"""The per-step constrained arms ``perstep_mf``, ``perstep_mar`` and ``perstep_mar`` + constrained SC (T-0068),
and the settling arm ``perstep_mar_qargmax`` (± constrained SC, T-0076).

SPEC §3.1 (with the 2026-09-29 correction), §3.4, §3.5, §3.7: at every denoise step of a constrained slot

- the **joint draw** x̂ ~ p(x | C) (T-0066) is fed back where accepted, a uniform id elsewhere (R0);
- acceptance (stock EntropyBound: Σ sorted − cummax ≤ bound) and ``confident`` (mean over CL < threshold) use
  **H(q)** of the constrained marginal (Mar) or stock's **H(p)** (Mf); positions ``≥ L_eff`` always use H(p);
- history, ``argmax_canvas`` and the committed canvas are the **constrained MAP**; convergence is a stable MAP
  history over ST and ``confident``, or the step cap; at convergence ``canvas ← MAP``;
- ``perstep_mar_qargmax`` is ``perstep_mar`` except its history row and ``argmax_canvas`` on a non-converging step:
  ``where(i < L_eff, argmax_v q_i(v), map_row)`` (first index on ties); convergence is the stock rule on that
  history, and at convergence ``argmax_canvas = canvas = MAP`` as for the other arms (T-0076);
- ``sc_embeds`` is ``p·E`` (stock) or ``q·E`` (constrained SC; ``q`` on ``[0, L_eff)``, ``p`` beyond), times
  stock's normalizer, kept only if CD's own convergence says the slot denoises next (ruling P9).

Every distribution is taken at stock's temperature for the slot's step on entry (ruling P3). Every random number
(joint draw, renoise ids) comes from the CD-owned generator the caller passes; the global RNG is never touched.

The work is split like T-0067's wrapper: ``prepare`` runs **before** the inner stock call on the pristine logits
(everything heavy: marginals, draw, MAP, entropies, the SC matmul), ``write`` runs **after** it and only combines
device tensors into the slot's rows with ``torch.where`` on the entry phase (the host never knows it). No host
synchronisation anywhere. Admission (ruling P1/P13) is ``requests.new_request``'s.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

from .budget import terminal_factor
from .draw import joint_draw
from .emission import class_log_weights_from, class_mass, log_transfer
from .mapdecode import class_max_of, constrained_map
from .marginals import constrained_marginals
from .requests import CdRequest
from .semiring import DEAD
from .tree import up_sweep

__all__ = [
    "PERSTEP_MODES",
    "PerStep",
    "Prepared",
    "finish",
    "new_perstep",
    "prepare",
    "write",
]

PERSTEP_MODES = ("perstep_mf", "perstep_mar", "perstep_mar_qargmax")
#: the CD generator's seed (any fixed value; it only has to be independent of the global RNG)
GENERATOR_SEED = 0x0068_CD5E


@dataclasses.dataclass(eq=False)
class PerStep:
    """A per-step request's arm and knobs, and its per-canvas accepted-count accumulator (device)."""

    marginal: bool  # Mar: accept/confident on H(q)
    constrained_sc: bool  # sc_embeds from q·E
    bound: float
    threshold: float
    acc: (
        torch.Tensor
    )  # int64 0-d: Σ over this canvas's denoise steps of accepted positions in [0, L_eff)
    #: T-0076 ``perstep_mar_qargmax``: history row = argmax of q on [0, L_eff) (host-known; implies marginal)
    q_argmax: bool = False


def new_perstep(req: CdRequest, inner: Any, device: torch.device) -> PerStep | None:
    """The arm of a live per-step request; ``None`` for end_projection or a request that is not live."""
    spec = req.spec
    if spec is None or spec.mode not in PERSTEP_MODES or not req.live:
        return None
    bound = spec.entropy_bound if spec.entropy_bound is not None else inner.entropy_bound
    thr = (
        spec.confidence_threshold
        if spec.confidence_threshold is not None
        else inner.confidence_threshold
    )
    return PerStep(
        marginal=spec.mode in ("perstep_mar", "perstep_mar_qargmax"),
        constrained_sc=spec.self_conditioning == "constrained",
        bound=float(bound),
        threshold=float(thr),
        acc=torch.zeros((), dtype=torch.int64, device=device),
        q_argmax=spec.mode == "perstep_mar_qargmax",
    )


@dataclasses.dataclass(eq=False)
class Prepared:
    """What ``prepare`` computed for one constrained slot in one call (all device tensors)."""

    map_row: (
        torch.Tensor
    )  # int64 [CL]: the MAP on [0, valid), stock's argmax of the padded rows (0) beyond
    map_empty: torch.Tensor  # 0-d bool: Z = 0 (the MAP's or the marginals')
    tokens: torch.Tensor  # int64 [valid]: the MAP (for δ*)
    l_pick: torch.Tensor  # int64 0-d: the L_eff this call used
    fed: torch.Tensor  # int64 [CL]: where(accept, x̂, uniform ids)
    confident: torch.Tensor  # 0-d bool
    n_acc: torch.Tensor  # int64 0-d: accepted positions in [0, L_eff)
    soft: torch.Tensor  # [CL, hidden] fp32: (π @ E) * normalizer
    hl0: torch.Tensor  # history length on entry
    #: int64 [CL]: the history row; ``map_row`` itself except under ``perstep_mar_qargmax`` (argmax of q on
    #: [0, L_eff), ``map_row`` beyond)
    hist_row: torch.Tensor
    q_argmax: bool  # host-known: the arm is perstep_mar_qargmax (finish's argmax_canvas)


def _temperature(inner: Any, step0: torch.Tensor, max_steps: float) -> torch.Tensor:
    """Stock's schedule at the slot's step on entry, the same ops as ``_compiled_sample_step`` phase 1."""
    remaining = (max_steps - step0.float()).clamp(min=1.0)
    temp = inner.t_min + (inner.t_max - inner.t_min) * (remaining / max_steps)
    return temp.clamp(min=1e-10)


def _accept(h: torch.Tensor, bound: float) -> torch.Tensor:
    """Stock EntropyBound over ``[CL]``: accepted iff the entropies ranked below sum to ≤ bound."""
    sorted_ent, sorted_idx = torch.sort(h, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= bound
    mask = torch.zeros_like(sorted_mask)
    mask.scatter_(0, sorted_idx, sorted_mask)
    return mask


def _soft(pi: torch.Tensor, inner: Any) -> torch.Tensor:
    """``(π @ E) * normalizer`` as stock forms it (vocab shard, TP all-reduce), fp32 ``[rows, hidden]``."""
    e = inner.embed_weight
    lo, hi = inner.sc_vocab_start, inner.sc_vocab_end
    soft = torch.matmul(pi[..., lo:hi].to(e.dtype), e[: hi - lo])
    if inner.tp_size > 1:
        soft = torch.ops.vllm.all_reduce(soft, group_name=inner.tp_group_name)
    return (soft * inner.normalizer).to(torch.float32)


def prepare(
    ps: PerStep,
    inner: Any,
    st: Any,
    slot: int,
    x: torch.Tensor,
    tables: Any,
    a_cur: torch.Tensor,
    r_cap: torch.Tensor,
    l_dev: torch.Tensor,
    cands: list[int],
    step0: torch.Tensor,
    generator: torch.Generator,
) -> Prepared:
    """Everything of one per-step call that depends only on the pristine logits ``x`` ``[valid, V]``."""
    valid, V = x.shape
    CL = int(st.canvas_length)
    dev = x.device
    scaled = x / _temperature(inner, step0, float(st.max_denoising_steps))

    # stock's H(p) per position, and p (needed for p·E unless every scored row takes q·E)
    lse = torch.logsumexp(scaled, dim=-1, keepdim=True)
    p = torch.exp(scaled - lse)
    hp = -torch.special.xlogy(p, p).sum(dim=-1)  # [valid]; a -inf logit contributes exactly 0
    # T-0073 D6 item 2 (peak memory): p is [valid, V]; it is not kept live across the L_eff loop, whose
    # marginals call is the peak. It is recomputed after the loop with the same ops (bitwise the same).
    del p

    # the l_eff-independent pieces, once per slot per call (T-0072 pass-through)
    cm = class_mass(scaled, tables)
    cw = class_log_weights_from(cm)
    cmax = class_max_of(scaled, tables)
    need_q = ps.marginal
    tokens = empty = l_pick = draw_tok = hq = q = qa = None
    for l_eff in cands:
        b = terminal_factor(tables, r_cap - l_eff)
        tree = up_sweep(log_transfer(scaled, tables, l_eff, class_mass=cm))
        d = joint_draw(
            scaled, tables, tree, a_cur, b, l_eff, generator=generator, class_log_weights=cw
        )
        m = constrained_map(scaled, tables, a_cur, b, l_eff, class_max=cmax)
        e = m.empty | d.empty
        c_hq = c_q = c_qa = None
        if need_q:
            marg = constrained_marginals(scaled, tables, tree, a_cur, b, l_eff, class_mass=cm)
            e = e | (marg.log_z <= DEAD)
            c_hq = marg.entropy
            c_q = marg.q if ps.constrained_sc else None
            if ps.q_argmax:
                # T-0076: argmax of this candidate's q (first index on ties); q itself is not kept
                c_qa = marg.q.argmax(dim=-1)
        if tokens is None:
            tokens, empty, draw_tok, hq, q, qa = m.tokens, e, d.tokens, c_hq, c_q, c_qa
            l_pick = torch.full((), l_eff, dtype=torch.int64, device=dev)
        else:
            hit = l_dev == l_eff
            tokens = torch.where(hit, m.tokens, tokens)
            empty = torch.where(hit, e, empty)
            draw_tok = torch.where(hit, d.tokens, draw_tok)
            if c_hq is not None and hq is not None:
                hq = torch.where(hit, c_hq, hq)
            if c_q is not None and q is not None:
                q = torch.where(hit, c_q, q)
            if c_qa is not None and qa is not None:
                qa = torch.where(hit, c_qa, qa)
            l_pick = torch.where(hit, l_eff, l_pick)
        # nothing of this candidate outlives its selection: the next candidate's marginals are the peak
        del tree, d, m, c_hq, c_q, c_qa
        if need_q:
            del marg
    assert tokens is not None and empty is not None and l_pick is not None
    assert draw_tok is not None
    del cm, cw, cmax
    p = torch.exp(scaled - lse)
    del scaled

    pos_v = torch.arange(valid, device=dev)
    scored = pos_v < l_pick  # [valid]
    h_v = torch.where(scored, hq, hp) if hq is not None else hp
    # padded rows (valid < CL) are stock's zero logits: uniform, H = log V (never committed)
    pad = CL - valid
    h = h_v.to(torch.float32)
    if pad:
        h = torch.cat([h, torch.full((pad,), math.log(V), dtype=h.dtype, device=dev)])
    accept = _accept(h, ps.bound)
    confident = h.mean() < ps.threshold
    pos = torch.arange(CL, device=dev)
    n_acc = (accept & (pos < l_pick)).sum()

    uniform = torch.randint(0, V, (CL,), generator=generator, dtype=torch.int64, device=dev)
    xhat = draw_tok.to(torch.int64)
    if pad:
        xhat = torch.cat([xhat, uniform[valid:]])
    fed = torch.where(accept, xhat, uniform)

    map_row = tokens.to(torch.int64)
    if pad:
        map_row = torch.cat([map_row, torch.zeros(pad, dtype=torch.int64, device=dev)])
    hist_row = map_row
    if ps.q_argmax:
        # T-0076: where(i < L_eff, argmax_v q_i(v), map_row); beyond L_eff exactly perstep_mar's row
        assert qa is not None
        hist_row = torch.where(scored, qa, map_row[:valid])
        if pad:
            hist_row = torch.cat([hist_row, map_row[valid:]])

    # self-conditioning: π = q on the scored rows under constrained SC, else p
    if ps.constrained_sc and q is not None:
        pi = torch.where(scored.unsqueeze(1), q, p)
        del q
    else:
        pi = p
    del p
    soft = _soft(pi, inner)
    del pi
    if pad:
        uni = torch.zeros((1, V), dtype=torch.float32, device=dev).log_softmax(dim=-1).exp()
        soft = torch.cat([soft, _soft(uni, inner).expand(pad, -1)])
    return Prepared(
        map_row=map_row,
        map_empty=empty,
        tokens=tokens,
        l_pick=l_pick,
        fed=fed,
        confident=confident,
        n_acc=n_acc,
        soft=soft,
        hl0=st.accepted_canvas_history_len[slot].clone(),
        hist_row=hist_row,
        q_argmax=ps.q_argmax,
    )


def write(
    ps: PerStep,
    pr: Prepared,
    st: Any,
    slot: int,
    ec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """After the inner call: history, stability and CD's convergence. Returns ``(conv, den, acc_entry)``;
    ``acc_entry`` is the canvas's accepted-count sum on entry (reported on its commit call). The caller
    decides this call's errors from ``conv`` and then writes the canvas rows (``finish``)."""
    s = slot
    den = ~ec
    ST = int(st.stability_threshold)
    hist = st.accepted_canvas_history
    wp = pr.hl0 % ST
    for i in range(ST):
        hist[s, i] = torch.where((wp == i) & den, pr.hist_row, hist[s, i])
    rows = hist[s]
    stable = (rows == rows[0:1]).all()
    hl_new = st.accepted_canvas_history_len[s]
    step_after = st.step[s]
    conv = den & ((stable & pr.confident & (hl_new >= ST)) | (step_after >= st.max_denoising_steps))
    acc_entry = ps.acc.clone()
    ps.acc.copy_(torch.where(den, ps.acc + pr.n_acc, torch.zeros_like(ps.acc)))
    return conv, den, acc_entry


def finish(
    pr: Prepared,
    st: Any,
    slot: int,
    ec: torch.Tensor,
    conv: torch.Tensor,
    force: torch.Tensor,
    eos: torch.Tensor,
) -> None:
    """The slot's canvas, argmax, phase, confident and sc rows (after the errors of this call are known)."""
    s = slot
    den = ~ec
    stock_argmax = st.argmax_canvas[s].clone()
    stock_canvas = st.canvas[s].clone()
    if pr.q_argmax:
        # T-0076: the history row while denoising; the MAP (= the canvas) at convergence
        argmax = torch.where(den, torch.where(conv, pr.map_row, pr.hist_row), stock_argmax)
    else:
        argmax = torch.where(den, pr.map_row, stock_argmax)
    canvas = torch.where(den, torch.where(conv, pr.map_row, pr.fed), stock_canvas)
    st.argmax_canvas[s] = torch.where(force, eos, argmax)
    st.canvas[s] = torch.where(force, eos, canvas)
    st.is_encoder_phase[s] = torch.where(ec, st.is_encoder_phase[s], conv) | force
    st.confident[s] = torch.where(den, pr.confident, st.confident[s])
    keep = den & ~conv & ~force
    sc = st.self_conditioning_embeds
    sc[s] = torch.where(keep, pr.soft.to(sc.dtype), torch.zeros_like(sc[s]))
