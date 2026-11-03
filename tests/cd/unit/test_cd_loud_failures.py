"""T-0067 criterion 7: every CD failure is loud and non-fatal (T-0062 decision 7).

For each class: no exception escapes ``add_request``/``__call__``; the request's first commit at or after the
trigger (and every commit before it: none) has EOS (1) as its first id, within ≤ 2 sampler calls of the
trigger; exactly one ``ASTRA_CD`` record whose ``error`` is the exact code (ruling S3 / T-0062 decision 12: T-0064
reason codes for ``CdRequestError``; ``unsupported_option``, ``infeasible``, ``z0``, ``empty_delta`` otherwise);
``error_count()`` rises by exactly 1; the other slots are bit-identical to the run with the failing request's
``astra_cd_*`` keys removed over the calls in which both runs schedule the same set.
Adversarial: log Z = −inf injected at step 7 (the capped converging step) in a mixed batch.
Guard in the other direction: a feasible request at the tightest feasible budget does not trip the path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, rows_differ
from tests.cd.unit._sampler_support import (
    CHAIN6,
    G1,
    Grammar,
    Records,
    brute_map,
    cd_xargs,
    error_count,
    patch_everywhere,
)
from tests.cd.unit._scenario import Spec, canvas_logits, run_pair

EOS = 1


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0067-loud-g1")


@pytest.fixture(scope="module")
def chain6() -> Grammar:
    return Grammar(CHAIN6, "t0067-loud-chain6")


def _expect_loud(
    cfg: Cfg,
    spec: Spec,
    fn: Callable[[int, str, int], torch.Tensor],
    *,
    trigger: int,
    records: Records,
    code: str,
    tag: str,
    n_calls: int = 10,
    allow_null_tag: bool = False,
    latch: bool = False,
    cd_ctx: Any = None,
) -> None:
    """``latch=True`` keeps the failing request scheduled after its EOS commit (as ``min_tokens`` or
    ``ignore_eos`` would): every later commit must still be EOS-first, and the batches never diverge."""
    before = error_count()
    specs = [spec, Spec("u1"), Spec("u2", params={"top_k": 3})]
    a, b = run_pair(
        cfg, specs, fn, n_calls=n_calls, end_on_eos=() if latch else ("c",), cd_ctx=cd_ctx
    )
    commits = [(i, s.committed("c")) for i, s in enumerate(a) if s.committed("c") is not None]
    after_trigger = [(i, ids) for i, ids in commits if i >= trigger]
    assert after_trigger, "the failing request never committed a canvas (it must end with EOS)"
    first_i, first_ids = after_trigger[0]
    assert first_ids[0] == EOS, (
        f"call {first_i}: the failing request committed {first_ids} — it continued instead of ending on EOS"
    )
    assert first_i <= trigger + 2, (
        f"EOS commit at call {first_i}, trigger at {trigger}: more than 2 calls"
    )
    ended = len(a) - 1 if latch else first_i
    if latch:
        later = [(i, ids) for i, ids in after_trigger if i > first_i]
        assert later, "fixture: no commit after the EOS commit (raise n_calls)"
        bad_later = [(i, ids) for i, ids in later if ids[0] != EOS]
        assert bad_later == [], f"the request continued after its EOS commit: {bad_later}"
    # other slots bit-identical up to (and including) the EOS commit; sets diverge afterwards
    for i in range(ended + 1):
        for r in ("u1", "u2"):
            bad = rows_differ(a[i], b[i], r)
            assert bad == [], f"call {i}: {r} differs from the keys-removed run in {bad}"
    errs = [r for r in records.all() if r.get("error") is not None]
    assert len(errs) == 1, f"expected exactly one error record, got {errs}"
    rec = errs[0]
    assert rec["error"] == code, f"error code {rec['error']!r}, expected {code!r} (ruling S3)"
    assert rec["record_tag"] == tag or (allow_null_tag and rec["record_tag"] is None), rec
    ws.contracts_cd().CdCanvasRecord.model_validate(rec)
    assert error_count() == before + 1, "the per-process error counter must rise by exactly one"


def _cfg() -> Cfg:
    return Cfg(CL=4, stability_threshold=1)


def _fn() -> Callable[[int, str, int], torch.Tensor]:
    return canvas_logits({"c": 11, "u1": 12, "u2": 13}, period=3)


@pytest.mark.parametrize(
    ("label", "mutate", "code"),
    [
        (
            "typo_key",
            lambda x: {
                **{k: v for k, v in x.items() if k != "astra_cd_mode"},
                "astra_cd_mdoe": x["astra_cd_mode"],
            },
            "unknown_key",
        ),
        ("sha_mismatch", lambda x: {**x, "astra_cd_automaton_sha256": "0" * 64}, "sha_mismatch"),
        ("bad_value", lambda x: {**x, "astra_cd_entropy_bound": -1.0}, "bad_value"),
    ],
)
def test_cd_request_error_is_loud_and_non_fatal(
    label: str, mutate: Any, code: str | None, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    tag = f"loud-{label}"
    xargs = mutate(cd_xargs(g1, tag))
    _expect_loud(
        _cfg(),
        Spec("c", xargs=xargs),
        _fn(),
        trigger=0,
        records=Records(caplog, capsys),
        code=code,
        tag=tag,
        allow_null_tag=(label == "typo_key"),
    )


@pytest.mark.parametrize(
    ("label", "params", "params_b"),
    [
        (
            "logprobs",
            {"logprobs": 5},
            {},
        ),  # the stock run drops logprobs: its path is not modelled on CPU
        ("top_k", {"top_k": 5}, None),
        ("top_p", {"top_p": 0.9}, None),
    ],
)
def test_unsupported_combination_is_loud(
    label: str, params: dict[str, Any], params_b: Any, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    tag = f"loud-{label}"
    spec = Spec("c", xargs=cd_xargs(g1, tag), params=params, params_b=params_b)
    _expect_loud(
        _cfg(),
        spec,
        _fn(),
        trigger=0,
        records=Records(caplog, capsys),
        code="unsupported_option",
        tag=tag,
    )


def test_infeasible_budget_is_loud(chain6: Grammar, caplog: Any, capsys: Any) -> None:
    """CHAIN6 needs 7 tokens (6 + stop) to reach F; max_tokens = 3."""
    spec = Spec("c", xargs=cd_xargs(chain6, "loud-budget"), params={"max_tokens": 3})
    _expect_loud(
        _cfg(),
        spec,
        _fn(),
        trigger=0,
        records=Records(caplog, capsys),
        code="infeasible",
        tag="loud-budget",
    )


def test_tightest_feasible_budget_does_not_trip(chain6: Grammar, caplog: Any, capsys: Any) -> None:
    """Guard's other direction: max_tokens = 7 = d(start) is feasible; no error, a constrained commit."""
    cfg = Cfg(CL=4, stability_threshold=1)
    fn = _fn()
    specs = [Spec("c", xargs=cd_xargs(chain6, "ok-budget"), params={"max_tokens": 7}), Spec("u1")]
    records = Records(caplog, capsys)
    a, _b = run_pair(cfg, specs, fn, n_calls=9)
    commits = [s.committed("c") for s in a if s.committed("c") is not None]
    assert commits and commits[0] == [1000, 1001, 1002, 1003], (
        commits
    )  # R_after = 3 forces the chain
    assert [r for r in records.all() if r.get("error")] == []


def test_log_z_neg_inf_at_step_7_in_a_mixed_batch(g1: Grammar, caplog: Any, capsys: Any) -> None:
    """Adversarial: cap = 7 steps, never confident; at the 7th denoise call (the capped converging step) the
    constrained slot's logits are −inf on every id but 1007, which G1 never admits: Z = 0."""
    cfg = Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=7)
    base = canvas_logits({"c": 61, "u1": 62, "u2": 63}, period=1)
    trigger = 6  # decode index of the 7th denoise call, where the cap forces convergence
    inject = trigger + 1  # its logits-call index (the prefill is call 0)

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        x = base(call, req, n)
        if req == "c" and call == inject:
            x = torch.full_like(x, float("-inf"))
            x[:, 1007] = 0.0
        return x

    # the preceding steps are feasible (the guard must not fire early): a MAP exists at every other step
    for c in range(1, inject):
        brute_map(g1, base(c, "c", 4), {0}, 4, 10**6)
    _expect_loud(
        cfg,
        Spec("c", xargs=cd_xargs(g1, "loud-logz")),
        fn,
        trigger=trigger,
        records=Records(caplog, capsys),
        code="z0",
        tag="loud-logz",
        n_calls=10,
    )


def test_empty_delta_star_is_loud(g1: Grammar, caplog: Any, capsys: Any, monkeypatch: Any) -> None:
    """δ* returning the empty set at canvas 0's commit is reported and never substituted (T-0065 criterion 7)."""
    from tests.cd.fakes.diffusion_sampler import cd_module

    budget = cd_module("budget")
    cd_module("sampler")
    calls = {"n": 0}

    def empty_delta_star(tables: Any, states: Any, committed_ids: Any, stop_ids: Any) -> Any:
        calls["n"] += 1
        z = torch.zeros_like(states, dtype=torch.bool)
        return budget.DeltaStar(states=z, empty=torch.tensor(True))

    patch_everywhere(monkeypatch, "delta_star", empty_delta_star)
    _expect_loud(
        _cfg(),
        Spec("c", xargs=cd_xargs(g1, "loud-delta")),
        _fn(),
        trigger=2,
        records=Records(caplog, capsys),
        code="empty_delta",
        tag="loud-delta",
    )
    assert calls["n"] >= 1, "cannot inject: δ* was not computed through T-0065's budget.delta_star"


# --------------------------------------------------------------------------- the latch (review M10)


@pytest.mark.parametrize("label", ["unsupported_option", "z0"])
def test_error_latch_holds_after_the_eos_commit(
    label: str, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    """The request stays scheduled after its EOS commit (min_tokens / ignore_eos): it never resumes."""
    records = Records(caplog, capsys)
    if label == "unsupported_option":
        spec = Spec("c", xargs=cd_xargs(g1, "latch-u"), params={"top_k": 5}, params_b=None)
        _expect_loud(
            _cfg(),
            spec,
            _fn(),
            trigger=0,
            records=records,
            code=label,
            tag="latch-u",
            n_calls=12,
            latch=True,
        )
        return
    cfg = Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=7)
    base = canvas_logits({"c": 61, "u1": 62, "u2": 63}, period=1)

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        x = base(call, req, n)
        if req == "c" and call == 7:
            x = torch.full_like(x, float("-inf"))
            x[:, 1007] = 0.0
        return x

    _expect_loud(
        cfg,
        Spec("c", xargs=cd_xargs(g1, "latch-z")),
        fn,
        trigger=6,
        records=records,
        code=label,
        tag="latch-z",
        n_calls=22,
        latch=True,
    )


# --------------------------------------------------------------------------- S9: the catch-all (`internal`)


class _RaiseOnRowWrite(torch.overrides.TorchFunctionMode):
    """Raises once, on the first host-indexed write (``st.argmax_canvas[int] = ...``) into ``argmax_canvas`` once
    armed: the CD post-step write-back. The stock step writes with a tensor index and is never hit."""

    def __init__(self, target: Any, armed: Any) -> None:
        super().__init__()
        self.target, self.armed, self.fired = target, armed, 0

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if (
            func is torch.Tensor.__setitem__
            and self.fired == 0
            and self.armed()
            and args[0] is self.target
            and isinstance(args[1], (int, slice))
        ):
            self.fired += 1
            raise RuntimeError("injected fault in post (T-0067 S9 row)")
        return func(*args, **(kwargs or {}))


def test_s9_raise_in_pre_is_internal(
    g1: Grammar, caplog: Any, capsys: Any, monkeypatch: Any
) -> None:
    from tests.cd.fakes.diffusion_sampler import cd_module

    cd_module("sampler")
    fired = {"n": 0}

    def boom(*a: Any, **k: Any) -> Any:
        fired["n"] += 1
        raise RuntimeError("injected fault in constrained_map (T-0067 S9 row)")

    patch_everywhere(monkeypatch, "constrained_map", boom)
    _expect_loud(
        _cfg(),
        Spec("c", xargs=cd_xargs(g1, "s9-pre")),
        _fn(),
        trigger=0,
        records=Records(caplog, capsys),
        code="internal",
        tag="s9-pre",
    )
    assert fired["n"] >= 1, "cannot inject: constrained_map (T-0066) was never called"


def test_s9_raise_in_post_is_internal(g1: Grammar, caplog: Any, capsys: Any) -> None:
    """Armed on the converging call (decode index 1) only; the fault hits CD's row write-back."""
    modes: list[_RaiseOnRowWrite] = []

    def ctx_for(w: Any) -> Any:
        import contextlib

        @contextlib.contextmanager
        def ctx() -> Any:
            mode = _RaiseOnRowWrite(w.ms.diffusion_states.argmax_canvas, lambda: w.calls == 2)
            modes.append(mode)
            with mode:
                yield

        return ctx

    _expect_loud(
        _cfg(),
        Spec("c", xargs=cd_xargs(g1, "s9-post")),
        _fn(),
        trigger=1,
        records=Records(caplog, capsys),
        code="internal",
        tag="s9-post",
        cd_ctx=ctx_for,
    )
    assert sum(m.fired for m in modes) >= 1, (
        "cannot inject: no host-indexed argmax_canvas write-back seen"
    )


@pytest.mark.parametrize("arm_at", [0, 1], ids=["denoise_call", "converging_call"])
def test_s9_stock_exceptions_propagate_unchanged(
    arm_at: int, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    """Ruling S9: the catch-all wraps ``pre()``/``post()`` only, never the inner stock call. A stock
    ``RuntimeError`` raised with a constrained request in the batch escapes the wrapper as the same object;
    no ``internal`` (or any) error record is emitted for it, not even after the requests are released (S11
    flushes), and the per-process error counter does not move."""
    from tests.cd.fakes.diffusion_sampler import World, stock

    armed = {"on": False, "raised": []}

    class RaisingStock(stock.DiffusionSampler):  # type: ignore[misc,name-defined]
        def __call__(self, *a: Any, **k: Any) -> Any:
            if armed["on"]:
                exc = RuntimeError("stock failure")
                armed["raised"].append(exc)  # type: ignore[attr-defined]
                raise exc
            return super().__call__(*a, **k)

    records = Records(caplog, capsys)
    before = error_count()
    fn = _fn()
    w = World(_cfg(), cd=True, sampler_cls=RaisingStock)
    w.add("c", xargs=cd_xargs(g1, "s9-stock"))
    w.add("u1")
    w.prefill()
    for _ in range(arm_at):
        w.decode(fn)
    armed["on"] = True
    with pytest.raises(RuntimeError, match="stock failure") as info:
        w.decode(fn)
    assert armed["raised"], "fixture: the stock sampler was never called"
    assert info.value is armed["raised"][-1], (  # type: ignore[index]
        f"the stock exception was replaced or wrapped: {info.value!r}"
    )
    armed["on"] = False
    for r in list(w.reqs):
        w.remove(r)
    errs = [r for r in records.all() if r.get("error") is not None]
    assert errs == [], f"a stock exception produced CD error records: {errs}"
    assert error_count() == before, "a stock exception bumped the CD error counter"
