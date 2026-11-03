"""Golden per-step traces for T-0076 criterion 9 (bit-identity of the pre-existing arms).

The fixture ``tests/cd/fixtures/perstep_golden_8416e39.pt`` was written by ``write_golden`` **before any T-0076
implementation** (at a tree whose code equals 8416e39; see ``tests/cd/fixtures/MANIFEST.md``). The test
``test_cd_perstep_golden.py`` re-runs the same scenarios with ``trace`` and compares every recorded tensor with
``torch.equal`` on its bit pattern.

Scenarios (all seeded; the logits are a pure function of (scenario, call)): the T-0067 CPU stand-in (vendored
v0.29.0 stock step), one constrained slot ``c`` and one unconstrained companion ``u``.

- ``budget``: CL 8, ``max_tokens`` 29 -> three full canvases and a fourth with ``L_eff`` = 5 < CL (budget).
- ``pad``: CL 8, the constrained slot's valid length is 5 < CL on every call (padded rows), ``max_tokens`` 40.

Per call the trace keeps the constrained slot's ``canvas``, ``accepted_canvas_history`` (+ length),
``argmax_canvas``, ``self_conditioning_embeds``, ``is_encoder_phase``, ``confident``, ``step``,
``draft_tokens[:CL]``, the committed ids, the device report the sampler packed for the slot (``int64``), the
CD-owned generator's state after the call (per-step arms), the global RNG state and the sequence of torch
function names called inside the sampler call (``ops``, via ``TorchFunctionMode``); plus every ``ASTRA_CD`` record
line, in order.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, World, random_live_logits
from tests.cd.unit._sampler_support import G1, LIVE, Grammar

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
#: The pin ``test_cd_perstep_golden.py`` compares with, op digest included. Re-recorded under T-0077b ruling M3/K8 at
#: 5321565 (the kernel dispatch check adds one ``getset_descriptor.__get__`` -- the ``is_cuda`` read in
#: ``cd/kernels/backend.use_kernels`` -- per ``constrained_marginals`` call; every value and record line equals the
#: 9c9c508 and 8416e39 traces).
FIXTURE = _FIXTURES / "perstep_golden_5321565.pt"
#: The T-0077a pin (ruling L3, live-state slicing); kept as the previous op-digest anchor, never re-recorded.
FIXTURE_9C9C508 = _FIXTURES / "perstep_golden_9c9c508.pt"
#: The original T-0076 pin, recorded before any T-0076 implementation; kept, never re-recorded. The golden module
#: still compares every traced value and record line with it (op digest excluded).
FIXTURE_8416E39 = _FIXTURES / "perstep_golden_8416e39.pt"

#: the pre-existing arms as (mode, self_conditioning)
GOLDEN_ARMS: dict[str, tuple[str, str]] = {
    "end_projection": ("end_projection", "model"),
    "perstep_mf": ("perstep_mf", "model"),
    "perstep_mar": ("perstep_mar", "model"),
    "perstep_mar_sc": ("perstep_mar", "constrained"),
}
SCENARIOS: dict[str, dict[str, Any]] = {
    "budget": {"CL": 8, "max_tokens": 29, "valid": None, "n_calls": 40, "seed": 11},
    "pad": {"CL": 8, "max_tokens": 40, "valid": 5, "n_calls": 30, "seed": 23},
}
#: T-0067's G1 (``_sampler_support``): forced structure, finals reachable, viable over several canvases
GRAMMAR_SPEC = G1
TENSOR_ROWS = (
    "canvas",
    "argmax_canvas",
    "accepted_canvas_history",
    "accepted_canvas_history_len",
    "self_conditioning_embeds",
    "is_encoder_phase",
    "confident",
    "step",
)


def _cfg(CL: int) -> Cfg:
    return Cfg(
        CL=CL,
        max_denoising_steps=6,
        stability_threshold=1,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=1.0,
        confidence_threshold=1e9,
    )


def _logits(seed: int):  # type: ignore[no-untyped-def]
    def fn(call: int, req: str, n: int) -> torch.Tensor:
        base = seed * 1000 + (call // 2) if req == "c" else seed * 1000 + 500 + call
        return random_live_logits(base, LIVE, n)

    return fn


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("ASTRA_CD "):
            self.lines.append(msg)


class _OpLog(torch.overrides.TorchFunctionMode):
    """The name of every torch function called inside one sampler call (criterion 9: no extra device op)."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.names.append(
            getattr(func, "__qualname__", None) or getattr(func, "__name__", repr(func))
        )
        return func(*args, **(kwargs or {}))


def op_digest(names: Sequence[str]) -> dict[str, Any]:
    """sha256 of the op-name sequence, its length, and per-name counts (for a readable diff on failure)."""
    import hashlib
    from collections import Counter

    return {
        "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "n": len(names),
        "counts": dict(sorted(Counter(names).items())),
    }


def trace(
    arm: str, scenario: str, *, mode: str | None = None, sc: str | None = None
) -> dict[str, Any]:
    """Run one (arm, scenario) and return its trace. ``mode``/``sc`` override the wire keys (used by T-0076's
    rows to run a new arm through the same scenario)."""
    m, s = GOLDEN_ARMS.get(arm, (None, None))
    mode = mode or m
    sc = sc or s
    assert mode is not None and sc is not None
    p = SCENARIOS[scenario]
    CL, valid = int(p["CL"]), p["valid"]
    g = Grammar(GRAMMAR_SPEC, "t0076-golden")
    xargs = ws.raw_xargs(
        g.payload, mode=mode, self_conditioning=sc, record_tag=f"golden-{arm}-{scenario}"
    )
    lines = _Lines()
    logger = logging.getLogger("vllm.astra_cd")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(lines)
    try:
        w = World(_cfg(CL), cd=True, seed=int(p["seed"]))
        reports: list[list[torch.Tensor]] = []
        orig = w.sampler._report

        def spy(jobs: Sequence[Any]) -> None:
            reports.append([j.report.detach().clone().cpu() for j in jobs])
            orig(jobs)

        w.sampler._report = spy
        w.add("c", xargs=xargs, max_tokens=int(p["max_tokens"]))
        w.add("u")
        if valid is not None:
            w.reqs["c"].valid_len = int(valid)
        w.prefill()
        op_logs: list[_OpLog] = []

        def ctx() -> _OpLog:
            log = _OpLog()
            op_logs.append(log)
            return log

        w.call_ctx = ctx
        fn = _logits(int(p["seed"]))
        calls: list[dict[str, Any]] = []
        emitted = 0
        for _ in range(int(p["n_calls"])):
            if "c" not in w.reqs:
                break
            n_before = len(reports)
            snap = w.decode(fn)
            slot = snap.slots["c"]
            rec: dict[str, Any] = {name: snap.state[name][slot].clone() for name in TENSOR_ROWS}
            rec["draft"] = snap.draft_tokens[slot, :CL].clone()
            ids = snap.committed("c")
            rec["committed"] = torch.tensor(ids if ids is not None else [], dtype=torch.int64)
            got = reports[n_before:]
            rec["report"] = got[0][0] if got and got[0] else torch.zeros(0, dtype=torch.int64)
            gens = getattr(w.sampler, "_generators", {})
            rec["gen_state"] = (
                next(iter(gens.values())).get_state().clone()
                if gens
                else torch.zeros(0, dtype=torch.uint8)
            )
            rec["rng_after"] = snap.rng_after.clone()
            rec["ops"] = op_digest(op_logs[-1].names)
            calls.append(rec)
            if ids is not None:
                emitted += len(ids)
                if emitted >= int(p["max_tokens"]) or ids[0] == 1:
                    w.remove("c")
        # flush the last call's reports (records are read one call late)
        w.sampler._flush()
    finally:
        logger.removeHandler(lines)
        logger.setLevel(old_level)
    return {"calls": calls, "records": list(lines.lines)}


def all_traces() -> dict[str, dict[str, Any]]:
    return {f"{arm}/{scen}": trace(arm, scen) for arm in GOLDEN_ARMS for scen in SCENARIOS}


def canvases(tr: dict[str, Any]) -> int:
    return sum(1 for c in tr["calls"] if c["committed"].numel() > 0)


def write_golden(path: Path = FIXTURE) -> dict[str, dict[str, Any]]:
    data = all_traces()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def describe(data: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    for key, tr in data.items():
        lens = [int(c["committed"].numel()) for c in tr["calls"] if c["committed"].numel()]
        out.append(
            f"{key}: calls={len(tr['calls'])} canvases={len(lens)} lens={lens} records={len(tr['records'])}"
        )
    return out


__all__ = [
    "FIXTURE",
    "FIXTURE_9C9C508",
    "FIXTURE_8416E39",
    "GOLDEN_ARMS",
    "SCENARIOS",
    "all_traces",
    "canvases",
    "describe",
    "trace",
    "write_golden",
]


def main(argv: Sequence[str]) -> int:
    """``python -m tests.cd.unit._golden_support [--write]``: dry run (describe) or write the fixture."""
    import time

    t0 = time.time()
    write = "--write" in argv
    data = write_golden() if write else all_traces()
    for line in describe(data):
        print(line)
    for key, tr in data.items():
        for r in tr["records"]:
            if '"error":null' not in r:
                print("ERROR RECORD", key, r[:200])
    print(f"wall {time.time() - t0:.1f} s ({'written ' + str(FIXTURE) if write else 'dry'})")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
