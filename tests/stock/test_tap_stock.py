"""T-0011 tier 1b, criterion 15 (and, under ``ASTRA_STOCK=image``, criterion 16's two CUDA rows).

The tap around the **real** v0.29.0 ``DiffusionSampler`` -- vendored eager copy on CPU torch here, the image's
compiled one on CUDA inside the pinned container. The hand-written fast-suite fake once encoded a false premise
(that ``num_sampled`` is host-readable, R4); here the real code is the oracle.

Run: ``ASTRA_STOCK_TESTS=1 uv run --group stock pytest tests/stock -q -m "not gpu"`` (tier 1b), and
``ASTRA_STOCK=image ASTRA_STOCK_DEVICE=cuda ASTRA_STOCK_REQUIRE_CUDA=1`` inside the container
(``tests/contract/test_vllm_tap_cuda_gpu.py``). With the variable set and the ``stock`` group missing,
``tests/stock/world.py`` raises at import: the run **fails**, it never skips.

Regime (every row): Cfg CL 8, 4 slots, V 64, stability_threshold 1 (ST 2), 12 max steps, always confident,
seed 0 unless stated, the tap's virtual clock (``ManualClock``); the device and stock source come from the
environment and are printed by ``world.regime()``.

Rows (the container pins the pass count, ``STOCK_TAP_PASSED`` in the CUDA contract module):
7 S8 liveness rows on real torch + 1 device-move row + 1 value-read-device row + 1 sync-attribution
row + 4 criterion-15 rows + 1 step-cap row (R15) + 1 blocking-event row (R13) + 2 criterion-16 rows = 18.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from typing import Any

import pytest

from tests.fakes.diffusion_sampler import validate_canvas_step
from tests.stock.world import (
    DEVICE,
    IMAGE,
    Cfg,
    Snap,
    SyncSpy,
    World,
    frame_mismatches,
    peaked_logits,
    plugin_attr,
    regime,
    snaps_differ,
    torch,  # via world.py, whose import names the missing `stock` group
)

CFG = Cfg()
TOKENS = {"r1": 11, "r2": 23, "r3": 37, "r4": 41}
TRUNCATED = 5  # r2's canvas is truncated near max_model_len: 5 of CL 8 logits


def _scenario(w: World, *, calls: int = 40) -> None:
    """≥ 3 requests × ≥ 2 canvases, a truncated canvas (r2), a prefill-only call, and a slot reuse
    (r1 finishes; r4 is admitted into r1's slot, LIFO), then every request is released."""
    fn = peaked_logits(TOKENS)
    w.add("r1")
    w.add("r2", valid_len=TRUNCATED)
    w.add("r3")
    w.prefill()
    for _ in range(calls // 2):
        w.decode(fn)
    w.remove("r1")
    slot = w.add("r4")
    assert slot == w.snaps[0].slots["r1"], "fixture: r4 must reuse r1's slot"
    w.prefill()  # prefill-only call (r4), the others not in the batch
    for _ in range(calls // 2):
        w.decode(fn)
    for r in list(w.reqs):
        w.remove(r)


def _commits(w: World) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = defaultdict(list)
    for s in w.snaps:
        for r in s.order:
            c = s.committed(r)
            if c is not None:
                out[r].append(c)
    return dict(out)


def _commit_steps(w: World) -> dict[str, list[int]]:
    """Per request, in order, the step each committed canvas had reached: the slot's ``step``
    *before* its commit call (stock resets it to 0 during the call; R10's "pre-reset step")."""
    out: dict[str, list[int]] = defaultdict(list)
    for s in w.snaps:
        for r in s.order:
            if s.committed(r) is not None:
                assert s.pre_step is not None
                out[r].append(int(s.pre_step[s.slots[r]]))
    return dict(out)


# --------------------------------------------------------------------------- S8 liveness on real torch


@pytest.mark.parametrize(
    "op", [".item()", ".tolist()", ".numpy()", "bool()", "int()", "float()", ".nonzero()"]
)
def test_the_torch_spy_fires_on_each_listed_sync(op: str) -> None:
    """Criterion 3 / S8 liveness, real torch: each op in the list is recorded (it is the same spy
    criterion 15's "zero syncs" row runs under, so a dead spy would make that row vacuous)."""
    t = torch.tensor([[1, 2], [3, 4]], device=DEVICE)
    run = {
        ".item()": lambda: t[0, 0].item(),
        ".tolist()": lambda: t.tolist(),
        ".numpy()": lambda: t.cpu().numpy() if DEVICE.type == "cpu" else t.numpy(),
        "bool()": lambda: bool(t[0, 0]),
        "int()": lambda: int(t[0, 0]),
        "float()": lambda: float(t[0, 0]),
        ".nonzero()": lambda: t.nonzero(),
    }[op]
    spy = SyncSpy()
    with spy, contextlib.suppress(TypeError, RuntimeError):
        run()
    assert spy.hits, f"{op} was not recorded ({regime()})"


def test_the_torch_spy_counts_device_to_host_moves_only_when_they_block() -> None:
    """S8: on CUDA a blocking ``.cpu()``/``.to('cpu')``/``copy_`` into a host tensor fires, a
    ``non_blocking`` copy into a pinned buffer and a same-device ``.to`` do not; on CPU no move is a
    device→host move at all."""
    t = torch.arange(6, device=DEVICE)
    spy = SyncSpy()
    with spy:
        t.to(DEVICE)
        t.to(torch.int32)
    assert spy.hits == [], spy.hits
    if DEVICE.type == "cuda":
        pinned = torch.empty(6, dtype=t.dtype).pin_memory()
        with spy:
            pinned.copy_(t, non_blocking=True)
        assert spy.hits == [], spy.hits
        for move in (
            lambda: t.cpu(),
            lambda: t.to("cpu"),
            lambda: torch.empty(6, dtype=t.dtype).copy_(t),
        ):
            spy.hits.clear()
            with spy:
                move()
            assert spy.hits, "a blocking device->host move was not recorded"
    else:
        with spy:
            t.cpu()
            torch.empty(6, dtype=t.dtype).copy_(t)
        assert spy.hits == [], f"CPU->CPU is not a device->host move: {spy.hits}"


def test_the_spy_counts_value_reads_only_on_the_worlds_device() -> None:
    """S8: a sync is a device->host read. ``SyncSpy(device="cuda")`` must not count ``.item()`` on a
    CPU tensor (inductor's wrappers read CPU scalar tensors on every call; tier 2 at 5fc032e
    counted 240 of them); the CPU stand-in (``SyncSpy(device="cpu")``, what every CPU run uses)
    must count it, with the device in the record. On CUDA, a CUDA tensor's ``.item()`` is counted
    and a CPU tensor's next to it is not."""
    host = torch.tensor([3, 4])
    on_cuda_world = SyncSpy(device="cuda")
    with on_cuda_world:
        host[0].item()
        bool(host[1])
        host.tolist()
    assert on_cuda_world.hits == [], [str(r) for r in on_cuda_world.records]
    stand_in = SyncSpy(device="cpu")
    with stand_in:
        host[0].item()
    assert stand_in.hits == ["item"], stand_in.hits
    assert stand_in.records[0].device == "cpu" and "on cpu" in str(stand_in.records[0])
    if torch.cuda.is_available():
        dev = torch.tensor([5, 6], device="cuda")
        spy = SyncSpy(device="cuda")
        with spy:
            dev[0].item()
            host[0].item()
        assert spy.hits == ["item"], [str(r) for r in spy.records]
        assert spy.records[0].device.startswith("cuda"), str(spy.records[0])


def test_the_spy_attributes_each_sync_to_the_tap_or_to_stock() -> None:
    """Liveness of the attribution the zero-syncs row relies on: a sync made by plugin code is
    charged to the tap, one made by stock code is charged to stock *even when the plugin's wrapper
    is further out on the stack* (``TapSampler`` delegates to the stock sampler), and each record
    names its innermost call site. The two callers are compiled under a plugin and a vLLM file name,
    so the row needs neither the image nor the tap."""
    t = torch.tensor([1, 2], device=DEVICE)
    ns: dict[str, Any] = {"t": t}
    stock_src = "def stock_step():\n    return t[0].item()\n"
    exec(compile(stock_src, "/x/vllm/model_executor/models/diffusion_gemma.py", "exec"), ns)
    plugin_src = (
        "def tap_wrapper():\n    return stock_step()\n\ndef tap_own():\n    return t[1].item()\n"
    )
    exec(compile(plugin_src, "/x/astra_vllm_plugin/astra_vllm_plugin/sampler.py", "exec"), ns)
    spy = SyncSpy()
    with spy:
        ns["tap_wrapper"]()
        ns["tap_own"]()
    assert [(r.op, r.from_plugin) for r in spy.records] == [("item", False), ("item", True)], [
        str(r) for r in spy.records
    ]
    assert "diffusion_gemma.py:2 stock_step" in spy.records[0].site, str(spy.records[0])
    assert "sampler.py:5 tap_own" in spy.records[1].site, str(spy.records[1])


# --------------------------------------------------------------------------- criterion 15


def test_every_frame_equals_the_rows_of_its_call() -> None:
    """Criterion 15: hz 1e6 on a 1 s clock step, so every call publishes and every frame is
    attributable to the call that staged it (the tap is drained after each call, outside the
    sampler call). For each decode call, each request in the batch has exactly one frame whose
    canvas/argmax/step equal the post-call rows, whose ``stable`` equals the float64 numpy
    reduction of the history, whose phase is commit iff ``num_sampled > 0`` (with the sampled row cut
    at ``num_sampled``) and whose ``canvas_idx`` counts the commits before it. A prefill call may
    publish nothing; anything it does publish is checked against its rows the same way. Every commit
    is published exactly once."""
    w = World(CFG, tap=True, seed=0)
    _scenario(w)
    bad: list[str] = []
    closed: dict[str, int] = defaultdict(int)
    commit_frames: dict[str, list[list[int]]] = defaultdict(list)
    for s in w.snaps:
        by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in s.frames:
            validate_canvas_step(f)
            by_req[f["request_id"]].append(f)
        if s.kind == "decode":
            assert set(by_req) <= set(s.order), f"call {s.call}: frames for {set(by_req)}"
            for r in s.order:
                frames = by_req.get(r, [])
                if len(frames) != 1:
                    bad.append(f"call {s.call} {r}: {len(frames)} frames (want 1)")
                    continue
                f = frames[0]
                for m in frame_mismatches(
                    f, s, r, canvas_idx=closed[r], max_steps=CFG.max_denoising_steps
                ):
                    bad.append(f"call {s.call} {r}: {m}")
                if s.committed(r) is not None:
                    closed[r] += 1
                    commit_frames[r].append(list(f["committed_tokens"]))
        elif s.kind == "prefill":
            for r, frames in by_req.items():
                for f in frames:
                    for m in frame_mismatches(
                        f, s, r, canvas_idx=closed[r], max_steps=CFG.max_denoising_steps
                    ):
                        bad.append(f"prefill call {s.call} {r}: {m}")
        else:  # release: nothing left unpublished when every call publishes
            assert all(f["phase"] != "commit" for f in s.frames), (
                f"release of {s.order} re-published a commit already on the wire: {s.frames}"
            )
    assert bad == [], f"{len(bad)} mismatches ({regime()}):\n" + "\n".join(bad[:30])
    commits = _commits(w)
    assert all(len(commits.get(r, [])) >= 2 for r in ("r1", "r2", "r3")), (
        "fixture: each request must close ≥ 2 canvases, got {r: len(c) for r, c in commits.items()}"
    )
    assert any(len(c) == TRUNCATED for c in commits["r2"]), "fixture: a truncated commit must occur"
    assert dict(commit_frames) == commits, "every commit exactly once, with its sampled row"


def test_every_commit_is_published_exactly_once_under_the_rate_limit() -> None:
    """Criterion 4 against the real sampler: hz 30 on a 12 ms clock step. A canvas here takes 3
    calls (ST 2 denoise steps + the commit), i.e. 36 ms > the 33.3 ms window, so no two commits of a
    request share a window and none may overrun; but most commit calls are suppressed, so the
    commits go out lagged (next publishing call, or the release, S11). n = 60 decode calls + 2
    prefills + 4 releases, seed 0. The commit frames per request, in order, equal the committed rows
    the sampler emitted, each exactly once."""
    w = World(CFG, tap=True, seed=0, hz=30.0, dt=0.012)
    _scenario(w, calls=60)
    commits = _commits(w)
    got: dict[str, list[list[int]]] = defaultdict(list)
    steps: dict[str, list[int]] = defaultdict(list)
    with_rows: list[dict[str, Any]] = []
    for f in w.frames:
        validate_canvas_step(f)
        if f["phase"] == "commit":
            got[f["request_id"]].append(list(f["committed_tokens"]))
            steps[f["request_id"]].append(f["step"])
            if any(f.get(k) is not None for k in ("canvas", "argmax", "stable")):
                with_rows.append(f)
    assert w.tap.commit_overrun == 0, "one commit per window per request never overruns"
    assert dict(got) == commits, f"published {dict(got)} != committed {commits}"
    assert with_rows == [], f"R10: commit frames carrying canvas/argmax/stable: {with_rows[:3]}"
    want = _commit_steps(w)
    assert dict(steps) == want, (
        f"R10: commit steps on the wire {dict(steps)} != the pre-reset steps {want} ({regime()})"
    )
    publishing = sum(1 for s in w.snaps if s.kind == "decode")
    assert len(w.frames) < publishing * 3, "fixture: the limiter must suppress calls here"


def test_the_tap_leaves_the_engine_bit_identical() -> None:
    """Criterion 15: under one torch seed, every ``SamplerOutput`` field, every state row,
    ``draft_tokens`` and the global RNG state are bit-identical with and without the tap, after
    every call of the scenario (the claim tier 3 cannot make: there the noise is unseeded)."""
    plain = World(CFG, tap=False, seed=7)
    _scenario(plain)
    tapped = World(CFG, tap=True, seed=7)
    _scenario(tapped)
    assert len(plain.snaps) == len(tapped.snaps)
    diffs = [
        (a.call, a.kind, snaps_differ(a, b)) for a, b in zip(plain.snaps, tapped.snaps, strict=True)
    ]
    diffs = [d for d in diffs if d[2]]
    assert diffs == [], f"the tap changed the engine ({regime()}): {diffs[:5]}"
    assert len(tapped.frames) > 0, "the tapped run published nothing: the comparison is vacuous"


def test_the_spy_records_zero_syncs_on_the_sampler_thread() -> None:
    """Criterion 15 / 3: the whole scenario with every sampler call and every release under the S8
    spy -- stock code included (it is sync-free by design) -- records nothing.

    On CUDA (criterion 16) the image's sample step is compiled by inductor on its first calls, and
    compilation syncs (PROGRESS 2026-09-30), so the same stock-only warm-up as 16(a) runs first,
    outside the spy. Every hit is recorded with its call site (``SyncSpy.records``): hits whose stack
    passes through ``astra_vllm_plugin`` are the tap's and are asserted on first; any hit left after
    the warm-up -- stock's included -- still fails the row (S8 counts every sync on the sampler
    thread), with the sites in the message so the run attributes them.

    The release lines of this configuration (hz 30, 4 ms per call, canvases closing every 2-3
    calls) show ``commit_overrun`` well above zero: several commits per 33 ms window, of which R5's
    single latch row keeps the last. The row pins the accounting -- every commit is either published
    or counted, never silent -- not the overrun itself."""
    if DEVICE.type == "cuda":
        _warm_up()
    w = World(CFG, tap=True, seed=0, hz=30.0, dt=0.004)
    spy = SyncSpy()
    w.call_ctx = lambda: spy
    _scenario(w)
    mine = [r for r in spy.records if r.from_plugin]
    assert mine == [], f"{len(mine)} host syncs inside the tap ({regime()}):\n" + "\n".join(
        map(str, mine[:20])
    )
    assert spy.hits == [], (
        f"{len(spy.hits)} host syncs on the sampler thread, none in the tap -- stock's after the "
        f"warm-up ({regime()}):\n" + "\n".join(sorted({str(r) for r in spy.records})[:20])
    )
    assert w.tap.errors == 0
    assert w.frames, "nothing published: the row is vacuous"
    commits = sum(len(c) for c in _commits(w).values())
    published = sum(1 for f in w.frames if f["phase"] == "commit")
    assert published + w.tap.commit_overrun == commits, (
        f"{commits} commits, {published} published, commit_overrun {w.tap.commit_overrun}: a commit "
        "vanished silently (criterion 4)"
    )


def test_a_canvas_closed_by_the_step_cap_publishes_valid_frames() -> None:
    """R15: on stock a denoise call sets ``step = cur_step + 1`` and converges at
    ``step_after >= max_denoising_steps``, so the converging denoise frame carries ``max_steps`` and
    the commit after it carries the pre-reset ``max_steps``. Never-confident configuration
    (``confidence_threshold=-1.0``: the mean entropy is never below it, so only the cap closes a
    canvas), max_steps 12, one request, hz 1e6 (every call publishes), n = 30 decodes, seed 0.
    Every frame validates, matches its call's rows, at least two canvases close, and every commit
    frame's step is 12."""
    cfg = Cfg(confidence_threshold=-1.0)
    w = World(cfg, tap=True, seed=0)
    fn = peaked_logits(TOKENS, period=1)
    w.add("r1")
    w.prefill()
    for _ in range(30):
        w.decode(fn)
    w.remove("r1")
    invalid: list[str] = []
    for f in w.frames:
        try:
            validate_canvas_step(f)
        except AssertionError as exc:
            invalid.append(f"({f['phase']}, step {f['step']}): {exc}")
    assert invalid == [], f"frames the contract refuses at the step cap ({regime()}): {invalid[:4]}"
    bad: list[str] = []
    closed = 0
    for s in w.snaps:
        if s.kind != "decode":
            continue
        for f in s.frames:
            bad += [
                f"call {s.call}: {m}"
                for m in frame_mismatches(
                    f, s, "r1", canvas_idx=closed, max_steps=cfg.max_denoising_steps
                )
            ]
        if s.committed("r1") is not None:
            closed += 1
    assert bad == [], "\n".join(bad[:20])
    commits = [f for f in w.frames if f["phase"] == "commit"]
    assert len(commits) >= 2, f"fixture: the cap must close >= 2 canvases in 30 calls: {commits}"
    assert [f["step"] for f in commits] == [cfg.max_denoising_steps] * len(commits)
    assert any(
        f["phase"] == "denoise" and f["step"] == cfg.max_denoising_steps for f in w.frames
    ), "fixture: the converging denoise frame at the cap must occur"


def test_the_backends_cuda_events_are_blocking(monkeypatch: Any) -> None:
    """R13: the consumer waits on each bundle's event; a default ``torch.cuda.Event()`` is a
    spin-wait in ``synchronize()`` (a core at 100 % for the whole decode). ``TorchTapBackend`` on a
    CUDA device must create its events with ``blocking=True``. ``torch.cuda.Event`` is replaced by a
    recorder, so this runs on a CPU-only torch as well (the backend is constructed for "cuda" but
    allocates nothing here)."""
    made: list[dict[str, Any]] = []

    class RecordingEvent:
        def __init__(self, *a: Any, **k: Any) -> None:
            made.append({"args": a, **k})

    monkeypatch.setattr(torch.cuda, "Event", RecordingEvent)
    backend = plugin_attr("tap", "TorchTapBackend")("cuda")
    ev = backend.event()
    assert isinstance(ev, RecordingEvent), "the backend did not build a torch.cuda.Event"
    assert made and made[-1].get("blocking") is True, (
        f"torch.cuda.Event created with {made[-1] if made else None}: pass blocking=True (R13)"
    )


# --------------------------------------------------------------------------- criterion 16 (image, CUDA)


def _require_image_cuda() -> None:
    if IMAGE and DEVICE.type == "cuda" and torch.cuda.is_available():
        return
    msg = f"criterion 16 runs only under ASTRA_STOCK=image ASTRA_STOCK_DEVICE=cuda ({regime()})"
    if os.environ.get("ASTRA_STOCK_REQUIRE_CUDA") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _warm_up(calls: int = 14) -> None:
    """≥ 12 stock-only calls with sync-debug **off**, so inductor compiles the image's sample step
    here (PROGRESS 2026-09-30: compile-time syncs), with the measured runs' Cfg and batch shape."""
    assert torch.cuda.get_sync_debug_mode() == 0
    w = World(CFG, tap=False, seed=0)
    fn = peaked_logits(TOKENS)
    for r in ("r1", "r2", "r3"):
        w.add(r, valid_len=TRUNCATED if r == "r2" else None)
    w.prefill()
    for _ in range(calls):
        w.decode(fn)
    torch.cuda.synchronize()


@contextlib.contextmanager
def _strict() -> Any:
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(0)


@contextlib.contextmanager
def _relaxed() -> Any:
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _relax_stock_bookkeeping(w: World) -> None:
    """Stock's own request bookkeeping is not the tap's to answer for: v0.29.0's
    ``DiffusionGemmaRequestStates.add_request``/``remove_request`` write
    ``self_conditioning_embeds[slot] = 0`` -- a Python scalar into a CUDA tensor, the very landmine
    sync-debug refuses. Run those two stock methods with sync-debug off, so a raise under "error"
    in bind/release can only come from the tap's own work around them."""
    st = w.ms.diffusion_states
    for name in ("add_request", "remove_request"):
        fn = getattr(st, name)

        def relaxed(*a: Any, _fn: Any = fn, **k: Any) -> Any:
            with _relaxed():
                return _fn(*a, **k)

        setattr(st, name, relaxed)


def test_cuda_fifty_calls_under_sync_debug_error_after_warm_up() -> None:
    """Criterion 16(a): after the warm-up, the tap is built and ≥ 50 calls (denoise, converging,
    commit) plus a release flush run under ``set_sync_debug_mode("error")`` -- and so do the binds
    (``ms.add_request``, where the tap resets the slot for its new occupant: a Python scalar written
    into a device tensor there is the CUDA landmine, PROGRESS 2026-10-01). Zero raises -- and since
    a raise inside the tap is swallowed by the catch-all, ``tap.errors == 0`` is the witness.
    Control first: a stock-only bind and release under "error", with stock's own bookkeeping
    relaxed (``_relax_stock_bookkeeping``), must not raise, else a raise in the tapped bind or
    release could not be attributed."""
    _require_image_cuda()
    fn = peaked_logits(TOKENS)
    _warm_up()
    control = World(CFG, tap=False, seed=0)
    _relax_stock_bookkeeping(control)
    control.bind_ctx = _strict
    control.add("r1")  # a TOKENS key: peaked_logits has logits only for r1-r4
    control.prefill()
    control.decode(fn)
    torch.cuda.synchronize()
    # only stock + tap work runs under "error": World.remove enters call_ctx around
    # ms.remove_request alone, and its snapshot (the harness's own .clone().cpu()) runs after
    control.call_ctx = _strict
    control.remove("r1")
    control.call_ctx = contextlib.nullcontext

    w = World(CFG, tap=True, seed=0, drain_each_call=False)
    _relax_stock_bookkeeping(w)
    w.bind_ctx = _strict
    for r in ("r1", "r2", "r3"):
        w.add(r, valid_len=TRUNCATED if r == "r2" else None)
    w.prefill()
    torch.cuda.synchronize()
    w.call_ctx = _strict
    for _ in range(25):
        w.decode(fn)
    w.remove("r1")  # the release flush, under "error" too (call_ctx)
    w.add("r4")  # a bind into the freed slot (slot reuse), under "error" (bind_ctx)
    w.call_ctx = contextlib.nullcontext  # stock's prefill is not under test here
    w.prefill()
    w.call_ctx = _strict
    for _ in range(25):
        w.decode(fn)
    w.remove("r2")
    w.call_ctx = contextlib.nullcontext
    w.bind_ctx = contextlib.nullcontext
    torch.cuda.synchronize()
    w.tap.drain()
    assert w.tap.errors == 0, f"the tap raised under sync-debug 'error' ({regime()})"
    assert sum(len(v) for v in _commits(w).values()) >= 3, "fixture: commits must occur"
    assert any(f["phase"] == "commit" for f in w.frames)


#: ~35-50 ms on a 5090 per call: the stream falls behind the host at every call
_LAG_SLEEP_CYCLES = 100_000_000


def test_cuda_the_lag_really_happens_and_frames_still_equal_the_rows(monkeypatch: Any) -> None:
    """Criterion 16(b): a GPU sleep after each call and no host read between calls (rows and
    outputs cloned on the device, read after the run). At least one staged event's ``query()`` is
    False right after its call (the lag path is real), and after a synchronize the frames equal the
    snapshots, call by call, request by request."""
    _require_image_cuda()
    fn = peaked_logits(TOKENS)
    _warm_up()
    events: list[Any] = []
    real_event = torch.cuda.Event

    def spy_event(*a: Any, **k: Any) -> Any:
        # a factory, not a subclass: torch.cuda.Event takes its arguments in __new__, so a
        # subclass __init__ forwarding them reaches object.__init__ and raises (tier 2, b0a575c)
        ev = real_event(*a, **k)
        events.append(ev)
        return ev

    monkeypatch.setattr(torch.cuda, "Event", spy_event)
    w = World(CFG, tap=True, seed=0, drain_each_call=False)
    for r in ("r1", "r2", "r3"):
        w.add(r, valid_len=TRUNCATED if r == "r2" else None)
    w.prefill()
    torch.cuda.synchronize()
    w.tap.drain()
    start = len(w.frames)
    queried: list[bool] = []
    devsnaps = []
    for _ in range(40):
        devsnaps.append(w.decode_unsynced(fn))
        torch.cuda._sleep(_LAG_SLEEP_CYCLES)
        queried += [bool(e.query()) for e in events[-4:]]
    assert events, (
        "the tap's backend created no torch.cuda.Event (patch it before the world is built)"
    )
    assert False in queried, "every event had completed by the next call: the lag never happened"
    torch.cuda.synchronize()
    w.tap.drain()
    frames = w.frames[start:]
    by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in frames:
        by_req[f["request_id"]].append(f)
    bad: list[str] = []
    closed: dict[str, int] = defaultdict(int)
    pos: dict[str, int] = defaultdict(int)
    for i, d in enumerate(devsnaps):
        snap = Snap(
            call=i,
            kind="decode",
            order=d["order"],
            slots=d["slots"],
            state={k: v.cpu() for k, v in d["state"].items()},
            draft_tokens=torch.zeros(1),
            out={k: v.cpu() for k, v in d["out"].items()},
            rng_after=torch.zeros(1),
            pre_step=d["pre_step"].cpu(),
        )
        for r in snap.order:
            if pos[r] >= len(by_req[r]):
                bad.append(f"call {i} {r}: no frame")
                continue
            f = by_req[r][pos[r]]
            pos[r] += 1
            bad += [
                f"call {i} {r}: {m}"
                for m in frame_mismatches(
                    f, snap, r, canvas_idx=closed[r], max_steps=CFG.max_denoising_steps
                )
            ]
            if snap.committed(r) is not None:
                closed[r] += 1
    assert bad == [], f"{len(bad)} mismatches under lag ({regime()}):\n" + "\n".join(bad[:30])
