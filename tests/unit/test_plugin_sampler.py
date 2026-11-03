"""T-0011 criteria 1-5 and 9 (tier 1a): the interception point and the hot path of the canvas tap.

Everything here is driven against the fakes in ``tests/fakes/diffusion_sampler.py``; the fast suite
has neither ``vllm`` nor ``torch`` (docs/tasks/T-0011.md "Tests to write"). Tier 1b
(``tests/stock/test_tap_stock.py``) runs the same tap around the real vendored v0.29.0 sampler.
The plugin modules are imported through ``load_plugin_module`` so that a module which does not exist
yet fails as an assertion naming the criterion rather than as a collection error.

Regime of every row: CPU, Python 3.11, no torch, the fakes' virtual host clock (``ManualClock``), no
randomness (n and the call pattern are stated per row; there is no seed because nothing is random).

**The API these tests fix** (ruling R1: the task names behaviours, the test-author fixes signatures
so the implementer and the reviewer read the same contract):

``astra_vllm_plugin.model_state``
    ``make_model_state_cls(base, *, tap) -> type`` builds ``AstraDiffusionGemmaModelState``, a
    subclass of ``base``. ``tap`` is a factory ``tap(slots) -> CanvasTap``; it is called once, when
    ``custom_sampler`` wraps (criterion 4: the host pool is allocated then). The model state owns
    the slot table as ``state.astra_slots`` because ``req_id`` only exists on the model-state side
    (criterion 9). Taking the base as an argument is what lets criterion 1 be proved on a host with
    no vLLM, and what lets T-0069 compose ``make_model_state_cls(make_cd_model_state_cls(...))``.

``astra_vllm_plugin.sampler``
    ``TapSampler(inner, tap)``. All of the wrapper's own attributes are **private**: criterion 2
    forbids any public attribute the inner sampler lacks, so the inner object is ``_inner``.

``astra_vllm_plugin.tap``
    ``SlotTable`` (``bind`` / ``options`` / ``release`` / ``info`` / ``epoch``),
    ``CanvasTap(publisher, *, slots, backend, hz, clock, queue_size)`` with the sampler-thread side
    reached through ``TapSampler`` and ``drain()`` on the consumer side (the constructor starts no
    thread), and ``StepPublisher``. ``backend`` is the injected torch/CUDA seam
    (``host_buffer`` / ``event`` / ``torch``) and ``clock`` a zero-argument monotonic-seconds
    callable -- both exist so criteria 3 and 4 are provable without a GPU.

**Device semantics (R4/R5).** ``SamplerOutput.num_sampled`` and ``sampled_token_ids`` are device
tensors in batch order, reused and zeroed by the sampler on every call; the host cannot read them
without a sync. Every commit is therefore published exactly once, *lagged*: no later than the slot's
next publishing call or its release (S11).
"""

from __future__ import annotations

import ast
import inspect
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fakes.diffusion_sampler import (
    RELEASE_LOG_EXPECTED,
    RELEASE_LOG_SUBMITTED,
    SPY,
    TAP_COUNTERS,
    BlockingPublisher,
    FakeDiffusionSampler,
    FakeHostBuffer,
    FakeInputBatch,
    FakeModelStateBase,
    FakeSamplerOutput,
    FakeTapBackend,
    NewRequestData,
    RecordingPublisher,
    Row,
    SamplingParams,
    SpyTensor,
    TapFactory,
    TapHarness,
    attr,
    load_plugin_module,
    make_two_level_base,
    model_state_cls,
    release_log_scenario,
    spy_report,
    validate_canvas_step,
)

PLUGIN_SRC = Path(__file__).resolve().parents[2] / "plugin" / "astra_vllm_plugin"

#: the runner probes a sampler by attribute, never by isinstance (model_runner.py:1462-1463)
PROBE_NAMES = (
    "sampling_states",
    "req_states",
    "logits_mode",
    "penalties_state",
    "apply_staged_writes",
    "add_request",
    "compute_nans",
    "get_logprobs_dims",
    "diffusion_states",
    "canvas_length",
    # the scaffold's two guesses, measured absent on the box 2026-09-13: parity here means
    # "absent on the wrapper too", which is criterion 2's explicit negative (and criterion 11's)
    "remove_request",
    "states",
)

#: the one name the tap must intercept (criterion 9: per-request options arrive here), so it is
#: exempt from the value-identity half of criterion 2 -- but not from hasattr parity, and not from
#: delegating (test_add_request_delegates_to_the_inner_sampler)
INTERCEPTED = ("add_request",)

#: criterion 1, negative: the tap never reaches into the diffusion sampler's construction inputs
FORBIDDEN_NAMES = (
    "EntropyBound",
    "gen_config",
    "t_min",
    "t_max",
    "embed_weight",
    "shard_indices",
)


# --------------------------------------------------------------------------- helpers


def _factory(criterion: str = "1") -> TapFactory:
    return TapFactory(hz=1000.0, criterion=criterion)


def _no_violations(where: str) -> None:
    assert SPY.violations() == {}, f"{where}: {spy_report()}"


# --------------------------------------------------------------------------- criterion 1


def test_custom_sampler_wraps_what_super_returned() -> None:
    """Criterion 1: the 2-tuple is preserved and both members are *super's own objects*."""
    cls = model_state_cls(FakeModelStateBase, _factory(), "1")
    assert cls.__name__ == "AstraDiffusionGemmaModelState"
    assert issubclass(cls, FakeModelStateBase)

    state = cls()
    handed_in = object()
    result = state.custom_sampler(handed_in)

    assert isinstance(result, tuple) and len(result) == 2, (
        "the runner unpacks two values at model_runner.py:449-452"
    )
    wrapped, second = result
    assert state.custom_sampler_calls == [handed_in], (
        "super().custom_sampler(sampler) must be called"
    )
    assert second is state.second, "the second member must be super's object, not a copy"
    assert wrapped is not state.sampler, "the first member must be the tap wrapper"
    assert attr(wrapped, "_inner", "1") is state.sampler, (
        "the wrapper must hold exactly the sampler object super returned (identity)"
    )


def test_custom_sampler_returns_none_when_super_does() -> None:
    """Criterion 1: ``None`` passthrough — the runner reads it as 'no custom sampler'."""
    factory = _factory()
    cls = model_state_cls(FakeModelStateBase, factory, "1")
    state = cls(result=None)
    assert state.custom_sampler(object()) is None
    assert factory.built == [], "no sampler to wrap: no tap, no pinned pool"


def test_custom_sampler_propagates_the_entropy_bound_validation_error() -> None:
    """Criterion 1: upstream's ValueError (diffusion_gemma.py:833) propagates unchanged."""
    boom = ValueError("DiffusionGemma requires an EntropyBound sampler_config")
    cls = model_state_cls(FakeModelStateBase, _factory(), "1")
    state = cls(raises=boom)
    with pytest.raises(ValueError) as excinfo:
        state.custom_sampler(object())
    assert excinfo.value is boom, "the exception object itself must propagate, not a rewrapping"


def test_custom_sampler_never_constructs_a_diffusion_sampler() -> None:
    """Criterion 1, negative: we wrap, we do not build; and we probe none of its inputs."""
    cls = model_state_cls(FakeModelStateBase, _factory(), "1")
    state = cls()
    before = FakeDiffusionSampler.instances
    state.custom_sampler(object())
    assert FakeDiffusionSampler.instances == before, (
        "the tap constructed a sampler; it must only wrap what super() returned"
    )
    probed = set(state.probes) | set(state.sampler.probes)
    assert probed.isdisjoint(FORBIDDEN_NAMES), (
        f"the tap read sampler construction inputs: {sorted(probed & set(FORBIDDEN_NAMES))}"
    )


def test_plugin_source_never_names_the_sampler_construction_inputs() -> None:
    """Criterion 1, negative: not in the code of the modules T-0011 owns.

    Identifiers and string constants only, from the AST: ``model.py`` is T-0015's file and its
    *comments* discuss ``EntropyBound`` and ``t_min``/``t_max`` precisely to record that we do not
    touch them, which a grep over raw text would flag.
    """
    offenders: dict[str, list[str]] = {}
    for module in ("sampler", "tap", "model_state"):
        path = PLUGIN_SRC / "astra_vllm_plugin" / f"{module}.py"
        assert path.exists(), f"T-0011 criterion 1: {module}.py does not exist yet"
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
        hit = sorted(names & set(FORBIDDEN_NAMES))
        if hit:
            offenders[path.name] = hit
    assert offenders == {}, f"the tap reaches into the sampler's construction inputs: {offenders}"


def test_the_factory_works_over_a_two_level_wrapper_base() -> None:
    """Criterion 1: any base, including the ``make_cd_model_state_cls(...)`` shape (T-0069's
    ``Tap(CD(stock))``). The tap wraps whatever the middle layer returned -- by identity -- and the
    request bookkeeping reaches every layer. n = 3 calls at hz 1000 on a 10 ms clock (all publish).
    """
    two_level = make_two_level_base(FakeModelStateBase)
    h = TapHarness(hz=1000.0, base=two_level, criterion="1")
    assert issubclass(type(h.ms), two_level), "the tap's model state must extend the given base"
    inner = attr(h.wrapper, "_inner", "1")
    assert isinstance(inner, two_level.layer_sampler_cls), (
        "the tap must wrap the middle layer's sampler, not reach past it to the stock one"
    )
    assert inner._layer_inner is h.sampler
    h.bind(0, "req-two")
    assert h.ms.layer_added == [(0, "req-two")], "add_request must reach the middle layer"
    assert h.ms.add_request_calls and h.ms.add_request_calls[0][0] == 0, "and the stock base"
    for i in range(3):
        h.step(step=i + 1)
        h.clock.advance(0.01)
    h.tap.drain()
    assert inner.layer_calls == 3, "every call goes through the middle layer exactly once"
    assert [m["request_id"] for m in h.frames()] == ["req-two"] * 3
    h.release("req-two")
    assert h.ms.layer_removed == ["req-two"] and h.ms.remove_request_calls == ["req-two"]
    _no_violations("two-level base")


# --------------------------------------------------------------------------- criterion 2


def test_wrapper_attribute_surface_equals_the_inner_sampler() -> None:
    """Criterion 2: ``hasattr`` parity over the runner's probe list, values are the inner's."""
    inner = FakeDiffusionSampler()
    sampler_mod = load_plugin_module("sampler", "2")
    wrapper = attr(sampler_mod, "TapSampler", "2")(inner, None)
    mismatched = [n for n in PROBE_NAMES if hasattr(wrapper, n) != hasattr(inner, n)]
    assert mismatched == [], (
        "the runner feature-detects by attribute; these names disagree with the inner sampler: "
        f"{mismatched}"
    )
    for name in PROBE_NAMES:
        if not hasattr(inner, name) or name in INTERCEPTED:
            continue
        got, want = getattr(wrapper, name), getattr(inner, name)
        if inspect.ismethod(want):
            # a bound method is rebuilt on every attribute access, so compare what it binds
            assert got.__func__ is want.__func__ and got.__self__ is inner, (
                f"{name} must be the inner sampler's own method, bound to the inner object"
            )
        else:
            assert got is want, f"{name} must be the inner object itself"
    # the two the runner reads at model_runner.py:1462-1463 and DiffusionSampler does not define
    assert not hasattr(wrapper, "compute_nans")
    assert not hasattr(wrapper, "get_logprobs_dims")


def test_wrapper_does_not_define_remove_request() -> None:
    """Criterion 2: ``DiffusionSampler`` has none — the scaffold's ``remove_request`` is the bug."""
    sampler_mod = load_plugin_module("sampler", "2")
    tap_sampler = attr(sampler_mod, "TapSampler", "2")
    assert "remove_request" not in vars(tap_sampler), (
        "TapSampler must not define remove_request: DiffusionSampler has none, so the wrapper "
        "would answer hasattr() differently from the object it stands in for"
    )
    assert not hasattr(tap_sampler(FakeDiffusionSampler(), None), "remove_request")


def test_wrapper_defines_no_public_attribute_the_inner_sampler_lacks() -> None:
    """Criterion 2, negative: the whole public surface belongs to the inner object."""
    inner = FakeDiffusionSampler()
    sampler_mod = load_plugin_module("sampler", "2")
    wrapper = attr(sampler_mod, "TapSampler", "2")(inner, None)
    extra = {n for n in dir(wrapper) if not n.startswith("_")} - set(dir(inner))
    assert extra == set(), (
        f"TapSampler adds public attributes the inner sampler lacks: {sorted(extra)}; "
        "keep the wrapper's own state private (_inner, _tap)"
    )


def test_add_request_delegates_to_the_inner_sampler() -> None:
    """Criterion 2/9: the tap reads ``sampling_params.extra_args`` on the way through, no more."""
    h = TapHarness(hz=1000.0, criterion="2")
    params = SamplingParams(extra_args={"astra_tap": True})
    assert h.wrapper.add_request(1, 7, params) == "inner-add", (
        "the wrapper must return whatever the inner sampler returned"
    )
    assert h.sampler.added == [(1, 7, params)], "arguments reach the inner sampler unchanged"


def test_call_forwards_its_arguments_to_the_inner_sampler() -> None:
    """Criterion 2: ``(logits, input_batch, draft_logits=None)`` reaches the inner unchanged."""
    t = TapHarness()
    logits, batch, draft = object(), object(), object()
    t.wrapper(logits, batch, draft)
    assert t.sampler.calls == [(logits, batch, draft)]


# --------------------------------------------------------------------------- criterion 3


@pytest.mark.parametrize(
    "op",
    [
        ".item()",
        ".tolist()",
        ".numpy()",
        "np.asarray()",
        "bool()",
        "int()",
        "float()",
        ".nonzero()",
        ".cpu()",
        ".to(cpu) blocking",
        "copy_ D2H blocking",
        "t[i] = python scalar",
    ],
)
def test_the_spy_fires_on_every_listed_sync(op: str) -> None:
    """Criterion 3 / S8 liveness: the instrument fires on each listed op and on a blocking D2H
    copy -- otherwise "zero syncs recorded" means nothing. Green before implementation by design:
    this row tests the fake, not the plugin (its real-torch twin is in ``tests/stock``)."""
    import numpy as np

    SPY.reset()
    t = SpyTensor([[1, 2], [3, 4]], name="probe", readonly=True)
    backend = FakeTapBackend()
    own = backend.torch.zeros(2, dtype="int32", device="cuda:0")
    run = {
        ".item()": lambda: t[0, 0].item(),
        ".tolist()": lambda: t.tolist(),
        ".numpy()": lambda: t.numpy(),
        "np.asarray()": lambda: np.asarray(t),
        "bool()": lambda: bool(t[0, 0]),
        "int()": lambda: int(t[0, 0]),
        "float()": lambda: float(t[0, 0]),
        ".nonzero()": lambda: t.nonzero(),
        ".cpu()": lambda: t.cpu(),
        ".to(cpu) blocking": lambda: t.to("cpu"),
        "copy_ D2H blocking": lambda: backend.host_buffer((2, 2)).copy_(t),
        # the CUDA landmine: a host-to-device scalar write, on a tensor the tap owns (not read-only)
        "t[i] = python scalar": lambda: own.__setitem__(0, 5),
    }[op]
    with pytest.raises(AssertionError, match="host sync"):
        run()
    assert len(SPY.hits) == 1, f"{op} must record exactly one hit, got {SPY.hits}"


def test_the_spy_does_not_fire_on_what_s8_allows() -> None:
    """S8, the other side of the guard: a same-device ``.to``, a dtype change, a device view and a
    ``non_blocking`` copy into a pinned buffer are not syncs (a spy that fired on these would make
    every implementation look guilty)."""
    SPY.reset()
    t = SpyTensor([[1, 2], [3, 4]], name="probe", readonly=True)
    backend = FakeTapBackend()
    assert t.to("cuda:0") is t
    t.to(backend.torch.int32)
    t[0]
    (t == t[0:1]).all(0)
    own = backend.torch.zeros(2, 2, dtype="int32", device="cuda:0")
    own[0].fill_(3)  # a device fill with a scalar argument is a kernel argument, not a copy
    own[1].zero_()
    own[0] = t[1]  # a device tensor value: device-to-device
    buf = backend.host_buffer((2, 2))
    buf.copy_(t, non_blocking=True)
    assert SPY.hits == [], SPY.hits
    with pytest.raises(AssertionError, match="before the CUDA event"):
        buf.tolist()  # and the copied values are not readable until an event is awaited
    ev = backend.event()
    ev.record()
    ev.synchronize()
    assert buf.tolist() == [[1, 2], [3, 4]]


def test_the_spy_refuses_writes_to_what_the_tap_only_reads() -> None:
    """Criterion 5(a) liveness: ``__setitem__`` and in-place ops on a read tensor, or a view of
    one, are recorded and refused; the same ops on a tap-owned tensor are allowed."""
    SPY.reset()
    t = SpyTensor([[1, 2], [3, 4]], name="state", readonly=True)
    for write in (
        lambda: t.__setitem__(0, 5),
        lambda: t[0].fill_(1),
        lambda: t.add_(1),
        lambda: t[1].copy_(t[0]),
        lambda: t.index_copy_(0, SpyTensor([0]), t),
    ):
        with pytest.raises(AssertionError, match="only reads"):
            write()
    assert len(SPY.writes) == 5
    own = FakeTapBackend().torch.zeros(2, 2, dtype="int64", device="cuda:0")
    own[0] = t[1]
    own.index_add_(0, SpyTensor([1]), t[0:1])
    assert own._arr.tolist() == [[3, 4], [1, 2]]
    assert len(SPY.writes) == 5


def test_no_host_sync_on_the_sampler_thread() -> None:
    """Criterion 3: 200 calls of one request, consumer never runs, spy tensors throughout.

    At 30 Hz over 200 ms of offered steps (1 ms apart) some calls are suppressed and some publish,
    so both paths are exercised (the scaffold synced per step per request, *before* the rate limit,
    which is the failure this row is aimed at). n = 200 calls, no commits.
    """
    t = TapHarness(hz=30.0, queue_size=64)
    t.bind()
    for i in range(200):
        t.step(step=i % t.states.max_denoising_steps)
        t.clock.advance(0.001)
    assert t.tap.submitted >= 5, (
        f"only {t.tap.submitted} steps were staged in 200 ms at 30 Hz; the test would be vacuous"
    )
    _no_violations("200 calls")
    assert t.tap.errors == 0, "the tap swallowed an exception; a host sync would show up here"
    assert t.backend.copies, "a publishing step must copy the canvas bundle D2H"
    assert all(c.non_blocking for c in t.backend.copies), "every D2H copy must be non_blocking"
    assert t.backend.events_recorded >= 1, "a CUDA event must be recorded after the copies"
    assert t.backend.events_awaited == 0, "the sampler thread must never wait on the event"
    assert t.publisher.sent == [], "nothing may be published without the consumer draining"


def test_two_hundred_calls_with_the_consumer_blocked_never_sync_and_never_block() -> None:
    """Criterion 3, **the primary adversarial test**.

    Regime: n = 200 sampler calls over two requests in one batch, commits every 7th call (req-a)
    and every 11th (req-b), hz = 30 on a 1 ms virtual clock (suppressed, publishing and commit
    calls all occur), queue 8, CL 16, ST 8. A consumer thread is started and blocks *inside*
    ``publisher.send`` for the whole run (a subscriber that stopped reading). The sampler calls run
    on their own thread with a 20 s deadline.

    Fails if the tap syncs on anything it reads (spy log), allocates host memory or writes a read
    tensor during a call, blocks the sampler thread behind the consumer (deadline), reads a host
    buffer on the sampler thread, or waits on an event there.
    """
    gate = threading.Event()
    pub = BlockingPublisher(gate)
    h = TapHarness(hz=30.0, queue_size=8, publisher=pub, max_num_reqs=2)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    stop = threading.Event()

    def consume() -> None:
        while not stop.is_set():
            h.tap.drain()
            stop.wait(0.001)

    def produce() -> None:
        for i in range(200):
            h.call(
                [
                    Row(0, 1 + i % 40, commit=(10, i) if i % 7 == 6 else None),
                    Row(1, 1 + i % 40, commit=(20, i) if i % 11 == 10 else None),
                ]
            )
            h.clock.advance(0.001)

    consumer = threading.Thread(target=consume, daemon=True)
    producer = threading.Thread(target=produce, daemon=True)
    consumer.start()
    try:
        # the consumer must actually be stuck in send() before the run counts as "blocked"
        h.call([Row(0, 1), Row(1, 1)])
        h.clock.advance(0.05)
        assert pub.entered.wait(5.0), (
            "the consumer never reached publisher.send: nothing was drained, the row is vacuous"
        )
        producer.start()
        producer.join(20.0)
        finished = not producer.is_alive()
    finally:
        gate.set()
        stop.set()
        consumer.join(5.0)
    assert finished, "the sampler thread blocked behind a stalled consumer (criterion 3 / 5c)"
    sampler_thread = producer.ident
    _no_violations("200 calls, consumer blocked")
    assert h.tap.errors == 0, f"the tap swallowed {h.tap.errors} exception(s); {spy_report()}"
    assert h.tap.submitted >= 10, f"only {h.tap.submitted} bundles staged: the row is vacuous"
    assert h.tap.dropped > 0, "a queue of 8 behind a stalled consumer must have dropped frames"
    assert all(c.non_blocking for c in h.backend.copies)
    assert consumer.ident not in {c.thread for c in h.backend.copies}, (
        "a D2H copy was issued on the consumer thread (copies belong to the sampler thread)"
    )
    assert any(c.thread == sampler_thread for c in h.backend.copies), "the producer staged nothing"
    for buf in h.backend.buffers:
        assert sampler_thread not in buf.read_threads, (
            "a host buffer was read on the sampler thread"
        )
    for ev in h.backend.events:
        assert sampler_thread not in ev.wait_threads, "the sampler thread waited on an event"
    h.tap.drain()
    for msg in h.frames():
        validate_canvas_step(msg)


def test_published_values_are_read_only_after_another_thread_awaits_the_event() -> None:
    """Criterion 3: the D2H bundle is decoded on the consumer thread, after the event."""
    t = TapHarness(hz=1000.0, queue_size=8)
    t.bind()
    sampler_thread: list[int] = []

    def run() -> None:
        sampler_thread.append(threading.get_ident())
        t.step(step=3)

    th = threading.Thread(target=run)
    th.start()
    th.join()

    assert t.publisher.sent == []
    drained = t.tap.drain()
    assert drained == 1 and len(t.publisher.sent) == 1
    assert t.backend.events_awaited == 1
    main = threading.get_ident()
    touched = [b for b in t.backend.buffers if b.copy_threads]
    assert touched, "no host buffer was written"
    for buf in touched:
        assert set(buf.copy_threads) == set(sampler_thread), "copies belong to the sampler thread"
        assert set(buf.read_threads) <= {main}, "values must be read on the consumer thread only"
    for ev in t.backend.events:
        assert set(ev.record_threads) <= set(sampler_thread)
        assert set(ev.wait_threads) <= {main}
    _no_violations("one step")


def test_published_bundle_is_a_few_kilobytes_not_the_whole_history() -> None:
    """Criterion 3/4 budget: canvas + argmax + a reduced [CL] mask + scalars, never [ST, CL].

    With ``CL=16`` and ``ST=8`` the accepted-canvas history alone is 128 scalars; the bundle the
    task's implementation notes describe is ``2 x CL`` ids + a ``[CL]`` mask + a ``[CL]`` latched
    commit row + a handful of scalars, i.e. at most ``3 * CL + 32 = 80``. Shipping the history
    fails this bound.
    """
    t = TapHarness(hz=1000.0, canvas_length=16)
    t.bind()
    t.step(step=1)
    assert t.tap.submitted == 1
    assert t.backend.scalars_copied <= 3 * 16 + 32, (
        f"{t.backend.scalars_copied} scalars copied for one step; reduce the stable mask on the "
        "device ((hist == hist[0:1]).all(0)) instead of shipping [ST, CL]"
    )


# --------------------------------------------------------------------------- criterion 4


def _publishes_over_one_second(*, steps: int, dt: float) -> tuple[int, TapHarness]:
    t = TapHarness(hz=30.0, queue_size=4096)
    t.bind()
    for i in range(steps):
        t.step(step=i % t.states.max_denoising_steps)
        t.clock.advance(dt)
    return t.tap.submitted, t


def test_publish_count_is_bounded_by_hz_and_buffers_are_allocated_once() -> None:
    """Criterion 4: 1 kHz offered, hz=30 → ≤ 31 copies/pushes per second per request."""
    count, t = _publishes_over_one_second(steps=1000, dt=0.001)
    assert count <= 31, f"{count} publishes in 1.0 s at hz=30"
    assert count >= 25, f"only {count} publishes in 1.0 s at hz=30: the limiter is too aggressive"
    assert t.backend.events_recorded == count
    # allocated once: the number of host buffers must not grow with the number of steps
    first_round_allocations = len(t.backend.allocations)
    for i in range(1000):
        t.step(step=i % t.states.max_denoising_steps)
        t.clock.advance(0.001)
    assert len(t.backend.allocations) == first_round_allocations, (
        "host buffers are allocated per step; pre-allocate a pool (criterion 4)"
    )


def test_raising_the_offered_step_rate_tenfold_does_not_raise_the_copy_count() -> None:
    """Criterion 4, negative: the limiter runs on the clock, not on the step counter.

    Both arms offer exactly 1.0 s of steps. At 33.3 ms spacing the 1 ms grid admits a publish
    every 34 ms and the 0.1 ms grid every 33.4 ms, so the finer grid may legitimately fit one
    more inside the second — but not ten times more.
    """
    slow, _ = _publishes_over_one_second(steps=1000, dt=0.001)
    fast, _ = _publishes_over_one_second(steps=10000, dt=0.0001)
    assert fast <= 31
    assert fast <= slow + 1, f"10x the offered rate produced {fast} publishes against {slow}"


def test_host_buffers_are_allocated_when_the_sampler_is_wrapped_never_during_a_call() -> None:
    """Criterion 4: the pinned pool exists before the first call and never grows.

    n = 600 calls over three occupants of two slots (binds, commits, releases, a reuse), hz 30 on
    a 1 ms clock. A pool that is allocated lazily on the first publishing call fails the first
    assertion; one that grows on commit, release or reuse fails the second.
    """
    h = TapHarness(hz=30.0, queue_size=16)
    assert h.allocations_at_wrap > 0, (
        "no host buffer was allocated when custom_sampler wrapped the sampler (criterion 4)"
    )
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    for i in range(300):
        h.call([Row(0, 1 + i % 40, commit=(1, 2) if i % 9 == 8 else None), Row(1, 1 + i % 40)])
        h.clock.advance(0.001)
    h.release("req-a")
    h.bind(0, "req-c")
    for i in range(300):
        h.call([Row(0, 1 + i % 40, commit=(3, 4) if i % 13 == 12 else None), Row(1, 1 + i % 40)])
        h.clock.advance(0.001)
    h.release("req-b")
    h.release("req-c")
    assert len(h.backend.allocations) == h.allocations_at_wrap, (
        f"host buffers allocated during calls: {h.backend.allocations[h.allocations_at_wrap :]}"
    )
    _no_violations("600 calls")


def test_suppressed_calls_issue_no_device_to_host_copy() -> None:
    """Criterion 4: a call the limiter suppresses stages nothing -- not even for a commit.

    hz 30; call 1 publishes at t = 0, calls 2-5 follow 1 ms apart (inside the 33 ms window),
    call 4 commits. Copies and recorded events must not move after call 1."""
    h = TapHarness(hz=30.0, queue_size=16)
    h.bind()
    h.step(step=1)
    copies, events = len(h.backend.copies), h.backend.events_recorded
    assert copies > 0, "the first call must publish (otherwise this row is vacuous)"
    for s, commit in ((2, False), (3, False), (4, True), (1, False)):
        h.clock.advance(0.001)
        h.step(step=s, commit=commit)
    assert len(h.backend.copies) == copies, "a suppressed call issued a D2H copy"
    assert h.backend.events_recorded == events, "a suppressed call recorded an event"
    _no_violations("suppressed calls")


@pytest.mark.parametrize("vary", ["history_len", "step"])
def test_per_call_device_work_has_a_fixed_size(vary: str) -> None:
    """Criterion 4: per-call device work is independent of history length and step count.

    The fake meters every element produced by a device op (``SPY.work``). Two otherwise identical
    publishing calls -- one at ``history_len`` 1 / step 1, the other at ``history_len`` 8 / step 40
    -- must cost the same. A tap that slices the history by its (device) length, or accumulates
    per-step tensors, fails.
    """
    work: list[int] = []
    sizes: list[int] = []
    for value in (1, 8) if vary == "history_len" else (1, 40):
        h = TapHarness(hz=1000.0, canvas_length=8, history_steps=8)
        h.bind()
        if vary == "history_len":
            h.states.set_slot(0, history=[[5] * 8] * 8, history_len=value)
        h.step(step=1)  # warm call (anything first-call-only happens here)
        h.clock.advance(0.01)
        before = SPY.work
        h.step(step=value if vary == "step" else 2)
        work.append(SPY.work - before)
        sizes.append(h.backend.copies[-1].n if h.backend.copies else -1)
    assert work[0] == work[1], f"device work varies with {vary}: {work}"
    assert sizes[0] == sizes[1], f"copy size varies with {vary}: {sizes}"


def test_a_suppressed_commit_is_published_exactly_once_at_the_next_publishing_call() -> None:
    """Criterion 4 (rewritten, R5): every commit is published **exactly once**, lagged.

    hz 30. Call 1 publishes (t = 0). Calls 2 and 3 are 1 ms apart, inside the window; call 3 is a
    commit of ``(91, 92)`` at step 3 -- the tap cannot know that without a sync, so nothing may be
    staged. Call 4, 40 ms later, is the first call of the next canvas and publishes: the commit
    must be on the wire after it, once, with ``phase="commit"``, its step (3), the committed ids
    cut at ``num_sampled`` (2 of CL 16) and the canvas index it closes (0); the denoise frame of
    the new canvas carries ``canvas_idx`` 1 and comes after the commit.
    """
    h = TapHarness(hz=30.0, queue_size=64)
    h.bind()
    h.step(step=1)
    h.clock.advance(0.001)
    h.step(step=2)
    h.clock.advance(0.001)
    h.step(step=3, commit=True)
    h.tap.drain()
    assert [m["phase"] for m in h.frames()] == ["denoise"], (
        "only call 1 published; a commit at a suppressed call cannot be known yet"
    )
    h.clock.advance(0.040)
    h.step(step=1)
    for _ in range(5):  # later publishing calls must not publish it again
        h.clock.advance(0.040)
        h.step(step=2)
    h.tap.drain()
    frames = h.frames()
    commits = h.commit_frames()
    assert len(commits) == 1, f"the commit was published {len(commits)} times: {frames}"
    c = commits[0]
    assert tuple(c["committed_tokens"]) == (91, 92), "cut at num_sampled, never the whole row"
    assert c["step"] == 3 and c["canvas_idx"] == 0 and c["request_id"] == "req-a"
    i = frames.index(c)
    assert frames[i + 1]["phase"] == "denoise" and frames[i + 1]["canvas_idx"] == 1, (
        f"the next canvas's first frame must follow the commit with canvas_idx 1: {frames}"
    )
    assert h.tap.commit_overrun == 0
    _no_violations("lagged commit")


def test_a_commit_still_unpublished_at_release_is_flushed_by_the_release() -> None:
    """Criterion 4 + S11: "no later than the slot's next publishing call or its release".

    hz 30: call 1 publishes, call 2 (1 ms later, suppressed) commits ``(7, 8, 9)``; then the
    request finishes. The release stages at most one extra bundle and the commit reaches the wire
    exactly once with no further sampler call."""
    h = TapHarness(hz=30.0, queue_size=64)
    h.bind()
    h.step(step=1)
    h.clock.advance(0.001)
    h.step(step=2, commit=True, tokens=(7, 8, 9))
    events = h.backend.events_recorded
    h.release("req-a")
    assert h.backend.events_recorded - events <= 1, "≤ 1 bundle at release (criterion 4)"
    h.tap.drain()
    commits = h.commit_frames("req-a")
    assert [tuple(c["committed_tokens"]) for c in commits] == [(7, 8, 9)], h.frames()
    assert commits[0]["canvas_idx"] == 0
    _no_violations("release flush")


def test_every_commit_is_published_exactly_once_over_a_mixed_run() -> None:
    """Criterion 4: exactly once, whatever the limiter does. n = 120 calls, hz 30, clock steps of
    1, 5 and 40 ms in a fixed pattern, six commits (some at publishing calls, some at suppressed
    ones), then the release. The commit frames must be exactly the six, in order."""
    h = TapHarness(hz=30.0, queue_size=256)
    h.bind()
    expected: list[tuple[int, ...]] = []
    dts = (0.001, 0.005, 0.040)
    for i in range(120):
        toks: tuple[int, ...] | None = None
        if i % 20 == 19:
            toks = (100 + i, 200 + i)
            expected.append(toks)
        h.step(step=1 + i % 20, tokens=toks)
        h.clock.advance(dts[i % 3])
    h.release("req-a")
    h.tap.drain()
    got = [tuple(c["committed_tokens"]) for c in h.commit_frames()]
    assert got == expected, f"commits on the wire {got} != committed {expected}"
    assert [c["canvas_idx"] for c in h.commit_frames()] == list(range(len(expected)))
    assert h.tap.commit_overrun == 0, "one commit per window never overruns"
    _no_violations("mixed run")


@pytest.mark.parametrize("hz", [1000.0, 30.0], ids=["every-call-publishes", "commit-suppressed"])
def test_commit_advances_the_canvas_index(hz: float) -> None:
    """Criterion 4/9 (follows R5): the commit closes canvas 0 and the next canvas is index 1.

    Three calls: a denoise step, the commit, the first step of the next canvas. At hz 1000 every
    call publishes and the commit is the frame of its own call; at hz 30 the commit call (1 ms
    after the first) is suppressed and the commit is published, lagged, when the third call (40 ms
    later) publishes. Both regimes put the same three frames on the wire, in this order.
    """
    t = TapHarness(hz=hz, queue_size=64)
    t.bind()
    t.step(step=1)
    t.clock.advance(0.001 if hz < 100 else 0.01)
    t.step(step=2, commit=True)
    t.clock.advance(0.040)
    t.step(step=1)
    t.tap.drain()
    seen = [(m["phase"], m["canvas_idx"]) for m in t.publisher.sent]
    assert seen == [("denoise", 0), ("commit", 0), ("denoise", 1)], f"(phase, canvas_idx): {seen}"


def test_two_commits_in_one_rate_window_are_both_published_or_counted() -> None:
    """Criterion 4, adversarial: two commits inside one 33 ms window. Both are published, or
    ``commit_overrun`` increments for the one that is not. Never silent.

    hz 30: publish at t = 0; commit ``(11, 12)`` at +1 ms and ``(21, 22)`` at +3 ms, both
    suppressed; publish at +43 ms. Whatever the design keeps, the committed ids that reach the wire
    are an in-order subsequence of the two, every missing one is counted, and the canvas after
    them is index 2 (two canvases closed).
    """
    h = TapHarness(hz=30.0, queue_size=64)
    h.bind()
    h.step(step=1)
    h.clock.advance(0.001)
    h.step(step=2, tokens=(11, 12))
    h.clock.advance(0.001)
    h.step(step=1)
    h.clock.advance(0.001)
    h.step(step=2, tokens=(21, 22))
    h.clock.advance(0.040)
    h.step(step=1)
    h.tap.drain()
    got = [tuple(c["committed_tokens"]) for c in h.commit_frames()]
    assert got in ([(11, 12), (21, 22)], [(11, 12)], [(21, 22)]), f"commit frames {got}"
    overrun = attr(h.tap, "commit_overrun", "4")
    assert len(got) + overrun == 2, (
        f"{len(got)} commits published and commit_overrun={overrun}: a commit vanished silently"
    )
    last = h.frames()[-1]
    assert last["phase"] == "denoise" and last["canvas_idx"] == 2, h.frames()
    _no_violations("two commits in a window")


# --------------------------------------------------------------------------- criterion 5


def test_output_object_identity_and_no_mutation() -> None:
    """Criterion 5(a): ``out is inner_out`` and no device tensor is written to.

    The spy refuses every in-place operation and ``__setitem__`` on what the tap reads (state rows,
    sampler outputs, batch tensors) and records it; the log is the witness, because the catch-all
    swallows the raise.
    """
    t = TapHarness(hz=1000.0)
    t.bind()
    out = t.step(step=1, commit=True)
    assert out is t.sampler.last_output, "the runner must receive the inner SamplerOutput itself"
    assert isinstance(out, FakeSamplerOutput)
    assert t.tap.errors == 0, f"the tap raised: {spy_report()}"
    _no_violations("one commit call")


def test_tap_exception_is_swallowed_and_counted() -> None:
    """Criterion 5(b): the tap never breaks decoding, and never fails silently either.

    The state rows raise on any tap-side access for one publishing call (the fault is on the
    tensors themselves, so a tap that cached references at wrap time is faulted too). One faulted
    call may raise once before delegating and once after (guarded separately): 1 or 2.
    """
    t = TapHarness(hz=1000.0, publisher=RecordingPublisher())
    t.bind()
    t.states.fault_switch.exc = RuntimeError
    try:
        out = t.step(step=1)
    finally:
        t.states.fault_switch.exc = None
    assert out is t.sampler.last_output
    assert 1 <= t.tap.errors <= 2, (
        f"errors={t.tap.errors}: a swallowed tap failure must be counted, not discarded"
    )


def test_an_inner_exception_propagates_as_the_same_object() -> None:
    """Criterion 5(b): the inner call is never wrapped -- its exception is the engine's business,
    and it reaches the engine as the *same object*, after exactly one inner call."""
    h = TapHarness(hz=1000.0)
    h.bind()
    boom = RuntimeError("CUDA error: an illegal memory access was encountered")
    h.sampler.raises = boom
    with pytest.raises(RuntimeError) as excinfo:
        h.step(step=1)
    assert excinfo.value is boom, "re-raised as a different object (wrapped or copied)"
    assert len(h.sampler.calls) == 1, "the inner was retried or skipped"


def test_a_raise_anywhere_in_the_tap_still_runs_the_inner_exactly_once() -> None:
    """Criterion 5(b): a raise *before* delegating still runs the inner exactly once, and a raise
    after still returns the inner output.

    Everything the tap could read raises for one call: the batch's every attribute except the two
    the inner sampler reads, the state rows and the sampler's output tensors. Wherever the tap's
    work starts, it raises there. n = 1 faulted call, then 1 clean call.
    """
    h = TapHarness(hz=1000.0)
    h.bind()
    batch = FakeInputBatch([0], cl=h.states.canvas_length)
    batch.fault.exc = KeyError
    h.states.fault_switch.exc = KeyError
    try:
        out = h.call([Row(0, 1)], batch=batch)
    finally:
        h.states.fault_switch.exc = None
        batch.fault.exc = None
    assert len(h.sampler.calls) == 1, f"the inner ran {len(h.sampler.calls)} times"
    assert out is h.sampler.last_output, "the inner output must be returned despite the raise"
    assert h.tap.errors >= 1, "the raise was swallowed without being counted"


def test_a_raise_after_a_copy_returns_the_buffers_to_the_pool() -> None:
    """Criterion 5(b): 1000 injected raises leave the pool size unchanged.

    ``event.record()`` -- which comes after the copies -- raises on 1000 publishing calls, one per
    occupant (a raise latches the slot off until its next occupant, so each raise needs a fresh
    request). Every call still returns the inner output, each raise is counted once, no host
    buffer is allocated, and afterwards a clean request still publishes: a pool that leaked one
    bundle per raise would be empty long before 1000.
    """
    h = TapHarness(hz=1000.0, queue_size=16)
    h.backend.fail_record = 1000
    h.backend.fail_record_exc = RuntimeError
    for i in range(3000):
        if h.backend.record_raises >= 1000:
            break
        h.bind(0, f"req-{i}")
        out = h.step(step=1)
        assert out is h.sampler.last_output
        h.release(f"req-{i}")
        h.clock.advance(0.01)
    assert h.backend.record_raises == 1000, (
        f"only {h.backend.record_raises} record() raises were provoked; the row is vacuous"
    )
    assert h.tap.errors == h.backend.record_raises, (
        f"{h.backend.record_raises} raises but errors={h.tap.errors}: each raise counts once"
    )
    assert len(h.backend.allocations) == h.allocations_at_wrap, "the pool grew after raises"
    h.tap.drain()
    before = len(h.frames())
    h.bind(1, "req-clean")
    h.clock.advance(0.01)
    h.step(slot=1, step=1)
    h.tap.drain()
    assert len(h.frames("req-clean")) == 1 and len(h.frames()) == before + 1, (
        "after 1000 raises a clean request could not publish: the pool leaked"
    )


def test_one_traceback_per_exception_type_per_process(caplog: pytest.LogCaptureFixture) -> None:
    """Criterion 5(b): the counter counts every raise, the log carries one traceback per type.

    Two fresh exception types (so no earlier row in this process has logged them): ``TypeA``
    raised 3 times, ``TypeB`` twice, each from ``event.record()`` on a fresh occupant. Records must
    propagate to the root logger (caplog).
    """

    class TypeA(RuntimeError):
        pass

    class TypeB(RuntimeError):
        pass

    h = TapHarness(hz=1000.0)
    caplog.set_level(logging.DEBUG)
    plan = [TypeA, TypeA, TypeB, TypeA, TypeB]
    for i, exc in enumerate(plan):
        h.backend.fail_record = 1
        h.backend.fail_record_exc = exc
        h.bind(0, f"req-{i}")
        h.step(step=1)
        h.release(f"req-{i}")
        h.clock.advance(0.01)
    assert h.backend.record_raises == 5
    assert h.tap.errors == 5

    def tracebacks(t: type) -> int:
        n = 0
        for r in caplog.records:
            if (r.exc_info and r.exc_info[0] is t) or (
                t.__name__ in r.getMessage() and "Traceback" in r.getMessage()
            ):
                n += 1
        return n

    assert tracebacks(TypeA) == 1, f"TypeA logged {tracebacks(TypeA)} tracebacks (want 1)"
    assert tracebacks(TypeB) == 1, f"TypeB logged {tracebacks(TypeB)} tracebacks (want 1)"


def _latch_run(*, fault: bool) -> TapHarness:
    """req-a on slot 0 and req-b on slot 1, hz 1000, 10 ms apart. Call 3 faults the state rows
    (if ``fault``) while both slots are in the batch; calls 4-6 are clean. Then req-a finishes and
    req-c takes slot 0 for calls 7-8."""
    h = TapHarness(hz=1000.0, queue_size=64)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    for i in range(1, 7):
        if fault and i == 3:
            h.states.fault_switch.exc = RuntimeError
        try:
            h.call([Row(0, i), Row(1, i)])
        finally:
            h.states.fault_switch.exc = None
        h.clock.advance(0.01)
    h.release("req-a")
    h.bind(0, "req-c")
    for i in range(1, 3):
        h.call([Row(0, i)])
        h.clock.advance(0.01)
    h.tap.drain()
    return h


def test_a_raise_latches_the_affected_slots_off_until_their_next_occupant() -> None:
    """Criterion 5(b): after a raise, the affected slots publish nothing more for their current
    occupants -- a half-staged slot could otherwise emit a frame pairing state from two steps --
    and the latch clears for the next occupant of the slot."""
    h = _latch_run(fault=True)
    assert h.tap.errors >= 1
    a_steps = [m["step"] for m in h.frames("req-a")]
    b_steps = [m["step"] for m in h.frames("req-b")]
    assert all(s < 3 for s in a_steps + b_steps), (
        f"latched slots kept publishing: req-a steps {a_steps}, req-b steps {b_steps}"
    )
    assert [m["step"] for m in h.frames("req-c")] == [1, 2], (
        "the slot's next occupant must publish again (the latch is per occupant, not forever)"
    )


def test_a_tap_that_does_not_raise_never_trips_the_latch() -> None:
    """Criterion 5(b), the twin: the identical run without the fault publishes every call of every
    occupant -- a latch that trips on something other than a raise fails here."""
    h = _latch_run(fault=False)
    assert h.tap.errors == 0, spy_report()
    assert [m["step"] for m in h.frames("req-a")] == [1, 2, 3, 4, 5, 6]
    assert [m["step"] for m in h.frames("req-b")] == [1, 2, 3, 4, 5, 6]
    assert [m["step"] for m in h.frames("req-c")] == [1, 2]


def test_the_publisher_thread_guards_each_frame_separately() -> None:
    """Criterion 5(b) / S14: a raise while sending one frame costs that frame only, is counted,
    and never escapes ``drain()``. Three frames queued; the second ``send`` raises."""
    pub = RecordingPublisher(OSError("socket went away"), fail_on={2})
    h = TapHarness(hz=1000.0, publisher=pub)
    h.bind()
    for s in (1, 2, 3):
        h.step(step=s)
        h.clock.advance(0.01)
    h.tap.drain()
    assert [m["step"] for m in h.frames()] == [1, 3], "the frames around the bad one must go out"
    assert h.tap.errors == 1


def test_nothing_the_tap_does_raises_into_the_engine_loop() -> None:
    """Criterion 5(b): request bookkeeping is guarded too -- malformed ``extra_args``, a release of
    an unknown id, a double release and a release while the state rows are faulted all return
    super's value and raise nothing."""
    h = TapHarness(hz=1000.0)
    assert h.wrapper.add_request(0, 7, SamplingParams(extra_args=42)) == "inner-add"
    assert (
        h.wrapper.add_request(1, 7, SamplingParams(extra_args={"astra_tap": "yes?"})) == "inner-add"
    )
    assert h.ms.add_request(0, NewRequestData(req_id="req-x")) == "super-add"
    h.step(step=1)
    assert h.release("never-added") == "super-remove"
    h.states.fault_switch.exc = RuntimeError
    try:
        assert h.release("req-x") == "super-remove"
    finally:
        h.states.fault_switch.exc = None
    assert h.release("req-x") == "super-remove"


def test_full_queue_drops_oldest_without_blocking() -> None:
    """Criterion 5(c): a slow subscriber costs frames, never decode time."""
    t = TapHarness(hz=1000.0, queue_size=4)
    t.bind()
    for i in range(10):
        t.step(step=i)
        t.clock.advance(0.01)
    assert t.tap.dropped == 6, f"expected 6 dropped frames with a queue of 4, got {t.tap.dropped}"
    t.tap.drain()
    steps = [m["step"] for m in t.publisher.sent]
    assert steps == [6, 7, 8, 9], f"the *oldest* frames must be dropped, kept {steps}"


def test_warmup_and_dummy_runs_publish_nothing() -> None:
    """Criterion 5(d): the capture pass calls the sampler with a dummy batch (runner l.843)."""
    t = TapHarness(hz=1000.0)
    # dummy/profile run: no request was ever registered for the slot
    t.step(step=0)
    t.clock.advance(0.01)
    assert t.tap.submitted == 0, "a slot with no req_id must publish nothing"
    t.bind(req_id="_warmup_0")
    t.step(step=1)
    t.tap.drain()
    assert t.publisher.sent == [], "_warmup_-prefixed request ids publish nothing"
    assert t.tap.errors == 0


def test_per_request_opt_out_through_vllm_xargs_publishes_nothing() -> None:
    """Implementation notes / criterion 13: ``vllm_xargs={"astra_tap": false}`` is the B16 A arm."""
    t = TapHarness(hz=1000.0)
    t.bind(req_id="req-off", extra_args={"astra_tap": False})
    for i in range(5):
        t.step(step=i, commit=(i == 4))
        t.clock.advance(0.01)
    t.release("req-off")  # added with R5: nor at its commit, nor at the release flush (S11)
    t.tap.drain()
    assert t.publisher.sent == []
    assert t.backend.copies == [], "an opted-out request must not even be copied D2H"


def test_the_tap_is_on_by_default_with_no_xargs_at_all() -> None:
    """Implementation notes: the default is tap-on and no xargs must be required."""
    t = TapHarness(hz=1000.0)
    t.bind(req_id="req-plain", extra_args=None)
    t.step(step=0)
    t.tap.drain()
    assert [m["request_id"] for m in t.publisher.sent] == ["req-plain"]


# --------------------------------------------------------------------------- criterion 9


@pytest.mark.parametrize("model_state_first", [True, False])
def test_slot_registration_tolerates_either_arrival_order(model_state_first: bool) -> None:
    """Criterion 9: ``req_id`` and ``extra_args`` arrive from two call sites, order unspecified."""
    tapmod = load_plugin_module("tap", "9")
    slots = attr(tapmod, "SlotTable", "9")()
    if model_state_first:
        slots.bind(2, "req-x")
        slots.options(2, {"astra_tap": True})
    else:
        slots.options(2, {"astra_tap": True})
        slots.bind(2, "req-x")
    info = slots.info(2)
    assert info is not None
    assert info.request_id == "req-x"
    assert info.enabled is True
    assert info.canvas_idx == 0


@pytest.mark.parametrize("sampler_first", [False, True])
def test_either_arrival_order_publishes_through_the_whole_stack(sampler_first: bool) -> None:
    """Criterion 9, end to end: whichever of ``add_request`` (model state) and ``add_request``
    (sampler, carrying ``extra_args``) runs first, the opt-out is honoured and the id is right."""
    h = TapHarness(hz=1000.0)
    h.bind(0, "req-on", sampler_first=sampler_first)
    h.bind(1, "req-off", extra_args={"astra_tap": False}, sampler_first=sampler_first)
    h.call([Row(0, 1), Row(1, 1)])
    h.tap.drain()
    assert [m["request_id"] for m in h.frames()] == ["req-on"]


def test_slot_table_reports_nothing_before_the_request_id_arrives() -> None:
    """Criterion 9: until a slot has a ``req_id`` nothing may be published for it."""
    tapmod = load_plugin_module("tap", "9")
    slots = attr(tapmod, "SlotTable", "9")()
    slots.options(1, {"astra_tap": True})
    info = slots.info(1)
    assert info is None or info.request_id is None


def test_release_bumps_the_slot_epoch() -> None:
    """Criterion 9: the epoch is part of a frame's identity, fixed when the frame is staged."""
    tapmod = load_plugin_module("tap", "9")
    slots = attr(tapmod, "SlotTable", "9")()
    slots.bind(0, "req-old")
    first = slots.epoch(0)
    slots.release("req-old")
    slots.bind(0, "req-new")
    assert slots.epoch(0) != first, "a reused slot must be a new epoch"


def test_slot_reuse_never_pairs_one_occupants_id_with_the_others_data() -> None:
    """Criterion 9 (rewritten, R6), adversarial: free the slot *after* frames are staged, *before*
    they are drained, and let a new request take it.

    The two occupants write different payloads (canvas/argmax all 1s vs all 2s, CL 4, ST 2). Each
    occupant's frames carry its own id **and** its own payload; the new occupant starts at
    ``canvas_idx`` 0; no frame pairs one occupant's id with the other's data. The old occupant's
    frames are *published* under its own id (R6: dropping them would also drop the final commit
    S11 stages at release). n = 2 + 2 calls, hz 1000, 10 ms apart, one drain at the end.
    """
    t = TapHarness(hz=1000.0, queue_size=16, canvas_length=4, history_steps=2)
    t.bind(slot=0, req_id="req-old")
    for s in (1, 2):
        t.step(slot=0, step=s, canvas=[1, 1, 1, 1], argmax=[1, 1, 1, 1])
        t.clock.advance(0.01)
    assert t.tap.submitted >= 2, "the old occupant's frames must be staged before the slot is freed"
    t.release("req-old")
    t.bind(slot=0, req_id="req-new")
    for s in (1, 2):
        t.step(slot=0, step=s, canvas=[2, 2, 2, 2], argmax=[2, 2, 2, 2])
        t.clock.advance(0.01)
    t.tap.drain()

    old, new = t.frames("req-old"), t.frames("req-new")
    assert [m["step"] for m in old if m["phase"] == "denoise"][:2] == [1, 2], (
        f"the old occupant's staged frames must be published under its own id: {t.frames()}"
    )
    assert [m["step"] for m in new] == [1, 2]
    mixed = [m for m in old if tuple(m["canvas"]) != (1, 1, 1, 1)] + [
        m for m in new if tuple(m["canvas"]) != (2, 2, 2, 2) or tuple(m["argmax"]) != (2, 2, 2, 2)
    ]
    assert mixed == [], f"frames pairing one occupant's id with the other's payload: {mixed}"
    assert all(m["canvas_idx"] == 0 for m in new), (
        "the new occupant starts at canvas_idx 0, never the previous occupant's index"
    )
    assert {m["request_id"] for m in t.frames()} == {"req-old", "req-new"}


def test_a_request_re_added_after_preemption_continues_its_canvas_index() -> None:
    """Criterion 9: a ``req_id`` re-added after preemption continues its ``canvas_idx`` without
    going backwards (bounded tombstones, as in S10).

    req-p commits twice on slot 0 (canvases 0 and 1 closed), is preempted (released), another
    request uses slot 0, and req-p is re-added on slot 1. n = 7 calls, hz 1000, 10 ms apart.
    """
    h = TapHarness(hz=1000.0, queue_size=64)
    h.bind(0, "req-p")
    for s, toks in ((1, None), (2, (5, 6)), (1, None), (2, (7, 8)), (1, None)):
        h.step(slot=0, step=s, tokens=toks)
        h.clock.advance(0.01)
    h.release("req-p")
    h.bind(0, "req-other")
    h.step(slot=0, step=1)
    h.clock.advance(0.01)
    h.bind(1, "req-p")
    h.step(slot=1, step=1)
    h.tap.drain()
    idx = [m["canvas_idx"] for m in h.frames("req-p")]
    assert idx[:5] == [0, 0, 1, 1, 2], f"req-p before preemption: {idx}"
    assert idx[-1] >= 2 and idx == sorted(idx), (
        f"req-p's canvas_idx went backwards after it was re-added: {idx}"
    )
    assert [m["canvas_idx"] for m in h.frames("req-other")] == [0]


def test_model_state_binds_the_request_id_and_delegates_to_super() -> None:
    """Criterion 9: ``req_id`` comes from ``DiffusionGemmaModelState.add_request``."""
    cls = model_state_cls(FakeModelStateBase, _factory("9"), "9")
    state = cls()
    assert state.add_request(3, NewRequestData(req_id="req-zz")) == "super-add"
    slots = attr(state, "astra_slots", "9")
    bound = slots.info(3)
    assert bound is not None and bound.request_id == "req-zz", (
        "the model state's add_request is where req_id enters the tap (criterion 9)"
    )
    epoch = slots.epoch(3)
    assert state.remove_request("req-zz") == "super-remove"
    assert state.add_request_calls == [(3, state.add_request_calls[0][1])]
    assert state.remove_request_calls == ["req-zz"]
    after = slots.info(3)
    assert after is None or after.request_id is None, "a removed request leaves no occupant"
    assert slots.epoch(3) != epoch, "freeing the slot must move its epoch on"


def test_published_values_travel_through_the_backends_pinned_buffers() -> None:
    """Criteria 3/4: the published values come out of ``backend.host_buffer`` buffers. A tap that
    bypassed them (its own host arrays) would leave ``backend.buffers`` unwritten while still
    publishing -- and every readiness check in this file would be vacuous."""
    h = TapHarness(hz=1000.0)
    h.bind()
    h.step(step=1)
    h.tap.drain()
    assert h.frames(), "nothing published"
    assert any(b.copy_threads for b in h.backend.buffers), (
        "frames were published but no FakeHostBuffer was written: the tap is not copying through "
        "backend.host_buffer"
    )
    assert all(isinstance(b, FakeHostBuffer) for b in h.backend.buffers)


# --------------------------------------------------------------------------- review round 1 (R10-R13)
#
# Rows added by the test-author after review round 1 (docs/tasks/T-0011.md, rulings R10-R13). Every
# expected value is hand-computed from the call pattern stated in the row's docstring.

_ROWS = ("canvas", "argmax", "stable")


def _rowless(m: dict[str, Any]) -> bool:
    return all(m.get(k) is None for k in _ROWS)


class _HookLock:
    """Stands in for the tap's queue/pool lock: after every release, ``hook()`` runs (the lock is
    already free, as it would be when the sampler thread is scheduled at that instant)."""

    def __init__(self, real: Any, hook: Callable[[], None]) -> None:
        self._real = real
        self._hook = hook

    def acquire(self, *a: Any, **k: Any) -> bool:
        return bool(self._real.acquire(*a, **k))

    def release(self) -> None:
        self._real.release()
        self._hook()

    def locked(self) -> bool:
        return bool(self._real.locked())

    def __enter__(self) -> _HookLock:
        self._real.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def test_a_bundle_restaged_during_drain_never_lends_its_identity_to_another_payload() -> None:
    """Criterion 9 / R13, adversarial, deterministic: ``drain`` must read a bundle's identity and kind
    *before* its buffer goes back to the pool.

    Regime: CL 4, ST 2, hz 1000, queue 16, 10 ms apart. req-a (slot 0) stages one frame whose
    canvas/argmax are all 1s. The tap's lock is replaced by one that, every time it is released
    *inside* ``drain``, lets the sampler thread stage one req-b frame (slot 1, all 2s) -- three times
    in total. The pool is a free list, so the buffer drain just returned is the one the next
    staging takes. Each occupant's frames must carry its own id and its own payload: req-a exactly
    one frame of 1s, req-b exactly three frames of 2s. A drain that decodes ``bundle.meta`` after
    returning the bundle publishes req-a's 1s under req-b's id.
    """
    h = TapHarness(hz=1000.0, queue_size=16, canvas_length=4, history_steps=2)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    h.step(slot=0, step=1, canvas=[1, 1, 1, 1], argmax=[1, 1, 1, 1])
    h.clock.advance(0.01)
    state = {"in_drain": False, "busy": False, "fired": 0}

    def hook() -> None:
        if not state["in_drain"] or state["busy"] or state["fired"] >= 3:
            return
        state["busy"] = True
        try:
            state["fired"] += 1
            h.step(slot=1, step=state["fired"], canvas=[2, 2, 2, 2], argmax=[2, 2, 2, 2])
            h.clock.advance(0.01)
        finally:
            state["busy"] = False

    real = attr(h.tap, "_lock", "9")
    h.tap._lock = _HookLock(real, hook)
    state["in_drain"] = True
    try:
        h.tap.drain()
    finally:
        state["in_drain"] = False
        h.tap._lock = real
    h.tap.drain()
    assert state["fired"] == 3, (
        f"only {state['fired']} stagings ran inside drain: the tap's lock is not `_lock`, so this "
        "row is vacuous -- tell the test-author the lock's name"
    )
    a, b = h.frames("req-a"), h.frames("req-b")
    swapped = [m for m in a if tuple(m["canvas"]) != (1, 1, 1, 1)] + [
        m for m in b if tuple(m["canvas"]) != (2, 2, 2, 2) or tuple(m["argmax"]) != (2, 2, 2, 2)
    ]
    assert swapped == [], f"frames pairing one occupant's id with the other's payload: {swapped}"
    assert [m["step"] for m in a] == [1], f"req-a's one frame: {a}"
    assert [m["step"] for m in b] == [1, 2, 3], f"req-b's three frames, in order: {b}"
    assert h.tap.errors == 0, spy_report()


def test_concurrent_staging_and_draining_never_pair_an_id_with_another_occupants_payload() -> None:
    """Criterion 9 / R13, stress twin of the row above: a producer thread alternates req-a (slot 0,
    canvas all 1s) and req-b (slot 1, all 2s), 1500 calls each at hz 1000 on a 10 ms clock, while a
    consumer thread drains continuously; the interpreter switch interval is 1 µs so the two threads
    interleave inside ``drain``; queue 1, so the pool is 3 buffers and a returned one is reused at
    once. Every frame's payload must be its own request's; nothing raises. This is a regression
    guard, not the detector: the window is a few bytecodes wide and three runs against 4768ba1 (which
    has the R13 defect) stayed green -- the deterministic row above is the one that catches it."""
    import sys

    h = TapHarness(hz=1000.0, queue_size=1, canvas_length=4, history_steps=2)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    payload = {"req-a": (1, 1, 1, 1), "req-b": (2, 2, 2, 2)}
    stop = threading.Event()

    def consume() -> None:
        while not stop.is_set():
            h.tap.drain()

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    try:
        for i in range(1, 1501):
            h.step(slot=0, step=i % 40, canvas=[1] * 4, argmax=[1] * 4)
            h.step(slot=1, step=i % 40, canvas=[2] * 4, argmax=[2] * 4)
            h.clock.advance(0.01)
    finally:
        stop.set()
        consumer.join(10.0)
        sys.setswitchinterval(old)
    h.tap.drain()
    frames = h.frames()
    assert len(frames) >= 100, f"only {len(frames)} frames: the row is vacuous"
    wrong = [
        m
        for m in frames
        if tuple(m["canvas"]) != payload[m["request_id"]]
        or tuple(m["argmax"]) != payload[m["request_id"]]
    ]
    assert wrong == [], f"{len(wrong)} frames carry the other occupant's payload: {wrong[:3]}"
    assert h.tap.errors == 0, spy_report()


def test_a_suppressed_commit_followed_by_a_commit_on_a_publishing_call_is_never_silent() -> None:
    """Criterion 4, adversarial, the second ordering: the first commit lands on a suppressed call,
    the second on the *next publishing* call (the other row has both suppressed).

    hz 30: publish at t = 0 (step 1); +1 ms the canvas commits ``(11, 12)`` at step 2 (suppressed);
    +40 ms the next canvas commits ``(21, 22)`` at step 1 on a publishing call; +40 ms a denoise call
    publishes. Both commits are published, or ``commit_overrun`` counts the one that is not: the
    committed ids on the wire are an in-order subsequence of the two, ``published + overrun == 2``,
    each published commit carries the canvas index it closed ((11, 12) closes 0, (21, 22) closes 1)
    and the denoise frame after them is index 2."""
    h = TapHarness(hz=30.0, queue_size=64)
    h.bind()
    h.step(step=1)
    h.clock.advance(0.001)
    h.step(step=2, tokens=(11, 12))
    h.clock.advance(0.040)
    h.step(step=1, tokens=(21, 22))
    h.clock.advance(0.040)
    h.step(step=1)
    h.tap.drain()
    commits = h.commit_frames()
    got = [tuple(c["committed_tokens"]) for c in commits]
    assert got in ([(11, 12), (21, 22)], [(11, 12)], [(21, 22)]), f"commit frames {got}"
    overrun = attr(h.tap, "commit_overrun", "4")
    assert len(got) + overrun == 2, (
        f"{len(got)} commits published and commit_overrun={overrun}: a commit vanished silently"
    )
    closes = {(11, 12): 0, (21, 22): 1}
    assert [c["canvas_idx"] for c in commits] == [closes[t] for t in got], h.frames()
    last = h.frames()[-1]
    assert last["phase"] == "denoise" and last["canvas_idx"] == 2, h.frames()
    _no_violations("suppressed commit, then a commit on a publishing call")


def _r10_run(regime: str) -> tuple[TapHarness, int]:
    """The commit of ``(7, 8)`` closes canvas 0 after the canvas reached step 3; returns the harness
    and that pre-reset step. ``own-call``: hz 1000, every call publishes. ``lagged``: hz 30, the
    commit call (1 ms after a publish) is suppressed and the next call (40 ms later) publishes it.
    ``release``: hz 30, the commit call is suppressed and the request then finishes."""
    h = TapHarness(hz=1000.0 if regime == "own-call" else 30.0, queue_size=64)
    h.bind()
    h.step(step=1)
    h.clock.advance(0.01 if regime == "own-call" else 0.0005)
    h.step(step=2)
    h.clock.advance(0.01 if regime == "own-call" else 0.0005)
    h.step(step=3, tokens=(7, 8))
    if regime == "release":
        h.release("req-a")
    else:
        h.clock.advance(0.040)
        h.step(step=1)
    h.tap.drain()
    return h, 3


@pytest.mark.parametrize("regime", ["own-call", "lagged", "release"])
def test_commit_frames_carry_no_rows_and_the_pre_reset_step(regime: str) -> None:
    """R10 (amends criteria 4 and 6): one wire shape per phase.

    A ``commit`` frame carries ``committed_tokens`` and **none** of canvas/argmax/stable, whether it
    was published on its own call or lagged; its ``step`` is the step the canvas had reached (3),
    not the post-call value -- stock resets ``step`` to 0 at a commit and re-inits the canvas with
    noise (the fake does the same since R10). Every ``denoise`` frame carries all three rows and no
    committed ids. n = 3-4 calls per regime, CL 16, ST 8."""
    h, reached = _r10_run(regime)
    commits = h.commit_frames()
    assert len(commits) == 1, f"{regime}: {h.frames()}"
    c = commits[0]
    assert tuple(c["committed_tokens"]) == (7, 8) and c["canvas_idx"] == 0
    assert _rowless(c), (
        f"{regime}: a commit frame carries {[k for k in _ROWS if c.get(k) is not None]}; R10 says "
        "a commit frame has none of canvas/argmax/stable (the HUD draws it from committed_tokens)"
    )
    assert c["step"] == reached, (
        f"{regime}: commit step {c['step']}, want the pre-reset step {reached} (stock resets step "
        "to 0 during the commit call, so the post-call value is the wrong one)"
    )
    denoise = [m for m in h.frames() if m["phase"] == "denoise"]
    assert denoise, "fixture: denoise frames must be published"
    for m in denoise:
        assert all(m.get(k) is not None for k in _ROWS), f"{regime}: denoise frame missing rows {m}"
        assert not m.get("committed_tokens"), f"{regime}: denoise frame with committed ids {m}"
    for m in h.frames():
        validate_canvas_step(m)
    _no_violations(f"R10 {regime}")


def test_a_full_queue_keeps_the_release_flush_and_drops_the_oldest_step_frames() -> None:
    """R11 (amends criterion 5(c)), the reviewer's probe: final frames are protected.

    Regime: queue 2, hz 30, CL 16. req-a (slot 0) publishes at t = 0, commits ``(7, 8, 9)`` 1 ms
    later on a suppressed call, then finishes: its release stages the S11 flush, so the queue holds
    [a-step, a-release]. req-b (slot 1) then publishes three times 40 ms apart with nobody
    draining. Each of b's bundles must evict the oldest *non-release* bundle -- a-step, then b1,
    then b2 -- never the release: 3 dropped, the commit on the wire exactly once, no overrun, and
    of req-b only its newest frame (step 3)."""
    h = TapHarness(hz=30.0, queue_size=2)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    h.step(slot=0, step=1)
    h.clock.advance(0.001)
    h.step(slot=0, step=2, tokens=(7, 8, 9))
    h.release("req-a")
    for s in (1, 2, 3):
        h.clock.advance(0.040)
        h.step(slot=1, step=s)
    h.tap.drain()
    got = [tuple(c["committed_tokens"]) for c in h.commit_frames("req-a")]
    assert got == [(7, 8, 9)], (
        f"req-a's commits on the wire {got}: the release flush was evicted from a full queue while "
        f"step frames were still in it (dropped={h.tap.dropped}, "
        f"commit_overrun={h.tap.commit_overrun})"
    )
    assert h.tap.dropped == 3, f"dropped={h.tap.dropped}, want a-step, b1 and b2"
    assert h.tap.commit_overrun == 0
    assert [m["step"] for m in h.frames("req-b")] == [3]


def test_a_release_flush_dropped_from_a_queue_of_releases_is_counted_as_an_overrun() -> None:
    """R11: a release bundle is dropped only when the queue holds nothing else, and then it counts
    in ``commit_overrun`` as well as ``dropped`` (the commit it carried is gone: never silent).

    Regime: queue 1, hz 30. Call 1 (t = 0) has req-a (slot 0) and req-b (slot 1), both publish; the
    drain empties the queue. Call 2 (+1 ms, suppressed for both) commits ``(7, 8)`` for req-a and
    ``(5, 6)`` for req-b. Both requests finish with nobody draining in between: the second release
    meets a queue holding only the first. Exactly one of the two commits reaches the wire, and the
    other is in both counters (+1 each)."""
    h = TapHarness(hz=30.0, queue_size=1)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    h.call([Row(0, 1), Row(1, 1)])
    h.tap.drain()
    dropped0, overrun0 = h.tap.dropped, h.tap.commit_overrun
    h.clock.advance(0.001)
    h.call([Row(0, 2, commit=(7, 8)), Row(1, 2, commit=(5, 6))])
    h.release("req-a")
    h.release("req-b")
    h.tap.drain()
    got = sorted(tuple(c["committed_tokens"]) for c in h.commit_frames())
    assert got in ([(7, 8)], [(5, 6)]), f"commits on the wire: {got}"
    assert h.tap.dropped - dropped0 == 1, f"dropped moved by {h.tap.dropped - dropped0}"
    assert h.tap.commit_overrun - overrun0 == 1, (
        f"a release flush was dropped but commit_overrun moved by "
        f"{h.tap.commit_overrun - overrun0}: the lost commit is silent (R11)"
    )


def test_each_release_logs_one_line_with_the_request_id_and_the_counters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R12 (amends criterion 14): at each release the tap logs one line with the request's
    ``req_id`` and its counters ``submitted``, ``published``, ``dropped``, ``errors`` and
    ``commit_overrun``, at INFO or above (the tier-3 driver parses them from the engine log).

    Scenario and hand-computed values: ``release_log_scenario`` in the fakes. Each counter is read
    as ``<name>=<n>`` or ``<name>: <n>`` (the one format constraint this row imposes)."""
    import re

    caplog.set_level(logging.DEBUG)
    release_log_scenario()
    lines = [
        r
        for r in caplog.records
        if "req-log" in r.getMessage() and "commit_overrun" in r.getMessage()
    ]
    assert len(lines) == 1, (
        f"{len(lines)} log lines name req-log and commit_overrun at its release (want exactly 1): "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    rec = lines[0]
    assert rec.levelno >= logging.INFO, f"logged at {rec.levelname}; the engine log drops DEBUG"
    assert rec.name.startswith("vllm."), (
        f"R14: the release line must go through a logger under vLLM's `vllm` hierarchy (so vLLM's "
        f"handler writes it to the engine log); it went through {rec.name!r}"
    )
    msg = rec.getMessage()
    got: dict[str, int] = {}
    for name in TAP_COUNTERS:
        m = re.search(rf"\b{name}\b\s*[=:]\s*(\d+)", msg)
        assert m, f"counter {name} missing from the release line: {msg!r}"
        got[name] = int(m.group(1))
    assert got["submitted"] in RELEASE_LOG_SUBMITTED, f"submitted {got['submitted']}: {msg!r}"
    assert {k: got[k] for k in RELEASE_LOG_EXPECTED} == RELEASE_LOG_EXPECTED, msg


def test_tombstone_and_rate_limiter_tables_stay_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 9 ("bounded tombstones, as in S10") and the rate limiter's per-request keys: with the
    tombstone cap patched to 8 (``astra_vllm_plugin.tap.TOMBSTONES``), 300 requests come and go on
    two slots (bind, three calls, a commit on every third request, release; a drain every 10
    requests). No container the tap, its slot table or any plugin object it holds keeps may exceed
    cap + queue + 2 = 14 entries -- one that grows with the number of requests reaches ~300.

    The "too tight" side in the same run: req-p closes two canvases, is preempted, three other
    requests come and go, and req-p re-added continues at canvas_idx >= 2."""
    from collections import deque

    tapmod = load_plugin_module("tap", "9")
    attr(tapmod, "TOMBSTONES", "9")
    monkeypatch.setattr(tapmod, "TOMBSTONES", 8)
    h = TapHarness(hz=1000.0, queue_size=4)

    def sizes() -> dict[str, int]:
        objs = {"tap": h.tap, "slots": h.slots}
        for name, v in vars(h.tap).items():
            if type(v).__module__.startswith("astra_vllm_plugin") and hasattr(v, "__dict__"):
                objs[f"tap.{name}"] = v
        out: dict[str, int] = {}
        for oname, obj in objs.items():
            for name, v in vars(obj).items():
                if isinstance(v, (dict, list, set, deque)):
                    out[f"{oname}.{name}"] = len(v)
        return out

    h.bind(0, "req-p")
    for s, toks in ((1, None), (2, (5, 6)), (1, None), (2, (7, 8))):
        h.step(slot=0, step=s, tokens=toks)
        h.clock.advance(0.01)
        h.tap.drain()  # queue 4: drain as we go, so nothing here is lost to the queue
    h.release("req-p")
    h.tap.drain()
    for i in range(3):
        h.bind(0, f"req-mid-{i}")
        h.step(slot=0, step=1)
        h.clock.advance(0.01)
        h.release(f"req-mid-{i}")
        h.tap.drain()
    h.bind(1, "req-p")
    h.step(slot=1, step=1)
    h.clock.advance(0.01)
    h.release("req-p")
    h.tap.drain()
    idx = [m["canvas_idx"] for m in h.frames("req-p")]
    assert idx[-1] >= 2 and idx == sorted(idx), f"req-p's canvas_idx after re-add: {idx}"

    for i in range(300):
        slot = i % 2
        rid = f"req-{i}"
        h.bind(slot, rid)
        for s in (1, 2, 3):
            h.step(slot=slot, step=s, tokens=(1, 2) if s == 3 and i % 3 == 0 else None)
            h.clock.advance(0.01)
        h.release(rid)
        if i % 10 == 9:
            h.tap.drain()
    h.tap.drain()
    big = {k: n for k, n in sizes().items() if n > 8 + 4 + 2}
    assert big == {}, f"tables that grow with the number of requests (cap 8, queue 4): {big}"
    assert h.tap.errors == 0, spy_report()


class _SetupFault(RuntimeError):
    """Raised while the tap is being set up (its own type: one traceback per type per process)."""


class _PinnedAllocFault(RuntimeError):
    """A pinned host allocation that fails while ``custom_sampler`` wraps."""


@pytest.mark.parametrize("where", ["tap factory", "pinned allocation"])
def test_a_raise_while_setting_up_the_tap_returns_supers_objects_unwrapped(
    where: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 5(b) ("nothing ever raises into the engine loop") at set-up: a raise in the ``tap=``
    factory, or in the pinned-pool allocation ``custom_sampler`` performs when it wraps (criterion
    4), must not propagate. ``custom_sampler`` returns super's two objects themselves, unwrapped,
    the error is logged at WARNING or above with its type, and the model state's request
    bookkeeping keeps returning super's values afterwards."""
    caplog.set_level(logging.DEBUG)
    exc_type: type[BaseException]
    if where == "tap factory":
        exc_type = _SetupFault

        def factory(slots: Any) -> Any:
            raise _SetupFault("injected: the tap factory failed")

        cls = model_state_cls(FakeModelStateBase, factory, "5")
    else:
        exc_type = _PinnedAllocFault
        tf = _factory("5")
        tf.backend.fail_host_buffer = _PinnedAllocFault("injected: cudaHostAlloc failed")
        cls = model_state_cls(FakeModelStateBase, tf, "5")
    state = cls()
    result = state.custom_sampler(object())
    assert isinstance(result, tuple) and len(result) == 2, f"{where}: got {result!r}"
    assert result[0] is state.sampler and result[1] is state.second, (
        f"{where}: the tap could not be set up, so the runner must get super's own objects back"
    )
    logged = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and (
            (r.exc_info and r.exc_info[0] is exc_type)
            or exc_type.__name__ in r.getMessage()
            or "injected" in r.getMessage()
        )
    ]
    assert logged, f"{where}: the set-up failure was swallowed without a log line naming it"
    assert state.add_request(0, NewRequestData(req_id="req-after")) == "super-add"
    assert state.remove_request("req-after") == "super-remove"


def test_binding_and_releasing_slots_never_write_a_python_scalar_into_a_device_tensor() -> None:
    """The CUDA landmine (implementation notes, PROGRESS 2026-10-01): ``t[i] = <python scalar>`` on a
    CUDA tensor is a host-to-device sync, and the tap's bind/release run on the engine thread. The
    fake records it as a host sync (liveness: ``test_the_spy_fires_on_every_listed_sync``). n = 4
    requests over two slots, binds, a commit, releases and a slot reuse; zero syncs, zero errors."""
    h = TapHarness(hz=1000.0, queue_size=16)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    h.call([Row(0, 1), Row(1, 1)])
    h.clock.advance(0.01)
    h.call([Row(0, 2, commit=(4, 5)), Row(1, 2)])
    h.release("req-a")
    h.bind(0, "req-c")
    h.clock.advance(0.01)
    h.call([Row(0, 1), Row(1, 3)])
    h.release("req-b")
    h.release("req-c")
    h.tap.drain()
    _no_violations("bind/release")
    assert h.tap.errors == 0, "a swallowed raise in bind/release (the spy raises on a sync)"


# --------------------------------------------------------------------------- review round 2 (R14)


class _TripFault(RuntimeError):
    """The raise that latches req-trip's slot off (its own type: one traceback per type)."""


def test_release_lines_skip_warmup_and_opted_out_requests_and_include_tripped_ones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R14 (R12's edge cases): releases of ``_warmup_`` and opted-out requests log nothing; a request
    whose slot was latched off by a raise does log -- its ``errors`` count is the point.

    hz 1000, 10 ms apart: ``_warmup_0`` (slot 0) and ``req-off`` (slot 1, ``astra_tap: false``) each
    run one call and are released; then ``req-trip`` (slot 0) has its one publishing call fault at
    ``event.record()`` and is released."""
    caplog.set_level(logging.DEBUG)
    h = TapHarness(hz=1000.0, queue_size=16)
    h.bind(0, "_warmup_0")
    h.step(slot=0, step=1)
    h.release("_warmup_0")
    h.clock.advance(0.01)
    h.bind(1, "req-off", extra_args={"astra_tap": False})
    h.step(slot=1, step=1)
    h.release("req-off")
    h.clock.advance(0.01)
    h.bind(0, "req-trip")
    h.backend.fail_record = 1
    h.backend.fail_record_exc = _TripFault
    h.step(slot=0, step=1)
    assert h.tap.errors == 1, "fixture: the faulted call must count one error"
    h.release("req-trip")

    def lines(rid: str) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if rid in r.getMessage() and "commit_overrun" in r.getMessage()
        ]

    assert lines("_warmup_0") == [], "a warm-up request's release must log nothing (R14)"
    assert lines("req-off") == [], "an opted-out request's release must log nothing (R14)"
    trip = lines("req-trip")
    assert len(trip) == 1, f"a tripped request logs its release line once (R14): {trip}"
    assert "errors=1" in trip[0] or "errors: 1" in trip[0], trip[0]


def test_a_queue_of_release_bundles_drops_the_incoming_step_bundle() -> None:
    """R14 (R11's edge case): with the queue holding only release bundles, an incoming step bundle
    is the one dropped (counted in ``dropped``); the release stays and ``commit_overrun`` does not
    move (that occupant's next bundle carries its latch, so nothing is lost).

    Regime: queue 1, hz 30, CL 16. Call 1 (t = 0): req-a (slot 0) and req-b (slot 1) publish; drain.
    Call 2 (+1 ms, suppressed): req-a commits ``(7, 8)``; req-a finishes -> the queue is [a-release].
    Call 3 (+40 ms): req-b publishes step 3 into the full queue -> dropped. Drain: a's commit goes
    out once, b's step-3 frame never. Call 4 (+40 ms): req-b publishes step 4 and it goes out."""
    h = TapHarness(hz=30.0, queue_size=1)
    h.bind(0, "req-a")
    h.bind(1, "req-b")
    h.call([Row(0, 1), Row(1, 1)])
    h.tap.drain()
    dropped0, overrun0 = h.tap.dropped, h.tap.commit_overrun
    h.clock.advance(0.001)
    h.call([Row(0, 2, commit=(7, 8)), Row(1, 2)])
    h.release("req-a")
    h.clock.advance(0.040)
    h.step(slot=1, step=3)
    h.tap.drain()
    assert [tuple(c["committed_tokens"]) for c in h.commit_frames("req-a")] == [(7, 8)], (
        f"the release flush was evicted by an incoming step bundle: {h.frames()}"
    )
    assert h.tap.dropped - dropped0 == 1, f"dropped moved by {h.tap.dropped - dropped0}, want 1"
    assert h.tap.commit_overrun == overrun0, "dropping a step bundle loses no commit (R14)"
    assert [m["step"] for m in h.frames("req-b")] == [1], h.frames("req-b")
    h.clock.advance(0.040)
    h.step(slot=1, step=4)
    h.tap.drain()
    assert [m["step"] for m in h.frames("req-b")] == [1, 4], h.frames("req-b")
