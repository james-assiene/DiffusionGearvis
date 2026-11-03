"""Proving the stream landed where we asked: T-0035 criteria 4, 5 and 6 (findings F4, F5).

The most expensive failure mode in the whole audio path is silence that looks like success. The
2026-09-14 first-contact audit reproduced it: a ``PIPEWIRE_NODE`` naming a node the session
manager will not honour links the stream to ``auto_null`` and plays happily into nothing, and a
monitor capture of an absent node yields seconds of exactly 0.0 with no error. Nothing in the
device noticed either. This module is the contract that it must:

4. an unknown node name is refused **before** the stream opens, naming what was asked for and
   what exists;
5. after ``start()``, when a node was requested, the device finds its own ``Stream/*/Audio``
   node in ``pw-dump`` and checks the link peer, polling for a bounded settle window (≤ 500 ms,
   because links are made asynchronously); a peer that is not the requested node raises with
   *requested* and *actual* in the message; a ``pw-dump`` that is unavailable while a node was
   explicitly requested is an error too — an explicit request that cannot be verified is not
   honoured blindly;
6. ``sink != source`` places the two halves separately: ``PIPEWIRE_NODE`` is per-process and is
   reserved for the **input** half, and the output half is placed afterwards through the graph
   by the injectable seam ``_place_output``. The single-target path stays byte-for-byte the old
   one for ``source is None`` and for a source equal to the sink or to the sink's ``.monitor``
   (the null-sink rig, T-0021 criterion 9's bit-exact echo path).

**Backends.** The fake leg is here: ``sounddevice`` is ``tests/fakes/sounddevice_stub.py`` and
the graph is the recorded JSON of ``tests/fixtures/pipewire/`` injected at the module's single
shelling-out seam ``_pw_dump()``. The ``real`` leg of the same assertions is group B —
``tests/integration/test_audio_device_first_contact.py``, marked ``audio``, which needs
prerequisite P1 (``libportaudio2``) and ``scripts/audio_rig.sh up``; it is not parametrised in
here because with PortAudio absent group A cannot reach a device at all.

Seams pinned by this module (docs/tasks/T-0035.md criterion 6 and Implementation notes):
``_pw_dump() -> list[dict]`` at module level, and ``PortAudioDevice._place_output(sink)`` — the
named, injectable placement seam with the subprocess (``pw-link``) behind it.

Regime: PortAudio absent (P1), no graph read (every dump is recorded, libpipewire 1.0.5,
2026-09-14), 48 kHz mono f32, block 480 frames, requested latency 30 ms; the settle-window rows
assert on wall time measured with ``time.monotonic()`` on this 32-core box.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import pytest

from local_astra.conductor.audio_device import PortAudioDevice, open_device, parse_link_peers
from tests.fakes.audio_device import attr_or_fail
from tests.fakes.pipewire_dumps import (
    AUTO_NULL,
    MONITOR,
    SINK,
    WEBCAM,
    count_pw_dump_calls,
    load_dump,
    patch_pw_dump,
    peer_set,
)
from tests.fakes.sounddevice_stub import installed, last_stream

pytestmark = [pytest.mark.contract]

MODULE = "local_astra.conductor.audio_device"
ABSENT = "astra-no-such-sink"
#: criterion 5: the settle window is bounded at 500 ms; the slack is for a loaded box.
SETTLE_CEILING_S = 2.0


def is_named_error(exc: BaseException) -> bool:
    return type(exc).__module__ == MODULE and not isinstance(
        exc, KeyError | IndexError | AttributeError | TypeError
    )


def record_placement(monkeypatch: Any, *, fails: bool = False) -> list[str]:
    """Replace the placement seam; return the list of sinks it was asked to place the output on."""
    attr_or_fail(
        PortAudioDevice,
        "_place_output",
        "T-0035 criterion 6: the named, injectable placement seam (pw-link behind it), so "
        "group B can replace the mechanism without touching a caller",
    )
    placed: list[str] = []

    def place(self: Any, sink: str, *args: Any, **kwargs: Any) -> None:
        placed.append(sink)
        if fails:
            raise RuntimeError("pw-link failed (injected)")

    monkeypatch.setattr(PortAudioDevice, "_place_output", place)
    return placed


def test_an_unknown_sink_name_is_refused_before_the_stream_opens(monkeypatch: Any) -> None:
    """Criterion 4: name a node that is not there and nothing opens at all.

    ``dump_no_sink.json`` is this box's graph before the rig is up: the only sink is
    ``auto_null``. Asking for ``astra-no-such-sink`` must raise, quoting the name asked for and
    the names that exist, and must never fall back to the default device — the fallback is
    exactly how the audit ended up playing into ``auto_null``. The stub records that no stream
    was ever constructed, which is the difference between "refused" and "refused afterwards".
    """
    patch_pw_dump(monkeypatch, "dump_no_sink")
    with installed() as stub:
        with pytest.raises(Exception) as caught:
            open_device(sink=ABSENT, block_ms=10)
        message = str(caught.value)
        assert is_named_error(caught.value), (
            f"an unknown node raised {type(caught.value).__name__}: criterion 4 asks for a named "
            f"error of audio_device"
        )
        assert ABSENT in message and AUTO_NULL in message, (
            f"the refusal must quote the name asked for ({ABSENT!r}) and the sink/source names "
            f"that do exist ({AUTO_NULL!r}); it said: {message}"
        )
        assert stub.streams == [], (
            f"{len(stub.streams)} stream(s) were constructed before the refusal: criterion 4 "
            f"refuses *before* the stream opens"
        )


def test_a_stream_that_landed_on_auto_null_raises(monkeypatch: Any) -> None:
    """Adversarial, headline, F4: the failure the audit actually reproduced.

    ``dump_auto_null.json``: ``astra-test-sink`` exists (so criterion 4 passes it), the stream
    opens fine, and the links end on ``auto_null``. ``start()`` must raise naming **both** the
    requested and the actual peer — today it returns normally and every subsequent number
    (latency, peak, xruns) is measured on a stream nobody can hear.
    """
    patch_pw_dump(monkeypatch, "dump_auto_null")
    with installed():
        device = open_device(sink=SINK, block_ms=10)
        start = time.monotonic()
        with pytest.raises(Exception) as caught:
            device.start()
        elapsed = time.monotonic() - start
        device.stop()
    message = str(caught.value)
    assert is_named_error(caught.value), (
        f"the placement check raised {type(caught.value).__name__}; criterion 5 asks for a named "
        f"error"
    )
    assert SINK in message and AUTO_NULL in message, (
        f"the error must print requested ({SINK!r}) and actual ({AUTO_NULL!r}); it said: {message}"
    )
    assert elapsed < SETTLE_CEILING_S, (
        f"start() took {elapsed:.2f} s to refuse; the settle window is bounded at 500 ms "
        f"(criterion 5), not a retry loop with no end"
    )


def test_the_settle_window_tolerates_a_late_link(monkeypatch: Any) -> None:
    """Criterion 5: links are asynchronous, so the check polls — and a late link is fine.

    The graph answers "no links yet" for the first polls and then shows the rig linked. A check
    that read ``pw-dump`` once would fail here, and on the rig it would fail intermittently,
    which is worse: an intermittent refusal teaches people to pass ``--force``.
    """
    calls = patch_pw_dump(monkeypatch, "dump_not_linked_yet", "dump_not_linked_yet", "dump_rig_ok")
    with installed():
        device = open_device(sink=SINK, source=MONITOR, block_ms=10)
        start = time.monotonic()
        device.start()
        elapsed = time.monotonic() - start
        device.stop()
    assert elapsed < SETTLE_CEILING_S, f"start() waited {elapsed:.2f} s for a link that came"
    assert len(calls) >= 3, (
        f"_pw_dump was called {len(calls)} time(s): the settle window is a poll, not a single "
        f"read (criterion 5, and not a sleep-and-hope either)"
    )


def test_a_missing_pw_dump_with_an_explicit_node_is_an_error_not_a_shrug(monkeypatch: Any) -> None:
    """Criterion 5's last sentence: an explicit request that cannot be verified is not honoured.

    ``pw-dump`` is missing (a container without pipewire-utils, a stripped box). The device may
    refuse at ``open_device()`` or at ``start()`` — both are "it said so" — but it may not open a
    stream on the default sink and report success.
    """
    patch_pw_dump(monkeypatch, FileNotFoundError("pw-dump: command not found"))
    with installed():
        with pytest.raises(Exception) as caught:
            device = open_device(sink=SINK, block_ms=10)
            try:
                device.start()
            finally:
                device.stop()
        assert is_named_error(caught.value), (
            f"a missing pw-dump surfaced as {type(caught.value).__module__}."
            f"{type(caught.value).__name__}: criterion 5 wants a named error of audio_device "
            f"that says the explicit request could not be verified"
        )
        assert "pw-dump" in str(caught.value).lower()


def test_no_node_requested_means_no_pw_dump_call(monkeypatch: Any) -> None:
    """Criterion 4, pinned unchanged: the plain-default path costs no subprocess.

    Nobody named a node, so there is nothing to verify and nothing to refuse: the default device
    is what the session manager decides, exactly as today. This row is the fence around the new
    checks — a device that shells out on every ``start()`` would add a process spawn to the
    Conductor's startup path and, worse, would make the callback-free promise of criterion 10(d)
    harder to keep.
    """
    calls = count_pw_dump_calls(monkeypatch)
    with installed():
        device = open_device(block_ms=10)
        device.start()
        device.stop()
    assert calls == [], f"_pw_dump was called {len(calls)} time(s) with no node requested"


def test_the_null_sink_rig_path_is_byte_for_byte_the_old_one(monkeypatch: Any) -> None:
    """Criterion 6's carve-out, pinned unchanged: the rig that is measured and working.

    ``sink=astra-test-sink``, ``source=astra-test-sink.monitor`` is T-0021 criterion 9's
    bit-exact echo path and the regime of every audio number in the tree. What the stream sees at
    construction must stay exactly what ``_pipewire_target`` produces today —
    ``PIPEWIRE_NODE=astra-test-sink`` and ``stream.capture.sink = true`` — and no output re-link
    may be attempted, because both halves are one node already.

    It also pins the trap in criterion 4: ``astra-test-sink.monitor`` is **not** a node name; no
    PipeWire graph contains one. A refusal that looked the *source* up among the source nodes
    would refuse the only rig this project has.
    """
    patch_pw_dump(monkeypatch, "dump_rig_ok", required=False)
    placed: list[str] = []
    if hasattr(PortAudioDevice, "_place_output"):
        placed = record_placement(monkeypatch)
    with installed() as stub:
        device = open_device(sink=SINK, source=MONITOR, block_ms=10, latency_ms=30.0)
        device.start()
        try:
            stream = last_stream(stub)
            assert stream.kind == "duplex", "a source makes the device duplex (T-0021 crit. 1)"
            assert stream.env["PIPEWIRE_NODE"] == SINK, (
                f"PIPEWIRE_NODE={stream.env['PIPEWIRE_NODE']!r} at construction, expected "
                f"{SINK!r}: a monitor capture connects to the sink's own node"
            )
            assert stream.env["PIPEWIRE_LATENCY"] == "480/48000", (
                f"PIPEWIRE_LATENCY={stream.env['PIPEWIRE_LATENCY']!r}, expected the 480-frame "
                f"request of a 10 ms block at 48 kHz"
            )
            props = stream.env["PIPEWIRE_PROPS"] or ""
            assert re.search(r"stream\.capture\.sink\s*=\s*true", props), (
                f"PIPEWIRE_PROPS={props!r}: without stream.capture.sink=true the capture half "
                f"lands on the default source — the webcam mic (T-0021 ruling A11)"
            )
        finally:
            device.stop()
    assert placed == [], (
        f"the output half was re-linked to {placed} on the single-node rig path: criterion 6 "
        f"leaves that path byte-for-byte alone"
    )


def test_sink_differs_from_source_places_the_output_half_and_the_input_half_separately(
    monkeypatch: Any,
) -> None:
    """Adversarial, F5: two halves, one ``PIPEWIRE_NODE`` — so the output half needs a mechanism.

    ``sink=astra-test-sink``, ``source=`` the webcam mic. ``PIPEWIRE_NODE`` is per-process, so
    criterion 6 reserves it for the **input** half and places the output half afterwards through
    the graph. Today both halves are pointed at the *source's* node and the acoustic regime
    T-0023 needs (real sink → speaker → mic) has no working device path at all: the output would
    go wherever the session manager put it.
    """
    placed = record_placement(monkeypatch)
    patch_pw_dump(monkeypatch, "dump_sink_differs")
    with installed() as stub:
        device = open_device(sink=SINK, source=WEBCAM, block_ms=10)
        device.start()
        try:
            stream = last_stream(stub)
            assert stream.env["PIPEWIRE_NODE"] == WEBCAM, (
                f"PIPEWIRE_NODE={stream.env['PIPEWIRE_NODE']!r}: the one per-process variable is "
                f"the input half's (criterion 6)"
            )
            props = stream.env["PIPEWIRE_PROPS"] or ""
            assert not re.search(r"stream\.capture\.sink\s*=\s*true", props), (
                f"PIPEWIRE_PROPS={props!r}: {WEBCAM!r} is a real source, not a sink's monitor"
            )
            assert placed == [SINK], (
                f"the placement seam was called with {placed}, expected [{SINK!r}]: the output "
                f"half is linked to the requested sink after start() (criterion 6)"
            )
        finally:
            device.stop()


def test_a_placement_that_cannot_be_verified_raises_rather_than_defaulting(
    monkeypatch: Any,
) -> None:
    """Criterion 6's last sentence: a seam that returns is not a proof.

    The placement seam succeeds (``pw-link`` exited 0) and the graph still shows the output half
    on ``auto_null``. ``start()`` raises: it never leaves the output half on the default sink,
    and it never trusts the mechanism over the graph. There is deliberately no opt-out — an
    escape hatch here is the failure mode (Implementation notes).
    """
    placed = record_placement(monkeypatch)
    patch_pw_dump(monkeypatch, "dump_auto_null")
    with installed():
        device = open_device(sink=SINK, source=WEBCAM, block_ms=10)
        with pytest.raises(Exception) as caught:
            device.start()
        device.stop()
    assert is_named_error(caught.value), (
        f"unverified placement raised {type(caught.value).__name__}, not a named error"
    )
    assert SINK in str(caught.value) and AUTO_NULL in str(caught.value), (
        f"the error must name requested and actual; it said: {caught.value}"
    )
    assert placed == [SINK], (
        f"the seam was called with {placed}: the refusal must come from the *verification*, not "
        f"from skipping the placement"
    )


def test_two_peers_on_one_half_is_a_placement_failure_naming_both(monkeypatch: Any) -> None:
    """Adversarial, ruling A9: ``pw-link`` succeeded *in addition to* the session manager.

    ``dump_two_peers.json`` is the graph after the sequence the ruling names: the session manager
    linked our output half to ``auto_null`` (this box's default sink) when the stream appeared,
    then ``_place_output`` ran ``pw-link`` and added ``astra-test-sink``. **Both** links are
    live, so half of every block plays into a node nobody asked for, and none of the numbers
    measured afterwards belong to the requested sink alone.

    Criterion 5's proof is therefore set equality, not membership: the half's peer set must be
    exactly ``{requested}``. The ruling is equally clear about the remedy — this is a
    ``PlacementError`` naming both peers, not a silent ``pw-link -d``: unlinking somebody else's
    link from inside ``start()`` is a second uninspected mechanism in the place this task exists
    to make inspectable.

    Red before the ruling is implemented for the right reason: the auto_null links come first in
    the fixture, so a last-link-wins parser answers ``astra-test-sink``, the check passes and
    ``start()`` returns normally.
    """
    placed = record_placement(monkeypatch)
    patch_pw_dump(monkeypatch, "dump_two_peers")
    with installed():
        device = open_device(sink=SINK, source=WEBCAM, block_ms=10)
        start = time.monotonic()
        with pytest.raises(Exception) as caught:
            device.start()
        elapsed = time.monotonic() - start
        device.stop()
    message = str(caught.value)
    assert is_named_error(caught.value), (
        f"an output half linked to two nodes raised {type(caught.value).__name__}: ruling A9 "
        f"asks for a PlacementError. Message: {message}"
    )
    assert type(caught.value).__name__ == "PlacementError", (
        f"the error is {type(caught.value).__name__}; the placement of a stream half that landed "
        f"on two nodes is a PlacementError (ruling A9)"
    )
    assert AUTO_NULL in message and SINK in message, (
        f"the error must name both peers of the output half ({AUTO_NULL!r} and {SINK!r}) so the "
        f"reader can see which extra link to remove; it said: {message}"
    )
    assert placed == [SINK], (
        f"the seam was called with {placed}: the refusal comes from the verification, which is "
        f"what makes it a proof rather than a report of what pw-link returned"
    )
    assert elapsed < SETTLE_CEILING_S, (
        f"start() took {elapsed:.2f} s to refuse a graph that will never change: two peers is "
        f"not a state the 500 ms settle window can resolve"
    )


def test_the_verified_peer_set_is_exactly_the_requested_node(monkeypatch: Any) -> None:
    """Ruling A9, the same rule stated on the graph the device accepts (non-vacuity).

    The rig graph the previous row's fixture is derived from: the output half's peers are exactly
    ``{astra-test-sink}`` and the input half's exactly ``{the webcam}``, which is why
    ``start()`` returns there and raises on ``dump_two_peers.json``. Asserted through the parser
    on the same dump the device verifies, so "set equality" is pinned as the rule and not as an
    accident of one fixture.
    """
    placed = record_placement(monkeypatch)
    patch_pw_dump(monkeypatch, "dump_sink_differs")
    with installed():
        device = open_device(sink=SINK, source=WEBCAM, block_ms=10)
        device.start()
        device.stop()
    assert placed == [SINK]
    peers = parse_link_peers(load_dump("dump_sink_differs", pid=os.getpid()), os.getpid())
    assert peer_set(peers, "output") == {SINK}, (
        f"dump_sink_differs.json links the output half to {SINK!r} alone; the device accepted "
        f"this graph, so that acceptance means set equality and nothing weaker: {peers!r}"
    )
    assert peer_set(peers, "input") == {WEBCAM}


def test_a_refused_placement_leaves_no_stream_running(monkeypatch: Any) -> None:
    """Ruling A13 (the reviewer's one-liner): ``start()`` that raises owns no open stream.

    Every other row here proves ``start()`` refuses a stream that landed somewhere nobody asked
    for. This one is about what is left behind when it does: the PortAudio stream has already
    been constructed and started by the time the graph is read, so a ``PlacementError`` thrown
    past it leaves a live callback playing into ``auto_null`` — the exact state the refusal
    exists to prevent — plus a PortAudio thread that outlives the object and an ``atexit`` hook
    that fires into a finalised interpreter. The caller cannot clean it up either: ``start()``
    raised, so no caller has reason to believe a stream exists, and the audit's own reproduction
    script did not call ``stop()``.

    So the device stops **and** closes the stream before the error leaves ``start()``. Asserted
    on the stub's own counters, with no ``device.stop()`` in between: ``stops`` and ``closes``
    are each at least one, and the stub reports the stream inactive.
    """
    patch_pw_dump(monkeypatch, "dump_auto_null")
    with installed() as stub:
        device = open_device(sink=SINK, block_ms=10)
        with pytest.raises(Exception) as caught:
            device.start()
        assert type(caught.value).__name__ == "PlacementError", (
            f"start() raised {type(caught.value).__name__} on dump_auto_null.json; this row is "
            f"about what a PlacementError leaves behind"
        )
        assert len(stub.streams) == 1, (
            f"{len(stub.streams)} stream(s) were constructed: non-vacuity — the point of this "
            f"row is that a stream *had* been opened before the graph was read"
        )
        stream = last_stream(stub)
        assert stream.starts == 1, "non-vacuity: the stream was started before the refusal"
        assert stream.stops >= 1, (
            "start() raised PlacementError with the stream still running: nothing calls stop() "
            "after a start() that raised, so the callback keeps playing into auto_null and the "
            "atexit hook outlives the object (ruling A13, reviewer's one-liner)"
        )
        assert stream.closes >= 1, (
            "the stream was stopped but never closed: a closed PortAudio stream is what releases "
            "the device and the callback thread"
        )
        assert not stream.active, f"the stream is still active after a refused start(): {stream!r}"
        device.stop()
