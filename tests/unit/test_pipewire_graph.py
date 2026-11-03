"""The three pure ``pw-dump`` parsers of T-0035: criteria 3, 4 and 5 (findings F3 and F4).

No subprocess, no device, no graph: every input is recorded JSON from
``tests/fixtures/pipewire/`` (libpipewire 1.0.5, this box, 2026-09-14 — see its ``MANIFEST.md``
for what was recorded and what was constructed, and for the values read off the files by hand
that this module asserts against). Keeping the parsing pure is what makes criteria 4 and 5
testable before prerequisite P1.

The three functions, pinned here so the implementer builds to them (docs/tasks/T-0035.md,
Implementation notes: "Keep the parsers pure"):

``parse_nodes(dump) -> dict[str, list[str]]``
    ``{"sinks": [...], "sources": [...]}``, sorted node names of the ``Audio/Sink`` and
    ``Audio/Source`` nodes. This is the list criterion 4 prints when it refuses a name, and the
    list the smoke script shows (criterion 8). A ``Video/Source`` is not an audio source; nodes
    with a ``node.name`` and no ``media.class`` (``Dummy-Driver``, ``Freewheel-Driver``) are in
    neither list. Monitors are **not** nodes: no PipeWire graph contains a node named
    ``astra-test-sink.monitor``, which is why criterion 4 may not look a ``.monitor`` source up
    in this list.

``parse_link_peers(dump, pid) -> dict[str, str | None]``
    the peers of *that process's* stream halves: key ``"output"`` for its
    ``Stream/Output/Audio`` node, ``"input"`` for its ``Stream/Input/Audio`` node; the value is
    the peer node's name, or ``None`` when the half exists but is not linked yet. A half the
    process does not have is **absent** from the mapping, so "not linked yet" and "no such half"
    stay distinguishable inside the settle window.

``parse_quantum(dump, pid) -> float | None``
    the quantum **requested** by that process's stream node, in **ms**, or ``None`` when the
    graph does not say. Ruling A13: ``node.latency`` is the ``PIPEWIRE_LATENCY`` the client asked
    for, echoed back by the graph verbatim (even outside the graph's own clamp); ``pw-dump``
    exposes no live quantum, so this parser reads a *request* and nothing else. The live graph
    quantum is a ``pw-top`` measurement, printed by the smoke script.

The pid matters and is the part a parser is most likely to fake: in a real dump the **node**
objects carry no pid at all, only ``client.id``; the pid lives on the ``Client`` object as
``application.process.id`` (MANIFEST.md, "The pid"). A parser that returns the first stream node
it finds passes nothing here — ``test_a_different_pid_sees_nothing_of_ours`` is that mutation.

A malformed dump raises the module's own named error: never ``KeyError``/``IndexError``, and
never a silent empty answer that would let criterion 5 conclude "no link, therefore fine".
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.fakes.audio_device import attr_or_fail, import_or_fail
from tests.fakes.pipewire_dumps import AUTO_NULL, SINK, WEBCAM, load_dump, peer_set

MODULE = "local_astra.conductor.audio_device"
#: any pid at all: the parsers are pure, so it need not be a live process
PID = 4242
OTHER_PID = 909090


def parser(name: str) -> Any:
    module = import_or_fail(MODULE, "T-0035 criteria 3-5")
    return attr_or_fail(module, name, f"T-0035: the pure parser {name}(...) over a pw-dump")


def is_named_error(exc: BaseException) -> bool:
    """A named error of the module under test, not a leaked builtin (criteria 4, 5)."""
    return type(exc).__module__ == MODULE and not isinstance(
        exc, KeyError | IndexError | AttributeError | TypeError
    )


def test_node_names_are_extracted_from_a_recorded_dump() -> None:
    """Criterion 4's "the sink/source names that *do* exist", read off ``dump_rig_ok.json``.

    By hand from the fixture (MANIFEST.md): two ``Audio/Sink`` nodes, ``astra-test-sink`` and
    ``auto_null``; one ``Audio/Source`` node, the Logitech C925e webcam mic. The v4l2
    ``Video/Source`` node and the two driver nodes (``Dummy-Driver``, ``Freewheel-Driver``, which
    have a name and no ``media.class``) are in neither list, and neither are the two
    ``Stream/*/Audio`` nodes of our own client.
    """
    nodes = parser("parse_nodes")(load_dump("dump_rig_ok"))
    assert sorted(nodes["sinks"]) == [SINK, AUTO_NULL], (
        f"sinks: {nodes['sinks']}, expected {[SINK, AUTO_NULL]} from dump_rig_ok.json"
    )
    assert sorted(nodes["sources"]) == [WEBCAM], (
        f"sources: {nodes['sources']}, expected exactly the webcam mic — a Video/Source is not "
        f"an audio source and the driver nodes have no media.class"
    )
    flat = list(nodes["sinks"]) + list(nodes["sources"])
    assert "astra-conductor" not in flat, "a Stream/*/Audio node is not a sink or a source"
    assert f"{SINK}.monitor" not in flat, (
        "a monitor is not a node: no PipeWire graph contains a node named "
        f"{SINK}.monitor, so criterion 4 may not look one up here"
    )


def test_the_link_peer_of_this_process_stream_node_is_found() -> None:
    """Criterion 5 on the rig graph: both halves landed on ``astra-test-sink``.

    Read off ``dump_rig_ok.json`` by hand: the output node's two links end on the sink's
    playback ports, the input node's link starts at the sink's own ``monitor_FL`` port — a
    monitor capture links from the **sink node**, so the input peer is ``astra-test-sink``, not
    ``astra-test-sink.monitor``.
    """
    peers = parser("parse_link_peers")(load_dump("dump_rig_ok", pid=PID), PID)
    assert sorted(peers) == ["input", "output"], (
        f"peers of pid {PID} are {peers}: this process has exactly two stream halves in "
        f"dump_rig_ok.json"
    )
    assert peer_set(peers, "output") == {SINK}, (
        f"the output half's peers are {peers['output']!r}, expected exactly {{{SINK!r}}} "
        f"(dump_rig_ok.json: its two links end on that node's playback ports)"
    )
    assert peer_set(peers, "input") == {SINK}, (
        f"the input half's peers are {peers['input']!r}, expected exactly {{{SINK!r}}}"
    )


def test_a_different_pid_sees_nothing_of_ours() -> None:
    """The pid is not decoration: another process has no stream halves in this dump.

    A parser that returns "the first ``Stream/*/Audio`` node it finds" would report the rig's
    peers to any caller, and criterion 5's check on a box with a browser playing audio would
    verify somebody else's stream.
    """
    peers = parser("parse_link_peers")(load_dump("dump_rig_ok", pid=PID), OTHER_PID)
    assert peers == {}, f"pid {OTHER_PID} owns no stream node in dump_rig_ok.json, got {peers}"


def test_the_auto_null_dump_reports_auto_null() -> None:
    """Adversarial, F4: the failure the audit reproduced must be visible in the parser.

    ``dump_auto_null.json`` is the graph as it was when ``PIPEWIRE_NODE`` named a node the
    session manager would not honour: ``astra-test-sink`` exists, our stream node carries
    ``target.object = "astra-test-sink"``, and the two links end on **``auto_null``**. A parser
    that reports what was asked for — from ``target.object``, say — passes every other row here
    and fails this one, which is the whole point of criterion 5.
    """
    dump = load_dump("dump_auto_null", pid=PID)
    peers = parser("parse_link_peers")(dump, PID)
    assert peer_set(peers, "output") == {AUTO_NULL}, (
        f"the output half's peers are {peers.get('output')!r}; in dump_auto_null.json the links "
        f"end on {AUTO_NULL!r} while target.object still says {SINK!r} (finding F4)"
    )
    assert "input" not in peers, (
        f"there is no Stream/Input/Audio node in dump_auto_null.json, so the mapping must not "
        f"claim an input half: {peers}"
    )
    assert SINK in parser("parse_nodes")(dump)["sinks"], (
        "non-vacuity: the requested sink does exist in this dump — the stream landed elsewhere"
    )


def test_an_absent_node_is_absent() -> None:
    """Criterion 4: ``dump_no_sink.json`` is the graph before ``scripts/audio_rig.sh up``."""
    dump = load_dump("dump_no_sink", pid=PID)
    nodes = parser("parse_nodes")(dump)
    assert SINK not in nodes["sinks"], f"{SINK!r} must not be in {nodes['sinks']}"
    assert nodes["sinks"] == [AUTO_NULL], (
        f"the only sink on this box is {AUTO_NULL!r} (every ALC1220 analog profile reports "
        f"available: no), got {nodes['sinks']}"
    )
    assert parser("parse_link_peers")(dump, PID) == {}


def test_a_half_that_is_not_linked_yet_is_empty_not_missing() -> None:
    """Criterion 5's settle window needs the two states apart, and the fixture has them.

    ``dump_not_linked_yet.json`` is ``dump_rig_ok.json`` a few milliseconds earlier: both stream
    nodes exist, no ``Link`` object does. "Linked to nothing yet" (poll again) must not read as
    "linked to the right thing" (proceed) nor as "no such half" (give up).
    """
    peers = parser("parse_link_peers")(load_dump("dump_not_linked_yet", pid=PID), PID)
    assert sorted(peers) == ["input", "output"], (
        f"both halves exist in dump_not_linked_yet.json, so both keys are present; got {peers}"
    )
    assert peer_set(peers, "output") == frozenset(), (
        f"the output half is linked to nothing yet, which is an empty peer set, not "
        f"{peers['output']!r} (ruling A9)"
    )
    assert peer_set(peers, "input") == frozenset(), (
        f"the input half is linked to nothing yet, got {peers['input']!r}"
    )


def test_the_requested_quantum_is_read_from_the_recorded_dump() -> None:
    """Criterion 3 under ruling A13: ``node.latency`` of *our* stream node, in ms, or ``None``.

    What this reads is the **request**: ``node.latency`` is the ``PIPEWIRE_LATENCY`` the client
    set, echoed back by the graph, so a value here proves the plugin took our request and nothing
    about what the graph is running at.

    Hand-computed from the fixtures: ``480/48000 s = 10.0 ms`` in ``dump_rig_ok.json`` (the
    480-frame block this device asks for) and ``1024/48000 s = 21.333… ms`` in
    ``dump_quantum_1024.json`` (a client that requested PipeWire's default-sized block — the same
    field, a different request). ``dump_no_sink.json`` has no stream node of ours at all, so the
    answer is ``None`` (unknown), not 0.0 and not a guess.
    """
    parse_quantum = parser("parse_quantum")
    assert parse_quantum(load_dump("dump_rig_ok", pid=PID), PID) == pytest.approx(10.0, abs=1e-6)
    assert parse_quantum(load_dump("dump_quantum_1024", pid=PID), PID) == pytest.approx(
        1000.0 * 1024 / 48000, abs=1e-6
    )
    assert parse_quantum(load_dump("dump_no_sink", pid=PID), PID) is None


def test_malformed_dump_does_not_crash_the_parser() -> None:
    """Criteria 4 and 5: bad JSON is a named error, never a ``KeyError`` and never a shrug.

    ``pw-dump`` can be truncated (killed mid-write), absent, or — on a future PipeWire — shaped
    differently. The three parsers are the boundary where ``json.loads``' ``Any`` is narrowed
    (Implementation notes), so the boundary is what is tested: a truncated string, a JSON object
    instead of the array ``pw-dump`` emits, ``None``, and a list of things that are not objects
    all raise an error defined in ``audio_device`` itself. An object with no ``info.props`` — the
    ``Core``, ``Metadata`` and ``Device`` entries of every real dump have shapes the parsers do
    not care about — is skipped in silence, as ``dump_recorded_base.json`` already proves.
    """
    functions = [
        (parser("parse_nodes"), ()),
        (parser("parse_link_peers"), (PID,)),
        (parser("parse_quantum"), (PID,)),
    ]
    broken: list[Any] = ['[{"id": 1, "type": "PipeWire:Inter', {"nodes": []}, None, [1, 2, 3], 7]
    for func, extra in functions:
        for bad in broken:
            with pytest.raises(Exception) as caught:
                func(bad, *extra)
            assert is_named_error(caught.value), (
                f"{func.__name__}({json.dumps(bad) if bad is not None else 'None'!s:.40}…) raised "
                f"{type(caught.value).__module__}.{type(caught.value).__name__}; a malformed dump "
                f"is a named error of audio_device, not a leaked KeyError/IndexError/TypeError"
            )
    well_formed_but_empty = [{"id": 1, "type": "PipeWire:Interface:Node"}, {"id": 2}]
    assert parser("parse_nodes")(well_formed_but_empty) == {"sinks": [], "sources": []}
    assert parser("parse_link_peers")(well_formed_but_empty, PID) == {}
    assert parser("parse_quantum")(well_formed_but_empty, PID) is None


def test_an_output_half_linked_to_two_nodes_reports_both() -> None:
    """Adversarial, ruling A9: the residual a single-name answer cannot say.

    ``dump_two_peers.json`` is ``dump_sink_differs.json`` with the two links the session manager
    had already made to ``auto_null`` still live beside the two ``pw-link`` made to
    ``astra-test-sink`` (MANIFEST.md gives every edit and the hand count: peers ``auto_null`` and
    ``astra-test-sink`` for the output half, the webcam for the input half). The output half
    therefore plays into two nodes at once and half the signal goes somewhere nobody asked for.

    The auto_null links come **first** in the file, as they did in time, so a parser that keeps
    the last matching link per half answers ``astra-test-sink`` — exactly what was requested —
    and the placement check of criterion 5 finds nothing wrong. That is the mutation this row
    exists to kill: the answer has to be both names.
    """
    dump = load_dump("dump_two_peers", pid=PID)
    peers = parser("parse_link_peers")(dump, PID)
    assert peer_set(peers, "output") == {AUTO_NULL, SINK}, (
        f"the output half's peers are {peers.get('output')!r}; dump_two_peers.json links it to "
        f"both {AUTO_NULL!r} (links 139, 144) and {SINK!r} (links 138, 143), and a placement "
        f"that reports only the requested one is the residual of ruling A9"
    )
    assert peer_set(peers, "input") == {WEBCAM}, (
        f"the input half's peers are {peers.get('input')!r}: only the webcam link (171) ends on "
        f"it, so the two-peer answer above is about the output half and not about the parser "
        f"collapsing every link in the graph into one set"
    )
