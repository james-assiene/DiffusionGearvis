"""The reported quantum says where it came from: T-0035 criterion 3 as ruled in A13.

``quantum_ms`` used to be documented as "graph quantum actually negotiated". The confirmation
review (ruling A13) established that this cannot be measured from ``pw-dump`` at all:
``node.latency`` is the per-node **request** — the ``PIPEWIRE_LATENCY`` this process asked for,
echoed back verbatim by the graph, even outside the graph's own clamp — and ``pw-dump`` exposes no
live quantum anywhere. A number is therefore only honest here if it carries its provenance, and
the provenance ``"pw-dump"`` (which claimed a measurement) is **retired**.

Criterion 3 as ruling A13 states it, which is what every row below asserts:

* ``quantum_source: str`` is exactly one of ``"request"``, ``"callback"``, ``"unknown"``;
* ``"request"`` when a node was named: the value is the ``PIPEWIRE_LATENCY`` we asked for, read
  back from the graph, so it says the plugin honoured the request — not that the graph runs there;
* ``"callback"`` on the plain-default path: the frame count PortAudio handed the callback;
* ``"unknown"``/``0.0`` when neither — never the request without the label, and never 0.0 beside a
  label that names a provenance;
* the string ``"pw-dump"`` is not a value ``quantum_source`` can take.

The live graph quantum is not here at all: it is a group-B ``pw-top -b -n 2`` measurement printed
by the smoke script (``tests/unit/test_audio_smoke_script.py``, "graph quantum (pw-top)").

Seams this module pins, so the implementer builds to them (docs/tasks/T-0035.md, Implementation
notes): the module-level ``_pw_dump() -> list[dict]`` — the single function that shells out, and
therefore the single injection point for a recorded dump — and the pure parser
``parse_quantum(dump, pid) -> float | None``, which reads a *request*.

Regime: PortAudio absent (P1), ``sounddevice`` is ``tests/fakes/sounddevice_stub.py``; the graph
is the recorded JSON of ``tests/fixtures/pipewire/`` (libpipewire 1.0.5, 2026-09-14) with the
recorded pid rewritten to this process's; 48 kHz mono f32, requested block 480 frames = 10.0 ms.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import audio_device as audio_device_module
from local_astra.conductor.audio_device import PortAudioDevice, open_device
from tests.fakes.audio_device import attr_or_fail
from tests.fakes.pipewire_dumps import (
    SINK,
    count_pw_dump_calls,
    load_dump,
    patch_pw_dump,
    without_node_latency,
)
from tests.fakes.sounddevice_stub import installed, last_stream, wire_prefill

#: The three provenances ruling A13 allows, and no fourth.
SOURCES = ("request", "callback", "unknown")
#: The provenance the ruling retired: it claimed a measurement pw-dump cannot make.
RETIRED = "pw-dump"
#: 480 frames at 48 kHz — what the device asks for, what dump_rig_ok.json echoes back.
REQUEST_MS = 10.0


def source_of(device: PortAudioDevice) -> str:
    return str(
        attr_or_fail(
            device,
            "quantum_source",
            'T-0035 criterion 3 / ruling A13: "request" / "callback" / "unknown" beside the number',
        )
    )


def quantum_source_literals() -> set[str]:
    """Every string ``audio_device.py`` can assign to ``quantum_source``, read off its AST.

    An independent oracle for "the enum has exactly three values": the runtime rows below can
    only show the three they reach, so this one reads the module's own source and refuses any
    other literal — in particular the retired ``"pw-dump"``. Names are resolved through
    module-level constants, so ``_SOURCE_REQUEST = "request"`` is allowed; anything else assigned
    to ``quantum_source`` is unverifiable and says so.
    """
    path = Path(str(audio_device_module.__file__))
    tree = ast.parse(path.read_text(), filename=str(path))
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value

    def targets_quantum_source(target: ast.expr) -> bool:
        return isinstance(target, ast.Attribute) and target.attr == "quantum_source"

    def literal(value: ast.expr) -> str:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Name) and value.id in constants:
            return constants[value.id]
        raise AssertionError(
            f"{path.name} line {getattr(value, 'lineno', '?')}: quantum_source is assigned "
            f"something this row cannot read ({ast.dump(value)[:80]}…). Ruling A13 makes the "
            f"provenance a closed set of three literals ({', '.join(SOURCES)}); please assign "
            f"one of them, or a module-level constant holding one"
        )

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and targets_quantum_source(node.target):
            if node.value is not None:
                found.add(literal(node.value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if targets_quantum_source(target):
                    found.add(literal(node.value))
                elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for slot, value in zip(target.elts, node.value.elts, strict=False):
                        if targets_quantum_source(slot):
                            found.add(literal(value))
    if not found:
        raise AssertionError(
            f"nothing in {path.name} assigns a string to self.quantum_source: criterion 3 asks "
            f"for the provenance beside the number"
        )
    return found


def test_a_named_node_reports_the_request_and_says_so(monkeypatch: Any) -> None:
    """Ruling A13: a node was named, the graph echoed our request back, the label says so.

    ``dump_rig_ok.json`` carries ``node.latency = "480/48000"`` on this process's stream node —
    which is the ``PIPEWIRE_LATENCY`` the device set when it opened, not a measurement of the
    graph (A13: ``pw-dump`` exposes no live quantum). So the number is 10.0 ms, numerically the
    request, and the only thing that makes it honest is the label: ``quantum_source ==
    "request"``. ``"pw-dump"`` is the retired provenance — it asserted that the value came from
    the graph's negotiation, which is the claim A13 found to be false — and it must never appear.
    """
    patch_pw_dump(monkeypatch, "dump_rig_ok")
    with installed() as stub:
        device = open_device(sink=SINK, block_ms=10)
        device.start()
        wire_prefill(last_stream(stub), device).drive(3, frames=480, taken=480)
        try:
            assert source_of(device) != RETIRED, (
                f'quantum_source is "{RETIRED}": ruling A13 retired that value — node.latency is '
                f"the PIPEWIRE_LATENCY we asked for, echoed back, so calling it a pw-dump "
                f"measurement fabricates a negotiation that never happened"
            )
            assert source_of(device) == "request", (
                f"quantum_source is {source_of(device)!r} on a graph that carries our own "
                f'node.latency: ruling A13 names that provenance "request"'
            )
            assert device.quantum_ms == pytest.approx(REQUEST_MS, abs=1e-9), (
                f"quantum_ms is {device.quantum_ms}; dump_rig_ok.json echoes back 480/48000 s = "
                f"{REQUEST_MS} ms, which is what a request-sourced number reports"
            )
        finally:
            device.stop()


def test_an_unreadable_graph_reports_unknown_not_the_request(monkeypatch: Any) -> None:
    """Criterion 3: no value in the graph ⇒ ``0.0`` and ``"unknown"``, never 10.0.

    The graph answers, and the links verify, but no node in it carries ``node.latency`` (the
    recorded rig dump with that property removed — a real possibility: on this box every node's
    ``params.ProcessLatency`` is empty, see the fixtures' MANIFEST). "The graph would not say"
    is an honest, non-empty answer; reporting the request *without* the graph having echoed it
    back would be a number with no source at all.
    """
    patch_pw_dump(monkeypatch, without_node_latency(load_dump("dump_rig_ok", pid=os.getpid())))
    with installed() as stub:
        device = open_device(sink=SINK, block_ms=10)
        device.start()
        wire_prefill(last_stream(stub), device).drive(3, frames=480, taken=480)
        try:
            assert source_of(device) == "unknown", (
                f"quantum_source is {source_of(device)!r} for a graph that carries no quantum at "
                f"all (criterion 3)"
            )
            assert device.quantum_ms == 0.0, (
                f"quantum_ms is {device.quantum_ms}: an unknown quantum is 0.0, and in "
                f"particular it is not the {REQUEST_MS} ms that was requested"
            )
        finally:
            device.stop()


def test_quantum_source_is_reported_beside_the_number(monkeypatch: Any) -> None:
    """Ruling A13: the enum is exactly three values, and value and label never disagree.

    Two devices, one assertion each way: on ``dump_rig_ok.json`` the graph echoes our
    ``480/48000`` request (10.0 ms, read off the fixture by hand — see its MANIFEST) and the
    source is ``"request"``; with the same dump stripped of ``node.latency`` the source is
    ``"unknown"`` and the value is 0.0. A source that names a provenance must come with a
    positive number, and ``"unknown"`` must come with 0.0 — the two can never disagree, which is
    what makes ``quantum_ms == 0.0`` readable in a T-0023 regime block.

    The closed set is asserted twice over: at runtime, and against the module's own source, where
    the retired ``"pw-dump"`` must not appear as a value ``quantum_source`` can take (it may of
    course still appear as the *command* the module shells out to, which is a different thing).
    """
    readable = load_dump("dump_rig_ok", pid=os.getpid())
    for dump, expected_source, expected_ms in (
        (readable, "request", REQUEST_MS),
        (without_node_latency(readable), "unknown", 0.0),
    ):
        patch_pw_dump(monkeypatch, dump)
        with installed() as stub:
            device = open_device(sink=SINK, block_ms=10)
            device.start()
            wire_prefill(last_stream(stub), device).drive(1, frames=480, taken=480)
            try:
                assert source_of(device) in SOURCES, (
                    f"quantum_source is {source_of(device)!r}; ruling A13 names exactly three "
                    f"values: {SOURCES}"
                )
                assert source_of(device) == expected_source
                assert device.quantum_ms == pytest.approx(expected_ms, abs=0.01)
                assert (source_of(device) == "unknown") == (device.quantum_ms == 0.0), (
                    f"quantum_source={source_of(device)!r} beside quantum_ms={device.quantum_ms}: "
                    f'"unknown" means 0.0 and 0.0 means "unknown" (criterion 3)'
                )
            finally:
                device.stop()
    literals = quantum_source_literals()
    assert RETIRED not in literals, (
        f'audio_device.py can still set quantum_source to "{RETIRED}" (literals it assigns: '
        f"{sorted(literals)}). Ruling A13 retired that provenance: node.latency is the request "
        f'echoed back, so the value is "request"'
    )
    assert literals <= set(SOURCES), (
        f"audio_device.py assigns quantum_source the literals {sorted(literals)}; ruling A13 "
        f"closes the set at {SOURCES}"
    )


def test_the_default_path_reports_the_callback_frame_count(monkeypatch: Any) -> None:
    """Ruling A13's third enum value, per A1 and A10(a): no node ⇒ ``"callback"``.

    Nobody named a node, so nothing reads ``pw-dump`` (criterion 4, pinned again here by the
    counter: the plain-default path costs no subprocess). The only thing the device knows about
    the quantum is the frame count PortAudio handed it, and the provenance must say exactly
    that: ``quantum_source == "callback"`` beside ``quantum_ms == 10.0`` for a 480-frame
    callback at 48 kHz — which is *numerically* the request, and is honest only because the
    label says where it came from. The two values are one statement: a callback that reported
    ``"request"`` (the graph was never read, so nothing echoed anything back) or ``0.0`` (the
    "unknown" value, while a frame count is in hand) would be the same fabrication criterion 3
    forbids in the other direction.
    """
    calls = count_pw_dump_calls(monkeypatch)
    with installed() as stub:
        device = open_device(block_ms=10)
        device.start()
        wire_prefill(last_stream(stub), device).drive(1, frames=480, taken=480)
        try:
            assert source_of(device) == "callback", (
                f"quantum_source is {source_of(device)!r} after one 480-frame callback with no "
                f'node requested; ruling A1 names this path\'s provenance "callback" — the '
                f'graph was never read, so it cannot be "request", and a frame count is not '
                f'"unknown"'
            )
            assert device.quantum_ms == pytest.approx(REQUEST_MS, abs=1e-9), (
                f"quantum_ms is {device.quantum_ms} after a 480-frame callback at "
                f"{device.sample_rate} Hz; the callback's own frame count is "
                f'{REQUEST_MS} ms (480/48000 s) and that is what "callback" means'
            )
        finally:
            device.stop()
    assert calls == [], (
        f"_pw_dump was called {len(calls)} time(s) with no node requested: ruling A1 — no node "
        f'means no graph read, which is why this path reports "callback" at all'
    )
