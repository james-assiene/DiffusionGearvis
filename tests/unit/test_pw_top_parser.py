"""T-0035 ruling A14 — ``pw-top``'s follower marker is on the NAME column, not the ID column.

``scripts/demo/audio_smoke.py`` prints the *live* graph quantum (ruling A13: ``pw-dump``'s
``node.latency`` is only the request we made) by reading ``pw-top -b -n 2``'s QUANT column for
our stream node and for the driver of its graph.  pw-top 1.0.5 prints one row per node as::

    %s %4.1u %6.1u %6.1u %s %s %s %s  %3.1u %16.16s %s%s
    S   ID  QUANT   RATE    WAIT    BUSY   W/Q   B/Q  ERR FORMAT           NAME

the trailing ``%s%s`` being *marker* + *name*, with the marker ``" + "`` for a node that
follows a driver and ``""`` for the driver itself.  A parser that looks for the marker in the
ID column labels **every** node a driver, and the report then says "driver 480 frames" about
our own stream while never naming the driver's quantum — the one number the line exists to
carry.

Pinned here (behaviour of the script's public helpers, not their internals):

* ``pw_top_rows(text) -> [{"id": int, "quant": str, "rate": str, "name": str, "driver": bool}]``
  — one dict per node row of the **last** batch iteration;
* ``pw_top_quantum(ours: set[int]) -> str`` — a sentence naming our node's quantum and the
  driver's, in frames and in ms, or "not readable (…)" when no row carries a quantum.

Fixtures: ``tests/fixtures/pipewire/pw_top_idle.txt`` is a real ``pw-top -b -n 2`` of this box
(read-only, 2026-09-14, pw 1.0.5 — every node idle, QUANT 0, because ``auto_null`` is the only
sink and nothing was playing).  ``pw_top_active.txt`` is *constructed* from it by
``make_pw_top_fixture.py`` at pw-top's own column positions, since a follower row cannot be
recorded on a box with no running graph.  Both are described in
``tests/fixtures/pipewire/MANIFEST.md``.

Regime: no PortAudio, no PipeWire graph touched by these rows — pure text in, strings out.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "demo" / "audio_smoke.py"
FIXTURES = ROOT / "tests" / "fixtures" / "pipewire"
IDLE = FIXTURES / "pw_top_idle.txt"
ACTIVE = FIXTURES / "pw_top_active.txt"

# Read off tests/fixtures/pipewire/pw_top_active.txt by hand (last block only):
#   S   52   1024  48000 …    S16LE 2 48000 alsa_input.usb-046d_Logitech_Webcam_C925e_…
#   R  137    480  48000 …    F32LE 1 48000  + astra-conductor
WEBCAM = "alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo"
OURS_ID = 137  # the Stream/Output/Audio node of dump_rig_ok.json
DRIVER_ID = 52
OTHER_FOLLOWER_ID = 145  # another client's stream: a follower, and not ours
LAST_BLOCK_IDS = [29, 30, 45, 52, 137, 145, 123]
FIRST_BLOCK_ONLY_ID = 56  # the v4l2 node: present in the first block, absent from the last


def load_script() -> ModuleType:
    """Import the script by path, as the other smoke rows do."""
    if not SCRIPT.exists():
        raise AssertionError("scripts/demo/audio_smoke.py does not exist (T-0035 criterion 8)")
    spec = importlib.util.spec_from_file_location("astra_demo_audio_smoke_pwtop", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("scripts/demo/audio_smoke.py is not importable by path: no loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_of(text: str) -> list[dict[str, Any]]:
    module = load_script()
    if not hasattr(module, "pw_top_rows"):
        raise AssertionError("scripts/demo/audio_smoke.py has no pw_top_rows(text) (ruling A13)")
    result = module.pw_top_rows(text)
    assert isinstance(result, list), f"pw_top_rows returned {type(result).__name__}, not a list"
    return list(result)


def quantum_of(text: str, ours: set[int], monkeypatch: pytest.MonkeyPatch) -> str:
    """``pw_top_quantum(ours)`` with ``pw-top``'s output replaced by a fixture.

    Only the ``pw-top`` invocation is intercepted; anything else the script shells out to runs
    for real, so this does not hide a second subprocess behind the fake.
    """
    module = load_script()
    if not hasattr(module, "pw_top_quantum"):
        raise AssertionError("scripts/demo/audio_smoke.py has no pw_top_quantum(ours) (A13)")
    real_run = subprocess.run

    def fake_run(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "pw-top":
            return subprocess.CompletedProcess(list(argv), 0, text, "")
        return real_run(argv, *args, **kwargs)  # type: ignore[no-any-return,call-overload]

    monkeypatch.setattr(subprocess, "run", fake_run)
    said = module.pw_top_quantum(ours)
    assert isinstance(said, str), f"pw_top_quantum returned {type(said).__name__}, not a str"
    return said


def test_the_follower_row_is_not_a_driver() -> None:
    """Ruling A14, the headline: the ``+`` sits before NAME, so node 137 is a **follower**.

    The two rows are read off the fixture by hand (see this module's header).  A parser that
    takes the marker from the ID column reports ``driver=True`` here — which is exactly the
    defect A14 names, and is what today's code does.
    """
    rows = {row["id"]: row for row in rows_of(ACTIVE.read_text())}
    assert set(rows) == set(LAST_BLOCK_IDS), (
        f"pw_top_rows saw ids {sorted(rows)}, expected {sorted(LAST_BLOCK_IDS)} — the node rows "
        f"of the last batch iteration of tests/fixtures/pipewire/pw_top_active.txt"
    )
    follower = rows[OURS_ID]
    assert follower["driver"] is False, (
        f"row {follower!r} for node {OURS_ID} is labelled driver, but pw-top printed it as "
        f'" + astra-conductor": the marker precedes NAME in pw-top 1.0.5\'s row format '
        f"(%s %4.1u %6.1u %6.1u %s %s %s %s  %3.1u %16.16s %s%s), not in the ID column. "
        f"Ruling A14."
    )
    assert rows[OTHER_FOLLOWER_ID]["driver"] is False, (
        f'row {rows[OTHER_FOLLOWER_ID]!r} carries the same " + " marker before its name'
    )
    assert (follower["quant"], follower["rate"]) == ("480", "48000"), follower
    assert follower["name"] == "astra-conductor", (
        f"the follower's name came out {follower['name']!r}: the marker is not part of the name"
    )


def test_the_driver_row_is_a_driver() -> None:
    """The other half of A14: fixing the follower must not relabel the driver.

    Node 52 (the C925e) carries no marker, QUANT 1024 at RATE 48000.
    """
    rows = {row["id"]: row for row in rows_of(ACTIVE.read_text())}
    driver = rows[DRIVER_ID]
    assert driver["driver"] is True, (
        f"row {driver!r} for node {DRIVER_ID} has no follower marker in the fixture and must be "
        f"the driver; a parser that calls everything a follower is as wrong as A14's defect"
    )
    assert (driver["quant"], driver["rate"]) == ("1024", "48000"), driver
    assert driver["name"] == WEBCAM, driver


def test_only_the_last_block_is_reported() -> None:
    """``pw-top -b -n 2``: the first iteration is an unstarted counter, the second is the read.

    The fixture's first block contains node 56 (the v4l2 source) and the last does not, so a
    parser that accumulates both — or keeps the first — is visible here rather than silently
    reporting a stale quantum.
    """
    rows = rows_of(ACTIVE.read_text())
    ids = [row["id"] for row in rows]
    assert ids == LAST_BLOCK_IDS, (
        f"pw_top_rows returned ids {ids}; the last block of the fixture is {LAST_BLOCK_IDS} "
        f"(node {FIRST_BLOCK_ONLY_ID} appears only in the first block)"
    )
    assert FIRST_BLOCK_ONLY_ID not in ids


def test_the_recorded_idle_fixture_is_every_node_idle() -> None:
    """The recording itself: six nodes, QUANT 0, no follower — this box with no graph running."""
    rows = rows_of(IDLE.read_text())
    assert [row["id"] for row in rows] == [29, 30, 45, 56, 52, 123], rows
    assert {row["quant"] for row in rows} == {"0"}, (
        f"the recorded idle fixture is meant to have QUANT 0 everywhere; got {rows!r}"
    )
    assert all(row["driver"] for row in rows), (
        "no row in the recorded fixture carries a follower marker, so none of them is a follower"
    )


def test_an_idle_graph_is_reported_as_not_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial: an all-idle graph must not be reported as a quantum of 0 frames.

    Every node idle means the graph is not running; printing "driver 0 frames (0.0 ms)" would
    put a number in a T-0023 regime block that is not a measurement at all (CLAUDE.md: a number
    without its regime cannot be compared).  ``pw_top_quantum`` says so instead.
    """
    said = quantum_of(IDLE.read_text(), {OURS_ID}, monkeypatch)
    assert "not readable" in said.lower(), f"pw_top_quantum said {said!r} for an all-idle graph"
    assert "idle" in said.lower(), (
        f"pw_top_quantum said {said!r}: the reason (every node idle, QUANT 0) is what tells the "
        f"reader to start the graph rather than to install pw-top"
    )
    assert "480" not in said and "1024" not in said, said


def test_the_quantum_line_names_ours_and_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ruling A13/A14 together: our node's quantum *and* the driver's, in frames and in ms.

    Hand-computed from the fixture: ours 480 frames at 48000 Hz = 1000·480/48000 = **10.0 ms**;
    the driver 1024 frames at 48000 Hz = 1000·1024/48000 = 21.333… = **21.3 ms** at one
    decimal.  Before A14 both rows are "driver", so our own node is announced as the driver and
    the real driver's 21.3 ms never appears under its own name.
    """
    said = quantum_of(ACTIVE.read_text(), {OURS_ID}, monkeypatch)
    lowered = said.lower()
    assert "ours 480 frames (10.0 ms)" in lowered, (
        f"pw_top_quantum said {said!r}: node {OURS_ID} is ours and pw-top gives it 480 frames "
        f"at 48000 Hz = 10.0 ms"
    )
    assert "driver 1024 frames (21.3 ms)" in lowered, (
        f"pw_top_quantum said {said!r}: node {DRIVER_ID} is the driver of the graph in the "
        f"fixture (no follower marker) and runs 1024 frames at 48000 Hz = 21.3 ms"
    )
    assert "not readable" not in lowered, said
    assert "2048" not in lowered, (
        f"pw_top_quantum said {said!r}: node {OTHER_FOLLOWER_ID} is another client's stream — a "
        f"follower, and not ours — so its 2048 frames belong to nobody in this report. A parser "
        f"that reads the marker from the ID column calls it a driver and prints it (A14)."
    )


def test_a_row_of_ours_is_never_labelled_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial, A14: our stream node must not be presented as the graph's driver.

    This is the user-visible shape of the bug: the report claimed "driver 480 frames" — the
    quantum we asked for, dressed as the quantum the graph runs at, which is precisely the
    confusion ruling A13 introduced the pw-top line to end.
    """
    said = quantum_of(ACTIVE.read_text(), {OURS_ID}, monkeypatch)
    assert "driver 480" not in said.lower(), (
        f"pw_top_quantum said {said!r}: 480 frames is *our* stream node (137, a follower); the "
        f"driver in the fixture is node 52 at 1024 frames"
    )


def test_pw_top_rows_survives_junk() -> None:
    """Neither an empty text nor pw-top's own truncated output may raise (ruling A8)."""
    assert rows_of("") == []
    assert rows_of("S   ID  QUANT   RATE\n") == []
    assert rows_of("garbage\n\n   \n") == []


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
