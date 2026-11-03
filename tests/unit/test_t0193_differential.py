"""T-0193 A-7 (negative): with no ``astra.turn.stop`` sent, a session is byte-identical to BASE (docs/tasks/T-0193.md
A-7, "Tests to write -> A -> test_t0193_differential.py").

**Method** (T-0181 A11's): a differential over ``BASE..TIP``. ``BASE`` is main at 1076bc57 (``T-0193: group A0 merged;
notes for group A``, where group A branched; a literal), run in a subprocess over ``git archive BASE`` (T-0114's
``_base_tree``, origin checked; ``tests`` from this checkout, so the same rig drives both); ``TIP`` is the working tree,
in-process. The rig is ``_t0181_rig.mic_rig`` (read-only reuse: the session over the fakes, ``FakeAudioIO`` capture, a
reactive ``FakeStt`` behind a recorder, a socketless ``HudServer``); no stop is ever sent in a regime.

**Regimes** (A-7's three, the ``_t0181_rig`` shapes, each one session): ``voice`` (a spoken question answered, then a
second page greets), ``typed`` (an ``astra.text`` question answered, then a second page greets) and ``live_loop``
(``--tools live`` over the fake live browser, loop on: step 1 speaks and clicks, step 2 answers; a second page greets).

**Byte-identical sections** (A-7's list, nothing excepted): every frame fed to STT, every prompt and request body,
the machine's events, states and commands, every published message, every page's wire output (hello replies
included), history, ``turns.jsonl``, every session-log record, the TTS requests and the memory store. Normalised only
for what differs per run (session id, tmp dir, log path, ``git_commit``, the log writer's pid and byte count, the log
state's byte and record counts).

**Anti-vacuity.** A separate probe session types a question, waits until it speaks, sends a raw stop for its turn and
reads whether a ``UserStop`` was stepped and history marked ``[interrupted]``: on the working tree it must be, at BASE
(where the message is ``Unknown``) it must not, so the rows are red until group A lands.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, ``TurnPolicy(thinking=True)`` (tools off) or the
loop policy (live), 200 ms per word, ``FakeAudioIO`` 80 ms STT frames, n = 1 per regime, deterministic. No measurement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0181_differential as d81
from tests.unit._t0181_rig import LONG_ANSWER, QUESTION, mic_rig

#: A-7's BASE: main at 1076bc57, where T-0193 group A branched. A literal.
BASE_SHA = "1076bc570efa46e0c6843724a8364926ddbfa33b"
REGIMES = ("voice", "typed", "live_loop")
SECTIONS = (*d81.SECTIONS, "mic_positions", "mic_rows")


async def _probe(tmp: Path) -> dict[str, Any]:
    """The anti-vacuity probe: a typed ``LONG_ANSWER`` speaking for 400 ms, then a raw ``astra.turn.stop`` for its turn
    (``Unknown`` at BASE) and one second. Did a ``UserStop`` step, and did history end ``[interrupted]``?"""
    from tests.unit._tool_turn_support import answer

    tmp.mkdir(parents=True, exist_ok=True)
    async with mic_rig(tmp, script=[answer(LONG_ANSWER)]) as rig:
        await rig.advance(400, step=40)
        await rig.type_question(QUESTION)
        await rig.until(
            lambda: rig.state.state is TurnState.SPEAKING, "probe: never spoke", limit_ms=6_000
        )
        await rig.advance(400, step=40)
        rig.page.receive(json.dumps({"type": "astra.turn.stop", "turn_id": rig.state.turn_id}))
        await rig.pump()
        await rig.advance(1_000, step=40)
        stepped = len(rig.stepped("user.stop"))
        last = rig.state.history[-1].assistant_spoken if rig.state.history else None
    return {"stepped": stepped, "marked": bool(last and last.endswith("[interrupted]"))}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every regime (T-0181's drivers, which send no stop) and the probe. Runs in-process on the working tree and in
    a subprocess on the BASE tree."""
    return {
        "voice": await d81._voice(tmp / "voice"),
        "typed": await d81._typed(tmp / "typed"),
        "live_loop": await d81._live_loop(tmp / "live_loop"),
        "probe": await _probe(tmp / "probe"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0193_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0193_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import asyncio

    tmp = tmp_path_factory.mktemp("t0193_now")
    return json.loads(json.dumps(asyncio.run(_capture(tmp / "run"))))


#: T-0199 criterion 11 (census bullet, 2026-10-29): every voice-onset frame lookup writes exactly one note with these
#: keys. The only log record this comparison drops; every other record stays byte-compared with BASE.
ONSET_CLOCK_KEYS = {"what", "onset_ms", "at_session_ms", "method"}


def _without_onset_clock(records: list[Any]) -> list[Any]:
    """The log section minus exactly the ``note`` records with ``what == "onset.clock"`` (each asserted to have
    exactly criterion 11's keys). ``log.stopped``'s ``records`` counts every record written, so it is reduced by
    exactly the number dropped; every other field and record is compared as written. A no-op on BASE."""
    kept: list[Any] = []
    dropped = 0
    for r in records:
        data = r.get("data") if isinstance(r, dict) else None
        if r.get("kind") == "note" and isinstance(data, dict) and data.get("what") == "onset.clock":
            assert set(data) == ONSET_CLOCK_KEYS, (
                f"T-0199 criterion 11: an onset.clock note has exactly {sorted(ONSET_CLOCK_KEYS)}; got {sorted(data)}"
            )
            dropped += 1
            continue
        kept.append(r)
    if dropped:
        kept = [
            {**r, "data": {**r["data"], "records": r["data"]["records"] - dropped}}
            if r.get("kind") == "log.stopped"
            else r
            for r in kept
        ]
    return kept


def _same(expected: Any, got: Any, what: str) -> None:
    if not isinstance(expected, list) or not isinstance(got, list):
        expected, got = [expected], [got]
    first = next((i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None)
    assert got == expected, (
        f"T-0193 A-7 {what} differs from {BASE_SHA[:8]} (lengths {len(expected)} vs {len(got)}; first difference "
        f"at {first}: base {json.dumps(expected[first] if first is not None else None)[:800]} / "
        f"now {json.dumps(got[first] if first is not None else None)[:800]})"
    )


def test_the_tree_under_comparison_has_the_stop(base: dict[str, Any], now: dict[str, Any]) -> None:
    """Anti-vacuity for every row below: in the probe a stop is stepped and marks history on the working tree, and
    not at BASE, where the message is unknown."""
    assert base["probe"] == {"stepped": 0, "marked": False}, (
        f"premise: BASE {BASE_SHA[:8]} has no astra.turn.stop; {base['probe']}"
    )
    assert now["probe"] == {"stepped": 1, "marked": True}, (
        "T-0193 A-7 anti-vacuity: the working tree must implement the stop (A-2 routes it, the machine marks history) "
        f"so the 'no stop, no change' rows compare a tree that has it; probe {now['probe']}"
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_a7_with_no_stop_sent_nothing_differs(
    base: dict[str, Any], now: dict[str, Any], regime: str
) -> None:
    """A-7: in each regime every section equals BASE's, hello replies included (nothing is ignored but T-0213's one
    hello settings state, R-5 named re-pin). The premises say
    each regime ran its turn; no ``user.stop`` is stepped. The probe row says the tree has the stop."""
    b, n = base[regime], now[regime]
    assert len(n["wire"]) == 2, f"premise: two pages greeted; {len(n['wire'])}"
    assert len(n["records"]) == 1, f"premise: one turn line; {len(n['records'])}"
    if regime == "live_loop":
        assert len(n["executed"]) == 1, f"premise: the live click ran; {n['executed']}"
    stepped = [s["event"]["type"] for s in n["machine"]]
    assert "user.stop" not in stepped, f"A-7 ({regime}): no UserStop is stepped"
    for section in SECTIONS:
        if section == "log":
            _same(
                _without_onset_clock(b[section]),
                _without_onset_clock(n[section]),
                f"{regime} 'log'",
            )
            continue
        if section == "wire":
            # T-0213 R-5 named re-pin: the one settings state per hello, asserted last (T-0181's helper), set aside
            _same(
                b[section],
                d81._wire_without_settings_state(regime, n[section]),
                f"{regime} {section!r}",
            )
            continue
        _same(b[section], n[section], f"{regime} {section!r}")
    if regime == "live_loop":
        _same(b["executed"], n["executed"], "live_loop executed calls")
    assert now["probe"]["stepped"] == 1, (
        f"T-0193 A-7 anti-vacuity: the working tree routes a stop; probe {now['probe']}"
    )
