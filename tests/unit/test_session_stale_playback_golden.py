"""T-0092a criterion 6's golden: a session whose every run ends inside its own turn is unchanged.

The golden was captured at the literal base SHA **0304f3fa0afc2f5a03aacb0ff6d91e75464b2a31**
(``main``, 2026-10-12, the commit branch ``T-0092a-stale-drains`` was cut from), by
``tests/fixtures/session/capture_t0092a_golden.py``, which refuses to write unless
``local_astra/conductor/session.py`` is byte-identical to that commit's blob. The fixture records
the SHA and the blob it was captured against; it is never compared against a moving ref
(``.claude/rules/tests.md``, "A scope or contract pin compares against a fixed commit").

The scripted session (criterion 6, verbatim): a spoken turn, a typed turn, a UI action, an A2UI
surface, and a barge-in with a timely ``playback.stopped`` — then one more spoken turn, so the
barge-in's relisten turn also ends. In every one of them the run's terminal is handled while its own
turn is current. What must be identical to the base SHA, section by section:

* ``machine``  — every event the machine was stepped with, the state it returned and its commands;
* ``hud``      — every HUD message the session published;
* ``records``  — every ``turns.jsonl`` line;
* ``requests`` — every LLM request body, as the real ``VllmClient.request_body`` builds it from
  what the fake backend was handed (closes by delegation: a field the client adds is in it).

Normalised before comparing, and only these: the per-process ``session_id`` (random), the
``Regime.git_commit`` (moves with every commit) and nothing else. Times are the ``ScriptedClock``'s.
T-0180 B-R3 (2026-10-28): the ``hud`` section leaves out messages whose ``type`` is exactly ``astra.thought`` (the
base published none); every other type compares as before, and ``test_t0180_differential.py`` (B8) pins the thoughts.

This module is green on 0304f3f by construction (it *is* the base). It is the "too wide" guard of
criterion 6: a session that drops, re-orders or rewrites a same-turn terminal, or flushes a run
that ended in its own turn, changes the ``machine`` section here. Regime: CPU over fakes,
``ScriptedClock``, thinking on, 200 ms/word, n = 1, deterministic.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts import events as E
from local_astra.contracts.turn import MachineState, TurnState
from tests.fakes import FakeMemory, FakeStt
from tests.unit import _pre_t0155_persona as P
from tests.unit._hud_rig import Recorder, ui
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit.test_session_loop import (
    POLICY,
    RecordingAudio,
    RecordingLlm,
    RecordingMachine,
    RecordingPerception,
    RecordingTts,
    Rig,
    ScriptedClock,
    Session,
    SessionConfig,
    answer,
    frame,
    require_session,
    thought,
)
from tests.unit.test_turn_invariants import state_without_t0112

REPO = Path(__file__).resolve().parents[2]

#: The base commit of T-0092a (literal; never ``main`` or a merge base).
BASE_SHA = "0304f3fa0afc2f5a03aacb0ff6d91e75464b2a31"
#: ``git rev-parse 0304f3f:local_astra/conductor/session.py`` — the session the golden ran.
BASE_SESSION_BLOB = "837acd3287c642d057087c8808c3d1a7102a4094"
GOLDEN = REPO / "tests" / "fixtures" / "session" / "t0092a_golden.json"
#: T-0108 criterion 10's two additive turn-record extras, discounted under that task's name.
T0108_EXTRAS = frozenset({"ui_valid_final", "ui_shown"})
#: T-0127 criterion 5's additive turn-record extra, stripped beside them (re-scope R3).
T0127_EXTRAS = frozenset({"ui_complete_final"})

MS_PER_WORD = 200
SECTIONS = ("machine", "hud", "records", "requests")

SPOKEN = [
    thought("the screen shows a stack trace"),
    answer("It is a stack trace. "),
    answer("The error is on line four."),
]
TYPED = [answer("A typed answer here. ")]
CLICKED = [answer("You pressed next. ")]
SURFACE = [answer("Here is one half. "), ui()]
BARGED = [answer("This answer is long enough to still be playing at the cut. ")]
AFTER = [answer("After the barge in. ")]


class StateMachine(RecordingMachine):
    """The production machine; logs the full returned state as well (``RecordingMachine`` keeps
    only its ``TurnState``)."""

    def __init__(self) -> None:
        super().__init__(POLICY)
        self.full: list[tuple[Any, MachineState, tuple[Any, ...]]] = []

    def step(self, state: MachineState, event: E.Event) -> Any:
        new, cmds = super().step(state, event)
        self.full.append((event, new, tuple(cmds)))
        return new, cmds


def _refusing() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the golden built a body and then tried to send it: {request.url}")

    return httpx.MockTransport(handler)


@asynccontextmanager
async def golden_rig(tmp_path: Path) -> AsyncIterator[Rig]:
    require_session()
    clock = ScriptedClock()
    machine = StateMachine()
    llm = RecordingLlm(SPOKEN)
    llm.clock = clock
    tts = RecordingTts(ms_per_word=MS_PER_WORD)
    audio = RecordingAudio(clock)
    published = Recorder()
    turns_path = tmp_path / "turns.jsonl"
    session = Session(
        machine,
        llm,
        FakeStt(),
        tts,
        audio,
        RecordingPerception([frame(0, "a")]),
        FakeMemory(),
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns_path),
        publish=published,
    )
    rig = Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=session.stt,
        tts=tts,
        audio=audio,
        perception=session.perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
    )
    rig.published = published  # type: ignore[attr-defined]
    await session.start()
    try:
        yield rig
    finally:
        await session.aclose()


async def run_scenario(rig: Rig) -> None:
    """Criterion 6's session. Every run's terminal is handled inside its own turn."""
    # 1. a spoken turn
    rig.llm.script = SPOKEN
    await rig.ask("what is on this screen")
    await rig.speak_out()
    await rig.advance(300)
    # 2. a typed turn
    rig.llm.script = TYPED
    rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="and what does it mean"))
    await rig.speak_out()
    await rig.advance(300)
    # 3. a UI action
    rig.llm.script = CLICKED
    rig.submit(
        E.UiAction(t_ms=rig.clock.now_ms, surface_id="lesson_1", name="next", context={"step": 2})
    )
    await rig.speak_out()
    await rig.advance(300)
    # 4. an A2UI surface
    rig.llm.script = SURFACE
    await rig.ask("can you illustrate one half")
    await rig.speak_out()
    await rig.advance(300)
    # 5. a barge-in with a timely playback.stopped, then the relisten turn runs to its end
    rig.llm.script = BARGED
    await rig.ask("tell me more")
    await rig.until(
        lambda: (
            rig.state.state is TurnState.SPEAKING
            and any(source for _, source in rig.audio.plays if source and "long enough" in source)
        ),
        "the barge-in turn never reached the device",
        limit_ms=10_000,
    )
    await rig.advance(600, step=100)
    now = rig.clock.now_ms
    rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
    await rig.until(lambda: rig.state.state is TurnState.LISTENING, "the takeover never relistened")
    rig.llm.script = AFTER
    await rig.words("and then")
    await rig.end_of_turn()
    await rig.speak_out()


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


#: T-0180 B-R3: the one HUD message type the ``hud`` section drops, compared by exact equality of ``type`` (never a
#: prefix). Every other type compares as before; ``test_t0180_differential.py`` (B8) pins the thoughts themselves.
HUD_IGNORED_TYPE = "astra.thought"


def hud_section(messages: list[Any]) -> list[Any]:
    """T-0180 B-R3: every published message, dumped, but for those whose ``type`` is exactly ``HUD_IGNORED_TYPE``."""
    dumped = [_dump(m) for m in messages]
    return [m for m in dumped if not (isinstance(m, dict) and m.get("type") == HUD_IGNORED_TYPE)]


def capture_sections(rig: Rig) -> dict[str, Any]:
    machine = rig.machine
    assert isinstance(machine, StateMachine)
    client = VllmClient(transport=_refusing())
    requests = [
        client.request_body(prompt, request_id, options, xargs)
        for (request_id, prompt, options), xargs in zip(
            rig.llm.generates, rig.llm.xargs_log, strict=True
        )
    ]
    return {
        "machine": [
            {
                "event": _dump(ev),
                "state": state_without_t0112(new),
                "commands": [_dump(c) for c in cmds],
            }
            for ev, new, cmds in machine.full
        ],
        "hud": hud_section(rig.published.messages),  # type: ignore[attr-defined]
        "records": [json.loads(line) for line in rig.lines()],
        "requests": requests,
    }


def normalise(tree: Any, session_id: str) -> Any:
    """Replace the random session id everywhere and the regime's git commit; nothing else."""
    text = json.dumps(tree, sort_keys=True).replace(session_id, "<session>")
    out = json.loads(text)
    for record in out.get("records", []):
        regime = record.get("regime")
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def capture(tmp_path: Path) -> dict[str, Any]:
    async with golden_rig(tmp_path) as rig:
        await run_scenario(rig)
        await rig.pump()
        sections = capture_sections(rig)
        session_id = rig.session.session_id
        failures = list(rig.session.failures)
    assert not failures, f"premise: the scripted session ran without failures: {failures}"
    return normalise(sections, session_id)


def _first_difference(expected: list[Any], got: list[Any]) -> str:
    for i, (a, b) in enumerate(zip(expected, got, strict=False)):
        if a != b:
            return (
                f"first difference at index {i}:\n  base 0304f3f: "
                f"{json.dumps(a, sort_keys=True)[:1500]}\n  now:          "
                f"{json.dumps(b, sort_keys=True)[:1500]}"
            )
    return f"lengths differ: base 0304f3f has {len(expected)}, now {len(got)}"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    assert GOLDEN.is_file(), f"the golden fixture {GOLDEN} is missing (see tests/fixtures/session/)"
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["base_sha"] == BASE_SHA and data["session_py_blob"] == BASE_SESSION_BLOB, (
        "the golden must be the one captured at the literal base SHA, against its session.py; "
        f"the fixture says {data['base_sha']} / {data['session_py_blob']}"
    )
    return data


def test_the_golden_was_captured_against_the_base_sessions_bytes():
    """Provenance: the blob the fixture names is 0304f3f's ``session.py`` (skipped without it)."""
    try:
        blob = subprocess.run(
            ["git", "rev-parse", f"{BASE_SHA}:local_astra/conductor/session.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"base commit {BASE_SHA} is not in this clone; provenance cannot be re-checked")
    assert blob == BASE_SESSION_BLOB, (
        f"{BASE_SHA}:local_astra/conductor/session.py is {blob}, not the recorded {BASE_SESSION_BLOB}"
    )


def without_persona(got: dict[str, Any], *, persona: str | None = None) -> dict[str, Any]:
    """T-0155 criterion 7: the observed sections with each system message's one persona sentence turned back into
    the pre-T-0155 one (``_pre_t0155_persona.normalise_system_messages``: raises on zero or more than one per
    message). The fixture is never touched; this is the one declared difference."""
    out, seen = P.normalise_system_messages(got, persona=persona)
    assert seen > 0, "premise: the capture holds system messages"
    return dict(out)


async def test_persona_is_the_only_allowed_difference(tmp_path, golden):
    """T-0155 criterion 7 (adversarial, either world): the request bodies with the persona made Gearvis normalise
    back to the golden's; with one more prompt character changed they do not; with two persona sentences per system
    message the normaliser raises."""
    got = await capture(tmp_path)
    want = golden["sections"]["requests"]
    assert without_persona(P.as_t0155(got), persona=P.T0155_NAME)["requests"] == want
    mutant = without_persona(P.one_more_character(got), persona=P.T0155_NAME)["requests"]
    assert mutant != want, (
        "T-0155 criterion 7: a one-character prompt change outside the persona passes the golden"
    )
    with pytest.raises(P.PersonaCountError):
        without_persona(P.two_personas(got), persona=P.T0155_NAME)


async def test_every_run_ending_in_its_own_turn_is_identical_to_the_base_sha(tmp_path, golden):
    """Criterion 6: machine events/commands/states, HUD, turn records and LLM bodies unchanged (T-0155: but for the
    persona sentence, the one declared difference)."""
    # T-0177 B-P1: the published astra.vision.frame messages (shape checked) are the one declared HUD difference.
    got = without_vision_frames(without_persona(await capture(tmp_path)))
    # T-0092b criteria 5-6 (moving pin, widened by exactly T-0092b's two additions, as T-0089 E1 / T-0088 E10):
    # every record gains ``regime.tools == "off"`` and ``tool == null``. Both must be present; both are removed
    # before the comparison, so nothing else about a record may move.
    for i, record in enumerate(got["records"]):
        assert "tool" in record and record["tool"] is None, (
            f"T-0092b criterion 5: record {i} carries tool == null; it has {record.get('tool', '<absent>')!r}"
        )
        assert record["regime"].get("tools") == "off", (
            f"T-0092b criterion 5: record {i} carries regime.tools == 'off'; it has "
            f"{record['regime'].get('tools', '<absent>')!r}"
        )
        del record["tool"]
        del record["regime"]["tools"]
        for key in T0108_EXTRAS | T0127_EXTRAS:
            record["extras"].pop(key, None)
    for section in SECTIONS:
        expected = golden["sections"][section]
        assert got[section] == expected, (
            f"criterion 6: the {section!r} section differs from the base SHA {BASE_SHA[:7]} in a "
            f"session where every run ends inside its own turn — "
            f"{_first_difference(expected, got[section])}"
        )


async def test_the_scenario_is_what_criterion_6_names(golden):
    """The golden is not vacuous: it holds each turn kind, and every terminal reached the machine."""
    machine = golden["sections"]["machine"]
    types = [step["event"]["type"] for step in machine]
    assert types.count("playback.drained") == 5 and types.count("playback.stopped") == 1, (
        "five drained runs (spoken, typed, ui, surface, after-barge-in) and one barge-in stop; "
        f"got {types.count('playback.drained')} drains and {types.count('playback.stopped')} stops"
    )
    inputs = [record["input"] for record in golden["sections"]["records"]]
    assert sorted(inputs) == sorted(["voice", "text", "ui", "voice", "voice", "voice"]), inputs
    hud_types = {m["type"] for m in golden["sections"]["hud"]}
    assert {"astra.ui.surface", "astra.spoken.played"} <= hud_types, sorted(hud_types)
    assert any(r["outcome"] == "interrupted" for r in golden["sections"]["records"])


async def test_the_golden_scenario_is_deterministic_over_two_runs(tmp_path):
    """Premise: the capture is reproducible in one process, so a red golden is a change, not noise."""
    first = await capture(tmp_path / "a")
    second = await capture(tmp_path / "b")
    for section in SECTIONS:
        assert first[section] == second[section], (
            f"the {section!r} section is not deterministic — {_first_difference(first[section], second[section])}"
        )


def test_the_hud_section_drops_exactly_astra_thought():
    """T-0180 B-R3 (adversarial): the ``hud`` filter is type-exact. One example of every Astra message model plus four
    near-miss types written by hand go in; only the ``AstraThought`` example comes out missing, every other entry
    comes out dumped, unchanged and in order. A prefix or substring filter, or one that drops nothing, fails here."""
    from local_astra.contracts.protocol import ASTRA_MESSAGE_TYPES, AstraThought
    from tests.unit._examples import example_of

    models = [example_of(m) for m in ASTRA_MESSAGE_TYPES]
    thoughts = [m for m in models if isinstance(m, AstraThought)]
    assert len(thoughts) == 1, "premise: AstraThought is one of the Astra message models"
    near = [
        {"type": "astra.thoughts", "text": "a"},
        {"type": "astra.thought.x", "text": "b"},
        {"type": "astra.thought ", "text": "c"},
        {"type": "Astra.Thought", "text": "d"},
    ]
    inputs: list[Any] = [*models, *near]
    want = [_dump(m) for m in inputs if m is not thoughts[0]]
    assert len(want) == len(inputs) - 1
    assert hud_section(inputs) == want, (
        "T-0180 B-R3: the hud capture drops messages of type exactly 'astra.thought' and nothing else"
    )


if __name__ == "__main__":  # pragma: no cover - the generator imports ``capture`` instead
    raise SystemExit("run tests/fixtures/session/capture_t0092a_golden.py")
