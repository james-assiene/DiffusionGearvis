"""T-0092b criterion 6's golden (tools = off changes nothing) and criterion 7's scope negatives.

The golden was captured at the literal base SHA **a502871d…** — ``a502871``, the T-0092a merge commit T-0092b
branched from (the branch head d2e2879 changes only docs on top of it, so every owned file is byte-identical to
a502871's) — by ``tests/fixtures/session/capture_t0092b_golden.py``. That script refuses to write unless every
file T-0092b owns hashes to its a502871 blob. The fixture records the SHA and the blobs; it is never compared
against a moving ref (``.claude/rules/tests.md``).

The scripted session (criterion 6, verbatim): a spoken turn, a typed turn, a UI action, an A2UI surface, a
barge-in, a TTS error and an over-long retry, with ``tools = off`` (no kit), the session log on. Identical to
the base, section by section:

* ``machine``  — every event the machine was stepped with, the full state it returned, its commands;
* ``hud``      — every HUD message published;
* ``prompts``  — every ``Prompt`` the fake backend was handed (prefills and generates, in order);
* ``requests`` — every request body, built by the real ``VllmClient.request_body`` from those prompts
  (generates as sent; prefills as ``request_body(prefix, …, LlmOptions(max_tokens=1))``, the client's rule);
* ``log``      — ``kind``/``data`` of every ``llm.*``, ``machine.step`` and ``tts.*`` session-log record;
* ``records``  — every ``turns.jsonl`` line, which may differ from the base **only** by the two added keys
  ``regime.tools == "off"`` and ``tool == null`` (criterion 6's last bullet), and must carry both.

Normalised before comparing, and only these: the random ``session_id`` and the temporary directory
and the session log's file name, which carries the wall time
(→ ``<session>``, ``<tmp>``, ``<log>``), and ``Regime.git_commit`` (→ ``<git>``). Times are the ``ScriptedClock``'s.

T-0180 B-R3 (2026-10-28): the ``hud`` section leaves out messages whose ``type`` is exactly ``astra.thought`` (the
base published none); every other type compares as before, and ``test_t0180_differential.py`` (B8) pins the thoughts.

Regime: CPU over fakes, ``ScriptedClock``, profile ``lean`` defaults, thinking on, 200 ms/word, n = 1,
deterministic (a row below re-captures twice).
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
from local_astra.contracts.llm import LlmOptions
from local_astra.contracts.turn import MachineState, TurnState
from tests import _git_range as gr
from tests.fakes import FakeMemory, FakeStt
from tests.fakes.llm import http_error
from tests.unit import _pre_t0155_persona as P
from tests.unit._hud_rig import Recorder, ui
from tests.unit._session_log_rig import StepClock, read_records
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

#: The base commit of T-0092b: T-0092a's merge (literal; never ``main`` or a merge base).
BASE_SHA = "a502871"
#: ``git rev-parse a502871:<path>`` of every file T-0092b owns — the code the golden ran.
BASE_BLOBS = {
    "local_astra/conductor/session.py": "d23f8f2ac0be4c197c206dec87ff764f40cb00d0",
    "local_astra/conductor/prompt_builder.py": "a9d9279fd19c1db3aedd2ee7d0ba987482584b6b",
    "local_astra/cli.py": "a3efeef0ae3463ab6391108419bc8eec5c31806e",
    "local_astra/config.py": "34defda2223a05808dc09defca94d5f4a4ccc1a0",
    "local_astra/telemetry/turns_log.py": "c6d794a503fed2b1524a8f6a7bd27edb000e66cd",
    "astra.toml": "c75929ed0d8663b4524fd353d05b0eb7ae9d9236",
}
GOLDEN = REPO / "tests" / "fixtures" / "session" / "t0092b_tools_off_golden.json"
#: T-0108 criterion 10's two additive turn-record extras, discounted under that task's name.
T0108_EXTRAS = frozenset({"ui_valid_final", "ui_shown"})
#: T-0127 criterion 5's additive turn-record extra, stripped beside them (re-scope R2).
T0127_EXTRAS = frozenset({"ui_complete_final"})

MS_PER_WORD = 200
SECTIONS = ("machine", "hud", "prompts", "requests", "log")
LOG_KINDS = ("llm.", "machine.step", "tts.")

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
TTS_FAILS = [answer("This sentence never reaches a speaker. ")]
RETRIED = [answer("Answered on the retry. ")]


class StateMachine(RecordingMachine):
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
    """Tools off: the session is built exactly as at the base — no kit argument at all."""
    require_session()
    from local_astra.telemetry.session_log import SessionLog

    clock = ScriptedClock()
    machine = StateMachine()
    llm = RecordingLlm(SPOKEN)
    llm.clock = clock
    tts = RecordingTts(ms_per_word=MS_PER_WORD)
    audio = RecordingAudio(clock)
    published = Recorder()
    turns_path = tmp_path / "turns.jsonl"
    log = SessionLog(tmp_path / "logs", clock=StepClock())
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
        session_log=log,
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
    rig.log = log  # type: ignore[attr-defined]
    await session.start()
    try:
        yield rig
    finally:
        await session.aclose()


async def run_scenario(rig: Rig) -> None:
    """Criterion 6's session: spoken, typed, UI action, surface, barge-in, TTS error, over-long retry."""
    rig.llm.script = SPOKEN
    await rig.ask("what is on this screen")
    await rig.speak_out()
    await rig.advance(300)
    rig.llm.script = TYPED
    rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="and what does it mean"))
    await rig.speak_out()
    await rig.advance(300)
    rig.llm.script = CLICKED
    rig.submit(
        E.UiAction(t_ms=rig.clock.now_ms, surface_id="lesson_1", name="next", context={"step": 2})
    )
    await rig.speak_out()
    await rig.advance(300)
    rig.llm.script = SURFACE
    await rig.ask("can you illustrate one half")
    await rig.speak_out()
    await rig.advance(300)
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
    await rig.advance(300)
    # a TTS error: the synthesis raises, the turn ends on the machine's error row
    rig.tts.fail = True
    rig.llm.script = TTS_FAILS
    await rig.ask("say something")
    await rig.speak_out()
    rig.tts.fail = False
    await rig.advance(300)
    # an over-long prompt: the first generate is refused with HTTP 413, the session retries once
    rig.llm.fail_generates(http_error(413), None)
    rig.llm.script = RETRIED
    await rig.ask("one more question")
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


def capture_sections(rig: Rig, log_records: list[dict[str, Any]]) -> dict[str, Any]:
    machine = rig.machine
    assert isinstance(machine, StateMachine)
    client = VllmClient(transport=_refusing())
    prompts = [
        {"phase": "prefill", "request_id": rid, "prompt": _dump(p)} for rid, p in rig.llm.prefills
    ] + [
        {"phase": "generate", "request_id": rid, "prompt": _dump(p), "options": _dump(o)}
        for rid, p, o in rig.llm.generates
    ]
    requests = [
        client.request_body(prefix, rid, LlmOptions(max_tokens=1))
        for rid, prefix in rig.llm.prefills
    ] + [
        client.request_body(prompt, rid, options, xargs)
        for (rid, prompt, options), xargs in zip(rig.llm.generates, rig.llm.xargs_log, strict=True)
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
        "prompts": prompts,
        "requests": requests,
        "log": [
            {"kind": r["kind"], "data": r["data"]}
            for r in log_records
            if str(r.get("kind", "")).startswith(LOG_KINDS)
        ],
        "records": [json.loads(line) for line in rig.lines()],
    }


def normalise(tree: Any, session_id: str, tmp_path: Path, log_path: Path) -> Any:
    text = (
        json.dumps(tree, sort_keys=True)
        .replace(str(log_path), "<log>")
        .replace(session_id, "<session>")
        .replace(str(tmp_path), "<tmp>")
    )
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
        session_id = rig.session.session_id
        failures = list(rig.session.failures)
        log_path = rig.log.path  # type: ignore[attr-defined]
    assert not failures, f"premise: the scripted session ran without failures: {failures}"
    sections = capture_sections(rig, read_records(log_path))
    return normalise(sections, session_id, tmp_path, Path(log_path))


def _first_difference(expected: list[Any], got: list[Any]) -> str:
    for i, (a, b) in enumerate(zip(expected, got, strict=False)):
        if a != b:
            return (
                f"first difference at index {i}:\n  base {BASE_SHA}: "
                f"{json.dumps(a, sort_keys=True)[:1500]}\n  now:          "
                f"{json.dumps(b, sort_keys=True)[:1500]}"
            )
    return f"lengths differ: base {BASE_SHA} has {len(expected)}, now {len(got)}"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    assert GOLDEN.is_file(), f"the golden fixture {GOLDEN} is missing (see tests/fixtures/session/)"
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["base_sha"] == BASE_SHA and data["blobs"] == BASE_BLOBS, (
        "the golden must be the one captured at the literal base SHA, against its owned files; "
        f"the fixture says {data['base_sha']} / {data['blobs']}"
    )
    return data


def test_the_golden_was_captured_against_the_base_owned_files():
    """Provenance: the blobs the fixture names are a502871's (skipped when the commit is absent)."""
    for path, blob in BASE_BLOBS.items():
        try:
            got = subprocess.run(
                ["git", "rev-parse", f"{BASE_SHA}:{path}"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip(
                f"base commit {BASE_SHA} is not in this clone; provenance cannot be re-checked"
            )
        assert got == blob, f"{BASE_SHA}:{path} is {got}, not the recorded {blob}"


def without_persona(got: dict[str, Any], *, persona: str | None = None) -> dict[str, Any]:
    """T-0155 criterion 7: the observed sections with each system message's one persona sentence turned back into
    the pre-T-0155 one (``_pre_t0155_persona.normalise_system_messages``: raises on zero or more than one per
    message). The fixture is never touched; this is the one declared difference."""
    out, seen = P.normalise_system_messages(got, persona=persona)
    assert seen > 0, "premise: the capture holds system messages"
    return dict(out)


async def test_tools_off_is_identical_to_the_base_sha(tmp_path, golden):
    """Criterion 6: machine, HUD, prompts, request bodies and the llm/machine/tts log records are unchanged (T-0155:
    but for the persona sentence, the one declared difference)."""
    # T-0177 B-P1: the published astra.vision.frame messages (shape checked) are the one declared HUD difference.
    got = without_vision_frames(without_persona(await capture(tmp_path)))
    for section in SECTIONS:
        expected = golden["sections"][section]
        assert got[section] == expected, (
            f"T-0092b criterion 6: with tools = off the {section!r} section differs from the base "
            f"SHA {BASE_SHA} — {_first_difference(expected, got[section])}"
        )


async def test_persona_is_the_only_allowed_difference(tmp_path, golden):
    """T-0155 criterion 7 (adversarial, either world): the capture with its persona made Gearvis normalises back to
    the golden; with one more prompt character changed it does not (prompts, requests and log all differ); with two
    persona sentences per system message the normaliser raises."""
    got = without_vision_frames(await capture(tmp_path))  # T-0177 B-P1
    post = without_persona(P.as_t0155(got), persona=P.T0155_NAME)
    for section in SECTIONS:
        assert post[section] == golden["sections"][section], (
            f"the persona-only capture differs in {section!r}"
        )
    mutant = without_persona(P.one_more_character(got), persona=P.T0155_NAME)
    for section in ("prompts", "requests", "log"):
        assert mutant[section] != golden["sections"][section], (
            f"T-0155 criterion 7: a one-character prompt change outside the persona passes the {section!r} golden"
        )
    with pytest.raises(P.PersonaCountError):
        without_persona(P.two_personas(got), persona=P.T0155_NAME)


async def test_tools_off_turn_records_differ_only_by_the_two_added_keys(tmp_path, golden):
    """Criterion 6, last bullet: each record equals the base's plus exactly ``regime.tools == "off"`` and
    ``tool == null`` — both present (criterion 5 writes them on every line), nothing else moved."""
    got = (await capture(tmp_path))["records"]
    expected = golden["sections"]["records"]
    assert len(got) == len(expected), (
        f"one record per turn, as at the base: {len(expected)} vs {len(got)}"
    )
    for i, (base, now) in enumerate(zip(expected, got, strict=True)):
        assert "tool" in now and now["tool"] is None, (
            f"T-0092b criterion 6: record {i} carries the added key tool == null; it has "
            f"{now.get('tool', '<absent>')!r}"
        )
        assert now["regime"].get("tools") == "off", (
            f"T-0092b criterion 6: record {i} carries regime.tools == 'off'; it has "
            f"{now['regime'].get('tools', '<absent>')!r}"
        )
        stripped = {k: v for k, v in now.items() if k != "tool"}
        stripped["regime"] = {k: v for k, v in now["regime"].items() if k != "tools"}
        stripped["extras"] = {
            k: v for k, v in now["extras"].items() if k not in T0108_EXTRAS | T0127_EXTRAS
        }
        assert stripped == base, (
            f"T-0092b criterion 6: record {i} differs from the base by more than the two added keys — "
            f"{_first_difference([base], [stripped])}"
        )


def test_the_scenario_is_what_criterion_6_names(golden):
    """The golden is not vacuous: every turn kind criterion 6 lists is in it."""
    sections = golden["sections"]
    inputs = sorted(r["input"] for r in sections["records"])
    assert inputs.count("text") == 1 and inputs.count("ui") == 1 and inputs.count("voice") >= 5, (
        inputs
    )
    outcomes = [r["outcome"] for r in sections["records"]]
    assert "interrupted" in outcomes, f"a barge-in: {outcomes}"
    assert "error" in outcomes, f"the TTS error ends its turn on the error row: {outcomes}"
    assert {"astra.ui.surface", "astra.spoken.played"} <= {m["type"] for m in sections["hud"]}
    retried = [r for r in sections["records"] if r["extras"].get("prompt_error_status") == 413]
    assert len(retried) == 1 and retried[0]["outcome"] == "spoken", "the over-long retry answered"
    assert sum(1 for p in sections["prompts"] if p["phase"] == "generate") >= 9, (
        "eight turn generates plus the retry's second attempt"
    )
    kinds = {e["kind"] for e in sections["log"]}
    assert {"llm.request", "llm.commit", "llm.done", "machine.step", "tts.utterance"} <= kinds, (
        kinds
    )
    assert all(p["prompt"]["tools"] is None for p in sections["prompts"]), (
        "tools off: no tools field"
    )


async def test_the_golden_scenario_is_deterministic_over_two_runs(tmp_path):
    """Premise: the capture is reproducible in one process, so a red golden is a change, not noise."""
    first = await capture(tmp_path / "a")
    second = await capture(tmp_path / "b")
    for section in (*SECTIONS, "records"):
        assert first[section] == second[section], (
            f"the {section!r} section is not deterministic — "
            f"{_first_difference(first[section], second[section])}"
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


# --- criterion 7: files T-0092b must leave alone (pinned to the literal base SHA) ----------------

UNCHANGED = (
    "local_astra/contracts",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/toolcall",
    "local_astra/conductor/tool_executor.py",
    # ``local_astra/conductor/executor.py`` left this list with T-0092c criterion 5 (moving pin): its tool handlers
    # become live when tools are on. The tools-off behaviour stays pinned by
    # ``test_session_loop.py::test_t0089_tool_commands_have_named_inert_handlers`` (unedited, still listed below) and
    # by this module's golden replay.
    "tests/unit/test_session_loop.py",
)


#: T-0092b's merge commit 4f8d4e0; BASE = ``git merge-base 4f8d4e0^1 4f8d4e0^2`` = d2e2879 (the branch head on top of
#: a502871, docs only; not ``BASE_SHA``, which stays the golden's provenance), TIP = ``4f8d4e0^2`` (T-0094b criterion 2).
T0092B_MERGE = "4f8d4e0"
T0092B_BASE = "d2e2879"
T0092B_TIP = "04e07f6"
#: Power pair of the scope row (T-0094b ruling P2): T-0091's d55cfce (parent a7c00a8) edited ``toolcall/live.py``.
SCOPE_PRIOR_PARENT = "a7c00a8"
SCOPE_PRIOR = "d55cfce"
#: Power pair of the new-file row (ruling P5): T-0091's 52d6b54 (parent c986d33) added ``conductor/tool_executor.py``.
NEW_FILE_PRIOR_PARENT = "c986d33"
NEW_FILE_PRIOR = "52d6b54"
CONDUCTOR = "local_astra/conductor"


def test_the_files_criterion_7_names_are_unchanged_from_the_base():
    """Criterion 7: ``contracts/**``, ``llm_client.py``, ``turn_machine.py``, ``toolcall/**`` and
    ``tool_executor.py`` unchanged (``executor.py`` dropped by T-0092c criterion 5, see ``UNCHANGED``) and
    ``test_session_loop.py`` (which holds the inert-handler row) unedited, over T-0092b's closed range (T-0094b). The
    untracked-file half of the old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0092B_BASE, T0092B_TIP, T0092B_MERGE)
    changed = gr.changed(T0092B_BASE, T0092B_TIP, *UNCHANGED)
    assert not changed, (
        f"T-0092b criterion 7: these paths are unchanged in {T0092B_BASE}..{T0092B_TIP}; changed {changed}"
    )


def test_the_criterion_7_guard_has_power():
    """T-0094b criterion 5 (adversarial): over d55cfce's parent and d55cfce the same predicate over ``UNCHANGED``
    reports ``live.py`` under the toolcall entry of the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(SCOPE_PRIOR_PARENT, SCOPE_PRIOR)
    assert gr.is_ancestor(SCOPE_PRIOR, T0092B_BASE)
    toolcall = next(p for p in UNCHANGED if p.endswith("/toolcall"))
    assert f"{toolcall}/live.py" in gr.changed(SCOPE_PRIOR_PARENT, SCOPE_PRIOR, *UNCHANGED)


def _new_conductor_files(base: str, tip: str) -> list[str]:
    """The implementation note's predicate over two commits: ``*.py`` files directly under the conductor package at
    ``tip`` that are not there at ``base``."""

    def listed(sha: str) -> set[str]:
        out = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{sha}:{CONDUCTOR}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        return {n for n in out if n.endswith(".py")}

    return sorted(listed(tip) - listed(base))


def test_no_new_file_under_the_conductor_package():
    """Implementation note: no new conductor file (the T-0080a fork-point guard excuses none), between T-0092b's own
    BASE and TIP trees (T-0094b: never the working tree)."""
    gr.prove_range(T0092B_BASE, T0092B_TIP, T0092B_MERGE)
    base_rev, tip_rev = T0092B_BASE, T0092B_TIP
    assert gr.changed(base_rev, tip_rev, CONDUCTOR), "premise: two different trees"
    added = _new_conductor_files(base_rev, tip_rev)
    assert not added, f"T-0092b: no new file under local_astra/conductor/; added {added}"


def test_the_new_file_row_has_power():
    """T-0094b criterion 5 (adversarial): over 52d6b54's parent and 52d6b54 the same predicate names the added
    ``tool_executor.py``; a predicate that is always empty fails here."""
    gr.require_commits(NEW_FILE_PRIOR_PARENT, NEW_FILE_PRIOR)
    assert gr.is_ancestor(NEW_FILE_PRIOR, T0092B_BASE)
    assert _new_conductor_files(NEW_FILE_PRIOR_PARENT, NEW_FILE_PRIOR) == ["tool_executor.py"]


def test_the_new_file_row_compares_two_different_trees():
    """T-0094b too-strict twin: the conductor package did change in T-0092b's range (its owned ``session.py``), so a
    pair reduced to TIP vs TIP, which sees nothing at all, fails here."""
    gr.prove_range(T0092B_BASE, T0092B_TIP, T0092B_MERGE)
    assert f"{CONDUCTOR}/session.py" in gr.changed(T0092B_BASE, T0092B_TIP, CONDUCTOR)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("run tests/fixtures/session/capture_t0092b_golden.py")
