"""T-0098: the offline native-prompt A/B -- arm texts, request fidelity, the spoken definition and the
pre-registered decision rule (unit; CPU; no engine).

Every row is ``pending(task="T-0098")`` until ``/implement T-0098`` starts. The harness does not exist yet, so the
rows are red on the named assertion in ``harness()``, never on an ImportError at collection (it is loaded lazily, by
path, as ``tests/unit/test_t0093_report.py`` loads T-0093's).

**The assumed surface** (``scripts/gpu/t0098_native_prompt_ab.py``; importing it opens no socket and starts
nothing). The task names the behaviour but not these names; they are the test-author's and are stated here so the
implementer can honour them or say why not:

- ``load_cases() -> list[Case]``: the 53 cases in run order -- the dev split's 40 tool cases then its 8 no-tool
  cases (``tests/eval/toolcall/cases.jsonl`` file order), then T-0093's 5 ``PLAIN_QUESTIONS`` in their order. A case
  has at least ``case_id``, ``kind`` (``tool`` | ``no_tool`` | ``plain``) and ``text`` (T-0093's ``Case`` shape).
- ``arm_prompts(case, kit) -> Mapping[str, Prompt]``: the six pass-1 prompts of one case, keyed ``A0 A1 D1 D2 C2
  C1``. ``kit`` is a ``session.ToolKit`` (only ``kit.tools`` matters to the prompt).
- ``classify(commits) -> Mapping``: one trial's reading from its commit list (each commit a dict with ``segment``,
  ``text``, ``thought_open``, ``canvas_idx``, as ``trials.jsonl`` stores them), with keys ``routed``, ``spoken``,
  ``spoken_prefix``, ``thought_open`` and ``anomaly`` (criterion 5).
- ``build_report(run_dir) -> dict`` (reads files only) and ``render_markdown(report) -> str`` (report.md's text).
  The report keys these rows read:
  ``report["candidates"][c][p]`` (bool) for ``c`` in ``C2 C1`` and ``p`` in ``P1``..``P7`` plus ``"passes"``;
  ``report["readings"]["R1"]`` / ``["R2"]`` (bool: the reading holds); ``report["winner"]``: ``"C2"``, ``"C1"`` or
  no change (``None`` or the string ``"no change"``); ``report["void_arms"]``: the void arms' names;
  ``report["arms"][arm]["tools"][tool]["spoken"] == {"k": spoken, "n": routed}``.

**The run directory these rows build by hand** (criterion 4): ``trials.jsonl``, one line per request, with
``case_id, kind, gt, arm, repeat, order, system_sha256, commits, token_ids, error, wall_ms, first_canvas_ms``
(``kind`` and ``gt`` are extras the criterion does not list: they agree with ``cases.jsonl`` by ``case_id``, so a
report that joins on the case id reads the same thing), written in the rotated order of criterion 3; ``order`` is the
arm's slot (0-5) in its (case, repeat) block. ``manifest.json`` holds ``regime`` and ``arms`` (name -> ``text``,
``sha256``).

**The baseline run** (every decision row edits it; all numbers hand-computed). 53 cases x 6 arms x 5 repeats = 1590
lines; 265 per arm. Outcomes (the commit lists are in ``commits_for``):

- ``A0`` (control, the finding): every ``type_at`` trial is SILENT (the plan sentence inside a closed thought, then
  the call), except type-05 repeat 0, a call inside an unclosed thought (as live type-05 was); every other tool trial
  speaks. So ``type_at`` routed 49/50, spoken 0/49 (R1 holds: 0 <= 0.3); pooled route recall 199/200; click + scroll
  + key spoken 100/100; valid 199/199, exact 199/199; false tools 0/65; anomalies 0; thought_open 1.
- ``A1`` (positive control): every tool trial speaks, except type-01..type-06 repeat 0, which answer with no call
  (unrouted, thought closed). ``type_at`` routed 44/50, spoken 44/44 (R2 holds); pooled recall 194/200. A1 is
  deliberately *worse* than A0 on recall, so a P3 read against A1 (a reviewer mutation) changes outcomes.
- ``D1``, ``D2``, ``C2``, ``C1``: every tool trial speaks; no-tool and plain trials answer with no call. C2 and C1
  pass P1-P7 (type_at 50/50, majority in 10/10 cases; recall 200/200 >= 0.995 - 0.02; type recall 50/50 >= 0.98 -
  0.04; valid and exact 200/200; 0 false tools; 0 anomalies; thought_open 0 <= 1 + 4), so the winner is **C2**.

Regime: CPU, pure, hand-built data; no measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import VllmClient
from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.eval.cd import twopass
from tests.unit._pre_t0155_persona import (
    PRE_T0155_A0_SHA256,
    pre_t0155_fixture,
    run_main_against_live_persona,
)
from tests.unit._pre_t0155_persona import (
    PRE_T0155_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,  # T-0155 criterion 8: the harness's own persona
)

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit (the
#: pre-T-0155 texts patched into session, twopass and, through session, each harness loaded by path inside a row).
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0098_native_prompt_ab.py"
CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"

#: Evidence (T-0098, planner 2026-10-14): the live system text at 4b0bd92, identical on every case; 2744 characters.
A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
A0_CHARS = 2744
#: ``twopass.NATIVE_SYSTEM_PROMPT``; the task quotes the prefix ``e0b54a58...``.
A1_SHA256_PREFIX = "e0b54a58"
ARMS = ("A0", "A1", "D1", "D2", "C2", "C1")
CANDIDATES = ("C2", "C1")
PS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
REPEATS = 5

#: H-T98a (b), verbatim from the task file: the sentence C1 adds, and the paragraph as it must then read.
C1_SENTENCE = (
    "Say that sentence in your reply, after you have finished thinking: what you write while thinking is never "
    "heard."
)
C1_PARAGRAPH = (
    "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then "
    "call exactly one of the declared tools. Say that sentence in your reply, after you have finished thinking: "
    "what you write while thinking is never heard. Use at most one call per answer. When no action is needed, "
    "answer normally and call no tool."
)
#: The two paragraphs of ``NATIVE_TOOL_INSTRUCTIONS`` as they read at 2c584d9 (typed out, not read from the module).
LIVE_PARAGRAPH = (
    "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then "
    "call exactly one of the declared tools. Use at most one call per answer. When no action is needed, answer "
    "normally and call no tool."
)
COORDS_PARAGRAPH = (
    "Screen coordinates x and y are integers from 0 to 999, normalised to the attached screenshot: (0, 0) is its "
    "top-left corner and (999, 999) its bottom-right corner."
)

#: T-0093 J10, pre-registered there (typed out here; T-0098 reuses them unchanged).
PLAIN_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("plain-01", "What is the capital of Japan?"),
    ("plain-02", "Tell me a fun fact about octopuses."),
    ("plain-03", "How many minutes are there in a day?"),
    ("plain-04", "Who painted the Mona Lisa?"),
    ("plain-05", "How are you doing today?"),
)


def harness() -> ModuleType:
    """The harness, loaded by path. Red here, by name, until T-0098 criterion 1 lands."""
    assert HARNESS.is_file(), f"T-0098 criterion 1: {HARNESS.relative_to(REPO)} does not exist yet"
    name = "_t0098_native_prompt_ab_unit"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    for attr in (
        "load_cases",
        "arm_prompts",
        "classify",
        "build_report",
        "render_markdown",
        "main",
    ):
        assert callable(getattr(module, attr, None)), (
            f"T-0098 surface: {HARNESS.name} has no callable {attr!r} (see this module's docstring)"
        )
    return module


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_arm_texts() -> dict[str, str]:
    """The six system texts, each built here by hand from the pieces the task names (criterion: "checked by
    constructing the expected string by hand, not by sha alone")."""
    p, ct, ss, ui, nti = (
        DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        UI_INSTRUCTIONS,
        pb.NATIVE_TOOL_INSTRUCTIONS,
    )
    return {
        "A0": "\n".join([p, ct, ss, ui, nti]),
        "A1": twopass.NATIVE_SYSTEM_PROMPT,
        "D1": "\n".join([p, ct, ss, nti]),
        "D2": "\n".join([p, ct, ui, nti]),
        "C2": "\n".join([p, ct, ss, ui]) + "\n\n" + nti,
        "C1": "\n".join([p, ct, ss, ui, C1_PARAGRAPH + "\n\n" + COORDS_PARAGRAPH]),
    }


def tool_kit(**kwargs: Any) -> Any:
    from tests.unit._tools_switch_support import make_kit

    return make_kit(**kwargs)


def case_rows() -> list[dict[str, Any]]:
    """The 53 cases in run order, from the pinned file and the J10 list (an oracle independent of the harness)."""
    rows = [
        json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    dev = [r for r in rows if r.get("split") == "dev"]
    tool = [r for r in dev if r["kind"] == "tool"]
    no_tool = [r for r in dev if r["kind"] == "no_tool"]
    plain = [
        {"id": cid, "kind": "plain", "user": text, "gt": None, "tool": None, "reference_call": None}
        for cid, text in PLAIN_QUESTIONS
    ]
    return [*tool, *no_tool, *plain]


def harness_case(h: ModuleType, case_id: str) -> Any:
    found = [c for c in h.load_cases() if c.case_id == case_id]
    assert len(found) == 1, f"criterion 3: load_cases() holds {case_id!r} {len(found)} times"
    return found[0]


# =====================================================================================================
# Criterion 3: the cases
# =====================================================================================================


def test_load_cases_is_the_dev_design() -> None:
    """Criterion 3 and 10: 40 dev tool + 8 dev no-tool + the 5 plain questions = 53, in file order, and not one case
    of the test split. Expected ids from the file itself (``split == "dev"``) and J10's list."""
    h = harness()
    got = [(c.case_id, c.kind) for c in h.load_cases()]
    want = [(r["id"], r["kind"]) for r in case_rows()]
    assert len(want) == 53 and want[:2] == [("search-01", "tool"), ("search-02", "tool")]
    assert got == want, f"criterion 3: the run's cases are {got[:5]}..., not {want[:5]}..."
    rows = [
        json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    test_ids = {r["id"] for r in rows if r.get("split") == "test"}
    assert len(test_ids) == 140 - 48, "premise: the file's test split"
    assert not test_ids & {cid for cid, _ in got}, "criterion 10: a test-split case is in the run"


# =====================================================================================================
# The arms (H-T98a (a), (b))
# =====================================================================================================


def test_arm_texts_and_shas() -> None:
    """A0 is the builder's text (sha ``3605c012...``, 2744 characters); A1 is twopass (``e0b54a58...``); D1, D2, C2 are
    each one exact edit of A0 and C1 carries the approved sentence verbatim, at the approved place. Every expected
    string is built here by hand; the shas are the task's literals."""
    h = harness()
    want = expected_arm_texts()
    # premises: the hand-typed paragraphs are the live constant, and A0 is the 4b0bd92 text
    assert pb.NATIVE_TOOL_INSTRUCTIONS == LIVE_PARAGRAPH + "\n\n" + COORDS_PARAGRAPH
    assert sha256(want["A0"]) == A0_SHA256 and len(want["A0"]) == A0_CHARS
    assert sha256(want["A1"]).startswith(A1_SHA256_PREFIX)
    assert (
        C1_SENTENCE in C1_PARAGRAPH
        and C1_PARAGRAPH.replace(C1_SENTENCE + " ", "") == LIVE_PARAGRAPH
    )
    assert len(set(want.values())) == 6, "premise: the six arms are six texts"

    kit = tool_kit()
    builder_a0 = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools).system_message().text()
    assert builder_a0 == want["A0"], "premise: the builder's text is the hand join"

    for case_id in ("type-01", "search-01"):
        prompts = h.arm_prompts(harness_case(h, case_id), kit)
        assert set(prompts) == set(ARMS), f"H-T98a (a): the arms are {sorted(prompts)}"
        for arm in ARMS:
            first = prompts[arm].messages[0]
            assert first.role == "system", f"{arm}: messages[0] is {first.role}"
            got = first.text()
            assert got == want[arm], (
                f"{case_id} {arm}: the system text is not the hand-built one ({len(got)} chars)"
            )
        assert sha256(prompts["A0"].messages[0].text()) == A0_SHA256
        assert C1_SENTENCE in prompts["C1"].messages[0].text()
        # one edit each, measured against A0 (so an arm that is A0 itself, or A0 with two edits, is caught)
        a0 = prompts["A0"].messages[0].text()
        assert prompts["D1"].messages[0].text() == a0.replace("\n" + UI_INSTRUCTIONS, "", 1)
        assert prompts["D2"].messages[0].text() == a0.replace("\n" + pb.SPOKEN_STYLE, "", 1)
        assert prompts["C2"].messages[0].text() == a0.replace(
            "\n" + pb.NATIVE_TOOL_INSTRUCTIONS, "\n\n" + pb.NATIVE_TOOL_INSTRUCTIONS, 1
        )
        assert prompts["C1"].messages[0].text() == a0.replace(LIVE_PARAGRAPH, C1_PARAGRAPH, 1)


def _without_system(prompt: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = prompt.model_dump(mode="json")
    dumped["messages"] = dumped["messages"][1:]
    return dumped


def _body_without_system(body: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(body))
    out["messages"] = out["messages"][1:]
    return out


@pytest.mark.parametrize("case_id", ["type-01", "search-01"])
def test_bodies_differ_only_in_system(case_id: str) -> None:
    """Criterion 2: for an image case (type-01) and a text-only case (search-01), every arm's prompt -- and the body
    the real ``VllmClient.request_body`` makes of it -- equals A0's in everything but ``messages[0]``, and differs
    there. The tools ride on every arm (``tool_choice: "none"``); the image case carries an image, the text-only case
    the ``[screen]`` note. (The wire bodies the harness actually sends are checked in the integration smoke.)"""
    h = harness()
    kit = tool_kit()
    prompts = h.arm_prompts(harness_case(h, case_id), kit)
    client = VllmClient()
    a0 = prompts["A0"]
    a0_body = client.request_body(a0, "r")
    assert a0_body["tools"] == list(kit.tools) and a0_body["tool_choice"] == "none"
    assert "seed" not in a0_body
    user = a0.messages[-1]
    kinds = [type(p).__name__ for p in user.content]
    if case_id == "type-01":
        assert "ImagePart" in kinds, f"the image case sends no image: {kinds}"
    else:
        assert "ImagePart" not in kinds and pb.NO_IMAGE_STATEMENT in user.text(), (
            "the text-only case lacks the [screen] note"
        )
    for arm in ARMS[1:]:
        p = prompts[arm]
        assert _without_system(p) == _without_system(a0), (
            f"{case_id} {arm}: differs outside messages[0]"
        )
        assert p.messages[0] != a0.messages[0], f"{case_id} {arm}: messages[0] is A0's"
        body = client.request_body(p, "r")
        assert _body_without_system(body) == _body_without_system(a0_body), f"{case_id} {arm}: body"
        assert body["messages"][0] != a0_body["messages"][0]


# =====================================================================================================
# Criterion 2: A0 is what a Session sends
# =====================================================================================================


class _NoMemory:
    """No recollections (criterion 2: "no memory hits"), never an uncaptioned episode."""

    def __init__(self) -> None:
        self._next = 0

    async def ingest(self, episode: Any) -> int:
        self._next += 1
        return self._next

    async def search(self, query: Any) -> Sequence[Any]:
        return ()

    async def set_caption(self, episode_id: int, caption: str) -> None:
        return None

    async def forget_since(self, t0_ms: int) -> int:
        return 0

    async def next_uncaptioned(self) -> Any:
        return None


def _session_prompt(tmp_path: Path, row: Mapping[str, Any], kit: Any) -> tuple[Any, dict[str, Any]]:
    """One typed turn through a real ``Session`` over the fakes (virtual clock), the case's own frame as the turn's
    screen (built as T-0093 built it: ts 0, the fixture's size, phash = sha256(jpeg)[:16]); returns pass 1's prompt
    and the body the real client would send for it."""
    import asyncio

    from local_astra.contracts.perception import Frame, Source
    from tests.fakes import FakeStt
    from tests.fakes.llm import is_caption_prompt
    from tests.unit._tools_switch_support import one_typed_turn
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
        require_session,
    )

    require_session()
    frames = []
    if row.get("screen"):
        jpeg = (CASES.parent / row["screen"]).read_bytes()
        w, hgt = row["screen_size"]
        frames.append(
            Frame(
                source=Source.SCREEN,
                ts_ms=0,
                width=int(w),
                height=int(hgt),
                jpeg=jpeg,
                phash=hashlib.sha256(jpeg).hexdigest()[:16],
            )
        )
    clock = ScriptedClock()
    llm = RecordingLlm([answer("Okay.")])
    llm.clock = clock
    turns = tmp_path / "turns.jsonl"
    machine = RecordingMachine(POLICY)
    stt, tts, audio = FakeStt(), RecordingTts(ms_per_word=200), RecordingAudio(clock)
    perception, memory = RecordingPerception(frames), _NoMemory()
    session = Session(
        machine,
        llm,
        stt,
        tts,
        audio,
        perception,
        memory,
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns),
        tool_kit=kit,
    )
    rig = Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=stt,
        tts=tts,
        audio=audio,
        perception=perception,
        memory=memory,
        session=session,
        turns_path=turns,
    )

    async def go() -> None:
        await session.start()
        try:
            await one_typed_turn(rig, str(row["user"]))
        finally:
            await session.aclose()

    asyncio.run(go())
    passes = [
        (rid, p)
        for rid, p, _ in llm.generates
        if not is_caption_prompt(p) and not p.continue_final_message
    ]
    assert len(passes) == 1, f"premise: one pass-1 generate, got {len(passes)}"
    rid, prompt = passes[0]
    bodies = [b for b in llm.bodies if b["vllm_xargs"]["astra_request_id"] == rid]
    assert len(bodies) == 1
    return prompt, bodies[0]


@pytest.mark.parametrize("case_id", ["type-01", "search-01"])
def test_a0_body_equals_session_body(tmp_path: Path, case_id: str) -> None:
    """Criterion 2: the harness's A0 prompt equals the pass-1 prompt a ``Session`` builds for the same typed turn
    (same frame, no memory hits, no assists, the ``[screen]`` note on the text-only case), and the real client's body
    for it is byte-equal to the session's body once the request id is the same."""
    h = harness()
    kit = tool_kit()
    row = next(r for r in case_rows() if r["id"] == case_id)
    session_prompt, session_body = _session_prompt(tmp_path, row, kit)
    a0 = h.arm_prompts(harness_case(h, case_id), kit)["A0"]
    assert a0 == session_prompt, f"criterion 2 ({case_id}): the A0 prompt is not the session's"
    rid = session_body["vllm_xargs"]["astra_request_id"]
    mine = VllmClient().request_body(a0, rid)
    assert json.dumps(mine, sort_keys=True) == json.dumps(session_body, sort_keys=True)


# =====================================================================================================
# Criteria 5 and 11: one trial's reading
# =====================================================================================================


def commit(segment: str, text: str, *, open_: bool = False, idx: int = 0) -> dict[str, Any]:
    return {"segment": segment, "text": text, "thought_open": open_, "canvas_idx": idx}


TYPE_CALL = (
    '<|tool_call>call:type_at{press_enter:false,text:<|"|>anna.berg@northwind.example<|"|>,x:500,y:302}'
    "<tool_call|>"
)
PLAN = "The user wants the email typed. I will use `type_at` with `press_enter=false`."


def test_spoken_definition_rows() -> None:
    """Criterion 11's readings, on hand-built commit lists (the 4b0bd92 and dev type-01 shapes):

    1. the plan sentence only inside a closed thought, then the call: routed, **not** spoken (a reading that counts
       thought text is caught here);
    2. ``answer<channel|>`` then the call (dev type-01's stray marker): spoken, prefix ``answer``;
    3. the call inside an unclosed thought, no tool commit: unrouted, thought_open, not spoken;
    4. the same with a ``tool`` commit flagged ``thought_open``: still unrouted (J8: a tool commit *outside* an open
       thought);
    5. whitespace-only answer text ``"\\n\\n"``: routed, not spoken (a reading without ``.strip()`` is caught);
    6. a special literal alone (``<channel|>``): not spoken;
    7. answer text but no call: unrouted, so not spoken (spoken implies routed);
    8. a spoken ``call:`` is a voiced anomaly; a plain sentence is not;
    9. ruling N4: an ``answer`` commit flagged ``thought_open`` is never spoken, on a routed trial too, and only the
       ``thought_open == False`` answer text forms the prefix (an open ``I will type`` followed by a closed
       ``Typing it now.`` reads ``Typing it now.``).
    """
    h = harness()
    rows: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        (
            "plan sentence in a closed thought",
            [commit("thought", PLAN), commit("tool", TYPE_CALL)],
            {"routed": True, "spoken": False, "thought_open": False},
        ),
        (
            "answer<channel|><|tool_call>",
            [
                commit("thought", "OK."),
                commit("answer", "answer<channel|>"),
                commit("tool", TYPE_CALL),
            ],
            {"routed": True, "spoken": True, "spoken_prefix": "answer", "thought_open": False},
        ),
        (
            "call inside an unclosed thought",
            [commit("thought", PLAN, open_=True), commit("thought", TYPE_CALL, open_=True, idx=1)],
            {"routed": False, "spoken": False, "thought_open": True},
        ),
        (
            "tool commit flagged thought_open",
            [commit("thought", PLAN, open_=True), commit("tool", TYPE_CALL, open_=True)],
            {"routed": False, "spoken": False, "thought_open": True},
        ),
        (
            "whitespace-only answer",
            [commit("thought", PLAN), commit("answer", "\n\n"), commit("tool", TYPE_CALL)],
            {"routed": True, "spoken": False},
        ),
        (
            "a special literal alone",
            [commit("thought", PLAN), commit("answer", "<channel|>"), commit("tool", TYPE_CALL)],
            {"routed": True, "spoken": False},
        ),
        (
            # N4: an answer commit flagged thought_open never feeds the spoken prefix (the session's _AnswerCleaner)
            "an answer flagged thought_open, then the call",
            [
                commit("thought", PLAN, open_=True),
                commit("answer", "I will type the email address.", open_=True),
                commit("tool", TYPE_CALL),
            ],
            {"routed": True, "spoken": False},
        ),
        (
            "an open answer, then a closed one",
            [
                commit("thought", "OK."),
                commit("answer", "I will type", open_=True),
                commit("answer", "Typing it now.", idx=1),
                commit("tool", TYPE_CALL, idx=1),
            ],
            {"routed": True, "spoken": True, "spoken_prefix": "Typing it now."},
        ),
        (
            "an answer and no call",
            [commit("thought", "OK."), commit("answer", "I will type it now.")],
            {"routed": False, "spoken": False, "thought_open": False},
        ),
    ]
    for name, commits, want in rows:
        got = h.classify(commits)
        for key, value in want.items():
            assert got[key] == value, f"criterion 11 ({name}): {key} is {got[key]!r}, not {value!r}"
    spoken = h.classify(
        [
            commit("thought", "OK."),
            commit("answer", "I will type the email address."),
            commit("tool", TYPE_CALL),
        ]
    )
    assert spoken["spoken"] is True and spoken["anomaly"] is False
    assert spoken["spoken_prefix"].strip() == "I will type the email address."
    voiced = h.classify(
        [
            commit("thought", "OK."),
            commit("answer", "I will call:type_at now."),
            commit("tool", TYPE_CALL),
        ]
    )
    assert voiced["spoken"] is True and voiced["anomaly"] is True, (
        "criterion 5: a spoken call: is an anomaly"
    )


# =====================================================================================================
# Criteria 6 and 7: the report and the decision rule, on hand-built runs
# =====================================================================================================

SPEAK, SILENT, UNCLOSED, NONE, FALSE, ANOMALY, INVALID, INEXACT, ERROR = (
    "speak",
    "silent",
    "unclosed",
    "none",
    "false",
    "anomaly",
    "invalid",
    "inexact",
    "error",
)


def native_call(name: str, arguments: Mapping[str, Any]) -> str:
    def value(v: Any) -> str:
        return f'<|"|>{v}<|"|>' if isinstance(v, str) else json.dumps(v)

    body = ",".join(f"{k}:{value(v)}" for k, v in arguments.items())
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def commits_for(outcome: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ref = row.get("reference_call")
    call = native_call(ref["name"], ref["arguments"]) if ref else ""
    if outcome == SPEAK:
        return [
            commit("thought", "OK."),
            commit("answer", "I will do that.<channel|>"),
            commit("tool", call),
        ]
    if outcome == SILENT:
        return [commit("thought", PLAN), commit("tool", call)]
    if outcome == UNCLOSED:
        return [commit("thought", PLAN + call, open_=True)]
    if outcome == NONE:
        return [commit("thought", "OK."), commit("answer", "Okay.")]
    if outcome == FALSE:
        wrong = native_call("google_search", {"query": "something"})
        return [
            commit("thought", "OK."),
            commit("answer", "Let me look that up."),
            commit("tool", wrong),
        ]
    if outcome == ANOMALY:
        return [
            commit("thought", "OK."),
            commit("answer", "I will call:it now."),
            commit("tool", call),
        ]
    if (
        outcome == INVALID
    ):  # click_at with an undeclared argument: schema-invalid, still inside the bbox
        assert ref["name"] == "click_at"
        bad = native_call("click_at", {**ref["arguments"], "z": 1})
        return [commit("thought", "OK."), commit("answer", "I will click it."), commit("tool", bad)]
    if outcome == INEXACT:  # type_at with the wrong text: schema-valid, not exact
        assert ref["name"] == "type_at"
        wrong = native_call("type_at", {**ref["arguments"], "text": "zzz wrong text"})
        return [
            commit("thought", "OK."),
            commit("answer", "I will type it."),
            commit("tool", wrong),
        ]
    if outcome == ERROR:
        return []
    raise AssertionError(outcome)


Plan = dict[tuple[str, str, int], str]


def baseline() -> Plan:
    plan: Plan = {}
    for arm in ARMS:
        for row in case_rows():
            for r in range(REPEATS):
                if row["kind"] != "tool":
                    o = NONE
                elif arm == "A0" and row["tool"] == "type_at":
                    o = UNCLOSED if (row["id"], r) == ("type-05", 0) else SILENT
                elif arm == "A1" and row["id"] in {f"type-0{i}" for i in range(1, 7)} and r == 0:
                    o = NONE
                else:
                    o = SPEAK
                plan[(arm, row["id"], r)] = o
    return plan


def type_ids() -> list[str]:
    return [f"type-{i:02d}" for i in range(1, 11)]


def write_run(run_dir: Path, plan: Plan) -> Path:
    texts = expected_arm_texts()
    rows = case_rows()
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in range(REPEATS):
        for i, row in enumerate(rows):
            off = (i + r) % 6
            for slot in range(6):
                arm = ARMS[(off + slot) % 6]
                outcome = plan[(arm, row["id"], r)]
                lines.append(
                    json.dumps(
                        {
                            "case_id": row["id"],
                            "kind": row["kind"],
                            "gt": row.get("gt"),
                            "arm": arm,
                            "repeat": r,
                            "order": slot,
                            "system_sha256": sha256(texts[arm]),
                            "commits": commits_for(outcome, row),
                            "token_ids": [] if outcome == ERROR else [2, 3, 5, 106],
                            "error": "ReadTimeout('mock')" if outcome == ERROR else None,
                            "wall_ms": 640.0,
                            "first_canvas_ms": None if outcome == ERROR else 310.0,
                        }
                    )
                )
    assert len(lines) == 1590
    (run_dir / "trials.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "regime": {
            "profile": "lean",
            "split": "dev",
            "repeats": REPEATS,
            "seed": None,
            "concurrency": 1,
            "tools": "native:bbfc53ccdd80",
        },
        "arms": {a: {"text": t, "sha256": sha256(t)} for a, t in texts.items()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return run_dir


def report_for(tmp_path: Path, edit: Callable[[Plan], None] | None = None) -> dict[str, Any]:
    plan = baseline()
    if edit is not None:
        edit(plan)
    h = harness()
    report = h.build_report(write_run(tmp_path / "run", plan))
    assert isinstance(report, dict)
    return report


def no_change(report: Mapping[str, Any]) -> bool:
    return report["winner"] in (None, "no change")


def cand(report: Mapping[str, Any], c: str) -> Mapping[str, Any]:
    return report["candidates"][c]


def test_baseline_selects_c2(tmp_path: Path) -> None:
    """The baseline (module docstring): R1 and R2 hold, C2 and C1 pass every P, the winner is C2 (C2 is preferred
    over C1 when both pass); A0's ``type_at`` reads 0 spoken of 49 routed, C2's 50 of 50."""
    report = report_for(tmp_path)
    assert report["readings"]["R1"] is True and report["readings"]["R2"] is True
    for c in CANDIDATES:
        failing = [p for p in PS if cand(report, c)[p] is not True]
        assert failing == [] and cand(report, c)["passes"] is True, f"{c} fails {failing}"
    assert report["winner"] == "C2"
    assert list(report["void_arms"]) == []
    assert report["arms"]["A0"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 49}
    assert report["arms"]["C2"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    assert report["arms"]["A1"]["tools"]["type_at"]["spoken"] == {"k": 44, "n": 44}


def _type_speak(plan: Plan, arm: str, spoken: Mapping[str, int], rest: str = SILENT) -> None:
    """``spoken[case]`` of the case's 5 trials speak (repeats 0..k-1); the others are ``rest``."""
    for cid in type_ids():
        for r in range(REPEATS):
            plan[(arm, cid, r)] = SPEAK if r < spoken.get(cid, 0) else rest


@pytest.mark.parametrize(
    ("spoken", "p1"),
    [
        ({cid: 4 for cid in type_ids()}, True),  # 40/50 = 0.80, 10 cases at 4/5
        (
            {**{cid: 4 for cid in type_ids()}, "type-10": 3},
            False,
        ),  # 39/50 = 0.78, 10 cases still a majority
    ],
    ids=["40of50-passes", "39of50-fails"],
)
def test_decision_rule_p1_rate_boundary(tmp_path: Path, spoken: dict[str, int], p1: bool) -> None:
    """P1 at its boundary, both directions: C2's ``type_at`` spoken 40/50 = 0.80 passes, 39/50 = 0.78 fails, with every
    type case speaking in a majority of its routed trials in both rows (3/5 and 4/5), so only the rate moves. Every
    other P holds (the silent trials are routed, valid and exact). Winner: C2 when P1 holds, else C1."""
    report = report_for(tmp_path, lambda plan: _type_speak(plan, "C2", spoken))
    assert sum(spoken.values()) in (40, 39)
    assert cand(report, "C2")["P1"] is p1
    assert [p for p in PS[1:] if cand(report, "C2")[p] is not True] == []
    assert report["arms"]["C2"]["tools"]["type_at"]["spoken"] == {
        "k": sum(spoken.values()),
        "n": 50,
    }
    assert report["winner"] == ("C2" if p1 else "C1")


@pytest.mark.parametrize(
    ("spoken", "p1"),
    [
        # 7 cases at 5/5 = 35, then 2, 2, 1: 40/50 = 0.80, but only 7 cases spoke in a majority -> fails
        (
            {**{f"type-0{i}": 5 for i in range(1, 8)}, "type-08": 2, "type-09": 2, "type-10": 1},
            False,
        ),
        # 8 cases at 5/5 = 40, two at 0: 40/50 and 8 of 10 cases -> passes (the clause's own boundary)
        ({**{f"type-0{i}": 5 for i in range(1, 9)}, "type-09": 0, "type-10": 0}, True),
    ],
    ids=["7-cases-fails", "8-cases-passes"],
)
def test_decision_rule_case_majority(tmp_path: Path, spoken: dict[str, int], p1: bool) -> None:
    """P1's second clause alone: 40/50 spoken (the rate holds in both rows) concentrated in 7 cases fails; in 8 cases
    passes. A clause read as "spoke in any trial" (reviewer mutation: majority -> any) passes the first row (10 cases
    spoke at least once) and is caught here."""
    report = report_for(tmp_path, lambda plan: _type_speak(plan, "C2", spoken))
    assert sum(spoken.values()) == 40
    assert report["arms"]["C2"]["tools"]["type_at"]["spoken"] == {"k": 40, "n": 50}
    assert cand(report, "C2")["P1"] is p1
    assert report["winner"] == ("C2" if p1 else "C1")


def test_decision_rule_spoken_is_over_routed(tmp_path: Path) -> None:
    """P1's denominator is the routed trials (criterion 5, "over its routed trials"). C2 ``type_at``: type-01 and type-02
    answer with no call at repeat 0 (unrouted) and speak at 1-3, silent at 4 (3/4 each); type-03..type-09 speak at
    0-3, silent at 4 (4/5 each); type-10 speaks 5/5. Spoken 6 + 28 + 5 = 39 of 48 routed = 0.8125 >= 0.80: passes,
    10/10 cases in a majority. Over all 50 trials it would read 39/50 = 0.78 and fail (reviewer mutation). P3 holds:
    type recall 48/50 = 0.96 >= 0.98 - 0.04, pooled 198/200 = 0.99 >= 0.995 - 0.02."""

    def edit(plan: Plan) -> None:
        for cid in type_ids():
            for r in range(REPEATS):
                if cid in ("type-01", "type-02"):
                    o = NONE if r == 0 else (SPEAK if r <= 3 else SILENT)
                elif cid == "type-10":
                    o = SPEAK
                else:
                    o = SPEAK if r <= 3 else SILENT
                plan[("C2", cid, r)] = o

    report = report_for(tmp_path, edit)
    assert report["arms"]["C2"]["tools"]["type_at"]["spoken"] == {"k": 39, "n": 48}
    assert cand(report, "C2")["P1"] is True and cand(report, "C2")["P3"] is True
    assert report["winner"] == "C2"


def test_decision_rule_unclosed_calls_fail_p3_and_p7(tmp_path: Path) -> None:
    """Adversarial (criterion 11, the task's named failure mode): C2 moves 6 ``type_at`` calls into unclosed thoughts
    (type-01..type-06, repeat 0) and speaks on every other trial. Spoken reads 44/44 = 1.0 and every case speaks in a
    majority of its routed trials, so P1 holds -- but type recall 44/50 = 0.88 < 0.98 - 0.04 and pooled 194/200 = 0.97 <
    0.995 - 0.02 (P3 fails), and thought_open 6 > A0's 1 + 4 (P7 fails). Winner C1."""

    def edit(plan: Plan) -> None:
        for i in range(1, 7):
            plan[("C2", f"type-0{i}", 0)] = UNCLOSED

    report = report_for(tmp_path, edit)
    c2 = cand(report, "C2")
    assert report["arms"]["C2"]["tools"]["type_at"]["spoken"] == {"k": 44, "n": 44}
    assert c2["P1"] is True
    assert c2["P3"] is False and c2["P7"] is False and c2["passes"] is False
    assert report["winner"] == "C1"


def test_decision_rule_p3_type_clause_against_a0(tmp_path: Path) -> None:
    """P3's ``type_at`` clause alone, read against A0: C2 has 4 ``type_at`` calls in unclosed thoughts. Type recall
    46/50 = 0.92 < A0's 49/50 - 0.04 = 0.94: fails; pooled 196/200 = 0.98 >= 0.995 - 0.02 holds; thought_open 4 <= 1 + 4
    (P7 holds). Against A1 (44/50 - 0.04 = 0.84; pooled 0.97 - 0.02) it would pass -- the reviewer mutation "P3 against
    A1" is caught here. Winner C1."""

    def edit(plan: Plan) -> None:
        for i in range(1, 5):
            plan[("C2", f"type-0{i}", 0)] = UNCLOSED

    report = report_for(tmp_path, edit)
    c2 = cand(report, "C2")
    assert c2["P3"] is False, "P3: type recall 46/50 against A0's 49/50"
    assert c2["P7"] is True and c2["P1"] is True
    assert report["winner"] == "C1"


def test_decision_rule_p7_alone(tmp_path: Path) -> None:
    """P7 alone. A0 gets 4 more unrouted trials that are not thoughts (click-01..click-04 repeat 0 answer with no call):
    A0 recall 195/200 = 0.975, click+scroll+key spoken 96/96, thought_open still 1. C2: 6 calls in unclosed thoughts,
    2 each in type (type-01, type-02), search (search-01, search-02) and click (click-05, click-06), repeat 0. C2
    pooled recall 194/200 = 0.97 >= 0.975 - 0.02 = 0.955; type recall 48/50 = 0.96 >= 0.98 - 0.04 (P3 holds); spoken
    48/48; but thought_open 6 > 1 + 4 = 5: P7 fails. Winner C1."""

    def edit(plan: Plan) -> None:
        for i in range(1, 5):
            plan[("A0", f"click-0{i}", 0)] = NONE
        for cid in ("type-01", "type-02", "search-01", "search-02", "click-05", "click-06"):
            plan[("C2", cid, 0)] = UNCLOSED

    report = report_for(tmp_path, edit)
    c2 = cand(report, "C2")
    assert [p for p in PS if c2[p] is not True] == ["P7"], {p: c2[p] for p in PS}
    assert report["winner"] == "C1"


@pytest.mark.parametrize(("silent", "p2"), [(4, True), (6, False)], ids=["96of100", "94of100"])
def test_decision_rule_p2(tmp_path: Path, silent: int, p2: bool) -> None:
    """P2: C2 click + scroll + key spoken (100 trials, all routed) against A0's 100/100 - 0.05: ``silent`` click trials
    (click-01.., repeat 0) are silent. 96/100 passes, 94/100 fails. Winner C2 or C1."""

    def edit(plan: Plan) -> None:
        for i in range(1, silent + 1):
            plan[("C2", f"click-{i:02d}", 0)] = SILENT

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P2"] is p2
    assert report["winner"] == ("C2" if p2 else "C1")


@pytest.mark.parametrize(("n", "p4"), [(3, True), (5, False)], ids=["197of200", "195of200"])
def test_decision_rule_p4_valid(tmp_path: Path, n: int, p4: bool) -> None:
    """P4, valid: ``n`` click calls carry an undeclared argument (schema-invalid, still inside the bbox, so exact by
    ``score.exact``). Valid 197/200 = 0.985 >= 1 - 0.02 passes; 195/200 = 0.975 fails."""

    def edit(plan: Plan) -> None:
        for i in range(1, n + 1):
            plan[("C2", f"click-{i:02d}", 0)] = INVALID

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P4"] is p4
    assert report["winner"] == ("C2" if p4 else "C1")


@pytest.mark.parametrize(("n", "p4"), [(5, True), (7, False)], ids=["195of200", "193of200"])
def test_decision_rule_p4_exact(tmp_path: Path, n: int, p4: bool) -> None:
    """P4, exact: ``n`` type calls type the wrong text (valid, not exact). Exact 195/200 = 0.975 >= 1 - 0.03 passes;
    193/200 = 0.965 fails. Valid stays 200/200."""

    def edit(plan: Plan) -> None:
        for i in range(1, n + 1):
            plan[("C2", f"type-{i:02d}", 0)] = INEXACT

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P4"] is p4
    assert report["winner"] == ("C2" if p4 else "C1")


@pytest.mark.parametrize(
    ("a0_false", "c2_false", "p5"),
    [
        (0, 2, True),  # 2 <= 5 and 2 <= 0 + 2
        (0, 3, False),  # 3 <= 5 but 3 > 0 + 2
        (4, 6, False),  # 6 <= 4 + 2 but 6 > 5
        (4, 5, True),  # 5 <= 5 and 5 <= 6
    ],
)
def test_decision_rule_p5(tmp_path: Path, a0_false: int, c2_false: int, p5: bool) -> None:
    """P5, both clauses in both directions: false tools on the 65 no-tool + plain trials, C2 against A0. The false
    tools fall on chat-01.. then plain-01.. at repeat 0 (plain trials count: "routed on a no-tool or plain trial")."""
    order = [f"chat-0{i}" for i in range(1, 9)] + [cid for cid, _ in PLAIN_QUESTIONS]

    def edit(plan: Plan) -> None:
        for cid in order[:a0_false]:
            plan[("A0", cid, 0)] = FALSE
        for cid in list(reversed(order))[:c2_false]:
            plan[("C2", cid, 0)] = FALSE

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P5"] is p5
    assert report["winner"] == ("C2" if p5 else "C1")


def test_decision_rule_p6(tmp_path: Path) -> None:
    """P6: one C2 trial speaks ``I will call:it now.`` before its call -- a voiced anomaly; P6 fails, winner C1."""

    def edit(plan: Plan) -> None:
        plan[("C2", "type-03", 2)] = ANOMALY

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P6"] is False
    assert report["winner"] == "C1"


def test_decision_rule_neither_passes_and_diagnostics_never_win(tmp_path: Path) -> None:
    """Winner order and N2: C2 and C1 both silent on every ``type_at`` trial (P1 0/50) while D1 and D2 are perfect: the
    winner is "no change", never a diagnostic arm. And when C2 fails while C1 passes, C1 wins (covered above)."""

    def edit(plan: Plan) -> None:
        _type_speak(plan, "C2", {})
        _type_speak(plan, "C1", {})

    report = report_for(tmp_path, edit)
    assert cand(report, "C2")["P1"] is False and cand(report, "C1")["P1"] is False
    assert no_change(report), f"winner {report['winner']!r}"
    assert "no change" in harness().render_markdown(report)


@pytest.mark.parametrize(
    ("spoken", "r1"), [(14, True), (15, False)], ids=["14of49-holds", "15of49-fails"]
)
def test_r1_failing_selects_nothing(tmp_path: Path, spoken: int, r1: bool) -> None:
    """R1 and N3: A0 ``type_at`` speaks on ``spoken`` of its 49 routed trials (type-01, type-02, then type-03 from
    repeat 0; type-05 repeat 0 stays unclosed). 14/49 = 0.286 <= 0.3 holds and C2 wins; 15/49 = 0.306 fails: no
    candidate is selected though C2 passes every P, and report.md reads "finding not reproduced". (Over all 50 trials
    15/50 = 0.30 would hold: the denominator is the routed trials.)"""

    def edit(plan: Plan) -> None:
        left = spoken
        for cid in ("type-01", "type-02", "type-03", "type-04"):
            for r in range(REPEATS):
                if left > 0:
                    plan[("A0", cid, r)] = SPEAK
                    left -= 1

    report = report_for(tmp_path, edit)
    assert report["arms"]["A0"]["tools"]["type_at"]["spoken"] == {"k": spoken, "n": 49}
    assert report["readings"]["R1"] is r1
    assert cand(report, "C2")["passes"] is True
    md = harness().render_markdown(report)
    if r1:
        assert report["winner"] == "C2"
        assert "finding not reproduced" not in md
    else:
        assert no_change(report), f"N3: R1 failed but {report['winner']!r} was selected"
        assert "finding not reproduced" in md


@pytest.mark.parametrize(
    ("spoken", "r2"), [(31, True), (30, False)], ids=["31of44-holds", "30of44-fails"]
)
def test_r2_failing_still_selects(tmp_path: Path, spoken: int, r2: bool) -> None:
    """R2: A1 ``type_at`` speaks on ``spoken`` of its 44 routed trials, silent on the rest (the 6 unrouted stay). 31/44
    = 0.705 >= 0.7 holds; 30/44 = 0.682 fails, report.md carries "eval baseline not reproduced", and selection
    proceeds against A0: the winner is still C2."""

    def edit(plan: Plan) -> None:
        left = spoken
        for cid in type_ids():
            for r in range(REPEATS):
                if plan[("A1", cid, r)] == NONE:
                    continue
                plan[("A1", cid, r)] = SPEAK if left > 0 else SILENT
                left -= 1

    report = report_for(tmp_path, edit)
    assert report["arms"]["A1"]["tools"]["type_at"]["spoken"] == {"k": spoken, "n": 44}
    assert report["readings"]["R2"] is r2
    assert report["winner"] == "C2"
    md = harness().render_markdown(report)
    assert ("eval baseline not reproduced" in md) is (not r2)


@pytest.mark.parametrize(("errors", "void"), [(5, False), (6, True)], ids=["5of265", "6of265"])
def test_errored_arm_voids(tmp_path: Path, errors: int, void: bool) -> None:
    """Criterion 5: an arm with more than 2 % errored trials is void. C2 has 265 trials; ``errors`` of them error (on
    plain trials, so no rate of a P moves). 5/265 = 1.9 % is not void and C2 wins; 6/265 = 2.3 % voids C2, which then
    cannot be selected: the winner is C1."""
    plain = [cid for cid, _ in PLAIN_QUESTIONS]
    slots = [(cid, r) for r in range(REPEATS) for cid in plain][:errors]

    def edit(plan: Plan) -> None:
        for cid, r in slots:
            plan[("C2", cid, r)] = ERROR

    report = report_for(tmp_path, edit)
    assert ("C2" in report["void_arms"]) is void
    assert "C1" not in report["void_arms"] and "A0" not in report["void_arms"]
    assert report["winner"] == ("C1" if void else "C2")


# =====================================================================================================
# T-0155 criterion 8: the frozen harness refuses today's persona
# =====================================================================================================


def test_harness_refuses_live_persona(
    pre_t0155: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T-0155 criterion 8: with the pre-T-0155 patch undone, the T-0098 table's ``main`` sees the live Gearvis A0
    (sha ``869f8f96...``, the generator of T-0155 note 3), exits 2 with its A0-drift refusal naming both shas, and
    opens no socket and writes no run directory. The wording is the harness's own (``main``'s A0 check)."""
    from tests.unit.test_persona_name import A0_T0155_SHA256

    out = tmp_path / "t0098-out"
    _, rc, said, attempts = run_main_against_live_persona(
        pre_t0155, harness, lambda h: h._T93, ["--table", "T-0098"], out, monkeypatch, capsys
    )
    assert rc == 2, f"T-0155 criterion 8: exit {rc}: {said[-400:]!r}"
    assert attempts == [], f"T-0155 criterion 8: sockets before the refusal: {attempts}"
    assert not out.exists(), "T-0155 criterion 8: the refusal wrote a run directory"
    assert f"the A0 system text has sha256 {A0_T0155_SHA256}, not {PRE_T0155_A0_SHA256}" in said, (
        f"T-0155 criterion 8: not the A0-drift refusal: {said[-400:]!r}"
    )
