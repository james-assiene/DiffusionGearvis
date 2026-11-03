"""The human's own recorded turn, diagnosable from the log, the record and the pane.

T-0054 criteria 19-21 (joint). The failure this module exists for is ruling A33 of T-0052, in
the human's words: *"the turn speaks its prose, the interface data is silently dropped, and no
log line records that a refusal happened."* T-0052 fixed the first half — the JSON no longer
reaches the speakers — and left the second half open on purpose: the count lives on the
per-turn ``SpeechGate`` and nothing reads it. Half of "undiagnosable" therefore survives until
this task lands, and these rows are what closes it.

Three real components per row and no hand-copied expectation:

* the **real ``VllmClient``** over an ``httpx.MockTransport`` SSE stub replays
  ``tests/fixtures/llm/flight_canvases.json`` — turn ``t6`` of the desk session of 2026-09-16,
  reconstructed by T-0052 from ``logs/sessions/20260916-181929-3900425.jsonl``. It is the oracle
  for what a commit's fence lifecycle *is*: no row writes ``"inferred"`` because a session log
  once said so;
* a **real ``Session``** over ``tests/fakes/`` turns those commits into a session log, a
  ``turns.jsonl`` line and a wire trace;
* the **real ``hud/src/ui/fence.js``** in ``node`` decides what the pane says about the message
  that reached the wire, so the last hop of criterion 19(d)/20 is the page's own code and not a
  Python restatement of it.

"Fake backend" here means no engine produced the text; the recorded canvases stand in for the
model, exactly as ``tests/contract/test_fence_reassembly.py`` (T-0052) does. There is no
``backend=real`` arm: T-0054's regime section says so — the live arm is T-0052's criterion 16
eval under ``/launch``, and it is not owed here.

Regime: CPU only, n = 1 per row, no seed, no timing claimed, ``ScriptedClock``, ``tests/fakes/``,
node v18.19.1 for the page half. Profile lean, canvas length 256, 48 steps, thinking on — the
fixture's own recorded regime, which is what makes the numbers below comparable to that night's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.llm_client import UI_OPEN, VllmClient
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from tests.fakes.llm import flight_canvases, sse_transport
from tests.unit._hud_rig import Recorder, answer, frame, running, thought
from tests.unit._session_log_rig import (
    SPEECH_REFUSED,
    fence_messages,
    make_log,
    of_kind,
    read_records,
)

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
FENCE_JS = REPO / "hud" / "src" / "ui" / "fence.js"
NODE = shutil.which("node")

FIX = flight_canvases()
REC = FIX["recorded"]
FLIGHT: list[tuple[str | None, str | None]] = [
    (c["reasoning"], c["content"]) for c in FIX["canvases"]
]
PROMPT = Prompt(
    messages=(Message(role="user", content=(TextPart(text="build me a flight booking ui"),)),),
    prefix_len=1,
)

#: The question the turn asks. Short and word-separable: the rig delivers one STT word at a time.
QUESTION = "build me a flight booking interface"


async def commits_of(canvases: list[tuple[str | None, str | None]], *, rid: str) -> list[Commit]:
    """The real client's own commits for a recorded stream — the oracle for every fence value."""
    client = VllmClient(transport=sse_transport(canvases))
    return [c async for c in client.generate(PROMPT, rid, None)]


def commit_records(records: list[dict[str, Any]], *, segment: str) -> list[dict[str, Any]]:
    return [
        r for r in of_kind(records, "llm.commit") if r.get("data", {}).get("segment") == segment
    ]


def describe_fence(message: Any) -> dict[str, Any]:
    """``describeFence`` from ``hud/src/ui/fence.js``, run in node over one decoded message.

    The page's half of criteria 19(d) and 20 is JavaScript, so this is where the two halves of
    the task meet: the message is dumped exactly as ``contracts/codec.encode`` would put it on
    the wire and handed to the module the page itself imports. A Python re-statement of the
    status table would prove nothing about what a human at the desk sees.
    """
    assert NODE is not None, (
        "node >= 18.19 is a dev prerequisite (docs/tasks/T-0008.md Regime); the page half of "
        "T-0054 criteria 19-20 cannot be checked without it"
    )
    assert FENCE_JS.is_file(), (
        "T-0054 criterion 14 adds hud/src/ui/fence.js exporting "
        "describeFence(message) -> { status, label, canvases, refused } — DOM-free, clock-free, "
        "and a pure function of one decoded message (docs/tasks/T-0054.md, Owned files, group B)"
    )
    payload = message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message)
    script = (
        f"import {{ describeFence }} from {json.dumps(FENCE_JS.as_uri())};\n"
        f"const out = describeFence({json.dumps(payload)});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        "hud/src/ui/fence.js must export describeFence(message) and must not throw on a message "
        f"the server really published.\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n"
        f"{proc.stderr}"
    )
    return dict(json.loads(proc.stdout))


# --- criterion 19: the fixed path -----------------------------------------------------------


async def test_the_flight_turn_is_reassembled_and_the_record_says_so(tmp_path) -> None:
    """Criterion 19: the log, the turn line, the wire and the pane all say "reassembled".

    Every expectation comes from the real client's own commits, computed in this test: canvas 1
    opens the fence (409 characters, ``fence_close="open"``) and canvas 2 closes it structurally
    (``fence_close="inferred"``, ``fence_continued=True``, 1159 characters — the fenced body the
    model never wrote a ``</a2ui>`` for). Nothing below is transcribed from a session log.
    """
    commits = await commits_of(FLIGHT, rid="t6-gen")
    ui_commits = [c for c in commits if c.segment == "ui"]
    assert len(ui_commits) == 2, (
        "setup: T-0052's carry gives this turn two ui commits, one per canvas that carried the "
        f"fence; got {[(c.canvas_idx, c.fence_close) for c in commits]}"
    )
    last_ui = ui_commits[-1]
    assert (last_ui.canvas_idx, last_ui.fence_close, last_ui.fence_continued) == (
        2,
        "inferred",
        True,
    ), (
        "setup: the oracle is the real client and it says canvas 2 closed the carried fence "
        f"structurally; it said {(last_ui.canvas_idx, last_ui.fence_close, last_ui.fence_continued)}"
    )
    expected_canvases = len({c.canvas_idx for c in ui_commits})

    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=commits,
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()
        record = rig.records()[-1]

    written = read_records(Path(log.path))

    # (a) the llm.commit record for canvas 2 carries the lifecycle, taken from the Commit.
    canvas2 = [r for r in commit_records(written, segment="ui") if r["data"]["canvas_idx"] == 2]
    assert canvas2, (
        "criterion 1/19(a): no llm.commit record for canvas 2's ui commit. "
        f"ui commit records: {[r['data'].get('canvas_idx') for r in commit_records(written, segment='ui')]}"
    )
    assert canvas2[-1]["data"].get("fence_close") == last_ui.fence_close, (
        "criterion 19(a): the record must carry the Commit's own fence_close ('inferred' — the "
        "client closed the fence at the structural end of the JSON because the model never wrote "
        f"the marker). The record says {canvas2[-1]['data'].get('fence_close')!r}. "
        "session.py copies commit fields one by one, which is where the two T-0052 added stop."
    )
    assert canvas2[-1]["data"].get("fence_continued") is True, (
        "criterion 19(a): fence_continued says this canvas continued a fence opened in an "
        "earlier one — the single fact that distinguishes a reassembled surface from a fresh "
        f"one. The record says {canvas2[-1]['data'].get('fence_continued')!r}"
    )

    # (b) the turn line answers "was this surface reassembled?" on its own.
    assert record.extras.get("ui_canvases") == expected_canvases == 2, (
        "criterion 19(b): the turn line counts the canvases that produced a ui commit; the real "
        f"client produced {expected_canvases} and the line says "
        f"{record.extras.get('ui_canvases')!r} (extras: {sorted(record.extras)})"
    )
    assert record.extras.get("ui_fence") == "inferred", (
        "criterion 19(b): ui_fence is the FenceClose of the *last* ui commit of the turn; the "
        f"line says {record.extras.get('ui_fence')!r}"
    )

    # (c) exactly one fence message on the wire carries the final tuple.
    fences = fence_messages(recorder.messages)
    assert fences, (
        "criterion 19(c): nothing on the wire tells the page this surface arrived in pieces. "
        f"The session published {sorted(set(recorder.types()))}"
    )
    final = [m for m in fences if (m.close, m.canvases, m.continued) == ("inferred", 2, True)]
    assert len(final) == 1, (
        "criterion 19(c): exactly one astra.ui.fence carries (close='inferred', canvases=2, "
        f"continued=True); the wire holds {[(m.close, m.canvases, m.continued) for m in fences]}"
    )
    assert fences[-1] is final[0], (
        "criterion 9: the last fence message of the turn is the final tuple — a message "
        f"published after it would be stale. Wire: {[(m.close, m.canvases) for m in fences]}"
    )
    assert final[0].speech_refused == 0, (
        "the fixed path speaks only the 152 characters of prose, so nothing is refused; the "
        f"message claims {final[0].speech_refused}"
    )

    # (d) the pane, decided by the page's own module.
    described = describe_fence(final[0])
    assert described.get("status") == "reassembled", (
        "criterion 19(d)/15: close in {written, inferred} with continued=true reads "
        f"'reassembled'; hud/src/ui/fence.js returned {described!r}"
    )
    assert "2" in str(described.get("label", "")), (
        "criterion 19(d): the label names how many canvases the surface was reassembled from "
        f"('reassembled from 2 canvases'); it says {described.get('label')!r}"
    )
    assert described.get("canvases") == 2 and described.get("refused") == 0, (
        f"criterion 14: describeFence echoes the message's own numbers; it returned {described!r}"
    )


# --- criterion 20: the pre-fix path, the row that closes ruling A33 ---------------------------


async def test_the_pre_fix_turn_leaves_a_refusal_record_and_a_refused_only_pane(tmp_path) -> None:
    """Criterion 20 — **the single most important row in this task**.

    The same fixture driven through the path T-0052 group A no longer takes: canvas 2's 902
    characters arrive as one ``answer`` commit and no ``ui`` commit is produced at all, which is
    the world the fence carry regressing puts the turn back into. That is the state the human
    reported — the prose is spoken, the interface data is dropped, and *nothing anywhere says a
    refusal happened*. After this task three facts are written down, and this row is the three.

    Three oracles, none of them a session log:

    * ``sources[0]`` is compared against the fixture's recorded ``tts_utterance_source`` — the
      810-character substring whose TTS rendering is the 807 characters Pocket TTS actually said
      that night. ``SpeechGate.refused_sources`` retains the committed *source*, so that is the
      string the record must carry byte-exact; the 807-character rendering is asserted separately
      never to reach TTS, which is the fixture's own negative oracle ("no test may ever allow
      that string to be spoken");
    * ``count`` is compared against an independent ``SpeechGate`` fed the same text — the gate is
      not this task's file and is used here as an oracle, never re-implemented;
    * the pane's status comes from ``hud/src/ui/fence.js`` in node.

    Criterion 20's own words: "a regression that deletes the tap turns this row red and nothing
    else".
    """
    from local_astra.conductor.speech_gate import SpeechGate

    text = REC["answer_piece_canvas2"]

    # The independent oracle: what a gate outside this session does with the same bytes.
    oracle = SpeechGate()
    oracle_sentences = [
        *oracle.feed(Commit(request_id="o", canvas_idx=2, segment="answer", text=text))
    ]
    oracle_sentences += [*oracle.close()]
    assert oracle.refused_count == 1 and len(oracle.refused_sources) == 1, (
        "setup: the voice guard refuses exactly one sentence of this canvas (T-0052 criterion 8). "
        f"It refused {oracle.refused_count}"
    )
    assert oracle.refused_sources[0] == REC["tts_utterance_source"], (
        "setup: the gate's retained source is the 810-character substring the fixture recorded "
        "as what Pocket TTS was handed; if this drifts the fixture is no longer that turn"
    )

    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=[thought(FIX["canvases"][0]["reasoning"], canvas_idx=0), answer(text, canvas_idx=2)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()
        record = rig.records()[-1]
        spoken = list(rig.tts.texts())

    written = read_records(Path(log.path))

    # Fact 1 — a log line records that a refusal happened, with its evidence.
    refusals = of_kind(written, SPEECH_REFUSED)
    assert refusals, (
        "criterion 20 / T-0052 ruling A33: 'no log line records that a refusal happened'. The "
        "guard fired on this very turn and the session log holds "
        f"{sorted(set(r['kind'] for r in written))}. SpeechGate.refused_count is read by nobody, "
        "and the gate is dropped at IDLE, so the only copy of the number goes with it."
    )
    last = refusals[-1]["data"]
    assert last.get("count") == oracle.refused_count == 1, (
        f"criterion 5: the record carries the gate's running count; it says {last.get('count')!r}"
    )
    assert last.get("truncated") is False, (
        "criterion 5: truncated is true only past MAX_REFUSED_SOURCES = 8; one refusal is not "
        f"truncated. The record says {last.get('truncated')!r}"
    )
    sources = last.get("sources")
    assert isinstance(sources, list) and sources, (
        f"criterion 5: sources is the gate's retained list; the record says {sources!r}"
    )
    assert sources[0] == REC["tts_utterance_source"], (
        "criterion 5/20: sources[0] must be the refused source **byte-exact** — it is the only "
        "way a human tells a true refusal from a false positive (T-0054 'What stays invisible' "
        f"item 4). Expected {len(REC['tts_utterance_source'])} characters starting "
        f"{REC['tts_utterance_source'][:40]!r}; got {len(sources[0])} starting {sources[0][:40]!r}"
    )

    # The fixture's negative oracle: the 807 characters Pocket TTS said that night, never again.
    assert REC["tts_utterance_text"] not in "\n".join(spoken), (
        "the 807 characters of A2UI JSON the human heard on 2026-09-16 were handed to TTS again. "
        "T-0052's guard is what stops that; this row is what proves the guard was not disabled "
        f"to make the refusal record easy. TTS was asked for {[t[:60] for t in spoken]}"
    )

    # Fact 2 — the turn line says it, so a bench reader never has to open the log.
    assert record.extras.get("speech_refused") == 1, (
        "criterion 3/20: the turn line carries speech_refused. It says "
        f"{record.extras.get('speech_refused')!r} (extras: {sorted(record.extras)}). Reading "
        "SpeechGate.refused_count at _write_record time reads from an object _sync_gate already "
        "set to None — accumulate onto _TurnLog as the count changes (criterion 7)."
    )
    assert record.extras.get("ui_canvases") == 0 and record.extras.get("ui_fence") is None, (
        "criterion 20: in the regression world this turn produces no ui commit at all, so the "
        "line must read 'never existed' rather than 'ended cleanly'; it says "
        f"({record.extras.get('ui_canvases')!r}, {record.extras.get('ui_fence')!r})"
    )

    # Fact 3 — the page says it out loud, so a human at the desk never opens a log file.
    fences = fence_messages(recorder.messages)
    assert len(fences) == 1, (
        "criterion 9/20: exactly one astra.ui.fence reaches the wire for this turn — the tuple "
        f"changes once (a refusal) and is published once. Got {len(fences)}: "
        f"{[(m.close, m.canvases, m.speech_refused) for m in fences]}"
    )
    message = fences[0]
    assert (message.close, message.speech_refused) == (None, 1), (
        "criterion 20: the message says close=null (nothing was shown) and speech_refused=1; it "
        f"says {(message.close, message.speech_refused)}"
    )
    assert message.turn_id == record.turn_id, (
        f"criterion 8: the message names its turn; it says {message.turn_id!r} for {record.turn_id!r}"
    )

    described = describe_fence(message)
    assert described.get("status") == "refused-only", (
        "criterion 15/20: close=null with speech_refused>0 is the row that reads 'part of the "
        "reply was interface data and was not spoken; nothing was shown'. hud/src/ui/fence.js "
        f"returned {described!r}"
    )
    assert described.get("refused") == 1, (
        f"criterion 14: describeFence echoes the count it was sent; it returned {described!r}"
    )
    assert "1" in str(described.get("label", "")), (
        "criterion 15: the label names how many pieces of interface data were not spoken; it "
        f"says {described.get('label')!r}"
    )


async def test_the_pre_fix_turn_with_its_half_surface_still_records_the_refusal(tmp_path) -> None:
    """The same regression with canvas 1's ``ui`` commit still there — the half surface the human
    actually saw.

    Criterion 20 pins the wire shape ``close: null`` and the pane ``refused-only``, which is only
    reachable when the turn produced **no** ui commit at all. The night of 2026-09-16 the human
    saw half an interface *and* heard the JSON, so this is the same regression one notch less
    severe, and the two facts that must survive it are the ones a diagnosis needs: the refusal is
    still recorded, and the pane no longer claims completeness. It is here rather than in the row
    above because a panel that answered ``refused-only`` to a turn that *did* show something
    would be lying in the other direction.
    """
    opening = await commits_of(FLIGHT[:2], rid="t6-gen")
    ui_commits = [c for c in opening if c.segment == "ui"]
    assert len(ui_commits) == 1 and ui_commits[0].canvas_idx == 1, (
        f"setup: canvas 1 alone opens the fence and publishes 409 characters; got "
        f"{[(c.canvas_idx, c.segment, c.fence_close) for c in opening]}"
    )

    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=[*opening, answer(REC["answer_piece_canvas2"], canvas_idx=2)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()
        record = rig.records()[-1]

    written = read_records(Path(log.path))
    assert of_kind(written, SPEECH_REFUSED), (
        "criterion 5: the refusal is recorded whether or not the turn also showed something; "
        f"the log holds {sorted(set(r['kind'] for r in written))}"
    )
    assert record.extras.get("speech_refused") == 1, (
        f"criterion 3: the turn line says so too; extras={record.extras}"
    )
    assert record.extras.get("ui_fence") == ui_commits[0].fence_close == "open", (
        "criterion 3: the last ui commit of this turn left its fence open — the canvas ended "
        f"inside it — so the line must say 'open'; it says {record.extras.get('ui_fence')!r}"
    )

    fences = fence_messages(recorder.messages)
    assert fences, "criterion 9: the turn showed a surface and refused a sentence and said neither"
    last = fences[-1]
    assert (last.close, last.speech_refused) == ("open", 1), (
        f"criterion 9: the final tuple carries both facts; it carries {(last.close, last.speech_refused)}"
    )
    described = describe_fence(last)
    assert described.get("status") == "incomplete", (
        "criterion 15: close='open' reads 'the interface was cut off — showing what arrived'. "
        f"It must not read 'refused-only' (something *was* shown) and must not read 'whole'. "
        f"hud/src/ui/fence.js returned {described!r}"
    )
    assert described.get("refused") == 1, (
        "criterion 15's last row: a refusal is reported next to the surface whatever the close "
        f"value is; describeFence returned {described!r}"
    )


# --- adversarial ------------------------------------------------------------------------------


async def test_a_surface_abandoned_at_the_bound_is_not_reported_as_reassembled(tmp_path) -> None:
    """An abandoned carry must never read as a success.

    A panel that maps every non-null ``close`` to "reassembled" passes criterion 19 and fails
    here; so does a turn record that reports the *first* fence value it saw rather than the last.
    The commits come from the real client driven past ``UI_FENCE_MAX_CANVASES`` (T-0052 D3), so
    "abandoned" is the client's own verdict and not a literal written here.
    """
    junk = '{"id":"ghost","component":"Column","children":["x' + "y" * 40 + '","kids":['
    endless: list[tuple[str | None, str | None]] = [(None, UI_OPEN + junk)]
    endless += [(None, junk) for _ in range(9)]
    endless += [(None, "And that is the table.")]

    commits = await commits_of(endless, rid="t1-gen")
    ui_commits = [c for c in commits if c.segment == "ui"]
    assert ui_commits and ui_commits[-1].fence_close == "abandoned", (
        "setup: the carry must really be abandoned at the bound before this row means anything; "
        f"the client said {[(c.canvas_idx, c.fence_close) for c in ui_commits]}"
    )
    expected_canvases = len({c.canvas_idx for c in ui_commits})

    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=commits,
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
    ) as rig:
        await rig.ask("draw me an endless table")
        await rig.speak_out()
        record = rig.records()[-1]

    assert record.extras.get("ui_fence") == "abandoned", (
        "criterion 4: the turn line reports the *last* ui commit's fence, and the last one was "
        f"abandoned. It says {record.extras.get('ui_fence')!r} — a writer that keeps the first "
        "value it saw reports 'open' here and calls a lost surface a live one."
    )
    assert record.extras.get("ui_canvases") == expected_canvases >= 1, (
        f"criterion 4: {expected_canvases} canvases produced a ui commit; the line says "
        f"{record.extras.get('ui_canvases')!r}"
    )

    fences = fence_messages(recorder.messages)
    assert fences, (
        "criterion 9: a surface abandoned at the bound is exactly what the page needs told"
    )
    described = describe_fence(fences[-1])
    assert described.get("status") == "abandoned", (
        "criterion 15: close='abandoned' reads 'the interface never finished — what you see is "
        f"incomplete'. hud/src/ui/fence.js returned {described!r}"
    )
    assert described.get("status") not in {"whole", "reassembled"}, (
        "the adversarial point, stated as its own assertion: an abandoned carry must not be "
        "reported as a success. A describeFence that maps every non-null close to 'reassembled' "
        f"passes criterion 19 and must fail here. It returned {described!r}"
    )
