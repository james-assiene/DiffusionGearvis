"""T-0039 group B: typing a question into the real page and reading the answer.

The row set of ``docs/tasks/T-0039.md`` §4 "Group B", last bullet: the real ``hud/`` page in
headless Chrome against the real ``HudServer`` and the real ``Session`` over the recorders of
``tests/unit/_hud_rig.py``. The harness is ``tests/integration/test_hud_rehearsal.py``'s, imported
rather than copied, so the two modules cannot drift apart; what is new here is the input -- a
typed question over ``astra.text`` instead of the STT events of ``Live.ask``.

**Regime** (docs/tasks/T-0039.md §6): fakes and recorders from ``tests/unit/_hud_rig.py``, real
``Session`` on ``SessionClock`` (wall time -- the browser has a clock of its own and the two must
agree), real ``HudServer(port=0)``, the page served over loopback http with ``?source=canvas`` and
``?ws=<server url>``, headless Chrome, ``tts ms_per_word`` as each row sets it, n=1 per row. **No
number here is a latency claim**: every wait is a generous wall-clock ceiling, because a browser
cannot be single-stepped, and the ceilings are deadlines rather than measurements.

**Where the assertions come from.** The DOM attribute contract
(``[data-turn-state]``, ``[data-transcript]``, ``[data-utterance-id]``, and T-0039's own
``[data-ask-form]``/``[data-ask-input]``), what ``HudServer.publish`` fanned out, what the
``RecordingMachine`` was fed, and the ``turns.jsonl`` line. Nothing reads a private attribute of
``local_astra``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import shutil
import socketserver
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import AstraSpoken, AstraTranscript, AstraTurnState
from local_astra.contracts.turn import TurnState
from tests.integration.test_hud_headless import REPO, requires_chrome, serve_hud
from tests.integration.test_hud_rehearsal import (
    PAGE_TIMEOUT_S,
    TURN_TIMEOUT_S,
    Live,
    answer,
    open_page,
    page_url,
    thought,
    thought_dom,
    thought_leaks,
    thought_texts,
)

pytestmark = [pytest.mark.integration]

#: The question the human types. Distinctive, so "it appears in the transcript" is a substring
#: search rather than a statement about a field name.
QUESTION = "how many eighths are in three quarters"

#: The scripted answer. Four words at 400 ms/word is ~1.6 s of speech: long enough that the page
#: can be caught with the control disabled, short enough for a 30 s ceiling.
ANSWER = "There are six eighths. "

#: A thought whose text exists nowhere else (criterion 21): it must appear in no DOM, no
#: server->client message and no ``turns.jsonl`` line, on the text path exactly as on the spoken
#: one (CLAUDE.md: never store assistant thinking in history; the rehearsal's scan, extended).
SECRET_THOUGHT = "SECRETTHOUGHT the user is confusing quarters with eighths"


def script() -> list[Commit]:
    return [thought(SECRET_THOUGHT), answer(ANSWER)]


#: Types the question into the box and submits it the way Enter does: ``requestSubmit()`` raises
#: the form's own ``submit`` event, which is the event criterion 12 says must be prevented. The
#: probe on ``window`` is wiped by a navigation, so a form that is *not* prevented is caught here
#: as well as in ``tests/hud/ask.test.mjs``.
def type_and_submit_js(question: str) -> str:
    return """
(() => {
  window.__astra_no_reload = 1;
  const input = document.querySelector('[data-ask-input]');
  const form = document.querySelector('[data-ask-form]');
  if (!input) return 'no [data-ask-input] on the page';
  if (!form) return 'no [data-ask-form] on the page';
  if (input.disabled) return 'the input is disabled';
  input.focus();
  input.value = __QUESTION__;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  if (typeof form.requestSubmit === 'function') form.requestSubmit();
  else form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  return 'submitted';
})()""".replace("__QUESTION__", json.dumps(question))


ASK_STATE_JS = """
(() => {
  const input = document.querySelector('[data-ask-input]');
  return {
    present: Boolean(input),
    disabled: input ? Boolean(input.disabled) : null,
    value: input ? input.value : null,
    survived: window.__astra_no_reload === 1,
  };
})()"""


@requires_chrome
async def test_a_typed_question_opens_a_turn_and_the_answer_comes_back(tmp_path: Path):
    """Rows (a)-(d) and (f) of docs/tasks/T-0039.md §4: the whole path, in one turn.

    (a) the typed question is in ``[data-transcript]``, verbatim and with no ``[ui]`` prefix;
    (b) the scripted answer is in the spoken pane with its ``[data-utterance-id]``;
    (c) the chip walks ``IDLE -> ... -> SPEAKING -> IDLE``;
    (d) the input is disabled while the turn runs and enabled again at ``IDLE``;
    (f) ``turns.jsonl`` has exactly one line and it says ``input="text"``.
    """
    live = Live(tmp_path, script(), ms_per_word=400)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            before = await page.evaluate(ASK_STATE_JS)
            assert before["present"], (
                "criterion 10: hud/index.html gains a text box the panel mounts into, addressed "
                "as [data-ask-input]; the page has none"
            )
            assert before["disabled"] is False, (
                "criterion 13: the page is IDLE, so the box takes a question"
            )

            outcome = await page.evaluate(type_and_submit_js(QUESTION))
            assert outcome == "submitted", f"the page could not be typed into: {outcome!r}"

            # (d) while the turn runs the control is disabled -- caught during SPEAKING, which
            # lasts ~1.6 s of scripted speech.
            await page.until(
                "document.querySelector('[data-turn-state]')"
                ".getAttribute('data-turn-state') === 'SPEAKING'",
                "the chip reaches SPEAKING",
                timeout_s=TURN_TIMEOUT_S,
            )
            speaking = await page.evaluate(ASK_STATE_JS)
            assert speaking["disabled"] is True, (
                "criterion 13: the panel disables the input while the last astra.turn.state is "
                "not IDLE"
            )
            assert speaking["value"] == "", (
                f"criterion 12: submitting clears the input; it still reads {speaking['value']!r}"
            )
            assert speaking["survived"] is True, (
                "criterion 12 adversarial: the form's default was not prevented -- the page "
                "navigated, which drops the socket and kills the turn the user just started"
            )

            await live.idle()
            await page.until(
                "document.querySelector('[data-turn-state]')"
                ".getAttribute('data-turn-state') === 'IDLE'",
                "the chip comes back to IDLE",
                timeout_s=TURN_TIMEOUT_S,
            )
            after = await page.evaluate(ASK_STATE_JS)
            assert after["disabled"] is False, "criterion 13: IDLE re-enables the control"

            probe = await page.probe()

        # (a) the transcript pane shows what was typed, verbatim
        assert QUESTION in (probe["transcript"] or ""), (
            f"criterion 14: the typed question appears in [data-transcript], produced by the "
            f"existing reducer and renderer (hud/src/state.js is not modified). The pane reads "
            f"{probe['transcript']!r}"
        )
        assert "[ui]" not in (probe["transcript"] or ""), (
            "criterion 5: the turn's transcript is the typed text verbatim -- the [ui] prefix "
            "belongs to the click path"
        )
        # (b) the spoken pane shows the answer, with its utterance id
        spoken_texts = " | ".join(u["text"] for u in probe["spoken"])
        assert ANSWER.strip() in spoken_texts, (
            f"criterion 8/14: the answer reaches the spoken pane exactly as a spoken turn's does; "
            f"the pane holds {spoken_texts!r}"
        )
        assert all(u["utterance_id"] for u in probe["spoken"]), (
            "criterion 8: each utterance carries its [data-utterance-id]"
        )

        # (c) the states the server published, in order
        states = [m.state.name for m in live.wire if isinstance(m, AstraTurnState)]
        assert "THINKING" in states and "SPEAKING" in states, (
            f"criterion 5: a typed question drives the ordinary rows IDLE -> ... -> THINKING -> "
            f"SPEAKING; the server published {states}"
        )
        assert states[-1] == "IDLE", f"…and the turn ends back at IDLE; it published {states}"
        assert probe["chip"] == "IDLE", f"…and the chip says so; it reads {probe['chip']!r}"

        # (f) one line, and it says the input was typed
        lines = live.turn_lines()
        assert len(lines) == 1, f"criterion 9: one turn, one turns.jsonl line; got {lines}"
        assert lines[0]["input"] == "text", (
            f"criterion 9: TurnRecord.input is 'text' for a typed turn ('voice' for a spoken one, "
            f"'ui' for a click); the line says {lines[0].get('input')!r}"
        )
        # …and the line is the record of the turn this test drove, not an empty one. The record
        # itself cannot say so -- ``TurnRecord`` carries no transcript text (T-0032 ruling A1) --
        # so the anti-vacuity check is the session's own history for that turn, read the way
        # tests/unit/test_session_text_turn.py reads it (criterion 5: the typed text is the *user*
        # side of history, verbatim).
        history = live.session.state.history
        assert len(history) == 1, f"criterion 5: one typed turn, one history entry; got {history}"
        assert history[0].user_text == QUESTION, (
            f"criterion 5: the turn behind that turns.jsonl line is this test's typed question, "
            f"verbatim and with no [ui] prefix; history reads {history[0].user_text!r}"
        )
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_the_typed_turn_leaks_no_thought_and_synthesises_no_stt(tmp_path: Path):
    """Row (e) of §4, and the two negatives §5 names.

    The thought-leak scan of the rehearsal, extended to the text rows: ``SECRET_THOUGHT`` is
    committed by the LLM on this turn and may appear in no DOM, in nothing ``HudServer.publish``
    fanned out and in no ``turns.jsonl`` line (criterion 21). The second half is the
    implementation note "never synthesise ``stt.*`` events for typed text": the endpointing rows
    belong to a microphone, and a machine that saw one here would have been driven by a fake.

    **Criterion 21 narrowed by T-0180** (B9, ruling R-1): on the wire the thought may appear only
    in the ``text`` of an ``astra.thought``, and (anti-vacuity) it **does** appear there, since
    the script committed it; it stays absent from every other message and field, from history
    and from ``turns.jsonl``. **The DOM half (T-0180 group C):** the thought may appear in the DOM
    only inside ``[data-model-thoughts]``, and (anti-vacuity) it **does** appear there; everywhere
    else in the DOM it stays absent.
    """
    live = Live(tmp_path, script(), ms_per_word=200)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            outcome = await page.evaluate(type_and_submit_js(QUESTION))
            assert outcome == "submitted", f"the page could not be typed into: {outcome!r}"
            await live.until(
                lambda: live.session.state.state is not TurnState.IDLE,
                "the typed question never opened a turn",
                timeout_s=PAGE_TIMEOUT_S,
            )
            await live.idle()
            await asyncio.sleep(0.2)
            outside, inside = await thought_dom(page)

        # Anti-vacuity: the thought really did enter the conductor on this turn.
        thoughts = [
            e
            for e in live.machine.events(E.LlmCommit)
            if e.segment == "thought" and "SECRETTHOUGHT" in e.text
        ]
        assert thoughts, (
            "the turn machine never saw the thought commit, so the scans below prove nothing; it "
            f"saw {[(e.segment, e.text[:40]) for e in live.machine.events(E.LlmCommit)]}"
        )
        assert "SECRETTHOUGHT" not in outside, (
            "criterion 21 (narrowed by T-0180 C): the assistant's thinking reached the DOM outside "
            "[data-model-thoughts]"
        )
        assert SECRET_THOUGHT in inside, (
            "T-0180 B9 anti-vacuity (DOM): the committed thought is not in the turn's "
            f"[data-model-thoughts]; the disclosures read {inside[:200]!r}"
        )
        leaked = thought_leaks(live.wire, "SECRETTHOUGHT")
        assert not leaked, (
            "criterion 21 (narrowed by T-0180): the thought went out on the wire outside "
            f"astra.thought's text: {leaked}"
        )
        shown = thought_texts(live.wire)
        assert any(SECRET_THOUGHT in text for text in shown), (
            "T-0180 B9 anti-vacuity: the committed thought is in no astra.thought text; the "
            f"astra.thought texts on the wire were {shown}"
        )
        history = json.dumps([h.model_dump(mode="json") for h in live.session.state.history])
        assert "SECRETTHOUGHT" not in history, "T-0180 B9: the thought reached history"
        recorded = live.turns_path.read_text() if live.turns_path.exists() else ""
        assert recorded.strip(), "the turn must have been recorded at all"
        assert "SECRETTHOUGHT" not in recorded, (
            f"criterion 21: the thought was written to turns.jsonl:\n{recorded}"
        )

        # The typed question is never voiced and never becomes assistant text (criterion 21).
        spoken = [m for m in live.wire if isinstance(m, AstraSpoken)]
        assert spoken, "the turn did speak an answer"
        assert not any(QUESTION in m.text for m in spoken), (
            f"criterion 21: the typed question was handed to TTS and came back as an "
            f"astra.spoken: {[m.text for m in spoken]}"
        )

        # No stt.* event was synthesised for the typed text (§5, and T-0032's endpointer). The
        # one onset criterion 5 *does* ask for is not an stt.* event and is not in this list.
        for kind in (E.SttWord, E.SttEndOfTurn, E.SttFlushed):
            assert live.machine.events(kind) == [], (
                f"§5: the session hands the machine the end-of-turn state directly, as the [ui] "
                f"path already does; it synthesised {kind.__name__} events for typed text: "
                f"{live.machine.events(kind)}"
            )

        # One final transcript for the question, and nothing after it (criterion 8).
        transcripts = [m for m in live.wire if isinstance(m, AstraTranscript)]
        finals = [m for m in transcripts if m.final]
        assert len(finals) == 1, (
            f"criterion 8: one astra.transcript(final=True) for the typed question; got "
            f"{[(m.text, m.final) for m in transcripts]}"
        )
        assert finals[0].text == QUESTION, (
            f"criterion 5: verbatim, with no [ui] prefix; it published {finals[0].text!r}"
        )
        assert transcripts[-1] is finals[0], (
            f"criterion 8: nothing follows the final transcript -- a final followed by partials "
            f"is the breach: {[(m.text, m.final) for m in transcripts]}"
        )
    finally:
        hud.shutdown()
        await live.aclose()


# --- T-0039 ruling A19/A20: the two defects only a real browser shows -----------------------------


#: What the page says about the capture, read the way ``tests/integration/test_hud_headless.py``'s
#: DOM contract pins it: the criterion-7 banner and ``[data-capture-size]``.
CAPTURE_JS = """
(() => {
  const banner = document.querySelector('#capture-banner');
  const size = document.querySelector('[data-capture-size]');
  return {
    banner: banner ? banner.textContent : null,
    banner_hidden: banner ? Boolean(banner.hidden) : null,
    capture_size: size ? size.getAttribute('data-capture-size') : null,
    frames_sent: Number(
      (document.querySelector('[data-frames-sent]') || {}).getAttribute
        ? document.querySelector('[data-frames-sent]').getAttribute('data-frames-sent')
        : 0,
    ),
  };
})()"""

FOCUS_JS = """
(() => {
  const input = document.querySelector('[data-ask-input]');
  if (!input) return 'no [data-ask-input] on the page';
  input.focus();
  return document.activeElement === input ? 'focused' : 'the box refused the focus';
})()"""


def serve_dir(directory: Path) -> tuple[socketserver.TCPServer, int]:
    """``serve_hud()`` for a *copy* of ``hud/`` — the row below serves a page with a file missing."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


@requires_chrome
async def test_typing_a_question_does_not_stop_the_screen_capture(tmp_path: Path):
    """T-0039 ruling A19(1)/A20: the keystrokes of a question must not reach main.js's shortcut.

    ``hud/src/main.js`` binds a document-level ``keydown`` that stops the capture on a bare ``m``
    (the mute shortcut), and ``main.js`` is zero-diff here (criterion 20). So every question with
    an ``m`` in it — including this module's — kills perception halfway through the typing, and
    the turn the user is about to open sees no screen. The keystrokes are delivered through
    ``Input.dispatchKeyEvent``, the browser's own key pipeline: a ``dispatchEvent`` built in
    ``Runtime.evaluate`` is untrusted, inserts no character and would never reach that listener,
    so it could not see this defect at all.

    No number here is a latency claim; every wait is a generous wall-clock ceiling.
    """
    assert "m" in QUESTION, "this row is about the mute shortcut: the question must contain an 'm'"
    live = Live(tmp_path, script(), ms_per_word=200)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "(document.querySelector('[data-capture-size]') || {})"
                ".getAttribute('data-capture-size')",
                "the synthetic capture reports its size",
                timeout_s=PAGE_TIMEOUT_S,
            )
            before = await page.evaluate(CAPTURE_JS)
            assert before["banner"] == "SYNTHETIC CAPTURE", (
                f"the ?source=canvas capture is running and labelled before anything is typed; "
                f"the banner reads {before['banner']!r}"
            )
            assert before["capture_size"], "…and the page reports the capture's size"

            assert await page.evaluate(FOCUS_JS) == "focused"
            await asyncio.to_thread(page.chrome.type_text, QUESTION)

            typed = await page.evaluate("(document.querySelector('[data-ask-input]') || {}).value")
            assert typed == QUESTION, (
                f"anti-vacuity: the keystrokes must really have landed in the box, or the capture "
                f"assertions below hold of a page nobody typed into; it reads {typed!r}"
            )

            after = await page.evaluate(CAPTURE_JS)
            assert after["banner"] == before["banner"] and after["banner_hidden"] is False, (
                "A19(1): typing the question stopped the capture — main.js's bare-'m' mute "
                f"shortcut fired while the user was typing. The banner now reads "
                f"{after['banner']!r} (hidden={after['banner_hidden']!r})"
            )
            assert after["capture_size"] == before["capture_size"], (
                f"A19(1): [data-capture-size] changed from {before['capture_size']!r} to "
                f"{after['capture_size']!r} while a question was typed: the capture was torn down"
            )
            # …and the capture is still *running*, not merely still labelled: frames keep going
            # out. ``stopCapture`` leaves the pipeline throwing "capture is not ready".
            await page.until(
                "Number(document.querySelector('[data-frames-sent]')"
                f".getAttribute('data-frames-sent')) > {int(before['frames_sent'])}",
                "the page keeps sending frames after the question was typed",
                timeout_s=PAGE_TIMEOUT_S,
            )

            # And the same key pipeline submits: Enter in the box opens the turn.
            await asyncio.to_thread(page.chrome.enter)
            await live.until(
                lambda: live.session.state.state is not TurnState.IDLE,
                "Enter pressed in the box (as a real key event) never opened a turn",
                timeout_s=PAGE_TIMEOUT_S,
            )
            await live.idle()
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_the_form_never_navigates_when_the_panel_is_absent(tmp_path: Path):
    """T-0039 ruling A19(2)/A20: the ``<form>`` is live before — and without — the panel.

    The page is served from a copy of ``hud/`` with ``hud/src/ask/index.js`` deleted, which is
    exactly the window every real load passes through (the panel is an ES module fetched after
    the document) and the state a failed panel load leaves the page in for good. A bare ``<form>``
    submits natively: the page reloads, ``?ws=`` and ``?source=`` are replaced by the form's own
    (empty) query string, the WebSocket drops, and the HUD comes back pointed at the default
    Conductor with no capture.

    Two probes, because the fix has two halves: a real Enter keystroke in the box (which a
    ``disabled`` input swallows) and an explicit ``form.requestSubmit()`` (which only a
    non-navigating form — ``method="dialog"`` — can absorb). Neither may navigate.
    """
    page_root = tmp_path / "hud-without-the-ask-panel"
    shutil.copytree(REPO / "hud", page_root)
    panel = page_root / "src" / "ask" / "index.js"
    assert panel.exists(), (
        "criterion 10 pins hud/src/ask/index.js as the panel module; this row removes it, and it "
        "was not there to remove"
    )
    panel.unlink()

    live = Live(tmp_path, script(), ms_per_word=200)
    await live.start()
    hud, hud_port = serve_dir(page_root)
    try:
        url = page_url(hud_port, live.server.url)
        async with open_page(url) as page:
            marked = await page.evaluate(
                "(() => { window.__astra_no_reload = 1;"
                " return { search: location.search,"
                " form: Boolean(document.querySelector('[data-ask-form]')),"
                " input: Boolean(document.querySelector('[data-ask-input]')) }; })()"
            )
            assert marked["form"] and marked["input"], (
                "criterion 10: the box and its form are markup in hud/index.html, present even "
                f"with the panel module missing; the page has {marked!r}"
            )
            assert "ws=" in marked["search"] and "source=" in marked["search"], marked

            assert await page.evaluate(FOCUS_JS) in ("focused", "the box refused the focus")
            await asyncio.to_thread(page.chrome.type_text, "hello")
            await asyncio.to_thread(page.chrome.enter)
            await page.evaluate(
                "(() => { const f = document.querySelector('[data-ask-form]');"
                " if (f && typeof f.requestSubmit === 'function') f.requestSubmit();"
                " return 'submitted'; })()"
            )
            await asyncio.sleep(1.0)  # a navigation is not instantaneous; a generous ceiling

            state = await page.evaluate(
                "({ survived: window.__astra_no_reload === 1, search: location.search })"
            )
        assert state["survived"] is True, (
            "T-0039 ruling A19(2): the form navigated — the page reloaded, the socket dropped and "
            "the turn the user just started would have died with it. The panel cannot be what "
            "prevents this: it is not loaded yet on every real page load, and never on a failed "
            "one (criterion 12: submitting never navigates)"
        )
        assert "ws=" in state["search"] and "source=" in state["search"], (
            f"A19(2): the native submit replaced the query string; the page now reads "
            f"{state['search']!r} and has lost ?ws= and/or ?source= — it would reconnect to the "
            f"default Conductor with no capture"
        )
    finally:
        hud.shutdown()
        await live.aclose()
