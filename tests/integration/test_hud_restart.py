"""T-0162 group D: the Restart button and the page-wide clear, in the real page.

``docs/tasks/T-0162.md`` D-1 to D-4 and *Tests to write* "test_hud_restart.py": two turns, then
Restart → confirm: the page is empty, the capture stops, and the next prompt has no history; a
reconnect replays the clear; *adversarial*: Restart mid-canvas, and the aborted frames never
reappear. Group A's server half (``astra.session.reset`` → ``astra.session.cleared``, merged at
f94bf8d5) is real here; what these rows pin is the page.

**Regime** (docs/tasks/T-0162.md §Regime): the rig of ``tests/integration/test_hud_rehearsal.py``
imported, not copied -- fakes and recorders of ``tests/unit/_hud_rig.py`` behind a real ``Session``
on ``SessionClock`` (wall time), real ``HudServer(port=0)``, the page over loopback http with
``?source=canvas`` (the synthetic capture starts at load), headless Chrome (``google-chrome``),
fake LLM (``RecordingLlm``, one script replayed per request), ``ms_per_word=60``, n=1 per row, no
GPU, no model, no network off loopback. **No number here is a latency claim**: every wait is a
generous wall-clock ceiling.

**What the page is driven by.** Content arrives the way it does in a session: spoken turns through
``Live.ask``. Two things the fake-LLM rig cannot produce are published on ``HudServer.publish`` in
their wire shape, exactly as the session and the canvas relay publish them: an inline tool call
(``AstraToolCall``; tools are off in this rig) and denoising frames (``AstraCanvasStep``; the fake
LLM has no plugin tap). The page cannot tell the difference, and the server's own reset filter
(group A ruling (a)) sees the relay's request ids exactly as it would live.

**DOM contract** (the hooks ``tests/hud/session.test.mjs`` pins): ``[data-restart-button]`` in the
header, ``[data-restart-confirm]`` (hidden until pressed) with ``[data-restart-confirm-yes]`` and
``[data-restart-cancel]``; T-0154's thread hooks; the Tools pane's ``[data-tools-call]``; the
denoising view's ``[data-denoise-root]``/``[data-denoise-label]``/``[data-denoise-empty]``.

Activated with group D's implementation (2026-10-27): no ``pending`` marker.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from typing import Any

import pytest

from local_astra.conductor.hud_server import HudConnection
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart
from local_astra.contracts.protocol import (
    AstraCanvasStep,
    AstraSessionCleared,
    AstraSpoken,
    AstraToolCall,
    AstraTranscript,
)
from local_astra.contracts.turn import TurnState
from tests.integration.test_hud_headless import CHROME, Chrome, requires_chrome, serve_hud
from tests.integration.test_hud_rehearsal import (
    BUTTON_LABEL,
    PAGE_TIMEOUT_S,
    TURN_TIMEOUT_S,
    Live,
    Page,
    answer,
    light_click_js,
    open_page,
    page_url,
    thought,
    ui_commit,
)

pytestmark = [pytest.mark.integration]

#: Distinctive questions, so "the next prompt has no history" is a substring search.
Q1 = "please describe the walrus colony"
Q2 = "and how many penguins are nearby"
Q3 = "what time is it on the moon"
ANSWER = "The colony is large. "
SECRET_THOUGHT = "SECRETTHOUGHT restart rows never show this"

#: The confirm text: the question verbatim (D-1), the memory wipe (H-T162-1) and the agent browser's
#: start page (group A ruling (c), 2026-10-27). The task gives the first two verbatim and the
#: browser clause only as a fact the text must state, so it is matched by its two nouns.
CONFIRM_QUESTION = "Start a new session?"
CONFIRM_MEMORY = "everything Gearvis remembers"
#: Ruling (b), group D reviews 2026-10-27: the wording is fixed; its key phrases are pinned.
CONFIRM_RULED = (
    "clears the conversation and everything Gearvis remembers",
    "stops screen sharing",
    "Gearvis's own browser, if it is open, back to its start page",
    "Session logs already written stay on disk.",
)

MID_CANVAS_RID = "chatcmpl-t0162d-midcanvas-0000000a"
PRE_RESET_RID = "chatcmpl-t0162d-turn2-0000000b"


def script() -> list[Any]:
    """A thought, one spoken answer and a card: every turn leaves a bubble, a line and a card."""
    return [thought(SECRET_THOUGHT), answer(ANSWER), ui_commit()]


def frame(rid: str, step: int, canvas_idx: int = 0) -> AstraCanvasStep:
    """A CL 8 denoise frame: argmax 100+p, canvas equal on the first ``step % 9`` positions."""
    kept = min(8, step % 9)
    return AstraCanvasStep(
        request_id=rid,
        canvas_idx=canvas_idx,
        step=step,
        max_steps=48,
        canvas_length=8,
        canvas=tuple(100 + p if p < kept else 900 + p for p in range(8)),
        argmax=tuple(100 + p for p in range(8)),
        stable="00",
        phase="denoise",
    )


# --- the page side ------------------------------------------------------------------------------

#: Everything D-2 says is cleared and D-2 says is kept, plus the capture (D-3), in one read.
STATE_JS = """
(() => {
  const all = (s) => [...document.querySelectorAll(s)];
  const one = (s) => document.querySelector(s);
  const visible = (el) => Boolean(el) && (el.checkVisibility ? el.checkVisibility() : el.offsetParent !== null);
  const root = one('[data-denoise-root]');
  const button = one('[data-restart-button]');
  const confirm = one('[data-restart-confirm]');
  const log = one('[data-log-toggle]');
  const banner = one('#capture-banner');
  return {
    articles: all('[data-thread] article[data-turn-id]').map((a) => a.getAttribute('data-turn-id')),
    bubbles: all('[data-transcript]').length,
    utterances: all('[data-utterance-id]').length,
    tool_cards: all('[data-tool-call-id]').length,
    card_wraps: all('[data-ui-surface]').length,
    card_roots: all('[data-a2ui-root]').length,
    inert_cards: all('[data-ui-surface]').filter((w) => w.inert || w.closest('[inert]')).map((w) => w.getAttribute('data-ui-surface')),
    tools_calls: all('[data-tools-call]').length,
    denoise_sections: root ? root.children.length : null,
    denoise_label: (one('[data-denoise-label]') || {}).textContent || '',
    denoise_empty: (one('[data-denoise-empty]') || {}).textContent || '',
    meter: (one('#thinking-meter') || {}).textContent || '',
    thread_text: (one('[data-thread]') || {}).textContent || '',
    restart: button ? { text: button.textContent.trim(), disabled: button.disabled, in_header: Boolean(button.closest('header')) } : null,
    confirm_visible: visible(confirm),
    confirm_text: confirm ? confirm.textContent.replace(/\\s+/g, ' ').trim() : null,
    banner_visible: Boolean(banner) && !banner.hidden,
    mute_disabled: one('#mute-button').disabled,
    capture_size: one('[data-capture-size]').getAttribute('data-capture-size'),
    preview_hidden: one('#preview').hidden,
    frames_sent: Number(one('[data-frames-sent]').getAttribute('data-frames-sent') || 0),
    theme: document.documentElement.getAttribute('data-theme'),
    log_checked: log ? log.checked : null,
    status: (one('#conn-status') || {}).textContent,
  };
})()"""


def click_js(selector: str) -> str:
    return """(() => {
  const el = document.querySelector(__SEL__);
  if (!el) return 'no ' + __SEL__ + ' on the page';
  if (el.disabled) return __SEL__ + ' is disabled';
  el.click();
  return 'clicked';
})()""".replace("__SEL__", json.dumps(selector))


async def state(page: Page) -> dict[str, Any]:
    value = await page.evaluate(STATE_JS)
    assert isinstance(value, dict) and "__exception__" not in value, f"the probe threw: {value!r}"
    return value


async def press(page: Page, key: str, text: str | None = None, code: int | None = None) -> None:
    await asyncio.to_thread(page.chrome.key, key, text=text, code=code)


async def restart(page: Page, live: Live) -> dict[str, Any]:
    """Press Restart, read the open confirm, press its Restart. Returns the confirm's state.

    Between the two clicks nothing may have reached the server (D-1's two steps, R6)."""
    outcome = await page.evaluate(click_js("[data-restart-button]"))
    assert outcome == "clicked", f"D-1: a header button labelled Restart: {outcome!r}"
    await page.until(
        "(() => { const c = document.querySelector('[data-restart-confirm]');"
        " return Boolean(c) && c.checkVisibility(); })()",
        "D-1: the inline confirm opens on the first click",
    )
    opened = await state(page)
    await asyncio.sleep(0.3)  # a reset sent by the first click would have been routed by now
    assert resets_seen(live) == [], (
        f"D-1 (two steps): the first click sent astra.session.reset; the machine saw {resets_seen(live)}"
    )
    outcome = await page.evaluate(click_js("[data-restart-confirm-yes]"))
    assert outcome == "clicked", f"D-1: the confirm's Restart: {outcome!r}"
    return opened


def resets_seen(live: Live) -> list[Any]:
    return live.machine.events(E.SessionReset)


def cleared_on_wire(live: Live) -> list[AstraSessionCleared]:
    return [m for m in live.wire if isinstance(m, AstraSessionCleared) and not m.replay]


def turn_ids(live: Live) -> list[str]:
    out: list[str] = []
    for m in live.wire:
        if isinstance(m, AstraTranscript) and m.turn_id not in out:
            out.append(m.turn_id)
    return out


def prompt_text(prompt: Any) -> str:
    return "\n".join(m.text() for m in prompt.messages)


def prompt_images(prompt: Any) -> int:
    return sum(isinstance(p, ImagePart) for m in prompt.messages for p in m.content)


# --- rows ---------------------------------------------------------------------------------------


@requires_chrome
async def test_two_turns_then_restart_empties_the_page_stops_the_capture_and_the_next_prompt_has_no_history(
    tmp_path,
):
    """D-1, D-2, D-3, D-4 and A-3 through the page: the whole path, once.

    Two spoken turns leave two articles (bubbles, lines, cards; the older card inert), an inline
    tool card and a Tools pane line (published), and a drawn denoising canvas (published). Restart
    → confirm sends one ``astra.session.reset`` (one ``SessionReset`` reaches the machine). Once the
    server's ``astra.session.cleared`` is out, the page holds nothing of either turn, the capture
    is stopped, and what D-2 keeps (theme, log checkbox, diagnostics counters) is unchanged. The
    third turn's prompt carries neither earlier question nor any image (the capture is stopped and
    the pre-reset frames are below the floor), and its card is the newest and not inert.
    """
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "Number(document.querySelector('[data-frames-sent]').getAttribute('data-frames-sent')) > 0",
                "premise: the synthetic capture sends frames before the first turn",
            )
            for question in (Q1, Q2):
                await live.ask(question)
                await live.idle()
            ids = turn_ids(live)
            assert len(ids) == 2, f"premise: two turns with a transcript; the wire names {ids}"
            live.server.publish(
                AstraToolCall(
                    turn_id=ids[1],
                    call_id="c-restart-1",
                    name="scroll",
                    arguments={"direction": "down"},
                    status="executed",
                    source="draft",
                    action="scrolled down",
                )
            )
            for step in (3, 4):
                live.server.publish(frame(PRE_RESET_RID, step))
            await page.until(
                "document.querySelectorAll('[data-tools-call]').length === 1"
                " && document.querySelector('[data-denoise-root]').children.length > 0"
                " && document.querySelectorAll('[data-thread] article[data-turn-id]').length === 2",
                "premise: two articles, one tool call in the Tools pane and a drawn canvas",
            )
            before = await state(page)
            assert before["tool_cards"] == 1 and before["card_wraps"] == 2, (
                f"premise: one inline tool card and two cards; {before}"
            )
            assert before["inert_cards"] == [f"turn-{ids[0]}"], (
                f"D-4 premise (T-0154 B8): before the reset the older card is inert; {before['inert_cards']}"
            )
            assert before["banner_visible"] and not before["mute_disabled"], (
                "premise: the capture runs before the reset"
            )
            assert before["restart"] is not None, "D-1: the page has a [data-restart-button]"
            assert before["restart"]["text"] == "Restart" and before["restart"]["in_header"], (
                f"D-1: a header button labelled 'Restart'; {before['restart']}"
            )
            assert not before["confirm_visible"], (
                "D-1: the confirm is closed until Restart is pressed"
            )
            first_prompt = live.llm.gen_prompts[0][1]
            assert prompt_images(first_prompt) >= 1, (
                "anti-vacuity: the pre-reset turn's prompt carried a frame, so 'no old frame' below "
                "is a statement about the reset, not about a rig that never sends one"
            )

            opened = await restart(page, live)
            text = opened["confirm_text"] or ""
            assert CONFIRM_QUESTION in text, (
                f"D-1: the confirm asks {CONFIRM_QUESTION!r}; it reads {text!r}"
            )
            assert CONFIRM_MEMORY in text, (
                f"H-T162-1 / R6: the confirm names the memory wipe; it reads {text!r}"
            )
            for phrase in CONFIRM_RULED:
                assert phrase in text, f"ruling (b): the confirm says {phrase!r}; it reads {text!r}"

            await live.until(
                lambda: len(cleared_on_wire(live)) == 1,
                "the server published astra.session.cleared",
            )
            await live.idle()
            await page.until(
                "document.querySelectorAll('[data-thread] article[data-turn-id]').length === 0",
                "D-2: the thread is emptied by astra.session.cleared",
            )
            await asyncio.sleep(0.3)
            after = await state(page)
            sent_then = after["frames_sent"]
            accepted_then = len(live.sink.accepted)
            await asyncio.sleep(1.2)
            later = await state(page)

        assert len(resets_seen(live)) == 1, (
            f"D-1: one confirmed Restart sends exactly one astra.session.reset; the machine saw "
            f"{len(resets_seen(live))}"
        )
        # D-2: cleared
        for key in (
            "bubbles",
            "utterances",
            "tool_cards",
            "card_wraps",
            "card_roots",
            "tools_calls",
        ):
            assert after[key] == 0, (
                f"D-2: {key} is cleared by astra.session.cleared; it reads {after[key]}"
            )
        assert after["articles"] == [], f"D-2: no article survives; {after['articles']}"
        assert after["denoise_sections"] == 0, (
            f"D-2: the denoising view's sections are gone; {after['denoise_sections']}"
        )
        assert after["denoise_label"] == "", (
            f"D-2: its label is empty again; {after['denoise_label']!r}"
        )
        assert after["denoise_empty"].startswith("no denoising data yet"), (
            f"D-2: the empty line reads its own copy; {after['denoise_empty']!r}"
        )
        assert after["meter"] == "—", f"D-2: the thinking meter shows nothing; {after['meter']!r}"
        assert SECRET_THOUGHT not in after["thread_text"] and Q1 not in after["thread_text"]
        assert not after["confirm_visible"], "the confirm closes after the reset"
        # D-3: the capture stops, as Mute does
        assert not after["banner_visible"], (
            "D-3: the synthetic-capture banner goes with the capture"
        )
        assert after["mute_disabled"], "D-3: Mute is disabled, as after Mute (nothing to stop)"
        assert after["capture_size"] == "" and after["preview_hidden"], (
            f"D-3: no capture size and no preview; {after['capture_size']!r}, hidden={after['preview_hidden']}"
        )
        assert later["frames_sent"] == sent_then and len(live.sink.accepted) == accepted_then, (
            f"D-3: no frame is sent after the reset (page {sent_then} -> {later['frames_sent']}, "
            f"server {accepted_then} -> {len(live.sink.accepted)})"
        )
        # D-2: kept
        assert after["theme"] == before["theme"], "D-2: the theme is kept"
        assert after["log_checked"] == before["log_checked"], "D-2: the log checkbox is kept"
        assert after["frames_sent"] >= before["frames_sent"], (
            f"D-2: the diagnostics counters are kept, not reset ({before['frames_sent']} -> {after['frames_sent']})"
        )
        assert after["restart"] is not None and not after["restart"]["disabled"], (
            "Restart stays usable"
        )
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_the_next_turn_after_restart_has_no_history_and_its_card_is_the_newest(tmp_path):
    """A-3 and D-4 seen from the page: after Restart → confirm, the third turn's prompt holds
    neither earlier question, nor the earlier answer, nor any image; its article is the only one,
    and its card is not inert and sends its action (one ``UiAction`` reaches the machine)."""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            for question in (Q1, Q2):
                await live.ask(question)
                await live.idle()
            assert ANSWER.strip() in prompt_text(live.llm.gen_prompts[-1][1]), (
                "anti-vacuity: the second turn's prompt carries the first turn's answer as history"
            )
            await restart(page, live)
            await live.until(
                lambda: len(cleared_on_wire(live)) == 1,
                "the server published astra.session.cleared",
            )
            await live.idle()
            n_prompts = len(live.llm.gen_prompts)
            await live.ask(Q3)
            await live.idle()
            await page.until(
                "document.querySelectorAll('[data-ui-surface]').length === 1",
                "the third turn's card renders",
                timeout_s=TURN_TIMEOUT_S,
            )
            third = await state(page)
            # The quiz's action button (the fixture's "Check answers", which raises check_quiz),
            # not its first button: that is a tab, which raises no action.
            clicked = await page.evaluate(light_click_js(BUTTON_LABEL))
            await live.until(
                lambda: len(live.machine.events(E.UiAction)) >= 1,
                "D-4: the first post-reset card's action reaches the server",
                timeout_s=PAGE_TIMEOUT_S,
            )
            await live.idle()

        prompts = live.llm.gen_prompts[n_prompts:]
        assert prompts, "premise: the third turn generated"
        text = prompt_text(prompts[0][1])
        assert Q3 in text, "premise: the third turn's prompt carries its own question"
        for old in (Q1, Q2, ANSWER.strip(), SECRET_THOUGHT):
            assert old not in text, (
                f"A-3: the post-reset prompt has no history; it contains {old!r}"
            )
        assert prompt_images(prompts[0][1]) == 0, (
            "A-3 / D-3: no old frame: the capture stopped at the reset and the held frames are below "
            f"the floor, yet the prompt carries {prompt_images(prompts[0][1])} image(s)"
        )
        assert len(third["articles"]) == 1, (
            f"D-2: only the new session's turn is on the page; {third['articles']}"
        )
        assert third["inert_cards"] == [], (
            f"D-4: the first new surface is the newest, not inert; {third['inert_cards']}"
        )
        assert clicked == "clicked", clicked
        third_turn = turn_ids(live)[2]
        actions = [(a.surface_id, a.name) for a in live.machine.events(E.UiAction)]
        assert actions[:1] == [(f"turn-{third_turn}", "check_quiz")], (
            f"D-4: the post-reset card's own action (the quiz's check_quiz) reaches the server; {actions}"
        )
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_a_page_that_missed_the_echo_clears_at_its_hello_and_a_repeated_replay_keeps_the_new_session(
    tmp_path,
):
    """D-2 "live or replayed", A-5: a page whose socket dropped at the moment of the reset.

    The page's connection is closed instead of being offered the live ``astra.session.cleared``
    (``HudConnection.offer`` wrapped for that one message); the reset is submitted exactly as
    ``HudServer.route`` submits another page's ``astra.session.reset``. ``ws.js`` reconnects inside
    the grace, the hello reply begins with the reset replayed, and the page must clear what it still
    shows. Then the guard's other side: a new turn, a second socket-only reconnect whose hello
    replays the *same* ``resets`` again, and the new turn's article must be the very same node.
    """
    live = Live(tmp_path, script(), ms_per_word=60)
    opened: list[HudConnection] = []
    real_open = live.server.open

    def tapped_open() -> HudConnection:
        connection = real_open()
        opened.append(connection)
        return connection

    live.server.open = tapped_open  # type: ignore[method-assign]
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await live.ask(Q1)
            await live.idle()
            await page.until(
                "document.querySelectorAll('[data-thread] article[data-turn-id]').length === 1",
                "premise: the first turn's article",
            )
            first = opened[0]
            real_offer = first.offer

            def drop_at_the_echo(message: Any) -> bool:
                if isinstance(message, AstraSessionCleared):
                    first.close(reason="T-0162 D test: the socket dropped as the reset happened")
                    return True
                return real_offer(message)

            first.offer = drop_at_the_echo  # type: ignore[method-assign]
            hellos = live.hellos
            live.session.submit(E.SessionReset(t_ms=live.now))
            await live.until(lambda: live.hellos >= hellos + 1, "the page greeted the server again")
            await page.connected()
            replay_head = live.hello_replies[-1][0] if live.hello_replies[-1] else None
            await page.until(
                "document.querySelectorAll('[data-thread] article[data-turn-id]').length === 0",
                "D-2: the replayed astra.session.cleared clears the page that missed the live one",
            )
            missed = await state(page)

            await live.ask(Q2)
            await live.idle()
            await page.until(
                "document.querySelectorAll('[data-thread] article[data-turn-id]').length === 1",
                "the new session's first turn renders",
            )
            await page.evaluate(
                "window.__astra_new = document.querySelector('[data-thread] article[data-turn-id]'); 1"
            )
            hellos = live.hellos
            opened[-1].close(reason="T-0162 D test: socket-only reconnect after the reset")
            await live.until(
                lambda: live.hellos >= hellos + 1, "the page greeted the server a third time"
            )
            await page.connected()
            await asyncio.sleep(0.5)
            second_head = live.hello_replies[-1][0] if live.hello_replies[-1] else None
            kept = await page.evaluate(
                "({ connected: Boolean(window.__astra_new && window.__astra_new.isConnected),"
                " n: document.querySelectorAll('[data-thread] article[data-turn-id]').length })"
            )

        assert isinstance(replay_head, AstraSessionCleared) and replay_head.replay, (
            f"premise (A-5): the hello reply begins with the reset, replayed; it began with {replay_head!r}"
        )
        assert missed["utterances"] == 0 and missed["bubbles"] == 0 and missed["card_wraps"] == 0, (
            f"D-2: nothing of the pre-reset turn survives the replayed clear; {missed}"
        )
        assert not missed["banner_visible"] and missed["mute_disabled"], (
            "D-3: the replayed reset stops the capture too (the next Share picks the new page)"
        )
        assert (
            isinstance(second_head, AstraSessionCleared)
            and second_head.resets == replay_head.resets
        ), f"premise: the second hello replays the same resets value; it began with {second_head!r}"
        assert kept == {"connected": True, "n": 1}, (
            f"D-2: a repeated resets value clears nothing — the new session's article must be the same "
            f"node after the second reconnect; {kept}"
        )
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_restart_mid_canvas_the_aborted_frames_never_reappear(tmp_path):
    """Adversarial (R1): Restart while the model is still denoising.

    The fake LLM is held, so the turn sits in THINKING with its request open; the relay's frames
    for that request are on the page. Restart → confirm aborts the request (A-2). Then the relay
    sends one more frame of the aborted request, and the LLM is released: nothing of the cut turn
    may reach the page -- no canvas section, no label, no utterance, no card -- in the next two
    seconds or ever after the clear.
    """
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    asking = None
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            live.llm.hold()
            asking = live.asking(Q1)
            await live.until(
                lambda: (
                    live.session.state.state is TurnState.THINKING and bool(live.llm.gen_prompts)
                ),
                "premise: the turn is THINKING with its generate request open",
            )
            request_id = live.llm.gen_prompts[-1][0]
            for step in (1, 2, 3):
                live.server.publish(frame(MID_CANVAS_RID, step))
                await asyncio.sleep(0.05)
            await page.until(
                "document.querySelector('[data-denoise-root]').children.length > 0",
                "premise: the mid-canvas frames are drawn",
            )
            await restart(page, live)
            await live.until(
                lambda: len(cleared_on_wire(live)) == 1,
                "the server published astra.session.cleared",
            )
            cut_at = len(live.wire)
            await page.until(
                "document.querySelector('[data-denoise-root]').children.length === 0",
                "D-2: the denoising view is cleared",
            )
            live.server.publish(frame(MID_CANVAS_RID, 4))  # a late frame of the aborted request
            live.server.publish(frame(MID_CANVAS_RID, 5, canvas_idx=1))
            live.llm.release()
            deadline = time.monotonic() + 2.0
            worst: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                now = await state(page)
                if (
                    now["denoise_sections"]
                    or now["utterances"]
                    or now["card_wraps"]
                    or now["denoise_label"]
                ):
                    worst = now
                    break
                await asyncio.sleep(0.1)
            await live.idle()

        assert ("abort", request_id) in [(what, rid) for what, rid, _ in live.llm.calls], (
            f"premise (A-2): the reset aborted the open request {request_id}; calls {live.llm.calls}"
        )
        assert worst is None, f"R1: the aborted turn repainted the page after the clear: {worst}"
        late_spoken = [m for m in live.wire[cut_at:] if isinstance(m, AstraSpoken)]
        assert late_spoken == [], f"A-6: no utterance of the cut turn after the echo: {late_spoken}"
    finally:
        live.llm.release()
        if asking is not None:
            with contextlib.suppress(Exception):
                await asking
        hud.shutdown()
        await live.aclose()


def _closed_port() -> int:
    """A loopback port nothing listens on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@requires_chrome
async def test_restart_is_disabled_while_the_socket_is_closed(tmp_path):
    """D-1: "The button is disabled while the socket is closed." A page pointed at a loopback port
    nobody serves never connects; its Restart must be disabled, and a click sends nothing (there is
    nothing to send to). The other direction is every row above: connected, it is enabled."""
    hud, hud_port = serve_hud()
    assert CHROME is not None
    chrome = Chrome(CHROME).__enter__()
    try:
        page = Page(chrome)
        await page.navigate(page_url(hud_port, f"ws://127.0.0.1:{_closed_port()}/ws"))
        await page.until(
            "Boolean(document.querySelector('[data-restart-button]'))",
            "D-1: the Restart button is rendered even before a socket opens",
        )
        await asyncio.sleep(1.0)
        s = await state(page)
        assert s["status"] != "connected", (
            f"premise: the page has no socket; status {s['status']!r}"
        )
        assert s["restart"] is not None and s["restart"]["disabled"], (
            f"D-1: Restart is disabled while the socket is closed; {s['restart']}"
        )
    finally:
        with contextlib.suppress(Exception):
            chrome.__exit__()
        hud.shutdown()


@requires_chrome
async def test_restart_never_fires_on_a_key_press_and_m_is_no_restart(tmp_path):
    """D-4 with the browser's own key pipeline (``Input.dispatchKeyEvent``, T-0039 ruling A19):
    r, R, Enter, Space and Escape typed into the page open no confirm and send no reset; then ``m``
    sends no reset either. (T-0181 B3, H-T181-4 re-scoped the old "``m`` still mutes" half: the ``m``
    shortcut is removed, and the row below pins that ``m`` does nothing.)"""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "Boolean(document.querySelector('[data-restart-button]'))",
                "D-1: the page has a Restart button (anti-vacuity for the key rows)",
            )
            await page.until(
                "!document.querySelector('#mute-button').disabled",
                "premise: the synthetic capture runs, so Mute is enabled",
            )
            await page.evaluate("document.activeElement && document.activeElement.blur(); 1")
            for key, text, code in (
                ("r", "r", None),
                ("R", "R", None),
                ("Enter", "\r", 13),
                (" ", " ", 32),
                ("Escape", None, 27),
            ):
                await press(page, key, text, code)
            await asyncio.sleep(0.5)
            keys = await state(page)
            await press(page, "m", "m")
            await asyncio.sleep(0.8)
            after_m = await state(page)

        assert not keys["confirm_visible"], "D-4: no key press opens the Restart confirm"
        assert not after_m["confirm_visible"], "D-4: m opens no Restart confirm"
        assert resets_seen(live) == [], (
            f"D-4: no key press sends astra.session.reset; {resets_seen(live)}"
        )
        assert cleared_on_wire(live) == [], "…and m is not a Restart"
    finally:
        hud.shutdown()
        await live.aclose()


#: A window-level capturing keydown listener: records every key that reached the page before any page
#: handler can stop it (the anti-vacuity premise of "`m` does nothing").
KEYS_SEEN_JS = (
    "(() => { window.__t0181_keys = []; if (!window.__t0181_keys_on) { window.__t0181_keys_on = true;"
    " window.addEventListener('keydown', (e) => window.__t0181_keys.push(e.key), true); } return true; })()"
)


@requires_chrome
async def test_m_on_the_page_does_nothing_and_the_capture_keeps_running(tmp_path):
    """T-0181 B3 (H-T181-4, James 2026-10-28: "remove it"): the ``m`` shortcut for Stop sharing is gone.
    ``m`` on the body with nothing focused, during a running capture, stops nothing (Mute stays enabled,
    the banner stays), opens no confirm and sends no reset; the key did reach the page."""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "!document.querySelector('#mute-button').disabled",
                "premise: the synthetic capture runs, so Stop sharing is enabled",
            )
            before = await state(page)
            await page.evaluate(KEYS_SEEN_JS)
            await page.evaluate("document.activeElement && document.activeElement.blur(); 1")
            await press(page, "m", "m")
            await asyncio.sleep(0.8)
            after = await state(page)
            seen = await page.evaluate("window.__t0181_keys || []")

        assert before["banner_visible"], f"premise: the capture banner shows before m: {before}"
        assert "m" in (seen or []), f"premise: the m keydown reached the page: {seen}"
        assert not after["mute_disabled"] and after["banner_visible"], (
            f"B3: m stopped the capture (no keydown stops it): {after}"
        )
        assert not after["confirm_visible"], "B3: m opens no Restart confirm"
        assert resets_seen(live) == [], f"B3: m sends no reset; {resets_seen(live)}"
    finally:
        hud.shutdown()
        await live.aclose()


# --- reviewer B (2026-10-27): the keyboard, a fresh page, a dropped socket, the lesser keys ------

ACTIVE_JS = """(() => {
  const a = document.activeElement;
  if (!a || a === document.body) return 'body';
  if (a.matches('[data-restart-cancel]')) return 'cancel';
  if (a.matches('[data-restart-confirm-yes]')) return 'yes';
  if (a.matches('[data-restart-button]')) return 'restart';
  return a.tagName.toLowerCase() + (a.id ? '#' + a.id : '');
})()"""

CONFIRM_OPEN_JS = (
    "(() => { const c = document.querySelector('[data-restart-confirm]');"
    " return Boolean(c) && c.checkVisibility(); })()"
)

#: Enter and Space as a keyboard sends them: Enter activates a focused button on keydown (and
#: repeats while held), Space on keyup.
ENTER = {"key": "Enter", "code": 13, "text": "\r"}
SPACE = {"key": " ", "code": 32, "text": " "}


async def raw_key(page: Page, kind: str, key: dict[str, Any], *, auto_repeat: bool = False) -> None:
    params: dict[str, Any] = {
        "type": kind,
        "key": key["key"],
        "windowsVirtualKeyCode": key["code"],
        "nativeVirtualKeyCode": key["code"],
        "autoRepeat": auto_repeat,
    }
    if kind == "keyDown" and key.get("text"):
        params["text"] = key["text"]
        params["unmodifiedText"] = key["text"]
    await asyncio.to_thread(page.chrome._cmd, "Input.dispatchKeyEvent", **params)


async def tap(page: Page, key: dict[str, Any]) -> None:
    await raw_key(page, "keyDown", key)
    await raw_key(page, "keyUp", key)


async def focus_restart(page: Page) -> None:
    outcome = await page.evaluate(
        "(() => { const b = document.querySelector('[data-restart-button]');"
        " if (!b) return 'no button'; b.focus(); return document.activeElement === b ? 'ok' : 'not focused'; })()"
    )
    assert outcome == "ok", f"the Restart button takes the focus: {outcome!r}"


async def escape(page: Page) -> None:
    await tap(page, {"key": "Escape", "code": 27})
    await page.until(f"!{CONFIRM_OPEN_JS}", "Escape closes the confirm")


@requires_chrome
async def test_the_keyboard_on_the_restart_button_opens_the_confirm_on_cancel_and_never_resets(
    tmp_path,
):
    """Reviewer B (2026-10-27), D-1/D-4 with the browser's own key pipeline.

    A focused button is activated by Enter (on keydown, repeated while held) and by Space (on
    keyup): that may open the confirm, but the focus must land on Cancel, so a held Enter cannot run
    on into the confirm's Restart. With the confirm already open, Enter on the header button again
    confirms nothing. Escape (inside) closes it and gives the focus back to Restart. Zero resets.
    """
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "(() => { const b = document.querySelector('[data-restart-button]'); return Boolean(b) && !b.disabled; })()",
                "the Restart button is enabled once the socket is open",
            )
            # Enter on Restart: opens the confirm, the focus on Cancel
            await focus_restart(page)
            await tap(page, ENTER)
            await page.until(CONFIRM_OPEN_JS, "Enter on the focused Restart opens the confirm")
            after_enter = await page.evaluate(ACTIVE_JS)
            await tap(page, {"key": "Escape", "code": 27})  # Escape inside the confirm
            await page.until(f"!{CONFIRM_OPEN_JS}", "Escape inside the confirm closes it")
            after_escape = await page.evaluate(ACTIVE_JS)
            # Space on Restart: the same
            await focus_restart(page)
            await tap(page, SPACE)
            await page.until(CONFIRM_OPEN_JS, "Space on the focused Restart opens the confirm")
            after_space = await page.evaluate(ACTIVE_JS)
            await escape(page)
            # Enter held down on Restart: one keydown, then six auto-repeats, then the keyup
            await focus_restart(page)
            await raw_key(page, "keyDown", ENTER)
            for _ in range(6):
                await raw_key(page, "keyDown", ENTER, auto_repeat=True)
                await asyncio.sleep(0.03)
            await raw_key(page, "keyUp", ENTER)
            await asyncio.sleep(0.5)
            held = len(resets_seen(live))
            if await page.evaluate(CONFIRM_OPEN_JS):
                await escape(page)
            # the confirm already open, Enter on the header button again
            outcome = await page.evaluate(click_js("[data-restart-button]"))
            assert outcome == "clicked", outcome
            await page.until(CONFIRM_OPEN_JS, "the click opens the confirm")
            await focus_restart(page)
            await tap(page, ENTER)
            await tap(page, ENTER)
            await asyncio.sleep(0.5)
            still_open = await page.evaluate(CONFIRM_OPEN_JS)

        assert after_enter == "cancel", (
            f"D-1: the open confirm's focus is on Cancel (the safe answer); it is on {after_enter!r}"
        )
        assert after_escape == "restart", (
            f"Escape inside the confirm returns the focus to Restart; it is on {after_escape!r}"
        )
        assert after_space == "cancel", (
            f"Space opens it with the focus on Cancel too; it is on {after_space!r}"
        )
        assert held == 0, (
            f"D-4: a held Enter on Restart never confirms; {held} reset(s) reached the server"
        )
        assert still_open, (
            "Enter on the header Restart with the confirm open neither confirms nor closes it"
        )
        assert resets_seen(live) == [], (
            f"D-4: no key press sent astra.session.reset; {resets_seen(live)}"
        )
        assert cleared_on_wire(live) == []
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_escape_from_the_page_an_outside_click_and_m_inside_the_confirm(tmp_path):
    """Reviewer B, lower severity: Escape typed with the focus outside the confirm closes it; a
    click outside it closes it; ``m`` typed inside it neither mutes nor closes it. Zero resets."""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(
                "(() => { const b = document.querySelector('[data-restart-button]'); return Boolean(b) && !b.disabled; })()"
                " && !document.querySelector('#mute-button').disabled",
                "Restart enabled and the capture running",
            )
            # Escape from the body
            assert await page.evaluate(click_js("[data-restart-button]")) == "clicked"
            await page.until(CONFIRM_OPEN_JS, "the confirm opens")
            await page.evaluate("document.activeElement && document.activeElement.blur(); 1")
            assert await page.evaluate(ACTIVE_JS) == "body", (
                "premise: the focus is outside the confirm"
            )
            await tap(page, {"key": "Escape", "code": 27})
            await page.until(f"!{CONFIRM_OPEN_JS}", "Escape from the page closes the confirm")
            # an outside click
            assert await page.evaluate(click_js("[data-restart-button]")) == "clicked"
            await page.until(CONFIRM_OPEN_JS, "the confirm opens again")
            await page.evaluate("document.querySelector('[data-thread]').click(); 1")
            await page.until(f"!{CONFIRM_OPEN_JS}", "a click outside the confirm closes it")
            # m inside the confirm
            assert await page.evaluate(click_js("[data-restart-button]")) == "clicked"
            await page.until(CONFIRM_OPEN_JS, "the confirm opens a third time")
            assert await page.evaluate(ACTIVE_JS) == "cancel", (
                "premise: the focus is inside the confirm"
            )
            await press(page, "m", "m")
            await asyncio.sleep(0.5)
            m_inside = await state(page)

        assert not m_inside["mute_disabled"] and m_inside["banner_visible"], (
            "keys typed in the confirm stay in it: m there does not mute"
        )
        assert m_inside["confirm_visible"], "…and does not close the confirm"
        assert resets_seen(live) == [], f"none of this sends a reset; {resets_seen(live)}"
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_a_page_opened_after_the_reset_keeps_its_capture(tmp_path):
    """Reviewer B, D-3's exception: a page opened *after* a reset hears it replayed as the very first
    message of its first hello. Its capture already belongs to the new session, so it keeps running
    and Mute stays enabled (a page that missed a live reset is stopped: the reconnect row above)."""
    live = Live(tmp_path, script(), ms_per_word=60)
    await live.start()
    hud, hud_port = serve_hud()
    url = page_url(hud_port, live.server.url)
    try:
        async with open_page(url) as first:
            await first.until(
                "(() => { const b = document.querySelector('[data-restart-button]');"
                " return Boolean(b) && !b.disabled; })()",
                "the first page's Restart is mounted and enabled",
            )
            await restart(first, live)
            await live.until(
                lambda: len(cleared_on_wire(live)) == 1,
                "the server published astra.session.cleared",
            )
            hellos = live.hellos
            async with open_page(url) as second:
                await live.until(lambda: live.hellos >= hellos + 1, "the second page said hello")
                await second.until(
                    "Number(document.querySelector('[data-frames-sent]').getAttribute('data-frames-sent')) > 0",
                    "the second page's capture sends frames",
                )
                await asyncio.sleep(1.5)
                fresh = await state(second)
            head = live.hello_replies[-1][0] if live.hello_replies[-1] else None

        assert isinstance(head, AstraSessionCleared) and head.replay, (
            f"premise (A-5): the fresh page's hello reply began with the reset, replayed; it began with {head!r}"
        )
        assert fresh["banner_visible"] and not fresh["mute_disabled"], (
            f"D-3 exception: a page opened after the reset keeps its capture and Mute; {fresh}"
        )
        assert fresh["capture_size"] != "", "…and is still capturing"
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_a_page_opened_after_a_reset_keeps_its_new_session_through_a_reconnect(tmp_path):
    """Reviewer A T-1, ruling (a): one tracker for the page.

    The page opens after a reset, so its very first message is that reset replayed -- which can
    arrive before the panels that mount asynchronously (cards, Tools pane, denoising view) are
    listening. It then runs a turn with a card, a tool call and denoising frames. A socket-only
    reconnect replays the *same* reset at the hello; nothing may be cleared: the article, the card,
    the Tools-pane line and the denoising section are the very same nodes afterwards.
    """
    live = Live(tmp_path, script(), ms_per_word=60)
    opened: list[HudConnection] = []
    real_open = live.server.open

    def tapped_open() -> HudConnection:
        connection = real_open()
        opened.append(connection)
        return connection

    live.server.open = tapped_open  # type: ignore[method-assign]
    await live.start()
    hud, hud_port = serve_hud()
    try:
        live.session.submit(E.SessionReset(t_ms=live.now))  # what route() does for a Restart
        await live.until(lambda: len(cleared_on_wire(live)) == 1, "the reset is echoed")
        await live.idle()
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await live.ask(Q1)
            await live.idle()
            turn = turn_ids(live)[-1]
            live.server.publish(
                AstraToolCall(
                    turn_id=turn,
                    call_id="c-after-reset",
                    name="scroll",
                    arguments={"direction": "down"},
                    status="executed",
                    source="draft",
                    action="scrolled down",
                )
            )
            for step in (2, 3):
                live.server.publish(frame("chatcmpl-t0162d-fresh-0000000c", step))
            await page.until(
                "document.querySelectorAll('[data-tools-call]').length === 1"
                " && document.querySelectorAll('[data-ui-surface]').length === 1"
                " && document.querySelector('[data-denoise-root]').children.length > 0",
                "premise: the new session's card, Tools-pane line and canvas are on the page",
            )
            await page.evaluate(
                """(() => { window.__t1 = {
  article: document.querySelector('[data-thread] article[data-turn-id]'),
  card: document.querySelector('[data-ui-surface]'),
  tool: document.querySelector('[data-tools-call]'),
  inline: document.querySelector('[data-tool-call-id]'),
  section: document.querySelector('[data-denoise-root]').firstElementChild,
}; return 1; })()"""
            )
            hellos = live.hellos
            opened[-1].close(reason="T-0162 D test: socket-only reconnect after a fresh page")
            await live.until(lambda: live.hellos >= hellos + 1, "the page greeted the server again")
            await page.connected()
            await asyncio.sleep(0.8)
            kept = await page.evaluate(
                "Object.fromEntries(Object.entries(window.__t1).map(([k, v]) => [k, Boolean(v && v.isConnected)]))"
            )
            head = live.hello_replies[-1][0] if live.hello_replies[-1] else None

        assert isinstance(head, AstraSessionCleared) and head.replay and head.resets == 1, (
            f"premise (A-5): the reconnect's hello replays the same reset; it began with {head!r}"
        )
        assert kept == {
            "article": True,
            "card": True,
            "tool": True,
            "inline": True,
            "section": True,
        }, (
            f"ruling (a): a replay of the reset the page already took clears nothing; still connected: {kept}"
        )
    finally:
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_restart_is_disabled_while_the_socket_is_down_and_enabled_on_reconnect(tmp_path):
    """Reviewer B, D-1 after load: the server drops the page's socket and refuses its reconnects
    for a while (``process_request`` answers 503). Restart goes disabled; once the server accepts
    again and the page reconnects, it is enabled."""
    live = Live(tmp_path, script(), ms_per_word=60)
    blocked = {"on": False}
    real_process = live.server._process_request
    opened: list[HudConnection] = []
    real_open = live.server.open

    def process(connection: Any, request: Any) -> Any:
        if blocked["on"]:
            return connection.respond(503, "T-0162 D test: reconnects refused for a while\n")
        return real_process(connection, request)

    def tapped_open() -> HudConnection:
        connection = real_open()
        opened.append(connection)
        return connection

    live.server._process_request = process  # type: ignore[method-assign]
    live.server.open = tapped_open  # type: ignore[method-assign]
    await live.start()
    hud, hud_port = serve_hud()
    enabled_js = "(() => { const b = document.querySelector('[data-restart-button]'); return Boolean(b) && !b.disabled; })()"
    disabled_js = "(() => { const b = document.querySelector('[data-restart-button]'); return Boolean(b) && b.disabled; })()"
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            await page.until(enabled_js, "D-1: Restart is enabled once the socket is open")
            blocked["on"] = True
            opened[-1].close(reason="T-0162 D test: the socket drops")
            await page.until(disabled_js, "D-1: Restart is disabled while the socket is down")
            down = await state(page)
            blocked["on"] = False
            await page.connected()
            await page.until(
                enabled_js,
                "D-1: Restart is enabled again after the reconnect",
                timeout_s=PAGE_TIMEOUT_S,
            )

        assert down["status"] != "connected", (
            f"premise: the page knew it was down; status {down['status']!r}"
        )
        assert resets_seen(live) == []
    finally:
        hud.shutdown()
        await live.aclose()
