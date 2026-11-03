"""T-0037 criterion 14: a page reloaded mid-turn shows the turn that is still being spoken.

The 2026-09-14 rehearsal's defect D3, landed as a test. Row (f) of
``tests/integration/test_hud_rehearsal.py`` (T-0036, **not touched here**) asserts only that a
fresh page *reconnects*; this module asserts what the user actually sees after the reload — the
turn's utterances in the spoken pane, the question in the transcript pane, the chip on the
server's state, and no thought anywhere — with the real ``hud/`` page in headless Chrome over the
real ``HudServer`` and the real ``Session``.

**The reload is a reload**, not a second browser: the same tab is navigated to the same URL, which
reconnects in a few hundred milliseconds. That matters — ``RECONNECT_GRACE_MS`` (4 s since T-0166,
counted from the last greeted page) bounds it, and a page that came back *after* the grace is a departure, which clears the journal on purpose (criterion 4).
Launching a second Chrome takes seconds and would test the wrong thing; the row asserts the gap it
measured, so a slow box fails with the reason spelled out rather than with an empty pane.

**Regime** (docs/tasks/T-0037.md): the T-0036 pattern — fakes and recorders from
``tests/unit/_hud_rig.py`` behind a real ``Session`` on ``SessionClock`` (wall time), real
``HudServer(port=0)``, the page served over loopback http with ``?source=canvas``, headless
Chrome, ``ms_per_word=400``, n = 1, budget ≤ 45 s, no GPU, no audio device, no model. No number
here is a latency claim: the waits are generous ceilings, not measurements.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import pytest

from local_astra.conductor.hud_server import RECONNECT_GRACE_MS, HudConnection
from local_astra.contracts import events as E
from local_astra.contracts.protocol import (
    AstraSpoken,
    AstraToolCall,
    AstraTranscript,
    AstraTurnState,
    AstraUiFence,
    AstraUiSurface,
)
from local_astra.contracts.turn import TurnState
from tests.integration.test_hud_headless import requires_chrome, serve_hud
from tests.integration.test_hud_rehearsal import (
    QUESTION,
    SECRET_THOUGHT,
    Live,
    Page,
    answer,
    open_page,
    page_url,
    thought,
    thought_dom,
    thought_leaks,
    thought_texts,
)

pytestmark = [pytest.mark.integration]

FIRST = "One two three four five six. "
SECOND = "Seven eight nine ten eleven twelve. "

#: Wall-clock ceilings. The whole row must fit in 45 s; these are the pieces.
SPEAKING_TIMEOUT_S = 20.0
RELOAD_TIMEOUT_S = 15.0

#: T-0154 B2/B10: the thread's articles, in DOM order, by turn id.
ARTICLES_JS = (
    "[...document.querySelectorAll('[data-thread] article[data-turn-id]')]"
    ".map((a) => a.getAttribute('data-turn-id'))"
)


def content_turns(wire: list[Any]) -> list[str]:
    """The turns that get an article (T-0154 B2), in first-seen order, read off the wire.

    Content is a user transcript, an utterance, a surface (its turn is its ``surface_id`` minus
    ``turn-``, implementation note 3), an action card or a fence notice; ``astra.turn.state`` and
    ``astra.spoken.played`` create nothing.
    """
    out: list[str] = []
    for m in wire:
        turn: str | None = None
        if isinstance(m, AstraTranscript | AstraSpoken | AstraToolCall | AstraUiFence):
            turn = m.turn_id
        elif isinstance(m, AstraUiSurface):
            turn = m.surface_id.removeprefix("turn-")
        if turn is not None and turn not in out:
            out.append(turn)
    return out


async def chip_on_the_servers_state(live: Live, page: Page, timeout_s: float = 10.0) -> str:
    """Wait until the chip reads the state the server is in, and say so if it never does.

    A browser cannot be single-stepped, so the chip is compared to the session's state as it
    converges rather than at one instant. The anti-vacuity half is in the caller: the reconnect's
    own hello reply must have carried an ``astra.turn.state``.
    """
    deadline = time.monotonic() + timeout_s
    chip = None
    while time.monotonic() < deadline:
        chip = (await page.probe())["chip"]
        if chip == live.session.state.state.name:
            return str(chip)
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"criterion 14: the reloaded page's chip reads {chip!r} while the server is in "
        f"{live.session.state.state.name} after {timeout_s:.0f} s"
    )


@requires_chrome
async def test_a_page_reloaded_mid_speaking_shows_the_open_turn(tmp_path):
    """Criterion 14, end to end: the reconnecting page is caught up on the open turn.

    Today this row fails on its first assertion — the fresh page's spoken pane is empty while TTS
    is still playing, which is the defect. The expectations are computed from the wire (what this
    server published for this turn) and never from the page, so a page that invented a plausible
    transcript would fail here too.

    **No barge-in before the reload**, deliberately: a ``user.speech_onset`` takes the machine
    through PAUSED and INTERRUPTED and then *opens the next turn* within the same handful of
    milliseconds (the wire reads ``PAUSED t1 · INTERRUPTED t1 · astra.spoken.played t1 ·
    LISTENING t2``), and criterion 4 says a new ``turn_id`` clears the journal. The strike's
    survival across a replay is therefore asserted where it is reachable — over the reducer in
    ``tests/hud/replay.test.mjs`` (criterion 12) and over the reply's ordering in
    ``tests/unit/test_hud_hello_replay.py`` (criterion 1). See this module's report for the
    criterion 14 / criterion 4 tension.

    **Extended by T-0154 B10** (G2: assertions added, none removed): after the reload the thread
    holds exactly one article (the one turn with content, read off the wire) and as many
    ``[data-utterance-id]`` elements as utterances were published — a count, because the set
    equality above is blind to a doubled line — and both counts equal the first page's. The
    anti-vacuity half: the reconnect's hello reply really re-sent the turn's spoken lines.
    """
    script = [thought(SECRET_THOUGHT), answer(FIRST), answer(SECOND)]
    live = Live(tmp_path, script, ms_per_word=400)
    await live.start()
    hud, hud_port = serve_hud()
    url = page_url(hud_port, live.server.url)
    asking = None
    try:
        async with open_page(url) as page:
            asking = live.asking(QUESTION)
            await live.until(
                lambda: len(live.spoken()) >= 2,
                "the turn never published its two utterances",
                timeout_s=SPEAKING_TIMEOUT_S,
            )
            await page.until(
                "document.querySelectorAll('[data-utterance-id]').length >= 2",
                "both utterances on the page before the reload",
            )
            before = await page.probe()
            articles_before = await page.evaluate(ARTICLES_JS)
            hellos_before = live.hellos
            assert before["spoken"], "premise: the first page showed the turn"
            assert live.session.state.state is TurnState.SPEAKING, (
                "premise: the reload must happen mid-turn; the session is in "
                f"{live.session.state.state}"
            )

            # --- the reload: same tab, same URL, a fresh page object and a fresh socket -------
            started = time.monotonic()
            await page.navigate(url)
            await page.connected()
            gap_s = time.monotonic() - started

            assert live.hellos == hellos_before + 1, (
                f"premise: the reloaded page greets the server once more; hellos={live.hellos}"
            )
            assert gap_s < RECONNECT_GRACE_MS / 1000, (
                f"premise: the reload took {gap_s:.2f} s, past the {RECONNECT_GRACE_MS} ms "
                "reconnect grace — the session was told its last page was gone, which clears the "
                "journal on purpose (criterion 4). This row needs a reload inside the window"
            )
            assert live.machine.events(E.ClientDisconnected) == [], (
                "T-0028 criterion 13: a reload inside the grace is not a departure; the machine "
                f"saw {live.machine.events(E.ClientDisconnected)}"
            )

            await page.until(
                "document.querySelectorAll('[data-utterance-id]').length >= 2",
                "criterion 14: the reloaded page shows the utterances of the turn that is still "
                "being spoken — today hello_reply() sends a live chip and nothing else, so the "
                "pane stays empty while TTS keeps playing",
                timeout_s=RELOAD_TIMEOUT_S,
            )
            probe = await page.probe()
            chip = await chip_on_the_servers_state(live, page)
            second_dom = await thought_dom(page)
            articles_after = await page.evaluate(ARTICLES_JS)

        # --- what the server published for this turn: the oracle -----------------------------
        spoken = {m.utterance_id: m.text for m in live.wire if isinstance(m, AstraSpoken)}
        assert len(spoken) >= 2, (
            f"premise: the turn published two spoken lines; it published {sorted(spoken)}"
        )
        shown = {u["utterance_id"]: u for u in probe["spoken"]}
        assert set(shown) == set(spoken), (
            "criterion 14: the reloaded page shows exactly the turn's utterances — no line "
            f"missing and none invented. The server published {sorted(spoken)}; the fresh page "
            f"shows {sorted(shown)}"
        )
        for utterance_id, text in spoken.items():
            assert text.split() == shown[utterance_id]["text"].split(), (
                f"criterion 3: what the page shows is what was published: {text!r} against "
                f"{shown[utterance_id]['text']!r}"
            )
        # T-0154 B10 (G2: added, nothing removed): the set equality above cannot see a doubled
        # line — two elements with one utterance_id are one key. The counts can.
        assert len(probe["spoken"]) == len(spoken), (
            "T-0154 B10: a reconnect replay does not change the utterance count; the server "
            f"published {len(spoken)} utterances and the reloaded page shows "
            f"{len(probe['spoken'])} [data-utterance-id] elements: "
            f"{[u['utterance_id'] for u in probe['spoken']]}"
        )
        expected_articles = content_turns(live.wire)
        assert len(expected_articles) == 1, (
            f"premise: the wire's content names one turn; it names {expected_articles}"
        )
        assert articles_after == expected_articles, (
            "T-0154 B2/B10: after the reload the thread holds exactly one article, for the one "
            f"turn with content ({expected_articles}); it holds {articles_after!r}"
        )
        assert articles_before == articles_after and len(before["spoken"]) == len(
            probe["spoken"]
        ), (
            "T-0154 B10: the reload changes neither the article count nor the utterance count: "
            f"articles {articles_before!r} -> {articles_after!r}, utterances "
            f"{len(before['spoken'])} -> {len(probe['spoken'])}"
        )
        assert any(
            isinstance(m, AstraSpoken) and m.utterance_id in spoken for m in live.hello_replies[-1]
        ), (
            "anti-vacuity (T-0154 B10): the reconnect's hello reply re-sent the turn's spoken "
            "lines, so a page that appended them again would have doubled them; it carried "
            f"{[getattr(m, 'type', m) for m in live.hello_replies[-1]]}"
        )
        assert QUESTION in " ".join(str(probe["transcript"] or "").split()), (
            "criterion 14: the user's question is back in the transcript pane; it reads "
            f"{probe['transcript']!r}"
        )
        assert probe["errors"] in (None, ""), (
            "criterion 6, negative: a replay that fits the journal puts no error strip up; the "
            f"page reads {probe['errors']!r}"
        )

        # --- the chip, and that it came from the server (criterion 1) --------------------------
        replayed_states = [m for m in live.hello_replies[-1] if isinstance(m, AstraTurnState)]
        assert replayed_states, (
            "criterion 1: the reconnect's hello reply carries the turn state; it carried "
            f"{[getattr(m, 'type', m) for m in live.hello_replies[-1]]}"
        )
        assert chip in {m.state.name for m in replayed_states} | {live.session.state.state.name}, (
            f"criterion 1: the chip ({chip}) is a state the server sent, not a page-side guess; "
            f"the hello reply carried {[m.state.name for m in replayed_states]}"
        )

        # --- and the thought never leaves the conductor (criterion 4) --------------------------
        thoughts = [
            e
            for e in live.machine.events(E.LlmCommit)
            if e.segment == "thought" and "SECRETTHOUGHT" in e.text
        ]
        assert thoughts, (
            "the turn machine never saw the thought commit, so the scans below prove nothing; "
            f"it saw {[(e.segment, e.text[:40]) for e in live.machine.events(E.LlmCommit)]}"
        )
        # Narrowed by T-0180 (B9 shape, ruling R-1; T-0037 criterion 4's amendment note): the thought
        # may appear on the wire only in an astra.thought's text and in the DOM only inside
        # [data-model-thoughts]; and (anti-vacuity) it does appear there: the reload is inside the
        # grace, so the hello reply replays the turn's thoughts (A4) and the page shows them (C1).
        outside, inside = second_dom
        assert "SECRETTHOUGHT" not in outside, (
            "criterion 4 (narrowed by T-0180): the assistant's thinking reached the reloaded "
            "page's DOM outside [data-model-thoughts]"
        )
        leaked = thought_leaks(live.wire, "SECRETTHOUGHT")
        assert not leaked, (
            "criterion 4 (narrowed by T-0180): the thought went out on the wire outside "
            f"astra.thought's text: {leaked}"
        )
        assert any(SECRET_THOUGHT in text for text in thought_texts(live.hello_replies[-1])), (
            "T-0180 A4 anti-vacuity: the reload's hello reply replayed no astra.thought with the "
            f"committed thought; it carried {[getattr(m, 'type', m) for m in live.hello_replies[-1]]}"
        )
        assert SECRET_THOUGHT in inside, (
            "T-0180 B9 anti-vacuity (DOM): the replayed thought is not in the reloaded page's "
            f"[data-model-thoughts]; the disclosures read {inside[:200]!r}"
        )
    finally:
        if asking is not None:
            with contextlib.suppress(Exception):
                await asking
        hud.shutdown()
        await live.aclose()


@requires_chrome
async def test_t0154_a_socket_only_reconnect_changes_neither_articles_nor_utterances(tmp_path):
    """T-0154 B10 (ruling O-B (2)): a reconnect *without* a page reload is idempotent too.

    The reload row above starts from a fresh DOM, so a thread that appends every replayed line
    again passes it. This row keeps the page and drops only its socket: the server closes the
    page's ``HudConnection`` (captured by wrapping the public ``HudServer.open``, which the socket
    handler calls once per connection), ``hud/src/ws.js`` reconnects after its 125–250 ms backoff,
    says ``astra.hello`` again and receives the T-0037 replay of the open turn — every
    ``astra.spoken`` and the ``astra.transcript`` it has already rendered, with ``replay: true``.
    The thread must then hold the same articles and the same utterance elements as before.

    Premises: the turn is still SPEAKING (the journal holds it), one more hello and only one, the
    gap from close to hello under ``RECONNECT_GRACE_MS`` and no ``ClientDisconnected`` (a
    reconnect, not a departure). Anti-vacuity: the hello reply carried the turn's spoken lines,
    so a doubling thread would have doubled them. Expected values come from the wire.
    """
    script = [thought(SECRET_THOUGHT), answer(FIRST), answer(SECOND)]
    live = Live(tmp_path, script, ms_per_word=400)
    opened: list[HudConnection] = []
    real_open = live.server.open

    def tapped_open() -> HudConnection:
        connection = real_open()
        opened.append(connection)
        return connection

    live.server.open = tapped_open  # type: ignore[method-assign]
    await live.start()
    hud, hud_port = serve_hud()
    url = page_url(hud_port, live.server.url)
    asking = None
    try:
        async with open_page(url) as page:
            asking = live.asking(QUESTION)
            await live.until(
                lambda: len(live.spoken()) >= 2,
                "the turn never published its two utterances",
                timeout_s=SPEAKING_TIMEOUT_S,
            )
            await page.until(
                "document.querySelectorAll('[data-utterance-id]').length >= 2",
                "both utterances on the page before the socket is dropped",
            )
            before = await page.probe()
            articles_before = await page.evaluate(ARTICLES_JS)
            hellos_before = live.hellos
            assert len(opened) == 1, f"premise: one connection so far; opened {len(opened)}"
            assert live.session.state.state is TurnState.SPEAKING, (
                "premise: the socket must drop mid-turn; the session is in "
                f"{live.session.state.state}"
            )

            # --- the socket-only reconnect: the server drops the connection, the page stays ---
            started = time.monotonic()
            opened[0].close(reason="T-0154 B10 test: socket-only reconnect")
            await live.until(
                lambda: live.hellos >= hellos_before + 1,
                "the page never greeted the server again after its socket was closed",
                timeout_s=RELOAD_TIMEOUT_S,
            )
            gap_s = time.monotonic() - started
            await page.connected()
            await asyncio.sleep(0.5)  # the replay burst is a handful of frames; let it land
            after = await page.probe()
            articles_after = await page.evaluate(ARTICLES_JS)

        assert live.hellos == hellos_before + 1, (
            f"premise: exactly one more hello after the socket dropped; hellos={live.hellos}"
        )
        assert len(opened) == 2, f"premise: one new connection; opened {len(opened)}"
        assert gap_s < RECONNECT_GRACE_MS / 1000, (
            f"premise: the reconnect took {gap_s:.2f} s, past the {RECONNECT_GRACE_MS} ms grace"
        )
        assert live.machine.events(E.ClientDisconnected) == [], (
            "T-0028 criterion 13: a reconnect inside the grace is not a departure; the machine "
            f"saw {live.machine.events(E.ClientDisconnected)}"
        )
        spoken = {m.utterance_id for m in live.wire if isinstance(m, AstraSpoken)}
        assert len(spoken) >= 2, f"premise: two spoken lines were published: {sorted(spoken)}"
        assert any(
            isinstance(m, AstraSpoken) and m.utterance_id in spoken for m in live.hello_replies[-1]
        ), (
            "anti-vacuity: the reconnect's hello reply re-sent the turn's spoken lines; it carried "
            f"{[getattr(m, 'type', m) for m in live.hello_replies[-1]]}"
        )
        expected_articles = content_turns(live.wire)
        assert len(expected_articles) == 1, (
            f"premise: the wire's content names one turn; it names {expected_articles}"
        )
        assert articles_before == expected_articles, (
            "T-0154 B2: before the socket dropped the thread holds one article for the one turn "
            f"with content ({expected_articles}); it holds {articles_before!r}"
        )
        assert articles_after == articles_before, (
            "T-0154 B10: a socket-only reconnect replay changes the article count or order: "
            f"{articles_before!r} -> {articles_after!r}"
        )
        ids_before = [u["utterance_id"] for u in before["spoken"]]
        ids_after = [u["utterance_id"] for u in after["spoken"]]
        assert len(ids_before) == len(spoken), (
            f"premise: the page showed each published utterance once before: {ids_before}"
        )
        assert ids_after == ids_before, (
            "T-0154 B10: a socket-only reconnect replay changes the utterance elements — the "
            f"published {sorted(spoken)} were shown as {ids_before} and now as {ids_after}"
        )
    finally:
        if asking is not None:
            with contextlib.suppress(Exception):
                await asking
        hud.shutdown()
        await live.aclose()
