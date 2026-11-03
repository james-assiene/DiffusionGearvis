"""T-0227 group B: the page follows the capture interval, and the panel's two new rows reach the session (Chrome half).

``docs/tasks/T-0227.md`` B-1 .. B-4 and *Tests to write* "Integration (Playwright, loopback), B". The node half
(the pipeline's setter, the state reader, the tick) is ``tests/hud/frame_interval.test.mjs``.

**Activated** on ``T-0227-a-server`` once group A was approved and group B was on main (merge 38fa1044). It was
pending until then: the panel rows need group A's catalog rows (``max_frames_per_turn``, ``frame_interval_ms``) and
the session applying them; the B-4 row had no ``TIP`` literal.

**The rig** is T-0213's (``tests/integration/test_hud_settings.py``): ``Live`` (a real ``Session`` + real
``HudServer(port=0)`` over the fakes, fake LLM), the page served from ``hud/`` over loopback with
``?source=canvas`` (``page_url``; ``canvas`` and ``synthetic`` are the same synthetic provider in ``main.js``, the
task's ``?source=synthetic``), headless Chrome over CDP with ``Network.enable``. The init script is T-0213's
recorder plus a frame log: every ``astra.frame`` the page sends is kept as ``[source, ts_ms]`` (the page clock,
B-3's measure); every other message is kept whole.

**Readings stated (the task leaves them open):**

- "The page sends about one frame per 1000 ms" is checked with B-3's own bounds at 1000 ms: every gap >= 1000 and
  the median <= 1300.
- A window "over the next N s" starts at the page's ``Date.now()`` read right after the row's source badge turned
  ``ui`` (the echo reached the panel), and holds the frames whose ``ts_ms`` is in ``[t0, t0 + N s]``.
- ``test_max_images_change_bounds_the_next_request``: the page's own capture is stopped first (Stop sharing): the
  synthetic pattern moves, so its frames would add scene changes of their own. Three frames with distinct phashes
  (>= 20 bits apart from each other and from every phash the page sent; the scene threshold is 10) are then posted
  on the page's socket, so they go through ``HudServer.route`` -> ``BrowserFrameSource.ingest`` -> ``SceneChanged``
  exactly as a capture's would: one at IDLE (the onset frame), two while the turn is LISTENING. A premise turn at
  the default (3) carries 3 current-turn images. At cap 2 the row asserts **exactly** 2 (the task says "at most 2";
  with three distinct frames and the T-0059 de-duplicate-then-cap rule, fewer than 2 is also a bug). "Current-turn
  images" are the ``ImagePart`` s of the final request's last message (docs/01 D9: the turn's images lead its
  message; history images sit in earlier messages).

Waits are on conditions with generous ceilings (T-0218), never a sleep for an effect; ``QUIET_S`` windows only prove
nothing more happened. No number here is a latency claim (task Regime: functional thresholds on the page clock;
headless Chrome, synthetic source, loopback, fake LLM, n = 1 window per interval, the verifier reruns x3). Browser
rows run only with ``ASTRA_BROWSER_TESTS=1``.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import random
import statistics
from typing import Any

import pytest

from local_astra.contracts.llm import ImagePart
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.integration.test_hud_rehearsal import Live, Page, answer
from tests.integration.test_hud_settings import (
    PAGE_WAIT_S,
    QUIET_S,
    browser,
    control_js,
    hello_state,
    init_js,
    open_panel,
    settings_page,
    settle,
    wait_live,
    wire_item,
    wire_states,
)

pytestmark = [pytest.mark.integration]

SET_TYPE = "astra.settings.set"
KEY_INTERVAL = "frame_interval_ms"
KEY_FRAMES = "max_frames_per_turn"

#: a cadence window's own ceiling: the window itself plus the generous T-0218 slack
WINDOW_SLACK_S = 15.0


# --- the recorder -----------------------------------------------------------------------------------------------


def frame_init_js() -> str:
    """T-0213's :func:`init_js` (``block=False``: the server's own states reach the page) that also keeps
    ``[source, ts_ms]`` of every ``astra.frame`` the page sends in ``frames`` and every other sent message in
    ``msgs``."""
    out = init_js(block=False)
    for old, new in (
        ("sent: [], ", "sent: [], msgs: [], frames: [], "),
        (
            "rec.sent.push(m && m.type); }",
            "rec.sent.push(m && m.type); if (m && m.type === 'astra.frame') rec.frames.push([m.source, m.ts_ms]);"
            " else rec.msgs.push(m); }",
        ),
    ):
        assert old in out, f"rig: INIT_JS no longer holds {old!r}"
        out = out.replace(old, new)
    return out


FRAMES_JS = (
    "(() => { const r = window.__t0213; return r && r.frames ? r.frames.slice() : null; })()"
)
MSGS_JS = "(() => { const r = window.__t0213; return r && r.msgs ? r.msgs.slice() : null; })()"


async def page_frames(page: Page) -> list[tuple[str, int]]:
    got = await page.evaluate(FRAMES_JS)
    assert isinstance(got, list), f"rig: the frame recorder is missing: {got!r}"
    return [(str(s), int(t)) for s, t in got]


async def sets_sent(page: Page) -> list[dict[str, Any]]:
    got = await page.evaluate(MSGS_JS)
    assert isinstance(got, list), f"rig: the message recorder is missing: {got!r}"
    return [m for m in got if isinstance(m, dict) and m.get("type") == SET_TYPE]


def screen_ts(frames: list[tuple[str, int]]) -> list[int]:
    return [t for s, t in frames if s == "screen"]


def gaps(ts: list[int]) -> list[int]:
    return [b - a for a, b in itertools.pairwise(ts)]


async def page_now(page: Page) -> int:
    got = await page.evaluate("Date.now()")
    assert isinstance(got, (int, float)), got
    return int(got)


async def window(page: Page, t0: int, span_ms: int) -> list[int]:
    """The screen frames with ``t0 <= ts_ms <= t0 + span_ms``, once the page has sent one past the window's end."""
    end = t0 + span_ms
    await settle(
        page,
        "(() => { const r = window.__t0213; return Boolean(r) && r.frames.some((f) => f[0] === 'screen' && f[1] > "
        f"{end}); }})()",
        timeout_s=span_ms / 1000 + WINDOW_SLACK_S,
    )
    return [t for t in screen_ts(await page_frames(page)) if t0 <= t <= end]


async def choose(page: Page, live: Live, key: str, value: int) -> int:
    """Choose ``value`` in ``key``'s panel select; wait for the echo (source ``ui`` on the row and the value in force
    on the wire); return the page clock at that moment."""
    sets_before = len(await sets_sent(page))
    did = await page.evaluate(control_js(key, value))
    assert did == f"selected {value}", f"B-4 / A-2: the {key} row's select offers {value}: {did!r}"
    await page.until(
        "(() => { const s = document.querySelector('[data-setting-key="
        + json.dumps(key)
        + "] [data-setting-source]');"
        " return Boolean(s) && s.getAttribute('data-setting-source') === 'ui' && "
        "(document.querySelector('[data-setting-key="
        + json.dumps(key)
        + "] [data-setting-value]') || {textContent: ''})"
        ".textContent.includes(" + json.dumps(str(value)) + "); })()",
        f"the echo of {key}={value} shows source=ui and the value on its row",
        timeout_s=PAGE_WAIT_S,
    )
    t0 = await page_now(page)
    await wait_live(
        live,
        lambda: bool(wire_states(live)) and wire_item(wire_states(live)[-1], key).value == value,
        f"the session applied {key}={value}",
    )
    await asyncio.sleep(QUIET_S)  # nothing-more window: a second set would show here
    sets = (await sets_sent(page))[sets_before:]
    assert len(sets) == 1, f"one choice, one astra.settings.set; the page sent {sets}"
    changed = sets[0].get("changes")
    assert changed == {key: value}, f"the set carries {{{key!r}: {value}}}; got {sets[0]}"
    assert type(changed[key]) is int, (
        f"T-0213 R-26 (a): the int is sent as a JSON number, not {type(changed[key]).__name__}"
    )
    item = wire_item(wire_states(live)[-1], key)
    assert item.source == "ui", f"A-9: after a set the source is ui; {item}"
    return t0


def check_cadence(ts: list[int], interval: int, *, min_gaps: int, what: str) -> None:
    g = gaps(ts)
    assert len(g) >= min_gaps, (
        f"{what}: only {len(ts)} frames in the window, so the cadence cannot be judged (gaps {g})"
    )
    assert min(g) >= interval, (
        f"{what}: every gap >= {interval} ms (B-3: the throttle is strict); gaps {g}"
    )
    med = statistics.median(g)
    assert med <= 1.3 * interval, f"{what}: median gap {med} ms <= {1.3 * interval} (B-3); gaps {g}"
    # Review finding (T-0227 B): the median cannot see a tick-quantisation rate cut (gaps ~1x and ~2x, the ~1x ones in
    # the majority); the mean gap, the same 1.3x bound applied to the rate, can (tests/hud/frame_interval.test.mjs).
    avg = statistics.fmean(g)
    assert avg <= 1.3 * interval, (
        f"{what}: mean gap {avg:.1f} ms <= {1.3 * interval} (B-3: the rate is not cut); gaps {g}"
    )


# --- B-1/B-2/B-3: the measured cadence follows the panel --------------------------------------------------------


@browser
async def test_interval_change_changes_the_measured_cadence(tmp_path):
    """B-1/B-2/B-3 (Tests to write): at the default the page sends about one frame per 1000 ms; choosing 250 in the
    panel sends one ``astra.settings.set {"frame_interval_ms": 250}``, the echo shows ``source=ui``, and over the next
    5 s of page clock every ``ts_ms`` gap is >= 250 with a median <= 325; choosing 2000, over the next 8 s every gap
    is >= 2000 with a median <= 2600."""
    async with settings_page(tmp_path, block=False, script=frame_init_js()) as (
        live,
        page,
        _chrome,
    ):
        state = await hello_state(live)
        item = wire_item(state, KEY_INTERVAL)
        assert (item.value, item.changeable) == (1000, True), (
            f"A-9 premise: the hello state carries frame_interval_ms = 1000, changeable; {item}"
        )
        t_start = await page_now(page)
        base = await window(page, t_start, 4000)
        check_cadence(base, 1000, min_gaps=2, what="default (1000 ms)")

        await open_panel(page, len(state.items))
        t0 = await choose(page, live, KEY_INTERVAL, 250)
        fast = await window(page, t0, 5000)
        check_cadence(fast, 250, min_gaps=10, what="after choosing 250 ms")

        t1 = await choose(page, live, KEY_INTERVAL, 2000)
        slow = await window(page, t1, 8000)
        check_cadence(slow, 2000, min_gaps=3, what="after choosing 2000 ms")


# --- A-5 through the page: the next request's image count follows the panel ------------------------------------

#: a few bytes the frame source accepts (base64 of a JPEG SOI plus padding; nothing here decodes it)
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"t0227" * 16 + b"\xff\xd9").decode()

POST_FRAME_JS = """(() => {
  const s = window.__t0213 && window.__t0213.sockets.at(-1);
  if (!s || s.readyState !== 1) return 'no open socket';
  s.send(JSON.stringify({type: 'astra.frame', source: 'screen', ts_ms: Date.now(), width: 640, height: 400,
    jpeg_b64: __B64__, phash: __PHASH__}));
  return 'sent';
})()"""

#: long enough that two frames can be posted while the turn is LISTENING (16 words x 150 ms)
QUESTION = (
    "which of these three screens shows the later date and the larger number please tell me now"
)


def _distinct_phashes(avoid: list[str], n: int, seed: int) -> list[str]:
    """``n`` 16-hex-digit phashes, each >= 20 bits from every other and from ``avoid`` (scene threshold: 10)."""
    rng = random.Random(seed)
    taken = [int(p, 16) for p in avoid]
    out: list[str] = []
    while len(out) < n:
        cand = rng.getrandbits(64)
        if all(bin(cand ^ t).count("1") >= 20 for t in taken):
            taken.append(cand)
            out.append(f"{cand:016x}")
    return out


async def post_frame(page: Page, live: Live, phash: str) -> None:
    before = len(live.sink.accepted)
    sent = await page.evaluate(
        POST_FRAME_JS.replace("__B64__", json.dumps(JPEG_B64)).replace(
            "__PHASH__", json.dumps(phash)
        )
    )
    assert sent == "sent", f"rig: could not post a frame on the page's socket: {sent!r}"
    await wait_live(
        live,
        lambda: len(live.sink.accepted) > before and live.sink.accepted[-1].phash == phash,
        f"rig: the server stored the posted frame {phash}",
    )


async def three_screen_turn(page: Page, live: Live, phashes: list[str]) -> Any:
    """Onset frame at IDLE, two more while LISTENING; returns the turn's final request."""
    assert live.session.state.state is TurnState.IDLE, "premise: the turn starts at IDLE"
    n_requests = len(live.llm.gen_prompts)
    await post_frame(page, live, phashes[0])
    asking = asyncio.ensure_future(live.ask(QUESTION, word_ms=150))
    try:
        await wait_live(
            live,
            lambda: live.session.state.state is TurnState.LISTENING,
            "premise: the spoken question opened a LISTENING turn",
        )
        await post_frame(page, live, phashes[1])
        await post_frame(page, live, phashes[2])
        assert live.session.state.state is TurnState.LISTENING, (
            "rig: both scene changes were posted while the user was still speaking"
        )
    finally:
        await asking
    await live.idle()
    await wait_live(
        live, lambda: len(live.llm.gen_prompts) > n_requests, "the turn sent its request"
    )
    _rid, request = live.llm.gen_prompts[-1]
    return request


def current_images(request: Any) -> list[ImagePart]:
    return [p for p in request.messages[-1].content if isinstance(p, ImagePart)]


@browser
async def test_max_images_change_bounds_the_next_request(tmp_path):
    """A-5 / Tests to write: premise, at the default a turn that sees 3 distinct screens sends 3 current-turn images;
    choosing 1 in the panel, the next such turn sends exactly 1; choosing 2, the next sends 2 (stated: exactly 2)."""
    async with settings_page(
        tmp_path,
        block=False,
        script=frame_init_js(),
        make_live=lambda p: Live(p, [answer("Fine. ")], ms_per_word=60),
    ) as (live, page, _chrome):
        state = await hello_state(live)
        item = wire_item(state, KEY_FRAMES)
        assert (item.value, item.changeable) == (3, True), (
            f"A-9 premise: the hello state carries max_frames_per_turn = 3, changeable; {item}"
        )
        # Stop the page's own (moving) capture, then wait until its last frame has landed.
        await settle(
            page,
            "(() => { const r = window.__t0213; return Boolean(r) && r.frames.length > 0; })()",
        )
        stopped = await page.evaluate(
            "(() => { const b = document.getElementById('mute-button'); if (!b || b.disabled) return 'no capture';"
            " b.click(); return 'stopped'; })()"
        )
        assert stopped == "stopped", f"rig: Stop sharing stops the synthetic capture: {stopped!r}"
        n_page = len(await page_frames(page))
        await settle(
            page,
            "(() => document.getElementById('mute-button').disabled)()",
        )
        await asyncio.sleep(QUIET_S)
        assert len(await page_frames(page)) == n_page, (
            "rig: the stopped capture sent no more frames"
        )
        pool = _distinct_phashes([f.phash for f in live.sink.accepted], 9, seed=0x0227)

        request = await three_screen_turn(page, live, pool[0:3])
        assert len(current_images(request)) == 3, (
            "premise (T-0059 at the default 3): three distinct screens give three current-turn images; got "
            f"{[p.phash for p in current_images(request)]}"
        )

        await open_panel(page, len(state.items))
        await choose(page, live, KEY_FRAMES, 1)
        request = await three_screen_turn(page, live, pool[3:6])
        got = [p.phash for p in current_images(request)]
        assert len(got) == 1, (
            f"A-5: with the cap at 1 the next turn carries exactly 1 current-turn image; {got}"
        )

        await choose(page, live, KEY_FRAMES, 2)
        request = await three_screen_turn(page, live, pool[6:9])
        got = [p.phash for p in current_images(request)]
        assert len(got) == 2, (
            f"A-5: with the cap at 2 the next three-screen turn carries 2 current-turn images (at most 2, and three "
            f"distinct screens fill both); {got}"
        )
        for p in current_images(request):
            assert p.phash in pool[6:9], (
                f"the images are this turn's own screens; {p.phash} is not in {pool[6:9]}"
            )


# --- B-4: no egress ------------------------------------------------------------------------------------------------


@browser
async def test_no_egress(tmp_path):
    """B-4: over a page load, the capture running, both new rows chosen in the panel (250 ms, then 2 images) and
    a 250 ms cadence window, every request the page makes is to loopback."""
    async with settings_page(tmp_path, block=False, script=frame_init_js()) as (live, page, chrome):
        state = await hello_state(live)
        await open_panel(page, len(state.items))
        t0 = await choose(page, live, KEY_INTERVAL, 250)
        await window(page, t0, 2000)
        await choose(page, live, KEY_FRAMES, 2)
        await asyncio.sleep(QUIET_S)  # nothing-happens window for an off-loopback request
        await asyncio.to_thread(chrome.drain, 1.0)
        requests = list(chrome.requests)
    assert requests, (
        "the CDP request log is empty, so this row would be vacuous (Network.enable before navigation)"
    )
    offenders = [
        u
        for u in requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert not offenders, f"B-4: zero non-loopback requests; got {offenders}"
    assert not [
        u for u in requests if u.split("://", 1)[0] not in {"http", "ws", "data", "blob"}
    ], requests


# --- B-4: frozen page files ----------------------------------------------------------------------------------------

#: Closed at group B's merge M = 38fa1044 (.claude/rules/tests.md: pin both ends): ``BASE = merge-base M^1 M^2``
#: (5b8afd19) and ``TIP = M^2`` (fd04b03b, the branch tip; its range holds B's implementation 3b26998d). Was
#: ``BASE = 0a9882a3`` (the branch point) and ``TIP = None`` while pending.
BASE = "5b8afd1967b1263037a829a018e965b4602e7ef3"
TIP: str | None = "fd04b03b293bba58caa1fe90a78e94a0933917cb"
#: B-4's byte-identical list: the settings panel, the socket, the vendor tree (absent, stays absent), the wire,
#: the assist gate (still 1000 / MAX_FPS), and the T-0008 throttle rows that must pass unedited.
FROZEN = (
    "hud/src/settings",
    "hud/src/ws.js",
    "hud/vendor",
    "hud/src/protocol.js",
    "hud/src/assists/gate.js",
    "hud/src/assists/index.js",
    "tests/hud/throttle.test.mjs",
)


def test_b4_frozen_page_files_are_byte_identical() -> None:
    """B-4: the frozen page files (and ``throttle.test.mjs``) are byte-identical over ``BASE..TIP``."""
    assert TIP is not None, (
        "T-0227 B-4: the TIP literal (group B's implementation commit) does not exist yet"
    )
    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE[:8]} is not an ancestor of {TIP[:8]}"
    assert BASE != TIP, "premise: the range is not empty"
    changed = gr.changed(BASE, TIP, *FROZEN)
    assert changed == [], (
        f"T-0227 B-4: frozen page files differ in {BASE[:8]}..{TIP[:8]}: {changed}"
    )
