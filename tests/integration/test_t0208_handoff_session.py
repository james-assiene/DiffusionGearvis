"""T-0208 group B, criteria B4-B7: the handoff scenarios H1-H3 through a real ``Session`` in live mode, the user's
shared image arriving through the production ingress, and the real web-mode executor on the scenario pages
(docs/tasks/T-0208.md, "Tests to write", ``tests/integration/test_t0208_handoff_session.py``).

**Gate.** T-0111c's autouse ``_browser_layer`` (imported through the T-0205 browser module): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; ``make browser-handoff`` (B8) runs it.

**The rig** (B4). ``StartPageServer(static_dir=SCENARIO_DIR)`` on loopback and T-0205 A7's web-mode
``BrowserToolExecutor`` (``web_executor``: ``web=True``, one allowed origin, ``echo="model"``) with the deny-all
resolver and connector (``DenyNet``), opened at the row's ``start_page``; T-0111g's ``Watch`` around it. The session
is T-0177's ``vision_rig`` with ``ingress=True`` (one real ``BrowserFrameSource`` is both the session's perception and
the ``HudServer``'s frame sink) over a live kit on the ``Watch``, the web loop budget of 20 steps
(``tsl.loop_policy(20)``) and the screen label on (``live_feedback(screen_label=True)``, cli's default). The row's
image is posted once as an ``astra.frame`` (source ``screen``, 1280x800) through the server's page, then
``astra.vision.select {source: "shared"}`` over the wire, then one ``UserText(request)``. The scripted fake LLM holds
each step; before releasing step k+1 the test computes its call from the current page's ``data-scn`` boxes
(test-side; a scroll first when the target is outside the viewport), so the script is the test's and the page is the
browser's. The ``data-scn`` ids and golden paths are T-0205's (D4, D5, D6).

**Q-2 (answered here).** ``BrowserFrameSource.ingest`` base64-decodes ``jpeg_b64`` and stores those bytes as
``Frame.jpeg``; it does not re-encode, and ``PromptBuilder`` writes ``data:image/jpeg;base64,<bytes>`` from
``Frame.jpeg`` unchanged. So the byte-equality rows compare the request bodies against **what the session stored for
that frame** (``rig.perception.get_frame(Source.SCREEN)``, the source the session reads), as the task's notes ask; each
row also asserts, as a named premise, that the stored bytes equal the committed file's bytes (if the ingress ever
re-encodes, that premise goes red alone and the stored-bytes rows still mean what they say).

**Reading the requests.** Every non-caption ``generate`` and every ``prefill`` of turn ``t1`` is read as its wire body
(``_wire``, the real client's ``request_body``). Step 1 is the turn generate ``…-t1-gen-<n>`` (and its prefill and
resends); "later" is every request under a step machine id ``…-t1-gen-s<k>…`` (steps, re-asks, resends, completion
checks). A later image must be one of the executor's own frames (``Watch.frames`` or the post-action
``latest_frame``) and must not be the shared bytes; the shared bytes' base64 appears in no later body.

**Readings this module pins where the task is silent** (reported):

- B4 "step 1 is a non-coordinate action (scroll)": ``scroll`` down, ``executed`` (it moves or not; the start page's
  geometry decides). Later scrolls go up or down toward the next target.
- Step texts: step 1 says one sentence (``SAY``); later steps say nothing (T-0205's ``answer("")``); the last step is a
  no-call answer (``DONE``) except in the checkout row, whose last step is the Checkout click (T-0205 A10's shape).
- The published ``astra.vision.frame`` at step 1 carries one image whose ``jpeg_b64`` is the stored bytes; every
  frame published for a later step has ``source: "browser"``.
- B5 "the drawn steam wand's coordinates": the centroid of the image's red pixels (R >= 170, G <= 90, B <= 90; B2's
  red circle around the snapped wand), read from the committed JPEG with Pillow and mapped to 0-999 on 1280x800. "No
  page event fires": capture-phase listeners on ``window`` for pointer, mouse, click, key, input and wheel events,
  installed test-side before the turn, read after the refusal (positive control: a synthetic ``mousedown`` is counted).
- B6 "traps through the executor" runs the executor directly (T-0205's ``opened``/``Rig``, no session: the trap is a
  page path, the session adds nothing) and judges with ``handoff_tasks.judge``.
- B4 completion check: "Merge T-0208-a2 is an ancestor" is read from the subjects of this checkout's history
  (``git log --format=%s``, a line starting ``Merge T-0208-a2``); after that merge the check fires once with no shared
  byte and a browser fresh frame; before it, no check request is sent.

BASE = 7e52c3aa. Written red ahead of ``handoff_tasks.py`` and the images (31933b4c, pending); activated at group B's
``/implement`` against 80075ec5.

Regime: Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, web mode on loopback, deny-all resolver and
connector, ``ScriptedClock`` for the session, loop 20, screen label on, n = 1.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import json
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import is_caption_prompt
from tests.integration.test_computer_live_session import (
    Watch,
    click,
    descendants,
    drive,
    non_loopback_sockets,
    type_at,
)
from tests.integration.test_t0205_scenarios_browser import (  # noqa: F401  (_browser_layer is the autouse gate)
    BOX_JS,
    STATE_JS,
    DenyNet,
    H,
    W,
    _browser_layer,
    _browser_marker_registered,
    _d6_to_cart,
    _trap_d4_bes878,
    _trap_d5_ridge,
    opened,
    run,
    scn_dir,
    unit,
    vdict,
    web_executor,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import b64, data_url, image_urls, live_feedback
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_session_live import _wire, last_user
from tests.unit.test_start_pages import new_server
from tests.unit.test_t0177_shared_frame import post, select, vision_rig

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.handoff_tasks"
HANDOFF_DIR = REPO / "local_astra" / "pages" / "handoff"
IDS = ("H1", "H2", "H3")
#: T-0177 A7 / B2 and T-0134 A3, typed from the tasks (never imported)
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
L_NOW = "[This screenshot is the browser screen now.]"
Q = '<|"|>'
SCROLL_DOWN = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
SCROLL_UP = f"<|tool_call>call:scroll{{direction:{Q}up{Q}}}<tool_call|>"
SAY = {
    "H1": "That is the steam wand of a Barista Express BES870; I will find the replacement. ",
    "H2": "That is the Trailhead Duo Stove; I will look for it. ",
    "H3": "Sam asked for a large pepperoni from Domino's, for pickup at W 44th St. ",
}
DONE = {
    "H1": "The Steam Wand Assembly for the BES870 is in your cart. ",
    "H2": "The Trailhead Duo Stove is in your cart. ",
    "H3": "A large pepperoni pizza is in your cart for pickup at the W 44th St store. ",
}
#: The T-0205 golden paths (data-scn id, text to type or None), after the step-1 scroll
PLANS: dict[str, list[tuple[str, str | None]]] = {
    "H1": [
        ("q", "Barista Express"),
        ("search", None),
        ("result-bes870", None),
        ("add-steam-wand-assembly", None),
        ("nav-cart", None),
    ],
    "H2": [("filter-1-2-person", None), ("add-trailhead-duo-stove", None), ("nav-cart", None)],
    "H3": [
        ("method-carryout", None),
        ("location", "Times Square"),
        ("find-stores", None),
        ("store-w-44th", None),
        ("size-large", None),
        ("add-pepperoni-pizza", None),
        ("upsell-no-thanks", None),
        ("nav-cart", None),
    ],
}
#: the adversarial step after the hand-back: never requested, never voiced
DEAD = "I will place the order now. "
CHECK_SAYS = "The cart shows the Trailhead Duo Stove. "
EVENTS = (
    "pointerdown",
    "pointerup",
    "mousedown",
    "mouseup",
    "click",
    "dblclick",
    "keydown",
    "keyup",
    "input",
    "wheel",
)
LISTEN_JS = (
    "(evs) => { window.__t0208_events = []; for (const e of evs) window.addEventListener(e, (ev) => "
    "window.__t0208_events.push(ev.type), true); return evs.length; }"
)


# =====================================================================================================
# the table and the image
# =====================================================================================================


def ho() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0208 B1/B3: {MODULE} does not exist yet") from exc
        raise


def row(hid: str) -> Any:
    rows = {r.handoff_id: r for r in ho().HANDOFFS}
    assert hid in rows, f"T-0208 B1: no {hid} row"
    return rows[hid]


def image_bytes(hid: str) -> bytes:
    r = row(hid)
    p = HANDOFF_DIR / r.image
    assert p.is_file(), f"T-0208 B2: {p.relative_to(REPO)} does not exist yet"
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == r.image_sha256, f"T-0208 B2: {r.image} sha mismatch"
    return data


def hjudge(hid: str, state: Any, voiced: str, shown: str = "") -> Any:
    return ho().judge(hid, state, voiced, shown)


def frame_message(hid: str, data: bytes) -> AstraFrame:
    """The ``astra.frame`` the HUD page posts for the shared image (16-hex phash, screen source, 1280x800)."""
    return AstraFrame(
        source=Source.SCREEN,
        ts_ms=0,
        width=W,
        height=H,
        jpeg_b64=base64.b64encode(data).decode(),
        phash=hashlib.sha256(data).hexdigest()[:16],
        title=row(hid).image,
        app="image viewer",
    )


def red_centroid(data: bytes) -> tuple[int, int]:
    """B5: the 0-999 point at the centroid of the image's red pixels (the circle around the snapped wand)."""
    import io

    from PIL import Image

    from local_astra.conductor.tool_executor import to_pixels

    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        px = rgb.load()
        assert px is not None
        xs = ys = n = 0
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                r, g, b = px[x, y]
                if r >= 170 and g <= 90 and b <= 90:
                    xs += x
                    ys += y
                    n += 1
    assert n >= 50, f"T-0208 B2/B5: no red circle drawn in the H1 image ({n} red samples)"
    cx, cy = xs / n, ys / n
    ux = min(range(1000), key=lambda c: abs(to_pixels(c, w) - cx))
    uy = min(range(1000), key=lambda c: abs(to_pixels(c, h) - cy))
    return ux, uy


def a2_merged() -> bool:
    """Whether this checkout's history holds a ``Merge T-0208-a2`` commit (B4's merge-aware row)."""
    from tests import _git_range as gr

    proc = gr._git("log", "--format=%s")
    assert proc.returncode == 0, proc.stderr
    return any(s.startswith("Merge T-0208-a2") for s in proc.stdout.splitlines())


# =====================================================================================================
# the session rig and the driver
# =====================================================================================================


@dataclass
class Turn:
    """What one handoff turn left behind, as plain data."""

    hid: str
    stored: bytes
    step1_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    later_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    check_bodies: list[dict[str, Any]] = field(default_factory=list)
    browser_jpegs: set[bytes] = field(default_factory=set)
    vision_frames: list[dict[str, Any]] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)
    cards: dict[str, Any] = field(default_factory=dict)
    state: Any = None
    url: str = ""
    voiced: str = ""
    spoken: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    clip_kinds: list[str] = field(default_factory=list)
    hand_backs: list[Any] = field(default_factory=list)
    steps: int = 0
    events_after_step1: list[str] | None = None
    control_events: list[str] | None = None
    step2_body: dict[str, Any] | None = None
    net: tuple[list[Any], list[Any]] = ([], [])
    sockets: list[str] = field(default_factory=list)


def results_of(rig: Any) -> list[Any]:
    return [e for e in rig.machine.events(E.ToolResult) if e.call_id.startswith("t1-call")]


async def call_for(ex: Any, sid: str, text: str | None = None) -> str:
    """The next step's call: a scroll toward ``sid`` while it is outside the viewport, else a click (or a type)."""
    b = await ex.driver.page.evaluate(BOX_JS, sid)
    assert b is not None, f'T-0205 A3: no data-scn="{sid}" on {ex.driver.page.url}'
    if b["bottom"] > H:
        return SCROLL_DOWN
    if b["top"] < 0:
        return SCROLL_UP
    x, y = unit(b["left"], b["right"], W), unit(b["top"], b["bottom"], H)
    return type_at(x, y, text, False) if text is not None else click(x, y)


async def wait_result(rig: Any, watch: Watch, k: int) -> None:
    await drive(
        rig,
        watch,
        lambda: len(results_of(rig)) >= k and watch.inflight == 0,
        f"T-0208: step {k}'s result never arrived",
    )


async def give_step(rig: Any, watch: Watch, k: int, script: list[Any]) -> None:
    """After result ``k``, script step ``k+1`` and release it; hold the one after."""
    rig.llm.step_scripts = [list(script)]
    rig.llm.release_steps()
    await drive(
        rig, watch, lambda: len(tsl.steps_of(rig)) >= k, f"T-0208: step {k + 1} was never requested"
    )
    rig.llm.hold_steps()


@contextlib.asynccontextmanager
async def session_on(
    tmp_path: Path, hid: str, *, completion_check: bool = False
) -> AsyncIterator[tuple[Any, Watch, Any, DenyNet, str]]:
    """The B4 rig: scenario server, web-mode executor at the row's start page, ``vision_rig`` with ingress."""
    srv = new_server(scn_dir())
    srv.start()
    net = DenyNet()
    saved = tsl.LOOP
    tsl.LOOP = tsl.loop_policy(20)
    try:
        async with web_executor(srv.origin, net) as ex:
            await run(ex.open(srv.origin + row(hid).start_page))
            watch = Watch(ex)
            kw: dict[str, Any] = {
                "live_feedback": live_feedback(screen_label=True, page_extent=False)
            }
            if completion_check:
                kw["completion_check"] = True
            async with vision_rig(
                tmp_path, watch, kit="live", ingress=True, hud_frames=[], session_kwargs=kw
            ) as rig:
                ex._now_ms = lambda: rig.clock.now_ms
                yield rig, watch, ex, net, srv.origin
            assert list(ex.egress_log) == [], ex.egress_log
    finally:
        tsl.LOOP = saved
        srv.stop()
    assert (net.resolves, net.connects) == ([], []), (
        f"T-0208 B7: the proxy seams were called: {net}"
    )


async def handoff_turn(
    tmp_path: Path,
    hid: str,
    *,
    step1: str = SCROLL_DOWN,
    checkout: bool = False,
    completion_check: bool = False,
    listen: bool = False,
    sockets: bool = False,
) -> Turn:
    """One handoff turn (B4): post the image, select shared, ``UserText(request)``, step 1 ``step1``, then the golden
    path computed from the page, then a no-call answer (or, with ``checkout``, the Checkout click)."""
    data = image_bytes(hid)
    async with session_on(tmp_path, hid, completion_check=completion_check) as (
        rig,
        watch,
        ex,
        net,
        _origin,
    ):
        page = ex.driver.page
        if listen:
            assert await page.evaluate(LISTEN_JS, list(EVENTS)) == len(EVENTS)
        await post(rig, frame_message(hid, data))
        stored_frame = await rig.perception.get_frame(Source.SCREEN)
        assert stored_frame is not None, "premise: the session's source holds the posted frame"
        stored = stored_frame.jpeg
        assert stored == data, (
            "Q-2 premise: BrowserFrameSource kept the posted JPEG bytes (it does not re-encode); "
            f"stored {len(stored)} bytes, posted {len(data)}"
        )
        await select(rig, "shared")
        assert rig.session.vision_source == "shared", "premise: the select reached the session"
        if completion_check:
            rig.llm.script_checks([answer(CHECK_SAYS)])
        rig.llm.script = [answer(SAY[hid]), tool(step1)]
        rig.llm.hold_steps()
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=row(hid).request))
        turn = Turn(hid=hid, stored=stored)
        await wait_result(rig, watch, 1)
        if listen:
            turn.events_after_step1 = list(await page.evaluate("window.__t0208_events"))
            await page.evaluate(
                "window.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}))"
            )
            turn.control_events = list(await page.evaluate("window.__t0208_events"))
        plan = list(PLANS[hid]) + ([("checkout", None)] if checkout else [])
        k = 1
        while True:
            await wait_result(rig, watch, k)
            if not plan:
                rig.llm.step_scripts = [[answer(DONE[hid])]]
                rig.llm.release_steps()
                break
            call = await call_for(ex, *plan[0])
            if call not in (SCROLL_DOWN, SCROLL_UP):
                plan.pop(0)
            if checkout and not plan:
                rig.llm.step_scripts = [[answer(""), tool(call)], [answer(DEAD)]]
                rig.llm.release_steps()
                k += 1
                break
            await give_step(rig, watch, k, [answer(""), tool(call)])
            if k == 1:
                turn.step2_body = _wire(tsl.step_request(rig, 2))
            k += 1
            assert k <= 20, "T-0208: the golden path ran past the web budget"
        await drive(
            rig,
            watch,
            lambda: watch.inflight == 0 and rig.state.state is TurnState.IDLE and bool(rig.lines()),
            "T-0208: the turn never ended",
            limit_s=120,
        )
        await rig.advance(2000)
        if sockets:
            turn.sockets = non_loopback_sockets(descendants())
        turn.state = await page.evaluate(STATE_JS)
        turn.url = page.url
        _collect(rig, watch, turn)
        turn.net = (list(net.resolves), list(net.connects))
    turn.extras = dict(rig.record("t1").extras)
    return turn


def _collect(rig: Any, watch: Watch, turn: Turn) -> None:
    llm = rig.llm
    for rid, prompt, _opts in llm.generates:
        if is_caption_prompt(prompt) or "t1-gen" not in rid:
            continue
        body = _wire(prompt)
        (turn.later_bodies if "t1-gen-s" in rid else turn.step1_bodies).append((rid, body))
    for rid, prefix in llm.prefills:
        if "t1-gen" not in rid:
            continue
        body = _wire(prefix)
        (turn.later_bodies if "t1-gen-s" in rid else turn.step1_bodies).append(
            (f"prefill:{rid}", body)
        )
    turn.check_bodies = [b for b in llm.check_bodies]
    turn.browser_jpegs = {f.jpeg for _g, f in watch.frames if f is not None}
    turn.browser_jpegs |= {f.jpeg for f in watch.after if f is not None}
    turn.vision_frames = [
        m.model_dump(mode="json")
        for m in rig.published.messages
        if getattr(m, "type", None) == "astra.vision.frame" and m.turn_id == "t1"
    ]
    turn.results = [(e.call_id, e.status) for e in results_of(rig)]
    turn.outcomes = list(watch.outcomes)
    turn.cards = {cid: rig.tool_calls(cid)[-1] for cid, _ in turn.results if rig.tool_calls(cid)}
    turn.voiced = " ".join(h.assistant_spoken or "" for h in rig.state.history)
    turn.spoken = rig.spoken_sources()
    turn.clip_kinds = [getattr(u, "kind", None) for u in rig.audio.clip_plays()]
    turn.hand_backs = [getattr(e, "hand_back", None) for e in results_of(rig)]
    turn.steps = len(tsl.steps_of(rig))


def decoded_images(body: dict[str, Any]) -> list[bytes]:
    out = []
    for url in image_urls(body):
        assert url.startswith("data:image/jpeg;base64,"), url[:40]
        out.append(base64.b64decode(url.split(",", 1)[1]))
    return out


def assert_one_source(turn: Turn) -> None:
    """B4: step 1 carries exactly the stored shared bytes then the shared label, and no browser bytes; no later body
    carries any shared byte, and every later image is one of the executor's own frames."""
    shared_b64 = b64(turn.stored)
    assert turn.step1_bodies, "premise: the step-1 request was read"
    for rid, body in turn.step1_bodies:
        imgs = decoded_images(body)
        assert imgs == [turn.stored], (
            f"T-0208 B4: {rid} carries {len(imgs)} image(s), not exactly the stored shared frame"
        )
        assert not (set(imgs) & turn.browser_jpegs), f"T-0208 B4: {rid} carries a browser frame"
        if rid.startswith("prefill:"):
            continue
        parts = last_user(body)["content"]
        at = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
        assert at and parts[at[-1]]["image_url"]["url"] == data_url(turn.stored), rid
        nxt = parts[at[-1] + 1] if at[-1] + 1 < len(parts) else {}
        assert nxt == {"type": "text", "text": SHARED_LABEL_SCREEN}, (
            f"T-0208 B4: the part after the shared image in {rid} is {nxt!r}"
        )
    assert turn.later_bodies, "premise: later requests were read"
    for rid, body in turn.later_bodies:
        assert shared_b64 not in json.dumps(body), f"T-0208 B4: {rid} carries the shared bytes"
        for img in decoded_images(body):
            assert img != turn.stored, rid
            assert img in turn.browser_jpegs, (
                f"T-0208 B4: {rid} carries an image that is not a browser frame"
            )
    for body in turn.check_bodies:
        assert shared_b64 not in json.dumps(body), (
            "T-0208 B4: a completion check carries the shared bytes"
        )
    by_step: dict[int, list[dict[str, Any]]] = {}
    for m in turn.vision_frames:
        by_step.setdefault(int(m["step"]), []).append(m)
    assert 1 in by_step, f"T-0208 B4: no astra.vision.frame at step 1; {sorted(by_step)}"
    for m in by_step[1]:
        assert m["source"] == "shared" and not m["blind"], m.get("source")
        assert [i.get("jpeg_b64") for i in m["images"]] == [shared_b64], (
            "step 1 shows the shared bytes"
        )
    later = [m for s, ms in by_step.items() if s >= 2 for m in ms]
    assert later, "premise: frames were published after step 1"
    assert all(m["source"] == "browser" for m in later), [(m["step"], m["source"]) for m in later]


_TURNS: dict[tuple[str, str], Turn] = {}


async def golden_turn(tmp_path: Path, hid: str) -> Turn:
    """The B4 golden turn for ``hid``, run once per process and shared by the rows that only read it."""
    key = ("golden", hid)
    if key not in _TURNS:
        _TURNS[key] = await handoff_turn(tmp_path, hid)
    return _TURNS[key]


# =====================================================================================================
# B4: the golden handoffs
# =====================================================================================================


@pytest.mark.parametrize("hid", IDS)
async def test_handoff_golden(tmp_path: Path, hid: str) -> None:
    """B4: step 1 (the shared image) is an executed ``scroll``; then the base scenario's golden path through the
    executor, every action executed; the turn ends with a no-call answer and ``judge(hid, __state, voiced)``
    succeeds with no flag. The turn line says ``vision_source: "shared"``."""
    turn = await golden_turn(tmp_path, hid)
    assert turn.results and turn.results[0] == ("t1-call", "executed"), turn.results[:1]
    assert turn.outcomes and "scroll" in str(turn.outcomes[0].action).lower(), turn.outcomes[:1]
    assert [s for _c, s in turn.results] == ["executed"] * len(turn.results), turn.results
    assert isinstance(turn.state, dict), turn.state
    v = hjudge(hid, turn.state, turn.voiced)
    assert v.success is True, (hid, turn.state, vdict(v), [o.action for o in turn.outcomes])
    assert not (v.trap or v.checkout_pressed or v.secret_typed or v.over_action), vdict(v)
    assert turn.extras.get("vision_source") == "shared", turn.extras.get("vision_source")
    assert turn.net == ([], []), turn.net


@pytest.mark.parametrize("hid", IDS)
async def test_no_shared_bytes_after_step1(tmp_path: Path, hid: str) -> None:
    """B4 (one source): step 1's body carries exactly the bytes the session stored for the posted frame, then
    ``SHARED_LABEL_SCREEN``, and no browser bytes; no later body (step, re-ask, resend, completion check) carries any
    shared byte and every later image is an executor frame; the published ``astra.vision.frame`` is ``shared`` at
    step 1 and ``browser`` from step 2."""
    turn = await golden_turn(tmp_path, hid)
    assert_one_source(turn)


async def test_h1_hand_back_at_checkout(tmp_path: Path) -> None:
    """B4 checkout (T-0192, T-0205 A10): H1 continued to the cart's Checkout click ends ``handed_back``: the click
    carries ``resolved["checkout"]``, its ``ToolResult`` ``hand_back == "checkout"``, one ``at_checkout`` clip, no
    step request after it (step requests == results - 1), the dead step never voiced; one source throughout."""
    turn = await handoff_turn(tmp_path, "H1", checkout=True)
    assert turn.url.endswith("/breville/checkout.html"), turn.url
    assert [s for _c, s in turn.results] == ["executed"] * len(turn.results), turn.results
    assert turn.outcomes[-1].resolved.get("checkout") is True, turn.outcomes[-1]
    assert turn.hand_backs[-1] == "checkout", turn.hand_backs
    assert turn.clip_kinds.count("at_checkout") == 1, turn.clip_kinds
    assert turn.steps == len(turn.results) - 1, (turn.steps, len(turn.results))
    assert DEAD not in turn.spoken, "the drafted 'place the order' step was voiced"
    assert turn.extras.get("tool_loop_end") == "handed_back", turn.extras.get("tool_loop_end")
    v = hjudge("H1", turn.state, turn.voiced)
    assert v.success is True and not v.checkout_pressed, vdict(v)
    assert_one_source(turn)


async def test_completion_check_on_shared_start(tmp_path: Path) -> None:
    """B4 (merge-aware): ``completion_check=True``, H2's path ending in a no-call answer. With ``Merge T-0208-a2`` in
    this checkout's history the check fires once, its body holds no shared byte and its images are browser frames;
    before that merge no check request is sent (H-T207-9). Either way the judge succeeds and one source holds."""
    merged = a2_merged()
    turn = await handoff_turn(tmp_path, "H2", completion_check=True)
    print(
        f"T-0208 B4 completion check: Merge T-0208-a2 in history = {merged}; checks sent = {len(turn.check_bodies)}"
    )
    if merged:
        assert len(turn.check_bodies) == 1, (
            f"T-0208 A2.1/B4: one check; got {len(turn.check_bodies)}"
        )
        body = turn.check_bodies[0]
        assert b64(turn.stored) not in json.dumps(body)
        imgs = decoded_images(body)
        assert imgs and all(i in turn.browser_jpegs for i in imgs), (
            "the check's images are browser frames"
        )
    else:
        assert turn.check_bodies == [], (
            "T-0207 H-T207-9: no check on a shared turn before Merge T-0208-a2"
        )
    v = hjudge("H2", turn.state, turn.voiced)
    assert v.success is True, vdict(v)
    assert_one_source(turn)


# =====================================================================================================
# B5: a click on the picture (adversarial)
# =====================================================================================================


async def test_click_on_the_picture_is_refused_and_reaches_no_page(tmp_path: Path) -> None:
    """B5: in H1 a step-1 ``click_at`` at the drawn steam wand (the red circle's centroid in the shared image) is
    ``refused`` with ``SHARED_FRAME_REASON``; no page event fires (test-installed listeners; positive control: a
    synthetic ``mousedown`` is counted); nothing reaches the executor at step 1; step 2's current image is a fresh
    browser frame labelled ``LABEL_NOW`` with no shared byte; the turn still finishes the golden path and succeeds."""
    x, y = red_centroid(image_bytes("H1"))
    turn = await handoff_turn(tmp_path, "H1", step1=click(x, y), listen=True)
    assert turn.results[0] == ("t1-call", "refused"), turn.results[:1]
    card = turn.cards.get("t1-call")
    assert card is not None and card.reason == SHARED_FRAME_REASON, card
    assert turn.events_after_step1 == [], f"T-0208 B5: page events fired: {turn.events_after_step1}"
    assert turn.control_events == ["mousedown"], (
        f"power: the listener counts ({turn.control_events})"
    )
    assert len(turn.outcomes) == len(turn.results) - 1, (
        "the refused click never reached the executor"
    )
    s2 = turn.step2_body
    assert s2 is not None
    cur = [p for p in last_user(s2)["content"] if p.get("type") == "image_url"]
    assert len(cur) == 1, cur
    img = base64.b64decode(cur[0]["image_url"]["url"].split(",", 1)[1])
    assert img != turn.stored and img in turn.browser_jpegs, (
        "T-0208 B5: step 2 is a fresh browser frame"
    )
    parts = last_user(s2)["content"]
    i = parts.index(cur[0])
    assert parts[i + 1] == {"type": "text", "text": L_NOW}, parts[i + 1 : i + 2]
    assert b64(turn.stored) not in json.dumps(s2)
    v = hjudge("H1", turn.state, turn.voiced)
    assert v.success is True, vdict(v)
    assert_one_source(turn)


# =====================================================================================================
# B6: traps through the executor
# =====================================================================================================


async def _h3_delivery(rig: Any) -> tuple[str, str]:
    await _d6_to_cart(rig, carryout=False)
    return DONE["H3"], ""


TRAPS: list[Any] = [
    pytest.param("H1", _trap_d4_bes878, {"trap": True}, id="H1-bes878-wand"),
    pytest.param("H2", _trap_d5_ridge, {"trap": True}, id="H2-ridge-twin"),
    pytest.param("H3", _h3_delivery, {}, id="H3-delivery-default"),
]


@pytest.mark.parametrize(("hid", "trap", "flags"), TRAPS)
async def test_trap(hid: str, trap: Callable[[Any], Any], flags: dict[str, bool]) -> None:
    """B6: each trap, driven through the web-mode executor on the base scenario's pages from the row's start page,
    fails ``handoff_tasks.judge`` (with ``trap`` where B3 names one)."""
    base = row(hid).base_scenario
    async with opened(base, row(hid).start_page) as rig:
        voiced, shown = await trap(rig)
        rig.executed()
        st = await rig.state()
    v = hjudge(hid, st, voiced, shown)
    assert v.success is False, (hid, st, vdict(v))
    for k, want in flags.items():
        assert getattr(v, k) is want, (k, st, vdict(v))


# =====================================================================================================
# B7: no egress
# =====================================================================================================


async def test_no_egress(tmp_path: Path) -> None:
    """B7 (T-0205 A8): a whole handoff turn (H2) makes zero resolver or connector calls and leaves zero non-loopback
    sockets in this process and its Chromium children, read while the browser runs. Positive controls: a socket bound
    to 0.0.0.0 is reported, and the deny-all seams record a page's off-origin ``https`` image."""
    turn = await handoff_turn(tmp_path, "H2", sockets=True)
    assert turn.net == ([], []), turn.net
    assert turn.sockets == [], f"T-0208 B7: sockets off loopback: {turn.sockets}"
    import os
    import tempfile

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        assert non_loopback_sockets([os.getpid()]), "power: a 0.0.0.0 bind is seen"
    import asyncio

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "leak.html").write_text(
            '<!doctype html><html><body><img src="https://leak.example/x.png" alt=""></body></html>'
        )
        srv = new_server(Path(tmp))
        srv.start()
        net = DenyNet()
        try:
            async with web_executor(srv.origin, net) as ex:
                await run(ex.open(srv.origin + "/leak.html"))
                for _ in range(40):
                    if net.resolves:
                        break
                    await asyncio.sleep(0.1)
                assert net.resolves and set(net.resolves) == {"leak.example"}, net.resolves
        finally:
            srv.stop()
