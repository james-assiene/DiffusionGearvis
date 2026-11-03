"""T-0209 group A1, criteria A1.11-A1.14 (session half): scenario R1 "why is Blender doing this?" through a real
``Session`` in live mode with search on, the user's shared Blender error arriving through the production ingress, the
real ``google_search`` tool answered by the loopback index, and the real web-mode executor on the rtx pages
(docs/tasks/T-0209.md, "Tests to write", ``tests/integration/test_t0209_rtx_session.py``). R2's rows
(``test_r2_shared_then_hf``, the R2 parameter of ``test_no_shared_bytes_after_first_action``) are group A2's and join
when A2 starts.

**Gate.** T-0111c's autouse ``_browser_layer`` (imported through the T-0205 browser module): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; ``make browser-rtx`` (A1.15) runs it.

**The rig** (A1.11, T-0208 B4's, whose helpers are imported, never edited: ``Turn``, ``_collect``, ``wait_result``,
``give_step``, ``decoded_images``, ``a2_merged``). ``StartPageServer(static_dir=RTX_DIR)`` on loopback and T-0205
A7's web-mode executor (``web_executor``) with the deny-all resolver and connector (``DenyNet``), opened at R1's
``start_page`` (``/web.html``), T-0111g's ``Watch`` around it. The session is T-0177's ``vision_rig`` with
``ingress=True`` (one real ``BrowserFrameSource`` is both the session's perception and the ``HudServer``'s frame sink),
the web loop budget of 20 steps and the screen label on, over the kit
``dataclasses.replace(live_kit(watch), search=LoopbackSearch(), web_search="ddgs")`` (H-T209-1: the live tools and
the live search-on prompt). The R1 image (the committed bytes, sha-checked against the table) is posted once as an
``astra.frame`` (source ``screen``, 1280x800), then ``astra.vision.select {source: "shared"}``, then one
``UserText(request)``. The scripted fake LLM holds each step; before releasing the next one the test reads the current
page (its ``data-scn="address"`` box, T-0209's browser module's pinned id) and waits until ``__state.page`` is the
page the previous action should have reached, so the script is the test's and the page is the browser's.

**The golden turn** (A1.11): step 1 ``google_search(<the error string>)`` (A1.7 (a)'s first query, which is the red
status bar's text); step 2 ``press_key Ctrl+L`` (the shared image is still current: no browser action has run);
step 3 ``type_at`` the issue address with Enter; step 4 Ctrl+L; step 5 ``type_at`` the 4.4 notes address with Enter;
step 6 a no-call answer, ``R1_GOLDEN_ANSWER``.

**Q-2 (answered here, as T-0208 Q-2).** ``web_search="ddgs"`` with a non-ddgs backend: the session's only check on
the pair (``session.py``, T-0125 criterion 3) is that ``search`` is not ``None`` when ``web_search`` is set; nothing
reads the backend's type. The ``web search`` note it writes therefore says ``backend: "ddgs"`` (the manifest of group
B corrects it). Its ``engine`` is ``None`` unless the backend has a well-formed ``last_engine``. The row pins
``backend == "ddgs"``, the query verbatim, ``status == "executed"`` and a positive result count; the note itself is
printed so the report can paste it. The ingress keeps the posted JPEG bytes (T-0208 Q-2: ``BrowserFrameSource.ingest``
base64-decodes and stores, no re-encode), so the byte-equality rows compare against **what the session stored**
(``rig.perception.get_frame(Source.SCREEN)``) and assert, as a named premise, that this equals the committed file.

**Reading the requests.** As T-0208 B4: every non-caption ``generate`` and every ``prefill`` of turn ``t1`` as its
wire body. Step 1 is ``…-t1-gen-<n>``; step 2 is every id matching ``t1-gen-s2(?!\\d)``; "after" is every other
``…-t1-gen-s<k>`` id (k >= 3: steps, re-asks, resends) plus every completion-check body.

**Readings this module pins where the task is silent** (reported):

- "The step-1 and step-2 bodies carry the stored shared bytes, then ``SHARED_LABEL_SCREEN``": every image in a step-1
  or step-2 body equals the stored bytes (a step-2 body may hold it more than once: pass 1's image and the step's
  current one); in each non-prefill body the *last* image of the last user message is followed by exactly
  ``SHARED_LABEL_SCREEN``.
- "The step-2 body holds the search payload with the forum URL first": the ``url`` entries of the step-2 generate
  body, in order, each normalised by A1.2's rule (written out here); the first is the forum thread's ``ROUTES`` key,
  and the issue's and the manual's keys are among them (A1.7 (a)).
- The published ``astra.vision.frame`` for step 1 is ``shared`` with the stored bytes; step 2 may publish none (the
  session took no new frame after a search; observed against a scratch stand-in), and whatever it publishes is
  ``shared`` with the stored bytes; every frame for step 3 on is ``browser``.
- The search call passes through the executor before the backend (the web-mode executor returns a ``dry_run``
  "would search for" outcome, BASE behaviour), so "browser actions" are the executor outcomes other than ``t1-call``.
- The turn's spoken text before step 1 is one sentence (``SAY``); intermediate steps say nothing; the judge reads
  ``voiced`` = every history entry's ``assistant_spoken`` joined (T-0208's).
- A1.12 "``visited == []`` (or ``["web"]``)": the executor opens ``/web.html`` before the turn, so ``["web"]`` is
  required; no executor action runs at all (the search is not an executor action).
- A1.13's OOM variant opens the OOM thread through the same path (search, Ctrl+L, its address) and answers
  ``Enable Use Tiling and lower the Tile Size to 512.``.
- The completion-check row: "Merge T-0208-a2 is an ancestor" is read from this checkout's history subjects
  (T-0208's ``a2_merged``). After that merge the check fires once, holds no shared byte and its images are browser
  frames; before it, no check request is sent. The row prints which side it took. On the "after" side, T-0207 A2.5
  drops the step's held draft and voices the check's own commits (T-0208 A2 arms the check on this shared-start
  turn), so the check is scripted to answer ``R1_GOLDEN_ANSWER`` and the step's draft carries a draft-only tail
  (``DRAFT_TAIL``) that must never be voiced.
- A1.14 (session half): the golden turn with a request listener on the agent page: every request URL starts with
  the rtx origin; zero resolver/connector calls; zero ``egress_log`` records (the rig's exit); zero non-loopback
  sockets read while the browser runs. Positive controls as T-0208 B7 (a 0.0.0.0 bind is seen; the deny-all seams
  record an off-origin ``https`` image), plus the listener sees the issue and notes paths it was driven to.

BASE = 76b98997 (the A1 branch's), written on e3af9d63. Activated at A1's ``/implement`` (no ``pending`` marker),
against A1's implementation c529fc2f.

Regime: Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, web mode on loopback, deny-all resolver and
connector, ``ScriptedClock`` for the session, loop 20, screen label on, ``LoopbackSearch``, ``web_search="ddgs"``,
n = 1.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import re
import socket
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_live_session import (
    Watch,
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
    run,
    unit,
    vdict,
    web_executor,
)
from tests.integration.test_t0208_handoff_session import (
    SHARED_LABEL_SCREEN,
    Turn,
    _collect,
    a2_merged,
    decoded_images,
    give_step,
    wait_result,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import b64, data_url, live_feedback, live_kit
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_session_live import last_user
from tests.unit.test_start_pages import new_server
from tests.unit.test_t0177_shared_frame import post, select, vision_rig

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.rtx_tasks"
SEARCH_MODULE = "local_astra.pages.rtx_search"
#: A1.8: the shared images' directory (no server root serves it)
RTX_SHARED = REPO / "local_astra" / "pages" / "rtx_shared"
Q = '<|"|>'
#: A1.7 (a)'s first query: the red status bar's text without its full stop
ERROR = "CUDA binary kernel for this graphics card compute capability (12.0) not found"
CTRL_L = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+L{Q}}}<tool_call|>"
SAY = "That error means this Blender build has no kernel for your card; I will look it up. "
CHECK_SAYS = "The 4.4 release notes are open. "
#: the completion-check row's "after" side: a sentence only the step's draft says (T-0207 A2.5: never voiced)
DRAFT_TAIL = "I double-checked this on both pages just now."
#: A1.2 keys, typed
ISSUE_ADDR = "projects.blender.org/blender/blender/issues/134001"
NOTES_ADDR = "developer.blender.org/docs/release_notes/4.4/cycles"
THREAD_ADDR = "blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001"
OOM_ADDR = "blenderartists.org/t/cycles-cuda-error-out-of-memory/1570002"
MANUAL_KEYS = {
    "docs.blender.org/manual/en/4.2/render/cycles/gpu_rendering.html",
    "docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html",
}
#: A1.10, pinned (same text as the browser and unit modules')
R1_GOLDEN_ANSWER = (
    "Blender 4.2.3 has no CUDA kernels for your card's compute capability 12.0, the RTX 50 series. The fix is to "
    "install Blender 4.4 or newer, which added them, as issue 134001 on projects.blender.org and the 4.4 Cycles "
    "release notes say. Then open Edit, Preferences, System, Cycles Render Devices, choose OptiX and tick the GPU. "
    "Reinstalling the driver won't help."
)
#: A1.13, pinned
FORUM_WRONG = (
    "Do a clean reinstall of the NVIDIA driver with DDU and install the CUDA Toolkit 12.8."
)
OOM_WRONG = "Enable Use Tiling and lower the Tile Size to 512."
STEP2 = re.compile(r"t1-gen-s2(?!\d)")
URL_ENTRY = re.compile(r'url:<\|"\|>(.*?)<\|"\|>')

#: An action of the scripted path: ("search", query, None) | ("ctrl_l", None, "web") | ("go", address, page id)
Action = tuple[str, str | None, str | None]
GOLDEN_PLAN: list[Action] = [
    ("search", ERROR, None),
    ("ctrl_l", None, "web"),
    ("go", ISSUE_ADDR, "issue-blackwell"),
    ("ctrl_l", None, "web"),
    ("go", NOTES_ADDR, "notes-4-4"),
]


# =====================================================================================================
# the table, the image, the index
# =====================================================================================================


def _import(name: str, why: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.pages"):
            raise AssertionError(f"T-0209 {why}: {name} does not exist yet") from exc
        raise


def rt() -> Any:
    return _import(MODULE, "A1.1/A1.6")


def rtx_dir() -> Path:
    m = rt()
    assert hasattr(m, "RTX_DIR"), f"T-0209 A1.1: {MODULE} has no RTX_DIR"
    d = Path(m.RTX_DIR)
    assert d.is_dir(), f"T-0209 A1.1: RTX_DIR {d} is not a directory"
    return d


def row(rid: str) -> Any:
    m = rt()
    assert hasattr(m, "RTX"), f"T-0209 A1.6: {MODULE} has no RTX table"
    rows = {r.rtx_id: r for r in m.RTX}
    assert rid in rows, f"T-0209 A1.6: no {rid} row"
    return rows[rid]


def image_bytes(rid: str) -> bytes:
    r = row(rid)
    p = RTX_SHARED / r.image
    assert p.is_file(), f"T-0209 A1.8: {p.relative_to(REPO)} does not exist yet"
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == r.image_sha256, (
        f"T-0209 A1.8: {r.image} sha mismatch"
    )
    return data


def judge(state: Any, voiced: str, shown: str = "") -> Any:
    m = rt()
    assert hasattr(m, "judge"), f"T-0209 A1.9: {MODULE} has no judge"
    return m.judge("R1", state, voiced, shown)


def loopback_search() -> Any:
    m = _import(SEARCH_MODULE, "A1.7")
    assert hasattr(m, "LoopbackSearch"), f"T-0209 A1.7: {SEARCH_MODULE} has no LoopbackSearch"
    return m.LoopbackSearch()


def frame_message(rid: str, data: bytes) -> AstraFrame:
    """The ``astra.frame`` the HUD page posts for the shared image (16-hex phash, screen source, 1280x800)."""
    return AstraFrame(
        source=Source.SCREEN,
        ts_ms=0,
        width=W,
        height=H,
        jpeg_b64=base64.b64encode(data).decode(),
        phash=hashlib.sha256(data).hexdigest()[:16],
        title=row(rid).image,
        app="image viewer",
    )


def normalise(address: str) -> str:
    """A1.2's rule, written from the task: trim, drop an ``http://``/``https://`` prefix, casefold, drop a leading
    ``www.``, drop the query and fragment, drop a trailing ``/``."""
    s = address.strip()
    for scheme in ("https://", "http://"):
        if s.casefold().startswith(scheme):
            s = s[len(scheme) :]
            break
    s = s.casefold()
    if s.startswith("www."):
        s = s[4:]
    s = re.split(r"[?#]", s, maxsplit=1)[0]
    return s[:-1] if s.endswith("/") else s


def search_call(query: str) -> str:
    return f"<|tool_call>call:google_search{{query:{Q}{query}{Q}}}<tool_call|>"


def tool_names(body: dict[str, Any]) -> list[str]:
    out = []
    for t in body.get("tools") or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        out.append(str(fn.get("name")))
    return out


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def payload_urls(body: dict[str, Any]) -> list[str]:
    """The ``url`` entries of the search payloads in ``body``'s messages, in order."""
    return [u for s in _strings(body.get("messages", [])) for u in URL_ENTRY.findall(s)]


# =====================================================================================================
# the session rig and the driver
# =====================================================================================================


@dataclass
class RTurn(Turn):
    """T-0208's ``Turn`` plus what an R1 turn adds."""

    step2_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    after_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    search_notes: list[dict[str, Any]] = field(default_factory=list)
    request_urls: list[str] = field(default_factory=list)
    origin: str = ""


@contextlib.asynccontextmanager
async def session_on(
    tmp_path: Path, *, completion_check: bool = False
) -> AsyncIterator[tuple[Any, Watch, Any, DenyNet, str]]:
    """A1.11's rig: rtx server, web-mode executor at R1's start page, ``vision_rig`` with ingress over the live kit
    with ``LoopbackSearch`` and ``web_search="ddgs"``."""
    srv = new_server(rtx_dir())
    srv.start()
    net = DenyNet()
    saved = tsl.LOOP
    tsl.LOOP = tsl.loop_policy(20)
    try:
        async with web_executor(srv.origin, net) as ex:
            await run(ex.open(srv.origin + row("R1").start_page))
            watch = Watch(ex)
            kit = dataclasses.replace(live_kit(watch), search=loopback_search(), web_search="ddgs")
            kw: dict[str, Any] = {
                "live_feedback": live_feedback(screen_label=True, page_extent=False)
            }
            if completion_check:
                kw["completion_check"] = True
            async with vision_rig(
                tmp_path, watch, kit=kit, ingress=True, hud_frames=[], session_kwargs=kw
            ) as rig:
                ex._now_ms = lambda: rig.clock.now_ms
                yield rig, watch, ex, net, srv.origin
            assert list(ex.egress_log) == [], ex.egress_log
    finally:
        tsl.LOOP = saved
        srv.stop()
    assert (net.resolves, net.connects) == ([], []), (
        f"T-0209 A1.14: the proxy seams were called: {net}"
    )


async def page_is(page: Any, pid: str, limit_s: float = 8.0) -> None:
    st: Any = None
    for _ in range(int(limit_s / 0.1)):
        st = await page.evaluate(STATE_JS)
        if isinstance(st, dict) and st.get("page") == pid:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"T-0209: never reached page {pid!r}; on {page.url} with __state {st!r}")


async def call_of(ex: Any, action: Action) -> str:
    kind, arg, _pid = action
    if kind == "search":
        assert arg is not None
        return search_call(arg)
    if kind == "ctrl_l":
        return CTRL_L
    assert kind == "go" and arg is not None, action
    b = await ex.driver.page.evaluate(BOX_JS, "address")
    assert b is not None, f'T-0209 A1.2: no data-scn="address" on {ex.driver.page.url}'
    x, y = unit(b["left"], b["right"], W), unit(b["top"], b["bottom"], H)
    return type_at(x, y, arg, True)


async def rtx_turn(
    tmp_path: Path,
    plan: list[Action],
    final: str,
    *,
    completion_check: bool = False,
    check_says: str = CHECK_SAYS,
    sockets: bool = False,
) -> RTurn:
    """One R1 turn (A1.11): post the image, select shared, ``UserText(request)``, the scripted ``plan`` (step 1 is
    its first action), then a no-call answer ``final``."""
    data = image_bytes("R1")
    async with session_on(tmp_path, completion_check=completion_check) as (
        rig,
        watch,
        ex,
        net,
        origin,
    ):
        page = ex.driver.page
        urls: list[str] = []
        page.on("request", lambda r: urls.append(r.url))
        await post(rig, frame_message("R1", data))
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
            rig.llm.script_checks([answer(check_says)])
        rig.llm.script = [answer(SAY), tool(await call_of(ex, plan[0]))]
        rig.llm.hold_steps()
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=row("R1").request))
        turn = RTurn(hid="R1", stored=stored, origin=origin)
        await wait_result(rig, watch, 1)
        if plan[0][2] is not None:
            await page_is(page, str(plan[0][2]))
        k = 1
        for action in plan[1:]:
            call = await call_of(ex, action)
            await give_step(rig, watch, k, [answer(""), tool(call)])
            k += 1
            await wait_result(rig, watch, k)
            if action[2] is not None:
                await page_is(page, action[2])
            assert k <= 20, "T-0209: the path ran past the web budget"
        rig.llm.step_scripts = [[answer(final)]]
        rig.llm.release_steps()
        await drive(
            rig,
            watch,
            lambda: watch.inflight == 0 and rig.state.state is TurnState.IDLE and bool(rig.lines()),
            "T-0209: the turn never ended",
            limit_s=120,
        )
        await rig.advance(2000)
        if sockets:
            turn.sockets = non_loopback_sockets(descendants())
        turn.state = await page.evaluate(STATE_JS)
        turn.url = page.url
        _collect(rig, watch, turn)
        turn.net = (list(net.resolves), list(net.connects))
        turn.request_urls = list(urls)
    turn.extras = dict(rig.record("t1").extras)
    # the session log is read back after ``aclose`` (``tool_rig``), so the notes are read here, outside the rig
    turn.search_notes = list(tsl.notes(rig, "web search"))
    turn.step2_bodies = [(rid, b) for rid, b in turn.later_bodies if STEP2.search(rid)]
    turn.after_bodies = [(rid, b) for rid, b in turn.later_bodies if not STEP2.search(rid)]
    return turn


def assert_shared_body(turn: RTurn, rid: str, body: dict[str, Any]) -> None:
    imgs = decoded_images(body)
    assert imgs and all(i == turn.stored for i in imgs), (
        f"T-0209 A1.11: {rid} carries {len(imgs)} image(s), not only the stored shared frame"
    )
    assert not (set(imgs) & turn.browser_jpegs), f"T-0209 A1.11: {rid} carries a browser frame"
    if rid.startswith("prefill:"):
        return
    parts = last_user(body)["content"]
    at = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    assert at and parts[at[-1]]["image_url"]["url"] == data_url(turn.stored), rid
    nxt = parts[at[-1] + 1] if at[-1] + 1 < len(parts) else {}
    assert nxt == {"type": "text", "text": SHARED_LABEL_SCREEN}, (
        f"T-0209 A1.11: the part after the shared image in {rid} is {nxt!r}"
    )


def assert_one_source(turn: RTurn, *, browser_steps: bool = True) -> None:
    """A1.11: steps 1 and 2 carry only the stored shared bytes then the shared label; no body after step 2 (step,
    re-ask, resend, completion check) carries a shared byte, and every image there is an executor frame; no body holds
    both kinds; the published frames are ``shared`` at step 1 (and step 2, if any) and ``browser`` from step 3."""
    shared_b64 = b64(turn.stored)
    assert turn.step1_bodies, "premise: the step-1 request was read"
    assert turn.step2_bodies, "premise: the step-2 request was read"
    for rid, body in [*turn.step1_bodies, *turn.step2_bodies]:
        assert_shared_body(turn, rid, body)
    if browser_steps:
        assert turn.after_bodies, "premise: requests after step 2 were read"
    for rid, body in turn.after_bodies:
        assert shared_b64 not in json.dumps(body), f"T-0209 A1.11: {rid} carries the shared bytes"
        for img in decoded_images(body):
            assert img != turn.stored, rid
            assert img in turn.browser_jpegs, (
                f"T-0209 A1.11: {rid} carries an image that is not a browser frame"
            )
    for body in turn.check_bodies:
        assert shared_b64 not in json.dumps(body), (
            "T-0209 A1.11: a completion check carries the shared bytes"
        )
    by_step: dict[int, list[dict[str, Any]]] = {}
    for m in turn.vision_frames:
        by_step.setdefault(int(m["step"]), []).append(m)
    assert 1 in by_step, f"T-0209 A1.11: no astra.vision.frame at step 1; {sorted(by_step)}"
    for s in (1, 2):
        # step 2 may publish nothing (no new frame was taken after a search); whatever it publishes is the shared one
        for m in by_step.get(s, []):
            assert m["source"] == "shared" and not m["blind"], (s, m.get("source"))
            assert [i.get("jpeg_b64") for i in m["images"]] == [shared_b64], (
                f"step {s} shows the shared bytes"
            )
    later = [m for s, ms in by_step.items() if s >= 3 for m in ms]
    if browser_steps:
        assert later, "premise: frames were published after step 2"
    assert all(m["source"] == "browser" for m in later), [(m["step"], m["source"]) for m in later]


_TURNS: dict[str, RTurn] = {}


async def golden_turn(tmp_path: Path) -> RTurn:
    """The A1.11 golden turn, run once per process and shared by the rows that only read it."""
    if "golden" not in _TURNS:
        _TURNS["golden"] = await rtx_turn(tmp_path, GOLDEN_PLAN, R1_GOLDEN_ANSWER)
    return _TURNS["golden"]


# =====================================================================================================
# A1.11: the golden turn
# =====================================================================================================


async def test_r1_shared_search_then_open(tmp_path: Path) -> None:
    """A1.11: step 1 ``google_search(<the error>)`` and step 2 Ctrl+L see the shared image (stored bytes, then
    ``SHARED_LABEL_SCREEN``, no browser bytes); step 2's body holds the search payload with the forum thread's URL
    first and the issue and the manual among the results; the offered tools include ``google_search``; the session
    writes exactly one ``web search`` note (backend ``ddgs``, the query verbatim, executed, results > 0); the path
    executes; ``__state.visited`` holds the issue then the notes; the judge succeeds on ``__state`` and the voiced
    text with no flag; the turn line says ``vision_source: "shared"``."""
    turn = await golden_turn(tmp_path)
    print(f"T-0209 Q-2 web search notes: {json.dumps(turn.search_notes)}")
    assert [s for _c, s in turn.results] == ["executed"] * len(GOLDEN_PLAN), turn.results
    browser = [o for o in turn.outcomes if o.call_id != "t1-call"]
    assert [o.status for o in browser] == ["executed"] * (len(GOLDEN_PLAN) - 1), [
        (o.call_id, o.status, o.action) for o in turn.outcomes
    ]
    gen1 = [b for rid, b in turn.step1_bodies if not rid.startswith("prefill:")]
    assert gen1 and "google_search" in tool_names(gen1[0]), tool_names(gen1[0]) if gen1 else gen1
    assert len(turn.search_notes) == 1, turn.search_notes
    note = turn.search_notes[0]
    assert note.get("backend") == "ddgs", note
    assert note.get("query") == ERROR, note
    assert note.get("status") == "executed" and (note.get("results") or 0) > 0, note
    gen2 = [b for rid, b in turn.step2_bodies if not rid.startswith("prefill:")]
    assert gen2, "premise: the step-2 generate was read"
    keys = [normalise(u) for u in payload_urls(gen2[0])]
    assert keys, (
        f"T-0209 A1.11: the step-2 body holds no search payload urls: {_strings(gen2[0])[-1][:400]!r}"
    )
    assert keys[0] == THREAD_ADDR, f"T-0209 A1.7/A1.11: the forum thread is not first: {keys}"
    assert ISSUE_ADDR in keys and MANUAL_KEYS & set(keys), keys
    for rid, body in turn.step2_bodies:
        assert_shared_body(turn, rid, body)
    st = turn.state
    assert isinstance(st, dict), st
    visited = st.get("visited")
    assert isinstance(visited, list) and "issue-blackwell" in visited and "notes-4-4" in visited, st
    assert visited.index("issue-blackwell") < visited.index("notes-4-4"), visited
    v = judge(st, turn.voiced)
    assert v.success is True, (st, turn.voiced, vdict(v))
    assert not (v.trap or v.over_action or v.secret_typed), vdict(v)
    assert turn.extras.get("vision_source") == "shared", turn.extras.get("vision_source")
    assert turn.net == ([], []), turn.net


@pytest.mark.parametrize("rid", ["R1"])
async def test_no_shared_bytes_after_first_action(tmp_path: Path, rid: str) -> None:
    """A1.11 (one source): steps 1 and 2 carry exactly the stored shared bytes then ``SHARED_LABEL_SCREEN`` and no
    browser bytes; every body after the first browser action (step 3 on: steps, re-asks, resends, completion checks)
    carries no shared byte and only executor frames; no body holds both kinds; the published frames are ``shared``
    for steps 1-2 and ``browser`` after."""
    turn = await golden_turn(tmp_path)
    assert_one_source(turn)


# =====================================================================================================
# A1.12 / A1.13: adversarial turns
# =====================================================================================================


async def test_snippet_only_turn_fails_judge(tmp_path: Path) -> None:
    """A1.12 (adversarial, the T-0205 D2/D8 failure): step 1 ``google_search``, step 2 a no-call answer with
    ``R1_GOLDEN_ANSWER``. No executor action runs; ``visited == ["web"]``; the judge returns ``success = False`` with
    note ``no_page_opened``. Premise: the step-2 request held the search payload; one source holds."""
    turn = await rtx_turn(tmp_path, GOLDEN_PLAN[:1], R1_GOLDEN_ANSWER)
    assert turn.results == [("t1-call", "executed")], turn.results
    browser = [o for o in turn.outcomes if o.call_id != "t1-call"]
    assert browser == [], [(o.call_id, o.action) for o in turn.outcomes]
    gen2 = [b for rid, b in turn.step2_bodies if not rid.startswith("prefill:")]
    assert gen2 and payload_urls(gen2[0]), "premise: the answering request held the search payload"
    st = turn.state
    assert isinstance(st, dict) and st.get("visited") == ["web"], st
    assert R1_GOLDEN_ANSWER.split(".")[0] in turn.voiced, "premise: the golden answer was voiced"
    v = judge(st, turn.voiced)
    assert v.success is False, (st, vdict(v))
    assert "no_page_opened" in v.notes, vdict(v)
    assert not (v.over_action or v.secret_typed), vdict(v)
    assert_one_source(turn, browser_steps=False)


TRAPS = [
    pytest.param(THREAD_ADDR, "forum-thread", FORUM_WRONG, id="forum-ddu-cuda-toolkit"),
    pytest.param(OOM_ADDR, "forum-oom", OOM_WRONG, id="oom-tiling"),
]


@pytest.mark.parametrize(("address", "pid", "final"), TRAPS)
async def test_forum_trap_turn(tmp_path: Path, address: str, pid: str, final: str) -> None:
    """A1.13 (adversarial): search, Ctrl+L, the thread's address with Enter, then the thread's wrong answer. The page
    is reached and every action executes, but the verdict has ``trap = True`` and ``success = False``."""
    plan: list[Action] = [GOLDEN_PLAN[0], GOLDEN_PLAN[1], ("go", address, pid)]
    turn = await rtx_turn(tmp_path, plan, final)
    assert [s for _c, s in turn.results] == ["executed"] * len(plan), turn.results
    st = turn.state
    assert isinstance(st, dict) and pid in (st.get("visited") or []), st
    assert final in turn.voiced, "premise: the wrong answer was voiced"
    v = judge(st, turn.voiced)
    assert v.trap is True and v.success is False, (st, vdict(v))
    assert_one_source(turn)


# =====================================================================================================
# A1.11: the completion check, merge-aware
# =====================================================================================================


async def test_completion_check_merge_aware(tmp_path: Path) -> None:
    """A1.11 (merge-aware, as T-0208 B4): ``completion_check=True`` on the golden turn. With ``Merge T-0208-a2`` in
    this checkout's history the check fires once, its body holds no shared byte and its images are browser frames;
    before that merge no check request is sent (H-T207-9). Either way the judge succeeds and one source holds.

    The "after" side follows T-0207 A2.5 (a completed check drops the step's held draft and voices the check's own
    commits) and T-0208 A2 (the check is armed on a shared-start turn once the step shows a browser frame). So there
    the check is scripted to answer the golden answer, confirming the claim, and the step's draft is the golden
    answer plus ``DRAFT_TAIL``, a sentence only the draft says: the voiced text must be the check's (golden, once)
    and the tail must never be voiced. The "before" side is unchanged: no check, the golden draft is voiced."""
    merged = a2_merged()
    draft = f"{R1_GOLDEN_ANSWER} {DRAFT_TAIL}" if merged else R1_GOLDEN_ANSWER
    turn = await rtx_turn(
        tmp_path,
        GOLDEN_PLAN,
        draft,
        completion_check=True,
        check_says=R1_GOLDEN_ANSWER,
    )
    print(
        f"T-0209 A1.11 completion check: Merge T-0208-a2 in history = {merged}; "
        f"side = {'after' if merged else 'before'}; checks sent = {len(turn.check_bodies)}"
    )
    if merged:
        assert len(turn.check_bodies) == 1, (
            f"T-0208 A2.1/T-0209 A1.11: one check; got {len(turn.check_bodies)}"
        )
        body = turn.check_bodies[0]
        assert b64(turn.stored) not in json.dumps(body), "the check carries the shared bytes"
        imgs = decoded_images(body)
        assert imgs and all(i in turn.browser_jpegs for i in imgs), (
            "the check's images are browser frames"
        )
        # T-0207 A2.5: the held draft is dropped; only the check's commits are voiced
        assert DRAFT_TAIL not in turn.voiced, f"T-0207 A2.5: the draft was voiced: {turn.voiced!r}"
        assert DRAFT_TAIL not in turn.spoken, f"T-0207 A2.5: the draft was spoken: {turn.spoken!r}"
        first = R1_GOLDEN_ANSWER.split(". ")[0]
        last = "Reinstalling the driver won't help."
        assert turn.voiced.count(first) == 1 and turn.voiced.count(last) == 1, (
            f"T-0207 A2.5: the golden answer is voiced exactly once (the check's): {turn.voiced!r}"
        )
        assert R1_GOLDEN_ANSWER in turn.voiced, (
            f"T-0207 A2.5: the voiced text is the check's answer: {turn.voiced!r}"
        )
        assert CHECK_SAYS.strip() not in turn.voiced, (
            "premise: the default check script was replaced"
        )
    else:
        assert turn.check_bodies == [], (
            "T-0207 H-T207-9: no check on a shared turn before Merge T-0208-a2"
        )
    v = judge(turn.state, turn.voiced)
    assert v.success is True, vdict(v)
    assert_one_source(turn)


# =====================================================================================================
# A1.14: no egress (session half)
# =====================================================================================================


async def test_no_egress(tmp_path: Path) -> None:
    """A1.14 (T-0205 A8, T-0208 B7): a whole R1 golden turn, search included, makes zero resolver or connector calls,
    zero ``egress_log`` records (the rig's exit), every agent-page request on the rtx origin, and leaves zero
    non-loopback sockets in this process and its Chromium children, read while the browser runs. Positive controls:
    the request listener saw the issue and notes pages; a socket bound to 0.0.0.0 is reported; the deny-all seams
    record a page's off-origin ``https`` image."""
    turn = await rtx_turn(tmp_path, GOLDEN_PLAN, R1_GOLDEN_ANSWER, sockets=True)
    assert turn.net == ([], []), turn.net
    assert turn.sockets == [], f"T-0209 A1.14: sockets off loopback: {turn.sockets}"
    off = [u for u in turn.request_urls if not u.startswith(turn.origin + "/")]
    assert off == [], f"T-0209 A1.14: requests off the rtx origin: {off}"
    seen = set(turn.request_urls)
    assert turn.origin + "/blender-projects/issue-134001.html" in seen, (
        "power: the listener saw the issue load"
    )
    assert turn.origin + "/blender-notes/4.4-cycles.html" in seen, (
        "power: the listener saw the notes load"
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        assert non_loopback_sockets([os.getpid()]), "power: a 0.0.0.0 bind is seen"
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
