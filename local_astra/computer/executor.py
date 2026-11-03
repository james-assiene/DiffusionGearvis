"""``BrowserToolExecutor``: a ``ToolExecutor`` in ``live`` mode that performs the v1 calls on a sandboxed browser
page (T-0111c; the session's ``tools = live`` wiring is T-0111g).

**Statuses** (criterion 1): ``executed`` when the action was dispatched and settled without an exception;
``failed`` when an exception happened after dispatch began or the browser died (the reason names the exception
type); ``refused`` when nothing was dispatched; ``dry_run`` only for ``google_search``, described exactly as
``DryRunToolExecutor`` does. Nothing is ever retried: after a dead browser the next call relaunches once and is
``failed: browser restarted`` without performing its action (criterion 10).

**Frames** (criteria 2-4, 9): ``frame()`` screenshots the active page's viewport and returns a ``Frame(source=
WINDOW)`` after the ``screen_max`` policy. Its ``phash`` is the 64-bit DCT perceptual hash XOR a random 64-bit nonce
drawn once per browser generation (ruling 683cc1e): equal for identical pixels within a generation, different across
a restart, so a page re-opened after a restart is clickable from a fresh frame. A coordinate
call must carry the geometry of an own frame of the current browser generation; a phash seen in an earlier
generation is refused for good, even when it is captured again. Its 0-999 coordinates
are mapped against the *viewport's* CSS size with ``tool_executor.to_pixels``, never against the (possibly
downscaled) image. After every executed action the frame is retaken after the settle, and its phash is
``resolved["frame_after"]``. T-0200: a capture whose pixels are near-uniform (``blank_frame``: any one colour) is
taken again every ``BLANK_RECAPTURE_INTERVAL_S`` until one is not, or until the budget is spent
(``BLANK_RECAPTURE_ACTION_S`` on an action's after-frame, the guarded-type stop and the overview-frame path, never
beyond ``ACTION_TIMEOUT_S`` less ``CAPTURE_RESERVE_S``; ``BLANK_RECAPTURE_FRAME_S`` in ``frame()``); the first
non-blank capture, or else the last successful one, is the frame and the only one registered. A painted first
capture costs one screenshot and the pixel check, nothing else. A re-capture that raises counts as one blank attempt;
the first screenshot raising propagates. Each capture that took more than one screenshot calls ``on_note`` once
(no URL); ``page_overview``'s whole-page picture is never re-taken. T-0203 B2: an action's late re-capture after a
followed navigation is a second ``_capture``-style capture (T-0200's re-capture applies); only the kept one (the
second, or the first when the second's first screenshot raises or no time is left) is registered, once.

**Active page** (criterion 15, ruling 683cc1e): ``execute()``'s ``ensure()`` may adopt a popup that opened after the
last settle. When the driver's ``page_serial`` changes across it within one generation, every non-search action is
``refused`` (``the active page changed; take a new frame``) and nothing is dispatched; a relaunch is checked first
(``failed: browser restarted``). A coordinate call whose frame was taken on another page than the active one is
refused the same way. ``ensure()`` runs under the same ``ACTION_TIMEOUT_S`` as the dispatch, so one ``execute()``
takes at most ``ACTION_TIMEOUT_S``; a timeout in either has the same outcome. A cold launch (the first call, or
the relaunch after a dead browser) happens inside that ``ensure()``, so it counts against the 8 s
``ACTION_TIMEOUT_S`` too.

**Cancellation** (criterion 11): a ``CancelledError`` propagates; any button or key this executor pressed and has not
released is released first.

**Options** (T-0111g A1-A3): ``headless`` is passed to ``PlaywrightDriver`` (default headless). ``allowed_origins``
narrows the fence from any loopback URL to exactly those ``scheme://host:port`` origins (``runtime.url_allowed``);
``open()`` refuses anything else with ``ValueError``. With ``end_when_closed`` the browser *ends for good* when, once
launched, the driver reports it disconnected or with no page open: the check runs before every ``ensure()`` and
again after it returns or raises (whatever it raised; nothing is then dispatched or captured), after
which ``closed`` is ``True``, nothing relaunches or opens a page, every non-search call is ``refused`` (``the browser
window was closed``), ``frame()`` and ``open()`` raise ``BrowserClosedError``, and ``google_search`` is still the dry
run. The executor also builds its ``PlaywrightDriver`` in end mode, so a last page that closes *during* an
``ensure()`` (while a popup is adopted or its load awaited) is not reopened either: the driver raises
``runtime.BrowserEndedError`` and the executor ends the same way. Without it the executor never reads the two facts
(``connected``, ``page_count``) and T-0111c's relaunch-once rule holds.

**Echo** (T-0134 A2): with ``echo="model"`` an executed ``click_at``/``type_at`` reports the 0-999 arguments it
executed (``clicked at (X, Y)``), the units of the call; ``resolved`` keeps ``x_px``/``y_px``. The default
``"pixels"`` reports the viewport pixels, as before.

**Page extent** (T-0134 A4): when the driver's *class* has ``layout_metrics`` (``runtime.PlaywrightDriver`` reads
CDP ``Page.getLayoutMetrics``, which page script cannot change), every own capture reads the metrics before and after
its screenshot; ``extent_of(phash)`` gives ``{"above_px", "below_px", "viewport_px"}`` (CSS px) for that frame, or
``None`` when the two reads differ, a read failed or timed out, or the driver has no such method. The capture itself
never fails because of the extent.

**Overview frame** (T-0151 A1): ``overview_frame_of(key)`` takes a ``page_overview`` picture from ``overview_of``'s
once-only store and registers it, apart from the screen frames, when it is unclipped, as wide as the viewport, of the
current generation and page, and ``latest_frame`` is still the own frame that was latest at the capture. A
``click_at``/``type_at`` on it maps against the captured page size, is refused when ``latest_frame`` moved on
(``OVERVIEW_STALE``) or the page size changed (``OVERVIEW_PAGE_CHANGED``), scrolls at most once to bring the point
into view, checks the page arrived (else ``OVERVIEW_NOT_REACHED``, nothing clicked) and acts at the viewport point;
its echo is BASE's with a ``scrolled down|up by N px and `` prefix after a scroll. CDP metrics and pixels only.

**Web mode** (T-0160 B-2..B-6; ``web=True`` with ``start_origin``): the ``PlaywrightDriver`` runs behind its egress
proxy (``runtime`` docstring) in a fresh profile; a ``user_data_dir`` is a ``ValueError``. ``open()`` also takes public
``https`` URLs (``fence.web_url_allowed``). ``Ctrl+T``/``Ctrl+L`` open ``<start origin>/web.html`` (``opened the
start page``) and ``Ctrl+V`` is refused (``PASTE_REASON``). Each executed action's ``resolved["hosts"]`` lists the
hosts the proxy tunnelled since the previous action (at most ``MAX_HOSTS``, the rest counted in ``hosts_more``).

**Password and card guard** (T-0160 B-3, in every mode): a ``type_at`` whose text is card-like
(``fence.card_like``) is refused before anything is dispatched (``CARD_TEXT_REASON``). Otherwise, when the driver's
class has ``focused_field`` (the real driver always does) or in web mode, the executor clicks, reads the focused
field within ``FOCUS_TIMEOUT_S`` and, before any key, stops with ``failed``: ``SECRET_FIELD_REASON`` when
``fence.secret_field`` names it, or in web mode ``FOCUS_UNKNOWN_REASON`` when it could not be read (no method, a
timeout, an exception). The frame is then re-captured, and ``resolved`` carries ``typed: 0`` and never the text. The
overview-frame path applies the same guard. Outside web mode a driver class without ``focused_field`` runs BASE's
path, and an unreadable focus there is typed into as BASE. In web mode a focus left on a ``body`` that is not
contenteditable (a click on a canvas or a bare background) is refused the same way, with ``focus_check: "body"``
(T-0210 A1); ``press_key`` never reads the focus, so a page's own shortcut for one of its keys acts as if the user
pressed it (T-0210 H-T210-2).

**Replace** (T-0189): when that same read names a single-line text ``input`` (``_single_line``) that is not a secret
field, ``COMBOS["Ctrl+A"]`` is pressed before the text, so ``type_at`` replaces the field's text, and ``resolved``
gains ``replaced: True``. A ``textarea``, contenteditable (a designMode ``body`` included), opaque or unreadable
focus is typed at the caret; outside web mode so is a plain ``body`` (T-0210 A1, A3).

**Navigated** (T-0174 B1): when the driver's class has ``navigation_mark`` and ``navigated_since``, an executed
``click_at``, ``type_at`` or ``press_key`` (the overview-frame click and web mode's start-page ``goto`` included)
carries ``resolved["navigated"]``: the mark is taken after ``ensure()`` and its checks, the answer read after the
action's last settle and before the post-action capture, and the key is set only from a ``bool`` (``None``, anything
else or a raise leave it absent). No other tool, outcome or mode carries it; ``hosts`` is not read to decide it.

**Following a navigation** (T-0203 B1-B6; supersedes T-0174 B1-1b (c)): when the driver's *class* also has
``follow_navigation`` and a mark was taken, a first read that is not ``False`` (``None``, ``True``, anything else or a
raise) is followed once (``follow_navigation(mark, min(NAV_FOLLOW_S, deadline - now))``, ``deadline`` T-0200's) and
read again; that second answer is ``navigated``. Order: settle, front, read, follow, re-read, checkout read, capture.
An action that sent Enter (``type_at`` with ``press_enter``, a ``press_key`` chord holding ``Enter``) and whose answer
is ``False`` reads once more after its capture; when that is not ``False`` it follows with what is left of the
call's ``NAV_FOLLOW_S`` (shared by both follows, never past ``deadline``), reads again, reads the checkout again
(``checkout`` is the OR of the two reads) and captures again. A follow that raises ``Exception`` is ignored; a
``CancelledError`` propagates. Each follow that answered other than ``"none"`` calls ``on_note`` with ``{"what":
"live frame: followed navigation", "path": "action", "result", "late", "waited_ms"}`` only. Every other action, and
every action on a driver class without ``follow_navigation``, makes exactly BASE's driver calls; nothing sleeps.

**Restart fallback** (T-0188 A1-A4): when the driver's class has ``replace_page``, a start-page navigation that
raises (``reset_page``, and web mode's ``Ctrl+T``/``Ctrl+L``) is followed by a replacement of the page at the same URL.
``reset_page`` returns ``ResetOutcome(replaced, reason)``; the reason class (``RESET_REASONS``) is decided from a
recorded ``beforeunload`` dialog event and the exception's type, never its message. When the replacement fails too,
``ResetPageError`` carries the classes only. A successful navigation calls nothing more than before.

**Checkout stop** (T-0192 A2-A8, web mode only): when the driver's class has ``main_frame_urls`` (the real driver
does), every ``execute()`` reads it after ``ensure()`` and its restart and page-serial checks, before any dispatch;
when a returned URL is a checkout (``fence.checkout_url``) or the read fails, ``click_at``, ``type_at``, ``scroll``,
``press_key`` (but ``Ctrl+T``/``Ctrl+L``), ``page_overview`` and the overview-frame path are ``refused``
(``CHECKOUT_REASON``) and nothing is dispatched or captured. An executed action reads it again after its last settle,
before the post-action capture, and carries ``resolved["checkout"] = True`` when a URL is a checkout. ``reset_page``
drains it. Only that boolean leaves the read: no URL, path or host is ever in an outcome, an event or a log.

**Submit stop** (T-0206 A3-A8, web mode only): when the driver's *class* has ``submit_facts``, a ``click_at``, the
focus click of a ``type_at`` and the overview-frame click read, after the page-serial check, the checkout check and
T-0204's moved guard (so what is read is what is pressed) and before anything is dispatched, what the click would
press (``submit_facts(x, y)`` at the CSS point, bounded by ``SUBMIT_TIMEOUT_S``). When ``fence.irreversible_submit``
says it sends a form holding filled personal data, the call is ``refused`` with ``SUBMIT_REASON`` and nothing is
dispatched or captured. When the driver's class has ``submit_facts_focused``, a ``press_key`` chord holding
Enter/NumpadEnter (the focused element's form, its default button as the control) or Space (only when the focused
element is itself the control) is refused the same way; other keys are never read. A ``type_at`` with
``press_enter`` reads the focused element's facts after the focus read and before any key, counting the focused field
as filled: when they send, the text is typed but Enter is not pressed, the outcome is ``executed`` with BASE's action
for ``press_enter=False`` and ``resolved["submit_withheld"] = True`` (a textarea target never withholds). A read that
answers ``None``, raises or times out lets the action proceed as BASE (fail open, H-T206-5) and calls ``on_note`` with
``{"what": "live click: submit check", "result": "unknown"}`` (``"live key: submit check"`` for a focused read) only.
Labels and the rest of the facts stay in the executor: only the refusal reason or the ``submit_withheld`` flag leaves
it. Outside web mode, or on a driver class without the reads, nothing is read and the calls are BASE's.

**Wait** (T-0202 B1-B6): ``wait`` (no arguments; any argument is ``bad arguments`` before anything is called)
takes ``scroll``'s path: the guard, the end-when-closed check, ``ensure()``, the restart, page-serial and checkout
checks under ``ACTION_TIMEOUT_S``, fronting, then ``asyncio.sleep(WAIT_S)`` where ``scroll`` dispatches, then the
settle, fronting, the after-settle checkout read and ``_capture`` (T-0200's action budget). No input is sent, no
navigation fact is read, and the outcome is ``executed`` with ``WAIT_ACTION``. Waits are counted per ``turn_id`` once
their sleep has begun: past ``MAX_CONSECUTIVE_WAITS`` in a row (only an executed ``click_at``, ``type_at``,
``scroll``, ``press_key``, ``page_overview`` or overview-frame action ends a run) a wait is refused
``WAIT_RUN_REASON``, past ``MAX_WAITS_PER_TURN`` in the turn ``WAIT_TURN_REASON``, before ``ensure()``. No other
action sleeps.

**Pre-click guard** (T-0204 B1-B7; supersedes T-0203 B6/B8.6, T-0200 C10 and T-0202 B10's "BASE calls" for the real
driver's ``click_at``/``type_at``, T-0204 B10): when ``PRECLICK_GUARD`` (read at call time) and the driver's *class*
has ``screenshot_region``, ``arm_shift_recorder``, ``shift_mark`` and ``shifts_since``, every own capture arms the
shift recorder before its screenshot and reads ``shift_mark()`` right after it, and the kept capture's screenshot (CSS
px, before the ``policy`` downscale) and mark are its frame's reference, keyed by phash (a phash captured again on the
same page replaces its reference with the newest capture and mark, B-R4), at most ``PRECLICK_REF_FRAMES``. An executed ``click_at``/``type_at`` on a screen
frame (not the overview-frame path) is checked after ``ensure()``, every refusal check and fronting, before anything is
dispatched: a ``screenshot_region`` of the box of half-size ``PRECLICK_HALF_PX`` around the point (clamped to the
viewport), bounded by ``PRECLICK_TIMEOUT_S``, then ``shifts_since(mark)``. The page *moved* when a shift rect (zero-area
rects dropped) intersects that box grown by ``PRECLICK_SHIFT_MARGIN_PX`` only when the crop failed, or when at least
``PRECLICK_CHANGED_SHARE`` of the crop's 8x8 blocks differ from the reference by a mean |luma| above
``PRECLICK_BLOCK_DELTA`` (T-0234 B1/R-2: a taken crop decides by pixels; with a shift near the point, any changed block
of the 3x3 centred on the point's block also counts, and a click executed despite it notes ``executed_despite_shift``). Moved: ``refused`` with ``MOVED_REASON`` (``session.FRESHNESS_REASON``'s text),
nothing dispatched or registered. After ``MAX_MOVED_REFUSALS`` moved refusals in a row in one ``turn_id`` the next call
is not checked (``allowed_after_cap``); an executed ``click_at``, ``type_at``, ``scroll``, ``press_key``, ``wait`` or
``page_overview`` of the turn, or a new turn, resets the run. No reference, or a crop that raised, timed out or could
not be compared with no shift near the point: the call runs as BASE (``unavailable``); with ``shifts_since`` unable to
tell, the pixels alone decide. Every verdict but a plain "not moved" calls ``on_note`` once with ``{"what", "result",
"shift", "changed_share", "ms"}`` only. A ``CancelledError`` propagates. Geometry and pixels only: no page text, URL or
DOM is read, and the model sees nothing new.

**Visual settle** (T-0204 C1-C10; part of **Frames**; supersedes the same rows as B10 for the real driver: BASE plus
one more ``screenshot`` per kept action capture, more only on a changing page): when ``STABLE_SCREEN`` (read at call
time; ``PRECLICK_GUARD`` does not gate it, C-R3) and the driver's *class* has B1's four signals, every action capture
(``_take("action")``: the after-frame of ``click_at``, ``type_at``, ``press_key``, ``scroll`` and ``wait``, the
overview-frame action, T-0203 B2's late re-capture and the guarded-type stop) keeps taking screenshots after T-0200's
blank loop has kept one, each ``STABLE_SCHEDULE_S`` after the previous one's start (the last interval repeating), until
two consecutive shots match: on T-0200's ``BLANK_GRID_W`` luma grid at most ``STABLE_MAX_CELLS`` cells differ by more
than ``STABLE_CELL_DELTA`` (from the third shot on, a cell that also differed in the previous pair is animation and is
not counted), and the shift recorder reports no non-zero-area layout shift since the previous shot's mark (C-R1; a
recorder that cannot tell leaves the cell rule alone). No shot starts past ``STABLE_CAP_S`` from the first or past
T-0200's deadline; each is bounded one interval past the cap and ``FRAME_BUILD_RESERVE_S`` short of the call's end, and
one that raises or times out ends the settle. The last successful shot is the kept capture, registered once with the
metrics read around it and its own shift mark (B's reference); an intermediate shot is never an own frame. A settle of
more than 2 shots, or one that ended unstable, calls ``on_note`` once with ``{"what", "path", "shots", "stable",
"waited_ms"}`` only. ``frame()`` and ``page_overview`` are not settled; a ``CancelledError`` propagates. What the
settle cannot see: a pause followed by a later paint passes it (that is network quiet's, ``wait``'s and the pre-click
guard's job).

The executor talks to the browser only through ``Driver`` (the seam pinned by
``tests/unit/test_computer_executor.py``); the Playwright implementation is ``runtime.PlaywrightDriver``.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from local_astra.computer import fence
from local_astra.computer.egress_proxy import Connect, Resolve
from local_astra.computer.runtime import (
    BrowserEndedError,
    PlaywrightDriver,
    failure_type_class,
    url_allowed,
)
from local_astra.conductor.tool_executor import DryRunToolExecutor, to_pixels
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from local_astra.perception import policy

__all__ = [
    "ACTION_TIMEOUT_S",
    "BLANK_GRID_W",
    "BLANK_LUMA_RANGE",
    "BLANK_RECAPTURE_ACTION_S",
    "BLANK_RECAPTURE_FRAME_S",
    "BLANK_RECAPTURE_INTERVAL_S",
    "CAPTURE_RESERVE_S",
    "CARD_TEXT_REASON",
    "CHECKOUT_REASON",
    "COMBOS",
    "FOCUS_TIMEOUT_S",
    "FOCUS_UNKNOWN_REASON",
    "FRONT_TIMEOUT_S",
    "MAX_CONSECUTIVE_WAITS",
    "MAX_HOSTS",
    "MAX_MOVED_REFUSALS",
    "MAX_OVERVIEW_SCREENS",
    "MAX_WAITS_PER_TURN",
    "MOVED_REASON",
    "NAV_FOLLOW_S",
    "OVERVIEW_NOT_REACHED",
    "OVERVIEW_PAGE_CHANGED",
    "OVERVIEW_STALE",
    "PAGE_CHANGED_REASON",
    "PASTE_REASON",
    "PRECLICK_BLOCK_DELTA",
    "PRECLICK_CHANGED_SHARE",
    "PRECLICK_GUARD",
    "PRECLICK_HALF_PX",
    "PRECLICK_REF_FRAMES",
    "PRECLICK_SHIFT_MARGIN_PX",
    "PRECLICK_TIMEOUT_S",
    "REFUSED_COMBOS",
    "RESET_REASONS",
    "SECRET_FIELD_REASON",
    "STABLE_CAP_S",
    "STABLE_CELL_DELTA",
    "STABLE_MAX_CELLS",
    "STABLE_SCHEDULE_S",
    "STABLE_SCREEN",
    "START_PAGE_ACTION",
    "SUBMIT_REASON",
    "SUBMIT_TIMEOUT_S",
    "WAIT_ACTION",
    "WAIT_RUN_REASON",
    "WAIT_S",
    "WAIT_TURN_REASON",
    "BrowserClosedError",
    "BrowserToolExecutor",
    "Driver",
    "ResetOutcome",
    "ResetPageError",
    "blank_frame",
    "phash",
    "start_page_url_for",
]

#: Bound of one action (dispatch, settle, post-action frame); well under ``TOOL_EXEC_TIMEOUT_MS = 10_000``.
ACTION_TIMEOUT_S = 8.0
#: T-0200 C1 (H-T200-1): a capture is blank when its 8-bit luma, box-downscaled to this many px wide, spans at
#: most ``BLANK_LUMA_RANGE`` (max - min, so one small element counts as content)
BLANK_GRID_W = 64
BLANK_LUMA_RANGE = 8
#: T-0200 C2/C3 (chosen, not measured; read at call time): the wait between two captures of one blank page, the
#: re-capture budgets of the action paths and of ``frame()``, and what an action keeps of ``ACTION_TIMEOUT_S``
BLANK_RECAPTURE_INTERVAL_S = 0.1
BLANK_RECAPTURE_ACTION_S = 3.0
BLANK_RECAPTURE_FRAME_S = 1.5
CAPTURE_RESERVE_S = 1.0
#: T-0200 RV1: what an action's re-capture leaves of ``ACTION_TIMEOUT_S`` for building the kept frame
FRAME_BUILD_RESERVE_S = 0.3
#: T-0200 C9: the note of a capture that was re-taken
BLANK_NOTE_WHAT = "live frame: blank capture retaken"
#: T-0198 C6: the bound on one ``bring_to_front()`` wait (inside ``ACTION_TIMEOUT_S``); a page key is never retried
FRONT_TIMEOUT_S = 1.0
#: Bound of each release (``mouse_up``, ``key_up``) on the failure/cancellation path; its timeout is suppressed.
RELEASE_TIMEOUT_S = 1.0
_COORD_MAX = 999
#: ``press_key`` combos (``tools_v1.json``) as Playwright key names, held in order and released in reverse.
COMBOS: Mapping[str, tuple[str, ...]] = {
    "Enter": ("Enter",),
    "Escape": ("Escape",),
    "Tab": ("Tab",),
    "Backspace": ("Backspace",),
    "Delete": ("Delete",),
    "Space": (" ",),
    "ArrowUp": ("ArrowUp",),
    "ArrowDown": ("ArrowDown",),
    "ArrowLeft": ("ArrowLeft",),
    "ArrowRight": ("ArrowRight",),
    "Home": ("Home",),
    "End": ("End",),
    "PageUp": ("PageUp",),
    "PageDown": ("PageDown",),
    "Ctrl+A": ("Control", "a"),
    "Ctrl+C": ("Control", "c"),
    "Ctrl+V": ("Control", "v"),
    "Ctrl+X": ("Control", "x"),
    "Ctrl+Z": ("Control", "z"),
    "Ctrl+Enter": ("Control", "Enter"),
}
#: Browser-chrome shortcuts (H-T111c-7): refused, nothing dispatched. Navigation arrives with c3's ``open_url``.
REFUSED_COMBOS = frozenset({"Alt+Tab", "Ctrl+T", "Ctrl+W", "Ctrl+L"})
_SCROLL = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}
#: Own frames remembered for the identity check; the oldest are forgotten first.
_MAX_OWN_FRAMES = 1024
#: T-0187: a coordinate call or action refused because the active page changed (``page_serial`` moved); the
#: session answers it with a fresh frame at the next step (``session.NEW_FRAME_REASONS``)
PAGE_CHANGED_REASON = "the active page changed; take a new frame"
#: The refusal of every non-search call once the browser has ended (T-0111g A2).
CLOSED_REASON = "the browser window was closed"
#: T-0134 A4: bound of one layout-metrics read; a read past it is a failed read (no extent for that frame).
#: Chosen, not measured: a CDP ``Page.getLayoutMetrics`` round trip is far below it on loopback.
METRICS_TIMEOUT_S = 0.5
#: T-0134 A1/A2: what an executed click/type reports: viewport pixels (BASE) or the call's own 0-999 units
Echo = Literal["pixels", "model"]
_ECHOES = ("pixels", "model")
_METRIC_KEYS = ("scroll_y", "viewport_height", "content_height")
#: T-0140 A3 (H-T140-2): the overview shows at most the top this-many screens of the page.
MAX_OVERVIEW_SCREENS = 4
#: T-0140 A3: a page at most this many screens tall fits; no overview is taken.
OVERVIEW_FITS_RATIO = 1.05
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
OVERVIEW_MOVED = "the page moved during the overview"
OVERVIEW_UNAVAILABLE = "page overview is not available"
#: Not pinned by the task: a metrics read before the capture that failed or gave no usable viewport.
OVERVIEW_NO_METRICS = "the page size could not be read"
#: Overview frames kept until read; the oldest are forgotten first.
_MAX_OVERVIEWS = 8
#: T-0151 A1-3/A1-4: the overview-frame refusals and the moved-page failure.
OVERVIEW_STALE = "the screen changed since the whole-page picture; take a new frame"
OVERVIEW_PAGE_CHANGED = "the page changed since the whole-page picture; take a new frame"
OVERVIEW_NOT_REACHED = "the page did not scroll to the point; nothing was clicked"
#: T-0160 B-3: the password and card guard.
CARD_TEXT_REASON = "will not type a card number; ask the user to type it themselves"
SECRET_FIELD_REASON = (
    "will not type into a password or card field; ask the user to type it themselves"
)
FOCUS_UNKNOWN_REASON = "could not check the field before typing; nothing was typed"
#: T-0160 B-3 (b), T-0194: bound of one ``focused_field()`` read (read at call time); a timeout or an exception is
#: retried once, so two reads fit in half of ``ACTION_TIMEOUT_S``.
FOCUS_TIMEOUT_S = 2.0
#: T-0194: ``resolved["focus_check"]`` on a web-mode ``FOCUS_UNKNOWN`` outcome, the last read's case (a fixed literal).
FocusCheck = Literal["missing", "timeout", "error", "none", "not_mapping", "unreadable", "body"]
#: T-0160 B-4: the web-mode navigation keys and the refused paste.
START_PAGE_COMBOS = frozenset({"Ctrl+T", "Ctrl+L"})
START_PAGE_PATH = "/web.html"
START_PAGE_ACTION = "opened the start page"


#: T-0202 B1 (H-T202-2): how long an executed ``wait`` sleeps (read at call time)
WAIT_S = 1.0
#: T-0202 B1: an executed wait's action, built once from ``WAIT_S`` at import (not from a patched value)
WAIT_ACTION = f"waited {WAIT_S:g} s"
#: T-0202 B3/B4 (H-T202-3, read at call time): waits in a row, and counted waits per turn
MAX_CONSECUTIVE_WAITS = 3
MAX_WAITS_PER_TURN = 6
#: T-0202 B3/B4 (H-T202-5): the refusals of a wait past either limit
WAIT_RUN_REASON = "you have already waited three times in a row; act on the screen or answer"
WAIT_TURN_REASON = "you have already waited six times in this request; act on the screen or answer"
#: T-0202 B3: the tools whose executed call ends a run of waits (the overview-frame path is a ``click_at``/``type_at``)
_WAIT_RUN_RESETS = frozenset({"click_at", "type_at", "scroll", "press_key", "page_overview"})

#: T-0203 B1/B3 (H-T203-1; chosen, not measured; read at call time): the follow budget of one call, shared by the
#: after-settle and the late follow, each clipped to T-0200's deadline
NAV_FOLLOW_S = 3.0
#: T-0203 A4's bound past a follow's own timeout (a follow that overruns it is ended here, as a raise)
_FOLLOW_SLACK_S = 0.1
#: T-0203 B5: the note of a follow that did not answer ``"none"``
FOLLOW_NOTE_WHAT = "live frame: followed navigation"
_FOLLOW_NOTED = frozenset({"loaded", "ended", "timeout"})

#: T-0204 B2 (James 2026-10-29 (b): on by default; read at call time): the pre-click "did the page move?" guard
PRECLICK_GUARD = True
#: T-0204 B2 (H-T204-2/3; chosen, not measured; read at call time): the crop's half-size around the click point,
#: the margin a shift rect may come within, the mean |luma delta| above which an 8x8 block changed, the share of
#: changed blocks that reads as moved, the bound of the whole check, and the references kept (oldest dropped first)
PRECLICK_HALF_PX = 64
PRECLICK_SHIFT_MARGIN_PX = 40
PRECLICK_BLOCK_DELTA = 16
PRECLICK_CHANGED_SHARE = 0.10
PRECLICK_TIMEOUT_S = 0.3
PRECLICK_REF_FRAMES = 4
#: T-0204 B4 (H-T204-3; read at call time): "moved" refusals in a row in one turn before a call goes unchecked
MAX_MOVED_REFUSALS = 2
#: T-0204 B3 (H-T204-1): the moved refusal, ``session.FRESHNESS_REASON``'s text byte for byte (a test pins both), so
#: T-0187's ``NEW_FRAME_REASONS`` path shows a fresh frame at the next step with no ``session.py`` change
MOVED_REASON = "the screen changed since this frame; take a new frame"
#: T-0204 B6: the note of a check that did not end in a plain "not moved"
PRECLICK_NOTE_WHAT = "live click: page moved check"
#: T-0204 B4: the tools whose executed call ends a run of moved refusals (the overview-frame path is a click/type)
_MOVED_RUN_RESETS = frozenset(
    {"click_at", "type_at", "scroll", "press_key", "wait", "page_overview"}
)
#: T-0204 B1: the driver-class capability the guard needs, all four
_GUARD_METHODS = ("screenshot_region", "arm_shift_recorder", "shift_mark", "shifts_since")
#: T-0204 B2 (b): the side of the crop's comparison blocks, px
_PRECLICK_BLOCK = 8

#: T-0204 C1 (James 2026-10-29 (a): on by default; read at call time): the visual settle before an action's
#: after-frame is kept; ``PRECLICK_GUARD`` does not gate it (ruling C-R3)
STABLE_SCREEN = True
#: T-0204 C2 (H-T204-5; chosen, not measured; read at call time): the start-to-start offsets of the settle's shots
#: (the last repeating), the |luma delta| above which a ``BLANK_GRID_W`` grid cell differs, the differing cells a
#: stable pair may have, and the settle's cap from its first shot
STABLE_SCHEDULE_S = (0.1, 0.25, 0.5)
STABLE_CELL_DELTA = 6
STABLE_MAX_CELLS = 4
STABLE_CAP_S = 1.0
#: T-0204 C9: the note of a settle that took more than 2 shots or ended unstable
STABLE_NOTE_WHAT = "live frame: waited for a stable screen"


def start_page_url_for(origin: str, *, web: bool) -> str:
    """T-0169: the one start-page URL on ``origin``: ``/web.html`` in web mode, else the catalogue ``/``."""
    return origin.rstrip("/") + (START_PAGE_PATH if web else "/")


PASTE_REASON = "pasting is off on real websites"
#: T-0192 A3: the refusal of a page action once the browser has reached a checkout page (web mode).
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
#: T-0206 A3/A4: the refusal of a press that would send a form holding filled personal data (web mode).
SUBMIT_REASON = "this button sends the form; the user presses it themselves"
#: T-0206 A3 (read at call time): bound of one submit read; a timeout, a raise or ``None`` fails open (H-T206-5).
SUBMIT_TIMEOUT_S = 0.3
#: T-0206 A3: the note of a click whose submit read failed (no URL, no label).
SUBMIT_NOTE_WHAT = "live click: submit check"
#: T-0206 A4 (not pinned; the key reads' counterpart of ``SUBMIT_NOTE_WHAT``): a focused read that failed.
SUBMIT_KEY_NOTE_WHAT = "live key: submit check"
#: T-0206 A4: the keys of a chord that make it read the focused element's form (Space: ``COMBOS["Space"]``).
_ENTER_KEYS = frozenset({"Enter", "NumpadEnter"})
_SPACE_KEY = " "
#: T-0206 A4: the end of a ``type_at`` action that pressed Enter (``_plan``, ``_click_action``)
_ENTER_SUFFIX = " and pressed Enter"
#: T-0160 B-6: hosts reported per action; the rest are counted in ``hosts_more``.
MAX_HOSTS = 32


@dataclass(frozen=True)
class _Overview:
    """T-0140 A3's stored picture plus what T-0151 A1-1 needs to register it, all read at the capture."""

    frame: Frame
    clipped: bool
    #: the captured picture's size in CSS px (its pixels before scaling; ``scale="css"``)
    css_size: tuple[int, int]
    generation: int
    page_serial: int
    viewport_height: float
    content_height: float
    #: the phash of ``latest_frame`` at the capture (``None``: no own frame yet)
    latest: str | None


@dataclass(frozen=True)
class _OverviewGeometry:
    """T-0151 A1-1: a registered overview frame, an own frame with page geometry (R-c)."""

    generation: int
    width: int
    height: int
    page_serial: int
    page_width: int
    page_height: int
    viewport_height: float
    content_height: float
    latest: str


class BrowserClosedError(RuntimeError):
    """``frame()`` or ``open()`` after the browser ended under ``end_when_closed`` (T-0111g A2)."""


#: T-0188 A2: why a start-page navigation failed, from event kinds and exception types only
RESET_REASONS = ("beforeunload", "timeout", "target_closed", "other")


@dataclass(frozen=True)
class ResetOutcome:
    """T-0188 A3: what ``reset_page`` did: ``replaced`` when the start-page navigation failed and the page was
    replaced with a fresh one, and then ``reason``, the class of that failure (``RESET_REASONS``); else ``False`` and
    ``None``."""

    replaced: bool
    reason: str | None


class ResetPageError(RuntimeError):
    """T-0188 A3: the start-page navigation failed and so did the replacement. ``reason`` is the navigation's class,
    ``fallback_reason`` the replacement's (``RESET_REASONS``); the args are the two classes only, never a driver
    message, and nothing is chained."""

    def __init__(self, reason: str, fallback_reason: str) -> None:
        super().__init__(reason, fallback_reason)
        self.reason = reason
        self.fallback_reason = fallback_reason


class Driver(Protocol):
    """The browser seam (``tests/unit/test_computer_executor.py`` docstring). Input calls dispatch; ``ensure``,
    ``screenshot``, ``settle`` and ``take_events`` do not."""

    viewport: tuple[int, int]
    generation: int
    page_serial: int

    async def ensure(self) -> None: ...
    async def screenshot(self) -> bytes: ...
    async def settle(self) -> None: ...
    async def mouse_move(self, x: float, y: float) -> None: ...
    async def mouse_down(self) -> None: ...
    async def mouse_up(self) -> None: ...
    async def key_down(self, key: str) -> None: ...
    async def key_up(self, key: str) -> None: ...
    async def type_text(self, text: str) -> None: ...
    async def scroll_by(self, dx: float, dy: float) -> None: ...
    async def goto(self, url: str) -> None: ...
    def take_events(self) -> list[dict[str, Any]]: ...


class _EndAwareDriver(Driver, Protocol):
    """The two facts read only under ``end_when_closed`` (``tests/unit/test_computer_live_options.py`` docstring):
    the current browser is running, and how many pages it has open (0 when not connected)."""

    @property
    def connected(self) -> bool: ...
    @property
    def page_count(self) -> int: ...


class _ReplaceDriver(Driver, Protocol):
    """T-0188 A1: the optional page replacement, looked up on the driver's class: a fresh page at ``url`` becomes the
    active one (``page_serial`` moves) and every other page is closed without running ``beforeunload``."""

    async def replace_page(self, url: str) -> None: ...


class _MetricsDriver(Driver, Protocol):
    """T-0134 A4 (O-1/O-2): the optional layout-metrics read, looked up on the driver's class; CSS px."""

    async def layout_metrics(self) -> Mapping[str, float]: ...


class _OverviewDriver(_MetricsDriver, Protocol):
    """T-0140 A3: the optional whole-page capture, looked up on the driver's class (as ``layout_metrics``)."""

    async def full_page_screenshot(self, max_height_css: int) -> bytes: ...


class _FocusDriver(Driver, Protocol):
    """T-0160 B-3 (b): the optional focused-field read, looked up on the driver's class; ``fence.secret_field``
    keys, or ``None`` when nothing holds the focus."""

    async def focused_field(self) -> Mapping[str, str] | None: ...


class _SubmitDriver(Driver, Protocol):
    """T-0206 A2: the optional submit reads, each looked up on the driver's class (as ``focused_field``):
    ``fence.irreversible_submit``'s facts at a CSS point, or for the focused element; ``None`` when unreadable."""

    async def submit_facts(self, x: float, y: float) -> Mapping[str, Any] | None: ...
    async def submit_facts_focused(self) -> Mapping[str, Any] | None: ...


class _NavigationDriver(Driver, Protocol):
    """T-0174 B1-1: the optional navigation observation, looked up on the driver's class (as ``focused_field``). A
    synchronous opaque mark of the active page's main-frame document, taken at dispatch start, and whether the active
    page now shows a different main-frame document than at that mark (``None`` when the driver cannot tell)."""

    def navigation_mark(self) -> object: ...
    def navigated_since(self, mark: object) -> bool | None: ...


class _FollowDriver(_NavigationDriver, Protocol):
    """T-0203 B1: the optional navigation follow (``runtime.PlaywrightDriver.follow_navigation``), looked up on the
    driver's class: wait for a navigation started after ``mark`` to commit and load, at most ``timeout_s``."""

    async def follow_navigation(self, mark: object, timeout_s: float) -> str: ...


class _CheckoutDriver(Driver, Protocol):
    """T-0192 A2: the optional main-frame URL read, looked up on the driver's class (as ``focused_field``): the
    main-frame navigation URLs since the previous read, the current one last; reading drains them."""

    def main_frame_urls(self) -> tuple[str, ...]: ...


class _GuardDriver(Driver, Protocol):
    """T-0204 B1: group A's driver signals, looked up on the driver's class (``runtime.PlaywrightDriver``): a region
    screenshot in viewport CSS px, the layout-shift recorder's arm, and its synchronous mark and read."""

    async def screenshot_region(self, x: float, y: float, w: float, h: float) -> bytes: ...
    async def arm_shift_recorder(self) -> bool: ...
    def shift_mark(self) -> object: ...
    def shifts_since(
        self, mark: object
    ) -> tuple[tuple[float, float, float, float], ...] | None: ...


@dataclass(frozen=True)
class _GuardRef:
    """T-0204 B1: one own frame's guard reference: the kept screenshot as captured (CSS px, before the ``policy``
    downscale; its luma is decoded only when a click on that frame is checked), the ``shift_mark()`` read right after
    it, and the page it was taken on."""

    jpeg: bytes
    mark: object
    page_serial: int


#: T-0174 B1-1: the tools whose executed outcome carries ``navigated``
_NAVIGATION_TOOLS = frozenset({"click_at", "type_at", "press_key"})
#: no navigation (or T-0204 shift) mark was taken (no capability, another tool, or the mark failed)
_NO_MARK: object = object()
#: one capture's kept screenshot, the metrics read around it and its T-0204 B1 shift mark (``_NO_MARK``: none)
_Shot = tuple[bytes, dict[str, float] | None, dict[str, float] | None, object]


@dataclass(frozen=True)
class _Plan:
    """What one validated call will dispatch, in order: a click, typed text, key chords, a scroll."""

    action: str
    resolved: dict[str, Any]
    point: tuple[int, int] | None = None
    text: str | None = None
    chords: tuple[tuple[str, ...], ...] = ()
    scroll: tuple[int, int] | None = None
    #: the driver's ``page_serial`` when the call's frame was taken (coordinate calls only)
    frame_page: int | None = None
    #: T-0151 A1-2: set when the call's frame is a registered overview frame; ``point`` is then the *page* point
    overview: _OverviewGeometry | None = None
    #: T-0151 A1-5: the call's own 0-999 arguments, for the echo of an overview-frame action
    units: tuple[int, int] | None = None
    #: T-0160 B-4: a URL the plan navigates the active page to (web mode's ``Ctrl+T``/``Ctrl+L``)
    goto: str | None = None
    #: T-0202 B1: the plan is a ``wait`` (no input; ``asyncio.sleep(WAIT_S)`` where an action dispatches)
    wait: bool = False


_DCT_N = 32
_DCT_K = np.arange(_DCT_N)
#: the orthogonal-up-to-scale DCT-II basis, ``C[u, x] = cos(pi * (2x + 1) * u / 2N)``
_DCT = np.cos(np.pi * (2 * _DCT_K[None, :] + 1) * _DCT_K[:, None] / (2 * _DCT_N))


def phash(img: Image.Image) -> str:
    """64-bit DCT perceptual hash, 16 lowercase hex characters: 32x32 grayscale, 2-D DCT-II, the 8x8 lowest
    frequencies compared with their median, MSB first in row-major order (the ``imagehash.phash`` construction).

    A difference hash (the HUD's) was tried first and is too coarse here: a smooth diagonal gradient and a uniform
    page both hash to all zeros, and frame identity is by phash (ruling a918d74: a phash seen before a restart is
    refused for good). Coefficients are rounded so a uniform page hashes the same whatever float noise is left."""
    gray = np.asarray(
        img.convert("L").resize((_DCT_N, _DCT_N), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low = np.round((_DCT @ gray @ _DCT.T)[:8, :8].ravel(), 6)
    bits = low > np.median(low)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def blank_frame(jpeg: bytes) -> bool:
    """T-0200 C1: whether a screenshot is near-uniform (any one colour), from its pixels only: 8-bit luma decoded in
    JPEG draft mode (a DCT-scaled decode, a few ms at most) and box-downscaled to ``BLANK_GRID_W`` px wide, aspect
    kept; blank iff ``max - min <= BLANK_LUMA_RANGE``. Reads no DOM, text, URL or title."""
    luma = _grid(jpeg)
    return bool(int(luma.max()) - int(luma.min()) <= BLANK_LUMA_RANGE)


def _grid(jpeg: bytes) -> np.ndarray[Any, np.dtype[np.int16]]:
    """T-0200 C1's luma grid (also T-0204 C2's): 8-bit luma decoded in JPEG draft mode and box-downscaled to
    ``BLANK_GRID_W`` px wide, aspect kept."""
    grid_w = BLANK_GRID_W
    img: Image.Image = Image.open(io.BytesIO(jpeg))
    w, h = img.size
    grid_h = max(1, round(h * grid_w / w))
    img.draft("L", (grid_w, grid_h))  # a no-op for anything but a JPEG
    small = img.convert("L").resize((grid_w, grid_h), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.int16)


def _luma(jpeg: bytes) -> Image.Image:
    """8-bit luma of a JPEG at full size, decoded in draft mode (grayscale only, no colour conversion)."""
    img: Image.Image = Image.open(io.BytesIO(jpeg))
    img.draft("L", img.size)  # a no-op for anything but a JPEG
    return img.convert("L")


def _point_blocks_changed(
    changed: npt.NDArray[np.bool_], box: tuple[int, int, int, int], point: tuple[int, int]
) -> bool:
    """T-0234 R-2: whether any of the 3x3 blocks centred on the point's block (on the crop's grid, clipped to it)
    changed."""
    bh, bw = changed.shape
    cx = min(max((point[0] - box[0]) // _PRECLICK_BLOCK, 0), bw - 1)
    cy = min(max((point[1] - box[1]) // _PRECLICK_BLOCK, 0), bh - 1)
    return bool(changed[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2].any())


def _changed_blocks(
    crop: bytes, reference: bytes, box: tuple[int, int, int, int], viewport: tuple[int, int]
) -> npt.NDArray[np.bool_] | None:
    """T-0204 B2 (b): per 8x8 block of the crop (on the crop's own grid, rows x cols; a partial edge block is not
    counted), whether its mean absolute luma difference from the same region ``box`` (viewport CSS px, ``x0, y0, x1,
    y1``) of the reference screenshot exceeds ``PRECLICK_BLOCK_DELTA``; the changed share is the mean. ``None`` when
    the two cannot be aligned (the reference is not viewport-sized, the crop not ``box``-sized, or no whole block
    fits)."""
    ref = _luma(reference)
    if ref.size != viewport:
        return None
    x0, y0, x1, y1 = box
    now = _luma(crop)
    if now.size != (x1 - x0, y1 - y0):
        return None
    bw, bh = (x1 - x0) // _PRECLICK_BLOCK, (y1 - y0) // _PRECLICK_BLOCK
    if bw == 0 or bh == 0:
        return None
    w, h = bw * _PRECLICK_BLOCK, bh * _PRECLICK_BLOCK
    a = np.asarray(now, dtype=np.int16)[:h, :w]
    b = np.asarray(ref.crop((x0, y0, x0 + w, y0 + h)), dtype=np.int16)
    blocks = (
        np.abs(a - b)
        .reshape(bh, _PRECLICK_BLOCK, bw, _PRECLICK_BLOCK)
        .mean(axis=(1, 3), dtype=np.float64)
    )
    changed: npt.NDArray[np.bool_] = blocks > PRECLICK_BLOCK_DELTA
    return changed


def _coord(args: Mapping[str, Any], key: str) -> int | None:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= _COORD_MAX:
        return None
    return v


class BrowserToolExecutor:
    """A live ``ToolExecutor`` (``contracts/protocols.py``) over a sandboxed browser (module docstring)."""

    mode = "live"

    def __init__(
        self,
        *,
        now_ms: Callable[[], int],
        viewport: tuple[int, int] = (1280, 800),
        guard: Callable[[ToolInvocation], str | None] | None = None,
        user_data_dir: str | os.PathLike[str] | None = None,
        driver: Driver | None = None,
        headless: bool = True,
        end_when_closed: bool = False,
        allowed_origins: frozenset[str] | None = None,
        echo: Echo = "pixels",
        web: bool = False,
        start_origin: str | None = None,
        proxy_resolve: Resolve | None = None,
        proxy_connect: Connect | None = None,
    ) -> None:
        if not isinstance(echo, str) or echo not in _ECHOES:
            raise ValueError(f"echo must be one of {_ECHOES}, not {echo!r} (T-0134 A1)")
        if web and user_data_dir is not None:
            raise ValueError(
                "web mode uses a fresh profile only; a user_data_dir is refused (T-0160 B-2)"
            )
        if web and start_origin is None:
            raise ValueError("web mode needs the start origin (T-0160 B-1)")
        #: T-0134 A2: the units an executed click/type reports (read by the session by duck typing, A5)
        self.echo: Echo = echo
        #: T-0160: web mode (module docstring)
        self.web = web
        self._start_origin = start_origin
        #: T-0162: the first URL ``open()`` navigated to (``reset_page``'s target outside web mode)
        self._start_url: str | None = None
        profile = fence.check_profile_dir(user_data_dir)
        self._now_ms = now_ms
        self._guard = guard
        self._egress: list[dict[str, str]] = []
        self._end_when_closed = end_when_closed
        self._allowed_origins = allowed_origins
        self._closed = False
        self._driver: Driver = (
            driver
            if driver is not None
            else PlaywrightDriver(
                viewport=viewport,
                on_egress=self._egress.append,
                user_data_dir=profile,
                headless=headless,
                allowed_origins=allowed_origins,
                end_when_closed=end_when_closed,
            )
            if not web
            else PlaywrightDriver(
                viewport=viewport,
                on_egress=self._egress.append,
                headless=headless,
                allowed_origins=allowed_origins,
                end_when_closed=end_when_closed,
                web=True,
                start_origin=start_origin,
                proxy_resolve=proxy_resolve,
                proxy_connect=proxy_connect,
            )
        )
        self._dry = DryRunToolExecutor()
        #: phash -> (generation, width, height, page_serial) of the frames this executor produced
        self._own: dict[str, tuple[int, int, int, int]] = {}
        #: phashes first seen in an earlier generation and captured again since: never accepted again
        self._retired: set[str] = set()
        self._latest: Frame | None = None
        #: (generation, nonce): the phash salt of the current browser generation (ruling 683cc1e)
        self._salt: tuple[int, int] | None = None
        #: T-0134 A4: phash -> the page extent read around that frame's screenshot (``None``: unknown)
        self._extents: dict[str, dict[str, float] | None] = {}
        #: T-0140 A3: overview id -> its scaled frame (and T-0151's capture facts), until read once; never an own frame
        self._overviews: dict[str, _Overview] = {}
        self._overview_n = 0
        #: T-0151 A1-1: phash -> a registered overview frame; apart from ``_own``, ``_extents`` and ``_latest``
        self._overview_frames: dict[str, _OverviewGeometry] = {}
        #: T-0188 A4: events drained to classify a failed start-page navigation, reported with the action's own
        self._carried: list[dict[str, Any]] = []
        #: T-0198: headed browsers come to the front (once per page key) when they act; never when headless
        self.headless = headless
        #: T-0198 C4: the session's per-turn switch (``False`` in a shared-screen turn)
        self.raise_window = True
        #: T-0198: page keys ``(turn_id, generation, page_serial)`` already fronted, or tried
        self._fronted: set[tuple[str, int, int]] = set()
        #: T-0200 C9: called once per capture that took more than one screenshot (set by the composition root)
        self.on_note: Callable[[dict[str, Any]], None] | None = None
        #: T-0202 B3/B4: the turn the wait counts belong to, its current run of waits and its counted waits
        self._wait_turn: str | None = None
        self._wait_run = 0
        self._wait_total = 0
        #: T-0204 B1: phash -> the guard reference of that own frame, at most ``PRECLICK_REF_FRAMES``
        self._refs: dict[str, _GuardRef] = {}
        #: T-0204 B4: the turn the moved refusals belong to, and its current run of them
        self._moved_turn: str | None = None
        self._moved_run = 0

    # ------------------------------------------------------------------ public surface

    @property
    def driver(self) -> Driver:
        return self._driver

    @property
    def latest_frame(self) -> Frame | None:
        return self._latest

    @property
    def egress_log(self) -> tuple[dict[str, str], ...]:
        """The blocked off-loopback attempts, each ``{"host", "scheme"}`` only."""
        return tuple(dict(r) for r in self._egress)

    @property
    def closed(self) -> bool:
        """``True`` once the browser ended under ``end_when_closed`` (module docstring); never ``False`` again."""
        return self._ended()

    def extent_of(self, phash: str) -> dict[str, float] | None:
        """T-0134 A4: the page extent captured with the own frame ``phash``, ``{"above_px", "below_px",
        "viewport_px"}`` in CSS px; ``None`` when that frame is unknown or its read was unstable or failed."""
        value = self._extents.get(phash)
        return dict(value) if value is not None else None

    def overview_of(self, key: object) -> Frame | None:
        """T-0140 A3: the scaled overview frame named by an executed ``page_overview``'s ``resolved["overview"]``,
        once; a second read, or an unknown id, is ``None``."""
        entry = self._overviews.pop(key, None) if isinstance(key, str) else None
        return entry.frame if entry is not None else None

    def overview_frame_of(self, key: object) -> Frame | None:
        """T-0151 A1-1: the overview named ``key``, read from ``overview_of``'s once-only store and registered as an
        *overview frame* (an own frame with page geometry, kept apart from the screen frames) only when the capture
        was not clipped, its CSS width is the viewport's, the generation and ``page_serial`` are the capture's, and
        ``latest_frame`` is still the own frame that was latest at the capture (there was one). Otherwise ``None``,
        and the picture is dropped."""
        entry = self._overviews.pop(key, None) if isinstance(key, str) else None
        if entry is None:
            return None
        d = self._driver
        latest = self._latest.phash if self._latest is not None else None
        if (
            entry.clipped
            or entry.css_size[0] != d.viewport[0]
            or entry.generation != d.generation
            or entry.page_serial != d.page_serial
            or entry.latest is None
            or latest != entry.latest
        ):
            return None
        frame = entry.frame
        assert frame.phash is not None  # every frame this executor builds has one
        self._overview_frames.pop(frame.phash, None)
        self._overview_frames[frame.phash] = _OverviewGeometry(
            generation=entry.generation,
            width=frame.width,
            height=frame.height,
            page_serial=entry.page_serial,
            page_width=entry.css_size[0],
            page_height=entry.css_size[1],
            viewport_height=entry.viewport_height,
            content_height=entry.content_height,
            latest=entry.latest,
        )
        while len(self._overview_frames) > _MAX_OVERVIEWS:
            self._overview_frames.pop(next(iter(self._overview_frames)))
        return frame

    async def frame(self) -> Frame:
        """An own frame of the active page (launches the browser if needed)."""
        if self._ended():
            raise BrowserClosedError(CLOSED_REASON)
        await self._ensure()
        return await self._capture("frame")

    async def open(self, url: str) -> None:
        """Navigate the active page (the harness's; c3 brings the ``open_url`` tool). Loopback http(s) only, and
        only the allowed origins when they are set; in web mode also a public ``https`` URL (T-0160)."""
        allowed = url_allowed(url, self._allowed_origins) or (
            self.web
            and self._start_origin is not None
            and fence.web_url_allowed(url, self._start_origin)
        )
        if not allowed or not url.lower().startswith(("http://", "https://")):
            raise ValueError(
                "open() takes loopback http(s) URLs of the allowed origins only (T-0111c, T-0111g, H-T111c-E)"
            )
        if self._ended():
            raise BrowserClosedError(CLOSED_REASON)
        await self._ensure()
        await self._driver.goto(url)
        await self._driver.settle()
        if self._start_url is None:
            self._start_url = url  # T-0162: the start index the harness opened first

    @property
    def start_page_url(self) -> str | None:
        """T-0169: the page a session starts on, Restart returns to (web mode) and Ctrl+T/Ctrl+L open (web mode),
        decided by this executor's ``web`` flag; ``None`` without a start origin."""
        if self._start_origin is None:
            return None
        return start_page_url_for(self._start_origin, web=self.web)

    async def reset_page(self) -> ResetOutcome:
        """T-0162 (reviewer B): back to the start page for a new session: ``/web.html`` in web mode, else the first
        page ``open()`` was given. A no-op when no browser is live (never launched, ended, or nothing opened).

        T-0188 A1-A3: when that navigation raises (a page that refuses to unload), the page is replaced
        (``_goto_or_replace``) and the outcome says so with the failure's class; when the replacement fails too,
        ``ResetPageError`` carries the classes and never a message."""
        url = self.start_page_url if self.web else self._start_url
        if url is None or self._driver.generation < 1 or self._ended():
            return ResetOutcome(replaced=False, reason=None)
        await self._ensure()
        reason, _events = await self._goto_or_replace(url)
        await self._driver.settle()
        if self._reads_checkout():
            # T-0192 R-A2: Restart is a way out; what led here does not refuse the next action
            with contextlib.suppress(Exception):
                cast(_CheckoutDriver, self._driver).main_frame_urls()
        return ResetOutcome(replaced=reason is not None, reason=reason)

    async def _goto_or_replace(self, url: str) -> tuple[str | None, list[dict[str, Any]]]:
        """T-0188 A1/A4: ``goto(url)``; ``(None, [])`` when it succeeds (nothing else is called). When it raises and
        the driver's class has ``replace_page``, the events recorded so far are drained to classify the failure
        (``_reset_reason``), the page is replaced at ``url``, and the class and the drained events are returned. A
        driver without the capability lets the exception through unchanged; a failed replacement raises
        ``ResetPageError`` outside the ``except`` (the exceptions' messages are never read, kept or chained:
        ``__cause__`` and ``__context__`` are ``None``, T-0188 A-R2)."""
        d = self._driver
        try:
            await d.goto(url)
            return None, []
        except Exception as exc:
            if getattr(type(d), "replace_page", None) is None:
                raise
            events = d.take_events()
            reason = _reset_reason(exc, events)
        fallback: str | None = None
        try:
            await cast(_ReplaceDriver, d).replace_page(url)
        except Exception as exc:
            fallback = failure_type_class(exc)
        if fallback is not None:
            # raised outside the ``except``: neither ``__cause__`` nor ``__context__`` reaches the driver's exception
            raise ResetPageError(reason, fallback)
        return reason, events

    async def close(self) -> None:
        """Close the browser this executor launched (an injected driver is its owner's to close)."""
        if isinstance(self._driver, PlaywrightDriver):
            await self._driver.close()

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        out = await self._execute(inv)
        # T-0202 B3: only an executed action of the same turn ends a run of waits (a new turn resets anyway)
        if out.status == "executed" and inv.name in _WAIT_RUN_RESETS:
            self._wait_counts(inv.turn_id)
            self._wait_run = 0
        # T-0204 B4: so does an executed action (a wait included) end a run of moved refusals
        if out.status == "executed" and inv.name in _MOVED_RUN_RESETS:
            self._moved_counts(inv.turn_id)
            self._moved_run = 0
        return out

    def _moved_counts(self, turn_id: str) -> None:
        """T-0204 B4: start the run of moved refusals afresh when ``turn_id`` is not the turn it belongs to."""
        if turn_id != self._moved_turn:
            self._moved_turn, self._moved_run = turn_id, 0

    def _wait_counts(self, turn_id: str) -> None:
        """T-0202 B3/B4: start the counts afresh when ``turn_id`` is not the turn they belong to."""
        if turn_id != self._wait_turn:
            self._wait_turn, self._wait_run, self._wait_total = turn_id, 0, 0

    def _wait_limit(self, inv: ToolInvocation) -> ToolOutcome | None:
        """T-0202 B3/B4: the refusal of a wait past the run or the per-turn limit (both read at call time); when
        both apply, the run's."""
        self._wait_counts(inv.turn_id)
        if self._wait_run >= MAX_CONSECUTIVE_WAITS:
            return _refused(inv, WAIT_RUN_REASON)
        if self._wait_total >= MAX_WAITS_PER_TURN:
            return _refused(inv, WAIT_TURN_REASON)
        return None

    async def _execute(self, inv: ToolInvocation) -> ToolOutcome:
        if inv.name == "google_search":
            return await self._dry.execute(inv)
        if self._ended():
            return _refused(inv, CLOSED_REASON)
        if inv.name == "page_overview":
            return await self._execute_overview(inv)
        plan = self._plan(inv)
        if isinstance(plan, ToolOutcome):
            return plan
        if self._guard is not None:
            verdict = self._guard(inv)
            if verdict is not None:
                return _refused(inv, verdict)
        if plan.wait:
            limited = self._wait_limit(inv)
            if limited is not None:
                return limited
        d = self._driver
        generation, serial = d.generation, d.page_serial
        self._carried = []
        # T-0200 C3: an after-frame's re-captures stop once less than ``CAPTURE_RESERVE_S`` of the call remains
        deadline = asyncio.get_running_loop().time() + ACTION_TIMEOUT_S - CAPTURE_RESERVE_S
        try:
            # one bound for the whole call: adoption, its load wait and the cap close in ensure(), then dispatch
            async with asyncio.timeout(ACTION_TIMEOUT_S):
                if await self._ensure_ended():
                    return _refused(inv, CLOSED_REASON)
                if generation != 0 and d.generation != generation:
                    return _failed_reason(inv, "browser restarted")
                if d.page_serial != serial or (
                    plan.frame_page is not None and plan.frame_page != d.page_serial
                ):
                    return _refused(inv, PAGE_CHANGED_REASON)
                if self._checkout_refuses(start_page=plan.goto is not None):
                    return _refused(inv, CHECKOUT_REASON)
                mark = self._navigation_mark(inv)
                if plan.overview is not None:
                    return await self._act_on_overview(inv, plan, mark, deadline=deadline)
                replaced: dict[str, Any] = {}
                await self._front(inv.turn_id)
                if plan.point is not None and self._guarding() and await self._moved(inv, plan):
                    return _refused(inv, MOVED_REASON)
                # T-0206 A3/A4: after the moved guard, so what is read is what is pressed
                if plan.point is not None and await self._click_sends(plan.point):
                    return _refused(inv, SUBMIT_REASON)
                if plan.point is None and await self._keys_send(plan.chords):
                    return _refused(inv, SUBMIT_REASON)
                if plan.text is not None and self._guards_focus():
                    stopped = await self._dispatch_guarded(
                        inv, plan, plan.resolved, deadline=deadline
                    )
                    if isinstance(stopped, ToolOutcome):
                        return stopped
                    replaced = stopped
                    if stopped.get("submit_withheld") is True:
                        # T-0206 A4: BASE's action for ``press_enter=False``; no Enter was sent
                        plan = replace(
                            plan, chords=(), action=plan.action.removesuffix(_ENTER_SUFFIX)
                        )
                elif plan.wait:
                    # T-0202 B3/B6: the wait counts once its sleep has begun (a cancellation propagates)
                    self._wait_run += 1
                    self._wait_total += 1
                    await asyncio.sleep(WAIT_S)
                else:
                    await self._dispatch(plan)
                await self._driver.settle()
                await self._front(inv.turn_id)
                after, navigated, arrived = await self._after_settle(plan, mark, deadline)
                events = self._carried + self._driver.take_events()
        except Exception as exc:
            return _failed(inv, exc)
        finally:
            self._carried = []
        resolved = dict(plan.resolved)
        resolved["frame_after"] = after.phash
        resolved.update(_report(events))
        resolved.update(self._hosts_report(events))
        resolved.update(navigated)
        resolved.update(replaced)
        resolved.update(arrived)
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action=plan.action,
            reason=None,
            resolved=resolved,
        )

    async def _front(self, turn_id: str) -> None:
        """T-0198: bring a headed browser's active page to the front, once per page key ``(turn_id, generation,
        page_serial)``, through the driver class's optional ``bring_to_front``. Best effort: the key is marked before
        the try and never retried; an ``Exception`` or a wait past ``FRONT_TIMEOUT_S`` is ignored, a cancellation
        propagates. Called only after every refusal check, before the first dispatch and after a settle."""
        if self.headless or not self.raise_window:
            return
        d = self._driver
        fn = getattr(type(d), "bring_to_front", None)
        if fn is None:
            return
        key = (turn_id, d.generation, d.page_serial)
        if key in self._fronted:
            return
        self._fronted.add(key)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(fn(d), FRONT_TIMEOUT_S)

    async def _ensure(self) -> None:
        """``driver.ensure()`` for ``frame()`` and ``open()``: ``BrowserClosedError`` when the browser ended inside it."""
        if await self._ensure_ended():
            raise BrowserClosedError(CLOSED_REASON)

    async def _ensure_ended(self) -> bool:
        """Run ``driver.ensure()`` and say whether the browser ended inside it (A3 ruling): under
        ``end_when_closed`` the end facts are re-checked after ``ensure()`` returns *or raises*, whatever it raised,
        so a last page that closed while ``ensure()`` adopted a popup or awaited a load ends the executor before any
        dispatch or capture. ``BrowserEndedError`` (the runtime refusing to reopen) ends it outright. Otherwise an
        exception propagates unchanged (in default mode ``_ended()`` reads no fact and stays ``False``)."""
        try:
            await self._driver.ensure()
        except BrowserEndedError:
            self._closed = True
            return True
        except Exception:
            if self._ended():
                return True
            raise
        return self._ended()

    def _ended(self) -> bool:
        """The end-on-close rule (A2), checked before and after every ``ensure()``. Only under ``end_when_closed`` and only
        once a browser has launched does it read the driver's ``connected`` and ``page_count``; the executor's own
        page-cap close never closes the last page, so a page count of 0 is someone else's doing."""
        if self._closed or not self._end_when_closed or self._driver.generation < 1:
            return self._closed
        d = cast(_EndAwareDriver, self._driver)
        if not d.connected or d.page_count == 0:
            self._closed = True
        return self._closed

    # ------------------------------------------------------------------ the page overview (T-0140 A3)

    async def _execute_overview(self, inv: ToolInvocation) -> ToolOutcome:
        """``page_overview``: the rules of every non-search action (the guard, ``ensure()``, the restart and
        page-serial checks, ``ACTION_TIMEOUT_S``), then a whole-page capture between two layout-metrics reads. The
        frame is kept for ``overview_of`` only: never an own frame, an extent or ``latest_frame``. Pixels and CDP
        layout metrics only (A4)."""
        cls = type(self._driver)
        if (
            getattr(cls, "full_page_screenshot", None) is None
            or getattr(cls, "layout_metrics", None) is None
        ):
            return _refused(inv, OVERVIEW_UNAVAILABLE)
        if self._guard is not None:
            verdict = self._guard(inv)
            if verdict is not None:
                return _refused(inv, verdict)
        d = cast(_OverviewDriver, self._driver)
        generation, serial = d.generation, d.page_serial
        try:
            async with asyncio.timeout(ACTION_TIMEOUT_S):
                if await self._ensure_ended():
                    return _refused(inv, CLOSED_REASON)
                if generation != 0 and d.generation != generation:
                    return _failed_reason(inv, "browser restarted")
                if d.page_serial != serial:
                    return _refused(inv, PAGE_CHANGED_REASON)
                if self._checkout_refuses(start_page=False):
                    return _refused(inv, CHECKOUT_REASON)
                before = await self._read_metrics()
                if before is None or before["viewport_height"] <= 0:
                    return _failed_reason(inv, OVERVIEW_NO_METRICS)
                content, viewport = before["content_height"], before["viewport_height"]
                if content <= OVERVIEW_FITS_RATIO * viewport:
                    return ToolOutcome(
                        call_id=inv.call_id,
                        status="executed",
                        action=OVERVIEW_FITS,
                        reason=None,
                        resolved={},
                    )
                screens = content / viewport
                clipped = screens > MAX_OVERVIEW_SCREENS
                height = math.ceil(min(content, MAX_OVERVIEW_SCREENS * viewport))
                jpeg = await d.full_page_screenshot(height)
                after = await self._read_metrics()
                if after != before:
                    return _failed_reason(inv, OVERVIEW_MOVED)
                frame, css_size = self._overview_frame(jpeg)
        except Exception as exc:
            return _failed(inv, exc)
        self._overview_n += 1
        key = f"overview-{self._overview_n}"
        self._overviews[key] = _Overview(
            frame=frame,
            clipped=clipped,
            css_size=css_size,
            generation=d.generation,
            page_serial=d.page_serial,
            viewport_height=viewport,
            content_height=content,
            latest=self._latest.phash if self._latest is not None else None,
        )
        while len(self._overviews) > _MAX_OVERVIEWS:
            self._overviews.pop(next(iter(self._overviews)))
        shown = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action=f"showed the whole page (about {screens:.1f} screens tall{shown}) as a picture before the screen",
            reason=None,
            resolved={"overview": key, "screens": screens, "clipped": clipped},
        )

    def _overview_frame(self, jpeg: bytes) -> tuple[Frame, tuple[int, int]]:
        """The whole-page picture scaled to fit within the viewport's CSS size (aspect kept, never upscaled, even
        sides), then the ``screen_max`` policy; its phash salted as an own frame's, but never registered here. With
        it, the captured picture's own size (CSS px, ``scale="css"``) for T-0151's registration."""
        img: Image.Image = Image.open(io.BytesIO(jpeg))
        img.load()
        w, h = img.size
        vw, vh = self._driver.viewport
        scale = min(1.0, vw / w, vh / h)
        tw, th = (
            (w, h)
            if scale >= 1.0
            else (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        )
        tw, th = policy.target_size(tw, th, policy.SCREEN_BUCKETS[0])
        if (tw, th) != (w, h):
            img = img.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=policy.JPEG_QUALITY)
            jpeg = buf.getvalue()
        generation = self._driver.generation
        if self._salt is None or self._salt[0] != generation:
            self._salt = (generation, secrets.randbits(64))
        frame = Frame(
            source=Source.WINDOW,
            ts_ms=self._now_ms(),
            width=img.width,
            height=img.height,
            jpeg=jpeg,
            phash=f"{int(phash(img), 16) ^ self._salt[1]:016x}",
        )
        return frame, (w, h)

    # ------------------------------------------------------------------ validation (pure)

    def _plan(self, inv: ToolInvocation) -> _Plan | ToolOutcome:
        args = inv.arguments
        if inv.name in ("click_at", "type_at"):
            frame = inv.frame
            if frame is None:
                return _refused(inv, "no frame")
            own = self._own.get(frame.phash) if frame.phash is not None else None
            ovg = (
                self._overview_frames.get(frame.phash)
                if own is None and frame.phash is not None
                else None
            )
            if (
                ovg is not None
                and frame.source == Source.WINDOW
                and (ovg.width, ovg.height) == (frame.width, frame.height)
            ):
                return self._plan_overview(inv, ovg)
            if (
                own is None
                or frame.source != Source.WINDOW
                or (own[1], own[2]) != (frame.width, frame.height)
            ):
                return _refused(inv, "frame not from this browser")
            if frame.phash in self._retired or own[0] != self._driver.generation:
                return _refused(inv, "frame from before a browser restart")
            x, y = _coord(args, "x"), _coord(args, "y")
            if x is None or y is None:
                return _refused(inv, "bad coordinates")
            vw, vh = self._driver.viewport
            point = (to_pixels(x, vw), to_pixels(y, vh))
            resolved: dict[str, Any] = {"x_px": point[0], "y_px": point[1]}
            # T-0134 A2: the echo is in the units of the arguments this executor received, never converted back
            shown = (x, y) if self.echo == "model" else point
            if inv.name == "click_at":
                return _Plan(
                    action=f"clicked at ({shown[0]}, {shown[1]})",
                    resolved=resolved,
                    point=point,
                    frame_page=own[3],
                )
            text, enter = args.get("text"), args.get("press_enter")
            if not isinstance(text, str) or not text or not isinstance(enter, bool):
                return _refused(inv, "bad arguments")
            if fence.has_control_character(text):
                return _refused(inv, "bad arguments: control character")
            if fence.card_like(text):
                return _refused(inv, CARD_TEXT_REASON)
            action = f"clicked at ({shown[0]}, {shown[1]}) and typed {len(text)} characters" + (
                " and pressed Enter" if enter else ""
            )
            return _Plan(
                action=action,
                resolved=resolved | {"text": text, "press_enter": enter},
                point=point,
                text=text,
                chords=(("Enter",),) if enter else (),
                frame_page=own[3],
            )
        if inv.name == "scroll":
            direction = args.get("direction")
            if not isinstance(direction, str) or direction not in _SCROLL:
                return _refused(inv, "bad arguments")
            sx, sy = _SCROLL[direction]
            vw, vh = self._driver.viewport
            dx, dy = sx * (vw * 3 // 4), sy * (vh * 3 // 4)
            return _Plan(
                action=f"scrolled {direction} by {abs(dx + dy)} px",
                resolved={"direction": direction, "dx_px": dx, "dy_px": dy},
                scroll=(dx, dy),
            )
        if inv.name == "press_key":
            combo = args.get("combo")
            if not isinstance(combo, str):
                return _refused(inv, "bad arguments")
            start = self.start_page_url if self.web else None
            if combo in START_PAGE_COMBOS and start is not None:
                return _Plan(action=START_PAGE_ACTION, resolved={"combo": combo}, goto=start)
            if self.web and combo == "Ctrl+V":
                return _refused(inv, PASTE_REASON)
            if combo in REFUSED_COMBOS:
                return _refused(inv, "not available in the browser")
            keys = COMBOS.get(combo)
            if keys is None:
                return _refused(inv, "bad arguments")
            return _Plan(action=f"pressed {combo}", resolved={"combo": combo}, chords=(keys,))
        if inv.name == "wait":
            if args:
                return _refused(inv, "bad arguments")
            return _Plan(action=WAIT_ACTION, resolved={}, wait=True)
        return _refused(inv, "unknown tool")

    def _plan_overview(self, inv: ToolInvocation, ovg: _OverviewGeometry) -> _Plan | ToolOutcome:
        """T-0151 A1-2/A1-3: a ``click_at``/``type_at`` on a registered overview frame, planned against the page
        (``point`` is the page point); BASE's argument validation; refused when the frame is of an earlier generation
        or ``latest_frame`` is no longer the own frame recorded at the capture. The action string is built after the
        scroll (``_act_on_overview``)."""
        args = inv.arguments
        if ovg.generation != self._driver.generation:
            return _refused(inv, "frame from before a browser restart")
        if self._latest is None or self._latest.phash != ovg.latest:
            return _refused(inv, OVERVIEW_STALE)
        x, y = _coord(args, "x"), _coord(args, "y")
        if x is None or y is None:
            return _refused(inv, "bad coordinates")
        point = (to_pixels(x, ovg.page_width), to_pixels(y, ovg.page_height))
        if inv.name == "click_at":
            return _Plan(
                action="",
                resolved={},
                point=point,
                frame_page=ovg.page_serial,
                overview=ovg,
                units=(x, y),
            )
        text, enter = args.get("text"), args.get("press_enter")
        if not isinstance(text, str) or not text or not isinstance(enter, bool):
            return _refused(inv, "bad arguments")
        if fence.has_control_character(text):
            return _refused(inv, "bad arguments: control character")
        if fence.card_like(text):
            return _refused(inv, CARD_TEXT_REASON)
        return _Plan(
            action="",
            resolved={"text": text, "press_enter": enter},
            point=point,
            text=text,
            chords=(("Enter",),) if enter else (),
            frame_page=ovg.page_serial,
            overview=ovg,
            units=(x, y),
        )

    async def _act_on_overview(
        self,
        inv: ToolInvocation,
        plan: _Plan,
        mark: object = _NO_MARK,
        *,
        deadline: float | None = None,
    ) -> ToolOutcome:
        """T-0151 A1-3/A1-4/A1-5, inside ``execute()``'s ``ACTION_TIMEOUT_S`` block after its closed, restart and
        page-serial checks: a metrics read that must match the capture's page (else ``OVERVIEW_PAGE_CHANGED``, nothing
        dispatched); at most one ``scroll_by`` bringing the page point into view, a settle and the moved-page check
        (two equal reads separated by a settle, the content and viewport heights unchanged, the point in view; O-A1-3;
        else ``failed`` ``OVERVIEW_NOT_REACHED``,
        nothing clicked, the frame captured); then the BASE click/type at the viewport point. CDP metrics only."""
        ovg, page_point = plan.overview, plan.point
        assert ovg is not None and page_point is not None and plan.units is not None
        d = self._driver
        before = await self._read_metrics()
        if (
            before is None
            or before["content_height"] != ovg.content_height
            or before["viewport_height"] != ovg.viewport_height
        ):
            return _refused(inv, OVERVIEW_PAGE_CHANGED)
        await self._front(inv.turn_id)
        px, py = page_point
        s, h, c = before["scroll_y"], before["viewport_height"], before["content_height"]
        s_now = s
        if not s <= py < s + h:
            target = min(max(py - h / 2, 0.0), max(0.0, c - h))
            await d.scroll_by(0, round(target - s))
            await d.settle()
            first = await self._read_metrics()
            await d.settle()
            second = await self._read_metrics()
            if (
                first is None
                or first != second
                or first["content_height"] != c
                or first["viewport_height"] != h
                or not first["scroll_y"] <= py < first["scroll_y"] + h
            ):
                await self._front(inv.turn_id)
                after = await self._capture("action", deadline)
                events = d.take_events()
                return ToolOutcome(
                    call_id=inv.call_id,
                    status="failed",
                    action=f"failed {inv.name}: {OVERVIEW_NOT_REACHED}",
                    reason=OVERVIEW_NOT_REACHED,
                    resolved={"frame_after": after.phash} | _report(events),
                )
            s_now = first["scroll_y"]
        scrolled = round(s_now - s)
        point = (px, round(py - s_now))
        # T-0206 A3: the overview-frame click reads what it would press at the viewport point
        if await self._click_sends(point):
            return _refused(inv, SUBMIT_REASON)
        shown = plan.units if self.echo == "model" else point

        def echo(enter: bool) -> str:
            action = _click_action(shown, plan.text, enter)
            if scrolled:
                action = f"scrolled {'down' if scrolled > 0 else 'up'} by {abs(scrolled)} px and {action}"
            return action

        action = echo(bool(plan.chords))
        resolved: dict[str, Any] = {
            "x_px": point[0],
            "y_px": point[1],
            "page_y_px": py,
            "scrolled_px": scrolled,
        }
        resolved |= plan.resolved
        viewport_plan = _Plan(
            action=action, resolved={}, point=point, text=plan.text, chords=plan.chords
        )
        if plan.text is not None and self._guards_focus():
            stopped = await self._dispatch_guarded(inv, viewport_plan, resolved, deadline=deadline)
            if isinstance(stopped, ToolOutcome):
                return stopped
            resolved.update(stopped)
            if stopped.get("submit_withheld") is True:
                # T-0206 A4: BASE's action for ``press_enter=False``; no Enter was sent
                action = echo(False)
                viewport_plan = replace(viewport_plan, action=action, chords=())
        else:
            await self._dispatch(viewport_plan)
        await d.settle()
        await self._front(inv.turn_id)
        after, navigated, arrived = await self._after_settle(viewport_plan, mark, deadline)
        events = d.take_events()
        resolved["frame_after"] = after.phash
        resolved.update(_report(events))
        resolved.update(self._hosts_report(events))
        resolved.update(navigated)
        resolved.update(arrived)
        return ToolOutcome(
            call_id=inv.call_id, status="executed", action=action, reason=None, resolved=resolved
        )

    # ------------------------------------------------------------------ the password and card guard (T-0160 B-3)

    def _guards_focus(self) -> bool:
        """The post-click check runs in web mode, and whenever the driver's class has ``focused_field``."""
        return self.web or getattr(type(self._driver), "focused_field", None) is not None

    async def _focused_secret(
        self,
    ) -> tuple[
        Literal["password", "card", "unknown"] | None, Mapping[str, Any] | None, FocusCheck | None
    ]:
        """The focused field's ``fence.secret_field`` kind; ``"unknown"`` when it could not be read (no method, a
        timeout past ``FOCUS_TIMEOUT_S``, an exception, a reply that is not a mapping). A cancellation propagates.
        Second: the attributes read, when they were a mapping (T-0189: they choose the select-all), else ``None``.
        Third (T-0194): on ``"unknown"`` the last read's case (``FocusCheck``), else ``None``. A read that times out
        or raises is retried once, with no driver call in between; the second answer is judged as a first one."""
        if getattr(type(self._driver), "focused_field", None) is None:
            return "unknown", None, "missing"
        driver = cast(_FocusDriver, self._driver)
        failed: FocusCheck = "timeout"
        for _ in range(2):
            try:
                async with asyncio.timeout(FOCUS_TIMEOUT_S):
                    attrs: object = await driver.focused_field()
            except TimeoutError:
                failed = "timeout"
                continue
            except Exception:
                failed = "error"
                continue
            break
        else:
            return "unknown", None, failed
        if attrs is None:
            # nothing holds the focus: typed into nothing outside web mode (BASE); unknown in web mode (fail closed)
            return ("unknown", None, "none") if self.web else (None, None, None)
        if not isinstance(attrs, Mapping):
            return "unknown", None, "not_mapping"
        kind = fence.secret_field(attrs)
        if kind is None and self.web and _focus_body(attrs):
            # T-0210 A1: a click that left the focus on the page itself (canvas, background) is not typed into
            return "unknown", attrs, "body"
        if kind is None and self.web and not _focus_readable(attrs):
            return "unknown", attrs, "unreadable"
        return kind, attrs, None

    async def _dispatch_guarded(
        self,
        inv: ToolInvocation,
        plan: _Plan,
        resolved: Mapping[str, Any],
        *,
        deadline: float | None = None,
    ) -> ToolOutcome | dict[str, Any]:
        """Click, check the focused field, then type: once typed, the keys the caller adds to ``resolved`` (it settles
        and captures), or the ``failed`` outcome when the field is a secret field or (web mode) could not be checked.
        Nothing is typed and no Enter is pressed then; the frame is re-captured and ``resolved`` holds neither the
        text nor any key. T-0189: a single-line text input (``_single_line``) is selected with ``COMBOS["Ctrl+A"]``
        before the text, so the text replaces it, and the caller's ``resolved`` gains ``replaced: True``."""
        assert plan.text is not None
        await self._dispatch(_Plan(action=plan.action, resolved={}, point=plan.point))
        kind, attrs, check = await self._focused_secret()
        if kind is None or (kind == "unknown" and not self.web):
            extra: dict[str, Any] = {}
            chords = plan.chords
            if await self._withholds_enter(plan, attrs):
                # T-0206 A4: the text goes in, Enter does not; the caller drops Enter from its action
                chords = ()
                extra["submit_withheld"] = True
            if kind is None and attrs is not None and _single_line(attrs):
                await self._dispatch(
                    _Plan(action=plan.action, resolved={}, chords=(COMBOS["Ctrl+A"],))
                )
                extra["replaced"] = True
            await self._dispatch(
                _Plan(action=plan.action, resolved={}, text=plan.text, chords=chords)
            )
            return extra
        reason = FOCUS_UNKNOWN_REASON if kind == "unknown" else SECRET_FIELD_REASON
        d = self._driver
        await d.settle()
        await self._front(inv.turn_id)
        after = await self._capture("action", deadline)
        events = d.take_events()
        out = {k: v for k, v in resolved.items() if k != "text"}
        if kind != "unknown":
            out["secret_field"] = kind
        elif check is not None:
            out["focus_check"] = check
        out["typed"] = 0
        out["frame_after"] = after.phash
        out.update(_report(events))
        out.update(self._hosts_report(events))
        return ToolOutcome(
            call_id=inv.call_id,
            status="failed",
            action=f"failed {inv.name}: {reason}",
            reason=reason,
            resolved=out,
        )

    def _hosts_report(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """T-0160 B-6, web mode only: ``hosts``, the unique hosts of the driver's ``host`` events (the proxy's
        tunnels since the previous action), at most ``MAX_HOSTS``, and ``hosts_more`` beyond that. Outside web mode
        nothing is reported."""
        if not self.web:
            return {}
        hosts = list(
            dict.fromkeys(str(e["host"]) for e in events if e.get("kind") == "host" and "host" in e)
        )
        out: dict[str, Any] = {"hosts": hosts[:MAX_HOSTS]}
        if len(hosts) > MAX_HOSTS:
            out["hosts_more"] = len(hosts) - MAX_HOSTS
        return out

    # ------------------------------------------------------------------ the navigated fact (T-0174 B1)

    def _navigation_mark(self, inv: ToolInvocation) -> object:
        """T-0174 B1-1: the driver's navigation mark at dispatch start, for ``click_at``/``type_at``/``press_key``
        only; ``_NO_MARK`` when the driver's class lacks the capability or the mark could not be taken."""
        cls = type(self._driver)
        if (
            inv.name not in _NAVIGATION_TOOLS
            or getattr(cls, "navigation_mark", None) is None
            or getattr(cls, "navigated_since", None) is None
        ):
            return _NO_MARK
        try:
            return cast(_NavigationDriver, self._driver).navigation_mark()
        except Exception:
            return _NO_MARK

    def _navigated(self, mark: object) -> dict[str, bool]:
        """T-0174 B1-1: ``{"navigated": bool}`` read after the action's last settle, or ``{}`` when no mark was taken
        or the driver cannot tell (``None``, a non-``bool``, or a failed read). The action is unaffected."""
        if mark is _NO_MARK:
            return {}
        return _navigated_key(self._read_navigated(mark))

    def _read_navigated(self, mark: object) -> object:
        """One ``navigated_since(mark)`` read; ``None`` when it raised (the driver cannot tell)."""
        try:
            return cast(_NavigationDriver, self._driver).navigated_since(mark)
        except Exception:
            return None

    async def _after_settle(
        self, plan: _Plan, mark: object, deadline: float | None
    ) -> tuple[Frame, dict[str, bool], dict[str, bool]]:
        """After an action's last settle and fronting: the navigated read, the T-0192 A4 checkout read and the
        post-action capture (BASE), with T-0203 B1-B5's follows when a mark was taken and the driver's class has
        ``follow_navigation`` (module docstring, Following a navigation). The kept capture is registered once."""
        if mark is _NO_MARK or getattr(type(self._driver), "follow_navigation", None) is None:
            navigated = self._navigated(mark)
            arrived = self._arrived()
            after = await self._capture("action", deadline)
            return after, navigated, arrived
        loop = asyncio.get_running_loop()
        end = deadline if deadline is not None else loop.time() + NAV_FOLLOW_S
        budget = NAV_FOLLOW_S
        value = self._read_navigated(mark)
        if value is not False:
            budget -= await self._follow(mark, min(budget, end - loop.time()), late=False)
            value = self._read_navigated(mark)
        arrived = self._arrived()
        shot = await self._take("action", deadline)
        if value is False and _sends_enter(plan) and self._read_navigated(mark) is not False:
            await self._follow(mark, min(budget, end - loop.time()), late=True)
            value = self._read_navigated(mark)
            again = self._arrived()  # main_frame_urls() drains: the two reads are ORed
            arrived = arrived or again
            retaken = await self._retake(deadline)
            if retaken is not None:
                shot = retaken
        return self._register(*shot), _navigated_key(value), arrived

    async def _follow(self, mark: object, timeout_s: float, *, late: bool) -> float:
        """T-0203 B1-B5: one ``follow_navigation(mark, timeout_s)``, skipped when no time is left; returns the seconds
        it took. An ``Exception`` (or an overrun past A4's bound) is ignored; a ``CancelledError`` propagates. A
        result other than ``"none"`` is noted (B5)."""
        if timeout_s <= 0:
            return 0.0
        loop = asyncio.get_running_loop()
        started = loop.time()
        driver = cast(_FollowDriver, self._driver)
        try:
            async with asyncio.timeout(timeout_s + _FOLLOW_SLACK_S):
                result: object = await driver.follow_navigation(mark, timeout_s)
        except Exception:
            return loop.time() - started
        waited = loop.time() - started
        if isinstance(result, str) and result in _FOLLOW_NOTED:
            self._note(
                {
                    "what": FOLLOW_NOTE_WHAT,
                    "path": "action",
                    "result": result,
                    "late": late,
                    "waited_ms": round(waited * 1000),
                }
            )
        return waited

    async def _retake(self, deadline: float | None) -> _Shot | None:
        """T-0203 B2.4/B3: the late re-capture (``_take("action", deadline)``, T-0200's re-capture included), bounded
        so the kept frame is still built inside ``ACTION_TIMEOUT_S``; ``None`` (keep the first capture) when no time
        is left or it raised. A ``CancelledError`` propagates."""
        if deadline is None:
            limit: float | None = None
        else:
            limit = deadline + CAPTURE_RESERVE_S - FRAME_BUILD_RESERVE_S
            if limit <= asyncio.get_running_loop().time():
                return None
        try:
            async with asyncio.timeout_at(limit):
                return await self._take("action", deadline)
        except Exception:
            return None

    # ------------------------------------------------------------------ the checkout stop (T-0192 A2-A4)

    def _reads_checkout(self) -> bool:
        """T-0192 A3/A7: web mode with a driver class that has ``main_frame_urls``; otherwise it is never called."""
        return self.web and getattr(type(self._driver), "main_frame_urls", None) is not None

    def _at_checkout(self) -> bool:
        """One ``main_frame_urls()`` read (it drains): whether any URL it returned is a checkout. Only this boolean
        leaves here; a read that raises or returns anything but a tuple of strings raises ``ValueError`` without
        its message."""
        try:
            urls: object = cast(_CheckoutDriver, self._driver).main_frame_urls()
        except Exception:
            urls = None
        if not isinstance(urls, tuple) or not all(isinstance(u, str) for u in urls):
            raise ValueError("main-frame URLs unreadable")
        return any(fence.checkout_url(u) for u in urls)

    def _checkout_refuses(self, *, start_page: bool) -> bool:
        """T-0192 A3, after ``ensure()`` and its restart and page-serial checks: refuse when the read names a
        checkout or cannot be made (fail closed). The read is made for web mode's start-page keys too (it drains)
        but never refuses them: ``Ctrl+T``/``Ctrl+L`` are a way out."""
        if not self._reads_checkout():
            return False
        try:
            at = self._at_checkout()
        except ValueError:
            at = True
        return at and not start_page

    def _arrived(self) -> dict[str, bool]:
        """T-0192 A4: ``{"checkout": True}`` when the read after the action's last settle names a checkout; ``{}``
        otherwise, outside web mode, without the capability, or when the read fails (the next call's read then
        refuses)."""
        if not self._reads_checkout():
            return {}
        try:
            return {"checkout": True} if self._at_checkout() else {}
        except ValueError:
            return {}

    # ------------------------------------------------------------------ the submit stop (T-0206 A3/A4)

    def _reads_submit(self, name: str) -> bool:
        """Web mode only, and only when the driver's *class* has the read ``name`` (as ``focused_field``)."""
        return self.web and getattr(type(self._driver), name, None) is not None

    async def _submit_read(self, point: tuple[int, int] | None) -> Mapping[str, Any] | None:
        """One submit read (at ``point``, else of the focused element) bounded by ``SUBMIT_TIMEOUT_S``; ``None``
        when it answered ``None`` or not a mapping, raised or timed out. A cancellation propagates."""
        driver = cast(_SubmitDriver, self._driver)
        try:
            async with asyncio.timeout(SUBMIT_TIMEOUT_S):
                if point is None:
                    got: object = await driver.submit_facts_focused()
                else:
                    got = await driver.submit_facts(*point)
        except Exception:
            return None
        return got if isinstance(got, Mapping) else None

    async def _click_sends(self, point: tuple[int, int]) -> bool:
        """T-0206 A3: whether the click at the CSS ``point`` presses a send control (``fence.irreversible_submit``).
        A failed read fails open (H-T206-5) with a note naming nothing of the page."""
        if not self._reads_submit("submit_facts"):
            return False
        facts = await self._submit_read(point)
        if facts is None:
            self._note({"what": SUBMIT_NOTE_WHAT, "result": "unknown"})
            return False
        return fence.irreversible_submit(facts)

    async def _keys_send(self, chords: tuple[tuple[str, ...], ...]) -> bool:
        """T-0206 A4: whether a ``press_key`` chord holding Enter/NumpadEnter (the focused element's form, its default
        button as the control) or Space (only when the focused element is itself the control) sends the form. Other
        keys are never read; a failed read fails open with a note."""
        keys = {key for chord in chords for key in chord}
        enter, space = bool(keys & _ENTER_KEYS), _SPACE_KEY in keys
        if not (enter or space) or not self._reads_submit("submit_facts_focused"):
            return False
        facts = await self._submit_read(None)
        if facts is None:
            self._note({"what": SUBMIT_KEY_NOTE_WHAT, "result": "unknown"})
            return False
        if not fence.irreversible_submit(facts):
            return False
        return enter or _focus_is_control(facts)

    async def _withholds_enter(self, plan: _Plan, attrs: Mapping[str, Any] | None) -> bool:
        """T-0206 A4: ``type_at`` + Enter, read after the focus read and before any key: whether Enter would send the
        focused field's form once the text is in (the focused field counts as filled). A textarea target never
        withholds (Enter is a newline there); a failed read fails open with a note."""
        if not any(key in _ENTER_KEYS for chord in plan.chords for key in chord):
            return False
        if not self._reads_submit("submit_facts_focused"):
            return False
        tag = attrs.get("tag") if attrs is not None else None
        if isinstance(tag, str) and tag.strip().lower() == "textarea":
            return False
        facts = await self._submit_read(None)
        if facts is None:
            self._note({"what": SUBMIT_KEY_NOTE_WHAT, "result": "unknown"})
            return False
        fields = facts.get("fields")
        if isinstance(fields, list) and plan.text:
            marked = [
                dict(f, filled=True) if isinstance(f, Mapping) and f.get("focused") is True else f
                for f in fields[: fence.MAX_FIELDS]
            ]
            facts = dict(facts, fields=marked)
        return fence.irreversible_submit(facts)

    # ------------------------------------------------------------------ the pre-click guard (T-0204 B)

    def _guarding(self) -> bool:
        """T-0204 B1/B7: ``PRECLICK_GUARD`` (read at call time) and all four driver signals on the driver's class."""
        cls = type(self._driver)
        return PRECLICK_GUARD and all(getattr(cls, n, None) is not None for n in _GUARD_METHODS)

    async def _arm(self) -> None:
        """T-0204 B1: arm the shift recorder on the active page before a capture's screenshot (the driver makes it
        idempotent per page and bounds it); a failure only leaves the page unarmed. A cancellation propagates."""
        with contextlib.suppress(Exception):
            await cast(_GuardDriver, self._driver).arm_shift_recorder()

    def _shift_mark(self) -> object:
        """T-0204 B1: ``shift_mark()`` right after a capture's screenshot; ``_NO_MARK`` when it raised (no reference)."""
        try:
            return cast(_GuardDriver, self._driver).shift_mark()
        except Exception:
            return _NO_MARK

    async def _moved(self, inv: ToolInvocation, plan: _Plan) -> bool:
        """T-0204 B2-B6: whether the page under ``plan.point`` moved since the frame ``inv`` carries (module
        docstring, Pre-click guard). After ``MAX_MOVED_REFUSALS`` moved verdicts in a row in the turn the call is not
        checked. Every verdict but a plain "not moved" is noted once. Never raises but ``CancelledError``."""
        assert plan.point is not None and inv.frame is not None
        loop = asyncio.get_running_loop()
        started = loop.time()
        self._moved_counts(inv.turn_id)
        if self._moved_run >= MAX_MOVED_REFUSALS:
            self._moved_note("allowed_after_cap", None, None, started)
            return False
        ref = self._refs.get(inv.frame.phash) if inv.frame.phash is not None else None
        if ref is None:
            self._moved_note("unavailable", None, None, started)
            return False
        d = cast(_GuardDriver, self._driver)
        px, py = plan.point
        half = PRECLICK_HALF_PX
        vw, vh = d.viewport
        box = (max(0, px - half), max(0, py - half), min(vw, px + half), min(vh, py + half))
        share: float | None = None
        near_point = False  # T-0234 R-2: any of the point's 3x3 blocks changed
        if box[2] > box[0] and box[3] > box[1]:
            try:
                # B5: a crop that raises, times out or cannot be compared leaves the pixel signal undecided
                async with asyncio.timeout_at(started + PRECLICK_TIMEOUT_S):
                    crop = await d.screenshot_region(
                        box[0], box[1], box[2] - box[0], box[3] - box[1]
                    )
                changed = _changed_blocks(crop, ref.jpeg, box, (vw, vh))
                if changed is not None:
                    share = float(np.count_nonzero(changed)) / float(changed.size)
                    near_point = _point_blocks_changed(changed, box, plan.point)
            except Exception:
                share, near_point = None, False
        shift = self._shift_near(ref.mark, plan.point)
        # T-0234 B1/B2 + R-2: a taken crop decides by pixels (with a shift near the point, the point's 3x3 blocks
        # count too); only without one does a shift near the point decide alone (BASE)
        if share is None:
            moved = shift is True
        else:
            moved = share >= PRECLICK_CHANGED_SHARE or (shift is True and near_point)
        if moved:
            self._moved_run += 1
            self._moved_note("refused", shift, share, started)
            return True
        if share is None:
            self._moved_note("unavailable", shift, None, started)
        elif shift is True:
            self._moved_note("executed_despite_shift", shift, share, started)  # T-0234 R-2
        return False

    def _shift_near(self, mark: object, point: tuple[int, int]) -> bool | None:
        """T-0204 B2 (a): whether a layout-shift rect since ``mark`` (previous or current, as the driver lists them)
        intersects the point's box grown by ``PRECLICK_SHIFT_MARGIN_PX``; zero-area rects are dropped (a source that
        moved in or out of the viewport) and a rect that cannot be read counts as near (fail safe). ``None`` when the
        driver cannot tell."""
        try:
            rects: object = cast(_GuardDriver, self._driver).shifts_since(mark)
        except Exception:
            return None
        if rects is None:
            return None
        if not isinstance(rects, tuple):
            return True
        reach = PRECLICK_HALF_PX + PRECLICK_SHIFT_MARGIN_PX
        x0, y0, x1, y1 = point[0] - reach, point[1] - reach, point[0] + reach, point[1] + reach
        for rect in rects:
            try:
                x, y, w, h = (float(v) for v in rect)
            except (TypeError, ValueError):
                return True
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                return True
            if w <= 0 or h <= 0:
                continue
            if x < x1 and x + w > x0 and y < y1 and y + h > y0:
                return True
        return False

    def _moved_note(
        self, result: str, shift: bool | None, share: float | None, started: float
    ) -> None:
        """T-0204 B6: the guard's note, exactly these five keys (no URL, host, title, phash or coordinate)."""
        self._note(
            {
                "what": PRECLICK_NOTE_WHAT,
                "result": result,
                "shift": shift,
                "changed_share": share,
                "ms": round((asyncio.get_running_loop().time() - started) * 1000),
            }
        )

    # ------------------------------------------------------------------ dispatch and frames

    async def _dispatch(self, plan: _Plan) -> None:
        """Dispatch ``plan``; on any exit with a button or a key still down, release it before propagating."""
        d = self._driver
        button = False
        held: list[str] = []
        try:
            if plan.goto is not None:
                _, self._carried = await self._goto_or_replace(plan.goto)
            if plan.point is not None:
                await d.mouse_move(*plan.point)
                button = True
                await d.mouse_down()
                await d.mouse_up()
                button = False
            if plan.text is not None:
                await d.type_text(plan.text)
            for chord in plan.chords:
                for key in chord:
                    held.append(key)
                    await d.key_down(key)
                for key in reversed(chord):
                    await d.key_up(key)
                    held.remove(key)
            if plan.scroll is not None:
                await d.scroll_by(*plan.scroll)
        finally:
            if button or held:
                await self._release(button, held)

    async def _release(self, button: bool, held: list[str]) -> None:
        """Release what is still down. Best effort: on this path the action has already failed or been
        cancelled, and a dead browser holds nothing. Each release is bounded by ``RELEASE_TIMEOUT_S``, so a hung
        browser cannot keep ``execute()`` past its action timeout here."""
        if button:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._driver.mouse_up(), RELEASE_TIMEOUT_S)
        for key in reversed(held):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._driver.key_up(key), RELEASE_TIMEOUT_S)

    def _note(self, data: dict[str, Any]) -> None:
        """T-0200 C9: hand a note to ``on_note``; whatever it raises is swallowed (a log never fails a capture)."""
        if self.on_note is None:
            return
        with contextlib.suppress(Exception):
            self.on_note(data)

    async def _read_metrics(self) -> dict[str, float] | None:
        """T-0134 A4 (O-2): one layout-metrics read, or ``None`` when the driver's class has no
        ``layout_metrics`` (nothing is read) or the read failed, timed out or gave anything but three finite
        numbers. A cancellation propagates."""
        if getattr(type(self._driver), "layout_metrics", None) is None:
            return None
        driver = cast(_MetricsDriver, self._driver)
        try:
            async with asyncio.timeout(METRICS_TIMEOUT_S):
                metrics = await driver.layout_metrics()
            values = {key: float(metrics[key]) for key in _METRIC_KEYS}
        except Exception:
            return None
        return values if all(math.isfinite(v) for v in values.values()) else None

    def _settling(self) -> bool:
        """T-0204 C1/C10: ``STABLE_SCREEN`` (read at call time) and B1's four driver signals on the driver's class;
        ``PRECLICK_GUARD`` does not gate it (C-R3)."""
        cls = type(self._driver)
        return STABLE_SCREEN and all(getattr(cls, n, None) is not None for n in _GUARD_METHODS)

    async def _shot(self, marking: bool = False) -> _Shot:
        """One screenshot between two layout-metrics reads (T-0134 A4; the second only when the first gave one).
        T-0204 B1: when the guard is on and the driver's class has its capability, the shift recorder is armed before
        the screenshot and ``shift_mark()`` is read right after it (``_NO_MARK`` otherwise); ``marking`` (an action
        capture that settles, T-0204 C-R1) does the same with the guard off."""
        guarding = marking or self._guarding()
        if guarding:
            await self._arm()
        metrics_before = await self._read_metrics()
        jpeg = await self._driver.screenshot()
        mark = self._shift_mark() if guarding else _NO_MARK
        metrics_after = await self._read_metrics() if metrics_before is not None else None
        return jpeg, metrics_before, metrics_after, mark

    async def _capture(
        self, path: Literal["action", "frame"], deadline: float | None = None
    ) -> Frame:
        """An own frame of the active page. T-0200 C2/C3/C8/C9: while the capture is blank (``blank_frame``) and the
        budget allows another wait, wait ``BLANK_RECAPTURE_INTERVAL_S`` and take it again; the budget is
        ``BLANK_RECAPTURE_ACTION_S`` (``path="action"``, never past ``deadline``, the call's ``ACTION_TIMEOUT_S`` end
        less ``CAPTURE_RESERVE_S``) or ``BLANK_RECAPTURE_FRAME_S``. The first screenshot raising propagates; a later
        one raising counts as a blank attempt. Only the kept capture is registered. A cancellation propagates."""
        return self._register(*await self._take(path, deadline))

    async def _take(self, path: Literal["action", "frame"], deadline: float | None = None) -> _Shot:
        """``_capture``'s screenshots (T-0200's blank re-capture and its note): the kept screenshot, the metrics
        read around it and its shift mark (T-0204 B1), not registered."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        settling = path == "action" and self._settling()
        shot_started = started
        jpeg, metrics_before, metrics_after, mark = await self._shot(settling)
        blank = blank_frame(jpeg)
        if blank:
            interval = BLANK_RECAPTURE_INTERVAL_S
            budget = BLANK_RECAPTURE_ACTION_S if path == "action" else BLANK_RECAPTURE_FRAME_S
            end = started + budget
            # RV1: a re-capture is bounded by the time left, not only in when it starts; every path stops one
            # interval past its budget, and the action paths also ``FRAME_BUILD_RESERVE_S`` short of the call's
            # ``ACTION_TIMEOUT_S`` end, whichever is sooner. Its ``TimeoutError`` is C8's raising re-capture.
            hard = end + interval
            if path == "action" and deadline is not None:
                hard = min(hard, deadline + CAPTURE_RESERVE_S - FRAME_BUILD_RESERVE_S)
                end = min(end, deadline)
            captures = 1
            while blank and loop.time() + interval <= end:
                await asyncio.sleep(interval)
                remaining = hard - loop.time()
                if remaining <= 0:
                    break
                captures += 1
                try:
                    shot_started = loop.time()
                    async with asyncio.timeout(remaining):
                        shot = await self._shot(settling)
                    shot_blank = blank_frame(shot[0])
                except Exception:
                    continue  # C8 (amended): one blank attempt; the last successful capture stays the kept one
                jpeg, metrics_before, metrics_after, mark = shot
                blank = shot_blank
            if captures > 1:
                self._note(
                    {
                        "what": BLANK_NOTE_WHAT,
                        "path": path,
                        "captures": captures,
                        "blank": blank,
                        "waited_ms": round((loop.time() - started) * 1000),
                    }
                )
        if settling:
            return await self._settle(
                (jpeg, metrics_before, metrics_after, mark), shot_started, deadline
            )
        return jpeg, metrics_before, metrics_after, mark

    async def _settle(self, kept: _Shot, first: float, deadline: float | None) -> _Shot:
        """T-0204 C2/C9 (module docstring, Visual settle): from the capture T-0200's blank loop kept (started at
        ``first``), more shots on ``STABLE_SCHEDULE_S`` (start to start, the last interval repeating) until a pair is
        stable or the next shot would start past ``STABLE_CAP_S`` from ``first`` or past ``deadline``; the last
        successful shot is returned, not registered. A shot is bounded one interval past the cap (and
        ``FRAME_BUILD_RESERVE_S`` short of the call's end, as T-0200's RV1); one that raises or times out ends the
        settle unstable. A cancellation propagates."""
        loop = asyncio.get_running_loop()
        schedule = tuple(STABLE_SCHEDULE_S)
        cap = STABLE_CAP_S
        grid = _grid(kept[0])
        previous: np.ndarray[Any, np.dtype[np.bool_]] | None = None
        shots, stable = 1, False
        start = first
        while True:
            interval = schedule[min(shots - 1, len(schedule) - 1)]
            nxt = start + interval
            if nxt - first > cap or (deadline is not None and nxt > deadline):
                break
            await asyncio.sleep(max(0.0, nxt - loop.time()))
            hard = first + cap + interval
            if deadline is not None:
                hard = min(hard, deadline + CAPTURE_RESERVE_S - FRAME_BUILD_RESERVE_S)
            start = loop.time()
            if hard <= start:
                break
            shots += 1
            try:
                async with asyncio.timeout_at(hard):
                    shot = await self._shot(True)
                now = _grid(shot[0])
            except Exception:
                break  # the last successful shot stays the kept one (T-0200 C8's raising re-capture)
            shifted = self._shifted(kept[3])
            kept = shot
            if now.shape != grid.shape:
                grid, previous = now, None
                continue
            differs = np.abs(now - grid) > STABLE_CELL_DELTA
            grid = now
            counted = differs if previous is None or shots < 3 else differs & ~previous
            previous = differs
            if int(np.count_nonzero(counted)) <= STABLE_MAX_CELLS and not shifted:
                stable = True
                break
        if shots > 2 or not stable:
            self._note(
                {
                    "what": STABLE_NOTE_WHAT,
                    "path": "action",
                    "shots": shots,
                    "stable": stable,
                    "waited_ms": round((loop.time() - first) * 1000),
                }
            )
        return kept

    def _shifted(self, mark: object) -> bool:
        """T-0204 C-R1: whether the shift recorder reports a non-zero-area layout-shift entry since ``mark`` (the
        previous shot's); ``False`` when it cannot tell (no mark, ``None``, a read that raised), so the cell rule
        alone decides. An entry that cannot be read counts as a shift."""
        if mark is _NO_MARK:
            return False
        try:
            rects: object = cast(_GuardDriver, self._driver).shifts_since(mark)
        except Exception:
            return False
        if rects is None:
            return False
        if not isinstance(rects, tuple):
            return True
        for rect in rects:
            try:
                x, y, w, h = (float(v) for v in rect)
            except (TypeError, ValueError):
                return True
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                return True
            if w > 0 and h > 0:
                return True
        return False

    def _register(
        self,
        jpeg: bytes,
        metrics_before: dict[str, float] | None,
        metrics_after: dict[str, float] | None,
        mark: object = _NO_MARK,
    ) -> Frame:
        """Build the own frame of a kept capture and register it (``_own``, ``_extents``, ``latest_frame``); with a
        shift mark, also its T-0204 B1 guard reference (``_remember_ref``)."""
        captured = jpeg
        img: Image.Image = Image.open(io.BytesIO(jpeg))
        img.load()
        w, h = img.size
        tw, th = policy.target_size(w, h, policy.SCREEN_BUCKETS[0])
        if (tw, th) != (w, h):
            img = img.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=policy.JPEG_QUALITY)
            jpeg = buf.getvalue()
        generation = self._driver.generation
        if self._salt is None or self._salt[0] != generation:
            self._salt = (generation, secrets.randbits(64))
        frame = Frame(
            source=Source.WINDOW,
            ts_ms=self._now_ms(),
            width=img.width,
            height=img.height,
            jpeg=jpeg,
            phash=f"{int(phash(img), 16) ^ self._salt[1]:016x}",
        )
        prev = self._own.pop(frame.phash, None)
        if prev is not None and prev[0] != generation:
            # seen before a restart (now only by a salt collision): refused for good
            self._retired.add(frame.phash)
        self._own[frame.phash] = (
            generation,
            frame.width,
            frame.height,
            self._driver.page_serial,
        )
        self._extents[frame.phash] = _extent(metrics_before, metrics_after)
        while len(self._own) > _MAX_OWN_FRAMES:
            oldest = next(iter(self._own))
            self._own.pop(oldest)
            self._extents.pop(oldest, None)
        if mark is not _NO_MARK:
            self._remember_ref(frame.phash, _GuardRef(captured, mark, self._driver.page_serial))
        self._latest = frame
        return frame

    def _remember_ref(self, key: str, ref: _GuardRef) -> None:
        """T-0204 B1: keep ``ref`` for the own frame ``key``, at most ``PRECLICK_REF_FRAMES`` (oldest dropped first).
        A frame captured again replaces its reference with the newest capture and mark and moves it to newest (B-R4):
        the session shows the model every capture, so the call carrying that phash acts on the newest; a click
        carrying an earlier, pre-change frame still reads that frame's own reference."""
        self._refs.pop(key, None)
        self._refs[key] = ref
        while len(self._refs) > max(1, PRECLICK_REF_FRAMES):
            self._refs.pop(next(iter(self._refs)))


def _extent(
    before: Mapping[str, float] | None, after: Mapping[str, float] | None
) -> dict[str, float] | None:
    """T-0134 A4: the page beyond the viewport from two metrics reads around one screenshot (CSS px); ``None``
    when either read is missing, the two differ (the page moved while it was captured) or the viewport is empty.
    ``above = scroll_y``, ``below = content_height - scroll_y - viewport_height`` (never negative)."""
    if before is None or after is None or dict(before) != dict(after):
        return None
    scroll_y, viewport = before["scroll_y"], before["viewport_height"]
    if viewport <= 0:
        return None
    below = max(0.0, before["content_height"] - scroll_y - viewport)
    return {"above_px": max(0.0, scroll_y), "below_px": below, "viewport_px": viewport}


#: T-0160 B-3, reviewer A item 3 (orchestrator ruling): the focused elements web mode types into without more checks.
#: T-0210 A1: ``body`` left this list; only a contenteditable ``body`` (designMode) is typed into.
_TYPABLE_TAGS = frozenset({"input", "textarea", "select"})


def _focus_body(attrs: Mapping[str, Any]) -> bool:
    """T-0210 A1: web mode: the focus is the page's ``body`` (tag trimmed and casefolded) and it is not
    contenteditable, as after a click on a canvas, a bare background or a non-focusable element."""
    tag = attrs.get("tag")
    editable = attrs.get("contenteditable")
    return (
        isinstance(tag, str)
        and tag.strip().casefold() == "body"
        and not (isinstance(editable, str) and editable.strip().lower() == "true")
    )


def _focus_readable(attrs: Mapping[str, Any]) -> bool:
    """Web mode: the focused element is one whose field can be judged: ``input``, ``textarea``, ``select`` or a
    contenteditable element (a designMode ``body`` included), and not a custom element with no readable shadow root
    (``opaque``)."""

    def get(key: str) -> str:
        value = attrs.get(key)
        return value.strip().lower() if isinstance(value, str) else ""

    if get("opaque") == "true":
        return False
    return get("tag") in _TYPABLE_TAGS or get("contenteditable") == "true"


#: T-0189: the ``input`` types that hold one line of text; a missing ``type`` reads as empty (a text input).
_SINGLE_LINE_TYPES = frozenset({"text", "search", "url", "email", "tel", "number", ""})


def _single_line(attrs: Mapping[str, Any]) -> bool:
    """T-0189: the focused element is a single-line text ``input`` (selected before typing). A ``textarea``, a
    contenteditable or opaque element, ``body``, ``select`` and every other ``input`` type are not."""

    def get(key: str) -> str:
        value = attrs.get(key)
        return value.strip().lower() if isinstance(value, str) else ""

    if get("opaque") == "true" or get("contenteditable") == "true":
        return False
    return get("tag") == "input" and get("type") in _SINGLE_LINE_TYPES


def _navigated_key(value: object) -> dict[str, bool]:
    """T-0174 B1-1: ``{"navigated": value}`` only from a ``bool``."""
    return {"navigated": value} if isinstance(value, bool) else {}


def _focus_is_control(facts: Mapping[str, Any]) -> bool:
    """T-0206 A4: Space presses the control only when the focused element is itself the control: not when the read
    says otherwise (``focused_control`` false) or marks a focused field."""
    if facts.get("focused_control") is False:
        return False
    fields = facts.get("fields")
    if isinstance(fields, list):
        return not any(isinstance(f, Mapping) and f.get("focused") is True for f in fields)
    return True


def _sends_enter(plan: _Plan) -> bool:
    """T-0203 B2: the plan sends Enter (``type_at`` with ``press_enter``, a ``press_key`` chord holding ``Enter``)."""
    return any("Enter" in chord for chord in plan.chords)


def _click_action(shown: tuple[int, int], text: str | None, enter: bool) -> str:
    """T-0151 A1-5: BASE's ``click_at``/``type_at`` action string (``_plan``'s), for an overview-frame action."""
    action = f"clicked at ({shown[0]}, {shown[1]})"
    if text is None:
        return action
    return f"{action} and typed {len(text)} characters" + (" and pressed Enter" if enter else "")


def _reset_reason(exc: BaseException, events: list[dict[str, Any]]) -> str:
    """T-0188 A2: ``beforeunload`` when a beforeunload dialog event was recorded during the attempt (any other dialog
    does not count), else the exception type's class (``runtime.failure_type_class``). Never the message."""
    if any(e.get("kind") == "dialog" and e.get("type") == "beforeunload" for e in events):
        return "beforeunload"
    return failure_type_class(exc)


def _report(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Group the driver's events into ``resolved`` keys: ``blocked`` (downloads, file choosers), ``dismissed``
    (dialogs) and ``closed_pages`` (popups closed by the page cap). Absent keys mean nothing happened."""
    blocked = [{"kind": e["kind"]} for e in events if e.get("kind") in ("download", "file_chooser")]
    dismissed = [{"dialog": str(e.get("type"))} for e in events if e.get("kind") == "dialog"]
    closed = sum(1 for e in events if e.get("kind") == "page_closed")
    out: dict[str, Any] = {}
    if blocked:
        out["blocked"] = blocked
    if dismissed:
        out["dismissed"] = dismissed
    if closed:
        out["closed_pages"] = closed
    return out


def _refused(inv: ToolInvocation, reason: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=inv.call_id,
        status="refused",
        action=f"refused {inv.name}: {reason}",
        reason=reason,
        resolved={},
    )


def _failed_reason(inv: ToolInvocation, reason: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=inv.call_id,
        status="failed",
        action=f"failed {inv.name}: {reason}",
        reason=reason,
        resolved={},
    )


def _failed(inv: ToolInvocation, exc: Exception) -> ToolOutcome:
    """The reason names the exception type only: Playwright messages carry call logs with URLs."""
    return _failed_reason(inv, f"{type(exc).__name__} during {inv.name}")
