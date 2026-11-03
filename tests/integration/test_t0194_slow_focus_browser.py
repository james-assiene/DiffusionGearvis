"""T-0194 criterion 9 end to end: on a page whose main thread is busy right after the click, the focus read before
typing must wait for the page, so a password field is still recognised and refused (docs/tasks/T-0194.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate, imported).
``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0194_slow_focus_browser.py -q -m browser``. The fixture
server binds 127.0.0.1 on an ephemeral port; no row touches the network.

**Rig.** ``t0194_slow_focus.html`` (``tests/fixtures/browser/``, MANIFEST.md) through T-0111c's loopback fixture
server and a default ``BrowserToolExecutor`` (not web mode: the real ``PlaywrightDriver`` always has
``focused_field``, so the guarded path runs). ``#q`` (text, ``old``) and ``#pw`` (password, ``old-secret``); each
field's ``mouseup`` schedules a 1.0 s main-thread busy-wait (``setTimeout 0``), so ``PlaywrightDriver.focused_field``
(a ``frame.evaluate``) answers only after about 1.0 s.

**Expected values by hand.** BASE (``FOCUS_TIMEOUT_S`` 0.5 s, one read): the read times out, and outside web mode an
unknown read types at the caret, so ``#pw`` reads ``old-secret`` + the typed text and the call is ``executed`` (the
bug the planner found). With the fix (2.0 s bound, one retry) the read answers ``type: password`` after ~1.0 s: the
call is ``failed`` with ``SECRET_FIELD_REASON``, ``secret_field: "password"``, ``typed: 0`` and ``#pw`` keeps
``old-secret``. Partner row: ``#q`` is read as a single-line text input and replaced (T-0189): ``new-q-7``.

**State.** Written ``pending(task="T-0194")`` and activated for ``/implement T-0194``.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless (versions.lock ``[browser]``), 1280x800 at DSF 1, one
browser per row, busy-wait 1.0 s, n = 1. BASE = 0b50f3a8 (611068bb's executor).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_astra.contracts.toolcall import ToolOutcome
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    execute,
    executor,
    inv,
    run,
    server,
)
from tests.integration.test_computer_web_browser import center, value_of

pytestmark = [pytest.mark.integration, pytest.mark.browser]

PAGE = "t0194_slow_focus.html"
#: T-0160 B-3 (h), exact words
SECRET = "will not type into a password or card field; ask the user to type it themselves"


async def type_into(ex: Any, origin: str, selector: str, text: str) -> ToolOutcome:
    await run(ex.open(f"{origin}/{PAGE}"))
    x, y = await center(ex, selector)
    f = await run(ex.frame())
    return await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=False))


async def busy_runs(ex: Any) -> list[int]:
    """The busy-waits the page ran (their lengths in ms): the premise that the page was slow after the click."""
    return list(await ex.driver.page.evaluate("window.__busyMs"))


async def test_slow_page_password_refused(server: str) -> None:  # noqa: F811  (the fixture)
    """Criterion 9, adversarial (the named failure: a slow page turns the focus read into "unknown", and outside web
    mode an unknown read types at the caret, into a password field): ``failed`` with ``SECRET_FIELD_REASON``,
    ``secret_field: "password"``, ``typed: 0``; ``#pw`` keeps ``old-secret``. BASE: ``#pw`` reads
    ``old-secret`` + the typed text."""
    async with executor() as ex:
        out = await type_into(ex, server, "#pw", "hunter2")
        await asyncio.sleep(0.3)
        got = await value_of(ex, "#pw")
        runs = await busy_runs(ex)
        assert runs and min(runs) >= 1000, f"premise: the page was busy after the click: {runs}"
        assert got == "old-secret", (
            f"T-0194 criterion 9: text landed in the password field on a slow page; #pw reads {got!r}; "
            f"outcome {out.status} {out.reason!r}"
        )
        assert (out.status, out.reason) == ("failed", SECRET), out
        assert out.resolved.get("secret_field") == "password", out.resolved
        assert out.resolved.get("typed") == 0, out.resolved


async def test_slow_page_text_replaced(server: str) -> None:  # noqa: F811  (the fixture)
    """Criterion 9, partner (the guard too wide: a slow plain field must still be typed into): ``#q`` (``old``) is
    ``executed`` and holds exactly the typed text, ``resolved["replaced"] is True`` (the read answered a single-line
    text input, T-0189). BASE: the read times out, the text is typed at the caret: ``oldnew-q-7``, no ``replaced``."""
    async with executor() as ex:
        out = await type_into(ex, server, "#q", "new-q-7")
        await asyncio.sleep(0.3)
        got = await value_of(ex, "#q")
        runs = await busy_runs(ex)
        assert runs and min(runs) >= 1000, f"premise: the page was busy after the click: {runs}"
        assert out.status == "executed", out
        assert got == "new-q-7", (
            f"T-0194 criterion 9: #q must hold only the typed text; got {got!r}"
        )
        assert out.resolved.get("replaced") is True, out.resolved
