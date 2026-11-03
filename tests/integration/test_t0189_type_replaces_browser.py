"""T-0189 end to end: ``type_at`` into a prefilled single-line input replaces its text in a real headless Chromium;
a textarea and a contenteditable keep their text; ``/web.html``'s t12 failure is repaired (docs/tasks/T-0189.md,
criteria 1, 2 and 6).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate, imported).
``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0189_type_replaces_browser.py -q -m browser``. Every
server binds 127.0.0.1 on an ephemeral port; no row touches the network.

**Two rigs.**

- ``t0189_fields.html`` (``tests/fixtures/browser/``, MANIFEST.md) through T-0111c's loopback fixture server and a
  default ``BrowserToolExecutor`` (not web mode: the real ``PlaywrightDriver`` always has ``focused_field``, so the
  guarded path runs and criterion 1 applies there too). Seven prefilled single-line inputs (``text``, ``search``,
  ``url``, ``email``, ``tel``, ``number``, no ``type``), a prefilled password field, a textarea and a contenteditable
  ``div``, each with a known old text. The multi-line editors are 200 px tall with one short line, so a click at their
  centre lands below the text and the caret goes to its end: BASE appends there, and so must the fix.
- ``/web.html`` (the real start page) in web mode through T-0160's rig (``make browser-web``'s): the real
  ``EgressProxy`` with the injected resolver (``shop.example`` -> a public-looking address) and a connector to the
  loopback TLS fixture server; ``ignore_https_errors`` is the one test-only relaxation (T-0160's module docstring).

**Expected values by hand.** After a ``type_at`` of ``new`` into an input holding ``old``: ``new`` (criterion 1;
BASE gives ``old + new``). Into the textarea / contenteditable: ``old + new`` (criterion 2). The password field keeps
``old-secret`` and the call is ``failed`` (criterion 3, as BASE). Criterion 6 replays t12 (``logs/sessions/
20261028-145330-1824860.jsonl``, s4-s14) with the rig's fixture host: ``type_at "photopy"`` + Enter is refused by the
page (no dot, ``web.html`` 47), then ``type_at "https://shop.example"`` + Enter: with the fix the field reads
``https://shop.example`` and the page navigates to ``https://shop.example/``; at BASE it reads
``photopyhttps://shop.example``, whose scheme ``photopyhttps:`` the page refuses (``web.html`` 48-53).

**State.** Rows that need the feature were written ``pending(task="T-0189")`` and activated for ``/implement
T-0189``; the textarea, contenteditable and password rows hold at BASE (BASE already appends / refuses) and are active
from the start: they kill a select-all on the wrong element.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless (versions.lock ``[browser]``), 1280x800 at DSF 1, one
browser per row, n = 1. BASE = 4bee1247.
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
    page_log,
    run,
    server,
)
from tests.integration.test_computer_web_browser import (  # noqa: F401  (rig, launches are fixtures)
    Net,
    Rig,
    center,
    launches,
    rig,
    value_of,
    wait_url,
    web_executor,
)

pytestmark = [pytest.mark.integration, pytest.mark.browser]

PAGE = "t0189_fields.html"
#: input id -> (old value in the fixture, the text typed); typed texts share no prefix with the old ones
INPUTS = {
    "in-text": ("old-text", "new-text-7"),
    "in-search": ("old-search", "new-search-7"),
    "in-url": ("https://old.example/", "https://new.example/x"),
    "in-email": ("old@example.org", "new@example.net"),
    "in-tel": ("0123456789", "987"),
    "in-number": ("12345", "678"),
    "in-notype": ("old-notype", "new-notype-7"),
}
MODIFIERS = frozenset({"Control", "Alt", "Shift", "Meta"})


async def open_fields(ex: Any, origin: str) -> None:
    await run(ex.open(f"{origin}/{PAGE}"))


async def type_into(ex: Any, selector: str, text: str, *, enter: bool = False) -> ToolOutcome:
    x, y = await center(ex, selector)
    f = await run(ex.frame())
    return await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=enter))


async def typed_keydowns(ex: Any, text: str) -> list[dict[str, Any]]:
    """The page's keydowns for the typed characters (the last ``len(text)`` non-modifier keydowns)."""
    downs = [
        e
        for e in await page_log(ex)
        if e.get("type") == "keydown" and e.get("key") not in MODIFIERS
    ]
    return downs[-len(text) :]


# =====================================================================================================
# criterion 1: a prefilled single-line input holds exactly the typed text
# =====================================================================================================


@pytest.mark.parametrize("field", list(INPUTS))
async def test_input_value_is_typed_text(server: str, field: str) -> None:  # noqa: F811  (the fixtures)
    """Criterion 1: after ``type_at`` on a prefilled ``<input>`` of each single-line type its value is exactly the
    typed text (BASE: old + typed); ``resolved["replaced"] is True``; the typed characters arrive without Control
    (no modifier leaked from the select-all)."""
    old, new = INPUTS[field]
    async with executor() as ex:
        await open_fields(ex, server)
        assert await value_of(ex, f"#{field}") == old, "premise: the fixture's old value"
        out = await type_into(ex, f"#{field}", new)
        assert out.status == "executed", out
        await asyncio.sleep(0.2)
        got = await value_of(ex, f"#{field}")
        assert got == new, (
            f"T-0189 criterion 1: #{field} must hold only the typed text {new!r}; got {got!r}"
        )
        assert out.resolved.get("replaced") is True, out.resolved
        keys = await typed_keydowns(ex, new)
        assert [k["key"] for k in keys] == list(new), keys
        assert not any(k.get("ctrlKey") for k in keys), f"Control held while typing: {keys}"


async def test_second_type_replaces_first(server: str) -> None:  # noqa: F811  (the fixtures)
    """Criterion 1, the t12 shape on a plain field: two ``type_at`` calls on the same input leave only the second
    call's text."""
    async with executor() as ex:
        await open_fields(ex, server)
        for text in ("photopy", "https://www.photopea.com"):
            out = await type_into(ex, "#in-text", text)
            assert out.status == "executed", out
        await asyncio.sleep(0.2)
        got = await value_of(ex, "#in-text")
        assert got == "https://www.photopea.com", f"T-0189 criterion 1: got {got!r}"


# =====================================================================================================
# criterion 2: multi-line and rich editors keep their text (adversarial)
# =====================================================================================================


async def test_textarea_keeps_old_text(server: str) -> None:  # noqa: F811  (the fixtures)
    """Criterion 2, adversarial (a select-all would wipe a whole document): the textarea reads its old text plus the
    typed text; no ``replaced``. Holds at BASE."""
    async with executor() as ex:
        await open_fields(ex, server)
        out = await type_into(ex, "#ta", "+ta-17")
        assert out.status == "executed", out
        await asyncio.sleep(0.2)
        got = await value_of(ex, "#ta")
        assert got == "old textarea+ta-17", f"T-0189 criterion 2: the textarea lost text: {got!r}"
        assert "replaced" not in out.resolved, out.resolved


async def test_contenteditable_keeps_old_text(server: str) -> None:  # noqa: F811  (the fixtures)
    """Criterion 2, adversarial: the contenteditable ``div`` reads its old text plus the typed text; no
    ``replaced``. Holds at BASE."""
    async with executor() as ex:
        await open_fields(ex, server)
        out = await type_into(ex, "#ce", "+ce-17")
        assert out.status == "executed", out
        await asyncio.sleep(0.2)
        got = await ex.driver.page.evaluate("document.getElementById('ce').textContent")
        assert got == "old editable+ce-17", f"T-0189 criterion 2: the editor lost text: {got!r}"
        assert "replaced" not in out.resolved, out.resolved


async def test_password_field_untouched(server: str) -> None:  # noqa: F811  (the fixtures)
    """Criterion 3: the prefilled password field is ``failed`` with ``typed: 0`` and keeps its value; no
    ``replaced``. Holds at BASE."""
    async with executor() as ex:
        await open_fields(ex, server)
        out = await type_into(ex, "#in-pw", "hunter2")
        assert out.status == "failed" and out.resolved.get("typed") == 0, out
        assert "replaced" not in out.resolved, out.resolved
        assert await value_of(ex, "#in-pw") == "old-secret"


# =====================================================================================================
# criterion 6: /web.html, the t12 reproduction
# =====================================================================================================


ADDRESS = "#address"


async def test_web_html_retype_navigates(rig: Rig, launches: list[dict[str, Any]]) -> None:  # noqa: F811  (the fixtures)
    """Criterion 6: on the real ``/web.html`` in web mode, ``type_at "photopy"`` + Enter is refused by the page (the
    URL stays, the field keeps ``photopy``); then ``type_at "https://shop.example"`` + Enter navigates to
    ``https://shop.example/`` (BASE: the field reads ``photopyhttps://shop.example`` and the page refuses it)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.open(f"{rig.start}/web.html"), 30)
        start = f"{rig.start}/web.html"
        out = await type_into(ex, ADDRESS, "photopy", enter=True)
        assert out.status == "executed", out
        await asyncio.sleep(1.0)
        assert ex.driver.page.url == start, f"premise: photopy is refused; {ex.driver.page.url}"
        assert await value_of(ex, ADDRESS) == "photopy", (
            "premise: the refused text stays in the field"
        )
        out = await type_into(ex, ADDRESS, "https://shop.example", enter=True)
        assert out.status == "executed", out
        url = await wait_url(ex, "https://shop.example/")
        field = None if url != start else await value_of(ex, ADDRESS)
        assert url == "https://shop.example/", (
            f"T-0189 criterion 6: the second type_at must replace the field and navigate; url {url!r}, "
            f"field {field!r}"
        )
        assert out.resolved.get("replaced") is True, out.resolved
        assert "Shop fixture" in await ex.driver.page.inner_text("body")
