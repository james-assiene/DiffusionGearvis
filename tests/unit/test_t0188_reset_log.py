"""T-0188 group B, criterion 6: ``_wipe_memory`` logs why a Restart replaced the page.

Expectations come from ``docs/tasks/T-0188.md`` criterion 6 and the group A contract it builds on (criteria 2-3,
rulings A-R2 and A-R4): ``reset_page()`` returns ``ResetOutcome(replaced, reason)``; when the replacement fails too it
raises ``ResetPageError(reason, fallback_reason)``, whose ``reason`` is the first navigation's class.

Observed only through the session log file and the published echo of a live Session over fakes
(``live_rig``) whose browser double scripts ``reset_page``:

- a recovered reset (``ResetOutcome(True, <class>)``) is one ``note`` record
  ``{"what": "live_browser.reset_page", "replaced": true, "reason": <class>}``;
- an unrecovered reset (``ResetPageError``) is one ``error`` record
  ``{"what": "live_browser.reset_page", "error": <type name>, "reason": <class>}``; the old shape (type name only)
  is gone;
- a plain reset (``ResetOutcome(False, None)``, or ``None`` as the integration ``RecordingExecutor`` of
  ``tests/integration/test_t0169_web_start_page_browser.py`` returns) logs no ``live_browser.reset_page`` record, as
  at BASE;
- every row: the reset completes, ``session.reset`` is logged exactly once and ``astra.session.cleared`` published
  once;
- adversarial (task: B15 / T-0111g, an exception's text can carry a URL): the double's exception carries a URL, a
  host and a marker in its message, its args and its ``__cause__``; none of them reaches the log file.

Surface fixed here where the task is silent (listed in the test-author's report): the record's data keys are exactly
the ones criterion 6 lists (no ``fallback_reason``, no message); for a ``reset_page`` exception that is not a
``ResetPageError`` (a driver without ``replace_page`` re-raises unchanged, A-R4) only the type name and the absence of
the message are pinned, plus ``reason`` (if present) being a class or ``None``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), live tools over ``FakeLiveBrowser``; n = 1, deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra.computer.executor import RESET_REASONS, ResetOutcome, ResetPageError
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig
from tests.unit.test_session_loop import answer
from tests.unit.test_session_reset import FIRST_A, answered_turn, do_reset, one_cleared

WHAT = "live_browser.reset_page"
URL = "https://www.photopea.com/private/doc-7731?token=s3cr3t"
HOST = "www.photopea.com"
MARKER = "ERR_ABORTED_MARKER_9f2c"
LEAK = f"Page.goto: net::{MARKER} at {URL} (host {HOST}) timeout"
FORBIDDEN = (URL, HOST, MARKER, "photopea", "s3cr3t", "net::")


class LeakyResetPageError(ResetPageError):
    """A ``ResetPageError`` whose text carries a URL and a host (a careless implementation, or a careless logger)."""

    def __init__(self, reason: str, fallback_reason: str) -> None:
        super().__init__(reason, fallback_reason)
        self.args = (reason, fallback_reason, LEAK)

    def __str__(self) -> str:
        return LEAK

    def __repr__(self) -> str:
        return f"LeakyResetPageError({LEAK!r})"


class ScriptedResetBrowser(FakeLiveBrowser):
    """``FakeLiveBrowser`` whose ``reset_page`` returns or raises what the row scripts."""

    def __init__(self, result: Any = None, *, raise_: BaseException | None = None) -> None:
        super().__init__([live_frame("previous-conversation")])
        self.result = result
        self.raise_ = raise_
        self.page_resets = 0

    async def reset_page(self) -> Any:
        self.calls.append(("reset_page",))
        self.page_resets += 1
        if self.raise_ is not None:
            raise self.raise_
        return self.result


def leaky(exc: BaseException) -> BaseException:
    """Chain a driver-looking exception carrying the URL, as ``raise ... from`` would."""
    exc.__cause__ = RuntimeError(LEAK)
    exc.__context__ = exc.__cause__
    return exc


async def run_reset(tmp_path: Path, browser: ScriptedResetBrowser) -> tuple[Any, str]:
    async with live_rig(tmp_path, browser, script=[answer(FIRST_A)]) as rig:
        await answered_turn(rig)
        await do_reset(rig)
        assert browser.page_resets == 1, f"premise: reset_page awaited once; {browser.page_resets}"
        one_cleared(rig)
        log = rig.session.session_log
        assert log is not None, "premise: the rig logs"
        path = log.path
    return rig, Path(path).read_text(encoding="utf-8")


def reset_page_records(rig: Any) -> list[tuple[str, dict[str, Any]]]:
    return [
        (r["kind"], r["data"])
        for r in rig.log_records
        if isinstance(r.get("data"), dict) and r["data"].get("what") == WHAT
    ]


def assert_reset_completed(rig: Any) -> None:
    resets = rig.kind("session.reset")
    assert len(resets) == 1, f"criterion 6: session.reset logged exactly once; got {resets}"
    assert resets[0]["resets"] == 1, resets


def assert_no_leak(text: str) -> None:
    for needle in FORBIDDEN:
        assert needle not in text, (
            f"criterion 6 / B15: {needle!r} (from the exception's text) reached the session log"
        )


# --- recovered: one note ------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", RESET_REASONS)
async def test_a_recovered_reset_is_one_note_with_replaced_and_reason(tmp_path, reason):
    rig, _ = await run_reset(tmp_path, ScriptedResetBrowser(ResetOutcome(True, reason)))
    assert reset_page_records(rig) == [
        ("note", {"what": WHAT, "replaced": True, "reason": reason})
    ], (
        f"criterion 6: a recovered reset ({reason}) is one note record; got {reset_page_records(rig)}"
    )
    assert_reset_completed(rig)


# --- unrecovered: one error with the type name and the reason ------------------------------------------


@pytest.mark.parametrize(
    ("reason", "fallback"),
    [
        ("beforeunload", "other"),
        ("timeout", "target_closed"),
        ("target_closed", "timeout"),
        ("other", "beforeunload"),
    ],
)
async def test_an_unrecovered_reset_is_one_error_with_type_and_first_reason(
    tmp_path, reason, fallback
):
    browser = ScriptedResetBrowser(raise_=ResetPageError(reason, fallback))
    rig, _ = await run_reset(tmp_path, browser)
    assert reset_page_records(rig) == [
        ("error", {"what": WHAT, "error": "ResetPageError", "reason": reason})
    ], (
        "criterion 6 (A-R4: reason is the first navigation's class, not fallback_reason): "
        f"got {reset_page_records(rig)}"
    )
    assert_reset_completed(rig)


async def test_the_old_type_only_error_shape_is_gone(tmp_path):
    rig, _ = await run_reset(
        tmp_path, ScriptedResetBrowser(raise_=ResetPageError("beforeunload", "other"))
    )
    records = reset_page_records(rig)
    assert ("error", {"what": WHAT, "error": "ResetPageError"}) not in records, (
        f"criterion 6: the type-name-only record is replaced by one carrying the reason; got {records}"
    )
    assert all("reason" in data for _, data in records), records


# --- plain: no record ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result", [ResetOutcome(False, None), None], ids=["outcome-not-replaced", "returns-none"]
)
async def test_a_plain_reset_logs_no_reset_page_record(tmp_path, result):
    """As at BASE; ``None`` is what the integration ``RecordingExecutor`` (T-0169) returns."""
    rig, _ = await run_reset(tmp_path, ScriptedResetBrowser(result))
    assert reset_page_records(rig) == [], (
        f"criterion 6: a plain reset adds no live_browser.reset_page record; got {reset_page_records(rig)}"
    )
    assert_reset_completed(rig)


# --- adversarial: no message, URL or host in the log file -------------------------------------------------


async def test_no_message_url_or_host_from_an_unrecovered_reset_reaches_the_log(tmp_path):
    exc = leaky(LeakyResetPageError("timeout", "other"))
    rig, text = await run_reset(tmp_path, ScriptedResetBrowser(raise_=exc))
    assert reset_page_records(rig) == [
        ("error", {"what": WHAT, "error": "LeakyResetPageError", "reason": "timeout"})
    ], reset_page_records(rig)
    assert_no_leak(text)
    assert_reset_completed(rig)


async def test_no_message_url_or_host_from_a_foreign_exception_reaches_the_log(tmp_path):
    """A driver without ``replace_page`` re-raises the driver error unchanged (A-R4): its text carries the URL."""
    rig, text = await run_reset(tmp_path, ScriptedResetBrowser(raise_=leaky(RuntimeError(LEAK))))
    records = reset_page_records(rig)
    assert len(records) == 1 and records[0][0] == "error", records
    data = records[0][1]
    assert data.get("error") == "RuntimeError", data
    assert set(data) == {"what", "error", "reason"}, (
        f"criterion 6: an unrecovered reset's error record carries a reason key; got {data}"
    )
    assert data["reason"] in (*RESET_REASONS, None), data
    assert_no_leak(text)
    assert_reset_completed(rig)
