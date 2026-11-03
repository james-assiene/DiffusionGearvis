"""T-0111g B4: the doctor's three ``browser.*`` rows over ``Facts``, and their collector (docs/tasks/T-0111g.md).

**The seam these rows pin** (listed in the test-author's report):

- ``Facts.browser: dict[str, str | None] | None = None``: ``None`` is "not gathered"; gathered, it has the keys
  ``playwright`` (the installed distribution version, ``None`` when not importable), ``chromium`` (the pinned
  executable's path, ``None`` when absent) and ``display`` (the ``DISPLAY`` or ``WAYLAND_DISPLAY`` value, ``None``
  when neither is set). The pinned versions come from ``Facts.lock`` (``collect_lock`` already flattens
  ``[browser]``: ``lock["playwright"] == "1.64.0"``).
- ``preflight.collect_browser() -> dict[str, Any]`` returns ``{"browser": {...}}``, like every ``collect_*``
  (``gather`` merges it); it reads the filesystem and the environment only.
- Rows ``browser.playwright``, ``browser.chromium``, ``browser.display``: PASS or WARN, never FAIL (tools are off by
  default); a WARN carries a fix.

**Q2 / re-scope R7.** ``tests/unit/_box_rig.py`` pins an exhaustive row list: ``by_name`` fails on any row not in
``REQUIRED_CHECKS`` ("unexpected rows ...; add them to _box_rig.REQUIRED_CHECKS first"), and
``test_doctor_verdicts.py::test_doctor_json_*`` asserts ``{names} == set(REQUIRED_CHECKS)``. Adding the three rows
before B4 exists would redden every ``by_name`` row today, so R7 (add the three names to ``REQUIRED_CHECKS`` and
``"browser"`` to ``FACT_FIELDS``) is made at activation; ``test_box_rig_names_the_browser_rows`` below is pending
and holds that edit to account.

Regime: CPU, pure ``evaluate`` over hand-built ``Facts``; the collector reads this box's filesystem/environment.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra import preflight
from tests.unit import _box_rig

ROWS = ("browser.playwright", "browser.chromium", "browser.display")
LOCK = {"playwright": "1.64.0", "chromium": "156.0.8078.4"}
CHROME = "/home/someone/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
GOOD = {"playwright": "1.64.0", "chromium": CHROME, "display": ":1"}


def rows(browser: Any, lock: Any = LOCK) -> dict[str, Any]:
    assert "browser" in getattr(preflight.Facts, "__dataclass_fields__", {}), (
        "T-0111g B4: Facts gains `browser: dict[str, str | None] | None = None` (test_doctor_browser docstring)"
    )
    checks = preflight.evaluate(preflight.Facts(browser=browser, lock=lock))  # type: ignore[call-arg]
    got = {c.name: c for c in checks if c.name.startswith("browser.")}
    assert sorted(got) == sorted(ROWS), (
        f"T-0111g B4: exactly the three browser rows; got {sorted(got)}"
    )
    assert all(c.status != "FAIL" for c in got.values()), "B4: WARN at worst"
    return got


def test_rows_pass_warn() -> None:
    """B4: present gives PASS; missing gives WARN with the fix; a version mismatch gives WARN naming both;
    not gathered gives WARN (never PASS)."""
    ok = rows(GOOD)
    assert {n: c.status for n, c in ok.items()} == dict.fromkeys(ROWS, "PASS"), ok
    assert CHROME in ok["browser.chromium"].detail

    missing = rows({"playwright": None, "chromium": None, "display": None})
    assert {n: c.status for n, c in missing.items()} == dict.fromkeys(ROWS, "WARN"), missing
    # T-0222 re-pin: R-1 (BASE pinned "uv sync --extra browser"; --extra X uninstalls the other extras)
    assert "uv sync --all-extras" in missing["browser.playwright"].fix, missing[
        "browser.playwright"
    ]
    assert "playwright install chromium" in missing["browser.chromium"].fix, missing[
        "browser.chromium"
    ]
    assert "--browser headless" in missing["browser.display"].fix, missing["browser.display"]

    old = rows({**GOOD, "playwright": "1.63.0"})["browser.playwright"]
    assert old.status == "WARN" and "1.63.0" in old.detail and "1.64.0" in old.detail, old
    assert old.fix, "a mismatch says how to fix it"

    ungathered = rows(None)
    assert all(c.status == "WARN" for c in ungathered.values()), ungathered


def test_every_fact_missing_is_still_warn_at_worst() -> None:
    """B4 (adversarial): with every fact missing, the lock unread and a mismatch, no browser row is FAIL."""
    rows({"playwright": "0.1", "chromium": None, "display": None}, lock=None)
    rows(None, lock=None)


def test_collectors_spawn_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B4 (mutation 15): ``collect_browser`` starts no process and no Playwright; it reads the environment
    (``DISPLAY``/``WAYLAND_DISPLAY``) and the filesystem (an empty ``PLAYWRIGHT_BROWSERS_PATH`` is "absent")."""

    def spawned(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"T-0111g B4: a doctor collector started a process: {a[:1]!r}")

    monkeypatch.setattr(subprocess, "Popen", spawned)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawned)
    monkeypatch.setattr(os, "posix_spawn", spawned, raising=False)
    monkeypatch.setattr(os, "fork", spawned, raising=False)
    for mod, name in (
        ("playwright.async_api", "async_playwright"),
        ("playwright.sync_api", "sync_playwright"),
    ):
        try:
            m = __import__(mod, fromlist=[name])
        except ImportError:
            continue
        monkeypatch.setattr(m, name, spawned)
    collect = getattr(preflight, "collect_browser", None)
    assert callable(collect), (
        "T-0111g B4: preflight.collect_browser() exists (test_doctor_browser docstring)"
    )
    empty = tmp_path / "no-browsers"
    empty.mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(empty))
    monkeypatch.setenv("DISPLAY", ":42")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    got = collect()
    assert set(got) == {"browser"}, got
    facts = got["browser"]
    assert set(facts) == {"playwright", "chromium", "display"}, facts
    assert facts["chromium"] is None, (
        f"an empty PLAYWRIGHT_BROWSERS_PATH holds no Chromium; {facts}"
    )
    assert facts["display"] == ":42", facts
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert collect()["browser"]["display"] == "wayland-0"
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert collect()["browser"]["display"] is None


def test_box_rig_names_the_browser_rows() -> None:
    """R7 (Q2): ``_box_rig.REQUIRED_CHECKS`` is exhaustive, so the three rows and the ``browser`` fact are added to
    it at activation; until then this row is red."""
    assert set(ROWS) <= set(_box_rig.REQUIRED_CHECKS), (
        "R7: add the browser rows to _box_rig.REQUIRED_CHECKS"
    )
    assert "browser" in _box_rig.FACT_FIELDS, "R7: add `browser` to _box_rig.FACT_FIELDS"
