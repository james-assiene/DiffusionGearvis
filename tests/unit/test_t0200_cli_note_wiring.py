"""T-0200 criterion 9 (cli side): a live session's browser executor has its ``on_note`` wired to the session's
*current* log (``_CurrentSessionLog(session).record("note", data, session_ms=clock.now_ms)``), so a HUD log toggle
that replaces ``Session.session_log`` is followed; nothing is wired outside live (docs/tasks/T-0200.md).

The rig is ``test_cli_tools_live.py``'s: the real ``build_session`` over the fake clients, ``SpyExecutor`` (a
``FakeLiveBrowser``) in place of ``BrowserToolExecutor`` wherever ``cli`` reaches it, ``_FixedClock`` (``now_ms`` =
424242), nothing served. The row calls the wired ``on_note`` itself, with a note of the executor's shape, and reads
what a recording log received.

Readings: "wires it (live only)" is read as: the executor ``cli`` builds for ``--tools live`` carries a callable
``on_note`` once ``build_session`` returns; ``--tools dry_run`` builds no browser executor, so there is nothing to
wire (the row checks that premise too). The record's ``turn_id`` is not pinned (the task gives none).

Regime: CPU over fakes; no browser, no socket.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra import cli
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import Events, _FixedClock, _out, argv, fake_live

NOTE = {
    "what": "live frame: blank capture retaken",
    "path": "action",
    "captures": 3,
    "blank": False,
    "waited_ms": 212,
}


class RecLog:
    """A session log that records every ``record`` call."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Any, dict[str, Any]]] = []

    def record(self, kind: str, data: Any, **kw: Any) -> None:
        self.records.append((kind, data, kw))


def _build(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, *flags: str
) -> tuple[list[Any], Any]:
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(cli, "SessionClock", _FixedClock)
    made = fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *flags))
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    return made, calls


def test_live_on_note_records_into_current_log(tmp_path, capsys, monkeypatch) -> None:
    """C9: the live executor's ``on_note`` records one ``note`` with the data unchanged and ``session_ms`` from the
    session clock into ``session.session_log``; after the log is replaced (the HUD toggle) the next note goes to the
    new log only; with no log nothing raises. ``--tools dry_run`` builds no browser executor."""
    made, calls = _build(tmp_path, monkeypatch, capsys, "--tools", "live", "--browser", "headless")
    assert len(made) == 1 and calls.graphs, "premise: one live executor, one graph"
    session = calls.graphs[-1].session
    hook = getattr(made[0], "on_note", None)
    assert callable(hook), "T-0200 C9: cli wires the live browser's on_note"

    first = RecLog()
    session.session_log = first
    hook(dict(NOTE))
    assert len(first.records) == 1, first.records
    kind, data, kw = first.records[0]
    assert kind == "note" and data == NOTE, first.records
    assert kw.get("session_ms") == 424_242, f"T-0200 C9: session_ms from the session clock; {kw}"
    assert "http" not in repr(first.records)

    second = RecLog()
    session.session_log = second
    hook(dict(NOTE, path="frame"))
    assert len(first.records) == 1 and len(second.records) == 1, (first.records, second.records)
    assert second.records[0][0] == "note" and second.records[0][1]["path"] == "frame"

    session.session_log = None
    hook(dict(NOTE))  # no log: nothing recorded, nothing raised

    made_dry, _ = _build(tmp_path, monkeypatch, capsys, "--tools", "dry-run")
    assert made_dry == [], (
        "premise: dry_run builds no browser executor (nothing to wire outside live)"
    )
