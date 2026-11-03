"""T-0201 B9: ``scripts/replay_screen_reminder.py <session.jsonl>... [--seq N ...] [--json]`` on the synthetic log
``tests/fixtures/t0201/requests.jsonl`` (MANIFEST.md there).

**The output these rows fix** (B9 names the columns, not the format; in the test-author's report):

- text: one line per labelled ``tool_step`` request, whitespace-separated, starting with the integer ``seq``, then
  ``step``, ``label key``, ``reminder key``, ``continuation chars``, ``delta chars``; the last token is ``ok`` or
  ``mismatch``. Other lines (headers, totals) never start with an integer.
- ``--json``: one object with ``"rows"``: a list of ``{"seq", "step", "label_key", "reminder_key",
  "continuation_chars", "delta_chars", "ok"}`` (``reminder_key`` ``null`` when the label is unknown, ``ok`` a bool).
- Rows: every ``llm.request`` with ``phase: "tool_step"`` and a ``live_label``, except the report step
  (``report_fact: true``: no reminder there, B4; reading R-R1). ``continuation chars`` is the logged final assistant
  text's length; ``delta chars`` is ``len("\\n\\n" + reminder)``.
- Exit 0 all rows ok; 1 any row not ok (an unknown label text is such a row, reported); 2 a missing or unreadable
  file, or no labelled ``tool_step`` request at all.

Expected values come from the fixture's own JSON and H-T201-1 typed in ``test_t0201_loop_reminder.py``.

Regime: CPU, no GPU, no network; the inputs are only read; n = 1.
"""

from __future__ import annotations

import json
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_t0201_loop_reminder import H_T201_1, SEP

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_screen_reminder.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0201" / "requests.jsonl"
#: seq -> (step, reminder key), from the fixture's MANIFEST
WANT = {
    11: (7, "updated"),
    12: (15, "updated"),
    13: (3, "unchanged"),
    14: (2, "now"),
    15: (4, "updated"),
    16: (5, "updated"),
    17: (2, "shared_screen"),
    18: (3, "shared_camera"),
}


def logged() -> dict[int, dict[str, Any]]:
    return {
        r["seq"]: r
        for r in map(json.loads, FIXTURE.read_text().splitlines())
        if r["kind"] == "llm.request"
    }


def continuation(rec: dict[str, Any]) -> str:
    content = rec["data"]["prompt"]["messages"][-1]["content"]
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.exists(), "T-0201 B9: scripts/replay_screen_reminder.py does not exist yet"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, cwd=REPO
    )


def text_rows(out: str) -> list[list[str]]:
    return [ln.split() for ln in out.splitlines() if ln.split() and ln.split()[0].isdigit()]


def test_rows_and_exit_zero() -> None:
    """B9: one row per labelled ``tool_step`` (not the step-1 ``generate``, the unlabelled step or the report step),
    each ``ok``, with the step, the reminder key by the label, the logged continuation's length and the reminder's
    delta; exit 0."""
    proc = run(str(FIXTURE))
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = text_rows(proc.stdout)
    recs = logged()
    assert [int(r[0]) for r in rows] == sorted(WANT), proc.stdout
    for r in rows:
        seq = int(r[0])
        step, key = WANT[seq]
        assert int(r[1]) == step and r[3] == key, r
        assert int(r[4]) == len(continuation(recs[seq])), r
        assert int(r[5]) == len(SEP + H_T201_1[key]), r
        assert r[-1] == "ok", r


def test_json_and_seq_filter() -> None:
    """B9 ``--json`` and ``--seq``: the object's rows carry the same values; ``--seq 12 17`` gives those two rows."""
    proc = run(str(FIXTURE), "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    rows = data["rows"]
    assert [r["seq"] for r in rows] == sorted(WANT)
    recs = logged()
    for r in rows:
        step, key = WANT[r["seq"]]
        assert (r["step"], r["reminder_key"], r["ok"]) == (step, key, True), r
        assert r["continuation_chars"] == len(continuation(recs[r["seq"]]))
        assert r["delta_chars"] == len(SEP + H_T201_1[key])
        assert "label_key" in r
    picked = run(str(FIXTURE), "--seq", "12", "17", "--json")
    assert picked.returncode == 0, picked.stderr
    assert [r["seq"] for r in json.loads(picked.stdout)["rows"]] == [12, 17]


def test_exit_two_on_nothing_or_unreadable(tmp_path: Path) -> None:
    """B9: no labelled ``tool_step`` request -> 2; a missing file -> 2; a line that is not JSON -> 2."""
    lines = FIXTURE.read_text().splitlines()
    none = tmp_path / "none.jsonl"
    none.write_text(
        "\n".join(ln for ln in lines if '"live_label"' not in ln or '"generate"' in ln) + "\n"
    )
    assert run(str(none)).returncode == 2
    assert run(str(tmp_path / "missing.jsonl")).returncode == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(lines[0] + "\nnot json\n")
    assert run(str(bad)).returncode == 2


def test_unknown_label_is_reported_exit_one(tmp_path: Path) -> None:
    """Adversarial: a labelled ``tool_step`` whose ``live_label.text`` is no known label is reported as a row (its
    seq present, not ok, ``reminder_key`` null in ``--json``) and the run exits 1; the other rows stay ok."""
    out = []
    for ln in FIXTURE.read_text().splitlines():
        r = json.loads(ln)
        if r.get("seq") == 13:
            r["data"]["live_label"]["text"] = "[This is some other label.]"
        out.append(json.dumps(r))
    path = tmp_path / "unknown.jsonl"
    path.write_text("\n".join(out) + "\n")
    proc = run(str(path), "--json")
    assert proc.returncode == 1, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = {r["seq"]: r for r in json.loads(proc.stdout)["rows"]}
    assert 13 in rows and rows[13]["ok"] is False and rows[13]["reminder_key"] is None, rows.get(13)
    assert all(r["ok"] for s, r in rows.items() if s != 13)


def test_a_wrong_helper_is_a_mismatch_and_no_socket(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Adversarial (the check is real, not a print): run in-process with ``loop.with_screen_reminder`` replaced by one
    that also changes the system message, every row is a mismatch and the exit is 1. No socket is opened (connect and
    create_connection raise)."""
    from local_astra.contracts.llm import Message, TextPart
    from local_astra.toolcall import loop

    real = getattr(loop, "with_screen_reminder", None)
    assert callable(real), "T-0201 B2: loop.with_screen_reminder does not exist yet"
    assert SCRIPT.exists(), "T-0201 B9: scripts/replay_screen_reminder.py does not exist yet"

    def wrong(prompt: Any, reminder: str) -> Any:
        got = real(prompt, reminder)
        system = Message(role="system", content=(TextPart(text="changed"),))
        return got.model_copy(update={"messages": (system, *got.messages[1:])})

    def no_socket(*a: Any, **k: Any) -> Any:
        raise AssertionError("T-0201 B9: the replay opened a socket")

    monkeypatch.setattr(socket.socket, "connect", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    for honest in (True, False):
        monkeypatch.setattr(loop, "with_screen_reminder", real if honest else wrong)
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE), "--json"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        rows = json.loads(capsys.readouterr().out)["rows"]
        if honest:
            assert exc.value.code in (0, None) and all(r["ok"] for r in rows), rows
        else:
            assert exc.value.code == 1 and rows and not any(r["ok"] for r in rows), rows
