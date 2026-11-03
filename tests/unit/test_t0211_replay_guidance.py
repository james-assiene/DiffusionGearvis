"""T-0211 B7: ``scripts/replay_live_guidance.py <session.jsonl>... [--to v4] [--json]`` on the real v3 live session
``tests/fixtures/t0211/v3_live_session.jsonl`` (MANIFEST.md there) and edited copies of it.

**The output these rows fix** (B7 names the checks and the exit codes, not the format; readings, stated here and in
the test-author's report):

- Rows: one per ``llm.request`` record of **every** phase (``prefill``, ``generate``, ``tool_step``,
  ``completion_check``, ...), in log order, file after file. The fixture gives seq 7, 14, 167, 201.
- Text: a row is a line whose first whitespace-separated token is the integer ``seq`` and whose last token is ``ok``
  or ``mismatch``; every other line (headers, ``# <file>`` lines, totals) never starts with an integer.
- ``--json``: one object ``{"rows": [...]}``, each row an object with at least ``"seq"`` (int), ``"phase"`` (str) and
  ``"ok"`` (bool), in the same order as the text rows.
- ``--to`` defaults to ``v4`` (the usage line writes ``[--to v4]``); ``--to v4`` and no ``--to`` give the same rows.
- A row is ``ok`` iff (B7): the system text contains the logged version (v3, with or without the wait sentence)
  exactly once; TO's system text is BASE's with ``" " + S_SHARED + " " + S_SOURCE`` inserted at the end of the
  guidance span (before the wait sentence); every other byte of the two ``VllmClient.request_body`` bodies is equal;
  every request in one file has the same system text.
- Exit 0 all rows ok; 1 any row not ok; 2 a missing or unreadable file, a line that is not a JSON object, or no
  ``llm.request`` at all.
- ``test_differing_system_texts_mismatch``: B7 does not say which row(s) of a file whose system texts differ are
  marked; the row asserts only exit 1 and at least one ``mismatch`` row.

Expected values come from the fixture's own JSON and the texts typed from the tasks (S_SHARED/S_SOURCE: T-0211 B1;
the wait sentence: T-0202 H-T202-4), never from the script. Every row was ``pending(task="T-0211")`` until
activation (``already_v4`` last, after B-R7): at BASE the script does not exist and each row fails first on the named ``AssertionError`` "... does not exist yet".

Regime: CPU, no GPU, no network (the in-process rows refuse sockets); the inputs are only read; n = 1.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from tests.unit import _t0202_pins as t0202

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_live_guidance.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0211" / "v3_live_session.jsonl"
#: the fixture's sha256, from its MANIFEST
FIXTURE_SHA = "a0119946f31010e5ea46db1196dd67744f37f12b93dc8ca2924cef5bd4765aee"
#: seq -> phase of the fixture's four requests, from its MANIFEST
WANT = {7: "prefill", 14: "generate", 167: "tool_step", 201: "completion_check"}
#: B1, typed from the task (never imported)
S_SHARED = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: "
    "to do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)
S_SOURCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)
WAIT = t0202.SENTENCE
V3 = pb.LIVE_TOOL_GUIDANCE_V3


def records(path: Path = FIXTURE) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path: Path, recs: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def system_text(rec: dict[str, Any]) -> str:
    part = rec["data"]["prompt"]["messages"][0]["content"][0]
    assert part["type"] == "text"
    text: str = part["text"]
    return text


def set_system_text(rec: dict[str, Any], text: str) -> None:
    rec["data"]["prompt"]["messages"][0]["content"][0]["text"] = text


def run(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.exists(), "T-0211 B7: scripts/replay_live_guidance.py does not exist yet"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, cwd=REPO
    )


def text_rows(out: str) -> list[list[str]]:
    return [ln.split() for ln in out.splitlines() if ln.split() and ln.split()[0].isdigit()]


def json_rows(proc: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
    data = json.loads(proc.stdout)
    assert isinstance(data, dict) and isinstance(data.get("rows"), list), proc.stdout[:400]
    rows: list[dict[str, Any]] = data["rows"]
    return rows


def test_fixture_premise() -> None:
    """Premise (the fixture is what its MANIFEST says): four requests of four phases, one system text containing v3
    once and ending with ``"\\n" + v3 + " " + WAIT``. Fails first on the script's absence, like every row here."""
    assert SCRIPT.exists(), "T-0211 B7: scripts/replay_live_guidance.py does not exist yet"
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    reqs = [r for r in records() if r["kind"] == "llm.request"]
    assert {r["seq"]: r["data"]["phase"] for r in reqs} == WANT
    texts = {system_text(r) for r in reqs}
    assert len(texts) == 1
    text = next(iter(texts))
    assert text.count(V3) == 1 and text.endswith("\n" + V3 + " " + WAIT)


@pytest.mark.parametrize("to", [[], ["--to", "v4"]], ids=["default", "to-v4"])
def test_fixture_rows_ok_exit_zero(to: list[str]) -> None:
    """B7: one text row per ``llm.request`` of every phase (seq 7, 14, 167, 201), each ending ``ok``; exit 0. The
    input is only read (its bytes are unchanged)."""
    proc = run(str(FIXTURE), *to)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = text_rows(proc.stdout)
    assert [int(r[0]) for r in rows] == list(WANT), proc.stdout
    assert all(r[-1] == "ok" for r in rows), proc.stdout
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA, (
        "the replay wrote its input"
    )


def test_json_shape() -> None:
    """B7 ``--json``: one object ``{"rows": [...]}``; each row has ``seq`` (int), ``phase`` (str) and ``ok`` (bool),
    the fixture's four in log order, all ok; two files give their rows file after file (eight)."""
    proc = run(str(FIXTURE), "--json")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = json_rows(proc)
    assert [(r["seq"], r["phase"], r["ok"]) for r in rows] == [
        (s, p, True) for s, p in WANT.items()
    ], rows
    for r in rows:
        assert type(r["seq"]) is int and isinstance(r["phase"], str) and type(r["ok"]) is bool, r
    two = run(str(FIXTURE), str(FIXTURE), "--json")
    assert two.returncode == 0, two.stderr
    assert [r["seq"] for r in json_rows(two)] == [*WANT, *WANT]


def test_without_wait_sentence_ok(tmp_path: Path) -> None:
    """B7: the logged guidance may be v3 without the wait sentence (a ``--wait-tool off`` session): the fixture with
    ``" " + WAIT`` removed from every system text still gives four ok rows and exit 0."""
    recs = records()
    for r in recs:
        if r["kind"] == "llm.request":
            text = system_text(r)
            assert text.endswith(" " + WAIT), "premise"
            set_system_text(r, text[: -len(" " + WAIT)])
    proc = run(str(write(tmp_path / "nowait.jsonl", recs)), "--json")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert [(r["seq"], r["ok"]) for r in json_rows(proc)] == [(s, True) for s in WANT]


def test_extra_byte_outside_system_text_is_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adversarial (the body comparison is real, not a print): run in-process with ``VllmClient.request_body``
    wrapped so that the TO body (its system text carries S_SHARED) gains one byte outside the system text (``model``
    + ``"x"``): every row is a mismatch and the exit is 1. The honest wrapper gives every row ok and exit 0. No socket
    is opened (connect and create_connection raise)."""
    assert SCRIPT.exists(), "T-0211 B7: scripts/replay_live_guidance.py does not exist yet"
    from local_astra.conductor.llm_client import VllmClient

    real = VllmClient.request_body
    calls = {"to": 0}

    def wrong(self: Any, prompt: Any, *a: Any, **k: Any) -> dict[str, Any]:
        body: dict[str, Any] = real(self, prompt, *a, **k)
        if S_SHARED in prompt.messages[0].text():
            calls["to"] += 1
            body["model"] = str(body["model"]) + "x"
        return body

    def no_socket(*a: Any, **k: Any) -> Any:
        raise AssertionError("T-0211 B7: the replay opened a socket")

    monkeypatch.setattr(socket.socket, "connect", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    for honest in (True, False):
        monkeypatch.setattr(VllmClient, "request_body", real if honest else wrong)
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE), "--json"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        rows = json.loads(capsys.readouterr().out)["rows"]
        assert [r["seq"] for r in rows] == list(WANT), rows
        if honest:
            assert exc.value.code in (0, None) and all(r["ok"] for r in rows), rows
        else:
            assert calls["to"] >= len(WANT), "premise: TO went through request_body for every row"
            assert exc.value.code == 1 and not any(r["ok"] for r in rows), rows


def test_to_body_is_base_plus_sentences_before_wait(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adversarial (reviewer mutation 3 carried to the replay: the sentences placed after the wait sentence, with the
    script's own check moved to match, would still print ok): run in-process with ``VllmClient.request_body`` recording
    the system text of every prompt it is given. The BASE prompts carry the logged system text unchanged; the TO prompts
    (those carrying S_SHARED) carry exactly the logged text with ``" " + S_SHARED + " " + S_SOURCE`` inserted right
    after v3's text, so before ``" " + WAIT``; both kinds are seen for the fixture's four requests."""
    assert SCRIPT.exists(), "T-0211 B7: scripts/replay_live_guidance.py does not exist yet"
    from local_astra.conductor.llm_client import VllmClient

    real = VllmClient.request_body
    seen: list[str] = []

    def recording(self: Any, prompt: Any, *a: Any, **k: Any) -> dict[str, Any]:
        seen.append(prompt.messages[0].text())
        body: dict[str, Any] = real(self, prompt, *a, **k)
        return body

    logged = system_text(next(r for r in records() if r["kind"] == "llm.request"))
    i = logged.index(V3) + len(V3)
    assert logged[i:] == " " + WAIT, "premise: the wait sentence follows v3 in the fixture"
    want_to = logged[:i] + " " + S_SHARED + " " + S_SOURCE + logged[i:]
    monkeypatch.setattr(VllmClient, "request_body", recording)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE), "--json"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert exc.value.code in (0, None) and all(r["ok"] for r in rows), rows
    to = [t for t in seen if S_SHARED in t]
    base = [t for t in seen if S_SHARED not in t]
    assert len(to) >= len(WANT) and len(base) >= len(WANT), (
        f"premise: BASE and TO bodies built per request; {len(base)} BASE, {len(to)} TO"
    )
    assert set(base) == {logged}, "T-0211 B7: BASE is the request as logged"
    assert set(to) == {want_to}, (
        "T-0211 B7: TO's system text is BASE's with ' ' + S_SHARED + ' ' + S_SOURCE at the end of the guidance span"
    )


@pytest.mark.parametrize(
    "how",
    [
        "absent",
        "twice",
        # B-R8 (row 1): needs B-R7 (a log already carrying v4 is a mismatch; fixed in 1d90e374); at 73483aba it
        # reported ok with a doubled delta
        "already_v4",
    ],
)
def test_v3_not_exactly_once_is_mismatch(tmp_path: Path, how: str) -> None:
    """B7: a system text with no v3 guidance (v3 replaced by v2's text in every request) or with v3 twice gives
    mismatch rows and exit 1 (not 2: the file is readable and holds requests).

    B-R7/B-R8 ``already_v4``: every request's v3 rewritten to v4 (v3 + ``" " + S_SHARED + " " + S_SOURCE``, typed
    from B1), the wait sentence kept after it: the log already carries the v4 delta, so every row is a mismatch and the
    exit is 1 (not an ok row whose TO text carries the delta twice)."""
    recs = records()
    for r in recs:
        if r["kind"] == "llm.request":
            text = system_text(r)
            assert text.count(V3) == 1, "premise"
            if how == "absent":
                new = text.replace(V3, pb.LIVE_TOOL_GUIDANCE_V2)
                assert V3 not in new, "premise: v2 is not v3"
            elif how == "already_v4":
                new = text.replace(V3, V3 + " " + S_SHARED + " " + S_SOURCE)
                assert new.endswith(S_SOURCE + " " + WAIT) and new.count(S_SHARED) == 1, (
                    "premise: v4 then the wait sentence"
                )
            else:
                new = V3 + "\n" + text
            set_system_text(r, new)
    proc = run(str(write(tmp_path / f"{how}.jsonl", recs)), "--json")
    assert proc.returncode == 1, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = json_rows(proc)
    assert [r["seq"] for r in rows] == list(WANT) and not any(r["ok"] for r in rows), rows


def test_differing_system_texts_mismatch(tmp_path: Path) -> None:
    """B7: two requests of one file with different system texts (the last request's text gains one leading
    character; v3 still once in each) give at least one mismatch row and exit 1. Power: the same file split in two,
    one request per file, is all ok (the per-file rule, not the edit, makes the mismatch)."""
    recs = records()
    last = max(i for i, r in enumerate(recs) if r["kind"] == "llm.request")
    set_system_text(recs[last], "X" + system_text(recs[last]))
    assert system_text(recs[last]).count(V3) == 1, "premise"
    proc = run(str(write(tmp_path / "differ.jsonl", recs)), "--json")
    assert proc.returncode == 1, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    rows = json_rows(proc)
    assert [r["seq"] for r in rows] == list(WANT) and not all(r["ok"] for r in rows), rows
    head = [r for r in recs if r["kind"] != "llm.request" or r is not recs[last]]
    tail = [recs[0], recs[last]]
    split = run(
        str(write(tmp_path / "a.jsonl", head)), str(write(tmp_path / "b.jsonl", tail)), "--json"
    )
    assert split.returncode == 0, f"exit {split.returncode}\n{split.stdout}\n{split.stderr}"
    assert all(r["ok"] for r in json_rows(split))


def test_exit_two_on_unreadable_or_empty(tmp_path: Path) -> None:
    """B7: exit 2 for a missing file, a line that is not a JSON object (an array, a number, a string; and a line that
    is not JSON), and a file with no ``llm.request`` (only ``log.started`` and ``log.stopped``); also when a bad file
    comes after a good one."""
    assert run(str(tmp_path / "missing.jsonl")).returncode == 2
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    for name, junk in (
        ("array", "[1, 2]"),
        ("number", "42"),
        ("string", '"llm.request"'),
        ("text", "not json"),
    ):
        bad = tmp_path / f"{name}.jsonl"
        bad.write_text("\n".join([lines[0], junk, *lines[1:]]) + "\n", encoding="utf-8")
        proc = run(str(bad))
        assert proc.returncode == 2, f"{name}: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    empty = write(tmp_path / "empty.jsonl", [r for r in records() if r["kind"] != "llm.request"])
    assert run(str(empty)).returncode == 2
    assert run(str(FIXTURE), str(tmp_path / "missing.jsonl")).returncode == 2


@pytest.mark.parametrize("wrong", ["delta_plus_byte", "delta_after_wait", "no_delta"])
def test_wrong_v4_is_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], wrong: str
) -> None:
    """B-R8 (row 2), adversarial (reviewer 2's surviving mutant: the "TO = BASE + delta" check made unreachable, so the
    only comparison left masks the system message): run in-process with ``prompt_builder.LIVE_GUIDANCE["v4"]``
    replaced by a wrong v4 (the mapping swapped for a dict). ``delta_plus_byte``: v3 + ``" " + S_SHARED + " " +
    S_SOURCE`` + ``"x"``; ``delta_after_wait``: v3 + ``" " + WAIT`` + the delta, so in the TO text the sentences come
    after the wait sentence; ``no_delta``: v3 alone. Each gives every row a mismatch and exit 1. Power: the same patch
    with the right v4 (typed from B1) gives every row ok and exit 0. No socket is opened."""
    assert SCRIPT.exists(), "T-0211 B7: scripts/replay_live_guidance.py does not exist yet"
    delta = " " + S_SHARED + " " + S_SOURCE
    texts = {
        "right": V3 + delta,
        "delta_plus_byte": V3 + delta + "x",
        "delta_after_wait": V3 + " " + WAIT + delta,
        "no_delta": V3,
    }

    def no_socket(*a: Any, **k: Any) -> Any:
        raise AssertionError("T-0211 B7: the replay opened a socket")

    monkeypatch.setattr(socket.socket, "connect", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    for name in ("right", wrong):
        monkeypatch.setattr(pb, "LIVE_GUIDANCE", {**pb.LIVE_GUIDANCE, "v4": texts[name]})
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE), "--json"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        rows = json.loads(capsys.readouterr().out)["rows"]
        assert [r["seq"] for r in rows] == list(WANT), rows
        if name == "right":
            assert exc.value.code in (0, None) and all(r["ok"] for r in rows), (
                f"premise: the right v4 through the same patch is ok; {rows}"
            )
        else:
            assert exc.value.code == 1 and not any(r["ok"] for r in rows), (
                f"T-0211 B7: a wrong v4 ({wrong}) must give every row a mismatch and exit 1; {rows}"
            )
