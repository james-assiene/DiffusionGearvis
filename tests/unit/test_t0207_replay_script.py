"""T-0207 A1.5: ``scripts/replay_completion_check.py <session.jsonl>... [--json]`` on the synthetic log
``tests/fixtures/t0207/session_claims.jsonl`` (MANIFEST.md there names every turn and what it is for).

**The output these rows fix** (A1.5 names the columns, not the format; in the test-author's report):

- ``--json``: one object with ``"rows"``: a list of objects whose keys are exactly A1.5's columns ``file, turn, seq,
  step, report_step, reasked_this_step, vision_source, would_check, last_action_unchanged, label_key,
  thought_present, claim_match, answer``. ``file`` names the input (compared by its file name); ``label_key`` is
  the T-0201 B3 key of the request's ``live_label`` (``updated``, ``unchanged``, ``now``, ``shared_screen``, ...).
- text: one line per row holding the row's ``turn`` and ``seq`` as whitespace-separated tokens, and a totals line
  (the word "total", any case). Rows are in log order within a file, files in argument order.
- ``seq`` is the seq of the turn's final ``llm.request`` (for a T-0172 re-asked step, the ``announce_reask``
  request: that is the turn's final request).
- ``answer`` is the final request's answer channel cut to 120 characters (an ellipsis after the cut is accepted).
- Exit 0 on success; 1 if a body check fails; 2 if any input is missing or unreadable (a line that is not JSON) or
  holds no row.

Expected values come from the fixture's MANIFEST (built by ``build_claims.py`` from the turns' intent, never from
the script) and from the task file. ``claim_match`` uses the script's pinned regex; only unambiguous rows are pinned:
"I have added ... to your cart." / "I have purchased ..." / "... proceeded to the checkout page" match; "I cannot see
the screen right now." and "I am looking at the search results ..." do not.

Adversarial: the declined-click + ``google_search``-last turn (t7) and the search-only turn (t4) are not rows;
``last_action_unchanged`` counts executed browser actions only (t2's refused ``type_at`` sits between two clicks with
one ``frame_after``) and compares with the one just before (t6's last equals its first, not its previous); a
``check_prompt`` that changes the system message makes every body check fail (exit 1); no socket is opened.

On James's log (``logs/sessions/20261029-201813-2716251.jsonl`` under the main checkout, read only; the row skips
where it is absent): the 12 rows A1.5 expects, at the seqs of an independent scan by the test-author (final request
of every live turn ending in a non-empty answer with no call after an executed browser action), with seq 2493 and
5538 ``would_check``/``last_action_unchanged`` true and ``report_step`` false, and seq 16197 ``report_step`` true.

Regime: CPU, no GPU, no network; the inputs are only read; n = 1.
"""

from __future__ import annotations

import json
import runpy
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_completion_check.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0207" / "session_claims.jsonl"
COMPLETION = "local_astra.toolcall.completion_check"

COLUMNS = {
    "file",
    "turn",
    "seq",
    "step",
    "report_step",
    "reasked_this_step",
    "vision_source",
    "would_check",
    "last_action_unchanged",
    "label_key",
    "thought_present",
    "claim_match",
    "answer",
}

T3_ANSWER = (
    "I have added the synthetic comic to your basket and proceeded to the checkout page. I stopped there because "
    "paying needs your own payment and delivery details."
)
#: from MANIFEST.md: turn -> the pinned fields of its row (``None`` for a field not pinned)
WANT: dict[str, dict[str, Any]] = {
    "t1": {
        "seq": 2493,
        "step": 4,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": True,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": True,
        "answer": "I have added volume 7 of Vinland Saga to your cart.",
    },
    "t2": {
        "seq": 5538,
        "step": 5,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": True,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": True,
        "answer": "I have purchased The Legend of Zelda: Tears of the Kingdom for you.",
    },
    "t3": {
        "seq": 16197,
        "step": 4,
        "report_step": True,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": True,
        "label_key": "unchanged",
        "thought_present": True,
        "claim_match": True,
        "answer": T3_ANSWER,
    },
    "t6": {
        "seq": 16305,
        "step": 4,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "shared",
        "would_check": False,
        "last_action_unchanged": False,
        "label_key": "shared_screen",
        "thought_present": False,
        "claim_match": None,
        "answer": "I pressed Tab and the form shows the next field.",
    },
    "t8": {
        "seq": 16367,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": True,
        "claim_match": False,
        "answer": "I cannot see the screen right now.",
    },
    "t9": {
        "seq": 16407,
        "step": 3,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": True,
        "claim_match": False,
        "answer": "I am looking at the search results for volume 8.",
    },
    "t10": {
        "seq": 16443,
        "step": 2,
        "report_step": False,
        "reasked_this_step": True,
        "vision_source": "browser",
        "would_check": False,
        "last_action_unchanged": None,
        "label_key": None,
        "thought_present": None,
        "claim_match": None,
        # the re-ask's own answer: its commits are joined by attempt_id, never by request_id (which would put the
        # step's announcement "I will click the checkout button." in front)
        "answer": "There is no checkout button on this screen.",
    },
    "t11": {
        "seq": 16467,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": False,  # the answer tripped T-0142b: no llm.done for the final request
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": None,
        "answer": "I have added the comic to the basket the basket the basket",
    },
    "t12": {
        "seq": 16489,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": False,  # aborted mid-stream: no llm.done for the final request, no note
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": True,
        "claim_match": None,
        "answer": "I have opened the help page.",
    },
    "t13": {
        "seq": 16533,
        "step": 3,
        "report_step": False,
        "reasked_this_step": False,  # step 2 was re-asked, not step 3
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": False,
        "answer": "The account page shows your name and address.",
    },
    "t14": {
        "seq": 16555,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,  # wait is a browser action (A1.5's list)
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": None,
        "answer": "The page has loaded and shows the comic list.",
    },
    "t15": {
        # T-0207 A2-R3: a checked step; the step's tool_step request (16579) is the final request, never the
        # completion_check request after it (16585)
        "seq": 16579,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "browser",
        "would_check": True,
        "last_action_unchanged": False,
        "label_key": "updated",
        "thought_present": False,
        "claim_match": None,
        "answer": "I have added the comic to your basket.",
    },
}
#: T-0208 A2.6 (additive): the shared-source turns t16-t19 (MANIFEST.md). ``would_check`` is left unpinned here
#: (``None``) so the T-0207 rows above keep their meaning; ``test_t0208_shared_would_check_follows_the_label`` pins
#: it per A2.6.
T0208_SHARED: dict[str, tuple[int, str, str, bool]] = {
    # turn: (final seq, label_key, answer, would_check per T-0208 A2.6)
    "t16": (16611, "updated", "The replacement steam wand is in your cart now.", True),
    "t17": (16635, "unchanged", "The page did not move; the stove list is still on screen.", True),
    "t18": (16659, "now", "This page shows the pizza order form.", True),
    "t19": (16683, "shared_camera", "The camera shows the coffee machine.", False),
}
for _turn, (_seq, _key, _answer, _would) in T0208_SHARED.items():
    WANT[_turn] = {
        "seq": _seq,
        "step": 2,
        "report_step": False,
        "reasked_this_step": False,
        "vision_source": "shared",
        "would_check": None,
        "last_action_unchanged": False,
        "label_key": _key,
        "thought_present": False,
        "claim_match": None,
        "answer": _answer,
    }
#: T-0207 A2-R3: the seq of t15's completion_check request (never a row's seq)
T15_CHECK_SEQ = 16585
NOT_ROWS = {"t0", "t4", "t5", "t7"}

#: James's log (A1.5 "Expected on James's log"); the seqs are the test-author's independent scan, 2026-10-30
JAMES_LOG = "logs/sessions/20261029-201813-2716251.jsonl"
JAMES_SEQS = [2493, 3820, 5538, 6126, 7408, 9099, 10542, 11399, 12042, 13434, 16197, 17849]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.exists(), "T-0207 A1.5: scripts/replay_completion_check.py does not exist yet"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, cwd=REPO
    )


def json_rows(*args: str) -> list[dict[str, Any]]:
    proc = run(*args, "--json")
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    data = json.loads(proc.stdout)
    rows = data["rows"]
    assert isinstance(rows, list)
    return rows


def cut_ok(got: str, full: str) -> bool:
    """A1.5 "the answer cut to 120 characters": the whole answer when it fits, else its first 120 characters
    (an ellipsis after them, or in their last place, is accepted)."""
    if len(full) <= 120:
        return got == full
    return got in (full[:120], full[:120] + "…", full[:119] + "…", full[:117] + "...")


# --------------------------------------------------------------------------------------------- rows


def test_fixture_premises() -> None:
    """The fixture holds the turns MANIFEST.md names (guards against a rebuilt fixture drifting)."""
    recs = [json.loads(ln) for ln in FIXTURE.read_text().splitlines()]
    turns = {r["turn_id"] for r in recs if r.get("turn_id")}
    assert turns == set(WANT) | NOT_ROWS, sorted(turns)
    final: dict[str, dict[str, Any]] = {}
    for r in recs:
        # T-0207 A2-R3: a completion_check request is never the turn's final request
        if r["kind"] == "llm.request" and r["data"].get("phase") != "completion_check":
            final[r["turn_id"]] = r
    for turn, want in WANT.items():
        assert final[turn]["seq"] == want["seq"], (turn, final[turn]["seq"])
    assert final["t3"]["data"].get("report_fact") is True
    assert final["t10"]["data"]["phase"] == "announce_reask"
    executed = [
        r["data"]["name"]
        for r in recs
        if r["kind"] == "tool.call" and r["turn_id"] == "t7" and r["data"]["status"] == "executed"
    ]
    assert executed == ["google_search"], executed
    assert len(T3_ANSWER) > 120
    for turn in ("t11", "t12"):
        rid = final[turn]["data"]["request_id"]
        dones = [r for r in recs if r["kind"] == "llm.done" and r["data"]["request_id"] == rid]
        assert dones == [], (turn, dones)
    notes = [r["data"]["what"] for r in recs if r["kind"] == "note" and r["turn_id"] == "t11"]
    assert notes == ["degenerate answer: request aborted"], notes
    assert not [r for r in recs if r["kind"] == "note" and r["turn_id"] == "t12"]
    t13 = [r["data"] for r in recs if r["kind"] == "llm.request" and r["turn_id"] == "t13"]
    assert [(d["phase"], d["step"]) for d in t13] == [
        ("generate", 1),
        ("tool_step", 2),
        ("announce_reask", 2),
        ("tool_step", 3),
    ]
    t14 = [r["data"]["name"] for r in recs if r["kind"] == "tool.call" and r["turn_id"] == "t14"]
    assert set(t14) == {"wait"}, t14


def test_exact_rows_json() -> None:
    """A1.5: the rows, their columns (exactly A1.5's) and every pinned value; exit 0."""
    rows = json_rows(str(FIXTURE))
    assert [r["turn"] for r in rows] == list(WANT), [r.get("turn") for r in rows]
    for r in rows:
        assert set(r) == COLUMNS, sorted(set(r) ^ COLUMNS)
        assert Path(str(r["file"])).name == FIXTURE.name, r["file"]
        want = WANT[r["turn"]]
        for key, value in want.items():
            if value is None:
                continue
            if key == "answer":
                assert cut_ok(r["answer"], value), (r["turn"], r["answer"])
            else:
                assert r[key] == value and type(r[key]) is type(value), (
                    r["turn"],
                    key,
                    r[key],
                    value,
                )


def test_text_rows_and_totals() -> None:
    """A1.5: the text output has one line per row (its turn and seq as tokens), none for the other turns, and a
    totals line."""
    proc = run(str(FIXTURE))
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.splitlines()
    for turn, want in WANT.items():
        hits = [ln for ln in lines if turn in ln.split() and str(want["seq"]) in ln.split()]
        assert len(hits) == 1, (turn, hits)
    for turn in NOT_ROWS:
        assert not any(turn in ln.split() for ln in lines), turn
    assert any("total" in ln.lower() for ln in lines), proc.stdout[-500:]


def test_two_files_in_argument_order(tmp_path: Path) -> None:
    """A1.5 ``<session.jsonl>...``: two inputs give both files' rows, in argument order, each naming its file."""
    other = tmp_path / "copy_of_claims.jsonl"
    shutil.copyfile(FIXTURE, other)
    rows = json_rows(str(other), str(FIXTURE))
    names = [Path(str(r["file"])).name for r in rows]
    assert names == [other.name] * len(WANT) + [FIXTURE.name] * len(WANT), names
    assert [r["seq"] for r in rows] == [w["seq"] for w in WANT.values()] * 2


def test_reask_answer_is_its_own_attempts() -> None:
    """t10: the row's answer is the ``announce_reask`` request's own commits (joined by ``attempt_id``), not the
    step's announcement in front of it (a join by ``request_id``)."""
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    assert rows["t10"]["answer"] == "There is no checkout button on this screen.", rows["t10"]
    assert rows["t10"]["reasked_this_step"] is True and rows["t10"]["would_check"] is False, rows[
        "t10"
    ]


def test_reasked_counts_only_the_final_step() -> None:
    """t13: step 2 was re-asked and its call executed; the final step 3 was not re-asked, so ``reasked_this_step``
    is false and the row would be checked."""
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    assert rows["t13"]["reasked_this_step"] is False, rows["t13"]
    assert rows["t13"]["would_check"] is True, rows["t13"]


def test_completion_check_request_is_never_the_final_request() -> None:
    """T-0207 A2-R3: t15 logs a ``phase: "completion_check"`` request after its final step. The row is the step's
    ``tool_step`` request (its seq, step and draft answer), never the check request; no row carries the check's seq
    or the check's answer. Premise: the fixture's check request is the turn's last ``llm.request``."""
    recs = [json.loads(ln) for ln in FIXTURE.read_text().splitlines()]
    t15 = [
        (r["seq"], r["data"]["phase"])
        for r in recs
        if r["kind"] == "llm.request" and r["turn_id"] == "t15"
    ]
    assert t15[-1] == (T15_CHECK_SEQ, "completion_check"), t15
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    assert "t15" in rows, sorted(rows)
    r = rows["t15"]
    assert (r["seq"], r["step"], r["would_check"]) == (16579, 2, True), r
    assert r["answer"] == "I have added the comic to your basket.", r["answer"]
    assert all(x["seq"] != T15_CHECK_SEQ for x in rows.values())
    assert "still empty" not in json.dumps(list(rows.values()))


def test_t0208_shared_would_check_follows_the_label() -> None:
    """T-0208 A2.6: on a shared-source turn ``would_check`` follows A2.1 (the step's own current image, read from the
    final request's ``label_key``): a browser key (t16 ``updated``, t17 ``unchanged``, t18 ``now``) is true, the
    shared camera label (t19) is false; t6 (``shared_screen``) stays false. Each would-check row's body check passes
    (the script exits 0 and the text output says ``ok`` on those rows). Premise: the four turns' turn lines say
    ``shared`` and each executed one ``scroll`` (so only the label rule can tell them apart)."""
    recs = [json.loads(ln) for ln in FIXTURE.read_text().splitlines()]
    for turn in T0208_SHARED:
        lines = [r for r in recs if r["kind"] == "turn.record" and r["turn_id"] == turn]
        assert [r["data"]["extras"]["vision_source"] for r in lines] == ["shared"], turn
        executed = [
            r["data"]["name"]
            for r in recs
            if r["kind"] == "tool.call"
            and r["turn_id"] == turn
            and r["data"]["status"] == "executed"
        ]
        assert executed == ["scroll"], (turn, executed)
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    got = {t: (rows[t]["label_key"], rows[t]["would_check"]) for t in [*T0208_SHARED, "t6"]}
    want = {t: (key, would) for t, (_s, key, _a, would) in T0208_SHARED.items()}
    want["t6"] = ("shared_screen", False)
    assert got == want, (
        f"T-0208 A2.6: would_check on shared-source turns follows the final label (browser key -> true, "
        f"shared_screen/shared_camera -> false); got {got}, want {want}"
    )
    proc = run(str(FIXTURE))
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr}"
    for turn, (seq, _key, _a, would) in T0208_SHARED.items():
        (line,) = [
            ln for ln in proc.stdout.splitlines() if turn in ln.split() and str(seq) in ln.split()
        ]
        assert (" ok " in f" {line} ") is would, (
            f"T-0208 A2.6: the body check runs (and passes) exactly on the would-check rows; {line!r}"
        )


def test_wait_is_a_browser_action() -> None:
    """t14: the only executed action is ``wait`` (A1.5 lists it): the turn is a row."""
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    assert "t14" in rows and rows["t14"]["would_check"] is True, sorted(rows)


@pytest.mark.parametrize("turn", ["t11", "t12"], ids=["degenerate-answer", "aborted"])
def test_unclean_final_request_is_a_row_never_checked(turn: str) -> None:
    """A2.1 as read from the log ("not aborted, no T-0142 trip"): a final request that never reached ``llm.done``
    (t11: the answer tripped T-0142b, a note; t12: aborted mid-stream, no note at all) is still a row (a non-empty
    answer, no call, after an executed click) but ``would_check`` is false. t13 and t14 guard the other side: a
    clean final step is checked."""
    rows = {r["turn"]: r for r in json_rows(str(FIXTURE))}
    assert turn in rows, sorted(rows)
    assert rows[turn]["would_check"] is False, rows[turn]
    assert rows["t13"]["would_check"] is True and rows["t14"]["would_check"] is True


def test_search_last_and_declined_click_is_not_a_row() -> None:
    """Adversarial: t7's only executed action is a ``google_search`` (its click was declined, never executed), and
    t4 is search-only: neither is a row, though both end with a non-empty answer and no call."""
    rows = json_rows(str(FIXTURE))
    assert not {"t4", "t7"} & {r["turn"] for r in rows}
    assert "t5" not in {r["turn"] for r in rows}, (
        "a turn whose final request ends with a call is not a row"
    )


# --------------------------------------------------------------------------------------------- exit codes


def _without_turns(tmp_path: Path, keep: set[str], name: str) -> Path:
    out = []
    for ln in FIXTURE.read_text().splitlines():
        r = json.loads(ln)
        if r.get("turn_id") is None or r["turn_id"] in keep:
            out.append(ln)
    path = tmp_path / name
    path.write_text("\n".join(out) + "\n")
    return path


def test_exit_two_on_no_row_or_unreadable(tmp_path: Path) -> None:
    """A1.5: exit 2 when a file holds no row (only t0, t4, t5, t7), is missing, or has a line that is not JSON;
    also when one of several inputs holds no row."""
    none = _without_turns(tmp_path, NOT_ROWS, "none.jsonl")
    assert run(str(none)).returncode == 2
    assert run(str(none), "--json").returncode == 2
    assert run(str(tmp_path / "missing.jsonl")).returncode == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(FIXTURE.read_text().splitlines()[0] + "\nnot json\n")
    assert run(str(bad)).returncode == 2
    assert run(str(FIXTURE), str(none)).returncode == 2, (
        "A1.5: 'a file ... holds no row' is per input"
    )


def test_body_check_is_real_and_no_socket(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """A1.5 body check, adversarial (the check is real, not a print): run in-process, the honest ``check_prompt``
    gives exit 0; one that also replaces the system message gives exit 1 (every would-check request's messages
    minus the last two no longer equal the step's). No socket is opened (connect and create_connection raise)."""
    import importlib

    assert SCRIPT.exists(), "T-0207 A1.5: scripts/replay_completion_check.py does not exist yet"
    try:
        cc = importlib.import_module(COMPLETION)
    except ModuleNotFoundError as exc:
        raise NotImplementedError(f"T-0207 A1.1: {COMPLETION} does not exist yet") from exc
    from local_astra.contracts.llm import Message, TextPart

    real = cc.check_prompt

    def wrong(sent: Any, said: str, frame: Any, *, report: bool) -> Any:
        got = real(sent, said, frame, report=report)
        system = Message(role="system", content=(TextPart(text="changed"),))
        return got.model_copy(update={"messages": (system, *got.messages[1:])})

    def no_socket(*a: Any, **k: Any) -> Any:
        raise AssertionError("T-0207 A1.5: the replay opened a socket")

    monkeypatch.setattr(socket.socket, "connect", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    for honest in (True, False):
        monkeypatch.setattr(cc, "check_prompt", real if honest else wrong)
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE), "--json"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        out = capsys.readouterr().out
        if honest:
            assert exc.value.code in (0, None), out[-1000:]
            assert [r["turn"] for r in json.loads(out)["rows"]] == list(WANT)
        else:
            assert exc.value.code == 1, (
                f"a wrong check_prompt went unnoticed: exit {exc.value.code}"
            )


def _wrong_variants() -> dict[str, Any]:
    from local_astra.contracts.llm import Message, TextPart

    def answer_dropped(got: Any, sent: Any, said: str) -> Any:
        # the second-last message is the continuation alone, not continuation + answer
        cont = sent.messages[-1].text() if sent.continue_final_message else ""
        msg = Message(role="assistant", content=(TextPart(text=cont),))
        return got.model_copy(update={"messages": (*got.messages[:-2], msg, got.messages[-1])})

    def answer_doubled(got: Any, sent: Any, said: str) -> Any:
        text = got.messages[-2].text() + said
        msg = Message(role="assistant", content=(TextPart(text=text),))
        return got.model_copy(update={"messages": (*got.messages[:-2], msg, got.messages[-1])})

    def text_first(got: Any, sent: Any, said: str) -> Any:
        img, text = got.messages[-1].content
        msg = Message(role="user", content=(text, img))
        return got.model_copy(update={"messages": (*got.messages[:-1], msg)})

    def image_dropped(got: Any, sent: Any, said: str) -> Any:
        msg = Message(role="user", content=(got.messages[-1].content[1],))
        return got.model_copy(update={"messages": (*got.messages[:-1], msg)})

    def image_replaced_by_text(got: Any, sent: Any, said: str) -> Any:
        # the instruction still second, but no image before it
        text = got.messages[-1].content[1]
        msg = Message(role="user", content=(TextPart(text="screen"), text))
        return got.model_copy(update={"messages": (*got.messages[:-1], msg)})

    return {
        "image_replaced_by_text": image_replaced_by_text,
        "answer_dropped": answer_dropped,
        "answer_doubled": answer_doubled,
        "text_first": text_first,
        "image_dropped": image_dropped,
    }


@pytest.mark.parametrize(
    "variant",
    ["answer_dropped", "answer_doubled", "text_first", "image_dropped", "image_replaced_by_text"],
)
def test_body_check_tail_mutants_fail(
    variant: str, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A1.5 body check, adversarial on its tail: a ``check_prompt`` whose second-last message is not the
    continuation extended by the answer (the answer dropped or doubled), or whose last message puts the text before
    the image (or drops the image), makes the run exit 1 (A1.5 names no per-row failure marker, so only the exit is pinned); the honest
    ``check_prompt`` exits 0 (``test_body_check_is_real_and_no_socket``)."""
    import importlib

    assert SCRIPT.exists(), "T-0207 A1.5: scripts/replay_completion_check.py does not exist yet"
    try:
        cc = importlib.import_module(COMPLETION)
    except ModuleNotFoundError as exc:
        raise NotImplementedError(f"T-0207 A1.1: {COMPLETION} does not exist yet") from exc
    real = cc.check_prompt
    change = _wrong_variants()[variant]

    def wrong(sent: Any, said: str, frame: Any, *, report: bool) -> Any:
        return change(real(sent, said, frame, report=report), sent, said)

    monkeypatch.setattr(cc, "check_prompt", wrong)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(FIXTURE)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    out = capsys.readouterr()
    assert exc.value.code == 1, f"{variant} went unnoticed: exit {exc.value.code}\n{out.out[-800:]}"


def test_script_never_imports_network_clients() -> None:
    """A1.5 "no socket": the script's own imports name no HTTP client or socket module."""
    assert SCRIPT.exists(), "T-0207 A1.5: scripts/replay_completion_check.py does not exist yet"
    import ast

    tree = ast.parse(SCRIPT.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    bad = [
        n
        for n in names
        if n.split(".")[0] in {"socket", "httpx", "requests", "urllib", "http", "aiohttp"}
    ]
    assert not bad, bad


# --------------------------------------------------------------------------------------------- James's log


def _james_log() -> Path:
    common = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for root in (REPO, Path(common).parent if common else REPO):
        p = root / JAMES_LOG
        if p.is_file():
            return p
    pytest.skip(f"{JAMES_LOG} is not on this machine (James's session log, never committed)")


def test_james_log_expected_rows() -> None:
    """A1.5 "Expected on James's log": 12 rows; seq 2493 and 5538 would_check and last_action_unchanged true,
    report_step false; seq 16197 report_step true."""
    assert SCRIPT.exists(), "T-0207 A1.5: scripts/replay_completion_check.py does not exist yet"
    rows = json_rows(str(_james_log()))
    assert [r["seq"] for r in rows] == JAMES_SEQS, [r["seq"] for r in rows]
    by = {r["seq"]: r for r in rows}
    for seq in (2493, 5538):
        assert by[seq]["would_check"] is True and by[seq]["last_action_unchanged"] is True, by[seq]
        assert by[seq]["report_step"] is False, by[seq]
    assert by[16197]["report_step"] is True, by[16197]
    assert by[2493]["answer"] == "I have added volume 7 of Vinland Saga to your cart."
    assert (
        by[5538]["answer"] == "I have purchased The Legend of Zelda: Tears of the Kingdom for you."
    )
