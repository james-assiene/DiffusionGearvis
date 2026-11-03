"""T-0120c: the compaction re-check harness's ``main`` end to end against a mock engine on loopback, the no-engine and
stop exits, the refusals with a live engine that must never be contacted, and the pre-launch calibration on the
c398fdf record (CPU only; ``integration``, ``slow``). Every row is red on the named assertion in ``harness()`` until
the harness exists.

The harness is loaded lazily by path (``tests.unit.test_t0120c_compaction_recheck.harness``, whose docstring states
the assumed surface) and ``main(argv)`` is called in-process: running anything under the GPU directory from a shell is
gated, and this run spends no GPU time.

**The mock engine** is T-0115a's (``tests/integration/test_t0115a_harness_smoke.py``'s ``Engine115a``, imported
read-only) with its script replaced: a real HTTP server on ``127.0.0.1:<ephemeral>`` in a thread serving ``/health``,
``/v1/models``, ``/version``, ``/metrics``, ``POST /tokenize`` and ``POST /v1/chat/completions`` in the arm-(b) SSE
shape, from token ids. **The script:** a request whose id is in the row's re-search set answers a closed thought,
``Let me search again.`` and a native ``google_search`` call (stop id 50); otherwise, if the prompt holds the
history's fact (typed by hand: ``19:40``, ``06:45``, ``1,187``, ``41.6``), ``The answer is <fact>.``; else
``Okay.``. Every reply is two canvases (R1: canvas 0 ends inside the thought). Toggles: ``faults`` (request id ->
``"500"``).

**Expected, by hand** (the healthy run with ``k`` scripted re-searches among the 40 old-COMPACT requests): 160 chat
requests in ``schedule()`` order -- 40 warm, 120 arm (old FULL 40, old COMPACT 40, kept COMPACT 40) -- at concurrency
1, unseeded. Old FULL correct 40/40 (C holds), kept COMPACT correct 40/40 (K holds), old COMPACT re-search k/40 and
unsupported (40 - k)/40 ("Okay." states no value, so invented 0/40, I holds). At k = 24 R holds: **"stub
re-searches"**; at k = 23 R fails: **"stub falls short: R"**. Both lists agree here (every fact is a digit form). The
ferry re-searches repeat the history's fourth query (``query_repeats_history`` true), the others do not.

**The calibration row lives here, not in ``tests/eval``** (a reading, flagged in the test-author report): criterion 10
names the eval module, but a non-``gpu`` row there breaks T-0046's path-only-selects-nothing invariant
(``tests/unit/test_pytest_config.py``). It reads ``T0120C_BASELINE_DIR`` and skips without it::

    T0120C_BASELINE_DIR=/home/user/demos/local_astra/.claude/worktrees/t0115-launch/logs/t0115a-loop-requests-c398fdf \\
        uv run pytest tests/integration/test_t0120c_harness_smoke.py -q -k calibration -rA -s

Regime: CPU, mock engine on loopback, real ``VllmClient`` (id mode, the pinned detokeniser) and ``loop.compact``;
no measurement -- the numbers it writes describe the mock, never the model. The calibration reads the c398fdf run
([Ra] in docs/cd/toolcall-loop-requests.md; n 160 arm lines).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import loop
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.eval.test_compaction_recheck import (
    check_report,
    classify,
    derive,
    holds_c4,
    is_invented,
)
from tests.integration.test_t0093_harness_smoke import native_call, require_clean_tree
from tests.integration.test_t0098_harness_smoke import _has_key, no_sockets
from tests.integration.test_t0115a_harness_smoke import (
    CHAT_PATH,
    STOP_TURN,
    THOUGHT_HEAD,
    THOUGHT_TAIL,
    Engine115a,
    _prompt_text,
)
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0115a_loop_requests import STUB_5, hand_text
from tests.unit.test_t0120c_compaction_recheck import (
    ACCEPTED_C4,
    ACCEPTED_T0115A,
    C398FDF_E_TEXTS_SHA256,
    COMPACT_KEYS,
    HISTORY_IDS,
    OLD_NOTE_TAIL,
    OLD_STUB_5,
    T0115A_HARNESS,
    WARM_FULL_KEYS,
    _load,
    found,
    harness,
    old_elided_payload,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
GATE_FILES = (
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/llm_client.py",
    "scripts/gpu/t0115a_loop_requests.py",
    "scripts/gpu/t0120c_compaction_recheck.py",
)
STOP_TOOL = 50
#: the facts, typed by hand from T-0115a's histories (docs/cd/toolcall-loop-requests.md)
FACTS = {"ferry": "19:40", "market": "06:45", "hall": "1,187", "lighthouse": "41.6"}
SCHEDULE_KEYS = ("row", "history", "placement", "arm", "repeat", "request_id")
CELLS = (("old", "FULL"), ("old", "COMPACT"), ("kept", "COMPACT"))
OUTCOMES = ("correct", "re_search", "unsupported", "other", "errored")
#: the ferry re-searches repeat the history's fourth query; the others are new queries
FERRY_QUERY = "halvard ferries official timetable"
MARKET_QUERY = "ostrava lane market second search"
HALL_QUERY = "corvane hall seats again"
#: criteria 7 and 12: the c398fdf T-0115a record, pinned by content (the task names requests.jsonl's sha; the other
#: three are the launch line's manifest/report prefixes, read in full from the files and typed in)
BASELINE_SHA256 = {
    "requests.jsonl": "dfd144d11b37b5df735072609379f2ce01dc60b66cddc8d1149acdb7b7bf9ffc",
    "manifest.json": "a33aa86bbf9bfe0a5b4bafd4f9b7019950919c6272642608846aacf6133ac1f7",
    "report.json": "abe7a945b9718abc5fa347e464c3a168dbf261014097f706d0775271bbd04cfb",
    "report.md": "fbbb2e3568be7759463275eb59bf61610b7c498b40c9707f5ce83130d000fd9a",
}
#: criterion 7, typed from the task
CALIBRATION_RE_SEARCH = 9
CALIBRATION_INVENTED = 0


def research_set(k: int) -> dict[str, str]:
    """``k`` scripted re-searches among the old-COMPACT requests: ferry r0-r9, market r0-r9, then hall r0.. (k <= 24)."""
    ids = [(f"t0120c-e-ferry-old-COMPACT-r{r}", FERRY_QUERY) for r in range(10)]
    ids += [(f"t0120c-e-market-old-COMPACT-r{r}", MARKET_QUERY) for r in range(10)]
    ids += [(f"t0120c-e-hall-old-COMPACT-r{r}", HALL_QUERY) for r in range(4)]
    assert 0 <= k <= len(ids)
    return dict(ids[:k])


class Engine120c(Engine115a):
    """T-0115a's mock engine with the T-0120c script (module docstring)."""

    def __init__(self, research: Mapping[str, str], **toggles: Any) -> None:
        super().__init__({h: (f,) for h, f in FACTS.items()}, **toggles)
        self.research = dict(research)

    def answer(self, body: Mapping[str, Any]) -> dict[str, Any]:
        kind = self.kind(body)
        rid = str((body.get("vllm_xargs") or {}).get("astra_request_id", ""))
        text = _prompt_text(body)
        hits = [hid for hid, fact in FACTS.items() if fact in text]
        if kind not in ("warm", "E") or len(hits) > 1:
            self.unknown.append(f"{rid}: kind={kind} hits={hits}")
        if rid in self.research:
            call = native_call(
                {"name": "google_search", "arguments": {"query": self.research[rid]}}
            )
            tail = [*self.encode(THOUGHT_TAIL + "Let me search again." + call), STOP_TOOL]
        else:
            said = f"The answer is {FACTS[hits[0]]}." if hits else "Okay."
            tail = [*self.encode(THOUGHT_TAIL + said), STOP_TURN]
        return {
            "kind": kind,
            "body": body,
            "request_id": rid,
            "fact_of": hits[0] if hits else None,
            "chunks": [self.encode(THOUGHT_HEAD), tail],
        }


@contextlib.contextmanager
def serve_120c(research: Mapping[str, str], **toggles: Any) -> Iterator[tuple[Engine120c, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine120c(research, **toggles)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0120c-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chat_posts(eng: Engine115a) -> int:
    return sum(1 for method, path in eng.requests if method == "POST" and path == CHAT_PATH)


def t0115a_base_messages() -> dict[str, list[Any]]:
    """The oracle for every sent body's ``messages[:-1]``: T-0115a's base prompt for each history (its
    ``_base_prompt`` on the session's builder, the lean profile, the real tool kit and the ``E_SCREEN_CASE`` row), as
    the client serialises it. T-0115a is loaded here by path, its own copy, so a patch to the harness's ``T0115A``
    (the M2b mutant) cannot reach the oracle."""
    from local_astra import cli
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.conductor.prompt_builder import PromptBuilder
    from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
    from local_astra.config import load

    t15 = _load(T0115A_HARNESS, "_t0115a_for_t0120c_smoke_oracle")
    profile = load().profiles["lean"]
    kit, _ = cli.load_tool_kit(profile)
    builder = PromptBuilder(
        DEFAULT_SYSTEM_PROMPT, thinking=bool(profile.diffusion.thinking), tools=kit.tools
    )
    rows = {
        str(r["id"]): r
        for r in (
            json.loads(x)
            for x in Path(t15.CASES_PATH).read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    }
    client = VllmClient()
    return {
        str(x.id): client.request_body(
            t15._base_prompt(builder, x.question, rows[str(t15.E_SCREEN_CASE)]), "oracle"
        )["messages"]
        for x in t15.E_HISTORIES
    }


def _message_texts(messages: list[Any]) -> list[str]:
    """Every text a list of chat messages carries (string content, or the ``text`` of each text part; images
    skipped)."""
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [str(p.get("text", "")) for p in content if p.get("type") == "text"]
    return out


def expected_tally(k: int) -> dict[str, dict[str, dict[str, int]]]:
    """By hand: old FULL 40 correct; old COMPACT k re-search, 40 - k unsupported; kept COMPACT 40 correct."""

    def cell(**counts: int) -> dict[str, int]:
        return {o: counts.get(o, 0) for o in OUTCOMES}

    return {
        "old": {"FULL": cell(correct=40), "COMPACT": cell(re_search=k, unsupported=40 - k)},
        "kept": {"COMPACT": cell(correct=40)},
    }


# =====================================================================================================
# The healthy run
# =====================================================================================================


@pytest.mark.parametrize(
    ("k", "want_outcome"), [(24, "stub re-searches"), (23, "stub falls short: R")]
)
def test_mock_run_end_to_end(tmp_path: Path, k: int, want_outcome: str) -> None:
    """Criteria 1-10 on one complete run (module docstring): exit 0; exactly 160 chat POSTs (no Q4, no Q5, no kept
    FULL) in ``schedule()`` order at concurrency 1, unseeded, loopback only; every body's final text is the
    hand-joined continuation -- warm steps 1-4, FULL steps 1-5, COMPACT with steps 1-3 the 132-char invite stub and no
    old stub; ``requests.jsonl`` mirrors the schedule with both outcomes and the re-search queries; the manifest
    carries the four gate shas, the c398fdf continuity (16 equal, 8 differ), the lists and the patterns;
    ``report.json`` is ``build_report(run_dir)`` and agrees with the eval oracle; k = 24 reads "stub re-searches",
    k = 23 "stub falls short: R" (the R bar at its boundary, end to end)."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0120c-compaction-recheck"
    research = research_set(k)
    with serve_120c(research) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    planned = list(h.schedule())
    assert chat_posts(eng) == 160, f"criterion 2: {chat_posts(eng)} chat POSTs, want exactly 160"
    assert Counter(c["kind"] for c in eng.chats) == Counter({"warm": 40, "E": 120}), Counter(
        c["kind"] for c in eng.chats
    )
    assert [c["request_id"] for c in eng.chats] == [e["request_id"] for e in planned], (
        "not the schedule's order"
    )
    assert eng.max_inflight == 1, f"criterion 1: concurrency {eng.max_inflight}"
    histories = {x.id: x for x in h.E_HISTORIES}
    base_messages = t0115a_base_messages()
    every_accepted = [s for strings in ACCEPTED_C4.values() for s in strings]
    for hid, msgs in base_messages.items():
        # premise of the oracle: no accepted string of any history, normalised, in the base prompt's texts
        hits = [found(t, every_accepted) for t in _message_texts(msgs) if found(t, every_accepted)]
        assert hits == [], f"premise: accepted strings in T-0115a's {hid} base prompt: {hits}"
    for c in eng.chats:
        b = c["body"]
        assert not _has_key(b, "seed"), "criterion 1: a seed field was sent"
        assert b.get("tools") and b.get("continue_final_message") is True, c["request_id"]
        entry = next(e for e in planned if e["request_id"] == c["request_id"])
        steps = tuple(h.e_steps(histories[entry["history"]], entry["placement"]))
        if entry["arm"] == "warm":
            want = hand_text(steps[:4])
        elif entry["arm"] == "FULL":
            want = hand_text(steps)
        else:
            want = hand_text(steps, stubbed=(0, 1, 2))
        assert b["messages"][:-1] == base_messages[entry["history"]], (
            f"{c['request_id']}: the body's messages before the final one are not T-0115a's base prompt "
            "(criterion 4: no accepted string outside the fact result)"
        )
        content = b["messages"][-1]["content"]
        assert content == want, (
            f"{c['request_id']}: the body's final text is not the hand-joined continuation"
        )
        if entry["arm"] == "COMPACT":
            assert (
                content.count(STUB_5) == 3
                and OLD_STUB_5 not in content
                and OLD_NOTE_TAIL not in content
            )
    assert not any(e["placement"] == "kept" and e["arm"] == "FULL" for e in planned), (
        "H-T120c-3: kept FULL sent"
    )

    lines = read_lines(out / "requests.jsonl")
    assert [tuple(x.get(key) for key in SCHEDULE_KEYS) for x in lines] == [
        tuple(e.get(key) for key in SCHEDULE_KEYS) for e in planned
    ], "requests.jsonl is not the schedule, in order"
    for x in lines:
        if x["arm"] == "warm":
            continue
        rid = str(x["request_id"])
        want_o = (
            "re_search"
            if rid in research
            else ("unsupported" if (x["placement"], x["arm"]) == ("old", "COMPACT") else "correct")
        )
        assert (x.get("outcome"), x.get("outcome_t0115a")) == (want_o, want_o), (
            rid,
            x.get("outcome"),
            x.get("outcome_t0115a"),
        )
        if rid in research:
            assert x.get("query") == research[rid], (rid, x.get("query"))
            assert x.get("query_repeats_history") is (research[rid] == FERRY_QUERY), (
                rid,
                x.get("query_repeats_history"),
            )
        if (x["placement"], x["arm"]) == ("old", "COMPACT"):
            assert x.get("invented") is False, (rid, x.get("invented"))

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for p in GATE_FILES:
        assert manifest["gate_shas"][p] == _sha_file(REPO / p), f"criterion 10: gate sha of {p}"
    shas = manifest["e_texts_sha256"]
    assert {key: shas[key] for key in WARM_FULL_KEYS} == {
        key: C398FDF_E_TEXTS_SHA256[key] for key in WARM_FULL_KEYS
    }
    assert all(shas[key] != C398FDF_E_TEXTS_SHA256[key] for key in COMPACT_KEYS), (
        "criterion 3: a COMPACT text is c398fdf's"
    )
    assert {key: list(v) for key, v in manifest["elided"].items()} == {
        f"{hid}/{p}": [0, 1, 2] for hid in HISTORY_IDS for p in ("old", "kept")
    }
    assert {key: sorted(v) for key, v in manifest["accepted"].items()} == {
        key: sorted(v) for key, v in ACCEPTED_C4.items()
    }
    assert {key: sorted(v) for key, v in manifest["accepted_t0115a"].items()} == {
        key: sorted(v) for key, v in ACCEPTED_T0115A.items()
    }
    assert set(manifest["invented_patterns"]) == set(HISTORY_IDS), manifest["invented_patterns"]
    assert manifest.get("prompt_tokens"), "criterion 3: the manifest records no prompt_tokens"

    stored = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "report.json is build_report(run_dir)"
    )
    want_tally = expected_tally(k)
    for key in ("tally", "tally_t0115a"):
        for p, a in CELLS:
            cell = stored[key][p][a]
            assert {o: cell.get(o, 0) for o in OUTCOMES} == want_tally[p][a], (key, p, a, cell)
    assert dict(stored["invented"]) == {"k": 0, "n": 40}, stored["invented"]
    assert list(stored["invented_answers"]) == [], stored["invented_answers"]
    assert dict(stored["bars"]) == {"C": True, "R": k >= 24, "I": True, "K": True}, stored["bars"]
    assert stored["outcome"] == want_outcome, stored["outcome"]
    assert stored["complete"] is True
    # the eval oracle (tests/eval/test_compaction_recheck.py) agrees with the report
    got = derive(lines, manifest)
    check_report(stored, got)
    assert got["tally"] == want_tally, got["tally"]
    md = (out / "report.md").read_text(encoding="utf-8")
    assert want_outcome in md and "40/40" in md, "report.md lacks the outcome or the k/n"


# =====================================================================================================
# Exits 1: no engine, and 10 errors in a row
# =====================================================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_engine_absent_exits_1(tmp_path: Path) -> None:
    """Criterion 1: nothing listens at the (loopback) ``--base-url``, so the harness finds no engine and exits 1 --
    not 0, not the refusal code 2 -- having sent no request and written no ``requests.jsonl`` line."""
    h = harness()
    require_clean_tree()
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    out = tmp_path / "t0120c-compaction-recheck"
    with loopback_only() as attempts:
        rc = h.main(["--base-url", f"http://127.0.0.1:{_free_port()}/v1", "--out-dir", str(out)])
    assert attempts == [], attempts
    assert rc == 1, f"criterion 1: no engine exits 1; got {rc}"
    jsonl = out / "requests.jsonl"
    assert not jsonl.exists() or read_lines(jsonl) == [], "requests were recorded without an engine"


def test_ten_errors_in_a_row_stop_the_run(tmp_path: Path) -> None:
    """Criterion 2: the first 10 requests of the schedule answer HTTP 500, so the run stops there: exactly 10 chat
    POSTs, 10 ``errored`` lines, exit 1, the manifest says why it stopped, and the report reads incomplete, hence
    "inconclusive" naming it (criterion 9)."""
    h = harness()
    require_clean_tree()
    first = [e["request_id"] for e in list(h.schedule())[:10]]
    out = tmp_path / "t0120c-compaction-recheck"
    with (
        serve_120c({}, faults=dict.fromkeys(first, "500")) as (eng, base_url),
        loopback_only() as attempts,
    ):
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], attempts
    assert chat_posts(eng) == 10, (
        f"criterion 2: {chat_posts(eng)} chat POSTs after 10 errors in a row"
    )
    assert rc == 1, f"a stopped run exits 1; got {rc}"
    lines = read_lines(out / "requests.jsonl")
    assert [x["request_id"] for x in lines] == first
    assert all(x.get("error") or x.get("status") != 200 for x in lines), lines[:2]
    assert all(x.get("outcome") == "errored" for x in lines if x["arm"] != "warm"), [
        x.get("outcome") for x in lines
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("stopped"), manifest.get("stopped")
    stored = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert stored["complete"] is False
    assert (
        str(stored["outcome"]).startswith("inconclusive:")
        and "incomplete" in str(stored["outcome"]).lower()
    )


# =====================================================================================================
# Refused: nothing sent, with an engine right there
# =====================================================================================================

REFUSED: dict[str, tuple[str, ...]] = {
    "dirty-tree": ("_t0120c_dirty_probe",),
    "non-empty-out-dir": ("t0120c-out",),
    "t0115a-differing-copy": ("t0115a_loop_requests",),
    "c398fdf-pin": ("market/old/FULL",),
    "old-stub": ("COMPACT", "stub"),
    "accepted-in-literal": ("ten journey card",),
}


@pytest.mark.parametrize("why", list(REFUSED))
def test_refused_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criteria 1, 3 and 4 with a live engine on loopback and the real tool kit: a dirty tree (naming the path), a
    non-empty out dir, a differing copy of the T-0115a harness, a warm/FULL text off its c398fdf pin, the old stub
    (no T-0120b), an accepted string in a pool literal. Each: exit 2, the mock never contacted (not even ``/models``),
    no socket, nothing written, the refusal names its cause."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0120c-out"
    probe = REPO / "tests" / "integration" / "_t0120c_dirty_probe.txt"
    if why == "non-empty-out-dir":
        out.mkdir()
        (out / "earlier.txt").write_text("an earlier run\n")
    elif why == "t0115a-differing-copy":
        copy = tmp_path / "copy" / "t0115a_loop_requests.py"
        copy.parent.mkdir()
        copy.write_bytes(
            (REPO / "scripts" / "gpu" / "t0115a_loop_requests.py").read_bytes() + b"# differs\n"
        )
        monkeypatch.setattr(h, "T0115A_PATH", copy)
    elif why == "c398fdf-pin":
        monkeypatch.setattr(
            h, "C398FDF_E_TEXTS_SHA256", {**C398FDF_E_TEXTS_SHA256, "market/old/FULL": "f" * 64}
        )
    elif why == "old-stub":
        monkeypatch.setattr(loop, "elided_payload", old_elided_payload)
    elif why == "accepted-in-literal":
        monkeypatch.setattr(
            h, "ACCEPTED", {**ACCEPTED_C4, "ferry": (*ACCEPTED_C4["ferry"], "ten journey card")}
        )
    with serve_120c({}) as (eng, base_url):
        try:
            if why == "dirty-tree":
                probe.write_text("dirty\n")
            with no_sockets() as attempts:
                rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
        finally:
            probe.unlink(missing_ok=True)
    said = capsys.readouterr()
    assert rc == 2, f"{why}: exit {rc}"
    assert attempts == [], f"{why}: sockets before the refusal: {attempts}"
    assert eng.requests == [], f"{why}: the engine was contacted: {eng.requests[:5]}"
    left = sorted(p.name for p in out.iterdir()) if out.exists() else []
    assert left in ([], ["earlier.txt"]), f"{why}: the refusal wrote {left}"
    text = said.out + said.err
    assert any(tok in text for tok in REFUSED[why]), f"{why}: {text[-400:]!r}"


# =====================================================================================================
# Criterion 7: the pre-launch calibration on the c398fdf record (CPU)
# =====================================================================================================


def test_calibration_c398fdf() -> None:
    """Criterion 7, on the c398fdf T-0115a record (``T0120C_BASELINE_DIR``; skips without it). First the baseline's
    four files have their pinned sha256 (reviewer mutation: calibration against an unpinned baseline). Then the
    harness's pure ``score_line`` on the 160 E arm lines: (a) ``outcome_t0115a`` equals the stored ``outcome`` on
    160/160; (b) old-COMPACT ``re_search`` 9/40, under both lists (R is list-independent); (c) old-COMPACT
    ``invented`` 0/40. Its criterion 4 ``outcome`` equals the eval oracle's on every line, and its ``invented`` the
    oracle's with the harness's own patterns. Prints old FULL, kept COMPACT and kept FULL correct under criterion 4's
    list, for the write-up's pre-registration."""
    raw = os.environ.get("T0120C_BASELINE_DIR", "").strip()
    if not raw:
        pytest.skip("T0120C_BASELINE_DIR is not set: no c398fdf record to calibrate on")
    base = Path(raw) if Path(raw).is_absolute() else REPO / raw
    for name, want in BASELINE_SHA256.items():
        assert _sha_file(base / name) == want, (
            f"criterion 7: {base / name} is not the pinned c398fdf file"
        )
    lines = [
        x
        for x in read_lines(base / "requests.jsonl")
        if x.get("row") == "E" and x.get("arm") in ("FULL", "COMPACT")
    ]
    assert len(lines) == 160, f"{len(lines)} E arm lines in the baseline"

    h = harness()
    scored = {str(x["request_id"]): h.score_line(x) for x in lines}
    mismatched = [
        (str(x["request_id"]), x["outcome"], scored[str(x["request_id"])]["outcome_t0115a"])
        for x in lines
        if scored[str(x["request_id"])]["outcome_t0115a"] != x["outcome"]
    ]
    assert mismatched == [], (
        f"criterion 7(a): {len(mismatched)}/160 not reproduced: {mismatched[:5]}"
    )
    old_compact = [x for x in lines if (x["placement"], x["arm"]) == ("old", "COMPACT")]
    assert len(old_compact) == 40
    re_search = sum(
        1 for x in old_compact if scored[str(x["request_id"])]["outcome"] == "re_search"
    )
    re_search15 = sum(
        1 for x in old_compact if scored[str(x["request_id"])]["outcome_t0115a"] == "re_search"
    )
    assert (re_search, re_search15) == (CALIBRATION_RE_SEARCH, CALIBRATION_RE_SEARCH), (
        re_search,
        re_search15,
    )
    invented = [
        str(x["request_id"]) for x in old_compact if scored[str(x["request_id"])]["invented"]
    ]
    assert len(invented) == CALIBRATION_INVENTED, f"criterion 7(c): flagged {invented}"

    patterns = {key: tuple(v) for key, v in h.INVENTED_PATTERNS.items()}
    off: list[tuple[str, str, Any, Any]] = []
    for x in lines:
        rid = str(x["request_id"])
        want = classify(x, ACCEPTED_C4[str(x["history"])], holds_c4)
        if scored[rid]["outcome"] != want:
            off.append((rid, "outcome", scored[rid]["outcome"], want))
        inv = is_invented(x, want, patterns[str(x["history"])])
        if (x["placement"], x["arm"]) == ("old", "COMPACT") and bool(
            scored[rid]["invented"]
        ) != inv:
            off.append((rid, "invented", scored[rid]["invented"], inv))
    assert off == [], f"the harness's scoring disagrees with the eval oracle: {off[:5]}"

    def correct(p: str, a: str) -> str:
        mine = [x for x in lines if (x["placement"], x["arm"]) == (p, a)]
        return f"{sum(1 for x in mine if scored[str(x['request_id'])]['outcome'] == 'correct')}/{len(mine)}"

    print(
        "T-0120c calibration on c398fdf ([Ra], n 40 per cell) under criterion 4's list: "
        f"old FULL correct {correct('old', 'FULL')}, kept COMPACT correct {correct('kept', 'COMPACT')}, "
        f"kept FULL correct {correct('kept', 'FULL')}; old COMPACT re-search {re_search}/40, invented "
        f"{len(invented)}/40"
    )
