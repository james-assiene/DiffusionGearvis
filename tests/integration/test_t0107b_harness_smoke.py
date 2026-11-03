"""T-0107b: the high-stakes A/B harness's ``main`` end to end against a mock engine on loopback (CPU only;
``integration``, ``slow``).

The rows that run the harness load it through ``tests.unit.test_t0107b_harness.harness``, which fails on a named
assertion when the harness is absent. One premise row (the mock places every
hand-built request correctly and can encode every scripted reply) needs no harness and is green.

The harness is loaded by path and ``main(argv)`` is called in-process (running anything under the GPU directory from
a shell is gated; this run spends no GPU time). **The mock engine** is T-0098's (``Engine``/``serve`` pattern: a real
HTTP server on ``127.0.0.1:<ephemeral>``, the recorded SSE shape from token ids), subclassed with T-0107b's script. It
places each chat request by its ``messages[0]`` text (A0 / H1, the hand-built texts), its stratum (S3:
``continue_final_message``; S2: an assistant turn before the last user turn; else S1), its case (the S1 or S3
request text; on S2 the history's user text) and script (the transcript), and counts the repeat as the number of
earlier requests with the same (stratum, case, script, arm). It answers with ``outcome_of`` of the unit module, the
script whose numbers are typed there by hand: A0 never asks; H1 asks on high_stakes S1 except three trials, over-asks
once, fails to follow through three times and acts once after a no -- every bar holds, so **"adopt H1"**.

Regime: CPU, mock engine on loopback, real ``PromptBuilder`` and ``VllmClient`` (id mode, the pinned detokeniser);
no measurement -- the numbers describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import functools
import json
import socket
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.eval.test_t0107b_high_stakes_ab import N_REQUESTS, check_report, derive, read_lines
from tests.integration.test_t0093_harness_smoke import require_clean_tree
from tests.integration.test_t0098_harness_smoke import (
    STOP_TOOL,
    STOP_TURN,
    Engine,
    _has_key,
    _text_of,
)
from tests.unit._pre_t0155_persona import pre_t0155_fixture
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0107b_harness import (
    BASELINE_ARMS,
    LEAVE,
    NO,
    NO_TOOL_ANSWER,
    YES,
    case_rows,
    expected_a0_prompt,
    expected_arm_texts,
    harness,
    native_call,
    outcome_of,
    question_of,
    tool_kit,
    units,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)
THOUGHT = "<|channel>thought\nOK.<channel|>"


class Engine107(Engine):
    """T-0098's mock engine with T-0107b's placement and script (module docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_of = {text: arm for arm, text in expected_arm_texts().items()}
        self.cases = case_rows()
        self.s1_texts = sorted(((c["user"], c) for c in self.cases), key=lambda t: -len(t[0]))
        self.s3_texts = sorted(
            ((c["s3"]["user"], c) for c in self.cases if c["s3"]), key=lambda t: -len(t[0])
        )
        self.seen: Counter[tuple[str, str, str | None, str]] = Counter()

    def place(
        self, body: Mapping[str, Any]
    ) -> tuple[str | None, str, dict[str, Any] | None, str | None]:
        """(arm, stratum, case, script) of one chat request."""
        messages = list(body.get("messages") or [])
        system = _text_of(messages[0].get("content")) if messages else ""
        arm = self.arm_of.get(system)
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        script: str | None = None
        if body.get("continue_final_message"):
            stratum, text, table = "S3", _text_of(messages[users[-1]].get("content")), self.s3_texts
        elif any(m.get("role") == "assistant" for m in messages[1 : users[-1]]):
            stratum, text, table = "S2", _text_of(messages[users[0]].get("content")), self.s1_texts
            last = _text_of(messages[users[-1]].get("content"))
            script = "yes" if YES in last else "no" if NO in last else None
        else:
            stratum, text, table = "S1", _text_of(messages[users[-1]].get("content")), self.s1_texts
        case = next((c for t, c in table if t in text), None)
        return arm, stratum, case, script

    def reply(self, kind: str, case: Mapping[str, Any]) -> list[int]:
        ref = case.get("reference_call")
        call = native_call(ref["name"], ref["arguments"]) if ref else ""
        if kind == "call":
            return [*self.encode(THOUGHT + call), STOP_TOOL]
        if kind == "ask":
            return [*self.encode(THOUGHT + question_of(case)), STOP_TURN]
        if kind == "ask_call":
            return [*self.encode(THOUGHT + question_of(case) + call), STOP_TOOL]
        if kind == "leave":
            return [*self.encode(THOUGHT + LEAVE), STOP_TURN]
        return [*self.encode(THOUGHT + NO_TOOL_ANSWER), STOP_TURN]

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        arm, stratum, case, script = self.place(body)
        if arm is None or case is None or (stratum == "S2" and script is None):
            self.unknown.append(
                f"arm={arm} stratum={stratum} case={case and case['id']} script={script}"
            )
            return {
                "body": body,
                "arm": arm,
                "stratum": stratum,
                "case_id": None,
                "ids": [STOP_TURN],
            }
        with self.lock:
            key = (stratum, str(case["id"]), script, arm)
            repeat = self.seen[key]
            self.seen[key] += 1
        kind = outcome_of(arm, stratum, case, script, repeat)
        return {
            "body": body,
            "arm": arm,
            "stratum": stratum,
            "case_id": case["id"],
            "script": script,
            "repeat": repeat,
            "ids": self.reply(kind, case),
        }


@contextlib.contextmanager
def serve_107() -> Iterator[tuple[Engine107, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine107()
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0107b-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@functools.cache
def _run(out: str) -> tuple[Any, int, list[str]]:
    """One healthy run per session (both rows read it)."""
    h = harness()
    with serve_107() as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", out])
    return eng, rc, list(attempts)


def healthy_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, int, list[str], Path]:
    harness()  # red here, by name, before any premise skip
    require_clean_tree()
    out = Path(str(tmp_path_factory.getbasetemp())) / "t0107b-mock-run"
    eng, rc, attempts = _run(str(out))
    return eng, rc, attempts, out


# =====================================================================================================
# Premise (no harness)
# =====================================================================================================


def test_mock_places_every_hand_built_request() -> None:
    """Premise: the mock places each hand-built A0 and H1 request (criterion 3's S1, S2 yes/no and S3 prompts, as the
    unit module builds them) at its arm, stratum, case and script, and encodes every reply it can send."""
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine107()
    kit = tool_kit()
    by_id = {c["id"]: c for c in case_rows()}
    texts = expected_arm_texts()
    client = VllmClient()
    for stratum, cid, script in units():
        case = by_id[cid]
        prompt = expected_a0_prompt(case, stratum, kit, script)
        for arm in ("A0", "H1"):
            body = json.loads(json.dumps(client.request_body(prompt, "t0107b-premise")))
            body["messages"][0]["content"] = [{"type": "text", "text": texts[arm]}]
            got = eng.place(body)
            assert (got[0], got[1], got[2] and got[2]["id"], got[3]) == (
                arm,
                stratum,
                cid,
                script,
            ), (
                stratum,
                cid,
                script,
                arm,
                got[:2],
            )
        for kind in ("call", "ask", "ask_call", "leave", "answer"):
            if kind in ("call", "ask_call") and not case.get("reference_call"):
                continue
            assert eng.reply(kind, case), (cid, kind)


# =====================================================================================================
# The healthy run
# =====================================================================================================


def test_post_count_equals_the_schedule(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Criterion 7 on one complete run: exit 0; no socket leaves loopback; every chat request placed; the POST count
    is ``len(schedule())`` == 740, in the schedule's order (stratum, case, script, arm); S1/S2 pass 1 (no prefill, no
    continuation), S3 a continuation; no seed; tools on; each pair differs only in ``messages[0]``; ``trials.jsonl``
    mirrors the requests line for line."""
    eng, rc, attempts, out = healthy_run(tmp_path_factory)
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"
    h = harness()
    items = list(h.schedule())
    chats = eng.chats
    posts = [p for m, p in eng.requests if m == "POST" and p == "/v1/chat/completions"]
    assert len(posts) == len(chats) == len(items) == N_REQUESTS, (len(posts), len(items))

    def field(x: Any, name: str) -> Any:
        return x[name] if isinstance(x, Mapping) else getattr(x, name)

    want = [
        (field(x, "stratum"), field(x, "case_id"), field(x, "script"), field(x, "arm"))
        for x in items
    ]
    assert [(c["stratum"], c["case_id"], c["script"], c["arm"]) for c in chats] == want
    for c in chats:
        b = c["body"]
        assert not _has_key(b, "seed") and b.get("max_tokens") != 1, c["case_id"]
        assert b.get("tool_choice") == "none" and b.get("tools"), "tools on"
        assert bool(b.get("continue_final_message")) is (c["stratum"] == "S3"), c["stratum"]
    for k in range(0, len(chats), 2):
        a, b = (json.loads(json.dumps(chats[k + i]["body"])) for i in (0, 1))
        for body in (a, b):
            body["messages"] = body["messages"][1:]
            (body.get("vllm_xargs") or {}).pop("astra_request_id", None)
        assert a == b, f"pair {k // 2} differs outside messages[0]"
    trials = read_lines(out / "trials.jsonl")
    assert [(t["stratum"], t["case_id"], t.get("script"), t["arm"]) for t in trials] == want
    assert [t["token_ids"] for t in trials] == [c["ids"] for c in chats]


def test_oracle_and_report_agree(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Criterion 8 on the mock run: ``report.json`` is ``build_report(run_dir)``; the eval module's oracle re-derives
    every metric, P1-P6, the void arms, completeness and the outcome from ``trials.jsonl`` and agrees with it; and
    both read the script's hand-typed numbers (``BASELINE_ARMS``) and **"adopt H1"**. The manifest holds both arms'
    texts, shas and lengths."""
    _, rc, _, out = healthy_run(tmp_path_factory)
    assert rc == 0
    h = harness()
    stored = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "report.json is build_report(run_dir)"
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    got = derive(read_lines(out / "trials.jsonl"), stopped=bool(manifest.get("stopped")))
    check_report(stored, got)
    for arm, metrics in BASELINE_ARMS.items():
        for m, v in metrics.items():
            assert got["metrics"][arm][m] == v, (arm, m, got["metrics"][arm][m], v)
    assert got["kind"] == "adopt" and stored["outcome"] == "adopt H1", stored["outcome"]
    texts = expected_arm_texts()
    for arm, text in texts.items():
        assert manifest["arms"][arm]["text"] == text and manifest["arms"][arm]["chars"] == len(
            text
        ), arm


def test_absent_engine_exits_one(tmp_path: Path) -> None:
    """An otherwise valid invocation over a clean tree against a loopback port where nothing listens: exit 1 (no
    engine), no socket off loopback, no trial written."""
    h = harness()
    require_clean_tree()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "t0107b-absent"
    with loopback_only() as attempts:
        rc = h.main(["--base-url", f"http://127.0.0.1:{port}/v1", "--out-dir", str(out)])
    assert attempts == [], attempts
    assert rc == 1, f"no engine: exit {rc}, want 1"
    trials = out / "trials.jsonl"
    assert not trials.exists() or trials.read_text(encoding="utf-8").strip() == ""
