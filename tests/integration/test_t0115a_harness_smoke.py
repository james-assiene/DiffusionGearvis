"""T-0115a: the loop-request harness's ``main`` end to end against a mock engine on loopback, the engine-down stop,
and the refusals that need a clean tree (CPU only; ``integration``, ``slow``). Every row is
red on the named assertion in ``harness()`` until the harness exists.

The harness is loaded lazily by path (``tests.unit.test_t0115a_loop_requests.harness``, whose docstring states the
assumed surface) and ``main(argv)`` is called in-process: running anything under the GPU directory from a shell is
gated, and this run spends no GPU time.

**The mock engine** is a real HTTP server on ``127.0.0.1:<ephemeral>`` in a thread, so the harness's real
``VllmClient`` (form G) and its raw loopback httpx (form A, Q5) run unpatched. It serves ``GET /health``,
``/v1/models``, ``/version``, ``/metrics`` (zero counters), ``POST /tokenize`` and ``POST /v1/chat/completions``
(the arm-(b) SSE shape of ``tests/fakes/vllm_sse.build_lines``, from token ids). It places each chat request by its
body alone: a ``role: tool`` message is form A; a final assistant message with an ``image_url`` part is Q5; a
continued final assistant string with 1, 4 or 5 rendered calls is Q4 G, warm or an E arm. **The script:** if any
accepted string of any T-0115a history (``E_HISTORIES``) is in the prompt's text, a closed thought and
``The answer is <fact>.``; otherwise a closed thought and ``Okay.`` -- "correct" only when the fact is present.
**Every reply is two canvases** (ruling R1): canvas 0 is the thought header and its first words (the canvas ends
inside the thought, so the client marks its commits ``thought_open``), canvas 1 closes the thought and answers. A
classifier that reads "any commit thought-open" instead of the final state counts every reply as not-final.

**Toggles** (each row names its own): ``faults`` (request id -> ``"500"``, an HTTP 500 before any byte of stream,
or ``"truncate"``, the whole reply -- the fact included -- then the stream ends without a finish_reason or
``[DONE]``); ``q5_status`` (every Q5 request answers that HTTP status with an error body, the engine staying up);
``health_flaps`` (after the first Q5 request, that many ``GET /health`` answer 503 before it recovers).

**Expected, by hand** (the healthy run): 263 chat requests in the order of ``schedule()`` -- Q4 G 30, A 30, warm 40,
E 160, Q5 3, the Q5 three last -- at concurrency 1, unseeded. Steps 1-3 are elided in COMPACT, so the old placement
loses the fact there: old FULL correct 40/40, old COMPACT correct 0/40 and unsupported 40/40 ("Okay." is a final
without the fact); kept FULL and COMPACT correct 40/40. EC holds (40 >= 28 both), E1 holds (40 >= 40 - 4), E2 fails
(0 + 0 < 32), E3 fails (40 > 8): **"elision hurts: E2, E3"** (the outcome names every failing bar). Q5: 3 accepted.
Q4: every reply is a final ("Okay."), so G and A each read final 30/30. No request errors, so every tally cell has
``errored`` 0 (ruling R4: the fifth outcome, counted in the cell).

**Readings this module makes where the task is open** (named in the test-author report): a complete run exits 0
whatever the outcome; after an engine-down void the harness still writes ``report.json`` from the rows it has (the
"earlier rows are kept"), and its exit code is only pinned as not the refusal code 2.

Regime: CPU, mock engine on loopback, real ``VllmClient`` (id mode, the pinned detokeniser) and ``loop.compact``;
no measurement -- the numbers it writes describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.fakes.vllm_sse import MODEL, StreamSpec, build_lines
from tests.integration.test_t0093_harness_smoke import Encoder, require_clean_tree
from tests.integration.test_t0098_harness_smoke import _has_key, _kit_loader, no_sockets
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0115a_loop_requests import (
    N_Q5,
    STUB_5,
    harness,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
GATE_FILES = ("local_astra/toolcall/loop.py", "local_astra/conductor/llm_client.py")
STOP_TURN = 106
#: the reply's two canvases (R1): canvas 0 ends inside the thought, canvas 1 closes it and answers
THOUGHT_HEAD = "<|channel>thought\nLet me read"
THOUGHT_TAIL = " the results.<channel|>"
CALL_TAG = "<|tool_call>call:"
CHAT_PATH = "/v1/chat/completions"
SCHEDULE_KEYS = ("row", "case_id", "history", "placement", "arm", "repeat")
OUTCOMES = ("correct", "re_search", "unsupported", "other", "errored")

EXPECTED_KINDS = Counter({"G": 30, "A": 30, "warm": 40, "E": 160, "Q5": 3})
EXPECTED_TALLY = {
    "old": {
        "FULL": {"correct": 40, "re_search": 0, "unsupported": 0, "other": 0, "errored": 0},
        "COMPACT": {"correct": 0, "re_search": 0, "unsupported": 40, "other": 0, "errored": 0},
    },
    "kept": {
        "FULL": {"correct": 40, "re_search": 0, "unsupported": 0, "other": 0, "errored": 0},
        "COMPACT": {"correct": 40, "re_search": 0, "unsupported": 0, "other": 0, "errored": 0},
    },
}
EXPECTED_Q4_FINAL = {"k": 30, "n": 30}
EXPECTED_BARS = {"EC": True, "E1": True, "E2": False, "E3": False}


def _sha_file(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, Mapping))
    return ""


def _prompt_text(body: Mapping[str, Any]) -> str:
    """Every message's text, tool messages and tool-call arguments included."""
    out: list[str] = []
    for m in body.get("messages") or []:
        out.append(_text(m.get("content")))
        for call in m.get("tool_calls") or []:
            out.append(str((call.get("function") or {}).get("arguments", "")))
    return "\n".join(out)


class Engine115a:
    """The scripted answers, every chat request in arrival order, and the in-flight high-water mark."""

    def __init__(
        self,
        accepted: Mapping[str, tuple[str, ...]],
        *,
        kill_on_q5: bool = False,
        faults: Mapping[str, str] | None = None,
        q5_status: int | None = None,
        health_flaps: int = 0,
    ) -> None:
        self.enc = Encoder()
        self.cache: dict[str, list[int]] = {}
        self.accepted = dict(accepted)
        self.kill_on_q5 = kill_on_q5
        self.faults = dict(faults or {})
        self.q5_status = q5_status
        self.health_flaps = health_flaps
        #: /health failures still to serve, armed by the first Q5 request
        self.flaps_pending = 0
        self.flaps_served = 0
        self.q5_seen = 0
        self.faults_served: list[tuple[str, str]] = []
        self.down = False
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.chats: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []
        self.unknown: list[str] = []

    def encode(self, text: str) -> list[int]:
        if text not in self.cache:
            self.cache[text] = self.enc.encode(text)
        return self.cache[text]

    @staticmethod
    def kind(body: Mapping[str, Any]) -> str:
        msgs = body.get("messages") or []
        if any(m.get("role") == "tool" for m in msgs):
            return "A"
        last = msgs[-1] if msgs else {}
        content = last.get("content")
        if (
            last.get("role") == "assistant"
            and isinstance(content, list)
            and any(isinstance(p, Mapping) and p.get("type") == "image_url" for p in content)
        ):
            return "Q5"
        if body.get("continue_final_message") and isinstance(content, str):
            n = content.count(CALL_TAG)
            return {1: "G", 4: "warm", 5: "E"}.get(n, f"continuation-with-{n}-calls")
        return "unplaced"

    def answer(self, body: Mapping[str, Any]) -> dict[str, Any]:
        kind = self.kind(body)
        text = _prompt_text(body)
        hits = [hid for hid, acc in self.accepted.items() if any(a in text for a in acc)]
        if kind not in EXPECTED_KINDS or len(hits) > 1:
            self.unknown.append(f"kind={kind} hits={hits}")
        fact = self.accepted[hits[0]][0] if hits else None
        answer = f"The answer is {fact}." if fact else "Okay."
        chunks = [self.encode(THOUGHT_HEAD), [*self.encode(THOUGHT_TAIL + answer), STOP_TURN]]
        return {
            "kind": kind,
            "body": body,
            "request_id": str((body.get("vllm_xargs") or {}).get("astra_request_id", "")),
            "fact_of": hits[0] if hits else None,
            "chunks": chunks,
        }

    def handler(self) -> type[BaseHTTPRequestHandler]:
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                engine.requests.append(("GET", self.path))
                if engine.down:
                    self._send(503, b'{"error":"engine dead"}', "application/json")
                elif self.path == "/health" and engine.flaps_pending > 0:
                    # a /health blip after a Q5 request: the engine is up, the probe fails this once
                    engine.flaps_pending -= 1
                    engine.flaps_served += 1
                    self._send(503, b'{"error":"health blip"}', "application/json")
                elif self.path == "/health":
                    self._send(200, b"", "text/plain")
                elif self.path == "/v1/models":
                    self._send(
                        200, json.dumps({"data": [{"id": MODEL}]}).encode(), "application/json"
                    )
                elif self.path == "/version":
                    self._send(
                        200, json.dumps({"version": "0.29.0-mock"}).encode(), "application/json"
                    )
                elif self.path == "/metrics":
                    text = (
                        f'vllm:prefix_cache_queries_total{{model_name="{MODEL}"}} 0\n'
                        f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} 0\n'
                        f'vllm:num_requests_running{{model_name="{MODEL}"}} 0\n'
                        f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0\n'
                    )
                    self._send(200, text.encode(), "text/plain")
                else:
                    self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                engine.requests.append(("POST", self.path))
                body = json.loads(
                    self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}"
                )
                if engine.down:
                    self._send(503, b'{"error":"engine dead"}', "application/json")
                    return
                if self.path == "/tokenize":
                    text = str(body.get("prompt", "")) or _prompt_text(body)
                    ids = engine.encode(text)
                    reply = {"count": len(ids), "max_model_len": 16384, "tokens": ids}
                    self._send(200, json.dumps(reply).encode(), "application/json")
                    return
                if self.path != "/v1/chat/completions":
                    self._send(404, b"{}", "application/json")
                    return
                with engine.lock:
                    engine.inflight += 1
                    engine.max_inflight = max(engine.max_inflight, engine.inflight)
                try:
                    chat = engine.answer(body)
                    with engine.lock:
                        engine.chats.append(chat)
                    if chat["kind"] == "Q5" and engine.kill_on_q5:
                        # the patched image path's crash: the EngineCore dies mid-request, no reply
                        engine.down = True
                        self.close_connection = True
                        return
                    if chat["kind"] == "Q5":
                        engine.q5_seen += 1
                        if engine.q5_seen == 1:
                            engine.flaps_pending = engine.health_flaps
                        if engine.q5_status is not None:
                            # the engine refuses the body and stays up
                            err = {"error": {"message": "mock refusal", "code": engine.q5_status}}
                            self._send(
                                engine.q5_status, json.dumps(err).encode(), "application/json"
                            )
                            return
                    fault = engine.faults.get(chat["request_id"])
                    if fault is not None:
                        engine.faults_served.append((chat["request_id"], fault))
                    if fault == "500":
                        err = {"error": {"message": "mock internal error", "code": 500}}
                        self._send(500, json.dumps(err).encode(), "application/json")
                        return
                    spec = StreamSpec(chunks=chat["chunks"])
                    if fault == "truncate":
                        # the whole reply (the fact included), then the stream stops without saying why
                        spec = StreamSpec(chunks=chat["chunks"], truncate=True, done=False)
                    lines = build_lines(spec, engine.enc.detok)
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for line in lines:
                        self.wfile.write(f"{line}\n\n".encode())
                        self.wfile.flush()
                finally:
                    with engine.lock:
                        engine.inflight -= 1

        return Handler


@contextlib.contextmanager
def serve_115a(
    accepted: Mapping[str, tuple[str, ...]], **toggles: Any
) -> Iterator[tuple[Engine115a, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine115a(accepted, **toggles)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0115a-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def accepted_of(h: Any) -> dict[str, tuple[str, ...]]:
    return {x.id: (str(x.fact), *(str(e) for e in x.equivalents)) for x in h.E_HISTORIES}


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def tally_of(lines: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    out = {p: {a: dict.fromkeys(OUTCOMES, 0) for a in ("FULL", "COMPACT")} for p in ("old", "kept")}
    for x in lines:
        if x.get("row") == "E" and x.get("arm") in ("FULL", "COMPACT"):
            out[x["placement"]][x["arm"]][x["outcome"]] += 1
    return out


# =====================================================================================================
# The healthy run
# =====================================================================================================


def test_main_runs_end_to_end_against_a_mock_engine(tmp_path: Path) -> None:
    """Criteria 1-6 on one complete run (module docstring): exit 0; 263 chat requests in ``schedule()`` order, Q5
    last, concurrency 1, no seed, loopback only; form G and the E prompts are ``VllmClient`` continuations, form A
    the ``role: tool`` body; the 40 old-placement COMPACT prompts carry the stub three times and no fact (the 120
    other E arm prompts carry it); ``requests.jsonl`` mirrors the schedule; the tally, the bars and the outcome read
    "elision hurts: E2, E3"; Q5 3 accepted; ``report.json`` is ``build_report(run_dir)``; the manifest carries the
    gate shas of this tree and the histories' accepted strings.

    T-0120b re-scope: ``STUB_5`` is the 132-char search stub (the harness's COMPACT arm is the product's
    ``loop.compact``), so the stub count and the shared-stub prefix check read the new form."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-loop-requests"
    acc = accepted_of(h)
    with serve_115a(acc) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    chats = eng.chats
    assert Counter(c["kind"] for c in chats) == EXPECTED_KINDS, Counter(c["kind"] for c in chats)
    assert [c["kind"] for c in chats[-N_Q5:]] == ["Q5"] * N_Q5, "criterion 3: Q5 is not last"
    assert eng.max_inflight == 1, f"criterion 1: concurrency {eng.max_inflight}"
    for c in chats:
        b = c["body"]
        assert not _has_key(b, "seed"), "criterion 1: a seed field was sent"
        assert b.get("tools"), f"{c['kind']}: tools off"
        if c["kind"] in ("G", "warm", "E"):
            assert (
                b.get("continue_final_message") is True and b.get("add_generation_prompt") is False
            )
            assert (b.get("vllm_xargs") or {}).get("astra_request_id"), "not sent by VllmClient"
        if c["kind"] == "A":
            assert "continue_final_message" not in b, "form A is not a continuation"
    e_bodies = [c for c in chats if c["kind"] == "E"]
    no_fact = [c for c in e_bodies if c["fact_of"] is None]
    assert len(no_fact) == 40, f"{len(no_fact)} E prompts without the fact; want the 40 old COMPACT"
    for c in no_fact:
        assert c["body"]["messages"][-1]["content"].count(STUB_5) == 3, "steps 1-3 not stubbed"

    lines = read_lines(out / "requests.jsonl")
    want = [tuple(e.get(k) for k in SCHEDULE_KEYS) for e in h.schedule()]
    assert [tuple(x.get(k) for k in SCHEDULE_KEYS) for x in lines] == want, (
        "requests.jsonl is not the schedule, in order"
    )
    assert tally_of(lines) == EXPECTED_TALLY, tally_of(lines)
    assert [x["reading"] for x in lines if x["row"] == "Q5"] == ["accepted"] * N_Q5

    manifest = json.loads((out / "manifest.json").read_text())
    for p in GATE_FILES:
        assert manifest["gate_shas"][p] == _sha_file(p), f"criterion 6: gate sha of {p}"
    got_acc = {k: tuple(v["accepted"]) for k, v in manifest["e_histories"].items()}
    assert {k: set(v) for k, v in got_acc.items()} == {k: set(v) for k, v in acc.items()}
    # N7 (ruling R3; reviewer mutant: the kept miss cut at the first stub, as old's is). The kept COMPACT shares its
    # stubs and step 4 with the same history's old COMPACT, sent earlier, so its predicted miss counts the tokens
    # after the two texts' common prefix. Oracle: the mock's own tokenizer on the bodies the mock received.
    predicted, prompt_tokens = manifest["predicted_miss_tokens"], manifest["prompt_tokens"]
    by_id = {c["request_id"]: c["body"] for c in chats}
    for x in h.E_HISTORIES:
        old_pred, kept_pred = predicted[f"{x.id}/old"], predicted[f"{x.id}/kept"]
        assert isinstance(old_pred, int) and isinstance(kept_pred, int), (x.id, old_pred, kept_pred)
        pt_old = prompt_tokens[f"E/{x.id}/old/COMPACT"]
        pt_kept = prompt_tokens[f"E/{x.id}/kept/COMPACT"]
        # the stub-start cut: the head up to the first stub is the same text in both COMPACT prompts
        stub_cut_kept = pt_kept - (pt_old - old_pred)
        assert 0 < kept_pred < stub_cut_kept, (
            f"{x.id}: kept predicted {kept_pred} is not below its stub-start cut {stub_cut_kept}"
        )
        assert kept_pred < old_pred, f"{x.id}: kept predicted {kept_pred} >= old {old_pred}"
        old_body = by_id[f"t0115a-e-{x.id}-old-COMPACT-r0"]
        kept_body = by_id[f"t0115a-e-{x.id}-kept-COMPACT-r0"]
        old_t = old_body["messages"][-1]["content"]
        kept_t = kept_body["messages"][-1]["content"]
        k = 0
        while k < min(len(old_t), len(kept_t)) and old_t[k] == kept_t[k]:
            k += 1
        assert old_t.index(STUB_5) < k < len(kept_t), (
            f"{x.id}: the two COMPACT texts do not share the stubs"
        )
        head = json.loads(json.dumps(kept_body))
        head["messages"][-1]["content"] = kept_t[:k]
        want = len(eng.encode(_prompt_text(kept_body))) - len(eng.encode(_prompt_text(head)))
        assert kept_pred == want, (
            f"{x.id}: kept predicted {kept_pred}, want {want} (tokens after the common prefix)"
        )

    stored = json.loads((out / "report.json").read_text())
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "report.json is build_report(run_dir)"
    )
    assert stored["E"]["tally"] == EXPECTED_TALLY
    assert stored["E"]["bars"] == EXPECTED_BARS
    outcome = str(stored["E"]["outcome"])
    assert outcome.startswith("elision hurts") and "E2" in outcome and "E3" in outcome, outcome
    assert "E1" not in outcome and "EC" not in outcome, outcome
    assert list(stored["Q5"]["readings"]) == ["accepted"] * N_Q5
    md = (out / "report.md").read_text()
    assert outcome in md and "0/40" in md and "40/40" in md, "criterion 6: report.md lacks the k/n"
    # R1 end to end: every reply's thought spans two canvases and closes; the Q4 step-2 replies are finals
    assert all(len(c["chunks"]) == 2 for c in chats), "the mock did not send two-canvas replies"
    for arm in ("G", "A"):
        assert stored["Q4"][arm]["outcomes"]["final"] == EXPECTED_Q4_FINAL, (
            f"R1/R7: Q4 {arm}: {stored['Q4'][arm]['outcomes']}"
        )


# =====================================================================================================
# Criterion 3: an engine-down void at Q5 stops the harness and keeps the earlier rows
# =====================================================================================================


def test_engine_down_stops_and_keeps_rows(tmp_path: Path) -> None:
    """Criterion 3 (reviewer mutation: Q5 first): the mock's EngineCore dies on the first Q5 request (the
    connection drops without a reply, and from then on every endpoint, ``/health`` included, answers 503). The
    harness sends no second Q5 request, and every earlier row is kept: ``requests.jsonl`` holds the 260 Q4 and E lines
    in schedule order and the Q5 line reads ``engine-down`` (none ``accepted``); ``report.json`` exists with the E
    outcome of the healthy run and an ``engine-down`` Q5 reading. Not a refusal (exit is not 2). Moved here from the
    unit module: it needs a mock engine (criterion 8)."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-loop-requests"
    with (
        serve_115a(accepted_of(h), kill_on_q5=True) as (eng, base_url),
        loopback_only() as attempts,
    ):
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], attempts
    assert rc != 2, "an engine-down void is not a refusal"
    kinds = Counter(c["kind"] for c in eng.chats)
    assert kinds["Q5"] == 1, f"criterion 3: {kinds['Q5']} Q5 requests after the engine died"
    assert kinds == EXPECTED_KINDS - Counter({"Q5": N_Q5 - 1}), kinds
    # R5: ``chats`` records only what a live engine answered; a POST sent after the engine died is answered 503
    # before it is placed, so the count comes from the full request log
    posts = sum(1 for method, path in eng.requests if method == "POST" and path == CHAT_PATH)
    want_posts = sum(EXPECTED_KINDS.values()) - (N_Q5 - 1)
    assert posts == want_posts, (
        f"R5: {posts} chat POSTs, want {want_posts} (the harness kept sending after engine-down)"
    )

    lines = read_lines(out / "requests.jsonl")
    earlier = [x for x in lines if x["row"] != "Q5"]
    want = [tuple(e.get(k) for k in SCHEDULE_KEYS) for e in h.schedule() if e["row"] != "Q5"]
    assert [tuple(x.get(k) for k in SCHEDULE_KEYS) for x in earlier] == want, (
        "the earlier rows were not kept"
    )
    assert lines[: len(earlier)] == earlier, "a Q5 line before the earlier rows"
    q5 = [x["reading"] for x in lines if x["row"] == "Q5"]
    assert q5 and q5[0] == "engine-down" and "accepted" not in q5, q5
    assert tally_of(lines) == EXPECTED_TALLY

    stored = json.loads((out / "report.json").read_text())
    assert stored["E"]["tally"] == EXPECTED_TALLY and stored["E"]["bars"] == EXPECTED_BARS
    assert "engine-down" in list(stored["Q5"]["readings"]), stored["Q5"]


# =====================================================================================================
# Ruling R4: errored requests are their own outcome, never correct, and 3/160 make E inconclusive
# =====================================================================================================

#: one fault in each of three cells: two truncations after the whole reply (the fact included) and one HTTP 500
ERRORED_FAULTS = {
    "t0115a-e-ferry-old-FULL-r0": "truncate",
    "t0115a-e-market-kept-COMPACT-r3": "truncate",
    "t0115a-e-hall-kept-FULL-r5": "500",
}
#: by hand: the healthy tally with each faulted request moved from correct to errored
ERRORED_TALLY = {
    "old": {
        "FULL": {"correct": 39, "re_search": 0, "unsupported": 0, "other": 0, "errored": 1},
        "COMPACT": {"correct": 0, "re_search": 0, "unsupported": 40, "other": 0, "errored": 0},
    },
    "kept": {
        "FULL": {"correct": 39, "re_search": 0, "unsupported": 0, "other": 0, "errored": 1},
        "COMPACT": {"correct": 39, "re_search": 0, "unsupported": 0, "other": 0, "errored": 1},
    },
}


def test_errored_requests_are_errored_and_make_e_inconclusive(tmp_path: Path) -> None:
    """Ruling R4 end to end. Three E requests fail, one per cell, each fact-bearing: two streams send the whole
    reply -- ``The answer is <fact>.`` in a closed thought's wake -- and then stop without a finish_reason (the client
    raises after yielding the fact; adversarial: the commits alone read correct), and one answers HTTP 500. Each line
    reads ``errored``, never correct; the tally (by hand, ``ERRORED_TALLY``) counts them in their cells; 3/160 errored
    makes E "inconclusive" naming the errored requests, though EC holds and E2/E3 fail as in the healthy run. The run
    is complete (3 errors are not 10 in a row), so it exits 0 and Q5 is sent."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-loop-requests"
    with (
        serve_115a(accepted_of(h), faults=ERRORED_FAULTS) as (eng, base_url),
        loopback_only() as attempts,
    ):
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], attempts
    assert sorted(eng.faults_served) == sorted(ERRORED_FAULTS.items()), (
        f"the mock did not serve every fault: {eng.faults_served}"
    )
    assert rc == 0, f"a complete run exits 0; got {rc}"
    lines = read_lines(out / "requests.jsonl")
    by_id = {str(x["request_id"]): x for x in lines}
    for rid in ERRORED_FAULTS:
        assert by_id[rid]["outcome"] == "errored", (
            f"R4: {rid} errored but reads {by_id[rid]['outcome']!r}"
        )
    assert tally_of(lines) == ERRORED_TALLY, tally_of(lines)
    stored = json.loads((out / "report.json").read_text())
    assert stored["E"]["tally"] == ERRORED_TALLY, stored["E"]["tally"]
    outcome = str(stored["E"]["outcome"])
    assert outcome.startswith("inconclusive") and "errored" in outcome.lower(), outcome
    assert list(stored["Q5"]["readings"]) == ["accepted"] * N_Q5
    md = (out / "report.md").read_text()
    assert "errored" in md.lower() and "1/40" in md, "R4: report.md lacks the errored k/n"


# =====================================================================================================
# Q5 "rejected" (R7) and a /health blip that is not engine-down (R8)
# =====================================================================================================


def test_q5_rejected_while_health_answers(tmp_path: Path) -> None:
    """Criterion 3 (R7): every Q5 request answers HTTP 400 and the engine stays up (``/health`` answers), so each
    reads ``rejected`` -- not accepted (the status is not 200), not engine-down (the engine answers) -- all three are
    sent (rejected is not a stop), and the earlier rows are those of the healthy run."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-loop-requests"
    with (
        serve_115a(accepted_of(h), q5_status=400) as (eng, base_url),
        loopback_only() as attempts,
    ):
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], attempts
    assert eng.q5_seen == N_Q5, f"{eng.q5_seen} Q5 requests; rejected is not a stop"
    lines = read_lines(out / "requests.jsonl")
    assert [x["reading"] for x in lines if x["row"] == "Q5"] == ["rejected"] * N_Q5
    stored = json.loads((out / "report.json").read_text())
    assert list(stored["Q5"]["readings"]) == ["rejected"] * N_Q5, stored["Q5"]
    assert stored["E"]["tally"] == EXPECTED_TALLY
    assert rc == 0, f"a complete run exits 0; got {rc}"


def test_q5_health_blip_reads_rejected_not_engine_down(tmp_path: Path) -> None:
    """Ruling R8: Q5 re-checks ``/health`` (up to 3 times, 1 s apart) before reading anything but accepted. The
    first Q5 request answers 400 and the next ``/health`` fails once, then recovers: that reads ``rejected``, not
    engine-down, and the harness goes on to send the other two Q5 requests (also rejected). A single check would
    read engine-down and stop after one."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-loop-requests"
    with (
        serve_115a(accepted_of(h), q5_status=400, health_flaps=1) as (eng, base_url),
        loopback_only() as attempts,
    ):
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], attempts
    assert eng.flaps_served == 1, f"the /health blip was not served ({eng.flaps_served})"
    readings = [x["reading"] for x in read_lines(out / "requests.jsonl") if x["row"] == "Q5"]
    assert readings == ["rejected"] * N_Q5, f"R8: a /health blip read as {readings}"
    assert eng.q5_seen == N_Q5, f"{eng.q5_seen} Q5 requests"
    assert rc == 0, f"a complete run exits 0; got {rc}"


# =====================================================================================================
# Criterion 1: the refusals (T-0115 criterion 2, where they apply), all before any socket
# =====================================================================================================

REFUSALS: dict[str, tuple[str, ...]] = {
    "non-loopback-url": ("loopback",),
    "dirty-tree": ("_t0115a_dirty_probe",),
    "non-empty-out-dir": ("t0115a-out",),
    "unpinned-cases": ("cases.jsonl",),
    "test-split-case": ("search-11",),
    "kit-unloadable": ("kit-unloadable-probe",),
    "repair-off": ("payloads missing (t0098 probe)", "repair"),
    "wrong-tool-set": ("bbfc53ccdd80", "000000000000"),
}


@pytest.mark.parametrize("why", list(REFUSALS))
def test_main_refuses_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criterion 1: each row breaks one condition on an otherwise valid invocation over a clean tree (the healthy row
    is the other direction): a non-loopback ``--base-url``; a dirty tree (naming the path); a non-empty
    ``--out-dir``; the ``cases.jsonl`` pin (``CASES_SHA256``); a test-split case id among ``DEV_CASE_IDS``
    (``search-11`` is a test-split id of the pinned file); a tool kit that will not load, has repair off, or is not
    the ``bbfc53ccdd80`` set. Each: exit 2, the live mock never contacted, no socket at all, nothing written, the
    refusal names its cause. Not applicable from T-0115: the scenario-table sha and ``--max-tool-steps``."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0115a-out"
    probe = REPO / "tests" / "integration" / "_t0115a_dirty_probe.txt"
    with serve_115a(accepted_of(h)) as (eng, base_url):
        argv = ["--base-url", base_url, "--out-dir", str(out)]
        if why == "non-loopback-url":
            argv[1] = "http://10.255.255.1:8001/v1"
        elif why == "non-empty-out-dir":
            out.mkdir()
            (out / "earlier.txt").write_text("an earlier run\n")
        elif why == "unpinned-cases":
            monkeypatch.setattr(h, "CASES_SHA256", "0" * 64)
        elif why == "test-split-case":
            monkeypatch.setattr(h, "DEV_CASE_IDS", (*h.DEV_CASE_IDS, "search-11"))
        elif why in ("kit-unloadable", "repair-off", "wrong-tool-set"):
            kind = {
                "kit-unloadable": "unloadable",
                "repair-off": "repair-off",
                "wrong-tool-set": "wrong",
            }[why]
            monkeypatch.setattr(cli, "load_tool_kit", _kit_loader(kind))
        try:
            if why == "dirty-tree":
                probe.write_text("dirty\n")
            with no_sockets() as attempts:
                rc = h.main(argv)
        finally:
            probe.unlink(missing_ok=True)
    said = capsys.readouterr()
    assert rc == 2, f"criterion 1 ({why}): exit {rc}"
    assert attempts == [], f"criterion 1 ({why}): sockets before the refusal: {attempts}"
    assert eng.requests == [], f"criterion 1 ({why}): the engine was contacted: {eng.requests}"
    left = sorted(p.name for p in out.iterdir()) if out.exists() else []
    assert left in ([], ["earlier.txt"]), f"criterion 1 ({why}): the refusal wrote {left}"
    text = said.out + said.err
    assert any(tok in text for tok in REFUSALS[why]), f"criterion 1 ({why}): {text[-300:]!r}"
