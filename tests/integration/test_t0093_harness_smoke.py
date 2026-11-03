"""T-0093 J16: a smoke run of the harness's ``main`` against a mock engine on loopback (CPU only; ``integration``, ``slow``).

The harness is loaded lazily by path (as ``tests/unit/test_t0093_report.py`` does) and ``main(argv)`` is called
in-process: running anything under the GPU directory from a shell is gated, and this run spends no GPU time. It lives
in ``tests/integration`` (not ``tests/eval``, where T-0046 requires every row to be ``gpu``-marked) and outside the fast
selection; it takes about two minutes.

**The mock engine** is a real HTTP server on ``127.0.0.1:<ephemeral>`` in a thread, so the harness's own transports
(``httpx.AsyncHTTPTransport`` for the session's ``VllmClient``, the unproxied urllib opener for ``/models``,
``/metrics`` and ``/version``) run unpatched, sockets included. It serves:

- ``GET /v1/models`` (the pinned model id), ``GET /version``;
- ``GET /metrics``: the prefix-cache counters, zero running/waiting requests and a ``vllm:cache_config_info`` line
  (``block_size="16"``, reported only). T-0096: **every** chat request, the prefill included, adds len(its rendered
  ids) to the queries and, to the hits, the longest prefix it shares with any earlier request's ids, counted in whole
  32-token blocks (chained block hashes, as the engine's prefix cache keys them) and only within its first len - 1
  ids, so a hit is always a multiple of 32;
- ``POST /v1/chat/completions``: the recorded arm-(b) SSE shape built by ``tests/fakes/vllm_sse.build_lines`` from
  token ids -- one canvas after a fixed 200 ms (the same in both arms, so the first-canvas ratio is not noise);
  tools on (the body carries ``tools``): a tool case answers ``<|channel>thought\\n...<channel|>`` + a sentence + the
  recorded stray ``<channel|>`` + the case's ``reference_call`` as a native call + stop id 50, as the T-0087 streams
  do; every other request answers a sentence + stop id 106. A ``max_tokens = 1`` request (the prefill) gets one
  stop id after ``PREFILL_DELAY_S`` (J12's slow-prefill mode: the session's ``PrefillAbort`` lands mid-prefill);
- ``POST /tokenize``: text form (``prompt``) -> the encoder's ids, as before; chat form (``messages``, T-0096) -> the
  ids of the same stand-in renderer the chat requests are counted with (``Renderer``: the template kwargs, the tools,
  each message's role and parts -- words and punctuation to ids from one shared vocabulary, one fixed 64-id
  placeholder run per image -- its end marker unless it is continued, and the generation prompt unless
  ``add_generation_prompt`` is false; ``add_special_tokens: true`` would prepend a BOS id, as the engine does).

**The ids** come from a greedy longest-match encoder over the pinned ``tokenizer.json`` (special literals first,
byte fallback last), and every encoded text is checked to decode back exactly through the real ``HostDetokenizer``
before it is served -- the oracle is the decoder the client itself uses.

**Script:** search-01 is silent with tools on (J15's THINKING call); the barge-in candidates speak eight words
(2.4 s at the fake TTS's 300 ms/word), long enough for the machine's 600 ms takeover; everything else says "Okay.".

**Preconditions, not patched around** (test-author report, 2026-10-12):

- ``main`` has no hook for a cut-down case list, so this runs all 60 cases x 2 arms (120 sessions);
- ``main`` refuses a dirty tree (by design), so the row skips, naming that, when ``git status`` is not clean;
- the tool kit is the real one (``cli.load_tool_kit``: the pinned 31 MB tokenizer from the local HF cache), which is
  why this is not a unit row; without it the row skips, naming the file.

Regime: CPU, mock engine on loopback, fake STT/TTS/audio/perception, real ``Session`` and ``VllmClient``; no
measurement -- the numbers it writes describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall.detok import HostDetokenizer, pinned_tokenizer_path
from local_astra.toolcall.special_tokens import SPECIAL_IDS
from tests.fakes.vllm_sse import MODEL, StreamSpec, build_lines
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0093_report import GATES, REPO, harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]

CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
FIRST_CANVAS_DELAY_S = 0.2
#: J12 (slow-prefill mode): every prefill takes this long, so the session's end-of-turn ``PrefillAbort`` lands while
#: it runs and the harness must defer it; pass 1 then waits for the prefill in both arms.
PREFILL_DELAY_S = 0.4
STOP_TOOL, STOP_TURN = 50, 106
LONG = "Here is what I am going to do now."


# --- ids --------------------------------------------------------------------------------------------------------


class Encoder:
    """Greedy longest-match over the pinned vocabulary; every result is checked by the real decoder."""

    def __init__(self) -> None:
        tok = json.loads(pinned_tokenizer_path().read_text(encoding="utf-8"))
        self.pieces: dict[str, int] = {}
        self.bytes: dict[int, int] = {}
        for piece, i in tok["model"]["vocab"].items():
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                self.bytes[int(piece[3:5], 16)] = i
            else:
                text = piece.replace("▁", " ")
                if text not in self.pieces or i < self.pieces[text]:
                    self.pieces[text] = i
        self.specials = {lit: i for i, lit in SPECIAL_IDS.items()}
        self.longest = 24
        self.detok = HostDetokenizer.load()

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        pos = 0
        while pos < len(text):
            special = next(
                (
                    lit
                    for lit in sorted(self.specials, key=len, reverse=True)
                    if text.startswith(lit, pos)
                ),
                None,
            )
            if special is not None:
                out.append(self.specials[special])
                pos += len(special)
                continue
            for size in range(min(self.longest, len(text) - pos), 0, -1):
                i = self.pieces.get(text[pos : pos + size])
                if i is not None:
                    out.append(i)
                    pos += size
                    break
            else:
                out += [self.bytes[b] for b in text[pos].encode("utf-8")]
                pos += 1
        assert self.detok.decode(out) == text, f"the encoder does not round-trip {text!r}"
        return out


def native_call(call: dict[str, Any]) -> str:
    def value(v: Any) -> str:
        if isinstance(v, str):
            return f'<|"|>{v}<|"|>'
        return json.dumps(v)

    body = ",".join(f"{k}:{value(v)}" for k, v in call["arguments"].items())
    return f"<|tool_call>call:{call['name']}{{{body}}}<tool_call|>"


# --- the stand-in chat renderer (T-0096) -----------------------------------------------------------------------

BLOCK = 32
IMAGE_RUN = 64
#: marker ids above every word id the vocabulary can hand out in one run
BOS, START, END, IMAGE, TOOLS, KWARGS, EXTRA = (10_000_000 + i for i in range(7))
#: the chat-form ``/tokenize`` body keeps exactly these (criterion 1c)
TOKENIZE_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
    }
)


class Renderer:
    """A deterministic stand-in for the chat template: one shared word vocabulary for the whole run."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.lock = threading.Lock()

    def words(self, text: str) -> list[int]:
        with self.lock:
            return [
                self.vocab.setdefault(w, len(self.vocab))
                for w in re.findall(r"\w+|[^\w\s]|\s+", text)
            ]

    def render(self, body: dict[str, Any]) -> list[int]:
        ids: list[int] = [BOS] if body.get("add_special_tokens") else []
        ids += [
            KWARGS,
            *self.words(json.dumps(body.get("chat_template_kwargs") or {}, sort_keys=True)),
        ]
        if body.get("tools"):
            ids += [TOOLS, *self.words(json.dumps(body["tools"], sort_keys=True))]
        messages = list(body.get("messages") or [])
        continued = bool(body.get("continue_final_message"))
        for i, m in enumerate(messages):
            ids += [START, *self.words(str(m.get("role")))]
            content = m.get("content")
            parts = (
                [{"type": "text", "text": content}] if isinstance(content, str) else content or []
            )
            for part in parts:
                if part.get("type") == "image_url":
                    ids += [IMAGE] * IMAGE_RUN
                else:
                    ids += self.words(str(part.get("text", "")))
            if not (continued and i == len(messages) - 1):
                ids.append(END)
        if body.get("add_generation_prompt", True) and not continued:
            ids += [START, *self.words("model")]
        return ids


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def extra_for(mode: str | None, body: dict[str, Any]) -> bool:
    """Whether the chat-form ``/tokenize`` adds one id to this body. The prefill's prompt ends in a user message with
    no text (its image at most); pass 1's carries the question there (the 4b0bd92 records)."""
    if mode is None:
        return False
    messages = list(body.get("messages") or [])
    if mode == "image":
        return any(p.get("type") == "image_url" for m in messages for p in _parts(m))
    asks = bool(messages) and any(
        p.get("type") == "text" and str(p.get("text", "")).strip() for p in _parts(messages[-1])
    )
    return {"all": True, "prefill": not asks, "generate": asks}[mode]


def lcp(a: list[int], b: list[int]) -> int:
    """The longest common prefix of two id lists, by brute force (the oracle for S)."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


# --- the mock engine ------------------------------------------------------------------------------------------


class Engine:
    """The scripted answers and the counters; ``serve()`` starts the loopback server."""

    def __init__(
        self, cases: list[Any], candidates: tuple[str, ...], *, extra_id: str | None = None
    ) -> None:
        self.enc = Encoder()
        self.renderer = Renderer()
        #: T-0096: the chat-form ``/tokenize`` returns one id too many (the fail-fast row)
        #: ``None``, ``all``, ``prefill`` / ``generate`` (the chat-form body without / with the question text in its last
        #: message), or ``image`` (a body carrying an image part)
        self.extra_id = extra_id
        #: block-hash chains of every earlier chat request's full blocks (the mock's prefix cache)
        self.blocks: set[int] = set()
        #: every chat request in arrival order: request id, prefill or not, rendered ids, query and hit deltas
        self.chats: list[dict[str, Any]] = []
        self.tokenize_bodies: list[dict[str, Any]] = []
        ref = {}
        for line in CASES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                ref[r["id"]] = r.get("reference_call")
        self.by_text: list[tuple[str, Any]] = sorted(
            ((json.dumps(c.text)[1:-1], c) for c in cases), key=lambda t: len(t[0]), reverse=True
        )
        self.ref = ref
        self.candidates = candidates
        self.lock = threading.Lock()
        self.queries = 0
        self.hits = 0
        self.requests: list[tuple[str, str]] = []
        self.unknown: list[str] = []
        self.tool_streams = 0

    def case_of(self, body: dict[str, Any]) -> Any:
        messages = json.dumps(body.get("messages"), ensure_ascii=False)
        return next((case for text, case in self.by_text if text in messages), None)

    def answer(self, body: dict[str, Any]) -> list[int]:
        thought = "<|channel>thought\nOK.<channel|>"
        case = self.case_of(body)
        tools_on = bool(body.get("tools"))
        if body.get("max_tokens") == 1:
            return [STOP_TURN]
        if body.get(
            "continue_final_message"
        ):  # a repair: the drafts here are valid, so none is expected
            call = self.ref.get(case.case_id) if case is not None else None
            if call:
                return [*self.enc.encode(native_call(call)), STOP_TOOL]
            return [STOP_TURN]
        if case is not None and tools_on and case.kind == "tool":
            call = self.ref[case.case_id]
            if case.case_id == "search-01":
                text = thought + native_call(call)  # J15: a silent call in THINKING
            else:
                sentence = LONG if case.case_id in self.candidates else "Okay."
                text = thought + sentence + "<channel|>" + native_call(call)
            with self.lock:
                self.tool_streams += 1
            return [*self.enc.encode(text), STOP_TOOL]
        return [*self.enc.encode(thought + "Okay."), STOP_TURN]

    def count(self, body: dict[str, Any]) -> None:
        """T-0096: the request's queries (its rendered length) and hits (whole cached blocks within len - 1)."""
        ids = self.renderer.render(body)
        with self.lock:
            chain: int | None = None
            hit = 0
            for k in range((len(ids) - 1) // BLOCK):
                chain = hash((chain, tuple(ids[k * BLOCK : (k + 1) * BLOCK])))
                if chain not in self.blocks:
                    break
                hit += BLOCK
            chain = None
            for k in range(len(ids) // BLOCK):
                chain = hash((chain, tuple(ids[k * BLOCK : (k + 1) * BLOCK])))
                self.blocks.add(chain)
            self.queries += len(ids)
            self.hits += hit
            self.chats.append(
                {
                    "request_id": (body.get("vllm_xargs") or {}).get("astra_request_id"),
                    "prefill": body.get("max_tokens") == 1,
                    "ids": ids,
                    "queries": len(ids),
                    "hits": hit,
                }
            )

    def metrics(self) -> str:
        with self.lock:
            q, h = self.queries, self.hits
        return (
            f'vllm:prefix_cache_queries_total{{model_name="{MODEL}"}} {q}\n'
            f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} {h}\n'
            f'vllm:num_requests_running{{model_name="{MODEL}"}} 0\n'
            f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0\n'
            f'vllm:cache_config_info{{block_size="16",cache_dtype="bfloat16",'
            f'enable_prefix_caching="True",model_name="{MODEL}"}} 1.0\n'
        )

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
                if self.path == "/v1/models":
                    self._send(
                        200, json.dumps({"data": [{"id": MODEL}]}).encode(), "application/json"
                    )
                elif self.path == "/metrics":
                    self._send(200, engine.metrics().encode(), "text/plain")
                elif self.path == "/version":
                    self._send(
                        200, json.dumps({"version": "0.29.0-mock"}).encode(), "application/json"
                    )
                else:
                    engine.unknown.append(f"GET {self.path}")
                    self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                engine.requests.append(("POST", self.path))
                body = json.loads(
                    self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}"
                )
                if self.path == "/tokenize":
                    if "messages" in body:  # T-0096: the chat form
                        engine.tokenize_bodies.append(body)
                        ids = engine.renderer.render(body) + (
                            [EXTRA] if extra_for(engine.extra_id, body) else []
                        )
                    else:
                        ids = engine.enc.encode(str(body.get("prompt", "")))
                    reply = {"count": len(ids), "max_model_len": 16384, "tokens": ids}
                    self._send(200, json.dumps(reply).encode(), "application/json")
                    return
                if self.path != "/v1/chat/completions":
                    engine.unknown.append(f"POST {self.path}")
                    self._send(404, b"{}", "application/json")
                    return
                ids = engine.answer(body)
                engine.count(body)
                lines = build_lines(StreamSpec(chunks=[ids]), engine.enc.detok)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for k, line in enumerate(lines):
                    if k == 1:
                        prefill = body.get("max_tokens") == 1
                        time.sleep(PREFILL_DELAY_S if prefill else FIRST_CANVAS_DELAY_S)
                    self.wfile.write(f"{line}\n\n".encode())
                    self.wfile.flush()

        return Handler


@contextlib.contextmanager
def serve(*, extra_id: str | None = None) -> Iterator[tuple[Engine, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    h = harness()
    eng = Engine(h.load_cases(), tuple(h.BARGE_IN_CANDIDATES), extra_id=extra_id)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0093-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def engine() -> Iterator[tuple[Engine, str]]:
    with serve() as served:
        yield served


def require_clean_tree() -> None:
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        pytest.skip(
            f"main refuses a dirty tree by design (EXIT_REFUSED); commit first. Dirty: {dirty[:200]}"
        )


def test_main_runs_end_to_end_against_a_mock_engine(
    engine: tuple[Engine, str], tmp_path: Path
) -> None:
    """J16: ``main`` exits 0 with every gate holding; the run directory is complete; at least one tools-on case is
    dry-run and one barge-in is delivered; no socket leaves loopback.

    T-0096: ``prefix_ids.jsonl`` has 120 lines and its ids are the ones the mock rendered
    for those very requests; the manifest's prefill and pass-1 tokens are the mock's own deltas; no ``integrity:``
    void; B11 holds per case with S the brute-force LCP of the sidecar's ids; the ``/tokenize`` bodies are the chat
    form with exactly the kept keys; the regime carries ``H-T96``, 32 and the engine's label. J12 still holds."""
    require_clean_tree()
    eng, base_url = engine
    h = harness()
    out = tmp_path / "t0093-live-smoke"
    started = time.perf_counter()
    with loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    wall = time.perf_counter() - started
    print(f"\nJ16 smoke: main -> {rc} in {wall:.1f} s; {len(eng.requests)} engine requests")

    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"the harness called endpoints the mock does not serve: {eng.unknown}"
    for name in ("turns.jsonl", "manifest.json", "report.json", "report.md", "prefix_ids.jsonl"):
        assert (out / name).is_file(), f"J16 / T-0096 criterion 1d: the run directory has no {name}"
    assert any((out / "sessions").glob("*.jsonl")), "J16: no session log was written"

    report = h.build_report(out)
    g = h.gates(report)
    assert set(g) == GATES, g
    failing = sorted(k for k, v in g.items() if v is not True)
    assert report["void"] == [] and failing == [], (
        f"J16: void {report['void'][:5]}, failing gates {failing}"
    )
    assert rc == h.EXIT_OK == 0, f"J16: main exited {rc}"

    manifest = json.loads((out / "manifest.json").read_text())
    cases = manifest["cases"]
    assert len(cases) == 120 and len(manifest["sessions"]) == 120, (
        len(cases),
        len(manifest["sessions"]),
    )
    assert not [c["case_id"] for c in cases if c.get("timed_out") or c.get("failures")], [
        (c["case_id"], c["arm"], c.get("failures"))
        for c in cases
        if c.get("timed_out") or c.get("failures")
    ]
    lines = [json.loads(x) for x in (out / "turns.jsonl").read_text().splitlines() if x.strip()]
    on = {c["session_id"] for c in cases if c["arm"] == "on"}
    dry = [
        x
        for x in lines
        if x["session_id"] in on and (x.get("tool") or {}).get("status") == "dry_run"
    ]
    assert dry, "J16: no tools-on case executed a dry run"
    barged = [c["case_id"] for c in cases if c.get("barge_in")]
    assert barged, "J16: no barge-in was delivered"
    print(
        f"J16 smoke: {len(dry)} dry runs, barge-ins delivered on {barged}, {eng.tool_streams} tool streams"
    )

    # --- J12, slow-prefill mode: the prefill completes, its abort is deferred, one request at a time, both arms --
    by_arm = {arm: [c for c in cases if c["arm"] == arm] for arm in ("on", "off")}
    deferred = {
        arm: sum(1 for c in rows if (c.get("deferred_prefill_aborts") or 0) >= 1)
        for arm, rows in by_arm.items()
    }
    waits = report["prefill_wait_ms"]
    print(f"J12 smoke: deferred aborts per arm {deferred}; prefill_wait_ms {waits}")
    not_done = [(c["case_id"], c["arm"]) for c in cases if c.get("prefill_done") is not True]
    assert not_done == [], f"J12: prefills that did not complete: {not_done[:5]}"
    crowded = [
        (c["case_id"], c["arm"], c.get("max_inflight")) for c in cases if c.get("max_inflight") != 1
    ]
    assert crowded == [], f"J12/J19: cases with other than one request in flight: {crowded[:5]}"
    assert deferred == {"on": 60, "off": 60}, (
        f"J12: every case's PrefillAbort lands mid-prefill and is deferred, in both arms; {deferred}"
    )
    floor = PREFILL_DELAY_S * 1000 / 2
    assert waits["on"]["p50"] >= floor and waits["off"]["p50"] >= floor, (
        f"J12: pass 1 waited for its prefill: {waits}"
    )
    assert abs(waits["on"]["p50"] - waits["off"]["p50"]) <= 100, (
        f"J12: both arms wait alike: {waits}"
    )

    # --- T-0096: what the harness recorded, against what the mock saw ----------------------------------------
    side = [json.loads(x) for x in (out / "prefix_ids.jsonl").read_text().splitlines() if x.strip()]
    assert len(side) == 120, (
        f"criterion 1d: one prefix_ids.jsonl line per (case, arm); got {len(side)}"
    )
    keys = {
        "case_id",
        "arm",
        "session_id",
        "prefill_request_id",
        "generate_request_id",
        "prefill_ids",
        "generate_ids",
    }
    assert all(keys <= set(r) for r in side), sorted(set(side[0]))
    regime = manifest["regime"]
    assert (regime.get("prefix_rule"), regime.get("prefix_block_tokens")) == ("H-T96", 32), regime
    assert str(regime.get("cache_config_block_size")) == "16", regime
    assert (report.get("prefix_rule"), report.get("prefix_block_tokens")) == ("H-T96", 32)
    integrity = [v for v in report["void"] if str(v).startswith("integrity:")]
    assert integrity == [], integrity

    meta = {(c["case_id"], c["arm"]): c for c in cases}
    assert [c for c in cases if c.get("prefills_before_pass1") != 1] == []
    pos = 0
    for row in side:
        found = []
        for rid, prefill in (
            (row["prefill_request_id"], True),
            (row["generate_request_id"], False),
        ):
            while pos < len(eng.chats) and not (
                eng.chats[pos]["request_id"] == rid and eng.chats[pos]["prefill"] is prefill
            ):
                pos += 1
            assert pos < len(eng.chats), f"criterion 1d: no {rid} request for {row['case_id']}"
            found.append(eng.chats[pos])
            pos += 1
        pre, gen = found
        c = meta[(row["case_id"], row["arm"])]
        tag = f"{row['case_id']} ({row['arm']})"
        assert row["prefill_ids"] == pre["ids"], (
            f"criterion 1c: {tag}'s prefill ids are not the engine's"
        )
        assert row["generate_ids"] == gen["ids"], (
            f"criterion 1c: {tag}'s pass-1 ids are not the engine's"
        )
        assert (c["prefill_query_tokens"], c["prefill_hit_tokens"]) == (
            pre["queries"],
            pre["hits"],
        ), f"criterion 1a: {tag}"
        assert (c["prefix_query_tokens"], c["prefix_hit_tokens"]) == (
            gen["queries"],
            gen["hits"],
        ), f"J2 numbers: {tag}"

    rows = {r["case_id"]: r for r in report["prefix_b11_cases"]["on"]}
    for row in (r for r in side if r["arm"] == "on"):
        s = lcp(row["prefill_ids"], row["generate_ids"])
        b11 = rows[row["case_id"]]
        assert b11["S"] == s and b11["holds"] is True, (row["case_id"], s, b11)

    assert len(eng.tokenize_bodies) == 240, len(eng.tokenize_bodies)
    for body in eng.tokenize_bodies:
        assert set(body) <= TOKENIZE_KEYS and {"model", "messages"} <= set(body), sorted(body)
        assert body.get("add_special_tokens") is False, body.get("add_special_tokens")


def test_main_fails_fast_when_tokenize_disagrees(tmp_path: Path) -> None:
    """T-0096 criterion 6: the mock's chat-form ``/tokenize`` returns one id more than the engine counted, so case 0's
    prefill reads P + 1 ids against P queried (5a). ``main`` stops after case 0 with exit 1 and writes a void report
    whose reason names case 0 and both numbers. "After that case" is read as: nothing of order 1 or later ran."""
    require_clean_tree()
    h = harness()
    first = h.load_cases()[0].case_id
    out = tmp_path / "t0093-live-failfast"
    with serve(extra_id="all") as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert rc == h.EXIT_FAILED == 1, f"criterion 6: exit {rc}"
    manifest = json.loads((out / "manifest.json").read_text())
    orders = sorted({c.get("order") for c in manifest["cases"]})
    assert orders == [0] and 1 <= len(manifest["cases"]) <= 2, (orders, len(manifest["cases"]))
    assert len([c for c in eng.chats if c["prefill"]]) <= 2, "criterion 6: cases ran after case 0"
    for name in ("report.json", "report.md"):
        assert (out / name).is_file(), f"criterion 6: no {name}"
    report = json.loads((out / "report.json").read_text())
    p = next(c["queries"] for c in eng.chats if c["prefill"])
    naming = [
        str(v)
        for v in report["void"]
        if first in str(v) and str(p) in str(v) and str(p + 1) in str(v)
    ]
    assert naming and naming[0].startswith("integrity:"), (first, p, report["void"])
    assert h.gates(report)["run_valid"] is False


def _fail_fast_run(
    tmp_path: Path, mode: str
) -> tuple[Engine, Any, int, dict[str, Any], dict[str, Any]]:
    require_clean_tree()
    h = harness()
    out = tmp_path / f"t0093-live-failfast-{mode}"
    with serve(extra_id=mode) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    for name in ("manifest.json", "report.json", "report.md"):
        assert (out / name).is_file(), f"criterion 6: no {name}"
    manifest = json.loads((out / "manifest.json").read_text())
    report = json.loads((out / "report.json").read_text())
    return eng, h, rc, manifest, report


def _mismatch(reasons: list[Any], case_id: str, n: int) -> list[str]:
    """Void reasons naming ``case_id`` with both ``n`` and ``n + 1`` as whole numbers."""
    out = []
    for v in map(str, reasons):
        nums = set(re.findall(r"\d+", v))
        if case_id in v and {str(n), str(n + 1)} <= nums:
            out.append(v)
    return out


@pytest.mark.parametrize("which", ["prefill", "generate"])
def test_fail_fast_names_the_call_that_disagreed(tmp_path: Path, which: str) -> None:
    """T-0096 criterion 6 with 5a vs 5b (ruling M6's test gap): only one of case 0's two ``/tokenize`` calls returns
    an extra id. ``main`` stops after case 0 with exit 1; the void report names the disagreeing request's numbers
    (P and P + 1 for the prefill, Q and Q + 1 for pass 1, P and Q from the mock's own counts) and not the other's."""
    eng, h, rc, manifest, report = _fail_fast_run(tmp_path, which)
    first = h.load_cases()[0].case_id
    assert rc == h.EXIT_FAILED == 1, f"criterion 6: exit {rc}"
    assert sorted({c.get("order") for c in manifest["cases"]}) == [0], manifest["cases"]
    pre = next(c for c in eng.chats if c["prefill"])
    gen = next(c for c in eng.chats if not c["prefill"])
    p, q = pre["queries"], gen["queries"]
    assert p != q, f"fixture: P and Q must differ to tell the reasons apart ({p})"
    bad, good = (p, q) if which == "prefill" else (q, p)
    named = _mismatch(report["void"], first, bad)
    assert named and named[0].startswith("integrity:"), (which, bad, report["void"])
    assert _mismatch(report["void"], first, good) == [], (which, good, report["void"])
    assert h.gates(report)["run_valid"] is False


def test_fail_fast_stops_at_the_first_image_case(tmp_path: Path) -> None:
    """T-0096 ruling M6 (6'): the chat-form ``/tokenize`` disagrees only on prompts that carry an image (image
    expansion), so case 0 and every text-only case before the first image case verify. ``main`` stops after that
    case (in run order: the first case with a screen), runs no later case, exits 1 and writes a void report naming
    it. Case 0 here has no image, so a case-0-only fail-fast would run all 60 cases."""
    eng, h, rc, manifest, report = _fail_fast_run(tmp_path, "image")
    cases = h.load_cases()
    assert cases[0].screen is None, f"fixture: case 0 ({cases[0].case_id}) carries an image"
    k = next(i for i, c in enumerate(cases) if c.screen is not None)
    first_image = cases[k].case_id
    assert rc == h.EXIT_FAILED == 1, f"ruling M6: exit {rc}"
    orders = sorted({c.get("order") for c in manifest["cases"]})
    assert orders == list(range(k + 1)), (
        f"ruling M6: stop at order {k} ({first_image}); ran {orders}"
    )
    assert len([c for c in eng.chats if c["prefill"]]) <= 2 * (k + 1), "a later case ran"
    named = [str(v) for v in report["void"] if first_image in str(v)]
    assert named and named[0].startswith("integrity:"), (first_image, report["void"])
    earlier = [str(v) for v in report["void"] if any(c.case_id in str(v) for c in cases[:k])]
    assert earlier == [], f"the text-only cases before {first_image} verify: {earlier}"
    assert h.gates(report)["run_valid"] is False
