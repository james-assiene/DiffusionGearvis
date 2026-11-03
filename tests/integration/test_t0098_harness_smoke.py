"""T-0098: the A/B harness's ``main`` end to end against a mock engine on loopback, and every refusal of criterion 1
(CPU only; ``integration``, ``slow``). ``pending(task="T-0098")`` until ``/implement T-0098`` starts.

The harness is loaded lazily by path (``tests.unit.test_t0098_prompt_ab.harness``) and ``main(argv)`` is called
in-process: running anything under the GPU directory from a shell is gated, and this run spends no GPU time.

**The mock engine** (modelled on T-0093's smoke, whose ``Encoder``, ``native_call`` and ``require_clean_tree`` are
imported read-only) is a real HTTP server on ``127.0.0.1:<ephemeral>`` in a thread, so the harness's real
``VllmClient`` and its unproxied transport run unpatched. It serves ``GET /v1/models`` (the pinned model id),
``GET /version``, ``GET /metrics`` (zero counters), ``POST /tokenize`` (text form), and ``POST /v1/chat/completions``:
the recorded arm-(b) SSE shape built by ``tests/fakes/vllm_sse.build_lines`` from token ids. Each request's arm is
read from its ``messages[0]`` text against the six texts built by hand in the unit module (an unknown system text is
recorded and fails the row), its case from the user text. The script:

- a tool case under **A0 or C2** whose tool is ``type_at``: SILENT -- the plan sentence inside a closed thought, then
  the case's ``reference_call`` as a native call, stop id 50 (the 4b0bd92 shape);
- every other tool case, every arm: SPEAKS -- a closed thought, ``I will do that.``, the recorded stray
  ``<channel|>``, the call, stop id 50 (the dev type-01 shape);
- no-tool and plain cases: a closed thought and ``Okay.``, stop id 106.

**Expected, by hand:** A0 ``type_at`` spoken 0/50 (R1 holds), A1 50/50 (R2 holds); C2 ``type_at`` 0/50, so C2 fails
P1 and passes P2-P7 (every other number equals A0's); C1 passes all; the winner is **C1**. D1 and D2 also pass all
and are never selected (N2). No errors, so no arm is void.

**Assumptions the task leaves open** (named in the test-author report): the split is chosen by ``--split`` (default
``dev``); a complete, non-void run exits 0 whatever the winner; ``trials.jsonl`` is written in request order; the
cases are pinned by a module constant ``CASES_SHA256`` that ``main`` checks (the unpinned-file row overrides it,
since the pinned file itself may not be edited); the rotation offset ``(case_index + repeat) mod 6`` indexes the arm
order ``A0 A1 D1 D2 C2 C1`` of the task's table, ``case_index`` being the case's position in ``load_cases()``, and
``order`` is the arm's slot (0-5) in its block; the body comparison ignores ``vllm_xargs.astra_request_id``, which is
per request by construction.

Regime: CPU, mock engine on loopback, real ``PromptBuilder`` and ``VllmClient`` (id mode, the pinned detokeniser);
no measurement -- the numbers it writes describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import threading
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.fakes.vllm_sse import MODEL, StreamSpec, build_lines
from tests.integration.test_t0093_harness_smoke import Encoder, native_call, require_clean_tree
from tests.unit._pre_t0155_persona import pre_t0155_fixture
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0098_prompt_ab import (
    A0_SHA256,
    ARMS,
    CANDIDATES,
    PS,
    REPEATS,
    REPO,
    expected_arm_texts,
    harness,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit (the
#: pre-T-0155 texts patched into session, twopass and, through session, each harness loaded by path inside a row).
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
STOP_TOOL, STOP_TURN = 50, 106
SILENT_ARMS = frozenset({"A0", "C2"})
TRIAL_KEYS = {
    "case_id",
    "arm",
    "repeat",
    "order",
    "system_sha256",
    "commits",
    "token_ids",
    "error",
    "wall_ms",
    "first_canvas_ms",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(p.get("text", "")) for p in content or [] if p.get("type") == "text")


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_has_key(v, key) for v in value)
    return False


def _outside_system(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``body`` without ``messages[0]`` and without the per-request ``vllm_xargs.astra_request_id``."""
    out: dict[str, Any] = json.loads(json.dumps(body))
    out["messages"] = out["messages"][1:]
    (out.get("vllm_xargs") or {}).pop("astra_request_id", None)
    return out


class Engine:
    """The scripted answers and every chat request, in arrival order."""

    def __init__(self) -> None:
        self.enc = Encoder()
        self.cache: dict[str, list[int]] = {}
        rows = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.rows = {r["id"]: r for r in rows}
        self.test_texts = {r["user"] for r in rows if r.get("split") == "test"}
        self.by_text: list[tuple[str, dict[str, Any]]] = sorted(
            ((r["user"], r) for r in rows if r.get("split") == "dev"),
            key=lambda t: len(t[0]),
            reverse=True,
        )
        from tests.unit.test_t0098_prompt_ab import PLAIN_QUESTIONS

        self.by_text += [(q, {"id": cid, "kind": "plain"}) for cid, q in PLAIN_QUESTIONS]
        self.arm_of = {text: arm for arm, text in expected_arm_texts().items()}
        self.lock = threading.Lock()
        self.chats: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []
        self.unknown: list[str] = []

    def encode(self, text: str) -> list[int]:
        if text not in self.cache:
            self.cache[text] = self.enc.encode(text)
        return self.cache[text]

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        system = _text_of(messages[0].get("content")) if messages else ""
        arm = self.arm_of.get(system)
        user = _text_of(messages[-1].get("content")) if messages else ""
        row = next((r for text, r in self.by_text if text in user), None)
        if arm is None or row is None:
            self.unknown.append(
                f"arm={arm} case={row and row['id']} system_sha={_sha(system)[:12]}"
            )
        thought = "<|channel>thought\nOK.<channel|>"
        if row is not None and row["kind"] == "tool":
            call = native_call(row["reference_call"])
            if arm in SILENT_ARMS and row["tool"] == "type_at":
                text = "<|channel>thought\nI will use type_at with press_enter.<channel|>" + call
            else:
                text = thought + "I will do that.<channel|>" + call
            ids = [*self.encode(text), STOP_TOOL]
        else:
            ids = [*self.encode(thought + "Okay."), STOP_TURN]
        return {
            "body": body,
            "arm": arm,
            "case_id": row["id"] if row else None,
            "system": system,
            "user": user,
            "ids": ids,
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
                if self.path == "/v1/models":
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
                if self.path == "/tokenize":
                    ids = engine.encode(str(body.get("prompt", "")))
                    reply = {"count": len(ids), "max_model_len": 16384, "tokens": ids}
                    self._send(200, json.dumps(reply).encode(), "application/json")
                    return
                if self.path != "/v1/chat/completions":
                    self._send(404, b"{}", "application/json")
                    return
                chat = engine.answer(body)
                with engine.lock:
                    engine.chats.append(chat)
                lines = build_lines(StreamSpec(chunks=[chat["ids"]]), engine.enc.detok)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for line in lines:
                    self.wfile.write(f"{line}\n\n".encode())
                    self.wfile.flush()

        return Handler


@contextlib.contextmanager
def serve() -> Iterator[tuple[Engine, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0098-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def no_sockets() -> Iterator[list[str]]:
    """Every outbound socket attempt, loopback included, is recorded and refused (criterion 1: "before any socket")."""
    attempts: list[str] = []
    mp = pytest.MonkeyPatch()

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{what} {args[1:] if what.startswith('socket.') else args}")
            raise AssertionError(f"criterion 1: a {what} before the refusal")

        return fn

    mp.setattr(socket.socket, "connect", refuse("socket.connect"))
    mp.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    mp.setattr(socket, "create_connection", refuse("create_connection"))
    mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    try:
        yield attempts
    finally:
        mp.undo()


# =====================================================================================================
# The healthy run
# =====================================================================================================


def test_main_runs_end_to_end_against_a_mock_engine(tmp_path: Path) -> None:
    """Criteria 2-4, 6, 7 and 10 on one complete run (1590 requests): exit 0; every request is pass 1 with the tools,
    no prefill, no seed, no test-split case; each (case, repeat) block sends the six arms in the rotated order and its
    bodies differ only in ``messages[0]``; ``trials.jsonl`` mirrors the requests line for line, raw ids included; the
    manifest carries the six hand-built texts and shas; the report reads the script's decision (C1)."""
    h = harness()
    require_clean_tree()
    out = tmp_path / "t0098-ab"
    with serve() as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out), "--split", "dev"])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    # --- the wire (criteria 2, 3, 10) -----------------------------------------------------------------------
    chats = eng.chats
    assert len(chats) == 53 * 6 * REPEATS == 1590, f"criterion 3: {len(chats)} chat requests"
    for c in chats:
        b = c["body"]
        assert b.get("max_tokens") != 1, "criterion 2: an anticipatory prefill was sent"
        assert not _has_key(b, "seed"), "criterion 3: a seed field was sent"
        assert not b.get("continue_final_message"), "pass 1 only: a continuation was sent"
        assert b.get("tool_choice") == "none" and b.get("tools"), "all tools on (bbfc53ccdd80)"
        assert c["user"] not in eng.test_texts and not any(
            t in c["user"] for t in eng.test_texts
        ), "criterion 10: a test-split case was sent"
    assert {_sha(c["system"]) for c in chats if c["arm"] == "A0"} == {A0_SHA256}

    order = [c.case_id for c in h.load_cases()]
    assert len(order) == 53
    seen: Counter[str] = Counter()
    for k in range(0, len(chats), 6):
        block = chats[k : k + 6]
        cid = block[0]["case_id"]
        assert {b["case_id"] for b in block} == {cid}, f"block {k // 6} mixes cases"
        repeat = seen[cid]
        seen[cid] += 1
        off = (order.index(cid) + repeat) % 6
        want = [ARMS[(off + j) % 6] for j in range(6)]
        assert [b["arm"] for b in block] == want, f"criterion 3: {cid} repeat {repeat}"
        ref = _outside_system(block[0]["body"])
        for b in block:
            body = _outside_system(b["body"])
            assert body == ref, f"criterion 2: {cid} {b['arm']} differs outside messages[0]"
    assert seen == Counter({cid: REPEATS for cid in order}), "criterion 3: 5 blocks per case"

    # --- the raw record (criterion 4) ---------------------------------------------------------------------
    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == 1590
    repeats: Counter[str] = Counter()
    for k, (t, c) in enumerate(zip(trials, chats, strict=True)):
        assert set(t) >= TRIAL_KEYS, f"criterion 4: line {k} lacks {sorted(TRIAL_KEYS - set(t))}"
        assert (t["case_id"], t["arm"]) == (c["case_id"], c["arm"]), f"line {k} is not request {k}"
        assert t["order"] == k % 6 and t["repeat"] == repeats[t["case_id"]] // 6, (
            k,
            t["order"],
            t["repeat"],
        )
        repeats[t["case_id"]] += 1
        assert t["system_sha256"] == _sha(c["system"])
        assert t["token_ids"] == c["ids"], f"criterion 4: line {k}'s token_ids are not the engine's"
        assert t["error"] in (None, ""), t["error"]
        assert 0 <= t["first_canvas_ms"] <= t["wall_ms"], (t["first_canvas_ms"], t["wall_ms"])
        assert t["commits"], f"line {k} recorded no commit"
        for m in t["commits"]:
            assert {"segment", "text", "thought_open", "canvas_idx"} <= set(m), m
    manifest = json.loads((out / "manifest.json").read_text())
    texts = expected_arm_texts()
    got = {a: (v["text"], v["sha256"]) for a, v in manifest["arms"].items()}
    assert got == {a: (t, _sha(t)) for a, t in texts.items()}, (
        "criterion 4: the manifest's arm texts"
    )
    assert manifest.get("regime"), "criterion 4: the manifest holds the regime"

    # --- the report and the decision (criteria 6, 7) ---------------------------------------------------------
    for name in ("report.json", "report.md"):
        assert (out / name).is_file(), f"criterion 6: no {name}"
    stored = json.loads((out / "report.json").read_text())
    rebuilt = h.build_report(out)
    assert json.loads(json.dumps(rebuilt)) == stored, (
        "criterion 6: report.json is build_report(run_dir)"
    )
    assert stored["readings"]["R1"] is True and stored["readings"]["R2"] is True
    assert list(stored["void_arms"]) == []
    assert stored["arms"]["A0"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert stored["arms"]["A1"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    assert stored["arms"]["C2"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert stored["arms"]["C1"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    c2, c1 = stored["candidates"]["C2"], stored["candidates"]["C1"]
    assert [p for p in PS if c2[p] is not True] == ["P1"], {p: c2[p] for p in PS}
    assert [p for p in PS if c1[p] is not True] == [] and c1["passes"] is True
    assert set(stored["candidates"]) == set(CANDIDATES), "N2: only C2 and C1 are candidates"
    assert stored["winner"] == "C1"


# =====================================================================================================
# Criterion 1: the refusals, all before any socket
# =====================================================================================================


def _kit_loader(kind: str) -> Any:
    from local_astra import cli
    from tests.unit._tools_switch_support import make_kit

    def load(profile: Any) -> Any:
        if kind == "unloadable":
            raise cli.ToolsUnavailable("kit-unloadable-probe: cannot load the host detokeniser")
        if kind == "repair-off":
            return make_kit(repair_off="payloads missing (t0098 probe)"), object()
        return make_kit(tools_version="0" * 64), object()

    return load


REFUSALS = {
    "non-loopback-url": "loopback",
    "dirty-tree": "_t0098_dirty_probe",
    "non-empty-out-dir": "t0098-out",
    "unpinned-cases": "cases.jsonl",
    "test-split": "split",
    "kit-unloadable": "kit-unloadable-probe",
    "repair-off": "payloads missing (t0098 probe)",
    "wrong-tool-set": "bbfc53ccdd80",
    "a0-sha-drift": "3605c012",
}


@pytest.mark.parametrize("why", list(REFUSALS))
def test_main_refuses_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criterion 1: exit 2, nothing sent to the (live) mock, no socket opened at all, no run directory written, and the
    refusal names its cause. Each row breaks exactly one condition on an otherwise valid invocation over a clean tree
    (the healthy row above is the other direction: the same invocation, unbroken, runs). The A0 row changes
    ``SPOKEN_STYLE`` by one sentence, so the builder's A0 text no longer hashes to ``3605c012...``."""
    h = harness()
    require_clean_tree()
    from local_astra import cli

    out = tmp_path / "t0098-out"
    probe = REPO / "tests" / "integration" / "_t0098_dirty_probe.txt"
    with serve() as (eng, base_url):
        argv = ["--base-url", base_url, "--out-dir", str(out)]
        if why == "non-loopback-url":
            argv[1] = "http://10.255.255.1:8001/v1"
        elif why == "non-empty-out-dir":
            out.mkdir()
            (out / "earlier.txt").write_text("an earlier run\n")
        elif why == "unpinned-cases":
            monkeypatch.setattr(h, "CASES_SHA256", "0" * 64)
        elif why == "test-split":
            argv += ["--split", "test"]
        elif why == "kit-unloadable":
            monkeypatch.setattr(cli, "load_tool_kit", _kit_loader("unloadable"))
        elif why == "repair-off":
            monkeypatch.setattr(cli, "load_tool_kit", _kit_loader("repair-off"))
        elif why == "wrong-tool-set":
            monkeypatch.setattr(cli, "load_tool_kit", _kit_loader("wrong"))
        elif why == "a0-sha-drift":
            monkeypatch.setattr(pb, "SPOKEN_STYLE", pb.SPOKEN_STYLE + " Keep it short.")
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
    token = REFUSALS[why]
    text = said.out + said.err
    assert token in text or (why == "wrong-tool-set" and "000000000000" in text), (
        f"criterion 1 ({why}): the refusal does not name its cause ({token!r}): {text[-300:]!r}"
    )
