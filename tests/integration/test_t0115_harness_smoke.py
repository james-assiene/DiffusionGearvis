"""T-0115: the loop probe's ``main`` end to end against a mock engine on loopback, scripted per step (CPU only;
``integration``, ``slow``; not in the fast suite).

The harness is loaded lazily by path (``tests.unit.test_t0115_loop_probe.load_harness``) and ``main(argv)`` is
called in-process: running anything under the GPU directory from a shell is gated, and this run spends no GPU time.
One run serves every row (it is cached for the module), because ``main`` has no hook for a cut-down schedule: all
118 turns run, each a fresh ``Session`` with the real ``VllmClient``.

**The mock engine** is T-0093's (``tests/integration/test_t0093_harness_smoke.py``: the greedy encoder over the
pinned tokenizer checked by the real decoder, the stand-in chat renderer that counts ``/metrics`` prefix queries and
whole-32-token-block hits, chat-form ``/tokenize``) with a per-step script instead of per-case answers. It finds the
scenario prompt by its text in the request (the texts come from the harness's ``SCENARIOS``) and the step from the
request itself: step 1 unless ``continue_final_message`` is set, else 1 + the number of ``<|tool_response>`` blocks in
the continued assistant text. Every canvas opens with a short thought. The script:

- S0, and an S6 prompt without maximal rows: a final "Okay." at step 1;
- S1: step 1 speaks "Let me check." and searches; step 2 answers with the prompt's ``fact`` -> 2 generates, 1 executed;
- S2: search, search (no text), then the fact -> 3 generates, 2 executed;
- S3: a click (speaking "I will fill it in."), a ``type_at``, then "Done." -> 3 generates, 2 executed;
- S4: a click, then "Done." -> 2 generates, 1 executed; step 2 is **three canvases**: its thought spans canvases 0
  and 1 (both end inside it, so the client marks their commits ``thought_open``) and closes in canvas 2, which
  answers "Done." -- the one multi-canvas reply, so a step's thought state is read end to end;
- S5, **the repeat loop**: every step clicks (500, 500) -> 8 executed (exact repeats 7), and the 9th generate's
  valid call is cancelled by the budget (R16): ``tool_loop_end == "budget"``, 9 generates;
- S6 prompts with maximal rows (5 x >= 300-char snippets, asserted as a premise), **the heavy turn**: step 1 says
  "Let me compare them." and searches ``libs``; every later step writes 1 200 chars of text, then searches ``libs``.

**The hand-computed over-budget step.** Each embedded step is its text (step 1: the 20-char prefix, at most 100
here; later steps cut to ``MAX_STEP_TEXT`` = 1 000), the call ``<|tool_call>call:google_search{query:<|"|>libs<|"|>}
<tool_call|>`` (12 + 5 + 13 + 1 + 6 + 5 + 4 + 5 + 1 + 12 = 64 chars) and the response block. The rows are maximal, so
the uncapped block is over ``MAX_RESPONSE`` = 1 500 and ``render_response`` caps every string at the largest common
length that fits; one more char of cap adds at most one char to each of the payload's 16 strings (the query and five
title / url / snippet triples), so the block is 1 485 .. 1 500 chars. After m executed steps the continuation is
``lo(m) = 64 + 1485 + (m - 1) * 2549`` .. ``hi(m) = 100 + 64 + 1500 + (m - 1) * 2564`` chars: m = 5 gives at most
11 920 <= 14 000 (step 6 is sent) and m = 6 at least 14 294 > 14 000 (``MAX_CONTINUATION``).

**T-0120b: the heavy turn compacts instead of ending over budget** (re-computed by hand; it no longer ends
``over_budget``). ``ScenarioSearch`` serves the i-th search ``rows[i]`` (one library each), whose rendered blocks are
``R = 1496, 1497, 1496, 1500, 1496, 1497, 1498, 1498`` for i = 0..7 (premise, checked in the row); the step-1 text
is the 20-char prefix, a later step's text 1 000 chars, every call 64 chars, so a later step embeds
``1064 + R[i]`` and an elided search response is the 132-char stub (T-0120b criterion 2; 5 results). Sent
continuations: step 2 1 580; step 3 4 141; step 4 6 701; step 5 9 265; step 6 11 825. Step 7: 14 386 > 14 000,
so ``compact`` (target 9 333, keep the last 2 of 6) elides index 0 (-1 364 -> 13 022), 1 (-1 365 -> 11 657),
2 (-1 364 -> 10 293) and 3 (-1 368 -> 8 925 <= 9 333, stop): sent at 8 925. Step 8: 8 925 + 1 064 + 1 498 =
11 487, fits, no compaction. Step 9: 11 487 + 1 064 + 1 498 = 14 049 > 14 000, so ``compact`` elides index 4
(-1 364 -> 12 685) and 5 (-1 365 -> 11 320; indices 6-7 are the kept tail), then (T-0178 B1) the report step
appends ``REPORT_SEPARATOR + REPORT_FACT`` (2 + 160 = 162 chars): sent at 11 320 + 162 = 11 482, while the
compaction note keeps ``chars_after`` 11 320 (T-0219 re-pin: T-0178 B1 report fact). Step 9's call is the
9th and is cancelled by ``max_tool_steps`` 8 (as S5's): ``tool_loop_end == "budget"``, 8 executed, 9 generates,
two ``tool loop compacted`` notes (step 7 elided [0, 1, 2, 3] 14 386 -> 8 925; step 9 elided [4, 5] 14 049 ->
11 320), ``tool_compactions == 2``, no over-budget note. The result holds for any step-1 text of 0..100 chars (the
nearest edge is step 9's 14 049 against 14 000).

**The expected outcome.** C1 40/40, C2 >= 60, nothing errored; G3 fails (S5's 20 budget turns, 20/100 > 10/100) and
G4 fails (20/20 S5 turns with a call executed >= 3 times > 4/20), while G1, G2 and G5 hold: "not ready" naming at
least G3 and G4. **Exactly** G3 and G4 (B6, recomputed after H-T115-10): the script never reads a response, so
the S3/S4 success responses change no step count; G1 holds (every step >= 2 generate is a final or a valid call:
S1 20 + S2 40 + S3 40 + S4 20 + S5 160 + S6 40 = 320 after T-0120b (was S6 25 = 305), all sane); G2 S4 20/20 and
S1 20/20; G4 exact 0 over S1-S4
(the two searches of S2 and the click/type of S3 differ); G5 20/20 and 20/20; G6 0 (the only thought commit text
is "OK." and S4 step 2's thought pieces, none of which holds a voiced text; histories are the played joins); G7
holds (no 413; the largest step prompt, S6's sixth, is under 14 336 tokens). So the outcome names G3 and G4 only.

**H-T115-10 on the wire.** The mock records each step >= 2 request's continued text: S3's steps 2-3 and S4's step 2
end in a success response (``screen_changed:true``); S5's steps keep ``screen_changed:false``.

**Preconditions, not patched around:** ``main`` refuses a dirty tree, so the rows skip, naming that, when ``git
status`` is not clean; the pinned tokenizer must be in the local HF cache (else skip, naming the file).

Regime: CPU, mock engine on loopback, fake STT/TTS/audio/perception, real ``Session`` and ``VllmClient``; no
measurement -- the numbers it writes describe the mock, never the model.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_astra.toolcall import loop
from local_astra.toolcall.detok import pinned_tokenizer_path
from local_astra.toolcall.loop import REPORT_FACT, REPORT_SEPARATOR
from tests.fakes.vllm_sse import MODEL, StreamSpec, build_lines
from tests.integration.test_t0093_harness_smoke import (
    STOP_TOOL,
    STOP_TURN,
    Engine,
    native_call,
    require_clean_tree,
)
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0115_loop_probe import BARS, SHAPE, load_harness
from tests.unit.test_t0115a_loop_requests import STUB_5

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FIRST_CANVAS_DELAY_S = 0.02
THOUGHT = "<|channel>thought\nOK.<channel|>"
STRAY = "<channel|>"
#: S4 step 2's thought, one piece per canvas: canvases 0 and 1 end inside it, canvas 2 closes it (then "Done.")
SPANNING_THOUGHT = ("<|channel>thought\nThe page", " is open now.", " Fine.<channel|>")
LONG = "".join(f"Library {i:02d} is fast and small. " for i in range(40))
assert len(LONG) == 1200
CLICK_TARGET = {"x": 500, "y": 500}

MAX_CONTINUATION = 14_000
#: B6: the clean mock's failing bars, by hand (module docstring, *The expected outcome*)
EXPECTED_FAILING = {"G3", "G4"}

# --- T-0120b: the compacting heavy S6 turn, by hand (module docstring) -------------------------------------------
COMPACTED_NOTE = "tool loop compacted"
OVER_BUDGET_NOTE = "tool loop switched off: the continuation is over budget"
#: rendered response block of the i-th search (rows[i]) for i = 0..7 (a premise, checked against render_response)
HEAVY_R = (1496, 1497, 1496, 1500, 1496, 1497, 1498, 1498)
#: the continuation sent for steps 2..9 (steps 7 and 9 after compaction)
HEAVY_SENT = {
    2: 1580,
    3: 4141,
    4: 6701,
    5: 9265,
    6: 11825,
    7: 8925,
    8: 11487,
    # T-0219 re-pin: T-0178 B1 report fact (step 9 is the report step; compaction's chars_after stays 11320)
    9: 11320 + len(REPORT_SEPARATOR + REPORT_FACT),
}
#: (step, newly elided indices, chars_before, chars_after) of each compaction note
HEAVY_COMPACTIONS = ((7, [0, 1, 2, 3], 14386, 8925), (9, [4, 5], 14049, 11320))
HEAVY_GENERATES = 9
HEAVY_EXECUTED = 8


def lo(m: int) -> int:
    return 64 + 1485 + (m - 1) * 2549


def hi(m: int) -> int:
    return 100 + 64 + 1500 + (m - 1) * 2564


def maximal(prompt: Any) -> bool:
    return bool(prompt.rows) and all(
        len(rs.results) >= 5 and all(len(r.snippet) >= 300 for r in rs.results)
        for rs in prompt.rows
    )


def _call(name: str, **arguments: Any) -> str:
    return native_call({"name": name, "arguments": arguments})


class LoopEngine(Engine):
    """T-0093's mock engine (counters, renderer, encoder) answering per scenario and step."""

    def __init__(self, scenarios: Any) -> None:
        super().__init__([], ())
        self.by_prompt: list[tuple[str, str, Any]] = sorted(
            (
                (json.dumps(p.text, ensure_ascii=False)[1:-1], s.scenario_id, p)
                for s in scenarios
                for p in s.prompts
            ),
            key=lambda t: len(t[0]),
            reverse=True,
        )
        self.answered: Counter[tuple[str, int]] = Counter()
        #: the continued assistant text of every step >= 2 request, by (scenario, step): what the model was sent
        self.continued: dict[tuple[str, int], list[str]] = {}

    def which(self, body: dict[str, Any]) -> tuple[str | None, Any]:
        messages = json.dumps(body.get("messages"), ensure_ascii=False)
        return next(((sid, p) for text, sid, p in self.by_prompt if text in messages), (None, None))

    @staticmethod
    def continued_text(body: dict[str, Any]) -> str:
        messages = list(body.get("messages") or [])
        content = messages[-1].get("content") if messages else ""
        if isinstance(content, str):
            return content
        return "".join(str(p.get("text", "")) for p in content or [] if isinstance(p, dict))

    def script(self, sid: str, prompt: Any, step: int) -> tuple[str, bool]:
        """(canvas text after the opening thought, ends in a call)."""
        if sid == "S1":
            if step == 1:
                return "Let me check." + STRAY + _call("google_search", query="first query"), True
            return prompt.fact or "Okay.", False
        if sid == "S2":
            if step == 1:
                return "Let me check." + STRAY + _call("google_search", query="first query"), True
            if step == 2:
                return _call("google_search", query="second query"), True
            return prompt.fact or "Okay.", False
        if sid == "S3":
            if step == 1:
                return "I will fill it in." + STRAY + _call("click_at", x=100, y=200), True
            if step == 2:
                return _call("type_at", x=100, y=240, text="Ada", press_enter=False), True
            return "Done.", False
        if sid == "S4":
            if step == 1:
                return "Okay." + STRAY + _call("click_at", x=412, y=300), True
            return "Done.", False
        if sid == "S5":
            if step == 1:
                return "Let me click." + STRAY + _call("click_at", **CLICK_TARGET), True
            return "I keep clicking." + _call("click_at", **CLICK_TARGET), True
        if sid == "S6" and maximal(prompt):
            if step == 1:
                return "Let me compare them." + STRAY + _call("google_search", query="libs"), True
            return LONG + _call("google_search", query="libs"), True
        return "Okay.", False

    def answer(self, body: dict[str, Any]) -> list[int]:
        if body.get("max_tokens") == 1:
            return [STOP_TURN]  # the anticipatory prefill
        step = 1
        if body.get("continue_final_message"):
            text = self.continued_text(body)
            if text.rstrip().endswith("<|tool_call>"):
                return [STOP_TURN]  # a repair: every scripted draft is valid, so none is expected
            step = 1 + text.count("<|tool_response>")
        sid, prompt = self.which(body)
        if sid is None:
            with self.lock:
                self.unknown.append("chat request matching no scenario prompt")
            return [*self.enc.encode(THOUGHT + "Okay."), STOP_TURN]
        with self.lock:
            self.answered[(sid, step)] += 1
            if step >= 2:
                self.continued.setdefault((sid, step), []).append(self.continued_text(body))
        text, is_call = self.script(sid, prompt, step)
        return [*self.enc.encode(THOUGHT + text), STOP_TOOL if is_call else STOP_TURN]

    def canvases(self, body: dict[str, Any]) -> list[list[int]]:
        """The response's canvases (one SSE chunk each). One canvas, except S4's step 2: its thought spans
        canvases 0 and 1 (``SPANNING_THOUGHT``) and closes in canvas 2, which answers "Done."."""
        ids = self.answer(body)
        if body.get("continue_final_message") and body.get("max_tokens") != 1:
            text = self.continued_text(body)
            sid, _ = self.which(body)
            if sid == "S4" and 1 + text.count("<|tool_response>") == 2:
                # the session log shows what the client made of it (``test_a_thought_spanning_canvases...``)
                *heads, tail = SPANNING_THOUGHT
                return [
                    *(self.enc.encode(h) for h in heads),
                    [*self.enc.encode(tail + "Done."), STOP_TURN],
                ]
        return [ids]

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
                    if "messages" in body:
                        engine.tokenize_bodies.append(body)
                        ids = engine.renderer.render(body)
                    else:
                        ids = engine.enc.encode(str(body.get("prompt", "")))
                    reply = {"count": len(ids), "max_model_len": 16384, "tokens": ids}
                    self._send(200, json.dumps(reply).encode(), "application/json")
                    return
                if self.path != "/v1/chat/completions":
                    engine.unknown.append(f"POST {self.path}")
                    self._send(404, b"{}", "application/json")
                    return
                chunks = engine.canvases(body)
                engine.count(body)
                lines = build_lines(StreamSpec(chunks=chunks), engine.enc.detok)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for k, line in enumerate(lines):
                    if k == 1:
                        time.sleep(FIRST_CANVAS_DELAY_S)
                    self.wfile.write(f"{line}\n\n".encode())
                    self.wfile.flush()

        return Handler


# =====================================================================================================
# The one run
# =====================================================================================================

_RUN: dict[str, Any] = {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _do_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    h = load_harness("_t0115_loop_probe_smoke")
    require_clean_tree()
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = LoopEngine(h.SCENARIOS)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0115-mock-engine", daemon=True)
    thread.start()
    out = tmp_path_factory.mktemp("t0115") / "t0115-loop-probe-smoke"
    started = time.perf_counter()
    try:
        with loopback_only() as attempts:
            rc = h.main(
                [
                    "--base-url",
                    f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "--out-dir",
                    str(out),
                ]
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    wall = time.perf_counter() - started
    print(f"\nT-0115 smoke: main -> {rc} in {wall:.1f} s; {len(eng.requests)} engine requests")
    for name in (
        "trials.jsonl",
        "steps.jsonl",
        "manifest.json",
        "report.json",
        "report.md",
        "queries.jsonl",
    ):
        assert (out / name).is_file(), f"criteria 3, 5, 6: the run directory has no {name}"
    return SimpleNamespace(
        h=h,
        rc=rc,
        out=out,
        eng=eng,
        attempts=list(attempts),
        trials=_jsonl(out / "trials.jsonl"),
        steps=_jsonl(out / "steps.jsonl"),
        report=json.loads((out / "report.json").read_text(encoding="utf-8")),
    )


def probe_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The module's one run, done on first use; a failure or skip is replayed to every later row."""
    if "error" in _RUN:
        raise _RUN["error"]
    if "result" not in _RUN:
        try:
            _RUN["result"] = _do_run(tmp_path_factory)
        except BaseException as exc:
            _RUN["error"] = exc
            raise
    result: SimpleNamespace = _RUN["result"]
    return result


def _key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (str(record["scenario"]), str(record["prompt_id"]), int(record["repeat"]))


def _heavy(run: SimpleNamespace) -> set[str]:
    s6 = next(s for s in run.h.SCENARIOS if s.scenario_id == "S6")
    return {p.prompt_id for p in s6.prompts if maximal(p)}


def expected_generates(run: SimpleNamespace) -> Callable[[tuple[str, str, int]], int]:
    per = {"S0": 1, "S1": 2, "S2": 3, "S3": 3, "S4": 2, "S5": 9}
    heavy = _heavy(run)

    def n(key: tuple[str, str, int]) -> int:
        sid, prompt_id, _ = key
        if sid == "S6":
            # T-0120b: the heavy turn compacts and runs to the step budget (module docstring)
            return HEAVY_GENERATES if prompt_id in heavy else 1
        return per[sid]

    return n


# =====================================================================================================
# Rows
# =====================================================================================================


def test_118_trial_lines_and_the_matching_step_lines_in_order(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 5: 118 trial lines in ``schedule()`` order; the step lines are the turns' generates grouped by turn
    in the same order, steps 1..k with k as scripted, each request id one of its turn's own; queries are logged;
    nothing left loopback and the mock saw no request it does not serve. (T-0120b re-scope: k of a heavy S6 turn is
    9, was 6, through ``expected_generates``.)"""
    run = probe_run(tmp_path_factory)
    assert run.attempts == [], f"a socket left loopback: {run.attempts}"
    assert run.eng.unknown == [], f"requests the mock does not serve: {run.eng.unknown[:5]}"
    keys = [_key(t) for t in run.trials]
    assert len(keys) == 118 == len(set(keys)), len(keys)
    assert keys == [tuple(x) for x in run.h.schedule()], "trials.jsonl is not in schedule order"
    assert Counter(k[0] for k in keys) == {sid: p * n for sid, (p, n) in SHAPE.items()}

    groups: list[tuple[tuple[str, str, int], list[int]]] = []
    for s in run.steps:
        if not groups or groups[-1][0] != _key(s):
            groups.append((_key(s), []))
        groups[-1][1].append(int(s["step"]))
    assert [g[0] for g in groups] == keys, "steps.jsonl is not grouped in trial order"
    n = expected_generates(run)
    wrong = [(k, got) for k, got in groups if got != list(range(1, n(k) + 1))]
    assert wrong == [], f"step lines differ from the script: {wrong[:5]}"
    rids = {_key(t): set(t["request_ids"]) for t in run.trials}
    stray = [s["request_id"] for s in run.steps if s["request_id"] not in rids[_key(s)]]
    assert stray == [], stray[:5]
    queries = _jsonl(run.out / "queries.jsonl")
    assert queries, "criterion 3: queries.jsonl is empty though S1, S2 and S6 search"


def test_a_scripted_repeat_loop_ends_budget_and_counts_repeats(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """S5 under the repeat script: every turn executes the same click 8 times, ends ``budget`` after 9 generates
    (the 9th call cancelled by the budget), and counts 7 exact repeats."""
    run = probe_run(tmp_path_factory)
    s5 = [t for t in run.trials if t["scenario"] == "S5"]
    assert len(s5) == 20
    for t in s5:
        tag = _key(t)
        assert t["tool_loop_end"] == "budget", (tag, t["tool_loop_end"])
        assert t["tool_steps_executed"] == 8, (tag, t["tool_steps_executed"])
        assert [c["step"] for c in t["executed"]] == list(range(1, 9)), (tag, t["executed"])
        assert all(
            c["name"] == "click_at" and c["arguments"] == CLICK_TARGET for c in t["executed"]
        ), tag
        assert t["repeats"]["exact"] == 7, (tag, t["repeats"])


def test_a_heavy_s6_compacts_twice_and_ends_at_the_step_budget(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """T-0120b re-scope of the heavy-S6 row (the module docstring's hand computation; it no longer ends over
    budget). Premises: the uncompacted step-7 continuation is over 14 000 (``hi(5) <= MAX_CONTINUATION < lo(6)``, as
    before) and the eight served row sets render to ``HEAVY_R``. Then per heavy turn: ``tool_loop_end == "budget"``,
    8 executed searches (steps 1..8), 9 generates whose sent continuations are ``HEAVY_SENT`` exactly (each <= 14
    000), two ``tool loop compacted`` notes with exactly the hand-computed step / elided / chars_before /
    chars_after, ``tool_compactions == 2`` on the turn line, and no over-budget note. On the wire: steps 7 and 8
    carry the 132-char stub 4 times, step 9 6 times, and step 8's continued text extends step 7's byte for byte
    (prefix stability, T-0120b criterion 3)."""
    run = probe_run(tmp_path_factory)
    heavy = _heavy(run)
    assert heavy, (
        "premise: S6 has a prompt with maximal rows (tests/unit/test_t0115_loop_probe.py pins it)"
    )
    assert hi(5) <= MAX_CONTINUATION < lo(6), (hi(5), lo(6))
    s6 = next(s for s in run.h.SCENARIOS if s.scenario_id == "S6")
    for p in s6.prompts:
        if p.prompt_id in heavy:
            got_r = tuple(
                len(loop.render_response("google_search", loop.search_payload(rows)))
                for rows in p.rows[:HEAVY_EXECUTED]
            )
            assert got_r == HEAVY_R, f"premise: the served blocks render to {got_r}"
    turns = [t for t in run.trials if t["scenario"] == "S6" and t["prompt_id"] in heavy]
    assert turns, "no heavy S6 turn ran"
    by_turn: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for s in run.steps:
        by_turn.setdefault(_key(s), []).append(s)
    for t in turns:
        tag = _key(t)
        assert t["tool_loop_end"] == "budget", (tag, t["tool_loop_end"])
        assert t["tool_steps_executed"] == HEAVY_EXECUTED, (tag, t["tool_steps_executed"])
        assert [c["step"] for c in t["executed"]] == list(range(1, HEAVY_EXECUTED + 1)), tag
        assert all(c["name"] == "google_search" for c in t["executed"]), tag
        assert t["over_budget_step"] is None, (tag, t["over_budget_step"])
        steps = by_turn[tag]
        assert [s["step"] for s in steps] == list(range(1, HEAVY_GENERATES + 1)), (
            tag,
            [s["step"] for s in steps],
        )
        sent = {int(s["step"]): int(s["continuation_chars"]) for s in steps[1:]}
        assert sent == HEAVY_SENT, (tag, sent)
        assert max(sent.values()) <= MAX_CONTINUATION, (tag, sent)
        log = _jsonl(run.out / "sessions" / str(t["session_log"]))
        notes = [r["data"] for r in log if r.get("kind") == "note"]
        compactions = [
            (n["step"], n["elided"], n["chars_before"], n["chars_after"])
            for n in notes
            if n.get("what") == COMPACTED_NOTE
        ]
        assert compactions == [tuple(c) for c in HEAVY_COMPACTIONS], (tag, compactions)
        assert not any(n.get("what") == OVER_BUDGET_NOTE for n in notes), tag
        lines = [r["data"]["extras"] for r in log if r.get("kind") == "turn.record"]
        assert [x.get("tool_compactions") for x in lines] == [2], (tag, lines)
    for step, stubs in ((7, 4), (8, 4), (9, 6)):
        texts = run.eng.continued.get(("S6", step), [])
        assert len(texts) == len(turns), (step, len(texts))
        assert all(x.count(STUB_5) == stubs for x in texts), (
            step,
            [x.count(STUB_5) for x in texts],
        )
    for at7, at8 in zip(run.eng.continued[("S6", 7)], run.eng.continued[("S6", 8)], strict=True):
        assert at8.startswith(at7), "criterion 3: step 8 does not extend step 7's continuation"


def test_a_thought_spanning_canvases_then_closing_is_final(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 5's outcome class end to end (docs/cd/toolcall-loop-probe.md: the ``thought_open`` of the step's
    last non-tool commit decides). Premise, from each S4 turn's own session log: step 2's commits carry
    ``thought_open`` on canvases 0 and 1 and not on the last one. Then its step line is ``final`` with step text
    "Done." (an "any commit thought_open" reduction makes it ``thought-open``), and no thought text was voiced."""
    run = probe_run(tmp_path_factory)
    s4 = [t for t in run.trials if t["scenario"] == "S4"]
    assert len(s4) == 20
    by_turn: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for s in run.steps:
        by_turn.setdefault(_key(s), []).append(s)
    for t in s4:
        tag = _key(t)
        step2 = next(s for s in by_turn[tag] if int(s["step"]) == 2)
        log = _jsonl(run.out / "sessions" / str(t["session_log"]))
        commits = [
            r["data"]
            for r in log
            if r.get("kind") == "llm.commit"
            and r["data"].get("attempt_id") == step2["attempt_id"]
            and r["data"].get("segment") != "tool"
        ]
        open_canvases = sorted({c["canvas_idx"] for c in commits if c.get("thought_open")})
        assert open_canvases == [0, 1] and commits[-1].get("thought_open") is False, (
            f"premise: {tag} step 2 is not a thought over two canvases closed in a third: {commits}"
        )
        assert step2["outcome"] == "final", (tag, step2["outcome"], commits)
        assert step2["step_text"] == "Done.", (tag, step2["step_text"])
        spoken = " ".join(step2["voiced"])
        assert "Done" in spoken and "page" not in spoken and "Fine" not in spoken, (tag, spoken)
        # A2' (orchestrator, 2026-10-17): the step line also records the raw TtsSpeak sources it voiced
        assert "voiced_sources" in step2, (tag, sorted(step2))
        sources = " ".join(step2["voiced_sources"])
        assert "Done" in sources and "page" not in sources and "Fine" not in sources, (tag, sources)


def test_report_json_carries_the_outcome(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Criterion 6: ``report.json`` carries ``outcome``; for this script it is "not ready" naming G3 and G4, with
    the run not void, both controls holding, and G1, G2 and G5 holding; ``main`` did not refuse."""
    run = probe_run(tmp_path_factory)
    report = run.report
    assert "outcome" in report, sorted(report)
    print(
        f"T-0115 smoke: outcome {report['outcome']!r}; bars { {b: report['bars'][b]['holds'] for b in BARS} }"
    )
    assert run.rc in (run.h.EXIT_OK, run.h.EXIT_FAILED), f"main exited {run.rc}"
    assert report["void"] is False, report.get("void")
    controls = {c: report["controls"][c] for c in ("C1", "C2")}
    assert controls == {"C1": True, "C2": True}, controls
    outcome = str(report["outcome"])
    assert outcome.startswith("not ready:"), outcome
    assert {"G3", "G4"} <= set(re.findall(r"\bG[1-7]\b", outcome)), outcome
    for bar in ("G1", "G2", "G5"):
        assert report["bars"][bar]["holds"] is True, (bar, report["bars"][bar])
    assert report["bars"]["G3"]["holds"] is False and report["bars"]["G4"]["holds"] is False
    # B6 (orchestrator, 2026-10-17): the exact failing set of the clean mock, by hand in the module docstring
    assert {b: report["bars"][b]["holds"] for b in BARS} == {
        b: b not in EXPECTED_FAILING for b in BARS
    }, report["bars"]
    assert set(re.findall(r"\bG[1-7]\b", outcome)) == EXPECTED_FAILING, outcome


def _last_response(text: str) -> str:
    blocks = re.findall(r"<\|tool_response>(.*?)<tool_response\|>", text, re.DOTALL)
    assert blocks, f"a step >= 2 request without a tool response: {text[-300:]!r}"
    return str(blocks[-1])


#: (scenario, step, tool) of the step requests whose last response is an action's (the mock's script)
SUCCESS_STEPS = (("S3", 2, "click_at"), ("S3", 3, "type_at"), ("S4", 2, "click_at"))


def test_action_responses_on_the_wire(tmp_path_factory: pytest.TempPathFactory) -> None:
    """H-T115-10 on the wire: every S3 step-2 and step-3 request and every S4 step-2 request carries a success
    response for the action just run (``screen_changed:true``, no ``screen_changed:false``, no ``dry_run``
    status), and every S5 step request still carries ``screen_changed:false`` (the repeat tempter), 20 turns each."""
    run = probe_run(tmp_path_factory)
    for sid, step, tool in SUCCESS_STEPS:
        sent = run.eng.continued.get((sid, step), [])
        assert len(sent) == 20, (sid, step, len(sent))
        for text in sent:
            block = _last_response(text)
            assert block.startswith(f"response:{tool}{{"), (sid, step, block)
            assert "screen_changed:true" in block and "screen_changed:false" not in block, (
                sid,
                step,
                block,
            )
            assert "dry_run" not in block, (sid, step, block)
    for step in range(2, 10):
        sent = run.eng.continued.get(("S5", step), [])
        assert len(sent) == 20, ("S5", step, len(sent))
        for text in sent:
            block = _last_response(text)
            assert block.startswith("response:click_at{") and "screen_changed:false" in block, (
                step,
                block,
            )
