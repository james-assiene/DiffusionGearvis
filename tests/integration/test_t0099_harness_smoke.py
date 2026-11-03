"""T-0099: a smoke run of the repair harness's ``main`` against a scripted mock engine on loopback (CPU; plumbing only).

The harness is loaded lazily by path (``tests/unit/test_t0099_report.harness``) and ``main(argv)`` is called
in-process: running anything under the GPU directory from a shell is gated, and this spends no GPU time. **No number
this module sees describes the model** (implementation notes: corrupting drafts in a mock is plumbing only, never in
the doc's numbers).

**The cut-down run.** ``main`` runs the stress set (``STRESS_PATH`` checked against ``STRESS_SHA256``, both read at
call time) k = ``K`` times, then arm F (``load_forced_cases()``). This module points ``STRESS_PATH``/``STRESS_SHA256``
at a 3-case fixture copied from the real stress rows and narrows arm F to the first two dev tool cases of the
harness's own ``load_forced_cases()``, with ``monkeypatch`` (the seam is stated in ``tests/unit/test_t0099_report.py``).

**The mock engine** (reuses T-0093's ``Encoder``/``native_call`` by import; a real HTTP server on 127.0.0.1) serves
``/v1/models``, ``/version``, ``/metrics`` (prefix counters, nothing running), ``/tokenize`` (text form) and
``/v1/chat/completions`` with scripted native drafts per case and scripted answers to the repair (a request with
``continue_final_message``):

- ``kout-02`` (K-out, "Ctrl+S"): pass 1 drafts ``press_key{combo: Control+C}`` (schema-invalid but an alias of a list
  entry, so repairable before and after T-0100); the repair answers, by repetition, rep 0 a valid
  ``press_key{combo: Ctrl+C}`` (coercion: dry_run), rep 1 the invalid F5 (``repair_failed``: final ``refused``), rep 2
  nothing for ``REPAIR_DELAY_S`` > 3 s (``repair_timeout``: final ``cancelled``). Until 0756893 this row was kout-01
  drafting F5; T-0100 (H-T99d) refuses F5 at detection, which the refusal row below checks;
- ``hard-03`` (T-hard): pass 1's draft has **no close** (``unterminated string``); the repair answers the exact call
  (final ``dry_run``, source ``repair``);
- ``under-02`` (U-under): pass 1's draft is a **nameless** call: refused, never repaired (final ``refused``);
- arm F (``search-01``, ``search-02``): pass 1 drafts the case's reference call (valid); the forced repair answers it
  again (final ``dry_run``; agreement).

The refusal row (T-0100 criterion 12) runs kin-01, ``kout-01`` drafting ``press_key{combo: F5}`` and ``cpix-03`` drafting
the a794a98 ``click_at{x:1088,y:748}``: both end ``refused`` at detection with no repair request.

The leak row pre-patches ``live.classify`` with a forcing wrapper before ``main`` (a patch that escaped arm F) on a
one-case fixture whose draft is valid (``kin-01``), and expects criterion 8's void from the unpatched judge.

Preconditions, not patched around (as in T-0093's smoke): ``main`` refuses a dirty tree, so the rows skip, naming
that, when ``git status`` is not clean; the tool kit is the real one (the pinned tokenizer from the local HF cache).

Regime: CPU, mock engine on loopback, fake STT/TTS/audio/perception, real ``Session`` and ``VllmClient``; no
measurement.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import live
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.fakes.vllm_sse import MODEL, StreamSpec, build_lines
from tests.integration.test_t0093_harness_smoke import Encoder, native_call, require_clean_tree
from tests.unit.test_session_no_egress import loopback_only
from tests.unit.test_t0099_report import FORCED, GATE_KEYS, REPO, harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]

STRESS = REPO / "tests" / "eval" / "toolcall_stress" / "cases.jsonl"
V1_CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
STOP_TOOL, STOP_TURN = 50, 106
FIRST_CANVAS_DELAY_S = 0.1
#: longer than the machine's 3000 ms repair deadline
REPAIR_DELAY_S = 3.6
THOUGHT = "<|channel>thought\nOK.<channel|>"

#: kin-01's draft is valid: arm S must execute it unrepaired (a forcing that entered arm S would repair it)
SMOKE_IDS = ("kin-01", "kout-02", "hard-03", "under-02")
#: T-0100: drafts refused at detection (H-T99d, H-T99e), plus the valid control
REFUSE_IDS = ("kin-01", "kout-01", "cpix-03")
N_S = 3 * len(SMOKE_IDS)
#: Q8 row: a forced repair this slow is still running when the machine's takeover fires (600 ms after the onset),
#: and is well inside the 3000 ms repair deadline; a fast one finishes while the turn is PAUSED
SLOW_REPAIR_S = 1.5
FAST_REPAIR_S = 0.02
LEAK_IDS = ("kin-01",)
ARM_F_N = 2


def _rows(ids: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {
        r["id"]: r
        for r in (
            json.loads(x) for x in STRESS.read_text(encoding="utf-8").splitlines() if x.strip()
        )
    }
    return [by_id[i] for i in ids]


def _v1_ref() -> dict[str, dict[str, Any]]:
    out = {}
    for x in V1_CASES.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            out[r["id"]] = r
    return out


DRAFTS: dict[str, str] = {
    "kout-01": native_call({"name": "press_key", "arguments": {"combo": "F5"}}),
    "kout-02": native_call({"name": "press_key", "arguments": {"combo": "Control+C"}}),
    #: the a794a98 cpix-03 pass-1 text, verbatim
    "cpix-03": "<|tool_call>call:click_at{x:1088,y:748}<tool_call|>",
    "hard-03": '<|tool_call>call:type_at{x:500,y:36,text:<|"|>{hdmi',
    "under-02": '<|tool_call>call:{text:<|"|>wireless mouse<|"|>}<tool_call|>',
    "kin-01": native_call({"name": "press_key", "arguments": {"combo": "Enter"}}),
}
#: the repair's answer per (case, repetition): a call, or ``None`` for "stall past the deadline"
REPAIRS: dict[tuple[str, int], dict[str, Any] | None] = {
    ("kout-02", 0): {"name": "press_key", "arguments": {"combo": "Ctrl+C"}},
    ("kout-02", 1): {"name": "press_key", "arguments": {"combo": "F5"}},
    ("kout-02", 2): None,
    **{
        ("hard-03", r): {
            "name": "type_at",
            "arguments": {"x": 500, "y": 36, "text": "{hdmi,usb-c} cable", "press_enter": True},
        }
        for r in range(3)
    },
    **{("kin-01", r): {"name": "press_key", "arguments": {"combo": "Enter"}} for r in range(3)},
}


def continuation(call: dict[str, Any]) -> str:
    """The repair's reply: the call *after* ``<|tool_call>``. The repair prompt already ends in that tag and the
    client starts the continuation with it, so a real engine continues with ``call:...`` (a reply that repeated the
    tag would log ``<|tool_call><|tool_call>call:...``, which ``repair_verdict`` refuses)."""
    text = native_call(call)
    assert text.startswith("<|tool_call>")
    return text.removeprefix("<|tool_call>")


class Engine:
    """Scripted answers keyed by the case's user text found in the request's messages."""

    def __init__(
        self,
        stress: list[dict[str, Any]],
        forced_ids: list[str],
        repair_delay: dict[str, float] | None = None,
    ) -> None:
        self.enc = Encoder()
        #: per arm-F case, the delay before the forced repair's first canvas
        self.repair_delay = dict(repair_delay or {})
        ref = _v1_ref()
        self.texts: list[tuple[str, str]] = sorted(
            [(json.dumps(r["user"])[1:-1], r["id"]) for r in stress]
            + [(json.dumps(ref[i]["user"])[1:-1], i) for i in forced_ids],
            key=lambda t: len(t[0]),
            reverse=True,
        )
        self.forced_ref = {i: ref[i]["reference_call"] for i in forced_ids}
        self.lock = threading.Lock()
        self.pass1: dict[str, int] = {}
        self.requests: list[tuple[str, str]] = []
        self.unknown: list[str] = []
        self.repairs: list[tuple[str, int]] = []

    def case_of(self, body: dict[str, Any]) -> str | None:
        messages = json.dumps(body.get("messages"), ensure_ascii=False)
        return next((cid for text, cid in self.texts if text in messages), None)

    def answer(self, body: dict[str, Any]) -> tuple[list[int], float]:
        """``(ids, delay before the first canvas)``."""
        if body.get("max_tokens") == 1:
            return [STOP_TURN], 0.0
        cid = self.case_of(body)
        if body.get("continue_final_message"):  # the repair (pass 1 continued)
            with self.lock:
                rep = self.pass1.get(cid or "", 1) - 1
                self.repairs.append((str(cid), rep))
            if cid in self.forced_ref:
                return [
                    *self.enc.encode(continuation(self.forced_ref[cid])),
                    STOP_TOOL,
                ], self.repair_delay.get(str(cid), FIRST_CANVAS_DELAY_S)
            if (str(cid), rep) not in REPAIRS:
                # unscripted (a draft the row expects never to be repaired): answer the pass-1 draft again, at once,
                # so a premature repair fails the row on its assertion rather than stalling it
                again = DRAFTS[str(cid)].removeprefix("<|tool_call>")
                return [*self.enc.encode(again), STOP_TOOL], FIRST_CANVAS_DELAY_S
            planned = REPAIRS[(str(cid), rep)]
            if planned is None:
                return [
                    *self.enc.encode(
                        continuation({"name": "press_key", "arguments": {"combo": "Enter"}})
                    ),
                    STOP_TOOL,
                ], REPAIR_DELAY_S
            return [*self.enc.encode(continuation(planned)), STOP_TOOL], FIRST_CANVAS_DELAY_S
        if cid is None or not body.get("tools"):
            return [*self.enc.encode(THOUGHT + "Okay."), STOP_TURN], FIRST_CANVAS_DELAY_S
        with self.lock:
            self.pass1[cid] = self.pass1.get(cid, 0) + 1
        draft = native_call(self.forced_ref[cid]) if cid in self.forced_ref else DRAFTS[cid]
        return [
            *self.enc.encode(THOUGHT + "Okay." + "<channel|>" + draft),
            STOP_TOOL,
        ], FIRST_CANVAS_DELAY_S

    @staticmethod
    def metrics() -> str:
        return (
            f'vllm:prefix_cache_queries_total{{model_name="{MODEL}"}} 0\n'
            f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} 0\n'
            f'vllm:num_requests_running{{model_name="{MODEL}"}} 0\n'
            f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0\n'
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
                    ids = engine.enc.encode(str(body.get("prompt", "")))
                    reply = {"count": len(ids), "max_model_len": 16384, "tokens": ids}
                    self._send(200, json.dumps(reply).encode(), "application/json")
                    return
                if self.path != "/v1/chat/completions":
                    engine.unknown.append(f"POST {self.path}")
                    self._send(404, b"{}", "application/json")
                    return
                ids, delay = engine.answer(body)
                lines = build_lines(StreamSpec(chunks=[ids]), engine.enc.detok)
                with contextlib.suppress(OSError):  # an aborted repair closes the socket under us
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for k, line in enumerate(lines):
                        if k == 1:
                            time.sleep(delay)
                        self.wfile.write(f"{line}\n\n".encode())
                        self.wfile.flush()

        return Handler


@contextlib.contextmanager
def serve(
    stress: list[dict[str, Any]],
    forced_ids: list[str],
    repair_delay: dict[str, float] | None = None,
) -> Iterator[tuple[Engine, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine(stress, forced_ids, repair_delay)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0099-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _cut_down(
    h: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ids: tuple[str, ...],
    pick: Any = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Point the harness at an ``ids``-only stress fixture and the arm-F cases ``pick(cases)`` chooses from the
    harness's own ``load_forced_cases()`` (default: the first ``ARM_F_N``)."""
    rows = _rows(ids)
    fixture = tmp_path / "stress_fixture.jsonl"
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode(
        "utf-8"
    )
    fixture.write_bytes(data)
    monkeypatch.setattr(h, "STRESS_PATH", fixture)
    monkeypatch.setattr(h, "STRESS_SHA256", hashlib.sha256(data).hexdigest())
    every = list(h.load_forced_cases())
    forced = pick(every) if pick is not None else every[:ARM_F_N]
    monkeypatch.setattr(h, "load_forced_cases", lambda: forced)
    return rows, [c.case_id for c in forced]


def _lines(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / "turns.jsonl").read_text().splitlines() if x.strip()]


def test_main_runs_both_arms_and_each_scripted_outcome_lands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plumbing: 4 stress cases x k = 3 (repetition-major) then 2 arm-F cases, one session each; every scripted outcome
    lands in its final status; G1-G8 hold in both arms; ``live.classify`` is the original object after ``main``; no
    socket leaves loopback."""
    h = harness()
    require_clean_tree()
    original = live.classify
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, SMOKE_IDS)
    out = tmp_path / "t0099-smoke"
    with serve(rows, forced_ids) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert live.classify is original, "criterion 3: main left live.classify patched"
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], eng.unknown
    for name in ("turns.jsonl", "manifest.json", "report.json", "report.md"):
        assert (out / name).is_file(), f"criterion 9: no {name}"

    manifest = json.loads((out / "manifest.json").read_text())
    cases = manifest["cases"]
    order = [(c["arm"], c["case_id"], c.get("rep")) for c in cases]
    expected_order = [("S", cid, rep) for rep in range(3) for cid in SMOKE_IDS] + [
        ("F", cid, 0) for cid in forced_ids
    ]
    assert [(a, cid) for a, cid, _ in order] == [(a, cid) for a, cid, _ in expected_order], order
    assert [rep for _, _, rep in order] == [rep for _, _, rep in expected_order], order
    assert [c["forced"] for c in cases] == [False] * N_S + [True] * ARM_F_N
    assert len(manifest["sessions"]) == N_S + ARM_F_N
    assert len({c["session_id"] for c in cases}) == N_S + ARM_F_N

    lines = {line["session_id"]: line for line in _lines(out)}
    status: dict[tuple[str, int], tuple[str | None, str | None]] = {}
    seen: dict[str, int] = {}
    for c in cases:
        rep = seen.get(c["case_id"], 0)
        seen[c["case_id"]] = rep + 1
        tool = lines[c["session_id"]].get("tool") or {}
        status[(c["case_id"], rep)] = (tool.get("status"), tool.get("reason"))
    # the valid arm-S draft runs unrepaired: forcing entered only for arm F
    assert [status[("kin-01", r)][0] for r in range(3)] == ["dry_run"] * 3
    assert not [r for r in eng.repairs if r[0] == "kin-01"], "arm S repaired a valid draft"
    assert status[("kout-02", 0)][0] == "dry_run"
    assert status[("kout-02", 1)][0] == "refused"
    assert status[("kout-02", 2)] == ("cancelled", "repair_timeout")
    assert [status[("hard-03", r)][0] for r in range(3)] == ["dry_run"] * 3
    assert [status[("under-02", r)][0] for r in range(3)] == ["refused"] * 3
    for cid in forced_ids:
        st, _ = status[(cid, 0)]
        # Q6 (with Q8's attempt that found the repair already done): cancelled only after a barge-in attempt
        barged = next(c for c in cases if c["case_id"] == cid).get("barge_in_attempted")
        assert st == ("cancelled" if barged else "dry_run"), (cid, st, barged)
    assert ("under-02", 0) not in eng.repairs, "a nameless draft was repaired"

    report = h.build_report(out)
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    for arm in ("S", "F"):
        assert set(g[arm]) == GATE_KEYS
        assert all(g[arm].values()), (arm, g[arm])
    s = report["S"]
    # Q5: the 3 nameless under-02 drafts are counted on their own and are not in the yield denominator, so the
    # judged tool turns are kin-01 x3 (valid) + kout-02 x3 + hard-03 x3 (invalid) = 9, 6 invalid
    assert (s["yield"]["k"], s["yield"]["n"]) == (6, 9)
    assert s["refused_drafts"] == 3
    assert s["repairs"]["n"] == 6
    assert s["repairs"]["outcomes"] == {"dry_run": 4, "refused": 1, "cancelled": 1}
    assert [c["combo"] for c in s["coercion"]["cases"]] == ["Ctrl+C"]
    assert report["F"]["repairs"]["n"] == ARM_F_N
    assert report["F"]["repair_timeout"] == 0
    assert rc == 0, f"main exited {rc}"


def test_h99_refusals_end_refused_with_no_repair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T-0100 criterion 12, plumbing: kout-01 drafting F5 and cpix-03 drafting x = 1088 end ``refused`` at detection
    with their reasons ("combo not in enum", "coordinate out of range"), no repair request reaches the engine and no
    ``tool_repair`` request is logged; the report counts the 6 in ``refused_drafts``, out of the yield denominator
    (yield 0/3: kin-01 x3 valid), with no repair and ``coercion.n == 0``; G1-G8 hold and ``main`` exits 0."""
    h = harness()
    require_clean_tree()
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, REFUSE_IDS)
    out = tmp_path / "t0099-refuse"
    with serve(rows, forced_ids) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], eng.unknown
    manifest = json.loads((out / "manifest.json").read_text())
    lines = {line["session_id"]: line for line in _lines(out)}
    want = {
        "kin-01": ("dry_run", None),
        "kout-01": ("refused", "combo not in enum"),
        "cpix-03": ("refused", "coordinate out of range"),
    }
    for c in manifest["cases"]:
        if c["arm"] != "S":
            continue
        tool = lines[c["session_id"]].get("tool") or {}
        assert (tool.get("status"), tool.get("reason")) == want[c["case_id"]], (c["case_id"], tool)
        records = [
            json.loads(x)
            for x in (out / "sessions" / c["session_log"]).read_text().splitlines()
            if x.strip()
        ]
        repairs = [
            r
            for r in records
            if r.get("kind") == "llm.request"
            and (r.get("data") or {}).get("phase") == "tool_repair"
        ]
        assert repairs == [], f"T-0100: {c['case_id']} logged a tool_repair request"
    assert not [r for r in eng.repairs if r[0] in ("kout-01", "cpix-03")], eng.repairs

    report = h.build_report(out)
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    for arm in ("S", "F"):
        assert all(g[arm].values()), (arm, g[arm])
    s = report["S"]
    assert s["refused_drafts"] == 6
    assert (s["yield"]["k"], s["yield"]["n"]) == (0, 3)
    assert s["repairs"]["n"] == 0
    assert s["coercion"]["n"] == 0
    assert rc == 0, f"main exited {rc}"


def test_a_leaked_patch_voids_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Criterion 8 end to end: a forcing patch that escaped into arm S (pre-applied here) makes arm S repair valid
    drafts; the report, re-judging with the unpatched ``classify``, voids the run and ``main`` exits 1."""
    h = harness()
    require_clean_tree()
    original = live.classify
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, LEAK_IDS)

    def leaked(text: str) -> live.Verdict:
        v = original(text)
        if v.status == "valid":
            return live.Verdict("repairable", v.name, v.arguments, FORCED, v.query_problem)
        return v

    out = tmp_path / "t0099-leak"
    try:
        live.classify = leaked  # type: ignore[assignment]
        with serve(rows, forced_ids) as (eng, base_url), loopback_only() as attempts:
            rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    finally:
        live.classify = original  # type: ignore[assignment]
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.repairs, (
        "fixture: the leaked patch should have sent arm S's valid drafts to the repair"
    )
    report = h.build_report(out)
    void = [str(v) for v in report["void"]]
    assert any("integrity: repair on a valid draft outside arm F" in v for v in void), void
    assert rc == 1, f"criterion 8: a void run exits 1, not {rc}"


def test_unpinned_stress_file_is_refused_before_any_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 2: a stress file that is not the pinned one is refused with exit 2 before any socket."""
    h = harness()
    require_clean_tree()
    fixture = tmp_path / "not_pinned.jsonl"
    fixture.write_bytes(STRESS.read_bytes() + b"\n")
    monkeypatch.setattr(h, "STRESS_PATH", fixture)
    with serve(_rows(SMOKE_IDS), []) as (eng, base_url), loopback_only():
        rc = h.main(["--base-url", base_url, "--out-dir", str(tmp_path / "refused")])
    assert rc == 2
    assert eng.requests == [], f"refusal after a socket: {eng.requests}"
    err = capsys.readouterr().err
    assert str(fixture) in err or fixture.name in err, (
        f"the refusal does not name the stress file: {err[-400:]}"
    )


def test_barge_ins_during_forced_repairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Criterion 4 with ruling Q8, plumbing: arm F is the first six pre-registered barge-in candidates (in run order);
    arm S is kin-01 x 3. The mock delays every forced repair by ``SLOW_REPAIR_S`` (still running when the machine's
    own takeover fires) except the second candidate's, ``FAST_REPAIR_S`` (its verdict lands while the turn is PAUSED).

    By hand under Q8: attempts on candidates 1-5 (the cap), none on the 6th; delivered 4 (1, 3, 4, 5: interrupted,
    a call on the line, the repair's ``LlmAbort`` logged); repair done before the takeover 1 (the 2nd: interrupted,
    but no repair left to abort); the 6th runs to a dry run. Q6: every cancelled arm-F turn had an attempt. G1-G8
    hold; 4 delivered reads "read". (Plumbing: these counts describe the mock's timing, never the model.)"""
    h = harness()
    require_clean_tree()
    candidates = tuple(h.BARGE_IN_CANDIDATES)

    def pick(every: list[Any]) -> list[Any]:
        chosen = [c for c in every if c.case_id in candidates][:6]
        assert len(chosen) == 6, (
            "fixture: fewer than six barge-in candidates among the dev tool cases"
        )
        return chosen

    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, ("kin-01",), pick)
    delay = {cid: SLOW_REPAIR_S for cid in forced_ids}
    delay[forced_ids[1]] = FAST_REPAIR_S
    out = tmp_path / "t0099-barge"
    with serve(rows, forced_ids, delay) as (eng, base_url), loopback_only() as attempts:
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], eng.unknown
    manifest = json.loads((out / "manifest.json").read_text())
    f_cases = [c for c in manifest["cases"] if c["arm"] == "F"]
    assert [c["case_id"] for c in f_cases] == forced_ids
    tried = [c["case_id"] for c in f_cases if c.get("barge_in_attempted")]
    assert tried == forced_ids[:5], f"Q8: attempts on the first five candidates only, got {tried}"
    lines = {line["session_id"]: line for line in _lines(out)}
    for c in f_cases:
        tool = lines[c["session_id"]].get("tool") or {}
        if tool.get("status") == "cancelled":
            assert c.get("barge_in_attempted"), (
                f"Q6: {c['case_id']} cancelled without a barge-in attempt"
            )
    last = lines[f_cases[-1]["session_id"]]
    assert (last.get("tool") or {}).get("status") == "dry_run", last.get("tool")

    report = h.build_report(out)
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    for arm in ("S", "F"):
        assert all(g[arm].values()), (arm, g[arm])
    row = report["F"]["barge_in"]
    assert row.get("attempted") == 5, row
    assert row.get("delivered") == 4, row
    assert row.get("repair_done_before_takeover") == 1, row
    assert row.get("reading") == "read", row
    assert rc == 0, f"main exited {rc}"


# =====================================================================================================
# T-0101 (added rows): ``main --warm-repair`` on the mock (criterion 6)
#
# A warm-up request is a continuation whose ``vllm_xargs`` carry ``astra_request_id`` ``warmup-...`` and
# ``astra_tap: "off"``; T-0099's ``Engine`` would treat it as an unscripted case repair (and fail on it), so
# ``WarmEngine`` answers it first -- with a valid press_key call, whatever the tool -- and records the order of every
# chat completion: ``warmup`` or ``session``. ``fail_tool`` answers that tool's warm-up with HTTP 500.
# =====================================================================================================

import io  # noqa: E402

WARM_OUT = {"name": "press_key", "arguments": {"combo": "Enter"}}


def _xargs(body: dict[str, Any]) -> dict[str, Any]:
    x = body.get("vllm_xargs")
    return x if isinstance(x, dict) else {}


def is_warmup(body: dict[str, Any]) -> bool:
    """A warm-up by either mark (so a request carrying only one of them is still counted, and then fails the row)."""
    x = _xargs(body)
    return str(x.get("astra_request_id", "")).startswith("warmup-") or x.get("astra_tap") == "off"


class WarmEngine(Engine):
    def __init__(self, *args: Any, fail_tool: str | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.order: list[str] = []
        self.warm_bodies: list[dict[str, Any]] = []
        sha = live.load_payloads().tools[fail_tool].automaton_sha256 if fail_tool else None
        self.fail_sha = sha

    def answer(self, body: dict[str, Any]) -> tuple[list[int], float]:
        if is_warmup(body):
            with self.lock:
                self.order.append("warmup")
                self.warm_bodies.append(body)
            return [*self.enc.encode(continuation(WARM_OUT)), STOP_TOOL], 0.0
        with self.lock:
            self.order.append("session")
        return super().answer(body)

    def handler(self) -> type[BaseHTTPRequestHandler]:
        base = super().handler()
        engine = self

        class Failing(base):  # type: ignore[valid-type, misc]
            def do_POST(self) -> None:
                if self.path == "/v1/chat/completions" and engine.fail_sha is not None:
                    raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                    body = json.loads(raw or b"{}")
                    if (
                        is_warmup(body)
                        and _xargs(body).get("astra_cd_automaton_sha256") == engine.fail_sha
                    ):
                        engine.requests.append(("POST", self.path))
                        with engine.lock:
                            engine.order.append("warmup")
                            engine.warm_bodies.append(body)
                        self._send(500, b'{"error": "mock warm-up failure"}', "application/json")
                        return
                    self.rfile = io.BytesIO(raw)  # the base handler reads the body again
                super().do_POST()

        return Failing


@contextlib.contextmanager
def serve_warm(
    stress: list[dict[str, Any]], forced_ids: list[str], fail_tool: str | None = None
) -> Iterator[tuple[WarmEngine, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = WarmEngine(stress, forced_ids, fail_tool=fail_tool)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0101-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main_warm(h: Any, argv: list[str]) -> int:
    """``h.main(argv + ["--warm-repair"])``; an argparse exit on the unknown flag is criterion 6 absent, said so as an
    assertion (the pending convention tolerates only that or NotImplementedError)."""
    try:
        return int(h.main([*argv, "--warm-repair"]))
    except SystemExit as exc:
        raise AssertionError(
            f"criterion 6: main does not accept --warm-repair (exit {exc.code})"
        ) from exc


def _first_log_started(out: Path, manifest: dict[str, Any]) -> str:
    first = out / "sessions" / str(manifest["sessions"][0]["log"])
    for x in first.read_text().splitlines():
        r = json.loads(x)
        if r.get("kind") == "log.started":
            return str(r["wall"])
    raise AssertionError(f"{first.name} has no log.started record")


def test_main_warm_repair_warms_each_tool_before_any_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Criterion 6, plumbing: with ``--warm-repair``, five warm-up requests (one per payload sha) reach the engine
    before any session's first request, each with the tap opt-out and a ``warmup-`` id; the manifest says
    ``repair_warmup.on`` with every tool ``ok`` and an ``ended_wall`` no later than the first session's
    ``log.started``; no warm-up id is in any session log; the run is not void and exits 0; loopback only."""
    from datetime import datetime

    h = harness()
    require_clean_tree()
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, ("kin-01",))
    out = tmp_path / "t0101-warm"
    with serve_warm(rows, forced_ids) as (eng, base_url), loopback_only() as attempts:
        rc = main_warm(h, ["--base-url", base_url, "--out-dir", str(out)])
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], eng.unknown
    payloads = live.load_payloads()
    n = len(payloads.tools)
    assert eng.order[:n] == ["warmup"] * n, eng.order[: n + 3]
    assert "warmup" not in eng.order[n:], "a warm-up request after a session's first request"
    shas = [_xargs(b).get("astra_cd_automaton_sha256") for b in eng.warm_bodies]
    assert sorted(shas) == sorted(p.automaton_sha256 for p in payloads.tools.values()), shas
    for b in eng.warm_bodies:
        x = _xargs(b)
        assert x.get("astra_tap") == "off", x
        assert str(x.get("astra_request_id", "")).startswith("warmup-"), x
        assert str(x.get("astra_cd_record_tag", "")).startswith("warmup:"), x
        assert b.get("continue_final_message") is True
    manifest = json.loads((out / "manifest.json").read_text())
    rw = manifest["regime"].get("repair_warmup")
    assert isinstance(rw, dict) and rw.get("on") is True, rw
    assert sorted(rw["per_tool"]) == sorted(payloads.tools), rw
    assert all(v.get("status") == "ok" for v in rw["per_tool"].values()), rw
    assert isinstance(rw.get("total_ms"), int | float), rw
    ended = datetime.fromisoformat(str(rw["ended_wall"]))
    started = datetime.fromisoformat(_first_log_started(out, manifest))
    assert ended <= started, (
        f"the warm-up ended at {ended}, after the first session started at {started}"
    )
    for log in (out / "sessions").glob("*.jsonl"):
        assert "warmup-" not in log.read_text(), f"criterion 2: a warm-up request id in {log.name}"
    report = h.build_report(out)
    assert report["regime"]["repair_warmup"]["on"] is True
    assert report["void"] == [], report["void"]
    assert rc == 0, f"main exited {rc}"


def test_main_without_warm_repair_sends_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Criterion 6: the flag is off by default -- no warm-up request, ``regime.repair_warmup == {"on": false}``."""
    h = harness()
    require_clean_tree()
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, ("kin-01",), lambda every: [])
    out = tmp_path / "t0101-cold"
    with serve_warm(rows, forced_ids) as (eng, base_url), loopback_only():
        rc = h.main(["--base-url", base_url, "--out-dir", str(out)])
    assert "warmup" not in eng.order, eng.order
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["regime"].get("repair_warmup") == {"on": False}, manifest["regime"]
    assert rc == 0, f"main exited {rc}"


def test_main_failed_warmup_voids_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Criterion 6: a mock that answers scroll's warm-up with HTTP 500 leaves scroll not ``ok``: the run is void
    (``integrity: repair warm-up incomplete (scroll: ...)``) and ``main`` exits 1; the other tools were still tried."""
    h = harness()
    require_clean_tree()
    rows, forced_ids = _cut_down(h, monkeypatch, tmp_path, ("kin-01",), lambda every: [])
    out = tmp_path / "t0101-warm-fail"
    with serve_warm(rows, forced_ids, fail_tool="scroll") as (eng, base_url), loopback_only():
        rc = main_warm(h, ["--base-url", base_url, "--out-dir", str(out)])
    assert eng.order.count("warmup") == len(live.load_payloads().tools), eng.order
    manifest = json.loads((out / "manifest.json").read_text())
    rw = manifest["regime"].get("repair_warmup") or {}
    assert str((rw.get("per_tool") or {}).get("scroll", {}).get("status", "")).startswith(
        "failed"
    ), rw
    void = [str(v) for v in h.build_report(out)["void"]]
    assert any("integrity: repair warm-up incomplete (scroll: failed" in v for v in void), void
    assert rc == 1, f"criterion 6: a void run exits 1, not {rc}"
