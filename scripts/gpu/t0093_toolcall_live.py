"""T-0093 -- the live tool-call eval, through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0093_toolcall_live.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the report half (``build_report``, ``gates``,
``render_markdown``) reads files only, and everything that talks to the engine is imported inside ``main``.

Refusals (exit 2), in this order and all **before any socket**:

1. a ``--base-url`` that is not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (``localhost`` is a name and is
   refused; nothing is resolved);
2. a dirty tree (the dirty paths are named);
3. an existing, non-empty ``--out-dir``;
4. a ``tests/eval/toolcall/cases.jsonl`` or ``tests/eval/surface_placement_cases.jsonl`` that is not the pinned one,
   or a dev split that is not 40 tool + 8 no-tool cases;
5. a tool kit that ``cli.load_tool_kit`` cannot load, whose repair is off (the amendment: a run with repair off is
   invalid), or whose tool set is not ``bbfc53ccdd80``.

What it runs (docs/tasks/T-0093.md criterion 1, rulings J0-J10):

- **Cases**, 60, in this fixed order: the dev split of ``tests/eval/toolcall`` (40 tool + 8 no-tool, file order), the 7
  Q12 cases of ``tests/eval/surface_placement_cases.jsonl`` (J1, file order), and the 5 plain questions of
  ``PLAIN_QUESTIONS`` (J10).
- **Arms.** Each case runs once with tools off and once with ``tools = dry_run``, alternated per case (case 0: on
  then off; case 1: off then on; ...), concurrency 1, in one boot. Each (case, arm) is one fresh ``Session`` with one
  typed turn (``UserText``): no history carries between cases, so every prompt is the eval case's own question over
  the live prompt. The tools-on kit comes from ``cli.load_tool_kit`` itself, inside the timed startup.
- **The session.** The real ``VllmClient`` (unproxied, loopback) behind ``_MeteredLlm``; STT, TTS, audio and
  perception are ``tests/fakes`` (``FakeStt``, ``FakeTts`` at its default 300 ms/word, ``FakeAudioIO`` on the
  session's own clock with only the playback seams exposed, ``FakePerception`` holding the case's screen as the
  turn's frame); memory is ``_CaseMemory`` (no recollections, except the Q12 case that carries its captions as hits,
  as T-0055 ran it; never an uncaptioned episode, so the idle captioner never sends a request).
- **Prefix hit (J2).** ``_MeteredLlm`` lets the turn's anticipatory prefill run to its end (the session's abort of
  it at end-of-turn is deferred, never forwarded while it is running), makes pass 1's generate wait for it, then reads
  ``/metrics`` before and after that generate: the per-case ``prefix_hit_tokens``/``prefix_query_tokens`` are that
  delta. The wait is recorded per case (``prefill_wait_ms``); it is inside ``llm.first_canvas`` in both arms. J12:
  this departs from a production typed turn, whose prefill is aborted when the turn starts thinking. That
  whole-request number is reported and, since H-T96, not gated.
- **Prefix gate (H-T96, T-0096): B11 as written.** Per case the manifest also gets the prefill's own ``/metrics`` delta
  (``prefill_query_tokens`` P, ``prefill_hit_tokens`` Hp: from the pre-case idle check's last reading m0 to pass 1's
  pre-generate reading m1, so the timed path gets no extra read) and ``prefills_before_pass1``. After
  ``session.aclose()``, outside every timed span, ``/tokenize`` (chat form, ``add_special_tokens: false``) gives the
  engine's ids for the last prefill's prompt and pass 1's; they go to ``prefix_ids.jsonl``. ``build_report`` derives
  S = LCP(prefill ids, pass-1 ids) and Hs = min(H, S); a tools-on case holds iff S > 0, Hs >= floor(S/32)*32 and
  Hs/S >= 9/10. A = 32 is fixed (ruling M5). Lengths that are not P and Q, a prefill count other than 1, differing
  image digests, a nonzero hit that is not a multiple of 32 or a missing sidecar line void the run; if case 0 or the first image case (ruling M6) fails
  the id checks the run stops there. A run without ``prefix_rule == "H-T96"`` or the sidecar is never scored.
- **Barge-ins (J13, J18).** On tools-on tool cases from the 15 ``BARGE_IN_CANDIDATES``, in run order, until
  ``BARGE_IN_N`` have been delivered: once the turn is SPEAKING with pass 1 done (or its repair running), a
  ``UserSpeechOnset`` is submitted and the machine's own takeover timer does the rest (production has no other source
  of a takeover). A barge-in is *delivered* when the turn's line reads ``outcome == "interrupted"`` and carries a call
  (a routed tool turn); ``manifest.barge_in`` is that. Fewer than five voids the run.
- **Stopping (J19).** After ``UNREACHABLE_STOP`` consecutive cases in which ``/metrics`` cannot be read, the run stops
  (exit 1); the manifest says why and the report is void.
- **Startup** per session: from before the tool kit is loaded (tools on) or the client is built (tools off) to
  ``Session.start()`` returning, wall clock.

Output (``--out-dir``, default ``logs/t0093-live-<sha7>/``): ``turns.jsonl`` (the sessions' own lines),
``sessions/*.jsonl`` (one ``SessionLog`` per session), ``manifest.json`` (J0: sessions, arms, cases, barge-ins,
per-case prefix tokens, startup times, regime), ``prefix_ids.jsonl`` (H-T96), ``report.json`` and ``report.md``;
``data/`` holds the sessions'
thumbnails. Exit codes: 0 every gate holds; 1 a gate fails, the run is void, or the engine is not there; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import http.client
import importlib
import ipaddress
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_astra.toolcall import score as _score
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

if TYPE_CHECKING:
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.config import Profile
    from local_astra.contracts.cd import CdRequest
    from local_astra.contracts.llm import Commit, LlmOptions, PrefillHandle, Prompt
    from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
Q12_PATH = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
#: sha256 of ``tests/eval/toolcall/cases.jsonl`` (as T-0087 pinned it)
CASES_SHA256 = "460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85"
#: sha256 of ``tests/eval/surface_placement_cases.jsonl`` (J1: the 7 Q12 cases)
Q12_SHA256 = "e152810b0c779a7eeeb3a5d84947d173cc8415400f36fa1cf7f007749158d6d8"
DEV_TOOL_CASES = 40
DEV_NO_TOOL_CASES = 8

#: The amendment: every turns.jsonl line of the tools-on arm reads exactly this, or the run is void.
TOOLS_ON = "native:bbfc53ccdd80:dry_run"
TOOLS_OFF = "off"
TOOLS_SHA12 = "bbfc53ccdd80"
#: ``Session._note_repair_off``'s note (T-0092b criterion 2).
REPAIR_OFF_NOTE = "tool repair off"
STALE_NOTE = "stale playback terminal"

#: J10, pre-registered before the run: five short factual or chit-chat questions that need no tool.
PLAIN_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("plain-01", "What is the capital of Japan?"),
    ("plain-02", "Tell me a fun fact about octopuses."),
    ("plain-03", "How many minutes are there in a day?"),
    ("plain-04", "Who painted the Mona Lisa?"),
    ("plain-05", "How are you doing today?"),
)

#: Criterion 2's scripted barge-ins (J18): 5, delivered on routed tools-on tool turns. 15 pre-registered candidates,
#: 3 per tool; a barge-in is attempted on each in run order until ``BARGE_IN_N`` have been delivered on a routed turn.
#: Chosen from ``logs/t0084-native-dev-794161d`` (T-0084e dev, pass 1): the click, type, scroll and key candidates
#: spoke a prefix in the native format there. No google_search case did in the native format (all ten called
#: silently; all ten spoke in the ``<cc>`` format), so the three search candidates are the first three dev searches
#: other than search-03 and search-06, which J18 names as excluded.
BARGE_IN_N = 5
BARGE_IN_CANDIDATES: tuple[str, ...] = (
    "search-01",
    "search-02",
    "search-04",
    "click-01",
    "click-03",
    "click-06",
    "type-03",
    "type-04",
    "type-08",
    "scroll-01",
    "scroll-02",
    "scroll-05",
    "key-01",
    "key-03",
    "key-04",
)
#: J19: the run stops after this many consecutive cases in which the engine cannot be reached.
UNREACHABLE_STOP = 3

#: Criterion 2 (approved unchanged at H-T93): the bars, exact.
PREFIX_HIT_MIN = Fraction(9, 10)
FIRST_CANVAS_MAX_RATIO = Fraction(11, 10)
FALSE_TOOL_MAX = Fraction(1, 13)

#: Criterion 3: the eval's numbers (dev split, T-0091), printed beside the live ones.
EVAL_REF = {
    "route_recall": "40/40",
    "draft_valid": "40/40",
    "draft_exact": "39/40",
    "spoken_prefix": "27/40",
}

GATE_KEYS = (
    "run_valid",
    "no_literal_voiced",
    "no_execute_on_interrupted",
    "prefix_hit_on",
    "first_canvas_ratio",
    "false_tool_rate",
    "no_execute_before_drain",
    "no_stale_terminal_reached_machine",
)

METRIC_QUERIES = "vllm:prefix_cache_queries_total"
METRIC_HITS = "vllm:prefix_cache_hits_total"
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"

#: H-T96 (T-0096): the ``prefix_hit_on`` gate is B11 as written. A is fixed at 32 from 4b0bd92 (ruling M5): never
#: derived from the run, never read from the engine's ``cache_config_info`` ``block_size`` label (reported only).
PREFIX_RULE = "H-T96"
PREFIX_BLOCK_TOKENS = 32
#: T-0096 criterion 1d: one line per (case, arm) with the engine's ids for the prefill's and pass 1's prompts
SIDECAR = "prefix_ids.jsonl"
PREDATES = "unread: run predates H-T96"
#: the chat-form ``/tokenize`` body keeps exactly these keys of ``VllmClient.request_body`` (criterion 1c)
TOKENIZE_KEEP = (
    "model",
    "messages",
    "tools",
    "chat_template_kwargs",
    "add_generation_prompt",
    "continue_final_message",
)

#: How long one turn may take end to end before the case is recorded as timed out (the machine's own
#: ``llm_timeout_ms`` is 20 s, the repair's 3 s; playback of a long answer at 300 ms/word adds to that).
CASE_TIMEOUT_S = 180.0
#: How long pass 1 waits for its own prefill before it goes anyway (recorded as not done).
PREFILL_WAIT_S = 60.0
#: How long the engine may stay busy before a case starts anyway (recorded as ``engine_idle_before: false``).
QUIESCE_S = 30.0

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _say(msg: str) -> None:
    print(f"t0093_live: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The report (pure: reads the run directory, opens no socket)
# =====================================================================================================


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _percentile(values: Sequence[Any], p: int) -> Any:
    """J9: nearest-rank, the ceil(p/100 * n)-th smallest value (integer arithmetic, no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-p * len(ordered) // 100))
    return ordered[rank - 1]


def _dist(values: Sequence[Any]) -> dict[str, Any]:
    return {"n": len(values), "p50": _percentile(values, 50), "p90": _percentile(values, 90)}


def _rate(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


def _span(record: Mapping[str, Any] | None, name: str) -> tuple[Any, Any] | None:
    if record is None:
        return None
    spans = record.get("spans")
    if not isinstance(spans, Mapping):
        return None
    pair = spans.get(name)
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        return None
    return pair[0], pair[1]


def _span_len(record: Mapping[str, Any] | None, name: str) -> Any:
    pair = _span(record, name)
    return None if pair is None else pair[1] - pair[0]


def _normalise(text: str) -> str:
    """J17: whitespace-normalised (every run of whitespace is one space; no leading or trailing space)."""
    return " ".join(text.split())


@dataclass
class _Turn:
    """One turn's session-log records (``log``, ``turn_id``), in ``seq`` order, and its turns.jsonl line."""

    log: str
    turn_id: str | None
    records: list[dict[str, Any]] = field(default_factory=list)
    line: dict[str, Any] | None = None

    def data(self, kind: str) -> list[dict[str, Any]]:
        return [r.get("data") or {} for r in self.records if r.get("kind") == kind]

    def steps(self) -> list[dict[str, Any]]:
        return self.data("machine.step")

    def commands(self, ctype: str) -> list[dict[str, Any]]:
        return [
            c
            for s in self.steps()
            for c in (s.get("commands") or [])
            if isinstance(c, dict) and c.get("type") == ctype
        ]

    def gen_ids(self) -> set[str]:
        ids = {
            str(d.get("request_id"))
            for d in self.data("llm.request")
            if d.get("phase") == "generate" and d.get("request_id") is not None
        }
        return ids or {f"{self.turn_id}-gen"}

    def tool_commits(self) -> list[dict[str, Any]]:
        """J8: pass 1's ``tool`` commit(s): ``llm.commit`` records of segment ``tool`` from a generate request."""
        ids = self.gen_ids()
        return [
            d
            for d in self.data("llm.commit")
            if d.get("segment") == "tool" and str(d.get("request_id")) in ids
        ]

    def routed(self) -> bool:
        return bool(self.tool_commits())

    def calls(self) -> list[dict[str, Any]]:
        return self.data("tool.call")

    def first_call(self) -> dict[str, Any] | None:
        calls = self.calls()
        return calls[0] if calls else None

    def final_status(self) -> str | None:
        calls = self.calls()
        if calls:
            return str(calls[-1].get("status"))
        tool = (self.line or {}).get("tool")
        return str(tool.get("status")) if isinstance(tool, Mapping) else None

    def interrupted(self) -> bool:
        if (self.line or {}).get("outcome") == "interrupted":
            return True
        for s in self.steps():
            event = s.get("event") or {}
            if s.get("state_after") == "INTERRUPTED" or event.get("type") == "user.takeover":
                return True
        return False

    def repaired(self) -> bool:
        return (
            any(d.get("phase") == "tool_repair" for d in self.data("llm.request"))
            or any(d.get("status") == "repairing" for d in self.calls())
            or bool(self.commands("tool.repair_start"))
        )

    def any_tool(self) -> bool:
        """A tool call of any kind: a pass-1 tool commit, a ``tool.call`` record, the line's ``tool``, a ``ToolExecute``."""
        return (
            self.routed()
            or bool(self.calls())
            or (self.line or {}).get("tool") is not None
            or bool(self.commands("tool.execute"))
        )


@dataclass
class _Case:
    """A manifest case and its turn."""

    meta: dict[str, Any]
    turn: _Turn


def _voiced(turn: _Turn) -> list[dict[str, Any]]:
    """Criterion 2 with J17 (human; supersedes J11's argument values). J4 scope: only ``TtsSpeak`` with non-empty
    text is checked (a refused or trimmed source travels with ``text=""``).

    - **Every turn:** no ``SPECIAL_TOKENS`` literal and no ``call:`` in the text, and none in the source, each side
      checked on its own.
    - **Tool turns** (a ``tool.call`` record): the concatenated voiced sources, whitespace-normalised, are a prefix of
      the turn's cleaned ``spoken_prefix`` from its first ``tool.call`` record (a prefix, not equal: a barge-in cuts
      the speech short). What the model says it will do, a value it will type included, is that prefix itself."""
    out: list[dict[str, Any]] = []
    sources: list[str] = []
    for c in turn.commands("tts.speak"):
        text = c.get("text")
        if not isinstance(text, str) or not text:
            continue
        raw = c.get("source")
        source = raw if isinstance(raw, str) else ""
        sources.append(source)
        for name, value in (("text", text), ("source", source)):
            hits = [lit for lit in SPECIAL_TOKENS if lit in value]
            if "call:" in value:
                hits.append("call:")
            if hits:
                out.append(
                    {
                        "log": turn.log,
                        "turn_id": turn.turn_id,
                        "utterance_id": c.get("utterance_id"),
                        "field": name,
                        "found": hits,
                    }
                )
    first = turn.first_call()
    if first is not None and sources:
        raw_prefix = first.get("spoken_prefix")
        prefix = _normalise(raw_prefix if isinstance(raw_prefix, str) else "")
        voiced = _normalise("".join(sources))
        if not prefix.startswith(voiced):
            out.append(
                {
                    "log": turn.log,
                    "turn_id": turn.turn_id,
                    "utterance_id": None,
                    "field": "sources",
                    "found": [f"voiced {voiced!r} is not a prefix of spoken_prefix {prefix!r}"],
                }
            )
    return out


def _executes_before_drain(turn: _Turn) -> list[dict[str, Any]]:
    """Amendment (T-0092a): a ``ToolExecute`` must come in or after a ``playback.drained`` step of its own turn.

    The one exception is T-0089 R1's THINKING row with nothing committed: a step taken in THINKING, in a turn that has
    voiced nothing, has nothing to drain (that row executes on ``llm.done`` and ends the turn)."""
    drained = False
    spoken = False
    out: list[dict[str, Any]] = []
    for s in turn.steps():
        event = s.get("event") or {}
        if event.get("type") == "playback.drained":
            drained = True
        commands = [c for c in (s.get("commands") or []) if isinstance(c, Mapping)]
        spoken = spoken or any(c.get("type") == "tts.speak" and c.get("text") for c in commands)
        for c in commands:
            if c.get("type") != "tool.execute":
                continue
            if drained or (s.get("state_before") == "THINKING" and not spoken):
                continue
            out.append(
                {
                    "log": turn.log,
                    "turn_id": turn.turn_id,
                    "call_id": c.get("call_id"),
                    "event": event.get("type"),
                    "state_before": s.get("state_before"),
                }
            )
    return out


def _stale(records: Sequence[Mapping[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    """J7: the stale-terminal notes of one session log, and those whose terminal (same type, ``played_ms`` and
    ``played_text``) appears in **any** later ``machine.step`` of the same session."""
    notes = [
        r
        for r in records
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == STALE_NOTE
    ]
    reached: list[dict[str, Any]] = []
    for note in notes:
        d = note.get("data") or {}
        key = (d.get("type"), d.get("played_ms"), d.get("played_text"))
        for r in records:
            if r.get("kind") != "machine.step" or int(r.get("seq", 0)) <= int(note.get("seq", 0)):
                continue
            ev = (r.get("data") or {}).get("event") or {}
            if (ev.get("type"), ev.get("played_ms"), ev.get("played_text")) == key:
                reached.append({"seq": note.get("seq"), "turn_id": note.get("turn_id"), **d})
                break
    return len(notes), reached


def _q12_expectations() -> dict[str, bool]:
    if not Q12_PATH.is_file():
        return {}
    return {str(r["id"]): bool(r["expect_surface"]) for r in _read_jsonl(Q12_PATH) if "id" in r}


def _call_dict(call: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": call.get("name"), "arguments": call.get("arguments")}


# --- H-T96 (T-0096): B11's prefix, from the sidecar's ids ---------------------------------------------------------


def _id_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not all(type(i) is int for i in value):
        return None
    return value


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """S: the number of leading ids two lists share (criterion 2)."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _read_sidecar(
    run_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]] | None:
    """``prefix_ids.jsonl`` by (case_id, arm), first line wins, and the void reasons of lines that do not parse;
    ``None`` when the run has no sidecar."""
    path = run_dir / SIDECAR
    if not path.is_file():
        return None
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    bad: list[str] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except ValueError:
            item = None
        if not isinstance(item, dict):
            bad.append(f"integrity: {SIDECAR} line {n} is unparsable")
            continue
        rows.setdefault((str(item.get("case_id")), str(item.get("arm"))), item)
    return rows, bad


def _ids_voids(meta: Mapping[str, Any], line: Mapping[str, Any] | None) -> list[str]:
    """Criterion 5 (f), (a), (b) for one (case, arm): a sidecar line with two id lists whose lengths are the
    ``/metrics`` query counts of the prefill (P) and of pass 1 (Q). Also the fail-fast check (criterion 6, M6)."""
    tag = f"{meta.get('case_id')} ({meta.get('arm')})"
    if line is None:
        return [f"integrity: {tag} has no {SIDECAR} line"]
    pre, gen = _id_list(line.get("prefill_ids")), _id_list(line.get("generate_ids"))
    if pre is None or gen is None:
        return [f"integrity: {tag}'s {SIDECAR} line has no readable id lists"]
    out: list[str] = []
    p, q = meta.get("prefill_query_tokens"), meta.get("prefix_query_tokens")
    if not isinstance(p, int) or len(pre) != p:
        out.append(
            f"integrity: {tag}: /tokenize gave {len(pre)} prefill ids, "
            f"/metrics counted {p} prefill query tokens"
        )
    if not isinstance(q, int) or len(gen) != q:
        out.append(
            f"integrity: {tag}: /tokenize gave {len(gen)} pass-1 ids, "
            f"/metrics counted {q} pass-1 query tokens"
        )
    return out


def _request_record(
    records: Sequence[Mapping[str, Any]], phase: str, request_id: Any
) -> dict[str, Any] | None:
    """The ``llm.request`` record of ``phase`` whose request id (the session's) or attempt id (the engine's, which
    the sidecar holds) is ``request_id``."""
    if request_id is None:
        return None
    for r in records:
        d = r.get("data") or {}
        if r.get("kind") != "llm.request" or d.get("phase") != phase:
            continue
        if request_id in (d.get("request_id"), d.get("attempt_id")):
            return dict(d)
    return None


def _digests_in(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        own = value.get("digest")
        found = [own] if isinstance(own, str) else []
        return found + [d for k, v in value.items() if k != "digest" for d in _digests_in(v)]
    if isinstance(value, list | tuple):
        return [d for v in value for d in _digests_in(v)]
    return []


def _message_digests(record: Mapping[str, Any]) -> list[list[str]]:
    messages = (record.get("prompt") or {}).get("messages") or []
    return [_digests_in(m.get("content")) if isinstance(m, Mapping) else [] for m in messages]


def _b11_voids(
    meta: Mapping[str, Any],
    line: Mapping[str, Any] | None,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Criterion 5's voids for one (case, arm), each naming the case, the arm and the numbers."""
    tag = f"{meta.get('case_id')} ({meta.get('arm')})"
    out = _ids_voids(meta, line)
    n = meta.get("prefills_before_pass1")
    if n != 1:
        out.append(f"integrity: {tag}: {n} prefill(s) started before pass 1, not 1")
    if line is not None:
        pre = _request_record(records, "prefill", line.get("prefill_request_id"))
        gen = _request_record(records, "generate", line.get("generate_request_id"))
        if pre is None or gen is None:
            out.append(
                f"integrity: {tag}: no llm.request record for the prefill "
                f"{line.get('prefill_request_id')} or pass 1 {line.get('generate_request_id')}"
            )
        else:
            a, b = _message_digests(pre), _message_digests(gen)
            k = min(len(a), len(b))
            if a[:k] != b[:k]:
                shown = [", ".join(d for m in x[:k] for d in m) for x in (a, b)]
                out.append(
                    f"integrity: {tag}: the image digests of the shared messages differ: "
                    f"prefill [{shown[0]}], pass 1 [{shown[1]}]"
                )
    odd = [
        f"{what} {v}"
        for what, v in (
            ("prefill hit", meta.get("prefill_hit_tokens")),
            ("pass-1 hit", meta.get("prefix_hit_tokens")),
        )
        if isinstance(v, int) and v != 0 and v % PREFIX_BLOCK_TOKENS != 0
    ]
    if odd:
        out.append(
            f"integrity: {tag}: {', '.join(odd)} not a multiple of {PREFIX_BLOCK_TOKENS} "
            f"(cache granularity is not the registered {PREFIX_BLOCK_TOKENS})"
        )
    return out


_B11_FIELDS = (
    "S",
    "P",
    "Hp",
    "H",
    "Q",
    "hit_on_prefix",
    "allowed_miss",
    "missed",
    "ratio",
    "holds",
    "hit_beyond_prefix",
    "prefill_prewarmed",
    "prefill_tail",
)


def _b11_row(
    meta: Mapping[str, Any], line: Mapping[str, Any] | None, voids: Sequence[str]
) -> dict[str, Any]:
    """Criterion 8's row for one (case, arm): S from the ids, Hs = min(H, S), and B11 per case (criterion 3)."""
    a = PREFIX_BLOCK_TOKENS
    p, hp = meta.get("prefill_query_tokens"), meta.get("prefill_hit_tokens")
    h, q = meta.get("prefix_hit_tokens"), meta.get("prefix_query_tokens")
    pre = _id_list((line or {}).get("prefill_ids"))
    gen = _id_list((line or {}).get("generate_ids"))
    s = _lcp(pre, gen) if pre is not None and gen is not None else None
    hs = min(h, s) if isinstance(h, int) and s is not None else None
    row: dict[str, Any] = {
        "case_id": meta.get("case_id"),
        "S": s,
        "P": p,
        "Hp": hp,
        "H": h,
        "Q": q,
        "hit_on_prefix": hs,
        "allowed_miss": s % a if s is not None else None,
        "missed": s - hs if s is not None and hs is not None else None,
        "ratio": float(Fraction(hs, s)) if hs is not None and s else None,
        "holds": None,
        "hit_beyond_prefix": h - hs if isinstance(h, int) and hs is not None else None,
        "prefill_prewarmed": (
            float(Fraction(hp, p)) if isinstance(hp, int) and isinstance(p, int) and p > 0 else None
        ),
        "prefill_tail": p - s if isinstance(p, int) and s is not None else None,
    }
    if voids or s is None or hs is None:
        row["reading"] = "unread: " + ("; ".join(voids) if voids else "no measurement")
        return row
    row["reading"] = "measured"
    row["holds"] = s > 0 and hs >= (s // a) * a and Fraction(hs, s) >= PREFIX_HIT_MIN
    return row


def _b11_report(
    run_dir: Path,
    manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    logs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Criteria 2-5, 8 and 12: the B11 report keys and their void reasons. A run without ``prefix_rule ==
    "H-T96"`` or without the sidecar predates H-T96: every row reads so, no void is added, nothing is re-scored."""
    rule = (manifest.get("regime") or {}).get("prefix_rule")
    side = _read_sidecar(run_dir)
    rows: dict[str, list[dict[str, Any]]] = {"on": [], "off": []}
    void: list[str] = []
    predates = side is None or rule != PREFIX_RULE
    if side is None or rule != PREFIX_RULE:
        for c in cases:
            if c.get("arm") in rows:
                row = {"case_id": c.get("case_id"), **dict.fromkeys(_B11_FIELDS)}
                rows[str(c["arm"])].append({**row, "reading": PREDATES})
    else:
        by_key, bad = side
        void += bad
        for c in cases:
            line = by_key.get((str(c.get("case_id")), str(c.get("arm"))))
            reasons = _b11_voids(c, line, logs.get(str(c.get("session_log"))) or [])
            void += reasons
            if c.get("arm") in rows:
                rows[str(c["arm"])].append(_b11_row(c, line, reasons))

    def pooled(arm: str) -> float | None:
        ok = [r for r in rows[arm] if r.get("holds") is not None]
        total = sum(r["S"] for r in ok)
        return float(Fraction(sum(r["hit_on_prefix"] for r in ok), total)) if total else None

    keys = {
        "prefix_rule": rule if rule == PREFIX_RULE else None,
        "prefix_block_tokens": None if predates else PREFIX_BLOCK_TOKENS,
        "prefix_b11_cases": rows,
        "prefix_b11": {arm: pooled(arm) for arm in ("on", "off")},
    }
    return keys, void


def build_report(run_dir: Path) -> dict[str, Any]:
    """The report of a run directory (``turns.jsonl``, ``sessions/*.jsonl``, ``manifest.json``; J0)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sessions: list[dict[str, Any]] = list(manifest.get("sessions") or [])
    cases: list[dict[str, Any]] = list(manifest.get("cases") or [])
    void: list[str] = []

    lines = _read_jsonl(run_dir / "turns.jsonl") if (run_dir / "turns.jsonl").is_file() else []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in lines:
        by_key.setdefault((str(line.get("session_id")), str(line.get("turn_id"))), []).append(line)

    # a log belongs to the session of the case that names it (first wins), then of the session entry that does
    arm_of_session: dict[str, str] = {}
    log_session: dict[str, str] = {}
    for c in cases:
        arm_of_session.setdefault(str(c.get("session_id")), str(c.get("arm")))
        log_session.setdefault(str(c.get("session_log")), str(c.get("session_id")))
    for s in sessions:
        arm_of_session.setdefault(str(s.get("session_id")), str(s.get("arm")))
        log_session.setdefault(str(s.get("log")), str(s.get("session_id")))
    stopped = manifest.get("stopped")
    if stopped:
        void.append(f"integrity: the run stopped early: {stopped}")

    # --- the session logs, grouped by (log, turn id) --------------------------------------------------
    logs: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(log_session):
        path = run_dir / "sessions" / name
        if not path.is_file():
            void.append(f"integrity: session log {name} is missing")
            continue
        records = sorted(_read_jsonl(path), key=lambda r: int(r.get("seq", 0)))
        logs[name] = records
        dropped = [r for r in records if r.get("kind") == "dropped"]
        if dropped:
            void.append(f"integrity: session log {name} dropped records ({len(dropped)} gap(s))")
    turns: dict[tuple[str, str | None], _Turn] = {}
    for name, records in logs.items():
        for r in records:
            tid = r.get("turn_id")
            key = (name, None if tid is None else str(tid))
            turns.setdefault(key, _Turn(log=name, turn_id=key[1])).records.append(r)
    for (name, tid), turn in turns.items():
        sid = log_session.get(name)
        found = by_key.get((str(sid), str(tid))) if sid is not None and tid is not None else None
        turn.line = found[0] if found else None

    # --- the cases ------------------------------------------------------------------------------------
    joined: list[_Case] = []
    for c in cases:
        key = (str(c.get("session_id")), str(c.get("turn_id")))
        found = by_key.get(key) or []
        if len(found) != 1:
            void.append(
                f"integrity: {c.get('case_id')} ({c.get('arm')}) has {len(found)} turns.jsonl lines, not 1"
            )
        name = str(c.get("session_log"))
        turn = turns.get((name, str(c.get("turn_id")))) or _Turn(
            log=name, turn_id=str(c.get("turn_id"))
        )
        if turn.line is None and found:
            turn.line = found[0]
        joined.append(_Case(meta=c, turn=turn))

    def arm_cases(arm: str, *kinds: str) -> list[_Case]:
        return [
            j
            for j in joined
            if j.meta.get("arm") == arm and (not kinds or j.meta.get("kind") in kinds)
        ]

    # --- void (amendment) -----------------------------------------------------------------------------
    on_sessions = {sid for sid, arm in arm_of_session.items() if arm == "on"}
    off_sessions = {sid for sid, arm in arm_of_session.items() if arm == "off"}
    if not arm_cases("on"):
        void.append("no tools-on case in the manifest")
    wrong_on = sorted(
        {
            str((line.get("regime") or {}).get("tools"))
            for line in lines
            if str(line.get("session_id")) in on_sessions
            and (line.get("regime") or {}).get("tools") != TOOLS_ON
        }
    )
    if wrong_on:
        void.append(f"a tools-on line reads tools={wrong_on}, not {TOOLS_ON!r}")
    for name, records in logs.items():
        if log_session.get(name) not in on_sessions:
            continue
        for r in records:
            d = r.get("data") or {}
            if r.get("kind") == "note" and d.get("what") == REPAIR_OFF_NOTE:
                void.append(f"repair off in {name}: {d.get('reason')}")
                break
    wrong_off = sorted(
        {
            str((line.get("regime") or {}).get("tools"))
            for line in lines
            if str(line.get("session_id")) in off_sessions
            and (line.get("regime") or {}).get("tools") != TOOLS_OFF
        }
    )
    if wrong_off:
        void.append(f"integrity: a tools-off line reads tools={wrong_off}, not {TOOLS_OFF!r}")

    # --- J19: errored and incomplete cases, in either arm ------------------------------------------------
    errored = [j for j in joined if (j.turn.line or {}).get("outcome") == "error"]
    timed_out = [j for j in joined if j.meta.get("timed_out")]
    not_idle = [j for j in joined if j.meta.get("engine_idle_before") is False]
    for j in joined:
        tag = f"{j.meta.get('case_id')} ({j.meta.get('arm')})"
        if (j.turn.line or {}).get("outcome") == "error":
            void.append(f"integrity: {tag} ended in an error outcome")
        if j.meta.get("failures"):
            void.append(f"integrity: {tag} recorded session failures: {j.meta.get('failures')}")
        if j.meta.get("timed_out"):
            void.append(f"integrity: {tag} timed out")
        if j.meta.get("prefill_done") is False:
            void.append(f"integrity: {tag} ran pass 1 without its prefill having finished")
        inflight = j.meta.get("max_inflight")
        if isinstance(inflight, int) and inflight > 1:
            void.append(f"integrity: {tag} had {inflight} requests in flight at once")

    # --- gate inputs over every turn of every session --------------------------------------------------
    voiced: list[dict[str, Any]] = []
    on_interrupted: list[dict[str, Any]] = []
    before_drain: list[dict[str, Any]] = []
    for turn in turns.values():
        voiced += _voiced(turn)
        before_drain += _executes_before_drain(turn)
        if turn.interrupted():
            for c in turn.commands("tool.execute"):
                on_interrupted.append(
                    {"log": turn.log, "turn_id": turn.turn_id, "call_id": c.get("call_id")}
                )
    dropped_total = 0
    reached: list[dict[str, Any]] = []
    for name, records in logs.items():
        n, hits = _stale(records)
        dropped_total += n
        reached += [{"log": name, **h} for h in hits]

    # --- criterion 3 (J5, J8) ------------------------------------------------------------------------
    on_tool = arm_cases("on", "tool")
    routed = [j for j in on_tool if j.turn.routed()]
    unbarged = [j for j in on_tool if not j.meta.get("barge_in")]
    # J18: a barge-in counts only when it was delivered on a routed tool turn; fewer than five voids the run
    barged = [j for j in on_tool if j.meta.get("barge_in") and j.turn.routed()]
    if len(barged) < BARGE_IN_N:
        void.append(
            f"integrity: {len(barged)} barge-in(s) delivered on routed tool turns, not {BARGE_IN_N}"
        )

    def draft_valid(j: _Case) -> bool:
        call = j.turn.first_call()
        return j.turn.routed() and call is not None and call.get("schema_valid") is True

    def draft_exact(j: _Case) -> bool:
        call, gt = j.turn.first_call(), j.meta.get("gt")
        if not j.turn.routed() or call is None or not isinstance(gt, Mapping):
            return False
        return bool(_score.exact(_call_dict(call), gt))

    searches = [j for j in routed if (j.turn.first_call() or {}).get("name") == "google_search"]
    spoken = [
        j for j in routed if str((j.turn.first_call() or {}).get("spoken_prefix") or "").strip()
    ]
    false_cases = arm_cases("on", "no_tool", "plain")

    repaired = [j for j in joined if j.turn.repaired()]
    outcomes: dict[str, int] = {}
    for j in repaired:
        status = j.turn.final_status() or "none"
        outcomes[status] = outcomes.get(status, 0) + 1
    repair_ms = [v for j in repaired if (v := _span_len(j.turn.line, "tool.repair")) is not None]

    action: list[Any] = []
    for j in joined:
        execute, done = _span(j.turn.line, "tool.execute"), _span(j.turn.line, "llm.done")
        if execute is not None and done is not None:
            action.append(execute[0] - done[1])

    expectations = _q12_expectations()

    def q12_chosen(arm: str, positive: bool) -> list[_Case]:
        chosen = []
        for j in arm_cases(arm, "q12"):
            cid = str(j.meta.get("case_id"))
            expect = expectations.get(cid, j.meta.get("expect_surface"))
            if expect is None:
                continue
            if cid in expectations and j.meta.get("expect_surface") not in (None, expect):
                void.append(
                    f"integrity: {cid}'s expect_surface in the manifest differs from J1's file"
                )
            if bool(expect) is positive:
                chosen.append(j)
        return chosen

    def q12_row(j: _Case, expect: bool) -> dict[str, Any]:
        """T-0097 criterion 5: one Q12 (case, arm), read from its own turn's ``ui.gate`` records (``seq`` order)."""
        records = j.turn.data("ui.gate")
        gated = [d for d in records if d.get("gated") is True]
        emitting = [d for d in gated if d.get("surface_emitted") is True]
        extras = (j.turn.line or {}).get("extras") or {}
        refused = (
            extras.get("speech_refused")
            if "speech_refused" in extras
            else len(j.turn.data("speech.refused"))
        )
        return {
            "case_id": j.meta.get("case_id"),
            "arm": j.meta.get("arm"),
            "expect_surface": expect,
            "shown": bool(emitting),
            "all_ui_valid": extras.get("ui_valid"),
            "gated": len(gated),
            "emitted": len(emitting),
            "failed": len(gated) - len(emitting),
            "not_gated": sum(1 for d in records if d.get("gated") is False),
            "first_emit_canvas": emitting[0].get("canvas_idx") if emitting else None,
            "speech_refused": refused,
        }

    # the selection runs in the order the J0 keys were computed, so the void reasons it adds are unchanged
    q12_groups = {
        (positive, arm): q12_chosen(arm, positive)
        for positive in (True, False)
        for arm in ("on", "off")
    }
    q12_rows: dict[tuple[bool, str], list[dict[str, Any]]] = {
        (positive, arm): [q12_row(j, positive) for j in chosen]
        for (positive, arm), chosen in q12_groups.items()
    }

    def q12_rate(positive: bool, field_: str) -> dict[str, dict[str, int]]:
        return {
            arm: _rate(
                sum(1 for r in q12_rows[positive, arm] if r[field_] is True),
                len(q12_rows[positive, arm]),
            )
            for arm in ("on", "off")
        }

    def q12_counts(positive: bool, arm: str) -> dict[str, int]:
        return {
            c: sum(r[c] for r in q12_rows[positive, arm])
            for c in ("gated", "emitted", "failed", "not_gated")
        }

    q12_cases = [r for arm in ("on", "off") for p in (True, False) for r in q12_rows[p, arm]]
    q12_inconsistent = [
        {"case_id": r["case_id"], "arm": r["arm"]}
        for r in q12_cases
        if (r["all_ui_valid"] is True and r["emitted"] == 0)
        or (r["all_ui_valid"] is False and r["failed"] == 0)
        or (r["all_ui_valid"] is None and r["gated"] > 0)
    ]

    def spans_of(arm: str, name: str) -> list[Any]:
        return [v for j in arm_cases(arm) if (v := _span_len(j.turn.line, name)) is not None]

    first_canvas: dict[str, Any] = {
        arm: _dist(spans_of(arm, "llm.first_canvas")) for arm in ("on", "off")
    }
    p_on, p_off = first_canvas["on"]["p50"], first_canvas["off"]["p50"]
    first_canvas["ratio"] = (p_on / p_off) if p_on is not None and p_off else None

    def prefix_cases(arm: str) -> list[dict[str, Any]]:
        return [
            {
                "case_id": j.meta.get("case_id"),
                "hit": j.meta.get("prefix_hit_tokens"),
                "query": j.meta.get("prefix_query_tokens"),
            }
            for j in arm_cases(arm)
        ]

    def pooled(rows: Sequence[Mapping[str, Any]]) -> float | None:
        ok = [r for r in rows if isinstance(r["hit"], int) and isinstance(r["query"], int)]
        total = sum(r["query"] for r in ok)
        return sum(r["hit"] for r in ok) / total if total else None

    prefix_rows = {arm: prefix_cases(arm) for arm in ("on", "off")}
    startup = {
        arm: _percentile(
            [
                s.get("startup_ms")
                for s in sessions
                if s.get("arm") == arm and s.get("startup_ms") is not None
            ],
            50,
        )
        for arm in ("on", "off")
    }
    stale_dropped = dropped_total
    # H-T96 (T-0096): B11 per case; its voids come after every other reason
    b11, b11_void = _b11_report(run_dir, manifest, cases, logs)
    void += b11_void
    return {
        "run_dir": str(run_dir),
        "git_commit": sorted({str((line.get("regime") or {}).get("git_commit")) for line in lines}),
        "regime": manifest.get("regime"),
        "void": void,
        "cases": {arm: len(arm_cases(arm)) for arm in ("on", "off")},
        "incomplete": {
            "errored": len(errored),
            "timed_out": len(timed_out),
            "not_idle": len(not_idle),
        },
        "route_recall": _rate(len(routed), len(on_tool)),
        "draft_valid": _rate(sum(draft_valid(j) for j in unbarged), len(unbarged)),
        "draft_exact": _rate(sum(draft_exact(j) for j in unbarged), len(unbarged)),
        "barge_in": {
            "n": len(barged),
            "cases": [j.meta.get("case_id") for j in barged],
            "draft_valid": _rate(sum(draft_valid(j) for j in barged), len(barged)),
            "draft_exact": _rate(sum(draft_exact(j) for j in barged), len(barged)),
            "executed": sum(1 for j in barged if j.turn.commands("tool.execute")),
        },
        "query_problem": _rate(
            sum(
                1 for j in searches if (j.turn.first_call() or {}).get("query_problem") is not None
            ),
            len(searches),
        ),
        "spoken_prefix": _rate(len(spoken), len(routed)),
        "false_tool": _rate(sum(1 for j in false_cases if j.turn.any_tool()), len(false_cases)),
        "false_tool_cases": [j.meta.get("case_id") for j in false_cases if j.turn.any_tool()],
        "repairs": {
            "n": len(repaired),
            "outcomes": dict(sorted(outcomes.items())),
            "ms": _dist(repair_ms),
        },
        "action_latency_ms": _dist(action),
        "q12_shown": q12_rate(True, "shown"),
        "q12_all_ui_valid": q12_rate(True, "all_ui_valid"),
        "q12_negative_shown": q12_rate(False, "shown"),
        "q12_negative_all_ui_valid": q12_rate(False, "all_ui_valid"),
        "q12_gates": {
            group: {arm: q12_counts(positive, arm) for arm in ("on", "off")}
            for group, positive in (("positive", True), ("negative", False))
        },
        "q12_cases": q12_cases,
        "q12_inconsistent": q12_inconsistent,
        "first_canvas_ms": first_canvas,
        "llm_done_ms": {arm: _dist(spans_of(arm, "llm.done")) for arm in ("on", "off")},
        "prefix_hit": {arm: pooled(prefix_rows[arm]) for arm in ("on", "off")},
        "prefix_hit_cases": prefix_rows,
        **b11,
        "prefill_wait_ms": {
            arm: _dist(
                [
                    j.meta["prefill_wait_ms"]
                    for j in arm_cases(arm)
                    if isinstance(j.meta.get("prefill_wait_ms"), int | float)
                ]
            )
            for arm in ("on", "off")
        },
        "startup_ms": startup,
        "stale_terminals": {"dropped": stale_dropped, "reached_machine": len(reached)},
        "stale_reached": reached,
        "voiced_literals": voiced,
        "execute_on_interrupted": on_interrupted,
        "execute_before_drain": before_drain,
    }


def _prefix_gate(report: Mapping[str, Any]) -> bool:
    """H-T96 (T-0096): B11 as written, per tools-on case -- a verified measurement, S > 0, Hs >= floor(S/32)*32 and
    Hs/S >= 9/10 (``holds`` of its ``prefix_b11_cases`` row). No tools-on case, an unread case, or a run that predates
    H-T96 fails; the whole request (J2) is never a fallback."""
    if report.get("prefix_rule") != PREFIX_RULE:
        return False
    rows = ((report.get("prefix_b11_cases") or {}).get("on")) or []
    return bool(rows) and all(r.get("holds") is True for r in rows)


def _ratio_gate(report: Mapping[str, Any]) -> bool:
    fc = report.get("first_canvas_ms") or {}
    p_on = (fc.get("on") or {}).get("p50")
    p_off = (fc.get("off") or {}).get("p50")
    if not isinstance(p_on, int | float) or not isinstance(p_off, int | float) or p_off <= 0:
        return False
    return Fraction(p_on) <= FIRST_CANVAS_MAX_RATIO * Fraction(p_off)


def _false_tool_gate(report: Mapping[str, Any]) -> bool:
    rate = report.get("false_tool") or {}
    k, n = rate.get("k"), rate.get("n")
    if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
        return False
    return Fraction(k, n) <= FALSE_TOOL_MAX


def gates(report: Mapping[str, Any]) -> dict[str, bool]:
    """Criterion 2 and the amendment, approved unchanged at H-T93: exactly ``GATE_KEYS``; ``True`` holds."""
    stale = report.get("stale_terminals") or {}
    return {
        "run_valid": report.get("void") == [],
        "no_literal_voiced": report.get("voiced_literals") == [],
        "no_execute_on_interrupted": report.get("execute_on_interrupted") == [],
        "prefix_hit_on": _prefix_gate(report),
        "first_canvas_ratio": _ratio_gate(report),
        "false_tool_rate": _false_tool_gate(report),
        "no_execute_before_drain": report.get("execute_before_drain") == [],
        "no_stale_terminal_reached_machine": stale.get("reached_machine") == 0,
    }


def _kn(rate: Mapping[str, Any] | None) -> str:
    if not rate:
        return "-"
    return f"{rate.get('k')}/{rate.get('n')}"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _dist_cells(d: Mapping[str, Any] | None) -> str:
    d = d or {}
    return f"n {d.get('n', 0)}, p50 {_fmt(d.get('p50'))}, p90 {_fmt(d.get('p90'))}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """report.md: the gates, then criterion 3's numbers beside the eval's (dev split, T-0091)."""
    g = gates(report)
    commits = ", ".join(f"`{c}`" for c in report.get("git_commit") or []) or "-"
    out = [f"# T-0093 live tool-call eval, run at {commits}", ""]
    out.append(
        "Dry run; nothing executed (`tools = dry_run`: every call goes to `DryRunToolExecutor`)."
    )
    out.append("")
    regime = report.get("regime")
    if isinstance(regime, Mapping):
        out += ["## Regime", ""]
        out += [f"- {k}: `{v}`" for k, v in regime.items()]
        out.append("")
    void = report.get("void") or []
    out += ["## Void", ""]
    out += [f"- {v}" for v in void] if void else ["- none: the run is valid"]
    incomplete = report.get("incomplete") or {}
    out += [
        "",
        "## Errored and incomplete cases (J19; both arms)",
        "",
        f"- errored cases (outcome error): {incomplete.get('errored', 0)}",
        f"- timed-out cases: {incomplete.get('timed_out', 0)}",
        f"- not-idle cases (engine busy before the case; reported, not void): "
        f"{incomplete.get('not_idle', 0)}",
    ]
    out += ["", "## Hard gates (criterion 2 and the amendment; H-T93)", ""]
    out += ["| gate | holds | reading |", "|---|---|---|"]
    fc = report.get("first_canvas_ms") or {}
    stale = report.get("stale_terminals") or {}
    b11_on = (report.get("prefix_b11_cases") or {}).get("on") or []
    if report.get("prefix_rule") != PREFIX_RULE or any(
        r.get("reading") == PREDATES for r in b11_on
    ):
        prefix_reading = (
            f"B11-prefix (H-T96): {PREDATES} (the whole request, J2, is reported, not gated)"
        )
    else:
        failing = [r for r in b11_on if r.get("holds") is not True]
        listed = ", ".join(
            f"{r.get('case_id')} (S {_fmt(r.get('S'))}, Hs {_fmt(r.get('hit_on_prefix'))}, allowed miss "
            f"{_fmt(r.get('allowed_miss'))}{'' if r.get('holds') is False else ', unread'})"
            for r in failing
        )
        prefix_reading = (
            f"B11-prefix (H-T96): {len(failing)} tools-on case(s) failing or unread"
            + (f": {listed}" if listed else "")
            + ("" if b11_on else "; no tools-on case")
        )
    readings = {
        "run_valid": f"{len(void)} void reason(s)",
        "no_literal_voiced": f"{len(report.get('voiced_literals') or [])} offending TtsSpeak field(s)",
        "no_execute_on_interrupted": f"{len(report.get('execute_on_interrupted') or [])} ToolExecute",
        "prefix_hit_on": prefix_reading,
        "first_canvas_ratio": f"p50 on {_fmt((fc.get('on') or {}).get('p50'))} / off "
        f"{_fmt((fc.get('off') or {}).get('p50'))} = {_fmt(fc.get('ratio'))} (bar <= 1.10)",
        "false_tool_rate": f"{_kn(report.get('false_tool'))} {report.get('false_tool_cases') or []} (bar <= 1/13)",
        "no_execute_before_drain": f"{len(report.get('execute_before_drain') or [])} ToolExecute",
        "no_stale_terminal_reached_machine": f"{stale.get('reached_machine')} reached, "
        f"{stale.get('dropped')} dropped",
    }
    for key in GATE_KEYS:
        out.append(f"| {key} | {'yes' if g.get(key) else '**NO**'} | {readings[key]} |")
    out += ["", "## Reported against the eval (criterion 3; not gated)", ""]
    out += ["| metric | live, tools on | eval (dev) |", "|---|---|---|"]
    out.append(
        f"| native route recall | {_kn(report.get('route_recall'))} | {EVAL_REF['route_recall']} |"
    )
    out.append(
        f"| draft valid (not barged in) | {_kn(report.get('draft_valid'))} | {EVAL_REF['draft_valid']} |"
    )
    out.append(
        f"| draft exact, score.exact (not barged in) | {_kn(report.get('draft_exact'))} | "
        f"{EVAL_REF['draft_exact']} |"
    )
    out.append(f"| query_problem (routed google_search) | {_kn(report.get('query_problem'))} | - |")
    out.append(
        f"| spoken prefix (routed) | {_kn(report.get('spoken_prefix'))} | {EVAL_REF['spoken_prefix']} |"
    )
    barge = report.get("barge_in") or {}
    out += [
        "",
        f"Barge-ins delivered: {barge.get('n', 0)} ({barge.get('cases') or []}); their drafts valid "
        f"{_kn(barge.get('draft_valid'))}, exact {_kn(barge.get('draft_exact'))}; executed {barge.get('executed', 0)}.",
    ]
    repairs = report.get("repairs") or {}
    out += [
        "",
        "## Repairs and action latency",
        "",
        f"- repairs fired: {repairs.get('n', 0)}; outcomes by final status: {repairs.get('outcomes') or {}}",
        f"- `tool.repair` ms: {_dist_cells(repairs.get('ms'))}",
        f"- action latency (`tool.execute` start - pass 1 `llm.done` end) ms: "
        f"{_dist_cells(report.get('action_latency_ms'))}",
        "",
        "## Q12 surface uptake (J1)",
        "",
        "| | tools on | tools off |",
        "|---|---|---|",
    ]
    q12_gates = report.get("q12_gates") or {}

    def gate_cells(cell: Mapping[str, Any] | None) -> str:
        cell = cell or {}
        return " / ".join(_fmt(cell.get(c)) for c in ("gated", "emitted", "failed", "not_gated"))

    for label, name in (
        (
            "surface shown (primary; Q12 / T-0055 `surface_accepted`; component count not recorded)",
            "q12_shown",
        ),
        ("every ui commit valid (secondary; `extras.ui_valid`)", "q12_all_ui_valid"),
    ):
        rate = report.get(name) or {}
        out.append(f"| {label} | {_kn(rate.get('on'))} | {_kn(rate.get('off'))} |")
    positive_gates = q12_gates.get("positive") or {}
    out.append(
        "| ui.gate on positive cases: gated / emitted / failed / not gated | "
        f"{gate_cells(positive_gates.get('on'))} | {gate_cells(positive_gates.get('off'))} |"
    )
    for label, name in (
        ("negative control, surface shown", "q12_negative_shown"),
        ("negative control, every ui commit valid", "q12_negative_all_ui_valid"),
    ):
        rate = report.get(name) or {}
        out.append(f"| {label} | {_kn(rate.get('on'))} | {_kn(rate.get('off'))} |")
    negative_gates = q12_gates.get("negative") or {}
    out.append(
        "| ui.gate on negative controls: gated / emitted / failed / not gated | "
        f"{gate_cells(negative_gates.get('on'))} | {gate_cells(negative_gates.get('off'))} |"
    )
    out += [
        "",
        "Shown: the turn's own log has a `ui.gate` record with `gated` and `surface_emitted` true (the session "
        "publishes on success and never retracts). Every ui commit valid: the session's AND over the turn's gated "
        "ui commits. Not comparable with T-0055: `ui.gate` records no component count and no `root`. Reported, "
        "not gated.",
    ]
    mismatched = report.get("q12_inconsistent") or []
    if mismatched:
        listed = ", ".join(
            f"{e.get('case_id')} ({e.get('arm')})" if isinstance(e, Mapping) else str(e)
            for e in mismatched
        )
        out += [
            "",
            f"`extras.ui_valid` inconsistent with the `ui.gate` log (reported, not voided): {listed}",
        ]
    done = report.get("llm_done_ms") or {}
    wait = report.get("prefill_wait_ms") or {}
    hit = report.get("prefix_hit") or {}
    startup = report.get("startup_ms") or {}
    b11 = report.get("prefix_b11") or {}
    measured = {
        arm: [
            r
            for r in ((report.get("prefix_b11_cases") or {}).get(arm) or [])
            if r.get("holds") is not None
        ]
        for arm in ("on", "off")
    }

    def beyond(arm: str) -> str:
        rows = measured[arm]
        return str(sum(r["hit_beyond_prefix"] for r in rows)) if rows else "-"

    def prewarmed(arm: str) -> str:
        shares = [
            r["prefill_prewarmed"] for r in measured[arm] if r.get("prefill_prewarmed") is not None
        ]
        return _fmt(_percentile(shares, 50))

    def tail(arm: str) -> str:
        tails = [r["prefill_tail"] for r in measured[arm] if r.get("prefill_tail") is not None]
        return f"min {min(tails)}, max {max(tails)}" if tails else "-"

    out += [
        "",
        "## Latency, prefix cache, startup (typed turns: relative to `llm.start`, not B3)",
        "",
        "| | tools on | tools off |",
        "|---|---|---|",
        f"| `llm.first_canvas` ms | {_dist_cells(fc.get('on'))} | {_dist_cells(fc.get('off'))} |",
        f"| `llm.done` ms | {_dist_cells(done.get('on'))} | {_dist_cells(done.get('off'))} |",
        f"| pass 1's wait for its prefill, ms (inside `llm.first_canvas`) | {_dist_cells(wait.get('on'))} | "
        f"{_dist_cells(wait.get('off'))} |",
        f"| B11-prefix pooled ΣHs/ΣS (H-T96; Hs = min(H, S)) | {_fmt(b11.get('on'))} | {_fmt(b11.get('off'))} |",
        "| prefix hit, whole request (J2), reported, not gated since H-T96: pooled Σhit/Σquery | "
        f"{_fmt(hit.get('on'))} | {_fmt(hit.get('off'))} |",
        f"| hit beyond the prefix (H - Hs), tokens, total | {beyond('on')} | {beyond('off')} |",
        f"| prefill pre-warmed share (Hp/P), p50 | {prewarmed('on')} | {prewarmed('off')} |",
        f"| prefill tail (P - S), tokens | {tail('on')} | {tail('off')} |",
        f"| session startup ms, p50 | {_fmt(startup.get('on'))} | {_fmt(startup.get('off'))} |",
        "",
        f"Stale terminals: {stale.get('dropped')} dropped, {stale.get('reached_machine')} reached the machine.",
        "",
    ]
    return "\n".join(out)


# =====================================================================================================
# Refusals (no socket)
# =====================================================================================================


def check_loopback(base_url: str) -> None:
    """``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1``; any name (``localhost`` too) is refused without being
    resolved. ``ValueError`` otherwise."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {base_url!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"base_url host {host!r} is not a loopback IP literal") from None
    ok = (
        addr in ipaddress.ip_network("127.0.0.0/8")
        if addr.version == 4
        else addr == ipaddress.ip_address("::1")
    )
    if not ok:
        raise ValueError(f"base_url host {host!r} is not in 127.0.0.0/8 or ::1")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str  # tool | no_tool | plain | q12
    text: str
    gt: dict[str, Any] | None = None
    screen: str | None = None
    screen_size: tuple[int, int] | None = None
    expect_surface: bool | None = None
    memory_captions: tuple[str, ...] = ()


def load_cases() -> list[Case]:
    """The 60 cases in run order. ``ValueError`` on a file that is not the pinned one or a split of the wrong size."""
    for path, pinned in ((SET_DIR / "cases.jsonl", CASES_SHA256), (Q12_PATH, Q12_SHA256)):
        got = _sha256(path)
        if got != pinned:
            raise ValueError(f"{path} has sha256 {got}, not the pinned {pinned}")
    rows = [r for r in _read_jsonl(SET_DIR / "cases.jsonl") if r.get("split") == "dev"]
    tool = [r for r in rows if r.get("kind") == "tool"]
    no_tool = [r for r in rows if r.get("kind") == "no_tool"]
    if len(tool) != DEV_TOOL_CASES or len(no_tool) != DEV_NO_TOOL_CASES:
        raise ValueError(
            f"the dev split has {len(tool)} tool and {len(no_tool)} no-tool cases, "
            f"not {DEV_TOOL_CASES} and {DEV_NO_TOOL_CASES}"
        )
    out: list[Case] = []
    for r in [*tool, *no_tool]:
        size = r.get("screen_size")
        out.append(
            Case(
                case_id=str(r["id"]),
                kind=str(r["kind"]),
                text=str(r["user"]),
                gt=r.get("gt"),
                screen=r.get("screen"),
                screen_size=(int(size[0]), int(size[1])) if size else None,
            )
        )
    for r in _read_jsonl(Q12_PATH):
        out.append(
            Case(
                case_id=str(r["id"]),
                kind="q12",
                text=str(r["request"]),
                expect_surface=bool(r["expect_surface"]),
                memory_captions=tuple(str(c) for c in r.get("memory_captions") or ()),
            )
        )
    out += [Case(case_id=cid, kind="plain", text=text) for cid, text in PLAIN_QUESTIONS]
    missing = sorted(set(BARGE_IN_CANDIDATES) - {c.case_id for c in out if c.kind == "tool"})
    if missing:
        raise ValueError(f"barge-in candidates not among the dev tool cases: {missing}")
    return out


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """J20 (as P15): a 3xx is never followed (urllib raises it as an ``HTTPError``, read here as unavailable)."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _get_text(url: str, timeout_s: float = 10.0) -> str | None:
    """The body of a 200 GET, or ``None``: refused, redirected, cut short (J20: ``http.client`` errors too)."""
    try:
        with _OPENER.open(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            body: bytes = resp.read()
            return body.decode("utf-8", errors="replace")
    except (OSError, ValueError, http.client.HTTPException):
        return None


def read_metrics(base_url: str) -> dict[str, int] | None:
    """The four counters/gauges this harness reads off ``/metrics`` (summed over label sets); ``None`` when the
    endpoint or a counter is unavailable. Reading it spends nothing (gpu-services rule)."""
    text = _get_text(_root(base_url) + "/metrics")
    if text is None:
        return None
    out: dict[str, int] = {}
    for name in (METRIC_QUERIES, METRIC_HITS, METRIC_RUNNING, METRIC_WAITING):
        pat = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        vals = [float(m.group(1)) for m in re.finditer(pat, text, re.MULTILINE)]
        if not vals and name in (METRIC_QUERIES, METRIC_HITS):
            return None
        out[name] = round(sum(vals))
    return out


class _MeteredLlm:
    """The session's ``LlmBackend``: the real ``VllmClient``, plus J2's measurement around pass 1.

    - ``prefill`` runs the client's prefill in its own task, so the session's end-of-turn ``PrefillAbort`` (task cancel
      plus ``abort``) cannot cut it short: the abort is deferred and recorded, and the prefill finishes.
    - Pass 1's ``generate`` (``cd is None``) waits for every prefill still running, reads ``/metrics``, streams, and
      reads ``/metrics`` again. The repair (``cd`` set) and ``tokenize`` go straight through.
    """

    def __init__(self, inner: VllmClient, base_url: str) -> None:
        self.inner = inner
        self.base_url = base_url
        self._prefills: dict[str, asyncio.Task[PrefillHandle]] = {}
        self.deferred_aborts: list[str] = []
        #: request ids whose abort reached the client, and prefills that ran to their end without one
        self._forwarded: set[str] = set()
        self._finished: set[str] = set()
        self.generates: list[dict[str, Any]] = []
        #: H-T96: every prefill's prompt by request id, in start order (the ``/tokenize`` calls after close)
        self.prefill_prompts: dict[str, Prompt] = {}
        self.max_inflight = 0
        self._inflight = 0

    def _enter(self) -> None:
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)

    def _leave(self) -> None:
        self._inflight -= 1

    async def _run_prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        self._enter()
        try:
            handle = await self.inner.prefill(prefix, request_id)
        finally:
            self._leave()
        # ``VllmClient.prefill`` returns normally when it was aborted, so a return alone is not a finished prefill:
        # it finished on its own only if no abort for it ever reached the client (test-author finding, J12/J19).
        if request_id not in self._forwarded:
            self._finished.add(request_id)
        return handle

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        self.prefill_prompts[request_id] = prefix
        task = asyncio.ensure_future(self._run_prefill(prefix, request_id))
        self._prefills[request_id] = task
        return await asyncio.shield(task)

    def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: CdRequest | None = None,
    ) -> AsyncIterator[Commit]:
        return self._generate(prompt, request_id, options, xargs, cd)

    async def _generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None,
        xargs: Mapping[str, str] | None,
        cd: CdRequest | None,
    ) -> AsyncIterator[Commit]:
        row: dict[str, Any] = {"request_id": request_id, "pass1": cd is None}
        self.generates.append(row)
        if cd is None:
            # H-T96: pass 1's prompt and its own prefill (the last one started before it), for ``/tokenize`` after close
            row["prompt"] = prompt
            row["prefill_request_id"] = next(reversed(self._prefills), None)
            started = time.perf_counter()
            pending = [t for t in self._prefills.values() if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=PREFILL_WAIT_S)
            row["prefill_wait_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            row["prefills"] = len(self._prefills)
            row["prefill_done"] = self.prefills_finished()
            row["m1"] = await asyncio.to_thread(read_metrics, self.base_url)
        self._enter()
        try:
            async for commit in self.inner.generate(prompt, request_id, options, xargs, cd=cd):
                yield commit
        finally:
            self._leave()
        if cd is None:
            row["m2"] = await asyncio.to_thread(read_metrics, self.base_url)

    async def tokenize(self, text: str) -> list[int]:
        return await self.inner.tokenize(text)

    def prefills_finished(self) -> bool:
        """J12: there was a prefill, and every prefill so far ran to its end on its own -- not cancelled, not failed,
        and never aborted (an aborted ``VllmClient.prefill`` returns normally, so the return is not the evidence)."""
        return bool(self._prefills) and all(
            t.done() and not t.cancelled() and t.exception() is None and rid in self._finished
            for rid, t in self._prefills.items()
        )

    async def abort(self, request_id: str) -> None:
        task = self._prefills.get(request_id)
        if task is not None and not task.done():
            self.deferred_aborts.append(request_id)
            return
        self._forwarded.add(request_id)
        await self.inner.abort(request_id)

    async def aclose(self) -> None:
        live = [t for t in self._prefills.values() if not t.done()]
        for task in live:
            task.cancel()
        await asyncio.gather(*self._prefills.values(), return_exceptions=True)
        await self.inner.aclose()

    def prefix_tokens(self) -> tuple[int | None, int | None]:
        """J2: Σ(hits) and Σ(queries) of pass 1's ``/metrics`` deltas; ``(None, None)`` when none was read."""
        hits = queries = 0
        seen = False
        for row in self.generates:
            m1, m2 = row.get("m1"), row.get("m2")
            if not row.get("pass1") or m1 is None or m2 is None:
                continue
            seen = True
            hits += m2[METRIC_HITS] - m1[METRIC_HITS]
            queries += m2[METRIC_QUERIES] - m1[METRIC_QUERIES]
        return (hits, queries) if seen else (None, None)


class _PlaybackOnly:
    """``FakeAudioIO`` with only the ``AudioIO`` playback seams: no ``stt_blocks``, so no capture pump runs (typed turns
    need none, and the fake's capture side would spin at real time for nothing)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        played: int = await self.inner.flush()
        return played

    def played_ms(self) -> float:
        return float(self.inner.played_ms())

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def playback_events(self) -> Any:
        return self.inner.playback_events()

    def is_rendering(self, t_ms: float) -> bool:
        return bool(self.inner.is_rendering(t_ms))

    def capture(self) -> Any:
        return self.inner.capture()


class _CaseMemory:
    """No recollections, except a Q12 case's captions as hits (T-0055's ``_memory``); never an uncaptioned episode,
    so the idle captioner sends no request."""

    def __init__(self, hits: Sequence[MemoryHit] = ()) -> None:
        self.hits = tuple(hits)
        self._next = 0

    async def ingest(self, episode: Episode) -> int:
        self._next += 1
        return self._next

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]:
        return self.hits

    async def set_caption(self, episode_id: int, caption: str) -> None:
        return None

    async def forget_since(self, t0_ms: int) -> int:
        return 0

    async def next_uncaptioned(self) -> Episode | None:
        return None


def _fake(module: str, name: str) -> Any:
    """A class from ``tests/fakes`` (imported by name: the fakes are not typed to --strict)."""
    return getattr(importlib.import_module(f"tests.fakes.{module}"), name)


def _memory_hits(case: Case) -> list[MemoryHit]:
    from local_astra.contracts.memory import Episode, MemoryHit
    from local_astra.contracts.perception import Source

    return [
        MemoryHit(
            episode=Episode(
                ts_ms=1_000 + i,
                source=Source.SCREEN,
                thumbnail_path=f"/tmp/t0055-{i}.jpg",
                phash=f"c{i}" * 8,
                window_title="Exercises and Problems in Calculus",
                caption=caption,
            ),
            score=0.9 - 0.1 * i,
            matched=("calculus",),
        )
        for i, caption in enumerate(case.memory_captions)
    ]


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Profile
    manifest: dict[str, Any]
    barge_ins: int = 0


#: How long ``/metrics`` may stay unreadable before the engine counts as unreachable for this case (J19).
UNREACHABLE_AFTER_S = 3.0


async def _wait_idle(base_url: str) -> tuple[bool, bool, dict[str, int] | None]:
    """``(idle, reachable, last)`` before a case: idle once ``/metrics`` shows nothing running or waiting; unreachable
    when ``/metrics`` cannot be read at all for ``UNREACHABLE_AFTER_S``. ``last`` is the last reading taken (H-T96's
    m0: the one that showed the engine idle, when it did), ``None`` when none could be read."""
    started = time.monotonic()
    deadline = started + QUIESCE_S
    reached = False
    last: dict[str, int] | None = None
    while True:
        m = await asyncio.to_thread(read_metrics, base_url)
        reached = reached or m is not None
        last = m if m is not None else last
        if m is not None and m[METRIC_RUNNING] + m[METRIC_WAITING] == 0:
            return True, True, m
        if not reached and time.monotonic() - started >= UNREACHABLE_AFTER_S:
            return False, False, last
        if time.monotonic() >= deadline:
            return False, reached, last
        await asyncio.sleep(0.2)


def _tokenize_chat(base_url: str, body: Mapping[str, Any]) -> list[int] | None:
    """H-T96 (criterion 1c): the engine's ids for a chat request's prompt -- ``POST /tokenize`` at the server root, in
    chat form: exactly the kept keys of ``body`` (``VllmClient.request_body``'s) and ``add_special_tokens: false``.
    Loopback through the unproxied, non-redirecting opener; ``None`` on any failure. Called after the session is
    closed, never on a timed path."""
    kept = {k: body[k] for k in TOKENIZE_KEEP if k in body}
    kept["add_special_tokens"] = False
    request = urllib.request.Request(
        _root(base_url) + "/tokenize",
        data=json.dumps(kept).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=60.0) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    return _id_list(payload.get("tokens") if isinstance(payload, dict) else None)


def _cache_block_size(base_url: str) -> str | None:
    """The ``block_size`` label of ``vllm:cache_config_info`` (H-T96: reported only, never A)."""
    text = _get_text(_root(base_url) + "/metrics")
    found = re.search(
        r'^vllm:cache_config_info\{[^}]*\bblock_size="([^"]*)"', text or "", re.MULTILINE
    )
    return found.group(1) if found else None


async def _barge(session: Any, stop: asyncio.Event) -> bool:
    """Submit one ``UserSpeechOnset`` once the turn is SPEAKING with pass 1 done (or its repair running)."""
    from local_astra.contracts import events as E
    from local_astra.contracts.turn import TurnState

    while not stop.is_set():
        state = session.state
        if state.state is TurnState.SPEAKING and (state.llm_done or state.tool is not None):
            now = session.clock.now_ms
            session.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
            return True
        await asyncio.sleep(0.005)
    return False


async def _one(run: _Run, case: Case, arm: str, index: int) -> bool:
    """Run one (case, arm) in a fresh session and record it; ``False`` when the engine could not be reached."""
    import httpx

    from local_astra import cli
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.telemetry.session_log import SessionLog

    idle, reachable, m0 = await _wait_idle(run.base_url)
    sessions_dir = run.out / "sessions"
    started = time.perf_counter()
    kit = detokenizer = None
    if arm == "on":
        kit, detokenizer = cli.load_tool_kit(run.profile)
    inner = VllmClient(
        VllmEndpoint(base_url=run.base_url),
        transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
        detokenizer=detokenizer,
    )
    llm = _MeteredLlm(inner, run.base_url)
    clock = SessionClock()
    frames = []
    if case.screen is not None and case.screen_size is not None:
        jpeg = (SET_DIR / case.screen).read_bytes()
        frames.append(
            Frame(
                source=Source.SCREEN,
                ts_ms=0,
                width=case.screen_size[0],
                height=case.screen_size[1],
                jpeg=jpeg,
                phash=hashlib.sha256(jpeg).hexdigest()[:16],
            )
        )
    log = SessionLog(sessions_dir)
    session = Session(
        TurnMachine(TurnPolicy(thinking=run.profile.diffusion.thinking)),
        llm,
        _fake("stt", "FakeStt")(),
        _fake("tts", "FakeTts")(),
        _PlaybackOnly(_fake("audio", "FakeAudioIO")(clock)),
        _fake("perception", "FakePerception")(frames),
        _CaseMemory(_memory_hits(case)),
        clock=clock,
        config=SessionConfig(
            run.out / "data",
            turns_path=run.out / "turns.jsonl",
            profile=run.profile,
            audio_path="null",
            source=Source.SCREEN,
        ),
        session_log=log,
        log_dir=sessions_dir,
        tool_kit=kit,
    )
    await session.start()
    startup_ms = round((time.perf_counter() - started) * 1000.0, 1)
    attempt = (
        arm == "on"
        and case.kind == "tool"
        and case.case_id in BARGE_IN_CANDIDATES
        and run.barge_ins < BARGE_IN_N
    )
    session.turn_written.clear()
    stop = asyncio.Event()
    barge = asyncio.ensure_future(_barge(session, stop)) if attempt else None
    session.submit(E.UserText(t_ms=clock.now_ms, text=case.text))
    timed_out = False
    try:
        await asyncio.wait_for(session.turn_written.wait(), timeout=CASE_TIMEOUT_S)
    except TimeoutError:
        timed_out = True
    stop.set()
    submitted = bool(await barge) if barge is not None else False
    await session.aclose()
    await llm.aclose()

    line: dict[str, Any] | None = None
    turns_path = run.out / "turns.jsonl"
    if turns_path.is_file():
        mine = [r for r in _read_jsonl(turns_path) if r.get("session_id") == session.session_id]
        line = mine[0] if mine else None
    # J18: delivered on a routed tool turn (the turn's call was published, so its line carries it)
    delivered = bool(
        line and line.get("outcome") == "interrupted" and line.get("tool") is not None and submitted
    )
    if delivered:
        run.barge_ins += 1
    hit, query = llm.prefix_tokens()
    pass1 = [g for g in llm.generates if g.get("pass1")]
    # --- H-T96 (T-0096 criterion 1): after close, outside every timed span ---------------------------------------
    first = pass1[0] if pass1 else {}
    m1 = first.get("m1")
    p_query = p_hit = None
    if m0 is not None and m1 is not None:
        p_query = m1[METRIC_QUERIES] - m0[METRIC_QUERIES]
        p_hit = m1[METRIC_HITS] - m0[METRIC_HITS]
    prefill_rid = first.get("prefill_request_id")
    prefill_prompt = llm.prefill_prompts.get(prefill_rid) if prefill_rid is not None else None
    gen_prompt = first.get("prompt")
    prefill_ids = generate_ids = None
    if prefill_prompt is not None:
        body = inner.request_body(prefill_prompt, str(prefill_rid))
        prefill_ids = await asyncio.to_thread(_tokenize_chat, run.base_url, body)
    if gen_prompt is not None:
        body = inner.request_body(gen_prompt, str(first.get("request_id")))
        generate_ids = await asyncio.to_thread(_tokenize_chat, run.base_url, body)
    with (run.out / SIDECAR).open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "arm": arm,
                    "session_id": session.session_id,
                    "prefill_request_id": prefill_rid,
                    "generate_request_id": first.get("request_id"),
                    "prefill_ids": prefill_ids,
                    "generate_ids": generate_ids,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    run.manifest["sessions"].append(
        {
            "arm": arm,
            "session_id": session.session_id,
            "log": log.path.name,
            "startup_ms": startup_ms,
        }
    )
    run.manifest["cases"].append(
        {
            "case_id": case.case_id,
            "kind": case.kind,
            "arm": arm,
            "order": index,
            "session_id": session.session_id,
            "session_log": log.path.name,
            "turn_id": line.get("turn_id") if line else None,
            "gt": case.gt,
            "barge_in": delivered,
            "barge_in_attempted": submitted,
            "expect_surface": case.expect_surface,
            "prefix_hit_tokens": hit,
            "prefix_query_tokens": query,
            "prefill_query_tokens": p_query,
            "prefill_hit_tokens": p_hit,
            "prefills_before_pass1": first.get("prefills"),
            "prefill_wait_ms": pass1[-1].get("prefill_wait_ms") if pass1 else None,
            "prefill_done": pass1[-1].get("prefill_done") if pass1 else None,
            "deferred_prefill_aborts": len(llm.deferred_aborts),
            "max_inflight": llm.max_inflight,
            "engine_idle_before": idle,
            "engine_reachable": reachable,
            "timed_out": timed_out,
            "failures": [repr(f) for f in session.failures],
        }
    )
    _write_json(run.out / "manifest.json", run.manifest)
    _say(
        f"{index:02d} {case.case_id} [{arm}] outcome={line.get('outcome') if line else None} "
        f"tool={(line or {}).get('tool', {}) and (line or {}).get('tool', {}).get('status')} "
        f"hit={hit}/{query} startup={startup_ms}ms" + (" BARGE-IN" if delivered else "")
    )
    return reachable


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def _run_all(run: _Run, cases: Sequence[Case]) -> None:
    """Every case in both arms, alternated; J19: stop after ``UNREACHABLE_STOP`` consecutive unreachable cases (the
    manifest says why, and ``build_report`` voids the run on it)."""
    unreachable = 0
    # T-0096 criterion 6 and ruling M6: the fail-fast checkpoints are case 0 (text or tools) and the first case, in
    # run order, whose prompt carries an image (image expansion)
    first_image = next(
        (i for i, c in enumerate(cases) if c.screen is not None and c.screen_size is not None),
        None,
    )
    for i, case in enumerate(cases):
        order = ("on", "off") if i % 2 == 0 else ("off", "on")
        for arm in order:
            unreachable = 0 if await _one(run, case, arm, i) else unreachable + 1
            if i in (0, first_image):
                # the checkpoint's ids must match its own query counts (5a, 5b, 5f), or nothing more runs
                side = _read_sidecar(run.out)
                meta = run.manifest["cases"][-1]
                line = side[0].get((case.case_id, arm)) if side is not None else None
                reasons = _ids_voids(meta, line)
                if reasons:
                    run.manifest["stopped"] = (
                        f"order {i} failed the H-T96 prefix verification (criterion 6, ruling M6): {'; '.join(reasons)}"
                    )
                    _write_json(run.out / "manifest.json", run.manifest)
                    _say(f"stopping: {run.manifest['stopped']}")
                    return
            if unreachable >= UNREACHABLE_STOP:
                run.manifest["stopped"] = (
                    f"the engine could not be reached for {unreachable} consecutive cases "
                    f"(last: {case.case_id} [{arm}])"
                )
                _write_json(run.out / "manifest.json", run.manifest)
                _say(f"stopping: {run.manifest['stopped']}")
                return


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    out: Path = args.out_dir or REPO / "logs" / f"t0093-live-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
        return EXIT_REFUSED
    try:
        cases = load_cases()
    except (OSError, ValueError, KeyError) as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED

    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    try:
        kit, _ = cli.load_tool_kit(profile)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(
            f"refusing: the tool repair is off ({kit.repair_off}); a run with repair off is invalid"
        )
        return EXIT_REFUSED
    if kit.tools_version[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {kit.tools_version[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = _get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    with contextlib.suppress(ValueError, KeyError, TypeError):
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if read_metrics(args.base_url) is None:
        _say(f"{_root(args.base_url)}/metrics has no prefix-cache counters; J2 cannot be measured")
        return EXIT_FAILED

    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": model,
            "engine_version": _get_text(_root(args.base_url) + "/version"),
            "versions_lock_sha256": _sha256(REPO / "services" / "versions.lock"),
            "cases_sha256": CASES_SHA256,
            "q12_sha256": Q12_SHA256,
            "tools": TOOLS_ON,
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": profile.diffusion.thinking,
            "arms": "tools off vs tools = dry_run, alternated per case (case 0 on first)",
            "n_per_case_per_arm": 1,
            "seed": None,
            "concurrency": 1,
            "input": "typed (UserText); fixture frames; fake STT/TTS (300 ms/word)/audio/perception; no audio",
            "plain_questions": [q for _, q in PLAIN_QUESTIONS],
            "barge_in": f"first {BARGE_IN_N} delivered among {list(BARGE_IN_CANDIDATES)}",
            "prefix_rule": PREFIX_RULE,
            "prefix_block_tokens": PREFIX_BLOCK_TOKENS,
            "cache_config_block_size": _cache_block_size(args.base_url),
        },
        "sessions": [],
        "cases": [],
    }
    _write_json(out / "manifest.json", manifest)
    run = _Run(base_url=args.base_url, out=out, profile=profile, manifest=manifest)
    asyncio.run(_run_all(run, cases))

    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    g = gates(report)
    for key in GATE_KEYS:
        _say(f"gate {key}: {'holds' if g[key] else 'FAILS'}")
    _say(f"written to {out}")
    return EXIT_OK if all(g.values()) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
