"""Shared helpers for the T-0080b driver rows (two-pass tool-call driver over the loopback fake). Not a test module.

API the rows assume (test-author proposal at 02df09d, "APIs confirmed" in the rulings of docs/tasks/T-0080b.md;
the field names below that the proposal did not spell out are this module's pins -- see the report):

``local_astra.eval.cd.twopass``
  ``TwoPassConfig(stage, cells, base_url, engine_log, out_dir, repo, expected_commit, calibration, set_dir, limit=None,
  ...)`` -- ``stage`` ``"dev"``/``"test"`` (the split of ``set_dir/cases.jsonl``, in file order, first ``limit``);
  ``cells`` ``["<thought>/<scope>/<arm>", ...]`` (K-D2: scope ``named``); ``calibration`` ``{arm: frozen file}``.
  ``run(cfg) -> result`` with ``exit_code``, ``artifact`` (Path), ``stopped`` (``None`` or a string). A non-loopback
  ``base_url`` raises ``ValueError`` before any socket opens.
  ``derive(rows, policy)`` -> ``select`` rows (``{"policy", "case", "tool_case", "called", "schema_valid", "exact",
  "cs", "pass2_wall_ms", "e2e_wall_ms", "steps"}``) for ``"draft_only"`` and ``"cd_on_invalid:<cell>"``, offline.
Artifact (JSONL): a regime line first (carrying the system prompt's sha256 and ``tools_v1.json``'s sha256), then one
  pass-1 row per case and one pass-2 row per (case, cell), then trailer line(s); a ``stopped`` value is written in
  the regime or a trailer when the run stops. Rows carry ``"pass"`` (1 or 2) and ``"case"`` (the case id); pass-1
  rows ``route`` (the router's kind, or a mapping with ``kind``) and ``wall_ms``; pass-2 rows ``cell``, ``requests``
  (2 when the name picker ran), ``cs``, ``error``, ``schema_valid``, ``exact``, ``wall_ms`` and
  ``prefix_hit_tokens`` (the delta of ``vllm:prefix_cache_hits_total`` around the request; ``None`` when
  ``/metrics`` has no counters -- never guessed).
Requests: pass 1 = system message (the one fixed system prompt) + user message (text, plus one
  ``image_url`` part for a case with a screen), ``chat_template_kwargs.enable_thinking`` true, ``max_tokens`` 1024,
  ``return_token_ids``, no ``tools``, no ``vllm_xargs``. T1/T2 = pass 1's body with ``vllm_xargs`` added and
  ``max_tokens`` 1024/256. T4 = pass 1's messages + ``{"role": "assistant", "content": spoken_prefix + "<cc>"}``,
  ``continue_final_message`` true, ``add_generation_prompt`` false, ``max_tokens`` 256, ``vllm_xargs``. The name
  picker's step 2 is the same body as step 1 with the picked tool's automaton.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.cd import _compile_cache as cc
from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import FakeVllmServer, Reply, cd_keys

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
CASES_FILE = SET_DIR / "cases.jsonl"
CALIBRATION = {"perstep_mar": REPO / "docs" / "cd" / "calibration-bfcl.json"}
QARGMAX_FILE = REPO / "docs" / "cd" / "calibration-bfcl-perstep_mar_qargmax.json"
ARM = "perstep_mar"
CELLS = (f"T1/named/{ARM}", f"T2/named/{ARM}", f"T4/named/{ARM}")
TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")
THOUGHTS = ("T1", "T2", "T4")
EMPTY_HEADER = "<|channel>thought\n<channel|>"
CC = "<cc>"
#: the pass-2 replies' step counter (``executed steps`` = steps - k0, k0 = 0 without a seeded draft)
P2_STEPS = 7


# --------------------------------------------------------------------------- code under test


def _install_toolcall_cache() -> ModuleType:
    """Wrap ``local_astra.decoding.toolcall.compile_toolcall`` with a memo + the on-disk cache of
    ``tests/cd/_compile_cache.py`` (same directory, key over the decoding sources and the tokenizer). The wrapper
    calls the real function on a miss; the compiler is deterministic (T-0063 criterion 9).

    It closes by delegation (``.claude/rules/tests.md``): every keyword is forwarded to the real function, and the
    memo key and disk digest hold *every* argument with the real signature's defaults resolved, so a keyword added
    later (T-0084a's ``call_format``) is forwarded and separates cache entries without editing this wrapper. A
    value that cannot be keyed (not str/int/float/bool/None) bypasses the cache and goes straight through."""
    import inspect

    tc = cs.need("local_astra.decoding.toolcall")
    if getattr(tc.compile_toolcall, "_t0080b_cache", False):
        return tc
    orig = tc.compile_toolcall
    sig = inspect.signature(orig)
    amod = cs.need(f"{cs.VENDOR_PKG}.automaton")
    memo: dict[tuple[str, ...], Any] = {}
    lock = threading.Lock()

    def compile_toolcall(tools: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind(tools, *args, **kwargs)
        except TypeError:
            return orig(tools, *args, **kwargs)  # let the real function raise its own error
        bound.apply_defaults()
        named = {k: v for k, v in bound.arguments.items() if k != next(iter(sig.parameters))}
        if args or not all(
            isinstance(v, (str, int, float, bool, type(None))) for v in named.values()
        ):
            return orig(tools, *args, **kwargs)
        key = (str(tools.version), *(f"{k}={named[k]!r}" for k in sorted(named)))
        with lock:
            hit = memo.get(key)
        if hit is not None:
            return hit
        path = None
        if cc.disk_enabled():
            digest = hashlib.sha256(
                "\x1f".join(
                    ["toolcall", *key, cc._TOKENIZER_SHA256, cc._implementation_digest()]
                ).encode()
            ).hexdigest()
            path = cc.cache_dir() / f"toolcall-{digest}.npz"
        if path is not None and path.is_file():
            a = amod.load(str(path))
        else:
            a = orig(tools, **kwargs)
            if path is not None:
                cc._save(amod, a, path)
        with lock:
            memo[key] = a
        return a

    compile_toolcall._t0080b_cache = True  # type: ignore[attr-defined]
    compile_toolcall.__wrapped__ = orig  # type: ignore[attr-defined]
    tc.compile_toolcall = compile_toolcall
    return tc


def twopass() -> ModuleType:
    """The driver module (compile caches installed first). Its absence is an assertion, not an error."""
    cs.grammars()
    _install_toolcall_cache()
    name = "local_astra.eval.cd.twopass"
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise AssertionError(f"T-0080b not implemented: {name} does not exist") from None
    for attr in ("TwoPassConfig", "run", "derive", "select"):
        if not hasattr(m, attr):
            raise AssertionError(f"T-0080b not implemented: twopass.{attr} is missing")
    return m


@functools.cache
def toolset() -> Any:
    return cs.need("local_astra.toolcall.tools").load()


def compiled(scope: str, thought: str) -> Any:
    return _install_toolcall_cache().compile_toolcall(toolset(), scope=scope, thought=thought)


@functools.cache
def payload_index() -> Mapping[str, tuple[str, str]]:
    """Encoded automaton -> (scope, thought) for every live scope (each tool and the name picker) x thought.
    ``union`` is deliberately absent: a union payload reads as unknown (K-D2: never sent)."""
    from tests.cd import _wire_support as ws

    out: dict[str, tuple[str, str]] = {}
    for scope in (*TOOLS, "name"):
        for thought in THOUGHTS:
            out[ws.encode(compiled(scope, thought))] = (scope, thought)
    return out


def bucket_of(scope: str, thought: str) -> int:
    return int(compiled(scope, thought).n_states_bucket)


# --------------------------------------------------------------------------- the eval set


@functools.cache
def all_cases() -> tuple[Mapping[str, Any], ...]:
    return tuple(json.loads(line) for line in CASES_FILE.read_text().splitlines() if line.strip())


def split(stage: str) -> list[Mapping[str, Any]]:
    return [c for c in all_cases() if c["split"] == stage]


@functools.cache
def by_id() -> Mapping[str, Mapping[str, Any]]:
    return {c["id"]: c for c in all_cases()}


def user_text(body: Mapping[str, Any]) -> str:
    """The text of the (first) user message: a string, or the ``text`` parts of a content list."""
    for m in body.get("messages") or []:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    return ""


def image_parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]


def case_of(body: Mapping[str, Any]) -> str | None:
    """The case (either split) whose request text the body carries; ``None`` if none or ambiguous."""
    text = user_text(body)
    exact = [c["id"] for c in all_cases() if c["user"] == text.strip()]
    if len(exact) == 1:
        return str(exact[0])
    hits = [c["id"] for c in all_cases() if c["user"] in text]
    return str(hits[0]) if len(hits) == 1 else None


def ordered_args(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    order = list(toolset().parameters(name)["properties"])
    return {k: values[k] for k in order if k in values}


def call_text(name: str, values: Mapping[str, Any]) -> str:
    args = ordered_args(name, values) if name in TOOLS else dict(values)
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def gt_call(case: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """A call that is exact for the case, built from its ground truth by hand rules (bbox centre, keywords)."""
    gt = case["gt"]
    name = gt["name"]
    if name == "google_search":
        return name, {"query": " ".join(gt["keywords"])}
    if name == "click_at":
        x0, y0, x1, y1 = gt["bbox"]
        return name, {"x": (x0 + x1) // 2, "y": (y0 + y1) // 2}
    if name == "type_at":
        x0, y0, x1, y1 = gt["bbox"]
        return name, {
            "x": (x0 + x1) // 2,
            "y": (y0 + y1) // 2,
            "text": gt["text"],
            "press_enter": gt["press_enter"],
        }
    if name == "scroll":
        return name, {"direction": gt["direction"]}
    if name == "press_key":
        return name, {"combo": gt["combo"]}
    raise AssertionError(name)


# --------------------------------------------------------------------------- scripted model


def thought_of_body(body: Mapping[str, Any]) -> str:
    """The thought variant a pass-2 body has the *shape* of (criterion 3): T4 continues the assistant prefix;
    T1 regenerates the thought (``max_tokens`` 1024); T2 forces it empty (256)."""
    if body.get("continue_final_message"):
        return "T4"
    return {1024: "T1", 256: "T2"}.get(body.get("max_tokens"), "?")  # type: ignore[arg-type]


@dataclasses.dataclass
class Sent:
    index: int
    kind: str  # "p1" | "p2"
    case: str | None
    scope: str | None  # pass 2: from the payload ("?" = unknown payload)
    thought: str | None  # pass 2: from the payload
    shape: str | None  # pass 2: from the body
    body: Mapping[str, Any]
    reply: Reply


@dataclasses.dataclass
class Scenario:
    """Scripted pass-1 answers and pass-2 outputs.

    Pass 1 (default, tool case): ``<|channel>thought\\n...<channel|>`` + ``spoken(case)`` + ``<cc>`` + the case's exact
    call. ``pass1`` overrides the whole text per case; a no-tool case without an override answers with no tag.
    Pass 2: the in-language output for the payload's (scope, thought): the header the variant needs, then the
    case's exact call (``name`` scope: ``{"name": "<the case's tool>"``). ``draft_tool`` gives the call/tool for a
    case without ground truth (a no-tool case answered with a tag). ``pass2`` overrides a reply by
    ``(case, thought)`` (the first matching request) -- for stop rows.
    """

    pass1: dict[str, str] = dataclasses.field(default_factory=dict)
    draft_tool: dict[str, tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=dict)
    pass2: dict[tuple[str, str], Reply] = dataclasses.field(default_factory=dict)
    pass1_reply: dict[str, Reply] = dataclasses.field(default_factory=dict)
    sent: list[Sent] = dataclasses.field(default_factory=list)
    on_first: Callable[[], None] | None = None
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    @staticmethod
    def spoken(case_id: str) -> str:
        return f"On it ({case_id}). "

    def pass1_text(self, case_id: str) -> str:
        if case_id in self.pass1:
            return self.pass1[case_id]
        case = by_id()[case_id]
        head = f"<|channel>thought\nThe user asked for {case_id}.<channel|>"
        if case["kind"] != "tool":
            return head + f"Here is my answer to {case_id}."
        name, args = gt_call(case)
        return head + self.spoken(case_id) + CC + call_text(name, args)

    def call_for(self, case_id: str) -> tuple[str, dict[str, Any]]:
        if case_id in self.draft_tool:
            return self.draft_tool[case_id]
        return gt_call(by_id()[case_id])

    def pass2_text(self, case_id: str, scope: str, thought: str) -> str:
        head = {
            "T1": f"<|channel>thought\nRe-thinking {case_id}.<channel|>{CC}",
            "T2": EMPTY_HEADER + CC,
            "T4": "",
        }[thought]
        name, args = self.call_for(case_id)
        if scope == "name":
            return head + json.dumps({"name": name})[:-1]  # '{"name": "<tool>"'
        return head + call_text(name, args)

    def __call__(self, index: int, body: Mapping[str, Any]) -> Reply:
        if index == 0 and self.on_first is not None:
            self.on_first()
        case = case_of(body)
        keys = cd_keys(body)
        if not keys:
            reply = self.pass1_reply.get(case or "") or Reply(
                text=self.pass1_text(case) if case else "", prefix_queries=5000, prefix_hits=77
            )
            with self._lock:
                self.sent.append(Sent(index, "p1", case, None, None, None, body, reply))
            return reply
        scope, thought = payload_index().get(str(keys.get("astra_cd_automaton")), ("?", "?"))
        shape = thought_of_body(body)
        override = None
        with self._lock:
            if case is not None and (case, shape) in self.pass2:
                override = self.pass2.pop((case, shape))
        if override is not None:
            reply = override
        elif case is None or scope == "?":
            reply = Reply(text="", prefix_queries=1000 + index, prefix_hits=900 + index)
        else:
            reply = Reply(
                text=self.pass2_text(case, scope, thought),
                steps=P2_STEPS,
                bucket=bucket_of(scope, thought),
                prefix_queries=1000 + index,
                prefix_hits=900 + index,
            )
        with self._lock:
            self.sent.append(Sent(index, "p2", case, scope, thought, shape, body, reply))
        return reply


def fake(scenario: Scenario, workdir: Path, **kw: Any) -> FakeVllmServer:
    return es.fake(scenario, workdir, **kw)


def run_driver(srv: FakeVllmServer, tmp_path: Path, **overrides: Any) -> Any:
    tp = twopass()
    repo, head = overrides.pop("repo", None), overrides.pop("expected_commit", None)
    if repo is None:
        repo, head = es.clean_repo(tmp_path)
    out = tmp_path / "artifacts"
    out.mkdir(exist_ok=True)
    cfg: dict[str, Any] = {
        "stage": "dev",
        "cells": list(CELLS),
        "base_url": srv.url,
        "engine_log": srv.engine_log,
        "out_dir": out,
        "repo": repo,
        "expected_commit": head,
        "calibration": dict(CALIBRATION),
        "set_dir": SET_DIR,
    }
    cfg.update(overrides)
    return tp.run(tp.TwoPassConfig(**cfg))


# --------------------------------------------------------------------------- artifact


def artifact_lines(path: Path | str) -> list[dict[str, Any]]:
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    assert lines, f"{path} is empty"
    return lines


def pass_rows(path: Path | str, which: int) -> list[dict[str, Any]]:
    return [x for x in artifact_lines(path)[1:] if x.get("pass") == which]


def stopped_marker(path: Path | str) -> str | None:
    for line in artifact_lines(path):
        if line.get("stopped"):
            return str(line["stopped"])
    return None


def route_kind(row: Mapping[str, Any]) -> str:
    r = row.get("route")
    if isinstance(r, Mapping):
        return str(r.get("kind"))
    return str(r)


def cells_in_order(sent: Sequence[Sent]) -> list[str]:
    """The cells of a case's pass-2 requests in arrival order (a two-request cell counted once)."""
    out: list[str] = []
    for s in sent:
        mode = (cd_keys(s.body).get("astra_cd_mode")) or "?"
        cell = f"{s.shape}/named/{mode}"
        if not out or out[-1] != cell:
            out.append(cell)
    return out


# --------------------------------------------------------------------------- T-0084c: the native format

#: the native call markers (pinned template; ids 48 / 49 / 50 checked against the tokenizer in the T-0084c rows)
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESPONSE = "<|tool_response>"
#: T-0084c's proposed native system prompt, verbatim from docs/tasks/T-0084c.md (Implementation notes)
NATIVE_SYSTEM_PROMPT_TEXT = (
    "You are Astra, a voice assistant that can see the user's screen and act on it.\n\n"
    "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then "
    "call exactly one of the declared tools. Use at most one call per answer. When no action is needed, answer "
    "normally and call no tool.\n\n"
    "Screen coordinates x and y are integers from 0 to 999, normalised to the attached screenshot: (0, 0) is its "
    "top-left corner and (999, 999) its bottom-right corner."
)


def native_cell(cell: str) -> str:
    """``T1/named/perstep_mar`` -> ``native/T1/named/perstep_mar`` (criterion 1's cell name)."""
    return f"native/{cell}"


def compiled_native(scope: str, thought: str) -> Any:
    return _install_toolcall_cache().compile_toolcall(
        toolset(), scope=scope, thought=thought, call_format="native"
    )


@functools.cache
def native_payload_index() -> Mapping[str, tuple[str, str]]:
    """Encoded native automaton -> (scope, thought), every live scope x thought (``union`` absent, as above)."""
    from tests.cd import _wire_support as ws

    out: dict[str, tuple[str, str]] = {}
    for scope in (*TOOLS, "name"):
        for thought in THOUGHTS:
            out[ws.encode(compiled_native(scope, thought))] = (scope, thought)
    return out


def payload_format(keys: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(format, scope, thought)`` of a constrained request's automaton: ``json`` or ``native`` from the two indexes
    (``("?", "?", "?")`` for an automaton in neither)."""
    payload = str(keys.get("astra_cd_automaton"))
    if payload in payload_index():
        return ("json", *payload_index()[payload])
    if payload in native_payload_index():
        return ("native", *native_payload_index()[payload])
    return ("?", "?", "?")


def native_call(name: str, values: Mapping[str, Any]) -> str:
    """The template's own rendering of a call, ``<|tool_call>call:NAME{...}<tool_call|>`` (the oracle, never the
    implementation)."""
    from tests.cd.oracle import gemma4_template as g

    return str(g.call_text(name, dict(values)))


def subset_set_dir(dest: Path, ids: Sequence[str], *, split: str = "dev") -> Path:
    """An eval-set directory holding exactly ``ids`` (the original case lines, re-labelled ``split``, in the order
    given) with its own MANIFEST.json sha and the original screens (symlinked)."""
    dest.mkdir(parents=True, exist_ok=True)
    lines = {json.loads(x)["id"]: x for x in CASES_FILE.read_text().splitlines() if x.strip()}
    out = []
    for cid in ids:
        obj = json.loads(lines[cid])
        obj["split"] = split
        out.append(json.dumps(obj, sort_keys=True))
    raw = ("\n".join(out) + "\n").encode()
    (dest / "cases.jsonl").write_bytes(raw)
    (dest / "MANIFEST.json").write_text(
        json.dumps({"cases_sha256": hashlib.sha256(raw).hexdigest()})
    )
    screens = dest / "screens"
    if not screens.exists():
        screens.symlink_to(SET_DIR / "screens", target_is_directory=True)
    return dest
