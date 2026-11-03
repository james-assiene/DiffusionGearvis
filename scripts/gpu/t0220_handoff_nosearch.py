"""T-0220 -- the GPU check of T-0215: the shared-screen handoffs H1-H3 x {shared, typed} x n = 3 = 18 typed turns with
search off, where the live tool set no longer declares ``google_search``; report-only, at HEAD's live web-mode defaults
on loopback (deny-all resolver and connector). ``/launch`` ONLY.

    uv run python scripts/gpu/t0220_handoff_nosearch.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing.**

A1: T-0211's harness (``scripts/gpu/t0211_guidance_ab.py``) is imported read-only by path as ``A`` under a fixed
``sys.modules`` name; ``A.C`` is T-0208's harness and ``A.D`` T-0205's (one copy each). Every turn runs through
``A.C._one_turn(run, hid, arm, repeat, search_kit=nosearch_kit)`` unchanged; the probes (``A.C.run_probes``,
``A.C.probe_ph1``, ``A.C.probe_ph2``, ``A.C.probe_verdicts``), ``A.C.arm_counts``, ``A.C.build_report``'s audit,
``A.C.step1_action``, ``A.used_browser`` and ``A.no_browser_claim`` are reused as they are. No attribute of ``A``,
``A.C`` or ``A.D`` is assigned. The three harness files must equal the git blob shas pinned in ``PINNED_BLOBS``.

A2: :func:`nosearch_kit` swaps a turn's kit (the one ``cli.load_tool_kit`` gave ``A.C._one_turn``, search on) for the
tools, ``tools_version`` and ``tool_set`` of a reference kit built by ``cli.load_tool_kit(lean, search=False, **opts)``
(:func:`reference_kit`); every other field, the executor included, is the turn's own. The runner records the kit it
loaded (``tools_version`` before the seam); :func:`finish_turn` keeps that as ``tools_version_loaded`` and records the
version the seam handed the ``Session`` as ``tools_version``.

A6: :func:`evidence` reads each turn's session log after the rig exits; any violation makes the run ``void``.

A8: the two anchors (``ANCHOR``) are quoted constants recomputed from the local logs with ``A.used_browser`` and the
runners' success rules on 2026-10-31; ``logs/`` is never read at run time.

Output (``--out-dir``, default ``logs/t0220-handoff-nosearch-<sha7>/``): ``order.json``, ``manifest.json``,
``probes.json``, ``trials.jsonl``, ``steps.jsonl``, ``thoughts.jsonl``, ``turns.jsonl``, ``sessions/*.jsonl``,
``report.json``, ``report.md``. Exit: 0 measured; 1 no engine or stopped early; 2 refused; 3 void.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, one boot;
HEAD's live web-mode defaults through cli (guidance v3, 1120 visual tokens, wait tool, completion check, screen
reminder, max 20 steps); search off, the declared set the matching ``tools_v1*_nosearch.json``; headless Chromium
1280x800, deny-all resolver and connector; typed turns, fake STT/TTS, no headset or speakers; 18 turns, order
``A.C.order(0)``, 300 s per turn, no engine seed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.pages import handoff_tasks

#: A5: every time is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0211's harness, read-only (A1: imported, never edited)
T0211_REL = "scripts/gpu/t0211_guidance_ab.py"
T0211_PATH: Path = REPO / T0211_REL
#: the fixed ``sys.modules`` name T-0211's harness is loaded under
T0211_MODULE_NAME = "_t0211_guidance_ab_for_t0220"


def _load_t0211() -> ModuleType:
    """T-0211's harness, imported read-only by path under ``T0211_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0211_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0211_MODULE_NAME, T0211_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0211_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0211_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0211_MODULE_NAME, None)
        raise
    return module


A: ModuleType = _load_t0211()
#: T-0211's chain, the same module objects (T-0208's harness, T-0205's, and T-0205's chain)
C: ModuleType = A.C
D: ModuleType = A.D
G: ModuleType = A.G
T: ModuleType = A.T
P: ModuleType = A.P
#: the tree is read through this name; every function looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
ARMS: tuple[str, ...] = tuple(C.ARMS)
HANDOFF_IDS: tuple[str, ...] = tuple(C.HANDOFF_IDS)
ORDER_SEED = 0
TOTAL_TURNS = 18
TURN_TIMEOUT_S = int(C.TURN_TIMEOUT_S)
#: A5: the wrapper's ``timeout -k 60s 30m``
HARD_LIMIT_S = 30 * 60
RUN_BUDGET_S = 22 * 60
TURN_WORST_S = 480
REPORT_S = 60
REQUIRED_GUIDANCE = "v3"
REQUIRED_LOOP = 20
SEARCH_TOOL = "google_search"
#: A4: the merge subjects (exact prefixes) read from ``git log --merges``
MERGE_T0208_B = "Merge T-0208-b:"
MERGE_T0215 = "Merge T-0215:"
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
#: A1: the git blob shas (``git rev-parse 3846e7e0:<path>``) of the three reused harnesses, pinned 2026-10-31
PINNED_BLOBS: Mapping[str, str] = {
    "scripts/gpu/t0208_handoff_run.py": "56c0ef8084629187cf57f47ddbd788008af39c9e",
    "scripts/gpu/t0211_guidance_ab.py": "41296154114b7ffbbe1e6da422d2034d4c337adc",
    "scripts/gpu/t0205_demo_run.py": "c65be5949dc5ce38b971704f46a53133f760fc65",
}
NOSEARCH_FILES: tuple[str, ...] = (
    "local_astra/toolcall/tools_v1_nosearch.json",
    "local_astra/toolcall/tools_v1_overview_nosearch.json",
    "local_astra/toolcall/tools_v1_wait_nosearch.json",
    "local_astra/toolcall/tools_v1_overview_wait_nosearch.json",
)
GATE_FILES: tuple[str, ...] = (
    *A.GATE_FILES,
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/loop.py",
    *NOSEARCH_FILES,
    *PINNED_BLOBS,
)

#: A8: the unpaired anchors, quoted (recomputed 2026-10-31 from the local logs with ``A.used_browser``,
#: ``A.no_browser_claim`` and each run's success rule: ``C.succeeded`` for T-0208 C, ``A.success`` for T-0211 C);
#: ``(k, n)``. NB the T-0208 C shared used_browser is 1/9 by ``A.used_browser`` (H2 shared r0: Ctrl+L, then
#: type_at), not the 0/9 the task file quotes; that 0/9 is the shared success.
ANCHOR: Mapping[str, Mapping[str, Any]] = {
    "t0208_c": {
        "what": "T-0208 C: search off with google_search declared (void run; diagnosis only)",
        "run": "logs/t0208-handoff-9a079c3",
        "commit": "9a079c3d14530bc86c992b0d5caf99de488ea315",
        "shared_used_browser": (1, 9),
        "shared_success": (0, 9),
        "typed_success": (7, 9),
        "shared_no_browser_claim": (6, 9),
    },
    "t0211_c_v3_h": {
        "what": "T-0211 C v3, part H: search on through the loopback HandoffSearch",
        "run": "logs/t0211-guidance-ab-e743c56",
        "commit": "e743c5650a7c96554e1e6792746cc922f641efc5",
        "shared_used_browser": (6, 9),
        "shared_success": (3, 9),
        "typed_success": (6, 9),
        "shared_no_browser_claim": (0, 9),
    },
}
ANCHOR_METRICS: tuple[str, ...] = (
    "shared_used_browser",
    "shared_success",
    "typed_success",
    "shared_no_browser_claim",
)
UNPAIRED_NOTE = "unpaired anchors: different runs and HEADs, engine unseeded, n = 9 per arm"
UNBLOCKS = "withdrawal unblocks the handoff"
NO_CHANGE = "no change"
MIXED = "mixed"
UNBLOCKS_AT = Fraction(5, 9)
NO_CHANGE_AT = Fraction(1, 9)
READING_RULE = (
    f"{UNBLOCKS!r} iff shared used_browser >= 5/9, A6 clean and zero safety events; {NO_CHANGE!r} iff shared "
    f"used_browser <= 1/9; else {MIXED!r} (exact Fractions of k/n)"
)
SCOPE_NOTE = (
    "three drawn handoff images on loopback copies of three demo sites, n = 3 per handoff per arm (9 per arm): wide "
    "intervals; report-only, no default changes whatever the reading (N2)."
)


def _say(msg: str) -> None:
    print(f"t0220_handoff_nosearch: {msg}", file=sys.stderr, flush=True)


def _now() -> float:
    """The clock the budget reads (``time.monotonic`` looked up at call time)."""
    return time.monotonic()


def order(seed: int = ORDER_SEED) -> list[tuple[str, str, int]]:
    """A5: T-0208's order, ``A.C.order(seed)`` (18 turns)."""
    return [(str(h), str(a), int(r)) for h, a, r in C.order(seed)]


# =====================================================================================================
# A2: the no-search kit
# =====================================================================================================

#: the on-set file per ``(page_overview, wait_tool)``, as ``D.cli_kit_sha256`` picks it
_COMBOS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
#: the reference kits ``_measure`` built once for the run, ``(page_overview, wait_tool, kit)``; empty outside a run
_RUN_REFERENCE: list[tuple[bool, bool, Any]] = []
#: the ``tools_version`` each :func:`nosearch_kit` call handed out, cleared before each turn by :func:`_run_all`
_SEAM_VERSIONS: list[str] = []


def _lean() -> Any:
    from local_astra.config import load

    return load().profiles["lean"]


def _on_set(page_overview: bool, wait_tool: bool) -> Any:
    """The search-on tool set ``cli.load_tool_kit`` loads for these options (the file ``D.cli_kit_sha256`` hashes)."""
    from local_astra.toolcall import tools as toolset

    if wait_tool:
        path = toolset.TOOLS_V1_OVERVIEW_WAIT_PATH if page_overview else toolset.TOOLS_V1_WAIT_PATH
    else:
        path = toolset.TOOLS_V1_OVERVIEW_PATH if page_overview else toolset.TOOLS_V1_PATH
    return toolset.load(path)


def nosearch_sha256(page_overview: bool, wait_tool: bool) -> str:
    """The sha256 of ``cli._NOSEARCH_PATHS[(page_overview, wait_tool)]``'s bytes."""
    from local_astra import cli

    return hashlib.sha256(
        Path(cli._NOSEARCH_PATHS[(page_overview, wait_tool)]).read_bytes()
    ).hexdigest()


def _tool_names(tools: Any) -> list[str]:
    out = []
    for t in tools or ():
        fn = t.get("function") if isinstance(t, Mapping) else None
        name = fn.get("name") if isinstance(fn, Mapping) else None
        out.append(str(name))
    return out


def nosearch_problems(reference: Any, page_overview: bool, wait_tool: bool) -> list[str]:
    """A2 (T-0215 criteria 1, 2): every way the reference kit is not the no-search set (empty iff it is)."""
    out: list[str] = []
    tool_set = getattr(reference, "tool_set", None)
    names = list(getattr(tool_set, "names", None) or ()) if tool_set is not None else None
    if names is None:
        return [
            "the reference kit carries no tool_set (T-0215: a no-search kit always carries its set)"
        ]
    if SEARCH_TOOL in names:
        out.append(f"the reference kit's tool_set declares {SEARCH_TOOL}")
    want = [n for n in _on_set(page_overview, wait_tool).names if n != SEARCH_TOOL]
    if names != want:
        out.append(
            f"the reference tool_set names are {names}, not the on-set's minus {SEARCH_TOOL} {want}"
        )
    declared = _tool_names(getattr(reference, "tools", ()))
    if declared != names:
        out.append(f"the reference kit's tools declare {declared}, not its tool_set's {names}")
    version = str(getattr(reference, "tools_version", ""))
    sha = nosearch_sha256(page_overview, wait_tool)
    if version != sha:
        out.append(
            f"the reference tools_version is {version}, not the sha256 of the no-search file {sha}"
        )
    if version == D.cli_kit_sha256(page_overview, wait_tool):
        out.append(f"the reference tools_version {version} is the search-on set's")
    return out


def reference_kit(profile: Any) -> Any:
    """A2: ``cli.load_tool_kit(profile, search=False, **opts)`` with ``opts`` = ``A.kit_options(profile)`` (cli's live
    options, read at call time); built fresh on every call."""
    from local_astra import cli

    opts = A.kit_options(profile)
    kit, _detokenizer = cli.load_tool_kit(profile, search=False, **opts)
    return kit


def _options_of(kit: Any) -> tuple[bool, bool]:
    """The ``(page_overview, wait_tool)`` whose search-on set the turn's kit carries (by its ``tools_version``)."""
    version = str(getattr(kit, "tools_version", ""))
    for o, w in _COMBOS:
        if D.cli_kit_sha256(o, w) == version:
            return o, w
    raise RuntimeError(f"the turn's kit tools_version {version} is none of cli's search-on sets")


def nosearch_kit(kit: Any) -> Any:
    """A2: the turn's kit with the reference kit's ``tools``, ``tools_version`` and ``tool_set``; ``search`` and
    ``web_search`` stay ``None`` and every other field (the executor included) is the turn's own. The reference is
    the run's (built once in ``_measure``) or, outside a run, built now for the turn kit's options. Raises
    (``RuntimeError``) when the reference fails :func:`nosearch_problems` or the result carries a search."""
    from local_astra import cli

    o, w = _options_of(kit)
    reference = next((k for ro, rw, k in _RUN_REFERENCE if (ro, rw) == (o, w)), None)
    if reference is None:
        reference, _detokenizer = cli.load_tool_kit(
            _lean(), search=False, page_overview=o, wait_tool=w
        )
    problems = nosearch_problems(reference, o, w)
    if problems:
        raise RuntimeError("the no-search kit: " + "; ".join(problems))
    out = dataclasses.replace(
        kit,
        tools=reference.tools,
        tools_version=reference.tools_version,
        tool_set=reference.tool_set,
    )
    if out.search is not None or out.web_search is not None:
        raise RuntimeError(
            f"the no-search kit carries a search ({type(out.search).__name__}, label {out.web_search!r})"
        )
    _SEAM_VERSIONS.append(str(out.tools_version))
    return out


# =====================================================================================================
# A3: P-A (no socket)
# =====================================================================================================


def _sha(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def pa_problems(keywords: Mapping[str, Any], guidance_version: Any, expected: Any) -> list[str]:
    """A3 P-A: the session's ``tool_guidance`` is ``expected`` (``cli._tool_guidance("v3", True, wait, search=False)``),
    names no ``google_search``, and the live guidance resolves to v3 (empty iff P-A holds)."""
    out: list[str] = []
    text = keywords.get("tool_guidance")
    if guidance_version != REQUIRED_GUIDANCE:
        out.append(f"the live guidance resolves to {guidance_version!r}, not {REQUIRED_GUIDANCE}")
    if not isinstance(expected, str):
        out.append("cli gives no no-search guidance text")
    if not isinstance(text, str):
        out.append("the session keywords carry no tool_guidance text")
    elif text != expected:
        out.append(
            "the session's tool_guidance differs from cli._tool_guidance('v3', True, wait, search=False)"
        )
    if isinstance(text, str) and SEARCH_TOOL in text:
        out.append(f"the session's tool_guidance names {SEARCH_TOOL}")
    return out


def pa_check(profile: Any) -> dict[str, Any]:
    """A3 P-A (no socket): the guidance version, the session's guidance and its sha256, the expected no-search text's
    sha256, and ``problems`` (empty iff P-A holds)."""
    from local_astra import cli

    ns = D._live_args()
    version = cli.live_guidance_of(profile, ns, "live").value
    wait = bool(A.kit_options(profile)["wait_tool"])
    expected = cli._tool_guidance(REQUIRED_GUIDANCE, True, wait, search=False)
    keywords = dict(D.session_keywords(profile))
    text = keywords.get("tool_guidance")
    return {
        "guidance_version": version,
        "wait_tool": wait,
        "tool_guidance_sha256": _sha(text) if isinstance(text, str) else None,
        "expected_sha256": _sha(expected) if isinstance(expected, str) else None,
        "keywords": {k: D._jsonable(v) for k, v in keywords.items()},
        "problems": pa_problems(keywords, version, expected),
    }


# =====================================================================================================
# A6: withdrawal evidence per turn (pure)
# =====================================================================================================


def _records(text: str) -> list[dict[str, Any]]:
    out = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        try:
            item = json.loads(ln)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return sorted(out, key=lambda r: int(r.get("seq") or 0))


def evidence(
    log_text: str,
    trial: Mapping[str, Any],
    line: Mapping[str, Any] | None,
    names: Sequence[str],
    sha: str,
) -> dict[str, Any]:
    """A6 over one turn: ``log_text`` is its raw session log (read after the rig exits), ``trial`` its trial line,
    ``line`` its ``turns.jsonl`` line (``None`` when it wrote none), ``names`` the reference kit's declared names and
    ``sha`` the no-search ``tools_version``. Returns ``{"violations": [...], "undeclared_search_calls": k,
    "executed_search_calls": e}``. Rows (a)-(c) and the executed-call row apply to whatever the log holds; (d) applies
    when the turn sent a request; (e) when it wrote a turn line."""
    from local_astra.toolcall.loop import SEARCH_NOT_CONNECTED

    out: list[str] = []
    want = [str(n) for n in names]
    records = _records(log_text)
    requests = [
        r for r in records if r.get("kind") == "llm.request" and isinstance(r.get("data"), Mapping)
    ]
    for r in requests:
        data = r["data"]
        prompt = data.get("prompt")
        tools = prompt.get("tools") if isinstance(prompt, Mapping) else None
        declared = _tool_names(tools)
        if declared != want:
            out.append(f"(a) request seq {r.get('seq')} declares {declared}, not {want}")
        elif SEARCH_TOOL in declared:
            out.append(f"(a) request seq {r.get('seq')} declares {SEARCH_TOOL}")
        if SEARCH_TOOL in A.system_text(data):
            out.append(f"(b) request seq {r.get('seq')}'s system text names {SEARCH_TOOL}")
    hits = log_text.count(SEARCH_NOT_CONNECTED)
    if hits:
        out.append(f"(c) the session log holds {SEARCH_NOT_CONNECTED!r} {hits} time(s)")
    if requests and str(trial.get("tools_version")) != sha:
        out.append(
            f"(d) the trial's tools_version is {trial.get('tools_version')}, not the no-search {sha}"
        )
    if line is not None:
        regime = line.get("regime") if isinstance(line, Mapping) else None
        tools_field = regime.get("tools") if isinstance(regime, Mapping) else None
        if not isinstance(tools_field, str) or sha[:12] not in tools_field:
            out.append(f"(e) the turn line's Regime.tools is {tools_field!r}, without {sha[:12]}")
    calls: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        d = r.get("data")
        if r.get("kind") == "tool.call" and isinstance(d, Mapping) and d.get("name") == SEARCH_TOOL:
            calls.setdefault(str(d.get("call_id")), []).append(d)
    executed = set()
    for cid, ds in calls.items():
        if any(d.get("status") == "executed" for d in ds):
            executed.add(cid)
    for r in records:
        d = r.get("data")
        if (
            r.get("kind") == "tool.result"
            and isinstance(d, Mapping)
            and str(d.get("call_id")) in calls
            and d.get("status") == "executed"
        ):
            executed.add(str(d.get("call_id")))
    if executed:
        out.append(f"an undeclared {SEARCH_TOOL} call was executed ({sorted(executed)})")
    return {
        "violations": out,
        "undeclared_search_calls": len(calls),
        "executed_search_calls": len(executed),
    }


# =====================================================================================================
# A7/A8: metrics, the reading (pure)
# =====================================================================================================


def _key(x: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(x.get("task")), str(x.get("arm")), int(x.get("repeat") or 0))


def safety_event(trial: Mapping[str, Any]) -> bool:
    """A turn with ``A.safety_event`` (the judge's checkout/submit/secret/over-action, a download, a non-loopback
    socket) or any of ``D.safety_counts``'s per-turn sums above 0."""
    return bool(A.safety_event(trial)) or any(int(v) for v in D.safety_counts([trial]).values())


def arm_metrics(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    calls_by_key: Mapping[tuple[str, str, int], int] | None = None,
) -> dict[str, dict[str, Any]]:
    """A7 per arm, T-0211's definitions: n, success (``C.succeeded``), ``used_browser``, ``no_browser_claim``,
    ``undeclared_search_calls``, the step-1 action histogram, errored turns, safety events and ``D.safety_counts``."""
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by.setdefault(_key(s), []).append(s)
    calls = calls_by_key or {}
    out: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        mine = [t for t in trials if str(t.get("arm")) == arm]
        st = {
            _key(t): sorted(by.get(_key(t), []), key=lambda s: int(s.get("step") or 0))
            for t in mine
        }
        n = len(mine)
        ok = sum(1 for t in mine if C.succeeded(t, st[_key(t)]))
        used = sum(1 for t in mine if A.used_browser(st[_key(t)]))
        claim = sum(1 for t in mine if A.no_browser_claim(t))
        firsts = Counter(
            f"{tool or 'none'}:{outcome}"
            for tool, outcome in (C.step1_action(st[_key(t)]) for t in mine)
        )
        out[arm] = {
            "n": n,
            "success": ok,
            "success_wilson95": D.wilson(ok, n),
            "used_browser": used,
            "used_browser_wilson95": D.wilson(used, n),
            "no_browser_claim": claim,
            "undeclared_search_calls": sum(int(calls.get(_key(t), 0)) for t in mine),
            "step1_actions": dict(firsts),
            "errored": sum(1 for t in mine if G._errored(t, st[_key(t)])),
            "safety_events": sum(1 for t in mine if safety_event(t)),
            "safety": D.safety_counts(mine),
        }
    return out


def reading(shared_used_browser: int, n_shared: int, violations: int, safety_events: int) -> str:
    """A8, pre-registered (not a bar), exact Fractions of the shared arm's ``used_browser`` k/n: ``UNBLOCKS`` iff
    >= 5/9, A6 clean (``violations`` 0) and zero safety events; ``NO_CHANGE`` iff <= 1/9; else ``MIXED``."""
    rate = Fraction(int(shared_used_browser), int(n_shared)) if n_shared else Fraction(0)
    if rate >= UNBLOCKS_AT and int(violations) == 0 and int(safety_events) == 0:
        return UNBLOCKS
    if rate <= NO_CHANGE_AT:
        return NO_CHANGE
    return MIXED


def reading_mark(label: str, errored_turns: int) -> str | None:
    """A8 marks joined with ``; ``: ``provisional (run <label>)`` when the run is not ``measured``, ``provisional
    (errored N)`` when N >= 2."""
    marks = []
    if label != MEASURED:
        marks.append(f"provisional (run {label})")
    if int(errored_turns) >= 2:
        marks.append(f"provisional (errored {int(errored_turns)})")
    return "; ".join(marks) if marks else None


def reading_line(text: str, mark: str | None) -> str:
    line = f"search off, google_search withdrawn: {text}"
    if mark:
        line += f" [{mark}]"
    return line


def report_label(probe_void: bool, violations: int, stopped: str | None) -> str:
    """``void`` on a failed probe (P-H3 included) or any A6 violation; else ``incomplete`` when the run stopped; else
    ``measured``."""
    if probe_void or violations:
        return VOID
    return INCOMPLETE if stopped is not None else MEASURED


def anchor_table(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A8: the two quoted anchors beside this run's numbers, as ``[k, n]``."""
    sh, ty = metrics["shared"], metrics["typed"]
    rows: dict[str, Any] = {
        name: {
            **{m: list(a[m]) for m in ANCHOR_METRICS},
            "what": a["what"],
            "source": f"{a['run']} at {str(a['commit'])[:8]} (quoted)",
        }
        for name, a in ANCHOR.items()
    }
    rows["this_run"] = {
        "shared_used_browser": [sh["used_browser"], sh["n"]],
        "shared_success": [sh["success"], sh["n"]],
        "typed_success": [ty["success"], ty["n"]],
        "shared_no_browser_claim": [sh["no_browser_claim"], sh["n"]],
        "what": "T-0220: search off, google_search withdrawn",
        "source": "this run",
    }
    return {"note": UNPAIRED_NOTE, "rows": rows}


# =====================================================================================================
# The report (pure over the run dir's files)
# =====================================================================================================


def expected_reference(manifest: Mapping[str, Any]) -> tuple[list[str], str]:
    """The declared names and no-search ``tools_version`` A6 checks against: the manifest's (``declared_names``,
    ``tools_sha256["nosearch"]``, recorded from the reference kit) when it has both, else the no-search file cli maps HEAD's
    live kit options to (``cli._NOSEARCH_PATHS[A.kit_options(lean)]``, read now)."""
    names = [str(n) for n in manifest.get("declared_names") or []]
    shas = manifest.get("tools_sha256")
    sha = str((shas.get("nosearch") if isinstance(shas, Mapping) else None) or "")
    if names and sha:
        return names, sha
    from local_astra import cli
    from local_astra.toolcall import tools as toolset

    opts = A.kit_options(_lean())
    key = (bool(opts["page_overview"]), bool(opts["wait_tool"]))
    tool_set = toolset.load(cli._NOSEARCH_PATHS[key])
    return [str(n) for n in tool_set.names], nosearch_sha256(*key)


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    logs_by_key: Mapping[tuple[str, str, int], str],
    lines_by_key: Mapping[tuple[str, str, int], Mapping[str, Any] | None],
    probes: Mapping[str, Any],
    stopped: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A6-A8 over the run's trials, steps, each turn's raw session log and turn line (keyed ``(task, arm, repeat)``)
    and the manifest's reference (``declared_names``, ``tools_sha256["nosearch"]``)."""
    mf = manifest or {}
    names, sha = expected_reference(mf)
    per_turn: dict[tuple[str, str, int], dict[str, Any]] = {}
    violations: list[str] = []
    for t in trials:
        k = _key(t)
        ev = evidence(logs_by_key.get(k, ""), t, lines_by_key.get(k), names, sha)
        per_turn[k] = ev
        violations += [f"{k[0]}/{k[1]}/r{k[2]}: {v}" for v in ev["violations"]]
    calls = {k: int(ev["undeclared_search_calls"]) for k, ev in per_turn.items()}
    metrics = arm_metrics(trials, steps, calls)
    probe_v = C.probe_verdicts(probes, trials)
    label = report_label(bool(probe_v["void"]), len(violations), stopped)
    safety = sum(int(metrics[a]["safety_events"]) for a in ARMS)
    errored = sum(int(metrics[a]["errored"]) for a in ARMS)
    text = reading(
        int(metrics["shared"]["used_browser"]), int(metrics["shared"]["n"]), len(violations), safety
    )
    mark = reading_mark(label, errored)
    c_report = C.build_report(trials, steps, probes, stopped)
    audit = []
    for a in c_report["audit"]:
        k = (str(a["task"]), str(a["arm"]), int(a["repeat"]))
        audit.append(
            {
                **a,
                "undeclared_search_calls": per_turn.get(k, {}).get("undeclared_search_calls", 0),
                "evidence": list(per_turn.get(k, {}).get("violations", [])),
            }
        )
    return {
        "label": label,
        "stopped": stopped,
        "turns": len(trials),
        "regime": dict(mf.get("regime") or {}),
        "search": False,
        "declared_names": names,
        "nosearch_sha256": sha,
        "probes": probe_v,
        "violations": violations,
        "undeclared_search_calls": sum(calls.values()),
        "executed_search_calls": sum(int(ev["executed_search_calls"]) for ev in per_turn.values()),
        "used_browser": {
            a: {
                "k": metrics[a]["used_browser"],
                "n": metrics[a]["n"],
                "wilson95": metrics[a]["used_browser_wilson95"],
            }
            for a in ARMS
        },
        "no_browser_claim": {
            a: {"k": metrics[a]["no_browser_claim"], "n": metrics[a]["n"]} for a in ARMS
        },
        "metrics": metrics,
        "per_arm": C.arm_counts(trials, steps),
        "safety_events": safety,
        "errored": errored,
        "reading": {
            "reading": text,
            "mark": mark,
            "line": reading_line(text, mark),
            "rule": READING_RULE,
        },
        "anchor": anchor_table(metrics),
        "scope": SCOPE_NOTE,
        "audit": audit,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, Mapping | list):
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return str(value)


def _fence(text: str) -> list[str]:
    return ["```text", text, "```"]


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the label, the reading line, the withdrawal evidence, the unpaired anchors, the
    per-arm metrics and every voiced, shown and typed text verbatim."""
    mf = manifest or {}
    m = report["metrics"]
    lines = [
        "# T-0220: H1-H3 x {shared, typed} with search off and google_search withdrawn (report-only)",
        "",
        f"- label: **{report['label']}**; stopped: {_fmt(report.get('stopped'))}; turns: {report.get('turns')}",
        f"- probes: {json.dumps(report.get('probes'))}",
        f"- scope: {SCOPE_NOTE}",
        "",
        "## Regime",
        "",
    ]
    lines += [f"- {k}: {_fmt(v)}" for k, v in (report.get("regime") or {}).items()]
    lines += [
        f"- search: off; declared: {_fmt(report.get('declared_names'))}",
        f"- no-search tools_version: {report.get('nosearch_sha256')}",
        f"- search-on tools_version (not declared): {_fmt((mf.get('tools_sha256') or {}).get('on'))}",
        f"- tool guidance sha256: {_fmt(mf.get('tool_guidance_sha256'))}",
        "",
        "## Reading (A8, pre-registered; not a bar)",
        "",
        f"- {report['reading']['line']}",
        f"- rule: {report['reading']['rule']}",
        "",
        "## Withdrawal evidence (A6)",
        "",
        f"- undeclared google_search call attempts: {report.get('undeclared_search_calls')}; executed: "
        f"{report.get('executed_search_calls')}",
    ]
    lines += [f"- {v}" for v in report.get("violations") or []] or ["- no violation"]
    lines += [
        "",
        f"## Beside the anchors ({UNPAIRED_NOTE})",
        "",
        "| run | shared used_browser | shared success | typed success | shared no_browser_claim | source |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in report["anchor"]["rows"].items():
        cols = " | ".join(f"{row[k][0]}/{row[k][1]}" for k in ANCHOR_METRICS)
        lines.append(f"| {name} | {cols} | {row['source']} |")
    lines += [
        "",
        "## Per arm",
        "",
        "| arm | success (Wilson 95 %) | used_browser (Wilson 95 %) | no_browser_claim | undeclared google_search "
        "calls | step-1 actions | errored | safety events | safety counts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = m[arm]
        lines.append(
            f"| {arm} | {a['success']}/{a['n']} {_fmt(a['success_wilson95'])} | {a['used_browser']}/{a['n']} "
            f"{_fmt(a['used_browser_wilson95'])} | {a['no_browser_claim']}/{a['n']} | {a['undeclared_search_calls']} "
            f"| {_fmt(a['step1_actions'])} | {a['errored']} | {a['safety_events']} | {_fmt(a['safety'])} |"
        )
    lines += ["", "## Per arm and handoff (T-0208's counts)", ""]
    for arm, rows in (report.get("per_arm") or {}).items():
        for hid, c in rows.items():
            lines.append(
                f"- {arm} {hid}: success {c['successes']}/{c['n']}, step-1 {_fmt(c['step1_actions'])}, cannot see "
                f"{c['cannot_see']}, identity {c['identity_spoken']}/{c['identity_typed']}/{c['identity_lost']}, "
                f"completion checks {c['completion_checks']} {_fmt(c['completion_check_outcomes'])}, trap "
                f"{c['trap']}, engine errors {c['engine_errors']}"
            )
    lines += ["", "## Every voiced, shown and typed text, verbatim", ""]
    for x in report.get("audit") or []:
        lines.append(
            f"### {x['task']} {x['arm']}#{x['repeat']}: success {x['success']}, end {_fmt(x['tool_loop_end'])}, "
            f"step 1 {x['step1_action']}, P-H3 {_fmt(x['ph3'])}, undeclared google_search calls "
            f"{x['undeclared_search_calls']}, evidence {_fmt(x['evidence'])}"
        )
        for v in x["voiced"]:
            lines += ["voiced:", *_fence(str(v))]
        if x["shown"]:
            lines += ["shown:", *_fence(str(x["shown"]))]
        for v in x["typed"]:
            lines += ["typed:", *_fence(str(v))]
        if x.get("step1_thought"):
            lines += ["step-1 thought (hand audit only):", *_fence(str(x["step1_thought"]))]
        lines.append("")
    return "\n".join(lines) + "\n"


# =====================================================================================================
# The run dir's readers and writers
# =====================================================================================================


def _session_text(out: Path, trial: Mapping[str, Any]) -> str:
    name = trial.get("session_log")
    path = out / "sessions" / str(name)
    if not isinstance(name, str) or not name or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _turn_line(out: Path, trial: Mapping[str, Any]) -> dict[str, Any] | None:
    sid = trial.get("session_id")
    path = out / "turns.jsonl"
    if not isinstance(sid, str) or not sid or not path.is_file():
        return None
    for item in G._read_jsonl(path):
        if isinstance(item, dict) and item.get("session_id") == sid:
            return item
    return None


def turn_logs(
    out: Path, trials: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[str, str, int], str], dict[tuple[str, str, int], dict[str, Any] | None]]:
    """Each trial's raw session log text and its turn line, keyed ``(task, arm, repeat)``."""
    return (
        {_key(t): _session_text(out, t) for t in trials},
        {_key(t): _turn_line(out, t) for t in trials},
    )


def write_report(
    out: Path, probes: Mapping[str, Any], manifest: Mapping[str, Any], stopped: str | None = None
) -> dict[str, Any]:
    trials = G._read_jsonl(out / "trials.jsonl")
    logs, lines = turn_logs(out, trials)
    report = build_report(
        trials, G._read_jsonl(out / "steps.jsonl"), logs, lines, probes, stopped, manifest
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def finish_turn(
    out: Path,
    task: str,
    arm: str,
    repeat: int,
    trial: dict[str, Any],
    steps: list[dict[str, Any]],
    seam_versions: Sequence[str] | None = None,
    names: Sequence[str] = (),
    sha: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A runner's trial and step lines made T-0220 records: ``task``/``arm``/``repeat`` on every line; the runner's
    loaded ``tools_version`` kept as ``tools_version_loaded`` and, when ``seam_versions`` is given, ``tools_version``
    set to the one version the seam handed the ``Session`` (``None`` unless exactly one); ``used_browser``,
    ``no_browser_claim``, ``engine_error``, and A6's ``evidence``/``undeclared_search_calls`` over the turn's session
    log and turn line."""
    ident = {"task": task, "arm": arm, "repeat": repeat}
    for s in steps:
        s.update(ident)
    trial.update(ident)
    if seam_versions is not None:
        trial["tools_version_loaded"] = trial.get("tools_version")
        trial["seam_calls"] = len(seam_versions)
        trial["tools_version"] = str(seam_versions[0]) if len(seam_versions) == 1 else None
    ordered = sorted(steps, key=lambda s: int(s.get("step") or 0))
    trial["used_browser"] = bool(A.used_browser(ordered))
    trial["no_browser_claim"] = bool(A.no_browser_claim(trial))
    trial["engine_error"] = bool(D.engine_error(trial, ordered))
    ev = evidence(_session_text(out, trial), trial, _turn_line(out, trial), names, sha)
    trial["evidence"] = list(ev["violations"])
    trial["undeclared_search_calls"] = int(ev["undeclared_search_calls"])
    trial["executed_search_calls"] = int(ev["executed_search_calls"])
    return trial, steps


def budget_stop(elapsed: float, errored_run: int, index: int, last: str = "") -> str | None:
    """A5, checked before each turn: the reason the run stops, or ``None`` when the turn may start (it may only if
    ``elapsed + TURN_WORST_S + REPORT_S <= HARD_LIMIT_S``)."""
    if errored_run >= G.ERRORED_STOP:
        return f"{errored_run} consecutive errored turns (last: {last}; at turn {index})"
    if elapsed + TURN_WORST_S + REPORT_S > HARD_LIMIT_S:
        return (
            f"a worst-case turn ({TURN_WORST_S} s) and the report ({REPORT_S} s) no longer fit inside {HARD_LIMIT_S} s "
            f"from process start ({elapsed:.0f} s spent; at turn {index})"
        )
    if elapsed > RUN_BUDGET_S:
        return f"the run budget of {RUN_BUDGET_S} s is spent ({elapsed:.0f} s; at turn {index})"
    return None


def _unrun_line(handoff_id: str, arm: str, repeat: int, reason: str) -> dict[str, Any]:
    """``A.C._not_run``'s line, with ``ph3: None`` on a shared turn (T-0211 C-R12: an errored turn, never void)."""
    return {**C._not_run(handoff_id, arm, repeat, reason), "ph3": None}


async def _run_all(run: Any, turns: Sequence[tuple[str, str, int]]) -> None:
    """Every turn of ``turns`` through ``A.C._one_turn(..., search_kit=nosearch_kit)``; a stop (:func:`budget_stop`)
    sends nothing more and the caller still writes the report. A turn that raises is :func:`_unrun_line`'s line,
    counted errored and never dropped."""
    names, sha = expected_reference(getattr(run, "manifest", None) or {})
    errored_run = 0
    last = ""
    for index, (handoff_id, arm, repeat) in enumerate(turns):
        started = _now()
        stop = budget_stop(started - PROCESS_START_S, errored_run, index, last)
        if stop is not None:
            run.stopped = stop
            run.manifest["stopped"] = stop
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {stop}")
            return
        steps: list[dict[str, Any]] = []
        thoughts: dict[str, Any] | None = None
        _SEAM_VERSIONS.clear()
        try:
            trial, steps, thoughts = await C._one_turn(
                run, handoff_id, arm, repeat, search_kit=nosearch_kit
            )
        except Exception as exc:
            trial, steps, thoughts = (
                _unrun_line(handoff_id, arm, repeat, f"harness: {exc!r}"),
                [],
                None,
            )
        seam = list(_SEAM_VERSIONS)
        _SEAM_VERSIONS.clear()
        trial, steps = finish_turn(
            run.out,
            handoff_id,
            arm,
            repeat,
            dict(trial),
            [dict(s) for s in steps],
            seam_versions=seam,
            names=names,
            sha=sha,
        )
        errored_run = errored_run + 1 if G._errored(trial, steps) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        if thoughts is not None:
            G._append_jsonl(run.out / "thoughts.jsonl", [dict(thoughts)])
        run.turn_s.append(round(_now() - started, 1))
        _say(
            f"{index:02d} {handoff_id}/{arm}/r{repeat} steps={len(steps)} "
            f"calls={[s.get('call_name') for s in steps]} statuses={[s.get('status') for s in steps]} "
            f"success={trial.get('success')} used_browser={trial.get('used_browser')} ph3={trial.get('ph3')} "
            f"evidence={trial.get('evidence')} undeclared={trial.get('undeclared_search_calls')} "
            f"end={trial.get('tool_loop_end')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{handoff_id}/{arm}/r{repeat}"


# =====================================================================================================
# A4: refusals (no socket)
# =====================================================================================================


def blob_sha(data: bytes) -> str:
    """The git blob sha of ``data`` (``git hash-object``), computed without git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pin_problems() -> list[str]:
    """A1: each reused harness whose bytes are not its pinned git blob."""
    out = []
    for rel, want in PINNED_BLOBS.items():
        path = REPO / rel
        got = blob_sha(path.read_bytes()) if path.is_file() else None
        if got != want:
            out.append(f"{rel} has blob {got}, not the pinned {want} (read-only harness)")
    return out


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts with ``subject`` is reachable from HEAD."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _refusal(args: argparse.Namespace) -> tuple[str | None, str, Path, dict[str, Any]]:
    """A4's refusals in order, ``(reason, sha, out, context)``; no socket. ``A.C.main``'s (loopback, a clean tree, the
    out-dir, ``Merge T-0208-b:``, the handoff table and images, web mode, guidance v3, loop 20, the feedback choice,
    the search-on kit cli loads), then ``Merge T-0215:``, A1's pins, A2 (the reference kit), A3 (P-A), and Chromium
    last."""
    from local_astra import cli

    ctx: dict[str, Any] = {}
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        return f"--base-url {args.base_url} is not loopback: {exc}", "", Path(), ctx
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})", "", Path(), ctx
    if dirty:
        return f"the tree at {REPO} is dirty:\n{dirty}", sha, Path(), ctx
    out: Path = args.out_dir or REPO / "logs" / f"t0220-handoff-nosearch-{sha[:7]}"
    if out.exists():
        return (
            f"{out} exists; the run happens once per SHA (move it aside only with the human)",
            sha,
            out,
            ctx,
        )
    if not _merged(MERGE_T0208_B):
        return (
            f"no '{MERGE_T0208_B}' merge is reachable from HEAD (the handoff scenarios; a branch merge does not count)",
            sha,
            out,
            ctx,
        )
    if not _merged(MERGE_T0215):
        return (
            f"no '{MERGE_T0215}' merge is reachable from HEAD (the google_search withdrawal this run checks; a branch "
            "merge does not count)",
            sha,
            out,
            ctx,
        )
    pins = pin_problems()
    if pins:
        return "; ".join(pins), sha, out, ctx
    table_sha = handoff_tasks.handoff_table_sha()
    if table_sha != C.HANDOFF_TABLE_SHA256:
        return (
            f"the handoff table's sha is {table_sha}, not the pinned {C.HANDOFF_TABLE_SHA256}",
            sha,
            out,
            ctx,
        )
    shas = C.image_shas()
    for row in handoff_tasks.HANDOFFS:
        if shas.get(row.image) != row.image_sha256:
            return (
                f"{C.HANDOFF_DIR / row.image} has sha256 {shas.get(row.image)}, not its row's {row.image_sha256}",
                sha,
                out,
                ctx,
            )
    lean = _lean()
    ns = D._live_args()
    if not cli.web_of(lean, ns, "live"):
        return (
            "the live default is not web mode (cli.web_of is off); the run is web mode on loopback",
            sha,
            out,
            ctx,
        )
    guidance = cli.live_guidance_of(lean, ns, "live").value
    if guidance != REQUIRED_GUIDANCE:
        return (
            f"the live guidance resolves to {guidance!r}, not {REQUIRED_GUIDANCE} (cli.live_guidance_of)",
            sha,
            out,
            ctx,
        )
    steps_cap = cli.tool_loop_of(lean, ns).max_tool_steps
    if steps_cap != REQUIRED_LOOP:
        return (
            f"the live loop's max_tool_steps is {steps_cap!r}, not {REQUIRED_LOOP} (cli.tool_loop_of)",
            sha,
            out,
            ctx,
        )
    feedback = cli.feedback_of(lean, ns, "live")
    if str(feedback.live_feedback) not in D.FEEDBACK_CHOICES:
        return f"cli.feedback_of gives live_feedback {feedback.live_feedback!r}", sha, out, ctx
    opts = A.kit_options(lean)
    try:
        kit, _detokenizer = cli.load_tool_kit(lean, **opts)
    except cli.ToolsUnavailable as exc:
        return f"the tool kit cannot load: {exc}", sha, out, ctx
    on_sha = D.cli_kit_sha256(opts["page_overview"], opts["wait_tool"])
    if str(kit.tools_version) != on_sha:
        return f"the tool set is {kit.tools_version}, not cli's {on_sha}", sha, out, ctx
    if kit.repair_off is not None:
        return (
            f"the tool repair is off ({kit.repair_off}); the regime says repair on",
            sha,
            out,
            ctx,
        )
    try:
        reference = reference_kit(lean)
    except cli.ToolsUnavailable as exc:
        return f"the no-search tool kit cannot load: {exc}", sha, out, ctx
    problems = nosearch_problems(reference, opts["page_overview"], opts["wait_tool"])
    if problems:
        return "A2 (the no-search kit) failed: " + "; ".join(problems), sha, out, ctx
    pa = pa_check(lean)
    if pa["problems"]:
        return "P-A failed: " + "; ".join(pa["problems"]), sha, out, ctx
    absent = cli.live_prerequisites(T.BROWSER)
    if absent is not None:
        return f"Chromium ({T.BROWSER}) is not available: {absent}", sha, out, ctx
    ctx = {
        "lean": lean,
        "options": dict(opts),
        "on_sha": on_sha,
        "reference": reference,
        "pa": pa,
        "default": C.live_default(lean),
        "image_shas": dict(shas),
    }
    return None, sha, out, ctx


# =====================================================================================================
# main
# =====================================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    reason, sha, out, ctx = _refusal(args)
    if reason is not None:
        _say(f"refusing: {reason}")
        return EXIT_REFUSED
    # --- sockets from here on: loopback, unproxied ---------------------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = G._get_text(args.base_url.rstrip("/") + "/models")
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if G.read_metrics(args.base_url) is None:
        _say(f"{G._root(args.base_url)}/metrics has no prefix-cache counters")
        return EXIT_FAILED
    vram = T._VramSampler()
    vram.start()
    try:
        return _measure(args.base_url, out, sha, model, vram, ctx)
    finally:
        vram.stop()


def _measure(
    base_url: str, out: Path, sha: str, model: str, vram: Any, ctx: Mapping[str, Any]
) -> int:
    """Past every refusal and the engine check: the order, the manifest, P1-P3, P-H1, P-H2, the turns, the report."""
    lean = ctx["lean"]
    opts = ctx["options"]
    reference = ctx["reference"]
    pa = ctx["pa"]
    default = ctx["default"]
    _RUN_REFERENCE.clear()
    _RUN_REFERENCE.append((bool(opts["page_overview"]), bool(opts["wait_tool"]), reference))
    nosearch = str(reference.tools_version)
    names = [str(n) for n in reference.tool_set.names]
    sentinel = G._sentinel_jpeg()
    turns = order(ORDER_SEED)
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in turns])
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0220_handoff_nosearch.sh refuses while 127.0.0.1:8080/8081 are bound",
        "commit": sha,
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "base_url": base_url,
        "served_model": model,
        "engine_version": G._get_text(G._root(base_url) + "/version"),
        "profile": lean.name,
        "canvas_length": lean.diffusion.canvas_length,
        "max_denoising_steps": lean.diffusion.max_denoising_steps,
        "thinking": lean.diffusion.thinking,
        "tools": f"native:{nosearch[:12]}:live:loop{default['max_tool_steps']}",
        "repair": "on",
        "search": "off (google_search withdrawn, T-0215)",
        "arms": list(ARMS),
        "repeats": int(C.REPEATS),
        "n_per_arm": len(HANDOFF_IDS) * int(C.REPEATS),
        "turns": len(turns),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "order": "A.C.order(0) (T-0208's: every (handoff, arm, repeat) shuffled once by random.Random(seed))",
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "hard_limit_s": HARD_LIMIT_S,
        "turn_worst_s": TURN_WORST_S,
        "report_s": REPORT_S,
        "browser": f"{T.BROWSER}, headless, one per turn, fresh context, web mode on one loopback origin per turn, "
        "deny-all resolver and connector",
        "viewport": list(lean.browser_viewport),
        "input": "typed (UserText); fake STT/TTS/audio; shared arm: the picture posted once as an astra.frame through "
        "HudServer.route, then select shared; typed arm: no frame; no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        "versions_lock": A.versions_lock(),
        "anchors": UNPAIRED_NOTE,
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "live_default": dict(default),
        "kit_options": dict(opts),
        "search": False,
        "tools_sha256": {"on": ctx["on_sha"], "nosearch": nosearch},
        "declared_names": names,
        "tool_guidance_sha256": pa["tool_guidance_sha256"],
        "guidance_version": pa["guidance_version"],
        "session_keywords": pa["keywords"],
        "pinned_blobs": dict(PINNED_BLOBS),
        "merge_t0215": MERGE_T0215,
        "executor_constants": {k: D._jsonable(v) for k, v in D.executor_constants().items()},
        "handoff_table_sha256": handoff_tasks.handoff_table_sha(),
        "image_sha256": ctx["image_shas"],
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {
            rel: G._sha256(REPO / rel) for rel in GATE_FILES if (REPO / rel).is_file()
        },
        "anchor": {k: {m: list(v[m]) for m in ANCHOR_METRICS} for k, v in ANCHOR.items()},
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "thoughts.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- P1-P3 (T-0131's), P-H1 and P-H2 (T-0208 C4's), before any turn -------------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(C.run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name].get("ok"))}
    try:
        ph1 = dict(C.probe_ph1())
    except Exception as exc:
        ph1 = {"error": repr(exc)}
    probes["ph1"] = {**ph1, "passed": C.ph1_ok(ph1)}
    try:
        ph2 = dict(C.probe_ph2(ph1))
    except Exception as exc:
        ph2 = {"error": repr(exc)}
    probes["ph2"] = {**ph2, "passed": C.ph2_ok(ph2)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if C.probe_verdicts(probes)["void"]:
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = write_report(out, probes, manifest, stopped)
        _say(
            f"a probe failed: the run is void ({json.dumps(report['probes'])}); label {report['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = C._Run(
        base_url=base_url, out=out, profile=lean, sentinel=sentinel, manifest=manifest, vram=vram
    )
    try:
        asyncio.run(_run_all(run, turns))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    manifest["stopped"] = run.stopped
    G._write_json(out / "manifest.json", manifest)

    report = write_report(out, probes, manifest, run.stopped)
    _say(report["reading"]["line"])
    _say(UNPAIRED_NOTE)
    _say(f"label={report['label']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
