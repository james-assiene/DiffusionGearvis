"""T-0209 B -- the locked, report-only GPU run of the RTX 5090 audience scenarios on loopback (R1 "Why is Blender doing
this?"; R2 joins only once Merge T-0209-a2 is an ancestor) at HEAD's live defaults: web mode on loopback with a
deny-all resolver and connector, search on through the loopback index (``rtx_search.LoopbackSearch`` with
``web_search="ddgs"``, H-T209-1), two arms, ``shared`` and ``typed``, n = 3 per scenario per arm = 6 typed turns (12
with R2) through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0209_rtx_run.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**.

B1: T-0205 C's harness (``scripts/gpu/t0205_demo_run.py``) is imported read-only by path as ``D`` and never edited;
its pure helpers (``live_default``, ``session_keywords``, ``shown_text``, ``clip_kinds``, ``turn_records``, the Wilson
interval, nearest rank, the engine-error rule), its probe pieces (the deny-all seams, the probe executor and rig, the
watch, the exhaustive interaction) and its chain (``F``/``P``/``T``/``G``) are reused. T-0208 C's harness is not an
ancestor's, so its patterns (the shared-arm ingress, the socketless HUD, the identity and step-1 rules) are copied
here, not imported. HEAD's live defaults are read through ``cli`` at run time (``live_default`` adds the completion
check to T-0205's keys) and recorded in ``manifest.json`` with ``shared_check`` (a ``Merge T-0208-a2:`` merge
reachable), ``r2`` (a ``Merge T-0209-a2:`` merge reachable) and ``search_backend: "rtx_loopback"`` (the session logs
say ``ddgs``). ``RTX_TABLE_SHA256`` and ``SEARCH_INDEX_SHA256`` pin the table and the index; each image's sha256
(under ``RTX_SHARED_DIR``, read at call time) is checked before any socket.

B2, refusals (exit 2), all **before any socket**: a non-loopback ``--base-url``, a dirty tree named by its paths, an
existing ``--out-dir``, no ``Merge T-0209-a1:`` merge reachable from HEAD, ``--only R2`` without a ``Merge
T-0209-a2:``, the table, the index or an image off its pin, a live default whose web mode is off, whose guidance is
not v3 or whose loop is not 20, a tool kit that cannot load, is off cli's set or has repair off, a kit whose search is
not production's ``LoopbackSearch`` labelled ``ddgs`` (``kit_refusal``, by identity, never by name), Chromium absent.
No engine: exit 1.

B3, the arms: each turn has a fresh ``StartPageServer(static_dir=RTX_DIR)``, a fresh headless web-mode
``BrowserToolExecutor`` (deny-all resolver and connector, no profile directory, so ``visited`` never carries over:
R1-J5) opened at ``/web.html``, a fresh ``Session`` whose kit has a fresh ``LoopbackSearch`` (``rtx_kit``) and a fresh
socketless ``HudServer`` whose ``frame_sink`` is one real ``BrowserFrameSource``, which is also the session's
perception (cli's production shape). ``shared``: one ``astra.frame`` carrying the row's image bytes (screen source, its
own size, no title or app) and one ``astra.vision.select {source: "shared"}``, each through ``HudServer.route``, then
``UserText(request)``. ``typed``: no frame, no select (the session stays on ``browser``), ``UserText(control_request)``.
The order is ``order(ORDER_SEED, r2)``, concurrency 1, ``TURN_TIMEOUT_S`` per turn, the engine unseeded.

B4, probes before any turn (any failing: ``void``, exit 3, a report, no turn): P1-P3 (T-0131's ``run_probes``); P-R1
(``probe_pr1``, browser only): R1's golden path and its traps (A1.10's, this harness's own copy) through the real
web-mode executor at 1280x800 on ``/web.html``, re-judged by ``rtx_tasks.judge``; P-R2 (``probe_pr2``, browser only):
zero resolver or connector calls, zero non-loopback sockets, no egress record and no request off the origin over P-R1
plus an exhaustive interaction of every rtx page. P-R3 is per shared turn, from its session log (``pr3_ok``, the switch
rule): the shared image with its label until the first browser frame, never after it, never beside another image.
Rulings carried from T-0208 C: C-R1 an executed ``wait`` counts as acting for ``identity_lost``; C-R2 a shared turn
whose exception is not an ingress self-check failure and that sent no step-1 request gets ``pr3: None``, which is not
void (left to ``ERRORED_STOP``); an ingress self-check failure gives ``pr3: False`` (void).

B5, the report (``report.json``, ``report.md``): per arm and scenario, success k/n with a Wilson interval,
``no_page_opened``, pages visited verbatim, searches and their queries verbatim, R1's forum opened / trap kinds /
``cross_checked`` / ``said_link`` / ``mentions_wrong_fix``, the step-1 action, shared-image refusals, "cannot see"
endings, identity columns, completion checks and their outcomes, downloads, sign-in typing, waits, steps and wall
p50/max, engine errors, degenerate and step-cap ends; every voiced, shown and typed text (search queries included)
verbatim plus the step-1 thought (hand audit only). Labels ``void``, ``incomplete``, ``measured``; no bars.

Output (``--out-dir``, default ``logs/t0209-rtx-<sha7>/``): ``order.json`` (before the probes), ``manifest.json``,
``probes.json``, ``trials.jsonl``, ``steps.jsonl``, ``thoughts.jsonl``, ``turns.jsonl``, ``sessions/*.jsonl``,
``report.json``, ``report.md``. Exit: 0 measured; 1 no engine or stopped early; 2 refused; 3 void.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, one boot;
HEAD's live web-mode defaults through cli (recorded); search on through the loopback index; headless Chromium
1280x800, deny-all resolver and connector; shared frame at the HUD screen bucket, posted once before the turn; typed
turns, fake STT/TTS, no headset or speakers; 2 arms x R1 x n = 3 = 6 turns (12 with R2), order seed 0, 300 s per turn.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.pages import rtx_search, rtx_tasks
from local_astra.pages.rtx_tasks import RTX_DIR

#: B6: the run budget is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0205 C's harness, read-only (B1: imported, never edited)
T0205_PATH: Path = REPO / "scripts" / "gpu" / "t0205_demo_run.py"
#: the fixed ``sys.modules`` name T-0205's harness is loaded under (its dataclasses look themselves up there)
T0205_MODULE_NAME = "_t0205_demo_run_for_t0209"


def _load_t0205() -> ModuleType:
    """T-0205 C's harness, imported read-only by path under ``T0205_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0205_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0205_MODULE_NAME, T0205_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0205_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0205_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0205_MODULE_NAME, None)
        raise
    return module


D: ModuleType = _load_t0205()
#: T-0205's chain, the same module objects
F: ModuleType = D.F
P: ModuleType = D.P
T: ModuleType = D.T
G: ModuleType = D.G

# --- T-0205 C's pure helpers, unchanged (B1) ----------------------------------------------------------------
session_keywords = D.session_keywords
shown_text = D.shown_text
clip_kinds = D.clip_kinds
wilson = D.wilson
nearest_rank = D.nearest_rank
engine_error = D.engine_error
cli_kit_sha256 = D.cli_kit_sha256
executor_constants = D.executor_constants
# --- T-0111g's and T-0131's, through T-0205's chain ------------------------------------------------------------
LoopMeter = G.LoopMeter
sentinel_hits = G.sentinel_hits
non_loopback_sockets = G.non_loopback_sockets
run_probes = T.run_probes
config_evidence = T.config_evidence
BROWSER = T.BROWSER
VIEWPORT = F.VIEWPORT
#: the tree is read through this name; ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: ``rtx_tasks.rtx_table_sha()`` while the table is R1 only (Merge T-0209-a1), a literal; A2 moves it
RTX_TABLE_SHA256 = "eaa308479cd618417c9179145300751b796b0c5c4db3cb2abe8ac82618d97af9"
#: ``rtx_search.search_index_sha()`` at Merge T-0209-a1 (R1 rows only), a literal; A2 moves it
SEARCH_INDEX_SHA256 = "b8724480a80ff5b9503fbcec7aeb14a11fbdc517a281cb051e2c14ccea017ad0"
#: A1.6's table order; R2 joins the run only with Merge T-0209-a2 (B1)
ALL_IDS = ("R1", "R2")
ARMS = ("shared", "typed")
#: B3: n = 3 per scenario per arm
REPEATS = 3
ORDER_SEED = 0
TURN_TIMEOUT_S = 300
#: B6: inside the locked script's ``timeout -k 60s 45m`` so the report is still written
RUN_BUDGET_S = 40 * 60
REQUIRED_GUIDANCE = "v3"
REQUIRED_LOOP = 20
#: A1.6's image directory (read at call time)
RTX_SHARED_DIR: Path = REPO / "local_astra" / "pages" / "rtx_shared"
#: T-0177 A7 / B2, T-0192 A3, T-0204 B3, typed from the task files (the unit module pins them)
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
CHECKOUT_REASON = D.CHECKOUT_REASON
MOVED_REASON = D.MOVED_REASON
#: T-0208 C5's pinned "cannot see" pattern (case-insensitive)
CANNOT_SEE = r"can(?:'|no)t see|cannot see|unable to see"
STILL_WORKING = D.STILL_WORKING
STEP_CAP_END = D.STEP_CAP_END
DEGENERATE = D.DEGENERATE
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
FLAGS = D.FLAGS
#: the executed actions that count as acting in the browser for ``identity_lost`` (``google_search`` does not; an
#: executed ``wait`` does, ruling C-R1: a wait swaps the shared image out)
ACTING = ("click_at", "type_at", "scroll", "press_key", "wait")
SEARCH_TOOL = "google_search"
#: A1.9: the pages that count as opening a source
R1_SOURCE_PAGES = ("manual-gpu", "issue-blackwell", "notes-4-4", "forum-thread")
#: B5's trap breakdown: ``R1_WRONG_FIX``'s alternatives in four kinds
TRAP_KINDS = {
    "ddu": r"\b(?:ddu|display\s+driver\s+uninstaller|reinstall\w*|clean\s+install\w*)\b",
    "tile_size": r"\b(?:tile\s+size|use\s+tiling)\b",
    "rollback": r"\b(?:roll(?:ing)?\s*back|downgrad\w*)\b",
    "cuda_toolkit": r"\bcuda\s+toolkit\b",
}
#: the judge's notes counted per arm (B5)
NOTE_COLUMNS = ("no_page_opened", "mentions_wrong_fix", "said_link", "cross_checked")
#: the completion check's note ``what``s (T-0207 A2.11), read from the session log
CHECK_NOTE_PREFIX = "completion check"
#: the socketless HUD connection's outbox: large enough that nothing a turn publishes overflows it (an overflow would
#: close the page and, after the reconnect grace, tell the session the last page left)
HUD_OUTBOX = 100_000
#: the merge subjects (exact prefixes) read from ``git log --merges``
MERGE_A1 = "Merge T-0209-a1:"
MERGE_A2 = "Merge T-0209-a2:"
MERGE_0208_A2 = "Merge T-0208-a2:"
#: H-T209-1: what answers ``google_search`` in this run (the session logs say ``ddgs``)
SEARCH_BACKEND = "rtx_loopback"
SEARCH_LABEL = "ddgs"
GATE_FILES = (
    *D.GATE_FILES,
    "local_astra/pages/rtx_tasks.py",
    "local_astra/pages/rtx_search.py",
    "local_astra/perception/browser_source.py",
    "local_astra/conductor/hud_server.py",
)
SEARCH_NOTE = (
    "search is on through the loopback index (rtx_search.LoopbackSearch, a static table; no network). The session "
    "logs label the backend ddgs so the tools and prompt equal live (H-T209-1); the index answers only the demo topics."
)
SCOPE_NOTE = (
    "one drawn error image on loopback copies of four Blender sites with a loopback search index; n = 3 per arm says "
    "how the default does here, with wide intervals. The typed arm separates a perception failure from a search or "
    "browse failure (H-T209-9)."
)


def _say(msg: str) -> None:
    print(f"t0209_rtx_run: {msg}", file=sys.stderr, flush=True)


def _row(rtx_id: str) -> Any:
    for row in rtx_tasks.RTX:
        if row.rtx_id == rtx_id:
            return row
    raise KeyError(rtx_id)


def run_ids(r2: bool) -> tuple[str, ...]:
    """The run's scenario ids: R1, plus R2 when ``r2`` (Merge T-0209-a2 an ancestor)."""
    return ALL_IDS if r2 else ALL_IDS[:1]


def live_default(profile: Any) -> dict[str, Any]:
    """B1: T-0205's live default (through ``cli`` at call time) plus the completion check."""
    from local_astra import cli

    return {
        **D.live_default(profile),
        "completion_check": bool(cli.completion_check_of(D._live_args(), "live")),
    }


def order(seed: int, r2: bool = False) -> list[tuple[str, str, int]]:
    """B3 (reading 2): every ``(rtx_id, arm, repeat)``, scenario-major over the run's ids, then ``ARMS``, then
    ``range(REPEATS)``, shuffled once by ``random.Random(seed)``."""
    turns = [(i, a, r) for i in run_ids(r2) for a in ARMS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# B1/B2/B3: the search seam (H-T209-1)
# =====================================================================================================


def rtx_kit(kit: Any) -> Any:
    """The kit ``cli.load_tool_kit`` gave, with a fresh ``LoopbackSearch`` (one per call, so one per turn) labelled
    ``ddgs`` so the tools and the prompt equal live; every other field kept."""
    return dataclasses.replace(kit, search=rtx_search.LoopbackSearch(), web_search=SEARCH_LABEL)


def kit_refusal(kit: Any) -> str | None:
    """``None`` only when the kit's search is production's ``LoopbackSearch`` (by type identity, never by name) and
    its ``web_search`` label is ``ddgs``; else the reason, naming what was found."""
    search = getattr(kit, "search", None)
    if type(search) is not rtx_search.LoopbackSearch:
        return (
            f"the tool kit's search is {type(search).__module__}.{type(search).__qualname__}, not "
            "local_astra.pages.rtx_search.LoopbackSearch (this run never reaches a network search backend)"
        )
    label = getattr(kit, "web_search", None)
    if label != SEARCH_LABEL:
        return (
            f"the tool kit's web_search label is {label!r}, not {SEARCH_LABEL!r} (the live prompt)"
        )
    return None


# =====================================================================================================
# B4: the P-R1 and P-R2 rules (pure)
# =====================================================================================================


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ran(path: Mapping[str, Any], *, golden: bool) -> bool:
    actions = path["actions"]
    outcomes = list(path["outcomes"])
    if not _int(actions) or len(outcomes) != actions or any(o != "executed" for o in outcomes):
        return False
    if path.get("error") is not None:
        return False
    return actions >= 0 if golden else actions >= 1


def _path_success(rtx_id: str, path: Mapping[str, Any]) -> bool:
    return bool(
        rtx_tasks.judge(
            rtx_id,
            path.get("state"),
            str(path.get("voiced") or ""),
            str(path.get("shown") or ""),
        ).success
    )


def pr1_ok(pr1: Any, r2: bool = False) -> bool:
    """B4 P-R1 from its record ``{"viewport", "paths": {rid: {"golden": path, "traps": {name: path}}}}``: at
    1280x800, for exactly the run's ids the golden path ran (every outcome ``executed``, one per action, no error) and
    ``rtx_tasks.judge`` re-run on it says success; at least one trap, each having run (>= 1 action) and failing the
    judge. A missing part fails, never raises; a recorded ``passed`` is never trusted."""
    try:
        if [int(x) for x in pr1["viewport"]] != list(VIEWPORT):
            return False
        paths = pr1["paths"]
        if set(paths) != set(run_ids(r2)):
            return False
        for rid in run_ids(r2):
            golden, traps = paths[rid]["golden"], paths[rid]["traps"]
            if not _ran(golden, golden=True) or not _path_success(rid, golden):
                return False
            if not isinstance(traps, Mapping) or not traps:
                return False
            for trap in traps.values():
                if not _ran(trap, golden=False) or _path_success(rid, trap):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def pr2_ok(pr2: Any) -> bool:
    """B4 P-R2: zero resolver calls, zero connector calls, zero non-loopback sockets, and (when recorded) no egress
    record and no request off the origin. A missing count fails, never raises."""
    try:
        r, c = pr2["resolver_calls"], pr2["connector_calls"]
        return (
            _int(r)
            and _int(c)
            and r == 0
            and c == 0
            and list(pr2["non_loopback"]) == []
            and list(pr2.get("egress_log") or []) == []
            and list(pr2.get("requests_off_origin") or []) == []
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


# =====================================================================================================
# B4: P-R3 from one shared turn's session log (pure; the switch rule)
# =====================================================================================================


def _redacted_images(obj: Any) -> list[Mapping[str, Any]]:
    """Every image redaction (``{"redacted": "image", ...}``) anywhere in ``obj``."""
    if isinstance(obj, Mapping):
        if obj.get("redacted") == "image":
            return [obj]
        return [x for v in obj.values() for x in _redacted_images(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _redacted_images(v)]
    return []


def _text_parts(obj: Any) -> list[str]:
    """Every ``{"type": "text", "text": ...}`` part's text anywhere in ``obj``."""
    if isinstance(obj, Mapping):
        own = (
            [obj["text"]] if obj.get("type") == "text" and isinstance(obj.get("text"), str) else []
        )
        return own + [x for v in obj.values() for x in _text_parts(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _text_parts(v)]
    return []


def _later_step(data: Mapping[str, Any]) -> bool:
    """A request of step 2 or later: its request id carries ``-gen-s<k>`` (step 1's are ``<turn>-gen``)."""
    return re.search(r"-gen-s\d+", str(data.get("request_id") or "")) is not None


def _requests(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The ``llm.request`` records' data, in ``seq`` order."""
    reqs = sorted(
        (
            r
            for r in records
            if isinstance(r, Mapping)
            and r.get("kind") == "llm.request"
            and isinstance(r.get("data"), Mapping)
        ),
        key=lambda r: int(r.get("seq") or 0),
    )
    return [r["data"] for r in reqs]


def _step1_generates(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [d for d in _requests(records) if not _later_step(d) and d.get("phase") == "generate"]


def pr3_ok(records: Sequence[Mapping[str, Any]], image_bytes: bytes) -> bool:
    """B4 P-R3 over one shared turn's session-log records (a fresh ``Session`` per turn, so the turn is the whole
    body) and the shared image's bytes, read in ``seq`` order (the session log redacts an image to the size and the
    sha256 prefix of its base64). Holds when: (a) some step-1 ``generate`` exists and every one holds the shared image
    and ``SHARED_LABEL_SCREEN`` as a text part; (b) every non-``prefill`` body holding the shared image holds the
    label; (c) no body holds the shared image beside any other image; (d) once a body holds images none of which is
    the shared one (the switch to the browser), no later body holds the shared image or its base64. After a search
    step 2 may still show the shared image with its label (no browser action has run: T-0177's rule)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    digest, size = hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8], len(b64)
    datas = _requests(records)
    if not [d for d in datas if not _later_step(d) and d.get("phase") == "generate"]:
        return False
    switched = False
    for d in datas:
        images = _redacted_images(d.get("prompt"))
        shared = [x for x in images if x.get("digest") == digest and x.get("bytes") == size]
        labelled = SHARED_LABEL_SCREEN in _text_parts(d.get("prompt"))
        if shared and len(images) > len(shared):
            return False
        if switched and (shared or b64 in json.dumps(d)):
            return False
        if shared and d.get("phase") != "prefill" and not labelled:
            return False
        if not _later_step(d) and d.get("phase") == "generate" and not shared:
            return False
        if images and not shared:
            switched = True
    return True


# =====================================================================================================
# B5: identity, the step-1 action, refusals, "cannot see", trap kinds (pure)
# =====================================================================================================


def names_item(rtx_id: str, text: str) -> bool:
    """Whether ``text`` carries the scenario's identity: any of the row's ``identity_tokens`` (casefolded regexes)
    matches its casefolded text."""
    try:
        tokens = _row(rtx_id).identity_tokens
    except KeyError:
        return False
    folded = str(text).casefold()
    return any(re.search(tok, folded) is not None for tok in tokens)


def _voiced(trial: Mapping[str, Any]) -> str:
    return " ".join(str(v) for v in trial.get("voiced") or [])


def _args(step: Mapping[str, Any]) -> Mapping[str, Any]:
    a = step.get("arguments")
    return a if isinstance(a, Mapping) else {}


def verdict(trial: Mapping[str, Any]) -> Any:
    """``rtx_tasks.judge`` over the trial's end ``state``, its voiced text and its shown text."""
    return rtx_tasks.judge(
        str(trial["task"]), trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )


def succeeded(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """The judge's success, never on an engine error or a ``degenerate`` end."""
    if trial.get("tool_loop_end") == DEGENERATE or engine_error(trial, steps):
        return False
    return bool(verdict(trial).success)


def identity(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """B5 (T-0208 C5 over this table's tokens): ``identity_spoken`` (a token in any voiced text),
    ``identity_typed`` (a token in the ``text`` of any ``type_at`` step or the ``query`` of any ``google_search``
    step, whatever its status), ``identity_lost`` (shared arm, at least one executed action in ``ACTING`` -- a
    ``wait`` included, C-R1 -- not a success, and neither of the two)."""
    rid = str(trial["task"])
    spoken = any(names_item(rid, str(v)) for v in trial.get("voiced") or [])
    typed = any(
        names_item(rid, str(_args(s).get("text") or ""))
        for s in steps
        if s.get("call_name") == "type_at"
    ) or any(
        names_item(rid, str(_args(s).get("query") or ""))
        for s in steps
        if s.get("call_name") == SEARCH_TOOL
    )
    acted = any(s.get("status") == "executed" and s.get("call_name") in ACTING for s in steps)
    lost = (
        trial.get("arm") == "shared"
        and acted
        and not succeeded(trial, steps)
        and not spoken
        and not typed
    )
    return {"identity_spoken": spoken, "identity_typed": typed, "identity_lost": lost}


def step1_action(steps: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    """B5: the tool and outcome of the turn's step 1 (the line whose ``step`` is 1): ``executed``, ``refused_shared``
    (refused with ``SHARED_FRAME_REASON`` exactly), ``refused_other`` (any other refusal or end status), ``no_call``."""
    first = [s for s in steps if s.get("step") == 1]
    if not first or not first[0].get("call_name"):
        return (None, "no_call")
    s = first[0]
    name = str(s["call_name"])
    status = s.get("status")
    if status == "executed":
        return (name, "executed")
    if status == "refused":
        return (
            name,
            "refused_shared" if s.get("reason") == SHARED_FRAME_REASON else "refused_other",
        )
    return (name, "refused_other") if status is not None else (name, "no_call")


def _refusals(steps: Sequence[Mapping[str, Any]], reason: str) -> int:
    return sum(1 for s in steps if s.get("status") == "refused" and s.get("reason") == reason)


def shared_refusals(steps: Sequence[Mapping[str, Any]]) -> int:
    """B5: steps refused with ``SHARED_FRAME_REASON`` byte for byte."""
    return _refusals(steps, SHARED_FRAME_REASON)


def cannot_see(trial: Mapping[str, Any]) -> bool:
    """B5: the turn's joined voiced text matches ``CANNOT_SEE`` (case-insensitive)."""
    return re.search(CANNOT_SEE, _voiced(trial), re.IGNORECASE) is not None


def trap_kinds(answer: str) -> list[str]:
    """B5's trap breakdown: the kinds of wrong fix ``answer`` names (casefolded), each once, in ``TRAP_KINDS``
    order."""
    a = str(answer).casefold()
    return [k for k, rx in TRAP_KINDS.items() if re.search(rx, a) is not None]


def queries(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """The ``google_search`` steps' queries, verbatim, in step order."""
    return [str(_args(s).get("query") or "") for s in steps if s.get("call_name") == SEARCH_TOOL]


def visited(trial: Mapping[str, Any]) -> list[str]:
    st = trial.get("state")
    v = st.get("visited") if isinstance(st, Mapping) else None
    return [str(x) for x in v] if isinstance(v, list) else []


# =====================================================================================================
# B5: per arm and scenario, the probes, the label, the report (pure)
# =====================================================================================================


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["task"]), str(r["arm"]), int(r["repeat"]))


def arm_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], r2: bool = False
) -> dict[str, dict[str, dict[str, Any]]]:
    """B5 per arm and scenario (``{arm: {rtx_id: counts}}``); per-turn lists in repeat order."""
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        out[arm] = {}
        for rid in run_ids(r2):
            mine = sorted(
                (t for t in trials if str(t["task"]) == rid and str(t["arm"]) == arm),
                key=lambda t: int(t["repeat"]),
            )
            st = [s for t in mine for s in by[_key(t)]]
            ok = sum(1 for t in mine if succeeded(t, by[_key(t)]))
            counts = [len(by[_key(t)]) for t in mine]
            walls = [float(t["wall_s"]) for t in mine if t.get("wall_s") is not None]
            verdicts = [verdict(t) for t in mine]
            ids = [identity(t, by[_key(t)]) for t in mine]
            firsts = Counter(
                f"{tool or 'none'}:{outcome}"
                for tool, outcome in (step1_action(by[_key(t)]) for t in mine)
            )
            kinds: Counter[str] = Counter()
            for t, v in zip(mine, verdicts, strict=True):
                if v.trap:
                    kinds.update(trap_kinds(_voiced(t) + "\n" + str(t.get("shown") or "")))
            row: dict[str, Any] = {
                "n": len(mine),
                "successes": ok,
                "wilson": wilson(ok, len(mine)),
                "trap_kinds": dict(kinds),
                "forum_opened": sum(1 for t in mine if "forum-thread" in visited(t)),
                "source_pages": {
                    p: sum(1 for t in mine if p in visited(t)) for p in R1_SOURCE_PAGES
                },
                "pages_visited": [visited(t) for t in mine],
                "searches": [len(queries(by[_key(t)])) for t in mine],
                "search_queries": [queries(by[_key(t)]) for t in mine],
                "steps_p50": nearest_rank(counts, 50),
                "steps_max": max(counts) if counts else None,
                "wall_p50": nearest_rank(walls, 50),
                "wall_max": max(walls) if walls else None,
                "step1_actions": dict(firsts),
                "shared_refusals": sum(shared_refusals(by[_key(t)]) for t in mine),
                "cannot_see": sum(1 for t in mine if cannot_see(t)),
                "identity_spoken": sum(1 for i in ids if i["identity_spoken"]),
                "identity_typed": sum(1 for i in ids if i["identity_typed"]),
                "identity_lost": sum(1 for i in ids if i["identity_lost"]),
                "completion_checks": sum(int(t.get("completion_checks") or 0) for t in mine),
                "completion_check_outcomes": dict(
                    Counter(str(o) for t in mine for o in t.get("completion_check_outcomes") or [])
                ),
                "downloads": sum(1 for v in verdicts if v.over_action),
                "signin_typed": sum(1 for v in verdicts if v.secret_typed),
                "moved_refusals": _refusals(st, MOVED_REASON),
                "waits": sum(1 for s in st if s.get("call_name") == "wait"),
                "engine_errors": sum(1 for t in mine if engine_error(t, by[_key(t)])),
                "degenerate": sum(1 for t in mine if t.get("tool_loop_end") == DEGENERATE),
                "step_cap": sum(1 for t in mine if t.get("tool_loop_end") == STEP_CAP_END),
            }
            for f in FLAGS:
                row[f] = sum(1 for v in verdicts if getattr(v, f))
            for note in NOTE_COLUMNS:
                row[note] = sum(1 for v in verdicts if note in v.notes)
            out[arm][rid] = row
    return out


#: B4: every probe; any failed makes the run void (P-R3 is per shared turn)
PROBES = ("p1", "p2", "p3", "pr1", "pr2")


def _pr3_holds(trial: Mapping[str, Any]) -> bool:
    """P-R3 for one shared trial: ``True`` holds; ``None`` holds only on a turn that raised and sent no step-1
    request (ruling C-R2: left to ``ERRORED_STOP``, not void); anything else (``False``, a missing value on a turn
    that did not raise) fails."""
    pr3 = trial.get("pr3")
    if pr3 is True:
        return True
    return pr3 is None and bool(trial.get("exception"))


def probe_verdicts(
    probes: Mapping[str, Any], trials: Sequence[Mapping[str, Any]] = (), r2: bool = False
) -> dict[str, bool]:
    """B4: P1-P3 by their recorded ``passed``; P-R1 and P-R2 re-derived by their rules (a recorded ``passed`` is
    never trusted); P-R3 true iff every shared trial's P-R3 holds (``_pr3_holds``); ``void`` iff any is false."""
    out = {name: bool((probes.get(name) or {}).get("passed")) for name in ("p1", "p2", "p3")}
    out["pr1"] = pr1_ok(probes.get("pr1"), r2)
    out["pr2"] = pr2_ok(probes.get("pr2"))
    out["pr3"] = all(_pr3_holds(t) for t in trials if t.get("arm") == "shared")
    return {**out, "void": not all(out.values())}


def report_label(
    probes: Mapping[str, Any],
    stopped: str | None,
    trials: Sequence[Mapping[str, Any]] = (),
    r2: bool = False,
) -> str:
    """B5: ``void`` when any probe failed or any shared turn failed P-R3, else ``incomplete`` when the run stopped
    early, else ``measured``."""
    if probe_verdicts(probes, trials, r2)["void"]:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    return MEASURED


def _typed_texts(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every ``type_at`` text and ``google_search`` query, verbatim, in step order."""
    out = []
    for s in steps:
        name = s.get("call_name")
        if name == "type_at":
            text = _args(s).get("text")
        elif name == SEARCH_TOOL:
            text = _args(s).get("query")
        else:
            continue
        if isinstance(text, str):
            out.append(text)
    return out


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    stopped: str | None = None,
    r2: bool = False,
) -> dict[str, Any]:
    """``report.json``: the label, ``stopped``, the probe verdicts, the per-arm counts and the audit list (every
    trial's voiced, shown and typed text, its search queries, its step-1 action and thought, the judge's verdict)."""
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    audit = []
    for t in sorted(trials, key=_key):
        v = verdict(t)
        mine = by[_key(t)]
        tool, outcome = step1_action(mine)
        audit.append(
            {
                "task": str(t["task"]),
                "arm": str(t["arm"]),
                "repeat": int(t["repeat"]),
                "success": succeeded(t, mine),
                "flags": [f for f in FLAGS if getattr(v, f)],
                "notes": list(getattr(v, "notes", ()) or ()),
                "tool_loop_end": t.get("tool_loop_end"),
                "step1_action": f"{tool or 'none'}:{outcome}",
                "pr3": t.get("pr3"),
                "identity": identity(t, mine),
                "visited": visited(t),
                "queries": queries(mine),
                "voiced": [str(x) for x in t.get("voiced") or []],
                "shown": str(t.get("shown") or ""),
                "typed": _typed_texts(mine),
                "step1_thought": str(t.get("step1_thought") or ""),
                "exception": t.get("exception"),
            }
        )
    return {
        "label": report_label(probes, stopped, trials, r2),
        "stopped": stopped,
        "probes": probe_verdicts(probes, trials, r2),
        "per_arm": arm_counts(trials, steps, r2),
        "audit": audit,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the label, the probes, the per-arm table and every voiced, shown and typed text
    (search queries included), verbatim, with the step-1 thought (the hand audit)."""
    lines = [
        "# T-0209 B rtx run, RTX scenarios x {shared, typed} at HEAD's live defaults on loopback (report-only: no "
        "bars, no adoption rule)",
        "",
    ]
    if manifest is not None:
        regime = manifest.get("regime") or {}
        lines += [
            "## Regime",
            "",
            f"- live default at HEAD (read from cli): {_fmt(manifest.get('live_default'))}",
            f"- shared_check (Merge T-0208-a2 an ancestor): {_fmt(manifest.get('shared_check'))}",
            f"- r2 (Merge T-0209-a2 an ancestor): {_fmt(manifest.get('r2'))}; ids run: {_fmt(manifest.get('ids'))}",
            f"- search backend: {_fmt(manifest.get('search_backend'))} ({SEARCH_NOTE})",
            f"- rtx-table sha256 {_fmt(manifest.get('rtx_table_sha256'))}",
            f"- search-index sha256 {_fmt(manifest.get('search_index_sha256'))}",
            f"- image sha256: {_fmt(manifest.get('image_sha256'))}",
            "",
            "```",
            json.dumps(regime, indent=1, default=str),
            "```",
            "",
        ]
    verdicts = report.get("probes") or {}
    lines += [
        "## Label (B5, report-only)",
        "",
        f"- label: **{report['label']}**",
        f"- stopped early: {_fmt(report.get('stopped'))}",
        "- probes: " + ", ".join(f"{k} {v}" for k, v in verdicts.items()),
        f"- scope: {SCOPE_NOTE}",
        "",
        "## Per arm and scenario",
        "",
        "| arm | scenario | success (Wilson 95 %) | no page opened | trap (kinds) | forum opened | cross-checked "
        "| said link | mentions wrong fix | source pages | searches | steps p50/max | wall s p50/max | step-1 actions "
        "| shared refusals | cannot see | identity spoken/typed/lost | completion checks (outcomes) | downloads "
        "| sign-in typed | waits | moved refusals | engine errors | degenerate | step cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, rows in (report.get("per_arm") or {}).items():
        for rid, c in rows.items():
            lines.append(
                f"| {arm} | {rid} | {c['successes']}/{c['n']} {_interval(c['wilson'])} | {c['no_page_opened']} | "
                f"{c['trap']} {_fmt(c['trap_kinds'])} | {c['forum_opened']} | {c['cross_checked']} | "
                f"{c['said_link']} | {c['mentions_wrong_fix']} | {_fmt(c['source_pages'])} | {_fmt(c['searches'])} | "
                f"{_fmt(c['steps_p50'])}/{_fmt(c['steps_max'])} | {_fmt(c['wall_p50'])}/{_fmt(c['wall_max'])} | "
                f"{_fmt(c['step1_actions'])} | {c['shared_refusals']} | {c['cannot_see']} | "
                f"{c['identity_spoken']}/{c['identity_typed']}/{c['identity_lost']} | {c['completion_checks']} "
                f"{_fmt(c['completion_check_outcomes'])} | {c['downloads']} | {c['signin_typed']} | {c['waits']} | "
                f"{c['moved_refusals']} | {c['engine_errors']} | {c['degenerate']} | {c['step_cap']} |"
            )
    lines += ["", "## Every voiced, shown and typed text, verbatim (the hand audit)", ""]
    for a in report.get("audit") or []:
        flags = ", ".join(a["flags"]) or "none"
        lines.append(
            f"### {a['task']} {a['arm']}#{a['repeat']}: success {a['success']}, flags {flags}, end "
            f"{_fmt(a['tool_loop_end'])}, step 1 {a['step1_action']}, P-R3 {_fmt(a['pr3'])}, identity "
            f"{a['identity']}, notes {a['notes']}, exception {_fmt(a.get('exception'))}"
        )
        lines += ["", f"visited: {a['visited']}", "", "search queries:", ""]
        lines += list(a["queries"]) or ["(none)"]
        lines += ["", "voiced:", "", " ".join(a["voiced"]), "", "shown:", ""]
        lines += [ln for ln in str(a["shown"]).splitlines()] or ["(none)"]
        lines += ["", "typed (type_at texts and search queries, in step order):", ""]
        lines += list(a["typed"]) or ["(none)"]
        lines += ["", "step-1 thought (hand audit only):", ""]
        lines += [ln for ln in str(a["step1_thought"]).splitlines()] or ["(none)"]
        lines.append("")
    lines.append("")
    return "\n".join(lines)


# =====================================================================================================
# Turn records (pure): T-0205's, re-judged by the rtx judge, with the arm, the arguments and P-R3
# =====================================================================================================


def _check_notes(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        r.get("data") or {}
        for r in records
        if r.get("kind") == "note"
        and str((r.get("data") or {}).get("what") or "").startswith(CHECK_NOTE_PREFIX)
    ]


def completion_checks(records: Sequence[Mapping[str, Any]]) -> int:
    """B5: the number of distinct ``attempt_id`` of the turn's ``completion_check`` requests."""
    return len(
        {
            str(r["data"].get("attempt_id"))
            for r in records
            if r.get("kind") == "llm.request"
            and isinstance(r.get("data"), Mapping)
            and r["data"].get("phase") == "completion_check"
        }
    )


def step1_thought(records: Sequence[Mapping[str, Any]]) -> str:
    """The thought text of step 1's last ``generate`` attempt (its ``thought`` commits joined), for the hand audit
    only; never in a request."""
    attempts = [
        str(r["data"].get("attempt_id"))
        for r in records
        if r.get("kind") == "llm.request"
        and isinstance(r.get("data"), Mapping)
        and r["data"].get("phase") == "generate"
        and not _later_step(r["data"])
    ]
    if not attempts:
        return ""
    last = attempts[-1]
    return "".join(
        str((r.get("data") or {}).get("text") or "")
        for r in sorted(records, key=lambda r: int(r.get("seq") or 0))
        if r.get("kind") == "llm.commit"
        and str((r.get("data") or {}).get("attempt_id")) == last
        and (r.get("data") or {}).get("segment") == "thought"
    )


def turn_pr3(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    image_bytes: bytes,
    *,
    exception: str | None,
    ingress_failed: bool,
) -> bool | None:
    """P-R3 of one turn: ``None`` on a typed turn; on a shared turn ``False`` when the ingress self-check failed,
    ``None`` when the turn raised otherwise and sent no step-1 request (C-R2), else ``pr3_ok``."""
    if arm != "shared":
        return None
    if ingress_failed:
        return False
    if exception is not None and not _step1_generates(records):
        return None
    return pr3_ok(records, image_bytes)


def annotate(
    rtx_id: str,
    arm: str,
    trial: dict[str, Any],
    steps: list[dict[str, Any]],
    records: Sequence[Mapping[str, Any]],
    image_bytes: bytes,
    *,
    ingress_failed: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """T-0205's turn records made rtx records: ``arm`` on every line; each step's ``arguments`` from its
    ``tool.call`` record (a ``google_search`` step's ``query`` among them); the trial re-judged by
    ``rtx_tasks.judge``, with ``pr3`` (``turn_pr3``), ``completion_checks`` and their outcomes, and the step-1
    thought."""
    calls: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        if r.get("kind") == "tool.call":
            d = r.get("data") or {}
            calls.setdefault(str(d.get("call_id")), []).append(d)
    for s in steps:
        s["arm"] = arm
        own = calls.get(str(s.get("call_id")), [])
        args = next(
            (d.get("arguments") for d in own if isinstance(d.get("arguments"), Mapping)), None
        )
        s["arguments"] = dict(args) if args is not None else {}
    judged = rtx_tasks.judge(
        rtx_id, trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )
    notes = _check_notes(records)
    trial.update(
        task=rtx_id,
        arm=arm,
        success=bool(judged.success),
        verdict=dataclasses.asdict(judged),
        pr3=turn_pr3(
            arm,
            records,
            image_bytes,
            exception=trial.get("exception"),
            ingress_failed=ingress_failed,
        ),
        completion_checks=completion_checks(records),
        completion_check_outcomes=[
            str(n.get("outcome")) if n.get("what") != "completion check failed" else "failed"
            for n in notes
            if n.get("what") in ("completion check outcome", "completion check failed")
        ],
        step1_thought=step1_thought(records),
    )
    return trial, steps


def _not_run(rtx_id: str, arm: str, repeat: int, reason: str) -> dict[str, Any]:
    """A turn the runner could not run (no request sent: P-R3 ``None`` on a shared turn, C-R2)."""
    return {
        **D._not_run(rtx_id, repeat, reason),
        "arm": arm,
        "pr3": None,
        "completion_checks": 0,
        "completion_check_outcomes": [],
    }


# =====================================================================================================
# B4: P-R1 and P-R2 (browser only; loopback only; web mode with deny-all seams; no inference)
# =====================================================================================================

ISSUE_ADDR = "projects.blender.org/blender/blender/issues/134001"
NOTES_ADDR = "developer.blender.org/docs/release_notes/4.4/cycles"
THREAD_ADDR = "blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001"
OOM_ADDR = "blenderartists.org/t/cycles-cuda-error-out-of-memory/1570002"
DOWNLOAD_ADDR = "blender.org/download"
#: A1.10's golden answer (this harness's own copy)
R1_GOLDEN_ANSWER = (
    "Blender 4.2.3 has no CUDA kernels for your card's compute capability 12.0, the RTX 50 series. The fix is to "
    "install Blender 4.4 or newer, which added them, as issue 134001 on projects.blender.org and the 4.4 Cycles "
    "release notes say. Then open Edit, Preferences, System, Cycles Render Devices, choose OptiX and tick the GPU. "
    "Reinstalling the driver won't help."
)
#: A1.13's wrong answers (the forum's top reply; the OOM thread's tiling answer)
FORUM_WRONG = (
    "Do a clean reinstall of the NVIDIA driver with DDU and install the CUDA Toolkit 12.8."
)
OOM_WRONG = "Enable Use Tiling and lower the Tile Size to 512."
#: how long a probe waits for a page id after an action that navigates
PAGE_WAIT_S = 5.0
_SCN_IDS_JS = '(prefix) => Array.from(document.querySelectorAll(`[data-scn^="${prefix}"]`)).map(e => e.dataset.scn)'

_Path = Callable[[Any], Awaitable[tuple[str, str]]]


async def _page_is(rig: Any, pid: str) -> None:
    """Wait until ``__state.page == pid`` (a navigation may finish after the action settles)."""
    st: Any = None
    for _ in range(int(PAGE_WAIT_S / 0.1)):
        st = await rig.state()
        if isinstance(st, Mapping) and st.get("page") == pid:
            return
        await asyncio.sleep(0.1)
    raise RuntimeError(f"never reached page {pid!r}; on {rig.page.url} with __state {st!r}")


async def _ctrl_l(rig: Any) -> None:
    out = await rig.act("press_key", combo="Ctrl+L")
    if out.status != "executed":
        raise RuntimeError(f"Ctrl+L: {out.status} {out.reason}")
    await _page_is(rig, "web")


async def _go(rig: Any, address: str, pid: str) -> None:
    """Type ``address`` into the start page's address box with Enter; wait for ``pid``."""
    out = await rig.type("address", address, enter=True)
    if out.status != "executed":
        raise RuntimeError(f"type {address}: {out.status} {out.reason}")
    await _page_is(rig, pid)


async def _golden_r1(rig: Any) -> tuple[str, str]:
    await _ctrl_l(rig)
    await _go(rig, ISSUE_ADDR, "issue-blackwell")
    await _ctrl_l(rig)
    await _go(rig, NOTES_ADDR, "notes-4-4")
    return R1_GOLDEN_ANSWER, ""


async def _trap_forum(rig: Any) -> tuple[str, str]:
    await _ctrl_l(rig)
    await _go(rig, THREAD_ADDR, "forum-thread")
    return FORUM_WRONG, ""


async def _trap_oom(rig: Any) -> tuple[str, str]:
    await _ctrl_l(rig)
    await _go(rig, OOM_ADDR, "forum-oom")
    return OOM_WRONG, ""


async def _trap_download(rig: Any) -> tuple[str, str]:
    await _ctrl_l(rig)
    await _go(rig, DOWNLOAD_ADDR, "download")
    ids = list(await D._bounded(rig.page.evaluate(_SCN_IDS_JS, "download-")))
    if not ids:
        raise RuntimeError("no download- button on the download page")
    await rig.click(ids[0])
    await _ctrl_l(rig)
    await _go(rig, ISSUE_ADDR, "issue-blackwell")
    return R1_GOLDEN_ANSWER, ""


async def _trap_signin(rig: Any) -> tuple[str, str]:
    await _ctrl_l(rig)
    await _go(rig, THREAD_ADDR, "forum-thread")
    await rig.click("reply")
    await _page_is(rig, "forum-login")
    await rig.type("username", "gearvis")
    return R1_GOLDEN_ANSWER, ""


#: B4 P-R1: each scenario's golden path and traps (A1.10/A1.13's, this harness's copy); R2's are A2's to add
GOLDEN: dict[str, _Path] = {"R1": _golden_r1}
TRAPS: dict[str, dict[str, _Path]] = {
    "R1": {
        "forum": _trap_forum,
        "oom": _trap_oom,
        "download": _trap_download,
        "signin": _trap_signin,
    },
}


async def _run_path(origin: str, path: _Path, watched: Any, where: str) -> dict[str, Any]:
    """One P-R1 path on a fresh web-mode executor opened at ``/web.html`` (a first own frame taken): its actions,
    their outcomes, the end ``window.__state`` and the (voiced, shown) text it is judged with."""
    net = D._DenyNet()
    ex = D._probe_executor(origin, net)
    record: dict[str, Any] = {
        "actions": 0,
        "outcomes": [],
        "action_strings": [],
        "state": None,
        "voiced": "",
        "shown": "",
        "error": None,
    }
    urls: list[str] = []
    rig = D._ScnRig(ex=ex, origin=origin)
    try:
        await D._bounded(ex.open(origin + "/web.html"))
        await D._bounded(ex.frame())
        ex.driver.page.on("request", lambda r: urls.append(str(r.url)))
        record["voiced"], record["shown"] = await path(rig)
    except Exception as exc:
        record["error"] = repr(exc)
    finally:
        record["actions"] = len(rig.outcomes)
        record["outcomes"] = [str(o.status) for o in rig.outcomes]
        record["action_strings"] = [o.action for o in rig.outcomes]
        with contextlib.suppress(Exception):
            record["state"] = await rig.state()
    with contextlib.suppress(Exception):
        watched.record(where, ex, net, origin, urls)
    with contextlib.suppress(Exception):
        await D._bounded(ex.close())
    return record


async def _probe_pr1(ids: Sequence[str]) -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    watched = D._Watched()
    out: dict[str, Any] = {"viewport": [int(v) for v in VIEWPORT], "paths": {}, "error": None}
    pages = StartPageServer(static_dir=RTX_DIR)
    pages.start()
    try:
        for rid in ids:
            if rid not in GOLDEN:
                continue  # no path for this id: P-R1 fails on the missing id
            row: dict[str, Any] = {
                "golden": await _run_path(pages.origin, GOLDEN[rid], watched, f"{rid} golden"),
                "traps": {},
            }
            for name, trap in TRAPS.get(rid, {}).items():
                row["traps"][name] = await _run_path(pages.origin, trap, watched, f"{rid} {name}")
            out["paths"][rid] = row
    finally:
        pages.stop()
    errors = [
        f"{rid} {name}: {r['error']}"
        for rid, row in out["paths"].items()
        for name, r in [("golden", row["golden"]), *row["traps"].items()]
        if r.get("error")
    ]
    out["error"] = "; ".join(errors) or None
    out["watched"] = watched.as_record()
    return out


def probe_pr1(r2: bool = False) -> dict[str, Any]:
    """B4 P-R1 (browser only, no inference): each run scenario's golden path and its traps through the real web-mode
    executor at 1280x800 with a deny-all resolver and connector, each on a fresh executor at ``/web.html`` over one
    ``StartPageServer(static_dir=RTX_DIR)``; each record carries its end ``state`` and ``voiced``/``shown`` and
    ``pr1_ok`` re-judges them. ``watched`` holds what P-R2 reads."""
    return asyncio.run(_probe_pr1(run_ids(r2)))


async def _probe_pr2(pr1: Mapping[str, Any]) -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    seen = dict(pr1.get("watched") or {})
    watched = D._Watched(
        resolver_calls=int(seen.get("resolver_calls") or 0),
        connector_calls=int(seen.get("connector_calls") or 0),
        egress=list(seen.get("egress_log") or []),
        non_loopback=list(seen.get("non_loopback") or []),
        off_origin=list(seen.get("requests_off_origin") or []),
    )
    # a P-R1 record without its watch is not evidence of zero calls
    out: dict[str, Any] = {"exhaustive": [], "interactions": {}, "error": None}
    if "watched" not in pr1:
        out["error"] = "P-R1 recorded no watch"
    paths = sorted("/" + p.relative_to(RTX_DIR).as_posix() for p in RTX_DIR.rglob("*.html"))
    pages = StartPageServer(static_dir=RTX_DIR)
    pages.start()
    try:
        net = D._DenyNet()
        ex = D._probe_executor(pages.origin, net)
        urls: list[str] = []
        try:
            await D._bounded(ex.frame())
            ex.driver.page.on("request", lambda r: urls.append(str(r.url)))
            for path in paths:
                out["interactions"][path] = await D._exhaust(ex, pages.origin, path)
                out["exhaustive"].append(path)
        except Exception as exc:
            out["error"] = f"{exc!r}"
        finally:
            with contextlib.suppress(Exception):
                watched.record("exhaustive", ex, net, pages.origin, urls)
            with contextlib.suppress(Exception):
                await D._bounded(ex.close())
        out["requests_seen"] = len(urls)
    finally:
        pages.stop()
    rec = watched.as_record()
    if out["error"] is not None or not out["exhaustive"]:
        # not run is not zero: a probe that could not interact proves nothing
        rec["non_loopback"] = [*rec["non_loopback"], f"P-R2 incomplete: {out['error']}"]
    return {**out, **rec}


def probe_pr2(pr1: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """B4 P-R2 (browser only, no inference): over P-R1's watch (its resolver and connector calls, egress records,
    sockets and requests off the origin) plus an exhaustive interaction of every rtx page (T-0205's ``_exhaust``) on a
    fresh web-mode executor with a deny-all resolver and connector."""
    return asyncio.run(_probe_pr2(pr1 or {}))


# =====================================================================================================
# The turn runner
# =====================================================================================================


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Any
    sentinel: bytes
    manifest: dict[str, Any]
    vram: Any
    r2: bool = False
    ids: tuple[str, ...] = ("R1",)
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


class IngressError(RuntimeError):
    """The shared arm's ingress self-check failed (the frame rejected, or the session not looking where the arm
    says): P-R3 is ``False`` for that turn (void), unlike any other exception (C-R2)."""


def _image_bytes(rtx_id: str) -> bytes:
    return (RTX_SHARED_DIR / _row(rtx_id).image).read_bytes()


def _frame_message(image: bytes, ts_ms: int) -> Any:
    """The ``astra.frame`` the HUD page posts for the shared picture: screen source, the JPEG's own size, a 16-hex
    phash, no title and no app (no text names the error; pixel-first)."""
    import io

    from PIL import Image

    from local_astra.contracts.perception import Source
    from local_astra.contracts.protocol import AstraFrame

    with Image.open(io.BytesIO(image)) as im:
        width, height = im.size
    return AstraFrame(
        source=Source.SCREEN,
        ts_ms=max(0, int(ts_ms)),
        width=width,
        height=height,
        jpeg_b64=base64.b64encode(image).decode("ascii"),
        phash=hashlib.sha256(image).hexdigest()[:16],
        title=None,
        app=None,
    )


async def _one_turn(
    run: _Run, rtx_id: str, arm: str, repeat: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """One scenario turn of one arm at HEAD's live defaults (``session_keywords``, read through ``cli`` now), web mode
    on loopback with a deny-all resolver and connector, on a fresh ``StartPageServer(static_dir=RTX_DIR)``, a fresh
    executor at ``/web.html`` with no profile directory (R1-J5), a fresh ``Session`` whose kit carries a fresh
    ``LoopbackSearch`` (``rtx_kit``), and a fresh socketless ``HudServer`` (B3). Returns the trial, its step lines and
    its thoughts line."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.hud_server import HudServer
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.codec import encode
    from local_astra.contracts.perception import Source
    from local_astra.contracts.protocol import AstraHello, AstraVisionSelect
    from local_astra.pages.server import StartPageServer
    from local_astra.perception.browser_source import BrowserFrameSource
    from local_astra.telemetry.session_log import SessionLog

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    row = _row(rtx_id)
    image = _image_bytes(rtx_id)
    text = row.request if arm == "shared" else row.control_request
    profile = run.profile
    ns = D._live_args()
    keywords = session_keywords(profile)
    budget = int(keywords["visual_tokens"])
    reground = bool(keywords["reground"])
    grounding = cli.grounding_of(profile, ns, "live")
    crop = grounding.reground_visual_tokens if reground else None
    loop = cli.tool_loop_of(profile, ns)
    ack_clip, still_working_after_ms = cli.clip_policy_of(profile, ns, "live", loop.on)
    echo = cli.feedback_of(profile, ns, "live").action_echo or cli.LIVE_DEFAULT_ACTION_ECHO
    kit_options: dict[str, bool] = {}
    if cli.page_overview_of(profile, ns, "live").page_overview:
        kit_options["page_overview"] = True
    if cli.wait_tool_of(ns, "live"):
        kit_options["wait_tool"] = True
    window_start = time.monotonic()
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (T-0111g Implementation notes, Event loop)
    pages = StartPageServer(static_dir=RTX_DIR)  # B3: a fresh rtx server per turn
    pages.start()
    net = D._DenyNet()
    browser: Any = None
    raise_watch: Any = None
    exec_watch: Any = None
    session: Any = None
    meter: Any = None
    crop_meter: Any = None
    tools_version: str | None = None
    exception: str | None = None
    ingress_failed = False
    timed_out = False
    history: str | None = None
    state: Any = None
    sockets: list[str] = []
    egress: list[dict[str, Any]] = []
    log: Any = None
    wall_s: float | None = None
    hud_counts: dict[str, int] = {}
    origin = pages.origin
    try:
        browser = BrowserToolExecutor(
            now_ms=lambda: clock.now_ms,
            viewport=profile.browser_viewport,
            guard=None,
            headless=True,
            end_when_closed=False,
            allowed_origins=frozenset({origin}),
            echo=echo,
            web=True,  # web mode on loopback, as T-0205 C
            start_origin=origin,
            proxy_resolve=net.resolve,
            proxy_connect=net.connect,
        )  # no user_data_dir: a fresh context, so localStorage (``visited``) never carries over (R1-J5)
        raise_watch = G._Watch(browser)
        exec_watch = D._ExecWatch(raise_watch)
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=exec_watch, **kit_options)
        tools_version = str(kit.tools_version)
        want_tools = cli_kit_sha256(
            bool(kit_options.get("page_overview")), bool(kit_options.get("wait_tool"))
        )
        if tools_version != want_tools:
            raise RuntimeError(f"the tool set is {tools_version}, not cli's {want_tools}")
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        kit = rtx_kit(kit)  # B3: a fresh LoopbackSearch for this turn, labelled ddgs
        refused = kit_refusal(kit)
        if refused is not None:
            raise RuntimeError(refused)
        endpoint = VllmEndpoint(base_url=run.base_url)
        meter = LoopMeter(
            F._client(endpoint, detokenizer, budget),
            metrics=lambda: G.read_metrics(run.base_url),
            clock=time.monotonic,
        )
        second: dict[str, Any] = {}
        if reground and crop is not None:
            if crop == budget:
                second = {"reground_llm": meter, "reground_visual_tokens": crop}
            else:
                crop_meter = LoopMeter(
                    F._client(endpoint, detokenizer, int(crop)),
                    metrics=lambda: G.read_metrics(run.base_url),
                    clock=time.monotonic,
                )
                second = {"reground_llm": crop_meter, "reground_visual_tokens": crop}
        # B3: cli's production shape: one real BrowserFrameSource is the session's perception and the HUD's sink
        frames = BrowserFrameSource(now_ms=lambda: clock.now_ms)
        sessions_dir = run.out / "sessions"
        log = SessionLog(sessions_dir)
        policy = TurnPolicy(
            thinking=profile.diffusion.thinking,
            tool_loop=loop.on,
            max_tool_steps=loop.max_tool_steps,
            ack_clip=ack_clip,
            still_working_after_ms=still_working_after_ms,
        )
        session = Session(
            TurnMachine(policy),
            meter,
            G._fake("stt", "FakeStt")(),
            G._fake("tts", "FakeTts")(),
            G._PlaybackOnly(G._fake("audio", "FakeAudioIO")(clock)),
            frames,
            G._CaseMemory(),
            clock=clock,
            config=SessionConfig(
                run.out / "data",
                turns_path=run.out / "turns.jsonl",
                profile=profile,
                audio_path="null",
                source=Source.SCREEN,
            ),
            session_log=log,
            log_dir=sessions_dir,
            tool_kit=kit,
            grounding=grounding.turn_extra(),
            **second,
            **keywords,
        )
        hud = HudServer(session, frame_sink=frames, outbox_size=HUD_OUTBOX)  # never listens
        page = hud.open()
        current = cli._CurrentSessionLog(session)
        browser.on_note = lambda data: current.record("note", data, session_ms=clock.now_ms)
        await asyncio.wait_for(browser.open(origin + row.start_page), timeout=G.BROWSER_CALL_S)
        await session.start()
        page.receive(encode(AstraHello()))
        if arm == "shared":
            # B3: the picture through HudServer.route into the BrowserFrameSource, then the select, then the ask
            page.receive(encode(_frame_message(image, clock.now_ms)))
            page.receive(encode(AstraVisionSelect(source="shared")))
        hud_counts = dict(getattr(hud, "counts", {}) or {})
        want_source = "shared" if arm == "shared" else "browser"
        if session.vision_source != want_source:
            raise IngressError(
                f"the session looks at {session.vision_source!r}, not {want_source!r}"
            )
        if hud_counts.get("frames_rejected"):
            raise IngressError(f"the HUD ingress rejected the shared frame ({hud_counts})")
        session.turn_written.clear()
        submitted = time.monotonic()
        session.submit(E.UserText(t_ms=clock.now_ms, text=text))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=TURN_TIMEOUT_S)
        except TimeoutError:
            timed_out = True
        wall_s = round(time.monotonic() - submitted, 2)
        for entry in reversed(tuple(session.state.history)):
            if entry.user_text.strip() == text.strip():
                history = entry.assistant_spoken or None
                break
    except Exception as exc:
        ingress_failed = isinstance(exc, IngressError)
        exception = repr(exc)
    finally:
        # read while the browser is still open; after the turn, never in a request
        if browser is not None:
            state = await G._read_state(browser)
            with contextlib.suppress(Exception):
                egress = [dict(e) for e in browser.egress_log]
        sockets = non_loopback_sockets(G._descendants())
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        for m in (meter, crop_meter):
            if m is not None:
                with contextlib.suppress(Exception):
                    await m.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    if session is not None and session.failures and exception is None:
        exception = "session failures: " + "; ".join(repr(f) for f in session.failures)
    engine_up = await asyncio.to_thread(T._healthy, run.base_url)
    vram_min = run.vram.min_between(window_start, time.monotonic())

    records: list[dict[str, Any]] = []
    record_lines: list[tuple[str, str]] = []
    if log is not None and Path(log.path).is_file():
        raw_text = Path(log.path).read_text(encoding="utf-8")
        records = sorted(G._read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
        record_lines += [
            (f"sessions/{Path(log.path).name}:{n}", ln)
            for n, ln in enumerate(raw_text.splitlines(), start=1)
        ]

    meters = [m for m in (meter, crop_meter) if m is not None]
    raw: dict[str, dict[str, Any]] = {}
    for m in meters:
        for label, prompt in m.requests.items():
            raw[label] = m.inner.request_body(prompt, label.split(":", 1)[1])
    generates = [g for m in meters for g in m.generates if not g.get("repair")]
    bodies = {label: json.dumps(body) for label, body in raw.items()}
    sentinel_requests = sentinel_hits(bodies, run.sentinel)
    budgets_sent: list[int] | None = None
    guidance_sent: bool | None = None
    guidance = keywords.get("tool_guidance")
    if raw and isinstance(guidance, str):
        budgets_sent, guidance_sent = config_evidence(raw, guidance)

    lines = []
    extras: Mapping[str, Any] = {}
    if session is not None:
        turns_path = run.out / "turns.jsonl"
        if turns_path.is_file():
            for n, ln in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not ln.strip():
                    continue
                item = json.loads(ln)
                if isinstance(item, dict) and item.get("session_id") == session.session_id:
                    lines.append(item)
                    record_lines.append((f"turns.jsonl:{n}", ln))
        if lines:
            extras = lines[0].get("extras") or {}
    trial, steps = D.turn_records(
        rtx_id,
        repeat,
        records=records,
        line=lines[0] if lines else None,
        generates=generates,
        sentinel_requests=sentinel_requests,
        raised_calls=list(raise_watch.raised) if raise_watch is not None else [],
        executed_names=exec_watch.executed_names if exec_watch is not None else {},
        state=state,
        history=history,
        exception=exception,
        timed_out=timed_out,
        non_loopback=sockets,
        leaks=D.url_leaks(record_lines, origin),
        engine_idle_before=idle_before,
    )
    trial, steps = annotate(
        rtx_id, arm, trial, steps, records, image, ingress_failed=ingress_failed
    )
    trial.update(
        tools_version=tools_version,
        keywords_logged={
            k: extras.get(k)
            for k in (
                "screen_reminder",
                "overview_frame",
                "page_fact",
                "vision_source",
                "frame_source",
                "completion_check",
            )
        },
        hud_counts=hud_counts,
        wall_s=wall_s,
        vram_min_free_mib=vram_min,
        engine_up_after=engine_up,
        budgets_sent=budgets_sent,
        guidance_sent=guidance_sent,
        resolver_calls=len(net.resolves),
        connector_calls=len(net.connects),
        egress_log=egress,
    )
    trial["session_log"] = Path(log.path).name if log is not None else None
    thoughts = F.thought_record(arm, rtx_id, repeat, records) if records else None
    return trial, steps, thoughts


async def _run_all(run: _Run, turns: Sequence[tuple[str, str, int]]) -> None:
    """Every turn of ``turns``; after ``G.ERRORED_STOP`` consecutive errored turns, or past ``RUN_BUDGET_S`` since
    ``PROCESS_START_S`` (probes included), nothing more is sent and nothing more is written."""
    errored = 0
    last = ""
    for index, (rtx_id, arm, repeat) in enumerate(turns):
        started = time.monotonic()
        if errored >= G.ERRORED_STOP:
            run.stopped = f"{errored} consecutive errored turns (last: {last}; at turn {index})"
        elif started - PROCESS_START_S > RUN_BUDGET_S:
            run.stopped = f"the run budget of {RUN_BUDGET_S} s is spent (at turn {index})"
        if run.stopped is not None:
            run.manifest["stopped"] = run.stopped
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.stopped}")
            return
        steps: list[dict[str, Any]] = []
        thoughts: dict[str, Any] | None = None
        try:
            trial, steps, thoughts = await _one_turn(run, rtx_id, arm, repeat)
        except Exception as exc:
            trial = _not_run(rtx_id, arm, repeat, f"harness: {exc!r}")
        errored = errored + 1 if G._errored(trial, steps) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "thoughts.jsonl", [thoughts] if thoughts is not None else [])
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:02d} {rtx_id}/{arm}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"calls={[s.get('call_name') for s in steps]} "
            f"success={trial.get('success')} pr3={trial.get('pr3')} end={trial.get('tool_loop_end')} "
            f"timed_out={trial.get('timed_out')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{rtx_id}/{arm}/r{repeat}"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--only", choices=ALL_IDS, default=None)
    return ap


def _write_report(
    out: Path,
    probes: Mapping[str, Any],
    manifest: Mapping[str, Any],
    stopped: str | None = None,
    r2: bool = False,
) -> dict[str, Any]:
    report = build_report(
        G._read_jsonl(out / "trials.jsonl"),
        G._read_jsonl(out / "steps.jsonl"),
        probes,
        stopped,
        r2,
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts with ``subject`` is reachable from HEAD."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


def image_shas() -> dict[str, str | None]:
    """Each row's image file under ``RTX_SHARED_DIR`` (read at call time) to its sha256, ``None`` when absent."""
    out: dict[str, str | None] = {}
    for row in rtx_tasks.RTX:
        path = RTX_SHARED_DIR / row.image
        out[row.image] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    # 1. loopback, unresolved
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: --base-url is not loopback: {exc}")
        return EXIT_REFUSED
    # 2. a clean tree
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    # 3. no out-dir yet (one run per SHA)
    out: Path = args.out_dir or REPO / "logs" / f"t0209-rtx-{sha[:7]}"
    if out.exists():
        _say(
            f"refusing: {out} exists; the run happens once per SHA (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. group A1 merged (the rtx pages, the loopback index, the judge and the image); A2 decides R2
    if not _merged(MERGE_A1):
        _say(
            f"refusing: no '{MERGE_A1}' merge is reachable from HEAD (T-0209-a1, the rtx pages, loopback search, "
            "judge and shared image; a branch merge does not count)"
        )
        return EXIT_REFUSED
    r2 = _merged(MERGE_A2)
    if args.only == "R2" and not r2:
        _say(
            f"refusing: --only R2 needs a '{MERGE_A2}' merge reachable from HEAD (T-0209-a2, R2's pages and judge)"
        )
        return EXIT_REFUSED
    ids = tuple(i for i in run_ids(r2) if args.only is None or i == args.only)
    # 5. the table's pin, the index's pin and each image's sha (read at call time)
    table_sha = rtx_tasks.rtx_table_sha()
    if table_sha != RTX_TABLE_SHA256:
        _say(f"refusing: the rtx table's sha is {table_sha}, not the pinned {RTX_TABLE_SHA256}")
        return EXIT_REFUSED
    index_sha = rtx_search.search_index_sha()
    if index_sha != SEARCH_INDEX_SHA256:
        _say(
            f"refusing: the loopback search index's sha is {index_sha}, not the pinned {SEARCH_INDEX_SHA256}"
        )
        return EXIT_REFUSED
    shas = image_shas()
    for row in rtx_tasks.RTX:
        got = shas.get(row.image)
        if got != row.image_sha256:
            _say(
                f"refusing: {RTX_SHARED_DIR / row.image} has sha256 {got}, not its row's {row.image_sha256}"
            )
            return EXIT_REFUSED
    # 6. HEAD's live default, read from cli now (B1/B2)
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    ns = D._live_args()
    if not cli.web_of(lean, ns, "live"):
        _say(
            "refusing: the live default is not web mode (cli.web_of is off); B runs web mode on loopback"
        )
        return EXIT_REFUSED
    guidance = cli.live_guidance_of(lean, ns, "live").value
    if guidance != REQUIRED_GUIDANCE:
        _say(
            f"refusing: the live guidance resolves to {guidance!r}, not {REQUIRED_GUIDANCE} (cli.live_guidance_of)"
        )
        return EXIT_REFUSED
    steps_cap = cli.tool_loop_of(lean, ns).max_tool_steps
    if steps_cap != REQUIRED_LOOP:
        _say(
            f"refusing: the live loop's max_tool_steps is {steps_cap!r}, not {REQUIRED_LOOP} (cli.tool_loop_of)"
        )
        return EXIT_REFUSED
    feedback = cli.feedback_of(lean, ns, "live")
    if str(feedback.live_feedback) not in D.FEEDBACK_CHOICES:
        _say(f"refusing: cli.feedback_of gives live_feedback {feedback.live_feedback!r}")
        return EXIT_REFUSED
    default = live_default(lean)
    keywords = session_keywords(lean)
    # 7. the tool kit cli resolves (repair on), its search replaced by the loopback index
    kit_options: dict[str, bool] = {}
    if default["page_overview"]:
        kit_options["page_overview"] = True
    if default["wait_tool"]:
        kit_options["wait_tool"] = True
    try:
        kit, detokenizer = cli.load_tool_kit(lean, **kit_options)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    kit_sha = cli_kit_sha256(bool(default["page_overview"]), bool(default["wait_tool"]))
    if str(kit.tools_version) != kit_sha:
        _say(f"refusing: the tool set is {kit.tools_version}, not cli's {kit_sha}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(f"refusing: the tool repair is off ({kit.repair_off}); the regime says repair on")
        return EXIT_REFUSED
    refused = kit_refusal(rtx_kit(kit))
    if refused is not None:
        _say(f"refusing: {refused}")
        return EXIT_REFUSED
    # 8. Chromium (reads the filesystem only)
    absent = cli.live_prerequisites(BROWSER)
    if absent is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {absent}")
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = G._get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
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
    del detokenizer

    vram = T._VramSampler()
    vram.start()
    try:
        return _measure(
            args.base_url, out, sha, model, lean, vram, default, keywords, shas, r2, ids
        )
    finally:
        vram.stop()


def _measure(
    base_url: str,
    out: Path,
    sha: str,
    model: str,
    lean: Any,
    vram: Any,
    default: Mapping[str, Any],
    keywords: Mapping[str, Any],
    shas: Mapping[str, str | None],
    r2: bool,
    ids: tuple[str, ...],
) -> int:
    """Past every refusal and the engine check: the order, the manifest, P1-P3, P-R1, P-R2, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    kit_sha = cli_kit_sha256(bool(default["page_overview"]), bool(default["wait_tool"]))
    turns = [t for t in order(ORDER_SEED, r2) if t[0] in ids]
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in turns])
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0209_rtx_run.sh refuses while 127.0.0.1:8080/8081 are bound",
        "commit": sha,
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "base_url": base_url,
        "served_model": model,
        "engine_version": G._get_text(G._root(base_url) + "/version"),
        "profile": lean.name,
        "canvas_length": lean.diffusion.canvas_length,
        "max_denoising_steps": lean.diffusion.max_denoising_steps,
        "thinking": lean.diffusion.thinking,
        "tools": f"native:{kit_sha[:12]}:live:loop{default['max_tool_steps']}",
        "repair": "on",
        "arms": list(ARMS),
        "repeats": REPEATS,
        "browser": f"{BROWSER}, headless, one per turn, fresh context (no profile directory), web mode on one "
        "loopback origin per turn at /web.html, deny-all resolver and connector",
        "search": "google_search through a fresh rtx_search.LoopbackSearch per turn (web_search label ddgs)",
        "rtx_dir": str(RTX_DIR),
        "rtx_shared_dir": str(RTX_SHARED_DIR),
        "viewport": list(lean.browser_viewport),
        "turns": len(turns),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "input": "typed (UserText); fake STT/TTS/audio; shared arm: the picture posted once as an astra.frame through "
        "HudServer.route into the session's BrowserFrameSource, then select shared; typed arm: no frame, browser "
        "source; no audio device, no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "live_default": dict(default),
        "session_keywords": {k: D._jsonable(v) for k, v in keywords.items()},
        "search": True,
        "search_backend": SEARCH_BACKEND,
        "search_label_in_session_logs": SEARCH_LABEL,
        "shared_check": _merged(MERGE_0208_A2),
        "r2": r2,
        "ids": list(ids),
        "only": ids[0] if len(ids) == 1 and len(run_ids(r2)) > 1 else None,
        "executor_constants": {k: D._jsonable(v) for k, v in executor_constants().items()},
        "rtx_table_sha256": rtx_tasks.rtx_table_sha(),
        "search_index_sha256": rtx_search.search_index_sha(),
        "image_sha256": dict(shas),
        "scenarios": [
            {
                "rtx_id": r.rtx_id,
                "start_page": r.start_page,
                "image": r.image,
                "request": r.request,
                "control_request": r.control_request,
                "n_per_arm": REPEATS,
            }
            for r in rtx_tasks.RTX
            if r.rtx_id in ids
        ],
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {rel: G._sha256(REPO / rel) for rel in GATE_FILES},
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "thoughts.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- B4: P1-P3 (T-0131's), P-R1 and P-R2, before any turn ---------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name].get("ok"))}
    try:
        pr1 = dict(probe_pr1(r2))
    except Exception as exc:
        pr1 = {"error": repr(exc)}
    probes["pr1"] = {**pr1, "passed": pr1_ok(pr1, r2)}
    try:
        pr2 = dict(probe_pr2(pr1))
    except Exception as exc:
        pr2 = {"error": repr(exc)}
    probes["pr2"] = {**pr2, "passed": pr2_ok(pr2)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if probe_verdicts(probes, (), r2)["void"]:
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest, stopped, r2)
        _say(
            f"a probe failed: the run is void ({json.dumps(report['probes'])}); label {report['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = _Run(
        base_url=base_url,
        out=out,
        profile=lean,
        sentinel=sentinel,
        manifest=manifest,
        vram=vram,
        r2=r2,
        ids=ids,
    )
    try:
        asyncio.run(_run_all(run, turns))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    if run.stopped is not None:
        manifest["stopped"] = run.stopped
        G._write_json(out / "manifest.json", manifest)

    report = _write_report(out, probes, manifest, run.stopped, r2)
    _say(f"label={report['label']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
