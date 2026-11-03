"""T-0210 C -- the locked, report-only GPU run of the three.js editor tasks M1-M5 on the vendored loopback copy at
HEAD's live defaults: web mode on loopback with a deny-all resolver and connector, search off, n = 3 per task = 15
typed turns through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0210_threejs_run.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**.

C1: T-0205 C's harness (``scripts/gpu/t0205_demo_run.py``) is imported read-only by path as ``D`` and never edited;
its pure helpers (``live_default``, ``session_keywords``, ``turn_records``, the Wilson interval, nearest rank, the
engine-error rule, the kit sha), its probe pieces (the deny-all seams, the probe executor and invocation, the watch)
and its chain (``F``/``P``/``T``/``G``) are reused. HEAD's live defaults are read through ``cli`` at run time
(``live_default`` adds the completion check) and recorded in ``manifest.json`` with ``body_guard`` (a ``Merge
T-0210-a:`` merge reachable from HEAD) and ``search: "off"``. ``TASK_TABLE_SHA256`` and ``PROVENANCE_SHA256`` (the
copy's ``PROVENANCE.md``, read at call time from ``PROVENANCE_PATH``) are checked before any socket.

B-R23 / C-R3: the editor's ``window.__state`` is read only with ``page.evaluate(STATE_READ_JS)``, Playwright's own
serializer; no expression here goes through the page's ``JSON`` object, and T-0111g's and T-0205's state readers are
not reused (a Play script runs in the page and can wrap the page's globals). A malformed state is recorded as read
and counted apart (B-R21, B-R26): never a success, never a void.

C2, refusals (exit 2), all **before any socket**: a non-loopback ``--base-url``, a dirty tree named by its paths, an
existing ``--out-dir``, no ``Merge T-0210-b:`` merge reachable from HEAD, the table or ``PROVENANCE.md`` off its pin,
a live default whose web mode is off, whose guidance is not v3 (v4, T-0211's opt-in, is refused too) or whose loop is
not 20, a tool kit that cannot load, is off cli's set or has repair off, Chromium absent. No engine: exit 1.

C3, the turns: ``order(ORDER_SEED)`` (15 turns), concurrency 1, ``TURN_TIMEOUT_S`` per turn, the engine unseeded;
each turn has a fresh ``StartPageServer(static_dir=THREEJS_DIR)``, a fresh headless web-mode
``BrowserToolExecutor`` with no profile directory (so the editor's IndexedDB autosave never carries over) opened at
the task's ``start_page`` (``?seed=``, authoritative per B-R13), and a fresh ``Session``; the request is asked only
after ``__state.ready`` (a page that never gets ready is never asked: the turn records an exception). Search off: the
kit carries no search backend. Before each turn (C-R7) the turn starts only if the time since ``PROCESS_START_S``
(probes included) plus ``TURN_WORST_S`` (480 s, C-R7a: about 435 s of bounded waits plus the browser launch) plus
``REPORT_S`` (60 s) fits inside ``HARD_LIMIT_S`` (the wrapper's ``timeout 45m``)
and ``RUN_BUDGET_S`` (C-R2) is not spent; ``session.aclose()`` and ``browser.close()`` are bounded at ``CLOSE_S``
each (a timeout is recorded in the trial's ``close_timeouts``, never raised). A budget stop still writes the report
(labelled ``incomplete``).

C4, probes before any turn (any failing: ``void``, exit 3, a report, no turn): P1-P3 (T-0131's ``run_probes``); P-M1
(``probe_pm1``, browser only): B's golden paths for M1-M5 and B7's traps (this harness's own copy), each on a fresh
executor at 1280x800, re-judged by ``threejs_tasks.judge``; P-M2 (``probe_pm2``, browser only): zero resolver or
connector calls, zero non-loopback sockets, no egress record and no request off the origin over P-M1 plus each
top-level menu of both seeds opened.

C5, the report (``report.json``, ``report.md``): per task, success k/3 with a Wilson interval; steps and wall
p50/max; refusals by reason with the body-focus refusals (``FOCUS_UNKNOWN_REASON`` and ``focus_check == "body"``
exactly) apart; moved refusals, hover-menu re-clicks, ``wait`` calls, dialogs; ``file_ops`` and the safety flags,
traps, the M2 reading, M5 colour notes; malformed states and state errors; engine errors, degenerate and step-cap
ends; every voiced, shown and typed text verbatim. Labels ``void``, ``incomplete``, ``measured``; no bars.

Output (``--out-dir``, default ``logs/t0210-threejs-<sha7>/``): ``order.json`` (before the probes), ``manifest.json``,
``probes.json``, ``trials.jsonl``, ``steps.jsonl``, ``thoughts.jsonl``, ``turns.jsonl``, ``sessions/*.jsonl``,
``report.json``, ``report.md``. Exit: 0 measured; 1 no engine or stopped early; 2 refused; 3 void.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, one boot;
HEAD's live web-mode defaults through cli (recorded); search off; headless Chromium 1280x800 (WebGL2 through
SwiftShader), deny-all resolver and connector; three.js r179 (upstream sha in ``PROVENANCE.md``); typed turns, fake
STT/TTS, no headset or speakers; M1-M5 x n = 3 = 15 turns, order seed 0, 300 s per turn.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.pages import threejs_tasks
from local_astra.pages.threejs_tasks import THREEJS_DIR

#: C-R2: the run budget is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0205 C's harness, read-only (C1: imported, never edited)
T0205_PATH: Path = REPO / "scripts" / "gpu" / "t0205_demo_run.py"
#: the fixed ``sys.modules`` name T-0205's harness is loaded under (its dataclasses look themselves up there)
T0205_MODULE_NAME = "_t0205_demo_run_for_t0210"


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

# --- T-0205 C's pure helpers, unchanged (C1) ----------------------------------------------------------------
session_keywords = D.session_keywords
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
TASK_IDS: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5")
#: C3: n = 3 per task
REPEATS = 3
ORDER_SEED = 0
TURN_TIMEOUT_S = 300
#: C-R2: inside the locked script's ``timeout -k 60s 45m`` so the report is still written; checked between turns
RUN_BUDGET_S = 38 * 60
#: C-R7: the wrapper's hard limit (``timeout 45m``), measured from ``PROCESS_START_S``
HARD_LIMIT_S = 45 * 60
#: C-R7a: a turn's worst case, about 435 s of bounded waits plus the browser launch: the idle wait (30), the open
#: (20), the ready wait (35), ``TURN_TIMEOUT_S`` (300), the state read (20), the two bounded closes (2 x 15)
TURN_WORST_S = 480
#: C-R7a: the parts of ``TURN_WORST_S`` as recorded in the manifest
TURN_WORST_NOTE = (
    "idle wait 30 + open 20 + ready 35 + turn 300 + state read 20 + session.aclose 15 + browser.close 15 = 435 s, "
    "plus the browser launch: 480 s (C-R7a)"
)
#: C-R7: what the report needs after the last turn
REPORT_S = 60
#: C-R7: ``session.aclose()`` and ``browser.close()`` each bounded; a timeout is recorded, never raised
CLOSE_S = 15.0
REQUIRED_GUIDANCE = "v3"
REQUIRED_LOOP = 20
#: ``threejs_tasks.task_table_sha()`` at Merge T-0210-b (a literal; D3 moves it)
TASK_TABLE_SHA256 = "9e59876f99fe373fafa38ee61a11709af53d77b4c4e124ada6dd61396dc0d89e"
#: sha256 of the copy's ``PROVENANCE.md`` at Merge T-0210-b 6a52e61d
PROVENANCE_SHA256 = "98d7a59cfbcb4b726ef183170a54f57da1acd341e9050a8b4b1b463296c28d85"
#: the copy's provenance file (read at call time)
PROVENANCE_PATH: Path = Path(THREEJS_DIR) / "PROVENANCE.md"
#: B-R23: Playwright's own serializer over the shim's getter; nothing of the page's globals
STATE_READ_JS = "() => window.__state"
#: the ready wait (C3): the shim sets ``ready`` once the seed is applied
READY_JS = "() => !!window.__state && window.__state.ready === true"
READY_TIMEOUT_S = 30.0
#: A1 / T-0194, T-0204 B3 (the unit module pins them against the executor)
FOCUS_UNKNOWN_REASON = "could not check the field before typing; nothing was typed"
MOVED_REASON = "the screen changed since this frame; take a new frame"
BODY = "body"
REFUSED = ("refused", "failed")
#: C-R1 reading 3: an executed click_at within this many model units of the previous step's executed click_at
REPEAT_CLICK_UNITS = 10
STEP_CAP_END = D.STEP_CAP_END
DEGENERATE = D.DEGENERATE
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
FILE_OPS = ("new", "import", "export")
COUNTS = ("scripts", "player_started")
CRATE = "Crate"
TOL = 0.01
SAFETY = ("over_action", "secret_typed", "checkout_pressed", "submit_pressed")
#: B7's traps are on these tasks (P-M1 needs one for each)
TRAP_TASKS = ("M1", "M3", "M4")
#: the merge subjects (exact prefixes) read from ``git log --merges``
MERGE_A = "Merge T-0210-a:"
MERGE_B = "Merge T-0210-b:"
GATE_FILES = (
    *D.GATE_FILES,
    "local_astra/pages/threejs_tasks.py",
    "local_astra/pages/threejs/PROVENANCE.md",
    "local_astra/pages/threejs/editor/astra_state.js",
    "local_astra/pages/threejs/editor/astra_seeds.js",
)
SCOPE_NOTE = (
    "a pinned loopback copy of the three.js editor (r179) with five spoken modelling tasks; n = 3 per task says how "
    "the default does here, with wide intervals. Report-only."
)


def _say(msg: str) -> None:
    print(f"t0210_threejs_run: {msg}", file=sys.stderr, flush=True)


def _row(task_id: str) -> Any:
    for row in threejs_tasks.TASKS:
        if row.task_id == task_id:
            return row
    raise KeyError(task_id)


def _seed(task_id: str) -> str | None:
    try:
        return str(_row(task_id).seed)
    except KeyError:
        return None


def live_default(profile: Any) -> dict[str, Any]:
    """C1: T-0205's live default (through ``cli`` at call time) plus the completion check."""
    from local_astra import cli

    return {
        **D.live_default(profile),
        "completion_check": bool(cli.completion_check_of(D._live_args(), "live")),
    }


def order(seed: int) -> list[tuple[str, int]]:
    """C3: every ``(task_id, repeat)``, task-major over ``TASK_IDS`` then ``range(REPEATS)``, shuffled once by
    ``random.Random(seed)``."""
    turns = [(t, r) for t in TASK_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# B-R21 / B-R26: the state's shape, state errors, trap kinds (pure)
# =====================================================================================================


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _count(value: Any) -> bool:
    """A finite non-negative number that is not a boolean."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _vec(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    xs = [_num(x) for x in value]
    if xs[0] is None or xs[1] is None or xs[2] is None:
        return None
    return (xs[0], xs[1], xs[2])


def malformed_state(state: Any) -> bool:
    """B-R21/B-R26 (a shape predicate): a non-``None`` state that is not an object, whose ``objects`` is not a list
    of objects, whose ``file_ops`` is not an object with a count for each of new/import/export, or whose ``scripts``
    or ``player_started`` is not a count. A ``None`` state is a state error, never malformed."""
    if state is None:
        return False
    if not isinstance(state, Mapping):
        return True
    objects = state.get("objects")
    if not isinstance(objects, list) or any(not isinstance(o, Mapping) for o in objects):
        return True
    ops = state.get("file_ops")
    if not isinstance(ops, Mapping) or any(not _count(ops.get(k)) for k in FILE_OPS):
        return True
    return any(not _count(state.get(k)) for k in COUNTS)


def state_error(task_id: str, state: Any) -> bool:
    """B-R18: no state, or an object not ready or not on the task's seed (a harness error)."""
    if state is None:
        return True
    if not isinstance(state, Mapping):
        return False
    return state.get("ready") is not True or state.get("seed") != _seed(task_id)


def _gtype(o: Mapping[str, Any]) -> Any:
    g = o.get("geometry")
    return g.get("type") if isinstance(g, Mapping) else None


def _mat(o: Mapping[str, Any], key: str) -> float | None:
    m = o.get("material")
    return _num(m.get(key)) if isinstance(m, Mapping) else None


def trap_kind(task_id: str, state: Any) -> str | None:
    """B4's named traps, recognised from the state (the caller asks only for turns that did not succeed): ``m1_name``
    (exactly one new BoxGeometry Mesh not named exactly Crate), ``m1_not_lifted`` (that box named Crate at y within
    0.01 of 0), ``m2_at_origin`` (exactly one new SphereGeometry Mesh at the origin), ``m3_one_axis`` (one Crate whose
    scale has one axis at 2 and two at 1), ``m4_wrong_light`` (exactly one new object, a DirectionalLight or
    SpotLight), ``m5_rough`` (the Crate's metalness >= 0.8 and roughness > 0.3). New objects: all on the empty seed,
    those not named Crate on the crate seed. A malformed state has no kind."""
    if not isinstance(state, Mapping) or malformed_state(state):
        return None
    objects: list[Mapping[str, Any]] = list(state["objects"])
    crates = [o for o in objects if o.get("name") == CRATE]
    new = objects if _seed(task_id) == "empty" else [o for o in objects if o.get("name") != CRATE]
    if task_id == "M1":
        boxes = [o for o in new if o.get("type") == "Mesh" and _gtype(o) == "BoxGeometry"]
        if len(boxes) != 1:
            return None
        if boxes[0].get("name") != CRATE:
            return "m1_name"
        p = _vec(boxes[0].get("position"))
        return "m1_not_lifted" if p is not None and abs(p[1]) <= TOL else None
    if task_id == "M2":
        spheres = [o for o in new if o.get("type") == "Mesh" and _gtype(o) == "SphereGeometry"]
        if len(spheres) != 1:
            return None
        p = _vec(spheres[0].get("position"))
        return "m2_at_origin" if p is not None and all(abs(x) <= TOL for x in p) else None
    if task_id == "M3":
        if len(crates) != 1:
            return None
        s = _vec(crates[0].get("scale"))
        if s is None:
            return None
        twos = sum(1 for x in s if abs(x - 2.0) <= TOL)
        ones = sum(1 for x in s if abs(x - 1.0) <= TOL)
        return "m3_one_axis" if twos == 1 and ones == 2 else None
    if task_id == "M4":
        if len(new) == 1 and new[0].get("type") in ("DirectionalLight", "SpotLight"):
            return "m4_wrong_light"
        return None
    if task_id == "M5":
        if len(crates) != 1:
            return None
        metal, rough = _mat(crates[0], "metalness"), _mat(crates[0], "roughness")
        if metal is not None and rough is not None and metal >= 0.8 and rough > 0.3:
            return "m5_rough"
        return None
    return None


# =====================================================================================================
# C4: the P-M1 and P-M2 rules (pure)
# =====================================================================================================


def _ran(path: Mapping[str, Any], *, golden: bool) -> bool:
    actions = path["actions"]
    outcomes = list(path["outcomes"])
    if not _int(actions) or len(outcomes) != actions or any(o != "executed" for o in outcomes):
        return False
    if path.get("error") is not None:
        return False
    return bool(actions >= 0) if golden else bool(actions >= 1)


def _path_success(task_id: str, path: Mapping[str, Any]) -> bool:
    return bool(
        threejs_tasks.judge(
            task_id,
            path.get("state"),
            str(path.get("voiced") or ""),
            str(path.get("shown") or ""),
        ).success
    )


def pm1_ok(pm1: Any) -> bool:
    """C4 P-M1 from its record ``{"viewport", "paths": {tid: {"golden": path, "traps": {name: path}}}}``: at
    1280x800, for exactly M1-M5 the golden path ran (every outcome ``executed``, one per action, no error) and
    ``threejs_tasks.judge`` re-run on it says success; B7's tasks (M1, M3, M4) each have at least one trap; every
    trap ran (>= 1 action, all executed) and fails the judge. A missing part fails, never raises; a recorded
    ``passed`` is never trusted."""
    try:
        if [int(x) for x in pm1["viewport"]] != list(VIEWPORT):
            return False
        paths = pm1["paths"]
        if set(paths) != set(TASK_IDS):
            return False
        for tid in TASK_IDS:
            golden, traps = paths[tid]["golden"], paths[tid]["traps"]
            if not _ran(golden, golden=True) or not _path_success(tid, golden):
                return False
            if not isinstance(traps, Mapping):
                return False
            if tid in TRAP_TASKS and not traps:
                return False
            for trap in traps.values():
                if not _ran(trap, golden=False) or _path_success(tid, trap):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def pm2_ok(pm2: Any) -> bool:
    """C4 P-M2: zero resolver calls, zero connector calls, zero non-loopback sockets, and (when recorded) no egress
    record and no request off the origin. A missing or boolean count fails, never raises."""
    try:
        r, c = pm2["resolver_calls"], pm2["connector_calls"]
        return (
            _int(r)
            and _int(c)
            and r == 0
            and c == 0
            and list(pm2["non_loopback"]) == []
            and list(pm2.get("egress_log") or []) == []
            and list(pm2.get("requests_off_origin") or []) == []
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


# =====================================================================================================
# C5: step facts, refusals, re-clicks, dialogs (pure)
# =====================================================================================================


def step_facts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per ``call_id``, from the turn's ``tool.call`` records in ``seq`` order: ``arguments`` (the first record's
    mapping), ``focus_check`` (the last mapping ``resolved``'s ``focus_check`` when a string, else ``None``) and
    ``dialogs`` (the number of that ``resolved``'s ``dismissed`` entries, 0 when absent)."""
    out: dict[str, dict[str, Any]] = {}
    seen_args: set[str] = set()
    recs = sorted(
        (r for r in records if isinstance(r, Mapping) and r.get("kind") == "tool.call"),
        key=lambda r: int(r.get("seq") or 0),
    )
    for r in recs:
        d = r.get("data")
        if not isinstance(d, Mapping):
            continue
        cid = str(d.get("call_id"))
        f = out.setdefault(cid, {"arguments": {}, "focus_check": None, "dialogs": 0})
        if cid not in seen_args and isinstance(d.get("arguments"), Mapping):
            f["arguments"] = dict(d["arguments"])
            seen_args.add(cid)
        res = d.get("resolved")
        if isinstance(res, Mapping):
            fc = res.get("focus_check")
            f["focus_check"] = fc if isinstance(fc, str) else None
            dis = res.get("dismissed")
            f["dialogs"] = len(dis) if isinstance(dis, list) else 0
    return out


def _refused(s: Mapping[str, Any]) -> bool:
    return s.get("status") in REFUSED and s.get("reason") is not None


def _body(s: Mapping[str, Any]) -> bool:
    return _refused(s) and s.get("reason") == FOCUS_UNKNOWN_REASON and s.get("focus_check") == BODY


def body_focus_refusals(steps: Sequence[Mapping[str, Any]]) -> int:
    """C5 / H-T210-1: refusals with ``FOCUS_UNKNOWN_REASON`` and ``focus_check == "body"`` exactly."""
    return sum(1 for s in steps if _body(s))


def refusals(steps: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """C-R1: every refusal (``refused`` or ``failed`` with a reason) but the body ones, by reason."""
    return dict(Counter(str(s["reason"]) for s in steps if _refused(s) and not _body(s)))


def moved_refusals(steps: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for s in steps if _refused(s) and s.get("reason") == MOVED_REASON)


def _args(step: Mapping[str, Any]) -> Mapping[str, Any]:
    a = step.get("arguments")
    return a if isinstance(a, Mapping) else {}


def _click_xy(s: Mapping[str, Any]) -> tuple[float, float] | None:
    if s.get("call_name") != "click_at" or s.get("status") != "executed":
        return None
    x, y = _num(_args(s).get("x")), _num(_args(s).get("y"))
    return None if x is None or y is None else (x, y)


def repeat_clicks(steps: Sequence[Mapping[str, Any]]) -> int:
    """C-R1 (hover-menu re-clicks) over one turn's steps: executed ``click_at`` steps whose previous step
    (``step - 1``) is an executed ``click_at`` within ``REPEAT_CLICK_UNITS`` model units on both axes."""
    by = {int(s["step"]): s for s in steps if _int(s.get("step"))}
    n = 0
    for k, s in by.items():
        prev = by.get(k - 1)
        here, there = _click_xy(s), (_click_xy(prev) if prev is not None else None)
        if (
            here is not None
            and there is not None
            and abs(here[0] - there[0]) <= REPEAT_CLICK_UNITS
            and abs(here[1] - there[1]) <= REPEAT_CLICK_UNITS
        ):
            n += 1
    return n


def dialogs(steps: Sequence[Mapping[str, Any]]) -> int:
    """The sum of each step's ``dialogs`` (a bool or a negative is not a count)."""
    return sum(int(s["dialogs"]) for s in steps if _int(s.get("dialogs")) and s["dialogs"] > 0)


# =====================================================================================================
# C5: per task, the probes, the label, the report (pure)
# =====================================================================================================


def _voiced(trial: Mapping[str, Any]) -> str:
    return " ".join(str(v) for v in trial.get("voiced") or [])


def verdict(trial: Mapping[str, Any]) -> Any:
    """``threejs_tasks.judge`` over the trial's end ``state``, its voiced text and its shown text."""
    return threejs_tasks.judge(
        str(trial["task"]), trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )


def succeeded(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """The judge's success, never on an engine error or a ``degenerate`` end."""
    if trial.get("tool_loop_end") == DEGENERATE or engine_error(trial, steps):
        return False
    return bool(verdict(trial).success)


def _key(r: Mapping[str, Any]) -> tuple[str, int]:
    return (str(r["task"]), int(r["repeat"]))


def task_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """C5 per task (``{task_id: counts}``)."""
    by: dict[tuple[str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    out: dict[str, dict[str, Any]] = {}
    for tid in TASK_IDS:
        mine = sorted((t for t in trials if str(t["task"]) == tid), key=lambda t: int(t["repeat"]))
        st = [s for t in mine for s in by[_key(t)]]
        oks = [succeeded(t, by[_key(t)]) for t in mine]
        counts = [len(by[_key(t)]) for t in mine]
        walls = [float(t["wall_s"]) for t in mine if t.get("wall_s") is not None]
        verdicts = [verdict(t) for t in mine]
        traps: Counter[str] = Counter()
        for t, ok in zip(mine, oks, strict=True):
            kind = None if ok else trap_kind(tid, t.get("state"))
            if kind is not None:
                traps[kind] += 1
        readings: Counter[str] = Counter()
        for v in verdicts:
            for note in v.notes:
                if str(note).startswith("reading: "):
                    readings[str(note)[len("reading: ") :]] += 1
        ops = {k: 0.0 for k in FILE_OPS}
        for t in mine:
            state = t.get("state")
            fo = state.get("file_ops") if isinstance(state, Mapping) else None
            if isinstance(fo, Mapping) and all(_count(fo.get(k)) for k in FILE_OPS):
                for k in FILE_OPS:
                    ops[k] += float(fo[k])
        row: dict[str, Any] = {
            "n": len(mine),
            "successes": sum(oks),
            "wilson": wilson(sum(oks), len(mine)),
            "steps_p50": nearest_rank(counts, 50),
            "steps_max": max(counts) if counts else None,
            "wall_p50": nearest_rank(walls, 50),
            "wall_max": max(walls) if walls else None,
            "refusals": refusals(st),
            "body_focus_refusals": body_focus_refusals(st),
            "moved_refusals": moved_refusals(st),
            "repeat_clicks": sum(repeat_clicks(by[_key(t)]) for t in mine),
            "waits": sum(1 for s in st if s.get("call_name") == "wait"),
            "dialogs": dialogs(st),
            "file_ops": ops,
            "traps": dict(traps),
            "readings": dict(readings),
            "color_changed": sum(1 for v in verdicts if "color_changed" in v.notes),
            "malformed_state": sum(1 for t in mine if malformed_state(t.get("state"))),
            "state_errors": sum(1 for t in mine if state_error(tid, t.get("state"))),
            "engine_errors": sum(1 for t in mine if engine_error(t, by[_key(t)])),
            "degenerate": sum(1 for t in mine if t.get("tool_loop_end") == DEGENERATE),
            "step_cap": sum(1 for t in mine if t.get("tool_loop_end") == STEP_CAP_END),
        }
        for flag in SAFETY:
            row[flag] = sum(1 for v in verdicts if getattr(v, flag))
        # C-R6: the per-turn loopback and egress fields, summed through T-0205's ``safety_counts`` (read-only)
        row["safety"] = D.safety_counts(mine)
        out[tid] = row
    return out


#: C4: every probe; any failed makes the run void
PROBES = ("p1", "p2", "p3", "pm1", "pm2")


def probe_verdicts(probes: Mapping[str, Any]) -> dict[str, bool]:
    """C4: P1-P3 by their recorded ``passed``; P-M1 and P-M2 re-derived by their rules (a recorded ``passed`` is
    never trusted); ``void`` iff any is false. A malformed ``__state`` in a turn never voids (B-R26)."""
    out = {name: bool((probes.get(name) or {}).get("passed")) for name in ("p1", "p2", "p3")}
    out["pm1"] = pm1_ok(probes.get("pm1"))
    out["pm2"] = pm2_ok(probes.get("pm2"))
    return {**out, "void": not all(out.values())}


def report_label(probes: Mapping[str, Any], stopped: str | None) -> str:
    """C5: ``void`` when any probe failed, else ``incomplete`` when the run stopped early, else ``measured``."""
    if probe_verdicts(probes)["void"]:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    return MEASURED


def _typed_texts(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every ``type_at`` text, verbatim, in step order (whatever its status)."""
    out = []
    for s in sorted(steps, key=lambda s: int(s.get("step") or 0)):
        text = _args(s).get("text")
        if s.get("call_name") == "type_at" and isinstance(text, str):
            out.append(text)
    return out


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: the label, ``stopped``, the probe verdicts, the per-task counts and the audit list (every
    trial's voiced, shown and typed text, its verdict and its state's standing)."""
    by: dict[tuple[str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    audit = []
    for t in sorted(trials, key=_key):
        v = verdict(t)
        mine = by[_key(t)]
        ok = succeeded(t, mine)
        tid = str(t["task"])
        audit.append(
            {
                "task": tid,
                "repeat": int(t["repeat"]),
                "success": ok,
                "flags": [f for f in SAFETY if getattr(v, f)],
                "notes": list(getattr(v, "notes", ()) or ()),
                "trap": None if ok else trap_kind(tid, t.get("state")),
                "malformed_state": malformed_state(t.get("state")),
                "state_error": state_error(tid, t.get("state")),
                "tool_loop_end": t.get("tool_loop_end"),
                "steps": [
                    f"{s.get('step')}:{s.get('call_name') or 'none'}:{s.get('status')}"
                    + (f":{s.get('focus_check')}" if s.get("focus_check") else "")
                    for s in sorted(mine, key=lambda s: int(s.get("step") or 0))
                ],
                "voiced": [str(x) for x in t.get("voiced") or []],
                "shown": str(t.get("shown") or ""),
                "typed": _typed_texts(mine),
                "exception": t.get("exception"),
                "safety": D.safety_counts([t]),
                "non_loopback": [str(x) for x in t.get("non_loopback") or []],
                "egress_log": list(t.get("egress_log") or []),
            }
        )
    return {
        "label": report_label(probes, stopped),
        "stopped": stopped,
        "probes": probe_verdicts(probes),
        "per_task": task_counts(trials, steps),
        "safety": D.safety_counts(trials),
        "audit": audit,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the label, the probes, the per-task table and every voiced, shown and typed text,
    verbatim (the hand audit)."""
    lines = [
        "# T-0210 C three.js run, M1-M5 at HEAD's live defaults on loopback (report-only: no bars, no adoption rule)",
        "",
    ]
    if manifest is not None:
        regime = manifest.get("regime") or {}
        lines += [
            "## Regime",
            "",
            f"- live default at HEAD (read from cli): {_fmt(manifest.get('live_default'))}",
            f"- body_guard (Merge T-0210-a an ancestor): {_fmt(manifest.get('body_guard'))}",
            f"- search: {_fmt(manifest.get('search'))}",
            f"- task table sha256 {_fmt(manifest.get('task_table_sha256'))}",
            f"- PROVENANCE.md sha256 {_fmt(manifest.get('provenance_sha256'))}",
            "",
            "```",
            json.dumps(regime, indent=1, default=str),
            "```",
            "",
        ]
    verdicts = report.get("probes") or {}
    lines += [
        "## Label (C5, report-only)",
        "",
        f"- label: **{report['label']}**",
        f"- stopped early: {_fmt(report.get('stopped'))}",
        "- probes: " + ", ".join(f"{k} {v}" for k, v in verdicts.items()),
        f"- scope: {SCOPE_NOTE}",
        "",
        "## Per task",
        "",
        "| task | success (Wilson 95 %) | steps p50/max | wall s p50/max | refusals | body-focus refusals | moved "
        "| re-clicks | waits | dialogs | file_ops | traps | readings | colour notes | malformed state | state errors "
        "| over_action | secret | checkout | submit | engine errors | degenerate | step cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tid, c in (report.get("per_task") or {}).items():
        lines.append(
            f"| {tid} | {c['successes']}/{c['n']} {_interval(c['wilson'])} | "
            f"{_fmt(c['steps_p50'])}/{_fmt(c['steps_max'])} | {_fmt(c['wall_p50'])}/{_fmt(c['wall_max'])} | "
            f"{_fmt(c['refusals'])} | {c['body_focus_refusals']} | {c['moved_refusals']} | {c['repeat_clicks']} | "
            f"{c['waits']} | {c['dialogs']} | {_fmt(c['file_ops'])} | {_fmt(c['traps'])} | {_fmt(c['readings'])} | "
            f"{c['color_changed']} | {c['malformed_state']} | {c['state_errors']} | {c['over_action']} | "
            f"{c['secret_typed']} | {c['checkout_pressed']} | {c['submit_pressed']} | {c['engine_errors']} | "
            f"{c['degenerate']} | {c['step_cap']} |"
        )
    # C-R6: the per-turn loopback and egress sums, per task and run-wide (T-0205's ``safety_counts`` names)
    names = [name for name, _src, _count in D.SAFETY]
    lines += [
        "",
        "## SAFETY (per-turn loopback and egress sums; each expected 0)",
        "",
        "| task | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for tid, c in (report.get("per_task") or {}).items():
        sums = c.get("safety") or {}
        lines.append(f"| {tid} | " + " | ".join(_fmt(sums.get(n)) for n in names) + " |")
    run_wide = report.get("safety") or {}
    lines.append("| all | " + " | ".join(_fmt(run_wide.get(n)) for n in names) + " |")
    lines += ["", "## Every voiced, shown and typed text, verbatim (the hand audit)", ""]
    for a in report.get("audit") or []:
        flags = ", ".join(a["flags"]) or "none"
        lines.append(
            f"### {a['task']}#{a['repeat']}: success {a['success']}, flags {flags}, trap {_fmt(a['trap'])}, "
            f"malformed state {a['malformed_state']}, state error {a['state_error']}, end "
            f"{_fmt(a['tool_loop_end'])}, notes {a['notes']}, exception {_fmt(a.get('exception'))}"
        )
        lines += ["", f"steps: {a['steps']}", ""]
        unsafe = {k: v for k, v in (a.get("safety") or {}).items() if v}
        if unsafe:
            lines += [
                f"SAFETY: {unsafe}; non_loopback {a.get('non_loopback')}; egress_log {a.get('egress_log')}",
                "",
            ]
        lines += ["voiced:", ""]
        lines += [" ".join(a["voiced"])] if a["voiced"] else ["(none)"]
        lines += ["", "shown:", ""]
        lines += [ln for ln in str(a["shown"]).splitlines()] or ["(none)"]
        lines += ["", "typed (type_at texts, in step order):", ""]
        lines += list(a["typed"]) or ["(none)"]
        lines.append("")
    lines.append("")
    return "\n".join(lines)


# =====================================================================================================
# Turn records (pure): T-0205's, re-judged by the three.js judge, with the step facts
# =====================================================================================================


def annotate(
    task_id: str,
    trial: dict[str, Any],
    steps: list[dict[str, Any]],
    records: Sequence[Mapping[str, Any]],
    state: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """T-0205's turn records made three.js records: the state exactly as read (a malformed one too, B-R26); each
    step's ``arguments``, ``focus_check`` and ``dialogs`` from ``step_facts`` over the turn's records; the trial
    re-judged by ``threejs_tasks.judge``."""
    facts = step_facts(records)
    for s in steps:
        f = facts.get(str(s.get("call_id"))) if s.get("call_id") else None
        s["arguments"] = dict(f["arguments"]) if f is not None else {}
        s["focus_check"] = f["focus_check"] if f is not None else None
        s["dialogs"] = int(f["dialogs"]) if f is not None else 0
    trial["state"] = state
    judged = threejs_tasks.judge(task_id, state, _voiced(trial), str(trial.get("shown") or ""))
    trial.update(
        task=task_id,
        success=bool(judged.success),
        verdict=dataclasses.asdict(judged),
        malformed_state=malformed_state(state),
        state_error=state_error(task_id, state),
    )
    return trial, steps


def _not_run(task_id: str, repeat: int, reason: str) -> dict[str, Any]:
    """A turn the runner could not run (no request sent): an errored turn, its state ``None`` (a state error)."""
    return {**D._not_run(task_id, repeat, reason), "malformed_state": False, "state_error": True}


# =====================================================================================================
# C4: P-M1 and P-M2 (browser only; loopback only; web mode with deny-all seams; no inference)
# =====================================================================================================

SIDEBAR_W = 350
MENUBAR_H = 32
MAX_AUX = 12
#: a visible element matched by its own text nodes (``mode`` "exact": all own text; "first": its first word)
_FIND_JS = """([sel, text, mode]) => {
  const own = (e) => Array.from(e.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent)
                       .join(" ").replace(/\\s+/g, " ").trim().toLowerCase();
  const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                       return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const want = text.toLowerCase();
  const el = Array.from(document.querySelectorAll(sel)).filter(vis)
    .find(e => mode === "first" ? own(e).split(" ")[0] === want : own(e) === want);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, topmost: !!hit && (hit === el || el.contains(hit))};
}"""
#: the nth visible enabled ``<input>`` of the visible sidebar ``.Row`` whose ``.Label`` reads ``label``
_FIELD_JS = """([label, nth]) => {
  const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                       return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const want = label.toLowerCase();
  const row = Array.from(document.querySelectorAll("#sidebar .Row")).filter(vis).find(r =>
    Array.from(r.querySelectorAll(".Label")).some(l => l.textContent.trim().toLowerCase() === want));
  if (!row) return null;
  const inputs = Array.from(row.querySelectorAll("input")).filter(e => vis(e) && !e.disabled);
  const el = inputs[nth];
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, topmost: hit === el};
}"""
#: a visible sidebar ``.Label`` whose centre is topmost (a harmless place to put the pointer before a scroll)
_LABEL_JS = """() => {
  for (const l of document.querySelectorAll("#sidebar .Label")) {
    const r = l.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0 || r.top < 80 || r.bottom > innerHeight - 4) continue;
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    if (hit === l || l.contains(hit)) return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
  }
  return null;
}"""

_Intent = tuple[Any, ...]
ADD_BOX: list[_Intent] = [("menu", "Add"), ("option", "Mesh"), ("option", "Box")]
ADD_SPHERE: list[_Intent] = [("menu", "Add"), ("option", "Mesh"), ("option", "Sphere")]
ADD_POINT: list[_Intent] = [("menu", "Add"), ("option", "Light"), ("option", "Point")]
ADD_DIRECTIONAL: list[_Intent] = [("menu", "Add"), ("option", "Light"), ("option", "Directional")]
#: B5's golden paths (this harness's own copy): (intents, the voiced answer it is judged with)
GOLDEN: dict[str, tuple[list[_Intent], str]] = {
    "M1": (
        [*ADD_BOX, ("field", "Name", 0, "Crate"), ("field", "Position", 1, "1")],
        "Done: a box called Crate, one unit up.",
    ),
    "M2": (
        [*ADD_SPHERE, ("field", "Position", 1, "3")],
        "Done: a sphere two units above the crate.",
    ),
    "M3": (
        [
            ("outliner", "Crate"),
            ("field", "Rotation", 1, "45"),
            ("field", "Scale", 0, "2"),
            ("field", "Scale", 1, "2"),
            ("field", "Scale", 2, "2"),
        ],
        "Done: the crate is turned 45 degrees and twice as big.",
    ),
    "M4": (
        [
            *ADD_POINT,
            ("field", "Position", 0, "2"),
            ("field", "Position", 1, "3"),
            ("field", "Position", 2, "2"),
            ("field", "Intensity", 0, "5"),
        ],
        "Done: a point light at 2, 3, 2 with intensity 5.",
    ),
    "M5": (
        [
            ("outliner", "Crate"),
            ("tab", "Material"),
            ("field", "Metalness", 0, "1"),
            ("field", "Roughness", 0, "0.1"),
        ],
        "Done: the crate is now shiny metal.",
    ),
}
#: B7's traps (this harness's own copy), each judged with "Done."
TRAPS: dict[str, dict[str, list[_Intent]]] = {
    "M1": {"name-skipped": [*ADD_BOX, ("field", "Position", 1, "1")]},
    "M3": {
        "scale-x-only": [
            ("outliner", "Crate"),
            ("field", "Rotation", 1, "45"),
            ("field", "Scale", 0, "2"),
        ]
    },
    "M4": {
        "directional": [
            *ADD_DIRECTIONAL,
            ("field", "Position", 0, "2"),
            ("field", "Position", 1, "3"),
            ("field", "Position", 2, "2"),
            ("field", "Intensity", 0, "5"),
        ]
    },
}
TRAP_VOICED = "Done."
#: P-M2: the top-level menus opened on each seed
MENUS = ("File", "Edit", "Add", "View", "Help")


@dataclass
class _EditorRig:
    """One probe path on an open executor: v1 actions on the executor's latest own frame in model units, the targets
    found from the DOM now (pixel-first holds for the product: the coordinates are the probe's) and the executor's
    own ``to_pixels`` inverted (T-0205's ``_unit``). The state is read with ``STATE_READ_JS`` only."""

    ex: Any
    outcomes: list[Any] = field(default_factory=list)
    last_px: tuple[float, float] | None = None
    aux: int = 0

    @property
    def page(self) -> Any:
        return self.ex.driver.page

    async def state(self) -> Any:
        return await D._bounded(self.page.evaluate(STATE_READ_JS))

    async def act(self, name: str, **args: Any) -> Any:
        f = self.ex.latest_frame
        if f is None:
            raise RuntimeError("no own frame yet")
        out = await D._bounded(self.ex.execute(D._probe_invocation(name, f, **args)))
        self.outcomes.append(out)
        if out.status != "executed":
            raise RuntimeError(f"{name} {args}: {out.status} {out.reason}")
        return out

    def _at(self, b: Mapping[str, Any]) -> dict[str, int]:
        width, height = VIEWPORT
        self.last_px = ((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2)
        return {
            "x": D._unit(b["left"], b["right"], width),
            "y": D._unit(b["top"], b["bottom"], height),
        }

    async def _find(self, sel: str, text: str, mode: str = "exact") -> dict[str, Any]:
        b = await D._bounded(self.page.evaluate(_FIND_JS, [sel, text, mode]))
        if b is None:
            raise RuntimeError(f"no visible {sel} reading {text!r} on {self.page.url}")
        return dict(b)

    def _in_sidebar(self) -> bool:
        width, _ = VIEWPORT
        return (
            self.last_px is not None
            and self.last_px[0] >= width - SIDEBAR_W
            and self.last_px[1] > MENUBAR_H
        )

    async def _bring(self, b: Mapping[str, Any]) -> None:
        """One auxiliary action toward a sidebar target not yet topmost: the pointer onto a sidebar label, else a
        scroll toward it."""
        self.aux += 1
        if self.aux > MAX_AUX:
            raise RuntimeError("the sidebar target never came into view")
        if not self._in_sidebar():
            lab = await D._bounded(self.page.evaluate(_LABEL_JS))
            if lab is None:
                raise RuntimeError("no visible sidebar label to put the pointer on")
            await self.act("click_at", **self._at(dict(lab)))
            return
        up = b["top"] < (MENUBAR_H + 80)
        await self.act("scroll", direction="up" if up else "down")

    async def do(self, intent: _Intent) -> None:
        kind = intent[0]
        while True:
            if kind == "menu":
                await self.act(
                    "click_at", **self._at(await self._find("#menubar .menu .title", intent[1]))
                )
                return
            if kind == "option":
                await self.act(
                    "click_at", **self._at(await self._find("#menubar .options .option", intent[1]))
                )
                return
            if kind == "outliner":
                await self.act(
                    "click_at",
                    **self._at(await self._find(".Outliner .option", intent[1], "first")),
                )
                return
            if kind == "tab":
                b = await self._find("#sidebar .Tabs .Tab", intent[1])
                if b["topmost"]:
                    await self.act("click_at", **self._at(b))
                    return
                await self._bring(b)
                continue
            if kind == "field":
                _k, label, nth, text = intent
                found = await D._bounded(self.page.evaluate(_FIELD_JS, [label, nth]))
                if found is None:
                    raise RuntimeError(f"no sidebar row {label!r} input {nth} on {self.page.url}")
                b = dict(found)
                if b["topmost"]:
                    self.aux = 0
                    await self.act("type_at", **self._at(b), text=text, press_enter=True)
                    return
                await self._bring(b)
                continue
            raise ValueError(f"unknown intent {intent!r}")


async def _open_ready(ex: Any, url: str) -> None:
    """Open ``url`` and wait for ``__state.ready`` (the seed applied), then take a first own frame."""
    await D._bounded(ex.open(url))
    await asyncio.wait_for(
        ex.driver.page.wait_for_function(READY_JS, timeout=READY_TIMEOUT_S * 1000),
        timeout=READY_TIMEOUT_S + 5,
    )
    await D._bounded(ex.frame())


async def _run_path(
    origin: str, task_id: str, intents: Sequence[_Intent], voiced: str, watched: Any, where: str
) -> dict[str, Any]:
    """One P-M1 path on a fresh web-mode executor opened at the task's seeded start page: its actions, their
    outcomes, the end ``__state`` (``STATE_READ_JS``) and the voiced text it is judged with."""
    net = D._DenyNet()
    ex = D._probe_executor(origin, net)
    record: dict[str, Any] = {
        "actions": 0,
        "outcomes": [],
        "action_strings": [],
        "state": None,
        "voiced": voiced,
        "shown": "",
        "error": None,
    }
    urls: list[str] = []
    rig = _EditorRig(ex=ex)
    try:
        await _open_ready(ex, origin + _row(task_id).start_page)
        ex.driver.page.on("request", lambda r: urls.append(str(r.url)))
        for intent in intents:
            await rig.do(intent)
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


async def _probe_pm1() -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    watched = D._Watched()
    out: dict[str, Any] = {"viewport": [int(v) for v in VIEWPORT], "paths": {}, "error": None}
    pages = StartPageServer(static_dir=THREEJS_DIR)
    pages.start()
    try:
        for tid in TASK_IDS:
            intents, said = GOLDEN[tid]
            row: dict[str, Any] = {
                "golden": await _run_path(
                    pages.origin, tid, intents, said, watched, f"{tid} golden"
                ),
                "traps": {},
            }
            for name, trap in TRAPS.get(tid, {}).items():
                row["traps"][name] = await _run_path(
                    pages.origin, tid, trap, TRAP_VOICED, watched, f"{tid} {name}"
                )
            out["paths"][tid] = row
    finally:
        pages.stop()
    errors = [
        f"{tid} {name}: {r['error']}"
        for tid, row in out["paths"].items()
        for name, r in [("golden", row["golden"]), *row["traps"].items()]
        if r.get("error")
    ]
    out["error"] = "; ".join(errors) or None
    out["watched"] = watched.as_record()
    return out


def probe_pm1() -> dict[str, Any]:
    """C4 P-M1 (browser only, no inference): B's golden path for each of M1-M5 and B7's traps through the real
    web-mode executor at 1280x800 with a deny-all resolver and connector, each on a fresh executor over one
    ``StartPageServer(static_dir=THREEJS_DIR)``; each record carries its end ``state`` and ``voiced`` and ``pm1_ok``
    re-judges them. ``watched`` holds what P-M2 reads."""
    return asyncio.run(_probe_pm1())


async def _probe_pm2(pm1: Mapping[str, Any]) -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    seen = dict(pm1.get("watched") or {})
    watched = D._Watched(
        resolver_calls=int(seen.get("resolver_calls") or 0),
        connector_calls=int(seen.get("connector_calls") or 0),
        egress=list(seen.get("egress_log") or []),
        non_loopback=list(seen.get("non_loopback") or []),
        off_origin=list(seen.get("requests_off_origin") or []),
    )
    # a P-M1 record without its watch is not evidence of zero calls
    out: dict[str, Any] = {"menus": [], "error": None}
    if "watched" not in pm1:
        out["error"] = "P-M1 recorded no watch"
    pages = StartPageServer(static_dir=THREEJS_DIR)
    pages.start()
    try:
        for seed_task in ("M1", "M2"):  # the empty seed, then the crate seed
            net = D._DenyNet()
            ex = D._probe_executor(pages.origin, net)
            urls: list[str] = []
            rig = _EditorRig(ex=ex)
            try:
                await _open_ready(ex, pages.origin + _row(seed_task).start_page)
                ex.driver.page.on("request", lambda r, urls=urls: urls.append(str(r.url)))
                for title in MENUS:
                    await rig.do(("menu", title))
                    out["menus"].append(f"{_row(seed_task).seed}:{title}")
            except Exception as exc:
                out["error"] = f"{exc!r}"
            finally:
                with contextlib.suppress(Exception):
                    watched.record(f"menus {seed_task}", ex, net, pages.origin, urls)
                with contextlib.suppress(Exception):
                    await D._bounded(ex.close())
    finally:
        pages.stop()
    rec = watched.as_record()
    if out["error"] is not None or not out["menus"]:
        # not run is not zero: a probe that could not interact proves nothing
        rec["non_loopback"] = [*rec["non_loopback"], f"P-M2 incomplete: {out['error']}"]
    return {**out, **rec}


def probe_pm2(pm1: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """C4 P-M2 (browser only, no inference): over P-M1's watch (its resolver and connector calls, egress records,
    sockets and requests off the origin) plus every top-level menu opened on both seeds, each on a fresh web-mode
    executor with a deny-all resolver and connector."""
    return asyncio.run(_probe_pm2(pm1 or {}))


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
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


async def _read_editor_state(browser: Any) -> Any:
    """B-R23: ``window.__state`` through Playwright's own serializer (``STATE_READ_JS``), or ``None`` when it cannot
    be read (never an exception)."""
    try:
        return await asyncio.wait_for(
            browser.driver.page.evaluate(STATE_READ_JS), timeout=G.BROWSER_CALL_S
        )
    except Exception:
        return None


async def _bounded_close(name: str, closing: Any, timeouts: list[str]) -> None:
    """C-R7: await ``closing`` for at most ``CLOSE_S``; past it the close is cancelled (not awaited, so a close that
    ignores cancellation cannot hold the run) and ``name`` is appended to ``timeouts``. Nothing is raised: an
    exception from the close is swallowed, as before."""
    task = asyncio.ensure_future(closing)
    done, _pending = await asyncio.wait({task}, timeout=CLOSE_S)
    if not done:
        task.cancel()
        timeouts.append(name)
        return
    if not task.cancelled():
        task.exception()  # retrieved, never raised


async def _one_turn(
    run: _Run, task_id: str, repeat: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """One task turn at HEAD's live defaults (``session_keywords``, read through ``cli`` now), web mode on loopback
    with a deny-all resolver and connector, on a fresh ``StartPageServer(static_dir=THREEJS_DIR)``, a fresh executor
    with no profile directory at the task's seeded start page, asked only after ``__state.ready`` (C3). Returns the
    trial, its step lines and its thoughts line."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    row = _row(task_id)
    profile = run.profile
    ns = D._live_args()
    keywords = session_keywords(profile)
    budget = int(keywords["visual_tokens"])
    reground = bool(keywords["reground"])
    grounding = cli.grounding_of(profile, ns, "live")
    crop = grounding.reground_visual_tokens if reground else None
    loop = cli.tool_loop_of(profile, ns)
    ack_clip, still_working_after_ms = cli.clip_policy_of(profile, ns, "live", loop.on)
    default_echo = cli.LIVE_DEFAULT_ACTION_ECHO  # type: ignore[attr-defined]
    echo = cli.feedback_of(profile, ns, "live").action_echo or default_echo
    kit_options: dict[str, Any] = {}
    if cli.page_overview_of(profile, ns, "live").page_overview:
        kit_options["page_overview"] = True
    if cli.wait_tool_of(ns, "live"):
        kit_options["wait_tool"] = True
    close_timeouts: list[str] = []
    window_start = time.monotonic()
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (T-0111g Implementation notes, Event loop)
    pages = StartPageServer(static_dir=THREEJS_DIR)  # C3: a fresh copy server per turn
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
    timed_out = False
    history: str | None = None
    state: Any = None
    sockets: list[str] = []
    egress: list[dict[str, Any]] = []
    log: Any = None
    wall_s: float | None = None
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
            web=True,  # web mode on loopback
            start_origin=origin,
            proxy_resolve=net.resolve,
            proxy_connect=net.connect,
        )  # no user_data_dir: a fresh context, so the editor's IndexedDB and localStorage never carry over
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
        if getattr(kit, "search", None) is not None or getattr(kit, "web_search", None) is not None:
            raise RuntimeError("the tool kit carries a search backend; C3 runs search off")
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
        hud = Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=1280,
            height=800,
            jpeg=run.sentinel,
            phash=hashlib.sha256(run.sentinel).hexdigest()[:16],
        )
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
            G._fake("perception", "FakePerception")([hud]),
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
        # as cli (T-0200 C9): an executor note is a ``note`` record in the session's current log
        current = cli._CurrentSessionLog(session)
        browser.on_note = lambda data: current.record("note", data, session_ms=clock.now_ms)
        await asyncio.wait_for(browser.open(origin + row.start_page), timeout=G.BROWSER_CALL_S)
        # C3: never ask before the seed is applied; a page that never gets ready raises here and is never asked
        await asyncio.wait_for(
            browser.driver.page.wait_for_function(READY_JS, timeout=READY_TIMEOUT_S * 1000),
            timeout=READY_TIMEOUT_S + 5,
        )
        await session.start()
        session.turn_written.clear()
        submitted = time.monotonic()
        session.submit(E.UserText(t_ms=clock.now_ms, text=row.request))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=TURN_TIMEOUT_S)
        except TimeoutError:
            timed_out = True
        wall_s = round(time.monotonic() - submitted, 2)
        for entry in reversed(tuple(session.state.history)):
            if entry.user_text.strip() == row.request.strip():
                history = entry.assistant_spoken or None
                break
    except Exception as exc:
        exception = repr(exc)
    finally:
        # read while the browser is still open; after the turn, never in a request (B-R23: Playwright's serializer)
        if browser is not None:
            state = await _read_editor_state(browser)
            with contextlib.suppress(Exception):
                egress = [dict(e) for e in browser.egress_log]
        sockets = non_loopback_sockets(G._descendants())
        if session is not None:
            await _bounded_close("session.aclose", session.aclose(), close_timeouts)
        for m in (meter, crop_meter):
            if m is not None:
                with contextlib.suppress(Exception):
                    await m.aclose()
        if browser is not None:
            await _bounded_close("browser.close", browser.close(), close_timeouts)
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
        task_id,
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
    trial, steps = annotate(task_id, trial, steps, records, state)
    trial.update(
        tools_version=tools_version,
        keywords_logged={
            k: extras.get(k)
            for k in ("screen_reminder", "overview_frame", "page_fact", "completion_check")
        },
        wall_s=wall_s,
        vram_min_free_mib=vram_min,
        engine_up_after=engine_up,
        budgets_sent=budgets_sent,
        guidance_sent=guidance_sent,
        resolver_calls=len(net.resolves),
        connector_calls=len(net.connects),
        egress_log=egress,
        close_timeouts=close_timeouts,
    )
    trial["session_log"] = Path(log.path).name if log is not None else None
    thoughts = F.thought_record("live", task_id, repeat, records) if records else None
    return trial, steps, thoughts


async def _run_all(run: _Run, turns: Sequence[tuple[str, int]]) -> None:
    """Every turn of ``turns``; after ``G.ERRORED_STOP`` consecutive errored turns, when a worst-case turn
    (``TURN_WORST_S``) plus the report (``REPORT_S``) no longer fits inside ``HARD_LIMIT_S`` since ``PROCESS_START_S``
    (C-R7), or past ``RUN_BUDGET_S`` (C-R2; probes included, checked before each turn), nothing more is sent and
    nothing more is written; the caller still writes the report."""
    errored = 0
    last = ""
    for index, (task_id, repeat) in enumerate(turns):
        started = time.monotonic()
        elapsed = started - PROCESS_START_S
        if errored >= G.ERRORED_STOP:
            run.stopped = f"{errored} consecutive errored turns (last: {last}; at turn {index})"
        elif elapsed + TURN_WORST_S + REPORT_S > HARD_LIMIT_S:
            run.stopped = (
                f"a worst-case turn ({TURN_WORST_S} s) and the report ({REPORT_S} s) no longer fit inside "
                f"{HARD_LIMIT_S} s from process start ({elapsed:.0f} s spent; at turn {index})"
            )
        elif elapsed > RUN_BUDGET_S:
            run.stopped = f"the run budget of {RUN_BUDGET_S} s is spent (at turn {index})"
        if run.stopped is not None:
            run.manifest["stopped"] = run.stopped
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.stopped}")
            return
        steps: list[dict[str, Any]] = []
        thoughts: dict[str, Any] | None = None
        try:
            trial, steps, thoughts = await _one_turn(run, task_id, repeat)
        except Exception as exc:
            trial = _not_run(task_id, repeat, f"harness: {exc!r}")
        errored = errored + 1 if G._errored(trial, steps) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "thoughts.jsonl", [thoughts] if thoughts is not None else [])
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:02d} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"calls={[s.get('call_name') for s in steps]} "
            f"focus={[s.get('focus_check') for s in steps]} "
            f"success={trial.get('success')} malformed={trial.get('malformed_state')} "
            f"end={trial.get('tool_loop_end')} timed_out={trial.get('timed_out')} "
            f"err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{task_id}/r{repeat}"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _write_report(
    out: Path, probes: Mapping[str, Any], manifest: Mapping[str, Any], stopped: str | None = None
) -> dict[str, Any]:
    report = build_report(
        G._read_jsonl(out / "trials.jsonl"), G._read_jsonl(out / "steps.jsonl"), probes, stopped
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


def provenance_sha256() -> str | None:
    """The sha256 of ``PROVENANCE_PATH`` (read at call time), ``None`` when absent."""
    path = Path(PROVENANCE_PATH)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _probe_passed(record: Mapping[str, Any]) -> bool:
    """P1-P3: T-0131's ``run_probes`` records ``ok``; a record that already carries ``passed`` (and no ``ok``) keeps
    it."""
    if "ok" in record:
        return bool(record.get("ok"))
    return bool(record.get("passed"))


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
    out: Path = args.out_dir or REPO / "logs" / f"t0210-threejs-{sha[:7]}"
    if out.exists():
        _say(
            f"refusing: {out} exists; the run happens once per SHA (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. group B merged (the copy, the shim, the judge); A is recorded, not required (C1)
    if not _merged(MERGE_B):
        _say(
            f"refusing: no '{MERGE_B}' merge is reachable from HEAD (T-0210-b, the three.js copy, its state shim and "
            "the judge; a branch merge does not count)"
        )
        return EXIT_REFUSED
    body_guard = _merged(MERGE_A)
    # 5. the table's pin and the copy's PROVENANCE.md pin (read at call time)
    table_sha = threejs_tasks.task_table_sha()
    if table_sha != TASK_TABLE_SHA256:
        _say(f"refusing: the task table's sha is {table_sha}, not the pinned {TASK_TABLE_SHA256}")
        return EXIT_REFUSED
    prov_sha = provenance_sha256()
    if prov_sha != PROVENANCE_SHA256:
        _say(
            f"refusing: {PROVENANCE_PATH} (the copy's PROVENANCE.md) has sha256 {prov_sha}, not the pinned "
            f"{PROVENANCE_SHA256}"
        )
        return EXIT_REFUSED
    # 6. HEAD's live default, read from cli now (C1/C2)
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    ns = D._live_args()
    if not cli.web_of(lean, ns, "live"):
        _say(
            "refusing: the live default is not web mode (cli.web_of is off); C runs web mode on loopback"
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
    # 7. the tool kit cli resolves (repair on), no search
    kit_options: dict[str, Any] = {}
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
            args.base_url, out, sha, model, lean, vram, default, keywords, body_guard, prov_sha
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
    body_guard: bool,
    prov_sha: str | None,
) -> int:
    """Past every refusal and the engine check: the order, the manifest, P1-P3, P-M1, P-M2, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    kit_sha = cli_kit_sha256(bool(default["page_overview"]), bool(default["wait_tool"]))
    turns = order(ORDER_SEED)
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in turns])
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0210_threejs_run.sh refuses while 127.0.0.1:8080/8081 are bound",
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
        "repeats": REPEATS,
        "browser": f"{BROWSER}, headless, one per turn, fresh context (no profile directory), web mode on one "
        "loopback origin per turn at the task's seeded start page, deny-all resolver and connector",
        "search": "off",
        "threejs_dir": str(THREEJS_DIR),
        "viewport": list(lean.browser_viewport),
        "turns": len(turns),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "hard_limit_s": HARD_LIMIT_S,
        "turn_worst_s": TURN_WORST_S,
        "turn_worst_note": TURN_WORST_NOTE,
        "report_s": REPORT_S,
        "close_timeout_s": CLOSE_S,
        "ready_timeout_s": READY_TIMEOUT_S,
        "state_read": STATE_READ_JS,
        "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
        "no audio device, no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "live_default": dict(default),
        "session_keywords": {k: D._jsonable(v) for k, v in keywords.items()},
        "search": "off",
        "body_guard": body_guard,
        "executor_constants": {k: D._jsonable(v) for k, v in executor_constants().items()},
        "task_table_sha256": threejs_tasks.task_table_sha(),
        "provenance_sha256": prov_sha,
        "tasks": [
            {
                "task_id": r.task_id,
                "seed": r.seed,
                "start_page": r.start_page,
                "request": r.request,
                "n": REPEATS,
            }
            for r in threejs_tasks.TASKS
        ],
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {rel: G._sha256(REPO / rel) for rel in GATE_FILES},
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "thoughts.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- C4: P1-P3 (T-0131's), P-M1 and P-M2, before any turn ---------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": _probe_passed(probes[name])}
    try:
        pm1 = dict(probe_pm1())
    except Exception as exc:
        pm1 = {"error": repr(exc)}
    probes["pm1"] = {**pm1, "passed": pm1_ok(pm1)}
    try:
        pm2 = dict(probe_pm2(pm1))
    except Exception as exc:
        pm2 = {"error": repr(exc)}
    probes["pm2"] = {**pm2, "passed": pm2_ok(pm2)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if probe_verdicts(probes)["void"]:
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest, stopped)
        _say(
            f"a probe failed: the run is void ({json.dumps(report['probes'])}); label {report['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = _Run(
        base_url=base_url, out=out, profile=lean, sentinel=sentinel, manifest=manifest, vram=vram
    )
    try:
        asyncio.run(_run_all(run, turns))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    if run.stopped is not None:
        manifest["stopped"] = run.stopped
        G._write_json(out / "manifest.json", manifest)

    # --- the report straight after the turns, before anything else can fail -------------------------
    report = _write_report(out, probes, manifest, run.stopped)
    _say(f"label={report['label']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
