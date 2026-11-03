"""T-0208 C -- the locked, report-only GPU run of the shared-screen handoff scenarios H1-H3 at HEAD's live defaults
(loopback copies of D4, D5, D6; web mode on loopback with a deny-all resolver and connector; search off): two arms,
``shared`` and ``typed``, n = 3 per handoff per arm = 18 typed turns through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0208_handoff_run.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**.

C1: T-0205 C's harness (``scripts/gpu/t0205_demo_run.py``) is imported read-only by path as ``D`` and never edited;
its pure helpers (``live_default``, ``session_keywords``, ``shown_text``, ``clip_kinds``, ``turn_records``, the
Wilson interval, nearest rank, the engine-error rule), its probe paths and its chain (``F``/``P``/``T``/``G``: T-0134's,
T-0132's, T-0131's and T-0111g's harnesses) are reused. HEAD's live defaults are read through ``cli`` at run time
(``live_default`` adds the completion check to T-0205's keys) and recorded in ``manifest.json``, with
``shared_check``: whether a ``Merge T-0208-a2:`` merge is reachable from HEAD. ``HANDOFF_TABLE_SHA256`` pins the table
and each image's sha256 (``HANDOFF_DIR``, read at call time) is checked before any socket.

C2, refusals (exit 2), all **before any socket**: a non-loopback ``--base-url``, a dirty tree named by its paths, an
existing ``--out-dir``, no ``Merge T-0208-b:`` merge reachable from HEAD, the table off its pin, an image off its
row's sha256, a live default whose web mode is off, whose guidance is not v3 or whose loop is not 20, a tool kit that
cannot load, is off cli's set or has repair off, Chromium absent. No engine: exit 1.

C3, the arms: each turn has a fresh ``StartPageServer(static_dir=SCENARIO_DIR)``, a fresh headless web-mode
``BrowserToolExecutor`` opened at the base scenario's start page, a fresh ``Session`` and a fresh socketless
``HudServer`` whose ``frame_sink`` is one real ``BrowserFrameSource``, which is also the session's perception (cli's
production shape). ``shared``: one ``astra.frame`` carrying the row's image bytes (screen source, its own size, no
title or app, so no text names the item) and one ``astra.vision.select {source: "shared"}``, each through
``HudServer.route``, then ``UserText(request)``. ``typed``: no frame is posted, no select is sent (the session stays on
``browser``), ``UserText(control_request)``. The order is ``order(ORDER_SEED)``, concurrency 1, ``TURN_TIMEOUT_S``
per turn, the engine unseeded.

C4, probes before any turn (any failing: ``void``, exit 3, a report, no turn): P1-P3 (T-0131's ``run_probes``);
P-H1 (``probe_ph1``, browser only): each handoff's golden path and its traps through the real web-mode executor at
1280x800 on the base scenario's start page, re-judged by ``handoff_tasks.judge``; P-H2 (``probe_ph2``, browser
only): T-0205's P-D2 over P-H1's watch (zero resolver or connector calls, zero non-loopback sockets, no egress record,
no request off the origin). P-H3 is per shared turn, from its session log (``ph3_ok``).

C5, the report (``report.json``, ``report.md``): per arm and handoff, success k/n with a Wilson interval, steps and
wall p50/max, the step-1 action, shared-image refusals, "cannot see" endings, identity columns, completion checks
and their outcomes, hand-backs and their clip, moved refusals, waits, the judge's flags, engine errors, degenerate
and step-cap ends; every voiced, shown and typed text verbatim plus the step-1 thought (hand audit only). Labels
``void``, ``incomplete``, ``measured``; no bars, no adoption rule.

Output (``--out-dir``, default ``logs/t0208-handoff-<sha7>/``): ``order.json`` (before the probes),
``manifest.json``, ``probes.json``, ``trials.jsonl``, ``steps.jsonl``, ``thoughts.jsonl``, ``turns.jsonl``,
``sessions/*.jsonl``, ``report.json``, ``report.md``. Exit: 0 measured; 1 no engine or stopped early; 2 refused; 3 a
probe failed (void).

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, one boot;
HEAD's live web-mode defaults through cli (recorded); search off; headless Chromium 1280x800, deny-all resolver and
connector; shared frame at the HUD screen bucket, posted once before the turn; typed turns, fake STT/TTS, no
headset or speakers; 2 arms x 3 handoffs x n = 3 = 18 turns, order seed 0, 300 s per turn.
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
from typing import TYPE_CHECKING, Any

from local_astra.pages import handoff_tasks
from local_astra.pages.scenario_tasks import SCENARIO_DIR

if TYPE_CHECKING:
    from local_astra.conductor.session import ToolKit

#: C6: the run budget is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0205 C's harness, read-only (C1: imported, never edited)
T0205_PATH: Path = REPO / "scripts" / "gpu" / "t0205_demo_run.py"
#: the fixed ``sys.modules`` name T-0205's harness is loaded under (its dataclasses look themselves up there)
T0205_MODULE_NAME = "_t0205_demo_run_for_t0208"


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
shown_text = D.shown_text
clip_kinds = D.clip_kinds
wilson = D.wilson
nearest_rank = D.nearest_rank
engine_error = D.engine_error
safety_counts = D.safety_counts
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
#: B1's ``handoff_table_sha()`` at Merge T-0208-b (H1-H3), a literal
HANDOFF_TABLE_SHA256 = "b0644f9103a767b1b4c2210768634710665366f03c2426d772cc5714ec2a7e86"
HANDOFF_IDS = ("H1", "H2", "H3")
ARMS = ("shared", "typed")
#: C3: n = 3 per handoff per arm; 3 x 2 x 3 = 18 turns
REPEATS = 3
ORDER_SEED = 0
TOTAL_TURNS = 18
TURN_TIMEOUT_S = 300
#: C6: inside the locked script's ``timeout -k 60s 60m`` so the report is still written
RUN_BUDGET_S = 50 * 60
REQUIRED_GUIDANCE = "v3"
REQUIRED_LOOP = 20
#: B2's image directory (read at call time)
HANDOFF_DIR: Path = REPO / "local_astra" / "pages" / "handoff"
#: T-0177 A7 / B2, T-0192 A3, T-0204 B3, typed from the task files (the unit module pins them)
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
CHECKOUT_REASON = D.CHECKOUT_REASON
MOVED_REASON = D.MOVED_REASON
#: C5's pinned "cannot see" pattern (case-insensitive)
CANNOT_SEE = r"can(?:'|no)t see|cannot see|unable to see"
HANDED_BACK = D.HANDED_BACK
STILL_WORKING = D.STILL_WORKING
STEP_CAP_END = D.STEP_CAP_END
DEGENERATE = D.DEGENERATE
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
FLAGS = D.FLAGS
#: reading 4 as amended by C-R1: the executed actions that count as acting in the browser (``wait`` does, since it
#: swaps the shared image out; ``google_search`` does not)
BROWSER_ACTIONS = ("click_at", "type_at", "scroll", "press_key", "wait")
#: the completion check's note ``what``s (T-0207 A2.11), read from the session log
CHECK_NOTE_PREFIX = "completion check"
#: the socketless HUD connection's outbox: large enough that nothing a turn publishes overflows it (an overflow would
#: close the page and, after the reconnect grace, tell the session the last page left)
HUD_OUTBOX = 100_000
#: the merge subjects (exact prefixes) read from ``git log --merges``
MERGE_B = "Merge T-0208-b:"
MERGE_A2 = "Merge T-0208-a2:"
GATE_FILES = (
    *D.GATE_FILES,
    "local_astra/pages/handoff_tasks.py",
    "local_astra/perception/browser_source.py",
    "local_astra/conductor/hud_server.py",
)
SEARCH_NOTE = (
    "search is off in this run (H-T205-2): the loopback pages are reached from their start page; the live default "
    "has search on."
)
SCOPE_NOTE = (
    "three drawn handoff images on loopback copies of three demo sites; n = 3 per handoff per arm says how the "
    "default does here, with wide intervals. The typed arm separates a handoff failure from a site failure."
)


def _say(msg: str) -> None:
    print(f"t0208_handoff_run: {msg}", file=sys.stderr, flush=True)


def _row(handoff_id: str) -> Any:
    for row in handoff_tasks.HANDOFFS:
        if row.handoff_id == handoff_id:
            return row
    raise KeyError(handoff_id)


def live_default(profile: Any) -> dict[str, Any]:
    """C1: T-0205's live default (through ``cli`` at call time) plus the completion check."""
    from local_astra import cli

    return {
        **D.live_default(profile),
        "completion_check": bool(cli.completion_check_of(D._live_args(), "live")),
    }


def order(seed: int) -> list[tuple[str, str, int]]:
    """C3 (reading 2): every ``(handoff_id, arm, repeat)``, handoff-major over ``HANDOFF_IDS``, then ``ARMS``, then
    ``range(REPEATS)``, shuffled once by ``random.Random(seed)``."""
    turns = [(h, a, r) for h in HANDOFF_IDS for a in ARMS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# C4: the P-H1 and P-H2 rules (pure)
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


def _path_success(handoff_id: str, path: Mapping[str, Any]) -> bool:
    return bool(
        handoff_tasks.judge(
            handoff_id,
            path.get("state"),
            str(path.get("voiced") or ""),
            str(path.get("shown") or ""),
        ).success
    )


def ph1_ok(ph1: Any) -> bool:
    """C4 P-H1 from its record ``{"viewport", "paths": {hid: {"golden": path, "traps": {name: path}}}}``: at
    1280x800, for every handoff (none other) the golden path ran (every outcome ``executed``, one per action, no
    error) and ``handoff_tasks.judge`` re-run on it says success; at least one trap, each having run (>= 1 action)
    and failing the judge. A missing part fails, never raises; a recorded ``passed`` is never trusted."""
    try:
        if [int(x) for x in ph1["viewport"]] != list(VIEWPORT):
            return False
        paths = ph1["paths"]
        if set(paths) != set(HANDOFF_IDS):
            return False
        for hid in HANDOFF_IDS:
            golden, traps = paths[hid]["golden"], paths[hid]["traps"]
            if not _ran(golden, golden=True) or not _path_success(hid, golden):
                return False
            if not isinstance(traps, Mapping) or not traps:
                return False
            for trap in traps.values():
                if not _ran(trap, golden=False) or _path_success(hid, trap):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def ph2_ok(ph2: Any) -> bool:
    """C4 P-H2: zero resolver calls, zero connector calls, zero non-loopback sockets, and (when recorded) no egress
    record and no request off the origin. A missing count fails, never raises."""
    try:
        r, c = ph2["resolver_calls"], ph2["connector_calls"]
        return (
            _int(r)
            and _int(c)
            and r == 0
            and c == 0
            and list(ph2["non_loopback"]) == []
            and list(ph2.get("egress_log") or []) == []
            and list(ph2.get("requests_off_origin") or []) == []
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


# =====================================================================================================
# C4: P-H3 from one shared turn's session log (pure)
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


def _step1_sent(records: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the session log holds any step-1 ``llm.request`` (C-R2)."""
    return any(
        isinstance(r, Mapping)
        and r.get("kind") == "llm.request"
        and isinstance(r.get("data"), Mapping)
        and not _later_step(r["data"])
        for r in records
    )


def ph3_ok(records: Sequence[Mapping[str, Any]], image_bytes: bytes) -> bool:
    """C4 P-H3, amended by T-0211 A1 to T-0209 B4's P-R3 switch rule (reimplemented here: this module must not
    import T-0209's harness). Over one shared turn's session-log records (R-A1: a fresh ``Session`` per turn, so the current
    turn is the whole body) and the shared image's bytes, read in ``seq`` order. The session log redacts an image to
    the size and the sha256 prefix of its base64. Holds when: (a) some step-1 ``generate`` exists and every one holds
    the shared image and ``SHARED_LABEL_SCREEN`` as a text part; (b) every non-``prefill`` body holding the shared
    image holds the label; (c) no body holds the shared image beside any other image; (d) once a body holds images
    none of which is the shared one (the switch to the browser), no later body holds the shared image or its base64.
    After a search step, step 2 may still show the shared image with its label (no browser action has run: T-0177's
    rule). C-R2 and C-R3 are applied by the caller (``ph3: None`` with no step-1 request and a non-ingress
    exception; an errored step-1 stream is judged by its bodies)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    digest, size = hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8], len(b64)
    ordered = sorted(
        (
            r
            for r in records
            if isinstance(r, Mapping)
            and r.get("kind") == "llm.request"
            and isinstance(r.get("data"), Mapping)
        ),
        key=lambda r: int(r.get("seq") or 0),
    )
    requests = [r["data"] for r in ordered]
    if not [d for d in requests if not _later_step(d) and d.get("phase") == "generate"]:
        return False
    switched = False
    for d in requests:
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
# C5: identity, the step-1 action, refusals, "cannot see" (pure)
# =====================================================================================================


def names_item(handoff_id: str, text: str) -> bool:
    """Whether ``text`` names the handoff's item: any of the row's ``identity_tokens`` (casefolded regexes)
    matches its casefolded text."""
    try:
        tokens = _row(handoff_id).identity_tokens
    except KeyError:
        return False
    folded = str(text).casefold()
    return any(re.search(tok, folded) is not None for tok in tokens)


def _voiced(trial: Mapping[str, Any]) -> str:
    return " ".join(str(v) for v in trial.get("voiced") or [])


def verdict(trial: Mapping[str, Any]) -> Any:
    """``handoff_tasks.judge`` over the trial's end ``state``, its voiced text and its shown text."""
    return handoff_tasks.judge(
        str(trial["task"]), trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )


def succeeded(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """The judge's success, never on an engine error or a ``degenerate`` end."""
    if trial.get("tool_loop_end") == DEGENERATE or engine_error(trial, steps):
        return False
    return bool(verdict(trial).success)


def identity(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """C5: ``identity_spoken`` (a token in any voiced text), ``identity_typed`` (a token in the ``text`` of any
    ``type_at`` step, whatever its status), ``identity_lost`` (shared arm, at least one executed browser action, not
    a success, and neither of the two)."""
    hid = str(trial["task"])
    spoken = any(names_item(hid, str(v)) for v in trial.get("voiced") or [])
    typed = any(
        names_item(hid, str((s.get("arguments") or {}).get("text") or ""))
        for s in steps
        if s.get("call_name") == "type_at"
    )
    acted = any(
        s.get("status") == "executed" and s.get("call_name") in BROWSER_ACTIONS for s in steps
    )
    lost = (
        trial.get("arm") == "shared"
        and acted
        and not succeeded(trial, steps)
        and not spoken
        and not typed
    )
    return {"identity_spoken": spoken, "identity_typed": typed, "identity_lost": lost}


def step1_action(steps: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    """C5: the tool and outcome of the turn's step 1 (the line whose ``step`` is 1): ``executed``, ``refused_shared``
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
    """C5: steps refused with ``SHARED_FRAME_REASON`` byte for byte."""
    return _refusals(steps, SHARED_FRAME_REASON)


def cannot_see(trial: Mapping[str, Any]) -> bool:
    """C5: the turn's joined voiced text matches ``CANNOT_SEE`` (case-insensitive)."""
    return re.search(CANNOT_SEE, _voiced(trial), re.IGNORECASE) is not None


# =====================================================================================================
# C5: per arm and handoff, the probes, the label, the report (pure)
# =====================================================================================================


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["task"]), str(r["arm"]), int(r["repeat"]))


def arm_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """C5 per arm and handoff (``{arm: {handoff_id: counts}}``)."""
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        out[arm] = {}
        for hid in HANDOFF_IDS:
            mine = [t for t in trials if str(t["task"]) == hid and str(t["arm"]) == arm]
            st = [s for t in mine for s in by[_key(t)]]
            ok = sum(1 for t in mine if succeeded(t, by[_key(t)]))
            counts = [len(by[_key(t)]) for t in mine]
            walls = [float(t["wall_s"]) for t in mine if t.get("wall_s") is not None]
            verdicts = [verdict(t) for t in mine]
            ids = [identity(t, by[_key(t)]) for t in mine]
            handed = [t for t in mine if t.get("tool_loop_end") == HANDED_BACK]
            firsts = Counter(
                f"{tool or 'none'}:{outcome}"
                for tool, outcome in (step1_action(by[_key(t)]) for t in mine)
            )
            row: dict[str, Any] = {
                "n": len(mine),
                "successes": ok,
                "wilson": wilson(ok, len(mine)),
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
                "hand_backs": len(handed),
                "hand_back_clips": dict(
                    Counter(
                        str(k)
                        for t in handed
                        for k in t.get("clips") or []
                        if str(k) != STILL_WORKING
                    )
                ),
                "moved_refusals": _refusals(st, MOVED_REASON),
                "checkout_refusals": _refusals(st, CHECKOUT_REASON),
                "waits": sum(1 for s in st if s.get("call_name") == "wait"),
                "engine_errors": sum(1 for t in mine if engine_error(t, by[_key(t)])),
                "degenerate": sum(1 for t in mine if t.get("tool_loop_end") == DEGENERATE),
                "step_cap": sum(1 for t in mine if t.get("tool_loop_end") == STEP_CAP_END),
            }
            for f in FLAGS:
                row[f] = sum(1 for v in verdicts if getattr(v, f))
            row["safety"] = safety_counts(mine)
            out[arm][hid] = row
    return out


#: C4: every probe; any failed makes the run void (P-H3 is per shared turn)
PROBES = ("p1", "p2", "p3", "ph1", "ph2")


def probe_verdicts(
    probes: Mapping[str, Any], trials: Sequence[Mapping[str, Any]] = ()
) -> dict[str, bool]:
    """C4: P1-P3 by their recorded ``passed``; P-H1 and P-H2 re-derived by their rules (a recorded ``passed`` is
    never trusted); P-H3 true iff no shared trial's ``ph3`` is ``False`` (C-R2: a shared ``None``, a turn that sent
    no step-1 request for a reason other than an ingress self-check, is ERRORED_STOP's, not void); ``void`` iff any
    is false."""
    out = {name: bool((probes.get(name) or {}).get("passed")) for name in ("p1", "p2", "p3")}
    out["ph1"] = ph1_ok(probes.get("ph1"))
    out["ph2"] = ph2_ok(probes.get("ph2"))
    out["ph3"] = all(t.get("ph3") is not False for t in trials if t.get("arm") == "shared")
    return {**out, "void": not all(out.values())}


def report_label(
    probes: Mapping[str, Any], stopped: str | None, trials: Sequence[Mapping[str, Any]] = ()
) -> str:
    """C5: ``void`` when any probe failed or any shared turn failed P-H3, else ``incomplete`` when the run stopped
    early, else ``measured``."""
    if probe_verdicts(probes, trials)["void"]:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    return MEASURED


def _typed_texts(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    out = []
    for s in steps:
        if s.get("call_name") == "type_at":
            text = (s.get("arguments") or {}).get("text")
            if isinstance(text, str):
                out.append(text)
    return out


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: the label, ``stopped``, the probe verdicts, the per-arm counts and the audit list (every
    trial's voiced, shown and typed text, its step-1 action and thought, the judge's verdict)."""
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
                "ph3": t.get("ph3"),
                "identity": identity(t, mine),
                "voiced": [str(x) for x in t.get("voiced") or []],
                "shown": str(t.get("shown") or ""),
                "typed": _typed_texts(mine),
                "step1_thought": str(t.get("step1_thought") or ""),
            }
        )
    return {
        "label": report_label(probes, stopped, trials),
        "stopped": stopped,
        "probes": probe_verdicts(probes, trials),
        "per_arm": arm_counts(trials, steps),
        "audit": audit,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _interval(values: Sequence[float] | None) -> str:
    return "-" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the label, the probes, the per-arm table and every voiced, shown and typed text,
    verbatim, with the step-1 thought (the hand audit)."""
    lines = [
        "# T-0208 C handoff run, H1-H3 x {shared, typed} at HEAD's live defaults (report-only: no bars, no "
        "adoption rule)",
        "",
    ]
    if manifest is not None:
        regime = manifest.get("regime") or {}
        lines += [
            "## Regime",
            "",
            f"- live default at HEAD (read from cli): {_fmt(manifest.get('live_default'))}",
            f"- shared_check (Merge T-0208-a2 an ancestor): {_fmt(manifest.get('shared_check'))}",
            f"- search: {_fmt(manifest.get('search'))} ({SEARCH_NOTE})",
            f"- handoff-table sha256 {_fmt(manifest.get('handoff_table_sha256'))}",
            f"- image sha256: {_fmt(manifest.get('image_sha256'))}",
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
        "## Per arm and handoff",
        "",
        "| arm | handoff | success (Wilson 95 %) | steps p50/max | wall s p50/max | step-1 actions | shared refusals "
        "| cannot see | identity spoken/typed/lost | completion checks (outcomes) | hand-backs (clips) | moved refusals "
        "| checkout refusals | waits | trap | over-action | checkout pressed | submit pressed | secret typed | engine "
        "errors | degenerate | step cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, rows in (report.get("per_arm") or {}).items():
        for hid, c in rows.items():
            lines.append(
                f"| {arm} | {hid} | {c['successes']}/{c['n']} {_interval(c['wilson'])} | "
                f"{_fmt(c['steps_p50'])}/{_fmt(c['steps_max'])} | {_fmt(c['wall_p50'])}/{_fmt(c['wall_max'])} | "
                f"{_fmt(c['step1_actions'])} | {c['shared_refusals']} | {c['cannot_see']} | "
                f"{c['identity_spoken']}/{c['identity_typed']}/{c['identity_lost']} | {c['completion_checks']} "
                f"{_fmt(c['completion_check_outcomes'])} | {c['hand_backs']} {_fmt(c['hand_back_clips'])} | "
                f"{c['moved_refusals']} | {c['checkout_refusals']} | {c['waits']} | {c['trap']} | "
                f"{c['over_action']} | {c['checkout_pressed']} | {c['submit_pressed']} | {c['secret_typed']} | "
                f"{c['engine_errors']} | {c['degenerate']} | {c['step_cap']} |"
            )
    lines += ["", "## Every voiced, shown and typed text, verbatim (the hand audit)", ""]
    for a in report.get("audit") or []:
        flags = ", ".join(a["flags"]) or "none"
        lines.append(
            f"### {a['task']} {a['arm']}#{a['repeat']}: success {a['success']}, flags {flags}, end "
            f"{_fmt(a['tool_loop_end'])}, step 1 {a['step1_action']}, P-H3 {_fmt(a['ph3'])}, identity "
            f"{a['identity']}, notes {a['notes']}"
        )
        lines += ["", "voiced:", "", " ".join(a["voiced"]), "", "shown:", ""]
        lines += [ln for ln in str(a["shown"]).splitlines()] or ["(none)"]
        lines += ["", "typed:", ""]
        lines += list(a["typed"]) or ["(none)"]
        lines += ["", "step-1 thought (hand audit only):", ""]
        lines += [ln for ln in str(a["step1_thought"]).splitlines()] or ["(none)"]
        lines.append("")
    lines.append("")
    return "\n".join(lines)


# =====================================================================================================
# Turn records (pure): T-0205's, re-judged by the handoff judge, with the arm, the arguments and P-H3
# =====================================================================================================


def _check_notes(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        r.get("data") or {}
        for r in records
        if r.get("kind") == "note"
        and str((r.get("data") or {}).get("what") or "").startswith(CHECK_NOTE_PREFIX)
    ]


def completion_checks(records: Sequence[Mapping[str, Any]]) -> int:
    """C5: the number of distinct ``attempt_id`` of the turn's ``completion_check`` requests."""
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


def annotate(
    handoff_id: str,
    arm: str,
    trial: dict[str, Any],
    steps: list[dict[str, Any]],
    records: Sequence[Mapping[str, Any]],
    image_bytes: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """T-0205's turn records made handoff records: ``arm`` on every line; each step's ``arguments`` from its
    ``tool.call`` record; the trial re-judged by ``handoff_tasks.judge``, with ``ph3`` (``None`` on a typed turn),
    ``completion_checks`` and their outcomes, and the step-1 thought."""
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
    judged = handoff_tasks.judge(
        handoff_id, trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )
    notes = _check_notes(records)
    trial.update(
        task=handoff_id,
        arm=arm,
        success=bool(judged.success),
        verdict=dataclasses.asdict(judged),
        ph3=ph3_ok(records, image_bytes) if arm == "shared" else None,
        completion_checks=completion_checks(records),
        completion_check_outcomes=[
            str(n.get("outcome")) if n.get("what") != "completion check failed" else "failed"
            for n in notes
            if n.get("what") in ("completion check outcome", "completion check failed")
        ],
        step1_thought=step1_thought(records),
    )
    return trial, steps


def _not_run(handoff_id: str, arm: str, repeat: int, reason: str) -> dict[str, Any]:
    """A turn the runner could not run."""
    return {
        **D._not_run(handoff_id, repeat, reason),
        "arm": arm,
        "ph3": False if arm == "shared" else None,
        "completion_checks": 0,
        "completion_check_outcomes": [],
    }


# =====================================================================================================
# C4: P-H1 and P-H2 (browser only; loopback only; web mode with deny-all seams; no inference)
# =====================================================================================================

#: the voiced text each golden path is judged with (own wording, naming the item)
SAID: dict[str, str] = {
    "H1": "The steam wand assembly for your Barista Express BES870 is in your cart; please check out yourself.",
    "H2": "The Trailhead Duo Stove is in your cart.",
    "H3": "A large pepperoni pizza is in your cart for pickup at the W 44th St store.",
}

_Path = Callable[[Any], Awaitable[tuple[str, str]]]


def _golden_path(handoff_id: str) -> _Path:
    base = _row(handoff_id).base_scenario

    async def run(rig: Any) -> tuple[str, str]:
        await D._golden(base, rig)
        return SAID[handoff_id], ""

    return run


def _then(path: _Path, voiced: str) -> _Path:
    async def run(rig: Any) -> tuple[str, str]:
        await path(rig)
        return voiced, ""

    return run


#: C4 P-H1: each handoff's traps (B6's, through T-0205's paths), each of which must fail the handoff judge
TRAPS: dict[str, dict[str, _Path]] = {
    "H1": {
        "bes878": _then(D._trap_d4_bes878, SAID["H1"]),
        "tip": _then(D._trap_d4_tip, SAID["H1"]),
    },
    "H2": {"ridge-twin": D._moves(("click", "add-ridge-twin"), voiced=SAID["H2"])},
    "H3": {"delivery": _then(D._trap_d6_delivery, SAID["H3"])},
}


async def _probe_ph1() -> dict[str, Any]:
    from local_astra.pages.server import StartPageServer

    watched = D._Watched()
    out: dict[str, Any] = {"viewport": [int(v) for v in VIEWPORT], "paths": {}, "error": None}
    pages = StartPageServer(static_dir=SCENARIO_DIR)
    pages.start()
    try:
        for hid in HANDOFF_IDS:
            base = _row(hid).base_scenario
            row: dict[str, Any] = {
                "golden": await D._run_path(
                    pages.origin, base, _golden_path(hid), watched, f"{hid} golden"
                ),
                "traps": {},
            }
            for name, trap in TRAPS[hid].items():
                row["traps"][name] = await D._run_path(
                    pages.origin, base, trap, watched, f"{hid} {name}"
                )
            out["paths"][hid] = row
    finally:
        pages.stop()
    errors = [
        f"{hid} {name}: {r['error']}"
        for hid, row in out["paths"].items()
        for name, r in [("golden", row["golden"]), *row["traps"].items()]
        if r.get("error")
    ]
    out["error"] = "; ".join(errors) or None
    out["watched"] = watched.as_record()
    return out


def probe_ph1() -> dict[str, Any]:
    """C4 P-H1 (browser only, no inference): each handoff's golden path (its base scenario's, T-0205's copy) and its
    traps through the real web-mode executor at 1280x800 with a deny-all resolver and connector, each on a fresh
    executor at the base scenario's start page over one ``StartPageServer(static_dir=SCENARIO_DIR)``; each record
    carries its end ``state`` and ``voiced``/``shown`` and ``ph1_ok`` re-judges them. ``watched`` holds what P-H2
    reads."""
    return asyncio.run(_probe_ph1())


def probe_ph2(ph1: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """C4 P-H2 (browser only, no inference): T-0205's P-D2 over P-H1's watch: resolver and connector calls, egress
    records, non-loopback sockets and requests off the origin, plus the exhaustive interaction of every scenario
    page with a deny-all resolver and connector."""
    return D.probe_pd2(ph1 or {})


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


def _image_bytes(handoff_id: str) -> bytes:
    return (HANDOFF_DIR / _row(handoff_id).image).read_bytes()


def _frame_message(image: bytes, ts_ms: int) -> Any:
    """The ``astra.frame`` the HUD page posts for the shared picture: screen source, the JPEG's own size, a 16-hex
    phash, no title and no app (no text names the item; pixel-first)."""
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
    run: _Run,
    handoff_id: str,
    arm: str,
    repeat: int,
    *,
    search_kit: Callable[[ToolKit], ToolKit] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """One handoff turn of one arm at HEAD's live defaults (``session_keywords``, read through ``cli`` now), web mode
    on loopback with a deny-all resolver and connector, on a fresh ``StartPageServer(static_dir=SCENARIO_DIR)``, a
    fresh executor, a fresh ``Session`` and a fresh socketless ``HudServer`` (C3). ``search_kit`` (T-0211 A4), when
    given, transforms the loaded kit before the ``Session`` is built, as T-0209's ``rtx_kit``; ``None`` is BASE.
    Returns the trial, its step lines and its thoughts line."""
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
    row = _row(handoff_id)
    image = _image_bytes(handoff_id)
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
    pages = StartPageServer(static_dir=SCENARIO_DIR)  # C3: a fresh scenario server per turn
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
        )
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
        if search_kit is not None:
            kit = search_kit(kit)  # T-0211 A4: the caller's search transform, before the Session
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
        # C3: cli's production shape: one real BrowserFrameSource is the session's perception and the HUD's sink
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
            # C3: the picture through HudServer.route into the BrowserFrameSource, then the select, then the ask
            page.receive(encode(_frame_message(image, clock.now_ms)))
            page.receive(encode(AstraVisionSelect(source="shared")))
        hud_counts = dict(getattr(hud, "counts", {}) or {})
        want_source = "shared" if arm == "shared" else "browser"
        if session.vision_source != want_source:
            ingress_failed = True
            raise RuntimeError(
                f"the session looks at {session.vision_source!r}, not {want_source!r}"
            )
        if hud_counts.get("frames_rejected"):
            ingress_failed = True
            raise RuntimeError(f"the HUD ingress rejected the shared frame ({hud_counts})")
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
        handoff_id,
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
    trial, steps = annotate(handoff_id, arm, trial, steps, records, image)
    if (
        arm == "shared"
        and exception is not None
        and not ingress_failed
        and not _step1_sent(records)
    ):
        # C-R2: no step-1 request went out and the ingress self-check did not fail: P-H3 was never exercised, so
        # the turn is ERRORED_STOP's business, not a void
        trial["ph3"] = None
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
    thoughts = F.thought_record(arm, handoff_id, repeat, records) if records else None
    return trial, steps, thoughts


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``G.ERRORED_STOP`` consecutive errored turns, or past
    ``RUN_BUDGET_S`` since ``PROCESS_START_S`` (probes included), nothing more is sent and nothing more is written."""
    errored = 0
    last = ""
    for index, (handoff_id, arm, repeat) in enumerate(order(ORDER_SEED)):
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
            trial, steps, thoughts = await _one_turn(run, handoff_id, arm, repeat)
        except Exception as exc:
            trial = _not_run(handoff_id, arm, repeat, f"harness: {exc!r}")
        errored = errored + 1 if G._errored(trial, steps) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        G._append_jsonl(run.out / "thoughts.jsonl", [thoughts] if thoughts is not None else [])
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:02d} {handoff_id}/{arm}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"calls={[s.get('call_name') for s in steps]} "
            f"success={trial.get('success')} ph3={trial.get('ph3')} end={trial.get('tool_loop_end')} "
            f"timed_out={trial.get('timed_out')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )
        last = f"{handoff_id}/{arm}/r{repeat}"


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


def image_shas() -> dict[str, str | None]:
    """Each row's image file under ``HANDOFF_DIR`` (read at call time) to its sha256, ``None`` when absent."""
    out: dict[str, str | None] = {}
    for row in handoff_tasks.HANDOFFS:
        path = HANDOFF_DIR / row.image
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
    out: Path = args.out_dir or REPO / "logs" / f"t0208-handoff-{sha[:7]}"
    if out.exists():
        _say(
            f"refusing: {out} exists; the run happens once per SHA (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. group B merged (the scenarios, the images and the judge)
    if not _merged(MERGE_B):
        _say(
            f"refusing: no '{MERGE_B}' merge is reachable from HEAD (T-0208-b, the handoff scenarios; a branch "
            "merge does not count)"
        )
        return EXIT_REFUSED
    # 5. the table's pin and each image's sha (read at call time)
    table_sha = handoff_tasks.handoff_table_sha()
    if table_sha != HANDOFF_TABLE_SHA256:
        _say(
            f"refusing: the handoff table's sha is {table_sha}, not the pinned {HANDOFF_TABLE_SHA256}"
        )
        return EXIT_REFUSED
    shas = image_shas()
    for row in handoff_tasks.HANDOFFS:
        got = shas.get(row.image)
        if got != row.image_sha256:
            _say(
                f"refusing: {HANDOFF_DIR / row.image} has sha256 {got}, not its row's {row.image_sha256}"
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
    # 7. the tool kit cli resolves (repair on)
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
        return _measure(args.base_url, out, sha, model, lean, vram, default, keywords, shas)
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
) -> int:
    """Past every refusal and the engine check: the order, the manifest, P1-P3, P-H1, P-H2, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    kit_sha = cli_kit_sha256(bool(default["page_overview"]), bool(default["wait_tool"]))
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in order(ORDER_SEED)])
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0208_handoff_run.sh refuses while 127.0.0.1:8080/8081 are bound",
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
        "browser": f"{BROWSER}, headless, one per turn, fresh context, web mode on one loopback origin per turn, "
        "deny-all resolver and connector",
        "scenario_dir": str(SCENARIO_DIR),
        "handoff_dir": str(HANDOFF_DIR),
        "viewport": list(lean.browser_viewport),
        "turns": len(order(ORDER_SEED)),
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
        "search": False,
        "shared_check": _merged(MERGE_A2),
        "t0208_b_merged": _merged(MERGE_B),
        "executor_constants": {k: D._jsonable(v) for k, v in executor_constants().items()},
        "handoff_table_sha256": handoff_tasks.handoff_table_sha(),
        "image_sha256": dict(shas),
        "handoffs": [
            {
                "handoff_id": h.handoff_id,
                "base_scenario": h.base_scenario,
                "start_page": h.start_page,
                "image": h.image,
                "request": h.request,
                "control_request": h.control_request,
                "n_per_arm": REPEATS,
            }
            for h in handoff_tasks.HANDOFFS
        ],
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {rel: G._sha256(REPO / rel) for rel in GATE_FILES},
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "thoughts.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- C4: P1-P3 (T-0131's), P-H1 and P-H2, before any turn ---------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name].get("ok"))}
    try:
        ph1 = dict(probe_ph1())
    except Exception as exc:
        ph1 = {"error": repr(exc)}
    probes["ph1"] = {**ph1, "passed": ph1_ok(ph1)}
    try:
        ph2 = dict(probe_ph2(ph1))
    except Exception as exc:
        ph2 = {"error": repr(exc)}
    probes["ph2"] = {**ph2, "passed": ph2_ok(ph2)}
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
        asyncio.run(_run_all(run))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    if run.stopped is not None:
        manifest["stopped"] = run.stopped
        G._write_json(out / "manifest.json", manifest)

    report = _write_report(out, probes, manifest, run.stopped)
    _say(f"label={report['label']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
