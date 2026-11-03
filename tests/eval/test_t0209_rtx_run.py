"""T-0209 B7: the loopback GPU run on the RTX scenarios (R1 only until Merge T-0209-a2; two arms, ``shared`` and
``typed``, n = 3 per scenario per arm, HEAD's live web-mode defaults on loopback, search on through ``LoopbackSearch``),
re-derived from the run's records and compared with ``report.json`` (``gpu``, ``eval``; report-only: no bars).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0209_rtx_run.sh``)::

    T0209_RUN_DIR=logs/t0209-rtx-<sha7> uv run pytest tests/eval/test_t0209_rtx_run.py -q -m gpu -rA

Every run row skips when ``T0209_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time
but are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness** (B7, as T-0208 C7). It never imports the harness (nor any script under
``scripts/``) and, from ``local_astra``, imports only ``local_astra.pages.rtx_tasks.judge`` (B7 allows it; the judge
is pinned by ``tests/unit/test_t0209_rtx_judge.py``). The ids, arms, reasons, labels, identity tokens, shas and image
names below are typed from docs/tasks/T-0209.md (A1.6, A1.9, B1-B5), T-0208 (C4, C5), T-0177 A7 / B2, T-0192 A3 and
T-0204 B3; ``tests/unit/test_t0209_rtx_harness.py`` pins them against production and runs every pure row against
this module too (``eval-oracle``), with hand-computed expectations, so a wrong oracle is caught on CPU.

**Record fields read** (the harness surface is stated in the unit module's docstring):

- every trial and step line carries ``task`` (the rtx id), ``arm`` (``"shared"`` or ``"typed"``) and ``repeat``;
- a trial carries T-0205 C's trial fields (``state``, ``voiced`` (a list), ``shown``, ``clips``, ``tool_loop_end``,
  ``timed_out``, ``engine_up_after``, ``wall_s``, ``session_log``), plus ``completion_checks`` and ``pr3`` (the P-R3
  verdict of a shared turn, ``None`` for a typed one);
- a step carries T-0205 C's step fields (``step``, ``http_status``, ``retries``, ``status``, ``reason``,
  ``call_name``) plus ``arguments`` (the call's arguments as its ``tool.call`` record logged them; a
  ``google_search`` step's ``query`` among them);
- ``probes.json``: ``p1``-``p3`` (T-0131's, each with ``passed``), ``pr1``, ``pr2`` (shapes in ``pr1_ok`` /
  ``pr2_ok``);
- ``report.json``: ``{"label", "stopped", "probes", "per_arm": {arm: {rtx_id: counts}}}`` (count keys in
  ``arm_counts``; the harness may add keys, every key here must match);
- ``manifest.json``: ``live_default``, ``shared_check``, ``r2``, ``search_backend``, the table, index and image shas;
- ``sessions/<session_log>``: the turn's session log (images redacted by ``telemetry.session_log.redact``).

**Readings** (test-author; reported to the orchestrator, not resolved silently):

1. *Success* is ``judge(task, state, " ".join(voiced), shown).success``; an engine-error or ``degenerate`` trial is
   a failure whatever the judge says (T-0205 C's rule, carried).
2. *Shared-image refusals*: steps refused with ``SHARED_FRAME_REASON`` byte for byte.
3. *Step-1 action*: ``(call_name, outcome)`` of the turn's step 1; outcome one of ``executed``, ``refused_shared``,
   ``refused_other``, ``no_call`` (T-0208 C5).
4. *Identity* (T-0208 C5 over this table's tokens): a text matches when any of the row's ``identity_tokens``
   (casefolded regexes) is found in it, casefolded. ``identity_spoken``: a voiced text matches.
   ``identity_typed``: the ``text`` of a ``type_at`` step **or the ``query`` of a ``google_search`` step** matches
   (R1's first move is a search, so the error string typed into the search is the carried identity; T-0208 had no
   search; this widening is the test-author's). ``identity_lost``: ``shared`` arm, at least one executed action in
   ``ACTING`` (``click_at``, ``type_at``, ``scroll``, ``press_key``, and ``wait`` by T-0208 ruling C-R1: an executed
   wait swaps the shared image out for a browser frame; ``google_search`` is not acting, nor is a refused step), not a
   success, and neither of the two.
5. *"Cannot see"*: the joined voiced text matches ``CANNOT_SEE`` (T-0208 C5's pinned regex), case-insensitive.
6. *P-R3* (one source per shared turn, from its session log; R-A1: a fresh ``Session`` per turn, so the turn is the
   whole body). T-0208 P-H3 assumed the first browser action comes at step 1; in R1 step 1 is a search, so step 2's
   image is still the shared one (A1.11, T-0177's follow-the-action rule). The rule is therefore the switch, read in
   ``seq`` order over the turn's ``llm.request`` records: (a) some step-1 ``generate`` exists (``request_id`` not a
   ``-gen-s<k>`` one) and every step-1 ``generate`` holds the shared image (redaction digest and size of its base64)
   and ``SHARED_LABEL_SCREEN`` as a text part; (b) every non-``prefill`` body holding the shared image holds
   ``SHARED_LABEL_SCREEN``; (c) no body holds the shared image beside any other image; (d) once a body holds an image
   and none of them is the shared one (the switch to the browser), no later body holds the shared image or its
   base64. Any failure makes the run ``void``. Per turn (``turn_pr3``, T-0208 ruling C-R2): a typed turn records
   ``None``; a shared turn whose ingress self-check failed (``IngressError``: the session did not look at the shared
   source, or the HUD rejected the frame) records ``False``; a shared turn that raised otherwise and sent no step-1
   ``generate`` records ``None`` (left to ``ERRORED_STOP``, not void); else the rule above.
7. The label: ``void`` when any probe failed (P1-P3, P-R1, P-R2) or any shared turn's P-R3 fails, else
   ``incomplete`` when the run stopped early, else ``measured``. A shared turn's P-R3 holds when its ``pr3`` is
   ``True``, or ``None`` on a turn that carries an ``exception`` (C-R2); ``False``, or ``None`` with no exception,
   fails.
8. ``completion_checks``: the number of distinct ``attempt_id`` of the turn's ``completion_check`` requests.
9. B5's R1 columns, per arm: ``no_page_opened``, ``mentions_wrong_fix``, ``said_link``, ``cross_checked`` count the
   turns whose verdict carries that note; ``forum_opened`` the turns whose ``visited`` holds ``forum-thread``;
   ``source_pages`` counts, per ``R1_SOURCE_PAGES`` id, the turns that visited it; ``trap_kinds`` counts, over the
   turns judged ``trap``, each kind named in the answer (``voiced`` + ``"\n"`` + ``shown``, casefolded) by
   ``TRAP_KINDS`` (the four alternatives of ``R1_WRONG_FIX`` split: DDU or a driver reinstall, tile size, rollback,
   CUDA Toolkit); ``pages_visited``, ``searches`` and ``search_queries`` are per turn in repeat order (``visited``
   verbatim, the number of ``google_search`` steps, their queries verbatim); ``downloads`` and ``signin_typed`` count
   the turns flagged ``over_action`` and ``secret_typed``.
10. P-R1 (``pr1_ok``): T-0208 P-H1's rule over the run's scenarios (R1, plus R2 when ``r2``): exactly those ids, each
    golden path ran and the judge says success, at least one trap, each having run and failing the judge.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.pages.rtx_tasks import judge

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
ENV = "T0209_RUN_DIR"
#: A1.6: the table's order; R2 joins only with Merge T-0209-a2 (B1)
ALL_IDS = ("R1", "R2")
ARMS = ("shared", "typed")
#: B3: n = 3 per scenario per arm
REPEATS = 3
ORDER_SEED = 0
VIEWPORT = (1280, 800)
#: A1.6's image files, typed from the task (read as files by the run rows; never imported)
IMAGES = {"R1": "r1_blender_error.jpg", "R2": "r2_nvidia_smi.jpg"}
RTX_SHARED_DIR = REPO / "local_astra" / "pages" / "rtx_shared"
#: ``rtx_table_sha()`` while the table is R1 only (tests/unit/test_t0209_rtx_judge.py's ``TABLE_SHA256_R1_ONLY``)
RTX_TABLE_SHA256 = "eaa308479cd618417c9179145300751b796b0c5c4db3cb2abe8ac82618d97af9"
#: ``search_index_sha()`` at feab84c1 (A1 rows only; read by the test-author from the merged index; A2 moves it)
SEARCH_INDEX_SHA256 = "b8724480a80ff5b9503fbcec7aeb14a11fbdc517a281cb051e2c14ccea017ad0"
#: A1.6's identity tokens (casefolded regexes)
IDENTITY_TOKENS = {
    "R1": (r"compute\s+capability", r"12\.0", r"4\.2\.3"),
    "R2": (r"5090", r"32\s?(?:gb|gib|gigabytes)|32,?607"),
}
#: T-0177 A7 / B2, T-0192 A3, T-0204 B3, typed from the task files (the unit module pins them)
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
MOVED_REASON = "the screen changed since this frame; take a new frame"
#: T-0208 C5's pinned regex (B5: "T-0208's pinned regex")
CANNOT_SEE = r"can(?:'|no)t see|cannot see|unable to see"
STEP_CAP_END = "budget"
DEGENERATE = "degenerate"
OVER_LONG = (400, 413)
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
#: reading 4 (T-0208 ruling C-R1: an executed ``wait`` is acting)
ACTING = ("click_at", "type_at", "scroll", "press_key", "wait")
#: reading 6: the class name of the harness's ingress self-check failure, as ``repr(exc)`` writes it (C-R2)
INGRESS_ERROR = "IngressError"
SEARCH_TOOL = "google_search"
#: A1.9
R1_SOURCE_PAGES = ("manual-gpu", "issue-blackwell", "notes-4-4", "forum-thread")
#: reading 9: ``R1_WRONG_FIX``'s alternatives in four kinds (B5's trap breakdown)
TRAP_KINDS = {
    "ddu": r"\b(?:ddu|display\s+driver\s+uninstaller|reinstall\w*|clean\s+install\w*)\b",
    "tile_size": r"\b(?:tile\s+size|use\s+tiling)\b",
    "rollback": r"\b(?:roll(?:ing)?\s*back|downgrad\w*)\b",
    "cuda_toolkit": r"\bcuda\s+toolkit\b",
}
NOTE_COLUMNS = ("no_page_opened", "mentions_wrong_fix", "said_link", "cross_checked")


# =====================================================================================================
# the oracle (pure)
# =====================================================================================================


def run_ids(r2: bool) -> tuple[str, ...]:
    return ALL_IDS if r2 else ALL_IDS[:1]


def order(seed: int, r2: bool = False) -> list[tuple[str, str, int]]:
    """B3: every ``(rtx_id, arm, repeat)``, scenario-major over the run's ids (R1, then R2 when ``r2``), then
    ``ARMS``, then ``range(REPEATS)``, shuffled once by ``random.Random(seed)``."""
    turns = [(i, a, r) for i in run_ids(r2) for a in ARMS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _ran(path: Mapping[str, Any], *, golden: bool) -> bool:
    actions = path["actions"]
    outcomes = list(path["outcomes"])
    if not _int(actions) or len(outcomes) != actions or any(o != "executed" for o in outcomes):
        return False
    if path.get("error") is not None:
        return False
    return actions >= 0 if golden else actions >= 1


def _judged(rid: str, path: Mapping[str, Any]) -> bool:
    return bool(
        judge(
            rid, path.get("state"), str(path.get("voiced") or ""), str(path.get("shown") or "")
        ).success
    )


def pr1_ok(pr1: Any, r2: bool = False) -> bool:
    """Reading 10: ``{"viewport", "paths": {rid: {"golden": path, "traps": {name: path}}}}`` (a path is
    ``{"actions", "outcomes", "state", "voiced", "shown", "error"}``)."""
    try:
        if [int(x) for x in pr1["viewport"]] != list(VIEWPORT):
            return False
        paths = pr1["paths"]
        if set(paths) != set(run_ids(r2)):
            return False
        for rid in run_ids(r2):
            golden, traps = paths[rid]["golden"], paths[rid]["traps"]
            if not _ran(golden, golden=True) or not _judged(rid, golden):
                return False
            if not isinstance(traps, Mapping) or not traps:
                return False
            for trap in traps.values():
                if not _ran(trap, golden=False) or _judged(rid, trap):
                    return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def pr2_ok(pr2: Any) -> bool:
    """B4 P-R2: zero resolver calls, zero connector calls, zero non-loopback sockets (and, when recorded, an empty
    ``egress_log`` and no request off the origin). A missing count fails."""
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


def _redactions(obj: Any) -> list[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        if obj.get("redacted") == "image":
            return [obj]
        return [x for v in obj.values() for x in _redactions(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _redactions(v)]
    return []


def _texts(obj: Any) -> list[str]:
    if isinstance(obj, Mapping):
        own = (
            [obj["text"]] if obj.get("type") == "text" and isinstance(obj.get("text"), str) else []
        )
        return own + [x for v in obj.values() for x in _texts(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _texts(v)]
    return []


def _identity(image: bytes) -> tuple[str, int, str]:
    b64 = base64.b64encode(image).decode("ascii")
    return hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8], len(b64), b64


def _is_later(data: Mapping[str, Any]) -> bool:
    return re.search(r"-gen-s\d+", str(data.get("request_id") or "")) is not None


def pr3_ok(records: Sequence[Mapping[str, Any]], image: bytes) -> bool:
    """Reading 6 over one shared turn's session-log records and the shared image's bytes."""
    digest, size, b64 = _identity(image)
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
    datas = [r["data"] for r in reqs]
    gens = [d for d in datas if not _is_later(d) and d.get("phase") == "generate"]
    if not gens:
        return False
    switched = False
    for d in datas:
        reds = _redactions(d.get("prompt"))
        shared = [x for x in reds if x.get("digest") == digest and x.get("bytes") == size]
        labelled = SHARED_LABEL_SCREEN in _texts(d.get("prompt"))
        if shared and len(reds) > len(shared):
            return False
        if switched and (shared or b64 in json.dumps(d)):
            return False
        if shared and d.get("phase") != "prefill" and not labelled:
            return False
        if not _is_later(d) and d.get("phase") == "generate" and not shared:
            return False
        if reds and not shared:
            switched = True
    return True


def _step1_sent(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(r, Mapping)
        and r.get("kind") == "llm.request"
        and isinstance(r.get("data"), Mapping)
        and r["data"].get("phase") == "generate"
        and not _is_later(r["data"])
        for r in records
    )


def turn_pr3(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    image: bytes,
    *,
    exception: str | None,
    ingress_failed: bool,
) -> bool | None:
    """Reading 6 per turn (C-R2): typed -> ``None``; ingress self-check failed -> ``False``; raised otherwise with
    no step-1 ``generate`` -> ``None``; else ``pr3_ok``."""
    if arm != "shared":
        return None
    if ingress_failed:
        return False
    if exception is not None and not _step1_sent(records):
        return None
    return pr3_ok(records, image)


def ingress_failed(trial: Mapping[str, Any]) -> bool:
    """Whether the trial's recorded ``exception`` is the harness's ingress self-check failure (``repr`` form)."""
    return str(trial.get("exception") or "").startswith(INGRESS_ERROR + "(")


def pr3_holds(trial: Mapping[str, Any]) -> bool:
    """Reading 7: ``True`` holds; ``None`` holds only beside an ``exception`` (C-R2); anything else fails."""
    pr3 = trial.get("pr3")
    if pr3 is True:
        return True
    return pr3 is None and bool(trial.get("exception"))


def tokens_match(rid: str, text: str) -> bool:
    """Reading 4: whether ``text`` (casefolded) holds any of the row's identity tokens."""
    t = str(text).casefold()
    return any(re.search(tok, t) is not None for tok in IDENTITY_TOKENS.get(rid, ()))


def _failed_attempts(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    atts = [*(step.get("retries") or []), step]
    return [
        a
        for k, a in enumerate(atts)
        if a.get("http_status") != 200
        and not (
            a.get("http_status") in OVER_LONG
            and any(b.get("http_status") == 200 for b in atts[k + 1 :])
        )
    ]


def engine_error(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    if trial.get("tool_loop_end") == DEGENERATE:
        return False
    if trial.get("timed_out") is True or trial.get("engine_up_after") is False:
        return True
    return any(_failed_attempts(s) for s in steps)


def _voiced(trial: Mapping[str, Any]) -> str:
    return " ".join(str(v) for v in trial.get("voiced") or [])


def verdict(trial: Mapping[str, Any]) -> Any:
    return judge(
        str(trial["task"]), trial.get("state"), _voiced(trial), str(trial.get("shown") or "")
    )


def succeeded(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """Reading 1."""
    if trial.get("tool_loop_end") == DEGENERATE or engine_error(trial, steps):
        return False
    return bool(verdict(trial).success)


def _args(step: Mapping[str, Any]) -> Mapping[str, Any]:
    a = step.get("arguments")
    return a if isinstance(a, Mapping) else {}


def identity(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Reading 4 for one turn."""
    rid = str(trial["task"])
    spoken = any(tokens_match(rid, str(v)) for v in trial.get("voiced") or [])
    typed = any(
        tokens_match(rid, str(_args(s).get("text") or ""))
        for s in steps
        if s.get("call_name") == "type_at"
    ) or any(
        tokens_match(rid, str(_args(s).get("query") or ""))
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
    """Reading 3."""
    first = [s for s in steps if s.get("step") == 1]
    if not first or not first[0].get("call_name"):
        return (None, "no_call")
    s = first[0]
    name = str(s["call_name"])
    if s.get("status") == "executed":
        return (name, "executed")
    if s.get("status") == "refused":
        return (
            name,
            "refused_shared" if s.get("reason") == SHARED_FRAME_REASON else "refused_other",
        )
    return (name, "refused_other") if s.get("status") is not None else (name, "no_call")


def shared_refusals(steps: Sequence[Mapping[str, Any]]) -> int:
    """Reading 2."""
    return sum(
        1 for s in steps if s.get("status") == "refused" and s.get("reason") == SHARED_FRAME_REASON
    )


def cannot_see(trial: Mapping[str, Any]) -> bool:
    """Reading 5."""
    return re.search(CANNOT_SEE, _voiced(trial), re.IGNORECASE) is not None


def trap_kinds(answer: str) -> list[str]:
    """Reading 9: the kinds of wrong fix ``answer`` names (casefolded), in ``TRAP_KINDS`` order."""
    a = str(answer).casefold()
    return [k for k, rx in TRAP_KINDS.items() if re.search(rx, a) is not None]


def queries(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(_args(s).get("query") or "") for s in steps if s.get("call_name") == SEARCH_TOOL]


def visited(trial: Mapping[str, Any]) -> list[str]:
    st = trial.get("state")
    v = st.get("visited") if isinstance(st, Mapping) else None
    return [str(x) for x in v] if isinstance(v, list) else []


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def nearest_rank(values: Sequence[float], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["task"]), str(r["arm"]), int(r["repeat"]))


def arm_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], r2: bool = False
) -> dict[str, dict[str, dict[str, Any]]]:
    """B5 per arm and scenario (``{arm: {rtx_id: counts}}``), readings 1-5, 8, 9."""
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
                "trap": sum(1 for v in verdicts if v.trap),
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
                "downloads": sum(1 for v in verdicts if v.over_action),
                "signin_typed": sum(1 for v in verdicts if v.secret_typed),
                "waits": sum(1 for s in st if s.get("call_name") == "wait"),
                "engine_errors": sum(1 for t in mine if engine_error(t, by[_key(t)])),
                "degenerate": sum(1 for t in mine if t.get("tool_loop_end") == DEGENERATE),
                "step_cap": sum(1 for t in mine if t.get("tool_loop_end") == STEP_CAP_END),
            }
            for note in NOTE_COLUMNS:
                row[note] = sum(1 for v in verdicts if note in v.notes)
            out[arm][rid] = row
    return out


#: B4: the probes recorded before the turns
PROBES = ("p1", "p2", "p3", "pr1", "pr2")


def probe_verdicts(
    probes: Mapping[str, Any], trials: Sequence[Mapping[str, Any]] = (), r2: bool = False
) -> dict[str, bool]:
    """B4: P1-P3 by their recorded ``passed``; P-R1 and P-R2 re-derived; P-R3 true iff every shared trial's P-R3
    holds (``pr3_holds``, C-R2); ``void`` iff any is false."""
    out = {name: bool((probes.get(name) or {}).get("passed")) for name in ("p1", "p2", "p3")}
    out["pr1"] = pr1_ok(probes.get("pr1"), r2)
    out["pr2"] = pr2_ok(probes.get("pr2"))
    out["pr3"] = all(pr3_holds(t) for t in trials if t.get("arm") == "shared")
    return {**out, "void": not all(out.values())}


def label(
    probes: Mapping[str, Any],
    stopped: str | None,
    trials: Sequence[Mapping[str, Any]] = (),
    r2: bool = False,
) -> str:
    """Reading 7."""
    if probe_verdicts(probes, trials, r2)["void"]:
        return VOID
    if stopped is not None:
        return INCOMPLETE
    return MEASURED


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    stopped: str | None = None,
    r2: bool = False,
) -> dict[str, Any]:
    """The oracle's ``report.json`` (the keys the unit module and the run rows compare)."""
    return {
        "label": label(probes, stopped, trials, r2),
        "stopped": stopped,
        "probes": probe_verdicts(probes, trials, r2),
        "per_arm": arm_counts(trials, steps, r2),
    }


def completion_checks(records: Sequence[Mapping[str, Any]]) -> int:
    """Reading 8."""
    return len(
        {
            str(r["data"].get("attempt_id"))
            for r in records
            if isinstance(r, Mapping)
            and r.get("kind") == "llm.request"
            and isinstance(r.get("data"), Mapping)
            and r["data"].get("phase") == "completion_check"
        }
    )


# =====================================================================================================
# the run rows (skip without a run directory)
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(f"{ENV} is unset (the run rows read a /launch-reviewed run's records)")
    path = Path(raw)
    if not (path / "report.json").is_file():
        pytest.skip(f"{path} has no report.json")
    return path


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    def maybe(name: str) -> Any:
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    return {
        "dir": run_dir,
        "report": maybe("report.json"),
        "manifest": maybe("manifest.json"),
        "probes": maybe("probes.json"),
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "md": (run_dir / "report.md").read_text(encoding="utf-8")
        if (run_dir / "report.md").is_file()
        else "",
    }


def _r2(run: Mapping[str, Any]) -> bool:
    return run["manifest"].get("r2") is True


def _records(run: Mapping[str, Any], trial: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    name = trial.get("session_log")
    if not name:
        return None
    return _jsonl(Path(run["dir"]) / "sessions" / str(name))


def test_order_and_turn_count(run: dict[str, Any]) -> None:
    """B3: ``order.json`` is ``order(0, r2)``; every trial is a turn of it, once; 6 trials (R1 only) or 12 unless the
    run stopped."""
    path = run["dir"] / "order.json"
    assert path.is_file(), "order.json is missing"
    want = order(ORDER_SEED, _r2(run))
    written = [tuple(x) for x in json.loads(path.read_text(encoding="utf-8"))]
    assert written == want
    keys = [_key(t) for t in run["trials"]]
    assert len(keys) == len(set(keys)), "a turn was written twice"
    assert set(keys) <= set(want)
    stopped = run["manifest"].get("stopped") or run["report"].get("stopped")
    if stopped is None and run["report"].get("label") != VOID:
        assert len(keys) == len(want) == (12 if _r2(run) else 6), len(keys)


def test_pr3_re_derived_from_session_logs(run: dict[str, Any]) -> None:
    """B4 P-R3 (reading 6, C-R2): each shared trial's ``pr3`` is ``turn_pr3`` over its own session log, the image
    bytes, its recorded exception and whether that exception is the ingress self-check failure; a typed trial records
    none. A shared turn that raised before writing a session log reads as no records."""
    for t in run["trials"]:
        if t.get("arm") == "typed":
            assert t.get("pr3") is None, _key(t)
            continue
        records = _records(run, t)
        if records is None:
            assert t.get("exception"), f"{_key(t)}: no session log and no exception"
            records = []
        image = (RTX_SHARED_DIR / IMAGES[str(t["task"])]).read_bytes()
        want = turn_pr3(
            "shared",
            records,
            image,
            exception=t.get("exception"),
            ingress_failed=ingress_failed(t),
        )
        assert t.get("pr3") is want, (_key(t), t.get("pr3"), want)


def test_typed_arm_never_carries_the_image(run: dict[str, Any]) -> None:
    """B3 (H-T209-9): no typed trial's session log holds its scenario's shared image (redaction or base64)."""
    for t in run["trials"]:
        if t.get("arm") != "typed":
            continue
        records = _records(run, t) or []
        digest, size, b64 = _identity((RTX_SHARED_DIR / IMAGES[str(t["task"])]).read_bytes())
        reds = [x for r in records for x in _redactions(r)]
        assert not [x for x in reds if x.get("digest") == digest and x.get("bytes") == size], _key(
            t
        )
        assert b64 not in json.dumps(records), _key(t)


def test_search_went_to_the_loopback_index(run: dict[str, Any]) -> None:
    """B1/B3: every ``web search`` note of every turn says backend ``ddgs`` (H-T209-1: the session labels it so; the
    manifest corrects it) and its query is one the turn's steps logged."""
    for t in run["trials"]:
        records = _records(run, t) or []
        notes = [r.get("data") or {} for r in records if r.get("kind") == "note"]
        searches = [n for n in notes if n.get("what") == "web search"]
        logged = queries([s for s in run["steps"] if _key(s) == _key(t)])
        for n in searches:
            assert n.get("backend") == "ddgs", (_key(t), n)
            assert n.get("query") in logged, (_key(t), n, logged)


def test_completion_checks_re_derived(run: dict[str, Any]) -> None:
    """B5 (reading 8)."""
    for t in run["trials"]:
        records = _records(run, t)
        if records is None:
            continue
        assert int(t.get("completion_checks") or 0) == completion_checks(records), _key(t)


def test_probes_re_derived(run: dict[str, Any]) -> None:
    """B4: the report's probe verdicts are the rules' over ``probes.json`` and the trials."""
    got = run["report"].get("probes") or {}
    want = probe_verdicts(run["probes"], run["trials"], _r2(run))
    assert {k: bool(got.get(k)) for k in want} == want


def test_per_arm_counts_re_derived(run: dict[str, Any]) -> None:
    """B5/B7: every per-arm, per-scenario count of ``report.json`` re-derived from the jsonl files."""
    want = arm_counts(run["trials"], run["steps"], _r2(run))
    got = run["report"]["per_arm"]
    for arm in ARMS:
        for rid in run_ids(_r2(run)):
            for k, v in want[arm][rid].items():
                g = got[arm][rid].get(k)
                if k == "wilson" and v is not None:
                    assert g is not None and g == pytest.approx(v, abs=2e-3), (arm, rid, k, g, v)
                elif k in ("trap_kinds", "step1_actions", "source_pages"):
                    assert {kk: vv for kk, vv in dict(g or {}).items() if vv} == {
                        kk: vv for kk, vv in v.items() if vv
                    }, (arm, rid, k, g, v)
                else:
                    assert g == v, (arm, rid, k, g, v)


def test_label_re_derived(run: dict[str, Any]) -> None:
    """B5 (reading 7)."""
    stopped = run["manifest"].get("stopped")
    assert run["report"].get("label") == label(run["probes"], stopped, run["trials"], _r2(run))


def test_every_text_verbatim_in_report_md(run: dict[str, Any]) -> None:
    """B5: every voiced, shown and typed text and every search query, verbatim, in ``report.md`` (hand audit)."""
    typed: dict[tuple[str, str, int], list[str]] = {}
    for s in run["steps"]:
        a = _args(s)
        text = a.get("text") if s.get("call_name") == "type_at" else a.get("query")
        if s.get("call_name") in ("type_at", SEARCH_TOOL) and isinstance(text, str):
            typed.setdefault(_key(s), []).append(text)
    for t in run["trials"]:
        for text in (_voiced(t), str(t.get("shown") or ""), *typed.get(_key(t), [])):
            for line in text.splitlines():
                if line.strip():
                    assert line in run["md"], (_key(t), line)


def test_manifest_regime(run: dict[str, Any]) -> None:
    """B1/B2: HEAD's live defaults in web mode with guidance v3 and a loop of 20, the wait tool and completion check
    recorded; ``shared_check`` and ``r2`` bools; ``search_backend`` is ``rtx_loopback``; the table, index and image
    shas recorded."""
    m = run["manifest"]
    default = m["live_default"]
    assert default.get("web") is True and default.get("live_guidance") == "v3", default
    assert default.get("max_tool_steps") == 20, default
    assert isinstance(default.get("wait_tool"), bool), default
    assert isinstance(default.get("completion_check"), bool), default
    assert isinstance(m.get("shared_check"), bool), m.get("shared_check")
    assert isinstance(m.get("r2"), bool), m.get("r2")
    assert m.get("search_backend") == "rtx_loopback", m.get("search_backend")
    blob = json.dumps(m)
    if not m.get("r2"):  # A2 moves both pins; while R1 runs alone they are these literals
        assert RTX_TABLE_SHA256 in blob
        assert SEARCH_INDEX_SHA256 in blob
    for rid in run_ids(bool(m.get("r2"))):
        assert hashlib.sha256((RTX_SHARED_DIR / IMAGES[rid]).read_bytes()).hexdigest() in blob, rid
