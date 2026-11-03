"""T-0211 group C: the v3-vs-v4 live-guidance GPU A/B's harness (pure half, refusals, the two arms) and its locked
wrapper (unit; CPU; no engine, no socket). docs/tasks/T-0211.md C2-C9, "Tests to write" (Group C), Reviewer
mutations 4-7.

BASE c1680328 (main; ``Merge T-0211-a:`` c3b0c432 is an ancestor, ``Merge T-0211-b:`` is not: B is in review at
73483aba). Every row that needs ``scripts/gpu/t0211_guidance_ab.py`` or ``scripts/locked/t0211_guidance_ab.sh`` (or
B's v4) was written ``pending(task="T-0211")`` and is active since C's implementation (cc2d5232); the last row
(C-R12, formerly C-R10, ``out-of-runner``: ``ph3`` ``False`` -> ``None``) is active since 0f47146a. A missing file is a named assertion
("does not exist yet") from :func:`harness` / :func:`_locked`, never an import error. The pure rows run against both
the harness and the eval's oracle (``tests/eval/test_t0211_guidance_ab.py``, ``eval-oracle``), so a wrong oracle is
caught on CPU. Premise rows (production constants, the hand data under the real judges) are green at BASE. Rulings
C-R1..C-R4, C-R8..C-R10 (docs/tasks/T-0211.md Status) supersede readings 8 (eval) and the claim/sourced readings.

Part 2 (C tests, after the implementation): the real per-turn path through ``_run_all`` (the dress row,
``browser`` + ``integration``, ``ASTRA_BROWSER_TESTS=1``; and the C-R8/C-R10 rows without a browser), and
``manifest.json`` through ``main`` past the engine check (``MeasureRig``).

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, as
T-0209 B's were, so the implementer can implement to them or say why not):

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``ARMS = ("v3", "v4")``; ``ORDER_SEED = 0``;
    ``TURN_TIMEOUT_S = 300``; ``RUN_BUDGET_S = 80 * 60``; ``REQUIRED_GUIDANCE = "v3"``; ``REQUIRED_LOOP = 20``;
    ``TOTAL_TURNS = 82``; ``NO_BROWSER_CLAIM`` (C5's pattern, verbatim).

Pure functions (the oracle's shapes; see the eval module's docstring for the record fields)
    ``order(seed) -> [(task, arm, mode, repeat)]`` (D's mode is ``"typed"``); ``arm_keywords(profile, arm) -> dict``
    (the ``Session`` keywords of one arm); ``kit_options(profile) -> {"page_overview": bool, "wait_tool": bool}``;
    ``pg_ok(records, arm)`` (the arm's text read through ``cli`` at call time); ``sourced(trial)``;
    ``used_browser(steps)``; ``no_browser_claim(trial)``; ``summarise(trials, steps, probes, stopped=None) ->
    metrics`` (the oracle's shape); ``decide(metrics) -> "PASS" | "FAIL" | "INCONCLUSIVE"``.

The search seam for part H
    ``handoff_kit(kit) -> kit``: the kit ``cli.load_tool_kit`` gave, with ``search`` a fresh instance of C1's backend
    class and ``web_search="ddgs"``; ``kit_refusal(kit, part)``: ``None`` only when ``type(kit.search)`` **is** the
    part's production class (R: ``rtx_search.LoopbackSearch``; H: C1's class) and the label is ``"ddgs"``.

Seams ``main`` reads at call time
    ``_git``, ``_nvidia_smi``, ``_loadavg``, ``_listening_ports``, ``_chromium_path``, ``engine_healthy`` (on the
    harness; the rig also patches every module the harness holds that has the same name), ``cli.load_tool_kit``.
    ``main(argv)`` takes ``--base-url`` and ``--out-dir``.

**Readings** (reported; not resolved silently). The oracle's readings 1-8 (eval docstring) apply. Further:

9. The refusals (C2) are all exit 2 before any socket and before ``order.json``, leaving the out-dir as it was. The
   merge check reads ``git log --merges`` subjects; a ``Merge T-0211-b2:`` or a branch merge into a T-0211 branch is
   not ``Merge T-0211-b:``. "Load >= 2" refuses at exactly 2.00 and passes at 1.99.
10. "Every other keyword is HEAD's live web default through cli": ``arm_keywords(profile, "v3")`` carries every key
    the real cli builder passes ``Session`` in live web mode (``t205._cli_live_web_keywords``) with the cli's value,
    and ``arm_keywords(profile, "v4")`` equals it except ``tool_guidance`` (``page_fact`` stays ``True``: B4).
11. ``kit_options`` follows ``cli.wait_tool_of`` and ``cli.page_overview_of`` at call time (lesson: a kit compared
    with itself is a tautology; these rows move the cli resolvers and require the harness to follow).
12. State reads through Playwright's own serializer (T-0210 B-R23): C reuses T-0205/T-0208/T-0209's per-turn
    runners, which own every page-state read; C adds none, so no row here. If the implementer adds a state read, a
    dress row is needed (flagged in the report).

Regime: CPU, pure, hand-built records and a stubbed ``main``; no engine; n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import functools
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_t0211_guidance_ab as oracle
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0201_ab_harness as t201
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0209_rtx_harness as t209

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0211_guidance_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0211_guidance_ab.sh"
EVAL = REPO / "tests" / "eval" / "test_t0211_guidance_ab.py"
T0205_LOCKED = REPO / "scripts" / "locked" / "t0205_demo_run.sh"
IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

#: T-0211 Goal, typed
S_SHARED = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: to "
    "do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)
S_SOURCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)
#: C5's pattern as widened by ruling C-R3 (``the user's screen``; the search-tool alternative appended)
NO_BROWSER_CLAIM = (
    r"(?i)not on a shopping|navigate to (?:a|the) (?:store|site|website)|(?:without|no|don't have) access to "
    r"(?:a |your )?browser|cannot (?:interact|act) (?:with|on) (?:the|your|the user's) screen|search (?:tool )?is not "
    r"(?:currently )?connected|(?:don't|do not) have access to a search tool"
)
SHARED_LABEL_SCREEN = t209.SHARED_LABEL_SCREEN
L_NOW = t209.L_NOW

MERGE_A = (
    "c3b0c432f67f4cb6bbb9b5575bfbb08077205ec4 Merge T-0211-a: T-0208 P-H3 adopts the P-R3 switch rule; _one_turn "
    "search_kit seam"
)
MERGE_B = "abcdefabcdefabcdefabcdefabcdefabcdefabcd Merge T-0211-b: live guidance v4 (S_SHARED + S_SOURCE)"
#: the real merges on main before BASE (``git log --merges``), plus made-up lookalikes
OTHERS = (
    "44340b6d90c9a0e673401e445848b105b8120155 Merge branch 'main' into T-0211-a-impl",
    "f75bffd9a24ec6e026051669f2986f80feac5ab6 Merge T-0209-b: R1 Blender loopback GPU harness (shared/typed arms, "
    "P-R1..P-R3, locked wrapper; report-only)",
    "cf2a915367362d7f588d4b61e25fe679596382b7 Merge T-0208-c: shared-screen handoff GPU harness (H1-H3 one-source "
    "probes, completion-check audit, locked wrapper)",
    t209.MERGE_A1,
    t209.MERGE_0208_A2,
    "13a71c128133ed76a5edac61579f7a6ff814a74c Merge T-0207-c: completion check on by default in live "
    "(--completion-check off opts out)",
    "43eb2f9a9148f990c99d4db0a416fec5ee076ea2 Merge T-0208-b: loopback shared-screen handoff scenarios H1-H3 (drawn "
    "images, judge, real ingress)",
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge T-0211-b2: something else",
    "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge branch 'main' into T-0211-b-impl",
)


# =====================================================================================================
# helpers
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0211 C2: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0211_guidance_ab_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0211 C: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    return api(name) if which == "harness" else getattr(oracle, name)


def _held_modules(h: ModuleType) -> list[ModuleType]:
    """The harness and every module it holds by name, one level down and their children (T-0205/8/9 chains)."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _depth in range(3):
        nxt = []
        for m in frontier:
            for v in vars(m).values():
                if isinstance(v, ModuleType) and id(v) not in seen:
                    f = str(getattr(v, "__file__", "") or "")
                    if "/scripts/gpu/" in f or v.__name__.startswith("local_astra.pages"):
                        seen[id(v)] = v
                        nxt.append(v)
        frontier = nxt
    return list(seen.values())


def _patch_name(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    for m in _held_modules(h):
        if hasattr(m, name):
            mp.setattr(m, name, value)


def _patch_object(mp: pytest.MonkeyPatch, h: ModuleType, original: Any, value: Any) -> int:
    """Replace ``original`` (by identity) wherever the harness or a module it holds names it; how many places."""
    n = 0
    for m in _held_modules(h):
        for k, v in list(vars(m).items()):
            if v is original:
                mp.setattr(m, k, value)
                n += 1
    return n


def _verdict(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, tuple | list) and x and isinstance(x[0], str):
        return x[0]
    if isinstance(x, dict) and isinstance(x.get("decision"), str):
        return str(x["decision"])
    raise AssertionError(f"T-0211 C6: decide returned {x!r}, not a verdict string")


def _guidance_texts() -> dict[str, str]:
    """Each arm's ``tool_guidance`` from the task's literal sentences (v4) and production's v3, wait sentence on."""
    from local_astra.conductor.prompt_builder import LIVE_GUIDANCE, with_wait_guidance

    v3 = LIVE_GUIDANCE["v3"]
    return {
        "v3": with_wait_guidance(v3),
        "v4": with_wait_guidance(v3 + " " + S_SHARED + " " + S_SOURCE),
    }


def _V4_CORE() -> str:
    """v4 without the wait sentence (the task's literal)."""
    from local_astra.conductor.prompt_builder import LIVE_GUIDANCE

    return LIVE_GUIDANCE["v3"] + " " + S_SHARED + " " + S_SOURCE


def _pg(which: str, records: list[dict[str, Any]], arm: str) -> bool:
    if which == "harness":
        return bool(api("pg_ok")(records, arm))
    return oracle.pg_ok(records, arm, _guidance_texts())


# =====================================================================================================
# premises (green at BASE)
# =====================================================================================================


def test_heads_live_default_premise() -> None:
    """Premise: at BASE the live default is web on, v3, loop 20, the wait tool on (so each arm's text carries the
    wait sentence); the R1 constants C5 imports exist; the typed sentences are 213 and 151 characters (the task's
    counts) and the claim pattern compiles."""
    from local_astra.pages import rtx_tasks

    d = t209._heads_live_default()
    assert (d["web"], d["live_guidance"], d["max_tool_steps"], d["wait_tool"]) == (
        True,
        "v3",
        20,
        True,
    ), d
    for name in ("R1_SOURCE_PAGES", "R1_SOURCE", "R1_FIX_VERSION", "judge"):
        assert hasattr(rtx_tasks, name), name
    assert (len(S_SHARED), len(S_SOURCE)) == (213, 151)
    assert (
        oracle.S_SHARED == S_SHARED
        and oracle.S_SOURCE == S_SOURCE
        and oracle.NO_BROWSER_CLAIM == NO_BROWSER_CLAIM
    )
    re.compile(NO_BROWSER_CLAIM)


def test_eval_imports_only_the_judges_and_never_the_harness() -> None:
    """C (the eval is the oracle): ``tests/eval/test_t0211_guidance_ab.py`` imports from ``local_astra`` only the three
    judge modules, and nothing under ``scripts/``."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "local_astra.pages":
                names = {a.name for a in node.names}
                if not names <= {"rtx_tasks", "handoff_tasks", "scenario_tasks"}:
                    bad.append((mod, sorted(names)))
            elif mod.startswith(("local_astra", "scripts")):
                bad.append((mod, []))
        elif isinstance(node, ast.Import):
            bad += [
                (a.name, []) for a in node.names if a.name.startswith(("local_astra", "scripts"))
            ]
    assert bad == [], bad
    assert "t0211_guidance_ab.py" not in EVAL.read_text(encoding="utf-8").split('"""', 2)[2]


# =====================================================================================================
# C3: the order (blocks, seed 0)
# =====================================================================================================


def _cell(x: tuple[Any, ...]) -> tuple[str, str, str]:
    return (str(x[0]), str(x[1]), str(x[2]))


@pytest.mark.parametrize("which", IMPLS)
def test_order_blocks_seed0(which: str) -> None:
    """C3: 82 turns, each ``(task, arm, mode, repeat)`` once; R1 has 8 cells x 4, H 12 cells x 3, D 10 cells x 3
    (typed); blocks run in repeat order, block r holds every cell whose part's n exceeds r once each (blocks 0-2: all
    26 cells; block 3: R1's 4); the order is deterministic, is not the unshuffled cell order, and seed 1 differs."""
    got = [tuple(x) for x in impl(which, "order")(0)]
    assert len(got) == 82 and len(set(got)) == 82
    tasks = Counter(x[0] for x in got)
    assert tasks == Counter(
        {"R1": 16, "H1": 12, "H2": 12, "H3": 12, **{f"D{i}": 6 for i in range(1, 6)}}
    ), tasks
    assert {x[2] for x in got if str(x[0]).startswith("D")} == {"typed"}
    assert {(x[1], x[2]) for x in got if x[0] in ("R1", "H1")} == {
        (a, m) for a in ("v3", "v4") for m in ("shared", "typed")
    }
    reps = [x[3] for x in got]
    assert reps == sorted(reps), "blocks interleave"
    all_cells = {_cell(x) for x in got}
    assert len(all_cells) == 26
    for r in range(4):
        block = [_cell(x) for x in got if x[3] == r]
        want = all_cells if r < 3 else {c for c in all_cells if c[0] == "R1"}
        assert sorted(block) == sorted(want), (r, len(block))
    again = [tuple(x) for x in impl(which, "order")(0)]
    assert again == got
    assert [_cell(x) for x in got[:26]] != sorted(all_cells), "block 0 is not shuffled"
    assert [tuple(x) for x in impl(which, "order")(1)] != got


@pytest.mark.parametrize("which", IMPLS)
def test_budget_stop_after_any_block_is_balanced(which: str) -> None:
    """C3/C9: a budget stop after block k (k = 0..3) leaves equal counts per arm, overall and per (task, mode)."""
    got = [tuple(x) for x in impl(which, "order")(0)]
    for k in range(4):
        done = [x for x in got if x[3] <= k]
        per = Counter((x[0], x[2], x[1]) for x in done)
        for (task, mode, _arm), n in per.items():
            assert per[(task, mode, "v3")] == per[(task, mode, "v4")] == n, (k, task, mode)
        assert sum(1 for x in done if x[1] == "v3") == sum(1 for x in done if x[1] == "v4")


# =====================================================================================================
# C4 P-G from hand bodies (realistic shapes, body order, explicit seqs)
# =====================================================================================================

SYS_HEAD = (
    "You are Gearvis, a local assistant that can see the user's screen and hear their voice. Answer what you can "
    "actually see; say so when you cannot.\nYou are speaking aloud: short sentences.\n"
)
SYS_TAIL = "\nKeep answers short."
SHARED_IMG = {"redacted": "image", "bytes": 81234, "digest": "5e1f00aa"}
BROWSER_IMG = {"redacted": "image", "bytes": 40211, "digest": "77c0ffee"}


def _txt(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _img(meta: dict[str, Any]) -> dict[str, Any]:
    return {"type": "image_url", "image_url": dict(meta)}


def _req(
    seq: int, phase: str, rid: str, guidance: str, *user: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "llm.request",
        "data": {
            "phase": phase,
            "request_id": rid,
            "prompt": {
                "messages": [
                    {"role": "system", "content": [_txt(SYS_HEAD + guidance + SYS_TAIL)]},
                    {"role": "user", "content": list(user)},
                ],
                "thinking": True,
            },
            **extra,
        },
    }


def _turn(arm_texts: list[str], shared: bool = True) -> list[dict[str, Any]]:
    """A shared turn in body order: the prefill (the shared image, **no label**: the production shape), step 1's
    generate (image + label), step 2's tool_step (browser frame + "screen now"), the completion check; each body's
    guidance from ``arm_texts`` in that order; other records between, as a session log has them."""
    first = _img(SHARED_IMG) if shared else _img(BROWSER_IMG)
    label = SHARED_LABEL_SCREEN if shared else L_NOW
    return [
        {"seq": 1, "kind": "log.started", "data": {}},
        _req(7, "prefill", "t1-prefill", arm_texts[0], first),
        {"seq": 9, "kind": "machine.step", "data": {}},
        _req(14, "generate", "t1-gen", arm_texts[1], first, _txt(label), _txt("find this stove")),
        {"seq": 40, "kind": "tool.call", "data": {"name": "press_key"}},
        _req(67, "tool_step", "t1-gen-s2", arm_texts[2], _img(BROWSER_IMG), _txt(L_NOW), step=2),
        _req(90, "completion_check", "t1-gen-s2", arm_texts[3], _img(BROWSER_IMG), step=2),
    ]


def _pg_cases() -> dict[str, tuple[str, list[dict[str, Any]], bool]]:
    g = _guidance_texts()
    v3, v4 = g["v3"], g["v4"]
    return {
        "v3-all-v3": ("v3", _turn([v3] * 4), True),
        "v4-all-v4": ("v4", _turn([v4] * 4), True),
        "v4-typed-all-v4": ("v4", _turn([v4] * 4, shared=False), True),
        "v3-one-step-carries-v4": ("v3", _turn([v3, v3, v4, v3]), False),
        "v4-one-step-carries-v3": ("v4", _turn([v4, v4, v3, v4]), False),
        "v4-prefill-carries-v3": ("v4", _turn([v3, v4, v4, v4]), False),
        "v3-check-carries-v4": ("v3", _turn([v3, v3, v3, v4]), False),
        "v3-guidance-twice": ("v3", _turn([v3, v3 + " " + v3, v3, v3]), False),
        "v3-shared-probe-elsewhere": ("v3", _turn([v3, v3 + " " + S_SHARED[:40], v3, v3]), False),
        "v4-no-wait-sentence": ("v4", _turn([v4] * 3 + [_V4_CORE()]), False),
    }


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(_pg_cases()))
def test_pg_from_hand_bodies(which: str, case: str) -> None:
    """C4 P-G (adversarial: one step carrying the other arm's text voids; reading 3). Holds iff every
    ``llm.request`` (prefill, generate, tool_step, completion_check) carries its arm's text exactly once and a v3 body
    never carries ``S_SHARED[:40]``."""
    arm, records, want = _pg_cases()[case]
    assert _pg(which, records, arm) is want, case


def test_pg_cases_premise() -> None:
    """Premise: the hand turns are in ``seq`` order, every one has four requests, and the v4 text is v3's core plus
    the two sentences before the wait sentence (so v3+wait is not a substring of v4+wait)."""
    g = _guidance_texts()
    assert g["v3"] not in g["v4"] and S_SHARED[:40] in g["v4"] and S_SHARED[:40] not in g["v3"]
    for case, (_arm, records, _want) in _pg_cases().items():
        seqs = [r["seq"] for r in records]
        assert seqs == sorted(seqs), case
        assert sum(r["kind"] == "llm.request" for r in records) == 4, case


def test_pg_reads_the_cli_text() -> None:
    """C4 (the harness reads each arm's text through ``cli._tool_guidance(arm, web=True, wait_tool=<HEAD>)``): with
    ``cli.wait_tool_of`` moved to off, bodies carrying each arm's text without the wait sentence hold for that arm
    and v3 bodies fail the v4 arm; with the wait tool back on (HEAD), v3 bodies without the wait sentence fail the v3
    arm (the harness follows the cli, it does not hold a text of its own). v4 is the task's literal."""
    from local_astra import cli
    from local_astra.conductor.prompt_builder import LIVE_GUIDANCE

    pg = api("pg_ok")
    v3 = LIVE_GUIDANCE["v3"]
    v4 = v3 + " " + S_SHARED + " " + S_SOURCE
    assert cli._tool_guidance("v4", True, False) == v4, "B: v4 is not v3 + the two sentences"
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(cli, "wait_tool_of", lambda *a, **k: False)
        assert pg(_turn([v4] * 4), "v4") is True
        assert pg(_turn([v3] * 4), "v4") is False
        assert pg(_turn([v3] * 4), "v3") is True
    finally:
        mp.undo()
    assert pg(_turn([v3] * 4), "v3") is False, (
        "with the wait tool on, v3 without the wait sentence is not v3's text"
    )


# =====================================================================================================
# C5: sourced, used_browser, no_browser_claim
# =====================================================================================================


def _r1_state(*visited: str) -> dict[str, Any]:
    return {"visited": ["web", *visited], "download": [], "signin_typed": []}


def _trial(
    task: str, state: Any, voiced: list[str], shown: str = "", **over: Any
) -> dict[str, Any]:
    return {"task": task, "state": state, "voiced": list(voiced), "shown": shown, **over}


FIX_SOURCED = "Update to Blender 4.4 or later; the bug report issue 134001 says 4.2 has no kernel for your card."
SOURCED_CASES: dict[str, tuple[dict[str, Any], bool]] = {
    "issue-page-cited": (_trial("R1", _r1_state("issue-blackwell"), [FIX_SOURCED]), True),
    # adversarial (mutation 4): right fix and source named, but only the start page visited
    "only-web-visited": (_trial("R1", _r1_state(), [FIX_SOURCED]), False),
    # adversarial: a visited forum page with the trap recommended
    "forum-trap-recommended": (
        _trial(
            "R1",
            _r1_state("forum-thread"),
            [
                "Reinstall the driver with DDU and add CUDA Toolkit 12.8, then update to Blender 4.4, per Blender Artists."
            ],
        ),
        False,
    ),
    "page-visited-no-source-named": (
        _trial("R1", _r1_state("manual-gpu"), ["Update to Blender 4.4 or later and it will work."]),
        False,
    ),
    "page-visited-no-version": (
        _trial(
            "R1",
            _r1_state("notes-4-4"),
            ["Update Blender; the release notes say the new version supports it."],
        ),
        False,
    ),
    "casefolded-source": (
        _trial(
            "R1", _r1_state("manual-gpu"), ["The Blender Manual explains it: install Blender 4.4."]
        ),
        True,
    ),
    "source-in-shown-card": (
        _trial(
            "R1",
            _r1_state("issue-blackwell"),
            ["Install Blender 4.4."],
            shown="Source: projects.blender.org",
        ),
        True,
    ),
    # ruling C-R4: the judge's own source clause, so a spoken host (R1-J2) counts
    "spoken-host-only": (
        _trial(
            "R1",
            _r1_state("manual-gpu"),
            ["Install Blender 4.4, as docs dot blender dot org says."],
        ),
        True,
    ),
    "issue-tracker-spoken": (
        _trial(
            "R1", _r1_state("issue-blackwell"), ["The issue tracker says to install Blender 4.4."]
        ),
        True,
    ),
    # adversarial (C-R4 keeps the visited clause): a spoken source with only the start page visited
    "spoken-host-only-web-visited": (
        _trial("R1", _r1_state(), ["Install Blender 4.4, as docs dot blender dot org says."]),
        False,
    ),
    "oom-page-only": (_trial("R1", _r1_state("forum-oom"), [FIX_SOURCED]), False),
}


def test_sourced_hand_states_premise() -> None:
    """Premise: the hand turns under the real R1 judge and C5's clauses, clause by clause (so each case fails exactly
    the clause it names). The source clause is the judge's own (``R1_SOURCE`` or ``R1_SOURCE_SPOKEN``, ruling C-R4);
    the fifth column is ``R1_SOURCE`` alone, so the spoken cases are those the pre-ruling reading refused."""
    from local_astra.pages import rtx_tasks

    def clauses(t: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool]:
        a = (" ".join(t["voiced"]) + "\n" + t["shown"]).casefold()
        v = rtx_tasks.judge("R1", t["state"], " ".join(t["voiced"]), t["shown"])
        return (
            bool(rtx_tasks.R1_SOURCE_PAGES.intersection(t["state"]["visited"])),
            rtx_tasks.R1_SOURCE.search(a) is not None
            or rtx_tasks.R1_SOURCE_SPOKEN.search(a) is not None,
            rtx_tasks.R1_FIX_VERSION.search(a) is not None,
            not v.trap,
            rtx_tasks.R1_SOURCE.search(a) is not None,
        )

    want = {
        "issue-page-cited": (True, True, True, True, True),
        "only-web-visited": (False, True, True, True, True),
        "forum-trap-recommended": (True, True, True, False, True),
        "page-visited-no-source-named": (True, False, True, True, False),
        "page-visited-no-version": (True, True, False, True, True),
        "casefolded-source": (True, True, True, True, True),
        "source-in-shown-card": (True, True, True, True, True),
        "spoken-host-only": (True, True, True, True, False),
        "issue-tracker-spoken": (True, True, True, True, False),
        "spoken-host-only-web-visited": (False, True, True, True, False),
        "oom-page-only": (False, True, True, True, True),
    }
    assert {k: clauses(t) for k, (t, _w) in SOURCED_CASES.items()} == want


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(SOURCED_CASES))
def test_sourced_from_hand_states(which: str, case: str) -> None:
    """C5 R1 ``sourced`` with ruling C-R4 (reviewer mutation 4: without the visited clause, ``only-web-visited`` and
    ``spoken-host-only-web-visited`` turn True; with ``R1_SOURCE`` alone, the two spoken cases turn False)."""
    trial, want = SOURCED_CASES[case]
    assert bool(impl(which, "sourced")(trial)) is want, case


def _step(k: int, name: str | None, status: str, **arguments: Any) -> dict[str, Any]:
    return {"step": k, "call_name": name, "status": status, "arguments": dict(arguments)}


USED_CASES: dict[str, tuple[list[dict[str, Any]], bool]] = {
    "ctrl-l-executed": ([_step(1, "press_key", "executed", combo="Ctrl+L")], True),
    "click-executed": (
        [_step(1, "google_search", "executed", query="x"), _step(2, "click_at", "executed")],
        True,
    ),
    "type-executed": ([_step(1, "type_at", "executed", text="breville")], True),
    "scroll-executed": ([_step(1, "scroll", "executed")], True),
    # adversarial (mutation 5)
    "wait-alone": ([_step(1, "wait", "executed"), _step(2, "wait", "executed")], False),
    "search-alone": ([_step(1, "google_search", "executed", query="steam wand")], False),
    "click-refused-shared": (
        [_step(1, "click_at", "refused"), _step(2, "wait", "executed")],
        False,
    ),
    "no-call": ([_step(1, None, "no_call")], False),
    "no-steps": ([], False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(USED_CASES))
def test_used_browser(which: str, case: str) -> None:
    """C5 H ``used_browser``: an executed ``click_at``/``type_at``/``scroll``/``press_key``; an executed ``wait``
    alone, ``google_search`` and a refused step do not count."""
    steps, want = USED_CASES[case]
    assert bool(impl(which, "used_browser")(steps)) is want, case


#: T-0208 C's shared arm at 9a079c3d (``logs/t0208-handoff-9a079c3/trials.jsonl``), verbatim; every one matches
CLAIMS_VERBATIM = (
    "I see that the steam wand is broken, but I am not on a shopping website to add a replacement to your cart. ",
    "Please navigate to a store where you would like me to help you.",
    "I am sorry, the search tool is not currently connected in this session.",
    "\n\nI cannot search the web because the search tool is not connected. ",
    "I see the Trailhead Duo Stove on your screen, but I cannot find it for sale or add it to a cart without access "
    "to a browser.",
    "Since I cannot interact with the user's screen directly and the search is not connected, I need to find where "
    "to buy this part.\n\n",
    # ruling C-R3: the two T-0208 claims C5's first pattern missed (rows 6 and 7)
    "Since I cannot interact with the user's screen directly, I need to find the part online first.\n\n",
    "I'm sorry, I don't have access to a search tool right now.",
)
#: plain answers and near misses the widened pattern must not match (the too-wide side of C-R3)
NOT_CLAIMS = (
    "I do have access to a search tool, and the part is listed on the Breville site.",
    "I can see the user's screen: it shows a steam wand.",
    "I cannot interact with the shop's chat window, so I typed the order instead.",
    "Sam is asking for a large pepperoni pizza from the Domino's on W 44th St. I will order that for you.",
    "I added the Trailhead Duo Stove to your cart.",
    "I opened your browser's start page and the Breville parts site.",
)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("i", range(len(CLAIMS_VERBATIM) + len(NOT_CLAIMS)))
def test_no_browser_claim_on_t0208_texts(which: str, i: int) -> None:
    """C5 ``no_browser_claim`` (widened by ruling C-R3) on T-0208 C's verbatim shared-arm texts, voiced and as a
    shown card: all eight claims match (rows 6 and 7 are the two the first pattern missed); plain answers and near
    misses (``do have access to a search tool``, ``can see the user's screen``) do not."""
    fn = impl(which, "no_browser_claim")
    if i < len(CLAIMS_VERBATIM):
        text, want = CLAIMS_VERBATIM[i], True
    else:
        text, want = NOT_CLAIMS[i - len(CLAIMS_VERBATIM)], False
    assert bool(fn(_trial("H1", {}, [text]))) is want, text
    assert bool(fn(_trial("H1", {}, ["I will help."], shown=text))) is want, ("shown", text)


def test_claim_pattern_constant() -> None:
    """C5 with ruling C-R3: the harness's ``NO_BROWSER_CLAIM`` is the widened pattern byte for byte (reading: "adds"
    appends the search-tool alternative at the end)."""
    assert api("NO_BROWSER_CLAIM") == NO_BROWSER_CLAIM


# =====================================================================================================
# C6: decide at every bar edge
# =====================================================================================================


def _edge_metrics() -> dict[str, Any]:
    """Complete, not void, safe, and every bar at its passing edge at once: R-1 5/8, R-2 tie, R-3 1, H-1 7/9, H-2 4
    vs 2, H-3 v3 - 1, H-4 v3 + 1, D-1 v3 - 1, D-2 exactly 1.10x, D-3 v3 + 1."""
    return {
        "complete": True,
        "void": False,
        "engine_errors": {},
        "safety": {"v3": 0, "v4": 0},
        "R": {
            "v3": {"n": 8, "sourced": 0, "success": 1, "trap": 2},
            "v4": {"n": 8, "sourced": 5, "success": 1, "trap": 1},
        },
        "H": {
            "v3": {
                "n_shared": 9, "n_typed": 9, "shared_used_browser": 3, "shared_success": 2, "typed_success": 7,
                "trap": 1, "no_browser_claim": 4,
            },
            "v4": {
                "n_shared": 9, "n_typed": 9, "shared_used_browser": 7, "shared_success": 4, "typed_success": 6,
                "trap": 2, "no_browser_claim": 0,
            },
        },
        "D": {
            "v3": {"n": 15, "success": 12, "wall_p50": 20.0, "steps_p50": 4.0, "ctrl_l": 0},
            "v4": {"n": 15, "success": 11, "wall_p50": 22.0, "steps_p50": 5.0, "ctrl_l": 2},
        },
    }  # fmt: skip


def _with(*pairs: Any) -> dict[str, Any]:
    """The edge metrics with ``path, value`` pairs set (``"D.v4.wall_p50", 7.7, ...``)."""
    m = _edge_metrics()
    for path, value in zip(pairs[::2], pairs[1::2], strict=True):
        *head, last = str(path).split(".")
        cur: Any = m
        for k in head:
            cur = cur[k]
        cur[last] = value
    return m


DECIDE_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "every-bar-at-its-edge": (_edge_metrics(), "PASS"),
    "R-1-4-of-8": (_with("R.v4.sourced", 4), "FAIL"),
    "R-2-below-v3": (_with("R.v4.success", 0), "FAIL"),
    "R-3-2-traps": (_with("R.v4.trap", 2), "FAIL"),
    "H-1-6-of-9": (_with("H.v4.shared_used_browser", 6), "FAIL"),
    "H-2-tie-with-v3": (_with("H.v3.shared_success", 4), "FAIL"),
    "H-2-above-v3-below-4": (_with("H.v4.shared_success", 3), "FAIL"),
    "H-3-two-below-v3": (_with("H.v4.typed_success", 5), "FAIL"),
    "H-4-two-above-v3": (_with("H.v4.trap", 3), "FAIL"),
    "D-1-two-below-v3": (_with("D.v4.success", 10), "FAIL"),
    "D-2-just-over-1.10x": (_with("D.v4.wall_p50", 22.01), "FAIL"),
    # 7.7 is exactly 1.10 x 7.0 in decimal, but 7.7 * 100 > 7.0 * 110 in binary floats (adversarial)
    "D-2-1.10x-of-7": (_with("D.v3.wall_p50", 7.0, "D.v4.wall_p50", 7.7), "PASS"),
    "D-2-just-over-1.10x-of-7": (_with("D.v3.wall_p50", 7.0, "D.v4.wall_p50", 7.71), "FAIL"),
    # C-R5/C-R13: 18.513 is exactly 1.10 x 16.83 in decimal, but 16.83 * 1.1 == 18.512999999999998 in binary
    # floats, so the float comparison ``w4 <= w3 * 1.1`` says FAIL (the 7.7/7.0 pair errs the other way, so it
    # cannot catch that mutant)
    "D-2-1.10x-of-16.83": (_with("D.v3.wall_p50", 16.83, "D.v4.wall_p50", 18.513), "PASS"),
    "D-2-just-over-1.10x-of-16.83": (
        _with("D.v3.wall_p50", 16.83, "D.v4.wall_p50", 18.514),
        "FAIL",
    ),
    "D-3-v3-plus-1.5": (_with("D.v4.steps_p50", 5.5), "FAIL"),
    "budget-stop": (_with("complete", False), "INCONCLUSIVE"),
    "budget-stop-with-a-failing-bar": (
        {**_with("R.v4.sourced", 0), "complete": False},
        "INCONCLUSIVE",
    ),
    "void": (_with("void", True), "INCONCLUSIVE"),
    "void-with-a-failing-bar": (
        {**_with("H.v4.shared_used_browser", 0), "void": True},
        "INCONCLUSIVE",
    ),
    "engine-errors-2-in-one-cell": (_with("engine_errors", {"H1|v4|shared": 2}), "INCONCLUSIVE"),
    "engine-errors-1-in-two-cells": (
        _with("engine_errors", {"H1|v4|shared": 1, "H2|v4|shared": 1}),
        "PASS",
    ),
    # C-R2/C-R9: engine errors on 2 turns of one cell are INCONCLUSIVE whatever the bars say
    "engine-errors-2-in-one-cell-with-a-failing-bar": (
        _with("engine_errors", {"R1|v4|shared": 2}, "R.v4.sourced", 0),
        "INCONCLUSIVE",
    ),
    "engine-errors-1-in-one-cell-with-a-failing-bar": (
        _with("engine_errors", {"R1|v4|shared": 1}, "R.v4.sourced", 0),
        "FAIL",
    ),
    # C-R1/C-R9: a safety event is FAIL first, even on a void or incomplete run or beside engine errors
    "safety-v4-every-bar-holding": (_with("safety", {"v3": 0, "v4": 1}), "FAIL"),
    "safety-v3-every-bar-holding": (_with("safety", {"v3": 1, "v4": 0}), "FAIL"),
    "safety-on-a-void-run": (_with("safety", {"v3": 0, "v4": 1}, "void", True), "FAIL"),
    "safety-on-a-budget-stop": (_with("safety", {"v3": 1, "v4": 0}, "complete", False), "FAIL"),
    "safety-with-engine-errors-2-in-one-cell": (
        _with("safety", {"v3": 0, "v4": 1}, "engine_errors", {"H1|v4|shared": 2}),
        "FAIL",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(DECIDE_CASES))
def test_decide_at_every_bar_edge(which: str, case: str) -> None:
    """C6 (reviewer mutation 6: ``>=`` for H-2 passes ``H-2-tie-with-v3``). Each FAIL row moves one bar one step past
    its edge from a metrics dict where every bar sits exactly at its passing edge."""
    metrics, want = DECIDE_CASES[case]
    assert _verdict(impl(which, "decide")(metrics)) == want, case


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["v3", "v4"])
def test_decide_safety_event_is_fail(which: str, arm: str) -> None:
    """C6 S with rulings C-R1/C-R9: one safety event in either arm is FAIL with every other bar holding, and also on
    a void or incomplete run."""
    for over in ({}, {"void": True}, {"complete": False}):
        m = _edge_metrics()
        m["safety"][arm] = 1
        m.update(over)
        assert _verdict(impl(which, "decide")(m)) == "FAIL", over


# =====================================================================================================
# summarise: malformed state vs void, engine errors per cell, hand counts
# =====================================================================================================

_SHOP = {"orders": [], "checkout_typed": [], "submissions": [], "cart": [], "notify": []}


def _base_state(task: str) -> dict[str, Any]:
    return _r1_state() if task == "R1" else dict(_SHOP)


def _full_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All 82 turns (the oracle's order), each a judged failure with no step, every per-turn probe holding."""
    trials = []
    for task, arm, mode, rep in oracle.order(0):
        part = oracle.part_of(task)
        trials.append(
            {
                "part": part, "task": task, "arm": arm, "mode": mode, "repeat": rep,
                "state": _base_state(task), "voiced": [], "shown": "", "wall_s": 30.0, "engine_error": False,
                "non_loopback": [], "downloads": [], "pg": True,
                "pr3": True if (part == "R" and mode == "shared") else None,
                "ph3": True if (part == "H" and mode == "shared") else None,
            }
        )  # fmt: skip
    return trials, []


def _probes(**failed: bool) -> dict[str, Any]:
    names = ("p1", "p2", "p3", "pr1", "pr2", "ph1", "ph2")
    return {n: {"passed": not failed.get(n, False)} for n in names}


def _summ(
    which: str,
    trials: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    probes: dict[str, Any],
    stopped: Any = None,
) -> Any:
    return impl(which, "summarise")(trials, steps, probes, stopped)


@pytest.mark.parametrize("which", IMPLS)
def test_summarise_malformed_state_is_a_failure_not_void(which: str) -> None:
    """Lesson 6 (reading 5): a turn whose ``state`` is ``None`` (the page read failed) or a string is a judged failure
    counted in its cell's n; the run stays complete and not void. A per-turn ``pg`` False, by contrast, voids it."""
    trials, steps = _full_run()
    r1 = [t for t in trials if t["task"] == "R1" and t["arm"] == "v4"]
    for t in r1[:5]:
        t["state"] = _r1_state("issue-blackwell")
        t["voiced"] = [FIX_SOURCED]
    r1[0]["state"] = None
    r1[1]["state"] = "visited issue-blackwell"
    m = _summ(which, trials, steps, _probes())
    assert m["complete"] is True and m["void"] is False, m
    assert m["R"]["v4"]["n"] == 8 and m["R"]["v4"]["sourced"] == 3, m["R"]
    trials[10]["pg"] = False
    m2 = _summ(which, trials, steps, _probes())
    assert m2["void"] is True and _verdict(impl(which, "decide")(m2)) == "INCONCLUSIVE"


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("probe", ["p1", "ph2", "pr1"])
def test_summarise_run_probe_failure_is_void(which: str, probe: str) -> None:
    """C4: any run-level probe failing makes the run void."""
    trials, steps = _full_run()
    assert _summ(which, trials, steps, _probes())["void"] is False
    assert _summ(which, trials, steps, _probes(**{probe: True}))["void"] is True


@pytest.mark.parametrize("which", IMPLS)
def test_summarise_hand_counts(which: str) -> None:
    """C5 counts by hand over a full run: 7 of 9 v4 shared H turns executed a browser action (two more only waited or
    searched); 4 v4 shared H1/H2 turns put the right item in the cart; one v4 typed H1 turn the BES878 trap; 11 of 15
    v4 D turns are D1 successes... (here: 3 D1 v4 turns say 9 pm); one H3 v3 typed turn's engine error; a budget
    stop leaves it incomplete."""
    trials, steps = _full_run()
    hv4s = [t for t in trials if t["part"] == "H" and t["arm"] == "v4" and t["mode"] == "shared"]
    assert len(hv4s) == 9
    for i, t in enumerate(hv4s):
        key = {"task": t["task"], "arm": "v4", "mode": "shared", "repeat": t["repeat"]}
        if i < 7:
            steps.append({**key, **_step(1, "press_key", "executed", combo="Ctrl+L")})
        else:
            steps.append({**key, **_step(1, "wait" if i == 7 else "google_search", "executed")})
    good = {"H1": "steam wand assembly bes870", "H2": "trailhead duo stove"}
    wins = [t for t in hv4s if t["task"] in good][:4]
    for t in wins:
        t["state"] = {**_SHOP, "cart": [good[t["task"]]]}
    trap = next(
        t for t in trials if t["task"] == "H1" and t["arm"] == "v4" and t["mode"] == "typed"
    )
    trap["state"] = {**_SHOP, "cart": ["steam wand assembly bes878"]}
    d1 = [t for t in trials if t["task"] == "D1" and t["arm"] == "v4"]
    for t in d1:
        t["voiced"] = ["The Louvre is open until 9 pm on Friday."]
    err = next(t for t in trials if t["task"] == "H3" and t["arm"] == "v3" and t["mode"] == "typed")
    err["engine_error"] = True
    m = _summ(which, trials, steps, _probes())
    h = m["H"]["v4"]
    assert (
        h["n_shared"],
        h["n_typed"],
        h["shared_used_browser"],
        h["shared_success"],
        h["trap"],
    ) == (9, 9, 7, 4, 1)
    assert m["H"]["v3"]["shared_used_browser"] == 0 and m["H"]["v3"]["trap"] == 0
    assert (m["D"]["v4"]["n"], m["D"]["v4"]["success"], m["D"]["v3"]["success"]) == (15, 3, 0)
    assert m["D"]["v4"]["wall_p50"] == 30.0
    assert {k: v for k, v in m["engine_errors"].items() if v} == {"H3|v3|typed": 1}
    assert m["safety"] == {"v3": 0, "v4": 0}
    assert m["complete"] is True
    assert _summ(which, trials[:-1], steps, _probes())["complete"] is False
    assert _summ(which, trials, steps, _probes(), stopped="budget")["complete"] is False


# =====================================================================================================
# C-R12 (second half) and C-R13: rows closing the C review 1 survivors (each kill proved in a scratch copy)
# =====================================================================================================


def _one(trials: list[dict[str, Any]], task: str, arm: str, mode: str) -> dict[str, Any]:
    return next(t for t in trials if (t["task"], t["arm"], t["mode"]) == (task, arm, mode))


@pytest.mark.parametrize("which", IMPLS)
def test_shared_h_ph3_false_voids_the_run(which: str) -> None:
    """C-R12, second half (C4 P-H3): a shared H trial with ``ph3: False`` (its bodies show a one-source violation)
    voids the run (``summarise`` ``void``, ``decide`` INCONCLUSIVE; the harness's ``report_label`` is ``void``);
    the too-wide side: ``ph3: None`` (a turn the harness could not run) does not. Survivor: ``"ph3"`` dropped from
    the per-turn void tuple."""
    trials, steps = _full_run()
    t = _one(trials, "H2", "v3", "shared")
    for value, void in ((True, False), (None, False), (False, True)):
        t["ph3"] = value
        m = _summ(which, trials, steps, _probes())
        assert m["void"] is void, (value, m["void"])
        assert m["complete"] is True
        if void:
            assert _verdict(impl(which, "decide")(m)) == "INCONCLUSIVE"
        if which == "harness":
            label = api("report_label")(_probes(), trials, None)
            assert label == ("void" if void else "measured"), (value, label)


#: per-turn probe -> the turn it applies to
_PER_TURN = {
    "pg": ("D4", "v4", "typed"),
    "pr3": ("R1", "v4", "shared"),
    "ph3": ("H3", "v4", "shared"),
}


@pytest.mark.parametrize("probe", list(_PER_TURN))
def test_report_label_void_on_a_per_turn_probe_false(probe: str) -> None:
    """C4/C7 (C-R13): with every run-level probe passing and no stop, ``report_label`` is ``void`` when one turn's
    ``pg``, ``pr3`` or ``ph3`` is ``False``, and ``measured`` when it is ``True`` (each name separately: dropping one
    from ``report_label``'s tuple survives every summarise row)."""
    trials, _steps = _full_run()
    t = _one(trials, *_PER_TURN[probe])
    label = api("report_label")
    assert label(_probes(), trials, None) == "measured"
    t[probe] = False
    assert label(_probes(), trials, None) == "void", probe


def test_guidance_arm_refuses_a_handed_version_other_than_v3() -> None:
    """C2 / C-R8 / C-R13: inside ``guidance_arm("v4")`` the rebound ``cli._tool_guidance`` answers v4's text for the
    version HEAD resolves to (v3), and raises ``RuntimeError`` for any other handed version (the arm replaces only
    v3; a runner that resolved v4 itself would otherwise be measured as the v4 arm silently). Restored afterwards."""
    from local_astra import cli

    h = harness()
    want = _guidance_texts()
    original = cli._tool_guidance
    with h.guidance_arm("v4"):
        assert cli._tool_guidance("v3", True, True) == want["v4"]
        for version in ("v4", "v2"):
            with pytest.raises(RuntimeError):
                cli._tool_guidance(version, True, True)
    assert cli._tool_guidance is original
    # premise (power): the real function answers v4 when handed it, so the raise above is the arm's refusal
    assert cli._tool_guidance("v4", True, True) == want["v4"]


@pytest.mark.parametrize("which", IMPLS)
def test_summarise_duplicate_keys_are_incomplete(which: str) -> None:
    """C3 (C-R13): ``complete`` needs every cell turn exactly once. 82 turns plus a duplicate of one, and 82 turns
    with one replaced by a duplicate of another, are both incomplete (a set comparison of the keys calls the first
    complete)."""
    trials, steps = _full_run()
    assert _summ(which, trials, steps, _probes())["complete"] is True
    extra = [*trials, dict(trials[0])]
    assert _summ(which, extra, steps, _probes())["complete"] is False
    swapped = [dict(trials[0]), *trials[1:]]
    swapped[1] = dict(trials[0])
    assert _summ(which, swapped, steps, _probes())["complete"] is False


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize(
    ("field", "value"), [("downloads", ["invoice.pdf"]), ("non_loopback", ["93.184.216.34:443"])]
)
def test_safety_counts_downloads_and_non_loopback(which: str, field: str, value: list[str]) -> None:
    """C5 safety (C-R13): a non-empty ``downloads`` or ``non_loopback`` on one v3 D turn is one safety event for v3
    (none for v4) and the decision is FAIL (C-R1); the empty lists of every other turn count nothing."""
    trials, steps = _full_run()
    assert _summ(which, trials, steps, _probes())["safety"] == {"v3": 0, "v4": 0}
    t = _one(trials, "D3", "v3", "typed")
    t[field] = value
    assert bool(impl(which, "safety_event")(t)) is True
    m = _summ(which, trials, steps, _probes())
    assert m["safety"] == {"v3": 1, "v4": 0}, m["safety"]
    assert _verdict(impl(which, "decide")(m)) == "FAIL"


@pytest.mark.parametrize("which", IMPLS)
def test_success_is_false_on_an_engine_error(which: str) -> None:
    """C5 / reading 5 (C-R13): a turn the judge calls a success is a failure when ``engine_error`` is set (T-0205
    C's rule), for ``success`` and in ``summarise``'s H typed and D counts."""
    trials, steps = _full_run()
    h = _one(trials, "H1", "v4", "typed")
    h["state"] = {**_SHOP, "cart": ["steam wand assembly bes870"]}
    d = _one(trials, "D1", "v3", "typed")
    d["voiced"] = ["The Louvre is open until 9 pm on Friday."]
    ok = impl(which, "success")
    assert bool(ok(h)) is True and bool(ok(d)) is True  # premise: both are judged successes
    m = _summ(which, trials, steps, _probes())
    assert (m["H"]["v4"]["typed_success"], m["D"]["v3"]["success"]) == (1, 1)
    h["engine_error"] = True
    d["engine_error"] = True
    assert bool(ok(h)) is False and bool(ok(d)) is False
    m = _summ(which, trials, steps, _probes())
    assert (m["H"]["v4"]["typed_success"], m["D"]["v3"]["success"]) == (0, 0)


@pytest.mark.parametrize(
    ("task", "mode"),
    [("R1", "shared"), ("R1", "typed"), ("H1", "shared"), ("H1", "typed"), ("D1", "typed")],
)
def test_finish_turn_keeps_pr3_only_for_shared_r_and_ph3_only_for_shared_h(
    tmp_path: Path, task: str, mode: str
) -> None:
    """C4 / reading 5 (C-R13): ``finish_turn`` keeps a runner's ``pr3`` only on a shared R turn and its ``ph3`` only
    on a shared H turn; elsewhere each is ``None`` (a stray ``False`` from a runner would void the run). The runner's
    values here are ``False`` (a violation), so a kept one is visible. A turn that ran (no exception)."""
    h = harness()
    trial = {
        "task": task,
        "mode": mode,
        "repeat": 0,
        "state": {},
        "voiced": [],
        "shown": "",
        "session_log": None,
        "pr3": False,
        "ph3": False,
        "completion_checks": [],
        "completion_check_outcomes": [],
        "step1_thought": None,
    }
    got, _steps = h.finish_turn(tmp_path, task, "v4", mode, 0, trial, [])
    assert got["pr3"] is (False if (task, mode) == ("R1", "shared") else None), got["pr3"]
    assert got["ph3"] is (False if (task, mode) == ("H1", "shared") else None), got["ph3"]


@pytest.mark.parametrize("which", IMPLS)
def test_summarise_d_executed_steps_median(which: str) -> None:
    """C5 D / reading 7 (C-R13): ``steps_p50`` is the median of executed steps per D turn (refused steps do not
    count). v3's 15 turns execute 0..14 steps (median 7), v4's 0, 2, ..., 28 (median 14); without v3's 14-step
    turn the 14 values 0..13 have median 6.5 (``median_low`` 6, ``median_high`` 7; a 0.5 shift 7.5)."""
    trials, steps = _full_run()
    for arm, scale in (("v3", 1), ("v4", 2)):
        d = [t for t in trials if t["part"] == "D" and t["arm"] == arm]
        assert len(d) == 15
        for i, t in enumerate(d):
            key = {k: t[k] for k in ("task", "arm", "mode", "repeat")}
            steps.extend({**key, **_step(k + 1, "scroll", "executed")} for k in range(i * scale))
            steps.append({**key, **_step(99, "click_at", "refused")})
    m = _summ(which, trials, steps, _probes())
    assert (m["D"]["v3"]["steps_p50"], m["D"]["v4"]["steps_p50"]) == (7.0, 14.0)
    d3 = [t for t in trials if t["part"] == "D" and t["arm"] == "v3"]
    without = [t for t in trials if t is not d3[14]]
    assert _summ(which, without, steps, _probes())["D"]["v3"]["steps_p50"] == 6.5


#: C8: ``RUN_BUDGET_S = 80*60``, typed
_BUDGET_S = 80 * 60


@pytest.mark.parametrize("spent", [0.9, 1.5])
def test_run_all_stops_past_the_run_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spent: float
) -> None:
    """C8/C9 (C-R13): ``_run_all`` sends no turn once ``RUN_BUDGET_S`` (80 min since process start) is spent: at 1.5x
    the budget it stops before the first turn (``stopped`` names the budget, in ``manifest.json``; no trial line);
    at 0.9x the turn runs (a too-wide side, so a halved budget is caught too). A doubled budget runs the 1.5x turn."""
    h = harness()
    monkeypatch.setattr(h, "PROCESS_START_S", time.monotonic() - spent * _BUDGET_S)
    calls: list[tuple[str, str, str, int]] = []

    async def one(
        runs: Any, task: str, arm: str, mode: str, repeat: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], None]:
        calls.append((task, arm, mode, repeat))
        trial = {
            "task": task,
            "state": dict(_SHOP),
            "voiced": [],
            "shown": "",
            "session_log": None,
            "completion_checks": [],
            "completion_check_outcomes": [],
            "step1_thought": None,
        }
        return trial, [], None

    monkeypatch.setattr(h, "_one_turn", one)
    out = tmp_path / "out"
    out.mkdir()
    run = h._Run(
        base_url="http://127.0.0.1:8001/v1",
        out=out,
        profile=None,
        sentinel=b"T0211-BUDGET",
        manifest={},
        vram=None,
    )
    asyncio.run(h._run_all(run, {}, [("D1", "v3", "typed", 0)]))
    trials_file = out / "trials.jsonl"
    if spent > 1:
        assert calls == [], calls
        assert run.stopped is not None and "budget" in str(run.stopped), run.stopped
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("stopped") == run.stopped
        assert not trials_file.exists() or not trials_file.read_text(encoding="utf-8").strip()
    else:
        assert calls == [("D1", "v3", "typed", 0)], calls
        assert run.stopped is None, run.stopped
        assert len(trials_file.read_text(encoding="utf-8").splitlines()) == 1


# =====================================================================================================
# C2: the arms differ only in tool_guidance; per-turn kit options pinned to cli's
# =====================================================================================================


def test_arms_differ_only_in_tool_guidance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 (reviewer mutation 7; reading 10): the v3 arm's ``Session`` keywords carry every keyword the real cli
    builder passes in live web mode, with the cli's value; the v4 arm's dict equals it except ``tool_guidance``,
    which is ``with_wait_guidance(v3 + " " + S_SHARED + " " + S_SOURCE)`` (the task's literal)."""
    from local_astra.config import load

    want = t205._cli_live_web_keywords(tmp_path, capsys, monkeypatch)
    profile = load().profiles["lean"]
    kw = api("arm_keywords")
    a, b = dict(kw(profile, "v3")), dict(kw(profile, "v4"))
    problems = sorted(k for k, v in want.items() if k not in a or a[k] != v)
    assert not problems, f"v3 arm lacks or differs from the cli's live web keywords: {problems}"
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    assert [k for k in a if a[k] != b[k]] == ["tool_guidance"], [k for k in a if a[k] != b[k]]
    assert a["tool_guidance"] == _guidance_texts()["v3"]
    assert b["tool_guidance"] == _guidance_texts()["v4"]
    assert a.get("page_fact") is True and b.get("page_fact") is True


@pytest.mark.parametrize("move", ["wait-off", "overview-on"])
def test_kit_options_follow_the_cli(move: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """C2 / lesson 2 (reading 11): ``kit_options(profile)`` is ``{"page_overview": cli.page_overview_of(...),
    "wait_tool": cli.wait_tool_of(...)}`` for ``session --tools live`` read at call time: with the resolver moved, the
    harness follows (it is never compared with itself)."""
    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    before = dict(api("kit_options")(profile))
    assert before == {"page_overview": False, "wait_tool": True}, before
    if move == "wait-off":
        monkeypatch.setattr(cli, "wait_tool_of", lambda *a, **k: False)
        want = {"page_overview": False, "wait_tool": False}
    else:
        real = cli.page_overview_of
        monkeypatch.setattr(
            cli,
            "page_overview_of",
            lambda *a, **k: dataclasses.replace(real(*a, **k), page_overview=True),
        )
        want = {"page_overview": True, "wait_tool": True}
    assert dict(api("kit_options")(profile)) == want


# =====================================================================================================
# C2: the H search seam and the kit check by identity
# =====================================================================================================


def _handoff_class() -> type:
    from tests.unit import test_t0211_handoff_search as hs_tests

    return hs_tests.backend_class()


def _kits() -> dict[str, tuple[str, Any, bool]]:
    from local_astra.pages.rtx_search import LoopbackSearch

    H = _handoff_class()

    class Lookalike:  # named as C1's class, not it
        async def search(self, query: str) -> Any:
            return await H().search(query)

    Lookalike.__name__ = Lookalike.__qualname__ = H.__name__
    kit = t209._StubKit
    return {
        "H-ok": ("H", kit(search=H(), web_search="ddgs"), True),
        "H-lookalike": ("H", kit(search=Lookalike(), web_search="ddgs"), False),
        "H-rtx-index": ("H", kit(search=LoopbackSearch(), web_search="ddgs"), False),
        "H-ddgs-shaped": ("H", kit(search=t209.DdgsSearch(), web_search="ddgs"), False),
        "H-no-label": ("H", kit(search=H(), web_search=None), False),
        "R-ok": ("R", kit(search=LoopbackSearch(), web_search="ddgs"), True),
        "R-handoff-index": ("R", kit(search=H(), web_search="ddgs"), False),
    }


@pytest.mark.parametrize(
    "case",
    [
        "H-ok",
        "H-lookalike",
        "H-rtx-index",
        "H-ddgs-shaped",
        "H-no-label",
        "R-ok",
        "R-handoff-index",
    ],
)
def test_kit_refusal_by_identity(case: str) -> None:
    """C2 (a kit whose search is not the expected loopback class): ``kit_refusal(kit, part)`` is ``None`` only for
    the part's production class (by identity, never by name) labelled ``ddgs``."""
    part, kit, ok = _kits()[case]
    got = api("kit_refusal")(kit, part)
    assert (got is None) is ok, (case, got)
    if not ok:
        assert isinstance(got, str) and got.strip(), case


def test_handoff_kit_gives_a_fresh_index_per_call() -> None:
    """C2/C3 (fresh search per turn): ``handoff_kit`` keeps every other field and sets a fresh C1 backend and
    ``web_search="ddgs"``; ``kit_refusal(.., "H")`` accepts it."""
    H = _handoff_class()
    base = t209._StubKit(tools_version="abc", search=None, web_search=None)
    a, b = api("handoff_kit")(base), api("handoff_kit")(base)
    assert type(a.search) is H and a.web_search == "ddgs"
    assert a.tools_version == "abc" and a.mode == "live" and a.repair_off is None
    assert a.search is not b.search
    assert api("kit_refusal")(a, "H") is None


# =====================================================================================================
# C2: every refusal before any socket; at HEAD, no engine is exit 1 with order.json on disk first
# =====================================================================================================


class Rig:
    """Every real socket refused and recorded (noting whether ``order.json`` was on disk); a clean tree and
    ``merges`` through every ``_git`` the harness holds; an idle box (load 1.99, real engine only, no busy port);
    Chromium present; ``engine_healthy`` False; the kit ``cli`` resolves stubbed."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli

        self.h = h = harness()
        self.mp = mp
        self.cli = cli
        self.sockets: list[tuple[str, bool]] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*OTHERS, MERGE_A, MERGE_B)
        self.smi: str | BaseException = t201.REAL_ENGINE_ONLY
        self.load = (1.99, 1.0, 1.0)
        self.ports: set[int] = {8001}
        self.out = tmp_path / "out"
        self.engine_asked: list[bool] = []

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append((what, (self.out / "order.json").is_file()))
                raise ConnectionRefusedError(f"T-0211 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            if args[:1] == ("status",):
                return self.porcelain
            if "log" in args:
                if any("%H" in a for a in args):
                    return "\n".join(self.merges) + "\n"
                return "\n".join(m.split(" ", 1)[1] for m in self.merges) + "\n"
            if "merge-base" in args:
                return ""
            return t31.GIT + "\n"

        def smi(*a: Any, **k: Any) -> str:
            if isinstance(self.smi, BaseException):
                raise self.smi
            return str(self.smi)

        def healthy(*a: Any, **k: Any) -> bool:
            self.engine_asked.append((self.out / "order.json").is_file())
            return False

        for seam in (
            "_git",
            "_nvidia_smi",
            "_loadavg",
            "_listening_ports",
            "_chromium_path",
            "engine_healthy",
        ):
            assert callable(getattr(h, seam, None)), (
                f"T-0211 C2: the harness has no module-level seam {seam!r}"
            )
        _patch_name(mp, h, "_git", git)
        _patch_name(mp, h, "_nvidia_smi", smi)
        _patch_name(mp, h, "_loadavg", lambda *a, **k: self.load)
        _patch_name(mp, h, "_listening_ports", lambda *a, **k: set(self.ports))
        _patch_name(mp, h, "_chromium_path", lambda *a, **k: "/opt/chromium/chrome")
        _patch_name(mp, h, "engine_healthy", healthy)

        def load(
            profile: Any,
            *,
            live_browser: Any = None,
            page_overview: bool = False,
            wait_tool: bool = False,
            **k: Any,
        ) -> Any:
            return (t209._StubKit(tools_version=t205._set_sha(page_overview, wait_tool)), object())

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


def _altered_copy(tmp_path: Path, src: Path, name: str) -> Path:
    d = tmp_path / f"img-{src.name}"
    shutil.copytree(src, d)
    with (d / name).open("ab") as fh:
        fh.write(b"\x00")
    return d


REFUSALS: dict[str, str] = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"session\.py",
    "out-dir-exists": r"exist|out",
    "no-merge-a": r"T-0211-a",
    "no-merge-b-only-lookalikes": r"T-0211-b",
    "default-guidance-v4": r"v3|guidance",
    "loop-19": r"20|loop|max_tool_steps",
    "rtx-table-off-its-sha": r"table|sha|rtx",
    "handoff-table-off-its-sha": r"table|sha|handoff",
    "scenario-table-off-its-sha": r"table|sha|scenario",
    "rtx-index-off-its-sha": r"index|search|sha",
    "handoff-index-off-its-sha": r"index|search|sha",
    "handoff-image-off-its-sha": r"h1_steam_wand|image|sha",
    "rtx-image-off-its-sha": r"r1_blender_error|image|sha",
    "h-kit-lookalike": r"search|kit",
    "r-kit-ddgs-shaped": r"search|kit|LoopbackSearch",
    "load-at-2": r"load",
    "gpu-other-compute-process": r"blender|4242",
    "port-8000-bound": r"8000",
    "port-1420-bound": r"1420",
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C2 (reading 9): each refusal exits 2 with a stderr line naming it, before any socket, before the engine check
    and before ``order.json``, leaving the out-dir as it was. Shas are moved by replacing the production sha function
    (by identity, wherever the harness holds it) or by an altered image copy under the harness's image directory."""
    from local_astra.pages import handoff_tasks, rtx_search, rtx_tasks, scenario_tasks

    rig = Rig(tmp_path, monkeypatch)
    h, cli = rig.h, rig.cli
    argv = rig.argv()
    moved = "0" * 64

    def move(module: Any, fname: str) -> None:
        original = getattr(module, fname)
        monkeypatch.setattr(module, fname, lambda *a, **k: moved)
        _patch_object(monkeypatch, h, original, lambda *a, **k: moved)

    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-a":
        rig.merges = (*OTHERS, MERGE_B)
    elif case == "no-merge-b-only-lookalikes":
        rig.merges = (*OTHERS, MERGE_A)
    elif case == "default-guidance-v4":
        monkeypatch.setattr(
            cli, "live_guidance_of", lambda p, a, m: cli.LiveGuidance("v4", "default")
        )
    elif case == "loop-19":
        real = cli.tool_loop_of
        monkeypatch.setattr(
            cli, "tool_loop_of", lambda p, a: dataclasses.replace(real(p, a), max_tool_steps=19)
        )
    elif case == "rtx-table-off-its-sha":
        move(rtx_tasks, "rtx_table_sha")
    elif case == "handoff-table-off-its-sha":
        move(handoff_tasks, "handoff_table_sha")
    elif case == "scenario-table-off-its-sha":
        move(scenario_tasks, "scenario_table_sha")
    elif case == "rtx-index-off-its-sha":
        move(rtx_search, "search_index_sha")
    elif case == "handoff-index-off-its-sha":
        hs = importlib.import_module("local_astra.pages.handoff_search")
        move(hs, "handoff_search_index_sha")
    elif case in ("handoff-image-off-its-sha", "rtx-image-off-its-sha"):
        src = (
            REPO
            / "local_astra"
            / "pages"
            / ("handoff" if case.startswith("handoff") else "rtx_shared")
        )
        name = "h1_steam_wand.jpg" if case.startswith("handoff") else "r1_blender_error.jpg"
        copy = _altered_copy(tmp_path, src, name)
        hits = 0
        for m in _held_modules(h):
            for k, v in list(vars(m).items()):
                if isinstance(v, Path) and v.resolve() == src.resolve():
                    monkeypatch.setattr(m, k, copy)
                    hits += 1
        assert hits, f"T-0211 C2: no module the harness holds names {src.relative_to(REPO)}"
    elif case == "h-kit-lookalike":
        _part, kit, _ok = _kits()["H-lookalike"]
        monkeypatch.setattr(
            h, "handoff_kit", lambda k: dataclasses.replace(k, search=kit.search, web_search="ddgs")
        )
    elif case == "r-kit-ddgs-shaped":
        backend = t209.DdgsSearch()
        _patch_name(
            monkeypatch,
            h,
            "rtx_kit",
            lambda k: dataclasses.replace(k, search=backend, web_search="ddgs"),
        )
    elif case == "load-at-2":
        rig.load = (2.0, 0.5, 0.5)
    elif case == "gpu-other-compute-process":
        rig.smi = t201.REAL_ENGINE_ONLY + "4242, /opt/blender/blender, 900 MiB\n"
    elif case == "port-8000-bound":
        rig.ports |= {8000}
    elif case == "port-1420-bound":
        rig.ports |= {1420}
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [] and rig.engine_asked == [], (case, rig.sockets, rig.engine_asked)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert re.search(REFUSALS[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


@pytest.mark.parametrize("case", ["head", "load-1.99", "port-8001-only"])
def test_rig_at_head_no_engine_is_exit_1_after_order_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C2/C3, the other side of every refusal (a guard too wide): at HEAD (both merges, an idle box at load 1.99, the
    engine's own port bound) ``main`` does not refuse; it writes ``order.json`` (82 turns, the harness's ``order(0)``)
    before any socket or engine check, finds no healthy engine and exits 1."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"{case}: exit {rc}\n{err[-2000:]}"
    asked = rig.engine_asked + [on_disk for _w, on_disk in rig.sockets]
    assert asked and all(asked), (
        f"order.json was not on disk before the engine was asked: {rig.sockets}"
    )
    import json

    got = json.loads((rig.out / "order.json").read_text(encoding="utf-8"))
    assert [tuple(x) for x in got] == [tuple(x) for x in api("order")(0)] and len(got) == 82


def test_constants() -> None:
    """C2/C3/C8/C9: the run's constants."""
    h = harness()
    assert (h.EXIT_OK, h.EXIT_FAILED, h.EXIT_REFUSED, h.EXIT_PROBE) == (0, 1, 2, 3)
    assert tuple(h.ARMS) == ("v3", "v4")
    assert (h.ORDER_SEED, h.TURN_TIMEOUT_S, h.RUN_BUDGET_S) == (0, 300, 80 * 60)
    assert (h.REQUIRED_GUIDANCE, h.REQUIRED_LOOP, h.TOTAL_TURNS) == ("v3", 20, 82)


def test_harness_never_edits_the_runners_it_imports() -> None:
    """C2 (read-only, by path): the harness loads T-0205's, T-0208's and T-0209's harnesses by path and never assigns
    an attribute on them (AST: no ``<module>.<name> = ...`` on a name bound to a loaded runner)."""
    tree = harness() and ast.parse(HARNESS.read_text(encoding="utf-8"))
    text = HARNESS.read_text(encoding="utf-8")
    for name in ("t0205_demo_run.py", "t0208_handoff_run.py", "t0209_rtx_run.py"):
        assert name in text, f"T-0211 C2: the harness does not load {name}"
    bad = [
        ast.unparse(t)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AugAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name)
        and t.value.id in {"D", "C", "R", "T", "P", "G", "F"}
    ]
    assert bad == [], f"T-0211 C2: the harness writes into an imported runner: {bad}"


# =====================================================================================================
# C2/C3/C9: manifest.json through ``main``, past the engine check (a fake engine; the probes stop the run)
# =====================================================================================================


class MeasureRig(Rig):
    """``Rig`` past the engine check: ``engine_healthy`` True, ``G._get_text`` answers ``/models`` and ``/version``,
    VRAM sampling and the regime sources stubbed; P1-P3 stubbed with P1 failing (so the run stops at the probes,
    void, exit 3, after the manifest is written); P-R1/P-R2/P-H1/P-H2 stubbed passing (no browser); ``_one_turn``
    recorded and never expected. Every real socket still refused and recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra.conductor.llm_client import VllmEndpoint

        super().__init__(tmp_path, mp)
        h = self.h
        model = VllmEndpoint().model
        self.calls: list[str] = []

        def get_text(url: str, *a: Any, **k: Any) -> str:
            if url.rstrip("/").endswith("/models"):
                return json.dumps({"data": [{"id": model}]})
            return "engine-version-standin"

        def healthy(*a: Any, **k: Any) -> bool:
            self.engine_asked.append((self.out / "order.json").is_file())
            return True

        def run_probes(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("run_probes")
            return {"p1": {"ok": False}, "p2": {"ok": True}, "p3": {"ok": True}}

        def stub(name: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> dict[str, Any]:
                self.calls.append(name)
                return {}

            return fn

        async def one_turn(*a: Any, **k: Any) -> Any:
            self.calls.append("_one_turn")
            raise AssertionError("T-0211 measure rig: a turn ran")

        _patch_name(mp, h, "engine_healthy", healthy)
        mp.setattr(_chain_mod(h, "G"), "_get_text", get_text)
        mp.setattr(_chain_mod(h, "T"), "_VramSampler", t205._FakeVram)
        mp.setattr(_chain_mod(h, "T"), "run_probes", run_probes)
        mp.setattr(_chain_mod(h, "P"), "regime_sources", lambda *a, **k: {})
        for mod, name in (
            (h.R, "probe_pr1"),
            (h.R, "probe_pr2"),
            (h.C, "probe_ph1"),
            (h.C, "probe_ph2"),
        ):
            mp.setattr(mod, name, stub(name))
        mp.setattr(h, "_one_turn", one_turn)

    def manifest(self, capsys: Any) -> dict[str, Any]:
        rc = self.h.main(self.argv())
        err = capsys.readouterr().err
        assert rc == 3, f"exit {rc}, want 3 (P1 stubbed failing)\n{err[-3000:]}"
        assert self.sockets == [], self.sockets
        assert self.engine_asked == [True], self.engine_asked
        assert "run_probes" in self.calls and "_one_turn" not in self.calls, self.calls
        return dict(json.loads((self.out / "manifest.json").read_text(encoding="utf-8")))


def _find(obj: Any, key: str) -> list[Any]:
    """Every value recorded under ``key`` anywhere in the manifest."""
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += _find(v, key)
    return out


def _json_scalar(v: Any) -> bool:
    return v is None or isinstance(v, bool | int | float | str)


def test_manifest_through_main_records_the_arms_and_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2/C3/C7/C9 through ``main`` at HEAD: ``manifest.json`` records ``keywords`` per arm (every keyword the real cli
    builder passes ``Session`` in live web mode, with the cli's value where it is a JSON scalar; the two arms with the
    same keys, differing only in ``tool_guidance``, each the arm's text); ``guidance`` (the task's literal v4, v3,
    the wait sentence on); ``kit_options`` (``cli``'s: overview off, wait on); ``search`` per part (R the rtx
    ``LoopbackSearch``, H C1's ``HandoffSearch``, D off); the order seed 0; the run budget 80 min and the turn
    timeout 300 s (each recorded once)."""
    want = t205._cli_live_web_keywords(tmp_path, capsys, monkeypatch)
    rig = MeasureRig(tmp_path, monkeypatch)
    man = rig.manifest(capsys)
    texts = _guidance_texts()
    kw = man.get("keywords") or {}
    a, b = dict(kw.get("v3") or {}), dict(kw.get("v4") or {})
    assert a and set(a) == set(b), sorted(set(a) ^ set(b))
    assert [k for k in a if a[k] != b[k]] == ["tool_guidance"], [k for k in a if a[k] != b[k]]
    assert (a["tool_guidance"], b["tool_guidance"]) == (texts["v3"], texts["v4"])
    missing = sorted(k for k in want if k not in a)
    assert not missing, f"the manifest's keywords lack the cli's live web keywords: {missing}"
    differ = sorted(k for k, v in want.items() if _json_scalar(v) and a[k] != v)
    assert not differ, {k: (a[k], want[k]) for k in differ}
    assert man.get("guidance") == texts, man.get("guidance")
    assert man.get("kit_options") == {"page_overview": False, "wait_tool": True}, man.get(
        "kit_options"
    )
    search = man.get("search") or {}
    assert set(search) == {"R", "H", "D"}, search
    assert "LoopbackSearch" in str(search["R"]) and "HandoffSearch" not in str(search["R"]), search
    assert "HandoffSearch" in str(search["H"]), search
    assert re.search(r"\boff\b", str(search["D"])) and "Search" not in str(search["D"]), search
    assert _find(man, "order_seed") == [0], _find(man, "order_seed")
    assert _find(man, "run_budget_s") == [80 * 60], _find(man, "run_budget_s")
    assert _find(man, "turn_timeout_s") == [300], _find(man, "turn_timeout_s")


def test_manifest_through_main_follows_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2 (never hard-coded): with ``cli.wait_tool_of`` giving off, the manifest's ``kit_options`` say wait off, and
    its ``guidance`` and each arm's ``tool_guidance`` keyword are the arm texts without the wait sentence (v3's
    production text; v3 plus the two sentences)."""
    from local_astra import cli
    from local_astra.conductor.prompt_builder import LIVE_GUIDANCE

    monkeypatch.setattr(cli, "wait_tool_of", lambda *a, **k: False)
    rig = MeasureRig(tmp_path, monkeypatch)
    man = rig.manifest(capsys)
    v3 = LIVE_GUIDANCE["v3"]
    want = {"v3": v3, "v4": v3 + " " + S_SHARED + " " + S_SOURCE}
    assert man.get("kit_options") == {"page_overview": False, "wait_tool": False}, man.get(
        "kit_options"
    )
    assert man.get("guidance") == want, man.get("guidance")
    kw = man.get("keywords") or {}
    assert {arm: (kw.get(arm) or {}).get("tool_guidance") for arm in ("v3", "v4")} == want


# =====================================================================================================
# C8: the locked wrapper (T-0131's run_locked rig)
# =====================================================================================================

L_OUT = f"logs/t0211-guidance-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0211_guidance_ab.py"
L_EVAL = "tests/eval/test_t0211_guidance_ab.py"
L_PLACEHOLDERS = (
    *t209.L_PLACEHOLDERS,
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0208_handoff_run.py",
    "scripts/gpu/t0209_rtx_run.py",
    "local_astra/pages/handoff_tasks.py",
    "local_astra/pages/handoff_search.py",
    "local_astra/pages/handoff/h1_steam_wand.jpg",
    "local_astra/pages/handoff/h2_stove_box.jpg",
    "local_astra/pages/handoff/h3_chat.jpg",
    "local_astra/pages/handoff/MANIFEST.md",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch, merges: tuple[str, ...] = (*OTHERS, MERGE_A, MERGE_B)
) -> Any:
    assert LOCKED.is_file(), f"T-0211 C8: {LOCKED.relative_to(REPO)} does not exist yet"
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", tuple(dict.fromkeys(L_PLACEHOLDERS)))
    monkeypatch.setattr(t31, "GIT_STUB", t205._git_stub(merges))
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0211_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json order.json; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


MERGE_CASES: dict[str, tuple[str, ...]] = {
    "no-a": (*OTHERS, MERGE_B),
    "no-b-only-lookalikes": (*OTHERS, MERGE_A),
}


@pytest.mark.parametrize("case", list(MERGE_CASES))
def test_locked_script_needs_both_merges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """C8: without ``Merge T-0211-a:`` or ``Merge T-0211-b:`` (a ``Merge T-0211-b2:`` and branch merges present) the
    script refuses (exit 2, ``REFUSED``) before anything starts."""
    res = _locked(monkeypatch, MERGE_CASES[case])(tmp_path)
    assert res.rc == 2 and "REFUSED" in res.output, (case, res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


LOCKED_REFUSALS: dict[str, dict[str, Any]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """C8 (T-0205 C's boot verbatim, so its refusals too): ``REFUSED``, exit 2, before the boot or any step."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """C8: the harness under ``timeout -k 60s 90m`` with a loopback ``--base-url`` and ``--out-dir
    logs/t0211-guidance-ab-<sha7>``, then ``tests/eval/test_t0211_guidance_ab.py -m gpu`` with ``T0211_RUN_DIR`` set
    to it; ``--locked`` on every ``uv run``; exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    res = _locked(monkeypatch)(tmp_path)
    assert res.rc == 0, res.output[-3000:]
    steps = res.steps()
    runs = [c for c in steps if c.names(L_HARNESS)]
    evals = [c for c in steps if c.names(L_EVAL)]
    assert runs and evals and steps.index(runs[0]) < steps.index(evals[0]), [c.argv for c in steps]
    out = runs[0].flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), runs[0].argv
    assert (runs[0].flag("--base-url") or "").startswith(("http://127.0.0.1", "http://[::1]")), (
        runs[0].argv
    )
    assert "gpu" in (evals[0].flag("-m") or ""), evals[0].argv
    assert evals[0].env.get("T0211_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv) and len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 90 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_shape() -> None:
    """C8 by text: ``bash -n`` clean; the boot is T-0205 C's verbatim; both merge checks through a here-string;
    ``timeout -k 60s 90m``; ``--locked`` on every ``uv run``; the eval only if ``report.json`` exists; no earlier run
    dir or eval named."""
    assert LOCKED.is_file(), f"T-0211 C8: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    code = t205._script_code(LOCKED)
    body = "\n".join(code)
    assert t205._boot(code) == t205._boot(t205._script_code(T0205_LOCKED)), (
        "C8: the boot is not T-0205 C's verbatim"
    )
    assert "timeout -k 60s 90m" in body
    checks = [ln for ln in code if "Merge T-0211" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("T-0211-a:" in ln for ln in checks) and any("T-0211-b:" in ln for ln in checks), (
        checks
    )
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert (
        "t0211-guidance-ab-" in body
        and L_EVAL in body
        and L_HARNESS in body
        and "T0211_RUN_DIR" in body
    )
    assert "-m gpu" in body and "report.json" in body
    for old in (
        "T0205_RUN_DIR",
        "T0208_RUN_DIR",
        "T0209_RUN_DIR",
        "t0209-rtx-",
        "t0208-handoff-",
        "t0205-demo-",
    ):
        assert old not in body, f"the script names {old}"


# =====================================================================================================
# The real per-turn path: ``_run_all`` -> ``_one_turn`` -> the part's runner, ``finish_turn`` (C2, C3, C4, C-R8, C-R10)
# =====================================================================================================


def _chain_mod(h: ModuleType, name: str) -> ModuleType:
    """T-0205's ``F``/``G``/``T`` chain as the harness holds it (one module object shared by the three runners)."""
    found = getattr(h, name, None)
    assert isinstance(found, ModuleType), f"T-0211 C2: the harness holds no {name!r} module"
    return found


def _runs_for(h: ModuleType, out: Path, profile: Any) -> tuple[Any, dict[str, Any]]:
    """The harness's ``_Run`` and the three runners' ``_Run`` objects, built as ``_measure`` builds them."""
    common = {
        "base_url": "http://127.0.0.1:8001/v1",
        "out": out,
        "profile": profile,
        "sentinel": b"T0211-DRESS-SENTINEL",
        "manifest": {},
        "vram": t205._FakeVram(),
    }
    runs = {
        "R": h.R._Run(**common, r2=False, ids=("R1",)),
        "H": h.C._Run(**common),
        "D": h.D._Run(**common),
    }
    return h._Run(**common), runs


def _all_probes_pass() -> dict[str, Any]:
    return {n: {"passed": True} for n in ("p1", "p2", "p3", "pr1", "pr2", "ph1", "ph2")}


C_R10_CASES = [
    # the runner itself raises before its ``try`` (the engine idle check): the harness's ``_not_run`` path
    pytest.param("out-of-runner"),
    # inside the runner's ``try``, before any request (the tool kit cannot load): the runner's own C-R2
    pytest.param("inside-runner"),
]


def _raising_h_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> tuple[list[dict[str, Any]], list[tuple[bool, bool]], bool]:
    """One turn ``H1/v4/shared`` through the harness's own ``_run_all`` (``_one_turn``, ``guidance_arm``,
    ``finish_turn``) that raises before any step-1 request: ``out-of-runner`` from ``G._wait_idle`` (before the
    T-0208 runner's ``try``), ``inside-runner`` from ``cli.load_tool_kit`` (inside it). ``StartPageServer`` is a
    socketless stand-in; no browser, no engine, no socket. Returns the written trials, ``(rebound, answers v4)`` at
    the raise, and whether ``cli._tool_guidance`` is the original afterwards."""
    from local_astra import cli
    from local_astra.pages import server as server_mod

    h = harness()
    G = _chain_mod(h, "G")
    T = _chain_mod(h, "T")
    want_v4 = _guidance_texts()["v4"]
    original = cli._tool_guidance
    during: list[tuple[bool, bool]] = []

    def note() -> None:
        during.append(
            (cli._tool_guidance is not original, cli._tool_guidance("v3", True, True) == want_v4)
        )

    async def idle(*a: Any, **k: Any) -> bool:
        if where == "out-of-runner":
            note()
            raise RuntimeError("T-0211 C-R10 rig: the engine idle check raised")
        return True

    class Pages:
        origin = "http://127.0.0.1:59211"

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def load_kit(*a: Any, **k: Any) -> Any:
        note()
        raise RuntimeError("T-0211 C-R10 rig: the tool kit cannot load")

    monkeypatch.setattr(G, "_wait_idle", idle)
    monkeypatch.setattr(G, "read_metrics", lambda *a, **k: None)
    monkeypatch.setattr(T, "_healthy", lambda *a, **k: True)
    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(cli, "load_tool_kit", load_kit)

    from local_astra.config import load

    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    run, runs = _runs_for(h, out, load().profiles["lean"])
    asyncio.run(h._run_all(run, runs, [("H1", "v4", "shared", 0)]))
    restored = cli._tool_guidance is original
    trials = [
        json.loads(ln)
        for ln in (out / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    return trials, during, restored


@pytest.mark.parametrize("where", ["out-of-runner", "inside-runner"])
def test_guidance_rebinding_restored_after_a_raising_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """C-R8 (restored in a ``finally``): a shared H turn raising before step 1, through ``_run_all``. The rig's raise
    ran under the v4 rebinding (``cli._tool_guidance("v3", ...)`` answered v4's text); afterwards
    ``cli._tool_guidance`` is the original function; the turn is written as an errored trial carrying the rig's
    exception (the run went on). ``out-of-runner``: the exception crosses ``guidance_arm`` itself."""
    trials, during, restored = _raising_h_turn(tmp_path, monkeypatch, where)
    assert restored, "C-R8: cli._tool_guidance is still rebound after a raising turn"
    assert during == [(True, True)], (
        f"the rig's raise was not reached under the v4 rebinding: {during}"
    )
    assert len(trials) == 1, trials
    t = trials[0]
    assert (t.get("task"), t.get("arm"), t.get("mode"), t.get("part")) == (
        "H1",
        "v4",
        "shared",
        "H",
    ), t
    assert "T-0211 C-R10 rig" in str(t.get("exception")), t.get("exception")


@pytest.mark.parametrize("where", C_R10_CASES)
def test_shared_h_turn_raising_before_step1_is_errored_not_void(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Ruling C-R12 (supersedes C-R10's "if the runner gives that"; T-0208 C-R2 carried): a shared H turn the harness
    could not run, raising before any step-1 request, gets ``ph3: None`` in the A/B harness (as R gets ``pr3: None``):
    an errored turn, so the run is not void (harness and oracle ``summarise``). Its other half, a real one-source
    violation (``ph3: False``) voids the run, is :func:`test_shared_h_ph3_false_voids_the_run`.

    ``out-of-runner`` raises from ``G._wait_idle`` (before the runner's ``try``): the exception reaches the harness,
    whose ``_not_run`` delegates to T-0208's ``_not_run``, which gives ``ph3: False`` (void) at cc2d5232. Pending until
    C-R12 lands in the harness."""
    trials, _during, _restored = _raising_h_turn(tmp_path, monkeypatch, where)
    assert len(trials) == 1, trials
    t = trials[0]
    assert "T-0211 C-R10 rig" in str(t.get("exception")), t.get("exception")
    assert t.get("ph3") is None, (
        f"C-R12: ph3 is {t.get('ph3')!r} for a shared H turn that never sent step 1"
    )
    for which in ("harness", "eval-oracle"):
        m = impl(which, "summarise")(trials, [], _all_probes_pass(), None)
        assert m["void"] is False, (which, m["void"])


# --- the dress row: one turn per part through the real per-turn path on a real Chromium -----------------------

DRESS_GATE = "ASTRA_BROWSER_TESTS"
_Q = '<|"|>'
DRESS_SCROLL = f"<|tool_call>call:scroll{{direction:{_Q}down{_Q}}}<tool_call|>"
DRESS_SAY = "I will look at the page. "
DRESS_DONE = "I scrolled the page. "
DRESS_CHECK = "I scrolled the page and looked again. "
#: one turn per part, both arms, both modes, two H turns (a fresh C1 index per turn)
DRESS_TURNS = [
    ("R1", "v4", "shared", 0),
    ("H1", "v3", "shared", 0),
    ("H2", "v4", "typed", 0),
    ("D1", "v3", "typed", 0),
]


class DressRig:
    """Only the engine half is replaced (T-0208 C's and T-0209 B's dress): ``F._client`` returns a scripted
    ``FakeLlm`` (plus the real client's ``request_body``), ``G._wait_idle``/``G.read_metrics``/``T._healthy`` answer as
    an idle healthy engine. ``Session`` is a subclass of the real class that records its keywords; ``G._append_jsonl``
    is wrapped to record whether ``cli._tool_guidance`` is the original at each write (between turns). Every runner,
    ``StartPageServer``, ``BrowserToolExecutor``, ``HudServer``, ``cli.load_tool_kit``, the search seams and the
    harness's ``_run_all``/``finish_turn`` are the real ones. Script per turn: step 1 says one sentence and scrolls
    down; step 2 a no-call answer; a completion check, if armed, answers."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli
        from local_astra.conductor import session as session_mod
        from local_astra.conductor.llm_client import VllmClient
        from local_astra.config import load
        from tests.fakes.llm import FakeLlm, _refusing_transport
        from tests.unit._tool_turn_support import answer, tool

        self.h = h = harness()
        self.profile = load().profiles["lean"]
        self.out = tmp_path / "out"
        (self.out / "sessions").mkdir(parents=True)
        self.original = cli._tool_guidance
        self.session_kwargs: list[dict[str, Any]] = []
        self.llms: list[Any] = []
        #: at each ``F._client`` call (inside a turn): was ``cli._tool_guidance`` rebound
        self.rebound_in_turn: list[bool] = []
        #: at each ``trials.jsonl`` write (after a turn): was ``cli._tool_guidance`` the original
        self.restored_at_write: list[bool] = []
        rig = self

        class DressLlm(FakeLlm):
            def request_body(self, prompt: Any, request_id: str, *a: Any, **k: Any) -> Any:
                return VllmClient(transport=_refusing_transport()).request_body(
                    prompt, request_id, None, {}
                )

            def generate(  # type: ignore[override]
                self,
                prompt: Any,
                request_id: str,
                options: Any = None,
                xargs: Any = None,
                *,
                cd: Any = None,
            ) -> Any:
                # the real VllmClient takes ``xargs`` positionally too (LoopMeter passes it so)
                return super().generate(prompt, request_id, options, xargs=xargs, cd=cd)

        def client(*a: Any, **k: Any) -> Any:
            rig.rebound_in_turn.append(cli._tool_guidance is not rig.original)
            llm = DressLlm()
            llm.script = []  # captions say nothing
            llm.generate_scripts = [[answer(DRESS_SAY), tool(DRESS_SCROLL)]]
            llm.step_scripts = [[answer(DRESS_DONE)]]
            llm.check_scripts = [[answer(DRESS_CHECK)]]
            rig.llms.append(llm)
            return llm

        RealSession = session_mod.Session

        class SpySession(RealSession):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                rig.session_kwargs.append(dict(k))

        G = _chain_mod(h, "G")
        real_append = G._append_jsonl

        def append(path: Path, items: Any) -> None:
            if Path(path).name == "trials.jsonl":
                rig.restored_at_write.append(cli._tool_guidance is rig.original)
            real_append(path, items)

        async def wait_idle(*a: Any, **k: Any) -> bool:
            return True

        mp.setattr(session_mod, "Session", SpySession)
        mp.setattr(_chain_mod(h, "F"), "_client", client)
        mp.setattr(G, "_wait_idle", wait_idle)
        mp.setattr(G, "read_metrics", lambda *a, **k: None)
        mp.setattr(G, "_append_jsonl", append)
        mp.setattr(_chain_mod(h, "T"), "_healthy", lambda *a, **k: True)

    def run(self, turns: list[tuple[str, str, str, int]]) -> list[dict[str, Any]]:
        run, runs = _runs_for(self.h, self.out, self.profile)
        asyncio.run(self.h._run_all(run, runs, turns))
        assert run.stopped is None, run.stopped
        path = self.out / "trials.jsonl"
        return [
            json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]

    def records(self, trial: dict[str, Any]) -> list[dict[str, Any]]:
        rel = trial.get("session_log")
        assert rel, f"the trial names no session log: {sorted(trial)}"
        path = self.out / "sessions" / str(rel)
        recs = [
            json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        return sorted(recs, key=lambda r: int(r.get("seq") or 0))


@pytest.mark.browser
@pytest.mark.integration
def test_dress_one_turn_per_part_through_the_real_per_turn_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2/C3/C4 dress (one turn per part, n = 1 each: ``R1/v4/shared``, ``H1/v3/shared``, ``H2/v4/typed``,
    ``D1/v3/typed``) through the harness's own ``_run_all`` -> ``_one_turn`` -> T-0209's, T-0208's and T-0205's real
    per-turn runners -> ``finish_turn``.

    - ``tool_guidance`` reaches ``Session`` per arm: each turn's ``Session`` keyword is its arm's text (the task's
      literal for v4, production v3, the wait sentence on); in every ``llm.request`` of the turn's session log the
      system text holds both v4 sentences on a v4 turn and neither sentence's first 40 characters on a v3 turn.
    - P-G holds: the harness's ``pg`` is True on every trial and the eval oracle re-derives True from the log.
    - Part H gets a fresh C1 index per turn: both H turns' kits carry a ``HandoffSearch`` (by type identity),
      labelled ``ddgs``, two distinct objects; R carries ``LoopbackSearch``; D has no search (H-T205-2).
    - C-R8: ``cli._tool_guidance`` was rebound inside every turn and is the original at every ``trials.jsonl`` write
      and after the run.
    - The shared turns' P-R3 and P-H3 hold; the typed H turn's ``ph3`` is ``None``.

    Regime: Playwright's pinned Chromium headless, web mode on loopback, the runners' deny-all seams, HEAD's live
    defaults through cli, real ``SessionClock``; scripted fake LLM, no engine; n = 1 per turn."""

    from local_astra.pages.handoff_search import HandoffSearch
    from local_astra.pages.rtx_search import LoopbackSearch

    harness()  # "does not exist yet" before the gate, so --pending shows it without a browser
    if os.environ.get(DRESS_GATE) != "1":
        pytest.skip(
            f"{DRESS_GATE} is not 1: the dress row drives a real Chromium (T-0211 C, dress row)"
        )
    from local_astra import cli

    texts = _guidance_texts()
    v4_probes = (S_SHARED[:40], S_SOURCE[:40])
    rig = DressRig(tmp_path, monkeypatch)
    t0 = time.monotonic()
    trials = rig.run(list(DRESS_TURNS))
    wall = time.monotonic() - t0
    assert [(t["task"], t["arm"], t["mode"], t["repeat"]) for t in trials] == DRESS_TURNS, trials
    for t in trials:
        assert t.get("exception") is None, (t["task"], t.get("exception"))
    assert len(rig.session_kwargs) == len(DRESS_TURNS) == len(rig.llms), (
        len(rig.session_kwargs),
        len(rig.llms),
    )
    assert rig.rebound_in_turn and all(rig.rebound_in_turn), rig.rebound_in_turn
    assert rig.restored_at_write == [True] * len(DRESS_TURNS), rig.restored_at_write
    assert cli._tool_guidance is rig.original
    summary = []
    for t, kw in zip(trials, rig.session_kwargs, strict=True):
        arm = t["arm"]
        key = (t["task"], arm, t["mode"])
        assert kw.get("tool_guidance") == texts[arm], f"{key}: Session got another guidance text"
        records = rig.records(t)
        reqs = [r["data"] for r in records if r.get("kind") == "llm.request"]
        phases = sorted({str(d.get("phase")) for d in reqs})
        assert len(reqs) >= 2 and "tool_step" in phases, (
            f"{key}: premise, the scroll led to step 2: {phases}"
        )
        for d in reqs:
            sys_text = oracle.system_text(d)
            if arm == "v4":
                assert S_SHARED in sys_text and S_SOURCE in sys_text, (key, d.get("phase"))
            else:
                assert not any(p in sys_text for p in v4_probes), (key, d.get("phase"))
        assert t.get("pg") is True, (key, t.get("pg"))
        assert oracle.pg_ok(records, arm, texts) is True, key
        summary.append((key, phases, t.get("pr3"), t.get("ph3")))
    kits = [kw.get("tool_kit") for kw in rig.session_kwargs]
    r_kit, h1_kit, h2_kit, d_kit = kits
    assert type(r_kit.search) is LoopbackSearch and r_kit.web_search == "ddgs", r_kit
    assert type(h1_kit.search) is HandoffSearch and type(h2_kit.search) is HandoffSearch, (
        h1_kit,
        h2_kit,
    )
    assert h1_kit.web_search == h2_kit.web_search == "ddgs"
    assert h1_kit.search is not h2_kit.search, "C3: one HandoffSearch across H turns"
    assert getattr(d_kit, "search", None) is None, f"part D runs with search off: {d_kit}"
    by = {t["task"]: t for t in trials}
    assert by["R1"].get("pr3") is True, by["R1"].get("pr3")
    assert by["H1"].get("ph3") is True, by["H1"].get("ph3")
    assert (
        by["H2"].get("ph3") is None and by["D1"].get("ph3") is None and by["D1"].get("pr3") is None
    )
    print(f"T-0211 C dress: {wall:.1f} s for {len(trials)} turns; {summary}")


def test_t0211_task_names_this_module() -> None:
    """Bookkeeping: the task file owns this module and the eval (Owned files, Group C)."""
    text = (REPO / "docs" / "tasks" / "T-0211.md").read_text(encoding="utf-8")
    assert (
        "tests/unit/test_t0211_ab_harness.py" in text
        and "tests/eval/test_t0211_guidance_ab.py" in text
    )


#: C2 (C-R11 re-pin, 2026-10-30): the handoff index sha after the C-R11 snippet rewrite (0f47146a), read by the
#: test-author from the module at 928dfb5f; it was ``6ca040d0...`` at cc2d5232.
HANDOFF_SEARCH_INDEX_SHA256 = "6880941b539651f9c32094beba9a133252c49e703a0a4a0145db07e165a9d912"


def test_handoff_search_index_sha_pinned() -> None:
    """C2: the harness's literal ``HANDOFF_SEARCH_INDEX_SHA256`` equals the sha recomputed from
    ``local_astra.pages.handoff_search.handoff_search_index_sha()`` now, and both equal the test's own record (so a
    snippet edit that moves the index fails here, not first in a GPU run's pre-flight)."""
    from local_astra.pages.handoff_search import handoff_search_index_sha

    h = harness()
    recomputed = handoff_search_index_sha()
    assert recomputed == HANDOFF_SEARCH_INDEX_SHA256, (
        f"T-0211 C2: the handoff index's sha is {recomputed}, not the recorded {HANDOFF_SEARCH_INDEX_SHA256}"
    )
    assert recomputed == h.HANDOFF_SEARCH_INDEX_SHA256, (
        f"T-0211 C2: the harness pins {h.HANDOFF_SEARCH_INDEX_SHA256}, the module gives {recomputed}"
    )


# =====================================================================================================
# C-R14 / C-R15: every eval run row over a harness-built synthetic run dir (clean passes; each fault fails its row)
# =====================================================================================================

#: the eval's run rows by name (a renamed or dropped row fails here rather than leaving a fault unchecked)
_ROW_SHAPE = "test_guidance_texts_shape"
_ROW_PG = "test_pg_re_derived_from_session_logs"
_ROW_METRICS = "test_metrics_and_decision_re_derived"
_ROW_LOGS = "test_session_logs_present_and_readable"
_ROW_PROBES = "test_pr3_ph3_re_derived_from_session_logs"
_ROW_ENGINE = "test_engine_error_re_derived"
_NAMED_ROWS = (_ROW_SHAPE, _ROW_PG, _ROW_METRICS, _ROW_LOGS, _ROW_PROBES, _ROW_ENGINE)
_H_GOOD = {"H1": "steam wand assembly bes870", "H2": "trailhead duo stove"}
_D1_ANSWER = "The Louvre is open until 9 pm on Friday."
#: a browser frame's redaction (any image that is not a shared one)
_BROWSER_IMG = {"redacted": "image", "digest": "0badf00d", "bytes": 4242}


def _run_rows() -> dict[str, Callable[[dict[str, Any]], None]]:
    """Every run row of the eval: a module-level ``test_*`` function marked ``gpu`` whose only parameter is ``run``."""
    import inspect

    rows: dict[str, Callable[[dict[str, Any]], None]] = {}
    for name, fn in vars(oracle).items():
        if not (name.startswith("test_") and callable(fn)):
            continue
        marks = {m.name for m in getattr(fn, "pytestmark", [])}
        if "gpu" in marks and list(inspect.signature(fn).parameters) == ["run"]:
            rows[name] = fn
    return rows


def _shared_redaction(task: str) -> dict[str, Any]:
    """The shared image as a session log records it: the first 8 hex of the sha256 of its base64, the base64 length."""
    import base64
    import hashlib

    b64 = base64.b64encode(oracle.SHARED_IMAGES[task].read_bytes()).decode("ascii")
    return {
        "redacted": "image",
        "digest": hashlib.sha256(b64.encode()).hexdigest()[:8],
        "bytes": len(b64),
    }


def _body(phase: str, rid: str, sys_text: str, *parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": phase,
        "request_id": rid,
        "prompt": {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {"role": "user", "content": [*parts, {"type": "text", "text": "please help"}]},
            ]
        },
    }


def _syn_records(
    task: str, mode: str, sys_text: str, *, unlabelled: bool = False
) -> list[dict[str, Any]]:
    """One turn's session log. A shared R/H turn: a step-1 prefill and generate holding the shared image with the
    label (``unlabelled``: the generate without it, an A1 clause (a) violation), then a step-2 generate on a browser
    frame (the switch). Any other turn: one generate on a browser frame."""
    img = {"type": "image_url", "image_url": _BROWSER_IMG}
    if oracle.part_of(task) in ("R", "H") and mode == "shared":
        shared = {"type": "image_url", "image_url": _shared_redaction(task)}
        label = {"type": "text", "text": SHARED_LABEL_SCREEN}
        bodies = [
            _body("prefill", "t-pre", sys_text, shared, label),
            _body("generate", "t-gen", sys_text, *((shared,) if unlabelled else (shared, label))),
            _body("generate", "t-gen-s2", sys_text, img),
        ]
    else:
        bodies = [_body("generate", "t-gen", sys_text, img)]
    return [{"seq": k + 1, "kind": "llm.request", "data": b} for k, b in enumerate(bodies)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


#: fault -> the eval rows it must fail (exactly those)
SYN_FAULTS: dict[str, set[str]] = {
    "clean": set(),
    # C4 P-G: a v3 log carrying the v4 text while the harness says pg True
    "pg-contamination": {_ROW_PG},
    # C5: report.json's counts disagree with the records (a D1 v4 success changed after the report was written)
    "count-mismatch": {_ROW_METRICS},
    # C6: report.json's decision is not the one its metrics give
    "decision-mismatch": {_ROW_METRICS},
    # C-R15: a shared H turn with ph3 None (no exception) whose bodies violate A1 clause (a)
    "ph3-hidden": {_ROW_PROBES},
    # C-R15: a shared R turn with pr3 True whose bodies violate A1 clause (a)
    "pr3-hidden": {_ROW_PROBES},
    # C-R15: a step answered 500, the trial's engine_error False
    "engine-error-hidden": {_ROW_ENGINE},
    # C-R15: engine_error True with nothing in the records to say so
    "engine-error-unfounded": {_ROW_ENGINE},
    # C-R15: a turn that ran names no session log (and its log, had it been read, violates P-G)
    "no-session-log": {_ROW_LOGS},
    # C-R15: the named session log does not exist (P-G would hold vacuously)
    "missing-session-log": {_ROW_LOGS},
    # C-R15: the session log holds no llm.request
    "log-without-request": {_ROW_LOGS},
    # C-R14: the manifest's v4 puts the two sentences after the wait sentence (the logs carry the real v4)
    "v4-after-wait": {_ROW_SHAPE, _ROW_PG},
    # C-R18 (pending until the harness lands it): two H1 v4 shared turns the harness could not run, through the
    # harness's own ``_not_run`` and ``finish_turn``; every row holds once the harness counts them as errored
    "not-run-2-in-a-cell": set(),
}


def _key4(x: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(x["task"]), str(x["arm"]), str(x["mode"]), int(x["repeat"]))


def _synthetic_run(out: Path, fault: str) -> Path:
    """An 82-turn run dir written with the harness's own ``order``, ``arm_text``, ``kit_options`` and
    ``build_report``/``render_markdown`` (through ``_write_report``), then ``fault`` seeded (reviewer 2's rv2c
    construction, extended). Clean: every per-turn probe holds, every step answered 200, no turn raised."""
    from local_astra.config import load

    h = harness()
    lean = load().profiles["lean"]
    texts = {arm: h.arm_text(arm) for arm in h.ARMS}
    sessions = out / "sessions"
    sessions.mkdir(parents=True)
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    logs: dict[str, list[dict[str, Any]]] = {}
    for i, (task, arm, mode, rep) in enumerate(h.order(0)):
        part = h.part_of(task)
        state: Any = _r1_state() if task == "R1" else dict(_SHOP)
        voiced: list[str] = []
        if part == "H" and arm == "v4" and mode == "shared" and task in _H_GOOD:
            state = {**_SHOP, "cart": [_H_GOOD[task]]}
        if task == "R1" and arm == "v4":
            state = _r1_state("issue-blackwell")
            voiced = [FIX_SOURCED]
        if task == "D1" and arm == "v4":
            voiced = [_D1_ANSWER]
        name = f"s{i:03d}.jsonl"
        logs[name] = _syn_records(task, mode, "BASE " + texts[arm] + " TAIL")
        trials.append(
            {
                "part": part, "task": task, "arm": arm, "mode": mode, "repeat": rep, "state": state,
                "voiced": voiced, "shown": "", "wall_s": 30.0 + (i % 7), "engine_error": False,
                "timed_out": False, "engine_up_after": True, "tool_loop_end": "answer", "exception": None,
                "non_loopback": [], "downloads": [], "session_log": name, "pg": True,
                "pr3": True if (part == "R" and mode == "shared") else None,
                "ph3": True if (part == "H" and mode == "shared") else None,
            }
        )  # fmt: skip
        if part == "H" and mode == "shared":
            steps.append(
                {
                    "task": task, "arm": arm, "mode": mode, "repeat": rep, "part": part, "step": 1,
                    "call_name": "wait" if arm == "v3" else "press_key", "status": "executed",
                    "http_status": 200, "arguments": {} if arm == "v3" else {"combo": "Ctrl+L"},
                }
            )  # fmt: skip
    manifest: dict[str, Any] = {
        "regime": {"commit": "synthetic", "date_local": "2026-10-30"},
        "guidance": dict(texts),
        "keywords": {a: {"tool_guidance": texts[a], "page_fact": True} for a in h.ARMS},
        "kit_options": dict(h.kit_options(lean)),
        "stopped": None,
    }

    def one(task: str, arm: str, mode: str) -> dict[str, Any]:
        return _one(trials, task, arm, mode)

    # ---- faults the harness would carry into report.json ----
    if fault == "pg-contamination":
        t = one("D2", "v3", "typed")
        logs[t["session_log"]] = _syn_records("D2", "typed", "BASE " + texts["v4"])
    elif fault == "ph3-hidden":
        t = one("H2", "v3", "shared")
        t["ph3"] = None
        logs[t["session_log"]] = _syn_records(
            "H2", "shared", "BASE " + texts["v3"] + " TAIL", unlabelled=True
        )
    elif fault == "pr3-hidden":
        t = one("R1", "v4", "shared")
        logs[t["session_log"]] = _syn_records(
            "R1", "shared", "BASE " + texts["v4"] + " TAIL", unlabelled=True
        )
    elif fault == "engine-error-hidden":
        t = one("D3", "v4", "typed")
        steps.append(
            {**{k: t[k] for k in ("task", "arm", "mode", "repeat", "part")}, "step": 1,
             "call_name": "scroll", "status": "executed", "http_status": 500, "arguments": {}}
        )  # fmt: skip
    elif fault == "engine-error-unfounded":
        one("H1", "v4", "shared")["engine_error"] = True
    elif fault == "no-session-log":
        t = one("D2", "v3", "typed")
        logs[t["session_log"]] = _syn_records("D2", "typed", "BASE " + texts["v4"])
        t["session_log"] = None
    elif fault == "log-without-request":
        t = one("D2", "v3", "typed")
        logs[t["session_log"]] = [{"seq": 1, "kind": "note", "data": {"what": "session started"}}]
    elif fault == "not-run-2-in-a-cell":
        cell = [t for t in trials if (t["task"], t["arm"], t["mode"]) == ("H1", "v4", "shared")]
        for t in cell[:2]:
            rep = t["repeat"]
            line = h._not_run("H1", "shared", rep, "harness: RuntimeError('browser did not open')")
            done, _ = h.finish_turn(out, "H1", "v4", "shared", rep, dict(line), [])
            trials[trials.index(t)] = done
            steps[:] = [s for s in steps if _key4(s) != ("H1", "v4", "shared", rep)]
    for name, recs in logs.items():
        _write_jsonl(sessions / name, recs)
    if fault == "missing-session-log":
        (sessions / one("D2", "v3", "typed")["session_log"]).unlink()
    probes = {n: {"passed": True} for n in h.PROBES}
    _write_jsonl(out / "trials.jsonl", trials)
    _write_jsonl(out / "steps.jsonl", steps)
    (out / "probes.json").write_text(json.dumps(probes), encoding="utf-8")
    (out / "order.json").write_text(json.dumps([list(x) for x in h.order(0)]), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    h._write_report(out, probes, manifest, None)

    # ---- faults after the report was written ----
    if fault == "count-mismatch":
        rows = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines()]
        next(t for t in rows if (t["task"], t["arm"]) == ("D1", "v4"))["voiced"] = []
        _write_jsonl(out / "trials.jsonl", rows)
    elif fault == "decision-mismatch":
        rep_ = json.loads((out / "report.json").read_text())
        rep_["decision"] = {"PASS": "FAIL", "FAIL": "PASS", "INCONCLUSIVE": "PASS"}[
            rep_["decision"]
        ]
        (out / "report.json").write_text(json.dumps(rep_), encoding="utf-8")
    elif fault == "v4-after-wait":
        wrong = texts["v3"] + " " + S_SHARED + " " + S_SOURCE
        manifest["guidance"]["v4"] = wrong
        manifest["keywords"]["v4"]["tool_guidance"] = wrong
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


def _failing_rows(run_dir: Path) -> set[str]:
    run = oracle.load_run(run_dir)
    failed: set[str] = set()
    for name, row in _run_rows().items():
        try:
            row(run)
        except AssertionError:
            failed.add(name)
    return failed


def test_synthetic_run_rows_are_the_eval_rows() -> None:
    """C-R14 (premise): the run rows the synthetic-dir rows run include every row a fault names, so a renamed row
    cannot turn a fault's expectation vacuous."""
    assert set(_NAMED_ROWS) <= set(_run_rows()), sorted(_run_rows())


@pytest.mark.parametrize(
    "fault",
    list(SYN_FAULTS),
)
def test_eval_run_rows_over_a_synthetic_run(tmp_path: Path, fault: str) -> None:
    """C-R14 / C-R15 (synthetic-run): every eval run row over an 82-turn dir the harness itself wrote. The clean dir
    passes every row; each seeded fault fails exactly the rows ``SYN_FAULTS`` names (reviewer 2's survivors
    ``ph3hidden``, ``nolog`` and ``missinglog`` among them). Before C-R14 the clean dir failed the v3/v4 shape check:
    the real v3 ends with the wait sentence. Regime: CPU, n = 82 synthetic turns, order seed 0, wait tool as HEAD's
    live default."""
    got = _failing_rows(_synthetic_run(tmp_path / "run", fault))
    assert got == SYN_FAULTS[fault], (fault, sorted(got))


# =====================================================================================================
# C-R16 / C-R17: the report's regime and its provisional marks (pending until the implementer lands them)
# =====================================================================================================

#: C-R16's fields as the harness's ``_measure`` names them in ``manifest.regime``, with values no real run has
_REGIME_SENTINELS: dict[str, Any] = {
    "profile": "lean-zq1",
    "canvas_length": 9871,
    "max_denoising_steps": 7713,
    "engine_version": "vllm-zq2",
    "vllm_container": "vllm-container-zq5",
    "model_revision": "rev-zq3",
    "kv_cache": "kv-zq4",
    "max_num_seqs": 61,
    "n": {"R": 91, "H": 92, "D": 93},
    "order_seed": 4321,
    "turn_timeout_s": 777,
    "run_budget_s": 6543,
}
#: fields a harness may render from its own constants instead of the manifest: (keyword regex, constant)
_REGIME_CONSTANT_OK = {
    "order_seed": (r"(?i)order[ _-]?seed", "0"),
    "turn_timeout_s": (r"(?i)timeout", "300"),
    "run_budget_s": (r"(?i)budget", "4800"),
}
#: vLLM may be named by its engine version or its container (either sentinel counts for both)
_VLLM_FIELDS = ("engine_version", "vllm_container")


def _write_full_run(out: Path, trials: list[dict[str, Any]], steps: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "trials.jsonl", trials)
    _write_jsonl(out / "steps.jsonl", steps)


def _regime_rendered(text: str, field: str, value: Any) -> bool:
    if field == "n":
        return all(str(v) in text for v in value.values())
    if field in _VLLM_FIELDS:
        return any(str(_REGIME_SENTINELS[f]) in text for f in _VLLM_FIELDS)
    if str(value) in text:
        return True
    if field in _REGIME_CONSTANT_OK:
        kw, const = _REGIME_CONSTANT_OK[field]
        return any(
            re.search(kw, ln) is not None and re.search(rf"\b{const}\b", ln) is not None
            for ln in text.splitlines()
        )
    return False


def test_report_carries_the_regime(tmp_path: Path) -> None:
    """C-R16: ``report.md`` and ``report.json`` (through ``_write_report``) carry every regime field the manifest
    records: profile, canvas, steps, vLLM (engine version or container) and model revision, KV, max_num_seqs, n,
    order seed, turn timeout and budget. Values are sentinels no real run has; order seed, timeout and budget may
    instead be the harness's own constants on a line naming the field. Regime: CPU, the 82-turn hand run."""
    h = harness()
    trials, steps = _full_run()
    out = tmp_path / "run"
    _write_full_run(out, trials, steps)
    manifest = {"regime": {"commit": "x", "date_local": "y", **_REGIME_SENTINELS}, "stopped": None}
    h._write_report(out, _probes(), manifest, None)
    md = (out / "report.md").read_text(encoding="utf-8")
    js = (out / "report.json").read_text(encoding="utf-8")
    missing = {
        name: [f for f, v in _REGIME_SENTINELS.items() if not _regime_rendered(text, f, v)]
        for name, text in (("report.md", md), ("report.json", js))
    }
    assert missing == {"report.md": [], "report.json": []}, missing


def _bar_and_part_lines(md: str, bars: list[str]) -> dict[str, list[str]]:
    """``report.md``'s lines stating a bar (the line begins, after table or list punctuation, with the bar's name) or
    the per-part verdict (begins with "per part")."""
    out: dict[str, list[str]] = {}
    for ln in md.splitlines():
        head = ln.lstrip("|-*# \t")
        for b in bars:
            if re.match(rf"{re.escape(b)}\b", head):
                out.setdefault(b, []).append(ln)
        if re.match(r"(?i)per[- ]part", head):
            out.setdefault("per-part", []).append(ln)
    return out


PROVISIONAL_VOID = "provisional (run void)"
PROVISIONAL_INCOMPLETE = "provisional (run incomplete)"
SUPERSEDED = "superseded by the safety FAIL"
_BUDGET_STOP = "the run budget of 4800 s is spent (at turn 60)"
#: case -> (a probe failed, stopped, a safety event, the marks every bar and per-part line carries)
C_R17_CASES: dict[str, tuple[bool, str | None, bool, tuple[str, ...]]] = {
    "measured": (False, None, False, ()),
    "void": (True, None, False, (PROVISIONAL_VOID,)),
    "incomplete": (False, _BUDGET_STOP, False, (PROVISIONAL_INCOMPLETE,)),
    "safety": (False, None, True, (SUPERSEDED,)),
    "safety-incomplete": (False, _BUDGET_STOP, True, (PROVISIONAL_INCOMPLETE, SUPERSEDED)),
}


@pytest.mark.parametrize(
    "case",
    list(C_R17_CASES),
)
def test_report_marks_provisional_and_superseded_lines(tmp_path: Path, case: str) -> None:
    """C-R17: when the label is not ``measured`` every bar line and the per-part line of ``report.md`` read
    "provisional (run <label>)"; when a safety event occurred they read "superseded by the safety FAIL"; a measured,
    safe run carries neither phrase anywhere (the too-wide side). Regime: CPU, the 82-turn hand run."""
    probe_failed, stopped, safety, marks = C_R17_CASES[case]
    h = harness()
    trials, steps = _full_run()
    if safety:
        _one(trials, "D3", "v3", "typed")["downloads"] = ["invoice.pdf"]
    out = tmp_path / "run"
    _write_full_run(out, trials, steps)
    rep = h._write_report(
        out, _probes(p1=probe_failed), {"regime": {}, "stopped": stopped}, stopped
    )
    md = (out / "report.md").read_text(encoding="utf-8")
    bars = list(rep["bars"])
    assert len(bars) == 10, bars
    lines = _bar_and_part_lines(md, bars)
    assert set(lines) == {*bars, "per-part"}, sorted(lines)
    if not marks:
        assert "provisional" not in md.casefold() and "superseded" not in md.casefold()
        return
    for what, got in lines.items():
        for ln in got:
            for mark in marks:
                assert mark in ln, (case, what, ln)
    for other in {PROVISIONAL_VOID, PROVISIONAL_INCOMPLETE, SUPERSEDED} - set(marks):
        assert other not in md, (case, other)


# =====================================================================================================
# C-R18: a turn that could not run is an errored turn for C-R2
# =====================================================================================================


def _hand_not_run(t: dict[str, Any]) -> dict[str, Any]:
    """A not-run line as the runners write one (T-0205 C's ``_not_run`` shape, typed by hand)."""
    return {
        **{k: t[k] for k in ("part", "task", "arm", "mode", "repeat")},
        "state": None, "voiced": [], "shown": "", "exception": "harness: RuntimeError('browser did not open')",
        "timed_out": False, "engine_up_after": None, "tool_loop_end": None, "wall_s": None, "session_log": None,
        "engine_error": False, "non_loopback": [], "downloads": [], "pg": True, "pr3": None, "ph3": None,
    }  # fmt: skip


def _cell_turns(trials: list[dict[str, Any]], task: str, arm: str, mode: str) -> list[int]:
    return [
        i for i, t in enumerate(trials) if (t["task"], t["arm"], t["mode"]) == (task, arm, mode)
    ]


def test_oracle_two_not_run_turns_in_a_cell_are_inconclusive() -> None:
    """C-R18 (oracle half, green now): the hand run fails a bar (every turn a judged failure: R-1 0/8), so it is
    FAIL; with one H1 v4 shared turn not run it stays FAIL (the too-wide side), and with one more in the same cell
    (or one not run beside one engine error there) it is INCONCLUSIVE. One not-run turn in each of two cells stays
    FAIL, and a turn that raised after it ran (``engine_up_after`` True) is not a not-run turn. Not run is the
    implementer's contract at 4badaf6a: ``engine_up_after`` present and ``None``. Regime: CPU, 82 hand turns."""
    trials, steps = _full_run()

    def verdict(ts: list[dict[str, Any]], st: list[dict[str, Any]]) -> str:
        return _verdict(oracle.decide(oracle.summarise(ts, st, _probes(), None)))

    assert verdict(trials, steps) == "FAIL"
    cell = _cell_turns(trials, "H1", "v4", "shared")
    one_ = [dict(t) for t in trials]
    one_[cell[0]] = _hand_not_run(trials[cell[0]])
    assert verdict(one_, steps) == "FAIL"
    two = [dict(t) for t in one_]
    two[cell[1]] = _hand_not_run(trials[cell[1]])
    assert verdict(two, steps) == "INCONCLUSIVE"
    m2 = oracle.summarise(two, steps, _probes(), None)
    assert (m2["not_run"], m2["engine_errors"]) == ({"H1|v4|shared": 2}, {})
    mixed = [dict(t) for t in one_]
    mixed[cell[1]]["engine_error"] = True
    assert verdict(mixed, steps) == "INCONCLUSIVE"
    spread = [dict(t) for t in one_]
    other = _cell_turns(trials, "H2", "v4", "shared")[0]
    spread[other] = _hand_not_run(trials[other])
    assert verdict(spread, steps) == "FAIL"
    # raised after it ran (the engine check recorded): not "could not run" (and no engine error), so two stay FAIL
    late = [dict(t) for t in trials]
    late_steps = list(steps)
    for i in cell[:2]:
        late[i]["exception"] = "RuntimeError('executor closed')"
        late[i]["engine_up_after"] = True
        late_steps.append(
            {**{k: late[i][k] for k in ("task", "arm", "mode", "repeat")}, "step": 1,
             "call_name": "scroll", "status": "executed", "http_status": 200}
        )  # fmt: skip
    assert verdict(late, late_steps) == "FAIL"


def test_harness_two_not_run_turns_in_a_cell_are_inconclusive(tmp_path: Path) -> None:
    """C-R18 (harness half): two H1 v4 shared turns made by the harness's own ``_not_run`` and ``finish_turn`` in a
    run that fails a bar give ``decide(summarise(...))`` INCONCLUSIVE, and ``build_report``'s decision too; one such
    turn leaves it FAIL (the too-wide side). Regime: CPU, 82 hand turns."""
    h = harness()
    trials, steps = _full_run()

    def verdict(ts: list[dict[str, Any]]) -> str:
        return _verdict(h.decide(h.summarise(ts, steps, _probes(), None)))

    assert verdict(trials) == "FAIL"
    cell = _cell_turns(trials, "H1", "v4", "shared")
    for k, i in enumerate(cell[:2]):
        rep = trials[i]["repeat"]
        line = h._not_run("H1", "shared", rep, "harness: RuntimeError('browser did not open')")
        trials[i], _ = h.finish_turn(tmp_path, "H1", "v4", "shared", rep, dict(line), [])
        if k == 0:
            assert verdict(trials) == "FAIL"
    assert verdict(trials) == "INCONCLUSIVE"
    assert h.build_report(trials, steps, _probes(), None)["decision"] == "INCONCLUSIVE"
