"""T-0209 group B: the loopback GPU run's harness (pure half, refusals, manifest, the two arms, a fresh browser per
turn) and its locked wrapper (unit; CPU; no engine, no socket). docs/tasks/T-0209.md B1-B7, "Tests to write" (B),
the Regime (B) and the Status line's R1-J5 ("each turn gets a fresh browser context, so ``visited`` never carries
over between turns").

BASE feab84c1 (main, ``T-0209 A1 merged``; Merge T-0209-a1 fb8db509, Merge T-0207-c 13a71c12, Merge T-0208-b
43eb2f9a, Merge T-0208-a2 60deb9c4 are ancestors; Merge T-0209-a2 and Merge T-0208-c are not). B runs **R1 only**
(A2 absent; "If A2 slips ... B and C run R1 only. The manifest says so").

**Activation.** The rows that need ``scripts/gpu/t0209_rtx_run.py`` or ``scripts/locked/t0209_rtx_run.sh`` carried
``pending(task="T-0209")`` until the implementation at 796c110f; the markers are removed (activation). A missing file
is a named assertion in ``harness()``/``api()``/``_locked()`` ("does not exist yet" / "has no ..."), never an import
error (every harness symbol is looked up inside the row). The pure rows also run against the eval's oracle
(``tests/eval/test_t0209_rtx_run.py``, ``eval-oracle``, not pending), so a wrong oracle is caught on CPU now. The
premises (production constants, the hand data under the real judge, frozen T-0205 files) are green at BASE. Every
expected value is computed by hand (comments say how) or comes from production code the rule does not own.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, as
T-0208 C's, so the implementer can implement to them or say why not).

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``ARMS = ("shared", "typed")``; ``REPEATS = 3``;
    ``ORDER_SEED = 0``; ``TURN_TIMEOUT_S = 300``; ``RUN_BUDGET_S = 40 * 60``; ``RTX_TABLE_SHA256``,
    ``SEARCH_INDEX_SHA256`` (the literals); ``SHARED_FRAME_REASON``; ``CANNOT_SEE`` (T-0208 C5's pattern); ``VOID``,
    ``INCOMPLETE``, ``MEASURED``; ``RTX_SHARED_DIR`` (the image directory, read at call time).

Pure functions (both the harness and the oracle; shapes in the oracle's docstrings)
    ``order(seed, r2=False) -> [(rtx_id, arm, repeat)]``; ``pr1_ok(record, r2=False)``, ``pr2_ok(record)``;
    ``pr3_ok(records, image_bytes)``; ``identity(trial, steps)``; ``step1_action(steps)``; ``shared_refusals(steps)``;
    ``cannot_see(trial)``; ``trap_kinds(answer)``; ``build_report(trials, steps, probes, stopped=None, r2=False) ->
    {"label", "stopped", "probes", "per_arm": {arm: {rtx_id: counts}}}``; ``session_keywords(profile)``.

The search seam (B1/B2/B3, H-T209-1)
    ``rtx_kit(kit) -> kit``: the kit ``cli.load_tool_kit`` gave, with ``search`` a fresh
    ``local_astra.pages.rtx_search.LoopbackSearch()`` and ``web_search="ddgs"`` (``dataclasses.replace``);
    ``kit_refusal(kit) -> str | None``: ``None`` only when ``type(kit.search)`` **is** production's ``LoopbackSearch``
    and ``kit.web_search == "ddgs"``; ``main`` and ``_one_turn`` both look ``rtx_kit`` and ``kit_refusal`` up on the
    module at call time.

Seams ``main`` and ``_one_turn`` read at call time
    ``_git``; every ``cli`` resolver, ``cli.load_tool_kit``, ``cli.live_prerequisites``; T-0205's chain ``F``/``G``/
    ``T``/``P`` (by name on the harness or on the T-0205 module it holds); ``run_probes``, ``probe_pr1()``,
    ``probe_pr2(pr1)``; ``_one_turn(run, rtx_id, arm, repeat) -> (trial, steps, thoughts)`` over ``_Run``;
    ``HudServer``, ``BrowserFrameSource``, ``Session``, ``StartPageServer``, ``BrowserToolExecutor``,
    ``TurnMachine``/``TurnPolicy`` imported at call time from their modules. ``main(argv)`` takes ``--base-url``,
    ``--out-dir`` and ``--only R1|R2``.

**Readings** (reported to the orchestrator; not resolved silently)
    1. B2's refusals, all exit 2 before any socket, leaving the out-dir as it was: a non-loopback ``--base-url``, a
       dirty tree, an existing out-dir, no ``Merge T-0209-a1:`` merge reachable from HEAD (``git log --merges``
       subjects; a branch merge ``Merge branch 'main' into T-0209-a1-tests7`` or a ``Merge T-0209-a1b:`` does not
       count), the table off ``RTX_TABLE_SHA256``, the index off ``SEARCH_INDEX_SHA256``, an image off its row's
       ``image_sha256`` (read under ``RTX_SHARED_DIR``), web mode off, guidance not v3, loop not 20, a kit
       ``kit_refusal`` refuses, ``--only R2`` without a ``Merge T-0209-a2:``. ``SEARCH_INDEX_SHA256`` is
       ``b8724480...`` read off ``search_index_sha()`` at BASE (the task pins no index sha; A2 moves it).
    2. ``order`` is scenario-major over the run's ids (R1; R1, R2 with ``r2``) x ``ARMS`` x ``range(3)``, then one
       ``random.Random(0).shuffle`` (T-0208's construction). 6 turns R1-only, 12 with R2.
    3. The ``shared`` arm: one real ``BrowserFrameSource`` is the session's perception and the ``HudServer``'s
       ``frame_sink``; the image is one ``astra.frame`` that ``HudServer.route`` hands to ``ingest``; then one
       ``astra.vision.select {source: "shared"}`` routed to ``Session.apply_vision_select``; then
       ``UserText(request)``. The ``typed`` arm: no frame carrying the image, no ``shared`` select,
       ``UserText(control_request)``. Both: a fresh ``StartPageServer(static_dir=RTX_DIR)``, a fresh web-mode executor
       (deny-all resolver and connector, no ``user_data_dir``) opened at ``/web.html``, a fresh ``Session`` and a fresh
       ``LoopbackSearch`` per turn; the executor is closed at the end of its turn (R1-J5).
    4. P-R3, identity, the trap breakdown, the R1 columns: the eval oracle's readings 4, 6 and 9. P-R3 is T-0208
       P-H3 adapted to a search-first turn (the switch rule); identity counts a ``google_search`` query as typed.
    5. The manifest records ``live_default`` (T-0205's keys plus ``completion_check``), ``shared_check`` (a ``Merge
       T-0208-a2:`` merge reachable), ``r2`` (a ``Merge T-0209-a2:`` merge reachable), ``search_backend:
       "rtx_loopback"``, and the table, index and image shas anywhere in the manifest.
    6. The wrapper accepts ``Merge T-0209-a1:`` only (no other T-0209 or T-0205 merge check); its boot is T-0205
       C's verbatim; it checks only files listed in ``L_PLACEHOLDERS``.

Regime: CPU, pure, hand-built records and a stubbed ``main``/``_one_turn``; no engine; n = 1. The dress row
(``browser`` + ``integration``, ``ASTRA_BROWSER_TESTS=1``) is the only row that opens a browser.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import contextlib
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.telemetry.session_log import redact
from tests.eval import test_t0209_rtx_run as oracle
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0209_rtx_judge as J

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0209_rtx_run.py"
LOCKED = REPO / "scripts" / "locked" / "t0209_rtx_run.sh"
EVAL = REPO / "tests" / "eval" / "test_t0209_rtx_run.py"
T0205_LOCKED = REPO / "scripts" / "locked" / "t0205_demo_run.sh"
IMAGE_DIR = REPO / "local_astra" / "pages" / "rtx_shared"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

ARMS = ("shared", "typed")
#: tests/unit/test_t0209_rtx_judge.py's literal (R1 only)
TABLE_SHA256 = J.TABLE_SHA256_R1_ONLY
#: ``search_index_sha()`` at BASE feab84c1 (reading 1)
INDEX_SHA256 = "b8724480a80ff5b9503fbcec7aeb14a11fbdc517a281cb051e2c14ccea017ad0"
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
MOVED_REASON = "the screen changed since this frame; take a new frame"
L_NOW = "[This screenshot is the browser screen now.]"
CANNOT_SEE = r"can(?:'|no)t see|cannot see|unable to see"
#: the real merge subjects on main (``git log --merges``), and made-up lookalikes
MERGE_A1 = (
    "fb8db509e3e83c8cb2a6e896cd0f3ed27bd2b1cc Merge T-0209-a1: R1 'Why is Blender doing this?' loopback scenario "
    "(pages, loopback search, judge, shared error image)"
)
MERGE_0208_A2 = (
    "60deb9c476d2664d7b7684ad0cb3a56d91e14b52 Merge T-0208-a2: completion check on shared-start turns once the step "
    "shows a browser frame; a wait alone does not arm (R-A2)"
)
MERGE_A2 = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0209-a2: R2 a coding model that fits this card"
MERGE_B = "abcdefabcdefabcdefabcdefabcdefabcdefabcd Merge T-0209-b: the loopback GPU run harness"
OTHERS = (
    "40ed3c38879c23736266dfb65662b4f7167a4c99 Merge branch 'main' into T-0209-a1-tests7",
    "e2087acba419df2cb8729368106a9599a6a0351f Merge branch 'main' into T-0209-a1-impl3",
    "13a71c128133ed76a5edac61579f7a6ff814a74c Merge T-0207-c: completion check on by default in live "
    "(--completion-check off opts out)",
    "43eb2f9a9148f990c99d4db0a416fec5ee076ea2 Merge T-0208-b: loopback shared-screen handoff scenarios H1-H3 (drawn "
    "images, judge, real ingress)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0209-a1b: something else",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0209-a2-hf' into T-0209-a2-impl",
)


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0209 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0209_rtx_run_unit", HARNESS)
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
        f"T-0209 B: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api(name)
    return getattr(oracle, name)


def _tree() -> ast.Module:
    assert HARNESS.is_file(), f"T-0209 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    return ast.parse(HARNESS.read_text(encoding="utf-8"))


def _t0205_modules(h: ModuleType) -> list[ModuleType]:
    return [
        v
        for v in vars(h).values()
        if isinstance(v, ModuleType)
        and str(getattr(v, "__file__", "") or "").endswith("t0205_demo_run.py")
    ]


def _chain(h: ModuleType, name: str) -> ModuleType:
    """T-0205's ``F``/``G``/``T``/``P`` as the harness reaches them (its own name, or the T-0205 module's)."""
    found = getattr(h, name, None)
    if isinstance(found, ModuleType):
        return found
    for d in _t0205_modules(h):
        found = getattr(d, name, None)
        if isinstance(found, ModuleType):
            return found
    raise AssertionError(f"T-0209 B1: the harness reaches no {name!r} (T-0205's chain) by name")


def _patch_everywhere(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    mp.setattr(h, name, value, raising=False)
    targets = [*_t0205_modules(h)]
    with contextlib.suppress(AssertionError):
        targets.append(_chain(h, "G"))
    for t in targets:
        if hasattr(t, name):
            mp.setattr(t, name, value)


def _values_of(obj: Any, key: str) -> list[Any]:
    return t205._values_of(obj, key)


def _rtx_row(rid: str = "R1") -> Any:
    from local_astra.pages import rtx_tasks

    return next(r for r in rtx_tasks.RTX if r.rtx_id == rid)


# =====================================================================================================
# Premises (green at BASE): production constants, the oracle's literals, the eval's imports
# =====================================================================================================


def test_reasons_table_index_and_image_premise() -> None:
    """Premise: the label, reasons, table sha, index sha and tokens typed here and in the eval oracle are
    production's; the committed R1 image is its row's sha; A2 is absent (the table is R1 only)."""
    from local_astra.computer import executor
    from local_astra.conductor import session
    from local_astra.pages import rtx_search, rtx_tasks

    assert session.SHARED_LABEL_SCREEN == SHARED_LABEL_SCREEN == oracle.SHARED_LABEL_SCREEN
    assert session.SHARED_FRAME_REASON == SHARED_FRAME_REASON == oracle.SHARED_FRAME_REASON
    assert executor.CHECKOUT_REASON == CHECKOUT_REASON == oracle.CHECKOUT_REASON
    assert executor.MOVED_REASON == MOVED_REASON == oracle.MOVED_REASON
    assert oracle.CANNOT_SEE == CANNOT_SEE
    assert rtx_tasks.RTX_IDS == ("R1",), "premise: A2 is absent, B runs R1 only"
    assert rtx_tasks.rtx_table_sha() == TABLE_SHA256 == oracle.RTX_TABLE_SHA256
    assert rtx_search.search_index_sha() == INDEX_SHA256 == oracle.SEARCH_INDEX_SHA256
    assert tuple(sorted(rtx_tasks.R1_SOURCE_PAGES)) == tuple(sorted(oracle.R1_SOURCE_PAGES))
    for row in rtx_tasks.RTX:
        data = (IMAGE_DIR / row.image).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row.image_sha256, row.image
        assert oracle.IMAGES[row.rtx_id] == row.image
        assert oracle.IDENTITY_TOKENS[row.rtx_id] == tuple(row.identity_tokens)


def test_trap_kinds_partition_r1_wrong_fix_premise() -> None:
    """Premise (reading 9): the four trap kinds together match exactly where production's ``R1_WRONG_FIX`` matches,
    over a corpus naming each alternative and some near misses (so the breakdown never drops a trap)."""
    from local_astra.pages import rtx_tasks

    corpus = [
        "reinstall the driver",
        "a clean install of the driver",
        "use DDU",
        "Display Driver Uninstaller",
        "install the CUDA Toolkit 12.8",
        "roll back the driver",
        "rolling back to 566",
        "rollback",
        "downgrade the driver",
        "lower the tile size",
        "use tiling",
        "install Blender 4.4",
        "cuda kernels",
        "tiles",
        "the toolkit",
        "install it",
    ]
    for text in corpus:
        t = text.casefold()
        assert bool(oracle.trap_kinds(t)) == (rtx_tasks.R1_WRONG_FIX.search(t) is not None), text


def test_eval_imports_only_the_judge_and_never_the_harness() -> None:
    """B7 (C7-style): the eval module imports, from ``local_astra``, only ``rtx_tasks.judge``; nothing from
    ``scripts`` or ``tests``; loads nothing by path; names no harness file in code."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    doc = tree.body[0].value if isinstance(tree.body[0], ast.Expr) else None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith(("local_astra", "scripts", "tests", "importlib")), (
                    a.name
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("scripts", "tests", "importlib")), mod
            if mod.startswith("local_astra"):
                assert mod == "local_astra.pages.rtx_tasks", mod
                assert [a.name for a in node.names] == ["judge"], [a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not doc:
            assert "t0209_rtx_run.py" not in node.value and "scripts/gpu" not in node.value, (
                node.value[:80]
            )


#: content pins (sha256 at feab84c1, by the test-author with ``sha256sum``): B1 imports T-0205 C's helpers and never
#: edits them; B6 copies T-0205 C's boot (C7: every earlier harness byte-identical)
FROZEN_SHA256 = {
    "scripts/gpu/t0205_demo_run.py": "426d1c243eb35395d22470d69542474efc43f3122a5966b5178ec8e1d2fe29c8",
    "scripts/locked/t0205_demo_run.sh": "17a663ff87ce2ec2542ec155c13335cd32cfd4790e7f78771bcca4cfd16350c1",
}


@pytest.mark.parametrize("rel", list(FROZEN_SHA256))
def test_t0205_files_unchanged(rel: str) -> None:
    """B1/C7 negative (green at BASE and through group B)."""
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == FROZEN_SHA256[rel], f"T-0209 B1: {rel} changed ({got})"


# =====================================================================================================
# B1: import, constants, the search seam
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """B1: importing the harness opens no socket; it has ``main``."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"


def test_constants() -> None:
    """B1-B6: exits, arms, repeats, seed, the turn timeout, the run budget inside ``timeout 45m``, the pins, the
    reason, the regex, the labels, the image directory."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )
    assert tuple(api("ARMS")) == ARMS
    assert (api("REPEATS"), api("ORDER_SEED"), api("TURN_TIMEOUT_S")) == (3, 0, 300)
    assert api("RUN_BUDGET_S") == 40 * 60 < 45 * 60, "B6: the budget sits inside timeout 45m"
    assert api("RTX_TABLE_SHA256") == TABLE_SHA256
    assert api("SEARCH_INDEX_SHA256") == INDEX_SHA256
    assert api("SHARED_FRAME_REASON") == SHARED_FRAME_REASON
    assert api("CANNOT_SEE") == CANNOT_SEE
    assert (api("VOID"), api("INCOMPLETE"), api("MEASURED")) == ("void", "incomplete", "measured")
    assert Path(api("RTX_SHARED_DIR")).resolve() == IMAGE_DIR.resolve()


def test_no_real_search_backend_is_reachable() -> None:
    """B1/C7 (the scoped search exception untouched): no code names a network search backend or the cli's search
    starters; the code names ``LoopbackSearch``."""
    names, _ = t205._code(_tree())
    hits = {
        "_search_backend",
        "search_start",
        "DdgsSearch",
        "BraveApiSearch",
        "ddgs_backend",
        "brave_api",
    } & names
    assert not hits, f"B1: the harness reaches a network search backend: {sorted(hits)}"
    assert "LoopbackSearch" in HARNESS.read_text(encoding="utf-8"), (
        "B3: the harness never names LoopbackSearch"
    )


@dataclasses.dataclass(frozen=True)
class _StubKit:
    """A kit shaped as ``ToolKit`` for the fields the harness reads and replaces (``dataclasses.replace`` works)."""

    tools_version: str = ""
    repair_off: Any = None
    mode: str = "live"
    search: Any = None
    web_search: Any = None


class DdgsSearch:
    """A ``DdgsSearch``-shaped object (adversarial, B2): same method, a different backend; never a real one."""

    async def search(self, query: str) -> Any:
        raise AssertionError("T-0209 B2: a DdgsSearch-shaped backend was asked to search")


def _lookalike() -> Any:
    """A class *named* ``LoopbackSearch`` that is not production's (the check is on identity, not name)."""
    from local_astra.pages import rtx_search

    class LoopbackSearch:
        async def search(self, query: str) -> Any:
            return await rtx_search.LoopbackSearch().search(query)

    return LoopbackSearch()


def _kits() -> dict[str, tuple[Any, bool]]:
    from local_astra.pages.rtx_search import LoopbackSearch

    return {
        "loopback-ddgs": (_StubKit(search=LoopbackSearch(), web_search="ddgs"), True),
        "ddgs-shaped": (_StubKit(search=DdgsSearch(), web_search="ddgs"), False),
        "named-lookalike": (_StubKit(search=_lookalike(), web_search="ddgs"), False),
        "no-search": (_StubKit(search=None, web_search="ddgs"), False),
        "loopback-no-web-search": (_StubKit(search=LoopbackSearch(), web_search=None), False),
        "loopback-other-label": (_StubKit(search=LoopbackSearch(), web_search="brave"), False),
    }


@pytest.mark.parametrize("case", list(_kits()))
def test_kit_refusal_by_identity(case: str) -> None:
    """B2 (adversarial: a ``DdgsSearch``-shaped object, and a lookalike named ``LoopbackSearch``): ``kit_refusal``
    returns ``None`` only for production's ``LoopbackSearch`` with ``web_search="ddgs"``; else a non-empty reason."""
    kit, ok = _kits()[case]
    got = api("kit_refusal")(kit)
    if ok:
        assert got is None, (case, got)
    else:
        assert isinstance(got, str) and got.strip(), (case, got)


def test_rtx_kit_gives_a_fresh_loopback_and_ddgs_label() -> None:
    """B3 (H-T209-1): ``rtx_kit`` keeps every other field, sets production's ``LoopbackSearch`` (a fresh one per
    call) and ``web_search="ddgs"``; ``kit_refusal`` accepts the result."""
    from local_astra.pages.rtx_search import LoopbackSearch

    base = _StubKit(tools_version="abc", search=None, web_search=None)
    a = api("rtx_kit")(base)
    b = api("rtx_kit")(base)
    assert type(a.search) is LoopbackSearch and a.web_search == "ddgs", a
    assert a.tools_version == "abc" and a.repair_off is None and a.mode == "live", a
    assert a.search is not b.search, "B3: one LoopbackSearch shared across turns"
    assert api("kit_refusal")(a) is None


# =====================================================================================================
# B3: the order (seed 0; R1 only without A2)
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("r2", [False, True])
def test_order_seed0_over_both_arms(which: str, r2: bool) -> None:
    """B3 (reading 2): R1-only gives 6 turns, with R2 12; each (scenario, arm) three times, repeats 0-2; seed 0 is the
    hand construction; deterministic; another seed differs; arms interleave; no R2 turn without ``r2``."""
    order = impl(which, "order")
    ids = ("R1", "R2") if r2 else ("R1",)
    want = [(i, a, r) for i in ids for a in ARMS for r in range(3)]
    random.Random(0).shuffle(want)
    got = [tuple(t) for t in order(0, r2=r2)]
    assert got == want
    assert len(got) == (12 if r2 else 6)
    assert Counter((i, a) for i, a, _ in got) == {(i, a): 3 for i in ids for a in ARMS}
    assert got == [tuple(t) for t in order(0, r2=r2)] and got != [tuple(t) for t in order(1, r2=r2)]
    arms = [a for _, a, _ in got]
    assert arms != sorted(arms) and arms != sorted(arms, reverse=True), arms


def test_order_hand_literal_r1_only() -> None:
    """Premise for the oracle (reading 2): ``random.Random(0).shuffle`` of the six R1 turns, written out."""
    want = [("R1", a, r) for a in ARMS for r in range(3)]
    random.Random(0).shuffle(want)
    assert oracle.order(0) == want and oracle.order(0, r2=False) == want


# =====================================================================================================
# B4: P-R3 from hand-made session-log bodies (the switch rule)
# =====================================================================================================

IMG = b"\xff\xd8\xff\xe0" + b"T-0209 hand-made shared picture of a Blender error" + b"\xff\xd9"
B1 = b"\xff\xd8\xff\xe0" + b"T-0209 browser frame one" + b"\xff\xd9"
B2 = b"\xff\xd8\xff\xe0" + b"T-0209 browser frame two" + b"\xff\xd9"
Q = "why is Blender doing this?"
SEARCH_PAYLOAD = "google_search results: https://blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001"


def _img(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        "phash": "p" + hashlib.sha256(data).hexdigest()[:15],
    }


def _txt(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


_SEQ = iter(range(1, 1_000_000))


def _req(rid: str, phase: str, *parts: dict[str, Any], step: int | None = None) -> dict[str, Any]:
    """One ``llm.request`` record as the session log writes it (images redacted by production's ``redact``)."""
    seq = next(_SEQ)
    data: dict[str, Any] = {
        "phase": phase,
        "request_id": rid,
        "attempt_id": f"a-{seq}",
        "prompt": {
            "messages": [
                {"role": "system", "content": [_txt("You are Gearvis.")]},
                {"role": "user", "content": list(parts)},
            ]
        },
    }
    if step is not None:
        data["step"] = step
    return redact({"kind": "llm.request", "turn_id": "t1", "seq": seq, "data": data})


def _pr3_case(case: str) -> list[dict[str, Any]]:
    """The case's records in **body order**, ``seq`` 1..n in that order (review of 0ce94b69: the case bodies built
    after the shared ``s3`` took a later ``seq`` from the module counter, so ``pr3_ok``'s seq sort put them after the
    switch and clause (d) decided every case; (b) and (c) were never exercised alone)."""
    records = _pr3_bodies(case)
    for n, r in enumerate(records, start=1):
        r["seq"] = n
    return records


def _pr3_bodies(case: str) -> list[dict[str, Any]]:
    prefill = _req("t1-prefill", "prefill", _img(IMG), _txt(SHARED_LABEL_SCREEN))
    gen = _req("t1-gen", "generate", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q))
    # step 2 after a search: no browser action yet, so the shared image is still current (A1.11)
    s2 = _req(
        "t1-gen-s2",
        "tool_step",
        _img(IMG),
        _txt(SHARED_LABEL_SCREEN),
        _txt(Q),
        _txt(SEARCH_PAYLOAD),
        step=2,
    )
    s3 = _req("t1-gen-s3", "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=3)
    check = _req("t1-gen-s3", "completion_check", _img(B1), _img(B2), _txt(Q), step=3)
    if case == "search-then-browser":
        return [prefill, gen, s2, s3, check]
    if case == "browser-at-step-2":  # T-0208's shape: the first action at step 1
        return [
            prefill,
            gen,
            _req("t1-gen-s2", "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=2),
            check,
        ]
    if case == "step2-shared-without-label":  # the shared image presented as if it were the browser
        bad = _req("t1-gen-s2", "tool_step", _img(IMG), _txt(L_NOW), _txt(Q), step=2)
        return [prefill, gen, bad, s3]
    if case == "shared-after-switch":
        late = _req("t1-gen-s4", "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q), step=4)
        return [prefill, gen, s2, s3, late]
    if case == "base64-in-text-after-switch":
        hidden = _txt("earlier: " + base64.b64encode(IMG).decode("ascii"))
        return [prefill, gen, s2, s3, _req("t1-gen-s4", "tool_step", _img(B2), hidden, step=4)]
    if case == "check-carries-shared":
        # the first body with a browser frame also carries the shared image, with its label: (c) alone (no
        # browser-only body comes before it, so the rule has not switched and (d) cannot decide it; the label is
        # there, so (b) cannot either)
        both = _req(
            "t1-gen-s2",
            "completion_check",
            _img(B1),
            _img(IMG),
            _txt(SHARED_LABEL_SCREEN),
            _txt(Q),
            step=2,
        )
        return [prefill, gen, s2, both]
    if (
        case == "check-browser-only"
    ):  # check-carries-shared's twin: the same check without the shared image or its label
        return [
            prefill,
            gen,
            s2,
            _req("t1-gen-s2", "completion_check", _img(B1), _txt(Q), step=2),
        ]
    if (
        case == "step2-labelled"
    ):  # step2-shared-without-label's twin: the shared label, not the browser's
        ok = _req("t1-gen-s2", "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q), step=2)
        return [prefill, gen, ok, s3]
    if case == "step2-shared-only":  # step2-both's twin: the same body without the browser image
        alone = _req("t1-gen-s2", "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), step=2)
        return [prefill, gen, alone, s3]
    if case == "step2-both":
        both = _req(
            "t1-gen-s2", "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), _img(B1), step=2
        )
        return [prefill, gen, both, s3]
    if case == "step1-no-label":
        return [prefill, _req("t1-gen", "generate", _img(IMG), _txt(Q)), s2, s3]
    if case == "step1-browser-not-shared":
        return [
            _req("t1-prefill", "prefill", _img(B1)),
            _req("t1-gen", "generate", _img(B1), _txt(L_NOW), _txt(Q)),
            s3,
        ]
    if case == "no-step1-generate":
        return [s2, s3]
    raise KeyError(case)


#: case -> whether P-R3 holds (by hand, from reading 6's four clauses)
PR3_CASES = {
    "search-then-browser": True,
    "browser-at-step-2": True,
    "step2-labelled": True,
    "step2-shared-only": True,
    "check-browser-only": True,
    "step2-shared-without-label": False,  # (b)
    "shared-after-switch": False,  # (d)
    "base64-in-text-after-switch": False,  # (d)
    "check-carries-shared": False,  # (c)
    "step2-both": False,  # (c)
    "step1-no-label": False,  # (a), its label half: a non-prefill shared body without the label, as (b)
    "step1-browser-not-shared": False,  # (a)
    "no-step1-generate": False,  # (a)
}

#: a failing case -> (its clause, its passing twin). The twin differs from the case only in that clause's detail, so
#: the case fails for that clause (each pair by hand from ``_pr3_bodies``).
PR3_TWINS = {
    "step2-shared-without-label": ("b", "step2-labelled"),
    "step1-no-label": ("b", "search-then-browser"),
    "step2-both": ("c", "step2-shared-only"),
    "check-carries-shared": ("c", "check-browser-only"),
    "shared-after-switch": ("d", "search-then-browser"),
}


def _image_redactions(obj: Any) -> list[Mapping[str, Any]]:
    """Every ``{"redacted": "image", ...}`` in ``obj`` (production's ``redact`` shape)."""
    if isinstance(obj, Mapping):
        if obj.get("redacted") == "image":
            return [obj]
        return [x for v in obj.values() for x in _image_redactions(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _image_redactions(v)]
    return []


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(PR3_CASES))
def test_pr3_rule_from_hand_bodies(which: str, case: str) -> None:
    """B4 P-R3 (reading 6; reviewer mutation: T-0208's P-H3 copied unchanged voids every R1 golden turn, the
    ``search-then-browser`` case): the shared image with its label until the first browser frame, never after it,
    never beside another image."""
    got = impl(which, "pr3_ok")(_pr3_case(case), IMG)
    assert got is PR3_CASES[case], (case, got)


@pytest.mark.parametrize("case", list(PR3_CASES))
def test_pr3_case_bodies_are_in_seq_order(case: str) -> None:
    """Premise for the P-R3 rows: each case's records carry ``seq`` 1..n in the order the bodies were sent, so the
    rule's seq sort reads them as written (at 0ce94b69 the (b)/(c) bodies were read after the switch)."""
    records = _pr3_case(case)
    assert [r["seq"] for r in records] == list(range(1, len(records) + 1)), case


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(PR3_TWINS))
def test_pr3_case_fails_only_its_own_clause(which: str, case: str) -> None:
    """B4 P-R3 (review of 0ce94b69; adversarial): each failing case has a passing twin differing only in its clause's
    detail; for a (b) or (c) case no browser-only body precedes the offending body, so (d) cannot decide it. Hence a
    rule without (b) passes ``step2-shared-without-label`` and ``step1-no-label``, and one without (c) passes
    ``step2-both`` and ``check-carries-shared``."""
    clause, twin = PR3_TWINS[case]
    pr3 = impl(which, "pr3_ok")
    assert pr3(_pr3_case(twin), IMG) is True, (twin, "the twin must hold")
    assert pr3(_pr3_case(case), IMG) is False, case
    if clause == "d":
        return
    # the shared image's redaction digest: sha256 of its base64 text, first 8 hex (production's ``redact``)
    b64 = base64.b64encode(IMG).decode("ascii")
    digest = hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8]
    records = _pr3_case(case)
    reds = [_image_redactions(r["data"]["prompt"]) for r in records]
    assert all(x.get("bytes") is not None and x.get("digest") for rs in reds for x in rs), reds
    carrying = [i for i, rs in enumerate(reds) if any(x["digest"] == digest for x in rs)]
    assert carrying, case
    last = carrying[-1]
    browser_only = [
        records[i]["seq"]
        for i in range(last)
        if reds[i] and all(x["digest"] != digest for x in reds[i])
    ]
    assert not browser_only, (case, "a browser-only body precedes the offending body", browser_only)


def test_pr3_out_of_seq_order_is_read_in_seq_order() -> None:
    """Oracle premise: the rule reads records by ``seq``; the clean turn shuffled is still clean, and a shared body
    whose ``seq`` is after the switch is still void."""
    clean = _pr3_case("search-then-browser")
    assert oracle.pr3_ok(list(reversed(clean)), IMG) is True
    late = _pr3_case("shared-after-switch")
    assert oracle.pr3_ok(list(reversed(late)), IMG) is False


RAISED = "RuntimeError('the browser died')"
INGRESS = "IngressError(\"the session looks at 'browser', not 'shared'\")"
#: case -> (arm, records case or None for no records, exception, ingress_failed, want), each by hand from T-0208
#: ruling C-R2 (reading 6's per-turn paragraph)
TURN_PR3_CASES: dict[str, tuple[str, str | None, str | None, bool, bool | None]] = {
    "typed-is-none": ("typed", "search-then-browser", None, False, None),
    "typed-raised-is-none": ("typed", None, RAISED, False, None),
    "shared-clean": ("shared", "search-then-browser", None, False, True),
    "shared-bad-bodies": ("shared", "shared-after-switch", None, False, False),
    # no exception and no request: the rule itself fails (a), not None
    "shared-no-records-no-exception": ("shared", None, None, False, False),
    # C-R2: raised, no step-1 request -> None (not void)
    "shared-raised-no-records": ("shared", None, RAISED, False, None),
    "shared-raised-later-steps-only": ("shared", "no-step1-generate", RAISED, False, None),
    # C-R2, the guard's other side: raised after step 1 was sent -> the rule still reads the bodies
    "shared-raised-after-step1-clean": ("shared", "search-then-browser", RAISED, False, True),
    "shared-raised-after-step1-bad": ("shared", "shared-after-switch", RAISED, False, False),
    # C-R2: the ingress self-check failure is False whatever the records say
    "shared-ingress-no-records": ("shared", None, INGRESS, True, False),
    "shared-ingress-clean-bodies": ("shared", "search-then-browser", INGRESS, True, False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(TURN_PR3_CASES))
def test_turn_pr3_per_turn(which: str, case: str) -> None:
    """B4 P-R3 per turn (T-0208 C-R2): typed ``None``; ingress self-check failure ``False``; raised with no step-1
    ``generate`` ``None``; else ``pr3_ok``."""
    arm, records_case, exception, ingress, want = TURN_PR3_CASES[case]
    records = _pr3_case(records_case) if records_case is not None else []
    got = impl(which, "turn_pr3")(arm, records, IMG, exception=exception, ingress_failed=ingress)
    assert got is want, (case, got)


def test_oracle_reads_the_ingress_failure_off_the_recorded_exception() -> None:
    """Oracle premise: ``repr`` of the harness's ``IngressError`` is what the oracle recognises; no other class is."""
    h_exc = api("IngressError")("the HUD ingress rejected the shared frame ({})")
    assert oracle.ingress_failed({"exception": repr(h_exc)}) is True
    assert oracle.ingress_failed({"exception": RAISED}) is False
    assert oracle.ingress_failed({"exception": "NotIngressError('x')"}) is False
    assert oracle.ingress_failed({"exception": None}) is False


# =====================================================================================================
# B5: identity, step 1, refusals, "cannot see", trap kinds
# =====================================================================================================

SAFE = J.SAFE
GOLD = J.GOLD
GOLDEN = J.R1_GOLDEN_ANSWER
FORUM_WRONG = J.FORUM_WRONG
OOM_WRONG = J.OOM_WRONG
ERROR = "CUDA binary kernel for this graphics card compute capability (12.0) not found"
SAY = "That error means this Blender build has no kernel for your card; I will look it up."
ISSUE_ADDR = "projects.blender.org/blender/blender/issues/134001"
NOTES_ADDR = "developer.blender.org/docs/release_notes/4.4/cycles"
THREAD_ADDR = "blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001"
OOM_ADDR = "blenderartists.org/t/cycles-cuda-error-out-of-memory/1570002"
SNIPPET_ONLY = {"page": "web", "visited": ["web"], **SAFE}
FORUM_STATE = {"page": "forum-thread", "visited": ["web", "web", "forum-thread"], **SAFE}
OOM_STATE = {"page": "forum-oom", "visited": ["web", "web", "forum-oom"], **SAFE}
DOWNLOAD_STATE = {**GOLD, "download": [{"page": "download", "item": "linux"}]}
LOGIN_STATE = {
    "page": "forum-login",
    "visited": ["web", "forum-login"],
    **SAFE,
    "signin_typed": ["password"],
}
FORUM_ROLLBACK = FORUM_WRONG + " Or roll back the driver."


def _trial(
    rid: str, arm: str, state: Any, voiced: list[str], repeat: int = 0, **over: Any
) -> dict[str, Any]:
    return {
        "task": rid,
        "arm": arm,
        "repeat": repeat,
        "state": copy.deepcopy(state),
        "voiced": list(voiced),
        "shown": "",
        "clips": [],
        "tool_loop_end": None,
        "timed_out": False,
        "engine_up_after": True,
        "wall_s": 10.0,
        "completion_checks": 0,
        "pr3": True if arm == "shared" else None,
        **over,
    }


def _st(
    k: int,
    name: str | None,
    status: str | None,
    reason: str | None = None,
    rid: str = "R1",
    arm: str = "shared",
    repeat: int = 0,
    **arguments: Any,
) -> dict[str, Any]:
    return {
        "task": rid,
        "arm": arm,
        "repeat": repeat,
        "step": k,
        "http_status": 200,
        "retries": [],
        "status": status,
        "reason": reason,
        "call_name": name,
        "arguments": dict(arguments),
    }


def test_hand_states_premise() -> None:
    """Premise: the hand states under the real judge, as the hand counts below assume (each by A1.9)."""
    from local_astra.pages.rtx_tasks import judge

    g = judge("R1", GOLD, SAY + " " + GOLDEN)
    assert (
        g.success
        and not g.trap
        and {"cross_checked", "said_link", "mentions_wrong_fix"} <= set(g.notes)
    )
    s = judge("R1", SNIPPET_ONLY, GOLDEN)
    assert not s.success and not s.trap and "no_page_opened" in s.notes and "said_link" in s.notes
    f = judge("R1", FORUM_STATE, FORUM_ROLLBACK)
    assert (
        not f.success and f.trap and "said_link" not in f.notes and "cross_checked" not in f.notes
    )
    o = judge("R1", OOM_STATE, OOM_WRONG)
    assert not o.success and o.trap and "no_page_opened" not in o.notes
    d = judge("R1", DOWNLOAD_STATE, GOLDEN)
    assert not d.success and d.over_action and not d.trap and "cross_checked" in d.notes
    li = judge("R1", LOGIN_STATE, "I can't see the page.")
    assert not li.success and li.secret_typed and "mentions_wrong_fix" not in li.notes


SEARCH_ERR = ("google_search", "executed")
#: case -> (trial, steps, (spoken, typed, lost)), each by hand from reading 4
IDENTITY_CASES: dict[str, tuple[dict[str, Any], list[dict[str, Any]], tuple[bool, bool, bool]]] = {
    "spoken": (
        _trial("R1", "shared", GOLD, [SAY, GOLDEN]),
        [_st(1, *SEARCH_ERR, query="rtx")],
        (True, False, False),
    ),
    # the error string typed into the search carries the identity (reading 4's widening)
    "searched": (
        _trial("R1", "shared", FORUM_STATE, ["Done."]),
        [_st(1, *SEARCH_ERR, query=ERROR), _st(2, "press_key", "executed", combo="Ctrl+L")],
        (False, True, False),
    ),
    "typed-at": (
        _trial("R1", "shared", FORUM_STATE, ["Done."]),
        [_st(1, "type_at", "executed", text="blender 4.2.3 rtx 5090", x=1, y=1)],
        (False, True, False),
    ),
    "lost": (
        _trial("R1", "shared", FORUM_STATE, [FORUM_WRONG]),
        [
            _st(1, *SEARCH_ERR, query="rtx 5090 blender"),
            _st(2, "press_key", "executed", combo="Ctrl+L"),
        ],
        (False, False, True),
    ),
    # T-0208 ruling C-R1: an executed wait swaps the shared image out, so a wait-only failed turn naming nothing is lost
    "a-wait-alone-is-acting": (
        _trial("R1", "shared", SNIPPET_ONLY, ["It is a driver thing."]),
        [_st(1, "wait", "executed")],
        (False, False, True),
    ),
    # the guard's other side: a refused wait ran nothing, so it is not acting
    "a-refused-wait-is-not-acting": (
        _trial("R1", "shared", SNIPPET_ONLY, ["It is a driver thing."]),
        [_st(1, "wait", "refused", "at most 3 waits in a row")],
        (False, False, False),
    ),
    "a-search-alone-is-not-acting": (
        _trial("R1", "shared", SNIPPET_ONLY, ["It is a driver thing."]),
        [_st(1, *SEARCH_ERR, query="rtx 5090 blender")],
        (False, False, False),
    ),
    "typed-arm-is-never-lost": (
        _trial("R1", "typed", FORUM_STATE, [FORUM_WRONG]),
        [_st(1, "press_key", "executed", arm="typed", combo="Ctrl+L")],
        (False, False, False),
    ),
    # "12.8" is not "12.0"; "4.2" is not "4.2.3"
    "near-tokens-are-not-tokens": (
        _trial("R1", "shared", FORUM_STATE, ["Install CUDA Toolkit 12.8 for Blender 4.2."]),
        [_st(1, "press_key", "executed", combo="Ctrl+L")],
        (False, False, True),
    ),
    "casefolded": (
        _trial("R1", "shared", FORUM_STATE, ["Your COMPUTE CAPABILITY is the issue."]),
        [_st(1, "press_key", "executed", combo="Ctrl+L")],
        (True, False, False),
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(IDENTITY_CASES))
def test_identity_columns_from_hand_steps(which: str, case: str) -> None:
    """B5 identity columns (T-0208 C5 over R1's tokens; reading 4)."""
    trial, steps, want = IDENTITY_CASES[case]
    got = dict(impl(which, "identity")(trial, steps))
    assert (
        bool(got.get("identity_spoken")),
        bool(got.get("identity_typed")),
        bool(got.get("identity_lost")),
    ) == want, (case, got)


def _action(x: Any) -> tuple[Any, Any]:
    if isinstance(x, dict):
        return (x.get("tool"), x.get("outcome"))
    return tuple(x)  # type: ignore[return-value]


STEP1_CASES: dict[str, tuple[list[dict[str, Any]], tuple[str | None, str]]] = {
    "search": (
        [_st(1, *SEARCH_ERR, query=ERROR), _st(2, "press_key", "executed")],
        ("google_search", "executed"),
    ),
    "refused-shared": (
        [_st(1, "click_at", "refused", SHARED_FRAME_REASON)],
        ("click_at", "refused_shared"),
    ),
    "refused-moved": ([_st(1, "type_at", "refused", MOVED_REASON)], ("type_at", "refused_other")),
    "no-call": ([_st(1, None, None)], (None, "no_call")),
    "no-step-line": ([], (None, "no_call")),
    "by-step-number": (
        [_st(2, "press_key", "executed"), _st(1, "click_at", "refused", SHARED_FRAME_REASON)],
        ("click_at", "refused_shared"),
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(STEP1_CASES))
def test_step1_action_classifier(which: str, case: str) -> None:
    """B5: the step-1 action is its tool and outcome."""
    steps, want = STEP1_CASES[case]
    assert _action(impl(which, "step1_action")(steps)) == want, case


@pytest.mark.parametrize("which", IMPLS)
def test_shared_refusals_exact_reason(which: str) -> None:
    """B5 (adversarial, exact reason equality): of five refusal-like steps exactly one is a shared-image refusal."""
    steps = [
        _st(1, "click_at", "refused", SHARED_FRAME_REASON),
        _st(2, "click_at", "refused", CHECKOUT_REASON),
        _st(3, "click_at", "refused", MOVED_REASON),
        _st(4, "click_at", "refused", SHARED_FRAME_REASON + "; and more"),
        _st(5, "click_at", "executed", SHARED_FRAME_REASON),
    ]
    assert impl(which, "shared_refusals")(steps) == 1


CANNOT_SEE_CASES = {
    "I can't see the error.": True,
    "I cannot see the picture.": True,
    "I'm unable to see your screen.": True,
    "I can see the Blender error.": False,
    "": False,
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("text", list(CANNOT_SEE_CASES))
def test_cannot_see_ending(which: str, text: str) -> None:
    """B5: T-0208's pinned regex over the voiced text."""
    trial = _trial("R1", "shared", SNIPPET_ONLY, [text] if text else [])
    assert bool(impl(which, "cannot_see")(trial)) is CANNOT_SEE_CASES[text], text


TRAP_KIND_CASES = {
    FORUM_WRONG: ["ddu", "cuda_toolkit"],
    FORUM_ROLLBACK: ["ddu", "rollback", "cuda_toolkit"],
    OOM_WRONG: ["tile_size"],
    "Downgrade to the 566 driver.": ["rollback"],
    "Install Blender 4.4 and pick OptiX.": [],
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("text", list(TRAP_KIND_CASES))
def test_trap_kinds(which: str, text: str) -> None:
    """B5's trap breakdown (reading 9): DDU or a driver reinstall, tile size, rollback, CUDA Toolkit; each named
    kind once, in that order; one kind per answer however often named (OOM_WRONG names tiling and tile size)."""
    got = list(impl(which, "trap_kinds")(text))
    assert got == TRAP_KIND_CASES[text], (text, got)


# =====================================================================================================
# B4/B5: P-R1, P-R2, the label, and the per-arm counts by hand
# =====================================================================================================


def _path(state: Any, voiced: str, actions: int = 4, **over: Any) -> dict[str, Any]:
    return {
        "actions": actions,
        "outcomes": ["executed"] * actions,
        "state": copy.deepcopy(state),
        "voiced": voiced,
        "shown": "",
        "error": None,
        **over,
    }


def _pr1(**over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "viewport": [1280, 800],
        "paths": {
            "R1": {
                "golden": _path(GOLD, GOLDEN),
                "traps": {
                    "forum": _path(FORUM_STATE, FORUM_WRONG, actions=2),
                    "oom": _path(OOM_STATE, OOM_WRONG, actions=2),
                    "download": _path(DOWNLOAD_STATE, GOLDEN, actions=5),
                    "signin": _path(LOGIN_STATE, GOLDEN, actions=3),
                },
            }
        },
    }
    rec.update(over)
    return rec


PR2_OK = {
    "resolver_calls": 0,
    "connector_calls": 0,
    "non_loopback": [],
    "egress_log": [],
    "requests_off_origin": [],
}


def _probes(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "p1": {"ok": True, "passed": True},
        "p2": {"ok": True, "passed": True},
        "p3": {"ok": True, "passed": True},
        "pr1": _pr1(),
        "pr2": dict(PR2_OK),
    }
    base.update(over)
    return base


def _mut_pr1(case: str) -> dict[str, Any]:
    rec = _pr1()
    if case == "golden-fails":
        rec["paths"]["R1"]["golden"] = _path(SNIPPET_ONLY, GOLDEN, actions=0)
    elif case == "trap-passes":
        rec["paths"]["R1"]["traps"]["forum"] = _path(GOLD, GOLDEN)
    elif case == "golden-not-executed":
        rec["paths"]["R1"]["golden"]["outcomes"][1] = "refused"
    elif case == "r1-missing":
        del rec["paths"]["R1"]
    elif case == "r2-without-a2":
        rec["paths"]["R2"] = copy.deepcopy(rec["paths"]["R1"])
    elif case == "no-traps":
        rec["paths"]["R1"]["traps"] = {}
    elif case == "viewport":
        rec["viewport"] = [1440, 900]
    else:
        raise KeyError(case)
    return rec


LABEL_CASES: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str | None, str]] = {
    "measured": (_probes(), [], None, "measured"),
    "typed-pr3-none-is-not-void": (
        _probes(),
        [_trial("R1", "typed", GOLD, [GOLDEN])],
        None,
        "measured",
    ),
    "a-shared-turn-fails-pr3": (
        _probes(),
        [
            _trial("R1", "shared", GOLD, [GOLDEN]),
            _trial("R1", "shared", GOLD, [GOLDEN], 1, pr3=False),
        ],
        None,
        "void",
    ),
    "a-shared-turn-pr3-missing": (
        _probes(),
        [_trial("R1", "shared", GOLD, [GOLDEN], pr3=None)],
        None,
        "void",
    ),
    # C-R2, the other side: None beside an exception (no step-1 request) is left to ERRORED_STOP, not void
    "a-shared-turn-pr3-none-after-an-exception-is-not-void": (
        _probes(),
        [
            _trial("R1", "shared", GOLD, [GOLDEN]),
            _trial(
                "R1",
                "shared",
                SNIPPET_ONLY,
                [],
                1,
                pr3=None,
                exception="RuntimeError('the browser died')",
            ),
        ],
        None,
        "measured",
    ),
    # an exception does not excuse a False (only a None)
    "a-shared-turn-pr3-false-after-an-exception": (
        _probes(),
        [
            _trial(
                "R1",
                "shared",
                SNIPPET_ONLY,
                [],
                pr3=False,
                exception="RuntimeError('the browser died')",
            )
        ],
        None,
        "void",
    ),
    # C-R2: the ingress self-check failure is pr3 False, so void
    "ingress-failure": (
        _probes(),
        [
            _trial(
                "R1",
                "shared",
                SNIPPET_ONLY,
                [],
                pr3=False,
                exception="IngressError(\"the session looks at 'browser', not 'shared'\")",
            )
        ],
        None,
        "void",
    ),
    "pr1-golden-fails": (_probes(pr1=_mut_pr1("golden-fails")), [], None, "void"),
    "pr1-trap-passes": (_probes(pr1={**_mut_pr1("trap-passes"), "passed": True}), [], None, "void"),
    "pr1-golden-not-executed": (_probes(pr1=_mut_pr1("golden-not-executed")), [], None, "void"),
    "pr1-r1-missing": (_probes(pr1=_mut_pr1("r1-missing")), [], None, "void"),
    "pr1-r2-without-a2": (_probes(pr1=_mut_pr1("r2-without-a2")), [], None, "void"),
    "pr1-no-traps": (_probes(pr1=_mut_pr1("no-traps")), [], None, "void"),
    "pr1-viewport": (_probes(pr1=_mut_pr1("viewport")), [], None, "void"),
    "pr2-resolver": (_probes(pr2={**PR2_OK, "resolver_calls": 1}), [], None, "void"),
    "pr2-connector": (
        _probes(pr2={**PR2_OK, "connector_calls": 2, "passed": True}),
        [],
        None,
        "void",
    ),
    "pr2-non-loopback": (
        _probes(pr2={**PR2_OK, "non_loopback": ["93.184.216.34:443"]}),
        [],
        None,
        "void",
    ),
    "pr2-egress": (_probes(pr2={**PR2_OK, "egress_log": [{"host": "x"}]}), [], None, "void"),
    "p1-failed": (_probes(p1={"ok": False, "passed": False}), [], None, "void"),
    "stopped": (_probes(), [], "the run budget is spent", "incomplete"),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(LABEL_CASES))
def test_labels_and_probe_rules(which: str, case: str) -> None:
    """B4/B5 (reading 7): any probe failing (P1-P3, P-R1 and P-R2 re-derived, a recorded ``passed`` never trusted) or
    any shared turn whose ``pr3`` is not ``True`` is ``void``; else a stopped run is ``incomplete``; else
    ``measured``. P-R1 with an R2 path but ``r2`` false is void (the run's ids exactly)."""
    probes, trials, stopped, want = LABEL_CASES[case]
    report = impl(which, "build_report")(trials, [], probes, stopped, r2=False)
    assert report["label"] == want, (case, report.get("label"), report.get("probes"))


def _hand_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Six R1 turns; every count in ``HAND_COUNTS`` is by hand from these lines."""
    trials = [
        # shared 0: the golden path; success; 6 step lines; 60 s; one completion check
        _trial("R1", "shared", GOLD, [SAY, GOLDEN], 0, wall_s=60.0, completion_checks=1),
        # shared 1: snippet only (no page opened); 2 step lines; 20 s
        _trial("R1", "shared", SNIPPET_ONLY, [GOLDEN], 1, wall_s=20.0),
        # shared 2: step 1 refused on the picture, then the forum's trap plus a rollback; 5 step lines; 40 s
        _trial("R1", "shared", FORUM_STATE, [FORUM_ROLLBACK], 2, wall_s=40.0),
        # typed 0: the OOM thread's trap; a wait; 5 step lines; 50 s
        _trial("R1", "typed", OOM_STATE, [OOM_WRONG], 0, wall_s=50.0),
        # typed 1: the golden answer but a download pressed; step cap; 2 step lines; 30 s
        _trial("R1", "typed", DOWNLOAD_STATE, [GOLDEN], 1, wall_s=30.0, tool_loop_end="budget"),
        # typed 2: sign-in typed, "can't see", timed out (an engine error); 1 step line; 300 s
        _trial(
            "R1", "typed", LOGIN_STATE, ["I can't see the page."], 2, wall_s=300.0, timed_out=True
        ),
    ]
    steps = [
        _st(1, *SEARCH_ERR, repeat=0, query=ERROR),
        _st(2, "press_key", "executed", repeat=0, combo="Ctrl+L"),
        _st(3, "type_at", "executed", repeat=0, text=ISSUE_ADDR, x=1, y=1, press_enter=True),
        _st(4, "press_key", "executed", repeat=0, combo="Ctrl+L"),
        _st(5, "type_at", "executed", repeat=0, text=NOTES_ADDR, x=1, y=1, press_enter=True),
        _st(6, None, None, repeat=0),
        _st(1, *SEARCH_ERR, repeat=1, query="blender rtx 5090 no gpu"),
        _st(2, None, None, repeat=1),
        _st(1, "click_at", "refused", SHARED_FRAME_REASON, repeat=2, x=5, y=5),
        _st(2, *SEARCH_ERR, repeat=2, query="rtx 5090 blender"),
        _st(3, "press_key", "executed", repeat=2, combo="Ctrl+L"),
        _st(4, "type_at", "executed", repeat=2, text=THREAD_ADDR, x=1, y=1, press_enter=True),
        _st(5, None, None, repeat=2),
        _st(1, *SEARCH_ERR, arm="typed", repeat=0, query="cycles out of memory"),
        _st(2, "press_key", "executed", arm="typed", repeat=0, combo="Ctrl+L"),
        _st(
            3,
            "type_at",
            "executed",
            arm="typed",
            repeat=0,
            text=OOM_ADDR,
            x=1,
            y=1,
            press_enter=True,
        ),
        _st(4, "wait", "executed", arm="typed", repeat=0),
        _st(5, None, None, arm="typed", repeat=0),
        _st(1, *SEARCH_ERR, arm="typed", repeat=1, query="compute capability 12.0 blender 4.2.3"),
        _st(2, None, None, arm="typed", repeat=1),
        _st(1, None, None, arm="typed", repeat=2),
    ]
    return trials, steps


HAND_COUNTS: dict[str, dict[str, Any]] = {
    "shared": {
        "n": 3,
        "successes": 1,
        "trap": 1,
        "trap_kinds": {"ddu": 1, "rollback": 1, "cuda_toolkit": 1},  # only the trap turn, shared 2
        "no_page_opened": 1,  # shared 1
        "mentions_wrong_fix": 3,  # the golden answer names "reinstalling" (declined), so 0, 1 and 2
        "said_link": 2,  # shared 0 and 1 (projects.blender.org)
        "cross_checked": 1,  # shared 0
        "forum_opened": 1,
        "source_pages": {"issue-blackwell": 1, "notes-4-4": 1, "forum-thread": 1},
        "pages_visited": [
            ["web", "issue-blackwell", "notes-4-4"],
            ["web"],
            ["web", "web", "forum-thread"],
        ],
        "searches": [1, 1, 1],
        "search_queries": [[ERROR], ["blender rtx 5090 no gpu"], ["rtx 5090 blender"]],
        "steps_p50": 5,  # 6, 2, 5: the second smallest
        "steps_max": 6,
        "wall_p50": 40.0,
        "wall_max": 60.0,
        "step1_actions": {"google_search:executed": 2, "click_at:refused_shared": 1},
        "shared_refusals": 1,
        "cannot_see": 0,
        "identity_spoken": 2,  # 0 and 1 say "compute capability", "4.2.3"
        "identity_typed": 1,  # 0 searched the error string
        "identity_lost": 1,  # 2: acted (Ctrl+L), failed, named nothing
        "completion_checks": 1,
        "downloads": 0,
        "signin_typed": 0,
        "waits": 0,
        "engine_errors": 0,
        "step_cap": 0,
    },
    "typed": {
        "n": 3,
        "successes": 0,
        "trap": 1,
        "trap_kinds": {"tile_size": 1},
        "no_page_opened": 0,
        "mentions_wrong_fix": 2,  # typed 0 and 1
        "said_link": 1,
        "cross_checked": 1,  # typed 1 visited the issue and the notes (the download does not undo the visit)
        "forum_opened": 0,
        "source_pages": {"issue-blackwell": 1, "notes-4-4": 1},
        "pages_visited": [
            ["web", "web", "forum-oom"],
            ["web", "issue-blackwell", "notes-4-4"],
            ["web", "forum-login"],
        ],
        "searches": [1, 1, 0],
        "search_queries": [["cycles out of memory"], ["compute capability 12.0 blender 4.2.3"], []],
        "steps_p50": 2,  # 5, 2, 1
        "steps_max": 5,
        "wall_p50": 50.0,  # 30, 50, 300
        "wall_max": 300.0,
        "step1_actions": {"google_search:executed": 2, "none:no_call": 1},
        "shared_refusals": 0,
        "cannot_see": 1,
        "identity_spoken": 1,
        "identity_typed": 1,
        "identity_lost": 0,
        "downloads": 1,
        "signin_typed": 1,
        "waits": 1,
        "engine_errors": 1,
        "step_cap": 1,
    },
}
#: Wilson 95 % of 1/3 and 0/3 (z = 1.959964), by hand
WILSON_1_OF_3 = (0.0615, 0.7923)
WILSON_0_OF_3 = (0.0, 0.5615)


@pytest.mark.parametrize("which", IMPLS)
def test_per_arm_counts_by_hand(which: str) -> None:
    """B5: per arm, R1's columns of a hand-made run (``HAND_COUNTS``): success, Wilson, ``no_page_opened``,
    ``mentions_wrong_fix`` (the hand-audit count), the trap breakdown, ``said_link``, ``cross_checked``, the forum and
    the source pages opened, pages visited verbatim, searches and queries per turn, step 1, refusals, "cannot see",
    identity, checks, downloads, sign-in typing, waits, steps and wall, engine errors and step-cap ends."""
    trials, steps = _hand_run()
    report = impl(which, "build_report")(trials, steps, _probes(), None, r2=False)
    per = report["per_arm"]
    for arm, want in HAND_COUNTS.items():
        got = per[arm]["R1"]
        for k, v in want.items():
            g = got.get(k, "<absent>")
            if isinstance(v, dict):
                g = {kk: vv for kk, vv in dict(g or {}).items() if vv}
            assert g == v, (arm, k, g, v)
    assert per["shared"]["R1"]["wilson"] == pytest.approx(list(WILSON_1_OF_3), abs=1e-3)
    assert per["typed"]["R1"]["wilson"] == pytest.approx(list(WILSON_0_OF_3), abs=1e-3)
    assert report["label"] == "measured", report["label"]


# =====================================================================================================
# B2: refusals before any socket; the rig at HEAD goes to the engine
# =====================================================================================================


class Rig:
    """Every real socket refused and recorded; a clean tree and ``merges`` through the harness's ``_git`` (and
    T-0205's chain's); the kit ``cli`` resolves (repair on); Chromium reported present."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli

        self.h = h = harness()
        self.cli = cli
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*OTHERS, MERGE_0208_A2, MERGE_A1)
        self.out = tmp_path / "out"
        self.kits: list[dict[str, Any]] = []

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0209 refusal rig: no {what}")

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

        _patch_everywhere(mp, h, "_git", git)

        def load(
            profile: Any,
            *,
            live_browser: Any = None,
            page_overview: bool = False,
            wait_tool: bool = False,
            **k: Any,
        ) -> Any:
            self.kits.append({"page_overview": page_overview, "wait_tool": wait_tool, **k})
            return (_StubKit(tools_version=t205._set_sha(page_overview, wait_tool)), object())

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)

    def argv(self, *extra: str) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out), *extra]


def _image_copy(tmp_path: Path, alter: bool = False) -> Path:
    """The committed images copied under ``tmp_path``; ``alter`` appends one byte to R1's."""
    from local_astra.pages import rtx_tasks

    d = tmp_path / "rtx-images"
    d.mkdir()
    for row in rtx_tasks.RTX:
        shutil.copy(IMAGE_DIR / row.image, d / row.image)
    if alter:
        with (d / _rtx_row("R1").image).open("ab") as fh:
            fh.write(b"\x00")
    return d


REFUSALS = (
    "url-not-loopback",
    "dirty-tree",
    "out-dir-exists",
    "no-merge-a1",
    "table-off-its-sha",
    "index-off-its-sha",
    "image-off-its-sha",
    "web-off",
    "guidance-v2",
    "loop-19",
    "kit-search-ddgs-shaped",
    "kit-search-lookalike",
    "only-r2-without-a2",
)
REFUSAL_NAMES = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"session\.py",
    "no-merge-a1": r"T-0209-a1",
    "table-off-its-sha": r"table|sha",
    "index-off-its-sha": r"index|search|sha",
    "image-off-its-sha": r"r1_blender_error",
    "web-off": r"web",
    "guidance-v2": r"v3|guidance",
    "loop-19": r"20|loop|max_tool_steps",
    "kit-search-ddgs-shaped": r"LoopbackSearch|search",
    "kit-search-lookalike": r"LoopbackSearch|search",
    "only-r2-without-a2": r"R2|T-0209-a2",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """B2 (reading 1): each refusal exits 2 with a stderr line naming it, before any socket, leaving the out-dir as it
    was. "No Merge T-0209-a1": the log holds T-0207-c's, T-0208-b's and -a2's merges, the branch merges into the
    A1/A2 branches and a ``Merge T-0209-a1b:``, but not A1's. The kit rows move ``rtx_kit`` (looked up at call time)
    so the replaced kit carries a ``DdgsSearch``-shaped backend or a lookalike named ``LoopbackSearch``."""
    from local_astra.pages import rtx_search, rtx_tasks

    rig = Rig(tmp_path, monkeypatch)
    cli = rig.cli
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-a1":
        rig.merges = (*OTHERS, MERGE_0208_A2)
    elif case == "table-off-its-sha":
        moved = (
            dataclasses.replace(rtx_tasks.RTX[0], request="why is Blender slow?"),
            *rtx_tasks.RTX[1:],
        )
        monkeypatch.setattr(rtx_tasks, "RTX", moved)
        if hasattr(rig.h, "RTX"):
            monkeypatch.setattr(rig.h, "RTX", moved)
        assert rtx_tasks.rtx_table_sha() != TABLE_SHA256, "premise: the table moved"
    elif case == "index-off-its-sha":
        t, u, s, k = rtx_search.SEARCH_ROWS[0]
        moved_rows = (
            (t, u, s + " Edit Preferences System Cycles Render Devices.", k),
            *rtx_search.SEARCH_ROWS[1:],
        )
        monkeypatch.setattr(rtx_search, "SEARCH_ROWS", moved_rows)
        if hasattr(rig.h, "SEARCH_ROWS"):
            monkeypatch.setattr(rig.h, "SEARCH_ROWS", moved_rows)
        assert rtx_search.search_index_sha() != INDEX_SHA256, "premise: the index moved"
    elif case == "image-off-its-sha":
        assert hasattr(rig.h, "RTX_SHARED_DIR"), (
            "T-0209 B1: the harness has no module-level RTX_SHARED_DIR"
        )
        monkeypatch.setattr(rig.h, "RTX_SHARED_DIR", _image_copy(tmp_path, alter=True))
    elif case == "web-off":
        monkeypatch.setattr(cli, "web_of", lambda *a, **k: False)
    elif case == "guidance-v2":
        monkeypatch.setattr(
            cli, "live_guidance_of", lambda p, a, m: cli.LiveGuidance("v2", "default")
        )
    elif case == "loop-19":
        real = cli.tool_loop_of
        monkeypatch.setattr(
            cli, "tool_loop_of", lambda p, a: dataclasses.replace(real(p, a), max_tool_steps=19)
        )
    elif case in ("kit-search-ddgs-shaped", "kit-search-lookalike"):
        backend = DdgsSearch() if case == "kit-search-ddgs-shaped" else _lookalike()
        assert hasattr(rig.h, "rtx_kit"), "T-0209 B3: the harness has no module-level rtx_kit"
        monkeypatch.setattr(
            rig.h,
            "rtx_kit",
            lambda kit: dataclasses.replace(kit, search=backend, web_search="ddgs"),
        )
    elif case == "only-r2-without-a2":
        argv += ["--only", "R2"]
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = rig.h.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in REFUSAL_NAMES:
        assert re.search(REFUSAL_NAMES[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


@pytest.mark.parametrize("case", ["head", "no-t0208-a2-merge", "only-r1", "image-copy-intact"])
def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """B2, the other side of every refusal (a guard too wide): at HEAD (A1 merged, A2 absent, web on, v3, loop 20,
    every pin intact, a clean tree, the real ``rtx_kit``) ``main`` does not refuse; it asks for the engine (a socket
    attempt) and, finding none, exits 1. Neither A2 nor T-0208-a2 is required; ``--only R1`` is accepted; an intact
    copy of the images under ``RTX_SHARED_DIR`` passes."""
    rig = Rig(tmp_path, monkeypatch)
    argv = rig.argv()
    if case == "no-t0208-a2-merge":
        rig.merges = (*OTHERS, MERGE_A1)
    elif case == "only-r1":
        argv += ["--only", "R1"]
    elif case == "image-copy-intact":
        assert hasattr(rig.h, "RTX_SHARED_DIR"), (
            "T-0209 B1: the harness has no module-level RTX_SHARED_DIR"
        )
        monkeypatch.setattr(rig.h, "RTX_SHARED_DIR", _image_copy(tmp_path))
    rc = rig.h.main(argv)
    err = capsys.readouterr().err
    assert rc == 1, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


# =====================================================================================================
# B1-B5 through main: order.json, the manifest, a probe failing is void
# =====================================================================================================


class MeasureRig(Rig):
    """``Rig`` past the engine check: ``/models`` lists the served model, ``/metrics`` answers; VRAM sampling and the
    regime sources stubbed; P1-P3, P-R1 and P-R2 stubbed (passing unless a row says otherwise); ``_one_turn``
    recorded and never expected. Every real socket still refused and recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra.conductor.llm_client import VllmEndpoint

        super().__init__(tmp_path, mp)
        h = self.h
        model = VllmEndpoint().model
        self.probes = _probes()
        self.calls: list[str] = []
        self.order_on_disk: list[bool] = []

        def get_text(url: str, *a: Any, **k: Any) -> str:
            if url.rstrip("/").endswith("/models"):
                return json.dumps({"data": [{"id": model}]})
            return "engine-version-standin"

        _patch_everywhere(mp, h, "_get_text", get_text)
        _patch_everywhere(
            mp,
            h,
            "read_metrics",
            lambda *a, **k: {"prefix_cache_hits": 0, "prefix_cache_queries": 0},
        )
        mp.setattr(_chain(h, "T"), "_VramSampler", t205._FakeVram)
        mp.setattr(_chain(h, "P"), "regime_sources", lambda *a, **k: {})

        def run_probes(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("run_probes")
            self.order_on_disk.append((self.out / "order.json").is_file())
            return {n: dict(self.probes[n]) for n in ("p1", "p2", "p3")}

        def pr1(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("probe_pr1")
            return copy.deepcopy(self.probes["pr1"])

        def pr2(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("probe_pr2")
            return copy.deepcopy(self.probes["pr2"])

        async def one_turn(*a: Any, **k: Any) -> Any:
            self.calls.append("_one_turn")
            raise AssertionError("T-0209 rig: a turn ran")

        for seam, fn in (
            ("run_probes", run_probes),
            ("probe_pr1", pr1),
            ("probe_pr2", pr2),
            ("_one_turn", one_turn),
        ):
            assert hasattr(h, seam), f"T-0209 B: the harness has no module-level seam {seam!r}"
            mp.setattr(h, seam, fn)

    def run(self, capsys: Any, want: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rc = self.h.main(self.argv())
        err = capsys.readouterr().err
        assert rc == want, f"exit {rc}, want {want}\n{err[-3000:]}"
        assert self.sockets == [], self.sockets
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        return manifest, report


@pytest.mark.parametrize(
    "case", ["pr1-golden-fails", "pr1-trap-passes", "pr2-connector", "p1-failed"]
)
def test_probe_failure_is_void_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """B4: a failing P-R1, P-R2 (or P1) gives exit 3 and a ``report.json`` labelled ``void``; no turn runs."""
    rig = MeasureRig(tmp_path, monkeypatch)
    if case == "pr1-golden-fails":
        rig.probes["pr1"] = _mut_pr1("golden-fails")
    elif case == "pr1-trap-passes":
        rig.probes["pr1"] = {**_mut_pr1("trap-passes"), "passed": True}
    elif case == "pr2-connector":
        rig.probes["pr2"] = {**PR2_OK, "connector_calls": 1, "passed": True}
    elif case == "p1-failed":
        rig.probes["p1"] = {"ok": False}
    _manifest, report = rig.run(capsys, 3)
    assert report.get("label") == "void", report
    assert "_one_turn" not in rig.calls, rig.calls
    assert {"run_probes", "probe_pr1", "probe_pr2"} <= set(rig.calls), rig.calls


def test_order_json_written_before_the_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3 (reading 2): ``order.json`` is on disk when the probes run and holds ``order(0)`` R1-only: 6 entries."""
    rig = MeasureRig(tmp_path, monkeypatch)
    rig.probes["pr2"] = {**PR2_OK, "resolver_calls": 1}  # stop at the probes
    rig.run(capsys, 3)
    assert rig.order_on_disk == [True], rig.order_on_disk
    written = [tuple(x) for x in json.loads((rig.out / "order.json").read_text(encoding="utf-8"))]
    want = [("R1", a, r) for a in ARMS for r in range(3)]
    random.Random(0).shuffle(want)
    assert written == want


def _heads_live_default() -> dict[str, Any]:
    from local_astra import cli

    return {
        **t205._heads_live_default(),
        "completion_check": cli.completion_check_of(argparse.Namespace(), "live"),
    }


def test_heads_live_default_premise() -> None:
    """Premise (green): at BASE the live default is web on, v3, loop 20, the wait tool and the completion check on."""
    d = _heads_live_default()
    assert (
        d["web"],
        d["live_guidance"],
        d["max_tool_steps"],
        d["wait_tool"],
        d["completion_check"],
    ) == (
        True,
        "v3",
        20,
        True,
        True,
    ), d


@pytest.mark.parametrize("t0208_a2", ["merged", "absent"])
def test_manifest_records_heads_live_defaults_and_the_search_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    t0208_a2: str,
) -> None:
    """B1 (reading 5): ``live_default`` is HEAD's as ``cli`` resolves it (the wait tool and completion check among
    them); ``shared_check`` is ``True`` exactly when a ``Merge T-0208-a2:`` merge is reachable; ``r2`` is ``False``
    (only a branch merge into T-0209-a2-impl is in the log); ``search_backend`` is ``"rtx_loopback"`` (the session
    logs say ``ddgs``, H-T209-1); the table sha, the index sha and the R1 image's sha are recorded."""
    want = _heads_live_default()
    rig = MeasureRig(tmp_path, monkeypatch)
    if t0208_a2 == "absent":
        rig.merges = (*OTHERS, MERGE_A1)
    rig.probes["pr2"] = {
        **PR2_OK,
        "resolver_calls": 1,
    }  # stop at the probes: the manifest is written before them
    manifest, _ = rig.run(capsys, 3)
    got = manifest.get("live_default") or {}
    problems = {
        k: (got.get(k, "<absent>"), v) for k, v in want.items() if got.get(k, "<absent>") != v
    }
    assert not problems, f"B1: live_default differs from cli's: {problems}"
    assert manifest.get("shared_check") is (t0208_a2 == "merged"), manifest.get("shared_check")
    assert manifest.get("r2") is False, manifest.get("r2")
    assert manifest.get("search_backend") == "rtx_loopback", manifest.get("search_backend")
    blob = json.dumps(manifest)
    assert TABLE_SHA256 in blob, "B1: the table sha is not recorded"
    assert INDEX_SHA256 in blob, "B1: the index sha is not recorded"
    assert _rtx_row("R1").image_sha256 in blob, "B1: the R1 image's sha256 is not recorded"


def test_manifest_follows_cli_at_run_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B1 (never hard-coded): with ``cli.wait_tool_of`` and ``cli.completion_check_of`` giving off, the manifest
    records both off and the kit asked for has no wait tool."""
    from local_astra import cli

    monkeypatch.setattr(cli, "wait_tool_of", lambda a, m: False)
    monkeypatch.setattr(cli, "completion_check_of", lambda a, m: False)
    rig = MeasureRig(tmp_path, monkeypatch)
    rig.probes["pr2"] = {**PR2_OK, "resolver_calls": 1}
    manifest, _ = rig.run(capsys, 3)
    d = manifest.get("live_default") or {}
    assert d.get("wait_tool") is False and d.get("completion_check") is False, d
    assert rig.kits and not any(k.get("wait_tool") for k in rig.kits), rig.kits


# =====================================================================================================
# Drift: every live web keyword cli passes reaches the harness's Session
# =====================================================================================================


def test_no_live_web_keyword_of_the_cli_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B1 (adversarial, the T-0201 D drift): every setting the cli passes ``Session`` at HEAD for ``--tools live`` in
    web mode is in ``session_keywords`` with the cli's value; search is not a Session keyword (it rides the kit)."""
    from local_astra.config import load

    want = t205._cli_live_web_keywords(tmp_path, capsys, monkeypatch)
    assert want.get("completion_check") is True and want.get("screen_reminder") is True, sorted(
        want
    )
    got = dict(api("session_keywords")(load().profiles["lean"]))
    problems = sorted(k for k, v in want.items() if k not in got or got[k] != v)
    assert not problems, f"drift: lacking or differing from the cli's live web keywords: {problems}"
    assert "search" not in got and "web_search" not in got, sorted(got)


# =====================================================================================================
# B3 through the real ``_one_turn``: the two arms, the search kit, a fresh browser per turn (R1-J5)
# =====================================================================================================

ORIGIN = "http://127.0.0.1:59209"


class TurnRig:
    """``_one_turn``'s seams replaced (T-0208 C's rig): ``BrowserToolExecutor``, ``StartPageServer``, ``Session``,
    ``TurnMachine``, ``TurnPolicy`` by recording fakes; ``HudServer`` by a subclass of the real one whose ``open()``
    gives a socketless page that decodes each message and hands it to the **real** ``HudServer.route``;
    ``BrowserFrameSource.ingest`` wrapped to record its caller. ``cli.load_tool_kit`` gives a ``_StubKit``; T-0205's
    chain stubbed. Every real socket is refused and recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli
        from local_astra.computer import executor as executor_mod
        from local_astra.conductor import hud_server as hud_mod
        from local_astra.conductor import session as session_mod
        from local_astra.conductor import turn_machine as tm
        from local_astra.config import load
        from local_astra.contracts.codec import decode
        from local_astra.contracts.perception import Source
        from local_astra.contracts.protocol import AstraHello
        from local_astra.pages import server as server_mod
        from local_astra.perception.browser_source import BrowserFrameSource

        self.h = h = harness()
        self.profile = load().profiles["lean"]
        self.events: list[Any] = []
        self.sockets: list[str] = []
        self.executors: list[Any] = []
        self.servers: list[Any] = []
        self.sessions: list[Any] = []
        self.huds: list[Any] = []
        self.ingests: list[tuple[str, str, Any, bytes]] = []
        self.selects: list[tuple[str, str, str]] = []
        self.at_submit: dict[str, Any] = {}
        self.waits: list[tuple[str, float]] = []
        #: the keyword options of each ``cli.load_tool_kit`` call (T-0208 C review: the per-turn kit must be cli's)
        self.kit_opts: list[dict[str, Any]] = []
        self.state: Any = copy.deepcopy(GOLD)
        #: C-R2 rows: the select reaches the session but is not applied (the ingress self-check must fail), or the
        #: executor's ``open`` raises before any request
        self.ignore_select = False
        self.open_error: Exception | None = None
        self.BrowserFrameSource = BrowserFrameSource
        self.tmp = tmp_path
        rig = self

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                rig.sockets.append(what)
                raise ConnectionRefusedError(f"T-0209 turn rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        real_ingest = BrowserFrameSource.ingest

        def ingest(self_: Any, msg: Any) -> Any:
            f = sys._getframe(1)
            rig.ingests.append(
                (
                    Path(f.f_code.co_filename).name,
                    f.f_code.co_name,
                    msg.source,
                    base64.b64decode(msg.jpeg_b64),
                )
            )
            rig.events.append("ingest")
            return real_ingest(self_, msg)

        mp.setattr(BrowserFrameSource, "ingest", ingest)
        RealHud = hud_mod.HudServer

        class Page:
            def __init__(self, server: Any) -> None:
                self.server = server
                self.errors: list[Any] = []

            def receive(self, raw: str | bytes) -> None:
                message = decode(raw)
                if isinstance(message, AstraHello):
                    rig.events.append("hello")
                    return
                self.server.route(message, self)

            def _error(self, code: Any, message: Any = None) -> None:
                self.errors.append(code)

            def close(self, *a: Any, **k: Any) -> None:
                return None

        class SpyHud(RealHud):  # type: ignore[misc, valid-type]
            def __init__(self, session: Any, **kw: Any) -> None:
                super().__init__(session, **kw)
                rig.huds.append(self)

            async def start(self) -> None:
                raise AssertionError(
                    "T-0209 turn rig: the HUD server need not listen (socketless open())"
                )

            def open(self) -> Any:  # type: ignore[override]
                return Page(self)

        class FakePage:
            async def evaluate(self, js: str, *a: Any) -> Any:
                rig.events.append("state-read")
                return copy.deepcopy(rig.state)

        class FakeExec:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs
                self.on_note: Any = None
                self.egress_log: list[dict[str, Any]] = []
                self.driver = types.SimpleNamespace(page=FakePage())
                self.closed = 0
                rig.executors.append(self)

            async def open(self, url: str) -> None:
                rig.events.append(("open", id(self), url))
                if rig.open_error is not None:
                    raise rig.open_error

            async def execute(self, inv: Any) -> Any:
                raise AssertionError("T-0209 turn rig: nothing executes")

            async def close(self) -> None:
                self.closed += 1
                rig.events.append(("browser.close", id(self)))

        class FakeServer:
            def __init__(self, static_dir: Any = None) -> None:
                self.static_dir = static_dir
                self.stopped = 0
                rig.servers.append(self)

            @property
            def origin(self) -> str:
                return ORIGIN

            def start(self) -> None:
                rig.events.append("server.start")

            def stop(self) -> None:
                self.stopped += 1
                rig.events.append("server.stop")

        def policy(*args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(kind="policy", kwargs=kwargs)

        def machine(*args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(kind="machine", args=args, kwargs=kwargs)

        class FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args, self.kwargs = args, kwargs
                self.perception = args[5] if len(args) > 5 else kwargs.get("perception")
                self.session_log = kwargs.get("session_log")
                self.config = kwargs["config"]
                self.session_id = f"s-t0209-turn-rig-{len(rig.sessions)}"
                self.turn_written = asyncio.Event()
                self.failures: list[Any] = []
                self.state = types.SimpleNamespace(history=[])
                self.vision_source = "browser"
                self.publisher: Any = None
                self._task: Any = None
                rig.sessions.append(self)

            def set_publisher(self, fn: Any) -> None:
                self.publisher = fn

            def apply_vision_select(self, message: Any) -> Any:
                f = sys._getframe(1)
                source = str(getattr(message.source, "value", message.source))
                rig.selects.append((Path(f.f_code.co_filename).name, f.f_code.co_name, source))
                rig.events.append(("select", source))
                if not rig.ignore_select:
                    self.vision_source = source
                return types.SimpleNamespace(source=source)

            async def start(self) -> None:
                rig.events.append("session.start")
                await self.session_log.start()

            def submit(self, event: Any) -> None:
                rig.events.append(("submit", event.text))
                self._task = asyncio.get_running_loop().create_task(rig.script(self, event))

            async def aclose(self) -> None:
                if self._task is not None:
                    self._task.cancel()
                    with contextlib.suppress(BaseException):
                        await self._task
                await self.session_log.aclose()
                rig.events.append("session.aclose")

        def load_tool_kit(profile: Any, *, live_browser: Any = None, **opts: Any) -> Any:
            rig.kit_opts.append(dict(opts))
            sha = t205._set_sha(opts.get("page_overview", False), opts.get("wait_tool", False))
            return _StubKit(tools_version=sha), object()

        class FakeMeter:
            def __init__(self, inner: Any, **kw: Any) -> None:
                self.inner = inner
                self.requests: dict[str, Any] = {}
                self.generates: list[dict[str, Any]] = []

            async def aclose(self) -> None:
                return None

        async def wait_idle(*a: Any, **k: Any) -> bool:
            return True

        self.Source = Source
        mp.setattr(executor_mod, "BrowserToolExecutor", FakeExec)
        mp.setattr(server_mod, "StartPageServer", FakeServer)
        mp.setattr(session_mod, "Session", FakeSession)
        mp.setattr(hud_mod, "HudServer", SpyHud)
        mp.setattr(tm, "TurnPolicy", policy)
        mp.setattr(tm, "TurnMachine", machine)
        mp.setattr(cli, "load_tool_kit", load_tool_kit)
        mp.setattr(
            _chain(h, "F"), "_client", lambda *a, **k: types.SimpleNamespace(request_body=None)
        )
        _patch_everywhere(mp, h, "LoopMeter", FakeMeter)
        mp.setattr(_chain(h, "G"), "_wait_idle", wait_idle)
        mp.setattr(_chain(h, "G"), "_descendants", lambda: [])
        mp.setattr(_chain(h, "T"), "_healthy", lambda *a, **k: True)
        _patch_everywhere(mp, h, "non_loopback_sockets", lambda pids: [])
        mp.setattr(h, "asyncio", t205._AsyncioProxy(self.waits, lambda: 10.0))
        self.out = tmp_path / "out"
        (self.out / "sessions").mkdir(parents=True)

    async def script(self, s: Any, event: Any) -> None:
        frame = None
        if s.perception is not None and hasattr(s.perception, "get_frame"):
            frame = await s.perception.get_frame(self.Source.SCREEN)
        self.at_submit = {"vision_source": s.vision_source, "frame": frame, "text": event.text}
        log, tid = s.session_log, "t1"
        log.record(
            "llm.request",
            {"phase": "generate", "request_id": f"{tid}-gen", "attempt_id": "a1"},
            turn_id=tid,
        )
        line = {
            "session_id": s.session_id,
            "turn_id": tid,
            "extras": {"tool_loop_end": None, "tool_steps": 1, "vision_source": s.vision_source},
        }
        with Path(s.config.turns_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        s.state.history.append(types.SimpleNamespace(user_text=event.text, assistant_spoken="ok"))
        s.turn_written.set()

    def run(self, rid: str, arm: str, repeat: int = 0) -> dict[str, Any]:
        h = self.h
        assert hasattr(h, "_Run") and hasattr(h, "_one_turn"), (
            "T-0209 B3: the harness has no _Run/_one_turn"
        )
        known = {
            "base_url": "http://127.0.0.1:8001/v1",
            "out": self.out,
            "profile": self.profile,
            "sentinel": b"T0209-TURN-RIG-SENTINEL",
            "manifest": {},
            "vram": t205._FakeVram(),
            "r2": False,
        }
        names = {f.name for f in dataclasses.fields(h._Run)}
        run = h._Run(**{k: v for k, v in known.items() if k in names})
        trial, _steps, _thoughts = asyncio.run(h._one_turn(run, rid, arm, repeat))
        assert self.sockets == [], self.sockets
        return dict(trial)


def test_shared_arm_goes_through_the_production_ingress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3 (reading 3; reviewer mutation 9, the select skipped): one ``astra.frame`` with R1's image bytes is routed
    by ``HudServer.route`` into the ``BrowserFrameSource`` that is both the server's ``frame_sink`` and the session's
    perception; then one ``select shared`` routed by ``HudServer.route``; then ``UserText(request)``. At submit the
    session looks at ``shared`` and its perception holds the image's bytes (screen source, 1280x800). The executor
    opens ``/web.html`` on a server whose root is ``RTX_DIR``."""
    from local_astra.pages.rtx_tasks import RTX_DIR

    rig = TurnRig(tmp_path, monkeypatch)
    row = _rtx_row("R1")
    image = (IMAGE_DIR / row.image).read_bytes()
    trial = rig.run("R1", "shared")
    assert trial.get("exception") is None, trial.get("exception")
    s = rig.sessions[-1]
    assert isinstance(s.perception, rig.BrowserFrameSource), type(s.perception).__name__
    assert rig.huds and rig.huds[-1].frame_sink is s.perception and rig.huds[-1].session is s
    mine = [i for i in rig.ingests if i[3] == image]
    assert [(f, fn, src.value) for f, fn, src, _ in mine] == [
        ("hud_server.py", "route", "screen")
    ], rig.ingests
    assert [x for x in rig.selects if x[2] == "shared"] == [("hud_server.py", "route", "shared")], (
        rig.selects
    )
    ev = rig.events
    submit = ("submit", row.request)
    assert submit in ev, ev
    assert ev.index("ingest") < ev.index(("select", "shared")) < ev.index(submit), ev
    frame = rig.at_submit.get("frame")
    assert rig.at_submit.get("vision_source") == "shared", rig.at_submit
    assert frame is not None and frame.jpeg == image, (
        "B3: the session's perception does not hold the image"
    )
    assert (frame.width, frame.height) == (1280, 800), (frame.width, frame.height)
    assert [Path(sv.static_dir).resolve() for sv in rig.servers] == [Path(RTX_DIR).resolve()]
    opens = [(e[0], e[2]) for e in ev if isinstance(e, tuple) and e[0] == "open"]
    assert opens == [("open", ORIGIN + "/web.html")], opens
    assert (trial.get("task"), trial.get("arm")) == ("R1", "shared"), trial


def test_typed_arm_posts_no_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B3 (H-T209-9): no frame carrying the image is ingested, no ``shared`` select, the session looks at
    ``browser``, its perception holds no frame with the image's bytes; the request is ``control_request``."""
    rig = TurnRig(tmp_path, monkeypatch)
    row = _rtx_row("R1")
    image = (IMAGE_DIR / row.image).read_bytes()
    trial = rig.run("R1", "typed")
    assert trial.get("exception") is None, trial.get("exception")
    assert not [i for i in rig.ingests if i[3] == image], "B3: the typed arm posted the image"
    assert not [x for x in rig.selects if x[2] == "shared"], rig.selects
    assert rig.at_submit.get("vision_source") == "browser", rig.at_submit
    frame = rig.at_submit.get("frame")
    assert frame is None or frame.jpeg != image, "B3: the typed arm's perception holds the image"
    assert [e for e in rig.events if isinstance(e, tuple) and e[0] == "submit"] == [
        ("submit", row.control_request)
    ]
    assert (trial.get("task"), trial.get("arm")) == ("R1", "typed"), trial


def _void_alone(which: str, trial: dict[str, Any]) -> str:
    return str(impl(which, "build_report")([trial], [], _probes(), None, r2=False)["label"])


def test_ingress_self_check_failure_is_pr3_false_and_void(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3/B4 (T-0208 C-R2; adversarial): the shared select reaches the session through ``HudServer.route`` but is
    not applied, so the session still looks at ``browser``. The turn must not be asked (no submit); its exception
    is the harness's ``IngressError``; ``pr3`` is ``False``; and that trial alone voids the run, by the harness and
    by the oracle (which recognises the ingress failure from the recorded ``exception`` alone)."""
    rig = TurnRig(tmp_path, monkeypatch)
    rig.ignore_select = True
    trial = rig.run("R1", "shared")
    assert [x for x in rig.selects if x[2] == "shared"], (
        "the rig's select never reached the session"
    )
    assert not [e for e in rig.events if isinstance(e, tuple) and e[0] == "submit"], rig.events
    exc = str(trial.get("exception") or "")
    assert exc.startswith("IngressError("), exc
    assert trial.get("pr3") is False, trial.get("pr3")
    assert oracle.ingress_failed(trial) is True
    for which in ("harness", "eval-oracle"):
        assert _void_alone(which, trial) == "void", which


def test_other_exception_before_step1_is_pr3_none_and_not_void(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3/B4 (T-0208 C-R2, the other side): a shared turn whose browser fails to open raises before any request;
    ``pr3`` is ``None`` (left to ``ERRORED_STOP``) and that trial alone does not void the run, by the harness and by
    the oracle."""
    rig = TurnRig(tmp_path, monkeypatch)
    rig.open_error = RuntimeError("T-0209 turn rig: the browser died")
    trial = rig.run("R1", "shared")
    exc = str(trial.get("exception") or "")
    assert "the browser died" in exc and not exc.startswith("IngressError("), exc
    assert trial.get("pr3") is None, trial.get("pr3")
    assert oracle.ingress_failed(trial) is False
    for which in ("harness", "eval-oracle"):
        assert _void_alone(which, trial) == "measured", which


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_session_gets_every_cli_live_keyword_and_the_loopback_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arm: str
) -> None:
    """B1/B3 drift through the real ``_one_turn`` (both arms): every setting the cli passes ``Session`` for ``--tools
    live`` in web mode reaches this turn's ``Session`` with the cli's value; the session's kit has production's
    ``LoopbackSearch`` and ``web_search="ddgs"`` (search on through it); the executor is web mode with both deny-all
    proxy seams, the rtx origin as its start origin, and no profile directory; the turn waits 300 s."""
    from local_astra.pages.rtx_search import LoopbackSearch

    want = t205._cli_live_web_keywords(tmp_path / "cli", capsys, monkeypatch)
    rig = TurnRig(tmp_path, monkeypatch)
    rig.run("R1", arm)
    kwargs = rig.sessions[-1].kwargs
    got = {k: v for k, v in kwargs.items() if k not in t205.CLI_PLUMBING}
    problems = sorted(k for k, v in want.items() if k not in got or got[k] != v)
    assert not problems, (
        f"drift: the turn's Session lacks or differs on the cli's live keywords: {problems}"
    )
    kit = kwargs.get("tool_kit")
    assert kit is not None and type(kit.search) is LoopbackSearch, (
        f"B3: the turn's kit search is {kit!r}"
    )
    assert kit.web_search == "ddgs", kit
    kw = rig.executors[-1].kwargs
    assert kw.get("web") is True and kw.get("start_origin") == ORIGIN, kw
    assert callable(kw.get("proxy_resolve")) and callable(kw.get("proxy_connect")), sorted(kw)
    assert kw.get("user_data_dir") is None, (
        "R1-J5: a profile directory would carry localStorage across turns"
    )
    assert [t for q, t in rig.waits if q.endswith("Event.wait")] == [300], rig.waits


def _cli_live_kit_options(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, bool]:
    """The ``{wait_tool, page_overview}`` options the cli itself passes ``load_tool_kit`` at HEAD for ``session
    --tools live`` (web mode; T-0132's build rig, the real loader wrapped to record), absent keys read as ``False``.
    The rig's patches are undone before returning."""
    from local_astra import cli
    from tests.unit import test_cli_t0132 as c132

    seen: list[dict[str, Any]] = []
    real = cli.load_tool_kit

    def spy(profile: Any, *a: Any, **k: Any) -> Any:
        if k.get("live_browser") is not None:
            seen.append({n: v for n, v in k.items() if n != "live_browser"})
        return real(profile, *a, **k)

    monkeypatch.setattr(cli, "load_tool_kit", spy)
    calls, _ = c132.build(tmp_path / "cli-kit", capsys, monkeypatch)
    assert calls.graphs[-1].web is True, "premise: the cli built a web-mode live session"
    monkeypatch.undo()
    capsys.readouterr()
    assert len(seen) == 1, f"premise: the cli's live build loads one kit with a browser; {seen}"
    return {n: bool(seen[0].get(n, False)) for n in ("wait_tool", "page_overview")}


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_kit_options_equal_the_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arm: str
) -> None:
    """B1/B3 (T-0208 C review finding, repeated for T-0209 B; adversarial): the options ``_one_turn`` passes
    ``cli.load_tool_kit`` equal those the cli itself passes for ``--tools live`` at HEAD, ``{wait_tool,
    page_overview}``, on both arms. The harness's own tool-set self-check compares the kit with a sha derived from the
    same options, so it cannot catch a dropped ``wait_tool``; this row does. Hand premise at HEAD (CLAUDE.md: the
    ``wait`` tool on by default since T-0202 D; ``--page-overview`` off by default, T-0140): wait on, overview off;
    and the cli's resolvers agree (``cli.wait_tool_of(ns, "live")``, ``cli.page_overview_of``)."""
    from local_astra import cli
    from local_astra.config import load

    want = _cli_live_kit_options(tmp_path, capsys, monkeypatch)
    assert want == {"wait_tool": True, "page_overview": False}, want
    ns = argparse.Namespace(tools="live")
    lean = load().profiles["lean"]
    assert {
        "wait_tool": bool(cli.wait_tool_of(ns, "live")),
        "page_overview": bool(cli.page_overview_of(lean, ns, "live").page_overview),
    } == want
    rig = TurnRig(tmp_path, monkeypatch)
    trial = rig.run("R1", arm)
    assert trial.get("exception") is None, trial.get("exception")
    assert len(rig.kit_opts) == 1, rig.kit_opts
    got = {n: bool(rig.kit_opts[0].get(n, False)) for n in ("wait_tool", "page_overview")}
    assert got == want, (
        f"{arm}: the turn's kit options {rig.kit_opts[0]} differ from the cli's {want}"
    )


def test_not_run_turn_is_pr3_none_and_not_void() -> None:
    """Orchestrator ruling for T-0209 B (consistent with T-0208 C-R2): a turn the runner could not run (``_not_run``,
    e.g. a harness crash after the try) is an errored turn, not a void run, so its ``pr3`` is ``None`` on both arms
    and that trial alone leaves the run ``measured``, by the harness and by the oracle."""
    nr = api("_not_run")
    for arm in ARMS:
        trial = nr("R1", arm, 0, "T-0209 test: the harness crashed after the try")
        assert "pr3" in trial and trial["pr3"] is None, (arm, trial.get("pr3"))
        assert trial.get("arm") == arm, trial
        for which in ("harness", "eval-oracle"):
            assert _void_alone(which, dict(trial)) == "measured", (arm, which)


def test_each_turn_gets_a_fresh_browser_server_session_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3 + R1-J5 (adversarial: a harness that keeps one executor, server or index for the run): two turns, two
    distinct executors each opened once at ``/web.html`` and closed once before the next turn opens, two servers on
    ``RTX_DIR`` each stopped, two sessions, two distinct ``LoopbackSearch`` objects."""
    rig = TurnRig(tmp_path, monkeypatch)
    rig.run("R1", "shared", 0)
    rig.run("R1", "typed", 0)
    assert len(rig.executors) == 2 and rig.executors[0] is not rig.executors[1], rig.executors
    assert [e.closed for e in rig.executors] == [1, 1], [e.closed for e in rig.executors]
    first, second = (id(e) for e in rig.executors)
    ev = rig.events
    assert ev.index(("browser.close", first)) < ev.index(("open", second, ORIGIN + "/web.html")), ev
    assert [e for e in ev if isinstance(e, tuple) and e[0] == "open"] == [
        ("open", first, ORIGIN + "/web.html"),
        ("open", second, ORIGIN + "/web.html"),
    ]
    assert len(rig.servers) == 2 and [s.stopped for s in rig.servers] == [1, 1]
    assert len(rig.sessions) == 2
    searches = [s.kwargs["tool_kit"].search for s in rig.sessions]
    assert searches[0] is not searches[1], "B3: one LoopbackSearch shared across turns"


# =====================================================================================================
# B6: the locked wrapper (T-0131's run_locked rig, as T-0205's and T-0208's rows)
# =====================================================================================================

L_OUT = f"logs/t0209-rtx-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0209_rtx_run.py"
L_EVAL = "tests/eval/test_t0209_rtx_run.py"
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0205_demo_run.py",
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0132_crop_budget_ab.py",
    "scripts/gpu/t0131_grounding_ab.py",
    "scripts/gpu/t0111g_live_smoke.py",
    "scripts/gpu/t0138_suite_ab.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_wait.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/tools_v1_overview_wait.json",
    "local_astra/pages/scenario_tasks.py",
    "local_astra/pages/suite_tasks.py",
    "local_astra/pages/rtx_tasks.py",
    "local_astra/pages/rtx_search.py",
    "local_astra/pages/rtx_shared/r1_blender_error.jpg",
    "local_astra/pages/rtx_shared/MANIFEST.md",
    "local_astra/eval/cd/stats.py",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch,
    merges: tuple[str, ...] = (*OTHERS, MERGE_0208_A2, MERGE_A1, MERGE_B),
) -> Any:
    assert LOCKED.is_file(), f"T-0209 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", t205._git_stub(merges))
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0209_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json order.json; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


MERGE_CASES: dict[str, tuple[str, ...]] = {
    "no-a1-only-lookalikes": (*OTHERS, MERGE_0208_A2, MERGE_B),
    "no-a1-branch-merges-only": OTHERS,
}


@pytest.mark.parametrize("case", list(MERGE_CASES))
def test_locked_script_needs_merge_a1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """B6 (adversarial, reading 6): without ``Merge T-0209-a1:`` -- the log holding T-0207-c's, T-0208's, a made-up
    ``Merge T-0209-b:``, the branch merges into the T-0209 branches and a ``Merge T-0209-a1b:`` -- the script refuses
    (exit 2, ``REFUSED``) before anything starts."""
    res = _locked(monkeypatch, MERGE_CASES[case])(tmp_path)
    assert res.rc == 2 and "REFUSED" in res.output, (case, res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_refuses_vllm_sh_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B6 (T-0205 C's boot regime): ``services/vllm.sh`` changed since 806da12 refuses before the boot."""
    res = _locked(monkeypatch)(tmp_path, vllm_drift=True)
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert "services/vllm.sh changed since 806da12" in res.output, res.output[-2000:]
    assert res.started() == [], f"ran before the refusal: {res.started()}"


LOCKED_REFUSALS: dict[str, dict[str, Any]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """B6 (T-0205 C's boot, verbatim, so its refusals too): ``REFUSED``, exit 2, before the boot or any step."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


@pytest.mark.parametrize("merges", ["all", "a1-only"])
def test_locked_script_steps_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, merges: str
) -> None:
    """B6, the other side of every refusal (``a1-only``: no T-0208-a2 and no T-0209-b merge, neither required): the
    harness under ``timeout -k 60s 45m`` with a loopback ``--base-url`` and ``--out-dir logs/t0209-rtx-<sha7>``, then
    ``tests/eval/test_t0209_rtx_run.py -m gpu`` with ``T0209_RUN_DIR`` set to it; ``--locked`` on every ``uv run``;
    exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    chosen = (*OTHERS, MERGE_0208_A2, MERGE_A1, MERGE_B) if merges == "all" else (*OTHERS, MERGE_A1)
    res = _locked(monkeypatch, chosen)(tmp_path)
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
    assert evals[0].env.get("T0209_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv) and len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 45 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_shape() -> None:
    """B6 by text: ``bash -n`` clean; the boot is T-0205 C's verbatim; the ``Merge T-0209-a1:`` check through a
    here-string; no T-0205 merge check left from the fork; ``timeout -k 60s 45m``; ``--locked`` on every ``uv run``;
    the eval only if ``report.json`` exists; T-0205's eval and run dir never named."""
    assert LOCKED.is_file(), f"T-0209 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    code = t205._script_code(LOCKED)
    body = "\n".join(code)
    assert t205._boot(code) == t205._boot(t205._script_code(T0205_LOCKED)), (
        "B6: the boot is not T-0205 C's verbatim"
    )
    assert "timeout -k 60s 45m" in body
    checks = [ln for ln in code if "Merge T-0209" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("T-0209-a1:" in ln for ln in checks), checks
    assert not [ln for ln in code if "Merge T-0205" in ln], "the fork kept T-0205's merge checks"
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert "t0209-rtx-" in body and L_EVAL in body and L_HARNESS in body and "T0209_RUN_DIR" in body
    assert "-m gpu" in body and "report.json" in body
    for old in ("test_t0205_demo_run", "T0205_RUN_DIR", "t0205-demo-"):
        assert old not in body, f"the script names {old}"


# =====================================================================================================
# Dress row: the harness's real ``_one_turn`` on a real Session, HudServer + BrowserFrameSource, the real web-mode
# executor on the rtx root, the real ``LoopbackSearch``, a scripted fake LLM (browser; CPU)
# =====================================================================================================

DRESS_GATE = "ASTRA_BROWSER_TESTS"
_QQ = '<|"|>'
CTRL_L = f"<|tool_call>call:press_key{{combo:{_QQ}Ctrl+L{_QQ}}}<tool_call|>"
DRESS_TYPED_SAY = "I will look that error up. "
DRESS_TYPED_FINAL = (
    "Blender 4.2.3 has no kernels for compute capability 12.0. Install Blender 4.4, then choose OptiX under Edit, "
    "Preferences, System, Cycles Render Devices, as the second reply on blenderartists.org says."
)


def _search_call(query: str) -> str:
    return f"<|tool_call>call:google_search{{query:{_QQ}{query}{_QQ}}}<tool_call|>"


@dataclasses.dataclass(frozen=True)
class _Go:
    """A step placeholder: type ``address`` into the start page's address box with Enter, its coordinates read from
    the live page (``data-scn="address"``) when the step's request is sent."""

    address: str


class DressRig:
    """Only the engine half of ``_one_turn`` is replaced (T-0208 C's dress): ``F._client`` returns a scripted
    ``FakeLlm`` (plus the real client's ``request_body``); ``G._wait_idle``/``G.read_metrics``/``T._healthy`` answer as
    an idle healthy engine. ``Session``, ``HudServer`` and ``BrowserToolExecutor`` are subclasses of the real classes
    that only record; ``StartPageServer``, ``BrowserFrameSource``, ``TurnMachine``, ``cli.load_tool_kit``, the
    harness's ``rtx_kit`` and ``LoopbackSearch`` are untouched."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra.computer import executor as executor_mod
        from local_astra.conductor import hud_server as hud_mod
        from local_astra.conductor import session as session_mod
        from local_astra.conductor.llm_client import VllmClient
        from local_astra.config import load
        from tests.fakes.llm import FakeLlm, _refusing_transport
        from tests.integration.test_computer_live_session import type_at
        from tests.integration.test_t0205_scenarios_browser import BOX_JS, H, W, unit
        from tests.unit._tool_turn_support import answer, tool

        self.h = h = harness()
        self.profile = load().profiles["lean"]
        self.tmp = tmp_path
        self.sessions: list[Any] = []
        self.session_kwargs: list[dict[str, Any]] = []
        self.selects: list[str] = []
        self.llms: list[Any] = []
        self.execs: list[Any] = []
        self.plans: list[dict[str, Any]] = []
        self.answer, self.tool = answer, tool
        rig = self

        class DressLlm(FakeLlm):
            def request_body(self, prompt: Any, request_id: str, *a: Any, **k: Any) -> Any:
                return VllmClient(transport=_refusing_transport()).request_body(
                    prompt, request_id, None, {}
                )

            async def generate(  # type: ignore[override]
                self,
                prompt: Any,
                request_id: str,
                options: Any = None,
                xargs: Any = None,
                *,
                cd: Any = None,
            ) -> Any:
                if (
                    re.search(r"-gen-s\d+", request_id)
                    and self.step_scripts
                    and isinstance(self.step_scripts[0], _Go)
                ):
                    go = self.step_scripts[0]
                    page = rig.execs[-1].driver.page
                    b = await page.evaluate(BOX_JS, "address")
                    assert b is not None, f'T-0209 dress: no data-scn="address" on {page.url}'
                    x, y = unit(b["left"], b["right"], W), unit(b["top"], b["bottom"], H)
                    self.step_scripts[0] = [answer(""), tool(type_at(x, y, go.address, True))]
                async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
                    yield c

        def client(*a: Any, **k: Any) -> Any:
            plan = rig.plans[len(rig.llms)]
            llm = DressLlm()
            llm.script = []  # captions say nothing
            llm.generate_scripts = [[answer(plan["say"]), tool(_search_call(plan["query"]))]]
            llm.step_scripts = [
                [answer(""), tool(CTRL_L)],
                _Go(plan["address"]),
                [answer(plan["final"])],
            ]
            llm.check_scripts = [[answer(plan["final"])]]
            rig.llms.append(llm)
            return llm

        RealSession = session_mod.Session
        RealHud = hud_mod.HudServer
        RealExec = executor_mod.BrowserToolExecutor

        class SpySession(RealSession):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                rig.sessions.append(self)
                rig.session_kwargs.append(dict(k))

            def apply_vision_select(self, message: Any) -> Any:
                rig.selects.append(str(getattr(message.source, "value", message.source)))
                return super().apply_vision_select(message)

        class SpyExec(RealExec):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                rig.execs.append(self)

        async def wait_idle(*a: Any, **k: Any) -> bool:
            return True

        mp.setattr(session_mod, "Session", SpySession)
        mp.setattr(hud_mod, "HudServer", RealHud)
        mp.setattr(executor_mod, "BrowserToolExecutor", SpyExec)
        mp.setattr(_chain(h, "F"), "_client", client)
        mp.setattr(_chain(h, "G"), "_wait_idle", wait_idle)
        mp.setattr(_chain(h, "G"), "read_metrics", lambda *a, **k: None)
        mp.setattr(_chain(h, "T"), "_healthy", lambda *a, **k: True)

    def run(self, arm: str, plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.plans.append(plan)
        h = self.h
        out = self.tmp / f"out-{arm}"
        (out / "sessions").mkdir(parents=True)
        known = {
            "base_url": "http://127.0.0.1:8001/v1",
            "out": out,
            "profile": self.profile,
            "sentinel": b"T0209-DRESS-SENTINEL",
            "manifest": {},
            "vram": t205._FakeVram(),
            "r2": False,
        }
        names = {f.name for f in dataclasses.fields(h._Run)}
        run = h._Run(**{k: v for k, v in known.items() if k in names})
        trial, steps, _thoughts = asyncio.run(h._one_turn(run, "R1", arm, 0))
        trial = dict(trial)
        trial["_out"] = out
        return trial, list(steps)


def _dress_bodies(
    llm: Any,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """(step-1 bodies, later bodies) of one turn: every non-caption generate and every prefill, as its wire body;
    "later" is a ``-gen-s<k>`` id (steps, re-asks, checks)."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit.test_session_live import _wire

    first: list[tuple[str, dict[str, Any]]] = []
    later: list[tuple[str, dict[str, Any]]] = []
    items = [(rid, p) for rid, p, _o in llm.generates if not is_caption_prompt(p)]
    items += [(f"prefill:{rid}", p) for rid, p in llm.prefills]
    for rid, prompt in items:
        (later if re.search(r"-gen-s\d+", rid) else first).append((rid, _wire(prompt)))
    return first, later


def _tool_names(body: dict[str, Any]) -> list[str]:
    out = []
    for t in body.get("tools") or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        out.append(str(fn.get("name")))
    return out


def _search_notes(trial: dict[str, Any]) -> list[dict[str, Any]]:
    name = trial.get("session_log")
    assert name, f"the trial names no session log: {sorted(trial)}"
    path = Path(trial["_out"]) / "sessions" / str(name)
    recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [
        r.get("data") or {}
        for r in recs
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == "web search"
    ]


@pytest.mark.browser
@pytest.mark.integration
def test_dress_r1_shared_and_typed_through_the_real_one_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3/B4 dress (R1, one ``shared`` and one ``typed`` turn, n = 1 each) through the harness's own ``_one_turn``.
    Script per turn: step 1 says one sentence and calls ``google_search(<the error string>)`` (answered by the real
    ``LoopbackSearch``); step 2 Ctrl+L; step 3 ``type_at`` an address (coordinates from the live ``data-scn="address"``
    box) with Enter; step 4 a no-call answer; a completion check, if armed, answers the same.

    Shared turn (golden: the issue page): the session looked at ``shared``; step 1's wire body carries exactly the R1
    file's bytes and ``SHARED_LABEL_SCREEN``, and offers ``google_search``; step 2 (after the search, no browser
    action yet) still carries only the shared image with its label; no body from step 3 on carries the shared base64;
    the harness's ``pr3`` is true; exactly one ``web search`` note (backend ``ddgs``, the query verbatim); ``visited``
    holds the issue page, and the judge (the oracle's ``verdict``) succeeds on the trial.

    Typed turn (the forum thread), after the shared turn (R1-J5): no select, no shared byte, ``pr3`` ``None``; its
    ``visited`` starts at ``web``, holds ``forum-thread`` and holds **none** of the shared turn's pages (a carried-over
    ``localStorage`` would hold ``issue-blackwell``); a distinct ``LoopbackSearch`` and executor. Both: no exception,
    no session failure, zero resolver/connector calls, no non-loopback socket.

    Regime: Playwright's pinned Chromium headless, web mode on loopback, the harness's deny-all seams, HEAD's live
    defaults through cli, real ``SessionClock``; scripted fake LLM, no engine; n = 1 per arm."""
    import time

    from local_astra.pages.rtx_search import LoopbackSearch

    harness()  # named "does not exist yet" before the gate, so --pending shows it without a browser
    if os.environ.get(DRESS_GATE) != "1":
        pytest.skip(
            f"{DRESS_GATE} is not 1: the dress row drives a real Chromium (T-0209 B, dress row)"
        )
    image = (IMAGE_DIR / _rtx_row("R1").image).read_bytes()
    shared_b64 = base64.b64encode(image).decode("ascii")
    shared_url = "data:image/jpeg;base64," + shared_b64

    rig = DressRig(tmp_path, monkeypatch)
    t0 = time.monotonic()
    trial, _steps = rig.run(
        "shared", {"say": SAY + " ", "query": ERROR, "address": ISSUE_ADDR, "final": GOLDEN}
    )
    shared_wall = time.monotonic() - t0
    assert trial.get("exception") is None, trial.get("exception")
    assert len(rig.sessions) == 1 and len(rig.llms) == 1, (rig.sessions, rig.llms)
    assert not rig.sessions[0].failures, rig.sessions[0].failures
    assert rig.selects == ["shared"], rig.selects
    first, later = _dress_bodies(rig.llms[0])
    gens1 = [(rid, b) for rid, b in first if not rid.startswith("prefill:")]
    assert gens1, f"no step-1 generate was sent; trial {trial}"
    for rid, body in gens1:
        from tests.unit._live_browser_support import image_urls

        assert image_urls(body) == [shared_url], (
            f"{rid}: step 1 does not carry exactly the shared image"
        )
        assert SHARED_LABEL_SCREEN in json.dumps(body), f"{rid}: no SHARED_LABEL_SCREEN at step 1"
        assert "google_search" in _tool_names(body), _tool_names(body)
    s2 = [(rid, b) for rid, b in later if re.search(r"-gen-s2(?!\d)", rid)]
    rest = [(rid, b) for rid, b in later if not re.search(r"-gen-s2(?!\d)", rid)]
    assert s2 and rest, f"premise: the scripted path reached steps 2 and 3: {[r for r, _ in later]}"
    for rid, body in s2:
        from tests.unit._live_browser_support import image_urls

        urls = image_urls(body)
        assert urls and set(urls) == {shared_url}, (
            f"{rid}: step 2 (after the search) is not the shared image only"
        )
        assert SHARED_LABEL_SCREEN in json.dumps(body), rid
    for rid, body in rest:
        assert shared_b64 not in json.dumps(body), (
            f"{rid}: a body after the first browser action carries the shared image"
        )
    assert trial.get("pr3") is True, f"P-R3 is {trial.get('pr3')!r}"
    notes = _search_notes(trial)
    print(f"T-0209 B dress web search notes: {json.dumps(notes)}")
    assert (
        len(notes) == 1 and notes[0].get("backend") == "ddgs" and notes[0].get("query") == ERROR
    ), notes
    st = trial.get("state") or {}
    assert st.get("visited", [None])[0] == "web" and "issue-blackwell" in st.get("visited", []), st
    assert oracle.verdict(trial).success is True, (oracle.verdict(trial), trial.get("voiced"))
    assert trial.get("resolver_calls") == 0 and trial.get("connector_calls") == 0, trial
    assert not trial.get("non_loopback"), trial.get("non_loopback")

    t1 = time.monotonic()
    typed, _tsteps = rig.run(
        "typed",
        {
            "say": DRESS_TYPED_SAY,
            "query": ERROR,
            "address": THREAD_ADDR,
            "final": DRESS_TYPED_FINAL,
        },
    )
    typed_wall = time.monotonic() - t1
    assert typed.get("exception") is None, typed.get("exception")
    assert len(rig.sessions) == 2 and not rig.sessions[1].failures, rig.sessions[1].failures
    assert rig.selects == ["shared"], f"the typed turn sent a select: {rig.selects}"
    first2, later2 = _dress_bodies(rig.llms[1])
    for rid, body in first2 + later2:
        assert shared_b64 not in json.dumps(body), f"typed {rid}: carries the shared image"
    assert typed.get("pr3") is None, typed.get("pr3")
    tv = (typed.get("state") or {}).get("visited") or []
    assert tv and tv[0] == "web" and "forum-thread" in tv, tv
    assert not ({"issue-blackwell", "notes-4-4"} & set(tv)), (
        f"R1-J5: the typed turn's visited carried over: {tv}"
    )
    assert len(rig.execs) == 2 and rig.execs[0] is not rig.execs[1], rig.execs
    kits = [k.get("tool_kit") for k in rig.session_kwargs]
    assert all(type(k.search) is LoopbackSearch and k.web_search == "ddgs" for k in kits), kits
    assert kits[0].search is not kits[1].search, "B3: one LoopbackSearch across turns"
    assert typed.get("resolver_calls") == 0 and typed.get("connector_calls") == 0, typed
    assert not typed.get("non_loopback"), typed.get("non_loopback")
    print(
        f"T-0209 B dress: shared {shared_wall:.1f} s (steps {trial.get('tool_steps')}, end {trial.get('tool_loop_end')}, "
        f"checks {trial.get('completion_checks')}, visited {st.get('visited')}), typed {typed_wall:.1f} s (visited {tv})"
    )


def test_t0209_task_names_this_module() -> None:
    """Activation bookkeeping (tests.md): the task file names both B modules under "Tests to write"."""
    text = (REPO / "docs" / "tasks" / "T-0209.md").read_text(encoding="utf-8")
    tests = text.split("## Tests to write", 1)[1].split("\n## ", 1)[0]
    assert "tests/unit/test_t0209_rtx_harness.py" in tests
    assert "tests/eval/test_t0209_rtx_run.py" in tests
