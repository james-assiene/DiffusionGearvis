"""T-0208 group C: the GPU handoff run's harness (pure half, refusals, manifest, the two arms' ingress) and its locked
wrapper (unit; CPU; no engine, no browser, no socket). docs/tasks/T-0208.md C1-C7, "Tests to write" (C).

Activated at C's ``/implement`` against 988ffc7e (the ``pending`` markers below are gone). One extra row, the dress
row (``test_dress_h1_...``, ``browser`` + ``integration``, runs only with ``ASTRA_BROWSER_TESTS=1``), drives the
harness's real ``_one_turn`` on a real ``Session``/``HudServer``/``BrowserFrameSource`` and the real web-mode executor
with a scripted fake LLM; it is the only row here that opens a browser.

BASE e3f4e424 (main, ``T-0207 C merged``; T-0208 A1, B, A2, T-0202 D and T-0207 C merged). Every harness row needs
``scripts/gpu/t0208_handoff_run.py`` and every wrapper row ``scripts/locked/t0208_handoff_run.sh``; neither exists at
BASE, so those rows carry ``pending(task="T-0208")`` and, run with ``--pending``, are red on a named assertion in
``harness()``/``api()``/``_locked()`` ("does not exist yet" / "has no ..."), never on an import error (every harness
symbol is looked up inside the row). The pure rows also run against the eval's oracle (``tests/eval/
test_t0208_handoff_run.py``, ``eval-oracle``, not pending), so a wrong oracle is caught on CPU. Every expected value
is computed by hand (comments say how) or comes from production code the rule does not own (``redact`` builds the
logged image parts), never read off an implementation.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not; the shapes of the pure functions are the eval oracle's
docstrings, which these rows also run).

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``HANDOFF_IDS`` (H1-H3); ``ARMS = ("shared",
    "typed")``; ``REPEATS = 3``; ``ORDER_SEED = 0``; ``TOTAL_TURNS = 18``; ``TURN_TIMEOUT_S = 300``; ``RUN_BUDGET_S =
    50 * 60``; ``HANDOFF_TABLE_SHA256`` (the literal); ``SHARED_FRAME_REASON``; ``CANNOT_SEE`` (C5's pattern);
    ``VOID``, ``INCOMPLETE``, ``MEASURED``; ``HANDOFF_DIR`` (the image directory, read at call time).

Pure functions (both the harness and the oracle)
    ``order(seed) -> [(handoff_id, arm, repeat)]``; ``ph1_ok(record)``, ``ph2_ok(record)``; ``ph3_ok(records,
    image_bytes)``; ``identity(trial, steps) -> {"identity_spoken", "identity_typed", "identity_lost"}``;
    ``step1_action(steps) -> (tool | None, outcome)`` (a ``{"tool", "outcome"}`` mapping is accepted too);
    ``shared_refusals(steps) -> int``; ``cannot_see(trial) -> bool``; ``build_report(trials, steps, probes,
    stopped=None) -> {"label", "stopped", "probes", "per_arm": {arm: {handoff_id: counts}}}``; ``session_keywords
    (profile)`` (T-0205's, or the harness's own, every live web keyword cli passes ``Session``).

Seams ``main`` and ``_one_turn`` read at call time
    ``_git``; every ``cli`` resolver, ``cli.load_tool_kit``, ``cli.live_prerequisites``; ``G._get_text``,
    ``G.read_metrics``, ``T._VramSampler``, ``P.regime_sources`` (T-0205's chain, reachable as ``h.F``/``h.G``/
    ``h.T``/``h.P`` or through the imported T-0205 module); ``run_probes``, ``probe_ph1()``, ``probe_ph2(ph1)``;
    ``_one_turn(run, handoff_id, arm, repeat) -> (trial, steps, thoughts)`` over ``_Run`` (T-0205's fields);
    ``HudServer`` (``local_astra.conductor.hud_server``), ``BrowserFrameSource``, ``Session``, ``StartPageServer``,
    ``BrowserToolExecutor``, ``TurnMachine``/``TurnPolicy`` imported at call time from their modules (as T-0205 C).
    ``main(argv)`` takes ``--base-url`` and ``--out-dir``.

**Readings** (reported to the orchestrator; not resolved silently)
    1. C2's refusals, all exit 2 before any socket: a non-loopback ``--base-url``, a dirty tree, an existing out-dir,
       no ``Merge T-0208-b:`` merge reachable from HEAD (a branch merge ``Merge commit '...' into T-0208-b-impl`` or a
       ``Merge T-0208-bb:`` does not count), the table off ``HANDOFF_TABLE_SHA256``, an image whose bytes are off its
       row's ``image_sha256`` (read from ``HANDOFF_DIR``), web mode off, guidance not v3, loop not 20. A2 is not
       required (C records it).
    2. ``order`` is handoff-major over ``HANDOFF_IDS`` x ``ARMS`` x ``range(3)``, then one ``random.Random(0)
       .shuffle`` (T-0205's construction with the arm inside the handoff). The task says only "shuffled across arms
       with seed 0"; this construction is the test-author's.
    3. The ``shared`` arm (C3): one real ``BrowserFrameSource`` is the session's perception (``Session``'s sixth
       positional, as T-0205's ``FakePerception``) **and** the ``HudServer``'s ``frame_sink``; the image is posted as
       one ``astra.frame`` that ``HudServer.route`` hands to ``ingest`` (a socketless ``HudServer(...).open()``
       connection, T-0177's ``vision_rig`` shape; ``start()`` is never needed); then one ``astra.vision.select
       {source: "shared"}`` that ``HudServer.route`` hands to ``Session.apply_vision_select``; then
       ``UserText(request)``. The stored frame is the file's bytes (Q-2: the ingress does not re-encode). The
       ``typed`` arm: no frame carrying the image is ingested, no ``shared`` select, ``UserText(control_request)``.
    4. P-H3, identity, the step-1 classifier, "cannot see": the eval oracle's readings 3-6 (its docstring). P-H3 is
       amended by T-0211 A1 (2026-10-30) to T-0209 B4's P-R3 switch rule, read in ``seq`` order (the ``_ph3_case``
       table is re-pinned; A2 compares it with P-R3 on both tables; A4's ``search_kit`` seam rows follow it). Amended
       by C-R1 (2026-10-30): an executed ``wait`` **is** an "executed browser action" for ``identity_lost`` (a wait
       swaps the shared picture out); the row ``c-r1-wait-then-no-call-is-lost`` pins it, and a refused wait is not
       acting. C-R2: a shared turn whose exception is not an ingress self-check failure and that sent no step-1
       request records ``ph3: None`` and voids nothing; an ingress self-check failure (the session not on
       ``shared``, ``frames_rejected > 0``) records ``ph3: False`` (void). C-R3: a step-1 stream that errored after
       its request was logged is judged by the bodies (``ph3: True`` when they hold), the engine error counted.
    5. The manifest records ``live_default`` (T-0205's keys plus ``completion_check``), ``shared_check`` (whether a
       ``Merge T-0208-a2`` merge is reachable from HEAD), the table sha and each image's sha256 (anywhere in the
       manifest), and search ``False``/``"off"``.
    6. The wrapper's ``Merge T-0208-b:`` check accepts that subject only; the T-0208-c merge and A2 are neither
       required nor refused. Its boot is T-0205 C's verbatim (the code lines from ``make plugin-wheel`` to ``device
       memory after boot``, less lines naming a task); it checks only files listed in ``L_PLACEHOLDERS``.

Regime: CPU, pure, hand-built records and a stubbed ``main``/``_one_turn``; no engine, no browser; n = 1.
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
import random
import re
import shutil
import socket
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.telemetry.session_log import redact
from tests.eval import test_t0208_handoff_run as oracle
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0208_handoff_judge as J

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0208_handoff_run.py"
LOCKED = REPO / "scripts" / "locked" / "t0208_handoff_run.sh"
EVAL = REPO / "tests" / "eval" / "test_t0208_handoff_run.py"
T0205_LOCKED = REPO / "scripts" / "locked" / "t0205_demo_run.sh"
IMAGE_DIR = REPO / "local_astra" / "pages" / "handoff"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

IDS = ("H1", "H2", "H3")
ARMS = ("shared", "typed")
#: tests/unit/test_t0208_handoff_judge.py's hand-computed literal
TABLE_SHA256 = J.TABLE_SHA256
#: T-0177 A7 / B2, T-0192 A3, T-0204 B3, typed from the task files
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
MOVED_REASON = "the screen changed since this frame; take a new frame"
L_NOW = "[This screenshot is the browser screen now.]"
#: C5, typed from the task
CANNOT_SEE = r"can(?:'|no)t see|cannot see|unable to see"
#: the real merge subjects on main (``git log --merges``), and made-up lookalikes
MERGE_B = (
    "43eb2f9a9148f990c99d4db0a416fec5ee076ea2 Merge T-0208-b: loopback shared-screen handoff scenarios H1-H3 (drawn "
    "images, judge, real ingress)"
)
MERGE_A2 = (
    "60deb9c476d2664d7b7684ad0cb3a56d91e14b52 Merge T-0208-a2: completion check on shared-start turns once the step "
    "shows a browser frame; a wait alone does not arm (R-A2)"
)
MERGE_C = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0208-c: the GPU handoff run harness"
OTHERS = (
    "b64a5cf1974f917ce1ced3de8d36bf539d1809be Merge T-0208-a1: Share click to the real session, end to end (tests only)",
    "f6ea68d3c96d4a0afc97f87413d699c197e7936d Merge commit 'ec698e0c' into T-0208-b-impl",
    "d41c0e37a5d750cf114afe37810bb8d70913ce0a Merge branch 'main' into T-0208-a2-tests3",
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge branch 'T-0208-b-handoff' into T-0208-b-impl",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0208-bb: something else",
    "e89189d3bd76846cf274e299ac9d209584cf46b2 Merge T-0205-c: loopback GPU demo-run harness (D1-D9 x3, HEAD live "
    "defaults, report-only, locked)",
)


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0208 C1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0208_handoff_run_unit", HARNESS)
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
        f"T-0208 C: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api(name)
    return getattr(oracle, name)


def _tree() -> ast.Module:
    assert HARNESS.is_file(), f"T-0208 C1: {HARNESS.relative_to(REPO)} does not exist yet"
    return ast.parse(HARNESS.read_text(encoding="utf-8"))


def _t0205_modules(h: ModuleType) -> list[ModuleType]:
    """The T-0205 harness module(s) the T-0208 harness holds (C1 lets it import T-0205's helpers)."""
    return [
        v
        for v in vars(h).values()
        if isinstance(v, ModuleType)
        and str(getattr(v, "__file__", "") or "").endswith("t0205_demo_run.py")
    ]


def _chain(h: ModuleType, name: str) -> ModuleType:
    """T-0205's ``F``/``G``/``T``/``P`` as the T-0208 harness reaches them (its own name, or the T-0205 module's)."""
    found = getattr(h, name, None)
    if isinstance(found, ModuleType):
        return found
    for d in _t0205_modules(h):
        found = getattr(d, name, None)
        if isinstance(found, ModuleType):
            return found
    raise AssertionError(f"T-0208 C1: the harness reaches no {name!r} (T-0205's chain) by name")


def _patch_everywhere(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    """``name`` replaced on the harness and on every T-0205/T-0111g module it holds that has it."""
    mp.setattr(h, name, value, raising=False)
    targets = [*_t0205_modules(h)]
    with contextlib.suppress(AssertionError):
        targets.append(_chain(h, "G"))
    for t in targets:
        if hasattr(t, name):
            mp.setattr(t, name, value)


def _values_of(obj: Any, key: str) -> list[Any]:
    return t205._values_of(obj, key)


# =====================================================================================================
# Premises (green at BASE): the typed constants are production's; the eval imports only the judge
# =====================================================================================================


def test_reasons_and_table_premise() -> None:
    """Premise: the label, reasons and table sha typed here and in the eval oracle are production's, byte for byte;
    each committed image is its row's sha."""
    from local_astra.computer import executor
    from local_astra.conductor import session
    from local_astra.pages import handoff_tasks

    assert session.SHARED_LABEL_SCREEN == SHARED_LABEL_SCREEN == oracle.SHARED_LABEL_SCREEN
    assert session.SHARED_FRAME_REASON == SHARED_FRAME_REASON == oracle.SHARED_FRAME_REASON
    assert executor.CHECKOUT_REASON == CHECKOUT_REASON == oracle.CHECKOUT_REASON
    assert executor.MOVED_REASON == MOVED_REASON == oracle.MOVED_REASON
    assert handoff_tasks.handoff_table_sha() == TABLE_SHA256 == oracle.HANDOFF_TABLE_SHA256
    assert oracle.CANNOT_SEE == CANNOT_SEE
    for row in handoff_tasks.HANDOFFS:
        data = (IMAGE_DIR / row.image).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row.image_sha256, row.image
        assert oracle.IMAGES[row.handoff_id] == row.image
        assert tuple(t.casefold() for t in row.identity_tokens) == row.identity_tokens


def test_eval_imports_only_the_judge_and_never_the_harness() -> None:
    """C7: the eval module imports, from ``local_astra``, only ``handoff_tasks.judge``; it imports nothing from
    ``scripts`` or ``tests``, loads nothing by path, and names no harness file in code."""
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
                assert mod == "local_astra.pages.handoff_tasks", mod
                assert [a.name for a in node.names] == ["judge"], [a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not doc:
            assert "t0208_handoff_run.py" not in node.value and "scripts/gpu" not in node.value, (
                node.value[:80]
            )


#: content pins (sha256 at e3f4e424, by the test-author with ``sha256sum``): C1 imports T-0205 C's helpers and never
#: edits them; C6 copies T-0205 C's boot
FROZEN_SHA256 = {
    "scripts/gpu/t0205_demo_run.py": "426d1c243eb35395d22470d69542474efc43f3122a5966b5178ec8e1d2fe29c8",
    "scripts/locked/t0205_demo_run.sh": "17a663ff87ce2ec2542ec155c13335cd32cfd4790e7f78771bcca4cfd16350c1",
}


@pytest.mark.parametrize("rel", list(FROZEN_SHA256))
def test_t0205_files_unchanged(rel: str) -> None:
    """C1 negative (green at BASE and through group C): T-0205 C's harness and wrapper keep their bytes."""
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == FROZEN_SHA256[rel], f"T-0208 C1: {rel} changed ({got})"


# =====================================================================================================
# C1: import, constants, search never constructed
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """C1: importing the harness opens no socket (connect, bind or name lookup, loopback included); it has ``main``."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"


def test_constants() -> None:
    """C1-C6: exits, ids, arms, repeats, seed, turns, turn timeout, the run budget inside ``timeout 60m``, the table
    pin, the reasons, the regex, the labels."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )
    assert tuple(api("HANDOFF_IDS")) == IDS
    assert tuple(api("ARMS")) == ARMS
    assert (api("REPEATS"), api("ORDER_SEED"), api("TOTAL_TURNS")) == (3, 0, 18)
    assert api("TURN_TIMEOUT_S") == 300
    assert api("RUN_BUDGET_S") == 50 * 60 < 60 * 60, "C6: the budget sits inside timeout 60m"
    assert api("HANDOFF_TABLE_SHA256") == TABLE_SHA256
    assert api("SHARED_FRAME_REASON") == SHARED_FRAME_REASON
    assert api("CANNOT_SEE") == CANNOT_SEE
    assert (api("VOID"), api("INCOMPLETE"), api("MEASURED")) == ("void", "incomplete", "measured")


def test_search_never_constructed() -> None:
    """C3 (search off, H-T205-2): no code names a search backend."""
    names, _ = t205._code(_tree())
    hits = {"_search_backend", "DdgsSearch", "BraveApiSearch", "ddgs_backend", "brave_api"} & names
    assert not hits, f"C3: the harness reaches a search backend: {sorted(hits)}"


# =====================================================================================================
# C3: the order over both arms
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
def test_order_18_seed0_over_both_arms(which: str) -> None:
    """C3 (reading 2): 18 turns, each (handoff, arm) three times with repeats 0-2; seed 0 gives the hand construction;
    deterministic; another seed differs; both arms interleave (the shuffle is across arms, not per arm)."""
    order = impl(which, "order")
    want = [(h, a, r) for h in IDS for a in ARMS for r in range(3)]
    random.Random(0).shuffle(want)
    got = [tuple(t) for t in order(0)]
    assert got == want
    assert len(got) == 18 and Counter((h, a) for h, a, _ in got) == {
        (h, a): 3 for h in IDS for a in ARMS
    }
    assert got == [tuple(t) for t in order(0)] and got != [tuple(t) for t in order(1)]
    arms = [a for _, a, _ in got]
    assert arms != sorted(arms) and arms != sorted(arms, reverse=True), arms


# =====================================================================================================
# C4: P-H3 from hand-made session-log bodies
# =====================================================================================================

#: a hand-made shared image and browser frames (bytes only; the rule never decodes them)
IMG = b"\xff\xd8\xff\xe0" + b"T-0208 hand-made shared picture of a broken steam wand" + b"\xff\xd9"
B1 = b"\xff\xd8\xff\xe0" + b"T-0208 browser frame one" + b"\xff\xd9"
B2 = b"\xff\xd8\xff\xe0" + b"T-0208 browser frame two" + b"\xff\xd9"
Q = "this part of my coffee machine broke, put the replacement in my cart"


def _img(data: bytes) -> dict[str, Any]:
    """An image part as the wire carries it (a data URL); the session log redacts it (``_req``)."""
    return {
        "type": "image_url",
        "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        "phash": "p" + hashlib.sha256(data).hexdigest()[:15],
    }


def _txt(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


_SEQ = iter(range(1, 1_000_000))


def _req(rid: str, phase: str, *parts: dict[str, Any], step: int | None = None) -> dict[str, Any]:
    """One ``llm.request`` record as the session log writes it: the prompt's images redacted by production's
    ``redact`` (size and digest of the base64, the frame's phash)."""
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


def test_redacted_parts_premise() -> None:
    """Premise: production's ``redact`` replaces the data URL by a redaction (size and digest of the base64), so a
    rule over the log can tell the shared image from a browser frame; no base64 survives."""
    rec = _req("t1-gen", "generate", _img(IMG), _txt(SHARED_LABEL_SCREEN))
    blob = json.dumps(rec)
    assert base64.b64encode(IMG).decode() not in blob
    url = rec["data"]["prompt"]["messages"][1]["content"][0]["url"]
    b64 = base64.b64encode(IMG).decode()
    assert url["redacted"] == "image" and url["bytes"] == len(b64), url
    assert url["digest"] == hashlib.sha256(b64.encode()).hexdigest()[:8], url


#: T-0211 A1: step 2 after a search: no browser action has run, so the shared image is still current (T-0177's rule)
SEARCH_PAYLOAD = "google_search results: https://www.example-parts.test/breville/bes870/steam-wand"


def _ph3_case(case: str) -> list[dict[str, Any]]:
    """The case's records in **body order** with explicit ``seq`` (T-0211 A2; T-0209's review of 0ce94b69: a
    module-global counter gave bodies built late a later ``seq``, so the rule's seq sort read them after the switch
    and clause (d) decided the (b) and (c) cases). ``seq`` is 1..n in list order, except the two ``seq-*`` cases,
    whose list order and ``seq`` order differ on purpose."""
    records = _ph3_bodies(case)
    explicit = _PH3_EXPLICIT_SEQ.get(case)
    seqs = explicit if explicit is not None else list(range(1, len(records) + 1))
    assert len(seqs) == len(records), case
    for r, n in zip(records, seqs, strict=True):
        r["seq"] = n
    return records


#: the ``seq-*`` cases' seqs, in list order. In list order both read "shared step, then browser step"; by ``seq`` the
#: adversarial one reads "browser step, then shared step" (the switch body takes the earlier ``seq``).
_PH3_EXPLICIT_SEQ = {
    "seq-switch-before-shared": [1, 2, 4, 3, 5],
    "seq-shared-before-switch": [1, 2, 3, 4, 5],
}


def _ph3_bodies(case: str) -> list[dict[str, Any]]:
    prefill = _req("t1-prefill", "prefill", _img(IMG), _txt(SHARED_LABEL_SCREEN))
    gen = _req("t1-gen", "generate", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q))
    s2 = _req("t1-gen-s2", "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=2)
    check = _req("t1-gen-s2", "completion_check", _img(B1), _img(B2), _txt(Q), step=2)
    # the T-0211 shape: a search at step 1, so step 2 still shows the shared image with its label; the browser at 3
    s2_shared = _req(
        "t1-gen-s2",
        "tool_step",
        _img(IMG),
        _txt(SHARED_LABEL_SCREEN),
        _txt(Q),
        _txt(SEARCH_PAYLOAD),
        step=2,
    )
    s3 = _req("t1-gen-s3", "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=3)
    check3 = _req("t1-gen-s3", "completion_check", _img(B1), _img(B2), _txt(Q), step=3)
    if case == "clean":
        return [prefill, gen, s2, check]
    if (
        case == "clean-step1-reask"
    ):  # step 1's re-ask is a step-1 body: it may carry the shared image
        reask = _req(
            "t1-gen", "announce_reask", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q), step=1
        )
        return [prefill, gen, reask, s2]
    if (
        case == "search-step-keeps-shared-labelled"
    ):  # T-0211 A1: the voided T-0208 C turn shape now holds
        return [prefill, gen, s2_shared, s3, check3]
    if case == "shared-unlabelled-non-prefill":  # (b): the twin above with the step-2 label dropped
        bad = _req("t1-gen-s2", "tool_step", _img(IMG), _txt(Q), _txt(SEARCH_PAYLOAD), step=2)
        return [prefill, gen, bad, s3, check3]
    if (
        case == "shared-beside-browser-image"
    ):  # (c): the twin above with a browser frame added to step 2
        bad = _req(
            "t1-gen-s2",
            "tool_step",
            _img(IMG),
            _txt(SHARED_LABEL_SCREEN),
            _img(B1),
            _txt(Q),
            _txt(SEARCH_PAYLOAD),
            step=2,
        )
        return [prefill, gen, bad, s3, check3]
    if (
        case == "shared-after-browser-only-body"
    ):  # (d): the twin's two later steps in the other order
        browser = _req("t1-gen-s2", "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=2)
        late = _req(
            "t1-gen-s3",
            "tool_step",
            _img(IMG),
            _txt(SHARED_LABEL_SCREEN),
            _txt(Q),
            _txt(SEARCH_PAYLOAD),
            step=3,
        )
        return [prefill, gen, browser, late, check3]
    if (
        case == "no-step1-generate"
    ):  # (a): the twin without its step-1 generate (a prefill is not one)
        return [prefill, s2_shared, s3, check3]
    if case in ("seq-switch-before-shared", "seq-shared-before-switch"):
        # the same five bodies in the same list order (shared step, browser step); only ``seq`` (and the step number
        # each request id carries, kept consistent with ``seq``) differs between the two
        adversarial = case == "seq-switch-before-shared"
        shared_rid, browser_rid = (
            ("t1-gen-s3", "t1-gen-s2") if adversarial else ("t1-gen-s2", "t1-gen-s3")
        )
        shared = _req(
            shared_rid, "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q), step=2
        )
        browser = _req(browser_rid, "tool_step", _img(B1), _txt(L_NOW), _txt(Q), step=3)
        last = _req("t1-gen-s4", "completion_check", _img(B1), _img(B2), _txt(Q), step=4)
        return [prefill, gen, shared, browser, last]
    if case == "later-step-carries-shared":  # (b): a step-2 shared image without its label
        return [prefill, gen, _req("t1-gen-s2", "tool_step", _img(IMG), _txt(Q), step=2)]
    if (
        case == "later-step-shared-labelled"
    ):  # its twin: the label kept (no browser body yet: holds under A1)
        return [
            prefill,
            gen,
            _req("t1-gen-s2", "tool_step", _img(IMG), _txt(SHARED_LABEL_SCREEN), _txt(Q), step=2),
        ]
    if case == "later-text-hides-shared-base64":
        # the base64 in a text part of a browser body: production's ``redact`` turns it into the shared image's own
        # redaction (size and digest of its base64), so the body holds the shared image beside a browser frame: (c)
        hidden = _txt("earlier: " + base64.b64encode(IMG).decode("ascii"))
        return [prefill, gen, _req("t1-gen-s2", "tool_step", _img(B1), hidden, step=2)]
    if (
        case == "base64-in-text-after-switch"
    ):  # after the switch: (c) and (d) (redacted as in the case above)
        hidden = _txt("earlier: " + base64.b64encode(IMG).decode("ascii"))
        return [prefill, gen, s2, _req("t1-gen-s3", "tool_step", _img(B2), hidden, step=3)]
    if case == "check-carries-shared":
        return [
            prefill,
            gen,
            s2,
            _req("t1-gen-s2", "completion_check", _img(B1), _img(IMG), step=2),
        ]
    if case == "step1-no-label":
        return [prefill, _req("t1-gen", "generate", _img(IMG), _txt(Q)), s2]
    if case == "step1-browser-not-shared":
        return [
            _req("t1-prefill", "prefill", _img(B1)),
            _req("t1-gen", "generate", _img(B1), _txt(L_NOW), _txt(Q)),
            s2,
        ]
    if case == "step1-both":
        return [
            prefill,
            _req("t1-gen", "generate", _img(IMG), _txt(SHARED_LABEL_SCREEN), _img(B1), _txt(Q)),
            s2,
        ]
    if case == "prefill-both":
        return [
            _req("t1-prefill", "prefill", _img(B1), _img(IMG), _txt(SHARED_LABEL_SCREEN)),
            gen,
            s2,
        ]
    if case == "realistic-prefill-unlabelled":
        # the shape of every real logged prefill (T-0211 A review): the shared image with NO label; (b) exempts a
        # prefill, so the turn holds (a rule applying (b) to the prefill too refuses every real turn)
        return [_req("t1-prefill", "prefill", _img(IMG)), gen, s2]
    if case == "step1-generate-browser-only":
        # (a) per step-1 generate: the prefill holds the labelled shared image, the generate a browser frame only
        return [prefill, _req("t1-gen", "generate", _img(B1), _txt(Q)), s2]
    if case == "step1-generate-label-no-image":
        # (a) per step-1 generate: the generate keeps the label but holds no image at all
        return [prefill, _req("t1-gen", "generate", _txt(SHARED_LABEL_SCREEN), _txt(Q)), s2]
    raise KeyError(case)


#: case -> whether P-H3 holds (by hand, from T-0211 A1's four clauses (a)-(d), read in ``seq`` order)
PH3_CASES = {
    "clean": True,
    "clean-step1-reask": True,
    "search-step-keeps-shared-labelled": True,
    "seq-shared-before-switch": True,
    "later-step-shared-labelled": True,
    "shared-unlabelled-non-prefill": False,  # (b)
    "later-step-carries-shared": False,  # (b)
    "step1-no-label": False,  # (a), its label half (a non-prefill shared body without the label: (b) too)
    "shared-beside-browser-image": False,  # (c)
    "step1-both": False,  # (c)
    "prefill-both": False,  # (c)
    "check-carries-shared": False,  # (c) and (d)
    "shared-after-browser-only-body": False,  # (d)
    "seq-switch-before-shared": False,  # (d), only by seq
    "later-text-hides-shared-base64": False,  # (c): the redacted base64 is the shared image beside B1
    "base64-in-text-after-switch": False,  # (c) and (d)
    "no-step1-generate": False,  # (a)
    "step1-browser-not-shared": False,  # (a)
    "realistic-prefill-unlabelled": True,  # (b) exempts the prefill: the real logged prefill shape
    "step1-generate-browser-only": False,  # (a), on the generate (the prefill is clean)
    "step1-generate-label-no-image": False,  # (a), on the generate (the prefill is clean)
}

#: a failing case -> (its clause, its passing twin); the twin differs from the case only in that clause's detail
#: (each pair by hand from ``_ph3_bodies``)
PH3_TWINS = {
    "shared-unlabelled-non-prefill": ("b", "search-step-keeps-shared-labelled"),
    "later-step-carries-shared": ("b", "later-step-shared-labelled"),
    "step1-no-label": ("b", "clean"),
    "shared-beside-browser-image": ("c", "search-step-keeps-shared-labelled"),
    "step1-both": ("c", "clean"),
    "prefill-both": ("c", "clean"),
    "shared-after-browser-only-body": ("d", "search-step-keeps-shared-labelled"),
    "seq-switch-before-shared": ("d", "seq-shared-before-switch"),
    "no-step1-generate": ("a", "search-step-keeps-shared-labelled"),
    # the twin keeps the same prefill and step 2; only the step-1 generate's shared image (with its label) differs
    "step1-generate-browser-only": ("a", "clean"),
    "step1-generate-label-no-image": ("a", "clean"),
}

#: the cases the BASE harness rule (C4's three clauses, no switch) decides otherwise than A1 (measured at BASE 175a29b0:
#: each was red there on the assertion, never on an import); their ``harness`` rows were pending until T-0211 A's
#: ``/implement`` (activated 2026-10-30); kept as the record of which rows the A1 rule change turns
_PH3_RED_AT_BASE = {
    "search-step-keeps-shared-labelled",
    "seq-shared-before-switch",
    "later-step-shared-labelled",
}


def _ph3_params(cases: Any) -> list[Any]:
    return [
        pytest.param(which, case, id=f"{which}-{case}")
        for which in ("harness", "eval-oracle")
        for case in cases
    ]


def _image_redactions(obj: Any) -> list[Any]:
    """Every ``{"redacted": "image", ...}`` in ``obj`` (production's ``redact`` shape)."""
    if isinstance(obj, dict):
        if obj.get("redacted") == "image":
            return [obj]
        return [x for v in obj.values() for x in _image_redactions(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _image_redactions(v)]
    return []


@pytest.mark.parametrize(("which", "case"), _ph3_params(PH3_CASES))
def test_ph3_rule_from_hand_bodies(which: str, case: str) -> None:
    """T-0211 A1 (re-pinned from T-0208 C4): P-H3 is T-0209 B4's P-R3 switch rule, read in ``seq`` order: (a) a
    step-1 ``generate`` exists and each holds the shared image and ``SHARED_LABEL_SCREEN``; (b) every non-prefill body
    holding the shared image holds the label; (c) never beside another image; (d) after the first browser-only body,
    no body holds the shared image or its base64. The too-wide side: a search step that keeps the labelled shared
    image (the voided T-0208 C turn shape) holds; the too-narrow side: each clause has a failing case."""
    got = impl(which, "ph3_ok")(_ph3_case(case), IMG)
    assert got is PH3_CASES[case], (case, got)


@pytest.mark.parametrize("case", list(PH3_CASES))
def test_ph3_case_bodies_carry_their_seq(case: str) -> None:
    """Premise for the P-H3 rows: each case's ``seq`` is 1..n in list order, except the two ``seq-*`` cases, which
    hold the same seqs in a different list order (the adversarial one's list order is not its ``seq`` order)."""
    seqs = [r["seq"] for r in _ph3_case(case)]
    assert sorted(seqs) == list(range(1, len(seqs) + 1)), (case, seqs)
    if case == "seq-switch-before-shared":
        assert seqs != sorted(seqs), seqs
    else:
        assert seqs == sorted(seqs), (case, seqs)


@pytest.mark.parametrize(("which", "case"), _ph3_params(PH3_TWINS))
def test_ph3_case_fails_only_its_own_clause(which: str, case: str) -> None:
    """T-0211 A2 (adversarial; T-0209's review of 0ce94b69): each failing case has a passing twin that differs only in
    its clause's detail. For a (b) or (c) case no browser-only body precedes the offending body, so (d) cannot decide
    it; hence a rule without (b) passes ``shared-unlabelled-non-prefill``, one without (c) passes
    ``shared-beside-browser-image``, one without the seq sort passes ``seq-switch-before-shared``."""
    clause, twin = PH3_TWINS[case]
    ph3 = impl(which, "ph3_ok")
    assert ph3(_ph3_case(twin), IMG) is True, (twin, "the twin must hold")
    assert ph3(_ph3_case(case), IMG) is False, case
    if clause not in ("b", "c"):
        return
    b64 = base64.b64encode(IMG).decode("ascii")
    digest = hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8]
    records = sorted(_ph3_case(case), key=lambda r: r["seq"])
    reds = [_image_redactions(r["data"]["prompt"]) for r in records]
    carrying = [i for i, rs in enumerate(reds) if any(x["digest"] == digest for x in rs)]
    assert carrying, case
    offending = [
        i
        for i in carrying
        if (clause == "c" and len(reds[i]) > 1)
        or (
            clause == "b"
            and records[i]["data"]["phase"] != "prefill"
            and SHARED_LABEL_SCREEN not in json.dumps(records[i]["data"]["prompt"])
        )
    ]
    assert offending, (case, "no body breaks the clause")
    first = offending[0]
    browser_only = [
        records[i]["seq"]
        for i in range(first)
        if reds[i] and all(x["digest"] != digest for x in reds[i])
    ]
    assert not browser_only, (case, "a browser-only body precedes the offending body", browser_only)


def _t209() -> ModuleType:
    """T-0209's unit module (its ``_pr3_case`` table, ``IMG`` and harness loader; T-0209's modules never import
    this one, so there is no cycle)."""
    from tests.unit import test_t0209_rtx_harness as t209

    return t209


def _pr3_impls() -> dict[str, Any]:
    from tests.eval import test_t0209_rtx_run as pr3_oracle

    return {"t0209-harness": _t209().api("pr3_ok"), "t0209-eval-oracle": pr3_oracle.pr3_ok}


#: T-0209's P-R3 table cases the BASE P-H3 harness rule decides otherwise (measured at BASE 175a29b0; their
#: ``harness`` rows were pending until T-0211 A's ``/implement``; kept as the record)
_PR3_RED_AT_BASE = {
    "search-then-browser",
    "step2-labelled",
    "step2-shared-only",
    "check-browser-only",
}


def _a2_params() -> list[Any]:
    from tests.unit import test_t0209_rtx_harness as t209

    out = []
    for which in ("harness", "eval-oracle"):
        for table, cases in (("ph3", list(PH3_CASES)), ("pr3", list(t209.PR3_CASES))):
            for case in cases:
                out.append(pytest.param(which, table, case, id=f"{which}-{table}-{case}"))
    return out


@pytest.mark.parametrize(("which", "table", "case"), _a2_params())
def test_ph3_equals_pr3_on_both_tables(which: str, table: str, case: str) -> None:
    """T-0211 A2: on every case of both hand-made body tables (this module's ``_ph3_case``, T-0209's ``_pr3_case``),
    ``ph3_ok(case, IMG) == pr3_ok(case, IMG)`` for T-0209's harness rule and its eval oracle. The adversarial
    ``seq-switch-before-shared`` case (the switch body takes an earlier ``seq`` than a later shared body; list order
    says otherwise) is in this table, so a P-H3 without the seq sort differs from P-R3 there. Both sides are also
    pinned to the hand verdict, so two equally wrong rules cannot agree silently."""
    t209 = _t209()
    if table == "ph3":
        records, image, want = _ph3_case(case), IMG, PH3_CASES[case]
    else:
        records, image, want = t209._pr3_case(case), t209.IMG, t209.PR3_CASES[case]
    pr3s = {name: fn(copy.deepcopy(records), image) for name, fn in _pr3_impls().items()}
    assert all(v is want for v in pr3s.values()), (case, "P-R3 premise", pr3s, want)
    got = impl(which, "ph3_ok")(copy.deepcopy(records), image)
    assert got is want, (table, case, got, pr3s)


def test_harness_does_not_import_t0209() -> None:
    """T-0211 A2: the rule is reimplemented in place; ``t0208_handoff_run.py`` never names T-0209's harness (it would
    be an import cycle: T-0209 may import T-0208)."""
    src = HARNESS.read_text(encoding="utf-8")
    assert "t0209" not in src.casefold(), "T-0211 A2: the T-0208 harness names t0209"


# =====================================================================================================
# C5: identity columns, the step-1 classifier, shared refusals, "cannot see"
# =====================================================================================================

FAILED_H1 = {"page": "home", "cart": [], **J.SAFE}


def _trial(
    hid: str, arm: str, state: Any, voiced: list[str], repeat: int = 0, **over: Any
) -> dict[str, Any]:
    return {
        "task": hid,
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
        "ph3": True if arm == "shared" else None,
        **over,
    }


def _st(
    k: int,
    name: str | None,
    status: str | None,
    reason: str | None = None,
    hid: str = "H1",
    arm: str = "shared",
    repeat: int = 0,
    **arguments: Any,
) -> dict[str, Any]:
    return {
        "task": hid,
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


SCROLL = ("scroll", "executed")
#: case -> (trial, steps, (spoken, typed, lost)), each expectation by hand from C5's three definitions
IDENTITY_CASES: dict[str, tuple[dict[str, Any], list[dict[str, Any]], tuple[bool, bool, bool]]] = {
    # H1_SAID names the Barista Express; the golden cart succeeds
    "spoken": (
        _trial("H1", "shared", J.H1_GOLD, [J.H1_SAID]),
        [_st(1, *SCROLL)],
        (True, False, False),
    ),
    # nothing said names it; the typed query does (casefolded: "BES870" matches "bes870")
    "typed-only": (
        _trial("H1", "shared", FAILED_H1, ["Done."]),
        [_st(1, *SCROLL), _st(2, "type_at", "executed", text="BES870 steam wand", x=1, y=1)],
        (False, True, False),
    ),
    "lost": (
        _trial("H1", "shared", FAILED_H1, ["I added the part to your cart."]),
        [_st(1, *SCROLL), _st(2, "click_at", "executed", x=5, y=5)],
        (False, False, True),
    ),
    "typed-arm-is-never-lost": (
        _trial("H1", "typed", FAILED_H1, ["I added the part to your cart."]),
        [_st(1, *SCROLL, arm="typed")],
        (False, False, False),
    ),
    "a-search-is-not-a-browser-action": (
        _trial("H1", "shared", FAILED_H1, ["Done."]),
        [_st(1, "google_search", "executed", query="coffee machine part")],
        (False, False, False),
    ),
    "a-refused-click-is-not-acting": (
        _trial("H1", "shared", FAILED_H1, ["Done."]),
        [_st(1, "click_at", "refused", SHARED_FRAME_REASON, x=5, y=5)],
        (False, False, False),
    ),
    # C-R1: an executed ``wait`` is acting (it swaps the shared picture out for a browser frame); step 2 answers with
    # no call, the cart is empty, the item never named: lost
    "c-r1-wait-then-no-call-is-lost": (
        _trial("H1", "shared", FAILED_H1, ["I could not find it."]),
        [_st(1, "wait", "executed"), _st(2, None, None)],
        (False, False, True),
    ),
    # its too-wide side: a refused wait is not acting
    "c-r1-refused-wait-is-not-acting": (
        _trial("H1", "shared", FAILED_H1, ["I could not find it."]),
        [_st(1, "wait", "refused", "max 3 waits in a row"), _st(2, None, None)],
        (False, False, False),
    ),
    "a-success-is-not-lost": (
        _trial("H1", "shared", J.H1_GOLD, ["Done, it is in your cart."]),
        [_st(1, *SCROLL)],
        (False, False, False),
    ),
    # H3's token needs "large" and "pepperoni" in the same text
    "h3-pepperoni-alone-is-lost": (
        _trial(
            "H3", "shared", {**J.H3_GOLD, "method": "delivery"}, ["Sam wants a pepperoni pizza."]
        ),
        [_st(1, *SCROLL, hid="H3")],
        (False, False, True),
    ),
    "h3-both-words-any-order": (
        _trial("H3", "shared", {**J.H3_GOLD, "method": "delivery"}, ["one pepperoni, large"]),
        [_st(1, *SCROLL, hid="H3")],
        (True, False, False),
    ),
    "h2-casefolded": (
        _trial("H2", "shared", {**J.H2_GOLD, "cart": []}, ["The TRAILHEAD stove is sold out."]),
        [_st(1, *SCROLL, hid="H2")],
        (True, False, False),
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(IDENTITY_CASES))
def test_identity_columns_from_hand_steps(which: str, case: str) -> None:
    """C5: ``identity_spoken`` (a token in any voiced text), ``identity_typed`` (a token in any ``type_at`` text),
    ``identity_lost`` (shared arm, an executed browser action, a failure, neither of the others)."""
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
    "executed": ([_st(1, *SCROLL), _st(2, "click_at", "executed")], ("scroll", "executed")),
    "refused-shared": (
        [_st(1, "click_at", "refused", SHARED_FRAME_REASON), _st(2, *SCROLL)],
        ("click_at", "refused_shared"),
    ),
    # adversarial: a checkout refusal at step 1 is another refusal, not a shared one
    "refused-checkout": (
        [_st(1, "click_at", "refused", CHECKOUT_REASON)],
        ("click_at", "refused_other"),
    ),
    "refused-moved": ([_st(1, "type_at", "refused", MOVED_REASON)], ("type_at", "refused_other")),
    "no-call": ([_st(1, None, None)], (None, "no_call")),
    "no-step-line": ([], (None, "no_call")),
    # step 1 is the line whose ``step`` is 1, not the first line
    "by-step-number": (
        [_st(2, *SCROLL), _st(1, "click_at", "refused", SHARED_FRAME_REASON)],
        ("click_at", "refused_shared"),
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(STEP1_CASES))
def test_step1_action_classifier(which: str, case: str) -> None:
    """C5: the step-1 action is its tool and outcome: executed, refused with ``SHARED_FRAME_REASON``, another
    refusal, no call."""
    steps, want = STEP1_CASES[case]
    got = _action(impl(which, "step1_action")(steps))
    assert got == want, (case, got)


@pytest.mark.parametrize("which", IMPLS)
def test_checkout_reason_is_not_a_shared_refusal(which: str) -> None:
    """Adversarial (exact reason equality): of a shared refusal, a ``CHECKOUT_REASON`` refusal, a moved refusal, a
    reason that only starts with the shared one and an executed click, exactly one is a shared-image refusal."""
    steps = [
        _st(1, "click_at", "refused", SHARED_FRAME_REASON),
        _st(2, "click_at", "refused", CHECKOUT_REASON),
        _st(3, "click_at", "refused", MOVED_REASON),
        _st(4, "click_at", "refused", SHARED_FRAME_REASON + "; and more"),
        _st(5, "click_at", "executed", SHARED_FRAME_REASON),
    ]
    assert impl(which, "shared_refusals")(steps) == 1
    assert impl(which, "shared_refusals")([_st(1, "click_at", "refused", CHECKOUT_REASON)]) == 0


CANNOT_SEE_CASES = {
    "I can't see your browser.": True,
    "I cannot see the picture.": True,
    "I'm unable to see the cart.": True,
    "I CANNOT SEE it": True,
    "I can see the steam wand.": False,
    "I can now see the cart.": False,
    "": False,
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("text", list(CANNOT_SEE_CASES))
def test_cannot_see_ending(which: str, text: str) -> None:
    """C5: a turn whose voiced text (one utterance here, so "ending" and "any" agree) matches the pinned regex."""
    trial = _trial("H1", "shared", FAILED_H1, [text] if text else [])
    assert bool(impl(which, "cannot_see")(trial)) is CANNOT_SEE_CASES[text], text


# =====================================================================================================
# C4/C5: P-H1, P-H2, the label, and the per-arm counts by hand
# =====================================================================================================


def _path(state: Any, voiced: str, actions: int = 3, **over: Any) -> dict[str, Any]:
    return {
        "actions": actions,
        "outcomes": ["executed"] * actions,
        "state": copy.deepcopy(state),
        "voiced": voiced,
        "shown": "",
        "error": None,
        **over,
    }


TRAP_STATES = {
    "H1": {**J.H1_GOLD, "cart": ["steam wand assembly bes878"]},
    "H2": {**J.H2_GOLD, "cart": ["ridge twin"]},
    "H3": {**J.H3_GOLD, "method": "delivery"},
}


def _ph1(**over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "viewport": [1280, 800],
        "paths": {
            hid: {
                "golden": _path(*J.GOLD[hid]),
                "traps": {"lookalike": _path(TRAP_STATES[hid], J.GOLD[hid][1], actions=4)},
            }
            for hid in IDS
        },
    }
    rec.update(over)
    return rec


PH2_OK = {
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
        "ph1": _ph1(),
        "ph2": dict(PH2_OK),
    }
    base.update(over)
    return base


def test_hand_probes_premise() -> None:
    """Premise: every golden state succeeds and every trap state fails under the real judge."""
    from local_astra.pages.handoff_tasks import judge

    for hid in IDS:
        assert judge(hid, J.GOLD[hid][0], J.GOLD[hid][1]).success is True, hid
        assert judge(hid, TRAP_STATES[hid], J.GOLD[hid][1]).success is False, hid


def _mut_ph1(case: str) -> dict[str, Any]:
    rec = _ph1()
    if case == "golden-fails":
        rec["paths"]["H2"]["golden"] = _path({**J.H2_GOLD, "cart": []}, J.H2_SAID)
    elif case == "trap-passes":
        rec["paths"]["H1"]["traps"]["lookalike"] = _path(J.H1_GOLD, J.H1_SAID)
    elif case == "golden-not-executed":
        rec["paths"]["H3"]["golden"]["outcomes"][1] = "refused"
    elif case == "a-handoff-missing":
        del rec["paths"]["H3"]
    elif case == "no-traps":
        rec["paths"]["H1"]["traps"] = {}
    elif case == "viewport":
        rec["viewport"] = [1440, 900]
    else:
        raise KeyError(case)
    return rec


LABEL_CASES: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str | None, str]] = {
    "measured": (_probes(), [], None, "measured"),
    "typed-ph3-none-is-not-void": (
        _probes(),
        [_trial("H1", "typed", J.H1_GOLD, [J.H1_SAID])],
        None,
        "measured",
    ),
    "a-shared-turn-fails-ph3": (
        _probes(),
        [
            _trial("H1", "shared", J.H1_GOLD, [J.H1_SAID]),
            _trial("H2", "shared", J.H2_GOLD, [J.H2_SAID], ph3=False),
        ],
        None,
        "void",
    ),
    # C-R2: a shared turn that sent no step-1 request records ph3 None; it voids nothing (ERRORED_STOP's business)
    "c-r2-shared-ph3-none-is-not-void": (
        _probes(),
        [
            _trial("H1", "shared", J.H1_GOLD, [J.H1_SAID]),
            _trial(
                "H2",
                "shared",
                {**J.H2_GOLD, "cart": []},
                [],
                ph3=None,
                exception="RuntimeError('open')",
            ),
        ],
        None,
        "measured",
    ),
    "ph1-golden-fails": (_probes(ph1=_mut_ph1("golden-fails")), [], None, "void"),
    "ph1-trap-passes": (_probes(ph1={**_mut_ph1("trap-passes"), "passed": True}), [], None, "void"),
    "ph1-golden-not-executed": (_probes(ph1=_mut_ph1("golden-not-executed")), [], None, "void"),
    "ph1-a-handoff-missing": (_probes(ph1=_mut_ph1("a-handoff-missing")), [], None, "void"),
    "ph1-no-traps": (_probes(ph1=_mut_ph1("no-traps")), [], None, "void"),
    "ph1-viewport": (_probes(ph1=_mut_ph1("viewport")), [], None, "void"),
    "ph2-resolver": (_probes(ph2={**PH2_OK, "resolver_calls": 1}), [], None, "void"),
    "ph2-connector": (
        _probes(ph2={**PH2_OK, "connector_calls": 2, "passed": True}),
        [],
        None,
        "void",
    ),
    "ph2-non-loopback": (
        _probes(ph2={**PH2_OK, "non_loopback": ["93.184.216.34:443"]}),
        [],
        None,
        "void",
    ),
    "p1-failed": (_probes(p1={"ok": False, "passed": False}), [], None, "void"),
    "stopped": (_probes(), [], "the run budget is spent", "incomplete"),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(LABEL_CASES))
def test_labels_and_probe_rules(which: str, case: str) -> None:
    """C4/C5: any probe failing (P1-P3, P-H1 re-derived, P-H2 re-derived, a recorded ``passed`` never trusted) or any
    shared turn failing P-H3 is ``void``; else a stopped run is ``incomplete``; else ``measured``. A typed turn's
    ``ph3: None`` voids nothing."""
    probes, trials, stopped, want = LABEL_CASES[case]
    report = impl(which, "build_report")(trials, [], probes, stopped)
    assert report["label"] == want, (case, report.get("label"), report.get("probes"))


def _hand_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Five turns; every count below is by hand."""
    trials = [
        # H1 shared 0: success; scroll, click, type "Barista Express"; one completion check; 30 s
        _trial("H1", "shared", J.H1_GOLD, [J.H1_SAID], 0, wall_s=30.0, completion_checks=1),
        # H1 shared 1: failure; step 1 refused on the picture, a scroll, a checkout refusal; nothing named; 50 s
        _trial("H1", "shared", FAILED_H1, ["I added the part."], 1, wall_s=50.0),
        # H1 shared 2: failure; no call at step 1; "can't see"; 10 s
        _trial("H1", "shared", FAILED_H1, ["I can't see your browser."], 2, wall_s=10.0),
        # H1 typed 0: success; typed and said "BES870"; handed back with the at_checkout clip
        _trial(
            "H1",
            "typed",
            J.H1_GOLD,
            ["The BES870 steam wand is in your cart."],
            0,
            wall_s=20.0,
            tool_loop_end="handed_back",
            clips=["still_working", "at_checkout"],
        ),
        # H2 shared 0: the Ridge Twin in the cart (a trap); only a search ran
        _trial("H2", "shared", {**J.H2_GOLD, "cart": ["ridge twin"]}, ["Done."], 0, wall_s=40.0),
    ]
    steps = [
        _st(1, *SCROLL, repeat=0),
        _st(2, "click_at", "executed", repeat=0, x=1, y=1),
        _st(3, "type_at", "executed", repeat=0, text="Barista Express", x=1, y=1),
        _st(1, "click_at", "refused", SHARED_FRAME_REASON, repeat=1, x=5, y=5),
        _st(2, *SCROLL, repeat=1),
        _st(3, "click_at", "refused", CHECKOUT_REASON, repeat=1, x=9, y=9),
        _st(1, None, None, repeat=2),
        _st(1, "type_at", "executed", arm="typed", text="BES870 steam wand", x=1, y=1),
        _st(2, "wait", "executed", arm="typed"),
        _st(1, "google_search", "executed", hid="H2", query="trailhead duo stove"),
    ]
    return trials, steps


#: by hand: arm -> handoff -> the counts checked (not every key)
HAND_COUNTS: dict[str, dict[str, dict[str, Any]]] = {
    "shared": {
        "H1": {
            "n": 3,
            "successes": 1,
            "steps_p50": 3,  # step counts 3, 3, 1: the second smallest
            "steps_max": 3,
            "wall_p50": 30.0,  # 10, 30, 50
            "wall_max": 50.0,
            "step1_actions": {
                "scroll:executed": 1,
                "click_at:refused_shared": 1,
                "none:no_call": 1,
            },
            "shared_refusals": 1,
            "checkout_refusals": 1,
            "cannot_see": 1,
            "identity_spoken": 1,
            "identity_typed": 1,
            "identity_lost": 1,
            "completion_checks": 1,
            "hand_backs": 0,
            "waits": 0,
            "trap": 0,
        },
        "H2": {
            "n": 1,
            "successes": 0,
            "step1_actions": {"google_search:executed": 1},
            "identity_lost": 0,
            "trap": 1,
            "shared_refusals": 0,
        },
        "H3": {"n": 0, "successes": 0, "wilson": None},
    },
    "typed": {
        "H1": {
            "n": 1,
            "successes": 1,
            "steps_p50": 2,
            "identity_spoken": 1,
            "identity_typed": 1,
            "identity_lost": 0,
            "hand_backs": 1,
            "hand_back_clips": {"at_checkout": 1},
            "waits": 1,
            "step1_actions": {"type_at:executed": 1},
        },
        "H2": {"n": 0, "successes": 0},
        "H3": {"n": 0, "successes": 0},
    },
}
#: Wilson 95 % of 1/3 and 1/1 (z = 1.959964), by hand
WILSON_1_OF_3 = (0.0615, 0.7923)
WILSON_1_OF_1 = (0.2065, 1.0)


@pytest.mark.parametrize("which", IMPLS)
def test_per_arm_counts_by_hand(which: str) -> None:
    """C5: per arm and handoff, the counts of a hand-made run (expectations by hand, in ``HAND_COUNTS``)."""
    trials, steps = _hand_run()
    report = impl(which, "build_report")(trials, steps, _probes(), None)
    per = report["per_arm"]
    for arm, rows in HAND_COUNTS.items():
        for hid, want in rows.items():
            got = per[arm][hid]
            for k, v in want.items():
                g = got.get(k, "<absent>")
                if isinstance(v, dict):
                    g = {kk: vv for kk, vv in dict(g or {}).items() if vv}
                assert g == v, (arm, hid, k, g, v)
    assert per["shared"]["H1"]["wilson"] == pytest.approx(list(WILSON_1_OF_3), abs=1e-3)
    assert per["typed"]["H1"]["wilson"] == pytest.approx(list(WILSON_1_OF_1), abs=1e-3)
    assert report["label"] == "measured", report["label"]


# =====================================================================================================
# C2: refusals before any socket; the rig at HEAD goes to the engine
# =====================================================================================================


class Rig:
    """Every real socket refused and recorded; a clean tree and ``merges`` through the harness's ``_git`` (and
    T-0111g's); the kit ``cli`` resolves with repair on; Chromium reported present."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli

        self.h = h = harness()
        self.cli = cli
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*OTHERS, MERGE_B, MERGE_A2)
        self.out = tmp_path / "out"
        self.kits: list[dict[str, Any]] = []
        self.git_calls: list[tuple[str, ...]] = []

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0208 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            self.git_calls.append(tuple(args))
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
            return (t205._kit(tools_version=t205._set_sha(page_overview, wait_tool)), object())

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


def _image_copy(tmp_path: Path, alter: str | None = None) -> Path:
    """The three committed images copied under ``tmp_path``; ``alter`` gets one byte appended."""
    d = tmp_path / "handoff-images"
    d.mkdir()
    for name in oracle.IMAGES.values():
        shutil.copy(IMAGE_DIR / name, d / name)
    if alter is not None:
        with (d / alter).open("ab") as fh:
            fh.write(b"\x00")
    return d


REFUSALS = (
    "url-not-loopback",
    "dirty-tree",
    "out-dir-exists",
    "no-merge-b",
    "table-off-its-sha",
    "image-off-its-sha",
    "web-off",
    "guidance-v2",
    "loop-19",
)
REFUSAL_NAMES = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"session\.py",
    "no-merge-b": r"T-0208-b",
    "table-off-its-sha": r"table|sha",
    "image-off-its-sha": r"h2_stove_box",
    "web-off": r"web",
    "guidance-v2": r"v3|guidance",
    "loop-19": r"20|loop|max_tool_steps",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C2 (reading 1): each refusal exits 2 with a stderr line naming it, before any socket, leaving the out-dir as it
    was. The live default is read through ``cli`` at run time, so each is made by moving a ``cli`` resolver. "No
    Merge T-0208-b": the log holds A1's and A2's merges, T-0205-c's, the branch merges into T-0208-b-impl and a
    ``Merge T-0208-bb:``, but not B's."""
    from local_astra.pages import handoff_tasks

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
    elif case == "no-merge-b":
        rig.merges = (*OTHERS, MERGE_A2, MERGE_C)
    elif case == "table-off-its-sha":
        moved = (
            dataclasses.replace(
                handoff_tasks.HANDOFFS[0], request="put any coffee part in my cart"
            ),
            *handoff_tasks.HANDOFFS[1:],
        )
        monkeypatch.setattr(handoff_tasks, "HANDOFFS", moved)
        if hasattr(rig.h, "HANDOFFS"):
            monkeypatch.setattr(rig.h, "HANDOFFS", moved)
        assert handoff_tasks.handoff_table_sha() != TABLE_SHA256, "premise: the table moved"
    elif case == "image-off-its-sha":
        assert hasattr(rig.h, "HANDOFF_DIR"), (
            "T-0208 C1: the harness has no module-level HANDOFF_DIR"
        )
        monkeypatch.setattr(rig.h, "HANDOFF_DIR", _image_copy(tmp_path, alter="h2_stove_box.jpg"))
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


@pytest.mark.parametrize("case", ["head", "no-a2-merge", "image-copy-intact"])
def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """The other side of every refusal (a guard too wide): at HEAD (B merged, web on, v3, loop 20, the table and
    images on their pins, a clean tree) ``main`` does not refuse; it asks for the engine (a socket attempt) and,
    finding none, exits 1. A2 is not required (``no-a2-merge``); an intact copy of the images under ``HANDOFF_DIR``
    passes (so the mismatch row's refusal is the altered byte, not the directory)."""
    rig = Rig(tmp_path, monkeypatch)
    if case == "no-a2-merge":
        rig.merges = (*OTHERS, MERGE_B)
    elif case == "image-copy-intact":
        assert hasattr(rig.h, "HANDOFF_DIR"), (
            "T-0208 C1: the harness has no module-level HANDOFF_DIR"
        )
        monkeypatch.setattr(rig.h, "HANDOFF_DIR", _image_copy(tmp_path))
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


# =====================================================================================================
# C1-C5 through main: order.json, the manifest, a probe failing is void
# =====================================================================================================


class MeasureRig(Rig):
    """``Rig`` past the engine check: ``/models`` lists the served model, ``/metrics`` answers; VRAM sampling and the
    regime sources stubbed; P1-P3, P-H1 and P-H2 stubbed (passing unless a row says otherwise); ``_one_turn``
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

        def ph1(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("probe_ph1")
            return copy.deepcopy(self.probes["ph1"])

        def ph2(*a: Any, **k: Any) -> dict[str, Any]:
            self.calls.append("probe_ph2")
            return copy.deepcopy(self.probes["ph2"])

        async def one_turn(*a: Any, **k: Any) -> Any:
            self.calls.append("_one_turn")
            raise AssertionError("T-0208 rig: a turn ran")

        for seam, fn in (
            ("run_probes", run_probes),
            ("probe_ph1", ph1),
            ("probe_ph2", ph2),
            ("_one_turn", one_turn),
        ):
            assert hasattr(h, seam), f"T-0208 C: the harness has no module-level seam {seam!r}"
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
    "case", ["ph1-golden-fails", "ph1-trap-passes", "ph2-connector", "p1-failed"]
)
def test_probe_failure_is_void_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C4: a failing P-H1, P-H2 (or P1) gives exit 3 and a ``report.json`` labelled ``void``; no turn runs."""
    rig = MeasureRig(tmp_path, monkeypatch)
    if case == "ph1-golden-fails":
        rig.probes["ph1"] = _mut_ph1("golden-fails")
    elif case == "ph1-trap-passes":
        rig.probes["ph1"] = {**_mut_ph1("trap-passes"), "passed": True}
    elif case == "ph2-connector":
        rig.probes["ph2"] = {**PH2_OK, "connector_calls": 1, "passed": True}
    elif case == "p1-failed":
        rig.probes["p1"] = {"ok": False}
    _manifest, report = rig.run(capsys, 3)
    assert report.get("label") == "void", report
    assert "_one_turn" not in rig.calls, rig.calls
    assert {"run_probes", "probe_ph1", "probe_ph2"} <= set(rig.calls), rig.calls


def test_order_json_written_before_the_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C3 (reading 2): ``order.json`` is on disk when the probes run and holds ``order(0)``: 18 ``[id, arm, repeat]``."""
    rig = MeasureRig(tmp_path, monkeypatch)
    rig.probes["ph2"] = {**PH2_OK, "resolver_calls": 1}  # stop at the probes
    rig.run(capsys, 3)
    assert rig.order_on_disk == [True], rig.order_on_disk
    written = [tuple(x) for x in json.loads((rig.out / "order.json").read_text(encoding="utf-8"))]
    want = [(h, a, r) for h in IDS for a in ARMS for r in range(3)]
    random.Random(0).shuffle(want)
    assert written == want


@pytest.mark.parametrize("case", ["one-shared-turn-fails-ph3", "every-shared-turn-holds"])
def test_post_run_void_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C4 (review finding 6): the probes pass, all 18 turns run, and one shared turn's ``ph3`` is ``False``: the
    report is ``void`` and ``main`` exits 3 (``EXIT_PROBE``), not 0 or 1. The other side: every shared turn's
    ``ph3`` ``True`` is ``measured``, exit 0."""
    rig = MeasureRig(tmp_path, monkeypatch)
    ran: list[tuple[str, str, int]] = []
    bad = next(k for k in oracle.order(0) if k[1] == "shared")  # the first shared turn in run order

    async def one_turn(run: Any, hid: str, arm: str, repeat: int) -> Any:
        ran.append((hid, arm, repeat))
        state, said = J.GOLD[hid]
        ph3: bool | None = None
        if arm == "shared":
            ph3 = not (case == "one-shared-turn-fails-ph3" and (hid, arm, repeat) == bad)
        trial = _trial(hid, arm, state, [said], repeat, ph3=ph3, exception=None, session_log=None)
        return trial, [_st(1, *SCROLL, hid=hid, arm=arm, repeat=repeat)], None

    monkeypatch.setattr(rig.h, "_one_turn", one_turn)
    want = 3 if case == "one-shared-turn-fails-ph3" else 0
    _manifest, report = rig.run(capsys, want)
    assert len(ran) == 18, ran
    assert report.get("label") == ("void" if want == 3 else "measured"), report.get("label")


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


@pytest.mark.parametrize("a2", ["a2-merged", "a2-absent"])
def test_manifest_records_heads_live_defaults_shared_check_and_shas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], a2: str
) -> None:
    """C1 (reading 5): ``manifest.json``'s ``live_default`` holds HEAD's live defaults as ``cli`` resolves them, the
    wait tool and the completion check among them; ``shared_check`` is ``True`` exactly when a ``Merge T-0208-a2``
    merge is in the history (``a2-absent``: only the branch merge ``... into T-0208-a2-tests3``); the table sha and
    every image's sha256 are recorded; search is recorded off."""
    from local_astra.pages import handoff_tasks

    want = _heads_live_default()
    rig = MeasureRig(tmp_path, monkeypatch)
    if a2 == "a2-absent":
        rig.merges = (*OTHERS, MERGE_B)
    rig.probes["ph2"] = {
        **PH2_OK,
        "resolver_calls": 1,
    }  # stop at the probes: the manifest is written before them
    manifest, _ = rig.run(capsys, 3)
    got = manifest.get("live_default") or {}
    problems = {
        k: (got.get(k, "<absent>"), v) for k, v in want.items() if got.get(k, "<absent>") != v
    }
    assert not problems, f"C1: live_default differs from cli's: {problems}"
    assert manifest.get("shared_check") is (a2 == "a2-merged"), manifest.get("shared_check")
    blob = json.dumps(manifest)
    assert TABLE_SHA256 in blob, "C1: the table sha is not recorded"
    for row in handoff_tasks.HANDOFFS:
        assert row.image_sha256 in blob, f"C1: {row.image}'s sha256 is not recorded"
    searches = _values_of(manifest, "search")
    assert searches and all(v in (False, "off") for v in searches), f"search recorded as {searches}"


def test_manifest_follows_cli_at_run_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C1 (never hard-coded): with ``cli.wait_tool_of`` and ``cli.completion_check_of`` giving off, the manifest
    records both off (and the kit asked for has no wait tool)."""
    from local_astra import cli

    monkeypatch.setattr(cli, "wait_tool_of", lambda a, m: False)
    monkeypatch.setattr(cli, "completion_check_of", lambda a, m: False)
    rig = MeasureRig(tmp_path, monkeypatch)
    rig.probes["ph2"] = {**PH2_OK, "resolver_calls": 1}
    manifest, _ = rig.run(capsys, 3)
    d = manifest.get("live_default") or {}
    assert d.get("wait_tool") is False and d.get("completion_check") is False, d
    assert rig.kits and not any(k.get("wait_tool") for k in rig.kits), rig.kits


# =====================================================================================================
# R-B1 drift: every live web keyword cli passes reaches the harness's Session
# =====================================================================================================


def test_no_live_web_keyword_of_the_cli_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C1/C3 (adversarial, the T-0201 D drift): every setting the cli passes ``Session`` at HEAD for ``--tools live``
    in web mode (the completion check and the screen reminder among them) is in ``session_keywords`` with the cli's
    value; no search keyword."""
    from local_astra.config import load

    want = t205._cli_live_web_keywords(tmp_path, capsys, monkeypatch)
    assert want.get("completion_check") is True and want.get("screen_reminder") is True, sorted(
        want
    )
    got = dict(api("session_keywords")(load().profiles["lean"]))
    problems = sorted(k for k, v in want.items() if k not in got or got[k] != v)
    assert not problems, f"R-B1: lacking or differing from the cli's live web keywords: {problems}"
    assert not got.get("search") and not got.get("web_search"), "search reaches the session"


# =====================================================================================================
# C3 through the real ``_one_turn``: the two arms. No socket, no browser, no engine.
# =====================================================================================================

ORIGIN = "http://127.0.0.1:59208"


class TurnRig:
    """``_one_turn``'s seams replaced: ``BrowserToolExecutor``, ``StartPageServer``, ``Session``, ``TurnMachine``,
    ``TurnPolicy`` by recording fakes; ``HudServer`` by a subclass of the real one whose ``open()`` gives a socketless
    page that decodes each message and hands it to the **real** ``HudServer.route`` (``start()`` refused);
    ``BrowserFrameSource.ingest`` wrapped to record its caller (the real one still runs). The fake session records
    ``apply_vision_select`` with its caller and, at submit, the frame its perception holds. ``cli.load_tool_kit``,
    T-0205's chain (``F._client``, ``LoopMeter``, ``G._wait_idle``, ``T._healthy``, the socket scan) by stubs.
    Every real socket is refused and recorded."""

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
        self.state: Any = copy.deepcopy(J.H1_GOLD)
        self.BrowserFrameSource = BrowserFrameSource
        #: every ``cli.load_tool_kit`` request ``_one_turn`` made (its keyword options)
        self.kits: list[dict[str, Any]] = []
        #: every kit object ``cli.load_tool_kit`` returned (T-0211 A4)
        self.kit_objs: list[Any] = []
        #: C-R2/C-R3 knobs: the executor's ``open`` raises; the ingress rejects every frame; the session ignores the
        #: select; ``script`` mode ("plain", or "stream-error": a logged step-1 body then a failed stream)
        self.open_raises = False
        self.reject_frames = False
        self.ignore_select = False
        self.mode = "plain"
        rig = self

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                rig.sockets.append(what)
                raise ConnectionRefusedError(f"T-0208 turn rig: no {what}")

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
            if rig.reject_frames:
                return (
                    None  # the sink's refusal value (HudServer.route counts it in frames_rejected)
                )
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
                    "T-0208 turn rig: the HUD server need not listen (socketless open())"
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
                rig.executors.append(self)

            async def open(self, url: str) -> None:
                rig.events.append(("open", url))
                if rig.open_raises:
                    raise RuntimeError("T-0208 turn rig: the browser failed to open the start page")

            async def execute(self, inv: Any) -> Any:
                raise AssertionError("T-0208 turn rig: nothing executes")

            async def close(self) -> None:
                rig.events.append("browser.close")

        class FakeServer:
            def __init__(self, static_dir: Any = None) -> None:
                self.static_dir = static_dir
                rig.servers.append(self)

            @property
            def origin(self) -> str:
                return ORIGIN

            def start(self) -> None:
                rig.events.append("server.start")

            def stop(self) -> None:
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
                self.clock = kwargs.get("clock")
                self.session_id = "s-t0208-turn-rig"
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
            rig.kits.append(dict(opts))
            sha = t205._set_sha(opts.get("page_overview", False), opts.get("wait_tool", False))
            kit = types.SimpleNamespace(tools_version=sha, repair_off=None)
            rig.kit_objs.append(kit)  # T-0211 A4: the object the search seam receives
            return kit, object()

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
        """At submit: the frame the session's perception holds and its vision choice; then one logged request, the
        ``turns.jsonl`` line, a history entry and ``turn_written``."""
        frame = None
        if s.perception is not None and hasattr(s.perception, "get_frame"):
            frame = await s.perception.get_frame(self.Source.SCREEN)
        self.at_submit = {"vision_source": s.vision_source, "frame": frame, "text": event.text}
        log, tid = s.session_log, "t1"
        data: dict[str, Any] = {"phase": "generate", "request_id": f"{tid}-gen", "attempt_id": "a1"}
        if self.mode == "stream-error":
            # C-R3: step 1's body goes out (logged, redacted by the log) holding the picture and the shared label;
            # then its stream breaks: the meter's row is LoopMeter's shape for a mid-stream exception (HTTP 0,
            # the error's repr), and the session records the failure
            assert frame is not None, "stream-error mode needs the shared frame at submit"
            data["prompt"] = {
                "messages": [
                    {"role": "system", "content": [_txt("You are Gearvis.")]},
                    {
                        "role": "user",
                        "content": [_img(frame.jpeg), _txt(SHARED_LABEL_SCREEN), _txt(event.text)],
                    },
                ]
            }
            s.args[1].generates.append(
                {
                    "request_id": "a1",
                    "repair": False,
                    "http_status": 0,
                    "error": "RemoteProtocolError('peer closed connection')",
                }
            )
            s.failures.append(RuntimeError("step 1 stream: peer closed connection"))
        log.record("llm.request", data, turn_id=tid)
        line = {
            "session_id": s.session_id,
            "turn_id": tid,
            "extras": {"tool_loop_end": None, "tool_steps": 1, "vision_source": s.vision_source},
        }
        with Path(s.config.turns_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        s.state.history.append(types.SimpleNamespace(user_text=event.text, assistant_spoken="ok"))
        s.turn_written.set()

    def run(self, hid: str, arm: str, **one_turn_kw: Any) -> dict[str, Any]:
        h = self.h
        assert hasattr(h, "_Run") and hasattr(h, "_one_turn"), (
            "T-0208 C3: the harness has no _Run/_one_turn"
        )
        known = {
            "base_url": "http://127.0.0.1:8001/v1",
            "out": self.out,
            "profile": self.profile,
            "sentinel": b"T0208-TURN-RIG-SENTINEL",
            "manifest": {},
            "vram": t205._FakeVram(),
        }
        names = {f.name for f in dataclasses.fields(h._Run)}
        run = h._Run(**{k: v for k, v in known.items() if k in names})
        self.last_run = run
        trial, steps, _thoughts = asyncio.run(h._one_turn(run, hid, arm, 0, **one_turn_kw))
        self.last_steps = [dict(s) for s in steps]
        assert self.sockets == [], self.sockets
        assert len(self.sessions) == 1 and len(self.executors) == 1, (self.sessions, self.executors)
        return dict(trial)


def _row(hid: str) -> Any:
    from local_astra.pages import handoff_tasks

    return next(r for r in handoff_tasks.HANDOFFS if r.handoff_id == hid)


@pytest.mark.parametrize("hid", IDS)
def test_shared_arm_goes_through_the_production_ingress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hid: str
) -> None:
    """C3 (reading 3; reviewer mutation 5, the select skipped): one ``astra.frame`` carrying the row's image bytes is
    routed by ``HudServer.route`` into the ``BrowserFrameSource`` that is both the server's ``frame_sink`` and the
    session's perception; then one ``select shared`` routed by ``HudServer.route`` to the session; then
    ``UserText(request)``. At submit the session looks at ``shared`` and its perception holds the image's bytes
    (screen source, 1280x800). The executor opens the base scenario's start page on the scenario server."""
    from local_astra.pages.scenario_tasks import SCENARIO_DIR

    rig = TurnRig(tmp_path, monkeypatch)
    row = _row(hid)
    image = (IMAGE_DIR / row.image).read_bytes()
    trial = rig.run(hid, "shared")
    assert trial.get("exception") is None, trial.get("exception")
    s = rig.sessions[-1]
    assert isinstance(s.perception, rig.BrowserFrameSource), (
        f"C3: the session's perception is {type(s.perception).__name__}, not a BrowserFrameSource"
    )
    assert rig.huds and rig.huds[-1].frame_sink is s.perception and rig.huds[-1].session is s, (
        "C3: the HudServer's frame sink and session are not the session's own"
    )
    mine = [i for i in rig.ingests if i[3] == image]
    assert [(f, fn, src.value) for f, fn, src, _ in mine] == [
        ("hud_server.py", "route", "screen")
    ], rig.ingests
    shared = [x for x in rig.selects if x[2] == "shared"]
    assert shared == [("hud_server.py", "route", "shared")], rig.selects
    ev = rig.events
    submit = ("submit", row.request)
    assert submit in ev, ev
    assert ev.index("ingest") < ev.index(("select", "shared")) < ev.index(submit), ev
    frame = rig.at_submit.get("frame")
    assert rig.at_submit.get("vision_source") == "shared", rig.at_submit
    assert frame is not None and frame.jpeg == image, (
        "C3: the session's perception does not hold the image"
    )
    assert (frame.width, frame.height) == (1280, 800), (frame.width, frame.height)
    assert [Path(sv.static_dir) for sv in rig.servers] == [Path(SCENARIO_DIR)], [
        sv.static_dir for sv in rig.servers
    ]
    opens = [e for e in ev if isinstance(e, tuple) and e[0] == "open"]
    assert opens == [("open", ORIGIN + row.start_page)], opens
    assert (trial.get("task"), trial.get("arm")) == (hid, "shared"), trial


@pytest.mark.parametrize("hid", IDS)
def test_typed_arm_posts_no_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hid: str
) -> None:
    """C3 (H-T208-9; reviewer mutation 6, the typed arm posting the image): no frame carrying the image is ingested,
    no ``shared`` select reaches the session, the session looks at ``browser`` and its perception holds no frame with
    the image's bytes; the request is the row's ``control_request``."""
    rig = TurnRig(tmp_path, monkeypatch)
    row = _row(hid)
    image = (IMAGE_DIR / row.image).read_bytes()
    trial = rig.run(hid, "typed")
    assert trial.get("exception") is None, trial.get("exception")
    assert not [i for i in rig.ingests if i[3] == image], "C3: the typed arm posted the image"
    assert not [x for x in rig.selects if x[2] == "shared"], rig.selects
    assert rig.at_submit.get("vision_source") == "browser", rig.at_submit
    frame = rig.at_submit.get("frame")
    assert frame is None or frame.jpeg != image, "C3: the typed arm's perception holds the image"
    assert [e for e in rig.events if isinstance(e, tuple) and e[0] == "submit"] == [
        ("submit", row.control_request)
    ], rig.events
    assert (trial.get("task"), trial.get("arm")) == (hid, "typed"), trial


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_session_gets_every_cli_live_keyword(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arm: str
) -> None:
    """C1/C3 drift through the real ``_one_turn``: every setting the cli passes ``Session`` for ``--tools live`` in
    web mode reaches this turn's ``Session`` with the cli's value (both arms); no search keyword. The executor is web
    mode with both deny-all proxy seams; the turn waits ``TURN_TIMEOUT_S`` = 300 s."""
    want = t205._cli_live_web_keywords(tmp_path / "cli", capsys, monkeypatch)
    rig = TurnRig(tmp_path, monkeypatch)
    rig.run("H2", arm)
    got = {k: v for k, v in rig.sessions[-1].kwargs.items() if k not in t205.CLI_PLUMBING}
    problems = sorted(k for k, v in want.items() if k not in got or got[k] != v)
    assert not problems, (
        f"R-B1: the turn's Session lacks or differs on the cli's live keywords: {problems}"
    )
    assert not got.get("search") and not got.get("web_search"), "search reaches the session"
    kw = rig.executors[-1].kwargs
    assert kw.get("web") is True and kw.get("start_origin") == ORIGIN, kw
    assert callable(kw.get("proxy_resolve")) and callable(kw.get("proxy_connect")), sorted(kw)
    assert [t for q, t in rig.waits if q.endswith("Event.wait")] == [300], rig.waits


@pytest.mark.parametrize("cli_kit", ["head", "cli-flipped"])
@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_kit_is_clis_wait_tool_and_page_overview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, cli_kit: str
) -> None:
    """C1/C3 (review finding 2): the one kit ``_one_turn`` asks ``cli.load_tool_kit`` for carries cli's resolved
    ``{wait_tool, page_overview}`` for live, both arms. ``head``: cli's own resolvers (by hand at HEAD: the wait
    tool on, the page overview off; the premise is asserted). ``cli-flipped``: ``cli.wait_tool_of`` gives off and
    ``cli.page_overview_of`` gives on, so a hard-coded kit is caught either way. An option not passed counts as
    ``False`` (``load_tool_kit``'s default)."""
    from local_astra import cli

    if cli_kit == "head":
        d = _heads_live_default()
        assert (d["wait_tool"], d["page_overview"]) == (True, False), d  # premise
        want = {"wait_tool": True, "page_overview": False}
    else:
        monkeypatch.setattr(cli, "wait_tool_of", lambda a, m: False)
        monkeypatch.setattr(
            cli,
            "page_overview_of",
            lambda *a, **k: types.SimpleNamespace(page_overview=True),
        )
        want = {"wait_tool": False, "page_overview": True}
    rig = TurnRig(tmp_path, monkeypatch)
    trial = rig.run("H1", arm)
    assert trial.get("exception") is None, trial.get("exception")
    assert len(rig.kits) == 1, rig.kits
    got = {k: bool(rig.kits[-1].get(k, False)) for k in want}
    assert got == want, (
        f"{arm}/{cli_kit}: the turn's kit asked for {rig.kits[-1]}, cli resolves {want}"
    )


# =====================================================================================================
# T-0211 A4: the optional search seam on ``_one_turn``
# =====================================================================================================

#: ``Session`` keywords whose values are per-run objects (compared by role, not by value)
_SEAM_OBJECT_KW = ("clock", "config", "session_log", "log_dir", "grounding", "tool_kit")


def _seam_snapshot(rig: TurnRig, trial: dict[str, Any]) -> dict[str, Any]:
    """Everything a turn writes or hands on, with per-run objects left out: the trial, its step lines, the session
    log's records (kind, turn and data; no wall-clock field), the ``load_tool_kit`` options, the ``Session``'s
    non-object keywords and the run's manifest."""
    logs = sorted((rig.out / "sessions").glob("*.jsonl"))
    out = str(rig.out)  # each rig has its own tmp dir; the log names it (``log.started``)
    records = [
        {k: r.get(k) for k in ("kind", "turn_id", "data")}
        for p in logs
        for r in (
            json.loads(ln.replace(out, "<out>"))
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    ]
    kwargs = rig.sessions[-1].kwargs
    return {
        "trial": json.loads(json.dumps(trial, default=repr)),
        "steps": json.loads(json.dumps(rig.last_steps, default=repr)),
        "records": records,
        "kit_options": [json.loads(json.dumps(k, default=repr)) for k in rig.kits],
        "session_kwargs": {
            k: repr(v) for k, v in sorted(kwargs.items()) if k not in _SEAM_OBJECT_KW
        },
        "session_kwarg_names": sorted(kwargs),
        "manifest": json.loads(json.dumps(rig.last_run.manifest, default=repr)),
    }


def _assert_seam_parameter() -> None:
    import inspect

    params = inspect.signature(api("_one_turn")).parameters
    assert "search_kit" in params, "T-0211 A4: _one_turn has no search_kit parameter"
    p = params["search_kit"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"T-0211 A4: search_kit is {p.kind}, not keyword-only"
    )
    assert p.default is None, f"T-0211 A4: search_kit defaults to {p.default!r}, not None"
    positional = [
        n for n, q in params.items() if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["run", "handoff_id", "arm", "repeat"], positional


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_session_gets_the_loaded_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """T-0211 A4 premise (green at BASE): with no transform, the turn's ``Session`` gets the very kit object
    ``cli.load_tool_kit`` returned, and ``load_tool_kit`` is asked once."""
    rig = TurnRig(tmp_path, monkeypatch)
    trial = rig.run("H1", arm)
    assert trial.get("exception") is None, trial.get("exception")
    assert len(rig.kit_objs) == 1, rig.kit_objs
    assert rig.sessions[-1].kwargs.get("tool_kit") is rig.kit_objs[0], (
        "the Session's kit is not the loaded one"
    )


def _seam_run(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, arm: str, name: str, kw: dict[str, Any]
) -> dict[str, Any]:
    with monkeypatch.context() as mp:
        rig = TurnRig(tmp, mp)
        trial = rig.run("H2", arm, **kw)
        assert trial.get("exception") is None, (name, trial.get("exception"))
        assert len(rig.kit_objs) == 1, (name, rig.kit_objs)
        assert rig.sessions[-1].kwargs.get("tool_kit") is rig.kit_objs[0], (
            f"{name}: the Session's kit is not the one load_tool_kit returned"
        )
        return _seam_snapshot(rig, trial)


#: equal-length run directories: the ``log.stopped`` record's ``data.bytes`` is the log file's own size, and the log
#: holds its own path, so directories of different lengths differ there for no reason of ``search_kit``'s
_SEAM_DIRS = ("omit", "none")


def _stopped_bytes(records: list[dict[str, Any]]) -> list[Any]:
    return [
        r["data"]["bytes"]
        for r in records
        if r.get("kind") == "log.stopped"
        and isinstance(r.get("data"), dict)
        and "bytes" in r["data"]
    ]


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_seam_snapshot_premise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Premise of the seam row below (green at BASE): two calls that both omit ``search_kit``, run in the two
    directories that row uses, give equal snapshots with ``log.stopped``'s ``bytes`` kept in, so a difference
    there is ``search_kit``'s and not the directory names'."""
    assert len({len(d) for d in _SEAM_DIRS}) == 1, _SEAM_DIRS
    a, b = (_seam_run(tmp_path / d, monkeypatch, arm, d, {}) for d in _SEAM_DIRS)
    assert a["records"], "premise: the turn logged no record"
    assert _stopped_bytes(a["records"]), (
        "premise: no log.stopped bytes, so the snapshot lost that power"
    )
    diff = sorted(k for k in a if a[k] != b[k])
    assert not diff, f"two runs that omit search_kit differ in {diff}"


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_search_seam_default_is_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """T-0211 A4: ``_one_turn`` takes one optional keyword-only ``search_kit`` (default ``None``). Called with
    ``search_kit=None`` it is the call without it (BASE): the same trial, step lines, session-log records,
    ``load_tool_kit`` options, ``Session`` keywords and manifest, and the ``Session`` gets the loaded kit itself."""
    _assert_seam_parameter()
    omit_dir, none_dir = _SEAM_DIRS
    omitted = _seam_run(tmp_path / omit_dir, monkeypatch, arm, "omitted", {})
    none = _seam_run(tmp_path / none_dir, monkeypatch, arm, "none", {"search_kit": None})
    assert omitted["records"], "premise: the turn logged no record"
    diff = sorted(k for k in omitted if omitted[k] != none[k])
    assert not diff, f"T-0211 A4: search_kit=None differs from BASE's call in {diff}"


@pytest.mark.parametrize("arm", ARMS)
def test_one_turn_search_seam_applies_the_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """T-0211 A4 (as T-0209's ``rtx_kit``): a ``search_kit`` transform is called once, with the kit
    ``cli.load_tool_kit`` returned, before any ``Session`` is built; the ``Session`` gets the transform's result."""
    _assert_seam_parameter()
    rig = TurnRig(tmp_path, monkeypatch)
    calls: list[tuple[Any, int]] = []

    def transform(kit: Any) -> Any:
        calls.append((kit, len(rig.sessions)))
        return types.SimpleNamespace(**vars(kit), search="T-0211 loopback search marker")

    trial = rig.run("H3", arm, search_kit=transform)
    assert trial.get("exception") is None, trial.get("exception")
    assert len(calls) == 1, f"the transform ran {len(calls)} times"
    got_kit, sessions_then = calls[0]
    assert got_kit is rig.kit_objs[0], "the transform did not get the loaded kit"
    assert sessions_then == 0, "the transform ran after the Session was built"
    session_kit = rig.sessions[-1].kwargs.get("tool_kit")
    assert session_kit is not rig.kit_objs[0], "the Session got the untransformed kit"
    assert getattr(session_kit, "search", None) == "T-0211 loopback search marker", session_kit


def _ingress_failure(rig: TurnRig, case: str) -> None:
    if case == "wrong-vision-source":
        rig.ignore_select = True  # the select is routed but the session stays on the browser
    elif case == "frames-rejected":
        rig.reject_frames = True  # the ingress refuses the picture: frames_rejected 1
    else:
        raise KeyError(case)


@pytest.mark.parametrize("which", IMPLS)
def test_shared_turn_that_never_sent_step1_is_ph3_none_not_void(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """C-R2: a shared turn whose ``browser.open`` raises before submit (so no step-1 request goes out, and the
    failure is not an ingress self-check) records an exception and ``ph3: None``; the run is not voided by it (the
    report, by the harness's rules and by the oracle's, is not ``void``: the errored turn is ERRORED_STOP's)."""
    rig = TurnRig(tmp_path, monkeypatch)
    rig.open_raises = True
    trial = rig.run("H1", "shared")
    assert ("submit", _row("H1").request) not in rig.events, rig.events
    assert trial.get("exception"), "C-R2: the failed open is not recorded as the turn's exception"
    assert "ph3" in trial and trial["ph3"] is None, (
        f"C-R2: a shared turn that sent no step-1 request gets ph3 None, not {trial.get('ph3')!r}"
    )
    report = impl(which, "build_report")([trial], [], _probes(), None)
    assert report["label"] != "void", (which, report.get("label"), report.get("probes"))


@pytest.mark.parametrize("case", ["wrong-vision-source", "frames-rejected"])
def test_ingress_self_check_failure_is_ph3_false_and_void(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """C-R2's other side (the guard too wide): an ingress self-check failure (the session not looking at
    ``shared`` after the select, or the HUD ingress rejecting the picture) still gives the shared turn ``ph3:
    False``, never ``None``, and the report is ``void``."""
    rig = TurnRig(tmp_path, monkeypatch)
    _ingress_failure(rig, case)
    trial = rig.run("H1", "shared")
    assert ("submit", _row("H1").request) not in rig.events, rig.events
    assert trial.get("exception"), f"{case}: the self-check did not fail the turn"
    assert trial.get("ph3") is False, f"{case}: ph3 is {trial.get('ph3')!r}, not False"
    for which in ("harness", "eval-oracle"):
        report = impl(which, "build_report")([trial], [], _probes(), None)
        assert report["label"] == "void", (which, case, report.get("label"))


@pytest.mark.parametrize("which", IMPLS)
def test_step1_stream_error_after_the_request_holds_ph3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """C-R3: a shared turn whose step-1 body went out (logged with the picture and ``SHARED_LABEL_SCREEN``) and whose
    stream then errored gives ``ph3: True`` (the bodies held it); the engine error is counted (1 of 1 turn) and the
    turn is not a success, though the end state is the golden one (``rig.state`` is ``H1_GOLD``)."""
    rig = TurnRig(tmp_path, monkeypatch)
    rig.mode = "stream-error"
    trial = rig.run("H1", "shared")
    assert rig.at_submit.get("frame") is not None, rig.at_submit
    assert trial.get("ph3") is True, f"C-R3: ph3 is {trial.get('ph3')!r} after a logged step-1 body"
    steps = rig.last_steps
    assert [(s.get("step"), s.get("http_status")) for s in steps] == [(1, 0)], steps  # premise
    report = impl(which, "build_report")([trial], steps, _probes(), None)
    row = report["per_arm"]["shared"]["H1"]
    assert (row["n"], row["engine_errors"], row["successes"]) == (1, 1, 0), row
    assert report["label"] != "void", report.get("probes")


# =====================================================================================================
# C6: the locked wrapper (T-0131's run_locked rig, as T-0205's rows)
# =====================================================================================================

L_OUT = f"logs/t0208-handoff-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0208_handoff_run.py"
L_EVAL = "tests/eval/test_t0208_handoff_run.py"
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
    "local_astra/pages/handoff_tasks.py",
    "local_astra/pages/handoff/h1_steam_wand.jpg",
    "local_astra/pages/handoff/h2_stove_box.jpg",
    "local_astra/pages/handoff/h3_chat.jpg",
    "local_astra/pages/handoff/MANIFEST.md",
    "local_astra/eval/cd/stats.py",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch, merges: tuple[str, ...] = (*OTHERS, MERGE_B, MERGE_A2, MERGE_C)
) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0208 script, harness, eval and merges."""
    assert LOCKED.is_file(), f"T-0208 C6: {LOCKED.relative_to(REPO)} does not exist yet"
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", t205._git_stub(merges))
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0208_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json order.json; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


MERGE_CASES: dict[str, tuple[str, ...]] = {
    "no-b-only-a2-and-c": (*OTHERS, MERGE_A2, MERGE_C),
    "no-b-branch-merges-only": OTHERS,
}


@pytest.mark.parametrize("case", list(MERGE_CASES))
def test_locked_script_needs_merge_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """C6 (adversarial, reading 6): without ``Merge T-0208-b:`` -- the log holding A1's, A2's and C's merges, the
    branch merges into T-0208-b-impl and a ``Merge T-0208-bb:`` -- the script refuses (exit 2, ``REFUSED``) before
    anything starts."""
    res = _locked(monkeypatch, MERGE_CASES[case])(tmp_path)
    assert res.rc == 2 and "REFUSED" in res.output, (case, res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_refuses_vllm_sh_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C6 (T-0205 C's boot regime): ``services/vllm.sh`` changed since 806da12 refuses with "services/vllm.sh changed
    since 806da12", exit 2, before the boot or any step; the script holds that check."""
    res = _locked(monkeypatch)(tmp_path, vllm_drift=True)
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert "services/vllm.sh changed since 806da12" in res.output, res.output[-2000:]
    assert res.started() == [], f"ran before the refusal: {res.started()}"
    code = "\n".join(t205._script_code(LOCKED))
    assert "git diff 806da12..HEAD -- services/vllm.sh" in code, (
        "C6: the vllm.sh drift check is missing"
    )


LOCKED_REFUSALS: dict[str, dict[str, Any]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """C6 (T-0205 C's boot, verbatim, so its refusals too): ``REFUSED``, exit 2, before the boot or any step."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


@pytest.mark.parametrize("merges", ["all", "b-only"])
def test_locked_script_steps_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, merges: str
) -> None:
    """C6, the other side of every refusal (B reachable; ``b-only``: neither A2 nor C, so neither is required): the
    harness under ``timeout -k 60s 60m`` with a loopback ``--base-url`` and ``--out-dir logs/t0208-handoff-<sha7>``,
    then ``tests/eval/test_t0208_handoff_run.py -m gpu`` with ``T0208_RUN_DIR`` set to it (``report.json`` exists);
    every ``uv run`` uses ``--locked``; exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    chosen = (*OTHERS, MERGE_B, MERGE_A2, MERGE_C) if merges == "all" else (*OTHERS, MERGE_B)
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
    assert evals[0].env.get("T0208_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv) and len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 60 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_shape() -> None:
    """C6 by text: ``bash -n`` clean; the boot is T-0205 C's verbatim; the ``Merge T-0208-b:`` check through a
    here-string, never a pipe into ``grep``, and no T-0205 merge check left from the fork; ``timeout -k 60s 60m``;
    ``--locked`` on every ``uv run``; the eval only if ``report.json`` exists; T-0205's eval and run dir never named."""
    assert LOCKED.is_file(), f"T-0208 C6: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    code = t205._script_code(LOCKED)
    body = "\n".join(code)
    assert t205._boot(code) == t205._boot(t205._script_code(T0205_LOCKED)), (
        "C6: the boot is not T-0205 C's verbatim"
    )
    assert "timeout -k 60s 60m" in body
    checks = [ln for ln in code if "Merge T-0208" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert any("T-0208-b:" in ln for ln in checks), checks
    assert not [ln for ln in code if "Merge T-0205" in ln], "the fork kept T-0205's merge checks"
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert (
        "t0208-handoff-" in body
        and L_EVAL in body
        and L_HARNESS in body
        and "T0208_RUN_DIR" in body
    )
    assert "-m gpu" in body and "report.json" in body
    for old in ("test_t0205_demo_run", "T0205_RUN_DIR", "t0205-demo-"):
        assert old not in body, f"the script names {old}"


# =====================================================================================================
# Dress row: the harness's real ``_one_turn`` on a real Session, real HudServer + BrowserFrameSource, the real
# web-mode executor on the scenario root, a scripted fake LLM, the harness's own deny-all seams (browser; CPU).
# =====================================================================================================

DRESS_GATE = "ASTRA_BROWSER_TESTS"
_Q = '<|"|>'
DRESS_SCROLL = f"<|tool_call>call:scroll{{direction:{_Q}down{_Q}}}<tool_call|>"
DRESS_SAY = "That is the steam wand of a Barista Express BES870; I will find the replacement. "
DRESS_DONE = "I scrolled the page. "
DRESS_CHECK = "I scrolled the page and looked again. "


class DressRig:
    """Only the engine half of ``_one_turn`` is replaced: ``F._client`` returns a scripted ``FakeLlm`` (plus the real
    client's ``request_body``, which the harness calls after the turn), ``G._wait_idle``/``G.read_metrics``/
    ``T._healthy`` answer as an idle healthy engine would, and ``run.vram`` is T-0205's fake. ``Session`` and
    ``HudServer`` are subclasses of the real classes that only record (the instance, every submitted event, every
    ``apply_vision_select``, the socketless connection); ``BrowserToolExecutor``, ``StartPageServer``,
    ``BrowserFrameSource``, ``TurnMachine``, ``cli.load_tool_kit`` and the harness's ``_DenyNet`` are untouched."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra.conductor import hud_server as hud_mod
        from local_astra.conductor import session as session_mod
        from local_astra.conductor.llm_client import VllmClient
        from local_astra.config import load
        from tests.fakes.llm import FakeLlm, _refusing_transport
        from tests.unit._tool_turn_support import answer, tool

        self.h = h = harness()
        self.profile = load().profiles["lean"]
        self.tmp = tmp_path
        self.sessions: list[Any] = []
        self.submitted: list[Any] = []
        self.selects: list[str] = []
        self.connections: list[Any] = []
        self.llms: list[Any] = []
        self.published: list[Any] = []
        #: (request id, the session's ``vision_source`` when that generate was sent)
        self.source_at: list[tuple[str, str]] = []
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
                s = rig.sessions[-1] if rig.sessions else None
                rig.source_at.append((request_id, str(getattr(s, "vision_source", None))))
                return super().generate(prompt, request_id, options, xargs=xargs, cd=cd)

        def client(*a: Any, **k: Any) -> Any:
            llm = DressLlm()
            llm.script = []  # captions say nothing
            llm.generate_scripts = [[answer(DRESS_SAY), tool(DRESS_SCROLL)]]
            llm.step_scripts = [[answer(DRESS_DONE)]]
            llm.check_scripts = [[answer(DRESS_CHECK)]]
            rig.llms.append(llm)
            return llm

        RealSession = session_mod.Session
        RealHud = hud_mod.HudServer

        class SpySession(RealSession):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                rig.sessions.append(self)

            def submit(self, event: Any) -> None:
                rig.submitted.append(event)
                super().submit(event)

            def apply_vision_select(self, message: Any) -> Any:
                rig.selects.append(str(getattr(message.source, "value", message.source)))
                return super().apply_vision_select(message)

        class SpyHud(RealHud):  # type: ignore[misc, valid-type]
            def open(self) -> Any:
                conn = super().open()
                rig.connections.append(conn)
                return conn

            def publish(self, message: Any) -> None:
                # recorded at publish: the socketless outbox holds only the newest vision frame (T-0177 B8)
                rig.published.append(message)
                super().publish(message)

        async def wait_idle(*a: Any, **k: Any) -> bool:
            return True

        mp.setattr(session_mod, "Session", SpySession)
        mp.setattr(hud_mod, "HudServer", SpyHud)
        mp.setattr(_chain(h, "F"), "_client", client)
        mp.setattr(_chain(h, "G"), "_wait_idle", wait_idle)
        mp.setattr(_chain(h, "G"), "read_metrics", lambda *a, **k: None)
        mp.setattr(_chain(h, "T"), "_healthy", lambda *a, **k: True)

    def run(self, hid: str, arm: str) -> dict[str, Any]:
        h = self.h
        out = self.tmp / f"out-{arm}"
        (out / "sessions").mkdir(parents=True)
        known = {
            "base_url": "http://127.0.0.1:8001/v1",
            "out": out,
            "profile": self.profile,
            "sentinel": b"T0208-DRESS-SENTINEL",
            "manifest": {},
            "vram": t205._FakeVram(),
        }
        names = {f.name for f in dataclasses.fields(h._Run)}
        run = h._Run(**{k: v for k, v in known.items() if k in names})
        trial, _steps, _thoughts = asyncio.run(h._one_turn(run, hid, arm, 0))
        return dict(trial)


def _dress_bodies(
    llm: Any,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """(step-1 bodies, later bodies) of one turn: every non-caption generate and every prefill, as its wire body
    (the real ``VllmClient.request_body``); "later" is a ``-gen-s<k>`` id (steps, re-asks, checks)."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit.test_session_live import _wire

    first: list[tuple[str, dict[str, Any]]] = []
    later: list[tuple[str, dict[str, Any]]] = []
    items = [(rid, p) for rid, p, _o in llm.generates if not is_caption_prompt(p)]
    items += [(f"prefill:{rid}", p) for rid, p in llm.prefills]
    for rid, prompt in items:
        (later if re.search(r"-gen-s\d+", rid) else first).append((rid, _wire(prompt)))
    return first, later


def _dress_images(body: dict[str, Any]) -> list[str]:
    from tests.unit._live_browser_support import image_urls

    return image_urls(body)


@pytest.mark.browser
@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get(DRESS_GATE) != "1",
    reason=f"{DRESS_GATE} is not 1: the dress row drives a real Chromium (T-0208 C, dress row)",
)
def test_dress_h1_shared_and_typed_through_the_real_one_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3/C4 dress (H1, one ``shared`` and one ``typed`` turn, n = 1 each): the paths the stubbed ``TurnRig`` never
    ran -- the real hello/greet, ``HudConnection.receive``, ``HudServer.route`` into the real ``BrowserFrameSource``
    and into the real ``Session.apply_vision_select`` -- driven by the harness's own ``_one_turn``. Script: step 1
    says ``DRESS_SAY`` and scrolls down (a non-coordinate action, executed in the real browser); step 2 is a no-call
    answer; a completion check, if armed, answers ``DRESS_CHECK``.

    Shared turn: the session's ``vision_source`` was ``"shared"`` when step 1's generate was sent; step 1's wire body
    carries exactly one image, the H1 file's bytes as a JPEG data URL, and ``SHARED_LABEL_SCREEN`` as text; no later
    body carries the shared base64; P-H3 (the harness's own verdict from the session log) is true; the HUD rejected
    no frame; no ``ClientDisconnected`` was submitted and the connection stayed open; exactly one select
    (``shared``) reached the session; the first published ``astra.vision.frame`` is ``shared``. Typed turn: no
    select, ``vision_source`` ``"browser"`` at step 1, no body carries the shared base64, ``ph3`` is ``None``.
    Both: no exception, no session failure, zero resolver/connector calls and no non-loopback socket.

    Regime: Playwright's pinned Chromium headless, web mode on loopback, the harness's deny-all seams, HEAD's live
    defaults through cli, real ``SessionClock``; scripted fake LLM, no engine; n = 1 per arm. Wall time printed."""
    import time

    from local_astra.contracts import events as E

    image = (IMAGE_DIR / _row("H1").image).read_bytes()
    shared_b64 = base64.b64encode(image).decode("ascii")
    shared_url = "data:image/jpeg;base64," + shared_b64

    rig = DressRig(tmp_path, monkeypatch)
    t0 = time.monotonic()
    trial = rig.run("H1", "shared")
    shared_wall = time.monotonic() - t0
    assert trial.get("exception") is None, trial.get("exception")
    assert len(rig.sessions) == 1 and len(rig.llms) == 1, (rig.sessions, rig.llms)
    session, llm = rig.sessions[0], rig.llms[0]
    assert not session.failures, session.failures
    # the select through the real HudConnection.receive -> HudServer.route -> Session.apply_vision_select
    assert rig.selects == ["shared"], rig.selects
    # vision source at step 1 (the first non-caption, non-step generate)
    first, later = _dress_bodies(llm)
    step1_ids = {rid for rid, _b in first if not rid.startswith("prefill:")}
    assert step1_ids, f"no step-1 generate was sent: {llm.generates}; trial {trial}"
    at_step1 = [src for rid, src in rig.source_at if rid in step1_ids]
    assert at_step1 and set(at_step1) == {"shared"}, rig.source_at
    # step-1 body: exactly the shared image, then the shared label
    for rid, body in first:
        if rid.startswith("prefill:"):
            continue
        urls = _dress_images(body)
        assert urls == [shared_url], (
            f"{rid}: step 1 carries {len(urls)} image(s); shared among them: {shared_url in urls}"
        )
        assert SHARED_LABEL_SCREEN in json.dumps(body), f"{rid}: no SHARED_LABEL_SCREEN in step 1"
    assert later, "premise: the scripted scroll led to a step-2 request"
    for rid, body in later:
        assert shared_b64 not in json.dumps(body), f"{rid}: a later body carries the shared image"
    assert trial.get("ph3") is True, f"P-H3 is {trial.get('ph3')!r}"
    counts = trial.get("hud_counts") or {}
    assert counts.get("frames_rejected") == 0, counts
    assert not [e for e in rig.submitted if isinstance(e, E.ClientDisconnected)], rig.submitted
    assert rig.connections and all(not c.closed for c in rig.connections), [
        c.close_reason for c in rig.connections
    ]
    frames = [m for m in rig.published if getattr(m, "type", None) == "astra.vision.frame"]
    seen = [(f.step, str(getattr(f.source, "value", f.source))) for f in frames]
    step1_frames = [f for f in frames if f.step == 1]
    assert step1_frames and all(
        str(getattr(f.source, "value", f.source)) == "shared"
        and [i.jpeg_b64 for i in f.images] == [shared_b64]
        for f in step1_frames
    ), seen
    assert all(src == "browser" for step, src in seen if step >= 2), seen
    n_published = len(rig.published)
    assert trial.get("resolver_calls") == 0 and trial.get("connector_calls") == 0, trial
    assert not trial.get("non_loopback"), trial.get("non_loopback")

    t1 = time.monotonic()
    typed = rig.run("H1", "typed")
    typed_wall = time.monotonic() - t1
    assert typed.get("exception") is None, typed.get("exception")
    assert len(rig.sessions) == 2 and len(rig.llms) == 2, (rig.sessions, rig.llms)
    assert not rig.sessions[1].failures, rig.sessions[1].failures
    assert rig.selects == ["shared"], f"the typed turn sent a select: {rig.selects}"
    first2, later2 = _dress_bodies(rig.llms[1])
    ids2 = {rid for rid, _b in first2 if not rid.startswith("prefill:")}
    assert ids2, "no typed step-1 generate"
    assert {src for rid, src in rig.source_at if rid in ids2} == {"browser"}, rig.source_at
    for rid, body in first2 + later2:
        assert shared_b64 not in json.dumps(body), f"typed {rid}: carries the shared image"
    assert typed.get("ph3") is None, typed.get("ph3")
    typed_frames = [
        (m.step, str(getattr(m.source, "value", m.source)))
        for m in rig.published[n_published:]
        if getattr(m, "type", None) == "astra.vision.frame"
    ]
    assert typed_frames and all(src == "browser" for _s, src in typed_frames), typed_frames
    assert typed.get("resolver_calls") == 0 and typed.get("connector_calls") == 0, typed
    assert not typed.get("non_loopback"), typed.get("non_loopback")
    print(
        f"T-0208 C dress: shared {shared_wall:.1f} s (steps {trial.get('tool_steps')}, "
        f"end {trial.get('tool_loop_end')}, checks {trial.get('completion_checks')}), "
        f"typed {typed_wall:.1f} s (steps {typed.get('tool_steps')}, end {typed.get('tool_loop_end')}); "
        f"shared ids step-1 {sorted(r for r, _b in first)} later {sorted(r for r, _b in later)}"
    )


def test_t0208_task_names_this_module() -> None:
    """Activation bookkeeping (tests.md): the task file names both C modules under "Tests to write"."""
    text = (REPO / "docs" / "tasks" / "T-0208.md").read_text(encoding="utf-8")
    tests = text.split("## Tests to write", 1)[1].split("\n## ", 1)[0]
    assert "tests/unit/test_t0208_handoff_harness.py" in tests
    assert "tests/eval/test_t0208_handoff_run.py" in tests
