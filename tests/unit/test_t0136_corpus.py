"""T-0136 group A1 (unit, the pure parts): the outcome-label calibration corpus builder
``scripts/build_outcome_corpus.py`` (docs/tasks/T-0136.md, A1-1..A1-8, "Tests to write (A1)").

**Status.** Activated for group A1 (rulings O-A1 in the task file; no ``pending`` markers). Until the builder exists
each builder row fails on the named assertion in :func:`builder`, never on an ImportError at collection (the script
is loaded by path inside the helper). The probe-fixture rows and the A1-8 scope row are green at BASE_A1.

**O-A1.** (3) ``--out`` is exactly ``logs/t0136-outcome-corpus-<sha7>``: any other name under ``logs/`` is refused
too. (4) t0132 is dropped; t0134 records (no ``frame_kind``, no ``cart_after``) are read as live frames with the cart
check skipped. (7) A1-7 determinism is judged on pair structure plus frames within JPEG noise (max per-pixel gray
|Δ| <= 8, no pixel over 24), and the builder records the max |Δ| it saw between two builds: pinned here as
``compare_builds`` below (where the CLI writes that number is not pinned).

**The surface these rows pin.** The task fixes behaviour, not names; these are the hooks a row needs, and the
implementer must provide them (precedent: T-0111c's and T-0151's harness modules).

==========================================================  =========================================================
``main(argv: Sequence[str] | None = None) -> int``          the CLI of A1-1; ``EXIT_REFUSED == 2``
module-level ``_git(*args: str) -> str``                    T-0111g's seam (``scripts/gpu/t0111g_live_smoke.py``):
                                                            the dirty-tree check reads ``_git("status", ...)`` and
                                                            the sha7 comes from ``_git("rev-parse", ...)``; no other
                                                            process starts before a refusal
``read_sequences(run_dir: Path)``                           A1-2: a mapping ``(run, arm, task, repeat) -> [record,
                                                            ...]``; ``run`` is ``run_dir.name``; the value
                                                            is the logged executed step records (``status ==
                                                            "executed"``), as logged, in ``step`` order
``pair_id(before_sha, after_sha, point, prev_point) -> str``  A1-4: 16 lowercase hex; ``point`` a pair of ints (a tuple
                                                            or a JSON list), ``prev_point`` the same or ``None``
``first_divergence(logged, replayed) -> int | None``        A1-3: the index of the first logged step that the replay
                                                            did not reproduce (``None``: all reproduced); a replayed
                                                            item is ``{"action", "cart_after"}``
``compare_builds(out_a: Path, out_b: Path) -> Mapping``     O-A1 (7): two out dirs' pairs matched on their structure
                                                            (every ``pairs.jsonl`` field except ``pair_id``,
                                                            ``before``, ``after``); ``structure_equal`` (bool),
                                                            ``max_abs_delta`` (int: the largest per-pixel gray |Δ|
                                                            over every matched pair's ``before`` and ``after``) and
                                                            ``pixels_over_24`` (int: pixels with |Δ| > 24, summed)
==========================================================  =========================================================

**Refusals (A1-1).** "Before any browser starts" is observed, not inferred: every process start (``subprocess.Popen``,
``asyncio.create_subprocess_exec`` and the event loop's ``subprocess_exec``, which is where Playwright starts its
driver and Chromium) is recorded and refused, so a builder that gets past a refusal it should have made reaches a
process start and the row names it. A refusal also leaves the out dir uncreated. The companion row (a guard too wide)
is the same set-up with every condition passing: the builder must not refuse, and must reach a process start.

Expected values are by hand from the task file and from the real records of ``logs/t0151-overview-frame-ab-4471255/
steps.jsonl`` (S2 arm B repeat 11, the over-click of the Goal; S3 arm B repeat 4), copied as literals: no row reads
``logs/``.

BASE_A1 = 6405e0d (main when the A1 branch was cut).

**A1c (docs/tasks/T-0136.md, "Amendment A1c/A2b", A1c-1..A1c-6, A1c-N).** The rows under "A1c" at the end. Activated
for group A1c (O-A1c; no ``pending`` markers): until the builder lands each builder row fails on a named assertion;
the fixture, count and scope rows hold at BASE_A1c. The surface the A1c rows pin (names the implementer must provide; the task fixes behaviour):

==========================================================  =========================================================
module-level ``PROBE_PAGES: tuple[Path, ...]``              A1c-3: the probe pages in run order, ``(outcomes.html,
                                                            outcomes_b.html)`` under ``tests/fixtures/t0136/``; read
                                                            at call time (a row replaces it with tmp pages)
``probe_table() -> list[dict]``                             A1c-3: the run plan, every page's table in
                                                            ``PROBE_PAGES`` order, each entry a copy of its table
                                                            entry plus ``probe_page`` (the file name). ``--probes``
                                                            runs exactly ``probe_table()`` (A1's caret row already
                                                            patches it)
``probe_url(origin: str, probe: Mapping) -> str``           A1c-3: ``origin + "/" + probe_page + query`` (no query:
                                                            nothing appended, so page a's URL is A1's)
``frame_groups(frames: Mapping[str, bytes]) -> dict``       A1c-4: frame sha256 -> cluster id over the given encoded
                                                            frames (decoded to full-resolution gray)
``<out>/frame_groups.json``                                 A1c-4: that mapping for every frame of the build
pair records ``probe`` (the probe id, as A1 writes it) and  A1c-3; the manifest regime gains ``probe_pages_sha256``
``probe_page``                                              and keeps ``probe_page_sha256``
==========================================================  =========================================================

A ``pre`` is an object with integer ``x``/``y`` (model units); the rows also give it ``x_px``/``y_px`` (by hand) and
``target``. A refused table (A1c-3) exits 2 before any process starts, like A1-1's refusals, and names the probe.

BASE_A1c = d4adf4f (main when the A1c branch was cut).

Regime: CPU only, no browser, no network (loopback included), n = 1.
"""

from __future__ import annotations

import asyncio
import asyncio.base_events
import functools
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts" / "build_outcome_corpus.py"
PROBE_PAGE = REPO / "tests" / "fixtures" / "t0136" / "outcomes.html"
#: The commit group A1 branched from (main at "T-0136: plan ..."). A literal, never a moving ref.
BASE_A1 = "6405e0d"
#: The implementation tip for the A1-8 scope row; starts at BASE_A1 (no implementation yet) and is bumped by the
#: test-author or the orchestrator to the last implementation commit before review (the implementer never edits tests).
TIP_A1 = "bb80cde"

CLASSES = ("none", "near", "elsewhere", "page")


# =====================================================================================================
# Loading the builder by path
# =====================================================================================================


def _load(name: str) -> ModuleType:
    assert BUILDER.is_file(), f"T-0136 A1-1: {BUILDER.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location(name, BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{BUILDER.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


@functools.cache
def builder() -> ModuleType:
    return _load("_t0136_build_outcome_corpus_unit")


def api(name: str, module: ModuleType | None = None) -> Any:
    b = builder() if module is None else module
    assert hasattr(b, name), (
        f"T-0136 A1: {BUILDER.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(b, name)


# =====================================================================================================
# Hand-built steps.jsonl records (the shape of logs/t0151-overview-frame-ab-4471255/steps.jsonl)
# =====================================================================================================


def rec(
    task: str,
    arm: str | None,
    repeat: int,
    step: int,
    *,
    name: str | None = None,
    arguments: dict[str, Any] | None = None,
    action: str | None = None,
    status: str | None = "executed",
    frame_kind: str | None = None,
    scrolled_px: int | None = None,
    cart_after: Any = "absent",
) -> dict[str, Any]:
    """One logged step record; ``status=None`` is a step that ended without a call (the logged shape), ``"refused"``
    a refused call. ``arm=None`` omits the key; ``cart_after="absent"`` and ``frame_kind="absent"`` omit those keys
    (t0134's records carry neither)."""
    executed = status == "executed"
    r: dict[str, Any] = {
        "task": task,
        "repeat": repeat,
        "step": step,
        "http_status": 200,
        "status": status,
        "reason": None,
        "call_name": name,
        "executed": executed,
        "executed_name": name if executed else None,
        "executed_arguments": arguments if executed else None,
        "action": action if executed else None,
        "frame_kind": frame_kind,
        "scrolled_px": scrolled_px,
        "visual_tokens": 1120,
        "reground": False,
        "has_image": True,
    }
    if arm is not None:
        r["arm"] = arm
    if cart_after != "absent":
        r["cart_after"] = cart_after
    if frame_kind == "absent":
        del r["frame_kind"]
    return r


C1 = "clicked at (423, 631)"


#: S2 arm B repeat 11 (t0151), the Goal's over-click, verbatim in the fields A1 reads.
S2_B11 = [
    rec(
        "S2",
        "B",
        11,
        1,
        name="click_at",
        arguments={"x": 423, "y": 631},
        action="clicked at (423, 631)",
        cart_after=["blue mug"],
    ),
    rec(
        "S2",
        "B",
        11,
        2,
        name="click_at",
        arguments={"x": 423, "y": 631},
        action="clicked at (423, 631)",
        cart_after=["blue mug", "blue mug"],
    ),
    rec("S2", "B", 11, 3, status=None, cart_after=None),
]
#: S3 arm B repeat 4 (t0151), verbatim in the fields A1 reads.
S3_B4 = [
    rec(
        "S3",
        "B",
        4,
        1,
        name="type_at",
        arguments={"press_enter": True, "text": "Ada Lovelace", "x": 305, "y": 335},
        action="clicked at (305, 335) and typed 12 characters and pressed Enter",
        cart_after=None,
    ),
    rec(
        "S3",
        "B",
        4,
        2,
        name="type_at",
        arguments={"press_enter": True, "text": "ada@example.com", "x": 305, "y": 448},
        action="clicked at (305, 448) and typed 15 characters and pressed Enter",
        cart_after=None,
    ),
    rec(
        "S3",
        "B",
        4,
        3,
        name="type_at",
        arguments={"press_enter": True, "text": "Hello from Astra", "x": 305, "y": 624},
        action="clicked at (305, 624) and typed 16 characters and pressed Enter",
        cart_after=None,
    ),
    rec(
        "S3",
        "B",
        4,
        4,
        name="click_at",
        arguments={"x": 141, "y": 816},
        action="clicked at (141, 816)",
        cart_after=None,
    ),
    rec("S3", "B", 4, 5, status=None, cart_after=None),
]


def write_run(root: Path, name: str, records: Sequence[Mapping[str, Any]]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return d


# =====================================================================================================
# A1-2: steps.jsonl into ordered sequences
# =====================================================================================================


def test_read_sequences_groups_orders_and_keeps_executed_only(tmp_path: Path) -> None:
    """A1-2: each logged ``(run, arm, task, repeat)`` is one sequence of its executed steps in ``step`` order.

    By hand: S2/B/11 has steps 1, 2 executed (3 ended without a call); S3/B/4 has 1-4; S2/A/11 (the same task and
    repeat in the other arm: a key without ``arm`` would merge it into S2/B/11) has a refused step 1 and an executed
    scroll at 2 and click at 3, written in the order 3, 1, 2 (file order is not step order). The records of the
    three turns are interleaved in the file."""
    s2a = [
        rec(
            "S2",
            "A",
            11,
            3,
            name="click_at",
            arguments={"x": 425, "y": 630},
            action="clicked at (425, 630)",
            cart_after=["blue mug"],
        ),
        rec("S2", "A", 11, 1, status="refused"),
        rec(
            "S2",
            "A",
            11,
            2,
            name="scroll",
            arguments={"direction": "down"},
            action="scrolled down by 600 px",
            cart_after=[],
        ),
    ]
    lines = [
        S2_B11[0],
        S3_B4[0],
        s2a[0],
        S2_B11[1],
        S3_B4[1],
        s2a[1],
        S3_B4[2],
        S2_B11[2],
        s2a[2],
        S3_B4[3],
        S3_B4[4],
    ]
    run = write_run(tmp_path, "t0151-overview-frame-ab-4471255", lines)
    seqs = api("read_sequences")(run)
    got = {tuple(k): [int(r["step"]) for r in v] for k, v in dict(seqs).items()}
    assert got == {
        ("t0151-overview-frame-ab-4471255", "B", "S2", 11): [1, 2],
        ("t0151-overview-frame-ab-4471255", "B", "S3", 4): [1, 2, 3, 4],
        ("t0151-overview-frame-ab-4471255", "A", "S2", 11): [2, 3],
    }, got


def test_read_sequences_keeps_the_logged_fields(tmp_path: Path) -> None:
    """A1-2: the replay reads ``executed_name``, ``executed_arguments`` and ``frame_kind`` (and A1-3 the ``action``
    and ``cart_after``) from the records as logged; an overview-frame step keeps ``frame_kind == "overview"`` and its
    ``scrolled_px``."""
    s5 = rec(
        "S5",
        "B",
        1,
        1,
        name="click_at",
        arguments={"x": 159, "y": 936},
        action="scrolled down by 420 px and clicked at (159, 936)",
        frame_kind="overview",
        scrolled_px=420,
        cart_after=["cotton tea towel"],
    )
    run = write_run(tmp_path, "run-x", [s5, *S3_B4[:2]])
    seqs = dict(api("read_sequences")(run))
    (s5_seq,) = [v for k, v in seqs.items() if tuple(k)[2] == "S5"]
    (r,) = s5_seq
    assert (
        r["executed_name"],
        r["executed_arguments"],
        r["frame_kind"],
        r["scrolled_px"],
        r["action"],
        r["cart_after"],
    ) == (
        "click_at",
        {"x": 159, "y": 936},
        "overview",
        420,
        "scrolled down by 420 px and clicked at (159, 936)",
        ["cotton tea towel"],
    )
    (s3_seq,) = [v for k, v in seqs.items() if tuple(k)[2] == "S3"]
    assert [x["executed_arguments"]["text"] for x in s3_seq] == ["Ada Lovelace", "ada@example.com"]


def test_read_sequences_t0134_shape(tmp_path: Path) -> None:
    """A1-2 on the t0134 shape (O-A1 (4)): records with no ``frame_kind`` and no ``cart_after`` key are kept as
    logged, in step order; the key carries the arm and the run dir name. (t0132, which logs no
    ``executed_arguments``, is dropped by O-A1 (4) and has no row.)"""
    rows = [
        rec(
            "S2",
            "A",
            3,
            2,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
        rec(
            "S2",
            "A",
            3,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
        rec("S2", "A", 3, 3, status=None, frame_kind="absent"),
    ]
    assert all("frame_kind" not in r and "cart_after" not in r for r in rows), (
        "premise: the t0134 shape"
    )
    run = write_run(tmp_path, "t0134-feedback-ab-e59aa69", rows)
    seqs = dict(api("read_sequences")(run))
    got = {tuple(k): [int(r["step"]) for r in v] for k, v in seqs.items()}
    assert got == {("t0134-feedback-ab-e59aa69", "A", "S2", 3): [1, 2]}, got
    (seq,) = seqs.values()
    assert [r["executed_arguments"] for r in seq] == [{"x": 423, "y": 631}] * 2


# =====================================================================================================
# A1-4: pair_id
# =====================================================================================================

SHA_B = "a" * 64
SHA_A = "b" * 64


def test_pair_id_shape_and_determinism() -> None:
    """A1-4: ``pair_id`` is 16 lowercase hex characters and the same for the same four components, whether the
    points are tuples or the JSON lists they come back as (dedupe across a written and a re-read record)."""
    pid = api("pair_id")
    a = pid(SHA_B, SHA_A, (542, 505), None)
    assert isinstance(a, str) and re.fullmatch(r"[0-9a-f]{16}", a), a
    assert pid(SHA_B, SHA_A, (542, 505), None) == a
    assert pid(SHA_B, SHA_A, [542, 505], None) == a, (
        "a JSON list point keys differently from the tuple"
    )
    b = pid(SHA_B, SHA_A, (542, 505), (542, 505))
    assert pid(SHA_B, SHA_A, [542, 505], [542, 505]) == b


def test_pair_id_is_the_same_in_another_process(tmp_path: Path) -> None:
    """A1-4/A1-7 (determinism across runs): ``pair_id`` does not depend on the process (no ``hash()``, no
    ``id()``, no set order): two interpreters with different ``PYTHONHASHSEED`` give this process's value."""
    builder()
    want = api("pair_id")(SHA_B, SHA_A, (542, 505), (100, 200))
    probe = tmp_path / "probe_pair_id.py"
    probe.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('b', {str(BUILDER)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['b'] = m; spec.loader.exec_module(m)\n"
        f"print(m.pair_id({SHA_B!r}, {SHA_A!r}, (542, 505), (100, 200)))\n",
        encoding="utf-8",
    )
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=60,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip() == want, (seed, out.stdout, want)


VARIANTS = {
    "before": ("c" * 64, SHA_A, (542, 505), None),
    "after": (SHA_B, "c" * 64, (542, 505), None),
    "before-after-swapped": (SHA_A, SHA_B, (542, 505), None),
    "point-x": (SHA_B, SHA_A, (543, 505), None),
    "point-y": (SHA_B, SHA_A, (542, 504), None),
    "point-xy-swapped": (SHA_B, SHA_A, (505, 542), None),
    "prev-point-origin": (SHA_B, SHA_A, (542, 505), (0, 0)),
    "prev-point-same": (SHA_B, SHA_A, (542, 505), (542, 505)),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_pair_id_separates_each_component(variant: str) -> None:
    """A1-4: the key covers ``before``, ``after``, ``point`` and ``prev_point``, in order: changing any one of them
    (including ``None`` versus ``(0, 0)`` and swapping ``before`` and ``after``) gives another id. A mutant that
    drops ``prev_point`` would merge a first click with its repeat (S2#11's two identical clicks)."""
    pid = api("pair_id")
    base = pid(SHA_B, SHA_A, (542, 505), None)
    assert pid(*VARIANTS[variant]) != base, variant


def test_pair_id_has_no_delimiter_collision() -> None:
    """A1-4 (adversarial): a serialisation that concatenates digits without separators collides ``(1, 23)`` with
    ``(12, 3)`` and ``prev_point (12, 3)`` with a point boundary; the key must not."""
    pid = api("pair_id")
    assert pid(SHA_B, SHA_A, (1, 23), None) != pid(SHA_B, SHA_A, (12, 3), None)
    assert pid(SHA_B, SHA_A, (1, 2), (34, 5)) != pid(SHA_B, SHA_A, (1, 23), (4, 5))


# =====================================================================================================
# A1-3: the reproduction check and truncation at the first divergence
# =====================================================================================================


def rp(action: str, cart: Any = None) -> dict[str, Any]:
    return {"action": action, "cart_after": cart}


TRUNCATION = {
    "all-reproduced": (S2_B11[:2], [rp(C1, ["blue mug"]), rp(C1, ["blue mug", "blue mug"])], None),
    "action-differs-at-1": (
        S2_B11[:2],
        [rp(C1, ["blue mug"]), rp("clicked at (423, 632)", ["blue mug", "blue mug"])],
        1,
    ),
    "cart-differs-at-0": (S2_B11[:2], [rp(C1, []), rp(C1, ["blue mug"])], 0),
    # the replay's cart is a strict extension of the logged one: not equal (a prefix/startswith mutant passes it)
    "cart-longer-at-0": (
        S2_B11[:2],
        [rp(C1, ["blue mug", "blue mug"]), rp(C1, ["blue mug", "blue mug"])],
        0,
    ),
    "cart-order": (
        [
            rec(
                "S1",
                "A",
                0,
                1,
                name="click_at",
                arguments={"x": 1, "y": 1},
                action="clicked at (1, 1)",
                cart_after=["a", "b"],
            )
        ],
        [rp("clicked at (1, 1)", ["b", "a"])],
        0,
    ),
    # adversarial (the task's mutant): steps after the first divergence reproduce again; the answer is still 1
    "reproduces-again-after": (
        [
            *S2_B11[:2],
            rec(
                "S2",
                "B",
                11,
                3,
                name="click_at",
                arguments={"x": 423, "y": 631},
                action=C1,
                cart_after=["blue mug", "blue mug", "blue mug"],
            ),
        ],
        [rp(C1, ["blue mug"]), rp(C1, ["blue mug"]), rp(C1, ["blue mug", "blue mug", "blue mug"])],
        1,
    ),
    # "when logged": t0134's records carry no cart_after, so only the action is compared
    "cart-not-logged": (
        [rec("S2", "A", 3, 1, name="click_at", arguments={"x": 423, "y": 631}, action=C1)],
        [rp(C1, ["blue mug", "blue mug"])],
        None,
    ),
    # a page without a cart: logged None and replayed None are equal
    "cart-none-both": (S3_B4[:1], [rp(S3_B4[0]["action"], None)], None),
    # the replay stopped after step 1 (the browser failed, say): step 2 was not reproduced
    "replay-shorter": (S2_B11[:2], [rp(C1, ["blue mug"])], 1),
}


@pytest.mark.parametrize("case", list(TRUNCATION))
def test_first_divergence(case: str) -> None:
    """A1-3: a step is reproduced when the replay's action string equals the logged ``action`` and its
    ``cart_after`` equals the logged one when logged; the sequence diverges at the first step that is not."""
    logged, replayed, want = TRUNCATION[case]
    got = api("first_divergence")(logged, replayed)
    assert got == want, (case, got, want)


# =====================================================================================================
# O-A1 (7): the max |Δ| recorded between two builds
# =====================================================================================================


def _jpeg(arr: Any) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").convert("RGB").save(
        buf, format="JPEG", quality=90
    )
    return buf.getvalue()


def _base_frames() -> tuple[Any, Any]:
    """A flat 128 grey 1280x800 frame, and the same with a black 200x100 block at (100, 100): flat areas so that
    the JPEG round trip is exact to a few grey levels, and the block's edges are the only detail."""
    import numpy as np

    before = np.full((800, 1280), 128, dtype=np.int16)
    after = before.copy()
    after[100:200, 100:300] = 0
    return before, after


def _fake_build(out: Path, pairs_spec: Sequence[tuple[Any, Any, tuple[int, int]]]) -> Path:
    """An out dir in the pinned layout: frames by sha256, ``pairs.jsonl`` with every A1-4 field, a manifest."""
    (out / "frames").mkdir(parents=True)
    rows = []
    prev = None
    for i, (b, a, point) in enumerate(pairs_spec):
        shas = []
        for img in (b, a):
            data = _jpeg(img)
            sha = hashlib.sha256(data).hexdigest()
            (out / "frames" / f"{sha}.jpg").write_bytes(data)
            shas.append(sha)
        rows.append(
            {
                "pair_id": api("pair_id")(shas[0], shas[1], point, prev),
                "before": shas[0],
                "after": shas[1],
                "point": list(point),
                "prev_point": None if prev is None else list(prev),
                "frame_kind": None,
                "scrolled_px": None,
                "occurrences": 1,
                "run": "t0151-overview-frame-ab-4471255",
                "arm": "B",
                "task": "S2",
                "repeat": 11,
                "step": i + 1,
                "oracle": {
                    "cart_before": [],
                    "cart_after": ["blue mug"],
                    "scroll_y_before": 0,
                    "scroll_y_after": 0,
                    "url_changed": False,
                },
            }
        )
        prev = point
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "sequences": 1,
                "reproduced_steps": len(rows),
                "truncated_sequences": 0,
                "pairs": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return out


ORACLE_1 = {
    "cart_before": [],
    "cart_after": ["blue mug"],
    "scroll_y_before": 0,
    "scroll_y_after": 0,
    "url_changed": False,
}
SRC = ("t0151-overview-frame-ab-4471255", "B", "S2")


def _png(arr: Any) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _fake_build_records(out: Path, specs: Sequence[Mapping[str, Any]]) -> Path:
    """An out dir from explicit record specs: ``before``/``after`` arrays, ``point``, ``prev`` (default None),
    ``sources`` (a list of ``(run, arm, task, repeat, step)``, default one), ``oracle`` (default ``ORACLE_1``),
    ``fmt`` (``"jpeg"``, or ``"png"`` for exact pixel values). ``occurrences`` is ``len(sources)``; the record's
    own source fields are its first source's, as one build would write a deduped record."""
    (out / "frames").mkdir(parents=True)
    rows = []
    for s in specs:
        enc = _png if s.get("fmt") == "png" else _jpeg
        ext = "png" if s.get("fmt") == "png" else "jpg"
        shas = []
        for img in (s["before"], s["after"]):
            data = enc(img)
            sha = hashlib.sha256(data).hexdigest()
            (out / "frames" / f"{sha}.{ext}").write_bytes(data)
            shas.append(sha)
        sources = [list(x) for x in s.get("sources", [(*SRC, 11, 1)])]
        prev = s.get("prev")
        rows.append(
            {
                "pair_id": api("pair_id")(shas[0], shas[1], s["point"], prev),
                "before": shas[0],
                "after": shas[1],
                "point": list(s["point"]),
                "prev_point": None if prev is None else list(prev),
                "frame_kind": None,
                "scrolled_px": None,
                "occurrences": len(sources),
                "run": sources[0][0],
                "arm": sources[0][1],
                "task": sources[0][2],
                "repeat": sources[0][3],
                "step": sources[0][4],
                "sources": sources,
                "oracle": dict(s.get("oracle", ORACLE_1)),
            }
        )
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    n = sum(r["occurrences"] for r in rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {"sequences": 1, "reproduced_steps": n, "truncated_sequences": 0, "pairs": len(rows)}
        ),
        encoding="utf-8",
    )
    return out


def test_compare_builds_sees_a_change_in_before_only(tmp_path: Path) -> None:
    """Review gap 1a (kills ``cb-ignore-before``): build b's ``before`` has a 200x100 block 100 grey levels brighter,
    its ``after`` is a's: the recorded max |Δ| and count over 24 are the ``before`` frames' (independent decode)."""
    before, after = _base_frames()
    changed = before.copy()
    changed[500:600, 800:1000] = 228
    a = _fake_build_records(
        tmp_path / "a", [{"before": before, "after": after, "point": (542, 505)}]
    )
    b = _fake_build_records(
        tmp_path / "b", [{"before": changed, "after": after, "point": (542, 505)}]
    )
    want_max, want_over = _decoded_max(before, changed)
    assert want_max >= 90 and want_over >= 18_000, (want_max, want_over)
    got = dict(api("compare_builds")(a, b))
    assert (got["structure_equal"], got["max_abs_delta"], got["pixels_over_24"]) == (
        True,
        want_max,
        want_over,
    ), (
        got,
        want_max,
        want_over,
    )


def test_compare_builds_oracle_is_structure(tmp_path: Path) -> None:
    """Review gap 1b (kills ``cb-ignore-oracle``): builds identical except one occurrence's oracle block (the cart
    after the click) are not structurally equal; the companion with equal oracles is."""
    before, after = _base_frames()
    spec = {"before": before, "after": after, "point": (542, 505)}
    a = _fake_build_records(tmp_path / "a", [spec])
    b = _fake_build_records(
        tmp_path / "b", [dict(spec, oracle=dict(ORACLE_1, cart_after=["red mug"]))]
    )
    c = _fake_build_records(tmp_path / "c", [spec])
    assert dict(api("compare_builds")(a, b))["structure_equal"] is False
    assert dict(api("compare_builds")(a, c))["structure_equal"] is True


def test_compare_builds_matches_per_occurrence(tmp_path: Path) -> None:
    """Review gap 1c (kills ``cb-per-record``; O-A1b (4)): the same two source occurrences, deduped in build b only
    (one record with ``occurrences == 2``) and not in build a (two records, their frames within JPEG noise of each
    other), are structurally equal, and the recorded max |Δ| is the noise (at most 8, none over 24)."""
    before, after = _base_frames()
    s1, s2 = (*SRC, 0, 1), (*SRC, 3, 1)
    a = _fake_build_records(
        tmp_path / "a",
        [
            {"before": before, "after": after, "point": (542, 505), "sources": [s1]},
            {"before": before + 3, "after": after + 3, "point": (542, 505), "sources": [s2]},
        ],
    )
    b = _fake_build_records(
        tmp_path / "b",
        [{"before": before, "after": after, "point": (542, 505), "sources": [s1, s2]}],
    )
    got = dict(api("compare_builds")(a, b))
    want = max(_decoded_max(before, before + 3)[0], _decoded_max(after, after + 3)[0])
    assert 1 <= want <= 8, want
    assert (got["structure_equal"], got["max_abs_delta"], got["pixels_over_24"]) == (
        True,
        want,
        0,
    ), (got, want)


def test_compare_builds_superset_is_not_equal(tmp_path: Path) -> None:
    """Review gap 1d (kills ``cb-struct-keys-only``): build b holds every pair of a plus one more; not equal, in
    either order."""
    before, after = _base_frames()
    p1 = {"before": before, "after": after, "point": (542, 505)}
    p2 = {
        "before": after,
        "after": after,
        "point": (542, 505),
        "prev": (542, 505),
        "sources": [(*SRC, 11, 2)],
    }
    a = _fake_build_records(tmp_path / "a", [p1])
    b = _fake_build_records(tmp_path / "b", [p1, p2])
    assert dict(api("compare_builds")(a, b))["structure_equal"] is False
    assert dict(api("compare_builds")(b, a))["structure_equal"] is False


@pytest.mark.parametrize(("delta", "over"), [(24, 0), (25, 1)])
def test_compare_builds_over_24_boundary(tmp_path: Path, delta: int, over: int) -> None:
    """Review gap 1e (kills ``cb-over-ge`` and ``cb-over-25``): lossless frames (PNG, so the pixel values are exact)
    that differ in one pixel by exactly ``delta``: ``max_abs_delta == delta`` and ``pixels_over_24`` is 0 at 24
    ("over 24" is strictly greater) and 1 at 25."""
    import numpy as np

    flat = np.full((800, 1280), 100, dtype=np.int16)
    bumped = flat.copy()
    bumped[400, 640] = 100 + delta
    a = _fake_build_records(
        tmp_path / "a", [{"before": flat, "after": flat, "point": (542, 505), "fmt": "png"}]
    )
    b = _fake_build_records(
        tmp_path / "b", [{"before": flat, "after": bumped, "point": (542, 505), "fmt": "png"}]
    )
    got = dict(api("compare_builds")(a, b))
    assert (got["structure_equal"], got["max_abs_delta"], got["pixels_over_24"]) == (
        True,
        delta,
        over,
    ), got


def _decoded_max(a: Any, b: Any) -> tuple[int, int]:
    """Independent oracle: decode both JPEGs as the builds store them and take max |Δ| and the count over 24."""
    import numpy as np
    from PIL import Image

    def g(x: Any) -> Any:
        with Image.open(io.BytesIO(_jpeg(x))) as im:
            return np.asarray(im.convert("L"), dtype=np.int16)

    d = np.abs(g(a) - g(b))
    return int(d.max()), int((d > 24).sum())


def test_compare_builds_identical_builds(tmp_path: Path) -> None:
    """O-A1 (7): two builds with the same frames and structure: structure equal, max |Δ| 0, no pixel over 24."""
    before, after = _base_frames()
    spec = [(before, after, (542, 505)), (after, after, (542, 505))]
    a = _fake_build(tmp_path / "a", spec)
    b = _fake_build(tmp_path / "b", spec)
    got = dict(api("compare_builds")(a, b))
    assert (got["structure_equal"], got["max_abs_delta"], got["pixels_over_24"]) == (True, 0, 0), (
        got
    )


def test_compare_builds_reports_jpeg_noise(tmp_path: Path) -> None:
    """O-A1 (7): build b's frames are build a's plus a uniform +3 grey (JPEG-noise scale): the structure is equal,
    the recorded max |Δ| is the decoded one (by an independent decode here, a few grey levels, at most 8), and no
    pixel is over 24. A recorder that compares bytes (any difference = 255) or reports 0 fails."""
    before, after = _base_frames()
    a = _fake_build(tmp_path / "a", [(before, after, (542, 505))])
    b = _fake_build(tmp_path / "b", [(before + 3, after + 3, (542, 505))])
    want = max(_decoded_max(before, before + 3)[0], _decoded_max(after, after + 3)[0])
    assert 1 <= want <= 8, f"premise: +3 survives the JPEG round trip as 1..8 grey levels: {want}"
    got = dict(api("compare_builds")(a, b))
    assert (got["structure_equal"], got["max_abs_delta"], got["pixels_over_24"]) == (
        True,
        want,
        0,
    ), (got, want)


def test_compare_builds_reports_a_large_change(tmp_path: Path) -> None:
    """O-A1 (7), adversarial (the mutant lets frames differ by a large |Δ|): build b's ``after`` has a 200x100 block
    100 grey levels brighter. The recorded max |Δ| is the decoded one (about 100, far above 8) and the count over
    24 covers the block (at least 90 % of its 20 000 pixels): a compare that ignores ``after`` or caps |Δ| fails."""
    before, after = _base_frames()
    changed = after.copy()
    changed[500:600, 800:1000] = 228
    a = _fake_build(tmp_path / "a", [(before, after, (542, 505))])
    b = _fake_build(tmp_path / "b", [(before, changed, (542, 505))])
    want_max, want_over = _decoded_max(after, changed)
    assert want_max >= 90 and want_over >= 18_000, (want_max, want_over)
    got = dict(api("compare_builds")(a, b))
    assert got["structure_equal"] is True, got
    assert (got["max_abs_delta"], got["pixels_over_24"]) == (want_max, want_over), (
        got,
        want_max,
        want_over,
    )


def test_compare_builds_structure_differs(tmp_path: Path) -> None:
    """O-A1 (7): a build missing the second pair, or with a pair at another point, is not structurally equal
    (whatever its frames)."""
    before, after = _base_frames()
    a = _fake_build(tmp_path / "a", [(before, after, (542, 505)), (after, after, (542, 505))])
    b = _fake_build(tmp_path / "b", [(before, after, (542, 505))])
    c = _fake_build(tmp_path / "c", [(before, after, (542, 505)), (after, after, (543, 505))])
    assert dict(api("compare_builds")(a, b))["structure_equal"] is False
    assert dict(api("compare_builds")(a, c))["structure_equal"] is False


# =====================================================================================================
# A1-1: the refusals, before any browser starts
# =====================================================================================================


class _Spawned(AssertionError):
    pass


def _spy_processes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record and refuse every process start (sync and asyncio); returns the list the spy appends argv to."""
    started: list[str] = []

    def refuse(where: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            argv = " ".join(
                str(a) for a in args if isinstance(a, (str, bytes, os.PathLike, list, tuple))
            )
            started.append(f"{where}: {argv}")
            raise _Spawned(f"T-0136 refusal rig: no process may start here ({where}: {argv[:200]})")

        return fn

    monkeypatch.setattr(subprocess.Popen, "__init__", refuse("subprocess.Popen"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse("asyncio.create_subprocess_exec"))
    monkeypatch.setattr(
        asyncio.subprocess, "create_subprocess_exec", refuse("asyncio.create_subprocess_exec")
    )
    monkeypatch.setattr(
        asyncio, "create_subprocess_shell", refuse("asyncio.create_subprocess_shell")
    )
    monkeypatch.setattr(
        asyncio.base_events.BaseEventLoop, "subprocess_exec", refuse("loop.subprocess_exec")
    )
    monkeypatch.setattr(
        asyncio.base_events.BaseEventLoop, "subprocess_shell", refuse("loop.subprocess_shell")
    )
    return started


def _fake_git(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, porcelain: str, sha: str
) -> None:
    assert callable(getattr(module, "_git", None)), (
        f"T-0136 A1-1: {BUILDER.name} has no module-level _git (T-0111g's seam; see this module's docstring)"
    )

    def git(*args: str) -> str:
        if args[:1] == ("status",):
            return porcelain
        if args[:1] == ("rev-parse",):
            return (sha[:7] if any(a.startswith("--short") for a in args) else sha) + "\n"
        raise AssertionError(f"T-0136 refusal rig: unexpected git call {args}")

    monkeypatch.setattr(module, "_git", git)


@pytest.fixture
def fake_sha() -> Iterator[str]:
    """A fresh 40-hex sha per row, so the corpus dir ``logs/t0136-outcome-corpus-<sha7>`` of a row never names a real
    run; the dir is removed afterwards if a row made it."""
    sha = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    yield sha
    shutil.rmtree(REPO / "logs" / f"t0136-outcome-corpus-{sha[:7]}", ignore_errors=True)


def _run_main(module: ModuleType, argv: list[str]) -> tuple[int | None, BaseException | None]:
    try:
        return int(api("main", module)(argv)), None
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), None
    except BaseException as exc:  # the spy's refusal surfacing from inside main
        return None, exc


REFUSAL_CASES = (
    "dirty-tree",
    "dirty-tree-untracked",
    "out-outside-repo",
    "out-in-a-foreign-logs-dir",
    "out-escapes-logs",
    "out-sibling-of-logs",
    "out-is-logs-parent",
    "out-wrong-name-under-logs",
    "out-other-sha7-under-logs",
    "out-nested-under-the-right-name",
    "browser-extra-missing",
)
REFUSAL_WORDS = {
    "dirty-tree": r"dirty|clean|uncommitted|commit",
    "dirty-tree-untracked": r"dirty|clean|uncommitted|commit|untracked",
    "out-outside-repo": r"logs",
    "out-in-a-foreign-logs-dir": r"logs",
    "out-escapes-logs": r"logs",
    "out-sibling-of-logs": r"logs",
    "out-is-logs-parent": r"logs",
    "out-wrong-name-under-logs": r"t0136-outcome-corpus",
    "out-other-sha7-under-logs": r"t0136-outcome-corpus",
    "out-nested-under-the-right-name": r"t0136-outcome-corpus",
    "browser-extra-missing": r"browser|playwright",
}


@pytest.mark.parametrize("case", REFUSAL_CASES)
def test_refusals_exit_2_before_any_browser(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_sha: str,
) -> None:
    """A1-1: a dirty tree (a modified file; an untracked one), an out dir not under the repository's ``logs/`` (a
    tmp dir; a directory *named* ``logs`` elsewhere, which a ``"logs" in parts`` check accepts; ``logs/../x``, which
    an unresolved prefix check accepts; ``logs-other/``, which a string ``startswith`` accepts; the repo root), and
    a missing ``browser`` extra (``playwright`` not importable) each exit 2 before any process starts, leave the out
    dir uncreated, and say why on stderr."""
    run = write_run(tmp_path, "t0151-overview-frame-ab-4471255", S2_B11)
    good_out = REPO / "logs" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    out = good_out
    porcelain = ""
    if case == "browser-extra-missing":
        for name in ("playwright", "playwright.async_api", "playwright.sync_api"):
            monkeypatch.setitem(sys.modules, name, None)
        module = _load("_t0136_build_outcome_corpus_no_playwright")
    else:
        module = builder()
    if case == "dirty-tree":
        porcelain = " M local_astra/conductor/session.py\n"
    elif case == "dirty-tree-untracked":
        porcelain = "?? scripts/stray.py\n"
    elif case == "out-outside-repo":
        out = tmp_path / f"t0136-outcome-corpus-{fake_sha[:7]}"
    elif case == "out-in-a-foreign-logs-dir":
        out = tmp_path / "logs" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    elif case == "out-escapes-logs":
        out = REPO / "logs" / ".." / f"t0136-outcome-corpus-{fake_sha[:7]}"
    elif case == "out-sibling-of-logs":
        out = REPO / f"logs-{fake_sha[:7]}" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    elif case == "out-is-logs-parent":
        out = REPO
    elif case == "out-wrong-name-under-logs":
        out = REPO / "logs" / f"t0136-corpus-{fake_sha[:7]}"
    elif case == "out-other-sha7-under-logs":
        other = "0" * 7 if fake_sha[:7] != "0" * 7 else "1" * 7
        out = REPO / "logs" / f"t0136-outcome-corpus-{other}"
    elif case == "out-nested-under-the-right-name":
        out = good_out / "inner"
    _fake_git(module, monkeypatch, porcelain, fake_sha)
    started = _spy_processes(monkeypatch)
    existed = out.exists()
    rc, exc = _run_main(module, [str(run), "--out", str(out)])
    err = capsys.readouterr().err
    assert exc is None, f"{case}: main raised instead of refusing: {type(exc).__name__}: {exc}"
    assert started == [], f"{case}: a process started before the refusal: {started}"
    assert rc == 2 and api("EXIT_REFUSED", module) == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert out.exists() == existed, f"{case}: the refusal created {out}"
    assert not (REPO / f"logs-{fake_sha[:7]}").exists(), f"{case}: created a logs- sibling"
    assert not good_out.exists(), f"{case}: the refusal created the corpus dir {good_out}"
    assert re.search(REFUSAL_WORDS[case], err, flags=re.I), (
        f"{case}: the refusal does not say why: {err!r}"
    )


def test_passing_values_get_past_the_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_sha: str,
) -> None:
    """A1-1, the other side of every refusal row (a guard too wide): a clean tree, the out dir
    ``logs/t0136-outcome-corpus-<sha7>`` under the repository and ``playwright`` importable is *not* refused: the
    builder goes on to start the browser (here: the spy refuses the process, so main fails or raises, but not with
    exit 2 before trying)."""
    run = write_run(tmp_path, "t0151-overview-frame-ab-4471255", S2_B11)
    module = builder()
    _fake_git(module, monkeypatch, "", fake_sha)
    started = _spy_processes(monkeypatch)
    out = REPO / "logs" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    rc, exc = _run_main(module, [str(run), "--out", str(out)])
    err = capsys.readouterr().err
    assert not (rc == 2 and not started), (
        f"the passing set-up was refused: exit {rc}\n{err[-2000:]}"
    )
    assert started, (
        f"the builder never tried to start the browser (exit {rc}, {exc!r})\n{err[-2000:]}"
    )
    assert any(re.search(r"playwright|node|chrom", s, flags=re.I) for s in started), started


# =====================================================================================================
# A1-5: the probe fixture (green at BASE_A1: the fixture is the test-author's)
# =====================================================================================================


def probe_table() -> list[dict[str, Any]]:
    text = re.sub(r"<!--.*?-->", "", PROBE_PAGE.read_text(encoding="utf-8"), flags=re.S)
    m = re.search(r'<script type="application/json" id="probes">(.*?)</script>', text, flags=re.S)
    assert m is not None, "the probe table is missing from outcomes.html"
    table = json.loads(m.group(1))
    assert isinstance(table, list)
    return table


def _px(v: int, size: int) -> int:
    """``round(v * (size - 1) / 999)`` with halves up, in exact arithmetic (the executor's documented mapping)."""
    q = Fraction(v * (size - 1), 999)
    return int(q + Fraction(1, 2)) if q >= 0 else -int(-q + Fraction(1, 2))


def test_probe_table_has_eight_per_class_and_the_caret() -> None:
    """A1-5: at least 8 probes per intended class (none: blank, static text, hover-only; near: checkbox, own-label
    button, typed input; elsewhere: toast at the bottom, badge at the top right; page: a navigating link), plus
    exactly one caret-blink probe (530 ms); ids and points are unique; every pixel point is the model point mapped
    by hand at 1280x800."""
    table = probe_table()
    plain = [p for p in table if p["tool"] != "caret"]
    by_class = {c: [p for p in plain if p["intended"] == c] for c in CLASSES}
    assert {c: len(v) >= 8 for c, v in by_class.items()} == dict.fromkeys(CLASSES, True), {
        c: len(v) for c, v in by_class.items()
    }
    assert {p["intended"] for p in plain} == set(CLASSES)
    kinds = {c: {p["kind"] for p in v} for c, v in by_class.items()}
    assert kinds["none"] >= {"blank", "text", "hover"}, kinds
    assert kinds["near"] >= {"checkbox", "label", "typed"}, kinds
    assert kinds["elsewhere"] >= {"toast", "badge"}, kinds
    assert kinds["page"] == {"link"}, kinds
    carets = [p for p in table if p["tool"] == "caret"]
    assert len(carets) == 1 and carets[0]["wait_ms"] == 530, carets
    assert len({p["id"] for p in table}) == len(table)
    assert len({(p["x_px"], p["y_px"]) for p in table}) == len(table), "two probes share a point"
    for p in table:
        assert p["tool"] in ("click_at", "type_at", "caret"), p
        assert 0 <= p["x"] <= 999 and 0 <= p["y"] <= 999, p
        assert (p["x_px"], p["y_px"]) == (_px(p["x"], 1280), _px(p["y"], 800)), p
        if p["tool"] == "type_at":
            assert isinstance(p["text"], str) and p["text"] and p["press_enter"] is False, p
    assert {p["kind"] for p in plain if p["tool"] == "type_at"} == {"typed"}


#: sha256 of the canonical JSON (sort_keys, compact) of the A1 probe table's 36 entries at 2688714 (the corpus run's
#: fixture): the A1b follow-up appends probes and must leave these, hence their pair ids, as they were.
A1_TABLE_SHA = "b4ad3949fd1e625bdb1dee57fd6375e5b006e0275a95515f5a5fd40220c2770d"
#: sha256 of outcomes.html at 2688714 with comments and the probe table's contents removed: everything that renders.
A1_BODY_SHA = "236b12d4e7dea35bb119483d0f9272f35840016878b33205f3a4d7cca58b149b"
#: O-A2 / Q-1: the follow-up adds at least 72 ``none`` probes to A1's 9 (8 plus none of the caret).
A1B_NONE_FLOOR = 9 + 72


def _ninth(x: int, y: int) -> int:
    return (0 if x < 427 else 1 if x < 854 else 2) + 3 * (0 if y < 267 else 1 if y < 534 else 2)


def test_probe_table_a1_probes_unchanged() -> None:
    """A1b (O-A2): the first 36 entries of the table are A1's, byte for byte in canonical JSON, so the corpus run's
    pair ids for them stand; and the rendered page is A1's (only the heading gained ``id="title"``, which no CSS rule
    selects): the page with comments and the probe table's contents removed, and that id taken out, hashes to A1's."""
    table = probe_table()
    blob = json.dumps(table[:36], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == A1_TABLE_SHA, "an A1 probe changed or moved"
    text = re.sub(r"<!--.*?-->", "", PROBE_PAGE.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(
        r'(<script type="application/json" id="probes">).*?(</script>)', r"\1\2", text, flags=re.S
    )
    assert "#title" not in text, (
        "a CSS rule or script selects the new heading id (it would change pixels)"
    )
    body = text.replace('<h1 id="title">', "<h1>")
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == A1_BODY_SHA, (
        "the rendered page changed"
    )


def test_probe_table_has_the_a1b_none_probes() -> None:
    """A1b (O-A2 / Q-1): at least 81 ``none`` probes (A1's 9 plus at least 72), every one a ``click_at`` (the caret
    stays the only caret), of the kinds blank, heading, static text and hover-only; at least 4 ``none`` probes in each
    of the nine ninths of 1280x800 (so a split by pair-id parity cannot leave a region empty); each non-blank ``none``
    probe names its target and each blank one names none."""
    table = probe_table()
    none = [p for p in table if p["intended"] == "none" and p["tool"] != "caret"]
    assert len(none) >= A1B_NONE_FLOOR, len(none)
    assert all(p["tool"] == "click_at" for p in none), [
        p["id"] for p in none if p["tool"] != "click_at"
    ]
    assert {p["kind"] for p in none} >= {"blank", "heading", "text", "hover"}, {
        p["kind"] for p in none
    }
    per = {k: 0 for k in range(9)}
    for p in none:
        per[_ninth(p["x_px"], p["y_px"])] += 1
    assert min(per.values()) >= 4, per
    for p in none:
        assert (p["target"] is None) is (p["kind"] == "blank"), p
    assert sum(p["tool"] == "caret" for p in table) == 1


def test_probe_elsewhere_triggers_sit_outside_the_near_box() -> None:
    """A1-5 (the fixture's own geometry, from its CSS by hand): every ``elsewhere`` trigger is in the left column
    (x <= 230), and what it changes (the toast at x 540..840 / 1000..1260, y 730..774; the badge at x 1060..1260,
    y 16..56; the status chip at x 1080..1260, y 450..490) is outside A2-3's +-320 x +-80 px near box of the click
    with at least 64 px to spare horizontally, so a 64 px cell grid cannot straddle the two."""
    boxes = {
        "#toast": (540, 730, 840, 774),
        "#toast-br": (1000, 730, 1260, 774),
        "#badge": (1060, 16, 1260, 56),
        "#status": (1080, 450, 1260, 490),
    }
    text = PROBE_PAGE.read_text(encoding="utf-8")
    # premise: the boxes above are the fixture's CSS (left/width from the id rule, top/height from the shared one)
    for rule in (
        "#toast { left: 540px; width: 300px; }",
        "#toast-br { left: 1000px; width: 260px; }",
        "display: none; top: 730px; height: 44px;",
        "left: 1060px; top: 16px; width: 200px; height: 40px;",
        "left: 1080px; top: 450px; width: 180px; height: 40px;",
        ".left { left: 30px; width: 200px;",
    ):
        assert rule in text, f"premise: outcomes.html no longer has {rule!r}"
    for p in probe_table():
        if p["intended"] != "elsewhere":
            continue
        assert p["x_px"] <= 230, p
        x0, y0, _x1, y1 = boxes[p["changes"]]
        nx1, ny0, ny1 = p["x_px"] + 320, p["y_px"] - 80, p["y_px"] + 80
        gap_x = x0 - nx1  # every trigger is left of what it changes
        gap_y = max(y0 - ny1, ny0 - y1)
        assert gap_x >= 64 or gap_y >= 64, (p["id"], gap_x, gap_y)


def test_probe_page_is_deterministic_and_offline() -> None:
    """A1-5's fixture rules: no timers, no animation (the only ``animation`` text is the neutraliser), no network
    API, no absolute or protocol-relative URL, no external resource; the only navigation is the relative
    ``?page=k`` link (8 of them)."""
    text = PROBE_PAGE.read_text(encoding="utf-8")
    code = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for word in (
        "setTimeout",
        "setInterval",
        "requestAnimationFrame",
        "requestIdleCallback",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "sendBeacon",
        "navigator.",
        "import(",
        "@keyframes",
        "@import",
        "url(",
        "<img",
        "<link",
        "<iframe",
        "src=",
    ):
        assert word not in code, f"outcomes.html uses {word!r}"
    assert not re.search(r"(https?:)?//[A-Za-z0-9]", code), "outcomes.html names a URL"
    assert re.findall(r"animation:\s*([^;]+);", code) == ["none !important"], (
        "an animation other than none"
    )
    assert re.findall(r'href="([^"]*)"', code) == [f"?page={k}" for k in range(1, 9)]


# =====================================================================================================
# A1-8: negative (green at BASE_A1)
# =====================================================================================================


def test_a1_changes_no_local_astra_and_no_existing_script() -> None:
    """A1-8: over ``BASE_A1..TIP_A1`` (two literals, never the working tree) no ``local_astra/`` file changes and no
    existing ``scripts/`` file is modified, deleted or renamed (the new builder is an addition). Green at BASE_A1
    (an empty range until ``TIP_A1`` is bumped to the implementation's last commit)."""
    gr.require_commits(BASE_A1, TIP_A1)
    assert gr.is_ancestor(BASE_A1, TIP_A1), (BASE_A1, TIP_A1)
    assert gr.changed(BASE_A1, TIP_A1, "local_astra") == []
    assert gr.changed(BASE_A1, TIP_A1, "scripts", diff_filter="a") == []
    added = gr.changed(BASE_A1, TIP_A1, "scripts", diff_filter="A")
    assert set(added) <= {"scripts/build_outcome_corpus.py"}, added


def test_a1_scope_predicate_has_power() -> None:
    """The A1-8 predicates see a change when there is one: T-0151 A1 (cad8538) changed ``local_astra/``, and the
    T-0140 B harness revision (d3f6b42) modified an existing ``scripts/`` file."""
    gr.require_commits("d49fa5b", "cad8538", "d3f6b42")
    assert gr.changed("d49fa5b", "cad8538", "local_astra") != []
    assert gr.changed("d3f6b42^", "d3f6b42", "scripts", diff_filter="a") != []


# =====================================================================================================
# A1c: the second probe page, the builder's several pages and frame groups (docs/tasks/T-0136.md, "Amendment
# A1c/A2b"). Fixture, count and scope rows hold at BASE_A1c (not pending); builder rows are pending.
# =====================================================================================================

PROBE_PAGE_B = REPO / "tests" / "fixtures" / "t0136" / "outcomes_b.html"
#: The commit group A1c branched from (main at "T-0136: amendment A1c/A2b planned ..."). A literal.
BASE_A1C = "d4adf4f"
#: The A1c implementation tip for the A1c-N scope row; starts at BASE_A1C (an empty range) and is bumped by the
#: test-author or the orchestrator to the last implementation commit before review (the implementer never edits tests).
TIP_A1C = "01d47f6"
#: A1c-N: sha256 of tests/fixtures/t0136/outcomes.html as A1b left it (7eb17f6), the page every A1/A1b pair id and
#: probe frame was built from. Measured by ``sha256sum`` at d4adf4f.
OUTCOMES_A_SHA256 = "9d10ee58f2f5c4ba8fc396ade522c134099d7a82b051138e4057fae1eb1dbd8b"
NEAR_HW, NEAR_HH = 320, 80  # A2-3 rule 5 at 1280x800: +-0.25 W x +-0.10 H, definitional
MARGIN = 64  # A1c-2: elsewhere rects lie outside the near box by at least this much
GATED_NONE_KINDS = ("n-ul", "n-btn", "n-tint", "n-pre")
TRIVIAL_KINDS = ("t-disabled", "t-image", "t-selected")
C_SMALL_KINDS = tuple(f"c-small-{k}" for k in ("qty", "count", "flash", "hint", "price", "glyph"))
GATED_NEAR_KINDS = (*C_SMALL_KINDS, "c-colour", "c-relabel", "c-focus")
STRESS_KINDS = ("s-shadow", "s-lift", "s-ghost", "s-strike", "s-luma", "s-row")


def _table_of(path: Path) -> list[dict[str, Any]]:
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    m = re.search(r'<script type="application/json" id="probes">(.*?)</script>', text, flags=re.S)
    assert m is not None, f"the probe table is missing from {path.name}"
    table = json.loads(m.group(1))
    assert isinstance(table, list)
    return table


def probe_table_b() -> list[dict[str, Any]]:
    """The probe table of ``outcomes_b.html``, parsed here (independently of the builder)."""
    return _table_of(PROBE_PAGE_B)


def _page_b_html() -> str:
    return PROBE_PAGE_B.read_text(encoding="utf-8")


def _element(html: str, el_id: str) -> tuple[str, str]:
    """``(tag, attributes)`` of the element with this id in the page's markup (by regex, test-side)."""
    m = re.search(rf'<(\w+) id="{re.escape(el_id)}"([^>]*)>', html)
    assert m is not None, f"outcomes_b.html has no element #{el_id}"
    return m.group(1), m.group(2)


def _box(attrs: str) -> tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` from an inline ``left/top/width/height`` style, in px."""
    m = re.search(r"left: (\d+)px; top: (\d+)px; width: (\d+)px; height: (\d+)px;", attrs)
    assert m is not None, attrs
    x, y, w, h = (int(v) for v in m.groups())
    return x, y, x + w, y + h


def _gray_of_hex(c: str) -> Fraction:
    """PIL's ``L`` conversion (ITU-R 601-2 luma, ``L = R * 299/1000 + G * 587/1000 + B * 114/1000``), exactly."""
    r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
    return Fraction(r * 299 + g * 587 + b * 114, 1000)


def _ninth_of_rect(rect: Sequence[int]) -> int | None:
    """The ninth of 1280x800 that wholly holds ``[left, top, right, bottom)``, or ``None`` when it straddles."""
    xs, ys = (0, 427, 854, 1280), (0, 267, 534, 800)
    x0, y0, x1, y1 = rect
    cols = [i for i in range(3) if xs[i] <= x0 and x1 <= xs[i + 1]]
    rows = [j for j in range(3) if ys[j] <= y0 and y1 <= ys[j + 1]]
    return (cols[0] + 3 * rows[0]) if cols and rows else None


def test_outcomes_html_byte_identical() -> None:
    """A1c-N: ``outcomes.html`` is byte-identical to its A1b version (sha256 literal), so the A1/A1b probe frames and
    pair ids hold. (A content pin, not a range: it must hold after every later merge too.)"""
    got = hashlib.sha256(PROBE_PAGE.read_bytes()).hexdigest()
    assert got == OUTCOMES_A_SHA256, f"outcomes.html changed: sha256 {got}"


def test_probe_ids_unique_across_both_pages() -> None:
    """A1c-1: every probe id is unique across both pages (the builder refuses a duplicate, A1c-3; the fixture must
    not carry one)."""
    ids = [p["id"] for p in probe_table()] + [p["id"] for p in probe_table_b()]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert dup == [], dup


def test_probe_table_b_fields_and_mapping() -> None:
    """A1c-1's probe fields on every page-b probe: ``id``, ``intended`` (one of the four classes), ``kind``, ``stress``
    (a bool), ``tool`` (``click_at``: page b has no typing probe), ``x``/``y`` in 0..999 with ``x_px``/``y_px`` the
    hand mapping at 1280x800, a ``query`` that starts with ``?scene=`` and names a scene the page has, ``target``
    (null only for blank space), ``changes`` (four ints) exactly on near and elsewhere probes; a ``pre`` has integer
    ``x``/``y`` with ``x_px``/``y_px`` mapped the same way and a ``target``. ``stress`` is true exactly on the stress
    kinds."""
    html = _page_b_html()
    for p in probe_table_b():
        assert p["intended"] in CLASSES and isinstance(p["stress"], bool), p
        assert p["tool"] == "click_at", p
        assert 0 <= p["x"] <= 999 and 0 <= p["y"] <= 999, p
        assert (p["x_px"], p["y_px"]) == (_px(p["x"], 1280), _px(p["y"], 800)), p
        q = p["query"]
        assert isinstance(q, str) and q.startswith("?scene="), p
        assert f'<section id="s-{q[len("?scene=") :]}"' in html, (p["id"], q)
        assert p["stress"] is (p["kind"] in STRESS_KINDS), p
        has_rect = p["intended"] in ("near", "elsewhere")
        assert ("changes" in p) is has_rect, p
        if has_rect:
            c = p["changes"]
            assert (
                len(c) == 4 and all(isinstance(v, int) for v in c) and c[0] < c[2] and c[1] < c[3]
            ), p
        if (p["target"] is None) != (p.get("target_kind") == "blank"):
            raise AssertionError(f"{p['id']}: target null exactly for blank space: {p}")
        if "pre" in p:
            pre = p["pre"]
            assert all(type(pre[k]) is int for k in ("x", "y")), p
            assert (pre["x_px"], pre["y_px"]) == (_px(pre["x"], 1280), _px(pre["y"], 800)), p
            assert isinstance(pre["target"], str), p


def test_probe_table_b_minimum_counts() -> None:
    """A1c-2's minimum counts in the page-b table, per kind (the kinds are the page's legend, A1c-1):

    - gated ``none`` >= 52: N-UL >= 24 distinct links, N-BTN >= 8, N-TINT >= 6, N-PRE >= 14 (each a distinct ``pre``
      element);
    - trivial ``none`` (report-only): >= 2 each of a disabled control, an image and an already-selected option, with
      no ``pre``;
    - gated ``near`` >= 46: C-SMALL >= 24 over six kinds (>= 4 each), C-COLOUR >= 8, C-RELABEL >= 8, C-FOCUS >= 6;
    - gated ``elsewhere`` >= 16; gated ``page`` >= 12 (>= 8 keep the bar, >= 4 replace everything);
    - stress >= 18, >= 3 of each of the six kinds.
    Every kind has the class A1c-2 gives it (stress: hovers ``none``, strikethrough and equal-luma ``near``)."""
    table = probe_table_b()
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for p in table:
        by_kind.setdefault(p["kind"], []).append(p)
    n = {k: len(v) for k, v in by_kind.items()}
    want_class = dict.fromkeys(GATED_NONE_KINDS + TRIVIAL_KINDS, "none")
    want_class |= dict.fromkeys(GATED_NEAR_KINDS, "near")
    want_class |= {"e-small": "elsewhere", "p-bar": "page", "p-full": "page"}
    want_class |= {"s-shadow": "none", "s-lift": "none", "s-ghost": "none", "s-row": "none"}
    want_class |= {"s-strike": "near", "s-luma": "near"}
    assert set(n) == set(want_class), sorted(set(n) ^ set(want_class))
    for p in table:
        assert p["intended"] == want_class[p["kind"]], p
    floors = {
        "n-ul": 24,
        "n-btn": 8,
        "n-tint": 6,
        "n-pre": 14,
        **dict.fromkeys(TRIVIAL_KINDS, 2),
        **dict.fromkeys(C_SMALL_KINDS, 4),
        "c-colour": 8,
        "c-relabel": 8,
        "c-focus": 6,
        "e-small": 16,
        "p-bar": 8,
        "p-full": 4,
        **dict.fromkeys(STRESS_KINDS, 3),
    }
    short = {k: (n.get(k, 0), f) for k, f in floors.items() if n.get(k, 0) < f}
    assert short == {}, short
    assert sum(n[k] for k in GATED_NONE_KINDS) >= 52, n
    assert sum(n[k] for k in C_SMALL_KINDS) >= 24, n
    assert sum(n[k] for k in GATED_NEAR_KINDS) >= 46, n
    assert n["p-bar"] + n["p-full"] >= 12, n
    assert sum(n[k] for k in STRESS_KINDS) >= 18, n
    assert len({p["target"] for p in by_kind["n-ul"]}) == n["n-ul"], "two N-UL probes share a link"
    pres = [p["pre"]["target"] for p in by_kind["n-pre"]]
    assert len(set(pres)) == len(pres), "two N-PRE probes share a pre element"
    assert all("pre" in p for p in by_kind["n-pre"]), "an N-PRE probe has no pre"
    assert not any("pre" in p for p in table if p["kind"] != "n-pre"), "a pre outside N-PRE"


def test_probe_table_b_n_ul_links() -> None:
    """A1c-2's N-UL spread, each fact read from the page's markup (the link's inline style), not from the table's
    own claim, which must agree with it: >= 10 links with a width >= 400 px, >= 4 that wrap onto two lines (a box two
    line-heights tall that wraps), font sizes 13, 16, 20 and 24 px with >= 3 links each, and the pointer in the
    first quarter (``x_px - left < width / 4``), the middle (within 3/8..5/8 of the width; A1c-2 says "middle
    quarter", read as the central quarter) and the last quarter (``> 3/4``) of >= 6 links each, on the link's first
    line. Every N-UL link is an ``<a>`` with no ``href`` (so it neither navigates nor takes focus)."""
    html = _page_b_html()
    links = [p for p in probe_table_b() if p["kind"] == "n-ul"]
    fonts: dict[int, int] = {}
    quarters = {"first": 0, "middle": 0, "last": 0}
    wide = wrapped = 0
    for p in links:
        tag, attrs = _element(html, p["target"][1:])
        assert tag == "a" and "href" not in attrs, (p["id"], tag, attrs)
        x0, y0, x1, y1 = _box(attrs)
        fs = int(re.search(r"font-size: (\d+)px", attrs).group(1))  # type: ignore[union-attr]
        lh = int(re.search(r"line-height: (\d+)px", attrs).group(1))  # type: ignore[union-attr]
        width = x1 - x0
        lines = (y1 - y0) // lh
        assert (y1 - y0) % lh == 0 and lines in (1, 2), (p["id"], attrs)
        assert lines == 1 or "ul-wrap" in attrs, (p["id"], "a two-line box that does not wrap")
        assert (p["font_px"], p["width_px"], p["lines"]) == (fs, width, lines), p
        assert x0 <= p["x_px"] < x1 and y0 <= p["y_px"] < y0 + lh, (
            p["id"],
            "pointer not on the first line",
        )
        frac = Fraction(p["x_px"] - x0, width)
        q = (
            "first"
            if frac < Fraction(1, 4)
            else "last"
            if frac > Fraction(3, 4)
            else "middle"
            if Fraction(3, 8) <= frac <= Fraction(5, 8)
            else None
        )
        assert q == p["quarter"], (p["id"], frac, p["quarter"])
        quarters[q] += 1
        fonts[fs] = fonts.get(fs, 0) + 1
        wide += width >= 400
        wrapped += lines == 2
    assert wide >= 10 and wrapped >= 4, (wide, wrapped)
    assert {f: fonts.get(f, 0) >= 3 for f in (13, 16, 20, 24)} == dict.fromkeys(
        (13, 16, 20, 24), True
    ), fonts
    assert min(quarters.values()) >= 6, quarters


def test_probe_table_b_tint_colour_and_luma_values() -> None:
    """A1c-2's colour bounds, by hand from the page's CSS with PIL's luma: every N-TINT chip's background and hover
    tint (both in the page's style rules for that chip) differ by 8..30 in gray; every C-COLOUR pair differs by >= 30
    (and the icon's ``.on`` rule names the second colour); every equal-luma stress swap differs by <= 24."""
    html = _page_b_html()
    for p in probe_table_b():
        if p["kind"] == "n-tint":
            a, b = p["tint"]
            sel = p["target"]
            assert f"{sel} {{ background: {a}; }}" in html, (p["id"], a)
            assert f"{sel}:hover {{ background: {b}; }}" in html, (p["id"], b)
            assert 8 <= abs(_gray_of_hex(a) - _gray_of_hex(b)) <= 30, (p["id"], a, b)
        elif p["kind"] == "c-colour":
            a, b = p["colours"]
            assert re.search(rf"{re.escape(p['changes_el'])}\.on {{ fill: {b};", html), (p["id"], b)
            assert abs(_gray_of_hex(a) - _gray_of_hex(b)) >= 30, (p["id"], a, b)
        elif p["kind"] == "s-luma":
            a, b = p["colours"]
            assert f"{p['changes_el']} {{ background: {a}; }}" in html, (p["id"], a)
            assert f"{p['changes_el']}.on {{ background: {b}; }}" in html, (p["id"], b)
            assert abs(_gray_of_hex(a) - _gray_of_hex(b)) <= 24, (p["id"], a, b)


def test_probe_table_b_pre_and_trivial_targets() -> None:
    """A1c-2's N-PRE and trivial targets, checked against the markup: an N-PRE ``pre`` is a hover-only link (an
    ``<a>`` with no ``href``) and the probe's target covers >= 4 disabled controls (the element has ``disabled``), >= 4
    inline SVG images, >= 4 already-selected options (a segment with class ``on``) and >= 2 blank points; the trivial
    probes click such a disabled control, image or selected option on a fresh load (no ``pre``)."""
    html = _page_b_html()
    kinds = {"disabled": 0, "image": 0, "selected": 0, "blank": 0}

    def kind_of(sel: str | None) -> str:
        if sel is None:
            return "blank"
        tag, attrs = _element(html, sel[1:])
        if re.search(r"\bdisabled\b", attrs):
            return "disabled"
        if tag == "svg":
            return "image"
        if re.search(r'class="[^"]*\bseg on\b', attrs):
            return "selected"
        return f"other:{tag}"

    for p in probe_table_b():
        if p["kind"] == "n-pre":
            tag, attrs = _element(html, p["pre"]["target"][1:])
            assert tag == "a" and "href" not in attrs, (p["id"], "the pre is not a hover-only link")
            k = kind_of(p["target"])
            assert k == p["target_kind"], (p["id"], k)
            kinds[k] += 1
        elif p["kind"] in TRIVIAL_KINDS:
            assert "pre" not in p and kind_of(p["target"]) == p["kind"][len("t-") :], p
    assert kinds["disabled"] >= 4 and kinds["image"] >= 4 and kinds["selected"] >= 4, kinds
    assert kinds["blank"] >= 2, kinds


def test_probe_table_b_page_links() -> None:
    """A1c-2's page probes against the markup: each target is a link whose ``href`` is the probe's ``dest`` (a
    relative ``?scene=`` URL of a scene the page has), every destination is distinct, the bar-keeping ones (``d``
    scenes) are the scenes the page's script shows the site bar on and the full replacements (``f`` scenes) are not."""
    html = _page_b_html()
    pages = [p for p in probe_table_b() if p["intended"] == "page"]
    dests = [p["dest"] for p in pages]
    assert len(set(dests)) == len(dests), "two page probes share a destination"
    bar_rule = re.search(
        r"if \(/\^\(([^)]*)\)\$/\.test\(name\)\) document\.getElementById\(\"site-bar\"\)", html
    )
    assert bar_rule is not None, (
        "premise: the page's script shows the site bar by a scene-name pattern"
    )
    shows_bar = re.compile(rf"^({bar_rule.group(1)})$")
    for p in pages:
        tag, attrs = _element(html, p["target"][1:])
        assert tag == "a" and f'href="{p["dest"]}"' in attrs, (p["id"], attrs)
        scene = p["dest"][len("?scene=") :]
        assert f'<section id="s-{scene}"' in html, p
        assert bool(shows_bar.match(scene)) is (p["kind"] == "p-bar"), (p["id"], scene)
        assert shows_bar.match(p["query"][len("?scene=") :]), (
            "the page probes' own scene must show the bar"
        )


def test_probe_b_near_and_elsewhere_geometry() -> None:
    """A1c-2's geometry, by hand from A2-3's definitional near box (+-320 x +-80 px around the click, closed): every
    gated ``near`` probe's ``changes`` rect lies inside the box; every ``elsewhere`` rect lies outside it with a
    margin of >= 64 px, and the elsewhere rects, each wholly inside one ninth, cover >= 6 ninths.

    Adversarial (the task's failure mode, a near change read as ``none`` or as ``elsewhere``): a rect one pixel past
    the box would be a near probe a correct detector reads as ``elsewhere``; a rect inside the 64 px margin would be an
    elsewhere probe a 64 px cell grid can straddle."""
    ninths = set()
    for p in probe_table_b():
        if p["intended"] not in ("near", "elsewhere") or p["stress"]:
            continue
        x, y = p["x_px"], p["y_px"]
        x0, y0, x1, y1 = p["changes"]
        if p["intended"] == "near":
            assert x - NEAR_HW <= x0 and x1 <= x + NEAR_HW, (p["id"], "x", p["changes"])
            assert y - NEAR_HH <= y0 and y1 <= y + NEAR_HH, (p["id"], "y", p["changes"])
        else:
            gap = max(
                x0 - (x + NEAR_HW), (x - NEAR_HW) - x1, y0 - (y + NEAR_HH), (y - NEAR_HH) - y1
            )
            assert gap >= MARGIN, (p["id"], gap)
            k = _ninth_of_rect(p["changes"])
            assert k is not None, (p["id"], "the rect straddles two ninths", p["changes"])
            ninths.add(k)
    assert len(ninths) >= 6, sorted(ninths)


def test_probe_b_gated_probes_are_independent_by_construction() -> None:
    """A1c-2's independence, the part that can be checked without a browser: no two gated probes share (scene,
    ``pre`` element, target, point), so no two can produce the same (before, after) frames, and no two page-b probes
    share (scene, ``pre`` point, point), so the builder's ``pair_id`` (frames, point, previous point) cannot merge two
    of them into one record (A1c-6, "exactly one pair"). The pixel-level check (jitter clusters of the real frames)
    is the browser module's."""
    table = probe_table_b()
    gated = [p for p in table if not p["stress"] and p["kind"] not in TRIVIAL_KINDS]
    keys = [
        (p["query"], (p.get("pre") or {}).get("target"), p["target"], p["x_px"], p["y_px"])
        for p in gated
    ]
    assert len(set(keys)) == len(keys), "two gated probes are the same action on the same scene"
    pts = [
        (
            p["query"],
            (p.get("pre") or {}).get("x_px"),
            (p.get("pre") or {}).get("y_px"),
            p["x_px"],
            p["y_px"],
        )
        for p in table
    ]
    dup = sorted({k for k in pts if pts.count(k) > 1}, key=str)
    assert dup == [], dup


#: Every site.css rule A1c-1 says the page copies, by selector as site.css writes it -> the selector in the page.
SITE_RULES = {
    "a": "a",
    "a:hover": "a:hover",
    "button,\n.button": "button",
    "button:hover": "button:hover",
    "input,\ntextarea": "input",
    "input:focus,\ntextarea:focus": "input:focus",
}


def _css_block(css: str, selector: str) -> dict[str, str]:
    m = re.search(rf"(?:^|\n)\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", css)
    assert m is not None, f"no rule {selector!r}"
    decls = {}
    for d in m.group(1).split(";"):
        if d.strip():
            k, v = d.split(":", 1)
            decls[k.strip()] = " ".join(v.split())
    return decls


def test_probe_page_b_copies_the_site_styles() -> None:
    """A1c-1: the page's link, button and input rules are site.css's (``a``/``a:hover``, ``button``/``button:hover``,
    ``input``/``input:focus``), declaration for declaration, with site.css's ``:root`` variables written in (read from
    site.css here, so a site.css change reddens this row); the page cites site.css in a comment and links nothing
    (no ``<link``, no ``local_astra/pages`` URL)."""
    site = (REPO / "local_astra" / "pages" / "static" / "assets" / "site.css").read_text(
        encoding="utf-8"
    )
    root = _css_block(site, ":root")
    html = _page_b_html()
    style = re.search(r"<style>(.*?)</style>", html, flags=re.S)
    assert style is not None
    css = re.sub(r"/\*.*?\*/", "", style.group(1), flags=re.S)
    assert "site.css" in style.group(1), "the style element does not cite site.css"
    for site_sel, page_sel in SITE_RULES.items():
        want = {
            k: re.sub(r"var\((--[\w-]+)\)", lambda m: root[m.group(1)], v)
            for k, v in _css_block(site, site_sel).items()
        }
        assert _css_block(css, page_sel) == want, (page_sel, _css_block(css, page_sel), want)
    assert "<link" not in html and "local_astra/pages/" not in re.sub(
        r"<!--.*?-->|/\*.*?\*/", "", html, flags=re.S
    )


def test_probe_page_b_is_deterministic_and_offline() -> None:
    """A1c-1 (the same scan as page a's, plus what a scene choice could read): no timers, no animation other than the
    neutraliser, no network API, no absolute or protocol-relative URL (no SVG namespace URL either), no external
    resource, no autofocus or scripted focus, nothing but ``location.search`` a script could vary on (no clock,
    random, storage, cookie or history); the only navigation is relative ``?scene=`` links to scenes the page has;
    nothing is focused at load (no ``autofocus``)."""
    code = re.sub(r"<!--.*?-->", "", _page_b_html(), flags=re.S)
    for word in (
        "setTimeout",
        "setInterval",
        "requestAnimationFrame",
        "requestIdleCallback",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "sendBeacon",
        "navigator.",
        "import(",
        "@keyframes",
        "@import",
        "@font-face",
        "url(",
        "<img",
        "<link",
        "<iframe",
        "src=",
        "xmlns",
        "autofocus",
        ".focus(",
        "Math.random",
        "performance.",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "history.",
    ):
        assert word not in code, f"outcomes_b.html uses {word!r}"
    assert not re.search(r"(https?:)?//[A-Za-z0-9]", code), "outcomes_b.html names a URL"
    assert re.findall(r"animation:\s*([^;]+);", code) == ["none !important"], "an animation"
    assert not re.search(r"\bDate\b", code), "outcomes_b.html reads a clock"
    assert "transition-property: none !important" in code
    hrefs = re.findall(r'href="([^"]*)"', code)
    assert hrefs and all(re.fullmatch(r"\?scene=[a-z0-9-]+", h) for h in hrefs), hrefs
    for h in hrefs:
        assert f'<section id="s-{h[len("?scene=") :]}"' in code, h


def test_probe_pages_order_a_then_b() -> None:
    """A1c-3: ``PROBE_PAGES`` is ``(outcomes.html, outcomes_b.html)``, in that order; ``probe_table()`` is page a's
    table, unchanged and in its order, then page b's, each entry carrying its ``probe_page`` and otherwise equal to
    the table entry (so page a's probes make exactly A1's calls)."""
    pages = tuple(Path(p).resolve() for p in api("PROBE_PAGES"))
    assert pages == (PROBE_PAGE.resolve(), PROBE_PAGE_B.resolve()), pages
    plan = api("probe_table")()
    a, b = probe_table(), probe_table_b()
    assert [p.get("id") for p in plan] == [p["id"] for p in a] + [p["id"] for p in b], (
        "not page a's table then page b's"
    )
    for got, (want, page) in zip(
        plan, [(p, "outcomes.html") for p in a] + [(p, "outcomes_b.html") for p in b], strict=True
    ):
        assert got.get("probe_page") == page, got
        assert {k: v for k, v in got.items() if k != "probe_page"} == want, (got, want)


def test_probe_urls() -> None:
    """A1c-3: each probe loads ``origin + "/" + <page> + <query>``: page a's (no query) is ``/outcomes.html``
    exactly as at A1, page b's is ``/outcomes_b.html?scene=...``."""
    url = api("probe_url")
    origin = "http://127.0.0.1:43210"
    a0 = dict(probe_table()[0], probe_page="outcomes.html")
    assert url(origin, a0) == "http://127.0.0.1:43210/outcomes.html", url(origin, a0)
    for p in probe_table_b()[:3] + probe_table_b()[-3:]:
        got = url(origin, dict(p, probe_page="outcomes_b.html"))
        assert got == f"http://127.0.0.1:43210/outcomes_b.html{p['query']}", (p["id"], got)


def _page_with(table: Sequence[Mapping[str, Any]]) -> str:
    return (
        '<!doctype html>\n<html><head><script type="application/json" id="probes">\n'
        + json.dumps(list(table), indent=1)
        + "\n</script></head><body></body></html>\n"
    )


def _bad_b(case: str) -> tuple[list[dict[str, Any]], str, str]:
    """Page b's table with one defect: ``(table, the probe id the refusal must name, a word it must say)``."""
    tb = [dict(p) for p in probe_table_b()]
    pre_i = next(i for i, p in enumerate(tb) if "pre" in p)
    pid = tb[pre_i]["id"]
    if case == "duplicate-id-across-pages":
        tb[3]["id"] = probe_table()[0]["id"]
        return tb, tb[3]["id"], r"duplicat|unique|twice"
    if case == "duplicate-id-within-b":
        tb[3]["id"] = tb[2]["id"]
        return tb, tb[2]["id"], r"duplicat|unique|twice"
    if case == "query-without-question-mark":
        tb[0]["query"] = "scene=ul1"
        return tb, tb[0]["id"], r"query"
    if case == "query-starting-with-a-path":
        tb[0]["query"] = "/outcomes_b.html?scene=ul1"
        return tb, tb[0]["id"], r"query"
    pre = dict(tb[pre_i]["pre"])
    if case == "pre-without-y":
        del pre["y"]
    elif case == "pre-float-x":
        pre["x"] = 62.5
    elif case == "pre-string-x":
        pre["x"] = "62"
    elif case == "pre-not-an-object":
        tb[pre_i]["pre"] = [pre["x"], pre["y"]]
        return tb, pid, r"\bpre\b"
    tb[pre_i]["pre"] = pre
    return tb, pid, r"\bpre\b"


BAD_TABLE_CASES = (
    "duplicate-id-across-pages",
    "duplicate-id-within-b",
    "query-without-question-mark",
    "query-starting-with-a-path",
    "pre-without-y",
    "pre-float-x",
    "pre-string-x",
    "pre-not-an-object",
)


@pytest.mark.parametrize("case", BAD_TABLE_CASES)
def test_bad_probe_tables_exit_2_before_any_browser(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_sha: str,
) -> None:
    """A1c-3's refusals: with ``--probes`` and the probe pages (``PROBE_PAGES``, replaced by page a and a tmp page b
    with one defect) holding a probe id twice (across the pages, or within one), a ``query`` that does not start with
    ``?``, or a ``pre`` without integer ``x``/``y`` (``y`` missing, a float, a string, not an object), the builder
    exits 2 before any process starts, leaves the out dir uncreated, and names the probe and the reason on stderr."""
    table, pid, word = _bad_b(case)
    bad_b = tmp_path / "outcomes_b.html"
    bad_b.write_text(_page_with(table), encoding="utf-8")
    run = write_run(tmp_path, "empty-run", [])
    out = REPO / "logs" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    module = builder()
    monkeypatch.setattr(module, "PROBE_PAGES", (PROBE_PAGE, bad_b), raising=False)
    _fake_git(module, monkeypatch, "", fake_sha)
    started = _spy_processes(monkeypatch)
    rc, exc = _run_main(module, [str(run), "--out", str(out), "--probes"])
    err = capsys.readouterr().err
    assert exc is None, f"{case}: main raised instead of refusing: {type(exc).__name__}: {exc}"
    assert started == [], f"{case}: a process started before the refusal: {started}"
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert not out.exists(), f"{case}: the refusal created {out}"
    assert pid in err and re.search(word, err, flags=re.I), (case, pid, err[-2000:])


@pytest.mark.parametrize("pages", ["the-fixture-pages", "a-tmp-copy-of-page-b"])
def test_valid_probe_pages_are_not_refused(
    pages: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_sha: str,
) -> None:
    """A1c-3, the other side of the refusals (a guard too wide): the fixture pages as they are, and a tmp page b with
    page b's table unchanged, are not refused: the builder goes on to start the browser (the spy refuses the process,
    so main fails or raises, but not with exit 2 before trying). Holds at BASE_A1c, which ignores page b."""
    run = write_run(tmp_path, "empty-run", [])
    module = builder()
    if pages == "a-tmp-copy-of-page-b":
        copy_b = tmp_path / "outcomes_b.html"
        copy_b.write_text(_page_with(probe_table_b()), encoding="utf-8")
        monkeypatch.setattr(module, "PROBE_PAGES", (PROBE_PAGE, copy_b), raising=False)
    _fake_git(module, monkeypatch, "", fake_sha)
    started = _spy_processes(monkeypatch)
    out = REPO / "logs" / f"t0136-outcome-corpus-{fake_sha[:7]}"
    rc, exc = _run_main(module, [str(run), "--out", str(out), "--probes"])
    err = capsys.readouterr().err
    assert not (rc == 2 and not started), f"{pages}: refused: exit {rc}\n{err[-2000:]}"
    assert started, f"{pages}: the builder never tried to start the browser (exit {rc}, {exc!r})"


def _png_frame(arr: Any) -> tuple[str, bytes]:
    data = _png(arr)
    return hashlib.sha256(data).hexdigest(), data


def _frames(*arrays: Any) -> tuple[list[str], dict[str, bytes]]:
    shas, frames = [], {}
    for a in arrays:
        sha, data = _png_frame(a)
        shas.append(sha)
        frames[sha] = data
    assert len(frames) == len(arrays), "premise: distinct synthetic frames"
    return shas, frames


def _partition(groups: Mapping[str, str]) -> set[frozenset[str]]:
    by: dict[str, set[str]] = {}
    for sha, gid in groups.items():
        by.setdefault(gid, set()).add(sha)
    return {frozenset(v) for v in by.values()}


def test_frame_groups_jitter_rule() -> None:
    """A1c-4 on synthetic lossless frames (PNG, exact gray values; 64x48 flat gray 100 unless said):

    - |Δ| 8 merges and 9 does not: B is A with one pixel at 108, E is A with another pixel at 109 (E against B: 9 at
      E's pixel); A and B form one cluster, E its own;
    - a uniform +8 merges with A, a uniform +9 does not;
    - transitive closure: C is B with that pixel at 116 (C~B, C≁A at 16): A, B and C are one cluster (adversarial: a
      clusterer comparing each frame with a representative only splits C off);
    - different sizes never merge: F is A with one more row of 100, identical on the overlap;
    - the id of a cluster is its lexicographically smallest sha256, a singleton's its own sha;
    - the result does not depend on the order of the input mapping.
    Adversarial (mutation 21, sha-only groups): every frame its own cluster fails the first and third points."""
    import numpy as np

    fg = api("frame_groups")
    a = np.full((48, 64), 100, dtype=np.int16)
    b = a.copy()
    b[10, 10] = 108
    e = a.copy()
    e[20, 20] = 109
    c = b.copy()
    c[10, 10] = 116
    f = np.full((49, 64), 100, dtype=np.int16)
    (sa, sb, se), frames = _frames(a, b, e)
    got = dict(fg(frames))
    assert set(got) == set(frames), "not every frame is mapped"
    assert _partition(got) == {frozenset({sa, sb}), frozenset({se})}, _partition(got)
    assert got[sa] == got[sb] == min(sa, sb) and got[se] == se, got
    (s0, s8), frames = _frames(a, a + 8)
    assert _partition(dict(fg(frames))) == {frozenset({s0, s8})}, "a uniform +8 did not merge"
    (s0, s9), frames = _frames(
        a, a + 9
    )  # alone: a + 8 beside it would chain them (|a+9 - (a+8)| = 1)
    assert _partition(dict(fg(frames))) == {frozenset({s0}), frozenset({s9})}, "a uniform +9 merged"
    (sa, sb, sc), frames = _frames(a, b, c)
    got = dict(fg(frames))
    assert _partition(got) == {frozenset({sa, sb, sc})}, "the chain A~B~C is not one cluster"
    assert set(got.values()) == {min(sa, sb, sc)}, got
    (sa, sf), frames = _frames(a, f)
    assert _partition(dict(fg(frames))) == {frozenset({sa}), frozenset({sf})}, "sizes merged"
    (sa, sb, sc, se, sf), frames = _frames(a, b, c, e, f)
    forward = dict(fg(frames))
    backward = dict(fg(dict(reversed(list(frames.items())))))
    assert forward == backward, "the groups depend on the input order"
    assert _partition(forward) == {frozenset({sa, sb, sc}), frozenset({se}), frozenset({sf})}
    assert forward[se] == se and forward[sf] == sf and forward[sc] == min(sa, sb, sc)


def test_frame_groups_decode_jpeg_at_full_resolution() -> None:
    """A1c-4: the rule is judged on the decoded full-resolution gray frames, not on bytes: two JPEG encodings of the
    same flat frame (quality 90 and 80, different bytes) merge; the frame with a 200x100 block 100 levels brighter
    (the A1 compare rows' large change) does not. Expected by an independent decode in this row."""
    import numpy as np
    from PIL import Image

    def jpeg(arr: Any, q: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(
            buf, format="JPEG", quality=q
        )
        return buf.getvalue()

    base = np.full((800, 1280), 128, dtype=np.int16)
    block = base.copy()
    block[500:600, 800:1000] = 228
    data = [jpeg(base, 90), jpeg(base, 80), jpeg(block, 90)]
    shas = [hashlib.sha256(d).hexdigest() for d in data]
    assert len(set(shas)) == 3, "premise: three different byte strings"
    dec = [np.asarray(Image.open(io.BytesIO(d)).convert("L"), dtype=np.int16) for d in data]
    assert int(np.abs(dec[0] - dec[1]).max()) <= 8 < int(np.abs(dec[0] - dec[2]).max()), "premise"
    got = dict(api("frame_groups")(dict(zip(shas, data, strict=True))))
    assert _partition(got) == {frozenset(shas[:2]), frozenset(shas[2:])}, _partition(got)


def test_a1c_scope() -> None:
    """A1c-N: over ``BASE_A1C..TIP_A1C`` (two literals) no ``local_astra/`` file changes and, among ``scripts/``, only
    ``build_outcome_corpus.py`` (A1-8's "no existing scripts/ file" is lifted for that file only). Green at BASE_A1C
    (an empty range until ``TIP_A1C`` is bumped); the A1-8 power row shows the predicates see changes."""
    gr.require_commits(BASE_A1C, TIP_A1C)
    assert gr.is_ancestor(BASE_A1C, TIP_A1C), (BASE_A1C, TIP_A1C)
    assert gr.changed(BASE_A1C, TIP_A1C, "local_astra") == []
    touched = gr.changed(BASE_A1C, TIP_A1C, "scripts")
    assert set(touched) <= {"scripts/build_outcome_corpus.py"}, touched
