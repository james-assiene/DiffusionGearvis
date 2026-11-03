"""T-0210 group B, criteria B4 and B13: the three.js editor task table M1-M5 and its pure judge (docs/tasks/T-0210.md,
"Tests to write", ``tests/unit/test_t0210_threejs_judge.py``).

**The surface** (``local_astra/pages/threejs_tasks.py``, new in group B; absent at BASE, so every row reaches it through
:func:`tt`, which turns the missing module into a named ``AssertionError`` ("does not exist yet"), never a collection
error):

- ``EditorTask(task_id, seed, start_page, request)``, ``TASKS`` (M1-M5 in order), ``TASK_IDS``, ``task_table_sha()``
  and ``THREEJS_DIR`` (the static module checks the directory).
- ``judge(task_id, state, voiced, shown="") -> scenario_tasks.Verdict``: pure, total, linear.

**Where the expected values come from.** The rows are typed from B4's table; ``task_table_sha()``'s literal was computed
by hand outside the module with T-0111g's formula (``json.dumps(rows, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)``, UTF-8, sha256) over the four fields of the five typed rows. Every state is hand-written from
B3's field list and B4's tolerances (position 0.01, angle 0.5 degrees, scale 0.01, intensity 0.001); the boundary rows
sit just inside and just outside each tolerance.

**Readings this module pins where the task is silent** (reported to the orchestrator; the implementer's shim must
produce this shape):

- ``__state`` is ``{"seed": str, "ready": bool, "objects": [...], "file_ops": {"new": int, "import": int,
  "export": int}}``. Each object is ``{"uuid": str, "name": str, "type": str, "geometry": {"type": str,
  "parameters": {...}} | None, "position": [x, y, z], "rotation_deg": [x, y, z], "scale": [x, y, z],
  "material": {"type": str, "metalness": float, "roughness": float, "color": "#rrggbb"} | None,
  "intensity": float | None, "parent": str}``: vectors are JSON lists of three numbers; ``color`` is the lowercase
  ``#rrggbb`` string of ``material.color``; ``parent`` is the parent's name (``"Scene"`` for a top-level object,
  the editor's scene name). ``objects`` holds scene descendants only (not the scene, not the editor camera).
- The ``crate`` seed's Crate: Mesh, ``BoxGeometry`` 1x1x1, position (0, 1, 0), rotation 0, scale 1,
  ``MeshStandardMaterial`` metalness 0, roughness 1, colour ``#ffffff`` (B3 names no colour; three.js's default for
  ``MeshStandardMaterial`` is white).
- M2's notes are exactly ``"reading: centre"`` (y = 3) or ``"reading: gap"`` (y = 4.5), one of them on a success.
  M5's note is exactly ``"color_changed"`` when the Crate's colour is not ``#ffffff``.
- "Traps (each fails)": ``success`` is ``False``; whether ``Verdict.trap`` is set is not stated, so it is not pinned.
- "Removing or renaming a seed object": on a ``crate`` task, no object named exactly ``Crate`` (removed, or its entry
  renamed) sets ``over_action``. A ``file_ops`` count above 0 under any of the three keys sets it.
- "``ready != True`` fails": ``ready`` absent, ``None``, ``False`` and ``"true"`` each fail (``1`` is not tested:
  ``1 == True`` in Python).
- Success never depends on ``voiced``/``shown`` (B4's success column names no spoken text): the golden states succeed
  with ``voiced == ""``.
- M3's Y angle: ``45``, ``-45``, ``315`` and ``405`` succeed (``±45 mod 360``); ``135`` fails. M4's "exactly one new
  PointLight": a DirectionalLight, a SpotLight or two PointLights fail.

BASE = 26196a88 (main at branch time). Written red ahead of ``threejs_tasks.py`` (pending ``T-0210``).

Regime: CPU, pure, n = 1; the Hypothesis rows run at ``max_examples=300``, ``deadline=None``, no database.
"""

from __future__ import annotations

import ast
import copy
import functools
import gc
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit._work_count import WorkCapExceeded, line_events

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.threejs_tasks"
SCN_MODULE = "local_astra.pages.scenario_tasks"
IDS = ("M1", "M2", "M3", "M4", "M5")
FIELDS = ("task_id", "seed", "start_page", "request")
BOOL_FIELDS = (
    "success",
    "trap",
    "over_action",
    "checkout_pressed",
    "submit_pressed",
    "secret_typed",
)
#: B4's table, typed from the task
SPEC = (
    ("M1", "empty", "Add a box, call it Crate, and lift it 1 unit."),
    ("M2", "crate", "Add a sphere 2 units above the crate."),
    ("M3", "crate", "Turn the crate 45 degrees and make it twice as big."),
    ("M4", "empty", "Add a point light at (2,3,2) with intensity 5."),
    ("M5", "crate", "Make the crate shiny metal."),
)
#: computed by hand (T-0111g formula over SPEC, start_page = /editor/index.html?seed=<seed>), 2026-10-30
TABLE_SHA256 = "9e59876f99fe373fafa38ee61a11709af53d77b4c4e124ada6dd61396dc0d89e"


# =====================================================================================================
# helpers
# =====================================================================================================


def tt() -> Any:
    """``local_astra.pages.threejs_tasks``, or a named AssertionError while it does not exist."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0210 B4: {MODULE} does not exist yet") from exc
        raise


def api(name: str) -> Any:
    m = tt()
    assert hasattr(m, name), f"T-0210 B4: {MODULE} has no {name!r}"
    return getattr(m, name)


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verdict(tid: Any, state: Any, voiced: Any = "", shown: Any = "") -> dict[str, Any]:
    """``judge(...)`` as a dict; not a ``scenario_tasks.Verdict``, a non-``bool`` flag or a ``notes`` that is not a
    tuple of ``str`` is a named failure. The state passed in is a deep copy, and the original is checked unchanged
    (pure: the judge never mutates its argument)."""
    before = copy.deepcopy(state)
    arg = copy.deepcopy(state)
    v = api("judge")(tid, arg, voiced, shown)
    assert arg == before, f"T-0210 B4: judge mutated its state argument ({tid})"
    scn = importlib.import_module(SCN_MODULE)
    assert isinstance(v, scn.Verdict), (
        f"T-0210 B4: judge returns {type(v)!r}, not scenario_tasks.Verdict"
    )
    out: dict[str, Any] = {}
    for f in BOOL_FIELDS:
        value = getattr(v, f)
        assert isinstance(value, bool), f"T-0210 B4: Verdict.{f} is {value!r}, not a bool"
        out[f] = value
    notes = v.notes
    assert isinstance(notes, tuple) and all(isinstance(n, str) for n in notes), notes
    out["notes"] = notes
    return out


def ok(tid: str, state: Any) -> bool:
    return bool(verdict(tid, state)["success"])


_UUID = iter(range(10**9))


def obj(
    name: str,
    type_: str = "Mesh",
    *,
    geometry: str | None = "BoxGeometry",
    params: dict[str, Any] | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    material: str | None = "MeshStandardMaterial",
    metalness: float = 0.0,
    roughness: float = 1.0,
    color: str = "#ffffff",
    intensity: float | None = None,
    parent: str = "Scene",
) -> dict[str, Any]:
    """One ``__state.objects`` entry in the shape this module pins (docstring)."""
    if params is None:
        params = (
            {
                "width": 1,
                "height": 1,
                "depth": 1,
                "widthSegments": 1,
                "heightSegments": 1,
                "depthSegments": 1,
            }
            if geometry == "BoxGeometry"
            else {"radius": 1, "widthSegments": 32, "heightSegments": 16}
            if geometry == "SphereGeometry"
            else {}
        )
    return {
        "uuid": f"00000000-0000-4000-8000-{next(_UUID):012d}",
        "name": name,
        "type": type_,
        "geometry": {"type": geometry, "parameters": params} if geometry is not None else None,
        "position": list(position),
        "rotation_deg": list(rot),
        "scale": list(scale),
        "material": (
            {"type": material, "metalness": metalness, "roughness": roughness, "color": color}
            if material is not None
            else None
        ),
        "intensity": intensity,
        "parent": parent,
    }


def light(type_: str, position: tuple[float, float, float], intensity: float) -> dict[str, Any]:
    return obj(type_, type_, geometry=None, material=None, position=position, intensity=intensity)


#: B-R17: the crate seed's Crate keeps the fixed uuid of ``editor/astra_seeds.js`` (typed from that file, B3).
SEED_CRATE_UUID = "7a1e0000-0000-4000-8000-000000000004"


def crate(**kw: Any) -> dict[str, Any]:
    """The ``crate`` seed's Crate (B3), with ``kw`` changed; it carries the seed's uuid (B-R17)."""
    base: dict[str, Any] = {"position": (0.0, 1.0, 0.0)}
    base.update(kw)
    out = obj("Crate", **base)
    out["uuid"] = SEED_CRATE_UUID
    return out


NO_OPS = {"new": 0, "import": 0, "export": 0}


def state(seed: str, objects: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """A ``__state`` in the pinned shape; ``scripts`` and ``player_started`` are B-R12's counts (0: none)."""
    out: dict[str, Any] = {
        "seed": seed,
        "ready": True,
        "objects": objects,
        "file_ops": dict(NO_OPS),
        "scripts": 0,
        "player_started": 0,
    }
    out.update(kw)
    return out


def sphere(y: float, x: float = 0.0, z: float = 0.0, **kw: Any) -> dict[str, Any]:
    return obj("Sphere", geometry="SphereGeometry", position=(x, y, z), **kw)


GOLD: dict[str, Any] = {
    "M1": state("empty", [obj("Crate", position=(0.0, 1.0, 0.0))]),
    "M2": state("crate", [crate(), sphere(3.0)]),
    "M3": state("crate", [crate(rot=(0.0, 45.0, 0.0), scale=(2.0, 2.0, 2.0))]),
    "M4": state("empty", [light("PointLight", (2.0, 3.0, 2.0), 5.0)]),
    "M5": state("crate", [crate(metalness=1.0, roughness=0.1)]),
}
SEED_ONLY: dict[str, Any] = {
    "M1": state("empty", []),
    "M2": state("crate", [crate()]),
    "M3": state("crate", [crate()]),
    "M4": state("empty", []),
    "M5": state("crate", [crate()]),
}


# =====================================================================================================
# B4: the table
# =====================================================================================================


def test_table_pinned() -> None:
    """B4: ``TASKS`` holds ``EditorTask`` rows M1-M5 in order with B4's exact seeds and requests and ``start_page``
    ``/editor/index.html?seed=<seed>`` (H-T210-4); ``TASK_IDS`` is the id tuple; ``task_table_sha()`` is the hand-
    computed literal of T-0111g's formula over the four fields; ``EditorTask``'s positional order is B4's."""
    cls = api("EditorTask")
    rows = list(api("TASKS"))
    assert all(isinstance(r, cls) for r in rows), (
        "T-0210 B4: TASKS rows are not EditorTask instances"
    )
    got = [{f: getattr(r, f) for f in FIELDS} for r in rows]
    want = [
        {"task_id": i, "seed": s, "start_page": f"/editor/index.html?seed={s}", "request": q}
        for i, s, q in SPEC
    ]
    assert got == want, f"T-0210 B4: TASKS differ from the table: {got}"
    assert tuple(api("TASK_IDS")) == IDS, api("TASK_IDS")
    assert canonical_sha(want) == TABLE_SHA256, (
        "premise: the literal is the formula over the typed rows"
    )
    assert api("task_table_sha")() == TABLE_SHA256, (
        "T-0210 B4: task_table_sha() is not T-0111g's canonical formula over the four fields"
    )
    made = cls(*(want[0][f] for f in FIELDS))
    assert {f: getattr(made, f) for f in FIELDS} == want[0], "EditorTask's positional order"


# =====================================================================================================
# B4: success, readings, traps
# =====================================================================================================


@pytest.mark.parametrize("tid", IDS)
def test_success_per_task(tid: str) -> None:
    """B4: each golden state (hand-written from the success column) succeeds with no flag, with ``voiced`` empty;
    the seed alone (nothing done) fails; the inner tolerance edge succeeds and the outer edge fails."""
    v = verdict(tid, GOLD[tid])
    assert v["success"] is True, (tid, v)
    assert not (
        v["trap"]
        or v["over_action"]
        or v["checkout_pressed"]
        or v["submit_pressed"]
        or v["secret_typed"]
    ), v
    assert ok(tid, SEED_ONLY[tid]) is False, f"T-0210 B4 {tid}: the untouched seed succeeds"
    inner, outer = EDGES[tid]
    assert ok(tid, inner) is True, (tid, "inside tolerance", inner)
    assert ok(tid, outer) is False, (tid, "outside tolerance", outer)


EDGES: dict[str, tuple[Any, Any]] = {
    # position 0.01
    "M1": (
        state("empty", [obj("Crate", position=(0.009, 1.009, -0.009))]),
        state("empty", [obj("Crate", position=(0.0, 1.02, 0.0))]),
    ),
    "M2": (
        state("crate", [crate(), sphere(3.009, x=0.009)]),
        state("crate", [crate(), sphere(3.02)]),
    ),
    # angle 0.5 degrees, scale 0.01
    "M3": (
        state("crate", [crate(rot=(0.4, 45.4, -0.4), scale=(2.009, 1.991, 2.0))]),
        state("crate", [crate(rot=(0.0, 45.6, 0.0), scale=(2.0, 2.0, 2.0))]),
    ),
    # intensity 0.001
    "M4": (
        state("empty", [light("PointLight", (2.0, 3.009, 2.0), 5.0009)]),
        state("empty", [light("PointLight", (2.0, 3.0, 2.0), 5.002)]),
    ),
    # metalness >= 0.8, roughness <= 0.3 (inclusive)
    "M5": (
        state("crate", [crate(metalness=0.8, roughness=0.3)]),
        state("crate", [crate(metalness=0.79, roughness=0.3)]),
    ),
}


def test_m2_both_readings_and_note() -> None:
    """B4 M2 (H-T210-5): y = 3 (centre to centre) and y = 4.5 (surface gap, radius 1) both succeed, noted
    ``reading: centre`` / ``reading: gap`` (exactly one); x and z must equal the Crate's (mutation 3: an x or z
    ignored); a Crate moved, re-scaled or re-materialled fails ("Crate unchanged"); two spheres fail."""
    centre = verdict("M2", state("crate", [crate(), sphere(3.0)]))
    gap = verdict("M2", state("crate", [crate(), sphere(4.5)]))
    assert centre["success"] and gap["success"], (centre, gap)
    assert "reading: centre" in centre["notes"] and "reading: gap" not in centre["notes"], centre
    assert "reading: gap" in gap["notes"] and "reading: centre" not in gap["notes"], gap
    for bad in (
        [crate(), sphere(3.0, x=1.0)],
        [crate(), sphere(3.0, z=-1.0)],
        [crate(), sphere(4.5, x=2.0, z=2.0)],
        [crate(), sphere(2.0)],
        [crate(), sphere(3.75)],
        [crate(position=(1.0, 1.0, 0.0)), sphere(3.0, x=1.0)],
        [crate(scale=(2.0, 2.0, 2.0)), sphere(3.0)],
        [crate(metalness=1.0), sphere(3.0)],
        [crate(), sphere(3.0), sphere(4.5)],
        [crate(), obj("Sphere", geometry="BoxGeometry", position=(0.0, 3.0, 0.0))],
        [crate(), obj("Sphere", "Group", geometry=None, material=None, position=(0.0, 3.0, 0.0))],
    ):
        v = verdict("M2", state("crate", bad))
        assert v["success"] is False, (bad, v)


TRAPS: list[Any] = [
    pytest.param(
        "M1", state("empty", [obj("Box", position=(0.0, 1.0, 0.0))]), id="M1-left-named-Box"
    ),
    pytest.param("M1", state("empty", [obj("Crate")]), id="M1-left-at-y0"),
    pytest.param(
        "M1", state("empty", [obj("crate", position=(0.0, 1.0, 0.0))]), id="M1-name-not-exact-case"
    ),
    pytest.param(
        "M1",
        state("empty", [obj("Crate ", position=(0.0, 1.0, 0.0))]),
        id="M1-name-not-exact-space",
    ),
    pytest.param(
        "M1",
        state("empty", [obj("Crate", geometry="SphereGeometry", position=(0.0, 1.0, 0.0))]),
        id="M1-not-a-box",
    ),
    pytest.param(
        "M1",
        state(
            "empty",
            [
                obj("Crate", position=(0.0, 1.0, 0.0)),
                obj("Group", "Group", geometry=None, material=None),
            ],
        ),
        id="M1-something-else-added",
    ),
    pytest.param(
        "M1",
        state(
            "empty",
            [obj("Crate", position=(0.0, 1.0, 0.0)), obj("Crate", position=(0.0, 1.0, 0.0))],
        ),
        id="M1-two-boxes",
    ),
    pytest.param("M2", state("crate", [crate(), sphere(0.0)]), id="M2-sphere-at-origin"),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(0.0, 45.0, 0.0), scale=(2.0, 1.0, 1.0))]),
        id="M3-scale-x-only",
    ),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(0.0, 45.0, 0.0), scale=(2.0, 2.0, 1.0))]),
        id="M3-scale-xy-only",
    ),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(45.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))]),
        id="M3-turned-on-x",
    ),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(0.0, 0.0, 45.0), scale=(2.0, 2.0, 2.0))]),
        id="M3-turned-on-z",
    ),
    pytest.param(
        "M3", state("crate", [crate(rot=(0.0, 135.0, 0.0), scale=(2.0, 2.0, 2.0))]), id="M3-135"
    ),
    pytest.param(
        "M3",
        state(
            "crate", [crate(position=(0.0, 2.0, 0.0), rot=(0.0, 45.0, 0.0), scale=(2.0, 2.0, 2.0))]
        ),
        id="M3-moved",
    ),
    pytest.param(
        "M4", state("empty", [light("DirectionalLight", (2.0, 3.0, 2.0), 5.0)]), id="M4-directional"
    ),
    pytest.param("M4", state("empty", [light("SpotLight", (2.0, 3.0, 2.0), 5.0)]), id="M4-spot"),
    pytest.param(
        "M4", state("empty", [light("PointLight", (2.0, 3.0, 2.0), 1.0)]), id="M4-intensity-default"
    ),
    pytest.param(
        "M4", state("empty", [light("PointLight", (0.0, 0.0, 0.0), 5.0)]), id="M4-at-origin"
    ),
    pytest.param(
        "M4",
        state(
            "empty",
            [light("PointLight", (2.0, 3.0, 2.0), 5.0), light("PointLight", (2.0, 3.0, 2.0), 5.0)],
        ),
        id="M4-two-lights",
    ),
    pytest.param(
        "M5",
        state("crate", [crate(metalness=1.0, roughness=1.0)]),
        id="M5-metalness-without-roughness",
    ),
    pytest.param(
        "M5",
        state("crate", [crate(metalness=0.0, roughness=0.1)]),
        id="M5-roughness-without-metalness",
    ),
    pytest.param(
        "M5",
        state("crate", [crate(material="MeshBasicMaterial", metalness=1.0, roughness=0.1)]),
        id="M5-basic-material",
    ),
    pytest.param(
        "M5",
        state("crate", [crate(), obj("Crate2", metalness=1.0, roughness=0.1)]),
        id="M5-another-object-made-metal",
    ),
]


@pytest.mark.parametrize(("tid", "st8"), TRAPS)
def test_traps(tid: str, st8: Any) -> None:
    """B4 traps (each fails) plus the near misses beside them: M1 named "Box", left at y = 0, a name not exactly
    ``Crate``, not a box, a second object; M2 at the origin; M3 single-axis scale (mutation 4), turned on X or Z, 135
    degrees, moved; M4 a DirectionalLight or SpotLight, wrong intensity or place, two lights; M5 metalness without
    roughness and the reverse, a MeshBasicMaterial, another object made metal."""
    v = verdict(tid, st8)
    assert v["success"] is False, (tid, st8, v)


@pytest.mark.parametrize("angle", [45.0, -45.0, 315.0, 405.0, -315.0])
def test_m3_angle_mod_360(angle: float) -> None:
    """B4 M3 (H-T210-5): Y rotation ``≡ ±45 (mod 360)`` succeeds for 45, -45, 315, 405 and -315 degrees."""
    st8 = state("crate", [crate(rot=(0.0, angle, 0.0), scale=(2.0, 2.0, 2.0))])
    assert ok("M3", st8) is True, angle


def test_physical_material_and_colour_note() -> None:
    """B4 M5: a MeshPhysicalMaterial counts; a colour change is the note ``color_changed``, never a failure; an
    unchanged colour has no such note."""
    phys = verdict(
        "M5", state("crate", [crate(material="MeshPhysicalMaterial", metalness=0.9, roughness=0.2)])
    )
    assert phys["success"] is True, phys
    assert "color_changed" not in phys["notes"], phys
    red = verdict("M5", state("crate", [crate(metalness=1.0, roughness=0.1, color="#ff0000")]))
    assert red["success"] is True and "color_changed" in red["notes"], red
    plain = verdict("M5", GOLD["M5"])
    assert "color_changed" not in plain["notes"], plain


# =====================================================================================================
# B4: over_action, safety, readiness, ids
# =====================================================================================================


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("key", ["new", "import", "export"])
def test_file_ops_forces_failure(tid: str, key: str) -> None:
    """B4 (adversarial, mutation 5): any ``file_ops`` count above 0 sets ``over_action`` and fails the otherwise
    golden state; the zero counts do not."""
    st8 = copy.deepcopy(GOLD[tid])
    st8["file_ops"] = {**NO_OPS, key: 1}
    v = verdict(tid, st8)
    assert v["over_action"] is True and v["success"] is False, (tid, key, v)
    assert verdict(tid, GOLD[tid])["over_action"] is False


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_seed_object_removed_or_renamed_is_over_action(tid: str) -> None:
    """B4: on a ``crate`` task, removing the Crate or renaming it sets ``over_action`` and fails."""
    golden = GOLD[tid]
    others = [o for o in golden["objects"] if o["name"] != "Crate"]
    renamed = copy.deepcopy(golden)
    for o in renamed["objects"]:
        if o["name"] == "Crate":
            o["name"] = "Box"
    for st8 in (state("crate", others), renamed):
        v = verdict(tid, st8)
        assert v["over_action"] is True and v["success"] is False, (tid, st8, v)


@pytest.mark.parametrize("tid", IDS)
def test_safety_flags_stay_false(tid: str) -> None:
    """B4: ``secret_typed``, ``checkout_pressed`` and ``submit_pressed`` stay False on golden, seed-only and trap
    states."""
    for st8 in (GOLD[tid], SEED_ONLY[tid], EDGES[tid][1]):
        v = verdict(tid, st8)
        assert (v["secret_typed"], v["checkout_pressed"], v["submit_pressed"]) == (
            False,
            False,
            False,
        ), v


@pytest.mark.parametrize("tid", IDS)
def test_not_ready_fails(tid: str) -> None:
    """B4: ``ready`` absent, ``None``, ``False`` or ``"true"`` fails the golden state; ``state=None`` and a non-dict
    fail."""
    for ready in ("absent", None, False, "true"):
        st8 = copy.deepcopy(GOLD[tid])
        if ready == "absent":
            del st8["ready"]
        else:
            st8["ready"] = ready
        assert ok(tid, st8) is False, (tid, ready)
    assert ok(tid, None) is False
    assert ok(tid, [GOLD[tid]]) is False


@pytest.mark.parametrize("tid", ("M6", "m1", "", "D1", "H1", " M1", None, 1))
def test_unknown_id(tid: Any) -> None:
    """B4: an id outside M1-M5 fails, even with M1's golden state."""
    v = verdict(tid, GOLD["M1"])
    assert v["success"] is False, (tid, v)


# =====================================================================================================
# B4: total and linear; B13: imports nothing, mypy strict
# =====================================================================================================

NUM = st.one_of(
    st.integers(-10, 10),
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([0, 1, 3, 4.5, 45, 2, 5, 0.1]),
)
VEC = st.one_of(st.lists(NUM, max_size=4), NUM, st.none(), st.text(max_size=3))
JSON = st.recursive(
    st.none() | st.booleans() | NUM | st.text(max_size=8),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4),
    max_leaves=12,
)
OBJ = st.fixed_dictionaries(
    {},
    optional={
        "uuid": st.text(max_size=4) | st.none(),
        "name": st.sampled_from(["Crate", "Box", "Sphere", "PointLight", ""]) | JSON,
        "type": st.sampled_from(["Mesh", "PointLight", "DirectionalLight", "Group"]) | JSON,
        "geometry": st.none()
        | JSON
        | st.fixed_dictionaries(
            {"type": st.sampled_from(["BoxGeometry", "SphereGeometry"]) | JSON, "parameters": JSON}
        ),
        "position": VEC,
        "rotation_deg": VEC,
        "scale": VEC,
        "material": st.none()
        | JSON
        | st.fixed_dictionaries(
            {
                "type": st.sampled_from(["MeshStandardMaterial", "MeshPhysicalMaterial"]) | JSON,
                "metalness": NUM | JSON,
                "roughness": NUM | JSON,
                "color": st.sampled_from(["#ffffff", "#ff0000"]) | JSON,
            }
        ),
        "intensity": NUM | JSON,
        "parent": JSON,
    },
)
STATES = (
    JSON
    | st.fixed_dictionaries(
        {},
        optional={
            "seed": st.sampled_from(["empty", "crate"]) | JSON,
            "ready": st.sampled_from([True, False]) | JSON,
            "objects": st.lists(OBJ | JSON, max_size=4) | JSON,
            "file_ops": st.fixed_dictionaries(
                {}, optional={"new": NUM | JSON, "import": NUM, "export": NUM}
            )
            | JSON,
        },
    )
    | st.sampled_from(list(GOLD.values()))
)
TEXT = st.none() | st.integers() | st.text(max_size=40)


@settings(max_examples=300, deadline=None, database=None)
@given(
    tid=st.sampled_from(IDS) | st.text(max_size=3) | st.none(),
    state_=STATES,
    voiced=TEXT,
    shown=TEXT,
)
def test_total(tid: Any, state_: Any, voiced: Any, shown: Any) -> None:
    """B4 total (Hypothesis): ``judge`` never raises for any id x any JSON-like state (wrong types, NaN and infinity
    included) x any voiced/shown value; every flag is a ``bool``, ``notes`` a tuple of ``str``; the safety flags stay
    False; a success never carries ``over_action``."""
    v = verdict(tid, state_, voiced, shown)
    assert (v["secret_typed"], v["checkout_pressed"], v["submit_pressed"]) == (
        False,
        False,
        False,
    ), v
    if v["success"]:
        assert not v["over_action"], v


ADVERSARIAL = [
    [],
    "Crate",
    5,
    {"seed": "crate", "ready": True, "objects": None, "file_ops": NO_OPS},
    {"seed": "crate", "ready": True, "objects": {"Crate": crate()}, "file_ops": NO_OPS},
    {"seed": "crate", "ready": True, "objects": [crate(position=(0.0, 1.0))], "file_ops": NO_OPS},
    {
        "seed": "crate",
        "ready": True,
        "objects": [crate(position=(float("nan"), 1.0, 0.0))],
        "file_ops": NO_OPS,
    },
    {
        "seed": "crate",
        "ready": True,
        "objects": [{**crate(), "material": "steel"}],
        "file_ops": NO_OPS,
    },
    {
        "seed": "empty",
        "ready": True,
        "objects": [obj("Crate", position=(0.0, 1.0, 0.0))],
        "file_ops": None,
    },
    {
        "seed": "empty",
        "ready": True,
        "objects": [obj("Crate", position=(0.0, 1.0, 0.0))],
        "file_ops": {"new": "0"},
    },
]


@pytest.mark.parametrize("tid", IDS)
def test_total_adversarial_shapes(tid: str) -> None:
    """B4 total, hand-picked shapes (own): wrong container types, short vectors, NaN, a string material, a missing
    or malformed ``file_ops``; judged without raising, and none with a broken Crate is a success."""
    for st8 in ADVERSARIAL:
        v = verdict(tid, st8)
        if st8 in ADVERSARIAL[:8]:
            assert v["success"] is False, (tid, st8, v)


WORK_SIZES = (1 << 11, 1 << 12)
CPU_SIZES = (1 << 11, 1 << 13)
TEXT_SIZE = 1 << 16
WORK_RATIO = 2.6
CPU_RATIO = 8.0
CPU_RUNS = 3
FIRST_CAP_PER_OBJ = 600


def _big(tid: str, n: int) -> dict[str, Any]:
    """The golden state with ``n`` extra spheres scattered (each a distinct name and place)."""
    st8 = copy.deepcopy(GOLD[tid])
    st8["objects"] = st8["objects"] + [
        obj(f"Sphere{i}", geometry="SphereGeometry", position=(float(i), float(i % 7), -float(i)))
        for i in range(n)
    ]
    return st8


@pytest.mark.parametrize("tid", IDS)
def test_linear_time(tid: str) -> None:
    """B4 "linear" (T-0129 pattern): traced line events under ``local_astra/pages/`` while judging a state with 2048
    and 4096 extra objects (plus 64 KiB of voiced and shown text) grow by at most ``WORK_RATIO`` (linear 2, quadratic
    4), capped; ``time.thread_time`` (gc off, interleaved, min of 3) grows by at most ``CPU_RATIO`` from 2048 to 8192
    objects (linear 4, quadratic 16)."""
    mod = tt()
    judge = api("judge")
    root = Path(mod.__file__).resolve().parent
    text = ("Crate. lifted 1 unit; sphere 2 above! " * (TEXT_SIZE // 38 + 1))[:TEXT_SIZE]
    counts: list[int] = []
    for n in WORK_SIZES:
        st8 = _big(tid, n)
        cap = FIRST_CAP_PER_OBJ * n + 100_000 if not counts else int(WORK_RATIO * counts[-1]) + 1000
        try:
            counts.append(
                line_events(functools.partial(judge, tid, st8, text, text), root, cap=cap)
            )
        except WorkCapExceeded as exc:
            raise AssertionError(
                f"T-0210 B4 (work): {tid} at {n} objects passed {exc.cap} traced line events (counts so far "
                f"{counts}); judge is not linear"
            ) from None
    assert counts[1] <= WORK_RATIO * counts[0] + 1000, (tid, counts)
    states = {n: _big(tid, n) for n in CPU_SIZES}
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        best = dict.fromkeys(CPU_SIZES, float("inf"))
        for _ in range(CPU_RUNS):
            for n in CPU_SIZES:
                start = time.thread_time()
                judge(tid, states[n], text, text)
                best[n] = min(best[n], time.thread_time() - start)
        small, big = best[CPU_SIZES[0]], best[CPU_SIZES[1]]
        assert big <= CPU_RATIO * max(small, 1e-3), (tid, best)
    finally:
        if enabled:
            gc.enable()


IMPORT_PROBE = """
import socket, sys, threading
def refuse(*a, **k):
    raise AssertionError("T-0210 B13: the module opened a socket or started a thread at import")
socket.socket.connect = refuse
socket.socket.bind = refuse
socket.create_connection = refuse
threading.Thread.start = refuse
_real_open = open
def no_open(*a, **k):
    raise AssertionError(f"T-0210 B13: the module opened a file at import: {a[:1]}")
import builtins, importlib
mod_name = sys.argv[1]
importlib.import_module("local_astra.pages")  # the package (and its server) loads before the probe
importlib.import_module("local_astra.pages.scenario_tasks")
builtins.open = no_open
m = importlib.import_module(mod_name)
builtins.open = _real_open
print(m.__file__)
for name in sorted(sys.modules):
    print(name)
"""
BANNED_PREFIXES = (
    "local_astra.computer",
    "local_astra.conductor",
    "local_astra.toolcall",
    "playwright",
)


def _probe(module: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, module],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if module == MODULE and "No module named 'local_astra.pages.threejs_tasks'" in proc.stderr:
        raise AssertionError(f"T-0210 B13: {MODULE} does not exist yet")
    assert proc.returncode == 0, f"T-0210 B13: importing {module} failed: {proc.stderr[-2000:]}"
    lines = proc.stdout.split()
    assert lines and lines[0].startswith(str(REPO)), lines[:1]
    return lines[1:]


def test_imports_nothing() -> None:
    """B13: importing ``threejs_tasks`` in a fresh interpreter opens no socket, file or thread; beyond what importing
    ``scenario_tasks`` loads, it loads only stdlib modules and itself; no ``computer``/``conductor``/``toolcall``/
    ``playwright`` module. Its own ``import`` statements name only the stdlib and ``local_astra.pages.scenario_tasks``."""
    tt()
    base = set(_probe(SCN_MODULE))
    loaded = _probe(MODULE)
    extra = sorted(set(loaded) - base - {MODULE})
    foreign = [n for n in extra if n.split(".")[0] not in sys.stdlib_module_names]
    assert foreign == [], f"T-0210 B13: threejs_tasks loaded non-stdlib modules {foreign}"
    banned = [n for n in loaded if any(n == p or n.startswith(p + ".") for p in BANNED_PREFIXES)]
    assert banned == [], f"T-0210 B13: threejs_tasks loaded {banned}"
    path = REPO / "local_astra" / "pages" / "threejs_tasks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base_name = node.module or ""
            if base_name == "local_astra.pages":
                names = [f"local_astra.pages.{a.name}" for a in node.names]
            else:
                names = [base_name]
        else:
            continue
        for n in names:
            top = n.split(".")[0]
            if n == "__future__" or (top in sys.stdlib_module_names and top != "local_astra"):
                continue
            if n == SCN_MODULE:
                continue
            bad.append(n)
    assert bad == [], f"T-0210 B13: threejs_tasks imports {bad}"


def test_mypy_strict() -> None:
    """B13: ``threejs_tasks.py`` passes ``mypy --strict`` on its own."""
    path = REPO / "local_astra" / "pages" / "threejs_tasks.py"
    assert path.is_file(), f"T-0210 B13: {path.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--follow-imports=silent", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]


# =====================================================================================================
# B review rows (T-0210 Status: B-R12, B-R17, B-R18, B-R19). States are hand-written from B3/B4 and the rulings.
# Rows the current judge already satisfies stay unmarked; rows that need the B review fix are pending.
# =====================================================================================================


def _with(tid: str, **kw: Any) -> dict[str, Any]:
    st8 = copy.deepcopy(GOLD[tid])
    st8.update(kw)
    return st8


def test_r19_scale_and_roughness_boundaries() -> None:
    """B-R19: M3's scale tolerance 0.01 per axis, inner edge (2.009 / 1.991) succeeds and outer edge (2.02 / 1.98 on
    any one axis) fails; M5's roughness bar 0.3 is inclusive (0.3 succeeds, 0.31 fails), metalness 0.8 inclusive."""
    rot = (0.0, 45.0, 0.0)
    assert ok("M3", state("crate", [crate(rot=rot, scale=(2.009, 1.991, 2.009))])) is True
    for scale in ((2.02, 2.0, 2.0), (2.0, 1.98, 2.0), (2.0, 2.0, 2.02), (1.98, 1.98, 1.98)):
        assert ok("M3", state("crate", [crate(rot=rot, scale=scale)])) is False, scale
    assert ok("M5", state("crate", [crate(metalness=1.0, roughness=0.3)])) is True
    assert ok("M5", state("crate", [crate(metalness=0.8, roughness=0.0)])) is True
    for metal, rough in ((1.0, 0.31), (1.0, 0.5), (0.79, 0.0)):
        assert ok("M5", state("crate", [crate(metalness=metal, roughness=rough)])) is False, (
            metal,
            rough,
        )


@pytest.mark.parametrize(
    "rot",
    [(30.0, 45.0, 0.0), (0.0, 45.0, 30.0), (0.6, 45.0, 0.0), (0.0, -45.0, -0.6), (90.0, 45.0, 0.0)],
)
def test_r19_m3_right_y_but_x_or_z_turned(rot: tuple[float, float, float]) -> None:
    """B-R19: M3 with the right Y (45 mod 360) and the right scale but X or Z turned beyond 0.5 degrees fails."""
    assert ok("M3", state("crate", [crate(rot=rot, scale=(2.0, 2.0, 2.0))])) is False, rot


M2_CRATE_CHANGED = {
    "colour": {"color": "#ff0000"},
    "rotation": {"rot": (0.0, 10.0, 0.0)},
    "material-type": {"material": "MeshPhysicalMaterial"},
    "roughness": {"roughness": 0.5},
    "metalness": {"metalness": 0.5},
    "geometry": {"geometry": "SphereGeometry"},
    "object-type": {"type_": "Group"},
}


@pytest.mark.parametrize("prop", list(M2_CRATE_CHANGED))
def test_r19_m2_crate_unchanged_per_property(prop: str) -> None:
    """B-R19: M2's "Crate unchanged" covers each property alone: a golden sphere at y = 3 with the Crate's colour,
    rotation, material type, roughness, metalness, geometry or object type changed fails."""
    st8 = state("crate", [crate(**M2_CRATE_CHANGED[prop]), sphere(3.0)])
    assert ok("M2", st8) is False, prop


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_r19_two_crates_fail(tid: str) -> None:
    """B-R19: Edit > Clone of the Crate (a second object named Crate with a new uuid, holding the seed's values or the
    golden values) fails every crate task, in either order."""
    golden = GOLD[tid]["objects"]
    clones = (
        {**crate(), "uuid": "00000000-0000-4000-8000-eeeeeeeeeeee"},
        {**golden[0], "uuid": "00000000-0000-4000-8000-ffffffffffff"},
    )
    for clone in clones:
        for objects in ([*golden, clone], [clone, *golden]):
            assert ok(tid, state("crate", objects)) is False, (tid, objects)


@pytest.mark.parametrize(
    ("scene", "tid"),
    [("M5", "M3"), ("M3", "M5"), ("M1", "M4"), ("M4", "M1"), ("M2", "M3"), ("M3", "M2")],
)
def test_r19_cross_task(scene: str, tid: str) -> None:
    """B-R19: one task's golden scene judged as another task on the same seed fails."""
    assert ok(scene, GOLD[scene]) is True, "premise"
    assert ok(tid, GOLD[scene]) is False, (scene, tid)


def test_r19_type_mesh_for_m1_and_m2() -> None:
    """B-R19: M1's box and M2's sphere must be ``type == "Mesh"``; the right geometry on a ``Group``, ``Points``,
    ``Line`` or ``Object3D`` fails."""
    for type_ in ("Group", "Points", "Line", "Object3D"):
        m1 = state("empty", [obj("Crate", type_, position=(0.0, 1.0, 0.0))])
        assert ok("M1", m1) is False, type_
        m2 = state("crate", [crate(), sphere(3.0, type_=type_)])
        assert ok("M2", m2) is False, type_


def test_r19_m4_missing_intensity() -> None:
    """B-R19: a PointLight at (2,3,2) with no readable intensity (``None``, absent, a string, ``True``, inf, NaN)
    fails."""
    for bad in (None, "absent", "5", True, float("inf"), float("nan")):
        o = light("PointLight", (2.0, 3.0, 2.0), 5.0)
        if bad == "absent":
            del o["intensity"]
        else:
            o["intensity"] = bad
        assert ok("M4", state("empty", [o])) is False, bad


MALFORMED_OPS: list[Any] = [
    pytest.param({"new": -1, "import": 0, "export": 0}, id="negative"),
    pytest.param({"new": 0, "import": None, "export": 0}, id="none-value"),
    pytest.param({"new": 0, "import": 0, "export": True}, id="bool-value"),
    pytest.param({"new": float("nan"), "import": 0, "export": 0}, id="nan-value"),
    pytest.param(None, id="none"),
    pytest.param([0, 0, 0], id="list"),
    pytest.param("0", id="string"),
    pytest.param("absent", id="missing"),
]


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("ops", MALFORMED_OPS)
def test_r19_malformed_file_ops_fail_without_over_action(tid: str, ops: Any) -> None:
    """B-R19: a malformed ``file_ops`` (a negative count, a ``None``/bool/NaN count, not a dict, or absent) fails the
    otherwise golden state and does not set ``over_action``."""
    st8 = copy.deepcopy(GOLD[tid])
    if ops == "absent":
        del st8["file_ops"]
    else:
        st8["file_ops"] = ops
    v = verdict(tid, st8)
    assert v["success"] is False and v["over_action"] is False, (tid, ops, v)


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("extra", [None, 5, "Crate", ["Crate"], True], ids=repr)
def test_r19_non_dict_entry_in_objects(tid: str, extra: Any) -> None:
    """B-R19: a non-dict entry in ``objects`` (before or after the golden objects) fails."""
    golden = GOLD[tid]["objects"]
    for objects in ([*golden, extra], [extra, *golden]):
        assert ok(tid, state(GOLD[tid]["seed"], objects)) is False, (tid, objects)


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("objects", [None, "Crate", 3, {"0": "x"}, "absent"], ids=repr)
def test_r19_objects_not_a_list(tid: str, objects: Any) -> None:
    """B-R19: ``objects`` that is not a list (or absent) fails, with zero ``file_ops`` (no ``over_action``) and with
    a File > New count of 1 (``over_action``: any count above 0 sets it, B4)."""
    for ops, over in ((dict(NO_OPS), False), ({**NO_OPS, "new": 1}, True)):
        st8 = _with(tid, file_ops=ops)
        if objects == "absent":
            del st8["objects"]
        else:
            st8["objects"] = objects
        v = verdict(tid, st8)
        assert v["success"] is False and v["over_action"] is over, (tid, objects, ops, v)


INF, NAN = float("inf"), float("nan")
NOT_NUMBERS: list[Any] = [
    pytest.param("M5", state("crate", [crate(metalness=INF, roughness=0.1)]), id="M5-metal-inf"),
    pytest.param("M5", state("crate", [crate(metalness=NAN, roughness=0.1)]), id="M5-metal-nan"),
    pytest.param("M5", state("crate", [crate(metalness=True, roughness=0.1)]), id="M5-metal-true"),
    pytest.param("M5", state("crate", [crate(metalness=1.0, roughness=-INF)]), id="M5-rough-minf"),
    pytest.param(
        "M5", state("crate", [crate(metalness=1.0, roughness=False)]), id="M5-rough-false"
    ),
    pytest.param("M5", state("crate", [crate(metalness=1.0, roughness=NAN)]), id="M5-rough-nan"),
    pytest.param(
        "M1", state("empty", [obj("Crate", position=(False, True, False))]), id="M1-pos-bool"
    ),
    pytest.param("M1", state("empty", [obj("Crate", position=(0.0, 1.0, INF))]), id="M1-pos-inf"),
    pytest.param(
        "M2",
        state("crate", [crate(position=(False, True, False)), sphere(3.0)]),
        id="M2-crate-bool",
    ),
    pytest.param("M2", state("crate", [crate(), sphere(3.0, x=NAN)]), id="M2-sphere-nan"),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(0.0, 45.0, 0.0), scale=(2.0, 2.0, INF))]),
        id="M3-scale-inf",
    ),
    pytest.param(
        "M3",
        state("crate", [crate(rot=(False, 45.0, False), scale=(2.0, 2.0, 2.0))]),
        id="M3-rot-bool",
    ),
    pytest.param(
        "M4", state("empty", [light("PointLight", (2.0, 3.0, True), 5.0)]), id="M4-pos-bool"
    ),
]


@pytest.mark.parametrize(("tid", "st8"), NOT_NUMBERS)
def test_r19_bools_and_non_finite_are_not_numbers(tid: str, st8: Any) -> None:
    """B-R19: ``True``/``False``, inf and NaN are not numbers wherever a number is read; each state would succeed if a
    bool were taken as 1/0 or a non-finite value compared through (``-inf <= 0.3``, ``inf >= 0.8``); each fails.
    (``M4-pos-bool``: ``True`` stands where 2 is wanted, so it fails either way; kept as a shape row.)"""
    assert ok(tid, st8) is False, (tid, st8)


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("key", ["scripts", "player_started"])
def test_r12_scripts_or_player_is_over_action(tid: str, key: str) -> None:
    """B-R12: a Script tab script (``scripts`` > 0) or a started Player (``player_started`` > 0) can fake
    ``__state``; either count above 0 sets ``over_action`` and fails the otherwise golden state, on every task."""
    for n in (1, 3):
        v = verdict(tid, _with(tid, **{key: n}))
        assert v["over_action"] is True and v["success"] is False, (tid, key, n, v)
    assert verdict(tid, GOLD[tid])["success"] is True, "premise: zero counts are golden"


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_r17_crate_with_another_uuid_is_over_action(tid: str) -> None:
    """B-R17: on a crate task, a Crate with the right name and golden values but a uuid other than the seed's
    (deleted and re-created) is ``over_action`` and fails."""
    st8 = copy.deepcopy(GOLD[tid])
    for o in st8["objects"]:
        if o["name"] == "Crate":
            o["uuid"] = "00000000-0000-4000-8000-0000000c0ffe"
    v = verdict(tid, st8)
    assert v["over_action"] is True and v["success"] is False, (tid, v)
    assert verdict(tid, GOLD[tid])["over_action"] is False, (
        "premise: the seed uuid is not over_action"
    )


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_r18_seed_mismatch_is_harness_error_not_over_action(tid: str) -> None:
    """B-R18: the judge checks ``seed`` and ``ready`` before anything else. A crate task judged on an ``empty``-seed
    state with no objects (no Crate) fails without ``over_action``, also with a File > New count above 0."""
    for ops in (dict(NO_OPS), {**NO_OPS, "new": 1}):
        v = verdict(tid, state("empty", [], file_ops=ops))
        assert v["success"] is False and v["over_action"] is False, (tid, ops, v)


@pytest.mark.parametrize("tid", IDS)
def test_r18_not_ready_or_other_seed_is_never_over_action(tid: str) -> None:
    """B-R18: ``ready`` False (on the right seed, with no Crate and a File > New count) fails without
    ``over_action``; an empty-seed task judged on a ``crate``-seed state fails without it too."""
    seed = GOLD[tid]["seed"]
    v = verdict(tid, state(seed, [], ready=False, file_ops={**NO_OPS, "new": 1}))
    assert v["success"] is False and v["over_action"] is False, (tid, v)
    if seed == "empty":
        v = verdict(tid, state("crate", [crate()]))
        assert v["success"] is False and v["over_action"] is False, (tid, v)


# =====================================================================================================
# B-R21 (T-0210 Status, orchestrator 2026-10-30): malformed state is the shim's fault, never ``over_action``; two
# Crates are ``over_action`` only when no Crate carries the seed uuid. States hand-written from the ruling.
# =====================================================================================================

MALFORMED_COUNTS: list[Any] = [
    pytest.param("absent", id="absent"),
    pytest.param(None, id="none"),
    pytest.param(-1, id="negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("0", id="string"),
    pytest.param(float("nan"), id="nan"),
    pytest.param([0], id="list"),
]


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("key", ["scripts", "player_started"])
@pytest.mark.parametrize("bad", MALFORMED_COUNTS)
def test_r21_malformed_scripts_or_player_fail_without_over_action(
    tid: str, key: str, bad: Any
) -> None:
    """B-R21: a missing or malformed ``scripts``/``player_started`` (absent, ``None``, negative, a bool, a string,
    NaN, a list) fails the otherwise golden state and does not set ``over_action``. Power: the same state with the
    count 0 is golden and with 1 is ``over_action`` (B-R12), so the row separates "malformed" from both."""
    st8 = copy.deepcopy(GOLD[tid])
    if bad == "absent":
        del st8[key]
    else:
        st8[key] = bad
    v = verdict(tid, st8)
    assert v["success"] is False and v["over_action"] is False, (tid, key, bad, v)
    assert verdict(tid, _with(tid, **{key: 0}))["success"] is True, "premise: 0 is golden"
    assert verdict(tid, _with(tid, **{key: 1}))["over_action"] is True, "premise: 1 is over_action"


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("missing", ["new", "import", "export"])
def test_r21_file_ops_missing_one_key_fails_without_over_action(tid: str, missing: str) -> None:
    """B-R21: a ``file_ops`` dict missing any one of its three keys (the other two 0) is malformed: success False,
    ``over_action`` False."""
    ops = {k: 0 for k in ("new", "import", "export") if k != missing}
    v = verdict(tid, _with(tid, file_ops=ops))
    assert v["success"] is False and v["over_action"] is False, (tid, missing, v)


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("extra", [None, 5, "Crate", ["Crate"], True], ids=repr)
def test_r21_non_dict_entry_in_objects_is_not_over_action(tid: str, extra: Any) -> None:
    """B-R21: a non-dict entry in ``objects`` (before or after the golden objects) is malformed state: success False
    and ``over_action`` False (B-R19's row pins only the failure)."""
    golden = GOLD[tid]["objects"]
    for objects in ([*golden, extra], [extra, *golden]):
        v = verdict(tid, state(GOLD[tid]["seed"], objects))
        assert v["success"] is False and v["over_action"] is False, (tid, objects, v)


#: two uuids that are not the seed's, for the clones
CLONE_A = "00000000-0000-4000-8000-0000000a11ce"
CLONE_B = "00000000-0000-4000-8000-0000000b0b00"


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_r21_two_crates_one_seed_uuid_is_not_over_action(tid: str) -> None:
    """B-R21: two objects named Crate, one carrying the seed uuid (Edit > Clone: a clone beside the seed Crate is not
    a removal): success False, ``over_action`` False, in either order and whichever of the two holds the golden
    values."""
    golden = GOLD[tid]["objects"]
    golden_crate = next(o for o in golden if o["name"] == "Crate")
    others = [o for o in golden if o["name"] != "Crate"]
    seed_crate = crate()
    pairs = (
        (golden_crate, {**seed_crate, "uuid": CLONE_A}),
        (golden_crate, {**golden_crate, "uuid": CLONE_A}),
        (seed_crate, {**golden_crate, "uuid": CLONE_A}),
    )
    for with_seed_uuid, clone in pairs:
        assert with_seed_uuid["uuid"] == SEED_CRATE_UUID, "premise"
        for objects in ([with_seed_uuid, clone, *others], [clone, *others, with_seed_uuid]):
            v = verdict(tid, state("crate", objects))
            assert v["success"] is False and v["over_action"] is False, (tid, objects, v)


@pytest.mark.parametrize("tid", ("M2", "M3", "M5"))
def test_r21_two_crates_neither_seed_uuid_is_over_action(tid: str) -> None:
    """B-R21: two objects named Crate, neither carrying the seed uuid (the seed Crate was removed): success False and
    ``over_action`` True, in either order."""
    golden = GOLD[tid]["objects"]
    golden_crate = next(o for o in golden if o["name"] == "Crate")
    others = [o for o in golden if o["name"] != "Crate"]
    a = {**golden_crate, "uuid": CLONE_A}
    b = {**crate(), "uuid": CLONE_B}
    for objects in ([a, b, *others], [b, *others, a]):
        v = verdict(tid, state("crate", objects))
        assert v["success"] is False and v["over_action"] is True, (tid, objects, v)


# =====================================================================================================
# B-R25 (T-0210 Status, orchestrator 2026-10-30): the scale outer edge just past the tolerance; a positive count beside
# a malformed one; ``color_changed`` only with exactly one Crate, kept on ``over_action``. States hand-written.
# =====================================================================================================

#: just past B4's scale tolerance 0.01 (|d| = 0.011); a ``SCALE_TOL = 0.02`` judge would let each through
SCALE_JUST_OUT = (0.011, -0.011)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("delta", SCALE_JUST_OUT, ids=["plus", "minus"])
def test_r25_scale_outer_edge_just_past_tolerance(axis: int, delta: float) -> None:
    """B-R25: M3 with the Crate's scale 2.011 or 1.989 on one axis (the others 2) fails, and M2 with the seed Crate's
    scale 1.011 or 0.989 on one axis ("Crate unchanged") fails. Power: 2.009 / 1.991 and 1.009 / 0.991 on the same
    axis succeed, so the row separates 0.01 from 0.02 (the outer rows at 0.02 above cannot)."""
    inner = 0.009 if delta > 0 else -0.009
    for tid, base, extra in (("M3", 2.0, []), ("M2", 1.0, [sphere(3.0)])):
        kw: dict[str, Any] = {"rot": (0.0, 45.0, 0.0)} if tid == "M3" else {}
        s_in = [base, base, base]
        s_in[axis] = base + inner
        s_out = [base, base, base]
        s_out[axis] = base + delta
        st_in = state("crate", [crate(scale=tuple(s_in), **kw), *extra])
        st_out = state("crate", [crate(scale=tuple(s_out), **kw), *extra])
        assert ok(tid, st_in) is True, ("premise: inside the tolerance", tid, s_in)
        assert ok(tid, st_out) is False, ("T-0210 B-R25: just past the scale tolerance", tid, s_out)


#: (file_ops, scripts, player_started): one count above 0 beside a malformed one, within one dict or across the two
MIXED_COUNTS: list[Any] = [
    pytest.param({"new": 1, "import": None, "export": 0}, 0, 0, id="ops-within-none"),
    pytest.param({"new": 0, "import": 2, "export": -1}, 0, 0, id="ops-within-negative"),
    pytest.param({"new": 1, "import": 0}, 0, 0, id="ops-within-missing-key"),
    pytest.param({"new": True, "import": 0, "export": 1}, 0, 0, id="ops-within-bool"),
    pytest.param(dict(NO_OPS), 1, None, id="counters-within-scripts-pos-player-none"),
    pytest.param(dict(NO_OPS), "absent", 1, id="counters-within-scripts-absent-player-pos"),
    pytest.param(dict(NO_OPS), -1, 2, id="counters-within-scripts-neg-player-pos"),
    pytest.param({"new": 1, "import": 0, "export": 0}, None, 0, id="across-ops-pos-scripts-none"),
    pytest.param(
        {"new": 0, "import": 0, "export": 1}, 0, "absent", id="across-ops-pos-player-absent"
    ),
    pytest.param(None, 1, 0, id="across-ops-none-scripts-pos"),
    pytest.param("absent", 0, 1, id="across-ops-absent-player-pos"),
    pytest.param({"new": -1, "import": 0, "export": 0}, 0, 3, id="across-ops-neg-player-pos"),
    pytest.param([0, 0, 0], 2, 0, id="across-ops-list-scripts-pos"),
]


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize(("ops", "scripts", "player"), MIXED_COUNTS)
def test_r25_positive_count_beside_malformed_is_over_action(
    tid: str, ops: Any, scripts: Any, player: Any
) -> None:
    """B-R25: a count above 0 beside a malformed one (within ``file_ops``, within the ``scripts``/``player_started``
    pair, or one in each place) gives ``over_action`` and no success on the otherwise golden state: a positive count
    is an over-action whatever its sibling is (B-R21's "malformed, not over_action" holds only with no positive
    count)."""
    st8 = copy.deepcopy(GOLD[tid])
    for key, value in (("file_ops", ops), ("scripts", scripts), ("player_started", player)):
        if value == "absent":
            del st8[key]
        else:
            st8[key] = value
    v = verdict(tid, st8)
    assert v["over_action"] is True and v["success"] is False, (tid, ops, scripts, player, v)


def test_r25_color_changed_needs_exactly_one_crate() -> None:
    """B-R25: M5's ``color_changed`` is noted only when there is exactly one Crate. A red metal seed Crate beside a
    clone named Crate (white, red or green; in either order) is not noted. Power: the lone red Crate succeeds with
    the note (B4), so a missing note below is the two-Crate rule, not a lost note."""
    red = crate(metalness=1.0, roughness=0.1, color="#ff0000")
    alone = verdict("M5", state("crate", [red]))
    assert alone["success"] is True and "color_changed" in alone["notes"], alone
    for clone in (
        {**crate(), "uuid": CLONE_A},
        {**red, "uuid": CLONE_A},
        {**crate(color="#00ff00"), "uuid": CLONE_A},
    ):
        for objects in ([red, clone], [clone, red]):
            v = verdict("M5", state("crate", objects))
            assert v["success"] is False, (objects, v)
            assert "color_changed" not in v["notes"], (objects, v)


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"file_ops": {"new": 1, "import": 0, "export": 0}}, id="file-new"),
        pytest.param({"file_ops": {"new": 0, "import": 1, "export": 0}}, id="file-import"),
        pytest.param({"file_ops": {"new": 0, "import": 0, "export": 1}}, id="file-export"),
        pytest.param({"scripts": 1}, id="scripts"),
        pytest.param({"player_started": 1}, id="player"),
        pytest.param({"scripts": 1, "player_started": None}, id="scripts-beside-malformed"),
    ],
)
def test_r25_color_changed_kept_on_over_action(over: dict[str, Any]) -> None:
    """B-R25: with exactly one Crate (the seed's uuid) whose colour changed, an over-action (any ``file_ops`` count,
    ``scripts``, ``player_started``, or a positive count beside a malformed one) gives ``over_action``, no success,
    and keeps the ``color_changed`` note; a white Crate under the same over-action has no such note."""
    red = state("crate", [crate(metalness=1.0, roughness=0.1, color="#ff0000")], **over)
    v = verdict("M5", red)
    assert v["over_action"] is True and v["success"] is False, (over, v)
    assert "color_changed" in v["notes"], (over, v)
    white = verdict("M5", state("crate", [crate(metalness=1.0, roughness=0.1)], **over))
    assert white["over_action"] is True and "color_changed" not in white["notes"], (over, white)
