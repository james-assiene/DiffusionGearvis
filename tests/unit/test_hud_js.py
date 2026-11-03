"""Runs the page's JavaScript unit layer (``node --test tests/hud``) from the fast suite.

T-0008 criteria 1-12 are mostly page behaviour; the parts that can be tested without a browser
live in ``tests/hud/*.test.mjs`` and are run here so a Python-only ``make check`` still sees them.
Node is a dev prerequisite (18.19.1 on this box, ``docs/tasks/T-0008.md`` Regime); the page
itself needs no node — it is buildless ES modules served over http.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
JS_TESTS = Path(__file__).resolve().parents[1] / "hud"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None,
    reason="node >= 18.19 is a dev prerequisite for the HUD JS unit layer "
    "(docs/tasks/T-0008.md Regime: node v18.19.1); `node --test tests/hud` could not run",
)


def test_node_test_runner_is_new_enough_for_the_test_runner_flag():
    """``node --test`` needs node >= 18; the box pins 18.19.1 (T-0008 Regime)."""
    out = subprocess.run([str(NODE), "--version"], capture_output=True, text=True, timeout=60)
    major = int(out.stdout.strip().lstrip("v").split(".")[0])
    assert major >= 18, f"node --test needs node >= 18.19, found {out.stdout.strip()}"


def test_the_hud_javascript_unit_tests_pass():
    """Criteria 1-12 (the DOM-free half): protocol, throttle, state, phash, ws url, panels."""
    files = sorted(p.name for p in JS_TESTS.glob("*.test.mjs"))
    assert files, "tests/hud/*.test.mjs is the JS unit layer of T-0008 and must not be empty"
    proc = subprocess.run(
        [str(NODE), "--test", "tests/hud"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "`node --test tests/hud` failed (T-0008 criteria 1-12). Modules under hud/src/ must be "
        "importable in node with no DOM (implementation notes: keep DOM out of state.js, "
        f"protocol.js, frames.js, phash.js, ws.js).\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_the_denoise_module_is_part_of_the_node_layer():
    """T-0128c group B (*Tests to write*: this module "names the new node module only").

    ``tests/hud/denoise.test.mjs`` holds B2/B4-B9's node rows for ``hud/src/denoise/``. Its rows run
    through the T-0128c group B gate (strict xfail until ``docs/tasks/T-0128c.md`` records
    "group B: tests …, activated <sha>"), so ``node --test tests/hud`` above stays at exit 0 while
    they are pending and runs them for real once B is activated. This row only makes sure the file
    is in the directory the runner reads, so a rename cannot quietly drop B's node rows.
    """
    files = {p.name for p in JS_TESTS.glob("*.test.mjs")}
    assert "denoise.test.mjs" in files, (
        "T-0128c B: tests/hud/denoise.test.mjs is the node half of the denoising view's tests and "
        f"must sit in tests/hud/ where `node --test tests/hud` finds it; found {sorted(files)}"
    )


def test_the_renderer_core_modules_are_part_of_the_node_layer():
    """T-0164 group A (*Owned files*: this module "names the two new node modules only").

    ``tests/hud/ui_render.test.mjs`` (A2-A13) and ``tests/hud/ui_engine.test.mjs`` (A1, A9, A10) hold the
    node rows of our own A2UI renderer, ``hud/src/ui/render.js`` and ``engine.js``. They run through the
    T-0164 group A gate (strict xfail until ``docs/tasks/T-0164.md`` records "group A: tests …, activated
    <sha>"), so ``node --test tests/hud`` above stays at exit 0 while they are pending. This row only keeps
    the two files where the runner reads them.
    """
    files = {p.name for p in JS_TESTS.glob("*.test.mjs")}
    missing = {"ui_render.test.mjs", "ui_engine.test.mjs"} - files
    assert not missing, (
        f"T-0164 A: {sorted(missing)} must sit in tests/hud/ where `node --test tests/hud` finds them; "
        f"found {sorted(files)}"
    )


def test_the_v2_cards_module_is_part_of_the_node_layer():
    """T-0184 group B (*Owned files*: this module "registers the new node module").

    ``tests/hud/ui_cards_v2.test.mjs`` holds B1, B2, B4-B7 and B11's node rows for the v2 cards. They
    run through the T-0184 group B gate (strict xfail until ``docs/tasks/T-0184.md``'s Status records
    "group B: tests …, activated <sha>"), so ``node --test tests/hud`` stays at exit 0 while they are
    pending. This row only keeps the file where the runner reads it.
    """
    files = {p.name for p in JS_TESTS.glob("*.test.mjs")}
    assert "ui_cards_v2.test.mjs" in files, (
        "T-0184 B: tests/hud/ui_cards_v2.test.mjs must sit in tests/hud/ where `node --test "
        f"tests/hud` finds it; found {sorted(files)}"
    )


def test_the_tools_pane_module_is_part_of_the_node_layer():
    """T-0162 group B (*Owned files*: this module "names the new node module only").

    ``tests/hud/tools.test.mjs`` holds B-1 .. B-5's node rows for ``hud/src/tools/``. They run
    through the T-0162 group B gate (strict xfail until ``docs/tasks/T-0162.md`` records
    "group B: tests …, activated <sha>"), so ``node --test tests/hud`` above stays at exit 0 while
    they are pending. This row only keeps the file where the runner reads it.
    """
    files = {p.name for p in JS_TESTS.glob("*.test.mjs")}
    assert "tools.test.mjs" in files, (
        "T-0162 B: tests/hud/tools.test.mjs is the node half of the Tools pane's tests and must "
        f"sit in tests/hud/ where `node --test tests/hud` finds it; found {sorted(files)}"
    )


def test_the_model_thoughts_module_is_part_of_the_node_layer():
    """T-0180 group C (*Owned files*: this module registers the new node module).

    ``tests/hud/thoughts.test.mjs`` holds C1-C6's node rows for the "Model thoughts" disclosure
    (``hud/src/panels/thoughts.js``). They run through the T-0180 group C gate (strict xfail until
    ``docs/tasks/T-0180.md``'s Status records "group C: tests …, activated <sha>"), so ``node --test
    tests/hud`` stays at exit 0 while they are pending. This row only keeps the file where the
    runner reads it.
    """
    files = {p.name for p in JS_TESTS.glob("*.test.mjs")}
    assert "thoughts.test.mjs" in files, (
        "T-0180 C: tests/hud/thoughts.test.mjs must sit in tests/hud/ where `node --test "
        f"tests/hud` finds it; found {sorted(files)}"
    )
