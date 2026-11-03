"""T-0134 A8 (negative): with ``live_feedback=None`` and ``echo="pixels"`` (the defaults) sessions and executor action
strings equal BASE c29a8d9's in every mode, ``live`` included; the frozen paths are byte-identical over a closed
range; ``LIVE_TOOL_GUIDANCE`` and ``tools_v1.json`` keep their shas (docs/tasks/T-0134.md).

Green at BASE and must stay green (no ``pending`` marker): until the implementation lands the working tree *is*
BASE's code, and the byte-identity rows compare two fixed commits.

- **The differential** is T-0131's harness, imported, not copied: the base run is a subprocess over ``git archive
  c29a8d9`` (T-0114's ``_base_tree`` with its ``BASE_SHA`` monkeypatched; ``local_astra`` first on ``sys.path``, the
  origin printed and checked), once per module. Sections: T-0111g's ``_capture`` (off and ``dry_run``: tool turns,
  loop turns, a tools-off turn), T-0131's live scenario (a click, a scroll, a report, then a toolless turn), T-0132's
  live ReGround turn, and the executor's action strings and ``resolved`` (minus the salted ``frame_after`` phash) for
  a click, two types, a scroll and a key over T-0111c's fake driver. No new keyword is passed anywhere.
- **Byte identity** (A8's list): ``BASE_SHA..TIP_IMPL``, two literals, never the working tree. ``TIP_IMPL`` is the
  branch's last commit touching non-test files: b8dd882 (the rulings; no implementation yet), so the row holds
  trivially until the implementation lands. **TIP moves**: whoever commits implementation bumps it, and at the merge
  it closes to the last implementation commit. ``cli.py``/``config.py`` are T-0133's: if the branch merges main after
  T-0133, the merger re-derives BASE (``merge-base``) as the rules say.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, n = 1,
deterministic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0131_differential as t0131
from tests.unit import test_t0132_differential as t0132
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit.test_t0121_differential import _assert_equal

#: T-0134's BASE (docs/tasks/T-0134.md, "Branch ... from main at c29a8d9"). A literal, never a moving ref.
BASE_SHA = "c29a8d92d7859323e4e45c75eb744cc53639de97"
#: The implementation tip: acb0db3 (group A's last implementation commit, the repair's live_label record; before it
#: 04c48ab, the Makefile line, and 3437fc0). Bumped by whoever commits implementation (was b8dd882, the rulings).
TIP_IMPL = "acb0db3fe841f8c2cffb3e7cd2a775bf0339d1b5"
#: A8: sha256 of ``LIVE_TOOL_GUIDANCE`` (the spec gives ``768a973b…``; the full value computed at BASE)
GUIDANCE_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"

#: A8's list, written out from the criterion
FROZEN = (
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/perception/policy.py",
    "local_astra/pages",
    "hud",
    "services",
    "local_astra/cli.py",
    "local_astra/config.py",
)


async def _executor_actions() -> list[dict[str, Any]]:
    """The executor over T-0111c's fake driver, defaults only: action, status and ``resolved`` (without the salted
    ``frame_after``) of a click, a type, a type with Enter, a scroll and a key."""
    from tests.unit.test_computer_executor import FakeDriver, computer_executor, inv, own_frame

    fake = FakeDriver()
    ex = computer_executor().BrowserToolExecutor(
        now_ms=lambda: 1, viewport=fake.viewport, driver=fake
    )
    f = await own_frame(ex)
    out = []
    calls = [
        ("click_at", {"x": 500, "y": 250}),
        ("type_at", {"x": 120, "y": 640, "text": "hello", "press_enter": False}),
        ("type_at", {"x": 999, "y": 0, "text": "mugs", "press_enter": True}),
        ("scroll", {"direction": "down"}),
        ("press_key", {"combo": "Ctrl+A"}),
    ]
    for name, args in calls:
        frame = ex.latest_frame or f if name in ("click_at", "type_at") else None
        o = await ex.execute(inv(name, frame, **args))
        resolved = {k: v for k, v in o.resolved.items() if k != "frame_after"}
        out.append(
            {"status": o.status, "action": o.action, "reason": o.reason, "resolved": resolved}
        )
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the c29a8d9 tree."""
    return {
        "t0111g": await t0111g._capture(tmp / "t0111g"),
        "live": await t0131._live_turns(tmp / "live"),
        "live_reground": await t0132._live_reground(tmp / "live_reground"),
        "executor": await _executor_actions(),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0134_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0134_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


def _same(expected: Any, got: Any, what: str) -> None:
    _assert_equal(expected, got, f"T-0134 A8 {what} (base {BASE_SHA[:7]}; T-0121's helper):")


async def test_defaults_equal_c29a8d9(base: dict[str, Any], tmp_path: Path) -> None:
    """A8: with the defaults, off/``dry_run`` sessions (machine steps, HUD messages, prompts, request bodies,
    session-log records, turn lines, history), the live loop and toolless turns, a live ReGround turn, and the
    executor's action strings all equal c29a8d9's; no turn line gains ``live_feedback``."""
    now = observed(await _capture(tmp_path))  # T-0155 O-8: the persona sentence only
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _same(
                base["t0111g"][scenario][section],
                now["t0111g"][scenario][section],
                f"{scenario} {section!r}",
            )
    for section in t0107a.SECTIONS:
        _same(base["live"][section], now["live"][section], f"live {section!r}")
    for section in ("log", "records", "bodies", "pass2"):
        _same(
            base["live_reground"][section],
            now["live_reground"][section],
            f"live reground {section!r}",
        )
    _same(base["executor"], now["executor"], "executor actions")
    assert base["executor"][0]["action"] == "clicked at (640, 200)", "premise: BASE's pixel wording"
    for sec in (now["live"], now["live_reground"]):
        for rec in sec["records"]:
            assert "live_feedback" not in rec.get("extras", {}), rec.get("extras")
    assert now["live"]["records"][-1]["extras"].get("frame_source") == "window", "premise: live ran"


def test_frozen_paths_over_closed_range() -> None:
    """A8: the contracts, the fakes, the machine, the speech gate, the dry-run executor, the prompt builder,
    ``reground.py``, ``tools_v1.json``, ``policy.py``, the pages, the HUD, ``services/``, ``cli.py`` and
    ``config.py`` are byte-identical over ``BASE_SHA..TIP_IMPL`` (never the working tree), with no file added or
    removed. Power: the same predicate sees ``82da380..02ccff5``'s lock change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE_SHA, TIP_IMPL, *FROZEN, diff_filter="ADR") == []
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0134 A8: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN) == ["services/versions.lock"], "power"


def test_guidance_and_tools_sha_unchanged() -> None:
    """A8: ``LIVE_TOOL_GUIDANCE`` hashes to ``768a973b…`` (working tree and both ends) and ``tools_v1.json`` to
    ``bbfc53cc…`` at BASE and at TIP_IMPL."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert GUIDANCE_SHA.startswith("768a973b")
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == GUIDANCE_SHA
    for rev in (BASE_SHA, TIP_IMPL):
        blob = gr._git("show", f"{rev}:local_astra/toolcall/tools_v1.json")
        assert blob.returncode == 0, blob.stderr
        digest = hashlib.sha256(blob.stdout.encode()).hexdigest()
        assert digest == t0131.TOOLS_SHA and digest.startswith("bbfc53cc"), rev
        src = gr._git("show", f"{rev}:local_astra/conductor/prompt_builder.py")
        assert src.returncode == 0 and "LIVE_TOOL_GUIDANCE" in src.stdout, rev
