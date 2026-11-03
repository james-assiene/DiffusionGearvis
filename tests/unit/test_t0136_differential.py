"""T-0136 B-5/B-7 (negative): with ``action_outcome`` off (the keyword absent) sessions equal BASE_B's in every mode,
the label-on live paths included; with it on, the settled payload is byte-identical and the bodies differ only by
the clause; the frozen paths are byte-identical over the closed range; the label constants and tool shas are
unchanged (docs/tasks/T-0136.md).

- **The differential** is T-0134's harness, imported, not copied (``test_t0134_differential._capture``: T-0111g's
  off/``dry_run`` turns and the loop, T-0131's live scenario, T-0132's live ReGround turn, the executor's action
  strings), plus two label-on live scenarios this module adds, because T-0134's capture never turns the label on
  and the clause lives in the label: (a) the label with the page extent over decodable frames with large changes
  (two clicks, a report, a toolless turn); (b) the T-0151 overview frame with the label on. Run in a subprocess over
  ``git archive`` BASE_B (``local_astra`` from the archive, origin checked; ``tests`` from this tree) and in-process.
  Green at BASE_B and must stay green (no ``pending`` marker).
- **The payload with the keyword on** (activated for group B, red until implemented): the same scenario (a) on and off: every step request's
  continuation tail and every ``tool.result`` record are identical, and each body with the clause removed equals the
  off body.
- **Byte identity** over ``BASE_B..TIP_B`` (two literals, never the working tree). ``TIP_B`` equals BASE_B until
  implementation is committed; the test-author or the orchestrator bumps it to the last implementation commit.

Reviewer mutations aimed at here: 8 (the payload changed: ``test_payload_identical_with_keyword_on``), 10 (the
default ``True``: every T-0134 live scenario then raises or carries a clause, ``test_off_equals_base``). Mutation 9
(``pass False`` when off) is group C's (cli wiring); the session side of it is ``test_t0136_session.
test_explicit_false_is_off``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, n = 1.
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
from tests.unit import test_t0134_differential as t0134
from tests.unit._live_browser_support import (
    FakeLiveBrowser,
    feedback_rig,
    live_feedback,
    live_frame,
)
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins

#: T-0136 group B's BASE: 2366600 (branch T-0136-b-outcome-session, from T-0136-a2-outcome-detector at 2366600,
#: "T-0136: human ruling after the A2 review"). A literal, never a moving ref.
BASE_B = "236660079cd339dbec1ee08390c37c81d7db5485"
#: The implementation tip; equals BASE_B until implementation is committed. Bumped by the test-author or the
#: orchestrator to the last implementation commit; at the merge it closes to that commit.
TIP_B = "07e72c32ce75d4dcf4c75ad463f6df87c75fe2ee"  # 07e72c3, the B implementation (was BASE_B)
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
CLAUSE_STEM = "Compared with the previous screenshot"

#: B-7's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/toolcall",
    "local_astra/computer",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/pages",
    "services",
    "scripts",
)


# --------------------------------------------------------------------------------------------- label-on scenarios


def _scenes() -> Any:
    from tests.unit import test_t0136_session as s

    return s


async def _label_extent(
    tmp: Path, extra: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Any]:
    """(a): label + extent, decodable frames (white -> badge -> badge + corner), click, click, report; then a
    toolless turn. Returns the normalised sections and the rig."""
    from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0121_differential import _with_history
    from tests.unit.test_t0131_differential import _apply_guidance

    s = _scenes()
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser([s.W0, live_frame("t2")], post_frames=[s.BADGE, s.BADGE_CORNER])
    ext = {"above_px": 0.0, "below_px": 1600.0, "viewport_px": 800.0}
    browser.extents = {f.phash: dict(ext) for f in (s.W0, s.BADGE, s.BADGE_CORNER)}
    kw: dict[str, Any] = {"live_feedback": live_feedback(screen_label=True, page_extent=True)}
    kw.update(extra or {})
    async with feedback_rig(
        tmp,
        browser,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
        session_kwargs=kw,
        machine_cls=_PolicyMachine,
    ) as rig:
        _apply_guidance(rig)
        rig.llm.script_steps([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 2, turn="t1")
        await t0107a._until_turns(rig, 1, "the label-on loop turn")
        rig.llm.script = [answer("It is a small shop. ")]
        await rig.ask("what is on the page")
        await t0107a._until_turns(rig, 2, "the label-on toolless turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path), rig


async def _overview(tmp: Path) -> dict[str, Any]:
    """(b): the T-0151 overview frame with the label on: an overview click, a live click, a report."""
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0121_differential import _with_history
    from tests.unit.test_t0131_differential import _apply_guidance

    s = _scenes()
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser([s.W0], post_frames=[s.BADGE, s.BADGE_CORNER])
    browser.overview_image = s.OV
    browser.page_metrics = dict(s.DEALS)
    kw = {
        "live_feedback": live_feedback(screen_label=True, page_extent=False),
        "overview_frame": True,
    }
    async with feedback_rig(
        tmp,
        browser,
        script=[answer(tsl.S1), tool(s.OV_CLICK)],
        session_kwargs=kw,
        machine_cls=_PolicyMachine,
    ) as rig:
        _apply_guidance(rig)
        rig.llm.script_steps([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 2, turn="t1")
        await t0107a._until_turns(rig, 1, "the overview-frame turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE_B tree."""
    out = await t0134._capture(tmp / "t0134")
    out["label_extent"] = (await _label_extent(tmp / "label_extent"))[0]
    out["overview"] = await _overview(tmp / "overview")
    return out


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0136_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_B)
    tmp = tmp_path_factory.mktemp("t0136_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_B)
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
        f"origin proof: the base run imported local_astra from {data['origin']}"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


async def test_off_equals_base(base: dict[str, Any], tmp_path: Path) -> None:
    """B-7 (off, the default; no keyword): T-0134's sections (off/``dry_run``/live, the loop and toolless turns, live
    ReGround, the executor's action strings) and the two label-on live scenarios equal BASE_B's section for section
    (machine steps, HUD messages, prompts, request bodies, session-log records, turn lines, history); no turn line
    gains ``action_outcome``, no ``live_label`` an ``outcome``, no body the clause."""
    now = observed(await _capture(tmp_path))  # T-0155 O-8: the persona sentence only
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            t0134._same(
                base["t0111g"][scenario][section],
                now["t0111g"][scenario][section],
                f"{scenario} {section}",
            )
    for name in ("live", "label_extent", "overview"):
        for section in t0107a.SECTIONS:
            t0134._same(base[name][section], now[name][section], f"{name} {section}")
    for section in ("log", "records", "bodies", "pass2"):
        t0134._same(
            base["live_reground"][section], now["live_reground"][section], f"reground {section}"
        )
    t0134._same(base["executor"], now["executor"], "executor actions")
    for name in ("live", "label_extent", "overview", "live_reground"):
        for rec in now[name]["records"]:
            assert "action_outcome" not in rec.get("extras", {}), rec.get("extras")
        blob = json.dumps(now[name])
        assert CLAUSE_STEM not in blob, f"T-0136 B-7: the clause with the keyword off ({name})"
        assert '"outcome": {' not in blob, (
            f"T-0136 B-4: an outcome record with the keyword off ({name})"
        )
    assert "Page beyond this screen" in json.dumps(now["label_extent"]["requests"]), (
        "premise: scenario (a) ran with the label and the extent"
    )


async def test_payload_identical_with_keyword_on(tmp_path: Path) -> None:
    """B-5/B-7 (reviewer mutation 8, the payload changed; red until implemented): scenario (a) with ``action_outcome=True``
    and without: every step request's continuation tail (the settled payloads, ``screenshot`` key included) and
    every ``tool.result`` record are identical; each step body with the clause removed equals the off body; the on
    run placed at least one clause (premise)."""
    import re

    from tests.unit.test_t0134_label import all_bodies

    _, off = await _label_extent(tmp_path / "off")
    _, on = await _label_extent(tmp_path / "on", {"action_outcome": True})
    for k in (2, 3):
        assert tsl.tail(tsl.step_request(on, k)) == tsl.tail(tsl.step_request(off, k)), (
            f"T-0136 B-5: the step-{k} continuation (settled payload) changed with the keyword on"
        )
    assert on.kind("tool.result") == off.kind("tool.result"), (
        "T-0136 B-5: tool.result records changed"
    )
    on_bodies, off_bodies = all_bodies(on), all_bodies(off)
    assert json.dumps(on_bodies).count(CLAUSE_STEM) >= 1, "premise: the on run placed a clause"
    clause = re.compile(r" Compared with the previous screenshot: [a-z ()]+\.")
    cleaned = json.loads(clause.sub("", json.dumps(on_bodies)))
    assert cleaned == off_bodies, "T-0136 B-7: the bodies differ beyond the clause"


def test_label_constants_unchanged() -> None:
    """B-7: ``LABEL_NOW``/``LABEL_UPDATED``/``LABEL_UNCHANGED`` are T-0134's texts (typed), ``LIVE_TOOL_GUIDANCE``
    hashes to 768a973b..., ``tools_v1.json`` to bbfc53cc... and ``tools_v1_overview.json`` to 8048bdb7..., in the
    working tree and at both ends of the range."""
    from tests.unit._live_browser_support import session_mod

    s = session_mod()
    assert s.LABEL_NOW == "This screenshot is the browser screen now."
    assert s.LABEL_UPDATED == "This screenshot is the browser screen now, after your last action."
    assert s.LABEL_UNCHANGED == (
        "This screenshot is the browser screen now; your last action did not change it."
    )
    gr.require_commits(BASE_B, TIP_B)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    root = Path(__file__).resolve().parents[2] / "local_astra/toolcall"
    for name, sha in (("tools_v1.json", V1_SHA), ("tools_v1_overview.json", OV_SHA)):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        for rev in (BASE_B, TIP_B):
            blob = gr._git("show", f"{rev}:local_astra/toolcall/{name}")
            assert blob.returncode == 0
            assert hashlib.sha256(blob.stdout.encode()).hexdigest() == sha, (rev, name)


def test_frozen_paths_over_closed_range() -> None:
    """B-7: the frozen paths are byte-identical over ``BASE_B..TIP_B`` (two literals, never the working tree) and
    every path exists at BASE_B. Power: the same predicate sees 82da380..02ccff5's lock change."""
    gr.require_commits(BASE_B, TIP_B)
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B[:7]} is not an ancestor of {TIP_B[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_B, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at BASE_B"
        )
    changed = gr.changed(BASE_B, TIP_B, *FROZEN)
    assert changed == [], f"T-0136 B-7: frozen paths differ in {BASE_B[:7]}..{TIP_B[:7]}: {changed}"
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, "services"), "power"
