"""T-0151 A2-10 and A2-11 (negative): with the overview frame off the sessions equal BASE_A2's; the frozen paths are
byte-identical over the closed range; ``LIVE_TOOL_GUIDANCE``, ``tools_v1.json`` and ``tools_v1_overview.json`` keep
their shas; docs/02 §5 gains the ``**Spec delta (T-0151).**`` paragraph (docs/tasks/T-0151.md).

- **The differential** is T-0134's harness, imported, not copied (``test_t0134_differential._capture``: T-0111g's
  off/``dry_run`` turns and the loop, T-0131's live scenario, T-0132's live ReGround turn, the executor's action
  strings; sections include the HUD messages, request bodies, session-log records and turn lines), run in a subprocess
  over ``git archive`` BASE_A2 (origin checked) and in-process. Green at BASE_A2 and must stay green.
- **Off on a long page** (in-process, green at BASE_A2): a live session without the keyword on a 1.525-screen page
  takes no capture and reads no overview, and no record or line gains an ``overview_frame`` key.
- **Byte identity** over ``BASE_A2..TIP_A2``. ``TIP_A2`` starts at BASE_A2 (no implementation yet) and is bumped by the
  test-author or the orchestrator to the last implementation commit before review (the implementer never edits tests).
- ``test_docs_delta`` reads docs/02 at ``TIP_A2``: red until the paragraph is committed and ``TIP_A2`` bumped
  (activated for group A2: red until then). Its keywords are the test-author's reading of A2-10's list (see the report), never wording.

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
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0134_differential as t0134
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins

#: T-0151 group A2's BASE: main at 177c0a1 ("T-0151: group A1 merged; A2 next"), after ``Merge T-0151 group A1``
#: (e4ef5cf). A literal, never a moving ref.
BASE_A2 = "177c0a1"
#: The implementation tip; equals BASE_A2 until implementation is committed. Bumped by the test-author or the
#: orchestrator to the last implementation commit; at the merge it closes to that commit.
TIP_A2 = "0b1d5d9"
HEADER = "**Spec delta (T-0151).**"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"

#: A2-11's byte-identical list, written out from the criterion, plus the T-0150 and T-0140 session, cli and
#: differential modules ("unedited except the Q-2 re-pins"; no T-0150/T-0140 module needed a re-pin, see the report)
FROZEN = (
    "local_astra/toolcall",
    "local_astra/computer",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/degeneracy.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/perception",
    "local_astra/pages",
    "hud",
    "services",
    "tests/unit/test_t0150_session.py",
    "tests/unit/test_t0150_differential.py",
    "tests/unit/test_t0150_cli.py",
    "tests/unit/test_t0140_session.py",
    "tests/unit/test_t0140_differential.py",
    "tests/unit/test_t0140_cli.py",
    "tests/unit/test_t0140_executor.py",
    "scripts/gpu/t0150_auto_overview_ab.py",
    "scripts/locked/t0150_auto_overview_ab.sh",
    "tests/unit/test_t0150_ab_harness.py",
    "tests/eval/test_t0150_ab.py",
    "scripts/gpu/t0140_overview_ab.py",
    "scripts/locked/t0140_overview_ab.sh",
    "tests/unit/test_t0140_ab_harness.py",
    "tests/eval/test_t0140_ab.py",
)

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
    gr.require_commits(BASE_A2)
    tmp = tmp_path_factory.mktemp("t0151_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_A2)
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
    """A2-11 (off, the default; no keyword): off/``dry_run``/live sessions, the live loop and toolless turns, a live
    ReGround turn and the executor's action strings equal BASE_A2's section for section (request bodies, turn lines,
    session-log records, HUD messages); no turn line gains ``overview_frame`` and no request record gains it."""
    now = observed(await t0134._capture(tmp_path))  # T-0155 O-8: the persona sentence only
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            t0134._same(
                base["t0111g"][scenario][section],
                now["t0111g"][scenario][section],
                f"{scenario} {section}",
            )
    for section in t0107a.SECTIONS:
        t0134._same(base["live"][section], now["live"][section], f"live {section}")
    for section in ("log", "records", "bodies", "pass2"):
        t0134._same(
            base["live_reground"][section], now["live_reground"][section], f"reground {section}"
        )
    t0134._same(base["executor"], now["executor"], "executor actions")
    for sec in (now["live"], now["live_reground"]):
        for rec in sec["records"]:
            assert "overview_frame" not in rec.get("extras", {}), rec.get("extras")
    assert '"overview_frame"' not in json.dumps(now["live"]["log"]), (
        "A2-8: no overview_frame key when off"
    )


async def test_off_takes_no_capture_on_a_long_page(tmp_path: Path) -> None:
    """A2-11 (green at BASE_A2): a live session built without the keyword, on a 1.525-screen page, over a click turn
    and a toolless turn: no ``page_overview`` invocation reaches the browser, ``overview_frame_of`` and ``overview_of``
    are never called, the step-1 body carries the live frame, and no turn line or ``llm.request`` record has an
    ``overview_frame`` key, nor a ``tool.call`` record a ``frame_kind``."""
    from tests.unit import test_t0151_session as s

    browser = s.deals_browser(frames=(s.F0, s.G))
    async with s.rig_for(
        tmp_path, browser, on=False, script=[s.answer(s.S1), s.tool(s.TOWEL)]
    ) as rig:
        await s.run_turn(rig, ([s.answer(s.S3)],))
        await s.next_turn(rig, 2, "and now", [s.answer("Again. ")])
        assert s.image_urls(s.pass1_body(rig)) == [s.builder_url(s.F0)]
        lines = [json.loads(x) for x in rig.lines()]
    assert s.captures(browser) == [], browser.calls
    assert s.reads(browser, "overview_frame_of") == [] and s.reads(browser, "overview_of") == []
    assert all("overview_frame" not in x.get("extras", {}) for x in lines)
    assert not [d for d in rig.kind("llm.request") if "overview_frame" in d]
    assert "frame_kind" not in json.dumps(rig.kind("tool.call"))


def test_frozen_shas() -> None:
    """A2-11: ``LIVE_TOOL_GUIDANCE`` hashes to 768a973b..., ``tools_v1.json`` to bbfc53cc... and
    ``tools_v1_overview.json`` to 8048bdb7..., in the working tree and at both ends of the range (content pins: these
    hold forever)."""
    gr.require_commits(BASE_A2, TIP_A2)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    assert t0134.GUIDANCE_SHA.startswith("768a973b"), t0134.GUIDANCE_SHA
    root = Path(__file__).resolve().parents[2] / "local_astra/toolcall"
    for name, sha in (("tools_v1.json", V1_SHA), ("tools_v1_overview.json", OV_SHA)):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        for rev in (BASE_A2, TIP_A2):
            blob = gr._git("show", f"{rev}:local_astra/toolcall/{name}")
            assert (
                blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest() == sha
            ), (rev, name)


def test_frozen_paths_over_closed_range() -> None:
    """A2-11: the frozen paths are byte-identical over ``BASE_A2..TIP_A2`` (two literals, never the working tree), and
    every path exists at BASE_A2. Power: the same predicate sees 82da380..02ccff5's lock change."""
    gr.require_commits(BASE_A2, TIP_A2)
    assert gr.is_ancestor(BASE_A2, TIP_A2), f"{BASE_A2} is not an ancestor of {TIP_A2}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A2, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {BASE_A2}"
        )
    changed = gr.changed(BASE_A2, TIP_A2, *FROZEN)
    assert changed == [], f"T-0151 A2-11: frozen paths differ in {BASE_A2}..{TIP_A2}: {changed}"
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, "services"), "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_docs_delta() -> None:
    """A2-10: docs/02 §5 holds exactly one paragraph starting ``**Spec delta (T-0151).**`` that records (keywords, the
    test-author's reading of A2-10's list): the opt-in (``overview_frame``); the measured geometry and its regime (838,
    1220, Chromium) and the precision ("precision" or "coarser"); the 2.0 cap and the normal-frame fallback above it
    (``2.0`` and ``too_tall``); the exception to "never an own frame" ("own frame"); the B9 exemption ("B9"); the
    prefix-cache effect ("prefix", "cache", a re-prefill); inner scroll containers ("scroll container"); and the
    headed flash ("headed", "flash"). Read at ``TIP_A2``."""
    gr.require_commits(BASE_A2, TIP_A2)
    doc = _doc02_at(TIP_A2)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0151 A2-10: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "overview_frame": "overview_frame" in para or "--overview-frame" in para,
        "838": "838" in para,
        "1220": "1220" in para or "1,220" in para,
        "Chromium": "chromium" in low,
        "precision": "precision" in low or "coarser" in low,
        "2.0": "2.0" in para,
        "too_tall": "too_tall" in para,
        "own frame": "own frame" in low,
        "B9": "B9" in para,
        "prefix cache": "prefix" in low and "cache" in low,
        "re-prefill": "re-prefill" in low or "reprefill" in low,
        "scroll container": "scroll container" in low,
        "headed": "headed" in low,
        "flash": "flash" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0151 A2-10: the docs/02 delta does not record {missing}"
    assert doc.count(HEADER) == 1
