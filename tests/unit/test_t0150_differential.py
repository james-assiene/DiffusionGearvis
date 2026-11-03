"""T-0150 A9 and A10 (negative): with ``auto_overview`` off the sessions equal BASE_A's; the frozen paths are
byte-identical over the closed range; ``LIVE_TOOL_GUIDANCE``, ``tools_v1.json`` and ``tools_v1_overview.json`` keep
their shas; docs/02 gains the §5 paragraph and the §4 sentence (docs/tasks/T-0150.md).

- **The differential** is T-0134's harness, imported, not copied (``test_t0134_differential._capture``: T-0111g's
  off/``dry_run`` turns and the loop, T-0131's live scenario, T-0132's live ReGround turn, the executor's action
  strings; sections include the HUD messages, request bodies, session-log records and turn lines), run in a subprocess
  over ``git archive`` BASE_A (origin checked) and in-process. Green at BASE_A and must stay green.
- **Byte identity** over ``BASE_A..TIP_IMPL``. ``TIP_IMPL`` starts at BASE_A (no implementation yet) and is bumped by
  whoever commits implementation (the T-0140 pattern); at the merge it closes to the last implementation commit.
- ``test_docs_delta`` reads docs/02 at ``TIP_IMPL``: red until the paragraph is committed and ``TIP_IMPL`` bumped.

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

#: T-0150 group A's BASE: main at b9a6174 (the plan and H-T150-1..3 approved). A literal, never a moving ref.
BASE_SHA = "b9a6174"
#: The implementation tip; equals BASE_SHA until implementation is committed. Bumped by whoever commits
#: implementation; at the merge it closes to the last implementation commit.
TIP_IMPL = "a411a8f"
HEADER = "**Spec delta (T-0150).**"
S4_ANCHOR = "no epsilon transitions)"
S4_SENTENCE = (
    "Purity is relative to the injected segmenter; the session calls `step` exactly once per event."
)
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"

#: A10's list, written out from the criterion
FROZEN = (
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/tools.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
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
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0150_base")
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
        f"origin proof: the base run imported local_astra from {data['origin']}"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


async def test_off_equals_base(base: dict[str, Any], tmp_path: Path) -> None:
    """A10 (reviewer mutation 10's half, the key written when off): with ``auto_overview`` off (the default; no keyword)
    off/``dry_run``/live sessions, the live loop and toolless turns, a live ReGround turn and the executor's action
    strings equal BASE_A's section for section; no turn line gains ``auto_overview`` and no live request record gains
    ``overview``."""
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
            assert "auto_overview" not in rec.get("extras", {}), rec.get("extras")
    assert '"overview":' not in json.dumps(now["live"]["log"]), (
        "A5: no overview request key when off"
    )


def test_frozen_shas() -> None:
    """A10: ``LIVE_TOOL_GUIDANCE`` hashes to 768a973b..., ``tools_v1.json`` to bbfc53cc... and ``tools_v1_overview.json``
    to 8048bdb7..., in the working tree and at both ends of the range (content pins: these hold forever)."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    root = Path(__file__).resolve().parents[2] / "local_astra/toolcall"
    for name, sha in (("tools_v1.json", V1_SHA), ("tools_v1_overview.json", OV_SHA)):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        for rev in (BASE_SHA, TIP_IMPL):
            blob = gr._git("show", f"{rev}:local_astra/toolcall/{name}")
            assert (
                blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest() == sha
            ), (
                rev,
                name,
            )


def test_frozen_paths_over_closed_range() -> None:
    """A10: the frozen paths are byte-identical over ``BASE_A..TIP_IMPL`` (two literals, never the working tree), and
    every path exists at BASE_A. Power: the same predicate sees 82da380..02ccff5's lock change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), f"{BASE_SHA} is not an ancestor of {TIP_IMPL}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {BASE_SHA}"
        )
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], f"T-0150 A10: frozen paths differ in {BASE_SHA}..{TIP_IMPL}: {changed}"
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, "services"), "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_docs_delta() -> None:
    """A9: docs/02 §5 holds exactly one paragraph starting ``**Spec delta (T-0150).**`` that records the step-1
    prefix-cache effect (a re-prefill), the S4/S5-only reach with the measured heights (1220 and 800), and the headed
    flash; §4 holds H-T148-2's sentence verbatim, right after "...no epsilon transitions)". Keywords, never wording.
    Read at ``TIP_IMPL``."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    doc = _doc02_at(TIP_IMPL)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0150 A9: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "auto_overview": "auto_overview" in para or "--auto-overview" in para,
        "prefix cache": "prefix" in low and "cache" in low,
        "re-prefill": "re-prefill" in low or "reprefill" in low,
        "S4": "S4" in para,
        "S5": "S5" in para,
        "1220": "1220" in para or "1,220" in para,
        "800": "800" in para,
        "flash": "flash" in low,
        "headed": "headed" in low,
        "AUTO_OVERVIEW_TIMEOUT_MS": "AUTO_OVERVIEW_TIMEOUT_MS" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0150 A9: the docs/02 delta does not record {missing}"
    s4 = t7d.section(doc, "## 4. ")
    at = s4.find(S4_ANCHOR)
    assert at >= 0, "premise: the §4 anchor"
    after = s4[at + len(S4_ANCHOR) :].lstrip(" .")
    assert after.startswith(S4_SENTENCE), (
        f"T-0150 A9 (H-T148-2): the §4 sentence right after the anchor; found {after[:160]!r}"
    )
    assert doc.count(S4_SENTENCE) == 1
