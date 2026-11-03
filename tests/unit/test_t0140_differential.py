"""T-0140 A11 and A12 (negative): with the overview not offered the sessions equal BASE_A2's; the frozen paths are
byte-identical over each group's closed range; ``LIVE_TOOL_GUIDANCE`` and ``tools_v1.json`` keep their shas; docs/02
gains one §5 paragraph (docs/tasks/T-0140.md).

- **The differential** is T-0134's harness, imported, not copied (``test_t0134_differential._capture``: T-0111g's
  off/``dry_run`` turns, T-0131's live scenario, T-0132's live ReGround turn, the executor's action strings), run in a
  subprocess over ``git archive`` BASE_A2 (origin checked) and in-process. Green at BASE_A2 and must stay green.
- **Byte identity**: group A1 over ``bd5f661..8c553b3`` (merge-base and second parent of ``Merge T-0140 group A1``
  2dd9258); group A2 over ``BASE_A2..TIP_IMPL``. ``TIP_IMPL`` starts at BASE_A2 (no implementation yet) and is bumped by
  whoever commits implementation; at the merge it closes to the last implementation commit.
- ``test_doc02_delta_t0140`` (A11) reads docs/02 at ``TIP_IMPL``: red until the paragraph is committed and TIP bumped.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, n = 1.
"""

from __future__ import annotations

import hashlib
import json
import re
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

#: T-0140 group A2's BASE: main at 52d2452 (A1 merged at 2dd9258). A literal, never a moving ref.
BASE_SHA = "52d2452"
#: The implementation tip: 71c24b5 (A2's implementation and docs/02 delta; was BASE 52d2452). Bumped by whoever
#: commits implementation; at the merge it closes to the last implementation commit.
TIP_IMPL = "71c24b5"
#: Group A1's closed range: merge-base 2dd9258^1 2dd9258^2, and 2dd9258^2
A1_BASE = "bd5f66183e24aa4354045a69efc2139fd7ed0385"
A1_TIP = "8c553b3"
HEADER = "**Spec delta (T-0140"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"

#: A12's list, written out from the criterion, plus "Not owned"
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/degeneracy.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/thought_call.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/payloads",
    "local_astra/perception",
    "local_astra/pages",
    "hud",
    "services",
    "pyproject.toml",
    "uv.lock",
    "docs/cd/calibration-bfcl.json",
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0135_thought_call_ab.py",
    "tests/cd/unit/test_toolcall_live_parity.py",
)
#: A2's owned files, untouched by group A1
A2_OWNED = (
    "local_astra/conductor/session.py",
    "local_astra/toolcall/loop.py",
    "local_astra/cli.py",
    "local_astra/config.py",
    "docs/02-architecture.md",
)
NOT_OWNED_GLOBS = ("tests/unit/test_t0134_", "tests/unit/test_t0135_", "tests/unit/test_t0142_")


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
    tmp = tmp_path_factory.mktemp("t0140_base")
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


async def test_not_offered_equals_base(base: dict[str, Any], tmp_path: Path) -> None:
    """A12 (reviewer mutation 5's session half): with the overview not offered, off/``dry_run`` sessions, the live loop
    and toolless turns, a live ReGround turn and the executor's action strings equal BASE_A2's section for section;
    no turn line gains ``page_overview`` and no request record gains ``overview``."""
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
            assert "page_overview" not in rec.get("extras", {}), rec.get("extras")
    assert '"overview":' not in json.dumps(now["live"]["log"]), "A10: no overview request key"


def test_guidance_and_tools_sha_unchanged() -> None:
    """A12: ``LIVE_TOOL_GUIDANCE`` hashes to 768a973b… and ``tools_v1.json`` to bbfc53cc… in the working tree and at
    both ends of A2's range."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    p = Path(__file__).resolve().parents[2] / "local_astra/toolcall/tools_v1.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == V1_SHA
    for rev in (BASE_SHA, TIP_IMPL):
        blob = gr._git("show", f"{rev}:local_astra/toolcall/tools_v1.json")
        assert (
            blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest() == V1_SHA
        ), rev


def _check_range(lo: str, hi: str, paths: tuple[str, ...]) -> None:
    gr.require_commits(lo, hi)
    assert gr.is_ancestor(lo, hi), f"{lo[:7]} is not an ancestor of {hi[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", lo, "--", *paths).stdout.split()
    for p in paths:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {lo[:7]}"
        )
    assert gr.changed(lo, hi, *paths, diff_filter="ADR") == []
    changed = gr.changed(lo, hi, *paths)
    assert changed == [], f"T-0140 A12: frozen paths differ in {lo[:7]}..{hi[:7]}: {changed}"
    names = gr._git("diff", "--name-only", lo, hi).stdout.split()
    hit = [n for n in names if n.startswith(NOT_OWNED_GLOBS)]
    assert hit == [], f"T-0140 not owned: {hit}"


@pytest.mark.parametrize("group", ["A1", "A2"])
def test_frozen_paths_over_closed_range(group: str) -> None:
    """A12 and "Not owned": the frozen paths (and the T-0134/T-0135/T-0142 test modules) are byte-identical over each
    group's closed range, two literals, never the working tree; group A1 also left A2's files alone. Power: the same
    predicate sees 82da380..02ccff5's lock change."""
    if group == "A1":
        _check_range(A1_BASE, A1_TIP, FROZEN + A2_OWNED)
    else:
        _check_range(BASE_SHA, TIP_IMPL, FROZEN)
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN), "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0140() -> None:
    """A11: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0140``, naming the tool file and R-p,
    the judge's ``tool_set``, the full-page capture, ``--page-overview``, the placement before the current frame, both
    A8 reasons, the turn-line key, that inner scroll containers are not covered, ``position: fixed``, the headed flash
    and the O-A1-2 ``vh`` behaviour; removing it leaves docs/02 equal to BASE_A2's paragraph for paragraph. Keywords,
    never wording. Read at ``TIP_IMPL``."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    base_paras = t7d.paragraphs(_doc02_at(BASE_SHA))
    doc = _doc02_at(TIP_IMPL)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0140 A11: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "tools_v1_overview.json": "tools_v1_overview.json" in para,
        "R-p": "R-p" in para,
        "tool_set": "tool_set" in para,
        "full page": "full_page" in low or "full-page" in low or "full page" in low,
        "--page-overview": "--page-overview" in para,
        "before the current frame": "before the current frame" in low,
        "no screen this turn": "no screen this turn" in para,
        "no image slot": "no image slot for an overview" in para,
        "page_overview key": "page_overview" in para,
        "inner scroll containers": "scroll container" in low,
        "position: fixed": "fixed" in low,
        "flash": "flash" in low,
        "vh / moved": re.search(r"\bvh\b", para) is not None and "moved" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0140 A11: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == base_paras, "T-0140 A11: docs/02 changes only by the one delta paragraph"
