"""T-0179 group B, B-9 (negative: off and untripped are BASE) and B-10 (docs/02).

- **The differential** is T-0142b's harness, imported, not copied (``test_t0142b_differential._capture_guard`` and
  ``_compare``: T-0142's capture, the answer near misses and the out-of-scope paths; machine steps, HUD messages,
  prompts, request bodies, session-log records, turn lines, history and the ``machine.events()`` sequence). The base
  run is a subprocess over ``git archive BASE_B`` (T-0114's ``_base_tree``; origin checked), once per guard value.
  With ``degeneracy_guard=False``, and with the guard on and nothing tripping, every section equals BASE_B's. Both rows
  are green at BASE and must stay green; they kill a retry that fires without a trip or with the guard off, a
  ``degenerate_retries`` / ``system_clip`` key written on an untripped turn, and a clip on a clean turn.
- **Byte identity** over ``BASE_B..TIP_B`` (two literals, never the working tree): ``llm_client.py``,
  ``degeneracy.py``, ``speech_gate.py``, ``prompt_builder.py``, ``toolcall/``, ``computer/`` and
  ``tests/fakes/vllm_sse.py``. ``turn_machine.py`` changes only in the clip-kind wiring: read as "every hunk of its
  ``BASE_B..TIP_B`` diff names ``lost_thread``" (the test-author's reading; an empty diff passes). ``TIP_B`` starts at
  BASE (no implementation yet), so these rows hold trivially until it lands. **TIP moves**: whoever commits
  implementation bumps it; at the merge it closes to the last implementation commit.
- ``test_doc02_delta_t0179`` (B-10) reads docs/02 at ``TIP_B``: red until the paragraph is committed and TIP bumped.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps) where a scenario says so, n = 1, deterministic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0142b_differential as t42bd
from tests.unit._t0177_pins import observed as t0177_observed  # T-0177 A-P1 and B-P1

#: T-0179's BASE_A and BASE_B: main at aeac453f (T-0184's plan, after ``Merge T-0178``), the branch point. A literal.
BASE_B = "aeac453f3f0d7beccd5ac98641b0c6cbbf446af5"
#: The group B implementation tip (starts at BASE_B). Bumped by whoever commits implementation, closed at the merge.
#: ebaf4248 ("T-0179 B: one fresh degenerate retry ..."), then 6dc3feef ("T-0179 B-13a: no degenerate retry after a
#: published card"), then 8c5e0658 ("T-0179 B-13a: docs/02 retry conditions as one list"), the last group B
#: implementation commit
TIP_B = "8c5e065896ce1e883d4cf0cc5ccc2fc8f7fce78f"
GUARD = "degeneracy_guard"
HEADER = "**Spec delta (T-0179)"
FROZEN_B9 = (
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/degeneracy.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall",
    "local_astra/computer",
    "tests/fakes/vllm_sse.py",
)
#: T-0179 keys a clean or untripped turn must never carry
NEW_KEYS = ("degenerate_retries", "system_clip")

_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0142b_differential as m
out = pathlib.Path(sys.argv[1])
value = sys.argv[2] == "true"
sections = asyncio.run(m._capture_guard(out.parent / "run", value))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[bool, dict[str, Any]]:
    gr.require_commits(BASE_B)
    tmp = tmp_path_factory.mktemp("t0179_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_B)
        base_dir = tsl._base_tree(tmp / "base_tree")
    got: dict[bool, dict[str, Any]] = {}
    for value in (False, True):
        out = tmp / f"base_out_{value}" / "sections.json"
        out.parent.mkdir(parents=True)
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(out), "true" if value else "false"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"the base run ({GUARD}={value}) failed:\n{proc.stderr[-3000:]}"
        )
        data = json.loads(out.read_text())
        assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
            f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_B[:8]} tree"
        )
        got[value] = data["sections"]
    return got


def _compare(base: Any, now: Any, label: str, path: str = "") -> None:
    """Exact equality, section by section, so a failure names the first differing path. T-0142b's own ``_compare`` is
    not used: it normalises the observed side to its 0248cfd BASE (``_t0162c_pins.observed``), which BASE_B already
    postdates; here both sides are the same era and are compared raw."""
    if isinstance(base, dict) and isinstance(now, dict) and base.keys() == now.keys():
        for key in base:
            _compare(base[key], now[key], label, f"{path}/{key}")
        return
    if isinstance(base, list) and isinstance(now, list) and len(base) == len(now):
        for i, (b, n) in enumerate(zip(base, now, strict=True)):
            _compare(b, n, label, f"{path}[{i}]")
        return
    assert base == now, (
        f"{label}: differs from BASE {BASE_B[:8]} at {path}:\nbase {json.dumps(base)[:600]}\n"
        f"now  {json.dumps(now)[:600]}"
    )


def _no_new_keys(now: dict[str, Any]) -> None:
    dumped = json.dumps(now)
    for key in NEW_KEYS:
        assert f'"{key}"' not in dumped, f"T-0179 B-9: {key!r} on an untripped turn"
    assert "degenerate request: retrying once" not in dumped, "T-0179 B-9: a retry without a trip"
    assert "lost_thread" not in dumped, "T-0179 B-9: a clip on an untripped turn"


async def test_false_equal_base(base: dict[bool, dict[str, Any]], tmp_path: Path) -> None:
    """B-9: ``degeneracy_guard=False`` on every session gives exactly BASE_B's sections with ``False``. Green at
    BASE."""
    # T-0177 A-P1: the observed side only, A10's extras.vision_source == "browser" (never "shared") set aside
    now = t0177_observed(await t42bd._capture_guard(tmp_path, False))
    _compare(base[False], now, "T-0179 B-9 False")
    _no_new_keys(now)
    assert GUARD not in json.dumps(now)


async def test_on_without_trip_equal_base(base: dict[bool, dict[str, Any]], tmp_path: Path) -> None:
    """B-9: the guard on and nothing tripping gives exactly BASE_B's sections with the guard on, and every
    ``degeneracy_guard`` value is ``{"on": true, "tripped": 0}``. Green at BASE."""
    # T-0177 A-P1: the observed side only, A10's extras.vision_source == "browser" (never "shared") set aside
    now = t0177_observed(await t42bd._capture_guard(tmp_path, True))
    found = t42bd._guard_values(now)
    assert len(found) >= 10 and all(v == {"on": True, "tripped": 0} for v in found), found[:3]
    _compare(base[True], now, "T-0179 B-9 on, no trip")
    _no_new_keys(now)


def test_frozen_paths_over_closed_range() -> None:
    """B-9: the frozen paths are byte-identical over ``BASE_B..TIP_B`` (never the working tree), nothing added or
    removed under them. Power: the same predicate sees ``82da380..02ccff5``'s lock change."""
    gr.require_commits(BASE_B, TIP_B)
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B[:8]} is not an ancestor of {TIP_B[:8]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_B, "--", *FROZEN_B9).stdout.split()
    for p in FROZEN_B9:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE_B, TIP_B, *FROZEN_B9)
    assert changed == [], f"T-0179 B-9: frozen paths differ in {BASE_B[:8]}..{TIP_B[:8]}: {changed}"
    gr.require_commits("82da380", "02ccff5")
    assert gr.changed("82da380", "02ccff5", "services") == ["services/versions.lock"], "power"


def test_turn_machine_changes_only_the_clip_wiring() -> None:
    """B-9 / implementation notes: ``turn_machine.py`` changes over ``BASE_B..TIP_B`` only beside T-0178's clip
    emission, read as: every diff hunk names ``lost_thread``."""
    gr.require_commits(BASE_B, TIP_B)
    proc = gr._git("diff", "-U0", BASE_B, TIP_B, "--", "local_astra/conductor/turn_machine.py")
    assert proc.returncode == 0, proc.stderr
    hunks = re.split(r"(?m)^@@", proc.stdout)[1:]
    stray = [h for h in hunks if "lost_thread" not in h]
    assert stray == [], (
        f"T-0179 B-9: turn_machine.py hunks outside the clip-kind wiring: {stray[:2]}"
    )


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0179() -> None:
    """B-10: docs/02 §5 gains one paragraph starting ``**Spec delta (T-0179)`` recording the retry rule (once, nothing
    voiced), fresh sampling and why (no seed: HTTP 400, unseeded engine), the bounds (``llm_timeout`` shared, 3
    requests), the clip and its class (``lost_thread``, system audio, never history), the supersession of H-T142-1 and
    H-T142b-1, and the evidence (the 2026-10-27 session, t10). T-0142's and T-0142b's paragraphs stay verbatim; apart
    from T-0179's own delta paragraphs docs/02 equals BASE_B's paragraph for paragraph. Keywords, never wording."""
    gr.require_commits(BASE_B, TIP_B)
    base_paras = t7d.paragraphs(_doc02_at(BASE_B))
    kept = [
        p
        for p in base_paras
        if p.lstrip().startswith(("**Spec delta (T-0142)", "**Spec delta (T-0142b"))
    ]
    assert len(kept) == 2, f"premise: BASE has T-0142's and T-0142b's paragraphs; {len(kept)}"
    doc = _doc02_at(TIP_B)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0179 B-10: one §5 paragraph starting {HEADER!r} at {TIP_B[:8]}; {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "retry once": "retr" in low and "once" in low,
        "fresh sampling": "fresh" in low,
        "why: seed / 400 / unseeded": "seed" in low and ("400" in para or "unseeded" in low),
        "llm_timeout shared": "llm_timeout" in para,
        "3 requests bound": re.search(r"\b(3|three)\b", low) is not None,
        "lost_thread": "lost_thread" in para,
        "system audio / history": "history" in low and ("system" in low or "clip" in low),
        "H-T142-1": "H-T142-1" in para,
        "H-T142b-1": "H-T142b-1" in para,
        "supersedes": "supersed" in low,
        "evidence t10": "t10" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0179 B-10: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    for p in kept:
        assert p in rest, "T-0179 B-10: T-0142's and T-0142b's paragraphs stay verbatim"
    rest = [p for p in rest if not p.lstrip().startswith("**Spec delta (T-0179")]
    assert rest == base_paras, "T-0179 B-10: docs/02 changes only by T-0179's delta paragraphs"
