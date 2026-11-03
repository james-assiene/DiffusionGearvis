"""T-0142 A7 and A8 (negative): with the guard off the sessions equal BASE e1ae76e in every mode; with it on and no trip
they differ only by the A5 turn-line key; the frozen paths are byte-identical over a closed range; docs/02 gains one
§5 paragraph (docs/tasks/T-0142.md).

- **The differential** is T-0135's harness, imported, not copied (``test_t0135_differential._capture``: T-0111g's
  off/``dry_run`` tool, loop and tools-off turns, T-0131's live scenario, T-0132's live ReGround turn, T-0135's D-5
  turns), plus this task's near misses (``_near_miss``): a tools-off thought with 7 back-to-back ``the`` and a
  63-character dot run, a live step whose thought straddles 4 + 3 repeats across two canvases, and a live step
  carrying the thought commits of the most repetitive clean logged request in the fixture. The base run is a
  subprocess over ``git archive e1ae76e`` (T-0114's ``_base_tree``, ``BASE_SHA`` monkeypatched; origin checked).
  The guard is put on every session of a capture by wrapping ``Session.__init__`` (``setdefault``), so no harness
  is edited. Sections: machine steps (``machine.events()``), HUD messages, prompts, request bodies, session-log
  records, turn lines, history (``t0107a.SECTIONS``) and T-0132's ReGround sections.
- ``test_absent_equal_base`` is green at BASE and must stay green; ``test_false_equal_base`` and
  ``test_on_without_trip_differs_only_by_key`` are red until ``Session(degeneracy_guard=...)`` exists (and the key
  is written); the second kills a guard that trips at 7 (mutation 3) or judges per commit too wide.
- **Byte identity** (A8's list plus "This task does not own"): ``BASE_SHA..TIP_IMPL``, two literals, never the working
  tree. ``TIP_IMPL`` starts at BASE (no implementation yet), so the row holds trivially until it lands. **TIP moves**:
  whoever commits implementation bumps it; at the merge it closes to the last implementation commit.
- ``test_doc02_delta_t0142`` (A7) reads docs/02 at ``TIP_IMPL``: red until the paragraph is committed and TIP bumped.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps) where a scenario says so, n = 1, deterministic.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0135_differential as t0135
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool, tool_rig
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

#: T-0142's BASE: main at e1ae76e (after ``Merge T-0135`` and T-0134c), the branch point. A literal.
BASE_SHA = "e1ae76e36f70aa5404be7dbb14566296ac3fbc5a"
#: The implementation tip: 712fea3 on T-0142-impl (code and the A7 docs/02 paragraph; was BASE). Bumped by whoever
#: commits implementation.
TIP_IMPL = "712fea3c45eb2244688aa0ee426e324e225ec5cb"
HEADER = "**Spec delta (T-0142"
GUARD = "degeneracy_guard"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0142" / "thoughts.jsonl"

#: A8's list, written out from the criterion
FROZEN_A8 = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall",
    "local_astra/computer",
    "services",
)
#: "This task does not own", beyond A8
NOT_OWNED = (
    "local_astra/conductor/tool_executor.py",
    "local_astra/config.py",
    "hud",
    "scripts/gpu",
    "scripts/locked",
    "tests/unit/_live_browser_support.py",
)


def _clean_fixture_thoughts() -> list[Any]:
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    clean = [r for r in rows if r["class"] == "clean" and r["phase"] == "tool_step"]
    row = clean[0]  # the first repeat-2 tool step (the build script's order)
    return [
        thought(c["text"], c["canvas_idx"], open_=bool(c["thought_open"])) for c in row["thoughts"]
    ]


async def _near_miss(tmp: Path) -> dict[str, Any]:
    """Text one short of every rule, in tools off and live (by hand), plus a logged clean step."""
    tmp.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    async with tool_rig(tmp / "off", kit=None, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = [
            thought("Let me see" + " the" * 7 + " screen."),
            thought("Loading" + "." * 63 + " done."),
            answer("It shows a cart."),
        ]
        await rig.ask("what is on this screen")
        await t0107a._until_turns(rig, 1, "the near-miss tools-off turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out["off"] = _with_history(rig, tmp / "off", session_id, log_path)
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 6)]
    )
    async with live_rig(tmp / "live", browser, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
        rig.llm.script_steps(
            [
                thought("I look and" + " the" * 4, 0, open_=True),
                thought(" the" * 3 + " page moved.", 1, open_=True),
                answer(tsl.S2, 1),
                tool(tsl.CLICK2, canvas_idx=1),
            ],
            [*_clean_fixture_thoughts(), answer(tsl.S3)],
        )
        await rig.ask(tsl.QUESTION)
        await t0107a._until_turns(rig, 1, "the near-miss live turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out["live"] = _with_history(rig, tmp / "live", session_id, log_path)
    out["live"]["executed"] = [[i.call_id, dict(i.arguments)] for i in browser.invocations]
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the e1ae76e tree."""
    return {
        **await t0135._capture(tmp / "t0135"),
        "near_miss": await _near_miss(tmp / "near_miss"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0142_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0142_base")
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


def _session_mod() -> Any:
    return importlib.import_module("local_astra.conductor.session")


def _require_guard() -> None:
    params = inspect.signature(_session_mod().Session.__init__).parameters
    assert GUARD in params, (
        "T-0142 A3: Session(..., degeneracy_guard: bool = False) does not exist yet"
    )


async def _capture_with(tmp: Path, monkeypatch: pytest.MonkeyPatch, value: bool) -> dict[str, Any]:
    _require_guard()
    cls = _session_mod().Session
    real_init = cls.__init__

    @functools.wraps(
        real_init
    )  # keeps the signature visible to the rigs' seam checks (reground_rig)
    def init(self: Any, *a: Any, **k: Any) -> None:
        k.setdefault(GUARD, value)
        real_init(self, *a, **k)

    monkeypatch.setattr(cls, "__init__", init)
    return await _capture(tmp)


def _same(expected: Any, got: Any, what: str) -> None:
    _assert_equal(expected, got, f"T-0142 A8 {what} (base {BASE_SHA[:7]}; T-0121's helper):")


def _compare(base: dict[str, Any], now: dict[str, Any], label: str) -> None:
    now = observed(now)  # T-0155 O-8: the persona sentence only, on the observed side
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _same(
                base["t0111g"][scenario][section],
                now["t0111g"][scenario][section],
                f"{label} {scenario} {section!r}",
            )
    for name in ("live", "d5_live", "d5_dry_run"):
        for section in t0107a.SECTIONS:
            _same(base[name][section], now[name][section], f"{label} {name} {section!r}")
    for section in ("log", "records", "bodies", "pass2"):
        _same(
            base["live_reground"][section],
            now["live_reground"][section],
            f"{label} live reground {section!r}",
        )
    for name in ("off", "live"):
        for section in t0107a.SECTIONS:
            _same(
                base["near_miss"][name][section],
                now["near_miss"][name][section],
                f"{label} near miss {name} {section!r}",
            )
    _same(base["near_miss"]["live"]["executed"], now["near_miss"]["live"]["executed"], label)


def _strip_key(tree: Any, found: list[Any]) -> Any:
    if isinstance(tree, dict):
        if GUARD in tree:
            found.append(tree[GUARD])
        return {k: _strip_key(v, found) for k, v in tree.items() if k != GUARD}
    if isinstance(tree, list):
        return [_strip_key(v, found) for v in tree]
    return tree


def _turn_lines(sections: Any) -> int:
    n = 0
    if isinstance(sections, dict):
        for k, v in sections.items():
            n += len(v) if k == "records" and isinstance(v, list) else _turn_lines(v)
    return n


async def test_absent_equal_base(base: dict[str, Any], tmp_path: Path) -> None:
    """A8: with the keyword not passed, every scenario (off, ``dry_run``, live, ReGround, D-5, the near misses)
    equals e1ae76e's, and no record names the guard. Green at BASE."""
    now = await _capture(tmp_path)
    _compare(base, now, "absent")
    assert GUARD not in json.dumps(now), "T-0142 A5: no key without the guard"
    # premises: the near-miss turns ran the paths they name
    assert [c for c, _ in now["near_miss"]["live"]["executed"]] == ["t1-call", "t1-call-s2"]
    ends = [r["extras"].get("tool_loop_end") for r in now["near_miss"]["live"]["records"]]
    assert ends == ["final"], ends


async def test_false_equal_base(
    base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A8 (mutation 7): ``degeneracy_guard=False`` on every session gives exactly BASE's sections."""
    now = await _capture_with(tmp_path, monkeypatch, False)
    _compare(base, now, "False")
    assert GUARD not in json.dumps(now), "T-0142 A5: no key with the guard off"


async def test_on_without_trip_differs_only_by_key(
    base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A8 (mutation 3, a rule one short; a guard too wide): with the guard on and nothing tripping, every section
    equals BASE's once the ``degeneracy_guard`` key is removed, and that key is on every turn line with
    ``{"on": true, "tripped": 0}``."""
    raw = await _capture_with(tmp_path, monkeypatch, True)
    found: list[Any] = []
    now = _strip_key(raw, found)
    lines = _turn_lines(raw)
    assert lines >= 10, f"premise: the scenarios wrote turn lines; {lines}"
    assert found, "T-0142 A5: the guard on, turn lines carry degeneracy_guard"
    assert all(v == {"on": True, "tripped": 0} for v in found), [
        v for v in found if v != {"on": True, "tripped": 0}
    ][:3]
    records_keys = sum(
        1
        for sec in _iter_records(raw)
        for rec in sec
        if isinstance(rec.get("extras"), dict) and GUARD in rec["extras"]
    )
    assert records_keys == lines, (
        f"T-0142 A5: every turn line has the key ({records_keys} of {lines})"
    )
    _compare(base, now, "on, no trip")


def _iter_records(tree: Any) -> Any:
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == "records" and isinstance(v, list):
                yield v
            else:
                yield from _iter_records(v)


def test_frozen_paths_over_closed_range() -> None:
    """A8: the machine, the speech gate, ``llm_client.py``, the contracts, the fakes, the prompt builder,
    ``toolcall/``, ``computer/``, ``services/`` and every not-owned path (``tool_executor.py``, ``config.py``, the
    HUD, ``scripts/gpu``, ``scripts/locked``, ``_live_browser_support.py``) are byte-identical over
    ``BASE_SHA..TIP_IMPL`` (never the working tree), with no file added or removed under them. Power: the same
    predicate sees ``82da380..02ccff5``'s lock change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )
    frozen = FROZEN_A8 + NOT_OWNED
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *frozen).stdout.split()
    for p in frozen:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE_SHA, TIP_IMPL, *frozen, diff_filter="ADR") == []
    changed = gr.changed(BASE_SHA, TIP_IMPL, *frozen)
    assert changed == [], (
        f"T-0142 A8: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *frozen) == ["services/versions.lock"], "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0142() -> None:
    """A7: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0142``, recording the rule and its
    constants (3 tokens, 8 repeats, 64 characters), that only committed thought is judged, the abort and the turn's
    end, the pass-2 fallback, that there is no retry and the evidence counts (10 collapses, 1,976 clean); removing
    it leaves docs/02 equal to BASE's paragraph for paragraph. Keywords, never wording. Read at ``TIP_IMPL``."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    base_doc = _doc02_at(BASE_SHA)
    assert [p for p in t7d.paragraphs(base_doc) if p.lstrip().startswith(HEADER)] == [], "premise"
    doc = _doc02_at(TIP_IMPL)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0142 A7: one §5 paragraph starting {HEADER!r} in {TIP_IMPL[:7]}; found {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "MAX_UNIT_TOKENS / 3": "MAX_UNIT_TOKENS" in para and re.search(r"\b3\b", para) is not None,
        "MIN_REPEATS / 8": "MIN_REPEATS" in para and re.search(r"\b8\b", para) is not None,
        "MIN_CHAR_RUN / 64": "MIN_CHAR_RUN" in para and "64" in para,
        "committed thought": "committed thought" in low,
        "abort": "abort" in low,
        "LlmError / end": "llmerror" in low.replace("_", "").replace(" ", "")
        or "ends the turn" in low,
        "pass 2": "pass 2" in low or "pass-2" in low,
        "no retry": "retry" in low,
        "evidence 10": re.search(r"\b10\b", para) is not None,
        "evidence 1,976": "1,976" in para or "1976" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0142 A7: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0142 A7: docs/02 changes only by the one delta paragraph"
    )
