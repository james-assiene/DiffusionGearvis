"""T-0132 A5 (negative): in ``off`` and ``dry_run`` the printed ``--dry-run`` output, the request bodies, the turn lines
and the session-log records equal BASE a2d9ac6's; a Session built without the new keywords equals a2d9ac6's in
every mode, ``live`` included; the frozen paths are byte-identical over a closed range (docs/tasks/T-0132.md).

Green at BASE and must stay green (no ``pending`` marker): until the implementation lands the working tree *is*
BASE's code, and the byte-identity rows compare two fixed commits.

- **The differential** is T-0131's harness, imported, not copied: the base run is a subprocess over ``git archive
  a2d9ac6`` (T-0114's ``_base_tree`` with its ``BASE_SHA`` monkeypatched; ``local_astra`` first on ``sys.path``, the
  origin printed and checked), once per module. Sessions: T-0111g's ``_capture`` (tool turns, loop turns, a
  tools-off turn) and T-0131's live scenario (a click, a scroll, a report, then a toolless turn; no new keyword),
  plus one live ReGround turn (``reground=True, visual_tokens=1120``, an ``ok`` pass 2; no new keyword).
- **The printed ``--dry-run`` output** of ``astra session`` in ``off`` and ``dry_run`` (with and without
  ``--visual-tokens``, ``--no-reground``, ``--no-tool-loop``, and a ``tools = "live"`` profile overridden by
  ``--tools off``/``--tools dry-run``): stdout and the exit code, the temporary directory normalised; stderr too for
  every run that exits 0.
- **Reading (reported): B5 vs A5.** Group B adds ``budget``, ``images``, ``first_ms`` and ``tool_ms`` to every
  ``tool.reground`` record, also in a Session built without the new keywords. The live ReGround comparison therefore
  drops exactly those four keys from ``tool.reground`` records (and requires that nothing else was added); every
  other section is compared whole.
- **Byte identity** (A5's list): ``BASE_SHA..TIP_IMPL``, two literals, never the working tree. ``TIP_IMPL`` is the
  branch's last commit touching non-test files: 0155195 (group A's implementation; b5eeec7 before it), so the row holds trivially until the
  implementation lands. **TIP moves**: whoever commits implementation bumps it (as T-0131's did), and at the merge it
  closes to the last implementation commit.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, n = 1,
deterministic.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import _t0172_pins as t0172
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0131_differential as t0131
from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, reground_rig
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_t0121_differential import _assert_equal

#: T-0132's BASE (docs/tasks/T-0132.md, "Branch ... from main at a2d9ac6"). A literal, never a moving ref.
BASE_SHA = "a2d9ac660d6de3146006b15cdeae50d45d97c8ba"
#: The implementation tip: 5e6b78e (group B, branch ``T-0132-group-b-impl``). Bumped by whoever commits
#: implementation (was 0155195, group A; b5eeec7, the plan).
TIP_IMPL = "5e6b78e28fd0a9a6b20bd4d563579307dd86a1f1"
#: B5: the keys group B adds to ``tool.reground`` (the only allowed difference, see the docstring)
B5_KEYS = frozenset({"budget", "images", "first_ms", "tool_ms"})

#: A5's list, written out from the criterion
FROZEN = (
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/perception/policy.py",
    "services/vllm.sh",
    "services/versions.lock",
    "local_astra/computer",
    "local_astra/pages",
    "hud",
)
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
#: T-0183 re-scope (docs/tasks/T-0183.md, "Differentials against a BASE that has ``--visual-tokens``"; the T-0174 D4b
#: pattern on both sides): every case that set no budget passes ``--visual-tokens 1120`` on both sides (a2d9ac6
#: has the flag), since T-0183 makes 1120 the default in ``off``/``dry_run``; nothing else changes. The cases that
#: set a budget (``off-560``, ``dry_run-1120``, ``live-profile-560-off``) are unchanged.
VT1120 = ("--visual-tokens", "1120")
#: (case, flags, profile body) of the ``--dry-run`` comparison
DRY_RUNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("off", VT1120, ""),
    ("off-560", ("--visual-tokens", "560"), ""),
    ("off-no-reground", ("--no-reground", *VT1120), ""),
    ("off-reground-refused", ("--reground", *VT1120), ""),
    ("dry_run", ("--tools", "dry-run", *VT1120), ""),
    ("dry_run-1120", ("--tools", "dry-run", "--visual-tokens", "1120"), ""),
    ("dry_run-no-loop", ("--tools", "dry-run", "--no-tool-loop", *VT1120), ""),
    ("live-profile-off", ("--tools", "off", *VT1120), LIVE_PROFILE),
    ("live-profile-dry_run", ("--tools", "dry-run", *VT1120), LIVE_PROFILE),
    ("live-profile-560-off", ("--tools", "off"), LIVE_PROFILE + "visual_tokens = 560\n"),
)


# --------------------------------------------------------------------------------------------- the scenarios


def _dry_runs(tmp: Path) -> dict[str, Any]:
    """Each ``astra session --dry-run`` of ``DRY_RUNS``: stdout, stderr and exit code, ``tmp`` normalised."""
    from tests.unit import _tools_switch_support as T
    from tests.unit.test_cli_session import explode_clients, run_cli
    from tests.unit.test_cli_tools_live import Events, argv, fake_live

    out: dict[str, Any] = {}
    for case, flags, body in DRY_RUNS:
        where = tmp / case
        where.mkdir(parents=True, exist_ok=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
            T.patch_detok_load(mp, T.Calls(), where)
            explode_clients(mp)
            fake_live(mp, Events())
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = run_cli([*argv(where, *flags, config_body=body), "--dry-run"])
        out[case] = {
            "code": code,
            "stdout": stdout.getvalue().replace(str(where), "<tmp>"),
            "stderr": stderr.getvalue().replace(str(where), "<tmp>"),
        }
    return out


async def _live_reground(tmp: Path) -> dict[str, Any]:
    """One live ReGround turn (``reground=True, visual_tokens=1120``; no new keyword): a click whose pass 2 is ``ok``,
    then a report. The log records, turn lines, the step bodies and the pass-2 prompts, normalised."""
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [jpeg_frame("onset")], post_frames=[jpeg_frame("after-1"), jpeg_frame("after-2")]
    )
    pass2 = [
        answer("REGROUND-DIFF pass two. "),
        tool("<|tool_call>call:click_at{x:900,y:100}<tool_call|>"),
    ]
    async with reground_rig(
        tmp,
        browser,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
        session_kwargs={"visual_tokens": 1120},
    ) as rig:
        rig.llm.script_pass2(pass2)
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, str(Path(rig.log.path))
    per_process = {"log.started": {"pid": "<pid>"}, "log.stopped": {"bytes": "<bytes>"}}
    tree = {
        "log": [
            {"kind": r["kind"], "data": {**r["data"], **per_process.get(r["kind"], {})}}
            for r in rig.log_records
        ],
        "records": [json.loads(line) for line in rig.lines()],
        "bodies": list(rig.llm.bodies),
        "pass2": [p.model_dump(mode="json") for _, p in rig.llm.pass2_prompts],
    }
    text = (
        json.dumps(tree, sort_keys=True)
        .replace(log_path, "<log>")
        .replace(session_id, "<session>")
        .replace(str(tmp), "<tmp>")
    )
    out: dict[str, Any] = json.loads(text)
    for rec in out["records"]:
        regime = rec.get("regime")
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the a2d9ac6 tree."""
    return {
        "t0111g": await t0111g._capture(tmp / "t0111g"),
        "live": await t0131._live_turns(tmp / "live"),
        "live_reground": await _live_reground(tmp / "live_reground"),
        "dry_runs": _dry_runs(tmp / "dry_runs"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0132_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0132_base")
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
    _assert_equal(expected, got, f"T-0132 A5 {what} (base {BASE_SHA[:7]}; T-0121's helper):")


# --------------------------------------------------------------------------------------------- the rows


async def test_off_dry_run_equal_a2d9ac6(base: dict[str, Any], tmp_path: Path) -> None:
    """A5 (reviewer mutations 1, 2): in ``off`` and ``dry_run`` the printed ``--dry-run`` output (every case of
    ``DRY_RUNS``, a ``tools = "live"`` profile overridden by ``--tools`` included) and T-0111g's sessions (machine
    steps, HUD messages, prompts, request bodies, session-log records, turn lines, history) equal a2d9ac6's."""
    now = observed(_dry_runs(tmp_path / "dry_runs"))  # T-0155 O-8: the persona sentence only
    for case, _flags, _body in DRY_RUNS:
        b, n = base["dry_runs"][case], now[case]
        # T-0172 R11 pin move (B9): a dry_run --dry-run prints the default reask line last; off prints none
        if "dry_run" in case:
            n = {**n, "stdout": t0172.admit_reask_line(n["stdout"], "dry")}
        else:
            t0172.assert_no_reask(n["stdout"])
        assert (n["code"], n["stdout"]) == (b["code"], b["stdout"]), (
            f"T-0132 A5: astra session --dry-run ({case}) changed:\nbase {b}\nnow  {n}"
        )
        if b["code"] == 0:
            assert n["stderr"] == b["stderr"], (case, b["stderr"], n["stderr"])
    assert base["dry_runs"]["off"]["code"] == 0 and "tools=off" in base["dry_runs"]["off"]["stdout"]
    assert base["dry_runs"]["off-reground-refused"]["code"] == 2, (
        "premise: BASE refuses --reground in off"
    )
    sessions = observed(await t0111g._capture(tmp_path / "t0111g"))  # T-0155 O-8
    assert sessions["dry_run_lines"] == base["t0111g"]["dry_run_lines"]
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _same(
                base["t0111g"][scenario][section],
                sessions[scenario][section],
                f"{scenario} {section!r}",
            )


async def test_live_session_without_new_keywords_equal_a2d9ac6(
    base: dict[str, Any], tmp_path: Path
) -> None:
    """A5 (R-f: the default lives in ``cli``): a live Session built without the new keywords (T-0131's live
    scenario, and a ReGround turn at an explicit 1120) gives a2d9ac6's sections: no ``grounding`` key, no request
    body or record changes, except the four B5 keys on ``tool.reground`` records."""
    now = observed(
        await t0131._live_turns(tmp_path / "live")
    )  # T-0155 O-8: the persona sentence only
    for section in t0107a.SECTIONS:
        _same(base["live"][section], now[section], f"live {section!r}")
    assert all("grounding" not in r["extras"] for r in now["records"]), "no grounding key"
    rg = observed(await _live_reground(tmp_path / "live_reground"))  # T-0155 O-8
    b_rg = base["live_reground"]
    for rec in rg["log"]:
        if rec["kind"] == "tool.reground":
            extra = set(rec["data"]) - {
                k for r in b_rg["log"] if r["kind"] == "tool.reground" for k in r["data"]
            }
            assert extra <= B5_KEYS, f"T-0132 A5: tool.reground gained {sorted(extra - B5_KEYS)}"
            for k in B5_KEYS:
                rec["data"].pop(k, None)
    for section in ("log", "records", "bodies", "pass2"):
        _same(b_rg[section], rg[section], f"live reground {section!r}")
    regrounds = [r for r in rg["log"] if r["kind"] == "tool.reground"]
    assert len(regrounds) == 1 and regrounds[0]["data"]["outcome"] == "ok", (
        "premise: pass 2 ran, ok"
    )
    assert all("grounding" not in r["extras"] for r in rg["records"])
    assert rg["records"][-1]["extras"].get("visual_tokens") == 1120, "premise: the budget was set"


def test_frozen_paths_over_closed_range() -> None:
    """A5: the contracts, the fakes, the machine, the speech gate, the dry-run executor, the client, the prompt
    builder, ``reground.py``, ``loop.py``, ``tools_v1.json``, ``policy.py``, ``services/vllm.sh``,
    ``versions.lock``, ``computer/``, ``pages/`` and the HUD are byte-identical over ``BASE_SHA..TIP_IMPL`` (never
    the working tree), with no file added or removed. Power: the same predicate sees ``82da380..02ccff5``'s lock
    change."""
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
        f"T-0132 A5: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN) == ["services/versions.lock"], "power"


def test_tools_sha_unchanged() -> None:
    """A5: ``tools_v1.json`` hashes to ``bbfc53cc…`` at BASE and at TIP_IMPL."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    for rev in (BASE_SHA, TIP_IMPL):
        blob = gr._git("show", f"{rev}:local_astra/toolcall/tools_v1.json")
        assert blob.returncode == 0, blob.stderr
        digest = hashlib.sha256(blob.stdout.encode()).hexdigest()
        assert digest == t0131.TOOLS_SHA and digest.startswith("bbfc53cc"), rev
