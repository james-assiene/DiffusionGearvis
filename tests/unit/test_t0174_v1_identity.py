"""T-0174 group A, A6: with no flag, ``live_guidance = "v1"`` and ``--live-guidance v1`` a live session is BASE's,
byte for byte; tools off and dry-run are unchanged; the frozen paths are byte-identical over ``BASE..TIP``; the
docs/02 §4 invariants are unedited (docs/tasks/T-0174.md).

- **The differential** runs ``_capture`` below in-process and in a subprocess over ``git archive BASE`` (T-0114's
  ``_base_tree``, origin checked). Sections:
  - ``cli`` itself (T-0152's ``_cli_run``: ``astra session --tools live --browser headless``) for each source of
    v1 (``default``: no flag, no key; ``flag-v1``: ``--live-guidance v1``; ``profile-v1``: ``live_guidance = "v1"``),
    under ``--dry-run`` and at a real start: exit code, stdout, stderr, every ``Session`` and executor keyword;
  - a rig session built with exactly the keywords ``cli`` passed (T-0152's ``_rig_turn``: one live loop turn on a
    fitting and on a long page; machine steps, HUD messages, prompts, request bodies, session-log records, turn
    lines, history, browser invocations), and the T-0174 A5 two-turn rig's wire bodies (three tool steps, then a
    second turn);
  - T-0134c's other modes (off and ``dry_run`` through ``cli``, with live profiles overridden);
  - the executor's action strings and ``resolved`` over T-0111c's fake driver (T-0134's ``_executor_actions``).
  Each v1 source is compared with BASE's ``default``. At BASE the flag and the key do not exist, so BASE has only
  the default to compare with.
- **Reading (reported).** A4 says a flag or profile source prints exactly one ``live guidance`` line while A6 says
  the startup output is byte-identical to BASE for all three sources; both hold only if that one line is admitted.
  So for ``flag-v1`` and ``profile-v1`` exactly ``astra session: live guidance: v1 (flag|profile)`` is admitted (it
  must be present once, on stdout, and is the only difference); for ``default`` nothing is admitted.
- **Byte identity** over ``BASE..TIP`` (two literals) of ``turn_machine.py``, ``speech_gate.py``, ``session.py``,
  ``llm_client.py``, ``contracts/``, ``toolcall/`` and ``computer/``, and of docs/02 §4 and
  ``tests/unit/test_turn_invariants.py`` ("§4 invariants unedited"). ``TIP`` is f23bb148, group A's A2b implementation
  commit (bumped from BASE by the test-author). Power: the same predicate sees T-0172 group
  B's change to ``session.py`` in its merge.

**Group D (D1, D4; James's ruling 2026-10-27: the default is v2).** ``test_default_is_base`` is re-scoped to
``test_default_is_explicit_v2`` (the default equals ``--live-guidance v2`` everywhere, its one A4 line aside, and no
longer BASE's ``tool_guidance``); ``test_flag_v1_is_base`` adds that the v1 sources still give BASE's bytes and now
differ from the default. Both were written ahead as pending and activated for D's implementation. The BASE subprocess runs
with an empty ``HOME`` (the T-0163 Brave key file; see ``base``).

Every row but ``test_v1_sources_are_base`` was green at BASE and stays green. ``test_v1_sources_are_base`` was written
ahead as pending (at BASE ``cli`` refused the flag and the key: exit 2) and activated for group A (87172bda).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps), 1280x800 frames, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import _t0160_pins as t0160  # T-0178 A8
from tests.unit import _t0181_pins as t0181  # T-0181 A-R13
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit._t0177_pins import observed as t0177_observed  # T-0177 A-P1 and B-P1

#: Group A's BASE: d486bcb9 (main after Merge T-0172 group B, plus the T-0174 status commit). A literal.
BASE = "d486bcb9"
#: The implementation tip: f23bb148, group A's A2b implementation commit (bumped from 31b91d7d by the test-author).
TIP = "f23bb148"
#: T-0172 group B's merge and its first parent (the power row: ``session.py`` changed there)
POWER = ("1b8feee0", "2114a765")
FIELD = "live_guidance"
LINE_PREFIX = "astra session: live guidance: "

#: A6's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "local_astra/toolcall",
    "local_astra/computer",
    "tests/unit/test_turn_invariants.py",
)

#: case -> (flags, profile body after T-0152's ``LIVE_PROFILE``, the source the admitted line names or None)
CASES: dict[str, tuple[list[str], str, str | None]] = {
    "default": ([], "", None),
    "flag-v1": (["--live-guidance", "v1"], "", "flag"),
    "profile-v1": ([], f'{FIELD} = "v1"\n', "profile"),
    # T-0174 D: the explicit v2 source, the oracle for the default after the flip (D1)
    "flag-v2": (["--live-guidance", "v2"], "", "flag"),
    # T-0191 C3: the explicit v3 source, the oracle for the default after T-0191's flip (C1; web mode)
    "flag-v3": (["--live-guidance", "v3"], "", "flag"),
}
#: the version each case's admitted A4 line names (the default admits none)
VERSION = {"default": None, "flag-v1": "v1", "profile-v1": "v1", "flag-v2": "v2", "flag-v3": "v3"}
PAGES = ("fits", "long")


def _same(
    want: Any, got: Any, what: str, ref: str = f"T-0174 A6: differs from BASE {BASE}"
) -> None:
    assert got == want, (
        f"{ref}: {what}:\nwant {json.dumps(want)[:1500]}\ngot  {json.dumps(got)[:1500]}"
    )


_D_REF = "T-0174 D1: the default differs from --live-guidance v2"


async def _rigs(tmp: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from tests.contract import test_t0174_prefix_oracle as a5
    from tests.unit import test_t0152_overview_frame_default as t0152

    out: dict[str, Any] = {}
    for case, run in runs.items():
        if run["code"] != 0 or not run["_raw_kwargs"]:
            out[case] = None
            continue
        kw, echo = t0152.rig_kwargs(run)
        out[case] = {
            page: await t0152._rig_turn(tmp / case / page, kw, echo, page) for page in PAGES
        }
        two = await a5.two_turns(tmp / case / "two", kw, echo, alternate=False)
        text = json.dumps({"bodies": two["bodies"], "prefills": two["prefills"]})
        norm = json.loads(text.replace(two["session_id"], "<session>"))
        out[case]["two_turn_bodies"] = norm["bodies"]
        out[case]["two_turn_prefills"] = norm["prefills"]
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    from tests.unit import test_t0134_differential as t0134
    from tests.unit import test_t0134c_live_feedback_default as t0134c
    from tests.unit import test_t0152_overview_frame_default as t0152

    out: dict[str, Any] = {"cli": {}}
    runs: dict[str, dict[str, Any]] = {}
    for case, (flags, body, _) in CASES.items():
        # T-0179 A-5: the working tree's live session pins the guard off (BASE's); a no-op in the BASE run
        # T-0201 D3 re-pin: the reminder pinned off too (BASE's requests); a no-op in the BASE run
        dry = t0152._cli_run(
            tmp / "cli" / case / "dry",
            flags,
            body,
            dry=True,
            pin_guard_off=True,
            pin_reminder_off=True,
            pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE's kit and guidance)
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE's turns)
        )
        start = t0152._cli_run(
            tmp / "cli" / case / "start",
            flags,
            body,
            dry=False,
            pin_guard_off=True,
            pin_reminder_off=True,
            pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE's kit and guidance)
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE's turns)
        )
        runs[case] = start
        out["cli"][case] = {"dry": t0152._public(dry), "start": t0152._public(start)}
    out["rigs"] = asyncio.run(_rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    out["executor"] = asyncio.run(t0134._executor_actions())
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0174_v1_identity as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture, in BASE's tree (T-0131's ``git archive`` harness through ``tsl._base_tree``)."""
    gr.require_commits(BASE)
    tmp = tmp_path_factory.mktemp("t0174_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    # T-0163 reads the Brave key file under ``Path.home()`` at each call; in-process ``tests/unit/conftest.py``
    # points it nowhere, but this subprocess does not load that conftest, so on a box with a real key file BASE
    # built ``FallbackSearch`` where the working tree builds ``DdgsSearch``. An empty HOME is the conftest's world.
    home = tmp / "empty_home"
    home.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "HOME": str(home)},
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _capture(tmp_path_factory.mktemp("t0174_now"))


def _admit_line(
    run: dict[str, Any], source: str | None, what: str, version: str = "v1"
) -> dict[str, Any]:
    """``run`` without the one admitted ``live guidance <version> (<source>)`` line (none admitted for the
    default)."""
    out = dict(run)
    lines = str(run["out"]).splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if x.startswith(LINE_PREFIX)]
    assert LINE_PREFIX not in str(run["err"]), (
        f"T-0174 A4 ({what}): no live guidance line on stderr"
    )
    if source is None:
        assert found == [], f"T-0174 A6 ({what}): the default prints no live guidance line"
        return out
    want = f"{LINE_PREFIX}{version} ({source})"
    assert len(found) == 1 and lines[found[0]].rstrip("\n") == want, (
        f"T-0174 A4 ({what}): exactly one {want!r}; found {[lines[i] for i in found]}"
    )
    out["out"] = "".join(lines[: found[0]] + lines[found[0] + 1 :])
    return out


def _compare_case(base: dict[str, Any], now: dict[str, Any], case: str) -> None:
    _, _, source = CASES[case]
    for kind in ("dry", "start"):
        want = base["cli"]["default"][kind]
        got = now["cli"][case][kind]
        assert want["code"] == 0, f"premise: BASE's default {kind} exits 0"
        assert got["code"] == 0, (
            f"T-0174 A6 ({case} {kind}): exit {got['code']}\n{got['out'][-600:]}{got['err'][-600:]}"
        )
        got = _admit_line(got, source, f"{case} {kind}")
        if kind == "dry":
            # T-0178 A8: web mode's loop=on(20) back to BASE's 8, strictly, once group A exists
            got = {**got, "out": t0160.admit_web_loop(str(got["out"]))}
        got = t0181.run_without_mic_unavailable(got)  # T-0181 A-R13 pin move
        _same(want, got, f"cli {case} {kind}")
    want_rig, got_rig = base["rigs"]["default"], now["rigs"][case]
    assert want_rig is not None and got_rig is not None, f"premise: rigs built ({case})"
    # T-0177 A-P1: the observed side only, A10's extras.vision_source == "browser" (never "shared") set aside
    got_rig = t0177_observed(got_rig)
    for key in want_rig:
        if key in PAGES:
            for section in want_rig[key]:
                _same(want_rig[key][section], got_rig[key][section], f"rig {case} {key} {section}")
        else:
            _same(want_rig[key], got_rig[key], f"rig {case} {key}")
    assert sorted(want_rig) == sorted(got_rig), (sorted(want_rig), sorted(got_rig))


# --------------------------------------------------------------------------------------------- rows


def _guidance_kw(run: dict[str, Any]) -> Any:
    kws = run["session_kwargs"]
    assert kws, "premise: a session was built"
    return kws[-1].get("tool_guidance")


_C_REF = "T-0191 C1: the default differs from --live-guidance v3"


def test_default_is_explicit_v3(base: dict[str, Any], now: dict[str, Any]) -> None:
    """T-0191 C3 re-scope of T-0174 D1's ``test_default_is_explicit_v2`` (the default flipped from v2 to v3, web
    mode, the capture's ``LIVE_PROFILE``): with no flag and no key, ``cli``'s exit, stdout, stderr and keywords under
    ``--dry-run`` and at a real start, and the rig sessions built from those keywords (request bodies, session-log
    records, executor invocations, turn lines, HUD, the two-turn A5 bodies) equal those of ``--live-guidance v3`` with
    its one A4 line set aside, and the default prints no ``live guidance`` line. Too-narrow companions: the default
    is no longer BASE's (v1) and no longer ``--live-guidance v2``'s (the pre-flip default)."""
    for kind in ("dry", "start"):
        ref = now["cli"]["flag-v3"][kind]
        got = now["cli"]["default"][kind]
        assert ref["code"] == 0 and got["code"] == 0, (ref["code"], got["code"])
        _same(
            _admit_line(ref, "flag", f"flag-v3 {kind}", version="v3"),
            _admit_line(got, None, f"default {kind}"),
            f"cli {kind}",
            _C_REF,
        )
    start = now["cli"]["default"]["start"]
    assert _guidance_kw(start) != _guidance_kw(base["cli"]["default"]["start"]), (
        "T-0174 D1: the default session's tool_guidance is still BASE's (v1)"
    )
    assert _guidance_kw(start) != _guidance_kw(now["cli"]["flag-v2"]["start"]), (
        "T-0191 C1: the default session's tool_guidance is still v2's (the pre-flip default)"
    )
    _D_REF = _C_REF
    ref_rig, got_rig = now["rigs"]["flag-v3"], now["rigs"]["default"]
    assert ref_rig is not None and got_rig is not None, "premise: rigs built (default, flag-v3)"
    assert sorted(ref_rig) == sorted(got_rig), (sorted(ref_rig), sorted(got_rig))
    for key in ref_rig:
        if key in PAGES:
            for section in ref_rig[key]:
                _same(ref_rig[key][section], got_rig[key][section], f"rig {key} {section}", _D_REF)
        else:
            _same(ref_rig[key], got_rig[key], f"rig {key}", _D_REF)


@pytest.mark.parametrize("case", ["flag-v1", "profile-v1"])
def test_flag_v1_is_base(base: dict[str, Any], now: dict[str, Any], case: str) -> None:
    """T-0174 D1: after the flip, ``--live-guidance v1`` (and ``live_guidance = "v1"``) is how a session gets A6's
    BASE bytes: everything equals BASE's default, the one A4 line ``live guidance: v1 (flag|profile)`` aside (and it
    is printed), and the session's ``tool_guidance`` is not the new default's. The second half is what makes this a
    D row: before the flip the v1 source and the default build the same session."""
    _compare_case(base, now, case)
    assert _guidance_kw(now["cli"][case]["start"]) != _guidance_kw(
        now["cli"]["default"]["start"]
    ), (
        f"T-0174 D1 ({case}): v1 and the default build the same tool_guidance; the default is still v1"
    )


@pytest.mark.parametrize("case", ["flag-v1", "profile-v1"])
def test_v1_sources_are_base(base: dict[str, Any], now: dict[str, Any], case: str) -> None:
    """A6: ``--live-guidance v1`` and ``live_guidance = "v1"`` give BASE's default bytes everywhere, the one A4 line
    aside (module docstring)."""
    _compare_case(base, now, case)


def test_other_modes_unchanged(base: dict[str, Any], now: dict[str, Any]) -> None:
    """A6 (active): tools off and dry-run through ``cli`` (live profiles overridden too): output, client and session
    keywords, executors, the system message and a request body equal BASE's."""
    for case in base["other"]:
        for key in base["other"][case]:
            got = now["other"][case][key]
            if key == "session_kwargs":
                got = t0181.without_mic_unavailable(got)  # T-0181 A-R13 pin move
            _same(base["other"][case][key], got, f"other {case} {key}")
    flat = json.dumps(now["other"])
    assert "live guidance" not in flat and '"live_guidance"' not in flat


def test_executor_unchanged(base: dict[str, Any], now: dict[str, Any]) -> None:
    """A6 (active): the executor's action strings and ``resolved`` over the fake driver equal BASE's."""
    _same(base["executor"], now["executor"], "executor actions")


def test_frozen_paths_over_closed_range() -> None:
    """A6 (active): the frozen paths are byte-identical over ``BASE..TIP`` (two literals, never the working tree), and
    each exists at BASE. Power: the same predicate sees T-0172 group B's ``session.py`` change in its merge."""
    gr.require_commits(BASE, TIP, *POWER)
    assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {BASE}"
        )
    changed = gr.changed(BASE, TIP, *FROZEN)
    assert changed == [], f"T-0174 A6: frozen paths differ in {BASE}..{TIP}: {changed}"
    assert "local_astra/conductor/session.py" in gr.changed(POWER[0], POWER[1], *FROZEN), "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_section4_invariants_unedited() -> None:
    """A6 (active): docs/02 §4 (the turn machine and its invariants) is byte-identical at BASE and TIP."""
    gr.require_commits(BASE, TIP)
    at_base = t7d.section(_doc02_at(BASE), "## 4. ")
    at_tip = t7d.section(_doc02_at(TIP), "## 4. ")
    assert at_base.strip() and "invariant" in at_base.lower(), "premise: §4 found at BASE"
    assert at_tip == at_base, "T-0174 A6: docs/02 §4 changed over BASE..TIP"
