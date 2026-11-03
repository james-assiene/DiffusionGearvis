"""T-0114 criterion 14, the CLI and session half: the loop flags, the policy, the refusal, the validation and labels.

Spec (docs/tasks/T-0114.md criterion 14, H-T114-2, H-T114-4, H-T114-6):

- ``astra session --tool-loop``, ``--no-tool-loop`` and ``--max-tool-steps N``; N < 1 is an argparse error.
- ``build_session`` passes ``TurnPolicy(thinking=…, tool_loop=…, max_tool_steps=…)``; since T-0182 group B (B-R5,
  the CLI-built policy moves) also ``ack_clip=False`` (filler clips off by default; no row here runs live).
- ``--tool-loop`` with tools ``off`` is refused at startup with a message naming ``--tools dry-run``.
- ``Session(...)`` raises ``ValueError`` for a policy with ``tool_loop=True`` and ``max_tool_steps < 1`` or
  ``tool_result_timeout_ms <= TOOL_EXEC_TIMEOUT_MS`` (10 000, H-T114-4).
- Loop on: ``Regime.tools`` gains ``:loop<max>`` and the ``--dry-run`` tools line gains `` loop=on(<max>)``.
  Loop off: both unchanged (those rows hold now and are not pending).

Readings (in the test-author's report for the reviewer):

- R14-a (adversarial, H-T114-2 "refused, not silently inert"): ``--max-tool-steps N`` when the *resolved* loop is off
  (no ``--tool-loop`` and the profile's ``tool_loop`` false, or ``--no-tool-loop``) is **refused** — a non-zero exit, a
  message naming ``--tool-loop``, no session constructed. A profile's own ``max_tool_steps`` with the loop off is not
  refused (the field always has a value, so it cannot express intent).
- R14-b: a profile with ``tool_loop = true`` and tools off is refused like the flag (the same inert loop).
- R14-c: ``:loop<max>`` is appended last, after ``:repair_off`` when that is present.
- R14-d: flags override the profile, as ``--tools`` does (ruling H-A4): ``--no-tool-loop`` beats ``tool_loop = true``,
  ``--max-tool-steps`` beats the profile's ``max_tool_steps``.

The CLI rows drive ``cli.main`` over the fakes (``tests/unit/_tools_switch_support.py``): concrete clients replaced,
``Session`` spied, ``_serve`` returns at once with the graph recorded. Nothing is served and no device opens.

Re-scoped by T-0123 (the loop on by default whenever tools are on; docs/tasks/T-0123.md "Named re-scopes"): the
tools-on rows that assumed the loop off by default moved to ``test_cli_tool_loop_default.py`` or gained an explicit
opt-out; every H-T114-2, R14-b, R14-d, Session-validation and regime row is unedited. The re-scoped rows are
active since T-0123's activation.

Regime: CPU over fakes, ``ScriptedClock``; no measurement.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.conductor.turn_machine import TurnPolicy
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import explode_clients, run_cli

#: H-T114-4, from the task text (not read from the module under test)
TOOL_EXEC_TIMEOUT_MS = 10_000
#: the tools-on dry-run line today (T-0092b criterion 1), written out by hand
LINE_ON = f"tools=dry_run native:{T.TOOLS_SHA12} repair=on"
LINE_OFF = "tools=off"
#: T-0183 A3a: every ``off``/``dry_run`` tools line ends in the resolved budget, 1120 by default (no row here sets a
#: budget); written out from the criterion
VT = " visual_tokens=1120"


# --- helpers -----------------------------------------------------------------------------------------------------


def _config(
    tmp_path: Path,
    *,
    tools: str | None = None,
    tool_loop: bool | None = None,
    max_tool_steps: int | None = None,
    thinking: bool | None = None,
) -> Path:
    lines = ["profile = 'lean'", "[profiles.lean]"]
    if tool_loop is not None or max_tool_steps is not None:
        from local_astra.config import Profile

        missing = {"tool_loop", "max_tool_steps"} - set(Profile.model_fields)
        assert not missing, (
            f"T-0114 criterion 14: Profile lacks {sorted(missing)}, so astra.toml cannot set them"
        )
    if tools is not None:
        lines.append(f'tools = "{tools}"')
    if tool_loop is not None:
        lines.append(f"tool_loop = {'true' if tool_loop else 'false'}")
    if max_tool_steps is not None:
        lines.append(f"max_tool_steps = {max_tool_steps}")
    if thinking is not None:
        lines += ["[profiles.lean.diffusion]", f"thinking = {'true' if thinking else 'false'}"]
    path = tmp_path / "astra.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: Path, *flags: str
) -> tuple[int, T.Calls, str]:
    """``astra session`` over the fakes, stopped before serving."""
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    code = run_cli(T.session_argv(config, tmp_path, *flags))
    return code, calls, ""


def _started_policy(tmp_path, monkeypatch, capsys, config: Path, *flags: str) -> TurnPolicy:
    code, calls, _ = _start(tmp_path, monkeypatch, config, *flags)
    out = capsys.readouterr()
    assert code == 0 and calls.graphs, (
        f"premise: `astra session {' '.join(flags)}` starts (exit {code}):\n{out.out}{out.err}"
    )
    policy = calls.graphs[-1].session.machine.policy
    assert isinstance(policy, TurnPolicy), policy
    return policy


def _refused(tmp_path, monkeypatch, capsys, config: Path, *flags: str) -> str:
    """Assert the start is refused before any session exists; return what it said."""
    code, calls, _ = _start(tmp_path, monkeypatch, config, *flags)
    out = capsys.readouterr()
    message = out.out + out.err
    assert code != 0, (
        f"refused: a non-zero exit; `astra session {' '.join(flags)}` exited {code}:\n{message}"
    )
    assert "unrecognized arguments" not in message, (
        f"T-0114 criterion 14: the flags exist; this is argparse not knowing them:\n{message}"
    )
    assert not calls.session_kwargs and not calls.graphs, (
        "refused: no session was constructed or served"
    )
    return message


def _dry_run_tools_line(tmp_path, monkeypatch, capsys, config: Path, *flags: str) -> str:
    explode_clients(monkeypatch)
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    code = run_cli([*T.session_argv(config, tmp_path), "--dry-run", *flags])
    out = capsys.readouterr()
    assert code == 0, f"--dry-run exits 0; it exited {code}:\n{out.out}{out.err}"
    lines = [line for line in (out.out + out.err).splitlines() if line.startswith("tools=")]
    assert len(lines) == 1, f"one tools line; the dry run printed:\n{out.out}{out.err}"
    return lines[0]


def _session(tmp_path: Path, policy: TurnPolicy, kit: Any) -> Any:
    """``test_session_loop``'s rig (as ``T.session_rig``), with the machine built on ``policy``."""
    from tests.fakes import FakeMemory, FakeStt
    from tests.unit.test_session_loop import (
        RecordingAudio,
        RecordingLlm,
        RecordingMachine,
        RecordingPerception,
        RecordingTts,
        Rig,
        ScriptedClock,
        Session,
        SessionConfig,
        answer,
        frame,
        require_session,
    )

    require_session()
    assert T.KIT_KWARG in inspect.signature(Session.__init__).parameters
    clock = ScriptedClock()
    machine = RecordingMachine(policy)
    llm = RecordingLlm([answer("Done. ")], prefix_tokens=1000)
    llm.clock = clock
    tts = RecordingTts(ms_per_word=200)
    audio = RecordingAudio(clock)
    turns_path = tmp_path / "turns.jsonl"
    session = Session(
        machine,
        llm,
        FakeStt(),
        tts,
        audio,
        RecordingPerception([frame(0, "a")]),
        FakeMemory(),
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns_path),
        tool_kit=kit,
    )
    return Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=session.stt,
        tts=tts,
        audio=audio,
        perception=session.perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
    )


async def _regime_tools(tmp_path: Path, policy: TurnPolicy, kit: Any) -> str:
    rig = _session(tmp_path, policy, kit)
    await rig.session.start()
    try:
        await T.one_typed_turn(rig)
    finally:
        await rig.session.aclose()
    return str(rig.records()[-1].regime.tools)


# --- the flags -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_max_tool_steps_below_1_or_not_an_int_is_an_argparse_error(
    tmp_path, monkeypatch, capsys, bad
):
    """Criterion 14: ``--max-tool-steps N`` with N < 1 is an argparse error (exit 2, naming the flag), not an
    "unrecognized arguments" error (which is what an absent flag also gives, with exit 2)."""
    code, calls, _ = _start(
        tmp_path,
        monkeypatch,
        _config(tmp_path),
        "--tools",
        "dry-run",
        "--tool-loop",
        f"--max-tool-steps={bad}",
    )
    err = capsys.readouterr().err
    assert code == 2, f"T-0114 criterion 14: an argparse error exits 2; got {code}:\n{err}"
    assert "unrecognized arguments" not in err, f"the flag must exist:\n{err}"
    assert "--max-tool-steps" in err, f"the error names the flag:\n{err}"
    assert not calls.session_kwargs, "no session was constructed"


def test_tool_loop_flag_with_dry_run_tools_passes_the_policy(tmp_path, monkeypatch, capsys):
    """Criterion 14: ``build_session`` passes ``TurnPolicy(thinking=…, tool_loop=True, max_tool_steps=8)`` — the
    profile's thinking and default bound; every other field the ``TurnPolicy`` default."""
    policy = _started_policy(
        tmp_path, monkeypatch, capsys, _config(tmp_path), "--tools", "dry-run", "--tool-loop"
    )
    assert policy == TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8, ack_clip=False), (
        policy
    )


@pytest.mark.parametrize("n", [1, 3])
def test_max_tool_steps_flag_reaches_the_policy_and_1_is_legal(tmp_path, monkeypatch, capsys, n):
    """Criterion 14: ``--max-tool-steps N`` is the policy's ``max_tool_steps``; ``1`` is the smallest legal value
    (the too-wide side of the N < 1 guard). Thinking off in the profile travels too."""
    policy = _started_policy(
        tmp_path,
        monkeypatch,
        capsys,
        _config(tmp_path, thinking=False),
        "--tools",
        "dry-run",
        "--tool-loop",
        "--max-tool-steps",
        str(n),
    )
    assert policy == TurnPolicy(thinking=False, tool_loop=True, max_tool_steps=n, ack_clip=False), (
        policy
    )


def test_the_profile_turns_the_loop_on_and_sets_the_bound(tmp_path, monkeypatch, capsys):
    """Criterion 14: with no loop flag the profile's ``tool_loop``/``max_tool_steps`` decide."""
    config = _config(tmp_path, tools="dry_run", tool_loop=True, max_tool_steps=5)
    policy = _started_policy(tmp_path, monkeypatch, capsys, config)
    assert policy == TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=5, ack_clip=False), (
        policy
    )


def test_flags_override_the_profile(tmp_path, monkeypatch, capsys):
    """Criterion 14, reading R14-d: ``--no-tool-loop`` beats ``tool_loop = true``; ``--max-tool-steps`` beats the
    profile's bound."""
    config = _config(tmp_path, tools="dry_run", tool_loop=True, max_tool_steps=5)
    off = _started_policy(tmp_path, monkeypatch, capsys, config, "--no-tool-loop")
    # criterion 14 names all three fields: the profile's bound is passed even with the loop off (it is inert there,
    # and only an explicit ``--max-tool-steps`` with the loop off is refused, reading R14-a)
    assert off == TurnPolicy(thinking=True, tool_loop=False, max_tool_steps=5, ack_clip=False), off
    two = _started_policy(tmp_path, monkeypatch, capsys, config, "--max-tool-steps", "2")
    assert two == TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=2, ack_clip=False), two


def test_no_tool_loop_flag_is_accepted_with_tools_off(tmp_path, monkeypatch, capsys):
    """Criterion 14: ``--no-tool-loop`` exists and, with tools off, starts the session as today."""
    policy = _started_policy(tmp_path, monkeypatch, capsys, _config(tmp_path), "--no-tool-loop")
    assert policy == TurnPolicy(thinking=True, ack_clip=False), policy


@pytest.mark.parametrize("tools", [None])
def test_with_no_loop_flag_the_policy_is_todays(tmp_path, monkeypatch, capsys, tools):
    """Criterion 14, loop off: no loop flag and no profile key gives ``TurnPolicy(thinking=…)`` with the loop off,
    tools off. Holds now. (T-0123: the tools-on ``[dry_run]`` row moved to
    ``test_cli_tool_loop_default.py::test_resolution_truth_table[p-dry/unset/none]``, as loop on, 8.)"""
    policy = _started_policy(
        tmp_path, monkeypatch, capsys, _config(tmp_path, tools=tools, thinking=False)
    )
    assert policy == TurnPolicy(thinking=False, ack_clip=False), policy
    assert policy.tool_loop is False


# --- the refusal (H-T114-2) ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [("--tool-loop",), ("--tool-loop", "--tools", "off")],
    ids=["tools-default-off", "tools-flag-off"],
)
def test_tool_loop_with_tools_off_is_refused_naming_tools_dry_run(
    tmp_path, monkeypatch, capsys, flags
):
    """Criterion 14 / H-T114-2: ``--tool-loop`` with tools off is refused at startup, naming ``--tools dry-run``,
    and no session is constructed."""
    message = _refused(tmp_path, monkeypatch, capsys, _config(tmp_path), *flags)
    assert "--tools dry-run" in message, (
        f"T-0114 H-T114-2: the refusal names `--tools dry-run`:\n{message}"
    )


def test_tool_loop_flag_beside_a_dry_run_profile_overridden_off_is_refused(
    tmp_path, monkeypatch, capsys
):
    """Criterion 14 / H-T114-2: the *resolved* tools setting decides — a ``dry_run`` profile with ``--tools off``
    and ``--tool-loop`` is a loop with nothing to execute."""
    message = _refused(
        tmp_path,
        monkeypatch,
        capsys,
        _config(tmp_path, tools="dry_run"),
        "--tools",
        "off",
        "--tool-loop",
    )
    assert "--tools dry-run" in message, message


def test_profile_tool_loop_with_tools_off_is_refused(tmp_path, monkeypatch, capsys):
    """Reading R14-b (H-T114-2): ``tool_loop = true`` in the profile with tools off is the same inert loop."""
    message = _refused(tmp_path, monkeypatch, capsys, _config(tmp_path, tool_loop=True))
    assert "--tools dry-run" in message, message


@pytest.mark.parametrize(
    ("profile_loop", "flags", "names"),
    [
        # T-0123: the former [no-loop-flag] row (tools on, no loop setting) now starts with the loop on; it is
        # test_cli_tool_loop_default.py::test_max_tool_steps_with_tools_on_and_no_loop_setting_starts
        pytest.param(
            True,
            ("--tools", "dry-run", "--no-tool-loop", "--max-tool-steps", "3"),
            "--tool-loop",
            id="no-tool-loop-beats-profile",
        ),
        pytest.param(
            False,
            ("--tools", "dry-run", "--max-tool-steps", "3"),
            "--tool-loop",
            id="profile-opt-out",
        ),
        pytest.param(
            None,
            ("--max-tool-steps", "3"),
            "--tools dry-run",
            id="tools-off-too",
        ),
    ],
)
def test_adversarial_max_tool_steps_without_the_loop_is_refused(
    tmp_path, monkeypatch, capsys, profile_loop, flags, names
):
    """Reading R14-a (adversarial; H-T114-2 "refused rather than silently inert"): an explicit ``--max-tool-steps``
    while the resolved loop is off would be carried and ignored, with an unchanged label — exactly the silent
    inertness H-T114-2 rules out. It is refused and no session starts. With tools on the loop is off only by
    explicit choice (``--no-tool-loop``, or ``tool_loop = false``), and the refusal names ``--tool-loop``
    (T-0123 4(c)); with tools off it names ``--tools dry-run`` (T-0123 4(d), H-T123-3)."""
    message = _refused(
        tmp_path, monkeypatch, capsys, _config(tmp_path, tool_loop=profile_loop), *flags
    )
    assert names in message, (
        f"T-0114 R14-a / T-0123 4(c)-(d): the refusal names `{names}`:\n{message}"
    )


def test_adversarial_companion_max_tool_steps_with_a_profile_loop_starts(
    tmp_path, monkeypatch, capsys
):
    """Reading R14-a, the too-wide side: ``--max-tool-steps 3`` with the loop on from the *profile* (no flag) is
    not refused — a refusal keyed on "``--tool-loop`` absent from argv" instead of the resolved loop fails here."""
    config = _config(tmp_path, tools="dry_run", tool_loop=True)
    policy = _started_policy(tmp_path, monkeypatch, capsys, config, "--max-tool-steps", "3")
    assert policy == TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=3, ack_clip=False), (
        policy
    )


# --- Session validation ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        TurnPolicy(tool_loop=True, max_tool_steps=0),
        TurnPolicy(tool_loop=True, max_tool_steps=-1),
        TurnPolicy(tool_loop=True, tool_result_timeout_ms=TOOL_EXEC_TIMEOUT_MS),
        TurnPolicy(tool_loop=True, tool_result_timeout_ms=TOOL_EXEC_TIMEOUT_MS - 1),
        TurnPolicy(tool_loop=True, tool_result_timeout_ms=0),
    ],
    ids=["steps-0", "steps-neg", "timeout-equal", "timeout-below", "timeout-0"],
)
def test_session_refuses_an_unsound_loop_policy(tmp_path, policy):
    """Criterion 14: ``Session(...)`` raises ``ValueError`` for ``tool_loop=True`` with ``max_tool_steps < 1`` or
    ``tool_result_timeout_ms <= TOOL_EXEC_TIMEOUT_MS`` (10 000, H-T114-4: the execute deadline must fall before R18)."""
    try:
        _session(tmp_path, policy, T.make_kit())
    except ValueError:
        return
    raise AssertionError(f"T-0114 criterion 14: Session(...) must raise ValueError for {policy}")


@pytest.mark.parametrize(
    "policy",
    [
        TurnPolicy(
            tool_loop=True, max_tool_steps=1, tool_result_timeout_ms=TOOL_EXEC_TIMEOUT_MS + 1
        ),
        TurnPolicy(tool_loop=True),
        TurnPolicy(tool_loop=False, max_tool_steps=0, tool_result_timeout_ms=0),
    ],
    ids=["loop-on-at-the-bounds", "loop-on-defaults", "loop-off-unchecked"],
)
def test_session_accepts_a_sound_or_loop_off_policy(tmp_path, policy):
    """Criterion 14, the too-wide side: the smallest legal loop policy (1 step, 10 001 ms) and the defaults construct;
    the check applies only with ``tool_loop=True``, so a loop-off policy is not validated (a loop-off session is
    unchanged). Holds now."""
    rig = _session(tmp_path, policy, T.make_kit())
    assert rig.session.machine.policy == policy


def test_the_h_t114_4_constant_is_below_the_r18_default():
    """Premise of the timeout row: the default ``tool_result_timeout_ms`` (15 000) is above 10 000, so the default
    loop policy is legal."""
    assert TurnPolicy().tool_result_timeout_ms > TOOL_EXEC_TIMEOUT_MS


def test_session_module_exports_the_h_t114_4_constant():
    """H-T114-4 / implementation notes: ``TOOL_EXEC_TIMEOUT_MS`` is a module constant of ``session.py``, 10 000."""
    assert getattr(session_mod, "TOOL_EXEC_TIMEOUT_MS", None) == TOOL_EXEC_TIMEOUT_MS


# --- the labels -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [3, 8])
def test_regime_tools_gains_loop_max_with_the_loop_on(tmp_path, steps):
    """Criterion 14 / H-T114-6: loop on, repair on — ``native:<sha12>:dry_run:loop<max>``."""
    got = asyncio.run(
        _regime_tools(tmp_path, TurnPolicy(tool_loop=True, max_tool_steps=steps), T.make_kit())
    )
    assert got == f"{T.REGIME_ON}:loop{steps}", got


def test_regime_tools_appends_loop_after_repair_off(tmp_path):
    """Criterion 14 / H-T114-6, reading R14-c: ``:loop<max>`` is the last field, after ``:repair_off``."""
    kit = T.make_kit(repair_off="payload stale (test)")
    got = asyncio.run(_regime_tools(tmp_path, TurnPolicy(tool_loop=True, max_tool_steps=3), kit))
    assert got == f"{T.REGIME_REPAIR_OFF}:loop3", got


@pytest.mark.parametrize(
    ("policy", "repair_off", "want"),
    [
        (TurnPolicy(), None, T.REGIME_ON),
        (TurnPolicy(tool_loop=False, max_tool_steps=3), None, T.REGIME_ON),
        (
            TurnPolicy(tool_loop=False, max_tool_steps=3),
            "payload stale (test)",
            T.REGIME_REPAIR_OFF,
        ),
    ],
    ids=["defaults", "bound-set-loop-off", "repair-off-loop-off"],
)
def test_regime_tools_is_unchanged_with_the_loop_off(tmp_path, policy, repair_off, want):
    """Criterion 14, loop off: ``Regime.tools`` is byte-for-byte today's, whatever ``max_tool_steps`` says. Holds now."""
    got = asyncio.run(_regime_tools(tmp_path, policy, T.make_kit(repair_off=repair_off)))
    assert got == want, got


@pytest.mark.parametrize(
    ("cfg", "flags", "steps"),
    [
        ({}, ("--tools", "dry-run", "--tool-loop", "--max-tool-steps", "3"), 3),
        ({}, ("--tools", "dry-run", "--tool-loop"), 8),
        ({"tools": "dry_run", "tool_loop": True, "max_tool_steps": 5}, (), 5),
        # T-0123 criterion 6: the former loop-off [tools-flag] / [tools-profile] rows, now on by default
        pytest.param({}, ("--tools", "dry-run"), 8),
        pytest.param({"tools": "dry_run"}, (), 8),
    ],
    ids=["flags", "flag-default-bound", "profile", "tools-flag", "tools-profile"],
)
def test_dry_run_tools_line_gains_loop_on_max(tmp_path, monkeypatch, capsys, cfg, flags, steps):
    """Criterion 14: with the loop on the ``--dry-run`` tools line gains `` loop=on(<max>)``."""
    line = _dry_run_tools_line(tmp_path, monkeypatch, capsys, _config(tmp_path, **cfg), *flags)
    assert line == f"{LINE_ON} loop=on({steps}){VT}", line


@pytest.mark.parametrize(
    ("cfg", "flags", "want"),
    [
        ({}, (), LINE_OFF + VT),
        ({}, ("--tools", "dry-run", "--no-tool-loop"), LINE_ON + VT),
        ({"tools": "dry_run", "tool_loop": False}, (), LINE_ON + VT),
    ],
    ids=["tools-off", "tools-flag", "tools-profile"],
)
def test_dry_run_tools_line_is_unchanged_with_the_loop_off(
    tmp_path, monkeypatch, capsys, cfg, flags, want
):
    """Criterion 14, loop off: the tools line is byte-for-byte today's. Holds now. T-0123 (H-T123-7): with tools on
    the loop is off only by ``--no-tool-loop`` or ``tool_loop = false``, and the line stays byte-identical (no
    ``loop=off``); the rows without either moved to ``test_dry_run_tools_line_gains_loop_on_max``."""
    line = _dry_run_tools_line(tmp_path, monkeypatch, capsys, _config(tmp_path, **cfg), *flags)
    assert line == want, line
    assert not re.search(r"\bloop=", line), line
