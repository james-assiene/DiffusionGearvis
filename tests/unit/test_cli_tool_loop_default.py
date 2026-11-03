"""T-0123: the tool-result loop is on by default whenever tools are on (human ruling, 2026-10-18).

Spec (docs/tasks/T-0123.md criteria 2-4, 6, 7). With ``tools`` the resolved tools mode (``--tools`` over
``Profile.tools``), ``req`` the loop flag when given else ``Profile.tool_loop`` (``None`` when absent), ``steps``
``--max-tool-steps`` when given else ``Profile.max_tool_steps``:

- tools ``dry_run``: loop = ``req`` if not ``None``, else **True**; tools ``off``: loop = **False**;
- refused, exit 2, before any session exists: (a)/(b) an explicit loop request (``--tool-loop``, or
  ``tool_loop = true`` without ``--no-tool-loop``) with tools off, naming ``--tools dry-run``; (c) an explicit
  ``--max-tool-steps`` with tools on and the loop off *by explicit choice*, naming ``--tool-loop``; (d) an explicit
  ``--max-tool-steps`` with tools off, naming ``--tools dry-run``; (e) a profile's own ``max_tool_steps`` never;
- ``build_session`` passes ``TurnPolicy(thinking=..., tool_loop=loop, max_tool_steps=steps)``; since T-0182 group B
  (B-R5, the CLI-built policy moves) also ``ack_clip=False`` (filler clips off by default; no row here runs live, so
  ``still_working_after_ms`` stays None).

The truth table below is the oracle, written out by hand from those rules (4 tools x 7 loop x 3 steps = 84 rows).
Every row also runs ``--dry-run``: one resolver feeds the policy, the refusal and the tools line, so a row whose
dry run disagrees with its start is "two resolvers that disagree on one row" (a named mutation).

Readings (in the test-author's report):

- When 4(a)/(b) and 4(d) both apply (an explicit loop request *and* ``--max-tool-steps`` with tools off), every
  one of them names ``--tools dry-run``, so the row asserts that and nothing about which text wins; the
  differential (test_t0123_differential.py) pins that it is the unchanged 4(a)/(b) text.
- "Naming ``--tool-loop``" (4(c)) is matched as the flag itself, not as the tail of ``--no-tool-loop``.
- Criterion 7's help wording is matched on the phrases the criterion states ("on whenever tools are on", "off
  with tools off", "with the loop on"), whitespace-normalised, not on an exact sentence.

Rows that need T-0123 (red at 68d5566) are marked ``pending=P`` in the table; they are active since T-0123's
activation.

Regime: CPU over the CLI rig's fakes (``tests/unit/_tools_switch_support.py``), ``ScriptedClock`` for the label
turn; profile ``lean`` from a temporary ``astra.toml``, thinking on; no GPU, no network; no measurement.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnPolicy
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tool_loop import _config, _regime_tools
from tests.unit.test_t0123_differential import base_fixture

#: the tools-on dry-run line today (T-0092b criterion 1), written out by hand
LINE_ON = f"tools=dry_run native:{T.TOOLS_SHA12} repair=on"
LINE_OFF = "tools=off"
#: T-0183 A3a: every ``off``/``dry_run`` tools line ends in the resolved budget, 1120 by default (none of these rows
#: sets a budget); written out from the criterion
VT = " visual_tokens=1120"
#: the two flags a refusal may name
TOOLS_DRY = "--tools dry-run"
TOOL_LOOP = "--tool-loop"
#: ``--tool-loop`` as a flag, not the tail of ``--no-tool-loop``
TOOL_LOOP_FLAG = re.compile(r"(?<![\w-])--tool-loop(?![\w-])")

# --- the axes (docs/tasks/T-0123.md "Tests to write"): profile keys, argv ----------------------------------------

TOOLS_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "p-off": ({}, ()),  # profile off (no key), no flag
    "p-dry": ({"tools": "dry_run"}, ()),  # profile dry_run, no flag
    "f-dry": ({}, ("--tools", "dry-run")),  # --tools dry-run over profile off
    "f-off": ({"tools": "dry_run"}, ("--tools", "off")),  # --tools off over profile dry_run
}
TOOLS_ON = frozenset({"p-dry", "f-dry"})
LOOP_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "unset": ({}, ()),
    "p-true": ({"tool_loop": True}, ()),
    "p-false": ({"tool_loop": False}, ()),
    "f-on": ({}, ("--tool-loop",)),
    "f-off": ({}, ("--no-tool-loop",)),
    "p-false+f-on": ({"tool_loop": False}, ("--tool-loop",)),
    "p-true+f-off": ({"tool_loop": True}, ("--no-tool-loop",)),
}
STEPS_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "none": ({}, ()),
    "f3": ({}, ("--max-tool-steps", "3")),
    "p5": ({"max_tool_steps": 5}, ()),
}


def start(loop: bool, steps: int) -> tuple[str, bool, int]:
    return ("start", loop, steps)


def refused(flag: str) -> tuple[str, str]:
    return ("refused", flag)


def row(tools: str, loop: str, steps: str, want: tuple[Any, ...], *, pending: bool = False) -> Any:
    return pytest.param(
        tools,
        loop,
        steps,
        want,
        id=f"{tools}/{loop}/{steps}",
    )


P = True  # this row needs T-0123 (it is red at 68d5566)

# fmt: off
TRUTH_TABLE = [
    # --- profile off, no flag: tools off --------------------------------------------------------------------------
    row("p-off", "unset",        "none", start(False, 8)),
    row("p-off", "unset",        "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("p-off", "unset",        "p5",   start(False, 5)),                         # 4(e)
    row("p-off", "p-true",       "none", refused(TOOLS_DRY)),                      # 4(b)
    row("p-off", "p-true",       "f3",   refused(TOOLS_DRY)),                      # 4(b)
    row("p-off", "p-true",       "p5",   refused(TOOLS_DRY)),                      # 4(b)
    row("p-off", "p-false",      "none", start(False, 8)),
    row("p-off", "p-false",      "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("p-off", "p-false",      "p5",   start(False, 5)),
    row("p-off", "f-on",         "none", refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "f-on",         "f3",   refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "f-on",         "p5",   refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "f-off",        "none", start(False, 8)),
    row("p-off", "f-off",        "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("p-off", "f-off",        "p5",   start(False, 5)),
    row("p-off", "p-false+f-on", "none", refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "p-false+f-on", "f3",   refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "p-false+f-on", "p5",   refused(TOOLS_DRY)),                      # 4(a)
    row("p-off", "p-true+f-off", "none", start(False, 8)),                         # 4(b) needs no --no-tool-loop
    row("p-off", "p-true+f-off", "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("p-off", "p-true+f-off", "p5",   start(False, 5)),
    # --- profile dry_run, no flag: tools on ------------------------------------------------------------------------
    row("p-dry", "unset",        "none", start(True, 8), pending=P),               # criterion 3, the ruling
    row("p-dry", "unset",        "f3",   start(True, 3), pending=P),               # criterion 3
    row("p-dry", "unset",        "p5",   start(True, 5), pending=P),
    row("p-dry", "p-true",       "none", start(True, 8)),
    row("p-dry", "p-true",       "f3",   start(True, 3)),
    row("p-dry", "p-true",       "p5",   start(True, 5)),
    row("p-dry", "p-false",      "none", start(False, 8)),                         # criterion 3, opt-out
    row("p-dry", "p-false",      "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("p-dry", "p-false",      "p5",   start(False, 5)),                         # 4(e)
    row("p-dry", "f-on",         "none", start(True, 8)),
    row("p-dry", "f-on",         "f3",   start(True, 3)),
    row("p-dry", "f-on",         "p5",   start(True, 5)),
    row("p-dry", "f-off",        "none", start(False, 8)),
    row("p-dry", "f-off",        "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("p-dry", "f-off",        "p5",   start(False, 5)),
    row("p-dry", "p-false+f-on", "none", start(True, 8)),                          # R14-d
    row("p-dry", "p-false+f-on", "f3",   start(True, 3)),
    row("p-dry", "p-false+f-on", "p5",   start(True, 5)),
    row("p-dry", "p-true+f-off", "none", start(False, 8)),                         # R14-d
    row("p-dry", "p-true+f-off", "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("p-dry", "p-true+f-off", "p5",   start(False, 5)),
    # --- --tools dry-run over profile off: tools on ----------------------------------------------------------------
    row("f-dry", "unset",        "none", start(True, 8), pending=P),               # criterion 3, the ruling
    row("f-dry", "unset",        "f3",   start(True, 3), pending=P),               # criterion 3
    row("f-dry", "unset",        "p5",   start(True, 5), pending=P),
    row("f-dry", "p-true",       "none", start(True, 8)),
    row("f-dry", "p-true",       "f3",   start(True, 3)),
    row("f-dry", "p-true",       "p5",   start(True, 5)),
    row("f-dry", "p-false",      "none", start(False, 8)),
    row("f-dry", "p-false",      "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("f-dry", "p-false",      "p5",   start(False, 5)),
    row("f-dry", "f-on",         "none", start(True, 8)),
    row("f-dry", "f-on",         "f3",   start(True, 3)),
    row("f-dry", "f-on",         "p5",   start(True, 5)),
    row("f-dry", "f-off",        "none", start(False, 8)),
    row("f-dry", "f-off",        "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("f-dry", "f-off",        "p5",   start(False, 5)),
    row("f-dry", "p-false+f-on", "none", start(True, 8)),
    row("f-dry", "p-false+f-on", "f3",   start(True, 3)),
    row("f-dry", "p-false+f-on", "p5",   start(True, 5)),
    row("f-dry", "p-true+f-off", "none", start(False, 8)),
    row("f-dry", "p-true+f-off", "f3",   refused(TOOL_LOOP)),                      # 4(c)
    row("f-dry", "p-true+f-off", "p5",   start(False, 5)),
    # --- --tools off over profile dry_run: tools off ---------------------------------------------------------------
    row("f-off", "unset",        "none", start(False, 8)),                         # resolved tools decide
    row("f-off", "unset",        "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("f-off", "unset",        "p5",   start(False, 5)),
    row("f-off", "p-true",       "none", refused(TOOLS_DRY)),                      # 4(b)
    row("f-off", "p-true",       "f3",   refused(TOOLS_DRY)),
    row("f-off", "p-true",       "p5",   refused(TOOLS_DRY)),
    row("f-off", "p-false",      "none", start(False, 8)),
    row("f-off", "p-false",      "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("f-off", "p-false",      "p5",   start(False, 5)),
    row("f-off", "f-on",         "none", refused(TOOLS_DRY)),                      # 4(a)
    row("f-off", "f-on",         "f3",   refused(TOOLS_DRY)),
    row("f-off", "f-on",         "p5",   refused(TOOLS_DRY)),
    row("f-off", "f-off",        "none", start(False, 8)),
    row("f-off", "f-off",        "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("f-off", "f-off",        "p5",   start(False, 5)),
    row("f-off", "p-false+f-on", "none", refused(TOOLS_DRY)),                      # 4(a)
    row("f-off", "p-false+f-on", "f3",   refused(TOOLS_DRY)),
    row("f-off", "p-false+f-on", "p5",   refused(TOOLS_DRY)),
    row("f-off", "p-true+f-off", "none", start(False, 8)),
    row("f-off", "p-true+f-off", "f3",   refused(TOOLS_DRY), pending=P),           # 4(d)
    row("f-off", "p-true+f-off", "p5",   start(False, 5)),
]
# fmt: on


# --- helpers -------------------------------------------------------------------------------------------------------


def _compose(tools: str, loop: str, steps: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    cfg: dict[str, Any] = {}
    argv: tuple[str, ...] = ()
    for axis, key in ((TOOLS_AXIS, tools), (LOOP_AXIS, loop), (STEPS_AXIS, steps)):
        keys, flags = axis[key]
        cfg.update(keys)
        argv += flags
    return cfg, argv


def _launch(tmp_path: Path, capsys: Any, config: Path, *flags: str) -> tuple[int, T.Calls, str]:
    """``astra session`` over the fakes, stopped before serving: (exit code, calls, everything it printed)."""
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        T.fake_clients(mp, calls)
        T.stop_before_serving(mp, calls)
        T.patch_detok_load(mp, calls, tmp_path)
        code = run_cli(T.session_argv(config, tmp_path, *flags))
    out = capsys.readouterr()
    return code, calls, out.out + out.err


def _dry(tmp_path: Path, capsys: Any, config: Path, *flags: str) -> tuple[int, str]:
    """``astra session --dry-run``, every concrete client exploding: (exit code, everything it printed)."""
    with pytest.MonkeyPatch.context() as mp:
        explode_clients(mp)
        T.patch_detok_load(mp, T.Calls(), tmp_path)
        code = run_cli([*T.session_argv(config, tmp_path), "--dry-run", *flags])
    out = capsys.readouterr()
    return code, out.out + out.err


def _tools_line(printed: str) -> str:
    lines = [line for line in printed.splitlines() if line.startswith("tools=")]
    assert len(lines) == 1, f"one tools line; the dry run printed:\n{printed}"
    return lines[0]


def _names(message: str, flag: str) -> bool:
    if flag == TOOL_LOOP:
        return bool(TOOL_LOOP_FLAG.search(message))
    return flag in message


def _assert_started(
    tmp_path: Path,
    capsys: Any,
    config: Path,
    flags: tuple[str, ...],
    want: TurnPolicy,
    tools_on: bool,
) -> None:
    code, calls, printed = _launch(tmp_path, capsys, config, *flags)
    assert code == 0 and calls.graphs, (
        f"T-0123: `astra session {' '.join(flags)}` starts (exit {code}):\n{printed}"
    )
    policy = calls.graphs[-1].session.machine.policy
    assert policy == want, (
        f"T-0123 criterion 2: the policy handed to TurnMachine is {want}; got {policy}"
    )
    kit = calls.kit()
    assert (kit is not None) == tools_on, f"tools {'on' if tools_on else 'off'}: the kit is {kit!r}"


def _assert_refused(
    tmp_path: Path, capsys: Any, config: Path, flags: tuple[str, ...], flag: str
) -> None:
    code, calls, printed = _launch(tmp_path, capsys, config, *flags)
    assert "unrecognized arguments" not in printed, printed
    assert code == 2, (
        f"T-0123 criterion 4: `astra session {' '.join(flags)}` is refused with exit 2; got {code}:\n{printed}"
    )
    assert not calls.session_kwargs and not calls.graphs, (
        "T-0123 criterion 4: refused before any session is constructed or served"
    )
    assert _names(printed, flag), f"T-0123 criterion 4: the refusal names `{flag}`:\n{printed}"


def _expected_line(tools_on: bool, loop: bool, steps: int) -> str:
    if not tools_on:
        return LINE_OFF + VT
    return f"{LINE_ON} loop=on({steps}){VT}" if loop else LINE_ON + VT


# --- criteria 2-4: the truth table ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("tools", "loop", "steps", "want"), TRUTH_TABLE)
def test_resolution_truth_table(tmp_path, capsys, tools, loop, steps, want):
    """Criteria 2-4: the exact ``TurnPolicy`` (and the kit iff tools are on), or a refusal with exit 2 naming its
    flag; and the ``--dry-run`` of the same argv agrees (its tools line, or the same refusal)."""
    cfg, flags = _compose(tools, loop, steps)
    config = _config(tmp_path, **cfg)
    tools_on = tools in TOOLS_ON
    d_code, d_printed = _dry(tmp_path, capsys, config, *flags)
    if want[0] == "start":
        _, on, n = want
        _assert_started(
            tmp_path,
            capsys,
            config,
            flags,
            TurnPolicy(thinking=True, tool_loop=on, max_tool_steps=n, ack_clip=False),
            tools_on,
        )
        assert d_code == 0, f"the dry run of a starting argv exits 0; got {d_code}:\n{d_printed}"
        line = _tools_line(d_printed)
        assert line == _expected_line(tools_on, on, n), (
            f"T-0123 criteria 2/6: the dry-run tools line follows the resolved loop; got {line!r}"
        )
    else:
        _, flag = want
        _assert_refused(tmp_path, capsys, config, flags, flag)
        assert d_code == 2 and _names(d_printed, flag), (
            f"the dry run is refused the same way (exit 2, naming `{flag}`); got {d_code}:\n{d_printed}"
        )


def test_the_truth_table_is_the_whole_product():
    """The table has exactly one row per axis combination (4 x 7 x 3 = 84), so no cell is silently missing."""
    ids = [(p.values[0], p.values[1], p.values[2]) for p in TRUTH_TABLE]
    assert len(ids) == len(set(ids)) == 84
    assert {i[0] for i in ids} == set(TOOLS_AXIS)
    assert {i[1] for i in ids} == set(LOOP_AXIS)
    assert {i[2] for i in ids} == set(STEPS_AXIS)


# --- adversarial rows ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [None, False], ids=["thinking-default", "thinking-off"])
def test_adversarial_tools_off_default_policy_is_exactly_todays(tmp_path, capsys, thinking):
    """Criterion 5 (mutation: ``tool_loop=True`` passed with tools off and the setting unset): the default
    session (no tools key, no loop key, no flag) gets ``TurnPolicy(thinking=...)`` exactly, ``tool_loop is False``,
    ``max_tool_steps`` 8, no kit; its dry-run line is ``tools=off`` with no ``loop=``."""
    config = _config(tmp_path, thinking=thinking)
    want_thinking = thinking is not False
    code, calls, printed = _launch(tmp_path, capsys, config)
    assert code == 0 and calls.graphs, printed
    policy = calls.graphs[-1].session.machine.policy
    assert policy == TurnPolicy(thinking=want_thinking, ack_clip=False), policy
    assert policy.tool_loop is False and policy.max_tool_steps == 8, policy
    assert calls.kit() is None
    d_code, d_printed = _dry(tmp_path, capsys, config)
    assert d_code == 0 and _tools_line(d_printed) == LINE_OFF + VT, d_printed


@pytest.mark.parametrize(
    ("cfg", "flags"),
    [
        ({"tool_loop": True}, ()),
        ({"tools": "dry_run", "tool_loop": True}, ("--tools", "off")),
        ({"tool_loop": True, "max_tool_steps": 5}, ()),
    ],
    ids=["tools-default-off", "tools-flag-off-over-dry-run-profile", "with-a-profile-bound"],
)
def test_adversarial_explicit_profile_true_with_tools_off_still_refused(
    tmp_path, capsys, cfg, flags
):
    """Criterion 4(b) / H-T123-2 (mutation: an explicit profile ``true`` with tools off treated as inert, now that
    ``None`` is the "on when tools are on" value): ``tool_loop = true`` with resolved tools off is refused, exit 2,
    naming ``--tools dry-run``, start and dry run alike."""
    config = _config(tmp_path, **cfg)
    _assert_refused(tmp_path, capsys, config, flags, TOOLS_DRY)
    d_code, d_printed = _dry(tmp_path, capsys, config, *flags)
    assert d_code == 2 and TOOLS_DRY in d_printed, d_printed


@pytest.mark.parametrize(
    ("cfg", "flags", "want", "tools_on"),
    [
        pytest.param(
            {"tools": "dry_run"},
            ("--tools", "off"),
            dict(thinking=True, tool_loop=False, max_tool_steps=8, ack_clip=False),
            False,
            id="flag-off-over-dry-run-profile",
        ),
        pytest.param(
            {},
            ("--tools", "dry-run"),
            dict(thinking=True, tool_loop=True, max_tool_steps=8, ack_clip=False),
            True,
            id="flag-dry-run-over-off-profile",
        ),
    ],
)
def test_adversarial_resolved_tools_decide_not_the_profiles(
    tmp_path, capsys, cfg, flags, want, tools_on
):
    """Criterion 2 (mutation: the default keyed on ``profile.tools`` instead of the resolved tools): a
    ``dry_run`` profile overridden by ``--tools off`` has the loop off and no kit; an ``off`` profile with
    ``--tools dry-run`` has it on, 8 steps. Each direction fails one of the two keyings."""
    # built here, not at collection: before T-0182 B the ``ack_clip`` keyword is a TypeError, not a module error
    want = TurnPolicy(**want)
    config = _config(tmp_path, **cfg)
    _assert_started(tmp_path, capsys, config, flags, want, tools_on)
    d_code, d_printed = _dry(tmp_path, capsys, config, *flags)
    assert d_code == 0, d_printed
    assert _tools_line(d_printed) == _expected_line(tools_on, want.tool_loop, want.max_tool_steps)


@pytest.mark.parametrize(
    ("cfg", "flags"),
    [
        ({}, ("--tools", "dry-run", "--max-tool-steps", "3")),
        ({"tools": "dry_run"}, ("--max-tool-steps", "3")),
    ],
    ids=["tools-flag", "tools-profile"],
)
def test_max_tool_steps_with_tools_on_and_no_loop_setting_starts(tmp_path, capsys, cfg, flags):
    """Criterion 3 (re-scope of T-0114 R14-a's ``[no-loop-flag]`` row; mutation: R14-a keyed on "``--tool-loop``
    absent from argv"): with tools on and no loop setting the loop is on, so ``--max-tool-steps 3`` is no longer
    refused: ``TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=3)`` and ``loop=on(3)``."""
    config = _config(tmp_path, **cfg)
    want = TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=3, ack_clip=False)
    _assert_started(tmp_path, capsys, config, flags, want, True)
    d_code, d_printed = _dry(tmp_path, capsys, config, *flags)
    assert d_code == 0 and _tools_line(d_printed) == f"{LINE_ON} loop=on(3){VT}", d_printed


@pytest.mark.parametrize(
    ("cfg", "flags"),
    [
        ({}, ("--max-tool-steps", "3")),
        ({"tools": "dry_run"}, ("--tools", "off", "--max-tool-steps", "3")),
        ({}, ("--no-tool-loop", "--max-tool-steps", "3")),
        ({"tool_loop": False}, ("--max-tool-steps", "3")),
    ],
    ids=["tools-default-off", "tools-flag-off", "no-tool-loop", "profile-false"],
)
def test_max_tool_steps_with_tools_off_refused_names_tools_dry_run(tmp_path, capsys, cfg, flags):
    """Criterion 4(d) / H-T123-3 (mutation: R14-a dropped for tools off): an explicit ``--max-tool-steps`` with
    resolved tools off is refused, exit 2, before any session, naming ``--tools dry-run`` (the 68d5566 text
    advised ``--tool-loop``, itself refused with tools off). The too-wide side is the truth table's ``*/p5``
    rows: a profile's own bound with tools off starts (4(e))."""
    config = _config(tmp_path, **cfg)
    _assert_refused(tmp_path, capsys, config, flags, TOOLS_DRY)
    d_code, d_printed = _dry(tmp_path, capsys, config, *flags)
    assert d_code == 2 and TOOLS_DRY in d_printed, d_printed


# --- criterion 6: labels -------------------------------------------------------------------------------------------


def _point_repair(mp: pytest.MonkeyPatch, tmp_path: Path, kind: str) -> None:
    if kind == "repair_off":
        T.point_cli_at(mp, payloads=tmp_path / "nowhere" / "native_T4.json")


def _labels(tmp_path: Path, capsys: Any, kind: str, *flags: str) -> tuple[TurnPolicy, str, str]:
    """For ``--tools dry-run`` plus ``flags``: the policy, the ``Regime.tools`` a typed turn on that policy and the
    CLI's kit writes, and the dry-run tools line with the temporary directory written ``<tmp>``."""
    config = _config(tmp_path)
    argv = ("--tools", "dry-run", *flags)
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        _point_repair(mp, tmp_path, kind)
        T.fake_clients(mp, calls)
        T.stop_before_serving(mp, calls)
        T.patch_detok_load(mp, calls, tmp_path)
        code = run_cli(T.session_argv(config, tmp_path, *argv))
        out = capsys.readouterr()
        assert code == 0 and calls.graphs, f"premise: the tools-on start runs:\n{out.out}{out.err}"
        policy = calls.graphs[-1].session.machine.policy
        rig_dir = tmp_path / f"rig{len(flags)}"
        rig_dir.mkdir()
        regime = asyncio.run(_regime_tools(rig_dir, policy, calls.kit()))
    with pytest.MonkeyPatch.context() as mp:
        _point_repair(mp, tmp_path, kind)
        explode_clients(mp)
        T.patch_detok_load(mp, T.Calls(), tmp_path)
        d_code = run_cli([*T.session_argv(config, tmp_path), "--dry-run", *argv])
    d_out = capsys.readouterr()
    assert d_code == 0, f"premise: the tools-on dry run exits 0:\n{d_out.out}{d_out.err}"
    line = _tools_line(d_out.out + d_out.err).replace(str(tmp_path), "<tmp>")
    return policy, regime, line


@pytest.mark.parametrize(
    ("kind", "hand_regime"),
    [
        pytest.param("repair_on", T.REGIME_ON, id="repair_on"),
        pytest.param("repair_off", T.REGIME_REPAIR_OFF, id="repair_off"),
    ],
)
def test_default_tools_on_labels_carry_loop8_and_no_tool_loop_restores_todays(
    tmp_path, capsys, kind, hand_regime
):
    """Criterion 6 / H-T114-6 / H-T123-4 / H-T123-7: a default tools-on session (``--tools dry-run``, no loop
    setting) reads ``native:<sha12>:dry_run:loop8`` (``:repair_off`` before ``:loop8``) and its dry-run line is
    today's plus `` loop=on(8)``; with ``--no-tool-loop`` both are byte-identical to today's. "Today's" is the
    68d5566 fixture, itself checked against the hand-written label."""
    today = base_fixture()["labels"][kind]
    assert today["regime_tools"] == hand_regime, (
        today
    )  # the fixture agrees with the hand-written value
    if kind == "repair_on":
        assert today["tools_line"] == LINE_ON, today
    else:
        assert today["tools_line"].startswith(
            f"tools=dry_run native:{T.TOOLS_SHA12} repair=off("
        ), today

    policy, regime, line = _labels(tmp_path, capsys, kind)
    assert policy == TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8, ack_clip=False), (
        policy
    )
    assert regime == f"{hand_regime}:loop8", f"T-0123 criterion 6: Regime.tools is {regime!r}"
    assert line == f"{today['tools_line']} loop=on(8){VT}", (
        f"T-0123 criterion 6 (T-0183 A3a: plus the budget): the tools line is {line!r}"
    )

    off_policy, off_regime, off_line = _labels(tmp_path, capsys, kind, "--no-tool-loop")
    assert off_policy == TurnPolicy(
        thinking=True, tool_loop=False, max_tool_steps=8, ack_clip=False
    ), off_policy
    assert off_regime == today["regime_tools"], (
        f"--no-tool-loop restores today's label; got {off_regime!r}"
    )
    assert off_line == today["tools_line"] + VT, (
        f"--no-tool-loop restores today's line (T-0183 A3a: plus the budget); got {off_line!r}"
    )


# --- criterion 7: help ---------------------------------------------------------------------------------------------


def _option_help(text: str, option: str) -> str:
    """The help block of ``option`` in argparse's output, whitespace-normalised and lower-cased."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if re.match(rf"\s+{re.escape(option)}\b", line)), None
    )
    assert start is not None, f"`astra session --help` lists {option}:\n{text}"
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if re.match(r"\s{2}-", line) or not line.strip():
            break
        block.append(line)
    return " ".join(" ".join(block).split()).lower()


def test_help_text_states_the_default(capsys):
    """Criterion 7: ``--tool-loop``'s help says the default is on whenever tools are on and off with tools off
    (not the 68d5566 "default: the profile's ``tool_loop``, off"); ``--max-tool-steps`` says "with the loop on"."""
    code = run_cli(["session", "--help"])
    text = capsys.readouterr().out
    assert code == 0, text
    loop = _option_help(text, "--tool-loop")
    assert "on whenever tools are on" in loop, f"T-0123 criterion 7: --tool-loop help: {loop!r}"
    assert "off with tools off" in loop, f"T-0123 criterion 7: --tool-loop help: {loop!r}"
    assert "`tool_loop`, off)" not in loop, f"the old 'default off' claim is gone: {loop!r}"
    steps = _option_help(text, "--max-tool-steps")
    assert "with the loop on" in steps, f"T-0123 criterion 7: --max-tool-steps help: {steps!r}"
