"""T-0178 group A: the web step budget (docs/tasks/T-0178.md A1-A6, rulings H-T178-2 and H-T178-6).

The oracle is the task's resolution order, written out by hand as ``WANT`` below: ``--max-tool-steps N`` if given;
else, in live with web on, ``Profile.max_tool_steps_web`` (default 20); else ``Profile.max_tool_steps`` (default 8).
Every cell is checked three ways: ``cli.tool_loop_of`` on a hand-built namespace, the ``TurnPolicy`` a real start
hands to the machine (over the CLI rig's fakes, stopped before serving), and the ``--dry-run`` tools line.

Readings (in the test-author's report):

- A2's "byte-identical to BASE" for the non-web modes is pinned here on the lines and values group A can move (the
  tools line, the policy, the regime suffix, the absence of an A4 line); the whole-output BASE differentials are the
  existing rows that ``_t0160_pins`` and the A8 rigs carry (``test_t0136_cli``, ``test_t0137_differential``,
  ``test_t0152_overview_frame_default``, ``test_t0169_web_start_page``, ``test_t0174_v1_identity``).
- A3's turn-line ``regime.tools`` is read through ``test_cli_tool_loop._regime_tools`` (one typed turn on a rig
  session built with the policy ``cli`` resolved, the default ``dry_run`` kit): its ``:loop<N>`` suffix comes from
  the policy alone (``Session._tools_regime``), which is what group A changes.
- A4's real-start line is checked to appear once on stdout among the tools block; its exact neighbour is pinned for
  ``--dry-run`` (the line right after the tools line, as A4 says).

Regime: CPU over the CLI rig's fakes (``tests/unit/_tools_switch_support.py``, ``test_cli_tools_live.fake_live``), no
browser, nothing served, profile ``lean`` from a temporary ``astra.toml``; no measurement.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pydantic
import pytest

from local_astra import cli
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.config import Profile
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tool_loop import _regime_tools
from tests.unit.test_cli_tools_live import Events, argv, fake_live


def require_key() -> None:
    """The A1 key; its absence is a named assertion, never a config-load error inside a rig."""
    assert "max_tool_steps_web" in Profile.model_fields, (
        "T-0178 A1: Profile.max_tool_steps_web does not exist yet"
    )


LINE_PREFIX = "astra session: tool loop: "
#: mode -> extra argv
MODES: dict[str, list[str]] = {
    "off": [],
    "dry_run": ["--tools", "dry-run"],
    "live-no-web": ["--tools", "live", "--browser", "headless", "--no-web"],
    "live-web": ["--tools", "live", "--browser", "headless"],
}
#: T-0182 group B (B-R5, the CLI-built policy moves): ``cli`` builds ``ack_clip=False`` (filler clips off by default)
#: and, in live with the loop on, ``still_working_after_ms = Profile.still_working_after_s * 1000`` (default 20 s).
STILL_WORKING_LIVE_MS = 20_000


def _clip_fields(mode: str, on: bool) -> dict[str, Any]:
    """The two T-0182 fields of the CLI-built policy for a start in ``mode`` with the loop ``on`` (by hand, B2/B4)."""
    live = mode.startswith("live")
    return {
        "ack_clip": False,
        "still_working_after_ms": STILL_WORKING_LIVE_MS if live and on else None,
    }


#: override -> (extra argv, profile body)
OVERRIDES: dict[str, tuple[list[str], str]] = {
    "none": ([], ""),
    "flag5": (["--max-tool-steps", "5"], ""),
    "key3": ([], "max_tool_steps = 3\n"),
    "webkey12": ([], "max_tool_steps_web = 12\n"),
}
REFUSED = "refused"
#: (mode, override) -> (loop on, max_tool_steps) or REFUSED, by hand from A1/A2/A5 and H-T178-2
WANT: dict[tuple[str, str], Any] = {
    ("off", "none"): (False, 8),
    ("off", "flag5"): REFUSED,  # A5: --max-tool-steps with tools off, BASE's refusal
    ("off", "key3"): (False, 3),  # a profile's own key is never refused (inert)
    ("off", "webkey12"): (False, 8),  # max_tool_steps_web is inert where it does not apply
    ("dry_run", "none"): (True, 8),
    ("dry_run", "flag5"): (True, 5),
    ("dry_run", "key3"): (True, 3),
    ("dry_run", "webkey12"): (True, 8),
    ("live-no-web", "none"): (True, 8),
    ("live-no-web", "flag5"): (True, 5),
    ("live-no-web", "key3"): (True, 3),
    ("live-no-web", "webkey12"): (True, 8),
    ("live-web", "none"): (True, 20),  # A1, H-T178-6
    ("live-web", "flag5"): (True, 5),  # the flag first
    ("live-web", "key3"): (True, 20),  # H-T178-2: max_tool_steps does not apply in web mode
    ("live-web", "webkey12"): (True, 12),
}
#: A4: the mode's default
MODE_DEFAULT = {"off": 8, "dry_run": 8, "live-no-web": 8, "live-web": 20}
#: BASE's two refusal texts (``cli.tool_loop_of`` at 0fd25414), the oracle of A5
REFUSE_TOOLS_OFF = (
    "--max-tool-steps is refused with tools off (the loop runs only with tools on, so it "
    "would be ignored); add --tools dry-run, or drop --max-tool-steps"
)
REFUSE_LOOP_OFF = (
    "--max-tool-steps is refused with the tool loop off (it would be ignored); add "
    "--tool-loop, or drop --max-tool-steps"
)


# --- rigs ------------------------------------------------------------------------------------------------------------


def _dry(tmp: Path, flags: list[str], body: str) -> tuple[int, str, str]:
    tmp.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISPLAY", ":99")
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        T.patch_detok_load(mp, calls, tmp)
        explode_clients(mp)
        fake_live(mp, Events())
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli([*argv(tmp, *flags, config_body=body), "--dry-run"])
    return code, out.getvalue(), err.getvalue()


def _start(tmp: Path, flags: list[str], body: str) -> tuple[int, T.Calls, str, str]:
    tmp.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISPLAY", ":99")
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        T.fake_clients(mp, calls)
        T.stop_before_serving(mp, calls)
        T.patch_detok_load(mp, calls, tmp)
        fake_live(mp, Events())
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli(argv(tmp, *flags, config_body=body))
    return code, calls, out.getvalue(), err.getvalue()


def _tools_line(out: str) -> tuple[int, str]:
    lines = out.splitlines()
    at = [i for i, x in enumerate(lines) if x.startswith("tools=")]
    assert len(at) == 1, f"one tools line under --dry-run; got:\n{out}"
    return at[0], lines[at[0]]


def _loop_word(line: str) -> str | None:
    words = [w for w in line.split() if w.startswith("loop=")]
    assert len(words) <= 1, line
    return words[0] if words else None


def _budget_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(LINE_PREFIX)]


def _namespace(mode: str, flag: int | None) -> argparse.Namespace:
    tools = {"off": "off", "dry_run": "dry_run", "live-no-web": "live", "live-web": "live"}[mode]
    return argparse.Namespace(
        tools=tools,
        tool_loop=None,
        max_tool_steps=flag,
        no_web=mode == "live-no-web",
        browser="headless",
    )


def _profile(override: str) -> Profile:
    keys: dict[str, Any] = {}
    if override == "key3":
        keys["max_tool_steps"] = 3
    if override == "webkey12":
        keys["max_tool_steps_web"] = 12
    return Profile(name="lean", **keys)


# --- A1/A2/A5: resolution ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "override"), sorted(WANT), ids=[f"{m}/{o}" for m, o in sorted(WANT)]
)
def test_resolution_by_mode(tmp_path, mode, override):
    """A1/A2 (H-T178-2): the hand table ``WANT``, through ``cli.tool_loop_of``, the started session's ``TurnPolicy``
    and the ``--dry-run`` tools line's ``loop=on(<N>)``; the refused cell is refused with exit 2 and BASE's text."""
    want = WANT[(mode, override)]
    flags, body = OVERRIDES[override]
    assert "max_tool_steps_web" in Profile.model_fields, "T-0178 A1: Profile.max_tool_steps_web"
    flag = 5 if override == "flag5" else None
    resolved = cli.tool_loop_of(_profile(override), _namespace(mode, flag))
    argv_ = [*MODES[mode], *flags]
    code, calls, out, err = _start(tmp_path / "start", argv_, body)
    if want == REFUSED:
        assert resolved.refusal == REFUSE_TOOLS_OFF, resolved
        assert code == 2 and REFUSE_TOOLS_OFF in err and not calls.graphs, (code, err)
        return
    on, steps = want
    assert (resolved.on, resolved.max_tool_steps, resolved.refusal) == (on, steps, None), (
        f"T-0178 A1: tool_loop_of for {mode}/{override} is {resolved}; want on={on} steps={steps}"
    )
    assert code == 0 and calls.graphs, f"{mode}/{override}: exit {code}\n{out}{err}"
    policy = calls.graphs[-1].session.machine.policy
    assert policy == TurnPolicy(
        thinking=True, tool_loop=on, max_tool_steps=steps, **_clip_fields(mode, on)
    ), f"T-0178 A1/A2: the TurnPolicy built for {mode}/{override} is {policy}"
    d_code, d_out, d_err = _dry(tmp_path / "dry", argv_, body)
    assert d_code == 0, d_err
    _, line = _tools_line(d_out)
    assert _loop_word(line) == (f"loop=on({steps})" if on else None), line


def test_the_table_is_the_whole_product():
    assert set(WANT) == {(m, o) for m in MODES for o in OVERRIDES}


def test_profile_max_tool_steps_does_not_leak_into_web(tmp_path):
    """Adversarial (H-T178-2; mutation: web mode reads ``max_tool_steps`` when the profile sets it): ``max_tool_steps
    = 3`` with web on resolves to 20; the same profile with ``--no-web`` resolves to 3."""
    body = "max_tool_steps = 3\n"
    for mode, steps in (("live-web", 20), ("live-no-web", 3)):
        code, calls, _out, err = _start(tmp_path / mode, MODES[mode], body)
        assert code == 0 and calls.graphs, err
        got = calls.graphs[-1].session.machine.policy.max_tool_steps
        assert got == steps, (
            f"T-0178 H-T178-2: {mode} with max_tool_steps = 3 resolves to {got}, not {steps}"
        )


def test_profile_max_tool_steps_above_web_default_does_not_leak_into_web(tmp_path):
    """Amendment A1b (H-T178-2; mutation: web mode reads ``max(max_tool_steps_web, max_tool_steps)``, which the
    ``max_tool_steps = 3`` row above cannot see): an explicit ``max_tool_steps = 30`` with web on still resolves to 20,
    through ``cli.tool_loop_of`` and the started session's ``TurnPolicy``; with ``--no-web`` the same profile resolves
    to 30 (the key is read where it applies, so the row is not vacuous)."""
    require_key()
    body = "max_tool_steps = 30\n"
    profile = Profile(name="lean", max_tool_steps=30)
    for mode, steps in (("live-web", 20), ("live-no-web", 30)):
        resolved = cli.tool_loop_of(profile, _namespace(mode, None))
        assert (resolved.on, resolved.max_tool_steps, resolved.refusal) == (True, steps, None), (
            f"T-0178 A1b: tool_loop_of for {mode} with max_tool_steps = 30 is {resolved}; want {steps}"
        )
        code, calls, _out, err = _start(tmp_path / mode, MODES[mode], body)
        assert code == 0 and calls.graphs, err
        got = calls.graphs[-1].session.machine.policy.max_tool_steps
        assert got == steps, (
            f"T-0178 A1b: {mode} with max_tool_steps = 30 starts with {got} steps, not {steps}"
        )


# --- A3: web mode is visible -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "steps"),
    [
        pytest.param([], 20, id="default"),
        pytest.param(["--max-tool-steps", "5"], 5, id="flag5"),  # holds at BASE: the flag wins
    ],
)
def test_web_dry_run_line_and_regime_say_20(tmp_path, flags, steps):
    """A3: in live with web on, the ``--dry-run`` tools line reads ``loop=on(20)`` and a turn on the policy ``cli``
    resolved records ``regime.tools`` ending ``:loop20``; with ``--max-tool-steps 5`` both read 5."""
    code, out, err = _dry(tmp_path / "dry", [*MODES["live-web"], *flags], "")
    assert code == 0, err
    _, line = _tools_line(out)
    assert _loop_word(line) == f"loop=on({steps})" and " egress=web " in line, line
    code, calls, out, err = _start(tmp_path / "start", [*MODES["live-web"], *flags], "")
    assert code == 0 and calls.graphs, err
    policy = calls.graphs[-1].session.machine.policy
    rig_dir = tmp_path / "rig"
    rig_dir.mkdir()
    regime = asyncio.run(_regime_tools(rig_dir, policy, T.make_kit()))
    assert regime.endswith(f":loop{steps}"), f"T-0178 A3: regime.tools is {regime!r}"


# --- A2: the non-web modes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["off", "dry_run", "live-no-web", "live-browser_web-false"])
def test_non_web_modes_byte_identical_to_base(tmp_path, mode):
    """A2 (negative): off, dry_run, live ``--no-web`` and live with ``browser_web = false`` keep BASE's 8: the policy
    is ``TurnPolicy(thinking=True, tool_loop=<BASE's>, max_tool_steps=8)``, the tools line's loop word is BASE's
    (``loop=on(8)`` or none for off), the regime ends ``:loop8`` where the loop is on, and no A4 line is printed.
    ``TurnPolicy.max_tool_steps`` still defaults to 8."""
    assert "max_tool_steps_web" in Profile.model_fields, "T-0178 A1: Profile.max_tool_steps_web"
    assert TurnPolicy(thinking=True).max_tool_steps == 8
    flags, body = (
        (MODES["live-web"], "browser_web = false\n")
        if mode == "live-browser_web-false"
        else (MODES[mode], "")
    )
    on = mode != "off"
    code, calls, out, err = _start(tmp_path / "start", flags, body)
    assert code == 0 and calls.graphs, err
    policy = calls.graphs[-1].session.machine.policy
    assert policy == TurnPolicy(
        thinking=True, tool_loop=on, max_tool_steps=8, **_clip_fields(mode, on)
    ), policy
    assert _budget_lines(out + err) == [], out
    d_code, d_out, d_err = _dry(tmp_path / "dry", flags, body)
    assert d_code == 0, d_err
    _, line = _tools_line(d_out)
    assert _loop_word(line) == ("loop=on(8)" if on else None), line
    assert _budget_lines(d_out + d_err) == [], d_out
    if on:
        rig_dir = tmp_path / "rig"
        rig_dir.mkdir()
        assert asyncio.run(_regime_tools(rig_dir, policy, T.make_kit())).endswith(":loop8")


# --- A4: the budget line ---------------------------------------------------------------------------------------------

#: (mode, argv, body) -> the expected line, or None (by hand from A4)
LINE_CASES = [
    ("live-web", [], "", None),
    ("live-web", ["--max-tool-steps", "20"], "", None),  # adversarial: equals web's default
    (
        "live-web",
        ["--max-tool-steps", "8"],
        "",
        f"{LINE_PREFIX}8 steps (--max-tool-steps)",
    ),  # adversarial
    ("live-web", [], "max_tool_steps_web = 12\n", f"{LINE_PREFIX}12 steps (max_tool_steps_web)"),
    ("live-web", [], "max_tool_steps = 3\n", None),  # inert in web mode (H-T178-2)
    ("dry_run", [], "", None),
    ("dry_run", ["--max-tool-steps", "8"], "", None),
    ("dry_run", ["--max-tool-steps", "20"], "", f"{LINE_PREFIX}20 steps (--max-tool-steps)"),
    ("dry_run", [], "max_tool_steps = 3\n", f"{LINE_PREFIX}3 steps (max_tool_steps)"),
    ("dry_run", [], "max_tool_steps_web = 12\n", None),  # inert outside web mode
    ("live-no-web", [], "max_tool_steps = 3\n", f"{LINE_PREFIX}3 steps (max_tool_steps)"),
    ("live-no-web", [], "", None),
    ("off", [], "", None),
]


@pytest.mark.parametrize(
    ("mode", "flags", "body", "want"),
    [
        pytest.param(
            *case,
            id=str(i),
        )
        for i, case in enumerate(LINE_CASES)
    ],
)
def test_budget_line_only_when_not_default(tmp_path, mode, flags, body, want):
    """A4: ``astra session: tool loop: <N> steps (<source>)`` exactly when N differs from the mode's default (8, or
    20 in web mode), right after the tools line under ``--dry-run`` and once in a real start's output; none with
    defaults in any mode."""
    if "max_tool_steps_web" in body:
        require_key()
    argv_ = [*MODES[mode], *flags]
    code, out, err = _dry(tmp_path / "dry", argv_, body)
    assert code == 0, err
    got = _budget_lines(out + err)
    assert got == ([want] if want else []), f"T-0178 A4 ({mode} {flags} {body!r}): {got}\n{out}"
    if want:
        at, _ = _tools_line(out)
        assert out.splitlines()[at + 1] == want, (
            f"T-0178 A4: the line follows the tools line:\n{out}"
        )
    code, _calls, out, err = _start(tmp_path / "start", argv_, body)
    assert code == 0, err
    assert [x for x in out.splitlines() if x.startswith(LINE_PREFIX)] == ([want] if want else []), (
        out
    )
    assert LINE_PREFIX not in err, err


# --- A1: the key, A5: refusals, A6: help ------------------------------------------------------------------------------


def test_profile_key_validated():
    """A1: ``max_tool_steps_web: int = Field(default=20, ge=1)``: 0 and -1 raise at load, 1 and 12 load;
    ``max_tool_steps`` keeps 8."""
    f = Profile.model_fields.get("max_tool_steps_web")
    assert f is not None and f.default == 20, "T-0178 A1: Profile.max_tool_steps_web = 20"
    assert Profile(name="lean").max_tool_steps == 8
    for bad in (0, -1):
        with pytest.raises(pydantic.ValidationError):
            Profile(name="lean", max_tool_steps_web=bad)
    for ok in (1, 12):
        assert Profile(name="lean", max_tool_steps_web=ok).max_tool_steps_web == ok  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("flags", "body", "text"),
    [
        (["--max-tool-steps", "3"], "", REFUSE_TOOLS_OFF),
        (["--tools", "dry-run", "--no-tool-loop", "--max-tool-steps", "3"], "", REFUSE_LOOP_OFF),
    ],
    ids=["tools-off", "loop-off"],
)
def test_refusals_unchanged(tmp_path, flags, body, text):
    """A5 (negative; holds at BASE): ``--max-tool-steps`` with tools off or the loop off is refused with BASE's text
    (exit 2, nothing built)."""
    code, calls, _out, err = _start(tmp_path / "r", flags, body)
    assert code == 2 and text in err and not calls.graphs, (code, err)


def test_profile_web_key_never_refused(tmp_path):
    """A5: a profile's own ``max_tool_steps_web`` is never refused: tools off, dry_run, and dry_run with the loop off
    all start."""
    require_key()
    for i, f in enumerate(([], ["--tools", "dry-run"], ["--tools", "dry-run", "--no-tool-loop"])):
        code, calls, _out, err = _start(tmp_path / f"ok{i}", f, "max_tool_steps_web = 12\n")
        assert code == 0 and calls.graphs, (
            f"T-0178 A5: max_tool_steps_web is never refused ({f}): {err}"
        )


def test_help_says_20_in_web(capsys):
    """A6: the ``--max-tool-steps`` help says "default 8; 20 in live with web on" (whitespace-normalised)."""
    from tests.unit.test_cli_tool_loop_default import _option_help

    assert run_cli(["session", "--help"]) == 0
    text = _option_help(capsys.readouterr().out, "--max-tool-steps")
    assert "default 8; 20 in live with web on" in text, f"T-0178 A6: {text!r}"
