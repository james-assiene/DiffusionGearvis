"""T-0134c: live mode defaults to T-0134 arm B (action echo in model units plus the "screen now" label; page extent
available but off), resolved in ``cli`` with a source per value (docs/tasks/T-0134c.md, A1-A8).

**The interfaces these rows pin** (the task's; readings in the test-author report):

- ``config.LIVE_DEFAULT_FEEDBACK == "label"``, ``config.LIVE_DEFAULT_ACTION_ECHO == "model"``,
  ``config.LiveFeedbackChoice`` = ``Literal["off", "label", "label+extent"]``, ``config.ActionEcho`` =
  ``Literal["pixels", "model"]``; ``Profile.live_feedback`` and ``Profile.action_echo`` default ``None``; every other
  ``Profile`` field equals BASE's (A1). Load refuses an unknown value (naming the allowed ones) and a non-``None``
  value with an explicit ``tools`` other than ``live`` (naming ``live``).
- ``cli.feedback_of(profile, args, mode) -> cli.Feedback`` (frozen dataclass, fields in the order
  ``live_feedback, live_feedback_from, action_echo, action_echo_from``; ``session_value()``) (A2). **Reading:** the
  flags' argparse dests are ``args.live_feedback`` and ``args.action_echo``, read with ``getattr`` (T-0135 calls it
  with an empty ``Namespace()``).
- ``astra session --live-feedback {off,label,label+extent}`` and ``--action-echo {pixels,model}`` (A3); either one in
  a resolved mode other than ``live`` exits 2 naming ``live``.
- ``cli`` builds ``BrowserToolExecutor(..., echo=<resolved>)`` and ``Session(..., live_feedback=<session_value()>)``
  in live, and passes no ``live_feedback`` keyword in ``off``/``dry_run`` (A4). The rows read the keywords ``cli``
  passes (``fake_live``'s spy executor records them) and then run a rig session built with exactly those values
  over the fake LLM and ``FakeLiveBrowser`` (the T-0133 ``test_default_click_sends_no_reground`` pattern: a
  ``cli``-built graph is never driven on its real clock); the real executor's wording is read over T-0111c's fake
  driver (``test_t0134_echo.make_echo``).
- The ``feedback`` line (A5a), live only, immediately after the ``grounding`` line, literal:
  ``astra session: feedback: live_feedback=<v> (<src>), action_echo=<v> (<src>)``.

The rigs are T-0131's, T-0132's, T-0133's, T-0134's and T-0111g's (``test_cli_t0131``, ``test_cli_t0132`` with
``build``/``Built``, ``test_t0132_live_default``, ``test_t0133_live_reground_off``, ``test_t0134_label``,
``test_t0134_echo``, ``test_cli_tools_live``, ``_live_browser_support``, ``test_session_tool_loop``'s
``_base_tree``), imported, not copied.

Activated at write time (single group, no ``pending`` marker): the behaviour rows fail at BASE 795a16e on a named
``AssertionError``. ``test_other_modes_unchanged``, ``test_grounding_unchanged``,
``test_session_and_executor_defaults_unchanged`` and ``test_frozen_paths_over_closed_range`` are green at BASE.
BASE is 795a16e (the plan commit; ``git diff ca730f5 795a16e`` touches only ``PROGRESS.md`` and ``docs/tasks/``, so
its code is the task's BASE ca730f5's). ``TIP_IMPL`` is bumped by the test-author after each implementation commit.

**Re-scopes (Q-1, named at activation):** R1 ``test_t0131_config.py::test_defaults_unchanged`` (subtracts the two new
fields; green before and after); R2 ``test_t0131_config.py::test_field_set_gains_reground_visual_tokens`` (the exact
field set gains them; red until implemented); R3 ``test_t0133_live_reground_off.py::test_constants`` (orchestrator
ruling 2026-10-23: the two new fields are subtracted from its BASE c29a8d9 comparison, every other field exact; the
module left ``FROZEN``). ``test_cli_tools_live.py``, ``test_t0132_live_default.py`` and the
``make browser-live`` modules pin neither the echo wording nor the label's absence in a ``cli``-built session (the
e2e builds its ``Session`` directly): no edit.

Regime: CPU over fakes, ``ScriptedClock`` for the turn rows, thinking on, 200 ms per word, loop on (8 steps),
1280x800 frames; nothing served; n = 1, deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import io
import json
import re
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.conductor import prompt_builder as pb
from local_astra.config import Profile, load
from tests import _git_range as gr
from tests.unit import _t0125_pins as t0125
from tests.unit import _t0160_pins as t0160
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0181_pins as t0181
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_feedback
from tests.unit._pre_t0155_persona import observed
from tests.unit._t0202_pins import admitted_tools_lines as tools_lines  # T-0202 D3 re-pin
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_session_live import last_user, pass1_body, step_body
from tests.unit.test_t0132_live_default import PREFIX as GROUNDING_PREFIX
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134_echo import BASE_ACTIONS, MODEL_ACTIONS, make_echo
from tests.unit.test_t0134_label import CLAUSE, F0, F1, F2, L1, LU, STEM, all_bodies, label_after

#: BASE: 795a16e (main at the T-0134c plan; its code is ca730f5's, see the docstring). A literal, never a moving ref.
BASE_SHA = "795a16e9301c64cb71ef8e5a1d2150464d995ed8"
#: The implementation tip: 62c3f0e, the approved commit (was 9b75498, then BASE). Bumped by the test-author after
#: each implementation commit.
TIP_IMPL = "62c3f0e8aafeae37c643cf72854c75f7bd2f0a31"
#: docs/02 as the implementation left it, read at a literal: 62c3f0e is the approved tip (``git diff 62c3f0e HEAD --
#: docs/02-architecture.md`` empty at the conversion). Never the working tree, so later docs/02 edits cannot redden
#: the docs row.
DOC_REV = "62c3f0e8aafeae37c643cf72854c75f7bd2f0a31"
HEADER = "**Spec delta (T-0134c"
FEEDBACK_PREFIX = "astra session: feedback: "
#: A5a, written out from the criterion
DEFAULT_FEEDBACK_LINE = (
    FEEDBACK_PREFIX + "live_feedback=label (default), action_echo=model (default)"
)
LIVE = MODE_FLAGS["live"]
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
LF_FLAG, AE_FLAG = "--live-feedback", "--action-echo"
LF_VALUES = ("off", "label", "label+extent")
AE_VALUES = ("pixels", "model")


# --------------------------------------------------------------------------------------------- seams


def require_flags() -> None:
    """Checked before any exit code is read: argparse's own exit 2 on an unknown flag would satisfy a refusal row,
    and its "unrecognized arguments: --live-feedback" would even name ``live``."""
    flags = flags_of(CLI)
    missing = [f for f in (LF_FLAG, AE_FLAG) if f not in flags]
    assert not missing, f"T-0134c A3: astra session {missing} do not exist yet"


def require_fields() -> None:
    missing = [f for f in ("live_feedback", "action_echo") if f not in Profile.model_fields]
    assert not missing, f"T-0134c A1: Profile.{missing} do not exist yet"


def require_resolver() -> tuple[Any, Any]:
    fn = getattr(cli, "feedback_of", None)
    cls = getattr(cli, "Feedback", None)
    assert callable(fn), "T-0134c A2: cli.feedback_of(profile, args, mode) does not exist yet"
    assert isinstance(cls, type) and dataclasses.is_dataclass(cls), (
        "T-0134c A2: cli.Feedback (a frozen dataclass) does not exist yet"
    )
    return fn, cls


def lf(screen_label: bool, page_extent: bool) -> Any:
    """``session.LiveFeedback(screen_label, page_extent)`` (T-0134's, frozen here)."""
    return live_feedback(screen_label=screen_label, page_extent=page_extent)


def feedback_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(FEEDBACK_PREFIX)]


def _names_live(text: str) -> bool:
    """``live`` in ``text`` once the flags' and keys' own spellings are removed (``--live-feedback`` and
    ``live_feedback`` contain ``live``, so a message naming only the flag does not name the mode)."""
    low = text.lower()
    for spelling in ("--live-feedback", "live_feedback", "live-feedback"):
        low = low.replace(spelling, "")
    return re.search(r"\blive\b", low) is not None


def executor_kwargs(calls: T.Calls) -> dict[str, Any]:
    """The keywords ``cli`` passed to ``BrowserToolExecutor`` (``fake_live``'s spy, the kit's live browser)."""
    kit = calls.kit()
    browser = getattr(kit, "live_browser", None)
    assert browser is not None and hasattr(browser, "kwargs"), (
        "premise: cli built the live executor"
    )
    return dict(browser.kwargs)


def session_feedback(calls: T.Calls) -> Any:
    kw = calls.session_kwargs[-1]
    assert "live_feedback" in kw, (
        f"T-0134c A4: cli passes Session(..., live_feedback=...) in live; it passed {sorted(kw)}"
    )
    return kw["live_feedback"]


async def _click_turn(tmp_path: Path, *, echo: str, session_kwargs: dict[str, Any]) -> Any:
    """One live loop turn on a rig session built with ``session_kwargs``: a click (``VALID_CLICK``, 500/250), then a
    report at step 2. Returns the rig after the turn (bodies and the turn line are read off it)."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2], echo=echo)
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
        session_kwargs=session_kwargs,
    ) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        return rig


# --------------------------------------------------------------------------------------------- A1: config


def test_constants_and_profile_fields(tmp_path):
    """A1: the two constants and literals exist with the criterion's values; ``Profile`` gains exactly
    ``live_feedback`` and ``action_echo`` (default ``None`` in every built-in profile), every other field, type and
    default equal to BASE's; ``config`` imports neither ``local_astra.computer`` nor ``conductor.session``."""
    assert getattr(config, "LIVE_DEFAULT_FEEDBACK", None) == "label", (
        "T-0134c A1: config.LIVE_DEFAULT_FEEDBACK = 'label' does not exist yet"
    )
    assert getattr(config, "LIVE_DEFAULT_ACTION_ECHO", None) == "model", (
        "T-0134c A1: config.LIVE_DEFAULT_ACTION_ECHO = 'model' does not exist yet"
    )
    choice, echo = getattr(config, "LiveFeedbackChoice", None), getattr(config, "ActionEcho", None)
    assert choice is not None and typing.get_args(choice) == LF_VALUES, choice
    assert echo is not None and typing.get_args(echo) == AE_VALUES, echo
    require_fields()
    for name in ("live_feedback", "action_echo"):
        f = Profile.model_fields[name]
        assert f.default is None and not f.is_required(), (name, f)
    for name, profile in load(None).profiles.items():
        assert profile.live_feedback is None and profile.action_echo is None, name  # type: ignore[attr-defined]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0133, "BASE_SHA", BASE_SHA)
        base = t0133._base_config_module(tmp_path)
    now = t0133._fields(Profile, "local_astra.config")
    then = t0133._fields(base.Profile, "t0133_base_config")
    for name in ("live_feedback", "action_echo"):
        now.pop(name)
    # T-0135b re-scope R4 (docs/tasks/T-0135b.md A1): T-0135b's ``thought_call_recovery`` is subtracted too (its own
    # row pins it); absent before T-0135b, so the row is green before and after
    now.pop("thought_call_recovery", None)
    # T-0140 re-pin (O-A2, docs/tasks/T-0140.md A5): exactly ``page_overview`` is subtracted too, and it must exist
    assert "page_overview" in now, "T-0140 A5: Profile.page_overview does not exist yet"
    now.pop("page_overview")
    # T-0150 re-pin R1 (docs/tasks/T-0150.md Q-2, A6): exactly ``auto_overview`` may be added (pinned by T-0150's own
    # rows); absent before T-0150, so the row is green before and after
    now.pop("auto_overview", None)
    # T-0151 re-pin R1 (docs/tasks/T-0151.md Q-2, A2-9): exactly ``overview_frame`` may be added (pinned by T-0151's
    # own rows); absent before T-0151 A2, so the row is green before and after
    now.pop("overview_frame", None)
    # T-0136 re-pin (O-C1, docs/tasks/T-0136.md C-1, Q-2): exactly ``action_outcome`` may be added (pinned by
    # test_t0136_cli.py); absent before T-0136 C, so the row is green before and after
    now.pop("action_outcome", None)
    now.pop(
        "repeat_guard", None
    )  # T-0137 re-pin (O-B2): absent before T-0137 B, green before and after
    # T-0125 pin move (Pins, amendment 2): exactly ``search`` is added, defaulting to None
    assert Profile.model_fields["search"].default is None, "T-0125 pin move: Profile.search = None"
    now.pop("search")
    now.pop(
        "browser_web", None
    )  # T-0160 re-pin (C-1): green before and after; pinned by test_cli_tools_web.py
    now = t0172.pop_reask_field(
        now
    )  # T-0172 R3 pin move (B9): exactly announced_call_reask: bool | None = None
    now = t0172.pop_live_guidance_field(now)  # T-0174 R-A1 pin move (A3): exactly live_guidance
    now = t0133.pop_max_tool_steps_web(now)  # T-0178 A7 pin move: exactly max_tool_steps_web
    now = t0133.pop_clip_policy_fields(now)  # T-0182 B pin move
    now = t0133.pop_frame_setting_fields(now)  # T-0227 R-2 re-pin
    assert now == then, "T-0134c A1: every other Profile field, type and default equals BASE's"
    probe = (
        "import sys, json; import local_astra.config; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('local_astra.computer') "
        "or m == 'local_astra.conductor.session')))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [], f"T-0134c A1: config imports {proc.stdout}"


def test_profile_key_refusals(tmp_path):
    """A1: an unknown value of either key is refused at load naming the allowed values; a valid value with
    ``tools = "off"`` or ``tools = "dry_run"`` is refused naming ``live`` (no silent no-op); with ``tools = "live"``
    every value of both keys loads and reads back."""
    require_fields()
    for key, junk, allowed in (
        ("live_feedback", '"full"', LF_VALUES),
        ("live_feedback", "true", LF_VALUES),
        ("action_echo", '"px"', AE_VALUES),
        ("action_echo", '"Model"', AE_VALUES),
    ):
        with pytest.raises(ValueError) as exc:
            load(_toml(tmp_path, f'tools = "live"\n{key} = {junk}\n'))
        assert all(v in str(exc.value) for v in allowed), (key, junk, str(exc.value))
    for tools in ("off", "dry_run"):
        for key, value in (
            ("live_feedback", "label"),
            ("live_feedback", "off"),
            ("action_echo", "model"),
        ):
            with pytest.raises(ValueError) as exc:
                load(_toml(tmp_path, f'tools = "{tools}"\n{key} = "{value}"\n'))
            assert _names_live(str(exc.value)), (tools, key, str(exc.value))
    for v in LF_VALUES:
        for e in AE_VALUES:
            got = load(
                _toml(tmp_path, f'tools = "live"\nlive_feedback = "{v}"\naction_echo = "{e}"\n')
            ).active()
            assert (got.live_feedback, got.action_echo) == (v, e), (v, e)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------- A2/A4: the default


def test_live_default_is_arm_b(tmp_path, capsys, monkeypatch):
    """A2/A4 (mutations 1, 3, 7): no flags and no keys resolve ``Feedback("label", "default", "model", "default")``
    (also with T-0135's empty ``Namespace()``); ``cli`` builds the executor with ``echo="model"`` and the session
    with ``LiveFeedback(True, False)``; that executor's click reads ``clicked at (500, 250)`` (the call's 0-999
    units, T-0134 A2)."""
    feedback_of, feedback = require_resolver()
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    want = feedback("label", "default", "model", "default")
    assert feedback_of(profile, argparse.Namespace(), "live") == want
    assert (
        feedback_of(profile, argparse.Namespace(live_feedback=None, action_echo=None), "live")
        == want
    )
    assert want.session_value() == lf(True, False)
    calls, _ = build(tmp_path, capsys, monkeypatch)
    ekw = executor_kwargs(calls)
    assert ekw.get("echo") == "model", (
        f"T-0134c A4: cli builds BrowserToolExecutor(..., echo='model') by default; kwargs {sorted(ekw)}"
    )
    assert session_feedback(calls) == lf(True, False), (
        f"T-0134c A4: the default session gets LiveFeedback(True, False); got {session_feedback(calls)!r}"
    )
    monkeypatch.undo()

    async def click() -> str:
        from tests.unit.test_t0134_echo import _outcomes

        ex, _ = make_echo(ekw["echo"])
        return str((await _outcomes(ex))["click"].action)

    assert asyncio.run(click()) == MODEL_ACTIONS["click"] == "clicked at (500, 250)"


#: (flags, profile body, expected Feedback fields, expected session_value as (screen_label, page_extent))
SOURCES = {
    "flag-off": (
        [LF_FLAG, "off"],
        "",
        ("off", "flag", "model", "default"),
        (False, False),
    ),
    "flag-label": ([LF_FLAG, "label"], "", ("label", "flag", "model", "default"), (True, False)),
    "profile-label+extent": (
        [],
        'live_feedback = "label+extent"\n',
        ("label+extent", "profile", "model", "default"),
        (True, True),
    ),
    "flag-over-profile": (
        [LF_FLAG, "off"],
        'live_feedback = "label+extent"\n',
        ("off", "flag", "model", "default"),
        (False, False),
    ),
    "echo-flag-pixels": (
        [AE_FLAG, "pixels"],
        "",
        ("label", "default", "pixels", "flag"),
        (True, False),
    ),
    "echo-flag-model-equal-to-default": (
        [AE_FLAG, "model"],
        "",
        ("label", "default", "model", "flag"),
        (True, False),
    ),
    "echo-profile-pixels": (
        [],
        'action_echo = "pixels"\n',
        ("label", "default", "pixels", "profile"),
        (True, False),
    ),
    "echo-flag-over-profile": (
        [AE_FLAG, "model"],
        'action_echo = "pixels"\n',
        ("label", "default", "model", "flag"),
        (True, False),
    ),
    "both-flags": (
        [LF_FLAG, "label+extent", AE_FLAG, "pixels"],
        "",
        ("label+extent", "flag", "pixels", "flag"),
        (True, True),
    ),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources_each_value(tmp_path, capsys, monkeypatch, case):
    """A2 (adversarial; mutation 6, one flag resetting the other value or its source): each value resolves on its
    own (flag, else profile, else default), a flag equal to the default is still ``flag``; the ``feedback`` line
    says the same, and what ``cli`` builds follows: ``Session(live_feedback=<session_value()>)`` and
    ``BrowserToolExecutor(echo=<action_echo>)``."""
    require_flags()
    feedback_of, feedback = require_resolver()
    flags, body, want, sv = SOURCES[case]
    lf_v, lf_src, ae_v, ae_src = want
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    line = FEEDBACK_PREFIX + f"live_feedback={lf_v} ({lf_src}), action_echo={ae_v} ({ae_src})"
    assert feedback_lines(out) == [line], f"T-0134c A2/A5a {case}: printed\n{out}"
    monkeypatch.undo()
    calls, _ = build(tmp_path / "build", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert session_feedback(calls) == lf(*sv), (case, session_feedback(calls))
    assert executor_kwargs(calls).get("echo") == ae_v, (case, executor_kwargs(calls))
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    ns = dict(zip(flags[::2], flags[1::2], strict=True))
    args = argparse.Namespace(live_feedback=ns.get(LF_FLAG), action_echo=ns.get(AE_FLAG))
    got = feedback_of(profile, args, "live")
    assert got == feedback(*want), f"T-0134c A2 {case}: {got}"
    assert got.session_value() == lf(*sv), (case, got.session_value())


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_non_live_unset(mode):
    """A2: in ``off`` and ``dry_run`` both values are ``None`` from ``"unset"`` and ``session_value()`` is
    ``None``, for a profile with no keys and for a live profile whose mode the flag overrode."""
    feedback_of, feedback = require_resolver()
    for profile in (Profile(name="lean"), Profile(name="lean", tools="live")):  # type: ignore[arg-type]
        got = feedback_of(profile, argparse.Namespace(), mode)
        assert got == feedback(None, "unset", None, "unset"), (mode, got)
        assert got.session_value() is None


# --------------------------------------------------------------------------------------------- A5: saying it


def test_feedback_line_exact(tmp_path, capsys, monkeypatch):
    """A5a (mutation 5): under ``--dry-run`` live prints exactly the default literal, on the line right after the
    ``grounding`` line; at a real start it comes right after the ``grounding`` line and before the regime line. A
    non-default run (flag and profile) prints ``live_feedback=label+extent (flag), action_echo=pixels
    (profile)``. Nothing is printed in ``off`` or ``dry_run``."""
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live")
    lines = out.splitlines()
    assert feedback_lines(out) == [DEFAULT_FEEDBACK_LINE], f"T-0134c A5a: printed\n{out}"
    (g,) = [i for i, line in enumerate(lines) if line.startswith(GROUNDING_PREFIX)]
    assert lines[g + 1] == DEFAULT_FEEDBACK_LINE, (
        f"T-0134c A5a: right after the grounding line\n{out}"
    )
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    lines = started.splitlines()
    assert feedback_lines(started) == [DEFAULT_FEEDBACK_LINE], f"A5a at a real start:\n{started}"
    (g,) = [i for i, line in enumerate(lines) if line.startswith(GROUNDING_PREFIX)]
    regime = [i for i, line in enumerate(lines) if line.startswith("astra session: regime:")]
    assert lines[g + 1] == DEFAULT_FEEDBACK_LINE and regime and g + 1 < regime[0], started
    monkeypatch.undo()
    require_flags()
    out = dry_run(
        tmp_path / "other",
        capsys,
        monkeypatch,
        "live",
        LF_FLAG,
        "label+extent",
        body='tools = "live"\naction_echo = "pixels"\n',
    )
    want = FEEDBACK_PREFIX + "live_feedback=label+extent (flag), action_echo=pixels (profile)"
    assert feedback_lines(out) == [want], out
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        other = dry_run(tmp_path / f"m-{mode}", capsys, monkeypatch, mode)
        assert "feedback:" not in other, f"T-0134c A5a: no feedback line with {mode}\n{other}"
        monkeypatch.undo()
        _, started, code = build_out(tmp_path / f"s-{mode}", capsys, monkeypatch, *MODE_FLAGS[mode])
        assert code == 0 and "feedback:" not in started, started


@pytest.mark.parametrize(
    "flags",
    [[], [LF_FLAG, "off"], [AE_FLAG, "pixels"], [LF_FLAG, "label+extent", AE_FLAG, "model"]],
    ids=["default", "feedback-off", "echo-pixels", "both"],
)
# T-0160 ruling 2: web on by default in live (egress=web)
def test_tools_line_unchanged(tmp_path, capsys, monkeypatch, flags):
    """A5b: the live ``--dry-run`` tools line is T-0133's (BASE's plus `` visual_tokens=1120``) whatever the feedback
    flags; ``cli.tools_line`` returns BASE's strings."""
    if flags:
        require_flags()
    out = dry_run(tmp_path, capsys, monkeypatch, "live", *flags)
    # T-0160 pin move (C-2): the default live line reads egress=web
    assert tools_lines(out) == [t0160.web_line(BASE_LINES["live"]) + " visual_tokens=1120"], out
    kit = T.make_kit()
    live = cli.LiveLine(browser="headless", viewport=(1280, 800))
    assert cli.tools_line(kit, 8, live) == BASE_LINES["live"]
    assert cli.tools_line(kit, 8) == BASE_LINES["dry_run"]
    assert cli.tools_line(None) == BASE_LINES["off"]


# --------------------------------------------------------------------------------------------- A4/A5c: on the wire


def test_default_body_has_one_label_no_extent(tmp_path, capsys, monkeypatch):
    """A4/A5c: a session built with exactly what a default ``cli`` live run passes (its ``live_feedback`` and its
    executor's ``echo``): the step-1 request has exactly one A3 label (``L1``) right after the current frame, step 2
    exactly one (``LU``) after the post-click frame, no body contains ``Page beyond this screen``, the click reads
    in 0-999 units, and the turn line carries ``live_feedback == {"screen_label": true, "page_extent": false,
    "echo": "model"}`` with no ``*_from`` key (R-k)."""
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch)
    feedback = session_feedback(calls)
    echo = executor_kwargs(calls).get("echo", "pixels")
    monkeypatch.undo()
    rig = asyncio.run(
        _click_turn(tmp_path / "turn", echo=echo, session_kwargs={"live_feedback": feedback})
    )
    assert label_after(pass1_body(rig), F0) == L1, (
        "T-0134c A4: step 1 carries one label after the frame"
    )
    s2 = step_body(rig, 2)
    assert label_after(s2, F1) == LU, "T-0134c A4: step 2 carries one label after the frame"
    for body in all_bodies(rig):
        assert CLAUSE not in json.dumps(body), "T-0134c A4: no page-extent clause by default"
    assert "clicked at (500, 250)" in json.dumps(s2), (
        "T-0134c A4: the default echo is in the call's 0-999 units"
    )
    extras = dict(rig.record("t1").extras)
    assert extras.get("live_feedback") == {
        "screen_label": True,
        "page_extent": False,
        "echo": "model",
    }, f"T-0134c A5c: the default turn line's live_feedback; extras {extras}"
    sources = [k for k in extras if k.endswith("_from") and ("feedback" in k or "echo" in k)]
    assert sources == [] and "action_echo" not in extras, (
        f"R-k: no source key on the turn line; {extras}"
    )


def test_off_is_arm_a_bodies(tmp_path, capsys, monkeypatch):
    """A6 (adversarial; mutation 2, ``off`` passing ``None``, and the mutant ``off`` passing ``LiveFeedback(True,
    False)`` or leaving the echo at ``model``): ``--live-feedback off --action-echo pixels`` makes ``cli`` pass
    ``LiveFeedback(False, False)`` (R-m) and an executor echo of ``pixels``. A session built with those has step-1
    and step-2 bodies equal to a session built with ``live_feedback=None`` and the executor's default echo (BASE's
    default live session), and its turn line differs by exactly the ``live_feedback`` key, ``{"screen_label":
    false, "page_extent": false, "echo": "pixels"}``. The real executor's click with that echo reads BASE's
    wording."""
    require_flags()
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch, LF_FLAG, "off", AE_FLAG, "pixels")
    feedback = session_feedback(calls)
    assert feedback == lf(False, False), (
        f"T-0134c A5c/R-m: --live-feedback off passes LiveFeedback(False, False); got {feedback!r}"
    )
    echo = executor_kwargs(calls).get("echo", "pixels")
    assert echo == "pixels", f"T-0134c A6: --action-echo pixels; the executor got echo={echo!r}"
    monkeypatch.undo()
    arm_a = asyncio.run(
        _click_turn(tmp_path / "a", echo=echo, session_kwargs={"live_feedback": feedback})
    )
    base = asyncio.run(_click_turn(tmp_path / "base", echo="pixels", session_kwargs={}))
    assert pass1_body(arm_a) == pass1_body(base), "T-0134c A6: step-1 bodies equal BASE's default"
    assert step_body(arm_a, 2) == step_body(base, 2), (
        "T-0134c A6: step-2 bodies equal BASE's default"
    )
    assert STEM not in json.dumps(all_bodies(arm_a)), "premise: no label anywhere"
    assert "clicked at (640, 200)" in json.dumps(step_body(arm_a, 2)), "premise: the pixel echo"
    a_extras, b_extras = dict(arm_a.record("t1").extras), dict(base.record("t1").extras)
    assert "live_feedback" not in b_extras, "premise: BASE's default turn line has no live_feedback"
    assert a_extras.pop("live_feedback", None) == {
        "screen_label": False,
        "page_extent": False,
        "echo": "pixels",
    }, f"T-0134c A5c: the off turn line records the state; extras {a_extras}"
    assert a_extras == b_extras, "T-0134c A6: the turn lines differ only by live_feedback"
    assert last_user(pass1_body(arm_a)) == last_user(pass1_body(base))

    async def clicks() -> tuple[str, str]:
        from tests.unit.test_t0134_echo import _outcomes

        ex, _ = make_echo(echo)
        dflt, _ = make_echo(None)
        return str((await _outcomes(ex))["click"].action), str(
            (await _outcomes(dflt))["click"].action
        )

    got, default = asyncio.run(clicks())
    assert got == default == BASE_ACTIONS["click"], (got, default)


# --------------------------------------------------------------------------------------------- A3: refusals


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize(
    "flag", [[LF_FLAG, "label"], [LF_FLAG, "off"], [AE_FLAG, "model"], [AE_FLAG, "pixels"]]
)
def test_flags_refused_outside_live(tmp_path, capsys, monkeypatch, mode, flag):
    """A3 (mutation 4, a flag accepted silently in ``dry_run``): either flag, with any value, in ``off`` or
    ``dry_run`` exits 2 with a message naming ``live`` (not argparse's usage error), builds no graph and starts no
    page server, with ``--dry-run`` and without. Also through a live profile whose mode ``--tools`` overrode."""
    require_flags()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        for extra in (["--dry-run"], []):
            monkeypatch.setenv("DISPLAY", ":99")
            calls = T.Calls()
            T.fake_clients(monkeypatch, calls)
            T.stop_before_serving(monkeypatch, calls)
            T.patch_detok_load(monkeypatch, calls, tmp_path)
            events = Events()
            fake_live(monkeypatch, events)
            code = run_cli([*argv(tmp_path, *flags, *flag, config_body=body), *extra])
            out, err = _out(capsys)
            assert code == 2 and _names_live(out + err), (
                f"T-0134c A3 {mode} {flag} {extra}: exit 2 naming live; got {code}\n{out}{err}"
            )
            assert "unrecognized arguments" not in err, err
            assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
            assert not [e for e in events if e[0].startswith("pages.")], list(events)
            monkeypatch.undo()


#: (case, profile keys) for the ruling row below
PROFILE_KEYS = {
    "live_feedback": 'live_feedback = "label"\n',
    "live_feedback-off": 'live_feedback = "off"\n',
    "action_echo": 'action_echo = "pixels"\n',
    "both": 'live_feedback = "label+extent"\naction_echo = "model"\n',
}


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
@pytest.mark.parametrize("case", list(PROFILE_KEYS))
def test_profile_keys_refused_when_tools_overridden(tmp_path, capsys, monkeypatch, case, override):
    """Orchestrator ruling 2026-10-23 (the ``reground_refusal`` pattern, T-0131 B9): a profile with
    ``tools = "live"`` and a ``live_feedback``/``action_echo`` key loads, but run with ``--tools off`` or
    ``--tools dry-run`` the resolved mode is not live, so the key would silently do nothing: ``cli`` exits 2
    naming ``live``, builds no graph and starts no page server, with ``--dry-run`` and without. Too-wide companion:
    the same profile with no override is accepted and builds."""
    require_fields()
    body = LIVE_PROFILE + PROFILE_KEYS[case]
    (tmp_path / "p").mkdir()
    assert load(_toml(tmp_path / "p", body)).active().tools == "live", "premise: the profile loads"
    for extra in (["--dry-run"], []):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        events = Events()
        fake_live(monkeypatch, events)
        code = run_cli([*argv(tmp_path, *override, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and _names_live(out + err), (
            f"ruling: profile {case} under {override} {extra} exits 2 naming live; got {code}\n{out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
        assert not [e for e in events if e[0].startswith("pages.")], list(events)
        monkeypatch.undo()
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs, "the same profile without an override is accepted"


@pytest.mark.parametrize(
    "junk", [[LF_FLAG, "full"], [LF_FLAG, "extent"], [AE_FLAG, "px"], [AE_FLAG, "Model"]]
)
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """A3: an unknown value exits 2 through argparse (its ``invalid choice`` message)."""
    require_flags()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *LIVE, *junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


# --------------------------------------------------------------------------------------------- A2: purity

_PURE = """
import argparse, builtins, json, socket, sys
from local_astra import cli
from local_astra.config import Profile
before = sorted(m for m in sys.modules if m.startswith("local_astra.computer"))
opened = []
real_open = builtins.open
def spy(*a, **k):
    opened.append(str(a[0]) if a else "?")
    return real_open(*a, **k)
class NoSocket:
    def __init__(self, *a, **k):
        raise AssertionError("feedback_of opened a socket")
socket.socket = NoSocket
builtins.open = spy
try:
    out = []
    for p, mode in ((Profile(name="lean", tools="live"), "live"), (Profile(name="lean"), "off"),
                    (Profile(name="lean"), "dry_run")):
        f = cli.feedback_of(p, argparse.Namespace(), mode)
        out.append(repr(f.session_value()))
    f = cli.feedback_of(Profile(name="lean", tools="live"),
                        argparse.Namespace(live_feedback="label+extent", action_echo="pixels"), "live")
    out.append(repr(f.session_value()))
finally:
    builtins.open = real_open
after = sorted(m for m in sys.modules if m.startswith("local_astra.computer"))
print(json.dumps({"before": before, "after": after, "opened": opened, "values": out}))
"""


def test_feedback_of_pure():
    """A2: ``feedback_of`` (and ``session_value()``) opens no socket and no file and imports no
    ``local_astra.computer`` module (``sys.modules`` before and after, in a fresh interpreter)."""
    require_resolver()
    proc = subprocess.run(
        [sys.executable, "-c", _PURE], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, f"T-0134c A2: feedback_of is not pure:\n{proc.stderr[-2000:]}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["before"] == [], f"premise: importing cli loads no computer module; {got['before']}"
    assert got["after"] == [], f"T-0134c A2: feedback_of imported {got['after']}"
    assert got["opened"] == [], f"T-0134c A2: feedback_of opened {got['opened']}"
    assert got["values"][1:3] == ["None", "None"], got["values"]
    assert "True" in got["values"][0] and "True" in got["values"][3], got["values"]


# --------------------------------------------------------------------------------------------- A7: docs and help


def test_help_texts_state_defaults():
    """A7: the ``--live-feedback`` help names the three values and the ``label`` default; the ``--action-echo``
    help names both values and the ``model`` default. Keywords, never wording."""
    require_flags()
    h = t0133._help_of(LF_FLAG)
    assert all(v in h for v in LF_VALUES) and "default" in h.lower() and "live" in h, h
    h = t0133._help_of(AE_FLAG)
    assert all(v in h for v in AE_VALUES) and "default" in h.lower(), h


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0134c():
    """A7: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0134c``, naming the constants, the
    flags, ``label+extent``, the human ruling, the refusal outside live, the ``feedback`` line and that ``off``
    passes ``LiveFeedback(False, False)``; removing it once leaves docs/02 equal to BASE's paragraph for paragraph
    (the T-0134 paragraph verbatim). Keywords, never wording. Both sides are fixed revisions (the T-0133 ``DOC_REV``
    conversion, done at approval): BASE 795a16e and ``DOC_REV`` 62c3f0e, never the working tree."""
    gr.require_commits(BASE_SHA, DOC_REV)
    base_doc = _doc02_at(BASE_SHA)
    doc = _doc02_at(DOC_REV)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0134c A7: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "LIVE_DEFAULT_FEEDBACK": "LIVE_DEFAULT_FEEDBACK" in para,
        "LIVE_DEFAULT_ACTION_ECHO": "LIVE_DEFAULT_ACTION_ECHO" in para,
        "--live-feedback": "--live-feedback" in para,
        "--action-echo": "--action-echo" in para,
        "label+extent": "label+extent" in para,
        "ruling": "ruling" in low or "ruled" in low,
        "feedback line": "feedback" in low and "line" in low,
        "LiveFeedback(False, False)": "LiveFeedback(False, False)" in para.replace("`", ""),
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0134c A7: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0134c A7: docs/02 changes only by the one delta paragraph (the T-0134 paragraph verbatim)"
    )


# --------------------------------------------------------------------------------------------- A8: other modes vs BASE

#: (case, flags, profile body) of the ``--dry-run`` and build comparisons (off and dry_run only)
#: T-0183 re-scope (docs/tasks/T-0183.md, "Differentials against a BASE that has ``--visual-tokens``"; the T-0174 D4b
#: pattern on both sides): every case passes ``--visual-tokens 1120`` on both sides (BASE has the flag), since T-0183
#: makes 1120 the default in ``off``/``dry_run``; nothing else changes.
VT1120 = ("--visual-tokens", "1120")
OTHER: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("off", VT1120, ""),
    ("dry_run", ("--tools", "dry-run", *VT1120), ""),
    ("dry_run-no-loop", ("--tools", "dry-run", "--no-tool-loop", *VT1120), ""),
    ("live-profile-off", ("--tools", "off", *VT1120), LIVE_PROFILE),
    ("live-profile-dry_run", ("--tools", "dry-run", *VT1120), LIVE_PROFILE),
)


def _norm(v: Any, tmp: str) -> Any:
    """``tmp`` and the imported tree's root (the BASE tree in the subprocess, the worktree here) replaced."""
    if not isinstance(v, str):
        return v
    root = str(Path(cli.__file__).resolve().parents[1])
    return v.replace(tmp, "<tmp>").replace(root, "<tree>")


def _describe(v: Any, tmp: str, depth: int = 0) -> Any:
    """A JSON-able description of a keyword value: primitives as they are, containers element-wise, a dataclass's
    fields one level deep, any other object by its type name."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return _norm(v, tmp)
    if isinstance(v, Path):
        return _norm(str(v), tmp)
    if depth >= 2:
        return f"<{type(v).__name__}>"
    if isinstance(v, dict):
        return {str(k): _describe(x, tmp, depth + 1) for k, x in sorted(v.items(), key=str)}
    if isinstance(v, (list, tuple)):
        return [_describe(x, tmp, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(str(_describe(x, tmp, depth + 1)) for x in v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            "<type>": type(v).__name__,
            **{
                f.name: _describe(getattr(v, f.name), tmp, depth + 1) for f in dataclasses.fields(v)
            },
        }
    return f"<{type(v).__name__}>"


def without_t0140_kit_field(got: list[Any], want: list[Any]) -> list[Any]:
    """T-0140 re-pin (O-A2, A6): ``ToolKit`` gains exactly one optional tool-set field whose default ``None`` means v1;
    its name is the implementer's. Each described ``tool_kit`` may carry at most one key that BASE's lacks, and only
    with the value ``None`` (off/``dry_run`` never offer the overview); that key is dropped, nothing else is."""
    out = []
    for i, kw in enumerate(got):
        kit = kw.get("tool_kit") if isinstance(kw, dict) else None
        base_kit = want[i].get("tool_kit") if i < len(want) and isinstance(want[i], dict) else None
        if isinstance(kit, dict) and isinstance(base_kit, dict):
            extra = [k for k in kit if k not in base_kit]
            assert len(extra) <= 1 and all(kit[k] is None for k in extra), (
                f"T-0140 A6: the kit gains at most one field, None outside an overview session; extra {extra}"
            )
            kw = {**kw, "tool_kit": {k: v for k, v in kit.items() if k not in extra}}
        out.append(kw)
    return out


def _other_modes(tmp: Path) -> dict[str, Any]:
    """For each ``OTHER`` case: the ``--dry-run`` output, and a build's output, client and session keywords, the
    executors made, the system message and one request body the session's own (real, refusing) client builds."""
    from local_astra.conductor import llm_client
    from tests.fakes.llm import _refusing_transport
    from tests.unit._live_browser_support import live_frame
    from tests.unit.test_cli_session import explode_clients

    out: dict[str, Any] = {}
    for case, flags, body in OTHER:
        where = tmp / case
        for sub in ("dry", "build"):
            (where / sub).mkdir(parents=True, exist_ok=True)
        w = str(where)
        stdout, stderr = io.StringIO(), io.StringIO()
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
            T.patch_detok_load(mp, T.Calls(), where)
            explode_clients(mp)
            fake_live(mp, Events())
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = run_cli([*argv(where / "dry", *flags, config_body=body), "--dry-run"])
        dry = {"code": code, "out": _norm(stdout.getvalue(), w), "err": _norm(stderr.getvalue(), w)}
        stdout, stderr = io.StringIO(), io.StringIO()
        made: list[Any] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
            mp.setenv("DISPLAY", ":99")
            calls = T.Calls()
            T.fake_clients(mp, calls)
            clients: list[Any] = []
            real = llm_client.VllmClient

            def vllm(
                *a: Any,
                _c: list[Any] = clients,
                _calls: T.Calls = calls,
                _real: Any = real,
                **k: Any,
            ) -> Any:
                _calls.vllm_kwargs.append(dict(k))
                client = _real(*a, transport=_refusing_transport(), **k)
                _c.append(client)
                return client

            mp.setattr(cli, "VllmClient", vllm)
            T.stop_before_serving(mp, calls)
            T.patch_detok_load(mp, calls, where)
            made = fake_live(mp, Events())
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                bcode = run_cli(argv(where / "build", *flags, config_body=body))
        assert bcode == 0 and calls.graphs, f"premise: {case} builds; {bcode}\n{stderr.getvalue()}"
        session = calls.graphs[-1].session
        prompt = session.prompts.full(
            session.prompts.prefix([], live_frame("onset")), "what is on the page", None, []
        )
        out[case] = {
            "dry": dry,
            "build": {
                "code": bcode,
                "out": _norm(stdout.getvalue(), w),
                "err": _norm(stderr.getvalue(), w),
            },
            "vllm_kwargs": [_describe(k, w) for k in calls.vllm_kwargs],
            "session_kwargs": [_describe(k, w) for k in calls.session_kwargs],
            "executors": len(made),
            "system": session.prompts.system_message().text(),
            "body": json.loads(json.dumps(clients[0].request_body(prompt, "x", None, {}))),
        }
    return out


def _grounding(tmp: Path) -> dict[str, Any]:
    """For each case of T-0133's live tables (``OPT_IN``, ``SCOPE``, ``OFF``): every ``grounding_of`` call (mode and
    result) and the printed ``grounding`` lines, under ``--dry-run`` and at a build."""
    from tests.unit.test_cli_session import explode_clients

    table: dict[str, tuple[list[str], str]] = {}
    for name, rows in (("opt_in", t0133.OPT_IN), ("scope", t0133.SCOPE), ("off", t0133.OFF)):
        for case, row in rows.items():
            table[f"{name}:{case}"] = (list(row[0]), row[1])
    out: dict[str, Any] = {}
    real = cli.grounding_of
    for case, (flags, body) in table.items():
        where = tmp / case.replace(":", "_")
        for sub in ("dry", "build"):
            (where / sub).mkdir(parents=True, exist_ok=True)
        seen: list[Any] = []

        def spy(profile: Any, args: Any, mode: str, _s: list[Any] = seen) -> Any:
            got = real(profile, args, mode)
            _s.append({"mode": mode, **dataclasses.asdict(got)})
            return got

        res: dict[str, Any] = {}
        for kind in ("dry", "build"):
            stdout = io.StringIO()
            with pytest.MonkeyPatch.context() as mp:
                mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
                mp.setenv("DISPLAY", ":99")
                calls = T.Calls()
                T.patch_detok_load(mp, calls, where)
                if kind == "dry":
                    explode_clients(mp)
                else:
                    T.fake_clients(mp, calls)
                    T.stop_before_serving(mp, calls)
                fake_live(mp, Events())
                mp.setattr(cli, "grounding_of", spy)
                extra = ["--dry-run"] if kind == "dry" else []
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    code = run_cli([*argv(where / kind, *LIVE, *flags, config_body=body), *extra])
            res[kind] = {
                "code": code,
                "lines": [
                    x for x in stdout.getvalue().splitlines() if x.startswith(GROUNDING_PREFIX)
                ],
                "kw": calls.session_kwargs[-1].get("grounding") if calls.session_kwargs else None,
            }
        res["calls"] = list(seen)
        out[case] = res
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    return {"other": _other_modes(tmp / "other"), "grounding": _grounding(tmp / "grounding")}


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0134c_live_feedback_default as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0134c_base")
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


def _tmpless(tree: Any, tmp: Path) -> Any:
    """``tree`` with the run's own tmp root replaced (the two runs have different roots)."""
    return json.loads(json.dumps(tree).replace(str(tmp), "<root>"))


# T-0125 pin move: ToolKit.web_search (None with search off)
def test_other_modes_unchanged(base: dict[str, Any], tmp_path: Path) -> None:
    """A8 (green at BASE; mutations 4 and 5's other half): for ``off`` and ``dry_run`` (plain, loop off, and a live
    profile under ``--tools off``/``--tools dry-run``) the printed ``--dry-run`` output, a build's output, the
    client and session keywords (no ``live_feedback`` keyword), the executors made (none), the system message and
    one request body equal BASE's."""
    # T-0155 O-8: the persona sentence only, on the observed side
    now = {"other": observed(_other_modes(tmp_path / "other"))}
    for case, *_ in OTHER:
        for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
            # T-0172 R3 pin move (B9): dry_run gains the reask line and keyword, off nothing
            got = t0172.admit_reask_other(case, key, now["other"][case][key])
            if key == "session_kwargs":
                # T-0142 R-10: A6 passes degeneracy_guard=True in every mode; only that key is dropped
                got = [{k: v for k, v in kw.items() if k != "degeneracy_guard"} for kw in got]
                got = t0125.kit_without_web_search(got)  # T-0125 pin move
                got = without_t0140_kit_field(got, base["other"][case][key])
                got = t0181.without_mic_unavailable(got)  # T-0181 A-R13 pin move
            assert got == base["other"][case][key], (
                f"T-0134c A8: {case} {key} differs from {BASE_SHA[:7]}:\n"
                f"base {json.dumps(base['other'][case][key])[:1500]}\n"
                f"now  {json.dumps(now['other'][case][key])[:1500]}"
            )
        assert all("live_feedback" not in kw for kw in now["other"][case]["session_kwargs"]), case
        assert "feedback:" not in now["other"][case]["dry"]["out"], case
    assert now["other"]["off"]["executors"] == 0 and now["other"]["off"]["body"]["messages"], (
        "premise"
    )


def test_grounding_unchanged(base: dict[str, Any], tmp_path: Path) -> None:
    """A8 (green at BASE): ``cli.grounding_of`` (every call: mode and resolved values) and the printed ``grounding``
    lines are BASE's for every input of T-0133's live tables, under ``--dry-run`` and at a build."""
    now = {"grounding": _grounding(tmp_path / "grounding")}
    assert set(now["grounding"]) == set(base["grounding"]), "premise: the same table"
    for case in now["grounding"]:
        got, want = now["grounding"][case], base["grounding"][case]
        assert got == want, (
            f"T-0134c A8: grounding for {case} differs from {BASE_SHA[:7]}: {got} vs {want}"
        )
        assert got["calls"] and got["dry"]["lines"], f"premise: {case} resolved and printed"


# --------------------------------------------------------------------------------------------- A8: frozen

#: A8's list, written out from the criterion, plus the pinned module that stays unedited (T-0133's left: R3)
FROZEN = (
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/perception",
    "local_astra/pages",
    "hud",
    "services",
    "scripts/gpu",
    "scripts/locked",
    "tests/unit/test_t0134_differential.py",
)


def test_session_and_executor_defaults_unchanged():
    """A8 (green at BASE): ``Session``'s ``live_feedback`` keyword still defaults to ``None`` and
    ``BrowserToolExecutor``'s ``echo`` to ``"pixels"`` (T-0132 R-f: the default lives in ``cli``)."""
    import inspect

    from local_astra.conductor import session as session_mod
    from tests.unit.test_computer_executor import computer_executor

    params = inspect.signature(session_mod.Session.__init__).parameters
    assert params["live_feedback"].default is None
    params = inspect.signature(computer_executor().BrowserToolExecutor.__init__).parameters
    assert params["echo"].default == "pixels"


def test_frozen_paths_over_closed_range():
    """A8 (green at BASE): A8's paths (``conductor/**``, ``computer/**``, ``toolcall/**``, ``contracts/**``, the
    fakes, perception, pages, the HUD, ``services/``, ``scripts/gpu``, ``scripts/locked``) and the pinned
    module (``test_t0134_differential.py``) are byte-identical over ``BASE_SHA..TIP_IMPL`` (two literals, never the working tree), with no file added
    or removed; ``LIVE_TOOL_GUIDANCE`` and ``tools_v1.json`` keep their shas. Power: the same predicate sees
    T-0111g's ``versions.lock`` change."""
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
        f"T-0134c A8: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    from tests.unit import test_t0134_differential as t0134

    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    blob = gr._git("show", f"{TIP_IMPL}:local_astra/toolcall/tools_v1.json")
    assert blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest().startswith(
        "bbfc53cc"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN), "power"
