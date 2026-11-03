"""T-0135b: live mode recovers a tool call left at the tail of an open thought by default (T-0135 arm B), resolved in
``cli`` with a source; ``--thought-call-recovery off`` restores arm A (docs/tasks/T-0135b.md, A1-A9).

**The interfaces these rows pin** (the task's; readings in the test-author report):

- ``config.LIVE_DEFAULT_THOUGHT_CALL_RECOVERY: bool = True``; ``Profile.thought_call_recovery: bool | None = None``;
  every other ``Profile`` field, type and default equals BASE's (A1). Load refuses a non-bool value (a
  ``ValueError`` naming ``true`` and ``false``; TOML ``"off"``, ``"true"``, ``1`` included, so a lax pydantic
  coercion fails the row) and a non-``None`` value with ``tools`` other than ``live`` (naming ``live``).
- ``cli.recovery_of(profile, args, mode) -> cli.Recovery``, a frozen dataclass with fields in the order
  ``thought_call_recovery, thought_call_recovery_from`` and ``session_value()`` (A2). **Reading:** the flag's
  argparse dest is ``args.thought_call_recovery`` holding the spelled choice (``"on"``/``"off"``), read with
  ``getattr``; ``Recovery.thought_call_recovery`` holds the resolved bool.
- ``astra session --thought-call-recovery {on,off}`` (A3); in a resolved mode other than ``live`` the flag, or the
  profile key under a ``--tools`` override, exits 2 naming ``live``.
- ``cli`` passes ``Session(..., recover_thought_calls=<session_value()>)`` in live, and no such keyword in
  ``off``/``dry_run`` (A4). As in T-0134c, the rows read the keywords ``cli`` passes (``T.Calls``' spy) and then run a
  rig session built with exactly the behaviour keywords ``cli`` passed (``CARRIED``: ``recover_thought_calls``,
  ``degeneracy_guard``, ``live_feedback``) over the fake LLM and ``FakeLiveBrowser``; a ``cli``-built graph is never
  driven on its real clock.
- The ``recovery`` line (A5a), live only, immediately after the T-0134c ``feedback`` line, literal:
  ``astra session: recovery: thought_call_recovery=<on|off> (<src>)``.

The rigs are T-0131's, T-0132's, T-0134c's, T-0135's, T-0142's and T-0111g's (``test_cli_t0131``, ``test_cli_t0132``
with ``build``, ``test_cli_tools_live``, ``test_t0134c_live_feedback_default``, ``test_t0135_live_recovery``,
``test_t0142_session``, ``_live_browser_support``), imported, not copied.

Activated at write time (single group, no ``pending`` marker): the behaviour rows fail at BASE 0248cfd on a named
``AssertionError``. ``test_other_modes_unchanged``, ``test_session_default_unchanged``,
``test_eval_exemption_adversarial`` and ``test_frozen_paths_over_closed_range`` are green at BASE. ``TIP_IMPL`` and
``DOC_REV`` are bumped by the test-author after each implementation commit.

**Re-scopes (Q-1, named at activation):** R1 ``test_t0131_config.py::test_defaults_unchanged`` (subtracts
``thought_call_recovery``; green before and after); R2 ``test_t0131_config.py::test_field_set_gains_reground_visual_tokens``
(the exact field set gains it; red until implemented); R3 ``test_t0133_live_reground_off.py::test_constants``
(subtracts it from the BASE c29a8d9 comparison; green before and after); R4
``test_t0134c_live_feedback_default.py::test_constants_and_profile_fields`` (subtracts it from the same comparison;
green before and after). ``test_cli_tools_live.py``, the T-0132/T-0133/T-0134c ``cli`` rows and the
``make browser-live`` modules (which build their ``Session`` directly, not through ``cli``) pin neither the set of
``cli``'s session keywords nor the absence of ``recover_thought_calls`` on a ``cli``-built turn line: no edit.

Regime: CPU over fakes, ``ScriptedClock`` for the turn rows, thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, ReGround off; nothing served; n = 1, deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
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
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0135_live_recovery as t0135
from tests.unit import test_t0142_session as t0142
from tests.unit._live_browser_support import feedback_rig
from tests.unit._pre_t0155_persona import observed
from tests.unit._t0202_pins import admitted_tools_lines as tools_lines  # T-0202 D3 re-pin
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, dry_run
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history
from tests.unit.test_t0132_live_default import PREFIX as GROUNDING_PREFIX
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import (
    DEFAULT_FEEDBACK_LINE,
    FEEDBACK_PREFIX,
    _names_live,
)

#: BASE: 0248cfd (main after ``Merge T-0142``, the task's BASE). A literal, never a moving ref.
BASE_SHA = "0248cfdaa56792836f2b2e7e7c72a9c5d9c68af3"
#: The implementation tip: b0b9ee1 (``T-0135b-impl``; was BASE). Bumped by the test-author after each implementation
#: commit.
TIP_IMPL = "b0b9ee14f1360e83fa1f1e0b74401ac15664a24e"
#: docs/02 as the implementation left it, read at a literal: b0b9ee1 holds the §5 paragraph (was BASE); never the
#: working tree, so later docs/02 edits cannot redden the docs row
DOC_REV = "b0b9ee14f1360e83fa1f1e0b74401ac15664a24e"
#: docs/02 the paragraph is added to: BASE, or the merge parent if T-0142b's paragraph lands first (the T-0134 A9
#: precedent); bumped with ``DOC_REV``
DOC_PARENT = "0248cfdaa56792836f2b2e7e7c72a9c5d9c68af3"
HEADER = "**Spec delta (T-0135b"
KEY = "recover_thought_calls"
FROM = "from_thought"
GUARD = "degeneracy_guard"
FIELD = "thought_call_recovery"
FLAG = "--thought-call-recovery"
RECOVERY_PREFIX = "astra session: recovery: "
#: A5a, written out from the criterion
DEFAULT_RECOVERY_LINE = RECOVERY_PREFIX + "thought_call_recovery=on (default)"
OFF_FLAG_LINE = RECOVERY_PREFIX + "thought_call_recovery=off (flag)"
#: A5b: BASE's grounding line for a default live session (T-0132 A4a with T-0133's ReGround off), by hand
DEFAULT_GROUNDING_LINE = GROUNDING_PREFIX + "visual_tokens=1120 (default), reground=off (default)"
LIVE = MODE_FLAGS["live"]
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
#: the behaviour keywords ``cli`` passes that the rig session is built with
CARRIED = (KEY, GUARD, "live_feedback")
S1, S3 = tsl.S1, tsl.S3


# --------------------------------------------------------------------------------------------- seams


def require_flag() -> None:
    """Checked before any exit code is read: argparse's own exit 2 on an unknown flag would satisfy a refusal row."""
    assert FLAG in flags_of(CLI), f"T-0135b A3: astra session {FLAG} does not exist yet"


def require_field() -> None:
    assert FIELD in Profile.model_fields, f"T-0135b A1: Profile.{FIELD} does not exist yet"


def require_resolver() -> tuple[Any, Any]:
    fn = getattr(cli, "recovery_of", None)
    cls = getattr(cli, "Recovery", None)
    assert callable(fn), "T-0135b A2: cli.recovery_of(profile, args, mode) does not exist yet"
    assert isinstance(cls, type) and dataclasses.is_dataclass(cls), (
        "T-0135b A2: cli.Recovery (a frozen dataclass) does not exist yet"
    )
    assert cls.__dataclass_params__.frozen, "T-0135b A2: Recovery is frozen"  # type: ignore[attr-defined]
    assert [f.name for f in dataclasses.fields(cls)] == [FIELD, FIELD + "_from"], (
        f"T-0135b A2: Recovery's fields; got {[f.name for f in dataclasses.fields(cls)]}"
    )
    return fn, cls


def recovery_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(RECOVERY_PREFIX)]


def carried(calls: T.Calls) -> dict[str, Any]:
    """The behaviour keywords ``cli`` passed to ``Session`` (``CARRIED``), those it passed only."""
    kw = calls.session_kwargs[-1]
    return {k: kw[k] for k in CARRIED if k in kw}


def session_recovery(calls: T.Calls) -> Any:
    kw = calls.session_kwargs[-1]
    assert KEY in kw, (
        f"T-0135b A4: cli passes Session(..., {KEY}=...) in live; it passed {sorted(kw)}"
    )
    return kw[KEY]


def _load_error(tmp_path: Path, body: str) -> str | None:
    """The message of the ``ValueError`` loading ``body`` raises, or ``None`` if it loads (never ``pytest.raises``,
    whose "did not raise" is not an ``AssertionError``)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        load(_toml(tmp_path, body))
    except ValueError as exc:
        return str(exc)
    return None


async def _tail_turn(tmp: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """T-0135's D-5 turn on a rig session built with ``kwargs``: a click at step 1, then a step 2 whose only commit
    is an open thought ending in a valid ``type_at``, then a report."""
    tmp.mkdir(parents=True, exist_ok=True)
    browser = t0135.browser3()
    async with feedback_rig(
        tmp, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs
    ) as rig:
        await t0135.run_turn(rig, ([t0135.d5()], [answer(S3)]))
        extras = dict(rig.record("t1").extras)
    # the session log is read once the rig has closed (flushed), as T-0135's rows do
    return {
        "invocations": t0135.invocations(browser),
        "records": t0135.records_of(rig, "t1-call-s2"),
        "all_records": list(rig.tool_records()),
        "extras": extras,
    }


async def _two_turns(tmp: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The D-5 loop turn, then a toolless live turn, on a rig session built with ``kwargs``; every section
    (T-0121's capture) and the executor's invocations."""
    tmp.mkdir(parents=True, exist_ok=True)
    browser = t0135.browser3()
    async with feedback_rig(
        tmp,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps([t0135.d5()], [answer(S3)])
        await rig.ask(tsl.QUESTION)
        await t0107a._until_turns(rig, 1, "the D-5 turn")
        rig.llm.script = [answer("It is a small shop. ")]
        await rig.ask("what is on the page")
        await t0107a._until_turns(rig, 2, "the toolless live turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out = _with_history(rig, tmp, session_id, log_path)
    out["invocations"] = t0135.invocations(browser)
    return out


#: A9's step 2: the logged S2#15 collapse (T-0142's fixture, every commit an open thought), then one more open
#: thought commit holding the valid D-5 ``type_at``, so the request's final open block ends in a tagged call
TRIP_STREAM = [
    *t0142.commits_of(t0142.S215),
    thought(" " + t0135.TYPE_D5, 19, open_=True),
]


async def _trip_turn(
    tmp: Path, kwargs: dict[str, Any], late: bool, *, retry_trips: bool = False
) -> dict[str, Any]:
    """T-0142's ``guard_rig`` (live kit, ``degeneracy_guard=kwargs[GUARD]``, ``LateLlm(late=late)``) with the other
    ``kwargs`` added to ``Session``; step 1 clicks, step 2 is ``TRIP_STREAM``, a step 3 report is scripted.

    ``retry_trips`` (T-0179 B-11): ``TRIP_STREAM`` is scripted for step 2's one degenerate retry as well (the guarded
    run); the unguarded power run never retries, so it keeps one ``TRIP_STREAM`` and the report."""
    from tests.unit import _tool_turn_support as tts

    tmp.mkdir(parents=True, exist_ok=True)
    others = {k: v for k, v in kwargs.items() if k != GUARD}
    real = tts.Session

    class WithCli(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in others.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    tts.Session = WithCli  # type: ignore[misc]
    try:
        async with t0142.guard_rig(tmp, guard=kwargs[GUARD], llm=t0142.LateLlm(late=late)) as rig:
            steps = (TRIP_STREAM, TRIP_STREAM) if retry_trips else (TRIP_STREAM,)
            await t0142.run(rig, *steps, [answer(S3)])
            extras = dict(rig.record("t1").extras)
    finally:
        tts.Session = real  # type: ignore[misc]
    # the session log is read once the rig has closed (flushed), as T-0142's rows do
    return {
        "notes": t0142.notes(rig),
        "errors": [e.message for e in t0142.errors(rig, "t1-gen-s2")],
        "invocations": t0135.invocations(rig.browser),
        "records": list(rig.tool_records()),
        "extras": extras,
    }


# --------------------------------------------------------------------------------------------- A1: config


def test_constant_and_profile_field(tmp_path):
    """A1: ``LIVE_DEFAULT_THOUGHT_CALL_RECOVERY`` is the bool ``True``; ``Profile.thought_call_recovery`` is
    ``bool | None`` defaulting to ``None`` (``None`` in every built-in profile); every other field, type and default
    equals BASE's; ``config`` imports nothing from ``local_astra.conductor`` or ``local_astra.computer``."""
    value = getattr(config, "LIVE_DEFAULT_THOUGHT_CALL_RECOVERY", None)
    assert value is True, (
        f"T-0135b A1: config.LIVE_DEFAULT_THOUGHT_CALL_RECOVERY = True does not exist yet; got {value!r}"
    )
    require_field()
    f = Profile.model_fields[FIELD]
    assert f.default is None and not f.is_required(), f
    assert f.annotation == (bool | None), f.annotation
    for name, profile in load(None).profiles.items():
        assert getattr(profile, FIELD) is None, name
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0133, "BASE_SHA", BASE_SHA)
        base = t0133._base_config_module(tmp_path)
    now = t0133._fields(Profile, "local_astra.config")
    then = t0133._fields(base.Profile, "t0133_base_config")
    now.pop(FIELD)
    # T-0140 re-pin (O-A2, docs/tasks/T-0140.md A5): exactly ``page_overview`` is subtracted too, and it must exist
    assert "page_overview" in now, "T-0140 A5: Profile.page_overview does not exist yet"
    now.pop("page_overview")
    # T-0150 re-pin R2 (docs/tasks/T-0150.md Q-2, A6): exactly ``auto_overview`` may be added (pinned by T-0150's own
    # rows); absent before T-0150, so the row is green before and after
    now.pop("auto_overview", None)
    # T-0151 re-pin R2 (docs/tasks/T-0151.md Q-2, A2-9): exactly ``overview_frame`` may be added (pinned by T-0151's
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
    )  # T-0172 R2 pin move (B9): exactly announced_call_reask: bool | None = None
    now = t0172.pop_live_guidance_field(now)  # T-0174 R-A2 pin move (A3): exactly live_guidance
    now = t0133.pop_max_tool_steps_web(now)  # T-0178 A7 pin move: exactly max_tool_steps_web
    now = t0133.pop_clip_policy_fields(now)  # T-0182 B pin move
    now = t0133.pop_frame_setting_fields(now)  # T-0227 R-2 re-pin
    assert now == then, "T-0135b A1: every other Profile field, type and default equals BASE's"
    probe = (
        "import sys, json; import local_astra.config; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('local_astra.computer', "
        "'local_astra.conductor')))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [], f"T-0135b A1: config imports {proc.stdout}"


def test_profile_key_refusals(tmp_path):
    """A1: a non-bool value is refused at load naming ``true`` and ``false`` (strings and integers a lax bool would
    coerce included); a non-``None`` value with ``tools = "off"``, ``"dry_run"`` or no ``tools`` key is refused
    naming ``live``; with ``tools = "live"`` both bools load and read back as bools."""
    require_field()
    for junk in ('"off"', '"on"', '"true"', '"false"', "1", "0", '"yes"'):
        msg = _load_error(
            tmp_path / f"junk-{junk.strip(chr(34))}", f'tools = "live"\n{FIELD} = {junk}\n'
        )
        assert msg is not None, f"T-0135b A1: {FIELD} = {junk} loaded; a non-bool is refused"
        low = msg.lower()
        assert "true" in low and "false" in low, (
            f"T-0135b A1: the refusal of {FIELD} = {junk} names true and false; got {msg!r}"
        )
    for tools in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        for value in ("true", "false"):
            msg = _load_error(tmp_path / f"t-{tools[9:12]}-{value}", f"{tools}{FIELD} = {value}\n")
            assert msg is not None and _names_live(msg), (
                f"T-0135b A1: {FIELD} = {value} with {tools.strip() or 'no tools key'} is refused naming live; "
                f"got {msg!r}"
            )
    for value, want in (("true", True), ("false", False)):
        (tmp_path / f"ok-{value}").mkdir()
        got = getattr(
            load(_toml(tmp_path / f"ok-{value}", f'tools = "live"\n{FIELD} = {value}\n')).active(),
            FIELD,
        )
        assert got is want, (value, got)


# --------------------------------------------------------------------------------------------- A2/A4: the default


def test_live_default_is_on(tmp_path, capsys, monkeypatch):
    """A2/A4 (mutation 1): no flag and no key resolve ``Recovery(True, "default")`` (also with T-0140's empty
    ``Namespace()``), ``session_value()`` is ``True``; ``off``/``dry_run`` resolve ``Recovery(None, "unset")`` with
    ``session_value()`` ``None``, for a live profile too. ``cli`` builds the default live session with
    ``recover_thought_calls=True``, and passes no such keyword in ``off``/``dry_run`` (mutation 5)."""
    recovery_of, recovery = require_resolver()
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    want = recovery(True, "default")
    assert recovery_of(profile, argparse.Namespace(), "live") == want
    assert recovery_of(profile, argparse.Namespace(**{FIELD: None}), "live") == want
    assert want.session_value() is True
    for mode in ("off", "dry_run"):
        for p in (Profile(name="lean"), profile):  # type: ignore[arg-type]
            got = recovery_of(p, argparse.Namespace(), mode)
            assert got == recovery(None, "unset") and got.session_value() is None, (mode, got)
    calls, _ = build(tmp_path / "live", capsys, monkeypatch)
    assert session_recovery(calls) is True, (
        f"T-0135b A4: the default live session gets {KEY}=True; got {session_recovery(calls)!r}"
    )
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        calls = build_mode(tmp_path / mode, capsys, monkeypatch, mode)
        assert KEY not in calls.session_kwargs[-1], (
            f"T-0135b A4: no {KEY} keyword in {mode}; got {calls.session_kwargs[-1].get(KEY)!r}"
        )


#: (flags, profile body, expected (value, source)): flag, else profile, else default; each source on its own
SOURCES = {
    "flag-off": ([FLAG, "off"], "", (False, "flag")),
    "flag-on-equal-to-default": ([FLAG, "on"], "", (True, "flag")),
    "profile-false": ([], f"{FIELD} = false\n", (False, "profile")),
    "profile-true": ([], f"{FIELD} = true\n", (True, "profile")),
    "flag-on-over-profile-false": ([FLAG, "on"], f"{FIELD} = false\n", (True, "flag")),
    "flag-off-over-profile-true": ([FLAG, "off"], f"{FIELD} = true\n", (False, "flag")),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources(tmp_path, capsys, monkeypatch, case):
    """A2/A5a (mutation 2, ``off`` mapped to ``None``): the flag, else the profile key, else the default; a flag
    equal to the default is still ``flag``. The ``recovery`` line says the same, and ``cli`` passes
    ``recover_thought_calls=<the bool>`` (``False`` for off, never ``None``)."""
    require_flag()
    require_field()
    recovery_of, recovery = require_resolver()
    flags, body, (value, src) = SOURCES[case]
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    word = "on" if value else "off"
    assert recovery_lines(out) == [RECOVERY_PREFIX + f"thought_call_recovery={word} ({src})"], (
        f"T-0135b A2/A5a {case}: printed\n{out}"
    )
    monkeypatch.undo()
    calls, _ = build(tmp_path / "build", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert session_recovery(calls) is value, (case, session_recovery(calls))
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    args = argparse.Namespace(**{FIELD: flags[1] if flags else None})
    got = recovery_of(profile, args, "live")
    assert got == recovery(value, src), f"T-0135b A2 {case}: {got}"
    assert got.session_value() is value, (case, got.session_value())


# --------------------------------------------------------------------------------------------- A5: saying it


def test_recovery_line_exact(tmp_path, capsys, monkeypatch):
    """A5a (mutation 4): under ``--dry-run`` live prints exactly the default literal on the line right after the
    ``feedback`` line, and at a real start too; ``--thought-call-recovery off`` prints the ``off (flag)`` literal;
    nothing is printed in ``off`` or ``dry_run`` (under ``--dry-run`` and at a start)."""
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live")
    lines = out.splitlines()
    assert recovery_lines(out) == [DEFAULT_RECOVERY_LINE], f"T-0135b A5a: printed\n{out}"
    (f,) = [i for i, line in enumerate(lines) if line.startswith(FEEDBACK_PREFIX)]
    assert lines[f + 1] == DEFAULT_RECOVERY_LINE, (
        f"T-0135b A5a: right after the feedback line\n{out}"
    )
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    lines = started.splitlines()
    assert recovery_lines(started) == [DEFAULT_RECOVERY_LINE], f"A5a at a real start:\n{started}"
    (f,) = [i for i, line in enumerate(lines) if line.startswith(FEEDBACK_PREFIX)]
    assert lines[f + 1] == DEFAULT_RECOVERY_LINE, started
    monkeypatch.undo()
    require_flag()
    out = dry_run(tmp_path / "off-flag", capsys, monkeypatch, "live", FLAG, "off")
    assert recovery_lines(out) == [OFF_FLAG_LINE], out
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        other = dry_run(tmp_path / f"m-{mode}", capsys, monkeypatch, mode)
        assert "recovery:" not in other, f"T-0135b A5a: no recovery line with {mode}\n{other}"
        monkeypatch.undo()
        _, started, code = build_out(tmp_path / f"s-{mode}", capsys, monkeypatch, *MODE_FLAGS[mode])
        assert code == 0 and "recovery:" not in started, started


@pytest.mark.parametrize("flags", [[], [FLAG, "off"], [FLAG, "on"]], ids=["default", "off", "on"])
# T-0160 ruling 2: web on by default in live (egress=web)
def test_existing_lines_unchanged(tmp_path, capsys, monkeypatch, flags):
    """A5b (mutation 4's other half, the recovery merged into the ``feedback`` line): the ``grounding`` line, the
    ``feedback`` line and the live ``--dry-run`` tools line are BASE's literals whatever the recovery flag, under
    ``--dry-run`` and (default) at a real start; ``cli.tools_line`` returns BASE's strings."""
    if flags:
        require_flag()
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live", *flags)
    # T-0160 pin move (C-2): the default live line reads egress=web
    assert tools_lines(out) == [t0160.web_line(BASE_LINES["live"]) + " visual_tokens=1120"], out
    assert [x for x in out.splitlines() if x.startswith(GROUNDING_PREFIX)] == [
        DEFAULT_GROUNDING_LINE
    ], out
    assert t0134c.feedback_lines(out) == [DEFAULT_FEEDBACK_LINE], out
    if not flags:
        monkeypatch.undo()
        _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
        assert code == 0, started
        assert [x for x in started.splitlines() if x.startswith(GROUNDING_PREFIX)] == [
            DEFAULT_GROUNDING_LINE
        ], started
        assert t0134c.feedback_lines(started) == [DEFAULT_FEEDBACK_LINE], started
    kit = T.make_kit()
    live = cli.LiveLine(browser="headless", viewport=(1280, 800))
    assert cli.tools_line(kit, 8, live) == BASE_LINES["live"]
    assert cli.tools_line(kit, 8) == BASE_LINES["dry_run"]
    assert cli.tools_line(None) == BASE_LINES["off"]


# --------------------------------------------------------------------------------------------- A4/A5c: the turn


def test_default_recovers_tail_call(tmp_path, capsys, monkeypatch):
    """A4/A5c (mutation 1): a session built with exactly the behaviour keywords a default ``cli`` live run passes:
    a step 2 whose only commit is an open thought tailed by a valid ``type_at`` executes that call (``t1-call-s2``,
    the D-5 arguments), every ``tool.call`` record of it carries ``from_thought: true``, and the turn line carries
    ``recover_thought_calls: true`` (a bare bool) with no source key (T-0134c R-k)."""
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch)
    kwargs = carried(calls)
    monkeypatch.undo()
    assert kwargs.get(KEY) is True, (
        f"T-0135b A4: a default cli live session gets {KEY}=True; cli passed {sorted(kwargs)} "
        f"({kwargs.get(KEY)!r})"
    )
    got = asyncio.run(_tail_turn(tmp_path / "turn", kwargs))
    assert got["invocations"] == [
        ("t1-call", tsl.CLICK_ARGS),
        ("t1-call-s2", t0135.TYPE_ARGS),
    ], f"T-0135b A4: the recovered call executes; executed {got['invocations']}"
    recs = got["records"]
    assert [r.get("status") for r in recs] == ["proposed", "executed"], recs
    assert all(r.get(FROM) is True for r in recs), (
        f"T-0135b A4: from_thought on every record; {recs}"
    )
    extras = got["extras"]
    assert extras.get(KEY) is True, f"T-0135b A5c: the turn line says {KEY}: true; {extras}"
    extra_keys = [k for k in extras if "recover" in k and k != KEY]
    assert extra_keys == [], f"R-k: no source key on the turn line; {extra_keys}"


def test_off_is_arm_a(tmp_path, capsys, monkeypatch):
    """A6 (adversarial; mutation 2, ``off`` passing ``None``, and the mutant that keeps ``True`` for ``off``):
    ``--thought-call-recovery off`` makes ``cli`` pass ``recover_thought_calls=False``. A session built with what
    ``cli`` passed runs the D-5 turn and a toolless live turn exactly as the same session without the keyword
    (BASE's default live session): machine steps, HUD, prompts, request bodies, session-log records, turn lines,
    history and the executor's invocations are equal once the key is removed; it executes no recovered call; the
    key sits only in turn-line ``extras``, ``False`` on both lines."""
    require_flag()
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch, FLAG, "off")
    kwargs = carried(calls)
    monkeypatch.undo()
    assert KEY in kwargs and kwargs[KEY] is False, (
        f"T-0135b A6/R-m analogue: --thought-call-recovery off passes {KEY}=False; got {kwargs.get(KEY, 'absent')!r}"
    )
    base_kwargs = {k: v for k, v in kwargs.items() if k != KEY}
    arm_a = asyncio.run(_two_turns(tmp_path / "a", kwargs))
    base = asyncio.run(_two_turns(tmp_path / "base", base_kwargs))
    a_clean, where = t0135._drop_key(arm_a)
    _, where_base = t0135._drop_key(base)
    assert where_base == [], f"premise: no key without the keyword; {where_base}"
    assert where and all(p[-1] == "extras" for p in where), (
        f"T-0135b A6: the key lives only in turn-line extras; found at {where}"
    )
    assert [rec["extras"].get(KEY) for rec in arm_a["records"]] == [False, False], (
        "T-0135b A5c: every live turn line, the toolless one too, says recover_thought_calls: false"
    )
    assert arm_a["invocations"] == [("t1-call", tsl.CLICK_ARGS)], (
        f"T-0135b A6: off executes no recovered call; {arm_a['invocations']}"
    )
    assert a_clean["invocations"] == base["invocations"]
    for section in t0107a.SECTIONS:
        _assert_equal(
            base[section], a_clean[section], f"T-0135b A6 off vs BASE default {section!r}:"
        )


@pytest.mark.parametrize("late", [False, True], ids=["abort-stops", "engine-serves-on"])
def test_tripped_thought_not_recovered(tmp_path, capsys, monkeypatch, late):
    """A9 (adversarial; the mutant recovers the tail after a trip): with the keywords a default ``cli`` live run
    passes, a step 2 whose open thought is the logged S2#15 collapse (T-0142's fixture) and whose last block ends in
    a valid tagged ``type_at`` trips T-0142's guard: one trip note for ``t1-gen-s2``, ``LlmError(t1-gen-s2,
    DEGENERATE_MESSAGE)``, no ``tool.call`` of ``t1-call-s2`` and no ``from_thought`` anywhere, only step 1's click
    executed. ``engine-serves-on``: the engine keeps streaming the aborted request (the tail arrives after the
    trip). Power: the same stream with the guard off is recovered.

    T-0179 B-11: the tripped step is retried once and the same stream is scripted for the retry, so it trips again:
    one trip note per attempt, ``LlmError(t1-gen-s2, DEGENERATE_RETRIED_MESSAGE)``, and neither attempt's tail call is
    recovered or executed."""
    t0142.degenerate_message()
    message = t0142.retried_message()
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch)
    kwargs = carried(calls)
    monkeypatch.undo()
    assert kwargs.get(KEY) is True and kwargs.get(GUARD) is True, (
        f"T-0135b A4/A9: a default cli live session gets {KEY}=True and {GUARD}=True; got "
        f"{ {k: kwargs.get(k, 'absent') for k in (KEY, GUARD)} }"
    )
    assert all(c.segment == "thought" and c.thought_open for c in TRIP_STREAM), (
        "premise: one open block"
    )
    power = asyncio.run(_trip_turn(tmp_path / "power", {**kwargs, GUARD: False}, late))
    assert ("t1-call-s2", t0135.TYPE_ARGS) in power["invocations"], (
        f"premise (power): unguarded, the stream's tail call is recovered; {power['invocations']}"
    )
    assert [r.get("call_id") for r in power["records"] if FROM in r] == ["t1-call-s2"] * 2, (
        "premise (power): the recovered call's records carry from_thought, so their absence below is real"
    )
    got = asyncio.run(_trip_turn(tmp_path / "guarded", kwargs, late, retry_trips=True))
    assert got["extras"].get(KEY) is True, f"premise: recovery was on; {got['extras']}"
    assert [n["request_id"] for n in got["notes"]] == ["t1-gen-s2", "t1-gen-s2"], got["notes"]
    assert got["errors"] == [message], (
        f"T-0135b A9 / T-0179 B-11: the turn ends with DEGENERATE_RETRIED_MESSAGE; {got['errors']}"
    )
    assert got["invocations"] == [("t1-call", tsl.CLICK_ARGS)], (
        f"T-0135b A9: a tripped request is never recovered; executed {got['invocations']}"
    )
    assert [r for r in got["records"] if r.get("call_id") == "t1-call-s2"] == [], got["records"]
    assert [r for r in got["records"] if FROM in r] == [], got["records"]
    assert got["extras"].get("tool_loop_end") == "degenerate", got["extras"].get("tool_loop_end")


# --------------------------------------------------------------------------------------------- A3: refusals


def _refused(tmp_path: Path, capsys: Any, monkeypatch: Any, flags: list[str], body: str) -> None:
    for extra in (["--dry-run"], []):
        where = tmp_path / ("dry" if extra else "start")
        where.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, where)
        events = Events()
        fake_live(monkeypatch, events)
        code = run_cli([*argv(where, *flags, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and _names_live(out + err), (
            f"T-0135b A3 {flags} {body!r} {extra}: exit 2 naming live; got {code}\n{out}{err}"
        )
        assert "unrecognized arguments" not in err, err
        assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
        assert not [e for e in events if e[0].startswith("pages.")], list(events)
        monkeypatch.undo()


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, value):
    """A3 (mutation 3, the flag accepted silently in ``dry_run``): ``--thought-call-recovery on|off`` with
    ``--tools off`` or ``--tools dry-run`` exits 2 with a message naming ``live`` (not argparse's usage error),
    builds no graph and starts no page server, with ``--dry-run`` and without; also through a live profile whose mode
    ``--tools`` overrode."""
    require_flag()
    _refused(tmp_path / "plain", capsys, monkeypatch, [*MODE_FLAGS[mode], FLAG, value], "")
    override = ["--tools", "off" if mode == "off" else "dry-run"]
    _refused(tmp_path / "over", capsys, monkeypatch, [*override, FLAG, value], LIVE_PROFILE)


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
@pytest.mark.parametrize("value", ["true", "false"])
def test_profile_key_refused_when_tools_overridden(tmp_path, capsys, monkeypatch, override, value):
    """A3 (the T-0134c orchestrator ruling): a live profile with ``thought_call_recovery`` loads, but run with
    ``--tools off`` or ``--tools dry-run`` the key would silently do nothing: exit 2 naming ``live``, no graph, no
    page server, with ``--dry-run`` and without. Too-wide companion: the same profile without an override builds."""
    require_field()
    body = LIVE_PROFILE + f"{FIELD} = {value}\n"
    (tmp_path / "p").mkdir()
    assert load(_toml(tmp_path / "p", body)).active().tools == "live", "premise: the profile loads"
    _refused(tmp_path / "over", capsys, monkeypatch, override, body)
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs, "the same profile without an override is accepted"


@pytest.mark.parametrize("junk", ["true", "yes", "ON", "default"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """A3: an unknown value exits 2 through argparse (its ``invalid choice`` message)."""
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *LIVE, FLAG, junk), "--dry-run"])
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
        raise AssertionError("recovery_of opened a socket")
socket.socket = NoSocket
builtins.open = spy
try:
    out = []
    for p, mode, ns in (
        (Profile(name="lean", tools="live"), "live", argparse.Namespace()),
        (Profile(name="lean"), "off", argparse.Namespace()),
        (Profile(name="lean"), "dry_run", argparse.Namespace()),
        (Profile(name="lean", tools="live"), "live", argparse.Namespace(thought_call_recovery="off")),
    ):
        out.append(repr(cli.recovery_of(p, ns, mode).session_value()))
finally:
    builtins.open = real_open
after = sorted(m for m in sys.modules if m.startswith("local_astra.computer"))
print(json.dumps({"before": before, "after": after, "opened": opened, "values": out}))
"""


def test_recovery_of_pure():
    """A2: ``recovery_of`` (and ``session_value()``) opens no socket and no file and imports no
    ``local_astra.computer`` module (``sys.modules`` before and after, in a fresh interpreter); an empty
    ``Namespace()`` resolves the default."""
    require_resolver()
    proc = subprocess.run(
        [sys.executable, "-c", _PURE], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, f"T-0135b A2: recovery_of is not pure:\n{proc.stderr[-2000:]}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["before"] == [], f"premise: importing cli loads no computer module; {got['before']}"
    assert got["after"] == [], f"T-0135b A2: recovery_of imported {got['after']}"
    assert got["opened"] == [], f"T-0135b A2: recovery_of opened {got['opened']}"
    assert got["values"] == ["True", "None", "None", "False"], got["values"]


# --------------------------------------------------------------------------------------------- A8: docs and help


def test_help_states_default():
    """A8: the ``--thought-call-recovery`` help names both values, the ``on`` default and ``live``. Keywords, never
    wording."""
    require_flag()
    h = t0133._help_of(FLAG)
    assert re.search(r"\bon\b", h) and re.search(r"\boff\b", h), h
    assert "default" in h.lower() and "live" in h, h


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0135b():
    """A8: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0135b``, naming the constant, the flag,
    the profile key, the ``Session`` keyword, ``from_thought``, the human ruling, the ``recovery`` line, that ``off``
    passes ``False`` and the A/B's success counts (66/100, 67/100); removing it once leaves docs/02 equal to
    ``DOC_PARENT``'s paragraph for paragraph (T-0135's and T-0134c's paragraphs verbatim). Keywords, never wording.
    Both sides are fixed revisions (``DOC_REV``, ``DOC_PARENT``), never the working tree."""
    gr.require_commits(DOC_PARENT, DOC_REV)
    base_doc = _doc02_at(DOC_PARENT)
    doc = _doc02_at(DOC_REV)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0135b A8: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    flat = para.replace("`", "")
    low = flat.lower()
    need = {
        "LIVE_DEFAULT_THOUGHT_CALL_RECOVERY": "LIVE_DEFAULT_THOUGHT_CALL_RECOVERY" in flat,
        "--thought-call-recovery": "--thought-call-recovery" in flat,
        "thought_call_recovery": re.search(r"(?<![-\w])thought_call_recovery\b", flat) is not None,
        "recover_thought_calls": "recover_thought_calls" in flat,
        "from_thought": "from_thought" in flat,
        "ruling": "ruling" in low or "ruled" in low,
        "recovery line": "recovery" in low and "line" in low,
        "False": "False" in flat,
        "live": re.search(r"\blive\b", low) is not None,
        "66/100": re.search(r"66\s*/\s*100", flat) is not None,
        "67/100": re.search(r"67\s*/\s*100", flat) is not None,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0135b A8: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0135b A8: docs/02 changes only by the one delta paragraph (T-0135's and T-0134c's verbatim)"
    )


# --------------------------------------------------------------------------------------------- A9: other modes vs BASE

#: A9 "for every input": ``grounding_of`` and ``feedback_of`` over a grid of profiles, argument namespaces (the new
#: dest included, which both must ignore) and modes. Executed in-process here and in the BASE tree's interpreter.
_GRID = """
def resolver_grid():
    import argparse, dataclasses
    from local_astra import cli
    from local_astra.config import Profile
    profiles = {
        "plain": dict(name="lean"),
        "dry": dict(name="lean", tools="dry_run"),
        "dry-vt560": dict(name="lean", tools="dry_run", visual_tokens=560),
        "live": dict(name="lean", tools="live"),
        "live-vt280": dict(name="lean", tools="live", visual_tokens=280),
        "live-reground": dict(name="lean", tools="live", reground=True),
        "live-crop560": dict(name="lean", tools="live", reground=True, reground_visual_tokens=560),
        "live-fb": dict(name="lean", tools="live", live_feedback="label+extent", action_echo="pixels"),
        "live-fb-off": dict(name="lean", tools="live", live_feedback="off"),
    }
    spaces = {
        "empty": {},
        "nones": dict(visual_tokens=None, reground=None, reground_visual_tokens=None, live_feedback=None,
                      action_echo=None),
        "vt560": dict(visual_tokens=560),
        "reground": dict(reground=True),
        "reground-off": dict(reground=False),
        "crop280": dict(reground=True, reground_visual_tokens=280),
        "fb": dict(live_feedback="off", action_echo="model"),
        "recovery-off": dict(thought_call_recovery="off"),
        "recovery-on-fb": dict(thought_call_recovery="on", live_feedback="label", action_echo="pixels"),
    }
    out = {}
    for pn, pk in profiles.items():
        p = Profile(**pk)
        for sn, sk in spaces.items():
            for mode in ("off", "dry_run", "live"):
                ns = argparse.Namespace(**sk)
                if mode != "live" and getattr(ns, "visual_tokens", None) is None and p.visual_tokens is None:
                    # T-0183 re-scope (the D4b pattern on both sides): outside live with no budget set, the
                    # namespace passes --visual-tokens 1120 (T-0183 makes 1120 the default there)
                    ns.visual_tokens = 1120
                g = cli.grounding_of(p, ns, mode)
                f = cli.feedback_of(p, ns, mode)
                out[pn + "|" + sn + "|" + mode] = {
                    "grounding": dataclasses.asdict(g),
                    "grounding_line": g.line(),
                    "turn_extra": g.turn_extra(),
                    "feedback": dataclasses.asdict(f),
                    "feedback_line": f.line(),
                    "feedback_session": repr(f.session_value()),
                }
    return out
"""

_RUNNER = (
    _GRID
    + """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0134c_live_feedback_default as m
out = pathlib.Path(sys.argv[1])
sections = {"other": m._other_modes(out.parent / "run" / "other"), "grid": resolver_grid()}
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""
)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture, in BASE's tree (T-0131's ``git archive`` harness through ``tsl._base_tree``)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0135b_base")
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


# T-0125 pin move: ToolKit.web_search (None with search off)
def test_other_modes_unchanged(base: dict[str, Any], tmp_path: Path) -> None:
    """A9 (green at BASE; mutations 4 and 5): for ``off`` and ``dry_run`` (plain, loop off, and a live profile under
    ``--tools off``/``--tools dry-run``) the printed ``--dry-run`` output, a build's output, the client and session
    keywords (no ``recover_thought_calls``), the executors made, the system message and one request body equal
    BASE's; and ``cli.grounding_of``/``cli.feedback_of`` and their lines equal BASE's over a grid of inputs and every
    mode."""
    # T-0155 O-8: the persona sentence only, on the observed side
    now_other = observed(t0134c._other_modes(tmp_path / "other"))
    for case, *_ in t0134c.OTHER:
        for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
            got, want = now_other[case][key], base["other"][case][key]
            # T-0172 R2 pin move (B9): dry_run gains the reask line and keyword, off nothing
            got = t0172.admit_reask_other(case, key, got)
            if key == "session_kwargs":
                got = t0125.kit_without_web_search(got)  # T-0125 pin move
                got = t0134c.without_t0140_kit_field(got, want)  # T-0140 re-pin (O-A2)
                got = t0181.without_mic_unavailable(got)  # T-0181 A-R13 pin move
            assert got == want, (
                f"T-0135b A9: {case} {key} differs from {BASE_SHA[:7]}:\n"
                f"base {json.dumps(want)[:1500]}\nnow  {json.dumps(got)[:1500]}"
            )
        assert all(KEY not in kw for kw in now_other[case]["session_kwargs"]), case
        assert "recovery:" not in now_other[case]["dry"]["out"], case
    ns: dict[str, Any] = {}
    exec(_GRID, ns)
    grid = ns["resolver_grid"]()
    assert set(grid) == set(base["grid"]) and len(grid) == 9 * 9 * 3, "premise: the same grid"
    for k in grid:
        assert grid[k] == base["grid"][k], f"T-0135b A9: resolvers differ from BASE at {k}"


def test_session_default_unchanged():
    """A9 (green at BASE): ``Session``'s ``recover_thought_calls`` keyword still defaults to ``None`` (the default
    lives in ``cli``, T-0132 R-f)."""
    import inspect

    from local_astra.conductor import session as session_mod

    params = inspect.signature(session_mod.Session.__init__).parameters
    assert KEY in params and params[KEY].default is None


def test_eval_exemption_adversarial():
    """A7 on CPU (green once the eval is corrected; mutation 6): ``tests/eval/test_t0135_ab.arm_setting_findings``
    exempts only the turn-line assertion and only for a trial that wrote no turn line; the manifest-level assertion
    still runs on a timed-out trial; a trial with a turn line but no ``_logged`` key fails whatever its ``success``;
    a trial that scanned nothing is not judged."""
    from tests.eval import test_t0135_ab as ev

    def trial(arm: str, **kw: Any) -> dict[str, Any]:
        row = {
            "arm": arm,
            "task": "S2",
            "repeat": 5,
            "scanned_requests": ["r"],
            "recover_thought_calls": arm == "B",
            "recover_thought_calls_logged": arm == "B",
            "timed_out": False,
            "session_id": "s-1",
            "success": True,
        }
        row.update(kw)
        return row

    f = ev.arm_setting_findings
    assert f([trial("A"), trial("B")]) == ([], [])
    no_line = {
        "recover_thought_calls_logged": None,
        "timed_out": True,
        "session_id": None,
        "success": False,
    }
    assert f([trial("B", **no_line)]) == ([], ["B S2#5"])
    assert f([trial("B", **{**no_line, "timed_out": False})]) == ([], ["B S2#5"]), (
        "session_id None alone"
    )
    assert f([trial("B", **{**no_line, "session_id": "s-9"})]) == ([], ["B S2#5"]), (
        "timed_out alone"
    )
    got, ex = f([trial("B", **{**no_line, "recover_thought_calls": False})])
    assert got == ["B S2#5: recover_thought_calls False"] and ex == ["B S2#5"], got
    for success in (True, False):
        got, ex = f([trial("A", recover_thought_calls_logged=None, success=success)])
        assert got == ["A S2#5: recover_thought_calls_logged None"] and ex == [], (success, got)
    got, _ = f([trial("B", recover_thought_calls_logged=False)])
    assert got == ["B S2#5: recover_thought_calls_logged False"], got
    assert f([trial("B", scanned_requests=[], recover_thought_calls=None)]) == ([], [])


# --------------------------------------------------------------------------------------------- A9: frozen

#: A9's list, written out from the criterion, plus the task's other not-owned paths and modules
FROZEN = (
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/contracts",
    "tests/fakes",
    "hud",
    "services",
    "scripts",
    "local_astra/perception",
    "local_astra/pages",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "tests/unit/_live_browser_support.py",
    "tests/unit/test_t0135_ab_harness.py",
    "tests/unit/test_t0135_differential.py",
    "tests/unit/test_t0135_live_recovery.py",
    "tests/unit/test_t0135_replay.py",
    "tests/unit/test_t0135_thought_call.py",
    "tests/unit/test_t0142_degeneracy.py",
    "tests/unit/test_t0142_differential.py",
    "tests/unit/test_t0142_session.py",
    "tests/eval/test_t0135_replay_t0132.py",
)


def test_frozen_paths_over_closed_range():
    """A9 (green at BASE): A9's paths (``conductor/**``, ``computer/**``, ``toolcall/**``, ``contracts/**``, the
    fakes, the HUD, ``services/``, ``scripts/``) and the task's other not-owned paths are byte-identical over
    ``BASE_SHA..TIP_IMPL`` (two literals, never the working tree), with no file added or removed;
    ``LIVE_TOOL_GUIDANCE`` keeps 768a973b and ``tools_v1.json`` bbfc53cc. Power: the same predicate sees T-0111g's
    ``versions.lock`` change."""
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
        f"T-0135b A9: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    guidance = hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest()
    assert guidance.startswith("768a973b"), guidance
    blob = gr._git("show", f"{TIP_IMPL}:local_astra/toolcall/tools_v1.json")
    assert blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest().startswith(
        "bbfc53cc"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN), "power"
