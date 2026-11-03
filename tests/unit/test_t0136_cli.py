"""T-0136 group C (cli, config, docs): ``--action-outcome``, its source, the refusals, the wiring, the startup line,
off is BASE, and the docs/02 delta (docs/tasks/T-0136.md, C-1...C-7).

**The surface these rows pin** (the task's names; readings stated):

- C-1: ``config.LIVE_DEFAULT_ACTION_OUTCOME = False``; ``Profile.action_outcome: bool | None = None``; TOML booleans
  only (a non-bool value is a load refusal naming ``action_outcome``); refused at load: the key with ``tools !=
  "live"`` (true or false; naming ``action_outcome`` and ``live``, reading **Rc-1**), and ``action_outcome = true``
  with ``live_feedback = "off"`` (naming ``action_outcome`` and ``live_feedback``, reading **Rc-1**).
- C-2: ``cli.action_outcome_of(profile, args, mode) -> ActionOutcomeChoice(action_outcome, action_outcome_from)``,
  read by those two field names (``choice``); in live: the flag, else the key, else ``LIVE_DEFAULT_ACTION_OUTCOME``
  read at call time (reading **Rc-2**: the row sets the constant in both ``config`` and ``cli``, so either module's
  namespace may be the one read); elsewhere ``(False, "unset")``; one resolver used by ``session_cmd`` and
  ``_build_graph``.
- C-3: exit 2 before anything is built: either flag outside live (naming ``live``); the key of a live profile whose
  mode ``--tools`` overrode (naming ``live``); on together with resolved live feedback ``off`` from any source pair
  (defaults included, by setting the constants), naming both (read: once ``_`` and ``-`` are spaces, the text holds
  "action outcome" and "live feedback").
- C-4: ``Session(action_outcome=True)`` only when on; off passes no keyword (reviewer mutation 9).
- C-5: the live-only line ``astra session: action outcome: off (default)`` / ``on (flag)`` / ``on (profile)`` /
  ``off (flag)`` / ``off (profile)``, right after the ``overview frame`` line (also when that line reads ``off (with
  ...)``) and, at a real start, before the regime line; ``--dry-run`` too.
- C-6: with the keyword off, everything else is BASE_C's (the ``cli`` side: every startup line but the new one, the
  ``--dry-run`` output, stderr, exit codes, every ``Session`` and executor keyword; the rig side, as T-0152 Rd-1: request
  bodies, turn lines, session-log records, HUD messages, prompts, history, browser invocations of a ``cli``-built
  default session on a long and on a fitting page; the other modes; the six other resolvers over a grid), and
  ``scripts/**``, ``conductor/**`` and ``computer/**`` are byte-identical over ``BASE_C..TIP_IMPL``.
- C-7: one docs/02 §5 paragraph ``**Spec delta (T-0136).**`` read at ``TIP_IMPL`` (reading **Rc-4**).

**Readings** (flagged in the report, not resolved):

- **Rc-1** the load refusals' wording is pinned only to the names (``action_outcome``; ``live`` for the mode refusal,
  ``live_feedback`` for the feedback refusal), as T-0151 C2.
- **Rc-2** "read at call time" names no module; T-0152's constant is read from ``cli``'s namespace. Both are set.
- **Rc-3 (sequencing; ruled O-C1)** the rows were written at d4adf4f, before group B. Group B's branch tip 6e63f98
  was merged into this branch at 4f59607 and BASE_C re-based to it, so ``conductor/**`` (B's ``session.py``) lies
  before the closed range. Rows that build a session with the outcome **on** still assert the premise
  ``"action_outcome" in Session``'s signature (``require_session_keyword``), now satisfied.
- **Rc-4 (C-7)** the calibrated values and test-split numbers do not exist yet (A2/A2b open), so the docs row pins
  the constant *names* (A2-3 plus A2b-4's ``UNDERLINE_ROWS``), the 12-string vocabulary's four stems, the clause, the
  six reasons and why overview-frame steps are skipped, the off-screen limitation, the rule-3' blind spots (A2b
  implementation notes: "The C-7 docs paragraph names them"; read as the words ``strikethrough``, ``luma`` and
  ``shadow``), the B-5 names, the corpus dir stem and a ``n/N`` fraction for the test split. Never the values.

**Green at BASE_C** (BASE-identity rows, no marker): ``test_default_resolution_is_base``,
``test_default_rigs_are_base``, ``test_other_modes_unchanged``, ``test_other_resolvers_unchanged`` and
``test_frozen_paths_over_closed_range``. Every other row is red at BASE_C on a named assertion. Activated for
``/implement T-0136`` group C (O-C1): no pending markers.

Regime: CPU over fakes, ``ScriptedClock`` for the rig turns (T-0152's ``_rig_turn``), thinking on, loop on (8 steps),
live, 1280x800 frames; nothing served; n = 1, deterministic. BASE_C = 6e63f98.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests import _git_range as gr
from tests.unit import _t0125_pins as t0125
from tests.unit import _t0160_pins as t0160
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0181_pins as t0181
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0134_differential as t0134
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0152_overview_frame_default as t0152
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import _names_live
from tests.unit.test_t0140_cli import session_help
from tests.unit.test_t0150_cli import refused_run
from tests.unit.test_t0151_cli import LIVE, LIVE_PROFILE
from tests.unit.test_t0151_cli import PREFIX as FRAME_PREFIX

#: BASE_C: 6e63f98, the T-0136 B branch tip (A2 detector + B session wiring), merged into this branch at 4f59607;
#: re-based from d4adf4f per Rc-3 (O-C1). A literal, never a moving ref.
BASE_C = "6e63f98"
#: The implementation tip; equals BASE_C until the implementation is committed. Bumped by the test-author or the
#: orchestrator to the last implementation commit (the implementer never edits tests).
TIP_IMPL = "aca89a8"
HEADER = "**Spec delta (T-0136).**"

PREFIX = "astra session: action outcome: "
ON, OFF = "--action-outcome", "--no-action-outcome"
REGIME = "astra session: regime:"

#: C-5's literals, typed from the criterion
OFF_DEFAULT = PREFIX + "off (default)"
ON_FLAG = PREFIX + "on (flag)"
ON_PROFILE = PREFIX + "on (profile)"
OFF_FLAG = PREFIX + "off (flag)"
OFF_PROFILE = PREFIX + "off (profile)"


# --------------------------------------------------------------------------------------------- seams


def require() -> Any:
    flags = set(re.findall(r"--[a-z][a-z-]+", session_help()))
    missing = [f for f in (ON, OFF) if f not in flags]
    assert not missing, f"T-0136 C-2: astra session {missing} do not exist yet"
    assert "action_outcome" in Profile.model_fields, (
        "T-0136 C-1: Profile.action_outcome does not exist yet"
    )
    fn = getattr(cli, "action_outcome_of", None)
    assert callable(fn), "T-0136 C-2: cli.action_outcome_of(profile, args, mode) does not exist yet"
    return fn


def require_session_keyword() -> None:
    """Reading Rc-3: rows that build a session with the outcome on need group B's ``Session(action_outcome=)``."""
    params = inspect.signature(cli.Session.__init__).parameters
    assert "action_outcome" in params, (
        "premise (T-0136 Rc-3): Session(action_outcome=) is group B's; merge group B before group C's on-rows"
    )


def choice(v: Any) -> tuple[Any, Any]:
    """The resolver's value and source, by C-2's field names."""
    return v.action_outcome, v.action_outcome_from


def lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def _spaced(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def names_both(text: str) -> bool:
    low = _spaced(text)
    return "action outcome" in low and "live feedback" in low


def _load(tmp_path: Path, name: str, body: str) -> Any:
    d = tmp_path / name
    d.mkdir()
    return load(_toml(d, body))


def _set_default(monkeypatch: Any, name: str, value: Any) -> None:
    """Reading Rc-2: a live default set in both modules' namespaces."""
    monkeypatch.setattr(config, name, value, raising=False)
    monkeypatch.setattr(cli, name, value, raising=False)


# =====================================================================================================
# C-1: config
# =====================================================================================================


def test_default_off(tmp_path, capsys, monkeypatch):
    """C-1/C-2/C-4/C-5 (reviewer mutation 10, the default ``True``): ``config.LIVE_DEFAULT_ACTION_OUTCOME is False``;
    the key defaults to ``None`` (annotation ``bool | None``) in every built-in profile; the resolver says ``(False,
    "default")``; the default build passes no ``action_outcome`` keyword; the default live line reads ``off
    (default)``."""
    fn = require()
    assert getattr(config, "LIVE_DEFAULT_ACTION_OUTCOME", "absent") is False, (
        "T-0136 C-1: config.LIVE_DEFAULT_ACTION_OUTCOME = False"
    )
    f = Profile.model_fields["action_outcome"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.action_outcome is None, name  # type: ignore[attr-defined]
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert choice(fn(profile, argparse.Namespace(), "live")) == (False, "default")
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    assert "action_outcome" not in calls.session_kwargs[-1], (
        f"T-0136 C-4: the default build passes no keyword; {sorted(calls.session_kwargs[-1])}"
    )
    monkeypatch.undo()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(out) == [OFF_DEFAULT], out


def test_profile_key_load_refusals(tmp_path):
    """C-1: the key with ``tools`` other than ``live`` (off, dry_run, absent), true or false, is refused at load naming
    ``action_outcome`` and ``live``; a non-bool value is refused naming ``action_outcome``; ``action_outcome = true``
    with ``live_feedback = "off"`` is refused naming both keys. Too-wide companions: the key alone (true, false) loads
    with its value; ``true`` beside ``live_feedback = "label"`` / ``"label+extent"`` loads; ``false`` beside
    ``live_feedback = "off"`` loads (only ``true`` with ``off`` is refused)."""
    require()
    for i, tools_line in enumerate(('tools = "off"\n', 'tools = "dry_run"\n', "")):
        for v in ("true", "false"):
            with pytest.raises(ValueError) as exc:
                _load(tmp_path, f"t{i}-{v}", f"{tools_line}action_outcome = {v}\n")
            msg = str(exc.value)
            assert _names_live(msg) and "action_outcome" in msg, (tools_line, v, msg)
    for j, junk in enumerate(('"on"', '"true"', "1", "0", '"yes"')):
        with pytest.raises(ValueError) as exc:
            _load(tmp_path, f"junk-{j}", f'tools = "live"\naction_outcome = {junk}\n')
        assert "action_outcome" in str(exc.value), (junk, str(exc.value))
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, "fb-off", 'tools = "live"\naction_outcome = true\nlive_feedback = "off"\n')
    msg = str(exc.value)
    assert "action_outcome" in msg and "live_feedback" in msg, msg
    for k, (body, want) in enumerate(
        (
            ('tools = "live"\naction_outcome = true\n', True),
            ('tools = "live"\naction_outcome = false\n', False),
            ('tools = "live"\naction_outcome = true\nlive_feedback = "label"\n', True),
            ('tools = "live"\naction_outcome = true\nlive_feedback = "label+extent"\n', True),
            ('tools = "live"\naction_outcome = false\nlive_feedback = "off"\n', False),
        )
    ):
        try:
            got = _load(tmp_path, f"ok-{k}", body).active().action_outcome
        except ValueError as e:
            raise AssertionError(f"T-0136 C-1: {body!r} loads; refused with {e}") from None
        assert got is want, body


# =====================================================================================================
# C-2/C-4/C-5: the resolver, the wiring, the line
# =====================================================================================================

#: (flags, profile body after ``LIVE_PROFILE``, resolved pair, line)
SOURCES = {
    "flag-on": ([ON], "", (True, "flag"), ON_FLAG),
    "flag-off": ([OFF], "", (False, "flag"), OFF_FLAG),
    "key-true": ([], "action_outcome = true\n", (True, "profile"), ON_PROFILE),
    "key-false": ([], "action_outcome = false\n", (False, "profile"), OFF_PROFILE),
    "flag-off-over-key": ([OFF], "action_outcome = true\n", (False, "flag"), OFF_FLAG),
    "flag-on-over-key": ([ON], "action_outcome = false\n", (True, "flag"), ON_FLAG),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources_and_line(tmp_path, capsys, monkeypatch, case):
    """C-2/C-5: the flag, else the key, each with its source (the resolver over the loaded profile and the argparse
    dest, and the ``--dry-run`` line, one line exactly)."""
    fn = require()
    flags, body, want, line = SOURCES[case]
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    flag = True if ON in flags else False if OFF in flags else None
    assert choice(fn(profile, argparse.Namespace(action_outcome=flag), "live")) == want
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    assert lines(out) == [line], out


@pytest.mark.parametrize("case", [c for c in SOURCES if not SOURCES[c][2][0]])
def test_off_passes_no_keyword(tmp_path, capsys, monkeypatch, case):
    """C-4 (adversarial; reviewer mutation 9, ``action_outcome=False`` passed when off): off by flag or key builds a
    ``Session`` with no ``action_outcome`` keyword, and prints its line at a real start."""
    require()
    flags, body, _, line = SOURCES[case]
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    assert "action_outcome" not in kw, f"T-0136 C-4: off passes no keyword; {sorted(kw)}"
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "s", capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body
    )
    assert code == 0 and lines(started) == [line], started


@pytest.mark.parametrize("case", [c for c in SOURCES if SOURCES[c][2][0]])
def test_on_passes_keyword(tmp_path, capsys, monkeypatch, case):
    """C-4: on by flag or key builds ``Session(action_outcome=True)`` (the keyword ``True`` itself), and the real start
    prints ``on (flag)`` / ``on (profile)``. Needs group B's keyword (Rc-3)."""
    require()
    require_session_keyword()
    flags, body, _, line = SOURCES[case]
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    assert kw.get("action_outcome") is True, (
        f"T-0136 C-4: Session(action_outcome=True); {sorted(kw)}"
    )
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "s", capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body
    )
    assert code == 0 and lines(started) == [line], started


@pytest.mark.parametrize("flag", [True, False, None])
@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_resolver_unset_outside_live(mode, flag) -> None:
    """C-2: outside live the resolver is ``(False, "unset")``, whatever the dest says and with a live profile carrying
    the key (the refusal of such a flag is ``session_cmd``'s, not the resolver's)."""
    fn = require()
    for profile in (
        Profile(name="lean"),
        Profile(name="lean", tools="live", action_outcome=True),  # type: ignore[arg-type, call-arg]
    ):
        got = choice(fn(profile, argparse.Namespace(action_outcome=flag), mode))
        assert got == (False, "unset"), (mode, flag, profile.tools, got)


def test_default_read_at_call_time(tmp_path, capsys, monkeypatch):
    """C-2 (reading Rc-2): with ``LIVE_DEFAULT_ACTION_OUTCOME`` set ``True`` after import, ``Namespace()`` in live
    resolves ``(True, "default")`` and the ``--dry-run`` line reads ``on (default)``; a flag or key still wins."""
    fn = require()
    _set_default(monkeypatch, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert choice(fn(profile, argparse.Namespace(), "live")) == (True, "default")
    assert choice(fn(profile, argparse.Namespace(action_outcome=False), "live")) == (False, "flag")
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(out) == [PREFIX + "on (default)"], out


def test_one_resolver(tmp_path, capsys, monkeypatch):
    """C-2 "one resolver": ``session_cmd`` and ``_build_graph`` both read the choice through ``cli.action_outcome_of``
    (a spy wrapping the real one sees at least two calls on a default build, every one in live and ``(False,
    "default")``); a spy forcing off from ``"flag"`` on a ``--action-outcome`` build leaves ``Session`` without the
    keyword and the line ``off (flag)`` (nothing re-derives the choice beside the resolver)."""
    real = require()
    seen: list[Any] = []

    def spy(p: Any, a: Any, m: str, **kw: Any) -> Any:
        got = real(p, a, m, **kw)
        seen.append((m, choice(got)))
        return got

    monkeypatch.setattr(cli, "action_outcome_of", spy)
    build(tmp_path / "b", capsys, monkeypatch)
    assert len(seen) >= 2 and all(x == ("live", (False, "default")) for x in seen), seen
    monkeypatch.undo()
    forced = cli.ActionOutcomeChoice(False, "flag")
    assert choice(forced) == (False, "flag"), (
        "premise: ActionOutcomeChoice(action_outcome, action_outcome_from)"
    )
    monkeypatch.setattr(
        cli, "action_outcome_of", lambda p, a, m, **kw: forced if m == "live" else real(p, a, m)
    )
    calls, _ = build(tmp_path / "f", capsys, monkeypatch, ON)
    assert "action_outcome" not in calls.session_kwargs[-1], sorted(calls.session_kwargs[-1])
    monkeypatch.undo()
    monkeypatch.setattr(
        cli, "action_outcome_of", lambda p, a, m, **kw: forced if m == "live" else real(p, a, m)
    )
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", ON)
    assert lines(out) == [OFF_FLAG], out


#: (flags, profile body after ``LIVE_PROFILE``) whose frame line differs; the new line follows it in each
PLACEMENT = {
    "default": ([], ""),
    "on-flag": ([ON], ""),
    "reground": (["--reground"], ""),
    "page": (["--page-overview"], ""),
    "no-frame": (["--no-overview-frame"], ""),
    "key-true+extent": ([], 'action_outcome = true\nlive_feedback = "label+extent"\n'),
}


@pytest.mark.parametrize("case", list(PLACEMENT))
def test_line_placement_dry_run(tmp_path, capsys, monkeypatch, case):
    """C-5: under ``--dry-run``, exactly one line, immediately after the ``overview frame`` line (whatever that line
    reads, ``off (with --reground)`` included)."""
    require()
    flags, body = PLACEMENT[case]
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    ls = out.splitlines()
    assert len(lines(out)) == 1, out
    (f,) = [i for i, x in enumerate(ls) if x.startswith(FRAME_PREFIX)]
    assert f + 1 < len(ls) and ls[f + 1].startswith(PREFIX), (
        f"T-0136 C-5: right after the frame line:\n{out}"
    )


@pytest.mark.parametrize("case", ["default", "reground", "page", "no-frame"])
def test_line_placement_real_start(tmp_path, capsys, monkeypatch, case):
    """C-5: at a real start (outcome off: no group-B premise), exactly one line, immediately after the ``overview
    frame`` line and before the regime line."""
    require()
    flags, body = PLACEMENT[case]
    _, started, code = build_out(
        tmp_path / "s", capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body
    )
    assert code == 0, started
    sl = started.splitlines()
    assert lines(started) == [OFF_DEFAULT], started
    (f,) = [i for i, x in enumerate(sl) if x.startswith(FRAME_PREFIX)]
    (r,) = [i for i, x in enumerate(sl) if x.startswith(REGIME)]
    assert sl[f + 1] == OFF_DEFAULT and f + 1 < r, started


def test_line_live_only(tmp_path, capsys, monkeypatch):
    """C-5 (needs the flags to exist, so the absence is not vacuous): no ``action outcome`` line in ``off`` or
    ``dry_run``, under ``--dry-run`` and at a real start; the live dry run has one."""
    require()
    assert lines(dry_run(tmp_path / "live", capsys, monkeypatch, "live")) == [OFF_DEFAULT]
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        assert "action outcome:" not in dry_run(tmp_path / mode, capsys, monkeypatch, mode)
        monkeypatch.undo()
        _, started, code = build_out(
            tmp_path / (mode + "-s"), capsys, monkeypatch, *MODE_FLAGS[mode]
        )
        assert code == 0 and "action outcome:" not in started, started


# =====================================================================================================
# C-3: startup refusals
# =====================================================================================================


def _refused(code: int, text: str, calls: Any) -> None:
    assert "unrecognized" not in text, f"argparse refused, not the task's refusal: {text}"
    assert code == 2, (code, text)
    assert not calls.graphs and not calls.session_kwargs, "T-0136 C-3: built before refusing"


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("flag", [ON, OFF])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, flag):
    """C-3: either flag in a resolved mode other than live exits 2 naming ``live`` and builds nothing (also through a
    live profile whose mode ``--tools`` overrode), with ``--dry-run`` and at a real start."""
    require()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        for dry in (True, False):
            code, text, calls = refused_run(
                tmp_path, capsys, monkeypatch, *flags, flag, body=body, dry=dry
            )
            _refused(code, text, calls)
            assert _names_live(text), text
            monkeypatch.undo()


@pytest.mark.parametrize("value", ["true", "false"])
@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
def test_key_refused_when_tools_overridden(tmp_path, capsys, monkeypatch, override, value):
    """C-3: a live profile with an ``action_outcome`` key (true or false) under a non-live ``--tools`` override exits 2
    naming ``live`` and builds nothing, with ``--dry-run`` and without; too-wide companion: the same profile with no
    override builds (``false``; ``true`` needs group B's keyword and is ``test_on_passes_keyword``'s)."""
    require()
    body = LIVE_PROFILE + f"action_outcome = {value}\n"
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *override, body=body, dry=dry
        )
        _refused(code, text, calls)
        assert _names_live(text), (override, dry, text)
        monkeypatch.undo()
    if value == "false":
        calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
        assert calls.graphs


#: on together with resolved live feedback ``off``: (flags, profile body, the defaults set) per source pair
#: (outcome source / feedback source); key/key is a load refusal (C-1)
WITH_OFF: dict[str, tuple[list[str], str, dict[str, Any]]] = {
    "flag/flag": ([ON, "--live-feedback", "off"], "", {}),
    "flag/key": ([ON], 'live_feedback = "off"\n', {}),
    "key/flag": (["--live-feedback", "off"], "action_outcome = true\n", {}),
    "default/flag": (["--live-feedback", "off"], "", {"LIVE_DEFAULT_ACTION_OUTCOME": True}),
    "default/key": ([], 'live_feedback = "off"\n', {"LIVE_DEFAULT_ACTION_OUTCOME": True}),
    "flag/default": ([ON], "", {"LIVE_DEFAULT_FEEDBACK": "off"}),
    "key/default": ([], "action_outcome = true\n", {"LIVE_DEFAULT_FEEDBACK": "off"}),
}


@pytest.mark.parametrize("pair_", list(WITH_OFF))
def test_refused_with_feedback_off(tmp_path, capsys, monkeypatch, pair_):
    """C-3 (adversarial; the mutant checks only the flags, or only the profile): the outcome on together with resolved
    live feedback ``off`` from each source pair, defaults included (set as constants, reading Rc-2), exits 2 naming
    both before anything is constructed, with ``--dry-run`` and at a real start."""
    require()
    flags, body, defaults = WITH_OFF[pair_]
    for dry in (True, False):
        for name, value in defaults.items():
            _set_default(monkeypatch, name, value)
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body, dry=dry
        )
        _refused(code, text, calls)
        assert names_both(text), (pair_, dry, text)
        monkeypatch.undo()


@pytest.mark.parametrize(
    "flags",
    [
        [OFF, "--live-feedback", "off"],
        ["--live-feedback", "off"],
        [ON, "--live-feedback", "label"],
        [ON, "--live-feedback", "label+extent"],
    ],
    ids=["off+fb-off", "default+fb-off", "on+label", "on+extent"],
)
def test_feedback_refusal_not_too_wide(tmp_path, capsys, monkeypatch, flags):
    """C-3 (too-wide companions): the outcome off (flag or default) with feedback ``off`` builds with no keyword; the
    outcome on with feedback ``label`` or ``label+extent`` builds with ``action_outcome=True`` (group B's premise,
    Rc-3)."""
    require()
    on = ON in flags
    if on:
        require_session_keyword()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags)
    kw = calls.session_kwargs[-1]
    if on:
        assert kw.get("action_outcome") is True, sorted(kw)
    else:
        assert "action_outcome" not in kw, sorted(kw)


# =====================================================================================================
# C-6: off is BASE_C (the capture runs here and, over ``git archive BASE_C``, in a subprocess)
# =====================================================================================================

#: name -> (BASE_C's flags, BASE_C's profile body, this tree's off-variant flags, its profile body); the profile body
#: follows ``LIVE_PROFILE``
OFF_CASES: dict[str, tuple[list[str], str, list[str], str]] = {
    "default": ([], "", [OFF], ""),
    "key-false": ([], "", [], "action_outcome = false\n"),
    "reground": (["--reground"], "", [OFF, "--reground"], ""),
    "page": (["--page-overview"], "", [OFF, "--page-overview"], ""),
    "auto": (["--auto-overview"], "", [OFF, "--auto-overview"], ""),
    "no-frame": (["--no-overview-frame"], "", [OFF, "--no-overview-frame"], ""),
    "feedback-off": (["--live-feedback", "off"], "", [OFF, "--live-feedback", "off"], ""),
    "feedback-off-key": (
        [],
        'live_feedback = "off"\n',
        [],
        'live_feedback = "off"\naction_outcome = false\n',
    ),
    "extent-pixels": (
        ["--live-feedback", "label+extent", "--action-echo", "pixels"],
        "",
        [OFF, "--live-feedback", "label+extent", "--action-echo", "pixels"],
        "",
    ),
    "recovery-off": (
        ["--thought-call-recovery", "off"],
        "",
        [OFF, "--thought-call-recovery", "off"],
        "",
    ),
}
#: the off-variant's line source per case
OFF_SOURCE = {
    c: ("profile" if "action_outcome = false" in v[3] else "flag") for c, v in OFF_CASES.items()
}
RIG_CASES = ("default", "key-false")


def resolver_grid() -> dict[str, Any]:
    """C-6: the six other resolvers (``grounding_of``, ``feedback_of``, ``recovery_of``, ``page_overview_of``,
    ``auto_overview_of``, ``overview_frame_of``) and their lines over a grid of BASE-valid profiles, namespaces (this
    task's dest included, which they must ignore) and modes."""
    profiles = {
        "plain": dict(name="lean"),
        "dry": dict(name="lean", tools="dry_run"),
        "live": dict(name="lean", tools="live"),
        "live-reground": dict(name="lean", tools="live", reground=True),
        "live-page": dict(name="lean", tools="live", page_overview=True),
        "live-frame-off": dict(name="lean", tools="live", overview_frame=False, auto_overview=True),
        "live-fb-off": dict(name="lean", tools="live", live_feedback="off"),
        "live-fb": dict(
            name="lean", tools="live", live_feedback="label+extent", thought_call_recovery=False
        ),
    }
    spaces = {
        "empty": {},
        "ao-on": dict(action_outcome=True),
        "ao-off": dict(action_outcome=False),
        "ao-on-fb-off": dict(action_outcome=True, live_feedback="off"),
        "ao-off-reground": dict(action_outcome=False, reground=True),
        "ao-on-frame-off": dict(action_outcome=True, overview_frame=False, page_overview=True),
    }
    out: dict[str, Any] = {}
    for pn, pk in profiles.items():
        p = Profile(**pk)  # type: ignore[arg-type]
        for sn, sk in spaces.items():
            for mode in ("off", "dry_run", "live"):
                ns = argparse.Namespace(**sk)
                if mode != "live" and p.visual_tokens is None:
                    # T-0183 re-scope (the D4b pattern on both sides): outside live with no budget set, the
                    # namespace passes --visual-tokens 1120 (T-0183 makes 1120 the default there)
                    ns.visual_tokens = 1120
                g = cli.grounding_of(p, ns, mode)
                f = cli.feedback_of(p, ns, mode)
                r = cli.recovery_of(p, ns, mode)
                o = cli.page_overview_of(p, ns, mode)
                a = cli.auto_overview_of(p, ns, mode)
                fr = cli.overview_frame_of(p, ns, mode)
                out[f"{pn}|{sn}|{mode}"] = {
                    "grounding": dataclasses.asdict(g),
                    "grounding_line": g.line(),
                    "turn_extra": g.turn_extra(),
                    "feedback": dataclasses.asdict(f),
                    "feedback_line": f.line(),
                    "feedback_session": repr(f.session_value()),
                    "recovery": dataclasses.asdict(r),
                    "recovery_line": r.line(),
                    "recovery_session": repr(r.session_value()),
                    "page": dataclasses.asdict(o),
                    "page_line": o.line(),
                    "auto": dataclasses.asdict(a),
                    "auto_line": a.line(),
                    "frame": [fr.overview_frame, fr.overview_frame_from],
                    "frame_line": fr.line(cli.overview_frame_yielded_to(fr, p, ns, mode)),
                }
    return out


async def _rigs(tmp: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, run in runs.items():
        if run["code"] != 0 or not run["_raw_kwargs"]:
            out[key] = None
            continue
        kw, echo = t0152.rig_kwargs(run)
        out[key] = {
            page: await t0152._rig_turn(tmp / key / page, kw, echo, page) for page in t0152.PAGES
        }
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario: per case, BASE_C's input (``base``) and the off-variant (``off``), each under ``--dry-run`` and
    at a real start; rigs for ``RIG_CASES``; the other modes; the resolver grid; the executor's action strings."""
    out: dict[str, Any] = {"cli": {}}
    runs: dict[str, dict[str, Any]] = {}
    for case, (bflags, bbody, oflags, obody) in OFF_CASES.items():
        out["cli"][case] = {}
        for variant, flags, body in (("base", bflags, bbody), ("off", oflags, obody)):
            # T-0174 D4b + T-0179 A-5: the working tree's live session pins v1 and the guard off (BASE_C's); a no-op in the BASE run
            dry = t0152._cli_run(
                tmp / "cli" / case / variant / "dry",
                flags,
                body,
                dry=True,
                pin_v1=True,
                pin_guard_off=True,
                pin_reminder_off=True,  # T-0201 D3 re-pin: the reminder pinned off (BASE_C's requests)
                pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE_C's kit and guidance)
                pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE_C's turns)
            )
            start = t0152._cli_run(
                tmp / "cli" / case / variant / "start",
                flags,
                body,
                dry=False,
                pin_v1=True,
                pin_guard_off=True,
                pin_reminder_off=True,  # T-0201 D3 re-pin: the reminder pinned off (BASE_C's requests)
                pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE_C's kit and guidance)
                pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE_C's turns)
            )
            out["cli"][case][variant] = {"dry": t0152._public(dry), "start": t0152._public(start)}
            if case in RIG_CASES:
                runs[f"{case}|{variant}"] = start
    out["rigs"] = asyncio.run(_rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    out["grid"] = resolver_grid()
    out["executor"] = asyncio.run(t0134._executor_actions())
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0136_cli as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE_C's capture, in BASE_C's tree (T-0131's ``git archive`` harness through ``tsl._base_tree``; origin
    checked)."""
    gr.require_commits(BASE_C)
    tmp = tmp_path_factory.mktemp("t0136_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_C)
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
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_C} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _capture(tmp_path_factory.mktemp("t0136_now"))


def _drop_line(text: str) -> list[str]:
    return [
        x
        for x in text.splitlines()
        if not x.startswith(PREFIX)
        and not x.startswith("astra session: repeat guard:")  # T-0137 re-pin (O-B2)
    ]


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0136 {what} differs from BASE_C {BASE_C}:\nbase {json.dumps(want)[:1500]}\nnow  {json.dumps(got)[:1500]}"
    )


def _cli_same(want: dict[str, Any], got: dict[str, Any], what: str) -> None:
    got = t0181.run_without_mic_unavailable(
        got
    )  # T-0181 A-R13 pin move: only the mic_unavailable keyword
    assert want["code"] == 0 and want["raised"] is None, (
        f"premise: BASE_C's run of {what} succeeds; {want['err']}"
    )
    _same(_drop_line(want["out"]), _drop_line(got["out"]), f"{what} out")
    for key in ("code", "raised", "err", "graphs", "session_kwargs", "executor_kwargs"):
        _same(want[key], got[key], f"{what} {key}")
    for kw in got["session_kwargs"]:
        assert "action_outcome" not in kw, (what, kw)


# T-0125 pin move: the default live search line, startup line, kit
@pytest.mark.parametrize("case", list(OFF_CASES))
# T-0160 ruling 2: web on by default in live (egress=web, executor web=True)
def test_default_resolution_is_base(base, now, case):
    """C-6 (green at BASE_C; reviewer mutation 9 on the default path): BASE_C's own input on this tree (the outcome off
    from its default) gives BASE_C's ``--dry-run`` output and startup output apart from the new line, its stderr,
    exit code, graphs and every ``Session`` and executor keyword (no ``action_outcome``)."""
    for kind in ("dry", "start"):
        _cli_same(
            base["cli"][case]["base"][kind],
            t0160.admit_live_run(  # T-0160 pin move
                t0125.admit_live_run(
                    t0172.admit_reask_run(now["cli"][case]["base"][kind], kind), kind
                ),  # T-0172 R7 pin move: the default reask line and keyword
                kind,  # T-0125 pin move
            ),
            f"C-6 {case} default {kind}",
        )


# T-0125 pin move: the default live search line, startup line, kit
@pytest.mark.parametrize("case", list(OFF_CASES))
# T-0160 ruling 2: web on by default in live (egress=web, executor web=True)
def test_off_is_base(base, now, case):
    """C-6 (adversarial; reviewer mutation 9: off passing ``action_outcome=False``): ``--no-action-outcome`` or
    ``action_outcome = false`` beside each case gives BASE_C's build for BASE_C's input: every line but the new one,
    stderr, exit code, graphs, every ``Session`` and executor keyword; the new line reads ``off (flag)`` /
    ``off (profile)`` immediately after the frame line."""
    for kind in ("dry", "start"):
        got = now["cli"][case]["off"][kind]
        assert got["code"] == 0 and got["raised"] is None, (
            f"T-0136 C-6 {case} {kind}: {got['err'][-800:]}"
        )
        assert lines(got["out"]) == [PREFIX + f"off ({OFF_SOURCE[case]})"], got["out"]
        ls = got["out"].splitlines()
        (f,) = [i for i, x in enumerate(ls) if x.startswith(FRAME_PREFIX)]
        assert ls[f + 1].startswith(PREFIX), got["out"]
        _cli_same(
            base["cli"][case]["base"][kind],
            t0160.admit_live_run(
                t0125.admit_live_run(t0172.admit_reask_run(got, kind), kind), kind
            ),  # T-0125, T-0160, T-0172 R7 pin moves
            f"C-6 {case} off {kind}",
        )


def _rig_same(want: dict[str, Any] | None, got: dict[str, Any] | None, what: str) -> None:
    assert want is not None, f"premise: BASE_C built the rig for {what}"
    assert got is not None, f"T-0136 C-6: {what} did not build (exit != 0 or no Session)"
    got = observed(got)  # T-0155 O-8: the persona sentence only, on the observed side
    # T-0172 R7 pin move (B7): a cli-built live session's turn lines carry the default announced_call_reask
    got = t0172.admit_reask_turn_key(got)
    for page in t0152.PAGES:
        for key in want[page]:
            _same(want[page][key], got[page][key], f"C-6 {what} {page} {key}")


@pytest.mark.parametrize("case", RIG_CASES)
def test_default_rigs_are_base(base, now, case):
    """C-6 (green at BASE_C): a ``cli``-built session from BASE_C's input, driven on a long and a fitting page, gives
    BASE_C's sections (machine steps, HUD messages, prompts, request bodies, session-log records, turn lines, history,
    wire bodies, browser invocations)."""
    _rig_same(base["rigs"][f"{case}|base"], now["rigs"][f"{case}|base"], f"{case} default")


@pytest.mark.parametrize("case", RIG_CASES)
def test_off_rigs_are_base(base, now, case):
    """C-6 (reviewer mutation 9 over the rig): ``--no-action-outcome`` / ``action_outcome = false`` sessions give
    BASE_C's default sections on both pages (no turn-line key, no record, no clause)."""
    _rig_same(base["rigs"][f"{case}|base"], now["rigs"][f"{case}|off"], f"{case} off")


# T-0125 pin move: ToolKit.web_search (None with search off)
def test_other_modes_unchanged(base, now):
    """C-6 (green at BASE_C): for ``off`` and ``dry_run`` (plain, loop off, a live profile under ``--tools off``/
    ``--tools dry-run``) T-0134c's sections equal BASE_C's; no ``action_outcome`` keyword, no line. The executor's own
    action strings equal BASE_C's."""
    for case, *_ in t0134c.OTHER:
        for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
            # T-0155 O-8: the persona sentence only, on the observed side
            got = observed(now["other"][case][key])
            got = t0172.admit_reask_other(case, key, got)  # T-0172 R7 pin move (B9)
            if key == "session_kwargs":
                got = t0125.kit_without_web_search(got)  # T-0125 pin move
                got = t0181.without_mic_unavailable(got)  # T-0181 A-R13 pin move
            _same(base["other"][case][key], got, f"C-6 {case} {key}")
        assert all("action_outcome" not in kw for kw in now["other"][case]["session_kwargs"])
        assert "action outcome:" not in now["other"][case]["dry"]["out"], case
    _same(base["executor"], now["executor"], "C-6 executor actions")


def test_other_resolvers_unchanged(base, now):
    """C-6 (green at BASE_C): the six other resolvers and their lines equal BASE_C's over the grid, this task's dest
    included."""
    assert set(now["grid"]) == set(base["grid"]) and len(now["grid"]) == 8 * 6 * 3, "premise"
    for k in now["grid"]:
        _same(base["grid"][k], now["grid"][k], f"C-6 resolvers at {k}")


# =====================================================================================================
# C-7 and C-6: docs and frozen paths
# =====================================================================================================


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


#: C-7's keywords (reading Rc-4): names, never values or wording
DOC_KEYS = (
    # A2-3 / A2b-4: the rule's calibrated constants
    "CELL_PX",
    "DRAFT_SCALE",
    "NOISE_DELTA",
    "HOVER_DELTA",
    "HOVER_HALF_W",
    "HOVER_HALF_H",
    "MIN_CELL_PIXELS",
    "PAGE_FRACTION",
    "UNDERLINE_ROWS",
    # the vocabulary and the clause
    "no visible change",
    "changed near your click",
    "changed elsewhere (",
    "page changed",
    "Compared with the previous screenshot",
    # B-2's reasons
    "not_pointer",
    "blind",
    "no_after_frame",
    "overview_frame",
    "scrolled",
    "decode",
    # B-5's contract
    "_TurnLog.action_outcomes",
    "action_outcome_unmeasured",
    "NO_VISIBLE_CHANGE",
    # the corpus regime
    "t0136-outcome-corpus-",
    "test split",
)


def test_doc02_delta_t0136():
    """C-7 (reading Rc-4): docs/02 §5 at ``TIP_IMPL`` holds exactly one paragraph starting ``**Spec delta
    (T-0136).**`` naming the constants, the four vocabulary stems and the clause, the six unmeasured reasons and the
    overview frame's coordinate system, the off-screen limitation, the rule-3' blind spots, the B-5 contract and the
    corpus with an ``n/N`` test-split fraction; removing it leaves BASE_C's docs/02 paragraph for paragraph. Both
    sides are literals, never the working tree."""
    gr.require_commits(BASE_C, TIP_IMPL)
    assert TIP_IMPL != BASE_C, "T-0136 C-7: TIP_IMPL is not bumped to the implementation tip yet"
    doc, base_doc = _doc02_at(TIP_IMPL), _doc02_at(BASE_C)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0136 C-7: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    flat = para.replace("`", "")
    low = flat.lower()
    missing = [k for k in DOC_KEYS if k not in flat]
    assert not missing, f"T-0136 C-7: the docs/02 delta does not record {missing}"
    for words in (
        ("off-screen", "off screen"),
        ("coordinate",),
        ("strikethrough",),
        ("luma",),
        ("shadow",),
    ):
        assert any(w in low for w in words), f"T-0136 C-7: the delta does not name {words[0]!r}"
    assert re.search(r"\b\d+\s*/\s*\d+\b", flat), (
        "T-0136 C-7: the delta states no test-split count n/N"
    )
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), "T-0136 C-7: docs/02 changes only by the one paragraph"
    assert doc.count(HEADER) == 1


#: C-6's frozen list, written out from the criterion
FROZEN = ("scripts", "local_astra/conductor", "local_astra/computer")
#: the power check's literal range: T-0136 A1's builder commit and its parent
POWER_PARENT, POWER = "800d39b", "bb80cde"


def test_frozen_paths_over_closed_range():
    """C-6 (green at BASE_C; Rc-3 on re-basing): ``scripts/**``, ``conductor/**`` and ``computer/**`` are
    byte-identical over ``BASE_C..TIP_IMPL`` (two literals), and each exists at BASE_C. Power: the same predicate sees
    the T-0136 A1 builder land in ``scripts/`` (800d39b..bb80cde)."""
    gr.require_commits(BASE_C, TIP_IMPL, POWER_PARENT, POWER)
    assert gr.is_ancestor(BASE_C, TIP_IMPL), f"{BASE_C} is not an ancestor of {TIP_IMPL}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_C, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE_C"
    changed = gr.changed(BASE_C, TIP_IMPL, *FROZEN)
    assert changed == [], f"T-0136 C-6: frozen paths differ in {BASE_C}..{TIP_IMPL}: {changed}"
    assert "scripts/build_outcome_corpus.py" in gr.changed(POWER_PARENT, POWER, *FROZEN), "power"
