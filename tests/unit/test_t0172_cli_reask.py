"""T-0172 group B, B9: the announced-call re-ask is on by default in both tools-on modes, resolved in ``cli`` with a
source, off with ``--announced-call-reask off`` (docs/tasks/T-0172.md).

**The interfaces these rows pin** (the task's; readings in the test-author's report):

- ``config.DEFAULT_ANNOUNCED_CALL_REASK: bool = True``; ``Profile.announced_call_reask: bool | None = None``. Load
  refuses a non-bool (naming ``true`` and ``false``; strings and integers a lax bool would coerce included) and a
  non-``None`` value with ``tools = "off"`` (or no ``tools`` key, whose default is off); ``dry_run`` and ``live``
  load both bools.
- ``cli.reask_of(profile, args, mode) -> cli.Reask``, a dataclass whose fields are, in order, the value and the
  source (``dataclasses.astuple`` is compared, never field names): flag (``on``/``off``, argparse dest
  ``announced_call_reask``, read with ``getattr``), else profile, else the default; ``(None, "unset")`` in ``off``.
- ``--announced-call-reask {on,off}``; with ``--tools off`` (or a profile key under a ``--tools off`` override) exit
  2, not argparse's usage error, no graph and no page server, with ``--dry-run`` and without.
- ``cli`` passes ``Session(..., reask_announced_calls=<bool>)`` in ``dry_run`` and ``live`` (``False`` for off, never
  ``None``), no such keyword in ``off``.
- The startup line, in both tools-on modes, under ``--dry-run`` and at a real start:
  ``astra session: reask: announced_call_reask=<on|off> (<src>)``; with defaults exactly
  ``astra session: reask: announced_call_reask=on (default)``; it is the last tools-on line: in live right after the
  ``search:`` line, and no ``astra session:`` line other than the regime line follows it (R-g, the reading of "the
  last tools-on startup line").

Pending rows fail at BASE a7811bed on a named ``AssertionError``. Active: ``test_other_lines_unchanged_default`` and
``test_profile_fields_otherwise_base``, green before and after.

Regime: CPU over fakes, nothing served; n = 1, deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests.unit import _t0201_pins as t0201  # T-0201 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0172_session_reask as sr
from tests.unit._tool_turn_support import answer
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run, tools_lines
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_t0132 import _toml
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out

BASE_SHA = sr.BASE_SHA
FIELD = "announced_call_reask"
FLAG = "--announced-call-reask"
KEY = sr.KEY
PREFIX = "astra session: reask: "
DEFAULT_LINE = PREFIX + "announced_call_reask=on (default)"
TOOLS_ON = ("dry_run", "live")
#: BASE's live tools-block lines before the search line (a7811bed, by hand from a default ``--dry-run``); the
#: ``search:`` line is counted, not spelled, because T-0163 (merged after BASE) owns its wording
BASE_LIVE_LINES = [
    "astra session: grounding: visual_tokens=1120 (default), reground=off (default)",
    "astra session: feedback: live_feedback=label (default), action_echo=model (default)",
    "astra session: recovery: thought_call_recovery=on (default)",
    "astra session: page overview: off (default)",
    "astra session: auto overview: off (default)",
    "astra session: overview frame: on (default)",
    "astra session: action outcome: off (default)",
    "astra session: repeat guard: off (default)",
]


# --------------------------------------------------------------------------------------------- seams


def require_flag() -> None:
    assert FLAG in flags_of(CLI), f"T-0172 B9: astra session {FLAG} does not exist yet"


def require_field() -> None:
    assert FIELD in Profile.model_fields, f"T-0172 B9: Profile.{FIELD} does not exist yet"


def require_resolver() -> tuple[Any, Any]:
    fn, cls = getattr(cli, "reask_of", None), getattr(cli, "Reask", None)
    assert callable(fn), "T-0172 B9: cli.reask_of(profile, args, mode) does not exist yet"
    assert isinstance(cls, type) and dataclasses.is_dataclass(cls), (
        "T-0172 B9: cli.Reask (a dataclass: value, source) does not exist yet"
    )
    assert len(dataclasses.fields(cls)) == 2, [f.name for f in dataclasses.fields(cls)]
    return fn, cls


def resolved(r: Any) -> tuple[Any, Any]:
    return dataclasses.astuple(r)  # type: ignore[return-value]


def reask_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def session_value(calls: T.Calls) -> Any:
    kw = calls.session_kwargs[-1]
    assert KEY in kw, f"T-0172 B9: cli passes Session(..., {KEY}=...); it passed {sorted(kw)}"
    return kw[KEY]


def _load_error(tmp_path: Path, body: str) -> str | None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        load(_toml(tmp_path, body))
    except ValueError as exc:
        return str(exc)
    return None


def assert_last_tools_line(out: str, mode: str, where: str) -> None:
    lines = out.splitlines()
    (i,) = [n for n, x in enumerate(lines) if x.startswith(PREFIX)]
    later = [x for x in lines[i + 1 :] if x.startswith("astra session: ") and "regime:" not in x]
    assert later == [], (
        f"T-0172 B9 (R-g, {mode} {where}): the reask line is the last tools-on line; after it {later}"
    )
    if mode == "live":
        assert lines[i - 1].startswith("astra session: search:"), (
            f"T-0172 B9 (R-g, live {where}): right after the search line; before it {lines[i - 1]!r}"
        )


# --------------------------------------------------------------------------------------------- config


def test_constant_and_profile_field():
    """B9: ``DEFAULT_ANNOUNCED_CALL_REASK`` is the bool ``True``; ``Profile.announced_call_reask`` is ``bool | None``,
    default ``None``, ``None`` in every built-in profile."""
    value = getattr(config, "DEFAULT_ANNOUNCED_CALL_REASK", None)
    assert value is True, (
        f"T-0172 B9: config.DEFAULT_ANNOUNCED_CALL_REASK = True does not exist yet; {value!r}"
    )
    require_field()
    f = Profile.model_fields[FIELD]
    assert f.default is None and not f.is_required() and f.annotation == (bool | None), f
    for name, profile in load(None).profiles.items():
        assert getattr(profile, FIELD) is None, name


def test_profile_fields_otherwise_base(tmp_path):
    """B9/B12 (active): every ``Profile`` field other than ``announced_call_reask`` has BASE's type and default."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0133, "BASE_SHA", BASE_SHA)
        base = t0133._base_config_module(tmp_path)
    now = t0133._fields(Profile, "local_astra.config")
    then = t0133._fields(base.Profile, "t0133_base_config")
    now.pop(FIELD, None)
    from tests.unit import _t0172_pins as pins

    now = pins.pop_live_guidance_field(now)  # T-0174 R-A3 pin move (A3): exactly live_guidance
    now = t0133.pop_max_tool_steps_web(now)  # T-0178 A7 pin move: exactly max_tool_steps_web
    now = t0133.pop_clip_policy_fields(now)  # T-0182 B pin move
    now = t0133.pop_frame_setting_fields(now)  # T-0227 R-2 re-pin
    # T-0163 merges into this branch before /implement and may add its own field; it is pinned by its own rows
    extra = sorted(set(now) - set(then))
    for k in extra:
        if "brave" in k:
            now.pop(k)
    assert now == then, (
        f"T-0172 B9: other Profile fields equal BASE's; differ in {sorted(set(now) ^ set(then))}"
    )


def test_profile_key_refusals(tmp_path):
    """B9: a non-bool is refused naming true and false; a non-``None`` value with ``tools = "off"`` or no tools key is
    refused; with ``dry_run`` and ``live`` both bools load and read back."""
    require_field()
    for junk in ('"off"', '"on"', '"true"', "1", "0", '"yes"'):
        msg = _load_error(
            tmp_path / f"j{junk.strip(chr(34))}", f'tools = "live"\n{FIELD} = {junk}\n'
        )
        assert msg is not None and "true" in msg.lower() and "false" in msg.lower(), (junk, msg)
    for tools in ('tools = "off"\n', ""):
        for value in ("true", "false"):
            msg = _load_error(tmp_path / f"o{len(tools)}{value}", f"{tools}{FIELD} = {value}\n")
            assert msg is not None, (
                f"T-0172 B9: {FIELD} = {value} with {tools.strip() or 'no tools key'} loaded"
            )
    for mode in ("dry_run", "live"):
        for value, want in (("true", True), ("false", False)):
            d = tmp_path / f"ok-{mode}-{value}"
            d.mkdir()
            got = getattr(load(_toml(d, f'tools = "{mode}"\n{FIELD} = {value}\n')).active(), FIELD)
            assert got is want, (mode, value, got)


# --------------------------------------------------------------------------------------------- resolver and default


def test_default_on_both_tools_modes(tmp_path, capsys, monkeypatch):
    """B9 (H-T172-2): no flag and no key resolve ``(True, "default")`` in ``dry_run`` and ``live`` (an empty
    ``Namespace()`` too), ``(None, "unset")`` in ``off``; ``cli`` builds both tools-on sessions with
    ``reask_announced_calls=True`` and passes no such keyword in ``off``."""
    reask_of, _ = require_resolver()
    for mode in TOOLS_ON:
        p = Profile(name="lean", tools=mode)  # type: ignore[arg-type]
        assert resolved(reask_of(p, argparse.Namespace(), mode)) == (True, "default"), mode
        assert resolved(reask_of(p, argparse.Namespace(**{FIELD: None}), mode)) == (True, "default")
    assert resolved(reask_of(Profile(name="lean"), argparse.Namespace(), "off")) == (None, "unset")
    for mode in ("dry_run", "live", "off"):
        monkeypatch.undo()
        calls = build_mode(tmp_path / mode, capsys, monkeypatch, mode)
        if mode == "off":
            assert KEY not in calls.session_kwargs[-1], calls.session_kwargs[-1].get(KEY)
        else:
            assert session_value(calls) is True, (mode, session_value(calls))


SOURCES = {
    "flag-off": ([FLAG, "off"], "", (False, "flag")),
    "flag-on": ([FLAG, "on"], "", (True, "flag")),
    "profile-false": ([], f"{FIELD} = false\n", (False, "profile")),
    "profile-true": ([], f"{FIELD} = true\n", (True, "profile")),
    "flag-on-over-profile-false": ([FLAG, "on"], f"{FIELD} = false\n", (True, "flag")),
    "flag-off-over-profile-true": ([FLAG, "off"], f"{FIELD} = true\n", (False, "flag")),
}


@pytest.mark.parametrize("mode", TOOLS_ON)
@pytest.mark.parametrize("case", list(SOURCES))
def test_sources(tmp_path, capsys, monkeypatch, mode, case):
    """B9: flag, else profile, else default; the line says the same; ``cli`` passes the bool (``False`` for off)."""
    require_flag()
    require_field()
    reask_of, _ = require_resolver()
    flags, body, (value, src) = SOURCES[case]
    profile_body = f'tools = "{mode}"\n' + body
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, mode, *flags, body=profile_body)
    word = "on" if value else "off"
    assert reask_lines(out) == [PREFIX + f"announced_call_reask={word} ({src})"], out
    monkeypatch.undo()
    calls = build_mode(tmp_path / "build", capsys, monkeypatch, mode, *flags, body=profile_body)
    assert session_value(calls) is value, (case, session_value(calls))
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", profile_body)).active()
    got = reask_of(profile, argparse.Namespace(**{FIELD: flags[1] if flags else None}), mode)
    assert resolved(got) == (value, src), got


# --------------------------------------------------------------------------------------------- the line


@pytest.mark.parametrize("mode", TOOLS_ON)
def test_reask_line_exact_and_last(tmp_path, capsys, monkeypatch, mode):
    """B9 (R-g): with defaults the line is exactly the default literal, once, and the last tools-on line, under
    ``--dry-run`` and at a real start; ``off`` prints none."""
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, mode)
    assert reask_lines(out) == [DEFAULT_LINE], f"T-0172 B9: printed\n{out}"
    assert_last_tools_line(out, mode, "--dry-run")
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *MODE_FLAGS[mode])
    assert code == 0 and reask_lines(started) == [DEFAULT_LINE], started
    assert_last_tools_line(started, mode, "start")
    monkeypatch.undo()
    other = dry_run(tmp_path / "off", capsys, monkeypatch, "off")
    assert "reask:" not in other, other


def _other_lines(out: str) -> list[str]:
    return [
        x
        for x in out.splitlines()
        if x.startswith("astra session: ") and not x.startswith(PREFIX) and "regime:" not in x
    ]


def test_other_lines_unchanged_default(tmp_path, capsys, monkeypatch):
    """B9 (active, green before and after): with the reask line set aside, a default ``--dry-run`` prints BASE's
    tools lines: live the eight literal lines then exactly one ``search:`` line, ``dry_run`` none; the ``tools=``
    lines are BASE's prefixes.

    T-0179 A-5 (the T-0174 D4b rig pattern): the live run passes ``--no-repeat-guard`` so it keeps BASE's guard (off)
    after the live default flipped to on; its ``off (flag)`` line is read back as BASE's ``off (default)`` (exactly
    once). The default itself is pinned by ``test_t0179_repeat_guard_default.py``."""
    live = dry_run(tmp_path / "live", capsys, monkeypatch, "live", "--no-repeat-guard")
    pinned = [x for x in live.splitlines() if x == "astra session: repeat guard: off (flag)"]
    assert len(pinned) == 1, f"T-0179 A-5: the pin prints its line once; {pinned}"
    live = live.replace(
        "astra session: repeat guard: off (flag)", "astra session: repeat guard: off (default)"
    )
    live = t0201.admit_default_line(
        live, "live --dry-run"
    )  # T-0201 D3 re-pin (admit): the default line
    live = t0207.admit_default_line(
        live, "live --dry-run"
    )  # T-0207 C3 re-pin (admit): the default line
    other = _other_lines(live)
    assert other[:-1] == BASE_LIVE_LINES and other[-1].startswith("astra session: search:"), other
    monkeypatch.undo()
    dry = dry_run(tmp_path / "dry", capsys, monkeypatch, "dry_run")
    assert _other_lines(dry) == [], dry
    assert tools_lines(dry) and tools_lines(dry)[0].startswith("tools=dry_run native:"), dry


@pytest.mark.parametrize("value", ["on", "off"])
def test_other_lines_unchanged_with_flag(tmp_path, capsys, monkeypatch, value):
    """B9: the flag changes only the reask line."""
    require_flag()
    base = _other_lines(dry_run(tmp_path / "base", capsys, monkeypatch, "live"))
    monkeypatch.undo()
    out = dry_run(tmp_path / "flag", capsys, monkeypatch, "live", FLAG, value)
    assert _other_lines(out) == base, out
    assert reask_lines(out) == [PREFIX + f"announced_call_reask={value} (flag)"], out


# --------------------------------------------------------------------------------------------- refusals


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
        assert code == 2, f"T-0172 B9 {flags} {body!r} {extra}: exit 2; got {code}\n{out}{err}"
        assert "unrecognized arguments" not in err and "invalid choice" not in err, err
        assert re.search(r"announced[-_]call[-_]reask", out + err), (
            f"the refusal names the flag or key: {out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
        assert not [e for e in events if e[0].startswith("pages.")], list(events)
        monkeypatch.undo()


@pytest.mark.parametrize("value", ["on", "off"])
def test_flag_refused_with_tools_off(tmp_path, capsys, monkeypatch, value):
    """B9: ``--announced-call-reask on|off`` with ``--tools off`` (plain, and over a live profile) exits 2."""
    require_flag()
    _refused(tmp_path / "plain", capsys, monkeypatch, [FLAG, value], "")
    _refused(
        tmp_path / "over", capsys, monkeypatch, ["--tools", "off", FLAG, value], 'tools = "live"\n'
    )


@pytest.mark.parametrize("value", ["true", "false"])
def test_profile_key_refused_under_tools_off(tmp_path, capsys, monkeypatch, value):
    """B9: a live profile with the key loads, but under ``--tools off`` it would do nothing: exit 2. Too-wide
    companion: under ``--tools dry-run`` the same profile builds (the key applies in dry_run)."""
    require_field()
    body = f'tools = "live"\nbrowser = "headless"\n{FIELD} = {value}\n'
    _refused(tmp_path / "over", capsys, monkeypatch, ["--tools", "off"], body)
    calls = build_mode(tmp_path / "ok", capsys, monkeypatch, "dry_run", body=body)
    assert session_value(calls) is (value == "true")


@pytest.mark.parametrize("junk", ["true", "yes", "ON", "default"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *MODE_FLAGS["live"], FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


_PURE = """
import argparse, builtins, json, socket, sys
from local_astra import cli
from local_astra.config import Profile
opened = []
real_open = builtins.open
def spy(*a, **k):
    opened.append(str(a[0]) if a else "?")
    return real_open(*a, **k)
class NoSocket:
    def __init__(self, *a, **k):
        raise AssertionError("reask_of opened a socket")
socket.socket = NoSocket
builtins.open = spy
try:
    out = [list(__import__("dataclasses").astuple(cli.reask_of(p, ns, mode))) for p, mode, ns in (
        (Profile(name="lean", tools="live"), "live", argparse.Namespace()),
        (Profile(name="lean", tools="dry_run"), "dry_run", argparse.Namespace(announced_call_reask="off")),
        (Profile(name="lean"), "off", argparse.Namespace()),
    )]
finally:
    builtins.open = real_open
print(json.dumps({"opened": opened, "values": out,
                  "computer": sorted(m for m in sys.modules if m.startswith("local_astra.computer"))}))
"""


def test_reask_of_pure():
    """B9: ``reask_of`` is pure: no socket, no file, no ``local_astra.computer`` import (a fresh interpreter)."""
    require_resolver()
    proc = subprocess.run(
        [sys.executable, "-c", _PURE], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["opened"] == [] and got["computer"] == [], got
    assert got["values"] == [[True, "default"], [False, "flag"], [None, "unset"]], got["values"]


def test_help_states_default():
    require_flag()
    h = t0133._help_of(FLAG)
    assert re.search(r"\bon\b", h) and re.search(r"\boff\b", h) and "default" in h.lower(), h


# --------------------------------------------------------------------------------------------- cli -> behaviour


def test_cli_default_session_reasks(tmp_path, capsys, monkeypatch):
    """B9 -> B1: a session built with the keyword a default ``cli`` dry-run build passes re-asks the smoke's step-2
    announcement once; with ``--announced-call-reask off`` it does not."""

    async def turn(tmp: Path, value: Any) -> Any:
        async with sr.reask_rig(tmp, reask=value, script=sr.STEP1) as rig:
            await sr.run(rig, [answer(sr.ANN_WEATHER)], reasks=[[answer(sr.R_STOPPED)]])
        return rig

    on = session_value(build_mode(tmp_path / "on", capsys, monkeypatch, "dry_run"))
    monkeypatch.undo()
    require_flag()
    off = session_value(build_mode(tmp_path / "off", capsys, monkeypatch, "dry_run", FLAG, "off"))
    monkeypatch.undo()
    assert (on, off) == (True, False), (on, off)
    assert len(asyncio.run(turn(tmp_path / "t-on", on)).llm.reask_prompts) == 1
    assert asyncio.run(turn(tmp_path / "t-off", off)).llm.reask_prompts == []
