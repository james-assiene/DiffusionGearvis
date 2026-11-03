"""T-0201 group D: the screen reminder on by default in live sessions, resolved in ``cli``; ``--screen-reminder off``
restores BASE; off outside live (docs/tasks/T-0201.md D1-D3; C's GPU A/B PASSED 2026-10-29 at 25b7bb56).

**The interfaces these rows pin** (the task's; readings R-D1..R-D5 in the test-author's report):

- ``astra session --screen-reminder {on,off}`` (D1). R-D1: the flag spells its value ``on``/``off`` like
  ``--thought-call-recovery``; a junk value is argparse's ``invalid choice`` (exit 2).
- ``cli`` builds every default live session with ``Session(..., screen_reminder=True)`` (D1); with
  ``--screen-reminder off`` it passes ``screen_reminder=False`` or no such keyword (both are BASE's ``Session``, B1).
  In ``off``/``dry_run`` it never passes a truthy ``screen_reminder`` (``Session`` refuses one there, R-S3 of B).
- R-D2 (the start line, "reports ``(flag|default)`` like ``--thought-call-recovery``"): live prints exactly one line
  ``astra session: screen reminder: [screen_reminder=]<on|off> (<flag|default>)`` under ``--dry-run`` and at a real
  start; the wording between the prefix and the value is the implementer's within that pattern (the two shapes of
  ``recovery:`` and ``repeat guard:``). It sits in the live tools block: after the ``grounding:`` line, before the
  ``search:`` line, and T-0172's ``reask:`` line stays the last tools-on line. ``off``/``dry_run`` print none.
- R-D3 (no profile key): D1 names a flag and a default only and Group D owns ``cli.py`` alone, so ``Profile`` gains no
  field (``config.py`` is not D's); the line's source is ``flag`` or ``default``.
- R-D4 ("off outside live"): the task does not say whether ``--screen-reminder`` with ``--tools off``/``dry-run`` is
  refused (the ``recovery_flag_refusal`` pattern) or ignored; the row accepts either, provided a refusal is exit 2
  naming the flag (not argparse's usage error) with no graph built, and an accepted run passes no truthy keyword and
  prints no line.
- R-D5 (D2 "byte-identical to BASE"): ``off`` and ``dry_run`` (T-0134c's ``OTHER`` cases: plain, loop off, a live
  profile under ``--tools off``/``dry-run``) and live with ``--screen-reminder off`` are compared with BASE 01f2b7fa
  in BASE's own tree (``git archive``, origin-proved): the ``--dry-run`` output, a build's output, the client and
  session keywords, the executors made, the system message and one request body. The only differences admitted:
  a ``screen_reminder: False`` session keyword and, live only, the one ``screen reminder: off (flag)`` line.
  Request bytes of a step under ``Session(screen_reminder=False)`` equal BASE's by B1's rows; here the cli-built
  session's step requests are compared with the keyword-absent session's over the same scenario.

Activated rows fail at BASE 01f2b7fa on a named ``AssertionError`` (no flag, no keyword, no line). Active and green
before and after: ``test_other_modes_byte_identical_to_base`` (a negative) and ``test_no_profile_field``.

Regime: CPU over fakes, nothing served, ``ScriptedClock`` (virtual time), thinking on, tools ``live`` (1280x800
frames, ReGround off, screen label on), loop on, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.config import Profile
from tests import _git_range as gr
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0201_session_reminder as sr
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_session_live import _wire
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0201_loop_reminder import STEM

#: BASE: main at 01f2b7fa (``T-0201 C run: PASS``), where group D branched. A literal, never a moving ref.
BASE_SHA = "01f2b7fa"

KW = "screen_reminder"
FLAG = "--screen-reminder"
LIVE = MODE_FLAGS["live"]
#: R-D2: the start line's pattern; group 1 the value, group 2 the source
LINE = re.compile(
    r"^astra session: screen reminder: (?:screen_reminder=)?(on|off) \((flag|default)\)$"
)
#: any line that talks about a screen reminder at all (so a second, differently worded line is caught)
ANY_LINE = re.compile(r"screen[ _]reminder", re.IGNORECASE)
#: the behaviour keywords ``cli`` passes that the rig session is built with (T-0135b's ``CARRIED`` pattern): the
#: reminder follows the screen label, which ``live_feedback`` turns on
CARRIED = (KW, "live_feedback")


# --------------------------------------------------------------------------------------------- seams


def require_flag() -> None:
    """Checked before any exit code is read: argparse's own exit 2 on an unknown flag would satisfy a refusal row."""
    assert FLAG in flags_of(CLI), f"T-0201 D1: astra session {FLAG} does not exist yet"


def session_kw(calls: T.Calls) -> Any:
    """``cli``'s ``screen_reminder`` keyword to the last ``Session``, or ``sr.ABSENT``."""
    assert calls.session_kwargs, "premise: cli constructed a Session"
    return calls.session_kwargs[-1].get(KW, sr.ABSENT)


def carried(calls: T.Calls) -> dict[str, Any]:
    kw = calls.session_kwargs[-1]
    return {k: kw[k] for k in CARRIED if k in kw}


def reminder_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith("astra session: ") and ANY_LINE.search(x)]


def build_live(tmp: Path, capsys: Any, monkeypatch: Any, *flags: str) -> T.Calls:
    calls = build_mode(tmp, capsys, monkeypatch, "live", *flags)
    monkeypatch.undo()
    return calls


def assert_line_in_block(out: str, want: str, where: str) -> None:
    """R-D2: exactly one screen-reminder line, equal to ``want``, after ``grounding:`` and before ``search:``; the
    ``reask:`` line stays the last ``astra session:`` line other than the regime line."""
    got = reminder_lines(out)
    assert got == [want], (
        f"T-0201 D1 ({where}): the screen reminder line is {want!r}; printed {got}\n{out}"
    )
    assert LINE.match(want), f"premise: {want!r} matches R-D2's pattern"
    lines = out.splitlines()
    (i,) = [n for n, x in enumerate(lines) if x == want]
    (g,) = [n for n, x in enumerate(lines) if x.startswith("astra session: grounding:")]
    (s,) = [n for n, x in enumerate(lines) if x.startswith("astra session: search:")]
    assert g < i < s, (
        f"T-0201 D1/R-D2 ({where}): in the live block, after grounding and before search\n{out}"
    )
    later = [
        x
        for x in lines[
            lines.index(next(x for x in lines if x.startswith("astra session: reask:"))) + 1 :
        ]
        if x.startswith("astra session: ") and "regime:" not in x
    ]
    assert later == [], (
        f"T-0172 B9 (R-g) kept ({where}): reask stays the last tools-on line; after it {later}"
    )


async def _three(tmp: Path, kwargs: dict[str, Any]) -> Any:
    """B's three-step scenario (step 2 ``updated``, step 3 ``unchanged``) on a rig session built with exactly
    ``kwargs`` (no default label: ``live_feedback`` comes from ``cli`` too)."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with sr.live(
        tmp,
        sr.three_browser(),
        script=[answer(sr.S1), tool(VALID_CLICK)],
        on=sr.ABSENT,
        extra=kwargs,
        feedback=None,
    ) as rig:
        await sr.run_steps(rig, sr.THREE_STEPS)
    return rig


# --------------------------------------------------------------------------------------------- D1: the default


def test_live_default_on(tmp_path, capsys, monkeypatch):
    """D1: a default live build (no flag) passes ``Session(..., screen_reminder=True)``, at the default and with a
    live profile (``tools = "live"`` in the config, no ``--tools``)."""
    calls = build_live(tmp_path / "flags", capsys, monkeypatch)
    got = session_kw(calls)
    assert got is True, (
        f"T-0201 D1: a default live session gets {KW}=True; cli passed "
        f"{'no keyword' if got is sr.ABSENT else repr(got)}"
    )
    calls = build_mode(tmp_path / "profile", capsys, monkeypatch, "off", body=t0134c.LIVE_PROFILE)
    monkeypatch.undo()
    assert calls.session_kwargs[-1].get("tool_kit") is not None, (
        "premise: the live profile built a live kit"
    )
    assert session_kw(calls) is True, f"T-0201 D1: the live profile's session gets {KW}=True"


def test_flag_off_and_on(tmp_path, capsys, monkeypatch):
    """D1: ``--screen-reminder off`` passes ``screen_reminder=False`` or no keyword (never ``None``, never truthy);
    ``--screen-reminder on`` passes ``True``. The flag changes no other ``Session`` keyword."""
    require_flag()
    default = build_live(tmp_path / "default", capsys, monkeypatch)
    off = build_live(tmp_path / "off", capsys, monkeypatch, FLAG, "off")
    on = build_live(tmp_path / "on", capsys, monkeypatch, FLAG, "on")
    got_off = session_kw(off)
    assert got_off is sr.ABSENT or got_off is False, (
        f"T-0201 D1: --screen-reminder off passes {KW}=False or nothing; got {got_off!r}"
    )
    assert session_kw(on) is True, (
        f"T-0201 D1: --screen-reminder on passes {KW}=True; got {session_kw(on)!r}"
    )
    w = {
        "default": str(tmp_path / "default"),
        "off": str(tmp_path / "off"),
        "on": str(tmp_path / "on"),
    }

    def rest(calls: T.Calls, tmp: str) -> Any:
        return {k: v for k, v in t0134c._describe(calls.session_kwargs[-1], tmp).items() if k != KW}

    assert rest(off, w["off"]) == rest(default, w["default"]) == rest(on, w["on"]), (
        "T-0201 D1: the flag changes only the screen_reminder keyword"
    )


@pytest.mark.parametrize("junk", ["true", "yes", "ON", "default"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """R-D1: the flag takes ``on``/``off`` only."""
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *LIVE, FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


def test_help_states_default():
    """D1: the help text names on, off and the default."""
    require_flag()
    h = t0133._help_of(FLAG)
    assert re.search(r"\bon\b", h) and re.search(r"\boff\b", h) and "default" in h.lower(), h


# --------------------------------------------------------------------------------------------- D1: the start line


def test_start_line_source(tmp_path, capsys, monkeypatch):
    """D1/R-D2: live prints the default line ``on (default)`` under ``--dry-run`` and at a real start; ``off (flag)``
    and ``on (flag)`` with the flag (a flag equal to the default is still ``flag``)."""
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live")
    monkeypatch.undo()
    got = reminder_lines(out)
    assert len(got) == 1 and LINE.match(got[0]), (
        f"T-0201 D1: live --dry-run prints one screen reminder line matching {LINE.pattern}; printed {got}"
    )
    m = LINE.match(got[0])
    assert m is not None and m.groups() == ("on", "default"), (
        f"T-0201 D1: the default line says on (default): {got}"
    )
    default_line = got[0]
    assert_line_in_block(out, default_line, "--dry-run")
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
    monkeypatch.undo()
    assert code == 0, started
    assert_line_in_block(started, default_line, "start")
    require_flag()
    for value in ("off", "on"):
        out = dry_run(tmp_path / f"flag-{value}", capsys, monkeypatch, "live", FLAG, value)
        monkeypatch.undo()
        want = default_line.replace("on (default)", f"{value} (flag)")
        assert_line_in_block(out, want, f"--dry-run {FLAG} {value}")


@pytest.mark.parametrize("value", ["on", "off"])
def test_flag_changes_only_its_line(tmp_path, capsys, monkeypatch, value):
    """D1: with the screen reminder line set aside, ``--screen-reminder <value>`` prints the default run's output."""
    require_flag()
    base = dry_run(tmp_path / "base", capsys, monkeypatch, "live")
    monkeypatch.undo()
    out = dry_run(tmp_path / "flag", capsys, monkeypatch, "live", FLAG, value)
    monkeypatch.undo()

    def rest(text: str, tmp: Path) -> list[str]:
        return [x.replace(str(tmp), "<tmp>") for x in text.splitlines() if not ANY_LINE.search(x)]

    assert rest(out, tmp_path / "flag") == rest(base, tmp_path / "base"), out
    assert len(reminder_lines(out)) == 1, out


# --------------------------------------------------------------------------------------------- D1: not outside live


def test_not_wired_outside_live(tmp_path, capsys, monkeypatch):
    """D1 "off outside live" (R-D4): ``off`` and ``dry_run`` (plain, and a live profile under ``--tools off``/
    ``dry-run``) pass no truthy ``screen_reminder`` and print no screen reminder line, under ``--dry-run`` and at a
    build. R-D4 (ruled refusal, review 2026-10-29): ``--screen-reminder on`` and ``off`` there exit 2, under
    ``--dry-run`` and at a start, with a message naming ``--screen-reminder`` (not argparse's usage error), and build
    no graph and no ``Session``; for ``--tools off`` and ``--tools dry-run`` alike, plain and over a live profile. Power
    row: the same build in live does pass ``True``."""
    assert session_kw(build_live(tmp_path / "power", capsys, monkeypatch)) is True, (
        "T-0201 D1: the live default passes screen_reminder=True (power row: the rest of this row is not vacuous)"
    )
    cases = {
        "off": ([], ""),
        "dry_run": (["--tools", "dry-run"], ""),
        "live-profile-off": (["--tools", "off"], t0134c.LIVE_PROFILE),
        "live-profile-dry": (["--tools", "dry-run"], t0134c.LIVE_PROFILE),
    }
    for case, (flags, body) in cases.items():
        out = dry_run(tmp_path / f"d-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        assert reminder_lines(out) == [], f"T-0201 D1: no screen reminder line in {case}\n{out}"
        calls = build_mode(tmp_path / f"b-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        assert session_kw(calls) in (sr.ABSENT, False), (
            f"T-0201 D1: {case} passes no truthy {KW}; got {session_kw(calls)!r}"
        )
    require_flag()
    for case, (flags, body) in cases.items():
        for value in ("on", "off"):
            for extra in (["--dry-run"], []):
                where = tmp_path / f"f-{case}-{value}-{len(extra)}"
                where.mkdir(parents=True)
                monkeypatch.setenv("DISPLAY", ":99")
                calls = T.Calls()
                T.fake_clients(monkeypatch, calls)
                T.stop_before_serving(monkeypatch, calls)
                T.patch_detok_load(monkeypatch, calls, where)
                fake_live(monkeypatch, Events())
                code = run_cli([*argv(where, *flags, FLAG, value, config_body=body), *extra])
                out, err = _out(capsys)
                monkeypatch.undo()
                what = f"{case} {FLAG} {value} {extra}"
                assert code == 2, (
                    f"T-0201 D1 (R-D4, {what}): the flag outside live is refused with exit 2; got {code}\n{out}{err}"
                )
                assert "unrecognized arguments" not in err and "invalid choice" not in err, (
                    f"{what}: not argparse's usage error\n{err}"
                )
                assert FLAG in out + err, (
                    f"T-0201 D1 (R-D4, {what}): the refusal names {FLAG}\n{out}{err}"
                )
                assert not calls.graphs and not calls.session_kwargs, (
                    f"T-0201 D1 (R-D4, {what}): a refused run built a graph or a Session"
                )
                assert reminder_lines(out) == [], (
                    f"{what}: a refused run prints no reminder line\n{out}"
                )


# --------------------------------------------------------------------------------------------- D1/D2: the requests


def test_default_cli_session_reminds(tmp_path, capsys, monkeypatch):
    """D1 -> B2/B3: a rig session built with exactly the ``screen_reminder``/``live_feedback`` keywords a default
    ``cli`` live build passes sends step 2 (``updated``) and step 3 (``unchanged``) with ``SEP + reminder`` after the
    last tool response (H-T201-1 typed by hand); step 1 has none."""
    kwargs = carried(build_live(tmp_path / "cli", capsys, monkeypatch))
    assert kwargs.get(KW) is True, (
        f"T-0201 D1: a default cli live session gets {KW}=True; cli passed {kwargs}"
    )
    assert "live_feedback" in kwargs, f"premise: cli passes live_feedback in live; {sorted(kwargs)}"
    rig = asyncio.run(_three(tmp_path / "turn", kwargs))
    assert STEM not in json.dumps(_wire(tsl.sent(rig, "t1-gen"))), (
        "T-0201 B4: step 1 carries no reminder"
    )
    for k, key in ((2, "updated"), (3, "unchanged")):
        sr.assert_reminded(tsl.step_request(rig, k), key, f"cli default, step {k}")


def test_flag_off_requests_are_base(tmp_path, capsys, monkeypatch):
    """D2 (negative; adversarial for a flag that is parsed but not wired, or wired to ``None``): with
    ``--screen-reminder off`` the rig session built with what ``cli`` passed sends every request (steps 1, 2 and
    3) byte for byte as the session without the keyword (BASE's ``Session``, B1), no reminder anywhere and
    no ``screen_reminder`` key in any ``llm.request`` record. Power row: the default build's bodies differ."""
    require_flag()
    off = carried(build_live(tmp_path / "cli-off", capsys, monkeypatch, FLAG, "off"))
    on = carried(build_live(tmp_path / "cli-on", capsys, monkeypatch))
    assert off.get(KW, sr.ABSENT) in (sr.ABSENT, False), off
    base_kwargs = {k: v for k, v in off.items() if k != KW}
    got = asyncio.run(_three(tmp_path / "off", off))
    want = asyncio.run(_three(tmp_path / "base", base_kwargs))
    power = asyncio.run(_three(tmp_path / "on", on))
    bodies, base_bodies = sr.every_body(got), sr.every_body(want)
    assert len(bodies) == 3, f"premise: steps 1-3 were sent; {len(bodies)}"
    assert bodies == base_bodies, "T-0201 D2: --screen-reminder off sends BASE's bodies"
    sr.assert_no_reminder(got, "--screen-reminder off")
    assert sr.every_body(power) != base_bodies, (
        "power: the default build's requests carry the reminder"
    )


def test_flag_off_live_build_is_base(base: dict[str, Any], tmp_path: Path) -> None:
    """D2 (negative, R-D5): a live build with ``--screen-reminder off`` equals BASE's default live build in BASE's
    tree: ``--dry-run`` output and a build's output (the one ``off (flag)`` line set aside), client and session
    keywords (a ``screen_reminder: False`` set aside), executors, system message and one request body."""
    require_flag()
    # T-0202 D3 re-pin: pin off (--wait-tool off appended on a tree with the flag, so the build is BASE's,
    # which has no wait tool; the screen reminder flag under test is unchanged)
    flags = t0202.pin_off([*LIVE, *t0134c.VT1120, FLAG, "off"])
    # T-0207 C3 re-pin: pin off (--completion-check off appended on a tree with that flag, so the session is
    # BASE's, which has no completion check; its one "off (flag)" line and a False keyword set aside below)
    check_pinned = t0207.pins(flags)
    flags = t0207.pin_off(flags)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0134c, "OTHER", (("live", tuple(flags), ""),))
        now = t0134c._other_modes(tmp_path / "now")["live"]
    if check_pinned:
        now = t0207.section_without_off_pin(
            now, "T-0201 D2 live --screen-reminder off"
        )  # T-0207 C3 re-pin
    want = base["live"]["live"]
    _compare(now, want, "live --screen-reminder off", live=True)


# --------------------------------------------------------------------------------------------- D2: other modes vs BASE


def _admit(got: dict[str, Any], key: str, live: bool) -> Any:
    v = got[key]
    if key == "session_kwargs":
        out = []
        for kw in v:
            assert kw.get(KW, False) is False, f"T-0201 D2: {KW}={kw.get(KW)!r} where BASE has none"
            out.append({k: x for k, x in kw.items() if k != KW})
        return out
    if key in ("dry", "build"):
        text = v["out"]
        lines = reminder_lines(text)
        if live:
            assert (
                len(lines) == 1
                and LINE.match(lines[0])
                and LINE.match(lines[0]).groups() == ("off", "flag")
            ), (  # type: ignore[union-attr]
                f"T-0201 D1: the off run prints one 'off (flag)' line; {lines}"
            )
        else:
            assert lines == [], f"T-0201 D1: no screen reminder line outside live; {lines}"
        return {**v, "out": "\n".join(x for x in text.split("\n") if x not in lines)}
    return v


def _compare(now: dict[str, Any], want: dict[str, Any], case: str, *, live: bool) -> None:
    for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
        got = _admit(now, key, live)
        assert got == want[key], (
            f"T-0201 D2: {case} {key} differs from {BASE_SHA}:\n"
            f"base {json.dumps(want[key])[:1500]}\nnow  {json.dumps(got)[:1500]}"
        )


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0134c_live_feedback_default as m
out = pathlib.Path(sys.argv[1])
other = m._other_modes(out.parent / "run" / "other")
m.OTHER = (("live", (*m.MODE_FLAGS["live"], *m.VT1120), ""),)
live = m._other_modes(out.parent / "run" / "live")
out.write_text(json.dumps({"origin": local_astra.__file__, "other": other, "live": live}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture in BASE's tree (T-0135b's ``git archive`` harness through ``tsl._base_tree``): T-0134c's
    ``OTHER`` cases and a default live build."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0201d_base")
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
    return {"other": data["other"], "live": data["live"]}


def test_other_modes_byte_identical_to_base(base: dict[str, Any], tmp_path: Path) -> None:
    """D2 (negative, active: green before and after): for ``off`` and ``dry_run`` (T-0134c's ``OTHER``: plain, loop
    off, a live profile under ``--tools off``/``--tools dry-run``) the ``--dry-run`` output, a build's output, client
    and session keywords, executors, system message and one request body equal BASE's (a ``screen_reminder: False``
    keyword is the one admitted difference). Power row: the same capture of a live build differs from BASE's
    ``off`` capture, so the comparison can see a difference."""
    now = t0134c._other_modes(tmp_path / "other")
    assert set(now) == set(base["other"]) and len(now) == len(t0134c.OTHER) == 5, sorted(now)
    for case in now:
        _compare(now[case], base["other"][case], case, live=False)
    assert base["live"]["live"]["session_kwargs"] != base["other"]["off"]["session_kwargs"], (
        "power: a live build's session keywords differ from off's"
    )


def test_no_profile_field():
    """R-D3 (active): D owns ``cli.py`` only; ``Profile`` gains no ``screen_reminder`` field."""
    assert KW not in Profile.model_fields, (
        f"T-0201 D: Profile.{KW} is not D's (config.py is not an owned file)"
    )
