"""T-0207 group C: the completion check on by default in live sessions, resolved in ``cli``; ``--completion-check
off`` restores BASE; never outside live (docs/tasks/T-0207.md C1-C3; B's GPU A/B PASSED 2026-10-30 at cb016fd8).

**The interfaces these rows pin** (C1's words; readings R-C1..R-C5 in the test-author's report):

- ``astra session --completion-check {on,off}`` (C1). A junk value is argparse's ``invalid choice`` (exit 2).
- ``cli`` builds every default live session with ``Session(..., completion_check=True)`` (C1); with
  ``--completion-check off`` it passes ``completion_check=False`` or no such keyword (both are A2's BASE ``Session``,
  A2.2). In ``off``/``dry_run`` it never passes a truthy ``completion_check`` (``Session`` refuses one there, A2.1).
- The start line, exactly C1's ``astra session: completion check: <on|off> (<flag|default>)``, printed once in live
  under ``--dry-run`` and at a real start. R-C1: C1 says it "sits before ``reask:``, and ``reask:`` stays last"; these
  rows pin that (index before the ``reask:`` line, no ``astra session:`` line other than the regime and HUD-page
  lines after ``reask:``) and, because rows that still hold already fix the neighbours (T-0125 criterion 3: the
  ``search:`` line follows every tools-block line; T-0174 A4: a guidance line sits immediately before ``reask:``),
  the line sits in the live block: after ``grounding:`` and before ``search:``. ``off``/``dry_run`` print none.
- R-C2 (no profile key): C1 names a flag and a default only and C owns ``cli.py`` alone, so ``Profile`` gains no
  field (``config.py`` is not C's); the line's source is ``flag`` or ``default``.
- C1 "the flag outside live is refused with exit 2, naming the flag" (not argparse's usage error), no graph and no
  ``Session`` built, for ``--tools off`` and ``--tools dry-run``, plain and over a live profile.
- R-C3 (C2 "byte-identical to BASE"): ``off`` and ``dry_run`` (T-0134c's ``OTHER`` cases) and live with
  ``--completion-check off`` are compared with BASE b4c6a6aa (R-C4: Merge T-0202-d; T-0207 C3 re-pin, was c9ecd1db)
  in BASE's own tree (``git archive``, origin-proved): the ``--dry-run`` output, a build's output, the client and session keywords, the executors made, the system message and
  one request body. The only differences admitted: a ``completion_check: False`` session keyword and, live only, the
  one ``completion check: off (flag)`` line. Turn behaviour under ``Session(completion_check=False)`` equals BASE's by
  A2.2's rows (``test_t0207_session_check.test_off_is_byte_identical_to_base``).
- R-C4 ("the turn line records it"): a session built with exactly the ``completion_check`` keyword a default ``cli``
  live build passes runs A2's claim turn (a click, then a no-call claim): the check fires once and the turn line's
  extras carry A2.11's ``completion_check: {on: true, asked: 1, outcome: "no_call"}``; with ``--completion-check off``
  no check request is sent and the key is absent.
- R-C5 (C3 census): the rows that pin the old default (no keyword, no line) are re-pinned in their own modules, each
  citing T-0207 C3, by ``tests/unit/_t0207_pins.py``.

Activated rows fail at BASE c9ecd1db on a named ``AssertionError`` (no flag, no keyword, no line). Active and green
before and after: ``test_other_modes_byte_identical_to_base`` (a negative), ``test_session_true_outside_live_still
_raises`` and ``test_no_profile_field``.

Regime: CPU over fakes, nothing served, ``ScriptedClock`` (virtual time), thinking on, tools ``live`` (1280x800
frames, ReGround off), loop on, n = 1, deterministic.
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
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0207_session_check as sc
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out

#: BASE: ``Merge T-0202-d`` at b4c6a6aa (wait tool on by default in live). R-C4: C merges after T-0202 D and its
#: BASE comparisons move to that merge, so a default live BASE build and a ``--completion-check off`` build both carry
#: the wait set (T-0207 C3 re-pin: was c9ecd1db, where group C branched). A literal, never a moving ref.
BASE_SHA = "b4c6a6aa50202e9d4c24081fd93773e8022bdcc7"

KW = "completion_check"
FLAG = "--completion-check"
LIVE = MODE_FLAGS["live"]
#: C1's start line, exactly; group 1 the value, group 2 the source
LINE = re.compile(r"^astra session: completion check: (on|off) \((flag|default)\)$")
#: any line that talks about a completion check at all (so a second, differently worded line is caught)
ANY_LINE = re.compile(r"completion[ _-]check", re.IGNORECASE)
ABSENT = sc.ABSENT


# --------------------------------------------------------------------------------------------- seams


def require_flag() -> None:
    """Checked before any exit code is read: argparse's own exit 2 on an unknown flag would satisfy a refusal row."""
    assert FLAG in flags_of(CLI), f"T-0207 C1: astra session {FLAG} does not exist yet"


def session_kw(calls: T.Calls) -> Any:
    """``cli``'s ``completion_check`` keyword to the last ``Session``, or ``ABSENT``."""
    assert calls.session_kwargs, "premise: cli constructed a Session"
    return calls.session_kwargs[-1].get(KW, ABSENT)


def check_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith("astra session: ") and ANY_LINE.search(x)]


def build_live(tmp: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = "") -> T.Calls:
    calls = build_mode(tmp, capsys, monkeypatch, "live", *flags, body=body)
    monkeypatch.undo()
    return calls


def assert_line_placed(out: str, want: str, where: str) -> None:
    """C1: exactly one completion check line, equal to ``want``, before the ``reask:`` line; ``reask:`` stays the last
    ``astra session:`` line other than the regime line."""
    got = check_lines(out)
    assert got == [want], (
        f"T-0207 C1 ({where}): the completion check line is {want!r}; printed {got}\n{out}"
    )
    assert LINE.match(want), f"premise: {want!r} matches C1's pattern"
    lines = out.splitlines()
    (i,) = [n for n, x in enumerate(lines) if x == want]
    (r,) = [n for n, x in enumerate(lines) if x.startswith("astra session: reask:")]
    assert i < r, f"T-0207 C1 ({where}): the completion check line sits before reask:\n{out}"
    # R-C1: a live-only line, so in the live block (after ``grounding:``); and before ``search:`` when that prints,
    # since T-0125 criterion 3 keeps search after every tools-block line and T-0174 A4 keeps the guidance line
    # immediately before reask (both rows still hold under C)
    (g,) = [n for n, x in enumerate(lines) if x.startswith("astra session: grounding:")]
    s = [n for n, x in enumerate(lines) if x.startswith("astra session: search:")]
    assert g < i and all(i < n for n in s), (
        f"T-0207 C1/R-C1 ({where}): after grounding: and before search: (T-0125 criterion 3)\n{out}"
    )
    later = [x for x in lines[r + 1 :] if x.startswith("astra session: ") and "regime:" not in x]
    later = [x for x in later if "HUD page" not in x]
    assert later == [], (
        f"T-0207 C1 ({where}): reask: stays the last tools-on line; after it {later}"
    )


# --------------------------------------------------------------------------------------------- C1: the default


def test_live_default_on(tmp_path, capsys, monkeypatch):
    """C1: a default live build (no flag) passes ``Session(..., completion_check=True)``, with ``--tools live``, with
    ``--no-web``, and with a live profile (``tools = "live"`` in the config, no ``--tools``)."""
    for name, flags in (("flags", ()), ("no-web", ("--no-web",))):
        got = session_kw(build_live(tmp_path / name, capsys, monkeypatch, *flags))
        assert got is True, (
            f"T-0207 C1 ({name}): a default live session gets {KW}=True; cli passed "
            f"{'no keyword' if got is ABSENT else repr(got)}"
        )
    calls = build_mode(tmp_path / "profile", capsys, monkeypatch, "off", body=t0134c.LIVE_PROFILE)
    monkeypatch.undo()
    assert calls.session_kwargs[-1].get("tool_kit") is not None, (
        "premise: the live profile built a live kit"
    )
    assert session_kw(calls) is True, f"T-0207 C1: the live profile's session gets {KW}=True"


def test_flag_off_and_on(tmp_path, capsys, monkeypatch):
    """C1: ``--completion-check off`` passes ``completion_check=False`` or no keyword (never ``None``, never truthy);
    ``--completion-check on`` passes ``True``. The flag changes no other ``Session`` keyword."""
    require_flag()
    default = build_live(tmp_path / "default", capsys, monkeypatch)
    off = build_live(tmp_path / "off", capsys, monkeypatch, FLAG, "off")
    on = build_live(tmp_path / "on", capsys, monkeypatch, FLAG, "on")
    got_off = session_kw(off)
    assert got_off is ABSENT or got_off is False, (
        f"T-0207 C1: --completion-check off passes {KW}=False or nothing; got {got_off!r}"
    )
    assert session_kw(on) is True, (
        f"T-0207 C1: --completion-check on passes {KW}=True; got {session_kw(on)!r}"
    )

    def rest(calls: T.Calls, tmp: Path) -> Any:
        return {
            k: v for k, v in t0134c._describe(calls.session_kwargs[-1], str(tmp)).items() if k != KW
        }

    assert (
        rest(off, tmp_path / "off")
        == rest(default, tmp_path / "default")
        == rest(on, tmp_path / "on")
    ), "T-0207 C1: the flag changes only the completion_check keyword"


@pytest.mark.parametrize("junk", ["true", "yes", "ON", "default"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """C1: the flag takes ``on``/``off`` only."""
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *LIVE, FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


def test_help_states_default():
    """C1: the help text names on, off and the default."""
    require_flag()
    h = t0133._help_of(FLAG)
    assert re.search(r"\bon\b", h) and re.search(r"\boff\b", h) and "default" in h.lower(), h


# --------------------------------------------------------------------------------------------- C1: the start line


def test_start_line_position_and_source(tmp_path, capsys, monkeypatch):
    """C1: live prints ``completion check: on (default)`` under ``--dry-run`` and at a real start, before ``reask:``,
    with ``reask:`` last; ``off (flag)`` and ``on (flag)`` with the flag (a flag equal to the default is still
    ``flag``)."""
    want = "astra session: completion check: on (default)"
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live")
    monkeypatch.undo()
    assert_line_placed(out, want, "--dry-run")
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
    monkeypatch.undo()
    assert code == 0, started
    assert_line_placed(started, want, "start")
    require_flag()
    for value in ("off", "on"):
        out = dry_run(tmp_path / f"flag-{value}", capsys, monkeypatch, "live", FLAG, value)
        monkeypatch.undo()
        assert_line_placed(
            out, f"astra session: completion check: {value} (flag)", f"--dry-run {FLAG} {value}"
        )
        _, started, code = build_out(
            tmp_path / f"start-{value}", capsys, monkeypatch, *LIVE, FLAG, value
        )
        monkeypatch.undo()
        assert code == 0, started
        assert_line_placed(
            started, f"astra session: completion check: {value} (flag)", f"start {FLAG} {value}"
        )


@pytest.mark.parametrize("value", ["on", "off"])
def test_flag_changes_only_its_line(tmp_path, capsys, monkeypatch, value):
    """C1: with the completion check line set aside, ``--completion-check <value>`` prints the default run's output."""
    require_flag()
    base = dry_run(tmp_path / "base", capsys, monkeypatch, "live")
    monkeypatch.undo()
    out = dry_run(tmp_path / "flag", capsys, monkeypatch, "live", FLAG, value)
    monkeypatch.undo()

    def rest(text: str, tmp: Path) -> list[str]:
        return [x.replace(str(tmp), "<tmp>") for x in text.splitlines() if not ANY_LINE.search(x)]

    assert rest(out, tmp_path / "flag") == rest(base, tmp_path / "base"), out
    assert len(check_lines(out)) == 1, out


# --------------------------------------------------------------------------------------------- C1: not outside live

OUTSIDE = {
    "off": ([], ""),
    "dry_run": (["--tools", "dry-run"], ""),
    "live-profile-off": (["--tools", "off"], t0134c.LIVE_PROFILE),
    "live-profile-dry": (["--tools", "dry-run"], t0134c.LIVE_PROFILE),
}


def test_not_wired_outside_live(tmp_path, capsys, monkeypatch):
    """C1/C2: ``off`` and ``dry_run`` (plain, and a live profile under ``--tools off``/``dry-run``) pass no truthy
    ``completion_check`` and print no completion check line, under ``--dry-run`` and at a build. Power row: the same
    build in live does pass ``True``."""
    assert session_kw(build_live(tmp_path / "power", capsys, monkeypatch)) is True, (
        "T-0207 C1: the live default passes completion_check=True (power row: the rest of this row is not vacuous)"
    )
    for case, (flags, body) in OUTSIDE.items():
        out = dry_run(tmp_path / f"d-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        assert check_lines(out) == [], f"T-0207 C1: no completion check line in {case}\n{out}"
        calls = build_mode(tmp_path / f"b-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        assert session_kw(calls) in (ABSENT, False), (
            f"T-0207 C1: {case} passes no truthy {KW}; got {session_kw(calls)!r}"
        )


@pytest.mark.parametrize("case", sorted(OUTSIDE))
@pytest.mark.parametrize("value", ["on", "off"])
@pytest.mark.parametrize("dry", [True, False], ids=["dry-run", "start"])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, case, value, dry):
    """C1: ``--completion-check on|off`` with ``--tools off`` or ``--tools dry-run`` (plain, and over a live profile)
    exits 2 under ``--dry-run`` and at a start, with a message naming ``--completion-check`` (not argparse's usage
    error), and builds no graph and no ``Session``."""
    require_flag()
    flags, body = OUTSIDE[case]
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli(
        [*argv(tmp_path, *flags, FLAG, value, config_body=body), *(["--dry-run"] if dry else [])]
    )
    out, err = _out(capsys)
    what = f"{case} {FLAG} {value} {'--dry-run' if dry else 'start'}"
    assert code == 2, (
        f"T-0207 C1 ({what}): the flag outside live is refused with exit 2; got {code}\n{out}{err}"
    )
    assert "unrecognized arguments" not in err and "invalid choice" not in err, (
        f"{what}: argparse's error\n{err}"
    )
    assert FLAG in out + err, f"T-0207 C1 ({what}): the refusal names {FLAG}\n{out}{err}"
    assert not calls.graphs and not calls.session_kwargs, (
        f"T-0207 C1 ({what}): a refused run built a graph or Session"
    )
    assert check_lines(out) == [], f"{what}: a refused run prints no completion check line\n{out}"


@pytest.mark.parametrize("mode", ["dry_run", "off"])
async def test_session_true_outside_live_still_raises(tmp_path: Path, mode: str) -> None:
    """A2.1 kept under C (active; green before and after): ``Session(completion_check=True)`` with tools ``dry_run``
    or off still raises ``ValueError`` naming ``live`` (A2's own row, run again here so a C change to ``session.py``
    that relaxed it for the cli default reddens a C row too)."""
    await sc.test_keyword_refused_outside_live(tmp_path, mode)


def test_no_profile_field():
    """R-C2 (active): C owns ``cli.py`` only; ``Profile`` gains no ``completion_check`` field."""
    assert KW not in Profile.model_fields, (
        f"T-0207 C: Profile.{KW} is not C's (config.py is not an owned file)"
    )


# --------------------------------------------------------------------------------------------- R-C4: the turn


def _carried(calls: T.Calls) -> dict[str, Any]:
    kw = calls.session_kwargs[-1]
    return {KW: kw[KW]} if KW in kw else {}


async def _claim_turn(tmp: Path, kwargs: dict[str, Any]) -> Any:
    """A2's claim turn (step 1 clicks on F0, step 2 claims with no call; the check, if asked, answers with no call) on
    A2's live rig built with exactly ``kwargs`` (``completion_check`` not passed unless ``cli`` passed it)."""
    return await sc.claim_turn(tmp, sc.ABSENT, extra=kwargs)


def test_default_cli_session_checks_and_turn_line_records_it(tmp_path, capsys, monkeypatch):
    """C1 -> A2 (R-C4): a session built with the ``completion_check`` keyword a default ``cli`` live build passes asks
    the check once on A2's claim turn (one ``llm.request`` with ``phase: "completion_check"``), voices the check's
    answer and never the draft claim, and its turn line's extras carry ``completion_check: {on: true, asked: 1,
    outcome: "no_call"}``."""
    kwargs = _carried(build_live(tmp_path / "cli", capsys, monkeypatch))
    assert kwargs.get(KW) is True, (
        f"T-0207 C1: a default cli live session gets {KW}=True; cli passed {kwargs}"
    )
    rig = asyncio.run(_claim_turn(tmp_path / "turn", kwargs))
    assert len(sc.check_requests(rig)) == 1, (
        f"T-0207 C1: the default cli session asks the completion check once; {len(sc.check_requests(rig))}"
    )
    spoken = sc.spoken_published(rig)
    assert sc.CHECK_SAYS in spoken and sc.CLAIM not in spoken, (
        f"T-0207 A2.5 via C1: voiced {spoken!r}"
    )
    assert sc.turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, (
        f"T-0207 C1/A2.11: the turn line records the check; extras key {sc.turn_key(rig)!r}"
    )


def test_flag_off_turn_is_base(tmp_path, capsys, monkeypatch):
    """C2 (negative): with ``--completion-check off`` the same claim turn sends no check request, voices the claim (as
    BASE) and its turn line has no ``completion_check`` key; the session built with ``cli``'s keywords sends the same
    request bodies as the keyword-absent session."""
    require_flag()
    kwargs = _carried(build_live(tmp_path / "cli-off", capsys, monkeypatch, FLAG, "off"))
    assert kwargs.get(KW, ABSENT) in (ABSENT, False), kwargs
    got = asyncio.run(_claim_turn(tmp_path / "off", kwargs))
    want = asyncio.run(_claim_turn(tmp_path / "base", {}))
    assert sc.check_requests(got) == [], "T-0207 C2: --completion-check off asks no check"
    assert sc.CLAIM in sc.spoken_published(got), (
        "T-0207 C2: with the check off the claim is voiced as at BASE"
    )
    assert sc.turn_key(got) is ABSENT, (
        f"T-0207 C2: no turn line key with the check off; {sc.turn_key(got)!r}"
    )
    # A2's dump (prefills, request bodies as sent, options, llm/note/tool records, HUD messages, machine log, turn
    # extras), with the session id and tmp normalised
    now, ref = sc._dump(got, tmp_path / "off"), sc._dump(want, tmp_path / "base")
    assert now["sent"] and now == ref, (
        "T-0207 C2: --completion-check off runs the keyword-absent session's turn: "
        f"{[k for k in ref if now.get(k) != ref[k]]} differ"
    )


# --------------------------------------------------------------------------------------------- C2: vs BASE


def _admit(got: dict[str, Any], key: str, live: bool) -> Any:
    v = got[key]
    if key == "session_kwargs":
        out = []
        for kw in v:
            assert kw.get(KW, False) is False, f"T-0207 C2: {KW}={kw.get(KW)!r} where BASE has none"
            out.append({k: x for k, x in kw.items() if k != KW})
        return out
    if key in ("dry", "build"):
        text = v["out"]
        lines = check_lines(text)
        if live:
            m = LINE.match(lines[0]) if len(lines) == 1 else None
            assert m is not None and m.groups() == ("off", "flag"), (
                f"T-0207 C1: the off run prints one 'off (flag)' line; {lines}"
            )
        else:
            assert lines == [], f"T-0207 C1: no completion check line outside live; {lines}"
        return {**v, "out": "\n".join(x for x in text.split("\n") if x not in lines)}
    return v


def _compare(now: dict[str, Any], want: dict[str, Any], case: str, *, live: bool) -> None:
    for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
        got = _admit(now, key, live)
        assert got == want[key], (
            f"T-0207 C2: {case} {key} differs from {BASE_SHA[:8]}:\n"
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
    tmp = tmp_path_factory.mktemp("t0207c_base")
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


def test_flag_off_live_build_is_base(base: dict[str, Any], tmp_path: Path) -> None:
    """C2 (negative, R-C3): a live build with ``--completion-check off`` equals BASE's default live build in BASE's
    tree: ``--dry-run`` output and a build's output (the one ``off (flag)`` line set aside), client and session
    keywords (a ``completion_check: False`` set aside), executors, system message and one request body."""
    require_flag()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0134c, "OTHER", (("live", (*LIVE, *t0134c.VT1120, FLAG, "off"), ""),))
        now = t0134c._other_modes(tmp_path / "now")["live"]
    _compare(now, base["live"]["live"], "live --completion-check off", live=True)


def test_other_modes_byte_identical_to_base(base: dict[str, Any], tmp_path: Path) -> None:
    """C2 (negative, active: green before and after): for ``off`` and ``dry_run`` (T-0134c's ``OTHER``: plain, loop
    off, a live profile under ``--tools off``/``--tools dry-run``) the ``--dry-run`` output, a build's output, client
    and session keywords, executors, system message and one request body equal BASE's (a ``completion_check: False``
    keyword is the one admitted difference). Power row: the same capture of a live build differs from BASE's ``off``
    capture, so the comparison can see a difference."""
    now = t0134c._other_modes(tmp_path / "other")
    assert set(now) == set(base["other"]) and len(now) == len(t0134c.OTHER) == 5, sorted(now)
    for case in now:
        _compare(now[case], base["other"][case], case, live=False)
    assert base["live"]["live"]["session_kwargs"] != base["other"]["off"]["session_kwargs"], (
        "power: a live build's session keywords differ from off's"
    )


# --------------------------------------------------------------------------------------------- R-C6: the resolver

RESOLVER = "completion_check_of"


def resolver() -> Any:
    """R-C6: ``cli.completion_check_of(args, mode) -> bool``; a named assertion (not an ``AttributeError``) before it
    exists."""
    from local_astra import cli

    fn = getattr(cli, RESOLVER, None)
    assert callable(fn), f"T-0207 R-C6: cli.{RESOLVER}(args, mode) -> bool does not exist yet"
    return fn


#: (namespace fields, mode) -> what R-C6 / C1 say: on by default in live, the flag's value in live, False elsewhere
RESOLVER_CASES: dict[str, tuple[dict[str, Any], str, bool]] = {
    "live-default-bare": ({"tools": "live"}, "live", True),
    "live-default-none": ({"tools": "live", "completion_check": None}, "live", True),
    "live-no-web-default": ({"tools": "live", "no_web": True}, "live", True),
    "live-flag-off": ({"tools": "live", "completion_check": "off"}, "live", False),
    "live-flag-on": ({"tools": "live", "completion_check": "on"}, "live", True),
    "off-default": ({"tools": "off"}, "off", False),
    "dry-default": ({"tools": "dry-run"}, "dry_run", False),
    "off-flag-on": ({"tools": "off", "completion_check": "on"}, "off", False),
    "dry-flag-on": ({"tools": "dry-run", "completion_check": "on"}, "dry_run", False),
    # the resolved mode decides, not ``args.tools`` (a live profile under ``--tools off`` resolves to off)
    "live-args-mode-off": ({"tools": "live"}, "off", False),
}


@pytest.mark.parametrize("case", list(RESOLVER_CASES))
def test_resolver_completion_check_of(case: str) -> None:
    """R-C6 + C1: ``cli.completion_check_of(Namespace(...), mode)`` is a ``bool``: ``True`` by default in live (with
    or without ``--no-web``), ``False`` with ``--completion-check off``, ``True`` with ``on``, and ``False`` outside
    live whatever the namespace says (the flag there is refused in ``session_cmd``, C1). Expected values typed from
    C1, not read from cli."""
    import argparse

    fields, mode, want = RESOLVER_CASES[case]
    got = resolver()(argparse.Namespace(**fields), mode)
    assert type(got) is bool, f"T-0207 R-C6 ({case}): -> bool; got {got!r}"
    assert got is want, (
        f"T-0207 R-C6/C1 ({case}): completion_check_of({fields}, {mode!r}) is {want}"
    )


def test_cli_build_resolves_through_resolver(tmp_path, capsys, monkeypatch):
    """R-C6 (adversarial: a default hard-coded beside the resolver): with ``cli.completion_check_of`` replaced by one
    that says off, a default live build passes no truthy ``completion_check``; replaced by one that says on, it passes
    ``True``; and the build asked it for ``live``. Power: the default build passes ``True`` (``test_live_default_on``)."""
    resolver()
    from local_astra import cli

    for say in (False, True):
        asked: list[str] = []

        def fake(args: Any, mode: str, _say: bool = say, _asked: list[str] = asked) -> bool:
            _asked.append(mode)
            return _say if mode == "live" else False

        monkeypatch.setattr(cli, RESOLVER, fake)
        got = session_kw(build_live(tmp_path / f"say-{say}", capsys, monkeypatch))
        assert "live" in asked, (
            f"T-0207 R-C6: the live build never asked cli.{RESOLVER}; asked {asked}"
        )
        if say:
            assert got is True, (
                f"T-0207 R-C6: resolver on -> completion_check=True; cli passed {got!r}"
            )
        else:
            assert got is ABSENT or got is False, (
                f"T-0207 R-C6: resolver off -> no truthy completion_check; cli passed {got!r}"
            )


def _harness_keywords(which: str) -> Any:
    """The T-0205 harness's ``session_keywords`` (C: ``scripts/gpu/t0205_demo_run.py``, D:
    ``scripts/gpu/t0205_real_sites.py``), called the way its own module's rows call it."""
    if which == "demo":
        from local_astra.config import load
        from tests.unit import test_t0205_demo_harness as h

        return dict(h.api("session_keywords")(load().profiles["lean"]))
    from tests.unit import test_t0205_real_sites_harness as h

    return dict(h.api("session_keywords")())


@pytest.mark.parametrize("which", ["demo", "real_sites"])
@pytest.mark.parametrize("on", [False, True], ids=["resolver-off", "resolver-on"])
def test_harness_keywords_resolve_through_cli(
    monkeypatch: pytest.MonkeyPatch, which: str, on: bool
) -> None:
    """R-C2 + R-C6 (never hard-coded): with ``cli.completion_check_of`` replaced at run time, the T-0205 C/D
    harness's ``session_keywords`` asks it for ``live`` in web mode and carries ``completion_check=True`` exactly when
    it says on; when it says off the keywords carry no truthy ``completion_check``. The off case alone would pass a
    harness that never asks, so the ask is asserted in both."""
    from local_astra import cli

    asked: list[tuple[Any, str]] = []

    def fake(args: Any, mode: str) -> bool:
        asked.append((getattr(args, "no_web", None), mode))
        return on if mode == "live" else False

    # raising=False: before C the attribute does not exist; the row then fails on the ask assertion below
    monkeypatch.setattr(cli, RESOLVER, fake, raising=False)
    got = _harness_keywords(which)
    assert any(mode == "live" and not no_web for no_web, mode in asked), (
        f"T-0207 R-C2/R-C6 ({which}): session_keywords never asked cli.{RESOLVER} for live web mode; asked {asked}"
    )
    if on:
        assert got.get(KW) is True, (
            f"T-0207 R-C2 ({which}): resolver on -> completion_check=True; got {got.get(KW, ABSENT)!r}"
        )
    else:
        assert not got.get(KW), (
            f"T-0207 R-C2 ({which}): resolver off -> no truthy completion_check; got {got.get(KW)!r}"
        )
