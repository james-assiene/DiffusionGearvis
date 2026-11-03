"""T-0211 group B, B3 (config), B4 (selection, through the real ``cli`` builder over fakes) and B8 (nothing else
moves) (docs/tasks/T-0211.md).

**The interfaces these rows pin** (the task's; readings stated where it is silent):

- B3: ``config.LIVE_GUIDANCE_VERSIONS == ("v1", "v2", "v3", "v4")``; ``config.DEFAULT_LIVE_GUIDANCE == "v3"``
  (until group D); ``Profile.live_guidance`` loads ``"v4"`` with ``tools = "live"`` and refuses it elsewhere; an
  unknown value is refused naming v1, v2, v3 and v4.
- B4, in live: with web on and v4 (``--live-guidance v4`` or ``live_guidance = "v4"``) the session's
  ``tool_guidance`` is ``LIVE_GUIDANCE["v4"]`` itself with ``--wait-tool off``, and
  ``prompt_builder.with_wait_guidance(LIVE_GUIDANCE["v4"])`` with the wait tool on; with ``--no-web`` or
  ``browser_web = false`` it is ``LIVE_GUIDANCE["v2"]`` itself (``is``) with the wait tool off, and
  ``with_wait_guidance(LIVE_GUIDANCE["v2"])`` with it on. ``page_fact=True`` for v2, v3 and v4 in both web states
  (**reading**, as T-0191: the resolved version under ``--no-web`` is still v4). ``--live-guidance v4`` with
  ``--tools off|dry-run`` (and ``live_guidance = "v4"`` under such an override) exits 2 with cli's own refusal before
  any graph, page server or browser (T-0174 A4's ``_refused``).
- B4 startup line: a flag or profile v4 source prints exactly ``astra session: live guidance: v4 (<source>)`` once,
  immediately before the reask line; **reading** (T-0191 Q2 carried over): with ``--no-web`` the same one line and
  nothing else; the default source prints nothing.
- B4 adversarial (``test_no_web_never_leaks_sentences``): a cli-built live v4 session under ``--no-web`` (and
  ``browser_web = false``), two turns, the first with three tool steps one of which is a refused ``Ctrl+L``: the first
  40 characters of S_SHARED and of S_SOURCE (typed from the task) appear in no request body, session-log record or turn
  line; power: the same rig in web mode carries both in every body.
- B8 (``test_v1_v2_v3_bytes_equal_base``): ``_capture`` runs here and in a subprocess over ``git archive BASE``
  (``local_astra`` from BASE, origin checked; T-0174's harness): no flag, ``v1``, ``v2`` and ``v3`` (flag and
  profile), in web mode and with ``--no-web`` / ``browser_web = false``, with the live defaults as they are at BASE
  (wait tool, screen reminder, completion check on; nothing pinned) and one wait-off case; ``cli``'s exit, stdout,
  stderr and every ``Session``/executor keyword under ``--dry-run`` and at a real start; the rig sessions built from
  those keywords (T-0152's ``_rig_turn`` on a fitting and a long page, T-0174 A5's two-turn bodies); tools off and
  dry-run; the executor's actions over T-0111c's fake driver. All byte for byte, nothing admitted.
- B8 frozen paths: byte-identical over ``RANGE_BASE..TIP`` (B-R3: ``RANGE_BASE`` = ``merge-base(main, TIP)``,
  ``TIP`` the implementation commit, both literals set at activation); docs/02 §4 unedited over the same range.

BASE is 175a29b0 (main at the branch point of the B tests; it holds T-0211's planning commit 81b2a5cb, docs only,
and no T-0211 code). ``test_default_still_v3``, ``test_v1_v2_v3_selection_unchanged``, ``test_flag_junk_exits_2``,
``test_page_fact_v2_v3_v4[v2|v3]``, ``test_v1_v2_v3_bytes_equal_base``, ``test_other_modes_and_executor_equal_base``,
``test_frozen_paths_unchanged`` and ``test_section4_invariants_unedited`` held at BASE. Every other row was
``pending(task="T-0211")`` and failed at BASE on a named ``AssertionError`` ("... does not exist yet": ``"v4"`` was
not in ``LIVE_GUIDANCE_VERSIONS`` and ``LIVE_TOOL_GUIDANCE_V4`` did not exist). Activated 2026-10-30 at 10a7d921.

Regime: CPU over fakes, nothing served, ``ScriptedClock``, thinking on, tools live, loop on, 1280x800 frames; n = 1,
deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra import config
from local_astra.conductor import prompt_builder as pb
from local_astra.config import load
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0174_cli_guidance as tc
from tests.unit import test_t0191_cli_guidance as g191
from tests.unit.test_cli_t0131 import MODE_FLAGS
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out

FLAG = "--live-guidance"
FIELD = "live_guidance"
PREFIX = "astra session: live guidance: "
LIVE_PROFILE = tc.LIVE_PROFILE
NO_WEB_KEY = "browser_web = false\n"
WAIT = "--wait-tool"
#: B1, typed from the task (never imported)
S_SHARED = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: "
    "to do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)
S_SOURCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)
#: B's BASE: main 175a29b0 at the branch point of the B tests (T-0211 planning 81b2a5cb is docs only). A literal.
BASE = "175a29b0"
#: The implementation tip: B's last product commit, closed by the test-author at activation (T-0094, B-R3). It is
#: 1d90e374 (the B-R7 replay script fix), which follows the implementation commit 10a7d921.
TIP = "1d90e374bf488885892ca555d0c92846d86ff967"
#: B-R3: B's branches merged main mid-way (f9c64be6, 1b262f44), so the frozen range starts at
#: ``git merge-base main 1d90e374`` (main 98c75e58, 2026-10-30; the same as for 10a7d921), not at BASE. A literal.
RANGE_BASE = "98c75e58ea1c46a06c131f723d8e825379ae6ce1"
#: B's first branch commit (the B6 premise rows), made before the mid-way merges of main
B_FIRST = "0f286491bf494435e694aa67e587713fcb9ef6ab"

WEB_STATES: dict[str, tuple[list[str], str]] = {
    "web": ([], ""),
    "no-web-flag": (["--no-web"], ""),
    "no-web-profile": ([], NO_WEB_KEY),
}
SOURCES = {
    "flag": lambda v: ([FLAG, v], ""),
    "profile": lambda v: ([], f'{FIELD} = "{v}"\n'),
}


def require_v4() -> str:
    """The named failure at BASE: v4 is not a version and its text does not exist."""
    versions = getattr(config, "LIVE_GUIDANCE_VERSIONS", ())
    assert "v4" in versions, (
        f"T-0211 B3: 'v4' in config.LIVE_GUIDANCE_VERSIONS does not exist yet; {versions!r}"
    )
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V4", None)
    assert isinstance(value, str), (
        "T-0211 B1: prompt_builder.LIVE_TOOL_GUIDANCE_V4 does not exist yet"
    )
    return value


def case_args(source: str, version: str, web: str) -> tuple[list[str], str]:
    flags, body = SOURCES[source](version)
    wflags, wbody = WEB_STATES[web]
    return [*flags, *wflags], body + wbody


@contextlib.contextmanager
def load_refusal_is_assertion() -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        raise AssertionError(
            f"T-0211 B3: the profile's live_guidance value was refused at load: {str(exc)[:300]}"
        ) from None


def built(
    tmp: Path, capsys: Any, monkeypatch: Any, flags: list[str], body: str, wait: str
) -> dict[str, Any]:
    """The ``Session`` keywords ``cli`` passed for a live profile with ``body`` and ``flags`` plus ``--wait-tool
    <wait>`` (nothing served); ``__system__`` is the built session's system text."""
    with load_refusal_is_assertion():
        calls = build_mode(
            tmp, capsys, monkeypatch, "off", *flags, WAIT, wait, body=LIVE_PROFILE + body
        )
    kw = calls.session_kwargs[-1]
    kw["__system__"] = calls.graphs[-1].session.prompts.system_message().text()
    monkeypatch.undo()
    return kw


def expect(text: str, wait: str) -> str:
    return pb.with_wait_guidance(text) if wait == "on" else text


def assert_guidance(kw: dict[str, Any], obj: str, wait: str, where: str) -> None:
    """``tool_guidance`` is ``obj`` itself (wait off) or ``with_wait_guidance(obj)`` (wait on); the system text ends
    with it once."""
    got = kw.get("tool_guidance")
    want = expect(obj, wait)
    if wait == "off":
        assert got is obj, (
            f"{where}: tool_guidance is the mapping's object itself; {str(got)[-160:]!r}"
        )
    else:
        assert got == want, (
            f"{where}: tool_guidance is with_wait_guidance(...); {str(got)[-200:]!r}"
        )
    text = kw["__system__"]
    assert text.endswith("\n" + want) and text.count(want) == 1, f"{where}: {text[-300:]!r}"


# =====================================================================================================
# B3: config
# =====================================================================================================


def test_config_v4(tmp_path: Path) -> None:
    """B3: the versions tuple is ``("v1", "v2", "v3", "v4")``; the default stays ``"v3"``; ``live_guidance = "v4"``
    loads with ``tools = "live"`` and is refused with ``tools`` off, dry_run or absent; an unknown value is refused
    naming all four."""
    require_v4()
    versions = config.LIVE_GUIDANCE_VERSIONS
    assert versions == ("v1", "v2", "v3", "v4") and isinstance(versions, tuple), versions
    assert config.DEFAULT_LIVE_GUIDANCE == "v3", config.DEFAULT_LIVE_GUIDANCE
    d = tmp_path / "ok"
    d.mkdir()
    got = getattr(load(tc._toml(d, f'{LIVE_PROFILE}{FIELD} = "v4"\n')).active(), FIELD)
    assert got == "v4", f"T-0211 B3: live_guidance = 'v4' loads and reads back; {got!r}"
    for tools in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        msg = tc._load_error(tmp_path / f"o{len(tools)}", f'{tools}{FIELD} = "v4"\n')
        assert msg is not None, f"T-0211 B3: v4 with {tools.strip() or 'no tools key'} loaded"
    for n, junk in enumerate(('"v5"', '"V4"', '"v4 "', '"4"', "4")):
        msg = tc._load_error(tmp_path / f"j{n}", f"{LIVE_PROFILE}{FIELD} = {junk}\n")
        assert msg is not None and all(v in msg for v in ("v1", "v2", "v3", "v4")), (
            f"T-0211 B3: {FIELD} = {junk} is refused naming v1, v2, v3 and v4; got {msg!r}"
        )


@pytest.mark.parametrize("junk", ["v5", "V4", "v4.0"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """B3/B4 (active; too-wide companion of the new choice): only the listed versions parse."""
    tc.T.patch_detok_load(monkeypatch, tc.T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = tc.run_cli([*argv(tmp_path, *MODE_FLAGS["live"], FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


# =====================================================================================================
# B4: which text the session carries
# =====================================================================================================


@pytest.mark.parametrize("wait", ["on", "off"])
@pytest.mark.parametrize("source", list(SOURCES))
def test_v4_text_web_live(tmp_path, capsys, monkeypatch, source, wait):
    """B4: v4 in web mode (flag or profile) gets ``LIVE_GUIDANCE["v4"]`` itself (wait off) or
    ``with_wait_guidance(LIVE_GUIDANCE["v4"])`` (wait on); the system text ends with it once and holds both
    sentences once."""
    v4 = require_v4()
    assert pb.LIVE_GUIDANCE.get("v4") is v4, (
        "T-0211 B3: LIVE_GUIDANCE['v4'] is LIVE_TOOL_GUIDANCE_V4"
    )
    flags, body = case_args(source, "v4", "web")
    kw = built(tmp_path, capsys, monkeypatch, flags, body, wait)
    assert_guidance(kw, v4, wait, f"T-0211 B4 ({source}, web, wait {wait})")
    for s in (S_SHARED, S_SOURCE):
        assert kw["__system__"].count(s) == 1, f"T-0211 B4: {s[:40]!r} once in the system text"


@pytest.mark.parametrize("wait", ["on", "off"])
@pytest.mark.parametrize("web", ["no-web-flag", "no-web-profile"])
@pytest.mark.parametrize("source", list(SOURCES))
def test_v4_no_web_is_v2_object(tmp_path, capsys, monkeypatch, source, web, wait):
    """B4 (adversarial: the sentences promise Ctrl+L and opening result pages, which ``--no-web`` refuses): v4 with
    ``--no-web`` or ``browser_web = false`` carries ``LIVE_GUIDANCE["v2"]`` itself (``is``, not an equal copy) with
    the wait tool off, and ``with_wait_guidance(v2)`` with it on; neither sentence is in the system text."""
    require_v4()
    flags, body = case_args(source, "v4", web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body, wait)
    assert_guidance(kw, pb.LIVE_GUIDANCE["v2"], wait, f"T-0211 B4 ({source}, {web}, wait {wait})")
    for s in (S_SHARED, S_SOURCE, pb.LIVE_GUIDANCE["v3"]):
        assert s not in kw["__system__"], f"T-0211 B4: {s[:40]!r} under {web}"


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_v1_v2_v3_selection_unchanged(tmp_path, capsys, monkeypatch, version, web):
    """B4/B8 (active): v1 and v2 carry ``LIVE_GUIDANCE[v]`` in both web states, and v3 carries v3 in web mode and
    v2's object without it (T-0191 A4), wait off; unchanged by T-0211."""
    flags, body = case_args("flag", version, web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body, "off")
    obj = pb.LIVE_GUIDANCE["v2" if version == "v3" and web != "web" else version]
    assert_guidance(kw, obj, "off", f"B8 ({version}, {web})")
    for s in (S_SHARED, S_SOURCE):
        assert s not in kw["__system__"]


@pytest.mark.parametrize("web", list(WEB_STATES))
def test_default_still_v3(tmp_path, capsys, monkeypatch, web):
    """B3/B4 (active; removed in D): with no flag and no key the resolved version is v3: web mode carries
    ``LIVE_GUIDANCE["v3"]`` (plus the wait sentence, on by default), ``--no-web`` / ``browser_web = false`` v2's
    object (plus the wait sentence); neither v4 sentence anywhere; the default source prints no guidance line."""
    assert config.DEFAULT_LIVE_GUIDANCE == "v3", config.DEFAULT_LIVE_GUIDANCE
    flags, body = WEB_STATES[web]
    for wait in ("on", "off"):
        kw = built(tmp_path / f"build-{wait}", capsys, monkeypatch, flags, body, wait)
        obj = pb.LIVE_GUIDANCE["v3" if web == "web" else "v2"]
        assert_guidance(kw, obj, wait, f"T-0211 B3 default ({web}, wait {wait})")
        for s in (S_SHARED, S_SOURCE):
            assert s not in kw["__system__"]
        assert kw.get("page_fact") is True
    out, err = tc.dry_out_err(
        tmp_path / "dry", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body
    )
    assert "live guidance" not in out + err, f"the default stays silent\n{out}{err}"


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_v4_refused_outside_live(tmp_path, capsys, monkeypatch, mode):
    """B4: ``--live-guidance v4`` with ``--tools off`` or ``--tools dry-run`` (plain, and over a live profile) exits
    2 with cli's own refusal naming the flag, before any graph, page server or browser; ``live_guidance = "v4"`` in a
    live profile under the same override is refused the same way; the same profile with no override builds a live
    session (too-wide companion)."""
    v4 = require_v4()
    tc._refused(tmp_path / "plain", capsys, monkeypatch, [*MODE_FLAGS[mode], FLAG, "v4"], "")
    spelled = {"off": "off", "dry_run": "dry-run"}[mode]
    tc._refused(
        tmp_path / "over", capsys, monkeypatch, ["--tools", spelled, FLAG, "v4"], LIVE_PROFILE
    )
    tc._refused(
        tmp_path / "key",
        capsys,
        monkeypatch,
        ["--tools", spelled],
        f'{LIVE_PROFILE}{FIELD} = "v4"\n',
    )
    kw = built(tmp_path / "ok", capsys, monkeypatch, [], f'{FIELD} = "v4"\n', "off")
    assert kw.get("tool_guidance") is v4


# =====================================================================================================
# B4: page_fact
# =====================================================================================================


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize(
    "version",
    ["v2", "v3", "v4"],
)
def test_page_fact_v2_v3_v4(tmp_path, capsys, monkeypatch, version, web):
    """B4: ``page_fact=True`` for v2, v3 and v4 (flag) in both web states (v4 under ``--no-web`` is still v4); the v2
    and v3 rows are active, the v4 row pending. v1 passes no keyword (``test_t0191_cli_guidance``)."""
    if version == "v4":
        require_v4()
    flags, body = case_args("flag", version, web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body, "off")
    assert kw.get("page_fact") is True, (
        f"T-0211 B4 ({version}, {web}): page_fact=True; got {kw.get('page_fact', '<absent>')!r}"
    )


@pytest.mark.parametrize("source", list(SOURCES))
def test_page_fact_v4_profile(tmp_path, capsys, monkeypatch, source):
    """B4: ``page_fact=True`` for a profile (and flag) v4 source in web mode and under ``browser_web = false``."""
    require_v4()
    for web in ("web", "no-web-profile"):
        flags, body = case_args(source, "v4", web)
        kw = built(tmp_path / web, capsys, monkeypatch, flags, body, "on")
        assert kw.get("page_fact") is True, (source, web, kw.get("page_fact", "<absent>"))


# =====================================================================================================
# B4: the startup line
# =====================================================================================================


@pytest.mark.parametrize("web", ["web", "no-web-flag"])
@pytest.mark.parametrize("source", list(SOURCES))
def test_startup_line_v4(tmp_path, capsys, monkeypatch, source, web):
    """B4: a flag or profile v4 source prints exactly ``astra session: live guidance: v4 (<source>)`` once, on
    stdout, immediately before the reask line, under ``--dry-run`` and at a real start; with ``--no-web`` the same
    line; nothing else on either stream differs from the v3 run of the same source but its version word."""
    require_v4()
    flags, body = case_args(source, "v4", web)
    want = f"{PREFIX}v4 ({source})"
    with load_refusal_is_assertion():
        out, err = tc.dry_out_err(
            tmp_path / "dry", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body
        )
    assert tc.guidance_lines(err) == [], err
    tc.assert_line_before_reask(out, want, "dry")
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "start",
        capsys,
        monkeypatch,
        *MODE_FLAGS["off"],
        *flags,
        body=LIVE_PROFILE + body,
    )
    assert code == 0, started
    tc.assert_line_before_reask(started, want, "start")
    monkeypatch.undo()
    f3, b3 = case_args(source, "v3", web)
    out3, err3 = tc.dry_out_err(tmp_path / "dry3", capsys, monkeypatch, *f3, body=LIVE_PROFILE + b3)

    def norm(text: str, where: Path) -> list[str]:
        return [
            x.replace(str(where), "<tmp>") for x in text.splitlines() if not x.startswith(PREFIX)
        ]

    assert norm(out, tmp_path / "dry") == norm(out3, tmp_path / "dry3"), (
        "T-0211 B4: v4's dry-run stdout differs from v3's beyond the guidance line"
    )
    assert norm(err, tmp_path / "dry") == norm(err3, tmp_path / "dry3"), err


# =====================================================================================================
# B4 adversarial: under --no-web neither sentence goes anywhere
# =====================================================================================================


def heads() -> list[str]:
    return [S_SHARED[:40], S_SOURCE[:40]]


def test_no_web_never_leaks_sentences(tmp_path: Path) -> None:
    """B4 adversarial: ``--live-guidance v4 --no-web`` (and ``browser_web = false``) through cli, two turns, three
    tool steps with a refused Ctrl+L: the first 40 characters of S_SHARED and of S_SOURCE appear in no request body, no
    session-log record and no turn line, and every body carries v2's text. Power: the same rig in web mode carries
    both in every body and in the log."""
    require_v4()
    v2 = pb.LIVE_GUIDANCE["v2"]
    for h in heads():
        assert len(h) == 40 and h not in v2 and h not in pb.LIVE_GUIDANCE["v3"], (
            f"premise: {h!r} is distinctive"
        )
    web_kw, web_echo = g191.cli_kwargs(tmp_path / "cli-web", [FLAG, "v4"])
    on = asyncio.run(g191.refused_ctrl_l_turns(tmp_path / "rig-web", web_kw, web_echo))
    assert on["turns"] == 2 and len(on["bodies"]) >= 5, (on["turns"], len(on["bodies"]))
    for h in heads():
        enc = json.dumps(h)[1:-1]
        assert all(enc in json.dumps(b) for b in on["bodies"]), f"power: web mode carries {h!r}"
        assert "session log" in g191.found_in(h, on), f"power: web mode's log carries {h!r}"
    for label, flags, body in (
        ("flag", [FLAG, "v4", "--no-web"], ""),
        ("profile", [FLAG, "v4"], NO_WEB_KEY),
    ):
        kw, echo = g191.cli_kwargs(tmp_path / f"cli-{label}", flags, body)
        got = asyncio.run(g191.refused_ctrl_l_turns(tmp_path / f"rig-{label}", kw, echo))
        assert got["turns"] == 2 and len(got["bodies"]) >= 5, (
            f"premise ({label}): two turns, 4 + 1 generates; {got['turns']}, {len(got['bodies'])}"
        )
        names = [n for n, _ in got["invocations"] if n != "page_overview"]
        assert names[:3] == ["click_at", "press_key", "click_at"], f"premise ({label}): {names}"
        assert "not available in the browser" in got["log"], f"premise ({label}): Ctrl+L refused"
        for h in heads():
            where = g191.found_in(h, got)
            assert where == [], f"T-0211 B4 ({label}): {h!r} leaked into {where}"
        assert all(json.dumps(v2)[1:-1] in json.dumps(b) for b in got["bodies"]), (
            f"T-0211 B4 ({label}): every body carries v2's text"
        )


# =====================================================================================================
# B8: no flag and v1-v3 bytes equal BASE, in both web states (active)
# =====================================================================================================

#: case -> (flags, profile body after T-0152's ``LIVE_PROFILE``); live defaults as at BASE, nothing pinned
CASES: dict[str, tuple[list[str], str]] = {
    "default-web": ([], ""),
    "default-no-web": (["--no-web"], ""),
    "default-no-web-key": ([], NO_WEB_KEY),
    "flag-v1-web": ([FLAG, "v1"], ""),
    "flag-v1-no-web": ([FLAG, "v1", "--no-web"], ""),
    "flag-v2-web": ([FLAG, "v2"], ""),
    "flag-v2-no-web": ([FLAG, "v2", "--no-web"], ""),
    "flag-v3-web": ([FLAG, "v3"], ""),
    "flag-v3-no-web": ([FLAG, "v3", "--no-web"], ""),
    "flag-v3-web-wait-off": ([FLAG, "v3", WAIT, "off"], ""),
    "profile-v3-web": ([], f'{FIELD} = "v3"\n'),
    "profile-v3-no-web-key": ([], f'{FIELD} = "v3"\n' + NO_WEB_KEY),
}


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    from tests.unit import test_t0134_differential as t0134
    from tests.unit import test_t0134c_live_feedback_default as t0134c
    from tests.unit import test_t0152_overview_frame_default as t0152

    out: dict[str, Any] = {"cli": {}}
    runs: dict[str, dict[str, Any]] = {}
    for case, (flags, body) in CASES.items():
        dry = t0152._cli_run(tmp / "cli" / case / "dry", flags, body, dry=True)
        start = t0152._cli_run(tmp / "cli" / case / "start", flags, body, dry=False)
        runs[case] = start
        out["cli"][case] = {"dry": t0152._public(dry), "start": t0152._public(start)}
    out["rigs"] = asyncio.run(g191._rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    out["executor"] = asyncio.run(t0134._executor_actions())
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0211_cli_guidance as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture: ``local_astra`` from ``git archive BASE`` (T-0174's harness), origin checked."""
    gr.require_commits(BASE)
    tmp = tmp_path_factory.mktemp("t0211_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
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
    return _capture(tmp_path_factory.mktemp("t0211_now"))


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0211 B8: differs from BASE {BASE}: {what}:\nwant {json.dumps(want)[:1500]}\n"
        f"got  {json.dumps(got)[:1500]}"
    )


@pytest.mark.parametrize("case", list(CASES))
def test_v1_v2_v3_bytes_equal_base(base: dict[str, Any], now: dict[str, Any], case: str) -> None:
    """B8 (active): with no flag, ``v1``, ``v2`` and ``v3``, in web mode and without it, cli's output and keywords
    (``--dry-run`` and a real start) and the rig sessions built from those keywords equal BASE's byte for byte
    (module docstring); nothing is admitted."""
    for kind in ("dry", "start"):
        want, got = base["cli"][case][kind], now["cli"][case][kind]
        assert want["code"] == 0, f"premise: BASE's {case} {kind} exits 0\n{want['err'][-600:]}"
        _same(want, got, f"cli {case} {kind}")
    want_rig, got_rig = base["rigs"][case], now["rigs"][case]
    assert want_rig is not None and got_rig is not None, f"premise: rigs built ({case})"
    assert sorted(want_rig) == sorted(got_rig), (sorted(want_rig), sorted(got_rig))
    for key in want_rig:
        if key in g191.PAGES:
            for section in want_rig[key]:
                _same(want_rig[key][section], got_rig[key][section], f"rig {case} {key} {section}")
        else:
            _same(want_rig[key], got_rig[key], f"rig {case} {key}")


def test_other_modes_and_executor_equal_base(base: dict[str, Any], now: dict[str, Any]) -> None:
    """B8 (active): tools off and dry-run through cli, and the executor's action strings and ``resolved`` over the
    fake driver, equal BASE's."""
    for case in base["other"]:
        for key in base["other"][case]:
            _same(base["other"][case][key], now["other"][case][key], f"other {case} {key}")
    _same(base["executor"], now["executor"], "executor actions")


def test_capture_sees_the_cases(now: dict[str, Any]) -> None:
    """Power premise (active): the capture distinguishes the cases it compares (web vs ``--no-web``, v3 vs v2, wait
    on vs off), so the BASE comparison covers distinct configurations."""
    cli = now["cli"]
    assert cli["flag-v3-web"]["dry"]["out"] != cli["flag-v3-no-web"]["dry"]["out"]
    assert cli["flag-v3-web"]["start"] != cli["flag-v2-web"]["start"]
    assert cli["flag-v3-web"]["start"] != cli["flag-v3-web-wait-off"]["start"]
    flat = json.dumps(now["rigs"]["flag-v3-web"])
    assert json.dumps(pb.LIVE_GUIDANCE["v3"])[1:-1] in flat, "premise: the v3 rig carries v3"


# =====================================================================================================
# B8: frozen paths and the §4 invariants (active)
# =====================================================================================================

#: B8's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/toolcall",
    "local_astra/computer",
    "local_astra/pages",
    "local_astra/contracts",
    "tests/fakes",
    "scripts/gpu",
    "scripts/locked",
    "tests/unit/test_turn_invariants.py",
)


def _blobs_at(rev: str) -> dict[str, str]:
    proc = gr._git("ls-tree", "-r", rev, "--", *FROZEN)
    assert proc.returncode == 0, proc.stderr
    out = {}
    for line in proc.stdout.splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def _range_premise() -> None:
    """B-R3 premise: BASE <= RANGE_BASE <= TIP, and B's first test commit lies in ``RANGE_BASE..TIP`` (the range
    still holds all of B's branch)."""
    gr.require_commits(BASE, RANGE_BASE, B_FIRST, TIP)
    assert gr.is_ancestor(BASE, RANGE_BASE), f"{BASE} is not an ancestor of {RANGE_BASE[:8]}"
    assert gr.is_ancestor(RANGE_BASE, TIP), f"{RANGE_BASE[:8]} is not an ancestor of {TIP[:8]}"
    assert gr.is_ancestor(B_FIRST, TIP) and not gr.is_ancestor(B_FIRST, RANGE_BASE), (
        f"premise: {B_FIRST[:8]} lies in {RANGE_BASE[:8]}..{TIP[:8]}"
    )


def test_frozen_paths_unchanged() -> None:
    """B8: every frozen path (``session.py``, ``turn_machine.py``, ``toolcall/**`` with ``loop.py`` and
    ``tools_v1*.json``, ``computer/**``, ``pages/**``, ``contracts/**``, ``tests/fakes/**``, ``scripts/gpu/*``,
    ``scripts/locked/*``) is byte-identical over ``RANGE_BASE..TIP`` (two literals; B-R3: ``RANGE_BASE`` is
    ``merge-base(main, TIP)`` because B's branches merged main mid-way)."""
    _range_premise()
    at_base = _blobs_at(RANGE_BASE)
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in at_base), (
            f"premise: {p} at {RANGE_BASE[:8]}"
        )
    assert (
        "local_astra/toolcall/loop.py" in at_base
        and "local_astra/toolcall/tools_v1.json" in at_base
    )
    at_tip = _blobs_at(TIP)
    changed = sorted(p for p in set(at_base) | set(at_tip) if at_base.get(p) != at_tip.get(p))
    assert changed == [], (
        f"T-0211 B8: frozen paths differ in {RANGE_BASE[:8]}..{TIP[:8]}: {changed}"
    )


def test_section4_invariants_unedited() -> None:
    """B8: docs/02 §4 is byte-identical over ``RANGE_BASE..TIP`` (B-R3)."""
    _range_premise()
    proc = gr._git("show", f"{RANGE_BASE}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    at_base = t7d.section(proc.stdout, "## 4. ")
    tip = gr._git("show", f"{TIP}:docs/02-architecture.md")
    assert tip.returncode == 0, tip.stderr
    assert at_base.strip() and "invariant" in at_base.lower(), "premise: §4 found at RANGE_BASE"
    assert t7d.section(tip.stdout, "## 4. ") == at_base, (
        f"T-0211 B8: docs/02 §4 changed in {RANGE_BASE[:8]}..{TIP[:8]}"
    )
