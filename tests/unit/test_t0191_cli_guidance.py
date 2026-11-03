"""T-0191 group A, A3 (config), A4 (selection by web mode, through the real ``cli`` builder over fakes) and A7 (nothing
else moves) (docs/tasks/T-0191.md).

**The interfaces these rows pin** (the task's; readings stated where it is silent):

- A3: ``config.LIVE_GUIDANCE_VERSIONS == ("v1", "v2", "v3")``; ``config.DEFAULT_LIVE_GUIDANCE == "v2"`` in group A,
  ``"v3"`` since group C (C1, James's ruling 2026-10-28 after the B smoke PASS);
  ``Profile.live_guidance`` loads ``"v3"`` with ``tools = "live"`` and refuses it elsewhere; an unknown value is
  refused naming v1, v2 and v3.
- A4: a live session built by ``cli`` gets ``tool_guidance``:
  - ``LIVE_GUIDANCE["v3"]`` for v3 (``--live-guidance v3`` or ``live_guidance = "v3"``) in web mode;
  - ``LIVE_GUIDANCE["v2"]`` itself (``is``, not an equal copy) for v3 with ``--no-web`` or ``browser_web = false``;
  - ``LIVE_GUIDANCE[v]`` for v1 and v2 in both web states (unchanged).
  ``--live-guidance v3`` with ``--tools off|dry-run`` exits 2 with ``cli``'s own refusal before any graph, page server
  or browser (T-0174 A4's ``_refused``). ``page_fact=True`` iff the resolved version is v2 or v3 (in both web states:
  **reading**, the version under ``--no-web`` is still v3 even though its text is v2's); v1 passes no keyword at all.
- A4 startup line: a flag or profile v3 source prints exactly ``astra session: live guidance: v3 (<source>)`` once,
  immediately before T-0172's reask line (T-0174 A4's place). **Reading, reported** (Q2 at its recommendation,
  "silently carry v2's text"): with ``--no-web`` the line is the same ``v3 (<source>)`` line, and no other line is
  added; the default source prints nothing.
- A4 adversarial (``test_no_web_never_leaks_sentence``): a cli-built live v3 ``--no-web`` session over the fakes, two
  turns, the first with three tool steps one of which is a refused ``Ctrl+L``: S (v3's added sentence) and its two
  halves appear in no request body, no session-log record and no turn line; the same rig in web mode carries S in
  every body (power).
- A7: the differential ``test_v1_v2_bytes_equal_base`` runs ``_capture`` here and in a subprocess over
  ``git archive BASE`` (``local_astra`` from BASE, origin checked; T-0174's harness): with no flag, ``v1`` and ``v2``
  (flag and profile), in web mode and with ``--no-web`` / ``browser_web = false``, ``cli``'s exit, stdout, stderr and
  every ``Session``/executor keyword under ``--dry-run`` and at a real start; the rig sessions built from those
  keywords (machine, HUD, prompts, request bodies, session-log records, turn lines, history, browser invocations;
  T-0152's ``_rig_turn`` on a fitting and a long page) and T-0174 A5's two-turn bodies; tools off and dry-run; the
  executor's actions over T-0111c's fake driver. All byte for byte, nothing admitted.
- A7 frozen paths: byte-identical at BASE and at ``TIP`` (closed to the implementation tip ea2ab9b6 before merge,
  T-0094; ``None`` would compare the working tree), and docs/02 §4 unedited.

**Group C (the default flip, C1/C3, re-scoped by the test-author 2026-10-28):** ``test_config_v3`` expects the
default ``"v3"``; ``test_default_still_v2`` is removed and replaced by ``test_default_is_v3`` (web mode: v3; with
``--no-web`` / ``browser_web = false``: v2's object itself; the default source silent); in
``test_v1_v2_bytes_equal_base[default-web]`` BASE's side moves only the guidance text (``v2_moved_to_v3``: every
occurrence of v2's bytes becomes v3's, byte for byte otherwise); ``test_explicit_v2_is_old_default`` pins C1's
"``--live-guidance v2`` reproduces the pre-flip default bytes" (BASE's ``default-web`` against today's
``flag-v2-web``, the one A4 line ``live guidance: v2 (flag)`` admitted). The ``default-no-web*`` rows stay as they
were (the default carries v2's bytes there).

BASE is 40466d09 (main after T-0181 A, the merge-base of the A branch after main was merged at 769c0d0a; it holds
T-0191's planning commit f1caef6a, docs only, and no T-0191 code; it was f1caef6a, which predates T-0181 A). ``test_default_still_v2`` (removed in C),
``test_v1_v2_selection_both_web_states``, ``test_page_fact_v1_v2_both_web_states``, ``test_flag_junk_still_exits_2``,
``test_v1_v2_bytes_equal_base``, ``test_frozen_paths_unchanged`` and ``test_section4_invariants_unedited`` hold at
BASE. Every other row was written ahead as pending and activated at A's ``/implement`` (2026-10-28); at BASE each fails on a named
``AssertionError`` (``--live-guidance v3`` is an argparse ``invalid choice``, ``live_guidance = "v3"`` is refused at
load, ``LIVE_TOOL_GUIDANCE_V3`` does not exist).

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
from local_astra.config import Profile, load
from tests import _git_range as gr
from tests.unit import _t0202_pins as t0202
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0174_cli_guidance as tc
from tests.unit.test_cli_t0131 import MODE_FLAGS
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out

FLAG = "--live-guidance"
FIELD = "live_guidance"
PREFIX = "astra session: live guidance: "
LIVE_PROFILE = tc.LIVE_PROFILE
NO_WEB_KEY = "browser_web = false\n"
#: A's BASE: 40466d09, main after T-0181 A merged (it changed session.py, protocol.py, the fakes, and cli's
#: ``mic_unavailable=`` for --silent) and the A branch's merge-base; it contains T-0191's planning commit f1caef6a
#: (docs only) and no T-0191 code. Was f1caef6a until main 40466d09 was merged at 769c0d0a. A literal.
BASE = "40466d09"
#: The implementation tip: A's last non-test commit, closed by the test-author before merge (T-0094).
TIP: str | None = "ea2ab9b6"

#: web state -> (extra flags, extra profile body)
WEB_STATES: dict[str, tuple[list[str], str]] = {
    "web": ([], ""),
    "no-web-flag": (["--no-web"], ""),
    "no-web-profile": ([], NO_WEB_KEY),
}
#: source -> (flags, profile body) selecting a version ``{v}``
SOURCES = {
    "flag": lambda v: ([FLAG, v], ""),
    "profile": lambda v: ([], f'{FIELD} = "{v}"\n'),
}


def guidance_map() -> Any:
    return pb.LIVE_GUIDANCE


def v3_text() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V3", None)
    assert isinstance(value, str), (
        "T-0191 A1: prompt_builder.LIVE_TOOL_GUIDANCE_V3 does not exist yet"
    )
    return value


def sentence() -> str:
    v3, v2 = v3_text(), pb.LIVE_TOOL_GUIDANCE_V2
    assert v3.startswith(v2 + " ") and len(v3) > len(v2) + 1, (
        f"T-0191 A1: v3 is v2 + ' ' + S; v3 is {v3!r}"
    )
    return v3[len(v2) + 1 :]


def case_args(source: str, version: str, web: str) -> tuple[list[str], str]:
    flags, body = SOURCES[source](version)
    wflags, wbody = WEB_STATES[web]
    return [*flags, *wflags], body + wbody


@contextlib.contextmanager
def load_refusal_is_assertion() -> Iterator[None]:
    """At BASE ``live_guidance = "v3"`` raises a pydantic ``ValidationError`` (a ``ValueError``) out of ``cli.main``'s
    ``load``; a pending row must fail on a named ``AssertionError`` instead."""
    try:
        yield
    except ValueError as exc:
        raise AssertionError(
            f"T-0191 A3: the profile's live_guidance value was refused at load: {str(exc)[:300]}"
        ) from None


def built(tmp: Path, capsys: Any, monkeypatch: Any, flags: list[str], body: str) -> dict[str, Any]:
    """The ``Session`` keywords ``cli`` passed for a live profile with ``body`` and ``flags`` (nothing served).

    T-0202 D3 re-pin: the live build pins ``--wait-tool off``, so ``tool_guidance`` is the selected ``LIVE_GUIDANCE``
    object itself (the ``is`` rows) whatever the wait-tool default is; the default's guidance with the wait sentence
    is ``test_t0202_cli_default.test_live_default_on``'s (v3 web, v2 under ``--no-web``)."""
    flags = t0202.pin_off(flags, body)
    with load_refusal_is_assertion():
        calls = build_mode(tmp, capsys, monkeypatch, "off", *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    kw["__system__"] = calls.graphs[-1].session.prompts.system_message().text()
    monkeypatch.undo()
    return kw


# =====================================================================================================
# A3: config
# =====================================================================================================


def test_config_v3(tmp_path: Path) -> None:
    """A3: the versions tuple is ``("v1", "v2", "v3")``; the default is ``"v3"`` (C1; it stayed ``"v2"`` in group
    A); ``live_guidance = "v3"`` loads
    with ``tools = "live"`` and is refused with ``tools`` off, dry_run or absent; an unknown value is refused naming
    v1, v2 and v3. Re-scoped by T-0211 B3 (pending for T-0211): the tuple gains ``"v4"``, the junk ``"v4"`` becomes
    ``"v5"`` and the refusal names all four."""
    versions = getattr(config, "LIVE_GUIDANCE_VERSIONS", None)
    assert versions == ("v1", "v2", "v3", "v4") and isinstance(versions, tuple), (
        f"T-0211 B3: config.LIVE_GUIDANCE_VERSIONS = ('v1', 'v2', 'v3', 'v4') does not exist yet; {versions!r}"
    )
    assert config.DEFAULT_LIVE_GUIDANCE == "v3", (
        f"T-0191 C1: config.DEFAULT_LIVE_GUIDANCE = 'v3'; {config.DEFAULT_LIVE_GUIDANCE!r}"
    )
    d = tmp_path / "ok"
    d.mkdir()
    got = getattr(load(tc._toml(d, f'{LIVE_PROFILE}{FIELD} = "v3"\n')).active(), FIELD)
    assert got == "v3", f"T-0191 A3: live_guidance = 'v3' loads and reads back; {got!r}"
    for tools in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        msg = tc._load_error(tmp_path / f"o{len(tools)}", f'{tools}{FIELD} = "v3"\n')
        assert msg is not None, f"T-0191 A3: v3 with {tools.strip() or 'no tools key'} loaded"
    for n, junk in enumerate(('"v5"', '"V3"', '"v3 "', '"3"', "3")):
        msg = tc._load_error(tmp_path / f"j{n}", f"{LIVE_PROFILE}{FIELD} = {junk}\n")
        assert msg is not None and all(v in msg for v in ("v1", "v2", "v3", "v4")), (
            f"T-0211 B3: {FIELD} = {junk} is refused naming v1, v2, v3 and v4; got {msg!r}"
        )


@pytest.mark.parametrize("junk", ["v5", "V3", "v3.0"])
def test_flag_junk_still_exits_2(tmp_path, capsys, monkeypatch, junk):
    """A3/A4 (active; too-wide companion of the new choice): only the listed versions parse. Re-scoped by T-0211 B3:
    the junk ``"v4"`` (a version now) becomes ``"v5"``."""
    tc.T.patch_detok_load(monkeypatch, tc.T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = tc.run_cli([*argv(tmp_path, *MODE_FLAGS["live"], FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


# =====================================================================================================
# A4: which text the session carries
# =====================================================================================================


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize("source", list(SOURCES))
def test_v3_text_only_in_web_live(tmp_path, capsys, monkeypatch, source, web):
    """A4 (adversarial: the sentence must not reach a browser that refuses Ctrl+L): v3 in web mode gets
    ``LIVE_GUIDANCE["v3"]`` and its system message ends with it once; v3 with ``--no-web`` or ``browser_web = false``
    gets ``LIVE_GUIDANCE["v2"]`` itself (``is``), and S is nowhere in its system message."""
    flags, body = case_args(source, "v3", web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body)
    m = guidance_map()
    v3 = v3_text()
    assert "v3" in m and m["v3"] is v3, "T-0191 A3: LIVE_GUIDANCE['v3'] is LIVE_TOOL_GUIDANCE_V3"
    text = kw["__system__"]
    if web == "web":
        assert kw.get("tool_guidance") is m["v3"], (
            f"T-0191 A4 ({source}, {web}): Session(tool_guidance=LIVE_GUIDANCE['v3']); got "
            f"{str(kw.get('tool_guidance'))[-120:]!r}"
        )
        assert text.endswith("\n" + v3) and text.count(v3) == 1, text[-300:]
    else:
        assert kw.get("tool_guidance") is m["v2"], (
            f"T-0191 A4 ({source}, {web}): v3 without web mode carries LIVE_GUIDANCE['v2'] itself; got "
            f"{str(kw.get('tool_guidance'))[-120:]!r}"
        )
        assert text.endswith("\n" + m["v2"]) and sentence() not in text, text[-300:]


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_v1_v2_selection_both_web_states(tmp_path, capsys, monkeypatch, version, web):
    """A4 (active): v1 and v2 (flag) carry ``LIVE_GUIDANCE[v]`` in both web states, unchanged."""
    flags, body = case_args("flag", version, web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body)
    assert kw.get("tool_guidance") is guidance_map()[version], (version, web)
    assert kw["__system__"].endswith("\n" + guidance_map()[version])


@pytest.mark.parametrize("web", list(WEB_STATES))
def test_default_is_v3(tmp_path, capsys, monkeypatch, web):
    """C1 (replaces A's ``test_default_still_v2``): with no flag and no key the resolved version is v3, so in web mode
    the session carries ``LIVE_GUIDANCE["v3"]`` (its system message ends with it once, v1 absent) and with
    ``--no-web`` or ``browser_web = false`` it carries ``LIVE_GUIDANCE["v2"]`` itself (``is``) with S nowhere in the
    system message; ``page_fact`` is ``True`` (v3, A4); the default source prints no ``live guidance`` line on
    either stream (dry-run). The no-web rows are the too-wide companion: a flip that leaked v3 into ``--no-web``
    fails them (each also asserts the default is ``"v3"``, so all three fail before the flip)."""
    m = guidance_map()
    v3 = v3_text()
    assert config.DEFAULT_LIVE_GUIDANCE == "v3", (
        f"T-0191 C1: config.DEFAULT_LIVE_GUIDANCE is {config.DEFAULT_LIVE_GUIDANCE!r}, not 'v3'"
    )
    flags, body = WEB_STATES[web]
    kw = built(tmp_path / "build", capsys, monkeypatch, flags, body)
    text = kw["__system__"]
    if web == "web":
        assert kw.get("tool_guidance") is m["v3"], (
            f"T-0191 C1 ({web}): the default session gets LIVE_GUIDANCE['v3']; got "
            f"{str(kw.get('tool_guidance'))[-120:]!r}"
        )
        assert text.endswith("\n" + v3) and text.count(v3) == 1, text[-300:]
    else:
        assert kw.get("tool_guidance") is m["v2"], (
            f"T-0191 C1 ({web}): without web mode the default carries LIVE_GUIDANCE['v2'] itself; got "
            f"{str(kw.get('tool_guidance'))[-120:]!r}"
        )
        assert text.endswith("\n" + m["v2"]) and sentence() not in text, text[-300:]
    assert pb.LIVE_TOOL_GUIDANCE not in text, f"T-0191 C1 ({web}): no v1 text in the default"
    assert kw.get("page_fact") is True, (
        f"T-0191 C1 ({web}): page_fact with v3; {kw.get('page_fact')!r}"
    )
    out, err = tc.dry_out_err(
        tmp_path / "dry", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body
    )
    assert "live guidance" not in out + err, (
        f"T-0191 C1 ({web}): the default stays silent\n{out}{err}"
    )


def _refused_v3(tmp_path: Path, capsys: Any, monkeypatch: Any, flags: list[str], body: str) -> None:
    tc._refused(tmp_path, capsys, monkeypatch, flags, body)


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_v3_refused_outside_live(tmp_path, capsys, monkeypatch, mode):
    """A4: ``--live-guidance v3`` with ``--tools off`` or ``--tools dry-run`` (plain, and over a live profile) exits 2
    with cli's own refusal naming the flag, before any graph, page server or browser; a live profile's
    ``live_guidance = "v3"`` under the same override is refused the same way; the same profile with no override
    builds a live session (too-wide companion)."""
    _refused_v3(tmp_path / "plain", capsys, monkeypatch, [*MODE_FLAGS[mode], FLAG, "v3"], "")
    spelled = {"off": "off", "dry_run": "dry-run"}[mode]
    _refused_v3(
        tmp_path / "over", capsys, monkeypatch, ["--tools", spelled, FLAG, "v3"], LIVE_PROFILE
    )
    body = f'{LIVE_PROFILE}{FIELD} = "v3"\n'
    _refused_v3(tmp_path / "key", capsys, monkeypatch, ["--tools", spelled], body)
    kw = built(tmp_path / "ok", capsys, monkeypatch, [], f'{FIELD} = "v3"\n')
    assert kw.get("tool_guidance") is guidance_map()["v3"]


# =====================================================================================================
# A4: page_fact rides with v2 and v3
# =====================================================================================================


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize("source", list(SOURCES))
def test_page_fact_v2_and_v3(tmp_path, capsys, monkeypatch, source, web):
    """A4 (amends T-0174 B2-1): a v3 session gets ``page_fact=True`` in both web states (the resolved version is v3
    whatever text it carries)."""
    flags, body = case_args(source, "v3", web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body)
    assert kw.get("page_fact") is True, (
        f"T-0191 A4 ({source}, {web}): page_fact=True with v3; got {kw.get('page_fact', '<absent>')!r}"
    )


@pytest.mark.parametrize("web", list(WEB_STATES))
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_page_fact_v1_v2_both_web_states(tmp_path, capsys, monkeypatch, version, web):
    """A4 (active): v2 gets ``page_fact=True``; v1 passes no ``page_fact`` keyword at all; in both web states."""
    flags, body = case_args("flag", version, web)
    kw = built(tmp_path, capsys, monkeypatch, flags, body)
    if version == "v2":
        assert kw.get("page_fact") is True, (version, web)
    else:
        assert "page_fact" not in kw, (
            f"T-0191 A4: v1 passes no page_fact keyword; {kw.get('page_fact')!r}"
        )


# =====================================================================================================
# A4: the startup line
# =====================================================================================================


@pytest.mark.parametrize("web", ["web", "no-web-flag"])
@pytest.mark.parametrize("source", list(SOURCES))
def test_startup_line_v3(tmp_path, capsys, monkeypatch, source, web):
    """A4: a flag or profile v3 source prints exactly ``astra session: live guidance: v3 (<source>)`` once, on
    stdout, immediately before the reask line, under ``--dry-run`` and at a real start; with ``--no-web`` the same
    one line (Q2: silently v2's text; module docstring) and nothing else on either stream differs from the v2 run
    of the same source but its version word."""
    flags, body = case_args(source, "v3", web)
    want = f"{PREFIX}v3 ({source})"
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
    f2, b2 = case_args(source, "v2", web)
    out2, err2 = tc.dry_out_err(tmp_path / "dry2", capsys, monkeypatch, *f2, body=LIVE_PROFILE + b2)

    def norm(text: str, where: Path) -> list[str]:
        return [
            x.replace(str(where), "<tmp>") for x in text.splitlines() if not x.startswith(PREFIX)
        ]

    assert norm(out, tmp_path / "dry") == norm(out2, tmp_path / "dry2"), (
        "T-0191 A4: v3's dry-run stdout differs from v2's beyond the guidance line"
    )
    assert norm(err, tmp_path / "dry") == norm(err2, tmp_path / "dry2"), err


# =====================================================================================================
# A4 adversarial: under --no-web the sentence goes nowhere
# =====================================================================================================

CLICK_A = "<|tool_call>call:click_at{x:500,y:250}<tool_call|>"
CLICK_B = "<|tool_call>call:click_at{x:120,y:640}<tool_call|>"
PRESS_CTRL_L = '<|tool_call>call:press_key{combo:<|"|>Ctrl+L<|"|>}<tool_call|>'
Q1 = "go to news.ycombinator.com"
Q2 = "what is on the page"


def cli_kwargs(tmp: Path, flags: list[str], body: str = "") -> tuple[dict[str, Any], str]:
    """The ``Session`` keywords and executor echo of a live ``cli`` real start over the fakes (T-0152's
    ``_cli_run``)."""
    from tests.unit import test_t0152_overview_frame_default as t0152

    run = t0152._cli_run(tmp, flags, body, dry=False)
    assert run["code"] == 0 and run["_raw_kwargs"], (
        f"T-0191 A4: astra session --tools live {' '.join(flags)} builds a session; exit {run['code']}\n"
        f"{run['out'][-600:]}{run['err'][-600:]}"
    )
    return t0152.rig_kwargs(run)


async def refused_ctrl_l_turns(tmp: Path, kwargs: dict[str, Any], echo: str) -> dict[str, Any]:
    """Turn 1: click, ``press_key Ctrl+L`` (refused ``not available in the browser``, as the executor does outside web
    mode), click, answer; turn 2: an answer. Returns every wire body (captioner prompts excluded), the session log's
    text, the turn lines and the browser invocations."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame, refused
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0134_label import next_turn

    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("t0191-onset"), live_frame("t0191-turn2")],
        post_frames=[live_frame(f"t0191-after-{k}") for k in (1, 2, 3)],
        echo=echo,
    )
    browser.outcome("t1-call-s2", lambda inv: refused(inv, "not available in the browser"))
    async with feedback_rig(
        tmp,
        browser,
        script=[answer("I will go there. "), tool(CLICK_A)],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps(
            [answer("Next. "), tool(PRESS_CTRL_L)],
            [answer("Next. "), tool(CLICK_B)],
            [answer("I could not leave this page. ")],
        )
        await rig.ask(Q1)
        await tsl.until_steps(rig, 3, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        await next_turn(rig, 2, Q2, [answer("A page about news. ")])
        bodies = [
            body
            for (_, prompt, _), body in zip(rig.llm.generates, rig.llm.bodies, strict=True)
            if not is_caption_prompt(prompt)
        ]
        lines = rig.lines()
        log_path = Path(rig.log.path)
        invocations = [(i.name, dict(i.arguments)) for i in browser.invocations]
    return {
        "bodies": bodies,
        "log": log_path.read_text(encoding="utf-8"),
        "lines": "\n".join(lines),
        "turns": len(lines),
        "invocations": invocations,
    }


def needles(s: str) -> list[str]:
    """S, and its two halves split at the middle space: each must be distinctive (premise checked by the row)."""
    words = s.split(" ")
    half = len(words) // 2
    return [s, " ".join(words[:half]), " ".join(words[half:])]


def found_in(needle: str, got: dict[str, Any]) -> list[str]:
    where = []
    encoded = json.dumps(needle)[1:-1]
    flat = json.dumps(got["bodies"])
    if needle in flat or encoded in flat:
        where.append("request bodies")
    if needle in got["log"] or encoded in got["log"]:
        where.append("session log")
    if needle in got["lines"] or encoded in got["lines"]:
        where.append("turn lines")
    return where


def test_no_web_never_leaks_sentence(tmp_path: Path) -> None:
    """A4 adversarial: ``--live-guidance v3 --no-web`` (and ``browser_web = false``) through cli, two turns, three tool
    steps with a refused Ctrl+L: S and each half of it appear in no request body, no session-log record and no turn
    line. Power: the same rig in web mode carries S in every body and in the log."""
    s = sentence()
    v2 = guidance_map()["v2"]
    for n in needles(s):
        assert len(n) >= 20 and n not in v2, f"premise: {n!r} is distinctive (not in v2)"
    web_kw, web_echo = cli_kwargs(tmp_path / "cli-web", [FLAG, "v3"])
    on = asyncio.run(refused_ctrl_l_turns(tmp_path / "rig-web", web_kw, web_echo))
    assert on["turns"] == 2 and len(on["bodies"]) >= 5, (on["turns"], len(on["bodies"]))
    assert all(json.dumps(s)[1:-1] in json.dumps(b) for b in on["bodies"]), (
        "power: in web mode every body carries S"
    )
    assert found_in(s, on) and "session log" in found_in(s, on), "power: web mode's log carries S"
    for label, flags, body in (
        ("flag", [FLAG, "v3", "--no-web"], ""),
        ("profile", [FLAG, "v3"], NO_WEB_KEY),
    ):
        kw, echo = cli_kwargs(tmp_path / f"cli-{label}", flags, body)
        got = asyncio.run(refused_ctrl_l_turns(tmp_path / f"rig-{label}", kw, echo))
        assert got["turns"] == 2 and len(got["bodies"]) >= 5, (
            f"premise ({label}): two turns, 4 + 1 generates; {got['turns']}, {len(got['bodies'])}"
        )
        # the overview-frame captures (T-0152, on by default) are the session's own invocations, not model steps
        names = [n for n, _ in got["invocations"] if n != "page_overview"]
        assert names[:3] == ["click_at", "press_key", "click_at"], f"premise ({label}): {names}"
        assert "not available in the browser" in got["log"], f"premise ({label}): Ctrl+L refused"
        for n in needles(s):
            where = found_in(n, got)
            assert where == [], f"T-0191 A4 ({label}): v3's sentence part {n!r} leaked into {where}"
        assert all(json.dumps(v2)[1:-1] in json.dumps(b) for b in got["bodies"]), (
            f"T-0191 A4 ({label}): every body carries v2's text"
        )


# =====================================================================================================
# A7: v1/v2/default bytes equal BASE, in both web states (active)
# =====================================================================================================

#: case -> (flags, profile body after T-0152's ``LIVE_PROFILE``)
CASES: dict[str, tuple[list[str], str]] = {
    "default-web": ([], ""),
    "default-no-web": (["--no-web"], ""),
    "default-no-web-key": ([], NO_WEB_KEY),
    "flag-v1-web": ([FLAG, "v1"], ""),
    "flag-v1-no-web": ([FLAG, "v1", "--no-web"], ""),
    "flag-v2-web": ([FLAG, "v2"], ""),
    "flag-v2-no-web": ([FLAG, "v2", "--no-web"], ""),
    "profile-v2-no-web-key": ([], f'{FIELD} = "v2"\n' + NO_WEB_KEY),
}
PAGES = ("fits", "long")


async def _rigs(tmp: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from tests.contract import test_t0174_prefix_oracle as a5
    from tests.unit import test_t0152_overview_frame_default as t0152

    out: dict[str, Any] = {}
    for case, run in runs.items():
        if run["code"] != 0 or not run["_raw_kwargs"]:
            out[case] = None
            continue
        kw, echo = t0152.rig_kwargs(run)
        out[case] = {
            page: await t0152._rig_turn(tmp / case / page, kw, echo, page) for page in PAGES
        }
        two = await a5.two_turns(tmp / case / "two", kw, echo, alternate=False)
        text = json.dumps({"bodies": two["bodies"], "prefills": two["prefills"]})
        norm = json.loads(text.replace(two["session_id"], "<session>"))
        out[case]["two_turn_bodies"] = norm["bodies"]
        out[case]["two_turn_prefills"] = norm["prefills"]
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    from tests.unit import test_t0134_differential as t0134
    from tests.unit import test_t0134c_live_feedback_default as t0134c
    from tests.unit import test_t0152_overview_frame_default as t0152

    out: dict[str, Any] = {"cli": {}}
    runs: dict[str, dict[str, Any]] = {}
    for case, (flags, body) in CASES.items():
        # T-0201 D3 re-pin: the working tree's live session pins the screen reminder off (BASE's requests)
        # T-0202 D3 re-pin: and the wait tool off (BASE's kit and guidance)
        dry = t0152._cli_run(
            tmp / "cli" / case / "dry",
            flags,
            body,
            dry=True,
            pin_reminder_off=True,
            pin_wait_off=True,
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE's turns)
        )
        start = t0152._cli_run(
            tmp / "cli" / case / "start",
            flags,
            body,
            dry=False,
            pin_reminder_off=True,
            pin_wait_off=True,
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE's turns)
        )
        runs[case] = start
        out["cli"][case] = {"dry": t0152._public(dry), "start": t0152._public(start)}
    out["rigs"] = asyncio.run(_rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    out["executor"] = asyncio.run(t0134._executor_actions())
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0191_cli_guidance as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture: ``local_astra`` from ``git archive BASE`` (T-0174's harness), origin checked."""
    gr.require_commits(BASE)
    tmp = tmp_path_factory.mktemp("t0191_base")
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
    return _capture(tmp_path_factory.mktemp("t0191_now"))


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0191 A7: differs from BASE {BASE}: {what}:\nwant {json.dumps(want)[:1500]}\n"
        f"got  {json.dumps(got)[:1500]}"
    )


def _escaped_forms(text: str, levels: int = 3) -> list[str]:
    """``text`` as it appears inside JSON nested 0..``levels`` times (each level is one ``json.dumps``), distinct."""
    forms, cur = [], text
    for _ in range(levels + 1):
        if cur not in forms:
            forms.append(cur)
        cur = json.dumps(cur)[1:-1]
    return forms


def v2_moved_to_v3(obj: Any, what: str) -> Any:
    """T-0191 C1/C3 pin move for a BASE capture of a web-mode default session: ``obj`` with every occurrence of
    ``LIVE_GUIDANCE["v2"]``'s bytes replaced by ``LIVE_GUIDANCE["v3"]``'s (at every JSON nesting level), nothing else
    touched. Premises: BASE's side carries v2's text at least once and v3's added sentence nowhere."""
    v2, v3 = guidance_map()["v2"], v3_text()
    s3 = sentence()
    text = json.dumps(obj)
    assert not any(f in text for f in _escaped_forms(s3)), (
        f"premise ({what}): BASE's capture already carries v3's sentence"
    )
    found = 0
    # deepest escaping first, so a shallower form never matches inside a deeper one's replacement
    for f2, f3 in reversed(list(zip(_escaped_forms(v2), _escaped_forms(v3), strict=True))):
        found += text.count(f2)
        text = text.replace(f2, f3)
    assert found >= 1, f"premise ({what}): BASE's capture carries v2's guidance text"
    return json.loads(text)


#: C3: the A7 cases whose expectation the flip moves (the web-mode default; v2 -> v3 text only)
C_MOVES_CASES = ("default-web",)


@pytest.mark.parametrize(
    "case",
    [c for c in CASES],
)
def test_v1_v2_bytes_equal_base(base: dict[str, Any], now: dict[str, Any], case: str) -> None:
    """A7 (active): with no flag, ``v1`` and ``v2``, in web mode and without it, cli's output and keywords
    (``--dry-run`` and a real start) and the rig sessions built from those keywords equal BASE's byte for byte
    (module docstring); nothing is admitted. Re-scoped by C3: for ``default-web`` BASE's side moves only the guidance
    text, v2's bytes to v3's (``v2_moved_to_v3``); every other byte is BASE's."""
    moved = case in C_MOVES_CASES
    for kind in ("dry", "start"):
        want, got = base["cli"][case][kind], now["cli"][case][kind]
        assert want["code"] == 0, f"premise: BASE's {case} {kind} exits 0\n{want['err'][-600:]}"
        if moved and kind == "start":
            want = v2_moved_to_v3(want, f"cli {case} {kind}")
        _same(want, got, f"cli {case} {kind}")
    want_rig, got_rig = base["rigs"][case], now["rigs"][case]
    if moved:
        want_rig = v2_moved_to_v3(want_rig, f"rig {case}")
    assert want_rig is not None and got_rig is not None, f"premise: rigs built ({case})"
    assert sorted(want_rig) == sorted(got_rig), (sorted(want_rig), sorted(got_rig))
    for key in want_rig:
        if key in PAGES:
            for section in want_rig[key]:
                _same(want_rig[key][section], got_rig[key][section], f"rig {case} {key} {section}")
        else:
            _same(want_rig[key], got_rig[key], f"rig {case} {key}")


def test_explicit_v2_is_old_default(base: dict[str, Any], now: dict[str, Any]) -> None:
    """C1 (holds before and after the flip): ``--live-guidance v2`` in web mode reproduces the pre-flip default's
    bytes: BASE's ``default-web`` cli output and keywords (``--dry-run`` and a real start) and its rig sessions equal
    today's ``flag-v2-web``, with only the one A4 line ``astra session: live guidance: v2 (flag)`` admitted (and it is
    printed exactly once)."""
    from tests.unit.test_t0174_v1_identity import _admit_line

    for kind in ("dry", "start"):
        want, got = base["cli"]["default-web"][kind], now["cli"]["flag-v2-web"][kind]
        assert want["code"] == 0 and got["code"] == 0, (want["code"], got["code"])
        got = _admit_line(got, "flag", f"flag-v2-web {kind}", version="v2")
        _same(want, got, f"C1: BASE default-web vs flag-v2-web, cli {kind}")
    want_rig, got_rig = base["rigs"]["default-web"], now["rigs"]["flag-v2-web"]
    assert want_rig is not None and got_rig is not None, "premise: rigs built"
    assert sorted(want_rig) == sorted(got_rig), (sorted(want_rig), sorted(got_rig))
    for key in want_rig:
        _same(want_rig[key], got_rig[key], f"C1: BASE default-web vs flag-v2-web, rig {key}")


def test_other_modes_and_executor_equal_base(base: dict[str, Any], now: dict[str, Any]) -> None:
    """A7 (active): tools off and dry-run through cli, and the executor's action strings and ``resolved`` over the
    fake driver, equal BASE's."""
    for case in base["other"]:
        for key in base["other"][case]:
            _same(base["other"][case][key], now["other"][case][key], f"other {case} {key}")
    _same(base["executor"], now["executor"], "executor actions")


def test_web_states_differ(now: dict[str, Any]) -> None:
    """Power premise (active): the capture sees the web state (the web and ``--no-web`` cases' cli output differ), so
    the BASE comparison covers two distinct configurations."""
    a = now["cli"]["flag-v2-web"]["dry"]["out"]
    b = now["cli"]["flag-v2-no-web"]["dry"]["out"]
    assert a != b, "premise: web mode shows in the dry-run output"


# =====================================================================================================
# A7: frozen paths and the §4 invariants (active)
# =====================================================================================================

#: A7's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/computer/executor.py",
    "local_astra/computer/runtime.py",
    "local_astra/pages/static/web.html",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "local_astra/toolcall",
    "tests/fakes",
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


def _blobs_now() -> dict[str, str]:
    listed = gr._git("ls-files", "--cached", "--others", "--exclude-standard", "--", *FROZEN)
    assert listed.returncode == 0, listed.stderr
    paths = [p for p in listed.stdout.splitlines() if (gr.REPO / p).is_file()]
    out = {}
    for p in paths:
        proc = gr._git("hash-object", "--", p)
        assert proc.returncode == 0, proc.stderr
        out[p] = proc.stdout.strip()
    return out


def test_frozen_paths_unchanged() -> None:
    """A7 (active): every frozen path is byte-identical at BASE and at TIP (the working tree while ``TIP`` is
    ``None``): same file set, same blob ids. Each listed path exists at BASE."""
    gr.require_commits(BASE)
    at_base = _blobs_at(BASE)
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in at_base), (
            f"premise: {p} exists at {BASE}"
        )
    if TIP is not None:
        gr.require_commits(TIP)
        assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
        at_tip = _blobs_at(TIP)
    else:
        at_tip = _blobs_now()
    changed = sorted(p for p in set(at_base) | set(at_tip) if at_base.get(p) != at_tip.get(p))
    assert changed == [], f"T-0191 A7: frozen paths differ from {BASE}: {changed}"


def test_section4_invariants_unedited() -> None:
    """A7 (active): docs/02 §4 is byte-identical at BASE and at TIP (the working tree while ``TIP`` is ``None``)."""
    gr.require_commits(BASE)
    proc = gr._git("show", f"{BASE}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    at_base = t7d.section(proc.stdout, "## 4. ")
    if TIP is not None:
        tip = gr._git("show", f"{TIP}:docs/02-architecture.md")
        assert tip.returncode == 0, tip.stderr
        text = tip.stdout
    else:
        text = (gr.REPO / "docs" / "02-architecture.md").read_text(encoding="utf-8")
    assert at_base.strip() and "invariant" in at_base.lower(), "premise: §4 found at BASE"
    assert t7d.section(text, "## 4. ") == at_base, "T-0191 A7: docs/02 §4 changed"


def test_profile_field_unchanged() -> None:
    """A3 (active): ``Profile.live_guidance`` stays ``str | None = None`` and ``None`` in every built-in profile."""
    f = Profile.model_fields[FIELD]
    assert f.default is None and f.annotation == (str | None), f
    for name, profile in load(None).profiles.items():
        assert getattr(profile, FIELD) is None, name
