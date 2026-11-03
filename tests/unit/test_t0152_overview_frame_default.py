"""T-0152: the overview frame becomes the live default; beside the auto overview, the page overview or ReGround the
default gives way (H-T152-1), an explicit frame beside them is still refused, and ``--no-overview-frame`` is BASE
(docs/tasks/T-0152.md, A1-A9).

**The surface these rows pin** (the task's names; readings stated):

- ``config.LIVE_DEFAULT_OVERVIEW_FRAME is True`` (A1); ``Profile.overview_frame: bool | None = None`` and the T-0151
  A2-9 load refusals as at BASE.
- ``cli.overview_frame_of(profile, args, mode)`` (A2). Its result is read by the two field names T-0151 gave it,
  ``overview_frame`` and ``overview_frame_from`` (``frame_pair``), never as "a two-field value": the task adds a
  field that records the yield, and **its name is the implementer's**, so these rows read the yield only through the
  A5 line, never through that field.
- The ``overview frame`` startup line (A5), exact literals typed from the criterion: ``on (default)``;
  ``off (with --reground)`` / ``off (with --page-overview)`` / ``off (with --auto-overview)``, joined with `` and ``
  in the order auto, page, reground, the flag spelling whatever the other option's source (H-T152-2).
- What is built (A4): the keyword ``overview_frame=True`` by default; no keyword when off by flag, key or yield.

**BASE** is ``BASE_SHA`` 1670ee4 (main at the branch, "T-0152: H-T152-1/2 approved"). The BASE side of every
comparison runs this module's own ``_capture`` in a subprocess over ``git archive BASE_SHA`` (T-0131's harness,
``tsl._base_tree``; origin checked), so "BASE's default live session" is BASE's ``cli`` building it, never a
session this module assembles by hand. The rig sessions (A6/A7) are built with exactly the keywords ``cli`` passed
(T-0134c's pattern: a ``cli``-built graph is never driven on its real clock) over the fake LLM and
``FakeLiveBrowser`` with a scripted 1.525-screen overview (T-0151's ``deals_browser`` values) or a fitting page.

**Readings** (flagged in the report, not resolved):

- **Rd-1 (A7a/A7b scope).** "byte-identical ... request bodies, turn lines, session-log records, executor action
  strings and HUD messages" is compared over rig sessions for the cases a rig can build from ``cli``'s keywords:
  the default, ``--no-overview-frame``, ``overview_frame = false`` and ``--auto-overview``. ``--reground`` and
  ``--page-overview`` need a pass-2 client or the overview kit that the rig would have to assemble itself; for those
  two the comparison is ``cli``'s side (the ``--dry-run`` output, the startup output, every ``Session`` and executor
  keyword, described), which with A7e's frozen ``conductor/**`` fixes the bodies.
- **Rd-2 (A5 help).** "states that the default is on and gives way to the other three": the ``--overview-frame``
  help names ``default``, says ``on`` after it (not ``else off``) and names the three flags. Keywords, never wording.
- **Rd-3 (A9 docs).** The paragraph is read at ``TIP_IMPL``, a literal bumped by the test-author or orchestrator to
  the implementation tip (``= BASE_SHA`` until then, so the row is red on its bump assertion).

Green at BASE (BASE-identity and regression rows): ``test_load_refusals_unchanged``, ``test_sources``,
``test_explicit_still_refused``, ``test_no_frame_beside_others_builds``, ``test_off_is_base``,
``test_other_modes_unchanged``, ``test_other_resolvers_unchanged``, ``test_resolver_pure`` and
``test_frozen_paths_over_closed_range``. Activated for ``/implement T-0152`` (rulings O-1, O-2; no pending
marker): every other row is red at BASE on a named assertion until the implementation lands. O-1: the resolver keeps
two fields and may take ``grounding=`` (the spies here forward keywords).

Regime: CPU over fakes, ``ScriptedClock`` for the rig turns, thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, deals-like page 1220/800 CSS px (1.525 screens), overview 838x798; nothing served; n = 1,
deterministic.
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
from tests.unit import _t0201_pins as t0201
from tests.unit import _t0202_pins as t0202
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0134_differential as t0134
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0151_ab_harness as t0151h
from tests.unit import test_t0151_session as s151
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, image_urls
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_cli_tools_live import Events, argv, fake_live
from tests.unit.test_session_live import last_user, pass1_body
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _with_history
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134_label import STEM, all_bodies
from tests.unit.test_t0151_cli import (
    AUTO_PREFIX,
    LIVE,
    LIVE_PROFILE,
    OFF,
    ON,
    PREFIX,
    lines,
    names_both,
)

#: BASE: 1670ee4 (main at the T-0152 branch, "T-0152: H-T152-1/2 approved (James)"). A literal, never a moving ref.
BASE_SHA = "1670ee4"
#: The implementation tip; equals BASE_SHA until the implementation is committed. Bumped by the test-author or the
#: orchestrator to the last implementation commit (the implementer never edits tests); at the merge it closes there.
TIP_IMPL = "9482cf2"
HEADER = "**Spec delta (T-0152).**"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"


#: A5's literals, typed from the criterion
ON_DEFAULT = PREFIX + "on (default)"
OFF_DEFAULT = PREFIX + "off (default)"
WITH = {
    "reground": PREFIX + "off (with --reground)",
    "page": PREFIX + "off (with --page-overview)",
    "auto": PREFIX + "off (with --auto-overview)",
}
PAGE_AND_REGROUND = PREFIX + "off (with --page-overview and --reground)"
REGIME = "astra session: regime:"

#: the other three, as a flag and as a profile key (no frame flag); the A5 line each must give
YIELDS: dict[str, tuple[list[str], str, str]] = {
    "reground-flag": (["--reground"], "", WITH["reground"]),
    "reground-key": ([], "reground = true\n", WITH["reground"]),
    "page-flag": (["--page-overview"], "", WITH["page"]),
    "page-key": ([], "page_overview = true\n", WITH["page"]),
    "auto-flag": (["--auto-overview"], "", WITH["auto"]),
    "auto-key": ([], "auto_overview = true\n", WITH["auto"]),
    "page+reground": (["--page-overview", "--reground"], "", PAGE_AND_REGROUND),
    "page-key+reground-flag": (["--reground"], "page_overview = true\n", PAGE_AND_REGROUND),
}

#: every live case whose ``cli`` side is captured at BASE and now: (flags, profile body after ``LIVE_PROFILE``)
CLI_CASES: dict[str, tuple[list[str], str]] = {
    "default": ([], ""),
    "no-frame-flag": ([OFF], ""),
    "no-frame-key": ([], "overview_frame = false\n"),
    **{k: (f, b) for k, (f, b, _) in YIELDS.items()},
    "no-frame+reground": ([OFF, "--reground"], ""),
    "no-frame+page": ([OFF, "--page-overview"], ""),
    "no-frame+auto": ([OFF, "--auto-overview"], ""),
}
#: A3: an explicit frame beside each of the three, from three source pairs (the task's three)
REFUSED: dict[str, tuple[list[str], str, str]] = {
    "frame-flag+reground-flag": ([ON, "--reground"], "", "reground"),
    "frame-flag+page-key": ([ON], "page_overview = true\n", "page overview"),
    "frame-key+auto-flag": (["--auto-overview"], "overview_frame = true\n", "auto overview"),
}
#: the cases a rig session is driven for (Rd-1), on a long and on a fitting page
RIG_CASES = ("default", "no-frame-flag", "no-frame-key", "auto-flag")
PAGES = {"long": s151.DEALS, "fits": s151.FITS}
#: keywords the rig owns (its clock, config, logs and kit); every other keyword ``cli`` passed is passed to the rig
RIG_OWNED = frozenset({"clock", "config", "log_dir", "session_log", "tool_kit"})


# --------------------------------------------------------------------------------------------- seams


def frame_pair(v: Any) -> tuple[Any, Any]:
    """The resolver's value and source, by T-0151's field names (a third field, the yield's record, is allowed)."""
    return v.overview_frame, v.overview_frame_from


def resolve(body: str, tmp: Path, **ns: Any) -> tuple[Any, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    profile = load(_toml(tmp, LIVE_PROFILE + body)).active()
    return frame_pair(cli.overview_frame_of(profile, argparse.Namespace(**ns), "live"))


def _flags_ns(flags: list[str]) -> dict[str, Any]:
    """The argparse dests the flags set (``BooleanOptionalAction``; the four of this task)."""
    dests = {
        "--reground": ("reground", True),
        "--page-overview": ("page_overview", True),
        "--auto-overview": ("auto_overview", True),
        ON: ("overview_frame", True),
        OFF: ("overview_frame", False),
    }
    return dict(dests[f] for f in flags)


#: T-0174 D4b: the flag a differential's working-tree rig passes so its live session keeps BASE's (v1) guidance
#: after the default flipped to v2, and the one startup line that flag prints (A4)
V1_PIN = ["--live-guidance", "v1"]
V1_PIN_LINE = "astra session: live guidance: v1 (flag)"


def _tree_has_guidance_selector() -> bool:
    """Whether the ``local_astra`` under test has T-0174's selector (the working tree does; a BASE tree before
    T-0174 A, run in a subprocess with this module, does not and is run without the pin)."""
    return isinstance(getattr(config, "LIVE_GUIDANCE_VERSIONS", None), tuple)


def _drop_v1_pin_line(text: str, where: str) -> str:
    """``text`` without the one A4 line the pin prints; exactly one is required (T-0174 D4b)."""
    lines = text.splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if x.rstrip("\n") == V1_PIN_LINE]
    assert len(found) == 1, f"T-0174 D4b ({where}): the v1 pin prints {V1_PIN_LINE!r} once; {found}"
    return "".join(lines[: found[0]] + lines[found[0] + 1 :])


#: T-0179 A-5: the flag a differential's working-tree rig passes so its live session keeps BASE's repeat guard (off)
#: after ``LIVE_DEFAULT_REPEAT_GUARD`` flipped to on, the line that flag prints, and the line BASE prints in its place
GUARD_OFF_PIN = ["--no-repeat-guard"]
GUARD_OFF_PIN_LINE = "astra session: repeat guard: off (flag)"
GUARD_BASE_LINE = "astra session: repeat guard: off (default)"


def _names_repeat_guard(flags: list[str], body: str) -> bool:
    """A case that sets the repeat guard itself (a flag or the profile key) is left as it is: the pin would override
    it, so it is not added."""
    return any("repeat-guard" in f for f in flags) or "repeat_guard" in body


def _swap_guard_pin_line(text: str, where: str) -> str:
    """``text`` with the pin's ``off (flag)`` line read back as BASE's ``off (default)`` line; exactly one is required
    (T-0179 A-5, composed with T-0174 D4b)."""
    lines = text.splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if x.rstrip("\n") == GUARD_OFF_PIN_LINE]
    assert len(found) == 1, (
        f"T-0179 A-5 ({where}): the guard-off pin prints {GUARD_OFF_PIN_LINE!r} once; {found}"
    )
    lines[found[0]] = GUARD_BASE_LINE + "\n"
    return "".join(lines)


def _cli_run(
    where: Path,
    flags: list[str],
    body: str,
    *,
    dry: bool,
    pin_v1: bool = False,
    pin_guard_off: bool = False,
    pin_reminder_off: bool = False,
    pin_wait_off: bool = False,
    pin_check_off: bool = False,
) -> dict[str, Any]:
    """``astra session --tools live --browser headless <flags>`` over the fakes, ``--dry-run`` or a real start
    stopped before serving: exit code, stdout, stderr (normalised), the described ``Session`` and executor keywords
    and how many graphs were built. An exception is recorded, never raised (a mutant may raise inside ``cli``).

    ``pin_v1`` (T-0174 D4b): on a tree with the guidance selector, append ``--live-guidance v1`` so the session is
    v1's (BASE's) whatever the default is, and remove the one ``live guidance: v1 (flag)`` line it prints from stdout
    (required present once); every other byte is compared as before. On a tree without the selector (a BASE run) it
    does nothing.

    ``pin_guard_off`` (T-0179 A-5, the same pattern, additive): on a tree with the guidance selector, append
    ``--no-repeat-guard`` unless the case sets the guard itself, so the live session keeps BASE's guard (off) whatever
    the default is, and read its ``repeat guard: off (flag)`` line back as BASE's ``off (default)`` line (required
    present once); every other byte is compared as before. A BASE run is untouched.

    ``pin_reminder_off`` (T-0201 D3 re-pin, the same pattern, additive): on a tree with ``--screen-reminder``, append
    ``--screen-reminder off`` unless the case sets it itself, so the live session keeps BASE's requests (no screen
    reminder) whatever the default is; remove the one ``screen reminder: off (flag)`` line (required present once) and
    set aside a ``screen_reminder: False`` keyword (``_t0201_pins``). A BASE run is untouched.

    ``pin_wait_off`` (T-0202 D3 re-pin, the same pattern, additive): on a tree with ``--wait-tool``, append
    ``--wait-tool off`` unless the case sets it itself, so the live session keeps BASE's kit (v1) and guidance (no
    wait sentence) whatever the default is; ``--wait-tool off`` prints nothing and passes no keyword, so nothing is
    set aside (``_t0202_pins``). A BASE run is untouched.

    ``pin_check_off`` (T-0207 C3 re-pin, the same pattern, additive): on a tree with ``--completion-check``, append
    ``--completion-check off`` unless the case sets it itself, so the live session keeps BASE's turns (no completion
    check, no ``completion check outcome`` note) whatever the default is; remove the one ``completion check: off
    (flag)`` line (required present once) and set aside a ``completion_check: False`` keyword (``_t0207_pins``). A
    BASE run is untouched."""
    check_pinned = pin_check_off and t0207.pins(flags, body)  # T-0207 C3 re-pin
    if check_pinned:
        flags = [*flags, *t0207.CHECK_OFF_PIN]  # T-0207 C3 re-pin
    if pin_wait_off:
        flags = t0202.pin_off(flags, body)  # T-0202 D3 re-pin
    pinned = pin_v1 and _tree_has_guidance_selector()
    guard_pinned = (
        pin_guard_off and _tree_has_guidance_selector() and not _names_repeat_guard(flags, body)
    )
    if guard_pinned:
        flags = [*flags, *GUARD_OFF_PIN]
    reminder_pinned = (
        pin_reminder_off and t0201.tree_has_flag() and not t0201.names_reminder(flags, body)
    )
    if reminder_pinned:
        flags = [*flags, *t0201.REMINDER_OFF_PIN]  # T-0201 D3 re-pin
    if pinned:
        flags = [*flags, *V1_PIN]
    where.mkdir(parents=True, exist_ok=True)
    w = str(where)
    stdout, stderr = io.StringIO(), io.StringIO()
    calls = T.Calls()
    made: list[Any] = []
    raised = None
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        mp.setenv("DISPLAY", ":99")
        T.patch_detok_load(mp, calls, where)
        if dry:
            explode_clients(mp)
        else:
            T.fake_clients(mp, calls)
            T.stop_before_serving(mp, calls)
        made = fake_live(mp, Events())
        args = [*argv(where, *LIVE, *flags, config_body=LIVE_PROFILE + body)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code: Any = run_cli([*args, "--dry-run"] if dry else args)
            except Exception as exc:  # recorded: a row compares it
                code, raised = None, f"{type(exc).__name__}: {exc}"
    printed = stdout.getvalue()
    if pinned and code == 0:
        assert "live guidance" not in stderr.getvalue(), (
            "T-0174 A4: no live guidance line on stderr"
        )
        printed = _drop_v1_pin_line(printed, str(where))
    if guard_pinned and code == 0:
        printed = _swap_guard_pin_line(printed, str(where))
    session_kwargs: list[Any] = calls.session_kwargs
    if reminder_pinned and code == 0:
        printed = t0201.drop_off_pin_line(printed, str(where))  # T-0201 D3 re-pin
        session_kwargs = t0201.kwargs_without_off(session_kwargs)
    if check_pinned and code == 0:
        printed = t0207.drop_off_pin_line(printed, str(where))  # T-0207 C3 re-pin
        session_kwargs = t0207.kwargs_without_off(session_kwargs)
    return {
        "code": code,
        "raised": raised,
        "out": t0134c._norm(printed, w),
        "err": t0134c._norm(stderr.getvalue(), w),
        "graphs": len(calls.graphs),
        "session_kwargs": [t0134c._describe(k, w) for k in session_kwargs],
        "executor_kwargs": [t0134c._describe(dict(getattr(m, "kwargs", {})), w) for m in made],
        "_raw_kwargs": calls.session_kwargs,
        "_made": made,
    }


def _public(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if not k.startswith("_")}


async def _rig_turn(tmp: Path, kwargs: dict[str, Any], echo: str, page: str) -> dict[str, Any]:
    """One live loop turn (step 1 says ``S1`` and clicks the towel at (160, 935); step 2 reports ``S3``) on a rig
    session built with ``kwargs``, over a browser whose page is ``PAGES[page]`` and whose overview is T-0151's ``OV``.
    Sections: machine, HUD, prompts, requests, log, records (turn lines), history (T-0121's ``_with_history``); the
    wire bodies; the browser invocations (name, call id, frame geometry)."""
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser([s151.F0], post_frames=[s151.F1, s151.F2], echo=echo)
    browser.overview_image = s151.OV
    browser.page_metrics = dict(PAGES[page])
    async with feedback_rig(
        tmp,
        browser,
        script=[answer(s151.S1), tool(s151.TOWEL)],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        await s151.run_turn(rig, ([answer(s151.S3)],))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
        bodies = all_bodies(rig)
        pass1 = pass1_body(rig)
    out = _with_history(rig, tmp, session_id, log_path)
    text = json.dumps({"bodies": bodies, "pass1": pass1}).replace(session_id, "<session>")
    out.update(json.loads(text.replace(str(tmp), "<tmp>")))
    out["invocations"] = [
        {
            "name": i.name,
            "call_id": i.call_id,
            "frame": None
            if i.frame is None
            else {
                "phash": i.frame.phash,
                "source": str(i.frame.source),
                "width": i.frame.width,
                "height": i.frame.height,
            },
        }
        for i in browser.invocations
    ]
    return out


def rig_kwargs(run: dict[str, Any]) -> tuple[dict[str, Any], str]:
    kw = {k: v for k, v in run["_raw_kwargs"][-1].items() if k not in RIG_OWNED}
    echo = str(dict(getattr(run["_made"][-1], "kwargs", {})).get("echo", "pixels"))
    return kw, echo


def resolver_grid() -> dict[str, Any]:
    """A7d: the other five resolvers (``grounding_of``, ``feedback_of``, ``recovery_of``, ``page_overview_of``,
    ``auto_overview_of``) and their lines over a grid of profiles, namespaces (this task's dest included, which they
    must ignore) and modes. Executed in-process here and in the BASE tree's interpreter."""
    profiles = {
        "plain": dict(name="lean"),
        "dry": dict(name="lean", tools="dry_run"),
        "live": dict(name="lean", tools="live"),
        "live-reground": dict(name="lean", tools="live", reground=True),
        "live-page": dict(name="lean", tools="live", page_overview=True),
        "live-auto": dict(name="lean", tools="live", auto_overview=True),
        "live-frame": dict(name="lean", tools="live", overview_frame=True),
        "live-frame-off": dict(name="lean", tools="live", overview_frame=False, reground=True),
        "live-fb": dict(
            name="lean", tools="live", live_feedback="label+extent", thought_call_recovery=False
        ),
    }
    spaces = {
        "empty": {},
        "frame-on": dict(overview_frame=True),
        "frame-off": dict(overview_frame=False),
        "reground": dict(reground=True),
        "page": dict(page_overview=True),
        "auto": dict(auto_overview=True),
        "frame-off-reground": dict(overview_frame=False, reground=True),
        "all-off": dict(
            overview_frame=False, reground=False, page_overview=False, auto_overview=False
        ),
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
                }
    return out


async def _rigs(tmp: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in RIG_CASES:
        run = runs[case]
        if run["code"] != 0 or not run["_raw_kwargs"]:
            out[case] = None
            continue
        kw, echo = rig_kwargs(run)
        out[case] = {page: await _rig_turn(tmp / case / page, kw, echo, page) for page in PAGES}
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree. ``cli`` runs its own
    event loop, so the ``cli`` runs happen outside any loop and the rig turns in one of their own."""
    out: dict[str, Any] = {"cli": {}, "refused": {}}
    runs: dict[str, dict[str, Any]] = {}
    for case, (flags, body) in CLI_CASES.items():
        # T-0174 D4b + T-0179 A-5: the working tree's live session pins v1 and the guard off (BASE's); a no-op in the BASE run
        dry = _cli_run(
            tmp / "cli" / case / "dry",
            flags,
            body,
            dry=True,
            pin_v1=True,
            pin_guard_off=True,
            pin_reminder_off=True,  # T-0201 D3 re-pin: the reminder pinned off (BASE's requests)
            pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE's kit and guidance)
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE's turns)
        )
        start = _cli_run(
            tmp / "cli" / case / "start",
            flags,
            body,
            dry=False,
            pin_v1=True,
            pin_guard_off=True,
            pin_reminder_off=True,  # T-0201 D3 re-pin
            pin_wait_off=True,  # T-0202 D3 re-pin
            pin_check_off=True,  # T-0207 C3 re-pin
        )
        runs[case] = start
        out["cli"][case] = {"dry": _public(dry), "start": _public(start)}
    for case, (flags, body, _) in REFUSED.items():
        out["refused"][case] = {
            kind: _public(_cli_run(tmp / "refused" / case / kind, flags, body, dry=kind == "dry"))
            for kind in ("dry", "start")
        }
    out["rigs"] = asyncio.run(_rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    out["grid"] = resolver_grid()
    out["executor"] = asyncio.run(t0134._executor_actions())
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0152_overview_frame_default as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture, in BASE's tree (T-0131's ``git archive`` harness through ``tsl._base_tree``)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0152_base")
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


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The same capture on the working tree, once per module."""
    return _capture(tmp_path_factory.mktemp("t0152_now"))


#: T-0136 re-pin (O-C1, docs/tasks/T-0136.md C-5, Q-2): T-0136's live-only line, printed right after the frame line;
#: absent at BASE and before T-0136 C, so dropping it keeps the rows green before and after
T0136_PREFIX = "astra session: action outcome: "


def _drop_frame_line(text: str) -> list[str]:
    return [
        x
        for x in text.splitlines()
        if not x.startswith(PREFIX)
        and not x.startswith(T0136_PREFIX)  # T-0136 re-pin
        and not x.startswith("astra session: repeat guard:")  # T-0137 re-pin (O-B2)
    ]


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0152 {what} differs from BASE {BASE_SHA}:\nbase {json.dumps(want)[:1500]}\n"
        f"now  {json.dumps(got)[:1500]}"
    )


# =====================================================================================================
# A1: the constant
# =====================================================================================================


def test_constant_and_profile():
    """A1 (reviewer mutation 1): ``config.LIVE_DEFAULT_OVERVIEW_FRAME is True`` and ``cli`` sees the same value; the
    ``Profile.overview_frame`` annotation and default are BASE's (``bool | None``, ``None``) in every built-in
    profile."""
    assert config.LIVE_DEFAULT_OVERVIEW_FRAME is True, (
        "T-0152 A1: config.LIVE_DEFAULT_OVERVIEW_FRAME = True"
    )
    assert cli.LIVE_DEFAULT_OVERVIEW_FRAME is True, "T-0152 A1: cli reads the same constant"
    f = Profile.model_fields["overview_frame"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.overview_frame is None, name


#: T-0151 A2-9's load refusals (one row each): (profile body, what the message names)
LOAD_REFUSALS = {
    "tools-off": ('tools = "off"\noverview_frame = true\n', "live"),
    "tools-dry_run": ('tools = "dry_run"\noverview_frame = false\n', "live"),
    "tools-absent": ("overview_frame = true\n", "live"),
    "not-a-bool": ('tools = "live"\noverview_frame = "on"\n', "overview_frame"),
    "true-with-auto": (
        'tools = "live"\noverview_frame = true\nauto_overview = true\n',
        "overview_frame",
    ),
    "true-with-page": (
        'tools = "live"\noverview_frame = true\npage_overview = true\n',
        "overview_frame",
    ),
    "true-with-reground": (
        'tools = "live"\noverview_frame = true\nreground = true\n',
        "overview_frame",
    ),
}


@pytest.mark.parametrize("case", list(LOAD_REFUSALS))
def test_load_refusals_unchanged(tmp_path, case):
    """A1 (green at BASE): each T-0151 A2-9 load refusal stands, naming ``overview_frame`` (and ``live`` for the mode
    refusals); C1's companion: ``overview_frame = false`` beside the same ``true`` key loads."""
    body, needle = LOAD_REFUSALS[case]
    with pytest.raises(ValueError) as exc:
        load(_toml(tmp_path, body))
    assert "overview_frame" in str(exc.value), str(exc.value)
    if needle == "live":
        assert t0134c._names_live(str(exc.value)), str(exc.value)
    if case.startswith("true-with-"):
        (tmp_path / "c1").mkdir()
        ok = load(
            _toml(tmp_path / "c1", body.replace("overview_frame = true", "overview_frame = false"))
        )
        assert ok.active().overview_frame is False


# =====================================================================================================
# A2/A4/A5: the default
# =====================================================================================================


def test_default_on(tmp_path, capsys, monkeypatch):
    """A2/A4/A5 (reviewer mutation 1): ``Namespace()`` in live resolves ``(True, "default")`` (also with every dest
    ``None``); the default ``cli`` build passes ``Session(overview_frame=True)`` (no auto overview, no ReGround, the
    v1 kit); the line is ``on (default)`` under ``--dry-run`` and at a real start, right after the ``auto overview``
    line and (at a real start) before the regime line. In ``off``/``dry_run`` the resolver gives ``(False,
    "unset")``."""
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert frame_pair(cli.overview_frame_of(profile, argparse.Namespace(), "live")) == (
        True,
        "default",
    ), "T-0152 A2: the live default is on"
    nones = argparse.Namespace(
        overview_frame=None, reground=None, page_overview=None, auto_overview=None
    )
    assert frame_pair(cli.overview_frame_of(profile, nones, "live")) == (True, "default")
    for mode in ("off", "dry_run"):
        got = frame_pair(cli.overview_frame_of(Profile(name="lean"), argparse.Namespace(), mode))
        assert got == (False, "unset"), (mode, got)
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    kw = calls.session_kwargs[-1]
    assert kw.get("overview_frame") is True, (
        f"T-0152 A4: Session(overview_frame=True); {sorted(kw)}"
    )
    assert not kw.get("auto_overview") and not kw.get("reground"), kw
    # T-0202 D3 re-pin: the default live kit is v1, or the wait set once the wait tool is on by default (never the
    # overview set: the overview frame offers no tool)
    assert calls.kit().tools_version == t0202.default_live_sha(page_overview=False)
    assert t0202.default_on() or calls.kit().tools_version == V1_SHA
    monkeypatch.undo()
    dry = t0134c.dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(dry) == [ON_DEFAULT], dry
    dl = dry.splitlines()
    (a,) = [i for i, x in enumerate(dl) if x.startswith(AUTO_PREFIX)]
    assert dl[a + 1] == ON_DEFAULT, dry
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    sl = started.splitlines()
    assert lines(started) == [ON_DEFAULT], started
    (a,) = [i for i, x in enumerate(sl) if x.startswith(AUTO_PREFIX)]
    (r,) = [i for i, x in enumerate(sl) if x.startswith(REGIME)]
    assert sl[a + 1] == ON_DEFAULT and a + 1 < r, started


def test_help_states_default_on():
    """A5 (reading Rd-2): the ``--overview-frame`` help says the default is on (``default`` then ``on``, no ``else
    off``) and names the three it gives way to. Keywords, never wording."""
    h = t0133._help_of(ON)
    low = h.lower()
    assert "else off" not in low, f"T-0152 A5: the help still says the default is off: {h}"
    assert re.search(r"\bdefault\b[^;]*\bon\b", low), f"T-0152 A5: the help says default on: {h}"
    for flag in ("--reground", "--page-overview", "--auto-overview"):
        assert flag in h, (flag, h)


#: (flags, profile body, resolved pair, line); none needs the flip, so the row is green at BASE
SOURCES = {
    "flag-on": ([ON], "", (True, "flag"), PREFIX + "on (flag)"),
    "flag-off": ([OFF], "", (False, "flag"), PREFIX + "off (flag)"),
    "key-true": ([], "overview_frame = true\n", (True, "profile"), PREFIX + "on (profile)"),
    "key-false": ([], "overview_frame = false\n", (False, "profile"), PREFIX + "off (profile)"),
    "flag-off-over-key": ([OFF], "overview_frame = true\n", (False, "flag"), PREFIX + "off (flag)"),
    "flag-on-over-key": ([ON], "overview_frame = false\n", (True, "flag"), PREFIX + "on (flag)"),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources(tmp_path, capsys, monkeypatch, case):
    """A2/A5 (green at BASE): the flag, else the key, each with its source and BASE's line; the build passes
    ``overview_frame=True`` exactly when on and no keyword when off."""
    flags, body, want, line = SOURCES[case]
    assert resolve(body, tmp_path / "p", **_flags_ns(flags)) == want
    out = t0134c.dry_run(
        tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body
    )
    assert lines(out) == [line], out
    monkeypatch.undo()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    if want[0]:
        assert kw.get("overview_frame") is True, sorted(kw)
    else:
        assert "overview_frame" not in kw, f"T-0152 A4: off passes no keyword; {sorted(kw)}"


# =====================================================================================================
# A2/A3/A5: the yield
# =====================================================================================================


@pytest.mark.parametrize("case", list(YIELDS))
def test_yields(tmp_path, capsys, monkeypatch, case):
    """A2/A3/A4/A5 (adversarial; reviewer mutations 2, 4 and 6): with the auto overview, the page overview or ReGround
    on (flag or key) and no frame flag, the resolver itself gives ``(False, "default")`` (a yield only in
    ``session_cmd`` leaves ``_build_graph`` passing ``overview_frame=True`` beside them, which ``Session`` refuses);
    the build succeeds and ``Session`` is constructed with no ``overview_frame`` keyword (and never ``True`` beside
    ``auto_overview=True``, the overview kit or ``reground=True``); the line is the exact ``off (with ...)`` literal,
    under ``--dry-run`` and at a real start."""
    flags, body, line = YIELDS[case]
    assert resolve(body, tmp_path / "p", **_flags_ns(flags)) == (False, "default"), (
        f"T-0152 A2: the default gives way inside overview_frame_of ({case})"
    )
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert calls.graphs and calls.session_kwargs, "premise: Session was constructed"
    kw = calls.session_kwargs[-1]
    assert "overview_frame" not in kw, f"T-0152 A4: off by yield passes no keyword; {sorted(kw)}"
    monkeypatch.undo()
    out = t0134c.dry_run(
        tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body
    )
    assert lines(out) == [line], f"T-0152 A5 ({case}):\n{out}"
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "s", capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body
    )
    assert code == 0 and lines(started) == [line], started


@pytest.mark.parametrize(
    "case, own", [("reground-key", "reground"), ("page-key", "page"), ("auto-key", "auto")]
)
def test_yield_line_uses_flag_spelling(now, case, own):
    """A5 (H-T152-2): with the other option from the profile, the frame line still names it by its flag spelling
    (``--reground``, ``--page-overview``, ``--auto-overview``), while that option's own startup line states its
    source, ``(profile)``. Read from the captured real-start output."""
    out = now["cli"][case]["start"]["out"]
    assert lines(out) == [WITH[own]], out
    prefix = {
        "reground": "astra session: grounding:",
        "page": "astra session: page overview:",
        "auto": AUTO_PREFIX,
    }[own]
    (other,) = [x for x in out.splitlines() if x.startswith(prefix)]
    assert "(profile)" in other, f"premise: the other option's line states its source; {other}"


@pytest.mark.parametrize("case", list(REFUSED))
def test_explicit_frame_never_yields(tmp_path, case):
    """A2/A3 (green at BASE; adversarial, reviewer mutation 3, the yield swallowing an explicit frame): the explicit
    flag or key beside the other option still resolves on, from its own source, so ``session_cmd``'s refusal sees
    it."""
    flags, body, _ = REFUSED[case]
    src = "flag" if ON in flags else "profile"
    # an explicit key beside a key of the other three is a load refusal (C1); the pairs here are flag/key mixed
    assert resolve(body, tmp_path / "e", **_flags_ns(flags)) == (True, src)


@pytest.mark.parametrize("case", list(REFUSED))
def test_explicit_still_refused(base, now, case):
    """A3 (green at BASE; adversarial, reviewer mutation 3): ``--overview-frame --reground``, ``--overview-frame``
    with ``page_overview = true`` and ``overview_frame = true`` with ``--auto-overview`` each exit 2 naming both,
    build nothing, with ``--dry-run`` and at a real start, with BASE's message byte for byte."""
    _, _, words = REFUSED[case]
    for kind in ("dry", "start"):
        got, want = now["refused"][case][kind], base["refused"][case][kind]
        assert got["code"] == 2 and got["raised"] is None, (case, kind, got)
        assert names_both(got["out"] + got["err"], words), (case, kind, got["err"])
        assert got["graphs"] == 0 and got["session_kwargs"] == [], (
            f"{case} {kind}: built before refusing"
        )
        _same(want, got, f"A3 {case} {kind}")


@pytest.mark.parametrize("other", ["--reground", "--page-overview", "--auto-overview"])
def test_no_frame_beside_others_builds(tmp_path, capsys, monkeypatch, other):
    """A3 (green at BASE): ``--no-overview-frame`` with each of the three builds, passes no ``overview_frame`` keyword
    and prints ``off (flag)``."""
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, OFF, other)
    assert calls.graphs and "overview_frame" not in calls.session_kwargs[-1]
    monkeypatch.undo()
    out = t0134c.dry_run(tmp_path / "d", capsys, monkeypatch, "live", OFF, other)
    assert lines(out) == [PREFIX + "off (flag)"], out


def test_one_resolver(tmp_path, capsys, monkeypatch):
    """A2 "one resolver" (reviewer mutation 2): ``session_cmd`` and ``_build_graph`` both read the frame through
    ``cli.overview_frame_of`` (a spy wrapping the real one sees at least two calls on a default build, every one in
    live and ``(True, "default")``), and a spy forcing ``(False, "default")`` on a default build leaves ``Session``
    without the keyword and the line ``off (default)`` (nothing re-derives the default beside the resolver)."""
    real = cli.overview_frame_of
    seen: list[Any] = []

    def spy(p: Any, a: Any, m: str, **kw: Any) -> Any:
        # O-1: the cli call sites may pass ``grounding=``; forwarded untouched
        got = real(p, a, m, **kw)
        seen.append((m, frame_pair(got)))
        return got

    monkeypatch.setattr(cli, "overview_frame_of", spy)
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    assert len(seen) >= 2 and all(x == ("live", (True, "default")) for x in seen), seen
    assert calls.session_kwargs[-1].get("overview_frame") is True
    monkeypatch.undo()
    forced = real(
        Profile(name="lean"), argparse.Namespace(), "off"
    )  # an off value of the real class
    monkeypatch.setattr(
        cli,
        "overview_frame_of",
        lambda p, a, m, **kw: (
            dataclasses.replace(forced, overview_frame_from="default") if m == "live" else forced
        ),
    )
    calls, _ = build(tmp_path / "f", capsys, monkeypatch)
    assert "overview_frame" not in calls.session_kwargs[-1]


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
        raise AssertionError("overview_frame_of opened a socket")
socket.socket = NoSocket
builtins.open = spy
try:
    out = []
    live = Profile(name="lean", tools="live")
    for ns in ({}, {"reground": True}, {"page_overview": True}, {"auto_overview": True},
               {"overview_frame": True}, {"overview_frame": False, "reground": True}):
        f = cli.overview_frame_of(live, argparse.Namespace(**ns), "live")
        out.append([f.overview_frame, f.overview_frame_from])
    for mode in ("off", "dry_run"):
        f = cli.overview_frame_of(Profile(name="lean"), argparse.Namespace(), mode)
        out.append([f.overview_frame, f.overview_frame_from])
finally:
    builtins.open = real_open
after = sorted(m for m in sys.modules if m.startswith("local_astra.computer"))
print(json.dumps({"before": before, "after": after, "opened": opened, "values": out}))
"""


def test_resolver_pure():
    """A2 (green at BASE): ``overview_frame_of``, over the yield's inputs too, opens no socket and no file and imports
    no ``local_astra.computer`` module (a fresh interpreter); outside live it is ``(False, "unset")``."""
    proc = subprocess.run(
        [sys.executable, "-c", _PURE], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, f"T-0152 A2: overview_frame_of is not pure:\n{proc.stderr[-2000:]}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["before"] == [] and got["after"] == [], got
    assert got["opened"] == [], got["opened"]
    assert got["values"][-2:] == [[False, "unset"], [False, "unset"]], got["values"]
    assert got["values"][4] == [True, "flag"] and got["values"][5] == [False, "flag"], got["values"]


# =====================================================================================================
# A5 over the capture: every other line is BASE's
# =====================================================================================================


# T-0125 pin move: the default live search line, startup line, kit
@pytest.mark.parametrize("case", [c for c in CLI_CASES if c in YIELDS])
# T-0160 ruling 2: web on by default in live (egress=web, executor web=True)
def test_yield_is_base(base, now, case):
    """A7b (reviewer mutations 4 and 6): each of the three (flag or key, and page with ReGround) without a frame flag
    gives BASE's build for the same input: every ``Session`` and executor keyword (no ``overview_frame``), the
    ``--dry-run`` output and the startup output, except the frame line, which reads the exact ``off (with ...)``
    literal where BASE read ``off (default)``."""
    _, _, line = YIELDS[case]
    for kind in ("dry", "start"):
        got, want = now["cli"][case][kind], base["cli"][case][kind]
        assert got["code"] == want["code"] == 0 and got["raised"] is None, (case, kind, got["err"])
        assert lines(want["out"]) == [OFF_DEFAULT], (
            f"premise: BASE printed off (default); {want['out']}"
        )
        assert lines(got["out"]) == [line], f"T-0152 A5 {case} {kind}:\n{got['out']}"
        got = t0172.admit_reask_run(
            got, kind
        )  # T-0172 R8 pin move: the default reask line and keyword
        got = t0125.admit_live_run(got, kind)  # T-0125 pin move
        got = t0160.admit_live_run(
            got, kind
        )  # T-0160 pin move (ruling 2: web on by default in live)
        got = t0181.run_without_mic_unavailable(got)  # T-0181 A-R13 pin move
        _same(_drop_frame_line(want["out"]), _drop_frame_line(got["out"]), f"A7b {case} {kind} out")
        assert got["out"].splitlines().index(line) == want["out"].splitlines().index(OFF_DEFAULT)
        for key in ("err", "graphs", "session_kwargs", "executor_kwargs"):
            _same(want[key], got[key], f"A7b {case} {kind} {key}")
    for kw in now["cli"][case]["start"]["session_kwargs"]:
        assert "overview_frame" not in kw, kw


def test_yield_rigs_are_base(base, now):
    """A7b over a rig (reading Rd-1): ``--auto-overview`` without a frame flag, driven on a long and a fitting page,
    gives BASE's sections (machine steps, HUD messages, prompts, request bodies, session-log records, turn lines,
    history, wire bodies, browser invocations); premise: on the long page the auto overview was shown (T-0150)."""
    for page in PAGES:
        # T-0155 O-8: the persona sentence only, on the observed side
        got, want = observed(now["rigs"]["auto-flag"][page]), base["rigs"]["auto-flag"][page]
        got = t0172.admit_reask_turn_key(got)  # T-0172 R8 pin move (B7): the default turn-line key
        for key in want:
            _same(want[key], got[key], f"A7b auto-flag {page} {key}")
    # this row also needs the flip: the frame line of the same input
    assert lines(now["cli"]["auto-flag"]["start"]["out"]) == [WITH["auto"]]


# =====================================================================================================
# A6: default behaviour end to end
# =====================================================================================================


def _ext(rec: dict[str, Any]) -> dict[str, Any]:
    return dict(rec.get("extras", {}))


def test_default_long_page_shows_overview(base, now):
    """A6 (integration over the fakes; reviewer mutation 1): a ``cli``-built default live session on a 1.525-screen
    page: step 1's final user message is BASE's default with the live frame replaced by the overview and the T-0134
    label replaced by ``OVERVIEW_FRAME_LABEL`` at 1.5 (typed from T-0151 A2-4); exactly one image; no T-0134 label
    and no ``live_label`` on the generate record; the turn line's ``overview_frame`` is ``shown`` 1 (all five keys);
    the step-1 ``click_at`` reaches the browser on the overview's geometry (executor phash, 838x798) and is
    executed."""
    rig, b = now["rigs"]["default"]["long"], base["rigs"]["default"]["long"]
    base_parts = last_user(b["pass1"])["content"]
    assert base_parts[0]["image_url"]["url"] == s151.builder_url(s151.F0), "premise: BASE shows F0"
    parts = last_user(rig["pass1"])["content"]
    want = [s151.same_shape(base_parts[0]), s151.text_part(s151.LABEL_15), *base_parts[2:]]
    assert parts == want, (
        f"T-0152 A6: the default step-1 final user message; got {s151.kinds(parts)}"
    )
    assert image_urls(rig["pass1"]) == [s151.ov_url()], "A6: exactly one image, the overview"
    assert STEM not in json.dumps(rig["pass1"]), "A6: no T-0134 label"
    gens = [
        r["data"]
        for r in rig["log"]
        if r["kind"] == "llm.request" and r["data"].get("request_id") == "t1-gen"
    ]
    assert gens and all("live_label" not in g for g in gens), gens
    (line,) = rig["records"]
    assert _ext(line).get("overview_frame") == s151.counts(shown=1), _ext(line)
    clicks = [i for i in rig["invocations"] if i["call_id"] == "t1-call"]
    assert len(clicks) == 1 and clicks[0]["name"] == "click_at", rig["invocations"]
    fr = clicks[0]["frame"]
    assert fr is not None and (fr["phash"], fr["width"], fr["height"]) == (
        s151.OV.phash,
        838,
        798,
    ), f"T-0152 A6: the step-1 click on the overview geometry; {fr}"
    statuses = [
        r["data"]["status"]
        for r in rig["log"]
        if r["kind"] == "tool.call" and r["data"].get("call_id") == "t1-call"
    ]
    assert statuses and statuses[-1] == "executed", statuses


def test_default_fitting_page(base, now):
    """A6 (reviewer mutation 1): on a fitting page the default's step-1 body is BASE's default byte for byte, and its
    turn line differs from BASE's by exactly the added ``overview_frame`` key, ``fits`` 1 (all five keys)."""
    # T-0155 O-8: the persona sentence only, on the observed side
    rig, b = observed(now["rigs"]["default"]["fits"]), base["rigs"]["default"]["fits"]
    rig = t0172.admit_reask_turn_key(rig)  # T-0172 R8 pin move (B7): the default turn-line key
    _same(b["pass1"], rig["pass1"], "A6 fitting step-1 body")
    (got,), (want,) = rig["records"], b["records"]
    assert "overview_frame" not in _ext(want), "premise: BASE's default line has no key"
    ext = _ext(got)
    assert ext.pop("overview_frame", None) == s151.counts(fits=1), _ext(got)
    _same({**want, "extras": _ext(want)}, {**got, "extras": ext}, "A6 fitting turn line")


# =====================================================================================================
# A7: off is BASE; other modes; other resolvers
# =====================================================================================================


# T-0125 pin move: the default live search line, startup line, kit
@pytest.mark.parametrize("case, src", [("no-frame-flag", "flag"), ("no-frame-key", "profile")])
# T-0160 ruling 2: web on by default in live (egress=web, executor web=True)
def test_off_is_base(base, now, case, src):
    """A7a (green at BASE; adversarial, reviewer mutation 5: off passing ``overview_frame=False`` or adding a turn-line
    key): ``--no-overview-frame`` and ``overview_frame = false`` give BASE's default live session, on a long page and
    on a fitting page: every section (request bodies, turn lines, session-log records, HUD messages, prompts, machine
    steps, history, browser invocations), the ``Session`` and executor keywords and the startup output, except the
    frame line, ``off (flag)``/``off (profile)`` where BASE read ``off (default)``."""
    for page in PAGES:
        # T-0155 O-8: the persona sentence only, on the observed side
        got, want = observed(now["rigs"][case][page]), base["rigs"]["default"][page]
        got = t0172.admit_reask_turn_key(got)  # T-0172 R8 pin move (B7): the default turn-line key
        for key in want:
            _same(want[key], got[key], f"A7a {case} {page} {key}")
        assert all("overview_frame" not in _ext(r) for r in got["records"]), got["records"]
    assert image_urls(now["rigs"][case]["long"]["pass1"]) == [s151.builder_url(s151.F0)], "premise"
    for kind in ("dry", "start"):
        got, want = observed(now["cli"][case][kind]), base["cli"]["default"][kind]  # T-0155 O-8
        assert lines(want["out"]) == [OFF_DEFAULT], "premise: BASE's default line"
        assert lines(got["out"]) == [PREFIX + f"off ({src})"], got["out"]
        got = t0172.admit_reask_run(
            got, kind
        )  # T-0172 R8 pin move: the default reask line and keyword
        got = t0125.admit_live_run(got, kind)  # T-0125 pin move
        got = t0160.admit_live_run(
            got, kind
        )  # T-0160 pin move (ruling 2: web on by default in live)
        got = t0181.run_without_mic_unavailable(got)  # T-0181 A-R13 pin move
        _same(_drop_frame_line(want["out"]), _drop_frame_line(got["out"]), f"A7a {case} {kind} out")
        for key in ("code", "err", "graphs", "session_kwargs", "executor_kwargs"):
            _same(want[key], got[key], f"A7a {case} {kind} {key}")


# T-0125 pin move: ToolKit.web_search (None with search off)
def test_other_modes_unchanged(base, now):
    """A7c (green at BASE): for ``off`` and ``dry_run`` (plain, loop off, a live profile under ``--tools off``/``--tools
    dry-run``) T-0134c's sections (the ``--dry-run`` output, a build's output, the client and session keywords, the
    executors made, the system message, one request body) equal BASE's; no ``overview_frame`` keyword, no frame
    line. The executor's own action strings equal BASE's."""
    for case, *_ in t0134c.OTHER:
        for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
            # T-0155 O-8: the persona sentence only, on the observed side
            got = observed(now["other"][case][key])
            got = t0172.admit_reask_other(case, key, got)  # T-0172 R8 pin move (B9)
            if key == "session_kwargs":
                got = t0125.kit_without_web_search(got)  # T-0125 pin move
                got = t0181.without_mic_unavailable(got)  # T-0181 A-R13 pin move
            _same(base["other"][case][key], got, f"A7c {case} {key}")
        assert all("overview_frame" not in kw for kw in now["other"][case]["session_kwargs"])
        assert "overview frame:" not in now["other"][case]["dry"]["out"], case
    _same(base["executor"], now["executor"], "A7c executor actions")


def test_other_resolvers_unchanged(base, now):
    """A7d (green at BASE): ``grounding_of``, ``feedback_of``, ``recovery_of``, ``page_overview_of`` and
    ``auto_overview_of`` and their lines equal BASE's over a grid of profiles, namespaces (this task's dest included)
    and every mode."""
    assert set(now["grid"]) == set(base["grid"]) and len(now["grid"]) == 9 * 8 * 3, "premise"
    for k in now["grid"]:
        _same(base["grid"][k], now["grid"][k], f"A7d resolvers at {k}")


# =====================================================================================================
# A8: the T-0151 harness at HEAD
# =====================================================================================================


def test_t0151_harness_refuses_at_head(tmp_path, monkeypatch, capsys):
    """A8 (a frozen record): with HEAD's real default, the T-0151 harness's ``live_default(lean)["overview_frame"] is
    True``, its ``live_default_findings`` names ``overview_frame``, and ``main`` exits ``EXIT_REFUSED`` (2) before any
    socket, leaving no out-dir, naming ``overview_frame``. The rig is the T-0151 harness test's no-socket ``_rig``,
    imported; it pins BASE_B's default (R3), so the real constant is put back first."""
    real = cli.LIVE_DEFAULT_OVERVIEW_FRAME
    rig = t0151h._rig(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr(cli, "LIVE_DEFAULT_OVERVIEW_FRAME", real)
    h = rig.h
    d = dict(h.live_default(load().profiles["lean"]))
    assert d["overview_frame"] is True, f"T-0152 A8: HEAD's default has the frame on; {d}"
    found = h.live_default_findings(d)
    assert any("overview_frame" in f for f in found), found
    rc = h.main(["--base-url", t0151h.t31.VALID_URL, "--out-dir", str(rig.out)])
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], rig.sockets
    assert not rig.out.exists(), "the refusal wrote the out-dir"
    assert "overview_frame" in err, err


# =====================================================================================================
# A9 and A7e: docs and frozen paths
# =====================================================================================================


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0152():
    """A9 (reading Rd-3): docs/02 §5 at ``TIP_IMPL`` holds exactly one paragraph starting ``**Spec delta
    (T-0152).**`` naming ``LIVE_DEFAULT_OVERVIEW_FRAME``, ``--no-overview-frame``, ``off (with --reground)`` and
    ``94/100``; removing it leaves BASE's docs/02 paragraph for paragraph (the T-0151, T-0150 and T-0140 paragraphs
    verbatim). Both sides are literals, never the working tree."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert TIP_IMPL != BASE_SHA, "T-0152 A9: TIP_IMPL is not bumped to the implementation tip yet"
    doc, base_doc = _doc02_at(TIP_IMPL), _doc02_at(BASE_SHA)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0152 A9: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    flat = para.replace("`", "")
    missing = [
        k
        for k in (
            "LIVE_DEFAULT_OVERVIEW_FRAME",
            "--no-overview-frame",
            "off (with --reground)",
            "94/100",
        )
        if k not in flat
    ]
    assert not missing, f"T-0152 A9: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), "T-0152 A9: docs/02 changes only by the one paragraph"
    assert doc.count(HEADER) == 1


#: A7e's list, written out from the criterion (A8: every file under ``scripts/``)
FROZEN = (
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/contracts",
    "tests/fakes",
    "tests/unit/_live_browser_support.py",
    "local_astra/perception",
    "local_astra/pages",
    "hud",
    "services",
    "scripts",
)


def test_frozen_paths_over_closed_range():
    """A7e/A8 (green at BASE; reviewer mutation 7): the frozen paths are byte-identical over ``BASE_SHA..TIP_IMPL``
    (two literals, never the working tree), no file added or removed, and each exists at BASE; ``LIVE_TOOL_GUIDANCE``,
    ``tools_v1.json`` and ``tools_v1_overview.json`` keep their shas at both ends. Power: the same predicate sees
    T-0111g's ``versions.lock`` change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), f"{BASE_SHA} is not an ancestor of {TIP_IMPL}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], f"T-0152 A7e: frozen paths differ in {BASE_SHA}..{TIP_IMPL}: {changed}"
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    for name, sha in (("tools_v1.json", V1_SHA), ("tools_v1_overview.json", OV_SHA)):
        for rev in (BASE_SHA, TIP_IMPL):
            blob = gr._git("show", f"{rev}:local_astra/toolcall/{name}")
            assert blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest() == sha
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN), "power"
