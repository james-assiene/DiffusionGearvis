"""T-0131 A3 and C1 (cli side): ``--visual-tokens``, the ``--dry-run`` suffix, the client's budget, the turn-line
key and the live guidance (docs/tasks/T-0131.md).

**The interfaces these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``astra session --visual-tokens {280,560,1120}`` (anything else exits 2) overrides ``Profile.visual_tokens``;
- ``cli`` constructs ``VllmClient(..., visual_tokens=N)`` by keyword (``None`` or absent when unset);
- the session receives the budget as the keyword ``Session(..., visual_tokens=N)`` ("only for logging"; the task
  does not name the seam, this is the reading), and every turn line in every mode carries
  ``extras["visual_tokens"] == N`` when set, no such key when unset; ``Regime`` gains nothing;
- the ``--dry-run`` tools line is BASE's plus `` visual_tokens=N`` appended at its end, in every mode, when set
  (the task's "the ``--dry-run`` line gains ` visual_tokens=N`"; the reading is: the ``tools=`` line, at the end);
  every other printed line is unchanged;
- with ``tools = live`` the session ``cli`` builds has ``LIVE_TOOL_GUIDANCE`` last in its system message
  (observed through ``session.prompts``, whatever the route), and with ``off``/``dry_run`` it does not.

The rigs are T-0111g's and T-0092b's (``test_cli_tools_live``, ``_tools_switch_support``), imported, not copied.

Every row is ``pending(task="T-0131")``; at BASE (f30365d) each fails on a named ``AssertionError`` (the flag, the
profile field or the constant is missing).

Regime: CPU over fakes; no browser, no service, nothing served; no measurement.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.config import Profile
from local_astra.contracts import events as E
from tests.unit import _t0160_pins as t0160

# T-0202 D3 re-pin: every tools-line assertion reads the line through ``admitted_tools_lines`` (the one default
# `` wait=on(default)`` token removed, the wait set's sha12 read back as v1's); identity with the wait tool off
from tests.unit import _t0202_pins as t0202
from tests.unit import _tool_turn_support as tts
from tests.unit import _tools_switch_support as T
from tests.unit._live_browser_support import live_rig
from tests.unit._tool_turn_support import answer, tool_rig
from tests.unit.test_cli_session import CLI, explode_clients, flags_of, run_cli
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live

SHA12 = T.TOOLS_SHA12
#: BASE's lines (f30365d), written out from ``cli.tools_line`` and the T-0111g/T-0114 rows
BASE_LINES = {
    "off": "tools=off",
    "dry_run": f"tools=dry_run native:{SHA12} repair=on loop=on(8)",
    "live": f"tools=live native:{SHA12} repair=on loop=on(8) browser=headless egress=loopback viewport=1280x800",
}
MODE_FLAGS = {
    "off": [],
    "dry_run": ["--tools", "dry-run"],
    "live": ["--tools", "live", "--browser", "headless"],
}


def require_seams() -> None:
    assert "--visual-tokens" in flags_of(CLI), (
        "T-0131 A3: astra session --visual-tokens does not exist yet"
    )
    assert "visual_tokens" in Profile.model_fields, (
        "T-0131 A1: Profile.visual_tokens does not exist yet"
    )


def guidance() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE", None)
    assert isinstance(value, str), "T-0131 C1: prompt_builder.LIVE_TOOL_GUIDANCE does not exist yet"
    return value


def dry_run(
    tmp_path: Path, capsys: Any, monkeypatch: Any, mode: str, *extra: str, body: str = ""
) -> str:
    """Stdout of ``astra session <mode> --dry-run``; exit 0 required."""
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *MODE_FLAGS[mode], *extra, config_body=body), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, f"{mode} {extra} --dry-run exits 0; got {code}\n{out}{err}"
    return out


def tools_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("tools=")]


def build(
    tmp_path: Path, capsys: Any, monkeypatch: Any, mode: str, *extra: str, body: str = ""
) -> T.Calls:
    """Build the graph through ``cli`` over the fakes (nothing served); the recorded constructor calls."""
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *MODE_FLAGS[mode], *extra, config_body=body))
    out, err = _out(capsys)
    assert code == 0, f"{mode} {extra}: exit 0; got {code}\n{out}{err}"
    assert calls.graphs and calls.vllm_kwargs and calls.session_kwargs, (
        "premise: the graph was built"
    )
    return calls


# --------------------------------------------------------------------------------------------- A3: the flag


def test_flag_overrides_profile(tmp_path, capsys, monkeypatch):
    """A3: the profile's ``visual_tokens`` is used when no flag is given, and ``--visual-tokens`` overrides it, both
    on the ``--dry-run`` line and in the client ``cli`` builds; 280 given explicitly counts as set. Junk on the
    flag (70, 0, 1121, ``many``) exits 2."""
    require_seams()
    body = "visual_tokens = 560\n"
    assert t0202.admitted_tools_lines(dry_run(tmp_path, capsys, monkeypatch, "off", body=body)) == [
        "tools=off visual_tokens=560"
    ]
    got = dry_run(tmp_path, capsys, monkeypatch, "off", "--visual-tokens", "1120", body=body)
    assert t0202.admitted_tools_lines(got) == ["tools=off visual_tokens=1120"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "off", "--visual-tokens", "280")
    assert t0202.admitted_tools_lines(got) == ["tools=off visual_tokens=280"], got
    # dry_run's exploding stand-ins (explode_clients) must not reach the builds below (found by the A/C
    # implementer at 24db2e2: _build_graph raised "criterion 15" on BrowserFrameSource)
    monkeypatch.undo()
    calls = build(tmp_path, capsys, monkeypatch, "off", "--visual-tokens", "1120", body=body)
    assert calls.vllm_kwargs[-1].get("visual_tokens") == 1120, calls.vllm_kwargs[-1]
    calls = build(tmp_path, capsys, monkeypatch, "off", body=body)
    assert calls.vllm_kwargs[-1].get("visual_tokens") == 560, calls.vllm_kwargs[-1]
    for junk in ("70", "0", "1121", "many"):
        explode_clients(monkeypatch)
        code = run_cli([*argv(tmp_path, "--visual-tokens", junk), "--dry-run"])
        _out(capsys)
        assert code == 2, f"T-0131 A3: --visual-tokens {junk} exits 2; got {code}"


#: T-0132 re-scope R1 (docs/tasks/T-0132.md, human ruling 2026-10-22: live defaults to 1120 + ReGround on). The
#: ``live`` parameter of the rows below asserts the new spec (group A, activated 2026-10-22); ``off`` and ``dry_run``
#: keep asserting BASE behaviour. T-0133 R2 then turned the live ReGround default off again (1120 kept).
LIVE_T0132 = "live"
#: T-0133 re-scope R2 (docs/tasks/T-0133.md, human ruling 2026-10-22: live defaults to 1120 with ReGround *off*):
#: the resolved suffix of the default live line (A4b), and that of a live line with ReGround on
LIVE_DEFAULT_SUFFIX = " visual_tokens=1120"
LIVE_REGROUND_SUFFIX = " visual_tokens=1120 reground=on"
#: T-0183 re-scope (docs/tasks/T-0183.md A3a, human ruling 2026-10-28: 1120 by default in every tools mode): the
#: ``off``/``dry_run`` line with no budget set is BASE's plus this resolved suffix, as the live line has been since
#: T-0132 A4b; and ``cli`` builds the client and the session at 1120 there too (A2). Written out from the criterion.
DEFAULT_SUFFIX = " visual_tokens=1120"
DEFAULT_BUDGET = 1120
GROUNDING_PREFIX = "astra session: grounding: "


@pytest.mark.parametrize(
    "mode",
    [
        "off",
        "dry_run",
        # T-0160 ruling 2: web on by default in live (the live line says egress=web)
        LIVE_T0132,
    ],
)
def test_dry_run_suffixes_only_when_set(tmp_path, capsys, monkeypatch, mode):
    """A3 (mutation 2's cli half): unset, the ``--dry-run`` tools line is BASE's, byte for byte; set, it is BASE's
    plus `` visual_tokens=560`` at its end; every other printed line is the same in both runs.

    T-0132 R1 (live): unset, the live line is BASE's plus `` visual_tokens=1120 reground=on`` (A4b); with
    ``--visual-tokens 560`` BASE's plus `` visual_tokens=560 reground=on``; every other line is the same except the
    T-0132 ``grounding`` line, which states the sources and so differs.

    T-0133 R2 (live): ReGround is off by default, so unset the tail is `` visual_tokens=1120`` and with
    ``--visual-tokens 560`` it is `` visual_tokens=560``.

    T-0183 re-scope (A3a): unset, the ``off`` and ``dry_run`` lines are BASE's plus `` visual_tokens=1120`` too."""
    require_seams()
    unset = dry_run(tmp_path, capsys, monkeypatch, mode)
    tail = LIVE_DEFAULT_SUFFIX if mode == "live" else DEFAULT_SUFFIX
    # T-0160 pin move (C-2): the live line reads egress=web, nothing else changes
    want = t0160.web_line(BASE_LINES[mode]) if mode == "live" else BASE_LINES[mode]
    assert t0202.admitted_tools_lines(unset) == [want + tail], unset
    got = dry_run(tmp_path, capsys, monkeypatch, mode, "--visual-tokens", "560")
    set_tail = " visual_tokens=560"
    assert t0202.admitted_tools_lines(got) == [want + set_tail], got

    def rest(out: str) -> list[str]:
        return [
            line
            for line in out.splitlines()
            if not line.startswith("tools=") and not line.startswith(GROUNDING_PREFIX)
        ]

    assert rest(got) == rest(unset), "T-0131 A3: only the tools line gains the suffix"


@pytest.mark.parametrize("mode", ["off", "dry_run", LIVE_T0132])
def test_client_constructed_with_budget(tmp_path, capsys, monkeypatch, mode):
    """A3: ``cli`` builds ``VllmClient(..., visual_tokens=560)`` with the flag (by keyword, the detokeniser and the
    endpoint as before), ``None``/absent without it, and the session receives the same value
    (``Session(..., visual_tokens=...)``). T-0132 R1 (live): without the flag both are 1120 (the live default).
    T-0183 re-scope (A2): without the flag both are 1120 in ``off`` and ``dry_run`` too."""
    require_seams()
    calls = build(tmp_path, capsys, monkeypatch, mode, "--visual-tokens", "560")
    kw = calls.vllm_kwargs[-1]
    assert kw.get("visual_tokens") == 560, (
        f"T-0131 A3: VllmClient gets visual_tokens=560; kwargs {kw}"
    )
    assert ("detokenizer" in kw) is True, "premise: the detokeniser keyword is still passed"
    assert calls.session_kwargs[-1].get("visual_tokens") == 560, (
        "T-0131 A3 (pinned seam): the session receives the budget as Session(..., visual_tokens=560)"
    )
    calls = build(tmp_path, capsys, monkeypatch, mode)
    unset = 1120 if mode == "live" else DEFAULT_BUDGET  # T-0132 R1; T-0183 A2 outside live
    assert calls.vllm_kwargs[-1].get("visual_tokens") == unset, calls.vllm_kwargs[-1]
    assert calls.session_kwargs[-1].get("visual_tokens") == unset, calls.session_kwargs[-1]


# --------------------------------------------------------------------------------------------- A3: the turn line


def with_budget(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    """Every rig session is built with ``visual_tokens=budget`` (the pinned seam)."""
    real = tts.Session
    assert "visual_tokens" in inspect.signature(real.__init__).parameters, (
        "T-0131 A3: Session(..., visual_tokens: int | None = None) does not exist yet (pinned seam)"
    )

    class Budgeted(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            k.setdefault("visual_tokens", budget)
            super().__init__(*a, **k)

    monkeypatch.setattr(tts, "Session", Budgeted)


async def _one_turn(tmp_path: Path, mode: str) -> Any:
    if mode == "live":
        async with live_rig(tmp_path, script=[answer("It is a shop. ")]) as rig:
            rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="what is on the page"))
            await rig.turn_done()
            return rig.record("t1")
    kit: Any = None if mode == "off" else "default"
    async with tool_rig(tmp_path, kit=kit, script=[answer("It is a shop. ")]) as rig:
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="what is on the page"))
        await rig.turn_done()
        return rig.record("t1")


@pytest.mark.parametrize("mode", ["off", "dry_run", "live"])
async def test_turn_line_key_only_when_set(tmp_path, monkeypatch, mode):
    """A3: with a budget every turn line in every mode carries ``visual_tokens: 560`` (``TurnRecord.extras``); the
    ``Regime`` gains no field. Unset (the next row's premise), the key is absent."""
    with_budget(monkeypatch, 560)
    record = await _one_turn(tmp_path / "set", mode)
    assert record.extras.get("visual_tokens") == 560, (
        f"T-0131 A3: the {mode} turn line carries visual_tokens 560; extras {record.extras}"
    )
    assert "visual_tokens" not in record.regime.model_dump(), "A3: the budget is not a Regime field"
    monkeypatch.undo()
    plain = await _one_turn(tmp_path / "unset", mode)
    assert "visual_tokens" not in plain.extras, plain.extras


# --------------------------------------------------------------------------------------------- C1: who gets the guidance


def guidance_v2() -> str:
    """T-0174 A2's v2 text, read from ``prompt_builder.LIVE_GUIDANCE`` (never from ``config``'s default)."""
    value = getattr(pb, "LIVE_GUIDANCE", {}).get("v2")
    assert isinstance(value, str), "T-0174 A2: prompt_builder.LIVE_GUIDANCE['v2'] does not exist"
    assert value != guidance(), "premise: v2 is not v1's text"
    return value


def guidance_v3() -> str:
    """T-0191 A1's v3 text, read from ``prompt_builder.LIVE_GUIDANCE`` (never from ``config``'s default)."""
    value = getattr(pb, "LIVE_GUIDANCE", {}).get("v3")
    assert isinstance(value, str), "T-0191 A3: prompt_builder.LIVE_GUIDANCE['v3'] does not exist"
    assert value.startswith(guidance_v2() + " ") and value != guidance_v2(), (
        "premise: v3 is v2 + ' ' + S"
    )
    return value


#: T-0174 D4 / T-0191 C3 (re-scopes of C1 at each default flip): ``live`` is the default (no flag, web mode: v3
#: since T-0191 C1, v2 since T-0174 D1), ``live-no-web`` the default with ``--no-web`` (v2's object, T-0191 A4),
#: ``live-v2`` is ``--live-guidance v2`` (the pre-flip default), ``live-v1`` is ``--live-guidance v1`` (T-0131's
#: text).
@pytest.mark.parametrize(
    "mode",
    [
        "off",
        "dry_run",
        "live",
        "live-no-web",
        "live-v2",
        "live-v1",
    ],
)
def test_live_gets_guidance_dry_run_does_not(tmp_path, capsys, monkeypatch, mode):
    """C1 (mutation 3), re-scoped by T-0174 D4 and T-0191 C3: the session ``cli`` builds for ``tools = live`` with
    no flag (web mode) has ``LIVE_GUIDANCE["v3"]`` last in its system message, once (and not v1); with ``--no-web``
    or ``--live-guidance v2`` it has ``LIVE_GUIDANCE["v2"]`` last, once, and v3's added sentence nowhere; with
    ``--live-guidance v1`` it has ``LIVE_TOOL_GUIDANCE`` (v1) last, once (and not v2); for ``dry_run`` and ``off``
    none is present (``dry_run`` ends with the native instructions, as at BASE)."""
    v1, v2 = guidance(), guidance_v2()
    live_flags = {
        "live": ([], "v3"),
        "live-no-web": (["--no-web"], "v2"),
        "live-v2": (["--live-guidance", "v2"], "v2"),
        "live-v1": (["--live-guidance", "v1"], "v1"),
    }
    if mode in live_flags:
        flags, version = live_flags[mode]
        calls = build(tmp_path, capsys, monkeypatch, "live", *flags)
        want = {"v1": v1, "v2": v2, "v3": guidance_v3() if version == "v3" else ""}[version]
        others = [v2] if version == "v1" else [v1]
    else:
        calls = build(tmp_path, capsys, monkeypatch, mode)
    text = calls.graphs[-1].session.prompts.system_message().text()
    if mode in live_flags:
        text = t0202.admit_system(
            text, mode
        )  # T-0202 D3 re-pin: the default wait sentence set aside (once)
        assert text.endswith(want) and text.count(want) == 1, (
            f"T-0131 C1 / T-0174 D4 / T-0191 C1 ({mode}): the live session's system message ends with "
            f"LIVE_GUIDANCE[{version}]; it ends {text[-160:]!r}"
        )
        for other in others:
            assert other not in text, f"T-0174 D4 ({mode}): the other version's text is absent"
        if version == "v2":
            v3 = getattr(pb, "LIVE_GUIDANCE", {}).get("v3")
            if isinstance(v3, str):
                assert v3[len(v2) :].strip() not in text, (
                    f"T-0191 A4 ({mode}): v3's sentence is absent"
                )
        assert text[: len(text) - len(want)].rstrip().endswith(pb.NATIVE_TOOL_INSTRUCTIONS)
    else:
        assert v1 not in text and v2 not in text, f"T-0131 C1: no guidance with tools = {mode}"
        if mode == "dry_run":
            assert text.endswith(pb.NATIVE_TOOL_INSTRUCTIONS)


LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'


@pytest.mark.parametrize("override", [["--tools", "dry-run"], ["--tools", "off"]])
def test_live_profile_overridden_carries_no_guidance(tmp_path, capsys, monkeypatch, override):
    """C1 precedence (ruling at 21361bd, an A/C gap): the guidance follows the *resolved* tools mode, not the
    profile's key. A profile with ``tools = "live"`` overridden by ``--tools dry-run`` or ``--tools off`` builds a
    session with no live guidance (neither v1 nor v2, hence not v3, which contains v2). Premise, re-scoped by T-0174
    D4 and T-0191 C3: the same profile with no flag is live (web mode) and carries the default guidance,
    ``LIVE_GUIDANCE["v3"]``, so the profile alone reaches the guidance and the override is what removes it."""
    v1, v2 = guidance(), guidance_v2()
    v3 = guidance_v3()
    calls = build(tmp_path / "plain", capsys, monkeypatch, "off", body=LIVE_PROFILE)
    text = calls.graphs[-1].session.prompts.system_message().text()
    text = t0202.admit_system(
        text, "live profile"
    )  # T-0202 D3 re-pin: the default wait sentence set aside (once)
    assert text.endswith(v3) and text.count(v3) == 1 and v1 not in text, (
        "T-0191 C1 premise: the live profile alone gives the default (v3) live guidance"
    )
    monkeypatch.undo()
    calls = build(tmp_path / "over", capsys, monkeypatch, "off", *override, body=LIVE_PROFILE)
    text = calls.graphs[-1].session.prompts.system_message().text()
    assert v1 not in text and v2 not in text, (
        f"T-0131 C1: a live profile under {override} carries no guidance"
    )
    if override[1] == "dry-run":
        assert text.endswith(pb.NATIVE_TOOL_INSTRUCTIONS), "premise: the override reached dry_run"
    else:
        assert pb.NATIVE_TOOL_INSTRUCTIONS not in text, "premise: the override reached off"


# --------------------------------------------------------------------------------------------- B9: --reground (pending)
#
# Group B's cli rows (orchestrator, 2026-10-22). The interfaces: ``astra session --reground/--no-reground``
# overrides ``Profile.reground``; the live ``--dry-run`` tools line gains `` reground=on`` at its end when it is
# on; ``cli`` builds ``Session(..., reground=<resolved>)``. Reading (reported): ``--reground`` with tools other than
# ``live`` is refused with a non-zero exit naming ``live``, as the profile key is at load (A1).

LIVE_REGROUND = 'tools = "live"\nreground = true\n'


def require_reground_flag() -> None:
    flags = flags_of(CLI)
    assert "--reground" in flags and "--no-reground" in flags, (
        "T-0131 B9: astra session --reground/--no-reground does not exist yet"
    )


# T-0160 ruling 2: web on by default in live (egress=web)
def test_reground_dry_run_suffix(tmp_path, capsys, monkeypatch):
    """B9: the live ``--dry-run`` line is BASE's plus `` reground=on`` with ``--reground`` or the profile's
    ``reground = true``; ``--no-reground`` over the profile's true, and the default, print BASE's line. With a budget
    too, both suffixes are present after BASE's line.

    T-0132 R1 (human ruling 2026-10-22): the unset budget is now 1120 and the unset ReGround on, so ``--reground``,
    the profile's true and the default all print BASE's plus `` visual_tokens=1120 reground=on``, and
    ``--no-reground`` over the profile's true prints BASE's plus `` visual_tokens=1120``.

    T-0133 R2 (human ruling 2026-10-22: ReGround off by default): ``--reground`` and the profile's true print
    BASE's plus `` visual_tokens=1120 reground=on``; ``--no-reground`` over the profile's true, and the default,
    print BASE's plus `` visual_tokens=1120``."""
    require_reground_flag()
    base = t0160.web_line(BASE_LINES["live"])  # T-0160 pin move (C-2)
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--reground")
    assert t0202.admitted_tools_lines(got) == [base + LIVE_REGROUND_SUFFIX], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", body=LIVE_REGROUND)
    assert t0202.admitted_tools_lines(got) == [base + LIVE_REGROUND_SUFFIX], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--no-reground", body=LIVE_REGROUND)
    assert t0202.admitted_tools_lines(got) == [base + " visual_tokens=1120"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live")
    assert t0202.admitted_tools_lines(got) == [base + " visual_tokens=1120"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--reground", "--visual-tokens", "560")
    (line,) = t0202.admitted_tools_lines(got)
    assert line.startswith(base + " ") and set(line[len(base) :].split()) == {
        "visual_tokens=560",
        "reground=on",
    }, line


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_reground_flag_refused_without_live(tmp_path, capsys, monkeypatch, mode):
    """B9/A1 (the silent no-op is the failure): ``--reground`` with ``off`` or ``dry_run`` is refused with exit
    code 2 (the orchestrator's ruling on reviewer B's A/C approval: "cli refuses with exit 2, naming ``live``";
    pinned exactly at the group B review round, not just non-zero) naming ``live``; ``--no-reground`` with the
    same mode is accepted (the too-wide direction), its line BASE's plus T-0183 A3a's `` visual_tokens=1120``."""
    require_reground_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    explode_clients(monkeypatch)
    code = run_cli([*argv(tmp_path, *MODE_FLAGS[mode], "--reground"), "--dry-run"])
    out, err = _out(capsys)
    assert code == 2 and "live" in (out + err), (
        f"T-0131 B9: refused with exit 2 naming live; {code}\n{out}{err}"
    )
    got = dry_run(tmp_path, capsys, monkeypatch, mode, "--no-reground")
    assert t0202.admitted_tools_lines(got) == [BASE_LINES[mode] + DEFAULT_SUFFIX], got


@pytest.mark.parametrize("override", [["--tools", "dry-run"], ["--tools", "off"]])
def test_profile_reground_refused_when_tools_not_live(tmp_path, capsys, monkeypatch, override):
    """B9 (ruling at 21361bd; ReGround never silently does nothing): a profile with ``tools = "live"`` and
    ``reground = true`` run with ``--tools dry-run`` or ``--tools off`` resolves ``reground`` true and the tools mode
    not live, so ``cli`` exits 2 naming ``live``, with ``--dry-run`` and without, and builds no graph. Companion (the
    too-wide direction): the same profile with no tools override is accepted."""
    for extra in (["--dry-run"], []):
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *override, config_body=LIVE_REGROUND), *extra])
        out, err = _out(capsys)
        assert code == 2 and "live" in (out + err), (
            f"T-0131 B9: reground from the profile with {override} {extra} exits 2 naming live; "
            f"got {code}\n{out}{err}"
        )
        assert not calls.graphs, f"T-0131 B9: a refused run builds no graph ({override} {extra})"
        monkeypatch.undo()
    calls = build(tmp_path, capsys, monkeypatch, "off", "--browser", "headless", body=LIVE_REGROUND)
    assert calls.graphs, "the same profile without an override is accepted"


def test_cli_passes_reground_to_session(tmp_path, capsys, monkeypatch):
    """B9: ``cli`` builds ``Session(..., reground=True)`` with ``--reground`` (and with the profile's true), and
    ``reground`` False or absent with ``--no-reground`` over the profile's true. T-0132 R1 (human ruling
    2026-10-22): by default (live, no flag, no profile key) it is now ``True``. T-0133 R2 (human ruling
    2026-10-22): by default it is False or absent again (live defaults to ReGround off)."""
    require_reground_flag()
    calls = build(tmp_path, capsys, monkeypatch, "live", "--reground")
    assert calls.session_kwargs[-1].get("reground") is True, calls.session_kwargs[-1]
    calls = build(tmp_path, capsys, monkeypatch, "live", body=LIVE_REGROUND)
    assert calls.session_kwargs[-1].get("reground") is True
    calls = build(tmp_path, capsys, monkeypatch, "live", "--no-reground", body=LIVE_REGROUND)
    assert not calls.session_kwargs[-1].get("reground", False)
    calls = build(tmp_path, capsys, monkeypatch, "live")
    assert not calls.session_kwargs[-1].get("reground", False), (
        f"T-0133 R2: live ReGround defaults to off; {calls.session_kwargs[-1]}"
    )
