"""T-0174 group A, A3 and A4: the live-guidance version in ``config`` and ``cli`` (docs/tasks/T-0174.md).

**The interfaces these rows pin** (the task's; readings stated where it is silent):

- A3: ``config.LIVE_GUIDANCE_VERSIONS == ("v1", "v2")``, ``config.DEFAULT_LIVE_GUIDANCE == "v1"`` in group A,
  ``"v2"`` since group D (D1, James's ruling 2026-10-27; the D4 re-scoped rows and the D rows were
  written ahead as pending and activated for D's implementation),
  ``Profile.live_guidance: str | None = None`` (``None`` in every built-in profile). Load refuses an unknown value
  naming ``v1`` and ``v2`` (strings other than the two, any case variant, and non-strings), and any non-``None``
  value when ``tools`` is not ``live`` (``off``, ``dry_run``, or no ``tools`` key).
- A4: ``--live-guidance {v1,v2}`` (argparse dest ``live_guidance``, read with ``getattr`` like the other resolvers);
  ``cli.live_guidance_of(profile, args, mode) -> cli.LiveGuidance`` with attributes ``value`` and ``source``: flag,
  else profile, else ``DEFAULT_LIVE_GUIDANCE``, ``(None, "unset")`` outside live; pure (no socket, no file, no
  ``local_astra.computer`` import).
- A4: the flag with a resolved mode other than live (``--tools off``, ``--tools dry-run``, also over a live profile)
  exits 2 with ``cli``'s own refusal (not argparse's usage error), naming the flag, before any graph, page server or
  browser; a live profile's ``live_guidance`` key under a ``--tools off|dry-run`` override is refused the same way
  (reading, by analogy with every other live-only key, T-0137 B-9 and T-0172 B9).
- A4: a live session is built with ``Session(..., tool_guidance=LIVE_GUIDANCE[value])`` (the mapping's object).
- A4: with the default source no ``live guidance`` line is printed (the BASE byte-identity of the whole output is
  ``test_t0174_v1_identity.py``); a flag or profile source prints exactly
  ``astra session: live guidance: <v> (<source>)`` once, on stdout, immediately before T-0172's reask line, which
  stays the last tools-on line (``--dry-run``: no ``astra session:`` line after it; real start: the regime line
  right after it). Reading: "immediately before" means the previous line of the same stream (stdout, where the
  reask line is printed).

``test_session_gets_guidance[default]`` and ``test_default_prints_no_line`` were green at BASE and stay green. Every
other row was written ahead as pending, failed at BASE on a named ``AssertionError``, and was activated for group A
(87172bda).

Regime: CPU over fakes, nothing served; n = 1, deterministic.
"""

from __future__ import annotations

import argparse
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
from tests.unit import _t0202_pins as t0202
from tests.unit import _tools_switch_support as T
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit._t0172_pins import REASK_LINE, REASK_PREFIX
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import MODE_FLAGS
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_t0132 import _toml
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out

FLAG = "--live-guidance"
FIELD = "live_guidance"
PREFIX = "astra session: live guidance: "
REGIME_PREFIX = "astra session: regime:"
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'


# --------------------------------------------------------------------------------------------- seams


def require_flag() -> None:
    assert FLAG in flags_of(CLI), f"T-0174 A4: astra session {FLAG} does not exist yet"


def require_field() -> None:
    assert FIELD in Profile.model_fields, f"T-0174 A3: Profile.{FIELD} does not exist yet"


def require_resolver() -> Any:
    fn = getattr(cli, "live_guidance_of", None)
    assert callable(fn), "T-0174 A4: cli.live_guidance_of(profile, args, mode) does not exist yet"
    assert isinstance(getattr(cli, "LiveGuidance", None), type), (
        "T-0174 A4: cli.LiveGuidance does not exist yet"
    )
    return fn


def guidance_map() -> Any:
    m = getattr(pb, "LIVE_GUIDANCE", None)
    assert m is not None, "T-0174 A2: prompt_builder.LIVE_GUIDANCE does not exist yet"
    return m


def pair(r: Any) -> tuple[Any, Any]:
    assert isinstance(r, cli.LiveGuidance), type(r)
    return r.value, r.source


def guidance_lines(text: str) -> list[str]:
    return [x for x in text.splitlines() if x.startswith(PREFIX)]


def _load_error(tmp_path: Path, body: str) -> str | None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        load(_toml(tmp_path, body))
    except ValueError as exc:
        return str(exc)
    return None


def dry_out_err(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> tuple[str, str]:
    """``astra session --tools live --browser headless <flags> --dry-run`` over the fakes: (stdout, stderr)."""
    from tests.unit.test_cli_session import explode_clients

    tmp_path.mkdir(parents=True, exist_ok=True)
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *MODE_FLAGS["live"], *flags, config_body=body), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, f"{flags} {body!r} --dry-run exits 0; got {code}\n{out}{err}"
    return out, err


def assert_line_before_reask(out: str, want: str, where: str) -> None:
    """A4: ``want`` once, right before the one reask line, which stays the last tools-on line."""
    lines = out.splitlines()
    got = guidance_lines(out)
    assert got == [want], f"T-0174 A4 ({where}): exactly one {want!r}; printed {got}\n{out}"
    found = [i for i, x in enumerate(lines) if x.startswith(REASK_PREFIX)]
    assert len(found) == 1, f"premise ({where}): one reask line; {found}"
    i = found[0]
    assert i > 0 and lines[i - 1] == want, (
        f"T-0174 A4 ({where}): the live guidance line is immediately before the reask line; before it "
        f"{lines[i - 1] if i else None!r}"
    )
    after = lines[i + 1 :]
    if where == "dry":
        assert not [x for x in after if x.startswith("astra session: ")], (
            f"T-0174 A4 (dry): the reask line stays the last tools-on line; after it {after}"
        )
    else:
        assert after and after[0].startswith(REGIME_PREFIX), (
            f"T-0174 A4 (start): the regime line follows the reask line; next {after[:1]}"
        )


# =====================================================================================================
# A3: config
# =====================================================================================================


def test_config_constants() -> None:
    """A3, re-scoped by D1/D4: the versions tuple, and the default (``"v1"`` in group A, ``"v2"`` since D1).
    Re-scoped by T-0191 A3: the tuple is ``("v1", "v2", "v3")``; and by T-0191 C3: the default is exactly ``"v3"``
    (T-0191 C1, James's ruling 2026-10-28). Re-scoped by T-0211 B3: the tuple is ``("v1", "v2", "v3", "v4")``
    (the default stays ``"v3"``)."""
    versions = getattr(config, "LIVE_GUIDANCE_VERSIONS", None)
    default = getattr(config, "DEFAULT_LIVE_GUIDANCE", None)
    assert versions == ("v1", "v2", "v3", "v4") and isinstance(versions, tuple), (
        f"T-0211 B3: config.LIVE_GUIDANCE_VERSIONS = ('v1', 'v2', 'v3', 'v4') does not exist yet; {versions!r}"
    )
    assert default == "v3", f"T-0191 C1: config.DEFAULT_LIVE_GUIDANCE = 'v3'; {default!r}"


def test_profile_field() -> None:
    """A3: ``Profile.live_guidance: str | None = None``, ``None`` in every built-in profile."""
    require_field()
    f = Profile.model_fields[FIELD]
    assert f.default is None and not f.is_required() and f.annotation == (str | None), f
    for name, profile in load(None).profiles.items():
        assert getattr(profile, FIELD) is None, name


def test_profile_load_refusals(tmp_path: Path) -> None:
    """A3: an unknown value is refused naming v1 and v2; a known value with ``tools`` not live is refused; with
    ``tools = "live"`` both versions load and read back. Re-scoped by T-0191 A3 (written ahead as pending for
    T-0191): the junk list's ``"v3"`` (a version now) becomes ``"v4"``, and the refusal names v1, v2 and v3.
    Re-scoped by T-0211 B3: ``"v4"`` is a version now, so the junk becomes ``"v5"``, and the refusal names all
    four."""
    require_field()
    for n, junk in enumerate(('"v5"', '"V1"', '"v2 "', '""', '"1"', "1", "true")):
        msg = _load_error(tmp_path / f"j{n}", f"{LIVE_PROFILE}{FIELD} = {junk}\n")
        assert msg is not None and all(v in msg for v in ("v1", "v2", "v3", "v4")), (
            f"T-0211 B3: {FIELD} = {junk} is refused naming v1, v2, v3 and v4 (v4 does not exist yet); got {msg!r}"
        )
    for tools in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        for value in ("v1", "v2"):
            msg = _load_error(tmp_path / f"o{len(tools)}{value}", f'{tools}{FIELD} = "{value}"\n')
            assert msg is not None, (
                f"T-0174 A3: {FIELD} = {value!r} with {tools.strip() or 'no tools key'} loaded"
            )
    for value in ("v1", "v2"):
        d = tmp_path / f"ok-{value}"
        d.mkdir()
        got = getattr(load(_toml(d, f'{LIVE_PROFILE}{FIELD} = "{value}"\n')).active(), FIELD)
        assert got == value, (value, got)


# =====================================================================================================
# A4: the resolver
# =====================================================================================================

#: (profile keywords, namespace, mode) -> (value, source)
TABLE: dict[str, tuple[dict[str, Any], dict[str, Any], str, tuple[Any, Any]]] = {
    # T-0191 C3: the default rows expect v3 (v2 since T-0174 D4, v1 in group A); the resolver names the version, the
    # web-dependent text choice is cli's (T-0191 A4)
    "live-default": ({"tools": "live"}, {}, "live", ("v3", "default")),
    "live-default-dest-none": ({"tools": "live"}, {FIELD: None}, "live", ("v3", "default")),
    "live-flag-v1": ({"tools": "live"}, {FIELD: "v1"}, "live", ("v1", "flag")),
    "live-flag-v2": ({"tools": "live"}, {FIELD: "v2"}, "live", ("v2", "flag")),
    "live-profile-v2": ({"tools": "live", FIELD: "v2"}, {}, "live", ("v2", "profile")),
    "live-profile-v1": ({"tools": "live", FIELD: "v1"}, {}, "live", ("v1", "profile")),
    "flag-v1-over-profile-v2": (
        {"tools": "live", FIELD: "v2"},
        {FIELD: "v1"},
        "live",
        ("v1", "flag"),
    ),
    "flag-v2-over-profile-v1": (
        {"tools": "live", FIELD: "v1"},
        {FIELD: "v2"},
        "live",
        ("v2", "flag"),
    ),
    "off": ({}, {}, "off", (None, "unset")),
    "dry_run": ({"tools": "dry_run"}, {}, "dry_run", (None, "unset")),
    "live-profile-resolved-off": ({"tools": "live", FIELD: "v2"}, {}, "off", (None, "unset")),
    "live-profile-resolved-dry": (
        {"tools": "live", FIELD: "v2"},
        {FIELD: "v2"},
        "dry_run",
        (None, "unset"),
    ),
}


#: T-0191 C3: the rows the default flip moves (pending until T-0191 C's implementation)
C_MOVES = ("live-default", "live-default-dest-none", "default")


def _c_params(cases: Any) -> list[Any]:
    return [c for c in cases]


@pytest.mark.parametrize("case", _c_params(TABLE))
def test_resolver_table(case: str) -> None:
    """A4: flag, else profile, else the default; ``(None, "unset")`` in every mode but live, whatever is set."""
    fn = require_resolver()
    require_field()
    pk, ns, mode, want = TABLE[case]
    profile = Profile(name="lean", **pk)
    assert pair(fn(profile, argparse.Namespace(**ns), mode)) == want, case


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
        raise AssertionError("live_guidance_of opened a socket")
socket.socket = NoSocket
builtins.open = spy
try:
    out = [[r.value, r.source] for r in (
        cli.live_guidance_of(Profile(name="lean", tools="live"), argparse.Namespace(), "live"),
        cli.live_guidance_of(Profile(name="lean", tools="live"), argparse.Namespace(live_guidance="v2"), "live"),
        cli.live_guidance_of(Profile(name="lean", tools="live", live_guidance="v2"), argparse.Namespace(), "live"),
        cli.live_guidance_of(Profile(name="lean"), argparse.Namespace(), "off"),
    )]
finally:
    builtins.open = real_open
print(json.dumps({"opened": opened, "values": out,
                  "computer": sorted(m for m in sys.modules if m.startswith("local_astra.computer"))}))
"""


def test_resolver_pure() -> None:
    """A4: ``live_guidance_of`` opens no socket and no file and imports no ``local_astra.computer`` module (a fresh
    interpreter). The default's value is ``v3`` since T-0191 C1 (C3 re-scope; ``v2`` since D1)."""
    require_resolver()
    proc = subprocess.run(
        [sys.executable, "-c", _PURE], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["opened"] == [] and got["computer"] == [], got
    assert got["values"] == [
        ["v3", "default"],
        ["v2", "flag"],
        ["v2", "profile"],
        [None, "unset"],
    ], got["values"]


# =====================================================================================================
# A4: the session gets LIVE_GUIDANCE[value]
# =====================================================================================================

#: case -> (flags, profile body after LIVE_PROFILE, version)
SOURCES: dict[str, tuple[list[str], str, str]] = {
    "default": ([], "", "v3"),  # T-0191 C3: web mode (v2 since T-0174 D1, v1 in group A)
    "flag-v1": ([FLAG, "v1"], "", "v1"),
    "flag-v2": ([FLAG, "v2"], "", "v2"),
    "profile-v1": ([], f'{FIELD} = "v1"\n', "v1"),
    "profile-v2": ([], f'{FIELD} = "v2"\n', "v2"),
    "flag-v1-over-profile-v2": ([FLAG, "v1"], f'{FIELD} = "v2"\n', "v1"),
    "flag-v2-over-profile-v1": ([FLAG, "v2"], f'{FIELD} = "v1"\n', "v2"),
}


@pytest.mark.parametrize("case", _c_params(SOURCES))
def test_session_gets_guidance(tmp_path, capsys, monkeypatch, case):
    """A4: ``cli`` builds the live session with ``tool_guidance`` the mapping's object for the resolved version, and
    its system message ends with that text once. ``default``, re-scoped by D4 (v2) and T-0191 C3: v3 (web mode)."""
    flags, body, version = SOURCES[case]
    want = guidance_map()[version]
    # T-0202 D3 re-pin: the wait tool pinned off, so tool_guidance is the mapping's object itself (``is``)
    calls = build_mode(
        tmp_path, capsys, monkeypatch, "off", *t0202.pin_off(flags, body), body=LIVE_PROFILE + body
    )
    kw = calls.session_kwargs[-1]
    assert kw.get("tool_guidance") is want, (
        f"T-0174 A4 ({case}): Session(tool_guidance=LIVE_GUIDANCE[{version!r}]); got {kw.get('tool_guidance')!r}"
    )
    text = calls.graphs[-1].session.prompts.system_message().text()
    assert text.endswith("\n" + want) and text.count(want) == 1, (
        f"T-0174 A4 ({case}): the system message ends with the {version} guidance"
    )
    if version == "v1":
        assert text.endswith(pb.LIVE_TOOL_GUIDANCE)


# =====================================================================================================
# A4: the startup line
# =====================================================================================================


def test_default_prints_no_line(tmp_path, capsys, monkeypatch):
    """A4 (active): with the default source neither stream carries a ``live guidance`` line, under ``--dry-run`` and
    at a real start; the reask line is printed once (premise)."""
    out, err = dry_out_err(tmp_path / "dry", capsys, monkeypatch)
    assert "live guidance" not in out + err, out + err
    assert out.splitlines().count(REASK_LINE) == 1, out
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *MODE_FLAGS["live"])
    assert code == 0 and "live guidance" not in started, started


# =====================================================================================================
# D1: the default is v2 (James's ruling, 2026-10-27)
# =====================================================================================================

_ABSENT = "<absent>"


def test_default_is_v3(tmp_path, capsys, monkeypatch):
    """T-0191 C3 re-scope of D1's ``test_default_is_v2``: ``config.DEFAULT_LIVE_GUIDANCE == "v3"``; ``cli`` with no
    flag and no key (web mode) builds the live session with ``tool_guidance`` the mapping's v3 object, its system
    message ends with v3 once and holds no v1 text; the default prints no ``live guidance`` line (dry-run and a real
    start, both streams).

    Equivalence with the explicit source: every ``Session`` keyword name of the default build is the
    ``--live-guidance v3`` build's, and ``page_fact`` (``True`` iff the resolved guidance is v2 or v3) is the same in
    both. The whole-byte equivalence (bodies, logs, turn lines) is
    ``test_t0174_v1_identity.test_default_is_explicit_v3``.
    """
    default = getattr(config, "DEFAULT_LIVE_GUIDANCE", None)
    assert default == "v3", f"T-0191 C1: config.DEFAULT_LIVE_GUIDANCE is {default!r}, not 'v3'"
    v1, v2, v3 = pb.LIVE_TOOL_GUIDANCE, guidance_map()["v2"], guidance_map()["v3"]
    assert v1 != v3 and v1 not in v3 and v3 != v2, "premise: v1, v2 and v3 are distinct texts"

    # T-0202 D3 re-pin: the guidance default with the wait tool pinned off (the wait sentence is T-0202 D's row)
    calls = build_mode(
        tmp_path / "default", capsys, monkeypatch, "off", *t0202.pin_off([]), body=LIVE_PROFILE
    )
    kw = calls.session_kwargs[-1]
    assert kw.get("tool_guidance") is v3, (
        f"T-0191 C1: with no flag the session gets LIVE_GUIDANCE['v3']; got {str(kw.get('tool_guidance'))[-80:]!r}"
    )
    text = calls.graphs[-1].session.prompts.system_message().text()
    assert text.endswith("\n" + v3) and text.count(v3) == 1 and v1 not in text, (
        f"T-0191 C1: the default system message ends with v3 and holds no v1; it ends {text[-200:]!r}"
    )
    monkeypatch.undo()
    explicit = build_mode(
        tmp_path / "flag-v3",
        capsys,
        monkeypatch,
        "off",
        *t0202.pin_off([FLAG, "v3"]),  # T-0202 D3 re-pin: as the default build above
        body=LIVE_PROFILE,
    )
    kw2 = explicit.session_kwargs[-1]
    assert kw2.get("tool_guidance") is v3, "premise: --live-guidance v3 gives v3 (T-0191 A)"
    assert sorted(kw) == sorted(kw2), (
        f"T-0191 C1: the default and --live-guidance v3 pass different Session keywords: "
        f"{sorted(set(kw) ^ set(kw2))}"
    )
    assert kw.get("page_fact", _ABSENT) is True and kw2.get("page_fact", _ABSENT) is True, (
        f"T-0191 C1 / A4: page_fact with v3, default ({kw.get('page_fact', _ABSENT)!r}) and "
        f"--live-guidance v3 ({kw2.get('page_fact', _ABSENT)!r})"
    )
    monkeypatch.undo()

    out, err = dry_out_err(tmp_path / "dry", capsys, monkeypatch)
    assert "live guidance" not in out + err, f"T-0174 D1: the default stays silent\n{out}{err}"
    assert out.splitlines().count(REASK_LINE) == 1, out
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *MODE_FLAGS["live"])
    assert code == 0 and "live guidance" not in started, started


@pytest.mark.parametrize("case", ["flag-v1", "profile-v1"])
def test_v1_source_prints_line_and_gets_v1(tmp_path, capsys, monkeypatch, case):
    """D1: after the flip ``--live-guidance v1`` or ``live_guidance = "v1"`` prints exactly the A4 line
    ``astra session: live guidance: v1 (flag|profile)`` before the reask line (dry-run), gets
    ``LIVE_TOOL_GUIDANCE`` (v1), and differs from the default's session (the default is v2): the second half is
    what moves at D. The BASE byte-identity is ``test_t0174_v1_identity.test_flag_v1_is_base``."""
    flags, body, version = SOURCES[case]
    assert version == "v1"
    out, _ = dry_out_err(tmp_path / "dry", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert_line_before_reask(out, f"{PREFIX}v1 ({'flag' if flags else 'profile'})", "dry")
    monkeypatch.undo()
    # T-0202 D3 re-pin: the wait tool pinned off, so tool_guidance is LIVE_TOOL_GUIDANCE itself (``is``)
    got = build_mode(
        tmp_path / "v1",
        capsys,
        monkeypatch,
        "off",
        *t0202.pin_off(flags, body),
        body=LIVE_PROFILE + body,
    )
    assert got.session_kwargs[-1].get("tool_guidance") is pb.LIVE_TOOL_GUIDANCE
    assert got.session_kwargs[-1].get("page_fact", False) is False, "B2-1: no page fact with v1"
    monkeypatch.undo()
    default = build_mode(tmp_path / "default", capsys, monkeypatch, "off", body=LIVE_PROFILE)
    assert default.session_kwargs[-1].get("tool_guidance") is not pb.LIVE_TOOL_GUIDANCE, (
        f"T-0174 D1 ({case}): the default still builds v1, so the v1 source is not distinguishable from it"
    )


NON_DEFAULT = [c for c in SOURCES if c != "default"]


@pytest.mark.parametrize("case", NON_DEFAULT)
def test_line_flag_or_profile(tmp_path, capsys, monkeypatch, case):
    """A4: a flag or profile source prints exactly ``astra session: live guidance: <v> (<source>)`` once, right
    before the reask line, which stays last; under ``--dry-run`` (stdout; stderr carries no such line) and at a
    real start."""
    require_flag()
    flags, body, version = SOURCES[case]
    source = "flag" if flags else "profile"
    want = f"{PREFIX}{version} ({source})"
    out, err = dry_out_err(tmp_path / "dry", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert guidance_lines(err) == [], err
    assert_line_before_reask(out, want, "dry")
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
    assert_line_before_reask(started, want, "start")


@pytest.mark.parametrize("case", NON_DEFAULT)
def test_line_changes_nothing_else(tmp_path, capsys, monkeypatch, case):
    """A4: with the one line set aside, a flag or profile ``--dry-run`` prints the default run's stdout and stderr
    (for v2 too: the version changes the session, not the other lines)."""
    require_flag()
    flags, body, _ = SOURCES[case]
    base_out, base_err = dry_out_err(tmp_path / "base", capsys, monkeypatch, body=LIVE_PROFILE)
    monkeypatch.undo()
    out, err = dry_out_err(tmp_path / "case", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)

    def norm(text: str, where: Path) -> list[str]:
        return [
            x.replace(str(where), "<tmp>") for x in text.splitlines() if not x.startswith(PREFIX)
        ]

    assert len(guidance_lines(out)) == 1, out
    assert norm(out, tmp_path / "case") == norm(base_out, tmp_path / "base"), out
    assert norm(err, tmp_path / "case") == norm(base_err, tmp_path / "base"), err


# =====================================================================================================
# A4: refusals outside live
# =====================================================================================================


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
        made = fake_live(monkeypatch, events)
        code = run_cli([*argv(where, *flags, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2, f"T-0174 A4 {flags} {body!r} {extra}: exit 2; got {code}\n{out}{err}"
        assert "unrecognized arguments" not in err and "invalid choice" not in err, (
            f"T-0174 A4: cli's own refusal, not argparse's usage error: {err}"
        )
        assert re.search(r"live[-_]guidance", out + err), (
            f"T-0174 A4: the refusal names the flag or key: {out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
        assert not [e for e in events if e[0].startswith(("pages.", "browser."))], list(events)
        assert made == [], "a refused run made a browser executor"
        monkeypatch.undo()


@pytest.mark.parametrize("value", ["v1", "v2"])
@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, value):
    """A4: ``--live-guidance v1|v2`` with ``--tools off`` or ``--tools dry-run`` exits 2 before any graph, page
    server or browser (plain, and over a live profile), under ``--dry-run`` and at a real start."""
    require_flag()
    _refused(tmp_path / "plain", capsys, monkeypatch, [*MODE_FLAGS[mode], FLAG, value], "")
    spelled = {"off": "off", "dry_run": "dry-run"}[mode]
    _refused(
        tmp_path / "over", capsys, monkeypatch, ["--tools", spelled, FLAG, value], LIVE_PROFILE
    )


@pytest.mark.parametrize("value", ["v1", "v2"])
@pytest.mark.parametrize("override", ["off", "dry-run"])
def test_profile_key_refused_under_override(tmp_path, capsys, monkeypatch, override, value):
    """A3/A4: a live profile with the key loads, but under ``--tools off|dry-run`` it would do nothing: exit 2, no
    graph. Too-wide companion: the same profile with no override builds a live session."""
    require_field()
    body = f'{LIVE_PROFILE}{FIELD} = "{value}"\n'
    _refused(tmp_path / "over", capsys, monkeypatch, ["--tools", override], body)
    # T-0202 D3 re-pin: the wait tool pinned off, so tool_guidance is the mapping's object itself (``is``)
    calls = build_mode(
        tmp_path / "ok", capsys, monkeypatch, "off", *t0202.pin_off([], body), body=body
    )
    assert calls.session_kwargs[-1].get("tool_guidance") is guidance_map()[value]


@pytest.mark.parametrize("junk", ["v5", "V1", "default", "2"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """A4: only the listed versions parse (argparse ``choices``). Re-scoped by T-0191 A3 (approved 2026-10-28): the
    junk value ``"v3"`` became ``"v4"`` because v3 is a version now; green before and after T-0191 A. Re-scoped by
    T-0211 B3: ``"v4"`` becomes ``"v5"``; green before and after T-0211 B."""
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *MODE_FLAGS["live"], FLAG, junk), "--dry-run"])
    _, err = _out(capsys)
    assert code == 2 and "invalid choice" in err, f"{junk}: {code}\n{err}"


def test_help_states_choices_and_default():
    """A4: the flag's help names both versions and the default."""
    require_flag()
    h = t0133._help_of(FLAG)
    assert re.search(r"\bv1\b", h) and re.search(r"\bv2\b", h) and "default" in h.lower(), h


def test_help_says_default_v3():
    """T-0191 C1/C3 re-scope of D4b's ``test_help_says_default_v2`` (C1 allows the help text to change at the flip):
    the flag's help states the fallback is v3 (``else v3``), and no longer ``else v2`` or ``else v1``."""
    require_flag()
    h = " ".join(t0133._help_of(FLAG).split())
    assert re.search(r"\belse v3\b", h) and not re.search(r"\belse v[12]\b", h), (
        f"T-0191 C1: --live-guidance help names v3 as the default; {h!r}"
    )
