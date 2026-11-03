"""T-0137 B-9 (cli, config): the opt-in, its source, the refusals, the wiring and the startup line
(docs/tasks/T-0137.md).

**The surface these rows pin** (the task's names; readings stated):

- ``config.LIVE_DEFAULT_REPEAT_GUARD = False``; ``Profile.repeat_guard: bool | None = None``, TOML booleans only;
  refused at load with ``tools != "live"`` (the message names ``live`` and ``repeat_guard``).
- ``astra session --repeat-guard`` / ``--no-repeat-guard`` (argparse dest ``repeat_guard``).
  ``cli.repeat_guard_of(profile, args, mode) -> RepeatGuard(repeat_guard, repeat_guard_from)``, read by those two field
  names; sources ``flag`` / ``profile`` / ``default``; outside live only "not on" is pinned. It reads
  ``LIVE_DEFAULT_REPEAT_GUARD`` from ``cli``'s namespace at call time and is the single resolver (``session_cmd`` and
  ``_build_graph`` both call it).
- Exit 2 before anything is built: ``--repeat-guard`` outside a resolved ``live`` mode, naming ``live``; the key of a
  live profile whose mode ``--tools`` overrode.
- ``Session`` gets ``repeat_guard=True`` only when on; off passes no keyword. The tools stay v1.
- The line, live only, ``--dry-run`` included: ``astra session: repeat guard: off (default)`` / ``on (flag)`` /
  ``on (profile)`` / ``off (flag)`` / ``off (profile)``, after the ``overview frame`` line and before the regime line.

**Readings** (flagged in the report):

- **C-R1 (placement beside T-0136).** T-0136 C-5 puts its ``action outcome`` line "right after the overview frame
  line" too. Ruled at activation (coordinator, after T-0136 merged at df658ba): the repeat-guard line goes right after
  T-0136's ``astra session: action outcome:`` line, which follows the overview frame line.
- **C-R2 (``--no-repeat-guard`` outside live).** B-9 refuses "on outside a resolved live mode". Pinned after review
  B (the T-0135b/T-0152 shape, as implemented in 486edba): either flag outside live exits 2 naming ``live``
  (``test_flag_refused_outside_live`` is parametrized over both).
- **C-R3 (combinations).** B-1 "none is refused": ``--repeat-guard`` beside ``--reground``, ``--page-overview``,
  ``--auto-overview`` and ``--overview-frame`` builds with ``repeat_guard=True``.

Regime: CPU, the cli rigs over fakes, nothing served, n = 1. Activated for ``/implement T-0137`` group B (no pending
marker): every row is red at BASE_B 4382224 on a named assertion (``require()``).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests.unit import _t0202_pins as t0202
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import _names_live
from tests.unit.test_t0140_cli import session_help
from tests.unit.test_t0150_cli import refused_run

V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
PREFIX = "astra session: repeat guard: "
FRAME_PREFIX = "astra session: overview frame: "
OUTCOME_PREFIX = "astra session: action outcome: "
REGIME = "astra session: regime:"
ON, OFF = "--repeat-guard", "--no-repeat-guard"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
LIVE = MODE_FLAGS["live"]


def require() -> Any:
    flags = set(re.findall(r"--[a-z][a-z-]+", session_help()))
    missing = [f for f in (ON, OFF) if f not in flags]
    assert not missing, f"T-0137 B-9: astra session {missing} do not exist yet"
    assert "repeat_guard" in Profile.model_fields, (
        "T-0137 B-9: Profile.repeat_guard does not exist yet"
    )
    assert hasattr(config, "LIVE_DEFAULT_REPEAT_GUARD"), (
        "T-0137 B-9: config.LIVE_DEFAULT_REPEAT_GUARD does not exist yet"
    )
    fn = getattr(cli, "repeat_guard_of", None)
    assert callable(fn), "T-0137 B-9: cli.repeat_guard_of(profile, args, mode) does not exist yet"
    return fn


def pair(v: Any) -> tuple[Any, Any]:
    return v.repeat_guard, v.repeat_guard_from


def lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def _load(tmp_path: Any, name: str, body: str) -> Any:
    d = tmp_path / name
    d.mkdir()
    return load(_toml(d, body))


def check_placement(out: str, line: str, *, regime: bool) -> None:
    """C-R1: the line once, after the overview frame line (T-0136's action outcome line may sit between), and before
    the regime line when there is one."""
    ls = out.splitlines()
    assert lines(out) == [line], out
    (f,) = [i for i, x in enumerate(ls) if x.startswith(FRAME_PREFIX)]
    at = ls.index(line)
    assert at == f + 2 and ls[f + 1].startswith(OUTCOME_PREFIX), (
        "T-0137 B-9 (C-R1, as ruled at activation): the repeat guard line goes right after T-0136's action outcome "
        f"line, which follows the overview frame line;\n{out}"
    )
    if regime:
        (r,) = [i for i, x in enumerate(ls) if x.startswith(REGIME)]
        assert at < r, out


# =====================================================================================================
# config
# =====================================================================================================


def test_constant_and_profile() -> None:
    """B-9 config: the constant is ``True`` since T-0179 A-1 (was ``False``; ``cli`` sees it); ``Profile.repeat_guard``
    is ``bool | None`` defaulting to ``None`` in every built-in profile."""
    require()
    assert config.LIVE_DEFAULT_REPEAT_GUARD is True, "T-0179 A-1: the live default is on"
    assert getattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", None) is True
    f = Profile.model_fields["repeat_guard"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.repeat_guard is None, name  # type: ignore[attr-defined]


def test_load_refusals(tmp_path: Any) -> None:
    """B-9: the key with ``tools`` off, dry_run or absent is refused at load naming ``live`` and ``repeat_guard``; a
    non-bool is refused; too-wide companions: ``true``/``false`` with live load with that value."""
    require()
    for i, tools_line in enumerate(('tools = "off"\n', 'tools = "dry_run"\n', "")):
        for v in ("true", "false"):
            with pytest.raises(ValueError) as exc:
                _load(tmp_path, f"t{i}-{v}", f"{tools_line}repeat_guard = {v}\n")
            assert _names_live(str(exc.value)) and "repeat_guard" in str(exc.value), str(exc.value)
    for j, junk in enumerate(('"on"', '"true"', "1", "0")):
        with pytest.raises(ValueError) as exc:
            _load(tmp_path, f"junk-{j}", f'tools = "live"\nrepeat_guard = {junk}\n')
        assert "repeat_guard" in str(exc.value), (junk, str(exc.value))
    for v, want in (("true", True), ("false", False)):
        got = _load(tmp_path, f"ok-{v}", f'tools = "live"\nrepeat_guard = {v}\n').active()
        assert got.repeat_guard is want  # type: ignore[attr-defined]


# =====================================================================================================
# the resolver, the wiring, the line
# =====================================================================================================

SOURCES = {
    "flag-on": ([ON], "", (True, "flag")),
    "flag-off": ([OFF], "", (False, "flag")),
    "profile-true": ([], "repeat_guard = true\n", (True, "profile")),
    "profile-false": ([], "repeat_guard = false\n", (False, "profile")),
    "flag-off-over-profile": ([OFF], "repeat_guard = true\n", (False, "flag")),
    "flag-on-over-profile": ([ON], "repeat_guard = false\n", (True, "flag")),
    "default": ([], "", (True, "default")),  # T-0179 A-2 (was off)
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources_and_wiring(tmp_path: Any, capsys: Any, monkeypatch: Any, case: str) -> None:
    """B-9 (reviewer mutation 11, off passing ``repeat_guard=False``): flag, else key, else default, each with its
    source; the dry-run line says it; the built ``Session`` gets ``repeat_guard=True`` only when on and no keyword
    otherwise; the tools stay v1."""
    fn = require()
    flags, body, (value, src) = SOURCES[case]
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    assert lines(out) == [PREFIX + ("on" if value else "off") + f" ({src})"], out
    monkeypatch.undo()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    if value:
        assert kw.get("repeat_guard") is True, (
            f"T-0137 B-9: Session(repeat_guard=True); {sorted(kw)}"
        )
    else:
        assert "repeat_guard" not in kw, f"T-0137 B-9: off passes no keyword; {sorted(kw)}"
    # T-0202 D3 re-pin: the tools stay the default live set (v1, the wait set once the wait tool is on by default)
    assert calls.kit().tools_version == t0202.default_live_sha(), "T-0137 B-9: the tools stay v1"
    assert t0202.default_on() or calls.kit().tools_version == V1_SHA
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    flag = True if ON in flags else False if OFF in flags else None
    assert pair(fn(profile, argparse.Namespace(repeat_guard=flag), "live")) == (value, src)


def test_resolver_not_on_outside_live() -> None:
    """B-9: outside a resolved live mode the resolver is never on (only "not on" is pinned), whatever the flag."""
    fn = require()
    for mode in ("off", "dry_run"):
        for flag in (None, True, False):
            value, _ = pair(fn(Profile(name="lean"), argparse.Namespace(repeat_guard=flag), mode))
            assert not value, (mode, flag)


def test_reads_default_at_call_time(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """B-9 (reviewer mutation 12, the constant captured at import): with ``cli.LIVE_DEFAULT_REPEAT_GUARD``
    monkeypatched to ``True`` the resolver gives ``(True, "default")`` and the default build passes
    ``repeat_guard=True``."""
    fn = require()
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REPEAT_GUARD", True)
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert pair(fn(profile, argparse.Namespace(), "live")) == (True, "default")
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    assert calls.session_kwargs[-1].get("repeat_guard") is True, sorted(calls.session_kwargs[-1])


def test_one_resolver(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """B-9 (adversarial; the mutant resolves in ``session_cmd`` only, so ``_build_graph`` passes nothing): a spy
    wrapping the real resolver sees at least two live calls on a default build; a spy forcing on (a real ``(True,
    "flag")`` value) with no flag in the argv makes the build pass ``repeat_guard=True`` and print ``on (flag)``."""
    real = require()
    seen: list[Any] = []

    def spy(p: Any, a: Any, m: str, **kw: Any) -> Any:
        got = real(p, a, m, **kw)
        seen.append((m, pair(got)))
        return got

    monkeypatch.setattr(cli, "repeat_guard_of", spy)
    build(tmp_path / "b", capsys, monkeypatch)
    # T-0179 A-2: the default build resolves on (was ``(False, "default")``)
    assert len(seen) >= 2 and all(x == ("live", (True, "default")) for x in seen), seen
    monkeypatch.undo()
    forced_on = real(
        Profile(name="lean", tools="live"), argparse.Namespace(repeat_guard=True), "live"
    )  # type: ignore[arg-type]
    assert pair(forced_on) == (True, "flag")
    off = real(Profile(name="lean"), argparse.Namespace(), "off")
    monkeypatch.setattr(
        cli, "repeat_guard_of", lambda p, a, m, **kw: forced_on if m == "live" else off
    )
    calls, _ = build(tmp_path / "f", capsys, monkeypatch)
    assert calls.session_kwargs[-1].get("repeat_guard") is True, (
        "T-0137 B-9: _build_graph does not read the resolver"
    )


def test_line_placement(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """B-9: exactly one line after the overview frame line under ``--dry-run`` and at a real start (there before the
    regime line, C-R1); none in ``off`` or ``dry_run``."""
    require()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", ON)
    check_placement(out, PREFIX + "on (flag)", regime=False)
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    # the real start prints the resolver lines twice (session_cmd, then the graph); the line once in each block
    blocks = started.split(FRAME_PREFIX)
    assert started.count(PREFIX) >= 1, started
    sl = started.splitlines()
    idx = [i for i, x in enumerate(sl) if x.startswith(PREFIX)]
    assert all(sl[i] == PREFIX + "on (default)" for i in idx), started  # T-0179 A-2 (was off)
    (r,) = [i for i, x in enumerate(sl) if x.startswith(REGIME)]
    assert idx and idx[-1] < r and len(blocks) - 1 == len(idx), started
    for i in idx:
        assert sl[i - 1].startswith(OUTCOME_PREFIX) and sl[i - 2].startswith(FRAME_PREFIX), started
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        assert "repeat guard:" not in dry_run(tmp_path / mode, capsys, monkeypatch, mode)


def test_other_lines_unchanged(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """B-9 "Unchanged" (in-process companion of the BASE differential): ``--repeat-guard`` changes the ``--dry-run``
    output and the real-start output only in the repeat guard line (the tools line, ``tools_line(...)`` and the regime
    string are the default's)."""
    require()
    outs = {}
    for name, flags in (("default", []), ("on", [ON])):
        monkeypatch.undo()
        d = dry_run(tmp_path / f"d-{name}", capsys, monkeypatch, "live", *flags)
        monkeypatch.undo()
        _, s, code = build_out(tmp_path / f"s-{name}", capsys, monkeypatch, *LIVE, *flags)
        assert code == 0, s
        outs[name] = (d, s)

    def strip(text: str, where: str) -> list[str]:
        return [x.replace(where, "<tmp>") for x in text.splitlines() if not x.startswith(PREFIX)]

    for k in (0, 1):
        a = strip(outs["default"][k], str(tmp_path / ("d-default" if k == 0 else "s-default")))
        b = strip(outs["on"][k], str(tmp_path / ("d-on" if k == 0 else "s-on")))
        assert a == b, f"T-0137 B-9: another line changed with --repeat-guard:\n{a}\n{b}"


@pytest.mark.parametrize(
    "other", ["--reground", "--page-overview", "--auto-overview", "--overview-frame"]
)
def test_combines(tmp_path: Any, capsys: Any, monkeypatch: Any, other: str) -> None:
    """B-1 / C-R3: ``--repeat-guard`` beside each of the four builds and passes ``repeat_guard=True``."""
    require()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, ON, other)
    assert calls.session_kwargs[-1].get("repeat_guard") is True, (
        other,
        sorted(calls.session_kwargs[-1]),
    )


# =====================================================================================================
# startup refusals
# =====================================================================================================


@pytest.mark.parametrize("flag", [ON, OFF])
@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_flag_refused_outside_live(
    tmp_path: Any, capsys: Any, monkeypatch: Any, mode: str, flag: str
) -> None:
    """B-9 / C-R2: ``--repeat-guard`` and ``--no-repeat-guard`` in a resolved mode other than live exit 2 naming
    ``live`` and build nothing (also
    through a live profile whose mode ``--tools`` overrode); checked after the flag exists, so argparse's exit 2 cannot
    satisfy it."""
    require()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        code, text, calls = refused_run(tmp_path, capsys, monkeypatch, *flags, flag, body=body)
        assert code == 2 and _names_live(text) and "unrecognized" not in text, (code, text)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
def test_key_refused_when_tools_overridden(
    tmp_path: Any, capsys: Any, monkeypatch: Any, override: list[str]
) -> None:
    """B-9: a live profile with a ``repeat_guard`` key under a non-live ``--tools`` override exits 2 naming ``live``
    and builds nothing, with ``--dry-run`` and without; too-wide companion: no override builds."""
    require()
    body = LIVE_PROFILE + "repeat_guard = true\n"
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *override, body=body, dry=dry
        )
        assert code == 2 and _names_live(text), (override, dry, code, text)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs and calls.session_kwargs[-1].get("repeat_guard") is True


def test_result_type() -> None:
    """B-9: ``repeat_guard_of`` returns a ``RepeatGuard`` with the two named fields (``repeat_guard: bool``,
    ``repeat_guard_from``)."""
    fn = require()
    got = fn(Profile(name="lean", tools="live"), argparse.Namespace(), "live")  # type: ignore[arg-type]
    assert type(got).__name__ == "RepeatGuard", type(got)
    names = (
        [f.name for f in dataclasses.fields(got)]
        if dataclasses.is_dataclass(got)
        else list(got._fields)
    )
    assert names[:2] == ["repeat_guard", "repeat_guard_from"], names
