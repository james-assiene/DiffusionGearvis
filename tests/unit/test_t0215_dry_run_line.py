"""T-0215 criterion 4 (and 6's line half): the ``--dry-run`` tools line of a live session with search off shows the
sha12 of the no-search set, and a real start's regime shows the same sha (docs/tasks/T-0215.md).

Readings (reported to the caller):

- "the real start's regime line shows the same sha": at aeaf81a5 the printed ``astra session: regime:`` line of a
  real start carries no tools sha at all (profile, mic, stt, served model, vram, ``wait=``). The tools sha a real
  start records is the per-turn ``Regime.tools`` (``native:<sha12>:<mode>``, T-0092b criterion 5) in
  ``turns.jsonl``. The rows pin that one, through a real turn of a ``Session`` given the cli-built kit, and pin that
  no stdout line of the real start names the on-set's sha12.
- The search-on lines are today's, typed out from a run at aeaf81a5 (criterion 6 is "byte-identical to main").

Regime: CPU over fakes (``test_cli_search.dry``/``real``), nothing served, no network, n = 1.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import tools
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_search import PROXY_ENV, dry, real

LIVE = ("--tools", "live", "--browser", "headless")
#: the on-sets' sha12s (sha256sum at aeaf81a5)
ON_SHA12 = {
    "v1": "bbfc53ccdd80",
    "overview": "8048bdb7aabe",
    "wait": "05de281e7bd2",
    "overview+wait": "72c3d84fda01",
}
TAIL = " repair=on loop=on(20) browser=headless egress=web viewport=1280x800 visual_tokens=1120"
#: main's live ``--dry-run`` tools lines with search on (aeaf81a5), by wait tool on/off
ON_LINES = {
    True: f"tools=live native:{ON_SHA12['wait']}{TAIL} wait=on(default)",
    False: f"tools=live native:{ON_SHA12['v1']}{TAIL}",
}
LINE = re.compile(r"tools=live native:([0-9a-f]{12}) ")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")


def tools_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith("tools=")]


def wait_flags(wait: bool) -> tuple[str, ...]:
    return () if wait else ("--wait-tool", "off")


def dry_line(tmp_path: Path, capsys: Any, *flags: str) -> str:
    with pytest.MonkeyPatch.context() as mp:
        code, out, err = dry(tmp_path, capsys, mp, *flags)
    assert code == 0, f"{flags}: exit {code}\n{out}{err}"
    lines = tools_lines(out)
    assert len(lines) == 1, f"premise: one tools line; {lines}"
    return lines[0]


def real_kit(tmp_path: Path, capsys: Any, *flags: str) -> tuple[Any, str]:
    with pytest.MonkeyPatch.context() as mp:
        code, out, err, calls, _ = real(tmp_path, capsys, mp, *flags)
        assert code == 0, f"{flags}: exit {code}\n{out}{err}"
        kit = calls.kit()
    return kit, out


@pytest.mark.parametrize("wait", [True, False], ids=["wait", "no-wait"])
def test_search_on_dry_run_line_is_main_s(tmp_path, capsys, wait: bool) -> None:
    """Criterion 6: with search on the live ``--dry-run`` tools line is main's, byte for byte."""
    got = dry_line(tmp_path, capsys, *LIVE, *wait_flags(wait))
    assert got == ON_LINES[wait], f"T-0215 criterion 6: {got!r}"


@pytest.mark.parametrize("wait", [True, False], ids=["wait", "no-wait"])
@pytest.mark.parametrize("how", ["flag", "profile"])
def test_search_off_dry_run_line_shows_no_search_sha(
    tmp_path, capsys, wait: bool, how: str
) -> None:
    """Criterion 4: ``--dry-run --tools live --search off`` prints main's line with only the sha12 changed, to one
    that is no on-set's; it is the sha12 of the set a real start with the same flags declares, and that set has no
    ``google_search``."""
    extra = ("--search", "off") if how == "flag" else ()
    body_flags = (*LIVE, *wait_flags(wait), *extra)
    with pytest.MonkeyPatch.context() as mp:
        code, out, err = dry(
            tmp_path / "d",
            capsys,
            mp,
            *body_flags,
            body='search = "off"\n' if how == "profile" else "",
        )
    assert code == 0, err
    lines = tools_lines(out)
    assert len(lines) == 1, lines
    got = lines[0]
    m = LINE.match(got)
    assert m, f"premise: a live tools line; {got!r}"
    sha12 = m.group(1)
    assert sha12 not in ON_SHA12.values(), (
        f"T-0215 criterion 4 [{how}, wait={wait}]: the line shows an on-set's sha12 ({sha12})"
    )
    on_sha12 = ON_SHA12["wait" if wait else "v1"]
    assert got == ON_LINES[wait].replace(on_sha12, sha12), (
        f"T-0215 criterion 4: only the sha12 moves\n  main {ON_LINES[wait]!r}\n  got  {got!r}"
    )
    with pytest.MonkeyPatch.context() as mp:
        code, rout, rerr, calls, _ = real(
            tmp_path / "r",
            capsys,
            mp,
            *body_flags,
            body='search = "off"\n' if how == "profile" else "",
        )
        assert code == 0, rerr
        kit = calls.kit()
    assert kit.tools_version[:12] == sha12, (
        f"T-0215 criterion 4: the real start's kit is {kit.tools_version[:12]}, the dry-run line {sha12}"
    )
    assert "google_search" not in [t["function"]["name"] for t in kit.tools]
    assert all(on not in rout for on in ON_SHA12.values()), (
        "no real-start line names an on-set's sha"
    )


@pytest.fixture
def off_kit(tmp_path: Path, capsys: Any) -> tuple[Any, str]:
    """The cli-built ``--search off`` live kit and its dry-run line's sha12 (built outside the event loop)."""
    line = dry_line(tmp_path / "d", capsys, *LIVE, "--search", "off")
    m = LINE.match(line)
    assert m, line
    kit, _out = real_kit(tmp_path / "r", capsys, *LIVE, "--search", "off")
    return kit, m.group(1)


async def test_regime_tools_is_the_no_search_sha(tmp_path, off_kit: tuple[Any, str]) -> None:
    """Criterion 4: a turn of a session given the real start's kit records ``Regime.tools ==
    native:<the dry-run line's sha12>:live``, and that sha12 is not the on-set's."""
    kit, sha12 = off_kit
    assert sha12 != ON_SHA12["wait"], (
        f"T-0215 criterion 4: the search-off line still shows the on-set's sha12 ({sha12})"
    )
    regime = await T.regime_tools_of(kit, tmp_path / "s")
    assert regime == f"native:{sha12}:live", f"T-0215 criterion 4: Regime.tools {regime!r}"


def test_premise_on_sha12s() -> None:
    """The on-set sha12 literals are the shipped files' (the oracle the rows compare against)."""
    import hashlib

    root = Path(tools.__file__).resolve().parent
    files = {
        "v1": "tools_v1.json",
        "overview": "tools_v1_overview.json",
        "wait": "tools_v1_wait.json",
        "overview+wait": "tools_v1_overview_wait.json",
    }
    got = {k: hashlib.sha256((root / f).read_bytes()).hexdigest()[:12] for k, f in files.items()}
    assert got == ON_SHA12, got
