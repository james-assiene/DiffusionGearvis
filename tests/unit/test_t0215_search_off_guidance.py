"""T-0215 (guidance side): with no search backend, no live guidance text names searching (docs/tasks/T-0215.md
criteria 5, 6, 8).

Every row builds a real live start through ``cli`` over the CLI fakes (``test_cli_search.real``), and reads two
things the session is handed: ``Session(tool_guidance=...)`` (the text fixed at construction) and the
``guidance_table`` ``cli`` passes to ``Session.init_settings`` (the per-version texts a Settings-panel switch picks
from, T-0213 A2a-4). ``init_settings`` is spied on the real ``Session`` class and delegates to it.

Expected texts are composed by hand from the ``prompt_builder`` literals the task names (``LIVE_TOOL_GUIDANCE_V3``,
``S_SHARED_GUIDANCE``, ``WAIT_TOOL_GUIDANCE``), in the order criterion 5 states: v3 + " " + S_SHARED (+ " " + the wait
sentence when the wait tool is on). "Names searching" is read as: the text contains ``google_search``, or the word
``search`` anywhere outside v3's own phrase "the current site's search box" (which criterion 5 keeps).

Rows that failed at aeaf81a5 carried ``pending(task="T-0215")`` until activation (T-0215 implement); the negative rows (criteria 6 and 8) pass there.

Regime: CPU over fakes, nothing served, no network, n = 1.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import session as session_mod
from tests.unit.test_cli_search import PROXY_ENV, real
from tests.unit.test_t0213_settings_wire import BARE_TOML, _cli_items, _pair

LIVE = ("--tools", "live", "--browser", "headless")
SITE_SEARCH_BOX = "the current site's search box"
VERSIONS = ("v1", "v2", "v3", "v4")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")


def names_search(text: str) -> bool:
    """Whether ``text`` names web searching: ``google_search``, or ``search`` outside v3's site-search-box phrase."""
    if "google_search" in text:
        return True
    return re.search(r"search", text.replace(SITE_SEARCH_BOX, ""), re.IGNORECASE) is not None


def start(
    tmp_path: Path, capsys: Any, *flags: str, body: str = ""
) -> tuple[str | None, dict[str, Any], str]:
    """A real live start; the session's ``tool_guidance``, the ``guidance_table`` ``init_settings`` got, stdout."""
    tables: list[Any] = []
    real_init = session_mod.Session.init_settings

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        tables.append(k.get("guidance_table"))
        return real_init(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session_mod.Session, "init_settings", spy)
        code, out, err, calls, _ = real(tmp_path, capsys, mp, *flags, body=body)
        assert code == 0, f"{flags}: exit {code}\n{out}{err}"
        guidance = calls.session_kwargs[-1].get("tool_guidance")
    assert len(tables) == 1 and tables[0] is not None, (
        f"premise: one init_settings call with a guidance table in live; {tables}"
    )
    return guidance, dict(tables[0]), out


def wait_flags(wait: bool) -> tuple[str, ...]:
    return ("--wait-tool", "on" if wait else "off")


def with_wait(text: str, wait: bool) -> str:
    return text + " " + pb.WAIT_TOOL_GUIDANCE if wait else text


def test_premises() -> None:
    """The literals the expected texts are composed from: v4 today is v3 + S_SHARED + S_SOURCE, S_SOURCE names
    google_search, and neither v3 nor S_SHARED nor the wait sentence names web searching (so the predicate is not
    vacuous and not over-wide)."""
    assert pb.LIVE_GUIDANCE["v4"] == (
        pb.LIVE_TOOL_GUIDANCE_V3 + " " + pb.S_SHARED_GUIDANCE + " " + pb.S_SOURCE_GUIDANCE
    )
    assert names_search(pb.S_SOURCE_GUIDANCE) and names_search(pb.LIVE_GUIDANCE["v4"])
    assert SITE_SEARCH_BOX in pb.LIVE_TOOL_GUIDANCE_V3
    for text in (
        pb.LIVE_TOOL_GUIDANCE_V3,
        pb.S_SHARED_GUIDANCE,
        pb.WAIT_TOOL_GUIDANCE,
        pb.LIVE_GUIDANCE["v1"],
        pb.LIVE_GUIDANCE["v2"],
    ):
        assert not names_search(text), text
    assert names_search("Use web search first.") and names_search("call google_search")


# =====================================================================================================
# Criterion 5: v4 with search off drops S_SOURCE
# =====================================================================================================


@pytest.mark.parametrize("wait", [True, False], ids=["wait", "no-wait"])
@pytest.mark.parametrize("how", ["flag", "profile"])
def test_v4_search_off_drops_s_source(tmp_path, capsys, wait: bool, how: str) -> None:
    """Criterion 5: ``--live-guidance v4`` with search off gives exactly v3 + " " + S_SHARED (+ " " + the wait
    sentence with the wait tool on); S_SOURCE never appears."""
    flags = (*LIVE, "--live-guidance", "v4", *wait_flags(wait))
    if how == "flag":
        guidance, _table, _out = start(tmp_path, capsys, *flags, "--search", "off")
    else:
        guidance, _table, _out = start(tmp_path, capsys, *flags, body='search = "off"\n')
    want = with_wait(pb.LIVE_TOOL_GUIDANCE_V3 + " " + pb.S_SHARED_GUIDANCE, wait)
    assert guidance == want, (
        f"T-0215 criterion 5 [{how}, wait={wait}]: tool_guidance\n  want {want!r}\n  got  {guidance!r}"
    )
    assert pb.S_SOURCE_GUIDANCE not in (guidance or "")


def test_v4_degraded_search_drops_s_source(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 5 on the degrade path: a default-on search that cannot start (``_search_backend`` raising
    ``ImportError``) is search off, so v4 drops S_SOURCE there too."""
    from local_astra import cli

    def missing(name: str) -> Any:
        raise ImportError("No module named 'ddgs' (T-0215 rig)")

    monkeypatch.setattr(cli, "_search_backend", missing)
    guidance, table, out = start(tmp_path, capsys, *LIVE, "--live-guidance", "v4")
    assert any(x.startswith("astra session: search: off (default") for x in out.splitlines()), out
    want = with_wait(pb.LIVE_TOOL_GUIDANCE_V3 + " " + pb.S_SHARED_GUIDANCE, True)
    assert guidance == want, f"T-0215 criterion 5 [degrade]: {guidance!r}"
    assert table["v4"][0] == want, f"T-0215 criterion 5 [degrade]: the table's v4 {table['v4']!r}"


@pytest.mark.parametrize("wait", [True, False], ids=["wait", "no-wait"])
@pytest.mark.parametrize("chosen", ["v3", "v4"])
def test_guidance_table_search_off_never_names_google_search(
    tmp_path, capsys, wait: bool, chosen: str
) -> None:
    """Criterion 5: with search off, every text of the guidance table (whichever version the session started on)
    names no web searching, so a Settings-panel switch to v4 cannot bring S_SOURCE back; the table's v4 is the
    session's v4 text exactly, and its page-fact flags are main's (v2, v3, v4 on)."""
    guidance, table, _out = start(
        tmp_path, capsys, *LIVE, "--search", "off", "--live-guidance", chosen, *wait_flags(wait)
    )
    assert set(table) == set(VERSIONS), f"premise: one entry per version; {sorted(table)}"
    for version, (text, _fact) in table.items():
        assert text is not None, f"premise: live has a text for {version}"
        assert not names_search(text), (
            f"T-0215 criterion 5 [wait={wait}]: the table's {version} names searching: {text!r}"
        )
    assert table["v4"][0] == with_wait(pb.LIVE_TOOL_GUIDANCE_V3 + " " + pb.S_SHARED_GUIDANCE, wait)
    assert table[chosen][0] == guidance, "the table's chosen version is the session's own text"
    assert {v: f for v, (_t, f) in table.items()} == {
        "v1": False,
        "v2": True,
        "v3": True,
        "v4": True,
    }
    assert SITE_SEARCH_BOX in table["v3"][0], "criterion 5: v3's site search box stays"


# =====================================================================================================
# Criterion 6 (negative): search on is main's, byte for byte
# =====================================================================================================


@pytest.mark.parametrize("wait", [True, False], ids=["wait", "no-wait"])
@pytest.mark.parametrize("chosen", ["default", "v3", "v4"])
def test_search_on_guidance_byte_identical(tmp_path, capsys, wait: bool, chosen: str) -> None:
    """Criterion 6: live with search on (the default) keeps main's texts: the session's guidance is the version's
    literal (+ the wait sentence), v4 still carries S_SOURCE, and the table holds every version's literal."""
    flags = (
        *LIVE,
        *(() if chosen == "default" else ("--live-guidance", chosen)),
        *wait_flags(wait),
    )
    guidance, table, out = start(tmp_path, capsys, *flags)
    assert "astra session: search: ddgs (default)" in out.splitlines(), "premise: search on"
    version = "v3" if chosen == "default" else chosen
    assert guidance == with_wait(pb.LIVE_GUIDANCE[version], wait), (
        f"T-0215 criterion 6 [{chosen}, wait={wait}]: {guidance!r}"
    )
    want_table = {
        "v1": (with_wait(pb.LIVE_TOOL_GUIDANCE, wait), False),
        "v2": (with_wait(pb.LIVE_TOOL_GUIDANCE_V2, wait), True),
        "v3": (with_wait(pb.LIVE_TOOL_GUIDANCE_V3, wait), True),
        "v4": (
            with_wait(
                pb.LIVE_TOOL_GUIDANCE_V3 + " " + pb.S_SHARED_GUIDANCE + " " + pb.S_SOURCE_GUIDANCE,
                wait,
            ),
            True,
        ),
    }
    assert table == want_table, (
        f"T-0215 criterion 6 [{chosen}, wait={wait}]: the guidance table moved"
    )


def test_search_off_no_web_is_v2(tmp_path, capsys) -> None:
    """Criterion 6/5 boundary (negative): under ``--no-web`` v4 is v2's text, with search on or off (no sentence
    names searching there at main either)."""
    for i, extra in enumerate(((), ("--search", "off"))):
        guidance, table, _out = start(
            tmp_path / f"n{i}", capsys, *LIVE, "--no-web", "--live-guidance", "v4", *extra
        )
        assert guidance == with_wait(pb.LIVE_TOOL_GUIDANCE_V2, True), (extra, guidance)
        assert table["v4"][0] == table["v3"][0] == with_wait(pb.LIVE_TOOL_GUIDANCE_V2, True)


# =====================================================================================================
# Criterion 8 (negative): the Settings panel's search row stays truthful
# =====================================================================================================


@pytest.mark.parametrize(
    "how,flags,body,source",
    [
        ("flag", ("--search", "off"), "", "flag"),
        ("profile", (), 'search = "off"\n', "profile"),
    ],
)
def test_settings_search_row_still_reports_off(
    tmp_path, capsys, monkeypatch, how: str, flags: tuple[str, ...], body: str, source: str
) -> None:
    """Criterion 8: in a search-off live session the settings state's ``search`` item still reports ``"off"`` with
    its source (the row is not edited; withdrawing the tool does not change what the panel says)."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML + body, *LIVE, *flags)
    assert _pair(items, "search") == ("off", source), f"T-0215 criterion 8 [{how}]"
