"""T-0217: the settings panel's relaunch hint never suggests a flag already in force (docs/tasks/T-0217.md).

Every expected text below is hand-typed from the task's criterion 1 table and R-27, checked against ``cli``'s
argparse at 423362b7 (``--search {off,ddgs}``, ``--browser {headed,headless}``, ``--wait-tool {on,off}``,
``--page-overview/--no-page-overview`` and ``--filler-clips/--no-filler-clips`` are ``BooleanOptionalAction``,
``--action-echo {pixels,model}``; ``--no-web`` is ``store_true`` with no ``--web``, and the ``tools`` row's other value
in live has no single flag the ruling accepts, so both are R-27 rows). Today's R-12 texts (criterion 3) are the ones
``test_t0213_settings_wire`` already pins. No expected value is computed from the catalog by a copy of the rule.

Criterion 4's oracle is ``cli`` itself: the hint is appended to the session's own flags (argparse: the last one
wins) and the session is built again; the row must then report the other value. A hint that names the value already
in force leaves the row where it was.

Regime: CPU over fakes (``_cli_graph``: a real ``cli`` start, nothing served, no socket), profile ``lean``; n = 1,
deterministic per row. No measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import protocol as P
from tests.unit.test_t0213_settings_wire import (
    BARE_TOML,
    LIVE_ARGS,
    LIVE_TOML,
    _cli_graph,
    _cli_items,
    _items_of,
    _pair,
)

LIVE = ("--tools", "live")

#: criterion 1: key -> (flags that put the value in force, the value in force, the inverse hint, the other value)
IN_FORCE: dict[str, tuple[tuple[str, ...], Any, str, Any]] = {
    "search": ((*LIVE, "--search", "off"), "off", "--search ddgs", "ddgs"),
    "browser": (LIVE_ARGS, "headless", "--browser headed", "headed"),
    "wait_tool": ((*LIVE, "--wait-tool", "off"), False, "--wait-tool on", True),
    "page_overview": ((*LIVE, "--page-overview"), True, "--no-page-overview", False),
    "action_echo": ((*LIVE, "--action-echo", "pixels"), "pixels", "--action-echo model", "model"),
    "filler_clips": ((*LIVE, "--filler-clips"), True, "--no-filler-clips", False),
}

#: every criterion-1 value in force at once, by flag
ALL_IN_FORCE_FLAGS = (
    *LIVE_ARGS,
    "--search",
    "off",
    "--wait-tool",
    "off",
    "--page-overview",
    "--action-echo",
    "pixels",
    "--filler-clips",
)

#: the same values from astra.toml, not from a flag (``wait_tool`` has no profile key in the catalog: not here)
PROFILE_IN_FORCE_TOML = (
    LIVE_TOML  # tools = "live", browser = "headless"
    + 'search = "off"\npage_overview = true\naction_echo = "pixels"\nfiller_clips = true\nbrowser_web = false\n'
)

#: criterion 3: today's R-12 texts, when the value in force is the other one (hand-typed, as test_t0213 pins them)
TODAY = {
    "search": "--search off",
    "browser": "--browser headless",
    "wait_tool": "--wait-tool off",
    "page_overview": "--page-overview",
    "action_echo": "--action-echo pixels",
    "filler_clips": "--filler-clips",
    "browser_web": "--no-web",
}

#: the opposite of every criterion-1 value, each set by an explicit flag (source ``flag``, value not the effect)
EXPLICIT_OPPOSITE_FLAGS = (
    *LIVE,
    "--browser",
    "headed",
    "--search",
    "ddgs",
    "--wait-tool",
    "on",
    "--no-page-overview",
    "--action-echo",
    "model",
    "--no-filler-clips",
)

#: the other value of each row, by the value in force (hand-typed): criterion 4's expectation after the hint
OTHER: dict[str, dict[Any, Any]] = {
    "search": {"off": "ddgs", "ddgs": "off"},
    "browser": {"headless": "headed", "headed": "headless"},
    "wait_tool": {False: True, True: False},
    "page_overview": {True: False, False: True},
    "action_echo": {"pixels": "model", "model": "pixels"},
    "filler_clips": {True: False, False: True},
    "browser_web": {True: False},
}


def _sub(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


# ------------------------------------------------------------------------------------------------- criterion 1


@pytest.mark.parametrize("key", list(IN_FORCE))
def test_inverse_hint_when_in_force(tmp_path, capsys, monkeypatch, key):
    """Criterion 1: in live, a one-way row whose value in force (set here by its own flag) equals its command's
    effect carries the flag that sets the other value; the row stays read-only with kind ``relaunch``. Kills the
    R-12 text sent whatever the value (today: ``--search off`` in a session started with ``--search off``)."""
    flags, value, hint, _other = IN_FORCE[key]
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    assert _pair(items, key) == (value, "flag"), f"premise: {key} is {value!r} by flag"
    item = items[key]
    assert item.relaunch == hint, (
        f"T-0217 criterion 1: {key} is already {value!r}; the hint must be {hint!r}, got {item.relaunch!r}"
    )
    assert item.kind == "relaunch" and item.changeable is False, (key, item.kind, item.changeable)


def test_inverse_hints_all_in_force_at_once(tmp_path, capsys, monkeypatch):
    """Criterion 1 with every criterion-1 value in force in one session: each row gets its own inverse, so a fix
    that inverts only the first in-force row it meets (or one per session) is caught."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *ALL_IN_FORCE_FLAGS)
    got = {key: items[key].relaunch for key in IN_FORCE}
    assert got == {key: row[2] for key, row in IN_FORCE.items()}, got


# ------------------------------------------------------------------------------------------------- criterion 2 (R-27)


@pytest.mark.parametrize(
    ("key", "flags", "value"),
    [("browser_web", (*LIVE, "--no-web"), False), ("tools", LIVE, "live")],
    ids=["browser_web", "tools"],
)
def test_no_hint_when_no_single_flag(tmp_path, capsys, monkeypatch, key, flags, value):
    """Criterion 2 (R-27, amends R-10): web off has no ``--web`` flag, and tools ``live`` in live has no single
    flag the ruling accepts for the other value, so the row carries no relaunch text; its kind stays ``relaunch``
    and it stays read-only. Kills ``--no-web`` sent in a ``--no-web`` session and ``--tools live`` in live."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    assert _pair(items, key) == (value, "flag"), f"premise: {key} is {value!r} by flag"
    item = items[key]
    assert item.relaunch is None, (
        f"T-0217 R-27: {key} = {value!r} has no hint; got {item.relaunch!r}"
    )
    assert item.kind == "relaunch" and item.changeable is False, (
        f"T-0217 R-27: the kind stays relaunch; got {item.kind!r}, changeable {item.changeable}"
    )


# ------------------------------------------------------------------------------------------------- criterion 3


@pytest.mark.parametrize(
    "flags", [LIVE, EXPLICIT_OPPOSITE_FLAGS], ids=["defaults", "explicit-opposites"]
)
def test_hint_unchanged_when_not_in_force(tmp_path, capsys, monkeypatch, flags):
    """Criterion 3 (negative): when the value in force differs from the command's effect, the text is exactly
    today's R-12 text: in a default live session (headed, search on, wait on, no overview, model echo, no fillers,
    web on) and in one where each of those opposite values is set by an explicit flag. Kills a fix that always
    sends the inverse, and one that keys on the source (``flag``) instead of the value."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    premise = {key: items[key].value for key in TODAY}
    assert premise == {
        "search": "ddgs",
        "browser": "headed",
        "wait_tool": True,
        "page_overview": False,
        "action_echo": "model",
        "filler_clips": False,
        "browser_web": True,
    }, f"premise: the other value in force; got {premise}"
    got = {key: items[key].relaunch for key in TODAY}
    assert got == TODAY, got


# ------------------------------------------------------------------------------------------------- criterion 5


#: criterion 5: the rows T-0217 must not touch, in live (hand-typed from test_t0213_settings_wire's R-12 pins)
UNTOUCHED_LIVE = {
    "max_tool_steps": "--max-tool-steps N",
    "max_tool_steps_web": "--max-tool-steps N",
    "engine": "--profile lean",
    "browser_viewport": "set browser_viewport in astra.toml, then relaunch diffusion_gearvis session",
}

#: R-10a: every live-only row outside live reads not_here (hand list, the same as test_t0213's)
R10A_NOT_HERE = (
    "page_overview",
    "wait_tool",
    "browser",
    "browser_viewport",
    "browser_web",
    "max_tool_steps",
    "max_tool_steps_web",
    "action_echo",
)


@pytest.mark.parametrize(
    ("config", "flags"),
    [
        (BARE_TOML, LIVE),
        (BARE_TOML, ALL_IN_FORCE_FLAGS),
        (BARE_TOML, (*ALL_IN_FORCE_FLAGS, "--no-web", "--max-tool-steps", "9")),
        (PROFILE_IN_FORCE_TOML, ()),
    ],
    ids=["defaults", "all-in-force", "all-in-force-no-web-steps-9", "profile-in-force"],
)
def test_parametric_and_engine_rows_unchanged(tmp_path, capsys, monkeypatch, config, flags):
    """Criterion 5 (negative): the parametric step-budget rows keep ``--max-tool-steps N``, the engine keeps
    ``--profile <the profile in force>`` and browser_viewport its config sentence, whatever else is in force
    (including a budget set by flag: 9 is not "N", and no inverse exists). Kills an in-force rule that reaches
    these rows."""
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    got = {key: items[key].relaunch for key in UNTOUCHED_LIVE}
    assert got == UNTOUCHED_LIVE, got
    kinds = {key: items[key].kind for key in UNTOUCHED_LIVE}
    assert kinds == dict.fromkeys(UNTOUCHED_LIVE, "relaunch"), kinds


@pytest.mark.parametrize("flags", [(), ("--tools", "dry-run")], ids=["off", "dry_run"])
def test_outside_live_r10a_unchanged(tmp_path, capsys, monkeypatch, flags):
    """Criterion 5 (negative): outside live R-10a holds as before: every live-only row is ``not_here`` and the
    ``tools`` row carries ``--tools live`` (there the value in force is not ``live``, so it is a real hint)."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    kinds = {key: items[key].kind for key in R10A_NOT_HERE}
    assert kinds == dict.fromkeys(R10A_NOT_HERE, "not_here"), kinds
    assert items["tools"].relaunch == "--tools live", items["tools"].relaunch
    assert items["tools"].kind == "relaunch", items["tools"].kind


@pytest.mark.parametrize("flags", [(), ("--filler-clips",)], ids=["off", "off-filler-clips"])
def test_r29_outside_live_text_stays(tmp_path, capsys, monkeypatch, flags):
    """R-29: the inverse rule applies in live only; outside live today's text stays, even where the value in force
    is the command's effect (search ``off`` with tools off; filler clips on by flag). Kills an in-force rule applied
    in every mode."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    filler = bool(flags)
    assert _pair(items, "search")[0] == "off", "premise: no search with tools off"
    assert _pair(items, "filler_clips")[0] is filler, "premise: filler clips"
    got = {key: items[key].relaunch for key in ("search", "filler_clips")}
    assert got == {"search": "--search off", "filler_clips": "--filler-clips"}, got


# ------------------------------------------------------------------------------------------------- criterion 4


#: sessions whose hints criterion 4 parses: (config, flags)
PARSE_SESSIONS = {
    "defaults": (BARE_TOML, LIVE),
    "explicit-opposites": (BARE_TOML, EXPLICIT_OPPOSITE_FLAGS),
    "all-in-force": (BARE_TOML, ALL_IN_FORCE_FLAGS),
    "profile-in-force": (PROFILE_IN_FORCE_TOML, ()),
}


@pytest.mark.parametrize("case", list(PARSE_SESSIONS))
def test_every_hint_parses(tmp_path, capsys, monkeypatch, case):
    """Criterion 4: every switch-row hint the session sends, appended to the session's own flags, parses with
    ``cli`` (the harness asserts exit 0, so an argparse error fails the row) and gives the other value. The oracle
    is ``cli``, not a copy of the table. In the in-force sessions today's texts name the value already in force, so
    the row stays where it was. R-28: the parametric rows (``--max-tool-steps N``), the engine and browser_viewport
    are excluded (criterion 5 leaves them unchanged)."""
    config, flags = PARSE_SESSIONS[case]
    items = _cli_items(_sub(tmp_path, "first"), capsys, monkeypatch, config, *flags)
    problems: list[str] = []
    checked = 0
    for n, (key, others) in enumerate(OTHER.items()):
        value = items[key].value
        hint = items[key].relaunch
        if hint is None:
            assert key == "browser_web" and value is False, (
                f"T-0217: only web off may have no hint here; {key} = {value!r} has none"
            )
            continue
        if key == "browser_web" and value is False:
            problems.append(f"browser_web: web off sends {hint!r}; R-27 says none")
            continue
        assert value in others, f"premise: {key} = {value!r} has a hand-typed other value"
        again = _cli_items(
            _sub(tmp_path, f"hint{n}"), capsys, monkeypatch, config, *flags, *hint.split(" ")
        )
        checked += 1
        if again[key].value != others[value]:
            problems.append(
                f"{key}: {value!r} + {hint!r} gives {again[key].value!r}, not {others[value]!r}"
            )
    assert checked >= 6, f"premise: every switch row was parsed; {checked}"
    assert problems == [], "T-0217 criterion 4:\n  " + "\n  ".join(problems)


# ------------------------------------------------------------------------------------------------- adversarial


def test_profile_source_in_force(tmp_path, capsys, monkeypatch):
    """Adversarial: the values set in astra.toml, not by flag, still give the inverse hints, and R-27's rows (web
    off, tools live, both from the profile) give none. Kills a fix that reads the session's ``args`` (no flag was
    given here) instead of the value in force. ``wait_tool`` has no profile key, so it is not in this row."""
    items = _cli_items(tmp_path, capsys, monkeypatch, PROFILE_IN_FORCE_TOML)
    want = {key: IN_FORCE[key] for key in IN_FORCE if key != "wait_tool"}
    for key, (_flags, value, _hint, _other) in want.items():
        assert _pair(items, key) == (value, "profile"), (
            f"premise: {key} = {value!r} from astra.toml"
        )
    assert _pair(items, "browser_web") == (False, "profile") and _pair(items, "tools") == (
        "live",
        "profile",
    ), "premise: web off and tools live from astra.toml"
    got = {key: items[key].relaunch for key in (*want, "browser_web", "tools")}
    expect = {key: row[2] for key, row in want.items()} | {"browser_web": None, "tools": None}
    assert got == expect, got


def test_hint_after_settings_sets(tmp_path, capsys, monkeypatch):
    """Adversarial: the in-force hint survives a settings set: a no-op set (``action_outcome`` to the value it has)
    and then a real change applied at IDLE (``action_outcome`` on, rev 1) both answer with a state whose search row
    is still ``--search ddgs`` in a ``--search off`` session. Kills a fix computed once at hello (or only in the
    hello reply) and lost when the state is rebuilt for a set's echo. The set is handed to the session's own set
    handler, the one the pump calls, since this harness runs no pump (reported)."""
    graph = _cli_graph(tmp_path, capsys, monkeypatch, BARE_TOML, *IN_FORCE["search"][0])
    session = graph.session
    assert _pair(_items_of(graph), "action_outcome") == (False, "default"), "premise"
    session._on_settings_set(P.AstraSettingsSet(rev=0, changes={"action_outcome": False}))
    state = session.settings_state_message()
    assert state.rev == 0, f"premise: a no-op set moves nothing; rev {state.rev}"
    items = {i.key: i for i in state.items}
    assert items["search"].relaunch == "--search ddgs", items["search"].relaunch
    session._on_settings_set(P.AstraSettingsSet(rev=0, changes={"action_outcome": True}))
    state = session.settings_state_message()
    items = {i.key: i for i in state.items}
    assert state.rev == 1 and items["action_outcome"].value is True, (
        f"premise: the change applied at IDLE; rev {state.rev}, {items['action_outcome'].value!r}"
    )
    assert items["search"].relaunch == "--search ddgs", items["search"].relaunch
    assert _items_of(graph)["search"].relaunch == "--search ddgs", "the hello after the set too"
