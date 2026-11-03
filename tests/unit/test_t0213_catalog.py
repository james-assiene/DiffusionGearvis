"""T-0213 A1-1, A1-2, A1-5: the settings catalog as data, its drift guard against the code, and A1's frozen paths.

The interfaces the rows pin are the task's (docs/tasks/T-0213.md A1-1/A1-2). Where the task is silent, the choice is
fixed here and listed in the test-author's report:

- ``local_astra/settings_catalog.py`` exposes the rows as ``CATALOG: tuple[SettingEntry, ...]`` in catalog order
  (ruling R-3).
- ``SettingEntry`` has the A1-1 fields plus ``default`` (R-3), ``flags`` (every spelling of ``flag``) and
  ``relaunch`` (the exact command a relaunch row shows, B-2 / A2-read ``SettingItem.relaunch``).
- ``NOT_A_SETTING`` is a tuple of ``(name, reason)`` pairs. ``name`` is either a ``diffusion_gearvis session`` option
  string (``"--data-dir"``) or a ``Profile``/``Diffusion`` field name (``"aec"``); ``reason`` is a non-empty str.
- ``flag`` is the exact CLI spelling and may name several options (``"--[no-]overview-frame"``, ``"--search
  off|ddgs"``). An option is covered when its own string is a token of some ``flag``, ``--[no-]x`` names both
  ``--x`` and ``--no-x``, and ``--no-x`` is covered by ``--x`` (argparse's ``BooleanOptionalAction`` and the
  ``--reground``/``--no-reground`` pair). A row with no flag (``user_instructions``, "panel only") holds no token.
- "Every ``Profile`` field that is a session/quality setting (``config.py:358-444``)": the guard scans every
  ``Profile`` field (the A1 review at 5718460a: the fields before ``tools``, ``name``, ``stt_backend``, ..., are not
  exempt); a field that is not a setting is a ``NOT_A_SETTING`` name with a reason.
- ``default`` holds the live default as a Python value. ``screen_reminder``, ``completion_check`` and ``wait_tool``
  have no constant in ``config.py``: their ``LIVE_DEFAULT_*`` constants live in ``cli.py`` (lines 1670, 1703,
  1744 at BASE), so the row compares against those (a reading: the task says ``config.py:41-122``). ``search``'s
  default ("on in live") has no constant of ``SEARCH_CHOICES``'s shape and is not pinned.
- The v1-changeable boolean rows have ``values`` equal to ``{False, True}`` as bools (``LiveSettings`` takes
  ``type(v) is bool``, A1-3).
- ``group`` is one of B-1's seven panel groups, spelled as in B-1.

The argparse tree is the real one: ``cli.main`` builds it inline (no ``build_parser`` exists at BASE), so
``_session_parser`` patches ``ArgumentParser.parse_args`` to hand back the root parser before anything runs.

BASE is 82ee4503 (main at the A1 test branch point; the T-0213 planning commit, docs only). The rows were written
ahead of A1 and activated at A1 (implementation d7a5e87d); at BASE they fail on a named ``AssertionError`` ("...
does not exist yet"). ``test_session_parser_capture_sees_the_known_flags`` holds at BASE (it proves the capture
seam).

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import create_model

from local_astra import cli, config
from local_astra.config import Diffusion, Profile
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]

#: main when T-0213 A1's tests branched (82ee4503, the planning commit; docs only).
BASE = "82ee450339b8fbf25c3e85246346eb365b05b28a"
#: A1's final implementation tip: 37a2020c (R-6/R-7, on top of d7a5e87d and 5718460a), bumped at the final
#: activation before merge (5718460a before that, the tip the A1 review read).
TIP: str | None = "37a2020cd490d484c0abd19d15858bb94ae5d363"
#: A1-5's range base: ``git merge-base main 37a2020c`` = c2665e1f (checked at the final activation). The A1 branch
#: merged main once (d9af1e40, bringing only c2665e1f, docs), so it is c2665e1f, not BASE; the range
#: c2665e1f..37a2020c holds every A1 product commit (d7a5e87d, 5718460a, 37a2020c).
RANGE_BASE = "c2665e1fbc83eab13d50acb46b62b7ad5d25e103"

KINDS = {"next_turn", "next_turn_cold", "relaunch", "fixed"}
#: B-1's groups, in panel order (written out from the criterion)
GROUPS = (
    "Seeing the screen",
    "Acting in the browser",
    "Checking its work",
    "Guidance and prompt",
    "Search and web",
    "Speech",
    "Engine (relaunch only)",
)

#: The catalog table's rows whose kind and v1 flag are unambiguous, written out by hand from the table.
#: ``RL / v1 read-only`` (search) is relaunch, not changeable; ``later`` rows are not v1-changeable.
TABLE_KIND: dict[str, tuple[str, bool]] = {
    "visual_tokens": ("next_turn_cold", True),
    "reground": ("next_turn", True),
    "live_feedback": ("next_turn", True),
    "action_echo": ("relaunch", False),
    "thought_call_recovery": ("next_turn", True),
    "page_overview": ("next_turn_cold", False),
    "auto_overview": ("next_turn", True),
    "overview_frame": ("next_turn", True),
    "action_outcome": ("next_turn", True),
    "repeat_guard": ("next_turn", True),
    "screen_reminder": ("next_turn", True),
    "wait_tool": ("next_turn_cold", False),
    "completion_check": ("next_turn", True),
    "announced_call_reask": ("next_turn", True),
    "live_guidance": ("next_turn_cold", True),
    "user_instructions": ("next_turn_cold", True),
    "search": ("relaunch", False),
    "browser_web": ("relaunch", False),
    "tools": ("relaunch", False),
    "max_tool_steps": ("relaunch", False),
    "filler_clips": ("relaunch", False),
    "thinking": ("fixed", False),
}

#: The quality flags the table catalogues: none of them may be waved through by ``NOT_A_SETTING`` (the guard's
#: too-wide direction: an exclusion list holding everything would make the drift guard vacuous).
QUALITY_FLAGS = (
    "--visual-tokens",
    "--reground",
    "--reground-visual-tokens",
    "--live-feedback",
    "--action-echo",
    "--thought-call-recovery",
    "--page-overview",
    "--auto-overview",
    "--overview-frame",
    "--action-outcome",
    "--repeat-guard",
    "--screen-reminder",
    "--wait-tool",
    "--completion-check",
    "--announced-call-reask",
    "--live-guidance",
    "--search",
    "--no-web",
    "--tools",
    "--max-tool-steps",
    "--filler-clips",
)


# --- lookups (named failures while the module is absent) -----------------------------------------------------


def _catalog_module() -> Any:
    try:
        return importlib.import_module("local_astra.settings_catalog")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "T-0213 A1-1: local_astra/settings_catalog.py does not exist yet"
        ) from exc


def _catalog() -> tuple[Any, ...]:
    mod = _catalog_module()
    assert hasattr(mod, "SettingEntry"), (
        "T-0213 A1-1: settings_catalog.SettingEntry does not exist yet"
    )
    rows = getattr(mod, "CATALOG", None)
    assert isinstance(rows, tuple) and rows, (
        "T-0213 A1-1: settings_catalog.CATALOG (a non-empty tuple of SettingEntry, assumed name) does not exist yet"
    )
    assert all(isinstance(r, mod.SettingEntry) for r in rows), "every CATALOG row is a SettingEntry"
    return rows


def _by_key() -> dict[str, Any]:
    rows = _catalog()
    keys = [r.key for r in rows]
    assert len(keys) == len(set(keys)), (
        f"catalog keys repeat: {sorted(k for k in keys if keys.count(k) > 1)}"
    )
    return {r.key: r for r in rows}


def _not_a_setting() -> dict[str, str]:
    mod = _catalog_module()
    raw = getattr(mod, "NOT_A_SETTING", None)
    assert isinstance(raw, tuple), (
        "T-0213 A1-2: settings_catalog.NOT_A_SETTING (a tuple) does not exist yet"
    )
    out: dict[str, str] = {}
    for item in raw:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"NOT_A_SETTING holds (name, reason) pairs (assumed shape); got {item!r}"
        )
        name, reason = item
        assert isinstance(name, str) and name, f"NOT_A_SETTING name {name!r}"
        assert isinstance(reason, str) and reason.strip(), f"NOT_A_SETTING[{name!r}] has no reason"
        out[name] = reason
    return out


# --- the real argparse tree ------------------------------------------------------------------------------------


class _Captured(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("captured")
        self.parser = parser


def _session_parser(monkeypatch: pytest.MonkeyPatch) -> argparse.ArgumentParser:
    """The real ``session`` sub-parser that ``cli.main`` builds, captured at its ``parse_args`` call."""

    def capture(self: argparse.ArgumentParser, *a: Any, **k: Any) -> Any:
        raise _Captured(self)

    with monkeypatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(_Captured) as info:
            cli.main(["session"])
    root = info.value.parser
    subs = [a for a in root._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(subs) == 1, "cli.main has one sub-command table"
    return subs[0].choices["session"]


def _session_options(parser: argparse.ArgumentParser) -> set[str]:
    out: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        out.update(s for s in action.option_strings if s.startswith("--"))
    return out


def _all_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Every option string of the parser, argparse's own ``-h``/``--help`` included (the stale-name check: an
    exclusion of a real option is not stale)."""
    return {s for action in parser._actions for s in action.option_strings}


_FLAG_TOKEN = re.compile(r"--(\[no-\])?([a-z0-9][a-z0-9-]*)")


def _flag_tokens(rows: tuple[Any, ...]) -> set[str]:
    tokens: set[str] = set()
    for r in rows:
        flag = r.flag or ""
        for m in _FLAG_TOKEN.finditer(flag):
            tokens.add("--" + m.group(2))
            if m.group(1):
                tokens.add("--no-" + m.group(2))
    return tokens


def _uncovered_flags(
    options: set[str], rows: tuple[Any, ...], excluded: dict[str, str]
) -> list[str]:
    tokens = _flag_tokens(rows)
    names = set(excluded)

    def covered(opt: str) -> bool:
        if opt in tokens or opt in names:
            return True
        if opt.startswith("--no-"):
            pos = "--" + opt[len("--no-") :]
            return pos in tokens or pos in names
        return False

    return sorted(o for o in options if not covered(o))


def _quality_fields(model: type[Profile]) -> list[str]:
    """Every ``Profile`` field, in declaration order (the review's reading of A1-2: the fields before ``tools``
    are not exempt; those that are not settings sit in ``NOT_A_SETTING``)."""
    return list(model.model_fields)


def _uncovered_fields(
    fields: list[str], rows: tuple[Any, ...], excluded: dict[str, str]
) -> list[str]:
    known = (
        {r.key for r in rows}
        | {r.profile_key for r in rows if r.profile_key is not None}
        | set(excluded)
    )
    return sorted(f for f in fields if f not in known)


# --- A1-1: the shape --------------------------------------------------------------------------------------------


def test_session_parser_capture_sees_the_known_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise (holds at BASE): the capture seam reaches the real session parser, not an empty one."""
    opts = _session_options(_session_parser(monkeypatch))
    for flag in (
        *QUALITY_FLAGS,
        "--data-dir",
        "--serve-hud",
        "--no-serve-hud",
        "--no-reground",
        "--dry-run",
    ):
        assert flag in opts, f"premise: {flag} is a session option at BASE"


def test_entry_has_the_a1_1_fields_and_is_frozen() -> None:
    rows = _catalog()
    fields = (
        "key", "label", "group", "kind", "v1_changeable", "values", "flag", "profile_key", "what",
        "useful_when", "not_useful_when", "evidence", "cost", "default",
    )  # fmt: skip
    for r in rows:
        for f in fields:
            assert hasattr(r, f), f"T-0213 A1-1: SettingEntry.{f} does not exist yet ({r!r})"
        assert r.kind in KINDS, f"{r.key}: kind {r.kind!r}"
        assert type(r.v1_changeable) is bool, f"{r.key}: v1_changeable {r.v1_changeable!r}"
        assert isinstance(r.values, tuple), f"{r.key}: values is a tuple"
        assert r.profile_key is None or isinstance(r.profile_key, str)
        for f in ("key", "label", "what", "useful_when", "not_useful_when", "evidence", "cost"):
            assert isinstance(getattr(r, f), str), f"{r.key}.{f} is a str"
        assert r.group in GROUPS, f"{r.key}: group {r.group!r} is not one of B-1's {GROUPS}"
        if r.v1_changeable:
            assert r.kind in ("next_turn", "next_turn_cold"), (
                f"{r.key}: a changeable row is a next-turn kind"
            )
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass or frozen pydantic model, either refusal
        rows[0].key = "mutated"  # type: ignore[misc]


def test_table_rows_and_kinds() -> None:
    """Every unambiguous row of the task's catalog table is a key, with the table's kind and v1 flag."""
    by_key = _by_key()
    missing = sorted(set(TABLE_KIND) - set(by_key))
    assert not missing, f"T-0213 A1-1: catalog rows missing: {missing}"
    wrong = {
        k: ((by_key[k].kind, by_key[k].v1_changeable), want)
        for k, want in TABLE_KIND.items()
        if (by_key[k].kind, by_key[k].v1_changeable) != want
    }
    assert not wrong, f"(got, table) per key: {wrong}"


def test_catalog_imports_nothing_from_conductor_computer_or_search() -> None:
    _catalog_module()
    src = (REPO / "local_astra" / "settings_catalog.py").read_text(encoding="utf-8")
    banned = ("local_astra.conductor", "local_astra.computer", "local_astra.search")
    for node in ast.walk(ast.parse(src)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names = [base] + [f"{base}.{a.name}" for a in node.names]
            assert node.level == 0 or not any(
                a.name in ("conductor", "computer", "search") for a in node.names
            ), "relative import of a banned package"
        for n in names:
            assert not any(n == b or n.startswith(b + ".") for b in banned), (
                f"A1-1: settings_catalog imports {n}"
            )


# --- A1-2: the drift guard --------------------------------------------------------------------------------------


def test_every_session_flag_is_catalogued_or_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, excluded = _catalog(), _not_a_setting()
    options = _session_options(_session_parser(monkeypatch))
    missing = _uncovered_flags(options, rows, excluded)
    assert not missing, (
        f"T-0213 A1-2: session flags neither catalogued nor in NOT_A_SETTING: {missing}"
    )


def test_every_profile_quality_field_is_catalogued() -> None:
    rows, excluded = _catalog(), _not_a_setting()
    fields = _quality_fields(Profile)
    assert "visual_tokens" in fields and "browser_web" in fields, (
        "premise: the span holds the quality fields"
    )
    assert fields[: fields.index("tools")] == list(PRE_TOOLS_FIELDS), (
        f"premise: the fields before tools are {PRE_TOOLS_FIELDS}; got {fields[: fields.index('tools')]}"
    )
    missing = _uncovered_fields(fields, rows, excluded)
    assert not missing, (
        f"T-0213 A1-2: Profile fields neither catalogued nor in NOT_A_SETTING: {missing}"
    )


#: ``Profile``'s fields declared before ``tools`` at 5718460a, written out by hand (the engine and service fields).
PRE_TOOLS_FIELDS = (
    "name", "stt_backend", "tts_backend", "talker", "aec", "vram_budget_gib", "services", "diffusion",
)  # fmt: skip


def test_pre_tools_fields_are_each_accounted_for() -> None:
    """The drift guard scans every field: each field before ``tools`` is a catalog key, a ``profile_key`` or a
    ``NOT_A_SETTING`` name with a reason, one by one (so dropping one exclusion reddens this row by name)."""
    rows, excluded = _catalog(), _not_a_setting()
    for f in PRE_TOOLS_FIELDS:
        assert _uncovered_fields([f], rows, excluded) == [], (
            f"T-0213 A1-2: Profile.{f} is neither catalogued nor in NOT_A_SETTING"
        )


def test_guard_fails_on_an_extra_field_before_tools() -> None:
    """Adversarial: a field declared *before* ``tools`` (pydantic keeps declaration order, a subclass's fields go
    last, so the model is rebuilt with the new field first) is reported. A guard that scans only from ``tools``
    onwards misses it."""
    rows, excluded = _catalog(), _not_a_setting()
    names = list(Profile.model_fields)
    fields_in = {n: (Profile.model_fields[n].annotation, Profile.model_fields[n]) for n in names}
    front = create_model(
        "FrontProfile",
        zz_engine_knob=(int | None, None),
        **fields_in,  # type: ignore[call-overload]
    )
    scanned = _quality_fields(front)
    assert scanned.index("zz_engine_knob") < scanned.index("tools"), "premise: it sits before tools"
    assert _uncovered_fields(scanned, rows, excluded) == ["zz_engine_knob"]


def test_guard_fails_on_an_extra_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial: a flag added to the real ``session`` parser (through argparse itself, as a new feature would)
    is reported by the guard. Proves the guard is not vacuous."""
    rows, excluded = _catalog(), _not_a_setting()
    real_add = argparse._ActionsContainer.add_argument

    def add_argument(self: Any, *args: Any, **kwargs: Any) -> Any:
        action = real_add(self, *args, **kwargs)
        if "--serve-hud" in args:
            real_add(self, "--zz-sharper-clicks", action="store_true")
        return action

    monkeypatch.setattr(argparse._ActionsContainer, "add_argument", add_argument)
    options = _session_options(_session_parser(monkeypatch))
    assert "--zz-sharper-clicks" in options, "premise: the injected flag reached the session parser"
    assert _uncovered_flags(options, rows, excluded) == ["--zz-sharper-clicks"]


def test_guard_fails_on_an_extra_profile_field() -> None:
    """Adversarial: a new ``Profile`` field is reported by the guard."""
    rows, excluded = _catalog(), _not_a_setting()
    extended = create_model(
        "ExtendedProfile", __base__=Profile, zz_sharper_clicks=(bool | None, None)
    )
    assert _uncovered_fields(_quality_fields(extended), rows, excluded) == ["zz_sharper_clicks"]


def test_not_a_setting_is_not_too_wide(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard's too-wide direction: no quality flag and no catalog key is excluded, and every exclusion names
    a real session option or a real ``Profile``/``Diffusion`` field (no stale or invented names)."""
    rows, excluded = _catalog(), _not_a_setting()
    options = _all_option_strings(_session_parser(monkeypatch))
    fields = set(Profile.model_fields) | set(Diffusion.model_fields)
    for flag in QUALITY_FLAGS:
        assert flag not in excluded, f"{flag} is a catalogued setting, not NOT_A_SETTING"
    keys = {r.key for r in rows}
    both = sorted(keys & set(excluded))
    assert not both, f"catalogued and excluded at once: {both}"
    stale = sorted(n for n in excluded if n not in options and n not in fields)
    assert not stale, (
        f"NOT_A_SETTING names that are neither a session option nor a profile field: {stale}"
    )
    # a profile_key is a top-level ``Profile`` field, or ``diffusion.<field>`` for the nested engine table
    # (the implementation's spelling for ``thinking``, the TOML path ``[diffusion] thinking``)
    for r in rows:
        pk = r.profile_key
        if pk is None:
            continue
        if pk.startswith("diffusion."):
            assert pk.removeprefix("diffusion.") in Diffusion.model_fields, (
                f"{r.key}: profile_key {pk!r}"
            )
        else:
            assert pk in Profile.model_fields, f"{r.key}: profile_key {pk!r}"


def test_values_match_config_constants() -> None:
    by_key = _by_key()
    pins = {
        "visual_tokens": config.VISUAL_TOKENS,
        "live_feedback": config.LIVE_FEEDBACK_CHOICES,
        "action_echo": config.ACTION_ECHOES,
        "live_guidance": config.LIVE_GUIDANCE_VERSIONS,
        "search": config.SEARCH_CHOICES,
    }
    for key, want in pins.items():
        assert by_key[key].values == want, (
            f"{key}: values {by_key[key].values!r} != config {want!r}"
        )
    for key in (
        "reground", "thought_call_recovery", "auto_overview", "overview_frame", "action_outcome",
        "repeat_guard", "screen_reminder", "completion_check", "announced_call_reask",
    ):  # fmt: skip
        vals = by_key[key].values
        assert set(vals) == {False, True} and all(type(v) is bool for v in vals), (
            f"{key}: values {vals!r}"
        )
    if "reground_visual_tokens" in by_key:
        vals = by_key["reground_visual_tokens"].values
        assert vals and set(vals) <= set(config.VISUAL_TOKENS), (
            f"reground_visual_tokens: values {vals!r}"
        )


def test_defaults_match_live_default_constants() -> None:
    by_key = _by_key()
    pins: dict[str, object] = {
        "visual_tokens": config.LIVE_DEFAULT_VISUAL_TOKENS,
        "reground": config.LIVE_DEFAULT_REGROUND,
        "live_feedback": config.LIVE_DEFAULT_FEEDBACK,
        "action_echo": config.LIVE_DEFAULT_ACTION_ECHO,
        "thought_call_recovery": config.LIVE_DEFAULT_THOUGHT_CALL_RECOVERY,
        "page_overview": config.LIVE_DEFAULT_PAGE_OVERVIEW,
        "auto_overview": config.LIVE_DEFAULT_AUTO_OVERVIEW,
        "overview_frame": config.LIVE_DEFAULT_OVERVIEW_FRAME,
        "action_outcome": config.LIVE_DEFAULT_ACTION_OUTCOME,
        "repeat_guard": config.LIVE_DEFAULT_REPEAT_GUARD,
        "announced_call_reask": config.DEFAULT_ANNOUNCED_CALL_REASK,
        "live_guidance": config.DEFAULT_LIVE_GUIDANCE,
        # the three without a config constant: cli's (a reading, see the module docstring)
        "screen_reminder": cli.LIVE_DEFAULT_SCREEN_REMINDER,
        "completion_check": cli.LIVE_DEFAULT_COMPLETION_CHECK,
        "wait_tool": cli.LIVE_DEFAULT_WAIT_TOOL,
        # the tool-step budgets live in cli too; the catalog holds them as literals (it may not import cli)
        "max_tool_steps": cli.DEFAULT_TOOL_STEPS,
        "thinking": Diffusion().thinking,
        "user_instructions": "",
        "browser_web": Profile.model_fields["browser_web"].default,
        "filler_clips": Profile.model_fields["filler_clips"].default,
    }
    if "reground_visual_tokens" in by_key:
        pins["reground_visual_tokens"] = config.LIVE_DEFAULT_REGROUND_VISUAL_TOKENS
    if "max_tool_steps_web" in by_key:
        pins["max_tool_steps_web"] = cli.DEFAULT_TOOL_STEPS_WEB
    wrong = {
        k: (by_key[k].default, v)
        for k, v in pins.items()
        if by_key[k].default != v or type(by_key[k].default) is not type(v)
    }
    assert not wrong, f"(catalog default, constant) per key: {wrong}"


# --- A1-1: the text --------------------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def test_text_lengths() -> None:
    """``what`` is at most 2 sentences and 300 code points; every text field is non-empty."""
    for r in _catalog():
        assert r.what.strip(), f"{r.key}: what is empty"
        assert len(r.what) <= 300, f"{r.key}: what is {len(r.what)} code points"
        n = len(_SENTENCE_END.findall(r.what.strip()))
        assert 1 <= n <= 2, f"{r.key}: what has {n} sentences: {r.what!r}"
        assert r.label.strip(), f"{r.key}: label is empty"
        for f in ("useful_when", "not_useful_when", "evidence", "cost"):
            assert getattr(r, f).strip(), (
                f"{r.key}.{f} is empty (write '—' where the table has none)"
            )


def test_every_entry_cites_a_task_id() -> None:
    no_id = [r.key for r in _catalog() if not re.search(r"\bT-\d{4}[a-z]?\b", r.evidence)]
    assert not no_id, f"T-0213 A1-1: evidence without a task id: {no_id}"


def test_relaunch_rows_have_a_command() -> None:
    """B-2/A2-read: a relaunch row carries the exact command in ``relaunch`` (the implementation's field, the
    ``SettingItem.relaunch`` the panel shows); a row the panel can change carries none."""
    seen = 0
    for r in _catalog():
        if r.kind != "relaunch":
            if r.v1_changeable:
                assert r.relaunch is None, (
                    f"{r.key}: a panel-changeable row has no relaunch command"
                )
            continue
        seen += 1
        assert isinstance(r.relaunch, str) and "diffusion_gearvis" in r.relaunch, (
            f"{r.key}: a relaunch row names its command (diffusion_gearvis ...); relaunch={r.relaunch!r}"
        )
        if r.flag:
            assert any(f in r.relaunch for f in _flag_spellings(r)), (
                f"{r.key}: the relaunch command uses the row's own flag {r.flag!r}: {r.relaunch!r}"
            )
        assert not r.v1_changeable, f"{r.key}: a relaunch row is never v1-changeable"
    assert seen >= 5, (
        f"premise: the table has relaunch rows (search, browser_web, tools, ...); saw {seen}"
    )


def _flag_spellings(r: Any) -> set[str]:
    return {m.group(0).replace("[no-]", "") for m in _FLAG_TOKEN.finditer(r.flag)} | {
        m.group(0).replace("[no-]", "no-") for m in _FLAG_TOKEN.finditer(r.flag)
    }


def test_flags_are_spellings_of_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SettingEntry.flags`` (the implementation's every-spelling field) holds only real session options, and the
    row's ``flag`` is one of them (or both are empty: a setting no flag sets)."""
    options = _all_option_strings(_session_parser(monkeypatch))
    for r in _catalog():
        assert isinstance(r.flags, tuple), f"{r.key}: flags is a tuple"
        if not r.flag:
            assert r.flags == (), f"{r.key}: no flag, so no spellings: {r.flags!r}"
            continue
        bad = sorted(f for f in r.flags if f not in options)
        assert not bad, f"{r.key}: flags that are not session options: {bad}"
        assert r.flags, f"{r.key}: flag {r.flag!r} but no spellings"
        assert r.flag.split()[0].replace("[no-]", "") in r.flags, (
            f"{r.key}: {r.flag!r} not in {r.flags!r}"
        )


def test_thinking_row_is_fixed() -> None:
    row = _by_key()["thinking"]
    assert row.kind == "fixed" and row.v1_changeable is False
    assert "<|think|>" in row.what + row.evidence + row.not_useful_when + row.cost, (
        "the thinking row says why: <|think|> is the first system token (A2a-6)"
    )


# --- A1-5: frozen paths -----------------------------------------------------------------------------------------

#: A1-5's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/cli.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "hud",
)


def _blobs_at(rev: str) -> dict[str, str]:
    proc = gr._git("ls-tree", "-r", rev, "--", *FROZEN)
    assert proc.returncode == 0, proc.stderr
    out = {}
    for line in proc.stdout.splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def test_frozen_paths_unchanged() -> None:
    """A1-5: ``cli.py``, ``session.py``, ``turn_machine.py``, ``contracts/**`` and ``hud/**`` are byte-identical
    over ``RANGE_BASE..TIP`` (two literals; ``TIP`` is A1's implementation commit, ``RANGE_BASE`` its merge-base
    with main)."""
    assert TIP is not None, (
        "T-0213 A1-5: the TIP literal (A1's implementation commit) does not exist yet"
    )
    gr.require_commits(BASE, RANGE_BASE, TIP)
    assert gr.is_ancestor(BASE, RANGE_BASE), f"{BASE[:8]} is not an ancestor of {RANGE_BASE[:8]}"
    assert gr.is_ancestor(RANGE_BASE, TIP), f"{RANGE_BASE[:8]} is not an ancestor of {TIP[:8]}"
    assert not gr.is_ancestor(TIP, RANGE_BASE), "premise: the range is not empty"
    at_base = _blobs_at(RANGE_BASE)
    assert "local_astra/cli.py" in at_base and any(p.startswith("hud/") for p in at_base), "premise"
    at_tip = _blobs_at(TIP)
    changed = sorted(p for p in set(at_base) | set(at_tip) if at_base.get(p) != at_tip.get(p))
    assert changed == [], (
        f"T-0213 A1-5: frozen paths differ in {RANGE_BASE[:8]}..{TIP[:8]}: {changed}"
    )


def test_frozen_paths_power_premise() -> None:
    """A1-5 power: the same blob comparison sees a known change to a frozen path (T-0213 A1's own range does change
    ``prompt_builder.py``, which lies under ``conductor/`` but outside FROZEN; ``cli.py`` changed in some commit
    before BASE). The guard cannot pass by seeing nothing."""
    gr.require_commits(RANGE_BASE, TIP)
    proc = gr._git("ls-tree", "-r", RANGE_BASE, "--", "local_astra/conductor/prompt_builder.py")
    tip = gr._git("ls-tree", "-r", TIP, "--", "local_astra/conductor/prompt_builder.py")
    assert proc.returncode == 0 and tip.returncode == 0
    assert proc.stdout.split()[2] != tip.stdout.split()[2], (
        "premise: ls-tree blobs see A1's prompt_builder change"
    )


# --- R-7: the catalog wording (A1 review at 5718460a) -----------------------------------------------------------


def test_r7_v4_evidence_names_the_shared_handoff_and_the_typed_result() -> None:
    """R-7: the live_guidance row's v4 evidence says it was worse on the shared handoff (3/9 to 1/9) and gives the
    typed result (6/9 to 7/9), so the panel does not read as "v4 failed everything"."""
    row = _by_key()["live_guidance"]
    text = row.evidence
    assert "T-0211" in text and "shared handoff" in text, text
    for n in ("3/9", "1/9", "6/9", "7/9"):
        assert n in text, f"R-7: v4 evidence lacks {n}: {text!r}"
    assert "typed" in text, f"R-7: v4 evidence names the typed arm: {text!r}"
    assert text.index("3/9") < text.index("1/9") and text.index("6/9") < text.index("7/9"), (
        f"R-7: each result reads before -> after: {text!r}"
    )


def test_r7_reground_crop_budget_cites_the_crop_result() -> None:
    """R-7: ``reground_visual_tokens``' useful_when quotes the crop-budget result (T-0132 C: crop 280 64/100, crop
    1120 66/100), not the main-budget 280 result (T-0131 D: 5 -> 15/25), which belongs to the reground row."""
    row = _by_key()["reground_visual_tokens"]
    text = row.useful_when
    assert "T-0132" in text, f"R-7: useful_when cites T-0132: {text!r}"
    for n in ("64/100", "66/100"):
        assert n in text, f"R-7: useful_when lacks {n}: {text!r}"
    assert "15/25" not in text, f"R-7: the main-budget result is not this row's: {text!r}"


# --- T-0227 named re-pin: the parity tables gain the two frame rows (docs/tasks/T-0227.md A-2) ------------------

#: T-0227 A-2, typed from the criterion: (kind, v1 flag), the config constant names for values and default, the flag
T0227_ROWS: dict[str, tuple[str, bool, str, str, str]] = {
    "max_frames_per_turn": (
        "next_turn_cold",
        True,
        "FRAMES_PER_TURN_CHOICES",
        "DEFAULT_MAX_FRAMES_PER_TURN",
        "--max-frames-per-turn",
    ),
    "frame_interval_ms": (
        "next_turn",
        True,
        "FRAME_INTERVALS_MS",
        "DEFAULT_FRAME_INTERVAL_MS",
        "--frame-interval-ms",
    ),
}


def test_t0227_rows_join_the_parity_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-0227 re-pin of ``test_table_rows_and_kinds``, ``test_values_match_config_constants`` and
    ``test_defaults_match_live_default_constants``: the two rows have the A-2 kind and v1 flag, ``values`` equal the
    config constant (ints), ``default`` equals the config default (an int); each flag is a session option, catalogued
    (A1-2's drift guard sees it) and not waved through by ``NOT_A_SETTING``; each ``profile_key`` is a Profile field."""
    by_key = _by_key()
    missing = sorted(set(T0227_ROWS) - set(by_key))
    assert not missing, f"T-0227 A-2: catalog rows missing: {missing}"
    opts = _session_options(_session_parser(monkeypatch))
    excluded = set(_not_a_setting())
    for key, (kind, v1, values_name, default_name, flag) in T0227_ROWS.items():
        r = by_key[key]
        assert (r.kind, r.v1_changeable) == (kind, v1), (key, r.kind, r.v1_changeable)
        values = getattr(config, values_name, None)
        default = getattr(config, default_name, None)
        assert values is not None and default is not None, (
            f"T-0227 A-1: config.{values_name} / config.{default_name}"
        )
        assert r.values == values and all(type(v) is int for v in r.values), (key, r.values)
        assert r.default == default and type(r.default) is int, (key, r.default)
        assert r.flag == flag and flag in opts and flag not in excluded, (key, r.flag)
        assert r.profile_key == key and key in Profile.model_fields, (key, r.profile_key)
