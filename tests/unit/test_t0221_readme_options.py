"""T-0221: the settings catalog's notes brought up to date, and the README options table generated from it.

Derived from docs/tasks/T-0221.md (criteria 1-4) and nothing else. Interfaces fixed by the planner's brief:

- ``scripts/gen_readme_options.py`` exposes ``render(catalog) -> str``, the README block *between* the two markers
  (markers excluded), and ``main()``, which rewrites ``README.md`` in place. It is loaded by path, lazily, so this
  module collects before the script exists.
- The markers are ``<!-- options:begin -->`` and ``<!-- options:end -->``; the README text strictly between them is
  byte-identical to ``render(CATALOG)`` (criterion 3).

Readings where the task is open (listed in the test-author's report):

- "real notes" (criterion 1) is checked for ``useful_when`` and ``not_useful_when`` only, as the criterion says;
  ``cost`` may stay "—".
- The README "Default" cell is only required to be non-empty (the task does not say whether it shows ``default`` or
  ``default_note``). The "Changes" cell must say "restart" and carry the flag for a row with a relaunch command,
  and "panel" and the flag for a panel-changeable row; the fixed rows and ``context_helpers`` are not constrained.
- Groups: each group name appears in the block in ``GROUPS`` order and every entry's row lies after its own group's
  name and before the next group's (group headings or group rows, either).
- The reground spot row wants the T-0212 numbers in ``useful_when`` (35/96 before 10/74) and "n = 15" and
  "three.js" anywhere in the row's text fields.

The structural pin (criterion 4) is a literal written from ``git show 325d7a99:local_astra/settings_catalog.py``
executed as an isolated module (``_t0221_base_catalog``) at test-writing time; every field other than the five text
fields is pinned, in catalog order.

Rows that failed at BASE 325d7a99 (main at the T-0221 planning commit) carried a T-0221 pending marker, removed at
activation; they failed with an ``AssertionError``.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra import settings_catalog as sc
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gen_readme_options.py"
README = REPO / "README.md"
BEGIN = "<!-- options:begin -->"
END = "<!-- options:end -->"
REGEN = "uv run python scripts/gen_readme_options.py"

#: main at the T-0221 planning commit (the test branch point).
BASE = "325d7a99"

TEXT_FIELDS = ("what", "useful_when", "not_useful_when", "evidence", "cost")

#: Criterion 4: the SettingEntry field names at BASE, in order.
BASE_FIELDS = (
    "key", "label", "group", "kind", "v1_changeable", "values", "default", "default_note", "flag", "flags",
    "profile_key", "what", "useful_when", "not_useful_when", "evidence", "cost", "relaunch",
)  # fmt: skip

#: Criterion 4: every non-text field of every entry at BASE, in catalog order:
#: (key, label, group, kind, v1_changeable, values, default, default_note, flag, flags, profile_key, relaunch)
_S = "diffusion_gearvis session --tools live"
BASE_STRUCTURE: tuple[tuple[Any, ...], ...] = (
    ("visual_tokens", "Visual tokens per image", "Seeing the screen", "next_turn_cold", True, (280, 560, 1120), 1120, "1120 in live (T-0132 A1)", "--visual-tokens", ("--visual-tokens",), "visual_tokens", None),
    ("reground", "ReGround", "Seeing the screen", "next_turn", True, (False, True), False, "off (T-0133)", "--reground", ("--reground", "--no-reground"), "reground", None),
    ("reground_visual_tokens", "ReGround crop tokens", "Seeing the screen", "next_turn", True, (280, 560, 1120), 280, "280 with ReGround on at a 1120 main budget (T-0133 A5), else the main budget", "--reground-visual-tokens", ("--reground-visual-tokens",), "reground_visual_tokens", None),
    ("live_feedback", "Screen-now label", "Seeing the screen", "next_turn", True, ("off", "label", "label+extent"), "label", "label (T-0134c)", "--live-feedback", ("--live-feedback",), "live_feedback", None),
    ("page_overview", "Page overview tool", "Seeing the screen", "next_turn_cold", False, (False, True), False, "off", "--page-overview", ("--page-overview", "--no-page-overview"), "page_overview", f"{_S} --page-overview"),
    ("auto_overview", "Automatic page overview", "Seeing the screen", "next_turn", True, (False, True), False, "off", "--auto-overview", ("--auto-overview", "--no-auto-overview"), "auto_overview", None),
    ("overview_frame", "Overview frame", "Seeing the screen", "next_turn", True, (False, True), True, "on, on 1.05–2.0-screen pages (T-0152)", "--overview-frame", ("--overview-frame", "--no-overview-frame"), "overview_frame", None),
    ("context_helpers", "Context helpers (page text, PDF text, cells, accessibility tree)", "Seeing the screen", "next_turn", False, (), False, "off (D12a)", "", (), None, None),
    ("action_echo", "Action echo units", "Acting in the browser", "relaunch", False, ("pixels", "model"), "model", "model", "--action-echo", ("--action-echo",), "action_echo", f"{_S} --action-echo pixels"),
    ("thought_call_recovery", "Thought-call recovery", "Acting in the browser", "next_turn", True, (False, True), True, "on (T-0135b)", "--thought-call-recovery", ("--thought-call-recovery",), "thought_call_recovery", None),
    ("wait_tool", "Wait tool", "Acting in the browser", "next_turn_cold", False, (False, True), True, "on (T-0202 D)", "--wait-tool", ("--wait-tool",), None, f"{_S} --wait-tool off"),
    ("max_tool_steps", "Tool steps per turn", "Acting in the browser", "relaunch", False, (), 8, "8, web 20 (T-0178)", "--max-tool-steps", ("--max-tool-steps",), "max_tool_steps", f"{_S} --max-tool-steps N"),
    ("max_tool_steps_web", "Tool steps per turn (web)", "Acting in the browser", "relaunch", False, (), 20, "20 in live with web on (T-0178)", "--max-tool-steps", ("--max-tool-steps",), "max_tool_steps_web", f"{_S} --max-tool-steps N"),
    ("browser", "Agent browser window", "Acting in the browser", "relaunch", False, ("headed", "headless"), "headed", "headed", "--browser", ("--browser",), "browser", f"{_S} --browser headless"),
    ("browser_viewport", "Agent browser size", "Acting in the browser", "relaunch", False, (), (1280, 800), "1280×800", "", (), "browser_viewport", "set browser_viewport in astra.toml, then relaunch diffusion_gearvis session"),
    ("tools", "Tools mode", "Acting in the browser", "relaunch", False, ("off", "dry_run", "live"), "off", "off (profile)", "--tools", ("--tools",), "tools", _S),
    ("action_outcome", "Action outcome", "Checking its work", "next_turn", True, (False, True), False, "off", "--action-outcome", ("--action-outcome", "--no-action-outcome"), "action_outcome", None),
    ("repeat_guard", "Repeat guard", "Checking its work", "next_turn", True, (False, True), True, "on (T-0179, James's ruling)", "--repeat-guard", ("--repeat-guard", "--no-repeat-guard"), "repeat_guard", None),
    ("screen_reminder", "Screen reminder", "Checking its work", "next_turn", True, (False, True), True, "on (T-0201 D)", "--screen-reminder", ("--screen-reminder",), None, None),
    ("completion_check", "Completion check", "Checking its work", "next_turn", True, (False, True), True, "on (T-0207 C)", "--completion-check", ("--completion-check",), None, None),
    ("announced_call_reask", "Re-ask an announced call", "Checking its work", "next_turn", True, (False, True), True, "on (T-0172)", "--announced-call-reask", ("--announced-call-reask",), "announced_call_reask", None),
    ("live_guidance", "Screenshot guidance", "Guidance and prompt", "next_turn_cold", True, ("v1", "v2", "v3", "v4"), "v3", "v3 (T-0191 C)", "--live-guidance", ("--live-guidance",), "live_guidance", None),
    ("user_instructions", "Your instructions to Gearvis", "Guidance and prompt", "next_turn_cold", True, (), "", "empty", "", (), None, None),
    ("thinking", "Thinking", "Guidance and prompt", "fixed", False, (True,), True, "on", "", (), "diffusion.thinking", None),
    ("search", "Web search", "Search and web", "relaunch", False, ("off", "ddgs"), "ddgs", "on in live (T-0125)", "--search", ("--search",), "search", f"{_S} --search off"),
    ("browser_web", "Real websites", "Search and web", "relaunch", False, (False, True), True, "web on", "--no-web", ("--no-web",), "browser_web", f"{_S} --no-web"),
    ("filler_clips", "Filler clips", "Speech", "relaunch", False, (False, True), False, "off (James 2026-10-28)", "--filler-clips", ("--filler-clips", "--no-filler-clips"), "filler_clips", "diffusion_gearvis session --filler-clips"),
    ("honest_ending_clips", "Honest-ending clips (out_of_steps, not_finished, lost_thread, at_checkout)", "Speech", "fixed", False, (True,), True, "on", "", (), None, None),
    ("engine", "Engine: canvas, steps, max_num_seqs, KV, max-model-len, --diffusion-config, STT/TTS backend, profile", "Engine (relaunch only)", "relaunch", False, (), None, "per `services/versions.lock` and the profile", "--profile", ("--profile",), None, "diffusion_gearvis down && diffusion_gearvis up --profile lean"),
)  # fmt: skip
_STRUCT_FIELDS = (
    "key", "label", "group", "kind", "v1_changeable", "values", "default", "default_note", "flag", "flags",
    "profile_key", "relaunch",
)  # fmt: skip

BASE_GROUPS = (
    "Seeing the screen",
    "Acting in the browser",
    "Checking its work",
    "Guidance and prompt",
    "Search and web",
    "Speech",
    "Engine (relaunch only)",
)
BASE_KINDS = ("next_turn", "next_turn_cold", "relaunch", "fixed")
BASE_RELAUNCH_IN_FORCE = {
    "search": ("off", "--search ddgs"),
    "browser": ("headless", "--browser headed"),
    "wait_tool": (False, "--wait-tool on"),
    "page_overview": (True, "--no-page-overview"),
    "action_echo": ("pixels", "--action-echo model"),
    "filler_clips": (True, "--no-filler-clips"),
    "browser_web": (False, None),
    "tools": ("live", None),
}

#: The keys whose useful_when or not_useful_when is "—" (or empty) at BASE: these rows are T-0221's to fix.
DASH_AT_BASE = frozenset(
    {
        "page_overview",
        "action_echo",
        "wait_tool",
        "browser",
        "browser_viewport",
        "tools",
        "screen_reminder",
        "announced_call_reask",
        "thinking",
        "honest_ending_clips",
        "engine",
    }
)


def _by_key() -> dict[str, sc.SettingEntry]:
    return {e.key: e for e in sc.CATALOG}


def _row_text(e: sc.SettingEntry) -> str:
    return " ".join(getattr(e, f) for f in TEXT_FIELDS)


def _load_script() -> ModuleType:
    assert SCRIPT.is_file(), f"T-0221 criterion 2: {SCRIPT.relative_to(REPO)} does not exist yet"
    name = "_t0221_gen_readme_options"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    assert callable(getattr(mod, "render", None)), "gen_readme_options.render(catalog) is missing"
    assert callable(getattr(mod, "main", None)), "gen_readme_options.main() is missing"
    return mod


def _block(text: str) -> str:
    assert text.count(BEGIN) == 1, (
        f"README must hold {BEGIN!r} exactly once (holds it {text.count(BEGIN)}x)"
    )
    assert text.count(END) == 1, (
        f"README must hold {END!r} exactly once (holds it {text.count(END)}x)"
    )
    b = text.index(BEGIN) + len(BEGIN)
    e = text.index(END)
    assert b <= e, "the begin marker comes before the end marker"
    return text[b:e]


# --- criterion 1: the notes ----------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\b(todo|tbd|fixme|xxx|placeholder|lorem)\b", re.IGNORECASE)
_ONLY_PUNCT = re.compile(r"^[\s\-–—_.?…/*]*$")
_TRIVIAL = frozenset({"n/a", "na", "none", "nothing", "-", "—", "–", "?", "..."})


def _is_placeholder(s: str) -> bool:
    t = s.strip()
    return (
        not t
        or bool(_ONLY_PUNCT.match(t))
        or t.lower().rstrip(".") in _TRIVIAL
        or bool(_PLACEHOLDER.search(t))
    )


def test_placeholder_detector_power() -> None:
    """Premise: the detector refuses the shapes the criterion names and accepts the task's own example."""
    for bad in ("", " ", "—", "-", "–", " — ", "TBD", "todo: write", "n/a", "...", "?"):
        assert _is_placeholder(bad), repr(bad)
    for good in (
        "always on: switching it per request would make every reply slow",
        "slow-loading pages",
        "almost never",
    ):
        assert not _is_placeholder(good), repr(good)


@pytest.mark.parametrize(
    "key",
    [row[0] for row in BASE_STRUCTURE],
)
def test_every_entry_has_real_notes(key: str) -> None:
    """Criterion 1: no ``useful_when``/``not_useful_when`` is empty, "—", "-" or a placeholder."""
    e = _by_key()[key]
    for f in ("useful_when", "not_useful_when"):
        v = getattr(e, f)
        assert not _is_placeholder(v), f"T-0221 criterion 1: {key}.{f} is a placeholder: {v!r}"


def test_no_dash_constant_left_in_the_notes() -> None:
    """Criterion 1, adversarial (the drift the task names): across the whole catalog, not one note is a dash. The
    per-key rows above cover the same ground one row at a time; this one fails if a new row is added with "—"."""
    bad = [
        f"{e.key}.{f}"
        for e in sc.CATALOG
        for f in ("useful_when", "not_useful_when")
        if _is_placeholder(getattr(e, f))
    ]
    assert not bad, f"T-0221 criterion 1: placeholder notes: {bad}"


_TASK_ID = re.compile(r"\bT-(\d{4})([a-z]?)\b")


def _task_file_exists(num: str, suffix: str) -> bool:
    tasks = REPO / "docs" / "tasks"
    return (tasks / f"T-{num}{suffix}.md").is_file() or (tasks / f"T-{num}.md").is_file()


@pytest.mark.parametrize("key", ["max_frames_per_turn", "frame_interval_ms"])
def test_t0227_rows_have_real_notes(key: str) -> None:
    """T-0227 A-2 under T-0221's rules: the two new rows have real ``useful_when``/``not_useful_when`` notes, every
    text field non-empty and short (one sentence or a fragment: at most 300 code points here, a reading)."""
    found = [e for e in sc.CATALOG if e.key == key]
    assert len(found) == 1, f"T-0227 A-2: catalog row {key} does not exist yet"
    e = found[0]
    for f in ("useful_when", "not_useful_when"):
        assert not _is_placeholder(getattr(e, f)), f"T-0227 A-2: {key}.{f} is a placeholder"
    for f in TEXT_FIELDS:
        v = getattr(e, f)
        assert v.strip() and len(v) <= 300, f"T-0227 A-2: {key}.{f} is empty or long ({len(v)})"


def test_quoted_runs_exist() -> None:
    """Criterion 1: every ``T-NNNN`` cited in ``evidence``/``useful_when``/``not_useful_when`` has a task file
    (``docs/tasks/T-NNNN.md``, or ``T-NNNNx.md`` for a lettered sub-task)."""
    cited = 0
    missing = []
    for e in sc.CATALOG:
        for f in ("evidence", "useful_when", "not_useful_when"):
            for m in _TASK_ID.finditer(getattr(e, f)):
                cited += 1
                if not _task_file_exists(m.group(1), m.group(2)):
                    missing.append(f"{e.key}.{f}: {m.group(0)}")
    assert cited >= 29, f"premise: the catalog cites task ids (saw {cited})"
    assert not missing, f"T-0221 criterion 1: cited ids without a docs/tasks file: {missing}"


def test_quoted_runs_detector_power() -> None:
    """Premise: the existence check refuses an id that has no task file."""
    assert _task_file_exists("0212", "") and _task_file_exists("0111", "g")
    assert not _task_file_exists("9999", "")


def test_reground_states_the_small_target_result() -> None:
    """Criterion 1: reground is useful on small or dense targets and quotes T-0212's misses 35/96 -> 10/74 (n = 15
    per arm, three.js); it is no longer a copy of the crop row's text (the drift the task names)."""
    rows = _by_key()
    e = rows["reground"]
    u = e.useful_when
    assert u != rows["reground_visual_tokens"].useful_when, (
        f"T-0221 criterion 1: reground.useful_when is still the crop row's text: {u!r}"
    )
    assert "T-0212" in u, f"reground.useful_when cites T-0212: {u!r}"
    for n in ("35/96", "10/74"):
        assert n in u, f"reground.useful_when lacks {n}: {u!r}"
    assert u.index("35/96") < u.index("10/74"), f"reads before -> after: {u!r}"
    assert re.search(r"\b(small|dense)\b", u, re.IGNORECASE), f"names small or dense targets: {u!r}"
    text = _row_text(e)
    assert re.search(r"n\s*=\s*15\b", text), f"reground states n = 15 per arm: {text!r}"
    assert "three.js" in text, f"reground names the three.js editor: {text!r}"


def test_reground_not_useful_on_ordinary_pages() -> None:
    """Criterion 1: not useful on ordinary pages (T-0132 C 65 vs 64/100); refused with the overview frame."""
    n = _by_key()["reground"].not_useful_when
    assert "T-0132" in n, f"reground.not_useful_when cites T-0132: {n!r}"
    assert re.search(r"\b65\b", n) and "64/100" in n, f"quotes 65 vs 64/100: {n!r}"
    assert "overview frame" in n, f"says it is refused with the overview frame: {n!r}"
    assert re.search(r"request", _row_text(_by_key()["reground"])), (
        "states the extra request per click"
    )


def test_reground_crop_tokens_keep_their_crop_budget_text() -> None:
    """Criterion 1 (preservation): reground_visual_tokens keeps the crop-budget result (T-0132 C crop 280 64/100 vs
    1120 66/100). Holds at BASE by design."""
    u = _by_key()["reground_visual_tokens"].useful_when
    assert "T-0132" in u and "64/100" in u and "66/100" in u, u


def test_action_outcome_says_no_measured_gain() -> None:
    """Criterion 1: action_outcome says no measured gain (T-0212 outcome 0/15, misses 23/69 vs base 35/96; T-0214
    with ReGround no sign beyond ReGround alone); for experiments. The stale "T-0212 measures it" is gone."""
    e = _by_key()["action_outcome"]
    text = _row_text(e)
    assert "measures it" not in text, (
        f"T-0221: the stale 'T-0212 measures it' is still there: {text!r}"
    )
    assert "no measured gain" in text, text
    for s in ("T-0212", "0/15", "23/69", "35/96", "T-0214"):
        assert s in text, f"action_outcome lacks {s}: {text!r}"
    assert "experiment" in e.useful_when, f"useful_when says for experiments: {e.useful_when!r}"


def test_search_says_off_withdraws_the_tool() -> None:
    """Criterion 1: search off withdraws the ``google_search`` tool (T-0215); useful for questions that need the
    web; off for private or offline work."""
    e = _by_key()["search"]
    text = _row_text(e)
    assert "T-0215" in text, f"search cites T-0215: {text!r}"
    assert "google_search" in text, text
    assert re.search(r"withdr[ae]w", text, re.IGNORECASE), (
        f"search says off withdraws the tool: {text!r}"
    )
    assert "web" in e.useful_when, e.useful_when
    assert re.search(r"private|offline", e.not_useful_when), e.not_useful_when


# --- criterion 4: structure unchanged --------------------------------------------------------------------------


#: T-0227 amends criterion 4: exactly these two rows are inserted directly after ``visual_tokens``, in this order
T0227_INSERTED = ("max_frames_per_turn", "frame_interval_ms")


# T-0227 re-pin (named, docs/tasks/T-0227.md Owned files): criterion 4 as amended by T-0227
def test_catalog_structure_unchanged() -> None:
    """Criterion 4 (as amended by T-0227): the 29 T-0221 rows keep key, label, group, kind, v1_changeable, values,
    default, default_note, flag, flags, profile_key and relaunch, and their relative order, equal BASE's (literal
    pinned from ``git show 325d7a99``), and exactly ``max_frames_per_turn``, ``frame_interval_ms`` are inserted
    directly after ``visual_tokens``. The base pin (``test_catalog_structure_pin_is_base``) stays."""
    assert tuple(f.name for f in dataclasses.fields(sc.SettingEntry)) == BASE_FIELDS
    keys = [e.key for e in sc.CATALOG]
    base_keys = [r[0] for r in BASE_STRUCTURE]
    i = base_keys.index("visual_tokens") + 1
    assert keys == [*base_keys[:i], *T0227_INSERTED, *base_keys[i:]], (
        f"T-0227 amended criterion 4: the two rows follow visual_tokens and nothing else moves; got {keys}"
    )
    got = tuple(
        tuple(getattr(e, f) for f in _STRUCT_FIELDS)
        for e in sc.CATALOG
        if e.key not in T0227_INSERTED
    )
    assert [r[0] for r in got] == [r[0] for r in BASE_STRUCTURE], (
        "T-0221 criterion 4: keys or order changed"
    )
    wrong = {
        g[0]: {
            f: (gv, bv)
            for f, gv, bv in zip(_STRUCT_FIELDS, g, b, strict=True)
            if gv != bv or type(gv) is not type(bv)
        }
        for g, b in zip(got, BASE_STRUCTURE, strict=True)
    }
    wrong = {k: v for k, v in wrong.items() if v}
    assert not wrong, f"T-0221 criterion 4: non-text fields changed (now, base): {wrong}"
    assert sc.GROUPS == BASE_GROUPS and sc.KINDS == BASE_KINDS
    assert dict(sc.RELAUNCH_IN_FORCE) == BASE_RELAUNCH_IN_FORCE


def test_catalog_structure_pin_is_base() -> None:
    """Premise: the literal pin is BASE's catalog, re-read from ``git show`` and executed as an isolated module."""
    gr.require_commits(BASE)
    proc = gr._git("show", f"{BASE}:local_astra/settings_catalog.py")
    assert proc.returncode == 0, proc.stderr
    name = "_t0221_base_catalog"
    mod = ModuleType(name)
    mod.__file__ = f"<{BASE}:local_astra/settings_catalog.py>"
    sys.modules[name] = mod
    try:
        exec(compile(proc.stdout, mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    base = tuple(tuple(getattr(e, f) for f in _STRUCT_FIELDS) for e in mod.CATALOG)
    assert base == BASE_STRUCTURE
    assert len(base) == 29


# --- criteria 2 and 3: the README table ------------------------------------------------------------------------


def test_readme_options_block_matches_the_catalog() -> None:
    """Criterion 3: the README block between the markers is byte-identical to ``render(CATALOG)``; the failure
    names the command to run."""
    mod = _load_script()
    want = mod.render(sc.CATALOG)
    assert isinstance(want, str) and want.strip(), "render(CATALOG) returns the block text"
    got = _block(README.read_text(encoding="utf-8"))
    assert got == want, f"README options table is out of date with the catalog; run: {REGEN}"


def test_render_follows_the_catalog_text() -> None:
    """Criterion 3, adversarial: render reads the catalog (a changed note changes the block and shows up in it), so
    the drift test cannot pass against a frozen string."""
    mod = _load_script()
    base = mod.render(sc.CATALOG)
    marker = "only when the moon is full (T-0221 probe)"
    i = [e.key for e in sc.CATALOG].index("action_outcome")
    changed = list(sc.CATALOG)
    changed[i] = dataclasses.replace(changed[i], useful_when=marker)
    out = mod.render(tuple(changed))
    assert out != base and marker in out and marker not in base
    j = [e.key for e in sc.CATALOG].index("browser")
    fewer = tuple(e for k, e in enumerate(sc.CATALOG) if k != j)
    assert sc.CATALOG[j].label not in mod.render(fewer), "a dropped entry drops its row"


_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _cells(line: str) -> list[str]:
    s = line.strip()
    assert s.startswith("|") and s.endswith("|"), line
    return [c.strip().replace("\\|", "|") for c in _CELL_SPLIT.split(s[1:-1])]


def _table(block: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """(header cells, [(line index, cells)]) of every data row in ``block``; one header (repeated per group is
    allowed, equal each time)."""
    lines = block.splitlines()
    header: list[str] | None = None
    rows: list[tuple[int, list[str]]] = []
    k = 0
    while k < len(lines):
        ln = lines[k]
        if ln.strip().startswith("|"):
            nxt = lines[k + 1] if k + 1 < len(lines) else ""
            if re.match(r"^\s*\|?\s*:?-{3,}", nxt):
                h = _cells(ln)
                assert header is None or h == header, f"every table has one header: {h} vs {header}"
                header = h
                k += 2
                continue
            rows.append((k, _cells(ln)))
        k += 1
    assert header is not None, "the block holds a markdown table with a header row"
    return header, rows


def test_readme_table_has_one_row_per_entry_in_catalog_order_and_groups() -> None:
    """Criterion 2: columns Setting, Default, Useful when, Not useful when, Changes; one row per entry in catalog
    order, under its group; the notes are the catalog's; Changes says panel or restart, with the flag."""
    block = _block(README.read_text(encoding="utf-8"))
    header, rows = _table(block)
    assert [h.lower() for h in header] == [
        "setting",
        "default",
        "useful when",
        "not useful when",
        "changes",
    ], header
    entry_rows = [
        (i, c) for i, c in rows if len(c) == 5 and any(e.label in c[0] for e in sc.CATALOG)
    ]
    assert len(entry_rows) == len(sc.CATALOG), (
        f"one row per catalog entry: {len(entry_rows)} rows vs {len(sc.CATALOG)} entries"
    )
    lines = block.splitlines()
    group_at = {}
    for g in sc.GROUPS:
        hits = [
            n for n, ln in enumerate(lines) if g in ln and not any(g in e.label for e in sc.CATALOG)
        ]
        assert hits, f"group {g!r} is named in the block"
        group_at[g] = hits[0]
    assert sorted(group_at.values()) == [group_at[g] for g in sc.GROUPS], "groups in panel order"
    for (i, c), e in zip(entry_rows, sc.CATALOG, strict=True):
        assert e.label in c[0], f"row order: {c[0]!r} is not {e.label!r}"
        gi = sc.GROUPS.index(e.group)
        nxt = group_at[sc.GROUPS[gi + 1]] if gi + 1 < len(sc.GROUPS) else len(lines)
        assert group_at[e.group] < i < nxt, f"{e.key}: its row lies under {e.group!r}"
        assert c[1], f"{e.key}: Default cell is empty"
        assert e.useful_when in c[2], f"{e.key}: Useful when is the catalog's: {c[2]!r}"
        assert e.not_useful_when in c[3], f"{e.key}: Not useful when is the catalog's: {c[3]!r}"
        ch = c[4]
        assert ch, f"{e.key}: Changes cell is empty"
        if e.relaunch is not None:
            assert "restart" in ch.lower(), (
                f"{e.key}: a relaunch row says it needs a restart: {ch!r}"
            )
            if e.flag:
                assert e.flag in ch, f"{e.key}: Changes names the flag {e.flag}: {ch!r}"
        elif e.v1_changeable:
            assert "panel" in ch.lower() and "restart" not in ch.lower(), (
                f"{e.key}: a panel row says live from the panel: {ch!r}"
            )
            if e.flag:
                assert e.flag in ch, f"{e.key}: Changes names the flag {e.flag}: {ch!r}"


_HEADING = re.compile(r"^(#{1,6})\s")


def _check_options_section(text: str) -> None:
    """R-1 (durable form of criterion 2's placement): exactly one "Options at a glance" heading, under
    ``### Settings`` and before the next ``## ``; each marker once, inside that section (before the next heading of
    the same or a higher level), begin before end. Unrelated README edits elsewhere leave it true."""
    lines = text.splitlines()
    settings = [n for n, ln in enumerate(lines) if ln.strip() == "### Settings"]
    assert len(settings) == 1, "README has one '### Settings' heading"
    heads = [n for n, ln in enumerate(lines) if re.match(r"^#{2,6}\s.*Options at a glance", ln)]
    assert len(heads) == 1, f"README has one 'Options at a glance' heading (has {len(heads)})"
    h = heads[0]
    next_h2 = next(
        (n for n, ln in enumerate(lines) if n > settings[0] and re.match(r"^##\s", ln)), len(lines)
    )
    assert settings[0] < h < next_h2, (
        "'Options at a glance' lies under '### Settings', before the next '## ' section"
    )
    m = _HEADING.match(lines[h])
    assert m is not None
    level = len(m.group(1))
    sec_end = next(
        (
            n
            for n in range(h + 1, len(lines))
            if (hm := _HEADING.match(lines[n])) and len(hm.group(1)) <= level
        ),
        len(lines),
    )
    sec_end = min(sec_end, next_h2)
    begins = [n for n, ln in enumerate(lines) if BEGIN in ln]
    ends = [n for n, ln in enumerate(lines) if END in ln]
    assert len(begins) == 1, f"{BEGIN!r} appears once (appears on {len(begins)} lines)"
    assert len(ends) == 1, f"{END!r} appears once (appears on {len(ends)} lines)"
    assert text.count(BEGIN) == 1 and text.count(END) == 1, "each marker appears once"
    assert h < begins[0] < ends[0] < sec_end, (
        "the markers lie inside the 'Options at a glance' section, begin before end"
    )


def test_readme_options_section_lies_under_settings() -> None:
    """Criterion 2 / ruling R-1: the README's options section and its markers are where criterion 2 puts them. The
    block's content is pinned by ``test_readme_options_block_matches_the_catalog``."""
    _check_options_section(README.read_text(encoding="utf-8"))


_GOOD = (
    "# T\n\n## Run\n\n### Settings\n\nProse.\n\n#### Options at a glance\n\n"
    f"{BEGIN}\n| a |\n{END}\n\n## Real websites\n\nMore.\n"
)


def test_options_section_check_accepts_a_good_readme_with_unrelated_edits() -> None:
    """R-1 power, too-strict direction: a README with the section in place passes, and stays passing after edits
    elsewhere (a new "Known limits" note, a reworded paragraph)."""
    _check_options_section(_GOOD)
    edited = _GOOD.replace("Prose.", "Prose, reworded.").replace(
        "## Real websites", "## Known limits\n\nA note.\n\n## Real websites"
    )
    _check_options_section(edited)


_PLACED = "lies under '### Settings'"
_INSIDE = "inside the 'Options at a glance' section"


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        pytest.param(
            _GOOD.replace("### Settings\n", "### Other\n"), "one '### Settings'", id="no-settings"
        ),
        pytest.param(
            _GOOD.replace(f"#### Options at a glance\n\n{BEGIN}\n| a |\n{END}\n", "").replace(
                "More.\n", f"More.\n\n### Options at a glance\n\n{BEGIN}\n| a |\n{END}\n"
            ),
            _PLACED,
            id="section-outside-settings",
        ),
        pytest.param(
            _GOOD.replace("#### Options at a glance", "## Options at a glance"),
            _PLACED,
            id="heading-is-a-new-h2",
        ),
        pytest.param(
            _GOOD.replace(f"{END}\n", f"{END}\n{BEGIN}\n{END}\n"),
            "appears once",
            id="duplicate-markers",
        ),
        pytest.param(
            _GOOD.replace(f"{END}\n", f"{END}\n{END}\n"), "appears once", id="duplicate-end"
        ),
        pytest.param(
            _GOOD.replace(f"{BEGIN}\n| a |\n{END}", f"{END}\n| a |\n{BEGIN}"),
            _INSIDE,
            id="end-before-begin",
        ),
        pytest.param(
            _GOOD.replace(f"{END}\n", "").replace("More.\n", f"More.\n{END}\n"),
            _INSIDE,
            id="end-outside-section",
        ),
        pytest.param(
            _GOOD.replace("Prose.\n\n", "Prose.\n\n#### Options at a glance\n\nOld.\n\n"),
            "one 'Options at a glance' heading",
            id="duplicate-heading",
        ),
    ],
)
def test_options_section_check_refuses_a_misplaced_section(bad: str, reason: str) -> None:
    """R-1 power, too-wide direction: the check refuses a section outside Settings, duplicate markers or headings,
    markers out of order or outside the section, each for its own reason."""
    with pytest.raises(AssertionError, match=re.escape(reason)):
        _check_options_section(bad)


def test_main_rewrites_only_the_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 2: ``main()`` rewrites README.md in place: a stale block becomes ``render(CATALOG)`` and every
    byte outside the markers is kept. Run on a copy (script under ``<tmp>/scripts/``, README at ``<tmp>``, cwd
    ``<tmp>``) so the real README is never written."""
    mod = _load_script()
    want = mod.render(sc.CATALOG)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    head = "# Title\n\n### Settings\n\n#### Options at a glance\n\n"
    tail = "\n\nTrailing text that main() must keep, | pipes | and all.\n"
    (tmp_path / "README.md").write_text(f"{head}{BEGIN}\nstale row\n{END}{tail}", encoding="utf-8")
    name = "_t0221_gen_copy"
    spec = importlib.util.spec_from_file_location(name, tmp_path / "scripts" / SCRIPT.name)
    assert spec is not None and spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    sys.modules[name] = copy
    try:
        spec.loader.exec_module(copy)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", [str(tmp_path / "scripts" / SCRIPT.name)])
        rc = copy.main()
    finally:
        sys.modules.pop(name, None)
    assert rc in (None, 0), f"main() returns 0 or None: {rc!r}"
    out = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert out == f"{head}{BEGIN}{want}{END}{tail}", (
        "main() replaces the block and keeps every other byte"
    )
