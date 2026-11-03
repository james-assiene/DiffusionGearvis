"""T-0179 group A: the repeat guard on by default in live (docs/tasks/T-0179.md A-1 to A-4, B-10's A line).

T-0137's cli rigs are imported (``test_t0137_cli``: ``dry_run``, ``build``, ``build_out``, ``check_placement``,
``session_help``), and A-4's byte identity runs T-0137's differential capture (``test_t0137_differential._capture``:
the cli runs under ``--dry-run`` and at a real start, rig turns with request bodies, turn lines, session-log records,
executor action strings and HUD messages, and the other modes) in a subprocess over ``git archive BASE_A`` and
in-process here. The in-process capture passes ``--no-repeat-guard`` on every live case that does not set the guard
itself (the A-5 rig pin in ``test_t0152_overview_frame_default._cli_run``) and reads its ``off (flag)`` line back as
BASE's ``off (default)``; ``_other_modes`` (off, dry_run) passes no flag at all.

**Pending and active rows.** The rows that need the flip (the line, the keyword, the help, the docs/02 line) are
``pending(task="T-0179")`` and red at BASE on a named assertion. The off rows, the outside-live rows, the byte
identity and the frozen paths hold at BASE and are active.

Not tested (in the report): A-1's "with its comment updated" (a comment), A-6 (``make browser-live``, not in the fast
suite; the verifier runs it).

Regime: CPU, the cli rigs over fakes, nothing served, n = 1, deterministic.
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

from local_astra import cli
from local_astra.config import Profile
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0137_cli as t37
from tests.unit import test_t0137_differential as t37d
from tests.unit._t0177_pins import observed as t0177_observed  # T-0177 A-P1 and B-P1
from tests.unit._t0181_pins import capture_without_mic_unavailable  # T-0181 A-R13

#: main at aeac453f, T-0179's branch point (BASE_A = BASE_B). A literal.
BASE_A = "aeac453f3f0d7beccd5ac98641b0c6cbbf446af5"
#: The group A implementation tip (starts at BASE_A). Bumped by whoever commits group A, closed at the merge.
TIP_A = "30d314526b37cbbd14029a10918cb76b13ccad4e"  # 30d31452, "T-0179 A: repeat guard on by default in live"
PREFIX = "astra session: repeat guard: "
ON_DEFAULT = PREFIX + "on (default)"
FROZEN_A4 = (
    "local_astra/conductor/progress.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall",
    "local_astra/computer",
)
HEADER_A = "**Spec delta (T-0179 A)"


# --- A-2: the line, the sources, the keyword -----------------------------------------------------------------------


def test_default_line_and_position(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """A-2: in live with defaults the line is ``on (default)``, once, at T-0137 B-9's position (right after the action
    outcome line), under ``--dry-run``; at a real start every printed block says the same before the regime line."""
    out = t37.dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    t37.check_placement(out, ON_DEFAULT, regime=False)
    monkeypatch.undo()
    _, started, code = t37.build_out(tmp_path / "s", capsys, monkeypatch, *t37.LIVE)
    assert code == 0, started
    got = [x for x in started.splitlines() if x.startswith(PREFIX)]
    assert got and all(x == ON_DEFAULT for x in got), f"T-0179 A-2: {got}"


@pytest.mark.parametrize(
    ("flags", "body", "want"),
    [
        (["--no-repeat-guard"], "", "off (flag)"),
        ([], "repeat_guard = false\n", "off (profile)"),
    ],
    ids=["flag", "profile"],
)
def test_off_sources(
    tmp_path: Any, capsys: Any, monkeypatch: Any, flags: list[str], body: str, want: str
) -> None:
    """A-2 / A-3 / H-T179-5: ``--no-repeat-guard`` prints ``off (flag)`` and a profile ``false`` prints ``off
    (profile)``; both build a ``Session`` with no ``repeat_guard`` keyword. Active (holds at BASE)."""
    out = t37.dry_run(
        tmp_path / "d", capsys, monkeypatch, "live", *flags, body=t37.LIVE_PROFILE + body
    )
    assert t37.lines(out) == [PREFIX + want], out
    monkeypatch.undo()
    calls, _ = t37.build(tmp_path / "b", capsys, monkeypatch, *flags, body=t37.LIVE_PROFILE + body)
    assert "repeat_guard" not in calls.session_kwargs[-1], sorted(calls.session_kwargs[-1])


def test_build_graph_passes_true_by_default(tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
    """A-2 (adversarial: the constant flipped in ``session_cmd`` only, so the line says on and the graph passes
    nothing): a default live build passes ``repeat_guard=True`` *and* prints ``on (default)``; the resolver gives
    ``(True, "default")`` for a live profile without the key."""
    calls, _built = t37.build(tmp_path / "b", capsys, monkeypatch)
    assert calls.session_kwargs[-1].get("repeat_guard") is True, (
        f"T-0179 A-2: _build_graph passes repeat_guard=True by default in live; {sorted(calls.session_kwargs[-1])}"
    )
    got = cli.repeat_guard_of(Profile(name="lean", tools="live"), argparse.Namespace(), "live")  # type: ignore[arg-type]
    assert (got.repeat_guard, got.repeat_guard_from) == (True, "default"), got


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_off_and_dry_run_unchanged(tmp_path: Any, capsys: Any, monkeypatch: Any, mode: str) -> None:
    """A-2 / reviewer mutation 8 (the default leaking into dry_run): outside live there is no line and no keyword,
    and the resolver gives ``(False, "unset")``; the flag there is still refused (exit 2). Active."""
    out = t37.dry_run(tmp_path / "d", capsys, monkeypatch, mode)
    assert "repeat guard:" not in out, out
    got = cli.repeat_guard_of(Profile(name="lean"), argparse.Namespace(), mode)
    assert (got.repeat_guard, got.repeat_guard_from) == (False, "unset"), got
    monkeypatch.undo()
    from tests.unit.test_cli_t0131 import MODE_FLAGS
    from tests.unit.test_t0150_cli import refused_run

    code, text, _calls = refused_run(
        tmp_path / "r", capsys, monkeypatch, *MODE_FLAGS[mode], "--repeat-guard"
    )
    assert code == 2 and "repeat guard" in text, (code, text)


def test_flag_stays_boolean_optional() -> None:
    """A-3 / H-T179-5: ``--repeat-guard`` and ``--no-repeat-guard`` both exist (``BooleanOptionalAction``). Active."""
    flags = set(re.findall(r"--[a-z][a-z-]+", t37.session_help()))
    assert {"--repeat-guard", "--no-repeat-guard"} <= flags, sorted(
        f for f in flags if "repeat" in f
    )


def test_help_says_default_on() -> None:
    """A-3: the ``--repeat-guard`` help says the default in live is on (keywords: ``default`` then ``on``, no longer
    ``else off``)."""
    text = " ".join(t37.session_help().split())
    marker = "--repeat-guard, --no-repeat-guard"
    at = text.find(marker)
    assert at >= 0, "premise: the flag's help entry is in the help"
    # the next option's entry (a ``--tools`` mention inside this entry's text is not one)
    nxt = re.search(r" --(?!tools\b)[a-z][a-z-]+", text[at + len(marker) :])
    block = text[at : at + len(marker) + (nxt.start() if nxt else 700)]
    assert "repeat_of_step" in block, f"premise: the entry is the guard's: {block[:200]!r}"
    assert "else off" not in block, (
        f"T-0179 A-3: the help still says the default is off: {block[:400]!r}"
    )
    assert re.search(r"default[^.;]*\bon\b", block), (
        f"T-0179 A-3: the help says default on: {block[:400]!r}"
    )


# --- A-4: off is BASE ----------------------------------------------------------------------------------------------

_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0137_differential as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_A)
    tmp = tmp_path_factory.mktemp("t0179a_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_A)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_A[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


def _compare(base: Any, now: Any, path: str = "") -> None:
    if isinstance(base, dict) and isinstance(now, dict) and base.keys() == now.keys():
        for key in base:
            _compare(base[key], now[key], f"{path}/{key}")
        return
    if isinstance(base, list) and isinstance(now, list) and len(base) == len(now):
        for i, (b, n) in enumerate(zip(base, now, strict=True)):
            _compare(b, n, f"{path}[{i}]")
        return
    assert base == now, (
        f"T-0179 A-4: differs from BASE {BASE_A[:8]} at {path}:\nbase {json.dumps(base)[:600]}\n"
        f"now  {json.dumps(now)[:600]}"
    )


def test_no_repeat_guard_is_base(base: dict[str, Any], tmp_path: Path) -> None:
    """A-4: with ``--no-repeat-guard`` (live) and with no flag (off, dry_run), T-0137's capture -- cli stdout and
    stderr under ``--dry-run`` and at a real start, ``Session`` and executor keywords, rig turns (request bodies, turn
    lines, session-log records, executor action strings, HUD messages) and the other modes -- equals BASE_A's byte for
    byte (the pin's one line read back). Active: holds at BASE; red if ``--no-repeat-guard`` stops being BASE."""
    # T-0177 A-P1: the observed side only, A10's extras.vision_source == "browser" (never "shared") set aside
    now = t0177_observed(t37d._capture(tmp_path))
    now = capture_without_mic_unavailable(
        now
    )  # T-0181 A-R13 pin move: only Session's mic_unavailable keyword
    assert now["cli"], "premise: the capture ran the cli cases"
    _compare(base, now)


def test_frozen_paths_over_closed_range() -> None:
    """A-4: ``progress.py``, ``session.py``, ``turn_machine.py``, ``prompt_builder.py``, ``toolcall/`` (the
    ``tools_v1*.json`` included) and ``computer/`` are byte-identical over ``BASE_A..TIP_A`` (never the working tree),
    and ``LIVE_TOOL_GUIDANCE`` keeps 768a973b. Power: ``82da380..02ccff5`` changed the lock."""
    import hashlib

    gr.require_commits(BASE_A, TIP_A)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:8]} is not an ancestor of {TIP_A[:8]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A, "--", *FROZEN_A4).stdout.split()
    for p in (
        *FROZEN_A4,
        "local_astra/toolcall/tools_v1.json",
        "local_astra/toolcall/tools_v1_overview.json",
    ):
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE_A, TIP_A, *FROZEN_A4)
    assert changed == [], f"T-0179 A-4: frozen paths differ in {BASE_A[:8]}..{TIP_A[:8]}: {changed}"
    guidance = hashlib.sha256(cli.LIVE_TOOL_GUIDANCE.encode("utf-8")).hexdigest()
    assert guidance.startswith("768a973b"), guidance
    gr.require_commits("82da380", "02ccff5")
    assert gr.changed("82da380", "02ccff5", "services") == ["services/versions.lock"], "power"


# --- B-10, group A's line --------------------------------------------------------------------------------------------


def test_doc02_delta_t0179_a() -> None:
    """B-10 (group A): docs/02 gains one paragraph starting ``**Spec delta (T-0179 A)`` near T-0137's (after it, in
    the same section), recording the default on, the C result (no change: 61/100 vs 62/100, or 38423a5) and James's
    ruling. Read at ``TIP_A``; T-0137's paragraph stays verbatim."""
    gr.require_commits(BASE_A, TIP_A)
    proc = gr._git("show", f"{TIP_A}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    paras = t7d.paragraphs(proc.stdout)
    base_paras = t7d.paragraphs(gr._git("show", f"{BASE_A}:docs/02-architecture.md").stdout)
    (t0137,) = [p for p in base_paras if p.lstrip().startswith("**Spec delta (T-0137).**")]
    assert t0137 in paras, "T-0179 B-10: T-0137's paragraph stays verbatim"
    found = [i for i, p in enumerate(paras) if p.lstrip().startswith(HEADER_A)]
    assert len(found) == 1, (
        f"T-0179 B-10: one paragraph starting {HEADER_A!r} at {TIP_A[:8]}; {len(found)}"
    )
    para = paras[found[0]]
    assert found[0] > paras.index(t0137), "T-0179 B-10: the A line sits after T-0137's paragraph"
    low = para.lower()
    need = {
        "default on": "default" in low and re.search(r"\bon\b", low) is not None,
        "C result": "38423a5" in para or ("61" in para and "62" in para),
        "James": "James" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0179 B-10: the A line does not record {missing}"
