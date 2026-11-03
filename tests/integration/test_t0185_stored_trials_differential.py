"""T-0185 criteria 10(b) and 12 (first bullet): the full stored-trial differential, and the bound on a huge literal.

**10(b), "fulldiff".** Every ``ui`` commit of every stored ``trials.jsonl`` (``logs/*/`` and ``.claude/worktrees/*/logs/*/``
of the main checkout, found through ``git rev-parse --git-common-dir`` so that a worktree run sees the same corpus, and
of this checkout when it differs) goes, in commit order, through one fresh live ``UiGate()`` and one fresh frozen BASE
gate (``tests/unit/_frozen_ui_gate_t0185.py``, 7646b8b6) per trial: ``final=False``, then the last commit again with
``final=True`` in the same gate pair (so calls = commits + trials, which reproduces T-0184 C's count); turn id = the
trial's ``request_id`` (else ``<file>:<line>``), the commit's ``canvas_idx`` (T-0184 C1's harness shape). Every
``GateResult`` must be field-equal. It prints ``files=…, trials=…, calls=…, diffs=…`` (at T-0184 C: 21 files, 827
trials, 2988 calls) and skips, saying so, when there are no stored trials. Green at BASE by construction; unmarked.

**12, first bullet.** The gate on a canvas with a 10^5-digit literal takes <= 1 s, median of 5, serial, on each of
three canvases (the strict path: basic ``Slider.value``; json_repair's path: the same with the key bare; and the canvas
cut right after the literal). The regime is printed beside the medians. Pending: BASE raises on the first and third.

Not in the fast suite (``tests/integration``); CPU only, no service, no socket. Run it from a real worktree (the corpus
lookup uses git). Regime: Python 3.11.16, json_repair 0.63.4, default int-string limit 4300.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from tests import _git_range as gr
from tests.unit import _frozen_ui_gate_t0185 as frozen

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
GATE_FIELDS = tuple(f.name for f in dataclasses.fields(frozen.GateResult))
HUGE = 10**5


def _snapshot(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in GATE_FIELDS:
        assert hasattr(result, name), f"GateResult lost BASE's field {name!r}"
        value = getattr(result, name)
        if name == "surface":
            value = None if value is None else value.model_dump(mode="python")
        elif name == "report":
            value = dataclasses.asdict(value)
        out[name] = value
    return out


def _short(value: Any, limit: int = 600) -> str:
    try:
        text = repr(value)
    except ValueError:
        return "<unprintable: holds an int past the int-string limit>"
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} characters)"


def _first_difference(a: dict[str, Any], b: dict[str, Any]) -> str:
    for name in GATE_FIELDS:
        if a.get(name) != b.get(name):
            return f"field {name!r}:\n  tip:  {_short(a.get(name))}\n  base: {_short(b.get(name))}"
    return "(no field differs)"


def _roots() -> list[Path]:
    """This checkout, and the main checkout when this is a linked worktree (the stored runs live there)."""
    roots = [REPO]
    common = gr._git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if common.returncode == 0 and common.stdout.strip():
        main = Path(common.stdout.strip()).parent
        if main.resolve() != REPO.resolve():
            roots.append(main)
    return roots


def _trial_files() -> list[Path]:
    found: dict[Path, Path] = {}
    for root in _roots():
        for pattern in ("logs/*/trials.jsonl", ".claude/worktrees/*/logs/*/trials.jsonl"):
            for path in root.glob(pattern):
                found.setdefault(path.resolve(), path)
    return sorted(found.values())


def _ui_trials(path: Path) -> list[tuple[str, list[tuple[str, int]]]]:
    """``(turn id, [(text, canvas_idx), …])`` per trial of ``path`` with at least one ``ui`` commit, in file order."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        commits = row.get("commits") if isinstance(row, dict) else None
        if not isinstance(commits, list):
            continue
        ui = [
            (c["text"], int(c.get("canvas_idx") or 0))
            for c in commits
            if isinstance(c, dict) and c.get("segment") == "ui" and isinstance(c.get("text"), str)
        ]
        if ui:
            out.append((str(row.get("request_id") or f"{path.name}:{n}"), ui))
    return out


def test_every_stored_trial_gives_base_results() -> None:
    """Criterion 10(b): 0 diffs between the live gate and the frozen BASE gate over every stored ``ui`` commit."""
    files = _trial_files()
    if not files:
        pytest.skip(
            "no stored trials.jsonl under logs/*/ or .claude/worktrees/*/logs/*/ of this checkout or main"
        )
    trials = calls = 0
    for path in files:
        for turn, commits in _ui_trials(path):
            trials += 1
            tip, base = UiGate(), frozen.UiGate()
            last_text, last_idx = commits[-1]
            for k, (text, idx, final) in enumerate(
                [*((t, i, False) for t, i in commits), (last_text, last_idx, True)]
            ):
                calls += 1
                a = _snapshot(tip.gate(text, turn_id=turn, canvas_idx=idx, final=final))
                b = _snapshot(base.gate(text, turn_id=turn, canvas_idx=idx, final=final))
                assert a == b, f"{path} trial {turn} call {k}: {_first_difference(a, b)}"
    print(f"\nT-0185 fulldiff: files={len(files)}, trials={trials}, calls={calls}, diffs=0")
    if trials == 0:
        pytest.skip(f"{len(files)} trials.jsonl found, none with a ui commit")


def _huge_canvases() -> dict[str, str]:
    lit = "7" * HUGE
    head = (
        '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":'
        '"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}\n'
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        '{"id":"root","component":"Column","children":["t","x"]},'
        '{"id":"t","component":"Text","text":"Kept"},'
        '{"id":"x","component":"Slider","label":"Volume",'
    )
    tail = ',"min":0,"max":10}]}}'
    return {
        "strict": f'{head}"value":{lit}{tail}',
        "bare-key": f"{head}value:{lit}{tail}",
        "cut-after": f'{head}"value":{lit}',
    }


def test_huge_literal_canvas_is_bounded() -> None:
    """Criterion 12, first bullet: <= 1 s per gate call on a 10^5-digit literal, median of 5 (fresh gate each)."""
    regime = (
        f"python {platform.python_version()}, int-string limit {sys.get_int_max_str_digits()}, "
        f"load {os.getloadavg()[0]:.2f}, n=5, serial, fresh UiGate per call"
    )
    medians: dict[str, float] = {}
    for name, raw in _huge_canvases().items():
        times = []
        for _ in range(5):
            gate = UiGate()
            t0 = time.perf_counter()
            try:
                gate.gate(raw, turn_id="h", canvas_idx=0, final=name != "cut-after")
            except Exception as err:
                raise AssertionError(
                    f"{name}: the gate raised {type(err).__name__}: {str(err)[:120]}"
                ) from None
            times.append(time.perf_counter() - t0)
        medians[name] = statistics.median(times)
    print(
        "\nT-0185 criterion 12: "
        + ", ".join(f"{k} {v * 1000:.1f} ms" for k, v in medians.items())
        + f" ({regime})"
    )
    slow = {k: v for k, v in medians.items() if v > 1.0}
    assert not slow, f"criterion 12: median above 1 s: {slow} ({regime})"
