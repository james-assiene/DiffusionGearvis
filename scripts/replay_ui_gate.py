#!/usr/bin/env python3
"""Replay a session log's ``ui`` commits through the UI gate on CPU (T-0127 criterion 9).

Usage: ``uv run python scripts/replay_ui_gate.py <session.jsonl>``

The pre-desk check: for each turn, in log order, every ``llm.commit`` of the ``ui`` segment is
gated by a fresh ``UiGate`` with the logged ``turn_id`` and ``canvas_idx`` and ``final=False``,
which is what ``Session._gate_surface`` does at the desk. Per turn it prints the last call's
``complete``, ``lost`` and unreachable ids and the turn's repairs (structural, by component rule,
and the rest: json_repair log entries and salvage edits), then ``k/n complete``.

Exit status: 0 when every turn's last call is complete, 1 otherwise, 2 on a usage error.
Read-only (the log is only read), no network, no GPU.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.conductor import ui_gate as ui_gate_module
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.ui.component_repair import reserved_label_id


def ui_commits(path: Path) -> dict[str, list[dict[str, Any]]]:
    """``turn_id -> [ui commit data, ...]`` in log order (turns in order of first commit)."""
    turns: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data = record.get("data") or {}
        if record.get("kind") != "llm.commit" or data.get("segment") != "ui":
            continue
        turns.setdefault(str(record.get("turn_id")), []).append(data)
    return turns


def rules_of(before: Sequence[Any], after: Sequence[Any]) -> Counter[str]:
    """The component rules one ``repair_components`` call applied, counted as the stage counts."""
    out: Counter[str] = Counter()
    if after is before:
        return out  # the stage changed nothing
    # only the rewritten components: an object passed through is the identical input object
    untouched = {id(c) for c in before}
    by_id = {c.get("id"): c for c in after if isinstance(c, Mapping) and id(c) not in untouched}
    for c in before:
        if not isinstance(c, Mapping) or id(c) in {id(x) for x in after}:
            continue
        new = by_id.get(c.get("id"))
        if new is None:
            continue
        kind = c.get("component")
        if kind == "Card":
            out["CC"] += 1
        elif kind == "Button":
            rb = new.get("child") == reserved_label_id(str(c.get("id"))) and "action" not in c
            out["RB" if rb else "BC"] += 1
        elif kind in ("Row", "Column", "List"):
            out["LI"] += 1
        elif kind == "TextField":
            out["TF"] += 1
        elif kind == "DateTimeInput":
            out["DT"] += 1
        elif kind == "ChoicePicker":
            out["CPO"] += int(new.get("options") != c.get("options"))
            out["CPV"] += int(new.get("value") != c.get("value"))
    return out


class RuleCounter:
    """Counts rules through the gate's own call of the component stage.

    The stage is looked up at call time, so a patched stage (the eval's "CC disabled") is the one
    counted. A repair the gate reverts after grounding is re-counted by the gate on the original
    alone and subtracted; this mirrors that, so the counts are the repairs that were kept.
    """

    def __init__(self) -> None:
        self.rules: Counter[str] = Counter()
        self._last: Sequence[Any] = ()
        self._inner = ui_gate_module.repair_components

    def __call__(self, components: list[Any], validator: Any) -> tuple[list[Any], int]:
        out, edits = self._inner(components, validator)
        counted = rules_of(components, out)
        revert = len(components) == 1 and any(components[0] is c for c in self._last)
        if revert:
            self.rules.subtract(counted)
        else:
            self.rules.update(counted)
            self._last = components
        return out, edits


def replay(path: Path) -> int:
    turns = ui_commits(path)
    counter = RuleCounter()
    ui_gate_module.repair_components = counter  # type: ignore[assignment]
    complete = 0
    try:
        for turn_id, commits in turns.items():
            counter.rules = Counter()
            gate = UiGate()
            results: list[GateResult] = [
                gate.gate(
                    c.get("text", ""),
                    turn_id=turn_id,
                    canvas_idx=int(c.get("canvas_idx", 0)),
                    final=False,
                )
                for c in commits
            ]
            last = results[-1]
            complete += int(last.complete)
            structural = sum(r.structural_repairs for r in results)
            component = sum(r.component_repairs for r in results)
            other = sum(r.repaired for r in results) - structural - component
            rules = ", ".join(f"{k}={v}" for k, v in sorted(counter.rules.items()) if v)
            print(
                f"{turn_id}: complete={last.complete} lost={list(last.lost)} "
                f"unreachable={list(last.unreachable)} root={last.root} "
                f"canvases={len(results)} surface={'yes' if last.surface else 'no'} "
                f"repairs: structural={structural} component={component}"
                f"{f' ({rules})' if rules else ''} other={other}"
            )
            for problem in last.report.problems:
                print(f"    {problem}")
    finally:
        ui_gate_module.repair_components = counter._inner  # type: ignore[assignment]
    print(f"{complete}/{len(turns)} complete")
    return 0 if complete == len(turns) else 1


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print("usage: uv run python scripts/replay_ui_gate.py <session.jsonl>", file=sys.stderr)
        return 2
    return replay(Path(argv[0]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
