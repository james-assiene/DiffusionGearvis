"""Markdown report built from artifacts only (T-0070 criteria 7, 8(f), 9).

``build(paths)`` reads nothing but the given ``.jsonl`` artifacts: every figure is recomputed from their
rows. A success rate is never printed without the parse rate on the same row, every rate carries its
Wilson 95 % interval, and ``n`` is the artifact's full record count (failures included). Paired arms of
the same task and cut get the pre-registered exact McNemar comparisons, with the "both parsed"
restriction beside them. The unconstrained BFCL arm is printed under both normalisations (ruling E2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd import stats as _stats
from local_astra.eval.cd.runner import read_artifact

__all__ = ["build"]


def _rate(k: int, n: int) -> str:
    return _stats.fmt_rate(k, n) if n else "n/a"


def _count(rows: Sequence[Mapping[str, Any]], pred: Any) -> int:
    return sum(1 for r in rows if pred(r))


def _ok_key(task: str) -> str:
    return "exact_call" if task == "bfcl" else "solved"


def _label(regime: Mapping[str, Any]) -> str:
    bits = [str(regime.get("arm"))]
    if regime.get("whitespace"):
        bits.append(f"ws={regime['whitespace']}")
    if regime.get("label"):
        bits.append(str(regime["label"]))
    return " ".join(bits)


def _table_rows(
    regime: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    scores_key: str,
    note: str,
    stopped: str | None = None,
) -> str:
    """One table row. A stopped artifact's row says so in every cell that carries a rate (E11)."""
    task = str(regime.get("task"))
    n = len(rows)
    ok = _count(rows, lambda r: bool((r.get(scores_key) or {}).get(_ok_key(task))))
    parsed = _count(rows, lambda r: bool(r.get("parse")))
    errors = _count(rows, lambda r: bool(r.get("error")))
    cs_known = [r for r in rows if r.get("cs") is not None]
    cs = _rate(_count(cs_known, lambda r: r.get("cs") is True), n) if cs_known else "n/a"
    cells = [f"{_label(regime)}{note}", str(n), _rate(ok, n), _rate(parsed, n)]
    if task == "bfcl":
        sv = _count(rows, lambda r: bool((r.get(scores_key) or {}).get("schema_valid")))
        ac = sum(int((r.get(scores_key) or {}).get("arg_correct", 0)) for r in rows)
        at = sum(int((r.get(scores_key) or {}).get("arg_total", 0)) for r in rows)
        cells += [_rate(sv, n), f"{_rate(ac, at)} ({ac}/{at})"]
    cells += [cs, _rate(errors, n)]
    if stopped:
        letter = stopped.split(":", 1)[0]
        cells[0] += f" **STOPPED ({letter})**"
        cells[1] = f"{n} of {regime.get('n')} (stopped)"
        cells[2:] = [f"{c} (stopped)" for c in cells[2:]]
    return "| " + " | ".join(cells) + " |"


def _header(task: str) -> list[str]:
    ok = "exact call" if task == "bfcl" else "solved"
    cols = ["arm", "n", ok, "parsed"]
    if task == "bfcl":
        cols += ["schema valid", "arg acc (arg_correct/arg_total)"]
    cols += ["CS (host replay)", "errors"]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def _outcomes(task: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    return {
        str(r["id"]): {
            "ok": bool((r.get("scores") or {}).get(_ok_key(task))),
            "parsed": bool(r.get("parse")),
        }
        for r in rows
    }


def build(paths: Sequence[Path | str]) -> str:
    loaded = []
    for p in paths:
        regime, rows, other = read_artifact(p)
        loaded.append((Path(p), regime, rows, other))
    out = ["# CD evaluation report", ""]
    groups: dict[
        tuple[str, str, Any],
        list[tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]],
    ] = {}
    for item in loaded:
        regime = item[1]
        key = (str(regime.get("task")), str(regime.get("cut")), regime.get("whitespace") or "json")
        groups.setdefault(key, []).append(item)
    for (task, cut, ws), items in groups.items():
        out += [f"## {task} / {cut}" + (f" / whitespace {ws}" if task == "bfcl" else ""), ""]
        out += _header(task)
        notes: list[str] = []
        for path, regime, rows, other in items:
            trailer = next((x for x in reversed(other) if x.get("kind") == "trailer"), {})
            stopped = regime.get("stopped") or trailer.get("stopped")
            out.append(_table_rows(regime, rows, "scores", "", stopped))
            if (
                task == "bfcl"
                and regime.get("arm") == "unconstrained"
                and all("scores_published" in r for r in rows)
            ):
                out.append(
                    _table_rows(
                        regime,
                        rows,
                        "scores_published",
                        " (published normalisation: ci_enums off, expand_wildcard off)",
                        stopped,
                    )
                )
            n_regime = regime.get("n")
            line = (
                f"- `{path.name}`: commit {regime.get('commit')}, n = {n_regime} "
                f"(rows written {len(rows)}), max_tokens {regime.get('max_tokens')}, "
                f"thinking {'on' if regime.get('enable_thinking') else 'off'}, "
                f"calibration sha {regime.get('calibration_sha')}, knobs {regime.get('knobs')}"
            )
            if regime.get("excluded_ids"):
                line += f", excluded above max_states: {len(regime['excluded_ids'])}"
            notes.append(line)
            if "header_prefix_rate" in trailer and regime.get("arm") == "unconstrained":
                rate = trailer["header_prefix_rate"]
                shown = "n/a (no token ids)" if rate is None else f"{rate:.3f}"
                notes.append(
                    f"  - header-prefix rate (first {trailer.get('header_prefix_window', 20)} "
                    f"unconstrained records open with `<|channel>`, from token ids): {shown} "
                    "(reported, not a stop)"
                )
            if stopped:
                notes.append(f"  - **stopped: {stopped}** (partial artifact)")
        out += ["", *notes, ""]
        by_arm = {str(r[1].get("arm")): r[2] for r in items}
        pairs = [(a, b) for a, b in _stats.PREREGISTERED_PAIRS if a in by_arm and b in by_arm]
        if pairs:
            out += [
                f"Paired exact McNemar ({_ok_key(task)}), Bonferroni over {_stats.N_COMPARISONS}:",
                "",
                "| A vs B | n | b (A only) | c (B only) | p | p (Bonf.) | both parsed: n, b, c, p (Bonf.) |",
                "|---|---|---|---|---|---|---|",
            ]
            for a, b in pairs:
                try:
                    r = _stats.paired(_outcomes(task, by_arm[a]), _outcomes(task, by_arm[b]))
                except ValueError as exc:
                    out.append(f"| {a} vs {b} | not paired: {exc} | | | | | |")
                    continue
                bp = r.both_parsed
                bp_cell = (
                    f"{bp.n}, {bp.b}, {bp.c}, {bp.p:.4g} ({bp.p_bonferroni:.4g})"
                    if bp is not None
                    else "n/a"
                )
                out.append(
                    f"| {a} vs {b} | {r.n} | {r.b} | {r.c} | {r.p:.4g} | {r.p_bonferroni:.4g} | "
                    f"{bp_cell} |"
                )
            out.append("")
    return "\n".join(out) + "\n"
