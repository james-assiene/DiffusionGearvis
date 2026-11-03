"""T-0172 A4: the announced-call census over the session logs on this box (eval, CPU; docs/tasks/T-0172.md).

The corpus and the labels are ``tests/fixtures/t0172/labels.jsonl`` (built by ``build_labels.py``, stdlib only,
never importing the rule; its docstring defines the label and the ambiguous cases). The logs are read here through
the builder's own reader (``session_files``/``requests_of``), so the census sees exactly the corpus that was
labelled: every ``logs/**/sessions/*.jsonl`` whose name's timestamp is at most the fixture's cut-off.

Rows:

- ``test_corpus_matches_the_fixture``: the logs still hold the labelled corpus (per-directory file counts, request
  and no-call counts, and every labelled row's answer at its ``(source, seq)``). Green without the module.
- ``test_census_tail`` (A4 thresholds, ``tail``): precision 1.0 (0 hits on labelled negatives *and* on every
  unlabelled tools-on no-call answer, a negative by definition), recall >= 0.95 over the 108 positives, both smoke
  records hit, 0 hits on the 45 tools-off no-call answers. The ``all`` recall is printed (``-rA`` or ``-s``), not
  gated, beside ``tail``'s.
- ``test_replay_script_agrees``: ``scripts/replay_announced_calls.py --json`` over the logs root reports the same
  number of no-call answers and the same ``tail``/``all`` hit counts as this census (a file outside the corpus,
  e.g. a later session, would make the counts differ: the row skips, naming it, rather than compare different
  corpora).

**Private chats:** files directly under ``logs/sessions/`` are read and counted here, never copied; failure
messages name their rows by file and seq only, never by text.

The logs are untracked: each row looks under this checkout's ``logs/`` and then the main checkout's (the git common
dir's parent) and skips, naming the paths tried, when neither holds the corpus.

Regime: logs on the 5090 box as of 2026-10-27 (cut-off ``20261027-123442``), main 3e4d45b7; source runs lean, CL
256, 48 steps, thinking on, NVFP4 rev 2ea8372, vLLM 0.29.0, tools native, loop 8, unseeded. CPU, no GPU, no network.

Run (CPU; the ``gpu`` marker only gates selection, as the T-0142b and T-0170 evals)::

    uv run pytest tests/eval/test_t0172_census.py -q -m gpu -rA
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "tests" / "fixtures" / "t0172" / "labels.jsonl"
BUILDER = REPO / "tests" / "fixtures" / "t0172" / "build_labels.py"
SCRIPT = REPO / "scripts" / "replay_announced_calls.py"
MODULE = "local_astra.toolcall.announced_call"


def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("t0172_build_labels", BUILDER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[0]["provenance"], lines[1:]


def logs_root() -> Path:
    b = builder()
    prov, _rows = fixture()
    tried = b.logs_roots()
    for root in tried:
        if not root.is_dir():
            continue
        dirs = Counter(p.relative_to(root).parent.as_posix() for p in b.session_files(root))
        if dict(dirs) == prov["dirs"]:
            return root
    pytest.skip(
        f"T-0172 A4: no logs/ holding the labelled corpus ({prov['files']} session files up to "
        f"{prov['cutoff']} in {len(prov['dirs'])} directories); tried {[str(r) for r in tried]}"
    )


def corpus(root: Path) -> list[dict[str, Any]]:
    """Every ``generate``/``tool_step`` request of the corpus, with ``source``, ``private`` and ``no_call``."""
    b = builder()
    out: list[dict[str, Any]] = []
    for path in b.session_files(root):
        rel = path.relative_to(root).as_posix()
        for r in b.requests_of(path):
            r["source"] = rel
            r["private"] = b.is_private(rel)
            r["no_call"] = b.is_no_call_answer(r)
            out.append(r)
    return out


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise NotImplementedError(f"T-0172 A1: {MODULE} does not exist yet") from exc
        raise


def test_corpus_matches_the_fixture() -> None:
    prov, rows = fixture()
    reqs = corpus(logs_root())
    no_call = [r for r in reqs if r["no_call"]]
    assert len(reqs) == prov["requests"] == 6681
    assert sum(r["tools_on"] for r in no_call) == prov["tools_on_no_call_answers"] == 1987
    assert sum(not r["tools_on"] for r in no_call) == prov["tools_off_no_call_answers"] == 45
    by_key = {(r["source"], r["seq"]): r for r in reqs}
    for row in rows:
        got = by_key.get((row["source"], row["seq"]))
        assert got is not None and got["no_call"] and got["tools_on"], (row["source"], row["seq"])
        assert got["answer"] == row["answer"], (row["source"], row["seq"])


def test_census_tail() -> None:
    m = _mod()
    prov, rows = fixture()
    reqs = corpus(logs_root())
    labels = {(r["source"], r["seq"]): r["label"] for r in rows}
    on = [r for r in reqs if r["no_call"] and r["tools_on"]]
    off = [r for r in reqs if r["no_call"] and not r["tools_on"]]
    assert len(on) == 1987 and len(off) == 45, (len(on), len(off))

    def hits(mode: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in pool if m.announces(r["answer"], mode=mode) is not None]

    pos = [r for r in on if labels.get((r["source"], r["seq"])) is True]
    assert len(pos) == prov["positives"] == 108
    tail_hits = hits("tail", on)
    false = [
        (r["source"], r["seq"])
        for r in tail_hits
        if labels.get((r["source"], r["seq"])) is not True
    ]
    assert not false, (
        f"A4 precision 1.0 (tail): hits on negatives (labelled or by definition): {false}"
    )
    recall_tail = sum(labels.get((r["source"], r["seq"])) is True for r in tail_hits) / len(pos)
    all_hits = hits("all", on)
    recall_all = sum(labels.get((r["source"], r["seq"])) is True for r in all_hits) / len(pos)
    false_all = [
        (r["source"], r["seq"]) for r in all_hits if labels.get((r["source"], r["seq"])) is not True
    ]
    print(
        f"\nT-0172 A4 census (n = {len(on)} tools-on no-call answers, {len(pos)} labelled positive): "
        f"tail recall {recall_tail:.3f} ({round(recall_tail * len(pos))}/{len(pos)}), precision "
        f"{'1.000' if not false else 'below 1'}; all recall {recall_all:.3f} "
        f"({round(recall_all * len(pos))}/{len(pos)}), false hits {len(false_all)} (reported, not gated)"
    )
    missed = sorted(
        (r["source"], r["seq"]) for r in pos if m.announces(r["answer"], mode="tail") is None
    )
    assert recall_tail >= 0.95, f"A4 recall (tail) {recall_tail:.3f} < 0.95; missed: {missed}"
    smoke = [r for r in rows if r["smoke"]]
    assert len(smoke) == 2
    for s in smoke:
        got = m.announces(s["answer"], mode="tail")
        assert got is not None and got.verb == "search", (s["source"], s["seq"], got)
    off_hits = [
        (r["source"], r["seq"]) for r in off if m.announces(r["answer"], mode="tail") is not None
    ]
    assert not off_hits, f"A4: hits on tools-off requests: {off_hits}"


def test_replay_script_agrees() -> None:
    m = _mod()
    root = logs_root()
    if not SCRIPT.exists():
        raise NotImplementedError("T-0172 A3: scripts/replay_announced_calls.py does not exist yet")
    reqs = corpus(root)
    on = [r for r in reqs if r["no_call"] and r["tools_on"]]
    want_tail = sum(m.announces(r["answer"], mode="tail") is not None for r in on)
    want_all = sum(m.announces(r["answer"], mode="all") is not None for r in on)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(root), "--json"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = json.loads(proc.stdout)
    if doc["scanned"] != len(reqs):
        pytest.skip(
            f"the logs root holds sessions outside the labelled corpus ({doc['scanned']} requests scanned, "
            f"{len(reqs)} in the corpus): the script reads every file, the census only the corpus"
        )
    assert doc["no_call_answers"] == len(on) == 1987, doc["no_call_answers"]
    assert doc["hits"] == {"tail": want_tail, "all": want_all}, doc["hits"]
