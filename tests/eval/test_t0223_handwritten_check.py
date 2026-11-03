"""T-0223 group B: the GPU check that Gearvis finds the planted mistake on each handwritten page (10 pages x n = 3 =
30 typed turns, a shared-screen frame per turn, the tutor instructions through the settings path); re-derived from
the run's files and compared with ``report.json`` (``gpu``; report-only: no bar, nothing adopts, no default changes).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0223_handwritten_check.sh``)::

    T0223_RUN_DIR=logs/t0223-handwritten-<sha7> uv run pytest tests/eval/test_t0223_handwritten_check.py -q -m gpu -rA

Every run row skips when ``T0223_RUN_DIR`` is unset or the directory has no ``report.json``. The rows spend no GPU
time; they are ``gpu`` because the data exists only after a ``/launch``-reviewed run. Binding lesson (T-0214 C2):
the unit module (``tests/unit/test_t0223_check_harness.py``) calls :func:`load_run` and every run row here over a
synthetic run dir built with the T-0223 harness's own writers, on CPU: the clean dir passes every row and each
seeded fault fails its own.

**The oracle.** This module imports no harness and nothing under ``scripts/`` or ``tests/``. From ``local_astra``
it imports only the product's A2UI gate (``local_astra.conductor.ui_gate.UiGate``: "the text of any A2UI surface
shown" is what that gate emits, and a hand parser would disagree with it on every canvas the gate repairs). Every
other definition is typed from docs/tasks/T-0223.md B2-B6.

Files read (shapes the harness writes; test-author readings, reported to the orchestrator)
=========================================================================================

- ``order.json``: 30 ``[page_id, repeat]``, the page-major list (``M1..M5, P1..P5`` x ``0..2``) shuffled once by
  ``random.Random(0)`` (B2 ``ORDER_SEED = 0``).
- ``manifest.json``: ``regime`` (the full regime, B5), ``attempts_sha256``, ``page_sha256`` (``{id: sha256}``),
  ``stopped`` (``None`` or why the run stopped early).
- ``probes.json``: ``{"P-S": {"passed": bool, ...}, "P0": {"passed": bool, ...}}``.
- ``trials.jsonl``: one line per executed or not-run turn, in run order: ``page``, ``repeat``, ``session_log`` (a
  file name under ``sessions/``, ``None`` for a not-run turn), ``exception``, ``timed_out``, ``frame_sha256``
  (sha256 of the posted JPEG bytes), ``stored_sha256`` (sha256 of the JPEG the session's perception held when
  ``UserText`` was submitted), ``frame_digest`` (the session log's form of the posted image: sha256 of its base64
  text, first 8 hex, ``telemetry.session_log._redacted``), ``user_text_ms`` (session clock at ``UserText``).
- ``turns.jsonl``: the sessions' turn lines (``session_log`` absolute path, ``spans``, ``regime``).
- ``sessions/<session_log>``: the raw session logs.
- ``report.json``: ``label``; ``violations`` (strings ``"<page>/r<repeat>: <what>"``); ``pages`` =
  ``{id: {"located": k, "spoken_only_located": k, "n": 3}}``; ``turns`` = one entry per turn of ``order.json``
  (``page``, ``repeat``, ``located``, ``spoken_only_located``, ``spoken``, ``shown``); ``reading`` (the one line);
  ``latency`` = ``{"p50_ms": int | None, "max_ms": int | None, "n": int}``; ``regime``.
- ``report.md``: the reading line verbatim and every spoken text verbatim.

Readings (test-author rulings; reported, not resolved silently)
===============================================================

1. **Spoken text** (B4 "every ``assistant_spoken`` of the turn"): the ``source`` of every ``tts.speak`` command in
   the turn's ``machine.step`` records, in ``seq`` order (the committed substring the speech invariant checks; the
   turn line carries no spoken text). Clips (``play_clip``) are never text.
2. **Shown text**: every ``llm.commit`` of segment ``ui`` gated in log order by one fresh ``UiGate`` per turn
   (T-0205 C3's ``shown_text``), the ``Text`` components' strings, each once, first-seen order.
3. **Judged text** = the spoken sources and the shown lines joined with single spaces. **Located** = every
   ``judge_keywords`` entry is satisfied (ruling R-2: an entry is ``|``-separated alternatives, any one of which,
   casefolded, is a substring of the casefolded judged text) and no ``false_blame_keywords`` phrase (plain) is.
   ``attempts.json`` is ``{"attempts": [...]}`` (a bare list is read too). A not-run, errored, timed-out or empty turn is not located; n stays 3.
4. **B3 (a)** is checked on what the log carries: every ``prefill``/``generate`` ``llm.request`` holds exactly one
   image part, whose digest is the trial's ``frame_digest``; and ``stored_sha256 == frame_sha256``. B3 (c)'s
   ``max_soft_tokens == 1120`` is in the HTTP body, which the session log does not record: not re-derived here (the
   report's regime must say 1120).
5. **Latency**: ``spans["tts.first_audio"][0] - user_text_ms`` (both on the session clock) over turns whose line has
   that span; p50 is ``statistics.median_low`` (an even count is ambiguous in the task; the low median is a real
   observation).
6. **The label**: ``void`` when a probe's ``passed`` is false or any executed turn has a B3 violation; else
   ``incomplete`` when a turn of ``order.json`` has no trial, or a trial is not-run, errored or timed out, or
   ``manifest.stopped`` is set; else ``measured``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0223_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
#: group A's pinned ground truth (the harness pins its sha256; the report's regime must name the same)
ATTEMPTS = REPO / "demos" / "handwritten" / "attempts.json"

# ---- typed from docs/tasks/T-0223.md ------------------------------------------------------------------------------
PAGE_IDS = ("M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5")
REPEATS = 3
TOTAL_TURNS = 30
ORDER_SEED = 0
VISUAL_TOKENS = 1120
#: criterion 5's two texts
QUESTION = "Here's my attempt. Can you check it and tell me where I went wrong?"
TUTOR_INSTRUCTIONS = (
    "You are a patient tutor. Point to the line with the mistake and explain why it is wrong, then give the "
    "corrected step."
)
#: T-0213 A1-4's header (typed: ``prompt_builder.USER_INSTRUCTIONS_HEADER``)
USER_INSTRUCTIONS_HEADER = (
    "The user's own instructions follow. Apply them unless they conflict with anything above."
)
BLOCK = USER_INSTRUCTIONS_HEADER + "\n" + TUTOR_INSTRUCTIONS
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
READING_RE = re.compile(r"(\d+) of 10 pages located in ≥ 2 of 3 turns")


# =====================================================================================================
# loading
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _json(path: Path) -> Any:
    assert path.is_file(), f"T-0223: {path.name} is missing from the run dir"
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, attempts_path: Path | None = None) -> dict[str, Any]:
    """Everything the run rows read from one run directory (the unit module calls it on a synthetic dir with its
    own ``attempts_path``)."""
    d = Path(run_dir)
    ap = Path(attempts_path) if attempts_path is not None else ATTEMPTS
    sessions: dict[str, list[dict[str, Any]]] = {}
    for p in sorted((d / "sessions").glob("*.jsonl")) if (d / "sessions").is_dir() else []:
        recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        sessions[p.name] = sorted(recs, key=lambda r: int(r.get("seq") or 0))
    raw = ap.read_bytes() if ap.is_file() else b""
    attempts = json.loads(raw) if raw else []
    if isinstance(attempts, Mapping):
        attempts = attempts.get("attempts") or []
    return {
        "dir": d,
        "order": _json(d / "order.json") if (d / "order.json").is_file() else None,
        "manifest": _json(d / "manifest.json"),
        "probes": _json(d / "probes.json") if (d / "probes.json").is_file() else {},
        "report": _json(d / "report.json"),
        "md": (d / "report.md").read_text(encoding="utf-8") if (d / "report.md").is_file() else "",
        "trials": _jsonl(d / "trials.jsonl"),
        "turns": _jsonl(d / "turns.jsonl"),
        "sessions": sessions,
        "attempts": {str(a["id"]): a for a in attempts},
        "attempts_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(f"{ENV} is unset (the run rows read a /launch-reviewed run's files)")
    path = Path(raw)
    if not (path / "report.json").is_file():
        pytest.skip(f"{path} has no report.json")
    return path


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    return load_run(run_dir)


# =====================================================================================================
# the definitions (pure)
# =====================================================================================================


def order_oracle(seed: int = ORDER_SEED) -> list[tuple[str, int]]:
    turns = [(p, r) for p in PAGE_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


def _key(x: Mapping[str, Any]) -> tuple[str, int]:
    return (str(x.get("page")), int(x.get("repeat") or 0))


def _name(key: Sequence[Any]) -> str:
    return f"{key[0]}/r{key[1]}"


def spoken_of(records: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for r in records:
        if r.get("kind") != "machine.step":
            continue
        data = r.get("data") or {}
        for c in data.get("commands") or []:
            if isinstance(c, Mapping) and c.get("type") == "tts.speak":
                src = c.get("source")
                if isinstance(src, str) and src:
                    out.append(src)
    return out


def shown_of(records: Sequence[Mapping[str, Any]]) -> list[str]:
    ui = [
        r
        for r in records
        if r.get("kind") == "llm.commit" and (r.get("data") or {}).get("segment") == "ui"
    ]
    by_turn: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for r in ui:
        by_turn.setdefault(str(r.get("turn_id")), {}).setdefault(
            str(r["data"].get("request_id")), []
        ).append(r)
    texts: list[str] = []
    seen: set[str] = set()
    for requests in by_turn.values():
        gate = UiGate()
        for commits in requests.values():
            for r in commits:
                d = r["data"]
                result = gate.gate(
                    str(d.get("text") or ""),
                    turn_id=str(r.get("turn_id")),
                    canvas_idx=int(d.get("canvas_idx") or 0),
                    final=False,
                )
                if result.surface is None:
                    continue
                for msg in result.surface.messages:
                    update = msg.get("updateComponents") if isinstance(msg, Mapping) else None
                    comps = update.get("components") if isinstance(update, Mapping) else None
                    for c in comps if isinstance(comps, list) else []:
                        if (
                            isinstance(c, Mapping)
                            and c.get("component") == "Text"
                            and isinstance(c.get("text"), str)
                            and c["text"] not in seen
                        ):
                            seen.add(c["text"])
                            texts.append(c["text"])
    return texts


def entry_hit(entry: str, folded: str) -> bool:
    """Ruling R-2: a ``judge_keywords`` entry is ``|``-separated alternatives; any one, casefolded, as a substring
    of the casefolded judged text satisfies it (empty alternatives are never a hit)."""
    return any(alt and alt.casefold() in folded for alt in entry.split("|"))


def judge_text(
    spoken: Sequence[str], shown: Sequence[str], page: Mapping[str, Any]
) -> tuple[bool, bool]:
    """(located, spoken-only located) under reading 3."""

    def located(text: str) -> bool:
        t = text.casefold()
        keys = [str(k) for k in page.get("judge_keywords") or []]
        blame = [str(k) for k in page.get("false_blame_keywords") or []]
        if not keys or not text.strip():
            return False
        return all(entry_hit(k, t) for k in keys) and not any(b.casefold() in t for b in blame)

    return located(" ".join([*spoken, *shown])), located(" ".join(spoken))


def llm_errors(records: Sequence[Mapping[str, Any]]) -> int:
    """R-3: ``error`` records about an LLM call (``what`` starting ``llm``)."""
    return sum(
        1
        for r in records
        if r.get("kind") == "error"
        and str((r.get("data") or {}).get("what") or "").startswith("llm")
    )


def status(trial: Mapping[str, Any], records: Sequence[Mapping[str, Any]] = ()) -> str:
    """R-3: errored on an exception, an ``llm`` error record in the turn's log, a non-200 attempt
    (``http_non_200``) or ``engine_up_after is False``."""
    if not trial.get("session_log"):
        return "not_run"
    if (
        trial.get("exception")
        or llm_errors(records) > 0
        or int(trial.get("http_non_200") or 0) > 0
        or trial.get("engine_up_after") is False
    ):
        return "errored"
    if trial.get("timed_out"):
        return "timed_out"
    return "ran"


def _system_text(prompt: Mapping[str, Any]) -> str:
    for m in prompt.get("messages") or []:
        if m.get("role") == "system":
            return "".join(
                str(p.get("text") or "") for p in m.get("content") or [] if p.get("type") == "text"
            )
    return ""


def _images(prompt: Mapping[str, Any]) -> list[Any]:
    out = []
    for m in prompt.get("messages") or []:
        for p in m.get("content") or []:
            if isinstance(p, Mapping) and p.get("type") == "image_url":
                url = p.get("url")
                out.append(url.get("digest") if isinstance(url, Mapping) else None)
    return out


def _turn_line(run: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for line in run["turns"]:
        if Path(str(line.get("session_log") or "")).name == name:
            return line
    for r in run["sessions"].get(name, []):
        if r.get("kind") == "turn.record":
            return r.get("data")
    return None


def turn_violations(run: Mapping[str, Any], trial: Mapping[str, Any]) -> list[str]:
    """B3 (a)-(c) for one executed turn, from its log (reading 4)."""
    name = str(trial.get("session_log"))
    tag = _name(_key(trial))
    recs = run["sessions"].get(name)
    if recs is None:
        return [f"{tag}: session log {name} missing"]
    out: list[str] = []
    if trial.get("stored_sha256") != trial.get("frame_sha256"):
        out.append(f"{tag}: the stored frame is not the posted one")
    reqs = [r["data"] for r in recs if r.get("kind") == "llm.request"]
    if not reqs:
        out.append(f"{tag}: no llm.request")
    systems: set[str] = set()
    for d in reqs:
        prompt = d.get("prompt") or {}
        if d.get("phase") in ("prefill", "generate"):
            imgs = _images(prompt)
            if imgs != [trial.get("frame_digest")]:
                out.append(f"{tag}: {d.get('phase')} {d.get('request_id')} images {imgs}")
        sys_text = _system_text(prompt)
        systems.add(sys_text)
        if not sys_text.endswith(BLOCK):
            out.append(f"{tag}: {d.get('request_id')} system text does not end with the block")
        if prompt.get("tools"):
            out.append(f"{tag}: {d.get('request_id')} carries tools")
    line = _turn_line(run, name)
    regime = (line or {}).get("regime") or {}
    if regime.get("settings_rev") != 1 or regime.get("instructions_edited") is not True:
        out.append(f"{tag}: turn line settings {regime.get('settings_rev')}")
    shas = {hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] for s in systems}
    if regime.get("system_sha12") not in shas or len(shas) != 1:
        out.append(f"{tag}: system_sha12 {regime.get('system_sha12')} vs {sorted(shas)}")
    changes = [r for r in recs if r.get("kind") == "settings.change"]
    user = [
        int(r.get("seq") or 0)
        for r in recs
        if r.get("kind") == "machine.step"
        and ((r.get("data") or {}).get("event") or {}).get("type") == "user.text"
    ]
    if (
        len(changes) != 1
        or (changes[0].get("data") or {}).get("applied_at") != "immediate"
        or not user
        or int(changes[0].get("seq") or 0) > min(user)
    ):
        out.append(
            f"{tag}: settings.change {[(c.get('data') or {}).get('applied_at') for c in changes]}"
        )
    return out + body_violations(trial)


def body_violations(trial: Mapping[str, Any]) -> list[str]:
    """R-5 (F4): the body summaries the meter recorded in ``trials.jsonl`` (``bodies``: ``phase``, ``images`` as
    sha256 of each image, ``max_soft_tokens``, ``tools``), when present: a generate body exists; every
    prefill/generate body holds exactly the posted JPEG; every body carries ``max_soft_tokens == 1120`` and no
    tools."""
    bodies = trial.get("bodies")
    if bodies is None:
        return []
    tag = _name(_key(trial))
    out: list[str] = []
    if not any(b.get("phase") == "generate" for b in bodies):
        out.append(f"{tag}: no generate body")
    for b in bodies:
        if b.get("phase") in ("prefill", "generate") and list(b.get("images") or []) != [
            trial.get("frame_sha256")
        ]:
            out.append(f"{tag}: body {b.get('label')} images {b.get('images')}")
        if b.get("max_soft_tokens") != VISUAL_TOKENS:
            out.append(f"{tag}: body {b.get('label')} max_soft_tokens {b.get('max_soft_tokens')}")
        if b.get("tools"):
            out.append(f"{tag}: body {b.get('label')} carries tools")
    return out


def first_word_ms(run: Mapping[str, Any], trial: Mapping[str, Any]) -> int | None:
    recs = run["sessions"].get(str(trial.get("session_log")), [])
    if status(trial, recs) != "ran":
        return None
    line = _turn_line(run, str(trial.get("session_log")))
    span = ((line or {}).get("spans") or {}).get("tts.first_audio")
    if not span or trial.get("user_text_ms") is None:
        return None
    return int(span[0]) - int(trial["user_text_ms"])


def derive(run: Mapping[str, Any]) -> dict[str, Any]:
    trials = {_key(t): t for t in run["trials"]}
    turns: dict[tuple[str, int], tuple[bool, bool]] = {}
    violations: list[str] = []
    latencies: list[int] = []
    incomplete = bool((run["manifest"] or {}).get("stopped"))
    for key in order_oracle():
        t = trials.get(key)
        if t is None or status(t, run["sessions"].get(str(t.get("session_log")), [])) != "ran":
            incomplete = True
            turns[key] = (False, False)
            continue
        violations += turn_violations(run, t)
        recs = run["sessions"].get(str(t["session_log"]), [])
        turns[key] = judge_text(spoken_of(recs), shown_of(recs), run["attempts"][key[0]])
        ms = first_word_ms(run, t)
        if ms is not None:
            latencies.append(ms)
    pages = {
        p: {
            "located": sum(turns[(p, r)][0] for r in range(REPEATS)),
            "spoken_only_located": sum(turns[(p, r)][1] for r in range(REPEATS)),
        }
        for p in PAGE_IDS
    }
    probes_ok = all(bool((v or {}).get("passed")) for v in (run["probes"] or {}).values())
    if not probes_ok or set(run["probes"] or {}) != {"P-S", "P0"} or violations:
        label = VOID
    elif incomplete:
        label = INCOMPLETE
    else:
        label = MEASURED
    return {
        "turns": turns,
        "pages": pages,
        "reading_n": sum(1 for p in PAGE_IDS if pages[p]["located"] >= 2),
        "violations": violations,
        "label": label,
        "p50": statistics.median_low(latencies) if latencies else None,
        "max": max(latencies) if latencies else None,
        "n_latency": len(latencies),
    }


# =====================================================================================================
# the run rows
# =====================================================================================================


def test_order_re_derived(run: dict[str, Any]) -> None:
    """B2: ``order.json`` is the page-major list shuffled by ``Random(0)``; each turn once."""
    got = [tuple(x) for x in run["order"] or []]
    assert got == order_oracle(), got
    assert len(set(got)) == TOTAL_TURNS


def test_attempts_pinned(run: dict[str, Any]) -> None:
    """B1/B4: the keywords came from the pinned ``attempts.json`` (its sha256 in the manifest and the report's
    regime equals the file's), with every page judged by non-empty keywords."""
    sha = run["attempts_sha256"]
    assert sha is not None
    assert run["manifest"].get("attempts_sha256") == sha
    assert (run["report"].get("regime") or {}).get("attempts_sha256") == sha
    assert sorted(run["attempts"]) == sorted(PAGE_IDS)
    for p, a in run["attempts"].items():
        assert a.get("judge_keywords"), p
        assert isinstance(a.get("false_blame_keywords"), list), p


def test_evidence_re_derived(run: dict[str, Any]) -> None:
    """B3: the report's violations name exactly the turns this oracle finds."""
    mine = derive(run)["violations"]
    got = list(run["report"].get("violations") or []) + list(
        run["report"].get("body_violations") or []
    )
    assert {v.split(":", 1)[0] for v in got} == {v.split(":", 1)[0] for v in mine}, (got, mine)


def test_verdicts_re_derived(run: dict[str, Any]) -> None:
    """B4: each turn's located / spoken-only located equals the oracle's."""
    mine = derive(run)["turns"]
    got = {_key(t): t for t in run["report"].get("turns") or []}
    assert set(got) == set(mine), sorted(set(got) ^ set(mine))
    for key, (loc, spk) in mine.items():
        assert (bool(got[key].get("located")), bool(got[key].get("spoken_only_located"))) == (
            loc,
            spk,
        ), (_name(key), got[key])


def test_k_of_3_re_derived(run: dict[str, Any]) -> None:
    """B5: per page k/3 located and k/3 spoken-only, n = 3."""
    mine = derive(run)["pages"]
    got = run["report"].get("pages") or {}
    for p in PAGE_IDS:
        assert int(got[p]["located"]) == mine[p]["located"], (p, got[p])
        assert int(got[p]["spoken_only_located"]) == mine[p]["spoken_only_located"], (p, got[p])
        assert int(got[p]["n"]) == REPEATS


def test_reading_re_derived(run: dict[str, Any]) -> None:
    """B5: the one reading line, in the report and verbatim in ``report.md``."""
    n = derive(run)["reading_n"]
    line = str(run["report"].get("reading") or "")
    m = READING_RE.search(line)
    assert m is not None, line
    assert int(m.group(1)) == n, (line, n)
    assert line in run["md"]


def test_latency_re_derived(run: dict[str, Any]) -> None:
    """B5: time to first spoken word p50/max over the turns that spoke."""
    d = derive(run)
    lat = run["report"].get("latency") or {}
    assert (lat.get("p50_ms"), lat.get("max_ms"), lat.get("n")) == (
        d["p50"],
        d["max"],
        d["n_latency"],
    ), lat


def test_label_re_derived(run: dict[str, Any]) -> None:
    """B6: void / incomplete / measured."""
    assert run["report"].get("label") == derive(run)["label"]


def test_regime_recorded(run: dict[str, Any]) -> None:
    """B5: the regime names 1120 visual tokens, tools off, search off, seed 0, n = 3."""
    reg = run["report"].get("regime") or {}
    assert reg.get("visual_tokens") == VISUAL_TOKENS, reg
    assert reg.get("tools") in ("off", None) or reg.get("tools") == "off", reg
    assert reg.get("search") in (False, "off"), reg
    assert reg.get("order_seed") == ORDER_SEED and reg.get("n") == REPEATS, reg


def test_texts_verbatim(run: dict[str, Any]) -> None:
    """B5: every spoken source of every executed turn is in ``report.md`` verbatim."""
    for t in run["trials"]:
        if status(t, run["sessions"].get(str(t.get("session_log")), [])) != "ran":
            continue
        for s in spoken_of(run["sessions"].get(str(t["session_log"]), [])):
            assert s in run["md"], (_name(_key(t)), s)


#: the run rows, by name (the unit module runs each over its synthetic dirs)
RUN_ROWS = (
    test_order_re_derived,
    test_attempts_pinned,
    test_evidence_re_derived,
    test_verdicts_re_derived,
    test_k_of_3_re_derived,
    test_reading_re_derived,
    test_latency_re_derived,
    test_label_re_derived,
    test_regime_recorded,
    test_texts_verbatim,
)
