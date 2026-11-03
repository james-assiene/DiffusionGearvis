"""T-0226: Gearvis on real student attempts (one typed turn per page of a private, sha-pinned manifest; tools off,
1120 visual tokens, the "Education tutor" built-in through the settings path); every A4 check, A5 verdict,
A7 count, the reading line, latency, thinking chars and canvases and the A8 label re-derived from the run's files and
compared with ``report.json`` (``gpu``; report-only: no bar, nothing adopts, no default changes).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0226_real_attempts.sh``)::

    T0226_RUN_DIR=logs/t0226-real-attempts-<sha7> T0226_MANIFEST=<the private manifest> \
        uv run pytest tests/eval/test_t0226_real_attempts.py -q -m gpu -rA

Every run row skips when ``T0226_RUN_DIR`` or ``T0226_MANIFEST`` is unset, or the run dir has no ``report.json``.
The rows spend no GPU time; they are ``gpu`` because the data exists only after a ``/launch``-reviewed run. Binding
lesson (T-0214 C2, T-0223 B6): ``tests/unit/test_t0226_real_attempts.py`` calls :func:`load_run` and every row of
``RUN_ROWS`` over a synthetic run dir built with the harness's own writers on CPU, so this module is not first
exercised on the GPU.

**The oracle.** This module imports no harness and nothing under ``scripts/`` or ``tests/``. From ``local_astra`` it
imports only the product's A2UI gate (``UiGate``: the shown card text is what that gate emits). Every other
definition is typed from docs/tasks/T-0226.md A3-A8 (and T-0223 B3/R-2/R-3, which A4/A5 carry over).

**Texts (T-0229 A3).** The tutor text and the typed question are read from the run's own regime
(``user_instructions``, ``question``) and must be one of ``KNOWN_TEXTS``: T-0225's text with T-0226's question (the
T-0226 B run at 2789ec90) or T-0229's text with "Can you double-check my work?". Every request's system text must end
with the header and the recorded text, and the turn's one ``user.text`` must be the recorded question.

Privacy: this file names no private path, page or keyword. The manifest is read only from ``T0226_MANIFEST``.

Files read (the shapes the harness writes; test-author readings, reported to the orchestrator)
=============================================================================================

- the private manifest (``T0226_MANIFEST``): ``{"version": 1, "pages": [{id, png, sha256, kind, key,
  judge_keywords}]}`` (A1). Its sha256 must equal the run's ``manifest.json["manifest_sha256"]``.
- ``order.json``: the page ids in run order; an entry may be an id or ``[id, 0]`` (both read).
- ``manifest.json`` (the run's, A9): ``manifest_sha256``, ``manifest_dir_name`` (the basename only), ``regime``,
  ``stopped`` (``None`` or why the run stopped early). Never the manifest's absolute path, never a PNG file name.
- ``probes.json``: ``{"P-S": {"passed": bool, ...}, "P0": {"passed": bool, "image_tokens": {id: {"size": [w, h],
  "image_tokens": int | None, "expected_soft_tokens": int}}, ...}}``.
- ``trials.jsonl``: T-0223's raw trial per page (``page``, ``repeat`` 0, ``session_log``, ``exception``,
  ``timed_out``, ``frame_sha256``, ``stored_sha256``, ``frame_digest``, ``user_text_ms``, ``bodies``,
  ``http_non_200``, ``engine_up_after``) plus ``thought_tokens`` (A6, ``None`` when ``/tokenize`` failed).
- ``turns.jsonl``, ``sessions/<session_log>``: the sessions' turn lines and raw logs.
- ``report.json``: ``label``; ``violations`` (and/or ``body_violations``), strings ``"<page>/r0: <what>"``;
  ``pages``: one row per page of ``order.json`` with ``id``, ``kind``, ``verdict``, ``spoken_only_verdict``,
  ``claims_mistake``, ``says_correct``, ``spoken`` (list), ``shown``, ``matched``, ``thought_chars``,
  ``thought_canvases``, ``thought_tokens``, ``first_word_ms``, ``frame_size``, ``image_tokens``; ``counts``
  (``mistake``: ``n``, ``located``, ``missed``, ``missed_said_correct``, ``errored``; ``correct``: ``n``,
  ``said_correct``, ``claimed_mistake``, ``no_verdict``, ``errored``; ``unclear``: ``n``, ``claims_mistake``,
  ``says_correct``, ``errored``);
  ``reading``; ``latency`` (``p50_ms``, ``max_ms``, ``n``, ``cold_first_turn_ms``); ``thinking`` (``p50_chars``,
  ``max_chars``); ``phrases`` (``NEGATED_CLAIM``, ``NO_MISTAKE``, ``CLAIM``: lists); ``regime``.
- ``report.md``: the reading line, every spoken text verbatim, the three phrase lists, ``## Hand audit (main
  session)`` with one line per page.

Readings (test-author rulings; reported, not resolved silently)
===============================================================

1. **Verdict strings**: ``located``, ``missed``, ``said_correct``, ``claimed_mistake``, ``no_verdict``, ``errored``.
   A ``mistake`` page that missed carries ``said_correct: true`` when ``says_correct``. An ``unclear`` page that ran
   has verdict ``no_verdict`` and both flags.
2. **Errored** (A5): status (T-0223 R-3: not run, an exception, an ``llm`` error record, a non-200 attempt, the
   engine down after, timed out) is not ``ran``, or the judged text is empty. The spoken-only verdict is the same
   kind rule over the spoken text alone; it is ``errored`` exactly when the verdict is.
3. **Phrase lists**: the report's three lists are the pre-registered constants; each must contain every phrase the
   task lists as an example (the "e.g." lists, read as the minimum). The verdicts are re-derived with the report's
   lists, so a list edited after the fact changes nothing unless it also changes the verdicts.
4. **Clauses**: casefold, ``’``/``‘`` become ``'``, split on ``.``, ``!``, ``?``, ``;``, ``,`` (R-4 a), a newline
   and the words ``but``, ``however``, ``mais``, ``cependant`` (R-4 a; word boundaries); a clause ending in ``?`` is a
   question, neither correct nor a claim (R-4 a); phrase matching is a substring test inside a clause; within one
   clause a negated claim wins over a no-mistake phrase. Keyword entries use T-0223 R-2
   (casefold substring of the judged text, ``|`` alternatives, no other normalisation).
5. **Thinking** (A6): ``thought_chars`` is the sum of ``len(text)`` over the turn's ``llm.commit`` records with
   ``segment == "thought"``; ``thought_canvases`` is their number; a turn that ran without a thought has 0.
   ``thinking.p50_chars`` is ``statistics.median_low`` over the turns that ran, ``max_chars`` their max.
6. **Latency**: T-0223's (``spans["tts.first_audio"][0] - user_text_ms``, ``median_low``) over turns that ran and
   spoke; the cold first turn is the first of ``order.json``.
7. **Label** (A8): ``void`` when a probe is missing or failed, or any executed turn has an A4 violation; else
   ``incomplete`` when a page of the order has no trial or its status is not ``ran``, or the run stopped; else
   ``measured``. An empty turn that ran is errored for its verdict but does not by itself make the run incomplete.
8. **Reading**: "L of M mistake pages located; S of C correct pages said correct, F claimed a mistake; U unclear
   pages reported", with M, C and U the pages of each kind in the manifest.
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

ENV_RUN = "T0226_RUN_DIR"
ENV_MANIFEST = "T0226_MANIFEST"

# ---- typed from docs/tasks/T-0226.md, T-0229.md and hud/src/settings/prompts.js (T-0225, T-0229) ---------------------------------
ORDER_SEED = 0
VISUAL_TOKENS = 1120
KINDS = ("mistake", "correct", "unclear")
#: T-0229 A3: the (tutor text, question) a run carries is read from the run's own regime and must be one of these
#: literals, so the T-0226 B run (2789ec90, T-0225's text) and a T-0229 run both re-derive.
#: the T-0226 B pair (docs/tasks/T-0226.md, T-0225's built-in), historical
T0226_QUESTION = "Here's my work on this page. Is it right? If there's a mistake, tell me where."
T0226_TUTOR_INSTRUCTIONS = (
    "You are a patient tutor. When I show you my work, find the line with the mistake, point to "
    "it, and explain why it is wrong. Ask me a guiding question before giving the fix, then give "
    "the corrected step and the correct final answer. Speak in plain words, not symbols, and "
    "keep each reply short."
)
#: the T-0229 pair (docs/tasks/T-0229.md A1, A2), current
QUESTION = "Can you double-check my work?"
TUTOR_INSTRUCTIONS = (
    "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, point to it, "
    "explain why, and ask me a guiding question before giving the fix; then give the corrected step and the "
    "correct final answer. If everything is right, say so plainly and don't invent a problem. Speak in plain "
    "words, not symbols, and keep each reply short."
)
KNOWN_TEXTS = (
    (T0226_TUTOR_INSTRUCTIONS, T0226_QUESTION),
    (TUTOR_INSTRUCTIONS, QUESTION),
)
USER_INSTRUCTIONS_HEADER = (
    "The user's own instructions follow. Apply them unless they conflict with anything above."
)
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
LOCATED, MISSED, SAID_CORRECT, CLAIMED, NO_VERDICT, ERRORED = (
    "located",
    "missed",
    "said_correct",
    "claimed_mistake",
    "no_verdict",
    "errored",
)
#: A5's example phrases, read as the minimum of each constant (reading 3)
REQUIRED_NEGATED_CLAIM = (
    "not correct",
    "isn't correct",
    "is not correct",
    "not right",
    "isn't right",
    "not quite",
    "pas correct",
    "pas juste",
    "n'est pas",
)
REQUIRED_NO_MISTAKE = (
    "no mistake",
    "no mistakes",
    "no error",
    "no errors",
    "don't see any",
    "do not see any",
    "didn't find any",
    "couldn't find any",
    "can't find any",
    "nothing wrong",
    "nothing is wrong",
    "nothing went wrong",
    "is correct",
    "are correct",
    "all correct",
    "looks correct",
    "looks right",
    "is right",
    "well done",
    "checks out",
    "aucune erreur",
    "pas d'erreur",
    "pas de faute",
    "c'est correct",
    "c'est juste",
    "tout est juste",
    "bravo",
    # R-4 (a)
    "don't see a",
    "do not see a",
    "didn't find a",
    "can't find a",
    "couldn't find a",
)
REQUIRED_CLAIM = (
    "mistake",
    "error",
    "wrong",
    "incorrect",
    "should be",
    "should have",
    "forgot",
    "missing",
    "instead of",
    "erreur",
    "faute",
    "faux",
    "fausse",
    "oublié",
    "au lieu de",
)
READING_RE = re.compile(
    r"(\d+) of (\d+) mistake pages located; (\d+) of (\d+) correct pages said correct, (\d+) claimed a "
    r"mistake; (\d+) unclear pages reported"
)
HAND_AUDIT_HEADING = "## Hand audit (main session)"


# =====================================================================================================
# loading
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _json(path: Path) -> Any:
    assert path.is_file(), f"T-0226: {path.name} is missing from the run dir"
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Everything the run rows read: the run dir's files and the private manifest (the unit module calls it on a
    synthetic dir and a synthetic manifest in ``tmp_path``)."""
    d = Path(run_dir)
    mp = Path(manifest_path)
    sessions: dict[str, list[dict[str, Any]]] = {}
    for p in sorted((d / "sessions").glob("*.jsonl")) if (d / "sessions").is_dir() else []:
        recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        sessions[p.name] = sorted(recs, key=lambda r: int(r.get("seq") or 0))
    raw = mp.read_bytes()
    manifest = json.loads(raw)
    return {
        "dir": d,
        "manifest_path": mp,
        "order": _json(d / "order.json") if (d / "order.json").is_file() else None,
        "run_manifest": _json(d / "manifest.json"),
        "run_manifest_text": (d / "manifest.json").read_text(encoding="utf-8"),
        "probes": _json(d / "probes.json") if (d / "probes.json").is_file() else {},
        "report": _json(d / "report.json"),
        "md": (d / "report.md").read_text(encoding="utf-8") if (d / "report.md").is_file() else "",
        "trials": _jsonl(d / "trials.jsonl"),
        "turns": _jsonl(d / "turns.jsonl"),
        "sessions": sessions,
        "pages": {str(p["id"]): p for p in manifest.get("pages") or []},
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture(scope="module")
def run() -> dict[str, Any]:
    raw = os.environ.get(ENV_RUN)
    man = os.environ.get(ENV_MANIFEST)
    if not raw or not man:
        pytest.skip(
            f"{ENV_RUN} or {ENV_MANIFEST} is unset (the rows read a /launch-reviewed run's files)"
        )
    path = Path(raw)
    if not (path / "report.json").is_file():
        pytest.skip(f"{path} has no report.json")
    return load_run(path, Path(man))


# =====================================================================================================
# the definitions (pure)
# =====================================================================================================


def order_oracle(ids: Sequence[str], seed: int = ORDER_SEED) -> list[str]:
    """A3: the sorted page ids shuffled once by ``random.Random(seed)``."""
    out = sorted(str(i) for i in ids)
    random.Random(seed).shuffle(out)
    return out


def _oid(entry: Any) -> str:
    return str(entry[0]) if isinstance(entry, list | tuple) else str(entry)


def spoken_of(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every ``tts.speak`` command's ``source`` in the turn's ``machine.step`` records (T-0223 reading 1)."""
    out: list[str] = []
    for r in sorted(records, key=lambda r: int(r.get("seq") or 0)):
        if r.get("kind") != "machine.step":
            continue
        for c in (r.get("data") or {}).get("commands") or []:
            if isinstance(c, Mapping) and c.get("type") == "tts.speak":
                src = c.get("source")
                if isinstance(src, str) and src:
                    out.append(src)
    return out


def shown_of(records: Sequence[Mapping[str, Any]]) -> str:
    """The ``Text`` strings of every surface the A2UI gate emits for the turn's ``ui`` commits, each once, joined
    with newlines (T-0223 reading 2)."""
    ui = sorted(
        (
            r
            for r in records
            if r.get("kind") == "llm.commit" and (r.get("data") or {}).get("segment") == "ui"
        ),
        key=lambda r: int(r.get("seq") or 0),
    )
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
    return "\n".join(texts)


def entry_hit(entry: str, folded: str) -> bool:
    """T-0223 R-2: any non-empty ``|`` alternative, casefolded, is a substring of the casefolded judged text."""
    return any(alt and alt.casefold() in folded for alt in str(entry).split("|"))


def _norm(text: str) -> str:
    return text.casefold().replace("’", "'").replace("‘", "'")


#: A5 as amended by R-4 (a): the delimiters, captured so a clause's own end is known
CLAUSE_DELIMS = r"([.!?;,\n]|\bbut\b|\bhowever\b|\bmais\b|\bcependant\b)"


def clause_flags(text: str, phrases: Mapping[str, Sequence[str]]) -> tuple[bool, bool]:
    """A5's clause rule (reading 4): ``(claims_mistake, says_correct)``."""
    neg = [_norm(p) for p in phrases["NEGATED_CLAIM"]]
    nom = [_norm(p) for p in phrases["NO_MISTAKE"]]
    cla = [_norm(p) for p in phrases["CLAIM"]]
    claim = correct = False
    parts = re.split(CLAUSE_DELIMS, _norm(text))
    for i in range(0, len(parts), 2):
        clause = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if delim == "?":
            continue  # R-4 (a): a question is neither a correct clause nor a claim clause
        if any(p in clause for p in neg):
            claim = True
        elif any(p in clause for p in nom):
            correct = True
        elif any(p in clause for p in cla):
            claim = True
    return claim, correct and not claim


def verdict_of(
    kind: str,
    ran: bool,
    spoken: Sequence[str],
    shown: str,
    keywords: Sequence[str],
    phrases: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """A5 for one page (readings 1, 2)."""
    judged = " ".join([*[s for s in spoken if s], *([shown] if shown else [])])
    spoken_text = " ".join(s for s in spoken if s)

    def one(text: str) -> tuple[str, bool, bool, bool]:
        claims, says = clause_flags(text, phrases)
        if kind == "mistake":
            folded = text.casefold()
            ok = bool(keywords) and all(entry_hit(k, folded) for k in keywords)
            return (LOCATED if ok else MISSED), claims, says, (not ok) and says
        if kind == "correct":
            return (
                (CLAIMED if claims else SAID_CORRECT if says else NO_VERDICT),
                claims,
                says,
                False,
            )
        return NO_VERDICT, claims, says, False

    if not ran or not judged.strip():
        claims, says = clause_flags(judged, phrases)
        return {
            "verdict": ERRORED,
            "spoken_only_verdict": ERRORED,
            "claims_mistake": claims,
            "says_correct": says,
            "said_correct": False,
        }
    v, claims, says, said = one(judged)
    sv, _c, _s, _x = one(spoken_text)
    return {
        "verdict": v,
        "spoken_only_verdict": sv,
        "claims_mistake": claims,
        "says_correct": says,
        "said_correct": said,
    }


def llm_errors(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for r in records
        if r.get("kind") == "error"
        and str((r.get("data") or {}).get("what") or "").startswith("llm")
    )


def status(trial: Mapping[str, Any] | None, records: Sequence[Mapping[str, Any]] = ()) -> str:
    """T-0223 R-3."""
    if trial is None:
        return "missing"
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
            content = m.get("content")
            if isinstance(content, str):
                return content
            return "".join(
                str(p.get("text") or "") for p in content or [] if p.get("type") == "text"
            )
    return ""


def _images(prompt: Mapping[str, Any]) -> list[Any]:
    out = []
    for m in prompt.get("messages") or []:
        content = m.get("content")
        for p in content if isinstance(content, list) else []:
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


def texts_of(run: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """T-0229 A3: the ``(user_instructions, question)`` the run's report regime records (the run's
    ``manifest.json`` regime when the report has none)."""
    reg = run["report"].get("regime") or (run["run_manifest"] or {}).get("regime") or {}
    tutor, question = reg.get("user_instructions"), reg.get("question")
    return (
        tutor if isinstance(tutor, str) else None,
        question if isinstance(question, str) else None,
    )


def turn_violations(run: Mapping[str, Any], trial: Mapping[str, Any]) -> list[str]:
    """A4 = T-0223 B3 (a)-(c) with the tutor built-in block, from the log and the trial's body summaries."""
    name = str(trial.get("session_log"))
    tag = f"{trial.get('page')}/r0"
    recs = run["sessions"].get(name)
    if recs is None:
        return [f"{tag}: session log {name} missing"]
    out: list[str] = []
    tutor, question = texts_of(run)
    block = USER_INSTRUCTIONS_HEADER + "\n" + str(tutor)
    if tutor is None or question is None:
        out.append(f"{tag}: the regime records no user_instructions or question")
    typed = [
        ((r.get("data") or {}).get("event") or {}).get("text")
        for r in recs
        if r.get("kind") == "machine.step"
        and ((r.get("data") or {}).get("event") or {}).get("type") == "user.text"
    ]
    if typed != [question]:
        out.append(f"{tag}: typed {typed!r}, not the recorded question")
    if trial.get("stored_sha256") is None or trial.get("stored_sha256") != trial.get(
        "frame_sha256"
    ):
        out.append(f"{tag}: the stored frame is not the posted one")
    reqs = [r["data"] for r in recs if r.get("kind") == "llm.request"]
    if not reqs:
        out.append(f"{tag}: no llm.request")
    systems: set[str] = set()
    for d in reqs:
        prompt = d.get("prompt") or {}
        if d.get("phase") in ("prefill", "generate") and _images(prompt) != [
            trial.get("frame_digest")
        ]:
            out.append(f"{tag}: {d.get('phase')} images {_images(prompt)}")
        text = _system_text(prompt)
        systems.add(text)
        if not text.endswith(block):
            out.append(f"{tag}: {d.get('request_id')} system text does not end with the block")
        if prompt.get("tools"):
            out.append(f"{tag}: carries tools")
    regime = (_turn_line(run, name) or {}).get("regime") or {}
    if regime.get("settings_rev") != 1 or regime.get("instructions_edited") is not True:
        out.append(f"{tag}: turn line settings_rev {regime.get('settings_rev')}")
    shas = {hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] for s in systems}
    if regime.get("system_sha12") not in shas or len(shas) != 1:
        out.append(f"{tag}: system_sha12 {regime.get('system_sha12')}")
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
        out.append(f"{tag}: settings.change")
    bodies = trial.get("bodies")
    if bodies is not None:
        if not any(b.get("phase") == "generate" for b in bodies):
            out.append(f"{tag}: no generate body")
        for b in bodies:
            if b.get("phase") in ("prefill", "generate") and list(b.get("images") or []) != [
                trial.get("frame_sha256")
            ]:
                out.append(f"{tag}: body {b.get('label')} images")
            if b.get("max_soft_tokens") != VISUAL_TOKENS:
                out.append(f"{tag}: body {b.get('label')} max_soft_tokens")
            if b.get("tools"):
                out.append(f"{tag}: body {b.get('label')} tools")
    return out


def thought_of(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Reading 5: ``(chars, canvases)`` of the ``segment == "thought"`` commits."""
    texts = [
        str((r.get("data") or {}).get("text") or "")
        for r in records
        if r.get("kind") == "llm.commit" and (r.get("data") or {}).get("segment") == "thought"
    ]
    return sum(len(t) for t in texts), len(texts)


def first_word_ms(run: Mapping[str, Any], trial: Mapping[str, Any]) -> int | None:
    line = _turn_line(run, str(trial.get("session_log")))
    span = ((line or {}).get("spans") or {}).get("tts.first_audio")
    if not span or trial.get("user_text_ms") is None:
        return None
    return int(span[0]) - int(trial["user_text_ms"])


def phrases_of(run: Mapping[str, Any]) -> dict[str, list[str]]:
    got = run["report"].get("phrases") or {}
    return {k: [str(p) for p in got.get(k) or []] for k in ("NEGATED_CLAIM", "NO_MISTAKE", "CLAIM")}


def derive(run: Mapping[str, Any]) -> dict[str, Any]:
    pages = run["pages"]
    phrases = phrases_of(run)
    trials = {str(t.get("page")): t for t in run["trials"]}
    order = [_oid(x) for x in run["order"] or []]
    per: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    latencies: list[int] = []
    thinking: list[int] = []
    cold: int | None = None
    incomplete = bool((run["run_manifest"] or {}).get("stopped"))
    for index, pid in enumerate(order):
        t = trials.get(pid)
        recs = run["sessions"].get(str((t or {}).get("session_log")), [])
        st = status(t, recs)
        ran = st == "ran"
        if not ran:
            incomplete = True
        page = pages[pid]
        spoken = spoken_of(recs) if t is not None and t.get("session_log") else []
        shown = shown_of(recs) if t is not None and t.get("session_log") else ""
        v = verdict_of(
            str(page["kind"]), ran, spoken, shown, list(page.get("judge_keywords") or []), phrases
        )
        chars, canvases = thought_of(recs)
        ms = first_word_ms(run, t) if ran and t is not None else None
        if ran and t is not None:
            violations += turn_violations(run, t)
            thinking.append(chars)
            if ms is not None:
                latencies.append(ms)
                if index == 0:
                    cold = ms
        per[pid] = {**v, "thought_chars": chars, "thought_canvases": canvases, "first_word_ms": ms}
    counts: dict[str, dict[str, int]] = {
        "mistake": {"n": 0, "located": 0, "missed": 0, "missed_said_correct": 0, "errored": 0},
        "correct": {"n": 0, "said_correct": 0, "claimed_mistake": 0, "no_verdict": 0, "errored": 0},
        "unclear": {"n": 0, "claims_mistake": 0, "says_correct": 0, "errored": 0},
    }
    for pid, page in pages.items():
        kind = str(page["kind"])
        row = per.get(pid) or {"verdict": ERRORED, "claims_mistake": False, "says_correct": False}
        c = counts[kind]
        c["n"] += 1
        if row["verdict"] == ERRORED:
            c["errored"] += 1
        elif kind == "mistake":
            c["located" if row["verdict"] == LOCATED else "missed"] += 1
            c["missed_said_correct"] += int(bool(row.get("said_correct")))
        elif kind == "correct":
            c[row["verdict"]] += 1
        else:
            c["claims_mistake"] += int(bool(row["claims_mistake"]))
            c["says_correct"] += int(bool(row["says_correct"]))
    probes = run["probes"] or {}
    probes_ok = set(probes) >= {"P-S", "P0"} and all(
        bool((probes.get(n) or {}).get("passed")) for n in ("P-S", "P0")
    )
    if not probes_ok or violations:
        label = VOID
    elif incomplete:
        label = INCOMPLETE
    else:
        label = MEASURED
    m, c, u = counts["mistake"], counts["correct"], counts["unclear"]
    reading = (
        f"{m['located']} of {m['n']} mistake pages located; {c['said_correct']} of {c['n']} correct pages said "
        f"correct, {c['claimed_mistake']} claimed a mistake; {u['n']} unclear pages reported"
    )
    return {
        "per": per,
        "counts": counts,
        "violations": violations,
        "label": label,
        "reading": reading,
        "p50": statistics.median_low(latencies) if latencies else None,
        "max": max(latencies) if latencies else None,
        "n_latency": len(latencies),
        "cold": cold,
        "thinking_p50": statistics.median_low(thinking) if thinking else None,
        "thinking_max": max(thinking) if thinking else None,
    }


def _report_pages(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(p.get("id")): p for p in run["report"].get("pages") or []}


# =====================================================================================================
# the run rows
# =====================================================================================================


def test_manifest_sha_re_checked(run: dict[str, Any]) -> None:
    """A8: the private manifest given to the eval is the one the run pinned (its sha256 in the run's
    ``manifest.json``); the run's manifest records only its sha and directory basename (A9)."""
    rm = run["run_manifest"]
    assert rm.get("manifest_sha256") == run["manifest_sha256"], (
        rm.get("manifest_sha256"),
        run["manifest_sha256"],
    )
    mdir = run["manifest_path"].resolve().parent
    assert rm.get("manifest_dir_name") == mdir.name, rm.get("manifest_dir_name")
    text = run["run_manifest_text"]
    assert str(mdir) not in text and str(run["manifest_path"].resolve()) not in text
    for page in run["pages"].values():
        assert Path(str(page["png"])).name not in text, "the run's manifest.json names a PNG"


def test_order_re_derived(run: dict[str, Any]) -> None:
    """A3: ``order.json`` is the sorted ids shuffled once by ``Random(0)``; each page once (n = 1)."""
    got = [_oid(x) for x in run["order"] or []]
    assert got == order_oracle(list(run["pages"])), got


def test_evidence_re_derived(run: dict[str, Any]) -> None:
    """A4: the report's violations name exactly the pages this oracle finds."""
    mine = derive(run)["violations"]
    rep = run["report"]
    got = list(rep.get("violations") or []) + list(rep.get("body_violations") or [])

    def pages(vs: Sequence[str]) -> set[str]:
        return {str(v).split(":", 1)[0].split("/", 1)[0] for v in vs}

    assert pages(got) == pages(mine), (got, mine)


def test_verdicts_re_derived(run: dict[str, Any]) -> None:
    """A5: each page's verdict, spoken-only verdict and both flags equal the oracle's."""
    mine = derive(run)["per"]
    got = _report_pages(run)
    assert set(got) == set(mine), sorted(set(got) ^ set(mine))
    for pid, want in mine.items():
        row = got[pid]
        assert row.get("verdict") == want["verdict"], (pid, row.get("verdict"), want)
        assert row.get("spoken_only_verdict") == want["spoken_only_verdict"], (pid, row, want)
        assert bool(row.get("claims_mistake")) == want["claims_mistake"], (pid, row, want)
        assert bool(row.get("says_correct")) == want["says_correct"], (pid, row, want)
        if want["verdict"] == MISSED:
            assert bool(row.get("said_correct")) == want["said_correct"], (pid, row, want)


def test_counts_re_derived(run: dict[str, Any]) -> None:
    """A7: the counts per kind."""
    mine = derive(run)["counts"]
    got = run["report"].get("counts") or {}
    for kind, want in mine.items():
        for k, v in want.items():
            assert int((got.get(kind) or {}).get(k, -1)) == v, (kind, k, got.get(kind), want)


def test_reading_re_derived(run: dict[str, Any]) -> None:
    """A7: the reading line, in ``report.json`` and verbatim in ``report.md``."""
    line = str(run["report"].get("reading") or "")
    assert READING_RE.search(line), line
    assert derive(run)["reading"] in line, (line, derive(run)["reading"])
    assert line in run["md"]


def test_latency_re_derived(run: dict[str, Any]) -> None:
    """A6/A7: first-word p50/max over the turns that spoke; the cold first turn marked."""
    d = derive(run)
    lat = run["report"].get("latency") or {}
    assert (lat.get("p50_ms"), lat.get("max_ms"), lat.get("n")) == (
        d["p50"],
        d["max"],
        d["n_latency"],
    ), lat
    assert lat.get("cold_first_turn_ms") == d["cold"], lat
    for pid, want in d["per"].items():
        assert _report_pages(run)[pid].get("first_word_ms") == want["first_word_ms"], pid


def test_thinking_re_derived(run: dict[str, Any]) -> None:
    """A6: thought chars and canvases per page (only ``segment == "thought"`` commits) and their p50/max; the token
    count is the trial's ``thought_tokens``."""
    d = derive(run)
    got = _report_pages(run)
    trials = {str(t.get("page")): t for t in run["trials"]}
    for pid, want in d["per"].items():
        if trials.get(pid) is None or not trials[pid].get("session_log"):
            continue
        assert int(got[pid].get("thought_chars")) == want["thought_chars"], (pid, got[pid])
        assert int(got[pid].get("thought_canvases")) == want["thought_canvases"], (pid, got[pid])
        assert got[pid].get("thought_tokens") == trials[pid].get("thought_tokens"), pid
    th = run["report"].get("thinking") or {}
    assert (th.get("p50_chars"), th.get("max_chars")) == (d["thinking_p50"], d["thinking_max"]), th


def test_label_re_derived(run: dict[str, Any]) -> None:
    """A8: void / incomplete / measured."""
    assert run["report"].get("label") == derive(run)["label"]


def test_phrase_lists_recorded(run: dict[str, Any]) -> None:
    """A5/A7: the three phrase lists are in the report and verbatim in ``report.md``, each holding the task's
    examples (reading 3)."""
    ph = phrases_of(run)
    for name, required in (
        ("NEGATED_CLAIM", REQUIRED_NEGATED_CLAIM),
        ("NO_MISTAKE", REQUIRED_NO_MISTAKE),
        ("CLAIM", REQUIRED_CLAIM),
    ):
        missing = [p for p in required if p not in ph[name]]
        assert missing == [], (name, missing)
        for p in ph[name]:
            assert p in run["md"], (name, p)


def test_regime_recorded(run: dict[str, Any]) -> None:
    """A7: the regime names 1120 visual tokens, tools off, search off, order seed 0, n = 1 and the manifest's sha."""
    reg = run["report"].get("regime") or {}
    assert reg.get("visual_tokens") == VISUAL_TOKENS, reg
    assert reg.get("tools") == "off", reg
    assert reg.get("search") in (False, "off"), reg
    assert reg.get("order_seed") == ORDER_SEED and reg.get("n") == 1, reg
    assert reg.get("manifest_sha256") == run["manifest_sha256"], reg


def test_tutor_text_and_question_recorded(run: dict[str, Any]) -> None:
    """T-0229 A3: the regime records the tutor text and the question in force, as a pair this module knows (the
    T-0226 B pair of 2789ec90, or T-0229's), and the run's ``manifest.json`` regime agrees with the report's.
    The turns are checked against the recorded pair by :func:`turn_violations` (A4)."""
    pair = texts_of(run)
    assert pair in KNOWN_TEXTS, ("an unknown or mixed (user_instructions, question) pair", pair)
    rm = (run["run_manifest"] or {}).get("regime") or {}
    for key, value in zip(("user_instructions", "question"), pair, strict=True):
        if key in rm:
            assert rm[key] == value, (key, rm[key], value)


def test_texts_verbatim_and_hand_audit(run: dict[str, Any]) -> None:
    """A7: every spoken text of every executed turn is in ``report.md`` verbatim; the hand-audit section has one line
    per page."""
    md = run["md"]
    for t in run["trials"]:
        for s in spoken_of(run["sessions"].get(str(t.get("session_log")), [])):
            assert s in md, (t.get("page"), s)
    assert HAND_AUDIT_HEADING in md
    section = md.split(HAND_AUDIT_HEADING, 1)[1].split("\n## ", 1)[0]
    for pid in run["pages"]:
        hits = [
            ln
            for ln in section.splitlines()
            if re.search(rf"(?<![\w-]){re.escape(pid)}(?![\w-])", ln)
        ]
        assert len(hits) == 1, (pid, hits)


#: the run rows, by name (the unit module runs each over its synthetic dirs)
RUN_ROWS = (
    test_manifest_sha_re_checked,
    test_order_re_derived,
    test_evidence_re_derived,
    test_verdicts_re_derived,
    test_counts_re_derived,
    test_reading_re_derived,
    test_latency_re_derived,
    test_thinking_re_derived,
    test_label_re_derived,
    test_phrase_lists_recorded,
    test_regime_recorded,
    test_tutor_text_and_question_recorded,
    test_texts_verbatim_and_hand_audit,
)
