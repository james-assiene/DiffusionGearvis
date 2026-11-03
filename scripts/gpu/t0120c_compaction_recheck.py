"""T-0120c -- the compaction re-check: with T-0120b's invite stub, does the model search again when the result that
held the fact was elided? ``/launch`` ONLY (``scripts/locked/t0120c_compaction_recheck.sh``).

    uv run python scripts/gpu/t0120c_compaction_recheck.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing.** It is a thin harness over T-0115a's
(``scripts/gpu/t0115a_loop_requests.py``, loaded read-only by path under a fixed ``sys.modules`` name): the histories,
the arms (``e_steps``, ``e_prompts``, ``elided_steps``), ``E_BUDGET``, the step texts, ``classify_e``'s rules,
``_base_prompt``, ``_one``'s request path and timing, the ``/metrics`` deltas, the ``/tokenize`` counts and
``_first_per_cell``/``_placement_cost`` are T-0115a's objects, unchanged. The COMPACT arm is T-0115a's
``step_prompt(base, loop.compact(steps, E_BUDGET))``: this file carries no stub of its own.

Refusals (exit 2), all **before any socket** (the ``/models`` request is the first), in this order:

1. a ``--base-url`` that is not loopback (T-0093's ``check_loopback``);
2. a dirty tree (the dirty paths are named);
3. an existing, non-empty ``--out-dir``;
4. a T-0115a harness at ``T0115A_PATH`` whose sha256 is not ``T0115A_SHA256`` (re-read at call time);
5. a tool kit that will not load, has repair off, or is not ``bbfc53ccdd80``;
6. a case id the harness reads that is not a dev-split case of ``cases.jsonl``; a ``cases.jsonl`` off its pin (both
   read from T-0115a's attributes at call time);
7. an E budget that does not elide exactly steps 1-3 of every history;
8. criterion 4's literal check: an accepted string (normalised) in a question, step text, query, lead, pool item,
   filler or stub, or a FULL text whose accepted string is not only inside the fact result;
9. criterion 3's text check: a warm/FULL continuation text off its c398fdf sha256, or a COMPACT text that equals
   c398fdf's, lacks T-0120b's 132-char invite stub exactly 3 times, or carries an old-form stub.

What it runs (docs/tasks/T-0120c.md; the pre-registration is ``docs/cd/compaction-recheck.md``): 160 E requests
(``schedule()``): per history (T-0115a order), placement old then kept, a warm (the step-4 prompt) before repeats 0, 2,
4, 6, 8; old sends FULL then COMPACT in even repeats and COMPACT then FULL in odd; kept sends COMPACT only (no kept
FULL, H-T120c-3). No Q4, no Q5. Concurrency 1, unseeded, ``LlmOptions()``. 10 errored requests in a row stop the run.

Each line keeps ``_one``'s outcome (T-0115a's list and matcher) as ``outcome_t0115a`` and carries ``outcome``
recomputed from the same commits with criterion 4's list and normaliser (``score_line``), ``invented`` (criterion 6)
and, on a re-search, the query and ``query_repeats_history``.

Output (``--out-dir``, default ``logs/t0120c-compaction-recheck-<sha7>/``): ``manifest.json``, ``requests.jsonl``,
``report.json`` (``build_report(run_dir)``) and ``report.md``. Exit codes: 0 the run completed (whatever the
outcome); 1 the run stopped or the engine is not there; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.contracts.llm import Prompt
from local_astra.toolcall import loop

REPO = Path(__file__).resolve().parents[2]
#: T-0115a's harness, read-only; ``main`` re-hashes the file here at call time
T0115A_PATH: Path = REPO / "scripts" / "gpu" / "t0115a_loop_requests.py"
#: the fixed ``sys.modules`` name T-0115a is loaded under (its dataclasses look themselves up there while built)
T0115A_MODULE_NAME = "_t0115a_loop_requests_for_t0120c"
#: sha256 of scripts/gpu/t0115a_loop_requests.py at bef1c65 (unchanged since c398fdf; criteria 1 and 12)
T0115A_SHA256 = "dbbad65fba98e698d3a0157d0ed1523b17ca7648730bbf296599081dd4adde1f"


def _load_t0115a() -> ModuleType:
    """T-0115a's harness, imported read-only by path under ``T0115A_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0115A_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0115A_MODULE_NAME, T0115A_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0115A_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0115A_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0115A_MODULE_NAME, None)
        raise
    return module


T0115A: ModuleType = _load_t0115a()

# --- reused from T-0115a, unchanged (the same objects) ----------------------------------------------------------
E_HISTORIES = T0115A.E_HISTORIES
e_steps = T0115A.e_steps
e_prompts = T0115A.e_prompts
elided_steps = T0115A.elided_steps
STEP_TEXTS = T0115A.STEP_TEXTS
E_BUDGET: int = T0115A.E_BUDGET

SEARCH = "google_search"
WARM = "warm"
PLACEMENTS: tuple[str, ...] = ("old", "kept")
#: placement -> the arms sent (H-T120c-3: no kept FULL)
ARMS_SENT: dict[str, tuple[str, ...]] = {"old": ("FULL", "COMPACT"), "kept": ("COMPACT",)}
#: the three cells the bars read
CELLS: tuple[tuple[str, str], ...] = (("old", "FULL"), ("old", "COMPACT"), ("kept", "COMPACT"))
ERRORED = "errored"
OUTCOMES: tuple[str, ...] = ("correct", "re_search", "unsupported", "other", ERRORED)
BARS: tuple[str, ...] = ("C", "R", "I", "K")
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2

#: criterion 8 (H-T120c-1), exact fractions of each cell's n (40 on a complete run)
C_MIN = Fraction(36, 40)
R_MIN = Fraction(24, 40)
I_MAX = 0
K_MIN = Fraction(38, 40)
#: criterion 9 (H-T120c-5): this many errored arm requests (of 120) make the outcome inconclusive
ERRORED_INCONCLUSIVE = 3
N_ARM, N_WARM = 120, 40

#: the files whose bytes decide what was sent and how it was read (criterion 10: the eval skips on drift)
GATE_FILES: tuple[str, ...] = (
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/llm_client.py",
    "scripts/gpu/t0115a_loop_requests.py",
    "scripts/gpu/t0120c_compaction_recheck.py",
)

#: T-0120b criterion 2's stub block for a 5-result search response (132 chars). A check, not a stub: the COMPACT arm
#: is the product's ``loop.compact``; ``main`` refuses unless every COMPACT text holds this block exactly 3 times.
INVITE_STUB_BLOCK = (
    '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
    "results:5}<tool_response|>"
)
#: the old note's tail (the c398fdf stub, before T-0120b); a COMPACT text carrying it is refused
OLD_NOTE_TAIL = 'earlier result omitted<|"|>'

#: The c398fdf manifest's ``e_texts_sha256`` (manifest sha256 a33aa86b...): the sha256 of each final assistant text
#: (criterion 3). ``main`` reads it at call time.
C398FDF_E_TEXTS_SHA256: dict[str, str] = {
    "ferry/old/warm": "66facd5dd58801b24adb68717528a72926ebe38c62990543a4badebe70d98ccd",
    "ferry/old/FULL": "1e45eb586bd069c06f835dda293938aded6aba44701bacc0dd4d9fa82f346b6c",
    "ferry/old/COMPACT": "83388470c421a5b0bfc8cf88d714cacf05821c18301addd392d6d45c4ebba498",
    "ferry/kept/warm": "70391737d178aada7d23208f5b8f9d40bbc083f03253f42111d852707ab1751f",
    "ferry/kept/FULL": "60d4fae5a3b904aa7f0efef258445fff9d9eaee40435c98ef69f145a920322fa",
    "ferry/kept/COMPACT": "fd8cb5a28c260095c62b7f7c537a56e092ed98a5fd162fcb58671222a6606a4a",
    "market/old/warm": "1e29d6cc4c4a32cb9bbf4227c3d2705a69d03bdbfff5b3110361fd1543d359da",
    "market/old/FULL": "e0b7ebece939a2fc13aee6393df1048b00f14bd2cd304511e0aec120f88f80cf",
    "market/old/COMPACT": "5b20c1c45242d689e3e664b6c99150c5da275aa2d3ffc20b8e094f74dcf4b36a",
    "market/kept/warm": "7fe6893b950a45075703ea09668ecb3da29683b6dcd6aab34fc9a8a880f11177",
    "market/kept/FULL": "31a7b9c3db9816d29cdde1de54b3d107435c006263ac7fa388f61b07b5f2d46d",
    "market/kept/COMPACT": "624cf0628b6baf91d7fd6f7731968de55347ee95493eae8adc1274344d936832",
    "hall/old/warm": "4a1ab2a7a927d4b8723efc9cc4fb0f92b993ac49077614097cee9e82af4da803",
    "hall/old/FULL": "71628ecac085d8268a9c95590f48d18d019bdbc2ea196ce9e7231ae52007e825",
    "hall/old/COMPACT": "d814bc395067f4b65551b13c6f7497fc5b35fc5547073ef9376dcdb82a2839b1",
    "hall/kept/warm": "916f5b4e2de138f16cbcd7137d43e13b156c56a2122ee0132d44444b33d36689",
    "hall/kept/FULL": "63759caf6da10b752ce7bf1a29edf99748a1ab65b990b72bef74e29387366a41",
    "hall/kept/COMPACT": "8dab2148485459191a52692c1e85a912a43049ff50709ab7abbf93fd861a56eb",
    "lighthouse/old/warm": "836cc54f1c612161c56b51aa7c029ce2665de7356c5cdb36944c4d4b17bac2a2",
    "lighthouse/old/FULL": "4373591fdba11ebbce7c8cbb2b31e3c5fa5e4aa35b603ab96f4ea8686f907b01",
    "lighthouse/old/COMPACT": "d033f78e34fe3c93ae3ebecf9ef789f1d5e00e353633a127bdb39d3bfd8b87a8",
    "lighthouse/kept/warm": "9fddaac86b009b5be428d95e758e1ce44bd80adb15797b5c1c9175877315dd23",
    "lighthouse/kept/FULL": "3e726817514b40faaa3351ab5d9b08fce67ae0990ba8233b8888523db2a1aa04",
    "lighthouse/kept/COMPACT": "496e7fc308226896ee5995d3c3b26e81ca1552f7c2c5877ee1f176e4c1e825df",
}
#: c398fdf's ``/tokenize`` counts of the warm and FULL prompts (its manifest's ``prompt_tokens``); report-only:
#: the report flags any count of this run that differs (criterion 3)
C398FDF_PROMPT_TOKENS: dict[str, int] = {
    "E/ferry/old/warm": 3212,
    "E/ferry/old/FULL": 3587,
    "E/ferry/kept/warm": 3200,
    "E/market/old/warm": 3193,
    "E/market/old/FULL": 3562,
    "E/market/kept/warm": 3175,
    "E/hall/old/warm": 3168,
    "E/hall/old/FULL": 3543,
    "E/hall/kept/warm": 3159,
    "E/lighthouse/old/warm": 3162,
    "E/lighthouse/old/FULL": 3536,
    "E/lighthouse/kept/warm": 3161,
}
#: c398fdf's old-placement one-compaction cost (docs/cd/toolcall-loop-requests.md): measured miss median, predicted
C398FDF_OLD_MISS_TOKENS = 929.5
C398FDF_OLD_PREDICTED_MISS_TOKENS = 874.5

#: Criterion 4 (H-T120c-2): the complete pre-registered accepted lists, verbatim. Nothing is derived from them.
ACCEPTED: dict[str, tuple[str, ...]] = {
    "ferry": (
        "19:40",
        "7:40 pm",
        "7:40pm",
        "7:40 p.m",
        "7.40 pm",
        "7.40pm",
        "7.40 p.m",
        "7:40 in the evening",
        "twenty to eight",
        "seven forty pm",
        "seven forty p.m",
        "seven forty p m",
        "seven forty in the evening",
        "nineteen forty",
    ),
    "market": ("06:45", "6:45", "6.45", "quarter to seven", "six forty five"),
    "hall": (
        "1,187",
        "1187",
        "1 187",
        "one thousand one hundred and eighty seven",
        "one thousand one hundred eighty seven",
        "one thousand, one hundred and eighty seven",
        "one thousand, one hundred eighty seven",
        "eleven hundred and eighty seven",
        "eleven hundred eighty seven",
    ),
    "lighthouse": ("41.6", "forty one point six"),
}

# --- criterion 6 (H-T120c-4): the invented-value patterns, run with re.search on normalise(answer) ---------------
_HOUR = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_MINUTE = (
    r"(?:oh|o|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty"
    r"|forty|fifty)(?: (?:one|two|three|four|five|six|seven|eight|nine))?"
)
_MERIDIEM = r"(?:[ap]\.? ?m\b\.?|o['’]? ?clock\b|in the (?:morning|afternoon|evening)\b|at night\b)"
_DIGIT_HOUR = r"(?<![\d.,:])(?:[01]?\d|2[0-3])"
_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen"
    r"|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
)
_LENGTH_UNIT = r"(?:metres?|meters?|m|feet|foot|ft)\b"
_CLOCK_TIME: tuple[str, ...] = (
    # a digit h:mm / h.mm
    _DIGIT_HOUR + r"[:.][0-5]\d(?!\d)",
    # a digit hour (minutes optional) with am/pm, o'clock, in the morning|afternoon|evening, at night
    _DIGIT_HOUR + r"(?:[:.][0-5]\d)? ?" + _MERIDIEM,
    # an hour word (minute words optional) with the same
    r"\b" + _HOUR + r"(?: " + _MINUTE + r")? ?" + _MERIDIEM,
    # quarter/half/ten/five/twenty/twenty five to|past an hour word
    r"\b(?:quarter|half|ten|five|twenty|twenty five) (?:to|past) " + _HOUR + r"\b",
    # at <digit hour or hour word> (not "at one of", "at one point")
    r"\bat (?:(?:[01]?\d|2[0-3])(?!\d|[.,]\d)|" + _HOUR + r"\b)(?! (?:of|point)\b)",
)
INVENTED_PATTERNS: dict[str, tuple[str, ...]] = {
    "ferry": _CLOCK_TIME,
    "market": _CLOCK_TIME,
    "hall": (
        # a number >= 100 in digits (1,187 / 2,000 / 1187 / 240), not a decade (1960s, 1960's)
        r"(?<![\d.,])(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d{2,})(?!\d|[.,]\d|'?s\b)",
        r"\b(?:hundred|thousand)s?\b",
    ),
    "lighthouse": (
        # a number in digits or words followed by a length unit
        r"(?<![\d.,])\d+(?:[.,]\d+)? ?" + _LENGTH_UNIT,
        r"\b" + _NUMBER_WORD + r"(?: (?:and |point )?" + _NUMBER_WORD + r")* " + _LENGTH_UNIT,
    ),
}

_WS = re.compile(r"\s+")
_QUERY = re.compile(r'query:<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def _say(msg: str) -> None:
    print(f"t0120c: {msg}", file=sys.stderr, flush=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =====================================================================================================
# Criterion 4's matcher and criterion 6's filter (pure)
# =====================================================================================================


def normalise(text: str) -> str:
    """Criterion 4's normalisation: casefold, every ``-`` and ``–`` a space, runs of whitespace one space, stripped."""
    return _WS.sub(" ", text.casefold().replace("-", " ").replace("–", " ")).strip()


def _bounded(needle: str, hay: str) -> bool:
    """``needle`` in ``hay`` with no letter or digit on either side (R2)."""
    if not needle:
        return False
    return re.search(r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])", hay) is not None


def holds_c4(answer: str, accepted: str) -> bool:
    """Criterion 4's match: both texts normalised, then bounded."""
    return _bounded(normalise(accepted), normalise(answer))


def _classify_c4(commits: Sequence[Mapping[str, Any]], accepted: Sequence[str]) -> str:
    """``classify_e``'s rules (T-0115a's, called unchanged) with criterion 4's list and matcher: with no accepted
    string ``classify_e`` reads re-search / other / unsupported exactly as it would; an ``unsupported`` answer is then
    ``correct`` iff it holds an accepted string under criterion 4's match."""
    base = str(T0115A.classify_e(commits, ()))
    if base != "unsupported":
        return base
    answer = str(T0115A._answer_text(commits))
    return "correct" if any(holds_c4(answer, a) for a in accepted) else "unsupported"


def _errored(line: Mapping[str, Any]) -> bool:
    """Ruling R4, as ``_one`` reads it: an error recorded or a status other than 200."""
    return line.get("error") is not None or line.get("status") != 200


def _query(commits: Sequence[Mapping[str, Any]]) -> str | None:
    tools = [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]
    if not tools:
        return None
    found = _QUERY.search(str(tools[0].get("text", "")))
    return found.group(1) if found else None


def score_line(line: Mapping[str, Any]) -> dict[str, Any]:
    """One E line of T-0115a's ``requests.jsonl`` shape -> ``outcome`` (criterion 4), ``outcome_t0115a``
    (``classify_e`` with T-0115a's list, and R4), ``invented`` (criterion 6; old COMPACT only, False elsewhere) and,
    on a ``re_search`` line, ``query`` and ``query_repeats_history``. Pure."""
    hid = str(line.get("history"))
    history = next(x for x in T0115A.E_HISTORIES if x.id == hid)
    commits: list[Mapping[str, Any]] = list(line.get("commits") or [])
    if _errored(line):
        outcome = outcome15 = ERRORED
    else:
        outcome = _classify_c4(commits, ACCEPTED[hid])
        outcome15 = str(T0115A.classify_e(commits, history.accepted))
    invented = False
    if (
        (line.get("placement"), line.get("arm")) == ("old", "COMPACT")
        and outcome != ERRORED
        and not T0115A._ends_in_thought(commits)
    ):
        if outcome == "correct":
            invented = True
        else:
            text = normalise(str(T0115A._answer_text(commits)))
            invented = any(re.search(p, text) is not None for p in INVENTED_PATTERNS[hid])
    out: dict[str, Any] = {
        "outcome": outcome,
        "outcome_t0115a": outcome15,
        "invented": invented,
    }
    if outcome == "re_search":
        query = _query(commits)
        out["query"] = query
        out["query_repeats_history"] = query is not None and normalise(query) in {
            normalise(q) for q in history.queries
        }
    return out


# =====================================================================================================
# Criteria 8 and 9: the bars and the outcome (pure, exact)
# =====================================================================================================


def _cell(
    tally: Mapping[str, Mapping[str, Mapping[str, int]]], p: str, a: str
) -> Mapping[str, int]:
    return (tally.get(p) or {}).get(a) or {}


def _rate(cell: Mapping[str, int], key: str) -> Fraction:
    n = sum(int(v) for v in cell.values())
    return Fraction(int(cell.get(key, 0)), n) if n else Fraction(0)


def recheck_outcome(
    tally: Mapping[str, Mapping[str, Mapping[str, int]]], invented: int, complete: bool
) -> dict[str, Any]:
    """H-T120c-1 and -5 on ``tally[placement][arm][outcome]`` (criterion 4's list):

    - C: old FULL correct >= 36/40; R: old COMPACT re-search >= 24/40; I: old COMPACT invented = 0; K: kept COMPACT
      correct >= 38/40 (exact fractions of each cell's n);
    - "inconclusive: <reasons>" if C fails, if 3 or more arm requests errored, or if the run is incomplete (every
      reason named); else "stub re-searches" iff R, I and K hold; else "stub falls short: <every failing bar>"."""
    bars = {
        "C": _rate(_cell(tally, "old", "FULL"), "correct") >= C_MIN,
        "R": _rate(_cell(tally, "old", "COMPACT"), "re_search") >= R_MIN,
        "I": int(invented) <= I_MAX,
        "K": _rate(_cell(tally, "kept", "COMPACT"), "correct") >= K_MIN,
    }
    n_errored = sum(int(_cell(tally, p, a).get(ERRORED, 0)) for p, a in CELLS)
    n_arm = sum(int(v) for p, a in CELLS for v in _cell(tally, p, a).values())
    reasons: list[str] = []
    if not bars["C"]:
        full = _cell(tally, "old", "FULL")
        reasons.append(
            f"C fails (old FULL correct {int(full.get('correct', 0))}/{sum(full.values())}, "
            "below 36/40: the model does not use the 5-step history)"
        )
    if n_errored >= ERRORED_INCONCLUSIVE:
        reasons.append(
            f"{n_errored}/{n_arm} arm requests errored (>= {ERRORED_INCONCLUSIVE} make it inconclusive)"
        )
    if not complete:
        reasons.append("the run is incomplete")
    if reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    elif bars["R"] and bars["I"] and bars["K"]:
        outcome = "stub re-searches"
    else:
        outcome = "stub falls short: " + ", ".join(b for b in ("R", "I", "K") if not bars[b])
    return {"bars": bars, "outcome": outcome}


# =====================================================================================================
# The schedule (pure)
# =====================================================================================================


def schedule() -> list[dict[str, Any]]:
    """The 160 requests in send order: per history (T-0115a order), per placement (old, kept), a warm before
    repeats 0, 2, 4, 6, 8; old FULL then COMPACT in even repeats, COMPACT then FULL in odd; kept COMPACT only."""
    out: list[dict[str, Any]] = []
    for history in T0115A.E_HISTORIES:
        for placement in PLACEMENTS:
            for repeat in range(T0115A.E_REPEATS):
                base = {"row": "E", "history": history.id, "placement": placement}
                if repeat % T0115A.WARM_EVERY == 0:
                    out.append({**base, "arm": WARM, "repeat": repeat})
                arms = ARMS_SENT[placement]
                for arm in arms if repeat % 2 == 0 else arms[::-1]:
                    out.append({**base, "arm": arm, "repeat": repeat})
    for entry in out:
        entry["request_id"] = (
            f"t0120c-e-{entry['history']}-{entry['placement']}-{entry['arm']}-r{entry['repeat']}"
        )
    return out


# =====================================================================================================
# The pre-socket checks of criteria 3 and 4 (pure)
# =====================================================================================================


def literal_problems(accepted: Mapping[str, Sequence[str]]) -> list[str]:
    """Criterion 4: no accepted string (of any history), normalised, occurs in a question, step text, query, lead,
    pool item, filler sentence or stub; in each FULL text one of the history's own occurs only inside the fact
    result (the fact step's first result's snippet). Returns the problems, empty when none."""
    every = [s for strings in accepted.values() for s in strings]

    def found(text: str, strings: Sequence[str]) -> list[str]:
        t = normalise(text)
        return [s for s in strings if _bounded(normalise(s), t)]

    problems: list[str] = []
    first = T0115A.e_steps(T0115A.E_HISTORIES[0], "old")[0].response
    stub = loop.render_response(SEARCH, loop.elided_payload(first))
    for where, text in (
        *((f"step text {i + 1}", t) for i, t in enumerate(T0115A.STEP_TEXTS)),
        *(("short filler", f) for f in T0115A.SHORT_FILLER),
        ("the invite stub", INVITE_STUB_BLOCK),
        ("the stub loop.compact writes", stub),
    ):
        if hits := found(text, every):
            problems.append(f"{where}: {hits}")
    for history in T0115A.E_HISTORIES:
        literals: list[tuple[str, str]] = [("question", str(history.question))]
        literals += [(f"query {i + 1}", q) for i, q in enumerate(history.queries)]
        for kind, items in (("lead", history.leads), ("pool item", history.pool)):
            literals += [(f"{kind} {i}", " | ".join(item)) for i, item in enumerate(items)]
        literals += [("filler", f) for f in history.filler]
        for where, text in literals:
            if hits := found(text, every):
                problems.append(f"{history.id} {where}: {hits}")
        own = tuple(accepted.get(history.id, ()))
        for placement in PLACEMENTS:
            fact_step = T0115A.FACT_STEP[placement]
            for i, step in enumerate(T0115A.e_steps(history, placement)):
                where = f"{history.id}/{placement} step {i + 1}"
                if hits := found(step.text + loop.render_call(step.name, step.arguments), every):
                    problems.append(f"{where} text or call: {hits}")
                results = step.response["results"]
                if i == fact_step:
                    if not found(str(results[0]["snippet"]), own):
                        problems.append(f"{where}: no accepted string inside the fact result")
                    rest = [str(r[k]) for r in results[1:] for k in ("title", "url", "snippet")]
                    rest += [str(results[0]["title"]), str(results[0]["url"])]
                    rest.append(str(step.response["query"]))
                    if hits := [h for t in rest for h in found(t, every)]:
                        problems.append(f"{where}: outside the fact result: {hits}")
                elif hits := found(loop.render_response(step.name, step.response), every):
                    problems.append(f"{where} response: {hits}")
    return problems


def continuity_problems(texts: Mapping[str, str], pins: Mapping[str, str]) -> list[str]:
    """Criterion 3 on the 24 final texts (``"<history>/<placement>/<arm>"``): warm and FULL hash to their c398fdf
    pin; COMPACT does not, holds ``INVITE_STUB_BLOCK`` exactly 3 times and no old-form stub."""
    problems: list[str] = []
    off = [k for k, t in texts.items() if not k.endswith("/COMPACT") and _sha(t) != pins.get(k)]
    if off:
        problems.append(f"warm/FULL texts off their c398fdf pin: {', '.join(off)}")
    bad = [
        f"{k} ({t.count(INVITE_STUB_BLOCK)} invite stubs"
        + (", an old-form stub" if OLD_NOTE_TAIL in t else "")
        + (", equals c398fdf's" if _sha(t) == pins.get(k) else "")
        + ")"
        for k, t in texts.items()
        if k.endswith("/COMPACT")
        and (t.count(INVITE_STUB_BLOCK) != 3 or OLD_NOTE_TAIL in t or _sha(t) == pins.get(k))
    ]
    if bad:
        problems.append(
            "COMPACT texts without T-0120b's invite stub exactly 3 times (the stub loop.compact writes is "
            f"not T-0120b's): {'; '.join(bad)}"
        )
    return problems


# =====================================================================================================
# The report (pure: reads the run directory, opens no socket, writes nothing)
# =====================================================================================================


def _empty_tally() -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for p, a in CELLS:
        out.setdefault(p, {})[a] = dict.fromkeys(OUTCOMES, 0)
    return out


def build_report(run_dir: Path) -> dict[str, Any]:
    """``report.json``'s content from ``requests.jsonl`` (and ``manifest.json`` when present): both tallies, the
    invented count and its answers verbatim, the bars, the outcome, and the report-only rows. Writes nothing."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    lines = [x for x in T0115A._read_jsonl(run_dir / "requests.jsonl") if x.get("row") == "E"]
    arm_lines = [x for x in lines if (x.get("placement"), x.get("arm")) in CELLS]
    tally, tally15 = _empty_tally(), _empty_tally()
    for x in arm_lines:
        p, a = str(x["placement"]), str(x["arm"])
        if x.get("outcome") in OUTCOMES:
            tally[p][a][str(x["outcome"])] += 1
        if x.get("outcome_t0115a") in OUTCOMES:
            tally15[p][a][str(x["outcome_t0115a"])] += 1
    old_compact = [x for x in arm_lines if (x["placement"], x["arm"]) == ("old", "COMPACT")]
    flagged = [
        {
            "request_id": str(x["request_id"]),
            "answer": str(T0115A._answer_text(x.get("commits") or [])),
        }
        for x in old_compact
        if x.get("invented")
    ]
    planned = schedule()
    written = {
        "warm": sum(1 for x in lines if x.get("arm") == WARM),
        "arm": len(arm_lines),
        "all": len(lines),
    }
    want = {
        "warm": sum(1 for e in planned if e["arm"] == WARM),
        "arm": sum(1 for e in planned if e["arm"] != WARM),
        "all": len(planned),
    }
    complete = written == want and not manifest.get("stopped")
    rule = recheck_outcome(tally, len(flagged), complete)
    n_errored = sum(tally[p][a][ERRORED] for p, a in CELLS)

    re_search = [x for x in arm_lines if x.get("outcome") == "re_search"]
    by_history = {
        h.id: {
            "k": sum(
                1 for x in old_compact if x["history"] == h.id and x.get("outcome") == "re_search"
            ),
            "n": sum(1 for x in old_compact if x["history"] == h.id),
        }
        for h in T0115A.E_HISTORIES
    }
    repeats = {
        "k": sum(1 for x in re_search if x.get("query_repeats_history")),
        "n": len(re_search),
    }

    predicted: Mapping[str, Any] = manifest.get("predicted_miss_tokens") or {}
    cells = T0115A._first_per_cell(lines)
    cost = {
        p: T0115A._placement_cost(
            [c for (_, cp), c in cells.items() if cp == p],
            [
                n
                for k, v in predicted.items()
                if str(k).endswith(f"/{p}") and (n := T0115A._num(v)) is not None
            ],
        )
        for p in PLACEMENTS
    }
    cost["c398fdf_old"] = {
        "compact_miss_tokens_first_per_cell_median": C398FDF_OLD_MISS_TOKENS,
        "predicted_miss_tokens_median": C398FDF_OLD_PREDICTED_MISS_TOKENS,
    }
    tokens: Mapping[str, Any] = manifest.get("prompt_tokens") or {}
    differ = {
        k: {"this_run": tokens.get(k), "c398fdf": v}
        for k, v in C398FDF_PROMPT_TOKENS.items()
        if k in tokens and tokens.get(k) != v
    }
    return {
        "task": "T-0120c",
        "regime": manifest.get("regime", {}),
        "requests": {"planned": want, "written": written},
        "complete": complete,
        "stopped": manifest.get("stopped"),
        "tally": tally,
        "tally_t0115a": tally15,
        "invented": {"k": len(flagged), "n": sum(tally["old"]["COMPACT"].values())},
        "invented_answers": flagged,
        "errored": {"k": n_errored, "n": len(arm_lines)},
        "bars": rule["bars"],
        "outcome": rule["outcome"],
        "thresholds": {
            "C": "old FULL correct >= 36/40 (else inconclusive)",
            "R": "old COMPACT re-search >= 24/40 (c398fdf: 9/40)",
            "I": "old COMPACT invented = 0/40",
            "K": "kept COMPACT correct >= 38/40",
            "errored": f">= {ERRORED_INCONCLUSIVE}/120 errored arm requests make it inconclusive",
        },
        "re_search_by_history": by_history,
        "query_repeats_history": repeats,
        "cost": cost,
        "prompt_tokens_differing_from_c398fdf": differ,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """``report.md``: every k/n beside its threshold, the flagged answers verbatim, and the regime."""
    out = [
        "# T-0120c -- compaction re-check (T-0120b invite stub)",
        "",
        f"**Outcome:** {report['outcome']}",
        "",
        f"Regime: {json.dumps(report.get('regime', {}), ensure_ascii=False)}",
        "",
        f"Requests planned {report['requests']['planned']}, written {report['requests']['written']}; "
        f"complete: {report['complete']}; stopped: {report.get('stopped') or 'no'}",
        "",
    ]
    for key, title in (
        ("tally", "criterion 4's list"),
        ("tally_t0115a", "T-0115a's list (like for like with c398fdf)"),
    ):
        out += [
            f"## Tally under {title}",
            "",
            "| placement | arm | " + " | ".join(OUTCOMES) + " |",
            "|---|---|" + "---|" * len(OUTCOMES),
        ]
        for p, a in CELLS:
            cell = report[key][p][a]
            n = sum(cell.values())
            out.append(f"| {p} | {a} | " + " | ".join(f"{cell[o]}/{n}" for o in OUTCOMES) + " |")
        out.append("")
    out += ["| bar | rule | holds |", "|---|---|---|"]
    for b in BARS:
        out.append(f"| {b} | {report['thresholds'][b]} | {report['bars'][b]} |")
    inv = report["invented"]
    out += [
        "",
        f"Invented (old COMPACT): {inv['k']}/{inv['n']}; errored arm requests "
        f"{report['errored']['k']}/{report['errored']['n']}.",
        "",
        "Flagged answers, verbatim:",
        "",
    ]
    out += [f"- `{x['request_id']}`: {x['answer']!r}" for x in report["invented_answers"]] or [
        "- none"
    ]
    rq = report["query_repeats_history"]
    out += [
        "",
        "Report-only: old COMPACT re-searches per history "
        + ", ".join(f"{h} {v['k']}/{v['n']}" for h, v in report["re_search_by_history"].items())
        + f"; re-search queries repeating one of the history's {rq['k']}/{rq['n']}.",
        "",
        "Cost per placement (report-only; old is the one-compaction cost; c398fdf old: "
        f"{C398FDF_OLD_MISS_TOKENS} measured, {C398FDF_OLD_PREDICTED_MISS_TOKENS} predicted):",
        "",
        "| placement | first COMPACT miss tokens (n) | predicted | done ms diff (first pair) |",
        "|---|---|---|---|",
    ]
    for p in PLACEMENTS:
        c = report["cost"][p]
        out.append(
            f"| {p} | {T0115A._fmt(c['compact_miss_tokens_first_per_cell_median'])} "
            f"({c['compact_miss_tokens_first_per_cell_n']}) | {T0115A._fmt(c['predicted_miss_tokens_median'])} "
            f"| {T0115A._fmt(c['done_ms_diff_median'])} |"
        )
    differ = report["prompt_tokens_differing_from_c398fdf"]
    out += [
        "",
        "Warm/FULL prompt-token counts differing from c398fdf (report-only): "
        + (
            ", ".join(f"{k} {v['this_run']} vs {v['c398fdf']}" for k, v in differ.items()) or "none"
        ),
        "",
    ]
    return "\n".join(out)


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied; reached from main only)
# =====================================================================================================


async def _run_schedule(run: Any) -> None:
    """The schedule, in order, at concurrency 1, each request through T-0115a's ``_one``. ``STOP_AFTER_ERRORS``
    errored requests in a row stop the run."""
    try:
        for entry in schedule():
            line = await T0115A._one(run, entry)
            scored = score_line(line)
            # _one classified with T-0115a's list and matcher: kept as outcome_t0115a; outcome is criterion 4's
            line["outcome_t0115a"] = line["outcome"]
            line.update({k: v for k, v in scored.items() if k != "outcome_t0115a"})
            T0115A._write_line(run, line)
            _say(
                f"{entry['request_id']}: {line['outcome']} (t0115a {line['outcome_t0115a']}) "
                f"done={line['done_ms']}ms" + (f" ERROR {line['error']}" if line["error"] else "")
            )
            run.errors_in_a_row = 0 if line["error"] is None else run.errors_in_a_row + 1
            if run.errors_in_a_row >= T0115A.STOP_AFTER_ERRORS:
                run.manifest["stopped"] = (
                    f"{run.errors_in_a_row} consecutive requests errored (last: {entry['request_id']})"
                )
                _say(f"stopping: {run.manifest['stopped']}")
                return
    finally:
        await run.client.aclose()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    t93 = T0115A._T93
    # --- refusals, before any socket ------------------------------------------------------------------
    try:
        t93.check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: not a loopback URL: {exc}")
        return EXIT_REFUSED
    try:
        dirty = t93._git("status", "--porcelain").rstrip("\n")
        sha = t93._git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    out: Path = args.out_dir or REPO / "logs" / f"t0120c-compaction-recheck-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
        return EXIT_REFUSED
    t15_path = Path(T0115A_PATH)
    try:
        t15_sha = hashlib.sha256(t15_path.read_bytes()).hexdigest()
    except OSError as exc:
        _say(f"refusing: the T-0115a harness {t15_path} cannot be read: {exc}")
        return EXIT_REFUSED
    if t15_sha != T0115A_SHA256:
        _say(
            f"refusing: the T-0115a harness {t15_path} has sha256 {t15_sha}, not the pinned {T0115A_SHA256} "
            "(criterion 12: it stays byte-identical)"
        )
        return EXIT_REFUSED

    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    try:
        kit, detokenizer = cli.load_tool_kit(profile)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(
            f"refusing: the tool repair is off ({kit.repair_off}); a run with repair off is invalid"
        )
        return EXIT_REFUSED
    if kit.tools_version[:12] != T0115A.TOOLS_SHA12:
        _say(f"refusing: the tool set is {kit.tools_version[:12]}, not {T0115A.TOOLS_SHA12}")
        return EXIT_REFUSED
    cases_path: Path = T0115A.CASES_PATH
    try:
        raw_cases = cases_path.read_bytes()
        rows = {
            str(r["id"]): r
            for r in (json.loads(x) for x in raw_cases.decode("utf-8").splitlines() if x.strip())
        }
    except (OSError, ValueError, KeyError) as exc:
        _say(f"refusing: {cases_path} cannot be read: {exc}")
        return EXIT_REFUSED
    screen_case = str(T0115A.E_SCREEN_CASE)
    read_ids = list(dict.fromkeys((*T0115A.DEV_CASE_IDS, screen_case)))
    not_dev = [cid for cid in read_ids if (rows.get(cid) or {}).get("split") != T0115A.SPLIT]
    if not_dev:
        _say(
            f"refusing: case ids that are not dev-split cases of {cases_path.name}: {', '.join(not_dev)} "
            "(the test split is never read)"
        )
        return EXIT_REFUSED
    got_sha = hashlib.sha256(raw_cases).hexdigest()
    if got_sha != T0115A.CASES_SHA256:
        _say(
            f"refusing: {cases_path.name} has sha256 {got_sha}, not the pinned {T0115A.CASES_SHA256}"
        )
        return EXIT_REFUSED
    elided = {
        f"{x.id}/{p}": list(T0115A.elided_steps(x, p))
        for x in T0115A.E_HISTORIES
        for p in PLACEMENTS
    }
    off = {k: v for k, v in elided.items() if v != [0, 1, 2]}
    if off:
        _say(f"refusing: E_BUDGET {T0115A.E_BUDGET} does not elide exactly steps 1-3: {off}")
        return EXIT_REFUSED
    accepted = {k: tuple(v) for k, v in ACCEPTED.items()}
    problems = literal_problems(accepted)
    if problems:
        _say(
            "refusing: criterion 4: accepted strings in the literals, or not only in the fact result:\n  "
            + "\n  ".join(problems)
        )
        return EXIT_REFUSED

    import httpx

    from local_astra.conductor import prompt_builder as pb
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT

    try:
        endpoint = VllmEndpoint(base_url=args.base_url, request_timeout_s=T0115A.REQUEST_TIMEOUT_S)
    except ValueError as exc:
        _say(f"refusing: not a loopback URL the client accepts: {exc}")
        return EXIT_REFUSED
    thinking = bool(profile.diffusion.thinking)
    builder = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, thinking=thinking, tools=kit.tools)
    e_arms: dict[tuple[str, str, str], Prompt] = {}
    for history in T0115A.E_HISTORIES:
        base = T0115A._base_prompt(builder, history.question, rows[screen_case])
        for placement in PLACEMENTS:
            for arm, prompt in T0115A.e_prompts(base, history, placement).items():
                e_arms[(history.id, placement, arm)] = prompt
    texts = {f"{h}/{p}/{a}": prompt.messages[-1].text() for (h, p, a), prompt in e_arms.items()}
    problems = continuity_problems(texts, dict(C398FDF_E_TEXTS_SHA256))
    if problems:
        _say("refusing: criterion 3 (regime continuity with c398fdf):\n  " + "\n  ".join(problems))
        return EXIT_REFUSED

    client = VllmClient(endpoint, transport=httpx.AsyncHTTPTransport(), detokenizer=detokenizer)
    sent = {(e["history"], e["placement"], e["arm"]) for e in schedule()}
    bodies = {
        f"E/{h}/{p}/{a}": client.request_body(prompt, "body")
        for (h, p, a), prompt in e_arms.items()
        if (h, p, a) in sent
    }
    # the predicted miss (T-0115a's reading, R3). old: the COMPACT prompt's tokens after the first elided block (its
    # warm, steps 1-4 in full, is cached up to there). kept: from where it diverges from the same history's old
    # COMPACT, sent earlier
    cut_bodies: dict[str, dict[str, Any]] = {}
    for history in T0115A.E_HISTORIES:
        for placement in PLACEMENTS:
            compact_body = bodies[f"E/{history.id}/{placement}/COMPACT"]
            text = compact_body["messages"][-1]["content"]
            if placement == "old":
                first = T0115A.e_steps(history, placement)[0].response
                cut_at = text.index(loop.render_response(SEARCH, loop.elided_payload(first)))
            else:
                old_text = bodies[f"E/{history.id}/old/COMPACT"]["messages"][-1]["content"]
                cut_at = len(os.path.commonprefix([text, old_text]))
            cut = json.loads(json.dumps(compact_body))
            cut["messages"][-1]["content"] = text[:cut_at]
            cut_bodies[f"{history.id}/{placement}"] = cut

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    listing = t93._get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if endpoint.model not in served:
        _say(
            f"no engine serving {endpoint.model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    # outside every timed span: the prompts' engine token counts and the predicted misses
    prompt_tokens = {key: T0115A._tokens(args.base_url, body) for key, body in bodies.items()}
    predicted: dict[str, int | None] = {}
    for key, cut in cut_bodies.items():
        whole = prompt_tokens.get(f"E/{key}/COMPACT")
        head = T0115A._tokens(args.base_url, cut)
        predicted[key] = None if whole is None or head is None else whole - head

    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": "T-0120c",
        "rule": "H-T120c-1",
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": endpoint.model,
            "engine_version": t93._get_text(t93._root(args.base_url) + "/version"),
            "versions_lock_sha256": t93._sha256(REPO / "services" / "versions.lock"),
            "cases_sha256": T0115A.CASES_SHA256,
            "split": T0115A.SPLIT,
            "tools": f"native:{kit.tools_version[:12]}",
            "repair": "on",
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": thinking,
            "seed": None,
            "concurrency": 1,
            "options": "LlmOptions() as the session sends it",
            "requests": len(schedule()),
            "e_budget": T0115A.E_BUDGET,
            "compact": "loop.compact(steps, E_BUDGET) at its defaults (T-0115a's e_prompts; the T-0120b stub)",
            "warm": "the step-4 prompt before repeats 0, 2, 4, 6, 8 per (history, placement)",
            "e_screen_case": screen_case,
            "t0115a_sha256": t15_sha,
            "accepted_match": "criterion 4: casefold, every - and – a space, whitespace collapsed, no letter or "
            "digit next to the match (R2); the complete pre-registered lists (accepted)",
            "outcome_t0115a": "T-0115a's _one / classify_e with its own lists (accepted_t0115a), unchanged",
            "thought_open": "the last commit's (the final state, R1)",
            "errored": f"non-200, timeout, transport error or a stream without a finish_reason; never correct; "
            f">= {ERRORED_INCONCLUSIVE}/120 arm requests make it inconclusive (H-T120c-5)",
            "differences_from_c398fdf": "the commit; loop.py (the stub only); a fresh boot with E first; no "
            "Q4/Q5; no kept FULL",
        },
        "gate_shas": {p: t93._sha256(REPO / p) for p in GATE_FILES},
        "accepted": {k: list(v) for k, v in accepted.items()},
        "accepted_t0115a": {x.id: list(x.accepted) for x in T0115A.E_HISTORIES},
        "invented_patterns": {k: list(v) for k, v in INVENTED_PATTERNS.items()},
        "e_texts_sha256": {k: _sha(t) for k, t in texts.items()},
        "elided": elided,
        "prompt_tokens": prompt_tokens,
        "predicted_miss_tokens": predicted,
    }
    T0115A._write_json(out / "manifest.json", manifest)
    raw = httpx.AsyncClient(
        base_url=args.base_url,
        transport=httpx.AsyncHTTPTransport(),
        timeout=T0115A.REQUEST_TIMEOUT_S,
        trust_env=False,
    )
    run = T0115A._Run(
        out=out,
        base_url=args.base_url,
        client=client,
        raw=raw,
        detokenizer=detokenizer,
        manifest=manifest,
        g_prompts={},
        a_bodies={},
        e_arms=e_arms,
        q5_bodies={},
        prompt_tokens=prompt_tokens,
    )

    async def go() -> None:
        try:
            await _run_schedule(run)
        finally:
            await raw.aclose()

    asyncio.run(go())
    manifest["requests_written"] = run.written
    T0115A._write_json(out / "manifest.json", manifest)

    report = build_report(out)
    T0115A._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _say(f"outcome: {report['outcome']}")
    _say(f"written to {out}")
    if manifest.get("stopped") or not report["complete"]:
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
