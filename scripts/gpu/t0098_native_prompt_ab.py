"""T-0098 -- native tool turns under the live prompt: an offline, session-free prompt A/B on dev. ``/launch`` ONLY.

    uv run python scripts/gpu/t0098_native_prompt_ab.py --base-url http://127.0.0.1:8001/v1 --split dev

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the report half (``classify``, ``build_report``,
``render_markdown``) reads files only, and the engine is reached only from ``main``.

Refusals (exit 2), all **before any socket** (criterion 1):

1. a ``--base-url`` that is not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (T-0093's ``check_loopback``);
2. a dirty tree (the dirty paths are named);
3. an existing, non-empty ``--out-dir``;
4. a ``--split`` other than ``dev`` (the test split is never read);
5. a ``tests/eval/toolcall/cases.jsonl`` whose sha256 is not ``CASES_SHA256`` (or a dev split of the wrong size);
6. a tool kit that ``cli.load_tool_kit`` cannot load, whose repair is off, or whose tool set is not ``bbfc53ccdd80``;
7. an A0 system text (``PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools).system_message()``) whose sha256 is not
   ``A0_SHA256``: the live prompt drifted since 4b0bd92, so the run would not measure the finding;
8. arm prompts whose request bodies differ anywhere but ``messages[0]`` (criterion 2; a harness bug, never expected).

What it runs (docs/tasks/T-0098.md criteria 2-4, H-T98a):

- **Cases** (``load_cases``), 53, in this order: the dev split's 40 tool cases, its 8 no-tool cases (file order), then
  T-0093's 5 ``PLAIN_QUESTIONS``. T-0093's own ``load_cases`` is reused by import; its Q12 cases are dropped.
- **Arms** (``arm_prompts``): one pass-1 ``Prompt`` per arm, built as a fresh ``Session`` builds it for one typed turn
  (``PromptBuilder.prefix`` with no history, the case's frame, no assists; ``full`` with the case's text, no memory hits,
  so a text-only case carries the ``[screen]`` note). That prompt is A0; every other arm is A0 with ``messages[0]``
  replaced (no module global is patched). A0 live; A1 ``twopass.NATIVE_SYSTEM_PROMPT``; D1 A0 without
  ``UI_INSTRUCTIONS``; D2 A0 without ``SPOKEN_STYLE``; C2 A0 with ``"\\n\\n"`` before ``NATIVE_TOOL_INSTRUCTIONS``; C1
  A0 with H-T98a's sentence after "...then call exactly one of the declared tools.".
- **Design.** 5 repeats; within a repeat the cases in order; per (case, repeat) the six arms from offset
  ``(case_index + repeat) mod 6`` of ``ARMS``. Concurrency 1, one boot, pass 1 only (``VllmClient.generate`` in id
  mode with the kit's detokeniser, ``LlmOptions()`` as the session sends it): no prefill, no repair, no seed.
- **Record** (criterion 4): ``trials.jsonl``, one line per request, in request order: the case, the arm, the repeat,
  ``order`` (the arm's slot 0-5 in its block), the sha256 of the ``messages[0]`` text actually sent, every commit
  (``segment``, ``text``, ``thought_open``, ``canvas_idx``, ``stop_id``), every ``token_ids`` id the engine streamed
  (read off the wire by a tee on the transport, stop id included), ``error``, ``wall_ms`` and ``first_canvas_ms``.
  ``manifest.json`` holds the regime and the six arm texts with their shas.
- **Stopping.** After ``STOP_AFTER_ERRORS`` consecutive errored requests the run stops (exit 1); the manifest says why.

Output (``--out-dir``, default ``logs/t0098-prompt-ab-<sha7>/``): ``manifest.json``, ``trials.jsonl``, ``report.json``
(``build_report(run_dir)``) and ``report.md``. Exit codes: 0 the run completed (whatever the winner); 1 the run stopped
or the engine is not there; 2 refused.

**Two arm tables (T-0098c criterion 1).** ``TABLES`` is keyed by task; ``--table`` picks one, default ``T-0098``, under
which every name above behaves as at 3ee1636. The manifest records ``table`` and ``rule``; ``build_report`` reads the
table from the manifest (no key = T-0098). ``--table T-0098c`` (H-T98c) runs the UI-block A/B:

    uv run python scripts/gpu/t0098_native_prompt_ab.py --base-url http://127.0.0.1:8001/v1 --split dev --table T-0098c

- **Arms** A0 D1 D3 D4 C3 C5 C4 (T-0098c *Arms*): A0 live; D1 without ``UI_INSTRUCTIONS`` (U); D3 with only U's bullets
  (from ``- Between the fences``); D4 with only U's first half; C3 the tool paragraph before U; C5 U with
  ``"what you are showing."`` -> ``"what you are showing or doing."`` (N8: the wrap is kept); C4 U with H-T98c's line
  after ``"an empty or decorative surface is worse than none.\\n"``. C5 and C4 apply exactly one replacement (an anchor
  occurring 0 or >= 2 times is refused, never a double edit) and C4's U is at most ``UI_MAX_CHARS`` code points.
- **Cases**, 60: T-0098's 53 (indices unchanged), then the 7 Q12 cases of ``tests/eval/surface_placement_cases.jsonl``
  (pinned ``Q12_SHA256``, 4 positive + 3 negative) in file order, through T-0093's own loader. A Q12 case has no frame
  (the ``[screen]`` note); ``flight_booking_with_memory`` carries its captions as memory hits exactly as T-0093's
  ``_memory_hits`` builds them, with no ``now_ms`` (N10: the session passes none).
- **Rotation** from offset ``(case_index + repeat) mod 7`` of the table order; 2100 requests.
- **Surface shown** (criterion 5, ``ui_reading``): a fresh ``UiGate()`` per trial gates every ``ui`` commit in commit
  order as ``session._gate_surface`` does (``final=False``, ``turn_id`` = the trial's request id); shown = any call
  returned a surface. Placement failure = a ``ui`` commit flagged ``thought_open`` or a thought holding ``UI_OPEN``.
- **Rule** (criterion 7): P1-P7 as T-0098 over the same 53-case denominators, P8-P11 on the UI channel, readings R1,
  R2c (label only) and R3; winner C3, then C5, then C4, only when R1 and R3 hold on a complete run.

**A third table (T-0103, H-T103): the C3 confirmation.** ``--table T-0103`` runs A0 against C3 only (T-0098c's texts,
both pinned: ``A0_SHA256`` and ``C3_SHA256``), on T-0098c's 60 cases, with a per-case repeats map on the table: 40 on
the four Q12 positive cases, 5 on the other 56 (880 requests; ``schedule``). Repeats are outermost; a repeat runs the
cases scheduled for it in index order; per (case, repeat) the two arms from offset ``(case_index + repeat) mod 2``.
It also refuses a C3 that is not exactly ``"\\n".join([P, CT, SS, N, U])`` with sha ``C3_SHA256``, and a repeats map
other than 40/5. The rule is T-0098c's with P8, P10, P11 and R3 scaled by reading S (``_ui_conditions_103``), readings
R1 and R3 only, and one ``outcome``: "C3 confirmed", "C3 not confirmed: fails P...", or "inconclusive: <reason>";
``winner`` is C3 only when confirmed. The T-0098 and T-0098c paths are unchanged.

**A fourth table (T-0108a, H-T108a): the required-fields A/B.** ``--table T-0108a`` runs A0 against K1 and K2, two
UI blocks that name the interactive components' required fields (T-0108a *Arms*): each is U with E1 (one sentence
made shorter), then K1's or K2's anchor replaced, every edit an exact single-occurrence replacement (an anchor found
0 or >= 2 times is refused) and every U' at most ``UI_MAX_CHARS`` code points; all three texts are pinned
(``A0_SHA256``, ``K1_SHA256``, ``K2_SHA256``). T-0098c's 60 cases, T-0103's 40/5 repeats map (1320 requests), arms
from offset ``(case_index + repeat) mod 3``. Each Q12 positive trial is also read for a **complete card**
(``ui_reading``: the last gated call returned a surface and nothing was pruned on it, read through the gate's
``_repair``). The rule (``_conditions_108a``, ``_report_108a``): Q1/Q2 on complete cards, U1/U2 on last-OK and shown,
P3-P7 and P9-P11 as T-0103, P1/P2 dropped (D16), controls R3 and R4; one ``outcome``: "K<i> selected", "no change:
..." or "inconclusive: <reason>". The manifest also records each arm's U' with its engine token count (root
``/tokenize``, before the first request) and the sha256 of the three gate files. The earlier tables are unchanged:
the T-0103-specific branches are table attributes (``conditions``, ``report``, ``positive_repeats``,
``thresholds``, ``reading``) whose defaults keep the older output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import UI_OPEN
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.conductor.ui_gate import UiGate, _TooDeep
from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.eval.cd import twopass
from local_astra.toolcall import native, score
from local_astra.toolcall import tools as toolset
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals

if TYPE_CHECKING:
    import httpx

    from local_astra.conductor.llm_client import VllmClient

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
CASES_PATH = SET_DIR / "cases.jsonl"
T0093_PATH = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"


def _load_t0093() -> ModuleType:
    """T-0093's harness, imported read-only by path (it is not a package). Importing it opens no socket."""
    name = "_t0093_toolcall_live_for_t0098"
    found = sys.modules.get(name)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(name, T0093_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0093_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look themselves up here while they are built
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_T93 = _load_t0093()
#: T-0093 J10, reused unchanged
PLAIN_QUESTIONS: tuple[tuple[str, str], ...] = _T93.PLAIN_QUESTIONS

#: sha256 of ``tests/eval/toolcall/cases.jsonl`` (as T-0087 pinned it; criterion 1)
CASES_SHA256 = "460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85"
#: The live system text at 4b0bd92 (T-0098 *Evidence*): 2744 characters, identical on every case.
A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
TOOLS_SHA12 = "bbfc53ccdd80"
SPLIT = "dev"

#: The arm table of H-T98a, in its order (the rotation indexes it).
ARMS: tuple[str, ...] = ("A0", "A1", "D1", "D2", "C2", "C1")
ROLES: dict[str, str] = {
    "A0": "control: the live system message",
    "A1": "positive control: twopass.NATIVE_SYSTEM_PROMPT (the eval prompt)",
    "D1": "diagnostic, never selected: A0 without UI_INSTRUCTIONS",
    "D2": "diagnostic, never selected: A0 without SPOKEN_STYLE",
    "C2": 'candidate 1: A0 with "\\n\\n" (not "\\n") before NATIVE_TOOL_INSTRUCTIONS',
    "C1": "candidate 2: A0 with H-T98a's sentence added to the instructions",
}
#: Winner order (H-T98a (c)): C2 if it passes, else C1. D1 and D2 are never candidates (N2).
CANDIDATES: tuple[str, ...] = ("C2", "C1")
PS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
REPEATS = 5

#: H-T98a (b), verbatim: the sentence C1 adds, after ``C1_ANCHOR`` in the instructions paragraph.
C1_ANCHOR = "then call exactly one of the declared tools."
C1_SENTENCE = (
    "Say that sentence in your reply, after you have finished thinking: what you write while thinking is never "
    "heard."
)

#: Criterion 5: an arm with more than 2 % errored trials is void.
VOID_ERROR_RATE = Fraction(2, 100)
#: Criterion 6: the reproduction readings (labels, not gates).
R1_MAX = Fraction(3, 10)
R2_MIN = Fraction(7, 10)
#: Criterion 7 (H-T98a (c)): the decision rule's bars, exact.
P1_SPOKEN_MIN = Fraction(8, 10)
P1_MAJORITY_CASES = 8
P2_MARGIN = Fraction(5, 100)
P3_POOLED_MARGIN = Fraction(2, 100)
P3_TYPE_MARGIN = Fraction(4, 100)
P4_VALID_MARGIN = Fraction(2, 100)
P4_EXACT_MARGIN = Fraction(3, 100)
P5_MAX = 5
P5_OVER_A0 = 2
P7_OVER_A0 = 4

TOOL_TAG = "<|tool_call>"
TOOL_NAMES: tuple[str, ...] = ("google_search", "click_at", "type_at", "scroll", "press_key")
#: P2's pool
CSK = ("click_at", "scroll", "press_key")
NO_CHANGE = "no change"

#: Criterion 4's stop rule: this many consecutive errored requests stop the run (the engine is gone).
STOP_AFTER_ERRORS = 10

# --- T-0098c (H-T98c) -------------------------------------------------------------------------------------------
T0098 = "T-0098"
T0098C = "T-0098c"
#: J1's Q12 file and its pin (T-0093's, restated here so a drift is refused by this harness too)
Q12_PATH = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
Q12_SHA256 = "e152810b0c779a7eeeb3a5d84947d173cc8415400f36fa1cf7f007749158d6d8"
Q12_POSITIVE_N, Q12_NEGATIVE_N = 4, 3
#: The arm table of H-T98c, in its order (the rotation indexes it).
ARMS_C: tuple[str, ...] = ("A0", "D1", "D3", "D4", "C3", "C5", "C4")
ROLES_C: dict[str, str] = {
    "A0": "control: the live system message",
    "D1": "positive diagnostic, never selected: A0 without UI_INSTRUCTIONS",
    "D3": "diagnostic, never selected: A0 with only UI_INSTRUCTIONS' bullets (first half removed)",
    "D4": "diagnostic, never selected: A0 with only UI_INSTRUCTIONS' first half (bullets removed)",
    "C3": "candidate 1: NATIVE_TOOL_INSTRUCTIONS before UI_INSTRUCTIONS (wording unchanged)",
    "C5": 'candidate 2: UI_INSTRUCTIONS\' last bullet ends "what you are showing or doing."',
    "C4": "candidate 3: UI_INSTRUCTIONS with H-T98c's added line after the speech-only sentence",
}
#: Winner order (H-T98c (d)): C3, then C5, then C4. D1, D3, D4 are never candidates (N7).
CANDIDATES_C: tuple[str, ...] = ("C3", "C5", "C4")
PS_C: tuple[str, ...] = (*PS, "P8", "P9", "P10", "P11")
#: Where UI_INSTRUCTIONS' bullets start (D3 keeps U[i:], D4 keeps U[:i-1]).
U_BULLETS_ANCHOR = "- Between the fences"
#: C5 (N8: the wrap-preserving spelling of H-T98c's edit).
C5_ANCHOR = "what you are showing."
C5_REPLACEMENT = "what you are showing or doing."
#: C4: H-T98c (a)'s line, verbatim, after the anchor.
C4_ANCHOR = "an empty or decorative surface is worse than none.\n"
C4_ADDED = (
    "Every reply says at least one sentence aloud, after you have finished thinking, whatever else\n"
    "it contains.\n"
)
#: Criterion 2: C4's UI block may not exceed this many code points (N9: it is 1977).
UI_MAX_CHARS = 2000
#: Criterion 6's readings for T-0098c.
R2C_MIN = Fraction(7, 10)
R3_MIN_SHOWN = 14
#: Criterion 7's UI-channel bars, exact.
P8_TOTAL_MARGIN = 2
P8_CASE_MIN = 3
P9_NO_BELONG_OVER_A0 = 1
P9_TOOL_OVER_A0 = 2
P10_OVER_A0 = 1
P11_OVER_A0 = 2

# --- T-0103 (H-T103) --------------------------------------------------------------------------------------------
T0103 = "T-0103"
#: T-0103 *Arms*: C3's text, ``"\n".join([P, CT, SS, N, U])`` (2744 characters, A0's pieces in another order).
C3_SHA256 = "72c3b92bcff2e5924901e42475342826ffc38efefa116f66138f8aa43bad9308"
ARMS_103: tuple[str, ...] = ("A0", "C3")
ROLES_103: dict[str, str] = {
    "A0": "control: the live system message",
    "C3": "the candidate being confirmed: NATIVE_TOOL_INSTRUCTIONS before UI_INSTRUCTIONS (T-0098c's C3)",
}
CANDIDATES_103: tuple[str, ...] = ("C3",)
#: T-0103 *Design*: repeats on each Q12 positive case (every other case keeps ``REPEATS``).
POSITIVE_REPEATS_103 = 40
#: T-0103's per-arm denominators of the scaled conditions: 160 Q12 positive trials, 40 per positive case, 175 Q12.
Q12_POSITIVE_TRIALS_103 = Q12_POSITIVE_N * POSITIVE_REPEATS_103
Q12_CASE_TRIALS_103 = POSITIVE_REPEATS_103
Q12_TRIALS_103 = Q12_POSITIVE_TRIALS_103 + Q12_NEGATIVE_N * REPEATS
#: Criterion 5, reading S (H-T103 (c)): T-0098c's margins as proportions of their denominators, exact.
P8_TOTAL_SHARE_103 = Fraction(1, 10)
P8_CASE_SHARE_103 = Fraction(3, 5)
P10_SHARE_103 = Fraction(1, 20)
P11_SHARE_103 = Fraction(2, 35)
R3_SHARE_103 = Fraction(7, 10)
P8_TOTAL_MARGIN_103 = P8_TOTAL_SHARE_103 * Q12_POSITIVE_TRIALS_103  # 16 of 160
P8_CASE_MIN_103 = P8_CASE_SHARE_103 * Q12_CASE_TRIALS_103  # 24 of 40
P10_OVER_A0_103 = P10_SHARE_103 * Q12_POSITIVE_TRIALS_103  # 8 of 160
P11_OVER_A0_103 = P11_SHARE_103 * Q12_TRIALS_103  # 10 of 175
R3_MIN_SHOWN_103 = R3_SHARE_103 * Q12_POSITIVE_TRIALS_103  # 112 of 160
CONFIRMED = "C3 confirmed"

# --- T-0108a (H-T108a) ------------------------------------------------------------------------------------------
T0108A = "T-0108a"
#: T-0108a *Arms*: K1's and K2's system texts (2873 / 2865 characters; A0 keeps ``A0_SHA256``).
K1_SHA256 = "22a2ba7e9a022268fb41f847fa3cfad1e0a123f5c323cb120c63d7526fac0702"
K2_SHA256 = "23a34c3656547eebaf2f89c65cc6a0a04367f8dcd93924d3bb4b8be79ced687c"
ARMS_108A: tuple[str, ...] = ("A0", "K1", "K2")
ROLES_108A: dict[str, str] = {
    "A0": "control: the live system message",
    "K1": "candidate 1: UI_INSTRUCTIONS with E1, then F1 after the flashcard sentence",
    "K2": "candidate 2: UI_INSTRUCTIONS with E1, then the no-table sentence replaced by a shorter one, F1 and F2",
}
#: Selection (H-T108a (c)): the higher complete count; a tie goes to the first, K1.
CANDIDATES_108A: tuple[str, ...] = ("K1", "K2")
#: Criterion 5's conditions per candidate (P1, P2 and R1 are dropped per D16).
PS_108A: tuple[str, ...] = (
    "Q1",
    "Q2",
    "U1",
    "U2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P9",
    "P10",
    "P11",
)
#: E1, the shared edit (makes room)
E1_ANCHOR = (
    "Copy that catalogId exactly, use the same surfaceId in every message, give the tree a component\n"
    'with id "root", and define every id you reference in children in the same fence.'
)
E1_REPLACEMENT = (
    'Copy that catalogId exactly, keep one surfaceId, give one component id "root" and define every\n'
    "id you reference in the same fence."
)
#: F1 and F2, verbatim
F1 = (
    'Required: TextField label; DateTimeInput value ""; ChoicePicker options [{"label":"A","value":"a"}]\n'
    'and value []; Button child (a Text id) and action {"event":{"name":"go"}}.\n'
)
F2 = "Text variant: h1-h5, caption or body; children are ids, never objects.\n"
#: K1: F1 after this anchor
K1_ANCHOR = "flashcard is a Card.\n"
#: K2: this anchor replaced by ``K2_LEAD + F1 + F2``
K2_ANCHOR = (
    "There is no table, chart, maths or flashcard component: a table is a Column of Rows of Text, a\n"
    "flashcard is a Card.\n"
)
K2_LEAD = "A table is a Column of Rows of Text.\n"
#: Criterion 5, exact (denominators: 160 Q12 positive trials, 40 per positive case)
Q1_SHARE_108A = Fraction(3, 20)
Q1_MARGIN_108A = Q1_SHARE_108A * Q12_POSITIVE_TRIALS_103  # 24 of 160
#: a positive case is protected for Q2 / U1 / U2 iff A0 >= 3/5 of its 40 (24), as T-0103's P8
PROTECT_MIN_108A = P8_CASE_MIN_103  # 24 of 40
#: U1 / U2: K >= A0 - 1/10 of 160 (U2 is T-0103's P8)
U_MARGIN_108A = P8_TOTAL_MARGIN_103  # 16 of 160
#: R4 (the degradation reproduced): A0 complete <= 3/5 of 160
R4_SHARE_108A = Fraction(3, 5)
R4_MAX_COMPLETE_108A = R4_SHARE_108A * Q12_POSITIVE_TRIALS_103  # 96 of 160
#: The files that decide a complete card (criterion 8: the eval row reproduces only at the same shas)
GATE_FILES: tuple[str, ...] = (
    "local_astra/conductor/ui_gate.py",
    "local_astra/ui/bracket_repair.py",
    "local_astra/ui/a2ui_schema.py",
)
SALVAGE_DROPPED = "(dropped by salvage)"

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _say(msg: str) -> None:
    print(f"t0098_ab: {msg}", file=sys.stderr, flush=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =====================================================================================================
# The cases and the arms (pure; reads the pinned files only)
# =====================================================================================================


def load_cases(*, table: str = T0098) -> list[Any]:
    """The run's cases in run order (``TABLES[table].cases``). Under T-0098, the 53: dev tool, dev no-tool (file
    order), then the 5 plain questions; T-0093's ``Case`` objects (``case_id``, ``kind``, ``text``, ``gt``,
    ``screen``, ``screen_size``). ``ValueError`` on a cases file that is not ``CASES_SHA256``."""
    return TABLES[table].cases()


def _load_cases_98() -> list[Any]:
    got = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    if got != CASES_SHA256:
        raise ValueError(f"{CASES_PATH} has sha256 {got}, not the pinned {CASES_SHA256}")
    return [c for c in _T93.load_cases() if c.kind in ("tool", "no_tool", "plain")]


def _load_cases_c() -> list[Any]:
    """T-0098c criterion 3: T-0098's 53 (indices unchanged), then the 7 Q12 cases in file order (T-0093's loader,
    kind ``q12``, with ``expect_surface`` and ``memory_captions``). ``ValueError`` on a Q12 file that is not
    ``Q12_SHA256`` or that is not 4 positive + 3 negative."""
    base = _load_cases_98()
    got = hashlib.sha256(Q12_PATH.read_bytes()).hexdigest()
    if got != Q12_SHA256:
        raise ValueError(f"{Q12_PATH} has sha256 {got}, not the pinned {Q12_SHA256}")
    q12 = [c for c in _T93.load_cases() if c.kind == "q12"]
    positive = sum(1 for c in q12 if c.expect_surface)
    negative = len(q12) - positive
    if (positive, negative) != (Q12_POSITIVE_N, Q12_NEGATIVE_N):
        raise ValueError(
            f"{Q12_PATH.name} holds {positive} positive + {negative} negative Q12 cases, not "
            f"{Q12_POSITIVE_N} + {Q12_NEGATIVE_N}"
        )
    return [*base, *q12]


def arm_texts(tools: tuple[dict[str, Any], ...] | None, *, table: str = T0098) -> dict[str, str]:
    """The table's system texts, keyed by arm (``TABLES[table].texts``). ``ValueError`` when an arm cannot be built
    exactly as its table defines it."""
    return TABLES[table].texts(tools)


def _arm_texts_98(tools: tuple[dict[str, Any], ...] | None) -> dict[str, str]:
    """The six system texts. A0 is the builder's own; the edits are made from the pieces the builder joins, after
    checking that the builder still joins exactly those pieces (``ValueError`` otherwise)."""
    a0 = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=tools).system_message().text()
    p, ct, ss, ui, nti = (
        DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        pb.UI_INSTRUCTIONS,
        pb.NATIVE_TOOL_INSTRUCTIONS,
    )
    if a0 != "\n".join([p, ct, ss, ui, nti]):
        raise ValueError(
            "the builder's system message is no longer persona, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS "
            "and NATIVE_TOOL_INSTRUCTIONS joined by a newline; the arms cannot be built as H-T98a defines them"
        )
    if nti.count(C1_ANCHOR) != 1:
        raise ValueError(f"NATIVE_TOOL_INSTRUCTIONS does not hold {C1_ANCHOR!r} exactly once")
    c1_nti = nti.replace(C1_ANCHOR, C1_ANCHOR + " " + C1_SENTENCE, 1)
    texts = {
        "A0": a0,
        "A1": twopass.NATIVE_SYSTEM_PROMPT,
        "D1": "\n".join([p, ct, ss, nti]),
        "D2": "\n".join([p, ct, ui, nti]),
        "C2": "\n".join([p, ct, ss, ui]) + "\n\n" + nti,
        "C1": "\n".join([p, ct, ss, ui, c1_nti]),
    }
    if len(set(texts.values())) != len(ARMS):
        raise ValueError("two arms have the same system text")
    return texts


def _one_edit(arm: str, u: str, anchor: str, replacement: str, *, rule: str = "H-T98c") -> str:
    """``u`` with its one occurrence of ``anchor`` replaced; ``ValueError`` naming ``arm`` on 0 or >= 2 occurrences
    (a refusal, never a double edit or a silent no-op)."""
    found = u.count(anchor)
    if found != 1:
        raise ValueError(
            f"{arm}: UI_INSTRUCTIONS holds the anchor {anchor!r} {found} times, not exactly once; "
            f"{arm} cannot be built as {rule} defines it"
        )
    return u.replace(anchor, replacement, 1)


def _arm_texts_c(tools: tuple[dict[str, Any], ...] | None) -> dict[str, str]:
    """The seven T-0098c system texts (T-0098c *Arms*), from the pieces the builder joins, after checking that it
    still joins exactly those pieces (``ValueError`` otherwise, and on any anchor that is not there exactly once)."""
    a0 = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=tools).system_message().text()
    p, ct, ss, u, nti = (
        DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        pb.UI_INSTRUCTIONS,
        pb.NATIVE_TOOL_INSTRUCTIONS,
    )
    if a0 != "\n".join([p, ct, ss, u, nti]):
        raise ValueError(
            "the builder's system message is no longer persona, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS "
            "and NATIVE_TOOL_INSTRUCTIONS joined by a newline; the arms cannot be built as H-T98c defines them"
        )
    if u.count(U_BULLETS_ANCHOR) != 1:
        raise ValueError(
            f"D3/D4: UI_INSTRUCTIONS holds {U_BULLETS_ANCHOR!r} {u.count(U_BULLETS_ANCHOR)} times, not once"
        )
    i = u.index(U_BULLETS_ANCHOR)
    if i == 0 or u[i - 1] != "\n":
        raise ValueError(f"D3/D4: {U_BULLETS_ANCHOR!r} does not start a line of UI_INSTRUCTIONS")
    u_c5 = _one_edit("C5", u, C5_ANCHOR, C5_REPLACEMENT)
    u_c4 = _one_edit("C4", u, C4_ANCHOR, C4_ANCHOR + C4_ADDED)
    if len(u_c4) > UI_MAX_CHARS:
        raise ValueError(
            f"C4: its UI_INSTRUCTIONS would be {len(u_c4)} code points, over the {UI_MAX_CHARS} limit"
        )
    texts = {
        "A0": a0,
        "D1": "\n".join([p, ct, ss, nti]),
        "D3": "\n".join([p, ct, ss, u[i:], nti]),
        "D4": "\n".join([p, ct, ss, u[: i - 1], nti]),
        "C3": "\n".join([p, ct, ss, nti, u]),
        "C5": "\n".join([p, ct, ss, u_c5, nti]),
        "C4": "\n".join([p, ct, ss, u_c4, nti]),
    }
    if len(set(texts.values())) != len(ARMS_C):
        raise ValueError("two arms have the same system text")
    return texts


def _arm_texts_103(tools: tuple[dict[str, Any], ...] | None) -> dict[str, str]:
    """T-0103's two texts: T-0098c's A0 and C3, by T-0098c's own construction (``_arm_texts_c``)."""
    texts = _arm_texts_c(tools)
    return {arm: texts[arm] for arm in ARMS_103}


def _c3_problems_103(texts: Mapping[str, str]) -> list[str]:
    """Criterion 3 on what the table built: C3 is exactly ``"\\n".join([P, CT, SS, N, U])`` of the builder's pieces
    as they are now, with sha256 ``C3_SHA256``. The problems found, none when it holds."""
    c3 = texts.get("C3", "")
    want = "\n".join(
        [
            DEFAULT_SYSTEM_PROMPT,
            pb.CLOSE_THE_THOUGHT,
            pb.SPOKEN_STYLE,
            pb.NATIVE_TOOL_INSTRUCTIONS,
            pb.UI_INSTRUCTIONS,
        ]
    )
    problems: list[str] = []
    if c3 != want:
        problems.append(
            "C3: the text is not exactly persona, CLOSE_THE_THOUGHT, SPOKEN_STYLE, NATIVE_TOOL_INSTRUCTIONS and "
            "UI_INSTRUCTIONS joined by a newline"
        )
    if _sha(c3) != C3_SHA256:
        problems.append(f"C3: the text has sha256 {_sha(c3)}, not the pinned {C3_SHA256}")
    return problems


def _arm_texts_108a(tools: tuple[dict[str, Any], ...] | None) -> dict[str, str]:
    """T-0108a's three texts (T-0108a *Arms*): A0 the builder's own; K1 and K2 are A0 with U replaced by U', built
    from the pieces the builder joins after checking that it still joins exactly those pieces. U' is U with E1,
    then K1's anchor followed by F1 (K1) or K2's anchor replaced by ``K2_LEAD + F1 + F2`` (K2); every edit replaces
    exactly one occurrence. ``ValueError`` naming the arm on an anchor found 0 or >= 2 times or a U' over
    ``UI_MAX_CHARS`` code points."""
    a0 = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=tools).system_message().text()
    p, ct, ss, u, nti = (
        DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        pb.UI_INSTRUCTIONS,
        pb.NATIVE_TOOL_INSTRUCTIONS,
    )
    if a0 != "\n".join([p, ct, ss, u, nti]):
        raise ValueError(
            "the builder's system message is no longer persona, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS "
            "and NATIVE_TOOL_INSTRUCTIONS joined by a newline; the arms cannot be built as H-T108a defines them"
        )
    edits = {"K1": (K1_ANCHOR, K1_ANCHOR + F1), "K2": (K2_ANCHOR, K2_LEAD + F1 + F2)}
    texts = {"A0": a0}
    for arm, (anchor, replacement) in edits.items():
        u_e1 = _one_edit(arm, u, E1_ANCHOR, E1_REPLACEMENT, rule="H-T108a")
        u_k = _one_edit(arm, u_e1, anchor, replacement, rule="H-T108a")
        if len(u_k) > UI_MAX_CHARS:
            raise ValueError(
                f"{arm}: its UI_INSTRUCTIONS would be {len(u_k)} code points, over the {UI_MAX_CHARS} limit"
            )
        texts[arm] = "\n".join([p, ct, ss, u_k, nti])
    if len(set(texts.values())) != len(ARMS_108A):
        raise ValueError("two arms have the same system text")
    return texts


def _k_problems_108a(texts: Mapping[str, str]) -> list[str]:
    """Criterion 3 on what the table built: K1 and K2 have the pinned sha256 of T-0108a *Arms*. The problems found,
    none when they hold."""
    problems: list[str] = []
    for arm, pin in (("K1", K1_SHA256), ("K2", K2_SHA256)):
        got = _sha(texts.get(arm, ""))
        if got != pin:
            problems.append(
                f"{arm}: the text has sha256 {got}, not the pinned {pin} (T-0108a *Arms*)"
            )
    return problems


def _u_primes(texts: Mapping[str, str]) -> dict[str, str]:
    """Each arm's UI block U' as it sits in its text: what lies between ``P, CT, SS`` and ``N`` (T-0108a's arms
    differ only there; ``ValueError`` when a text does not have that shape)."""
    head = "\n".join([DEFAULT_SYSTEM_PROMPT, pb.CLOSE_THE_THOUGHT, pb.SPOKEN_STYLE]) + "\n"
    tail = "\n" + pb.NATIVE_TOOL_INSTRUCTIONS
    out: dict[str, str] = {}
    for arm, text in texts.items():
        if not (text.startswith(head) and text.endswith(tail)) or len(text) < len(head) + len(tail):
            raise ValueError(f"{arm}: the text is not P, CT, SS, U', N joined by a newline")
        out[arm] = text[len(head) : len(text) - len(tail)]
    return out


def _frames(case: Any) -> tuple[Frame, ...]:
    """The case's screen as the turn's one frame, built as T-0093 built it (ts 0, the fixture's size, phash =
    sha256(jpeg)[:16]); none for a text-only case."""
    if case.screen is None or case.screen_size is None:
        return ()
    jpeg = (SET_DIR / case.screen).read_bytes()
    return (
        Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=int(case.screen_size[0]),
            height=int(case.screen_size[1]),
            jpeg=jpeg,
            phash=hashlib.sha256(jpeg).hexdigest()[:16],
        ),
    )


def arm_prompts(
    case: Any, kit: Any, *, thinking: bool = True, table: str = T0098
) -> dict[str, Prompt]:
    """The table's pass-1 prompts of one case, keyed by arm. A0 is what a fresh ``Session`` with ``kit`` builds for
    this typed turn (criterion 2); every other arm is A0 with ``messages[0]`` replaced. A case with memory captions
    (T-0098c's ``flight_booking_with_memory``) carries them as the hits T-0093's ``_CaseMemory`` serves, and, as the
    session, no ``now_ms`` (N10)."""
    tools = kit.tools
    builder = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, thinking=thinking, tools=tools)
    prefix = builder.prefix([], None, AssistFlags(), None, frames=_frames(case))
    captions = getattr(case, "memory_captions", ())
    hits = tuple(_T93._memory_hits(case)) if captions else ()
    a0 = builder.full(prefix, case.text, None, hits)
    texts = arm_texts(tools, table=table)
    out: dict[str, Prompt] = {"A0": a0}
    for arm in TABLES[table].arms[1:]:
        system = Message(role="system", content=(TextPart(text=texts[arm]),))
        out[arm] = a0.model_copy(update={"messages": (system, *a0.messages[1:])})
    return out


def _without_system(body: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(body))
    out["messages"] = out["messages"][1:]
    (out.get("vllm_xargs") or {}).pop("astra_request_id", None)
    return out


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_has_key(v, key) for v in value)
    return False


def check_fidelity(bodies: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Criterion 2 on one case's six request bodies: equal to A0's outside ``messages[0]``, different inside it; no
    ``seed``; no prefill (``max_tokens == 1``); no continuation. The problems found, none when it holds."""
    problems: list[str] = []
    ref = _without_system(bodies["A0"])
    for arm, body in bodies.items():
        if _has_key(body, "seed"):
            problems.append(f"{arm}: the body carries a seed")
        if body.get("max_tokens") == 1 or body.get("continue_final_message"):
            problems.append(f"{arm}: the body is not pass 1")
        if arm != "A0":
            if _without_system(body) != ref:
                problems.append(f"{arm}: the body differs from A0's outside messages[0]")
            if body["messages"][0] == bodies["A0"]["messages"][0]:
                problems.append(f"{arm}: messages[0] is A0's")
    return problems


# =====================================================================================================
# The arm tables (T-0098c criterion 1, N5): one harness, keyed by task
# =====================================================================================================


@dataclass(frozen=True)
class ArmTable:
    """One task's A/B: its arms (table order, which the rotation indexes), their roles, the candidates in winner
    order, the arm-text builder, the A0 pin, the case loader and its sources, the P-conditions and the rule id.

    The fields with defaults (T-0103) default to what T-0098 does: no UI channel, request ids ``t0098-...``, out-dir
    stem ``t0098-prompt-ab``, ``REPEATS`` on every case (``repeats`` is ``None``) and no check on the texts beyond
    the A0 pin."""

    task: str
    rule: str
    arms: tuple[str, ...]
    roles: Mapping[str, str]
    candidates: tuple[str, ...]
    a0_sha256: str
    texts: Callable[[tuple[dict[str, Any], ...] | None], dict[str, str]]
    cases: Callable[[], list[Any]]
    case_sources: tuple[str, ...]
    ps: tuple[str, ...]
    #: the Q12 cases and the UI-channel scoring (T-0098c criteria 5-6)
    ui_channel: bool = False
    request_prefix: str = "t0098"
    out_stem: str = "t0098-prompt-ab"
    #: case id -> repeats; ``None``: ``REPEATS`` on every case
    repeats: Mapping[str, int] | None = None
    #: problems with the texts the builder returned (refused in ``main``, before any socket)
    text_problems: Callable[[Mapping[str, str]], list[str]] | None = None
    #: the repeats a Q12 positive case must have when ``repeats`` is set (``main`` refuses another map)
    positive_repeats: int = POSITIVE_REPEATS_103
    #: the candidate's UI-channel conditions against A0 (``None``: T-0098c's ``_ui_conditions`` on a UI table)
    conditions: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, bool]] | None = None
    #: the table's own readings, outcome and extras (``None``: the T-0098 / T-0098c report)
    report: Callable[..., dict[str, Any]] | None = None
    #: the manifest's ``thresholds`` and ``reading`` when ``repeats`` is set
    thresholds: Mapping[str, str] | None = None
    reading: str | None = None
    #: T-0108a: the complete-card keys per arm, and each U' (engine tokens) and the gate-file shas in the manifest
    complete_cards: bool = False

    def repeats_of(self, case_id: str) -> int:
        return REPEATS if self.repeats is None else int(self.repeats.get(case_id, REPEATS))


TABLES: dict[str, ArmTable] = {
    T0098: ArmTable(
        task=T0098,
        rule="H-T98a",
        arms=ARMS,
        roles=ROLES,
        candidates=CANDIDATES,
        a0_sha256=A0_SHA256,
        texts=_arm_texts_98,
        cases=_load_cases_98,
        case_sources=("tests/eval/toolcall/cases.jsonl (dev)", "T-0093 J10 PLAIN_QUESTIONS"),
        ps=PS,
    ),
    T0098C: ArmTable(
        task=T0098C,
        rule="H-T98c",
        arms=ARMS_C,
        roles=ROLES_C,
        candidates=CANDIDATES_C,
        a0_sha256=A0_SHA256,
        texts=_arm_texts_c,
        cases=_load_cases_c,
        case_sources=(
            "tests/eval/toolcall/cases.jsonl (dev)",
            "T-0093 J10 PLAIN_QUESTIONS",
            "tests/eval/surface_placement_cases.jsonl (J1, Q12)",
        ),
        ps=PS_C,
        ui_channel=True,
        request_prefix="t0098c",
        out_stem="t0098c-prompt-ab",
    ),
}
# TABLES[T0103] is registered below ``_case_table_c``: its repeats map is read from the pinned case files.


def arm_order(case_index: int, repeat: int, table: str = T0098) -> tuple[str, ...]:
    """Criterion 3: the (case, repeat) block's arm order, from offset ``(case_index + repeat) mod len(arms)`` of the
    table order."""
    arms = TABLES[table].arms
    offset = (case_index + repeat) % len(arms)
    return tuple(arms[(offset + slot) % len(arms)] for slot in range(len(arms)))


@dataclass(frozen=True)
class Slot:
    """One request of the run's schedule."""

    case_index: int
    case_id: str
    repeat: int
    #: the arm's slot in its (case, repeat) block
    order: int
    arm: str
    request_id: str


def schedule(table: str = T0098, cases: Sequence[Any] | None = None) -> list[Slot]:
    """The run's requests in request order (T-0098 criterion 3, T-0103 *Design*): repeats outermost; within a repeat
    the cases scheduled for it (``repeat < TABLES[table].repeats_of(case)``) in index order; per (case, repeat) the
    arms in ``arm_order``. ``cases`` defaults to ``load_cases(table=table)``."""
    t = TABLES[table]
    run_cases = list(load_cases(table=table) if cases is None else cases)
    per_case = [t.repeats_of(str(c.case_id)) for c in run_cases]
    out: list[Slot] = []
    for repeat in range(max(per_case, default=0)):
        for i, case in enumerate(run_cases):
            if repeat >= per_case[i]:
                continue
            for slot, arm in enumerate(arm_order(i, repeat, table)):
                rid = f"{t.request_prefix}-r{repeat}-{case.case_id}-{arm}"
                out.append(Slot(i, str(case.case_id), repeat, slot, arm, rid))
    return out


def ui_reading(
    commits: Sequence[Mapping[str, Any]], *, request_id: str, cards: bool = True
) -> dict[str, Any]:
    """T-0098c criterion 5 on one trial's commits, session-free and faithful (N6): a fresh ``UiGate()`` gates every
    ``ui`` commit in commit order exactly as ``session._gate_surface`` calls it (``final=False``, the trial's request
    id as ``turn_id``, the commit's ``canvas_idx``), whatever the commit's ``thought_open``.

    - ``shown``: at least one call returned a surface (never retracted by a later failing canvas);
    - ``placement_failure``: a ``ui`` commit flagged ``thought_open``, or a ``thought`` commit holding ``UI_OPEN``
      (the T-0055 defect);
    - ``every_ui_valid``: every call returned a surface (``None`` when nothing was gated); reported, not gated;
    - ``gated`` / ``emitted`` / ``failed``: the trial's ``ui.gate`` counts (calls; calls with a surface; without);
    - T-0108a criterion 4 (``_card_reading``; skipped with ``cards=False``, as the earlier tables read nothing of
      it): ``last_ok``, ``pruned_last``, ``complete``, and, reported only, ``pruned_kinds``, ``forward_refs`` and
      ``as_written_clean``.
    """
    gate = UiGate()
    gated = emitted = 0
    last: Any = None
    last_raw: str | None = None
    for c in commits:
        if c.get("segment") != "ui":
            continue
        raw = str(c.get("text", ""))
        result = gate.gate(
            raw,
            turn_id=request_id,
            canvas_idx=int(c.get("canvas_idx") or 0),
            final=False,
        )
        last, last_raw = result, raw
        gated += 1
        emitted += int(result.surface is not None)
    placement = any(
        c.get("segment") == "ui" and bool(c.get("thought_open")) for c in commits
    ) or any(c.get("segment") == "thought" and UI_OPEN in str(c.get("text", "")) for c in commits)
    return {
        "shown": emitted > 0,
        "placement_failure": placement,
        "every_ui_valid": (emitted == gated) if gated else None,
        "gated": gated,
        "emitted": emitted,
        "failed": gated - emitted,
        **(_card_reading(gate, last, last_raw) if cards else {}),
    }


def _card_reading(gate: UiGate, last: Any, raw: str | None) -> dict[str, Any]:
    """T-0108a criterion 4 (H-T108a (a)) on the trial's last gated call, with the same per-trial gate:

    - ``last_ok``: the last call returned a surface (no ``ui`` commit: false);
    - ``pruned_last``: the ids (``#<index>`` for an id-less one) that are schema-invalid among the components of
      ``gate._repair(raw).messages`` (after salvage), plus ``_repair(raw).dropped`` (what salvage threw away);
      ``_repair`` is read, never changed;
    - ``complete``: ``last_ok`` and nothing in ``pruned_last``;
    - reported, not counted: ``pruned_kinds`` (the component kinds of ``pruned_last``), ``forward_refs`` (ids the
      last canvas references and does not define) and ``as_written_clean`` (complete, and the canvas parsed as
      written: the fixer made no edit).
    """
    out: dict[str, Any] = {
        "last_ok": False,
        "pruned_last": [],
        "complete": False,
        "pruned_kinds": [],
        "forward_refs": [],
        "as_written_clean": False,
    }
    if last is None or raw is None:
        return out
    out["last_ok"] = last.surface is not None
    try:
        rep = gate._repair(raw)
    except (_TooDeep, RecursionError):
        # the gate answered this canvas with no surface (T-0108 criterion 9): nothing can be read from it
        return out
    v = gate.validator
    components = list(v.iter_components(rep.messages))
    pruned = set(v.invalid_component_ids(components)) | set(rep.dropped)
    kind_of: dict[str, str] = {}
    for i, c in enumerate(components):
        if isinstance(c, Mapping):
            cid = c.get("id")
            label = cid if isinstance(cid, str) and cid else f"#{i}"
            kind_of.setdefault(label, str(c.get("component")))
    defined = {c.get("id") for c in components if isinstance(c, Mapping)}
    referenced = {ref for c in components if isinstance(c, Mapping) for ref in v._children(c)}
    out["pruned_last"] = sorted(pruned)
    out["complete"] = bool(out["last_ok"]) and not pruned
    out["pruned_kinds"] = sorted({kind_of.get(cid, SALVAGE_DROPPED) for cid in pruned})
    out["forward_refs"] = sorted(str(r) for r in referenced - defined)
    out["as_written_clean"] = bool(out["complete"]) and rep.edits == 0
    return out


# =====================================================================================================
# One trial's reading (criteria 5 and 11)
# =====================================================================================================


def classify(commits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Criterion 5 on one trial's commits (J8, I15):

    - ``routed``: a ``tool`` commit outside an open thought;
    - ``spoken_prefix``: ``drop_special_literals`` of the ``answer`` commits with ``thought_open`` false, joined (N4; the session's
      ``_AnswerCleaner.spoken_prefix``); ``spoken``: routed **and** that prefix, stripped, is not empty;
    - ``thought_open``: the last ``thought`` commit carries ``thought_open`` (the thought never closed);
    - ``anomaly``: a spoken prefix holding ``call:`` or a special literal;
    - ``call``: the routed call parsed by ``native.parse_call`` (``None`` when it does not parse or is unrouted).
    """
    tools = [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]
    routed = bool(tools)
    prefix = drop_special_literals(
        "".join(
            str(c.get("text", ""))
            for c in commits
            if c.get("segment") == "answer" and not c.get("thought_open")
        )
    )
    stripped = prefix.strip()
    spoken = routed and stripped != ""
    thoughts = [c for c in commits if c.get("segment") == "thought"]
    thought_open = bool(thoughts) and bool(thoughts[-1].get("thought_open"))
    anomaly = spoken and ("call:" in stripped or any(lit in stripped for lit in SPECIAL_TOKENS))
    call: dict[str, Any] | None = None
    if routed:
        text = str(tools[0].get("text", ""))
        text = text[len(TOOL_TAG) :] if text.startswith(TOOL_TAG) else text
        parsed = native.parse_call(text)
        call = None if isinstance(parsed, score.ParseFailure) else parsed
    return {
        "routed": routed,
        "spoken": spoken,
        "spoken_prefix": prefix,
        "thought_open": thought_open,
        "anomaly": anomaly,
        "call": call,
    }


# =====================================================================================================
# The report (pure: reads the run directory and the pinned cases file, opens no socket)
# =====================================================================================================


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _case_table() -> dict[str, dict[str, Any]]:
    """case id -> {kind, tool, gt}: the dev split of the pinned file plus J10's plain questions."""
    out: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(CASES_PATH):
        if r.get("split") == SPLIT:
            out[str(r["id"])] = {"kind": r.get("kind"), "tool": r.get("tool"), "gt": r.get("gt")}
    for cid, _ in PLAIN_QUESTIONS:
        out[cid] = {"kind": "plain", "tool": None, "gt": None}
    return out


def _case_table_c() -> dict[str, dict[str, Any]]:
    """T-0098's case table plus the 7 Q12 cases (kind ``q12``, ``expect_surface`` from J1's file)."""
    out = _case_table()
    for r in _read_jsonl(Q12_PATH):
        out[str(r["id"])] = {
            "kind": "q12",
            "tool": None,
            "gt": None,
            "expect_surface": bool(r["expect_surface"]),
        }
    return out


def _design_repeats(positive: Mapping[str, bool], positive_repeats: int) -> dict[str, int]:
    """T-0103 / T-0108a *Design*: case id -> repeats, ``positive_repeats`` on a Q12 positive case, ``REPEATS``
    elsewhere."""
    return {cid: positive_repeats if pos else REPEATS for cid, pos in positive.items()}


def _design_repeats_103(positive: Mapping[str, bool]) -> dict[str, int]:
    """T-0103 *Design*: case id -> repeats, ``POSITIVE_REPEATS_103`` on a Q12 positive case, ``REPEATS`` elsewhere."""
    return _design_repeats(positive, POSITIVE_REPEATS_103)


# TABLES[T0103] and TABLES[T0108A] are registered below their rules (``_report_103``, ``_report_108a``).


def _kn(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


def _frac(rate: Mapping[str, int]) -> Fraction:
    return Fraction(rate["k"], rate["n"]) if rate["n"] else Fraction(0)


def _dist(values: Sequence[Any]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _T93._percentile(values, 50),
        "p90": _T93._percentile(values, 90),
    }


@dataclass
class _Trial:
    line: dict[str, Any]
    kind: str
    tool: str | None
    gt: Mapping[str, Any] | None
    reading: dict[str, Any]
    valid: bool = False
    exact: bool = False
    #: T-0098c only: the Q12 case's expectation and the trial's surface reading (``ui_reading``)
    expect_surface: bool | None = None
    ui: dict[str, Any] | None = None

    @property
    def errored(self) -> bool:
        return self.line.get("error") not in (None, "")


@dataclass
class _ArmStats:
    trials: list[_Trial] = field(default_factory=list)

    def tool_trials(self, tool: str | None = None) -> list[_Trial]:
        return [t for t in self.trials if t.kind == "tool" and (tool is None or t.tool == tool)]

    def routed(self, tool: str | None = None) -> list[_Trial]:
        return [t for t in self.tool_trials(tool) if t.reading["routed"]]


def _rates(trials: Sequence[_Trial], routed: Sequence[_Trial]) -> dict[str, Any]:
    spoken = [t for t in routed if t.reading["spoken"]]
    per_case: dict[str, dict[str, int]] = {}
    for t in trials:
        row = per_case.setdefault(
            str(t.line.get("case_id")), {"spoken": 0, "routed": 0, "trials": 0}
        )
        row["trials"] += 1
        row["routed"] += int(t.reading["routed"])
        row["spoken"] += int(t.reading["routed"] and t.reading["spoken"])
    return {
        "spoken": _kn(len(spoken), len(routed)),
        "routed": _kn(len(routed), len(trials)),
        "valid": _kn(sum(t.valid for t in routed), len(routed)),
        "exact": _kn(sum(t.exact for t in routed), len(routed)),
        "thought_open": _kn(sum(t.reading["thought_open"] for t in trials), len(trials)),
        "anomalies": sum(t.reading["anomaly"] for t in trials),
        "errored": _kn(sum(t.errored for t in trials), len(trials)),
        "per_case": per_case,
        "majority_cases": sum(
            1 for r in per_case.values() if r["routed"] and 2 * r["spoken"] > r["routed"]
        ),
        "first_canvas_ms": _dist(
            [
                t.line["first_canvas_ms"]
                for t in trials
                if isinstance(t.line.get("first_canvas_ms"), int | float)
            ]
        ),
        "prefix_words": _dist([len(t.reading["spoken_prefix"].split()) for t in spoken]),
    }


def _ui_channel(
    st: _ArmStats, tool_trials: Sequence[_Trial], no_tool_plain: Sequence[_Trial]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """T-0098c criterion 6's per-arm UI numbers (report keys) and what criterion 7 reads of them (stats keys)."""

    def ui(t: _Trial, key: str) -> int:
        return int(bool((t.ui or {}).get(key)))

    q12 = [t for t in st.trials if t.kind == "q12"]
    positive = [t for t in q12 if t.expect_surface]
    negative = [t for t in q12 if not t.expect_surface]
    no_belong = [*no_tool_plain, *negative]
    per_case: dict[str, dict[str, int]] = {}
    for t in q12:
        row = per_case.setdefault(str(t.line.get("case_id")), {"k": 0, "n": 0})
        row["n"] += 1
        row["k"] += ui(t, "shown")
    gated_trials = [t for t in st.trials if (t.ui or {}).get("gated")]
    gate_counts = {
        key: sum(int((t.ui or {}).get(key) or 0) for t in st.trials)
        for key in ("gated", "emitted", "failed")
    }
    out = {
        "q12_positive_shown": _kn(sum(ui(t, "shown") for t in positive), len(positive)),
        "q12_negative_shown": _kn(sum(ui(t, "shown") for t in negative), len(negative)),
        "q12_per_case": per_case,
        "surfaces_no_belong": _kn(sum(ui(t, "shown") for t in no_belong), len(no_belong)),
        "surfaces_no_tool_plain": _kn(
            sum(ui(t, "shown") for t in no_tool_plain), len(no_tool_plain)
        ),
        "surfaces_tool": _kn(sum(ui(t, "shown") for t in tool_trials), len(tool_trials)),
        "placement_failures_q12_positive": _kn(
            sum(ui(t, "placement_failure") for t in positive), len(positive)
        ),
        "placement_failures": _kn(
            sum(ui(t, "placement_failure") for t in st.trials), len(st.trials)
        ),
        "q12_tool_calls": _kn(sum(int(t.reading["routed"]) for t in q12), len(q12)),
        "every_ui_valid": _kn(
            sum(ui(t, "every_ui_valid") for t in gated_trials), len(gated_trials)
        ),
        "ui_gate": gate_counts,
    }
    stats = {
        "q12_pos_shown": out["q12_positive_shown"]["k"],
        "q12_pos_per_case": {
            cid: per_case[cid]["k"]
            for cid in dict.fromkeys(str(t.line.get("case_id")) for t in positive)
        },
        "surf_no_belong": out["surfaces_no_belong"]["k"],
        "surf_tool": out["surfaces_tool"]["k"],
        "placement_pos": out["placement_failures_q12_positive"]["k"],
        "q12_routed": out["q12_tool_calls"]["k"],
    }
    return out, stats


def _complete_cards(st: _ArmStats) -> tuple[dict[str, Any], dict[str, Any]]:
    """T-0108a criteria 4 and 7 per arm, on its Q12 positive trials: complete and last-OK (k/n, and per case k/n),
    and, reported only, the as-written-clean rate, the pruned kinds (trials pruning each kind, per arm and per case)
    and the trials whose last canvas holds a forward reference. Report keys and what the rule reads (stats keys)."""
    positive = [t for t in st.trials if t.kind == "q12" and t.expect_surface]
    complete: dict[str, dict[str, int]] = {}
    last_ok: dict[str, dict[str, int]] = {}
    kinds: dict[str, int] = {}
    kinds_case: dict[str, dict[str, int]] = {}
    for t in positive:
        ui = t.ui or {}
        cid = str(t.line.get("case_id"))
        for table, key in ((complete, "complete"), (last_ok, "last_ok")):
            row = table.setdefault(cid, {"k": 0, "n": 0})
            row["n"] += 1
            row["k"] += int(bool(ui.get(key)))
        per = kinds_case.setdefault(cid, {})
        for kind in ui.get("pruned_kinds") or ():
            kinds[kind] = kinds.get(kind, 0) + 1
            per[kind] = per.get(kind, 0) + 1

    def count(key: str) -> dict[str, int]:
        return _kn(sum(int(bool((t.ui or {}).get(key))) for t in positive), len(positive))

    out = {
        "q12_positive_complete": count("complete"),
        "q12_positive_last_ok": count("last_ok"),
        "q12_per_case_complete": complete,
        "q12_per_case_last_ok": last_ok,
        "q12_positive_as_written_clean": count("as_written_clean"),
        "q12_positive_forward_refs": _kn(
            sum(int(bool((t.ui or {}).get("forward_refs"))) for t in positive), len(positive)
        ),
        "q12_pruned_kinds": dict(sorted(kinds.items())),
        "q12_pruned_kinds_per_case": {
            cid: dict(sorted(v.items())) for cid, v in kinds_case.items()
        },
    }
    stats = {
        "complete": out["q12_positive_complete"]["k"],
        "last_ok": out["q12_positive_last_ok"]["k"],
        "complete_per_case": {cid: v["k"] for cid, v in complete.items()},
        "last_ok_per_case": {cid: v["k"] for cid, v in last_ok.items()},
    }
    return out, stats


def _ui_conditions(s: Mapping[str, Any], a0: Mapping[str, Any]) -> dict[str, bool]:
    """T-0098c criterion 7's P8-P11 for one candidate, against A0 in the same run."""
    a0_cases: Mapping[str, int] = a0["q12_pos_per_case"]
    s_cases: Mapping[str, int] = s["q12_pos_per_case"]
    return {
        "P8": s["q12_pos_shown"] >= a0["q12_pos_shown"] - P8_TOTAL_MARGIN
        and all(
            s_cases.get(cid, 0) >= P8_CASE_MIN for cid, k in a0_cases.items() if k >= P8_CASE_MIN
        ),
        "P9": s["surf_no_belong"] <= a0["surf_no_belong"] + P9_NO_BELONG_OVER_A0
        and s["surf_tool"] <= a0["surf_tool"] + P9_TOOL_OVER_A0,
        "P10": s["placement_pos"] <= a0["placement_pos"] + P10_OVER_A0,
        "P11": s["q12_routed"] <= a0["q12_routed"] + P11_OVER_A0,
    }


def _protected_103(a0: Mapping[str, Any]) -> dict[str, bool]:
    """T-0103 criterion 5: a positive case is protected iff A0 shows it in >= 3/5 of its 40 trials (>= 24)."""
    a0_cases: Mapping[str, int] = a0["q12_pos_per_case"]
    return {cid: k >= P8_CASE_MIN_103 for cid, k in a0_cases.items()}


def _ui_conditions_103(s: Mapping[str, Any], a0: Mapping[str, Any]) -> dict[str, bool]:
    """T-0103 criterion 5's P8-P11 for C3 against A0 in the same run: P8, P10 and P11 by reading S (the T-0098c
    margins as the same proportions of 160, 40 and 175), P9 verbatim."""
    s_cases: Mapping[str, int] = s["q12_pos_per_case"]
    return {
        "P8": s["q12_pos_shown"] >= a0["q12_pos_shown"] - P8_TOTAL_MARGIN_103
        and all(
            s_cases.get(cid, 0) >= P8_CASE_MIN_103
            for cid, protected in _protected_103(a0).items()
            if protected
        ),
        "P9": s["surf_no_belong"] <= a0["surf_no_belong"] + P9_NO_BELONG_OVER_A0
        and s["surf_tool"] <= a0["surf_tool"] + P9_TOOL_OVER_A0,
        "P10": s["placement_pos"] <= a0["placement_pos"] + P10_OVER_A0_103,
        "P11": s["q12_routed"] <= a0["q12_routed"] + P11_OVER_A0_103,
    }


def _protected_108a(a0_cases: Mapping[str, int]) -> dict[str, bool]:
    """T-0108a criterion 5: a positive case is protected iff A0's count there is >= 24 of its 40 (A0's, never K's)."""
    return {cid: k >= PROTECT_MIN_108A for cid, k in a0_cases.items()}


def _conditions_108a(s: Mapping[str, Any], a0: Mapping[str, Any]) -> dict[str, bool]:
    """T-0108a criterion 5's UI conditions for one K against A0 in the same run: Q1 and Q2 on complete cards, U1 on
    last-OK, U2 = T-0103's P8 on shown, P9-P11 as T-0103 (P9 verbatim, P10 and P11 by reading S)."""

    def per_case(key: str) -> bool:
        k_cases: Mapping[str, int] = s[key]
        return all(
            k_cases.get(cid, 0) >= PROTECT_MIN_108A
            for cid, protected in _protected_108a(a0[key]).items()
            if protected
        )

    t103 = _ui_conditions_103(s, a0)
    return {
        "Q1": s["complete"] >= a0["complete"] + Q1_MARGIN_108A,
        "Q2": per_case("complete_per_case"),
        "U1": s["last_ok"] >= a0["last_ok"] - U_MARGIN_108A and per_case("last_ok_per_case"),
        "U2": t103["P8"],
        "P9": t103["P9"],
        "P10": t103["P10"],
        "P11": t103["P11"],
    }


def _ratio(share: Fraction, n: int) -> str:
    """A scaled threshold as the ``"k/n"`` string of ``share`` x ``n`` (exact; every T-0103 one is a whole count)."""
    k = share * n
    return f"{k.numerator}/{n}" if k.denominator == 1 else f"{k}*{n}"


def _whole(value: Fraction | int) -> int | str:
    f = Fraction(value)
    return f.numerator if f.denominator == 1 else str(f)


THRESHOLDS_103: dict[str, str] = {
    "P8_total_margin": _ratio(P8_TOTAL_SHARE_103, Q12_POSITIVE_TRIALS_103),
    "P8_protected_min": _ratio(P8_CASE_SHARE_103, Q12_CASE_TRIALS_103),
    "P10_margin": _ratio(P10_SHARE_103, Q12_POSITIVE_TRIALS_103),
    "P11_margin": _ratio(P11_SHARE_103, Q12_TRIALS_103),
    "R3_min": _ratio(R3_SHARE_103, Q12_POSITIVE_TRIALS_103),
}


def _report_103(
    *,
    table: ArmTable,
    manifest: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    complete: bool,
    arms: Mapping[str, _ArmStats],
    out_arms: dict[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
    candidates: dict[str, dict[str, Any]],
    notes: list[str],
) -> dict[str, Any]:
    """T-0103 criteria 5 and 7: the readings R1 and R3, the outcome, the winner and the report's extras."""
    a0, s = stats["A0"], stats["C3"]
    readings = {
        "R1": a0["type_spoken"] <= R1_MAX,
        "R3": a0["q12_pos_shown"] >= R3_MIN_SHOWN_103,
    }
    void_arms = sorted(a for a in table.arms if stats[a]["void"])
    failing = [p for p in table.ps if candidates["C3"][p] is not True]
    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete run")
    for arm in void_arms:
        reasons.append(f"void {arm} (> 2 % of its trials errored)")
    if not readings["R1"]:
        reasons.append("R1: finding not reproduced")
    if not readings["R3"]:
        reasons.append("R3: UI baseline not reproduced")
    if reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    elif failing:
        outcome = "C3 not confirmed: fails " + ", ".join(failing)
    else:
        outcome = CONFIRMED
    if a0["void"]:
        notes.append("A0 is void: C3 cannot be compared against it")

    protected = _protected_103(a0)
    positive_cases: dict[str, Any] = {}
    for cid, flag in protected.items():
        positive_cases[cid] = {
            arm: out_arms[arm]["q12_per_case"].get(cid, _kn(0, 0)) for arm in table.arms
        } | {"protected": flag}
    blocks: dict[str, Any] = {}
    for arm in table.arms:
        positive = [t for t in arms[arm].trials if t.kind == "q12" and t.expect_surface]
        early = [t for t in positive if int(t.line.get("repeat", -1)) < REPEATS]
        late = [t for t in positive if int(t.line.get("repeat", -1)) >= REPEATS]
        blocks[arm] = {
            f"r0-{REPEATS - 1}": _kn(
                sum(bool((t.ui or {}).get("shown")) for t in early), len(early)
            ),
            f"r{REPEATS}-{POSITIVE_REPEATS_103 - 1}": _kn(
                sum(bool((t.ui or {}).get("shown")) for t in late), len(late)
            ),
        }
    thresholds = {
        "P8_total": {
            "margin": THRESHOLDS_103["P8_total_margin"],
            "A0": a0["q12_pos_shown"],
            "bar": _whole(a0["q12_pos_shown"] - P8_TOTAL_MARGIN_103),
            "C3": s["q12_pos_shown"],
        },
        "P8_per_case": {
            "protected_min": THRESHOLDS_103["P8_protected_min"],
            "protected": sorted(cid for cid, flag in protected.items() if flag),
            "C3": dict(s["q12_pos_per_case"]),
        },
        "P10": {
            "margin": THRESHOLDS_103["P10_margin"],
            "A0": a0["placement_pos"],
            "bar": _whole(a0["placement_pos"] + P10_OVER_A0_103),
            "C3": s["placement_pos"],
        },
        "P11": {
            "margin": THRESHOLDS_103["P11_margin"],
            "A0": a0["q12_routed"],
            "bar": _whole(a0["q12_routed"] + P11_OVER_A0_103),
            "C3": s["q12_routed"],
        },
        "R3": {"min": THRESHOLDS_103["R3_min"], "A0": a0["q12_pos_shown"]},
    }
    return {
        "task": T0103,
        "table": table.task,
        "rule": table.rule,
        "regime": manifest.get("regime") or {},
        "arm_texts": {
            a: {"sha256": v.get("sha256"), "chars": len(str(v.get("text", "")))}
            for a, v in (manifest.get("arms") or {}).items()
            if isinstance(v, Mapping)
        },
        "trials": len(lines),
        "complete": complete,
        "void_arms": void_arms,
        "arms": out_arms,
        "candidates": candidates,
        "readings": readings,
        "positive_cases": positive_cases,
        "q12_positive_shown_by_block": blocks,
        "thresholds": thresholds,
        "outcome": outcome,
        "winner": "C3" if outcome == CONFIRMED else NO_CHANGE,
        "notes": notes,
    }


THRESHOLDS_108A: dict[str, str] = {
    "Q1_margin": _ratio(Q1_SHARE_108A, Q12_POSITIVE_TRIALS_103),
    "protected_min": _ratio(P8_CASE_SHARE_103, Q12_CASE_TRIALS_103),
    "U1_margin": _ratio(P8_TOTAL_SHARE_103, Q12_POSITIVE_TRIALS_103),
    "U2_margin": _ratio(P8_TOTAL_SHARE_103, Q12_POSITIVE_TRIALS_103),
    # P9 verbatim: of the 80 trials where no surface belongs (8 no-tool, 5 plain, 3 Q12 negative cases x 5) and of
    # the 200 tool trials (40 x 5)
    "P9_no_belong_over_A0": f"{P9_NO_BELONG_OVER_A0}/80",
    "P9_tool_over_A0": f"{P9_TOOL_OVER_A0}/200",
    "P10_margin": _ratio(P10_SHARE_103, Q12_POSITIVE_TRIALS_103),
    "P11_margin": _ratio(P11_SHARE_103, Q12_TRIALS_103),
    "R3_min": _ratio(R3_SHARE_103, Q12_POSITIVE_TRIALS_103),
    "R4_max": _ratio(R4_SHARE_108A, Q12_POSITIVE_TRIALS_103),
}
SPEECH_LABEL = "reported, not gated (D16)"


def _report_108a(
    *,
    table: ArmTable,
    manifest: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    complete: bool,
    arms: Mapping[str, _ArmStats],
    out_arms: dict[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
    candidates: dict[str, dict[str, Any]],
    notes: list[str],
) -> dict[str, Any]:
    """T-0108a criteria 5 and 7: the controls R3 and R4, the void rule, the outcome, the winner (the higher complete
    count among the Ks that pass, a tie to K1) and the report's extras."""
    a0 = stats["A0"]
    readings = {
        "R3": a0["q12_pos_shown"] >= R3_MIN_SHOWN_103,
        "R4": a0["complete"] <= R4_MAX_COMPLETE_108A,
    }
    void_arms = sorted(a for a in table.arms if stats[a]["void"])
    failing = {k: [c for c in table.ps if candidates[k][c] is not True] for k in table.candidates}
    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete run")
    for arm in void_arms:
        reasons.append(f"void {arm} (> 2 % of its trials errored)")
    if not readings["R3"]:
        reasons.append("R3: UI baseline not reproduced")
    if not readings["R4"]:
        reasons.append("R4: degradation not reproduced (A0's cards are already complete)")
    passing = [k for k in table.candidates if not failing[k]]
    winner: str = NO_CHANGE
    if reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    elif passing:
        best = max(stats[k]["complete"] for k in passing)
        winner = next(k for k in passing if stats[k]["complete"] == best)
        outcome = (
            f"{winner} selected ("
            + ", ".join(f"{k} complete {stats[k]['complete']}" for k in passing)
            + f"; A0 complete {a0['complete']})"
        )
    else:
        outcome = "no change: " + "; ".join(
            f"{k} fails {', '.join(failing[k])}" for k in table.candidates
        )

    positive_cases: dict[str, Any] = {}
    protected: dict[str, dict[str, bool]] = {
        "Q2": _protected_108a(a0["complete_per_case"]),
        "U1": _protected_108a(a0["last_ok_per_case"]),
        "U2": _protected_108a(a0["q12_pos_per_case"]),
    }
    for cid in a0["complete_per_case"]:
        positive_cases[cid] = {
            key: {a: out_arms[a][src].get(cid, _kn(0, 0)) for a in table.arms}
            for key, src in (
                ("complete", "q12_per_case_complete"),
                ("last_ok", "q12_per_case_last_ok"),
                ("shown", "q12_per_case"),
            )
        }

    def bar(key: str, margin: Fraction | int, sign: int) -> dict[str, Any]:
        return {"A0": a0[key], "bar": _whole(a0[key] + sign * margin)} | {
            k: stats[k][key] for k in table.candidates
        }

    thresholds = {
        "Q1": {"margin": THRESHOLDS_108A["Q1_margin"]} | bar("complete", Q1_MARGIN_108A, 1),
        "Q2": {
            "protected_min": THRESHOLDS_108A["protected_min"],
            "protected": sorted(c for c, f in protected["Q2"].items() if f),
        },
        "U1": {
            "margin": THRESHOLDS_108A["U1_margin"],
            "protected_min": THRESHOLDS_108A["protected_min"],
            "protected": sorted(c for c, f in protected["U1"].items() if f),
        }
        | bar("last_ok", U_MARGIN_108A, -1),
        "U2": {
            "margin": THRESHOLDS_108A["U2_margin"],
            "protected_min": THRESHOLDS_108A["protected_min"],
            "protected": sorted(c for c, f in protected["U2"].items() if f),
        }
        | bar("q12_pos_shown", U_MARGIN_108A, -1),
        "P9": {
            "no_belong_over_A0": THRESHOLDS_108A["P9_no_belong_over_A0"],
            "tool_over_A0": THRESHOLDS_108A["P9_tool_over_A0"],
        },
        "P10": {"margin": THRESHOLDS_108A["P10_margin"]} | bar("placement_pos", P10_OVER_A0_103, 1),
        "P11": {"margin": THRESHOLDS_108A["P11_margin"]} | bar("q12_routed", P11_OVER_A0_103, 1),
        "R3": {"min": THRESHOLDS_108A["R3_min"], "A0": a0["q12_pos_shown"]},
        "R4": {"max": THRESHOLDS_108A["R4_max"], "A0": a0["complete"]},
    }
    speech = {
        "label": SPEECH_LABEL,
        "arms": {
            a: {
                "spoken": out_arms[a]["spoken"],
                "type_at_spoken": ((out_arms[a].get("tools") or {}).get("type_at") or {}).get(
                    "spoken"
                ),
            }
            for a in table.arms
        },
    }
    if a0["void"]:
        notes.append("A0 is void: no K can be compared against it")
    return {
        "task": T0108A,
        "table": table.task,
        "rule": table.rule,
        "regime": manifest.get("regime") or {},
        "arm_texts": {
            a: {"sha256": v.get("sha256"), "chars": len(str(v.get("text", "")))}
            for a, v in (manifest.get("arms") or {}).items()
            if isinstance(v, Mapping)
        },
        "u_prime": manifest.get("u_prime") or {},
        "trials": len(lines),
        "complete": complete,
        "void_arms": void_arms,
        "arms": out_arms,
        "candidates": candidates,
        "readings": readings,
        "positive_cases": positive_cases,
        "protected": protected,
        "thresholds": thresholds,
        "speech": speech,
        "outcome": outcome,
        "winner": winner,
        "notes": notes,
    }


TABLES[T0103] = ArmTable(
    task=T0103,
    rule="H-T103",
    arms=ARMS_103,
    roles=ROLES_103,
    candidates=CANDIDATES_103,
    a0_sha256=A0_SHA256,
    texts=_arm_texts_103,
    cases=_load_cases_c,
    case_sources=TABLES[T0098C].case_sources,
    ps=PS_C,
    ui_channel=True,
    request_prefix="t0103",
    out_stem="t0103-c3-confirm",
    repeats=_design_repeats_103(
        {cid: bool(meta.get("expect_surface")) for cid, meta in _case_table_c().items()}
    ),
    text_problems=_c3_problems_103,
    conditions=_ui_conditions_103,
    report=_report_103,
    thresholds=THRESHOLDS_103,
    reading="S (H-T103 (c)): T-0098c's P8, P10, P11 and R3 margins as the same proportions",
)

TABLES[T0108A] = ArmTable(
    task=T0108A,
    rule="H-T108a",
    arms=ARMS_108A,
    roles=ROLES_108A,
    candidates=CANDIDATES_108A,
    a0_sha256=A0_SHA256,
    texts=_arm_texts_108a,
    cases=_load_cases_c,
    case_sources=TABLES[T0098C].case_sources,
    ps=PS_108A,
    ui_channel=True,
    request_prefix="t0108a",
    out_stem="t0108a-ui-fields-ab",
    repeats=_design_repeats(
        {cid: bool(meta.get("expect_surface")) for cid, meta in _case_table_c().items()},
        POSITIVE_REPEATS_103,
    ),
    text_problems=_k_problems_108a,
    positive_repeats=POSITIVE_REPEATS_103,
    conditions=_conditions_108a,
    report=_report_108a,
    thresholds=THRESHOLDS_108A,
    reading=(
        "complete card (H-T108a (a)): the last gated call returned a surface and nothing is pruned on it -- "
        "schema-invalid after salvage or dropped by salvage, read through UiGate._repair; forward-reference "
        "prunes are reported, not counted; U2, P9, P10 and P11 as T-0103 (reading S); P1, P2 and R1 dropped (D16)"
    ),
    complete_cards=True,
)


def build_report(run_dir: Path) -> dict[str, Any]:
    """Criteria 5-7 from ``trials.jsonl`` (and ``manifest.json`` when present). Reads files only. The arm table is
    the manifest's ``table`` (no key, or no manifest: T-0098, whose report is exactly as at 3ee1636)."""
    run_dir = Path(run_dir)
    manifest: dict[str, Any] = {}
    if (run_dir / "manifest.json").is_file():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    table = TABLES[str(manifest.get("table") or T0098)]
    ui_channel = table.ui_channel
    lines = _read_jsonl(run_dir / "trials.jsonl")
    cases = _case_table_c() if ui_channel else _case_table()
    tool_set = toolset.load()
    notes: list[str] = []

    arms: dict[str, _ArmStats] = {a: _ArmStats() for a in table.arms}
    keys: set[tuple[str, str, int]] = set()
    duplicates = 0
    for line in lines:
        arm = str(line.get("arm"))
        cid = str(line.get("case_id"))
        key = (cid, arm, int(line.get("repeat", -1)))
        duplicates += key in keys
        keys.add(key)
        meta = cases.get(cid) or {
            "kind": line.get("kind"),
            "tool": line.get("tool"),
            "gt": line.get("gt"),
        }
        reading = classify(line.get("commits") or [])
        trial = _Trial(
            line=line,
            kind=str(meta["kind"]),
            tool=meta.get("tool"),
            gt=meta.get("gt"),
            reading=reading,
        )
        call = reading["call"]
        if reading["routed"] and call is not None:
            trial.valid = bool(score.schema_valid(call, tool_set))
            trial.exact = isinstance(trial.gt, Mapping) and bool(score.exact(call, trial.gt))
        if ui_channel:
            trial.expect_surface = meta.get("expect_surface")
            request_id = line.get("request_id") or f"{table.request_prefix}-r{key[2]}-{cid}-{arm}"
            trial.ui = ui_reading(
                line.get("commits") or [], request_id=str(request_id), cards=table.complete_cards
            )
        arms.setdefault(arm, _ArmStats()).trials.append(trial)
    expected = {
        (cid, a, r) for cid in cases for a in table.arms for r in range(table.repeats_of(cid))
    }
    complete = keys == expected and duplicates == 0 and len(lines) == len(expected)
    if not complete:
        notes.append(
            f"incomplete run: {len(lines)} trial lines, {len(expected - keys)} (case, arm, repeat) missing, "
            f"{duplicates} duplicated; no candidate is selected"
        )
    if manifest.get("stopped"):
        notes.append(f"the run stopped: {manifest['stopped']}")

    out_arms: dict[str, Any] = {}
    stats: dict[str, dict[str, Any]] = {}
    for arm, st in arms.items():
        tool_trials = st.tool_trials()
        routed = st.routed()
        # P5's denominators: the no-tool and plain trials (T-0098c: Q12 trials never count here)
        other = [t for t in st.trials if t.kind not in ("tool", "q12")]
        errored = sum(t.errored for t in st.trials)
        void = bool(st.trials) and Fraction(errored, len(st.trials)) > VOID_ERROR_RATE
        csk_routed = [t for t in routed if t.tool in CSK]
        type_all, type_routed = st.tool_trials("type_at"), st.routed("type_at")
        pooled = _rates(tool_trials, routed)
        type_rates = _rates(type_all, type_routed)
        per_other: dict[str, dict[str, int]] = {}
        for t in other:
            row = per_other.setdefault(str(t.line.get("case_id")), {"false_tool": 0, "trials": 0})
            row["trials"] += 1
            row["false_tool"] += int(t.reading["routed"])
        arm_out = {
            "role": table.roles.get(arm),
            "trials": len(st.trials),
            "errored": _kn(errored, len(st.trials)),
            "void": void,
            "route_recall": _kn(len(routed), len(tool_trials)),
            "spoken": pooled["spoken"],
            "csk_spoken": _kn(sum(t.reading["spoken"] for t in csk_routed), len(csk_routed)),
            "valid": pooled["valid"],
            "exact": pooled["exact"],
            "thought_open": pooled["thought_open"],
            "false_tools": _kn(sum(t.reading["routed"] for t in other), len(other)),
            "anomalies": sum(t.reading["anomaly"] for t in st.trials),
            "first_canvas_ms": _dist(
                [
                    t.line["first_canvas_ms"]
                    for t in st.trials
                    if isinstance(t.line.get("first_canvas_ms"), int | float)
                ]
            ),
            "prefix_words": pooled["prefix_words"],
            "tools": {tool: _rates(st.tool_trials(tool), st.routed(tool)) for tool in TOOL_NAMES},
            "no_tool_cases": per_other,
        }
        out_arms[arm] = arm_out
        stats[arm] = {
            "void": void,
            "type_spoken": _frac(type_rates["spoken"]),
            "type_majority": type_rates["majority_cases"],
            "csk_spoken": _frac(arm_out["csk_spoken"]),
            "recall": _frac(arm_out["route_recall"]),
            "type_recall": _frac(type_rates["routed"]),
            "valid": _frac(pooled["valid"]),
            "exact": _frac(pooled["exact"]),
            "false": arm_out["false_tools"]["k"],
            "anomalies": arm_out["anomalies"],
            "thought_open": pooled["thought_open"]["k"],
        }
        if ui_channel:
            ui_out, ui_stats = _ui_channel(st, tool_trials, other)
            arm_out.update(ui_out)
            stats[arm].update(ui_stats)
        if table.complete_cards:
            card_out, card_stats = _complete_cards(st)
            arm_out.update(card_out)
            stats[arm].update(card_stats)

    a0 = stats["A0"]
    candidates: dict[str, dict[str, Any]] = {}
    for c in table.candidates:
        s = stats[c]
        p: dict[str, Any] = {
            "P1": s["type_spoken"] >= P1_SPOKEN_MIN and s["type_majority"] >= P1_MAJORITY_CASES,
            "P2": s["csk_spoken"] >= a0["csk_spoken"] - P2_MARGIN,
            "P3": s["recall"] >= a0["recall"] - P3_POOLED_MARGIN
            and s["type_recall"] >= a0["type_recall"] - P3_TYPE_MARGIN,
            "P4": s["valid"] >= a0["valid"] - P4_VALID_MARGIN
            and s["exact"] >= a0["exact"] - P4_EXACT_MARGIN,
            "P5": s["false"] <= P5_MAX and s["false"] <= a0["false"] + P5_OVER_A0,
            "P6": s["anomalies"] == 0,
            "P7": s["thought_open"] <= a0["thought_open"] + P7_OVER_A0,
        }
        if table.conditions is not None:
            p.update(table.conditions(s, a0))
        elif ui_channel:
            p.update(_ui_conditions(s, a0))
        # the table's conditions only, in its order (T-0108a drops P1 and P2 per D16)
        p = {k: p[k] for k in table.ps}
        p["void"] = s["void"]
        p["passes"] = all(p[k] for k in table.ps) and not s["void"] and not a0["void"]
        candidates[c] = p
    if table.report is not None:
        return table.report(
            table=table,
            manifest=manifest,
            lines=lines,
            complete=complete,
            arms=arms,
            out_arms=out_arms,
            stats=stats,
            candidates=candidates,
            notes=notes,
        )
    if ui_channel:
        readings = {
            "R1": a0["type_spoken"] <= R1_MAX,
            "R2c": stats["D1"]["type_spoken"] >= R2C_MIN,
            "R3": a0["q12_pos_shown"] >= R3_MIN_SHOWN,
        }
    else:
        readings = {
            "R1": a0["type_spoken"] <= R1_MAX,
            "R2": stats["A1"]["type_spoken"] >= R2_MIN,
        }
    if a0["void"]:
        notes.append("A0 is void: no candidate can be compared against it")
    winner: str = NO_CHANGE
    if readings["R1"] and readings.get("R3", True) and complete:
        winner = next((c for c in table.candidates if candidates[c]["passes"]), NO_CHANGE)
    if ui_channel:
        if not readings["R1"]:
            notes.append("R1: finding not reproduced; no candidate is selected")
        if not readings["R3"]:
            notes.append(
                "R3: UI baseline not reproduced: the guard cannot detect a regression; no candidate is selected"
            )
        if not readings["R2c"]:
            notes.append("R2c: UI-block cause not reproduced (label only)")
        return {
            "task": T0098C,
            "table": table.task,
            "rule": table.rule,
            "regime": manifest.get("regime") or {},
            "arm_texts": {
                a: {"sha256": v.get("sha256"), "chars": len(str(v.get("text", "")))}
                for a, v in (manifest.get("arms") or {}).items()
                if isinstance(v, Mapping)
            },
            "trials": len(lines),
            "complete": complete,
            "void_arms": sorted(a for a in table.arms if stats[a]["void"]),
            "arms": out_arms,
            "candidates": candidates,
            "readings": readings,
            "winner": winner,
            "notes": notes,
        }
    return {
        "task": "T-0098",
        "regime": manifest.get("regime") or {},
        "arm_texts": {
            a: {"sha256": v.get("sha256"), "chars": len(str(v.get("text", "")))}
            for a, v in (manifest.get("arms") or {}).items()
            if isinstance(v, Mapping)
        },
        "trials": len(lines),
        "complete": complete,
        "void_arms": sorted(a for a in ARMS if stats[a]["void"]),
        "arms": out_arms,
        "candidates": candidates,
        "readings": readings,
        "winner": winner,
        "notes": notes,
    }


# --- report.md ---------------------------------------------------------------------------------------------------


def _r(rate: Mapping[str, Any] | None) -> str:
    if not isinstance(rate, Mapping) or "k" not in rate:
        return "-"
    k, n = rate["k"], rate["n"]
    return f"{k}/{n} ({k / n:.3f})" if n else f"{k}/{n}"


def _d(dist: Mapping[str, Any] | None) -> str:
    if not isinstance(dist, Mapping):
        return "-"
    return f"{dist.get('p50')} / {dist.get('p90')} (n={dist.get('n')})"


def _yes(v: Any) -> str:
    return "pass" if v is True else "FAIL"


def render_markdown(report: Mapping[str, Any]) -> str:
    if report.get("task") == T0108A:
        return _render_markdown_108a(report)
    if report.get("task") == T0103:
        return _render_markdown_103(report)
    if report.get("task") == T0098C:
        return _render_markdown_c(report)
    arms: Mapping[str, Any] = report.get("arms") or {}
    regime = report.get("regime") or {}
    out = ["# T-0098 native-prompt A/B (dev): report", ""]
    out.append(
        "Regime: "
        + (", ".join(f"{k}={v}" for k, v in regime.items()) if regime else "(no manifest)")
    )
    out.append("")
    out.append(f"Trials: {report.get('trials')}; complete: {report.get('complete')}")
    for note in report.get("notes") or []:
        out.append(f"- note: {note}")
    out.append("")

    readings = report.get("readings") or {}
    a0_type = ((arms.get("A0") or {}).get("tools") or {}).get("type_at") or {}
    a1_type = ((arms.get("A1") or {}).get("tools") or {}).get("type_at") or {}
    out.append("## Reproduction readings (labels, not gates)")
    out.append("")
    out.append(
        f"- R1 (A0 `type_at` spoken <= 0.3): {_r(a0_type.get('spoken'))}: "
        + ("holds" if readings.get("R1") else "fails")
    )
    out.append(
        f"- R2 (A1 `type_at` spoken >= 0.7): {_r(a1_type.get('spoken'))}: "
        + ("holds" if readings.get("R2") else "fails")
    )
    if readings.get("R1") is False:
        out.append("")
        out.append(
            "**R1: finding not reproduced.** No candidate is selected (N3); the follow-up is a T-0093-harness "
            "re-run with n >= 3."
        )
    if readings.get("R2") is False:
        out.append("")
        out.append("**R2: eval baseline not reproduced.** Selection proceeds against A0.")
    out.append("")

    winner = report.get("winner")
    out.append("## Decision (criterion 7, against A0 in this run)")
    out.append("")
    out.append("| | " + " | ".join(PS) + " | void | passes |")
    out.append("|---" * (len(PS) + 3) + "|")
    for c in CANDIDATES:
        p = (report.get("candidates") or {}).get(c) or {}
        out.append(
            f"| {c} | "
            + " | ".join(_yes(p.get(k)) for k in PS)
            + f" | {p.get('void')} | {_yes(p.get('passes'))} |"
        )
    out.append("")
    out.append(
        "**Winner: "
        + (f"{winner}" if winner in CANDIDATES else "no change")
        + "**"
        + ("" if winner in CANDIDATES else " (neither candidate is selected)")
    )
    out.append("")
    out.append(f"Void arms (> 2 % errored): {', '.join(report.get('void_arms') or []) or 'none'}")
    out.append("")

    out.append("## The numbers behind the rule (k/n; spoken, valid and exact over routed trials)")
    out.append("")
    out.append(
        "| arm | type spoken | type majority cases | click+scroll+key spoken | route recall | type recall | valid | "
        "exact | false tools | anomalies | thought_open (tool trials) | errored |"
    )
    out.append("|---" * 12 + "|")
    for arm in ARMS:
        a = arms.get(arm) or {}
        t = (a.get("tools") or {}).get("type_at") or {}
        out.append(
            f"| {arm} | {_r(t.get('spoken'))} | {t.get('majority_cases')} | {_r(a.get('csk_spoken'))} | "
            f"{_r(a.get('route_recall'))} | {_r(t.get('routed'))} | {_r(a.get('valid'))} | {_r(a.get('exact'))} | "
            f"{_r(a.get('false_tools'))} | {a.get('anomalies')} | {_r(a.get('thought_open'))} | "
            f"{_r(a.get('errored'))} |"
        )
    out.append("")

    out.append("## Per arm and tool")
    out.append("")
    out.append(
        "| arm | tool | spoken/routed | routed/trials | valid | exact | thought_open | anomalies | "
        "first canvas ms p50/p90 | prefix words p50/p90 |"
    )
    out.append("|---" * 10 + "|")
    for arm in ARMS:
        for tool in TOOL_NAMES:
            t = ((arms.get(arm) or {}).get("tools") or {}).get(tool) or {}
            out.append(
                f"| {arm} | {tool} | {_r(t.get('spoken'))} | {_r(t.get('routed'))} | {_r(t.get('valid'))} | "
                f"{_r(t.get('exact'))} | {_r(t.get('thought_open'))} | {t.get('anomalies')} | "
                f"{_d(t.get('first_canvas_ms'))} | {_d(t.get('prefix_words'))} |"
            )
    out.append("")

    out.append("## Per case: spoken k / routed of 5 (tool cases)")
    out.append("")
    case_ids: list[str] = []
    for arm in ARMS:
        for tool in TOOL_NAMES:
            for cid in (((arms.get(arm) or {}).get("tools") or {}).get(tool) or {}).get(
                "per_case"
            ) or {}:
                if cid not in case_ids:
                    case_ids.append(cid)
    out.append("| case | " + " | ".join(ARMS) + " |")
    out.append("|---" * (len(ARMS) + 1) + "|")
    for cid in case_ids:
        cells = []
        for arm in ARMS:
            row = None
            for tool in TOOL_NAMES:
                row = row or (
                    (((arms.get(arm) or {}).get("tools") or {}).get(tool) or {}).get("per_case")
                    or {}
                ).get(cid)
            cells.append(f"{row['spoken']}/{row['routed']}" if row else "-")
        out.append(f"| {cid} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Per case: false tools of 5 (no-tool and plain cases)")
    out.append("")
    other_ids: list[str] = []
    for arm in ARMS:
        for cid in (arms.get(arm) or {}).get("no_tool_cases") or {}:
            if cid not in other_ids:
                other_ids.append(cid)
    out.append("| case | " + " | ".join(ARMS) + " |")
    out.append("|---" * (len(ARMS) + 1) + "|")
    for cid in other_ids:
        cells = []
        for arm in ARMS:
            row = ((arms.get(arm) or {}).get("no_tool_cases") or {}).get(cid)
            cells.append(f"{row['false_tool']}/{row['trials']}" if row else "-")
        out.append(f"| {cid} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def _render_markdown_c(report: Mapping[str, Any]) -> str:
    """report.md for a T-0098c run (criterion 6): the readings with their labels, the decision table P1-P11, the
    numbers behind it as k/n, the UI channel per arm and the per-case tables."""
    arms_order = ARMS_C
    arms: Mapping[str, Any] = report.get("arms") or {}
    regime = report.get("regime") or {}
    out = ["# T-0098c UI-block prompt A/B (dev): report", ""]
    out.append(
        "Regime: "
        + (", ".join(f"{k}={v}" for k, v in regime.items()) if regime else "(no manifest)")
    )
    out.append("")
    out.append(
        f"Table: {report.get('table')}; rule: {report.get('rule')}; trials: {report.get('trials')}; "
        f"complete: {report.get('complete')}"
    )
    for note in report.get("notes") or []:
        out.append(f"- note: {note}")
    out.append("")

    readings = report.get("readings") or {}

    def arm(a: str) -> Mapping[str, Any]:
        return arms.get(a) or {}

    def type_spoken(a: str) -> Any:
        return ((arm(a).get("tools") or {}).get("type_at") or {}).get("spoken")

    out.append("## Readings")
    out.append("")
    out.append(
        f"- R1 (A0 `type_at` spoken <= 0.3): {_r(type_spoken('A0'))}: "
        + ("holds" if readings.get("R1") else "fails")
    )
    out.append(
        f"- R2c (D1 `type_at` spoken >= 0.7; label only): {_r(type_spoken('D1'))}: "
        + ("holds" if readings.get("R2c") else "fails")
    )
    out.append(
        f"- R3 (A0 Q12 positive shown >= 14/20): {_r(arm('A0').get('q12_positive_shown'))}: "
        + ("holds" if readings.get("R3") else "fails")
    )
    if readings.get("R1") is False:
        out.append("")
        out.append("**R1: finding not reproduced.** No candidate is selected.")
    if readings.get("R2c") is False:
        out.append("")
        out.append(
            "**R2c: UI-block cause not reproduced.** A label only; selection proceeds against A0."
        )
    if readings.get("R3") is False:
        out.append("")
        out.append(
            "**R3: UI baseline not reproduced: the guard cannot detect a regression.** No candidate is selected."
        )
    out.append("")

    winner = report.get("winner")
    out.append("## Decision (criterion 7, against A0 in this run; winner order C3, C5, C4)")
    out.append("")
    out.append("| | " + " | ".join(PS_C) + " | void | passes |")
    out.append("|---" * (len(PS_C) + 3) + "|")
    for c in CANDIDATES_C:
        p = (report.get("candidates") or {}).get(c) or {}
        out.append(
            f"| {c} | "
            + " | ".join(_yes(p.get(k)) for k in PS_C)
            + f" | {p.get('void')} | {_yes(p.get('passes'))} |"
        )
    out.append("")
    out.append(
        "**Winner: "
        + (f"{winner}" if winner in CANDIDATES_C else "no change")
        + "**"
        + ("" if winner in CANDIDATES_C else " (no candidate is selected; D1, D3, D4 never are)")
    )
    out.append("")
    out.append(f"Void arms (> 2 % errored): {', '.join(report.get('void_arms') or []) or 'none'}")
    out.append("")

    out.append("## The numbers behind P1-P7 (k/n; spoken, valid and exact over routed tool trials)")
    out.append("")
    out.append(
        "| arm | type spoken | type majority cases | click+scroll+key spoken | route recall | type recall | valid | "
        "exact | false tools (no-tool + plain) | anomalies | thought_open (tool trials) | errored |"
    )
    out.append("|---" * 12 + "|")
    for a in arms_order:
        x = arm(a)
        t = (x.get("tools") or {}).get("type_at") or {}
        out.append(
            f"| {a} | {_r(t.get('spoken'))} | {t.get('majority_cases')} | {_r(x.get('csk_spoken'))} | "
            f"{_r(x.get('route_recall'))} | {_r(t.get('routed'))} | {_r(x.get('valid'))} | {_r(x.get('exact'))} | "
            f"{_r(x.get('false_tools'))} | {x.get('anomalies')} | {_r(x.get('thought_open'))} | "
            f"{_r(x.get('errored'))} |"
        )
    out.append("")

    out.append(
        "## The UI channel (P8-P11; shown = the real UiGate returned a surface for some ui commit)"
    )
    out.append("")
    out.append(
        "| arm | Q12 positive shown | Q12 negative shown | surfaces where none belongs | surfaces on no-tool + "
        "plain | surfaces on tool trials | placement failures (Q12 positive) | placement failures (all) | "
        "tool calls on Q12 | every ui valid (gated trials) | ui.gate gated / emitted / failed |"
    )
    out.append("|---" * 11 + "|")
    for a in arms_order:
        x = arm(a)
        g = x.get("ui_gate") or {}
        out.append(
            f"| {a} | {_r(x.get('q12_positive_shown'))} | {_r(x.get('q12_negative_shown'))} | "
            f"{_r(x.get('surfaces_no_belong'))} | {_r(x.get('surfaces_no_tool_plain'))} | "
            f"{_r(x.get('surfaces_tool'))} | {_r(x.get('placement_failures_q12_positive'))} | "
            f"{_r(x.get('placement_failures'))} | {_r(x.get('q12_tool_calls'))} | "
            f"{_r(x.get('every_ui_valid'))} | {g.get('gated')} / {g.get('emitted')} / {g.get('failed')} |"
        )
    out.append("")

    out.append("## Per arm and tool")
    out.append("")
    out.append(
        "| arm | tool | spoken/routed | routed/trials | valid | exact | thought_open | anomalies | "
        "first canvas ms p50/p90 | prefix words p50/p90 |"
    )
    out.append("|---" * 10 + "|")
    for a in arms_order:
        for tool in TOOL_NAMES:
            t = (arm(a).get("tools") or {}).get(tool) or {}
            out.append(
                f"| {a} | {tool} | {_r(t.get('spoken'))} | {_r(t.get('routed'))} | {_r(t.get('valid'))} | "
                f"{_r(t.get('exact'))} | {_r(t.get('thought_open'))} | {t.get('anomalies')} | "
                f"{_d(t.get('first_canvas_ms'))} | {_d(t.get('prefix_words'))} |"
            )
    out.append("")

    def per_case_table(title: str, ids: list[str], cell: Callable[[str, str], str]) -> None:
        out.append(title)
        out.append("")
        out.append("| case | " + " | ".join(arms_order) + " |")
        out.append("|---" * (len(arms_order) + 1) + "|")
        for cid in ids:
            out.append(f"| {cid} | " + " | ".join(cell(a, cid) for a in arms_order) + " |")
        out.append("")

    tool_ids: list[str] = []
    for a in arms_order:
        for tool in TOOL_NAMES:
            for cid in ((arm(a).get("tools") or {}).get(tool) or {}).get("per_case") or {}:
                if cid not in tool_ids:
                    tool_ids.append(cid)

    def tool_cell(a: str, cid: str) -> str:
        for tool in TOOL_NAMES:
            row = (((arm(a).get("tools") or {}).get(tool) or {}).get("per_case") or {}).get(cid)
            if row:
                return f"{row['spoken']}/{row['routed']}"
        return "-"

    per_case_table("## Per case: spoken k / routed of 5 (tool cases)", tool_ids, tool_cell)

    other_ids: list[str] = []
    for a in arms_order:
        for cid in arm(a).get("no_tool_cases") or {}:
            if cid not in other_ids:
                other_ids.append(cid)

    def other_cell(a: str, cid: str) -> str:
        row = (arm(a).get("no_tool_cases") or {}).get(cid)
        return f"{row['false_tool']}/{row['trials']}" if row else "-"

    per_case_table("## Per case: false tools of 5 (no-tool and plain cases)", other_ids, other_cell)

    q12_ids: list[str] = []
    for a in arms_order:
        for cid in arm(a).get("q12_per_case") or {}:
            if cid not in q12_ids:
                q12_ids.append(cid)

    def q12_cell(a: str, cid: str) -> str:
        row = (arm(a).get("q12_per_case") or {}).get(cid)
        return f"{row['k']}/{row['n']}" if row else "-"

    per_case_table("## Per case: Q12 surface shown k / 5", q12_ids, q12_cell)
    return "\n".join(out)


def _render_markdown_103(report: Mapping[str, Any]) -> str:
    """report.md for a T-0103 run (criterion 7): the outcome, R1/R3, the decision table P1-P11 with the scaled bars
    as k/n, the numbers behind P1-P7, the UI channel, the per-case Q12 table and the repeat-block split."""
    arms_order = ARMS_103
    arms: Mapping[str, Any] = report.get("arms") or {}
    regime = report.get("regime") or {}
    readings = report.get("readings") or {}
    thresholds = report.get("thresholds") or {}

    def arm(a: str) -> Mapping[str, Any]:
        return arms.get(a) or {}

    def type_spoken(a: str) -> Any:
        return ((arm(a).get("tools") or {}).get("type_at") or {}).get("spoken")

    out = ["# T-0103 C3 confirmation (dev, A0 vs C3): report", ""]
    out.append(
        "Regime: "
        + (", ".join(f"{k}={v}" for k, v in regime.items()) if regime else "(no manifest)")
    )
    out.append("")
    out.append(
        f"Table: {report.get('table')}; rule: {report.get('rule')}; trials: {report.get('trials')}; "
        f"complete: {report.get('complete')}"
    )
    for note in report.get("notes") or []:
        out.append(f"- note: {note}")
    out.append("")
    out.append(f"**Outcome: {report.get('outcome')}** (winner: {report.get('winner')})")
    out.append("")
    out.append(f"Void arms (> 2 % errored): {', '.join(report.get('void_arms') or []) or 'none'}")
    out.append("")

    r3 = thresholds.get("R3") or {}
    out.append("## Controls")
    out.append("")
    out.append(
        f"- R1 (A0 `type_at` spoken <= 3/10): {_r(type_spoken('A0'))}: "
        + ("holds" if readings.get("R1") else "fails: finding not reproduced")
    )
    out.append(
        f"- R3 (A0 Q12 positive shown >= {r3.get('min')}): {_r(arm('A0').get('q12_positive_shown'))}: "
        + ("holds" if readings.get("R3") else "fails: UI baseline not reproduced")
    )
    out.append("")

    p = (report.get("candidates") or {}).get("C3") or {}
    p8, p10, p11 = (thresholds.get(k) or {} for k in ("P8_total", "P10", "P11"))
    p8c = thresholds.get("P8_per_case") or {}
    bars = {
        "P8": f"total >= A0 - {p8.get('margin')} = {p8.get('bar')}; protected (A0 >= {p8c.get('protected_min')}) "
        f"cases {', '.join(p8c.get('protected') or []) or 'none'} each >= {p8c.get('protected_min')}",
        "P9": "<= A0 + 1 of 80 where none belongs, <= A0 + 2 of 200 tool trials (verbatim)",
        "P10": f"<= A0 + {p10.get('margin')} = {p10.get('bar')}",
        "P11": f"<= A0 + {p11.get('margin')} = {p11.get('bar')}",
    }
    out.append("## Decision (criterion 5, C3 against A0 in this run)")
    out.append("")
    out.append("| condition | C3 | bar |")
    out.append("|---|---|---|")
    for k in PS_C:
        out.append(f"| {k} | {_yes(p.get(k))} | {bars.get(k, 'verbatim from T-0098c')} |")
    out.append(f"| void | {p.get('void')} | |")
    out.append("")

    out.append("## The numbers behind P1-P7 (k/n; spoken, valid and exact over routed tool trials)")
    out.append("")
    out.append(
        "| arm | type spoken | type majority cases | click+scroll+key spoken | route recall | type recall | valid | "
        "exact | false tools (no-tool + plain) | anomalies | thought_open (tool trials) | errored |"
    )
    out.append("|---" * 12 + "|")
    for a in arms_order:
        x = arm(a)
        t = (x.get("tools") or {}).get("type_at") or {}
        out.append(
            f"| {a} | {_r(t.get('spoken'))} | {t.get('majority_cases')} | {_r(x.get('csk_spoken'))} | "
            f"{_r(x.get('route_recall'))} | {_r(t.get('routed'))} | {_r(x.get('valid'))} | {_r(x.get('exact'))} | "
            f"{_r(x.get('false_tools'))} | {x.get('anomalies')} | {_r(x.get('thought_open'))} | "
            f"{_r(x.get('errored'))} |"
        )
    out.append("")

    out.append("## The UI channel (P8-P11)")
    out.append("")
    out.append(
        "| arm | Q12 positive shown | Q12 negative shown | surfaces where none belongs | surfaces on tool trials | "
        "placement failures (Q12 positive) | tool calls on Q12 | every ui valid (gated trials) | "
        "ui.gate gated / emitted / failed |"
    )
    out.append("|---" * 9 + "|")
    for a in arms_order:
        x = arm(a)
        g = x.get("ui_gate") or {}
        out.append(
            f"| {a} | {_r(x.get('q12_positive_shown'))} | {_r(x.get('q12_negative_shown'))} | "
            f"{_r(x.get('surfaces_no_belong'))} | {_r(x.get('surfaces_tool'))} | "
            f"{_r(x.get('placement_failures_q12_positive'))} | {_r(x.get('q12_tool_calls'))} | "
            f"{_r(x.get('every_ui_valid'))} | {g.get('gated')} / {g.get('emitted')} / {g.get('failed')} |"
        )
    out.append("")

    out.append("## Per positive case: Q12 surface shown k / 40")
    out.append("")
    out.append(
        "| case | " + " | ".join(arms_order) + f" | protected (A0 >= {p8c.get('protected_min')}) |"
    )
    out.append("|---" * (len(arms_order) + 2) + "|")
    for cid, row in (report.get("positive_cases") or {}).items():
        out.append(
            f"| {cid} | "
            + " | ".join(_r(row.get(a)) for a in arms_order)
            + f" | {'yes' if row.get('protected') else 'no'} |"
        )
    out.append("")

    out.append("## Q12 positive shown by repeat block (reported, not gated)")
    out.append("")
    blocks = report.get("q12_positive_shown_by_block") or {}
    names = list(next(iter(blocks.values()), {}) or {})
    out.append("| arm | " + " | ".join(names) + " |")
    out.append("|---" * (len(names) + 1) + "|")
    for a in arms_order:
        out.append(f"| {a} | " + " | ".join(_r((blocks.get(a) or {}).get(n)) for n in names) + " |")
    out.append("")

    out.append("## Per arm and tool")
    out.append("")
    out.append(
        "| arm | tool | spoken/routed | routed/trials | valid | exact | thought_open | anomalies |"
    )
    out.append("|---" * 8 + "|")
    for a in arms_order:
        for tool in TOOL_NAMES:
            t = (arm(a).get("tools") or {}).get(tool) or {}
            out.append(
                f"| {a} | {tool} | {_r(t.get('spoken'))} | {_r(t.get('routed'))} | {_r(t.get('valid'))} | "
                f"{_r(t.get('exact'))} | {_r(t.get('thought_open'))} | {t.get('anomalies')} |"
            )
    out.append("")
    return "\n".join(out)


def _render_markdown_108a(report: Mapping[str, Any]) -> str:
    """report.md for a T-0108a run (criterion 7): the outcome, R3/R4, the decision table with its bars as k/n, the
    complete / last-OK / shown counts per arm and per positive case, the pruned kinds, the tool numbers, the UI
    channel, the speech rates (reported, not gated) and the U' sizes."""
    arms_order = ARMS_108A
    arms: Mapping[str, Any] = report.get("arms") or {}
    regime = report.get("regime") or {}
    readings = report.get("readings") or {}
    th = report.get("thresholds") or {}

    def arm(a: str) -> Mapping[str, Any]:
        return arms.get(a) or {}

    out = ["# T-0108a required-fields prompt A/B (dev, A0 vs K1, K2): report", ""]
    out.append(
        "Regime: "
        + (", ".join(f"{k}={v}" for k, v in regime.items()) if regime else "(no manifest)")
    )
    out.append("")
    out.append(
        f"Table: {report.get('table')}; rule: {report.get('rule')}; trials: {report.get('trials')}; "
        f"complete: {report.get('complete')}"
    )
    for note in report.get("notes") or []:
        out.append(f"- note: {note}")
    out.append("")
    out.append(f"**Outcome: {report.get('outcome')}** (winner: {report.get('winner')})")
    out.append("")
    out.append(f"Void arms (> 2 % errored): {', '.join(report.get('void_arms') or []) or 'none'}")
    out.append("")

    r3, r4 = th.get("R3") or {}, th.get("R4") or {}
    out.append("## Controls")
    out.append("")
    out.append(
        f"- R3 (A0 Q12 positive shown >= {r3.get('min')}): {_r(arm('A0').get('q12_positive_shown'))}: "
        + ("holds" if readings.get("R3") else "fails: UI baseline not reproduced")
    )
    out.append(
        f"- R4 (A0 Q12 positive complete <= {r4.get('max')}): {_r(arm('A0').get('q12_positive_complete'))}: "
        + ("holds" if readings.get("R4") else "fails: degradation not reproduced")
    )
    out.append("")

    def prot(key: str) -> str:
        return ", ".join((th.get(key) or {}).get("protected") or []) or "none"

    q1, u1, u2 = (th.get(k) or {} for k in ("Q1", "U1", "U2"))
    p9, p10, p11 = (th.get(k) or {} for k in ("P9", "P10", "P11"))
    pmin = (th.get("Q2") or {}).get("protected_min")
    bars = {
        "Q1": f"complete >= A0 + {q1.get('margin')} = {q1.get('bar')}",
        "Q2": f"complete >= {pmin} on each protected case (A0 >= {pmin}): {prot('Q2')}",
        "U1": f"last-OK >= A0 - {u1.get('margin')} = {u1.get('bar')}; >= {pmin} on {prot('U1')}",
        "U2": f"shown >= A0 - {u2.get('margin')} = {u2.get('bar')}; >= {pmin} on {prot('U2')}",
        "P9": f"<= A0 + {p9.get('no_belong_over_A0')} where none belongs, <= A0 + {p9.get('tool_over_A0')} tool "
        "trials (verbatim)",
        "P10": f"<= A0 + {p10.get('margin')} = {p10.get('bar')}",
        "P11": f"<= A0 + {p11.get('margin')} = {p11.get('bar')}",
        "P6": "0 voiced anomalies over the arm's trials",
    }
    cands: Mapping[str, Any] = report.get("candidates") or {}
    out.append("## Decision (criterion 5, each K against A0 in this run)")
    out.append("")
    out.append("| condition | " + " | ".join(CANDIDATES_108A) + " | bar |")
    out.append("|---" * (len(CANDIDATES_108A) + 2) + "|")
    for k in PS_108A:
        cells = " | ".join(_yes((cands.get(c) or {}).get(k)) for c in CANDIDATES_108A)
        out.append(f"| {k} | {cells} | {bars.get(k, 'verbatim from T-0098')} |")
    out.append(
        "| void | "
        + " | ".join(str((cands.get(c) or {}).get("void")) for c in CANDIDATES_108A)
        + " | |"
    )
    out.append("")

    out.append("## Complete cards (Q12 positive trials)")
    out.append("")
    out.append(
        "| arm | complete | last-OK | shown | as written, clean (reported) | last canvas holds a forward ref "
        "(reported) | pruned kinds (trials) |"
    )
    out.append("|---" * 7 + "|")
    for a in arms_order:
        x = arm(a)
        kinds = (
            ", ".join(f"{k} {n}" for k, n in (x.get("q12_pruned_kinds") or {}).items()) or "none"
        )
        out.append(
            f"| {a} | {_r(x.get('q12_positive_complete'))} | {_r(x.get('q12_positive_last_ok'))} | "
            f"{_r(x.get('q12_positive_shown'))} | {_r(x.get('q12_positive_as_written_clean'))} | "
            f"{_r(x.get('q12_positive_forward_refs'))} | {kinds} |"
        )
    out.append("")

    protected = report.get("protected") or {}
    for title, key, flag in (
        ("complete", "complete", "Q2"),
        ("last-OK", "last_ok", "U1"),
        ("shown", "shown", "U2"),
    ):
        out.append(f"## Per positive case: {title} k / 40")
        out.append("")
        out.append("| case | " + " | ".join(arms_order) + f" | protected for {flag} |")
        out.append("|---" * (len(arms_order) + 2) + "|")
        for cid, row in (report.get("positive_cases") or {}).items():
            cells = " | ".join(_r((row.get(key) or {}).get(a)) for a in arms_order)
            mark = "yes" if (protected.get(flag) or {}).get(cid) else "no"
            out.append(f"| {cid} | {cells} | {mark} |")
        out.append("")

    out.append("## Pruned kinds per positive case (trials; reported, not gated)")
    out.append("")
    out.append("| case | " + " | ".join(arms_order) + " |")
    out.append("|---" * (len(arms_order) + 1) + "|")
    for cid in report.get("positive_cases") or {}:
        cells = []
        for a in arms_order:
            kinds = (arm(a).get("q12_pruned_kinds_per_case") or {}).get(cid) or {}
            cells.append(", ".join(f"{k} {n}" for k, n in kinds.items()) or "none")
        out.append(f"| {cid} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## The numbers behind P3-P7 (k/n; valid and exact over routed tool trials)")
    out.append("")
    out.append(
        "| arm | route recall | type recall | valid | exact | false tools (no-tool + plain) | anomalies | "
        "thought_open (tool trials) | errored |"
    )
    out.append("|---" * 9 + "|")
    for a in arms_order:
        x = arm(a)
        t = (x.get("tools") or {}).get("type_at") or {}
        out.append(
            f"| {a} | {_r(x.get('route_recall'))} | {_r(t.get('routed'))} | {_r(x.get('valid'))} | "
            f"{_r(x.get('exact'))} | {_r(x.get('false_tools'))} | {x.get('anomalies')} | "
            f"{_r(x.get('thought_open'))} | {_r(x.get('errored'))} |"
        )
    out.append("")

    out.append("## The UI channel (U2, P9-P11)")
    out.append("")
    out.append(
        "| arm | Q12 positive shown | Q12 negative shown | surfaces where none belongs | surfaces on tool trials | "
        "placement failures (Q12 positive) | tool calls on Q12 | ui.gate gated / emitted / failed |"
    )
    out.append("|---" * 8 + "|")
    for a in arms_order:
        x = arm(a)
        g = x.get("ui_gate") or {}
        out.append(
            f"| {a} | {_r(x.get('q12_positive_shown'))} | {_r(x.get('q12_negative_shown'))} | "
            f"{_r(x.get('surfaces_no_belong'))} | {_r(x.get('surfaces_tool'))} | "
            f"{_r(x.get('placement_failures_q12_positive'))} | {_r(x.get('q12_tool_calls'))} | "
            f"{g.get('gated')} / {g.get('emitted')} / {g.get('failed')} |"
        )
    out.append("")

    speech = report.get("speech") or {}
    out.append(f"## Speech before a tool call ({SPEECH_LABEL})")
    out.append("")
    out.append("| arm | spoken / routed (all tools) | `type_at` spoken / routed |")
    out.append("|---|---|---|")
    for a in arms_order:
        s = (speech.get("arms") or {}).get(a) or {}
        out.append(f"| {a} | {_r(s.get('spoken'))} | {_r(s.get('type_at_spoken'))} |")
    out.append("")

    u_prime = report.get("u_prime") or {}
    out.append("## U' (the UI block each arm sends)")
    out.append("")
    out.append("| arm | chars | engine tokens | sha256 |")
    out.append("|---|---|---|---|")
    for a in arms_order:
        u = u_prime.get(a) or {}
        out.append(
            f"| {a} | {u.get('chars', '-')} | {u.get('engine_tokens', '-')} | {u.get('sha256', '-')} |"
        )
    out.append("")
    return "\n".join(out)


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


def _tokenize_text(base_url: str, model: str, text: str) -> int | None:
    """The engine's token count for ``text``: ``POST /tokenize`` at the server root in text form,
    ``add_special_tokens: false`` (as ``VllmClient.tokenize`` sends it), through T-0093's unproxied, non-redirecting
    opener; ``None`` on any failure (recorded, never guessed). Reading it spends no inference."""
    import http.client
    import urllib.request

    request = urllib.request.Request(
        _T93._root(base_url) + "/tokenize",
        data=json.dumps({"model": model, "prompt": text, "add_special_tokens": False}).encode(
            "utf-8"
        ),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with _T93._OPENER.open(request, timeout=60.0) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    ids = _T93._id_list(payload.get("tokens") if isinstance(payload, dict) else None)
    return None if ids is None else len(ids)


def _make_tee() -> Any:
    """An ``httpx`` transport over an explicit ``httpx.AsyncHTTPTransport()`` (no proxy from the environment) that
    keeps each request's body and the raw bytes of its response, so the trial records the ids the engine streamed
    without reaching into the client."""
    import httpx

    class _TeeStream(httpx.AsyncByteStream):
        def __init__(self, inner: httpx.AsyncByteStream, sink: bytearray) -> None:
            self._inner = inner
            self._sink = sink

        async def __aiter__(self) -> AsyncIterator[bytes]:
            async for chunk in self._inner:
                self._sink.extend(chunk)
                yield chunk

        async def aclose(self) -> None:
            await self._inner.aclose()

    class _Tee(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self._inner = httpx.AsyncHTTPTransport()
            self.request_body: bytes = b""
            self.response_bytes = bytearray()

        def reset(self) -> None:
            self.request_body = b""
            self.response_bytes = bytearray()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            try:
                self.request_body = request.content
            except httpx.RequestNotRead:
                self.request_body = b""
            response = await self._inner.handle_async_request(request)
            stream = response.stream
            assert isinstance(stream, httpx.AsyncByteStream)
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_TeeStream(stream, self.response_bytes),
                extensions=response.extensions,
            )

        async def aclose(self) -> None:
            await self._inner.aclose()

        def token_ids(self) -> list[int]:
            ids: list[int] = []
            for line in bytes(self.response_bytes).decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    item = json.loads(payload)
                except ValueError:
                    continue
                choices = item.get("choices") if isinstance(item, dict) else None
                first = choices[0] if isinstance(choices, list) and choices else None
                got = first.get("token_ids") if isinstance(first, dict) else None
                if isinstance(got, list):
                    ids.extend(int(t) for t in got if type(t) is int)
            return ids

        def sent_system(self) -> str | None:
            try:
                body = json.loads(self.request_body or b"null")
            except ValueError:
                return None
            if not isinstance(body, dict) or not body.get("messages"):
                return None
            content = body["messages"][0].get("content")
            if isinstance(content, str):
                return content
            return "".join(str(p.get("text", "")) for p in content or [] if p.get("type") == "text")

    return _Tee()


@dataclass
class _Run:
    out: Path
    client: VllmClient
    tee: Any
    manifest: dict[str, Any]
    errors_in_a_row: int = 0
    written: int = 0
    table: str = T0098


async def _trial(
    run: _Run, case: Any, case_index: int, repeat: int, slot: int, arm: str, prompt: Prompt
) -> bool:
    """One request, recorded; ``False`` when it errored."""
    import httpx

    from local_astra.conductor.llm_client import LlmWireError
    from local_astra.contracts.llm import LlmOptions

    table = TABLES[run.table]
    ui_channel = table.ui_channel
    request_id = f"{table.request_prefix}-r{repeat}-{case.case_id}-{arm}"
    commits: list[dict[str, Any]] = []
    error: str | None = None
    first: float | None = None
    run.tee.reset()
    started = time.perf_counter()
    try:
        # the session's pass 1 exactly: LlmOptions(), no xargs (session.py _run_generate)
        async for c in run.client.generate(prompt, request_id, LlmOptions(), xargs=None):
            if first is None:
                first = time.perf_counter()
            commits.append(
                {
                    "segment": c.segment,
                    "text": c.text,
                    "thought_open": c.thought_open,
                    "canvas_idx": c.canvas_idx,
                    "stop_id": c.stop_id,
                }
            )
    except (httpx.HTTPError, httpx.StreamError, LlmWireError, OSError, ValueError) as exc:
        error = repr(exc)
    ended = time.perf_counter()
    sent = run.tee.sent_system()
    system = sent if sent is not None else prompt.messages[0].text()
    line = {
        "case_id": case.case_id,
        "kind": case.kind,
        "tool": (case.gt or {}).get("name") if case.kind == "tool" else None,
        "gt": case.gt,
        "case_index": case_index,
        "arm": arm,
        "repeat": repeat,
        "order": slot,
        "request_id": request_id,
        "system_sha256": _sha(system),
        "commits": commits,
        "token_ids": run.tee.token_ids(),
        "error": error,
        "wall_ms": round((ended - started) * 1000.0, 1),
        "first_canvas_ms": None if first is None else round((first - started) * 1000.0, 1),
    }
    if ui_channel and case.kind == "q12":
        line["expect_surface"] = case.expect_surface
    with (run.out / "trials.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    run.written += 1
    reading = classify(commits)
    shown = ""
    if ui_channel:
        shown = f" shown={ui_reading(commits, request_id=request_id, cards=False)['shown']}"
    _say(
        f"r{repeat} {case_index:02d} {case.case_id} [{arm} slot {slot}] routed={reading['routed']} "
        f"spoken={reading['spoken']} open={reading['thought_open']}{shown} wall={line['wall_ms']}ms"
        + (f" ERROR {error}" if error else "")
    )
    return error is None


async def _run_all(
    run: _Run, cases: Sequence[Any], prompts: Mapping[str, Mapping[str, Prompt]]
) -> None:
    """Criterion 3: the table's ``schedule`` -- repeats outermost, the cases scheduled for each repeat in order, the
    table's arms rotated per (case, repeat) (``arm_order``); concurrency 1."""
    try:
        for item in schedule(run.table, cases):
            case, arm, repeat = cases[item.case_index], item.arm, item.repeat
            ok = await _trial(
                run, case, item.case_index, repeat, item.order, arm, prompts[case.case_id][arm]
            )
            run.errors_in_a_row = 0 if ok else run.errors_in_a_row + 1
            if run.errors_in_a_row >= STOP_AFTER_ERRORS:
                run.manifest["stopped"] = (
                    f"{run.errors_in_a_row} consecutive requests errored "
                    f"(last: repeat {repeat}, {case.case_id} [{arm}])"
                )
                _say(f"stopping: {run.manifest['stopped']}")
                return
    finally:
        await run.client.aclose()


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--split", default=SPLIT, help="only 'dev' runs; anything else is refused")
    ap.add_argument(
        "--table",
        default=T0098,
        choices=tuple(TABLES),
        help="the arm table (T-0098c criterion 1); default T-0098",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    # --- refusals, before any socket ------------------------------------------------------------------
    try:
        _T93.check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: not a loopback URL: {exc}")
        return EXIT_REFUSED
    try:
        dirty = _T93._git("status", "--porcelain").rstrip("\n")
        sha = _T93._git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    table: ArmTable = TABLES[args.table]
    ui_channel = table.ui_channel
    out: Path = args.out_dir or REPO / "logs" / f"{table.out_stem}-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
        return EXIT_REFUSED
    if args.split != SPLIT:
        _say(
            f"refusing: --split {args.split!r}: only the {SPLIT} split runs (the test split is never read)"
        )
        return EXIT_REFUSED
    try:
        cases = load_cases(table=table.task)
    except (OSError, ValueError, KeyError) as exc:
        what = "cases.jsonl / surface_placement_cases.jsonl (Q12)" if ui_channel else "cases.jsonl"
        _say(f"refusing: {what}: {exc}")
        return EXIT_REFUSED
    if table.repeats is not None:
        design = _design_repeats(
            {str(c.case_id): c.kind == "q12" and bool(c.expect_surface) for c in cases},
            table.positive_repeats,
        )
        if dict(table.repeats) != design:
            diff = sorted(
                f"{cid}: {table.repeats.get(cid)} (design {design.get(cid)})"
                for cid in set(design) | set(table.repeats)
                if table.repeats.get(cid) != design.get(cid)
            )
            _say(
                f"refusing: the repeats map is not {table.positive_repeats} on the {Q12_POSITIVE_N} Q12 positive "
                f"cases and {REPEATS} on the other {len(design) - Q12_POSITIVE_N} ({table.task} *Design*): "
                + "; ".join(diff)
            )
            return EXIT_REFUSED

    from local_astra import cli
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
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
    if kit.tools_version[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {kit.tools_version[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED
    try:
        texts = arm_texts(kit.tools, table=table.task)
    except ValueError as exc:
        # when the arms cannot be built, say too whether A0 itself drifted (a changed U moves both)
        live_a0 = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools).system_message().text()
        drift = (
            ""
            if _sha(live_a0) == table.a0_sha256
            else f"; and the A0 system text has sha256 {_sha(live_a0)}, not {table.a0_sha256}"
        )
        _say(f"refusing: {exc}{drift}")
        return EXIT_REFUSED
    a0_sha = _sha(texts["A0"])
    if a0_sha != table.a0_sha256:
        _say(
            f"refusing: the A0 system text has sha256 {a0_sha}, not {table.a0_sha256} (the live prompt drifted "
            "since 4b0bd92, so this run would not measure the finding)"
        )
        return EXIT_REFUSED
    if table.text_problems is not None:
        text_problems = table.text_problems(texts)
        if text_problems:
            _say(f"refusing: {'; '.join(text_problems)}")
            return EXIT_REFUSED
    try:
        endpoint = VllmEndpoint(base_url=args.base_url)
    except ValueError as exc:
        _say(f"refusing: not a loopback URL the client accepts: {exc}")
        return EXIT_REFUSED
    tee = _make_tee()
    client = VllmClient(endpoint, transport=tee, detokenizer=detokenizer)
    thinking = bool(profile.diffusion.thinking)
    prompts = {c.case_id: arm_prompts(c, kit, thinking=thinking, table=table.task) for c in cases}
    for case in cases:
        bodies = {a: client.request_body(p, "fidelity") for a, p in prompts[case.case_id].items()}
        problems = check_fidelity(bodies)
        if problems:
            _say(f"refusing: criterion 2 does not hold for {case.case_id}: {'; '.join(problems)}")
            return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    listing = _T93._get_text(args.base_url.rstrip("/") + "/models")
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

    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": table.task,
        "table": table.task,
        "rule": table.rule,
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": endpoint.model,
            "engine_version": _T93._get_text(_T93._root(args.base_url) + "/version"),
            "versions_lock_sha256": _T93._sha256(REPO / "services" / "versions.lock"),
            "cases_sha256": CASES_SHA256,
            "split": SPLIT,
            "cases": len(cases),
            "tools": f"native:{kit.tools_version[:12]}",
            "repair": "on",
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": thinking,
            "arms": list(ARMS),
            "rotation": "per (case, repeat): arms from offset (case_index + repeat) mod 6 of the arm order",
            "repeats": REPEATS,
            "seed": None,
            "concurrency": 1,
            "pass": "pass 1 only: no prefill, no repair, no continuation; LlmOptions() as the session sends it",
            "input": "typed text over the fixture frame; no Session, no audio, no memory hits, no assists",
        },
        "arms": {
            a: {"text": t, "sha256": _sha(t), "role": table.roles[a]} for a, t in texts.items()
        },
        "cases": [{"case_id": c.case_id, "kind": c.kind, "index": i} for i, c in enumerate(cases)],
    }
    if ui_channel:
        manifest["regime"].update(
            {
                "q12_sha256": Q12_SHA256,
                "case_sources": list(table.case_sources),
                "arms": list(table.arms),
                "rotation": (
                    "per (case, repeat): arms from offset (case_index + repeat) "
                    f"mod {len(table.arms)} of the arm order"
                ),
                "input": (
                    "typed text; tool cases over the fixture frame, no-tool, plain and Q12 cases with no frame (the "
                    "[screen] note); no Session, no audio, no assists; flight_booking_with_memory carries its "
                    "captions as memory hits as T-0093's _CaseMemory serves them, with no now_ms (as the session)"
                ),
                "surface_shown": (
                    "a fresh UiGate() per trial gates every ui commit in commit order with final=False and the "
                    "request id as turn_id (session._gate_surface); shown = any call returned a surface"
                ),
            }
        )
        manifest["cases"] = [
            {"case_id": c.case_id, "kind": c.kind, "index": i}
            | ({"expect_surface": c.expect_surface} if c.kind == "q12" else {})
            for i, c in enumerate(cases)
        ]
    if table.repeats is not None:
        manifest["regime"].update(
            {
                "repeats": dict(table.repeats),
                "requests": len(schedule(table.task, cases)),
                "schedule": (
                    "repeats outermost; within a repeat the cases scheduled for it (repeat < the case's repeats) "
                    "in index order; per (case, repeat) the arms back to back in the rotation"
                ),
                "thresholds": dict(table.thresholds or {}),
                "reading": table.reading,
            }
        )
    if table.complete_cards:
        # T-0108a criterion 1: each arm's U' with its engine token count (root /tokenize, before the first
        # request), and the files that decide a complete card
        manifest["u_prime"] = {
            a: {
                "chars": len(u),
                "sha256": _sha(u),
                "engine_tokens": _tokenize_text(args.base_url, endpoint.model, u),
            }
            for a, u in _u_primes(texts).items()
        }
        manifest["gate_files_sha256"] = {p: _T93._sha256(REPO / p) for p in GATE_FILES}
    _write_json(out / "manifest.json", manifest)
    run = _Run(out=out, client=client, tee=tee, manifest=manifest, table=table.task)
    asyncio.run(_run_all(run, cases, prompts))
    manifest["trials_written"] = run.written
    _write_json(out / "manifest.json", manifest)

    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _say(
        f"readings: {report['readings']}; winner: {report['winner']}; void arms: {report['void_arms']}"
    )
    if "outcome" in report:
        _say(f"outcome: {report['outcome']}")
    _say(f"written to {out}")
    if manifest.get("stopped") or not report["complete"]:
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
