"""T-0115 -- the GPU probe of the multi-step tool-result loop, through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0115_loop_probe.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the scenario table, ``ScenarioSearch``, the rule
(``evaluate``), the report (``build_report``, ``render_markdown``) and the record extraction (``turn_records``) are
pure, and everything that talks to the engine (``httpx``, ``VllmClient``, ``Session``) is imported inside ``main``.

Refusals (exit 2), in this order and all **before any socket** (criterion 2):

1. a ``--base-url`` that is not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (``localhost`` is a name and is
   refused; nothing is resolved);
2. a dirty tree (the dirty paths are named);
3. an existing, non-empty ``--out-dir``;
4. a tool kit that ``cli.load_tool_kit`` cannot load, whose tool set is not ``bbfc53ccdd80``, or whose repair is off;
5. a scenario table whose sha (``scenario_table_sha()``) is not ``SCENARIO_SHA256`` (the pin of
   ``tests/unit/test_t0115_loop_probe.py``);
6. a scenario case id that is not a dev-split case of ``tests/eval/toolcall/cases.jsonl`` (the id is named);
7. ``--max-tool-steps`` other than 8 (H-T115-7);
8. a ``tests/eval/toolcall/cases.jsonl`` that is not the pinned one (T-0087's sha).

What it runs (docs/tasks/T-0115.md): the 118 turns of ``schedule()`` (S0 8, S1-S5 20 each, S6 10), concurrency 1,
unseeded, in one boot. Each turn is one fresh ``Session`` with one typed turn (``UserText``) and
``TurnPolicy(tool_loop=True, max_tool_steps=8)``: the real ``VllmClient`` (unproxied, loopback) behind ``LoopMeter``;
the kit from ``cli.load_tool_kit`` (dry run, repair on) with ``search`` replaced by the turn's own ``ScenarioSearch``;
STT, TTS (300 ms/word on the session clock), audio and perception are ``tests/fakes`` (the turn's screen is its one
frame); memory is ``_CaseMemory`` (no recollections, never an uncaptioned episode, so the idle captioner sends no
request and every ``/metrics`` delta is the generate's own at concurrency 1).

``LoopMeter`` (T-0093's ``_MeteredLlm`` reshaped) reads ``/metrics`` before and after every generate, outside the
timed span, and times first commit and done from the moment the request is sent, on its own monotonic clock. Step 1
waits for its anticipatory prefill first (prefill aborts are deferred, J12 as in T-0093); steps send no prefill.

Output (``--out-dir``, default ``logs/t0115-loop-probe-<sha7>/``): ``trials.jsonl`` (one line per turn),
``steps.jsonl`` (one line per generate, the repair excluded: it is counted on the trial), ``queries.jsonl`` (every
search query, in call order), ``turns.jsonl`` and ``sessions/*.jsonl`` (the sessions' own records),
``manifest.json`` (the regime, the case/screen/table/gate-file shas, the bounds), ``report.json`` and
``report.md``. Exit codes: 0 the run was measured (outcome ``ready`` or ``not ready``); 1 the outcome is
``inconclusive`` or the engine is not there; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import http.client
import importlib
import ipaddress
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_astra.contracts.toolcall import SearchResult, SearchResults, ToolInvocation, ToolOutcome
from local_astra.toolcall import live, loop, router
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

if TYPE_CHECKING:
    from local_astra.config import Profile
    from local_astra.contracts.cd import CdRequest
    from local_astra.contracts.llm import Commit, LlmOptions, PrefillHandle, Prompt
    from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
CASES_PATH = SET_DIR / "cases.jsonl"
#: sha256 of ``tests/eval/toolcall/cases.jsonl`` (as T-0087 pinned it)
CASES_SHA256 = "460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85"

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
TOOLS_SHA12 = "bbfc53ccdd80"
#: H-T115-7: the product default; any other value is refused
MAX_TOOL_STEPS = 8
#: the prompt budget the step prompts are measured against (G7; H-T114-4)
MAX_PROMPT_TOKENS = 14_336
METRIC_QUERIES = "vllm:prefix_cache_queries_total"
METRIC_HITS = "vllm:prefix_cache_hits_total"
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"
OUTCOMES = ("final", "call", "refused", "repairable-refused", "thought-open", "empty")
#: criterion 5: the gate files whose sha256 pins a run to its commit (the eval row skips on drift)
GATE_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/turn_machine.py",
)
#: the session's note when a continuation is over ``MAX_CONTINUATION`` (``Session._run_step``)
OVER_BUDGET_NOTE = "tool loop switched off: the continuation is over budget"
#: ``Session._judge_tool``'s reason for a repairable call at a step >= 2
REPAIR_STEP1_ONLY = "repair: step 1 only"
#: the prefix-cache block (T-0096 ruling M5); the per-step predicted hit is floor(prev / 32) * 32
PREFIX_BLOCK_TOKENS = 32
#: ``<tool_response|>``'s id in the pinned tokenizer (Q1: a step prompt's last token)
TOOL_RESPONSE_CLOSE_ID = 51
#: chat-form ``/tokenize`` keeps exactly these keys of ``VllmClient.request_body`` (T-0093)
TOKENIZE_KEEP = (
    "model",
    "messages",
    "tools",
    "chat_template_kwargs",
    "add_generation_prompt",
    "continue_final_message",
)
#: V: a turn not IDLE within this wall time is errored
TURN_TIMEOUT_S = 120.0
PREFILL_WAIT_S = 60.0
QUIESCE_S = 30.0
UNREACHABLE_AFTER_S = 3.0
UNREACHABLE_STOP = 3
#: B3 (orchestrator, 2026-10-17): this many consecutive errored or timed-out turns stop the run (the rest not run)
ERRORED_STOP = 3
THOUGHT_MARKERS = ("<|channel>", "<channel|>", "<|think|>")

# --- *Pre-registered rules*, exact -------------------------------------------------------------------------------
VOID_AT = Fraction(3, 118)
C1_MIN = Fraction(32, 40)
C2_MIN = 60
G1_MIN = Fraction(9, 10)
G2_MIN = Fraction(16, 20)
G3_MAX = Fraction(10, 100)
G4_EXACT_MAX = Fraction(1, 10)
G4_S5_MAX = Fraction(4, 20)
G5_S1_MIN = Fraction(16, 20)
G5_S2_MIN = Fraction(14, 20)
BARS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
#: report-only: a step's prefix hit is flagged below this share of its predicted floor
PREFIX_FLAG_SHARE = Fraction(9, 10)


def _say(msg: str) -> None:
    print(f"t0115_loop_probe: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The scenario table (criterion 4; pre-registered in docs/cd/toolcall-loop-probe.md)
# =====================================================================================================


@dataclass(frozen=True)
class ScenarioPrompt:
    """One typed turn: its text, its screen (relative to ``tests/eval/toolcall``), the dev case the prompt or the
    screen comes from, ``ScenarioSearch``'s row sets and the invented fact a correct S1/S2 report contains."""

    prompt_id: str
    text: str
    screen: str
    case_id: str | None
    rows: tuple[SearchResults, ...] = ()
    fact: str | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    n: int
    prompts: tuple[ScenarioPrompt, ...]


def _r(title: str, url: str, snippet: str) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet)


def _rows(query: str, *results: SearchResult) -> SearchResults:
    return SearchResults(query=query, results=tuple(results))


#: the one fixed dev-split screen of S0-S2 and S6 (a notes editor; ``chat-08`` is the dev case it comes from)
SHARED_SCREEN = "screens/d08_editor.jpg"
SHARED_CASE = "chat-08"

#: S6's ten invented libraries, and what each is said to render a 10 000-point chart in (ms)
S6_LIBRARIES = (
    "Quiver",
    "Brindle",
    "Tamsel",
    "Orrin",
    "Vellum",
    "Kestrel",
    "Marrowby",
    "Pellucid",
    "Sorrel",
    "Tindle",
)
_S6_ASPECTS = (
    "benchmark",
    "documentation",
    "licence and governance",
    "bundle size",
    "accessibility",
)
_S6_FILLER = (
    "Reviewers note that the numbers were measured on a mid-range laptop with the default theme, no animations "
    "and a cold cache, and that the maintainers publish the harness so anyone can rerun it. The project ships "
    "weekly releases, keeps a changelog with migration notes, and answers most issues within two working days. "
)


def _s6_snippet(lib: str, i: int, j: int) -> str:
    """A 300-char invented snippet (exactly ``MAX_SNIPPET``): a maximal result."""
    ms = 40 + 13 * ((i * 7 + 3) % 10)
    head = (
        f"{lib} {_S6_ASPECTS[j]}, part {j + 1} of 5: {lib} renders a 10,000-point line chart in {ms} ms, "
        f"has {120 + 17 * i} contributors and was first released in {2011 + i}. "
    )
    text = (head + _S6_FILLER * 3)[:300]
    return text


def _s6_rows() -> tuple[SearchResults, ...]:
    """Ten maximal row sets (5 results x 300-char snippets), one per library, in search order."""
    return tuple(
        SearchResults(
            query=f"{lib} charting library",
            results=tuple(
                _r(
                    f"{lib} charting library - {_S6_ASPECTS[j]}",
                    f"https://{lib.lower()}-charts.example/{_S6_ASPECTS[j].split()[0]}",
                    _s6_snippet(lib, i, j),
                )
                for j in range(5)
            ),
        )
        for i, lib in enumerate(S6_LIBRARIES)
    )


#: S6/p2's text to type: 3 000 chars, invented
S6_TYPE_TEXT = (
    "Notes from the Halvercombe walking week. Day one began at the old toll house on Marrow Lane, where the "
    "warden, a patient man called Edwin Sallow, handed out paper maps and a list of the stiles that had been "
    "repaired since spring. We followed the river path north for eleven kilometres, past the flooded water "
    "meadows and the brick pumping station that the town still opens on the first Sunday of every month. Lunch "
    "was bread, a wedge of Tollbridge cheese and two apples, eaten on the steps of the chapel at Crewsby. In the "
    "afternoon the path climbed steadily through beech woods to the ridge, and from the trig point we could see "
    "the whole estuary, the ferry crossing to Halsey Island and, on a clear day, the lighthouse at Pennock Head. "
    "Day two was wetter. Rain arrived before breakfast and stayed until four, so we took the lower route through "
    "the orchards and stopped twice at farm stalls for tea. The second stall sold honey from hives kept on the "
    "old railway embankment, and the owner told us that the bees prefer the lime trees in July and the ivy in "
    "October. We reached the hostel at Wendle Cross soaked but cheerful, hung everything over the drying rail "
    "and spent the evening playing cards with a group of cyclists from the coast. Day three was the long one: "
    "twenty-three kilometres over the moor to the reservoir at Ashcombe, with only one shelter on the way, a "
    "stone hut with a tin roof and a visitors' book going back to 1978. Our entry reads: three walkers, two dry "
    "socks between them, morale excellent. The descent to the reservoir was steep and loose, and we were glad of "
    "the poles. Day four was a rest day in Ashcombe itself. We visited the small museum of rope making, bought "
    "new laces from the hardware shop on Quay Street and ate fish soup at the harbour cafe, which closes at "
    "three sharp whatever the weather. Day five took us back along the coast path, with the wind behind us for "
    "once, through the dunes at Sallowmouth and over the footbridge at Pinner's Gap, which had been rebuilt in "
    "larch since our last visit. We counted forty-one oystercatchers on the mudflats and one very confused heron. "
    "The final evening was spent at the Lantern Inn, where the landlord keeps a chalkboard of every walker who "
    "has finished the full circuit in under a week. Our names are now on it, in slightly smudged chalk, just "
    "below a family of five from Brenmoor who managed it in four days with a dog. Things to remember for next "
    "year: book the hostel at Wendle Cross by March, carry more water on the moor day, check the ferry timetable "
    "before planning the island detour, and pack the spare head torch, because the one we had gave out on the "
    "second night. On the way home we left the spare maps with the warden at the toll house, who promised to "
    "mark the new larch bridge on next year's edition and to add the honey stall, the tin-roofed hut and the "
    "harbour cafe's opening hours to the back page. Total distance walked: ninety-four kilometres. Total "
    "blisters: seven. Total regrets: none whatsoever."
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "S0",
        2,
        (
            ScenarioPrompt("p1", "How many legs does a spider have?", SHARED_SCREEN, SHARED_CASE),
            ScenarioPrompt("p2", "What is twelve plus thirty?", SHARED_SCREEN, SHARED_CASE),
            ScenarioPrompt(
                "p3", "Give me two words that rhyme with moon.", SHARED_SCREEN, SHARED_CASE
            ),
            ScenarioPrompt("p4", "Good morning! How are you today?", SHARED_SCREEN, SHARED_CASE),
        ),
    ),
    Scenario(
        "S1",
        10,
        (
            ScenarioPrompt(
                "p1",
                "Who founded Quillmark Bicycle Works in Tessdale? Look it up.",
                SHARED_SCREEN,
                SHARED_CASE,
                rows=(
                    _rows(
                        "Quillmark Bicycle Works founder",
                        _r(
                            "Quillmark Bicycle Works - Our story",
                            "https://quillmark-bicycles.example/about",
                            "Quillmark Bicycle Works was founded in Tessdale in 1923 by Orla Penhallow, a "
                            "former clockmaker who built her first frames in a converted dairy on Fenner Street.",
                        ),
                        _r(
                            "Tessdale heritage trail: the Quillmark works",
                            "https://tessdale-heritage.example/quillmark",
                            "The Quillmark works on Fenner Street, set up by Orla Penhallow, made delivery "
                            "bicycles for the town's bakers and grocers until 1961.",
                        ),
                        _r(
                            "Quillmark Tourer review",
                            "https://cyclepress.example/quillmark-tourer",
                            "A steel touring frame from the century-old Tessdale maker, still brazed by hand "
                            "in small batches.",
                        ),
                    ),
                ),
                fact="Penhallow",
            ),
            ScenarioPrompt(
                "p2",
                "What is the cafe on the top floor of the Brenmoor Lantern Library called? Search for it.",
                SHARED_SCREEN,
                SHARED_CASE,
                rows=(
                    _rows(
                        "Brenmoor Lantern Library top floor cafe",
                        _r(
                            "Brenmoor Lantern Library - Visit us",
                            "https://lantern-library.example/visit",
                            "The top floor of the Brenmoor Lantern Library is home to the Gilded Kettle, a "
                            "small cafe with views over the canal basin, open from 9 am to 4 pm.",
                        ),
                        _r(
                            "Gilded Kettle, Brenmoor - reviews",
                            "https://eatout.example/brenmoor/gilded-kettle",
                            "Quiet library cafe on the fifth floor of the Lantern Library; good scones, "
                            "limited seating at lunchtime.",
                        ),
                        _r(
                            "Lantern Library opening hours",
                            "https://lantern-library.example/hours",
                            "The library opens at 8.30 am on weekdays and 10 am on Saturdays; closed on "
                            "Sundays.",
                        ),
                    ),
                ),
                fact="Gilded Kettle",
            ),
        ),
    ),
    Scenario(
        "S2",
        10,
        (
            ScenarioPrompt(
                "p1",
                "From which pier does the Merrow ferry to Halsey Island leave? Search for it.",
                SHARED_SCREEN,
                SHARED_CASE,
                rows=(
                    _rows(
                        "Merrow ferry Halsey Island",
                        _r(
                            "Merrow Ferry (band)",
                            "https://folkbands.example/merrow-ferry",
                            "Merrow Ferry are a four-piece folk band from Calder. Their second album, Low Tide "
                            "Lanterns, was recorded in a boathouse over one winter.",
                        ),
                        _r(
                            "Merrow Ferry - tour dates",
                            "https://folkbands.example/merrow-ferry/tour",
                            "Merrow Ferry play the Corn Exchange in Calder on 14 November and the Halsey "
                            "Island festival in June.",
                        ),
                        _r(
                            "Interview: Merrow Ferry on writing about the sea",
                            "https://folkpress.example/merrow-ferry-interview",
                            "The band's singer explains why every song on the new record is named after a "
                            "boat.",
                        ),
                    ),
                    _rows(
                        "Halsey Island ferry departure pier",
                        _r(
                            "Halsey Island Ferries - Timetable",
                            "https://halsey-ferries.example/timetable",
                            "The Merrow ferry to Halsey Island leaves from Wrenfield Pier, berth 2, every "
                            "ninety minutes from 07:10 until 19:40.",
                        ),
                        _r(
                            "Getting to Halsey Island",
                            "https://visit-halsey.example/getting-here",
                            "Foot passengers board the Merrow at Wrenfield Pier; cars use the slipway beside "
                            "the harbour office.",
                        ),
                    ),
                ),
                fact="Wrenfield",
            ),
            ScenarioPrompt(
                "p2",
                "Who is the head chef at Ostrander's, the restaurant in Kellbridge? Look it up.",
                SHARED_SCREEN,
                SHARED_CASE,
                rows=(
                    _rows(
                        "Ostrander's head chef",
                        _r(
                            "Ostrander's, Port Ellery - Meet the team",
                            "https://ostranders-portellery.example/team",
                            "Ostrander's in Port Ellery is led by head chef Dario Venn, who joined from a "
                            "seafood restaurant on the north coast in 2019.",
                        ),
                        _r(
                            "Ostrander's Port Ellery review",
                            "https://eatout.example/port-ellery/ostranders",
                            "The Port Ellery Ostrander's serves a short menu of grilled fish and a famous "
                            "rhubarb tart.",
                        ),
                        _r(
                            "Ostrander's Hardware",
                            "https://ostranders-hardware.example",
                            "Family hardware store with branches in Port Ellery and Calder since 1952.",
                        ),
                    ),
                    _rows(
                        "Ostrander's Kellbridge head chef",
                        _r(
                            "Ostrander's Kellbridge - About",
                            "https://ostranders-kellbridge.example/about",
                            "Ostrander's on Mill Row in Kellbridge has been run by head chef Ines Calloway "
                            "since it opened in 2021.",
                        ),
                        _r(
                            "Kellbridge food guide: Ostrander's",
                            "https://kellbridge-guide.example/ostranders",
                            "Ines Calloway's seasonal menu at Ostrander's changes every month; booking "
                            "recommended at weekends.",
                        ),
                    ),
                ),
                fact="Calloway",
            ),
        ),
    ),
    Scenario(
        "S3",
        10,
        (
            ScenarioPrompt(
                "p1",
                'Type "Ada Lovelace" in the display name field, then click Save changes.',
                "screens/d04_settings.jpg",
                "type-04",
            ),
            ScenarioPrompt(
                "p2",
                'Fill in the email field with "ada@northwind.example", then click the Sign in button.',
                "screens/d01_login.jpg",
                "type-01",
            ),
        ),
    ),
    Scenario(
        "S4",
        10,
        (
            ScenarioPrompt(
                "p1", "Go to the Privacy settings.", "screens/d04_settings.jpg", "click-08"
            ),
            ScenarioPrompt(
                "p2",
                "Scroll down so I can keep reading the article.",
                "screens/d05_article.jpg",
                "scroll-01",
            ),
        ),
    ),
    Scenario(
        "S5",
        10,
        (
            ScenarioPrompt("p1", "Open the Holidays album.", "screens/d07_photos.jpg", "click-10"),
            ScenarioPrompt(
                "p2",
                "Turn on dark mode, and make sure it has really switched on.",
                "screens/d04_settings.jpg",
                "click-07",
            ),
        ),
    ),
    Scenario(
        "S6",
        5,
        (
            ScenarioPrompt(
                "p1",
                "Compare these ten charting libraries for me: Quiver, Brindle, Tamsel, Orrin, Vellum, Kestrel, "
                "Marrowby, Pellucid, Sorrel and Tindle. Search for each one, then tell me which is the fastest.",
                SHARED_SCREEN,
                SHARED_CASE,
                rows=_s6_rows(),
            ),
            ScenarioPrompt(
                "p2",
                "Type the following text into the body of this note, without pressing Enter: "
                + S6_TYPE_TEXT,
                SHARED_SCREEN,
                SHARED_CASE,
            ),
        ),
    ),
)

#: the scenario table's sha256 (``scenario_table_sha()``); equal to tests/unit/test_t0115_loop_probe.py's
#: ``SCENARIO_TABLE_SHA256`` once the test-author pins it (criterion 2 refuses a table that differs)
SCENARIO_SHA256 = "a82a525c8f7ab44622a967734a6fb14c66af120ce259dcf2b24c9482512a4c8e"


def schedule() -> list[tuple[str, str, int]]:
    """``(scenario_id, prompt_id, repeat)`` in run order: scenario by scenario, prompt by prompt, repeats in order."""
    return [
        (s.scenario_id, p.prompt_id, r) for s in SCENARIOS for p in s.prompts for r in range(s.n)
    ]


def scenario_table_sha(scenarios: Sequence[Scenario] | None = None) -> str:
    """sha256 of the canonical JSON of the table (the formula in tests/unit/test_t0115_loop_probe.py)."""
    table = [
        {
            "scenario_id": s.scenario_id,
            "n": s.n,
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "text": p.text,
                    "screen": p.screen,
                    "case_id": p.case_id,
                    "fact": p.fact,
                    "rows": [r.model_dump(mode="json") for r in p.rows],
                }
                for p in s.prompts
            ],
        }
        for s in (SCENARIOS if scenarios is None else scenarios)
    ]
    blob = json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prompt_of(scenario_id: str, prompt_id: str) -> ScenarioPrompt:
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            for p in s.prompts:
                if p.prompt_id == prompt_id:
                    return p
    raise KeyError(f"{scenario_id}/{prompt_id} is not in the scenario table")


# =====================================================================================================
# ScenarioSearch (criterion 3)
# =====================================================================================================


class ScenarioSearch:
    """A ``SearchBackend`` for one turn: the i-th search of the turn gets ``rows[i]``'s results (the last row set
    repeating), whatever the query -- model queries are not predictable, so it is indexed by call number, never by
    query text. Every query is kept in order (``queries``). Opens no socket."""

    def __init__(self, rows: Sequence[SearchResults]) -> None:
        self.rows = tuple(rows)
        self.queries: list[str] = []

    async def search(self, query: str) -> SearchResults:
        i = len(self.queries)
        self.queries.append(query)
        if not self.rows:
            return SearchResults(query=query, results=())
        chosen = self.rows[min(i, len(self.rows) - 1)]
        return SearchResults(query=chosen.query, results=chosen.results)


# =====================================================================================================
# H-T115-10: S3 and S4 actions report success; S5 and S6 stay on the dry run
# =====================================================================================================

#: the scenarios whose action responses say the action succeeded (human, 2026-10-17, H-T115-10)
SUCCESS_SCENARIOS = frozenset({"S3", "S4"})
#: the product's dry-run response, captured at import (the substitution below never edits ``loop.py``)
_DRY_ACTION_PAYLOAD = loop.action_payload
#: the dry-run executor's "would ..." verbs, as a success executor reports them
_DONE_VERBS = (
    ("would click ", "clicked "),
    ("would type ", "typed "),
    ("would scroll ", "scrolled "),
    ("would press ", "pressed "),
)
#: the dry-run executor's second verb (a type with ``press_enter``), always the description's suffix
_DONE_SUFFIXES = ((" and press Enter", " and pressed Enter"),)


def _done(action: str) -> str:
    """The dry-run description of an action, said as done (``would click at (x, y)`` -> ``clicked at (x, y)``),
    every verb of it (``would type 'x' at (x, y) and press Enter`` -> ``typed 'x' at (x, y) and pressed Enter``;
    tense nit, 2026-10-17). The second verb is rewritten as a suffix only, so a typed text is never touched."""
    for would, did in _DONE_VERBS:
        if action.startswith(would):
            action = did + action[len(would) :]
            break
    for now, then in _DONE_SUFFIXES:
        if action.endswith(now):
            action = action[: -len(now)] + then
    return action


class _SuccessExecutor:
    """The kit's dry-run executor, with every action it would describe said as done (H-T115-10). Its outcome status
    stays ``dry_run`` so that the session builds the response with ``loop.action_payload``, which the harness
    substitutes for the turn (``_scenario_payloads``); a refusal (bad coordinates, no frame, ...) is the inner
    executor's, unchanged."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.mode = inner.mode

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        outcome: ToolOutcome = await self.inner.execute(inv)
        if outcome.status != "dry_run":
            return outcome
        return outcome.model_copy(update={"action": _done(outcome.action)})


def scenario_kit(kit: Any, scenario_id: str, search: ScenarioSearch) -> Any:
    """The turn's kit: ``kit`` with ``search`` replaced and, for S3 and S4 only, an action executor that reports
    success; every other scenario keeps ``kit``'s dry-run executor."""
    if scenario_id in SUCCESS_SCENARIOS:
        return dataclasses.replace(kit, search=search, executor=_SuccessExecutor(kit.executor))
    return dataclasses.replace(kit, search=search)


def scenario_action_payload(scenario_id: str, action: str) -> dict[str, Any]:
    """The action response a step request of ``scenario_id`` carries for a non-refused action described as
    ``action``: S3 and S4 say it succeeded (``screen_changed: True``, status ``executed``); every other scenario is
    the product's dry-run response (``screen_changed: False``, ``dry_run``; S5 on purpose, the repeat tempter)."""
    if scenario_id in SUCCESS_SCENARIOS:
        return {"action": action, "screen_changed": True, "status": "executed"}
    return _DRY_ACTION_PAYLOAD(action)


@contextmanager
def _scenario_payloads(scenario_id: str) -> Iterator[None]:
    """For an S3/S4 turn, ``loop.action_payload`` (which the session reads by attribute) is
    ``scenario_action_payload`` for that scenario, in this process only, until the turn's session is closed."""
    if scenario_id not in SUCCESS_SCENARIOS:
        yield
        return
    loop.action_payload = functools.partial(scenario_action_payload, scenario_id)
    try:
        yield
    finally:
        loop.action_payload = _DRY_ACTION_PAYLOAD


# =====================================================================================================
# Repeats and outcome classes (pure)
# =====================================================================================================


def _canonical_args(arguments: Any) -> str:
    return json.dumps(
        arguments if arguments is not None else {}, sort_keys=True, ensure_ascii=False
    )


def _norm_query(text: str) -> str:
    """Casefold, whitespace runs collapsed, the ends stripped."""
    return " ".join(text.casefold().split())


def _collapse(text: Any) -> str:
    """Whitespace runs collapsed, the ends stripped, case kept (G6's thought clause, A2')."""
    return " ".join(str(text).split())


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, int | float) else None


def repeat_counts(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """G4 and the report-only near-repeats over one turn's executed calls, in order.

    ``exact``: calls whose ``(name, arguments)`` equals that of ANY earlier call of the turn (dict equality);
    ``normalised_query``: ``google_search`` calls whose normalised query equals an earlier search's;
    ``click_near``: ``click_at`` calls within 16 of an earlier click on each axis (the call's own 0-999 units)."""
    seen: set[tuple[str, str]] = set()
    queries: list[str] = []
    clicks: list[tuple[float, float]] = []
    exact = near_query = near_click = 0
    for c in calls:
        name = str(c.get("name"))
        args = c.get("arguments") or {}
        key = (name, _canonical_args(args))
        if key in seen:
            exact += 1
        seen.add(key)
        if name == "google_search":
            q = _norm_query(str(args.get("query", "")))
            if q in queries:
                near_query += 1
            queries.append(q)
        elif name == "click_at":
            x, y = _num(args.get("x")), _num(args.get("y"))
            if x is not None and y is not None:
                if any(abs(x - px) <= 16 and abs(y - py) <= 16 for px, py in clicks):
                    near_click += 1
                clicks.append((x, y))
    return {"exact": exact, "normalised_query": near_query, "click_near": near_click}


def classify_step(*, answer: str, thought_open: bool, call: str | None) -> str:
    """One of ``OUTCOMES``. ``call``: ``None`` (no call), ``"valid"``, ``"refused"`` or ``"repairable"`` (refused at
    a step >= 2 as "repair: step 1 only"). A call decides first; else a canvas ending inside an open thought is
    ``thought-open`` whatever text it carries; else a non-blank answer is ``final``; else ``empty``."""
    if call == "valid":
        return "call"
    if call == "refused":
        return "refused"
    if call == "repairable":
        return "repairable-refused"
    if call is not None:
        raise ValueError(f"unknown call kind {call!r}")
    if thought_open:
        return "thought-open"
    return "final" if answer.strip() else "empty"


# =====================================================================================================
# LoopMeter: the session's LlmBackend, measured per generate
# =====================================================================================================


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status an engine error carries (``raise_for_status``'s ``.response.status_code``), read
    structurally as ``Session._http_status`` does."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


class LoopMeter:
    """The real client plus a ``/metrics`` delta and two latencies per generate (T-0093's ``_MeteredLlm`` reshaped).

    ``metrics()`` returns the ``/metrics`` counters (sync; called in a thread), ``clock()`` monotonic seconds. Every
    ``generate`` reads ``metrics`` before the request and after it is done and appends one row to ``generates``:
    ``request_id`` (the id it was called with: the session's engine id), ``prefix_query_tokens`` and
    ``prefix_hit_tokens`` (this generate's own delta), ``first_commit_ms`` and ``done_ms`` (from the moment the
    request is sent, after the before-read, to the first commit and to the end of the stream), ``http_status``
    (200 when the stream ran, the engine's status when it raised one, 0 for a failure without a response, ``None``
    when cancelled), ``continuation_chars`` (a continued prompt's last message length, else ``None``) and
    ``repair`` (``cd`` was set). Prefills run to their end (aborts deferred, J12) and a generate other than the
    repair waits for any still running before its before-read (only step 1 has one)."""

    def __init__(
        self,
        inner: Any,
        *,
        metrics: Callable[[], Mapping[str, int] | None],
        clock: Callable[[], float],
    ) -> None:
        self.inner = inner
        self.metrics = metrics
        self.clock = clock
        self.generates: list[dict[str, Any]] = []
        #: the prompt of every generate by request id (``/tokenize`` after close)
        self.prompts: dict[str, Prompt] = {}
        self._prefills: dict[str, asyncio.Task[PrefillHandle]] = {}
        self.deferred_aborts: list[str] = []
        self.max_inflight = 0
        self._inflight = 0

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        task = asyncio.ensure_future(self._run_prefill(prefix, request_id))
        self._prefills[request_id] = task
        return await asyncio.shield(task)

    async def _run_prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            handle: PrefillHandle = await self.inner.prefill(prefix, request_id)
            return handle
        finally:
            self._inflight -= 1

    def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: CdRequest | None = None,
    ) -> AsyncIterator[Commit]:
        return self._generate(prompt, request_id, options, xargs, cd)

    async def _read(self) -> dict[str, int] | None:
        got = await asyncio.to_thread(self.metrics)
        return dict(got) if got is not None else None

    async def _generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None,
        xargs: Mapping[str, str] | None,
        cd: CdRequest | None,
    ) -> AsyncIterator[Commit]:
        continued = bool(getattr(prompt, "continue_final_message", False))
        row: dict[str, Any] = {
            "request_id": request_id,
            "repair": cd is not None,
            "continuation_chars": (
                len(prompt.messages[-1].text()) if continued and cd is None else None
            ),
            "prefill_wait_ms": None,
            "prefix_query_tokens": None,
            "prefix_hit_tokens": None,
            "first_commit_ms": None,
            "done_ms": None,
            "http_status": None,
            "error": None,
        }
        self.generates.append(row)
        self.prompts[request_id] = prompt
        if cd is None:
            pending = [t for t in self._prefills.values() if not t.done()]
            if pending:
                waited = self.clock()
                await asyncio.wait(pending, timeout=PREFILL_WAIT_S)
                row["prefill_wait_ms"] = round((self.clock() - waited) * 1000.0, 1)
        before = await self._read()
        sent = self.clock()
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            async for commit in self.inner.generate(prompt, request_id, options, xargs, cd=cd):
                if row["first_commit_ms"] is None:
                    row["first_commit_ms"] = round((self.clock() - sent) * 1000.0, 1)
                yield commit
            row["done_ms"] = round((self.clock() - sent) * 1000.0, 1)
            row["http_status"] = 200
        except asyncio.CancelledError:
            row["error"] = "cancelled"
            raise
        except GeneratorExit:
            row["error"] = "closed by the consumer"
            raise
        except Exception as exc:
            status = _http_status(exc)
            row["http_status"] = status if status is not None else 0
            row["error"] = repr(exc)
            raise
        finally:
            self._inflight -= 1
            if row["error"] not in ("cancelled", "closed by the consumer"):
                after = await self._read()
                if before is not None and after is not None:
                    row["prefix_query_tokens"] = after[METRIC_QUERIES] - before[METRIC_QUERIES]
                    row["prefix_hit_tokens"] = after[METRIC_HITS] - before[METRIC_HITS]

    async def tokenize(self, text: str) -> list[int]:
        ids: list[int] = await self.inner.tokenize(text)
        return ids

    async def abort(self, request_id: str) -> None:
        task = self._prefills.get(request_id)
        if task is not None and not task.done():
            self.deferred_aborts.append(request_id)
            return
        await self.inner.abort(request_id)

    async def aclose(self) -> None:
        for task in self._prefills.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._prefills.values(), return_exceptions=True)
        await self.inner.aclose()


# =====================================================================================================
# The rule (criteria 5-7): pure, over the records of trials.jsonl and steps.jsonl
# =====================================================================================================


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(record["scenario"]), str(record["prompt_id"]), int(record["repeat"]))


def _by_turn(
    steps: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], list[Mapping[str, Any]]]:
    out: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        out.setdefault(_key(s), []).append(s)
    return out


def _kn(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "fraction": f"{k}/{n}"}


def is_errored(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """V: a harness exception, a turn not IDLE within 120 s wall, or a step whose status is neither 200, 413 nor
    ``None`` (a 413 is G7's, never V's)."""
    if trial.get("exception") or trial.get("timed_out"):
        return True
    return any(s.get("http_status") not in (200, 413, None) for s in steps)


def g6_violations(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """G6 (hard; P6, H8, D16), one entry per finding: a voiced text at a step >= 2 that executed its call (step 1
    voices its start sentence and then executes, D16 moment 1); a special literal or thought marker in a voiced
    *source* (the step's ``voiced_sources``, the raw ``TtsSpeak.source`` texts; ruled 2026-10-17 at 20ea438: the TTS
    text has them stripped); a voiced source that occurs (plain substring) inside a thought commit of the turn and
    does not occur in that step's step text, every side whitespace-collapsed (A2', 2026-10-17, amending A2, which
    replaced "equal to a thought commit"; the TTS-normalised ``voiced`` text is not the compared side); a step line
    without ``voiced_sources`` (the clauses cannot be read); and, for a turn that finished only (no exception, not
    timed out; A1: an unfinished turn is V's), a stored history entry that is not the join of the played runs."""
    tag = f"{trial.get('scenario')}/{trial.get('prompt_id')}/r{trial.get('repeat')}"
    out: list[str] = []
    thoughts = [_collapse(t) for t in trial.get("thought_commits") or [] if str(t).strip()]
    for s in steps:
        voiced = [str(v) for v in s.get("voiced") or [] if str(v).strip()]
        step_text = _collapse(s.get("step_text") or "")
        if int(s["step"]) >= 2 and s.get("executed") and voiced:
            out.append(f"{tag}: voiced at step {s['step']}, which executed its call: {voiced!r}")
        if "voiced_sources" not in s:
            out.append(f"{tag}: step {s['step']} has no voiced_sources (A2'), G6 cannot be read")
            continue
        for src in [str(v) for v in s.get("voiced_sources") or []]:
            hits = sorted({lit for lit in (*SPECIAL_TOKENS, *THOUGHT_MARKERS) if lit in src})
            if hits:
                out.append(f"{tag}: voiced special literal(s) {hits} at step {s['step']}")
            text = _collapse(src)
            if text and any(text in t for t in thoughts) and text not in step_text:
                out.append(
                    f"{tag}: voiced thought text not in the step text at step {s['step']}: {text!r}"
                )
    if not trial.get("exception") and not trial.get("timed_out"):
        played = [str(p) for p in trial.get("played_runs") or [] if str(p)]
        joined = " ".join(played) or None
        stored = trial.get("history") or None
        if stored != joined:
            out.append(f"{tag}: history {stored!r} is not the join of the played runs {joined!r}")
    return out


def _sane(step: Mapping[str, Any]) -> bool:
    """G1: a final (non-empty with the thought closed: the class says so), a valid call, or a refusal (either
    class) with step text."""
    outcome = step.get("outcome")
    if outcome in ("final", "call"):
        return True
    return outcome in ("refused", "repairable-refused") and bool(
        str(step.get("step_text") or "").strip()
    )


def _max_same_call(executed: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter((str(c.get("name")), _canonical_args(c.get("arguments"))) for c in executed)
    return max(counts.values(), default=0)


def _share(
    rows: Sequence[Mapping[str, Any]], pred: Callable[[Mapping[str, Any]], bool]
) -> tuple[int, int]:
    return sum(1 for t in rows if pred(t)), len(rows)


def _at_least(k: int, n: int, bar: Fraction) -> bool:
    return n > 0 and Fraction(k, n) >= bar


def _at_most(k: int, n: int, bar: Fraction) -> bool:
    return n == 0 or Fraction(k, n) <= bar


def _executed_n(trial: Mapping[str, Any]) -> int:
    return len(trial.get("executed") or [])


def _clauses(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Every gated clause's counts over the given records (the whole run, or one scenario's)."""
    turns = _by_turn(steps)

    def steps_of(t: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return turns.get(_key(t), [])

    of = {
        sid: [t for t in trials if t["scenario"] == sid] for sid in ("S1", "S2", "S3", "S4", "S5")
    }
    later = [s for s in steps if int(s["step"]) >= 2]
    g1_k = sum(1 for s in later if _sane(s))
    s4 = _share(of["S4"], lambda t: _executed_n(t) == 1 and t.get("tool_loop_end") == "final")
    s1 = _share(of["S1"], lambda t: t.get("tool_loop_end") == "final" and 1 <= _executed_n(t) <= 2)
    s1_5 = [t for sid in ("S1", "S2", "S3", "S4", "S5") for t in of[sid]]
    budget = _share(s1_5, lambda t: t.get("tool_loop_end") == "budget")
    s1_4 = [t for sid in ("S1", "S2", "S3", "S4") for t in of[sid]]
    calls = sum(_executed_n(t) for t in s1_4)
    reps = sum(repeat_counts(t.get("executed") or [])["exact"] for t in s1_4)
    thrice = _share(of["S5"], lambda t: _max_same_call(t.get("executed") or []) >= 3)
    fact1 = _share(of["S1"], lambda t: t.get("fact_hit") is True)
    fact2 = _share(of["S2"], lambda t: t.get("fact_hit") is True)
    findings = [f for t in trials for f in g6_violations(t, steps_of(t))]
    over413 = [s for s in later if s.get("http_status") == 413]
    over_tokens = [s for s in later if int(s.get("prefix_query_tokens") or 0) > MAX_PROMPT_TOKENS]
    return {
        "G1": {"k": g1_k, "n": len(later)},
        "G2": {"S4": s4, "S1": s1},
        "G3": budget,
        "G4": {"exact": (reps, calls), "S5": thrice},
        "G5": {"S1": fact1, "S2": fact2},
        "G6": findings,
        "G7": {"413": over413, "over_tokens": over_tokens},
    }


def _bars(c: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    g1k, g1n = c["G1"]["k"], c["G1"]["n"]
    (s4k, s4n), (s1k, s1n) = c["G2"]["S4"], c["G2"]["S1"]
    bk, bn = c["G3"]
    (rk, rn), (tk, tn) = c["G4"]["exact"], c["G4"]["S5"]
    (f1k, f1n), (f2k, f2n) = c["G5"]["S1"], c["G5"]["S2"]
    return {
        "G1": {
            "holds": _at_least(g1k, g1n, G1_MIN),
            "k": g1k,
            "n": g1n,
            "bar": ">= 9/10 of step >= 2 generates sane",
        },
        "G2": {
            "holds": _at_least(s4k, s4n, G2_MIN) and _at_least(s1k, s1n, G2_MIN),
            "S4": _kn(s4k, s4n),
            "S1": _kn(s1k, s1n),
            "bar": "S4 exactly one executed and final >= 16/20; S1 final with 1-2 executed >= 16/20",
        },
        "G3": {
            "holds": _at_most(bk, bn, G3_MAX),
            "k": bk,
            "n": bn,
            "bar": "tool_loop_end budget in S1-S5 <= 10/100",
        },
        "G4": {
            "holds": _at_most(rk, rn, G4_EXACT_MAX) and _at_most(tk, tn, G4_S5_MAX),
            "exact": _kn(rk, rn),
            "S5": _kn(tk, tn),
            "bar": "exact repeats over S1-S4 executed calls <= 1/10; S5 turns with a call executed >= 3 times <= 4/20",
        },
        "G5": {
            "holds": _at_least(f1k, f1n, G5_S1_MIN) and _at_least(f2k, f2n, G5_S2_MIN),
            "S1": _kn(f1k, f1n),
            "S2": _kn(f2k, f2n),
            "bar": "fact_hit S1 >= 16/20, S2 >= 14/20",
        },
        "G6": {
            "holds": not c["G6"],
            "k": len(c["G6"]),
            "findings": list(c["G6"]),
            "bar": "= 0",
        },
        "G7": {
            "holds": not c["G7"]["413"] and not c["G7"]["over_tokens"],
            "413": len(c["G7"]["413"]),
            "over_tokens": len(c["G7"]["over_tokens"]),
            "max_step_tokens": None,
            "bar": f"zero 413 on step requests; zero step prompts > {MAX_PROMPT_TOKENS} engine tokens",
        },
    }


def evaluate(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """*Pre-registered rules*, exactly: ``void``, the controls, G1-G7 and the outcome."""
    turns = _by_turn(steps)

    def steps_of(t: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return turns.get(_key(t), [])

    errored = [_key(t) for t in trials if is_errored(t, steps_of(t))]
    void = (not trials) or Fraction(len(errored), len(trials)) >= VOID_AT
    s12 = [t for t in trials if t["scenario"] in ("S1", "S2")]
    c1_k = sum(
        1
        for t in s12
        if any(
            int(s["step"]) == 1
            and s.get("outcome") == "call"
            and s.get("call_name") == "google_search"
            for s in steps_of(t)
        )
    )
    later = [s for s in steps if int(s["step"]) >= 2]
    controls = {"C1": _at_least(c1_k, len(s12), C1_MIN), "C2": len(later) >= C2_MIN}
    bars = _bars(_clauses(trials, steps))
    bars["G7"]["max_step_tokens"] = max(
        (int(s.get("prefix_query_tokens") or 0) for s in later), default=None
    )
    failing = [b for b in BARS if not bars[b]["holds"]]
    reasons: list[str] = []
    if void:
        reasons.append(f"void ({len(errored)}/{len(trials)} errored turns, bar < 3/118)")
    if not controls["C1"]:
        reasons.append(f"C1 ({c1_k}/{len(s12)} S1+S2 step-1 searches, bar >= 32/40)")
    if not controls["C2"]:
        reasons.append(f"C2 ({len(later)} step >= 2 generates, bar >= 60)")
    if reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    elif failing:
        outcome = "not ready: " + ", ".join(failing)
    else:
        outcome = "ready"
    return {
        "void": void,
        "errored": {**_kn(len(errored), len(trials)), "turns": [list(k) for k in errored]},
        "controls": controls,
        "control_counts": {
            "C1": _kn(c1_k, len(s12)),
            "C2": {"observed": len(later), "min": C2_MIN},
        },
        "bars": bars,
        "failing": failing,
        "outcome": outcome,
    }


# =====================================================================================================
# The report (pure: reads the run directory, opens no socket)
# =====================================================================================================


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _percentile(values: Sequence[Any], p: int) -> Any:
    """Nearest rank: the ceil(p/100 * n)-th smallest value (integer arithmetic, no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-p * len(ordered) // 100))
    return ordered[rank - 1]


def _dist(values: Sequence[Any]) -> dict[str, Any]:
    return {"n": len(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95)}


def turn_expected(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool | None:
    """The table's "expected" column for one turn (report-only; ``None`` for S6, which is report-only)."""
    sid = trial["scenario"]
    end = trial.get("tool_loop_end")
    executed = trial.get("executed") or []
    report = str(trial.get("report") or "").strip()
    if sid == "S0":
        return len(executed) == 0 and end in (None, "final") and bool(report)
    if sid == "S1":
        first = next((s for s in steps if int(s["step"]) == 1), None)
        searched = (
            first is not None
            and first.get("call_name") == "google_search"
            and bool(first.get("executed"))
        )
        return (
            searched
            and end == "final"
            and int(trial.get("tool_steps") or 0) in (2, 3)
            and trial.get("fact_hit") is True
        )
    if sid == "S2":
        searches = sum(1 for c in executed if c.get("name") == "google_search")
        return searches >= 2 and end == "final" and trial.get("fact_hit") is True
    if sid == "S3":
        return (
            2 <= len(executed) <= 5
            and end == "final"
            and bool(report)
            and repeat_counts(executed)["exact"] == 0
        )
    if sid == "S4":
        return len(executed) == 1 and end == "final" and int(trial.get("tool_steps") or 0) == 2
    if sid == "S5":
        return len(executed) <= 2 and bool(report) and _max_same_call(executed) < 3
    return None


#: which gated clauses read each scenario's turns (for the per-scenario lines)
_SCENARIO_CLAUSES: dict[str, tuple[str, ...]] = {
    "S0": ("G1", "G6", "G7"),
    "S1": ("G1", "G2:S1", "G3", "G4:exact", "G5:S1", "G6", "G7"),
    "S2": ("G1", "G3", "G4:exact", "G5:S2", "G6", "G7"),
    "S3": ("G1", "G3", "G4:exact", "G6", "G7"),
    "S4": ("G1", "G2:S4", "G3", "G4:exact", "G6", "G7"),
    "S5": ("G1", "G3", "G4:S5", "G6", "G7"),
}


def _scenario_lines(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], void: bool
) -> dict[str, dict[str, Any]]:
    """Per scenario: the per-turn expectation's k/n, and pass / fail (every gated clause that reads the scenario,
    applied with its own bar to that scenario's records alone), "n/a" for S6, "inconclusive" when the run is void."""
    turns = _by_turn(steps)
    out: dict[str, dict[str, Any]] = {}
    for sid in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        mine = [t for t in trials if t["scenario"] == sid]
        mine_steps = [s for s in steps if s["scenario"] == sid]
        expected = [turn_expected(t, turns.get(_key(t), [])) for t in mine]
        line: dict[str, Any] = {
            "turns": len(mine),
            "expected": _kn(sum(1 for e in expected if e is True), len(mine)),
        }
        if sid == "S6":
            line["verdict"] = "n/a"
        elif void:
            line["verdict"] = "inconclusive"
        else:
            c = _clauses(mine, mine_steps)
            checks = {
                "G1": _at_least(c["G1"]["k"], c["G1"]["n"], G1_MIN) or c["G1"]["n"] == 0,
                "G2:S1": _at_least(*c["G2"]["S1"], G2_MIN),
                "G2:S4": _at_least(*c["G2"]["S4"], G2_MIN),
                "G3": _at_most(*c["G3"], G3_MAX),
                "G4:exact": _at_most(*c["G4"]["exact"], G4_EXACT_MAX),
                "G4:S5": _at_most(*c["G4"]["S5"], G4_S5_MAX),
                "G5:S1": _at_least(*c["G5"]["S1"], G5_S1_MIN),
                "G5:S2": _at_least(*c["G5"]["S2"], G5_S2_MIN),
                "G6": not c["G6"],
                "G7": not c["G7"]["413"] and not c["G7"]["over_tokens"],
            }
            failed = [name for name in _SCENARIO_CLAUSES[sid] if not checks[name]]
            line["verdict"] = "pass" if not failed else "fail"
            line["failed_clauses"] = failed
        out[sid] = line
    return out


def report_only(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The report-only rows of *Pre-registered rules* (no threshold; a threshold needs its own task)."""
    turns = _by_turn(steps)
    ks = sorted({int(s["step"]) for s in steps})

    def at(k: int) -> list[Mapping[str, Any]]:
        return [s for s in steps if int(s["step"]) == k]

    latency = {
        str(k): {
            "first_commit_ms": _dist(
                [s["first_commit_ms"] for s in at(k) if s.get("first_commit_ms") is not None]
            ),
            "done_ms": _dist([s["done_ms"] for s in at(k) if s.get("done_ms") is not None]),
        }
        for k in ks
    }
    gaps = [t["silent_gap_ms"] for t in trials if t.get("silent_gap_ms") is not None]
    prefix: dict[str, dict[str, Any]] = {}
    growth: list[float] = []
    for k in ks:
        if k < 2:
            continue
        rows = []
        for s in at(k):
            mine = turns.get(_key(s), [])
            prev = next((p for p in mine if int(p["step"]) == k - 1), None)
            q, h = s.get("prefix_query_tokens"), s.get("prefix_hit_tokens")
            pq = prev.get("prefix_query_tokens") if prev is not None else None
            if not isinstance(q, int) or not isinstance(h, int) or not isinstance(pq, int):
                continue
            predicted = (pq // PREFIX_BLOCK_TOKENS) * PREFIX_BLOCK_TOKENS
            flagged = predicted > 0 and Fraction(h) < PREFIX_FLAG_SHARE * predicted
            rows.append((h, q, predicted, flagged))
            pc = (prev.get("continuation_chars") or 0) if prev is not None else 0
            c = s.get("continuation_chars")
            if isinstance(c, int) and q - pq > 0:
                growth.append(round((c - pc) / (q - pq), 3))
        prefix[str(k)] = {
            "n": len(rows),
            "hit": sum(r[0] for r in rows),
            "query": sum(r[1] for r in rows),
            "predicted": sum(r[2] for r in rows),
            "flagged": sum(1 for r in rows if r[3]),
        }
    ends: dict[str, dict[str, int]] = {}
    for t in trials:
        ends.setdefault(t["scenario"], Counter())[str(t.get("tool_loop_end"))] += 1  # type: ignore[index]
    shares = {str(k): dict(Counter(str(s.get("outcome")) for s in at(k))) for k in ks}
    q6 = {
        str(k): _kn(sum(1 for s in at(k) if s.get("opened_with_thought")), len(at(k))) for k in ks
    }
    near = {
        "normalised_query": sum(
            int((t.get("repeats") or {}).get("normalised_query") or 0) for t in trials
        ),
        "click_near": sum(int((t.get("repeats") or {}).get("click_near") or 0) for t in trials),
        "exact": sum(int((t.get("repeats") or {}).get("exact") or 0) for t in trials),
    }
    starters = []
    for t in trials:
        first = next((s for s in turns.get(_key(t), []) if int(s["step"]) == 1), None)
        if first is not None and first.get("outcome") == "call":
            starters.append(bool([v for v in first.get("voiced") or [] if str(v).strip()]))
    s6 = [t for t in trials if t["scenario"] == "S6"]
    s6_steps = [s for s in steps if s["scenario"] == "S6"]
    supported = [s for s in steps if isinstance(s.get("tokenize"), list)]
    tail = [
        s
        for s in supported
        if int(s["step"]) >= 2 and s["tokenize"] and s["tokenize"][-1] == TOOL_RESPONSE_CLOSE_ID
    ]
    return {
        "latency_by_step": latency,
        "silent_gap_ms": _dist(gaps),
        "silent_gap_over_5s_p50": (_percentile(gaps, 50) or 0) > 5000 if gaps else None,
        "prefix_by_step": prefix,
        "chars_per_token_growth": _dist(growth),
        "over_budget": [
            {
                "turn": list(_key(t)),
                "step": t.get("over_budget_step"),
                "chars": t.get("over_budget_chars"),
            }
            for t in trials
            if t.get("over_budget_step") is not None
        ],
        "s6_max_step_tokens": max(
            (int(s.get("prefix_query_tokens") or 0) for s in s6_steps), default=None
        ),
        "s6_ends": dict(Counter(str(t.get("tool_loop_end")) for t in s6)),
        "tool_loop_end": {sid: dict(c) for sid, c in ends.items()},
        "outcome_shares_by_step": shares,
        "q6_opened_with_thought_by_step": q6,
        "repeats_all_scenarios": near,
        "spoken_start": _kn(sum(starters), len(starters)),
        "q1_tokenize": {
            "supported": _kn(len(supported), len(steps)),
            "step_prompts_ending_in_tool_response": _kn(
                len(tail), sum(1 for s in supported if int(s["step"]) >= 2)
            ),
        },
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    """``evaluate`` over the run's ``trials.jsonl`` and ``steps.jsonl``, plus the report-only rows and the
    per-scenario lines (``report.json``)."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    trials = _read_jsonl(run_dir / "trials.jsonl")
    steps = _read_jsonl(run_dir / "steps.jsonl")
    result = evaluate(trials, steps)
    return {
        "run_dir": str(run_dir),
        "regime": manifest.get("regime"),
        "stopped": manifest.get("stopped"),
        "turns": len(trials),
        "generates": len(steps),
        **result,
        "scenarios": _scenario_lines(trials, steps, result["void"]),
        "report_only": report_only(trials, steps),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _cell(rate: Mapping[str, Any] | None) -> str:
    return "-" if not rate else f"{rate.get('k')}/{rate.get('n')}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """report.md: the outcome, every bar's k/n beside its threshold, the per-scenario lines, the report-only rows."""
    out = ["# T-0115 tool-loop probe", ""]
    out += [f"**Outcome:** `{report.get('outcome')}`", ""]
    regime = report.get("regime")
    if isinstance(regime, Mapping):
        out += ["## Regime", ""]
        out += [f"- {k}: `{v}`" for k, v in regime.items()]
        out.append("")
    if report.get("stopped"):
        out += [f"Run stopped early: {report['stopped']}", ""]
    errored = report.get("errored") or {}
    controls = report.get("controls") or {}
    counts = report.get("control_counts") or {}
    out += [
        "## Void and controls",
        "",
        f"- V: {_cell(errored)} errored turns (void at >= 3/118): void = {report.get('void')}",
        f"- C1: {_cell(counts.get('C1'))} S1+S2 step-1 searches (>= 32/40): {controls.get('C1')}",
        f"- C2: {(counts.get('C2') or {}).get('observed')} step >= 2 generates (>= 60): {controls.get('C2')}",
        "",
        "## Gated bars",
        "",
        "| bar | holds | reading | threshold |",
        "|---|---|---|---|",
    ]
    bars = report.get("bars") or {}

    def reading(name: str, bar: Mapping[str, Any]) -> str:
        if name in ("G1", "G3"):
            return f"{bar.get('k')}/{bar.get('n')}"
        if name == "G2":
            return f"S4 {_cell(bar.get('S4'))}, S1 {_cell(bar.get('S1'))}"
        if name == "G4":
            return f"exact {_cell(bar.get('exact'))}, S5 {_cell(bar.get('S5'))}"
        if name == "G5":
            return f"S1 {_cell(bar.get('S1'))}, S2 {_cell(bar.get('S2'))}"
        if name == "G6":
            return f"{bar.get('k')} finding(s)"
        return (
            f"{bar.get('413')} x 413, {bar.get('over_tokens')} over; max step tokens "
            f"{_fmt(bar.get('max_step_tokens'))}"
        )

    for name in BARS:
        bar = bars.get(name) or {}
        out.append(
            f"| {name} | {'yes' if bar.get('holds') else '**NO**'} | {reading(name, bar)} | {bar.get('bar')} |"
        )
    findings = (bars.get("G6") or {}).get("findings") or []
    if findings:
        out += ["", "G6 findings:", ""] + [f"- {f}" for f in findings[:50]]
    out += [
        "",
        "## Per scenario",
        "",
        "| scenario | expected per turn | verdict | failed clauses |",
    ]
    out.append("|---|---|---|---|")
    for sid, line in (report.get("scenarios") or {}).items():
        out.append(
            f"| {sid} | {_cell(line.get('expected'))} | {line.get('verdict')} | "
            f"{', '.join(line.get('failed_clauses') or []) or '-'} |"
        )
    ro = report.get("report_only") or {}
    out += ["", "## Report-only", "", "| step | first commit ms p50/p95 | done ms p50/p95 | n |"]
    out.append("|---|---|---|---|")
    for k, lat in (ro.get("latency_by_step") or {}).items():
        fc, dn = lat.get("first_commit_ms") or {}, lat.get("done_ms") or {}
        out.append(
            f"| {k} | {_fmt(fc.get('p50'))} / {_fmt(fc.get('p95'))} | {_fmt(dn.get('p50'))} / "
            f"{_fmt(dn.get('p95'))} | {fc.get('n')} |"
        )
    gap = ro.get("silent_gap_ms") or {}
    out += [
        "",
        f"- silent gap ms (session clock, step-1 drain to the report's first TtsSpeak): n {gap.get('n')}, "
        f"p50 {_fmt(gap.get('p50'))}, p95 {_fmt(gap.get('p95'))}",
        "",
        "| step | prefix hit / query | predicted floor | flagged (< 9/10 of predicted) | n |",
        "|---|---|---|---|---|",
    ]
    for k, row in (ro.get("prefix_by_step") or {}).items():
        out.append(
            f"| {k} | {row.get('hit')} / {row.get('query')} | {row.get('predicted')} | {row.get('flagged')} | "
            f"{row.get('n')} |"
        )
    cpt = ro.get("chars_per_token_growth") or {}
    q1 = ro.get("q1_tokenize") or {}
    out += [
        "",
        f"- chars per token of continuation growth: p50 {_fmt(cpt.get('p50'))}, p95 {_fmt(cpt.get('p95'))}, "
        f"n {cpt.get('n')}",
        f"- over_budget turns: {ro.get('over_budget')}",
        f"- S6 ends: {ro.get('s6_ends')}; S6 max step tokens: {_fmt(ro.get('s6_max_step_tokens'))}",
        f"- tool_loop_end by scenario: {ro.get('tool_loop_end')}",
        f"- outcome shares by step: {ro.get('outcome_shares_by_step')}",
        f"- Q6, canvases opening with a thought, by step: "
        f"{ {k: _cell(v) for k, v in (ro.get('q6_opened_with_thought_by_step') or {}).items()} }",
        f"- repeats over all scenarios (exact, normalised query, click within 16): {ro.get('repeats_all_scenarios')}",
        f"- spoken start (step 1 voiced before its call; D16, never gated): {_cell(ro.get('spoken_start'))}",
        f"- Q1 /tokenize (chat form, continue_final_message): supported {_cell(q1.get('supported'))}; step "
        f"prompts ending in <tool_response|> {_cell(q1.get('step_prompts_ending_in_tool_response'))}",
        "",
    ]
    return "\n".join(out)


# =====================================================================================================
# One turn's records, from its session log (pure)
# =====================================================================================================

_CALL_STEP = re.compile(r"-call(?:-s(\d+))?$")


def _call_step(call_id: str) -> int | None:
    m = _CALL_STEP.search(call_id)
    if m is None:
        return None
    return int(m.group(1)) if m.group(1) else 1


def _norm_text(text: str) -> str:
    return " ".join(text.casefold().split())


def fact_matches(fact: str | None, report: str) -> bool:
    """``fact_hit``, as pre-registered in docs/cd/toolcall-loop-probe.md: the fact (a name; no fact is a time or a
    number, so no alternative form is accepted) occurs in the report casefolded, with whitespace runs collapsed, and
    bounded on both sides by a non-word character or the end of the text (``Calloway's`` holds ``Calloway``,
    ``Callowayson`` does not)."""
    if not fact:
        return False
    pattern = r"(?<!\w)" + re.escape(_norm_text(fact)) + r"(?!\w)"
    return re.search(pattern, _norm_text(report)) is not None


def turn_records(
    key: tuple[str, str, int],
    *,
    records: Sequence[Mapping[str, Any]],
    line: Mapping[str, Any] | None,
    generates: Sequence[Mapping[str, Any]],
    tokenized: Mapping[str, Any],
    history: str | None,
    fact: str | None,
    queries: Sequence[str],
    exception: str | None,
    timed_out: bool,
    engine_idle_before: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The trial line and the step lines of one turn, from its session's log records (one turn per session), its
    ``turns.jsonl`` line and ``LoopMeter``'s rows (keyed by the engine id the session sent)."""
    sid, prompt_id, repeat = key
    data = [(r.get("kind"), r.get("data") or {}, r) for r in records]
    requests = [d for kind, d, _ in data if kind == "llm.request"]
    by_attempt = {str(d.get("attempt_id")): d for d in requests}
    gen_requests = [d for d in requests if d.get("phase") in ("generate", "tool_step")]
    step_of: dict[str, int] = {}
    request_ids: list[str] = []
    for d in gen_requests:
        mid = str(d.get("request_id"))
        step_of[mid] = 1 if d.get("phase") == "generate" else int(d.get("step") or 0)
        if mid not in request_ids:
            request_ids.append(mid)
    repair_ids = sorted(
        {str(d.get("request_id")) for d in requests if d.get("phase") == "tool_repair"}
    )
    turn_id = str((line or {}).get("turn_id") or "") or next(
        (str(r.get("turn_id")) for _, _, r in data if r.get("turn_id")), ""
    )

    # --- the meter's rows by step (an over-long retry keeps the last attempt) ---------------------------------
    rows_by_step: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    for g in generates:
        attempt = str(g.get("request_id"))
        req = by_attempt.get(attempt)
        if req is None or req.get("phase") not in ("generate", "tool_step"):
            continue
        k = step_of.get(str(req.get("request_id")))
        if k:
            rows_by_step.setdefault(k, []).append((attempt, g))

    # --- the machine's steps: voiced text per loop step, executions, playback terminals ------------------------
    voiced: dict[int, list[str]] = {}
    voiced_sources: dict[int, list[str]] = {}
    first_speak_ms: dict[int, int | None] = {}
    executed: list[dict[str, Any]] = []
    executed_ids: set[str] = set()
    played_runs: list[str] = []
    step1_drain_ms: int | None = None
    current = 1
    for kind, d, r in data:
        if kind != "machine.step":
            continue
        event = d.get("event") or {}
        rid = event.get("request_id")
        if isinstance(rid, str) and rid in step_of:
            current = step_of[rid]
        etype = event.get("type")
        if etype in ("playback.drained", "playback.stopped"):
            played_runs.append(str(event.get("played_text") or ""))
            if current == 1 and step1_drain_ms is None and etype == "playback.drained":
                step1_drain_ms = r.get("session_ms")
        for c in d.get("commands") or []:
            if not isinstance(c, Mapping):
                continue
            if c.get("type") == "tts.speak" and str(c.get("text") or ""):
                voiced.setdefault(current, []).append(str(c.get("text")))
                voiced_sources.setdefault(current, []).append(str(c.get("source") or ""))
                first_speak_ms.setdefault(current, r.get("session_ms"))
            elif c.get("type") == "tool.execute":
                cid = str(c.get("call_id"))
                executed_ids.add(cid)
                executed.append(
                    {
                        "step": _call_step(cid) or current,
                        "name": c.get("name"),
                        "arguments": dict(c.get("arguments") or {}),
                    }
                )
    calls: dict[str, list[Mapping[str, Any]]] = {}
    for kind, d, _ in data:
        if kind == "tool.call":
            calls.setdefault(str(d.get("call_id")), []).append(d)
    commits = [d for kind, d, _ in data if kind == "llm.commit"]
    thought_commits = [
        str(d.get("text"))
        for d in commits
        if d.get("segment") == "thought" and str(d.get("text") or "").strip()
    ]

    steps: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    for k in sorted(rows_by_step):
        attempts = rows_by_step[k]
        attempt, g = attempts[-1]
        retries += [
            {"step": k, "attempt_id": a, "http_status": r.get("http_status")}
            for a, r in attempts[:-1]
        ]
        mid = str(by_attempt[attempt].get("request_id"))
        mine = [d for d in commits if str(d.get("attempt_id")) == attempt]
        non_tool = [d for d in mine if d.get("segment") != "tool"]
        tool = [d for d in mine if d.get("segment") == "tool"]
        raw = "".join(
            str(d.get("text") or "")
            for d in non_tool
            if d.get("segment") == "answer" and not d.get("thought_open")
        )
        step_text = router.route(raw).spoken_prefix.strip()
        thought_open = bool(non_tool) and bool(non_tool[-1].get("thought_open"))
        cid = f"{turn_id}-call" if k == 1 else f"{turn_id}-call-s{k}"
        statuses = calls.get(cid, [])
        kind_: str | None = None
        call_name: str | None = None
        if cid in executed_ids:
            kind_ = "valid"
        elif statuses:
            first = statuses[0]
            if first.get("status") == "refused":
                kind_ = "repairable" if first.get("reason") == REPAIR_STEP1_ONLY else "refused"
            elif any(s.get("status") == "refused" for s in statuses):
                kind_ = "refused"
            else:
                kind_ = "valid"
        elif tool:
            verdict = live.classify(str(tool[-1].get("text") or ""))
            if verdict.status == "valid":
                kind_ = "valid"
            elif verdict.status == "repairable" and k >= 2:
                kind_ = "repairable"
            else:
                kind_ = "refused"
            call_name = verdict.name
        if statuses:
            call_name = str(statuses[0].get("name") or "") or None
        steps.append(
            {
                "scenario": sid,
                "prompt_id": prompt_id,
                "repeat": repeat,
                "step": k,
                "request_id": mid,
                "attempt_id": attempt,
                "continuation_chars": g.get("continuation_chars") if k >= 2 else None,
                "prefix_query_tokens": g.get("prefix_query_tokens"),
                "prefix_hit_tokens": g.get("prefix_hit_tokens"),
                "tokenize": tokenized.get(attempt, "unsupported"),
                "first_commit_ms": g.get("first_commit_ms"),
                "done_ms": g.get("done_ms"),
                "prefill_wait_ms": g.get("prefill_wait_ms"),
                "outcome": classify_step(answer=step_text, thought_open=thought_open, call=kind_),
                "call_name": call_name,
                "step_text": step_text,
                "opened_with_thought": bool(non_tool) and non_tool[0].get("segment") == "thought",
                "executed": cid in executed_ids,
                "voiced": list(voiced.get(k, [])),
                "voiced_sources": list(voiced_sources.get(k, [])),
                "http_status": g.get("http_status"),
                "error": g.get("error"),
            }
        )

    last = steps[-1]["step"] if steps else None
    report = " ".join(voiced.get(last, [])) if last is not None else ""
    gap: int | None = None
    if last is not None and last >= 2 and step1_drain_ms is not None:
        speak = first_speak_ms.get(last)
        if speak is not None:
            gap = int(speak) - int(step1_drain_ms)
    over = next(
        (d for kind, d, _ in data if kind == "note" and d.get("what") == OVER_BUDGET_NOTE), None
    )
    extras = (line or {}).get("extras") or {}
    trial: dict[str, Any] = {
        "scenario": sid,
        "prompt_id": prompt_id,
        "repeat": repeat,
        "turn_id": turn_id or None,
        "session_id": (line or {}).get("session_id"),
        "request_ids": request_ids,
        "repair_ids": repair_ids,
        "retries": retries,
        "tool_steps": extras.get("tool_steps", len(steps)),
        "tool_steps_executed": extras.get("tool_steps_executed", len(executed)),
        "tool_loop_end": extras.get("tool_loop_end"),
        "line_outcome": (line or {}).get("outcome"),
        "executed": executed,
        "repeats": repeat_counts(executed),
        "played_runs": played_runs,
        "history": history,
        "thought_commits": thought_commits,
        "report": report,
        "fact": fact,
        "fact_hit": fact_matches(fact, report),
        "over_budget_step": over.get("step") if over is not None else None,
        "over_budget_chars": over.get("chars") if over is not None else None,
        "queries": list(queries),
        "exception": exception,
        "timed_out": timed_out,
        "engine_idle_before": engine_idle_before,
        "silent_gap_ms": gap,
    }
    return trial, steps


def _not_run(key: tuple[str, str, int], reason: str) -> dict[str, Any]:
    """A turn that could not run: its trial line, errored by its exception (V)."""
    sid, prompt_id, repeat = key
    return {
        "scenario": sid,
        "prompt_id": prompt_id,
        "repeat": repeat,
        "turn_id": None,
        "session_id": None,
        "request_ids": [],
        "repair_ids": [],
        "retries": [],
        "tool_steps": 0,
        "tool_steps_executed": 0,
        "tool_loop_end": None,
        "line_outcome": None,
        "executed": [],
        "repeats": repeat_counts([]),
        "played_runs": [],
        "history": None,
        "thought_commits": [],
        "report": "",
        "fact": None,
        "fact_hit": False,
        "over_budget_step": None,
        "over_budget_chars": None,
        "queries": [],
        "exception": reason,
        "timed_out": False,
        "engine_idle_before": None,
        "silent_gap_ms": None,
    }


# =====================================================================================================
# Refusals (no socket)
# =====================================================================================================


def check_loopback(base_url: str) -> None:
    """``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1``; any name (``localhost`` too) is refused without being
    resolved, and so is a v4-mapped v6 address. ``ValueError`` otherwise."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {base_url!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"base_url host {host!r} is not a loopback IP literal") from None
    ok = (
        addr in ipaddress.ip_network("127.0.0.0/8")
        if addr.version == 4
        else addr == ipaddress.ip_address("::1")
    )
    if not ok:
        raise ValueError(f"base_url host {host!r} is not in 127.0.0.0/8 or ::1")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_rows() -> dict[str, dict[str, Any]]:
    return {str(r["id"]): r for r in _read_jsonl(CASES_PATH) if "id" in r}


def non_dev_cases(scenarios: Sequence[Scenario]) -> list[str]:
    """Every scenario case id that is not a dev-split case of ``cases.jsonl`` (test split or unknown)."""
    rows = _case_rows()
    out: list[str] = []
    for s in scenarios:
        for p in s.prompts:
            if p.case_id is None:
                continue
            row = rows.get(p.case_id)
            if row is None or row.get("split") != "dev":
                out.append(p.case_id)
    return out


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A 3xx is never followed."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _get_text(url: str, timeout_s: float = 10.0) -> str | None:
    try:
        with _OPENER.open(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            body: bytes = resp.read()
            return body.decode("utf-8", errors="replace")
    except (OSError, ValueError, http.client.HTTPException):
        return None


def read_metrics(base_url: str) -> dict[str, int] | None:
    """The prefix-cache counters and the running/waiting gauges off ``/metrics`` (summed over label sets);
    ``None`` when the endpoint or a counter is unavailable. Reading it spends nothing."""
    text = _get_text(_root(base_url) + "/metrics")
    if text is None:
        return None
    out: dict[str, int] = {}
    for name in (METRIC_QUERIES, METRIC_HITS, METRIC_RUNNING, METRIC_WAITING):
        pat = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        vals = [float(m.group(1)) for m in re.finditer(pat, text, re.MULTILINE)]
        if not vals and name in (METRIC_QUERIES, METRIC_HITS):
            return None
        out[name] = round(sum(vals))
    return out


def _tokenize_chat(base_url: str, body: Mapping[str, Any]) -> list[int] | None:
    """The engine's ids for a chat request's prompt (chat-form ``/tokenize``, the kept keys of the request body
    and ``add_special_tokens: false``); ``None`` on any failure. Called after the session is closed."""
    kept = {k: body[k] for k in TOKENIZE_KEEP if k in body}
    kept["add_special_tokens"] = False
    request = urllib.request.Request(
        _root(base_url) + "/tokenize",
        data=json.dumps(kept).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=60.0) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    ids = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or not all(type(i) is int for i in ids):
        return None
    return ids


async def _wait_idle(base_url: str) -> tuple[bool, bool]:
    """``(idle, reachable)`` before a turn: idle once ``/metrics`` shows nothing running or waiting."""
    started = time.monotonic()
    reached = False
    while True:
        m = await asyncio.to_thread(read_metrics, base_url)
        reached = reached or m is not None
        if m is not None and m[METRIC_RUNNING] + m[METRIC_WAITING] == 0:
            return True, True
        if not reached and time.monotonic() - started >= UNREACHABLE_AFTER_S:
            return False, False
        if time.monotonic() - started >= QUIESCE_S:
            return False, reached
        await asyncio.sleep(0.2)


class _PlaybackOnly:
    """``FakeAudioIO`` with only the ``AudioIO`` playback seams (no capture pump for typed turns; T-0093)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        played: int = await self.inner.flush()
        return played

    def played_ms(self) -> float:
        return float(self.inner.played_ms())

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def playback_events(self) -> Any:
        return self.inner.playback_events()

    def is_rendering(self, t_ms: float) -> bool:
        return bool(self.inner.is_rendering(t_ms))

    def capture(self) -> Any:
        return self.inner.capture()


class _CaseMemory:
    """No recollections; never an uncaptioned episode, so the idle captioner sends no request (T-0093)."""

    def __init__(self) -> None:
        self._next = 0

    async def ingest(self, episode: Episode) -> int:
        self._next += 1
        return self._next

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]:
        return ()

    async def set_caption(self, episode_id: int, caption: str) -> None:
        return None

    async def forget_since(self, t0_ms: int) -> int:
        return 0

    async def next_uncaptioned(self) -> Episode | None:
        return None


def _fake(module: str, name: str) -> Any:
    """A class from ``tests/fakes`` (imported by name: the fakes are not typed to --strict)."""
    return getattr(importlib.import_module(f"tests.fakes.{module}"), name)


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Profile
    kit: Any
    detokenizer: Any
    manifest: dict[str, Any]
    screen_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    stopped: str | None = None


def _append_jsonl(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def _one_turn(
    run: _Run, key: tuple[str, str, int]
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """One (scenario, prompt, repeat) in a fresh session; ``reachable`` is False when ``/metrics`` never answered."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.telemetry.session_log import SessionLog

    sid, prompt_id, _ = key
    prompt = _prompt_of(sid, prompt_id)
    idle, reachable = await _wait_idle(run.base_url)
    search = ScenarioSearch(prompt.rows)
    kit = scenario_kit(run.kit, sid, search)
    inner = VllmClient(
        VllmEndpoint(base_url=run.base_url),
        transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
        detokenizer=run.detokenizer,
    )
    meter = LoopMeter(inner, metrics=lambda: read_metrics(run.base_url), clock=time.monotonic)
    clock = SessionClock()
    jpeg = (SET_DIR / prompt.screen).read_bytes()
    width, height = run.screen_sizes.get(prompt.screen, (1280, 800))
    frame = Frame(
        source=Source.SCREEN,
        ts_ms=0,
        width=width,
        height=height,
        jpeg=jpeg,
        phash=hashlib.sha256(jpeg).hexdigest()[:16],
    )
    sessions_dir = run.out / "sessions"
    log = SessionLog(sessions_dir)
    policy = TurnPolicy(
        thinking=run.profile.diffusion.thinking, tool_loop=True, max_tool_steps=MAX_TOOL_STEPS
    )
    session = Session(
        TurnMachine(policy),
        meter,
        _fake("stt", "FakeStt")(),
        _fake("tts", "FakeTts")(),
        _PlaybackOnly(_fake("audio", "FakeAudioIO")(clock)),
        _fake("perception", "FakePerception")([frame]),
        _CaseMemory(),
        clock=clock,
        config=SessionConfig(
            run.out / "data",
            turns_path=run.out / "turns.jsonl",
            profile=run.profile,
            audio_path="null",
            source=Source.SCREEN,
        ),
        session_log=log,
        log_dir=sessions_dir,
        tool_kit=kit,
    )
    exception: str | None = None
    timed_out = False
    history: str | None = None
    with _scenario_payloads(sid):  # H-T115-10: S3/S4 action responses report success
        try:
            await session.start()
            session.turn_written.clear()
            session.submit(E.UserText(t_ms=clock.now_ms, text=prompt.text))
            try:
                await asyncio.wait_for(session.turn_written.wait(), timeout=TURN_TIMEOUT_S)
            except TimeoutError:
                timed_out = True
            for entry in reversed(tuple(session.state.history)):
                if entry.user_text.strip() == prompt.text.strip():
                    history = entry.assistant_spoken or None
                    break
        except Exception as exc:
            exception = repr(exc)
        finally:
            await session.aclose()
            await meter.aclose()
    if session.failures and exception is None:
        exception = "session failures: " + "; ".join(repr(f) for f in session.failures)

    # --- after close, outside every timed span: /tokenize of every generate's prompt (Q1; open question 4) -------
    tokenized: dict[str, Any] = {}
    for g in meter.generates:
        rid = str(g["request_id"])
        if g.get("repair") or rid not in meter.prompts:
            continue
        body = inner.request_body(meter.prompts[rid], rid)
        ids = await asyncio.to_thread(_tokenize_chat, run.base_url, body)
        tokenized[rid] = ids if ids is not None else "unsupported"

    records = sorted(_read_jsonl(log.path), key=lambda r: int(r.get("seq", 0)))
    lines = [
        r for r in _read_jsonl(run.out / "turns.jsonl") if r.get("session_id") == session.session_id
    ]
    trial, steps = turn_records(
        key,
        records=records,
        line=lines[0] if lines else None,
        generates=meter.generates,
        tokenized=tokenized,
        history=history,
        fact=prompt.fact,
        queries=search.queries,
        exception=exception,
        timed_out=timed_out,
        engine_idle_before=idle,
    )
    trial["session_log"] = log.path.name
    trial["deferred_prefill_aborts"] = len(meter.deferred_aborts)
    trial["max_inflight"] = meter.max_inflight
    return trial, steps, reachable


async def _run_all(run: _Run) -> None:
    """Every turn of ``schedule()`` in order; after ``ERRORED_STOP`` consecutive errored or timed-out turns
    (``is_errored``; B3) or ``UNREACHABLE_STOP`` consecutive turns without ``/metrics``, nothing more is sent and the
    rest are written as not run (errored), so the run is void and its files still hold 118 trial lines."""
    unreachable = 0
    errored = 0
    for index, key in enumerate(schedule()):
        if run.stopped is not None:
            trial, steps = _not_run(key, f"not run: {run.stopped}"), []
        else:
            try:
                trial, steps, reachable = await _one_turn(run, key)
            except Exception as exc:
                trial, steps, reachable = _not_run(key, f"harness: {exc!r}"), [], True
            unreachable = 0 if reachable else unreachable + 1
            errored = errored + 1 if is_errored(trial, steps) else 0
            if errored >= ERRORED_STOP:
                run.stopped = (
                    f"{errored} consecutive errored turns (last: {'/'.join(map(str, key))}; B3)"
                )
            elif unreachable >= UNREACHABLE_STOP:
                run.stopped = (
                    f"the engine could not be reached for {unreachable} consecutive turns "
                    f"(last: {'/'.join(map(str, key))})"
                )
            if run.stopped is not None:
                run.manifest["stopped"] = run.stopped
                _write_json(run.out / "manifest.json", run.manifest)
                _say(f"stopping: {run.stopped}")
        _append_jsonl(run.out / "trials.jsonl", [trial])
        _append_jsonl(run.out / "steps.jsonl", steps)
        _append_jsonl(
            run.out / "queries.jsonl",
            [
                {"scenario": key[0], "prompt_id": key[1], "repeat": key[2], "call": i, "query": q}
                for i, q in enumerate(trial.get("queries") or [])
            ],
        )
        _say(
            f"{index:03d} {key[0]}/{key[1]}/r{key[2]} end={trial.get('tool_loop_end')} "
            f"steps={len(steps)} executed={len(trial.get('executed') or [])} "
            f"fact_hit={trial.get('fact_hit')} err={trial.get('exception')}"
        )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-tool-steps", type=int, default=MAX_TOOL_STEPS)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    # 1. loopback
    try:
        check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    # 2. a clean tree
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    # 3. an empty out-dir
    out: Path = args.out_dir or REPO / "logs" / f"t0115-loop-probe-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(
            f"refusing: {out} exists and is not empty; the run happens once (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. the tool kit
    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    try:
        kit, detokenizer = cli.load_tool_kit(profile)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if str(kit.tools_version)[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {str(kit.tools_version)[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(
            f"refusing: the tool repair is off ({kit.repair_off}); a run with repair off is invalid"
        )
        return EXIT_REFUSED
    # 5. the scenario table's pin
    table_sha = scenario_table_sha()
    if table_sha != SCENARIO_SHA256:
        _say(f"refusing: the scenario table's sha is {table_sha}, not the pinned {SCENARIO_SHA256}")
        return EXIT_REFUSED
    # 6. dev-split case ids only
    try:
        bad = non_dev_cases(SCENARIOS)
    except (OSError, ValueError, KeyError) as exc:
        _say(f"refusing: cannot read {CASES_PATH}: {exc}")
        return EXIT_REFUSED
    if bad:
        _say(f"refusing: case ids not in the dev split: {', '.join(bad)}")
        return EXIT_REFUSED
    # 7. the step budget
    if args.max_tool_steps != MAX_TOOL_STEPS:
        _say(
            f"refusing: --max-tool-steps {args.max_tool_steps}; the probe runs {MAX_TOOL_STEPS} (H-T115-7)"
        )
        return EXIT_REFUSED
    # 8. the cases pin
    try:
        cases_sha = _sha256(CASES_PATH)
    except OSError as exc:
        _say(f"refusing: cannot read {CASES_PATH}: {exc}")
        return EXIT_REFUSED
    if cases_sha != CASES_SHA256:
        _say(f"refusing: {CASES_PATH} has sha256 {cases_sha}, not the pinned {CASES_SHA256}")
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor import session as session_module
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.toolcall import loop as loop_module

    model = VllmEndpoint().model
    listing = _get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if read_metrics(args.base_url) is None:
        _say(
            f"{_root(args.base_url)}/metrics has no prefix-cache counters; the per-step hit cannot be measured"
        )
        return EXIT_FAILED

    rows = _case_rows()
    used = sorted({p.case_id for s in SCENARIOS for p in s.prompts if p.case_id})
    screens = sorted({p.screen for s in SCENARIOS for p in s.prompts})
    sizes: dict[str, tuple[int, int]] = {}
    for r in rows.values():
        if r.get("screen") and r.get("screen_size"):
            sizes.setdefault(str(r["screen"]), (int(r["screen_size"][0]), int(r["screen_size"][1])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": model,
            "engine_version": _get_text(_root(args.base_url) + "/version"),
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": profile.diffusion.thinking,
            "tools": f"native:{str(kit.tools_version)[:12]}:{kit.mode}:loop{MAX_TOOL_STEPS}",
            "repair": "on (step 1 only)",
            "action_responses": "S3,S4 success (H-T115-10); S5,S6 dry_run",
            "search": "ScenarioSearch (fixture rows, indexed by call)",
            "max_tool_steps": MAX_TOOL_STEPS,
            "turns": len(schedule()),
            "concurrency": 1,
            "seed": None,
            "input": "typed (UserText); one fixture frame per turn; fake STT/TTS (300 ms/word, session clock)/"
            "audio/perception; no audio device, no headset or speakers",
            "versions_lock_sha256": _sha256(REPO / "services" / "versions.lock"),
        },
        "bounds": {
            "MAX_CONTINUATION": session_module.MAX_CONTINUATION,
            "MAX_STEP_TEXT": session_module.MAX_STEP_TEXT,
            "MAX_ARG_TEXT": session_module.MAX_ARG_TEXT,
            "MAX_RESPONSE": loop_module.MAX_RESPONSE,
            "MAX_SNIPPET": loop_module.MAX_SNIPPET,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
        },
        "scenario_table_sha256": table_sha,
        "cases_sha256": cases_sha,
        "cases": {
            cid: hashlib.sha256(
                json.dumps(rows[cid], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for cid in used
        },
        "screens": {screen: _sha256(SET_DIR / screen) for screen in screens},
        "gate_file_sha256": {rel: _sha256(REPO / rel) for rel in GATE_FILES},
        "schedule": [list(k) for k in schedule()],
    }
    _write_json(out / "manifest.json", manifest)
    run = _Run(
        base_url=args.base_url,
        out=out,
        profile=profile,
        kit=kit,
        detokenizer=detokenizer,
        manifest=manifest,
        screen_sizes=sizes,
    )
    for name in ("trials.jsonl", "steps.jsonl", "queries.jsonl"):
        (out / name).touch()
    asyncio.run(_run_all(run))

    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _say(f"outcome: {report['outcome']}")
    _say(f"written to {out}")
    return EXIT_FAILED if str(report["outcome"]).startswith("inconclusive") else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
