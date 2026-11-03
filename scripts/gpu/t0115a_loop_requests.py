"""T-0115a -- the tool-loop request-level rows: form G vs A (Q4), an image in a continued message (Q5) and the
compaction A/B (the E rows). ``/launch`` ONLY (``scripts/locked/t0115_loop_probe.sh`` step 3).

    uv run python scripts/gpu/t0115a_loop_requests.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the pure half (the histories, the arms, the
classifiers, the rule, the schedule and the report) reads files only, and the engine is reached only from ``main``
(everything that talks to it is imported there).

Refusals (exit 2), all **before any socket**, in T-0115 criterion 2's order where they apply:

1. a ``--base-url`` that is not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (T-0093's ``check_loopback``;
   no name is resolved, ``localhost`` included);
2. a dirty tree (the dirty paths are named);
3. an existing, non-empty ``--out-dir``;
4. a tool kit that ``cli.load_tool_kit`` cannot load, whose repair is off, or whose tool set is not ``bbfc53ccdd80``;
5. a case id in ``DEV_CASE_IDS`` that is not a dev-split case of ``tests/eval/toolcall/cases.jsonl`` (the test split
   is never read);
6. a ``cases.jsonl`` whose sha256 is not ``CASES_SHA256``;
7. an E budget that does not elide exactly steps 1-3 of every history (a harness bug, never expected).

What it runs (docs/tasks/T-0115a.md; the pre-registration is ``docs/cd/toolcall-loop-requests.md``):

- **Q4** (criterion 2, H-T111a-10): 6 step-2 cases (``Q4_CASES``: 2 ``google_search`` with fixed fake results, 2
  ``click_at``, 1 ``type_at``, 1 ``press_key`` with dry-run ``action_payload`` responses) x arms G and A x 5 = 60.
  G is ``loop.step_prompt(base, (step,))`` through ``VllmClient``; A is ``form_a_body``, the OpenAI ``role: tool``
  body (the assistant message carries step 1's text and the call, R6), raw JSON over loopback httpx (``Message``
  has no ``tool_calls``), read by the client's own id reader.
  Report-only: the step-2 outcome shares and the start-sentence repetition rate (``repeats_start``).
- **E** (criterion 4): 4 invented-fact histories (``E_HISTORIES``) x placements old/kept x 10 repeats x arms FULL and
  COMPACT = 160, plus 40 warm (the step-4 prompt). FULL is ``step_prompt(base, steps)``, COMPACT
  ``step_prompt(base, loop.compact(steps, E_BUDGET))`` -- the product's ``compact`` at its defaults, which elides
  steps 1-3. Per (history, placement) a warm is sent before repeats 0, 2, 4, 6 and 8; the arms alternate (FULL first
  in even repeats). Per request: the outcome (``errored`` if it errored, R4; else ``classify_e``, which reads the
  final state's ``thought_open``, R1, and the pre-registered accepted list, R2), prompt tokens (``/tokenize``,
  before the run), the ``/metrics`` prefix query and hit deltas, first-commit and done ms. 3 or more errored E
  requests make E inconclusive. The cost is reported per placement (R3); old is the one-compaction cost.
- **Q5** (criterion 3, H-T115a-3), **last**: 3 requests whose final, continued assistant message carries the step-1
  continuation of an action case and then an image part (the case's screen), raw JSON over loopback httpx. Each reads
  ``accepted`` (200 and a stream that ends), ``rejected`` (anything else while ``/health`` still answers) or
  ``engine-down`` (``/health`` fails 3 probes 1 s apart, R8: the patched image path killed the EngineCore). An engine-down void
  stops the harness; every earlier row is already on disk and is kept.

Concurrency 1 and unseeded throughout; ``LlmOptions()`` as the session sends it.

Output (``--out-dir``, default ``logs/t0115a-loop-requests-<sha7>/``): ``manifest.json``, ``requests.jsonl`` (one line
per request, warm included, in send order), ``report.json`` (``build_report(run_dir)``) and ``report.md``. Exit
codes: 0 the run completed (whatever the outcome, errored requests and Q5 rejected included); 1 the run stopped
(errors in a row, engine down) or the engine is not there; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from local_astra.contracts.llm import Prompt
from local_astra.contracts.toolcall import SearchResult, SearchResults
from local_astra.toolcall import loop
from local_astra.toolcall.loop import LoopStep
from local_astra.toolcall.special_tokens import drop_special_literals

if TYPE_CHECKING:
    import httpx

    from local_astra.conductor.llm_client import VllmClient
    from local_astra.toolcall.detok import HostDetokenizer

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
CASES_PATH = SET_DIR / "cases.jsonl"
T0093_PATH = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"


def _load_t0093() -> ModuleType:
    """T-0093's harness, imported read-only by path (as T-0098 does). Importing it opens no socket."""
    name = "_t0093_toolcall_live_for_t0115a"
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

#: sha256 of ``tests/eval/toolcall/cases.jsonl`` (as T-0087 pinned it; T-0098's pin)
CASES_SHA256 = "460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85"
TOOLS_SHA12 = "bbfc53ccdd80"
SPLIT = "dev"
#: the files whose bytes decide what was sent and how it was read (criterion 6: the eval skips on drift)
GATE_FILES: tuple[str, ...] = (
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/llm_client.py",
)

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
#: this many consecutive errored Q4/E requests stop the run (the engine is gone)
STOP_AFTER_ERRORS = 10
#: one request's timeout (a 5-step E prompt at CL 256 / 48 steps is several canvases)
REQUEST_TIMEOUT_S = 120.0

SEARCH = "google_search"
PLACEMENTS: tuple[str, ...] = ("old", "kept")
E_ARMS: tuple[str, ...] = ("FULL", "COMPACT")
WARM = "warm"
#: ruling R4: ``errored`` (non-200, timeout, transport error, a stream that ends without saying why) is never correct
ERRORED = "errored"
OUTCOMES: tuple[str, ...] = ("correct", "re_search", "unsupported", "other", ERRORED)
#: ruling R4: this many errored E arm requests (of 160) make the E outcome inconclusive (T-0115's V rule)
E_ERRORED_INCONCLUSIVE = 3
#: ruling R8: Q5 asks ``/health`` up to this many times, ``HEALTH_RETRY_S`` apart, before reading engine-down
HEALTH_CHECKS = 3
HEALTH_RETRY_S = 1.0
BARS: tuple[str, ...] = ("EC", "E1", "E2", "E3")
Q4_ARMS: tuple[str, ...] = ("G", "A")
Q4_OUTCOMES: tuple[str, ...] = ("final", "call", "thought_open", "empty")
Q5_READINGS: tuple[str, ...] = ("accepted", "rejected", "engine-down")

Q4_REPEATS = 5
E_REPEATS = 10
#: a warm (the step-4 prompt) is sent before every repeat r with r % WARM_EVERY == 0: 5 per (history, placement)
WARM_EVERY = 2
#: placement -> the 0-based step whose response holds the fact (criterion 4)
FACT_STEP: dict[str, int] = {"old": 0, "kept": 4}

#: criterion 5 (H-T115a-1, ruled by the human 2026-10-17), exact fractions of 40
EC_MIN = Fraction(28, 40)
E1_MARGIN = Fraction(4, 40)
E2_MIN = Fraction(32, 40)
E3_MAX = Fraction(8, 40)

#: The one budget passed to ``loop.compact`` (at its defaults otherwise: keep_last 2, target budget * 2 // 3 = 4666).
#: Every 5-step history renders to about 8 000 chars, so compaction acts; after eliding steps 1 and 2 it is still
#: about 5 300 > 4666, so it goes on to step 3; steps 4 and 5 are the kept tail. ``main`` checks the elided set.
E_BUDGET = 7000


def _say(msg: str) -> None:
    print(f"t0115a: {msg}", file=sys.stderr, flush=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =====================================================================================================
# Q4: the six step-2 cases (pure literals)
# =====================================================================================================


@dataclass(frozen=True)
class Q4Case:
    """One step-2 case: a dev-split case id and its executed step 1 (spoken text, the call, its response)."""

    case_id: str
    step: LoopStep


def _search_response(query: str, results: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    return loop.search_payload(
        SearchResults(
            query=query,
            results=tuple(SearchResult(title=t, url=u, snippet=s) for t, u, s in results),
        )
    )


def _px(v: int, size: int) -> int:
    from local_astra.conductor.tool_executor import to_pixels

    return to_pixels(v, size)


def _q4_cases() -> tuple[Q4Case, ...]:
    """T-0111a's set: 2 search, 2 click, 1 type, 1 press_key; the calls are the cases' reference calls, the action
    texts are what ``DryRunToolExecutor`` writes for them on the cases' 1280x800 screens."""
    lisbon = (
        (
            "Lisbon weather tomorrow - hourly forecast",
            "https://weather.example/lisbon/tomorrow",
            "Sunny spells in the morning, cloud building after 15:00. High 24 C, low 16 C. Light north-westerly wind.",
        ),
        (
            "Lisbon 10-day forecast",
            "https://forecast.example/pt/lisbon",
            "Dry for most of the week with temperatures a little above the seasonal average; a chance of showers on Friday.",
        ),
        (
            "Portugal weather warnings",
            "https://meteo.example/warnings",
            "No warnings are in force for the Lisbon district.",
        ),
    )
    museum = (
        (
            "Visit the British Museum - opening times",
            "https://museum.example/visit",
            "Open daily 10:00-17:00, Fridays until 20:30. Last entry 30 minutes before closing. Closed 24-26 December.",
        ),
        (
            "British Museum - plan your visit",
            "https://museum.example/plan",
            "Free entry; booking a timed ticket is recommended at weekends and during school holidays.",
        ),
        (
            "British Museum (Bloomsbury, London)",
            "https://maps.example/british-museum",
            "Great Russell St, London. Nearest stations: Tottenham Court Road, Holborn, Russell Square.",
        ),
    )
    w, h = 1280, 800
    return (
        Q4Case(
            "search-01",
            LoopStep(
                text="Let me look that up.",
                name=SEARCH,
                arguments={"query": "weather lisbon"},
                response=_search_response("weather lisbon", lisbon),
            ),
        ),
        Q4Case(
            "search-02",
            LoopStep(
                text="Let me check the opening hours.",
                name=SEARCH,
                arguments={"query": "british museum"},
                response=_search_response("british museum", museum),
            ),
        ),
        Q4Case(
            "click-01",
            LoopStep(
                text="Clicking Sign in.",
                name="click_at",
                arguments={"x": 500, "y": 499},
                response=loop.action_payload(f"would click at ({_px(500, w)}, {_px(499, h)})"),
            ),
        ),
        Q4Case(
            "click-03",
            LoopStep(
                text="Adding the Quiet One headphones to your cart.",
                name="click_at",
                arguments={"x": 789, "y": 380},
                response=loop.action_payload(f"would click at ({_px(789, w)}, {_px(380, h)})"),
            ),
        ),
        Q4Case(
            "type-02",
            LoopStep(
                text="Searching the shop for a USB-C hub.",
                name="type_at",
                arguments={"press_enter": True, "text": "usb-c hub", "x": 500, "y": 36},
                response=loop.action_payload(
                    f"would type {'usb-c hub'!r} at ({_px(500, w)}, {_px(36, h)}) and press Enter"
                ),
            ),
        ),
        Q4Case(
            "key-01",
            LoopStep(
                text="Undoing that.",
                name="press_key",
                arguments={"combo": "Ctrl+Z"},
                response=loop.action_payload("would press Ctrl+Z"),
            ),
        ),
    )


Q4_CASES: tuple[Q4Case, ...] = _q4_cases()
#: Q5's three requests: the action cases' step-1 continuations, each followed by the case's screen
Q5_CASE_IDS: tuple[str, ...] = ("click-01", "click-03", "type-02")
#: the one dev screen every E history is asked over (scroll-01's article screen)
E_SCREEN_CASE = "scroll-01"
#: every case id the harness reads from ``cases.jsonl`` (``main`` refuses one that is not dev-split)
DEV_CASE_IDS: tuple[str, ...] = (*(c.case_id for c in Q4_CASES), E_SCREEN_CASE)


# =====================================================================================================
# E: the four invented-fact histories (pure literals; docs/cd/toolcall-loop-requests.md has them verbatim)
# =====================================================================================================

Result = tuple[str, str, str]


@dataclass(frozen=True)
class EHistory:
    """One invented-fact question and the literals its 5 maximal ``google_search`` steps are built from.

    Step ``i`` (0-based) searches ``queries[i]``; its 5 results are a lead result -- ``fact_result`` at the
    placement's fact step, else ``leads[i]`` (relevant, without the fact) -- and four of ``pool`` (rotated by step).
    Every snippet is then padded with ``filler`` sentences until the rendered block is as long as ``MAX_RESPONSE``
    allows (``_maximal``). The accepted strings are ``(fact, *equivalents)``: the complete pre-registered list
    (ruling R2: the 12-hour and unpadded forms are listed here, never derived by the matcher; ``holds`` matches each
    one case-insensitively with number/word boundaries)."""

    id: str
    question: str
    fact: str
    equivalents: tuple[str, ...]
    queries: tuple[str, ...]
    fact_result: Result
    leads: tuple[Result, ...]
    pool: tuple[Result, ...]
    filler: tuple[str, ...]

    @property
    def accepted(self) -> tuple[str, ...]:
        return (self.fact, *self.equivalents)


#: The spoken text before each step's call, the same for every history: neutral, so that it reads as well after a
#: result that answers the question (old placement) as after one that does not (H-T115a limitation: synthetic).
STEP_TEXTS: tuple[str, ...] = (
    "Let me look that up.",
    "Let me check another source to be sure.",
    "I'll cross-check that with one more search.",
    "Let me look for an official page as well.",
    "One last search to confirm.",
)
#: short filler every history may use last, so a block can be brought to within a few chars of the bound
SHORT_FILLER: tuple[str, ...] = ("See details.", "More.")

E_HISTORIES: tuple[EHistory, ...] = (
    EHistory(
        id="ferry",
        question="When does the last Sunday ferry leave Pier 9 for Halvard Island?",
        fact="19:40",
        equivalents=("7:40 pm", "7:40pm", "7:40 p.m.", "twenty to eight"),
        queries=(
            "last sunday ferry pier 9 halvard island",
            "halvard island ferry timetable sunday",
            "pier 9 ferry departures halvard",
            "halvard ferries official timetable",
            "halvard island ferry sunday evening sailing",
        ),
        fact_result=(
            "Halvard Island Ferries - Sunday timetable",
            "https://halvardferries.example/timetable/sunday",
            "Sunday sailings from Pier 9: first departure 08:15, last departure 19:40. Return crossings from "
            "the island run hourly until 20:30.",
        ),
        leads=(
            (
                "Halvard Island Ferries - timetables",
                "https://halvardferries.example/timetable",
                "Timetables for weekday, Saturday and Sunday sailings from Pier 9. Choose a day to see the "
                "departures.",
            ),
            (
                "Getting to Halvard Island",
                "https://visithalvard.example/getting-here",
                "The island is reached by ferry from Pier 9 in about 35 minutes. Sailings are less frequent at "
                "weekends.",
            ),
            (
                "Pier 9 passenger terminal",
                "https://harbour.example/pier-9",
                "Pier 9 serves the Halvard and Moss Bay routes. The ticket office opens 30 minutes before the "
                "first sailing.",
            ),
            (
                "Halvard Island Ferries - about us",
                "https://halvardferries.example/about",
                "We operate two vessels on the Pier 9 to Halvard route all year round, weather permitting.",
            ),
            (
                "Halvard ferry - reviews",
                "https://travelreviews.example/halvard-ferry",
                "Friendly crew and a short crossing. Check the timetable before travelling on a Sunday.",
            ),
        ),
        pool=(
            (
                "Moss Bay ferry timetable",
                "https://mossbayferry.example/times",
                "Moss Bay sailings leave Pier 9 every two hours on weekdays.",
            ),
            (
                "Halvard Island - things to do",
                "https://visithalvard.example/see-and-do",
                "Coastal walks, the old lighthouse keeper's cottage and a small seafood market.",
            ),
            (
                "Harbour car parks",
                "https://harbour.example/parking",
                "Long-stay parking is available next to Pier 9; pay at the machine or by app.",
            ),
            (
                "Halvard Island weather",
                "https://weather.example/halvard",
                "Strong winds can cancel crossings; the operator posts cancellations on its website.",
            ),
            (
                "Ferry fares - Halvard route",
                "https://halvardferries.example/fares",
                "Adult returns, child fares and a ten-journey card. Bicycles travel free.",
            ),
            (
                "Island ferries in the region",
                "https://coastaltravel.example/ferries",
                "A guide to the passenger ferries that leave the city harbour, with links to each operator.",
            ),
            (
                "Halvard Island accommodation",
                "https://visithalvard.example/stay",
                "Two guesthouses and a campsite; book ahead in summer.",
            ),
            (
                "Pier 9 accessibility",
                "https://harbour.example/pier-9/access",
                "Step-free access to the pontoon; assistance on request at the ticket office.",
            ),
        ),
        filler=(
            "Timings can change on public holidays.",
            "Tickets can be bought on board or online.",
            "Allow ten minutes for boarding.",
            "Dogs on leads are welcome on deck.",
            "Check the operator's website before you travel.",
            "Updated regularly by the operator.",
        ),
    ),
    EHistory(
        id="market",
        question="What time does the Ostrava Lane farmers' market open on Saturdays?",
        fact="06:45",
        equivalents=("6:45", "6:45 am", "6:45am", "6:45 a.m.", "quarter to seven"),
        queries=(
            "ostrava lane farmers market saturday opening time",
            "ostrava lane market hours",
            "ostrava lane farmers market",
            "ostrava lane market council page",
            "ostrava lane saturday market opening",
        ),
        fact_result=(
            "Ostrava Lane Farmers' Market - opening times",
            "https://ostravalanemarket.example/times",
            "Saturday market: stalls open at 06:45 and trade until 13:00. The Wednesday evening market runs "
            "from 16:00 to 20:00.",
        ),
        leads=(
            (
                "Ostrava Lane Farmers' Market",
                "https://ostravalanemarket.example",
                "Local growers, bakers and cheesemakers every Saturday and Wednesday evening. See our opening "
                "times page.",
            ),
            (
                "Markets in the city - council guide",
                "https://citycouncil.example/markets",
                "Street markets, farmers' markets and night markets across the city, with links to each "
                "organiser.",
            ),
            (
                "Ostrava Lane market - stallholders",
                "https://ostravalanemarket.example/stalls",
                "Around forty stalls, most of them within thirty miles of the city.",
            ),
            (
                "Ostrava Lane market on a map",
                "https://maps.example/ostrava-lane",
                "Ostrava Lane, between the railway arches and the canal towpath.",
            ),
            (
                "Ostrava Lane market - visitor reviews",
                "https://localreviews.example/ostrava-lane",
                "Arrive early for the bread, the best loaves sell out quickly.",
            ),
        ),
        pool=(
            (
                "Canal towpath walks",
                "https://canaltrust.example/walks",
                "A flat walk along the canal passes Ostrava Lane and the old wharf.",
            ),
            (
                "Seasonal produce calendar",
                "https://growers.example/calendar",
                "What is in season each month, from asparagus to squash.",
            ),
            (
                "Parking near Ostrava Lane",
                "https://citycouncil.example/parking",
                "On-street parking is free before nine on weekend mornings.",
            ),
            (
                "Railway arches businesses",
                "https://archesquarter.example",
                "Cafes, workshops and a brewery under the arches.",
            ),
            (
                "Become a stallholder",
                "https://ostravalanemarket.example/apply",
                "Applications are open to producers who grow, rear or make what they sell.",
            ),
            (
                "Food markets guide",
                "https://foodguide.example/markets",
                "Our pick of the region's food markets, reviewed by readers.",
            ),
            (
                "Bus routes to the canal",
                "https://citybuses.example/routes",
                "Routes 4 and 12 stop at the canal bridge.",
            ),
            (
                "Ostrava Lane market - news",
                "https://ostravalanemarket.example/news",
                "New stalls this season and a summer music programme.",
            ),
        ),
        filler=(
            "Cash and cards are accepted at most stalls.",
            "Bring a reusable bag.",
            "Opening times may vary in winter.",
            "Dogs are welcome on leads.",
            "Run by a community cooperative.",
            "Check the website for changes.",
        ),
    ),
    EHistory(
        id="hall",
        question="How many seats does the main auditorium of Corvane Hall have?",
        fact="1,187",
        equivalents=(
            "1187",
            "1 187",
            "one thousand one hundred and eighty-seven",
            "one thousand one hundred eighty-seven",
        ),
        queries=(
            "corvane hall main auditorium seats",
            "corvane hall capacity",
            "corvane hall auditorium seating plan",
            "corvane hall venue information",
            "corvane hall main auditorium number of seats",
        ),
        fact_result=(
            "Corvane Hall - venue facts",
            "https://corvanehall.example/venue",
            "The main auditorium seats 1,187, with a further 240 seats in the studio theatre and standing room "
            "in the foyer bar.",
        ),
        leads=(
            (
                "Corvane Hall - seating plan",
                "https://corvanehall.example/seating",
                "Stalls, circle and balcony. Choose a performance to see the seats that are still available.",
            ),
            (
                "Corvane Hall",
                "https://encyclopedia.example/corvane-hall",
                "A concert hall opened in the 1960s and refurbished twice; home of the city's symphony "
                "orchestra.",
            ),
            (
                "Corvane Hall - hire the venue",
                "https://corvanehall.example/hire",
                "The auditorium and studio theatre are available for conferences and private events.",
            ),
            (
                "Corvane Hall - accessibility",
                "https://corvanehall.example/access",
                "Wheelchair spaces in the stalls and circle; hearing loops in both performance spaces.",
            ),
            (
                "Corvane Hall - visitor reviews",
                "https://localreviews.example/corvane-hall",
                "Great acoustics; the balcony is steep, so book the circle if you can.",
            ),
        ),
        pool=(
            (
                "What's on at Corvane Hall",
                "https://corvanehall.example/whats-on",
                "Orchestral concerts, comedy and family shows this season.",
            ),
            (
                "Corvane Hall refurbishment",
                "https://citynews.example/corvane-refurb",
                "The latest refurbishment replaced the stage floor and the ventilation.",
            ),
            (
                "Getting to Corvane Hall",
                "https://corvanehall.example/directions",
                "A short walk from the central station; the nearest car park is on Mill Street.",
            ),
            (
                "City symphony orchestra",
                "https://citysymphony.example",
                "Our season at Corvane Hall runs from September to June.",
            ),
            (
                "Corvane Hall box office",
                "https://corvanehall.example/tickets",
                "Book online, by phone or in person; the box office opens at noon.",
            ),
            (
                "Concert halls in the region",
                "https://culture.example/halls",
                "A guide to the region's concert halls and their resident ensembles.",
            ),
            (
                "Corvane Hall bars and food",
                "https://corvanehall.example/food",
                "Pre-show dining in the foyer restaurant; interval drinks can be ordered in advance.",
            ),
            (
                "Corvane Hall history",
                "https://corvanehall.example/history",
                "From the opening night to the present day, in photographs.",
            ),
        ),
        filler=(
            "Seat availability changes as tickets are sold.",
            "Booking fees may apply.",
            "Doors open one hour before the performance.",
            "Children's tickets are half price.",
            "Large bags cannot be taken into the auditorium.",
            "Information supplied by the venue.",
        ),
    ),
    EHistory(
        id="lighthouse",
        question="How tall is the Tessaly Point lighthouse?",
        fact="41.6",
        equivalents=("forty-one point six",),
        queries=(
            "tessaly point lighthouse height",
            "tessaly point lighthouse",
            "how tall is tessaly point lighthouse",
            "tessaly point lighthouse heritage listing",
            "tessaly point lighthouse tower height metres",
        ),
        fact_result=(
            "Tessaly Point lighthouse - heritage record",
            "https://heritage.example/tessaly-point",
            "Tower height 41.6 metres from base to vane; the light stands 63 metres above high water and is "
            "visible for 22 nautical miles.",
        ),
        leads=(
            (
                "Tessaly Point lighthouse",
                "https://encyclopedia.example/tessaly-point",
                "A white masonry tower on the headland, first lit in the nineteenth century and automated in "
                "the 1980s.",
            ),
            (
                "Visit Tessaly Point",
                "https://coastpath.example/tessaly-point",
                "The lighthouse grounds are open in summer; the tower itself is not open to the public.",
            ),
            (
                "Tessaly Point lighthouse - photographs",
                "https://photos.example/tessaly",
                "Pictures of the tower, the keepers' cottages and the fog signal house.",
            ),
            (
                "Lighthouses of the north coast",
                "https://lighthouses.example/north-coast",
                "Tessaly Point is one of seven lights on the north coast, all now automated.",
            ),
            (
                "Tessaly Point walk",
                "https://walks.example/tessaly-point",
                "A circular walk from the village to the headland and back along the cliffs.",
            ),
        ),
        pool=(
            (
                "Tessaly village",
                "https://tessaly.example",
                "A fishing village with a harbour, two pubs and a small museum.",
            ),
            (
                "Coastal birdwatching",
                "https://birds.example/coast",
                "Gannets and puffins can be seen from the headland in early summer.",
            ),
            (
                "Lighthouse keepers' cottages",
                "https://holidaylets.example/keepers-cottage",
                "Self-catering cottages next to the light, sleeping up to six.",
            ),
            (
                "North coast weather",
                "https://weather.example/north-coast",
                "Sea fog is common on the headland in spring.",
            ),
            (
                "Maritime museum",
                "https://maritime.example",
                "Exhibits on shipwrecks, lifeboats and the keepers who ran the lights.",
            ),
            (
                "Coast path guide",
                "https://coastpath.example/guide",
                "Stage-by-stage descriptions of the north coast path.",
            ),
            (
                "Lighthouse lenses explained",
                "https://lighthouses.example/lenses",
                "How the large glass lenses of old lights focus a small lamp into a beam.",
            ),
            (
                "Tessaly harbour",
                "https://tessaly.example/harbour",
                "Boat trips round the headland leave from the harbour wall.",
            ),
        ),
        filler=(
            "The light is maintained by the coastal authority.",
            "The grounds close at dusk.",
            "There is no parking on the headland.",
            "Information from the heritage record.",
            "Stay on the marked paths near the cliffs.",
            "Last updated this year.",
        ),
    ),
)
_HISTORY_BY_ID: dict[str, EHistory] = {x.id: x for x in E_HISTORIES}


def _maximal(query: str, results: Sequence[Result], filler: Sequence[str]) -> dict[str, Any]:
    """A ``search_payload`` of ``results`` whose snippets are padded, round robin, with ``filler`` sentences (each
    snippet within ``MAX_SNIPPET``) while the rendered block stays within ``MAX_RESPONSE``: so ``render_response``
    never has to cut it, and the fact, near the start of its snippet, is kept whole."""
    snippets = [r[2] for r in results]
    pool = [*filler, *SHORT_FILLER]

    def payload() -> dict[str, Any]:
        return _search_response(
            query, [(t, u, s) for (t, u, _), s in zip(results, snippets, strict=True)]
        )

    def size() -> int:
        return len(loop.render_response(SEARCH, payload()))

    if size() > loop.MAX_RESPONSE or any(len(s) > loop.MAX_SNIPPET for s in snippets):
        raise ValueError(f"{query!r}: the unpadded results are already over the bound")
    k = 0
    grew = True
    while grew:
        grew = False
        for j in range(len(snippets)):
            for d in range(len(pool)):
                add = " " + pool[(k + d) % len(pool)]
                if len(snippets[j]) + len(add) > loop.MAX_SNIPPET:
                    continue
                snippets[j] += add
                if size() <= loop.MAX_RESPONSE:
                    k = (k + d + 1) % len(pool)
                    grew = True
                    break
                snippets[j] = snippets[j][: -len(add)]
    return payload()


@functools.cache
def _e_steps(history_id: str, placement: str) -> tuple[LoopStep, ...]:
    history = _HISTORY_BY_ID[history_id]
    fact_step = FACT_STEP[placement]
    steps: list[LoopStep] = []
    for i, query in enumerate(history.queries):
        lead = history.fact_result if i == fact_step else history.leads[i]
        others = [history.pool[(2 * i + k) % len(history.pool)] for k in range(4)]
        steps.append(
            LoopStep(
                text=STEP_TEXTS[i],
                name=SEARCH,
                arguments={"query": query},
                response=_maximal(query, [lead, *others], history.filler),
            )
        )
    return tuple(steps)


def e_steps(history: EHistory, placement: str) -> tuple[LoopStep, ...]:
    """Criterion 4: the history's 5 maximal ``google_search`` steps; the fact is in step 1's response (``old``) or
    in step 5's (``kept``)."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, not {placement!r}")
    return _e_steps(history.id, placement)


def e_prompts(base: Prompt, history: EHistory, placement: str) -> dict[str, Prompt]:
    """Criterion 4's three prompts on ``base`` (a step-1 prompt): ``warm`` the step-4 prompt, ``FULL``
    ``step_prompt(base, steps)``, ``COMPACT`` ``step_prompt(base, loop.compact(steps, E_BUDGET))`` -- the product's
    ``compact``, looked up on ``loop`` at call time, at its defaults."""
    steps = e_steps(history, placement)
    return {
        WARM: loop.step_prompt(base, steps[:4]),
        "FULL": loop.step_prompt(base, steps),
        "COMPACT": loop.step_prompt(base, loop.compact(steps, E_BUDGET)),
    }


def elided_steps(history: EHistory, placement: str) -> list[int]:
    """The 0-based steps whose response ``loop.compact(steps, E_BUDGET)`` replaces (``main`` refuses unless it is
    [0, 1, 2] for every history and placement)."""
    steps = e_steps(history, placement)
    compacted = loop.compact(steps, E_BUDGET)
    return [
        i for i, (a, b) in enumerate(zip(steps, compacted, strict=True)) if a.response != b.response
    ]


# =====================================================================================================
# Form A, the repetition reading and the classifiers (pure)
# =====================================================================================================

#: form A's one tool call id (the template resolves the tool message's name through it)
FORM_A_CALL_ID = "call_0"


def form_a_body(
    base: Prompt, step: LoopStep, request_id: str, client: VllmClient | None = None
) -> dict[str, Any]:
    """H-T111a-10's arm A: ``base``'s request body as ``VllmClient`` renders it (``LlmOptions()``), then an
    ``assistant`` message whose content is step 1's text (ruling R6: what form G shows the model before the call)
    and one OpenAI ``tool_calls`` entry (``arguments`` a JSON string), then a ``role: tool`` message with the same
    ``tool_call_id`` and the response as a JSON string. Not a continuation."""
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.contracts.llm import LlmOptions

    if base.continue_final_message:
        raise ValueError("form A builds on the step-1 prompt, not on a continuation")
    body = (client or VllmClient()).request_body(base, request_id, LlmOptions())
    body["messages"] = [
        *body["messages"],
        {
            "role": "assistant",
            "content": step.text,
            "tool_calls": [
                {
                    "id": FORM_A_CALL_ID,
                    "type": "function",
                    "function": {"name": step.name, "arguments": json.dumps(dict(step.arguments))},
                }
            ],
        },
        {"role": "tool", "tool_call_id": FORM_A_CALL_ID, "content": json.dumps(step.response)},
    ]
    return body


_WS = re.compile(r"\s+")
_FIRST_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)", re.DOTALL)


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def repeats_start(step1_text: str, step2_text: str) -> bool:
    """Criterion 2's report-only reading: the step-2 text begins with step 1's first sentence, both normalised
    (case folded, runs of whitespace collapsed). An empty step-1 text never repeats."""
    first_text = _normalise(step1_text)
    if not first_text:
        return False
    found = _FIRST_SENTENCE.match(first_text)
    first = found.group(0) if found else first_text
    return _normalise(step2_text).startswith(first)


CALL_TAG = "<|tool_call>call:"


def _call_name(text: str) -> str:
    body = text[text.index(CALL_TAG) + len(CALL_TAG) :] if CALL_TAG in text else text
    return body.split("{", 1)[0].strip()


def _answer_text(commits: Sequence[Mapping[str, Any]]) -> str:
    return drop_special_literals(
        "".join(str(c.get("text", "")) for c in commits if c.get("segment") == "answer")
    ).strip()


def _ends_in_thought(commits: Sequence[Mapping[str, Any]]) -> bool:
    """Ruling R1: the request's **final** state, the last commit's ``thought_open``. ``VllmClient.generate`` marks
    every commit of a canvas that ends inside a thought, so an earlier canvas's flag says nothing about the end."""
    return bool(commits) and bool(commits[-1].get("thought_open"))


def holds(answer: str, accepted: str) -> bool:
    """Ruling R2: ``accepted`` occurs in ``answer``, both case-folded, not inside a longer number or word (``16:45``
    does not hold ``6:45``). The forms themselves are the pre-registered list; nothing is derived here."""
    if not accepted:
        return False
    pattern = r"(?<![0-9a-z])" + re.escape(accepted.casefold()) + r"(?![0-9a-z])"
    return re.search(pattern, answer.casefold()) is not None


def classify_e(commits: Sequence[Mapping[str, Any]], accepted: Sequence[str]) -> str:
    """Criterion 4's outcome of one E request, from its commits (``Commit.model_dump()`` mappings). A request that
    errored is ``errored`` before this is asked (``_one``; ruling R4).

    - a ``tool`` commit outside an open thought decides it: ``google_search`` -> ``re_search``, any other call ->
      ``other`` (a call is not a final, whatever the text before it says);
    - else a request whose last commit ends inside an unclosed thought -> ``other`` (R1: the final state): it is not
      a final answer, so it is never correct, even when it quotes the fact;
    - else the answer text (special literals dropped, stripped): empty -> ``other``; holding an accepted string
      (``holds``) -> ``correct``; otherwise ``unsupported`` (a fact only inside a closed thought is not in the
      answer).
    """
    tools = [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]
    if tools:
        return "re_search" if _call_name(str(tools[0].get("text", ""))) == SEARCH else "other"
    if _ends_in_thought(commits):
        return "other"
    answer = _answer_text(commits)
    if not answer:
        return "other"
    return "correct" if any(holds(answer, a) for a in accepted) else "unsupported"


def classify_q4(commits: Sequence[Mapping[str, Any]]) -> str:
    """Criterion 2's step-2 outcome class: ``call`` (a tool commit outside an open thought), ``thought_open`` (the
    last commit ends inside a thought: R1's final state), ``final`` (a non-empty answer), else ``empty``."""
    if any(c.get("segment") == "tool" and not c.get("thought_open") for c in commits):
        return "call"
    if _ends_in_thought(commits):
        return "thought_open"
    return "final" if _answer_text(commits) else "empty"


def _spoken(commits: Sequence[Mapping[str, Any]]) -> str:
    """The step-2 text read for the repetition row: the answer commits outside an open thought, joined."""
    return drop_special_literals(
        "".join(
            str(c.get("text", ""))
            for c in commits
            if c.get("segment") == "answer" and not c.get("thought_open")
        )
    )


# =====================================================================================================
# Criterion 5: the E rule (pure, exact)
# =====================================================================================================


def _rate(cell: Mapping[str, int], *keys: str) -> Fraction:
    n = sum(int(v) for v in cell.values())
    return Fraction(sum(int(cell.get(k, 0)) for k in keys), n) if n else Fraction(0)


def e_outcome(tally: Mapping[str, Mapping[str, Mapping[str, int]]]) -> dict[str, Any]:
    """H-T115a-1 on ``tally[placement][arm][outcome]``, exact fractions of each cell's n (40 on a complete run):

    - EC: FULL correct >= 28/40 at both placements; otherwise the E rows are inconclusive;
    - E1 (kept, non-inferiority): COMPACT correct >= FULL correct - 4/40;
    - E2 (old, recovery): COMPACT correct + re-search >= 32/40;
    - E3 (old, harm): COMPACT unsupported <= 8/40.

    Each cell's n counts its ``errored`` requests (ruling R4), and ``E_ERRORED_INCONCLUSIVE`` (3) or more errored
    requests over the four cells make the outcome inconclusive whatever the bars say.

    ``outcome``: "inconclusive: <reasons>" when EC fails or 3+ requests errored; else "elision viable" iff all four
    hold; else "elision hurts: <every failing bar>"."""

    def cell(p: str, a: str) -> Mapping[str, int]:
        return (tally.get(p) or {}).get(a) or {}

    empty = [f"{p} {a}" for p in PLACEMENTS for a in E_ARMS if not sum(cell(p, a).values())]
    n_errored = sum(int(cell(p, a).get(ERRORED, 0)) for p in PLACEMENTS for a in E_ARMS)
    n_all = sum(int(v) for p in PLACEMENTS for a in E_ARMS for v in cell(p, a).values())
    bars = {
        "EC": not empty and all(_rate(cell(p, "FULL"), "correct") >= EC_MIN for p in PLACEMENTS),
        "E1": _rate(cell("kept", "COMPACT"), "correct")
        >= _rate(cell("kept", "FULL"), "correct") - E1_MARGIN,
        "E2": _rate(cell("old", "COMPACT"), "correct", "re_search") >= E2_MIN,
        "E3": bool(sum(cell("old", "COMPACT").values()))
        and _rate(cell("old", "COMPACT"), "unsupported") <= E3_MAX,
    }
    bars = {k: bool(v) for k, v in bars.items()}
    reasons: list[str] = []
    if n_errored >= E_ERRORED_INCONCLUSIVE:
        reasons.append(
            f"{n_errored}/{n_all} E requests errored (>= {E_ERRORED_INCONCLUSIVE} make E inconclusive)"
        )
    if not bars["EC"]:
        if empty:
            reasons.append(f"no requests in {', '.join(empty)}")
        else:
            low = [p for p in PLACEMENTS if _rate(cell(p, "FULL"), "correct") < EC_MIN]
            reasons.append(
                f"FULL correct below 28/40 at the {' and '.join(low)} placement "
                "(the model does not use 5-step histories)"
            )
    if reasons:
        outcome = "inconclusive: " + "; ".join(reasons)
    elif all(bars.values()):
        outcome = "elision viable"
    else:
        outcome = "elision hurts: " + ", ".join(b for b in BARS if not bars[b])
    return {"bars": bars, "outcome": outcome}


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _num(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def e_cost(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Criterion 5's report-only cost over E records:

    - ``compact_miss_tokens_median``: the median over COMPACT records of ``prefix_query_tokens -
      prefix_hit_tokens`` (warm and FULL records are not compaction costs);
    - ``first_commit_ms_diff_median`` / ``done_ms_diff_median``: the median of the *paired* COMPACT - FULL
      differences, paired by (history, placement, repeat);
    - ``ms_per_miss_token``: ``done_ms_diff_median / compact_miss_tokens_median`` (``None`` when the miss is not
      positive).
    """
    e = [r for r in records if r.get("row", "E") == "E"]
    misses: list[float] = []
    for r in e:
        if r.get("arm") != "COMPACT":
            continue
        q, h = _num(r.get("prefix_query_tokens")), _num(r.get("prefix_hit_tokens"))
        if q is not None and h is not None:
            misses.append(q - h)
    pairs: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for r in e:
        if r.get("arm") in E_ARMS:
            pairs.setdefault((r.get("history"), r.get("placement"), r.get("repeat")), {})[
                str(r["arm"])
            ] = r

    def diffs(key: str) -> list[float]:
        out: list[float] = []
        for v in pairs.values():
            if set(v) != set(E_ARMS):
                continue
            c, f = _num(v["COMPACT"].get(key)), _num(v["FULL"].get(key))
            if c is not None and f is not None:
                out.append(c - f)
        return out

    miss = _median(misses)
    done = _median(diffs("done_ms"))
    return {
        "compact_miss_tokens_median": miss,
        "compact_miss_n": len(misses),
        "first_commit_ms_diff_median": _median(diffs("first_commit_ms")),
        "done_ms_diff_median": done,
        "pairs_n": len(diffs("done_ms")),
        "ms_per_miss_token": (done / miss) if (done is not None and miss) else None,
    }


# =====================================================================================================
# The schedule (pure)
# =====================================================================================================


def _request_id(entry: Mapping[str, Any]) -> str:
    if entry["row"] == "Q4":
        return f"t0115a-q4-{entry['case_id']}-{entry['arm']}-r{entry['repeat']}"
    if entry["row"] == "E":
        return f"t0115a-e-{entry['history']}-{entry['placement']}-{entry['arm']}-r{entry['repeat']}"
    return f"t0115a-q5-{entry['case_id']}-r{entry['repeat']}"


def schedule() -> list[dict[str, Any]]:
    """Every request in send order (263):

    - Q4 (60): repeats outermost; per repeat the 6 cases in order; per (case, repeat) G then A when
      ``(case_index + repeat)`` is even, else A then G;
    - E (200): per history (in order), per placement (old, kept): before repeats 0, 2, 4, 6, 8 a warm; per repeat
      FULL then COMPACT when the repeat is even, else COMPACT then FULL (5 each way);
    - Q5 (3), last (H-T115a-3): the crash it may cause kills every request after it.
    """
    out: list[dict[str, Any]] = []
    for repeat in range(Q4_REPEATS):
        for i, case in enumerate(Q4_CASES):
            arms = Q4_ARMS if (i + repeat) % 2 == 0 else Q4_ARMS[::-1]
            for arm in arms:
                out.append({"row": "Q4", "case_id": case.case_id, "arm": arm, "repeat": repeat})
    for history in E_HISTORIES:
        for placement in PLACEMENTS:
            for repeat in range(E_REPEATS):
                base = {"row": "E", "history": history.id, "placement": placement}
                if repeat % WARM_EVERY == 0:
                    out.append({**base, "arm": WARM, "repeat": repeat})
                arms = E_ARMS if repeat % 2 == 0 else E_ARMS[::-1]
                for arm in arms:
                    out.append({**base, "arm": arm, "repeat": repeat})
    for repeat, case_id in enumerate(Q5_CASE_IDS):
        out.append({"row": "Q5", "case_id": case_id, "repeat": repeat})
    for entry in out:
        entry["request_id"] = _request_id(entry)
    return out


# =====================================================================================================
# The report (pure: reads the run directory, opens no socket, writes nothing)
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


def _kn(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


def _empty_tally() -> dict[str, dict[str, dict[str, int]]]:
    return {p: {a: dict.fromkeys(OUTCOMES, 0) for a in E_ARMS} for p in PLACEMENTS}


def _first_per_cell(
    e_lines: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float | None]]:
    """Per (history, placement) cell, in send order, after the cell's first warm (rulings 2026-10-17 and R3):
    ``miss`` the first COMPACT's ``prefix_query_tokens - prefix_hit_tokens`` (the one compaction the warm makes
    measurable); ``fc`` / ``done`` the COMPACT - FULL first-commit and done ms of the cell's **first pair** (the lowest
    repeat with both arms). ``None`` where not measured. A cell with no warm is skipped."""
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for x in e_lines:
        by_cell.setdefault((str(x.get("history")), str(x.get("placement"))), []).append(x)
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, mine in by_cell.items():
        warm_at = next((i for i, x in enumerate(mine) if x.get("arm") == WARM), None)
        if warm_at is None:
            continue
        after = mine[warm_at + 1 :]
        row: dict[str, float | None] = {"miss": None, "fc": None, "done": None}
        compact = next((x for x in after if x.get("arm") == "COMPACT"), None)
        if compact is not None:
            q, h = _num(compact.get("prefix_query_tokens")), _num(compact.get("prefix_hit_tokens"))
            row["miss"] = None if q is None or h is None else q - h
        by_rep: dict[Any, dict[str, Mapping[str, Any]]] = {}
        for x in after:
            if x.get("arm") in E_ARMS:
                by_rep.setdefault(x.get("repeat"), {}).setdefault(str(x["arm"]), x)
        both = sorted(r for r, v in by_rep.items() if set(v) == set(E_ARMS))
        if both:
            pair = by_rep[both[0]]
            for name, col in (("fc", "first_commit_ms"), ("done", "done_ms")):
                c, f = _num(pair["COMPACT"].get(col)), _num(pair["FULL"].get(col))
                row[name] = None if c is None or f is None else c - f
        out[key] = row
    return out


def _placement_cost(
    cells: Sequence[Mapping[str, float | None]], predicted: Sequence[float]
) -> dict[str, Any]:
    """Ruling R3: one placement's report-only cost from its cells' first COMPACT and first pair. At ``old`` this is
    the one-compaction cost; ``kept`` shares a cached prefix with the same history's old COMPACT (its predicted miss is
    counted from that divergence) and is reported separately."""
    misses = [m for c in cells if (m := c["miss"]) is not None]
    miss = _median(misses)
    done = _median([d for c in cells if (d := c["done"]) is not None])
    return {
        "compact_miss_tokens_first_per_cell_median": miss,
        "compact_miss_tokens_first_per_cell_n": len(misses),
        "predicted_miss_tokens_median": _median(predicted),
        "first_commit_ms_diff_median": _median([f for c in cells if (f := c["fc"]) is not None]),
        "done_ms_diff_median": done,
        "ms_per_miss_token": (done / miss) if (done is not None and miss) else None,
    }


def _q4_report(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in Q4_ARMS:
        mine = [x for x in lines if x.get("row") == "Q4" and x.get("arm") == arm]
        n = len(mine)
        per_case: dict[str, dict[str, Any]] = {}
        for x in mine:
            row = per_case.setdefault(
                str(x.get("case_id")), {**dict.fromkeys(Q4_OUTCOMES, 0), "repeats_start": 0, "n": 0}
            )
            row["n"] += 1
            if x.get("outcome") in Q4_OUTCOMES:
                row[str(x["outcome"])] += 1
            row["repeats_start"] += int(bool(x.get("repeats_start")))
        out[arm] = {
            "outcomes": {
                o: _kn(sum(1 for x in mine if x.get("outcome") == o), n) for o in Q4_OUTCOMES
            },
            "errored": _kn(sum(1 for x in mine if x.get("error")), n),
            "repeats_start": _kn(sum(1 for x in mine if x.get("repeats_start")), n),
            "per_case": per_case,
            "done_ms_median": _median(
                [v for x in mine if (v := _num(x.get("done_ms"))) is not None]
            ),
        }
    return out


def build_report(run_dir: Path) -> dict[str, Any]:
    """``report.json``'s content, from ``manifest.json`` and ``requests.jsonl`` alone: Q4 (report-only shares and
    the repetition rate per arm), E (the tally, the bars, the outcome and the cost) and Q5 (the readings in order).
    Writes nothing."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    lines = _read_jsonl(run_dir / "requests.jsonl")
    tally = _empty_tally()
    for x in lines:
        if x.get("row") == "E" and x.get("arm") in E_ARMS and x.get("outcome") in OUTCOMES:
            tally[str(x["placement"])][str(x["arm"])][str(x["outcome"])] += 1
    rule = e_outcome(tally)
    e_lines = [x for x in lines if x.get("row") == "E"]
    cost = e_cost(e_lines)
    predicted_by_key: Mapping[str, Any] = manifest.get("predicted_miss_tokens") or {}
    cells = _first_per_cell(e_lines)
    cost["compact_miss_tokens_first_per_cell_median"] = _median(
        [m for c in cells.values() if (m := c["miss"]) is not None]
    )
    cost["compact_miss_tokens_first_per_cell_n"] = sum(
        1 for c in cells.values() if c["miss"] is not None
    )
    cost["predicted_miss_tokens_median"] = _median(
        [n for v in predicted_by_key.values() if (n := _num(v)) is not None]
    )
    cost["by_placement"] = {
        p: _placement_cost(
            [c for (_, cp), c in cells.items() if cp == p],
            [
                n
                for k, v in predicted_by_key.items()
                if str(k).endswith(f"/{p}") and (n := _num(v)) is not None
            ],
        )
        for p in PLACEMENTS
    }
    planned = schedule()
    want = {row: sum(1 for e in planned if e["row"] == row) for row in ("Q4", "E", "Q5")}
    got = {row: sum(1 for x in lines if x.get("row") == row) for row in ("Q4", "E", "Q5")}
    readings = [str(x.get("reading")) for x in lines if x.get("row") == "Q5"]
    return {
        "task": "T-0115a",
        "regime": manifest.get("regime", {}),
        "requests": {"planned": want, "written": got},
        "complete": got == want and not manifest.get("stopped"),
        "stopped": manifest.get("stopped"),
        "Q4": _q4_report(lines),
        "E": {
            "tally": tally,
            "bars": rule["bars"],
            "outcome": rule["outcome"],
            "cost": cost,
            "thresholds": {
                "EC": "FULL correct >= 28/40 at both placements",
                "E1": "kept: COMPACT correct >= FULL correct - 4/40",
                "E2": "old: COMPACT correct + re-search >= 32/40",
                "E3": "old: COMPACT unsupported <= 8/40",
                "errored": f">= {E_ERRORED_INCONCLUSIVE}/160 errored E requests make E inconclusive",
            },
        },
        "Q5": {"readings": readings},
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """``report.md``: every k/n of ``report`` beside its threshold, and the regime."""
    e = report["E"]
    lines = [
        "# T-0115a -- tool-loop request rows (Q4, E, Q5)",
        "",
        f"**E outcome:** {e['outcome']}",
        "",
        f"Regime: {json.dumps(report.get('regime', {}), ensure_ascii=False)}",
        "",
        f"Requests planned {report['requests']['planned']}, written {report['requests']['written']}; "
        f"complete: {report['complete']}; stopped: {report.get('stopped') or 'no'}",
        "",
        "## E -- the compaction A/B (H-T115a-1)",
        "",
        "| placement | arm | correct | re-search | unsupported | other | errored |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in PLACEMENTS:
        for a in E_ARMS:
            cell = e["tally"][p][a]
            n = sum(cell.values())
            lines.append(f"| {p} | {a} | " + " | ".join(f"{cell[o]}/{n}" for o in OUTCOMES) + " |")
    lines += ["", "| bar | rule | holds |", "|---|---|---|"]
    for b in BARS:
        lines.append(f"| {b} | {e['thresholds'][b]} | {e['bars'][b]} |")
    c = e["cost"]
    lines += [
        "",
        "Cost (report-only; ruling R3). Per placement, from each cell's first COMPACT after its warm and its first "
        "pair; **old is the one-compaction cost**, kept shares a cached prefix with the same history's old COMPACT "
        "and is reported separately:",
        "",
        "| placement | first COMPACT miss tokens (n) | predicted | first-pair first-commit ms | "
        "first-pair done ms | ms per miss token |",
        "|---|---|---|---|---|---|",
    ]
    for p in PLACEMENTS:
        b = c["by_placement"][p]
        lines.append(
            f"| {p} | {_fmt(b['compact_miss_tokens_first_per_cell_median'])} "
            f"({b['compact_miss_tokens_first_per_cell_n']}) | {_fmt(b['predicted_miss_tokens_median'])} | "
            f"{_fmt(b['first_commit_ms_diff_median'])} | {_fmt(b['done_ms_diff_median'])} | "
            f"{_fmt(b['ms_per_miss_token'])} |"
        )
    lines += [
        "",
        "Cache sanity values (all cells and repeats pooled; expected near zero under prefix caching): "
        f"median COMPACT miss tokens {_fmt(c['compact_miss_tokens_median'])} (n {c['compact_miss_n']}); "
        f"first COMPACT per cell, both placements {_fmt(c['compact_miss_tokens_first_per_cell_median'])} "
        f"(n {c['compact_miss_tokens_first_per_cell_n']}); "
        f"predicted {_fmt(c['predicted_miss_tokens_median'])}; "
        f"paired COMPACT - FULL first-commit ms {_fmt(c['first_commit_ms_diff_median'])}, "
        f"done ms {_fmt(c['done_ms_diff_median'])} (n {c['pairs_n']}); "
        f"ms per miss token {_fmt(c['ms_per_miss_token'])}.",
        "",
        "## Q4 -- form G vs form A (report-only)",
        "",
        "| arm | " + " | ".join(Q4_OUTCOMES) + " | errored | repeats start |",
        "|---|" + "---|" * (len(Q4_OUTCOMES) + 2),
    ]
    for arm in Q4_ARMS:
        r = report["Q4"][arm]
        cells = [f"{r['outcomes'][o]['k']}/{r['outcomes'][o]['n']}" for o in Q4_OUTCOMES]
        lines.append(
            f"| {arm} | "
            + " | ".join(cells)
            + f" | {r['errored']['k']}/{r['errored']['n']}"
            + f" | {r['repeats_start']['k']}/{r['repeats_start']['n']} |"
        )
    lines += [
        "",
        "## Q5 -- an image part in a continued assistant message",
        "",
        "Readings, in order: " + (", ".join(report["Q5"]["readings"]) or "none sent"),
        "",
    ]
    return "\n".join(lines)


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied; imported and opened inside main only)
# =====================================================================================================


def _frame(case: Mapping[str, Any]) -> Any:
    from local_astra.contracts.perception import Frame, Source

    jpeg = (SET_DIR / str(case["screen"])).read_bytes()
    w, h = case["screen_size"]
    return Frame(
        source=Source.SCREEN,
        ts_ms=0,
        width=int(w),
        height=int(h),
        jpeg=jpeg,
        phash=hashlib.sha256(jpeg).hexdigest()[:16],
    )


def _base_prompt(builder: Any, text: str, case: Mapping[str, Any] | None) -> Prompt:
    """A step-1 prompt as a fresh session builds it for one typed turn: no history, the case's screen (if any) as
    the one frame, no assists, no memory hits."""
    from local_astra.contracts.perception import AssistFlags

    frames = (_frame(case),) if case is not None and case.get("screen") else ()
    prefix = builder.prefix([], None, AssistFlags(), None, frames=frames)
    full: Prompt = builder.full(prefix, text, None, ())
    return full


def _q5_body(
    client: VllmClient, base: Prompt, step: LoopStep, jpeg: bytes, request_id: str
) -> dict[str, Any]:
    """Q5: ``base``'s body, then a final assistant message continued from the step-1 continuation text and an
    image part (the case's screen after the dry-run action), ``continue_final_message`` on."""
    from local_astra.contracts.llm import LlmOptions

    body = client.request_body(base, request_id, LlmOptions())
    text = loop.step_prompt(base, (step,)).messages[-1].text()
    body["messages"] = [
        *body["messages"],
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                    },
                },
            ],
        },
    ]
    body["continue_final_message"] = True
    body["add_generation_prompt"] = False
    return body


def _commit_dict(c: Any) -> dict[str, Any]:
    out: dict[str, Any] = c.model_dump()
    return out


async def _read_raw(
    response: httpx.Response, detokenizer: HostDetokenizer, request_id: str
) -> AsyncIterator[dict[str, Any]]:
    """A raw tools-on stream read by the client's own id reader, as ``VllmClient.generate`` reads one (one commit
    per piece per canvas; a fence still open at the end published as ``ui``; the tool commit last)."""
    from local_astra.conductor import llm_client as lc
    from local_astra.contracts.llm import Commit

    reader = lc._IdReader(detokenizer, None)
    canvas_idx = 0
    async for line in response.aiter_lines():
        payload = lc._sse_payload(line)
        if payload is None:
            continue
        if payload == lc.SSE_DONE:
            break
        pieces, thought_open = reader.feed(json.loads(payload), canvas_idx)
        emitted = False
        for piece in pieces:
            if not piece.text:
                continue
            yield _commit_dict(
                Commit(
                    request_id=request_id,
                    canvas_idx=canvas_idx,
                    segment=piece.segment,
                    text=piece.text,
                    thought_open=thought_open,
                    fence_close=piece.fence,
                    fence_continued=piece.continued,
                )
            )
            emitted = True
        if emitted:
            canvas_idx += 1
    if not reader.finished:
        raise lc.LlmWireError(f"{request_id}: the stream ended without a finish_reason")
    fence = reader.fence
    shown = drop_special_literals(fence.body)
    if fence.open and shown:
        yield _commit_dict(
            Commit(
                request_id=request_id,
                canvas_idx=canvas_idx,
                segment="ui",
                text=shown,
                fence_close="open",
            )
        )
    tool = reader.tool_text()
    if tool is not None:
        yield _commit_dict(
            Commit(
                request_id=request_id,
                canvas_idx=reader.last_canvas,
                segment="tool",
                text=tool,
                stop_id=reader.stop_id,
            )
        )


@dataclass
class _Run:
    out: Path
    base_url: str
    client: VllmClient
    raw: httpx.AsyncClient
    detokenizer: HostDetokenizer
    manifest: dict[str, Any]
    #: (row, key) -> the request's prompt or body
    g_prompts: dict[str, Prompt]
    a_bodies: dict[str, dict[str, Any]]
    e_arms: dict[tuple[str, str, str], Prompt]
    q5_bodies: dict[str, dict[str, Any]]
    prompt_tokens: dict[str, int | None]
    errors_in_a_row: int = 0
    written: int = 0


def _metrics(base_url: str) -> dict[str, int] | None:
    found: dict[str, int] | None = _T93.read_metrics(base_url)
    return found


def _delta(before: dict[str, int] | None, after: dict[str, int] | None, name: str) -> int | None:
    if before is None or after is None or name not in before or name not in after:
        return None
    return after[name] - before[name]


def _healthy(base_url: str) -> bool:
    return _T93._get_text(_T93._root(base_url) + "/health") is not None


def _write_line(run: _Run, line: Mapping[str, Any]) -> None:
    with (run.out / "requests.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    run.written += 1


async def _one(run: _Run, entry: Mapping[str, Any]) -> dict[str, Any]:
    """One Q4 or E request, timed on this process's monotonic clock, with the ``/metrics`` deltas around it."""
    import httpx

    from local_astra.conductor import llm_client as lc
    from local_astra.contracts.llm import LlmOptions

    rid = str(entry["request_id"])
    commits: list[dict[str, Any]] = []
    error: str | None = None
    status: int | None = None
    before = _metrics(run.base_url)
    started = time.perf_counter()
    first: float | None = None
    try:
        if entry["row"] == "Q4" and entry["arm"] == "A":
            body = dict(run.a_bodies[str(entry["case_id"])])
            body["vllm_xargs"] = {**dict(body.get("vllm_xargs") or {}), "astra_request_id": rid}
            async with run.raw.stream(
                "POST", lc.CHAT_COMPLETIONS_PATH.lstrip("/"), json=body
            ) as resp:
                status = resp.status_code
                resp.raise_for_status()
                async for c in _read_raw(resp, run.detokenizer, rid):
                    first = time.perf_counter() if first is None else first
                    commits.append(c)
        else:
            if entry["row"] == "Q4":
                prompt = run.g_prompts[str(entry["case_id"])]
            else:
                prompt = run.e_arms[
                    (str(entry["history"]), str(entry["placement"]), str(entry["arm"]))
                ]
            async for c in run.client.generate(prompt, rid, LlmOptions(), xargs=None):
                first = time.perf_counter() if first is None else first
                commits.append(_commit_dict(c))
            status = 200
    except httpx.HTTPStatusError as exc:
        error = repr(exc)
        status = exc.response.status_code
    except (httpx.HTTPError, httpx.StreamError, lc.LlmWireError, OSError, ValueError) as exc:
        error = repr(exc)
    ended = time.perf_counter()
    after = _metrics(run.base_url)
    line: dict[str, Any] = {k: entry[k] for k in entry}
    key = _prompt_key(entry)
    line.update(
        {
            "status": status,
            "error": error,
            "commits": commits,
            "prompt_tokens": run.prompt_tokens.get(key),
            "prefix_query_tokens": _delta(before, after, _T93.METRIC_QUERIES),
            "prefix_hit_tokens": _delta(before, after, _T93.METRIC_HITS),
            "first_commit_ms": None if first is None else round((first - started) * 1000.0, 1),
            "done_ms": round((ended - started) * 1000.0, 1),
        }
    )
    if entry["row"] == "E":
        # ruling R4: a request that errored is errored, whatever its commits hold (a truncated stream can carry the
        # fact before the client raises)
        line["outcome"] = (
            ERRORED
            if error is not None or status != 200
            else classify_e(commits, _HISTORY_BY_ID[str(entry["history"])].accepted)
        )
    else:
        step = _Q4_BY_ID[str(entry["case_id"])].step
        text2 = _spoken(commits)
        line["outcome"] = classify_q4(commits)
        line["step2_text"] = text2
        line["repeats_start"] = repeats_start(step.text, text2)
    return line


async def _q5(run: _Run, entry: Mapping[str, Any]) -> dict[str, Any]:
    """One Q5 request: accepted (200 and a stream that ends), rejected (anything else while ``/health`` answers) or
    engine-down (``/health`` does not answer after it, on any of ``HEALTH_CHECKS`` probes ``HEALTH_RETRY_S`` apart:
    ruling R8, so a blip is not read as a dead EngineCore)."""
    import httpx

    from local_astra.conductor import llm_client as lc

    rid = str(entry["request_id"])
    body = dict(run.q5_bodies[str(entry["case_id"])])
    body["vllm_xargs"] = {**dict(body.get("vllm_xargs") or {}), "astra_request_id": rid}
    status: int | None = None
    error: str | None = None
    ended_cleanly = False
    text_bytes = 0
    started = time.perf_counter()
    try:
        async with run.raw.stream("POST", lc.CHAT_COMPLETIONS_PATH.lstrip("/"), json=body) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                text_bytes += len(line)
                payload = lc._sse_payload(line)
                if payload == lc.SSE_DONE:
                    ended_cleanly = True
                    break
                if payload and status == 200:
                    with_finish = lc._first_choice(json.loads(payload))
                    if with_finish is not None and with_finish.get("finish_reason") is not None:
                        ended_cleanly = True
    except (httpx.HTTPError, httpx.StreamError, OSError, ValueError) as exc:
        error = repr(exc)
    ended = time.perf_counter()
    # ruling R8: one failed probe is not engine-down; ask up to HEALTH_CHECKS times, HEALTH_RETRY_S apart
    healthy = False
    checks = 0
    while checks < HEALTH_CHECKS:
        if checks:
            await asyncio.sleep(HEALTH_RETRY_S)
        checks += 1
        if _healthy(run.base_url):
            healthy = True
            break
    if not healthy:
        reading = "engine-down"
    elif status == 200 and ended_cleanly and error is None:
        reading = "accepted"
    else:
        reading = "rejected"
    line: dict[str, Any] = {k: entry[k] for k in entry}
    line.update(
        {
            "reading": reading,
            "status": status,
            "error": error,
            "health_after": healthy,
            "health_checks": checks,
            "response_chars": text_bytes,
            "done_ms": round((ended - started) * 1000.0, 1),
        }
    )
    return line


async def _run_all(run: _Run) -> None:
    """The schedule, in order, at concurrency 1. Q4/E: ``STOP_AFTER_ERRORS`` errored requests in a row stop the run
    (Q5 is then not sent: the engine is gone). Q5: an engine-down reading stops the run."""
    try:
        for entry in schedule():
            if entry["row"] == "Q5":
                line = await _q5(run, entry)
                _write_line(run, line)
                _say(f"{entry['request_id']}: {line['reading']} (status {line['status']})")
                if line["reading"] == "engine-down":
                    run.manifest["stopped"] = f"engine down after {entry['request_id']} (Q5)"
                    _say(f"stopping: {run.manifest['stopped']}")
                    return
                continue
            line = await _one(run, entry)
            _write_line(run, line)
            _say(
                f"{entry['request_id']}: {line['outcome']} done={line['done_ms']}ms"
                + (f" ERROR {line['error']}" if line["error"] else "")
            )
            run.errors_in_a_row = 0 if line["error"] is None else run.errors_in_a_row + 1
            if run.errors_in_a_row >= STOP_AFTER_ERRORS:
                run.manifest["stopped"] = (
                    f"{run.errors_in_a_row} consecutive requests errored (last: {entry['request_id']})"
                )
                _say(f"stopping: {run.manifest['stopped']}")
                return
    finally:
        await run.client.aclose()


_Q4_BY_ID: dict[str, Q4Case] = {c.case_id: c for c in Q4_CASES}


def _prompt_key(entry: Mapping[str, Any]) -> str:
    if entry["row"] == "Q4":
        return f"Q4/{entry['case_id']}/{entry['arm']}"
    return f"E/{entry['history']}/{entry['placement']}/{entry['arm']}"


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _tokens(base_url: str, body: Mapping[str, Any]) -> int | None:
    ids = _T93._tokenize_chat(base_url, body)
    return None if ids is None else len(ids)


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
    out: Path = args.out_dir or REPO / "logs" / f"t0115a-loop-requests-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
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
    if kit.tools_version[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {kit.tools_version[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED
    try:
        raw_cases = CASES_PATH.read_bytes()
        rows = {
            str(r["id"]): r
            for r in (json.loads(x) for x in raw_cases.decode("utf-8").splitlines() if x.strip())
        }
    except (OSError, ValueError, KeyError) as exc:
        _say(f"refusing: {CASES_PATH.relative_to(REPO)} cannot be read: {exc}")
        return EXIT_REFUSED
    not_dev = [cid for cid in DEV_CASE_IDS if (rows.get(cid) or {}).get("split") != SPLIT]
    if not_dev:
        _say(
            f"refusing: case ids that are not dev-split cases of {CASES_PATH.name}: {', '.join(not_dev)} "
            "(the test split is never read)"
        )
        return EXIT_REFUSED
    got_sha = hashlib.sha256(raw_cases).hexdigest()
    if got_sha != CASES_SHA256:
        _say(f"refusing: {CASES_PATH.name} has sha256 {got_sha}, not the pinned {CASES_SHA256}")
        return EXIT_REFUSED
    wrong_tool = [c.case_id for c in Q4_CASES if rows[c.case_id].get("tool") != c.step.name]
    if wrong_tool:
        _say(f"refusing: Q4 cases whose tool is not their step's: {wrong_tool}")
        return EXIT_REFUSED
    elided = {f"{x.id}/{p}": elided_steps(x, p) for x in E_HISTORIES for p in PLACEMENTS}
    off = {k: v for k, v in elided.items() if v != [0, 1, 2]}
    if off:
        _say(f"refusing: E_BUDGET {E_BUDGET} does not elide exactly steps 1-3: {off}")
        return EXIT_REFUSED

    import httpx

    from local_astra.conductor import prompt_builder as pb
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT

    try:
        endpoint = VllmEndpoint(base_url=args.base_url, request_timeout_s=REQUEST_TIMEOUT_S)
    except ValueError as exc:
        _say(f"refusing: not a loopback URL the client accepts: {exc}")
        return EXIT_REFUSED
    thinking = bool(profile.diffusion.thinking)
    builder = pb.PromptBuilder(DEFAULT_SYSTEM_PROMPT, thinking=thinking, tools=kit.tools)
    client = VllmClient(endpoint, transport=httpx.AsyncHTTPTransport(), detokenizer=detokenizer)
    g_prompts: dict[str, Prompt] = {}
    a_bodies: dict[str, dict[str, Any]] = {}
    q5_bodies: dict[str, dict[str, Any]] = {}
    bodies: dict[str, dict[str, Any]] = {}
    for case in Q4_CASES:
        row = rows[case.case_id]
        base = _base_prompt(builder, str(row["user"]), row)
        g_prompts[case.case_id] = loop.step_prompt(base, (case.step,))
        a_bodies[case.case_id] = form_a_body(base, case.step, "body", client)
        bodies[f"Q4/{case.case_id}/G"] = client.request_body(g_prompts[case.case_id], "body")
        bodies[f"Q4/{case.case_id}/A"] = a_bodies[case.case_id]
        if case.case_id in Q5_CASE_IDS:
            jpeg = (SET_DIR / str(row["screen"])).read_bytes()
            q5_bodies[case.case_id] = _q5_body(client, base, case.step, jpeg, "body")
    e_arms: dict[tuple[str, str, str], Prompt] = {}
    cut_bodies: dict[str, dict[str, Any]] = {}
    for history in E_HISTORIES:
        base = _base_prompt(builder, history.question, rows[E_SCREEN_CASE])
        for placement in PLACEMENTS:
            arms = e_prompts(base, history, placement)
            for arm, prompt in arms.items():
                e_arms[(history.id, placement, arm)] = prompt
                bodies[f"E/{history.id}/{placement}/{arm}"] = client.request_body(prompt, "body")
            # the predicted miss. old: the COMPACT prompt's tokens after the first elided block (its warm, steps
            # 1-4 in full, is cached up to there). kept (ruling R3): the kept COMPACT shares its stubs and step 4 with
            # the same history's old COMPACT, sent earlier, so its predicted miss counts from where the two diverge
            compact_body = bodies[f"E/{history.id}/{placement}/COMPACT"]
            text = compact_body["messages"][-1]["content"]
            if placement == "old":
                cut_at = text.index(
                    loop.render_response(
                        SEARCH, loop.elided_payload(e_steps(history, placement)[0].response)
                    )
                )
            else:
                old_text = bodies[f"E/{history.id}/old/COMPACT"]["messages"][-1]["content"]
                cut_at = len(os.path.commonprefix([text, old_text]))
            cut = json.loads(json.dumps(compact_body))
            cut["messages"][-1]["content"] = text[:cut_at]
            cut_bodies[f"{history.id}/{placement}"] = cut

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
    # outside every timed span: the prompts' engine token counts and the predicted misses
    prompt_tokens = {key: _tokens(args.base_url, body) for key, body in bodies.items()}
    predicted: dict[str, int | None] = {}
    for key, cut in cut_bodies.items():
        whole = prompt_tokens.get(f"E/{key}/COMPACT")
        head = _tokens(args.base_url, cut)
        predicted[key] = None if whole is None or head is None else whole - head

    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": "T-0115a",
        "rule": "H-T115a-1",
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": endpoint.model,
            "engine_version": _T93._get_text(_T93._root(args.base_url) + "/version"),
            "versions_lock_sha256": _T93._sha256(REPO / "services" / "versions.lock"),
            "cases_sha256": CASES_SHA256,
            "split": SPLIT,
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
            "e_budget": E_BUDGET,
            "compact": "loop.compact(steps, E_BUDGET) at its defaults (keep_last 2, target budget * 2 // 3)",
            "warm": f"the step-4 prompt before every repeat r with r % {WARM_EVERY} == 0, per (history, placement)",
            "e_screen_case": E_SCREEN_CASE,
            "q5_cases": list(Q5_CASE_IDS),
            "accepted_match": "the complete pre-registered list per history (e_histories), case-insensitive, "
            "not inside a longer number or word (R2)",
            "thought_open": "the last commit's (the final state, R1)",
            "errored": f"non-200, timeout, transport error or a stream without a finish_reason; never correct; "
            f">= {E_ERRORED_INCONCLUSIVE}/160 make E inconclusive (R4)",
            "q5_health": f"up to {HEALTH_CHECKS} /health probes {HEALTH_RETRY_S:g} s apart (R8)",
            "form_a": "assistant content = step 1's text, plus one tool_calls entry (R6)",
            "predicted_miss": "old: tokens after the first stub; kept: tokens after the divergence from the "
            "same history's old COMPACT (R3)",
        },
        "gate_shas": {p: _T93._sha256(REPO / p) for p in GATE_FILES},
        "e_histories": {
            x.id: {"question": x.question, "fact": x.fact, "accepted": list(x.accepted)}
            for x in E_HISTORIES
        },
        "e_texts_sha256": {
            f"{x.id}/{p}/{arm}": _sha(e_arms[(x.id, p, arm)].messages[-1].text())
            for x in E_HISTORIES
            for p in PLACEMENTS
            for arm in (WARM, *E_ARMS)
        },
        "elided": elided,
        "prompt_tokens": prompt_tokens,
        "predicted_miss_tokens": predicted,
        "q4_cases": [
            {"case_id": c.case_id, "tool": c.step.name, "text": c.step.text} for c in Q4_CASES
        ],
    }
    _write_json(out / "manifest.json", manifest)
    raw = httpx.AsyncClient(
        base_url=args.base_url,
        transport=httpx.AsyncHTTPTransport(),
        timeout=REQUEST_TIMEOUT_S,
        trust_env=False,
    )
    run = _Run(
        out=out,
        base_url=args.base_url,
        client=client,
        raw=raw,
        detokenizer=detokenizer,
        manifest=manifest,
        g_prompts=g_prompts,
        a_bodies=a_bodies,
        e_arms=e_arms,
        q5_bodies=q5_bodies,
        prompt_tokens=prompt_tokens,
    )

    async def go() -> None:
        try:
            await _run_all(run)
        finally:
            await raw.aclose()

    asyncio.run(go())
    manifest["requests_written"] = run.written
    _write_json(out / "manifest.json", manifest)

    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _say(f"E outcome: {report['E']['outcome']}; Q5: {report['Q5']['readings']}")
    _say(f"written to {out}")
    if manifest.get("stopped") or not report["complete"]:
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
