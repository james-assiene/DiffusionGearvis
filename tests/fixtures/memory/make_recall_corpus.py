#!/usr/bin/env python3
"""Build ``recall_corpus.json``: 40 hand-written captions plus 360 filler ones (T-0053).

Why it exists: criteria 13 and 14 need a corpus where the *document frequency* of a term is known
by construction, because that is the thing the new retrieval rule is allowed to look at. The
constraints below are asserted here, so a caption edited later either keeps them or fails the
build:

* every caption contains the standalone word ``a`` — so ``"Make a flat looking UI."`` matches the
  whole corpus on ``a`` alone, at **every** corpus size (criterion 14);
* no caption contains ``make``/``flat``/``look``/``ui`` in any form the ``porter`` stemmer folds
  together — so ``a`` is the *only* term that query can match;
* ``flight`` appears in exactly one caption and ``glasses`` in exactly one, and neither caption
  contains any other word of its query ("what time was that flight", "where did I leave my
  glasses") — that is what makes those queries match on a **single content term** (criterion 13);
* ``desk`` appears in exactly three captions, and those three are the ``small_subset``: inside that
  subset ``desk`` is universal (3/3) and inside the 400-episode corpus it is rare (3/400). One term,
  two verdicts, decided only by document frequency — the row a bare score floor cannot pass;
* the words of all four fixture queries (``what time was that flight``, ``where did I leave my
  glasses``, ``Make a flat looking UI.``, ``desk``) appear nowhere else, so hand-counting the
  document frequencies is possible and is done in ``MANIFEST.md``.

``ts_ms`` are synthetic and ordered so that the flight and glasses episodes are **not** among the
three newest: a retrieval that has fallen through to recency (the origin bug) cannot return them
by accident.

Usage (from the repository root)::

    uv run python tests/fixtures/memory/make_recall_corpus.py
    uv run python tests/fixtures/memory/make_recall_corpus.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recall_corpus.json"

BASE_TS_MS = 1_768_000_000_000  # arbitrary, fixed: 2026-01-09T22:26:40Z
STEP_MS = 60_000

#: Words that must not occur (after crude stemming) anywhere in the corpus: every word of the
#: three queries, minus the two that are planted on purpose.
FORBIDDEN = {
    "make",
    "flat",
    "look",
    "ui",
    "what",
    "time",
    "was",
    "that",
    "where",
    "did",
    "leave",
    "my",
    "i",
}

CAPTIONS: list[str] = [
    "A terminal window running a package install, with a progress bar and green status lines.",
    "A spreadsheet of monthly expenses, one column per category, totals at the bottom.",
    "A code editor showing a Python file with a function and its docstring.",
    "A browser on a news article about a harbour redevelopment, with a large photograph.",
    "A chat window with a short conversation and an unread badge on the sidebar.",
    "A map application centred on a coastal town, with a route drawn in blue.",
    "A pair of reading glasses resting on a desk beside a closed paperback.",
    "A flight confirmation on the screen of a laptop on a desk, seat fourteen C, "
    "boarding at six forty in the evening.",
    "A mug of coffee on a desk next to a stack of unopened envelopes.",
    "A photograph of a kitchen counter with a chopping board, onions and a knife.",
    "A recipe for lentil soup, ingredients on the left and steps on the right.",
    "A calendar month view with three coloured events on consecutive days.",
    "A PDF of a rental agreement, clause four highlighted in yellow.",
    "A music player with an album cover and a progress slider near the end.",
    "A video call with four participants and a shared presentation slide.",
    "A settings panel with toggles for notifications and a search box at the top.",
    "A photograph of a bicycle leaning against a brick wall.",
    "A dashboard of server graphs, one line spiking around midnight.",
    "A page of a calculus textbook with exercises on parametrized curves.",
    "A page of a chemistry textbook with a diagram of an atom.",
    "A shopping cart with three items and an estimated delivery date.",
    "A photograph of a dog asleep on a sofa in the afternoon sun.",
    "A bank statement with a list of card payments and a closing balance.",
    "A drawing application with a half finished sketch of a house.",
    "A weather forecast for the week with rain on Thursday and Friday.",
    "A train ticket with a platform number and a seat reservation.",
    "A podcast episode page with a description and a subscribe button.",
    "A photograph of a bookshelf, spines in many colours, one shelf empty.",
    "A git history graph with several branches and a merge commit.",
    "A note application with a bulleted shopping list and two checked items.",
    "A crossword puzzle half completed, a pencil resting on the page.",
    "A photograph of a park bench under a large chestnut tree.",
    "A spreadsheet of a football league table sorted by points.",
    "A slide deck title page with a company logo and a date.",
    "A photograph of a bowl of ramen with an egg and spring onions.",
    "A support ticket thread with a reply from an engineer and a status of open.",
    "A photograph of a whiteboard covered in boxes and arrows.",
    "A language learning app with a vocabulary card and an audio button.",
    "A photograph of a rain streaked window with a city blurred behind it.",
    "A printer dialog with a page range and a two sided printing option.",
]

#: Deterministic filler, expanded to 360 captions. Every one starts with "A " (so ``a`` stays
#: universal) and uses vocabulary disjoint from the four fixture queries.
_SUBJECTS = ("report", "invoice", "diagram", "photograph", "form", "receipt", "poster", "chart")
_SCENES = ("on a tablet", "in a browser tab", "on a projector", "in a print preview", "on a phone")
_DETAILS = (
    "a blue header",
    "a narrow margin",
    "a serif heading",
    "a small footer",
    "a numbered caption",
    "a dotted border",
    "a shaded row",
    "a rounded corner",
    "a thin rule",
)


def filler(count: int) -> list[str]:
    out = []
    for index in range(count):
        subject = _SUBJECTS[index % len(_SUBJECTS)]
        scene = _SCENES[(index // len(_SUBJECTS)) % len(_SCENES)]
        detail = _DETAILS[(index // (len(_SUBJECTS) * len(_SCENES))) % len(_DETAILS)]
        out.append(f"A {subject} number {index} {scene}, with {detail}.")
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _stem(word: str) -> str:
    """Crude suffix strip, wider than porter on purpose: it only has to over-reject."""
    for suffix in ("ings", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)]
    return word


def _check(captions: list[str]) -> None:
    for caption in captions:
        words = _words(caption)
        assert "a" in words, caption
        for word in words:
            assert _stem(word) not in FORBIDDEN and word not in FORBIDDEN, (word, caption)
    flight = [c for c in captions if "flight" in _words(c)]
    glasses = [c for c in captions if "glasses" in _words(c)]
    desk = [c for c in captions if "desk" in _words(c)]
    assert len(flight) == 1 and len(glasses) == 1 and len(desk) == 3, (
        len(flight),
        len(glasses),
        len(desk),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fillers = filler(360)
    _check(CAPTIONS + fillers)
    episodes = [
        {
            "key": f"c{index:02d}",
            "ts_ms": BASE_TS_MS + index * STEP_MS,
            "caption": caption,
        }
        for index, caption in enumerate(CAPTIONS)
    ]
    filler_rows = [
        {
            "key": f"f{index:03d}",
            "ts_ms": BASE_TS_MS - (index + 1) * STEP_MS,
            "caption": caption,
        }
        for index, caption in enumerate(fillers)
    ]
    payload = {
        "provenance": {
            "generator": "tests/fixtures/memory/make_recall_corpus.py",
            "hand_written": len(episodes),
            "filler": len(filler_rows),
            "note": (
                "captions invented for T-0053 criteria 13/14; no user data. The constraints they "
                "satisfy are asserted by the generator and listed in MANIFEST.md."
            ),
        },
        "queries": {
            "rejected": "Make a flat looking UI.",
            "flight": "what time was that flight",
            "glasses": "where did I leave my glasses",
            "universal_in_the_small_subset": "desk",
        },
        "small_subset": ["c06", "c07", "c08"],
        "flight_key": "c07",
        "glasses_key": "c06",
        "episodes": episodes,
        "filler": filler_rows,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("ok" if same else "DIFFERS")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(episodes)} + {len(filler_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
