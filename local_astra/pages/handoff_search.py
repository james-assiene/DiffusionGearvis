"""The loopback search index for the shared-screen handoffs H1-H3 (T-0211 C1): a pure, static ``SearchBackend``.

``HandoffSearch`` answers ``google_search`` in part H of the T-0211 v3-vs-v4 GPU A/B (``scripts/gpu/t0211_guidance_ab.py``)
with rows from ``SEARCH_ROWS``. It is not a network exception: it opens nothing, reads no file and imports no network
module. Only tests and the T-0211 harness construct it; ``cli`` never does. It has the shape of ``rtx_search``
(T-0209 A1.7), which it does not edit.

- ``SEARCH_ROWS`` is a tuple of ``(title, url, snippet, keywords)``, one row per handoff. Each URL is ``https://`` and
  its host words, typed into the scenarios start page's address box (``scenarios.js``'s ``words()`` and ``IGNORED``),
  leave exactly one site shown: the handoff's start page (H1 ``/breville/``, H2 ``/outdoor/stoves.html``, H3
  ``/pizza/``).
- No snippet names a handoff's item: no word of any ``handoff_tasks`` identity token appears in one, casefolded
  (C-R11), so a shared turn cannot learn the item from search. Keywords are never rendered and may keep those words.
- No snippet names a target item's price or a trap item (the steam wand tip, the BES878 / Barista Pro lookalike, the
  Ridge Twin, the wings). Every snippet is at most ``toolcall.loop.MAX_SNIPPET`` characters, and all the rows together
  render within ``toolcall.loop.MAX_RESPONSE`` beside any of the handoff requests, so the renderer cuts nothing.
- ``rank(query)``: words are casefolded alphanumeric runs, with a trailing "s" dropped from words longer than three
  characters. Query words shorter than two characters and the query-side stop words are ignored; each query word
  counts once. A word scores 2 for a title hit, else 1 for a snippet or keyword hit. Rows with a positive score come
  out best first, ties in row order, at most five.

The rows are invented for the demo copies, as on the pages.
"""

from __future__ import annotations

import hashlib
import json
import re

from local_astra.contracts.toolcall import SearchResult, SearchResults

__all__ = ["SEARCH_ROWS", "HandoffSearch", "handoff_search_index_sha", "rank"]

#: ``(title, url, snippet, keywords)``
SearchRow = tuple[str, str, str, tuple[str, ...]]

SEARCH_ROWS: tuple[SearchRow, ...] = (
    (
        "Breville Parts and Accessories - Espresso Machine Spare Parts",
        "https://www.breville.com/us/en/parts-accessories/espresso.html",
        "Genuine replacement parts for Breville espresso machines: steam wand assemblies, portafilters and gaskets. "
        "Find parts by your machine's model number.",
        (
            "coffee",
            "machine",
            "part",
            "replacement",
            "broke",
            "broken",
            "repair",
            "spare",
            "steam",
            "wand",
        ),
    ),
    (
        "Camping Stoves - Pinecone Outdoor Co-op",
        "https://www.pinecone-outdoor.com/camping/stoves",
        "Backpacking and two-burner camping stoves from the co-op. Compare stoves by burners, weight and fuel type.",
        ("stove", "camping", "burner", "trailhead", "duo", "hiking", "gear", "cheap", "under"),
    ),
    (
        "Domino's Pizza - Order Pizza Online for Delivery or Carryout",
        "https://www.dominos.com/en/pages/order",
        "Order pizza online from a Domino's store near you for delivery or carryout. Choose your store, then pick a "
        "size and toppings.",
        ("order", "pizza", "pepperoni", "large", "pickup", "store", "food", "dinner"),
    ),
)

_MAX_RESULTS = 5
_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "to", "and", "for", "of", "in", "on", "at", "me", "my", "find", "show", "add", "search",
        "page", "please", "this", "that", "it", "put", "what", "is", "are", "be", "you", "your", "with", "from",
        "online", "buy", "website", "site", "www", "com", "http", "https",
    }
)  # fmt: skip
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.casefold()):
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _terms(query: str) -> list[str]:
    terms: list[str] = []
    for w in _words(query):
        if len(w) >= 2 and w not in _STOP_WORDS and w not in terms:
            terms.append(w)
    return terms


_INDEX: tuple[tuple[frozenset[str], frozenset[str]], ...] = tuple(
    (frozenset(_words(title)), frozenset(_words(snippet + " " + " ".join(keywords))))
    for title, _url, snippet, keywords in SEARCH_ROWS
)


def rank(query: str) -> list[SearchRow]:
    """The rows matching ``query``, best first; pure and deterministic; never raises for a ``str``."""
    terms = _terms(query) if isinstance(query, str) else []
    scored: list[tuple[int, int]] = []
    for i, (title_words, other_words) in enumerate(_INDEX):
        score = sum(2 if t in title_words else 1 if t in other_words else 0 for t in terms)
        if score > 0:
            scored.append((-score, i))
    scored.sort()
    return [SEARCH_ROWS[i] for _s, i in scored[:_MAX_RESULTS]]


def handoff_search_index_sha() -> str:
    """sha256 of the canonical JSON of ``SEARCH_ROWS`` (keywords as lists)."""
    rows = [[t, u, s, list(k)] for t, u, s, k in SEARCH_ROWS]
    blob = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class HandoffSearch:
    """A ``contracts.protocols.SearchBackend`` over ``SEARCH_ROWS``; opens nothing."""

    async def search(self, query: str) -> SearchResults:
        rows = rank(query)
        return SearchResults(
            query=query,
            results=tuple(SearchResult(title=t, url=u, snippet=s) for t, u, s, _k in rows),
        )
