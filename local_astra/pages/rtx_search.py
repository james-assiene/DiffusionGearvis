"""The loopback search index for the RTX scenarios (T-0209 A1.7): a pure, static ``SearchBackend``.

``LoopbackSearch`` answers ``google_search`` in the R1/R2 rigs and the B harness with rows from ``SEARCH_ROWS``. It is
not a network exception: it opens nothing, reads no file and imports no network module. Only tests and the T-0209
harness construct it; ``cli`` never does.

- ``SEARCH_ROWS`` is a tuple of ``(title, url, snippet, keywords)``. Every URL normalises (A1.2) to a
  ``rtx_tasks.ROUTES`` key, so the model can open a result by typing its address into the start-page stand-in. Every
  snippet is at most ``toolcall.loop.MAX_SNIPPET`` (300) characters, and the rows are short enough that any five of
  them render within ``toolcall.loop.MAX_RESPONSE`` (1500) beside a query of up to about 160 characters (measured
  2026-10-30: 1334 characters plus the query for the largest five), so the renderer cuts nothing (a cut URL could not be
  typed into the address box).
- ``rank(query)``: words are casefolded alphanumeric runs, with a trailing "s" dropped from words longer than three
  characters (``pages/static/search.html``'s stem). Query words shorter than two characters, and the query-side stop
  words (``search.html``'s, plus "blender", "model" and "huggingface"), are ignored; each query word counts once. A
  word scores 2 for a title hit, else 1 for a snippet or keyword hit. Rows with a positive score come out best first,
  ties in row order, at most five.
- No R1 snippet names the setting path, and the forum thread's snippet carries its (wrong) top answer, so an answer
  built from snippets alone can be the trap but never a full success.

The facts in the rows are invented for the demo copies, as on the pages.
"""

from __future__ import annotations

import hashlib
import json
import re

from local_astra.contracts.toolcall import SearchResult, SearchResults

__all__ = ["SEARCH_ROWS", "LoopbackSearch", "rank", "search_index_sha"]

#: ``(title, url, snippet, keywords)``
SearchRow = tuple[str, str, str, tuple[str, ...]]

SEARCH_ROWS: tuple[SearchRow, ...] = (
    (
        "RTX 5090 not showing in Cycles Render Devices - Blender Artists",
        "https://blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001",
        'On 4.2.3: "CUDA binary kernel for this graphics card compute capability (12.0) not found". Top answer: '
        "reinstall the driver with DDU, add CUDA Toolkit 12.8.",
        ("forum", "question", "driver", "no", "compatible", "gpus"),
    ),
    (
        "RTX 5090 not listed in Cycles Render Devices on 4.2.3",
        "https://projects.blender.org/blender/blender/issues/134001",
        "Resolved. 4.2 builds have no kernel for compute capability 12.0; fixed in 4.4.0.",
        ("bug", "report", "tracker"),
    ),
    (
        "GPU Rendering - Blender Manual",
        "https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html",
        "Supported hardware, device setup, kernel and compute capability errors.",
        ("cycles", "render", "cuda", "optix", "nvidia", "documentation"),
    ),
    (
        "Blender 4.4: Cycles - Release Notes",
        "https://developer.blender.org/docs/release_notes/4.4/cycles",
        "Added support for NVIDIA Blackwell GPUs (RTX 50 series).",
        ("changelog", "release", "notes", "kernels"),
    ),
    (
        "Cycles CUDA error: out of memory - Blender Artists",
        "https://blenderartists.org/t/cycles-cuda-error-out-of-memory/1570002",
        "Answer: enable Use Tiling and lower Tile Size to 512.",
        ("forum", "vram", "crash"),
    ),
    (
        "Download Blender 4.4.1",
        "https://www.blender.org/download/",
        "Windows, macOS and Linux. Free and open source.",
        ("installer", "release"),
    ),
)

_MAX_RESULTS = 5
_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "to", "and", "for", "of", "in", "on", "at", "me", "my", "find", "show", "add", "search",
        "page", "please", "blender", "model", "huggingface",
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
    """The rows matching ``query``, best first (A1.7); pure and deterministic; never raises for a ``str``."""
    terms = _terms(query) if isinstance(query, str) else []
    scored: list[tuple[int, int]] = []
    for i, (title_words, other_words) in enumerate(_INDEX):
        score = sum(2 if t in title_words else 1 if t in other_words else 0 for t in terms)
        if score > 0:
            scored.append((-score, i))
    scored.sort()
    return [SEARCH_ROWS[i] for _s, i in scored[:_MAX_RESULTS]]


def search_index_sha() -> str:
    """sha256 of the canonical JSON of ``SEARCH_ROWS`` (keywords as lists)."""
    rows = [[t, u, s, list(k)] for t, u, s, k in SEARCH_ROWS]
    blob = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LoopbackSearch:
    """A ``contracts.protocols.SearchBackend`` over ``SEARCH_ROWS``; opens nothing."""

    async def search(self, query: str) -> SearchResults:
        rows = rank(query)
        return SearchResults(
            query=query,
            results=tuple(SearchResult(title=t, url=u, snippet=s) for t, u, s, _k in rows),
        )
