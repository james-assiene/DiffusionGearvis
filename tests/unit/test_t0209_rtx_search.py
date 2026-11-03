"""T-0209 group A1, criteria A1.7 and A1.16 (``rtx_search``): the pure loopback search index that answers
``google_search`` in the R1/R2 rigs (docs/tasks/T-0209.md, "Tests to write", ``tests/unit/test_t0209_rtx_search.py``).
A2.3's R2 rankings are group A2's and are added here when A2 starts.

**The surface** (``local_astra/pages/rtx_search.py``, new in A1, reached through :func:`rs`, a named
``AssertionError`` while absent): ``SEARCH_ROWS`` (a tuple of ``(title, url, snippet, keywords)``),
``search_index_sha()``, ``rank(query)`` and ``LoopbackSearch`` (``search(query) -> SearchResults``).

**Oracles.** The three A1.7 (a) queries, (b) and (c) are typed from the task, with ``R1_WRONG_FIX`` and ``R1_SETTING``
typed from A1.9 (never imported). The URL normaliser is A1.2's rule written out here (trim, drop an ``http://`` or
``https://`` prefix, casefold, drop a leading ``www.``, drop query and fragment, drop a trailing ``/``). Each result
row is identified by the page its URL routes to (``ROUTES`` -> path -> A1.4's page id, typed here). The ranking
differential (``test_rank_matches_the_rule``) is a brute-force oracle of A1.7's scoring rule written from the task and
``pages/static/search.html``'s word rule, over the real rows and a fixed query list.

**Readings this module pins where the task is silent** (reported):

- ``rank(query)`` returns the ranked rows best first, each either a ``SEARCH_ROWS`` element or its index;
  ``LoopbackSearch().search(query)`` returns them as ``SearchResult(title, url, snippet)`` in that order.
- ``keywords`` is a ``str`` or a sequence of ``str``; its words count as snippet words (score 1).
- The stem is ``search.html``'s: a trailing "s" is dropped from words longer than three characters; query words
  shorter than two characters are ignored; stop words are checked after stemming; each query word counts once.
- A1 rows are one per page id: ``forum-thread``, ``issue-blackwell``, ``manual-gpu``, ``notes-4-4``, ``forum-oom``,
  ``download``; each URL starts with ``https://``.
- ``search_index_sha()`` is a 64-hex sha256 that changes when a row changes (its formula is not pinned).

BASE = 76b98997. Activated at A1's ``/implement`` (no ``pending`` marker), against A1's implementation c529fc2f.

Regime: CPU, pure, n = 1; Hypothesis at ``max_examples=200``, ``deadline=None``, no database.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.rtx_search"
TASKS = "local_astra.pages.rtx_tasks"
#: A1.7 (a), typed
QUERIES_A = (
    "CUDA binary kernel for this graphics card compute capability (12.0) not found",
    "blender cycles no compatible GPUs found compute capability 12.0",
    "blender 4.2 rtx 5090 not showing cycles render devices",
)
#: A1.9, typed (applied to casefolded text)
R1_WRONG_FIX = re.compile(
    r"\b(?:reinstall\w*|clean\s+install\w*|ddu|display\s+driver\s+uninstaller|cuda\s+toolkit|roll(?:ing)?\s*back"
    r"|downgrad\w*|tile\s+size|use\s+tiling)\b"
)
R1_SETTING = re.compile(r"cycles\s+render\s+devices|preferences\W+(?:\w+\W+){0,3}system")
#: A1.2 paths -> A1.4 page ids, typed
PAGE_OF_PATH = {
    "/blender-manual/": "manual-home",
    "/blender-manual/gpu-rendering.html": "manual-gpu",
    "/blender-projects/": "projects-home",
    "/blender-projects/issue-134001.html": "issue-blackwell",
    "/blender-notes/": "notes-home",
    "/blender-notes/4.4-cycles.html": "notes-4-4",
    "/blender-notes/download.html": "download",
    "/blender-artists/": "forum-home",
    "/blender-artists/thread-rtx-5090.html": "forum-thread",
    "/blender-artists/thread-oom.html": "forum-oom",
}
A1_ROW_PAGES = (
    "forum-thread",
    "issue-blackwell",
    "manual-gpu",
    "notes-4-4",
    "forum-oom",
    "download",
)
#: ``pages/static/search.html``'s stop words, plus A1.7's three
STOP_WORDS = frozenset(
    {
        "the", "a", "an", "to", "and", "for", "of", "in", "on", "at", "me", "my", "find", "show", "add", "search",
        "page", "please", "blender", "model", "huggingface",
    }
)  # fmt: skip
NETWORK = frozenset(
    {"socket", "ssl", "httpx", "ddgs", "requests", "aiohttp", "urllib.request", "http.client"}
)


# =====================================================================================================
# helpers
# =====================================================================================================


def _import(name: str, why: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.pages"):
            raise AssertionError(f"T-0209 {why}: {name} does not exist yet") from exc
        raise


def rs() -> Any:
    return _import(MODULE, "A1.7")


def api(name: str) -> Any:
    m = rs()
    assert hasattr(m, name), f"T-0209 A1.7: {MODULE} has no {name!r}"
    return getattr(m, name)


def routes() -> dict[str, str]:
    m = _import(TASKS, "A1.2")
    assert hasattr(m, "ROUTES"), f"T-0209 A1.2: {TASKS} has no ROUTES"
    return dict(m.ROUTES)


def normalise(text: str) -> str:
    """A1.2's normalisation, written from the task."""
    t = text.strip()
    t = re.sub(r"^(?i:https?://)", "", t)
    t = t.casefold()
    if t.startswith("www."):
        t = t[4:]
    t = re.split(r"[?#]", t, maxsplit=1)[0]
    if t.endswith("/"):
        t = t[:-1]
    return t


def rows() -> list[tuple[str, str, str, Any]]:
    got = api("SEARCH_ROWS")
    assert isinstance(got, tuple) and got, "T-0209 A1.7: SEARCH_ROWS is a non-empty tuple"
    out = []
    for r in got:
        assert len(r) == 4, f"T-0209 A1.7: a row is (title, url, snippet, keywords): {r!r}"
        title, url, snippet, keywords = r
        assert all(isinstance(x, str) for x in (title, url, snippet)), r
        out.append((title, url, snippet, keywords))
    return out


def page_of(url: str) -> str | None:
    path = routes().get(normalise(url))
    return PAGE_OF_PATH.get(path) if path is not None else None


def run_search(backend: Any, query: str) -> Any:
    got = backend.search(query)
    if inspect.isawaitable(got):
        got = asyncio.run(got)
    return got


def search(query: str) -> list[tuple[str, str, str]]:
    from local_astra.contracts.toolcall import SearchResults

    got = run_search(api("LoopbackSearch")(), query)
    assert isinstance(got, SearchResults), f"T-0209 A1.7: search returns {type(got)!r}"
    return [(r.title, r.url, r.snippet) for r in got.results]


def ranked(query: str) -> list[tuple[str, str, str, Any]]:
    table = rows()
    out = []
    for item in api("rank")(query):
        if isinstance(item, int) and not isinstance(item, bool):
            out.append(table[item])
        else:
            out.append(tuple(item))
    return out


def pages_of(query: str) -> list[str | None]:
    return [page_of(u) for _t, u, _s in search(query)]


def kw_text(keywords: Any) -> str:
    return keywords if isinstance(keywords, str) else " ".join(str(k) for k in keywords)


# =====================================================================================================
# the protocol, determinism, bounds
# =====================================================================================================


def test_is_a_search_backend() -> None:
    """A1.7: ``LoopbackSearch()`` satisfies ``contracts.protocols.SearchBackend`` (``isinstance``)."""
    from local_astra.contracts.protocols import SearchBackend

    assert isinstance(api("LoopbackSearch")(), SearchBackend)


def test_deterministic() -> None:
    """A1.7: two instances and two calls give the same results; ``search`` returns ``rank``'s rows in order with the
    query echoed; ``search_index_sha()`` is a stable 64-hex string."""
    from local_astra.contracts.toolcall import SearchResults

    for q in (*QUERIES_A, "optix", "tile size"):
        a, b = search(q), search(q)
        assert a == b, q
        assert [(t, u, s) for t, u, s, _k in ranked(q)] == a, q
        got = run_search(api("LoopbackSearch")(), q)
        assert isinstance(got, SearchResults) and got.query == q, got
    sha = api("search_index_sha")()
    assert isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), sha
    assert api("search_index_sha")() == sha


@pytest.mark.parametrize("query", QUERIES_A)
def test_r1_queries_rank_the_forum_first(query: str) -> None:
    """A1.7 (a) (reviewer mutation 6: the forum row moved below the issue): for each query ``forum-thread`` is first
    and ``issue-blackwell`` and ``manual-gpu`` are in the (at most five) results."""
    got = pages_of(query)
    assert 1 <= len(got) <= 5, (query, got)
    assert got[0] == "forum-thread", f"T-0209 A1.7 (a): {query!r} ranks {got}"
    assert "issue-blackwell" in got and "manual-gpu" in got, (query, got)


def test_forum_snippet_is_the_wrong_fix() -> None:
    """A1.7 (b): the forum-thread row's snippet matches ``R1_WRONG_FIX`` (casefolded)."""
    hits = [r for r in rows() if page_of(r[1]) == "forum-thread"]
    assert len(hits) == 1, hits
    assert R1_WRONG_FIX.search(hits[0][2].casefold()), hits[0][2]


def test_no_snippet_names_the_setting() -> None:
    """A1.7 (c) (reviewer mutation 7: a snippet containing "Cycles Render Devices"): no snippet matches
    ``R1_SETTING`` (casefolded), so a snippet-only answer can never be a full success."""
    bad = [r[2] for r in rows() if R1_SETTING.search(r[2].casefold())]
    assert bad == [], bad


def test_rows_route_and_fit() -> None:
    """A1.7: every row URL is ``https://`` and normalises to a ``ROUTES`` key; every snippet is at most
    ``toolcall.loop.MAX_SNIPPET``; the A1 rows are one per page id ``forum-thread``, ``issue-blackwell``,
    ``manual-gpu``, ``notes-4-4``, ``forum-oom``, ``download``."""
    from local_astra.toolcall.loop import MAX_SNIPPET

    keys = routes()
    for title, url, snippet, _k in rows():
        assert url.startswith("https://"), url
        assert normalise(url) in keys, f"T-0209 A1.7: {url} normalises to no ROUTES key"
        assert len(snippet) <= MAX_SNIPPET, (url, len(snippet))
        assert title.strip(), url
    blender = sorted(p for p in (page_of(r[1]) for r in rows()) if p is not None)
    assert blender == sorted(A1_ROW_PAGES), blender


#: A1.11 step 1 / A1.8: the red status bar's text with its full stop (QUERIES_A[0] is the same text without it)
R1_ERROR_QUERY = "CUDA binary kernel for this graphics card compute capability (12.0) not found."
_URL_IN_BLOCK = re.compile(r'url:<\|"\|>(.*?)<\|"\|>')


def _rendered_urls(query: str) -> tuple[list[str], list[str]]:
    """(the backend's result URLs, the URLs as the session renders them into the ``google_search`` response):
    ``loop.search_payload`` then ``loop.render_response``, the path ``conductor/session.py`` takes."""
    from local_astra.contracts.toolcall import SearchResults
    from local_astra.toolcall import loop

    got = run_search(api("LoopbackSearch")(), query)
    assert isinstance(got, SearchResults), type(got)
    block = loop.render_response("google_search", loop.search_payload(got))
    return [r.url for r in got.results], _URL_IN_BLOCK.findall(block)


@pytest.mark.parametrize("query", [*QUERIES_A, R1_ERROR_QUERY])
def test_rendered_response_keeps_urls_typeable(query: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1.7 + A1.11 (payload bound): ``toolcall.loop.render_response`` caps a response at ``MAX_RESPONSE`` chars by
    shortening every string to a common length, which (found by the implementer) had cut result URLs so the model
    could not type them. For each A1.7 (a) query and the R1 error query the rendered ``google_search`` response
    carries every result URL whole: the same strings, in order, as the backend returned, each a ``SEARCH_ROWS`` URL
    (https, no whitespace, no ``…``) that normalises to a ``ROUTES`` key, i.e. typeable into ``web.html``.

    Positive control: with ``MAX_RESPONSE`` patched down to 700 the same predicate sees a cut (or dropped) URL, so
    this row can fail."""
    from local_astra.toolcall import loop

    row_urls = {r[1] for r in rows()}
    keys = routes()
    want, got = _rendered_urls(query)
    assert want, f"T-0209 A1.7: {query!r} returned no results"
    assert got == want, (
        f"T-0209 A1.7: rendered URLs differ from the results (cut by MAX_RESPONSE?): {got} vs {want}"
    )
    for url in got:
        assert url in row_urls, f"T-0209 A1.7: {url!r} is not a SEARCH_ROWS URL"
        assert "…" not in url and not re.search(r"\s", url) and url.startswith("https://"), url
        assert normalise(url) in keys, f"T-0209 A1.7: rendered {url!r} normalises to no ROUTES key"
    monkeypatch.setattr(loop, "MAX_RESPONSE", 700)
    want_small, got_small = _rendered_urls(query)
    assert got_small != want_small, (
        "power: a 700-char bound did not cut any URL; the predicate cannot see a cut"
    )


@pytest.mark.parametrize(
    "query",
    ["", "   ", "zzzzqx qqqqvw", "a b c x y", "\n\t", "kubernetes helm chart"],
)
def test_unmatched_returns_nothing(query: str) -> None:
    """A1.7: an empty query, or one matching no row, returns zero results."""
    assert search(query) == [], query


def test_stop_words_only_return_nothing() -> None:
    """A1.7 (adversarial): a query made only of stop words (``search.html``'s, plus "blender", "model" and
    "huggingface", in any case and with a plural "s") returns nothing."""
    q = "The Blender BLENDERS models HuggingFace pages, please find me a search for my page on Blender"
    assert search(q) == [], search(q)


def test_stuffed_query_returns_at_most_five() -> None:
    """A1.7 (adversarial): a query of every title, snippet and keyword word returns at most five rows, best first."""
    words = " ".join(f"{t} {s} {kw_text(k)}" for t, _u, s, k in rows())
    got = search(words)
    assert 1 <= len(got) <= 5, len(got)


# =====================================================================================================
# the ranking rule (brute-force oracle)
# =====================================================================================================


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.casefold()):
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def oracle(query: str) -> list[str]:
    """A1.7's rule over the real rows: unique query words (>= 2 chars, not stop words); 2 for a title hit, else 1 for
    a snippet or keyword hit; positive scores best first, ties in row order; at most 5. Returns URLs."""
    terms = []
    for w in _words(query):
        if len(w) >= 2 and w not in STOP_WORDS and w not in terms:
            terms.append(w)
    scored = []
    for i, (title, url, snippet, keywords) in enumerate(rows()):
        tw = set(_words(title))
        sw = set(_words(snippet)) | set(_words(kw_text(keywords)))
        score = sum(2 if t in tw else 1 if t in sw else 0 for t in terms)
        if score > 0:
            scored.append((-score, i, url))
    return [u for _s, _i, u in sorted(scored)[:5]]


def test_rank_matches_the_rule() -> None:
    """A1.7 ``rank`` (differential against the brute-force oracle): for the A1.7 queries, every single word of every
    title, and a few mixed queries, ``search`` returns the oracle's URLs in the oracle's order."""
    qs = [
        *QUERIES_A,
        "optix kernels",
        "out of memory tile",
        "download 4.4.1",
        "Devices DEVICE devices",
    ]
    for title, _u, _s, _k in rows():
        qs += title.split()
    for q in qs:
        assert [u for _t, u, _s in search(q)] == oracle(q), q


# =====================================================================================================
# never raises, no network
# =====================================================================================================


@settings(max_examples=200, deadline=None, database=None)
@given(query=st.text(max_size=200) | st.sampled_from(QUERIES_A))
def test_never_raises(query: str) -> None:
    """A1.7 (Hypothesis): ``search`` never raises for any ``str`` and returns at most five results."""
    assert len(search(query)) <= 5


def test_no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """A1.7/A1.14 (a sentinel): with every socket entry point recording, building a ``LoopbackSearch`` and running
    searches touches none. Power: the sentinel records a direct ``socket.create_connection``."""
    calls: list[str] = []

    def rec(name: str) -> Any:
        def f(*a: Any, **k: Any) -> Any:
            calls.append(name)
            raise OSError(f"T-0209 sentinel: {name}")

        return f

    cls = api("LoopbackSearch")
    monkeypatch.setattr(socket.socket, "connect", rec("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", rec("connect_ex"))
    monkeypatch.setattr(socket.socket, "bind", rec("bind"))
    monkeypatch.setattr(socket.socket, "sendto", rec("sendto"))
    monkeypatch.setattr(socket, "create_connection", rec("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", rec("getaddrinfo"))
    backend = cls()
    for q in (*QUERIES_A, "", "download"):
        got = backend.search(q)
        if inspect.isawaitable(got):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(got)
            finally:
                loop.close()
    assert calls == [], f"T-0209 A1.7: LoopbackSearch touched the network: {calls}"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9))
    assert calls == ["create_connection"], "power: the sentinel records"


def test_imports_no_network_module() -> None:
    """A1.7/A1.16: ``rtx_search``'s own imports name only the stdlib, ``local_astra.contracts.toolcall`` and
    ``local_astra.pages.rtx_tasks``, and none of ``socket``/``ssl``/``httpx``/``ddgs``/``requests``/``aiohttp``/
    ``urllib.request``/``http.client``; in a fresh interpreter, importing it loads nothing under ``local_astra.search``,
    ``computer``, ``conductor`` or ``toolcall``, and no ``ddgs``/``httpx``/``ssl`` beyond what its two allowed
    imports already load."""
    path = REPO / "local_astra" / "pages" / "rtx_search.py"
    assert path.is_file(), f"T-0209 A1.7: {path.relative_to(REPO)} does not exist yet"
    bad: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [f"{mod}.{a.name}" if mod == "local_astra.pages" else mod for a in node.names]
        else:
            continue
        for n in names:
            if n in NETWORK or n.split(".")[0] in NETWORK:
                bad.append(n)
            elif n == "__future__" or n in ("local_astra.contracts.toolcall", TASKS):
                continue
            elif n.split(".")[0] == "local_astra" or n.split(".")[0] not in sys.stdlib_module_names:
                bad.append(n)
    assert bad == [], f"T-0209 A1.16: rtx_search imports {bad}"
    probe = (
        "import sys, importlib\n"
        "for m in sys.argv[1:-1]: importlib.import_module(m)\n"
        "before = set(sys.modules)\n"
        "importlib.import_module(sys.argv[-1])\n"
        "print('\\n'.join(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, "local_astra.contracts.toolcall", TASKS, MODULE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    extra = proc.stdout.split()
    banned = [
        n
        for n in extra
        if n.split(".")[0] in {"ddgs", "httpx", "ssl", "requests", "aiohttp", "playwright"}
        or n.startswith(
            (
                "local_astra.search",
                "local_astra.computer",
                "local_astra.conductor",
                "local_astra.toolcall",
            )
        )
    ]
    assert banned == [], f"T-0209 A1.7: importing rtx_search loaded {banned}"
