"""T-0205 group D: the real-site rehearsal harness's pure half, its refusals and its wrapper (unit; CPU; no engine,
no browser, no network, no socket). docs/tasks/T-0205.md D1r-D6r and "Tests to write" (D).

BASE 14733dcb (main, ``T-0207 B merged; status``). Every row that needs ``scripts/gpu/t0205_real_sites.py`` or
``scripts/locked/t0205_real_sites.sh`` is red on a named assertion in ``harness()``/``api()``/``wrapper_text()``
("does not exist yet" / "has no ..."), never on an import error: every harness symbol is looked up inside the row. The
D6r byte-identity rows are green at BASE (TIP == BASE until the merge). Expected values are computed by hand (comments
say how), never read off an implementation (none exists).

**Network.** No row opens a socket. Every row that calls ``main`` runs under ``Rig``: every real ``connect``,
``connect_ex``, ``bind``, ``create_connection`` and ``getaddrinfo`` is refused and recorded, and each row asserts the
record is empty. ``ASTRA_REAL_SITES=1`` is set only inside ``Rig``, where the engine check, the executor, the sandbox
check and the turn runner are all stubs. Nothing here reads ``~/.config/diffusion_gearvis/``.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not).

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``ENV_VAR = "ASTRA_REAL_SITES"``; ``REAL_SCENARIOS``: a tuple of
    ``(id, request, start)`` (D1r); ``REFUSED_IDS = frozenset({"D7"})``; ``DEFAULT_REPEATS = 1``; ``MAX_REPEATS = 2``;
    ``TURN_TIMEOUT_S = 300``; ``RUN_BUDGET_S = 3600`` (read at call time); ``LOGS_DIR``: the repo's ``logs/``
    directory (module level, read at call time, so a row points it at a tmp dir).

Pure functions
    ``final_kind(url) -> "challenge" | "checkout" | "other"``; ``final_host(url) -> str | None``;
    ``hosts_of(outcomes) -> {"hosts": sorted list, "hosts_more": int}``; ``refusal_counts(egress) -> {scheme: int}``;
    ``reached(expected_host, hosts) -> bool | None``;
    ``search_query_count(session_records) -> int``; ``turn_record(scenario_id, repeat, raw) -> dict``;
    ``shot_path(out_dir, scenario_id, repeat, step) -> Path``; ``scenario_label(scenario_id, done, repeats) -> str``;
    ``executor_refusal(executor) -> str | None``; ``session_keywords() -> dict`` (the ``Session`` settings a turn
    passes, read from ``cli`` at call time).

Seams ``main`` reads at call time (module level)
    ``_git(*args) -> str`` (``status --porcelain`` and ``rev-parse HEAD``); ``_sandbox_check() -> str | None`` (``None``
    when Chromium's sandbox works, else why not); ``_make_executor(*a, **k)`` (the executor a turn would get, as
    ``cli._start_live`` resolves it); ``engine_healthy(base_url) -> bool`` (sync; the first socket after every
    refusal); ``run_turn(scenario_id=, request=, start=, repeat=, **k)`` (async; returns the raw turn below;
    ``start`` is the live start page, a URL or path whose path is ``/web.html``, never the expected site).
    ``main(argv)`` takes ``--base-url``, ``--out-dir`` (default ``LOGS_DIR / f"t0205-real-{sha7}"``),
    ``--real-sites``, ``--repeats`` and ``--scenarios`` (comma-separated ids; default: every runnable id).

The raw turn (``run_turn``'s return, the input of ``turn_record``)
    ``{"outcomes": [{"step", "tool", "status", "reason", "resolved"}], "egress": [{"host", "scheme"}],
    "session_records": [session-log records], "final_url": str | None, "frames": [{"step", "jpeg": bytes}],
    "wall_s": float, "tool_loop_end": str | None, "error": str | None}``.

``turn_record``'s keys pinned here
    ``id``, ``repeat``, ``hosts``, ``hosts_more``, ``refusals`` (counts per scheme), ``refusals_total``,
    ``search_queries``, ``steps``, ``wall_s``, ``tool_loop_end``, ``moved_refusals``, ``waits``, ``hand_back``,
    ``last_action`` (``{"step", "tool"}`` of the last executed action, or ``None``), ``final_kind``, ``final_host``,
    ``expected_host``, ``reached``. Never a URL.

``report.json`` keys pinned here
    ``report["scenarios"][id]`` per table id, with ``label`` (``ran`` / ``incomplete`` / ``refused``) and, for a
    scenario that ran, ``final_kind``, ``hosts``, ``steps``, ``shots`` (the screenshot paths), ``expected_host`` and
    ``reached`` (n = 1: the turn's value).

**Readings** (reported to the orchestrator; not resolved silently)
    1. D2r's flag/env pair: ``--real-sites`` and ``ASTRA_REAL_SITES`` exactly ``"1"`` (``"0"``, ``"true"`` and
       ``"yes"`` refuse). The harness never sets the variable itself (the wrapper does, D5r).
    2. D7 is refused by id: ``--scenarios`` naming D7 anywhere exits 2 before any socket; the default run never asks
       ``run_turn`` for D7, and the report labels D7 ``refused``. Whether ``REAL_SCENARIOS`` holds a D7 row is open.
    3. ``start`` (D1r vs H-T205-5), RULED by the coordinator 2026-10-30: H-T205-5 wins. Every turn starts at the live
       start page ``/web.html`` and the model navigates or searches itself. The table's third column keeps D1r's
       sites (an https URL on the named site; D2 has none) as the scenario's *expected site*: ``run_turn`` is called
       with the start page and never with the site URL (no argument names the expected host); the record and the
       report carry ``expected_host`` (the column's host; ``None`` for D2).
    4. "Relative dates": no request names a month, a year or a day-of-month; D1 is the task's exact sentence; D9 says
       "next Friday" and "back Sunday". The other requests' wording is not pinned beyond their subject words.
    5. "Counts per refusal class", RULED by the coordinator 2026-10-30: the class is the refusal's ``scheme`` (the
       proxy reports ``{host, scheme}`` only). The record keeps ``refusals_total`` and ``refusals`` = counts per
       scheme, never per host; the counts sum to the total.
    6. "Nothing written outside ``logs/``": an ``--out-dir`` that does not resolve under ``LOGS_DIR`` exits 2; a full
       stubbed run creates files only under ``LOGS_DIR``.
    7. Steps = executed-or-refused outcomes recorded for the turn (``len(outcomes)``); moved refusals = outcomes with
       status ``refused`` and ``reason == MOVED_REASON`` exactly; waits = outcomes whose tool is ``wait`` and status
       ``executed``; ``hand_back`` = ``tool_loop_end == "handed_back"``; search queries = session-log ``note`` records
       whose ``what`` is ``"web search"``; hosts = outcomes' ``resolved["hosts"]`` only (a search note's result hosts
       were not reached).
    8. Screenshots: ``shots/<id>/<repeat>-<step>.jpg`` with ``repeat`` as passed to ``run_turn`` and ``step`` as in
       the raw frame; an id with a path separator or ``..`` raises ``ValueError``.
    9. Live keywords: ``session_keywords()`` is the harness's single source of the turn's ``Session`` settings; every
       setting the cli passes ``Session`` for ``session --tools live`` (web mode, search on) is in it with the cli's
       value.
    10. Independence (D1r, Groups): the harness imports neither group C's ``t0205_demo_run`` nor A/B's
        ``scenario_tasks``, and constructs no network path of its own (no search backend, no ``EgressProxy``, no
        injected ``proxy_resolve``/``proxy_connect``).
    11. "Reached" (D4r), RULED by the coordinator 2026-10-30: reached iff some reached host equals the expected
        site's host or is a subdomain of it (a ``.``-separated suffix); ``None`` (JSON ``null``) when there is no
        expected site (D2). ``notlouvre.fr`` and ``louvre.fr.evil.test`` do not reach ``louvre.fr``.
        I2, RULED 2026-10-30 (review round 1): ``turn_record``'s ``reached`` is computed over the reached hosts
        together with the final host (a turn that ends on www.louvre.fr with no outcome hosts reached louvre.fr).
    12. Review round 1 rows (I1/T4, I3, I4, I5, T1-T3) are at the end of the module, each with its ruling.

Regime: CPU, pure, hand-built records; no engine, no browser, no network; n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from local_astra.computer.executor import CHECKOUT_REASON, MOVED_REASON
from tests import _git_range as gr
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0205_real_sites.py"
WRAPPER = REPO / "scripts" / "locked" / "t0205_real_sites.sh"

#: D6r's BASE: main at 14733dcb ("T-0207 B merged; status"), the commit T-0205-d branched from.
BASE_D = "14733dcb6f9696c87b9332af6d794960454f6a7e"
#: D's implementation tip: 78b9d7db ("T-0205 D fix: ... (I1-I5)"), group D's last implementation commit. BASE_D is
#: its ancestor and BASE_D..TIP_D changes only D's two scripts and this test file (no local_astra/ path).
TIP_D = "78b9d7dbcb9a6fe490f9e8b9549f80575cc1914d"

#: D1r by hand: every scenario but D7 runs (D7 refused, H-T205-6)
RUNNABLE = ("D1", "D2", "D3", "D4", "D5", "D6", "D8", "D9")
GIT = t31.GIT
SHA7 = GIT[:7]
VALID_URL = t31.VALID_URL


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0205 D1r: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0205_real_sites_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0205 D: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


def wrapper_text() -> str:
    assert WRAPPER.is_file(), f"T-0205 D5r: {WRAPPER.relative_to(REPO)} does not exist yet"
    return WRAPPER.read_text(encoding="utf-8")


def _code_only(text: str) -> str:
    """Python source with comments and docstrings removed (a source-grep guard must not match prose)."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0] = ast.Pass()
    return ast.unparse(tree)


# =====================================================================================================
# D1r: import, constants, independence
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """D1r/D2r: importing the harness opens no socket (connect, bind or name lookup, loopback included); it has a
    ``main``."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"


def test_constants() -> None:
    """D2r/D4r: exits, the env variable, the D7 refusal, repeats 1 by default and 2 at most, 300 s per turn, a 60 min
    run budget."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED")) == (0, 1, 2)
    assert api("ENV_VAR") == "ASTRA_REAL_SITES"
    assert frozenset(api("REFUSED_IDS")) == frozenset({"D7"})
    assert (api("DEFAULT_REPEATS"), api("MAX_REPEATS")) == (1, 2)
    assert api("TURN_TIMEOUT_S") == 300
    assert api("RUN_BUDGET_S") == 60 * 60


def test_imports_neither_group_c_nor_a_b() -> None:
    """D1r ("written independently; it does not import group C") and Groups ("imports nothing from A or B"), reading
    10: no import of, and no load by path of, ``t0205_demo_run`` or ``scenario_tasks``; a fresh interpreter that loads
    the harness has neither in ``sys.modules``."""
    harness()
    code = _code_only(HARNESS.read_text(encoding="utf-8"))
    for name in ("t0205_demo_run", "scenario_tasks"):
        assert name not in code, f"D1r: the harness names {name} in code (imports group C or A/B)"
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_h', {str(HARNESS)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['_h'] = m; spec.loader.exec_module(m)\n"
        "print(sorted(n for n in sys.modules if 'scenario_tasks' in n or 't0205_demo_run' in n))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "[]", proc.stdout


def test_no_network_path_of_its_own() -> None:
    """D2r (reading 10; adversarial: a harness that builds its own search backend or proxy, or injects a resolver,
    is a new network path): the harness's code constructs no search backend, no ``EgressProxy`` and passes no
    ``proxy_resolve``/``proxy_connect``; it imports nothing from ``local_astra.search`` and not ``ddgs``."""
    harness()
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [
                a.name
                for a in node.names
                if a.name.split(".")[0] == "ddgs" or a.name.startswith("local_astra.search")
            ]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if (
                mod.split(".")[0] == "ddgs"
                or mod.startswith("local_astra.search")
                or (mod == "local_astra" and any(a.name == "search" for a in node.names))
            ):
                problems.append(f"from {mod} import ...")
        elif isinstance(node, ast.Call):
            problems += [
                f"{k.arg}=" for k in node.keywords if k.arg in ("proxy_resolve", "proxy_connect")
            ]
        elif isinstance(node, ast.Attribute) and node.attr in ("_search_backend", "EgressProxy"):
            problems.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in ("_search_backend", "EgressProxy"):
            problems.append(node.id)
    assert not problems, f"D2r: the harness opens a network path of its own: {problems}"


def test_harness_never_sets_the_env_variable() -> None:
    """D2r/D5r (reading 1; reviewer mutation 9's sibling: a harness that sets ``ASTRA_REAL_SITES`` itself makes the
    pair check vacuous): no assignment to ``os.environ["ASTRA_REAL_SITES"]``, no ``setdefault``/``putenv``/``update``
    naming it."""
    harness()
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        names = ("ASTRA_REAL_SITES", "ENV_VAR")
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and any(n in ast.unparse(node.slice) for n in names)
        ):
            bad.append(ast.unparse(node))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("setdefault", "putenv", "update")
            and any(n in ast.unparse(node) for n in names)
        ):
            bad.append(ast.unparse(node))
    assert not bad, f"D2r: the harness sets the consent variable itself: {bad}"


# =====================================================================================================
# D1r: REAL_SCENARIOS
# =====================================================================================================

#: D1r/H-T205-5 by hand: id -> (allowed hosts, path prefix)
STARTS = {
    "D1": ("louvre.fr", "/"),
    "D3": ("bbcgoodfood.com", "/"),
    "D4": ("breville.com", "/"),
    "D5": ("rei.com", "/"),
    "D6": ("dominos.com", "/"),
    "D8": ("nps.gov", "/pore"),
    "D9": ("google.com", "/travel/flights"),
}
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
]


def _rows() -> dict[str, tuple[str, Any]]:
    rows: dict[str, tuple[str, Any]] = {}
    for row in api("REAL_SCENARIOS"):
        sid, request, start = tuple(row)
        assert sid not in rows, f"{sid} listed twice"
        rows[str(sid)] = (str(request), start)
    return rows


def test_real_scenarios_ids() -> None:
    """D1r/H-T205-6: every runnable id D1-D6, D8, D9 has a row; no id outside D1-D9 (a D7 row may exist but never
    runs: reading 2)."""
    rows = _rows()
    assert set(RUNNABLE) <= set(rows), sorted(rows)
    assert set(rows) <= set(RUNNABLE) | {"D7"}, sorted(rows)


@pytest.mark.parametrize("sid", sorted(STARTS))
def test_real_scenario_starts(sid: str) -> None:
    """D1r starting sites (reading 3): an https URL on the named site (the host or a subdomain of it), its path under
    the named prefix (D8 nps.gov/pore, D9 google.com/travel/flights)."""
    _request, start = _rows()[sid]
    want_host, prefix = STARTS[sid]
    parts = urlsplit(str(start))
    host = (parts.hostname or "").lower()
    assert parts.scheme == "https", (sid, start)
    assert host == want_host or host.endswith("." + want_host), (sid, start)
    assert parts.path.startswith(prefix), (sid, start)


def test_d2_starts_at_the_start_page() -> None:
    """D1r: D2 runs with search on from the start page, so its start is no external https site."""
    _, start = _rows()["D2"]
    parts = urlsplit(str(start or ""))
    assert not (parts.scheme == "https" and parts.hostname), f"D2 starts on a site: {start!r}"


def test_requests_use_relative_dates() -> None:
    """D1r (reading 4; H-T205-3: D keeps James's relative wording): D1 exactly; D9 next Friday, back Sunday; no request
    names a month, a four-digit year or a day-of-month."""
    rows = _rows()
    assert rows["D1"][0] == "Is the Louvre open today, and until when?"
    d9 = rows["D9"][0].casefold()
    assert "next friday" in d9 and "back sunday" in d9, rows["D9"][0]
    for sid, (request, _start) in rows.items():
        words = re.findall(r"[a-z]+", request.casefold())
        assert not set(words) & set(MONTHS), (sid, request)
        assert not re.search(r"\b(?:19|20)\d\d\b", request), (sid, request)
        weekday_date = (
            r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\b"
        )
        assert not re.search(weekday_date, request, re.IGNORECASE), (sid, request)


SUBJECTS = {
    "D2": ("ramen", "seattle"),
    "D3": ("lasagna", "grocery"),
    "D4": ("breville", "steam wand"),
    "D5": ("camping stove", "cart"),
    "D6": ("pepperoni", "domino"),
    "D8": ("san francisco", "saturday"),
    "D9": ("sfo", "seattle"),
}


@pytest.mark.parametrize("sid", sorted(SUBJECTS))
def test_requests_keep_james_subject(sid: str) -> None:
    """D1r ("James's originals"): each request keeps the subject words of the loopback copy's request (A2)."""
    request = _rows()[sid][0].casefold()
    for word in SUBJECTS[sid]:
        assert word in request, (sid, word, request)


# =====================================================================================================
# D3r: the pure records
# =====================================================================================================

FINAL_KINDS = [
    (
        "https://www.google.com/sorry/index?continue=https://www.google.com/travel/flights",
        "challenge",
    ),
    ("https://www.amazon.com/errors/validateCaptcha?amzn=x", "challenge"),
    ("https://www.rei.com/checkout/x", "checkout"),
    ("https://www.rei.com/cart", "other"),
    ("https://www.louvre.fr/en/visit/hours", "other"),
]


@pytest.mark.parametrize(
    "url, want", FINAL_KINDS, ids=[w + ":" + urlsplit(u).path for u, w in FINAL_KINDS]
)
def test_final_kind(url: str, want: str) -> None:
    """D3r and Tests to write: ``/sorry/index`` and ``/errors/validateCaptcha`` are challenges, ``/checkout/x`` a
    checkout, ``/cart`` and an ordinary page other (the too-wide direction: a cart is not a checkout)."""
    with t31.no_sockets() as attempts:
        got = api("final_kind")(url)
    assert got == want, (url, got)
    assert attempts == []


def test_final_host_is_the_host_only() -> None:
    """D3r: only the host of the final URL is kept, lowercased."""
    assert api("final_host")("https://WWW.Rei.com/checkout/secret-path?q=1#f") == "www.rei.com"


def test_hosts_union_with_hosts_more() -> None:
    """D3r (reading 7): the sorted union of every outcome's ``resolved["hosts"]`` and the sum of their
    ``hosts_more``; an outcome without ``resolved`` adds nothing. By hand: {a, b, c}, 3 + 2 = 5."""
    outcomes = [
        {
            "step": 1,
            "tool": "click_at",
            "status": "executed",
            "resolved": {"hosts": ["b.example.com", "a.example.com"], "hosts_more": 3},
        },
        {
            "step": 2,
            "tool": "click_at",
            "status": "executed",
            "resolved": {"hosts": ["b.example.com", "c.example.com"]},
        },
        {
            "step": 3,
            "tool": "scroll",
            "status": "executed",
            "resolved": {"hosts": [], "hosts_more": 2},
        },
        {"step": 4, "tool": "press_key", "status": "refused", "reason": "x", "resolved": None},
    ]
    got = api("hosts_of")(outcomes)
    assert list(got["hosts"]) == ["a.example.com", "b.example.com", "c.example.com"]
    assert got["hosts_more"] == 5


def test_refusal_counts() -> None:
    """D3r (reading 5, RULED: the class is the scheme): proxy refusals as counts per scheme, never per host; the counts
    sum to the number of refusal records. By hand: 3 https + 1 http (two of the https on one host, still counted
    twice) -> {https: 3, http: 1}, total 4; none -> total 0."""
    egress = [
        {"host": "10.0.0.7", "scheme": "https"},
        {"host": "intranet.local", "scheme": "https"},
        {"host": "plain.example.org", "scheme": "http"},
        {"host": "10.0.0.7", "scheme": "https"},
    ]
    got = dict(api("refusal_counts")(egress))
    assert all(isinstance(v, int) and v >= 0 for v in got.values()), got
    assert sum(got.values()) == 4, got
    assert not {r["host"] for r in egress} & set(got), f"a refusal class is a host: {got}"
    assert {k: v for k, v in got.items() if v} == {"https": 3, "http": 1}, got
    assert sum(dict(api("refusal_counts")([])).values()) == 0


def _note(seq: int, what: str, **data: Any) -> dict[str, Any]:
    return {"seq": seq, "kind": "note", "data": {"what": what, **data}}


def test_search_query_count_only() -> None:
    """D3r (reading 7): the number of ``web search`` notes; a request, a step line and another note do not count.
    By hand: 2. The record carries the count, never the query text."""
    records = [
        {"seq": 1, "kind": "llm.request", "data": {"phase": "tool_step"}},
        _note(
            2, "web search", query="louvre hours QUERYTOKEN", hosts=["www.louvre.fr"], hosts_more=0
        ),
        _note(3, "completion check asked"),
        _note(4, "web search", query="louvre friday QUERYTOKEN", hosts=[], hosts_more=0),
        {"seq": 5, "kind": "tool.step", "data": {"what": "web search"}},
    ]
    assert api("search_query_count")(records) == 2


PATH_TOKEN = "zq7PATHTOKENx9"


def _raw(**over: Any) -> dict[str, Any]:
    raw = {
        "outcomes": [
            {
                "step": 1,
                "tool": "click_at",
                "status": "executed",
                "reason": None,
                "resolved": {"hosts": ["www.rei.com"]},
            },
            {
                "step": 2,
                "tool": "click_at",
                "status": "refused",
                "reason": MOVED_REASON,
                "resolved": {},
            },
            {
                "step": 3,
                "tool": "wait",
                "status": "executed",
                "reason": None,
                "resolved": {"hosts": ["cdn.rei.com"], "hosts_more": 1},
            },
            {
                "step": 4,
                "tool": "click_at",
                "status": "refused",
                "reason": CHECKOUT_REASON,
                "resolved": {},
            },
            {"step": 5, "tool": "type_at", "status": "executed", "reason": None, "resolved": {}},
            {
                "step": 6,
                "tool": "click_at",
                "status": "refused",
                "reason": MOVED_REASON + " ",
                "resolved": {},
            },
        ],
        "egress": [{"host": "10.1.2.3", "scheme": "https"}],
        "session_records": [
            _note(
                1,
                "web search",
                query=f"stove {PATH_TOKEN}",
                hosts=["evil-result.example.net"],
                hosts_more=4,
            ),
        ],
        "final_url": f"https://www.rei.com/checkout/{PATH_TOKEN}/pay?token={PATH_TOKEN}#{PATH_TOKEN}",
        "frames": [{"step": s, "jpeg": b"\xff\xd8JPEG" + bytes([s])} for s in (1, 2, 3)],
        "wall_s": 42.5,
        "tool_loop_end": "handed_back",
        "error": None,
    }
    return {**raw, **over}


def test_turn_record_by_hand() -> None:
    """D3r (reading 7; adversarial: a ``CHECKOUT_REASON`` refusal and a reason differing by one space are not moved
    refusals; a search note's result hosts were not reached). By hand from ``_raw``: hosts {www.rei.com,
    cdn.rei.com}, hosts_more 1; refusals total 1; search queries 1; steps 6; moved refusals 1; waits 1; hand-back
    yes; last executed action step 5 type_at; final kind checkout on www.rei.com."""
    rec = api("turn_record")("D5", 0, _raw())
    assert rec["id"] == "D5" and rec["repeat"] == 0
    assert list(rec["hosts"]) == ["cdn.rei.com", "www.rei.com"], rec["hosts"]
    assert "evil-result.example.net" not in json.dumps(rec)
    assert rec["hosts_more"] == 1
    assert {k: v for k, v in dict(rec["refusals"]).items() if v} == {"https": 1}, rec["refusals"]
    assert rec["refusals_total"] == 1
    assert "10.1.2.3" not in json.dumps(rec), (
        "D3r: a refused host is stored (counts per scheme only)"
    )
    assert rec["search_queries"] == 1
    assert rec["steps"] == 6
    assert rec["moved_refusals"] == 1, (
        "exact MOVED_REASON only; CHECKOUT_REASON and a near-miss are not moved"
    )
    assert rec["waits"] == 1
    assert rec["hand_back"] is True
    assert dict(rec["last_action"]) == {"step": 5, "tool": "type_at"}
    assert rec["wall_s"] == pytest.approx(42.5)
    assert rec["tool_loop_end"] == "handed_back"
    assert (rec["final_kind"], rec["final_host"]) == ("checkout", "www.rei.com")
    assert rec["expected_host"] == _expected_host("D5")
    assert rec["reached"] is True, "www.rei.com was reached; D5's expected site is rei.com's"


def test_turn_record_without_hand_back_or_action() -> None:
    """D3r: an answer end is no hand-back; a turn with no executed action has ``last_action`` None."""
    rec = api("turn_record")(
        "D1",
        1,
        _raw(
            outcomes=[],
            tool_loop_end="answer",
            final_url="https://www.louvre.fr/",
            session_records=[],
        ),
    )
    assert rec["hand_back"] is False
    assert rec["last_action"] is None
    assert (rec["steps"], rec["moved_refusals"], rec["waits"], rec["search_queries"]) == (
        0,
        0,
        0,
        0,
    )
    assert rec["final_kind"] == "other"
    assert rec["expected_host"] == _expected_host("D1")
    # I2 (RULED 2026-10-30, replacing the earlier reading): ``reached`` is over the reached hosts and the final host,
    # so a turn that ended on www.louvre.fr reached louvre.fr even with no outcome hosts
    assert rec["reached"] is True, "I2: the final host www.louvre.fr counts as reached"


def test_turn_record_d2_has_no_expected_site() -> None:
    """Reading 11 (RULED): D2 has no expected site, so its ``expected_host`` and ``reached`` are ``None`` (JSON null),
    whatever hosts it reached."""
    rec = api("turn_record")("D2", 0, _raw())
    assert rec["expected_host"] is None
    assert rec["reached"] is None
    assert json.loads(json.dumps(rec, default=str))["reached"] is None


def _expected_host(sid: str) -> str | None:
    """The host of the table's third column (the expected site, reading 3), lowercased; ``None`` when it has none."""
    start = _rows()[sid][1]
    host = urlsplit(str(start or "")).hostname
    return host.lower() if host else None


REACHED = [
    # (expected host, reached hosts, want), by hand from the ruling
    ("louvre.fr", ["louvre.fr"], True),
    ("louvre.fr", ["www.louvre.fr"], True),
    ("louvre.fr", ["cdn.example.com", "tickets.louvre.fr"], True),
    ("louvre.fr", ["notlouvre.fr"], False),
    ("louvre.fr", ["louvre.fr.evil.test"], False),
    ("louvre.fr", ["notlouvre.fr", "louvre.fr.evil.test", "louvre.com"], False),
    ("louvre.fr", [], False),
    ("www.rei.com", ["cdn.rei.com"], False),
    ("www.rei.com", ["www.rei.com"], True),
    (None, ["www.louvre.fr"], None),
    (None, [], None),
]


@pytest.mark.parametrize(
    "expected, hosts, want",
    REACHED,
    ids=[f"{e}<-{'+'.join(h) or 'none'}" for e, h, _ in REACHED],
)
def test_reached(expected: str | None, hosts: list[str], want: bool | None) -> None:
    """Reading 11 (RULED; both directions: a look-alike prefix ``notlouvre.fr`` and a suffix trick
    ``louvre.fr.evil.test`` are not reached, a sibling subdomain of ``www.rei.com`` is not its subdomain): reached iff
    some host equals the expected host or ends with ``"." + expected``; no expected site -> ``None``."""
    got = api("reached")(expected, hosts)
    assert got is want, (expected, hosts, got)


def test_no_record_holds_a_url_path() -> None:
    """D3r (adversarial; reviewer mutation 10: a path stored in a record): a unique token in the final URL's path,
    query and fragment, and in the search query, appears nowhere in the record (the report, its markdown and the jsonl:
    ``test_files_only_under_logs_shots_named_no_url_anywhere``)."""
    rec = api("turn_record")("D5", 0, _raw())
    assert PATH_TOKEN not in json.dumps(rec, default=str)
    assert "/checkout/" not in json.dumps(rec, default=str)


@pytest.mark.parametrize(
    "label_case",
    [
        ("D1", 1, 1, "ran"),
        ("D1", 2, 2, "ran"),
        ("D3", 0, 1, "incomplete"),
        ("D4", 1, 2, "incomplete"),
        ("D7", 0, 1, "refused"),
        ("D7", 0, 2, "refused"),
    ],
    ids=lambda c: f"{c[0]}-{c[1]}of{c[2]}",
)
def test_scenario_label(label_case: tuple[str, int, int, str]) -> None:
    """D4r: the only labels are ``ran`` (every repeat done), ``incomplete`` (fewer) and ``refused`` (D7)."""
    sid, done, repeats, want = label_case
    assert api("scenario_label")(sid, done, repeats) == want


# =====================================================================================================
# D3r: screenshot paths
# =====================================================================================================


def test_shot_path_shape(tmp_path: Path) -> None:
    """D3r (reading 8): ``<out>/shots/<id>/<repeat>-<step>.jpg``."""
    out = tmp_path / "logs" / f"t0205-real-{SHA7}"
    assert Path(api("shot_path")(out, "D4", 1, 7)) == out / "shots" / "D4" / "1-7.jpg"


@pytest.mark.parametrize("bad", ["../D1", "..", "D1/../../x", "/etc/D1", "D1/x", "D1\\..\\x"])
def test_shot_path_refuses_escapes(tmp_path: Path, bad: str) -> None:
    """D3r and Tests to write (adversarial: a ``../`` id writes outside the out-dir): an id holding a separator or
    ``..`` raises ``ValueError``; nothing is created."""
    out = tmp_path / "logs" / "o"
    with pytest.raises(ValueError):
        api("shot_path")(out, bad, 0, 1)
    assert not (tmp_path / "logs").exists() or not any((tmp_path / "logs").rglob("*.jpg"))


# =====================================================================================================
# D2r: executor checks (real executors, never launched)
# =====================================================================================================

START_ORIGIN = "http://127.0.0.1:8123"


def _executor(**k: Any) -> Any:
    from local_astra.computer.executor import BrowserToolExecutor

    base: dict[str, Any] = {
        "now_ms": lambda: 0,
        "headless": True,
        "allowed_origins": frozenset({START_ORIGIN}),
    }
    if k.get("web", True):
        base.update(web=True, start_origin=START_ORIGIN)
    else:
        base.update(web=False)
    k.pop("web", None)
    return BrowserToolExecutor(**base, **k)


def _resolve(*a: Any, **k: Any) -> Any:
    raise AssertionError("an injected resolver was called")


async def _connect(*a: Any, **k: Any) -> Any:
    raise AssertionError("an injected connector was called")


EXECUTORS: dict[str, Callable[[], Any]] = {
    "not-web": lambda: _executor(web=False),
    "injected-resolve": lambda: _executor(proxy_resolve=_resolve),
    "injected-connect": lambda: _executor(proxy_connect=_connect),
    "injected-both": lambda: _executor(proxy_resolve=_resolve, proxy_connect=_connect),
}


@pytest.mark.parametrize("case", sorted(EXECUTORS))
def test_executor_refusal_names_the_problem(case: str) -> None:
    """D2r: an executor not in web mode, or with an injected ``proxy_resolve``/``proxy_connect``, is refused with a
    reason naming it."""
    with t31.no_sockets() as attempts:
        ex = EXECUTORS[case]()
        why = api("executor_refusal")(ex)
    assert attempts == []
    assert isinstance(why, str) and why.strip(), (case, why)
    want = r"web" if case == "not-web" else r"proxy_resolve|proxy_connect|inject|resolver|connector"
    assert re.search(want, why, re.IGNORECASE), (case, why)


def test_executor_refusal_passes_a_clean_web_executor() -> None:
    """D2r, the too-wide direction: a web-mode executor with the real resolver and connector (none injected) is not
    refused."""
    with t31.no_sockets() as attempts:
        assert api("executor_refusal")(_executor()) is None
    assert attempts == []


# =====================================================================================================
# D2r/D4r: main under a rig that stubs everything; refusals before any socket
# =====================================================================================================


class Rig:
    """Every check passing (clean tree, sandbox ok, a clean web-mode executor, ``--real-sites`` and
    ``ASTRA_REAL_SITES=1``, the out-dir under a tmp ``LOGS_DIR``), every real socket refused and recorded;
    ``engine_healthy`` and ``run_turn`` recorded (``run_turn`` returns ``_raw`` with the scenario's own host)."""

    def __init__(
        self, tmp_path: Path, mp: pytest.MonkeyPatch, module: ModuleType | None = None
    ) -> None:
        self.h = h = harness() if module is None else module
        self.mp = mp
        self.sockets: list[str] = []
        self.calls: list[str] = []
        self.turns: list[tuple[str, int]] = []
        self.turn_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.porcelain = ""
        self.sandbox: str | None = None
        self.executor_case: str | None = None
        self.healthy = False
        self.logs = tmp_path / "logs"
        self.logs.mkdir()
        self.out = self.logs / f"t0205-real-{SHA7}"
        self.clean_executor = _executor()
        self.bad_executors = {k: f() for k, f in EXECUTORS.items()}
        mp.setenv("ASTRA_REAL_SITES", "1")

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0205 D rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            self.calls.append("git " + " ".join(args))
            if args[:1] == ("status",):
                return self.porcelain
            return GIT + "\n"

        def sandbox(*a: Any, **k: Any) -> str | None:
            self.calls.append("_sandbox_check")
            return self.sandbox

        def make_executor(*a: Any, **k: Any) -> Any:
            self.calls.append("_make_executor")
            if self.executor_case is None:
                return self.clean_executor
            return self.bad_executors[self.executor_case]

        def health(*a: Any, **k: Any) -> bool:
            self.calls.append("engine_healthy")
            return self.healthy

        async def run_turn(*a: Any, **k: Any) -> dict[str, Any]:
            sid = str(k.get("scenario_id", a[0] if a else None))
            repeat = int(k.get("repeat", a[3] if len(a) > 3 else -1))
            self.calls.append("run_turn")
            self.turns.append((sid, repeat))
            self.turn_args.append((tuple(a), dict(k)))
            return _raw(
                final_url=f"https://www.{sid.lower()}.example.com/checkout/{PATH_TOKEN}?q={PATH_TOKEN}"
            )

        for seam in (
            "_git",
            "_sandbox_check",
            "_make_executor",
            "engine_healthy",
            "run_turn",
            "LOGS_DIR",
        ):
            assert hasattr(h, seam), f"T-0205 D: the harness has no module-level seam {seam!r}"
        mp.setattr(h, "_git", git)
        mp.setattr(h, "_sandbox_check", sandbox)
        mp.setattr(h, "_make_executor", make_executor)
        mp.setattr(h, "engine_healthy", health)
        mp.setattr(h, "run_turn", run_turn)
        mp.setattr(h, "LOGS_DIR", self.logs)

    def argv(self, *extra: str, out: Path | None = None) -> list[str]:
        return ["--base-url", VALID_URL, "--out-dir", str(out or self.out), "--real-sites", *extra]

    def main(self, argv: list[str]) -> int:
        try:
            return int(self.h.main(argv))
        except SystemExit as exc:  # argparse refusals exit 2 by SystemExit
            return int(exc.code) if isinstance(exc.code, int) else 2


REFUSALS = (
    "flag-missing",
    "env-missing",
    "both-missing",
    "env-0",
    "env-true",
    "env-yes",
    "executor-not-web",
    "executor-injected-resolve",
    "executor-injected-connect",
    "sandbox-fails",
    "url-not-loopback",
    "url-localhost-lookalike",
    "dirty-tree",
    "out-dir-exists",
    "out-dir-outside-logs",
    "d7-alone",
    "d7-last",
    "d7-first",
    "d7-middle",
    "repeats-3",
)
REFUSAL_NAMES = {
    "flag-missing": r"--real-sites",
    "env-missing": r"ASTRA_REAL_SITES",
    "both-missing": r"real.sites|ASTRA_REAL_SITES",
    "env-0": r"ASTRA_REAL_SITES",
    "env-true": r"ASTRA_REAL_SITES",
    "env-yes": r"ASTRA_REAL_SITES",
    "executor-not-web": r"web",
    "executor-injected-resolve": r"proxy_resolve|inject|resolver",
    "executor-injected-connect": r"proxy_connect|inject|connector",
    "sandbox-fails": r"sandbox",
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "url-localhost-lookalike": r"loopback|nip\.io",
    "dirty-tree": r"session\.py",
    "out-dir-exists": r"exist",
    "out-dir-outside-logs": r"logs",
    "d7-alone": r"D7",
    "d7-last": r"D7",
    "d7-first": r"D7",
    "d7-middle": r"D7",
    "repeats-3": r"repeat|3",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """D2r/D4r (reviewer mutation 9: the env check dropped): each refusal exits 2 with a stderr line naming it,
    before any socket and before ``engine_healthy``/``run_turn``, leaving the out-dir as it was."""
    rig = Rig(tmp_path, monkeypatch)
    argv = rig.argv()
    if case == "flag-missing":
        argv.remove("--real-sites")
    elif case == "env-missing":
        monkeypatch.delenv("ASTRA_REAL_SITES")
    elif case == "both-missing":
        argv.remove("--real-sites")
        monkeypatch.delenv("ASTRA_REAL_SITES")
    elif case.startswith("env-"):
        monkeypatch.setenv(
            "ASTRA_REAL_SITES", {"env-0": "0", "env-true": "true", "env-yes": "yes"}[case]
        )
    elif case == "executor-not-web":
        rig.executor_case = "not-web"
    elif case == "executor-injected-resolve":
        rig.executor_case = "injected-resolve"
    elif case == "executor-injected-connect":
        rig.executor_case = "injected-connect"
    elif case == "sandbox-fails":
        rig.sandbox = "Chromium sandboxing failed!"
    elif case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "url-localhost-lookalike":
        argv[1] = "http://127.0.0.1.nip.io:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "out-dir-outside-logs":
        argv[3] = str(tmp_path / "elsewhere" / "run")
    elif case.startswith("d7-"):
        spec = {"d7-alone": "D7", "d7-last": "D1,D7", "d7-first": "D7,D1", "d7-middle": "D2,D7,D9"}[
            case
        ]
        argv += ["--scenarios", spec]
    elif case == "repeats-3":
        argv += ["--repeats", "3"]
    else:
        raise KeyError(case)
    tree_before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    rc = rig.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    for seam in ("engine_healthy", "run_turn"):
        assert seam not in rig.calls, (case, rig.calls)
    tree_after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert tree_after == tree_before, (
        f"{case}: the refusal wrote files ({set(tree_after) ^ set(tree_before)})"
    )
    assert err.strip(), f"{case}: the refusal says nothing"
    assert "unrecognized arguments" not in err, (
        f"{case}: argparse does not know an option ({err[-500:]})"
    )
    assert re.search(REFUSAL_NAMES[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


@pytest.mark.parametrize(
    "extra",
    [(), ("--repeats", "2"), ("--repeats", "1"), ("--scenarios", "D9,D1")],
    ids=["default", "repeats-2", "repeats-1", "d9-d1"],
)
def test_rig_passes_every_refusal_then_no_engine_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
) -> None:
    """D2r/D4r (the rig's premise, and the too-wide direction: the pair present, ``--repeats 2`` and a D7-free list
    are not refusals): every check passing and no healthy engine -> ``engine_healthy`` asked once, exit 1, no turn, no
    real socket."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.main(rig.argv(*extra))
    err = capsys.readouterr().err
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.calls.count("engine_healthy") == 1 and "run_turn" not in rig.calls, rig.calls
    assert "_sandbox_check" in rig.calls and "_make_executor" in rig.calls, rig.calls
    assert rig.sockets == [], rig.sockets


# =====================================================================================================
# D3r/D4r: a full stubbed run
# =====================================================================================================


def _full_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
    default_out: bool = False,
) -> tuple[Rig, int]:
    rig = Rig(tmp_path, monkeypatch)
    rig.healthy = True
    argv = rig.argv(*extra)
    if default_out:
        i = argv.index("--out-dir")
        del argv[i : i + 2]
    rc = rig.main(argv)
    err = capsys.readouterr().err
    assert rig.sockets == [], rig.sockets
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    return rig, rc


def test_default_run_is_n1_in_order_without_d7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4r (n = 1 by default, in order, D7 never run): one turn per runnable scenario, in ``REAL_SCENARIOS`` order."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys)
    order = [str(r[0]) for r in harness().REAL_SCENARIOS if str(r[0]) != "D7"]
    assert [sid for sid, _ in rig.turns] == order, rig.turns
    assert len({rep for _, rep in rig.turns}) == 1, rig.turns
    assert "D7" not in {sid for sid, _ in rig.turns}


def test_repeats_2_runs_each_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4r: ``--repeats 2`` runs every runnable scenario twice (two distinct repeat indices each), D7 never, each
    repeat's scenarios in table order."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys, "--repeats", "2")
    assert len(rig.turns) == 2 * len(RUNNABLE), rig.turns
    reps = sorted({rep for _, rep in rig.turns})
    assert len(reps) == 2, rig.turns
    order = [str(r[0]) for r in harness().REAL_SCENARIOS if str(r[0]) != "D7"]
    for rep in reps:
        assert [sid for sid, r in rig.turns if r == rep] == order, (rep, rig.turns)
    assert "D7" not in {sid for sid, _ in rig.turns}


def test_files_only_under_logs_shots_named_no_url_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D3r (reading 6, 8; adversarial, reviewer mutation 10): with the default out-dir, everything written is under
    ``LOGS_DIR/t0205-real-<sha7>/``; each step's frame is ``shots/<id>/<repeat>-<step>.jpg`` holding the frame's
    bytes; no file (jsonl, report, markdown) holds the token from the final URL's path and query, nor a
    ``/checkout/`` path."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys, default_out=True)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    out = rig.logs / f"t0205-real-{SHA7}"
    assert files, "the run wrote nothing"
    outside = [str(p) for p in files if out not in p.parents]
    assert not outside, f"D3r: written outside logs/t0205-real-<sha7>/: {outside}"
    shots = sorted(p.relative_to(out).as_posix() for p in files if p.suffix == ".jpg")
    want = sorted(f"shots/{sid}/{rep}-{s}.jpg" for sid, rep in rig.turns for s in (1, 2, 3))
    assert shots == want, (shots, want)
    for sid, rep in rig.turns:
        assert (out / "shots" / sid / f"{rep}-2.jpg").read_bytes() == b"\xff\xd8JPEG\x02"
    for p in files:
        if p.suffix == ".jpg":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        assert PATH_TOKEN not in text, f"D3r: {p.name} holds a URL path or query"
        assert "/checkout/" not in text, f"D3r: {p.name} holds a URL path"
    assert any(p.suffix == ".jsonl" for p in files), "no jsonl record written"
    assert (out / "report.json").is_file() and (out / "report.md").is_file()


def test_report_per_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4r: the report lists per scenario its label (``ran`` for the eight, ``refused`` for D7), ``final_kind``,
    hosts, steps and screenshot paths; the final host is reported, never the URL; the markdown names every
    scenario and every screenshot path."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys)
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    scen = report["scenarios"]
    assert scen["D7"]["label"] == "refused"
    md = (rig.out / "report.md").read_text(encoding="utf-8")
    for sid in RUNNABLE:
        row = scen[sid]
        assert row["label"] == "ran", (sid, row)
        assert "checkout" in json.dumps(row["final_kind"]), (sid, row)
        assert "www.rei.com" in json.dumps(row["hosts"]) and "cdn.rei.com" in json.dumps(
            row["hosts"]
        ), (sid, row)
        assert "6" in json.dumps(row["steps"]), (sid, row)
        shots = json.dumps(row["shots"])
        for s in (1, 2, 3):
            assert f"shots/{sid}/0-{s}.jpg" in shots or f"{sid}/0-{s}.jpg" in shots, (
                sid,
                row["shots"],
            )
            assert f"{sid}/0-{s}.jpg" in md, (sid, s)
        assert sid in md
        assert f"www.{sid.lower()}.example.com" in json.dumps(report), (
            f"{sid}: the final host is not reported"
        )
    labels = set(re.findall(r"\b(ran|incomplete|refused|measured|void|PASS|FAIL)\b", md))
    assert labels <= {"ran", "incomplete", "refused"}, (
        f"D4r: no label beyond ran/incomplete/refused: {labels}"
    )


def test_turns_start_at_the_start_page_never_the_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading 3 (RULED: H-T205-5 wins; adversarial: a harness that opens each site directly measures less of the
    demo): every ``run_turn`` call gets ``start`` = the live start page (a URL or path whose path is ``/web.html``),
    and no argument of any call names the scenario's expected site (its host or its https URL)."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys)
    assert len(rig.turn_args) == len(RUNNABLE), rig.turn_args
    for (args, kwargs), (sid, _rep) in zip(rig.turn_args, rig.turns, strict=True):
        assert "start" in kwargs, f"{sid}: run_turn was not given start= ({sorted(kwargs)})"
        start = str(kwargs["start"])
        assert urlsplit(start).path == "/web.html", f"{sid}: the turn starts at {start!r}"
        expected = _expected_host(sid)
        if expected is None:
            continue
        site = str(_rows()[sid][1])
        blob = repr(args) + repr(sorted(kwargs.items(), key=lambda kv: kv[0]))
        assert site not in blob and expected not in blob.lower(), (
            f"{sid}: run_turn was handed the expected site ({expected}): {blob[:500]}"
        )


def test_report_expected_host_and_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Readings 3 and 11 (RULED): the report records each scenario's ``expected_host`` (the table column's host, D2
    null) and ``reached``. Every stubbed turn reached www.rei.com and cdn.rei.com only, so by hand: D5 reached, D2
    null, every other scenario not reached."""
    rig, _ = _full_run(tmp_path, monkeypatch, capsys)
    scen = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))["scenarios"]
    for sid in RUNNABLE:
        row = scen[sid]
        assert "expected_host" in row and "reached" in row, (sid, sorted(row))
        assert row["expected_host"] == _expected_host(sid), (sid, row["expected_host"])
        want = None if sid == "D2" else sid == "D5"
        assert row["reached"] is want, (sid, row["reached"], want)


def test_budget_exhausted_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4r (``RUN_BUDGET_S`` read at call time): with no budget left, fewer than eight turns run, every scenario not
    run is labelled ``incomplete`` (never ``ran``), D7 stays ``refused``; still no socket."""
    rig = Rig(tmp_path, monkeypatch)
    rig.healthy = True
    monkeypatch.setattr(rig.h, "RUN_BUDGET_S", 0)
    rc = rig.main(rig.argv())
    capsys.readouterr()
    assert rig.sockets == []
    assert rc in (0, 1)
    assert len(rig.turns) < len(RUNNABLE), rig.turns
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    ran = {sid for sid, _ in rig.turns}
    for sid in RUNNABLE:
        want = "ran" if sid in ran else "incomplete"
        assert report["scenarios"][sid]["label"] == want, (sid, report["scenarios"][sid])
    assert report["scenarios"]["D7"]["label"] == "refused"


# =====================================================================================================
# Live keywords (the R-B1-style drift row, T-0207 B)
# =====================================================================================================

#: cli Session keywords that are plumbing, not settings (as T-0207 B's ``CLI_PLUMBING``)
CLI_PLUMBING = frozenset(
    {
        "clock",
        "config",
        "session_log",
        "log_dir",
        "tool_kit",
        "grounding",
        "mic_unavailable",
        "reground_llm",
        "reground_visual_tokens",
    }
)


def _cli_web_live_keywords(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The ``Session`` keywords the cli passes at HEAD for ``session --tools live`` in web mode (no ``--no-web``;
    T-0132's build rig over fakes, nothing served), less ``CLI_PLUMBING``. The rig's patches are undone."""
    from tests.unit import test_cli_t0132 as c132

    with t31.no_sockets() as attempts:
        calls, _ = c132.build(tmp_path / "cli", capsys, monkeypatch)
    assert attempts == [], attempts
    kwargs = dict(calls.session_kwargs[-1])
    monkeypatch.undo()
    capsys.readouterr()
    return {k: v for k, v in kwargs.items() if k not in CLI_PLUMBING}


def test_no_live_web_keyword_of_the_cli_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading 9 (adversarial, the drift in general: T-0201 D made the screen reminder a live default after T-0201
    C's extras were written, and those runs silently went without it): every setting the cli passes ``Session`` at
    HEAD for ``--tools live`` in web mode is in ``session_keywords()`` with the cli's value. The premise: the cli's
    keywords are non-empty and include the screen reminder on."""
    want = _cli_web_live_keywords(tmp_path, capsys, monkeypatch)
    assert want.get("screen_reminder") is True, sorted(want)
    # T-0207 C3 re-pin (R-C2): T-0207 C1 makes the completion check a live default; the cli passes
    # completion_check=True, so session_keywords() must carry it (resolved through cli, compared below with the rest)
    assert want.get("completion_check") is True, (
        f"T-0207 C1 premise: the cli passes completion_check=True for live web; {sorted(want)}"
    )
    with t31.no_sockets() as attempts:
        got = dict(api("session_keywords")())
    assert attempts == []
    problems = sorted(k for k, v in want.items() if k not in got or got[k] != v)
    assert not problems, (
        f"lacking or differing from the cli's live web keywords: {[(k, want[k], got.get(k)) for k in problems]}"
    )


# =====================================================================================================
# D5r: the locked wrapper
# =====================================================================================================


def _shell_code(text: str) -> list[str]:
    """The script's lines with comments removed (full-line ``#`` comments dropped)."""
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_wrapper_requires_the_d_merge_and_holds_the_export() -> None:
    """D5r: the wrapper refuses unless a merge whose subject starts ``Merge T-0205-d:`` is reachable from HEAD; it
    exports ``ASTRA_REAL_SITES=1`` itself (the token-gated command is the consent), exactly once, after that check;
    it runs the harness with ``--real-sites``; the output goes to ``logs/t0205-real-``; it is bounded by ``timeout``
    and refuses a dirty tree."""
    code = _shell_code(wrapper_text())
    joined = "\n".join(code)
    merge_lines = [i for i, ln in enumerate(code) if "Merge T-0205-d:" in ln]
    assert merge_lines, "D5r: no 'Merge T-0205-d:' ancestor check"
    assert re.search(r"git log --merges[^\n]*HEAD", joined), (
        "D5r: the merge check does not read HEAD's merges"
    )
    exports = [i for i, ln in enumerate(code) if re.search(r"\bexport\s+ASTRA_REAL_SITES=1\b", ln)]
    assert len(exports) == 1, f"D5r: export ASTRA_REAL_SITES=1 appears {len(exports)} times"
    assert exports[0] > merge_lines[0], "D5r: the consent is exported before the merge check"
    assert "scripts/gpu/t0205_real_sites.py" in joined
    assert re.search(r"t0205_real_sites\.py[\s\S]{0,400}--real-sites", joined), (
        "D5r: the harness runs without --real-sites"
    )
    assert "logs/t0205-real-" in joined
    assert re.search(r"\btimeout\b", joined)
    assert "git status --porcelain" in joined
    assert "services/vllm.sh" in joined, "D5r: boot as C (services/vllm.sh)"


def _stub_dir(tmp: Path, log: Path) -> Path:
    stubs = tmp / "bin"
    stubs.mkdir()
    for name in (
        "docker",
        "make",
        "ss",
        "uv",
        "timeout",
        "sg",
        "curl",
        "nvidia-smi",
        "setsid",
        "python",
        "python3",
    ):
        p = stubs / name
        p.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 1\n', encoding="utf-8")
        p.chmod(0o755)
    return stubs


def _git_in(repo: Path, *args: str) -> None:
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


@pytest.mark.parametrize(
    "subject",
    [
        None,
        "Merge main into T-0205-d-real",
        "Merge T-0205-c: demo run harness",
        "Merge T-0205-dx: other",
    ],
    ids=["no-merge", "branch-merge", "c-merge", "dx-merge"],
)
def test_wrapper_refuses_without_the_d_merge(tmp_path: Path, subject: str | None) -> None:
    """D5r (behavioural): in a throwaway clean repo holding the wrapper, a HEAD with no ``Merge T-0205-d:`` merge (none,
    a branch merge, group C's, a look-alike) exits 2 naming T-0205-d, and nothing on PATH (uv, docker, make, the
    engine) is ever run."""
    text = wrapper_text()
    repo = tmp_path / "repo"
    (repo / "scripts" / "locked").mkdir(parents=True)
    script = repo / "scripts" / "locked" / "t0205_real_sites.sh"
    script.write_text(text, encoding="utf-8")
    _git_in(repo, "init", "-q", "-b", "main")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-q", "-m", "base")
    if subject is not None:
        _git_in(repo, "switch", "-q", "-c", "side")
        (repo / "side.txt").write_text("x", encoding="utf-8")
        _git_in(repo, "add", "-A")
        _git_in(repo, "commit", "-q", "-m", "side")
        _git_in(repo, "switch", "-q", "main")
        _git_in(repo, "merge", "-q", "--no-ff", "-m", subject, "side")
    log = tmp_path / "stub.log"
    log.write_text("", encoding="utf-8")
    stubs = _stub_dir(tmp_path, log)
    bash = shutil.which("bash")
    assert bash
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "PATH": f"{stubs}:{os.environ.get('PATH', '')}",
    }
    env.pop("ASTRA_REAL_SITES", None)
    proc = subprocess.run(
        [bash, str(script)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])
    assert "T-0205-d" in proc.stderr, proc.stderr[-2000:]
    assert log.read_text(encoding="utf-8") == "", (
        f"D5r: a stub ran before the refusal: {log.read_text()}"
    )


# =====================================================================================================
# D6r: the negative, over the closed range BASE_D..TIP_D (14733dcb..78b9d7db, D's implementation)
# =====================================================================================================

FROZEN_D = (
    "local_astra/computer/fence.py",
    "local_astra/computer/egress_proxy.py",
    "local_astra/search",
)
#: D's own new files: the only ones allowed to appear under scripts/gpu and scripts/locked in D's range
D_OWNED_SCRIPTS = frozenset(
    {"scripts/gpu/t0205_real_sites.py", "scripts/locked/t0205_real_sites.sh"}
)


def test_frozen_paths_over_closed_range_d() -> None:
    """D6r (green at BASE_D): no ``local_astra/`` file changes in D; ``fence.py``, ``egress_proxy.py`` and
    ``search/**`` are byte-identical, and every earlier harness under ``scripts/gpu`` and ``scripts/locked`` is too
    (D's two files are the only ones that may appear), over ``BASE_D..TIP_D`` (two literals, never the working tree);
    each frozen path exists at BASE_D."""
    gr.require_commits(BASE_D, TIP_D)
    assert gr.is_ancestor(BASE_D, TIP_D), f"{BASE_D[:7]} is not an ancestor of {TIP_D[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_D, "--", *FROZEN_D).stdout.split()
    for p in FROZEN_D:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at BASE_D"
        )
    assert gr.changed(BASE_D, TIP_D, "local_astra") == [], "D6r: D changed local_astra/"
    assert gr.changed(BASE_D, TIP_D, *FROZEN_D) == []
    scripts = [
        p
        for p in gr.changed(BASE_D, TIP_D, "scripts/gpu", "scripts/locked")
        if p not in D_OWNED_SCRIPTS
    ]
    assert scripts == [], f"D6r: earlier harnesses differ in {BASE_D[:7]}..{TIP_D[:7]}: {scripts}"
    added_only = gr.changed(BASE_D, TIP_D, *sorted(D_OWNED_SCRIPTS), diff_filter="M")
    assert added_only == [], f"D6r: D's own scripts existed at BASE_D: {added_only}"


def test_frozen_predicate_can_see_a_change() -> None:
    """D6r power row: the same predicate over the last commit that touched ``egress_proxy.py`` before BASE_D (found
    from the fixed BASE_D, never a moving ref) and over the last that touched a ``scripts/gpu`` harness reports
    them."""
    gr.require_commits(BASE_D)
    for path in ("local_astra/computer/egress_proxy.py", "scripts/gpu"):
        last = gr._git("log", "-n", "1", "--format=%H", BASE_D, "--", path).stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{40}", last), (path, last)
        parent = gr._git("rev-parse", f"{last}^").stdout.strip()
        assert gr.changed(parent, last, path), (
            f"the predicate cannot see {last[:7]}'s change to {path}"
        )


# =====================================================================================================
# Review round 1 (2026-10-30): I1-I5, T1-T3 (rulings in brackets in each docstring)
# =====================================================================================================


def fresh_harness() -> ModuleType:
    """A fresh, uncached copy of the harness module (its per-process search cache starts empty; the shared
    ``harness()`` copy is never touched by rows that start search)."""
    assert HARNESS.is_file(), f"T-0205 D1r: {HARNESS.relative_to(REPO)} does not exist yet"
    name = f"_t0205_real_sites_fresh_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


# --- I1/T4: a screenshot per step, from the executor's own latest frame ----------------------------------


class _ShotInner:
    """A fake live executor: ``frame()`` is a fresh capture (counted); ``execute`` sets ``latest_frame`` to the
    frame after the action, as the real executor does, and the session then reads ``latest_frame`` without calling
    ``frame()`` again (session.py ``latest_frame`` reuse)."""

    ONSET = b"\xff\xd8onset-frame"

    def __init__(self) -> None:
        self.frame_calls = 0
        self.executes = 0
        self._latest: Any = None
        self.mode = "live"

    @property
    def latest_frame(self) -> Any:
        return self._latest

    async def frame(self) -> Any:
        self.frame_calls += 1
        self._latest = SimpleNamespace(jpeg=self.ONSET, phash="onset")
        return self._latest

    async def execute(self, inv: Any) -> Any:
        self.executes += 1
        self._latest = SimpleNamespace(jpeg=_step_jpeg(self.executes), phash=f"p{self.executes}")
        return SimpleNamespace(status="executed", reason=None, resolved={"hosts": []})


def _step_jpeg(step: int) -> bytes:
    return b"\xff\xd8after-step-" + str(step).encode() + b"\xff\xd9"


def test_watch_records_a_shot_per_step_from_latest_frame() -> None:
    """I1/T4 (reviewer: the live session reuses ``latest_frame`` after an action and never calls ``frame()``, so a
    watch that records only ``frame()`` writes step 0 and nothing else). After the onset ``frame()`` and three
    ``execute`` calls on a fake whose ``execute`` sets ``latest_frame``, the watch holds the shot of every step:
    step 0 the onset frame, step k the bytes of the frame after the k-th action, exactly. ``_Watch.frames``
    (step -> bytes) is the seam ``run_turn`` turns into the raw ``frames`` that ``main`` writes to
    ``shots/<id>/<rep>-<step>.jpg`` (that half is pinned by ``test_files_only_under_logs_shots_named_no_url_anywhere``
    and by ``test_watch_shots_land_at_their_paths``)."""
    inner = _ShotInner()
    watch = api("_Watch")(inner)

    async def drive() -> None:
        await watch.frame()
        for _ in range(3):
            await watch.execute(SimpleNamespace(name="click_at", args={}))

    asyncio.run(drive())
    assert inner.frame_calls == 1, "premise: frame() is called only at the onset"
    assert inner.executes == 3
    frames = {int(k): bytes(v) for k, v in dict(watch.frames).items()}
    want = {0: _ShotInner.ONSET, 1: _step_jpeg(1), 2: _step_jpeg(2), 3: _step_jpeg(3)}
    assert frames == want, (
        f"I1: shots per step {sorted(frames)}; want {sorted(want)} with the exact bytes"
    )


def test_watch_shots_land_at_their_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """I1/T4 end to end through ``main``: a stubbed turn that drives a real ``_Watch`` over ``_ShotInner`` (onset
    ``frame()``, then two ``execute``) and hands back what the watch recorded as the raw ``frames`` leaves
    ``shots/<id>/0-0.jpg`` (the onset), ``0-1.jpg`` and ``0-2.jpg`` holding exactly the bytes of the frame after each
    action."""
    rig = Rig(tmp_path, monkeypatch)
    rig.healthy = True
    h = rig.h

    async def run_turn(*a: Any, **k: Any) -> dict[str, Any]:
        inner = _ShotInner()
        watch = h._Watch(inner)
        await watch.frame()
        for _ in range(2):
            await watch.execute(SimpleNamespace(name="click_at", args={}))
        rig.turns.append((str(k["scenario_id"]), int(k["repeat"])))
        frames = [{"step": int(s), "jpeg": bytes(j)} for s, j in dict(watch.frames).items()]
        return _raw(frames=frames, outcomes=list(watch.outcomes))

    monkeypatch.setattr(h, "run_turn", run_turn)
    rc = rig.main(rig.argv("--scenarios", "D3"))
    err = capsys.readouterr().err
    assert rc == 0, err[-2000:]
    assert rig.sockets == []
    got = {p.name: p.read_bytes() for p in (rig.out / "shots" / "D3").glob("*.jpg")}
    want = {"0-0.jpg": _ShotInner.ONSET, "0-1.jpg": _step_jpeg(1), "0-2.jpg": _step_jpeg(2)}
    assert got == want, (sorted(got), sorted(want))


# --- I2: reached over the reached hosts and the final host -------------------------------------------------

FINAL_REACHED = [
    # (scenario, outcome hosts, final URL, want), by hand from the I2 ruling
    ("D1", [], "https://www.louvre.fr/en/visit", True),
    ("D1", [], "https://louvre.fr/", True),
    ("D1", ["cdn.example.com"], "https://tickets.louvre.fr/x", True),
    ("D1", [], "https://notlouvre.fr/", False),
    ("D1", [], "https://louvre.fr.evil.test/", False),
    ("D1", [], None, False),
    ("D1", ["www.louvre.fr"], "https://www.google.com/search?q=x", True),
    ("D5", [], "https://cdn.rei.com/", True),
    ("D2", [], "https://www.louvre.fr/", None),
]


@pytest.mark.parametrize(
    "sid, hosts, final_url, want",
    FINAL_REACHED,
    ids=[
        f"{s}-{'+'.join(h) or 'nohosts'}-{urlsplit(u or '').hostname}"
        for s, h, u, _ in FINAL_REACHED
    ],
)
def test_turn_record_reached_counts_the_final_host(
    sid: str, hosts: list[str], final_url: str | None, want: bool | None
) -> None:
    """I2 (RULED: ``reached`` over the reached hosts and the final host; both directions: the final host
    www.louvre.fr reaches D1 with no outcome hosts, while a look-alike final host, a suffix trick or no final URL
    does not; D2 stays ``None``)."""
    outcomes = [
        {
            "step": 1,
            "tool": "click_at",
            "status": "executed",
            "reason": None,
            "resolved": {"hosts": hosts},
        }
    ]
    rec = api("turn_record")(
        sid, 0, _raw(outcomes=outcomes, final_url=final_url, session_records=[])
    )
    assert rec["reached"] is want, (sid, hosts, final_url, rec["reached"])


# --- I3: search off -------------------------------------------------------------------------------------

PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "DDGS_PROXY",
    "PRIMP_PROXY",
)


def _unset_on_harness_line(code: list[str]) -> set[str]:
    """The variables ``env -u`` removes on the line that runs the harness, inline or through an array it expands."""
    arrays: dict[str, set[str]] = {}
    for ln in code:
        m = re.match(r"\s*(\w+)=\(env\s+(.*)\)\s*$", ln)
        if m:
            arrays[m.group(1)] = set(re.findall(r"-u\s+(\w+)", m.group(2)))
    holders = {
        m.group(1)
        for ln in code
        if (m := re.match(r"\s*(\w+)=\S*scripts/gpu/t0205_real_sites\.py", ln))
    }
    runs = [
        ln
        for ln in code
        if re.search(r"\buv\s+run\b|\bpython3?\b", ln)
        and (
            "scripts/gpu/t0205_real_sites.py" in ln
            or any(re.search(r"\$\{?" + v + r"\b", ln) for v in holders)
        )
    ]
    assert runs, "the wrapper never runs the harness"
    unset: set[str] | None = None
    for ln in runs:
        here = (
            set(re.findall(r"\benv\b[^|;&]*?-u\s+(\w+)", ln))
            if re.search(r"\benv\s+-u", ln)
            else set()
        )
        for name, names in arrays.items():
            if re.search(r"\$\{" + name + r"\[@\]\}", ln):
                here |= names
        unset = here if unset is None else unset & here
    return unset or set()


def test_wrapper_unsets_every_proxy_variable() -> None:
    """I3 (static): the line that runs the harness unsets every proxy variable the search backend or the egress
    proxy could read: the six usual ones and ``DDGS_PROXY`` and ``PRIMP_PROXY`` (ddgs/primp read their own; a
    proxy there turns search off, T-0125)."""
    unset = _unset_on_harness_line(_shell_code(wrapper_text()))
    missing = [v for v in PROXY_VARS if v not in unset]
    assert not missing, f"I3: the harness runs with {missing} still set (unset: {sorted(unset)})"


SEARCH_WHY = "the search extra is not installed (no module named ddgs)"
SEARCH_FIX = "uv sync --extra search"


def _find_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                found.append(v)
            found += _find_key(v, key)
    elif isinstance(value, list):
        for v in value:
            found += _find_key(v, key)
    return found


def _search_rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fails: bool,
) -> tuple[Rig, list[str]]:
    """A ``Rig`` over a fresh module, with ``cli.search_start`` patched (failing or starting a fake backend) and a
    stubbed turn that reports search as a real ``run_turn`` would (``search_off`` = "why; fix" when it failed)."""
    from local_astra import cli

    started: list[str] = []

    class _Backend:
        async def aclose(self) -> None:
            return None

    def search_start(backend: str) -> Any:
        started.append(backend)
        if fails:
            return None, (SEARCH_WHY, SEARCH_FIX)
        return (_Backend(), "keyless: duckduckgo"), None

    monkeypatch.setattr(cli, "search_start", search_start)
    rig = Rig(tmp_path, monkeypatch, module=fresh_harness())
    rig.healthy = True

    async def run_turn(*a: Any, **k: Any) -> dict[str, Any]:
        rig.turns.append((str(k["scenario_id"]), int(k["repeat"])))
        return _raw(search_off=f"{SEARCH_WHY}; {SEARCH_FIX}" if fails else None)

    monkeypatch.setattr(rig.h, "run_turn", run_turn)
    return rig, started


def test_search_failure_shows_in_manifest_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """I3 (RULED): when web search failed to start, the manifest's ``search`` says off (or records the failure),
    and ``report.json`` holds a ``search_off`` carrying the reason, which ``report.md`` shows too (beside the words
    "search off"/"search_off"). A rehearsal that silently ran without search would read as the model's choice."""
    rig, _ = _search_rig(tmp_path, monkeypatch, fails=True)
    rc = rig.main(rig.argv("--scenarios", "D2,D1"))
    err = capsys.readouterr().err
    assert rc == 0, err[-2000:]
    assert rig.sockets == []
    manifest = json.loads((rig.out / "manifest.json").read_text(encoding="utf-8"))
    search = (manifest.get("settings") or {}).get("search", manifest.get("search"))
    said_off = search in (None, False, "off") or SEARCH_WHY in json.dumps(search)
    recorded = any(SEARCH_WHY in json.dumps(v) for v in _find_key(manifest, "search_off"))
    assert said_off or recorded, f"I3: the manifest still says search {search!r} with search failed"
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    offs = [v for v in _find_key(report, "search_off") if v]
    assert offs and any(SEARCH_WHY in json.dumps(v) for v in offs), (
        f"I3: report.json shows no search_off with the reason ({offs})"
    )
    md = (rig.out / "report.md").read_text(encoding="utf-8")
    assert re.search(r"search[ _]off", md, re.IGNORECASE), (
        "I3: report.md does not say search was off"
    )
    assert SEARCH_WHY in md, "I3: report.md does not show why search was off"


def test_search_started_is_not_reported_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """I3, the too-wide direction: search that started is reported on (the manifest's ``search`` is ``ddgs``), and
    neither report says search was off."""
    rig, _ = _search_rig(tmp_path, monkeypatch, fails=False)
    rc = rig.main(rig.argv("--scenarios", "D2,D1"))
    err = capsys.readouterr().err
    assert rc == 0, err[-2000:]
    manifest = json.loads((rig.out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["search"] == "ddgs", manifest["settings"]
    assert not [v for v in _find_key(manifest, "search_off") if v]
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    assert not [v for v in _find_key(report, "search_off") if v], report
    md = (rig.out / "report.md").read_text(encoding="utf-8")
    assert not re.search(r"search[ _]off", md, re.IGNORECASE), md


# --- T2: search starts once per process ----------------------------------------------------------------


@pytest.mark.parametrize("fails", [True, False], ids=["first-fails", "first-starts"])
def test_search_starts_once_per_process(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    """T2 (D2r: search is built once per process through the cli's own start path): ``_search("ddgs")`` twice calls
    ``cli.search_start`` exactly once, also when the first call failed (a failure is not retried per turn), and
    both calls give the same answer."""
    from local_astra import cli

    calls: list[str] = []
    backend = object()

    def search_start(name: str) -> Any:
        calls.append(name)
        return (None, (SEARCH_WHY, SEARCH_FIX)) if fails else ((backend, "keyless: x"), None)

    monkeypatch.setattr(cli, "search_start", search_start)
    h = fresh_harness()
    first = h._search("ddgs")
    second = h._search("ddgs")
    assert calls == ["ddgs"], f"T2: cli.search_start called {len(calls)} times"
    if fails:
        assert first is None and second is None
    else:
        assert first is not None and first[0] is backend and second is first


# --- I4: the sandbox check launches Chromium as web mode does ------------------------------------------


class _FakeLaunch:
    def __init__(self, fail: BaseException | None = None) -> None:
        self.launches: list[dict[str, Any]] = []
        self.fail = fail
        outer = self

        class _Browser:
            async def close(self) -> None:
                return None

        class _Chromium:
            async def launch(self, *a: Any, **k: Any) -> Any:
                outer.launches.append({"args_pos": a, **k})
                if outer.fail is not None:
                    raise outer.fail
                return _Browser()

        class _Pw:
            chromium = _Chromium()

            async def stop(self) -> None:
                return None

        class _Ctx:
            async def __aenter__(self) -> Any:
                return _Pw()

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def start(self) -> Any:
                return _Pw()

        self.ctx = _Ctx

    def __call__(self) -> Any:
        return self.ctx()


def _patch_launch(monkeypatch: pytest.MonkeyPatch, fake: _FakeLaunch) -> None:
    import playwright.async_api as pw_api

    from local_astra import cli

    monkeypatch.setattr(pw_api, "async_playwright", fake)
    monkeypatch.setattr(cli, "live_prerequisites", lambda *a, **k: None)


def test_sandbox_check_launches_with_the_chromium_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """I4 (RULED): ``_sandbox_check`` launches Chromium headless with its sandbox on and with
    ``runtime.CHROMIUM_ARGS`` (among them ``--host-resolver-rules=MAP * ~NOTFOUND``, so even the check cannot
    resolve a name); the launcher is patched, so nothing starts."""
    from local_astra.computer import runtime

    fake = _FakeLaunch()
    _patch_launch(monkeypatch, fake)
    with t31.no_sockets() as attempts:
        why = api("_sandbox_check")()
    assert attempts == []
    assert why is None, why
    assert len(fake.launches) == 1, fake.launches
    kw = fake.launches[0]
    assert kw.get("chromium_sandbox") is True and kw.get("headless", True) is True, kw
    args = list(kw.get("args") or ())
    missing = [a for a in runtime.CHROMIUM_ARGS if a not in args]
    assert not missing, f"I4: the sandbox check launches without {missing}"
    assert any(a.startswith("--host-resolver-rules=MAP * ~NOTFOUND") for a in args), args


def test_sandbox_check_failure_names_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """I4 companion: a launch that fails is a reason naming the sandbox, not an exception."""
    fake = _FakeLaunch(fail=RuntimeError("No usable sandbox! see crbug"))
    _patch_launch(monkeypatch, fake)
    why = api("_sandbox_check")()
    assert isinstance(why, str) and re.search("sandbox", why, re.IGNORECASE), why


# --- I5: the executor's notes reach the session log ------------------------------------------------------

FAKE_ORIGIN = "http://127.0.0.1:5555"


class _FakePages:
    """The start-page server, never bound."""

    def __init__(self, *a: Any, **k: Any) -> None:
        self.started = False

    @property
    def origin(self) -> str:
        return FAKE_ORIGIN

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


PROBE_NOTE = {"what": "T-0205 I5 probe note", "detail": "blank capture re-taken"}


class _NoteProbe:
    """A web-mode executor stand-in (passes ``executor_refusal``; ``open``/``close`` do nothing). The row's patched
    ``Session.submit`` makes it emit one note through its own ``on_note`` at the moment the turn is submitted (as
    the real executor does for a re-taken blank capture during a turn, T-0200 C9)."""

    mode = "live"
    web = True
    closed = False
    latest_frame = None
    egress_log: tuple[dict[str, str], ...] = ()

    def __init__(self) -> None:
        self.driver = SimpleNamespace(
            _proxy_resolve=None,
            _proxy_connect=None,
            _user_data_dir=None,
            page=SimpleNamespace(url=FAKE_ORIGIN + "/web.html"),
        )
        self.start_page_url = FAKE_ORIGIN + "/web.html"
        self.on_note: Any = None
        self.opened: list[str] = []
        self.on_note_at_submit: Any = "never submitted"

    async def open(self, url: str) -> None:
        self.opened.append(url)

    async def frame(self) -> Any:
        raise AssertionError("I5 probe: frame() is never reached (the turn ends at submit)")

    async def execute(self, inv: Any) -> Any:
        raise AssertionError("I5 probe: execute() is never reached (the turn ends at submit)")

    async def close(self) -> None:
        return None


def test_run_turn_wires_the_executor_notes_into_the_session_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I5 (RULED: as ``cli.build_session`` does, T-0200 C9): the real ``run_turn`` sets the executor's own
    ``on_note`` (on the executor, not on a wrapper around it such as ``_Watch``), and a note the executor emits
    during the turn lands as a ``note`` record in the turn's session log. No browser, no engine, no socket: the
    start-page server and the executor are stand-ins, the detokeniser is the T-0088a subset, search start is
    patched to fail (no backend is built), the repair warm-up is a no-op, and ``Session.submit`` is patched to fire
    the probe's note and end the turn at once (so no request is ever sent)."""
    from local_astra import cli
    from local_astra.conductor import session as session_mod
    from local_astra.pages import server as pages_server
    from tests.unit import _tools_switch_support as T

    h = fresh_harness()
    probe = _NoteProbe()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    monkeypatch.setattr(pages_server, "StartPageServer", _FakePages)
    monkeypatch.setattr(h, "_make_executor", lambda *a, **k: probe)
    monkeypatch.setattr(cli, "search_start", lambda name: (None, (SEARCH_WHY, SEARCH_FIX)))

    async def no_warm(session: Any) -> None:
        return None

    def submit(self: Any, event: Any) -> None:
        probe.on_note_at_submit = probe.on_note
        if callable(probe.on_note):
            probe.on_note(dict(PROBE_NOTE))
        self.turn_written.set()

    monkeypatch.setattr(cli, "_warm_repair", no_warm)
    monkeypatch.setattr(session_mod.Session, "submit", submit)
    out = tmp_path / "logs" / "run"
    out.mkdir(parents=True)
    with t31.no_sockets() as attempts:
        raw = asyncio.run(
            h.run_turn(
                scenario_id="D1",
                request="Is the Louvre open today, and until when?",
                start="/web.html",
                repeat=0,
                base_url=VALID_URL,
                out_dir=out,
                timeout_s=5,
            )
        )
    assert attempts == [], attempts
    assert probe.opened and probe.on_note_at_submit != "never submitted", (
        f"premise: the turn never opened the browser or submitted (error {raw.get('error')!r})"
    )
    assert callable(probe.on_note_at_submit), (
        "I5: the turn ran with no on_note on the executor (cli.build_session sets it, T-0200 C9)"
    )
    notes = [
        r
        for r in raw.get("session_records") or ()
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == PROBE_NOTE["what"]
    ]
    assert len(notes) == 1, (
        f"I5: the executor's note is not in the session log ({raw.get('session_records')})"
    )


# --- T1: a URL in an error never reaches a file or stderr ------------------------------------------------

ERR_URL = f"https://errhost.example.org/{PATH_TOKEN}?q={PATH_TOKEN}"


@pytest.mark.parametrize("how", ["turn-error", "outcome-error", "run-turn-raises"])
def test_no_url_path_from_an_error_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], how: str
) -> None:
    """T1 (adversarial; kills ``scrub`` = identity): a URL with a unique path and query token in the turn's
    ``error``, in an outcome's ``error``/``reason``, or in the message of an exception ``run_turn`` raises, never
    reaches ``records.jsonl``, ``report.json``, ``report.md`` or stderr. Premise (non-vacuity): the error itself is
    recorded (its plain word ``boom`` is in ``records.jsonl``) for the two turn-level cases."""
    rig = Rig(tmp_path, monkeypatch)
    rig.healthy = True
    message = f"boom at {ERR_URL} (net::ERR_FAILED)"

    async def run_turn(*a: Any, **k: Any) -> dict[str, Any]:
        rig.turns.append((str(k["scenario_id"]), int(k["repeat"])))
        if how == "run-turn-raises":
            raise RuntimeError(message)
        if how == "turn-error":
            return _raw(error=message)
        outcome = {
            "step": 1,
            "tool": "click_at",
            "status": "refused",
            "reason": message,
            "error": message,
            "resolved": {},
        }
        return _raw(outcomes=[outcome])

    monkeypatch.setattr(rig.h, "run_turn", run_turn)
    rc = rig.main(rig.argv("--scenarios", "D1"))
    err = capsys.readouterr().err
    assert rc == 0, err[-2000:]
    assert rig.sockets == []
    texts = {
        name: (rig.out / name).read_text(encoding="utf-8")
        for name in ("records.jsonl", "report.json", "report.md")
    }
    texts["stderr"] = err
    for name, text in texts.items():
        assert PATH_TOKEN not in text, f"T1: {name} holds the URL's path or query"
    if how != "outcome-error":
        assert "boom" in texts["records.jsonl"], "premise: the error is recorded at all"


# --- T3: the harness builds the executor with the cli's keywords ----------------------------------------

#: keywords that differ by design: the clock (the cli's reads the session clock, the harness's the turn's)
DRIFT_EXEMPT = frozenset({"now_ms"})


def test_make_executor_keywords_match_the_cli_live_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3 (kills ``_make_executor(web=False)``; the drift in general): ``BrowserToolExecutor`` gets the same
    keywords from ``_make_executor`` as from ``cli._start_live`` for ``session --tools live --browser headless``,
    with the same start origin (so ``allowed_origins`` and ``start_origin`` compare too); only ``now_ms`` (a clock)
    is exempt. Nothing is constructed or bound: the executor class and the start-page server are captured."""
    import argparse

    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.config import load
    from local_astra.pages import server as pages_server

    built: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def capture(*a: Any, **k: Any) -> Any:
        built.append((a, dict(k)))
        return SimpleNamespace(web=k.get("web"))

    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", capture)
    monkeypatch.setattr(pages_server, "StartPageServer", _FakePages)
    profile = load(REPO / "astra.toml").profiles["lean"]
    args = argparse.Namespace(tools="live", browser="headless")
    with t31.no_sockets() as attempts:
        cli._start_live(profile, args, SimpleNamespace(now_ms=0))
        api("_make_executor")(profile, now_ms=lambda: 0, start_origin=FAKE_ORIGIN)
    assert attempts == []
    assert len(built) == 2, built
    (cli_a, cli_k), (h_a, h_k) = built
    assert cli_a == h_a == ()
    assert cli_k.get("web") is True, (
        f"premise: the cli's live start is web mode ({cli_k.get('web')})"
    )
    keys = (set(cli_k) | set(h_k)) - DRIFT_EXEMPT
    diff = sorted(k for k in keys if k not in cli_k or k not in h_k or cli_k[k] != h_k[k])
    assert not diff, (
        f"T3: the harness's executor differs from the cli's in {[(k, cli_k.get(k), h_k.get(k)) for k in diff]}"
    )
