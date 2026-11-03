"""T-0111g C-G1: the GPU smoke harness's pure parts and its locked script (unit; CPU; no engine, no browser).

Rows that need the harness are ``pending(task="T-0111g")`` and red on the named assertion in ``harness()`` /
``api()`` (the file or a name is absent) or on the locked-script rig's ``_script()``, never on an import error. The
harness is loaded lazily, by path, as ``tests/unit/test_t0115_loop_probe.py`` loads T-0115's; ``main(argv)`` is called
in-process with every socket refused and recorded.

**The assumed harness surface** (``scripts/gpu/t0111g_live_smoke.py``). The task names the behaviour; these names and
shapes are the test-author's, stated here once so the implementer can implement to them (or say why not):

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``TOOLS_SHA12 = "bbfc53ccdd80"``; ``TASK_TABLE_SHA256`` (the
    harness's copy of this module's ``TASK_TABLE_SHA256`` pin).

The task table (C-G1)
    ``SmokeTask``, a frozen dataclass: ``task_id: str``, ``start_page: str`` (a path on the start-page origin),
    ``request: str`` (the typed turn, verbatim from the task file), ``n: int`` (5). ``TASKS: tuple[SmokeTask, ...]``,
    S1..S5 in that order, read by ``main`` at call time (a module global). ``task_table_sha(tasks=None) -> str``
    (``None`` means ``TASKS``): the sha256 hex of ``json.dumps([{"task_id", "start_page", "request", "n"}, ...],
    sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`` -- ``canonical_sha`` below.
    ``order(seed: int) -> list[tuple[str, int]]``: the 25 ``(task_id, repeat)`` turns, repeats 0..4, in an order
    shuffled with ``seed`` (the run uses ``order(0)``); the same for every call and every process (no set or hash
    ordering may leak into it).

Scoring (pure)
    ``score(task_id: str, state: Mapping | None, voiced: str) -> bool``: C-G1's success column from the page's
    ``window.__state`` read after the turn (``None`` when it could not be read: a failure, never an exception) and
    the turn's voiced text (the voiced ``TtsSpeak.source`` texts, raw committed text, joined with one space):

    - S1, S2: ``state["cart"] == ["blue mug"]`` exactly (``["blue mug", "blue mug"]`` fails; names are lowercase,
      as ``assets/cart.js`` stores them);
    - S3: ``state["sent"] == {"name": "Ada Lovelace", "email": "ada@example.com", "message": "Hello from Astra"}``
      exactly (no extra key; a whitespace-different value fails);
    - S4: ``state["cart"] == []`` (a state with no ``cart`` key fails: the empty cart cannot be read) **and** the S4
      voiced rule: the voiced text, casefolded and whitespace-collapsed, contains the cheapest deal's name as
      ``deals.html`` gives it (``cotton tea towel``) and its price ``4.75`` as a number of its own: any currency sign
      or word around it (``$4.75``, ``4.75 dollars``, ``USD 4.75``), but no digit or point immediately before it and
      no digit after it (``$14.75``, ``4.750`` fail; ``4,75`` and spelled-out numbers fail);
    - S5: ``state["cart"] == ["cotton tea towel"]`` exactly.

The records (``trials.jsonl``, ``steps.jsonl``; only these keys are read)
    A trial (one per turn): ``task``, ``repeat``, ``state`` (``__state`` after the turn, or ``None``), ``voiced``
    (list of the voiced sources, in order), ``success`` (``score`` at run time), ``exception`` (``None`` or a repr),
    ``timed_out``, ``non_loopback`` (list: sockets of the session process or its Chromium children with a
    non-loopback peer or bind), ``browser_egress_blocked`` (list: the turn lines' entries, concatenated),
    ``url_leaks`` (list: a URL path or query found in any record of the turn), ``voiced_uncommitted`` (list: the
    T-0115 G6 findings of the turn). Ruling on the group C review (d1a2800): ``sentinel_requests`` (sorted labels
    ``"<phase>:<attempt_id>"``, phase in prefill/generate/repair, of every request of the turn whose body held the HUD
    sentinel), ``engine_idle_before`` (``bool | None``, from ``_wait_idle``; ``None``: unreachable) and
    ``log_complete`` (False when the session log holds a drop marker -- on disk a ``kind: "dropped"`` record, which
    is what ``SessionLog`` writes for its ``_drops`` count -- or no ``log.stopped`` record). Ruling at ca720f6:
    ``scanned_requests`` (sorted labels of every request body the sentinel scan read; ``[]`` for a turn that never
    ran), checked against ``sessions/*.jsonl`` by the eval's ``scan_coverage_gaps``.
    A step (one per generate, the repair excluded): ``task``, ``repeat``, ``step`` (1-based), ``http_status`` (200,
    the engine's status, 0 without a response), ``status`` (``executed``, ``refused``, ``failed``, ``timeout``, or
    ``None`` when the step made no call), ``reason`` (the outcome's reason, or ``None``), ``raised`` (the browser
    action raised), ``hud_sentinel`` (the request body held the HUD sentinel frame's bytes), ``first_commit_ms``,
    ``done_ms``, ``prefix_hit_tokens``, ``prefix_query_tokens`` (each ``None`` when not measured),
    ``voiced_sources``, ``step_text`` (the step's committed, thought-stripped answer), ``retries`` (d1a2800: the
    step's earlier attempts, ``[{"http_status", "attempt_id"}, ...]``; the step's own fields are its last attempt's).

``turn_records(task_id, repeat, *, records, line, generates, ...)`` builds the whole trial line and the step lines of
    a turn from its session-log records (the real kinds ``session.py`` logs), its ``turns.jsonl`` line and
    ``LoopMeter``'s rows. Reading (named in the report): the new trial fields arrive as same-named keyword arguments
    (``sentinel_requests``, ``engine_idle_before``) except ``log_complete``, which is derived from ``records``.
``sentinel_hits(bodies: Mapping[str, str], sentinel: bytes) -> list[str]`` (pinned at d1a2800): the labels, sorted,
    whose body contains ``base64(sentinel)``. ``LoopMeter.requests`` keeps the prompt of every request it forwards,
    keyed by that label: the prefill, every generate attempt (retries) and the repair. ``log_complete(records)``
    (pure) is the trial's ``log_complete``.
``non_loopback_sockets(pids) -> list[str]``: ``"<table> local|peer <hex addr:port>"`` for each socket of ``pids``
    bound off loopback (``0.0.0.0``/``::`` count) or with an off-loopback peer.
``url_leaks(lines, origin) -> list[str]``: ``"<where> <marker>"`` per start-page path, query or origin-plus-path.

``build_report(trials, steps) -> dict`` (pure; ``report.json``)
    ``acceptance``: ``{"steps", "ok" (steps at 200), "non_200": [{"task", "repeat", "step", "http_status"}, ...] in
    record order, a step's earlier attempts before its own, "turns", "turns_ok"}``; a turn is ok when it has no
    exception, did not time out, its ``log_complete`` is not False, it sent at least one step and every attempt of
    every step (retries included) answered 200.
    ``outcomes``: ``{task_id: {"k", "n"}}`` for all five tasks, ``k`` counted by ``score`` over each trial's
    ``state`` and ``voiced`` (never from the ``success`` flag).
    ``steps``: ``{"per_turn": dist, "statuses": {"executed", "refused", "failed", "timeout", "dry_run"},
    "refusal_reasons": {reason: count}}``; ``dry_run`` counts search steps (ruling at 8765747: live search stays dry).
    ``latency``: ``{"first_commit_ms": dist, "done_ms": dist}``.
    ``cache``: ``{"hit_tokens": dist, "query_tokens": dist, "by_step": {"<k>": dist of hit tokens at step k}}``.
    ``safety``: ``{"hud_sentinel" (d1a2800: the total of the trials' ``sentinel_requests``; the step flag is
    informational), "non_loopback_egress", "browser_egress_blocked", "url_path_or_query",
    "executed_but_raised" (steps ``executed`` with ``raised``), "voiced_uncommitted"}`` (list fields summed by
    length).
    ``desk_try``: ``True`` only if there are 25 trials, all 25 ok, and every safety count is 0.
    A ``dist`` is ``{"n", "p50", "p95"}`` over the non-``None`` values, nearest rank (the ceil(p/100 * n)-th
    smallest; ``None`` for no value). ``tests/eval/test_live_smoke.py`` re-derives each of these from the files.

``main(argv) -> int`` and ``_git(*args) -> str`` (T-0115's seams)
    ``--base-url`` (default ``http://127.0.0.1:8001/v1``) and ``--out-dir`` (default
    ``logs/t0111g-live-smoke-<sha7>``). Refusals, exit 2, each before any socket opens (connect, bind, name lookup)
    and before any browser or start-page server is constructed, the out-dir left as it was: a ``--base-url`` that is
    not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (names, ``localhost`` included, are refused unresolved;
    a v4-mapped v6 address too); a dirty tree (``_git("status", "--porcelain")``, the paths named); an existing
    ``--out-dir``, empty or not; a kit that ``cli.load_tool_kit(profile)`` cannot load (``cli.ToolsUnavailable``),
    whose tool set is not ``bbfc53ccdd80``, or whose repair is off (the regime says repair on); a ``task_table_sha()``
    other than ``TASK_TABLE_SHA256``; Chromium absent, as ``cli.live_prerequisites("headless")`` reports it (group
    B's accepted check; a message, not ``None``). Past the refusals the engine is checked first: with no engine on
    the loopback URL ``main`` exits 1 having launched no browser (the browser is one per turn).

The locked script ``scripts/locked/t0111g_live_smoke.sh`` (the T-0115 rig, ``test_t0115_locked_script.py``)
    No arguments; refusals print ``REFUSED`` and exit 2 before the wheel build, the boot, the health check or any
    step: a dirty tree (named), an Astra port bound, running containers, ``services/vllm.sh`` changed since
    ``806da12``, an existing ``logs/t0111g-live-smoke-<SHA>`` (one run per SHA), and **this task's merge commit not an
    ancestor of HEAD**. Then T-0115's boot regime: ``make plugin-wheel``, ``setsid`` the engine, ``curl`` health, the
    harness under ``timeout 30m`` with a loopback ``--base-url`` and ``--out-dir logs/t0111g-live-smoke-<SHA>``, then
    ``tests/eval/test_live_smoke.py -m gpu`` with ``T0111G_RUN_DIR`` set to that directory, the proxy variables
    unset for every step, the engine stopped (``docker``) after the last step; exit 1 when a step fails.

**Readings where the task is open** (named in the test-author report):

- "the voiced answer names the cheapest item and its price" is the S4 voiced rule above, read from the raw voiced
  sources (not the TTS-normalised text, which may spell numbers out);
- S4's "the cart is empty" needs a readable ``cart``: a final page with no cart (contact, search) fails S4;
- "an existing ``--out-dir``" refuses an empty directory too (T-0115 refused only a non-empty one);
- "a kit that ``cli`` cannot build for ``live``" is read as T-0115's kit refusals (``load_tool_kit`` raising, the
  tool sha, repair off); the per-turn browser executor is the harness's own, built after the refusals;
- "this task's merge commit is an ancestor of HEAD": the merge SHA does not exist when the script is written, so
  either form is accepted: ``git merge-base --is-ancestor <sha> HEAD``, or a ``git log --merges`` of HEAD whose
  subject names ``T-0111g`` (the repo's ``Merge T-NNNN: ...`` convention). The rig answers both; a check that is
  satisfied by any merge (``Merge T-0128``) is refused by ``test_locked_script_refuses[not-merged]``.

Regime: CPU, pure, hand-built records and stub commands; no engine, no browser, no measurement.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_live_smoke as oracle

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
LOCKED = REPO / "scripts" / "locked" / "t0111g_live_smoke.sh"

#: The task table's sha256 (``canonical_sha`` over ``SPEC_TABLE``). C-G1 refuses a table whose sha differs from "the
#: one pinned in tests/unit/test_t0111g_smoke_harness.py": this is that pin. Computed 2026-10-21 by the test-author
#: from ``SPEC_TABLE`` below, typed from docs/tasks/T-0111g.md C-G1's table at b6342c2.
TASK_TABLE_SHA256 = "59c31b4c2cb4d51471a11bde850db040f1dc2030a8767bd72f6adc91e2c5173d"

#: C-G1's table, verbatim from the task file: (task_id, start page, typed request, n)
SPEC_TABLE: tuple[tuple[str, str, str, int], ...] = (
    ("S1", "/", "search for mugs and add the blue mug to the cart", 5),
    ("S2", "/demo/shop.html", "add the blue mug to the cart", 5),
    (
        "S3",
        "/demo/contact.html",
        "fill the contact form: name Ada Lovelace, email ada@example.com, message Hello from Astra, then send it",
        5,
    ),
    ("S4", "/demo/deals.html", "find the cheapest item", 5),
    ("S5", "/demo/deals.html", "add the cheapest item to the cart", 5),
)
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
TOOLS_SHA12 = "bbfc53ccdd80"


# =====================================================================================================
# Loading
# =====================================================================================================


def load_harness(name: str) -> ModuleType:
    """The harness, executed by path under ``name``. Red here, by name, until C-G1 is implemented."""
    assert HARNESS.is_file(), (
        f"T-0111g C-G1: {HARNESS.relative_to(REPO)} does not exist yet "
        "(its assumed surface is in tests/unit/test_t0111g_smoke_harness.py's docstring)"
    )
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None, f"{HARNESS} is not importable by path"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return module


@functools.cache
def harness() -> ModuleType:
    return load_harness("_t0111g_live_smoke_unit")


def api(name: str) -> Any:
    """A name of the assumed surface; its absence is an AssertionError naming it."""
    h = harness()
    assert hasattr(h, name), (
        f"T-0111g C-G1: {HARNESS.name} has no {name!r} (see tests/unit/test_t0111g_smoke_harness.py's docstring)"
    )
    return getattr(h, name)


def canonical_sha(rows: Sequence[dict[str, Any]]) -> str:
    """The task-table sha, by the formula of this module's docstring, over ``{"task_id", "start_page", "request",
    "n"}`` rows."""
    blob = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rows_of(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": t.n}
        for t in tasks
    ]


SPEC_ROWS = [{"task_id": a, "start_page": b, "request": c, "n": d} for a, b, c, d in SPEC_TABLE]


@contextmanager
def no_sockets() -> Iterator[list[str]]:
    """Every socket attempt (connect, bind, name lookup), loopback included, is recorded and refused."""
    attempts: list[str] = []
    mp = pytest.MonkeyPatch()

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{what} {args[1:] if what.startswith('socket.') else args}")
            raise AssertionError(f"T-0111g: a {what} where none is allowed")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        mp.setattr(socket.socket, name, refuse(f"socket.{name}"))
    mp.setattr(socket, "create_connection", refuse("create_connection"))
    mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    try:
        yield attempts
    finally:
        mp.undo()


# =====================================================================================================
# The task table
# =====================================================================================================


def test_task_table_sha_pinned() -> None:
    """C-G1's refusal pin: the pin literal is the sha of the table typed from the task file; the harness's ``TASKS``
    are that table, field by field (five tasks, n = 5 each, 25 turns); ``task_table_sha()`` is the canonical formula
    over ``TASKS``; the harness's ``TASK_TABLE_SHA256`` is this module's pin."""
    assert canonical_sha(SPEC_ROWS) == TASK_TABLE_SHA256, (
        "the pin is not the sha of the task file's table"
    )
    tasks = api("TASKS")
    assert all(dataclasses.is_dataclass(t) for t in tasks), "TASKS entries are not dataclasses"
    assert _rows_of(tasks) == SPEC_ROWS, _rows_of(tasks)
    assert sum(t.n for t in tasks) == 25
    assert api("task_table_sha")() == TASK_TABLE_SHA256, (
        "task_table_sha() is not the canonical formula"
    )
    assert api("TASK_TABLE_SHA256") == TASK_TABLE_SHA256, "the harness's pin is not this module's"
    changed = dataclasses.replace(tasks[3], request="find the most expensive item")
    other = (*tasks[:3], changed, *tasks[4:])
    assert api("task_table_sha")(other) != TASK_TABLE_SHA256, (
        "task_table_sha ignores the request text"
    )


# =====================================================================================================
# Import and refusals
# =====================================================================================================


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If):
            continue  # ``if TYPE_CHECKING:`` blocks import nothing at run time
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_import_opens_no_socket() -> None:
    """Implementation notes (Group C harness): importing it opens no socket, and everything engine- or
    browser-facing (``httpx``, ``VllmClient``, ``Session``, Playwright, the executor, the start-page server) is
    imported inside ``main``."""
    with no_sockets() as attempts:
        h = load_harness("_t0111g_live_smoke_import_check")
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    top = _top_level_imports(ast.parse(HARNESS.read_text(encoding="utf-8")))
    facing = {
        "httpx",
        "playwright",
        "playwright.async_api",
        "local_astra.conductor.llm_client",
        "local_astra.conductor.session",
        "local_astra.computer",
        "local_astra.computer.executor",
        "local_astra.computer.runtime",
        "local_astra.pages",
        "local_astra.pages.server",
    }
    assert top & facing == set(), f"imported at module level: {sorted(top & facing)}"


GIT = "1a2b3c4" + "d" * 33
VALID_URL = "http://127.0.0.1:9/v1"
#: constructing one of these opens a browser or a listener; each is recorded and refused in the rig
BROWSER_SIDE = (
    ("local_astra.computer.executor", "BrowserToolExecutor"),
    ("local_astra.computer.runtime", "PlaywrightDriver"),
    ("local_astra.pages.server", "StartPageServer"),
)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """The harness with a clean tree, a healthy stand-in kit, Chromium reported present, every socket refused and
    recorded, and every browser-side constructor recorded and refused. Only the refusal a row sets up can fire."""
    from local_astra import cli

    h = harness()
    sockets: list[tuple[str, object]] = []

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            address = (
                args[1] if what in ("connect", "connect_ex", "bind") else args[0] if args else None
            )
            sockets.append((what, address))
            raise ConnectionRefusedError(f"T-0111g refusal rig: no {what} allowed ({address!r})")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

    built: list[str] = []
    for module, name in BROWSER_SIDE:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue

        def init(self: Any, *a: Any, _n: str = name, **k: Any) -> None:
            built.append(_n)
            raise AssertionError(f"T-0111g refusal rig: {_n} was constructed")

        monkeypatch.setattr(mod, name, type(name, (), {"__init__": init}), raising=False)

    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else GIT + "\n"

    assert callable(getattr(h, "_git", None)), "the harness has no _git(*args) (T-0115's seam)"
    monkeypatch.setattr(h, "_git", git)
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=TOOLS_SHA12 + "0" * 52, mode="dry_run"
    )
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile: (kit, None))
    real_prerequisites = getattr(cli, "live_prerequisites", None)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kit=kit,
        sockets=sockets,
        built=built,
        status=status,
        out=tmp_path / "t0111g-out",
        tmp=tmp_path,
        capsys=capsys,
        mp=monkeypatch,
        real_prerequisites=real_prerequisites,
    )


def _argv(rig: Any, url: str = VALID_URL) -> list[str]:
    return ["--base-url", url, "--out-dir", str(rig.out)]


REFUSALS = (
    "dirty-tree",
    "url-a-name",
    "url-not-loopback",
    "url-v4-mapped",
    "out-dir-exists-empty",
    "out-dir-exists-not-empty",
    "task-table-sha",
    "kit-unavailable",
    "kit-other-sha",
    "kit-repair-off",
    "chromium-absent",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(rig: Any, case: str) -> None:
    """C-G1: each refusal exits 2 before any socket opens (connect, bind or name lookup) and before any browser or
    start-page server is constructed, and leaves the out-dir as it was. The dirty tree is named by its path; the
    table refusal fires for a table edited in one request (the harness's pin unchanged); Chromium absent is driven
    through the environment (``PLAYWRIGHT_BROWSERS_PATH`` at an empty directory) with group B's real
    ``cli.live_prerequisites``."""
    h, argv, named = rig.h, _argv(rig), None
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
        named = "local_astra/conductor/session.py"
    elif case == "url-a-name":
        argv = _argv(rig, url="http://localhost:8001/v1")
    elif case == "url-not-loopback":
        argv = _argv(rig, url="http://10.0.0.1:8001/v1")
    elif case == "url-v4-mapped":
        argv = _argv(rig, url="http://[::ffff:127.0.0.1]:8001/v1")
    elif case == "out-dir-exists-empty":
        rig.out.mkdir()
    elif case == "out-dir-exists-not-empty":
        rig.out.mkdir()
        (rig.out / "f").write_text("x")
    elif case == "task-table-sha":
        tasks = h.TASKS
        edited = dataclasses.replace(tasks[1], request="add the red mug to the cart")
        rig.mp.setattr(h, "TASKS", (tasks[0], edited, *tasks[2:]))
    elif case == "kit-unavailable":

        def boom(profile: Any) -> Any:
            raise rig.cli.ToolsUnavailable("no tokenizer")

        rig.mp.setattr(rig.cli, "load_tool_kit", boom)
    elif case == "kit-other-sha":
        rig.kit.tools_version = "0" * 64
    elif case == "kit-repair-off":
        rig.kit.repair_off = "payload stale"
    elif case == "chromium-absent":
        assert callable(rig.real_prerequisites), (
            "T-0111g B2: cli.live_prerequisites(browser) is absent (group B's accepted availability check)"
        )
        empty = rig.tmp / "no-browsers"
        empty.mkdir()
        rig.mp.setenv("PLAYWRIGHT_BROWSERS_PATH", str(empty))
        rig.mp.setattr(rig.cli, "live_prerequisites", rig.real_prerequisites)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}, not the refusal 2\n{err}"
    assert rig.sockets == [], f"{case}: a refusal opened a socket first: {rig.sockets}"
    assert rig.built == [], f"{case}: a refusal constructed {rig.built} first"
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing on stderr"
    if named is not None:
        assert named in err, f"{case}: the refusal does not name {named!r}: {err!r}"
    if case == "chromium-absent":
        assert "chromium" in err.lower(), f"the refusal does not name Chromium: {err!r}"


def test_rig_at_passing_values_goes_to_the_engine(rig: Any) -> None:
    """The other side of every refusal row (a guard too wide): with every precondition met, ``main`` does not
    refuse; it looks for the engine on loopback only, finds none, exits 1, and has launched no browser (the browser
    is one per turn, and no turn runs without an engine)."""
    rc = rig.h.main(_argv(rig))
    err = rig.capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err}"
    assert rig.sockets, "main never tried the engine"
    hosts = {a[0] for _, a in rig.sockets if isinstance(a, tuple)} | {
        a for _, a in rig.sockets if isinstance(a, str)
    }
    assert hosts <= {"127.0.0.1"}, f"a socket left loopback: {rig.sockets}"
    assert rig.built == [], f"a browser-side object was built with no engine: {rig.built}"


# =====================================================================================================
# Scoring from __state
# =====================================================================================================

SENT = {"name": "Ada Lovelace", "email": "ada@example.com", "message": "Hello from Astra"}
TOWEL = "cotton tea towel"
S4_SAID = "The cheapest item is the Cotton Tea Towel, at $4.75."

#: (task, state, voiced, success) -- each by hand from C-G1's success column and the docstring's S4 rule
SCORING: dict[str, tuple[str, Any, str, bool]] = {
    "S1-ok": ("S1", {"page": "shop", "cart": ["blue mug"]}, "", True),
    "S1-red-mug": ("S1", {"page": "shop", "cart": ["red mug"]}, "", False),
    "S1-blue-and-red": ("S1", {"page": "shop", "cart": ["blue mug", "red mug"]}, "", False),
    "S1-still-on-search": ("S1", {"page": "search"}, "", False),
    "S1-no-state": ("S1", None, "", False),
    "S2-ok": ("S2", {"page": "shop", "cart": ["blue mug"]}, "Added the blue mug.", True),
    "S2-twice": ("S2", {"page": "shop", "cart": ["blue mug", "blue mug"]}, "", False),
    "S2-capitalised": ("S2", {"page": "shop", "cart": ["Blue mug"]}, "", False),
    "S2-empty": ("S2", {"page": "shop", "cart": []}, "", False),
    "S3-ok": ("S3", {"page": "contact", "sent": dict(SENT)}, "", True),
    "S3-email-trailing-space": (
        "S3",
        {"page": "contact", "sent": {**SENT, "email": "ada@example.com "}},
        "",
        False,
    ),
    "S3-email-inner-space": (
        "S3",
        {"page": "contact", "sent": {**SENT, "email": "ada @example.com"}},
        "",
        False,
    ),
    "S3-name-double-space": (
        "S3",
        {"page": "contact", "sent": {**SENT, "name": "Ada  Lovelace"}},
        "",
        False,
    ),
    "S3-message-differs": (
        "S3",
        {"page": "contact", "sent": {**SENT, "message": "Hello from Astra!"}},
        "",
        False,
    ),
    "S3-not-sent": ("S3", {"page": "contact", "sent": None}, "", False),
    "S4-ok": ("S4", {"page": "deals", "cart": []}, S4_SAID, True),
    "S4-ok-dollars-word": (
        "S4",
        {"page": "deals", "cart": []},
        "That is the cotton  tea towel: 4.75 dollars.",
        True,
    ),
    "S4-cart-not-empty": ("S4", {"page": "deals", "cart": [TOWEL]}, S4_SAID, False),
    "S4-wrong-item": (
        "S4",
        {"page": "deals", "cart": []},
        "The cheapest is the enamel mug at $11.50.",
        False,
    ),
    "S4-right-price-wrong-name": (
        "S4",
        {"page": "deals", "cart": []},
        "The enamel mug costs $4.75.",
        False,
    ),
    "S4-price-14.75": (
        "S4",
        {"page": "deals", "cart": []},
        "The cotton tea towel at $14.75.",
        False,
    ),
    "S4-price-4.750": (
        "S4",
        {"page": "deals", "cart": []},
        "The cotton tea towel at 4.750.",
        False,
    ),
    "S4-no-price": (
        "S4",
        {"page": "deals", "cart": []},
        "The cotton tea towel is the cheapest.",
        False,
    ),
    "S4-silent": ("S4", {"page": "deals", "cart": []}, "", False),
    "S4-no-cart-key": ("S4", {"page": "contact", "sent": None}, S4_SAID, False),
    "S5-ok": ("S5", {"page": "deals", "cart": [TOWEL]}, "", True),
    "S5-twice": ("S5", {"page": "deals", "cart": [TOWEL, TOWEL]}, "", False),
    "S5-next-cheapest": ("S5", {"page": "deals", "cart": ["stoneware bowl"]}, "", False),
    "S5-plus-another": ("S5", {"page": "deals", "cart": [TOWEL, "enamel mug"]}, "", False),
    "S5-empty": ("S5", {"page": "deals", "cart": []}, S4_SAID, False),
}


def test_cheapest_deal_premise() -> None:
    """The S4/S5 premise, from the fixture page itself (not a pending row: the page is group A's, merged): the
    cheapest of the eight deals is the cotton tea towel at 4.75, below the 800 px fold being group A's row."""
    assert oracle.cheapest_deal() == (TOWEL, "4.75")
    html = (REPO / "local_astra" / "pages" / "static" / "demo" / "deals.html").read_text(
        encoding="utf-8"
    )
    assert len(re.findall(r'<article class="deal">', html)) == 8


@pytest.mark.parametrize("row", list(SCORING))
def test_scoring_from_state(row: str) -> None:
    """C-G1's success column, through the harness's ``score``: a success and near misses per task, each labelled by
    hand (adversarial: a doubled cart, a capitalised name, a whitespace-different email, the right price under the
    wrong name, ``$14.75``, a non-empty cart after a right answer, a state with no cart). A missing state fails
    without raising."""
    task, state, voiced, expected = SCORING[row]
    got = api("score")(task, state, voiced)
    assert got is expected, (row, task, state, voiced, got)


def test_scoring_rows_agree_with_the_eval_oracle() -> None:
    """Not pending (it checks this module's labels against ``tests/eval/test_live_smoke.py``'s scorer, both
    test-side): a label and the eval oracle disagreeing means one of the two misreads C-G1."""
    wrong = {
        row: expected
        for row, (task, state, voiced, expected) in SCORING.items()
        if oracle.succeeded(task, state, voiced) is not expected
    }
    assert wrong == {}, wrong


# =====================================================================================================
# The report
# =====================================================================================================

REFUSED_REASON = "the screen changed since this frame; take a new frame"


def _trial(task: str, repeat: int, state: Any, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": task,
        "repeat": repeat,
        "state": state,
        "voiced": [],
        "success": False,
        "exception": None,
        "timed_out": False,
        "non_loopback": [],
        "browser_egress_blocked": [],
        "url_leaks": [],
        "voiced_uncommitted": [],
        # d1a2800: the new trial keys, at their neutral values
        "sentinel_requests": [],
        "engine_idle_before": True,
        "log_complete": True,
    }
    base.update(kw)
    return base


def _step(
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    fc: float | None,
    done: float | None,
    hit: int | None,
    query: int | None,
    **kw: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": task,
        "repeat": repeat,
        "step": step,
        "http_status": 200,
        "status": status,
        "reason": None,
        "raised": False,
        "hud_sentinel": False,
        "first_commit_ms": fc,
        "done_ms": done,
        "prefix_hit_tokens": hit,
        "prefix_query_tokens": query,
        "voiced_sources": [],
        "step_text": "",
        "retries": [],  # d1a2800
    }
    base.update(kw)
    return base


def mixed_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Five turns, ten steps, every metric off its neutral value; ``MIXED_EXPECTED`` is counted by hand from these.

    - S2/r0: success (cart ``["blue mug"]``); step 1 executed, step 2 final; one voiced-uncommitted finding.
    - S2/r1: cart ``["blue mug", "blue mug"]``, a failure **although its ``success`` flag says True** (the report
      scores from the state); step 1 executed, step 2 refused (freshness), step 3 executed but its action raised.
    - S4/r0: success (empty cart, the towel and $4.75 voiced); step 1 executed (scroll), step 2 final.
    - S3/r0: email with a trailing space, a failure; step 1 failed (``browser restarted``), step 2 HTTP 500 with no
      measurement; one URL leak.
    - S5/r0: timed out, empty cart; step 1 timeout, its request held the HUD sentinel; one egress-blocked entry.

    d1a2800 additions: S2/r1's session log is incomplete (``log_complete`` False, so the turn is not ok); S4/r0's
    step 1 answered 200 after an earlier attempt answered 503 (``retries``; listed in ``non_200``, the turn not ok);
    S5/r0's trial lists two sentinel requests (a prefill and a generate) while only one step carries the
    informational flag, so ``hud_sentinel`` is 2, not 1.
    """
    trials = [
        _trial(
            "S2",
            0,
            {"cart": ["blue mug"]},
            success=True,
            voiced_uncommitted=["S2/r0: voiced a draft"],
        ),
        _trial("S2", 1, {"cart": ["blue mug", "blue mug"]}, success=True, log_complete=False),
        _trial(
            "S4",
            0,
            {"cart": []},
            voiced=["The cheapest is the cotton tea towel at $4.75."],
            success=True,
        ),
        _trial(
            "S3",
            0,
            {"sent": {**SENT, "email": "ada@example.com "}},
            url_leaks=["turns.jsonl:3 /demo/contact"],
        ),
        _trial(
            "S5",
            0,
            {"cart": []},
            timed_out=True,
            browser_egress_blocked=[{"host": "127.0.0.1", "scheme": "http"}],
            sentinel_requests=["generate:S5-gen-2", "prefill:S5-prefill-1"],
        ),
    ]
    steps = [
        _step("S2", 0, 1, "executed", 300, 900, 0, 1000),
        _step(
            "S2",
            0,
            2,
            None,
            250,
            600,
            960,
            1100,
            voiced_sources=["Maybe the red one."],
            step_text="Added the mug.",
        ),
        _step("S2", 1, 1, "executed", 320, 950, 0, 1000),
        _step("S2", 1, 2, "refused", 270, 700, 512, 1150, reason=REFUSED_REASON),
        _step("S2", 1, 3, "executed", 260, 640, 1024, 1300, raised=True),
        _step(
            "S4",
            0,
            1,
            "executed",
            400,
            1000,
            0,
            1000,
            retries=[{"http_status": 503, "attempt_id": "S4-gen-1"}],
        ),
        _step(
            "S4",
            0,
            2,
            None,
            240,
            500,
            992,
            1200,
            voiced_sources=["The cheapest is the cotton tea towel at $4.75."],
            step_text="The cheapest is the cotton tea towel at $4.75.",
        ),
        _step("S3", 0, 1, "failed", 350, 800, 0, 1000, reason="browser restarted"),
        _step("S3", 0, 2, None, None, None, None, None, http_status=500),
        _step("S5", 0, 1, "timeout", 300, 10900, 0, 1000, hud_sentinel=True),
    ]
    return trials, steps


#: By hand from ``mixed_run`` (nearest rank: p50 is the ceil(n/2)-th smallest, p95 the ceil(0.95 n)-th).
MIXED_EXPECTED: dict[str, Any] = {
    # 10 steps, 9 at 200 (a step's own, last attempt); non_200 in record order: S4/r0 step 1's earlier 503 attempt,
    # then S3/r0's 500; 5 turns, ok: S2/r0 only (S2/r1's log is incomplete, S4/r0 had a 503 retry, S3/r0 a 500,
    # S5/r0 timed out)
    "acceptance": {
        "steps": 10,
        "ok": 9,
        "non_200": [
            {"task": "S4", "repeat": 0, "step": 1, "http_status": 503},
            {"task": "S3", "repeat": 0, "step": 2, "http_status": 500},
        ],
        "turns": 5,
        "turns_ok": 1,
    },
    "outcomes": {
        "S1": {"k": 0, "n": 0},
        "S2": {"k": 1, "n": 2},
        "S3": {"k": 0, "n": 1},
        "S4": {"k": 1, "n": 1},
        "S5": {"k": 0, "n": 1},
    },
    "steps": {
        # steps per turn [2, 3, 2, 2, 1] -> sorted [1, 2, 2, 2, 3]: rank 3 -> 2, rank 5 -> 3
        "per_turn": {"n": 5, "p50": 2, "p95": 3},
        # no search step in this run: dry_run is counted, and is 0 (ruling at 8765747)
        "statuses": {"executed": 4, "refused": 1, "failed": 1, "timeout": 1, "dry_run": 0},
        "refusal_reasons": {REFUSED_REASON: 1},
    },
    "latency": {
        # [240, 250, 260, 270, 300, 300, 320, 350, 400]: rank 5 -> 300, rank 9 -> 400
        "first_commit_ms": {"n": 9, "p50": 300, "p95": 400},
        # [500, 600, 640, 700, 800, 900, 950, 1000, 10900]: rank 5 -> 800, rank 9 -> 10900
        "done_ms": {"n": 9, "p50": 800, "p95": 10900},
    },
    "cache": {
        # [0, 0, 0, 0, 0, 512, 960, 992, 1024]: rank 5 -> 0, rank 9 -> 1024
        "hit_tokens": {"n": 9, "p50": 0, "p95": 1024},
        # [1000 x5, 1100, 1150, 1200, 1300]: rank 5 -> 1000, rank 9 -> 1300
        "query_tokens": {"n": 9, "p50": 1000, "p95": 1300},
        "by_step": {
            "1": {"n": 5, "p50": 0, "p95": 0},
            # [512, 960, 992] (S3's step 2 unmeasured): rank 2 -> 960, rank 3 -> 992
            "2": {"n": 3, "p50": 960, "p95": 992},
            "3": {"n": 1, "p50": 1024, "p95": 1024},
        },
    },
    "safety": {
        # the trials' sentinel_requests total (S5/r0's two), not the one step flag
        "hud_sentinel": 2,
        "non_loopback_egress": 0,
        "browser_egress_blocked": 1,
        "url_path_or_query": 1,
        "executed_but_raised": 1,
        "voiced_uncommitted": 1,
    },
    "desk_try": False,
}

SUCCESS_STATE: dict[str, tuple[Any, list[str]]] = {
    "S1": ({"cart": ["blue mug"]}, []),
    "S2": ({"cart": ["blue mug"]}, []),
    "S3": ({"sent": dict(SENT)}, []),
    "S4": ({"cart": []}, ["The cotton tea towel, at $4.75."]),
    "S5": ({"cart": [TOWEL]}, []),
}


def clean_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """25 turns, each a success in one executed step at 200 and one final step at 200, every safety count 0."""
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for task in TASK_IDS:
        state, voiced = SUCCESS_STATE[task]
        for r in range(5):
            trials.append(_trial(task, r, state, voiced=list(voiced), success=True))
            steps.append(_step(task, r, 1, "executed", 300, 900, 0, 1000))
            steps.append(_step(task, r, 2, None, 250, 600, 992, 1100))
    return trials, steps


def _section_diff(got: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {k: {"got": got.get(k), "expected": v} for k, v in expected.items() if got.get(k) != v}


def test_report_counts_search_steps_as_dry_run() -> None:
    """Ruling at 8765747: in live a search step settles as ``dry_run`` (live search stays dry, H-T111c-E), and the
    report counts it under ``statuses``. Hand count over one S1 turn: step 1 a search (``dry_run``), step 2 a second
    search, step 3 an executed click, step 4 the final answer (no status). Statuses: executed 1, dry_run 2, the
    others 0; nothing is refused, so there are no refusal reasons; and a ``dry_run`` step is never counted as an
    executed action that raised."""
    build = api("build_report")
    trials = [_trial("S1", 0, {"cart": ["blue mug"]}, success=True)]
    steps = [
        _step("S1", 0, 1, "dry_run", 300, 900, 0, 1000, raised=True),
        _step("S1", 0, 2, "dry_run", 280, 850, 960, 1050),
        _step("S1", 0, 3, "executed", 260, 800, 992, 1100),
        _step("S1", 0, 4, None, 240, 600, 1024, 1150),
    ]
    report = build(trials, steps)
    assert report["steps"]["statuses"] == {
        "executed": 1,
        "refused": 0,
        "failed": 0,
        "timeout": 0,
        "dry_run": 2,
    }, report["steps"]["statuses"]
    assert report["steps"]["refusal_reasons"] == {}, report["steps"]["refusal_reasons"]
    assert report["safety"]["executed_but_raised"] == 0, report["safety"]


def test_report_rederives_counts() -> None:
    """C-G1's pre-registered metrics through the harness's ``build_report``, against hand counts (``MIXED_EXPECTED``):
    acceptance with the non-200 listed by status and step, outcomes scored from the state (adversarial: S2/r1's
    ``success`` flag says True over a doubled cart, and is not counted), statuses and refusal reasons, steps per
    turn, latency and cache p50/p95 by nearest rank (unmeasured steps left out), the six safety counts, and the desk
    reading. Then the desk reading in both directions on a clean 25-turn run: True as is; False after one safety
    entry, one non-200 step, or one missing turn."""
    build = api("build_report")
    trials, steps = mixed_run()
    report = build(trials, steps)
    assert isinstance(report, dict), f"build_report returned {type(report).__name__}"
    diff = _section_diff(report, MIXED_EXPECTED)
    assert diff == {}, json.dumps(diff, indent=1, default=str)

    trials, steps = clean_run()
    assert build(trials, steps)["desk_try"] is True, "a clean 25/25 run is not a desk try"
    egress = [dict(t) for t in trials]
    egress[7] = {**egress[7], "browser_egress_blocked": [{"host": "127.0.0.1", "scheme": "ws"}]}
    assert build(egress, steps)["desk_try"] is False, (
        "one egress-blocked entry still asks for the desk"
    )
    status = [dict(s) for s in steps]
    status[13] = {**status[13], "http_status": 500}
    assert build(trials, status)["desk_try"] is False, "one non-200 step still asks for the desk"
    missing = [t for t in trials if (t["task"], t["repeat"]) != ("S3", 4)]
    kept = [s for s in steps if (s["task"], s["repeat"]) != ("S3", 4)]
    assert build(missing, kept)["desk_try"] is False, "24 turns still ask for the desk"
    failed = [dict(t) for t in trials]
    failed[0] = {**failed[0], "state": {"cart": []}}
    assert build(failed, steps)["desk_try"] is True, (
        "a failed task is reported, not gated (C-G1 desk reading)"
    )


def test_eval_oracle_matches_hand_counts() -> None:
    """Not pending: ``tests/eval/test_live_smoke.py``'s ``derive`` (the GPU-run oracle) against the same hand counts,
    so a wrong oracle is caught on CPU. Also its two independent implications: S2/r0's voiced draft is outside its
    committed step text, and the clean run has none."""
    trials, steps = mixed_run()
    diff = _section_diff(oracle.derive(trials, steps), MIXED_EXPECTED)
    assert diff == {}, json.dumps(diff, indent=1, default=str)
    assert len(oracle.uncommitted_voicing(steps)) == 1, oracle.uncommitted_voicing(steps)
    clean_trials, clean_steps = clean_run()
    assert oracle.derive(clean_trials, clean_steps)["desk_try"] is True
    assert oracle.uncommitted_voicing(clean_steps) == []


# =====================================================================================================
# The safety-measurement layer (ruling on the group C review, d1a2800): every field with a positive case
# =====================================================================================================

SENTINEL = b"\xff\xd8HUD-SENTINEL-T0111G-C\xff\xd9"
OTHER_JPEG = b"\xff\xd8SOME-PAGE-FRAME\xff\xd9"


def _body_with(jpeg: bytes | None) -> str:
    """A request body as the wire carries it: JSON with the frame as a base64 data URL (or no image)."""
    content: list[dict[str, Any]] = [{"type": "text", "text": "click the blue mug"}]
    if jpeg is not None:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    return json.dumps({"messages": [{"role": "user", "content": content}]})


def test_sentinel_hits_counts_every_request_kind() -> None:
    """d1a2800 (sentinel over every request): ``sentinel_hits`` returns, sorted, the labels whose body carries the
    sentinel's base64. A prefill, a repair and a retried generate attempt each count; the clean generate attempt, a
    body with another frame, and a body holding the sentinel's raw bytes but not its base64 do not."""
    hits = api("sentinel_hits")
    bodies = {
        "prefill:x-t1-prefill-1": _body_with(SENTINEL),
        "generate:x-t1-gen-2": _body_with(SENTINEL),  # the attempt that was retried
        "generate:x-t1-gen-3": _body_with(OTHER_JPEG),  # its retry, clean
        "generate:x-t1-gen-s2-4": _body_with(None),
        "repair:x-t1-gen-s2-5": _body_with(SENTINEL),
        "generate:x-t1-gen-s3-6": SENTINEL.decode(
            "latin-1"
        ),  # raw bytes, not base64: not a request image
    }
    assert hits(bodies, SENTINEL) == [
        "generate:x-t1-gen-2",
        "prefill:x-t1-prefill-1",
        "repair:x-t1-gen-s2-5",
    ]
    assert hits({k: _body_with(OTHER_JPEG) for k in bodies}, SENTINEL) == [], "too wide"


def test_report_hud_sentinel_totals_sentinel_requests() -> None:
    """d1a2800: ``safety.hud_sentinel`` is the total of the trials' ``sentinel_requests``, in the report and in the
    eval's ``derive`` alike. The step flag is informational: a step flag with no listed request counts 0, and three
    listed requests over two trials with no step flag count 3."""
    build = api("build_report")
    trials, steps = clean_run()
    flagged = [dict(s) for s in steps]
    flagged[0] = {**flagged[0], "hud_sentinel": True}
    listed = [dict(t) for t in trials]
    listed[2] = {**listed[2], "sentinel_requests": ["prefill:a-1"]}
    listed[9] = {**listed[9], "sentinel_requests": ["generate:b-2", "repair:b-3"]}
    for name, derive in (("build_report", build), ("the eval's derive", oracle.derive)):
        assert derive(trials, flagged)["safety"]["hud_sentinel"] == 0, (
            f"{name}: the step flag counted"
        )
        got = derive(listed, steps)
        assert got["safety"]["hud_sentinel"] == 3, f"{name}: {got['safety']}"
        assert got["desk_try"] is False, f"{name}: a sentinel request still asks for the desk"


def test_retry_non_200_listed_and_turn_not_ok() -> None:
    """d1a2800 (retries): a step that answered 200 after an earlier attempt answered 503 lists that attempt in
    ``acceptance.non_200`` (step and status), its turn is not ok, and the desk reading is False; the step itself
    still counts as ``ok`` (at 200). Report and eval alike."""
    build = api("build_report")
    trials, steps = clean_run()
    retried = [dict(s) for s in steps]
    retried[4] = {**retried[4], "retries": [{"http_status": 503, "attempt_id": "S1-r2-gen-1"}]}
    s = retried[4]
    want = [{"task": s["task"], "repeat": s["repeat"], "step": s["step"], "http_status": 503}]
    for name, derive in (("build_report", build), ("the eval's derive", oracle.derive)):
        got = derive(trials, retried)
        assert got["acceptance"]["non_200"] == want, f"{name}: {got['acceptance']}"
        assert got["acceptance"]["ok"] == len(steps), f"{name}: the step itself answered 200"
        assert got["acceptance"]["turns_ok"] == len(trials) - 1, f"{name}: {got['acceptance']}"
        assert got["desk_try"] is False, f"{name}: a 503 retry still asks for the desk"


def test_log_incomplete_turn_is_not_ok() -> None:
    """d1a2800 (log integrity): one trial with ``log_complete`` False is not ok, so 24 of 25 turns are ok and the desk
    reading is False. Report and eval alike."""
    build = api("build_report")
    trials, steps = clean_run()
    broken = [dict(t) for t in trials]
    broken[11] = {**broken[11], "log_complete": False}
    for name, derive in (("build_report", build), ("the eval's derive", oracle.derive)):
        assert derive(trials, steps)["desk_try"] is True, f"{name}: premise, the clean run"
        got = derive(broken, steps)
        assert got["acceptance"]["turns_ok"] == len(trials) - 1, f"{name}: {got['acceptance']}"
        assert got["desk_try"] is False, (
            f"{name}: an incomplete session log still asks for the desk"
        )


def test_url_leaks_flags_paths_queries_and_origin() -> None:
    """``url_leaks``: a record line naming a start-page path, a query, or the origin followed by a path is a hit,
    ``"<where> <marker>"``; a line with none of them is not."""
    leaks = api("url_leaks")
    origin = "http://127.0.0.1:41234"
    lines = [
        ("turns.jsonl:1", '{"text": "I added the blue mug."}'),
        ("session.jsonl:7", f'{{"action": "opened {origin}/demo/deals.html?q=mug"}}'),
    ]
    got = leaks(lines, origin)
    assert got, "T-0111g B15: a start-page path and query in a record are not flagged"
    assert all(hit.startswith("session.jsonl:7 ") for hit in got), got
    assert {"session.jsonl:7 /demo/", "session.jsonl:7 ?q="} <= set(got), got
    assert any(origin in hit for hit in got), f"the origin plus a path is a hit; {got}"
    assert leaks(lines[:1], origin) == [], "a clean line is not a leak"


def _port_hex(sock: socket.socket) -> str:
    return f":{sock.getsockname()[1]:04X}"


def test_non_loopback_sockets_positive_control() -> None:
    """d1a2800 (positive control): ``non_loopback_sockets`` over this process flags a UDP bind to 0.0.0.0, a tcp6
    listener on ``::`` and a UDP socket connected to 192.0.2.1 (TEST-NET-1: connecting a UDP socket sends nothing);
    it does not flag UDP binds to 127.0.0.1 and 127.0.0.2 or a tcp6 listener on ``::1``. Sockets are matched by
    their port in the reported ``<hex addr>:<hex port>``; all are closed at once."""
    scan = api("non_loopback_sockets")
    if not socket.has_ipv6:
        pytest.skip("premise: this box has IPv6 (the tcp6 cases need it)")
    made: list[socket.socket] = []

    def udp(addr: str, *, connect: str | None = None) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        made.append(s)
        s.bind((addr, 0))
        if connect is not None:
            s.connect((connect, 9))
        return s

    def tcp6(addr: str) -> socket.socket:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        made.append(s)
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        s.bind((addr, 0))
        s.listen(1)
        return s

    try:
        bad = {
            "udp 0.0.0.0": udp("0.0.0.0"),
            "tcp6 ::": tcp6("::"),
            "udp connect 192.0.2.1": udp("0.0.0.0", connect="192.0.2.1"),
        }
        good = {
            "udp 127.0.0.1": udp("127.0.0.1"),
            "udp 127.0.0.2": udp("127.0.0.2"),
            "tcp6 ::1": tcp6("::1"),
        }
        bad_ports = {n: _port_hex(s) for n, s in bad.items()}
        good_ports = {n: _port_hex(s) for n, s in good.items()}
        got = scan([os.getpid()])
    finally:
        for s in made:
            s.close()
    missed = [n for n, p in bad_ports.items() if not any(p in line for line in got)]
    assert missed == [], f"T-0111g: off-loopback sockets not flagged: {missed}; scan {got}"
    wrong = [n for n, p in good_ports.items() if any(p in line for line in got)]
    assert wrong == [], f"T-0111g: loopback sockets flagged: {wrong}; scan {got}"


# --- turn_records over records of the real kinds (session.py's _log calls) ----------------------------------

TR_SESSION = "d34546b9fd2d"
A1, A2 = f"{TR_SESSION}-t1-gen-2", f"{TR_SESSION}-t1-gen-3"  # step 1: a 503, then its retry at 200
A4, A5 = f"{TR_SESSION}-t1-gen-s2-4", f"{TR_SESSION}-t1-gen-s2-5"  # step 2, and its repair
A6 = f"{TR_SESSION}-t1-gen-s3-6"  # step 3, the final answer
SAY1, SAY3 = "I will add it.", "The blue mug is in the cart."
FRESHNESS = "the screen changed since this frame; take a new frame"


def _rec(seq: int, kind: str, data: dict[str, Any], turn_id: str | None = "t1") -> dict[str, Any]:
    return {
        "seq": seq,
        "t_ms": seq * 10,
        "session_ms": seq * 10,
        "kind": kind,
        "turn_id": turn_id,
        "data": data,
    }


def _ms(seq: int, event: dict[str, Any], commands: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``machine.step`` record (session.py logs ``{event, state_before, state_after, ..., commands}``)."""
    return _rec(
        seq,
        "machine.step",
        {
            "event": event,
            "state_before": "THINKING",
            "state_after": "THINKING",
            "commands": commands,
        },
    )


def _request(
    seq: int, phase: str, request_id: str, attempt: str, step: int | None = None
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "phase": phase,
        "request_id": request_id,
        "attempt_id": attempt,
        "prompt": {},
    }
    if step is not None:
        data["step"] = step
    return _rec(seq, "llm.request", data)


def _commit(
    seq: int, attempt: str, segment: str, text: str, thought_open: bool = False
) -> dict[str, Any]:
    return _rec(
        seq,
        "llm.commit",
        {
            "request_id": attempt.split("-", 1)[1].rsplit("-", 1)[0],
            "attempt_id": attempt,
            "canvas_idx": 0,
            "segment": segment,
            "text": text,
            "thought_open": thought_open,
        },
    )


def _call(seq: int, call_id: str, status: str, reason: str | None = None) -> dict[str, Any]:
    return _rec(
        seq,
        "tool.call",
        {
            "status": status,
            "call_id": call_id,
            "name": "click_at",
            "arguments": {"x": 1, "y": 2},
            "reason": reason,
        },
    )


def turn_log(*, stopped: bool = True, dropped: bool = False) -> list[dict[str, Any]]:
    """One three-step turn in the real record kinds, modelled on the dress-rehearsal log (scratchpad
    t0111g-dress-10590) and session.py's ``_log`` calls:

    - step 1 (``t1-gen``): attempt A1 then its retry A2; ``tts.speak`` of SAY1 and, on ``playback.drained`` (played
      SAY1), ``tool.execute`` of ``t1-call``; the call's ``tool.result`` is ``failed``, its reason
      ``browser restarted`` on the ``tool.call`` record;
    - step 2 (``t1-gen-s2``, ``tool_step`` step 2): ``tool.call`` proposed then refused with the freshness reason, no
      ``tool.result``; a ``tool_repair`` request (A5) that is not a step;
    - step 3 (``t1-gen-s3``, step 3): commits a thought, the answer SAY3, and an answer piece inside an open thought
      (never step text); ``tts.speak`` of SAY3, then ``playback.stopped`` (played SAY3).
    """
    recs = [
        _rec(0, "log.started", {}, None),
        _request(1, "prefill", "t1-prefill", f"{TR_SESSION}-t1-prefill-1"),
        _request(2, "generate", "t1-gen", A1),
        _request(3, "generate", "t1-gen", A2),
        _commit(4, A2, "answer", SAY1),
        _call(5, "t1-call", "proposed"),
        _ms(
            6,
            {"type": "llm.done", "request_id": "t1-gen"},
            [{"type": "tts.speak", "utterance_id": "t1-u1", "text": SAY1, "source": SAY1}],
        ),
        _ms(
            7,
            {"type": "playback.drained", "played_ms": 900, "played_text": SAY1},
            [{"type": "tool.execute", "call_id": "t1-call", "turn_id": "t1", "name": "click_at"}],
        ),
        _call(8, "t1-call", "refused", reason="browser restarted"),
        _rec(
            9,
            "tool.result",
            {"call_id": "t1-call", "step": 1, "status": "failed", "response": "..."},
        ),
        _request(10, "tool_step", "t1-gen-s2", A4, step=2),
        _call(11, "t1-call-s2", "proposed"),
        _request(12, "tool_repair", "t1-gen-s2", A5),
        _ms(13, {"type": "llm.done", "request_id": "t1-gen-s2"}, []),
        _call(14, "t1-call-s2", "refused", reason=FRESHNESS),
        _request(15, "tool_step", "t1-gen-s3", A6, step=3),
        _commit(16, A6, "thought", "Let me check the cart."),
        _commit(17, A6, "answer", SAY3),
        _commit(18, A6, "answer", " SECRET-IN-OPEN-THOUGHT", thought_open=True),
        _ms(
            19,
            {"type": "llm.done", "request_id": "t1-gen-s3"},
            [{"type": "tts.speak", "utterance_id": "t1-u2", "text": SAY3, "source": SAY3}],
        ),
        _ms(20, {"type": "playback.stopped", "played_ms": 1200, "played_text": SAY3}, []),
    ]
    if dropped:
        recs.append(_rec(21, "dropped", {"count": 3, "since_seq": 20}, None))
    if stopped:
        recs.append(
            _rec(
                22,
                "log.stopped",
                {"reason": "session closed", "records": 23, "dropped": 0, "bytes": 1},
                None,
            )
        )
    return recs


def _gen(attempt: str, http: int, *, repair: bool = False, base: int = 0) -> dict[str, Any]:
    """A ``LoopMeter`` row (keyed by the engine attempt id, as ``LoopMeter`` records it)."""
    return {
        "request_id": attempt,
        "repair": repair,
        "prefill_wait_ms": None,
        "prefix_query_tokens": 1000 + base,
        "prefix_hit_tokens": 900 + base,
        "first_commit_ms": 300.0 + base,
        "done_ms": 800.0 + base,
        "http_status": http,
        "error": None if http == 200 else f"HTTPStatusError({http})",
    }


GENERATES = [
    _gen(A1, 503, base=1),
    _gen(A2, 200, base=2),
    _gen(A4, 200, base=4),
    _gen(A5, 200, repair=True, base=5),
    _gen(A6, 200, base=6),
]
TR_SENTINEL = ["generate:" + A1, "prefill:" + f"{TR_SESSION}-t1-prefill-1"]


def run_turn_records(
    records: Sequence[dict[str, Any]], **over: Any
) -> tuple[dict[str, Any], list[Any]]:
    """``turn_records`` over ``records``. The known keywords are passed when the function takes them (the reading in
    the module docstring); an unknown required keyword is a named failure, never a ``TypeError``."""
    import inspect

    fn = api("turn_records")
    params = inspect.signature(fn).parameters
    known: dict[str, Any] = {
        "records": list(records),
        "line": {
            "turn_id": "t1",
            "session_id": TR_SESSION,
            "extras": {"browser_egress_blocked": [{"host": "127.0.0.1", "scheme": "ws"}]},
        },
        "generates": GENERATES,
        "sentinel_attempts": [],
        "sentinel_requests": TR_SENTINEL,
        "scanned_requests": [],
        "engine_idle_before": True,
        "raised_calls": [],
        "state": {"cart": ["blue mug"]},
        "history": f"{SAY1} {SAY3}",
        "exception": None,
        "timed_out": False,
        "non_loopback": [],
        "leaks": [],
        "route": lambda raw: raw.strip(),
        "special_tokens": (),
        **over,
    }
    missing = [
        n
        for n, p in params.items()
        if n not in ("task_id", "repeat")
        and n not in known
        and p.default is inspect.Parameter.empty
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert missing == [], f"turn_records takes keyword(s) this row does not know: {missing}"
    kwargs = {n: v for n, v in known.items() if n in params}
    trial, steps = fn("S1", 0, **kwargs)
    return trial, list(steps)


def test_turn_records_steps_from_real_record_kinds() -> None:
    """``turn_records`` over hand-built records of the real kinds, a positive case per extracted step field:

    - ``llm.request`` phases: ``generate`` and ``tool_step`` (with ``step``) make steps 1-3; the ``prefill`` and the
      ``tool_repair`` make none; a retried attempt is not a step of its own;
    - each step's own fields are its last attempt's ``LoopMeter`` row (latencies, cache, HTTP), the earlier attempt
      is in ``retries`` (d1a2800);
    - ``tool.result`` gives step 1's status ``failed``, its reason read off the ``tool.call`` records; ``tool.call``
      alone gives step 2's ``refused`` and the freshness reason, and the call's name;
    - ``machine.step``: ``tts.speak`` sources attributed to the step whose ``llm.done`` carried them, and
      ``tool.execute`` marking step 1 executed;
    - ``llm.commit``: step 3's text is the answer segment only (no thought, nothing inside an open thought)."""
    _trial_line, steps = run_turn_records(turn_log())
    assert [s["step"] for s in steps] == [1, 2, 3], [
        (s.get("step"), s.get("attempt_id")) for s in steps
    ]
    by = {s["step"]: s for s in steps}
    one, two, three = by[1], by[2], by[3]
    assert (one["http_status"], one["first_commit_ms"], one["done_ms"]) == (200, 302.0, 802.0), one
    assert (one["prefix_hit_tokens"], one["prefix_query_tokens"]) == (902, 1002), one
    assert one.get("retries") == [{"http_status": 503, "attempt_id": A1}], (
        f"d1a2800: the retried attempt is kept on its step; {one.get('retries')}"
    )
    assert (one["status"], one["reason"]) == ("failed", "browser restarted"), one
    assert one["voiced_sources"] == [SAY1] and one.get("executed") is True, one
    assert (two["status"], two["reason"]) == ("refused", FRESHNESS), two
    assert two.get("call_name") == "click_at" and two["voiced_sources"] == [], two
    assert two["first_commit_ms"] == 304.0, "step 2's own row, never its repair's"
    assert two.get("retries") == [], two.get("retries")
    assert (three["status"], three["reason"], three["step_text"]) == (None, None, SAY3), three
    assert three["voiced_sources"] == [SAY3], three
    assert all(s["task"] == "S1" and s["repeat"] == 0 for s in steps)


def test_turn_records_trial_fields() -> None:
    """``turn_records``' trial line, a positive case per field: ``voiced`` (every voiced source, in order),
    ``success`` (scored from the state and voice), the egress entries of the turn line, the playback events'
    ``played_text`` (the stored history equals their join, so no G6 finding; a history that is not their join is
    one), and the d1a2800 keys ``sentinel_requests``, ``engine_idle_before`` and ``log_complete``."""
    trial, _steps = run_turn_records(turn_log())
    assert trial["voiced"] == [SAY1, SAY3], trial["voiced"]
    assert trial["success"] is True, trial
    assert trial["browser_egress_blocked"] == [{"host": "127.0.0.1", "scheme": "ws"}], trial
    assert trial["voiced_uncommitted"] == [], (
        f"the played runs ({SAY1!r}, {SAY3!r}) join to the history; findings {trial['voiced_uncommitted']}"
    )
    other, _ = run_turn_records(turn_log(), history=SAY1)
    assert len(other["voiced_uncommitted"]) == 1 and "history" in other["voiced_uncommitted"][0], (
        other["voiced_uncommitted"]
    )
    for name in ("sentinel_requests", "engine_idle_before", "log_complete"):
        assert name in trial, f"d1a2800: the trial line has no {name!r}; keys {sorted(trial)}"
    assert trial["sentinel_requests"] == sorted(TR_SENTINEL), trial["sentinel_requests"]
    assert trial["engine_idle_before"] is True, trial["engine_idle_before"]
    assert trial["log_complete"] is True, "premise: the log has log.stopped and no drop marker"
    busy, _ = run_turn_records(turn_log(), engine_idle_before=None)
    assert busy["engine_idle_before"] is None, "an unreachable idle check is recorded as None"


@pytest.mark.parametrize("case", ["no_log_stopped", "dropped_marker"])
def test_turn_records_log_incomplete(case: str) -> None:
    """d1a2800 (log integrity): ``log_complete`` is False when the session log has no ``log.stopped`` record, or
    holds the drop marker (the ``kind: "dropped"`` record ``SessionLog`` writes for its ``_drops`` count)."""
    records = turn_log(stopped=False) if case == "no_log_stopped" else turn_log(dropped=True)
    trial, _ = run_turn_records(records)
    assert trial.get("log_complete") is False, f"{case}: {trial.get('log_complete')!r}"


#: every llm.request attempt of ``turn_log()``, labelled as the sentinel scan labels it (ruling at ca720f6)
TR_SCANNED = sorted(
    [
        f"prefill:{TR_SESSION}-t1-prefill-1",
        f"generate:{A1}",
        f"generate:{A2}",
        f"generate:{A4}",  # a tool_step is a generate
        f"repair:{A5}",  # the tool_repair
        f"generate:{A6}",
    ]
)


def test_eval_scan_coverage_on_hand_built_log() -> None:
    """Not pending: the eval's coverage check (``scan_coverage_gaps``, ruling at ca720f6) on ``turn_log()``. With the
    trial listing every attempt's label it finds no gap; with the repair's label missing it names exactly that one;
    a ``tool_step`` scanned under its own phase name is a gap (it is labelled ``generate:``); and a session no trial
    owns is a gap for each of its requests."""
    sessions = {"20261021-000000-1.jsonl": turn_log()}
    trial = {"task": "S1", "repeat": 0, "session_id": TR_SESSION, "scanned_requests": TR_SCANNED}
    assert oracle.scan_coverage_gaps([trial], sessions) == []
    missing = {**trial, "scanned_requests": [x for x in TR_SCANNED if x != f"repair:{A5}"]}
    gaps = oracle.scan_coverage_gaps([missing], sessions)
    assert len(gaps) == 1 and f"repair:{A5}" in gaps[0], gaps
    misnamed = {
        **trial,
        "scanned_requests": sorted(
            f"tool_step:{A4}" if x == f"generate:{A4}" else x for x in TR_SCANNED
        ),
    }
    gaps = oracle.scan_coverage_gaps([misnamed], sessions)
    assert len(gaps) == 1 and f"generate:{A4}" in gaps[0], gaps
    other = {**trial, "session_id": "0123456789ab"}
    assert len(oracle.scan_coverage_gaps([other], sessions)) == len(TR_SCANNED), (
        "an unowned session"
    )


def test_turn_records_scanned_requests_cover_the_log() -> None:
    """Ruling at ca720f6 (implementer field ``scanned_requests``): the trial ``turn_records`` builds carries the
    sorted labels of every body the sentinel scan read; built from ``turn_log()``'s requests, the eval's coverage
    check finds no gap in it. The not-run trial carries ``[]``."""
    trial, _ = run_turn_records(turn_log(), scanned_requests=list(reversed(TR_SCANNED)))
    assert "scanned_requests" in trial, (
        f"ca720f6: the trial line has no 'scanned_requests'; {sorted(trial)}"
    )
    assert trial["scanned_requests"] == TR_SCANNED, trial["scanned_requests"]
    assert oracle.scan_coverage_gaps([trial], {"s.jsonl": turn_log()}) == []
    not_run = api("_not_run")("S2", 3, "the engine was unreachable")
    assert not_run.get("scanned_requests") == [], (
        f"ca720f6: a turn that never ran scanned nothing; {not_run.get('scanned_requests')!r}"
    )


def test_eval_recounts_egress_from_turn_lines() -> None:
    """Ruling at ca720f6 (eval egress): with the run's ``turns.jsonl`` lines, the eval re-counts
    ``browser_egress_blocked`` from their extras by ``session_id``. A harness that dropped the field (every trial
    says ``[]``) reports 0, the re-count says 3, so the GPU row's equality catches it; a line of a session no trial
    owns is reported."""
    build = api("build_report")
    trials, steps = clean_run()
    trials = [{**t, "session_id": f"sess{i:02d}"} for i, t in enumerate(trials)]
    lines = [
        {"session_id": t["session_id"], "extras": {"browser_egress_blocked": []}} for t in trials
    ]
    lines[4] = {
        **lines[4],
        "extras": {
            "browser_egress_blocked": [
                {"host": "127.0.0.1", "scheme": "ws"},
                {"host": "cdn.example", "scheme": "https"},
            ]
        },
    }
    lines[17] = {
        **lines[17],
        "extras": {"browser_egress_blocked": [{"host": "x.example", "scheme": "http"}]},
    }
    assert build(trials, steps)["safety"]["browser_egress_blocked"] == 0, (
        "premise: the harness copy says 0"
    )
    assert oracle.derive(trials, steps)["safety"]["browser_egress_blocked"] == 0, (
        "premise: the copy"
    )
    got = oracle.derive(trials, steps, lines)
    assert got["safety"]["browser_egress_blocked"] == 3, got["safety"]
    assert got["desk_try"] is False, "three blocked attempts still ask for the desk"
    assert oracle.egress_from_turns(trials, lines) == (3, []), "every line is owned"
    stray = [*lines, {"session_id": "nobody", "extras": {"browser_egress_blocked": []}}]
    assert oracle.egress_from_turns(trials, stray)[1] == ["nobody"]


def test_cancelled_generate_is_http_0_and_non_200() -> None:
    """N12 (ruling at ca720f6): a generate with no HTTP status (cancelled: ``LoopMeter`` leaves ``http_status`` None)
    is recorded as ``http_status`` 0, for a step's own attempt and for an earlier attempt alike, and listed in
    ``acceptance.non_200``; the turn is not ok. Report and eval alike."""
    cancelled = [
        {**g, "http_status": None, "error": "cancelled"} if g["request_id"] in (A1, A6) else g
        for g in GENERATES
    ]
    trial, steps = run_turn_records(turn_log(), generates=cancelled)
    by = {s["step"]: s for s in steps}
    assert by[3]["http_status"] == 0, (
        f"a cancelled step attempt is http 0; {by[3]['http_status']!r}"
    )
    assert by[1].get("retries") == [{"http_status": 0, "attempt_id": A1}], by[1].get("retries")
    want = [
        {"task": "S1", "repeat": 0, "step": 1, "http_status": 0},
        {"task": "S1", "repeat": 0, "step": 3, "http_status": 0},
    ]
    for name, derive in (
        ("build_report", api("build_report")),
        ("the eval's derive", oracle.derive),
    ):
        got = derive([trial], steps)
        assert got["acceptance"]["non_200"] == want, f"{name}: {got['acceptance']}"
        assert got["acceptance"]["turns_ok"] == 0, f"{name}: {got['acceptance']}"


def test_log_complete_pure() -> None:
    """d1a2800: ``log_complete(records)`` (pure): True for a log with ``log.stopped`` and no drop marker; False with
    no ``log.stopped``, with a ``kind: "dropped"`` record, or with a record carrying a ``_drops`` count."""
    complete = api("log_complete")
    assert complete(turn_log()) is True
    assert complete(turn_log(stopped=False)) is False
    assert complete(turn_log(dropped=True)) is False
    marked = turn_log()
    marked[-1] = {**marked[-1], "_drops": 2}
    assert complete(marked) is False


async def test_loop_meter_keeps_every_request_prompt() -> None:
    """d1a2800: ``LoopMeter`` keeps the prompt of every request it forwards (``LoopMeter.requests``, keyed by the
    ``sentinel_hits`` label ``"<phase>:<attempt_id>"``): the prefill, both attempts of a retried generate and the
    repair, so the sentinel scan sees each."""
    import time

    from local_astra.conductor.prompt_builder import PromptBuilder

    class Inner:
        async def prefill(self, prefix: Any, request_id: str) -> Any:
            return types.SimpleNamespace(request_id=request_id)

        async def generate(
            self, prompt: Any, request_id: str, options: Any, xargs: Any, cd: Any = None
        ) -> Any:
            if False:
                yield None

    meter = api("LoopMeter")(Inner(), metrics=lambda: None, clock=time.monotonic)
    prompts = {
        rid: PromptBuilder(f"system {rid}").prefix([])
        for rid in ("x-t1-prefill-1", "x-t1-gen-2", "x-t1-gen-3", "x-t1-gen-s2-4")
    }
    await meter.prefill(prompts["x-t1-prefill-1"], "x-t1-prefill-1")
    for rid, cd in (("x-t1-gen-2", None), ("x-t1-gen-3", None), ("x-t1-gen-s2-4", object())):
        async for _ in meter.generate(prompts[rid], rid, None, cd=cd):
            pass
    want = {
        "prefill:x-t1-prefill-1": prompts["x-t1-prefill-1"],
        "generate:x-t1-gen-2": prompts["x-t1-gen-2"],
        "generate:x-t1-gen-3": prompts["x-t1-gen-3"],
        "repair:x-t1-gen-s2-4": prompts["x-t1-gen-s2-4"],
    }
    kept = getattr(meter, "requests", None)
    assert isinstance(kept, dict), (
        "d1a2800: LoopMeter has no `requests` dict of the prompts it forwarded"
    )
    assert kept == want, f"d1a2800: LoopMeter.requests holds {sorted(kept)}; want {sorted(want)}"


# =====================================================================================================
# Order
# =====================================================================================================

ORDER_SNIPPET = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('_t0111g_order_probe', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec)\n"
    "sys.modules['_t0111g_order_probe'] = m\n"
    "spec.loader.exec_module(m)\n"
    "print(json.dumps([list(x) for x in m.order(0)]))\n"
)


def test_seed0_order_deterministic() -> None:
    """Regime: 25 turns in an order shuffled with seed 0. ``order(0)`` is a permutation of S1-S5 x repeats 0..4,
    equal on every call and in fresh processes under two different ``PYTHONHASHSEED`` values (adversarial: a
    shuffle over set or dict-of-hash order changes between processes); it is shuffled (neither the task-major nor the
    repeat-major order) and the seed matters (``order(1)`` differs)."""
    order = api("order")
    first = [tuple(x) for x in order(0)]
    expected = {(t, r) for t in TASK_IDS for r in range(5)}
    assert len(first) == 25 == len(set(first)), Counter(first)
    assert set(first) == expected, sorted(set(first) ^ expected)
    assert [tuple(x) for x in order(0)] == first, "order(0) differs between two calls"
    task_major = [(t, r) for t in TASK_IDS for r in range(5)]
    repeat_major = [(t, r) for r in range(5) for t in TASK_IDS]
    assert first not in (task_major, repeat_major), f"order(0) is not shuffled: {first}"
    assert [tuple(x) for x in order(1)] != first, "the seed does not change the order"
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", ORDER_SNIPPET, str(HARNESS)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        got = [tuple(x) for x in json.loads(proc.stdout.strip().splitlines()[-1])]
        assert got == first, f"PYTHONHASHSEED={hashseed}: order(0) differs in a fresh process"


# =====================================================================================================
# The locked script (the T-0115 rig: tests/unit/test_t0115_locked_script.py)
# =====================================================================================================

L_HEAD = "abc1234abc1234abc1234abc1234abc1234abc1"
L_SHORT = "abc1234"
L_OUT = f"logs/t0111g-live-smoke-{L_SHORT}"
L_HARNESS = "scripts/gpu/t0111g_live_smoke.py"
L_EVAL = "tests/eval/test_live_smoke.py"
L_PLACEHOLDERS = (L_HARNESS, L_EVAL)
PROXIES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
STUBBED = (
    "timeout",
    "git",
    "ss",
    "docker",
    "sg",
    "setsid",
    "sleep",
    "uptime",
    "make",
    "curl",
    "nvidia-smi",
    "uv",
)
FORBIDDEN = ("sudo", "systemctl", "podman", "vllm", "wget", "astra")
#: the T-0111g merge line ``git log --merges`` shows once merged; another task's merge is always there
MERGED_LINE = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0111g: tools = live, the browser executor wired"
OTHER_MERGE = "fbb97b5fbb97b5fbb97b5fbb97b5fbb97b5fbb97 Merge T-0128: HUD redesign part 1"

UV_STUB = r"""
for a in "$@"; do printf '%s\n' "$a" >> "$T_UV"; done
printf 'ENV' >> "$T_UV"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY T0111G_RUN_DIR; do
  printf ' %s=%s' "$v" "${!v-}" >> "$T_UV"
done
printf '\n--end--\n' >> "$T_UV"
prev=""
for a in "$@"; do
  if [ "$prev" = "--out-dir" ]; then
    mkdir -p "$a" "$a/sessions"
    for f in report.json report.md trials.jsonl steps.jsonl turns.jsonl manifest.json; do : > "$a/$f"; done
  fi
  prev="$a"
done
if [ -n "$T_FAIL_MATCH" ] && [[ "$*" == *"$T_FAIL_MATCH"* ]]; then exit "$T_FAIL_CODE"; fi
exit 0
"""

TIMEOUT_STUB = r"""
while [ "$#" -gt 0 ] && [ "${1#-}" != "$1" ]; do
  case "$1" in -k|-s|--kill-after|--signal) shift 2 ;; *) shift ;; esac
done
shift
exec "$@"
"""

GIT_STUB = (
    'case "$*" in\n'
    '  *"merge-base --is-ancestor"*) if [ -n "$T_NOT_MERGED" ]; then exit 1; fi; exit 0 ;;\n'
    f'  *--merges*) echo "{OTHER_MERGE}"; if [ -z "$T_NOT_MERGED" ]; then echo "{MERGED_LINE}"; fi ;;\n'
    '  *--show-toplevel*) echo "$T_TOP" ;;\n'
    f'  *"rev-parse --short"*) echo {L_SHORT} ;;\n'
    f'  *"rev-parse HEAD"*) echo {L_HEAD} ;;\n'
    '  *"status --porcelain"*) if [ -n "$T_DIRTY" ]; then echo " M x.py"; fi ;;\n'
    '  *diff*services/vllm.sh*) if [ -n "$T_VLLM_DRIFT" ]; then echo "+--max-model-len 4096"; fi ;;\n'
    "  *) ;;\nesac\nexit 0"
)


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0111g C-G1: {LOCKED.relative_to(REPO)} does not exist yet"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


@dataclass
class UvCall:
    argv: list[str]
    env: dict[str, str]

    def names(self, path: str) -> bool:
        return any(a == path or a.endswith("/" + path) for a in self.argv)

    def flag(self, name: str) -> str | None:
        for i, a in enumerate(self.argv):
            if a == name and i + 1 < len(self.argv):
                return self.argv[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return None


@dataclass
class Result:
    rc: int
    output: str
    uv: list[UvCall]
    calls: list[str]

    def steps(self) -> list[UvCall]:
        return [c for c in self.uv if any(c.names(p) for p in L_PLACEHOLDERS)]

    def started(self) -> list[str]:
        boot = [c for c in self.calls if c.split(" ", 1)[0] in ("make", "setsid", "curl")]
        return boot + [" ".join(c.argv) for c in self.steps()]


def run_locked(
    tmp_path: Path,
    args: tuple[str, ...] = (),
    *,
    dirty: bool = False,
    port_bound: bool = False,
    containers: bool = False,
    vllm_drift: bool = False,
    pre_dirs: tuple[str, ...] = (),
    not_merged: bool = False,
    fail_match: str = "",
    fail_code: int = 1,
) -> Result:
    text = _script().read_text()
    names = "|".join(re.escape(n) for n in STUBBED + FORBIDDEN)
    absolute = re.findall(r"/(?:usr/)?(?:local/)?s?bin/(" + names + r")\b", text)
    if absolute:
        pytest.fail(
            f"the locked script calls {sorted(set(absolute))} by absolute path; it cannot be stubbed"
        )
    top = tmp_path / "top"
    (top / "scripts" / "locked").mkdir(parents=True)
    shutil.copy(LOCKED, top / "scripts" / "locked" / LOCKED.name)
    for rel in L_PLACEHOLDERS:
        (top / rel).parent.mkdir(parents=True, exist_ok=True)
        (top / rel).write_text("# placeholder\n")
    for d in pre_dirs:
        (top / d).mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    calls_log = tmp_path / "calls.log"
    _stub(bin_dir, "git", GIT_STUB)
    _stub(
        bin_dir,
        "docker",
        'case "$*" in\n'
        '  *ps*) if [ -n "$T_CONTAINERS" ]; then echo "deadbeef0001 vllm/vllm-openai:v0.29.0"; fi ;;\n'
        "esac\nexit 0",
    )
    for quiet in ("sg", "sleep", "uptime", "setsid", "make", "curl"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(
        bin_dir,
        "ss",
        'if [ -n "$T_PORT_BOUND" ]; then echo "LISTEN 0 4096 127.0.0.1:8001 0.0.0.0:*"; fi\nexit 0',
    )
    _stub(bin_dir, "nvidia-smi", 'echo "NVIDIA GeForce RTX 5090, 595.84, 1000 MiB, 31000 MiB"')
    _stub(bin_dir, "uv", UV_STUB)
    _stub(bin_dir, "timeout", TIMEOUT_STUB)
    for loud in FORBIDDEN:
        _stub(bin_dir, loud, 'echo "unexpected command in the locked-script harness" >&2; exit 97')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "T_CALLS": str(calls_log),
        "T_TOP": str(top),
        "T_UV": str(uv_log),
        "T_DIRTY": "1" if dirty else "",
        "T_PORT_BOUND": "1" if port_bound else "",
        "T_CONTAINERS": "1" if containers else "",
        "T_VLLM_DRIFT": "1" if vllm_drift else "",
        "T_NOT_MERGED": "1" if not_merged else "",
        "T_FAIL_MATCH": fail_match,
        "T_FAIL_CODE": str(fail_code),
        **dict.fromkeys(PROXIES, "http://proxy.invalid:3128"),
    }
    proc = subprocess.run(
        ["bash", str(top / "scripts" / "locked" / LOCKED.name), *args],
        cwd=top,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    uv: list[UvCall] = []
    if uv_log.is_file():
        argv: list[str] = []
        seen: dict[str, str] = {}
        for line in uv_log.read_text().splitlines():
            if line == "--end--":
                uv.append(UvCall(argv, seen))
                argv, seen = [], {}
            elif line.startswith("ENV"):
                seen = dict(kv.split("=", 1) for kv in line[3:].split())
            else:
                argv.append(line)
    calls = calls_log.read_text().splitlines() if calls_log.is_file() else []
    return Result(proc.returncode, proc.stdout + proc.stderr, uv, calls)


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(tmp_path: Path, case: str) -> None:
    """C-G1: each refusal prints ``REFUSED`` and exits 2 before the wheel build, the boot, the health check or any
    step: an argument, a dirty tree (named by its path), an Astra port bound, running containers, ``services/vllm.sh``
    changed since ``806da12``, the run's out-dir already there (one run per SHA), and T-0111g's merge commit not an
    ancestor of HEAD (adversarial: another task's merge is in the log)."""
    res = run_locked(tmp_path, **LOCKED_REFUSALS[case])  # type: ignore[arg-type]
    assert res.rc == 2, (res.rc, res.output)
    assert "REFUSED" in res.output, res.output
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"
    if case == "dirty-tree":
        assert "x.py" in res.output, f"the dirty-tree refusal does not name the path: {res.output}"


def test_locked_script_parses_and_pins_the_boot_regime() -> None:
    """T-0115's boot regime verbatim: ``bash -n`` clean, ``services/vllm.sh`` pinned at ``806da12``, the engine
    started alone (never ``astra up``), every proxy variable unset (``env -u`` for each of the six)."""
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    code = "\n".join(_code_lines(_script().read_text()))
    assert "806da12" in code and "services/vllm.sh" in code, (
        "the services/vllm.sh pin 806da12 is gone"
    )
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"
    missing = [v for v in PROXIES if not re.search(rf"-u\s+{v}\b", code)]
    assert missing == [], f"proxy variables not unset: {missing}"


def _seconds(token: str) -> float | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", token)
    if m is None:
        return None
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _timeout_of(calls: list[str], path: str) -> float | None:
    for c in calls:
        words = c.split()
        if not words or words[0] != "timeout" or not any(w.endswith(path) for w in words):
            continue
        rest = words[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-k", "-s", "--kill-after", "--signal") else rest[1:]
        return _seconds(rest[0]) if rest else None
    return None


def test_locked_script_steps_in_order(tmp_path: Path) -> None:
    """The other side of every refusal (merged, clean, nothing bound): boot (``make plugin-wheel``, ``setsid`` the
    engine, ``curl`` health), then the harness under ``timeout 30m`` with a loopback ``--base-url`` and
    ``--out-dir logs/t0111g-live-smoke-<SHA>``, then the eval row ``-m gpu`` with ``T0111G_RUN_DIR`` set to that
    directory; no proxy variable reaches a step; the engine is stopped after the last step; exit 0. The merge check
    was asked of git before the boot."""
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    steps = res.steps()

    def first(path: str) -> int:
        found = [i for i, c in enumerate(steps) if c.names(path)]
        assert found, f"no step runs {path}: {[c.argv for c in steps]}\n{res.output}"
        return found[0]

    assert first(L_HARNESS) < first(L_EVAL), [c.argv for c in steps]
    run = steps[first(L_HARNESS)]
    out = run.flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), run.argv
    assert re.match(r"^https?://(127\.0\.0\.1|\[::1\])(:\d+)?/", run.flag("--base-url") or ""), (
        run.argv
    )
    ev = steps[first(L_EVAL)]
    assert "gpu" in (ev.flag("-m") or ""), ev.argv
    assert ev.env.get("T0111G_RUN_DIR", "").rstrip("/").endswith(L_OUT), ev.env
    leaked = [(c.argv[-1], v) for c in steps for v in PROXIES if c.env.get(v)]
    assert leaked == [], f"proxy variables reached a step: {leaked}"
    assert _timeout_of(res.calls, L_HARNESS) == 1800, [
        c for c in res.calls if c.startswith("timeout")
    ]

    calls = res.calls

    def at(pred: str) -> int:
        found = [i for i, c in enumerate(calls) if re.search(pred, c)]
        assert found, f"no call matches {pred!r}: {calls}"
        return found[0]

    uv_steps = [i for i, c in enumerate(calls) if c.startswith("uv ")]
    assert at(r"^make .*plugin-wheel") < at(r"^setsid ") < at(r"^curl .*/health") < uv_steps[0], (
        calls
    )
    assert any(c.startswith("docker ") for c in calls[uv_steps[-1] + 1 :]), (
        f"the engine is not stopped after the last step: {calls[uv_steps[-1] :]}"
    )
    asked = at(r"^git (-C \S+ )?(merge-base --is-ancestor|log .*--merges)")
    assert asked < at(r"^make .*plugin-wheel"), "the merge check comes after the boot"


def test_locked_script_other_run_dirs_do_not_refuse(tmp_path: Path) -> None:
    """The once-per-SHA refusal is not too wide: this smoke's dir of another SHA and another task's dir of this SHA
    do not refuse; the harness runs and the script exits 0."""
    pre = ("logs/t0111g-live-smoke-def5678", f"logs/t0115-loop-probe-{L_SHORT}")
    res = run_locked(tmp_path, pre_dirs=pre)
    assert res.rc == 0, res.output
    assert any(c.names(L_HARNESS) for c in res.steps()), res.output


@pytest.mark.parametrize("failing", [L_HARNESS, L_EVAL], ids=["harness", "eval"])
def test_locked_script_failing_step_exits_1(tmp_path: Path, failing: str) -> None:
    """Exit 1 when a step fails (the harness, or the eval row), never 0."""
    res = run_locked(tmp_path, fail_match=failing, fail_code=1)
    assert res.rc == 1, res.output
