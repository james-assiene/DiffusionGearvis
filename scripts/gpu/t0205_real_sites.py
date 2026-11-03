"""T-0205 group D -- the real-site rehearsal: the demo requests (D1-D6, D8, D9; D7 refused) run on real public
websites in web mode, through the session's own egress proxy, headless, n = 1 (``--repeats 2`` at most). Report-only:
reachability, bot walls, steps, where each run ended, the hosts reached and a screenshot per step. ``/launch`` ONLY,
by James (H-T205-7), through ``scripts/locked/t0205_real_sites.sh``.

    ASTRA_REAL_SITES=1 uv run python scripts/gpu/t0205_real_sites.py --base-url http://127.0.0.1:8001/v1 --real-sites

GPU-spending and network-reaching by definition (scripts/gpu/, T-0016). It never boots an engine. **Importing this
file opens no socket and starts nothing**: the table (``REAL_SCENARIOS``), the classifiers (``final_kind``,
``final_host``, ``reached``), the record (``hosts_of``, ``refusal_counts``, ``search_query_count``, ``turn_record``),
the screenshot paths (``shot_path``), the labels (``scenario_label``), the executor check (``executor_refusal``) and
the live keywords (``session_keywords``) are pure; everything engine-, browser- or network-facing is imported inside
the runners.

Written independently of group C (``t0205_demo_run``) and imports nothing of groups A/B (the loopback scenario
pages and their judge): a real site has no ``__state``, so there is no judge here.

**Network (D2r).** This harness opens no network path of its own. Each turn builds the executor exactly as the
cli's live start does (``_make_executor``: every keyword resolved through ``cli``; web mode, a fresh profile, the
real resolver and connector, nothing injected), so every page byte goes through the session's own T-0160 egress
proxy, which the driver starts at launch with Chromium's sandbox on. Web search goes only through the cli's own
start path (``cli.search_start``, which builds the T-0111b/T-0125/T-0163 backend), once per process; the harness
never constructs a backend. Before any socket, exit 2 (``EXIT_REFUSED``) when: ``--real-sites`` or
``ASTRA_REAL_SITES=1`` is missing (both are required; the harness never sets the variable, the locked wrapper does);
``--repeats`` is outside 1..2; ``--scenarios`` names D7 (H-T205-6: refused unconditionally) or an unknown id;
``--base-url`` is not loopback; the tree is dirty (its paths named); the out-dir is not under ``logs/`` or exists;
Chromium's sandbox check fails; the resolved executor is not in web mode, has an injected ``proxy_resolve`` /
``proxy_connect`` or a profile directory. Then one ``/health`` read; no healthy engine is exit 1.

**Turns (D4r).** Scenarios in table order, then the next repeat; concurrency 1; ``TURN_TIMEOUT_S`` per turn; no turn
starts once ``RUN_BUDGET_S`` is spent (the scenarios not run are ``incomplete``, exit 1). Each turn: a fresh
start-page server (``pages/static/``, as ``astra session`` serves it), a fresh web-mode executor on the live start
page ``/web.html`` (H-T205-5, ruled: the model navigates or searches itself; the table's third column is only the
*expected site*), a fresh ``Session`` with HEAD's live web-mode keywords (``session_keywords``, read from ``cli`` at
call time), typed (one ``UserText``), fake STT/TTS/audio, the cli's repair warm-up first.

**Records (D3r).** Per turn (``records.jsonl``): the hosts reached (the union of the outcomes' ``resolved["hosts"]``
plus ``hosts_more``; reached through the executor watch, never the proxy's internals, Q-2), the proxy refusals as
counts per scheme, the number of web searches, steps, wall time, ``tool_loop_end``, moved refusals, waits,
hand-back, the last executed action, ``final_kind`` and the final host. A URL (path, query, fragment) is never
stored. Every step's frame JPEG goes to ``shots/<id>/<repeat>-<step>.jpg`` (step 0: the frame before the first
action). Nothing is written outside ``logs/``; the sessions' own logs (which hold the search queries, as a live
session's do) go to ``sessions/`` in the out-dir.

**Report** (``report.json``, ``report.md``): per scenario its label (``ran`` / ``incomplete`` / ``refused``), the
expected host and whether it was reached (``None`` for D2), ``final_kind``, hosts, steps, where it ended and the
screenshot paths. No judge, no bar. State the date and local time of the run beside every result: real sites change.

Regime (state it beside every number): profile lean, engine only; vLLM, torch, CUDA, driver, model and chat template
per ``services/versions.lock`` at HEAD; canvas 256, 48 steps, thinking on, prefix caching on, repair on; HEAD's live
web-mode defaults (``manifest.json``); headless Chromium (sandbox on), web mode, search on; typed turns, fake
STT/TTS/audio, no headset or speakers; n = 1 (at most 2); engine unseeded; 300 s per turn.

Exit: 0 every turn ran; 1 a run that stopped early or no healthy engine; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
import importlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
#: D2r: the consent variable; the locked wrapper exports it, this harness only reads it
ENV_VAR = "ASTRA_REAL_SITES"

#: D1r: ``(id, request, expected site)``. Requests are James's, in relative words (H-T205-3: no month, year or
#: day-of-month); the third column is the site the scenario is about (D2: none, it is found by search). Every turn
#: starts at the live start page ``/web.html`` (H-T205-5, ruled), never at that site.
REAL_SCENARIOS: tuple[tuple[str, str, str | None], ...] = (
    ("D1", "Is the Louvre open today, and until when?", "https://louvre.fr/"),
    (
        "D2",
        "find three well-reviewed ramen places in Seattle that are open past 9 pm on Friday",
        None,
    ),
    (
        "D3",
        "find a lasagna recipe on BBC Good Food and turn it into a grocery list",
        "https://bbcgoodfood.com/",
    ),
    (
        "D4",
        "my Breville Barista Express steam wand broke; find the replacement part and its price, and put it in "
        "the cart",
        "https://breville.com/",
    ),
    (
        "D5",
        "find a two-person camping stove under $60 with good reviews at REI and put it in the cart",
        "https://rei.com/",
    ),
    (
        "D6",
        "order a large pepperoni pizza from Domino's for pickup near Times Square",
        "https://dominos.com/",
    ),
    (
        "D8",
        "plan a day trip for two from San Francisco on Saturday: at most two hours' drive each way, under $200 "
        "in total, and every stop open on Saturday",
        "https://nps.gov/pore/",
    ),
    (
        "D9",
        "find the cheapest nonstop round trip from SFO to Seattle, leaving next Friday and back Sunday, and take "
        "it to the airline's checkout",
        "https://google.com/travel/flights",
    ),
)
#: H-T205-6: a guard miss on D7 would send a real request to a real business; refused in this task
REFUSED_IDS = frozenset({"D7"})
DEFAULT_REPEATS = 1
MAX_REPEATS = 2
TURN_TIMEOUT_S = 300
#: D4r: no turn starts once this much has passed since ``main`` started (read at call time)
RUN_BUDGET_S = 60 * 60
#: the repo's ``logs/`` (read at call time); nothing is written outside it
LOGS_DIR = REPO / "logs"
#: the live start page every turn starts at (H-T205-5)
START_PAGE = "/web.html"
#: the start origin of the executor built for ``main``'s check only: it is never launched, so nothing binds it
CHECK_ORIGIN = "http://127.0.0.1:9"
#: the session's step note of a web search (reading 7)
SEARCH_NOTE = "web search"
#: D3r: a final main-frame URL whose path reads as a bot wall
_CHALLENGE = re.compile(r"(?:^|/)sorry(?:/|$)|captcha|/cdn-cgi/challenge", re.IGNORECASE)
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>()]+")
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
_LOOPBACK_NAMES = ("localhost",)


def _say(msg: str) -> None:
    print(f"[t0205-real] {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# pure: classifiers and records
# =====================================================================================================


def final_kind(url: str | None) -> str:
    """D3r: ``challenge`` (a bot wall: ``/sorry/``, a captcha, a Cloudflare challenge), ``checkout`` (the executor's
    own rule, ``fence.checkout_url``) or ``other``. Reads the URL; stores nothing."""
    if not url:
        return "other"
    try:
        path = urlsplit(url).path
    except ValueError:
        return "other"
    if _CHALLENGE.search(path):
        return "challenge"
    from local_astra.computer import fence

    if fence.checkout_url(url):
        return "checkout"
    return "other"


def final_host(url: str | None) -> str | None:
    """D3r: the final URL's host only, lowercased; ``None`` without one."""
    if not url:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.lower() if host else None


def hosts_of(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D3r: the sorted union of every outcome's ``resolved["hosts"]`` and the sum of its ``hosts_more``."""
    hosts: set[str] = set()
    more = 0
    for outcome in outcomes:
        resolved = outcome.get("resolved") or {}
        hosts.update(str(h).lower() for h in resolved.get("hosts") or ())
        more += int(resolved.get("hosts_more") or 0)
    return {"hosts": sorted(hosts), "hosts_more": more}


def refusal_counts(egress: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """D3r (reading 5, ruled): the proxy's refusals counted per scheme, never per host."""
    counts: dict[str, int] = {}
    for record in egress:
        scheme = str(record.get("scheme") or "unknown")
        counts[scheme] = counts.get(scheme, 0) + 1
    return counts


def search_query_count(session_records: Sequence[Mapping[str, Any]]) -> int:
    """D3r: the number of the session log's ``web search`` notes (the queries stay in the session log)."""
    return sum(
        1
        for r in session_records
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == SEARCH_NOTE
    )


def reached(expected_host: str | None, hosts: Sequence[str]) -> bool | None:
    """Reading 11 (ruled): some reached host equals the expected host or is a subdomain of it; ``None`` when the
    scenario has no expected site."""
    if expected_host is None:
        return None
    want = expected_host.lower()
    return any(h.lower() == want or h.lower().endswith("." + want) for h in hosts)


def _row(scenario_id: str) -> tuple[str, str, str | None]:
    for row in REAL_SCENARIOS:
        if row[0] == scenario_id:
            return row
    raise KeyError(scenario_id)


def expected_host(scenario_id: str) -> str | None:
    """The host of the table's third column (the expected site), lowercased; ``None`` for D2 (and D7)."""
    try:
        site = _row(scenario_id)[2]
    except KeyError:
        return None
    return final_host(site)


def _moved_reason() -> str:
    from local_astra.computer.executor import MOVED_REASON

    return str(MOVED_REASON)


def scrub(text: str | None) -> str | None:
    """An error text with every URL replaced by its host only (a record never holds a URL path)."""
    if text is None:
        return None
    return _URL.sub(lambda m: f"<url {final_host(m.group(0)) or '?'}>", str(text))[:500]


def turn_record(scenario_id: str, repeat: int, raw: Mapping[str, Any]) -> dict[str, Any]:
    """D3r: the stored record of one turn (reading 7). Never a URL: the final URL gives its kind and host only."""
    outcomes = list(raw.get("outcomes") or ())
    egress = list(raw.get("egress") or ())
    moved = _moved_reason()
    executed = [o for o in outcomes if o.get("status") == "executed"]
    last = max(executed, key=lambda o: int(o.get("step") or 0)) if executed else None
    hosts = hosts_of(outcomes)
    expected = expected_host(scenario_id)
    tool_loop_end = raw.get("tool_loop_end")
    final_url = raw.get("final_url")
    last_host = final_host(final_url)
    return {
        "id": scenario_id,
        "repeat": repeat,
        "hosts": hosts["hosts"],
        "hosts_more": hosts["hosts_more"],
        "expected_host": expected,
        # I2 (RULED): over the reached hosts and the final host
        "reached": reached(expected, [*hosts["hosts"], *([last_host] if last_host else [])]),
        "refusals": refusal_counts(egress),
        "refusals_total": len(egress),
        "search_queries": search_query_count(raw.get("session_records") or ()),
        "steps": len(outcomes),
        "wall_s": raw.get("wall_s"),
        "tool_loop_end": tool_loop_end,
        "moved_refusals": sum(
            1 for o in outcomes if o.get("status") == "refused" and o.get("reason") == moved
        ),
        "waits": sum(1 for o in executed if o.get("tool") == "wait"),
        "hand_back": tool_loop_end == "handed_back",
        "last_action": (
            {"step": int(last.get("step") or 0), "tool": str(last.get("tool"))}
            if last is not None
            else None
        ),
        "final_kind": final_kind(final_url),
        "final_host": last_host,
        "error": scrub(raw.get("error")),
    }


def shot_path(out_dir: Path | str, scenario_id: str, repeat: int, step: int) -> Path:
    """D3r (reading 8): ``<out>/shots/<id>/<repeat>-<step>.jpg``; an id that is not a plain name (a separator,
    ``..``) raises ``ValueError`` and nothing is created."""
    if not isinstance(scenario_id, str) or not _SAFE_ID.fullmatch(scenario_id):
        raise ValueError(f"not a scenario id: {scenario_id!r}")
    out = Path(out_dir)
    path = out / "shots" / scenario_id / f"{int(repeat)}-{int(step)}.jpg"
    if out.resolve() not in path.resolve().parents:
        raise ValueError(f"{path} is outside {out}")
    return path


def scenario_label(scenario_id: str, done: int, repeats: int) -> str:
    """D4r: ``refused`` (D7), ``ran`` (every repeat done) or ``incomplete``."""
    if scenario_id in REFUSED_IDS:
        return "refused"
    return "ran" if done >= repeats else "incomplete"


def executor_refusal(executor: Any) -> str | None:
    """D2r: why the turn's executor may not reach real sites, or ``None``: not in web mode; an injected
    ``proxy_resolve``/``proxy_connect`` (a resolver or connector other than the egress proxy's own); a profile
    directory (web mode is a fresh profile)."""
    if getattr(executor, "web", False) is not True:
        return (
            "the executor is not in web mode (real sites go only through the web-mode egress proxy)"
        )
    driver = getattr(executor, "driver", None)
    injected = [
        name
        for name in ("proxy_resolve", "proxy_connect")
        if getattr(driver, "_" + name, None) is not None
    ]
    if injected:
        return (
            f"the executor has an injected {' and '.join(injected)} (an injected resolver/connector replaces the "
            "egress proxy's own)"
        )
    if getattr(driver, "_user_data_dir", None) is not None:
        return "the executor uses a profile directory (web mode is a fresh profile only)"
    return None


# =====================================================================================================
# HEAD's live web-mode settings, read from the cli at call time
# =====================================================================================================


def _profile() -> Any:
    """The lean profile as ``astra session`` at the repo root loads it."""
    from local_astra.config import load

    return load(REPO / "astra.toml").profiles["lean"]


def _live_args() -> argparse.Namespace:
    """``session --tools live --browser headless`` (web mode, search on: no ``--no-web``, no ``--search``); every
    cli resolver reads ``args`` with ``getattr``."""
    return argparse.Namespace(tools="live", browser="headless")


def session_keywords(profile: Any = None) -> dict[str, Any]:
    """Reading 9: the ``Session`` settings the cli passes for ``session --tools live`` in web mode, each resolved
    through the cli's own resolver at call time (``cli._build_graph``'s choices, keyword for keyword; plumbing such
    as the clock, the log and the kit is the turn's own)."""
    from local_astra import cli

    profile = _profile() if profile is None else profile
    args = _live_args()
    mode = "live"
    grounding = cli.grounding_of(profile, args, mode)
    keywords: dict[str, Any] = {
        "visual_tokens": grounding.visual_tokens,
        "reground": grounding.reground,
        "degeneracy_guard": True,
        "live_feedback": cli.feedback_of(profile, args, mode).session_value(),
        "recover_thought_calls": cli.recovery_of(profile, args, mode).session_value(),
    }
    if cli.auto_overview_of(profile, args, mode).auto_overview:
        keywords["auto_overview"] = True
    if cli.overview_frame_of(profile, args, mode, grounding=grounding).overview_frame:
        keywords["overview_frame"] = True
    if cli.action_outcome_of(profile, args, mode).action_outcome:
        keywords["action_outcome"] = True
    if cli.repeat_guard_of(profile, args, mode).repeat_guard:
        keywords["repeat_guard"] = True
    if cli.screen_reminder_of(args, mode).screen_reminder:
        keywords["screen_reminder"] = True
    if cli.completion_check_of(args, mode):
        keywords["completion_check"] = True  # T-0207 C (R-C2): resolved through cli at call time
    reask = cli.reask_of(profile, args, mode).value
    if reask is not None:
        keywords["reask_announced_calls"] = reask
    guidance = cli.live_guidance_of(profile, args, mode).value
    if guidance in ("v2", "v3"):
        keywords["page_fact"] = True
    keywords["tool_guidance"] = cli._tool_guidance(
        guidance, cli.web_of(profile, args, mode), cli.wait_tool_of(args, mode)
    )
    return keywords


def _manifest_settings(profile: Any) -> dict[str, Any]:
    """What the manifest records of HEAD's live web-mode default (every value from the cli)."""
    from local_astra import cli

    args = _live_args()
    loop = cli.tool_loop_of(profile, args)
    keywords = session_keywords(profile)
    return {
        "session_keywords": {k: _jsonable(v) for k, v in sorted(keywords.items())},
        "web": cli.web_of(profile, args, "live"),
        "search": cli.search_of(profile, args, "live").backend,
        "max_tool_steps": loop.max_tool_steps,
        "tool_loop": loop.on,
        "live_guidance": cli.live_guidance_of(profile, args, "live").value,
        "echo": cli.feedback_of(profile, args, "live").action_echo or cli.LIVE_DEFAULT_ACTION_ECHO,
        "page_overview": cli.page_overview_of(profile, args, "live").page_overview,
        "wait_tool": cli.wait_tool_of(args, "live"),
        "browser": "headless",
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


# =====================================================================================================
# seams main reads at call time
# =====================================================================================================


def _git(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=60,
    )
    return proc.stdout


def _sandbox_check() -> str | None:
    """``None`` when the pinned headless Chromium launches with its sandbox on (as web mode launches it), else why
    not. Launches the browser with no proxy and no page: nothing is fetched."""
    from local_astra import cli
    from local_astra.computer import runtime

    why = cli.live_prerequisites("headless")
    if why is not None:
        return f"the Chromium sandbox cannot be checked: {why}"

    async def probe() -> None:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True, chromium_sandbox=True, args=list(runtime.CHROMIUM_ARGS)
            )
            await browser.close()

    try:
        asyncio.run(probe())
    except Exception as exc:
        lines = [ln for ln in str(exc).splitlines() if ln.strip()]
        return f"Chromium did not launch with its sandbox on: {type(exc).__name__}: {lines[0] if lines else ''}"
    return None


def _make_executor(profile: Any, *, now_ms: Any, start_origin: str) -> Any:
    """The executor a turn gets, built as ``cli._start_live`` builds it for ``session --tools live --browser
    headless`` (every keyword resolved through the cli; web mode; nothing injected). Nothing is launched here."""
    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor

    args = _live_args()
    browser_mode = cli.browser_of(profile, args)
    echo = cli.feedback_of(profile, args, "live").action_echo or cli.LIVE_DEFAULT_ACTION_ECHO
    return BrowserToolExecutor(
        now_ms=now_ms,
        viewport=profile.browser_viewport,
        guard=None,
        headless=browser_mode == "headless",
        end_when_closed=browser_mode == "headed",
        allowed_origins=frozenset({start_origin}),
        echo=echo,
        web=cli.web_of(profile, args, "live"),
        start_origin=start_origin,
    )


def _root(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def engine_healthy(base_url: str) -> bool:
    """One ``GET /health`` on the engine (loopback; reading status spends nothing)."""
    try:
        with urllib.request.urlopen(_root(base_url) + "/health", timeout=10) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


# --- the turn ------------------------------------------------------------------------------------------


class _Watch:
    """The live browser seam the session gets: every name delegated to the real executor; records each execute's
    outcome (step, tool, status, reason, resolved) and the last frame the session took after each step."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.outcomes: list[dict[str, Any]] = []
        self.frames: dict[int, bytes] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute(self, inv: Any) -> Any:
        out = await self._inner.execute(inv)
        # I1: after an action the session reuses ``latest_frame`` and never calls ``frame()``; that frame is this
        # step's shot
        latest = getattr(self._inner, "latest_frame", None)
        jpeg = getattr(latest, "jpeg", None)
        if isinstance(jpeg, (bytes, bytearray)):
            self.frames[len(self.outcomes) + 1] = bytes(jpeg)
        self.outcomes.append(
            {
                "step": len(self.outcomes) + 1,
                "tool": str(inv.name),
                "status": getattr(out, "status", None),
                "reason": getattr(out, "reason", None),
                "resolved": dict(getattr(out, "resolved", None) or {}),
            }
        )
        return out

    async def frame(self) -> Any:
        frame = await self._inner.frame()
        jpeg = getattr(frame, "jpeg", None)
        if isinstance(jpeg, (bytes, bytearray)):
            self.frames[len(self.outcomes)] = bytes(jpeg)
        return frame


class _PlaybackOnly:
    """``FakeAudioIO`` with only the playback seams (typed turns: no capture pump)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        return int(await self.inner.flush())

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


class _NoMemory:
    """No recollections and never an uncaptioned episode (the idle captioner sends no request)."""

    def __init__(self) -> None:
        self._next = 0

    async def ingest(self, episode: Any) -> int:
        self._next += 1
        return self._next

    async def search(self, query: Any) -> Sequence[Any]:
        return ()

    async def set_caption(self, episode_id: int, caption: str) -> None:
        return None

    async def forget_since(self, t0_ms: int) -> int:
        return 0

    async def next_uncaptioned(self) -> Any:
        return None


def _fake(module: str, name: str) -> Any:
    return getattr(importlib.import_module(f"tests.fakes.{module}"), name)


def _hud_jpeg() -> bytes:
    """A synthetic HUD frame no page renders (magenta with a grey bar)."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 800), (255, 0, 255))
    ImageDraw.Draw(img).rectangle((0, 380, 1279, 420), fill=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


#: the web search backend, built once per process through the cli's start path (``[(backend, label)]``, or
#: ``[None]`` with the reason in ``_SEARCH_OFF``)
_SEARCH: list[Any] = []
_SEARCH_OFF: list[str] = []


def _search(backend: str) -> Any:
    if not _SEARCH:
        from local_astra import cli

        built, failure = cli.search_start(backend)
        _SEARCH.append(built)
        if failure is not None:
            _SEARCH_OFF.append(f"{failure[0]}; {failure[1]}")
            _say(f"web search is off: {failure[0]}")
    return _SEARCH[0]


async def _close_search() -> None:
    if _SEARCH and _SEARCH[0] is not None:
        closer = getattr(_SEARCH[0][0], "aclose", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                await closer()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


async def run_turn(
    *,
    scenario_id: str,
    request: str,
    start: str,
    repeat: int,
    base_url: str,
    out_dir: Path,
    timeout_s: float = TURN_TIMEOUT_S,
) -> dict[str, Any]:
    """One typed turn at HEAD's live web-mode default, from the live start page. Returns the raw turn (module
    docstring of the unit test): outcomes, egress refusals, session-log records, the final URL (never stored),
    frames, wall time, ``tool_loop_end`` and an error."""
    from local_astra import cli
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    profile = _profile()
    args = _live_args()
    keywords = session_keywords(profile)
    grounding = cli.grounding_of(profile, args, "live")
    loop = cli.tool_loop_of(profile, args)
    clock = SessionClock()
    pages = StartPageServer()  # the default root: ``/web.html`` as ``astra session`` serves it
    pages.start()
    browser: Any = None
    watch: _Watch | None = None
    session: Any = None
    clients: list[Any] = []
    log: Any = None
    error: str | None = None
    final_url: str | None = None
    egress: list[dict[str, str]] = []
    wall_s: float | None = None
    repair_off: str | None = None
    search_off: str | None = None
    sessions_dir = out_dir / "sessions"
    turns_path = out_dir / "session_turns.jsonl"
    try:
        browser = _make_executor(profile, now_ms=lambda: clock.now_ms, start_origin=pages.origin)
        why = executor_refusal(browser)
        if why is not None:
            raise RuntimeError(f"refused executor: {why}")
        start_url = browser.start_page_url
        if urlsplit(str(start_url)).path != urlsplit(start).path:
            raise RuntimeError(f"the executor's start page is not {start}")
        watch = _Watch(browser)
        extra: dict[str, bool] = {}
        if cli.page_overview_of(profile, args, "live").page_overview:
            extra["page_overview"] = True
        if cli.wait_tool_of(args, "live"):
            extra["wait_tool"] = True
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=watch, **extra)
        repair_off = kit.repair_off
        web = cli.search_of(profile, args, "live")
        if web.backend is not None:
            built = _search(web.backend)
            if built is not None:
                kit = dataclasses.replace(kit, search=built[0], web_search=web.backend)
            else:
                search_off = _SEARCH_OFF[0] if _SEARCH_OFF else "unavailable"
        endpoint = VllmEndpoint(base_url=base_url)
        llm = VllmClient(endpoint, detokenizer=detokenizer, visual_tokens=grounding.visual_tokens)
        clients.append(llm)
        second: dict[str, Any] = {}
        if grounding.reground:
            crop = grounding.reground_visual_tokens
            second = {
                "reground_llm": llm
                if crop == grounding.visual_tokens
                else VllmClient(endpoint, detokenizer=detokenizer, visual_tokens=crop),
                "reground_visual_tokens": crop,
            }
        ack_clip, still_working_after_ms = cli.clip_policy_of(profile, args, "live", loop.on)
        hud_jpeg = _hud_jpeg()
        hud = Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=1280,
            height=800,
            jpeg=hud_jpeg,
            phash=hashlib.sha256(hud_jpeg).hexdigest()[:16],
        )
        log = SessionLog(sessions_dir)
        session = Session(
            TurnMachine(
                TurnPolicy(
                    thinking=profile.diffusion.thinking,
                    tool_loop=loop.on,
                    max_tool_steps=loop.max_tool_steps,
                    ack_clip=ack_clip,
                    still_working_after_ms=still_working_after_ms,
                )
            ),
            llm,
            _fake("stt", "FakeStt")(),
            _fake("tts", "FakeTts")(),
            _PlaybackOnly(_fake("audio", "FakeAudioIO")(clock)),
            _fake("perception", "FakePerception")([hud]),
            _NoMemory(),
            clock=clock,
            config=SessionConfig(
                out_dir / "data",
                turns_path=turns_path,
                profile=profile,
                audio_path="null",
                source=Source.SCREEN,
            ),
            session_log=log,
            log_dir=sessions_dir,
            tool_kit=kit,
            grounding=grounding.turn_extra(),
            **second,
            **keywords,
        )
        # I5, as ``cli.build_session`` does (T-0200 C9): the executor's own notes go to the session's current log
        current = cli._CurrentSessionLog(session)
        browser.on_note = lambda data: current.record("note", data, session_ms=clock.now_ms)
        await asyncio.wait_for(browser.open(start_url), timeout=60)
        await cli._warm_repair(session)
        await session.start()
        session.turn_written.clear()
        submitted = time.monotonic()
        session.submit(E.UserText(t_ms=clock.now_ms, text=request))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=timeout_s)
        except TimeoutError:
            error = f"the turn did not end within {timeout_s} s"
        wall_s = round(time.monotonic() - submitted, 2)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if browser is not None:
            with contextlib.suppress(Exception):
                final_url = str(browser.driver.page.url)
            egress = [dict(r) for r in browser.egress_log]
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        for client in clients:
            with contextlib.suppress(Exception):
                await client.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    if session is not None and session.failures and error is None:
        error = "session failures: " + "; ".join(repr(f) for f in session.failures)
    records: list[dict[str, Any]] = []
    if log is not None and Path(log.path).is_file():
        records = sorted(_read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
    tool_loop_end: str | None = None
    if session is not None:
        for line in _read_jsonl(turns_path):
            if line.get("session_id") == session.session_id:
                tool_loop_end = (line.get("extras") or {}).get("tool_loop_end")
                break
    return {
        "outcomes": list(watch.outcomes) if watch is not None else [],
        "egress": egress,
        "session_records": records,
        "final_url": final_url,
        "frames": [
            {"step": step, "jpeg": jpeg}
            for step, jpeg in sorted((watch.frames if watch is not None else {}).items())
        ],
        "wall_s": wall_s,
        "tool_loop_end": tool_loop_end,
        "error": error,
        "repair_off": repair_off,
        "search_off": search_off,
        "session_log": Path(log.path).name if log is not None else None,
    }


# =====================================================================================================
# main
# =====================================================================================================


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--real-sites",
        action="store_true",
        help="consent to reach real websites (with ASTRA_REAL_SITES=1)",
    )
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument(
        "--scenarios", default=None, help="comma-separated ids (default: every runnable id)"
    )
    return ap


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr, flush=True)
    return EXIT_REFUSED


def _loopback(base_url: str) -> bool:
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _runnable() -> list[str]:
    return [row[0] for row in REAL_SCENARIOS if row[0] not in REFUSED_IDS]


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _one(values: list[Any]) -> Any:
    """n = 1: the turn's value; otherwise the list per repeat."""
    return values[0] if len(values) == 1 else values


def build_report(
    records: Mapping[str, Sequence[Mapping[str, Any]]], repeats: int, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """D4r: per scenario (every table id and D7) its label and, for a scenario that ran, what each turn reached."""
    scenarios: dict[str, Any] = {}
    ids = [row[0] for row in REAL_SCENARIOS] + sorted(REFUSED_IDS)
    for sid in sorted(set(ids), key=lambda s: (len(s), s)):
        recs = list(records.get(sid, ()))
        row: dict[str, Any] = {
            "label": scenario_label(sid, len(recs), repeats),
            "expected_host": expected_host(sid),
            "turns": len(recs),
        }
        if recs:
            row.update(
                reached=_one([r["reached"] for r in recs]),
                final_kind=_one([r["final_kind"] for r in recs]),
                final_host=_one([r["final_host"] for r in recs]),
                hosts=sorted({h for r in recs for h in r["hosts"]}),
                hosts_more=_one([r["hosts_more"] for r in recs]),
                steps=_one([r["steps"] for r in recs]),
                wall_s=_one([r["wall_s"] for r in recs]),
                tool_loop_end=_one([r["tool_loop_end"] for r in recs]),
                last_action=_one([r["last_action"] for r in recs]),
                hand_back=_one([r["hand_back"] for r in recs]),
                moved_refusals=_one([r["moved_refusals"] for r in recs]),
                waits=_one([r["waits"] for r in recs]),
                search_queries=_one([r["search_queries"] for r in recs]),
                refusals=_one([r["refusals"] for r in recs]),
                refusals_total=_one([r["refusals_total"] for r in recs]),
                errors=[r["error"] for r in recs if r.get("error")],
                shots=[s for r in recs for s in r.get("shots", ())],
            )
        scenarios[sid] = row
    return {
        "task": "T-0205 D",
        "manifest": dict(manifest),
        "search_off": manifest.get("search_off"),
        "scenarios": scenarios,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    manifest = report.get("manifest") or {}
    lines = [
        "# T-0205 D: real-site rehearsal",
        "",
        f"Commit {manifest.get('sha')}; started {manifest.get('started_local')}; finished "
        f"{manifest.get('finished_local')}. Real sites change: read every row with that date and time.",
        f"Repeats {manifest.get('repeats')}; web mode, headless, search {manifest.get('settings', {}).get('search')}; "
        f"loop {manifest.get('settings', {}).get('max_tool_steps')}; report-only, no judge.",
    ]
    if report.get("search_off"):
        lines.append(f"Search off: web search failed to start ({report['search_off']}).")
    if manifest.get("stopped"):
        lines.append(f"Stopped early: {manifest['stopped']}.")
    lines += [
        "",
        "| id | label | expected host | reached | final kind | final host | steps | ended | last action |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, row in report["scenarios"].items():
        lines.append(
            "| "
            + " | ".join(
                str(x)
                for x in (
                    sid,
                    row["label"],
                    row.get("expected_host"),
                    row.get("reached", "-"),
                    row.get("final_kind", "-"),
                    row.get("final_host", "-"),
                    row.get("steps", "-"),
                    row.get("tool_loop_end", "-"),
                    row.get("last_action", "-"),
                )
            )
            + " |"
        )
    for sid, row in report["scenarios"].items():
        if not row.get("turns"):
            continue
        lines += [
            "",
            f"## {sid}",
            "",
            f"- hosts: {', '.join(row['hosts']) or '(none)'} (+{row['hosts_more']} more)",
            f"- proxy refusals per scheme: {row['refusals']} (total {row['refusals_total']})",
            f"- web searches: {row['search_queries']}; moved refusals: {row['moved_refusals']}; waits: "
            f"{row['waits']}; hand-back: {row['hand_back']}; wall s: {row['wall_s']}",
        ]
        for err in row.get("errors") or ():
            lines.append(f"- error: {err}")
        lines.append("- screenshots:")
        lines += [f"  - {s}" for s in row["shots"]]
    return "\n".join(lines) + "\n"


def _now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    args = _parser().parse_args(argv)

    # --- refusals, before any socket and before anything is written ------------------------------------
    if not args.real_sites or os.environ.get(ENV_VAR) != "1":
        missing = []
        if not args.real_sites:
            missing.append("--real-sites")
        if os.environ.get(ENV_VAR) != "1":
            missing.append(f"{ENV_VAR}=1 (got {os.environ.get(ENV_VAR)!r})")
        return _refuse(
            f"real websites need both --real-sites and {ENV_VAR}=1; missing: {', '.join(missing)} "
            "(run scripts/locked/t0205_real_sites.sh through /launch)"
        )
    if not 1 <= args.repeats <= MAX_REPEATS:
        return _refuse(f"--repeats {args.repeats}: between 1 and {MAX_REPEATS} repeats only (D4r)")
    ids = _runnable()
    if args.scenarios is not None:
        asked = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
        refused = [s for s in asked if s in REFUSED_IDS]
        if refused:
            return _refuse(
                f"{', '.join(refused)} is refused in this task (H-T205-6: a guard miss would send a real request "
                "to a real business)"
            )
        unknown = [s for s in asked if s not in ids]
        if unknown or not asked:
            return _refuse(f"--scenarios {args.scenarios!r}: unknown ids {unknown}; known {ids}")
        ids = [s for s in ids if s in asked]
    if not _loopback(args.base_url):
        return _refuse(f"--base-url {args.base_url} is not a loopback address")
    dirty = _git("status", "--porcelain")
    if dirty.strip():
        return _refuse("dirty tree: " + " ".join(dirty.split()))
    sha = _git("rev-parse", "HEAD").strip()
    logs = Path(LOGS_DIR).resolve()
    out = (
        Path(args.out_dir).resolve() if args.out_dir is not None else logs / f"t0205-real-{sha[:7]}"
    )
    if logs not in out.parents:
        return _refuse(f"--out-dir {out} is not under logs/ ({logs}); nothing is written elsewhere")
    if out.exists():
        return _refuse(
            f"{out} exists; the run happens once per SHA (move it aside only with the human)"
        )
    why = _sandbox_check()
    if why is not None:
        return _refuse(f"Chromium sandbox: {why}")
    profile = _profile()
    why = executor_refusal(_make_executor(profile, now_ms=lambda: 0, start_origin=CHECK_ORIGIN))
    if why is not None:
        return _refuse(why)
    settings = _manifest_settings(profile)

    # --- the first socket: the engine's health ----------------------------------------------------------
    if not engine_healthy(args.base_url):
        _say(f"no healthy engine at {args.base_url}")
        return EXIT_FAILED

    out.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "sha": sha,
        "started_local": _now_local(),
        "base_url": args.base_url,
        "repeats": args.repeats,
        "scenarios": ids,
        "refused": sorted(REFUSED_IDS),
        "requests": {row[0]: row[1] for row in REAL_SCENARIOS},
        "expected_hosts": {row[0]: expected_host(row[0]) for row in REAL_SCENARIOS},
        "start_page": START_PAGE,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "settings": settings,
    }
    _write_json(out / "manifest.json", manifest)

    done: dict[str, list[dict[str, Any]]] = {sid: [] for sid in ids}
    stopped: str | None = None
    try:
        for repeat in range(args.repeats):
            for sid in ids:
                if time.monotonic() - started >= RUN_BUDGET_S:
                    stopped = f"the run budget of {RUN_BUDGET_S} s is spent"
                    break
                request = _row(sid)[1]
                _say(f"{sid}/r{repeat} start {_now_local()}")
                try:
                    raw = asyncio.run(
                        run_turn(
                            scenario_id=sid,
                            request=request,
                            start=START_PAGE,
                            repeat=repeat,
                            base_url=args.base_url,
                            out_dir=out,
                            timeout_s=TURN_TIMEOUT_S,
                        )
                    )
                except Exception as exc:
                    raw = {"error": f"harness: {type(exc).__name__}: {exc}"}
                record = turn_record(sid, repeat, raw)
                shots: list[str] = []
                for frame in raw.get("frames") or ():
                    path = shot_path(out, sid, repeat, int(frame["step"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(bytes(frame["jpeg"]))
                    shots.append(path.relative_to(out).as_posix())
                record["shots"] = shots
                record["repair_off"] = raw.get("repair_off")
                record["search_off"] = raw.get("search_off")
                record["session_log"] = raw.get("session_log")
                done[sid].append(record)
                with (out / "records.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                _say(
                    f"{sid}/r{repeat} steps={record['steps']} end={record['tool_loop_end']} "
                    f"final={record['final_kind']}@{record['final_host']} reached={record['reached']} "
                    f"error={record['error']}"
                )
            if stopped is not None:
                break
    finally:
        with contextlib.suppress(Exception):
            asyncio.run(_close_search())
    manifest["finished_local"] = _now_local()
    # I3 (RULED): search that failed to start reads as off, with its reason, never as the model's choice
    search_off = next(
        (str(r["search_off"]) for recs in done.values() for r in recs if r.get("search_off")),
        _SEARCH_OFF[0] if _SEARCH_OFF else None,
    )
    if search_off is not None:
        manifest["settings"] = {**manifest["settings"], "search": "off"}
        manifest["search_off"] = search_off
        _say(f"web search was off: {search_off}")
    if stopped is not None:
        manifest["stopped"] = stopped
        _say(f"stopping: {stopped}")
    _write_json(out / "manifest.json", manifest)
    report = build_report(done, args.repeats, manifest)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _say(f"report: {out / 'report.md'}")
    return EXIT_FAILED if stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
