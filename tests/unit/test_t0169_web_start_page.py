"""T-0169 criteria 1-5 (CPU): a live session in web mode opens ``<origin>/web.html`` first; session start, Restart
(``reset_page``) and Ctrl+T/Ctrl+L agree on one start URL; loopback mode and everything the model or the user is told
stay as at b372edaa (docs/tasks/T-0169.md).

**The rig.** A real ``BrowserToolExecutor`` over T-0111c's ``FakeDriver`` (``tests/unit/test_computer_executor.py``),
which records every ``goto`` and opens nothing; a duck-typed graph carrying ``pages.origin``,
``session.tool_kit.live_browser`` (the executor) and ``web``, handed to the real ``cli._open_start_page``. The
``--no-web`` / ``browser_web = false`` / default-live rows build the executor from exactly the keywords ``cli._start_live``
passes (T-0160's ``_built`` rig in ``test_cli_tools_web.py``), with the fake driver added.

**Expected URLs by hand.** ``ORIGIN = "http://127.0.0.1:45678"`` (the rig's start-page origin): web mode
``http://127.0.0.1:45678/web.html``, loopback mode ``http://127.0.0.1:45678/`` (b372edaa's ``pages.origin + "/"``).

**Readings (reported):**

- Criterion 2 is pinned by behaviour only: the URL ``_open_start_page`` navigates to equals the URL ``reset_page()``
  and the web-mode Ctrl+T plan navigate to. No row greps the implementation or names the shared definition.
- "The executor's ``web`` flag is the authority, not ``graph.web``" (implementation notes): a graph whose ``web``
  disagrees with its executor's opens the page the *executor's* mode names.
- Criterion 5: the sandbox hint is keyed on the open failing, as at b372edaa (``graph.web`` and the error text); the
  failing navigation must be the ``/web.html`` one. The b372edaa message shape is the oracle for "keeps its message".
- Criterion 3/4's "identical to b372edaa" is a differential: the same capture runs in-process and in a subprocess over
  ``git archive b372edaa`` (T-0152's harness, origin checked).

Rows green at b372edaa by design (negative, active): every loopback row, the executor-authority loopback direction, the
``--no-web`` sandbox row and the b372edaa differential. Every other row was pending until ``/implement T-0169`` (activated at c258890c).

Regime: CPU over fakes; no browser, no socket, no measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_astra import cli
from local_astra.computer import executor as executor_mod
from tests import _git_range as gr
from tests.unit import _t0160_pins as t0160  # T-0178 A8
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0201_pins as t0201  # T-0201 D3 re-pin
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tools_live import (
    ORIGIN,
    Events,
    StartBoom,
    _serve_rig,
    argv,
    fake_live,
)
from tests.unit.test_cli_tools_web import LIVE, SANDBOX_ERROR, _built
from tests.unit.test_computer_executor import FakeDriver, inv, own_frame

#: the real class, taken at import (the ``_built`` rig replaces the module attribute with its spy for one test)
REAL_EXECUTOR = executor_mod.BrowserToolExecutor
CLOCK_MS = 1_700_000_016_916
WEB_URL = "http://127.0.0.1:45678/web.html"
CATALOGUE_URL = "http://127.0.0.1:45678/"
#: b372edaa: a page on the start origin other than the start page (the model navigated there)
ELSEWHERE = "http://127.0.0.1:45678/search.html"
BASE_SHA = "b372edaa"
#: criterion 4's pins, written out (``test_persona_name.py``; the task: 768a973b, tools_v1 bbfc53cc)
LIVE_TOOL_GUIDANCE_SHA256 = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
TOOLS_V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------- the rig


def gotos(d: FakeDriver, since: int = 0) -> list[str]:
    return [str(c[1][0]) for c in d.calls[since:] if c[0] == "goto"]


def executor(
    d: FakeDriver, *, web: bool, origin: str = ORIGIN, start_origin: str | None = None
) -> Any:
    """As ``cli._start_live`` builds it (start origin = allowed origin = the start-page origin), over ``d``."""
    return REAL_EXECUTOR(
        now_ms=lambda: CLOCK_MS,
        viewport=d.viewport,
        driver=d,
        allowed_origins=frozenset({origin}),
        web=web,
        start_origin=start_origin if start_origin is not None else origin,
    )


def graph(ex: Any, *, web: bool, origin: str = ORIGIN) -> Any:
    return SimpleNamespace(
        pages=SimpleNamespace(origin=origin),
        session=SimpleNamespace(tool_kit=SimpleNamespace(live_browser=ex)),
        web=web,
    )


async def start(ex: Any, *, web: bool, origin: str = ORIGIN) -> None:
    await cli._open_start_page(graph(ex, web=web, origin=origin))


def expected(web: bool) -> str:
    return WEB_URL if web else CATALOGUE_URL


# =====================================================================================================
# criterion 1: web mode, session start
# =====================================================================================================


async def test_web_mode_first_navigation_is_web_html() -> None:
    """C1: with web mode on, the executor's first (and only) navigation at session start is ``<origin>/web.html``;
    nothing navigates to the catalogue ``<origin>/`` before it."""
    d = FakeDriver()
    ex = executor(d, web=True)
    await start(ex, web=True)
    assert gotos(d) == [WEB_URL], (
        f"T-0169 C1: a web-mode session opens {WEB_URL} exactly once and nothing before it; navigated {gotos(d)}"
    )


def test_web_mode_first_navigation_through_cli_kwargs(tmp_path, capsys, monkeypatch) -> None:
    """C1 through ``cli``: the executor built from exactly the keywords a default ``--tools live`` start passes
    (web on by default) opens ``<origin>/web.html`` first."""
    kw = _built(tmp_path, capsys, monkeypatch, *LIVE)
    assert kw.get("web") is True, f"premise: a default live start is in web mode; {kw}"
    monkeypatch.undo()
    d = FakeDriver()
    ex = REAL_EXECUTOR(**{**kw, "driver": d})
    asyncio.run(start(ex, web=True))
    assert gotos(d) == [WEB_URL], f"T-0169 C1: navigated {gotos(d)}"


# =====================================================================================================
# criterion 3: loopback mode unchanged (negative; green at b372edaa)
# =====================================================================================================


async def test_loopback_first_navigation_is_catalogue_direct() -> None:
    """C3 (negative): ``web=False`` (with the start origin known, as ``cli`` builds it) opens exactly ``<origin>/``."""
    d = FakeDriver()
    ex = executor(d, web=False)
    await start(ex, web=False)
    assert gotos(d) == [CATALOGUE_URL], f"T-0169 C3: loopback opens the catalogue; {gotos(d)}"


@pytest.mark.parametrize(
    ("extra", "body"),
    [(["--no-web"], ""), ([], "browser_web = false\n")],
    ids=["no-web-flag", "browser_web-false"],
)
def test_loopback_first_navigation_is_catalogue(tmp_path, capsys, monkeypatch, extra, body) -> None:
    """C3 (negative): the executor ``cli._start_live`` builds with ``--no-web`` or ``browser_web = false`` opens
    exactly ``<origin>/`` at session start, as at b372edaa."""
    kw = _built(tmp_path, capsys, monkeypatch, *LIVE, *extra, body=body)
    assert not kw.get("web"), f"premise: web off; {kw}"
    monkeypatch.undo()
    d = FakeDriver()
    ex = REAL_EXECUTOR(**{**kw, "driver": d})
    asyncio.run(start(ex, web=False))
    assert gotos(d) == [CATALOGUE_URL], f"T-0169 C3: {extra or body!r} opens {gotos(d)}"


async def test_executor_flag_is_the_authority_loopback() -> None:
    """Implementation notes (negative direction, green at b372edaa): a loopback executor under a graph that says
    ``web`` still opens the catalogue (the executor's flag decides, never ``graph.web``)."""
    d = FakeDriver()
    ex = executor(d, web=False)
    await start(ex, web=True)
    assert gotos(d) == [CATALOGUE_URL], gotos(d)


async def test_executor_flag_is_the_authority_web() -> None:
    """Implementation notes: a web-mode executor under a graph whose ``web`` is False opens ``/web.html`` (the
    executor's flag decides, never ``graph.web``)."""
    d = FakeDriver()
    ex = executor(d, web=True)
    await start(ex, web=False)
    assert gotos(d) == [WEB_URL], f"T-0169: the executor's web flag is the authority; {gotos(d)}"


# =====================================================================================================
# criterion 2: one source for the start URL (adversarial: the divergence that caused the bug)
# =====================================================================================================


@pytest.mark.parametrize("web", [pytest.param(True, id="web"), pytest.param(False, id="loopback")])
async def test_start_and_reset_agree(web: bool) -> None:
    """C2 (adversarial): session start, then the model navigates elsewhere on the start origin, then Restart
    (``reset_page()``): the start navigation and the reset navigation are the same URL, and it is the mode's start
    page. At b372edaa web mode started on ``/`` and reset to ``/web.html``; that divergence is the bug."""
    d = FakeDriver()
    ex = executor(d, web=web)
    await start(ex, web=web)
    first = gotos(d)
    await ex.open(ELSEWHERE)
    mark = len(d.calls)
    await ex.reset_page()
    after = gotos(d, mark)
    assert len(first) == 1 and len(after) == 1, f"one navigation each; start {first}, reset {after}"
    assert first[0] == after[0], (
        f"T-0169 C2: session start opened {first[0]} but Restart navigates to {after[0]}"
    )
    assert first[0] == expected(web), f"T-0169 C1/C3: the {web=} start page is {expected(web)}"


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
async def test_ctrl_t_target_equals_start_url(combo: str) -> None:
    """C2: in web mode the Ctrl+T/Ctrl+L plan navigates to the URL session start opened."""
    d = FakeDriver()
    ex = executor(d, web=True)
    await start(ex, web=True)
    first = gotos(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert out.status == "executed", out
    assert first == gotos(d, mark) == [WEB_URL], (
        f"T-0169 C2: start {first}, {combo} {gotos(d, mark)}; both must be {WEB_URL}"
    )


async def test_trailing_slash_origin() -> None:
    """C2 (adversarial): a start origin given with a trailing ``/`` still opens ``http://127.0.0.1:45678/web.html``
    at start and at Restart: never ``//web.html``."""
    d = FakeDriver()
    slashed = ORIGIN + "/"
    ex = executor(d, web=True, origin=slashed)
    await start(ex, web=True, origin=slashed)
    mark = len(d.calls)
    await ex.reset_page()
    got = gotos(d, 0)
    assert not [u for u in got if "//web.html" in u], f"T-0169: a doubled slash; navigated {got}"
    assert gotos(d)[:1] == [WEB_URL] and gotos(d, mark) == [WEB_URL], (
        f"T-0169 C1/C2: start and Restart both open {WEB_URL}; navigated {got}"
    )


# =====================================================================================================
# criterion 5: start failures keep their messages
# =====================================================================================================


class SandboxBoom(RuntimeError):
    pass


class FailingDriver(FakeDriver):
    """``goto`` records its URL, then raises ``error`` (the launch error surfaced by the first navigation)."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def goto(self, url: str) -> None:
        self._enter("goto", url)
        raise self.error


#: b372edaa's T-0111g message: the type and the first line of the error, collapsed (``_open_start_page``)
SANDBOX_FIRST = "BrowserType.launch: Target page, context or browser has been closed"


async def _failed_start(web: bool, error: BaseException) -> tuple[FailingDriver, str, list[str]]:
    d = FailingDriver(error)
    ex = executor(d, web=web)
    closes: list[str] = []

    async def close() -> None:
        closes.append("close")

    ex.close = close  # the injected driver is the test's; this records the await
    with pytest.raises(cli.LiveStartFailed) as got:
        await start(ex, web=web)
    return d, str(got.value), closes


async def test_sandbox_failure_on_web_html_keeps_hint() -> None:
    """C5: in web mode the failing navigation is the ``/web.html`` one; Chromium's sandbox error raises
    ``LiveStartFailed`` with ``WEB_SANDBOX_HINT`` (b372edaa's message, word for word) and the browser is closed."""
    d, message, closes = await _failed_start(True, SandboxBoom(SANDBOX_ERROR))
    assert gotos(d) == [WEB_URL], f"T-0169 C1/C5: the failing open was {gotos(d)}"
    want = (
        f"cannot open the start page in the live browser: SandboxBoom: {SANDBOX_FIRST}; "
        f"{cli.WEB_SANDBOX_HINT}"
    )
    assert message == want, f"T-0169 C5:\n got  {message!r}\n want {want!r}"
    assert closes == ["close"], f"T-0169 C5: the browser is closed once; {closes}"


async def test_other_failure_on_web_html_keeps_t0111g_message() -> None:
    """C5: another failure of the web-mode open of ``/web.html`` keeps T-0111g's message (no hint); closed."""
    d, message, closes = await _failed_start(True, StartBoom("no page\nsecond line"))
    assert gotos(d) == [WEB_URL], f"T-0169 C1/C5: the failing open was {gotos(d)}"
    assert message == "cannot open the start page in the live browser: StartBoom: no page", message
    assert closes == ["close"], closes


@pytest.mark.parametrize("error", ["sandbox", "other"])
async def test_no_web_failure_has_no_hint(error: str) -> None:
    """C5 (negative, green at b372edaa): with ``--no-web`` the same sandbox error, or another, gives T-0111g's
    message without the hint, on the catalogue open; the browser is closed."""
    exc = SandboxBoom(SANDBOX_ERROR) if error == "sandbox" else StartBoom("no page")
    d, message, closes = await _failed_start(False, exc)
    first = SANDBOX_FIRST if error == "sandbox" else "no page"
    name = type(exc).__name__
    assert gotos(d) == [CATALOGUE_URL], gotos(d)
    assert message == f"cannot open the start page in the live browser: {name}: {first}", message
    assert cli.WEB_SANDBOX_HINT not in message
    assert closes == ["close"], closes


# =====================================================================================================
# criteria 3 and 4: what the user and the model are told (negative; a differential against b372edaa)
# =====================================================================================================

#: (case, extra argv, profile body, open error) of the real-start rigs; ``None``: the start succeeds
SERVE_CASES: dict[str, tuple[list[str], str, str | None]] = {
    "web": ([], "", None),
    "no-web": (["--no-web"], "", None),
    "browser_web-false": ([], "browser_web = false\n", None),
    "web-start-failure": ([], "", "other"),
    "web-sandbox-failure": ([], "", "sandbox"),
    "no-web-start-failure": (["--no-web"], "", "other"),
    "no-web-sandbox-failure": (["--no-web"], "", "sandbox"),
}
#: T-0183 re-scope (docs/tasks/T-0183.md, "Differentials against a BASE that has ``--visual-tokens``"; the T-0174 D4b
#: pattern on both sides): the ``off``/``dry_run`` cases pass ``--visual-tokens 1120`` on both sides (b372edaa has the
#: flag), since T-0183 makes 1120 the default there; nothing else changes. The live cases are unchanged.
VT1120 = ["--visual-tokens", "1120"]
#: (case, extra argv, profile body) of the ``--dry-run`` rigs
DRY_CASES: dict[str, tuple[list[str], str]] = {
    "live-web": ([*LIVE], ""),
    "live-no-web": ([*LIVE, "--no-web"], ""),
    "live-browser_web-false": ([*LIVE], "browser_web = false\n"),
    "off": ([*VT1120], ""),
    "off-no-web": (["--no-web", *VT1120], ""),
    "dry_run": (["--tools", "dry-run", *VT1120], ""),
}


def _norm(text: str, where: Path) -> str:
    return text.replace(str(where), "<tmp>")


#: T-0179 A-5 (the T-0174 D4b pattern): the working tree's live runs pass ``--no-repeat-guard`` so the session keeps
#: BASE's guard (off) after ``LIVE_DEFAULT_REPEAT_GUARD`` flipped, and the line it prints is read back as BASE's
GUARD_OFF_PIN = "--no-repeat-guard"
GUARD_OFF_PIN_LINE = "astra session: repeat guard: off (flag)"
GUARD_BASE_LINE = "astra session: repeat guard: off (default)"


def _pin_guard_off() -> bool:
    """On the working tree only: a BASE run (b372edaa, before T-0174's guidance selector) is untouched."""
    from local_astra import config

    return isinstance(getattr(config, "LIVE_GUIDANCE_VERSIONS", None), tuple)


def _unpin_guard_line(text: str, where: Path) -> str:
    lines = text.splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if x.rstrip("\n") == GUARD_OFF_PIN_LINE]
    assert len(found) == 1, (
        f"T-0179 A-5 ({where}): the pin prints {GUARD_OFF_PIN_LINE!r} once; {found}"
    )
    lines[found[0]] = GUARD_BASE_LINE + "\n"
    return "".join(lines)


def _dry_run(where: Path, extra: list[str], body: str) -> dict[str, Any]:
    where.mkdir(parents=True, exist_ok=True)
    out, err = io.StringIO(), io.StringIO()
    pinned = _pin_guard_off() and "live" in extra  # T-0179 A-5: the live dry-run cases only
    if pinned:
        extra = [*extra, GUARD_OFF_PIN]
    # T-0202 D3 re-pin: pin off (the live dry-run cases append --wait-tool off on a tree with the flag, so the line is
    # b372edaa's, which has no wait tool; off and dry_run are not live; a no-op at b372edaa)
    extra = t0202.pin_off(extra, body, live_prefix=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        T.patch_detok_load(mp, T.Calls(), where)
        explode_clients(mp)
        fake_live(mp, Events())
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli([*argv(where, *extra, config_body=body), "--dry-run"])
    printed = out.getvalue()
    if pinned and code == 0:
        printed = _unpin_guard_line(printed, where)
    printed = t0201.text_without_default(
        printed, str(where)
    )  # T-0201 D3 re-pin (admit); BASE untouched
    printed = t0207.text_without_default(
        printed, str(where)
    )  # T-0207 C3 re-pin (admit); BASE untouched
    return {"code": code, "out": _norm(printed, where), "err": _norm(err.getvalue(), where)}


def _serve(where: Path, extra: list[str], body: str, error: str | None) -> dict[str, Any]:
    where.mkdir(parents=True, exist_ok=True)
    out, err = io.StringIO(), io.StringIO()
    exc: BaseException | None = None
    if error == "sandbox":
        exc = SandboxBoom(SANDBOX_ERROR)
    elif error == "other":
        exc = StartBoom("no page")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        events = Events()
        args = _serve_rig(where, mp, events, open_error=exc)
        if body:
            config = Path(args[args.index("--config") + 1])
            config.write_text(config.read_text() + body)
        pinned = _pin_guard_off()  # T-0179 A-5: every serve case is live
        if pinned:
            extra = [*extra, GUARD_OFF_PIN]
        # T-0202 D3 re-pin: pin off (every serve case is live; --wait-tool off on a tree with the flag keeps
        # b372edaa's session, which has no wait tool; a no-op at b372edaa)
        extra = t0202.pin_off(extra, body, live_prefix=True)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli([*args, *extra])
    printed = out.getvalue()
    if pinned and GUARD_OFF_PIN_LINE in printed:
        printed = _unpin_guard_line(printed, where)
    printed = t0201.text_without_default(
        printed, str(where)
    )  # T-0201 D3 re-pin (admit); BASE untouched
    printed = t0207.text_without_default(
        printed, str(where)
    )  # T-0207 C3 re-pin (admit); BASE untouched
    return {
        "code": code,
        "out": _norm(printed, where),
        "err": _norm(err.getvalue(), where),
        # the order of what started and stopped; the opened URL is criterion 1's, compared by the rows above
        "events": [e[0] for e in events],
    }


def _capture(tmp: Path) -> dict[str, Any]:
    """Every case, in-process here and in a subprocess over the b372edaa tree."""
    out: dict[str, Any] = {"dry": {}, "serve": {}}
    for case, (extra, body) in DRY_CASES.items():
        out["dry"][case] = _dry_run(tmp / "dry" / case, extra, body)
    for case, (extra, body, error) in SERVE_CASES.items():
        out["serve"][case] = _serve(tmp / "serve" / case, extra, body, error)
    guidance = cli.LIVE_TOOL_GUIDANCE
    out["guidance_sha256"] = hashlib.sha256(guidance.encode("utf-8")).hexdigest()
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0169_web_start_page as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """b372edaa's capture, in its own tree (``git archive``; the import origin is checked)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0169_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


def test_prompt_and_stdout_unchanged(base: dict[str, Any], tmp_path: Path) -> None:
    """C3/C4 (negative, green at b372edaa): every ``--dry-run`` stdout and stderr (live web, ``--no-web``,
    ``browser_web = false``, off, dry_run), every real-start stdout/stderr and start/stop order (web and loopback,
    succeeding, failing, failing on the sandbox), and ``LIVE_TOOL_GUIDANCE`` are byte-identical to b372edaa's."""
    now = _capture(tmp_path / "now")
    for kind in ("dry", "serve"):
        for case, want in base[kind].items():
            got = now[kind][case]
            # T-0172 R10 pin move (B9): every tools-on run (all serve cases are live) prints the default reask line,
            # the last tools-on line; off prints none
            if kind == "serve" or not case.startswith("off"):
                got = {
                    **got,
                    "out": t0172.admit_reask_line(
                        got["out"], "start" if kind == "serve" else "dry"
                    ),
                }
            else:
                t0172.assert_no_reask(got["out"])
            if kind == "dry" and case == "live-web":
                # T-0178 A8: web mode's loop=on(20) back to BASE's 8, strictly, once group A exists
                got = {**got, "out": t0160.admit_web_loop(got["out"])}
            assert got == want, (
                f"T-0169 C3/C4: {kind} {case} differs from {BASE_SHA}:\n"
                f"base {json.dumps(want)[:2000]}\nnow  {json.dumps(got)[:2000]}"
            )
    assert now["guidance_sha256"] == base["guidance_sha256"] == LIVE_TOOL_GUIDANCE_SHA256


def test_model_inputs_unchanged() -> None:
    """C4 (negative, green at b372edaa): ``LIVE_TOOL_GUIDANCE`` keeps 768a973b, ``tools_v1.json`` keeps bbfc53cc and
    ``pages/static/web.html`` is byte-identical to b372edaa's except T-0230's two substitutions (the placeholder and
    the hint, T-0230 R-3)."""
    from tests.unit.test_t0230_start_page import t0230_substitute

    gr.require_commits(BASE_SHA)
    guidance = hashlib.sha256(cli.LIVE_TOOL_GUIDANCE.encode("utf-8")).hexdigest()
    assert guidance == LIVE_TOOL_GUIDANCE_SHA256, guidance
    assert hashlib.sha256(T.TOOLS_V1.read_bytes()).hexdigest() == TOOLS_V1_SHA256
    rel = "local_astra/pages/static/web.html"
    got = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_SHA}:{rel}"], capture_output=True, check=True
    )
    want = t0230_substitute(got.stdout.decode("utf-8")).encode("utf-8")
    assert (REPO / rel).read_bytes() == want, (
        "T-0169 C4: web.html changed (by more than T-0230's two substitutions)"
    )
