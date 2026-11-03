"""T-0160 group C, C-1 to C-4 and C-7 (cli side): web mode on by default in ``--tools live``, ``--no-web`` and
``browser_web = false``, the ``--dry-run`` egress word, the startup notice and the sandbox failure message
(docs/tasks/T-0160.md).

**Readings the rows take (listed in the test-author's report):**

- C-1: ``_start_live`` passes ``web=`` and ``start_origin=pages.origin`` to ``BrowserToolExecutor`` in web mode. With
  ``--no-web`` (or ``browser_web = false``) the executor's ``web`` keyword is falsy, absent or ``False`` (B-8: such
  a launch is BASE's).
- C-3: the notice is printed once at a **real** start, after the start page opened (the T-0125 startup line's
  pattern: a start failure stays the only stderr line), never under ``--dry-run`` (C-2: every other ``--dry-run``
  line is BASE's) and never with ``--no-web``. Each C-3 clause is matched case-sensitively, whitespace-collapsed.
- C-4: "Chromium's sandbox error" is a launch failure whose message holds ``Chromium sandboxing failed!`` (planner
  finding 1, the message on this box). The message names the README section and says ``--no-web``. Too wide: any
  other web-mode start failure, and the same error with ``--no-web``, do not name the section.

Regime: CPU over the T-0111g fakes (spy executor, fake start pages, fake HUD); nothing launched, nothing on the
network, no measurement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.config import Profile
from tests.unit import _t0202_pins as t0202
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import CLI, explode_clients, flags_of, run_cli
from tests.unit.test_cli_tools_live import (
    LIVE_LINE,
    LOOPBACK_LIVE_LINE,
    ORIGIN,
    Events,
    _out,
    _serve_rig,
    argv,
    failure_lines,
    fake_live,
)

LIVE = ["--tools", "live", "--browser", "headless"]
#: C-3, verbatim from the criterion
NOTICE = (
    "can open public websites",
    "pages can contain instructions that mislead it",
    "never types passwords or card numbers",
    "asks before it buys, sends, posts or deletes",
    "no saved logins",
    "downloads and uploads are off",
    "Do not sign in to anything in its window",
    "--no-web turns this off",
)
SANDBOX_SECTION = "Real websites: the browser sandbox"
SANDBOX_ERROR = (
    "BrowserType.launch: Target page, context or browser has been closed\n"
    "[pid=4242][err] Chromium sandboxing failed!\n"
    "╔════════════╗\n║ BOX ║\n╚════════════╝"
)
WEB_HTML = Path(cli.__file__).resolve().parent / "pages" / "static" / "web.html"


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _notice_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if NOTICE[0] in line]


def _dry(tmp_path: Path, capsys: Any, monkeypatch: Any, *extra: str, body: str = "") -> str:
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events())
    try:
        code = run_cli([*argv(tmp_path, *extra, config_body=body), "--dry-run"])
    except (
        Exception
    ) as exc:  # e.g. the profile refuses an unknown key: reported as the row's failure
        raise AssertionError(f"T-0160 C-1/C-2: {extra} {body!r} --dry-run raised: {exc}") from exc
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    assert not _notice_lines(err), (
        f"T-0160 C-3 (reading): no notice under --dry-run; stderr:\n{err}"
    )
    return out


def _built(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *extra: str, body: str = ""
) -> dict[str, Any]:
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    made = fake_live(monkeypatch, Events())
    try:
        code = run_cli(argv(tmp_path, *extra, config_body=body))
    except (
        Exception
    ) as exc:  # e.g. the profile refuses an unknown key: reported as the row's failure
        raise AssertionError(f"T-0160 C-1: {extra} {body!r} did not start: {exc}") from exc
    out, err = _out(capsys)
    assert code == 0 and len(made) == 1, f"premise: one executor built; {code}\n{out}{err}"
    return dict(made[0].kwargs)


# --------------------------------------------------------------------------------------------- C-1


def test_profile_browser_web_defaults_true() -> None:
    """C-1: ``Profile.browser_web: bool = True``."""
    field = Profile.model_fields.get("browser_web")
    assert field is not None, "T-0160 C-1: Profile.browser_web does not exist yet"
    assert field.annotation is bool and field.default is True, (field.annotation, field.default)
    assert Profile(name="lean").browser_web is True  # type: ignore[attr-defined]


def test_no_web_flag_is_store_true() -> None:
    """C-1: ``--no-web`` (store_true): there is no ``--web`` counterpart."""
    flags = flags_of(CLI)
    assert "--no-web" in flags, "T-0160 C-1: astra session --no-web does not exist yet"
    assert "--web" not in flags, "T-0160 C-1: --no-web is store_true, not a BooleanOptionalAction"


def test_live_default_builds_web_executor(tmp_path, capsys, monkeypatch) -> None:
    """C-1: a default live start builds the executor with ``web=True`` and ``start_origin`` = the start-page origin."""
    kw = _built(tmp_path, capsys, monkeypatch, *LIVE)
    assert kw.get("web") is True, f"T-0160 C-1 (ruling 2): web on by default in live; {kw}"
    assert kw.get("start_origin") == ORIGIN, f"T-0160 C-1: start_origin=pages.origin; {kw}"
    assert frozenset(kw.get("allowed_origins") or ()) == frozenset({ORIGIN}), kw


@pytest.mark.parametrize(
    ("extra", "body"),
    [(["--no-web"], ""), ([], "browser_web = false\n"), (["--no-web"], "browser_web = true\n")],
    ids=["flag", "profile", "flag-over-profile"],
)
def test_no_web_builds_loopback_executor(tmp_path, capsys, monkeypatch, extra, body) -> None:
    """C-1 (the off direction): ``--no-web`` or ``browser_web = false`` gives an executor not in web mode."""
    kw = _built(tmp_path, capsys, monkeypatch, *LIVE, *extra, body=body)
    assert not kw.get("web"), f"T-0160 C-1: web off with {extra or body!r}; {kw}"
    # premise of the off rows (too-wide direction): the same rig without the switch is in web mode
    monkeypatch.undo()
    on = _built(tmp_path / "on", capsys, monkeypatch, *LIVE)
    assert on.get("web") is True, on


@pytest.mark.parametrize("mode", [[], ["--tools", "dry-run"]], ids=["off", "dry_run"])
def test_no_web_outside_live_is_accepted_and_inert(tmp_path, capsys, monkeypatch, mode) -> None:
    """C-1: ``--no-web`` with a resolved mode other than live is accepted (exit 0) and changes nothing printed."""
    with_flag = _dry(tmp_path / "a", capsys, monkeypatch, *mode, "--no-web")
    monkeypatch.undo()
    without = _dry(tmp_path / "b", capsys, monkeypatch, *mode)
    assert with_flag.replace(str(tmp_path / "a"), "<tmp>") == without.replace(
        str(tmp_path / "b"), "<tmp>"
    )


# --------------------------------------------------------------------------------------------- C-2


def test_dry_run_line_says_web_and_no_web_says_loopback(tmp_path, capsys, monkeypatch) -> None:
    """C-2: the live ``--dry-run`` tools line says ``egress=web`` in web mode, BASE's ``egress=loopback`` with
    ``--no-web`` or ``browser_web = false``; every other ``--dry-run`` line is identical between the two."""
    web = _dry(tmp_path / "w", capsys, monkeypatch, *LIVE)
    monkeypatch.undo()
    off = _dry(tmp_path / "n", capsys, monkeypatch, *LIVE, "--no-web")
    monkeypatch.undo()
    key = _dry(tmp_path / "k", capsys, monkeypatch, *LIVE, body="browser_web = false\n")
    # T-0202 D3 re-pin: the web line read through admit_line (the default wait token and the wait set's sha12);
    # --no-web is live too, so its line carries the same token
    tools = [t0202.admit_line(x) for x in web.splitlines() if x.startswith("tools=")]
    assert len(tools) == 1 and LIVE_LINE.match(tools[0]), f"T-0160 C-2: egress=web; {tools}"
    off = t0202.admit_text(off, "--no-web")
    key = t0202.admit_text(key, "browser_web = false")
    for out in (off, key):
        lines = [x for x in out.splitlines() if x.startswith("tools=")]
        assert len(lines) == 1 and LOOPBACK_LIVE_LINE.match(lines[0]), (
            f"T-0160 C-2: --no-web is BASE's; {lines}"
        )

    def rest(out: str, tmp: Path) -> list[str]:
        return [
            x.replace(str(tmp), "<tmp>") for x in out.splitlines() if not x.startswith("tools=")
        ]

    assert rest(web, tmp_path / "w") == rest(off, tmp_path / "n") == rest(key, tmp_path / "k"), (
        "T-0160 C-2: every other --dry-run line is byte-identical"
    )


# --------------------------------------------------------------------------------------------- C-3


def test_startup_notice_once_with_every_clause(tmp_path, capsys, monkeypatch) -> None:
    """C-3: one stderr notice at a real web-mode start, holding every clause; printed after the start page opened."""
    events = Events()
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events))
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    found = _notice_lines(err)
    assert len(found) == 1, f"T-0160 C-3: one notice line; stderr:\n{err}"
    collapsed = _collapse(found[0])
    missing = [c for c in NOTICE if c not in collapsed]
    assert missing == [], f"T-0160 C-3: the notice lacks {missing}: {found[0]!r}"
    assert ("browser.open", ORIGIN + "/") in events, list(events)


def test_no_notice_with_no_web(tmp_path, capsys, monkeypatch) -> None:
    """C-3 (too wide): ``--no-web`` prints no notice and no clause of it."""
    events = Events()
    code = run_cli([*_serve_rig(tmp_path, monkeypatch, events), "--no-web"])
    out, err = _out(capsys)
    assert code == 0, f"premise: --no-web starts; {code}\n{out}{err}"
    assert not [c for c in NOTICE[:3] if c in _collapse(err)], (
        f"no notice with --no-web; stderr:\n{err}"
    )


def test_web_html_carries_the_notice_first_three_clauses(tmp_path, capsys, monkeypatch) -> None:
    """C-7: ``/web.html`` carries the C-3 notice's first three clauses, the same words as the stderr notice."""
    run_cli(_serve_rig(tmp_path, monkeypatch, Events()))
    _, err = _out(capsys)
    (line,) = _notice_lines(err) or [""]
    page = _collapse(WEB_HTML.read_text(encoding="utf-8"))
    for clause in NOTICE[:3]:
        assert clause in _collapse(line) and clause in page, f"T-0160 C-7: {clause!r} in both"


# --------------------------------------------------------------------------------------------- C-4


class LaunchError(Exception):
    pass


def test_sandbox_failure_names_readme_section_and_no_web(tmp_path, capsys, monkeypatch) -> None:
    """C-4: a web-mode launch failing with Chromium's sandbox error exits 1 before the HUD binds, with one stderr line
    naming "Real websites: the browser sandbox" and saying ``--no-web`` runs without real websites; no notice."""
    events = Events()
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events, open_error=LaunchError(SANDBOX_ERROR)))
    out, err = _out(capsys)
    assert code == 1, f"T-0160 C-4: exit 1; {code}\n{out}{err}"
    assert not [e for e in events if e[0] == "hud.start"], f"before the HUD binds; {list(events)}"
    lines = failure_lines(err)
    assert len(lines) == 1, f"T-0160 C-4: one stderr line; stderr:\n{err}"
    assert SANDBOX_SECTION in lines[0] and "--no-web" in lines[0], lines[0]
    assert "BOX" not in err and not _notice_lines(err), err
    assert ("pages.stop",) in events, list(events)


@pytest.mark.parametrize("case", ["other_error_web", "sandbox_error_no_web"])
def test_sandbox_message_only_for_the_sandbox_error_in_web_mode(
    tmp_path, capsys, monkeypatch, case
) -> None:
    """C-4 (too wide): another start failure in web mode, or the sandbox error with ``--no-web``, keeps BASE's one
    line (it names the exception type) and does not name the README section."""
    message = "no page" if case == "other_error_web" else SANDBOX_ERROR
    extra = [] if case == "other_error_web" else ["--no-web"]
    events = Events()
    code = run_cli(
        [*_serve_rig(tmp_path, monkeypatch, events, open_error=LaunchError(message)), *extra]
    )
    out, err = _out(capsys)
    assert code == 1, f"{code}\n{out}{err}"
    lines = failure_lines(err)
    assert len(lines) == 1 and "LaunchError" in lines[0], f"BASE's one line; stderr:\n{err}"
    assert SANDBOX_SECTION not in err, (
        f"T-0160 C-4: the section only for the web-mode sandbox error; {err}"
    )
    # premise (both cases): the web-mode sandbox error on the same rig does name the section, so the quiet line
    # above is the discrimination, not an absent feature
    if True:
        monkeypatch.undo()
        on = run_cli(
            _serve_rig(
                tmp_path / "on", monkeypatch, Events(), open_error=LaunchError(SANDBOX_ERROR)
            )
        )
        _, err_on = _out(capsys)
        assert on == 1 and SANDBOX_SECTION in err_on, err_on


# --------------------------------------------------------------------------------------------- amendments


def test_notice_states_spoken_words_and_unmarked_fields(tmp_path, capsys, monkeypatch) -> None:
    """Amendment (reviewer A): the notice gains one clause after the C-3 list: what the user says can end up typed
    into websites, and password or code fields a page does not mark as such are not recognised. Matched on its
    content words (the exact sentence is the implementer's), case-insensitively, after the C-3 clauses."""
    code = run_cli(_serve_rig(tmp_path, monkeypatch, Events()))
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    (line,) = _notice_lines(err) or [""]
    low = _collapse(line).casefold()
    assert "typed into" in low and "website" in low, (
        f"amendment: spoken words typed into websites; {line!r}"
    )
    assert (
        re.search(r"not (?:recognised|recognized)", low) and "password" in low and "code" in low
    ), f"amendment: unmarked password or code fields are not recognised; {line!r}"
    missing = [c for c in NOTICE if c not in _collapse(line)]
    assert missing == [], f"the C-3 clauses stay; missing {missing}"


#: main at the group C branch point: holds group B's ``web.html`` (fixed ref, .claude/rules/tests.md)
WEB_HTML_BASE = "180c1c25"


def test_web_html_unchanged_by_group_c() -> None:
    """Amendment (reviewer A) / B-5: ``/web.html``'s three sentences stay as they are: the file is byte-identical to
    group B's (main at 180c1c25) except T-0230's two substitutions (the placeholder and the hint, T-0230 R-3), and it
    does not carry the new clause."""
    import subprocess

    from tests.unit.test_t0230_start_page import t0230_substitute

    rel = "local_astra/pages/static/web.html"
    root = WEB_HTML_BASE_ROOT
    got = subprocess.run(
        ["git", "-C", str(root), "show", f"{WEB_HTML_BASE}:{rel}"], capture_output=True
    )
    if got.returncode != 0:
        pytest.skip(f"{WEB_HTML_BASE} is not in this clone")
    want = t0230_substitute(got.stdout.decode("utf-8")).encode("utf-8")
    assert WEB_HTML.read_bytes() == want, (
        "B-5: web.html changed in group C (or by more than T-0230's two substitutions)"
    )
    assert not re.search(
        r"not (?:recognised|recognized)", _collapse(WEB_HTML.read_text()).casefold()
    )


WEB_HTML_BASE_ROOT = Path(__file__).resolve().parents[2]


class _ServeHud:
    def __init__(self, session: Any, **kwargs: Any) -> None:
        self.session, self.port, self.url = session, 0, "ws://127.0.0.1:0/v1/realtime"

    async def start(self) -> None:
        import asyncio
        import os
        import signal

        asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize("mode", [[], ["--tools", "dry-run"]], ids=["off", "dry_run"])
@pytest.mark.parametrize("no_web", [False, True], ids=["default", "no_web"])
def test_real_start_outside_live_is_not_web(tmp_path, capsys, monkeypatch, mode, no_web) -> None:
    """Reviewer B: outside live, web is inert at a **real** start too (not only under ``--dry-run``): ``--tools off``
    and ``--tools dry-run``, each with and without ``--no-web`` (accepted), serve with ``graph.web`` False and print
    no clause of the notice. Premise: the same rig in live prints it."""
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json")
    monkeypatch.setattr(cli, "HudServer", _ServeHud)
    fake_live(monkeypatch, Events())
    graphs: list[Any] = []
    real_run = cli.run_session

    async def spy(graph: Any) -> int:
        graphs.append(graph)
        return await real_run(graph)

    monkeypatch.setattr(cli, "run_session", spy)
    code = run_cli([*argv(tmp_path, *mode), *(["--no-web"] if no_web else [])])
    out, err = _out(capsys)
    assert code == 0 and len(graphs) == 1, f"premise: one real start served; {code}\n{out}{err}"
    assert getattr(graphs[0], "web", False) is False, (
        f"C-1: web is inert outside live ({mode}, --no-web={no_web})"
    )
    said = [c for c in NOTICE if c in _collapse(err)]
    assert said == [], f"C-1/C-3: no notice clause outside live; {said}\n{err}"
    graphs.clear()
    (tmp_path / "live").mkdir()
    code = run_cli([*argv(tmp_path / "live", *LIVE)])
    _, err_live = _out(capsys)
    assert code == 0 and graphs and getattr(graphs[-1], "web", None) is True, "premise: live is web"
    assert NOTICE[0] in _collapse(err_live), "premise: the live rig prints the notice"
