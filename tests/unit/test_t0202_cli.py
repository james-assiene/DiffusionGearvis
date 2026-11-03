"""T-0202 group B, B8/B9 (cli side): ``astra session --tools live --wait-tool on|off`` (default off), the kit it
selects, the guidance it composes and the line it prints (docs/tasks/T-0202.md).

**The surface these rows pin** (the task's; readings stated where it is silent):

- ``--wait-tool on|off`` (one flag with a value; H-T202-7: no profile key). Outside a resolved ``live`` mode either
  value exits 2 with ``cli``'s own refusal naming ``live``, before any graph (the ``--live-guidance`` rule).
- ``cli.load_tool_kit(profile, live_browser=..., page_overview=..., wait_tool=...)``: the kit's ``tools``,
  ``tools_version`` and its tool-set field all come from one set: v1 (``tool_set`` ``None``), overview, wait, or
  overview+wait. The wait sets' shas are computed here from the files' bytes (``hashlib``), the v1 and overview
  shas are A6's literals.
- With the flag on, ``Session(tool_guidance=...)`` is ``LIVE_GUIDANCE[v] + " " + WAIT_TOOL_GUIDANCE`` (v3 without
  web: v2's text), the H-T202-4 sentence typed here from the task.
- The ``--dry-run`` tools line gains `` wait=on(flag)`` (and shows the wait set's sha12); nothing when off.

**Readings (reported):**

- R1 ("the ``--dry-run`` and start lines add `` wait=on(flag)``"): at a real start no ``tools=`` line is printed;
  the rows pin that exactly one stdout line of a real start carries the token `` wait=on(flag)`` (any line), and
  none carries ``wait=`` when off. Under ``--dry-run`` the token is on the ``tools=`` line; its position on that
  line is not pinned (removing it and swapping the sha12 gives the off line).
- R2 (B9 "byte-identical to BASE"): the row compares ``--wait-tool off`` with no flag in this tree (stdout of
  ``--dry-run``, the kit, every ``Session`` keyword) and pins the v1 sha; the full BASE subprocess differential
  is T-0191's harness and is not repeated here.

Regime: CPU, the cli rigs over fakes, nothing served, n = 1.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor import prompt_builder as pb
from local_astra.config import Profile
from local_astra.toolcall import native, tools
from tests.unit import _t0160_pins as t0160
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, SHA12, dry_run, tools_lines
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_cli_t0132 import build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import _names_live

FLAG = "--wait-tool"
TOOLCALL = Path(tools.__file__).resolve().parent
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
#: H-T202-4, typed from the task
SENTENCE = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)
TOKEN = " wait=on(flag)"
#: T-0202 D3 re-pin: D1's token for a live run with no --wait-tool flag (the default on)
DEFAULT_TOKEN_D = " wait=on(default)"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'


def file_sha(name: str) -> str:
    return hashlib.sha256((TOOLCALL / name).read_bytes()).hexdigest()


def kit_fields(kit: Any) -> tuple[Any, ...]:
    """A kit compared field by field: ``ToolKit`` is ``eq=False`` (identity), so ``==`` between two built kits is
    always ``False``. The executor and browser by identity, the set by value."""
    return (
        kit.mode,
        kit.tools,
        kit.tools_version,
        id(kit.executor),
        id(kit.live_browser),
        kit.payloads is None,
        kit.calibration is None,
        kit.repair_off,
        kit_set(kit),
    )


def dry(tmp: Path, capsys: Any, monkeypatch: Any, *extra: str) -> str:
    """``dry_run``'s stdout with its own tmp directory spelled ``<tmp>`` (the data dir is printed), so two runs in
    two directories compare byte for byte otherwise."""
    out = dry_run(tmp, capsys, monkeypatch, "live", *extra)
    monkeypatch.undo()
    assert str(tmp) in out, "premise: the dry-run line names the run's own directory"
    return out.replace(str(tmp), "<tmp>")


def session_help() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        run_cli(["session", "--help"])
    return buf.getvalue()


def require() -> None:
    flags = set(flags_of(CLI)) | set(re.findall(r"--[a-z][a-z-]+", session_help()))
    assert FLAG in flags, "T-0202 B8: astra session --wait-tool does not exist yet"


def kit_set(kit: Any) -> Any:
    from tests.unit.test_t0140_session import kit_set_field

    return getattr(kit, kit_set_field())


#: (wait flag, overview flag) -> the file the kit comes from (None: v1)
SETS = {
    ("on", False): "tools_v1_wait.json",
    ("on", True): "tools_v1_overview_wait.json",
    ("off", False): None,
    ("off", True): "tools_v1_overview.json",
}


@pytest.mark.parametrize(
    "wait,overview", list(SETS), ids=["wait", "overview+wait", "v1", "overview"]
)
def test_flag_selects_set(tmp_path, capsys, monkeypatch, wait: str, overview: bool) -> None:
    """B8: the four combinations with ``--page-overview``: the cli-built kit's ``tools``, ``tools_version`` and tool
    set all come from one file; v1 is BASE's kit (``tool_set`` ``None``, sha bbfc53cc...)."""
    require()
    flags = [FLAG, wait, *(["--page-overview"] if overview else [])]
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags)
    kit = calls.kit()
    name = SETS[(wait, overview)]
    if name is None:
        assert kit.tools_version == V1_SHA and kit_set(kit) is None, (
            kit.tools_version,
            kit_set(kit),
        )
        assert kit.tools == tuple(native.tools_field(tools.load()))
        return
    sha = file_sha(name)
    if name == "tools_v1_overview.json":
        assert sha == OV_SHA
    want = tools.load(TOOLCALL / name)
    assert want.version == sha, "premise: a set's version is its file's sha256"
    assert kit.tools_version == sha, (
        f"T-0202 B8 {flags}: tools_version {kit.tools_version} != {name}'s {sha}"
    )
    assert kit.tools == tuple(native.tools_field(want)), f"T-0202 B8 {flags}: tools are {name}'s"
    js = kit_set(kit)
    assert js is not None and js.version == sha, (
        f"T-0202 B8 {flags}: the judge's set is {name}'s; {js!r}"
    )
    assert ("wait" in js.names) is (wait == "on"), js.names


def test_load_tool_kit_keyword(tmp_path, monkeypatch) -> None:
    """B8: ``load_tool_kit(profile, live_browser=..., page_overview=..., wait_tool=...)`` directly; with
    ``wait_tool=False`` (or absent) the kit equals BASE's call without the keyword."""
    import inspect

    assert "wait_tool" in inspect.signature(cli.load_tool_kit).parameters, (
        "T-0202 B8: load_tool_kit(..., wait_tool=...) does not exist yet"
    )
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    browser = object()
    plain, _ = cli.load_tool_kit(profile, live_browser=browser)  # type: ignore[arg-type]
    off, _ = cli.load_tool_kit(profile, live_browser=browser, wait_tool=False)  # type: ignore[arg-type]
    assert kit_fields(off) == kit_fields(plain), (kit_fields(off), kit_fields(plain))
    assert plain.tools_version == V1_SHA and kit_set(plain) is None
    on, _ = cli.load_tool_kit(profile, live_browser=browser, wait_tool=True)  # type: ignore[arg-type]
    both, _ = cli.load_tool_kit(profile, live_browser=browser, page_overview=True, wait_tool=True)  # type: ignore[arg-type]
    assert on.tools_version == file_sha("tools_v1_wait.json") == kit_set(on).version
    assert both.tools_version == file_sha("tools_v1_overview_wait.json") == kit_set(both).version
    assert on.mode == both.mode == "live" and on.executor is browser


GUIDANCE_CASES = {
    "v1": (["--live-guidance", "v1"], "v1"),
    "v2": (["--live-guidance", "v2"], "v2"),
    "v3": (["--live-guidance", "v3"], "v3"),
    "v3-no-web": (["--live-guidance", "v3", "--no-web"], "v2"),
}


@pytest.mark.parametrize("case", list(GUIDANCE_CASES))
def test_guidance_composed(tmp_path, capsys, monkeypatch, case: str) -> None:
    """B8: with the flag on the session's ``tool_guidance`` is the version's text, one space, then H-T202-4's
    sentence (v3 under ``--no-web`` carries v2's text); with it off, the version's text alone."""
    require()
    assert pb.WAIT_TOOL_GUIDANCE == SENTENCE, "premise: A4's constant is H-T202-4"
    flags, text_of = GUIDANCE_CASES[case]
    base = pb.LIVE_GUIDANCE[text_of]
    calls = build_mode(tmp_path / "on", capsys, monkeypatch, "live", *flags, FLAG, "on")
    got = calls.session_kwargs[-1].get("tool_guidance")
    assert got == base + " " + SENTENCE, f"T-0202 B8 {case}: tool_guidance {got!r}"
    monkeypatch.undo()
    calls = build_mode(tmp_path / "off", capsys, monkeypatch, "live", *flags, FLAG, "off")
    assert calls.session_kwargs[-1].get("tool_guidance") == base
    assert SENTENCE not in calls.graphs[-1].session.prompts.system_message().text()


def test_start_line_token(tmp_path, capsys, monkeypatch) -> None:
    """B8: ``--dry-run``: the tools line with the flag on is the off line plus `` wait=on(flag)`` with the wait set's
    sha12 in place of v1's; with ``--page-overview`` too, the overview+wait sha12. A real start prints the token on
    exactly one line; off prints no ``wait=`` anywhere. T-0202 D3 re-pin (D1): the default is the flag-on output
    with `` wait=on(default)`` in place of `` wait=on(flag)`` (B: the default was off's)."""
    require()
    off = dry(tmp_path / "off", capsys, monkeypatch, FLAG, "off")
    default = dry(tmp_path / "def", capsys, monkeypatch)
    on = dry(tmp_path / "on", capsys, monkeypatch, FLAG, "on")
    # T-0202 D3 re-pin (D's default, D1): the default now prints the on line with " wait=on(default)" in place of
    # B's flag token, so "off == default" becomes "the default is the flag-on output with the token renamed", and
    # off is still BASE's (no wait=)
    assert "wait=" not in off and default == on.replace(TOKEN, DEFAULT_TOKEN_D), (default, on)
    assert sum(x.split().count(DEFAULT_TOKEN_D.strip()) for x in default.splitlines()) == 1, default
    (line_off,) = tools_lines(off)
    (line_on,) = tools_lines(on)
    assert line_off == t0160.web_line(BASE_LINES["live"]) + " visual_tokens=1120", line_off
    assert line_on.count(TOKEN) == 1, (
        f"T-0202 B8: the dry-run tools line carries {TOKEN!r}: {line_on}"
    )
    sha12 = file_sha("tools_v1_wait.json")[:12]
    assert line_on.replace(TOKEN, "").replace(sha12, SHA12) == line_off, (line_on, line_off)
    rest_on = [x for x in on.splitlines() if not x.startswith("tools=")]
    rest_off = [x for x in off.splitlines() if not x.startswith("tools=")]
    assert rest_on == rest_off, "T-0202 B8: the flag changes only the tools line under --dry-run"
    both = dry_run(tmp_path / "both", capsys, monkeypatch, "live", FLAG, "on", "--page-overview")
    monkeypatch.undo()
    (line_both,) = tools_lines(both)
    assert (
        f"native:{file_sha('tools_v1_overview_wait.json')[:12]}" in line_both and TOKEN in line_both
    )
    _, started, code = build_out(
        tmp_path / "s", capsys, monkeypatch, *MODE_FLAGS["live"], FLAG, "on"
    )
    assert code == 0, started
    assert sum(1 for x in started.splitlines() if TOKEN.strip() in x.split()) == 1, started
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s0", capsys, monkeypatch, *MODE_FLAGS["live"])
    # T-0202 D3 re-pin (D's default, D1): a default real start carries exactly one wait= token, wait=on(default)
    # (B: none); --wait-tool off still prints no wait= anywhere
    start_tokens = [t for x in started.splitlines() for t in x.split() if t.startswith("wait=")]
    assert code == 0 and start_tokens == [DEFAULT_TOKEN_D.strip()], started
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "s-off", capsys, monkeypatch, *MODE_FLAGS["live"], FLAG, "off"
    )
    assert code == 0 and "wait=" not in started, started


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode: str, value: str) -> None:
    """B8: ``--wait-tool on|off`` with tools off or dry-run (also a live profile overridden by ``--tools``) exits 2
    naming ``live`` and builds nothing; with ``--dry-run`` and without."""
    require()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        for extra in (["--dry-run"], []):
            monkeypatch.setenv("DISPLAY", ":99")
            calls = T.Calls()
            T.fake_clients(monkeypatch, calls)
            T.stop_before_serving(monkeypatch, calls)
            T.patch_detok_load(monkeypatch, calls, tmp_path)
            fake_live(monkeypatch, Events())
            code = run_cli([*argv(tmp_path, *flags, FLAG, value, config_body=body), *extra])
            out, err = _out(capsys)
            low = (out + err).replace("--wait-tool", "")
            assert code == 2 and _names_live(low) and "unrecognized" not in err, (code, out, err)
            assert not calls.graphs and not calls.session_kwargs
            monkeypatch.undo()


def test_off_byte_identical_to_base(tmp_path, capsys, monkeypatch) -> None:
    """B9 (negative, reading R2): ``--wait-tool off`` gives the same ``--dry-run`` stdout, the same kit and the same
    ``Session`` keywords as no flag (v1's sha, BASE's guidance), with and without ``--page-overview``.

    T-0202 D3 re-pin (D's default, D1/D2): no flag is now ``--wait-tool on``'s build (``a`` below), and
    ``--wait-tool off`` (``b``) is BASE's: v1's (or overview's) kit and sha, the guidance without the sentence. So
    each equality between no flag and off is restated as no flag = on, and off = BASE's literal values; the default
    differs from off by exactly the set, the one sentence and the one printed token."""
    require()
    for extra in ([], ["--page-overview"]):
        a = build_mode(tmp_path / f"a{len(extra)}", capsys, monkeypatch, "live", *extra)
        monkeypatch.undo()
        b = build_mode(
            tmp_path / f"b{len(extra)}", capsys, monkeypatch, "live", *extra, FLAG, "off"
        )
        monkeypatch.undo()
        on = build_mode(  # T-0202 D3 re-pin: the flag-on build the default now equals
            tmp_path / f"on{len(extra)}", capsys, monkeypatch, "live", *extra, FLAG, "on"
        )
        monkeypatch.undo()
        ka, kb, kon = a.kit(), b.kit(), on.kit()
        assert (ka.tools, ka.tools_version, kit_set(ka)) == (  # T-0202 D3 re-pin: default = on
            kon.tools,
            kon.tools_version,
            kit_set(kon),
        )
        # T-0202 D3 re-pin: off is BASE's set (v1, or overview with --page-overview); the default is the wait set
        assert kb.tools_version == (OV_SHA if extra else V1_SHA)
        wait_name = "tools_v1_overview_wait.json" if extra else "tools_v1_wait.json"
        assert ka.tools_version == file_sha(wait_name), (ka.tools_version, wait_name)
        skip = {"tool_kit", "clock"}
        wa = {k: repr(v) for k, v in a.session_kwargs[-1].items() if k not in skip}
        wb = {k: repr(v) for k, v in b.session_kwargs[-1].items() if k not in skip}
        # T-0202 D3 re-pin: the same keywords; the default's guidance is off's plus one space and the sentence
        ga, gb = a.session_kwargs[-1]["tool_guidance"], b.session_kwargs[-1]["tool_guidance"]
        assert wa.keys() == wb.keys() and ga == gb + " " + SENTENCE, (
            wa.keys() ^ wb.keys(),
            ga[-200:],
        )
        assert gb == pb.LIVE_GUIDANCE["v3"] and SENTENCE not in gb, gb[-200:]
        x = dry(tmp_path / f"x{len(extra)}", capsys, monkeypatch, *extra)
        y = dry(tmp_path / f"y{len(extra)}", capsys, monkeypatch, *extra, FLAG, "off")
        # T-0202 D3 re-pin: the default's dry-run stdout is off's with the token appended to the tools line and the
        # wait set's sha12 in place of off's set's sha12, nothing else
        off_sha12 = (OV_SHA if extra else V1_SHA)[:12]
        (line_x,) = tools_lines(x)
        (line_y,) = tools_lines(y)
        assert line_x.split().count(DEFAULT_TOKEN_D.strip()) == 1, line_x
        assert (
            line_x.replace(DEFAULT_TOKEN_D, "").replace(file_sha(wait_name)[:12], off_sha12)
            == line_y
        ), (line_x, line_y)
        assert x.replace(line_x, line_y) == y, (x, y)
