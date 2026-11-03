"""T-0202 group D: the wait tool on by default in live sessions (web and ``--no-web``), ``--wait-tool off`` restores
BASE, never on outside live (docs/tasks/T-0202.md D1-D3; C's GPU A/B PASSED 2026-10-30 at eebbbf8b: probe ON 8/10 vs
OFF 1/10, suite 17 vs 18, safety 0).

**The surface these rows pin** (the task's; readings stated where it is silent):

- A default live build (no ``--wait-tool``) gets B's wait kit: ``tools_v1_wait.json`` (or
  ``tools_v1_overview_wait.json`` with ``--page-overview``), and ``Session(tool_guidance=...)`` is the resolved
  guidance text, one space, then H-T202-4's sentence (typed here from the task). ``--wait-tool on`` gives the same
  kit and guidance. ``--wait-tool off`` gives BASE's v1 kit (``tool_set`` ``None``, sha bbfc53cc...) and guidance.
- D1's start line: the token `` wait=on(default)`` with no flag, `` wait=on(flag)`` with ``--wait-tool on``; no
  ``wait=`` anywhere with ``--wait-tool off`` or outside live. R-D1 (reading): the token sits where B put
  `` wait=on(flag)``: on the ``--dry-run`` tools line, and on exactly one stdout line of a real start.
- R-D2 (reading, "never on outside live"): ``off``/``dry_run`` (plain, and a live profile overridden by ``--tools``)
  build no wait kit, compose no sentence and print no ``wait=``; B8's refusal of the flag there stands
  (``test_t0202_cli.test_flag_refused_outside_live``).
- D2 (negative): ``off``, ``dry_run`` and live ``--wait-tool off`` are compared with BASE c9ecd1db in BASE's own
  tree (``git archive``, origin-proved): ``--dry-run`` output, a build's output, client and session keywords,
  executors, the system message and one request body, with no admitted difference.
- Adversarial (the failure mode named by D's blast radius: "changes the tools field and the system prompt of every
  live request"): a default live build differs from BASE's default live build in exactly the tools field (the wait
  set's), the guidance suffix in the system message, the tools sha in printed lines and the start token, nothing else.

Regime: CPU over fakes, nothing served, thinking on, tools ``live`` (headless, 1280x800), n = 1, deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.toolcall import native, tools
from tests import _git_range as gr
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0133_live_reground_off as t0133
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run, tools_lines
from tests.unit.test_cli_t0131 import build as build_mode
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0202_cli import kit_set

#: BASE: main at c9ecd1db, where group D branched (wait tool off by default). A literal, never a moving ref.
BASE_SHA = "c9ecd1db"

FLAG = "--wait-tool"
LIVE = MODE_FLAGS["live"]
TOOLCALL = Path(tools.__file__).resolve().parent
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
#: H-T202-4, typed from the task
SENTENCE = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)
DEFAULT_TOKEN = "wait=on(default)"
FLAG_TOKEN = "wait=on(flag)"


def file_sha(name: str) -> str:
    return hashlib.sha256((TOOLCALL / name).read_bytes()).hexdigest()


def build_live(tmp: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = "") -> Any:
    calls = build_mode(tmp, capsys, monkeypatch, "live" if not body else "off", *flags, body=body)
    monkeypatch.undo()
    return calls


def kit_of(calls: Any) -> Any:
    kit = calls.session_kwargs[-1].get("tool_kit")
    assert kit is not None, "premise: cli built a tool kit"
    return kit


def guidance_of(calls: Any) -> Any:
    return calls.session_kwargs[-1].get("tool_guidance")


def assert_wait_kit(kit: Any, name: str, where: str) -> None:
    sha = file_sha(name)
    want = tools.load(TOOLCALL / name)
    assert kit.tools_version == sha, (
        f"T-0202 D1 ({where}): the live kit is {name}'s (sha {sha[:12]}); got {kit.tools_version[:12]}"
        + (" = v1, the wait tool is off" if kit.tools_version == V1_SHA else "")
    )
    assert kit.tools == tuple(native.tools_field(want)), (
        f"T-0202 D1 ({where}): the tools field is {name}'s"
    )
    js = kit_set(kit)
    assert js is not None and js.version == sha and "wait" in js.names, (
        f"T-0202 D1 ({where}): the judge's set is {name}'s and names wait; {js!r}"
    )


def assert_v1_kit(kit: Any, where: str) -> None:
    assert kit.tools_version == V1_SHA and kit_set(kit) is None, (
        f"T-0202 ({where}): BASE's v1 kit (sha bbfc53cc, tool_set None); got {kit.tools_version[:12]}, {kit_set(kit)!r}"
    )
    assert "wait" not in json.dumps(list(kit.tools)).replace("wait for", ""), (
        f"T-0202 ({where}): no wait declaration in the v1 tools field"
    )


def tokens(text: str) -> list[str]:
    """Every whitespace token in ``text`` that starts with ``wait=``."""
    return [t for line in text.splitlines() for t in line.split() if t.startswith("wait=")]


# --------------------------------------------------------------------------------------------- D1: the default


@pytest.mark.parametrize(
    "case,flags,text_of",
    [
        ("web", [], "v3"),
        ("no-web", ["--no-web"], "v2"),
    ],
)
def test_live_default_on(tmp_path, capsys, monkeypatch, case, flags, text_of):
    """D1: a default live build (no ``--wait-tool``), web and ``--no-web``: the session's kit is the wait set's
    (tools field, sha, judge set naming ``wait``), and its ``tool_guidance`` is the default guidance text (v3; v2's
    text under ``--no-web``), one space, then H-T202-4's sentence."""
    assert pb.WAIT_TOOL_GUIDANCE == SENTENCE, "premise: A4's constant is H-T202-4"
    calls = build_live(tmp_path / case, capsys, monkeypatch, *flags)
    assert_wait_kit(kit_of(calls), "tools_v1_wait.json", f"default live {case}")
    got = guidance_of(calls)
    want = pb.LIVE_GUIDANCE[text_of] + " " + SENTENCE
    assert got == want, (
        f"T-0202 D1 ({case}): default tool_guidance carries H-T202-4's sentence; got {got!r}"
    )
    system = calls.graphs[-1].session.prompts.system_message().text()
    assert system.count(SENTENCE) == 1, (
        f"T-0202 D1 ({case}): the system message carries the sentence once"
    )


def test_live_profile_default_on(tmp_path, capsys, monkeypatch):
    """D1: a live profile (``tools = "live"`` in the config, no ``--tools``) gets the wait kit and the sentence too."""
    calls = build_live(tmp_path / "profile", capsys, monkeypatch, body=t0134c.LIVE_PROFILE)
    assert kit_of(calls).mode == "live", "premise: the live profile built a live kit"
    assert_wait_kit(kit_of(calls), "tools_v1_wait.json", "live profile")
    assert guidance_of(calls).endswith(" " + SENTENCE), guidance_of(calls)


def test_overview_default_gets_overview_wait_set(tmp_path, capsys, monkeypatch):
    """D1 with B8's set table: ``--page-overview`` and no ``--wait-tool`` is the overview+wait set."""
    calls = build_live(tmp_path / "ov", capsys, monkeypatch, "--page-overview")
    assert_wait_kit(kit_of(calls), "tools_v1_overview_wait.json", "default live --page-overview")


def test_flag_off_turns_it_off(tmp_path, capsys, monkeypatch):
    """D1: ``--wait-tool off`` gives BASE's v1 kit and BASE's guidance (no sentence), web and ``--no-web``;
    ``--wait-tool on`` gives the same kit and guidance as the default. Power: the default differs from off."""
    for case, extra, text_of in (("web", [], "v3"), ("no-web", ["--no-web"], "v2")):
        off = build_live(tmp_path / f"off-{case}", capsys, monkeypatch, *extra, FLAG, "off")
        assert_v1_kit(kit_of(off), f"--wait-tool off {case}")
        assert guidance_of(off) == pb.LIVE_GUIDANCE[text_of], guidance_of(off)
        system = off.graphs[-1].session.prompts.system_message().text()
        assert SENTENCE not in system, f"T-0202 D1 ({case}): --wait-tool off sends no wait sentence"
        on = build_live(tmp_path / f"on-{case}", capsys, monkeypatch, *extra, FLAG, "on")
        default = build_live(tmp_path / f"def-{case}", capsys, monkeypatch, *extra)
        k_on, k_def = kit_of(on), kit_of(default)
        assert (k_on.tools, k_on.tools_version) == (k_def.tools, k_def.tools_version), (
            f"T-0202 D1 ({case}): --wait-tool on and the default give one kit"
        )
        assert guidance_of(on) == guidance_of(default), case
        assert k_def.tools_version != kit_of(off).tools_version, (
            f"T-0202 D1 ({case}): the default is on, so it differs from --wait-tool off"
        )


def test_help_states_default_on():
    """D1: ``--wait-tool``'s help no longer says the default is off, and says it is on."""
    h = t0133._help_of(FLAG)
    low = " ".join(h.lower().split())
    assert "default off" not in low, f"T-0202 D1: the help still states the old default: {h!r}"
    assert re.search(r"default[:;,]? on\b", low), (
        f"T-0202 D1: the help states the default on: {h!r}"
    )


# --------------------------------------------------------------------------------------------- D1: the start line


def test_start_line_source(tmp_path, capsys, monkeypatch):
    """D1/R-D1: under ``--dry-run`` the tools line carries `` wait=on(default)`` exactly once with no flag and
    `` wait=on(flag)`` with ``--wait-tool on`` (each with the wait set's sha12); ``--wait-tool off`` prints no
    ``wait=``. At a real start exactly one stdout token, ``wait=on(default)`` with no flag, ``wait=on(flag)`` with
    the flag; none with ``--wait-tool off``."""
    sha12 = file_sha("tools_v1_wait.json")[:12]
    default = dry_run(tmp_path / "def", capsys, monkeypatch, "live")
    monkeypatch.undo()
    on = dry_run(tmp_path / "on", capsys, monkeypatch, "live", FLAG, "on")
    monkeypatch.undo()
    off = dry_run(tmp_path / "off", capsys, monkeypatch, "live", FLAG, "off")
    monkeypatch.undo()
    (line_def,) = tools_lines(default)
    (line_on,) = tools_lines(on)
    assert tokens(default) == [DEFAULT_TOKEN] and DEFAULT_TOKEN in line_def.split(), (
        f"T-0202 D1: the default --dry-run tools line carries {DEFAULT_TOKEN!r} once: {tokens(default)}\n{default}"
    )
    assert f"native:{sha12}" in line_def, (
        f"T-0202 D1: the default tools line shows the wait set's sha12: {line_def}"
    )
    assert tokens(on) == [FLAG_TOKEN] and FLAG_TOKEN in line_on.split(), (tokens(on), on)
    assert line_on.replace(FLAG_TOKEN, DEFAULT_TOKEN) == line_def, (line_on, line_def)
    assert tokens(off) == [], f"T-0202 D1: --wait-tool off prints no wait= token: {tokens(off)}"
    for flags, want in (([], [DEFAULT_TOKEN]), ([FLAG, "on"], [FLAG_TOKEN]), ([FLAG, "off"], [])):
        _, started, code = build_out(
            tmp_path / f"s{len(flags)}{want}", capsys, monkeypatch, *LIVE, *flags
        )
        monkeypatch.undo()
        assert code == 0, started
        assert tokens(started) == want, (
            f"T-0202 D1 (start, {flags}): tokens {tokens(started)} != {want}\n{started}"
        )


# --------------------------------------------------------------------------------------------- D1: never outside live


def test_not_wired_outside_live(tmp_path, capsys, monkeypatch):
    """D1 "never on outside live" (R-D2): ``off`` and ``dry_run`` (plain, and a live profile under ``--tools off``/
    ``dry-run``), with no flag: no wait kit (``dry_run``'s kit is v1), no sentence in any ``tool_guidance``, no
    ``wait=`` printed under ``--dry-run``. Power row: the live default does get the wait kit."""
    power = build_live(tmp_path / "power", capsys, monkeypatch)
    assert kit_of(power).tools_version == file_sha("tools_v1_wait.json"), (
        "T-0202 D1: the live default carries the wait kit (power row: the rest of this row is not vacuous)"
    )
    cases = {
        "off": ([], ""),
        "dry_run": (["--tools", "dry-run"], ""),
        "live-profile-off": (["--tools", "off"], t0134c.LIVE_PROFILE),
        "live-profile-dry": (["--tools", "dry-run"], t0134c.LIVE_PROFILE),
    }
    for case, (flags, body) in cases.items():
        out = dry_run(tmp_path / f"d-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        assert tokens(out) == [], f"T-0202 D1: no wait= token in {case}\n{out}"
        calls = build_mode(tmp_path / f"b-{case}", capsys, monkeypatch, "off", *flags, body=body)
        monkeypatch.undo()
        kw = calls.session_kwargs[-1]
        assert SENTENCE not in str(kw.get("tool_guidance")), (
            f"T-0202 D1: no wait sentence in {case}"
        )
        kit = kw.get("tool_kit")
        if kit is not None:
            assert kit.mode != "live", f"premise: {case} is not live"
            assert_v1_kit(kit, f"{case}")
        system = calls.graphs[-1].session.prompts.system_message().text()
        assert SENTENCE not in system, f"T-0202 D1: no wait sentence in {case}'s system message"


# --------------------------------------------------------------------------------------------- D2: BASE differentials


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0134c_live_feedback_default as m
out = pathlib.Path(sys.argv[1])
other = m._other_modes(out.parent / "run" / "other")
m.OTHER = (("live", (*m.MODE_FLAGS["live"], *m.VT1120), ""),)
live = m._other_modes(out.parent / "run" / "live")
out.write_text(json.dumps({"origin": local_astra.__file__, "other": other, "live": live}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture in BASE's tree (T-0135b's ``git archive`` harness through ``tsl._base_tree``): T-0134c's
    ``OTHER`` cases and a default live build (wait tool off at BASE)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0202d_base")
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
    return {"other": data["other"], "live": data["live"]}


KEYS = ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body")


def _compare(now: dict[str, Any], want: dict[str, Any], case: str) -> None:
    for key in KEYS:
        assert now[key] == want[key], (
            f"T-0202 D2: {case} {key} differs from {BASE_SHA}:\n"
            f"base {json.dumps(want[key])[:1500]}\nnow  {json.dumps(now[key])[:1500]}"
        )


def _live_now(tmp: Path, *flags: str, pin_check_off: bool = False) -> dict[str, Any]:
    run = [*LIVE, *t0134c.VT1120, *flags]
    # T-0207 C3 re-pin: ``pin_check_off`` (BASE differentials) appends --completion-check off on a tree with that
    # flag, so the session is BASE's; its one "off (flag)" line and a False keyword are set aside below
    check_pinned = pin_check_off and t0207.pins(run)
    if check_pinned:
        run = t0207.pin_off(run)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t0134c, "OTHER", (("live", tuple(run), ""),))
        now = t0134c._other_modes(tmp)["live"]
    if check_pinned:
        now = t0207.section_without_off_pin(now, f"T-0202 live {flags}")  # T-0207 C3 re-pin
    return now


def test_other_modes_byte_identical_to_base(base: dict[str, Any], tmp_path: Path) -> None:
    """D2 (negative, active: green before and after): ``off`` and ``dry_run`` (T-0134c's ``OTHER``: plain, loop off, a
    live profile under ``--tools off``/``--tools dry-run``): ``--dry-run`` output, a build's output, client and
    session keywords, executors, the system message and one request body equal BASE's, with no admitted difference.
    Power row: BASE's live capture differs from its ``off`` capture."""
    now = t0134c._other_modes(tmp_path / "other")
    assert set(now) == set(base["other"]) and len(now) == len(t0134c.OTHER) == 5, sorted(now)
    for case in now:
        _compare(now[case], base["other"][case], case)
    assert base["live"]["live"]["session_kwargs"] != base["other"]["off"]["session_kwargs"], (
        "power: a live build's session keywords differ from off's"
    )


def test_flag_off_live_build_is_base(base: dict[str, Any], tmp_path: Path) -> None:
    """D2 (negative, active: green before and after): live ``--wait-tool off`` equals BASE's default live build
    (wait tool off at BASE) in every captured section: no ``wait=``, no sentence, v1's tools field."""
    _compare(
        _live_now(tmp_path / "off", FLAG, "off", pin_check_off=True),  # T-0207 C3 re-pin: pin off
        base["live"]["live"],
        "live --wait-tool off",
    )


def _tools_field(body: dict[str, Any]) -> Any:
    assert "tools" in body, f"premise: a live request body has a tools field; keys {sorted(body)}"
    return body["tools"]


def test_default_live_differs_from_base_only_by_wait(base: dict[str, Any], tmp_path: Path) -> None:
    """D1 + adversarial (D's blast radius: every live request's tools field and system prompt change): the default
    live build differs from BASE's default live build in exactly
    - the request body's ``tools`` field, which is the wait set's (``native.tools_field`` of ``tools_v1_wait.json``),
      everything else in the body equal once the system text's suffix is removed;
    - the system message, which is BASE's with `` <sentence>`` inserted once (removing it gives BASE's);
    - printed lines: the v1 sha12 becomes the wait set's and one `` wait=on(default)`` token is added;
    - the session's ``tool_guidance`` keyword gains the suffix and its kit is the wait set's.
    """
    now = _live_now(tmp_path / "def")
    want = base["live"]["live"]
    want_tools = json.loads(
        json.dumps(list(native.tools_field(tools.load(TOOLCALL / "tools_v1_wait.json"))))
    )
    assert _tools_field(now["body"]) == want_tools, (
        "T-0202 D1: the default live body's tools are the wait set's"
    )
    assert _tools_field(want["body"]) != want_tools, "power: BASE's live body has v1's tools"
    assert now["system"].count(SENTENCE) == 1 and SENTENCE not in want["system"], (
        "T-0202 D1: the default live system message carries H-T202-4 once; BASE's none"
    )
    assert now["system"].replace(" " + SENTENCE, "", 1) == want["system"], (
        "T-0202 D1 (adversarial): the system message changes only by the sentence suffix"
    )
    body_now = json.loads(json.dumps(now["body"]).replace(json.dumps(" " + SENTENCE)[1:-1], "", 1))
    body_now.pop("tools")
    body_base = dict(want["body"])
    body_base.pop("tools")
    assert body_now == body_base, (
        "T-0202 D1 (adversarial): the request body changes only in tools and the suffix"
    )
    sha_now, sha_base = file_sha("tools_v1_wait.json")[:12], V1_SHA[:12]
    for key in ("dry", "build"):
        # T-0207 C3 re-pin (admit): the one completion check default line (T-0207 C1) is the only other printed
        # difference; required exactly once on a tree with the flag, identity before
        text = t0207.admit_default_line(now[key]["out"], f"T-0202 D1 default live {key}")
        assert tokens(text) == [DEFAULT_TOKEN], (
            f"T-0202 D1 ({key}): one {DEFAULT_TOKEN} token: {tokens(text)}"
        )
        lines = []
        for line in text.split("\n"):
            parts = line.split(" ")
            lines.append(" ".join(p for p in parts if p != DEFAULT_TOKEN))
        restored = "\n".join(lines).replace(sha_now, sha_base)
        assert restored == want[key]["out"], (
            f"T-0202 D1 (adversarial, {key}): output differs from BASE beyond the sha12 and the token\n"
            f"base {want[key]['out']}\nnow  {text}"
        )
        assert now[key]["err"] == want[key]["err"] and now[key]["code"] == want[key]["code"], key
    assert now["vllm_kwargs"] == want["vllm_kwargs"] and now["executors"] == want["executors"]
    # T-0207 C3 re-pin (admit): a live session's completion_check=True keyword (T-0207 C1) set aside; a truthy one
    # anywhere else, or a second difference, still fails below
    (kw_now,) = t0207.kwargs_without_default(now["session_kwargs"])
    (kw_base,) = want["session_kwargs"]
    assert kw_now["tool_guidance"] == kw_base["tool_guidance"] + " " + SENTENCE, kw_now[
        "tool_guidance"
    ]
    rest_now = {k: v for k, v in kw_now.items() if k not in ("tool_guidance", "tool_kit")}
    rest_base = {k: v for k, v in kw_base.items() if k not in ("tool_guidance", "tool_kit")}
    assert rest_now == rest_base, "T-0202 D1 (adversarial): no other Session keyword changes"
