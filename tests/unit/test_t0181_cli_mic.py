"""T-0181 group A, A3 through ``cli``: availability is decided once, at construction, and ``cli`` says why a session
has no mic; every printed line stays BASE's (docs/tasks/T-0181.md A3, readings A-R1, A-R7, A-R13 and A-R15).

**Oracles, by hand from A3:**

- ``--silent``: ``"this session is silent: no audio device is open (--silent)"``;
- ``--no-mic``: ``"this session has no microphone (--no-mic)"``;
- ``--mic none``: ``"this session has no microphone (--mic none)"``;
- a resolved node (``--mic <node>``, ``ASTRA_AUDIO_SOURCE``, the graph's default source): available, so ``cli`` does
  **not** pass the keyword at all (A-R13: "cli passes ``mic_unavailable`` only when the mic is unavailable, never
  ``None``; an available-mic construction is byte-identical to BASE"). These rows are plain guard rows, not pending:
  they hold at BASE and must keep holding;
- ``Session(..., mic_unavailable=None)`` (no ``cli``) is still "derived", the same as omitting it (A-R7);
- an audio object with no capture seam (``Session`` alone, no ``cli``): ``"this audio path has no capture seam"``.

**Why the reason must come from ``cli``** (the adversarial half): every audio object ``cli`` builds is a
``CaptureSource`` -- ``--silent`` builds ``PipeWireAudioIO`` over a ``NullAudioDevice`` and ``--no-mic`` an
output-only ``PipeWireAudioIO``, both with ``stt_blocks`` -- so a ``Session`` that only derived availability would
call a silent or mic-less session available. The real-start rows give the session ``FakeAudioIO`` (also a
``CaptureSource``) in place of the device, so only the keyword can make it unavailable.

**Readings** (listed in the test-author's report):

- ``--silent --no-mic`` and ``--silent --mic none`` (the CLI rigs' usual argv is the first) give the ``--silent``
  reason: ``resolve_mic`` (T-0045) puts ``--silent`` first, and A3 says ``cli`` passes the *resolved* microphone
  (ruled A-R15);
- ``--no-mic`` / ``--mic none`` with ``ASTRA_AUDIO_SOURCE`` exported keep their reason (T-0045 ruling A14: the flag
  beats the variable);
- "every printed line is byte-identical to BASE" is checked on ``--dry-run`` output and on a real start stopped before
  serving (exit code, stdout, stderr); an available mic's ``Session`` keywords equal BASE's exactly, an unavailable
  one's once ``mic_unavailable`` is removed by the shared normaliser ``_t0181_pins.without_mic_unavailable`` (the
  task's "cli.py -- A: one keyword at ``Session(...)`` construction, nothing printed").

**BASE** is main at 74ae8b25 (T-0181 A's branch point; a literal, ``.claude/rules/tests.md``), run in a subprocess over
``git archive`` (T-0114's ``_base_tree``, origin checked), ``tests`` from this checkout. The unavailable differential
rows are red until A lands only through their anti-vacuity half (the TIP ``Session`` got the keyword); the available
ones are guard rows, green now.

Regime: CPU, ``cli.main`` over the CLI rig's fakes (``tests/unit/_tools_switch_support.py``), the device replaced by
``FakeAudioIO``, the default source pinned to the recorded box's node, nothing served, profile ``lean`` from a
temporary ``astra.toml``; the ``Session`` rows over ``_t0181_rig.mic_rig``; n = 1, deterministic.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import _t0197_pins as t0197  # T-0197 pin move
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit._mic_rig import BOX_MIC, ENV_MIC, ENV_SOURCE, RIG_MIC, pin_default_source
from tests.unit._t0181_pins import without_mic_unavailable
from tests.unit._t0181_rig import (
    MIC_KWARG,
    NO_SEAM,
    REASON_MIC_NONE,
    REASON_NO_MIC,
    REASON_SILENT,
    mic_rig,
    mic_states,
    triple,
)
from tests.unit._t0201_pins import capture_without_default  # T-0201 D3 re-pin
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tools_live import Events, fake_live

#: BASE, a literal: main when T-0181 group A branched (``T-0180: group B merged``)
BASE_SHA = "74ae8b25229c65494530c18a26e00e503a932c19"

#: case -> (argv after ``session``, ``ASTRA_AUDIO_SOURCE`` or None, the A3 reason or None for available)
CASES: dict[str, tuple[list[str], str | None, str | None]] = {
    "silent": (["--silent"], None, REASON_SILENT),
    "silent-no-mic": (["--silent", "--no-mic"], None, REASON_SILENT),
    "silent-mic-none": (["--silent", "--mic", "none"], None, REASON_SILENT),
    "no-mic": (["--no-mic"], None, REASON_NO_MIC),
    "no-mic-env": (
        ["--no-mic"],
        ENV_MIC,
        REASON_NO_MIC,
    ),  # adversarial: the variable does not undo the flag
    "mic-none": (["--mic", "none"], None, REASON_MIC_NONE),
    "mic-none-env": (["--mic", "none"], ENV_MIC, REASON_MIC_NONE),
    "mic-node": (["--mic", RIG_MIC], None, None),
    "env-node": ([], ENV_MIC, None),
    "default-node": ([], None, None),
}
#: tools mode -> extra argv (live headless, loopback start pages only)
MODES: dict[str, list[str]] = {
    "off": [],
    "dry_run": ["--tools", "dry-run"],
    "live": ["--tools", "live", "--browser", "headless", "--no-web"],
}
#: the cases run in every mode (one unavailable, one available); the rest run tools off
MODE_CASES = ("no-mic", "default-node")


def _runs() -> list[tuple[str, str]]:
    return [("off", c) for c in CASES] + [(m, c) for m in MODES if m != "off" for c in MODE_CASES]


RUNS = _runs()


def _available(case: str) -> bool:
    return CASES[case][2] is None


def _param(*values: str, case: str) -> Any:
    """One row: an unavailable case needs A3's keyword; an available case is a guard row (A-R13: byte-identical to
    BASE). Group A activated: no row is pending."""
    return pytest.param(*values, id="/".join(values))


def _run(
    tmp: Path, mode: str, case: str, *, dry: bool, pin_wait_off: bool = False
) -> dict[str, Any]:
    """``diffusion_gearvis session <mode> <case>`` over the fakes, ``--dry-run`` or a real start stopped before
    serving: exit code, stdout, stderr (tmp and tree roots normalised), how many graphs were built and each
    ``Session``'s keywords, described.

    ``pin_wait_off`` (T-0202 D3 re-pin, pin off): a live run on a tree with ``--wait-tool`` appends
    ``--wait-tool off`` so the session is BASE's (no wait tool, no wait sentence); a no-op on BASE's tree."""
    from local_astra import cli
    from tests.conftest import VirtualClock
    from tests.fakes.audio import FakeAudioIO
    from tests.unit import test_t0134c_live_feedback_default as t0134c

    flags, env, _ = CASES[case]
    tmp.mkdir(parents=True, exist_ok=True)
    w = str(tmp)
    calls = T.Calls()
    opened: list[Any] = []
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISPLAY", ":99")
        if env is None:
            mp.delenv(ENV_SOURCE, raising=False)
        else:
            mp.setenv(ENV_SOURCE, env)
        pin_default_source(mp, BOX_MIC)
        T.patch_detok_load(mp, calls, tmp)
        if dry:
            explode_clients(mp)
        else:
            T.fake_clients(mp, calls)
            T.stop_before_serving(mp, calls)

            def fake_open_audio(args: Any) -> Any:
                opened.append(getattr(args, "mic", None))
                return FakeAudioIO(VirtualClock())

            mp.setattr(cli, "open_audio", fake_open_audio, raising=False)
        fake_live(mp, Events())
        config = T.write_config(tmp)
        argv = [
            "--config",
            str(config),
            "session",
            "--no-log",
            "--data-dir",
            str(tmp / "data"),
            # T-0197 pin move: BASE's session served no HUD page; opt out on a tree that parses the flag
            *t0197.opt_out(),
            *flags,
            *MODES[mode],
        ]
        if pin_wait_off:
            # T-0202 D3 re-pin: pin off (live only; off and dry_run are not live, so nothing is appended)
            argv = t0202.pin_off(argv, live_prefix=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli([*argv, "--dry-run"] if dry else argv)
    return {
        "code": code,
        "out": t0134c._norm(out.getvalue(), w),
        "err": t0134c._norm(err.getvalue(), w),
        "graphs": len(calls.graphs),
        "opened": opened,
        "session_kwargs": [t0134c._describe(k, w) for k in calls.session_kwargs],
    }


def _capture(tmp: Path) -> dict[str, Any]:
    """Every run, ``--dry-run`` and a real start. Runs in-process here and in a subprocess on the BASE tree."""
    out: dict[str, Any] = {}
    for mode, case in RUNS:
        key = f"{mode}/{case}"
        out[key] = {
            # T-0202 D3 re-pin: pin off (the A3 rows compare with BASE 74ae8b25, which has no wait tool)
            "dry": _run(tmp / mode / case / "dry", mode, case, dry=True, pin_wait_off=True),
            "start": _run(tmp / mode / case / "start", mode, case, dry=False, pin_wait_off=True),
        }
    return json.loads(json.dumps(out, default=str))


def _mic_kwarg(run: dict[str, Any]) -> tuple[bool, Any]:
    """(the one ``Session``'s keywords hold ``mic_unavailable``, its value)."""
    assert run["code"] == 0 and run["graphs"] == 1 and len(run["session_kwargs"]) == 1, (
        f"premise: one real start, one Session; code {run['code']}, graphs {run['graphs']}\n{run['err'][-1500:]}"
    )
    kwargs = run["session_kwargs"][0]
    return MIC_KWARG in kwargs, kwargs.get(MIC_KWARG)


def _assert_reason(run: dict[str, Any], mode: str, case: str) -> None:
    want = CASES[case][2]
    present, value = _mic_kwarg(run)
    if want is None:
        assert not present, (
            f"T-0181 A-R13 ({mode}, {case}): a resolved node is available, so cli does not pass {MIC_KWARG} at all "
            f"(never None; the construction is byte-identical to BASE); got {MIC_KWARG}={value!r}"
        )
        return
    assert present and value == want, (
        f"T-0181 A3 ({mode}, {case}): cli passes Session(..., {MIC_KWARG}={want!r}); "
        f"got {'no such keyword' if not present else repr(value)}"
    )


# --- A3: the exact reasons ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", [_param(c, case=c) for c in sorted(CASES)])
def test_a3_cli_passes_the_exact_reason(tmp_path: Path, case: str) -> None:
    """A3 (A-R1/A-R13/A-R15): a real start, tools off: ``--silent`` (alone, or with ``--no-mic``/``--mic none``),
    ``--no-mic`` and ``--mic none`` (with or without ``ASTRA_AUDIO_SOURCE`` exported) construct the ``Session`` with
    their exact reason (pending); a resolved node (``--mic``, the variable, the default source) constructs it without
    the keyword (a guard row, green at BASE)."""
    run = _run(tmp_path, "off", case, dry=False)
    _assert_reason(run, "off", case)


@pytest.mark.parametrize(
    ("case", "mode"),
    [_param(c, m, case=c) for c in MODE_CASES for m in sorted(set(MODES) - {"off"})],
)
def test_a3_the_reason_reaches_the_session_in_every_tools_mode(
    tmp_path: Path, mode: str, case: str
) -> None:
    """A3/A-R13: whatever the tools mode (``dry_run``, live), the one ``Session(...)`` construction carries the reason
    when the mic is unavailable (pending) and no keyword when it is available (a guard row)."""
    run = _run(tmp_path, mode, case, dry=False)
    _assert_reason(run, mode, case)


@pytest.mark.parametrize(
    ("mic", "want"),
    [
        pytest.param(True, (False, True, None), id="capture-seam"),
        pytest.param(False, (False, False, NO_SEAM), id="no-capture-seam"),
    ],
)
async def test_a_r7_mic_unavailable_none_is_derived(
    tmp_path: Path, mic: bool, want: tuple[Any, Any, Any]
) -> None:
    """A-R7 (and A3's default): ``Session(..., mic_unavailable=None)`` is the same as omitting it. Over a
    ``CaptureSource`` the hello says available and live; over an audio object with no ``stt_blocks`` it says
    unavailable with ``"this audio path has no capture seam"``. A session that read ``None`` as "available" would fail
    the second; one that read it as "unavailable" would fail the first."""
    async with mic_rig(tmp_path, mic=mic, mic_unavailable=None) as rig:
        states = mic_states(rig.page.sent())
    assert [triple(m) for m in states] == [want], (
        f"T-0181 A-R7: mic_unavailable=None is derived; the hello says {[triple(m) for m in states]}, want {[want]}"
    )


# --- A3: every printed line is BASE's -------------------------------------------------------------------------------


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0181_cli_mic as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Every run at BASE, once per module."""
    from tests.unit import test_session_tool_loop as tsl

    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0181_cli_base")
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
        timeout=600,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Every run on the working tree, once per module, through the same JSON round trip."""
    # T-0201 D3 re-pin (admit): a live run's default screen reminder line and screen_reminder=True keyword only
    # T-0207 C3 re-pin (admit): a live run's default completion check line and completion_check=True keyword only
    return t0207.capture_without_default(
        capture_without_default(_capture(tmp_path_factory.mktemp("t0181_cli_now") / "run")),
        "t0181 now",
    )


@pytest.mark.parametrize(("mode", "case"), [_param(m, c, case=c) for m, c in RUNS])
def test_a3_every_printed_line_is_bases(
    base: dict[str, Any], now: dict[str, Any], mode: str, case: str
) -> None:
    """A3 "every printed line is byte-identical to BASE" (and the task's "one keyword at ``Session(...)``
    construction, nothing printed"): for ``--dry-run`` and a real start, the exit code, stdout, stderr, the graph count
    and the mic the device was opened for equal BASE's. The ``Session`` keywords: an available mic's equal BASE's
    exactly (A-R13, a guard row); an unavailable one's equal BASE's once ``without_mic_unavailable`` drops the one key,
    and (anti-vacuity, pending) the TIP ``Session`` was given the A3 keyword with its exact value, which BASE never
    was."""
    key = f"{mode}/{case}"
    for kind in ("dry", "start"):
        want, got = base[key][kind], now[key][kind]
        for kwargs in want["session_kwargs"]:
            assert MIC_KWARG not in kwargs, f"premise: BASE {BASE_SHA[:8]} has no {MIC_KWARG}"
        for field in ("code", "out", "err", "graphs", "opened"):
            assert got[field] == want[field], (
                f"T-0181 A3: {key} {kind} {field} differs from BASE {BASE_SHA[:8]}:\n"
                f"base {json.dumps(want[field])[:2000]}\nnow  {json.dumps(got[field])[:2000]}"
            )
        compared = (
            got["session_kwargs"]
            if _available(case)
            else without_mic_unavailable(got["session_kwargs"])
        )
        assert compared == want["session_kwargs"], (
            f"T-0181 A3/A-R13: {key} {kind}: cli adds one keyword at Session(...) only when the mic is unavailable "
            f"and changes no other:\nbase {json.dumps(want['session_kwargs'])[:2000]}\n"
            f"now  {json.dumps(compared)[:2000]}"
        )
    _assert_reason(now[key]["start"], mode, case)
