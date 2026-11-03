"""T-0177 A9 (negative; re-run at B10): the browser source and non-live sessions are byte-identical to BASE.

**The differential.** ``_capture`` drives ``test_t0177_shared_frame.vision_rig`` (T-0114's ``tool_rig`` plus a
socketless ``HudServer`` and one greeted page; the HUD perception holds the distinctive shared frame ``SH``) through
four sessions, the three regimes of A9:

- ``live_browser``: live, no select ever. Turn 1: a spoken answer and a ``click_at`` (executed on the fake live
  browser), then step 2 answers; turn 2: a plain answer.
- ``live_toggled``: the same two turns, with ``astra.vision.select`` ``shared`` then ``browser`` sent by the page
  before turn 1 and again between the turns (A9 "after a select to shared and back to browser between turns"). Every
  turn is therefore a browser turn.
- ``off`` and ``dry_run``: tools off (two plain answers) and dry_run (turn 1 a click recorded by the dry-run
  executor, step 2 an answer; turn 2 a plain answer), with the HUD frame ``SH`` and a select ``shared`` sent between
  the turns (A3: outside live a select changes nothing).

Selects are sent as raw JSON (``page.receive``), never through the asserting ``select`` helper, so the same capture
runs at BASE, where the codec answers ``Unknown`` and the server counts them ``unhandled``.

Per session it records the three sections A9 names:

- ``requests``: the wire body of every prefill (``max_tokens=1``) and every generate (step 1, tool steps, captions,
  re-asks), built by the real ``VllmClient.request_body`` from the prompts the fake LLM was handed, in order;
- ``lines``: every ``turns.jsonl`` line; in the two live regimes the A10 key ``extras.vision_source`` is removed
  (the only key A9 ignores), outside live nothing is removed (A10: non-live lines are unchanged);
- ``llm_request``: the ``data`` of every ``llm.request`` record of the session log.

Normalisation, applied to both sides alike: the session id, the session log path and the tmp dir become
placeholders, and ``regime.git_commit`` becomes ``<git>`` (the BASE tree is a ``git archive`` with no repository).

The BASE run is a subprocess over ``git archive BASE_SHA`` (T-0114's ``_base_tree``, its ``local_astra`` first on
``sys.path`` because the subprocess's cwd is the tree; ``tests`` comes from this checkout), origin checked. The four
``test_regime_equals_base[...]`` rows are green at BASE by design and must stay green at A's and B's merges (B10).

``test_the_toggled_regime_really_toggled`` (red until A3/A10 land) is the premise that makes ``live_toggled`` a different regime
from ``live_browser`` at TIP: both selects of each pair were routed (two ``vision.select`` notes per pair, A3/A-P4) and
every live line says ``vision_source: "browser"`` (A10). Without it an implementation that ignored selects would
pass the regime vacuously.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools live / off / dry_run, loop on (8
steps), 1280x800 browser frames, HUD frame 1568x590 (screen), n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import LlmOptions
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import b64, live_frame
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import _refusing_transport
from tests.unit.test_t0177_shared_frame import SELECT, SH, VisionBrowser, vision_rig

#: T-0177's BASE: main at 7646b8b6 ("PROGRESS: T-0182 clip render run"), the branch point. A literal.
BASE_SHA = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
REGIMES = ("live_browser", "live_toggled", "off", "dry_run")
LIVE = ("live_browser", "live_toggled")
KITS: dict[str, Any] = {
    "live_browser": "live",
    "live_toggled": "live",
    "off": None,
    "dry_run": "default",
}
Q1 = tsl.QUESTION
Q2 = "what is on my screen now"


def _raw_select(rig: Any, source: str) -> None:
    rig.page.receive(json.dumps({"type": SELECT, "source": source}))


async def _toggle(rig: Any) -> None:
    for source in ("shared", "browser"):
        _raw_select(rig, source)
        await rig.pump()
    await rig.advance(300)


async def _until_lines(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state.name == "IDLE" and len(rig.lines()) >= n,
        f"T-0177 A9 differential: {what} never ended",
        limit_ms=30_000,
    )
    await rig.advance(300)


def _bodies(rig: Any) -> list[dict[str, Any]]:
    client = VllmClient(transport=_refusing_transport())
    prefills = [
        {
            "phase": "prefill",
            "request_id": rid,
            "body": client.request_body(p, rid, LlmOptions(max_tokens=1)),
        }
        for rid, p in rig.llm.prefills
    ]
    generates = [
        {"phase": "generate", "request_id": rid, "body": client.request_body(p, rid, o, x)}
        for (rid, p, o), x in zip(rig.llm.generates, rig.llm.xargs_log, strict=True)
    ]
    return [*prefills, *generates]


def _normalise(tree: Any, session_id: str, tmp: Path, log_path: Path) -> Any:
    text = (
        json.dumps(tree, sort_keys=True)
        .replace(str(log_path), "<log>")
        .replace(session_id, "<session>")
        .replace(str(tmp), "<tmp>")
    )
    out = json.loads(text)
    for line in out["lines"]:
        regime = line.get("regime")
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _scenario(tmp: Path, regime: str) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    browser = VisionBrowser(
        [live_frame("t0177-d-onset"), live_frame("t0177-d-turn2")],
        post_frames=[live_frame(f"t0177-d-after-{k}") for k in (1, 2, 3)],
    )
    tool_turn = regime != "off"
    first = [answer(tsl.S1), tool(VALID_CLICK)] if tool_turn else [answer("A shop window. ")]
    async with vision_rig(tmp, browser, script=first, kit=KITS[regime], hud_frames=[SH]) as rig:
        if regime == "live_toggled":
            await _toggle(rig)
        if tool_turn:
            rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(Q1)
        if tool_turn:
            await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
        await _until_lines(rig, 1, f"{regime} turn 1")
        if regime == "live_toggled":
            await _toggle(rig)
        elif regime in ("off", "dry_run"):
            _raw_select(rig, "shared")
            await rig.pump()
            await rig.advance(300)
        rig.llm.script = [answer("A shop. ")]
        await rig.ask(Q2)
        await _until_lines(rig, 2, f"{regime} turn 2")
        session_id = rig.session.session_id
        log_path = Path(rig.log.path)
        requests = _bodies(rig)
        raw_lines = [json.loads(x) for x in rig.lines()]
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    premise = {
        "selects": [
            [r["data"].get("from"), r["data"].get("to")]
            for r in records
            # A-P4: the A3 record is a ``note`` with ``what: "vision.select"``
            if r.get("kind") == "note" and (r.get("data") or {}).get("what") == "vision.select"
        ],
        "vision_source": [(line.get("extras") or {}).get("vision_source") for line in raw_lines],
        "executed": [i.call_id for i in browser.invocations],
    }
    lines = json.loads(json.dumps(raw_lines))
    if regime in LIVE:
        for line in lines:
            (line.get("extras") or {}).pop("vision_source", None)
    sections = {
        "requests": requests,
        "lines": lines,
        "llm_request": [r["data"] for r in records if r.get("kind") == "llm.request"],
    }
    out: dict[str, Any] = _normalise(sections, session_id, tmp, log_path)
    out["premise"] = premise
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every regime, in order. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    return {regime: await _scenario(tmp / regime, regime) for regime in REGIMES}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0177_differential as m
out = pathlib.Path(sys.argv[1])
got = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": got}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's sections, once per module (a subprocess over ``git archive BASE_SHA``; origin checked)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0177_diff_base")
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
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """This tree's sections, once per module (in-process)."""
    got: dict[str, Any] = asyncio.run(_capture(tmp_path_factory.mktemp("t0177_diff_now")))
    return got


def _compare(base: Any, now: Any, label: str, path: str = "") -> None:
    """Exact equality, recursively, so a failure names the first differing path."""
    if isinstance(base, dict) and isinstance(now, dict) and base.keys() == now.keys():
        for key in base:
            _compare(base[key], now[key], label, f"{path}/{key}")
        return
    if isinstance(base, list) and isinstance(now, list) and len(base) == len(now):
        for i, (b, n) in enumerate(zip(base, now, strict=True)):
            _compare(b, n, label, f"{path}[{i}]")
        return
    assert base == now, (
        f"{label}: differs from BASE {BASE_SHA[:8]} at {path}:\nbase {json.dumps(base)[:800]}\n"
        f"now  {json.dumps(now)[:800]}"
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_regime_equals_base(base: dict[str, Any], now: dict[str, Any], regime: str) -> None:
    """A9: in ``regime`` the prefill/generate/step bodies, the turn lines (live: without ``vision_source``) and the
    ``llm.request`` records equal BASE's exactly. Premises: two turns; outside ``off`` a click and a tool step; the
    HUD frame is in the ``off`` / ``dry_run`` bodies and in no live body. Green at BASE by design."""
    got = now[regime]
    assert len(got["lines"]) == 2, f"premise: two turns; {len(got['lines'])}"
    phases = [r.get("phase") for r in got["llm_request"]]
    if regime != "off":
        assert "tool_step" in phases, f"premise: a tool step in {regime}; {phases}"
    dumped = json.dumps(got["requests"])
    if regime in LIVE:
        assert b64(SH.jpeg) not in dumped, "premise (B6): no live body carries the HUD frame"
        assert got["premise"]["executed"], "premise: the live click was executed"
    else:
        assert b64(SH.jpeg) in dumped, "premise: the HUD frame is the non-live model input"
    for section in ("requests", "lines", "llm_request"):
        _compare(base[regime][section], got[section], f"T-0177 A9 {regime} {section}")


def test_the_toggled_regime_really_toggled(now: dict[str, Any]) -> None:
    """Premise of A9's second regime (A3, A10): both select pairs were routed and recorded as changes
    (``browser->shared``, ``shared->browser``, twice), every live line says ``vision_source: "browser"``, and no
    select outside live wrote a record."""
    toggled = now["live_toggled"]["premise"]
    assert toggled["selects"] == [["browser", "shared"], ["shared", "browser"]] * 2, toggled
    for regime in LIVE:
        assert now[regime]["premise"]["vision_source"] == ["browser", "browser"], (
            regime,
            now[regime]["premise"],
        )
    assert now["live_browser"]["premise"]["selects"] == []
    for regime in ("off", "dry_run"):
        assert now[regime]["premise"]["selects"] == [], (regime, now[regime]["premise"])
        assert now[regime]["premise"]["vision_source"] == [None, None]


# =====================================================================================================
# A-P1: the pre-authorised normaliser the older differentials use (tests/unit/_t0177_pins.py)
# =====================================================================================================


def test_the_pin_strips_browser_only_under_extras() -> None:
    """A-P1: ``vision_source: "browser"`` goes from every ``extras`` dict, nested in lists and records; the same key
    elsewhere stays, and the input is not mutated. Expected trees written by hand."""
    from tests.unit._t0177_pins import without_browser_vision_source

    line = {"turn_id": "t1", "extras": {"frame_source": "window", "vision_source": "browser"}}
    record = {"kind": "turn", "data": {"extras": {"vision_source": "browser", "browser": "open"}}}
    other = {"vision_source": "browser", "msg": {"vision_source": "browser"}}
    tree = {"lines": [line], "log": [record], "other": other, "t": (line,)}
    got = without_browser_vision_source(tree)
    assert got == {
        "lines": [{"turn_id": "t1", "extras": {"frame_source": "window"}}],
        "log": [{"kind": "turn", "data": {"extras": {"browser": "open"}}}],
        "other": {"vision_source": "browser", "msg": {"vision_source": "browser"}},
        "t": ({"turn_id": "t1", "extras": {"frame_source": "window"}},),
    }, got
    assert line["extras"]["vision_source"] == "browser", "the input is not mutated"


def test_the_pin_never_strips_shared() -> None:
    """A-P1 adversarial: a browser-regime row whose turn silently became shared must still differ from BASE, so
    ``"shared"`` is kept."""
    from tests.unit._t0177_pins import without_browser_vision_source

    line = {"extras": {"frame_source": "screen", "vision_source": "shared"}}
    assert without_browser_vision_source([line]) == [line]


@pytest.mark.parametrize("value", ["Browser", "window", None, 1, ["browser"]])
def test_the_pin_refuses_any_other_shape(value: Any) -> None:
    """A-P1: anything but A10's two values is not the declared difference and raises, never passes."""
    from tests.unit._t0177_pins import VisionSourcePinError, without_browser_vision_source

    with pytest.raises(VisionSourcePinError):
        without_browser_vision_source({"extras": {"vision_source": value}})
