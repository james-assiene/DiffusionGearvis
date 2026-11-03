"""T-0227 A-11 (negative) and A-10: with no flag, no profile key and no set, nothing changes (docs/tasks/T-0227.md).

**The differential, as these rows read "sessions built through ``cli``"** (reading R-D1, reported): a full ``cli``
start over the CLI rig's fakes (``tests/unit/test_t0213_settings_wire._cli_graph``: nothing served, every client a
fake) is made twice per mode, (a) with no new flag and (b) with ``--max-frames-per-turn 3 --frame-interval-ms
1000``. What ``cli`` hands the session is captured by a spy on ``Session.init_settings`` (delegating to the real
method) and compared: the snapshot, the read-only rows, the ``TurnPolicy`` repr and the builder's ``Budget`` are
equal, and the sources differ only in the two keys' (``default`` vs ``flag``). Driving the ``cli`` graph itself
through turns needs a real clock and audio, so the turns run on the scripted-clock rig
(``tests/unit/_t0181_rig.mic_rig``) whose session is given **exactly** what ``cli`` handed ``init_settings`` in
each arm, plus a third arm with no ``init_settings`` call at all (a session as ``Session.__init__`` builds it, BASE's
shape). The scenario: a voice turn with two scene changes, then (dry_run, live) a typed tool turn (step 1 clicks,
step 2 answers). Every prefill, every generate (prompt and options), every request body and every turn line are
equal across the three arms, with the random session id normalised.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, profile ``lean``; tools off / dry_run / live (live kit over
``FakeLiveBrowser``, loop on); n = 1 per arm, deterministic. No measurement.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.prompt_builder import Budget
from local_astra.contracts import events as E
from local_astra.contracts.perception import Source
from local_astra.contracts.turn import TurnState

REPO = Path(__file__).resolve().parents[2]
KEYS = ("max_frames_per_turn", "frame_interval_ms")
_LOG_NAME = re.compile(r"logs/\d{8}-\d{6}-\d+\.jsonl")
EXPLICIT = ("--max-frames-per-turn", "3", "--frame-interval-ms", "1000")
TOMLS = {
    "off": "[profiles.lean]\n",
    "dry_run": '[profiles.lean]\ntools = "dry_run"\n',
    "live": '[profiles.lean]\ntools = "live"\nbrowser = "headless"\n',
}


def _capture(
    tmp_path: Path, capsys: Any, monkeypatch: Any, mode: str, *flags: str
) -> dict[str, Any]:
    from local_astra.conductor import session as session_mod
    from tests.unit.test_t0213_settings_wire import _cli_graph

    real = session_mod.Session.init_settings
    seen: list[tuple[Any, ...]] = []

    def spy(self: Any, settings: Any, sources: Any, others: Any, **kw: Any) -> Any:
        seen.append((settings, dict(sources), dict(others)))
        return real(self, settings, sources, others, **kw)

    with monkeypatch.context() as mp:
        mp.setattr(session_mod.Session, "init_settings", spy)
        graph = _cli_graph(tmp_path, capsys, monkeypatch, TOMLS[mode], *flags)
    assert len(seen) == 1, f"premise: cli calls init_settings once; {len(seen)}"
    settings, sources, others = seen[0]
    return {
        "settings": settings,
        "sources": sources,
        "others": others,
        "policy": repr(graph.session.machine.policy),
        "budget": graph.session.prompts.budget,
        "system": graph.session.prompts.system_message().text(),
    }


def _kit(mode: str) -> Any:
    from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit

    if mode == "off":
        return None, None
    if mode == "dry_run":
        return "default", None
    browser = FakeLiveBrowser(
        [jpeg_frame("t0227-d0"), jpeg_frame("t0227-d1")],
        post_frames=[jpeg_frame(f"t0227-d-after-{k}") for k in range(4)],
    )
    return live_kit(browser), browser


async def _scenario(tmp_path: Path, mode: str, given: dict[str, Any] | None) -> list[str]:
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._t0181_rig import mic_rig
    from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
    from tests.unit.test_session_loop import POLICY, frame

    kit, browser = _kit(mode)
    async with mic_rig(tmp_path, kit=kit, policy=tsl.LOOP if mode != "off" else POLICY) as rig:
        if browser is not None:
            browser.now_ms = lambda: rig.clock.now_ms
        if given is not None:
            rig.session.init_settings(given["settings"], given["sources"], given["others"])
        llm = rig.llm
        llm.script_generates([answer("Yes. ")], [answer(tsl.S1), tool(VALID_CLICK)])
        llm.script_steps([answer(tsl.S3)])
        await rig.onset()
        for tag in ("b", "c"):
            await rig.advance(1_200, step=200)
            fr = frame(rig.clock.now_ms, tag)
            rig.perception.add(fr)
            rig.submit(
                E.SceneChanged(t_ms=rig.clock.now_ms, source=Source.SCREEN.value, phash=fr.phash)
            )
            await rig.pump()
            await rig.words("compare these screens")
        await rig.end_of_turn()
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 1, "t1 never ended"
        )
        await rig.advance(500)
        if mode != "off":
            await rig.type_question("open the settings menu")
            await rig.until(
                lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 2,
                "t2 never ended",
                limit_ms=30_000,
            )
            await rig.advance(500)
        sid = rig.session.session_id
        out = [f"prefill {rid} {p.model_dump_json()}" for rid, p in llm.prefills]
        out += [
            f"generate {rid} {p.model_dump_json()} {o.model_dump_json()}"
            for rid, p, o in llm.generates
        ]
        out += [json.dumps(b, sort_keys=True) for b in llm.bodies]
        out += [f"line {line}" for line in rig.lines()]
    # the session id is random and the session log's file name carries the wall clock: both normalised
    return [
        _LOG_NAME.sub("logs/<log>.jsonl", s.replace(sid, "<sid>").replace(str(tmp_path), "<tmp>"))
        for s in out
    ]


@pytest.mark.parametrize("mode", ("off", "dry_run", "live"))
def test_default_bodies_identical(tmp_path, capsys, monkeypatch, mode) -> None:
    """A-11: no new flag vs the two flags at their defaults, through ``cli`` (reading R-D1): what ``cli`` hands the
    session is equal but for the two sources; the builder is ``Budget()`` and ``TurnPolicy`` is the same; and every
    request and turn line of the scenario is byte-identical across (a), (b) and a session given nothing."""
    a = _capture(tmp_path / "a", capsys, monkeypatch, mode)
    b = _capture(tmp_path / "b", capsys, monkeypatch, mode, *EXPLICIT)
    assert a["settings"] == b["settings"], (a["settings"], b["settings"])
    assert {k: getattr(a["settings"], k) for k in KEYS} == {
        "max_frames_per_turn": 3,
        "frame_interval_ms": 1000,
    }
    assert {k: a["sources"].get(k, "default") for k in KEYS} == dict.fromkeys(KEYS, "default"), a[
        "sources"
    ]
    assert {k: b["sources"].get(k) for k in KEYS} == dict.fromkeys(KEYS, "flag"), b["sources"]
    strip = {k: v for k, v in a["sources"].items() if k not in KEYS}
    assert strip == {k: v for k, v in b["sources"].items() if k not in KEYS}, (
        "A-11: only the two sources differ"
    )
    assert a["others"] == b["others"] and a["policy"] == b["policy"] and a["system"] == b["system"]
    assert a["budget"] == b["budget"] == Budget(), (a["budget"], b["budget"])
    runs = {
        "none": asyncio.run(_scenario(tmp_path / "s-none", mode, None)),
        "a": asyncio.run(_scenario(tmp_path / "s-a", mode, a)),
        "b": asyncio.run(_scenario(tmp_path / "s-b", mode, b)),
    }
    assert runs["none"], "premise: the scenario sent requests"
    for arm in ("a", "b"):
        diff = [
            (i, x, y)
            for i, (x, y) in enumerate(zip(runs["none"], runs[arm], strict=False))
            if x != y
        ]
        assert len(runs[arm]) == len(runs["none"]) and not diff, (
            f"A-11 ({mode}): arm {arm} differs from a session given nothing; first diff {diff[:1]}"
        )


async def test_scenario_premise_without_init_settings(tmp_path) -> None:
    """Premise (green at BASE): the scenario is deterministic: two runs of the same arm are byte-identical, so the
    differential above can only redden on a real difference."""
    one = await _scenario(tmp_path / "1", "live", None)
    two = await _scenario(tmp_path / "2", "live", None)
    assert one and one == two


def test_readme_block_regenerated() -> None:
    """A-10: ``gen_readme_options.py --check`` exits 0 with the two new rows in the catalog, and the README block
    names both labels (complements T-0221's own row)."""
    from local_astra.settings_catalog import CATALOG

    keys = [e.key for e in CATALOG]
    assert all(k in keys for k in KEYS), f"T-0227 A-2: catalog rows {KEYS} do not exist yet"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_readme_options.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"A-10: README options block is stale\n{proc.stdout}{proc.stderr}"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- options:begin -->", 1)[1].split("<!-- options:end -->", 1)[0]
    for label in ("Screens per turn (max images)", "Screen capture interval (ms)"):
        assert label in block, f"A-10: the README table has the {label!r} row"
