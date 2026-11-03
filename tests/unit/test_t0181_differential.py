"""T-0181 A11 (negative): with the mute never toggled, a session is byte-identical to BASE but for the hello's added
``astra.mic.state``; and the mute opens no socket (docs/tasks/T-0181.md A11, "Tests to write -> Group A").

**Method.** A differential over ``BASE..TIP``: ``BASE`` is main at 74ae8b25 (T-0181 group A's branch point, a literal;
A-R12: it becomes a closed range at merge), run in a subprocess over ``git archive BASE`` (T-0114's ``_base_tree``,
origin checked; ``tests`` from this checkout, so the same rig drives both); ``TIP`` is the working tree, in-process.
The rig is ``_t0181_rig.mic_rig`` (the session over the fakes, ``FakeAudioIO`` capture, a reactive ``FakeStt`` behind
a recorder, a socketless ``HudServer``); nothing in it names a T-0181 symbol, and no mute is ever sent.

**Regimes** (A11's three, each one session):

- ``voice``: a spoken question (``what quokka narwhal zebra`` on the recogniser at audio 1000 ms, 200 ms a word, with
  endpointer steps) answered; then a second page greets;
- ``typed``: an ``astra.text`` question over the wire, answered; then a second page greets;
- ``live_loop``: ``--tools live`` (the live kit over the fake live browser, loop on): an ``astra.text`` question, step 1
  speaks and clicks, step 2 answers; then a second page greets.

**Byte-identical sections** (A11's list): every frame fed to STT (``t_ms`` and the sha256 of its bytes), every prompt
and every request body the real ``VllmClient`` builds from it, the machine's events, states and commands (so no
``MicMuted`` is stepped), every published message, every page's wire output, history, ``turns.jsonl``, every session-log
record, the TTS requests and the memory store.

**The only thing ignored** is the hello's ``astra.mic.state``: removed from each page's wire output, and session-log
``wire.out`` records whose message is an ``astra.mic.state`` are removed (at most one per hello; the publisher's own
stream must carry none). Everything else is compared as captured, after the usual normalisation (session id, tmp dir,
log path, ``git_commit``, the log writer's pid and byte count).

**Anti-vacuity.** TIP's hello must carry the state exactly where A4 puts it (index 3 of each page's wire output; both
pages greet with no reset), and a separate probe session must zero the frames it feeds once a mute is sent (A5), while
BASE shows neither: so the rows are red until group A lands, and a TIP that never implemented the mute cannot pass.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, ``TurnPolicy(thinking=True)`` (tools off) or the
loop policy (live), 200 ms per word, ``FakeAudioIO`` 80 ms STT frames, n = 1 per regime, deterministic. No
measurement.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit._t0181_rig import STATE, MicRig, mic_rig, reactive_stt
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_tools_off_golden import normalise

#: A11's BASE: main at 74ae8b25 (``T-0180: group B merged``), where T-0181 group A branched. A literal.
BASE_SHA = "74ae8b25229c65494530c18a26e00e503a932c19"
REGIMES = ("voice", "typed", "live_loop")
SECTIONS = (
    "stt_fed",
    "prompts",
    "bodies",
    "machine",
    "hud",
    "wire",
    "history",
    "records",
    "log",
    "tts",
    "memory",
)
VOICE = "what quokka narwhal zebra"
V_START = 1000
TYPED = "what is on my screen"
#: A4: where the state sits in a page's wire output when the page greets with no reset before it
HELLO_INDEX = 3


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _is_hello_state_row(record: dict[str, Any]) -> bool:
    data = record.get("data") or {}
    msg = data.get("msg") if isinstance(data, dict) else None
    return record.get("kind") == "wire.out" and isinstance(msg, dict) and msg.get("type") == STATE


def _log_state(message: dict[str, Any]) -> dict[str, Any]:
    """``astra.log.state``'s ``bytes`` and ``records`` describe the log file so far: its size depends on the tmp path's
    length (BASE and TIP run in different directories), and its record count would count a hello state's ``wire.out``
    row (the one record A11 ignores). Both become placeholders; the records themselves are compared in ``log``."""
    if message.get("type") != "astra.log.state":
        return message
    return {**message, "bytes": "<bytes>", "records": "<records>"}


def _bodies(rig: MicRig) -> list[dict[str, Any]]:
    """Every request body the real ``VllmClient`` builds from the prompts the fake LLM was handed (T-0177 A9's)."""
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.contracts.llm import LlmOptions
    from tests.unit.test_session_live import _refusing_transport

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


def _capture_rig(rig: MicRig, tmp: Path) -> dict[str, Any]:
    """Every A11 section of one finished rig (after ``aclose``), normalised. ``mic_positions`` (where each page's
    wire output carried the state) and ``mic_rows`` (the ``wire.out`` rows removed) are kept apart for the premises."""
    session_id, log_path = rig.session.session_id, Path(rig.log.path)
    per_process = {"log.started": {"pid": "<pid>"}, "log.stopped": {"bytes": "<bytes>"}}
    wires = [[_dump(m) for m in page.sent()] for page in rig.pages]
    sections: dict[str, Any] = {
        "stt_fed": [[t, hashlib.sha256(pcm).hexdigest()] for pcm, t in rig.rec_stt.fed],
        "prompts": [
            {"phase": "prefill", "request_id": r, "prompt": _dump(p)} for r, p in rig.llm.prefills
        ]
        + [
            {"phase": "generate", "request_id": r, "prompt": _dump(p), "options": _dump(o)}
            for r, p, o in rig.llm.generates
        ],
        "bodies": _bodies(rig),
        "machine": [
            {"event": _dump(ev), "state": str(state), "commands": [_dump(c) for c in cmds]}
            for ev, state, cmds in rig.machine.log
        ],
        "hud": [_log_state(_dump(m)) for m in rig.published.messages],
        "wire": [[_log_state(m) for m in wire if m.get("type") != STATE] for wire in wires],
        "history": [_dump(h) for h in rig.state.history],
        "records": [json.loads(line) for line in rig.lines()],
        "log": [
            {"kind": r["kind"], "data": {**r["data"], **per_process.get(r["kind"], {})}}
            for r in rig.log_records
            if not _is_hello_state_row(r)
        ],
        "tts": [list(r) for r in rig.tts.requests],
        "memory": [
            {"id": k, "episode": _dump(v)} for k, v in sorted(rig.session.memory.episodes.items())
        ],
        "mic_positions": [
            [i for i, m in enumerate(wire) if m.get("type") == STATE] for wire in wires
        ],
        "mic_rows": sum(1 for r in rig.log_records if _is_hello_state_row(r)),
    }
    out: dict[str, Any] = normalise(sections, session_id, tmp, log_path)
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _until_lines(rig: MicRig, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0181 A11 differential: {what} never ended",
        limit_ms=30_000,
        step=40,
    )
    await rig.advance(300, step=40)


async def _voice(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with mic_rig(
        tmp, stt=reactive_stt(steps=True), script=[answer("A shop window. ")]
    ) as rig:
        rig.say(VOICE, V_START)
        await _until_lines(rig, 1, "the voice turn")
        rig.greet()
        await rig.advance(200, step=40)
    return _capture_rig(rig, tmp)


async def _typed(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with mic_rig(tmp, script=[answer("A shop. ")]) as rig:
        await rig.advance(400, step=40)
        await rig.type_question(TYPED)
        await _until_lines(rig, 1, "the typed turn")
        rig.greet()
        await rig.advance(200, step=40)
    return _capture_rig(rig, tmp)


async def _live_loop(tmp: Path) -> dict[str, Any]:
    from tests.unit._live_browser_support import live_frame, live_kit
    from tests.unit.test_t0177_shared_frame import VisionBrowser

    tmp.mkdir(parents=True, exist_ok=True)
    browser = VisionBrowser(
        [live_frame("t0181-d-onset")],
        post_frames=[live_frame(f"t0181-d-after-{k}") for k in (1, 2, 3)],
    )
    async with mic_rig(
        tmp,
        kit=live_kit(browser),
        policy=tsl.LOOP,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.advance(400, step=40)
        await rig.type_question(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
        await _until_lines(rig, 1, "the live tool-loop turn")
        rig.greet()
        await rig.advance(200, step=40)
    out = _capture_rig(rig, tmp)
    out["executed"] = [[i.call_id, dict(i.arguments)] for i in browser.invocations]
    return out


async def _probe(tmp: Path) -> dict[str, Any]:
    """The anti-vacuity probe, a separate session: a raw ``astra.mic.mute(true)`` over the wire (``Unknown`` at
    BASE), then 480 ms; did any all-zero frame reach ``stt.feed`` after it?"""
    tmp.mkdir(parents=True, exist_ok=True)
    async with mic_rig(tmp) as rig:
        await rig.advance(400, step=40)
        mark = len(rig.rec_stt.fed)
        rig.send_mute(True)
        await rig.pump()
        await rig.advance(480, step=40)
        zeroed = any(rig.rec_stt.zero()[mark:])
        fed = len(rig.rec_stt.fed) - mark
    return {"zeroed": zeroed, "fed": fed}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every regime and the probe. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    return {
        "voice": await _voice(tmp / "voice"),
        "typed": await _typed(tmp / "typed"),
        "live_loop": await _live_loop(tmp / "live_loop"),
        "probe": await _probe(tmp / "probe"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0181_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The BASE run of every regime, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0181_base")
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
    """The working tree's run of every regime, once per module, through the same JSON round trip as BASE's."""
    import asyncio

    tmp = tmp_path_factory.mktemp("t0181_now")
    return json.loads(json.dumps(asyncio.run(_capture(tmp / "run"))))


#: T-0199 criterion 11 (census bullet, 2026-10-29): every voice-onset frame lookup writes exactly one note with these
#: keys. The only log record this comparison drops; every other record stays byte-compared with BASE.
ONSET_CLOCK_KEYS = {"what", "onset_ms", "at_session_ms", "method"}


def _without_onset_clock(records: list[Any]) -> list[Any]:
    """The log section minus exactly the ``note`` records with ``what == "onset.clock"`` (each asserted to have
    exactly criterion 11's keys). ``log.stopped``'s ``records`` counts every record written, so it is reduced by
    exactly the number dropped; every other field and record is compared as written. A no-op on BASE."""
    kept: list[Any] = []
    dropped = 0
    for r in records:
        data = r.get("data") if isinstance(r, dict) else None
        if r.get("kind") == "note" and isinstance(data, dict) and data.get("what") == "onset.clock":
            assert set(data) == ONSET_CLOCK_KEYS, (
                f"T-0199 criterion 11: an onset.clock note has exactly {sorted(ONSET_CLOCK_KEYS)}; got {sorted(data)}"
            )
            dropped += 1
            continue
        kept.append(r)
    if dropped:
        kept = [
            {**r, "data": {**r["data"], "records": r["data"]["records"] - dropped}}
            if r.get("kind") == "log.stopped"
            else r
            for r in kept
        ]
    return kept


#: T-0213 R-5 named re-pin: A2r-3's one ``astra.settings.state`` is the last message of every hello reply. In a
#: page's wire output (the mic state already removed) the first page greets before any turn, so its hello is the
#: head: assist, turn, log state (T-0181's mic state removed), plus T-0177 A4's vision state in live; the second page
#: greets at the end of the session, so its hello is its whole wire output.
SETTINGS_STATE = "astra.settings.state"
FIRST_HELLO_LEN = {"voice": 3, "typed": 3, "live_loop": 4}


def _wire_without_settings_state(regime: str, wires: list[list[Any]]) -> list[list[Any]]:
    """T-0213 R-5 named re-pin: each page's wire output with its one hello settings state taken out, after asserting
    there is exactly one per page and that it closes that page's hello (FIRST_HELLO_LEN; the second page's last)."""
    positions = [[i for i, m in enumerate(w) if m.get("type") == SETTINGS_STATE] for w in wires]
    assert positions == [[FIRST_HELLO_LEN[regime]], [len(wires[1]) - 1]], (
        f"T-0213 A2r-3 (R-5, {regime}): one settings state per hello, its last message; positions {positions}"
    )
    return [[m for m in w if m.get("type") != SETTINGS_STATE] for w in wires]


def _same(expected: list[Any], got: list[Any], what: str) -> None:
    first = next((i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None)
    assert got == expected, (
        f"T-0181 A11 {what} differs from {BASE_SHA[:8]} (lengths {len(expected)} vs {len(got)}; first difference "
        f"at {first}: base {json.dumps(expected[first] if first is not None else None)[:800]} / "
        f"now {json.dumps(got[first] if first is not None else None)[:800]})"
    )


def test_the_tree_under_comparison_has_the_mute(base: dict[str, Any], now: dict[str, Any]) -> None:
    """Anti-vacuity for every row below: in the probe session a mute zeroes the frames fed to STT on the working tree
    (A5), and not at BASE, where the message is unknown. Both sides kept feeding (the pump never stopped)."""
    assert base["probe"]["zeroed"] is False and base["probe"]["fed"] > 0, (
        f"premise: BASE {BASE_SHA[:8]} has no mute; {base['probe']}"
    )
    assert now["probe"]["fed"] > 0, f"A5: the pump keeps feeding while muted; {now['probe']}"
    assert now["probe"]["zeroed"] is True, (
        "T-0181 A11 anti-vacuity: the working tree must implement the mute (A5: a mute zeroes the frames fed to "
        f"STT), so the 'no mute, no change' rows compare a tree that has it; probe {now['probe']}"
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_a11_with_the_mute_never_toggled_nothing_but_the_hello_state_differs(
    base: dict[str, Any], now: dict[str, Any], regime: str
) -> None:
    """A11: in each regime every section (frames fed to STT, prompts, request bodies, machine steps, published
    messages, every page's wire output without the hello's state, history, ``turns.jsonl``, every session-log record
    but a hello state's ``wire.out``, TTS, memory) equals BASE's. The hello's state is where A4 puts it, once per
    page, and never on the publisher's own stream; BASE has none. No ``MicMuted`` is stepped. The premises say each
    regime ran its turn (the transcript the machine started on, one turn line, the live click executed)."""
    b, n = base[regime], now[regime]
    assert b["mic_positions"] == [[] for _ in b["wire"]] and b["mic_rows"] == 0, (
        f"premise: BASE {BASE_SHA[:8]} sends no astra.mic.state; {b['mic_positions']}"
    )
    assert len(n["wire"]) == 2, f"premise: two pages greeted; {len(n['wire'])}"
    want_start = {"voice": VOICE, "typed": TYPED, "live_loop": tsl.QUESTION}[regime]
    got_starts = [
        c["transcript"] for s in n["machine"] for c in s["commands"] if c["type"] == "llm.start"
    ]
    assert got_starts == [want_start], f"premise: the {regime} turn ran; llm.start {got_starts}"
    assert len(n["records"]) == 1, f"premise: one turn line; {len(n['records'])}"
    assert len(n["stt_fed"]) > 10, f"premise: the capture fed STT; {len(n['stt_fed'])} frames"
    if regime == "live_loop":
        assert len(n["executed"]) == 1, f"premise: the live click ran; {n['executed']}"
    stepped = [s["event"]["type"] for s in n["machine"]]
    assert "mic.muted" not in stepped, f"T-0181 A11 ({regime}): no MicMuted is stepped"
    assert not any(m.get("type") == STATE for m in n["hud"]), (
        f"T-0181 A11 ({regime}): with the mute never toggled the session publishes no astra.mic.state"
    )
    for section in SECTIONS:
        if section == "log":
            _same(
                _without_onset_clock(b[section]),
                _without_onset_clock(n[section]),
                f"{regime} 'log'",
            )
            continue
        if section == "wire":
            # T-0213 R-5 named re-pin: the one settings state per hello, asserted last, set aside
            _same(
                b[section],
                _wire_without_settings_state(regime, n[section]),
                f"{regime} {section!r}",
            )
            continue
        _same(b[section], n[section], f"{regime} {section!r}")
    if regime == "live_loop":
        _same(b["executed"], n["executed"], "live_loop executed calls")
    # anti-vacuity, last so a red row before group A says the sections already agree
    assert n["mic_positions"] == [[HELLO_INDEX], [HELLO_INDEX]], (
        f"T-0181 A4/A11 ({regime}): each hello carries exactly one astra.mic.state, fourth (index {HELLO_INDEX}), "
        f"and nothing else sends one with the mute never toggled; positions {n['mic_positions']}"
    )
    assert n["mic_rows"] <= 2, (
        f"T-0181 A11 ({regime}): at most one wire.out record per hello for the state; {n['mic_rows']}"
    )


# --- no new socket --------------------------------------------------------------------------------------------------


async def test_a11_no_new_socket(tmp_path: Path) -> None:
    """A11 "no new socket": a session that greets, mutes, answers a typed question while muted, unmutes and greets a
    second page creates no socket object at all (every ``socket.socket`` construction is counted, ``socketpair`` and
    ``fromfd`` included, since they build ``socket.socket`` too). The counter's power is checked first; the mute's
    effect (zeroed frames) is asserted, so a session with no mute cannot pass vacuously."""
    made: list[tuple[Any, ...]] = []
    real_init = socket.socket.__init__

    def counting_init(self: socket.socket, *a: Any, **k: Any) -> None:
        made.append(a)
        real_init(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "__init__", counting_init)
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
        assert len(made) == 1, f"premise: the counter sees a socket; {made}"
        made.clear()
        async with mic_rig(tmp_path, script=[answer("A shop. ")]) as rig:
            await rig.advance(400, step=40)
            await rig.engage()
            await rig.type_question(TYPED)
            await _until_lines(rig, 1, "the muted typed turn")
            await rig.release()
            rig.greet()
            await rig.advance(200, step=40)
            zeroed = sum(rig.rec_stt.zero())
    assert zeroed > 0, "premise: the mute zeroed frames"
    assert made == [], (
        f"T-0181 A11: the mute opens no socket; {len(made)} socket(s) were created: {made[:5]}"
    )
