"""T-0177 group A, A3 and A4: the vision choice belongs to the session (docs/tasks/T-0177.md).

The rigs are ``test_t0177_shared_frame.vision_rig`` (T-0114's ``tool_rig`` plus a socketless ``HudServer`` from the
``_hud_rig`` seams, one greeted page ``rig.page``), live by default, ``kit=None`` for ``off`` and ``kit="default"``
for ``dry_run``. Selects go over the wire as raw JSON, so before A1 lands a row fails on a named assertion (the codec
answers ``Unknown`` and the server counts it ``unhandled``), never at collection. What a page was sent is
``conn.sent()`` (``_hud_rig`` docstring: what this connection would put on the wire, in order).

**BASE capture** (A4 "byte-identical to BASE"). The module fixture ``base_replies`` runs ``_hello_capture`` in a
subprocess over ``git archive BASE`` (T-0114's ``_base_tree``; its ``local_astra`` first on ``sys.path``, the
origin checked); each row runs it again in this process. Each capture greets a second page after one answered turn, in ``off``,
``dry_run`` and ``live``, and records the encoded hello reply. The only normalisation is ``astra.log.state``'s
``path`` and ``bytes`` (the session log's tmp path and its size, which depends on that path's length); ``records``
is kept. Non-live replies must be equal; the live reply must be BASE's with exactly one ``astra.vision.state`` after
it. B9 pin move (T-0177 B): the captured turn sent an image, so a frame exists and each reply then ends with one
replayed ``astra.vision.frame`` (``replay: true``); the rows compare ``now[:-1]`` (non-live) and ``now[:-2]`` (live)
with BASE and check that trailing frame's bytes against the fixture the turn's request carried.

Readings (listed in the report): ``choices`` and ``replay`` are not pinned beyond the set ``{browser, shared}``
(orchestrator ruling 2); the A3 record is a ``note`` with data ``{what: "vision.select", from, to}`` (ruling A-P4; the session log's kind
vocabulary is untouched).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / dry_run / live, loop on, n = 1.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import AstraHello, AstraSessionReset
from tests import _git_range as gr
from tests.unit import _t0181_pins as t0181
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import turn_extras
from tests.unit._tool_turn_support import answer
from tests.unit.test_t0177_shared_frame import SELECT, one_turn, select, vision_rig

#: T-0177's BASE: main at 7646b8b6 ("PROGRESS: T-0182 clip render run"), the branch point. A literal.
BASE_SHA = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
STATE = "astra.vision.state"
UNAVAILABLE = "vision_source_unavailable"
MODES = {"off": None, "dry_run": "default", "live": "live"}


def greet(rig: Any) -> Any:
    """A second page, greeted; its hello reply is ``page.sent()``."""
    page = rig.server.open()
    page.receive(encode(AstraHello()))
    return page


def _without_settings_state(reply: list[Any]) -> list[Any]:
    """T-0213 R-5 named re-pin: the reply with its one settings state (asserted last) set aside. Imported here, not
    at module level: the BASE capture subprocess imports this module against BASE's ``local_astra``, which has no
    settings catalog."""
    from tests.unit import test_t0213_settings_wire as t0213

    return t0213.hello_without_settings_state(reply)


def states(messages: list[Any]) -> list[Any]:
    return [m for m in messages if getattr(m, "type", None) == STATE]


def raw_select(page: Any, source: str) -> None:
    page.receive(json.dumps({"type": SELECT, "source": source}))


# =====================================================================================================
# A3: routing, the echo, the record
# =====================================================================================================


async def test_a_select_before_hello_is_fenced(tmp_path: Path) -> None:
    """A3 "behind the hello fence": a select on an ungreeted page is counted ``no_hello``, publishes no state and
    does not change the choice (that page's hello then reports ``browser``).

    At BASE the type is unknown to the codec, so the select is counted ``unhandled`` (not ``no_hello``) and the
    ungreeted page is told nothing: the row fails on the ``no_hello`` count. With A1 decoded, ``receive`` answers an
    ungreeted page's client message ``astra.error(code="no_hello")`` before ``route``; the row pins that the select is
    fenced there (not counted ``unhandled``, not routed, the sender told ``no_hello``)."""
    async with vision_rig(tmp_path) as rig:
        page = rig.server.open()
        before = dict(rig.server.counts)
        raw_select(page, "shared")
        await rig.pump()
        after = dict(rig.server.counts)
        assert (after["no_hello"], after["unhandled"], after["routed"]) == (
            before["no_hello"] + 1,
            before["unhandled"],
            before["routed"],
        ), (
            f"A3: the select is fenced by the hello (no_hello +1, not unhandled, not routed); {after}"
        )
        told = [(m.type, getattr(m, "code", None)) for m in page.sent()]
        assert told == [("astra.error", "no_hello")], (
            f"A3: the ungreeted sender is told no_hello; {told}"
        )
        assert states(rig.published.messages) == [], "A3: nothing published for a fenced select"
        page.receive(encode(AstraHello()))
        reply = states(page.sent())
        assert [s.source for s in reply] == ["browser"], (
            f"A3/A4: the fenced select changed nothing, the hello says browser; {reply}"
        )


async def test_every_routed_select_is_echoed_redundant_ones_included(tmp_path: Path) -> None:
    """A3 (T-0043 echo): browser (redundant), shared, shared (redundant), browser each publish exactly one
    ``astra.vision.state`` carrying the current choice, in order; ``choices`` is the two sources."""
    async with vision_rig(tmp_path) as rig:
        mark = len(rig.page.sent())
        for source in ("browser", "shared", "shared", "browser"):
            await select(rig, source)
        echoed = states(rig.page.sent()[mark:])
    assert [s.source for s in echoed] == ["browser", "shared", "shared", "browser"], echoed
    assert all(set(s.choices) == {"browser", "shared"} for s in echoed), echoed


async def test_the_echo_reaches_every_connection(tmp_path: Path) -> None:
    """A3: a select from one page is echoed to every greeted page, the sender included, once each."""
    async with vision_rig(tmp_path) as rig:
        other = greet(rig)
        marks = (len(rig.page.sent()), len(other.sent()))
        await select(rig, "shared", page=other)
        got = (states(rig.page.sent()[marks[0] :]), states(other.sent()[marks[1] :]))
    assert [[s.source for s in g] for g in got] == [["shared"], ["shared"]], got


async def test_route_hands_the_select_to_the_session(tmp_path: Path) -> None:
    """A3: ``hud_server.route`` hands ``astra.vision.select`` to ``session.apply_vision_select(message)`` (the seam
    the task names); a spy that delegates sees one call carrying the page's ``source``."""
    async with vision_rig(tmp_path) as rig:
        real = getattr(rig.session, "apply_vision_select", None)
        assert callable(real), "T-0177 A3: Session.apply_vision_select does not exist yet"
        seen: list[Any] = []

        def spy(message: Any) -> Any:
            seen.append(message)
            return real(message)

        rig.session.apply_vision_select = spy
        await select(rig, "shared")
    assert [getattr(m, "source", None) for m in seen] == ["shared"], seen


async def test_a_record_only_when_the_choice_changes(tmp_path: Path) -> None:
    """A3 as ruled by A-P4: a change writes one ``note`` record with data ``{what: "vision.select", from, to}``; a
    redundant select writes none. Over browser, shared, shared, browser that is exactly two notes. No record of any
    other kind carries ``what: "vision.select"``, and none is the session log's ``error`` for an unknown kind (the
    kind vocabulary is untouched)."""
    async with vision_rig(tmp_path) as rig:
        for source in ("browser", "shared", "shared", "browser"):
            await select(rig, source)
    notes = [d for d in rig.kind("note") if d.get("what") == "vision.select"]
    assert [(d.get("from"), d.get("to")) for d in notes] == [
        ("browser", "shared"),
        ("shared", "browser"),
    ], notes
    assert all(set(d) == {"what", "from", "to"} for d in notes), notes
    others = [
        r
        for r in rig.log_records
        if r.get("kind") != "note" and (r.get("data") or {}).get("what") == "vision.select"
    ]
    assert others == [], f"A-P4: only note records carry what=vision.select; {others}"
    assert rig.kind("vision.select") == [], "A-P4: no record of kind vision.select"
    unknown = [d for d in rig.kind("error") if "vision.select" in str(d.get("kind", ""))]
    assert unknown == [], f"A-P4: no record was refused as an unknown kind; {unknown}"


@pytest.mark.parametrize("mode", ["off", "dry_run"])
async def test_outside_live_a_select_answers_the_sender_only(tmp_path: Path, mode: str) -> None:
    """A3 "Outside live": a select is routed (not ``unhandled``), changes nothing, and answers the sender with one
    ``astra.error(code="vision_source_unavailable")``; the other page gets nothing; no ``astra.vision.state`` is
    published, sent, or put in a later hello."""
    async with vision_rig(tmp_path, kit=MODES[mode]) as rig:
        other = greet(rig)
        marks = (len(rig.page.sent()), len(other.sent()))
        before = dict(rig.server.counts)
        raw_select(rig.page, "shared")
        await rig.pump()
        assert rig.server.counts["unhandled"] == before["unhandled"], "A3: the select is routed"
        sender, bystander = rig.page.sent()[marks[0] :], other.sent()[marks[1] :]
        assert [(m.type, getattr(m, "code", None)) for m in sender] == [
            ("astra.error", UNAVAILABLE)
        ], sender
        assert bystander == [], f"A3: the error goes to the sender only; {bystander}"
        third = greet(rig)
        everything = [*rig.page.sent(), *other.sent(), *third.sent(), *rig.published.messages]
        assert states(everything) == [], "A3: no state is ever published outside live"


async def test_the_choice_survives_a_reset(tmp_path: Path) -> None:
    """A3 / T-0162 D-3 (B9's "The choice survives a reset", named for A by the task's tests list): after ``shared`` and
    a session reset, a new page's hello reports ``shared`` and the next turn's line says ``vision_source: "shared"``."""
    async with vision_rig(tmp_path, script=[answer("I cannot see it. ")]) as rig:
        await select(rig, "shared")
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        await rig.advance(500)
        assert [s.source for s in states(greet(rig).sent())][-1:] == ["shared"]
        await one_turn(rig)
        assert turn_extras(rig).get("vision_source") == "shared", turn_extras(rig)


# =====================================================================================================
# A4: the hello
# =====================================================================================================


async def test_the_live_hello_ends_with_the_current_choice(tmp_path: Path) -> None:
    """A4: in live the hello reply's last message is ``astra.vision.state`` with the current choice (``browser`` by
    default, ``shared`` after a select), also after a reset (T-0162's replayed reset first, the state still last)."""
    async with vision_rig(tmp_path) as rig:
        first = greet(rig).sent()
        await select(rig, "shared")
        second = greet(rig).sent()
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        await rig.advance(500)
        third = greet(rig).sent()
    # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside; the vision state is last before it
    first, second, third = (_without_settings_state(r) for r in (first, second, third))
    for reply, want in ((first, "browser"), (second, "shared"), (third, "shared")):
        assert reply and reply[-1].type == STATE and reply[-1].source == want, [
            m.type for m in reply
        ]
        assert len(states(reply)) == 1, "A4: one state per hello"
    assert third[0].type == "astra.session.cleared", "premise: T-0162's replayed reset leads"


# --- the BASE capture ---------------------------------------------------------------------------------------


def _normalise(raw: str, frames_published: int = 0) -> str:
    """``frames_published`` (B9 pin move): each published ``astra.vision.frame`` writes exactly one ``wire.out``
    record (T-0043 c7, B7's redacted record), so ``records`` is compared net of them; at BASE the count is 0."""
    obj = json.loads(raw)
    if obj.get("type") == "astra.log.state":
        obj["path"] = "<log path>" if obj.get("path") is not None else None
        obj["bytes"] = "<log bytes>"
        if isinstance(obj.get("records"), int):
            obj["records"] -= frames_published
    return json.dumps(obj, sort_keys=True)


async def _hello_capture(tmp: Path, modes: tuple[str, ...] = tuple(MODES)) -> dict[str, list[str]]:
    """Per mode: the encoded hello reply of a page greeted after one answered turn (and, outside live, after a raw
    select from the first page, which must change nothing)."""
    out: dict[str, list[str]] = {}
    for mode in modes:
        kit = MODES[mode]
        async with vision_rig(tmp / mode, script=[answer("A shop. ")], kit=kit) as rig:
            await rig.ask("what is on my screen")
            await rig.turn_done()
            await rig.advance(300)
            if mode != "live":
                raw_select(rig.page, "shared")
                await rig.pump()
            published = sum(
                1
                for m in rig.published.messages
                if getattr(m, "type", None) == "astra.vision.frame"
            )
            out[mode] = [_normalise(encode(m), published) for m in greet(rig).sent()]
    return out


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0177_vision_select as m
out = pathlib.Path(sys.argv[1])
got = asyncio.run(m._hello_capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "replies": got}))
"""


@pytest.fixture(scope="module")
def base_replies(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    """BASE's hello replies, once per module (a subprocess over ``git archive BASE_SHA``; origin checked)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0177_hello_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "replies.json"
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
        f"origin proof: the base run imported local_astra from {data['origin']}"
    )
    replies: dict[str, list[str]] = data["replies"]
    return replies


FRAME = "astra.vision.frame"


def _replayed_frame(raw: str, want: bytes, where: str) -> None:
    """B9: ``raw`` is the hello's replayed ``astra.vision.frame``: ``replay`` true, turn ``t1``, not blind, one image
    whose bytes are ``want`` (the fixture the turn's request carried)."""
    import base64

    frame = json.loads(raw)
    assert frame.get("type") == FRAME, (
        f"B9: the {where} hello reply ends with the latest {FRAME} (a turn was answered, so one was published); "
        f"last message {frame.get('type')}"
    )
    assert frame.get("replay") is True, f"B9: the replayed frame says replay: true; {frame}"
    images = frame.get("images") or []
    got = [base64.b64decode(i["jpeg_b64"]) for i in images if i.get("jpeg_b64")]
    assert (frame.get("turn_id"), frame.get("blind"), got) == ("t1", False, [want]), (
        f"B9: the replayed frame is turn 1's request image; {[(i.get('kind'), i.get('identity')) for i in images]}"
    )


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_non_live_hello_equals_base(
    base_replies: dict[str, list[str]], tmp_path: Path, mode: str
) -> None:
    """A4 "Outside live, every hello reply is byte-identical to BASE" (after a turn and a select from another page),
    as moved by B9 (pin move, T-0177 B): a frame exists (the turn's request carried the HUD frame ``SH``, and B7
    publishes in every mode), so the reply is BASE's followed by exactly one trailing ``astra.vision.frame`` with
    ``replay: true`` -- nothing else changes, nothing moves."""
    from tests.unit.test_t0177_shared_frame import SH

    now = asyncio.run(_hello_capture(tmp_path, modes=(mode,)))[mode]
    # T-0181 A4 pin move (A-R8): the one mic state, asserted right after the log state, set aside
    now = t0181.hello_without_mic_state(now)
    # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
    now = _without_settings_state(now)
    base = base_replies[mode]
    assert len(base) >= 3, f"premise: BASE's reply; {base}"
    assert FRAME not in "".join(base), "premise: BASE published no vision frame"
    _replayed_frame(now[-1], SH.jpeg, mode)
    assert now[:-1] == base, (
        f"A4/B9: the {mode} hello reply is BASE {BASE_SHA[:8]}'s plus the replayed frame:\nbase {base}\n"
        f"now  {[json.loads(m).get('type') for m in now]}"
    )


def test_the_live_hello_is_base_plus_the_state(
    base_replies: dict[str, list[str]], tmp_path: Path
) -> None:
    """A4: in live the hello reply is BASE's followed by exactly one ``astra.vision.state`` (``browser``), after
    everything T-0037/T-0043/T-0162 put there; B9 (pin move): then the latest ``astra.vision.frame`` with
    ``replay: true`` (the turn's step-1 image, the browser's own frame ``F0``) is last."""
    from tests.unit.test_t0177_shared_frame import F0

    now = asyncio.run(_hello_capture(tmp_path, modes=("live",)))["live"]
    # T-0181 A4 pin move (A-R8): the one mic state, asserted right after the log state, set aside
    now = t0181.hello_without_mic_state(now)
    # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
    now = _without_settings_state(now)
    base = base_replies["live"]
    assert len(base) >= 3, f"premise: BASE's live reply; {base}"
    _replayed_frame(now[-1], F0.jpeg, "live")
    state = json.loads(now[-2])
    assert (state.get("type"), state.get("source")) == (STATE, "browser"), state
    assert now[:-2] == base, (
        f"A4/B9: the live reply is BASE's plus the state and the frame;\nbase {base}\n"
        f"now  {[json.loads(m).get('type') for m in now]}"
    )
