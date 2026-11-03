"""T-0181 group A, A1: the two mic wire types, through the real codec, the page's mirror and the fake.

- ``AstraMicMute``: ``astra.mic.mute``, client->server only, ``muted: bool``;
- ``AstraMicState``: ``astra.mic.state``, server->client only, ``muted: bool``, ``available: bool``,
  ``reason: str | None``; malformed when ``available`` is false and ``muted`` true, when ``available`` is false and
  ``reason`` null, and when ``available`` is true and ``reason`` non-null.

Every verdict is the real ``codec.decode``'s on JSON written out here by hand, never a model this module built and
compared with itself. The parity rows run ``hud/src/protocol.js``'s ``decode`` in node on the same raw strings and
compare its outcome with the Python codec's (T-0008's runner, ``node`` >= 18.19); ``tests/hud/protocol.test.mjs``
holds the page-side rows, run by ``tests/unit/test_hud_js.py``.

Readings (listed in the test-author's report):

- "strict models" is read as the codec's ``Malformed`` for an unknown field *and* for a non-bool ``muted`` /
  ``available`` (JSON ``"true"``, ``1``, ``null``): pydantic's lax ``bool`` would coerce the first two, so the fields
  need a strict bool. ``muted`` and ``available`` are required (no default); ``reason`` defaults to null, because
  ``codec.encode`` is ``exclude_none`` and an available state must survive its own round trip without it;
- ``astra.mic.state`` carries no ``replay`` field (A1 lists exactly three);
- the docs/02 rows are keywords, never wording: §5 gains one paragraph starting ``**Spec delta (T-0181`` that names
  both types (the A0 paragraph sits in §4, before §5), and §2.1 names ``astra.mic.mute``; docs/01's "hardware-style
  mute (keyboard shortcut)" sentence becomes B3's "the mic mute and Stop sharing are buttons; there is no keyboard
  shortcut" (Owned files: docs/01, A, "once H-T181-4 is ruled"; it was ruled 2026-10-28).

No ``backend`` parametrisation: no service is involved (as ``test_t0177_vision_wire.py`` and
``test_protocol_codec.py``); the fake's half is checked against the real ``HudServer`` over a real session instead.

Regime: CPU, no socket except the stub's ephemeral loopback port, node v18.19.1 for the parity rows; n = 1,
deterministic.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import protocol
from local_astra.contracts.codec import Malformed, Unknown, decode, encode
from tests.fakes import hud_server as fake

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_JS = REPO / "hud" / "src" / "protocol.js"
DOC01 = REPO / "docs" / "01-design.md"
DOC02 = REPO / "docs" / "02-architecture.md"
NODE = shutil.which("node")

MUTE = "astra.mic.mute"
STATE = "astra.mic.state"
NO_SEAM = "this audio path has no capture seam"
REASON = "this session has no microphone (--no-mic)"


def model(name: str) -> Any:
    cls = getattr(protocol, name, None)
    assert cls is not None, f"T-0181 A1: contracts/protocol.py gains {name}"
    return cls


def state_wire(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"type": STATE, "muted": False, "available": True}
    base.update(over)
    return {k: v for k, v in base.items() if v is not ...}


# =====================================================================================================
# declaration and direction
# =====================================================================================================


def test_the_two_types_are_declared_with_exactly_their_fields() -> None:
    mute, state = model("AstraMicMute"), model("AstraMicState")
    assert mute.model_fields["type"].default == MUTE
    assert state.model_fields["type"].default == STATE
    assert set(mute.model_fields) == {"type", "muted"}, sorted(mute.model_fields)
    assert set(state.model_fields) == {"type", "muted", "available", "reason"}, sorted(
        state.model_fields
    )


def test_membership_in_the_union_the_tuple_and_one_direction_each() -> None:
    for name in ("AstraMicMute", "AstraMicState"):
        assert model(name) in protocol.ASTRA_MESSAGE_TYPES, f"A1: {name} is in ASTRA_MESSAGE_TYPES"
    assert MUTE in protocol.CLIENT_TO_SERVER and MUTE not in protocol.SERVER_TO_CLIENT, (
        "A1: astra.mic.mute is client to server only"
    )
    assert STATE in protocol.SERVER_TO_CLIENT and STATE not in protocol.CLIENT_TO_SERVER, (
        "A1: astra.mic.state is server to client only"
    )
    for raw in (json.dumps({"type": MUTE, "muted": True}), json.dumps(state_wire())):
        got = decode(raw)
        assert not isinstance(got, (Unknown, Malformed)), (
            f"A1: the AstraMessage union decodes {raw}; got {got!r}"
        )


# =====================================================================================================
# round trips
# =====================================================================================================


@pytest.mark.parametrize("muted", [True, False])
def test_the_mute_round_trips(muted: bool) -> None:
    raw = json.dumps({"type": MUTE, "muted": muted})
    got = decode(raw)
    assert isinstance(got, model("AstraMicMute")), got
    assert got.muted is muted
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw)


@pytest.mark.parametrize(
    "wire",
    [
        pytest.param(state_wire(), id="live"),
        pytest.param(state_wire(muted=True), id="muted"),
        pytest.param(state_wire(available=False, reason=NO_SEAM), id="unavailable"),
        pytest.param(state_wire(available=False, reason=REASON), id="no-mic"),
    ],
)
def test_the_state_round_trips(wire: dict[str, Any]) -> None:
    """A1: each legal state decodes to the model with the fields written here and re-encodes to the same JSON (an
    available state carries no ``reason`` on the wire: ``codec.encode`` is ``exclude_none``)."""
    raw = json.dumps(wire)
    got = decode(raw)
    assert isinstance(got, model("AstraMicState")), got
    assert (got.muted, got.available, got.reason) == (
        wire["muted"],
        wire["available"],
        wire.get("reason"),
    )
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw)


def test_an_explicit_null_reason_is_an_absent_one() -> None:
    """A1: ``reason: null`` on an available state is legal and equals the state without it."""
    with_null = decode(json.dumps(state_wire(reason=None)))
    without = decode(json.dumps(state_wire()))
    assert isinstance(with_null, model("AstraMicState")), with_null
    assert with_null == without


# =====================================================================================================
# strictness
# =====================================================================================================


MALFORMED: list[tuple[str, dict[str, Any], str]] = [
    ("mute-extra", {"type": MUTE, "muted": True, "source": "mic"}, "an unknown field"),
    ("mute-missing", {"type": MUTE}, "muted is required"),
    ("mute-string", {"type": MUTE, "muted": "true"}, "a string muted"),
    ("mute-int", {"type": MUTE, "muted": 1}, "an integer muted"),
    ("mute-null", {"type": MUTE, "muted": None}, "a null muted"),
    ("state-extra", state_wire(replay=True), "an unknown field (no replay on this type)"),
    (
        "state-unavailable-muted",
        state_wire(muted=True, available=False, reason=NO_SEAM),
        "available false and muted true",
    ),
    (
        "state-unavailable-null-reason",
        state_wire(available=False, reason=None),
        "available false and reason null",
    ),
    (
        "state-unavailable-no-reason",
        state_wire(available=False),
        "available false and reason absent (null)",
    ),
    (
        "state-available-with-reason",
        state_wire(reason="a reason"),
        "available true and a non-null reason",
    ),
    ("state-muted-string", state_wire(muted="false"), "a string muted"),
    ("state-muted-int", state_wire(muted=0), "an integer muted"),
    ("state-available-int", state_wire(available=1), "an integer available"),
    ("state-missing-muted", state_wire(muted=...), "muted is required"),
    ("state-missing-available", state_wire(available=...), "available is required"),
    ("state-reason-number", state_wire(available=False, reason=5), "a non-string reason"),
]


@pytest.mark.parametrize(("name", "wire", "why"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_strictly_decoded(name: str, wire: dict[str, Any], why: str) -> None:
    model("AstraMicState")
    got = decode(json.dumps(wire))
    assert isinstance(got, Malformed), f"T-0181 A1: {why} is malformed; got {got!r}"


def test_the_power_of_the_strictness_rows() -> None:
    """Each malformed row differs from a well-formed message by its one defect: the well-formed shapes decode (so a
    row cannot pass because its base was already broken)."""
    model("AstraMicState")
    for raw in (
        {"type": MUTE, "muted": True},
        {"type": MUTE, "muted": False},
        state_wire(),
        state_wire(muted=True),
        state_wire(available=False, reason=NO_SEAM),
    ):
        assert not isinstance(decode(json.dumps(raw)), (Malformed, Unknown)), raw


# =====================================================================================================
# parity with hud/src/protocol.js (node)
# =====================================================================================================

_NODE_SCRIPT = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const mod = await import(pathToFileURL(process.argv[1]).href);
const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map((raw) => { const d = mod.decode(raw); return [d.kind, d.type ?? null]; });
process.stdout.write(JSON.stringify(out));
"""


def _js_outcomes(raws: list[str]) -> list[list[Any]]:
    if NODE is None:
        pytest.skip(
            "node >= 18.19 is a dev prerequisite (T-0008 Regime); the parity rows cannot run"
        )
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", _NODE_SCRIPT, str(PROTOCOL_JS)],
        input=json.dumps(raws),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
    )
    assert proc.returncode == 0, f"node could not run protocol.js's decode:\n{proc.stderr[-2000:]}"
    out: list[list[Any]] = json.loads(proc.stdout)
    return out


def _py_outcome(raw: str) -> str:
    got = decode(raw)
    if isinstance(got, Malformed):
        return "malformed"
    if isinstance(got, Unknown):
        return "unknown"
    return "astra"


def test_the_page_mirror_agrees_with_pydantic_on_every_case() -> None:
    """A1: ``hud/src/protocol.js`` mirrors both types and decodes them strictly: on every well-formed and every
    malformed case above, the page's ``decode`` reaches the production codec's outcome, and the well-formed ones are
    ``kind: "astra"`` on both sides (an unmirrored type would be ``unknown`` on the page)."""
    model("AstraMicState")
    good = [
        json.dumps({"type": MUTE, "muted": True}),
        json.dumps({"type": MUTE, "muted": False}),
        json.dumps(state_wire()),
        json.dumps(state_wire(muted=True)),
        json.dumps(state_wire(reason=None)),
        json.dumps(state_wire(available=False, reason=NO_SEAM)),
    ]
    bad = [json.dumps(w) for _, w, _ in MALFORMED]
    raws = good + bad
    js = _js_outcomes(raws)
    py = [_py_outcome(r) for r in raws]
    assert py[: len(good)] == ["astra"] * len(good), f"premise: the good cases decode; {py}"
    assert py[len(good) :] == ["malformed"] * len(bad), (
        f"premise: the bad cases are malformed; {py}"
    )
    mismatches = [(raw[:120], p, j[0]) for raw, p, j in zip(raws, py, js, strict=True) if p != j[0]]
    assert not mismatches, (
        "T-0181 A1: hud/src/protocol.js disagrees with contracts/protocol.py on "
        f"{len(mismatches)} case(s) (raw, python, page): {mismatches[:8]}"
    )


def test_the_page_lists_the_state_and_not_the_mute() -> None:
    """A1: the page's ``SERVER_TO_CLIENT`` gains ``astra.mic.state`` (the set-equality row in
    ``test_hud_constants_parity.py`` then holds both ways) and never ``astra.mic.mute``."""
    model("AstraMicState")
    src = PROTOCOL_JS.read_text()
    m = re.search(r"export const SERVER_TO_CLIENT = \[(.*?)\];", src, re.DOTALL)
    assert m is not None, "premise: protocol.js declares SERVER_TO_CLIENT"
    listed = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert STATE in listed, "T-0181 A1: the page dispatches astra.mic.state"
    assert MUTE not in listed, "astra.mic.mute is client to server only"


# =====================================================================================================
# the fake moves with the wire
# =====================================================================================================


def test_the_fakes_state_round_trips_through_the_real_codec() -> None:
    """A1: ``tests/fakes/hud_server.mic_state`` builds messages the production codec encodes and decodes to
    themselves, with the wire fields written here by hand."""
    model("AstraMicState")
    live = fake.mic_state()
    muted = fake.mic_state(True)
    gone = fake.mic_state(available=False, reason=REASON)
    for m in (live, muted, gone):
        assert decode(encode(m)) == m, m
    assert json.loads(encode(live)) == {"type": STATE, "muted": False, "available": True}
    assert json.loads(encode(muted)) == {"type": STATE, "muted": True, "available": True}
    assert json.loads(encode(gone)) == {
        "type": STATE,
        "muted": False,
        "available": False,
        "reason": REASON,
    }


def test_the_stub_records_a_mute_and_serves_a_state() -> None:
    """A1: the stub records the page's ``astra.mic.mute`` (decoded by the real codec) and serves a scripted
    ``astra.mic.state`` after the hello, through a real loopback socket on an ephemeral port (a plain ``websockets``
    client plays the page)."""
    from websockets.asyncio.client import connect

    model("AstraMicMute")
    script = [fake.mic_state(True)]

    async def page(url: str) -> list[str]:
        got: list[str] = []
        async with connect(url) as ws:
            await ws.send(
                json.dumps({"type": "astra.hello", "protocol_version": 1, "client": "hud"})
            )
            got.append(str(await asyncio.wait_for(ws.recv(), 5)))
            await ws.send(json.dumps({"type": MUTE, "muted": False}))
            await asyncio.sleep(0.2)
        return got

    with fake.HudStub(script) as stub:
        received = asyncio.run(page(stub.url))
        assert stub.wait_for_mic_mutes(1), stub.received_types()
        (sent,) = stub.mic_mutes()
    assert isinstance(sent, model("AstraMicMute")) and sent.muted is False, sent
    assert [json.loads(r) for r in received] == [
        {"type": STATE, "muted": True, "available": True}
    ], received


@pytest.mark.parametrize(
    ("mic", "want"),
    [
        pytest.param(True, {}, id="mic"),
        pytest.param(False, {"available": False, "reason": NO_SEAM}, id="no-mic"),
    ],
)
async def test_the_fakes_state_has_the_shape_the_real_server_sends(
    tmp_path: Path, mic: bool, want: dict[str, Any]
) -> None:
    """A1 (the fake moves with the wire): the real ``HudServer`` over a real session answers a hello with an
    ``astra.mic.state`` whose wire bytes equal the fake's ``mic_state(...)`` for the same switch."""
    from tests.unit._t0181_rig import mic_rig, mic_states

    model("AstraMicState")
    async with mic_rig(tmp_path, mic=mic) as rig:
        states = mic_states(rig.page.sent())
    assert len(states) == 1, f"A4: a hello reply carries one astra.mic.state; {states}"
    assert encode(states[0]) == encode(fake.mic_state(**want)), (
        f"the fake's {encode(fake.mic_state(**want))} is not the real server's {encode(states[0])}"
    )


# =====================================================================================================
# the documents (A1's spec delta; docs/01's sentence, H-T181-4)
# =====================================================================================================


def test_docs02_section5_has_the_spec_delta_naming_both_types() -> None:
    """A1 / Owned files: docs/02 §5 gains one ``**Spec delta (T-0181`` paragraph naming both types. Keywords only."""
    text = DOC02.read_text()
    s5 = text.split("## 5. ", 1)[1].split("\n## 6. ", 1)[0]
    paras = [p for p in s5.split("\n\n") if p.lstrip().startswith("**Spec delta (T-0181")]
    assert len(paras) == 1, f"A1: one §5 paragraph starting '**Spec delta (T-0181'; {len(paras)}"
    for word in (MUTE, STATE):
        assert word in paras[0], f"A1: the §5 spec delta names {word}"


def test_docs02_section21_names_the_mic_mute() -> None:
    """Owned files: "A: the §2.1 mute sentence": §2.1 (the HUD's control events) names ``astra.mic.mute``."""
    text = DOC02.read_text()
    s21 = text.split("### 2.1", 1)[1].split("### 2.2", 1)[0]
    assert MUTE in s21, "T-0181 A: docs/02 §2.1's mute sentence names astra.mic.mute"


def test_docs01_drops_the_one_action_hardware_mute() -> None:
    """B3 / Owned files (docs/01, A): the "hardware-style mute (keyboard shortcut)" promise is replaced by "the mic
    mute and Stop sharing are buttons; there is no keyboard shortcut". Keywords: the old phrase is gone; the
    paragraph that held it names Stop sharing and says there is no keyboard shortcut."""
    text = DOC01.read_text()
    assert "hardware-style mute (keyboard shortcut)" not in text, (
        "T-0181 B3: docs/01's one-action keyboard mute is replaced (H-T181-4)"
    )
    para = next((p for p in text.split("\n\n") if "Perception is opt-in per source" in p), "")
    assert para, "premise: docs/01's perception paragraph is where it was"
    low = para.lower()
    assert "mic mute" in low and "stop sharing" in low and "no keyboard shortcut" in low, para[
        -400:
    ]
