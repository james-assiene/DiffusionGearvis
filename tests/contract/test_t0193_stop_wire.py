"""T-0193 group A, A-1 and A-9: the ``astra.turn.stop`` wire message, through the real codec, the page's mirror, the
fake and the documents (docs/tasks/T-0193.md, "Tests to write -> A -> test_t0193_stop_wire.py").

``AstraTurnStop``: ``astra.turn.stop``, client to server only, a single strict non-empty field ``turn_id: str``. An
unknown field, a missing, non-string or empty ``turn_id`` is malformed. It joins ``CLIENT_TO_SERVER`` and is mirrored
in ``hud/src/protocol.js``. No server-to-client type is added (the echo is the existing ``astra.turn.state``).

Every verdict is the real ``codec.decode``'s on JSON written out here by hand, never a model this module built and
compared with itself. The parity rows run ``hud/src/protocol.js``'s ``decode`` in node on the same raw strings and
compare its outcome with the Python codec's (T-0008's runner, ``node`` >= 18.19); ``tests/hud/protocol.test.mjs``
holds the page-side rows, run by ``tests/unit/test_hud_js.py``.

Readings (listed in the test-author's report):

- "strict" is read as the codec's ``Malformed`` for a non-string ``turn_id`` of every JSON kind (number, bool, null,
  list, object): pydantic's lax ``str`` would refuse most of these anyway, but not a bool-to-str coercion in some
  modes, so every kind is a row;
- whitespace is not empty: ``" "`` is a legal ``turn_id`` (A-1 says "non-empty", not "non-blank"; a stale or unknown
  turn id is the machine's no-op, A0-2, never the codec's refusal);
- the docs/02 rows are keywords, never wording: §2.1 names ``astra.turn.stop`` (A-9's sentence) and §5 gains one
  paragraph starting ``**Spec delta (T-0193`` that names ``astra.turn.stop``, ``UserStop`` and ``response.cancel``
  (Implementation notes: "a future Realtime adapter can map ``response.cancel`` onto ``UserStop`` (one sentence in
  §5)"); the A0 paragraph sits in §4, before §5, and is not counted.

No ``backend`` parametrisation: no service is involved (as ``test_t0181_mic_wire.py``); the fake's half is checked
against the real ``HudServer`` over a real session instead.

Regime: CPU, no socket except the stub's ephemeral loopback port, node >= 18.19 for the parity rows; n = 1,
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
DOC02 = REPO / "docs" / "02-architecture.md"
NODE = shutil.which("node")

STOP = "astra.turn.stop"

#: ``SERVER_TO_CLIENT`` at BASE 1076bc57 (``contracts/protocol.py``), written out by hand: A-1 adds no
#: server-to-client type
BASE_SERVER_TO_CLIENT = frozenset(
    {
        "astra.assist.state",
        "astra.turn.state",
        "astra.ui.surface",
        "astra.ui.fence",
        "astra.canvas.step",
        "astra.canvas.commit",
        "astra.canvas.synthetic",
        "astra.transcript",
        "astra.spoken",
        "astra.spoken.played",
        "astra.log.state",
        "astra.tool.call",
        "astra.session.cleared",
        "astra.error",
        "astra.vision.state",
        "astra.vision.frame",
        "astra.thought",
        "astra.mic.state",
    }
)
#: ``CLIENT_TO_SERVER`` at BASE 1076bc57, by hand; A-1 adds exactly ``astra.turn.stop``
BASE_CLIENT_TO_SERVER = frozenset(
    {
        "astra.hello",
        "astra.frame",
        "astra.frame.text",
        "astra.assist.toggle",
        "astra.ui.action",
        "astra.text",
        "astra.log.toggle",
        "astra.session.reset",
        "astra.vision.select",
        "astra.mic.mute",
    }
)


def model() -> Any:
    cls = getattr(protocol, "AstraTurnStop", None)
    assert cls is not None, "T-0193 A-1: contracts/protocol.py gains AstraTurnStop"
    return cls


def raw(**fields: Any) -> str:
    return json.dumps({"type": STOP, **fields})


# =====================================================================================================
# declaration and direction
# =====================================================================================================


def test_declared_with_exactly_its_one_field() -> None:
    cls = model()
    assert cls.model_fields["type"].default == STOP
    assert set(cls.model_fields) == {"type", "turn_id"}, sorted(cls.model_fields)
    assert cls.model_fields["turn_id"].is_required(), "A-1: turn_id has no default"


def test_client_to_server_only_and_no_new_server_type() -> None:
    """A-1: ``astra.turn.stop`` is in ``ASTRA_MESSAGE_TYPES`` and ``CLIENT_TO_SERVER`` only; ``CLIENT_TO_SERVER`` is
    BASE's plus exactly this type, and ``SERVER_TO_CLIENT`` is BASE's unchanged (the echo is ``astra.turn.state``)."""
    assert model() in protocol.ASTRA_MESSAGE_TYPES, "A-1: AstraTurnStop is in ASTRA_MESSAGE_TYPES"
    # T-0213 A2a-1 (named re-pin): + astra.settings.set, a later task's client->server type; A-1 adds only STOP
    later_in = {"astra.settings.set"}
    assert set(protocol.CLIENT_TO_SERVER) == BASE_CLIENT_TO_SERVER | {STOP} | later_in, sorted(
        set(protocol.CLIENT_TO_SERVER) ^ (BASE_CLIENT_TO_SERVER | {STOP} | later_in)
    )
    # T-0213 A2r-1 (R-5 named re-pin): + astra.settings.state, a later task's server->client type; A-1 adds none
    later = {"astra.settings.state"}
    assert set(protocol.SERVER_TO_CLIENT) == BASE_SERVER_TO_CLIENT | later, sorted(
        set(protocol.SERVER_TO_CLIENT) ^ (BASE_SERVER_TO_CLIENT | later)
    )
    names = [m.model_fields["type"].default for m in protocol.ASTRA_MESSAGE_TYPES]
    assert names.count(STOP) == 1, f"A-1: joins once; {names.count(STOP)}"


# =====================================================================================================
# round trips
# =====================================================================================================


@pytest.mark.parametrize("turn_id", ["t1", "t12", " ", "t-0193-ä"])
def test_round_trips(turn_id: str) -> None:
    wire = raw(turn_id=turn_id)
    got = decode(wire)
    assert isinstance(got, model()), got
    assert got.turn_id == turn_id
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == {"type": STOP, "turn_id": turn_id}


# =====================================================================================================
# strictness
# =====================================================================================================

MALFORMED: list[tuple[str, dict[str, Any], str]] = [
    ("missing", {}, "turn_id is required"),
    ("empty", {"turn_id": ""}, "an empty turn_id"),
    ("int", {"turn_id": 3}, "an integer turn_id"),
    ("float", {"turn_id": 1.5}, "a float turn_id"),
    ("bool", {"turn_id": True}, "a bool turn_id"),
    ("null", {"turn_id": None}, "a null turn_id"),
    ("list", {"turn_id": ["t1"]}, "a list turn_id"),
    ("object", {"turn_id": {"id": "t1"}}, "an object turn_id"),
    ("extra", {"turn_id": "t1", "x": 1}, "an unknown field"),
    (
        "extra-replay",
        {"turn_id": "t1", "replay": False},
        "an unknown field (no replay on this type)",
    ),
]


@pytest.mark.parametrize(("name", "fields", "why"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_strictly_decoded(name: str, fields: dict[str, Any], why: str) -> None:
    model()
    got = decode(raw(**fields))
    assert isinstance(got, Malformed), f"T-0193 A-1: {why} is malformed; got {got!r}"


def test_the_power_of_the_strictness_rows() -> None:
    """Each malformed row differs from a well-formed stop by its one defect: the well-formed shape decodes (so a row
    cannot pass because its base was already broken, e.g. the type unknown)."""
    model()
    assert isinstance(decode(raw(turn_id="t1")), model())
    assert not isinstance(decode(raw(turn_id="t1")), (Malformed, Unknown))


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


def _py_outcome(wire: str) -> str:
    got = decode(wire)
    if isinstance(got, Malformed):
        return "malformed"
    if isinstance(got, Unknown):
        return "unknown"
    return "astra"


def test_the_page_mirror_agrees_with_pydantic_on_every_case() -> None:
    """A-1: ``hud/src/protocol.js`` mirrors the type and decodes it strictly: on every well-formed and malformed case
    above the page's ``decode`` reaches the production codec's outcome, and the well-formed ones are
    ``kind: "astra"`` on both sides (an unmirrored type would be ``unknown`` on the page)."""
    model()
    good = [raw(turn_id="t1"), raw(turn_id=" "), raw(turn_id="t12")]
    bad = [raw(**f) for _, f, _ in MALFORMED]
    raws = good + bad
    js = _js_outcomes(raws)
    py = [_py_outcome(r) for r in raws]
    assert py[: len(good)] == ["astra"] * len(good), f"premise: the good cases decode; {py}"
    assert py[len(good) :] == ["malformed"] * len(bad), (
        f"premise: the bad cases are malformed; {py}"
    )
    mismatches = [(r[:120], p, j[0]) for r, p, j in zip(raws, py, js, strict=True) if p != j[0]]
    assert not mismatches, (
        "T-0193 A-1: hud/src/protocol.js disagrees with contracts/protocol.py on "
        f"{len(mismatches)} case(s) (raw, python, page): {mismatches[:8]}"
    )
    assert all(j[1] == STOP for j in js[len(good) :]), (
        f"the page names the type of a malformed stop (for its counter); {js[len(good) :]}"
    )


def test_the_page_never_dispatches_the_stop() -> None:
    """A-1: client to server only, so the page's ``SERVER_TO_CLIENT`` does not list it (and the set-equality row in
    ``test_hud_constants_parity.py`` keeps holding both ways)."""
    model()
    src = PROTOCOL_JS.read_text()
    m = re.search(r"export const SERVER_TO_CLIENT = \[(.*?)\];", src, re.DOTALL)
    assert m is not None, "premise: protocol.js declares SERVER_TO_CLIENT"
    listed = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert STOP not in listed, "astra.turn.stop is client to server only"
    assert "astra.turn.state" in listed, "premise: the echo is dispatched"


# =====================================================================================================
# the fake moves with the wire
# =====================================================================================================


def test_the_stub_records_a_stop() -> None:
    """``HudStub`` records the page's ``astra.turn.stop`` (decoded by the real codec) through a real loopback socket
    on an ephemeral port (a plain ``websockets`` client plays the page); a malformed stop is not counted as one."""
    from websockets.asyncio.client import connect

    model()

    async def page(url: str) -> None:
        async with connect(url) as ws:
            await ws.send(
                json.dumps({"type": "astra.hello", "protocol_version": 1, "client": "hud"})
            )
            await ws.send(raw(turn_id="t3"))
            await ws.send(raw(turn_id=""))
            await asyncio.sleep(0.2)

    with fake.HudStub() as stub:
        asyncio.run(page(stub.url))
        assert stub.wait_for_turn_stops(1), stub.received_types()
        stops = stub.turn_stops()
    assert len(stops) == 1, f"one well-formed stop, the empty one is malformed; {stops}"
    assert isinstance(stops[0], model()) and stops[0].turn_id == "t3", stops


async def test_the_stub_and_the_real_server_agree_on_the_same_bytes(tmp_path: Path) -> None:
    """The fake's twin (CLAUDE.md, a fake moves with the wire): the bytes the stub records as one stop are routed by
    the real ``HudServer`` over a real session into exactly one ``UserStop`` with that ``turn_id``."""
    from tests.unit._t0181_rig import mic_rig

    model()
    wire = raw(turn_id="t9")
    recorded = decode(wire)
    assert isinstance(recorded, model())
    async with mic_rig(tmp_path) as rig:
        rig.page.receive(wire)
        await rig.pump()
        stepped = rig.stepped("user.stop")
    assert [(e.type, e.turn_id) for e in stepped] == [("user.stop", recorded.turn_id)], stepped


# =====================================================================================================
# the documents (A-9)
# =====================================================================================================


def test_docs02_section5_has_the_spec_delta() -> None:
    """A-9: docs/02 §5 gains one ``**Spec delta (T-0193`` paragraph naming the message, the machine event it becomes
    and the Realtime ``response.cancel`` it may later be mapped from. Keywords only."""
    text = DOC02.read_text()
    s5 = text.split("## 5. ", 1)[1].split("\n## 6. ", 1)[0]
    paras = [p for p in s5.split("\n\n") if p.lstrip().startswith("**Spec delta (T-0193")]
    assert len(paras) == 1, f"A-9: one §5 paragraph starting '**Spec delta (T-0193'; {len(paras)}"
    for word in (STOP, "UserStop", "response.cancel"):
        assert word in paras[0], f"A-9: the §5 spec delta names {word}"


def test_docs02_section21_names_the_stop() -> None:
    """A-9: §2.1 (the HUD's control events) gains one sentence naming ``astra.turn.stop``."""
    text = DOC02.read_text()
    s21 = text.split("### 2.1", 1)[1].split("### 2.2", 1)[0]
    assert STOP in s21, "T-0193 A-9: docs/02 §2.1 names astra.turn.stop"
