"""T-0177 group A, A1 and A2: the three vision wire types, through the real codec and the page's mirror.

- ``AstraVisionSelect``: ``astra.vision.select``, client->server only, ``source: "browser" | "shared"``;
- ``AstraVisionState``: ``astra.vision.state``, server->client only, ``source``, ``choices``, ``replay = False``;
- ``AstraVisionFrame``: ``astra.vision.frame``, server->client only, ``turn_id``, ``step >= 1``, ``source``,
  ``blind``, ``images: list[VisionImage]`` (at most ``Budget.max_images``), ``history_images >= 0``,
  ``replay = False``;
- ``VisionImage``: ``identity``, ``kind`` in window/screen/camera/overview, ``width >= 1``, ``height >= 1``,
  ``jpeg_b64: str | None`` (null only with ``omitted: "too_large"``).

Every verdict is the real ``codec.decode``'s on JSON written out here by hand, never a model this module built and
compared with itself. The JS rows run ``hud/src/protocol.js``'s ``decode`` in node on the same raw strings and
compare its outcome with the Python codec's (the parity criterion; T-0008's runner, ``node`` >= 18.19).

Readings (listed in the test-author's report):

- "``jpeg_b64`` is null only when ... and then ``omitted: "too_large"`` is set" is read as an equivalence on the
  model: ``jpeg_b64 is None`` iff ``omitted == "too_large"``; ``omitted`` is ``Literal["too_large"]`` (absent by
  default). The size rule itself (null only when the message would exceed ``MAX_MESSAGE_BYTES``) is the
  publisher's, group B's ``too_large`` row;
- the ``images`` bound reads ``prompt_builder.Budget().max_images`` (4 at BASE) as the oracle of "at most
  ``Budget.max_images``";
- no fake or real service is involved except in the A2 rows, so there is no ``backend`` parametrisation here (as in
  ``test_protocol_codec.py`` and ``test_session_reset_wire.py``).

Activated with group A (the ``pending`` marker is off): at BASE the three types do not exist, so the codec answers
``Unknown`` and each row fails on its own assertion, never at collection.

Regime: CPU, no socket except the stub's loopback, node v18.19.1 for the parity rows; n = 1, deterministic.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.prompt_builder import Budget
from local_astra.contracts import protocol
from local_astra.contracts.codec import Malformed, Unknown, decode, encode
from tests.fakes import hud_server as fake

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_JS = REPO / "hud" / "src" / "protocol.js"
DOC02 = REPO / "docs" / "02-architecture.md"
NODE = shutil.which("node")

SELECT = "astra.vision.select"
STATE = "astra.vision.state"
FRAME = "astra.vision.frame"
#: the bound by hand from the oracle the criterion names (4 at BASE 7646b8b6)
MAX_IMAGES = Budget().max_images

JPEG = b"\xff\xd8\xff\xe0" + b"T0177-VISION|" * 8
B64 = base64.b64encode(JPEG).decode()


def model(name: str) -> Any:
    cls = getattr(protocol, name, None)
    assert cls is not None, f"T-0177 A1: contracts/protocol.py gains {name}"
    return cls


def image(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "identity": "capture:0123456789abcdef0123456789abcdef",
        "kind": "window",
        "width": 1280,
        "height": 800,
        "jpeg_b64": B64,
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not ...}


def frame_wire(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": FRAME,
        "turn_id": "t3",
        "step": 2,
        "source": "shared",
        "blind": False,
        "images": [image(kind="screen", width=1568, height=590)],
        "history_images": 1,
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not ...}


# =====================================================================================================
# declaration and direction
# =====================================================================================================


def test_the_three_types_are_declared_with_exactly_their_fields() -> None:
    select, state, frame = (
        model("AstraVisionSelect"),
        model("AstraVisionState"),
        model("AstraVisionFrame"),
    )
    img = model("VisionImage")
    assert select.model_fields["type"].default == SELECT
    assert state.model_fields["type"].default == STATE
    assert frame.model_fields["type"].default == FRAME
    assert set(select.model_fields) == {"type", "source"}, sorted(select.model_fields)
    assert set(state.model_fields) == {"type", "source", "choices", "replay"}, sorted(
        state.model_fields
    )
    assert set(frame.model_fields) == {
        "type",
        "turn_id",
        "step",
        "source",
        "blind",
        "images",
        "history_images",
        "replay",
    }, sorted(frame.model_fields)
    assert set(img.model_fields) == {
        "identity",
        "kind",
        "width",
        "height",
        "jpeg_b64",
        "omitted",
    }, sorted(img.model_fields)


def test_membership_in_the_union_the_tuple_and_one_direction_each() -> None:
    for name in ("AstraVisionSelect", "AstraVisionState", "AstraVisionFrame"):
        assert model(name) in protocol.ASTRA_MESSAGE_TYPES, f"A1: {name} is in ASTRA_MESSAGE_TYPES"
    assert SELECT in protocol.CLIENT_TO_SERVER and SELECT not in protocol.SERVER_TO_CLIENT, (
        "A1: astra.vision.select is client to server only"
    )
    for kind in (STATE, FRAME):
        assert kind in protocol.SERVER_TO_CLIENT and kind not in protocol.CLIENT_TO_SERVER, (
            f"A1: {kind} is server to client only"
        )
    for raw in (
        json.dumps({"type": SELECT, "source": "shared"}),
        json.dumps({"type": STATE, "source": "browser", "choices": ["browser", "shared"]}),
        json.dumps(frame_wire()),
    ):
        got = decode(raw)
        assert not isinstance(got, (Unknown, Malformed)), (
            f"A1: the AstraMessage union decodes {raw[:60]}; got {got!r}"
        )


# =====================================================================================================
# round trips
# =====================================================================================================


@pytest.mark.parametrize("source", ["browser", "shared"])
def test_the_select_round_trips(source: str) -> None:
    raw = json.dumps({"type": SELECT, "source": source})
    got = decode(raw)
    assert isinstance(got, model("AstraVisionSelect")), got
    assert got.source == source
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw)


def test_the_state_round_trips_and_replay_defaults_to_false() -> None:
    raw = json.dumps(
        {"type": STATE, "source": "shared", "choices": ["browser", "shared"], "replay": True}
    )
    got = decode(raw)
    assert isinstance(got, model("AstraVisionState")), got
    assert (got.source, list(got.choices), got.replay) == ("shared", ["browser", "shared"], True)
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw)
    live = decode(json.dumps({"type": STATE, "source": "browser", "choices": ["browser"]}))
    assert isinstance(live, model("AstraVisionState")) and live.replay is False, live


def test_the_frame_round_trips_with_every_field() -> None:
    raw = json.dumps(frame_wire(replay=True))
    got = decode(raw)
    assert isinstance(got, model("AstraVisionFrame")), got
    assert (got.turn_id, got.step, got.source, got.blind, got.history_images, got.replay) == (
        "t3",
        2,
        "shared",
        False,
        1,
        True,
    )
    (img,) = got.images
    assert (img.identity, img.kind, img.width, img.height) == (
        "capture:0123456789abcdef0123456789abcdef",
        "screen",
        1568,
        590,
    )
    assert base64.b64decode(img.jpeg_b64) == JPEG, "the bytes survive the codec"
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw)


def test_a_blind_frame_and_a_too_large_image_round_trip() -> None:
    blind = decode(
        json.dumps(frame_wire(source="browser", blind=True, images=[], history_images=0))
    )
    assert isinstance(blind, model("AstraVisionFrame")) and blind.blind is True, blind
    assert decode(encode(blind)) == blind
    big = image(jpeg_b64=None, omitted="too_large")
    got = decode(json.dumps(frame_wire(images=[big])))
    assert isinstance(got, model("AstraVisionFrame")), got
    (img,) = got.images
    assert img.jpeg_b64 is None and img.omitted == "too_large", img
    assert decode(encode(got)) == got, "the omitted image survives encode's exclude_none"


def test_every_kind_and_the_image_bound_is_accepted() -> None:
    for kind in ("window", "screen", "camera", "overview"):
        got = decode(json.dumps(frame_wire(images=[image(kind=kind)])))
        assert isinstance(got, model("AstraVisionFrame")), f"A1: kind {kind!r}: {got!r}"
    full = [image(identity=f"capture:{i:032x}") for i in range(MAX_IMAGES)]
    got = decode(json.dumps(frame_wire(images=full)))
    assert isinstance(got, model("AstraVisionFrame")) and len(got.images) == MAX_IMAGES, (
        f"A1: exactly Budget.max_images ({MAX_IMAGES}) images is allowed; {got!r}"
    )


# =====================================================================================================
# strictness
# =====================================================================================================


MALFORMED: list[tuple[str, dict[str, Any], str]] = [
    ("select-extra", {"type": SELECT, "source": "shared", "turn_id": "t1"}, "an unknown field"),
    ("select-source", {"type": SELECT, "source": "camera"}, "a source outside the literal"),
    ("select-missing", {"type": SELECT}, "source is required"),
    (
        "state-extra",
        {"type": STATE, "source": "browser", "choices": ["browser"], "badge": "x"},
        "an unknown field",
    ),
    (
        "state-source",
        {"type": STATE, "source": "screen", "choices": ["browser"]},
        "a source outside the literal",
    ),
    (
        "state-choice",
        {"type": STATE, "source": "browser", "choices": ["browser", "camera"]},
        "a choice outside the literal",
    ),
    ("frame-extra", frame_wire(thumbnail="x"), "an unknown field on the frame"),
    ("frame-step0", frame_wire(step=0), "step 0 (step >= 1)"),
    ("frame-step-neg", frame_wire(step=-1), "a negative step"),
    ("frame-history-neg", frame_wire(history_images=-1), "history_images < 0"),
    ("frame-source", frame_wire(source="screen"), "a source outside the literal"),
    (
        "frame-blind-with-images",
        frame_wire(blind=True),
        "blind: true with non-empty images",
    ),
    (
        "frame-too-many",
        frame_wire(images=[image(identity=f"capture:{i:032x}") for i in range(MAX_IMAGES + 1)]),
        "more than Budget.max_images images",
    ),
    ("frame-missing-blind", frame_wire(blind=...), "blind is required"),
    ("image-extra", frame_wire(images=[image(phash="ab")]), "an unknown field on an image"),
    ("image-kind", frame_wire(images=[image(kind="tab")]), "a kind outside the literal"),
    ("image-width0", frame_wire(images=[image(width=0)]), "width 0"),
    ("image-height0", frame_wire(images=[image(height=0)]), "height 0"),
    (
        "image-omitted-with-bytes",
        frame_wire(images=[image(omitted="too_large")]),
        "omitted without a null jpeg_b64",
    ),
    (
        "image-null-without-omitted",
        frame_wire(images=[image(jpeg_b64=None)]),
        "a null jpeg_b64 without omitted (the reading in the module docstring)",
    ),
    (
        "image-omitted-other",
        frame_wire(images=[image(jpeg_b64=None, omitted="hidden")]),
        "omitted outside the literal",
    ),
]


@pytest.mark.parametrize(("name", "wire", "why"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_strictly_decoded(name: str, wire: dict[str, Any], why: str) -> None:
    model("AstraVisionFrame")
    got = decode(json.dumps(wire))
    assert isinstance(got, Malformed), f"T-0177 A1: {why} is malformed; got {got!r}"


def test_the_power_of_the_strictness_rows() -> None:
    """Each malformed row differs from a well-formed message by its one defect: the well-formed shapes decode (so a
    row cannot pass because its base was already broken)."""
    model("AstraVisionFrame")
    for raw in (
        {"type": SELECT, "source": "shared"},
        {"type": STATE, "source": "browser", "choices": ["browser", "shared"]},
        frame_wire(),
        frame_wire(images=[image(jpeg_b64=None, omitted="too_large")]),
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
    """A1: ``hud/src/protocol.js`` mirrors the three types with field tables: on every well-formed and every
    malformed case above, the page's ``decode`` reaches the same outcome as the production codec, and the
    well-formed ones are ``kind: "astra"`` on both sides (an unmirrored type would be ``unknown`` on the page)."""
    model("AstraVisionFrame")
    good = [
        json.dumps({"type": SELECT, "source": "browser"}),
        json.dumps({"type": STATE, "source": "shared", "choices": ["browser", "shared"]}),
        json.dumps({"type": STATE, "source": "browser", "choices": ["browser"], "replay": True}),
        json.dumps(frame_wire()),
        json.dumps(frame_wire(replay=True, history_images=0)),
        json.dumps(frame_wire(blind=True, images=[], source="browser")),
        json.dumps(frame_wire(images=[image(jpeg_b64=None, omitted="too_large")])),
        json.dumps(
            frame_wire(images=[image(kind=k) for k in ("window", "screen", "camera", "overview")])
        ),
    ]
    bad = [json.dumps(w) for _, w, _ in MALFORMED]
    raws = good + bad
    js = _js_outcomes(raws)
    py = [_py_outcome(r) for r in raws]
    assert py[: len(good)] == ["astra"] * len(good), (
        f"premise: the good cases decode in Python; {py}"
    )
    assert py[len(good) :] == ["malformed"] * len(bad), (
        f"premise: the bad cases are malformed; {py}"
    )
    mismatches = [
        (json.loads(raw).get("type"), raw[:120], p, j[0])
        for raw, p, j in zip(raws, py, js, strict=True)
        if p != j[0]
    ]
    assert not mismatches, (
        "T-0177 A1: hud/src/protocol.js disagrees with contracts/protocol.py on "
        f"{len(mismatches)} case(s) (type, raw, python, page): {mismatches[:6]}"
    )


def test_the_page_lists_the_two_server_types() -> None:
    """A1: the page's ``SERVER_TO_CLIENT`` gains the two server->client types (the set-equality row in
    ``test_hud_constants_parity.py`` then holds both ways)."""
    model("AstraVisionState")
    src = PROTOCOL_JS.read_text()
    m = re.search(r"export const SERVER_TO_CLIENT = \[(.*?)\];", src, re.DOTALL)
    assert m is not None, "premise: protocol.js declares SERVER_TO_CLIENT"
    listed = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert {STATE, FRAME} <= listed, (
        f"T-0177 A1: the page dispatches {sorted({STATE, FRAME} - listed)} too"
    )
    assert SELECT not in listed, "astra.vision.select is client to server only"


# =====================================================================================================
# A2: the fake
# =====================================================================================================


def test_the_fake_builds_messages_the_real_codec_round_trips() -> None:
    """A2: ``vision_script`` / ``vision_frame`` / ``vision_image`` build messages that the production codec encodes
    and decodes to themselves, with the field values written here by hand."""
    for name in ("VisionImage", "AstraVisionState", "AstraVisionFrame"):
        model(name)
    img = fake.vision_image(JPEG, identity="capture:aa", kind="screen", width=1568, height=590)
    omitted = fake.vision_image(None, identity="capture:bb")
    frame = fake.vision_frame(
        "t2", 3, source="shared", images=[img, omitted], history_images=2, replay=True
    )
    script = fake.vision_script("shared", frames=[frame])
    assert [m.type for m in script] == [STATE, FRAME]
    for m in script:
        assert decode(encode(m)) == m, m
    state = json.loads(encode(script[0]))
    # ``codec.encode`` is ``exclude_none``, not ``exclude_defaults``: a live ``replay: false`` stays on the wire
    assert state == {
        "type": STATE,
        "source": "shared",
        "choices": ["browser", "shared"],
        "replay": False,
    }, state
    wire = json.loads(encode(frame))
    assert wire["images"][0]["jpeg_b64"] == B64 and wire["images"][1] == {
        "identity": "capture:bb",
        "kind": "window",
        "width": 1280,
        "height": 800,
        "omitted": "too_large",
    }, wire["images"]
    assert (wire["turn_id"], wire["step"], wire["history_images"], wire["replay"]) == (
        "t2",
        3,
        2,
        True,
    )


def test_the_stub_records_a_select_and_replays_a_vision_script() -> None:
    """A2: the stub records the page's ``astra.vision.select`` (decoded by the real codec) and serves a vision
    script after the hello, through a real loopback socket (a plain ``websockets`` client plays the page)."""
    import asyncio

    from websockets.asyncio.client import connect

    model("AstraVisionSelect")
    script = fake.vision_script("browser")

    async def page(url: str) -> list[str]:
        got: list[str] = []
        async with connect(url) as ws:
            await ws.send(
                json.dumps({"type": "astra.hello", "protocol_version": 1, "client": "hud"})
            )
            got.append(str(await asyncio.wait_for(ws.recv(), 5)))
            await ws.send(json.dumps({"type": SELECT, "source": "shared"}))
            await asyncio.sleep(0.2)
        return got

    with fake.HudStub(script) as stub:
        received = asyncio.run(page(stub.url))
        assert stub.wait_for_vision_selects(1), stub.received_types()
        (sel,) = stub.vision_selects()
    assert isinstance(sel, model("AstraVisionSelect")) and sel.source == "shared", sel
    assert [json.loads(r)["type"] for r in received] == [STATE], received


async def test_the_fakes_state_has_the_shape_the_real_server_sends(tmp_path: Path) -> None:
    """A2 (the fake moves with the wire): the real ``HudServer`` over a live session answers a hello with an
    ``astra.vision.state`` whose wire fields equal the fake's ``vision_state("browser")`` (``replay`` aside, whose
    hello value the task leaves open)."""
    from tests.unit.test_t0177_shared_frame import VisionBrowser, vision_rig

    model("AstraVisionState")
    async with vision_rig(tmp_path, VisionBrowser()) as rig:
        states = [m for m in rig.page.sent() if getattr(m, "type", None) == STATE]
    assert states, "A4: a live hello reply carries astra.vision.state"
    real = {k: v for k, v in json.loads(encode(states[-1])).items() if k != "replay"}
    ours = {
        k: v for k, v in json.loads(encode(fake.vision_state("browser"))).items() if k != "replay"
    }
    assert real == ours, f"the fake's state {ours} is not the real server's {real}"


# =====================================================================================================
# docs/02 (A1's spec delta; owned by group A)
# =====================================================================================================


def test_docs02_names_the_types_and_drops_the_never_built_attached_message() -> None:
    """A1 / Owned files: docs/02 §5 gains one ``**Spec delta (T-0177)`` paragraph naming the three types, and §2.1
    no longer promises ``astra.frame.attached`` (never implemented). Keywords, never wording."""
    text = DOC02.read_text()
    s21 = text.split("### 2.1", 1)[1].split("### 2.2", 1)[0]
    assert "astra.frame.attached" not in s21, (
        "A1: the never-built astra.frame.attached sentence is replaced"
    )
    s5 = text.split("## 5. ", 1)[1].split("\n## 6. ", 1)[0]
    paras = [p for p in s5.split("\n\n") if p.lstrip().startswith("**Spec delta (T-0177)")]
    assert len(paras) == 1, f"A1: one §5 paragraph starting '**Spec delta (T-0177)'; {len(paras)}"
    for word in (SELECT, STATE, FRAME, "VisionImage"):
        assert word in paras[0], f"A1: the spec delta names {word}"
