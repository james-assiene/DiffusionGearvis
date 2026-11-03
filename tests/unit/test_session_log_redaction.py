"""Redaction: structural, not field-name-based — and not over-broad (criteria 5 and 6).

The log is the one artefact that sees everything the session saw, so it is also the one that must
never hold a pixel. ``redact`` walks the *dumped tree*: a base64 JPEG or a ``data:`` URL is
replaced wherever it sits, whatever the key is called. The adversarial row hides one where no
field name suggests an image — a field-name-based redactor passes every other row in this module
and fails that one.

The other half is criterion 6: thinking **is** logged. A redactor that is over-broad — one that
drops long strings, or everything under ``text`` — passes the negative scan and destroys the
evidence the task exists to collect.

Regime: no measurement; pure functions and one ``SessionLog`` over ``tmp_path``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import AstraFrame, AstraUiSurface
from tests.unit._session_log_rig import (
    b64,
    check_redacted,
    data_url,
    jpeg,
    make_log,
    of_kind,
    read_records,
    scan_for_pixels,
    settle,
    symbol,
)

#: The thought canvas of criterion 6, and the typed question. Both must survive verbatim.
THOUGHT = (
    "The user asked for an illustration. I should build an A2UI surface with a Column and two "
    "Text children, then answer briefly. Let me check the catalog id first."
)
QUESTION = "can you illustrate a half plus a quarter"


def frame_message(payload: bytes, *, title: str = "tab a") -> AstraFrame:
    return AstraFrame(
        source=Source.SCREEN,
        ts_ms=12_000,
        width=1280,
        height=720,
        jpeg_b64=b64(payload),
        phash="0f1e2d3c4b5a6978",
        title=title,
        app="chromium",
    )


def find(tree: object, key: str) -> object:
    """The first value stored under ``key`` anywhere in a dumped tree."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = find(value, key)
            if found is not None:
                return found
    elif isinstance(tree, list | tuple):
        for item in tree:
            found = find(item, key)
            if found is not None:
                return found
    return None


def test_a_captured_frame_is_replaced_by_its_size_digest_and_phash():
    """Criterion 5: ``astra.frame.jpeg_b64`` — 200 KiB of base64 becomes five small fields."""
    redact = symbol("redact")
    payload = jpeg(200 * 1024, tag=b"a")
    dumped = frame_message(payload).model_dump(mode="json")
    original = copy.deepcopy(dumped)

    out = redact(dumped)
    check_redacted(out["jpeg_b64"], payload, what="astra.frame.jpeg_b64", phash="0f1e2d3c4b5a6978")
    assert dumped == original, "criterion 5: redact() is pure — it does not mutate its argument"
    assert out["width"] == 1280 and out["title"] == "tab a", (
        f"…and it changes nothing else about the record; got {out!r:.300}"
    )

    other = jpeg(200 * 1024, tag=b"b")
    out_other = redact(frame_message(other).model_dump(mode="json"))
    assert out_other["jpeg_b64"]["digest"] != out["jpeg_b64"]["digest"], (
        "criterion 5: the digest identifies the image — two different frames must not share one "
        f"(both say {out['jpeg_b64']['digest']!r}); that is what makes it worth writing"
    )


def test_a_data_url_image_part_inside_a_prompt_is_redacted_and_the_text_is_not():
    """Criterion 9: the prompt's text parts are verbatim; its image parts are redacted."""
    redact = symbol("redact")
    payload = jpeg(180 * 1024)
    prompt = Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(
                role="user",
                content=(
                    ImagePart(url=data_url(payload), phash="aabbccddeeff0011"),
                    TextPart(text=QUESTION),
                ),
            ),
        ),
        prefix_len=2,
    )

    out = redact(prompt.model_dump(mode="json"))
    part = out["messages"][1]["content"][0]
    check_redacted(part["url"], payload, what="Prompt image part url", phash="aabbccddeeff0011")
    assert part["url"]["mime"] == "image/jpeg", (
        f"criterion 5: the mime comes from the data: URL it replaced; got {part['url']['mime']!r}"
    )
    assert out["messages"][1]["content"][1]["text"] == QUESTION, (
        "criterion 5 is about pixels: the question the user asked is evidence and stays verbatim"
    )
    assert out["messages"][0]["content"][0]["text"] == "You are Astra.", (
        "…and so does the system prompt"
    )


def test_a_thought_canvas_is_logged_verbatim():
    """Criterion 6, the anti-over-redaction row: thinking IS logged.

    The project rule keeps thinking out of *history*, out of ``astra.spoken`` and out of TTS. The
    log is the artefact one reads when an answer went wrong, so the thought goes in whole. A
    redactor that truncates long strings passes every other row in this module.
    """
    redact = symbol("redact")
    record = {
        "request_id": "gen-1",
        "canvas_idx": 0,
        "segment": "thought",
        "text": THOUGHT,
        "thought_open": True,
    }
    out = redact(record)
    assert out["text"] == THOUGHT, (
        f"criterion 6: the thought canvas is logged verbatim, untruncated; got {out['text']!r}"
    )


def test_a_data_url_where_no_field_name_suggests_an_image_is_still_redacted():
    """Criterion 5, adversarial: structure, not field names.

    Two hiding places, both real: a window ``title`` that happens to hold a data URL (a page whose
    tab title was set from an image), and an ``AnnotatedImage`` source three levels deep inside
    ``AstraUiSurface.messages``, which is a free-form A2UI tree the gate never looks inside for
    URLs. A field-name-based redactor (``jpeg_b64``/``url``/``image``) passes every other row
    here and leaks both of these.
    """
    redact = symbol("redact")
    payload = jpeg(64 * 1024, tag=b"c")
    url = data_url(payload)

    titled = redact(frame_message(jpeg(1024), title=url).model_dump(mode="json"))
    check_redacted(titled["title"], payload, what="astra.frame.title (no field name says image)")

    surface = AstraUiSurface(
        surface_id="turn-t1",
        canvas_idx=1,
        messages=(
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "turn-t1",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["pic"]},
                        {
                            "id": "pic",
                            "component": "AnnotatedImage",
                            "properties": {"source": {"uri": url}, "caption": "one half"},
                        },
                    ],
                },
            },
        ),
    )
    out = redact(surface.model_dump(mode="json"))
    component = out["messages"][0]["updateComponents"]["components"][1]
    check_redacted(
        component["properties"]["source"]["uri"],
        payload,
        what="AnnotatedImage source nested three deep in AstraUiSurface.messages",
    )
    assert component["properties"]["caption"] == "one half", (
        "…and the caption beside it is untouched: redaction replaces images, not trees"
    )
    assert json.dumps(out).find("/9j/") == -1, (
        "criterion 5: nothing base64-JPEG survives anywhere in the dumped surface"
    )


async def test_no_log_file_ever_contains_a_jpeg_or_a_data_url(tmp_path):
    """Criterion 5, end to end: the bytes on disk, scanned.

    The payloads are offered through ``record`` exactly as the taps of criteria 7 and 9 would
    offer them, then the *file* is scanned for ``/9j/``, ``data:image`` and the original base64
    text. The scan is only evidence because the same file provably holds the thought canvas and
    the question: a log that dropped everything would pass it.
    """
    payload = jpeg(200 * 1024, tag=b"d")
    prompt = Prompt(
        messages=(
            Message(
                role="user",
                content=(ImagePart(url=data_url(payload)), TextPart(text=QUESTION)),
            ),
        ),
        prefix_len=1,
    )

    log = make_log(tmp_path)
    await log.start()
    log.record(
        "wire.in",
        {
            "n_bytes": 270_000,
            "outcome": "astra",
            "msg": frame_message(payload).model_dump(mode="json"),
        },
        turn_id="t1",
    )
    log.record(
        "llm.request",
        {"phase": "generate", "request_id": "gen-1", "prompt": prompt.model_dump(mode="json")},
        turn_id="t1",
    )
    log.record(
        "llm.commit",
        {"request_id": "gen-1", "canvas_idx": 0, "segment": "thought", "text": THOUGHT},
        turn_id="t1",
    )
    await settle(log)
    await log.aclose()

    leaks = scan_for_pixels(log.path)
    assert not leaks, (
        f"criterion 5: no file written by this task contains {leaks} — the log is a debug "
        f"artefact, and pixels are never persisted outside the memory store's thumbnails"
    )
    raw = Path(log.path).read_text(encoding="utf-8")
    assert b64(payload)[:64] not in raw, (
        "criterion 5: …nor the original base64 under any other name"
    )
    assert THOUGHT in raw, (
        "criterion 6: the thought canvas is in the same file, verbatim — otherwise the scan above "
        "is vacuous (a log that writes nothing passes it)"
    )
    assert QUESTION in raw, "…and so is the question that was asked"

    records = read_records(log.path)
    frame_record = of_kind(records, "wire.in")[0]
    redacted = find(frame_record["data"]["msg"], "jpeg_b64")
    check_redacted(redacted, payload, what="wire.in astra.frame", phash="0f1e2d3c4b5a6978")
