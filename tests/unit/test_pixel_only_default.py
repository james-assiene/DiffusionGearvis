"""Pixel-only is the default the whole pipeline honours (T-0012 criteria 11, 3, 8).

The headline negative of T-0012: with no checkbox ever touched, assist text sent by the HUD must
reach neither the prompt nor the turn record. "Off by default" is the kind of default that
silently flips — a merge that makes ``AssistState`` filter on read only, a HUD that starts eager,
a prompt builder that forwards what it is given — and every one of those failures still produces
a demo that looks right while the model is reading the DOM.

A **unit** test (ruling A12): it composes ``AssistState`` with ``PromptBuilder`` and the
``Regime`` schema in-process — no service, no socket, no clock — so criterion 11's headline
negative runs in the fast suite and in the Stop hook, where a default that flips gets caught the
same day. The "Tests to write" entry that filed it under ``tests/integration`` was wrong.
"""

from __future__ import annotations

import importlib
from typing import Any

from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.contracts.protocol import AstraAssistToggle, AstraFrameText
from local_astra.telemetry.turns_log import Regime, TurnRecord
from tests.fakes.perception import FakePerception

_MISSING = object()

#: One distinctive string per assist kind. Nothing else in the prompt can produce them, so a
#: single ``in`` over the serialised prompt is a complete answer to "did this text reach the model".
SENTINELS: dict[str, str] = {
    "page_text": "SENTINEL-DOM-b41f9c",
    "pdf_text": "SENTINEL-PDF-27a0d5",
    "cells": "SENTINEL-CELLS-9e3b17",
    "at_spi": "SENTINEL-A11Y-5c8d02",
}

REGIME = Regime(
    profile="lean",
    canvas_length=256,
    max_denoising_steps=48,
    thinking=True,
    vllm_version="v0.29.0",
    model="nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    stt="kyutai/stt-1b-en_fr",
    tts="pocket-tts",
    audio_path="headset",
)

FRAME = Frame(
    source=Source.SCREEN,
    ts_ms=1000,
    width=1920,
    height=1080,
    jpeg=b"\xff\xd8\xff\xe0screen",
    phash="0f0f0f0f0f0f0f0f",
)


def assist_symbol(name: str) -> Any:
    try:
        module = importlib.import_module("local_astra.perception.assists")
    except ModuleNotFoundError as exc:  # pragma: no cover - the pre-implementation path
        raise AssertionError(f"T-0012 adds local_astra/perception/assists.py ({exc})") from exc
    obj = getattr(module, name, _MISSING)
    assert obj is not _MISSING, f"T-0012 adds perception.assists.{name}"
    return obj


def with_assists(regime: Regime, flags: AssistFlags) -> Regime:
    assert hasattr(regime, "with_assists"), "T-0012 adds Regime.with_assists (criterion 7)"
    return regime.with_assists(flags)


def wire_text(kind: str, ts_ms: int = 1000) -> AstraFrameText:
    return AstraFrameText(source=Source.SCREEN, ts_ms=ts_ms, kind=kind, text=SENTINELS[kind])


def everything_the_model_sees(prompt: Any) -> str:
    """Prompt text *and* its serialised form — an assist smuggled into an image part or a field
    name still shows up here."""
    return "".join(m.text() for m in prompt.messages) + prompt.model_dump_json()


async def build_turn(state: Any, perception: FakePerception, at_ms: int = 1500):
    """One turn the way the Conductor will run it: freeze, prefill prefix, then the full request."""
    frame = await perception.get_frame(Source.SCREEN, at_ms=at_ms)
    assert frame is not None
    snapshot = state.freeze()
    builder = PromptBuilder("You are Astra.")
    prefix = builder.prefix((), frame, snapshot)
    prompt = builder.full(
        prefix,
        "what is the total on my screen",
        None,
        [],
        frame_text=state.texts_for(at_ms),
    )
    return snapshot, prompt


async def test_with_no_checkbox_touched_no_assist_text_reaches_the_model():
    """Criterion 11. Four kinds of assist text arrive; none of it may reach the model or the log."""
    state = assist_symbol("AssistState")()
    perception = FakePerception([FRAME])
    for kind in SENTINELS:
        assert state.accept_frame_text(wire_text(kind)) is None, f"{kind} was not ticked"

    snapshot, prompt = await build_turn(state, perception)

    seen = everything_the_model_sees(prompt)
    leaked = sorted(k for k, s in SENTINELS.items() if s in seen)
    assert leaked == [], f"pixel-only turn leaked assist text: {leaked}"
    assert snapshot == AssistFlags()
    assert prompt.assists == AssistFlags()

    record = TurnRecord(
        turn_id="t1",
        session_id="s1",
        regime=with_assists(REGIME, prompt.assists),
        outcome="spoken",
        spans={"stt.last_word": (900, 1000), "playback.first_sample": (2900, 2900)},
    )
    assert record.regime.assists == "pixels"
    assert record.regime.assist_flags == AssistFlags()
    assert TurnRecord.from_line(record.to_line()).regime == record.regime


async def test_one_box_ticked_admits_exactly_that_kind_and_the_record_says_so():
    """Criteria 3 and 8: one kind in, three kinds out, and the log derives from the same object.

    Criterion 8's point is single derivation: the badge comes from the ``Prompt`` the request was
    built from, never from a second copy of the state that may already have moved on — so the
    mid-turn toggle below must not change what this turn's record claims.
    """
    state = assist_symbol("AssistState")()
    perception = FakePerception([FRAME])
    state.toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
    for kind in SENTINELS:
        state.accept_frame_text(wire_text(kind))

    snapshot, prompt = await build_turn(state, perception)
    state.toggle(AstraAssistToggle(flags=AssistFlags(page_text=True, cells=True)))  # mid-turn

    seen = everything_the_model_sees(prompt)
    assert SENTINELS["page_text"] in seen, "the ticked assist reaches the model"
    assert [k for k in ("pdf_text", "cells", "at_spi") if SENTINELS[k] in seen] == []
    assert snapshot == AssistFlags(page_text=True)
    assert prompt.assists == AssistFlags(page_text=True)

    record = TurnRecord(
        turn_id="t2",
        session_id="s1",
        regime=with_assists(REGIME, prompt.assists),
        outcome="spoken",
        spans={"stt.last_word": (900, 1000), "playback.first_sample": (2900, 2900)},
    )
    assert record.regime.assist_flags == prompt.assists
    assert record.regime.assists == prompt.assists.badge() == "pixels+dom"
