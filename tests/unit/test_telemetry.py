"""Span names, thresholds and the turns.jsonl record (docs/02 §8, docs/03 §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_astra.bench.thresholds import THRESHOLDS
from local_astra.contracts.perception import AssistFlags
from local_astra.telemetry.spans import LATENCY_END_SPAN, LATENCY_START_SPAN, SPAN_NAMES
from local_astra.telemetry.turns_log import Regime, TurnRecord

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


def test_span_names_are_unique_and_cover_the_architecture_list():
    assert len(SPAN_NAMES) == len(set(SPAN_NAMES))
    for required in (
        "audio.capture",
        "stt.first_word",
        "stt.end_of_turn",
        "llm.prefill",
        "llm.first_canvas",
        "llm.done",
        "tts.first_audio",
        "playback.first_sample",
    ):
        assert required in SPAN_NAMES
    assert LATENCY_START_SPAN in SPAN_NAMES and LATENCY_END_SPAN in SPAN_NAMES
    assert LATENCY_START_SPAN == "stt.last_word", "B3 counts the end-of-turn decision delay"


def test_thresholds_are_unique_per_bench_metric_and_match_the_design_budgets():
    keys = [(t.bench, t.metric) for t in THRESHOLDS]
    assert len(keys) == len(set(keys))
    by = {(t.bench, t.metric): t.limit for t in THRESHOLDS}
    assert by[("B3", "last_word_to_first_audio_p50_ms")] == 1500  # G1 p50 (docs/01 §8)
    assert by[("B3", "last_word_to_first_audio_p95_ms")] == 2200
    assert by[("B4", "barge_in_pause_p95_ms")] == 300
    assert by[("B7", "vram_total_gib")] == 29.0


def test_turn_record_round_trips_and_computes_last_word_to_first_audio():
    rec = TurnRecord(
        turn_id="t1",
        session_id="s1",
        regime=REGIME,
        outcome="spoken",
        spans={
            "stt.last_word": (900, 1000),
            "stt.end_of_turn": (1600, 1600),
            "llm.first_canvas": (1730, 2500),
            "playback.first_sample": (2900, 2900),
        },
    )
    back = TurnRecord.from_line(rec.to_line())
    assert back == rec
    assert back.latency_ms(LATENCY_START_SPAN, LATENCY_END_SPAN) == 1900  # includes the decision
    assert back.latency_ms("stt.last_word", "stt.end_of_turn") == 600
    assert back.latency_ms("stt.end_of_turn", "tts.first_audio") is None
    assert back.unknown_spans() == []


def test_turn_record_flags_unknown_spans_and_bad_outcomes():
    rec = TurnRecord(
        turn_id="t", session_id="s", regime=REGIME, outcome="spoken", spans={"made.up": (0, 1)}
    )
    assert rec.unknown_spans() == ["made.up"]
    with pytest.raises(ValidationError):
        TurnRecord(turn_id="t", session_id="s", regime=REGIME, outcome="great", spans={})


def test_regime_requires_every_field_that_makes_a_number_comparable():
    required = {n for n, f in Regime.model_fields.items() if f.is_required()}
    assert required == {
        "profile",
        "canvas_length",
        "max_denoising_steps",
        "thinking",
        "vllm_version",
        "model",
        "stt",
        "tts",
        "audio_path",
    }


# --- T-0012: the regime carries the assist flags it was measured with (criteria 7, 8, 13) --------
#
# CLAUDE.md "state the whole regime beside every number": a latency or rubric number measured with
# DOM text on must never be comparable, by accident, to one measured pixel-only. The landmine is
# that ``Regime.assists`` already has a *plausible* default ("pixels"), so a forgotten assignment
# logs a false regime instead of failing — hence the validator and ``with_assists``.

_MISSING = object()


def regime_symbol(name: str):
    """A T-0012 attribute of ``Regime``, reported as a named criterion when it is absent."""
    value = getattr(Regime, name, _MISSING)
    assert value is not _MISSING, f"T-0012 adds Regime.{name} (criterion 7)"
    return value


def require_assist_flags_field() -> None:
    """Criterion 7: the field exists and is *defaulted* (criterion 13 keeps it out of `required`).

    Without this guard the ``ValidationError`` tests below would pass today for the wrong reason:
    ``Contract`` forbids extra fields, so any ``assist_flags=`` keyword raises before a validator
    could ever run.
    """
    field = Regime.model_fields.get("assist_flags")
    assert field is not None, "T-0012 adds Regime.assist_flags: AssistFlags = AssistFlags()"
    assert not field.is_required(), "criterion 13: the new field is defaulted, not required"
    assert field.get_default(call_default_factory=True) == AssistFlags(), "default is all-off"


def test_regime_has_a_defaulted_assist_flags_field_and_an_all_off_default():
    """Criterion 7 + 13. A pre-T-0012 record (no assist fields) stays valid and reads pixel-only."""
    require_assist_flags_field()
    assert REGIME.assist_flags == AssistFlags()
    assert REGIME.assists == "pixels"


@pytest.mark.parametrize(
    ("assists", "flags"),
    [
        ("pixels", AssistFlags(page_text=True)),  # the criterion-7 example
        ("pixels+dom", AssistFlags()),  # badge claims an assist that was off
        ("pixels+dom", AssistFlags(cells=True)),  # right shape, wrong kind
        ("pixels+dom+cells", AssistFlags(page_text=True)),  # half true
    ],
    ids=["pixels-vs-dom", "dom-vs-pixels", "wrong-kind", "half-true"],
)
def test_regime_badge_must_agree_with_the_flags(assists, flags):
    """Criterion 7: ``assists == assist_flags.badge()`` or the record is refused.

    A line whose badge disagrees with its flags is worse than a missing line: it is evidence for
    a claim ("this number was measured pixel-only") that the same line contradicts.
    """
    require_assist_flags_field()
    base = REGIME.model_dump()
    base.pop("assists", None)
    base.pop("assist_flags", None)
    with pytest.raises(ValidationError) as excinfo:
        Regime(**base, assists=assists, assist_flags=flags)
    assert {e["type"] for e in excinfo.value.errors()} != {"extra_forbidden"}, (
        "the error must come from the badge/flags validator, not from extra='forbid'"
    )


@pytest.mark.parametrize(
    ("flags", "badge"),
    [
        (AssistFlags(), "pixels"),
        (AssistFlags(page_text=True), "pixels+dom"),
        (AssistFlags(pdf_text=True), "pixels+pdf"),
        (AssistFlags(cells=True), "pixels+cells"),
        (AssistFlags(at_spi=True), "pixels+a11y"),
        (
            AssistFlags(page_text=True, pdf_text=True, cells=True, at_spi=True),
            "pixels+dom+pdf+cells+a11y",
        ),
    ],
)
def test_a_consistent_badge_and_flag_pair_is_accepted(flags, badge):
    """Criterion 7, the other half: the badge strings are hand-written from docs/12 §3."""
    require_assist_flags_field()
    base = REGIME.model_dump()
    base.pop("assists", None)
    base.pop("assist_flags", None)
    r = Regime(**base, assists=badge, assist_flags=flags)
    assert r.assists == badge and r.assist_flags == flags


@pytest.mark.parametrize(
    "flags",
    [
        AssistFlags(),
        AssistFlags(page_text=True),
        AssistFlags(cells=True, at_spi=True),
        AssistFlags(page_text=True, pdf_text=True, cells=True, at_spi=True),
    ],
)
def test_with_assists_sets_both_fields_and_leaves_the_rest_alone(flags):
    """Criterion 7: the only sanctioned way to set the pair — it cannot set one without the other."""
    regime_symbol("with_assists")
    out = REGIME.with_assists(flags)
    assert isinstance(out, Regime)
    assert out.assist_flags == flags
    assert out.assists == flags.badge()
    untouched = {k: v for k, v in out.model_dump().items() if k not in {"assists", "assist_flags"}}
    before = {k: v for k, v in REGIME.model_dump().items() if k not in {"assists", "assist_flags"}}
    assert untouched == before, "with_assists changes the two assist fields and nothing else"
    assert REGIME.assist_flags == AssistFlags(), "the source regime is frozen and unmoved"


def test_a_regime_round_trips_through_jsonl_with_its_flags():
    """The bench reader gets the structured flags, not just the badge string (docs/03 §7, Q9)."""
    regime_symbol("with_assists")
    flags = AssistFlags(page_text=True, cells=True)
    rec = TurnRecord(
        turn_id="t1",
        session_id="s1",
        regime=REGIME.with_assists(flags),
        outcome="spoken",
        spans={"stt.last_word": (900, 1000), "playback.first_sample": (2900, 2900)},
    )
    line = rec.to_line()
    payload = json.loads(line)
    assert payload["regime"]["assists"] == "pixels+dom+cells"
    assert payload["regime"]["assist_flags"] == {
        "page_text": True,
        "pdf_text": False,
        "cells": True,
        "at_spi": False,
    }
    back = TurnRecord.from_line(line)
    assert back == rec
    assert back.regime.assist_flags == flags
    assert back.regime.assists == flags.badge()


def test_this_task_adds_no_span():
    """Criterion 15: T-0012 adds no stage, so no span name and no bench-reader change."""
    assert not [s for s in SPAN_NAMES if "assist" in s or "crop" in s]


# --- T-0092b criterion 5: Regime.tools and TurnRecord.tool -----------------------------------------

#: A turns.jsonl line in the shape every line had before T-0092b: no ``regime.tools``, no ``tool``. Written by
#: hand from the pre-task model (and the same shape as tests/fixtures/session/t0092a_golden.json's records).
PRE_T0092B_LINE = (
    '{"turn_id":"t1","session_id":"s1","spans":{"stt.last_word":[900,1000]},"events":[],'
    '"regime":{"profile":"lean","canvas_length":256,"max_denoising_steps":48,"thinking":true,'
    '"vllm_version":"vllm/vllm-openai:v0.29.0","model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4",'
    '"stt":"kyutai/stt-1b-en_fr","tts":"pocket-tts@3.1.0","audio_path":"headset","assists":"pixels",'
    '"assist_flags":{"page_text":false,"pdf_text":false,"cells":false,"at_spi":false},'
    '"git_commit":"a502871","seed":null},"input":"voice","outcome":"spoken","canvases":2,'
    '"thought_tokens":0,"answer_tokens":0,"thought_closed":null,"session_log":null,"extras":{}}'
)


def test_regime_tools_defaults_to_off_and_is_not_required():
    """Criterion 5: ``Regime.tools: str = "off"`` — defaulted, so the required set above is unchanged."""
    field = Regime.model_fields.get("tools")
    assert field is not None, "T-0092b criterion 5: Regime.tools exists"
    assert not field.is_required() and field.default == "off"
    assert REGIME.tools == "off"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "value", ["off", "native:bbfc53ccdd80:dry_run", "native:bbfc53ccdd80:dry_run:repair_off"]
)
def test_regime_tools_round_trips_through_a_line(value):
    """Criterion 5's three values (written out by hand from the criterion) survive ``to_line``/``from_line``."""
    assert "tools" in Regime.model_fields, "T-0092b criterion 5: Regime.tools exists"
    regime = REGIME.model_copy(update={"tools": value})
    rec = TurnRecord(turn_id="t", session_id="s", regime=regime, outcome="spoken", spans={})
    back = TurnRecord.from_line(rec.to_line())
    assert json.loads(rec.to_line())["regime"]["tools"] == value
    assert back.regime.tools == value  # type: ignore[attr-defined]


def test_turn_record_tool_is_an_optional_dict_defaulting_to_none():
    """Criterion 5: ``TurnRecord.tool: dict[str, Any] | None = None`` (the schema only; T-0092c writes it)."""
    field = TurnRecord.model_fields.get("tool")
    assert field is not None, "T-0092b criterion 5: TurnRecord.tool exists"
    assert not field.is_required() and field.default is None
    rec = TurnRecord(
        turn_id="t",
        session_id="s",
        regime=REGIME,
        outcome="spoken",
        spans={},
        tool={"call_id": "t-call", "status": "dry_run"},  # type: ignore[call-arg]
    )
    back = TurnRecord.from_line(rec.to_line())
    assert back.tool == {"call_id": "t-call", "status": "dry_run"}  # type: ignore[attr-defined]
    assert (
        json.loads(
            TurnRecord(
                turn_id="t", session_id="s", regime=REGIME, outcome="spoken", spans={}
            ).to_line()
        )["tool"]
        is None
    ), "the key is written, as null, on a turn without a call"


def test_a_line_written_before_t0092b_still_parses_and_reads_tools_off():
    """Criterion 5: every earlier ``turns.jsonl`` line parses; it reads ``tools == "off"`` and ``tool is None``."""
    assert "tools" in Regime.model_fields and "tool" in TurnRecord.model_fields, (
        "T-0092b criterion 5: Regime.tools and TurnRecord.tool exist"
    )
    rec = TurnRecord.from_line(PRE_T0092B_LINE)
    assert rec.regime.tools == "off" and rec.tool is None  # type: ignore[attr-defined]
    golden = json.loads(
        (
            Path(__file__).resolve().parents[2] / "tests/fixtures/session/t0092a_golden.json"
        ).read_text()
    )
    for line in golden["sections"]["records"]:
        back = TurnRecord.from_line(json.dumps(line))
        assert back.regime.tools == "off" and back.tool is None  # type: ignore[attr-defined]


def test_outcome_keeps_its_pattern():
    """Criterion 5: ``outcome`` keeps ``^(spoken|interrupted|error|vetoed|silent)$`` — no tool outcome is added."""
    for ok in ("spoken", "interrupted", "error", "vetoed", "silent"):
        TurnRecord(turn_id="t", session_id="s", regime=REGIME, outcome=ok, spans={})
    for bad in ("tool", "dry_run", "refused", "cancelled"):
        with pytest.raises(ValidationError):
            TurnRecord(turn_id="t", session_id="s", regime=REGIME, outcome=bad, spans={})


# --- T-0092c criterion 7: the two tool spans --------------------------------------------------------------


def test_t0092c_the_tool_spans_are_named_unique_and_known_to_the_record():
    """T-0092c criterion 7: ``SPAN_NAMES`` (and ``SpanName``) gain ``tool.repair`` and ``tool.execute``; the names
    stay unique, and ``TurnRecord.unknown_spans`` accepts both (the bench reader knows them)."""
    import typing

    from local_astra.telemetry.spans import SpanName

    for name in ("tool.repair", "tool.execute"):
        assert name in SPAN_NAMES, f"T-0092c criterion 7: SPAN_NAMES gains {name!r}"
        assert name in typing.get_args(SpanName), f"T-0092c criterion 7: SpanName gains {name!r}"
    assert len(SPAN_NAMES) == len(set(SPAN_NAMES))
    assert set(typing.get_args(SpanName)) == set(SPAN_NAMES), (
        "the Literal and the tuple list the same names"
    )
    rec = TurnRecord(
        turn_id="t1",
        session_id="s",
        regime=REGIME,
        outcome="spoken",
        spans={"tool.repair": (10, 410), "tool.execute": (900, 900)},
    )
    assert rec.unknown_spans() == [], rec.unknown_spans()
