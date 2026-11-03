"""Pixel-first assist state: the ingress filter and the per-turn snapshot (T-0012, docs/12 §3).

Task: docs/tasks/T-0012.md. Criteria 1-6 and the "no clock, no asyncio, no I/O" implementation
rule live here. Nothing in this module touches a clock, a socket or the filesystem: timestamps
arrive as ``at_ms`` arguments, the same discipline the turn machine follows.

``local_astra/perception/assists.py`` lands with T-0012, so every symbol is fetched through
``assist_symbol`` — a missing name must read as a named, failed criterion, not as an ImportError
half-way through collection (the convention ``tests/unit/test_turn_invariants.py`` set for T-0003).
The rulings of 2026-09-13 (A9-A12, gaps T1-T6) add the stale-after-silence, fail-closed-ingress,
no-snapshot, kind-order, zero-entry and mid-turn-untick cases below.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.perception import AssistFlags, FrameText, Source
from local_astra.contracts.protocol import AstraAssistState, AstraAssistToggle, AstraFrameText
from local_astra.perception.policy import RING_BUFFER_S

MODULE = "local_astra.perception.assists"

#: The four assist kinds. ``AssistFlags`` field names and ``FrameText.kind`` values coincide.
KINDS: tuple[str, ...] = ("page_text", "pdf_text", "cells", "at_spi")

_MISSING = object()


def assist_module() -> Any:
    """Import ``perception.assists``, reporting its absence as a named criterion."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:  # pragma: no cover - the pre-implementation path
        raise AssertionError(f"T-0012 adds {MODULE.replace('.', '/')}.py ({exc})") from exc


def assist_symbol(name: str) -> Any:
    """Fetch a T-0012 symbol from ``perception.assists`` with a readable criterion message."""
    obj = getattr(assist_module(), name, _MISSING)
    assert obj is not _MISSING, f"T-0012 adds {MODULE.rsplit('.', 1)[-1]}.{name}"
    return obj


def new_state() -> Any:
    """A freshly constructed ``AssistState`` (criterion 1: no arguments needed, none accepted)."""
    return assist_symbol("AssistState")()


def flags(*on: str) -> AssistFlags:
    assert set(on) <= set(KINDS)
    return AssistFlags(**{k: True for k in on})


def wire_text(kind: str, text: str, ts_ms: int = 1000, url: str | None = None) -> AstraFrameText:
    return AstraFrameText(source=Source.SCREEN, ts_ms=ts_ms, kind=kind, text=text, url=url)


def drop_counts(state: Any) -> dict[str, int]:
    """The per-kind drop counter of criterion 3, whether exposed as a mapping or a call."""
    counter = getattr(state, "dropped", _MISSING)
    assert counter is not _MISSING, (
        "criterion 3: AssistState exposes a readable per-kind drop counter as `.dropped`"
    )
    if callable(counter):
        counter = counter()
    return {str(k): int(v) for k, v in dict(counter).items()}


def texts(state: Any, at_ms: int) -> tuple[FrameText, ...]:
    out = tuple(state.texts_for(at_ms))
    assert all(isinstance(t, FrameText) for t in out), "texts_for yields contracts.FrameText"
    return out


def kinds_of(state: Any, at_ms: int) -> tuple[str, ...]:
    return tuple(t.kind for t in texts(state, at_ms))


# --- criterion 1: all off unless ticked ----------------------------------------------------------


def test_a_new_session_is_pixel_only():
    """Criterion 1. The badge string is asked of the flags, never spelled out a second time."""
    state = new_state()
    assert state.flags == AssistFlags()
    assert not state.flags.any_on()
    assert state.flags.badge() == "pixels"  # hand-written: docs/12 §3 badge for pixels-only
    msg = state.state_message()
    assert isinstance(msg, AstraAssistState)
    assert msg.type == "astra.assist.state"
    assert msg.flags == AssistFlags()
    assert msg.badge == state.flags.badge()


def test_no_constructor_argument_or_environment_variable_can_turn_an_assist_on(monkeypatch):
    """Criterion 1: the *sole* transition to on is an ``astra.assist.toggle`` message.

    A config key or an env var that pre-arms an assist would make a recording dishonest without
    anybody ticking a box, which is the whole point of the pixel-first claim.
    """
    state_cls = assist_symbol("AssistState")
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    params = [
        p
        for n, p in inspect.signature(state_cls.__init__).parameters.items()
        if n != "self" and p.kind in named
    ]
    assert all(p.default is not inspect.Parameter.empty for p in params), (
        "AssistState() must be constructible with no arguments"
    )
    assert not any(
        "AssistFlags" in str(p.annotation) or p.name in {"flags", *KINDS} for p in params
    ), "criterion 1: no constructor argument may pre-arm an assist"
    for var in (
        "ASTRA_ASSISTS",
        "ASTRA_ASSIST_FLAGS",
        "ASTRA_ASSIST_PAGE_TEXT",
        "ASTRA_PAGE_TEXT",
        "ASTRA_ASSIST_CELLS",
        "ASTRA_ASSIST_AT_SPI",
        "ASTRA_ASSIST_PDF_TEXT",
    ):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("ASTRA_ASSISTS", "page_text,pdf_text,cells,at_spi")
    assert new_state().flags == AssistFlags(), "an environment variable must not arm an assist"


# --- criterion 2: toggle handling ----------------------------------------------------------------


@pytest.mark.parametrize(
    "on",
    [(), ("page_text",), ("pdf_text",), ("cells",), ("at_spi",), KINDS],
    ids=["none", "dom", "pdf", "cells", "a11y", "all"],
)
def test_toggle_replaces_the_flag_set_and_echoes_a_matching_badge(on):
    """Criterion 2: the echo carries the flags that are now in force and their own badge."""
    state = new_state()
    want = flags(*on)
    echo = state.toggle(AstraAssistToggle(flags=want))
    assert isinstance(echo, AstraAssistState)
    assert echo.type == "astra.assist.state"
    assert echo.flags == want
    assert echo.badge == want.badge()
    assert state.flags == want
    assert state.state_message() == echo, "state_message() is the same echo, recomputed"


def test_toggle_is_a_replace_not_a_merge():
    """Criterion 2 + implementation note: the message carries the *whole* flag set."""
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text", "cells")))
    echo = state.toggle(AstraAssistToggle(flags=flags("pdf_text")))
    assert state.flags == AssistFlags(pdf_text=True)
    assert state.flags.page_text is False and state.flags.cells is False
    assert echo.badge == "pixels+pdf"  # hand-written from contracts.perception.AssistFlags.badge


# --- criterion 3: the ingress filter -------------------------------------------------------------


def test_frame_text_for_an_off_kind_is_dropped_and_counted():
    """Criterion 3: four kinds off, one message per kind in, four ``None`` out, counted per kind.

    Defence in depth behind the prompt-side filter (T-0005): an eager or buggy HUD must not be
    able to put text in front of the model.
    """
    state = new_state()
    for i, kind in enumerate(KINDS):
        assert state.accept_frame_text(wire_text(kind, f"DROP-{kind}", ts_ms=1000 + i)) is None
    assert drop_counts(state) == dict.fromkeys(KINDS, 1)
    state.freeze()
    assert texts(state, 10_000) == (), "a dropped message must never be buffered"


def test_dropped_text_cannot_be_resurrected_by_ticking_the_box_later():
    """Criterion 3, the observable half: 'never buffered' means a later tick cannot surface it."""
    state = new_state()
    state.accept_frame_text(wire_text("page_text", "SECRET DOM", ts_ms=1000))
    state.toggle(AstraAssistToggle(flags=flags(*KINDS)))
    state.freeze()
    assert texts(state, 10_000) == ()


def test_accepted_frame_text_carries_the_message_payload():
    """Criterion 3, positive: an on kind becomes a ``FrameText`` with the message's own fields."""
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text")))
    msg = wire_text("page_text", "A1 12 A2 30", ts_ms=1234, url="https://example.invalid/sheet")
    got = state.accept_frame_text(msg)
    assert got == FrameText(
        source=Source.SCREEN,
        ts_ms=1234,
        kind="page_text",
        text="A1 12 A2 30",
        url="https://example.invalid/sheet",
    )
    assert drop_counts(state) == dict.fromkeys(KINDS, 0), "an accepted message is not a drop"


def test_the_drop_counter_has_a_zero_entry_for_every_kind():
    """A1 (ruling): ``dropped`` is keyed by assist kind and reads *zero* for kinds never dropped.

    A counter that only grows a key on the first drop is indistinguishable, from the caller's
    side, from a counter that lost the kind: ``dropped.get("cells", 0)`` reads 0 both when cells
    text was never refused and when the filter stopped looking at cells at all. The zero entries
    are what makes "nothing was dropped" a statement rather than an absence.
    """
    state = new_state()
    assert drop_counts(state) == dict.fromkeys(KINDS, 0), "a fresh state names every kind at zero"

    state.accept_frame_text(wire_text("cells", "DROP cells", ts_ms=1000))
    assert drop_counts(state) == {"page_text": 0, "pdf_text": 0, "cells": 1, "at_spi": 0}

    state.accept_frame_text(wire_text("cells", "DROP cells again", ts_ms=1100))
    state.accept_frame_text(wire_text("at_spi", "DROP a11y", ts_ms=1200))
    assert drop_counts(state) == {"page_text": 0, "pdf_text": 0, "cells": 2, "at_spi": 1}


@pytest.mark.parametrize("kind", ["badge", "any_on", "dom"], ids=["badge", "any_on", "dom"])
def test_a_kind_that_is_not_an_assist_flag_field_is_dropped_and_counted(kind):
    """A10 (ruling): the ingress filter fails closed. **Adversarial.**

    ``getattr(flags, kind)`` alone is true for every *method* on ``AssistFlags``: a bound method
    is truthy, so a message labelled ``kind="badge"`` or ``kind="any_on"`` walks past the filter
    with all four boxes unticked and its text is buffered as if the user had ticked one. ``"dom"``
    is the badge spelling of ``page_text`` — the name a HUD author is most likely to send by
    mistake. The wire type is a ``Literal``, so such a message can only arrive through a codec
    bug, a hand-built frame or a second client; ``model_construct`` is how the test reproduces
    that without asking the codec to accept something it must not accept.

    The ruling says "dropped and counted" without naming the key, so this pins what is
    observable: nothing admitted, nothing buffered, the four real kinds still at zero, and the
    refusal visible in the counter's total.
    """
    state = new_state()  # no box ever ticked
    message = AstraFrameText.model_construct(
        type="astra.frame.text",
        source=Source.SCREEN,
        ts_ms=1000,
        kind=kind,
        text=f"SMUGGLED-{kind}",
        url=None,
    )
    assert message.kind == kind, "model_construct built the message the filter must refuse"

    assert state.accept_frame_text(message) is None, f"kind={kind!r} must never be admitted"
    counts = drop_counts(state)
    assert {k: counts[k] for k in KINDS} == dict.fromkeys(KINDS, 0), "no real kind was dropped"
    assert sum(counts.values()) == 1, "the refusal is counted somewhere readable"

    state.toggle(AstraAssistToggle(flags=flags(*KINDS)))  # every box ticked afterwards
    state.freeze()
    assert "SMUGGLED" not in "".join(t.text for t in texts(state, 10_000)), "never buffered"


# --- criterion 4: turning an assist off purges what it captured ----------------------------------


def test_turning_an_assist_off_purges_text_captured_while_it_was_on():
    """Criterion 4, adversarial.

    The likely implementation filters on *read* only: the DOM dump buffered while the box was
    ticked survives the untick and reappears the moment the box is ticked again (or on any turn
    whose snapshot happens to include the kind). Ticking ``page_text`` a second time must not
    resurrect text captured before the untick.
    """
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text", "cells")))
    state.accept_frame_text(wire_text("page_text", "SECRET DOM", ts_ms=1000))
    state.accept_frame_text(wire_text("cells", "A1 12", ts_ms=1000))

    state.toggle(AstraAssistToggle(flags=flags("cells")))  # user unticks page text
    state.freeze()
    assert kinds_of(state, 5000) == ("cells",)

    state.toggle(AstraAssistToggle(flags=flags("page_text", "cells")))  # ticks it again
    state.freeze()
    got = texts(state, 5000)
    assert "SECRET DOM" not in "".join(t.text for t in got), (
        "criterion 4: unticking purges what that assist captured; it does not merely hide it"
    )
    assert [t.kind for t in got] == ["cells"]


# --- criterion 6: texts_for ----------------------------------------------------------------------


def test_texts_for_returns_one_newest_entry_per_on_kind_at_or_before_the_timestamp():
    """Criterion 6: newest at-or-before ``at_ms``, one per kind, off kinds absent."""
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text", "cells")))
    state.accept_frame_text(wire_text("page_text", "dom old", ts_ms=1000))
    state.accept_frame_text(wire_text("page_text", "dom new", ts_ms=2000))
    state.accept_frame_text(wire_text("page_text", "dom future", ts_ms=3000))
    state.accept_frame_text(wire_text("cells", "cells old", ts_ms=1500))
    state.accept_frame_text(wire_text("pdf_text", "pdf never on", ts_ms=1500))
    state.freeze()

    got = texts(state, 2500)
    assert {t.kind: t.text for t in got} == {"page_text": "dom new", "cells": "cells old"}
    assert len(got) == 2, "at most one entry per kind"
    assert texts(state, 2000) == texts(state, 2500), "'at or before' includes the boundary"
    assert {t.text for t in texts(state, 3000)} == {"dom future", "cells old"}
    assert texts(state, 999) == (), "nothing captured at or before the timestamp"


def test_texts_for_order_is_deterministic_and_is_not_arrival_order():
    """Criterion 6: 'a deterministic kind order' — so the prompt (and its cache) is reproducible.

    Arrival order is the accident the caller must not inherit: two sessions that saw the same
    assist text in different orders must produce the same prompt suffix.
    """
    forward, backward = new_state(), new_state()
    for state, order in ((forward, KINDS), (backward, tuple(reversed(KINDS)))):
        state.toggle(AstraAssistToggle(flags=flags(*KINDS)))
        for i, kind in enumerate(order):
            state.accept_frame_text(wire_text(kind, f"{kind} text", ts_ms=1000 + i))
        state.freeze()
    a, b = kinds_of(forward, 5000), kinds_of(backward, 5000)
    assert sorted(a) == sorted(KINDS), "all four on kinds are present exactly once"
    assert a == b, f"kind order must not depend on arrival order: {a} vs {b}"
    assert a == kinds_of(forward, 5000), "and it is stable across calls"


def test_the_buffer_is_bounded_to_the_ring_horizon():
    """Criterion 6: bounded by the shared 60 s horizon (.claude/rules/perception-memory.md).

    Ten horizons of one-per-second DOM dumps go in; what survives is the recent tail, and the
    oldest entries are the ones gone. Unbounded, 601 entries would still be retrievable.
    """
    horizon_ms = RING_BUFFER_S * 1000
    newest = 10 * horizon_ms
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text")))
    for ts in range(0, newest + 1, 1000):
        state.accept_frame_text(wire_text("page_text", f"dom at {ts}", ts_ms=ts))
    state.freeze()

    retained = sorted({t.ts_ms for ts in range(0, newest + 1, 1000) for t in texts(state, ts)})
    assert retained, "the recent entries survive"
    assert len(retained) <= horizon_ms // 1000 + 2, (
        f"bounded to the {RING_BUFFER_S} s horizon, got {len(retained)} of 601 entries"
    )
    assert min(retained) >= newest - horizon_ms - 1000, "the oldest entries are the ones gone"
    assert max(retained) == newest
    assert texts(state, newest - 5 * horizon_ms) == (), "five horizons back is gone"
    assert [t.text for t in texts(state, newest)] == [f"dom at {newest}"]


def test_texts_for_before_the_first_freeze_is_empty_even_with_every_box_ticked():
    """A2 (ruling): no snapshot means nothing is admitted.

    The tempting shortcut is to fall back to the live flags when no turn has been frozen. That
    hands a caller who forgot to freeze whatever the HUD ticked a millisecond ago, and makes the
    snapshot in the turn record a fiction; pixel-only is the safe answer.
    """
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags(*KINDS)))
    for i, kind in enumerate(KINDS):
        assert state.accept_frame_text(wire_text(kind, f"{kind} text", ts_ms=1000 + i)) is not None
    assert state.flags == flags(*KINDS), "the live flags really are all on"

    assert texts(state, 5000) == (), "texts_for() before any freeze() is empty"

    state.freeze()
    assert sorted(kinds_of(state, 5000)) == sorted(KINDS), "the same buffer after a freeze"


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        (KINDS, ("page_text", "pdf_text", "cells", "at_spi")),
        (("at_spi", "page_text"), ("page_text", "at_spi")),
        (("cells", "pdf_text"), ("pdf_text", "cells")),
    ],
    ids=["all-four", "a11y+dom", "cells+pdf"],
)
def test_the_kind_order_is_the_assist_flags_declaration_order(on, expected):
    """A3 (ruling): "a deterministic kind order" is ``AssistFlags``' field declaration order.

    Spelled out here from contracts/perception.py (``page_text, pdf_text, cells, at_spi``) rather
    than read back from the model, so an implementation that sorts the kinds alphabetically, or
    reverses them, or follows arrival order, fails. The order is load-bearing twice over: the
    prompt suffix must be reproducible for the prefix cache, and two recordings of the same
    session must produce the same bytes.
    """
    assert tuple(AssistFlags.model_fields) == KINDS, "criterion 12: flag field names and order"
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags(*on)))
    for i, kind in enumerate(reversed(KINDS)):  # arrival order is the reverse of the declaration
        state.accept_frame_text(wire_text(kind, f"{kind} text", ts_ms=1000 + i))
    state.freeze()
    assert kinds_of(state, 5000) == expected


def test_assist_text_older_than_the_horizon_is_not_served_as_current_context():
    """A9 (ruling), adversarial: the horizon is a window on the *turn's* time, not on the ingest.

    The HUD goes quiet — a background tab, a crashed extension, a user who closed the browser —
    and the last DOM dump it sent is ten minutes old. Bounding the buffer against the newest
    *ingest* keeps that dump forever, because it is the newest thing there is, and the model is
    handed a ten-minute-old page as a description of the screen in front of the user. The frame
    ring's semantics, which criterion 6 borrows, are a window on now: ``ts_ms >= at_ms -
    ASSIST_HORIZON_MS``.
    """
    horizon_ms = RING_BUFFER_S * 1000
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text")))
    state.accept_frame_text(wire_text("page_text", "DOM captured at t=1s", ts_ms=1_000))
    state.freeze()

    assert [t.text for t in texts(state, 1_000)] == ["DOM captured at t=1s"], "fresh at capture"
    assert [t.text for t in texts(state, 1_000 + horizon_ms)] == ["DOM captured at t=1s"], (
        "at exactly the horizon it is still inside the window"
    )
    assert texts(state, 1_000 + horizon_ms + 1) == (), "one ms past the horizon it is stale"
    assert texts(state, 1_000 + 10 * 60_000) == (), "ten minutes of HUD silence: nothing current"


def test_a_fresh_dump_from_the_same_kind_is_still_served_after_a_long_silence():
    """A9's other half: the horizon must not throw away text the HUD just sent.

    A cutoff applied to the wrong end (or to the whole buffer at ingest time) would make a long
    quiet session drop the dump that arrived one second before the turn.
    """
    horizon_ms = RING_BUFFER_S * 1000
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text")))
    state.accept_frame_text(wire_text("page_text", "stale dump", ts_ms=1_000))
    state.accept_frame_text(wire_text("page_text", "fresh dump", ts_ms=10 * horizon_ms))
    state.freeze()
    assert [t.text for t in texts(state, 10 * horizon_ms + 500)] == ["fresh dump"]


# --- criterion 5: the per-turn snapshot ----------------------------------------------------------


def test_a_toggle_after_freeze_does_not_change_the_turn_in_flight():
    """Criterion 5, adversarial: the user ticks every box while the answer is being spoken.

    The snapshot, the texts derived from it and the badge for that turn must all be the
    pre-toggle ones; the new flags take effect from the *next* ``freeze()``.
    """
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text")))
    state.accept_frame_text(wire_text("page_text", "dom for this turn", ts_ms=1000))
    snapshot = state.freeze()
    assert snapshot == AssistFlags(page_text=True)
    assert kinds_of(state, 5000) == ("page_text",)

    state.toggle(AstraAssistToggle(flags=flags(*KINDS)))  # mid-answer: every box ticked
    for kind in ("pdf_text", "cells", "at_spi"):
        state.accept_frame_text(wire_text(kind, f"{kind} mid-turn", ts_ms=1100))

    assert snapshot == AssistFlags(page_text=True), (
        "freeze() returns a value later toggles cannot reach"
    )
    assert snapshot.badge() == "pixels+dom"
    assert kinds_of(state, 5000) == ("page_text",), (
        "criterion 5: texts_for follows the frozen snapshot until the next freeze()"
    )

    assert state.freeze() == flags(*KINDS), "the next turn sees the new flags"
    assert sorted(kinds_of(state, 5000)) == sorted(KINDS)


def test_unticking_mid_turn_purges_the_payload_but_leaves_the_snapshot_alone():
    """A11 (ruling): criterion 4 beats criterion 5 on a mid-turn untick. **Adversarial.**

    The user unticks DOM text while the answer is being spoken. Criterion 4 is a privacy promise,
    not a scheduling hint, so the text goes at once — including from the turn in flight, whose
    ``texts_for`` must stop returning it. The frozen snapshot, in contrast, is immutable: the turn
    record has to keep saying which assists were on when the prompt was built, or the regime
    logged beside the number is not the regime the number was measured in.
    """
    state = new_state()
    state.toggle(AstraAssistToggle(flags=flags("page_text", "cells")))
    state.accept_frame_text(wire_text("page_text", "SECRET DOM", ts_ms=1000))
    state.accept_frame_text(wire_text("cells", "A1 12", ts_ms=1000))
    snapshot = state.freeze()
    assert kinds_of(state, 5000) == ("page_text", "cells"), "both are in this turn's payload"

    state.toggle(AstraAssistToggle(flags=flags("cells")))  # mid-answer: DOM text unticked

    assert snapshot == AssistFlags(page_text=True, cells=True), "the snapshot is immutable"
    assert snapshot.badge() == "pixels+dom+cells"  # hand-written from AssistFlags.badge
    got = texts(state, 5000)
    assert [t.kind for t in got] == ["cells"], "the payload is not: the untick takes effect now"
    assert "SECRET DOM" not in "".join(t.text for t in got)


# --- criterion 15 / implementation note: no clock, no asyncio, no I/O ----------------------------


def test_assist_state_has_no_clock_no_asyncio_and_no_io():
    """Implementation note + criterion 15: timestamps arrive as arguments; no socket, no writes."""
    module = assist_module()
    source = Path(inspect.getfile(module)).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = {"time", "datetime", "asyncio", "socket", "http", "httpx", "requests", "aiohttp"}
    assert not imported & forbidden, (
        f"{MODULE} must stay pure, imports {sorted(imported & forbidden)}"
    )
    assert "open(" not in source, "nothing under perception/ writes outside the data dir"
