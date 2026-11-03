"""``astra.assist.toggle`` / ``astra.assist.state`` on the wire (T-0012 criteria 2 and 12).

The wire is *unchanged* by this task: T-0012 adds a component behind the protocol, not a message.
So half of this module pins what must not move (the message-type sets, ``AssistFlags``' field
names and defaults, the badge strings) and half checks that the echo ``AssistState`` produces
survives ``AstraMessageAdapter`` unchanged.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from local_astra.contracts.codec import decode, encode
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.protocol import (
    ASTRA_MESSAGE_TYPES,
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    AstraAssistState,
    AstraAssistToggle,
    AstraMessageAdapter,
)
from tests.unit import _t0181_pins as t0181  # T-0181 A1: the mic pair, by name (empty at BASE)

pytestmark = pytest.mark.contract

KINDS = ("page_text", "pdf_text", "cells", "at_spi")

_MISSING = object()

#: Every combination of the four flags, smallest first: 16 cases, hand-enumerable.
ALL_FLAG_SETS: tuple[AssistFlags, ...] = tuple(
    AssistFlags(**{k: bool(mask & (1 << i)) for i, k in enumerate(KINDS)}) for mask in range(16)
)


def assist_symbol(name: str) -> Any:
    try:
        module = importlib.import_module("local_astra.perception.assists")
    except ModuleNotFoundError as exc:  # pragma: no cover - the pre-implementation path
        raise AssertionError(f"T-0012 adds local_astra/perception/assists.py ({exc})") from exc
    obj = getattr(module, name, _MISSING)
    assert obj is not _MISSING, f"T-0012 adds perception.assists.{name}"
    return obj


@pytest.mark.parametrize("flags", ALL_FLAG_SETS, ids=lambda f: f.badge())
def test_toggle_and_state_round_trip_through_the_codec(flags):
    """Criterion 2: the toggle in, the echo out, both unchanged by the adapter."""
    toggle = AstraAssistToggle(flags=flags)
    assert AstraMessageAdapter.validate_python(toggle.model_dump()) == toggle
    assert decode(encode(toggle)) == toggle

    state = assist_symbol("AssistState")()
    echo = state.toggle(AstraMessageAdapter.validate_python(toggle.model_dump()))
    assert isinstance(echo, AstraAssistState)
    assert echo.flags == flags and echo.badge == flags.badge()
    assert AstraMessageAdapter.validate_python(echo.model_dump()) == echo
    assert decode(encode(echo)) == echo
    assert state.state_message() == echo


def test_the_assist_messages_keep_their_direction():
    """A toggle is client→server and the state echo is server→client; T-0012 moves neither."""
    assert "astra.assist.toggle" in CLIENT_TO_SERVER
    assert "astra.assist.state" in SERVER_TO_CLIENT
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT)


def test_the_message_type_sets_are_unchanged_by_this_task():
    """Criterion 12, pinned by hand from contracts/protocol.py as of T-0012's planning.

    A wire change would require changing the fakes in ``tests/fakes/`` in the same PR (CLAUDE.md);
    this task does neither, so any diff here means the implementer went outside its owned files.

    **T-0030 spec change, not test accommodation.** docs/02 §5's message list gains
    ``astra.spoken.played`` (T-0030 criterion 1, and the ``/spec`` delta recorded under that
    task's "Out of scope"), so this inventory grows by one server->client type and
    ``len(SERVER_TO_CLIENT)`` goes 9 -> 10. The test-author updates the pin in the same PR as the
    contract (T-0030 implementation notes: the implementer may not, it is hook-enforced); the
    ``pending`` marker comes off when group A is implemented. ``CLIENT_TO_SERVER`` is untouched —
    that is criterion 1's negative half.

    **T-0039 spec change, same rule (ruling A8).** T-0039 criterion 1 adds ``astra.text``
    (``AstraText``: the question typed into the HUD box) to ``ASTRA_MESSAGE_TYPES`` and to
    ``CLIENT_TO_SERVER`` **only** — it is the first client->server type added since T-0012, so
    that set goes from five entries to six and the inventory gains one tag.
    ``len(SERVER_TO_CLIENT) == 10`` is unchanged, which is criterion 8's negative half ("no new
    server->client type"): the answer to a typed question comes back on exactly the messages a
    spoken turn uses. Following this docstring's own rule, the pin moves in the test-author's PR,
    ahead of ``contracts/protocol.py``; until group A is implemented this row is red, by design,
    for the same reason the ``EVENT_TYPES`` pin in ``tests/unit/test_turn_invariants.py`` is.

    **T-0043 spec change, same rule (criterion 16, ruling B7).** The session-log pair is added:
    ``astra.log.toggle`` (``AstraLogToggle``) to ``CLIENT_TO_SERVER`` **only**, taking that set
    from six entries to seven, and ``astra.log.state`` (``AstraLogState``) to ``SERVER_TO_CLIENT``
    **only**, taking ``len(SERVER_TO_CLIENT)`` from 10 to 11; the inventory gains both tags. It is
    deliberately a *new pair* and not a field on ``AstraAssistToggle``: the rows after this one
    pin ``AssistFlags``, the assist toggle's own fields and every badge string as unchanged, which
    is criterion 16's "the log flag is not an assist" and stays exactly as strict as it was.
    Following this docstring's own rule the pin moves in the test-author's change, ahead of
    ``contracts/protocol.py``.

    **T-0054 spec change, same rule (criterion 8, ruling A11).** ``astra.ui.fence``
    (``AstraUiFence``: what happened to this turn's interface, and how much of the reply the
    speech gate refused) joins ``ASTRA_MESSAGE_TYPES`` and ``SERVER_TO_CLIENT`` **only**, taking
    ``len(SERVER_TO_CLIENT)`` from 11 to 12; the inventory gains the tag. ``CLIENT_TO_SERVER`` is
    untouched — the pane says what the server said and never answers it, which is criterion 8's
    "server->client only, never replayed". T-0054 named four moving pins and this is the fifth
    (ruling A11); it moves here, in the test-author's change, ahead of ``contracts/protocol.py``,
    so until group A is implemented this row is red by design, exactly as it was for T-0039 and
    T-0043 above.

    What this row still catches after the move: any *further* type — a seventeenth message, a
    ``log_dir`` message, a type moved from one direction to the other, or the pair added to
    ``ASTRA_MESSAGE_TYPES`` without being declared in a direction set — reddens it, because the
    first two assertions are set equalities transcribed by hand from docs/02 §5 and the third is
    an exact count. This is a scope fence, not a description of ``protocol.py``.

    **T-0180 A1, same rule (activation pin move).** ``astra.thought`` (``AstraThought``: the
    turn's committed thought text for the "Model thoughts" disclosure) joins
    ``ASTRA_MESSAGE_TYPES`` and ``SERVER_TO_CLIENT`` **only**, taking ``len(SERVER_TO_CLIENT)``
    from 16 to 17; ``CLIENT_TO_SERVER`` is untouched (A1 "``SERVER_TO_CLIENT`` only").

    **T-0181 A1, same rule (census pin move).** ``astra.mic.mute`` joins ``CLIENT_TO_SERVER`` only
    and ``astra.mic.state`` joins ``SERVER_TO_CLIENT`` only (17 -> 18). The pair is read through
    ``_t0181_pins.MIC_TYPES``, which is empty on a tree without ``AstraMicState`` (BASE) and exactly
    the pair on a tree with it, so the row is green on both and still reddens on any third type,
    a mic type in the wrong direction, or ``astra.mic.mute`` added without its state.

    **T-0193 A-1, same rule (census pin move).** ``astra.turn.stop`` (``AstraTurnStop``) joins ``CLIENT_TO_SERVER``
    only; ``SERVER_TO_CLIENT`` is untouched (the echo is the existing ``astra.turn.state``). Red until group A lands.

    **T-0213 A2a-1, same rule (named re-pin).** ``astra.settings.set`` (``AstraSettingsSet``) joins ``CLIENT_TO_SERVER``
    only; ``SERVER_TO_CLIENT`` is untouched (the answer is the existing ``astra.settings.state``).
    """
    assert {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES} == {
        "astra.hello",
        "astra.frame",
        "astra.frame.text",
        "astra.assist.toggle",
        "astra.ui.action",
        "astra.assist.state",
        "astra.turn.state",
        "astra.ui.surface",
        "astra.canvas.step",
        "astra.canvas.commit",
        "astra.canvas.synthetic",
        "astra.transcript",
        "astra.spoken",
        "astra.spoken.played",
        "astra.error",
        "astra.text",
        # T-0043 criterion 16: the session-log pair, one type in each direction.
        "astra.log.toggle",
        "astra.log.state",
        # T-0054 criterion 8: the fence notice, server->client only and never journalled.
        "astra.ui.fence",
        # T-0090 criterion 1: the proposed action's lifecycle, server->client only.
        "astra.tool.call",
        # T-0162 A-1 (pre-authorised pin move): the Restart pair, one type in each direction.
        "astra.session.reset",
        "astra.session.cleared",
        # T-0177 A1 (pin move A-P2): the vision selector, one client->server type and two server->client.
        "astra.vision.select",
        "astra.vision.state",
        "astra.vision.frame",
        # T-0180 A1 (activation pin move): the turn's committed thought text, server->client only.
        "astra.thought",
        # T-0193 A-1 (census pin move): the HUD's stop, client->server only.
        "astra.turn.stop",
        # T-0213 A2r-1 (R-5 named re-pin): the read-only settings state, server->client only.
        "astra.settings.state",
        # T-0213 A2a-1 (named re-pin): the Settings panel's change set, client->server only.
        "astra.settings.set",
    } | t0181.MIC_TYPES
    assert (
        set(CLIENT_TO_SERVER)
        == {
            "astra.hello",
            "astra.frame",
            "astra.frame.text",
            "astra.assist.toggle",
            "astra.ui.action",
            "astra.text",
            "astra.log.toggle",  # T-0043 criterion 16: the page's egress, never the server's
            "astra.session.reset",  # T-0162 A-1: the page's Restart
            "astra.vision.select",  # T-0177 A1: the "Looking at" combo's choice
            "astra.turn.stop",  # T-0193 A-1: the "Interrupt Gearvis" button
            "astra.settings.set",  # T-0213 A2a-1: the Settings panel's change set
        }
        | t0181.MIC_CLIENT_TO_SERVER
    )  # T-0181 A1: + astra.mic.mute (empty at BASE)
    # T-0090: + astra.tool.call; T-0162: + astra.session.cleared; T-0177 A1: + astra.vision.state/frame;
    # T-0180 A1: + astra.thought (server->client only; CLIENT_TO_SERVER above is unchanged)
    # T-0181 A1: + astra.mic.state (server->client only; 17 at BASE)
    # T-0213 A2r-1 (R-5 named re-pin): + astra.settings.state (server->client only; CLIENT_TO_SERVER unchanged)
    assert len(SERVER_TO_CLIENT) == 18 + len(t0181.MIC_SERVER_TO_CLIENT)
    assert "astra.settings.state" in SERVER_TO_CLIENT
    assert "astra.settings.set" not in SERVER_TO_CLIENT  # T-0213 A2a-1: client->server only
    assert t0181.MIC_SERVER_TO_CLIENT <= SERVER_TO_CLIENT
    assert "astra.session.cleared" in SERVER_TO_CLIENT
    assert {"astra.vision.state", "astra.vision.frame"} <= SERVER_TO_CLIENT
    assert "astra.thought" in SERVER_TO_CLIENT


def test_assist_flags_field_names_and_defaults_are_unchanged():
    """Criterion 12: the four kinds, all defaulting to off. This is the pixel-first claim itself."""
    assert {n: f.default for n, f in AssistFlags.model_fields.items()} == dict.fromkeys(
        KINDS, False
    )
    assert AssistFlags().any_on() is False
    assert AstraAssistToggle(flags=AssistFlags()).flags == AssistFlags()


@pytest.mark.parametrize(
    ("flags", "badge"),
    [
        (AssistFlags(), "pixels"),
        (AssistFlags(page_text=True), "pixels+dom"),
        (AssistFlags(pdf_text=True), "pixels+pdf"),
        (AssistFlags(cells=True), "pixels+cells"),
        (AssistFlags(at_spi=True), "pixels+a11y"),
        (AssistFlags(page_text=True, at_spi=True), "pixels+dom+a11y"),
        (
            AssistFlags(page_text=True, pdf_text=True, cells=True, at_spi=True),
            "pixels+dom+pdf+cells+a11y",
        ),
    ],
)
def test_badge_strings_are_unchanged(flags, badge):
    """Criterion 12: the badge is what ``turns.jsonl`` and the HUD both show; it is a wire string."""
    assert flags.badge() == badge
