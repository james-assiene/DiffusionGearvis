"""Filler policy: pure decision table (docs/03 §3 "Filler policy", docs/11 §1.1). Task T-0013.

``decide(inputs) -> "none" | "clip:ack" | "clip:hmm" | "clip:still_working" | "generate_short"``

Rules the tests pin: never two fillers in a row (``last_filler_age_ms`` below the minimum gap
→ ``none``); never a filler once the first canvas has committed; thinking-mode turns always get
the acknowledgement at ``llm.start``; the interim spoken update fires once at
``interim_update_ms`` (docs/11 tier B) and never for single-canvas turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["none", "clip:ack", "clip:hmm", "clip:still_working", "generate_short"]


@dataclass(frozen=True)
class FillerInputs:
    state: str  # TurnState name
    elapsed_ms: int  # since llm.start
    predicted_first_canvas_ms: int  # EMA
    thinking_enabled: bool
    last_filler_age_ms: int | None  # None: no filler yet this turn
    first_canvas_committed: bool
    fillers_this_turn: int


@dataclass(frozen=True)
class FillerPolicyConfig:
    ack_threshold_ms: int = 300
    min_gap_ms: int = 2_500
    interim_update_ms: int = 4_000
    max_fillers_per_turn: int = 2


def decide(inputs: FillerInputs, cfg: FillerPolicyConfig | None = None) -> Decision:
    """Decide which filler (if any) the reflex lane should play *now*.

    Pure: the answer depends only on ``inputs`` and ``cfg``. No clock, no I/O, no module state —
    the caller owns the bookkeeping (how many fillers it has emitted this turn, and how long ago
    the last one was *emitted*, which is what ``last_filler_age_ms`` measures, ruling A1).

    The table, in order (docs/tasks/T-0013.md criteria 1–5 and rulings A1–A5):

    1. **Only while ``THINKING``** (A3): both fillers live inside the think (docs/02 §4 row 6,
       docs/11 §1.1), so every other state is ``none`` whatever the rest of the inputs say.
    2. **Criterion 3's three guards**, in this order and before anything can fire: a committed
       first canvas silences the lane; the per-turn cap ``max_fillers_per_turn``; the minimum gap
       ``min_gap_ms`` since the previous *emission* (inclusive, A2: an age of exactly ``min_gap_ms``
       is not "within" the gap).
    3. **The acknowledgement** (criteria 1–2), which is the *first* filler of the turn: with
       thinking on it always plays, whatever the EMA predicts (criterion 1, docs/01 D11); with
       thinking off it plays iff ``predicted_first_canvas_ms > ack_threshold_ms``, strictly
       (criterion 2, A2 — ``== ack_threshold_ms`` → ``none``, as in docs/02 §4 row 6).
    4. **The interim spoken update** (criterion 4, docs/11 §1.1 tier B): ``clip:still_working``
       at ``elapsed_ms >= interim_update_ms`` (inclusive, A2), *once*. "Once" is derived, not
       flagged (A1): the update has already gone out iff the previous filler was emitted at
       ``elapsed_ms - last_filler_age_ms >= interim_update_ms``.
    5. Anything the criteria do not license — a second acknowledgement mid-think, a second interim
       update — is ``none``. ``clip:hmm`` is not produced in v1 (A4; it stays in ``Decision`` for
       the later task that adds its rule) and ``generate_short`` is never produced in v1
       (criterion 5: reserved for the talker profile, docs/11 §3).

    A late first tick (A5): if the turn's first call arrives with ``elapsed_ms > 0`` and no filler
    emitted yet, **this implementation still acknowledges** — the acknowledgement is gated on
    "no filler has gone out this turn", not on ``elapsed_ms == 0``, so a tick that arrives late
    gets the ack it would have got at ``llm.start``. The interim update then follows only if that
    ack preceded ``interim_update_ms`` — a first tick at or after 4 s acks and never sends
    ``clip:still_working``, because rule 4 reads the ack's emission time as the interim's.
    (Unspecified and untested; stated here as A5 requires.)
    """
    cfg = FillerPolicyConfig() if cfg is None else cfg

    if inputs.state != "THINKING":  # A3: fillers belong to the think
        return "none"

    # criterion 3: the three guards
    if inputs.first_canvas_committed:
        return "none"
    if inputs.fillers_this_turn >= cfg.max_fillers_per_turn:
        return "none"
    age_ms = inputs.last_filler_age_ms
    if age_ms is not None and age_ms < cfg.min_gap_ms:
        return "none"

    # criteria 1–2: the acknowledgement, i.e. the first filler of the turn
    no_filler_yet = age_ms is None and inputs.fillers_this_turn == 0
    if no_filler_yet and (
        inputs.thinking_enabled or inputs.predicted_first_canvas_ms > cfg.ack_threshold_ms
    ):
        return "clip:ack"

    # criterion 4: the interim update, once (A1: "already sent" read off the previous emission)
    if inputs.elapsed_ms >= cfg.interim_update_ms:
        interim_already_sent = (
            age_ms is not None and inputs.elapsed_ms - age_ms >= cfg.interim_update_ms
        )
        if not interim_already_sent:
            return "clip:still_working"

    return "none"
