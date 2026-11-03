"""Frozen copy of BASE's ``local_astra/reflex/turn_predictor.py``, for T-0195 group A (A2's differential).

Source: commit 341215b7e08e2d895bb6180b2ca8eaee4fa66950 (main when T-0195 A branched, "T-0195 planned: a one-word
utterance opens a turn; mic/STT diagnostics"), blob a337d6469bbd36139c45210ba5d8b14701b223de (``git rev-parse
341215b7:local_astra/reflex/turn_predictor.py``), whole-file sha256
1f863a87e03bf1c762513fe2b9e2724f33153152c7ad1d387e8900b44d452071. Copied verbatim below the marker line, from the
source's line 14 on (its module docstring, lines 1-12, and the blank line 13 are dropped because a second docstring
would precede ``from __future__``); the copied body hashes to ``T0195_FROZEN_BODY_SHA256`` in
``tests/unit/test_t0195_estimated_onset.py``. Never edit it. ``BASE_SHA`` at the end names the commit.
``test_t0195_estimated_onset.py`` drives this ``Endpointer`` and the live one on the same event streams (A2). Import it
on use, never at module level.
"""

# ruff: noqa
# --- verbatim copy of 341215b7 from its line 14 on (its module docstring, lines 1-13, dropped) ---
from __future__ import annotations

from typing import Literal, TypeAlias

from local_astra.contracts.speech import SttEvent, SttStepEvent, SttWordEvent

#: Probability of "the user has stopped" above which the run ends (docs/02 §2.2, inherited from
#: Unmute). **Chosen, not measured here**: X2/Q5 (docs/02 §10, docs/03) is the experiment that
#: replaces it with an end-of-turn-delay / false-cut pair.
EOT_P_THRESHOLD = 0.6

#: Half-life of the pause-head EMA, in *steps* (one step = 80 ms of audio), stated as a half-life
#: rather than a bare α so the number carries its regime: 3 steps ≈ 240 ms of audio. One spurious
#: high head between low ones therefore cannot cross ``EOT_P_THRESHOLD`` on its own; a sustained
#: one crosses within four steps. Chosen, not measured (X2/Q5).
EMA_HALF_LIFE_STEPS = 3.0

#: The smoothing factor the half-life implies. Derived, never tuned independently of it.
EMA_ALPHA = 1.0 - 0.5 ** (1.0 / EMA_HALF_LIFE_STEPS)

#: Backstop: this much *step-clock* (audio) time after the last word, with no new word, ends the
#: run whatever the heads say — the server may send no usable head at all (T-0007a census A12).
#: Chosen, ≈ the end-of-turn column of the B3 budget (docs/03 §7); ruling A4 keeps it ≥ 400 ms so
#: the EMA has at least four steps of slack to tell a spike from a run of high heads.
EOT_SILENCE_MS = 800

#: What a *backstop* decision reports as ``p_end`` (ruling A7). The machine drops
#: ``stt.end_of_turn`` below ``TurnPolicy.eot_p_threshold``; on the backstop the EMA is by
#: construction not high (an unusable ``prs`` vector never moves it), so reporting it would make
#: the backstop inert at the seam. 1.0 is ``SttEndOfTurnEvent``'s own default, and it says what
#: is true: this decision was not taken on the heads, and the machine's threshold is a second
#: gate on the head path only.
BACKSTOP_P_END = 1.0

#: A run shorter than this (first word's ``start_ms`` → last word's ``end_ms``) never ends a turn:
#: a cough recognised as one 40 ms word is not a question — on the **head** path only
#: (ruling A8; the silence backstop is unconditional). Chosen, not measured.
MIN_SPEECH_MS = 200

#: Which entry of ``SttStepEvent.pause_heads`` carries the pause probability. **Unverified**: the
#: live server sends a 4-vector (T-0007a census A12, e.g. ``[1.3e-05, 1.4e-04, 2.8e-04, 5.8e-04]``
#: at speech) and T-0032 criterion 12 reads it at *silence* to choose the index from evidence.
#: Until then this is a guess, and a vector too short for it is treated as no signal at all —
#: never as a 0.0 that would mean "still speaking" for ever.
PAUSE_HEAD_INDEX = 2

#: The longest span a single word may claim before the endpointer refuses to open a run on it
#: (T-0051 criterion 5). A word is not always timed: when moshi-server sends a ``Word`` whose
#: ``EndWord`` never arrives, ``MoshiSttClient`` closes it at the *next* word's start, and on
#: 2026-09-16 that fabricated a word 21 040 ms long (``start_ms`` 247200 → ``end_ms`` 268240, log
#: seq 3844) which opened a turn 800 ms into the assistant's own reply. Band, from sources outside
#: this module: ≥ 1000 ms because a drawn-out real word reaches about a second and a guard below
#: that would discard speech; ≤ 5000 ms because it must refuse that 21 s fabrication with a wide
#: margin. 2000 ms is the value T-0051 recommends inside that band. The guard refuses the *word*
#: — such a span neither opens a run nor extends one, since freezing audio up to an invented end
#: is worse than ignoring it — and never closes a run already open on honest words.
MAX_WORD_MS = 2000

#: ``("onset", audio_ms)`` or ``("end_of_turn", t_eot_ms, p_end)``; both times are *audio* times
#: taken from the STT stream, never the session clock's (docs/02 §4).
Onset: TypeAlias = tuple[Literal["onset"], int]
EndOfTurn: TypeAlias = tuple[Literal["end_of_turn"], int, float]
Decision: TypeAlias = Onset | EndOfTurn


class TurnPredictor:
    def p_end(self, pcm_16k_f32: bytes, pause_heads: tuple[float, ...] | None) -> float:
        raise NotImplementedError("T-0007")


class Endpointer:
    """Kyutai pause heads + a silence backstop → onset and end-of-turn, in audio time.

    One per session; ``observe(event)`` is called with every ``SttEvent`` the client yields, in
    wire order, and returns at most one decision per event. The rules (T-0032 criterion 3):

    (a) the first word of a run is the onset, at that word's ``start_ms``;
    (b) an EMA of the pause head crossing ``EOT_P_THRESHOLD`` ends the run at the last word's
        ``end_ms`` — not at the step that noticed, which is later by construction;
    (c) ``EOT_SILENCE_MS`` of step-clock time after the last word **arrived** ends it too, so a
        server that sends no usable head still endpoints (T-0051 criterion 2; see
        :meth:`_arrival_of`);
    (d) one end of turn per run: the next needs an intervening onset;
    (e) a run with no words never ends a turn, however long the silence;
    (f) a word whose ``start_ms`` is at or before the last decided ``t_eot`` opens nothing and
        changes nothing: it is the tail of a sentence the machine has already frozen
        (T-0051 criterion 4);
    (g) a word claiming more than ``MAX_WORD_MS`` of audio opens nothing either: it was not timed
        by the recogniser (T-0051 criterion 5);
    (h) nor does a word whose ``end_estimated`` is set: nobody measured its end, so it says
        nothing about when the speaker stopped — and the bounded hold that releases it
        (``STT_OPEN_WORD_HOLD_MS``) can do so *during the assistant's reply*, where an onset is a
        barge-in through the STT tap (T-0051 criterion 12).

    The EMA is reset at each onset (ruling A9). Not because of the silence before the run —
    steps are ignored while no run is open, so those heads never enter the EMA at all — but
    because the value carried over from the *previous* run is precisely the one that crossed
    ``EOT_P_THRESHOLD``: without the reset the next utterance starts one step away from ending,
    and a single moderate head cuts the user off on their second sentence.
    """

    def __init__(self) -> None:
        self._open = False
        self._first_word_ms: int | None = None
        self._last_word_ms: int | None = None
        #: Step-clock time at which the last word of the open run was delivered (rule c).
        self._last_arrival_ms: int | None = None
        #: The most recent step's ``t_ms`` — this module's only clock.
        self._step_ms: int | None = None
        #: ``t_eot_ms`` of the last end of turn decided, or ``None`` before the first (rule f).
        self._last_t_eot_ms: int | None = None
        self._ema = 0.0

    def observe(self, event: SttEvent) -> Decision | None:
        if isinstance(event, SttWordEvent):
            return self._on_word(event)
        if isinstance(event, SttStepEvent):
            return self._on_step(event)
        return None

    # --- rules ---------------------------------------------------------------------------------

    def _on_word(self, event: SttWordEvent) -> Decision | None:
        """A word: refuse it outright (rules f, g, h) or record when the recogniser delivered it.

        The three refusals are one behaviour — *this word is not evidence about now* — and all
        come from the desk session of 2026-09-16 (T-0051 criteria 4, 5 and 12). A word at or before
        the decided ``t_eot`` is the tail of the frozen turn arriving late: read as an onset it
        re-opened the run twice in that session (seq 3093 → 3113, 3889 → 3909), and each
        re-opening cost a second end of turn, a second ``mark()`` and a third flush. A word
        claiming more than ``MAX_WORD_MS`` was never timed: it is a held word closed by the next
        one. A word whose ``end_estimated`` is set is that same word seen one layer earlier — the
        client had to supply its end, and ``MoshiSttClient`` now says so instead of leaving it to
        be inferred from an implausible duration; read as an onset during the assistant's reply it
        would take the machine SPEAKING → PAUSED through the STT tap, which T-0032's design ruling
        and T-0034 forbid. Refused means refused entirely — no onset, no extension of an open run,
        no arrival — so a fabricated span can neither open a turn nor hold one open. What it does
        **not** mean is suppressed: the word is still published by the client, stays in the
        transcript and in the log, and belongs to whatever turn its audio falls in. This rule
        drops the *decision*, never the word (T-0051 criterion 12's negative half).

        Its **negative** is as load-bearing as the guard (T-0051 criterion 5, the twin row): a
        normal word after thirty seconds of thought is the next question, and a guard that
        swallowed it would make the endpointer deaf after every pause — worse than the defect.
        """
        if not self._admits(event):
            return None
        arrival = self._arrival_of(event)
        if self._open:
            last = self._last_word_ms
            self._last_word_ms = event.end_ms if last is None else max(last, event.end_ms)
            previous = self._last_arrival_ms
            self._last_arrival_ms = arrival if previous is None else max(previous, arrival)
            return None
        self._open = True
        self._first_word_ms = event.start_ms
        self._last_word_ms = event.end_ms
        self._last_arrival_ms = arrival
        self._ema = 0.0
        return ("onset", event.start_ms)

    def _admits(self, event: SttWordEvent) -> bool:
        """Rules (f), (g) and (h): may this word be believed as evidence of speech happening now?"""
        if event.end_estimated:
            return False
        if event.end_ms - event.start_ms > MAX_WORD_MS:
            return False
        return self._last_t_eot_ms is None or event.start_ms > self._last_t_eot_ms

    def _arrival_of(self, event: SttWordEvent) -> int:
        """The step-clock time at which this word was delivered — the backstop's zero.

        The recogniser runs behind: when a word arrives at step-clock time ``a`` it has examined
        audio up to ``a - L`` and the word's ``end_ms`` **is** ``a - L``, whatever ``L`` is. So
        the audio it has examined after that word at a later step ``t`` is exactly ``t - a``, and
        ``t - end_ms`` — what this module subtracted before T-0051 — is that plus ``L``: a
        subtraction of two different clocks (the fed-audio clock and the time the word was
        spoken), always short by the emission delay. On 2026-09-16 that delay was ≈ 510 ms on the
        GPU, so ``EOT_SILENCE_MS`` of 800 ended the turn on ≈ 290 ms of examined silence and cut
        the human off mid-question; on the CPU fallback the delay is seconds. Taking the arrival
        needs no estimate of ``L`` and degrades correctly when it is large.

        ``max`` with the word's own ``end_ms`` is the refinement of ruling A8: if the step stream
        has a hole, the last step's ``t_ms`` is stale and a word delivered against it would look
        as though it arrived long ago, firing the backstop after the *hole* rather than after the
        word. A no-op on any gapless stream — the server emits a step every 80 ms of audio
        (docs/02 §2.2) — and the guard that T-0048's feed gate, which makes such holes on
        purpose, inherits.
        """
        step_ms = self._step_ms
        return event.end_ms if step_ms is None else max(step_ms, event.end_ms)

    def _on_step(self, event: SttStepEvent) -> Decision | None:
        """One step: update the EMA, then ask whether the run has ended (rules b, c, d, e).

        ``MIN_SPEECH_MS`` gates the **head** path only (ruling A8): there a burst of high heads
        can cross within four steps of a word arriving, so a cough recognised as one 40 ms word
        would open and close a turn. The silence backstop is unconditional — criterion 3(c) as
        written — because a one-word answer ("yes.") is a real turn, and gating it there would
        leave the machine in LISTENING until the user said something longer. Until T-0038's end
        times flow, every word's ``end_ms`` equals its ``start_ms``, so the run length measured
        here is between word *starts* and under-reports the utterance by one word.

        ``p_end`` is the decision's own verdict, and the two paths do not report the same thing
        (ruling A7): the head path reports the EMA that crossed, which the machine's
        ``eot_p_threshold`` then re-checks as a second gate; the backstop reports **1.0**,
        because the EMA is by construction *not* high there (an unusable ``prs`` vector never
        moves it at all) and a decision carrying 0.0 would be dropped by that gate — no
        ``FreezeAfter``, no ``mark()``, LISTENING for ever. If both hold on the same step the
        EMA is reported: the head crossed, and its value is the more informative of the two.
        """
        self._step_ms = event.t_ms  # the module's only clock, and a word's arrival stamp
        first, last = self._first_word_ms, self._last_word_ms
        arrival = self._last_arrival_ms
        if not self._open or first is None or last is None or arrival is None:
            return None  # rule (e): no words, no end of turn
        head = self._head(event.pause_heads)
        if head is not None:
            self._ema += EMA_ALPHA * (head - self._ema)
        crossed = (
            head is not None and last - first >= MIN_SPEECH_MS and self._ema >= EOT_P_THRESHOLD
        )
        silent = event.t_ms - arrival >= EOT_SILENCE_MS
        if not (crossed or silent):
            return None
        self._open = False  # rule (d): the next end of turn needs an intervening onset
        self._last_t_eot_ms = last  # rule (f): words from before this are of the frozen turn
        return ("end_of_turn", last, self._ema if crossed else BACKSTOP_P_END)

    @staticmethod
    def _head(pause_heads: tuple[float, ...]) -> float | None:
        """The pause probability, or ``None`` when the vector does not carry one (criterion 3c).

        ``prs=[]``, or a vector shorter than ``PAUSE_HEAD_INDEX``, is a *missing* signal: the EMA
        is left where it was and the backstop decides. Reading it as 0.0 would mean "still
        speaking" for ever; raising would take the reflex lane down mid-utterance.
        """
        if len(pause_heads) <= PAUSE_HEAD_INDEX:
            return None
        return float(pause_heads[PAUSE_HEAD_INDEX])


# --- end of verbatim copy ---

BASE_SHA = "341215b7e08e2d895bb6180b2ca8eaee4fa66950"
