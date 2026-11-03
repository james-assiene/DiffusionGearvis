"""Fake ``SttClient``: emits scripted events; words lag audio by ``lag_ms`` like moshi-server.

Two modes, because two kinds of test need two different things (T-0032, "Fixtures"):

**script mode** (the default, unchanged -- every existing session test depends on it)
    ``events()`` replays ``script`` once, as fast as the consumer asks for it, whether or not
    anybody ever called ``feed()``. A test that wants a word at a given moment submits it itself.

**reactive mode** (``FakeStt(..., reactive=True)``, T-0032's headline)
    the script is released against *fed audio time*: a word is delivered when
    ``fed_ms >= end_ms + lag_ms``, so **a session that never feeds audio hears nothing** -- which
    is the deafness T-0032 exists to close. In this mode the stream never ends on its own: it is
    the live wire, and ``close()`` is what stops it.

Both modes implement the ``Marker`` round trip the real client has
(``local_astra/conductor/stt_client.py``; ``tests/unit/test_stt_client.py``'s
``test_a_marker_echo_yields_flushed_after_every_word_that_preceded_it`` is the real-service half
of this behaviour): ``mark(t_eot_ms)`` makes a ``SttFlushedEvent`` arrive **after** every scripted
word with ``end_ms <= t_eot_ms`` has been emitted, never before. That ordering is the whole point
of the flush (docs/02 §4 row 6); a fake that echoed early would make the ENDPOINTING -> THINKING
row pass for the wrong reason. In script mode the replay is one-shot, so a mark asked for after
the script has run out has nothing left to order itself against and echoes at the end of it.

In **reactive mode** the echo also costs audio: the marker is released only once fed audio time
has passed ``t_eot_ms + FLUSH_BACKLOG_MS`` (T-0032 ruling A11). That is the live behaviour of the
server, not a convenience -- see the census facts below -- and it is what makes "the capture pump
must never stop" testable over the fakes: a session that paused feeding in ENDPOINTING (or in
SPEAKING, before the next turn's mark) now hangs here exactly as it would on the box, instead of
passing green and hanging on hardware. Script mode keeps the old, unconditional ordering, because
its tests feed no audio at all.

Two facts from the T-0007a live census, recorded here so nobody reads a green fast suite as a
statement about the box:

* the real server echoes a ``Marker`` **only while audio keeps arriving** (census run 3: the echo
  landed 3.0 s after the mark, on CPU STT, and never landed at all when feeding stopped). Reactive
  mode models the *dependency* (no fed audio, no echo) with a much smaller, chosen backlog --
  ``FLUSH_BACKLOG_MS`` = 500 ms of audio, not the measured 3 s -- so the fast suite stays fast; the
  delay itself is group B's measurement (criterion 13), never a number quoted from here;
* the server sends ``EndWord`` (T-0038), so until that lands a real word's ``end_ms`` equals its
  ``start_ms``. The words scripted here have honest durations, which makes ``t_eot_ms`` the true
  end of the utterance in the fast suite and a (documented) under-estimate on the box.

**A word the server never closed** (``FakeStt.open_word(...)``, or ``words(..., end_estimated=True)``,
T-0051 criteria 8/9): moshi-server sends a ``Word`` and, ~0.5 s later, the ``EndWord`` that closes
it. On 2026-09-16 one ``EndWord`` never came at all and ``MoshiSttClient`` held the word for
**21 040 ms**, publishing it at the head of the human's next sentence with an invented 21 s
duration (``logs/sessions/20260916-170934-3380576.jsonl``, seq 3844). T-0051 bounds that wait with
``STT_OPEN_WORD_HOLD_MS`` and marks such a word ``end_estimated=True``; this fake can script one so
a session test can see what the hold releases. The wire-level half of the same behaviour — a real
``Word`` with no ``EndWord`` reaching ``MoshiSttClient`` — lives in
``tests/unit/test_stt_client_open_word_hold.py`` and in
``tests/contract/test_speech_clients_contract.py``'s fake/real pair, which is the real-service row
``.claude/rules/tests.md`` owes for a behaviour added to a fake.

Steps (``steps=True``) model the delayed-stream ``Step`` frame: one every ``step_ms`` of *audio*
time whether or not anyone speaks, carrying a pause-head vector that is **high in silence and low
in speech**. The vector is uniform (every entry the same value) on purpose: ``PAUSE_HEAD_INDEX``
is documented as unverified until T-0032 criterion 12 reads the real ``prs`` vector off the box,
so this fake must not pretend to know which index carries the signal. The *length* (8) is
arbitrary for the same reason; the real length is criterion 12's evidence.

**A recogniser that hears the audio** (``FakeStt(..., reactive=True, hears_audio=True)``, T-0181 A,
opt-in, additive): by default the fake is *content-blind* -- a scripted word is released by fed
audio *time* whatever the fed samples were, which is what lets T-0181's adversarial rows check that
the session itself drops a word the muted span could not have produced. With ``hears_audio=True`` a
fed frame whose samples are all exactly zero is silence: a scripted word whose ``start_ms`` falls in
such a frame is never emitted (it is counted in ``unheard`` and still counts as resolved for a
marker's ordering), and a step whose audio position falls in one carries the silence head. The
real-service half of this behaviour is ``tests/contract/test_t0181_stt_zeros.py`` (``backend=real``,
``gpu``): after an utterance, exact zeros produce no word and a marker issued during them is echoed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence

from local_astra.contracts.speech import SttEvent, SttStepEvent, SttWordEvent

#: moshi steps every 80 ms of audio (docs/02 §2.2).
STEP_MS = 80
#: Pause-head values the fake puts on the wire. Chosen, not measured (T-0032 criterion 12 is the
#: measurement); they sit either side of ``EOT_P_THRESHOLD`` = 0.6 so an EMA has to move to cross.
SILENCE_HEAD = 0.9
SPEECH_HEAD = 0.05
#: Length of the ``prs`` vector the fake sends. Arbitrary; the real one is criterion 12's evidence.
HEAD_SLOTS = 8
#: Reactive mode only (T-0032 ruling A11): how much *fed audio* must arrive after the marker was
#: asked for -- and after ``t_eot_ms`` -- before the echo is released. Chosen, not measured: the
#: live server echoes a marker only while audio keeps arriving (T-0007a census A14, run 3 -- 3.0 s
#: on CPU STT with silence fed continuously, and never when feeding stopped), because the marker
#: is inserted at the end of the audio the server has *received* and comes back only once the
#: server has chewed past it. The property that matters for the session is that dependency, not
#: the magnitude; 500 ms is ~6 STT frames, long enough that a pump which pauses anywhere between
#: the end of turn and the echo never gets its flush, short enough to keep the fast suite fast.
FLUSH_BACKLOG_MS = 500


def require_end_estimated() -> None:
    """T-0051 criterion 8's field, or a named assertion saying which criterion owes it.

    An ``AssertionError`` rather than a pydantic ``ValidationError`` from inside a test: a row that
    dies of "extra fields not permitted" reads as a broken test, not as a missing feature.
    """
    assert "end_estimated" in SttWordEvent.model_fields, (
        "T-0051 criterion 8: local_astra/contracts/speech.py must add a serialised field "
        "`end_estimated: bool = False` to SttWordEvent - True exactly where MoshiSttClient "
        "estimates the end (closed by the next word's arrival, by the marker echo, by close(), "
        "by a disconnect, or by the new bounded hold) and False when a real EndWord closed it "
        "(docs/tasks/T-0051.md)"
    )


def word_event(
    text: str, start_ms: int, end_ms: int, *, end_estimated: bool = False
) -> SttWordEvent:
    """One ``SttWordEvent``; the new field is passed **only** when it is asked for.

    Every pre-T-0051 caller therefore builds exactly the object it always did (``extra="forbid"``
    would otherwise make this module stop working against today's contract), and a caller that
    asks for an estimated end before the field lands gets :func:`require_end_estimated`'s message.
    """
    if not end_estimated:
        return SttWordEvent(text=text, start_ms=start_ms, end_ms=end_ms)
    require_end_estimated()
    return SttWordEvent(text=text, start_ms=start_ms, end_ms=end_ms, end_estimated=True)


class FakeStt:
    def __init__(
        self,
        script: Sequence[SttEvent] = (),
        lag_ms: int = 500,
        *,
        reactive: bool = False,
        steps: bool = False,
        step_ms: int = STEP_MS,
        usable_heads: bool = True,
        hears_audio: bool = False,
    ) -> None:
        if hears_audio and not reactive:
            # script mode ignores fed audio altogether, so "hearing" it would silently do nothing
            raise ValueError("hears_audio=True needs reactive=True (T-0181 A)")
        self.script = list(script)
        self.lag_ms = lag_ms
        self.reactive = reactive
        self.steps = steps
        self.step_ms = step_ms
        #: ``False`` models the server T-0007a's census A12 warns about: a ``Step`` whose ``prs``
        #: vector carries no usable pause head at all, so the silence **backstop** is the only
        #: thing that can end a turn. That is the regime T-0051's criteria 1, 2 and 17 are about
        #: (ruling A7: on a server that does send usable heads they decide first, and a 600 ms
        #: pause is cut by the heads whatever the backstop does).
        self.usable_heads = usable_heads
        self.fed_ms = 0
        self.fed_blocks = 0
        self.closed = False
        #: T-0181 A: ``True`` -- an all-zero fed frame is silence (see the module docstring)
        self.hears_audio = hears_audio
        #: ``(start_ms, end_ms)`` of every fed all-zero frame, in feed order (``hears_audio`` only)
        self.silent_frames: list[tuple[int, int]] = []
        #: scripted words never emitted because their start fell in a silent frame (``hears_audio`` only)
        self.unheard: list[SttWordEvent] = []
        #: every ``mark`` the session asked for, in order (a test's oracle for criterion 5)
        self.marks: list[int] = []
        #: counted the way the real client counts it (``MoshiSttClient.stats``)
        self.dropped_events = 0
        self._emitted: set[int] = set()
        self._pending_marks: list[tuple[int, int]] = []
        self._next_step = 0
        self._step_origin: int | None = None
        self._marker_id = 0

    # --- the SttClient surface -----------------------------------------------------------------

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        # The server starts stepping when audio starts, so the 80 ms grid is anchored on the
        # first block fed -- not on zero, which would make an audio clock far from the session
        # clock (the offset row of ``tests/unit/test_session_listening.py``) replay its whole
        # history as steps.
        if self._step_origin is None:
            self._step_origin = t_ms
        if self.hears_audio and not any(pcm_24k_f32):
            # little-endian f32: an all-zero byte string is exactly +0.0 in every sample
            duration_ms = len(pcm_24k_f32) // 4 * 1000 // 24000
            self.silent_frames.append((int(t_ms), int(t_ms) + duration_ms))
        self.fed_ms = t_ms
        self.fed_blocks += 1

    async def mark(self, t_eot_ms: int) -> int:
        """Ask for a ``Marker`` echo; it becomes a ``SttFlushedEvent`` once the words are out."""
        self.marks.append(int(t_eot_ms))
        # The audio position at which the marker was inserted: reactive mode releases the echo
        # only once the session has fed FLUSH_BACKLOG_MS of audio *past* it (ruling A11).
        self._pending_marks.append((int(t_eot_ms), self.fed_ms))
        self._marker_id += 1
        return self._marker_id

    def stats(self) -> Mapping[str, int]:
        return {
            "blocks": self.fed_blocks,
            "dropped_blocks": 0,
            "dropped_events": self.dropped_events,
        }

    async def events(self) -> AsyncIterator[SttEvent]:
        if not self.reactive:
            async for event in self._replay():
                yield event
            return
        async for event in self._live():
            yield event

    async def close(self) -> None:
        self.closed = True

    # --- script mode ---------------------------------------------------------------------------

    async def _replay(self) -> AsyncIterator[SttEvent]:
        """The original behaviour: the script, once, in order, ignoring fed audio."""
        index = 0
        while index < len(self.script):
            await asyncio.sleep(0)
            if self.closed:
                return
            event = self.script[index]
            self._emitted.add(index)
            index += 1
            yield event
            for flush in self._ready_flushes():
                yield flush
        for flush in self._ready_flushes():
            yield flush

    # --- reactive mode -------------------------------------------------------------------------

    async def _live(self) -> AsyncIterator[SttEvent]:
        """The live wire: nothing arrives that the fed audio has not paid for yet."""
        while not self.closed:
            due = [*self._due_words(), *self._due_steps(), *self._ready_flushes()]
            if not due:
                await asyncio.sleep(0)
                continue
            for event in due:
                if self.closed:
                    return
                yield event

    def _due_words(self) -> list[SttEvent]:
        out: list[SttEvent] = []
        for index, event in enumerate(self.script):
            if index in self._emitted:
                continue
            if self._release_ms(event) <= self.fed_ms:
                self._emitted.add(index)
                if (
                    self.hears_audio
                    and isinstance(event, SttWordEvent)
                    and self._silent_at(event.start_ms)
                ):
                    self.unheard.append(event)  # resolved for a marker's ordering, never emitted
                    continue
                out.append(event)
        return out

    def _silent_at(self, t_ms: int) -> bool:
        """``hears_audio``: the fed frame holding audio time ``t_ms`` was all zeros."""
        return any(start <= t_ms < end for start, end in self.silent_frames)

    def _due_steps(self) -> list[SttEvent]:
        if not self.steps:
            return []
        origin = self._step_origin
        if origin is None:
            return []
        out: list[SttEvent] = []
        while origin + (self._next_step * self.step_ms) + self.lag_ms <= self.fed_ms:
            audio_ms = origin + self._next_step * self.step_ms
            self._next_step += 1
            head = SPEECH_HEAD if self._is_speech(audio_ms) else SILENCE_HEAD
            prs = () if not self.usable_heads else (head,) * HEAD_SLOTS
            # ``t_ms`` is the **fed-audio clock at which the step is delivered**, not the audio
            # position the model had reached — which is what ``MoshiSttClient`` stamps
            # (``SttStepEvent(..., t_ms=self._audio_clock_ms)``) and therefore what the endpointer
            # reads as its clock. The two differ by exactly ``lag_ms``, and that difference *is*
            # T-0051's defect: a fake that stamped the audio position would hand the endpointer a
            # stream in which a word's ``end_ms`` equals "now", i.e. a recogniser with no emission
            # delay at all, and no session-level row could ever see the bug that cut the human off
            # (T-0051 criteria 2 and 17, ruling A3).
            out.append(SttStepEvent(pause_heads=prs, t_ms=audio_ms + self.lag_ms))
        return out

    def _ready_flushes(self) -> list[SttEvent]:
        """A marker echoes only once every word it covers has already gone out.

        Reactive mode adds the second condition the live server has (ruling A11): the echo also
        needs ``FLUSH_BACKLOG_MS`` of *fed audio* past ``t_eot_ms`` **and** past the audio
        position at which the marker was inserted. The second half is the one with teeth: by the
        time an endpointer has waited out its silence backstop, fed audio is already well past
        ``t_eot_ms``, so a session that stops feeding the moment it marks would still be handed
        its flush. With both, a pump that pauses anywhere between the end of turn and the echo --
        in ENDPOINTING, or in SPEAKING before the next turn -- never leaves ENDPOINTING, here and
        on the box.
        """
        out: list[SttEvent] = []
        for pending in list(self._pending_marks):
            t_eot_ms, fed_at_mark = pending
            if self.reactive and self.fed_ms < max(t_eot_ms, fed_at_mark) + FLUSH_BACKLOG_MS:
                continue  # the backlog has not been paid for in audio yet
            covered = [
                index
                for index, event in enumerate(self.script)
                if isinstance(event, SttWordEvent) and event.end_ms <= t_eot_ms
            ]
            if any(index not in self._emitted for index in covered):
                continue
            self._pending_marks.remove(pending)
            out.append(self.flushed(t_eot_ms))
        return out

    def _release_ms(self, event: SttEvent) -> int:
        for field in ("end_ms", "t_eot_ms", "t_ms"):
            value = getattr(event, field, None)
            if value is not None:
                return int(value) + self.lag_ms
        return self.lag_ms

    def _is_speech(self, t_ms: int) -> bool:
        if self.hears_audio and self._silent_at(t_ms):
            return False  # zeros are silence, whatever the script says was spoken there
        return any(start <= t_ms < end for start, end in self.speech_spans())

    # --- scripting helpers ---------------------------------------------------------------------

    def say(self, text: str, start_ms: int, ms_per_word: int = 200) -> list[SttWordEvent]:
        """Append one spoken run to the script, in *audio* time. Reactive mode releases it."""
        spoken = self.words(text, start_ms, ms_per_word)
        self.script.extend(spoken)
        return spoken

    def speech_spans(self) -> list[tuple[int, int]]:
        """``(start_ms, end_ms)`` of every scripted word -- what this fake calls "speech"."""
        return [(e.start_ms, e.end_ms) for e in self.script if isinstance(e, SttWordEvent)]

    @staticmethod
    def flushed(t_eot_ms: int) -> SttEvent:
        """``SttFlushedEvent``: the ``Marker`` echo that confirms every word <= ``t_eot_ms``.

        T-0007a criterion 18 adds the member to ``contracts/speech.py``; CLAUDE.md requires the
        fake to script the new wire shape in the same change. The import is deferred and the
        absence is an *assertion* so that this module keeps importing -- and
        ``tests/contract/test_fakes_conform.py`` keeps passing -- until the contract lands.
        """
        from local_astra.contracts import speech

        cls = getattr(speech, "SttFlushedEvent", None)
        assert cls is not None, (
            "T-0007a criterion 18: contracts/speech.py must add SttFlushedEvent "
            '(type: Literal["flushed"], t_eot_ms: int >= 0) to the SttEvent union'
        )
        return cls(t_eot_ms=t_eot_ms)

    @staticmethod
    def words(
        text: str, start_ms: int, ms_per_word: int = 250, *, end_estimated: bool = False
    ) -> list[SttWordEvent]:
        out = []
        for i, w in enumerate(text.split()):
            s = start_ms + i * ms_per_word
            out.append(word_event(w, s, s + ms_per_word, end_estimated=end_estimated))
        return out

    @staticmethod
    def open_word(text: str, start_ms: int, end_ms: int) -> SttWordEvent:
        """A word moshi-server never closed, as the client publishes it once the hold expires.

        ``end_estimated=True`` is the whole point: T-0051 criterion 12 refuses an onset for such a
        word, so a session that hears one mid-reply must not start a turn on it.
        """
        return word_event(text, start_ms, end_ms, end_estimated=True)
