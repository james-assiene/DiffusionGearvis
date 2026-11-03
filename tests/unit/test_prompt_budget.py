"""Prompt-budget enforcement against the *engine's* token count (T-0020, criteria 1-3).

What this module pins
=====================
``docs/tasks/T-0020.md`` names the behaviour but not the seam, so — as ``test_session_loop.py``
does for T-0007 — the names below are the test-author's, fixed here so the session tests and the
implementation agree. Report anything wrong to the reviewer; the implementer never edits tests.

``local_astra/conductor/prompt_builder.py``

    @dataclass(frozen=True)
    class FitResult:
        prompt: Prompt              # what to send: the *full* prompt (prefix + query suffix)
        tokens: int                 # ``measure(prompt)`` — the engine's count, never a guess
        fits: bool                  # tokens <= budget.max_prompt_tokens
        trimmed: Mapping[str, int]  # {"history_turns": n, "images": n, "memory_hits": n}

    PromptBuilder.fit(*, history, frame, assists, transcript, tutor=None, memory_hits=(),
                      frame_text=(), resolve=None, measure, drop_history_turns=0) -> FitResult

    ``measure: Callable[[Prompt], int]`` is the **only** source of a token count — the engine's
    own (``usage.prompt_tokens``), or a tokenizer a contract test proves agrees with it
    (criterion 3). The builder never counts characters (B10). ``drop_history_turns`` is the
    session's "trim one more history turn and retry" lever after a 400/413 (criterion 4).

How the expectations are computed
=================================
Not by re-running the trimmer. Each case builds the prompt the spec says must come out, through
the *ordinary* ``prefix()``/``full()`` API on the reduced inputs, measures it with the fake
engine, and sets the budget between that number and the number of the prompt that keeps one
element more. Every test states both bounds, so "this budget forces exactly this cut" is visible
rather than asserted. The fake engine's tokenizer (``FakeEngineTokenizer``) is a stand-in for
vLLM's count: 4 tokens per message, 2 per whitespace word, 1120 per image (the maximum visual
budget of docs/12 §1). It is deliberately *not* proportional to characters, so a chars/4
estimator cannot pass these tests by accident.

Trimming order is **T-0020 ruling A1**, which resolves B2 (``docs/tasks/T-0005.md``) into a
cost-ordered ladder — B2's literal sequencing left its step (2) unreachable, because dropping
whole turns first leaves no history image to age. The rungs, each applied only while the measured
count is over budget:

1. history *pixels*, oldest first, each falling back to its caption line (B3: a frame degrades,
   it never vanishes) — the turn itself stays;
2. memory hits, lowest-ranked first;
3. whole history turns, oldest first.

Never the system message, the current user turn or the current frame. One test per rung, plus the
floor. A history frame degrades exactly as an unresolved one does (B9), so each rung's surviving
prompt is built through ``prefix()`` with a ``resolve`` that answers for the turns whose pixels
have not been spent yet — the ordinary API, not the trimmer's own answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.conductor import prompt_builder as pb_module
from local_astra.conductor.prompt_builder import Budget, PromptBuilder
from local_astra.config import Profile
from local_astra.contracts.llm import ImagePart, Prompt, TextPart
from local_astra.contracts.memory import Episode, MemoryHit
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.contracts.turn import HistoryTurn
from tests.unit.test_prompt_builder import _max_model_len

REPO = Path(__file__).resolve().parents[2]

SYSTEM = "You are Astra."
TRANSCRIPT = "and what does the second column add up to"


# --- the fake engine's count ---------------------------------------------------------------


class FakeEngineTokenizer:
    """Stand-in for ``usage.prompt_tokens``: structural, not character-proportional.

    4 tokens per message (the chat template's role turn), 2 per whitespace word, 1120 per image
    (docs/12 §1: the maximum visual-token budget per image). Records every prompt it was asked
    about, so a test can prove the builder measured rather than guessed.
    """

    PER_MESSAGE = 4
    PER_WORD = 2
    PER_IMAGE = 1120

    def __init__(self) -> None:
        self.calls: list[Prompt] = []

    def __call__(self, prompt: Prompt) -> int:
        self.calls.append(prompt)
        return self.count(prompt)

    @classmethod
    def count(cls, prompt: Prompt) -> int:
        total = 0
        for message in prompt.messages:
            total += cls.PER_MESSAGE
            for part in message.content:
                if isinstance(part, TextPart):
                    total += cls.PER_WORD * len(part.text.split())
                elif isinstance(part, ImagePart):
                    total += cls.PER_IMAGE
        return total


# --- fixtures built by hand ------------------------------------------------------------------


def frame(tag: str, ts_ms: int = 0) -> Frame:
    return Frame(
        source=Source.SCREEN,
        ts_ms=ts_ms,
        width=1280,
        height=720,
        jpeg=b"\xff\xd8" + tag.encode() * 8,
        phash=tag * 16,
        title=f"window {tag}",
        app="firefox",
    )


def history(n: int = 3) -> tuple[HistoryTurn, ...]:
    return tuple(
        HistoryTurn(
            user_text=f"h{i} what is this window about",
            assistant_spoken=f"a{i} it is a spreadsheet of quarterly costs",
            frame_phash=f"{i:x}" * 16,
            frame_caption=f"cap{i} a spreadsheet with twelve rows",
        )
        for i in range(n)
    )


def resolver(turns: Sequence[HistoryTurn]):
    pixels = {t.frame_phash: b"\xff\xd8" + (t.frame_phash or "x")[:1].encode() * 8 for t in turns}
    return lambda phash: pixels.get(phash)


def hits(*scores: float) -> tuple[MemoryHit, ...]:
    return tuple(
        MemoryHit(
            episode=Episode(
                ts_ms=1000 + i,
                source=Source.SCREEN,
                thumbnail_path=f"/tmp/{i}.jpg",
                phash=f"m{i}" * 8,
                window_title=f"memory window {i}",
                caption=f"episode {i} the invoice tab with the totals column",
            ),
            score=score,
            matched=("column",),
        )
        for i, score in enumerate(scores)
    )


def expected_prompt(
    builder: PromptBuilder,
    turns: Sequence[HistoryTurn],
    current: Frame,
    memory: Sequence[MemoryHit],
    *,
    resolve: Any = None,
) -> Prompt:
    """The prompt the spec says must come out, built through the ordinary API on the survivors."""
    prefix = builder.prefix(turns, current, AssistFlags(), resolve)
    return builder.full(prefix, TRANSCRIPT, None, memory)


def fit(builder: PromptBuilder, criterion: str, **kwargs: Any) -> Any:
    """Call the seam, reporting its absence or a stale signature as the criterion's failure."""
    method = getattr(builder, "fit", None)
    if method is None:
        raise AssertionError(
            f"T-0020 criterion {criterion}: PromptBuilder has no fit(); the declared budget is "
            "still enforced by nothing (docs/tasks/T-0020.md)"
        )
    try:
        return method(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0020 criterion {criterion}: PromptBuilder.fit does not accept the specified "
            f"signature ({sorted(kwargs)}): {exc}"
        ) from None


def trimmed(result: Any) -> Mapping[str, int]:
    counts = getattr(result, "trimmed", None)
    assert isinstance(counts, Mapping), (
        f"T-0020 criterion 5: FitResult.trimmed is a mapping of what was dropped; got {counts!r}"
    )
    for key in ("history_turns", "images", "memory_hits"):
        assert key in counts, f"FitResult.trimmed must report {key!r}; got {dict(counts)}"
    return counts


def texts(prompt: Prompt) -> str:
    return " ".join(m.text() for m in prompt.messages)


def images(prompt: Prompt) -> list[str | None]:
    return [p.phash for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]


def floor_is_intact(prompt: Prompt, current: Frame) -> None:
    """B2's floor: the system message, the current user turn and the current frame always stay."""
    body = texts(prompt)
    assert pb_module.CLOSE_THE_THOUGHT in prompt.messages[0].text(), (
        "B2: the system message is never dropped"
    )
    assert TRANSCRIPT in body, (
        "B2/criterion 4: the user's own turn is never truncated to make the prompt fit"
    )
    assert current.phash in images(prompt), (
        "B2: the current frame is never dropped — the failure this ordering avoids is the "
        "assistant going blind mid-turn to save tokens"
    )


# --- criterion 1 -----------------------------------------------------------------------------


def test_the_default_budget_is_strictly_inside_the_serving_window():
    """Criterion 1, in the task's own words ("strictly less than").

    The companion assertion in ``test_prompt_builder.py`` (``<=``) is the strict-xfail this task
    inherits; this one is the criterion's wording plus ruling A5's headroom: a prompt exactly the
    size of the window has no space left to decode into.
    """
    budget, window = Budget().max_prompt_tokens, _max_model_len()
    assert budget < window, (
        f"criterion 1: the declared budget ({budget}) must be strictly below --max-model-len "
        f"({window}) in services/vllm.sh"
    )
    #: Ruling A5: headroom for the answer is part of the bound. The profile declares no
    #: answer-token budget today (``LlmOptions.max_tokens`` is unset on the turn path), so the
    #: smallest honest unit of generated output is one canvas — if the implementer adds a real
    #: answer budget, this reads it instead of the canvas.
    profile = Profile(name="lean")
    headroom = getattr(profile.diffusion, "max_answer_tokens", profile.diffusion.canvas_length)
    assert budget + headroom < window, (
        f"ruling A5: budget ({budget}) plus room for the answer ({headroom}) must stay strictly "
        f"below the window ({window}); a prompt that fills the window has nothing to decode into"
    )


# --- criterion 3: the count is the engine's ---------------------------------------------------

#: Anything that would make a *second* source of truth for the token count (B5/B12).
BANNED = ("tokenizer", "tiktoken", "sentencepiece", "AutoTokenizer", "chars_per_token")


def _code(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def test_the_count_comes_from_the_engine_and_nothing_else():
    """Criterion 3, both halves.

    (a) Static: neither owned module grows a second tokenizer or a characters-per-token constant.
    (b) Behavioural: the *injected* count decides everything. A prompt that is enormous in
        characters but which the engine calls small is left whole; the same prompt with an engine
        that calls it large is trimmed. A chars/4 estimator gets (b) backwards.
    """
    for name in ("prompt_builder.py", "session.py"):
        source = _code(REPO / "local_astra" / "conductor" / name)
        for banned in BANNED:
            assert banned not in source, (
                f"criterion 3: {name} must not carry a second token count ({banned!r}); the "
                "number is the engine's usage.prompt_tokens or nothing"
            )
        assert not re.search(r"/\s*/?\s*4\b", source), (
            f"criterion 3: {name} contains a division by 4 — the 4-chars-per-token guess B10 "
            "deleted must not come back"
        )

    builder = PromptBuilder(SYSTEM)
    turns = history(3)
    current = frame("z")
    memory = hits(0.9, 0.5)
    wordy = tuple(
        t.model_copy(update={"user_text": t.user_text + " lorem" * 50_000}) for t in turns
    )

    kept = fit(
        builder,
        "3",
        history=wordy,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(wordy),
        measure=lambda prompt: 10,
    )
    assert all(f"h{i} " in texts(kept.prompt) for i in range(3)), (
        "criterion 3: a prompt the engine calls small is sent whole, however many characters it "
        f"has; got history {[m.role for m in kept.prompt.messages]}"
    )
    assert trimmed(kept) == {"history_turns": 0, "images": 0, "memory_hits": 0}
    assert kept.fits is True and kept.tokens == 10, (
        f"FitResult carries the engine's number, not a recount; got {kept.tokens}"
    )

    strict = FakeEngineTokenizer()
    cut = fit(
        builder,
        "3",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=strict,
    )
    assert strict.calls, "criterion 3: fit() must ask the engine's counter at least once"
    assert cut.tokens == FakeEngineTokenizer.count(cut.prompt), (
        "FitResult.tokens is the measured count of the prompt it returns"
    )


# --- criterion 2: the trimming order ----------------------------------------------------------


def test_the_oldest_history_pixels_go_first():
    """Criterion 2, rung (1) of ruling A1: pixels before anything else, oldest first.

    Three history turns; the oldest is already past ``image_age_turns`` and travels as its caption,
    so the oldest *pixels* on the wire are the middle turn's. The budget sits between the untrimmed
    prompt and the prompt whose middle frame has degraded — 1120 tokens of image against a caption
    line. What must **not** happen is the turn disappearing: its text is the cheap part and the
    user asked about it two turns ago (B3 — a frame degrades, it never vanishes).
    """
    builder = PromptBuilder(SYSTEM)
    turns, current, memory = history(3), frame("z"), hits(0.9, 0.4)
    whole = expected_prompt(builder, turns, current, memory, resolve=resolver(turns))
    # only the newest turn's pixels are still resolvable: the middle turn's have been spent
    expected = expected_prompt(builder, turns, current, memory, resolve=resolver(turns[2:]))
    budget = FakeEngineTokenizer.count(expected)
    assert FakeEngineTokenizer.count(whole) > budget, (
        "regime: degrading the oldest pixels must actually be needed "
        f"({FakeEngineTokenizer.count(whole)} vs {budget})"
    )

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )
    assert result.fits is True and result.tokens <= budget
    assert result.prompt == expected, (
        "rung (1): the surviving prompt is the one the ordinary API builds when the middle turn's "
        "pixels are no longer available — caption line in, image out"
    )
    assert images(result.prompt) == [turns[2].frame_phash, current.phash], (
        "the *oldest* pixels go first and the newest history frame stays; got "
        f"{images(result.prompt)}"
    )
    assert "h1 " in texts(result.prompt) and "cap1" in texts(result.prompt), (
        "B3: the turn keeps its text and its caption — only the pixels were spent"
    )
    assert "episode 0" in texts(result.prompt), "memory hits go after pixels, not before (A1)"
    assert trimmed(result) == {"history_turns": 0, "images": 1, "memory_hits": 0}
    floor_is_intact(result.prompt, current)


def test_the_memory_block_goes_after_the_pixels_and_before_the_history_text():
    """Criterion 2, rung (2) of ruling A1, and its two boundaries.

    The budget is set one hit above the all-captions prompt: every history frame has already
    degraded, so the next thing to go is the *lowest-ranked* memory hit — not the higher-ranked
    one, and not a history turn, whose text is what the user is following up on.
    """
    builder = PromptBuilder(SYSTEM)
    turns, current, memory = history(3), frame("z"), hits(0.9, 0.4)
    captioned = expected_prompt(builder, turns, current, memory)  # no resolver: pixels all spent
    expected = expected_prompt(builder, turns, current, memory[:1])
    budget = FakeEngineTokenizer.count(expected)
    assert FakeEngineTokenizer.count(captioned) > budget, (
        "regime: dropping one memory hit must actually be needed "
        f"({FakeEngineTokenizer.count(captioned)} vs {budget})"
    )

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )
    assert result.fits is True
    assert result.prompt == expected
    assert images(result.prompt) == [current.phash], (
        f"every history frame pays before the memory block does; got {images(result.prompt)}"
    )
    assert all(f"h{i} " in texts(result.prompt) for i in range(3)), (
        "A1: the memory block goes *before* any history turn does"
    )
    assert "episode 0" in texts(result.prompt), "the higher-ranked hit stays"
    assert "episode 1" not in texts(result.prompt), "the lowest-ranked hit is the one that goes"
    assert trimmed(result) == {"history_turns": 0, "images": 2, "memory_hits": 1}
    floor_is_intact(result.prompt, current)


def test_a_whole_history_turn_goes_only_when_captions_and_memory_are_not_enough():
    """Criterion 2, rung (3) of ruling A1: the last rung before the floor.

    Budget between "all captions, no memory hits" and that same prompt minus the oldest turn. Only
    now may a turn disappear, and it is the oldest. The pixels spent on the earlier rungs stay
    spent: dropping a turn must not re-admit a younger turn's image (the frame slots freed by the
    drop are not a reason to re-send 1120 tokens the budget has already refused).
    """
    builder = PromptBuilder(SYSTEM)
    turns, current, memory = history(3), frame("z"), hits(0.9, 0.4)
    captioned = expected_prompt(builder, turns, current, ())
    expected = expected_prompt(builder, turns[1:], current, ())
    budget = FakeEngineTokenizer.count(expected)
    assert FakeEngineTokenizer.count(captioned) > budget, (
        "regime: dropping the oldest turn must actually be needed "
        f"({FakeEngineTokenizer.count(captioned)} vs {budget})"
    )

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )
    assert result.fits is True
    assert result.prompt == expected, (
        "the oldest turn goes, the two newest stay as captioned text, and no image comes back"
    )
    assert "h0 " not in texts(result.prompt) and "cap0" not in texts(result.prompt), (
        "the dropped turn leaves nothing behind — that is a drop, not a degrade"
    )
    assert images(result.prompt) == [current.phash], (
        f"pixels already spent are not re-sent when a turn goes; got {images(result.prompt)}"
    )
    assert trimmed(result) == {"history_turns": 1, "images": 2, "memory_hits": 2}
    floor_is_intact(result.prompt, current)


def test_two_history_turns_go_when_one_is_not_enough():
    """Criterion 2, rung (3) again: it stops as soon as it fits, and not before."""
    builder = PromptBuilder(SYSTEM)
    turns, current, memory = history(3), frame("z"), hits(0.9, 0.4)
    one_left = expected_prompt(builder, turns[2:], current, ())
    two_left = expected_prompt(builder, turns[1:], current, ())
    budget = FakeEngineTokenizer.count(one_left)
    assert FakeEngineTokenizer.count(two_left) > budget, "regime: two turns must not fit"

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )
    assert result.fits is True
    assert result.prompt == one_left, "exactly the two oldest turns go, the newest is kept"
    assert trimmed(result) == {"history_turns": 2, "images": 2, "memory_hits": 2}
    floor_is_intact(result.prompt, current)


def test_memory_hits_go_lowest_ranked_first():
    """Criterion 2, step (3), with no history at all so the step is reached on its own.

    Three hits scored 0.9 / 0.5 / 0.2; the budget fits the floor plus one hit line. The 0.9 hit
    is the one that stays — "lowest-ranked first" is an order, not a truncation of the block.
    """
    builder = PromptBuilder(SYSTEM)
    current, memory = frame("z"), hits(0.9, 0.5, 0.2)
    all_three = expected_prompt(builder, (), current, memory)
    best_only = expected_prompt(builder, (), current, memory[:1])
    budget = FakeEngineTokenizer.count(best_only)
    assert FakeEngineTokenizer.count(all_three) > budget, "regime: three hits must not fit"

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=(),
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        measure=FakeEngineTokenizer(),
    )
    assert result.fits is True
    assert result.prompt == best_only, (
        "the surviving prompt is the one built from the highest-scored hit alone"
    )
    assert "episode 0" in texts(result.prompt), "the best-ranked hit stays"
    assert "episode 2" not in texts(result.prompt) and "episode 1" not in texts(result.prompt), (
        "the two lowest-ranked hits are the ones that go"
    )
    assert trimmed(result) == {"history_turns": 0, "images": 0, "memory_hits": 2}
    floor_is_intact(result.prompt, current)


def test_one_more_history_turn_goes_on_request():
    """Criterion 4's lever, seen from the builder: the session's retry after a 400/413.

    ``drop_history_turns=1`` removes one turn *before* any measuring, so a retry is a strictly
    smaller prompt even when the engine's count said the first attempt fit (the count the engine
    reported and the count it charged disagreed — see the session's adversarial test).
    """
    builder = PromptBuilder(SYSTEM)
    turns, current = history(3), frame("z")
    resolve = resolver(turns)
    expected = expected_prompt(builder, turns[1:], current, (), resolve=resolve)

    result = fit(
        builder,
        "4",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=(),
        resolve=resolve,
        measure=lambda prompt: 1,  # the engine says everything fits
        drop_history_turns=1,
    )
    assert result.prompt == expected, (
        "drop_history_turns is applied even when the measured count fits: it is the retry lever"
    )
    assert trimmed(result)["history_turns"] == 1
    floor_is_intact(result.prompt, current)


# --- adversarial ------------------------------------------------------------------------------


def test_a_count_over_the_limit_for_the_floor_never_touches_the_users_turn():
    """Adversarial (the task's named failure mode): the engine reports more than the budget for a
    prompt that cannot be made smaller.

    An engine whose count is always astronomical — a wrong ``--max-model-len``, a tokenizer that
    counts an image at its full patch cost, a bug — must not be able to talk the builder into
    deleting the user's question, the system message or the frame, and must not loop for ever.
    The honest outcome is ``fits=False`` with the floor prompt, which the session turns into a
    loud error (criterion 4) rather than a silently mutilated turn.
    """
    builder = PromptBuilder(SYSTEM, Budget(max_prompt_tokens=10))
    turns, current, memory = history(3), frame("z"), hits(0.9, 0.4)
    measure = FakeEngineTokenizer()

    result = fit(
        builder,
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolver(turns),
        measure=lambda prompt: measure(prompt) + 1_000_000,
    )
    assert result.fits is False, (
        "a prompt that cannot be made to fit must say so, not claim success"
    )
    floor_is_intact(result.prompt, current)
    assert len(measure.calls) <= 64, (
        "the search is bounded: an engine that never says yes must not spin "
        f"({len(measure.calls)} measurements for a 3-turn history)"
    )
    assert trimmed(result)["history_turns"] == 3, (
        "everything droppable was dropped before giving up; the floor is what is left"
    )


def test_a_rung_that_would_blind_the_model_is_skipped_not_spent():
    """Ruling A10: the last copy of the current frame's pixels is never spent.

    The screen has not changed since the previous turn, so the builder sends the pixels once — on
    the *history* turn — and the current user turn carries no image of its own (D9/B9: the cached
    image already holds them). Rung 1 aims at exactly those pixels, and spending them would make
    the model answer about a screen it cannot see, so the rule is a floor, not a preference.

    What makes the rule *observable* is that spending the rung here does not even save tokens: the
    frame degrades to a caption, the current turn must then attach the same pixels itself, and the
    prompt comes out **larger** than it started (1314 against 1300 by the fake engine's count).
    The rung must therefore fall through to the next step that removes something — dropping the
    turn, 1264 — and report that, never "one image trimmed" with nothing gone.

    The regime is the pre-emptive one (``max_rungs=1``, ruling A6: a count over budget spends one
    rung and sends). That is where the rule bites: with an unbounded ladder a wasted rung is
    invisible, because the next step reaches the same prompt anyway. Here a wasted rung *is* the
    whole budget for this request, so without the floor the turn goes out at 1314 tokens — over
    budget, with nothing removed and a record that says nothing was trimmed.
    """
    builder = PromptBuilder(SYSTEM)
    current = frame("z")
    shared = HistoryTurn(
        user_text="h0 what is this window about",
        assistant_spoken="a0 it is a spreadsheet of quarterly costs",
        frame_phash=current.phash,  # the same screen: one copy of the pixels, on the history turn
        frame_caption="cap0 a spreadsheet with twelve rows",
    )
    resolve = resolver((shared,))
    whole = expected_prompt(builder, (shared,), current, (), resolve=resolve)
    spent = expected_prompt(builder, (shared,), current, ())  # rung 1 taken: caption + re-attach
    expected = expected_prompt(builder, (), current, ())  # rung 3: the turn goes, pixels stay
    assert images(whole) == [current.phash] and images(expected) == [current.phash], (
        "regime: one copy of the pixels before the trim and one after — the floor is about the "
        f"last copy, not about the history turn; got {images(whole)} and {images(expected)}"
    )
    assert FakeEngineTokenizer.count(spent) > FakeEngineTokenizer.count(whole), (
        "regime: spending this rung *grows* the prompt "
        f"({FakeEngineTokenizer.count(whole)} → {FakeEngineTokenizer.count(spent)}), which is why "
        "a rung that removes nothing must not be counted as one"
    )
    budget = FakeEngineTokenizer.count(expected)
    assert FakeEngineTokenizer.count(whole) > budget, "regime: something must actually be trimmed"

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=(shared,),
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=(),
        resolve=resolve,
        measure=FakeEngineTokenizer(),
        max_rungs=1,
    )
    assert result.fits is True and result.tokens == budget, (
        f"the ladder must reach a prompt that fits; it stopped at {result.tokens} "
        f"(fits={result.fits}) against a budget of {budget}"
    )
    assert result.prompt == expected, (
        "the step that fits is the one that drops the turn; the pixels come back on the current "
        "user turn, which is the whole reason the rung below it may be skipped"
    )
    assert trimmed(result) == {"history_turns": 1, "images": 0, "memory_hits": 0}, (
        "ruling A10: a rung that removes nothing is skipped, not spent — the record must say one "
        f"history turn went and no image did; it says {dict(trimmed(result))}"
    )
    floor_is_intact(result.prompt, current)


def test_a_kept_history_frame_degrades_to_its_caption_and_never_vanishes():
    """B3 under trimming: a frame degrades from pixels to text, it never silently disappears.

    Whatever the trimmer drops, every history turn *still in the prompt* must carry either its
    image or its caption line. A turn whose text is kept while its frame is deleted without a
    caption is a model answering about a screen it was told about and cannot see.
    """
    builder = PromptBuilder(SYSTEM)
    turns, current, memory = (
        history(3),
        frame("z"),
        hits(
            0.9,
        ),
    )
    resolve = resolver(turns)
    whole = expected_prompt(builder, turns, current, memory, resolve=resolve)
    budget = FakeEngineTokenizer.count(whole) - FakeEngineTokenizer.PER_IMAGE

    result = fit(
        PromptBuilder(SYSTEM, Budget(max_prompt_tokens=budget)),
        "2",
        history=turns,
        frame=current,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=memory,
        resolve=resolve,
        measure=FakeEngineTokenizer(),
    )
    body = texts(result.prompt)
    kept_phashes = set(images(result.prompt))
    for index, turn in enumerate(turns):
        if f"h{index} " not in body:
            continue  # the whole turn went; that is a drop, not a vanished frame
        assert turn.frame_phash in kept_phashes or f"cap{index}" in body, (
            f"B3: history turn {index} is still in the prompt but its frame left no trace — a "
            "frame degrades to its caption, it never vanishes"
        )
    floor_is_intact(result.prompt, current)
