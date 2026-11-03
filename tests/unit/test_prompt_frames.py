"""T-0059 group B — a turn may carry more than one frame, concatenated before the transcript.

Criteria 8-17 of ``docs/tasks/T-0059.md``. Everything here is derived from that file's criteria
and design rulings (D1, D2, D4, D6, D7), from ``docs/02-architecture.md`` §5 and from
``.claude/rules/conductor.md``'s prompt-layout bullet — never from the builder's behaviour.

Names this module fixes
=======================
The task names three of them as contract; the fourth is the test-author's, fixed here so the
implementation and the group C tests agree (the house pattern of ``test_prompt_budget.py``).
Report anything wrong to the reviewer; the implementer never edits ``tests/``.

``local_astra/conductor/prompt_builder.py``

    def frame_label(index: int) -> str          # criterion 14, name is contract; pure, index only

        Ruling A5: ``frame_label(0) == ""`` and an empty label is **omitted**, so the first image
        of a turn is unlabelled exactly as today (T-0053 criterion 10's byte-identity row stays
        green) and labels start at the second image. The index is the position among the current
        images *actually attached*, not among the frames handed in: a frame deduplicated against
        the newest history image contributes no image, so the image that does go on the wire
        first is the one the model is told to treat as the first (A5: "the model refers to the
        unlabelled one as the first"). Labelling a lone image ``[screen 2]`` would make the
        prompt assert a screen that is not in it, which is the T-0053 class of bug.

    @dataclass(frozen=True)
    class Budget:
        max_frames_per_turn: int = 3            # criterion 11, name is contract; D2/ruling A1

    PromptBuilder.prefix(history, frame=None, assists, resolve=None, *,
                         spent_images=frozenset(), frames: Sequence[Frame] = ()) -> Prompt
    PromptBuilder.fit(*, history, frame=None, ..., frames: Sequence[Frame] = ()) -> FitResult

        ``frame`` stays accepted and ``frame=f`` is exactly ``frames=(f,)`` (implementation
        notes); exactly one of the two may be non-empty.

    FRAME_LABEL_*                               # a module-level ``str`` constant carrying the
                                                # wording family of the labels (criterion 14).
                                                # Discovered by name prefix, never called.

How the expectations are computed
=================================
By hand, or from the module's own *constants* — never from a copied literal of the system block
(T-0053 ruling A3: T-0055 is rewriting that block and owes a token count for it, so a test that
copied its text would pin a value this task does not own). The one literal taken from elsewhere
is ``[screen: {caption}]``, which T-0053 criterion 10 froze and which
``tests/unit/test_session_over_long_prompt.py`` already carries as its ``CAPTION`` canary.

The engine's token count is stood in for by ``FakeEngineTokenizer`` from
``tests/unit/test_prompt_budget.py`` (4 tokens per message, 2 per word, 1120 per image):
structural, not character-proportional, so a chars/4 estimator cannot pass these rows by
accident. The budgets in the ``fit()`` rows are set *between* two prompts built through the
ordinary ``prefix()``/``full()`` API, so "this budget forces exactly this cut" is visible rather
than asserted.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import re
from collections.abc import Sequence
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb_module
from local_astra.conductor.prompt_builder import (
    ANSWER_HEADROOM_TOKENS,
    CLOSE_THE_THOUGHT,
    JUST_NOW,
    MEMORY_HEADER,
    NO_IMAGE_STATEMENT,
    RECOLLECTION_MARKER,
    SPOKEN_STYLE,
    UNKNOWN_AGE,
    Budget,
    PromptBuilder,
    relative_age,
)
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.contracts.llm import ImagePart, Message, Part, Prompt, TextPart
from local_astra.contracts.memory import Episode, MemoryHit
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.contracts.turn import HistoryTurn
from tests.unit.test_prompt_budget import FakeEngineTokenizer
from tests.unit.test_prompt_builder import _max_model_len, _mm_image_limit

SYSTEM = "You are Astra."
TRANSCRIPT = "which of these two screens shows the later date"

#: T-0053 criterion 10 froze ``_caption_part``'s rendering; the same literal is the ``CAPTION``
#: canary of ``tests/unit/test_session_over_long_prompt.py``. Criterion 16 keeps it frozen here.
CAPTION_TEMPLATE = "[screen: {caption}]"


# --- fixtures built by hand ---------------------------------------------------------------------


def frame(tag: str, ts_ms: int = 0) -> Frame:
    """A frame whose bytes and hash are both derived from ``tag``, so no two can be confused."""
    return Frame(
        source=Source.SCREEN,
        ts_ms=ts_ms,
        width=1280,
        height=720,
        jpeg=b"\xff\xd8" + tag.encode() * 8,
        phash=tag * 16,
        title=f"window {tag}",
    )


def history(n: int = 3, *, captions: bool = True, phashes: Sequence[str] | None = None):
    tags = list(phashes) if phashes is not None else [f"{i:x}" * 16 for i in range(n)]
    return tuple(
        HistoryTurn(
            user_text=f"h{i} what is this window about",
            assistant_spoken=f"a{i} it is a page of a report",
            frame_phash=tags[i],
            frame_caption=f"cap{i} a page with twelve rows" if captions else None,
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


def data_url(jpeg: bytes) -> str:
    """The data URL the layout specifies (docs/02 §5); written out, not read from the module."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


# --- reaching the names the implementer owns -----------------------------------------------------


def _budget(criterion: str, **kwargs: Any) -> Budget:
    try:
        return Budget(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0059 criterion {criterion}: Budget has no {sorted(kwargs)} field. The builder's "
            "cap on current-turn frames is ``Budget.max_frames_per_turn`` (the name is contract) "
            f"and it is the authoritative one — only the builder sees the whole prompt: {exc}"
        ) from None


def _cap(budget: Budget | None = None, criterion: str = "11") -> int:
    value = getattr(budget or Budget(), "max_frames_per_turn", None)
    assert isinstance(value, int), (
        f"T-0059 criterion {criterion}: Budget.max_frames_per_turn does not exist; the cap on "
        "current-turn frames is the builder's and the engine's --limit-mm-per-prompt is why"
    )
    return value


def _frame_label(index: int, criterion: str = "14") -> str:
    fn = getattr(pb_module, "frame_label", None)
    assert callable(fn), (
        f"T-0059 criterion {criterion}: prompt_builder.frame_label(index) does not exist. It is a "
        "module-level pure function of the frame's index alone (D4: a label that spells the total "
        "changes when the next frame arrives and invalidates every cached block before it)"
    )
    return fn(index)


def _prefix(
    pb: PromptBuilder,
    hist: Sequence[HistoryTurn] = (),
    *,
    frames: Sequence[Frame] = (),
    frame: Frame | None = None,
    assists: AssistFlags | None = None,
    resolve: Any = None,
    criterion: str,
) -> Prompt:
    try:
        return pb.prefix(
            hist,
            frame,
            AssistFlags() if assists is None else assists,
            resolve,
            frames=tuple(frames),
        )
    except TypeError as exc:
        raise AssertionError(
            f"T-0059 criterion {criterion}: PromptBuilder.prefix does not accept the keyword-only "
            "``frames: Sequence[Frame] = ()`` parameter of the implementation notes (``frame`` "
            f"stays accepted and ``frame=f`` is exactly ``frames=(f,)``): {exc}"
        ) from None


def _fit(pb: PromptBuilder, criterion: str, **kwargs: Any) -> Any:
    try:
        return pb.fit(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0059 criterion {criterion}: PromptBuilder.fit does not accept the specified "
            f"signature ({sorted(kwargs)}); ``frames`` is keyword-only with default ``()``: {exc}"
        ) from None


def _label_constant(criterion: str = "14") -> str:
    """The module-level constant carrying the labels' wording family.

    Discovered by name prefix and **never called** (``.claude/rules/tests.md``: a test that
    invokes what it discovered dictates arity and can fire side effects) — only ``str`` values
    are considered, so a helper function named ``FRAME_LABEL_FOR`` cannot be reached from here.
    """
    found = {
        name: value
        for name, value in vars(pb_module).items()
        if name.startswith("FRAME_LABEL") and isinstance(value, str) and value
    }
    assert found, (
        f"T-0059 criterion {criterion}: no module-level FRAME_LABEL* string constant in "
        "prompt_builder.py. The labels' wording lives in a constant the tests read for its "
        "wording family, so the wording can be tuned without editing this module"
    )
    return max(found.values(), key=len)


# --- reading the prompt --------------------------------------------------------------------------


def images(prompt: Prompt) -> list[ImagePart]:
    return [p for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]


def phashes(prompt: Prompt) -> list[str | None]:
    return [p.phash for p in images(prompt)]


def current_images(prompt: Prompt) -> list[ImagePart]:
    """Images in the current user turn — the last message, which ``full()`` extends."""
    return [p for p in prompt.messages[-1].content if isinstance(p, ImagePart)]


def history_images(prompt: Prompt) -> list[ImagePart]:
    return [p for m in prompt.messages[:-1] for p in m.content if isinstance(p, ImagePart)]


def serialised(prompt: Prompt) -> list[tuple[str, str]]:
    """Every part of every message, serialised — criterion 9 compares these, never lengths."""
    return [(m.role, p.model_dump_json()) for m in prompt.messages for p in m.content]


def texts(prompt: Prompt) -> str:
    return " ".join(m.text() for m in prompt.messages)


def is_subsequence(small: Sequence[Any], large: Sequence[Any]) -> bool:
    it = iter(large)
    return all(item in it for item in small)


# --- criterion 8: N frames, oldest first, all before the text ------------------------------------


def test_three_frames_are_all_before_the_transcript_in_capture_order():
    """Criterion 8, on the exact part sequence.

    ``[img1][label2][img2][label3][img3]`` and nothing else in the prefix's user turn; then
    ``full()`` may only append after the last pixel, and the transcript is the last part of the
    last message (``.claude/rules/conductor.md``'s layout bullet, docs/01 D9).

    Ruling A5 settled the shape: the first image is **unlabelled**, exactly as a one-frame turn
    is today, and labels start at the second. That is what keeps T-0053 criterion 10's
    byte-identity row (``test_an_ordinary_turn_is_byte_identical``) green while criterion 8 still
    gets a positional label on every image after the first.
    """
    pb = PromptBuilder(SYSTEM)
    a, b, c = frame("a", 100), frame("b", 200), frame("c", 300)
    pre = _prefix(pb, (), frames=(a, b, c), criterion="8")

    assert _frame_label(0) == "", (
        "criterion 8/ruling A5: frame_label(0) is the empty string and an empty label is omitted "
        "— the first image of a turn is unlabelled, exactly as a one-frame turn is today"
    )
    expected: tuple[Part, ...] = ()
    for index, f in enumerate((a, b, c)):
        label = _frame_label(index)
        if label:
            expected = (*expected, TextPart(text=label))
        expected = (*expected, ImagePart(url=data_url(f.jpeg), phash=f.phash))
    assert pre.messages[-1].content == expected, (
        "criterion 8/A5: the current user message is image, label, image, label, image — oldest "
        "first, every image before any text the query contributes"
    )
    assert isinstance(pre.messages[-1].content[0], ImagePart), (
        "ruling A5: the turn opens on its first image with no label in front of it; a label "
        "there is what T-0053 criterion 10's byte-identity row reddens on"
    )
    assert [p.phash for p in current_images(pre)] == [a.phash, b.phash, c.phash]

    full = pb.full(pre, TRANSCRIPT, None, [])
    last = full.messages[-1]
    added = range(len(pre.messages[-1].content), len(last.content))
    pixels = [i for i, p in enumerate(last.content) if isinstance(p, ImagePart)]
    assert pixels and max(pixels) < min(added), (
        "criterion 8/D9: every ImagePart precedes every part full() added — the pixels are in "
        "the cached prefix, the query-dependent text is after them"
    )
    tail = last.content[-1]
    assert isinstance(tail, TextPart) and tail.text == TRANSCRIPT, (
        "criterion 8: the transcript is the last part of the last message"
    )


# --- criterion 9: the prefix grows by appending, byte-exactly ------------------------------------

#: The shapes a turn can grow through. Each is ``(history turns, resolve?, frames)``; the
#: k-frame prompt's **current user message** must be a strict prefix of the (k+1)-frame one's
#: (ruling A5), and the whole prompt too wherever no history image is evicted.
GROWTH_CASES = {
    "no-history": ((), False, ("a", "b", "c")),
    "history-with-pixels": (history(2), True, ("a", "b", "c")),
    "history-captions-only": (history(2), False, ("a", "b", "c")),
    # The first frame's pixels already ride on the newest history image (criterion 10), so the
    # first prefill attaches nothing and the second one appends the first image of the turn.
    "first-frame-rides-history": (history(1, phashes=("a" * 16,)), True, ("a", "b", "c")),
    # A scene that changes back to a screen already in this turn adds no image; the append
    # property must still hold across the repeat.
    "repeat": ((), False, ("a", "b", "a")),
}

#: ``(case, k)`` where the step from k to k+1 current frames evicts a history image, so the
#: *whole-prompt* byte-prefix property legitimately fails there (ruling A5's accepted cost).
#: Hand-derived from the budget arithmetic, not read off an implementation: ``_image_turns``
#: leaves ``max_images - len(current images)`` = ``4 - n`` slots, and only ``history-with-pixels``
#: offers more than one history image (two kept turns, both inside ``image_age_turns = 2``, both
#: resolvable). Its history image count runs 2, 2, 2, 1 as the turn gathers 0, 1, 2 and 3 current
#: images, so the single eviction is at the k=2 step. ``first-frame-rides-history`` offers one
#: history image and never loses it; the other three cases put no history image on the wire.
EVICTS = {("history-with-pixels", 2)}


@pytest.mark.parametrize("case", sorted(GROWTH_CASES))
@pytest.mark.parametrize("k", (0, 1, 2))
def test_each_extra_frame_only_appends_to_the_prefix(case: str, k: int):
    """Criterion 9, the headline adversarial row (docs/01 D9, docs/03 §2, B11 >= 0.9).

    Every prefill sent during a turn must be a byte-prefix of the next one and of the final
    request, so the mutation this row exists to catch is **a label that spells the total** —
    ``[screen 1 of 3]``. Under that mutation the first label changes when the third frame
    arrives, so the k-frame prompt stops being a prefix of the (k+1)-frame one and every cached
    block from the first image onwards misses: a latency regression that is invisible in the
    code and visible only in the engine's prefix-cache hit rate.

    The comparison is on the *serialised* parts, not on counts or lengths: a row that compared
    lengths would stay green under exactly that mutation, since the mutated prompt has the same
    number of parts. Growth is expected only when the extra frame contributes a new image, which
    the k-frame prompt itself says (its phashes), not a re-implementation of the dedup rule.

    Ruling A5 scopes the guaranteed property to the **current user message**; the whole prompt
    keeps it except where the third current image evicts a history image (``EVICTS``, the cost
    the ruling accepts and group C records as ``history_images_evicted``). The eviction
    expectation is asserted in both directions, so an implementation that evicts where it should
    not — or that keeps four images plus the current ones — reddens here too.
    """
    turns, use_resolver, tags = GROWTH_CASES[case]
    pb = PromptBuilder(SYSTEM)
    resolve = resolver(turns) if use_resolver else None
    frames = tuple(frame(tag, 100 * i) for i, tag in enumerate(tags))

    small = _prefix(pb, turns, frames=frames[:k], resolve=resolve, criterion="9")
    large = _prefix(pb, turns, frames=frames[: k + 1], resolve=resolve, criterion="9")

    head = [p.model_dump_json() for p in small.messages[-1].content]
    tail = [p.model_dump_json() for p in large.messages[-1].content]
    assert tail[: len(head)] == head, (
        f"criterion 9/D9/A5 [{case}, k={k}]: the {k}-frame turn's own message is not a prefix of "
        f"the {k + 1}-frame one. The first part that differs is index "
        f"{next((i for i, (x, y) in enumerate(zip(head, tail, strict=False)) if x != y), len(tail))}"
        " — a label that mentions the total does exactly this"
    )
    grows = frames[k].phash not in phashes(small)
    assert (len(tail) > len(head)) == grows, (
        f"criterion 9/10 [{case}, k={k}]: a frame whose phash is new adds parts and a repeat "
        f"adds none; expected growth={grows}, parts {len(head)} -> {len(tail)}"
    )

    evicted = [m.model_dump_json() for m in small.messages[:-1]] != [
        m.model_dump_json() for m in large.messages[:-1]
    ]
    assert evicted is ((case, k) in EVICTS), (
        f"criterion 9/11/A5 [{case}, k={k}]: the history block "
        f"{'changed' if evicted else 'did not change'} when the {k + 1}th frame arrived; history "
        f"images went {[i.phash for i in history_images(small)]} -> "
        f"{[i.phash for i in history_images(large)]}. The only step allowed to evict one is the "
        "third current image against two attached history images, and it evicts the oldest"
    )
    if not evicted:
        assert serialised(large)[: len(serialised(small))] == serialised(small), (
            f"criterion 9 [{case}, k={k}]: with no eviction the *whole* prefill is a byte-prefix "
            "of the next request, which is what B11 >= 0.9 is measured against"
        )
        assert small.shares_prefix_with(large), (
            f"criterion 9 [{case}, k={k}]: Prompt.shares_prefix_with must hold from each prefill "
            "to the next — that is the D9 promise the anticipatory prefill is paid for"
        )


def test_a_third_frame_evicts_the_oldest_history_image_and_still_appends():
    """Ruling A5's accepted cost, pinned in both halves (criteria 9 and 11).

    ``_image_turns`` leaves ``max_images - len(current images)`` slots, and the number of current
    images grows during the turn. With two attached history images — two kept turns, both inside
    ``image_age_turns = 2``, which is the most history the default budget ever shows as pixels
    and the session's ordinary state — the turn's third frame takes the fourth slot and the
    **oldest** history image degrades to its caption line (amendment B3: a frame degrades, it
    never vanishes).

    That costs one re-issue the cache from the evicted image onward, which the ruling accepts
    rather than reserve the turn's slots on every ordinary turn. What is *not* given up: the
    current user message still only appends, so the pixels and labels already prefilled are
    still there, byte for byte, in the request that follows. Both halves are asserted here, and
    the eviction is asserted to be the *oldest* image rather than merely "one of them".
    """
    pb = PromptBuilder(SYSTEM)
    turns = history(2)
    resolve = resolver(turns)
    frames = (frame("a", 100), frame("b", 200), frame("c", 300))
    two = _prefix(pb, turns, frames=frames[:2], resolve=resolve, criterion="9")
    three = _prefix(pb, turns, frames=frames, resolve=resolve, criterion="9")

    assert [i.phash for i in history_images(two)] == [t.frame_phash for t in turns], (
        "the premise: with two current images both history frames are still pixels"
    )
    assert [i.phash for i in history_images(three)] == [turns[1].frame_phash], (
        "criterion 11/A5: the third current image evicts the oldest history image, not the newest"
    )
    assert CAPTION_TEMPLATE.format(caption=turns[0].frame_caption) in texts(three), (
        "B3: the evicted history frame degrades to its caption line, it does not vanish"
    )
    assert len(images(three)) == Budget().max_images, (
        "criterion 11: the whole-prompt cap is what forces the eviction, and it is exactly met"
    )

    head = [p.model_dump_json() for p in two.messages[-1].content]
    tail = [p.model_dump_json() for p in three.messages[-1].content]
    assert tail[: len(head)] == head and len(tail) > len(head), (
        "criterion 9/A5: the turn's own message still only appends — everything already "
        "prefilled is still there, byte for byte, with the third frame's label and image after it"
    )


def test_the_final_request_still_extends_the_last_prefill():
    """Criterion 9's second half: prefix(all frames) -> full(...) is still an extension."""
    pb = PromptBuilder(SYSTEM)
    turns = history(2)
    frames = (frame("a"), frame("b"), frame("c"))
    pre = _prefix(pb, turns, frames=frames, resolve=resolver(turns), criterion="9")
    full = pb.full(pre, TRANSCRIPT, None, hits(0.9))
    assert pre.shares_prefix_with(full), "criterion 9: the final request extends the prefill"
    assert serialised(full)[: len(serialised(pre))] == serialised(pre)


# --- criterion 10: one image per distinct phash --------------------------------------------------


def test_a_repeated_phash_is_one_image():
    """Criterion 10: the scene changed back to a screen this turn already showed."""
    pb = PromptBuilder(SYSTEM)
    a, b = frame("a", 100), frame("b", 200)
    again = frame("a", 300)  # same screen, captured later: a distinct Frame, the same pixels
    pre = _prefix(pb, (), frames=(a, b, again), criterion="10")
    assert [p.phash for p in current_images(pre)] == [a.phash, b.phash], (
        "criterion 10: one ImagePart per distinct phash, in first-occurrence order"
    )
    labels = [p.text for p in pre.messages[-1].content if isinstance(p, TextPart)]
    assert labels == [_frame_label(1)], (
        "criterion 10/A5: a deduplicated frame contributes no image and therefore no label "
        "either; two images carry one label, because the first one is unlabelled"
    )


def test_a_current_frame_matching_the_newest_history_image_is_not_resent():
    """Criterion 10, generalising ``test_unchanged_frame_is_not_attached_twice`` (T-0005 B9).

    The pixels of the unchanged screen already ride on the newest history image, so attaching
    them again costs an image slot of four and tells the model nothing.
    """
    pb = PromptBuilder(SYSTEM)
    turns = history(1, phashes=("c" * 16,))
    c_again, d = frame("c", 200), frame("d", 300)
    pre = _prefix(pb, turns, frames=(c_again, d), resolve=resolver(turns), criterion="10")
    assert phashes(pre) == ["c" * 16, "d" * 16], (
        "criterion 10: the history copy carries the pixels; only the changed screen is attached"
    )
    assert [p.phash for p in current_images(pre)] == ["d" * 16]
    assert pre.messages[-1].content == (ImagePart(url=data_url(d.jpeg), phash=d.phash),), (
        "criterion 10/A5: the label index counts the images actually attached, so the single "
        "image that goes on the wire is the unlabelled first one — labelling it [screen 2] would "
        "make the prompt assert a screen that is not in it"
    )

    # Two-sided: without a resolver the history turn holds no pixels, so the unchanged screen
    # must go on the wire — otherwise the prompt shows nothing of what the user is asking about.
    bare = _prefix(pb, turns, frames=(c_again, d), criterion="10")
    assert [p.phash for p in current_images(bare)] == ["c" * 16, "d" * 16]


# --- criterion 11: two caps, both enforced, ours consistent with the engine's ---------------------


def test_ten_frames_yield_three_current_images_and_four_in_total():
    """Criterion 11, adversarial: the machine's cap is not the builder's, and only the builder
    sees the whole prompt. A fifth image is an HTTP 400 from vLLM, not a degradation.

    With three current images the history keeps exactly ``max_images - 3`` = one image slot, and
    the frames that lost theirs fall back to their caption line (amendment B3) — that is the
    discriminating assertion for ``slots = max_images - len(current images)`` against today's
    ``- (1 if frame else 0)``, which would put six images on the wire.
    """
    pb = PromptBuilder(SYSTEM)
    cap = _cap(criterion="11")
    turns = history(3)
    frames = tuple(frame(chr(ord("a") + i), 100 * i) for i in range(10))
    pre = _prefix(pb, turns, frames=frames, resolve=resolver(turns), criterion="11")

    current = [p.phash for p in current_images(pre)]
    assert len(current) == cap == 3, (
        f"criterion 11/ruling A1: at most {cap} current images per turn; got {len(current)}"
    )
    assert len(images(pre)) <= Budget().max_images == 4, (
        "criterion 11: at most max_images images in the whole prompt — the engine's "
        "--limit-mm-per-prompt is a hard ceiling"
    )
    assert len(history_images(pre)) == 1, (
        "criterion 11: the current frames win; one history image slot is left of the four"
    )
    assert len(set(current)) == len(current)
    assert is_subsequence(current, [f.phash for f in frames]), (
        "criterion 8/11: the frames that survive the cap keep their capture order"
    )
    assert current[0] == frames[0].phash, (
        "D1: a frame that has been sent in a prefill is never removed from the front; frames are "
        "never dropped from the middle and never reordered"
    )
    body = texts(pre)
    assert CAPTION_TEMPLATE.format(caption=turns[0].frame_caption) in body, (
        "B3: a history frame that lost its image slot degrades to its caption, it never vanishes"
    )


def test_the_builder_cap_never_exceeds_the_launchers_image_limit():
    """Criterion 11's parity row: ``services/vllm.sh`` passes
    ``--limit-mm-per-prompt '{"image": 4, "video": 1}'``, so a fifth image is an HTTP 400 before
    a token is decoded. The builder is the only place that sees the whole prompt, so its two
    caps must both sit under the launcher's number and cannot drift from it silently.

    Nothing in T-0059 changes ``services/vllm.sh`` (D2, and that file is T-0058's and the
    human's): this row only reads it.
    """
    limit = _mm_image_limit()
    budget = Budget()
    assert budget.max_images <= limit, (
        f"criterion 11: Budget.max_images ({budget.max_images}) is over the engine's image "
        f"limit ({limit}); one image over it is a 400, not a degraded prompt"
    )
    cap = _cap(criterion="11")
    assert 1 <= cap <= limit - 1, (
        f"criterion 11/D2: the per-turn frame cap ({cap}) must leave at least one of the "
        f"engine's {limit} image slots for the newest history image (D5: a multi-frame turn "
        "still ages into history as one image and one caption)"
    )
    assert cap == 3, "ruling A1: the cap is three; the human vetoes it in the task file, not here"


# --- criterion 12: fit() spends history, never the turn ------------------------------------------


def test_fit_spends_history_pixels_and_never_a_current_frame():
    """Criterion 12 / D6 / T-0020 ruling A1, with the budget set between two hand-built prompts.

    The floor grows from one image to at most three. Rung 1 spends history pixels oldest first;
    ``trimmed["images"]`` counts history images only. The lower bound is the prompt with the one
    history image spent (every history frame a caption — which is exactly what a spent slot
    produces), the upper bound is the prompt that keeps it.
    """
    pb = PromptBuilder(SYSTEM)
    turns = history(3)
    resolve = resolver(turns)
    frames = (frame("x", 100), frame("y", 200), frame("z", 300))

    keeps_it = pb.full(
        _prefix(pb, turns, frames=frames, resolve=resolve, criterion="12"), TRANSCRIPT, None, []
    )
    spends_it = pb.full(
        _prefix(pb, turns, frames=frames, resolve=None, criterion="12"), TRANSCRIPT, None, []
    )
    high, low = FakeEngineTokenizer.count(keeps_it), FakeEngineTokenizer.count(spends_it)
    assert low < high, "the history image really is the difference between the two prompts"

    measure = FakeEngineTokenizer()
    pb = PromptBuilder(SYSTEM, _budget("12", max_prompt_tokens=low, max_frames_per_turn=3))
    result = _fit(
        pb,
        "12",
        history=turns,
        frame=None,
        frames=frames,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        resolve=resolve,
        measure=measure,
    )
    assert result.fits is True and result.tokens == low
    assert result.prompt.messages == spends_it.messages, (
        "criterion 12: the one rung the budget pays for is the history image; the result is the "
        "prompt the ordinary API builds once that frame has degraded to its caption"
    )
    assert [p.phash for p in current_images(result.prompt)] == [f.phash for f in frames], (
        "criterion 12/D6: no rung removes a current-turn image"
    )
    assert result.trimmed["images"] == 1 and result.trimmed["history_turns"] == 0


def test_fit_keeps_all_three_current_frames_when_nothing_can_be_spent():
    """Criterion 12's floor, adversarial: a budget nothing can satisfy.

    The ladder runs out of rungs and returns ``fits=False`` with the user's turn intact — it
    never buys tokens with the screens the question is about. Asserted over *every* prompt the
    engine was asked to measure, not only the last one, so a ladder that drops a current frame
    and re-adds it cannot hide.
    """
    pb_default = PromptBuilder(SYSTEM)
    turns = history(3)
    resolve = resolver(turns)
    frames = (frame("x", 100), frame("y", 200), frame("z", 300))
    measure = FakeEngineTokenizer()
    pb = PromptBuilder(SYSTEM, _budget("12", max_prompt_tokens=1, max_frames_per_turn=3))
    result = _fit(
        pb,
        "12",
        history=turns,
        frame=None,
        frames=frames,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=hits(0.9, 0.4),
        resolve=resolve,
        measure=measure,
    )
    assert result.fits is False, "a budget of one token cannot be met by trimming history"
    wanted = [f.phash for f in frames]
    assert [p.phash for p in current_images(result.prompt)] == wanted
    assert TRANSCRIPT in texts(result.prompt), "T-0020 ruling A1: the user's turn is never spent"
    assert measure.calls, "criterion 12: fit() measures with the engine's count, it never guesses"
    for step, prompt in enumerate(measure.calls):
        assert [p.phash for p in current_images(prompt)] == wanted, (
            f"criterion 12/D6: rung {step} removed or reordered a current-turn image; the floor "
            "is the system message, the current user turn and every current frame"
        )
    expected_floor = pb_default.full(
        _prefix(pb_default, (), frames=frames, criterion="12"), TRANSCRIPT, None, []
    )
    assert result.prompt.messages == expected_floor.messages, (
        "criterion 12: the floor is the prompt with no history and no hits at all"
    )


def test_a_history_copy_of_a_current_frame_is_spendable_only_while_another_copy_survives():
    """Criterion 12 / D6, generalising ``_oldest_history_image`` from one hash to a set.

    Two past turns looked at screen ``c``; the current turn shows ``a`` and ``c``, so ``c`` is
    not attached a second time and its pixels ride on the newest history image. The *older*
    copy is ordinary history and is spendable; the last copy is the current screen and is not.

    Two-sided, which is the point (``.claude/rules/tests.md``): an implementation that protects
    every copy spends nothing and reddens the count; one that ignores the current frames' hashes
    spends both and blinds the model mid-turn, which reddens the per-rung assertion.
    """
    shared = "c" * 16
    turns = history(2, phashes=(shared, shared))
    resolve = resolver(turns)
    frames = (frame("a", 100), frame("c", 200))
    measure = FakeEngineTokenizer()
    pb = PromptBuilder(SYSTEM, _budget("12", max_prompt_tokens=1, max_frames_per_turn=3))
    result = _fit(
        pb,
        "12",
        history=turns,
        frame=None,
        frames=frames,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        resolve=resolve,
        measure=measure,
    )
    assert (
        phashes(_prefix(pb, turns, frames=frames, resolve=resolve, criterion="12")).count(shared)
        == 2
    ), "the premise: both history turns went on the wire as pixels before any trimming"
    for step, prompt in enumerate(measure.calls):
        assert shared in phashes(prompt), (
            f"criterion 12/D6: rung {step} spent the last copy of a current frame's pixels; a "
            "copy is spendable only while another copy of those pixels survives"
        )
        assert "a" * 16 in phashes(prompt), "the other current frame is never spendable at all"
    assert result.trimmed["images"] == 1, (
        "criterion 12/D6: exactly one copy of the shared screen was spendable, and it was spent "
        "— protecting both copies is the over-wide failure and costs history that was free"
    )
    assert sorted(phashes(result.prompt)) == sorted(["a" * 16, shared])


def test_every_current_frames_history_twin_is_protected_not_only_the_first():
    """Criterion 12 / D6, the *set* half of ``_oldest_history_image``'s generalisation
    (ruling A12 F1 — the row that was missing).

    Its sibling above gives both history copies one phash, so spending the protected copy makes
    the current frame stop riding history and be re-attached as a current image: the pixels never
    leave the wire and a builder that protects only the first current hash, or none at all, stays
    green. This fixture has no such escape. Two kept turns with **distinct** hashes ``a`` and
    ``b``, both resolvable, and two current frames of exactly those screens: ``b`` is the newest
    history image so the current ``b`` rides it, and ``a`` is attached as the turn's own image
    while its history twin sits in message 1.

    Neither history image is then spendable — ``a``'s history copy is the last one history holds
    of a hash the turn carries, and ``b``'s is the only copy of the pixels the turn is riding —
    so the image rung is **skipped** and the ladder pays with a whole history turn instead. That
    is the discriminating measurement: ``{"history_turns": 1, "images": 0}`` here, against
    ``{"history_turns": 0, "images": 1}`` for a builder that protects only ``sorted(current)[:1]``
    or protects nothing, which would spend ``b``'s pixels — the ones the current turn is riding
    — to save tokens.

    Note which copies count: ``_oldest_history_image`` looks only at the history messages, so the
    turn's own image of ``a`` does not make its history twin spendable. That is the conservative
    direction (the model never goes blind mid-turn) and it is the rule T-0005 already had, read
    over a set instead of one hash.
    """
    turns = history(2, phashes=("a" * 16, "b" * 16))
    resolve = resolver(turns)
    frames = (frame("a", 100), frame("b", 200))
    pb = PromptBuilder(SYSTEM)

    untrimmed = pb.full(
        _prefix(pb, turns, frames=frames, resolve=resolve, criterion="12"), TRANSCRIPT, None, []
    )
    one_turn_gone = pb.full(
        _prefix(pb, turns[1:], frames=frames, resolve=resolve, criterion="12"),
        TRANSCRIPT,
        None,
        [],
    )
    assert phashes(untrimmed) == ["a" * 16, "b" * 16, "a" * 16], (
        "the premise: the newest history image is b, so the current b rides it and the current a "
        "is attached alongside its own history twin"
    )
    high, low = (
        FakeEngineTokenizer.count(untrimmed),
        FakeEngineTokenizer.count(one_turn_gone),
    )
    budget = 4_000
    assert low <= budget < high, (
        f"the budget must force exactly one rung: the untrimmed prompt is {high} tokens and the "
        f"one with the oldest turn gone is {low}, so {budget} sits between them"
    )

    measure = FakeEngineTokenizer()
    fitted = PromptBuilder(SYSTEM, _budget("12", max_prompt_tokens=budget))
    result = _fit(
        fitted,
        "12",
        history=turns,
        frames=frames,
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        resolve=resolve,
        measure=measure,
    )
    assert dict(result.trimmed) == {"history_turns": 1, "images": 0, "memory_hits": 0}, (
        "criterion 12/D6: with every current frame's pixels riding or twinned by a history "
        "image, the image rung has nothing to spend and the ladder drops a whole history turn. "
        f"Got {dict(result.trimmed)} — ``images``: 1 means a history image whose hash the turn "
        "carries was spent, which is what protecting only one of the current hashes does"
    )
    assert result.fits is True and result.tokens == low
    assert turns[0].user_text not in texts(result.prompt), "the oldest history turn is the rung"
    assert turns[1].user_text in texts(result.prompt), "only one turn is paid, not both"
    for step, prompt in enumerate(measure.calls):
        assert {"a" * 16, "b" * 16} <= set(phashes(prompt)), (
            f"criterion 12/D6: rung {step} lost the last copy of a current frame's pixels; "
            f"the prompt carried {phashes(prompt)}"
        )


# --- criterion 13: the no-image statement is still exactly right ---------------------------------


@pytest.mark.parametrize(
    ("case", "frame_tags", "turns", "with_resolver", "expected"),
    (
        ("one-current-frame", ("a",), (), False, False),
        ("three-current-frames", ("a", "b", "c"), (), False, False),
        ("no-frames-history-image", (), history(1), True, False),
        ("no-frames-captions-only", (), history(1), False, True),
        ("no-frames-no-history", (), (), False, True),
    ),
)
def test_the_no_image_statement_is_present_only_when_no_image_is(
    case: str, frame_tags, turns, with_resolver: bool, expected: bool
):
    """Criterion 13 (T-0053 criterion 2's four rows, re-run with the new signature), two-sided.

    The predicate is "are there pixels in this prompt", never "how many frames were passed".
    The over-wide direction is the dangerous one: telling a human who is sharing their screen
    that nothing is visible is worse than the bug this task fixes.
    """
    pb = PromptBuilder(SYSTEM)
    frames = tuple(frame(tag, 100) for tag in frame_tags)
    resolve = resolver(turns) if with_resolver else None
    pre = _prefix(pb, turns, frames=frames, resolve=resolve, criterion="13")
    full = pb.full(pre, TRANSCRIPT, None, [])
    body = texts(full)
    assert (len(images(full)) == 0) is expected, f"[{case}] premise: pixels on the wire or not"
    assert (NO_IMAGE_STATEMENT in body) is expected, (
        f"criterion 13 [{case}]: the no-image statement must be "
        f"{'present' if expected else 'absent'} — it is emitted iff the assembled prompt carries "
        "no ImagePart at all"
    )
    if not expected:
        assert "you cannot see anything" not in body, (
            f"criterion 13 [{case}]: not a word of the statement survives when pixels are attached"
        )


# --- criterion 14: labels are positional, bounded and conditional --------------------------------


def test_a_label_is_a_function_of_the_index_alone():
    """Criterion 14 / D4. ``[screen 1 of 3]`` is the mutation this forbids at the source: the
    first label would change when the third frame arrives, and every cached block before it with
    it. Positional labels also keep the two clocks apart — a screen frame's ``ts_ms`` and a
    camera frame's are not necessarily on one clock, so a label carries no elapsed time at all
    (``.claude/rules/conductor.md``).

    Ruling A5: ``frame_label(0)`` is the empty string and is omitted, so the bounds and the
    "no number but your own position" rule apply to the labels that exist, index >= 1.
    """
    fn = getattr(pb_module, "frame_label", None)
    assert callable(fn), "criterion 14: prompt_builder.frame_label(index) does not exist"
    parameters = list(inspect.signature(fn).parameters)
    assert len(parameters) == 1, (
        f"criterion 14: frame_label is a function of the index alone; it takes {parameters}. A "
        "second parameter is where the total gets in"
    )

    cap = _cap(criterion="14")
    first = [_frame_label(i) for i in range(cap)]
    assert first == [_frame_label(i) for i in range(cap)], "criterion 14: frame_label is pure"
    assert first[0] == "", (
        "ruling A5: frame_label(0) is the empty string — the first image of a turn is unlabelled, "
        "which is what keeps a one-frame turn byte-identical to today (criteria 15/16/23)"
    )
    assert all(label.strip() for label in first[1:]), (
        "criterion 8/14: every image after the first carries a label; only index 0 is empty"
    )

    # Same index, different cap, different number of frames on the turn: the same string.
    for cap_value in (2, 3, 4, 10):
        pb = PromptBuilder(SYSTEM, _budget("14", max_frames_per_turn=cap_value))
        for count in range(1, min(cap_value, 3) + 1):
            frames = tuple(frame(chr(ord("a") + i), 100 * i) for i in range(count))
            pre = _prefix(pb, (), frames=frames, criterion="14")
            labels = [p.text for p in pre.messages[-1].content if isinstance(p, TextPart)]
            assert labels == [label for label in first[:count] if label], (
                f"criterion 14/D4/A5: with cap={cap_value} and {count} frame(s) the labels are "
                f"{labels}; they must be the non-empty frame_label(i) of the index alone, so the "
                "label of the second image is the same string whether the turn ends up carrying "
                "two images or three"
            )

    for index, label in enumerate(first[1:], start=1):
        numbers = {int(n) for n in re.findall(r"\d+", label)}
        assert numbers <= {index, index + 1}, (
            f"criterion 14/D4: frame_label({index}) = {label!r} contains {sorted(numbers)}; the "
            "only number a positional label may carry is its own position (0- or 1-based). A "
            "number that is the total changes when the next frame arrives"
        )
        assert len(label) <= 64, (
            f"criterion 14: each label is at most 64 code points; frame_label({index}) is "
            f"{len(label)}"
        )
    assert len(set(first)) == len(first), (
        "criterion 14: the labels distinguish the positions — identical labels cannot answer "
        "'which of the two screens shows the later date'"
    )

    family = _label_constant()
    literals = [chunk for chunk in re.split(r"\{[^{}]*\}", family) if chunk.strip()]
    assert literals, "criterion 14: the wording constant must carry some wording"
    for index, label in enumerate(first[1:], start=1):
        assert all(chunk in label for chunk in literals), (
            f"criterion 14: frame_label({index}) = {label!r} does not come from the module's "
            f"wording constant {family!r}; the tests read the constant so the wording can be "
            "tuned without editing this module"
        )


def test_labels_are_absent_when_the_cap_is_one():
    """Criterion 14's other half, and what makes the rollback lever byte-exact rather than
    approximately exact: labels are emitted **iff** ``budget.max_frames_per_turn > 1``."""
    pb = PromptBuilder(SYSTEM, _budget("14", max_frames_per_turn=1))
    f = frame("a", 100)
    pre = _prefix(pb, (), frames=(f,), criterion="14")
    assert pre.messages[-1].content == (ImagePart(url=data_url(f.jpeg), phash=f.phash),), (
        "criterion 14/15: at cap 1 the current turn is the image and nothing else — no label, "
        "no new part, same order as today"
    )
    assert not any(isinstance(p, TextPart) for p in pre.messages[-1].content)


# --- criterion 15: the rollback lever, prompt half (negative) ------------------------------------


def test_one_frame_per_turn_is_byte_identical_to_today():
    """Criterion 15. ``max_frames_per_turn = 1`` restores today's behaviour exactly.

    The expected value is **derived** (T-0053 ruling A3): the system block comes from
    ``system_prompt`` plus the module's own instruction constants, never from a copy of its text
    — T-0055 is rewriting that block concurrently and owes a token count for it, so a pin that
    copied the literal would assert a value this task does not own.
    """
    budget = _budget("15", max_frames_per_turn=1)
    pb = PromptBuilder(DEFAULT_SYSTEM_PROMPT, budget)
    f = frame("a", 100)
    system_text = "\n".join(
        [DEFAULT_SYSTEM_PROMPT, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS]
    )
    expected = (
        Message(role="system", content=(TextPart(text=system_text),)),
        Message(role="user", content=(ImagePart(url=data_url(f.jpeg), phash=f.phash),)),
    )
    via_frames = _prefix(pb, (), frames=(f,), criterion="15")
    via_frame = pb.prefix((), f, AssistFlags())
    assert via_frame.messages == expected, "criterion 15: frame= at cap 1 is today's prompt"
    assert via_frames.messages == expected, "criterion 15: frames=(f,) at cap 1 is today's prompt"
    assert via_frames.model_dump_json() == via_frame.model_dump_json(), (
        "the implementation notes: frame=f is exactly frames=(f,), one normaliser, no second "
        "code path"
    )
    assert (via_frames.prefix_len, via_frames.thinking, via_frames.assists) == (
        2,
        True,
        AssistFlags(),
    )


def test_one_frame_per_turn_with_history_is_byte_identical_to_today():
    """Criterion 15 over the shape that actually ships: history, captions and a resolver.

    The caption rendering is T-0053 criterion 10's frozen ``[screen: {caption}]`` (the ``CAPTION``
    canary of ``tests/unit/test_session_over_long_prompt.py``), written out here rather than read
    from ``_caption_part``: reading the helper would make the expectation circular.
    """
    budget = _budget("15", max_frames_per_turn=1)
    pb = PromptBuilder(SYSTEM, budget)
    turns = history(2)
    resolve = resolver(turns)
    f = frame("z", 500)
    via_frames = _prefix(pb, turns, frames=(f,), resolve=resolve, criterion="15")
    via_frame = pb.prefix(turns, f, AssistFlags(), resolve)
    assert via_frames.model_dump_json() == via_frame.model_dump_json()
    assert [m.role for m in via_frames.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert via_frames.messages[1].content == (
        TextPart(text=turns[0].user_text),
        ImagePart(
            url=data_url(b"\xff\xd8" + turns[0].frame_phash[:1].encode() * 8),
            phash=turns[0].frame_phash,
        ),
    )
    assert via_frames.messages[-1].content == (ImagePart(url=data_url(f.jpeg), phash=f.phash),), (
        "criterion 15: no label, no new part, same order"
    )


def test_frames_and_frame_are_not_both_accepted():
    """The implementation notes: exactly one of the two may be non-empty. Giving both is a
    caller bug — silently concatenating them would put a frame in the prompt twice under two
    different labels, which is the dedup rule failing where it is least visible.
    """
    pb = PromptBuilder(SYSTEM)
    f, g = frame("a"), frame("b")
    try:
        result = pb.prefix((), f, AssistFlags(), None, frames=(g,))
    except ValueError:
        return
    except TypeError as exc:
        raise AssertionError(
            "T-0059 criterion 16: PromptBuilder.prefix does not accept the keyword-only "
            f"``frames`` parameter at all: {exc}"
        ) from None
    raise AssertionError(
        "the implementation notes: exactly one of ``frame`` and ``frames`` may be non-empty; "
        f"passing both must raise ValueError, not build {phashes(result)}"
    )


# --- criterion 16: nothing in the cached region moves (negative) ---------------------------------


def test_the_cached_system_block_is_byte_identical():
    """Criterion 16 / D7. Compared by hash, derived from ``system_prompt`` plus the module's
    constants — never from a copied literal of the block (T-0053 ruling A3; T-0055 still owes
    this block's token count under ``/launch`` and adding "you may be shown several screens" here
    would invalidate a measurement that has not been taken yet).
    """
    pb = PromptBuilder(DEFAULT_SYSTEM_PROMPT)
    expected = "\n".join([DEFAULT_SYSTEM_PROMPT, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS])
    message = pb.system_message()
    assert message.role == "system" and len(message.content) == 1
    assert hashlib.sha256(message.text().encode()).hexdigest() == (
        hashlib.sha256(expected.encode()).hexdigest()
    ), "criterion 16/D7: the system block gains nothing from this task"

    for name in ("NO_IMAGE_STATEMENT", "MEMORY_HEADER", "RECOLLECTION_MARKER"):
        assert getattr(pb_module, name) not in message.text(), (
            f"criterion 16/D4: {name} must not reach the cached prefix"
        )
    label_names = [n for n in vars(pb_module) if n.startswith("FRAME_LABEL")]
    for name in label_names:
        value = getattr(pb_module, name)
        if isinstance(value, str) and value:
            assert value not in message.text(), (
                f"criterion 16/D7: {name} belongs to the turn's own user message, not to the "
                "prefix-cached system block"
            )


def test_every_constant_this_task_does_not_own_is_unchanged():
    """Criterion 16, the rest of the cached region and the contracts.

    The numbers are derived where they can be: ``max_images`` is the engine's
    ``--limit-mm-per-prompt`` image count and ``max_prompt_tokens`` is ``--max-model-len`` less
    the answer's headroom, both read from ``services/vllm.sh`` (the Budget docstring's own
    arithmetic). ``HistoryTurn`` and the wire parts are enumerated because that is what a
    negative criterion is: no new field, no renamed field (D5 — a multi-frame turn still ages to
    one image and one caption).
    """
    b = Budget()
    assert b.max_images == _mm_image_limit() == 4
    assert b.image_age_turns == 2 and b.max_history_turns == 3
    assert b.max_prompt_tokens + ANSWER_HEADROOM_TOKENS == _max_model_len()
    assert b.max_prompt_tokens == 14_336

    assert relative_age(1_000, None) == UNKNOWN_AGE
    assert relative_age(1_000_000 - 1_000, 1_000_000) == JUST_NOW
    assert MEMORY_HEADER.startswith("[") and RECOLLECTION_MARKER.startswith("- ")
    assert NO_IMAGE_STATEMENT.startswith("[screen]")

    assert set(HistoryTurn.model_fields) == {
        "user_text",
        "assistant_spoken",
        "interrupted",
        "frame_phash",
        "frame_caption",
    }, "criterion 16/D5: HistoryTurn carries one frame_phash and one frame_caption, as today"
    assert set(ImagePart.model_fields) == {"type", "url", "phash"}
    # T-0088 ruling E10: widened by exactly `tools` and `continue_final_message` (T-0088 criterion 1)
    assert set(Prompt.model_fields) == {
        "messages",
        "prefix_len",
        "thinking",
        "assists",
        "tools",
        "continue_final_message",
    }
    assert set(Frame.model_fields) == {
        "source",
        "ts_ms",
        "width",
        "height",
        "jpeg",
        "phash",
        "title",
        "app",
        "crop_of",
    }, "T-0059 changes no wire format, so no fake changes with it"


def test_the_frame_parameter_is_still_accepted_and_still_means_one_frame():
    """Criterion 16: every existing caller keeps working, positionally and by keyword."""
    pb = PromptBuilder(SYSTEM, _budget("16", max_frames_per_turn=1))
    f = frame("a", 100)
    positional = pb.prefix((), f, AssistFlags())
    keyword = pb.prefix(history=(), frame=f, assists=AssistFlags())
    assert positional.model_dump_json() == keyword.model_dump_json()
    assert [p.phash for p in current_images(positional)] == [f.phash]

    signature = inspect.signature(PromptBuilder.prefix)
    assert "frames" in signature.parameters, (
        "criterion 16: prefix keeps ``frame`` and gains a keyword-only ``frames``"
    )
    frames_param = signature.parameters["frames"]
    assert frames_param.kind is inspect.Parameter.KEYWORD_ONLY, (
        "the implementation notes: ``frames`` is keyword-only, so no positional caller shifts"
    )
    assert frames_param.default == (), "``frames`` defaults to the empty sequence"
    assert signature.parameters["frame"].default is None, (
        "criterion 16: ``frame`` gains a default of None so ``frames=`` can be used alone; its "
        "position and meaning are unchanged"
    )


# --- criterion 17: the added text is bounded -----------------------------------------------------


def test_the_added_text_is_bounded():
    """Criterion 17: the labels of a three-frame turn add at most 200 code points to the turn's
    own user message — two labels under ruling A5, since the first image is unlabelled.

    A **structural** bound, explicitly not a token claim (T-0044 ruling A2): the token cost of a
    label is criterion 24's, measured on the engine under ``/launch``. Read from the module's own
    function, so tuning the wording either stays inside the bound or reddens this row.
    """
    cap = _cap(criterion="17")
    labels = [label for label in (_frame_label(i, "17") for i in range(min(cap, 3))) if label]
    total = sum(len(label) for label in labels)
    assert len(labels) == 2, (
        "ruling A5: a three-frame turn carries two labels, the first image being unlabelled"
    )
    assert total <= 200, (
        f"criterion 17: the labels of a three-frame turn add {total} code points ({labels!r}); "
        "the bound is 200 and it is structural, not a token count"
    )

    pb = PromptBuilder(SYSTEM)
    frames = (frame("a", 100), frame("b", 200), frame("c", 300))
    pre = _prefix(pb, (), frames=frames, criterion="17")
    added = sum(len(p.text) for p in pre.messages[-1].content if isinstance(p, TextPart))
    assert added == total, (
        "criterion 17: the only text this task adds to the current turn is the labels themselves"
    )
