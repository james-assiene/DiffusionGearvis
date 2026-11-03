"""Prompt layout (docs/01 D9/D11, docs/11 §2). Task T-0005.

Layout (``.claude/rules/conductor.md``)::

    [<|think|>][system][history: spoken text, aged frames → captions]
    [<turn>user\\n<|image|>(frames, oldest first, each after the first behind its position label)]
    ‖ [screen state][TutorState][recollections, each marked and dated][assists][transcript]

Everything left of ‖ is the cached prefix warmed by the anticipatory prefill; the suffix depends
on the query. ``<|think|>`` is the *first* system-turn token and is a session setting.

This module is pure: no clocks, no I/O, no per-call state. The prefill prefix and the prefix of
the final request must be byte-identical or every cached block misses, so ``prefix()`` is a
function of its arguments only (memory thumbnails, which would need file I/O, are deferred to
T-0009 by amendment B4). History frame pixels arrive the same way: through the ``resolve``
callback ``prefix()`` takes, never by reading a ring buffer here (amendment B9).

Prompt *size* is controlled by the structural caps in ``Budget`` and, in ``fit()``, by the token
budget — measured by the **engine** (``usage.prompt_tokens``), passed in as the ``measure``
callable. Nothing in this module counts tokens: the character-counting estimator was deleted by
amendment B10 (unmeasured, untested, unreachable) and T-0020 replaced it with a number only the
engine can produce. ``prefix()`` and ``full()`` still trim nothing; ``fit()`` is the whole of the
budget's enforcement and its ladder is T-0020 ruling A1.
"""

from __future__ import annotations

import base64
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.contracts.llm import ImagePart, Message, Part, Prompt, TextPart
from local_astra.contracts.memory import MemoryHit
from local_astra.contracts.perception import AssistFlags, Frame, FrameText
from local_astra.contracts.turn import HistoryTurn, TutorState
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

#: System prompt fragment that closes the thought before answering (docs/11 §2.1). The exact
#: wording is tuned in X7; the *presence* of this instruction is asserted by a test.
CLOSE_THE_THOUGHT = (
    "Think step by step inside your thinking channel, then close it and give the spoken answer "
    "as plain sentences. Never leave the answer inside the thinking channel."
)

SPOKEN_STYLE = (
    "You are speaking aloud: short sentences, no markdown, no lists, numbers as words when short. "
    "If you also show something, put it in the ui channel, never in speech."
)

#: T-0092b criterion 3 (H11): appended last to the system message when the session's ``tools`` setting is on.
#: Paragraphs 2 and 3 of ``eval.cd.twopass.NATIVE_SYSTEM_PROMPT``, verbatim (H-T84-0); the persona sentence is
#: omitted because ``system_prompt`` already carries one. A literal, because the conductor may not import the
#: eval; ``tests/cd/unit/test_native_tool_instructions_parity.py`` pins it against twopass.
NATIVE_TOOL_INSTRUCTIONS = (
    "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then "
    "call exactly one of the declared tools. Use at most one call per answer. When no action is needed, answer "
    "normally and call no tool.\n\n"
    "Screen coordinates x and y are integers from 0 to 999, normalised to the attached screenshot: (0, 0) is its "
    "top-left corner and (999, 999) its bottom-right corner."
)

#: T-0131 C1/C2: appended after ``NATIVE_TOOL_INSTRUCTIONS``, last in the system message, only when a session is
#: built with ``tool_guidance`` (``cli`` passes it for ``tools = live`` only). (a) verify the post-action screenshot
#: before claiming success; (b) scroll before answering about a whole page. A literal, so the cached system message
#: is the same bytes in every turn and process; it asks for no narration and no confirmation (speaking policy).
LIVE_TOOL_GUIDANCE = (
    "After an action, the next screenshot shows the screen as it is now. Before saying an action worked, check "
    "that the new screenshot shows its effect (for example, the cart count changed); if it does not, try again or "
    "say it did not work. When asked about a whole page (the cheapest item, every item, a list), scroll to see the "
    "rest of the page before answering."
)

#: T-0174 A2: v1 with its first sentence replaced by one that points backwards and is true at every step (the
#: screenshot already in this turn is the post-action screen), so the model has no sentence to quote as a reason to
#: wait; A2b: the verification sentence says "the latest screenshot" instead of "the new screenshot". It quotes none of the session's label texts (H-T174-3). A literal, for the same cache reason as v1.
LIVE_TOOL_GUIDANCE_V2 = (
    "The latest screenshot in this turn already shows the screen as it is after any action you have taken. "
    "Before saying an action worked, check that the latest screenshot shows its effect (for example, the cart "
    "count changed); if it does not, try again or say it did not work. When asked about a whole page (the cheapest "
    "item, every item, a list), scroll to see the rest of the page before answering."
)

#: T-0191 A1/A2: v2 plus one sentence saying how to go to a different website (Ctrl+L opens the start page, whose
#: address box takes a web address). Carried only in web mode: ``cli`` passes v2's object under ``--no-web``, where
#: Ctrl+L is refused. A literal, for the same cache reason as v1.
LIVE_TOOL_GUIDANCE_V3 = (
    "The latest screenshot in this turn already shows the screen as it is after any action you have taken. "
    "Before saying an action worked, check that the latest screenshot shows its effect (for example, the cart "
    "count changed); if it does not, try again or say it did not work. When asked about a whole page (the cheapest "
    "item, every item, a list), scroll to see the rest of the page before answering. To leave the current website "
    "for another, press Ctrl+L, which opens the start page, then type the other site's web address (such as "
    "github.com) into the start page's address box and press Enter, not into the current site's search box."
)

#: T-0211 B1: a shared screen is the task, not the place to act; the agent's own browser is reached with Ctrl+L.
S_SHARED_GUIDANCE = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: "
    "to do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)

#: T-0211 B1 (James, 2026-10-30): a web-search answer is read on its page and cites it, in the final answer only.
S_SOURCE_GUIDANCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)

#: T-0211 B1: v3 plus the two sentences above. Web mode only: ``cli`` passes v2's object under ``--no-web``. A
#: concatenation of literals, so the cached system message is the same bytes in every turn and process.
LIVE_TOOL_GUIDANCE_V4 = LIVE_TOOL_GUIDANCE_V3 + " " + S_SHARED_GUIDANCE + " " + S_SOURCE_GUIDANCE

#: T-0215: v4 in a live session with no search backend (``google_search`` is not declared there), so no sentence
#: names searching: v3 plus S_SHARED only. ``cli`` passes it in place of v4 only when search is off.
LIVE_TOOL_GUIDANCE_V4_NOSEARCH = LIVE_TOOL_GUIDANCE_V3 + " " + S_SHARED_GUIDANCE

#: T-0174 A2: the live guidance versions, read-only; ``cli`` passes ``LIVE_GUIDANCE[value]`` as ``tool_guidance``
#: for the resolved ``--live-guidance`` (``config.LIVE_GUIDANCE_VERSIONS``), fixed at session construction (T-0191,
#: T-0211: v3 and v4 resolve to v2's object under ``--no-web``).
LIVE_GUIDANCE: Mapping[str, str] = MappingProxyType(
    {
        "v1": LIVE_TOOL_GUIDANCE,
        "v2": LIVE_TOOL_GUIDANCE_V2,
        "v3": LIVE_TOOL_GUIDANCE_V3,
        "v4": LIVE_TOOL_GUIDANCE_V4,
    }
)

#: T-0202 A4 (H-T202-4): appended to the live guidance only when the session offers the ``wait`` tool, so the model
#: looks again at a page still loading instead of telling the user to wait. A literal, for the same cache reason as
#: v1; it quotes none of the session's label texts.
WAIT_TOOL_GUIDANCE = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)


def with_wait_guidance(text: str) -> str:
    """T-0202 A4: ``text`` followed by one space and :data:`WAIT_TOOL_GUIDANCE` (pure)."""
    return text + " " + WAIT_TOOL_GUIDANCE


#: T-0213 A1-4: the fixed header of the user's own instructions block, appended last to the system message.
USER_INSTRUCTIONS_HEADER = (
    "The user's own instructions follow. Apply them unless they conflict with anything above."
)
#: T-0213 A1-4: the bound on the instructions, in code points after stripping surrounding whitespace.
MAX_USER_INSTRUCTIONS = 1000
#: T-0213 A1-4 (Q4): text the instructions may not contain: every special-token literal of the pinned tokenizer
#: (``toolcall.special_tokens.SPECIAL_TOKENS``, proved against the pinned files by ``test_host_detok_parity``;
#: ``services/versions.lock`` pins the chat template by sha only, so there is no template text to parse here) plus
#: the delimiter fragments the template is built from.
INSTRUCTIONS_FORBIDDEN: tuple[str, ...] = ("<|", "|>", "<turn", "turn|>", *SPECIAL_TOKENS)


def validate_instructions(text: str) -> str | None:
    """T-0213 A1-4: why ``text`` cannot be the user's instructions block, or ``None`` when it can (pure).

    Refused: more than :data:`MAX_USER_INSTRUCTIONS` code points once surrounding whitespace is stripped; any
    control or format character other than ``\\n`` and tab (R-2: C0/C1 controls, bidi and other format
    characters, surrogates), anywhere in the unstripped text; any :data:`INSTRUCTIONS_FORBIDDEN` text, also
    when whitespace splits it."""
    stripped = text.strip()
    if len(stripped) > MAX_USER_INSTRUCTIONS:
        return (
            f"your instructions are {len(stripped)} characters long; at most {MAX_USER_INSTRUCTIONS} are "
            "allowed (T-0213)"
        )
    for ch in text:  # the original text: the strip is the length bound's rule, not a licence (R-2)
        if ch not in "\n\t" and unicodedata.category(ch) in ("Cc", "Cf", "Cs"):
            return (
                f"your instructions contain the control character U+{ord(ch):04X}; only plain text, tabs and "
                "line breaks are allowed (T-0213)"
            )
    squeezed = "".join(stripped.split())
    for lit in INSTRUCTIONS_FORBIDDEN:
        if lit in stripped or lit in squeezed:
            return (
                f"your instructions contain {lit!r}, which is chat-template syntax the model reads as structure; "
                "remove it (T-0213)"
            )
    return None


#: Header of the recollection block in the query-dependent suffix (T-0053, criterion 1d/3). It is
#: a sentence, not a label: the bug it replaces was a bare ``[memory]`` over three captions, which
#: is typographically identical to a caption of the current screen, and the model read it as one.
MEMORY_HEADER = "[recalled, not on screen]"

#: Prefix of every recalled line (criterion 1c/3), ≤ 48 code points. Each line reads
#: ``{RECOLLECTION_MARKER}{relative_age(...)}: {caption}`` — marked, then dated, then the caption.
#:
#: This constant and :data:`MEMORY_HEADER` are kept terse on purpose (ruling A12): criterion 9
#: bounds the framing of a three-hit no-image turn, and the tightest regime is ``now_ms=None`` —
#: production until group C — where the unknown-age phrase is longer than a rendered age. Measured
#: on the three origin captions: 363 code points at a real ``now_ms``, 381 with none, bound 400.
RECOLLECTION_MARKER = "- recalled, not on screen, "

#: Emitted in the *user* turn, first in the query-dependent suffix, when the assembled prompt
#: carries no ``ImagePart`` at all (criterion 1b/2, D3/D4), ≤ 240 code points. The predicate is
#: "are there pixels in this prompt", never ``frame is not None``: ``_assemble`` legitimately omits
#: the current frame when its pixels already ride on the newest history image (D9/B9), and saying
#: "you cannot see" to a human who is sharing their screen is worse than the bug being fixed.
#: The second sentence is D4's minimum: you cannot state "nothing is visible" without saying what
#: follows from it. It names *the recollections* and not "anything below" (ruling A20): the assist
#: blocks also sit below this statement and they *are* on the screen, so the wider clause would
#: have told the model not to describe the page text it was given.
NO_IMAGE_STATEMENT = (
    "[screen] No screen or camera image is attached to this turn: you cannot see anything "
    "right now. Say so if you are asked what you see, and never describe the recollections "
    "below as if they were on the screen."
)

#: Wording of the per-frame position labels (T-0059 criterion 14). A format string, so the
#: wording can be tuned here without touching :func:`frame_label` or the tests that read it.
#:
#: The label carries **its own position and nothing else** (D4): a label naming the total — the
#: ``[screen 1 of 3]`` shape — changes when the next frame arrives, and with it every cached block
#: before it, which is a silent total prefix-cache miss on exactly the turns this task exists for.
#: It carries no elapsed time either: a screen frame's ``ts_ms`` and a camera frame's are not
#: necessarily on one clock (``.claude/rules/conductor.md``), and this module has no clock at all.
FRAME_LABEL_TEMPLATE = "[screen {position}]"


def frame_label(index: int) -> str:
    """The label of the current-turn image at ``index`` (0-based, oldest first). Pure, and a
    function of the index alone — no total, no count, no duration.

    ``frame_label(0)`` is the **empty string** and an empty label is omitted (ruling A5): the
    first image of a turn is unlabelled, exactly as a one-frame turn is today, so nothing in the
    cached prefix of an ordinary turn moves; labels start at the second image and the model reads
    the unlabelled one as the first. ``index`` counts the images this turn actually attaches, not
    the frames handed in (ruling A6): a frame whose pixels already ride on a history image
    contributes no image, and labelling the one image on the wire "[screen 2]" would make the
    prompt assert a screen it does not carry.
    """
    if index <= 0:
        return ""
    return FRAME_LABEL_TEMPLATE.format(position=index + 1)


#: What an age renders as with no clock (``now_ms=None``, the state production is in until T-0053
#: group C wires ``clock.now_ms``): *earlier, time not known* — never "just now", never a count
#: (criterion 5).
UNKNOWN_AGE = "at some earlier time"

#: An episode younger than a minute, and an episode from the future (criterion 6).
JUST_NOW = "just now"

_MINUTE_MS = 60_000
_HOUR_MS = 60 * _MINUTE_MS
_DAY_MS = 24 * _HOUR_MS


def relative_age(ts_ms: int, now_ms: int | None) -> str:
    """A human phrase for how long ago ``ts_ms`` was, given ``now_ms``. Pure: two integers in,
    a phrase out — this module imports no clock and no calendar (criterion 11), which is also why
    "yesterday" here is the *elapsed* 24-48 h window and not a calendar day.

    Ages **truncate** and never round up (ruling A7): 59 minutes is "59 minutes ago", not "an hour
    ago". Rounding up would let the prompt make exactly the class of small false statement this
    module exists to remove. An episode at or after ``now_ms`` — clock skew, a replayed log —
    renders as :data:`JUST_NOW`, never as a negative count and never as an exception.
    """
    if now_ms is None:
        return UNKNOWN_AGE
    elapsed = now_ms - ts_ms
    if elapsed < _MINUTE_MS:
        return JUST_NOW
    if elapsed < _HOUR_MS:
        minutes = elapsed // _MINUTE_MS
        return "a minute ago" if minutes == 1 else f"{minutes} minutes ago"
    if elapsed < _DAY_MS:
        hours = elapsed // _HOUR_MS
        return "an hour ago" if hours == 1 else f"{hours} hours ago"
    if elapsed < 2 * _DAY_MS:
        return "yesterday"
    return f"{elapsed // _DAY_MS} days ago"


#: Assist kind (``FrameText.kind``) → its ``AssistFlags`` field and the label it is sent under.
ASSIST_KINDS: tuple[tuple[str, str, str], ...] = (
    ("page_text", "page_text", "[page text]"),
    ("pdf_text", "pdf_text", "[pdf text]"),
    ("cells", "cells", "[cells]"),
    ("at_spi", "at_spi", "[accessibility tree]"),
)


#: Tokens kept free for the answer when ``max_prompt_tokens`` is fully spent (ruling A5/A8).
#:
#: **Chosen, not measured** (ruling A8): no bench has produced an answer-length number, so 2 048
#: is eight canvases of the lean profile's 256 — a round figure picked to be comfortably longer
#: than any answer seen by hand, and small enough to leave the prompt most of the window. It is
#: headroom, not a cap: the cap is the engine's ``--max-model-len``.
#:
#: The measurement it is waiting for: the p99 of answer tokens per turn over a B3 run
#: (``uv run astra bench B3``, ``turns.jsonl``), regime = profile lean, canvas 256,
#: max_denoising_steps 48, thinking on, the vLLM/model/driver pins of ``services/versions.lock``,
#: headset path, n and seed as the bench declares. When that number exists it replaces this one
#: and moves to ``config.py`` as ``profile.diffusion.max_answer_tokens`` (the A5 test already
#: reads that field when it appears, and falls back to ``canvas_length`` until then).
#:
#: The profile declares no answer budget today: ``LlmOptions.max_tokens`` is unset on the turn
#: path and ``Diffusion`` only declares ``canvas_length``, which is why the number lives here.
ANSWER_HEADROOM_TOKENS = 2_048


@dataclass(frozen=True)
class Budget:
    """The caps this module applies: the structural ones, and — through :meth:`PromptBuilder.fit`
    — the token budget, measured by the *engine* and never guessed here.

    ``max_prompt_tokens`` is 14 336 = 16 384 (``--max-model-len``, T-0014) − 2 048 of answer
    headroom (ruling A5): a prompt that fills the serving window has nothing left to decode into.
    The 24 000 it used to be exceeded the window, so every prompt the old estimator would have
    trimmed was rejected by vLLM before the trimming could matter; and that estimator guessed
    four characters per token, which is why B10 deleted it. Nothing counts here now: ``fit()``
    takes a ``measure`` callable and the only caller passes the engine's own number.
    """

    max_prompt_tokens: int = 14_336  # = 16384 - ANSWER_HEADROOM_TOKENS; see the class docstring
    max_images: int = 4
    image_age_turns: int = 2  # frames older than this become their caption line
    max_history_turns: int = 3
    #: Current-turn frames attached as pixels (T-0059, D2/ruling A1). 3 is the default of the
    #: ``max_frames_per_turn`` setting (T-0227: 1..4, from the Settings panel at a turn boundary,
    #: ``--max-frames-per-turn`` or the profile key; ``config.DEFAULT_MAX_FRAMES_PER_TURN``), not a
    #: constant: a session's builder carries the value in force. At 3 it takes three of the engine's
    #: four image slots, leaving one for the newest history image (D5); at 4 the turn's own screens
    #: take every slot and no history image stays one. There is a second cap of the
    #: same name on ``TurnPolicy`` — the machine's — and **this one is authoritative**: only the
    #: builder sees the whole prompt, and ``max_images`` is the launcher's hard ceiling, over
    #: which a request is an HTTP 400 rather than a degraded prompt. ``1`` is the rollback lever
    #: and restores today's single-frame prompt byte for byte, labels included (criterion 15).
    max_frames_per_turn: int = 3


@dataclass(frozen=True)
class FitResult:
    """What :meth:`PromptBuilder.fit` decided (T-0020 ruling A2).

    ``prompt`` is what to send — the *full* prompt, prefix plus query-dependent suffix. ``tokens``
    is ``measure(prompt)``, the engine's count of exactly that object. ``fits`` is False when the
    ladder ran out of rungs before the count came under budget: the floor is still returned (the
    system message, the current turn and the current frame are never spent), so the caller sends
    it and lets the engine judge, or reports loudly — it does **not** truncate the user's turn.
    ``trimmed`` counts what the ladder spent, and is what the turn record carries (criterion 5).
    """

    prompt: Prompt
    tokens: int
    fits: bool
    trimmed: Mapping[str, int]


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _caption_part(caption: str) -> TextPart:
    return TextPart(text=f"[screen: {caption}]")


def _memory_block(hits: Sequence[MemoryHit], now_ms: int | None) -> str:
    """The recalled episodes, each marked as a recollection and dated; thumbnails are T-0009
    (amendment B4) and stay deferred — an unlabelled thumbnail would be this bug again in pixels.

    Empty when nothing was recalled, so the header and the markers are spent with the lines they
    frame and no orphan header survives ``fit()``'s rung 2 (criterion 8).
    """
    lines = []
    for hit in hits:
        label = hit.episode.caption or hit.episode.window_title
        if label:
            # One physical line per hit (ruling A19): a captioner's caption runs to several lines
            # — the three origin ones carry a blank line and a keyword line — and every line the
            # marker does not reach sits at column zero, where a second line reading
            # ``[screen: …]`` is byte-identical to ``_caption_part``'s label for a real frame.
            # Collapsing loses nothing the model needs; indenting instead costs 24 code points the
            # criterion 9 bound does not have.
            label = " ".join(label.split())
            lines.append(f"{RECOLLECTION_MARKER}{relative_age(hit.episode.ts_ms, now_ms)}: {label}")
    return "\n".join([MEMORY_HEADER, *lines]) if lines else ""


def _assist_blocks(frame_text: Sequence[FrameText], assists: AssistFlags) -> list[str]:
    """Only the kinds whose HUD flag is on; everything else never reaches the model (D12a)."""
    blocks = []
    for kind, flag, label in ASSIST_KINDS:
        if not getattr(assists, flag):
            continue
        for item in frame_text:
            if item.kind == kind and item.text:
                blocks.append(f"{label}\n{item.text}")
    return blocks


def _image_count(prompt: Prompt) -> int:
    return sum(1 for m in prompt.messages for p in m.content if isinstance(p, ImagePart))


def _oldest_history_image(prompt: Prompt, current_phashes: Collection[str]) -> int | None:
    """Position (in the kept history) of the oldest history image that may be spent.

    History lives in every message but the last, which is the current user turn; the prompt runs
    ``[system, user 0, assistant 0, user 1, …]``, so history turn *i* is message ``1 + 2i``.

    The floor: every current frame's pixels stay on the wire. When a screen has not changed,
    ``_assemble`` does not attach that frame a second time and those pixels ride on the newest
    history image (D9/B9) — so an image carrying a current frame's hash is spendable only while
    another copy of it survives. The *last* copy of each current hash is the current screen and is
    never spent; T-0059 D6 generalises this from one hash to the set the turn carries.
    """
    found: list[tuple[int, str]] = []
    for position, message in enumerate(prompt.messages[:-1]):
        for part in message.content:
            if isinstance(part, ImagePart) and part.phash is not None:
                found.append(((position - 1) // 2, part.phash))
    attached = [phash for _, phash in found]
    for index, phash in found:
        if phash not in current_phashes or attached.count(phash) > 1:
            return index
    return None


def _normalise_frames(frame: Frame | None, frames: Sequence[Frame]) -> tuple[Frame, ...]:
    """The current turn's frames, oldest first — one code path for both spellings (T-0059).

    ``frame=f`` is exactly ``frames=(f,)``; giving both is a caller bug, not a concatenation:
    the same screen would go on the wire twice under two different position labels, which is the
    de-duplication rule failing where it is least visible.
    """
    if frame is not None and frames:
        raise ValueError(
            "prefix()/fit() take either ``frame`` or ``frames``, never both "
            "(``frame=f`` is exactly ``frames=(f,)``)"
        )
    return (frame,) if frame is not None else tuple(frames)


#: Default for ``prefix``'s ``assists``. It exists only because ``frame`` gained a default of
#: ``None`` (T-0059 criterion 16: ``frames=`` must be usable alone) and Python allows no
#: non-defaulted parameter after a defaulted one. ``AssistFlags`` is a frozen contract, so one
#: shared instance is safe; every caller in the tree still passes its own flags explicitly.
_NO_ASSISTS = AssistFlags()


def _lowest_ranked(hits: Sequence[MemoryHit]) -> int:
    """Index of the hit to drop first: the lowest score, and the later one when scores tie."""
    return min(range(len(hits)), key=lambda i: (hits[i].score, -i))


class PromptBuilder:
    def __init__(
        self,
        system_prompt: str,
        budget: Budget | None = None,
        thinking: bool = True,
        *,
        tools: tuple[dict[str, Any], ...] | None = None,
        tool_guidance: str | None = None,
        ui_block: str | None = None,
        user_instructions: str | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self.thinking = thinking
        #: T-0092b criterion 3: the native ``tools`` request field, or ``None`` for tools off. A session
        #: setting like ``thinking``: every ``Prompt`` this builder returns carries the same value, so the
        #: prefill and the generate of a turn render the same prefix (q8).
        self.tools = tools
        #: T-0131 C1: extra tool guidance appended last when tools are on (``LIVE_TOOL_GUIDANCE`` in live);
        #: ignored with tools off. A session setting, so the cached system message never changes mid-session.
        self.tool_guidance = tool_guidance
        #: T-0184 C6: the A2UI block of the system message, ``None`` for ``UI_INSTRUCTIONS`` (read
        #: when the message is built, as before) or e.g. ``UI_INSTRUCTIONS_V2``. A session setting:
        #: the default keeps BASE's bytes, and another block changes only its own span.
        self.ui_block = ui_block
        #: T-0213 A1-4: the user's own instructions, appended last under ``USER_INSTRUCTIONS_HEADER``; ``None``
        #: or empty keeps BASE's bytes. A session setting like ``tool_guidance``. Text that
        #: ``validate_instructions`` refuses never reaches a prompt: it is refused here.
        if user_instructions:
            reason = validate_instructions(user_instructions)
            if reason is not None:
                raise ValueError(reason)
        self.user_instructions = user_instructions

    def prefix(
        self,
        history: Sequence[HistoryTurn],
        frame: Frame | None = None,
        assists: AssistFlags = _NO_ASSISTS,
        resolve: Callable[[str], bytes | None] | None = None,
        *,
        spent_images: frozenset[int] = frozenset(),
        frames: Sequence[Frame] = (),
    ) -> Prompt:
        """Everything up to and including the current frames — identical between prefill and final.

        The last message is the *user* turn holding the frames; ``full()`` extends that same
        message. ``frames`` are this turn's screens, **oldest first**, at most
        ``max_frames_per_turn`` of them (3 by default; a setting since T-0227, not a constant),
        each after the first preceded by its :func:`frame_label`
        text part — the labels are part of the *prefix*, next to their image, so a later frame
        only ever appends and every prefill of the turn stays a byte-prefix of the next one (D1).
        ``frame=f`` is exactly ``frames=(f,)`` and giving both raises ``ValueError``.

        A frame whose ``phash`` equals the newest history frame still in image form is not
        attached again (the cached image already carries the pixels), and neither is a second
        frame of a screen this turn already attached; history frames older than
        ``image_age_turns`` appear as their caption line, and at most ``max_images`` images are
        kept in the whole prompt — the current turn's win, and history images are evicted
        oldest first to make room (ruling A5). Thinking is *not* a literal token here — the
        template emits it (``Prompt.thinking``).

        ``resolve`` maps a history turn's ``frame_phash`` to that frame's JPEG bytes (the pixels
        live in the perception ring / memory store, and loading them here would make this module
        do I/O and stop being deterministic). **Without a resolver — the default — a history
        frame degrades to its caption line**, exactly as an aged-out frame does (amendment B9):
        that is why this module never emits an image part the client cannot send. T-0007/T-0009
        pass a real resolver and the pixels come back. Determinism is the caller's duty: the
        prefill and the final request must pass a resolver that answers identically, or every
        cached block after the first differing image misses.

        ``spent_images`` names kept-history *positions* whose pixels the budget has already spent
        (T-0020 rung 1): they degrade to their caption line exactly as an aged-out or unresolved
        frame does, and the slot they free admits nothing older. Positions, not hashes, because
        two turns looking at the same screen carry the same hash and only the older one is spent.
        Empty for every caller but ``fit()``.
        """
        kept = (
            list(history[-self.budget.max_history_turns :]) if self.budget.max_history_turns else []
        )
        # No token-budget trimming here (amendment B10): the structural caps above are the whole
        # control, and the only honest size check is the engine's own count. ``fit()`` is the one
        # caller that measures, and it says which pixels it has already spent.
        messages = self._assemble(kept, _normalise_frames(frame, frames), resolve, spent_images)
        return Prompt(
            messages=tuple(messages),
            prefix_len=len(messages),
            thinking=self.thinking,
            assists=assists,
            tools=self.tools,
        )

    def full(
        self,
        prefix: Prompt,
        transcript: str,
        tutor: TutorState | None,
        memory_hits: Sequence[MemoryHit],
        frame_text: Sequence[FrameText] = (),
        *,
        now_ms: int | None = None,
    ) -> Prompt:
        """Extend the last user message with the query-dependent suffix, in this order:
        no-image statement (only when the assembled prompt carries no pixels at all), TutorState
        block, memory hits (marked and dated caption lines — thumbnails are T-0009, amendment B4),
        assist text (only kinds whose flag is on in ``prefix.assists``), transcript.
        ``prefix_len`` is unchanged and ``prefix.shares_prefix_with(full)`` holds.

        ``now_ms`` is the clock reading the *caller* took (T-0053 criterion 4): this module has no
        clock. Without it every recalled line says its age is unknown, which is honest; it never
        says "just now" (criterion 5). Everything added here is in the query-dependent suffix,
        after the pixels, so the cached prefix is byte-identical either way."""
        messages = self._extend(prefix, transcript, tutor, memory_hits, frame_text, now_ms)
        return Prompt(
            messages=tuple(messages),
            prefix_len=prefix.prefix_len,
            thinking=prefix.thinking,
            assists=prefix.assists,
            tools=self.tools,
        )

    def fit(
        self,
        *,
        history: Sequence[HistoryTurn],
        frame: Frame | None = None,
        assists: AssistFlags = _NO_ASSISTS,
        transcript: str,
        tutor: TutorState | None = None,
        memory_hits: Sequence[MemoryHit] = (),
        frame_text: Sequence[FrameText] = (),
        resolve: Callable[[str], bytes | None] | None = None,
        frames: Sequence[Frame] = (),
        measure: Callable[[Prompt], int],
        drop_history_turns: int = 0,
        max_rungs: int | None = None,
        now_ms: int | None = None,
    ) -> FitResult:
        """The full prompt, trimmed until ``measure`` says it fits (T-0020, rulings A1/A2).

        ``measure`` is the **only** source of a token count: the engine's own
        ``usage.prompt_tokens``, or something a contract test proves agrees with it. This module
        counts nothing (B10), so a caller with no number from the engine has nothing to fit
        against and should not call this.

        The ladder is ruling A1's, each rung applied only while the measured count is over
        ``budget.max_prompt_tokens`` — B2's literal order left its "age the images" step
        unreachable, because dropping whole turns first leaves no history image to age:

        1. history *pixels*, oldest first, each falling back to its caption line (B3: a frame
           degrades, it never vanishes) — the turn itself stays;
        2. memory hits, lowest-ranked first;
        3. whole history turns, oldest first.

        The system message, the current user turn (transcript included) and **every** current
        frame are never spent (T-0059 D6): a prompt that cannot be made to fit comes back with
        ``fits=False`` rather than with the user's question truncated, and no rung buys tokens
        with a screen the question is about. When a screen has not changed, its pixels ride on the
        newest history image rather than being attached twice (D9/B9), so the *last* image
        carrying that hash is part of the floor — the older copies of it are ordinary history
        pixels and are spent first, oldest to newest.

        ``drop_history_turns`` drops that many oldest turns *before* any measuring: it is the
        session's retry lever after the engine rejects a prompt it had reported as fitting, so
        the retry is strictly smaller even when the count says everything is fine.

        ``now_ms`` is passed through to ``full()`` unchanged: it dates the recollections and is
        not part of any rung (T-0053 criterion 4).

        ``max_rungs`` stops the ladder after that many rungs even if the count is still over
        budget; the result then carries ``fits=False`` and the caller decides. It exists because
        a caller may have only *one* number from the engine (the prefill counts the prefix, and
        ``PrefillHandle`` carries no per-image or per-message cost), so re-measuring after a rung
        returns the same figure and an unbounded ladder would spend the whole history to cover a
        one-token overshoot (ruling A6). With a bound, each attempt gives up one rung and lets
        the engine — which is the authority — say whether that was enough.
        """
        current = _normalise_frames(frame, frames)
        current_phashes = {f.phash for f in current}
        kept = (
            list(history[-self.budget.max_history_turns :]) if self.budget.max_history_turns else []
        )
        dropped_turns = min(drop_history_turns, len(kept))
        turns = kept[dropped_turns:]
        hits = list(memory_hits)
        spent: set[int] = set()  # positions in ``turns`` whose pixels are spent
        dropped_images = 0
        dropped_hits = 0

        def build() -> Prompt:
            prefix = self.prefix(
                turns, None, assists, resolve, spent_images=frozenset(spent), frames=current
            )
            return self.full(prefix, transcript, tutor, hits, frame_text, now_ms=now_ms)

        def counts() -> dict[str, int]:
            return {
                "history_turns": dropped_turns,
                "images": dropped_images,
                "memory_hits": dropped_hits,
            }

        prompt = build()
        tokens = measure(prompt)
        rungs = 0
        # Bounded by construction: every rung removes one element that can never come back, so
        # the loop runs at most (history images + memory hits + history turns) times, and fewer
        # when ``max_rungs`` says so.
        while tokens > self.budget.max_prompt_tokens:
            if max_rungs is not None and rungs >= max_rungs:
                # Out of rungs, not out of prompt: what is left is still over the measured budget.
                return FitResult(prompt=prompt, tokens=tokens, fits=False, trimmed=counts())
            oldest = _oldest_history_image(prompt, current_phashes)
            if oldest is not None:
                before = _image_count(prompt)
                spent.add(oldest)
                prompt = build()
                dropped_images += max(0, before - _image_count(prompt))
            elif hits:
                del hits[_lowest_ranked(hits)]
                dropped_hits += 1
                prompt = build()
            elif turns:
                turns = turns[1:]
                # Positions shift with the window; a turn that is gone cannot still be "spent".
                spent = {index - 1 for index in spent if index >= 1}
                dropped_turns += 1
                prompt = build()
            else:
                # The floor, and it is still over: say so instead of eating the user's turn.
                return FitResult(prompt=prompt, tokens=tokens, fits=False, trimmed=counts())
            rungs += 1
            tokens = measure(prompt)
        return FitResult(prompt=prompt, tokens=tokens, fits=True, trimmed=counts())

    def system_message(self) -> Message:
        """Identical whether thinking is on or off: the flag rides on the chat template kwarg
        (docs/01 D11), so both sessions share every cached block of the system turn.

        ``UI_INSTRUCTIONS`` is appended last (T-0044): the shown channel needs the fence, the
        example and the component names, and it belongs *here* — in the cached prefix — rather
        than in the query-dependent suffix, where it would be re-sent and re-attended on every
        turn. It is a literal (never assembled from a set), so these bytes are the same in every
        process and every turn; ``fit()`` never spends it, because the system message is the
        floor of the ladder and not a rung.
        """
        ui_block = UI_INSTRUCTIONS if self.ui_block is None else self.ui_block
        parts = [self.system_prompt, CLOSE_THE_THOUGHT, SPOKEN_STYLE, ui_block]
        if self.tools is not None:
            # T-0092b criterion 3: last, so tools off is today's text byte for byte and tools on extends it.
            parts.append(NATIVE_TOOL_INSTRUCTIONS)
            if self.tool_guidance is not None:
                # T-0131 C1: after the native instructions, last; absent, the text is BASE's byte for byte.
                parts.append(self.tool_guidance)
        instructions = (self.user_instructions or "").strip()
        if instructions:
            # T-0213 A1-4: the user's block is the last part, after everything the A/Bs measured.
            parts.append(USER_INSTRUCTIONS_HEADER + "\n" + instructions)
        text = "\n".join(parts)
        return Message(role="system", content=(TextPart(text=text),))

    # -- internals --------------------------------------------------------------------------

    def _image_turns(self, kept: Sequence[HistoryTurn], current_images: int) -> set[int]:
        """Indices into ``kept`` whose frame stays an image: young enough, and inside
        ``max_images`` once the current turn's images have taken their slots (they always win).
        The rest fall back to their caption line (amendment B3) — a frame degrades to text, it
        never vanishes. Counted per turn, not per hash: two turns that happen to share a phash
        are two images on the wire (a chat-completions request carries no server-side state).

        ``current_images`` is the number of images this turn attaches, **not** the number of
        frames it was handed (T-0059 criterion 11): a frame deduplicated against a history image
        costs no slot. The newest history image is kept first, so a turn that gathers a further
        frame evicts the *oldest* history image (ruling A5, and the one place where a re-issue
        legitimately misses the cache from the evicted image onwards).
        """
        slots = self.budget.max_images - current_images
        chosen: set[int] = set()
        for age, index in enumerate(reversed(range(len(kept))), start=1):
            if len(chosen) >= slots:
                break
            if kept[index].frame_phash is not None and age <= self.budget.image_age_turns:
                chosen.add(index)
        return chosen

    def _current_frames(
        self, frames: Sequence[Frame], newest_attached: str | None
    ) -> tuple[Frame, ...]:
        """The frames this turn attaches as pixels, oldest first: one per distinct phash, never a
        screen whose pixels the newest history image already carries, at most
        ``max_frames_per_turn`` of them. First occurrence wins and nothing is reordered (D1).
        """
        chosen: list[Frame] = []
        seen: set[str] = set()
        for frame in frames:
            if len(chosen) >= self.budget.max_frames_per_turn:
                break
            if frame.phash == newest_attached or frame.phash in seen:
                continue
            seen.add(frame.phash)
            chosen.append(frame)
        return tuple(chosen)

    @staticmethod
    def _newest_attached(
        kept: Sequence[HistoryTurn],
        as_image: set[int],
        pixels: Callable[[str], bytes | None],
    ) -> str | None:
        """phash of the newest history frame that really goes on the wire as pixels — the one a
        current frame of the same screen rides on. A frame the resolver cannot produce is not in
        image form, so it is skipped here exactly as ``_assemble`` skips it.
        """
        for index in reversed(range(len(kept))):
            phash = kept[index].frame_phash
            if index in as_image and phash is not None and pixels(phash) is not None:
                return phash
        return None

    def _assemble(
        self,
        kept: Sequence[HistoryTurn],
        frames: Sequence[Frame],
        resolve: Callable[[str], bytes | None] | None,
        spent_images: frozenset[int] = frozenset(),
    ) -> list[Message]:
        cache: dict[str, bytes | None] = {}

        def pixels(phash: str) -> bytes | None:
            """``resolve`` memoised for the length of one assembly, so asking twice which history
            frames have pixels costs one call each. Determinism across *calls* stays the caller's
            duty (see :meth:`prefix`)."""
            if phash not in cache:
                cache[phash] = resolve(phash) if resolve is not None else None
            return cache[phash]

        # The two caps depend on each other: the number of current images decides how many of the
        # ``max_images`` slots history keeps, and the newest history image on the wire decides
        # whether the turn's first frame is attached at all (D9/B9). Settle it by iteration — a
        # frame that rides a history image frees a slot, and a freed slot only ever admits an
        # *older* history image, so the newest attached hash can move at most once and the fixed
        # point is reached in at most two rounds. Bounded by construction, and the slot set is
        # recomputed from the settled frames afterwards so the two can never disagree.
        attached = self._current_frames(frames, None)
        for _ in range(len(frames) + 1):
            slots = self._image_turns(kept, len(attached)) - spent_images
            settled = self._current_frames(frames, self._newest_attached(kept, slots, pixels))
            if settled == attached:
                break
            attached = settled
        # Subtracting after the fact, not inside ``_image_turns``: a spent slot must not be
        # refilled by an older frame the cap had already excluded (T-0020 rung 1).
        as_image = self._image_turns(kept, len(attached)) - spent_images

        messages = [self.system_message()]
        for index, turn in enumerate(kept):
            parts: list[Part] = [TextPart(text=turn.user_text)]
            if turn.frame_phash is not None:
                jpeg = pixels(turn.frame_phash) if index in as_image else None
                if jpeg is not None:
                    parts.append(ImagePart(url=_data_url(jpeg), phash=turn.frame_phash))
                elif turn.frame_caption:
                    # Aged out, over the image cap, or unresolved: the frame degrades to text
                    # (amendments B3/B9) rather than travelling as a reference nothing can fetch.
                    parts.append(_caption_part(turn.frame_caption))
            messages.append(Message(role="user", content=tuple(parts)))
            messages.append(
                Message(role="assistant", content=(TextPart(text=turn.assistant_spoken),))
            )
        current: list[Part] = []
        for position, frame in enumerate(attached):
            # The label belongs to the prefix, immediately before its own image: a suffix label
            # would sit after the pixels ``full()`` has already been extended past, and the first
            # re-issue would stop being an extension of what was prefilled (D1). Emitted only
            # when the turn may carry more than one frame, and never for the first image
            # (``frame_label(0) == ""``), which is what keeps a one-frame turn today's prompt.
            label = frame_label(position) if self.budget.max_frames_per_turn > 1 else ""
            if label:
                current.append(TextPart(text=label))
            current.append(ImagePart(url=_data_url(frame.jpeg), phash=frame.phash))
        messages.append(Message(role="user", content=tuple(current)))
        return messages

    def _extend(
        self,
        prefix: Prompt,
        transcript: str,
        tutor: TutorState | None,
        hits: Sequence[MemoryHit],
        frame_text: Sequence[FrameText],
        now_ms: int | None = None,
    ) -> list[Message]:
        blocks: list[str] = []
        if _image_count(prefix) == 0:
            # Derived from the assembled prompt, not from ``frame is not None`` (criterion 2):
            # only the prompt knows whether any pixels are on the wire. Part of the current turn,
            # so ``fit()`` never spends it (criterion 8).
            blocks.append(NO_IMAGE_STATEMENT)
        if tutor is not None and tutor.as_prompt_block():
            blocks.append(tutor.as_prompt_block())
        memory = _memory_block(hits, now_ms)
        if memory:
            blocks.append(memory)
        blocks.extend(_assist_blocks(frame_text, prefix.assists))
        suffix: list[Part] = [TextPart(text=block + "\n") for block in blocks]
        suffix.append(TextPart(text=transcript))
        last = prefix.messages[-1]
        messages = list(prefix.messages[:-1])
        messages.append(Message(role=last.role, content=(*last.content, *suffix)))
        return messages
