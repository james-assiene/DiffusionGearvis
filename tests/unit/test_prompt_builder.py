"""Prompt layout D9/D11 and close-the-thought rule (docs/01, docs/11 §2). Task T-0005."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from local_astra.conductor import prompt_builder as prompt_builder_module
from local_astra.conductor.prompt_builder import (
    CLOSE_THE_THOUGHT,
    SPOKEN_STYLE,
    Budget,
    PromptBuilder,
)
from local_astra.contracts.llm import ImagePart, TextPart
from local_astra.contracts.memory import Episode, MemoryHit
from local_astra.contracts.perception import AssistFlags, Frame, FrameText, Source
from local_astra.contracts.turn import HistoryTurn, TutorState

# --- active: the rule text exists and says the right thing --------------------------------------------


def test_close_the_thought_instruction_names_both_halves_of_the_rule():
    assert "close" in CLOSE_THE_THOUGHT.lower() and "thinking channel" in CLOSE_THE_THOUGHT.lower()
    assert "never leave the answer inside the thinking channel" in CLOSE_THE_THOUGHT.lower()


def test_spoken_style_forbids_markdown_and_lists():
    assert "no markdown" in SPOKEN_STYLE.lower() and "no lists" in SPOKEN_STYLE.lower()


REPO = Path(__file__).resolve().parents[2]
VLLM_SH = REPO / "services" / "vllm.sh"


def _serving_flags() -> str:
    """``services/vllm.sh`` with whole-line comments stripped.

    The comments in that file quote *superseded* values (a ``--max-model-len 16384`` correction
    block that cites 32768 and 8192 among them), so a plain grep would happily read history.
    """
    return "\n".join(
        line for line in VLLM_SH.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def _max_model_len() -> int:
    """The context window vLLM is actually started with (T-0014 set it to 16384)."""
    match = re.search(r"--max-model-len\s+(\d+)", _serving_flags())
    assert match is not None, f"no --max-model-len in the executable part of {VLLM_SH}"
    return int(match.group(1))


def _mm_image_limit() -> int:
    """``--limit-mm-per-prompt`` image count; a prompt carrying one more image is a hard 400."""
    match = re.search(r"--limit-mm-per-prompt\s+'(?P<body>.*?)'", _serving_flags())
    assert match is not None, f"no --limit-mm-per-prompt in the executable part of {VLLM_SH}"
    payload = json.loads(match.group("body").replace('\\"', '"'))
    return int(payload["image"])


def test_budget_defaults_match_the_serving_flags():
    """Amendment B10: read the flags, do not restate them.

    The previous version of this test asserted ``max_prompt_tokens < 32768`` against a comment
    saying ``# --max-model-len`` — a window T-0014 superseded with 16384 — so it passed while
    being false about the thing its name claims to check. The numbers now come from
    ``services/vllm.sh``, which means a flag change breaks the test that depends on it.
    """
    b = Budget()
    assert b.max_images == _mm_image_limit(), (
        "the image cap is the engine's --limit-mm-per-prompt image limit: one image over it is "
        "an HTTP 400, not a degraded prompt"
    )
    assert b.image_age_turns == 2 and b.max_history_turns == 3


# T-0020 criterion 1 owns this assertion (T-0005 amendment B10 opened it as a strict xfail; it
# became T-0020's pending test, and the marker came off when /implement T-0020 started). Ruling
# A5 adds the answer's headroom to the bound, which is asserted in
# tests/unit/test_prompt_budget.py::test_the_default_budget_is_strictly_inside_the_serving_window.
def test_the_declared_prompt_budget_fits_inside_the_serving_window():
    """A budget larger than the window is not a budget: every prompt it would admit is rejected
    by the engine before any trimming could help."""
    assert Budget().max_prompt_tokens <= _max_model_len()


def test_no_estimated_token_count_trims_the_prompt():
    """Amendment B10: the structural caps are the *only* size control in this module.

    The deleted estimator guessed 4 characters per token and 1120 tokens per image, then dropped
    history turns and images until its own guess fit. Reinstating anything like it would drop
    context on a prompt this module cannot measure, so this feeds a prompt hundreds of times
    over ``max_prompt_tokens`` worth of characters and asserts nothing is dropped: the system
    message, every kept history turn, the current frame and the transcript all survive.
    Amendment B2's floor (never drop the system message, the current user turn or the current
    frame) holds a fortiori.
    """
    assert not hasattr(prompt_builder_module, "estimate_tokens"), (
        "estimate_tokens is deleted by B10: an unmeasured 4-chars-per-token guess that trims "
        "context is worse than no budget at all"
    )
    pb = PromptBuilder("sys")
    huge = "lorem ipsum " * 20_000  # ~240k characters per turn, 3 turns kept
    hist = tuple(
        HistoryTurn(
            user_text=f"q{i} {huge}",
            assistant_spoken=f"a{i} {huge}",
            frame_phash=f"{i:016x}",
            frame_caption=f"cap{i}",
        )
        for i in range(3)
    )
    frame = _frame(9000, "f" * 16)
    p = pb.prefix(hist, frame, AssistFlags(), resolve=_resolver())
    full = pb.full(p, "and the total? " + huge, None, [])
    assert [m.role for m in p.messages] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert CLOSE_THE_THOUGHT in p.messages[0].text(), "the system message is never dropped"
    text = "".join(m.text() for m in p.messages)
    assert all(f"q{i} " in text for i in range(3)), "no history turn was trimmed by a guess"
    images = [part for m in p.messages for part in m.content if isinstance(part, ImagePart)]
    assert images and images[-1].phash == "f" * 16, "the current frame is never dropped"
    assert "and the total?" in "".join(m.text() for m in full.messages)


# --- T-0005 (activated for /implement; formerly pending) ------------------------------------------


def _frame(ts: int, phash: str = "a" * 16) -> Frame:
    return Frame(
        source=Source.SCREEN, ts_ms=ts, width=1280, height=800, jpeg=b"\xff\xd8\xff", phash=phash
    )


#: JPEG bytes a resolver hands back for a *history* frame; deliberately different from the
#: current frame's bytes so the two can never be confused in an assertion.
HISTORY_JPEG = b"\xff\xd8\xff\xe0history"


def _resolver(known: dict[str, bytes] | None = None):
    """A history-frame resolver of the kind T-0007/T-0009 will inject (amendment B9).

    ``prefix(..., resolve=...)`` is how pixels for past turns reach the prompt without this
    module doing I/O. With ``known`` it answers only for those hashes (``None`` elsewhere),
    which is how a ring buffer that has already evicted a frame behaves.
    """
    if known is None:
        return lambda phash: HISTORY_JPEG
    return lambda phash: known.get(phash)


def _images(prompt):
    return [part for m in prompt.messages for part in m.content if isinstance(part, ImagePart)]


def test_system_message_contains_the_rules_and_no_literal_think_token():
    """docs/01 D11: the chat template emits <|think|> from a session-level kwarg, not the text."""
    pb = PromptBuilder("You are Astra.", thinking=True)
    sysm = pb.system_message()
    assert sysm.role == "system"
    assert "<|think|>" not in sysm.text()
    assert CLOSE_THE_THOUGHT in sysm.text() and SPOKEN_STYLE in sysm.text()
    p = pb.prefix((), _frame(1000), AssistFlags())
    assert p.thinking is True and p.chat_template_kwargs() == {"enable_thinking": True}
    assert PromptBuilder("x", thinking=False).prefix((), None, AssistFlags()).thinking is False


def test_prefix_is_system_history_then_one_user_turn_ending_with_the_frame():
    pb = PromptBuilder("sys")
    hist = (HistoryTurn(user_text="hello", assistant_spoken="hi"),)
    p = pb.prefix(hist, _frame(1000), AssistFlags(page_text=True))
    assert [m.role for m in p.messages] == ["system", "user", "assistant", "user"]
    last = p.messages[-1].content[-1]
    assert isinstance(last, ImagePart) and last.phash == "a" * 16
    assert p.prefix_len == len(p.messages)
    assert p.assists == AssistFlags(page_text=True)


def test_full_extends_the_last_user_message_in_order_state_memory_assists_transcript():
    pb = PromptBuilder("sys")
    pre = pb.prefix((), _frame(1000), AssistFlags(page_text=True))
    ep = Episode(
        ts_ms=10,
        source=Source.SCREEN,
        thumbnail_path="t.jpg",
        phash="b" * 16,
        caption="budget totals",
    )
    full = pb.full(
        pre,
        transcript="what is the total",
        tutor=TutorState(problem="sum the column", current_step=1, total_steps=2),
        memory_hits=[MemoryHit(episode=ep, score=1.0)],
        frame_text=[
            FrameText(source=Source.SCREEN, ts_ms=1000, kind="page_text", text="A1 12 A2 30")
        ],
    )
    assert pre.shares_prefix_with(full), "D9: the final request extends the prefill prefix"
    assert full.prefix_len == pre.prefix_len
    assert len(full.messages) == len(pre.messages), "no second consecutive user message"
    tail = "".join(
        p.text
        for p in full.messages[-1].content[len(pre.messages[-1].content) :]
        if isinstance(p, TextPart)
    )
    i_state = tail.index("[tutor state]")
    i_mem = tail.index("budget totals")
    i_assist = tail.index("A1 12")
    i_q = tail.index("what is the total")
    assert i_state < i_mem < i_assist < i_q
    assert tail.rstrip().endswith("what is the total")


def test_assist_text_is_dropped_when_its_flag_is_off():
    pb = PromptBuilder("sys")
    pre = pb.prefix((), _frame(1000), AssistFlags())  # nothing on
    full = pb.full(
        pre,
        "q",
        None,
        [],
        frame_text=[
            FrameText(source=Source.SCREEN, ts_ms=1000, kind="page_text", text="SECRET DOM")
        ],
    )
    assert "SECRET DOM" not in "".join(m.text() for m in full.messages)
    assert full.assists == AssistFlags()


def _aging_history():
    """Four turns, each with a frame and a caption; the builder keeps the last three."""
    return tuple(
        HistoryTurn(
            user_text=f"q{i}",
            assistant_spoken=f"a{i}",
            frame_phash=f"{i:016x}",
            frame_caption=f"cap{i}",
        )
        for i in range(4)
    )


def test_history_frames_age_into_captions_and_images_are_capped():
    """Criterion 5 with pixels, i.e. with the resolver amendment B9 added.

    Hand-computed with ``Budget(max_images=2, image_age_turns=2, max_history_turns=3)`` and four
    history turns: turns 1-3 are kept (q0 is over the history cap); the current frame takes one
    of the two image slots, leaving one for history; turn 1 is three turns old so it is past
    ``image_age_turns`` and becomes ``cap1``; of the two young turns the newest wins the single
    slot, so turn 3 stays pixels and turn 2 falls back to ``cap2`` (amendment B3: a frame
    degrades to text, it never vanishes).
    """
    pb = PromptBuilder("sys", Budget(max_images=2, image_age_turns=2, max_history_turns=3))
    p = pb.prefix(_aging_history(), _frame(9000, "f" * 16), AssistFlags(), resolve=_resolver())
    text = "".join(m.text() for m in p.messages)
    assert "q0" not in text, "history capped at three turns"
    assert [i.phash for i in _images(p)] == [f"{3:016x}", "f" * 16], "current frame always wins"
    assert "cap1" in text, "the oldest kept turn's frame aged into its caption"
    assert "cap2" in text, "B3: a frame dropped for the image cap falls back to its caption"
    assert "cap3" not in text, "the newest history frame is still an image, not a caption"


def test_without_a_resolver_every_history_frame_is_a_caption_line():
    """The mirror of the test above: B9's *default*.

    With no resolver the builder has no pixels for a past turn, so every history frame — young,
    aged or capped — degrades to its caption line and the only image is the current frame. This
    pins the default rather than merely permitting it: a builder that emitted an
    ``astra-frame:``-style reference here would serialise into a request the client rejects
    (that is H1), and one that emitted no text at all would make the past turns unreadable.
    """
    pb = PromptBuilder("sys", Budget(max_images=2, image_age_turns=2, max_history_turns=3))
    p = pb.prefix(_aging_history(), _frame(9000, "f" * 16), AssistFlags())
    text = "".join(m.text() for m in p.messages)
    assert [i.phash for i in _images(p)] == ["f" * 16], "only the current frame carries pixels"
    assert all(f"cap{i}" in text for i in (1, 2, 3)), "every kept history frame became a caption"
    assert all(i.url.startswith("data:") for i in _images(p)), "no reference URL is ever emitted"


def test_unchanged_frame_is_not_attached_twice():
    """Criterion 6: the *resolved* history image already carries the pixels, so an identical
    current frame is not sent a second time; a changed one is."""
    pb = PromptBuilder("sys")
    hist = (HistoryTurn(user_text="q", assistant_spoken="a", frame_phash="c" * 16),)
    p = pb.prefix(hist, _frame(2000, "c" * 16), AssistFlags(), resolve=_resolver())
    assert [i.phash for i in _images(p)] == ["c" * 16], "one copy of the unchanged screen"
    p2 = pb.prefix(hist, _frame(2000, "d" * 16), AssistFlags(), resolve=_resolver())
    assert [i.phash for i in _images(p2)] == ["c" * 16, "d" * 16], "a changed screen is sent"


def test_without_a_resolver_an_unchanged_frame_is_still_attached_once():
    """The mirror of criterion 6 under B9's default, and the failure mode it guards.

    Without a resolver the history turn holds no pixels at all, so "the cached image already
    carries them" is false and the current frame must go on the wire — otherwise the prompt
    would contain *no* picture of the screen the user is asking about. (This history turn has no
    caption either, so there is nothing to degrade to: the frame is simply absent from the past
    turn, and the present one must not be suppressed on account of it.)
    """
    pb = PromptBuilder("sys")
    hist = (HistoryTurn(user_text="q", assistant_spoken="a", frame_phash="c" * 16),)
    p = pb.prefix(hist, _frame(2000, "c" * 16), AssistFlags())
    assert [i.phash for i in _images(p)] == ["c" * 16]
    history_messages = p.messages[1:-1]
    assert not [
        part for m in history_messages for part in m.content if isinstance(part, ImagePart)
    ], "no ImagePart for a history frame without a resolver"
    assert isinstance(p.messages[-1].content[-1], ImagePart), "the image is the *current* turn's"


def test_a_resolver_that_cannot_produce_the_pixels_does_not_blind_the_prompt():
    """Adversarial, B9's named failure mode: a resolver is not a promise.

    The ring buffer may have evicted the frame, so ``resolve`` returns ``None``. That turn must
    fall back to its caption line (never to a reference the client would reject), and — because
    its pixels are *not* in the context — an identical current frame must still be attached. The
    tempting implementation, memoising "the newest history phash" before checking whether the
    bytes actually arrived, produces a prompt with no image of the current screen.
    """
    pb = PromptBuilder("sys")
    hist = (
        HistoryTurn(
            user_text="q0", assistant_spoken="a0", frame_phash="c" * 16, frame_caption="the sheet"
        ),
    )
    p = pb.prefix(hist, _frame(2000, "c" * 16), AssistFlags(), resolve=_resolver(known={}))
    assert [i.phash for i in _images(p)] == ["c" * 16]
    assert isinstance(p.messages[-1].content[-1], ImagePart), "the current screen is visible"
    assert "the sheet" in "".join(m.text() for m in p.messages), "the evicted frame kept its text"


# --- T-0005 extensions written 2026-09-13 (criteria 4-6; see the test-author report) -------------


def test_frame_is_attached_as_a_base64_jpeg_data_url_carrying_its_phash():
    """``ImagePart`` contract: ``data:image/jpeg;base64,…`` for local frames. The bytes must be
    the frame's own JPEG — a path or a placeholder would make the model blind."""
    frame = _frame(1000)
    p = PromptBuilder("sys").prefix((), frame, AssistFlags())
    img = p.messages[-1].content[-1]
    assert isinstance(img, ImagePart)
    assert img.url == "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode()
    assert img.phash == frame.phash


def test_prefix_is_deterministic_so_the_prefill_and_the_final_request_share_it():
    """D9 failure mode: any per-call state in the builder (a counter, a "already sent" memo, a
    timestamp) makes the final request's prefix differ from the prefilled one and every cached
    block miss. Same inputs -> byte-identical prompt."""
    pb = PromptBuilder("sys")
    hist = (HistoryTurn(user_text="hello", assistant_spoken="hi", frame_phash="e" * 16),)
    first = pb.prefix(hist, _frame(1000), AssistFlags(page_text=True))
    second = pb.prefix(hist, _frame(1000), AssistFlags(page_text=True))
    assert first == second
    assert PromptBuilder("sys").prefix(hist, _frame(1000), AssistFlags(page_text=True)) == first


def test_full_does_not_mutate_the_prefix_and_is_repeatable():
    pb = PromptBuilder("sys")
    pre = pb.prefix((), _frame(1000), AssistFlags())
    before = pre.model_dump()
    kw = {"tutor": None, "memory_hits": [], "frame_text": ()}
    one = pb.full(pre, "q", **kw)
    two = pb.full(pre, "q", **kw)
    assert pre.model_dump() == before, "the prefill prefix must survive full() untouched"
    assert one == two
    assert pre.shares_prefix_with(one)


def test_only_the_assist_kinds_whose_flag_is_on_reach_the_prompt():
    """Pixel-first (D12a): the flags are per *kind*, not a single on/off. A cells dump with the
    cells box unticked is a silent breach of the demo's honesty claim."""
    pb = PromptBuilder("sys")
    pre = pb.prefix((), _frame(1000), AssistFlags(page_text=True))
    full = pb.full(
        pre,
        "what is the total",
        None,
        [],
        frame_text=[
            FrameText(source=Source.SCREEN, ts_ms=1000, kind="page_text", text="A1 12 A2 30"),
            FrameText(source=Source.SCREEN, ts_ms=1000, kind="cells", text="SECRET CELLS"),
            FrameText(source=Source.SCREEN, ts_ms=1000, kind="at_spi", text="SECRET A11Y"),
        ],
    )
    text = "".join(m.text() for m in full.messages)
    assert "A1 12 A2 30" in text
    assert "SECRET CELLS" not in text and "SECRET A11Y" not in text
    assert full.assists == AssistFlags(page_text=True), "the turn log needs the flags that were on"


def test_a_frame_matching_an_aged_out_history_frame_is_attached_again():
    """Criterion 6 in full: the rule is "equals the newest history frame *still in image form*".
    Once that turn has aged into a caption the pixels are gone from the context, so an identical
    current frame must be sent — otherwise the model answers about a screen it can no longer see.
    """
    pb = PromptBuilder("sys", Budget(max_images=4, image_age_turns=1, max_history_turns=3))
    hist = (
        HistoryTurn(
            user_text="q0", assistant_spoken="a0", frame_phash="c" * 16, frame_caption="old shot"
        ),
        HistoryTurn(
            user_text="q1", assistant_spoken="a1", frame_phash="e" * 16, frame_caption="new shot"
        ),
    )
    p = pb.prefix(hist, _frame(3000, "c" * 16), AssistFlags(), resolve=_resolver())
    assert [i.phash for i in _images(p)] == ["e" * 16, "c" * 16]
    assert "old shot" in "".join(m.text() for m in p.messages), "the aged turn keeps its caption"


def test_without_a_resolver_the_same_scenario_is_two_captions_and_the_current_frame():
    """The mirror: with no pixels for either past turn, both become caption lines and the
    current frame is attached even though its phash matches turn 0 (B9 default + criterion 6)."""
    pb = PromptBuilder("sys", Budget(max_images=4, image_age_turns=1, max_history_turns=3))
    hist = (
        HistoryTurn(
            user_text="q0", assistant_spoken="a0", frame_phash="c" * 16, frame_caption="old shot"
        ),
        HistoryTurn(
            user_text="q1", assistant_spoken="a1", frame_phash="e" * 16, frame_caption="new shot"
        ),
    )
    p = pb.prefix(hist, _frame(3000, "c" * 16), AssistFlags())
    assert [i.phash for i in _images(p)] == ["c" * 16], "only the current frame is pixels"
    text = "".join(m.text() for m in p.messages)
    assert "old shot" in text and "new shot" in text, "both history frames became captions"


def test_the_system_message_is_identical_whether_thinking_is_on_or_off():
    """docs/01 D11 + CLAUDE.md: thinking rides on the template kwarg. If it leaked into the system
    text, a session with thinking off would share no prefix blocks with one that has it on."""
    on, off = PromptBuilder("sys", thinking=True), PromptBuilder("sys", thinking=False)
    assert on.system_message() == off.system_message()
    # and the same one message down: the whole cached prefix must be flag-independent
    assert (
        on.prefix((), _frame(1000), AssistFlags()).messages
        == off.prefix((), _frame(1000), AssistFlags()).messages
    )
