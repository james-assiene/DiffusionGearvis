"""The prompt the session builds (T-0007 criteria 9 and 10).

Layout is ``prompt_builder.py``'s and is not touched here (docs/01 D9): what this task owns is
*when* it is built (once, at ``llm.start``, from the frame captured at onset), *what* it is built
from (the frozen assists, the history-frame resolver of T-0005 B9) and that the prefill and the
final request share a byte-identical prefix — otherwise every cached block after the first
difference misses. The API is fixed in ``test_session_loop.py``.
"""

from __future__ import annotations

import pytest

from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.contracts import commands as C
from local_astra.contracts.llm import ImagePart, Prompt
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.protocol import AstraAssistToggle, AstraFrameText
from local_astra.contracts.turn import TurnState
from tests.unit.test_session_loop import answer, frame, running


class BuilderSpy:
    """Records every ``PromptBuilder.prefix``/``full`` call of the session's own builder.

    Criterion 9's "built **once**" is a statement about the *builder*, not about the two prompts:
    two calls that happen to return equal objects still re-resolve every history image and re-read
    the frame, and the equality that hides it is exactly what a unit test of the builder cannot
    see. So the pin is structural (ruling A10): one ``prefix`` call per turn, and the object handed
    to ``full`` is the object the prefill was given.
    """

    def __init__(self) -> None:
        self.prefixes: list[Prompt] = []
        self.full_args: list[Prompt] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_prefix, real_full = PromptBuilder.prefix, PromptBuilder.full

        def prefix(builder, *args, **kwargs):
            out = real_prefix(builder, *args, **kwargs)
            self.prefixes.append(out)
            return out

        def full(builder, prefix_arg, *args, **kwargs):
            self.full_args.append(prefix_arg)
            return real_full(builder, prefix_arg, *args, **kwargs)

        monkeypatch.setattr(PromptBuilder, "prefix", prefix)
        monkeypatch.setattr(PromptBuilder, "full", full)


async def test_the_prefill_prefix_is_the_final_requests_prefix(tmp_path, monkeypatch):
    """Criterion 9: built **once**, passed to ``prefill``, extended with ``full()`` at ``llm.start``.

    The assertion is on the objects that actually reached the backend, not on the builder:
    ``prefix.shares_prefix_with(full)`` on the two prompts the session sent. Ruling A8: equality
    with ``full.prefix()`` is unsatisfiable by design — ``full()`` extends the last user message of
    the prefix in place and leaves ``prefix_len`` alone, so the reconstructed prefix carries the
    query-dependent suffix and can never equal what the prefill was given.

    "Built once" is pinned structurally instead (ruling A10): one ``PromptBuilder.prefix`` call
    for the turn, and the object ``full()`` extended is the object ``prefill`` received. Rebuilding
    between the two (a re-read frame, a re-resolved history image) is invisible in a unit test of
    the builder and costs the whole prefix cache here.
    """
    spy = BuilderSpy()
    spy.install(monkeypatch)
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert len(rig.llm.prefills) == 1, f"one prefill per turn; got {rig.llm.prefills}"
        _, prefix = rig.llm.prefills[0]
        _, full = rig.llm.gen_prompts[0]
        assert prefix.shares_prefix_with(full), (
            "criterion 9: prefix.shares_prefix_with(full) must hold on the objects actually sent"
        )
        assert len(spy.prefixes) == 1, (
            "criterion 9: the prefix is built **once** per turn; PromptBuilder.prefix was called "
            f"{len(spy.prefixes)} times"
        )
        assert len(spy.full_args) == 1 and spy.full_args[0] is prefix, (
            "criterion 9, negative: the prefix is not rebuilt between prefill and generate — the "
            "prompt handed to full() must be the very object the prefill was given, not an equal "
            f"copy (full() got {spy.full_args})"
        )
        assert prefix.thinking is full.thinking is True, (
            "thinking is a session setting, never per request: <|think|> is the first system "
            "token and toggling it misses the whole prefix cache (CLAUDE.md)"
        )
        assert full.messages[-1].text().endswith("what is on this screen"), (
            "the transcript is the last thing in the query-dependent suffix (docs/01 D9)"
        )


async def test_frame_comes_from_onset_not_latest(tmp_path):
    """Criterion 9: ``SnapshotFrame(at_ms)`` resolves via ``get_frame(source, at_ms=onset_ms)``.

    A newer frame arrives while the user is still speaking. The answer must be about the screen
    the user was looking at when they started talking, so the newer frame may not replace it —
    ``get_frame(source)`` with no ``at_ms`` (i.e. "latest") is the failure.
    """
    old, new = frame(0, "a"), frame(10_000, "b")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[old]) as rig:
        onset_ms = await rig.onset()
        rig.perception.add(new)
        await rig.words("what is this")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        snapshots = [c.at_ms for c in rig.machine.commands(C.SnapshotFrame)]
        assert snapshots and snapshots[0] == onset_ms, (
            f"the machine snapshots at the onset ({onset_ms}); got {snapshots}"
        )
        assert rig.perception.frame_calls, "SnapshotFrame must reach PerceptionClient.get_frame"
        assert all(at_ms is not None for _, at_ms in rig.perception.frame_calls), (
            "criterion 9: never `latest` — every get_frame carries at_ms "
            f"({rig.perception.frame_calls})"
        )
        assert (Source.SCREEN, onset_ms) in rig.perception.frame_calls, (
            f"expected get_frame(SCREEN, at_ms={onset_ms}); got {rig.perception.frame_calls}"
        )
        images = [
            part
            for message in rig.llm.gen_prompts[0][1].messages
            for part in message.content
            if isinstance(part, ImagePart)
        ]
        assert [part.phash for part in images] == [old.phash], (
            f"the turn's prompt carries the frame captured at the onset; got {images}"
        )


async def test_the_history_frames_resolver_answers_for_both_requests(tmp_path, monkeypatch):
    """Criterion 9: the T-0005 B9 ``resolve`` callback is supplied and answers identically.

    Without a resolver a history frame degrades to its caption line — ``prompt_builder`` never
    invents an image part it cannot fill. So "the second turn's prompt carries the first turn's
    frame as pixels" is the observable form of "a resolver was supplied", and comparing the
    prefill's prompt with the request's covers "answers identically".
    """
    spy = BuilderSpy()
    spy.install(monkeypatch)
    first = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[first]) as rig:
        await rig.ask("first question")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn one never ended")

        # Ruling A8: stamped at the clock, not in the future — ``get_frame(at_ms=onset)`` rightly
        # refuses a frame captured after the onset it is asked about, and a frame with a different
        # tag is what makes "both phashes are there" a real assertion.
        second = frame(rig.clock.now_ms, "b")
        assert second.phash != first.phash, "the two turns must be looking at different screens"
        rig.perception.add(second)
        await rig.advance(1_000, step=250)
        await rig.ask("second question")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn two never ended")

        _, prefix = rig.llm.prefills[1]
        _, full = rig.llm.gen_prompts[1]
        phashes = [
            part.phash
            for message in full.messages
            for part in message.content
            if isinstance(part, ImagePart)
        ]
        assert first.phash in phashes, (
            "criterion 9: the history frame of turn 1 must come back as pixels through the "
            f"resolve callback (T-0005 B9); the second prompt carries {phashes}"
        )
        assert second.phash in phashes, "and the current frame, which is a different screen"
        prefix_images = [
            part.phash
            for message in prefix.messages
            for part in message.content
            if isinstance(part, ImagePart)
        ]
        assert prefix_images == phashes, (
            "the resolver must answer identically for the prefill and the final request — the "
            "prefill saw images "
            f"{prefix_images} and the request {phashes}; every cached block after the first "
            "differing image misses"
        )
        assert prefix.shares_prefix_with(full), "and the two must still share their prefix"
        assert len(spy.prefixes) == 2 and spy.full_args[1] is prefix, (
            "criterion 9 / ruling A10: one PromptBuilder.prefix call per turn, and turn two's "
            f"generate extends the object its prefill was given ({len(spy.prefixes)} calls)"
        )


async def test_assists_are_frozen_at_turn_open(tmp_path):
    """Criterion 10: ``AssistState.freeze()`` at turn open; a mid-turn toggle changes nothing.

    Page text is on when the turn opens and off again before the answer commits. The turn's
    prompt must still carry ``page_text=True`` **and** the DOM text, and the logged regime must
    still say ``pixels+dom`` — a turn whose badge drifts from the prompt it was answered with is
    a measurement nobody can compare later (docs/01 D12a, T-0012).
    """
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        assert rig.session.assists.flags == AssistFlags(), (
            "criterion 10, negative: assists are off by default (docs/01 D12a)"
        )
        rig.session.assists.toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        rig.session.assists.accept_frame_text(
            AstraFrameText(
                source=Source.SCREEN, ts_ms=0, kind="page_text", text="DOM TEXT OF THE TAB"
            )
        )
        await rig.onset()
        await rig.words("what is on this screen")
        # the box is unticked mid-turn: the turn in flight keeps the flags it froze
        rig.session.assists.toggle(AstraAssistToggle(flags=AssistFlags()))
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        _, full = rig.llm.gen_prompts[0]
        assert full.assists == AssistFlags(page_text=True), (
            f"criterion 10: the frozen flags travel in the Prompt; got {full.assists}"
        )
        assert "DOM TEXT OF THE TAB" in full.messages[-1].text(), (
            "the assist payload of the frozen kinds goes into the query-dependent suffix"
        )
        record = rig.records()[-1]
        assert record.regime.assist_flags == AssistFlags(page_text=True)
        assert record.regime.assists == "pixels+dom", (
            "criterion 10: the badge comes from Regime.with_assists(prompt.assists), the only "
            f"sanctioned setter; got {record.regime.assists!r}"
        )

        rig.llm.script = [answer("No. ")]
        await rig.advance(500, step=250)
        await rig.ask("and now")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the second turn never ended")
        assert rig.llm.gen_prompts[1][1].assists == AssistFlags(), (
            "the untick takes effect from the next freeze"
        )
        assert rig.records()[-1].regime.assists == "pixels"


# --- T-0092b criterion 3: the tools-on prompt (H11) ------------------------------------------------
#
# Assumed names are fixed in tests/unit/_tools_switch_support.py's docstring: ``NATIVE_TOOL_INSTRUCTIONS`` and
# ``PromptBuilder(..., tools=...)`` in prompt_builder.py, the kit injected as ``Session(..., tool_kit=...)``.

import hashlib  # noqa: E402

from local_astra.contracts.llm import LlmOptions  # noqa: E402
from local_astra.contracts.memory import Episode, MemoryHit  # noqa: E402
from local_astra.contracts.turn import HistoryTurn  # noqa: E402
from tests.unit import _tools_switch_support as T  # noqa: E402

SYSTEM = "You are a test assistant."


def _history(n: int) -> list[HistoryTurn]:
    return [
        HistoryTurn(
            user_text=f"q{i}", assistant_spoken=f"a{i}", frame_phash=None, frame_caption=None
        )
        for i in range(n)
    ]


def _hit() -> MemoryHit:
    return MemoryHit(
        episode=Episode(
            ts_ms=0,
            source=Source.SCREEN,
            thumbnail_path="/x.jpg",
            phash="p0",
            caption="a past screen",
        ),
        score=1.0,
    )


def require_tools(prompt: Prompt, where: str) -> None:
    """The check every builder output must pass: the session's tools field, as a tuple, equal to criterion 3's."""
    assert prompt.tools is not None, (
        f"T-0092b criterion 3: {where} carries the tools field; it carries None"
    )
    assert tuple(prompt.tools) == T.tools_field(), (
        f"T-0092b criterion 3: {where} carries tuple(native.tools_field(tools.load())); it differs"
    )


def test_native_tool_instructions_are_paragraphs_2_and_3_of_the_eval_prompt():
    """Criterion 3 / H11, cd-free half: the literal's sha256 is that of twopass's paragraphs 2-3 joined by one
    blank line, pinned (``tests/cd/unit/test_native_tool_instructions_parity.py`` re-derives it from twopass)."""
    text = T.native_instructions()
    assert hashlib.sha256(text.encode()).hexdigest() == T.NATIVE_TOOL_INSTRUCTIONS_SHA256, (
        "T-0092b criterion 3: NATIVE_TOOL_INSTRUCTIONS is H-T84-0's paragraphs 2-3, verbatim"
    )
    assert "You are Astra" not in text, "H11: the persona sentence is omitted"
    assert text.count("\n\n") == 1, "two paragraphs"


def test_the_system_message_is_todays_text_followed_by_the_instructions():
    """Criterion 3: tools on = today's system text, then ``NATIVE_TOOL_INSTRUCTIONS`` last; tools off unchanged."""
    off = PromptBuilder(SYSTEM).system_message().text()
    on = T.builder(SYSTEM, tools=T.tools_field()).system_message().text()
    instructions = T.native_instructions()
    assert on.startswith(off), "the tools-on system text begins with today's, byte for byte"
    assert on.endswith(instructions), "…and NATIVE_TOOL_INSTRUCTIONS is appended last"
    assert on[len(off) : len(on) - len(instructions)].strip() == "", (
        "nothing but a separator between them"
    )
    assert instructions not in off, "tools off: the system message does not carry the instructions"
    assert T.builder(SYSTEM, tools=T.tools_field()).system_message().text() == on, (
        "the same bytes every time"
    )


def test_every_prompt_the_builder_returns_carries_the_tools():
    """Criterion 3: ``prefix``, ``full``, ``fit`` (a trimming fit) and the over-long retry all carry the tools."""
    pb = T.builder(SYSTEM, tools=T.tools_field())
    hist = _history(3)
    prefix = pb.prefix(hist, frame(0, "a"), AssistFlags())
    require_tools(prefix, "prefix()")
    full = pb.full(prefix, "what is this", None, [_hit()])
    require_tools(full, "full()")
    assert prefix.shares_prefix_with(full), "criterion 3: shares_prefix_with holds with tools on"
    over = pb.fit(
        history=hist,
        frame=frame(0, "a"),
        assists=AssistFlags(),
        transcript="what is this",
        memory_hits=[_hit()],
        measure=lambda p: pb.budget.max_prompt_tokens + 1,
        max_rungs=1,
    )
    require_tools(over.prompt, "fit() after a rung")
    retry = pb.fit(
        history=hist,
        frame=frame(0, "a"),
        assists=AssistFlags(),
        transcript="what is this",
        measure=lambda p: 1,
        drop_history_turns=1,
        max_rungs=0,
    )
    require_tools(retry.prompt, "fit(drop_history_turns=1), the over-long retry")
    assert len(retry.prompt.messages) == len(prefix.messages) - 2, (
        "premise: the retry dropped one history turn"
    )


def test_the_tools_check_catches_a_builder_that_drops_them_in_full():
    """Adversarial (task file): a builder whose ``full()`` rebuilds the Prompt without ``tools`` — today's code
    path — must fail ``require_tools``. Run against a subclass that does exactly that."""
    base = type(T.builder(SYSTEM, tools=T.tools_field()))

    class DropsInFull(base):  # type: ignore[misc, valid-type]
        def full(self, *a, **k):  # type: ignore[no-untyped-def]
            return super().full(*a, **k).model_copy(update={"tools": None})

    pb = DropsInFull(SYSTEM, tools=T.tools_field())
    prefix = pb.prefix([], frame(0, "a"), AssistFlags())
    require_tools(prefix, "prefix()")
    with pytest.raises(AssertionError, match="full"):
        require_tools(pb.full(prefix, "q", None, []), "full()")


def test_tools_off_builder_prompts_carry_no_tools():
    """Criterion 3, the too-wide direction: without tools, no prompt carries a tools field."""
    pb = PromptBuilder(SYSTEM)
    prefix = pb.prefix(_history(1), frame(0, "a"), AssistFlags())
    assert prefix.tools is None and pb.full(prefix, "q", None, []).tools is None


async def test_a_dry_run_sessions_prefill_and_generate_carry_identical_tools(tmp_path):
    """Criteria 3-4: the turn's prefill and generate carry the same tools (q8); the bodies built by the real
    ``VllmClient.request_body`` carry identical tool fields; the prefix is shared; the system message ends in
    the instructions; thinking is untouched; two turns, so no turn changes the setting."""
    from local_astra.conductor.llm_client import VllmClient

    rig = T.session_rig(tmp_path, kit=T.make_kit(), script=[answer("Yes. ")])
    await rig.session.start()
    try:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn one never ended")
        await rig.advance(500, step=250)
        rig.llm.script = [answer("Again. ")]
        await rig.ask("and now")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn two never ended")
    finally:
        await rig.session.aclose()
    assert len(rig.llm.prefills) >= 2 and len(rig.llm.gen_prompts) == 2
    client = VllmClient()
    keys = ("tools", "tool_choice", "skip_special_tokens", "return_token_ids")
    for (_, prefix), (rid, full) in zip(rig.llm.prefills[:2], rig.llm.gen_prompts, strict=True):
        require_tools(prefix, "the prefill's prompt")
        require_tools(full, "the generate's prompt")
        assert prefix.shares_prefix_with(full), (
            "criterion 3: the prefix-parity row holds with tools on"
        )
        assert full.messages[0].text().endswith(T.native_instructions())
        pre_body = client.request_body(prefix, rid, LlmOptions(max_tokens=1))
        gen_body = client.request_body(full, rid, LlmOptions())
        assert {k: pre_body.get(k) for k in keys} == {k: gen_body.get(k) for k in keys}, (
            "q8: identical tool fields"
        )
        assert (
            gen_body["chat_template_kwargs"]
            == client.request_body(full.model_copy(update={"tools": None}), rid)[
                "chat_template_kwargs"
            ]
        ), "thinking keeps chat_template_kwargs untouched"
    assert rig.llm.gen_prompts[0][1].tools == rig.llm.gen_prompts[1][1].tools, "a session setting"


async def test_a_dry_run_sessions_over_long_retry_and_trimmed_prompt_keep_the_tools(tmp_path):
    """Criterion 3: the retry after a 413 and the pre-emptive ``fit()`` trim (prefill count over budget) send
    tools too."""
    from tests.fakes.llm import http_error

    rig = T.session_rig(tmp_path, kit=T.make_kit(), script=[answer("Yes. ")], prefix_tokens=20_000)
    rig.llm.fail_generates(http_error(413), None)
    await rig.session.start()
    try:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
    finally:
        await rig.session.aclose()
    assert len(rig.llm.generates) == 2, (
        f"premise: one refused generate, one retry; {len(rig.llm.generates)}"
    )
    for i, (_, prompt, _) in enumerate(rig.llm.generates):
        require_tools(prompt, f"generate {i} (trimmed{' retry' if i else ''})")


async def test_a_tools_off_session_sends_no_tools(tmp_path):
    """Criterion 1/6: without a kit nothing changes — no prompt and no body carries a tools field."""
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
    assert all(p.tools is None for _, p in rig.llm.prefills)
    assert all(p.tools is None for _, p, _ in rig.llm.generates)
    assert rig.llm.bodies == [], "the fake builds a body only for a tools/continuation prompt"
