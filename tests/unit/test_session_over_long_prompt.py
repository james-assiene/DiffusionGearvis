"""The over-long-prompt path in the session (T-0020, criteria 3, 4, 5).

The engine is the only thing that knows how long a prompt is, and it says so in two ways: the
prefill's ``usage.prompt_tokens`` (``PrefillHandle.prefix_tokens``; ``0`` means *not reported*,
T-0007 criterion 19) and an HTTP 400/413 on the request itself. This module drives whole turns
over ``tests/fakes/`` and asserts what the session does with each.

Seams this module pins (the test-author's, as ``test_session_loop.py`` does for T-0007)
=======================================================================================
* a 400/413 from ``LlmBackend.generate`` is an ``httpx.HTTPStatusError`` — what
  ``VllmClient.generate``'s ``raise_for_status`` raises, built by ``tests.fakes.llm.http_error``
  and proved identical to the real client's in ``tests/contract/test_over_long_prompt_error.py``;
* the session retries **once**, with a prompt the builder trimmed by one more history turn, and
  then gives up loudly (``HudError`` + ``CommitHistory``, which the machine already emits for
  ``llm.error`` — so what this task owes is the retry, and that the second failure still reaches
  the machine);
* ``TurnRecord.extras`` of an over-long turn carries
  ``prompt_over_budget: True``, ``prompt_error_status: int``, ``prompt_tokens: int`` (the
  engine's count when it reported one) and ``prompt_trimmed: {"history_turns": n, "images": n,
  "memory_hits": n}`` — the same mapping ``PromptBuilder.fit`` returns (criterion 5).

Everything else — the rig, the scripted clock, the recording fakes — is T-0007's and is imported,
not re-built.
"""

from __future__ import annotations

from typing import Any

from local_astra.conductor.prompt_builder import Budget
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import http_error
from tests.unit.test_session_loop import answer, frame, running

FRAMES = [frame(0, "a")]


def turn_generates(rig: Any) -> list[Any]:
    """The prompts of this session's *turn* generates (the idle captioner's are not turns)."""
    return [prompt for _, prompt in rig.llm.gen_prompts]


def history_messages(prompt: Any) -> int:
    """Messages between the system message and the current user turn: the history block."""
    return max(0, len(prompt.messages) - 2)


def body(prompt: Any) -> str:
    return " ".join(m.text() for m in prompt.messages)


def phashes(prompt: Any) -> list[str | None]:
    return [p.phash for m in prompt.messages for p in m.content if hasattr(p, "phash")]


def image_parts(message_or_prompt: Any) -> list[ImagePart]:
    """Every ``ImagePart`` in a prompt, or in one message."""
    messages = getattr(message_or_prompt, "messages", None) or (message_or_prompt,)
    return [p for m in messages for p in m.content if isinstance(p, ImagePart)]


#: The rig's frame is ``frame(0, "a")``, whose ``title`` is ``window a``; the session uses the
#: title as the history turn's caption, and the builder renders it ``[screen: <caption>]``.
CAPTION = "[screen: window a]"


async def two_turns_of_history(rig: Any) -> None:
    """Two ordinary answered turns, so the third turn has history the trimmer can spend."""
    for question in ("what is in the first column", "and the second"):
        rig.llm.script = [answer("A number. ")]
        await rig.ask(question)
        await rig.until(lambda: rig.state.state is TurnState.IDLE, f"turn {question!r} never ended")
    assert len(rig.state.history) == 2, (
        f"the rig must build two history turns before the over-long one; got {rig.state.history}"
    )


# --- criterion 4 ------------------------------------------------------------------------------


async def test_a_413_is_retried_once_with_a_trimmed_prompt(tmp_path):
    """Criterion 4, the happy half: one retry, and the turn is answered.

    The engine rejects the first request with 413 (over-long prompt) and accepts the second. The
    session must send a *smaller* prompt the second time — one more history turn gone — and the
    user must get their answer: a 400 that kills the turn is exactly what this task removes.
    """
    async with running(tmp_path, frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("The total is forty two. ")]
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("and what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the over-long turn never ended")

        prompts = turn_generates(rig)[2:]  # the first two turns' prompts are history now
        assert len(prompts) == 2, (
            "criterion 4: exactly one retry after a 413 — one request is a dead turn, three is a "
            f"retry loop against a 26B model; got {len(prompts)} requests for this turn"
        )
        first, retry = prompts
        assert history_messages(retry) == history_messages(first) - 2, (
            "the retry drops one more history turn (a user/assistant pair): "
            f"{history_messages(first)} → {history_messages(retry)} history messages"
        )
        assert "The total is forty two." in rig.tts.texts(), (
            "criterion 4: the recovered turn is answered and spoken, not silently abandoned"
        )
        assert not rig.machine.commands(C.HudError), (
            "a turn that recovered must not also report an error to the HUD"
        )
        assert rig.records()[-1].outcome == "spoken", rig.records()[-1].outcome


async def test_the_retry_never_truncates_the_users_own_turn(tmp_path):
    """Criterion 4's negative: the current question and the current frame survive the trim.

    Trimming the transcript or dropping the frame would make the prompt fit and make the answer
    about something the user did not ask (B2's floor: never the system message, the current user
    turn, or the current frame).
    """
    question = "and what does the second column add up to"
    async with running(tmp_path, frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Sixty. ")]
        rig.llm.fail_generates(http_error(400), None)
        await rig.ask(question)
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the over-long turn never ended")

        attempts = turn_generates(rig)[2:]
        assert len(attempts) == 2, (
            "criterion 4: the 400 must be retried, so there is a retry to inspect; got "
            f"{len(attempts)} request(s)"
        )
        retry = attempts[-1]
        assert history_messages(retry) < history_messages(attempts[0]), (
            "the retry is smaller than the request that was rejected"
        )
        assert question in body(retry), (
            f"the retry must carry the user's whole question; it carried {body(retry)[-200:]!r}"
        )
        assert FRAMES[0].phash in phashes(retry), (
            "the current frame's pixels are never what pays for the trim. (The screen has not "
            "changed since the last turn, so those pixels may ride on the history image the "
            "builder already sent — D9/B9 — which is why this is a phash test, not a part test.)"
        )
        assert "Think step by step" in retry.messages[0].text(), (
            "the system message is never dropped"
        )


async def test_the_retry_is_one_request_at_a_time_and_carries_a_fresh_id(tmp_path):
    """The §4 invariant the retry can break: *at most one active generate per session*.

    Two ways to get this wrong, both invisible in the "was it answered" test: starting the retry
    while the rejected request is still being consumed (two of the engine's three sequences on one
    turn), and re-using the rejected request's id — which T-0005 B13 records as the bug that lets a
    stale tombstone swallow the next request. So: never two live generates, and two distinct ids,
    neither of them re-used from an earlier turn.
    """
    async with running(tmp_path, frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Two. ")]
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("how many columns are there")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the over-long turn never ended")

        ids = [request_id for request_id, _ in rig.llm.gen_prompts]
        assert len(ids[2:]) == 2, f"criterion 4: one retry; got {len(ids[2:])} requests"
        assert len(set(ids)) == len(ids), (
            f"every request the engine sees carries a fresh id (T-0005 B13); got {ids}"
        )
        assert all(live <= 1 for _, live, _ in rig.llm.concurrency), (
            "docs/02 §4: at most one active generate per session — the retry waits for the "
            f"rejected request to be released; saw {rig.llm.concurrency}"
        )


async def test_a_second_413_ends_the_turn_loudly(tmp_path):
    """Criterion 4, adversarial: the retry is over-long too.

    No third request, and no dead turn: the machine's error row runs (``hud.error`` then the
    history commit), the record says ``error``, and the status is in the message so the HUD can
    say *why*.
    """
    async with running(tmp_path, frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("never sent. ")]
        rig.llm.fail_generates(http_error(413), http_error(413), None)
        await rig.ask("and the grand total")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "criterion 4: a second 413 must end the turn through the machine's error row",
            limit_ms=5_000,
        )

        prompts = turn_generates(rig)[2:]
        assert len(prompts) == 2, (
            f"one retry, then stop — never a third attempt; got {len(prompts)}"
        )
        errors = rig.machine.events(E.LlmError)
        assert errors, "the unrecoverable 413 must reach the machine as llm.error"
        assert "413" in errors[-1].message, (
            f"the status travels with the error, not a flattened message; got {errors[-1].message!r}"
        )
        hud = rig.machine.commands(C.HudError)
        assert hud and "413" in hud[-1].message, (
            f"criterion 4: HudError, so the turn is never silently dead; got {hud}"
        )
        assert rig.machine.commands(C.CommitHistory), (
            "criterion 4: the user's half of the turn is still committed"
        )
        assert rig.records()[-1].outcome == "error", rig.records()[-1].outcome


async def test_only_an_over_long_status_is_retried(tmp_path):
    """Criterion 4's boundary: a 413 is a retry; a transport failure is not.

    Retrying everything doubles the cost of every real outage on a box with three sequences, and
    T-0007 criterion 13 already pins that a raising backend ends the turn. Both halves live in one
    test because each on its own would be satisfied by the wrong implementation.
    """
    async with running(tmp_path / "retried", frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Yes. ")]
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("is that right")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the 413 turn never ended")
        assert len(turn_generates(rig)[2:]) == 2, "a 413 is retried once"

    async with running(tmp_path / "not_retried", frames=FRAMES) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Yes. ")]
        rig.llm.fail_generates(RuntimeError("the engine went away"))
        await rig.ask("is that right")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the failing turn never ended")
        assert len(turn_generates(rig)[2:]) == 1, (
            "a transport failure is not an over-long prompt: it is reported, not retried"
        )
        assert rig.records()[-1].outcome == "error"


# --- criterion 3: the engine's count, pre-emptively -------------------------------------------


async def test_only_the_engines_reported_count_can_trim_before_the_request(tmp_path):
    """Criterion 3 at the session: the prefill's ``usage.prompt_tokens`` is the trigger, and an
    unreported count (``0``) is not a number.

    Three regimes in one test, because each alone is satisfiable by the wrong implementation:
    a reported count far over the window trims before the request is sent; a reported count well
    under it does not; and ``0`` — which T-0007 criterion 19 pins as *not reported* — is neither
    an empty prefix nor a reason to trim.

    Ruling A9: *how far* over the budget the count is cannot change what is trimmed. The engine
    reports one number for the prefix and no per-image cost, so nothing can price a second rung;
    a count twice the window therefore spends the same single rung as ``budget + 1`` does (ruling
    A6, ``test_a_one_token_overshoot_spends_exactly_one_rung``), and the 400/413 path is what
    escalates. This row used to assert ``< 4`` here, which contradicted the A6 row on the same
    fixture.
    """
    async with running(tmp_path / "over", frames=FRAMES, prefix_tokens=2 * 16_384) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Trimmed. ")]
        await rig.ask("what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        prefix, sent = rig.llm.prefills[-1][1], turn_generates(rig)[-1]
        assert history_messages(prefix) == 4 and len(image_parts(prefix)) == 2, (
            "regime: two history turns, each carrying the screen; got "
            f"{history_messages(prefix)} history messages and {len(image_parts(prefix))} images"
        )
        assert history_messages(sent) == 4, (
            "ruling A9/A6: a count twice the window spends the same one rung as budget + 1 — the "
            f"rung is pixels, so every history message stays; {history_messages(sent)} of 4 did"
        )
        assert len(image_parts(sent)) == len(image_parts(prefix)) - 1, (
            "criterion 3: a prefix the engine counted at 32768 tokens is trimmed before it is "
            f"sent, by exactly one image; {len(image_parts(prefix))} → {len(image_parts(sent))}"
        )
        assert len(turn_generates(rig)[2:]) == 1, "trimming ahead of time costs no extra request"
        assert rig.records()[-1].extras.get("prompt_over_budget") is True, (
            f"criterion 5: a pre-emptive trim is logged like any other; got {rig.records()[-1].extras}"
        )
        assert rig.records()[-1].extras.get("prompt_tokens") == 2 * 16_384, (
            "criterion 5: the number in the record is the engine's, verbatim; got "
            f"{rig.records()[-1].extras}"
        )

    for name, tokens in (("under", 1_000), ("unreported", 0)):
        async with running(tmp_path / name, frames=FRAMES, prefix_tokens=tokens) as rig:
            await two_turns_of_history(rig)
            rig.llm.script = [answer("Whole. ")]
            await rig.ask("what is the total")
            await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
            sent = turn_generates(rig)[-1]
            assert history_messages(sent) == 4, (
                f"prefix_tokens={tokens} is no reason to drop history ({name}); the prompt went "
                f"out with {history_messages(sent)} history messages instead of 4"
            )
            assert rig.records()[-1].extras.get("prompt_over_budget") in (None, False)


async def test_a_count_that_changes_between_prefill_and_generate(tmp_path):
    """Adversarial: the prefill said it fits, the generate 413s anyway.

    The prefill counts the *prefix*; the request carries the query-dependent suffix on top, the
    template may differ, and the engine's window is shared with the KV budget — so "the number
    said yes" must never harden into "a 413 is impossible". The recovery path is the same one, and
    the record keeps both facts: the count the engine reported, and that it was rejected anyway.
    """
    async with running(tmp_path, frames=FRAMES, prefix_tokens=1_000) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Recovered. ")]
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        prompts = turn_generates(rig)[2:]
        assert len(prompts) == 2 and history_messages(prompts[1]) < history_messages(prompts[0]), (
            "a 413 after a fitting count is still retried with a smaller prompt; got "
            f"{[history_messages(p) for p in prompts]}"
        )
        extras = rig.records()[-1].extras
        assert extras.get("prefix_tokens") == 1_000, (
            f"the reported count stays in the record, wrong or not; got {extras}"
        )
        assert extras.get("prompt_over_budget") is True, (
            "the disagreement between the reported count and the engine's verdict is the thing a "
            f"later 'why did it forget X' needs; got {extras}"
        )


# --- criterion 5 ------------------------------------------------------------------------------


async def test_the_over_long_turn_records_the_count_and_what_was_trimmed(tmp_path):
    """Criterion 5: the record answers "why did it forget X" for this turn.

    The turn recovered, so nothing else in the record says anything happened — the extras are the
    only trace. They must name the status, the engine's count and *what* was dropped, with the
    counts the builder returned rather than a flag.
    """
    async with running(tmp_path, frames=FRAMES, prefix_tokens=1_234) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer("Fine. ")]
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        extras = rig.records()[-1].extras
        assert extras.get("prompt_over_budget") is True, extras
        assert extras.get("prompt_error_status") == 413, (
            f"criterion 5: the status the engine returned; got {extras}"
        )
        trimmed = extras.get("prompt_trimmed")
        assert isinstance(trimmed, dict), f"criterion 5: what was trimmed, as counts; got {extras}"
        assert set(trimmed) >= {"history_turns", "images", "memory_hits"}, trimmed
        assert trimmed["history_turns"] >= 1, (
            f"one more history turn went into the retry; the record must say so ({trimmed})"
        )
        line = rig.lines()[-1]
        assert "prompt_trimmed" in line, "the extras must survive the JSON round trip to the log"


# --- ruling A6: the magnitude of the pre-emptive trim ------------------------------------------


async def test_a_one_token_overshoot_spends_exactly_one_rung(tmp_path):
    """Ruling A6: +1 token over the budget costs one rung, not the whole history.

    ``PrefillHandle`` carries one number for the whole prefix and no per-image cost, so nothing
    can measure what a rung saves — which is exactly why the magnitude has to be pinned here. A
    session that trims to the floor on a one-token overshoot throws away 100 % of the history for
    1 token and the user's "what did I just ask you" stops working; a session that trims nothing
    sends a prompt the engine will reject.

    Hand-computed from the rig's two history turns, both carrying the same screen: the prefix the
    session prefilled holds two images and four history messages; one rung is the *oldest* turn's
    pixels, so the request that goes out holds four history messages, one image, and the caption
    line in place of the pixels it spent. The 413 that follows escalates by exactly one turn, and
    the second 413 ends the turn loudly.
    """
    budget = Budget().max_prompt_tokens
    async with running(tmp_path, frames=FRAMES) as rig:
        rig.llm.prefill_counts(1_000, 1_000, budget + 1)
        await two_turns_of_history(rig)
        rig.llm.script = [answer("never sent. ")]
        rig.llm.fail_generates(http_error(413), http_error(413), None)
        await rig.ask("what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the over-long turn never ended")

        assert [count for _, count in rig.llm.prefill_returns] == [1_000, 1_000, budget + 1], (
            f"regime: the third turn's prefill is the over-budget one; got {rig.llm.prefill_returns}"
        )
        prefix = rig.llm.prefills[-1][1]
        assert history_messages(prefix) == 4 and len(image_parts(prefix)) == 2, (
            "regime: the untrimmed prefix is two history turns, each carrying the screen; got "
            f"{history_messages(prefix)} history messages and {len(image_parts(prefix))} images"
        )

        attempts = turn_generates(rig)[2:]
        assert len(attempts) == 2, (
            f"one retry, then the loud error — never a third attempt; got {len(attempts)}"
        )
        sent, retry = attempts
        assert history_messages(sent) == 4, (
            "ruling A6: one token over budget spends one rung, and a rung is *pixels* — every "
            f"history message stays; {history_messages(sent)} of 4 survived"
        )
        assert len(image_parts(sent)) == 1, (
            "ruling A6: exactly one image is lost, not every image and not none; the request "
            f"carried {len(image_parts(sent))} of 2"
        )
        assert not image_parts(sent.messages[1]), (
            "the *oldest* history turn's pixels are the rung that is spent"
        )
        assert CAPTION in sent.messages[1].text(), (
            "B3: the frame degrades to its caption line, it does not vanish; the oldest history "
            f"message reads {sent.messages[1].text()!r}"
        )
        assert image_parts(sent.messages[3]), (
            "the newest history frame keeps its pixels — it is the screen the follow-up is about"
        )
        assert history_messages(retry) == 2, (
            "the 413 escalates by exactly one history turn (drop_history_turns=1); the retry "
            f"carried {history_messages(retry)} history messages"
        )
        assert rig.machine.commands(C.HudError), "the second 413 ends the turn loudly"
        assert rig.records()[-1].outcome == "error", rig.records()[-1].outcome


# --- ruling A7: the count is this turn's ------------------------------------------------------


async def test_a_stale_over_budget_count_does_not_trim_the_next_turn(tmp_path):
    """Ruling A7: the engine counts *each* prefix, so last turn's number decides nothing here.

    Turn three's prefill reports over budget and its request is trimmed; turn four's prefill is
    still in flight when its request is built — the ordinary case, since D9 aborts the
    anticipatory prefill at ``llm.start`` — so turn four has *no* count of its own and must go out
    whole. The check is ``shares_prefix_with``: a prompt that still extends the prefix the session
    prefilled for *that* turn is a prompt nothing trimmed, and it is the same property D9 needs
    for the cache to hit, so a stale count costs both the history and the prefix cache.
    """
    budget = Budget().max_prompt_tokens
    async with running(tmp_path, frames=FRAMES) as rig:
        rig.llm.prefill_counts(1_000, 1_000, budget + 1, 1_000)
        await two_turns_of_history(rig)

        rig.llm.script = [answer("Trimmed. ")]
        await rig.ask("what is the total")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the over-budget turn hung")
        over_prefix, over_sent = rig.llm.prefills[-1][1], turn_generates(rig)[-1]
        assert len(image_parts(over_sent)) == len(image_parts(over_prefix)) - 1, (
            "regime: the over-budget turn must really have been trimmed, or turn four's assertion "
            f"is vacuous; {len(image_parts(over_prefix))} → {len(image_parts(over_sent))} images"
        )

        rig.llm.script = [answer("Whole. ")]
        # Turn four's prefill has not answered when the request goes out — the anticipatory
        # prefill is aborted at ``llm.start`` by design (docs/01 D9), so "no count for this turn"
        # is the ordinary case, and it is the only one in which the previous turn's number is
        # still lying around to be misused.
        rig.llm.hold_prefill()
        await rig.ask("and the one before that")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the fitting turn hung")
        rig.llm.release_prefill()
        prefix, sent = rig.llm.prefills[-1][1], turn_generates(rig)[-1]

        assert [count for _, count in rig.llm.prefill_returns] == [1_000, 1_000, budget + 1], (
            "regime: turn four's prefill never reported — it was still in flight when the request "
            f"was built; got {rig.llm.prefill_returns}"
        )
        assert prefix.shares_prefix_with(sent), (
            "ruling A7: turn four fits, so its request still extends the prefix that was "
            "prefilled — the previous turn's over-budget count is not this turn's (and a prompt "
            "that no longer extends its own prefill misses every cached block, D9)"
        )
        assert len(image_parts(sent)) == len(image_parts(prefix)), (
            f"nothing is degraded on a turn that fits; {len(image_parts(prefix))} images in the "
            f"prefix, {len(image_parts(sent))} on the wire"
        )
        assert history_messages(sent) == history_messages(prefix), (
            f"and no history turn is dropped; {history_messages(prefix)} → {history_messages(sent)}"
        )
        assert rig.records()[-1].extras.get("prompt_over_budget") in (None, False), (
            f"turn four was never over budget; its record says {rig.records()[-1].extras}"
        )
