"""Request ids and the concurrency budget (T-0007 criteria 7 and 8).

``--max-num-seqs 3``: one generate + one anticipatory prefill + the idle captioner, and not one
more (``.claude/rules/conductor.md``, docs/03 I8). Ids are unique for the lifetime of the
``LlmBackend`` object, which the machine's own ``t{n}-gen`` cannot promise: ``initial()`` resets
``turn_seq`` to 0, so two sessions over one engine issue the same names (PROGRESS: "T-0007 must
make ids unique per turn", T-0005 B13). The API is fixed in ``test_session_loop.py``.
"""

from __future__ import annotations

import asyncio
import tempfile
from itertools import pairwise
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.perception import Source
from local_astra.contracts.turn import TurnState
from local_astra.memory.captioner import IDLE_BEFORE_CAPTION_MS
from tests.unit.test_session_loop import (
    RecordingLlm,
    answer,
    frame,
    require_session,
    running,
)

#: criterion 8: the executor's floor between two prefill re-issues for the same turn.
REISSUE_MIN_MS = 750


def backend_ids(llm: RecordingLlm) -> list[str]:
    """Every id the *backend* saw, in order: prefills, generates (turn and captioner)."""
    return [rid for rid, _ in llm.prefills] + [rid for rid, _, _ in llm.generates]


async def run_one_turn(rig, text: str = "hello there") -> None:
    await rig.ask(text)
    await rig.until(lambda: rig.state.state is TurnState.IDLE, f"turn {text!r} never ended")


async def test_request_ids_are_unique_for_the_backends_lifetime(tmp_path):
    """Criterion 7 over 20 turns, ten of them on a session whose machine starts from ``initial()``.

    Both sessions share one ``LlmBackend``. The second one's machine numbers its turns from 1
    again, so an executor that passes the machine's ``t1-prefill`` / ``t1-gen`` through to the
    engine issues every id twice — and ``abort(request_id)`` then aborts the wrong request.
    """
    llm = RecordingLlm([answer("Yes. ")])
    for index in range(2):
        async with running(
            tmp_path / f"s{index}", llm=llm, frames=[frame(0, "a")], ms_per_word=100
        ) as rig:
            for turn in range(10):
                await run_one_turn(rig, f"question number {turn}")

    ids = backend_ids(llm)
    assert len(ids) >= 40, f"20 turns owe a prefill and a generate each; got {len(ids)} ids"
    duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
    assert not duplicates, (
        "criterion 7: no prefill/generate/caption id is issued twice to the same LlmBackend; "
        f"these were reused: {duplicates}"
    )


async def test_aborting_a_finished_request_does_not_swallow_the_next_turn(tmp_path):
    """Criterion 7, adversarial — the T-0005 B13 tombstone, the most likely regression here.

    Two shapes of the same bug, in one test:

    1. the executor is told to abort a request that has already **finished** (the machine aborts
       this turn's prefill at end-of-turn, long after ``PrefillHandle`` came back, and the test
       aborts the finished *generate* on top of that);
    2. the next session's machine reissues the same ``t1-gen`` name.

    An executor that remembers "aborted" by request id — which it must, because ``generate`` is an
    async generator and abort has to be meaningful before the first ``__anext__`` (T-0005 M5) —
    and never forgets, swallows the next turn's commits. The turn then sits in THINKING until
    ``llm_timeout``: no speech, no history, an error line in ``turns.jsonl``.
    """
    llm = RecordingLlm([answer("First answer. ")])
    async with running(tmp_path / "a", llm=llm, frames=[frame(0, "a")], ms_per_word=100) as rig:
        await run_one_turn(rig, "first question")
        finished = [c.request_id for c in rig.machine.commands(C.LlmStart)]
        assert finished, "the first turn must have started a generate"
        # the request is done; the executor is asked to abort it anyway (idempotent by contract)
        await rig.session.executor.execute(C.LlmAbort(request_id=finished[-1]))
        await rig.pump()
        assert [t for _, t, _ in rig.tts.requests] == ["First answer."], rig.tts.texts()

    llm.script = [answer("Second answer. ")]
    async with running(tmp_path / "b", llm=llm, frames=[frame(0, "a")], ms_per_word=100) as rig:
        await rig.ask("second question")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "criterion 7: the next generate must yield its commits — a tombstone from the "
            "finished abort swallowed the new request",
            limit_ms=5_000,
        )
        assert [t for _, t, _ in rig.tts.requests] == ["Second answer."], (
            f"the second session spoke {rig.tts.texts()}"
        )
        assert rig.records()[-1].outcome == "spoken"


async def test_prefill_abort_at_end_of_turn_and_captioner_abort_at_onset_reach_the_backend(
    tmp_path,
):
    """Criterion 8: both aborts are real ``LlmBackend.abort`` calls, not bookkeeping.

    The captioner half needs an episode to caption, so the turn that ingests one runs first; the
    session then has to be idle for ``IDLE_BEFORE_CAPTION_MS`` before ``run_once()`` may start
    (T-0009 A1: the captioner owns no clock, the session does).
    """
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        await run_one_turn(rig, "what is this")
        prefill_ids = [c.request_id for c in rig.machine.commands(C.PrefillAbort)]
        assert prefill_ids, "the ENDPOINTING → THINKING row aborts the anticipatory prefill"
        assert rig.llm.aborted, "criterion 8: PrefillAbort calls LlmBackend.abort"

        assert rig.memory.episodes, (
            "commit_history stores the turn's frame as an Episode (docs/02 §2.2); the captioner "
            "has nothing to do otherwise"
        )
        for episode in rig.memory.episodes.values():
            assert Path(episode.thumbnail_path).is_file(), (
                "Episode.thumbnail_path is a path: the session writes the thumbnail under the "
                f"data dir (got {episode.thumbnail_path!r})"
            )

        rig.llm.hold()  # the caption request stays in flight until the onset aborts it
        await rig.advance(IDLE_BEFORE_CAPTION_MS + 1_000, step=500)
        captions = [rid for rid, _, _ in rig.llm.generates if rid not in prefill_ids]
        assert rig.llm.live_captions, (
            "criterion 8: after IDLE_BEFORE_CAPTION_MS in IDLE the session runs the idle "
            f"captioner (generates so far: {captions})"
        )
        in_flight = set(rig.llm.live_captions)
        before = list(rig.llm.aborted)
        await rig.onset()
        rig.llm.release()
        await rig.pump()
        assert set(rig.llm.aborted) - set(before) >= in_flight, (
            "criterion 8: CaptionerAbort at speech onset calls LlmBackend.abort for the caption "
            f"request ({in_flight} not among {set(rig.llm.aborted) - set(before)})"
        )


async def test_prefill_reissue_is_rate_limited(tmp_path):
    """Criterion 8: ≥ 750 ms between two prefill re-issues for the same turn.

    ``scene.changed`` while LISTENING makes the machine ask for a re-warm every time; rate
    limiting is the *executor's* job (the machine is pure and has no clock). Four scene changes
    inside 400 ms may cost at most one extra prefill; the lower bound is asserted on the call
    times, so an executor that uses ``PrefillPolicy.reissue_min_interval_ms`` (1000 ms) passes too.
    """
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        await rig.onset()
        for index in range(4):
            await rig.advance(100, step=100)
            now = rig.clock.now_ms
            rig.perception.add(frame(now, chr(ord("b") + index)))
            rig.submit(E.SceneChanged(t_ms=now, source=Source.SCREEN.value, phash=f"p{index}"))
            await rig.pump()
        await rig.advance(2_000, step=250)
        rig.perception.add(frame(rig.clock.now_ms, "z"))
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source=Source.SCREEN.value, phash="late"))
        await rig.pump()

        reissues = [c for c in rig.machine.commands(C.PrefillReissue)]
        assert len(reissues) == 5, f"the machine asks every time; got {len(reissues)}"
        times = rig.llm.times("prefill")
        gaps = [b - a for a, b in pairwise(times)]
        assert all(gap >= REISSUE_MIN_MS for gap in gaps), (
            f"criterion 8: prefill re-issues must be ≥ {REISSUE_MIN_MS} ms apart; call times "
            f"{times} give gaps {gaps}"
        )
        assert len(times) >= 2, (
            "the re-issue after a 2 s gap must go through — a rate limiter that never reopens is "
            f"not a rate limiter (prefill calls: {times})"
        )


# --- criterion 8 as a property ------------------------------------------------------------------

ACTIONS = ("ask", "scene", "takeover", "onset", "idle", "playout")


@given(actions=st.lists(st.sampled_from(ACTIONS), min_size=2, max_size=6))
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_at_most_one_generate_one_prefill_one_caption(actions):
    """Criterion 8 (Hypothesis, ``--hypothesis-seed=0``, ``max_examples=25`` — each example drives
    a real event loop, so the budget is small on purpose).

    ``FakeLlm.active`` is sampled at every call that changes what is live. The engine runs with
    ``--max-num-seqs 3`` and each denoising request costs a ~2.5 GiB sampler transient vLLM does
    not profile, so a fourth live request is a hang on the box, not a slowdown.
    """
    require_session()

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            async with running(
                Path(tmp),
                script=[answer("A short answer. ")],
                frames=[frame(0, "a")],
                ms_per_word=100,
            ) as rig:
                for index, action in enumerate(actions):
                    now = rig.clock.now_ms
                    if action == "ask":
                        await rig.ask(f"question {index}", onset=rig.state.state is TurnState.IDLE)
                    elif action == "scene":
                        rig.perception.add(frame(now, "s"))
                        rig.submit(
                            E.SceneChanged(t_ms=now, source=Source.SCREEN.value, phash=f"s{index}")
                        )
                    elif action == "takeover":
                        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
                    elif action == "onset":
                        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
                    elif action == "idle":
                        await rig.advance(IDLE_BEFORE_CAPTION_MS + 500, step=500)
                    else:
                        await rig.advance(1_000, step=100)
                    await rig.pump()

                    live = rig.llm.live_prefills | rig.llm.live_generates | rig.llm.live_captions
                    assert len(live) <= 3, (
                        f"after {actions[: index + 1]}: {len(live)} live requests {sorted(live)} "
                        "— the budget is one generate + one prefill + the captioner"
                    )
                for prefills, generates, captions in rig.llm.concurrency:
                    assert generates <= 1, f"two live generates: {rig.llm.concurrency}"
                    assert prefills <= 1, f"two live prefills: {rig.llm.concurrency}"
                    assert captions <= 1, f"two live captions: {rig.llm.concurrency}"
                    assert prefills + generates + captions <= 3

    asyncio.run(scenario())


def test_the_rig_itself_would_notice_a_fourth_request(tmp_path):
    """Guard for the property above: the counters it reads are not always empty.

    A concurrency assertion over an empty log passes vacuously — which is how "at most one
    generate" survives an executor that starts none at all.
    """
    require_session()

    async def scenario() -> None:
        async with running(
            tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")], ms_per_word=100
        ) as rig:
            await rig.ask("hello")
            await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
            assert any(g == 1 for _, g, _ in rig.llm.concurrency), (
                f"no live generate was ever observed: {rig.llm.concurrency}"
            )
            assert any(p == 1 for p, _, _ in rig.llm.concurrency), (
                f"no live prefill was ever observed: {rig.llm.concurrency}"
            )

    asyncio.run(scenario())
