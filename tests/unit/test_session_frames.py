"""T-0059 group C — the session keeps a turn's frames, and the record says so.

Criteria 18-23 of ``docs/tasks/T-0059.md``, driven over ``tests/fakes/`` and the scripted clock
(no GPU, no network, no real perception). Every expectation here comes from that file's criteria
and design rulings (D1, D3, D5, rulings A5, A6, A10) and from ``docs/02-architecture.md`` §5 —
never from what ``session.py`` does today. Group A (the machine accumulates) and group B (the
prompt carries them) are merged on this branch; what is missing is the middle: the session still
overwrites ``Session._frame`` on every snapshot, so a turn that saw three screens sends one.

Names this module fixes
=======================
The task names the ``extras`` keys; it does not name the session's private field, and this module
deliberately does not either — a row that asserted ``session._frames`` would pin a design the
implementer owns. What is pinned is what leaves the session: the prompt on the wire, the turn
line, and the number of requests.

``TurnRecord.extras`` (criterion 20, names are contract)

    "frames"                  ordered list of the perceptual hashes of the frames this turn
                              captured, oldest first, after the session's own de-duplication and
                              cap (criterion 18). The turn's first entry is the onset frame.
    "frames_kept"             how many of them the turn kept — ``len(extras["frames"])``. Two
                              keys and not one because a reader that only wants the count must
                              not have to parse the list, and because the pair is the diagnosis:
                              a build that reports a count its own list contradicts is broken.
    "scene_changes_seen"      how many ``scene.changed`` events reached the session while this
                              turn was the open one — including the ones the machine suppressed
                              as a repeat and the ones the cap dropped. ``seen > kept`` is
                              exactly "a screen the user showed did not reach the model", which
                              is the fact criterion 20 exists to make diagnosable.
    "history_images_evicted"  how many history images this turn's frames pushed off the prompt
                              (ruling A5, finding 2: a third current frame evicts the oldest
                              history image ahead of every pixel, and the cost is admitted on the
                              turn line rather than hidden). ``0`` on every turn in this module,
                              which have no history pixels to evict — the rows assert presence
                              and that number, not the eviction mechanism, which is group B's.

Interpretations, stated so the reviewer can overrule them
=========================================================
* ``scene_changes_seen`` counts events delivered **while a turn is open**. Every scene change in
  this module arrives while the turn is LISTENING, so no row depends on what the session does
  with a ``scene.changed`` that arrives at IDLE; that case is left to the implementer and is
  reported as an ambiguity.
* ``frames`` lists the frames the turn *captured*; the prompt's labels count the images actually
  *attached* (ruling A6), so the two may differ by one when a frame rides a history image. No row
  here creates that situation, and none asserts the two are equal in general.
* Criterion 22's "prefill re-issues sent ≤ frames appended" is read as: re-issues that reached
  the backend ≤ frames appended to the turn after the onset one.

How the expectations are computed
=================================
By hand from the fixtures. ``tests/unit/test_session_loop.frame(ts_ms, tag)`` derives both the
JPEG bytes and the phash from ``tag`` and ``ts_ms``, so no two frames in a row can be confused
and the expected phash list is written out from the frames the test itself created. The one
derived expectation is criterion 23's byte-identical prompt, assembled from the builder's own
constants — never from a copied literal of the system block (T-0053 ruling A3: T-0055 is
rewriting that block and owes a token count for it).
"""

from __future__ import annotations

import base64
from typing import Any

from local_astra.conductor.prompt_builder import (
    CLOSE_THE_THOUGHT,
    SPOKEN_STYLE,
    Budget,
)
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.turn import TurnState
from local_astra.telemetry.turns_log import TurnRecord
from tests.unit.test_session_loop import answer, frame, running

#: The four keys criterion 20 adds to every turn line.
FRAME_EXTRAS = ("frames", "frames_kept", "scene_changes_seen", "history_images_evicted")

#: T-0108 criterion 10's two, added to every turn line under that task's name (discounted).
T0108_EXTRAS = frozenset({"ui_valid_final", "ui_shown"})

#: T-0127 criterion 5's one, added to every turn line under that task's name (discounted).
T0127_EXTRAS = frozenset({"ui_complete_final"})

#: Every ``extras`` key a turn of this rig carries **today** (measured on this branch before
#: group C exists). Criterion 23's "the turn line differs only by the additive extras keys" is
#: the difference against this set: a fifth new key, or a lost old one, reddens
#: ``test_a_one_screen_turn_is_byte_identical_to_todays_prompt``.
TODAYS_EXTRAS = frozenset(
    {
        "capture_gated_blocks",
        "capture_xruns",
        "prefix_tokens",
        "prefix_tokens_reported",
        "speech_refused",
        "stt_feed_errors",
        "stt_frames_fed",
        "ui_canvases",
        "ui_fence",
        "ui_repairs",
        "ui_resamples",
        "ui_valid",
    }
)


# --- reading what left the session ---------------------------------------------------------------


def images(prompt: Prompt) -> list[ImagePart]:
    return [p for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]


def phashes(prompt: Prompt) -> list[str | None]:
    return [p.phash for p in images(prompt)]


def serialised(prompt: Prompt) -> list[tuple[str, str]]:
    """Every part of every message, serialised. The prefix relation is compared on these, never
    on lengths: two prompts with the same number of parts can differ in every byte."""
    return [(m.role, p.model_dump_json()) for m in prompt.messages for p in m.content]


def data_url(jpeg: bytes) -> str:
    """The data URL docs/02 §5 specifies, written out rather than read from the builder."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def extra(record: TurnRecord, key: str, criterion: str) -> Any:
    """One ``extras`` key, or a failure that names the criterion instead of a ``KeyError``."""
    assert key in record.extras, (
        f"T-0059 criterion {criterion}: the turn line must carry extras[{key!r}] — the frames a "
        "turn showed the model are the same class of fact as the badge state, and a screen that "
        "was dropped is only diagnosable if the line says so. Got "
        f"{sorted(record.extras)}"
    )
    return record.extras[key]


def turn_record(rig: Any, turn_id: str) -> TurnRecord:
    matching = [r for r in rig.records() if r.turn_id == turn_id]
    assert len(matching) == 1, (
        f"expected one line for {turn_id}; got {[r.turn_id for r in rig.records()]}"
    )
    return matching[0]


# --- driving -------------------------------------------------------------------------------------


async def announce(rig: Any, tag: str | None = None, *, phash: str | None = None) -> Frame | None:
    """A ``scene.changed`` at the clock's now.

    With a ``tag`` a *new* frame is put in the perception ring first, timestamped now, so the
    session's ``get_frame(source, at_ms=t)`` — "the latest frame at or before ``t``" — returns it;
    the announced hash is that frame's own unless ``phash`` overrides it (D3: the announced hash
    and the fetched one are not the same thing). Without a tag nothing new is captured and the
    ring answers with the frame it already held, which is the refetch case of criterion 18.
    """
    now = rig.clock.now_ms
    captured: Frame | None = None
    if tag is not None:
        captured = frame(now, tag)
        rig.perception.add(captured)
    announced = phash if phash is not None else (captured.phash if captured else "unseen" * 2)
    rig.submit(E.SceneChanged(t_ms=now, source=Source.SCREEN.value, phash=announced))
    await rig.pump()
    return captured


# --- criterion 19: two scene changes in one utterance reach the model -----------------------------


async def test_two_scene_changes_in_one_utterance_reach_the_model_as_three_images(tmp_path):
    """Criterion 19, the row the human asked for by name: *"I just switched tabs, compare them"*.

    Onset, ``scene.changed`` A, ``scene.changed`` B, end of turn. The final request must carry
    **three** ``ImagePart``s in capture order — the onset frame first, which is what keeps
    ``test_session_prompt.py::test_frame_comes_from_onset_not_latest`` the canary it is — the
    transcript must be the last part of the last message, the last prefill that actually reached
    the backend must be a byte-prefix of that request (D1: frames only ever *append*, which is
    why the images go before the text), and ``extras["frames"]`` must list the three hashes in
    the same order.

    Today the third assertion is the only one that holds: ``snapshot_frame`` overwrites
    ``Session._frame``, so the prompt carries the newest screen and the first two are gone.

    The changes are spaced beyond ``PrefillPolicy.reissue_min_interval_ms`` on purpose, so both
    re-issues reach the wire and "the last prefill is a prefix of the request" is a claim about a
    three-image prefill rather than about a one-image one. The rate-limited case is criterion
    21's row.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        await rig.words("which of these")
        await rig.advance(1_200, step=200)
        first = await announce(rig, "b")
        await rig.words("two screens")
        await rig.advance(1_200, step=200)
        second = await announce(rig, "c")
        await rig.words("shows the later date")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert first is not None and second is not None
        expected = [onset_frame.phash, first.phash, second.phash]
        assert len(rig.llm.gen_prompts) == 1, f"one request for the turn; got {rig.llm.gen_prompts}"
        _, request = rig.llm.gen_prompts[0]
        assert phashes(request) == expected, (
            "criterion 19: the turn saw three screens and the request must carry all three, "
            "oldest first — the onset frame, then the tab the user switched to, then the next "
            f"one. Got {phashes(request)}"
        )
        assert [p.url for p in images(request)] == [
            data_url(onset_frame.jpeg),
            data_url(first.jpeg),
            data_url(second.jpeg),
        ], "and the pixels are each frame's own, not the same frame three times"
        assert all(isinstance(p, ImagePart) for p in request.messages[-1].content[:1]), (
            "criterion 19 / docs/01 D9: the current turn's images lead its message"
        )
        last_part = request.messages[-1].content[-1]
        assert isinstance(last_part, TextPart) and last_part.text.endswith("later date"), (
            "the transcript is the last part of the last message — every image before the text "
            f"(.claude/rules/conductor.md's layout bullet). Got {last_part!r}"
        )

        _, sent = rig.llm.prefills[-1]
        assert phashes(sent) == expected, (
            "criterion 19: the last prefill actually sent carries the same three frames — it is "
            f"what the final request extends. Got {phashes(sent)}"
        )
        prefix_parts, request_parts = serialised(sent), serialised(request)
        assert request_parts[: len(prefix_parts)] == prefix_parts, (
            "criterion 19 / D1: the prefill is a byte-prefix of the final request, compared on "
            "the serialised parts. A frame that replaced another instead of appending breaks "
            "every cached block from the image onwards"
        )
        assert sent.shares_prefix_with(request), "and the D9 relation the contract states"

        record = turn_record(rig, "t1")
        assert extra(record, "frames", "19") == expected, (
            "criterion 19: extras['frames'] is the ordered phash list of what the turn showed "
            f"the model; got {record.extras.get('frames')!r}"
        )
        for phash in expected:
            assert rig.session._resolve_frame(phash) is not None, (
                "criterion 18: every frame's bytes enter the resolver's store, or the next "
                f"turn cannot send this screen back as a history image (missing {phash})"
            )
        assert len(rig.memory.episodes) == 1, (
            "D5 / ruling A1.6: a multi-frame turn still ages as **one** unit — one Episode per "
            f"turn, not one per frame. Got {len(rig.memory.episodes)}"
        )
        assert [e.phash for e in rig.memory.episodes.values()] == [second.phash], (
            "and the episode is the turn's newest frame, exactly as today"
        )


# --- criterion 20: the turn line carries all of them ---------------------------------------------


async def test_the_turn_line_lists_every_phash_and_says_how_many_were_dropped(tmp_path):
    """Criterion 20. Four distinct screens in one utterance, a cap of three.

    Expected by hand from D1 and ruling A4's arithmetic. The machine's ``turn_phashes`` holds at
    most ``cap - 1`` entries because the onset frame's hash reaches no event, so four changes
    give four snapshots; the session holds at most ``cap`` *frames*, appending while there is
    room and replacing the **last** slot afterwards. With the onset frame ``a`` and changes
    ``b, c, d, e`` that is ``[a, b, e]``: ``c`` took the live slot, ``d`` replaced it and ``e``
    replaced ``d``. Nothing is ever dropped from the middle and nothing is reordered.

    So ``scene_changes_seen`` is 4 and ``frames_kept`` is 3: the line says two screens were shown
    and did not reach the model, which is the whole point of the key — a dropped screen is
    otherwise invisible in the prompt, in the log and in the answer.
    """
    cap = Budget().max_frames_per_turn
    assert cap == 3, f"this row's hand arithmetic assumes the default cap of 3; Budget says {cap}"
    assert TurnPolicy().max_frames_per_turn == cap, (
        "the two caps are separate constants in modules that must not import each other, and "
        "their defaults are equal (Implementation notes); this row is written against that"
    )
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        captured = []
        for tag in ("b", "c", "d", "e"):
            await rig.advance(1_200, step=200)
            captured.append(await announce(rig, tag))
        await rig.words("what changed")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert len(rig.machine.commands(C.SnapshotFrame)) == 5, (
            "the onset snapshot plus one per distinct change (criterion 1); got "
            f"{[c.at_ms for c in rig.machine.commands(C.SnapshotFrame)]}"
        )
        record = turn_record(rig, "t1")
        kept = [onset_frame.phash, captured[0].phash, captured[3].phash]
        assert extra(record, "frames", "20") == kept, (
            "criterion 20 / D1: the first frames are never disturbed and the **last** slot is "
            f"the live one, so the turn keeps {kept}; got {record.extras.get('frames')!r}"
        )
        assert extra(record, "frames_kept", "20") == len(kept), (
            "criterion 20: frames_kept is the count of the list the same line carries; a count "
            f"that contradicts its own list is the bug this pair exists to catch ({record.extras})"
        )
        assert extra(record, "scene_changes_seen", "20") == 4, (
            "criterion 20: every scene.changed the turn saw is counted, including the ones the "
            f"cap dropped; got {record.extras.get('scene_changes_seen')!r}"
        )
        assert extra(record, "scene_changes_seen", "20") > extra(record, "frames_kept", "20"), (
            "criterion 20: seen > kept is the signal that a screen the user showed did not reach "
            "the model — the reason both numbers are on the line"
        )
        assert extra(record, "history_images_evicted", "20") == 0, (
            "ruling A5 finding 2: this turn has no history pixels, so it evicted none; the key "
            "is on every line so the cost of a third frame is observable when there are"
        )
        _, request = rig.llm.gen_prompts[0]
        assert phashes(request) == kept, (
            f"and the request carries exactly what the line claims; got {phashes(request)}"
        )


async def test_a_one_frame_turn_still_lists_its_one_frame(tmp_path):
    """Criterion 20's "on **every** turn". The ordinary turn, which is most of them.

    A line whose keys appear only on multi-frame turns is a line a bench reader cannot group by,
    and "no screen was dropped" would be indistinguishable from "this build does not say".
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.ask("what is on this screen")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        record = turn_record(rig, "t1")
        assert extra(record, "frames", "20") == [onset_frame.phash], (
            f"criterion 20: one frame, listed; got {record.extras.get('frames')!r}"
        )
        assert extra(record, "frames_kept", "20") == 1
        assert extra(record, "scene_changes_seen", "20") == 0, (
            "no scene changed during this utterance, and the line says so with a number rather "
            "than by omitting the key"
        )
        assert extra(record, "history_images_evicted", "20") == 0
        for key in FRAME_EXTRAS:
            assert isinstance(record.extras[key], list | int), (
                f"criterion 20: extras[{key!r}] must survive the JSON round trip of the turns "
                f"file as a list or an int; got {type(record.extras[key])}"
            )


# --- criterion 21: a suppressed re-issue never loses a frame --------------------------------------


async def test_a_rate_limited_reissue_still_sends_the_frame(tmp_path):
    """Criterion 21, **adversarial**, and the most likely wiring mistake in this group.

    Two scene changes 200 ms apart with ``PrefillPolicy.reissue_min_interval_ms`` unchanged at
    1 000 ms: the machine asks for a re-issue both times, ``open_prefill`` suppresses both, and
    ``snapshot_frame`` — which has **no** rate limit (ruling A10) — runs both times. So two
    snapshots happen, one prefill is on the wire for the whole turn, and the final request must
    still carry **both** new screens plus the onset one.

    A build that appends the frame in the re-issue path rather than in ``snapshot_frame`` loses
    both of them here and passes every other row in this module, because every other row spaces
    its changes beyond the limit. That asymmetry is the point of this row.

    The 1 000 ms limit is the code's; docs/01 D9 says ~700 ms. The discrepancy predates this task
    and is deliberately not changed (task *Ambiguities* 3) — which is why nothing here reads the
    number to decide how to drive: 200 ms is inside either value.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        prefills_at_onset = len(rig.llm.prefills)
        await rig.advance(200, step=100)
        first = await announce(rig, "b")
        await rig.advance(200, step=100)
        second = await announce(rig, "c")
        await rig.words("compare those two")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert prefills_at_onset == 1, "the anticipatory prefill opens at onset (docs/01 D9)"
        snapshots = [c.at_ms for c in rig.machine.commands(C.SnapshotFrame)]
        assert len(snapshots) == 3, (
            "criterion 21: the rate limit is on the *prefill*, not on the snapshot — both changes "
            f"are captured. Got snapshots at {snapshots}"
        )
        assert len(rig.llm.prefills) == 1, (
            "criterion 21: and exactly one prefill reaches the backend, because both re-issues "
            f"fall inside reissue_min_interval_ms. Got {[r for r, _ in rig.llm.prefills]}"
        )
        _, request = rig.llm.gen_prompts[0]
        assert first is not None and second is not None
        assert phashes(request) == [onset_frame.phash, first.phash, second.phash], (
            "criterion 21: a suppressed re-issue never loses a frame — the frames belong to the "
            "turn, not to the prefill that happened to be sent. A build that appends only in the "
            f"re-issue path sends one image here. Got {phashes(request)}"
        )
        record = turn_record(rig, "t1")
        assert extra(record, "frames", "21") == [
            onset_frame.phash,
            first.phash,
            second.phash,
        ], "and the line agrees with the wire"
        assert extra(record, "scene_changes_seen", "21") == 2
        assert extra(record, "frames_kept", "21") == 3


# --- criterion 22: the cost of the feature ---------------------------------------------------------


async def test_no_extra_request_when_the_scene_does_not_change(tmp_path):
    """Criterion 22, counted on the fake LLM's calls and never on time.

    Three statements: an utterance with no scene change costs one prefill; a change to a screen
    the turn has already shown costs **nothing** — no snapshot, no re-issue, no frame; and the
    re-issues that reached the backend never outnumber the frames appended after the onset one.

    The repeat is announced beyond the rate limit, so a re-issue *would* go out if the machine
    asked for one: the zero here is the de-duplication, not the rate limiter. (Ruling A10's
    observation stands: membership is exact equality, so this row announces the identical hash,
    which is the only case the criterion claims.)
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        await rig.words("look at this")
        await rig.advance(1_200, step=200)
        assert len(rig.llm.prefills) == 1, (
            "criterion 22: nothing changed on screen, so nothing was re-sent to the engine; got "
            f"{[r for r, _ in rig.llm.prefills]}"
        )

        changed = await announce(rig, "b")
        assert changed is not None
        assert len(rig.llm.prefills) == 2, "the one real change re-warms the prefix once"

        await rig.advance(1_200, step=200)
        before = (len(rig.llm.prefills), len(rig.machine.commands(C.SnapshotFrame)))
        await announce(rig, phash=changed.phash)
        assert (len(rig.llm.prefills), len(rig.machine.commands(C.SnapshotFrame))) == before, (
            "criterion 22: a change back to a screen this turn already showed costs no snapshot "
            f"and no engine request; before {before}, after "
            f"{(len(rig.llm.prefills), len(rig.machine.commands(C.SnapshotFrame)))}"
        )

        await rig.words("and now")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert len(rig.llm.times("generate")) == 1, (
            f"one generate for the turn; got {rig.llm.times('generate')}"
        )
        record = turn_record(rig, "t1")
        assert extra(record, "frames", "22") == [onset_frame.phash, changed.phash], (
            "criterion 22: the repeat added no second copy of a screen already in the prompt; "
            f"got {record.extras.get('frames')!r}"
        )
        assert extra(record, "scene_changes_seen", "22") == 2, (
            "both events are counted even though one cost nothing — that is what makes the "
            "saving visible on the line"
        )
        reissues = len(rig.llm.prefills) - 1
        appended = extra(record, "frames_kept", "22") - 1
        assert reissues <= appended, (
            f"criterion 22: re-issues on the wire ({reissues}) must not outnumber the frames "
            f"appended after the onset one ({appended})"
        )


# --- criterion 18: the session's own de-duplication and its reset -----------------------------------


async def test_a_frame_refetched_with_the_same_hash_is_not_appended_twice(tmp_path):
    """Criterion 18's session-level de-duplication (D3: three places, three different reasons).

    ``PerceptionClient.get_frame(source, at_ms)`` answers with the latest frame at or before
    ``at_ms``, so two scene changes close together — or one announced for a screen the ring never
    captured — genuinely return the **same** ``Frame`` object. The machine cannot see that: it
    dedups on the announced ``SceneChanged.phash``, which is a different hash from a different
    source and may not even correspond to a frame the ring holds. If the session appends what it
    fetched without checking, the prompt carries the same pixels two or three times, each under a
    different position label, and the model is asked to compare a screen with itself.

    Three changes here, all with distinct announced hashes so the machine emits a snapshot for
    each: the first and the third refetch a frame the turn already holds, the second brings a new
    one. Two frames survive; three changes were seen.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        await rig.advance(1_200, step=200)
        # nothing new in the ring: get_frame answers with the onset frame again
        await announce(rig, phash="onset-again" * 2)
        await rig.advance(1_200, step=200)
        fresh = await announce(rig, "b")
        await rig.advance(1_200, step=200)
        # still nothing newer: the ring answers with ``fresh`` a second time
        await announce(rig, phash="fresh-again" * 2)
        await rig.words("what is this")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert fresh is not None
        assert len(rig.machine.commands(C.SnapshotFrame)) == 4, (
            "the machine dedups on the announced hash, which differs every time here, so it asks "
            f"for four snapshots; got {[c.at_ms for c in rig.machine.commands(C.SnapshotFrame)]}"
        )
        _, request = rig.llm.gen_prompts[0]
        assert phashes(request) == [onset_frame.phash, fresh.phash], (
            "criterion 18: the session dedups on the **fetched** Frame.phash, so the same screen "
            f"is one image however often it was refetched. Got {phashes(request)}"
        )
        record = turn_record(rig, "t1")
        assert extra(record, "frames", "18") == [onset_frame.phash, fresh.phash]
        assert extra(record, "frames_kept", "18") == 2
        assert extra(record, "scene_changes_seen", "18") == 3, (
            "criterion 20: seen counts the events, kept counts the screens — here they differ "
            "because of the dedup, which is one of the two things the pair diagnoses"
        )


async def test_the_turn_after_a_barge_in_starts_with_one_frame(tmp_path):
    """Criterion 18's reset, driven through a real takeover (T-0003 criterion 15).

    Turn one gathers two screens and reaches THINKING; the user barges in, which opens turn two
    with its own onset snapshot. Turn two must start from **one** frame: a ``_frames`` list that
    is not keyed on the turn id keeps growing across turns and the second question is answered
    with the screens of the first, which is the T-0059 bug with the sign flipped.

    The ordering trap this row also covers: the closing turn's line is written *after* the
    opening turn's commands have run (``Session._handle``), so the record of turn one must carry
    turn one's frames and not the frame turn two just snapshotted. The takeover's onset frame is
    a screen neither turn had seen, so a line that borrowed it is visible.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        await rig.advance(1_200, step=200)
        second_screen = await announce(rig, "b")
        await rig.words("first question")
        rig.llm.hold()
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.THINKING, "turn one never thought")

        await rig.advance(300, step=100)
        now = rig.clock.now_ms
        barge_frame = frame(now, "d")
        rig.perception.add(barge_frame)
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.turn_id == "t2" and rig.state.state is TurnState.LISTENING, (
            f"the takeover opens a new turn; got {rig.state.turn_id} {rig.state.state}"
        )
        rig.llm.release()
        await rig.words("second question")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn two never ended")

        assert second_screen is not None
        assert len(rig.llm.gen_prompts) == 2, f"two requests; got {len(rig.llm.gen_prompts)}"
        _, second_request = rig.llm.gen_prompts[1]
        assert phashes(second_request) == [barge_frame.phash], (
            "criterion 18: a new turn starts with no frames, so turn two carries only the screen "
            f"it snapshotted at its own onset. Got {phashes(second_request)}"
        )
        first_line = turn_record(rig, "t1")
        assert extra(first_line, "frames", "18") == [onset_frame.phash, second_screen.phash], (
            "criterion 18: the closing turn's line carries the closing turn's frames — its record "
            "is written after the *next* turn's snapshot has already run. Got "
            f"{first_line.extras.get('frames')!r}"
        )
        second_line = turn_record(rig, "t2")
        assert extra(second_line, "frames", "18") == [barge_frame.phash]
        assert extra(second_line, "frames_kept", "18") == 1
        assert extra(second_line, "scene_changes_seen", "18") == 0, (
            "the scene change belonged to turn one and is not re-counted against turn two"
        )


# --- criterion 23 (negative): a one-screen turn is exactly what it is today --------------------------


async def test_a_one_screen_turn_is_byte_identical_to_todays_prompt(tmp_path):
    """Criterion 23. One frame, no scene change: nothing about this turn may move.

    The expected prompt is **derived** from the builder's own constants and from the frame the
    test itself made (T-0053 ruling A3 — the system block is T-0055's and is being rewritten with
    a token count owed, so a copied literal would pin a value this task does not own). Its shape
    is the layout of docs/02 §5: one system message, then the current user message holding the
    image and then the transcript, with ``prefix_len`` at the D9 boundary between them.

    In particular there is **no label**: ``frame_label(0)`` is empty and an empty label is
    omitted (ruling A5), so the first image of a turn is unlabelled exactly as today. A build
    that labels the lone image ``[screen 1]`` makes the ordinary turn's cached prefix move — a
    silent prefix-cache miss on every turn, in exchange for nothing.

    The turn line is compared the same way: it may gain criterion 20's four keys and nothing
    else, and it may lose none.
    """
    onset_frame = frame(0, "a")
    transcript = "what is on this screen"
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.ask(transcript)
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        _, request = rig.llm.gen_prompts[0]
        system_text = "\n".join(
            [DEFAULT_SYSTEM_PROMPT, CLOSE_THE_THOUGHT, SPOKEN_STYLE, UI_INSTRUCTIONS]
        )
        expected = (
            Message(role="system", content=(TextPart(text=system_text),)),
            Message(
                role="user",
                content=(
                    ImagePart(url=data_url(onset_frame.jpeg), phash=onset_frame.phash),
                    TextPart(text=transcript),
                ),
            ),
        )
        assert request.messages == expected, (
            "criterion 23: a one-screen turn's prompt is byte-identical to today's — no label, "
            "no extra part, same order. Got "
            f"{[[p.model_dump_json()[:80] for p in m.content] for m in request.messages]}"
        )
        assert request.prefix_len == 2, (
            "and the D9 boundary is still inside the last user message, after the image and "
            f"before the transcript; got prefix_len={request.prefix_len}"
        )
        _, sent = rig.llm.prefills[-1]
        prefill_expected = (
            expected[0],
            Message(
                role="user",
                content=(ImagePart(url=data_url(onset_frame.jpeg), phash=onset_frame.phash),),
            ),
        )
        assert sent.messages == prefill_expected, "the prefill is today's prefill too"

        record = turn_record(rig, "t1")
        added = set(record.extras) - TODAYS_EXTRAS - T0108_EXTRAS - T0127_EXTRAS
        assert added == set(FRAME_EXTRAS), (
            "criterion 23: the turn line differs from today's by criterion 20's additive keys "
            f"and nothing else; it gained {sorted(added)}"
        )
        assert set(record.extras) >= TODAYS_EXTRAS, (
            "criterion 23, the other direction: no existing key may disappear; missing "
            f"{sorted(TODAYS_EXTRAS - set(record.extras))}"
        )
        assert len(rig.memory.episodes) == 1, (
            "criterion 23: Episode writing is unchanged — one episode per turn (D5)"
        )
        thumbnails = sorted((tmp_path / "thumbnails").glob("*.jpg"))
        assert len(thumbnails) == 1, (
            f"criterion 23: one thumbnail, the turn's frame; got {[p.name for p in thumbnails]}"
        )


# --- ruling A18 F1 + F2: the trim path, and what the extra screens cost the history ---------------


async def test_the_trim_path_keeps_three_current_frames_and_charges_the_evicted_history_image(
    tmp_path,
):
    """Ruling A18 F1 and F2, in one row: the long-history turn, which is where this feature is
    dearest and where both survivors hid.

    Two answered turns leave two history frames that still have pixels, then the third turn
    gathers three screens and its prefill reports a count over ``Budget.max_prompt_tokens``, so
    ``_turn_prompt`` goes down the ``fit()`` branch instead of extending the prefix in place.
    That branch is the one ``frame=self._frame`` survives in: the untrimmed path never reads the
    frames again, so a build that hands ``fit()`` one frame sends **one** image while its own
    turn line claims three, and only on the turns with enough history to trim.

    The arithmetic, by hand from the builder's own constants and D6/ruling A5:

    * two history turns, both inside ``image_age_turns`` and both resolvable → two history images
      *could* travel;
    * three current images take three of ``max_images`` (4), and the current turn's always win,
      so history keeps one slot: **one** history image is displaced by the frames;
    * the over-budget count spends exactly one rung, and rung 1 is history pixels (ruling A6), so
      ``prompt_trimmed["images"] == 1``;
    * the two must not be charged twice — the image the ladder spent is the ladder's, the one the
      frames displaced is the feature's — so ``history_images_evicted == 1``.

    F2 is the second half: every other row in this module runs without history pixels and asserts
    zero, so a build that simply writes ``history_images_evicted = 0`` passes all of them. Here
    zero is wrong.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, frames=[onset_frame]) as rig:
        rig.llm.prefill_counts(1_000, 1_000, Budget().max_prompt_tokens + 1)
        for tag, question in (("h", "what is in the first column"), ("i", "and the second")):
            await rig.advance(500, step=250)
            rig.perception.add(frame(rig.clock.now_ms, tag))
            rig.llm.script = [answer("A number. ")]
            await rig.ask(question)
            await rig.until(lambda: rig.state.state is TurnState.IDLE, f"turn {question!r}")
        assert len(rig.state.history) == 2, (
            f"regime: two history turns before the long one; got {len(rig.state.history)}"
        )

        rig.llm.script = [answer("The second one. ")]
        await rig.advance(500, step=250)
        current = [frame(rig.clock.now_ms, "p")]
        rig.perception.add(current[0])
        await rig.onset()
        for tag in ("q", "r"):
            await rig.advance(1_200, step=200)
            captured = await announce(rig, tag)
            assert captured is not None
            current.append(captured)
        await rig.words("which of these is later")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the long turn never ended")

        counts = [count for _, count in rig.llm.prefill_returns]
        assert counts[-1] > Budget().max_prompt_tokens, (
            "regime: the third turn's prefill must report an over-budget count, or fit() never "
            f"runs and this row tests the wrong branch; the engine reported {counts}"
        )
        _, request = rig.llm.gen_prompts[-1]
        assert [p.phash for p in request.messages[-1].content if isinstance(p, ImagePart)] == [
            f.phash for f in current
        ], (
            "ruling A18 F1: the trimmed prompt still carries all three current screens, in "
            "capture order — fit() spends history, never the turn (criterion 12 / D6). A build "
            "that hands fit() a single frame sends one image here and says three on its line. "
            f"Got {[p.phash for p in request.messages[-1].content if isinstance(p, ImagePart)]}"
        )
        record = rig.records()[-1]
        trimmed = record.extras.get("prompt_trimmed")
        assert isinstance(trimmed, dict) and trimmed.get("images") == 1, (
            "ruling A6: one token over budget spends exactly one rung and a rung is history "
            f"pixels; the line says prompt_trimmed={trimmed!r}"
        )
        assert extra(record, "history_images_evicted", "20") == 1, (
            "ruling A18 F2 / A5 finding 2: of the two history images that could have travelled, "
            "one was spent by the token ladder and one was displaced by the current frames — and "
            "they are not the same image, so the feature is charged exactly one. The line says "
            f"{record.extras.get('history_images_evicted')!r}"
        )
        assert extra(record, "frames_kept", "20") == 3, "and the turn kept its three screens"


# --- ruling A18 F3: scene_changes_seen counts LISTENING only ---------------------------------------


async def test_a_scene_change_while_thinking_is_not_counted_against_the_turn(tmp_path):
    """Ruling A18 F3. The count is what makes criterion 20's inference sound, or unsound.

    A source publishes scene changes whenever the screen moves — while the user speaks, while the
    assistant thinks, while it talks. The machine only acts on them in LISTENING (a change after
    the question is asked cannot change the prompt that was already built), so counting the
    others makes ``seen > kept`` true on every ordinary live turn and the key stops meaning "a
    screen was dropped". Counted in LISTENING only, ``seen > kept`` means exactly the cap or the
    dedup, which is what criterion 20 claims.

    One change while listening and two while thinking: the turn dropped nothing, so its line must
    read ``seen == 1`` against ``kept == 2`` — a number below the frame count, which is the shape
    of a turn that lost no screen.
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        await rig.advance(1_200, step=200)
        listened = await announce(rig, "b")
        await rig.words("what is this")
        rig.llm.hold()
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.THINKING, "the turn never thought")

        for tag in ("c", "d"):
            await rig.advance(200, step=100)
            await announce(rig, tag)
        assert rig.state.state is TurnState.THINKING, (
            "regime: both extra changes must land while the machine is THINKING; it is in "
            f"{rig.state.state}"
        )
        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert listened is not None
        record = rig.records()[-1]
        assert extra(record, "scene_changes_seen", "20") == 1, (
            "ruling A18 F3: only the change delivered in LISTENING is this turn's — the two that "
            "arrived while the assistant was thinking could not have reached the prompt and must "
            f"not be charged to it. The line says {record.extras.get('scene_changes_seen')!r}"
        )
        assert extra(record, "frames", "20") == [onset_frame.phash, listened.phash], (
            "and the prompt was built from the screens the user was actually looking at while "
            f"speaking; got {record.extras.get('frames')!r}"
        )
        assert extra(record, "frames_kept", "20") == 2
        assert extra(record, "scene_changes_seen", "20") <= extra(record, "frames_kept", "20"), (
            "criterion 20's inference, the negative direction: a turn that dropped nothing never "
            "reads seen > kept"
        )
        _, request = rig.llm.gen_prompts[0]
        assert phashes(request) == [onset_frame.phash, listened.phash], (
            f"and nothing the model saw came from after the question; got {phashes(request)}"
        )


async def test_a_change_back_to_a_screen_already_shown_is_seen_but_not_kept(tmp_path):
    """Ruling A18 F3's companion: the direction that must stay true.

    Two changes in LISTENING that announce the **onset frame's own hash**. The machine cannot
    suppress them — ``turn_phashes`` never holds the onset frame's hash, because no event carries
    it (D3) — so it snapshots both times; the session refetches the same ``Frame`` and its own
    de-duplication drops both. The turn keeps one screen and saw two changes, so its line reads
    ``seen > kept``, which under ruling A18's narrowing now means exactly one thing: a screen the
    user showed did not reach the model.

    (One such change gives ``seen == 1`` against ``kept == 1`` — equal, not greater; two are what
    make the inference bite, which is why this row drives two.)
    """
    onset_frame = frame(0, "a")
    async with running(tmp_path, script=[answer("Yes. ")], frames=[onset_frame]) as rig:
        await rig.onset()
        for _ in range(2):
            await rig.advance(1_200, step=200)
            await announce(rig, phash=onset_frame.phash)
        await rig.words("and now")
        await rig.end_of_turn()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        record = rig.records()[-1]
        assert extra(record, "scene_changes_seen", "20") == 2, (
            "both changes were delivered in LISTENING and both are counted, however little they "
            f"cost; the line says {record.extras.get('scene_changes_seen')!r}"
        )
        assert extra(record, "frames", "20") == [onset_frame.phash], (
            "criterion 18: the fetched frame is the screen the turn already holds, so nothing is "
            f"appended; got {record.extras.get('frames')!r}"
        )
        assert extra(record, "frames_kept", "20") == 1
        assert extra(record, "scene_changes_seen", "20") > extra(record, "frames_kept", "20"), (
            "ruling A18 F3: seen > kept, and after the narrowing it means the cap or the dedup "
            "and nothing else"
        )
        _, request = rig.llm.gen_prompts[0]
        assert phashes(request) == [onset_frame.phash], (
            f"one screen, one image, however often it was re-announced; got {phashes(request)}"
        )
