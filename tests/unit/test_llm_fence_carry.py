"""The UI fence carries across the canvases of one request — and ends. T-0052 group A.

Origin: the human's desk session of 2026-09-16 18:19–18:21 local (`logs/sessions/
20260916-181929-3900425.jsonl`, turn `t6`, 3678 records). The model built a flight-booking
interface, it did not fit in one generation canvas, and the second half — 750 characters of A2UI
JSON — was classified `answer` and read aloud by Pocket TTS for 807 characters.

Every row here drives the **real** ``VllmClient`` over an ``httpx.MockTransport`` that replays SSE
chunks, so the thing under test is the shipping client from the SSE line parser down. The
recorded canvases come from ``tests/fixtures/llm/flight_canvases.json`` (provenance in that
directory's ``MANIFEST.md``); the synthetic canvases are hand-written and their expected splits
are computed by hand in the docstring that uses them, never by running a scanner in the test.

Two failure directions, and the rows are written so they fail *separately*:

* JSON reaches ``answer`` → it is spoken. That is what happened.
* the carry swallows the prose after the JSON → the turn goes silent. That is what a client
  that only threaded the fence would have done on this very log, because the model never wrote a
  closing ``</a2ui>`` (ruling A1). Criterion 1 asserts both halves with distinct messages.

Regime: pure CPU, n = 1 per row, no seed, no timing claimed.
"""

from __future__ import annotations

from collections import deque

import httpx
import pytest

from local_astra.conductor import llm_client
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, VllmClient
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from tests.fakes.llm import (
    carried_ui_chars,
    fence_lifecycle,
    flight_canvases,
    sse_body,
    sse_transport,
)

FIX = flight_canvases()
REC = FIX["recorded"]
PROMPT = Prompt(
    messages=(Message(role="user", content=(TextPart(text="build me a flight booking ui"),)),),
    prefix_len=1,
)

#: The three canvases of turn t6 as ``(reasoning, content)`` SSE deltas.
FLIGHT = [(c["reasoning"], c["content"]) for c in FIX["canvases"]]


# --- helpers (no implementation logic lives here) ------------------------------------------------


async def run(
    canvases: list[tuple[str | None, str | None]],
    *,
    request_id: str = "t6-gen",
    terminate: bool = True,
) -> list[Commit]:
    """Replay ``canvases`` through the real client and collect every commit."""
    client = VllmClient(transport=sse_transport(canvases, terminate=terminate))
    return [c async for c in client.generate(PROMPT, request_id, None)]


def seg(commits: list[Commit], segment: str) -> list[Commit]:
    return [c for c in commits if c.segment == segment]


def text_of(commits: list[Commit], segment: str, *, canvas: int | None = None) -> str:
    return "".join(
        c.text
        for c in commits
        if c.segment == segment and (canvas is None or c.canvas_idx == canvas)
    )


def ui_canvases(commits: list[Commit]) -> list[int]:
    return sorted({c.canvas_idx for c in seg(commits, "ui")})


def last_ui(commits: list[Commit], *, why: str) -> Commit:
    """The final ``ui`` commit of a run, or a readable failure instead of an IndexError."""
    ui = seg(commits, "ui")
    assert ui, f"{why}: no ui commit was emitted at all, so the fence ending cannot be read"
    return ui[-1]


def bound(name: str) -> int:
    """A carry bound, read from the module (criterion 6 / D3 says they are constants)."""
    value = getattr(llm_client, name, None)
    assert isinstance(value, int) and value > 0, (
        f"llm_client.{name} must be a module constant a test can read (docs/tasks/T-0052.md D3): "
        f"got {value!r}"
    )
    return value


def junk_canvas(i: int, size: int) -> str:
    """JSON that opens more brackets than it closes: it has no structural end, ever.

    ``{"id":"cN","text":"xxx…","kids":[`` — one ``{`` and one ``[`` left open per canvas, so a
    depth scanner never returns to zero however many of these arrive. This is the shape criterion
    6's bound exists for.
    """
    return '{"id":"c' + str(i) + '","text":"' + "x" * size + '","kids":['


# --- criterion 1: tonight's turn, replayed -------------------------------------------------------


async def test_the_flight_surface_arrives_whole_and_the_prose_is_still_spoken() -> None:
    """Criterion 1 (a)(b)(c), the recorded turn through the real client.

    (a) is satisfied by a client that says nothing at all, and (c) by one that speaks JSON, so the
    three clauses carry distinct messages: a silent client and a JSON-speaking one must fail on
    different lines. The numbers are the fixture's, measured from the log: the fenced body is
    409 + 750 = 1159 characters and 152 characters of prose follow it inside canvas 2.
    """
    commits = await run(FLIGHT)

    # (a) nothing JSON-shaped may be classified answer — that is what reached Pocket TTS.
    spoken = [c for c in seg(commits, "answer") if '{"' in c.text or '"component"' in c.text]
    assert not spoken, (
        "clause (a): interface data was classified `answer`, i.e. handed to the speech gate and "
        "spoken — the 2026-09-16 bug verbatim. Offending commits: "
        + repr([(c.canvas_idx, c.text[:60]) for c in spoken])
    )

    # (b) the 750-character JSON tail of canvas 2 is ui, and the whole fenced body is accounted for.
    ui_two = text_of(commits, "ui", canvas=2)
    assert REC["json_tail_canvas2"] in ui_two, (
        "clause (b): the 750-character JSON tail of canvas 2 is not in that canvas's ui text; the "
        f"fence did not carry across the canvas boundary. ui text of canvas 2: {ui_two[:80]!r}"
    )
    # (c) the prose that follows the JSON *in the same canvas* is still spoken.
    prose = text_of(commits, "answer", canvas=2)
    assert prose, (
        "clause (c): the turn went SILENT. Carrying the fence without a structural close swallows "
        "the 152 characters of prose that follow the JSON, because the model never wrote a "
        "closing </a2ui> (docs/tasks/T-0052.md ruling A1). That is the opposite failure, not a fix."
    )
    assert prose == REC["prose_tail_canvas2"], (
        "clause (c): the spoken tail must be the prose byte for byte (152 characters). A scanner "
        "that stops at the first depth-zero instead of consuming the trailing run of closers "
        "leaves a stray '}' at the front of it, and Pocket TTS says 'close brace'. got "
        f"{prose[:60]!r}"
    )

    # (b), the arithmetic — asserted last on purpose, so a client that swallowed the prose fails
    # on clause (c)'s message ("the turn went SILENT") rather than on a character count.
    assert carried_ui_chars([c.text for c in seg(commits, "ui")]) == REC["fenced_body_chars"], (
        "clause (b): the fenced body is 409 + 750 = "
        f"{REC['fenced_body_chars']} characters (fixture, measured from the log); the client "
        f"carried {carried_ui_chars([c.text for c in seg(commits, 'ui')])}"
    )


# --- criterion 3: the carry is per request -------------------------------------------------------


async def test_fence_state_does_not_survive_a_request() -> None:
    """Criterion 3: one client, two ``generate`` calls; the first ends with the fence open.

    Two clauses with distinct messages, because they fail in opposite directions:

    * *inside* one request the fence carries — the second canvas of request 1 continues the JSON
      and must be ``ui``. Red today: that is the whole task;
    * *between* requests nothing survives — request 2 opens with prose and its first commit is
      ``answer``. State that outlives a request is a cross-turn leak (architecture §4: at most one
      active generate per session) and it is the failure amendment M7 named: one bad canvas and
      every later answer of the session is classified ``ui`` and never voiced. A retry after a
      400/413 is a new ``generate`` call and starts clean (D1).
    """
    bodies = deque(
        [
            sse_body([(None, f"Here. {UI_OPEN}" + '{"a":[1'), (None, ",2")]),
            sse_body([(None, "Four."), (None, " And that is all.")]),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions"), f"unexpected call to {request.url}"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=bodies.popleft()
        )

    client = VllmClient(transport=httpx.MockTransport(handler))
    first = [c async for c in client.generate(PROMPT, "r1", None)]
    assert [(c.canvas_idx, c.segment, c.text) for c in first if c.canvas_idx == 1] == [
        (1, "ui", '{"a":[1,2')
    ], (
        "inside one request the fence carries: the second canvas continues the JSON and is ui. "
        f"got {[(c.canvas_idx, c.segment, c.text) for c in first]}"
    )

    second = [c async for c in client.generate(PROMPT, "r2", None)]
    assert second and second[0].segment == "answer" and second[0].text == "Four.", (
        "the second request started inside the first request's fence: fence state must live in "
        f"the generate frame, never on the client. got {[(c.segment, c.text) for c in second]}"
    )
    assert not seg(second, "ui"), "no ui commit can come from a request that never opened a fence"


# --- criteria 4 and 5: the two ways a fence ends -------------------------------------------------


async def test_an_explicit_close_in_the_next_canvas_closes_the_fence() -> None:
    """Criterion 4. Hand-computed: canvas 1 is ``Here it is. <a2ui>{"version":"v0.9.1",
    "createSurface":`` (depth 1, no structural end) and canvas 2 is ``{"surfaceId":"s"}</a2ui>
    That is the panel.`` — depth goes 1 → 2 → 1, so it never reaches zero before the marker and
    only the *written* close can end this fence.
    """
    one = "Here it is. " + UI_OPEN + '{"version":"v0.9.1","createSurface":'
    two = '{"surfaceId":"s"}' + UI_CLOSE + " That is the panel."
    commits = await run([(None, one), (None, two)])

    assert len(seg(commits, "ui")) == 2, (
        "a fence opened in one canvas and closed in the next is two ui commits, one per canvas "
        f"(progressive render, T-0010): got {[(c.canvas_idx, c.text[:40]) for c in commits]}"
    )
    answers = [c.text for c in seg(commits, "answer")]
    assert answers == ["Here it is. ", " That is the panel."], (
        f"the text either side of the fence is the answer, verbatim: got {answers}"
    )
    for piece in answers:
        for fragment in [UI_OPEN[:k] for k in range(1, 7)] + [UI_CLOSE[:k] for k in range(1, 8)]:
            assert fragment not in piece, (
                f"a fence marker fragment {fragment!r} reached the answer channel: {piece!r}"
            )


async def test_a_fence_the_model_never_closed_ends_where_the_json_ends() -> None:
    """Criterion 5, on the recorded payload and on a minimal synthetic one.

    Synthetic, computed by hand: ``<a2ui>{"a":1} And that is the table.`` — depth 1 at ``{``, 0 at
    ``}``; the remainder after whitespace starts with ``A``, not ``{``, so the fence ends there.
    The recorded payload ends at character 1159 of the fenced body (750 of canvas 2).

    The commit must also *say* the close was inferred rather than written, so a reader can tell a
    surface the model finished from one the client guessed the end of. The row asserts that by
    comparing with the written-close run, not by asserting a spelling (criterion 12).
    """
    inferred = await run([(None, UI_OPEN + '{"a":1} And that is the table.')])
    assert [(c.segment, c.text) for c in inferred] == [
        ("ui", '{"a":1}'),
        ("answer", " And that is the table."),
    ], (
        "a fence with no closing marker ends at the structural end of the JSON; everything after "
        f"it is the answer. got {[(c.segment, c.text) for c in inferred]}"
    )

    recorded = await run(FLIGHT)
    assert text_of(recorded, "answer", canvas=2) == REC["prose_tail_canvas2"]
    assert REC["json_tail_canvas2"] in text_of(recorded, "ui", canvas=2)

    written = await run([(None, UI_OPEN + '{"a":1}' + UI_CLOSE + " And that is the table.")])
    assert [(c.segment, c.text) for c in written] == [
        ("ui", '{"a":1}'),
        ("answer", " And that is the table."),
    ], "control: the same payload with the close written must split identically"
    assert fence_lifecycle(last_ui(inferred, why="inferred close")) != fence_lifecycle(
        last_ui(written, why="written close")
    ), (
        "criterion 5: the commit must record that the close was *inferred* structurally rather "
        "than written by the model — today both commits say exactly the same thing, which is why "
        "'was this surface finished?' is unanswerable"
    )


async def test_a_canvas_that_ends_on_a_top_level_object_boundary_keeps_the_fence_open() -> None:
    """Adversarial, criterion 5 / ruling A28.1: the highest-value row here — it *is* the bug.

    An A2UI surface is a run of concatenated top-level messages, so a canvas boundary landing
    exactly on the ``}}`` that closes one of them is an ordinary shape, not a contrived one: the
    recorded turn is a ``createSurface`` followed by component messages. At that boundary the
    scanner has depth zero with an empty remainder, and it must **defer** — ``after >= size``
    returns ``None`` — because the next canvas decides whether the surface resumed. A scanner that
    treats it as an ending closes the fence there, the next canvas starts outside the fence, and
    its ``updateComponents`` message is classified ``answer`` and read aloud. That is the
    2026-09-16 failure verbatim (criterion 1 (a)), reached through a different door.

    Hand-computed: the fence opens in canvas 0 and does not end there; canvas 1's JSON joins the
    same body; the structural end falls after the second message because ``" Here it is."`` does
    not resume with ``{``, so only that prose is spoken. Verified both ways against the
    implementation: green as shipped; with the deferral mutated to ``return closers`` canvas 1's
    JSON is spoken and both clauses go red.
    """
    create = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","root":"r"}}'
    update = '{"version":"v0.9.1","updateComponents":{"id":"r","text":"Paris"}}'

    commits = await run([(None, UI_OPEN + create), (None, update + " Here it is.")])

    spoken = [c for c in seg(commits, "answer") if '{"' in c.text or '"version"' in c.text]
    assert not spoken, (
        "the canvas boundary fell on a top-level object boundary, the fence was declared finished "
        "there, and the next canvas of interface data went to the speech gate — the reported bug, "
        "reached through the empty-remainder deferral. Offending commits: "
        + repr([(c.canvas_idx, c.text[:60]) for c in spoken])
    )
    assert text_of(commits, "answer") == " Here it is.", (
        "only the prose after the last message may be spoken, verbatim. got "
        f"{[(c.canvas_idx, c.segment, c.text[:60]) for c in commits]}"
    )
    assert last_ui(commits, why="deferred boundary").text == create + update, (
        "both top-level messages belong to one surface: the second canvas continues the fence "
        "opened in the first. got "
        f"{last_ui(commits, why='deferred boundary').text[:120]!r}"
    )


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ('{"id":"t","text":"use } to close"}', "a closing brace inside a string value"),
        ('{"id":"t","text":"say \\"hi\\" }"}', "an escaped quote, then a brace, inside a string"),
    ],
    ids=["brace-in-string", "escaped-quote"],
)
async def test_a_brace_inside_a_string_does_not_close_the_fence(payload: str, why: str) -> None:
    """Adversarial, criterion 5: the most likely failure of a depth scanner.

    Both payloads are one complete JSON object whose *string value* contains a ``}``. A scanner
    that counts brackets without tracking strings closes the fence early, puts the rest of the
    object into ``answer``, and the assistant reads ``to close"} Done.`` aloud. The expected split
    is hand-computed: the object is the ui piece, ``" Done."`` is the answer.

    Neither payload pins the **backslash** rule, and the ``[escaped-quote]`` case used to claim it
    did (ruling A28.2): its ``}`` sits outside the string either way, so a scanner that ignores
    ``\\`` cuts in exactly the same place. The row below is the one that sees it.
    """
    commits = await run([(None, UI_OPEN + payload + " Done.")])
    assert [(c.segment, c.text) for c in commits] == [("ui", payload), ("answer", " Done.")], (
        f"{why}: the fence closed in the wrong place. got {[(c.segment, c.text) for c in commits]}"
    )


async def test_a_brace_just_after_an_escaped_quote_stays_inside_the_string() -> None:
    r"""Adversarial, criterion 5 / ruling A28.2: the one payload the backslash rule decides.

    ``{"t":"say \"}\" now"}`` — hand-traced: the string opens at ``"say``, the ``\`` marks the
    next ``"`` as text, so the ``}`` that follows it is *inside* the string; the string really
    closes at the ``"`` before ``}``, and the object's own ``}`` is the structural end. A scanner
    that drops the escape flag sees the escaped ``"`` close the string, reads the very next ``}``
    as the end of the object at depth zero, and hands ``\" now"} Done.`` to the speech gate: the
    assistant reads punctuation and JSON aloud. The brace must sit immediately after the escaped
    quote for the two scanners to disagree, which is why the parametrised row above cannot see it.

    Verified both ways against the implementation: green as shipped; red with ``escaped = True``
    mutated to ``escaped = False``.
    """
    payload = '{"t":"say \\"}\\" now"}'

    commits = await run([(None, UI_OPEN + payload + " Done.")])

    assert [(c.segment, c.text) for c in commits] == [("ui", payload), ("answer", " Done.")], (
        "a brace immediately after an escaped quote is string content: the fence ended early and "
        "the tail of the object was handed to the speech gate. got "
        f"{[(c.segment, c.text) for c in commits]}"
    )


async def test_a_run_of_closing_braces_never_drives_the_scanner_backwards() -> None:
    """Adversarial, criterion 5 / the landmine list: a surplus run of closers still ends the fence.

    The recorded payload ends ``…}]}]}}}}`` — more closers than openers, because the model wrote
    ``{"Text","text":…`` where a key belonged. What this row pins is the outcome: the fence *ends*
    inside the canvas, the object is the ui piece, and the prose is spoken rather than swallowed.

    It does **not** pin the scanner's depth clamp, and its docstring used to claim it did (rulings
    A25 and A29). Here the object's own ``}`` returns depth to zero and the scanner ends there
    before any surplus closer is read, so deleting ``max(0, depth - 1)`` leaves all three
    assertions below green. The clamp is pinned by the row that follows and by nothing else. Where
    the surplus closers land is still not pinned — either side of the boundary is defensible.
    """
    commits = await run([(None, UI_OPEN + '{"a":1}}}} Stray prose.'), (None, "Next canvas.")])

    ui = text_of(commits, "ui")
    answer = text_of(commits, "answer")
    assert ui.startswith('{"a":1}'), f"the object itself must be the ui piece; got {ui!r}"
    assert "Stray prose." in answer, f"the prose after the JSON must be spoken; got {answer!r}"
    assert answer.replace("}", "").strip() == "Stray prose.Next canvas.", (
        "nothing but a run of surplus closing braces may share the answer channel with the prose; "
        f"got {answer!r}"
    )
    assert text_of(commits, "answer", canvas=1) == "Next canvas.", (
        "the fence must be closed by the end of canvas 0: the next canvas is ordinary speech"
    )


async def test_a_closer_before_any_opener_cannot_drive_the_scanner_below_zero() -> None:
    """Adversarial, criterion 5 / rulings A25 and A29: the only row that pins the depth clamp.

    A closer that arrives *before* any opener is the one shape that asks ``max(0, depth - 1)`` to
    do anything, and it is reachable rather than theoretical: the fixture's own recorded payload
    contains a key-less object and a run of four closers, and D3's reopen path restarts the
    carried body empty, so a stray ``}`` can lead a carry. Without the clamp that closer takes
    depth to −1; the object's own ``}`` brings it back to 0, which is not the zero the scanner
    tests for — it never dropped *to* zero from above — so no structural end is ever found, the
    carry swallows the prose and the turn goes **silent**. That is the failure ruling A1 exists to
    prevent, and ruling A29 measured that no other row in the suite catches it.

    Hand-computed split: the stray closer belongs to the JSON piece, as the trailing run does in
    the row above, so ui is ``}{"a":1}`` and ``" Prose."`` is spoken. Verified both ways against
    the implementation: green as shipped; with the clamp removed (``depth = depth - 1``) the whole
    canvas becomes one open ``ui`` commit and both clauses go red.
    """
    commits = await run([(None, UI_OPEN + '}{"a":1} Prose.')])

    assert text_of(commits, "answer") == " Prose.", (
        "the turn went SILENT: a closing brace before any opener drove the scanner below zero, so "
        "depth zero was never reachable again, no structural end was found, and the carry "
        "swallowed the prose. got "
        f"{[(c.segment, c.text) for c in commits]}"
    )
    assert text_of(commits, "ui") == '}{"a":1}', (
        "the stray leading closer belongs to the surface, not to speech: a scanner that hands it "
        "to the answer channel makes Pocket TTS say 'close brace'. got "
        f"{[(c.segment, c.text) for c in commits]}"
    )


# --- criterion 6: the bound ----------------------------------------------------------------------


async def test_an_endless_fence_is_abandoned_at_the_canvas_bound_and_speech_resumes() -> None:
    """Adversarial, criterion 6: ten canvases that never close and never balance, then prose.

    Each canvas leaves one ``{`` and one ``[`` open, so neither the written nor the structural
    close can ever fire. Without a bound the carry eats the rest of the response and the turn is
    silent from here on — exactly what amendment M7 predicted and what naive threading ships.
    """
    max_canvases = bound("UI_FENCE_MAX_CANVASES")
    canvases: list[tuple[str | None, str | None]] = [(None, UI_OPEN + junk_canvas(0, 20))]
    canvases += [(None, junk_canvas(i, 20)) for i in range(1, 10)]
    canvases += [(None, "And that is the table.")]

    commits = await run(canvases)

    carrying = ui_canvases(commits)
    assert len(carrying) <= 1 + max_canvases, (
        f"the carry may span the opening canvas plus at most UI_FENCE_MAX_CANVASES "
        f"({max_canvases}) more; it spanned {len(carrying)} canvases: {carrying}"
    )
    assert text_of(commits, "answer", canvas=10) == "And that is the table.", (
        "after the carry is abandoned the answer channel is live again; the prose canvas that "
        "follows must be spoken, or the bound has not actually released the channel"
    )
    lifecycles = [fence_lifecycle(c) for c in seg(commits, "ui")]
    assert lifecycles, "the carry produced no ui commit at all"
    assert len(set(lifecycles)) >= 2, (
        "a subsystem that turns itself off owes a published state change (.claude/rules/"
        "conductor.md): the abandoned carry must be marked differently from a live one — every "
        f"ui commit currently says the same thing ({lifecycles[:1]})"
    )
    assert lifecycles.count(lifecycles[-1]) == 1, (
        f"exactly one ui commit is the abandonment, and it is the last one; got {lifecycles}"
    )
    assert carried_ui_chars([c.text for c in seg(commits, "ui")]) <= bound("UI_FENCE_MAX_CHARS")


async def test_an_endless_fence_is_abandoned_at_the_character_bound() -> None:
    """Adversarial, criterion 6, the *other* bound: canvases so large the character limit fires
    before the canvas limit. Sized from the constants, so it keeps discriminating if they move.

    Two canvases of ``UI_FENCE_MAX_CHARS * 0.6`` exceed the character bound while the canvas bound
    (4 further canvases) would still allow five. A client that counts canvases but not characters
    carries all five and is caught here; one that counts characters but not canvases is caught by
    the row above.
    """
    max_chars = bound("UI_FENCE_MAX_CHARS")
    max_canvases = bound("UI_FENCE_MAX_CANVASES")
    size = int(max_chars * 0.6)
    canvases: list[tuple[str | None, str | None]] = [(None, UI_OPEN + junk_canvas(0, size))]
    canvases += [(None, junk_canvas(i, size)) for i in range(1, 10)]
    canvases += [(None, "And that is the table.")]

    commits = await run(canvases)

    carrying = ui_canvases(commits)
    assert len(carrying) < 1 + max_canvases, (
        f"the character bound ({max_chars}) must fire before the canvas bound ({max_canvases}): "
        f"two canvases of {size} characters already exceed it, yet the carry spanned {carrying}"
    )
    carried = carried_ui_chars([c.text for c in seg(commits, "ui")])
    assert carried <= max_chars + size, (
        f"carried {carried} characters of fenced text with UI_FENCE_MAX_CHARS={max_chars}: the "
        "carry must stop at the first canvas that crosses the bound, not keep going"
    )
    assert text_of(commits, "answer", canvas=10) == "And that is the table.", (
        "speech must resume after the abandonment"
    )


# --- criterion 7: a response that ends with the fence open ---------------------------------------


@pytest.mark.parametrize("ending", ["done", "stream-close", "abort"])
async def test_a_response_that_ends_with_the_fence_open_publishes_what_it_has(ending: str) -> None:
    """Criterion 7 / D2: on ``[DONE]``, on a stream that just stops, and on ``abort``, the carried
    fenced text is committed to the *shown* channel and never re-read as prose.

    §4's "never speak text the LLM has not committed" is not weakened by this: the text *is*
    committed — to the channel the gate rules on. Discarding it instead would leave the panel
    showing half a surface with no record of why the rest never came (D2).

    The row does not pin *which* commit carries the ending, only that the ending is announced:
    the last ui commit must say something different from a cleanly closed fence.
    """
    open_fence = "Look. " + UI_OPEN + '{"a":[1,2'
    control = await run([(None, "Look. " + UI_OPEN + '{"a":[1,2]}' + UI_CLOSE)], request_id="ctl")

    if ending == "abort":
        client = VllmClient(transport=sse_transport([(None, open_fence)]))
        commits = []
        async for commit in client.generate(PROMPT, "r1", None):
            commits.append(commit)
            await client.abort("r1")
    else:
        commits = await run([(None, open_fence)], terminate=ending == "done")

    fenced = text_of(commits, "ui")
    assert '{"a":[1,2' in fenced, (
        f"[{ending}] the carried fenced text was dropped: a surface the gate could have salvaged "
        f"is gone and nothing records why. ui text: {fenced!r}"
    )
    for piece in seg(commits, "answer"):
        assert "{" not in piece.text and "[" not in piece.text, (
            f"[{ending}] carried fenced text was reclassified as answer and would be spoken: "
            f"{piece.text!r}"
        )
    assert fence_lifecycle(last_ui(commits, why=ending)) != fence_lifecycle(
        last_ui(control, why="control")
    ), (
        f"[{ending}] a response that ended with the fence open must be distinguishable from one "
        "that closed it; today both commits are identical, so the panel shows a truncated surface "
        "as though it were finished"
    )


# --- criterion 12: the commit says how the fence ended --------------------------------------------


async def test_the_commit_says_how_the_fence_ended() -> None:
    """Criterion 12 / D4: the five endings a reader has to tell apart, asserted as *distinct*
    rather than as spellings — the task pins the distinction, the implementer picks the field.

    Mapping to the criterion's list: ``one_canvas`` is "opened and closed in one canvas" and "a
    close the model wrote"; ``continued`` is "a ui commit that continues a fence opened in an
    earlier canvas" (and the model wrote that close too, so it must still differ from
    ``one_canvas``); ``inferred`` is the structural close; ``open`` is a response that ended open;
    ``abandoned`` is the bound. "Was this surface reassembled, abandoned, or never there?" is
    unanswerable while any two of these say the same thing.
    """
    one_canvas = await run([(None, UI_OPEN + '{"a":1}' + UI_CLOSE + " Bye.")])
    straddle = await run([(None, UI_OPEN + '{"a":[1'), (None, "2]}" + UI_CLOSE + " Bye.")])
    inferred = await run([(None, UI_OPEN + '{"a":1} Bye.')])
    left_open = await run([(None, UI_OPEN + '{"a":[1')])
    endless = await run(
        [(None, UI_OPEN + junk_canvas(0, 20))] + [(None, junk_canvas(i, 20)) for i in range(1, 10)]
    )

    endings = {
        name: fence_lifecycle(last_ui(run_, why=name))
        for name, run_ in (
            ("one_canvas", one_canvas),
            ("continued", straddle),
            ("inferred", inferred),
            ("open", left_open),
            ("abandoned", endless),
        )
    }
    duplicates = [(a, b) for a in endings for b in endings if a < b and endings[a] == endings[b]]
    assert not duplicates, (
        "these fence endings are indistinguishable on the Commit contract: "
        f"{duplicates}. Commit currently reports {sorted(endings.values())[0]!r} for all of them "
        "(criterion 12 / design decision D4)."
    )


async def test_a_newline_before_the_written_close_is_still_a_close_the_model_wrote() -> None:
    r"""Criterion 12 / ruling A26: the pretty-printed fence, the only shape the tie rule reaches.

    Every case in the row above puts ``</a2ui>`` flush against the JSON, where the structural end
    and the written marker fall on the same index and candidate order settles it: the tie rule is
    never asked. A model that pretty-prints writes ``}\n</a2ui>``, and there the structural end
    precedes the marker by whitespace. Earliest-ending-wins then records ``inferred`` — *the
    client guessed where this surface stopped* — for a model that plainly closed its own fence,
    which is the single question criterion 12 exists to answer; and the newline it left behind
    leads the spoken channel.

    Asserted as a relation rather than a spelling, like the row above: the pretty-printed ending
    must equal the flush one and differ from the genuinely inferred one. Verified both ways
    against the implementation: green as shipped; with the whitespace tie rule deleted from
    ``_fence_cut`` all three clauses go red.
    """
    spaced = await run([(None, UI_OPEN + '{"a":1}\n' + UI_CLOSE + " Bye.")])
    flush = await run([(None, UI_OPEN + '{"a":1}' + UI_CLOSE + " Bye.")])
    inferred = await run([(None, UI_OPEN + '{"a":1} Bye.')])

    assert text_of(spaced, "answer") == " Bye.", (
        "the newline the model put between its JSON and its own </a2ui> belongs to the surface; "
        "cutting at the structural end instead leaks it into the spoken channel. got "
        f"{[(c.segment, c.text) for c in spaced]}"
    )
    assert fence_lifecycle(last_ui(spaced, why="pretty-printed")) == fence_lifecycle(
        last_ui(flush, why="flush")
    ), (
        "a close the model wrote is a close the model wrote whether or not whitespace precedes "
        "it: the pretty-printed fence is recorded differently from the flush one, so the panel "
        "reports a finished surface as the client's guess"
    )
    assert fence_lifecycle(last_ui(spaced, why="pretty-printed")) != fence_lifecycle(
        last_ui(inferred, why="inferred")
    ), (
        "the pretty-printed fence is recorded exactly like one the model never closed at all, so "
        "criterion 12's whole question — did the model finish this surface, or did we guess? — "
        "has no answer for the commonest formatting the model emits"
    )


# --- adversarial: the reopened fence and the split marker ----------------------------------------


async def test_a_chunk_that_reopens_the_fence_while_open_is_defined() -> None:
    """Adversarial, D3: a second ``<a2ui>`` while a carry is open (a re-render, or two surfaces).

    D3's landing: the carried fence ends as abandoned and a new one opens. Whatever else happens,
    nothing JSON-shaped may fall out into ``answer`` — that is the whole bug — and the second
    surface must survive, because it is the one the model meant.
    """
    commits = await run(
        [
            (None, "Look. " + UI_OPEN + '{"a":[1'),
            (None, UI_OPEN + '{"b":2} Done.'),
        ]
    )

    for piece in seg(commits, "answer"):
        assert "{" not in piece.text and "[" not in piece.text, (
            f"a reopened fence spilled interface data into the answer channel: {piece.text!r}"
        )
    assert '{"b":2}' in text_of(commits, "ui"), (
        "the surface the second fence opened must still be published; got "
        f"{[(c.canvas_idx, c.segment, c.text) for c in commits]}"
    )
    assert text_of(commits, "answer", canvas=1) == " Done.", (
        "the prose after the second fence's structural end is the answer, verbatim"
    )
    lifecycles = [fence_lifecycle(c) for c in seg(commits, "ui")]
    assert len(set(lifecycles)) >= 2, (
        f"the abandoned first carry must be announced as such, not silently replaced: {lifecycles}"
    )


@pytest.mark.parametrize("marker", [UI_OPEN, UI_CLOSE], ids=["open", "close"])
@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6])
async def test_a_marker_split_across_the_chunk_boundary_is_never_voiced(
    marker: str, cut: int
) -> None:
    """Adversarial: every interior split point of ``<a2ui>`` (5) and ``</a2ui>`` (6).

    A canvas can end anywhere, including inside a marker. Today ``_strip_markup`` *drops* the tail
    fragment, which is safe only because the fence state is thrown away with it; with a carry the
    fragment must be **held** and prepended to the next canvas, or the client either voices
    ``ui>`` or loses the opening of a real surface. The fragment must never appear in any answer
    text, whichever marker it belongs to.
    """
    if cut >= len(marker):
        pytest.skip(f"{marker!r} has no interior split point at {cut}")
    if marker == UI_OPEN:
        canvases = [
            (None, "Here. " + marker[:cut]),
            (None, marker[cut:] + '{"a":1}' + UI_CLOSE + " Done."),
        ]
        expected_answers = ["Here. ", " Done."]
    else:
        canvases = [
            (None, "Here. " + UI_OPEN + '{"a":[1]' + marker[:cut]),
            (None, marker[cut:] + " Done."),
        ]
        expected_answers = ["Here. ", " Done."]

    commits = await run(canvases)

    answers = [c.text for c in seg(commits, "answer")]
    for piece in answers:
        assert "a2u" not in piece and "<" not in piece and ">" not in piece, (
            f"a fragment of {marker!r} split at {cut} reached the answer channel: {piece!r}"
        )
    assert answers == expected_answers, (
        f"the marker split at {cut} must be reassembled across the canvas boundary and the prose "
        f"either side spoken verbatim; got {[(c.segment, c.text) for c in commits]}"
    )
    assert '{"a":1}' in text_of(commits, "ui") or '{"a":[1]' in text_of(commits, "ui"), (
        "the surface itself must survive the split marker"
    )


async def test_a_held_fragment_that_never_becomes_a_marker_is_dropped_not_voiced() -> None:
    """Adversarial, ruling A28.3: the held fragment the next canvas does *not* complete.

    Outside the fence a trailing ``<a2u`` is held, because it may be the start of ``<a2ui>``. When
    the next canvas begins with ordinary prose instead, the fragment was markup all along — the
    model wrote a stray angle bracket — and it must be dropped exactly as ``_strip_markup`` dropped
    it before the carry existed (L10). Keep it and the assistant reads ``Hello <a2uplain prose
    after.`` aloud: markup in the speech channel, which is the rule that clause exists to preserve.

    Every marker-split row above uses a fragment that *does* complete, so none of them reaches the
    drop. Verified both ways against the implementation: green as shipped; red with the clause
    mutated to ``rest = merged``, which voices the fragment.
    """
    commits = await run([(None, "Hello <a2u"), (None, "plain prose after.")])

    assert text_of(commits, "answer") == "Hello plain prose after.", (
        "a held fragment that the next canvas did not turn into a marker must be dropped, not "
        "spoken: markup reached the speech gate. got "
        f"{[(c.canvas_idx, c.segment, c.text) for c in commits]}"
    )
    assert not seg(commits, "ui"), (
        "no fence ever opened here, so nothing may be published to the surface channel. got "
        f"{[(c.canvas_idx, c.segment, c.text) for c in commits]}"
    )
