"""T-0134 A1, A3, A5, A6, A7 (A-label): the "this is the screen now" label in live requests, over the fakes
(docs/tasks/T-0134.md).

**The surface these rows pin** (named in the task; where it is silent the choice is fixed here):

- ``Session(..., live_feedback: LiveFeedback | None = None)``; ``session.LiveFeedback`` is a frozen dataclass with
  ``screen_label`` and ``page_extent`` (``_live_browser_support.live_feedback``);
- the label is one ``TextPart`` immediately after the current frame's ``ImagePart`` in the request's final ``user``
  message; on the wire (``VllmClient.request_body``) that is one ``{"type": "text"}`` part right after the
  ``image_url`` part whose URL is the builder's data URL of the frame (``builder_url``, T-0111g's oracle);
- the A5 turn-line key lives in ``TurnRecord.extras`` (like T-0111g's live keys): ``extras["live_feedback"] ==
  {"screen_label", "page_extent", "echo"}``;
- A5's per-request record: a ``note`` or an ``llm.request`` record whose ``data`` holds the label text as a value
  and the keys ``above_px``, ``below_px``, ``viewport_px`` (anywhere in ``data``; ``None`` allowed). The rows count
  such records, one per labelled request.

Exact texts, typed from A3: ``L1``, ``LU`` (after ``screenshot: "updated"``), ``LN`` (after ``"unchanged"``); with
``page_extent`` off ``{P}`` is empty.

Readings (reported): a turn whose frame rides on the newest history image (its final user message holds no image) has
no current frame in its final user message at step 1, so step 1 carries no label (A3: "never on a history image");
from step 2 the frame is placed first (``show_current_image``) and the label follows it. A captioner request may not
happen in a rig turn; the caption part of ``test_no_label_in_prefill_caption_or_pass2`` checks every caption prompt
there is.

Every row but none is ``pending(task="T-0134")``: at BASE the rig fails on the missing keyword (a named
``AssertionError``).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), 1280x800
frames, n = 1, deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts.llm import ImagePart, TextPart
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import is_caption_prompt
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FakeLiveBrowser,
    builder_url,
    feedback_rig,
    jpeg_frame,
    live_feedback,
    live_frame,
    refused,
)
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    tool,
)
from tests.unit.test_session_live import (
    CLICK2,
    S1,
    S2,
    S3,
    _wire,
    image_urls,
    last_user,
    pass1_body,
    step_body,
)

L1 = "[This screenshot is the browser screen now.]"
LU = "[This screenshot is the browser screen now, after your last action.]"
LN = "[This screenshot is the browser screen now; your last action did not change it.]"
#: every label begins so (A3); a substring of none of the rig's other texts
STEM = "This screenshot is the browser screen now"
CLAUSE = "Page beyond this screen"

F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")


def label_on() -> Any:
    return live_feedback(screen_label=True, page_extent=False)


# --------------------------------------------------------------------------------------------- reading bodies


def texts(body: dict[str, Any]) -> list[str]:
    """Every text of the body (string contents and text parts), in order."""
    out: list[str] = []
    for m in body["messages"]:
        c = m["content"]
        if isinstance(c, str):
            out.append(c)
        else:
            out.extend(str(p.get("text", "")) for p in c if p.get("type") == "text")
    return out


def labels(body: dict[str, Any]) -> list[str]:
    """The text parts that are labels (start with ``[`` + STEM), anywhere in the body."""
    out = []
    for m in body["messages"]:
        c = m["content"]
        if isinstance(c, list):
            out.extend(
                p["text"]
                for p in c
                if p.get("type") == "text" and str(p.get("text", "")).startswith("[" + STEM)
            )
    return out


def stem_count(body: dict[str, Any]) -> int:
    return json.dumps(body).count(STEM)


def label_after(body: dict[str, Any], frame: Any) -> str:
    """A3: the body's final user message holds ``frame``'s image part, and the part right after it is a text part;
    returns that text. Also checks there is exactly one label in the whole body."""
    user = last_user(body)
    parts = user["content"]
    assert isinstance(parts, list), parts
    url = builder_url(frame)
    at = [
        i
        for i, p in enumerate(parts)
        if p.get("type") == "image_url" and p["image_url"]["url"] == url
    ]
    assert len(at) == 1, (
        f"premise: the current frame is in the final user message once; parts {_kinds(parts)}"
    )
    i = at[0]
    assert i + 1 < len(parts) and parts[i + 1].get("type") == "text", (
        f"T-0134 A3: no text part right after the current frame; parts {_kinds(parts)}"
    )
    assert stem_count(body) == 1, (
        f"T-0134 A3: exactly one label per request; found {stem_count(body)} ({labels(body)})"
    )
    return str(parts[i + 1]["text"])


def _kinds(parts: list[dict[str, Any]]) -> list[str]:
    return [
        p["type"] if p["type"] != "text" else "text:" + str(p.get("text", ""))[:40] for p in parts
    ]


def all_bodies(rig: Any) -> list[dict[str, Any]]:
    """Every generate's wire body (step 1, steps, pass 2, captions), rebuilt by the real client."""
    return [_wire(p) for _, p, _ in rig.llm.generates]


async def three(rig: Any, steps: tuple[Any, ...], question: str = tsl.QUESTION) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    await tsl.until_steps(rig, len(steps), limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)


async def next_turn(
    rig: Any, n: int, question: str, script: list[Any], steps: tuple[Any, ...] = ()
) -> None:
    rig.llm.script = list(script)
    if steps:
        rig.llm.script_steps(*steps)
    await rig.ask(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"turn {n} never ended",
        limit_ms=20_000,
    )
    await rig.advance(300)


# =====================================================================================================
# A3: the label
# =====================================================================================================


async def test_step1_label_after_frame(tmp_path: Path) -> None:
    """A3 (reviewer mutation 1, label before the image): step 1's request carries exactly ``L1``, once, as the part
    right after the onset frame; the user's question follows unchanged."""
    browser = FakeLiveBrowser([F0])
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer("It is a shop. ")],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        body = pass1_body(rig)
        assert label_after(body, F0) == L1
        parts = last_user(body)["content"]
        assert [p["type"] for p in parts] == ["image_url", "text", "text"], _kinds(parts)
        assert parts[2]["text"] == "what is on the screen", (
            "the question is unchanged after the label"
        )


async def test_step_k_updated_and_unchanged(tmp_path: Path) -> None:
    """A3: after an executed click whose payload says ``"updated"`` step 2 carries ``LU`` after the post-action frame
    F1; after a refused click (``"unchanged"``) step 3 carries ``LN`` after the same F1. The payload's ``screenshot``
    key is unchanged (T-0111g B8). T-0187 (owned re-scope): the refusal reason is one outside the "take a new frame"
    set."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(pass1_body(rig), F0) == L1
        s2, s3 = step_body(rig, 2), step_body(rig, 3)
        assert label_after(s2, F1) == LU
        assert label_after(s3, F1) == LN
        tail2, tail3 = tsl.tail(tsl.step_request(rig, 2)), tsl.tail(tsl.step_request(rig, 3))
        assert '<|"|>updated<|"|>' in tail2 and '<|"|>unchanged<|"|>' in tail3, (
            "premise: the payloads say updated, then unchanged"
        )
        assert STEM not in tail2 and STEM not in tail3, (
            "R-b: the label is never in the continuation"
        )


async def test_label_follows_frame_placed_first(tmp_path: Path) -> None:
    """A3, the ``show_current_image`` branch: turn 2's onset frame has turn 1's bytes, so it rides on the history
    image and turn 2's own message holds no image. Step 1 of turn 2 then has no label (reading in the docstring);
    step 2's final user message is ``[F1, LU, <its own parts>]``; the history image has no label after it."""
    again = F0.model_copy(update={"ts_ms": 77_777})
    browser = FakeLiveBrowser([F0, again], post_frames=[F1])
    async with feedback_rig(
        tmp_path, browser, script=[answer("A shop. ")], session_kwargs={"live_feedback": label_on()}
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(
            rig,
            2,
            tsl.QUESTION,
            [answer(S1), tool(VALID_CLICK)],
            ([answer(S2), tool(CLICK2)], [answer(S3)]),
        )
        p1 = pass1_body(rig, "t2")
        assert image_urls(p1) == [builder_url(F0)], "premise: one image, the history one"
        assert "image_url" not in [p["type"] for p in last_user(p1)["content"]], (
            "premise: rides on history"
        )
        assert stem_count(p1) == 0, f"no label on a history image; {labels(p1)}"
        s2 = step_body(rig, 2, "t2")
        parts = last_user(s2)["content"]
        assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"] == builder_url(
            F1
        ), _kinds(parts)
        assert parts[1] == {"type": "text", "text": LU}, _kinds(parts)
        assert parts[2:] == last_user(p1)["content"], "the message's own parts follow unchanged"
        assert stem_count(s2) == 1, labels(s2)


async def test_no_label_on_history_image_or_blind_turn(tmp_path: Path) -> None:
    """A3 (reviewer mutation 2): turn 2 (a new frame F2, F0 in history) carries one label, after F2, none after
    F0. A blind turn (the onset ``frame()`` fails) carries no image and no label at step 1 or step 2."""
    browser = FakeLiveBrowser([F0, F2])
    async with feedback_rig(
        tmp_path, browser, script=[answer("A shop. ")], session_kwargs={"live_feedback": label_on()}
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(rig, 2, "and now", [answer("Still a shop. ")])
        body = pass1_body(rig, "t2")
        assert image_urls(body) == [builder_url(F0), builder_url(F2)], (
            "premise: history image and current"
        )
        assert label_after(body, F2) == L1
        history_users = [m for m in body["messages"] if m["role"] == "user"][:-1]
        for m in history_users:
            if isinstance(m["content"], list):
                assert not any(
                    str(p.get("text", "")).startswith("[" + STEM) for p in m["content"]
                ), m
    blind = FakeLiveBrowser([], post_frames=[F1])
    blind.frame_mode = "raise"
    scroll = '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
    async with feedback_rig(
        tmp_path / "blind",
        blind,
        script=[answer(S1), tool(scroll)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await three(rig, ([answer(S3)],))
        for b in (pass1_body(rig), step_body(rig, 2)):
            assert image_urls(b) == [], "premise: blind"
            assert stem_count(b) == 0, f"T-0134 A3: a blind turn gets no label; {labels(b)}"


async def test_no_label_in_prefill_caption_or_pass2(tmp_path: Path) -> None:
    """A3 (reviewer mutation 5, the label sent in the prefill): with the label on and ReGround on, no prefill, no
    caption request and no ReGround pass-2 request carries a label; step 1 does (premise)."""
    browser = FakeLiveBrowser([jpeg_frame("onset")], post_frames=[jpeg_frame("after-1")])
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on(), "reground": True},
        reground_llm=True,
    ) as rig:
        rig.llm.script_pass2(
            [answer(""), tool("<|tool_call>call:click_at{x:900,y:100}<tool_call|>")]
        )
        rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(2_000)
        assert stem_count(pass1_body(rig)) == 1, "premise: step 1 carries the label"
        assert rig.llm.prefills, "premise: a prefill was sent"
        for rid, p in rig.llm.prefills:
            assert STEM not in json.dumps(_wire(p)), (
                f"T-0134 A3/A7: the prefill {rid} carries a label"
            )
        assert rig.llm.pass2_prompts, "premise: pass 2 ran"
        for rid, p in rig.llm.pass2_prompts:
            assert STEM not in json.dumps(_wire(p)), (
                f"T-0134 A3: the pass-2 request {rid} carries a label"
            )
        for rid, p, _ in rig.llm.generates:
            if is_caption_prompt(p):
                assert STEM not in json.dumps(_wire(p)), (
                    f"T-0134 A3: the caption request {rid} carries a label"
                )


async def _prefills(tmp: Path, feedback: Any) -> list[str]:
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    kwargs = {"live_feedback": feedback} if feedback is not None else {}
    async with feedback_rig(
        tmp, browser, script=[answer("A shop. ")], session_kwargs=kwargs
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        return [json.dumps(_wire(p), sort_keys=True) for _, p in rig.llm.prefills]


async def test_prefill_body_identical(tmp_path: Path) -> None:
    """A7: for the same frame the anticipatory prefill bodies are byte-identical with the label on and with
    ``live_feedback=None`` (the label lives after the frame, in the query-dependent suffix)."""
    on = await _prefills(tmp_path / "on", label_on())
    off = await _prefills(tmp_path / "off", None)
    assert on and on == off, "T-0134 A7: the prefill changed with the label on"
    assert STEM not in "".join(on)


# =====================================================================================================
# A6: never spoken, never stored
# =====================================================================================================


async def test_label_never_voiced_or_stored(tmp_path: Path) -> None:
    """A6 (adversarial): with the label and the extent on (a sentinel extent of 7.3 screens below), across a
    three-step turn and a second turn, no ``TtsSpeak`` (text or source), no history entry, no HUD message, no
    ``Episode`` and no machine event carries the label stem, the clause or the sentinel number; the requests do
    (premise)."""
    browser = FakeLiveBrowser([F0, F2], post_frames=[F1])
    browser.extents = {
        f.phash: {"above_px": 0.0, "below_px": 5840.0, "viewport_px": 800.0} for f in (F0, F1, F2)
    }
    feedback = live_feedback(screen_label=True, page_extent=True)
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": feedback},
    ) as rig:
        await three(rig, ([answer(S2)],))
        await next_turn(rig, 2, "and now", [answer("Still a shop. ")])
        sent = json.dumps(all_bodies(rig))
        assert STEM in sent and "about 7.3 screens below" in sent, (
            "premise: the requests carry label and clause"
        )
        needles = (STEM, CLAUSE, "7.3 screens")
        speaks = [c.text + "|" + c.source for c in rig.machine.commands(C.TtsSpeak)]
        assert speaks, "premise: something was spoken"
        stores = {
            "TtsSpeak": json.dumps(speaks),
            "history": tsl.history_text(rig),
            "HUD": tsl.published_json(rig),
            "Episode": json.dumps(
                [e.model_dump(mode="json") for e in rig.memory.episodes.values()]
            ),
            "machine.events()": json.dumps([repr(e) for e in rig.machine.events()]),
        }
        for where, text in stores.items():
            for n in needles:
                assert n not in text, f"T-0134 A6: {n!r} reached {where}"


# =====================================================================================================
# A5: logging
# =====================================================================================================


def _label_records(rig: Any) -> list[dict[str, Any]]:
    """A5: note / llm.request records whose data holds a label text as a value and the three extent keys."""

    def walk(v: Any) -> tuple[list[str], set[str]]:
        found, keys = [], set()
        if isinstance(v, dict):
            for k, x in v.items():
                keys.add(k)
                f, ks = walk(x)
                found += f
                keys |= ks
        elif isinstance(v, list):
            for x in v:
                f, ks = walk(x)
                found += f
                keys |= ks
        elif isinstance(v, str) and v.startswith("[" + STEM):
            found.append(v)
        return found, keys

    out = []
    for r in rig.log_records:
        if r["kind"] in ("note", "llm.request"):
            found, keys = walk(r["data"])
            if found and {"above_px", "below_px", "viewport_px"} <= keys:
                out.append(r)
    return out


async def _two_step(tmp: Path, kwargs: dict[str, Any], echo: Any = "model") -> Any:
    browser = FakeLiveBrowser([F0], post_frames=[F1], echo=echo)
    async with feedback_rig(
        tmp, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs
    ) as rig:
        await three(rig, ([answer(S3)],))
    return rig


async def test_turn_line_key(tmp_path: Path) -> None:
    """A5: with ``live_feedback`` given every turn line's extras carry ``live_feedback == {"screen_label": True,
    "page_extent": False, "echo": "model"}`` (``echo`` from the browser by duck typing; ``None`` when the browser
    has none); with ``live_feedback=None`` given explicitly no key is added. Each labelled request has one record
    with its label text and the extent keys (``None`` here: the extent is off). No new log kind and no new span."""
    rig = await _two_step(tmp_path / "on", {"live_feedback": label_on()})
    assert rig.record("t1").extras.get("live_feedback") == {
        "screen_label": True,
        "page_extent": False,
        "echo": "model",
    }, rig.record("t1").extras
    recs = _label_records(rig)
    labelled = [b for b in all_bodies(rig) if stem_count(b)]
    assert len(labelled) == 2, "premise: step 1 and step 2 carry labels"
    assert len(recs) == len(labelled), f"T-0134 A5: one label record per labelled request; {recs}"
    no_echo = await _two_step(tmp_path / "noecho", {"live_feedback": label_on()}, echo=None)
    assert no_echo.record("t1").extras.get("live_feedback") == {
        "screen_label": True,
        "page_extent": False,
        "echo": None,
    }
    off = await _two_step(tmp_path / "off", {"live_feedback": None})
    assert "live_feedback" not in off.record("t1").extras
    assert _label_records(off) == [] and all(stem_count(b) == 0 for b in all_bodies(off))
    kinds_on = {r["kind"] for r in rig.log_records}
    kinds_off = {r["kind"] for r in off.log_records}
    assert kinds_on == kinds_off, f"T-0134 A5: no new log kind; {sorted(kinds_on ^ kinds_off)}"
    spans_on = set(json.loads(rig.lines()[0])["spans"])
    spans_off = set(json.loads(off.lines()[0])["spans"])
    assert spans_on == spans_off, f"T-0134 A5: no new span; {sorted(spans_on ^ spans_off)}"


# =====================================================================================================
# A1: the keywords
# =====================================================================================================


async def test_keywords_refused(tmp_path: Path) -> None:
    """A1: ``page_extent=True`` without the label raises ``ValueError`` (in ``LiveFeedback`` or in ``Session``); a
    ``live_feedback`` with a ``dry_run`` kit raises ``ValueError`` naming ``live`` (no silent no-op)."""
    try:
        bad = live_feedback(screen_label=False, page_extent=True)
    except ValueError:
        bad = None
    if bad is not None:
        with pytest.raises(ValueError):
            async with feedback_rig(
                tmp_path / "x", FakeLiveBrowser([F0]), session_kwargs={"live_feedback": bad}
            ):
                pass
    with pytest.raises(ValueError, match="live"):
        async with feedback_rig(
            tmp_path / "dry",
            FakeLiveBrowser([F0]),
            session_kwargs={"live_feedback": label_on()},
            kit=T.make_kit(),
        ):
            pass
    # companion (the guard is not too wide): a valid live session starts
    async with feedback_rig(
        tmp_path / "ok", FakeLiveBrowser([F0]), session_kwargs={"live_feedback": label_on()}
    ) as rig:
        assert rig.session is not None


def test_label_texts_are_the_criterion() -> None:
    """Not pending (a check on this module): the literals are A3's, so a typo here cannot hide a wrong label."""
    assert L1 == "[This screenshot is the browser screen now" + "." + "]"
    assert re.fullmatch(
        r"\[This screenshot is the browser screen now(\.|, after your last action\.|; your last "
        r"action did not change it\.)\]",
        LU,
    ) and LN.endswith("did not change it.]")
    assert isinstance(TextPart(text=L1), TextPart) and ImagePart


# =====================================================================================================
# Review A of group A (2026-10-22): o20 and the repair record
# =====================================================================================================


async def test_arm_b_known_extents_no_clause(tmp_path: Path) -> None:
    """Review A o20 (mutant: the clause computed although ``page_extent`` is False): arm B's feedback (label on,
    extent off) over a browser whose every frame has a **known** extent gives exactly L1, LU and LN, with no
    ``Page beyond this screen`` clause anywhere, and ``extent_of`` is never consulted; the label records carry
    ``None`` extent numbers. T-0187 (owned re-scope): the refusal reason is one outside the "take a new frame" set."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    known = {"above_px": 400.0, "below_px": 1200.0, "viewport_px": 800.0}
    browser.extents = {f.phash: dict(known) for f in (F0, F1, F2)}
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(pass1_body(rig), F0) == L1
        assert label_after(step_body(rig, 2), F1) == LU
        assert label_after(step_body(rig, 3), F1) == LN
        sent = json.dumps(all_bodies(rig))
        assert CLAUSE not in sent, "T-0134 A4: no clause without page_extent"
        assert not [c for c in browser.calls if c[0] == "extent_of"], browser.calls
    for r in _label_records(rig):
        found = json.dumps(r["data"])
        assert '"above_px": null' in found and '"viewport_px": null' in found, found


async def test_repair_inherits_pass1_label_and_record(tmp_path: Path) -> None:
    """Review A, A5 (ruling: the repair inherits pass 1's label and record): a live turn whose step-1 call is
    repairable (H-T84-2) sends a ``tool_repair`` request whose body carries the step-1 label L1 once, right after
    the onset frame, and whose ``llm.request`` record carries ``live_label`` equal to pass 1's (``generate``)
    record: the same text and the same px keys."""
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(REPAIRABLE_CLICK)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        rig.llm.script_continuation(tool(REPAIRED_CLICK))
        await three(rig, ([answer(S3)],))
        reps = rig.llm.continuations()
        assert reps, "premise: a repair request was sent"
        body = _wire(reps[0][1])
        assert stem_count(body) == 1 and L1 in labels(body), labels(body)
    reqs = [r["data"] for r in rig.log_records if r["kind"] == "llm.request"]
    gen = [d for d in reqs if d.get("phase") == "generate"]
    rep = [d for d in reqs if d.get("phase") == "tool_repair"]
    assert gen and rep, (
        f"premise: generate and tool_repair records; {[d.get('phase') for d in reqs]}"
    )
    assert gen[-1].get("live_label", {}).get("text") == L1, gen[-1].get("live_label")
    assert rep[-1].get("live_label") == gen[-1]["live_label"], (
        f"T-0134 A5: the repair record inherits pass 1's live_label; got {rep[-1].get('live_label')}"
    )
