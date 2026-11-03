"""Builds ``session_claims.jsonl``: a synthetic session log for ``scripts/replay_completion_check.py`` (T-0207 A1.5).

Run: ``uv run python tests/fixtures/t0207/build_claims.py > tests/fixtures/t0207/session_claims.jsonl``.

The record shapes are those of ``logs/sessions/20261029-201813-2716251.jsonl`` (read only, for the shapes):
``llm.request`` (``phase``, ``step``, ``request_id``, ``attempt_id``, ``options``, ``prompt``, ``live_label``,
``report_fact``), ``llm.commit`` (keyed by ``attempt_id``), ``llm.done``, ``tool.call`` (``status``,
``resolved.frame_after``), ``tool.result``, ``tts.utterance`` and ``turn.record`` (``extras.vision_source``). The
prompts are built with the real ``loop.step_prompt``, ``loop.with_report_fact``, ``announced_call.reask_prompt``,
``session.log_dump`` and ``session_log.redact``, so the images are redacted exactly as the session log redacts them
(``phash`` kept). No real user utterance, page text or URL; the only answer texts taken from the log are the two
the task file quotes (seq 2532 and 5593). The turns and what each is for are listed in ``MANIFEST.md``.
"""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Sequence
from typing import Any

from local_astra.conductor.session import log_dump
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.telemetry.session_log import redact
from local_astra.toolcall import loop
from local_astra.toolcall.announced_call import reask_prompt

L1 = "[This screenshot is the browser screen now.]"
LU = "[This screenshot is the browser screen now, after your last action.]"
LN = "[This screenshot is the browser screen now; your last action did not change it.]"
SHARED_SCREEN = (
    "[This screenshot is the user's shared screen now, not your browser; "
    "click_at and type_at cannot act on it.]"
)
#: T-0177 A7's camera label, typed (T-0208 A2.6's t19)
SHARED_CAMERA = "[This image is the user's camera now, not your browser; click_at and type_at cannot act on it.]"
SYSTEM = "You are a synthetic test assistant that can use a browser."
TOOLS = [
    {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
    for n in ("google_search", "click_at", "type_at", "press_key", "scroll", "wait")
]
OPTIONS = {"top_k": None, "top_p": None, "max_tokens": None, "stop": []}
SID = "f0f0f0f0f0f0"


class Log:
    def __init__(self) -> None:
        self.seq = 0
        self.lines: list[dict[str, Any]] = []
        self.emit("log.started", None, {"pid": 1})

    def emit(
        self, kind: str, turn: str | None, data: dict[str, Any], *, at: int | None = None
    ) -> int:
        if at is not None:
            assert at > self.seq, (at, self.seq)
            self.seq = at
        else:
            self.seq += 2  # the real log interleaves machine.step/wire.out records; gaps are normal
        rec: dict[str, Any] = {
            "seq": self.seq,
            "t_ms": self.seq,
            "wall": "2026-10-29T00:00:00+00:00",
            "session_ms": self.seq,
            "kind": kind,
            "turn_id": turn,
            "data": data,
        }
        if kind == "log.started":
            rec.pop("session_ms")
        self.lines.append(rec)
        return self.seq


def image(tag: str) -> ImagePart:
    data = base64.b64encode(b"\xff\xd8\xff\xe0" + tag.encode() * 40).decode()
    return ImagePart(
        url="data:image/jpeg;base64," + data, phash="capture:" + tag.encode().hex()[:32]
    )


def base(tag: str, label: str | None, question: str) -> Prompt:
    parts: list[Any] = [image(tag)]
    if label is not None:
        parts.append(TextPart(text=label))
    parts.append(TextPart(text=question))
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text=SYSTEM),)),
            Message(role="user", content=tuple(parts)),
        ),
        prefix_len=2,
        thinking=True,
        tools=TOOLS,
    )


#: one executed (or refused) action: (name, arguments, action text, frame_after, status, page)
Action = tuple[str, dict[str, Any], str, str, str, str | None]


def loop_step(text: str, a: Action) -> loop.LoopStep:
    name, args, action, _frame, status, page = a
    response: dict[str, Any] = {"action": action, "status": status}
    if page is not None:
        response["page"] = page
    if name != "google_search":
        response["screenshot"] = "updated"
    return loop.LoopStep(text=text, name=name, arguments=args, response=response)


def request(
    log: Log,
    turn: str,
    step: int,
    prompt: Prompt,
    label: str | None,
    *,
    phase: str | None = None,
    report: bool = False,
    attempt: int | None = None,
) -> str:
    rid = f"{turn}-gen" if step == 1 else f"{turn}-gen-s{step}"
    aid = f"{SID}-{rid}-{attempt if attempt is not None else log.seq + 2}"
    data: dict[str, Any] = {
        "phase": phase or ("generate" if step == 1 else "tool_step"),
        "step": step,
        "request_id": rid,
        "attempt_id": aid,
        "options": OPTIONS,
        "prompt": log_dump(prompt),
    }
    if report:
        data["report_fact"] = True
    if label is not None:
        data["live_label"] = {
            "text": label,
            "above_px": None,
            "below_px": None,
            "viewport_px": None,
        }
    log.emit("llm.request", turn, redact(data))
    return aid


def commits(
    log: Log, turn: str, rid: str, aid: str, pieces: Sequence[tuple[str, str]], **extra: Any
) -> None:
    for seg, text in pieces:
        data: dict[str, Any] = {
            "request_id": rid,
            "attempt_id": aid,
            "canvas_idx": 0,
            "segment": seg,
            "text": text,
            "thought_open": False,
            "fence_close": None,
            "fence_continued": False,
        }
        if seg == "tool":
            data["stop_id"] = 50
        if seg == "thought" and extra.get("implicit"):
            data["implicit_thought"] = True
        log.emit("llm.commit", turn, data)
    if extra.get("done", True):
        log.emit("llm.done", turn, {"request_id": rid, "commits": len(pieces)})


def call(log: Log, turn: str, step: int, said: str, a: Action) -> None:
    name, args, action, frame, status, _page = a
    cid = f"{turn}-call" if step == 1 else f"{turn}-call-s{step}"
    head = {
        "call_id": cid,
        "source": "draft",
        "name": name,
        "arguments": args,
        "parse": "ok",
        "schema_valid": True,
        "query_problem": None,
        "repair_ms": None,
    }
    log.emit(
        "tool.call",
        turn,
        {
            "status": "proposed",
            **head,
            "action": None,
            "reason": None,
            "resolved": None,
            "spoken_prefix": said,
        },
    )
    if status == "declined":
        log.emit(
            "tool.call",
            turn,
            {
                "status": "refused",
                **head,
                "action": None,
                "reason": "the user said no",
                "resolved": None,
                "spoken_prefix": said,
                "user_declined": True,
                "verdict_status": "valid",
            },
        )
        return
    if name == "google_search":
        resolved: dict[str, Any] = {"query": args["query"]}
    else:
        resolved = {
            **{k: v for k, v in args.items() if k != "text"},
            "frame_after": frame,
            "hosts": ["shop.test"],
        }
    reason = None if status == "executed" else "the field could not be checked; nothing was typed"
    log.emit(
        "tool.call",
        turn,
        {
            "status": status,
            **head,
            "action": action,
            "reason": reason,
            "resolved": resolved,
            "spoken_prefix": said,
        },
    )
    log.emit(
        "tool.result",
        turn,
        {
            "call_id": cid,
            "step": step,
            "status": status,
            "response": loop.render_response(name, loop_step(said, a).response),
        },
    )


def turn_record(log: Log, turn: str, vision: str, steps: int, executed: int, end: str) -> None:
    log.emit(
        "turn.record",
        turn,
        {
            "turn_id": turn,
            "session_id": SID,
            "input": "typed",
            "outcome": "spoken",
            "regime": {
                "profile": "lean",
                "thinking": True,
                "tools": "native:bbfc53ccdd80:live:search:loop20",
            },
            "extras": {
                "tool_steps": steps,
                "tool_steps_executed": executed,
                "tool_loop_end": end,
                "browser": "open",
                "vision_source": vision,
            },
        },
    )


def loop_turn(
    log: Log,
    turn: str,
    question: str,
    plan: Sequence[tuple[str, Sequence[tuple[str, str]], Action | None, str | None]],
    final: tuple[Sequence[tuple[str, str]], str | None],
    *,
    vision: str = "browser",
    report: bool = False,
    final_at: int | None = None,
    end: str = "final",
    final_call: Action | None = None,
    final_end: str | None = None,
) -> None:
    """``plan``: per step before the last, (the step's answer text, its other commits, its action, its label).
    ``final``: the last request's commits and its label. ``final_at`` pins the last request's seq."""
    executed: list[loop.LoopStep] = []
    first: Prompt | None = None
    for i, (said, other, action, label) in enumerate(plan, start=1):
        p = base(f"{turn}-s{i}", label, question)
        first = first or p
        sent = (
            p
            if i == 1
            else loop.step_prompt(first.model_copy(update={"messages": p.messages}), executed)
        )
        aid = request(log, turn, i, sent, label)
        rid = f"{turn}-gen" if i == 1 else f"{turn}-gen-s{i}"
        pieces = [*other, ("answer", said)] if said else list(other)
        if action is not None:
            pieces.append(("tool", loop.render_call(action[0], action[1])))
        commits(log, turn, rid, aid, pieces)
        if action is not None:
            call(log, turn, i, said, action)
            if action[4] != "declined":
                executed.append(loop_step(said, action))
            else:
                executed.append(
                    loop.LoopStep(
                        text=said,
                        name=action[0],
                        arguments=action[1],
                        response=loop.status_payload("refused", "the user said no"),
                    )
                )
    k = len(plan) + 1
    pieces, label = final
    p = base(f"{turn}-s{k}", label, question)
    assert first is not None
    sent = loop.step_prompt(first.model_copy(update={"messages": p.messages}), executed)
    if report:
        sent = loop.with_report_fact(sent)
    if final_at is not None:
        log.seq = final_at - 2
    aid = request(log, turn, k, sent, label, report=report)
    commits(log, turn, f"{turn}-gen-s{k}", aid, pieces, implicit=report, done=final_end is None)
    answer = "".join(t for s, t in pieces if s == "answer")
    if final_end == "degenerate":
        # T-0142b: the answer tripped; the request is aborted, no llm.done, nothing voiced
        log.emit(
            "note",
            turn,
            {
                "what": "degenerate answer: request aborted",
                "request_id": f"{turn}-gen-s{k}",
                "attempt_id": aid,
                "step": k,
                "segment": "answer",
            },
        )
    elif final_end == "aborted":
        # a barge-in / UserStop / timeout closed the turn mid-stream: no llm.done, nothing voiced, no note
        pass
    elif final_call is not None:
        call(log, turn, k, answer, (*final_call[:4], "cancelled", None))  # type: ignore[arg-type]
    elif answer:
        log.emit(
            "tts.utterance",
            turn,
            {
                "utterance_id": f"{turn}-u1",
                "text": answer,
                "source": answer,
                "kind": "speech",
                "turn_state": "SPEAKING",
            },
        )
    n_exec = sum(1 for _s, _o, a, _l in plan if a is not None and a[4] == "executed")
    turn_record(log, turn, vision, k, n_exec, end)


def tap(x: int, y: int, frame: str, page: str | None = None) -> Action:
    return ("click_at", {"x": x, "y": y}, f"clicked at ({x}, {y})", frame, "executed", page)


def build() -> list[dict[str, Any]]:
    log = Log()
    # t0: a chat turn, no tools: no row
    p = base("t0-s1", L1, "hello there")
    aid = request(log, "t0", 1, p, L1)
    commits(
        log, "t0", "t0-gen", aid, [("thought", "A greeting."), ("answer", "Hello! How can I help?")]
    )
    turn_record(log, "t0", "browser", 1, 0, "none")

    # t1, shaped like seq 2493: type_at, scroll, then "Add to cart" (the thought says so) leaves frame_after
    # identical, no new page loaded; the final step has no thought and claims the item was added.
    loop_turn(
        log,
        "t1",
        "buy volume 7 of the synthetic comic",
        [
            (
                "I will open the shop.",
                [("thought", "Open the shop first.")],
                (
                    "type_at",
                    {"press_enter": True, "text": "shop.test", "x": 488, "y": 451},
                    "clicked at (488, 451) and typed 9 characters and pressed Enter",
                    "1a1a1a1a1a1a1a1a",
                    "executed",
                    None,
                ),
                L1,
            ),
            (
                "",
                [("thought", "The results may be lower down.")],
                (
                    "scroll",
                    {"direction": "down"},
                    "scrolled down by 600 px",
                    "2b2b2b2b2b2b2b2b",
                    "executed",
                    None,
                ),
                LU,
            ),
            (
                "I will add volume 7 to the cart.",
                [
                    (
                        "thought",
                        'The screen shows volume 7. I should click the "Add to cart" button for this item.',
                    )
                ],
                tap(485, 281, "2b2b2b2b2b2b2b2b", "no new page loaded"),
                LU,
            ),
        ],
        ([("answer", "I have added volume 7 of Vinland Saga to your cart.")], LU),
        final_at=2493,
    )

    # t2, shaped like seq 5538: two clicks with the same frame_after, a refused type_at (another frame_after)
    # between them; the final step has no thought and claims a purchase that never happened.
    loop_turn(
        log,
        "t2",
        "buy the synthetic game in the store",
        [
            (
                "I will open the store.",
                [],
                (
                    "type_at",
                    {"press_enter": True, "text": "store.test", "x": 500, "y": 40},
                    "clicked at (500, 40) and typed 10 characters and pressed Enter",
                    "3c3c3c3c3c3c3c3c",
                    "executed",
                    None,
                ),
                L1,
            ),
            (
                "",
                [("thought", "Go to the basket.")],
                tap(900, 180, "4d4d4d4d4d4d4d4d", "no new page loaded"),
                LU,
            ),
            (
                "",
                [("thought", "Type the quantity.")],
                (
                    "type_at",
                    {"press_enter": False, "text": "1", "x": 300, "y": 300},
                    "failed type_at: the field could not be checked; nothing was typed",
                    "5e5e5e5e5e5e5e5e",
                    "refused",
                    None,
                ),
                LU,
            ),
            (
                "I will click the proceed to purchase button.",
                [],
                tap(825, 235, "4d4d4d4d4d4d4d4d", "no new page loaded"),
                LU,
            ),
        ],
        ([("answer", "I have purchased The Legend of Zelda: Tears of the Kingdom for you.")], LU),
        final_at=5538,
    )

    # t3, shaped like seq 16197: the report step (report_fact: true), labelled "unchanged", an implicit thought,
    # an answer longer than 120 characters that claims checkout was reached.
    loop_turn(
        log,
        "t3",
        "put the synthetic comic in the basket and check out",
        [
            (
                "I will open the shop.",
                [],
                (
                    "type_at",
                    {"press_enter": True, "text": "shop.test", "x": 488, "y": 451},
                    "clicked at (488, 451) and typed 9 characters and pressed Enter",
                    "6f6f6f6f6f6f6f6f",
                    "executed",
                    None,
                ),
                L1,
            ),
            ("", [("thought", "Click checkout.")], tap(735, 535, "7a7a7a7a7a7a7a7a"), LU),
            (
                "",
                [("thought", "Click checkout again.")],
                tap(735, 535, "7a7a7a7a7a7a7a7a", "no new page loaded"),
                LU,
            ),
        ],
        (
            [
                ("thought", " Keep it short."),
                (
                    "answer",
                    "I have added the synthetic comic to your basket and proceeded to the checkout page. I "
                    "stopped there because paying needs your own payment and delivery details.",
                ),
            ],
            LN,
        ),
        report=True,
        final_at=16197,
        end="report",
    )

    # t4: search only (no browser action executed): no row
    loop_turn(
        log,
        "t4",
        "when does the synthetic shop open",
        [
            (
                "I will search for the opening hours.",
                [],
                (
                    "google_search",
                    {"query": "synthetic shop opening hours"},
                    'searched the web for "synthetic shop opening hours" via duckduckgo: 3 results (shop.test)',
                    "",
                    "executed",
                    None,
                ),
                L1,
            ),
        ],
        (
            [
                ("thought", "The results say nine."),
                ("answer", "The shop opens at nine in the morning."),
            ],
            LU,
        ),
    )

    # t5: the final request ends with a call (cancelled when the turn closed): no row
    loop_turn(
        log,
        "t5",
        "open the second page of results",
        [
            (
                "",
                [("thought", "Scroll first.")],
                (
                    "scroll",
                    {"direction": "down"},
                    "scrolled down by 600 px",
                    "8b8b8b8b8b8b8b8b",
                    "executed",
                    None,
                ),
                L1,
            )
        ],
        (
            [
                ("thought", "Click next."),
                ("answer", "I will click the next page button."),
                ("tool", loop.render_call("click_at", {"x": 640, "y": 700})),
            ],
            LU,
        ),
        final_call=tap(640, 700, ""),
        end="closed",
    )

    # t6: the shared vision source (H-T207-9): a row, never a check. press_key, scroll, press_key: the last
    # frame_after equals the first one, not the one before it.
    loop_turn(
        log,
        "t6",
        "submit the synthetic form",
        [
            (
                "I will press Enter.",
                [],
                (
                    "press_key",
                    {"combo": "Enter"},
                    "pressed Enter",
                    "9c9c9c9c9c9c9c9c",
                    "executed",
                    None,
                ),
                SHARED_SCREEN,
            ),
            (
                "",
                [],
                (
                    "scroll",
                    {"direction": "down"},
                    "scrolled down by 600 px",
                    "0d0d0d0d0d0d0d0d",
                    "executed",
                    None,
                ),
                SHARED_SCREEN,
            ),
            (
                "",
                [],
                (
                    "press_key",
                    {"combo": "Tab"},
                    "pressed Tab",
                    "9c9c9c9c9c9c9c9c",
                    "executed",
                    None,
                ),
                SHARED_SCREEN,
            ),
        ],
        ([("answer", "I pressed Tab and the form shows the next field.")], SHARED_SCREEN),
        vision="shared",
    )

    # t7 (adversarial): a click the user declined (never executed), then an executed google_search last, then an
    # answer: no browser action executed, so no row
    loop_turn(
        log,
        "t7",
        "find the price of the synthetic comic",
        [
            (
                "I will click the price filter.",
                [],
                ("click_at", {"x": 686, "y": 321}, "", "", "declined", None),
                L1,
            ),
            (
                "I will search for the price.",
                [],
                (
                    "google_search",
                    {"query": "synthetic comic price"},
                    'searched the web for "synthetic comic price" via duckduckgo: 2 results (shop.test)',
                    "",
                    "executed",
                    None,
                ),
                L1,
            ),
        ],
        (
            [
                ("thought", "The results give a price."),
                ("answer", "The synthetic comic costs nine pounds."),
            ],
            LU,
        ),
    )

    # t8: one executed click (no action before it), a thought, an answer that claims nothing
    loop_turn(
        log,
        "t8",
        "what is on the synthetic page",
        [("", [("thought", "Click the menu.")], tap(400, 300, "1e1e1e1e1e1e1e1e"), L1)],
        (
            [
                ("thought", "The image looks blank."),
                ("answer", "I cannot see the screen right now."),
            ],
            LU,
        ),
    )

    # t9: two type_at with different frame_after, a thought, an answer that claims nothing
    loop_turn(
        log,
        "t9",
        "look for volume 8 of the synthetic comic",
        [
            (
                "I will search the shop.",
                [],
                (
                    "type_at",
                    {"press_enter": True, "text": "volume 8", "x": 444, "y": 42},
                    "clicked at (444, 42) and typed 8 characters and pressed Enter",
                    "2f2f2f2f2f2f2f2f",
                    "executed",
                    None,
                ),
                L1,
            ),
            (
                "",
                [("thought", "Type it again.")],
                (
                    "type_at",
                    {"press_enter": True, "text": "volume 8", "x": 444, "y": 42},
                    "clicked at (444, 42) and typed 8 characters and pressed Enter",
                    "3a3a3a3a3a3a3a3a",
                    "executed",
                    None,
                ),
                LU,
            ),
        ],
        (
            [
                ("thought", "Results are showing."),
                ("answer", "I am looking at the search results for volume 8."),
            ],
            LU,
        ),
    )

    # t10: the last step only announced an action, T-0172 re-asked it (phase announce_reask), and the re-ask
    # answered without a call: a row whose step was re-asked (H-T207-5: never a check)
    question = "open the synthetic basket"
    first = base("t10-s1", L1, question)
    aid = request(log, "t10", 1, first, L1)
    a1 = tap(1200, 40, "4b4b4b4b4b4b4b4b")
    commits(
        log,
        "t10",
        "t10-gen",
        aid,
        [
            ("thought", "The basket icon."),
            ("answer", "I will open the basket."),
            ("tool", loop.render_call("click_at", {"x": 1200, "y": 40})),
        ],
    )
    call(log, "t10", 1, "I will open the basket.", a1)
    p2 = base("t10-s2", LU, question)
    sent = loop.step_prompt(
        first.model_copy(update={"messages": p2.messages}),
        [loop_step("I will open the basket.", a1)],
    )
    aid = request(log, "t10", 2, sent, LU)
    said = "I will click the checkout button."
    for seg, text in (("thought", "Checkout next."), ("answer", said)):
        log.emit(
            "llm.commit",
            "t10",
            {
                "request_id": "t10-gen-s2",
                "attempt_id": aid,
                "canvas_idx": 0,
                "segment": seg,
                "text": text,
                "thought_open": False,
                "fence_close": None,
                "fence_continued": False,
            },
        )
    log.emit(
        "note",
        "t10",
        {
            "what": "announced call re-asked",
            "request_id": "t10-gen-s2",
            "step": 2,
            "verb": "click",
            "chars": len(said),
        },
    )
    reask_aid = f"{SID}-t10-gen-s2-{log.seq + 2}"
    log.emit(
        "llm.request",
        "t10",
        redact(
            {
                "phase": "announce_reask",
                "step": 2,
                "request_id": "t10-gen-s2",
                "attempt_id": reask_aid,
                "options": OPTIONS,
                "prompt": log_dump(reask_prompt(sent, said)),
            }
        ),
    )
    commits(
        log,
        "t10",
        "t10-gen-s2",
        reask_aid,
        [("answer", "There is no checkout button on this screen.")],
    )
    log.emit(
        "note",
        "t10",
        {
            "what": "announced call re-ask outcome",
            "request_id": "t10-gen-s2",
            "step": 2,
            "outcome": "no_call",
        },
    )
    turn_record(log, "t10", "browser", 2, 1, "final")

    # t11: the final step's answer tripped T-0142b (note, no llm.done): a row, never a check
    loop_turn(
        log,
        "t11",
        "add the synthetic comic to the basket",
        [("", [("thought", "Click add.")], tap(500, 500, "5c5c5c5c5c5c5c5c"), L1)],
        ([("answer", "I have added the comic to the basket the basket the basket")], LU),
        final_end="degenerate",
        end="degenerate",
    )

    # t12: the final step was aborted mid-stream (a barge-in: no llm.done, no note): a row, never a check
    loop_turn(
        log,
        "t12",
        "open the synthetic help page",
        [("", [("thought", "Click help.")], tap(60, 60, "6d6d6d6d6d6d6d6d"), L1)],
        ([("thought", "The help page."), ("answer", "I have opened the help page.")], LU),
        final_end="aborted",
        end="aborted",
    )

    # t13: step 2 was re-asked (T-0172) and the re-ask drafted a click that executed; the final step 3 answers with
    # no call. Only the final step counts for reasked_this_step: a row that would be checked.
    question = "open the synthetic account page"
    first = base("t13-s1", L1, question)
    aid = request(log, "t13", 1, first, L1)
    a1 = tap(1100, 40, "7e7e7e7e7e7e7e7e")
    commits(
        log,
        "t13",
        "t13-gen",
        aid,
        [
            ("answer", "I will open the menu."),
            ("tool", loop.render_call("click_at", {"x": 1100, "y": 40})),
        ],
    )
    call(log, "t13", 1, "I will open the menu.", a1)
    done = [loop_step("I will open the menu.", a1)]
    p2 = base("t13-s2", LU, question)
    sent2 = loop.step_prompt(first.model_copy(update={"messages": p2.messages}), done)
    aid = request(log, "t13", 2, sent2, LU)
    said = "I will click the account link."
    for seg, text in (("thought", "Account link."), ("answer", said)):
        log.emit(
            "llm.commit",
            "t13",
            {
                "request_id": "t13-gen-s2",
                "attempt_id": aid,
                "canvas_idx": 0,
                "segment": seg,
                "text": text,
                "thought_open": False,
                "fence_close": None,
                "fence_continued": False,
            },
        )
    log.emit(
        "note",
        "t13",
        {
            "what": "announced call re-asked",
            "request_id": "t13-gen-s2",
            "step": 2,
            "verb": "click",
            "chars": len(said),
        },
    )
    reask_aid = f"{SID}-t13-gen-s2-{log.seq + 2}"
    log.emit(
        "llm.request",
        "t13",
        redact(
            {
                "phase": "announce_reask",
                "step": 2,
                "request_id": "t13-gen-s2",
                "attempt_id": reask_aid,
                "options": OPTIONS,
                "prompt": log_dump(reask_prompt(sent2, said)),
            }
        ),
    )
    a2 = tap(900, 120, "8f8f8f8f8f8f8f8f")
    commits(
        log,
        "t13",
        "t13-gen-s2",
        reask_aid,
        [("tool", loop.render_call("click_at", {"x": 900, "y": 120}))],
    )
    log.emit(
        "note",
        "t13",
        {
            "what": "announced call re-ask outcome",
            "request_id": "t13-gen-s2",
            "step": 2,
            "outcome": "call",
        },
    )
    call(log, "t13", 2, said, a2)
    done.append(loop_step(said, a2))
    p3 = base("t13-s3", LU, question)
    sent3 = loop.step_prompt(first.model_copy(update={"messages": p3.messages}), done)
    aid = request(log, "t13", 3, sent3, LU)
    commits(
        log, "t13", "t13-gen-s3", aid, [("answer", "The account page shows your name and address.")]
    )
    turn_record(log, "t13", "browser", 3, 2, "final")

    # t14: the only executed action is wait (T-0202): a browser action, so a row
    loop_turn(
        log,
        "t14",
        "wait for the synthetic page to load",
        [
            (
                "I will wait for the page.",
                [],
                ("wait", {}, "waited 2 s", "9a9a9a9a9a9a9a9a", "executed", None),
                L1,
            )
        ],
        ([("answer", "The page has loaded and shows the comic list.")], LU),
    )

    # t15 (T-0207 A2-R3): a checked step as the A2 session logs it: step 2's commits (no llm.done of their own), then
    # the completion check's request (phase completion_check, a fresh attempt under the same machine id), its commits,
    # one llm.done for the step and the check's answer voiced. The step's request is the turn's final request.
    question = "add the synthetic comic to the basket"
    first = base("t15-s1", L1, question)
    aid = request(log, "t15", 1, first, L1)
    a1 = tap(610, 330, "5c5c5c5c5c5c5c5c")
    commits(
        log,
        "t15",
        "t15-gen",
        aid,
        [
            ("answer", "I will add it to the basket."),
            ("tool", loop.render_call("click_at", {"x": 610, "y": 330})),
        ],
    )
    call(log, "t15", 1, "I will add it to the basket.", a1)
    p2 = base("t15-s2", LU, question)
    sent = loop.step_prompt(
        first.model_copy(update={"messages": p2.messages}),
        [loop_step("I will add it to the basket.", a1)],
    )
    aid = request(log, "t15", 2, sent, LU)
    draft = "I have added the comic to your basket."
    commits(log, "t15", "t15-gen-s2", aid, [("answer", draft)], done=False)
    log.emit(
        "note",
        "t15",
        {
            "what": "completion check asked",
            "request_id": "t15-gen-s2",
            "step": 2,
            "chars": len(draft),
            "same_as_last": False,
            "report": False,
        },
    )
    check_aid = f"{SID}-t15-gen-s2-{log.seq + 2}"
    fresh = image("t15-fresh")
    log.emit(
        "llm.request",
        "t15",
        redact(
            {
                "phase": "completion_check",
                "step": 2,
                "request_id": "t15-gen-s2",
                "attempt_id": check_aid,
                "options": OPTIONS,
                "prompt": log_dump(
                    check_prompt_of(sent, draft, fresh),
                ),
                "fresh_frame": {"phash": fresh.phash, "same_as_last": False},
            }
        ),
    )
    checked = "The basket on this screen is still empty."
    commits(log, "t15", "t15-gen-s2", check_aid, [("answer", checked)])
    log.emit(
        "note",
        "t15",
        {
            "what": "completion check outcome",
            "request_id": "t15-gen-s2",
            "step": 2,
            "outcome": "no_call",
        },
    )
    log.emit(
        "tts.utterance",
        "t15",
        {
            "utterance_id": "t15-u1",
            "text": checked,
            "source": checked,
            "kind": "speech",
            "turn_state": "SPEAKING",
        },
    )
    turn_record(log, "t15", "browser", 2, 1, "final")

    # t16-t19 (T-0208 A2.6): shared-source turns (turn line vision_source "shared"); step 1 carries the shared
    # screen label and executes a scroll; the final request's label decides would_check: a browser key (updated,
    # unchanged, now) is a browser frame (true); shared_camera is the user's camera (false)
    for turn, final_label, said in (
        ("t16", LU, "The replacement steam wand is in your cart now."),
        ("t17", LN, "The page did not move; the stove list is still on screen."),
        ("t18", L1, "This page shows the pizza order form."),
        ("t19", SHARED_CAMERA, "The camera shows the coffee machine."),
    ):
        loop_turn(
            log,
            turn,
            "this part of my coffee machine broke, put the replacement in my cart",
            [
                (
                    "I will scroll the shop page.",
                    [],
                    (
                        "scroll",
                        {"direction": "down"},
                        "scrolled down by 600 px",
                        f"{turn}{turn}{turn}{turn}"[:16],
                        "executed",
                        None,
                    ),
                    SHARED_SCREEN,
                ),
            ],
            ([("answer", said)], final_label),
            vision="shared",
        )

    log.emit("log.stopped", None, {})
    return log.lines


def check_prompt_of(sent: Prompt, said: str, fresh: ImagePart) -> Prompt:
    """T-0207 A1's ``check_prompt`` over ``sent``, with ``fresh`` as the frame's image part (the builder's images are
    parts, not frames: the frame's bytes are the part's data URL payload, its phash the part's)."""
    from local_astra.contracts.perception import Frame, Source
    from local_astra.toolcall.completion_check import check_prompt

    jpeg = base64.b64decode(fresh.url.split(",", 1)[1])
    frame = Frame(
        source=Source.WINDOW, ts_ms=0, width=1280, height=800, jpeg=jpeg, phash=fresh.phash
    )
    return check_prompt(sent, said, frame, report=False)


if __name__ == "__main__":
    for r in build():
        sys.stdout.write(json.dumps(r) + "\n")
