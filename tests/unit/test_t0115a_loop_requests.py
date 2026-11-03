"""T-0115a: the tool-loop request-level rows -- form G vs A (Q4), an image in a continued message (Q5) and the
compaction A/B (the E rows) -- on the harness's pure surface (unit; CPU; no engine).

Every row that needs the harness is red on the named assertion in ``harness()``
(``scripts/gpu/t0115a_loop_requests.py`` does not exist yet) or on an assertion of its own, never on an import
error: the harness is loaded lazily, by path. Mock-engine runs live in
``tests/integration/test_t0115a_harness_smoke.py`` (criterion 8).

**The assumed harness surface** (the task names the behaviour; these names are the test-author's, stated so the
implementer can implement to them or say why not):

- ``E_HISTORIES``: 4 objects with ``.id`` (str, unique), ``.question`` (str), ``.fact`` (str) and ``.equivalents``
  (tuple of str: the pre-registered stated equivalents, e.g. ``"7:40 pm"`` for ``"19:40"``). The accepted strings of
  a history are ``(fact, *equivalents)``.
- ``e_steps(history, placement) -> tuple[LoopStep, ...]``: 5 maximal ``google_search`` steps (5 results each, the
  rendered response at the ``MAX_RESPONSE`` bound) from harness literals; ``placement`` is ``"old"`` (the fact in
  step 1's response) or ``"kept"`` (in step 5's).
- ``E_BUDGET: int``: the one budget passed to ``loop.compact`` (at its defaults otherwise) for every history and
  placement; it elides exactly steps 1-3.
- ``e_prompts(base, history, placement) -> {"warm", "FULL", "COMPACT"}`` of ``Prompt``: ``warm`` is
  ``step_prompt(base, steps[:4])``, ``FULL`` is ``step_prompt(base, steps)``, ``COMPACT`` is
  ``step_prompt(base, loop.compact(steps, E_BUDGET))``. ``base`` is any step-1 ``Prompt``.
- ``Q4_CASES``: 6 objects with ``.case_id`` (a dev-split id of ``tests/eval/toolcall/cases.jsonl``) and ``.step``
  (the executed step 1, a ``LoopStep``): 2 ``google_search`` (fixed fake results), 2 ``click_at``, 1 ``type_at``,
  1 ``press_key`` (dry-run ``action_payload`` responses). Arm G is ``step_prompt(base, (case.step,))`` through
  ``VllmClient``; arm A is ``form_a_body``.
- ``form_a_body(base, step, request_id) -> dict``: the raw OpenAI body -- ``base``'s messages as ``VllmClient``
  renders them, then an ``assistant`` message whose ``content`` is ``step.text`` (ruling R6, 2026-10-17: the OpenAI
  equivalent of what form G shows the model; it was "no content" before R6) and one ``tool_calls`` entry
  ``{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}``, then a ``role: tool`` message
  with the same ``tool_call_id`` and the response as a JSON string; the same non-message fields as
  ``VllmClient.request_body(base, ...)``; no ``continue_final_message``.
- ``repeats_start(step1_text, step2_text) -> bool``: the step-2 text begins with step 1's first sentence,
  normalised (case and runs of whitespace).
- ``classify_e(commits, accepted) -> "correct" | "re_search" | "unsupported" | "other"``; ``commits`` are
  ``Commit.model_dump()`` mappings as stored in ``requests.jsonl``. Ruling R1 (2026-10-17): "ends inside a thought"
  is the **final** state, the last commit's ``thought_open`` (``VllmClient.generate`` marks every commit of a canvas
  that ends inside a thought, so a thought closed in a later canvas leaves ``thought_open`` on the earlier commits).
  Ruling R2: matching is case-insensitive, with number/word boundaries (``16:45`` does not match ``6:45``), and
  accepts the obvious 12-hour and unpadded forms; the exact list is the implementer's (pre-registered in the write-up).
- ``classify_q4(commits) -> "final" | "call" | "thought_open" | "empty"`` (R7), the same final-state rule (R1).
- ``e_outcome(tally) -> {"bars": {"EC", "E1", "E2", "E3": bool}, "outcome": str}``; ``tally[placement][arm]`` maps
  each of the outcomes to a count of 40 (``placement`` in old/kept, ``arm`` in FULL/COMPACT). ``outcome`` is
  ``"elision viable"``, ``"elision hurts: <every failing bar>"`` or ``"inconclusive: <reason>"``. Ruling R4: a fifth
  outcome ``"errored"`` (non-200, timeout, transport error; never correct) is counted in the tally cell beside the
  four (``tally[p][a]["errored"]``, inside the cell's n of 40); errored E requests >= 3/160 make the outcome
  ``"inconclusive: ..."`` with ``errored`` named in the reason.
- ``e_cost(records) -> mapping`` with ``compact_miss_tokens_median`` (median over COMPACT records of
  ``prefix_query_tokens - prefix_hit_tokens``; warm and FULL records excluded), ``first_commit_ms_diff_median`` and
  ``done_ms_diff_median`` (median of the paired COMPACT - FULL differences, paired by
  ``(history, placement, repeat)``) and ``ms_per_miss_token``.
- ``schedule() -> list[dict]``, pure: every request in send order. Each entry has ``row`` (``"Q4"``, ``"E"`` or
  ``"Q5"``); Q4 entries ``case_id``, ``arm`` (``"G"``/``"A"``), ``repeat``; E entries ``history`` (the id),
  ``placement``, ``arm`` (``"warm"``/``"FULL"``/``"COMPACT"``) and ``repeat``.
- ``main(argv) -> int`` with ``--base-url`` and ``--out-dir``; exit 2 on a refusal, before any socket.
  ``CASES_SHA256`` (the ``cases.jsonl`` pin) and ``DEV_CASE_IDS`` (every case id the harness reads) are module
  constants that ``main`` checks. ``build_report(run_dir)`` writes nothing and returns ``report.json``'s content.
- ``requests.jsonl`` (one line per request, warm included, in send order) carries the schedule entry's keys plus
  ``request_id``; E lines ``outcome``, ``commits``, ``prompt_tokens``, ``prefix_query_tokens``,
  ``prefix_hit_tokens``, ``first_commit_ms``, ``done_ms``; Q5 lines ``reading`` (``"accepted"``, ``"rejected"`` or
  ``"engine-down"``). ``manifest.json`` has ``gate_shas`` (repo path -> sha256 of ``local_astra/toolcall/loop.py``
  and ``local_astra/conductor/llm_client.py``) and ``e_histories`` (id -> ``{"fact", "accepted"}``).
  ``report.json`` has ``E`` = ``{"tally", "bars", "outcome", "cost"}`` and ``Q5`` = ``{"readings": [...]}``.
  ``cost`` adds (ruling 2026-10-17) ``compact_miss_tokens_first_per_cell_median`` / ``_n`` (the first COMPACT of each
  (history, placement) after its warm: the cost reading) and ``predicted_miss_tokens_median`` (manifest
  ``predicted_miss_tokens``, measured cells only).
  **Per placement (ruling R3; key names the test-author's):** ``cost["by_placement"][placement]`` for ``placement``
  in old/kept holds ``compact_miss_tokens_first_per_cell_median`` and ``_n`` (that placement's cells only),
  ``predicted_miss_tokens_median`` (manifest keys ``"<history>/<placement>"`` of that placement),
  ``first_commit_ms_diff_median`` and ``done_ms_diff_median`` (the paired COMPACT - FULL difference of each cell's
  **first pair** after its warm -- the lowest repeat with both arms -- medianed over the placement's cells) and
  ``ms_per_miss_token`` (that done median / that first-COMPACT miss median, the write-up's definition). The pooled
  top-level keys stay (the all-COMPACT miss median and the all-pairs latency medians as cache sanity checks).
  ``report["Q4"][arm]`` (R7, the harness's shape) has ``outcomes[o]``, ``repeats_start`` and ``errored`` as
  ``{"k", "n"}`` over that arm's Q4 lines.

Expected values are hand-built: the FULL / warm / COMPACT texts are joined here from ``render_call`` and
``render_response`` (T-0113 proved them against the template oracle), the stub block is typed by hand from T-0120
criterion 4, the budget's elided set is computed from rendered lengths, and the bar thresholds are typed from
docs/tasks/T-0115a.md criterion 5 (H-T115a-1, ruled by the human 2026-10-17).

Regime: CPU, pure, hand-built data, ``loop.py`` at the working tree; no measurement.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from local_astra.toolcall import loop, native
from local_astra.toolcall import tools as toolset
from local_astra.toolcall.loop import LoopStep, render_call, render_response
from local_astra.toolcall.special_tokens import drop_special_literals
from tests.integration.test_t0098_harness_smoke import no_sockets

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0115a_loop_requests.py"
CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"

SURFACE = (
    "e_steps",
    "e_prompts",
    "classify_e",
    "classify_q4",
    "e_outcome",
    "e_cost",
    "form_a_body",
    "repeats_start",
    "schedule",
    "build_report",
    "main",
)
CONSTANTS = ("E_HISTORIES", "E_BUDGET", "Q4_CASES", "DEV_CASE_IDS", "CASES_SHA256")

PLACEMENTS = ("old", "kept")
#: the placement's fact-bearing step (0-based): old = step 1, kept = step 5 (criterion 4)
FACT_STEP = {"old": 0, "kept": 4}
OUTCOMES = ("correct", "re_search", "unsupported", "other")
#: ruling R4: the fifth E outcome (non-200, timeout, transport error), never correct
ERRORED = "errored"
BARS = ("EC", "E1", "E2", "E3")

#: T-0120b criterion 2 (re-scope of T-0120 criterion 4, H-T120b-0), typed by hand: the stub a 5-result search
#: response becomes. 16 (``<|tool_response>``) + 22 (``response:google_search``) + 6 (``{note:``) + 5 + 51 (the note)
#: + 5 + 11 (``,results:5}``) + 16 (``<tool_response|>``) = 132 chars (was 103 with the old note). The harness follows
#: the product ``loop.compact`` by design (``test_compact_arm_uses_product_compact``), so its rows follow the stub;
#: the c398fdf run used the old 103-char stub and its record is untouched (T-0120b criterion 10).
STUB_5 = (
    '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
    "results:5}<tool_response|>"
)
assert len(STUB_5) == 132, len(STUB_5)
#: "maximal" read as: the rendered response block within this many chars of MAX_RESPONSE (1500).
MAXIMAL_SLACK = 100

# --- criterion 5 (H-T115a-1), typed out; n = 40 per (placement, arm) ---------------------------------------------
N = 40
E1_MARGIN = 4
E2_MIN = 32
E3_MAX = 8
EC_MIN = 28

# --- criteria 2-4, typed out -----------------------------------------------------------------------------------
Q4_TOOLS = Counter({"google_search": 2, "click_at": 2, "type_at": 1, "press_key": 1})
Q4_N = 5
E_REPEATS = 10
N_Q4, N_E_ARMS, N_WARM, N_Q5 = 60, 160, 40, 3


# =====================================================================================================
# Loading
# =====================================================================================================


def harness() -> ModuleType:
    """The harness, loaded by path. Red here, by name, until T-0115a criterion 1 lands."""
    assert HARNESS.is_file(), f"T-0115a criterion 1: {HARNESS.relative_to(REPO)} does not exist yet"
    name = "_t0115a_loop_requests_unit"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    for attr in SURFACE:
        assert callable(getattr(module, attr, None)), (
            f"T-0115a surface: {HARNESS.name} has no callable {attr!r} (see this module's docstring)"
        )
    for attr in CONSTANTS:
        assert hasattr(module, attr), (
            f"T-0115a surface: {HARNESS.name} has no {attr!r} (see this module's docstring)"
        )
    return module


@functools.cache
def tools() -> tuple[dict[str, Any], ...]:
    return tuple(native.tools_field(toolset.load()))


def base_prompt(question: str = "Which of these is the right one?") -> Prompt:
    """A step-1 prompt built by hand: a system line and the user's question, tools on."""
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(role="user", content=(TextPart(text=question),)),
        ),
        prefix_len=2,
        tools=tools(),
    )


def accepted(history: Any) -> tuple[str, ...]:
    return (str(history.fact), *(str(e) for e in history.equivalents))


def hand_text(steps: Sequence[LoopStep], stubbed: Iterable[int] = ()) -> str:
    """The final assistant text, joined here: text + call + response per step, the responses at ``stubbed``
    replaced by the hand-typed stub, leading whitespace dropped (the template's trim)."""
    stubbed = set(stubbed)
    return "".join(
        drop_special_literals(s.text)
        + render_call(s.name, s.arguments)
        + (STUB_5 if i in stubbed else render_response(s.name, s.response))
        for i, s in enumerate(steps)
    ).lstrip()


def final_text(prompt: Prompt) -> str:
    assert prompt.continue_final_message, "a step prompt continues the final assistant message"
    last = prompt.messages[-1]
    assert last.role == "assistant"
    return last.text()


def commit(segment: str, text: str, *, thought_open: bool = False, idx: int = 0) -> dict[str, Any]:
    stop = 50 if segment == "tool" else None
    return Commit(
        request_id="t0115a-unit",
        canvas_idx=idx,
        segment=segment,  # type: ignore[arg-type]
        text=text,
        thought_open=thought_open,
        stop_id=stop,
    ).model_dump()


# =====================================================================================================
# Criterion 4: the E histories, the budget and the arms
# =====================================================================================================


def test_e_histories_by_hand() -> None:
    """Criterion 4: 4 histories; each placement is 5 maximal ``google_search`` steps (5 results, the rendered
    block within 100 chars of ``MAX_RESPONSE``); FULL is the base messages plus one continued assistant message
    whose text is the hand-joined text of the 5 steps, and warm the same for steps 1-4 (FULL extends warm byte for
    byte: the cache property the warm request depends on). The fact (or an equivalent) appears in exactly one
    rendered response -- step 1's for old, step 5's for kept -- after the ``MAX_RESPONSE`` cap, and in no step text,
    call or question."""
    h = harness()
    histories = list(h.E_HISTORIES)
    assert len(histories) == 4, f"criterion 4: 4 invented-fact questions, got {len(histories)}"
    assert len({x.id for x in histories}) == 4, "history ids are not unique"
    for hist in histories:
        acc = accepted(hist)
        assert all(acc), f"{hist.id}: an empty accepted string matches everything"
        base = base_prompt(str(hist.question))
        assert not any(a in str(hist.question) for a in acc), (
            f"{hist.id}: the question holds the fact"
        )
        for placement in PLACEMENTS:
            steps = tuple(h.e_steps(hist, placement))
            assert len(steps) == 5, f"{hist.id}/{placement}: {len(steps)} steps"
            for i, s in enumerate(steps):
                assert s.name == "google_search", f"{hist.id}/{placement} step {i + 1}: {s.name}"
                assert (
                    isinstance(s.response, Mapping)
                    and len(s.response["results"]) == loop.MAX_RESULTS
                ), f"{hist.id}/{placement} step {i + 1}: not a 5-result response"
                block = render_response(s.name, s.response)
                assert len(block) >= loop.MAX_RESPONSE - MAXIMAL_SLACK, (
                    f"{hist.id}/{placement} step {i + 1}: {len(block)} chars is not maximal"
                )
                assert not any(a in drop_special_literals(s.text) for a in acc), (
                    "fact in a step text"
                )
                assert not any(a in render_call(s.name, s.arguments) for a in acc), "fact in a call"
            holders = [
                i
                for i, s in enumerate(steps)
                if any(a in render_response(s.name, s.response) for a in acc)
            ]
            assert holders == [FACT_STEP[placement]], (
                f"{hist.id}/{placement}: the fact is in the rendered responses {holders}, "
                f"want only step {FACT_STEP[placement] + 1}"
            )
            arms = h.e_prompts(base, hist, placement)
            for name in ("warm", "FULL"):
                p = arms[name]
                assert p.messages[:-1] == base.messages, f"{name}: the step-1 messages changed"
                assert p.tools == base.tools and p.thinking == base.thinking, (
                    f"{name}: tools/thinking"
                )
            full, warm = final_text(arms["FULL"]), final_text(arms["warm"])
            assert full == hand_text(steps), (
                f"{hist.id}/{placement}: FULL is not the hand-joined 5 steps"
            )
            assert warm == hand_text(steps[:4]), (
                f"{hist.id}/{placement}: warm is not the hand-joined 4 steps"
            )
            assert full.startswith(warm), "FULL does not extend the warm request"


def test_e_budget_elides_exactly_steps_1_to_3() -> None:
    """Criterion 4 (adversarial: a budget that elides nothing makes the A/B vacuous). By hand, from rendered
    lengths: the FULL continuation ``L`` is over ``E_BUDGET`` (so ``compact`` acts at all), and after eliding steps
    1 and 2 it is still over the target ``E_BUDGET * 2 // 3`` (so it does not stop at 2); steps 4-5 are the kept
    tail (``keep_last`` 2). So the elided set is {0, 1, 2}. The COMPACT text then equals the hand-joined text with
    the hand-typed stub in place of responses 1-3 -- no fact at the old placement, the fact at the kept one -- and
    the product's ``compact`` agrees.

    T-0120b re-scope: the stub is the 132-char search form (``STUB_5``), so each elision saves 29 chars less than
    with the old 103-char stub; the elided set stays {0, 1, 2} (the T-0120b ruling: ``E_BUDGET`` unchanged)."""
    h = harness()
    budget = h.E_BUDGET
    assert isinstance(budget, int) and budget >= 1, f"E_BUDGET {budget!r}"
    target = budget * 2 // 3
    for hist in h.E_HISTORIES:
        acc = accepted(hist)
        base = base_prompt(str(hist.question))
        for placement in PLACEMENTS:
            steps = tuple(h.e_steps(hist, placement))
            full_len = len(hand_text(steps))
            saving = [len(render_response(s.name, s.response)) - len(STUB_5) for s in steps]
            assert full_len > budget, (
                f"{hist.id}/{placement}: L {full_len} <= budget {budget}: compact elides nothing"
            )
            assert all(x > 0 for x in saving[:3]), f"{hist.id}/{placement}: a stub is not shorter"
            assert full_len - saving[0] - saving[1] > target, (
                f"{hist.id}/{placement}: compact stops at 2 elided ({full_len - saving[0] - saving[1]} <= {target})"
            )
            compacted = loop.compact(steps, budget)
            elided = {
                i
                for i, (a, b) in enumerate(zip(steps, compacted, strict=True))
                if a.response != b.response
            }
            assert elided == {0, 1, 2}, (
                f"{hist.id}/{placement}: the product elides {sorted(elided)}"
            )
            text = final_text(h.e_prompts(base, hist, placement)["COMPACT"])
            assert text == hand_text(steps, stubbed=(0, 1, 2)), (
                f"{hist.id}/{placement}: COMPACT is not the hand-joined text with steps 1-3 stubbed"
            )
            assert text.count(STUB_5) == 3
            has_fact = any(a in text for a in acc)
            assert has_fact is (placement == "kept"), (
                f"{hist.id}/{placement}: fact present = {has_fact}"
            )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def test_compact_arm_uses_product_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 4 (reviewer mutation: a harness-local stub). (a) For every history and placement, COMPACT equals
    ``step_prompt(base, loop.compact(steps, E_BUDGET))``. (b) With the product's ``compact`` replaced (in
    ``loop`` and wherever the harness bound it) by a wrapper that marks its result, the COMPACT text carries the mark,
    and the wrapper saw ``E_BUDGET`` and no ``keep_last``/``target`` other than the defaults. (c) The harness's code
    (docstrings excluded) defines no ``compact``/``elided_payload`` and holds no stub-note literal."""
    h = harness()
    for hist in h.E_HISTORIES:
        base = base_prompt(str(hist.question))
        for placement in PLACEMENTS:
            steps = tuple(h.e_steps(hist, placement))
            want = loop.step_prompt(base, loop.compact(steps, h.E_BUDGET))
            assert h.e_prompts(base, hist, placement)["COMPACT"] == want, f"{hist.id}/{placement}"

    real = loop.compact
    seen: list[tuple[int, dict[str, Any]]] = []
    mark = {"note": "t0115a product-compact mark"}

    def marked(steps: Sequence[LoopStep], budget: int, **kwargs: Any) -> tuple[LoopStep, ...]:
        seen.append((budget, dict(kwargs)))
        out = list(real(steps, budget, **kwargs))
        s = out[0]
        out[0] = LoopStep(text=s.text, name=s.name, arguments=s.arguments, response=mark)
        return tuple(out)

    monkeypatch.setattr(loop, "compact", marked)
    for k, v in list(vars(h).items()):
        if v is real:
            monkeypatch.setattr(h, k, marked)
    hist = next(iter(h.E_HISTORIES))
    text = final_text(h.e_prompts(base_prompt(str(hist.question)), hist, "old")["COMPACT"])
    assert "t0115a product-compact mark" in text, "the COMPACT arm did not come from loop.compact"
    assert seen, "loop.compact was not called"
    for budget, kwargs in seen:
        assert budget == h.E_BUDGET, f"compact called with budget {budget}"
        assert kwargs.get("keep_last", loop.COMPACT_KEEP_LAST) == loop.COMPACT_KEEP_LAST, kwargs
        assert kwargs.get("target") is None, f"compact is not at its default target: {kwargs}"

    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    docs = _docstring_nodes(tree)
    defs = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.name in ("compact", "elided_payload")
    ]
    assert defs == [], f"the harness defines its own {defs}"
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docs
        and loop.ELIDED_NOTE in n.value
    ]
    assert literals == [], (
        "the harness carries its own stub note (docstrings excluded from this check)"
    )


# =====================================================================================================
# Criterion 2: Q4 -- form G vs form A
# =====================================================================================================


def test_q4_cases_by_tool() -> None:
    """Criterion 2: 6 step-2 cases -- 2 ``google_search``, 2 ``click_at``, 1 ``type_at``, 1 ``press_key`` (T-0111a's
    set) -- each a dev-split case of the pinned ``cases.jsonl`` whose tool is the step's; search steps carry a
    fixed fake results list, actions the dry-run ``action_payload``."""
    h = harness()
    rows = {
        r["id"]: r for r in (json.loads(x) for x in CASES.read_text().splitlines() if x.strip())
    }
    cases = list(h.Q4_CASES)
    assert Counter(c.step.name for c in cases) == Q4_TOOLS, Counter(c.step.name for c in cases)
    assert len({c.case_id for c in cases}) == 6
    for c in cases:
        row = rows.get(c.case_id)
        assert row is not None and row["split"] == "dev", f"{c.case_id}: not a dev-split case"
        assert row["tool"] == c.step.name, f"{c.case_id}: the case's tool is {row['tool']}"
        r = c.step.response
        if c.step.name == "google_search":
            assert isinstance(r, Mapping) and r.get("results"), f"{c.case_id}: no fixed results"
        else:
            assert r == loop.action_payload(r["action"]), (
                f"{c.case_id}: not a dry-run action payload"
            )


def test_form_a_body_shape() -> None:
    """Criterion 1 and H-T111a-10: form A is the OpenAI ``role: tool`` body -- the base messages exactly as
    ``VllmClient`` renders them, an assistant message whose ``content`` is step 1's text (ruling R6: re-scoped from
    "no content", so the start-repetition comparison compares like with like) and one ``tool_calls`` entry (``type``
    function, the step's name, ``arguments`` a JSON string of the step's arguments), then a ``tool`` message with the
    same ``tool_call_id`` carrying the response as JSON; no ``continue_final_message``, no seed; the tools fields and
    template kwargs as the G body's."""
    h = harness()
    for case in h.Q4_CASES:
        step = case.step
        base = base_prompt("Do the thing on the screen.")
        rid = f"t0115a-unit-{case.case_id}"
        body = h.form_a_body(base, step, rid)
        g = VllmClient().request_body(base, rid)
        n = len(base.messages)
        msgs = body["messages"]
        assert msgs[:n] == g["messages"], "the base messages differ from VllmClient's rendering"
        assert len(msgs) == n + 2, f"{len(msgs) - n} messages after the base, want 2"
        asst, tool = msgs[n], msgs[n + 1]
        assert step.text, f"{case.case_id}: step 1 has no text, so R6's pin would be vacuous"
        assert asst["role"] == "assistant" and asst.get("content") == step.text, (
            f"R6: form A's assistant content is step 1's text {step.text!r}: {asst}"
        )
        calls = asst["tool_calls"]
        assert isinstance(calls, list) and len(calls) == 1
        call = calls[0]
        assert call["type"] == "function" and call["function"]["name"] == step.name
        assert isinstance(call["function"]["arguments"], str), "OpenAI arguments are a JSON string"
        assert json.loads(call["function"]["arguments"]) == json.loads(
            json.dumps(dict(step.arguments))
        )
        assert tool["role"] == "tool" and tool["tool_call_id"] == call["id"] and call["id"]
        assert isinstance(tool["content"], str)
        assert json.loads(tool["content"]) == json.loads(json.dumps(step.response))
        assert "continue_final_message" not in body, "form A is not a continuation"
        assert "seed" not in json.dumps(body), "unseeded"
        for k in (
            "tools",
            "tool_choice",
            "skip_special_tokens",
            "return_token_ids",
            "stream",
            "chat_template_kwargs",
        ):
            assert body.get(k) == g[k], f"form A's {k} differs from the G body's"


@pytest.mark.parametrize(
    ("step1", "step2", "want"),
    [
        ("Let me look that up. One moment.", "let me  look that up. The answer is 4.", True),
        ("Let me look that up. One moment.", "Let me look that up.", True),
        ("Let me look that up. One moment.", "The answer is 4. Let me look that up.", False),
        ("Let me look that up. One moment.", "One moment. The answer is 4.", False),
    ],
)
def test_repeats_start(step1: str, step2: str, want: bool) -> None:
    """Criterion 2's report-only row: the step-2 text begins with step 1's *first* sentence, normalised (case and
    whitespace). A later sentence of step 1, or the first sentence elsewhere in step 2, is not a repeat."""
    assert harness().repeats_start(step1, step2) is want


# =====================================================================================================
# The schedule: counts, order, Q5 last
# =====================================================================================================


def test_schedule_counts_and_e_order() -> None:
    """Criteria 2 and 4: 60 Q4 requests (6 cases x G/A x 5), 160 E arm requests (4 x 2 x 10 x 2) and 40 warm,
    3 Q5 (263). Per (history, placement): FULL and COMPACT 10 each, one pair per repeat, in alternating order (FULL
    first in 5 repeats, COMPACT first in 5); each history's first E request is a warm, and a warm of the same
    (history, placement) precedes its first arm request (the miss measurement is relative to it)."""
    h = harness()
    s = list(h.schedule())
    rows = Counter(e["row"] for e in s)
    assert rows == Counter({"Q4": N_Q4, "E": N_E_ARMS + N_WARM, "Q5": N_Q5}), rows
    q4 = Counter((e["case_id"], e["arm"]) for e in s if e["row"] == "Q4")
    assert q4 == Counter({(c.case_id, a): Q4_N for c in h.Q4_CASES for a in ("G", "A")}), q4
    e = [x for x in s if x["row"] == "E"]
    assert sum(x["arm"] == "warm" for x in e) == N_WARM
    ids = [x.id for x in h.E_HISTORIES]
    for hid in ids:
        mine = [x for x in e if x["history"] == hid]
        assert mine[0]["arm"] == "warm", (
            f"{hid}: the first E request is not the step-4 (warm) prompt"
        )
        for placement in PLACEMENTS:
            here = [x for x in mine if x["placement"] == placement]
            first_arm = next(i for i, x in enumerate(here) if x["arm"] != "warm")
            assert any(x["arm"] == "warm" for x in here[:first_arm]), (
                f"{hid}/{placement}: no warm first"
            )
            arms = [x for x in here if x["arm"] != "warm"]
            assert Counter(x["arm"] for x in arms) == Counter(
                {"FULL": E_REPEATS, "COMPACT": E_REPEATS}
            )
            first: dict[Any, str] = {}
            for x in arms:
                first.setdefault(x["repeat"], x["arm"])
            assert len(first) == E_REPEATS, f"{hid}/{placement}: {len(first)} repeats"
            assert Counter(first.values()) == Counter({"FULL": 5, "COMPACT": 5}), (
                f"{hid}/{placement}: the arms do not alternate: {first}"
            )


def test_q5_runs_last() -> None:
    """Criterion 3 (H-T115a-3; reviewer mutation: Q5 run first): the 3 Q5 requests are the last 3 of the schedule,
    because a crash in the patched image path kills the EngineCore and every row after it."""
    s = list(harness().schedule())
    assert [e["row"] for e in s[-N_Q5:]] == ["Q5"] * N_Q5, [e["row"] for e in s[-5:]]
    assert sum(e["row"] == "Q5" for e in s) == N_Q5


# =====================================================================================================
# The outcome classifier
# =====================================================================================================

FACT = "19:40"
ACCEPTED = ("19:40", "7:40 pm", "7:40pm")
THOUGHT = "<|channel>thought\nLet me read the results.<channel|>"
SEARCH = render_call("google_search", {"query": "last ferry pier 9"})
CLICK = render_call("click_at", {"x": 412, "y": 300})

CLASSIFIER_CASES: dict[str, tuple[list[dict[str, Any]], str]] = {
    "fact-in-final": (
        [
            commit("thought", "Read the results."),
            commit("answer", "The last ferry leaves at 19:40."),
        ],
        "correct",
    ),
    "stated-equivalent": ([commit("answer", "It leaves at 7:40 pm tonight.")], "correct"),
    "re-search": ([commit("thought", "Not in these."), commit("tool", SEARCH)], "re_search"),
    "fact-in-prefix-then-search": (
        [commit("answer", "I think 19:40, let me check."), commit("tool", SEARCH)],
        "re_search",
    ),
    "final-without-fact": (
        [commit("answer", "I could not find the departure time.")],
        "unsupported",
    ),
    "fact-only-in-closed-thought": (
        [
            commit("thought", "The result says 19:40 but I am unsure."),
            commit("answer", "I am not sure."),
        ],
        "unsupported",
    ),
    "thought-open-quotes-fact": (
        [
            commit("answer", "The last ferry leaves at 19:40.", thought_open=True),
            commit("thought", "Wait, is 19:40 the weekday time", thought_open=True),
        ],
        "other",
    ),
    "thought-open-only": ([commit("thought", "It is 19:40, so", thought_open=True)], "other"),
    "another-call": ([commit("answer", "Opening it."), commit("tool", CLICK)], "other"),
    "empty": ([], "other"),
    # R1 (blocker): a thought closed over two canvases. VllmClient marks every commit of canvas 0 thought_open (that
    # canvas ends inside the thought); canvas 1 closes it and answers. The final state is closed: a final answer.
    "closed-thought-over-two-canvases": (
        [
            commit("thought", "Let me read the", thought_open=True, idx=0),
            commit("thought", " results first.", idx=1),
            commit("answer", "The last ferry leaves at 19:40.", idx=1),
        ],
        "correct",
    ),
    # R1, the other direction (a "first canvas decides" mutant): closed and answering in canvas 0, then a thought
    # re-opened in canvas 1 that the stream ends inside. The final state is open: not a final answer.
    "answer-then-thought-left-open": (
        [
            commit("answer", "The last ferry leaves at 19:40.", idx=0),
            commit("thought", "Wait, is that the weekday", thought_open=True, idx=1),
        ],
        "other",
    ),
    # R2: matching is case-insensitive
    "stated-equivalent-upper-case": (
        [commit("answer", "It leaves at 7:40 PM tonight.")],
        "correct",
    ),
}


@pytest.mark.parametrize("case", list(CLASSIFIER_CASES))
def test_outcome_classifier(case: str) -> None:
    """Criterion 4's outcome classes: correct = a final answer containing the fact or a pre-registered equivalent;
    re-search = a ``google_search`` call (a call is not a final, even after a prefix that quotes the fact);
    unsupported = a final answer without the fact (a fact only inside a closed thought is not in the answer); other =
    anything else -- another call, empty, and (adversarial; reviewer mutation) a canvas that ends inside an unclosed
    thought, even one that quotes the fact: it is not a final answer, so it is never correct."""
    commits, want = CLASSIFIER_CASES[case]
    assert harness().classify_e(commits, ACCEPTED) == want


#: R2 (2026-10-17), only the forms the ruling names: case-insensitive; ``6:45 AM`` and the unpadded ``6:45`` for
#: 06:45; ``7:40 PM`` for 19:40; and the boundary negative, ``16:45`` is not ``6:45``. Through each history's own
#: accepted strings, so the rows hold whether the forms live in the pre-registered list or in the matcher.
ACCEPTED_FORM_CASES: dict[str, tuple[str, str, str]] = {
    "06:45-as-6:45-AM": ("market", "Stalls open at 6:45 AM on Saturdays.", "correct"),
    "06:45-unpadded": ("market", "Stalls open at 6:45 on Saturdays.", "correct"),
    "19:40-as-7:40-PM": ("ferry", "The last ferry leaves at 7:40 PM.", "correct"),
    # adversarial (a guard too wide): an unbounded "6:45" would match inside 16:45
    "16:45-is-not-6:45": ("market", "The Wednesday market opens at 16:45.", "unsupported"),
    # N1b (reviewer mutant: the trailing boundary deleted): a longer number that *starts* with the fact is not it
    "41.65-is-not-41.6": ("lighthouse", "The tower is 41.65 metres tall.", "unsupported"),
    "19:401-is-not-19:40": ("ferry", "The timetable code is 19:401.", "unsupported"),
    # ... and the trailing boundary is not too wide: sentence punctuation after the fact still holds it
    "1,187-then-full-stop": ("hall", "The main auditorium seats 1,187.", "correct"),
}


@pytest.mark.parametrize("case", list(ACCEPTED_FORM_CASES))
def test_accepted_forms(case: str) -> None:
    """Ruling R2: the 12-hour and unpadded forms of a history's time are accepted, case-insensitively, and the
    match has number boundaries: ``16:45`` holds ``6:45`` as a substring and must not read as the market's 06:45
    (the too-wide direction; the other rows are the absent direction)."""
    hid, answer, want = ACCEPTED_FORM_CASES[case]
    h = harness()
    hist = next(x for x in h.E_HISTORIES if x.id == hid)
    assert h.classify_e([commit("answer", answer)], accepted(hist)) == want, (
        f"{hid} (accepted {accepted(hist)}): {answer!r}"
    )


Q4_CLASSIFIER_CASES: dict[str, tuple[list[dict[str, Any]], str]] = {
    "final": ([commit("thought", "Read it."), commit("answer", "It is sunny.")], "final"),
    "action-call": ([commit("answer", "Clicking it."), commit("tool", CLICK)], "call"),
    "search-call": ([commit("tool", SEARCH)], "call"),
    "thought-open": ([commit("thought", "Hmm, the page", thought_open=True)], "thought_open"),
    "closed-thought-no-answer": ([commit("thought", "Nothing to add.")], "empty"),
    "empty": ([], "empty"),
    # R1's twin: a thought closed over two canvases, then an answer -- the final state is closed
    "closed-thought-over-two-canvases": (
        [
            commit("thought", "Let me read the", thought_open=True, idx=0),
            commit("thought", " page first.", idx=1),
            commit("answer", "It is sunny tomorrow.", idx=1),
        ],
        "final",
    ),
    "answer-then-thought-left-open": (
        [
            commit("answer", "It is sunny.", idx=0),
            commit("thought", "Though the forecast", thought_open=True, idx=1),
        ],
        "thought_open",
    ),
}


@pytest.mark.parametrize("case", list(Q4_CLASSIFIER_CASES))
def test_q4_classifier(case: str) -> None:
    """Criterion 2's step-2 outcome classes (R7 pins them): ``call`` a tool commit; ``thought_open`` a canvas whose
    **final** state is inside a thought (R1); ``final`` a non-empty answer; else ``empty``."""
    commits, want = Q4_CLASSIFIER_CASES[case]
    assert harness().classify_q4(commits) == want


# =====================================================================================================
# Criterion 5: the E bars at their boundaries
# =====================================================================================================


def tally(**edits: tuple[int, ...]) -> dict[str, dict[str, dict[str, int]]]:
    """The baseline (every bar passing) with ``<placement>_<arm>=(correct, re_search, unsupported, other[, errored])``
    edits; every cell carries ``errored`` (ruling R4), 0 unless a fifth count is given, inside the cell's 40.

    Baseline: kept FULL 36/2/2/0, kept COMPACT 36/2/2/0, old FULL 36/2/2/0, old COMPACT 20/14/4/2. EC 36 >= 28
    both; E1 36 >= 36 - 4; E2 20 + 14 = 34 >= 32; E3 4 <= 8. Old COMPACT correct 20 < 32 = old FULL - 4: E1 taken
    within the old placement would fail the baseline (reviewer mutation)."""
    base: dict[str, tuple[int, ...]] = {
        "kept_FULL": (36, 2, 2, 0),
        "kept_COMPACT": (36, 2, 2, 0),
        "old_FULL": (36, 2, 2, 0),
        "old_COMPACT": (20, 14, 4, 2),
    }
    base.update(edits)
    out: dict[str, dict[str, dict[str, int]]] = {"old": {}, "kept": {}}
    for key, counts in base.items():
        assert sum(counts) == N, f"fixture {key} sums to {sum(counts)}"
        assert len(counts) in (4, 5), f"fixture {key}: {counts}"
        placement, arm = key.split("_")
        full = (*counts, 0) if len(counts) == 4 else counts
        out[placement][arm] = dict(zip((*OUTCOMES, ERRORED), full, strict=True))
    return out


#: case -> (edits, the failing bars, or None for inconclusive)
RULE_CASES: dict[str, tuple[dict[str, tuple[int, int, int, int]], frozenset[str] | None]] = {
    "baseline": ({}, frozenset()),
    "E1-32-of-36-passes": ({"kept_COMPACT": (32, 4, 4, 0)}, frozenset()),
    "E1-31-of-36-fails": ({"kept_COMPACT": (31, 5, 4, 0)}, frozenset({"E1"})),
    "E1-is-relative-31-of-35-passes": (
        {"kept_FULL": (35, 3, 2, 0), "kept_COMPACT": (31, 5, 4, 0)},
        frozenset(),
    ),
    "E2-32-passes": ({"old_COMPACT": (20, 12, 5, 3)}, frozenset()),
    "E2-31-fails": ({"old_COMPACT": (20, 11, 5, 4)}, frozenset({"E2"})),
    "E3-8-passes": ({"old_COMPACT": (18, 14, 8, 0)}, frozenset()),
    # 9 unsupported of 40 leaves at most 31 for correct + re-search, so E2 fails with it (see the report)
    "E3-9-fails": ({"old_COMPACT": (17, 14, 9, 0)}, frozenset({"E2", "E3"})),
    "EC-28-kept-holds": ({"kept_FULL": (28, 6, 6, 0)}, frozenset()),
    "EC-28-old-holds": ({"old_FULL": (28, 6, 6, 0)}, frozenset()),
    "EC-27-kept-inconclusive": ({"kept_FULL": (27, 7, 6, 0)}, None),
    "EC-27-old-inconclusive": ({"old_FULL": (27, 7, 6, 0)}, None),
}

BAR_RE = re.compile(r"\b(?:EC|E[123])\b")


@pytest.mark.parametrize("case", list(RULE_CASES))
def test_e_rule_boundaries(case: str) -> None:
    """Criterion 5 (H-T115a-1, exact fractions of 40), each boundary with every other bar at a passing value:
    E1 kept COMPACT >= kept FULL - 4 (36: 32 passes, 31 fails; relative, so 31 of 35 passes); E2 old COMPACT
    correct + re-search >= 32 (32 passes, 31 fails); E3 old COMPACT unsupported <= 8 (8 passes, 9 fails); EC FULL
    correct >= 28 at both placements (28 holds, 27 at either placement is inconclusive). The outcome is
    "elision viable" iff all hold, "elision hurts: <bars>" naming exactly the failing bars, else inconclusive."""
    edits, failing = RULE_CASES[case]
    got = harness().e_outcome(tally(**edits))
    outcome = str(got["outcome"])
    if failing is None:
        assert got["bars"]["EC"] is False, got["bars"]
        assert outcome.startswith("inconclusive"), outcome
        return
    assert got["bars"] == {b: b not in failing for b in BARS}, got["bars"]
    if not failing:
        assert outcome.startswith("elision viable"), outcome
    else:
        assert outcome.startswith("elision hurts"), outcome
        assert set(BAR_RE.findall(outcome)) == set(failing), outcome


#: Ruling R4: errored E requests >= 3/160 make E inconclusive; every bar is held at a passing value (the errored
#: requests replace an ``unsupported`` or ``other`` count, so no correct count moves). case -> (edits, inconclusive?)
ERRORED_RULE_CASES: dict[str, tuple[dict[str, tuple[int, ...]], bool]] = {
    # 2/160, in two cells: below the line
    "errored-2-of-160-viable": (
        {"kept_FULL": (36, 2, 1, 0, 1), "old_COMPACT": (20, 14, 4, 1, 1)},
        False,
    ),
    # 3/160, one in each of three cells (a per-cell count would see 1 and miss it)
    "errored-3-of-160-spread-inconclusive": (
        {
            "kept_FULL": (36, 2, 1, 0, 1),
            "kept_COMPACT": (36, 2, 1, 0, 1),
            "old_COMPACT": (20, 14, 4, 1, 1),
        },
        True,
    ),
    "errored-3-of-160-one-cell-inconclusive": ({"old_FULL": (36, 1, 0, 0, 3)}, True),
}


@pytest.mark.parametrize("case", list(ERRORED_RULE_CASES))
def test_e_errored_rule(case: str) -> None:
    """Ruling R4 (flagged for the human: it can only make the outcome more cautious): with every bar at a passing
    value, 2 errored E requests of 160 leave "elision viable" (all four bars hold), and 3 -- spread over three cells
    or in one -- make E "inconclusive", the reason naming the errored requests."""
    edits, inconclusive = ERRORED_RULE_CASES[case]
    t = tally(**edits)
    assert sum(t[p][a][ERRORED] for p in PLACEMENTS for a in ("FULL", "COMPACT")) == (
        3 if inconclusive else 2
    )
    got = harness().e_outcome(t)
    outcome = str(got["outcome"])
    if inconclusive:
        assert outcome.startswith("inconclusive"), outcome
        assert "errored" in outcome.lower(), (
            f"the reason does not name the errored requests: {outcome}"
        )
    else:
        assert got["bars"] == dict.fromkeys(BARS, True), got["bars"]
        assert outcome.startswith("elision viable"), outcome


def _e_line(hid: str, placement: str, arm: str, rep: int, outcome: str) -> dict[str, Any]:
    errored = outcome == ERRORED
    return {
        "row": "E",
        "request_id": f"t0115a-e-{hid}-{placement}-{arm}-r{rep}",
        "history": hid,
        "placement": placement,
        "arm": arm,
        "repeat": rep,
        "status": None if errored else 200,
        "error": "ReadTimeout('mock')" if errored else None,
        "outcome": outcome,
        "commits": [],
    }


def test_e_report_counts_errored(tmp_path: Path) -> None:
    """Ruling R4: the E report gives errored k/n per arm and placement -- ``tally[p][a]["errored"]``, inside the
    cell's n. Hand lines: old FULL 3 correct + 1 errored; old COMPACT 2 unsupported + 2 errored; kept FULL 4 correct;
    kept COMPACT 1 correct + 1 re-search + 1 errored. Expected tally typed by hand below (an errored line dropped
    from the tally, or counted as any of the four, differs)."""
    spec = {
        ("old", "FULL"): ["correct"] * 3 + [ERRORED],
        ("old", "COMPACT"): ["unsupported"] * 2 + [ERRORED] * 2,
        ("kept", "FULL"): ["correct"] * 4,
        ("kept", "COMPACT"): ["correct", "re_search", ERRORED],
    }
    lines = [
        _e_line("ferry", p, a, rep, o)
        for (p, a), outs in spec.items()
        for rep, o in enumerate(outs)
    ]
    run = tmp_path / "run"
    run.mkdir()
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    want = {
        "old": {
            "FULL": {"correct": 3, "re_search": 0, "unsupported": 0, "other": 0, "errored": 1},
            "COMPACT": {"correct": 0, "re_search": 0, "unsupported": 2, "other": 0, "errored": 2},
        },
        "kept": {
            "FULL": {"correct": 4, "re_search": 0, "unsupported": 0, "other": 0, "errored": 0},
            "COMPACT": {"correct": 1, "re_search": 1, "unsupported": 0, "other": 0, "errored": 1},
        },
    }
    got = harness().build_report(run)["E"]["tally"]
    assert got == want, got


#: R7: hand Q4 lines, by arm: (outcome, repeats_start, errored). The report's shares are k/n over the arm's lines.
Q4_REPORT_LINES: dict[str, list[tuple[str, bool, bool]]] = {
    "G": [("final", True, False), ("final", False, False), ("call", False, False)],
    "A": [
        ("final", True, False),
        ("thought_open", False, False),
        ("empty", False, False),
        ("empty", False, True),
    ],
}
#: By hand: G n 3 -- final 2, call 1, thought_open 0, empty 0, repeats_start 1, errored 0; A n 4 -- final 1,
#: thought_open 1, empty 2, call 0, repeats_start 1, errored 1. (Shares over all 7 Q4 lines, or over the other
#: arm, differ in every n.)
Q4_REPORT_WANT: dict[str, dict[str, Any]] = {
    "G": {
        "outcomes": {
            "final": {"k": 2, "n": 3},
            "call": {"k": 1, "n": 3},
            "thought_open": {"k": 0, "n": 3},
            "empty": {"k": 0, "n": 3},
        },
        "repeats_start": {"k": 1, "n": 3},
        "errored": {"k": 0, "n": 3},
    },
    "A": {
        "outcomes": {
            "final": {"k": 1, "n": 4},
            "call": {"k": 0, "n": 4},
            "thought_open": {"k": 1, "n": 4},
            "empty": {"k": 2, "n": 4},
        },
        "repeats_start": {"k": 1, "n": 4},
        "errored": {"k": 1, "n": 4},
    },
}


def test_q4_report_by_hand(tmp_path: Path) -> None:
    """Criterion 2's report-only Q4 row (R7): per arm, the outcome shares, the start-repetition rate and the errored
    count as k/n over that arm's lines, read from ``requests.jsonl`` by ``build_report``. Expected values by hand
    (``Q4_REPORT_WANT``); the lines interleave the arms in send order."""
    lines: list[dict[str, Any]] = []
    for i in range(4):
        for arm in ("G", "A"):
            if i < len(Q4_REPORT_LINES[arm]):
                outcome, rep_start, err = Q4_REPORT_LINES[arm][i]
                lines.append(
                    {
                        "row": "Q4",
                        "request_id": f"t0115a-q4-click-01-{arm}-r{i}",
                        "case_id": "click-01",
                        "arm": arm,
                        "repeat": i,
                        "status": None if err else 200,
                        "error": "ReadTimeout('mock')" if err else None,
                        "outcome": outcome,
                        "repeats_start": rep_start,
                        "commits": [],
                    }
                )
    run = tmp_path / "run"
    run.mkdir()
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    q4 = harness().build_report(run)["Q4"]
    for arm, want in Q4_REPORT_WANT.items():
        got = q4[arm]
        assert {o: got["outcomes"][o] for o in want["outcomes"]} == want["outcomes"], (arm, got)
        assert got["repeats_start"] == want["repeats_start"], (arm, got)
        assert got["errored"] == want["errored"], (arm, got)


# =====================================================================================================
# Criterion 5's cost rows (report-only), by hand
# =====================================================================================================


def _rec(
    hid: str, placement: str, arm: str, rep: int, q: int, hit: int, fc: int, done: int
) -> dict[str, Any]:
    return {
        "row": "E",
        "history": hid,
        "placement": placement,
        "arm": arm,
        "repeat": rep,
        "prefix_query_tokens": q,
        "prefix_hit_tokens": hit,
        "first_commit_ms": fc,
        "done_ms": done,
    }


def test_e_cost_by_hand() -> None:
    """Criterion 5, report-only cost: the median COMPACT miss (query - hit) over COMPACT records only -- a warm
    record (6000 missed: the cold prefix) and the FULL records (100 missed) are not COMPACT costs; and the *paired*
    median COMPACT - FULL differences, paired by (history, placement, repeat), not the difference of the medians.

    By hand: COMPACT misses 3000, 2500, 4000 -> median 3000 (with the warm counted: 3500). First commit pairs
    (FULL, COMPACT) (100, 300), (200, 210), (300, 320) -> differences 200, 10, 20 -> median 20 (unpaired: 300 - 200
    = 100). Done (1000, 1100), (2000, 2900), (3000, 3050) -> 100, 900, 50 -> median 100 (unpaired: 900). The
    records are shuffled so that pairing by position would be wrong."""
    records = [
        _rec("h1", "old", "warm", 0, 6000, 0, 500, 5000),
        _rec("h1", "old", "COMPACT", 0, 5000, 2000, 300, 1100),
        _rec("h2", "kept", "FULL", 1, 5100, 5000, 200, 2000),
        _rec("h1", "old", "FULL", 0, 5100, 5000, 100, 1000),
        _rec("h2", "kept", "COMPACT", 1, 5000, 2500, 210, 2900),
        _rec("h3", "old", "COMPACT", 0, 5000, 1000, 320, 3050),
        _rec("h3", "old", "FULL", 0, 5100, 5000, 300, 3000),
    ]
    got = harness().e_cost(records)
    assert got["compact_miss_tokens_median"] == 3000, got
    assert got["first_commit_ms_diff_median"] == 20, got
    assert got["done_ms_diff_median"] == 100, got
    assert "ms_per_miss_token" in got, got


#: Hand-built ``requests.jsonl`` E lines for the first-COMPACT-per-cell rows, in send order (the contract: one line
#: per request, warm included, in send order). Four (history, placement) cells -- h1 at **both** placements (ruling
#: R3: a history with both, so keying the first COMPACT by history alone is wrong), h2 kept, h3 old -- interleaved so
#: that "the first COMPACT of the file" is not "the first COMPACT of each cell". Each cell opens with a warm; FULL
#: right after the warm misses its fifth step only; the first COMPACT after the warm misses its tail (the one
#: compaction the warm makes measurable; h1/kept's is smaller: it shares h1/old's cached prefix); every later repeat is
#: a byte-identical prompt and hits (0 or a few tokens), with a near-zero paired latency difference.
#: Columns: history, placement, arm, repeat, prefix_query_tokens, prefix_hit_tokens, first_commit_ms, done_ms.
FIRST_PER_CELL_LINES = (
    ("h2", "kept", "warm", 0, 6200, 0, 900, 9000),
    ("h1", "old", "warm", 0, 6000, 0, 900, 9000),
    ("h2", "kept", "FULL", 0, 5900, 5100, 100, 1000),  # miss 800
    (
        "h2",
        "kept",
        "COMPACT",
        0,
        5000,
        2400,
        165,
        1130,
    ),  # miss 2600, the first COMPACT of the file; +65 / +130
    ("h1", "old", "FULL", 0, 6000, 5100, 100, 1000),  # miss 900
    ("h1", "old", "COMPACT", 0, 5000, 2000, 250, 1300),  # miss 3000; +150 / +300
    ("h3", "old", "warm", 0, 5900, 0, 900, 9000),
    ("h3", "old", "FULL", 0, 5800, 5100, 100, 1000),  # miss 700
    ("h3", "old", "COMPACT", 0, 5000, 1600, 270, 1340),  # miss 3400; +170 / +340
    ("h1", "old", "COMPACT", 1, 5000, 5000, 101, 1005),  # repeats hit from here on
    ("h1", "old", "FULL", 1, 6000, 6000, 100, 1000),  # h1/old pair r1: +1 / +5
    ("h2", "kept", "COMPACT", 1, 5000, 5000, 100, 1000),
    ("h3", "old", "COMPACT", 1, 5000, 5000, 100, 1000),
    ("h1", "kept", "warm", 0, 5300, 4200, 900, 9000),  # miss 1100: shares h1/old's prefix
    ("h1", "kept", "FULL", 0, 5900, 5100, 100, 1000),  # miss 800
    ("h1", "kept", "COMPACT", 0, 5000, 3800, 130, 1060),  # miss 1200; +30 / +60
    ("h1", "old", "warm", 2, 6000, 6000, 900, 9000),
    ("h1", "old", "FULL", 2, 6000, 6000, 100, 1000),
    (
        "h1",
        "old",
        "COMPACT",
        2,
        5000,
        4996,
        100,
        1002,
    ),  # miss 4, the last COMPACT of h1/old; pair r2: 0 / +2
    ("h1", "kept", "COMPACT", 1, 5000, 5000, 100, 1000),  # miss 0, the last COMPACT of h1/kept
    ("h1", "kept", "FULL", 1, 5900, 5900, 99, 999),  # h1/kept pair r1: +1 / +1
    ("h2", "kept", "COMPACT", 2, 5000, 4992, 100, 1000),  # miss 8
    ("h3", "old", "COMPACT", 2, 5000, 4998, 100, 1000),  # miss 2: the last COMPACT of h3/old
    ("h2", "kept", "COMPACT", 3, 5000, 5000, 100, 1000),  # miss 0: the last COMPACT of h2/kept
)
#: manifest ``predicted_miss_tokens`` (``/tokenize`` count after the first stub), one cell not measured (None)
FIRST_PER_CELL_PREDICTED = {
    "h1/old": 2900,
    "h1/kept": 1100,
    "h2/kept": 2500,
    "h3/old": 3500,
    "h4/kept": None,
}

#: By hand from FIRST_PER_CELL_LINES (pooled over both placements):
#: - first COMPACT after the warm, per cell: h1/old 3000, h1/kept 1200, h2/kept 2600, h3/old 3400
#:   -> sorted 1200 2600 3000 3400 -> (2600 + 3000) / 2 = 2800 (n 4);
#: - mutant "first COMPACT of the file": h2/kept 2600 alone -> 2600;
#: - mutant "the warm is the first request of the cell" (first line per cell, any arm): 6200, 6000, 5900, 1100
#:   -> sorted 1100 5900 6000 6200 -> 5950;
#: - mutant "the first non-warm line per cell" (warm skipped by arm, FULL taken): 800, 900, 700, 800 -> 800;
#: - mutant "miss not relative to the warm" (the first COMPACT's whole query, as if cold): 5000 x 4 -> 5000;
#: - mutant "the last COMPACT per cell": 4, 0, 2, 0 -> sorted 0 0 2 4 -> 1;
#: - mutant "keyed by history alone" (R3): h2 2600, h1 3000 (h1/kept skipped), h3 3400 -> 3000 (n 3);
#: - all-COMPACT median (the cache sanity check): 2600, 3000, 3400, 0, 0, 0, 1200, 4, 0, 8, 2, 0
#:   -> sorted 0 0 0 0 0 2 4 8 1200 2600 3000 3400 -> (2 + 4) / 2 = 3;
#: - predicted median: 2900, 1100, 2500, 3500 (None not measured) -> sorted 1100 2500 2900 3500 -> 2700.
FIRST_PER_CELL_MEDIAN = 2800
FIRST_PER_CELL_N = 4
ALL_COMPACT_MEDIAN = 3
PREDICTED_MEDIAN = 2700
FIRST_PER_CELL_MUTANT_VALUES = {
    "first COMPACT overall": 2600,
    "warm counted as the cell's first": 5950,
    "first non-warm line per cell": 800,
    "query not relative to the warm (cold)": 5000,
    "last COMPACT per cell": 1,
    "keyed by history alone": 3000,
}

#: R3, per placement, by hand from FIRST_PER_CELL_LINES (each cell's first pair is its repeat 0, FULL then COMPACT):
#: - old (h1/old, h3/old): first COMPACT misses 3000, 3400 -> 3200 (n 2); predicted 2900, 3500 -> 3200; first pairs
#:   first-commit +150, +170 -> 160; done +300, +340 -> 320; ms per miss token 320 / 3200 = 0.1 (each cell's own
#:   ratio is 0.1 too, so the median of ratios agrees);
#: - kept (h1/kept, h2/kept): 1200, 2600 -> 1900 (n 2); predicted 1100, 2500 -> 1800; first-commit +30, +65 -> 47.5;
#:   done +60, +130 -> 95; ms per miss token 95 / 1900 = 0.05 (cells 0.05 and 0.05).
#: Mutants: keyed by history alone -> kept 2600 (n 1); every pair, not the first -> old done 300, 5, 2, 340 -> 152.5,
#: first-commit 150, 1, 0, 170 -> 75.5; kept done 130, 60, 1 -> 60, first-commit 65, 30, 1 -> 30; placements pooled
#: -> done 300, 340, 60, 130 -> 215; the miss over every COMPACT of the placement -> old 3000 3400 0 0 4 2 -> 3.
BY_PLACEMENT_WANT: dict[str, dict[str, float | int]] = {
    "old": {
        "compact_miss_tokens_first_per_cell_median": 3200,
        "compact_miss_tokens_first_per_cell_n": 2,
        "predicted_miss_tokens_median": 3200,
        "first_commit_ms_diff_median": 160,
        "done_ms_diff_median": 320,
        "ms_per_miss_token": 0.1,
    },
    "kept": {
        "compact_miss_tokens_first_per_cell_median": 1900,
        "compact_miss_tokens_first_per_cell_n": 2,
        "predicted_miss_tokens_median": 1800,
        "first_commit_ms_diff_median": 47.5,
        "done_ms_diff_median": 95,
        "ms_per_miss_token": 0.05,
    },
}


def _first_per_cell_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    lines = [
        {
            "row": "E",
            "request_id": f"t0115a-e-{hid}-{p}-{arm}-r{rep}-{i}",
            "history": hid,
            "placement": p,
            "arm": arm,
            "repeat": rep,
            "prefix_query_tokens": q,
            "prefix_hit_tokens": hit,
            "first_commit_ms": fc,
            "done_ms": done,
        }
        for i, (hid, p, arm, rep, q, hit, fc, done) in enumerate(FIRST_PER_CELL_LINES)
    ]
    # the fixture's own premise, checked so the rows cannot go vacuous: every cell opens with its warm and has a
    # COMPACT; one history has both placements; every cell's first pair is its repeat 0
    cells = {(x["history"], x["placement"]) for x in lines}
    for cell in cells:
        mine = [x for x in lines if (x["history"], x["placement"]) == cell]
        assert mine[0]["arm"] == "warm" and any(x["arm"] == "COMPACT" for x in mine), cell
        assert {x["arm"] for x in mine if x["repeat"] == 0} >= {"FULL", "COMPACT"}, cell
    assert any((h, "old") in cells and (h, "kept") in cells for h, _ in cells), cells
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps({"predicted_miss_tokens": FIRST_PER_CELL_PREDICTED}), encoding="utf-8"
    )
    return run


def test_e_cost_first_compact_per_cell_by_hand(tmp_path: Path) -> None:
    """Criterion 5, report-only cost, as ruled 2026-10-17 (docs/tasks/T-0115a.md, last *Rulings* entry): under prefix
    caching every repeat after a cell's first request is a byte-identical prompt, so the cost of one compaction is read
    from **the first COMPACT request of each (history, placement) cell after its warm**:
    ``report["E"]["cost"]["compact_miss_tokens_first_per_cell_median"]`` (``..._n`` its count), beside
    ``predicted_miss_tokens_median`` (from the manifest); the all-COMPACT ``compact_miss_tokens_median`` stays as a
    cache sanity check and is near zero here.

    Hand-built lines (``FIRST_PER_CELL_LINES``, values by hand beside it), interleaved across cells. The expected
    pooled value (2800, n 4) differs from every mutant's: the first COMPACT of the file (2600), the warm as the cell's
    first (5950), the first non-warm line (800), the COMPACT's whole query as if no warm had run (5000), the last
    COMPACT per cell (1), and (R3) the first COMPACT keyed by history alone (3000, n 3). Read through
    ``build_report`` (the report is the contract; the value need not live in ``e_cost``)."""
    run = _first_per_cell_run(tmp_path)
    cost = harness().build_report(run)["E"]["cost"]
    got = cost.get("compact_miss_tokens_first_per_cell_median")
    named = {v: k for k, v in FIRST_PER_CELL_MUTANT_VALUES.items()}
    assert got == FIRST_PER_CELL_MEDIAN, (
        f"first COMPACT per cell after its warm: want {FIRST_PER_CELL_MEDIAN}, got {got!r}"
        + (f" (the value of the mutant {named[got]!r})" if got in named else "")
    )
    assert cost.get("compact_miss_tokens_first_per_cell_n") == FIRST_PER_CELL_N, cost
    # relative to the warm: the warm's own (cold) miss and the COMPACT's whole query are not the reading
    assert got != FIRST_PER_CELL_MUTANT_VALUES["warm counted as the cell's first"], cost
    assert got != FIRST_PER_CELL_MUTANT_VALUES["query not relative to the warm (cold)"], cost
    # the all-COMPACT median is the near-zero cache sanity check, kept beside the reading
    assert cost.get("compact_miss_tokens_median") == ALL_COMPACT_MEDIAN, cost
    assert cost.get("predicted_miss_tokens_median") == PREDICTED_MEDIAN, cost
    # R3: the cell is (history, placement), never the history alone
    assert got != FIRST_PER_CELL_MUTANT_VALUES["keyed by history alone"], cost


def test_e_cost_by_placement_first_pair_by_hand(tmp_path: Path) -> None:
    """Ruling R3 (2026-10-17): the report-only cost is split by placement -- ``report["E"]["cost"]["by_placement"]
    [placement]``, keys named in this module's docstring. The **old** placement's first COMPACT per cell is the
    one-compaction cost; the **kept** placement shares a cached prefix with the same history's old COMPACT and is
    reported separately. The paired latency (first-commit and done, COMPACT - FULL) and ``ms_per_miss_token`` come
    from each cell's **first pair** only. Every expected value is typed by hand beside ``BY_PLACEMENT_WANT``; it
    differs from the history-alone key (kept 2600, n 1), from every-pair latency (old done 152.5, kept 60), from the
    placements pooled (done 215) and from the all-COMPACT miss (old 3)."""
    run = _first_per_cell_run(tmp_path)
    cost = harness().build_report(run)["E"]["cost"]
    split = cost.get("by_placement")
    assert isinstance(split, Mapping) and set(split) >= set(PLACEMENTS), (
        f"R3: no per-placement cost under cost['by_placement']: {sorted(cost)}"
    )
    for placement, want in BY_PLACEMENT_WANT.items():
        got = split[placement]
        for key, value in want.items():
            assert got.get(key) == pytest.approx(value), (
                f"R3 {placement} {key}: want {value} (by hand), got {got.get(key)!r}"
            )


# =====================================================================================================
# Criterion 1: refusals that need no tree, and the import
# =====================================================================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://10.255.255.1:8001/v1",
        "http://localhost:8001/v1",
        "http://127.0.0.1.nip.io:8001/v1",
        "http://192.168.1.10:8001/v1",
    ],
)
def test_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str], url: str) -> None:
    """Criterion 1 (T-0115 criterion 2's first refusal): a ``--base-url`` that is not an IP literal in 127.0.0.0/8
    or ``::1`` is refused with exit 2 before any socket (no DNS lookup either), nothing is written, and the refusal
    says loopback. ``127.0.0.1.nip.io`` starts with ``http://127.0.0.1``, which ``VllmEndpoint``'s prefix check
    accepts: the harness must parse the host. The other refusals, which need a clean tree and a live mock, are
    ``tests/integration/test_t0115a_harness_smoke.py``'s."""
    h = harness()
    out = tmp_path / "t0115a-out"
    with no_sockets() as attempts:
        rc = h.main(["--base-url", url, "--out-dir", str(out)])
    said = capsys.readouterr()
    assert rc == 2, f"{url}: exit {rc}"
    assert attempts == [], f"{url}: sockets before the refusal: {attempts}"
    assert not out.exists() or list(out.iterdir()) == [], "the refusal wrote a run directory"
    assert "loopback" in (said.out + said.err).lower(), (said.out + said.err)[-300:]


def test_import_opens_no_socket() -> None:
    """Criterion 1: importing the harness opens no socket, loopback included (everything that talks to the engine is
    imported or opened inside ``main``)."""
    with no_sockets() as attempts:
        harness()
    assert attempts == [], attempts
