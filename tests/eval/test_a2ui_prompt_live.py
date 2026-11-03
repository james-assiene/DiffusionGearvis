"""Q12 — A2UI channel uptake: does the prompt actually make a surface happen? (T-0044 criterion 9)

``gpu``. Run through ``/launch`` on the 5090 box with the engine up::

    uv run pytest tests/eval/test_a2ui_prompt_live.py -q -m gpu -rA

What it does
============
Four cases (``a2ui_prompt_cases.jsonl``), n = 3 each, straight through the three real objects the
task is about and nothing else: the real ``PromptBuilder`` (so the block under test is the one in
the cached prefix), the real ``VllmClient`` (so the fence arrives the way the engine really emits
it, one SSE chunk = one committed canvas) and the real ``UiGate``. No ``Session``, no STT, no TTS
— a full-session run is the human's demo and is recorded in PROGRESS *Runs*, not asserted here
(task ambiguity 7).

The rows
========
* three **positives** ("show me a table of the first five primes and their squares", "make a
  flashcard for *ephemeral*", "illustrate one half and three quarters"): at least 2 of 3 runs
  must produce a ``ui`` segment that the gate turns into a surface with ≥ 1 accepted component.
  0 of 3 fails the row;
* one **negative** ("what is the capital of France"): 3 of 3 runs produce no ``ui`` segment and no
  ``<a2ui>`` anywhere in the answer text. A surface here means the block over-triggers, and the
  fix is the block's wording — never this assertion;
* **adversarial**: a surface whose fence spanned more than one canvas fails its own row and names
  the spec question (``llm_client`` does not carry fence state across chunks — amendments B7/M7 —
  while docs/12 §1.1 describes progressive rendering; task ambiguity 2). It must not be tolerated
  because the gate happened to salvage the first canvas.

Regime (state it with every number this produces)
=================================================
profile ``lean``; canvas 256, ``max_denoising_steps`` 48, thinking **on**, sampling = the
whitelist defaults only (``LlmOptions()``), ``xargs`` none — canvas and steps are engine-global
(``--diffusion-config``, ``services/versions.lock``); prompt built with **frame None, no history,
no memory hits, assists off** — deliberately *not* the pixel-first turn shape, so these numbers do
not describe the live demo; vLLM container, model and revision as ``services/versions.lock`` pins;
4 cases × n = 3 = 12 generations; seed n/a (the diffusion path 400s on ``seed``). Wall time per
case is recorded but is **not** a bench number (n = 3, no bench harness): quote it as an
observation only.

Every run writes one JSON line to ``logs/t0044-a2ui-eval-<git-sha>.jsonl`` (``logs/``, never
``artifacts/``): case, run index, ui commits, canvases spanned, surface accepted, component kinds,
``ui_repairs``, ``ui_resamples``, answer text, prompt/surface token counts from the engine's own
``/tokenize`` (``null`` when the endpoint does not answer — there is no local tokenizer, T-0044
ambiguity 4), wall time and the regime block. The per-turn session log (T-0043) is the diagnostic
for a miss in production: ``ui_valid: null`` means no ``ui`` segment ever reached the gate,
``ui_valid: false`` means one did and failed.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, VllmClient, VllmEndpoint
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.conductor.ui_gate import UiGate
from local_astra.config import Profile
from local_astra.contracts.llm import LlmOptions
from local_astra.contracts.perception import AssistFlags
from tests.fakes.llm import fence_lifecycle

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).with_name("a2ui_prompt_cases.jsonl")

#: Criterion 9: n = 3 per case; a positive case needs 2 of 3, a negative one 3 of 3.
RUNS_PER_CASE = 3
POSITIVE_MIN_SURFACES = 2

#: The whole generation, not a latency budget: a diffusion turn on this box takes seconds, and a
#: hung request must fail the eval rather than hang the box.
REQUEST_TIMEOUT_S = 180.0


def _cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


CASES = _cases()
POSITIVES = [c for c in CASES if c["expect_surface"]]
NEGATIVES = [c for c in CASES if not c["expect_surface"]]


@dataclass
class RunRecord:
    """One generation. Everything criterion 9 asks the harness to record."""

    case: str
    request: str
    expect_surface: bool
    run: int
    ui_commits: int = 0
    canvases_spanned: int = 0
    surface_accepted: bool = False
    components: list[str] = field(default_factory=list)
    ui_repairs: int = 0
    ui_resamples: int = 0
    ui_valid: bool | None = None
    answer_text: str = ""
    thought_canvases: int = 0
    #: criterion 9's literal check: a fence marker in the text that would be spoken
    fence_in_answer: bool = False
    #: T-0052 criterion 16: how the fence ended, as the Commit contract reports it (written /
    #: inferred / open / abandoned). Read by difference from Commit's pre-T-0052 fields, so the
    #: harness does not have to guess the field's spelling.
    fence_close: str = "none"
    #: the block's rule 'never name the fence in speech' (criterion 4), observed live
    names_the_fence_in_speech: bool = False
    prompt_tokens: int | None = None
    surface_tokens: int | None = None
    wall_s: float = 0.0
    gate_problems: list[str] = field(default_factory=list)
    regime: dict[str, Any] = field(default_factory=dict)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _regime() -> dict[str, Any]:
    lock = tomllib.loads((REPO / "services" / "versions.lock").read_text())
    profile = Profile(name="lean")
    return {
        "profile": profile.name,
        "canvas_length": profile.diffusion.canvas_length,
        "max_denoising_steps": profile.diffusion.max_denoising_steps,
        "thinking": profile.diffusion.thinking,
        "vllm_container": lock["gpu"]["vllm_container"],
        "model": lock["gpu"]["model"],
        "model_revision": lock["gpu"]["model_revision"],
        "attention_backend": lock["gpu"]["vllm_attention_backend"],
        "sampling": "whitelist defaults only (LlmOptions())",
        "xargs": None,
        "prompt_shape": "frame=None, history=[], memory_hits=(), assists off",
        "n_per_case": RUNS_PER_CASE,
        "seed": "n/a (the diffusion path 400s on seed)",
        "git_commit": _git_commit(),
        "harness": "PromptBuilder + VllmClient + UiGate (no Session, no STT/TTS)",
    }


async def _tokens(client: httpx.AsyncClient, model: str, text: str) -> int | None:
    """The **engine's** count of ``text``, or None. Never a local estimate (T-0005 B10)."""
    for path in ("/tokenize", "../tokenize"):
        try:
            response = await client.post(path, json={"model": model, "prompt": text})
            response.raise_for_status()
            return int(response.json()["count"])
        except (httpx.HTTPError, KeyError, ValueError):
            continue
    return None  # no local tokenizer stands in for the engine (T-0044 ambiguity 4)


async def _one_run(case: dict[str, Any], run: int, regime: dict[str, Any]) -> RunRecord:
    endpoint = VllmEndpoint(request_timeout_s=REQUEST_TIMEOUT_S)
    builder = PromptBuilder(DEFAULT_SYSTEM_PROMPT)
    prefix = builder.prefix([], None, AssistFlags())
    prompt = builder.full(prefix, case["request"], None, ())

    record = RunRecord(
        case=case["id"],
        request=case["request"],
        expect_surface=bool(case["expect_surface"]),
        run=run,
        regime=regime,
    )
    gate = UiGate()
    client = VllmClient(endpoint)
    request_id = f"t0044-{case['id']}-{run}"
    ui_canvases: set[int] = set()
    answer: list[str] = []
    ui_text: list[str] = []
    started = time.monotonic()
    try:
        async for commit in client.generate(prompt, request_id, LlmOptions()):
            if commit.segment == "answer" and not commit.thought_open:
                answer.append(commit.text)
            elif commit.segment == "thought":
                record.thought_canvases += 1
            elif commit.segment == "ui":
                record.ui_commits += 1
                ui_canvases.add(commit.canvas_idx)
                ui_text.append(commit.text)
                record.fence_close = repr(fence_lifecycle(commit)) or "none"
                result = gate.gate(
                    commit.text,
                    turn_id=request_id,
                    canvas_idx=commit.canvas_idx,
                    final=True,
                )
                record.ui_repairs += result.repaired
                record.ui_resamples += len(result.resample)
                record.ui_valid = result.report.ok
                record.gate_problems.extend(result.report.problems)
                if result.surface is not None:
                    record.surface_accepted = True
                    record.components = [
                        str(c.get("component"))
                        for m in result.surface.messages
                        if isinstance(m, dict) and isinstance(m.get("updateComponents"), dict)
                        for c in m["updateComponents"]["components"]
                    ]
    finally:
        await client.aclose()
    record.wall_s = round(time.monotonic() - started, 3)
    record.answer_text = "".join(answer)
    record.canvases_spanned = len(ui_canvases)
    record.fence_in_answer = UI_OPEN in record.answer_text or UI_CLOSE in record.answer_text
    record.names_the_fence_in_speech = "a2ui" in record.answer_text.lower()

    async with httpx.AsyncClient(base_url=endpoint.base_url, timeout=30.0) as http:
        record.prompt_tokens = await _tokens(
            http, endpoint.model, "\n".join(m.text() for m in prompt.messages)
        )
        record.surface_tokens = (
            await _tokens(http, endpoint.model, "".join(ui_text)) if ui_text else 0
        )
    return record


async def _all_runs() -> list[RunRecord]:
    regime = _regime()
    records = []
    for case in CASES:
        for run in range(RUNS_PER_CASE):
            records.append(await _one_run(case, run, regime))
    return records


@pytest.fixture(scope="session")
def eval_runs() -> list[RunRecord]:
    """12 generations, once per session, written to ``logs/`` as they are asserted on."""
    endpoint = VllmEndpoint()
    try:
        httpx.get(endpoint.base_url + "/models", timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:  # the box is not serving: skip, never a silent pass
        pytest.skip(f"vLLM is not answering at {endpoint.base_url} ({exc}); start it with /launch")

    records = asyncio.run(_all_runs())
    out = REPO / "logs" / f"t0044-a2ui-eval-{_git_commit()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))
    print(f"\n[T-0044 eval] {len(records)} runs -> {out}")
    for r in records:
        print(
            f"  {r.case:>20} #{r.run}: ui_commits={r.ui_commits} canvases={r.canvases_spanned} "
            f"surface={r.surface_accepted} repairs={r.ui_repairs} resamples={r.ui_resamples} "
            f"surface_tokens={r.surface_tokens} wall={r.wall_s}s"
        )
    return records


def _for(records: list[RunRecord], case_id: str) -> list[RunRecord]:
    return [r for r in records if r.case == case_id]


@pytest.mark.pending(task="T-0044")
@pytest.mark.parametrize("case", POSITIVES, ids=[c["id"] for c in POSITIVES])
def test_a_show_me_request_produces_an_accepted_surface(eval_runs, case):
    """Criterion 9, positives: ≥ 2 of 3 runs give a gated surface with ≥ 1 component.

    This is the row the origin of the task is about: on 2026-09-15 "illustrate this with A2UI"
    produced prose and ``ui_valid: null`` — no ``ui`` segment ever reached the gate. A case at
    0 of 3 fails; at 1 of 3 it fails too, and the answer is the block's wording, not this number.
    """
    runs = _for(eval_runs, case["id"])
    assert len(runs) == RUNS_PER_CASE, f"expected {RUNS_PER_CASE} runs, got {len(runs)}"

    accepted = [r for r in runs if r.surface_accepted and r.components]
    detail = "\n".join(
        f"  run {r.run}: ui_commits={r.ui_commits} valid={r.ui_valid} "
        f"components={r.components} problems={r.gate_problems[:2]} answer={r.answer_text[:120]!r}"
        for r in runs
    )
    assert len(accepted) >= POSITIVE_MIN_SURFACES, (
        f"criterion 9: {case['id']!r} produced a surface in {len(accepted)}/{RUNS_PER_CASE} runs "
        f"(needs {POSITIVE_MIN_SURFACES}). The prompt block is what changes, not this threshold."
        f"\n{detail}"
    )
    for r in accepted:
        assert r.ui_valid is True and r.components, r


@pytest.mark.pending(task="T-0044")
@pytest.mark.parametrize("case", NEGATIVES, ids=[c["id"] for c in NEGATIVES])
def test_a_plain_question_produces_no_surface(eval_runs, case):
    """Criterion 9, the negative: 3 of 3 runs, no ``ui`` segment and no fence in the answer.

    A block that makes the model draw a card for "what is the capital of France" has traded one
    failure for another: every turn then spends canvas on a surface nobody asked for, and the
    shown channel stops meaning anything. ``<a2ui>`` appearing in the *answer* text is the other
    half — it means the model wrote a fence the splitter did not recognise as one, which would be
    read aloud.
    """
    runs = _for(eval_runs, case["id"])
    assert len(runs) == RUNS_PER_CASE

    offenders = [r for r in runs if r.ui_commits or r.fence_in_answer]
    assert not offenders, (
        f"criterion 9: {case['id']!r} produced a surface (or a fence in speech) in "
        f"{len(offenders)}/{RUNS_PER_CASE} runs; the block over-triggers.\n"
        + "\n".join(
            f"  run {r.run}: ui_commits={r.ui_commits} fence_in_answer={r.fence_in_answer} "
            f"answer={r.answer_text[:160]!r}"
            for r in offenders
        )
    )


@pytest.mark.pending(task="T-0052")
def test_a_surface_that_spans_more_than_one_canvas_is_reassembled(eval_runs):
    """Criterion 16, inverted by T-0052 (ruling A4; T-0044 ambiguity Q2 closed in the direction it
    named — change the client, not the eval).

    This row used to **fail the run** when a fence spanned more than one canvas, because
    ``llm_client`` split the tail into ``answer`` (B7/M7) and the model was told to finish the
    fence inside one canvas. On 2026-09-16 the model wrote a flight-booking interface that needed
    two canvases and 807 characters of A2UI JSON were read out loud. After T-0052 a straddling
    fence is *supported*, so the assertion is the opposite one: a surface spanning k > 1 canvases
    is reassembled into one accepted surface, and no fence marker or fragment of the payload
    reaches the spoken channel.

    Stays ``pending`` until a ``/launch`` run takes it: same three positive cases plus the negative
    control, n = 3, regime in ``_regime()``.
    """
    straddled = [r for r in eval_runs if r.canvases_spanned > 1]
    for record in straddled:
        assert record.surface_accepted, (
            f"{record.case} run {record.run}: a fence spanning {record.canvases_spanned} canvases "
            f"produced no accepted surface. gate problems: {record.gate_problems}"
        )
        assert not record.fence_in_answer, (
            f"{record.case} run {record.run}: fence markup reached the spoken channel: "
            f"{record.answer_text[:160]!r}"
        )
        assert record.answer_text.strip(), (
            f"{record.case} run {record.run}: the surface was reassembled and the turn went "
            "SILENT — the carry swallowed the prose (T-0052 ruling A1)"
        )
        assert '{"' not in record.answer_text, (
            f"{record.case} run {record.run}: interface data is in the spoken text: "
            f"{record.answer_text[:160]!r}"
        )
        assert record.fence_close != "none", (
            f"{record.case} run {record.run}: the commit does not say how the fence ended "
            "(T-0052 criterion 12); reassembled, abandoned and never-there are indistinguishable"
        )
    print(
        "\n[T-0052 criterion 16] canvases_spanned distribution: "
        + repr(sorted(r.canvases_spanned for r in eval_runs if r.ui_commits))
        + " fence_close: "
        + repr(sorted({r.fence_close for r in eval_runs if r.ui_commits}))
    )


def test_no_run_says_the_fence_out_loud(eval_runs):
    """Criterion 4's spoken-channel rule, live: the model never names the fence in speech.

    Not pending, for the same reason as the row above: vacuously true when nothing is shown.

    ``split_segments`` routes a real ``<a2ui>`` to the ui channel, so the literal marker can never
    reach the answer text — what does reach it is the model *talking about* the mechanism ("I'll
    put that in the a2ui panel"), which TTS then reads aloud. The block tells it not to; this row
    says whether the telling worked. A failure is fixed in the block's wording (the spec), never
    by deleting the row.
    """
    talkative = [r for r in eval_runs if r.names_the_fence_in_speech or r.fence_in_answer]
    assert not talkative, "the spoken channel mentions the ui fence:\n" + "\n".join(
        f"  {r.case} run {r.run}: {r.answer_text[:200]!r}" for r in talkative
    )


def test_every_run_is_recorded_with_its_regime(eval_runs):
    """The harness's own contract (not pending): a number without its regime cannot be compared.

    CLAUDE.md: state the whole regime beside every number. This row fails if the log line loses
    the model, the canvas, the prompt shape or the commit — which is how an eval result becomes
    unusable three weeks later.
    """
    out = REPO / "logs" / f"t0044-a2ui-eval-{_git_commit()}.jsonl"
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == len(CASES) * RUNS_PER_CASE == len(eval_runs)

    required = (
        "profile",
        "canvas_length",
        "max_denoising_steps",
        "thinking",
        "vllm_container",
        "model",
        "model_revision",
        "prompt_shape",
        "n_per_case",
        "git_commit",
    )
    for line in lines:
        missing = [key for key in required if not line["regime"].get(key)]
        assert not missing, f"the log line for {line['case']} lost {missing} of its regime"
        for key in (
            "ui_commits",
            "canvases_spanned",
            "surface_accepted",
            "ui_repairs",
            "ui_resamples",
            "answer_text",
            "wall_s",
            "fence_close",
        ):
            assert key in line, f"criterion 9 asks the harness to record {key!r}"
