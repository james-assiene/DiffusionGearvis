"""T-0055 criteria 13-14 — does the model write the surface in its **reply**? (``gpu``, /launch)

Run on the 5090 box with the engine up, through ``/launch``: the selection is this module, marker
``gpu``, plus ``--pending`` so the rows below execute as ordinary tests instead of strict xfail.

The defect this measures
========================
Twice on 2026-09-16 the model built a complete, correctly fenced, root-bearing A2UI surface — 15
components, ``createSurface`` + ``updateComponents`` — **entirely inside its reasoning channel**
(``logs/sessions/20260916-181929-3900425.jsonl``, ``logs/sessions/20260916-215207-2034852.jsonl``
turn ``t1``), then wrote, still inside the thought, ``I will say "I have created a flight booking
interface for you."`` and said exactly that aloud. The thinking channel is discarded by design, so
no ``ui`` segment was produced and no ``ui.gate`` record exists in either log: the human heard a
fluent, confident claim and looked at an empty panel. T-0055 rewrites the instruction block so the
block says *where* the surface goes; this module is the only thing that can say whether the
rewording worked.

Two arms, one engine
====================
``new`` is the block as merged; ``baseline`` is the block that failed, read verbatim from
``tests/fixtures/prompts/ui_block_t0044.txt``. The arms are **interleaved per run** so engine
drift cannot land on one of them. The baseline arm is installed by setting
``local_astra.conductor.prompt_builder.UI_INSTRUCTIONS`` — ``prompt_builder`` imports the *name*,
so patching the attribute on **that** module is what takes effect; patching ``ui_instructions`` is
a no-op. This is a test-only mechanism and exists nowhere in the product.

Criterion 14 is why the baseline arm is not decoration: if the baseline does not reproduce the
defect in this regime, the run **fails with "baseline did not reproduce"** rather than passing. A
green light earned by a regime that cannot show the bug is worth nothing, and the finding would be
that reproduction needs the desk's conditions (memory hits, a live session, a frame) — which is
what case ``flight_booking_with_memory`` exists to test.

This fix is necessary and not sufficient
========================================
In the 21:52 log the thought-fence itself straddled a canvas boundary (``<a2ui>`` in canvas 1,
``</a2ui>`` in canvas 2). T-0055 moves the surface into the channel the client reads; **T-0052**
makes the client able to read it when it spans more than one canvas. ``canvases_spanned`` and the
fence-lifecycle field are recorded per run and read by the reviewer so the two can be told apart:
a surface that lands in the reply and spans two canvases is a T-0055 success and a T-0052
exercise. Record, in the PROGRESS *Runs* line, whether T-0052 groups A/B and T-0053 group A are
merged in the tree that produced these numbers.

Regime (state it with every number this produces)
=================================================
profile ``lean``; canvas 256, ``max_denoising_steps`` 48 (engine-global, ``--diffusion-config``),
thinking **on**, sampling = the whitelist defaults only (``LlmOptions()``), ``xargs`` none; vLLM
v0.29.0 container, ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev ``2ea8372``,
``--max-model-len 16384``, ``max_num_seqs 3``, ``TRITON_ATTN``, plugin and chat template per
``services/versions.lock``; prompts built with **frame None, history empty, assists off**, memory
hits empty except ``flight_booking_with_memory``; **2 arms x 7 cases x n = 5 = 70 generations**;
seed n/a — the diffusion path 400s on ``seed``, so these runs are **not reproducible** and that
must be said beside every number. Wall time is recorded as an observation only: n = 5, no bench
harness, and the two arms differ in prefix so the second pays a prefill (**do not** read latency
out of this module). One JSON line per run to ``logs/t0055-placement-<git-sha>.jsonl`` (``logs/``,
never ``artifacts/``) carrying the whole regime.

n = 5 cannot resolve a small effect and this module does not claim otherwise (ambiguity 3). It can
show a large one — 2-of-2 at the desk against ≤ 2-of-20 here — and criterion 14 is what stops a
weak run being read as a pass. A failure is a finding about the **wording** (fix the block, or the
spec) — never about the assertion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import time
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor import prompt_builder as pb_module
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, VllmClient, VllmEndpoint
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.conductor.ui_gate import UiGate
from local_astra.config import Profile
from local_astra.contracts.llm import LlmOptions
from local_astra.contracts.memory import Episode, MemoryHit
from local_astra.contracts.perception import AssistFlags, Source
from tests.fakes.llm import fence_lifecycle

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).with_name("surface_placement_cases.jsonl")
BASELINE_BLOCK_PATH = REPO / "tests" / "fixtures" / "prompts" / "ui_block_t0044.txt"

#: criterion 13's n. Seven cases, two arms.
RUNS_PER_CASE = 5
ARMS = ("new", "baseline")

#: **(a) placement.** Over the 20 positive runs of arm ``new``, at most this many may carry a
#: complete fence inside a ``thought`` commit. The desk saw 2 of 2.
MAX_THOUGHT_FENCE_RUNS = 2

#: **(b) uptake.** Per positive case, at least this many of ``RUNS_PER_CASE`` must end with a
#: gate-accepted surface carrying at least ``MIN_ACCEPTED_COMPONENTS`` components including
#: ``root``: an empty panel and a decorative one are both failures.
POSITIVE_MIN_ACCEPTED = 3
MIN_ACCEPTED_COMPONENTS = 3

#: **(c) no false claim.** Over the 20 positive runs, runs whose spoken answer claims a surface
#: while none was accepted. This is the human-visible harm of the origin and the number that
#: matters most.
MAX_FALSE_CLAIM_RUNS = 1

#: **Criterion 14.** The baseline arm must show the defect, or the run is inconclusive. Strictly
#: more than ``MAX_THOUGHT_FENCE_RUNS`` positive baseline runs must put the fence in the thought.
BASELINE_MIN_THOUGHT_FENCE_RUNS = MAX_THOUGHT_FENCE_RUNS + 1

#: Pre-registered, never tuned after seeing the answers: phrases in which the assistant asserts
#: that a surface exists. Matched case-folded against the spoken answer only. The matched phrase is
#: recorded per run so the reviewer can audit a claim the list called wrong — "here is" is
#: deliberately broad, because the failure it names ("I have created a flight booking interface
#: for you", said to a human looking at an empty panel) is the worst thing this product does.
CLAIM_PHRASES: tuple[str, ...] = (
    "i have created",
    "i've created",
    "i created",
    "i have made",
    "i've made",
    "i made you",
    "i have built",
    "i've built",
    "i have put",
    "i have prepared",
    "here is",
    "here's",
    "on the screen",
    "on your screen",
    "take a look",
    "shown below",
    "you can see",
)

#: The whole generation, not a latency budget: a diffusion turn takes seconds here, and a hung
#: request must fail the eval rather than hang the box.
REQUEST_TIMEOUT_S = 180.0


def _cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


CASES = _cases()
POSITIVES = [c for c in CASES if c["expect_surface"]]
NEGATIVES = [c for c in CASES if not c["expect_surface"]]


@dataclass
class RunRecord:
    """One generation. Everything criteria 13-14 ask the harness to record, plus the regime."""

    arm: str
    case: str
    request: str
    expect_surface: bool
    run: int
    #: (a): a *complete* fence — open and close — anywhere in this turn's thought commits
    thought_fence_complete: bool = False
    #: an opening fence in the thought that never closed there; recorded, not asserted
    thought_fence_open: bool = False
    thought_chars: int = 0
    thought_canvases: int = 0
    ui_commits: int = 0
    canvases_spanned: int = 0
    surface_accepted: bool = False
    components: list[str] = field(default_factory=list)
    has_root: bool = False
    ui_repairs: int = 0
    ui_resamples: int = 0
    ui_valid: bool | None = None
    gate_problems: list[str] = field(default_factory=list)
    answer_text: str = ""
    #: (c): the claim phrase found in the spoken answer, or ``None``
    claim_phrase: str | None = None
    fence_in_answer: bool = False
    #: T-0052's fence lifecycle, so a T-0055 success that exercised T-0052 is legible as such
    fence_close: str = "none"
    wall_s: float = 0.0
    regime: dict[str, Any] = field(default_factory=dict)

    @property
    def gated_surface(self) -> bool:
        """(b)'s definition of success: accepted, ``root`` present, not decorative."""
        return (
            self.surface_accepted
            and self.has_root
            and len(self.components) >= MIN_ACCEPTED_COMPONENTS
        )

    @property
    def false_claim(self) -> bool:
        """(c): the answer says a surface exists and no surface was accepted."""
        return self.claim_phrase is not None and not self.surface_accepted


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
        "chat_template_revision": lock["gpu"]["chat_template_revision"],
        "attention_backend": lock["gpu"]["vllm_attention_backend"],
        "sampling": "whitelist defaults only (LlmOptions())",
        "xargs": None,
        "prompt_shape": "frame=None, history=[], assists off; memory hits per case",
        "arms": list(ARMS),
        "n_per_case_per_arm": RUNS_PER_CASE,
        "seed": "n/a (the diffusion path 400s on seed) - runs are NOT reproducible",
        "git_commit": _git_commit(),
        "harness": "PromptBuilder + VllmClient + UiGate (no Session, no STT/TTS)",
        "baseline_block_sha256": (
            "2ec55d8df45d195f6508cedd8b6c6d9befaab402a6085bf710fb5dee91818e67"
        ),
    }


@contextlib.contextmanager
def _arm(name: str) -> Iterator[None]:
    """Install an arm's instruction block for the duration of a prompt build.

    ``prompt_builder`` does ``from …ui_instructions import UI_INSTRUCTIONS``, so the name it reads
    at call time is ``prompt_builder.UI_INSTRUCTIONS``: that is the attribute to patch, and
    patching ``ui_instructions.UI_INSTRUCTIONS`` would do nothing. Test-only.
    """
    original = pb_module.UI_INSTRUCTIONS
    try:
        if name == "baseline":
            pb_module.UI_INSTRUCTIONS = BASELINE_BLOCK_PATH.read_text(encoding="utf-8")
        elif name != "new":
            raise AssertionError(f"unknown arm {name!r}")
        yield
    finally:
        pb_module.UI_INSTRUCTIONS = original


def _memory(case: dict[str, Any]) -> tuple[MemoryHit, ...]:
    """The origin turn's memory hits, for the one case that carries them (criterion 14)."""
    return tuple(
        MemoryHit(
            episode=Episode(
                ts_ms=1_000 + i,
                source=Source.SCREEN,
                thumbnail_path=f"/tmp/t0055-{i}.jpg",
                phash=f"c{i}" * 8,
                window_title="Exercises and Problems in Calculus",
                caption=caption,
            ),
            score=0.9 - 0.1 * i,
            matched=("calculus",),
        )
        for i, caption in enumerate(case["memory_captions"])
    )


def _claim(answer: str) -> str | None:
    low = answer.lower()
    return next((phrase for phrase in CLAIM_PHRASES if phrase in low), None)


async def _one_run(arm: str, case: dict[str, Any], run: int, regime: dict[str, Any]) -> RunRecord:
    endpoint = VllmEndpoint(request_timeout_s=REQUEST_TIMEOUT_S)
    with _arm(arm):
        builder = PromptBuilder(DEFAULT_SYSTEM_PROMPT)
        prefix = builder.prefix([], None, AssistFlags())
        prompt = builder.full(prefix, case["request"], None, _memory(case))

    record = RunRecord(
        arm=arm,
        case=case["id"],
        request=case["request"],
        expect_surface=bool(case["expect_surface"]),
        run=run,
        regime=regime,
    )
    gate = UiGate()
    client = VllmClient(endpoint)
    request_id = f"t0055-{arm}-{case['id']}-{run}"
    ui_canvases: set[int] = set()
    answer: list[str] = []
    thought: list[str] = []
    started = time.monotonic()
    try:
        async for commit in client.generate(prompt, request_id, LlmOptions()):
            if commit.segment == "thought":
                record.thought_canvases += 1
                thought.append(commit.text)
            elif commit.segment == "answer" and not commit.thought_open:
                answer.append(commit.text)
            elif commit.segment == "ui":
                record.ui_commits += 1
                ui_canvases.add(commit.canvas_idx)
                record.fence_close = str(fence_lifecycle(commit) or "none")
                result = gate.gate(
                    commit.text, turn_id=request_id, canvas_idx=commit.canvas_idx, final=True
                )
                record.ui_repairs += result.repaired
                record.ui_resamples += len(result.resample)
                record.ui_valid = result.report.ok
                record.gate_problems.extend(result.report.problems)
                if result.surface is not None:
                    components = [
                        c
                        for m in result.surface.messages
                        if isinstance(m, dict) and isinstance(m.get("updateComponents"), dict)
                        for c in m["updateComponents"]["components"]
                    ]
                    record.surface_accepted = True
                    record.components = [str(c.get("component")) for c in components]
                    record.has_root = any(c.get("id") == "root" for c in components)
    finally:
        await client.aclose()

    record.wall_s = round(time.monotonic() - started, 3)
    record.answer_text = "".join(answer)
    record.canvases_spanned = len(ui_canvases)
    record.claim_phrase = _claim(record.answer_text)
    record.fence_in_answer = UI_OPEN in record.answer_text or UI_CLOSE in record.answer_text

    # (a): a *complete* fence in the thought. The thought is joined across canvases on purpose —
    # at the desk the thought-fence itself straddled canvases 1 to 2, and a fence that opened in
    # one thought canvas and closed in the next is exactly the defect, not a near miss.
    whole_thought = "".join(thought)
    record.thought_chars = len(whole_thought)
    opened = whole_thought.find(UI_OPEN)
    record.thought_fence_open = opened >= 0
    record.thought_fence_complete = opened >= 0 and UI_CLOSE in whole_thought[opened:]
    return record


async def _all_runs() -> list[RunRecord]:
    regime = _regime()
    records: list[RunRecord] = []
    for case in CASES:
        for run in range(RUNS_PER_CASE):
            for arm in ARMS:  # interleaved per run: engine drift cannot land on one arm
                records.append(await _one_run(arm, case, run, regime))
    return records


@pytest.fixture(scope="session")
def placement_runs() -> list[RunRecord]:
    """70 generations, once per session, written to ``logs/`` as they are asserted on."""
    endpoint = VllmEndpoint()
    try:
        httpx.get(endpoint.base_url + "/models", timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:  # the box is not serving: skip, never a silent pass
        pytest.skip(f"vLLM is not answering at {endpoint.base_url} ({exc}); start it with /launch")

    records = asyncio.run(_all_runs())
    out = REPO / "logs" / f"t0055-placement-{_git_commit()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))
    print(f"\n[T-0055 placement eval] {len(records)} runs -> {out}")
    for r in records:
        print(
            f"  {r.arm:>8} {r.case:>28} #{r.run}: thought_fence={r.thought_fence_complete} "
            f"ui_commits={r.ui_commits} accepted={r.surface_accepted} "
            f"components={len(r.components)} root={r.has_root} canvases={r.canvases_spanned} "
            f"claim={r.claim_phrase!r} wall={r.wall_s}s"
        )
    return records


def _runs(records: list[RunRecord], arm: str, case_id: str | None = None) -> list[RunRecord]:
    return [r for r in records if r.arm == arm and (case_id is None or r.case == case_id)]


def _positive(records: list[RunRecord], arm: str) -> list[RunRecord]:
    return [r for r in records if r.arm == arm and r.expect_surface]


def _detail(runs: list[RunRecord]) -> str:
    return "\n".join(
        f"  {r.arm}/{r.case} #{r.run}: thought_fence={r.thought_fence_complete} "
        f"thought_chars={r.thought_chars} ui_commits={r.ui_commits} "
        f"accepted={r.surface_accepted} root={r.has_root} components={r.components} "
        f"canvases={r.canvases_spanned} fence_close={r.fence_close} "
        f"repairs={r.ui_repairs} resamples={r.ui_resamples} problems={r.gate_problems[:2]} "
        f"answer={r.answer_text[:140]!r}"
        for r in runs
    )


@pytest.mark.pending(task="T-0055")
def test_the_surface_is_written_in_the_reply_not_in_the_thought(placement_runs):
    """Criterion 13 (a): of the 20 positive runs of arm ``new``, at most 2 fence in the thought.

    This is the defect itself, counted. At the desk it was 2 of 2; the threshold says the rewritten
    block must make it rare, not merely rarer. A failure here is a finding about the **wording** —
    the block, or the spec — never about this number.
    """
    runs = _positive(placement_runs, "new")
    assert len(runs) == len(POSITIVES) * RUNS_PER_CASE, (
        f"expected 20 positive runs, got {len(runs)}"
    )

    offenders = [r for r in runs if r.thought_fence_complete]
    assert len(offenders) <= MAX_THOUGHT_FENCE_RUNS, (
        f"criterion 13 (a): {len(offenders)}/{len(runs)} positive runs built a complete A2UI fence "
        f"inside the reasoning channel (threshold {MAX_THOUGHT_FENCE_RUNS}). That channel is "
        "discarded by design and the chat template's strip_thinking macro deletes it from the "
        "conversation, so those surfaces reached nobody. Fix the block's wording (and record the "
        "alternative arm of ambiguity 1); do not raise this number without a docs/tasks entry.\n"
        + _detail(offenders)
    )


@pytest.mark.pending(task="T-0055")
@pytest.mark.parametrize("case", POSITIVES, ids=[c["id"] for c in POSITIVES])
def test_a_positive_case_produces_a_gate_accepted_surface(placement_runs, case):
    """Criterion 13 (b): 3 of 5 runs give a gated surface with at least 3 components incl ``root``.

    An empty panel and a decorative one are both failures — the human's complaint was not "the
    surface was small", it was "there was nothing there while the assistant said there was".
    """
    runs = _runs(placement_runs, "new", case["id"])
    assert len(runs) == RUNS_PER_CASE, f"expected {RUNS_PER_CASE} runs, got {len(runs)}"

    good = [r for r in runs if r.gated_surface]
    assert len(good) >= POSITIVE_MIN_ACCEPTED, (
        f"criterion 13 (b): {case['id']!r} produced a usable surface in {len(good)}/"
        f"{RUNS_PER_CASE} runs (needs {POSITIVE_MIN_ACCEPTED}, at least "
        f"{MIN_ACCEPTED_COMPONENTS} components including root). The prompt block is what changes, "
        "not this threshold.\n" + _detail(runs)
    )


@pytest.mark.pending(task="T-0055")
def test_no_run_claims_a_surface_it_did_not_show(placement_runs):
    """Criterion 13 (c): at most 1 of the 20 positive runs says it made something and did not.

    **The number that matters most.** The origin is not "a fence went to the wrong channel"; it is
    a person hearing "I have created a flight booking interface for you" while looking at an empty
    panel. An assistant that sounds correct while being wrong about the present is the worst
    failure shape this product has, and it is the one the human met twice.
    """
    runs = _positive(placement_runs, "new")
    liars = [r for r in runs if r.false_claim]
    assert len(liars) <= MAX_FALSE_CLAIM_RUNS, (
        f"criterion 13 (c): {len(liars)}/{len(runs)} positive runs claimed a surface that was "
        f"never accepted (threshold {MAX_FALSE_CLAIM_RUNS}).\n"
        + "\n".join(
            f"  {r.case} #{r.run}: claim={r.claim_phrase!r} thought_fence="
            f"{r.thought_fence_complete} ui_commits={r.ui_commits} answer={r.answer_text[:200]!r}"
            for r in liars
        )
        + "\nIf a claim phrase matched an answer that was not claiming anything, say so to the "
        "reviewer with the run — CLAIM_PHRASES is pre-registered and is not tuned after the run."
    )


@pytest.mark.pending(task="T-0055")
@pytest.mark.parametrize("case", NEGATIVES, ids=[c["id"] for c in NEGATIVES])
def test_a_plain_question_produces_no_surface(placement_runs, case):
    """Criterion 13 (d): 0 accepted surfaces over the 15 negative runs of arm ``new``.

    A block that makes the model draw a card for "what is the capital of France" has traded one
    failure for another: every chatty turn then spends a canvas on a surface nobody asked for and
    the shown channel stops meaning anything. ``cloud_joke`` is the sharpest of the three.

    A fence in the *thought* on a negative case is recorded but does not fail this row: the model
    considering a surface and not writing one is the block working.
    """
    runs = _runs(placement_runs, "new", case["id"])
    assert len(runs) == RUNS_PER_CASE

    offenders = [r for r in runs if r.surface_accepted or r.ui_commits or r.fence_in_answer]
    assert not offenders, (
        f"criterion 13 (d): {case['id']!r} produced a surface (or a fence in the spoken text) in "
        f"{len(offenders)}/{RUNS_PER_CASE} runs; the block over-triggers.\n" + _detail(offenders)
    )


@pytest.mark.pending(task="T-0055")
def test_a_surface_built_in_the_thought_is_reported_not_forgiven(placement_runs):
    """Adversarial: a fence in the thought and none in the reply is a **failure of this task**.

    The tempting reading — "the model tried, the answer was good, close enough" — is the reading
    that produced two empty panels. A run whose thought carries a complete fence and whose reply
    carries none produced nothing the user can see, however fluent it sounded; and a run where the
    gate happened to accept something from a *later* canvas does not redeem the one that was
    thrown away, because the block is being judged on where it told the model to write.

    This row is deliberately per-run and not a proportion: it names the runs, so the reviewer sees
    each one rather than a rate that has been rounded into acceptability.
    """
    wasted = [
        r
        for r in _positive(placement_runs, "new")
        if r.thought_fence_complete and not r.gated_surface
    ]
    assert not wasted, (
        f"{len(wasted)} run(s) built a complete surface inside the thinking channel and showed "
        "the user nothing. The surface was erased by the model's own chat template "
        "(strip_thinking deletes the thought span from every re-rendered assistant message); no "
        "ui.gate record exists for these runs, exactly as in the two desk sessions.\n"
        + _detail(wasted)
        + "\nThe fix is the block's wording. It is NOT reading the fence out of the thought: the "
        "human ruled on 2026-09-16 that we do not extract from that channel "
        "(tests/unit/test_prompt_ui_block.py::test_no_module_outside_the_client_reads_the_fence)."
    )


@pytest.mark.pending(task="T-0055")
def test_the_baseline_arm_reproduced_the_defect(placement_runs):
    """Criterion 14: the run says it is inconclusive rather than passing on a regime that is blind.

    If the block that failed twice at the desk does **not** put the fence in the thought here, this
    harness cannot see the defect, and the ``new`` arm's green is worth nothing. The finding in
    that case is that reproduction needs the desk's conditions — memory hits, a live session, a
    frame — not that the wording works. Case ``flight_booking_with_memory`` carries the three
    calculus captions of the origin turn for exactly this reason: it is the only respect in which
    the 21:52 prompt differed from this harness's.

    The pair of counters is reported either way, so the delta between the arms is a number and not
    an impression.
    """
    baseline = _positive(placement_runs, "baseline")
    new = _positive(placement_runs, "new")
    assert len(baseline) == len(new) == len(POSITIVES) * RUNS_PER_CASE

    baseline_fences = [r for r in baseline if r.thought_fence_complete]
    new_fences = [r for r in new if r.thought_fence_complete]
    by_case = {
        c["id"]: (
            sum(1 for r in _runs(placement_runs, "baseline", c["id"]) if r.thought_fence_complete),
            sum(1 for r in _runs(placement_runs, "new", c["id"]) if r.thought_fence_complete),
        )
        for c in POSITIVES
    }
    print(
        "\n[T-0055 criterion 14] thought-fence runs (baseline, new) per positive case: "
        + json.dumps(by_case)
        + f"\n  totals: baseline {len(baseline_fences)}/{len(baseline)}, "
        f"new {len(new_fences)}/{len(new)}"
    )

    assert len(baseline_fences) >= BASELINE_MIN_THOUGHT_FENCE_RUNS, (
        "baseline did not reproduce: the block that failed twice at the desk put a complete fence "
        f"in the thought in only {len(baseline_fences)}/{len(baseline)} positive runs "
        f"(needs at least {BASELINE_MIN_THOUGHT_FENCE_RUNS} for this run to be able to see the "
        "defect at all). This is INCONCLUSIVE, not a pass: the new arm's numbers cannot be "
        "credited, and the finding is that the reproduction needs the desk's conditions — a live "
        f"session, a frame, the memory hits. Per-case (baseline, new): {json.dumps(by_case)}"
    )


def test_every_run_is_recorded_with_its_regime(placement_runs):
    """The harness's own contract (not pending): a number without its regime cannot be compared.

    CLAUDE.md: state the whole regime beside every number. This row fails if the log line loses the
    arm, the model, the canvas, the prompt shape or the commit — which is how an eval result
    becomes unusable three weeks later — and if the baseline arm silently ran against the current
    block (the fixture is the *point* of the A/B).
    """
    out = REPO / "logs" / f"t0055-placement-{_git_commit()}.jsonl"
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == len(CASES) * RUNS_PER_CASE * len(ARMS) == len(placement_runs)
    for line in lines:
        assert line["arm"] in ARMS
        for key in ("profile", "canvas_length", "model", "model_revision", "git_commit", "seed"):
            assert line["regime"][key] is not None, f"regime is missing {key}"

    assert BASELINE_BLOCK_PATH.read_text(encoding="utf-8") != pb_module.UI_INSTRUCTIONS, (
        "the baseline fixture is byte-identical to the block under test: either T-0055 has not "
        "changed the block, or the fixture was regenerated. Either way the two arms are the same "
        "arm and criterion 14 cannot mean anything (tests/fixtures/prompts/MANIFEST.md)"
    )
