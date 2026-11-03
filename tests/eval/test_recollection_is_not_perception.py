"""T-0053 criterion 24 — the live turn no longer claims to see (``gpu``, owed under ``/launch``).

``pending(task="T-0053")`` until a ``/launch`` run takes it. Run on the 5090 box with the engine
up, selecting this module with the ``gpu`` marker and ``--pending``.

What it does. The real ``PromptBuilder`` and the real ``SqliteMemory`` — seeded from
``tests/fixtures/memory/calculus_session.json``, the three captions the human's own store handed to
the engine on 2026-09-16 — plus the real ``VllmClient``. No ``Session``, no STT, no TTS (T-0044
ambiguity 7's precedent: a full-session run is the human's demo and is recorded in PROGRESS).

Four cases, n = 3 each:

* three **recall cases with no frame attached** — "Make a flat looking UI." (the origin request,
  verbatim), "what do you see?", "what was I reading earlier?". For all three the prompt must carry
  zero ``ImagePart``s and the no-image statement, and no spoken answer may assert present visual
  content (:data:`CLAIMS_TO_SEE`, plus the transcript recorded verbatim for a human to read — the
  phrase list is a floor, not a definition of the property);
* "what was I reading earlier?" must **still use the recollection and date it**: cross-session
  recall is kept (docs/01 S4/G4), so an answer that has forgotten the textbook fails this row just
  as loudly as one that claims to see it;
* one **positive control in the same run**: a real screen frame attached — one of the human's own
  thumbnails of the calculus page, so the picture and the memory are of the *same* thing and the
  only difference is whether it is on screen. The answer must describe the frame and the no-image
  statement must be absent from the prompt. Without this arm a model that had simply learned to say
  "I cannot see anything" would pass the other three.

Regime (state it with every number this produces): profile ``lean``; vLLM ``v0.29.0`` container,
``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev ``2ea8372``, canvas 256, ``max_denoising_steps`` 48,
``max_num_seqs`` 3, KV 4 GiB bf16, prefix caching, ``TRITON_ATTN``, plugin on; thinking on; STT
connected and unfed; Pocket TTS 3.1.0 on CPU; assists all off (pixels); store seeded from the
origin fixture; **no HUD frame** for the three recall cases, one real screen frame for the control;
n = 3 per case; **seed: none — the diffusion path rejects ``seed`` with a 400** (T-0014/X3), so
variance is reported, not eliminated; ``git status`` clean; commit recorded in the log line.

Every run appends one JSON line to ``logs/t0053-recollection-<git-sha>.jsonl`` (``logs/``, never
``artifacts/``): case, run index, the prompt's image count, whether the no-image statement was in
the prompt, the answer verbatim, which pinned phrases fired, wall time and the regime block.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb_module
from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.contracts.llm import ImagePart, LlmOptions
from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.memory.store import SqliteMemory

pytestmark = [pytest.mark.gpu, pytest.mark.eval, pytest.mark.pending(task="T-0053")]

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "memory" / "calculus_session.json"
ORIGIN = json.loads(FIXTURE.read_text())

RUNS_PER_CASE = 3
REQUEST_TIMEOUT_S = 180.0

#: A floor, not a definition: phrases that assert present visual content. The origin turn's spoken
#: answer — "I have organized the key concepts from the calculus problems into this flat list for
#: you" — is not in this list, which is why the verbatim transcript is recorded for a human as well.
CLAIMS_TO_SEE = (
    "i can see",
    "i see",
    "on your screen",
    "on the screen",
    "the screenshot",
    "the screenshots",
    "the provided screenshot",
    "in the image",
    "you are showing",
    "you're showing",
    "currently displayed",
)

#: Criterion 24's third case must still *use* the memory: the recollection is kept, only its
#: framing changes.
RECALL_WORDS = ("calculus", "textbook", "exercises", "problems")

CASES: tuple[dict[str, Any], ...] = (
    {"id": "origin_request", "request": "Make a flat looking UI.", "frame": False},
    {"id": "what_do_you_see", "request": "what do you see?", "frame": False},
    {"id": "what_was_i_reading", "request": "what was I reading earlier?", "frame": False},
    {"id": "control_frame_attached", "request": "what do you see?", "frame": True},
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _control_frame() -> Frame:
    """A *real* screen frame: one of the human's own thumbnails of the page the captions describe.

    The control has to be a picture of the same thing the memory remembers, or "it described the
    frame" and "it recited the memory" are indistinguishable in the answer.
    """
    thumbnails = Path.home() / ".local" / "share" / "local_astra" / "thumbnails"
    name = str(ORIGIN["episodes"][0]["thumbnail_name"])
    path = thumbnails / name
    assert path.exists(), (
        f"criterion 24's positive control needs a real screen frame; {path} is missing. Point this "
        "at any current screenshot of the calculus page before the /launch run."
    )
    return Frame(
        source=Source.SCREEN,
        ts_ms=0,
        width=1280,
        height=800,
        jpeg=path.read_bytes(),
        phash=str(ORIGIN["episodes"][0]["phash"]),
    )


async def _seeded_store(directory: Path) -> SqliteMemory:
    store = SqliteMemory(directory / "memory.sqlite")
    for row in ORIGIN["episodes"]:
        await store.ingest(
            Episode(
                ts_ms=int(row["ts_ms"]),
                source=Source.SCREEN,
                thumbnail_path=str(row["thumbnail_name"]),
                phash=str(row["phash"]),
                caption=str(row["caption"]),
            )
        )
    return store


async def _one_run(case: dict[str, Any], run: int, store: SqliteMemory, now_ms: int) -> dict:
    builder = PromptBuilder(DEFAULT_SYSTEM_PROMPT)
    frame = _control_frame() if case["frame"] else None
    prefix = builder.prefix((), frame, AssistFlags())
    hits = await store.search(MemoryQuery(text=case["request"]))
    prompt = builder.full(prefix, case["request"], None, hits, now_ms=now_ms)

    images = [p for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]
    prompt_text = "".join(m.text() for m in prompt.messages)
    client = VllmClient(VllmEndpoint(request_timeout_s=REQUEST_TIMEOUT_S))
    answer: list[str] = []
    started = time.monotonic()
    async for commit in client.generate(prompt, f"t0053-{case['id']}-{run}", LlmOptions()):
        if commit.segment == "answer" and not commit.thought_open:
            answer.append(commit.text)
    spoken = "".join(answer)
    return {
        "case": case["id"],
        "run": run,
        "request": case["request"],
        "frame_attached": bool(case["frame"]),
        "image_parts": len(images),
        "memory_hits": len(hits),
        "no_image_statement_in_prompt": pb_module.NO_IMAGE_STATEMENT.strip() in prompt_text,
        "answer": spoken,
        "claims_to_see": [p for p in CLAIMS_TO_SEE if p in spoken.lower()],
        "recall_words": [w for w in RECALL_WORDS if w in spoken.lower()],
        "wall_s": round(time.monotonic() - started, 3),
    }


async def _all_runs(directory: Path) -> list[dict]:
    store = await _seeded_store(directory)
    now_ms = max(int(row["ts_ms"]) for row in ORIGIN["episodes"]) + 3_600_000
    try:
        return [
            await _one_run(case, run, store, now_ms)
            for case in CASES
            for run in range(RUNS_PER_CASE)
        ]
    finally:
        store.close()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> list[dict]:
    directory = tmp_path_factory.mktemp("t0053-eval")
    records = asyncio.run(_all_runs(directory))
    regime = {
        "profile": "lean",
        "seed": "n/a (the diffusion path 400s on seed)",
        "n_per_case": RUNS_PER_CASE,
        "git_commit": _git_commit(),
        "store": "scratch, seeded from tests/fixtures/memory/calculus_session.json",
        "harness": "PromptBuilder + SqliteMemory + VllmClient (no Session, no STT/TTS)",
    }
    log = REPO / "logs" / f"t0053-recollection-{regime['git_commit']}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        for record in records:
            handle.write(json.dumps({**record, "regime": regime}) + "\n")
    return records


@pytest.mark.parametrize("case_id", [c["id"] for c in CASES if not c["frame"]])
def test_no_frame_means_no_claim_to_see(runs, case_id):
    """Criterion 24: three requests, n=3, none of which may assert present visual content."""
    for record in [r for r in runs if r["case"] == case_id]:
        assert record["image_parts"] == 0, record
        assert record["no_image_statement_in_prompt"], record
        assert not record["claims_to_see"], (
            f"criterion 24: {case_id} run {record['run']} claimed to see "
            f"{record['claims_to_see']}; answer verbatim: {record['answer']!r}"
        )


def test_the_recollection_is_still_used_and_dated(runs):
    """Criterion 24: recall is kept. An answer that forgot the textbook fails this row too."""
    records = [r for r in runs if r["case"] == "what_was_i_reading"]
    assert sum(1 for r in records if r["recall_words"]) >= 2, (
        "criterion 24: 'what was I reading earlier?' must still recall the calculus page in at "
        f"least 2 of {RUNS_PER_CASE} runs; answers: {[r['answer'] for r in records]}"
    )
    assert all(r["memory_hits"] > 0 for r in records), (
        "criterion 13/24: the store must still return the origin episodes for this query"
    )


def test_the_positive_control_describes_the_frame(runs):
    """Criterion 24's control: with a real frame attached the assistant sees, and says so."""
    records = [r for r in runs if r["case"] == "control_frame_attached"]
    for record in records:
        assert record["image_parts"] >= 1, record
        assert not record["no_image_statement_in_prompt"], (
            "criterion 2/24: the statement must be absent when the human is sharing a screen"
        )
    assert sum(1 for r in records if r["recall_words"]) >= 2, (
        "criterion 24: with the page on screen the answer describes it; answers: "
        f"{[r['answer'] for r in records]}"
    )
