"""T-0223 B -- the locked, report-only GPU check: does Gearvis find the planted mistake in a handwritten maths or
physics attempt shown as a shared screen? 10 pages x n = 3 = 30 typed turns through a real ``Session`` (tools off,
search off, 1120 visual tokens, the tutor instructions set through the Settings path). ``/launch`` ONLY.

    uv run python scripts/gpu/t0223_handwritten_check.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing.**

Reuse: T-0208 C's harness (``scripts/gpu/t0208_handoff_run.py``) is imported read-only by path as ``H`` and never
edited; through it T-0205's (``D``) and T-0111g's / T-0131's / T-0132's helpers (``G``, ``T``, ``P``): ``LoopMeter``,
the session fakes, ``_PlaybackOnly``, ``read_metrics``, ``_healthy``, the VRAM sampler, ``shown_text``,
``clip_kinds``.

The seam (planner, read in code): outside live there is no vision selector; the turn reads
``perception.get_frame(SCREEN)``. As in T-0208 C, one ``BrowserFrameSource`` is both the session's perception and
the frame sink of a socketless ``HudServer``; per turn: hello -> ``astra.settings.set {rev: 0, changes:
{user_instructions: TUTOR_INSTRUCTIONS}}`` -> wait for the ``astra.settings.state`` echo at ``rev == 1`` with nothing
refused (``SETTINGS_TIMEOUT_S``; a timeout errors the turn) -> one ``astra.frame`` (screen; the page PNG resized to
the HUD screen bucket, JPEG at ``policy.JPEG_QUALITY``, a 16-hex phash) -> ``UserText(QUESTION)``.

B1 refusals (exit 2), all before any socket: ``_refusal`` (P-S, CPU, a rig ``Session`` over a scripted fake LLM,
is the last of them). P0 (engine: one text and one image generate at 1120 answer 200) failing is void, exit 3.
B3 per-turn evidence (``evidence``) from the session log and the turn line; any violation on a turn that ran makes
the run void. The meter's rebuilt request bodies (``max_soft_tokens == 1120``, no ``tools``, the one image) are
checked too and reported as ``body_violations`` (also void; the session log does not carry the HTTP body, so the
eval cannot re-derive them). B4 judge (``judge``): judged text = spoken + shown (R-1, R-2). B5 report
(``build_report``/``write_report``, from the run dir only). B6 labels: void (exit 3), incomplete (exit 1), measured
(exit 0).

Output (``--out-dir``, default ``logs/t0223-handwritten-<sha7>/``): ``order.json`` (first), ``manifest.json``,
``probes.json`` (``P-S``, ``P0``), ``trials.jsonl``, ``turns.jsonl``, ``sessions/*.jsonl``, ``report.json``,
``report.md``.

Regime (state it beside every number): profile lean, canvas 256, 48 denoising steps, thinking on; vLLM 0.29.0,
NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, vllm.sh at 806da12,
one boot; tools off, search off, 1120 visual tokens, HUD screen bucket (portrait page ~1109x1568), JPEG 85, one
frame per turn; typed turns, FakeStt/FakeTts, STT down; 10 x 3 = 30 turns, order seed 0, fresh Session per turn,
engine unseeded, concurrency 1, 90 s per turn, 14 min run budget (R-4).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

#: B2: the run budget is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0208 C's harness, read-only (imported, never edited)
T0208_PATH: Path = REPO / "scripts" / "gpu" / "t0208_handoff_run.py"
T0208_MODULE_NAME = "_t0208_handoff_run_for_t0223"


def _load_t0208() -> ModuleType:
    """T-0208 C's harness, imported read-only by path under ``T0208_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0208_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0208_MODULE_NAME, T0208_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0208_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0208_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0208_MODULE_NAME, None)
        raise
    return module


H: ModuleType = _load_t0208()
D: ModuleType = H.D
F: ModuleType = H.F
P: ModuleType = H.P
T: ModuleType = H.T
G: ModuleType = H.G

LoopMeter = G.LoopMeter
shown_text = D.shown_text
clip_kinds = D.clip_kinds
#: the tree is read through this name; ``_refusal`` and ``main`` look it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"

# --- criterion 5's texts, verbatim -----------------------------------------------------------------------------
TUTOR_INSTRUCTIONS = (
    "You are a patient tutor. Point to the line with the mistake and explain why it is wrong, then give the "
    "corrected step."
)
QUESTION = "Here's my attempt. Can you check it and tell me where I went wrong?"

# --- B2: the run shape -----------------------------------------------------------------------------------------
PAGE_IDS: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5")
REPEATS = 3
TOTAL_TURNS = len(PAGE_IDS) * REPEATS
ORDER_SEED = 0
TURN_TIMEOUT_S = 90
#: inside the locked script's ``timeout -k 30s 15m`` so the report is still written
RUN_BUDGET_S = 14 * 60  # R-4 (amends B2): 30 turns at ~20 s fit
#: B2: no turn starts unless ``TURN_TIMEOUT_S`` plus this still fit inside ``RUN_BUDGET_S``
REPORT_MARGIN_S = 45
#: B2: how long a turn waits for the settings echo before it is errored
SETTINGS_TIMEOUT_S = 10.0
#: the budget in force (criterion 5, B1, B3 c)
VISUAL_TOKENS = 1120
#: the session's settings revision after the one set (a fresh Session starts at 0)
SETTINGS_REV = 1
#: the socketless HUD connection's outbox (nothing a turn publishes overflows it)
HUD_OUTBOX = 100_000
#: B1: the merge subjects (exact prefixes) read from ``git log --merges``
MERGE_A = "Merge T-0223-a:"
MERGE_A2A = "Merge T-0213-a2a:"
#: the probe names in ``probes.json``
PROBE_NAMES = ("P-S", "P0")

# --- B1: group A's assets, read at call time; their pins are literals ------------------------------------------
ATTEMPTS_PATH: Path = REPO / "demos" / "handwritten" / "attempts.json"
PAGES_DIR: Path = REPO / "demos" / "handwritten" / "out"
#: the marker a pin carries while it is unfilled; the harness refuses while any pin has it
PIN_PLACEHOLDER = "PLACEHOLDER-fill-after-Merge-T-0223-a"
#: sha256 of ``demos/handwritten/attempts.json`` at ``Merge T-0223-a:`` (91fa77d0), computed from main's file; a
#: changed file is refused, never run
ATTEMPTS_SHA256 = "37ba6b501b76e3abbf7a5e9fdada64fefbe07725ef6ef249ee95c5e223790e48"
#: sha256 of each ``demos/handwritten/out/<id>.png`` at ``Merge T-0223-a:`` (91fa77d0), computed likewise
PAGE_SHA256: dict[str, str] = {
    "M1": "74d7241c07ee75f6c67c1c2c01dedb0e6333a353845d10112978e3972141db7e",
    "M2": "a870b52546acad91a45f98a02a606d77d64d6f6624823032a677bdf878fc0b21",
    "M3": "6635443790ba1e4431e451eb483ed48488825150db65ab66ce3a1f8bfc4ed1d8",
    "M4": "170b9119269b4350d0e33a92276b8d1618fb708730ee420812184ca009ec3f76",
    "M5": "61628b606372e08da1f8985a01908c29421f1307309c534fb667c5e856feb816",
    "P1": "5a65cbb69d68d620ee9250b95759a75843e9d8962ea8da48b64b91dde62404bf",
    "P2": "b33866c25e1155ff29872ae2f0a13bc5361313ea5775bb62569e8c7475bb1b8b",
    "P3": "c85f1a5fd12a8fe3b0a1d69313c8096fd6315a0618f30c03e04fde89fb69a019",
    "P4": "1f33d03cb289a9cc3efc61b27e5d3d367044c22c6a168bc70dd4bfb22dee3d8d",
    "P5": "320f24609def69b907ea1d5019f4ef27e0f9a34c51f438fe8cd7168536178179",
}

#: stated in the report
FRAME_NOTE = (
    "the frame is the PNG resized by Pillow (bilinear) to policy.target_size at the HUD screen bucket and encoded at "
    "JPEG quality 85; the HUD page resizes on a browser canvas, so the pixels differ slightly from a real share."
)
LATENCY_NOTE = (
    "time to first spoken word = UserText stamp -> tts.first_audio span start, both on the session clock; with "
    "FakeTts this is the hand-off to TTS, not audio out of a speaker; p50 is the low median."
)
JUDGE_NOTE = (
    "located = every judge keyword entry of the page is satisfied (R-2: any of its |-separated alternatives is a "
    "casefolded substring) by the judged text (every spoken text of the turn plus the shown card text, joined with "
    "spaces; R-1) and no false-blame phrase is; thoughts and clips are never judged; not-run, errored, timed-out "
    "and empty turns are not located; n = 3 per page."
)
HAND_AUDIT_NOTE = (
    "hand audit: a column for a human reading answers that spell symbols out (e.g. 'cosine' for cos); it never "
    "feeds k/3."
)
SCOPE_NOTE = "report-only: ten designed pages, one planted mistake each, n = 3 per page; no bar, no adoption rule."
PS_NOTE = (
    "P-S runs before any socket, so a failing P-S is a refusal (exit 2) and no run dir exists; a failing P0 is void "
    "(exit 3). P-S's rig Session keeps its data in a temporary directory removed after the probe."
)

#: the last P-S record ``_refusal`` produced (``probes.json`` carries it)
LAST_PS: dict[str, Any] = {}


def _say(msg: str) -> None:
    print(f"t0223_handwritten_check: {msg}", file=sys.stderr, flush=True)


def instructions_block() -> str:
    """B2 P-S / B3 (b): the tail every system message must end with."""
    from local_astra.conductor.prompt_builder import USER_INSTRUCTIONS_HEADER

    return USER_INSTRUCTIONS_HEADER + "\n" + TUTOR_INSTRUCTIONS


def order(seed: int) -> list[tuple[str, int]]:
    """B2: every ``(page_id, repeat)`` of ``PAGE_IDS`` x ``range(REPEATS)``, page-major, shuffled once by
    ``random.Random(seed)``."""
    turns = [(p, r) for p in PAGE_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


def budget_stop(elapsed_s: float) -> bool:
    """B2: ``True`` when no further turn may start ``elapsed_s`` seconds after process start: a turn starts only
    while ``TURN_TIMEOUT_S`` + ``REPORT_MARGIN_S`` still fit inside ``RUN_BUDGET_S`` (705 s is the last start at 14 min)."""
    return elapsed_s + TURN_TIMEOUT_S + REPORT_MARGIN_S > RUN_BUDGET_S


# =====================================================================================================
# B1: group A's assets (read at call time)
# =====================================================================================================


def _sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pins_are_placeholders() -> bool:
    """Whether any pin still carries ``PIN_PLACEHOLDER`` (or a page id has no pin)."""
    if PIN_PLACEHOLDER in str(ATTEMPTS_SHA256):
        return True
    return any(PIN_PLACEHOLDER in str(PAGE_SHA256.get(p, PIN_PLACEHOLDER)) for p in PAGE_IDS)


def page_path(page_id: str) -> Path:
    return Path(PAGES_DIR) / f"{page_id}.png"


def load_attempts(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """``attempts.json`` by id: ``{"attempts": [...]}``, or a bare list."""
    raw = json.loads(Path(path or ATTEMPTS_PATH).read_text(encoding="utf-8"))
    items = raw.get("attempts") if isinstance(raw, Mapping) else raw
    if not isinstance(items, list):
        raise ValueError("attempts.json holds no list of attempts")
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            out[str(item["id"])] = dict(item)
    return out


def keyword_problem(page_id: str, page: Mapping[str, Any] | None) -> str | None:
    """B1: why a page's judge rows cannot judge (``None``: they can): ``judge_keywords`` a non-empty list of
    non-empty strings, ``false_blame_keywords`` a list of strings."""
    if page is None:
        return f"attempts.json has no page {page_id}"
    jk = page.get("judge_keywords")
    if not isinstance(jk, list) or not jk or not all(isinstance(k, str) and k for k in jk):
        return f"page {page_id} has no non-empty judge_keywords list"
    fb = page.get("false_blame_keywords")
    if not isinstance(fb, list) or not all(isinstance(k, str) and k for k in fb):
        return f"page {page_id} has no false_blame_keywords list"
    return None


# =====================================================================================================
# B4: the judge (pure)
# =====================================================================================================

#: ruling R-2: a ``judge_keywords`` entry may hold alternatives separated by this; any one satisfies it
KEYWORD_ALTERNATIVES = "|"


def keyword_hit(entry: str, folded: str) -> bool:
    """R-2: whether one ``judge_keywords`` entry is satisfied by the casefolded judged text: any of its
    ``|``-separated alternatives (empty ones never) is a casefold substring. No other normalisation."""
    return any(a and a.casefold() in folded for a in str(entry).split(KEYWORD_ALTERNATIVES))


def _joined(text: str | Sequence[str] | None) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return " ".join(str(t) for t in text)


def _locates(text: str, page: Mapping[str, Any]) -> tuple[bool, list[str]]:
    folded = text.casefold()
    keywords = [str(k) for k in page.get("judge_keywords") or []]
    blame = [str(k) for k in page.get("false_blame_keywords") or []]
    matched = [k for k in keywords if keyword_hit(k, folded)]
    located = (
        bool(text.strip())
        and bool(keywords)
        and len(matched) == len(keywords)
        and not any(b.casefold() in folded for b in blame)
    )
    return located, matched


def judge(
    text_spoken: str | Sequence[str] | None,
    text_shown: str | Sequence[str] | None,
    page: Mapping[str, Any],
) -> tuple[bool, bool, list[str]]:
    """B4 (R-1, R-2): ``(located, spoken_only_located, matched)``. The judged text is the spoken text (a string, or
    every spoken text of the turn) and the shown card text joined with single spaces. Located: every
    ``judge_keywords`` entry of ``page`` is satisfied (any of its ``|``-separated alternatives a casefolded
    substring) and no ``false_blame_keywords`` phrase is a casefolded substring; an empty judged text or a page
    without keywords is never located. ``spoken_only_located`` is the same rule over the spoken text alone;
    ``matched`` the entries the judged text satisfies, in the page's order."""
    spoken = _joined(text_spoken)
    shown = _joined(text_shown)
    judged = " ".join(t for t in (spoken, shown) if t)
    located, matched = _locates(judged, page)
    spoken_located, _ = _locates(spoken, page)
    return located, spoken_located, matched


# =====================================================================================================
# B2: the frame the HUD page would post (CPU)
# =====================================================================================================


def frame_jpeg(png_path: Path | str) -> tuple[bytes, int, int, str]:
    """The page PNG as the HUD would post it: RGB (transparency over white), resized with
    ``policy.target_size(..., bucket_for(SCREEN, w, h))``, JPEG at ``policy.JPEG_QUALITY``; with its size and its
    16-hex perceptual hash (the executor's DCT phash)."""
    from PIL import Image

    from local_astra.computer.executor import phash
    from local_astra.contracts.perception import Source
    from local_astra.perception import policy

    with Image.open(png_path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.split()[-1])
        else:
            rgb = im.convert("RGB")
    w0, h0 = rgb.size
    w, h = policy.target_size(w0, h0, policy.bucket_for(Source.SCREEN, w0, h0))
    resized = rgb.resize((w, h), Image.Resampling.BILINEAR) if (w, h) != (w0, h0) else rgb
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=policy.JPEG_QUALITY)
    return buf.getvalue(), w, h, phash(resized)


def frame_message(png_path: Path | str, now_ms: int) -> Any:
    """B2: the ``astra.frame`` the HUD page posts for the shared page: screen source, the resized JPEG's own size,
    a 16-hex phash, no title and no app (no text names the page)."""
    from local_astra.contracts.perception import Source
    from local_astra.contracts.protocol import AstraFrame

    jpeg, w, h, ph = frame_jpeg(png_path)
    return AstraFrame(
        source=Source.SCREEN,
        ts_ms=max(0, int(now_ms)),
        width=w,
        height=h,
        jpeg_b64=base64.b64encode(jpeg).decode("ascii"),
        phash=ph,
        title=None,
        app=None,
    )


def frame_digest(jpeg: bytes) -> str:
    """How the session log names the posted image: sha256 of its base64 text, first 8 hex
    (``telemetry.session_log._redacted``)."""
    b64 = base64.b64encode(jpeg).decode("ascii")
    return hashlib.sha256(b64.encode("utf-8")).hexdigest()[:8]


# =====================================================================================================
# The Session, as cli builds it for ``session --tools off`` (plumbing is the turn's own)
# =====================================================================================================

MODE = "off"


def _off_args() -> argparse.Namespace:
    """``session --tools off`` with no other flag (search is off with tools off)."""
    return argparse.Namespace(tools=MODE)


def session_keywords(profile: Any) -> dict[str, Any]:
    """Every non-plumbing keyword ``cli._build_graph`` passes ``Session`` for ``session --tools off``, each through
    cli's own resolver at call time (keywords cli passes only when set are passed only when set)."""
    from local_astra import cli

    ns = _off_args()
    grounding = cli.grounding_of(profile, ns, MODE)
    guidance = cli.live_guidance_of(profile, ns, MODE).value
    out: dict[str, Any] = {
        "tool_guidance": cli._tool_guidance(
            guidance, cli.web_of(profile, ns, MODE), cli.wait_tool_of(ns, MODE)
        ),
        "visual_tokens": grounding.visual_tokens,
        "reground": grounding.reground,
        "grounding": None,
        "degeneracy_guard": True,
    }
    reask = cli.reask_of(profile, ns, MODE).value
    if reask is not None:
        out["reask_announced_calls"] = reask
    if guidance in cli.PAGE_FACT_GUIDANCE:
        out["page_fact"] = True
    return out


def build_session(
    profile: Any,
    llm: Any,
    frames: Any,
    clock: Any,
    data_dir: Path,
    *,
    turns_path: Path,
    session_log: Any = None,
    log_dir: Path | None = None,
) -> Any:
    """One fresh tools-off ``Session`` (empty history, no memory), ``cli``'s shape, with FakeStt/FakeTts and a
    playback-only fake audio device; then ``init_settings`` with ``cli.settings_of`` (the Settings panel's
    snapshot), so the settings set goes through the session's own validation. P-S and the turns share it."""
    from local_astra import cli
    from local_astra.conductor.session import Session, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts.perception import Source

    ns = _off_args()
    grounding = cli.grounding_of(profile, ns, MODE)
    loop = cli.tool_loop_of(profile, ns)
    ack_clip, still_working_after_ms = cli.clip_policy_of(profile, ns, MODE, loop.on)
    session = Session(
        TurnMachine(
            TurnPolicy(
                thinking=profile.diffusion.thinking,
                tool_loop=loop.on,
                max_tool_steps=loop.max_tool_steps,
                ack_clip=ack_clip,
                still_working_after_ms=still_working_after_ms,
            )
        ),
        llm,
        G._fake("stt", "FakeStt")(),
        G._fake("tts", "FakeTts")(),
        G._PlaybackOnly(G._fake("audio", "FakeAudioIO")(clock)),
        frames,
        G._CaseMemory(),
        clock=clock,
        config=SessionConfig(
            data_dir,
            turns_path=turns_path,
            profile=profile,
            audio_path="null",
            source=Source.SCREEN,
        ),
        session_log=session_log,
        log_dir=log_dir,
        tool_kit=None,
        **session_keywords(profile),
    )
    session.init_settings(*cli.settings_of(profile, ns, MODE, grounding, profile_from_config=False))
    return session


def settings_message() -> Any:
    """B2: the one set, at the fresh session's revision 0."""
    from local_astra.contracts.protocol import AstraSettingsSet

    return AstraSettingsSet(rev=0, changes={"user_instructions": TUTOR_INSTRUCTIONS})


async def wait_settings(page: Any, timeout_s: float = SETTINGS_TIMEOUT_S) -> dict[str, Any]:
    """B2: wait for the page's ``astra.settings.state`` at ``rev == SETTINGS_REV`` with nothing refused; returns
    ``{rev, refused}``. Raises ``RuntimeError`` on a refusal and ``TimeoutError`` when no such echo came."""
    from local_astra.contracts.protocol import AstraSettingsState

    deadline = time.monotonic() + timeout_s
    while True:
        for msg in page.sent():
            if not isinstance(msg, AstraSettingsState):
                continue
            if msg.refused:
                raise RuntimeError(
                    "the settings set was refused: "
                    + "; ".join(f"{r.key}: {r.reason}" for r in msg.refused)
                )
            if msg.rev == SETTINGS_REV:
                return {"rev": msg.rev, "refused": []}
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no astra.settings.state at rev {SETTINGS_REV} in {timeout_s} s")
        await asyncio.sleep(0.02)


# =====================================================================================================
# B3: per-turn evidence (pure)
# =====================================================================================================


def system_text(prompt: Any) -> str | None:
    """The system message's text in a logged prompt (``None`` without one)."""
    if not isinstance(prompt, Mapping):
        return None
    for m in prompt.get("messages") or []:
        if isinstance(m, Mapping) and m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str):
                return content
            return "".join(
                str(p.get("text") or "")
                for p in content or []
                if isinstance(p, Mapping) and p.get("type") == "text"
            )
    return None


def logged_images(prompt: Any) -> list[Any]:
    """The digest of every image part of a logged prompt (the session log redacts an image's url to
    ``{"redacted": "image", "digest": ...}``); ``None`` for a part whose url is not redacted."""
    out: list[Any] = []
    if not isinstance(prompt, Mapping):
        return out
    for m in prompt.get("messages") or []:
        content = m.get("content") if isinstance(m, Mapping) else None
        for p in content if isinstance(content, list) else []:
            if isinstance(p, Mapping) and p.get("type") == "image_url":
                url = p.get("url")
                out.append(url.get("digest") if isinstance(url, Mapping) else None)
    return out


def _is_user_text(r: Mapping[str, Any]) -> bool:
    if r.get("kind") != "machine.step":
        return False
    event = (r.get("data") or {}).get("event")
    return isinstance(event, Mapping) and event.get("type") == "user.text"


def evidence(
    records: Sequence[Mapping[str, Any]],
    line: Mapping[str, Any] | None,
    trial: Mapping[str, Any],
) -> list[str]:
    """B3, over one turn that ran: its session-log records, its turn line and the runner's raw trial
    (``frame_sha256``, ``stored_sha256``, ``frame_digest``). Each violation is ``"<page>/r<repeat>: <what>"``.

    (a) the stored frame is the posted one; at least one ``llm.request``; every ``prefill``/``generate`` request
    holds exactly one image, the posted one (its logged digest). (b) every request's system text ends with
    ``instructions_block()`` and all are one text; the turn line has ``settings_rev == 1``,
    ``instructions_edited`` true and ``system_sha12`` that text's; exactly one ``settings.change``,
    ``applied_at == "immediate"``, logged before the ``UserText`` step. (c) no logged prompt carries ``tools``."""
    tag = f"{trial.get('page')}/r{trial.get('repeat')}"
    out: list[str] = []
    if trial.get("stored_sha256") is None or trial.get("stored_sha256") != trial.get(
        "frame_sha256"
    ):
        out.append(f"{tag}: the stored frame is not the posted one")
    reqs = sorted(
        (
            r
            for r in records
            if r.get("kind") == "llm.request" and isinstance(r.get("data"), Mapping)
        ),
        key=lambda r: int(r.get("seq") or 0),
    )
    datas = [r["data"] for r in reqs]
    if not datas:
        out.append(f"{tag}: no llm.request")
    block = instructions_block()
    systems: set[str] = set()
    for d in datas:
        prompt = d.get("prompt") or {}
        if d.get("phase") in ("prefill", "generate"):
            images = logged_images(prompt)
            if images != [trial.get("frame_digest")]:
                out.append(f"{tag}: {d.get('phase')} {d.get('request_id')} images {images}")
        text = system_text(prompt) or ""
        systems.add(text)
        if not text.endswith(block):
            out.append(
                f"{tag}: {d.get('request_id')} system text does not end with the instructions block"
            )
        if isinstance(prompt, Mapping) and prompt.get("tools"):
            out.append(f"{tag}: {d.get('request_id')} carries tools")
    regime = (line or {}).get("regime") or {}
    if regime.get("settings_rev") != SETTINGS_REV or regime.get("instructions_edited") is not True:
        out.append(
            f"{tag}: turn line settings_rev {regime.get('settings_rev')}, instructions_edited "
            f"{regime.get('instructions_edited')}"
        )
    shas = {hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] for s in systems}
    if regime.get("system_sha12") not in shas or len(shas) != 1:
        out.append(f"{tag}: system_sha12 {regime.get('system_sha12')} vs {sorted(shas)}")
    changes = [r for r in records if r.get("kind") == "settings.change"]
    user = [int(r.get("seq") or 0) for r in records if _is_user_text(r)]
    if (
        len(changes) != 1
        or (changes[0].get("data") or {}).get("applied_at") != "immediate"
        or not user
        or int(changes[0].get("seq") or 0) > min(user)
    ):
        out.append(
            f"{tag}: settings.change {[(c.get('data') or {}).get('applied_at') for c in changes]}"
        )
    return out


def body_summary(label: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """One request body as the meter would send it, reduced to what B3 checks: its phase (the label's prefix),
    the sha256 of every image it carries (decoded from its data URLs), its ``max_soft_tokens`` and whether it
    carries ``tools``."""
    images: list[str] = []
    for m in body.get("messages") or []:
        content = m.get("content") if isinstance(m, Mapping) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, Mapping) or part.get("type") != "image_url":
                continue
            url = str((part.get("image_url") or {}).get("url") or "")
            _, _, payload = url.partition("base64,")
            try:
                images.append(hashlib.sha256(base64.b64decode(payload)).hexdigest())
            except ValueError:
                images.append("undecodable")
    mm = body.get("mm_processor_kwargs")
    return {
        "label": label,
        "phase": label.split(":", 1)[0],
        "images": images,
        "max_soft_tokens": mm.get("max_soft_tokens") if isinstance(mm, Mapping) else None,
        "tools": "tools" in body,
    }


def body_evidence(trial: Mapping[str, Any]) -> list[str]:
    """B3 (a)/(c) over the meter's rebuilt bodies, when the trial carries them (``bodies``): at least one generate
    body; every ``prefill``/``generate`` body holds exactly one image, the posted JPEG (sha256); every body carries
    ``max_soft_tokens == 1120`` and no ``tools``. A trial without ``bodies`` gives nothing."""
    bodies = trial.get("bodies")
    if bodies is None:
        return []
    tag = f"{trial.get('page')}/r{trial.get('repeat')}"
    out: list[str] = []
    if not any(b.get("phase") == "generate" for b in bodies):
        out.append(f"{tag}: no generate body")
    for b in bodies:
        if b.get("phase") in ("prefill", "generate") and list(b.get("images") or []) != [
            trial.get("frame_sha256")
        ]:
            out.append(f"{tag}: body {b.get('label')} images {b.get('images')}")
        if b.get("max_soft_tokens") != VISUAL_TOKENS:
            out.append(f"{tag}: body {b.get('label')} max_soft_tokens {b.get('max_soft_tokens')}")
        if b.get("tools"):
            out.append(f"{tag}: body {b.get('label')} carries tools")
    return out


# =====================================================================================================
# Turn records (pure: they read the run dir)
# =====================================================================================================


def status(trial: Mapping[str, Any], records: Sequence[Mapping[str, Any]] = ()) -> str:
    """``not_run`` (no session log); ``errored`` (an exception, or R-3: any ``llm`` error record in the turn's
    session log, a non-200 attempt (``http_non_200``) or ``engine_up_after is False``); ``timed_out``; or ``ran``."""
    if not trial.get("session_log"):
        return "not_run"
    if (
        trial.get("exception")
        or llm_errors(records) > 0
        or int(trial.get("http_non_200") or 0) > 0
        or trial.get("engine_up_after") is False
    ):
        return "errored"
    if trial.get("timed_out"):
        return "timed_out"
    return "ran"


def spoken_texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every spoken text of the turn, in ``seq`` order: the non-empty ``source`` of each ``tts.speak`` command in
    its ``machine.step`` records. Clips are ``play_clip`` commands, never here; thoughts never reach ``tts.speak``."""
    out: list[str] = []
    steps = sorted(
        (r for r in records if r.get("kind") == "machine.step"),
        key=lambda r: int(r.get("seq") or 0),
    )
    for r in steps:
        for c in (r.get("data") or {}).get("commands") or []:
            if isinstance(c, Mapping) and c.get("type") == "tts.speak":
                src = c.get("source")
                if isinstance(src, str) and src:
                    out.append(src)
    return out


def _session_records(out: Path, name: str | None) -> list[dict[str, Any]] | None:
    if not name:
        return None
    path = Path(out) / "sessions" / str(name)
    if not path.is_file():
        return None
    return sorted(G._read_jsonl(path), key=lambda r: int(r.get("seq") or 0))


def turn_line(
    out: Path, name: str | None, records: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any] | None:
    """The turn line of the session whose log is ``name``: from ``turns.jsonl`` (by its ``session_log`` file name),
    else the log's own ``turn.record``."""
    if not name:
        return None
    for item in G._read_jsonl(Path(out) / "turns.jsonl"):
        if Path(str(item.get("session_log") or "")).name == str(name):
            return item
    for r in records:
        if r.get("kind") == "turn.record" and isinstance(r.get("data"), Mapping):
            return dict(r["data"])
    return None


def first_word_ms(line: Mapping[str, Any] | None, user_text_ms: Any) -> int | None:
    """B5: ``spans["tts.first_audio"][0] - user_text_ms`` (both on the session clock); ``None`` when the turn did
    not speak."""
    span = ((line or {}).get("spans") or {}).get("tts.first_audio")
    if not isinstance(span, list | tuple) or not span or not isinstance(user_text_ms, int | float):
        return None
    return int(span[0]) - int(user_text_ms)


def llm_errors(records: Sequence[Mapping[str, Any]]) -> int:
    """``error`` records about an LLM call (``what`` starting ``llm``)."""
    return sum(
        1
        for r in records
        if r.get("kind") == "error"
        and str((r.get("data") or {}).get("what") or "").startswith("llm")
    )


def finish_turn(
    out: Path,
    page_id: str,
    repeat: int,
    trial: Mapping[str, Any],
    attempts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One turn's trial line: the runner's raw trial (``page``, ``repeat``, ``session_log``, ``exception``,
    ``timed_out``, ``frame_sha256``, ``stored_sha256``, ``frame_digest``, ``user_text_ms``, ``settings_echo``,
    ``wall_s``, and any extra the runner keeps) plus what the run dir says: ``status``, ``spoken``, ``shown``,
    ``clips``, ``located``, ``spoken_only_located``, ``matched``, ``first_word_ms``, ``llm_errors``,
    ``violations`` (B3, on a turn that ran) and ``body_violations``. Idempotent: the report re-finishes every
    trial from its raw fields."""
    pages = attempts if attempts is not None else load_attempts()
    done: dict[str, Any] = dict(trial)
    done["page"], done["repeat"] = str(page_id), int(repeat)
    records = _session_records(out, done.get("session_log")) or []
    st = status(done, records)
    line = turn_line(out, done.get("session_log"), records) if st != "not_run" else None
    spoken = spoken_texts(records)
    shown = shown_text(records)
    located, spoken_located, matched = judge(spoken, shown, pages.get(str(page_id)) or {})
    ran = st == "ran"
    violations: list[str] = []
    if ran:
        if _session_records(out, done.get("session_log")) is None:
            violations.append(f"{page_id}/r{repeat}: session log {done.get('session_log')} missing")
        else:
            violations = evidence(records, line, done)
    done.update(
        status=st,
        spoken=spoken,
        shown=shown,
        clips=clip_kinds(records),
        located=bool(located and ran),
        spoken_only_located=bool(spoken_located and ran),
        matched=matched,
        first_word_ms=first_word_ms(line, done.get("user_text_ms")) if ran else None,
        llm_errors=llm_errors(records),
        violations=violations,
        body_violations=body_evidence(done) if ran else [],
    )
    return done


def not_run(page_id: str, repeat: int, reason: str) -> dict[str, Any]:
    """The raw trial of a turn that was never run (counted as errored, never located)."""
    return {
        "page": page_id,
        "repeat": int(repeat),
        "session_log": None,
        "exception": reason,
        "timed_out": False,
        "frame_sha256": None,
        "stored_sha256": None,
        "frame_digest": None,
        "user_text_ms": None,
        "settings_echo": None,
        "wall_s": None,
    }


# =====================================================================================================
# B5/B6: the label and the report (from the run dir)
# =====================================================================================================


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def reading_line(pages: Mapping[str, Mapping[str, Any]]) -> str:
    n = sum(1 for p in PAGE_IDS if int((pages.get(p) or {}).get("located") or 0) >= 2)
    return f"{n} of {len(PAGE_IDS)} pages located in ≥ 2 of {REPEATS} turns"


def build_report(out: Path) -> dict[str, Any]:
    """``report.json`` from the run dir alone: ``order.json``, ``manifest.json`` (``regime``, ``stopped``),
    ``probes.json``, ``trials.jsonl`` (each re-finished by ``finish_turn``), ``turns.jsonl``, ``sessions/`` and the
    pinned ``attempts.json``. ``label`` (B6: ``void`` when a probe is missing or failed or any turn that ran has a
    violation; ``incomplete`` when a turn of the order has no trial, a trial did not run cleanly, or the run
    stopped; else ``measured``), ``violations``, ``body_violations``, ``pages`` (k/3 located and spoken-only,
    n = 3), ``turns`` (one per turn of the order), ``reading``, ``latency`` (p50 = low median, max, n over the turns
    that spoke; the cold first turn marked; warm-only figures beside), engine errors, minimum free VRAM, ``regime``.
    No bar."""
    out = Path(out)
    attempts = load_attempts()
    turn_order = [
        (str(p), int(r)) for p, r in _read_json(out / "order.json", None) or order(ORDER_SEED)
    ]
    manifest = _read_json(out / "manifest.json", {}) or {}
    probes = _read_json(out / "probes.json", {}) or {}
    raw = G._read_jsonl(out / "trials.jsonl")
    finished = {
        (str(t.get("page")), int(t.get("repeat") or 0)): finish_turn(
            out, str(t.get("page")), int(t.get("repeat") or 0), t, attempts
        )
        for t in raw
    }
    turns: list[dict[str, Any]] = []
    violations: list[str] = []
    body_violations: list[str] = []
    latencies: list[int] = []
    warm: list[int] = []
    cold_ms: int | None = None
    incomplete = bool(manifest.get("stopped"))
    statuses: dict[str, int] = {}
    pages: dict[str, dict[str, int]] = {
        p: {"located": 0, "spoken_only_located": 0, "n": REPEATS} for p in PAGE_IDS
    }
    for index, key in enumerate(turn_order):
        t = finished.get(key)
        st = "missing" if t is None else str(t["status"])
        statuses[st] = statuses.get(st, 0) + 1
        if st != "ran":
            incomplete = True
        if t is None:
            turns.append(
                {
                    "page": key[0],
                    "repeat": key[1],
                    "index": index,
                    "status": st,
                    "located": False,
                    "spoken_only_located": False,
                    "matched": [],
                    "spoken": [],
                    "shown": "",
                    "clips": [],
                    "first_word_ms": None,
                    "violations": [],
                    "exception": "no trial line",
                    "hand_audit": None,
                }
            )
            continue
        violations += list(t["violations"])
        body_violations += list(t["body_violations"])
        if key[0] in pages:
            pages[key[0]]["located"] += int(bool(t["located"]))
            pages[key[0]]["spoken_only_located"] += int(bool(t["spoken_only_located"]))
        ms = t["first_word_ms"]
        if isinstance(ms, int):
            latencies.append(ms)
            if index == 0:
                cold_ms = ms
            else:
                warm.append(ms)
        turns.append(
            {
                "page": key[0],
                "repeat": key[1],
                "index": index,
                "status": st,
                "located": bool(t["located"]),
                "spoken_only_located": bool(t["spoken_only_located"]),
                "matched": list(t["matched"]),
                "spoken": list(t["spoken"]),
                "shown": str(t["shown"]),
                "clips": list(t["clips"]),
                "first_word_ms": ms,
                "violations": list(t["violations"]) + list(t["body_violations"]),
                "exception": t.get("exception"),
                "hand_audit": t.get("hand_audit"),
            }
        )
    probes_ok = set(probes) == set(PROBE_NAMES) and all(
        bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES
    )
    if not probes_ok or violations or body_violations:
        label = VOID
    elif incomplete:
        label = INCOMPLETE
    else:
        label = MEASURED
    regime = dict(manifest.get("regime") or {})
    regime.setdefault(
        "attempts_sha256", manifest.get("attempts_sha256") or _sha256_file(ATTEMPTS_PATH)
    )
    vram = [
        int(t["vram_min_free_mib"])
        for t in finished.values()
        if t.get("vram_min_free_mib") is not None
    ]
    return {
        "label": label,
        "stopped": manifest.get("stopped"),
        "probes": {n: bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES},
        "violations": violations,
        "body_violations": body_violations,
        "statuses": statuses,
        "pages": pages,
        "reading": reading_line(pages),
        "latency": {
            "p50_ms": statistics.median_low(latencies) if latencies else None,
            "max_ms": max(latencies) if latencies else None,
            "n": len(latencies),
            "cold_first_turn_ms": cold_ms,
            "warm_p50_ms": statistics.median_low(warm) if warm else None,
            "warm_max_ms": max(warm) if warm else None,
            "warm_n": len(warm),
            "what": LATENCY_NOTE,
        },
        "llm_errors": sum(int(t.get("llm_errors") or 0) for t in finished.values()),
        "http_non_200": sum(int(t.get("http_non_200") or 0) for t in finished.values()),
        "vram_min_free_mib": min(vram) if vram else None,
        "probe_vram_min_free_mib": manifest.get("probe_vram_min_free_mib"),
        "wall_s": manifest.get("wall_s"),
        "turns": turns,
        "regime": regime,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """``report.md``: the label, the reading line verbatim, the per-page table, latency, every turn's spoken and
    shown text verbatim with the entries it matched (the hand audit), and the regime."""
    lat = report.get("latency") or {}
    lines = [
        "# T-0223 B handwritten check: does Gearvis locate the planted mistake? (report-only: no bar)",
        "",
        f"- label: **{report['label']}**",
        f"- stopped early: {_fmt(report.get('stopped'))}",
        "- probes: " + ", ".join(f"{k} {v}" for k, v in (report.get("probes") or {}).items()),
        f"- turn statuses: {_fmt(report.get('statuses'))}",
        f"- B3 violations: {_fmt(report.get('violations'))}",
        f"- body violations: {_fmt(report.get('body_violations'))}",
        f"- llm.error records: {report.get('llm_errors')}; non-200 generates: {report.get('http_non_200')}",
        f"- min free VRAM MiB over the turns: {_fmt(report.get('vram_min_free_mib'))} (probes: "
        f"{_fmt(report.get('probe_vram_min_free_mib'))})",
        f"- wall time s: {_fmt(report.get('wall_s'))}",
        f"- scope: {SCOPE_NOTE}",
        f"- judge: {JUDGE_NOTE}",
        f"- frame: {FRAME_NOTE}",
        f"- probes: {PS_NOTE}",
        "",
        f"**Reading: {report['reading']}**",
        "",
        "## Per page",
        "",
        "| page | located k/3 | spoken-only k/3 |",
        "|---|---|---|",
    ]
    for pid, row in (report.get("pages") or {}).items():
        lines.append(
            f"| {pid} | {row['located']}/{row['n']} | {row['spoken_only_located']}/{row['n']} |"
        )
    lines += [
        "",
        "## Time to first spoken word",
        "",
        f"- {lat.get('what')}",
        f"- turns that spoke (n = {lat.get('n')}): p50 {_fmt(lat.get('p50_ms'))} ms, max "
        f"{_fmt(lat.get('max_ms'))} ms; cold first turn {_fmt(lat.get('cold_first_turn_ms'))} ms",
        f"- warm only (n = {lat.get('warm_n')}): p50 {_fmt(lat.get('warm_p50_ms'))} ms, max "
        f"{_fmt(lat.get('warm_max_ms'))} ms",
        "",
        "## Every turn",
        "",
        HAND_AUDIT_NOTE,
        "",
        "| # | page | repeat | status | located | spoken-only | matched | clips | first word ms | hand audit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in report.get("turns") or []:
        lines.append(
            f"| {a['index']}{' (cold)' if a['index'] == 0 else ''} | {a['page']} | {a['repeat']} | {a['status']} | "
            f"{a['located']} | {a['spoken_only_located']} | {', '.join(a['matched']) or '-'} | "
            f"{', '.join(a['clips']) or '-'} | {_fmt(a['first_word_ms'])} | {_fmt(a['hand_audit'])} |"
        )
    lines.append("")
    for a in report.get("turns") or []:
        lines += [
            f"### {a['page']}/r{a['repeat']}: {a['status']}, located {a['located']}, spoken-only "
            f"{a['spoken_only_located']}, violations {_fmt(a['violations'] or None)}, error {_fmt(a['exception'])}",
            "",
            "spoken:",
            "",
        ]
        lines += list(a["spoken"]) or ["(none)"]
        lines += ["", "shown:", ""]
        lines += [ln for ln in str(a["shown"]).splitlines()] or ["(none)"]
        lines.append("")
    lines += [
        "## Regime",
        "",
        "```",
        json.dumps(report.get("regime") or {}, indent=1, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(out: Path) -> dict[str, Any]:
    """``report.json`` and ``report.md`` under ``out``, from the run dir (``build_report``)."""
    out = Path(out)
    report = build_report(out)
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


# =====================================================================================================
# P-S (CPU, B1) and P0 (engine, B2)
# =====================================================================================================


def _ps_commits(request_id: str) -> list[Any]:
    from local_astra.contracts.llm import Commit

    return [
        Commit(request_id=request_id, canvas_idx=0, segment="thought", text="Checking the lines."),
        Commit(
            request_id=request_id,
            canvas_idx=1,
            segment="answer",
            text="Line three has the mistake.",
        ),
    ]


async def _probe_ps(png: Path | None) -> dict[str, Any]:
    from local_astra.conductor.hud_server import HudServer
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import SessionClock
    from local_astra.config import load
    from local_astra.contracts import events as E
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello
    from local_astra.perception.browser_source import BrowserFrameSource

    profile = load().profiles["lean"]
    llm = G._fake("llm", "FakeLlm")(_ps_commits("ps"))
    clock = SessionClock()
    frames = BrowserFrameSource(now_ms=lambda: clock.now_ms)
    out: dict[str, Any] = {"passed": False, "error": None}
    with tempfile.TemporaryDirectory(prefix="t0223-ps-") as tmp:
        session = build_session(
            profile, llm, frames, clock, Path(tmp), turns_path=Path(tmp) / "turns.jsonl"
        )
        try:
            hud = HudServer(session, frame_sink=frames, outbox_size=HUD_OUTBOX)
            page = hud.open()
            await session.start()
            page.receive(encode(AstraHello()))
            page.receive(encode(settings_message()))
            await wait_settings(page)
            posted_sha: str | None = None
            if png is not None:
                message = frame_message(png, clock.now_ms)
                posted_sha = hashlib.sha256(base64.b64decode(message.jpeg_b64)).hexdigest()
                page.receive(encode(message))
            session.turn_written.clear()
            session.submit(E.UserText(t_ms=clock.now_ms, text=QUESTION))
            await asyncio.wait_for(session.turn_written.wait(), timeout=20)
            prompts = [p for _, p, _ in llm.generates]
            if not prompts:
                raise RuntimeError("the rig session sent no generate")
            # the first body exactly as the real client would build it (construction sends nothing)
            body = VllmClient(VllmEndpoint(), visual_tokens=VISUAL_TOKENS).request_body(
                prompts[0], "ps"
            )
            system = body["messages"][0]
            content = system.get("content")
            text = (
                content
                if isinstance(content, str)
                else "".join(
                    str(p.get("text") or "") for p in content or [] if p.get("type") == "text"
                )
            )
            out["system_role"] = system.get("role")
            out["system_tail"] = text[-len(instructions_block()) - 40 :]
            out["max_soft_tokens"] = (body.get("mm_processor_kwargs") or {}).get("max_soft_tokens")
            out["tools"] = "tools" in body
            # R-5: the posted frame reaches the generate body (and every prefill body), as the one image
            client = VllmClient(VllmEndpoint(), visual_tokens=VISUAL_TOKENS)
            out["generate_images"] = body_summary("generate:ps", body)["images"]
            out["prefill_images"] = [
                body_summary("prefill:" + rid, client.request_body(prefix, rid))["images"]
                for rid, prefix in llm.prefills
            ]
            out["image_in_body"] = posted_sha is not None and all(
                images == [posted_sha]
                for images in [out["generate_images"], *out["prefill_images"]]
            )
            out["passed"] = (
                system.get("role") == "system"
                and text.endswith(instructions_block())
                and out["max_soft_tokens"] == VISUAL_TOKENS
                and not out["tools"]
                and out["image_in_body"]
            )
        except Exception as exc:
            out["error"] = repr(exc)
        finally:
            with contextlib.suppress(Exception):
                await session.aclose()
    return out


def probe_ps(png: Path | None = None) -> dict[str, Any]:
    """B2 P-S (CPU, no socket): a rig ``Session`` built exactly as the run's (``build_session``), over a scripted
    fake LLM; the settings set through ``HudServer.route``; one frame; a typed turn; its first body's system text
    must end with ``USER_INSTRUCTIONS_HEADER + "\\n" + TUTOR_INSTRUCTIONS`` (and carry ``max_soft_tokens`` 1120,
    no tools); R-5: the posted frame is the one image of the generate body and of every prefill body."""
    return asyncio.run(_probe_ps(png))


async def _probe_generate(base_url: str, jpeg: bytes | None, request_id: str) -> int:
    """One generate at ``VISUAL_TOKENS`` (with one image, or text only); its HTTP status, 0 without a response."""
    import httpx

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart

    parts: list[Any] = []
    if jpeg is not None:
        parts.append(
            ImagePart(url="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
        )
    parts.append(TextPart(text=T.PROBE_TEXT))
    prompt = Prompt(messages=(Message(role="user", content=tuple(parts)),), prefix_len=0)
    client = VllmClient(
        VllmEndpoint(base_url=base_url),
        transport=httpx.AsyncHTTPTransport(),
        visual_tokens=VISUAL_TOKENS,
    )
    try:
        async for _commit in client.generate(prompt, request_id, LlmOptions()):
            pass
    except Exception as exc:
        status_ = G._http_status(exc)
        _say(f"probe {request_id}: {exc!r}")
        return status_ if status_ is not None else 0
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
    return 200


def probe_p0(base_url: str, model: str, frames: Mapping[str, bytes]) -> dict[str, Any]:
    """B2 P0 (engine): one text and one image generate at ``max_soft_tokens`` 1120 must answer 200 (the image is
    the first page's frame), the engine healthy after. Also, for the report and never gating: each page frame's
    image-token count at 1120 (``/tokenize`` with and without it) beside ``policy.expected_soft_tokens``."""
    from PIL import Image

    from local_astra.perception.policy import expected_soft_tokens

    first = frames[PAGE_IDS[0]]
    text_status = asyncio.run(_probe_generate(base_url, None, "t0223-probe-p0-text"))
    image_status = asyncio.run(_probe_generate(base_url, first, "t0223-probe-p0-image"))
    healthy = T._healthy(base_url)
    tokens: dict[str, Any] = {}
    bare = T._tokenize_count(base_url, model, VISUAL_TOKENS, None)
    for pid, jpeg in frames.items():
        with Image.open(io.BytesIO(jpeg)) as im:
            w, h = im.size
        with_image = T._tokenize_count(base_url, model, VISUAL_TOKENS, jpeg)
        tokens[pid] = {
            "size": [w, h],
            "image_tokens": (
                with_image - bare if with_image is not None and bare is not None else None
            ),
            "expected_soft_tokens": expected_soft_tokens(w, h, VISUAL_TOKENS),
        }
    return {
        "passed": text_status == 200 and image_status == 200 and healthy,
        "text_status": text_status,
        "image_status": image_status,
        "healthy_after": healthy,
        "image_tokens": tokens,
    }


# =====================================================================================================
# B1: refusals (exit 2), all before any socket
# =====================================================================================================


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts with ``subject`` is reachable from HEAD."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


def default_out(sha: str) -> Path:
    return REPO / "logs" / f"t0223-handwritten-{sha[:7]}"


def _refusal(args: argparse.Namespace) -> str | None:
    """B1: why the run is refused (exit 2), or ``None``. Checked in order, all before any socket: a non-loopback
    ``--base-url``; a dirty tree (its paths named); an existing out-dir; no ``Merge T-0223-a:`` or ``Merge
    T-0213-a2a:`` merge reachable from HEAD; a pin still a placeholder; ``attempts.json`` or a page PNG off its
    pin; a page without non-empty ``judge_keywords`` or a ``false_blame_keywords`` list; ``TUTOR_INSTRUCTIONS``
    refused by ``prompt_builder.validate_instructions``; 1120 visual tokens not in force; P-S failing (its record
    is kept in ``LAST_PS``)."""
    LAST_PS.clear()
    try:
        G.check_loopback(str(args.base_url))
    except ValueError as exc:
        return f"--base-url is not loopback: {exc}"
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})"
    if dirty:
        return f"the tree at {REPO} is dirty:\n{dirty}"
    out = Path(args.out_dir) if getattr(args, "out_dir", None) else default_out(sha)
    if out.exists():
        return f"{out} exists; the run happens once per SHA (move it aside only with the human)"
    for subject, what in (
        (MERGE_A, "T-0223 group A, the attempts and their pages"),
        (MERGE_A2A, "T-0213 A2-apply, the settings set"),
    ):
        if not _merged(subject):
            return f"no '{subject}' merge is reachable from HEAD ({what}; a branch merge does not count)"
    if pins_are_placeholders():
        return (
            "the attempts.json / page sha256 pins are still placeholders; fill ATTEMPTS_SHA256 and PAGE_SHA256 "
            "from the tree at 'Merge T-0223-a:'"
        )
    got = _sha256_file(Path(ATTEMPTS_PATH))
    if got != ATTEMPTS_SHA256:
        return f"{ATTEMPTS_PATH} has sha256 {got}, not the pinned {ATTEMPTS_SHA256}"
    for pid in PAGE_IDS:
        got = _sha256_file(page_path(pid))
        if got != PAGE_SHA256.get(pid):
            return f"page {pid}: {page_path(pid)} has sha256 {got}, not the pinned {PAGE_SHA256.get(pid)}"
    try:
        attempts = load_attempts()
    except (OSError, ValueError) as exc:
        return f"attempts.json cannot be read: {exc}"
    for pid in PAGE_IDS:
        problem = keyword_problem(pid, attempts.get(pid))
        if problem is not None:
            return problem
    from local_astra.conductor import prompt_builder

    why = prompt_builder.validate_instructions(TUTOR_INSTRUCTIONS)
    if why is not None:
        return f"the tutor instructions are refused by validate_instructions: {why}"
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    budget = cli.grounding_of(lean, _off_args(), MODE).visual_tokens
    kw_budget = session_keywords(lean).get("visual_tokens")
    if budget != VISUAL_TOKENS or kw_budget != VISUAL_TOKENS:
        return (
            f"the visual-token budget in force is {budget} (Session keyword {kw_budget}), not {VISUAL_TOKENS} "
            "(cli.grounding_of, tools off)"
        )
    try:
        ps = probe_ps(page_path(PAGE_IDS[0]))
    except Exception as exc:
        ps = {"passed": False, "error": repr(exc)}
    LAST_PS.update(ps)
    if not ps.get("passed"):
        return f"the CPU settings probe P-S failed: {json.dumps(ps, default=str)}"
    return None


# =====================================================================================================
# The turn runner
# =====================================================================================================


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Any
    manifest: dict[str, Any]
    vram: Any
    attempts: dict[str, dict[str, Any]]
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


async def _one_turn(run: _Run, page_id: str, repeat: int) -> dict[str, Any]:
    """One typed turn on a fresh Session (tools off, search off, 1120), B2's sequence; its raw trial."""
    from local_astra.conductor.hud_server import HudServer
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.conductor.session import SessionClock
    from local_astra.contracts import events as E
    from local_astra.contracts.codec import encode
    from local_astra.contracts.perception import Source
    from local_astra.contracts.protocol import AstraHello
    from local_astra.perception.browser_source import BrowserFrameSource
    from local_astra.telemetry.session_log import SessionLog

    window_start = time.monotonic()
    idle_before = await G._wait_idle(run.base_url)
    clock = SessionClock()
    frames = BrowserFrameSource(now_ms=lambda: clock.now_ms)
    sessions_dir = run.out / "sessions"
    log = SessionLog(sessions_dir)
    meter: Any = None
    session: Any = None
    trial: dict[str, Any] = {
        "page": page_id,
        "repeat": int(repeat),
        "session_log": Path(log.path).name,
        "exception": None,
        "timed_out": False,
        "frame_sha256": None,
        "stored_sha256": None,
        "frame_digest": None,
        "user_text_ms": None,
        "settings_echo": None,
        "wall_s": None,
    }
    try:
        meter = LoopMeter(
            F._client(VllmEndpoint(base_url=run.base_url), None, VISUAL_TOKENS),
            metrics=lambda: G.read_metrics(run.base_url),
            clock=time.monotonic,
        )
        session = build_session(
            run.profile,
            meter,
            frames,
            clock,
            run.out / "data",
            turns_path=run.out / "turns.jsonl",
            session_log=log,
            log_dir=sessions_dir,
        )
        hud = HudServer(session, frame_sink=frames, outbox_size=HUD_OUTBOX)  # never listens
        page = hud.open()
        await session.start()
        page.receive(encode(AstraHello()))
        page.receive(encode(settings_message()))
        trial["settings_echo"] = await wait_settings(page)
        # B2: the frame after the settings echo, right before the question
        message = frame_message(page_path(page_id), clock.now_ms)
        posted = base64.b64decode(message.jpeg_b64)
        trial["frame_sha256"] = hashlib.sha256(posted).hexdigest()
        trial["frame_digest"] = frame_digest(posted)
        page.receive(encode(message))
        got = await frames.get_frame(Source.SCREEN)
        trial["stored_sha256"] = hashlib.sha256(got.jpeg).hexdigest() if got is not None else None
        session.turn_written.clear()
        submitted = time.monotonic()
        trial["user_text_ms"] = int(clock.now_ms)
        session.submit(E.UserText(t_ms=trial["user_text_ms"], text=QUESTION))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=TURN_TIMEOUT_S)
        except TimeoutError:
            trial["timed_out"] = True
        trial["wall_s"] = round(time.monotonic() - submitted, 2)
    except Exception as exc:
        trial["exception"] = repr(exc)
    finally:
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        if meter is not None:
            with contextlib.suppress(Exception):
                await meter.aclose()
    if session is not None and session.failures and trial["exception"] is None:
        trial["exception"] = "session failures: " + "; ".join(repr(f) for f in session.failures)
    if not Path(log.path).is_file():
        trial["session_log"] = None
        trial["exception"] = trial["exception"] or "no session log was written"
    bodies: list[dict[str, Any]] = []
    http_non_200 = 0
    if meter is not None:
        for label, prompt in meter.requests.items():
            bodies.append(
                body_summary(label, meter.inner.request_body(prompt, label.split(":", 1)[1]))
            )
        http_non_200 = sum(1 for g in meter.generates if g.get("http_status") not in (200, None))
    trial.update(
        bodies=bodies,
        http_non_200=http_non_200,
        engine_idle_before=idle_before,
        engine_up_after=await asyncio.to_thread(T._healthy, run.base_url),
        vram_min_free_mib=run.vram.min_between(window_start, time.monotonic()),
    )
    return trial


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``, concurrency 1. Before each: ``budget_stop`` (from ``PROCESS_START_S``,
    probes included) and ``G.ERRORED_STOP`` consecutive turns that did not run cleanly stop the run; every turn not
    run is then written as not run (counted as errored), so the denominator stays ``REPEATS`` per page."""
    errored = 0
    for index, (page_id, repeat) in enumerate(order(ORDER_SEED)):
        if run.stopped is None:
            elapsed = time.monotonic() - PROCESS_START_S
            if errored >= G.ERRORED_STOP:
                run.stopped = f"{errored} consecutive errored turns (at turn {index})"
            elif budget_stop(elapsed):
                run.stopped = (
                    f"the run budget of {RUN_BUDGET_S} s cannot fit a {TURN_TIMEOUT_S} s turn and the "
                    f"{REPORT_MARGIN_S} s report after {elapsed:.0f} s (at turn {index})"
                )
            if run.stopped is not None:
                run.manifest["stopped"] = run.stopped
                G._write_json(run.out / "manifest.json", run.manifest)
                _say(f"stopping: {run.stopped}")
        if run.stopped is not None:
            raw = not_run(page_id, repeat, run.stopped)
        else:
            started = time.monotonic()
            try:
                raw = await _one_turn(run, page_id, repeat)
            except Exception as exc:
                raw = not_run(page_id, repeat, f"harness: {exc!r}")
            run.turn_s.append(round(time.monotonic() - started, 1))
        trial = finish_turn(run.out, page_id, repeat, raw, run.attempts)
        trial["index"] = index
        errored = errored + 1 if trial["status"] != "ran" else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        if raw.get("session_log"):
            _say(
                f"{index:02d} {page_id}/r{repeat} status={trial['status']} located={trial['located']} "
                f"spoken_only={trial['spoken_only_located']} matched={len(trial['matched'])} "
                f"first_word_ms={trial['first_word_ms']} violations={len(trial['violations'])} "
                f"body_violations={len(trial['body_violations'])} err={trial.get('exception')} "
                f"wall={run.turn_s[-1] if run.turn_s else None}s"
            )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    why = _refusal(args)
    if why is not None:
        _say(f"refusing: {why}")
        return EXIT_REFUSED
    ps = dict(LAST_PS)
    sha = _git("rev-parse", "HEAD").strip()
    out: Path = args.out_dir or default_out(sha)

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.config import load

    model = VllmEndpoint().model
    listing = G._get_text(args.base_url.rstrip("/") + "/models")
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    lean = load().profiles["lean"]
    vram = T._VramSampler()
    vram.start()
    try:
        return _measure(args.base_url, out, sha, model, lean, vram, ps)
    finally:
        vram.stop()


def _measure(
    base_url: str, out: Path, sha: str, model: str, lean: Any, vram: Any, ps: Mapping[str, Any]
) -> int:
    """Past every refusal and the engine check: the order, the manifest, the probes, the turns, the report."""
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in order(ORDER_SEED)])
    attempts = load_attempts()
    frames: dict[str, bytes] = {}
    frame_sizes: dict[str, list[int]] = {}
    for pid in PAGE_IDS:
        jpeg, w, h, _ = frame_jpeg(page_path(pid))
        frames[pid] = jpeg
        frame_sizes[pid] = [w, h]
    attempts_sha = _sha256_file(Path(ATTEMPTS_PATH))
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0223_handwritten_check.sh refuses while 127.0.0.1:8080/8081 are bound",
        "arms": None,
        "commit": sha,
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "base_url": base_url,
        "served_model": model,
        "engine_version": G._get_text(G._root(base_url) + "/version"),
        "profile": lean.name,
        "canvas_length": lean.diffusion.canvas_length,
        "max_denoising_steps": lean.diffusion.max_denoising_steps,
        "thinking": lean.diffusion.thinking,
        "tools": "off",
        "search": "off",
        "visual_tokens": VISUAL_TOKENS,
        "jpeg_quality": 85,
        "frame_sizes": frame_sizes,
        "frames_per_turn": 1,
        "user_instructions": TUTOR_INSTRUCTIONS,
        "question": QUESTION,
        "pages": list(PAGE_IDS),
        "n": REPEATS,
        "turns": TOTAL_TURNS,
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "report_margin_s": REPORT_MARGIN_S,
        "attempts_sha256": attempts_sha,
        "input": "typed (UserText); FakeStt/FakeTts/fake audio, STT down; a fresh Session per turn (empty history "
        "and memory); hello -> astra.settings.set (user_instructions) -> echo rev 1 -> one astra.frame (screen) "
        "through HudServer.route into the session's BrowserFrameSource -> UserText; no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "session_keywords": {k: D._jsonable(v) for k, v in session_keywords(lean).items()},
        "attempts_sha256": attempts_sha,
        "page_sha256": {pid: _sha256_file(page_path(pid)) for pid in PAGE_IDS},
        "frame_sha256": {pid: hashlib.sha256(j).hexdigest() for pid, j in frames.items()},
        "notes": [FRAME_NOTE, LATENCY_NOTE, JUDGE_NOTE, PS_NOTE],
        "stopped": None,
        "wall_s": None,
    }
    G._write_json(out / "manifest.json", manifest)
    (out / "trials.jsonl").touch()
    (out / "turns.jsonl").touch()

    probes: dict[str, Any] = {"P-S": dict(ps)}
    probe_start = time.monotonic()
    try:
        probes["P0"] = probe_p0(base_url, model, frames)
    except Exception as exc:
        probes["P0"] = {"passed": False, "error": repr(exc)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES):
        manifest["stopped"] = "not run: a probe did not answer as required"
        for pid, repeat in order(ORDER_SEED):
            G._append_jsonl(
                out / "trials.jsonl",
                [
                    finish_turn(
                        out, pid, repeat, not_run(pid, repeat, manifest["stopped"]), attempts
                    )
                ],
            )
        manifest["wall_s"] = round(time.monotonic() - PROCESS_START_S, 1)
        G._write_json(out / "manifest.json", manifest)
        report = write_report(out)
        _say(f"P0 did not answer 200: the run is void; label {report['label']}")
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = _Run(
        base_url=base_url, out=out, profile=lean, manifest=manifest, vram=vram, attempts=attempts
    )
    try:
        asyncio.run(_run_all(run))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
        done = {
            (str(t.get("page")), int(t.get("repeat") or 0))
            for t in G._read_jsonl(out / "trials.jsonl")
        }
        for pid, repeat in order(ORDER_SEED):
            if (pid, repeat) not in done:
                G._append_jsonl(
                    out / "trials.jsonl",
                    [finish_turn(out, pid, repeat, not_run(pid, repeat, run.stopped), attempts)],
                )
    manifest["stopped"] = run.stopped
    manifest["wall_s"] = round(time.monotonic() - PROCESS_START_S, 1)
    G._write_json(out / "manifest.json", manifest)

    report = write_report(out)
    _say(f"label={report['label']}; {report['reading']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if report["label"] == INCOMPLETE else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
