"""T-0226 A -- the locked, report-only GPU check on real student attempts: the T-0223 check over a private,
sha-pinned manifest of phone-scanned pages. One typed, neutral question per page on a fresh tools-off ``Session``
(search off, 1120 visual tokens, the T-0225 "Education tutor" built-in set through the Settings path), the page shown
as one shared-screen frame. ``/launch`` ONLY.

    uv run python scripts/gpu/t0226_real_attempts.py --manifest <the private manifest> \
        --manifest-sha256 <64 lowercase hex> --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing.**

Reuse, not copy (docs/tasks/T-0226.md, implementation notes): T-0223 B's harness
(``scripts/gpu/t0223_handwritten_check.py``) is loaded read-only by path under a private module name
(``T0223_MODULE_NAME``) as ``C``; on that instance only, and once at import, ``TUTOR_INSTRUCTIONS``, ``QUESTION`` and
``page_path`` are replaced. T-0223 loaded under its own name is untouched. Through ``C``: ``_one_turn``, ``evidence``,
``body_evidence``, ``status``, ``spoken_texts``, ``shown_text``, ``clip_kinds``, ``first_word_ms``, ``finish_turn``
(always with ``attempts={}``, so T-0223's judge never reads ``attempts.json``), ``frame_jpeg``, ``probe_ps``,
``_probe_generate``, ``not_run`` and T-0111g's / T-0131's helpers (``C.G``, ``C.T``). The refusals, the order, the
budget gate, P0, the runner loop, the kind-aware judge (A5), the thinking measures (A6) and the report are this file's.

Privacy (binding): the manifest and its pages are read, never copied; nothing written names the manifest's path or a
PNG file name (the run's ``manifest.json`` holds the manifest's sha256 and its directory's basename only); stderr names
pages by id only; P-S runs on a synthetic white page in a temporary directory removed afterwards; frames reach
``<out>/data`` only through the Session's own storage.

Output (``--out-dir``, default ``logs/t0226-real-attempts-<sha7>/``): ``order.json`` (first), ``manifest.json``,
``probes.json`` (``P-S``, ``P0``), ``trials.jsonl``, ``turns.jsonl``, ``sessions/*.jsonl``, ``report.json``,
``report.md``.

Exit: 0 measured; 1 no engine, or incomplete; 2 refused (A2, before any socket); 3 void (P0 failed or an A4
violation).

Regime (state it beside every number): DiffusionGemma 26B-A4B NVFP4 (rev from ``services/versions.lock``), vLLM
0.29.0, profile lean, canvas 256, 48 denoising steps, thinking on, KV 4 GiB bf16, ``max_num_seqs`` 3, prefix caching
on, vllm.sh at 806da12, one boot, STT down; tools off, search off, 1120 visual tokens, HUD screen bucket, JPEG 85, one
frame per turn; typed turns, FakeStt/FakeTts; n = 1 per page, order seed 0, engine unseeded, fresh Session per turn,
concurrency 1, 90 s per turn, 20 min run budget from process start.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
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

#: A3: the run budget is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
LOGS_DIR: Path = REPO / "logs"
#: T-0223 B's harness, read-only (loaded by path under a private name, never edited, never mutated under its own)
T0223_PATH: Path = REPO / "scripts" / "gpu" / "t0223_handwritten_check.py"
T0223_MODULE_NAME = "_t0223_for_t0226"


def _load_t0223() -> ModuleType:
    """A fresh instance of T-0223's harness under ``T0223_MODULE_NAME`` (registered there so its dataclasses
    resolve). Importing it opens no socket."""
    spec = importlib.util.spec_from_file_location(T0223_MODULE_NAME, T0223_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0223_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0223_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0223_MODULE_NAME, None)
        raise
    return module


C: ModuleType = _load_t0223()
G: ModuleType = C.G
T: ModuleType = C.T

#: the tree is read through this name; the refusals and ``main`` look it up at call time
_git = G._git
frame_jpeg = C.frame_jpeg

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"

# --- A3's texts, verbatim ----------------------------------------------------------------------------------------
#: byte-equal to the ``education_tutor`` built-in in ``hud/src/settings/prompts.js`` (T-0229 A1; T-0225's is historical)
TUTOR_INSTRUCTIONS = (
    "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, "
    "point to it, explain why, and ask me a guiding question before giving the fix; then give the "
    "corrected step and the correct final answer. If everything is right, say so plainly and "
    "don't invent a problem. Speak in plain words, not symbols, and keep each reply short."
)
#: neutral: it names no mistake (controls exist); T-0229 A2
QUESTION = "Can you double-check my work?"

# --- A3: the run shape -------------------------------------------------------------------------------------------
ORDER_SEED = 0
REPEATS = 1
TURN_TIMEOUT_S = 90
RUN_BUDGET_S = 20 * 60
#: no turn starts unless ``TURN_TIMEOUT_S`` plus this still fit inside ``RUN_BUDGET_S`` (last start at 1065 s)
REPORT_MARGIN_S = 45
ERRORED_STOP = 3
VISUAL_TOKENS = 1120
KINDS = ("mistake", "correct", "unclear")
#: A2: the merge subjects (exact prefixes) read from ``git log --merges``; branch merges never match
REQUIRED_MERGES: tuple[tuple[str, str], ...] = (
    ("Merge T-0226-a:", "T-0226 group A, this harness"),
    ("Merge T-0223-b:", "T-0223 group B, the harness this one reuses"),
    ("Merge T-0213-a2a:", "T-0213 A2-apply, the settings set"),
    ("Merge T-0225:", "T-0225, the built-in instruction presets"),
)
PROBE_NAMES = ("P-S", "P0")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# --- A5: the judge's phrase lists, pre-registered (the task's examples, verbatim) ---------------------------------
#: reads as a claim despite a negation
NEGATED_CLAIM: tuple[str, ...] = (
    "not correct",
    "isn't correct",
    "is not correct",
    "not right",
    "isn't right",
    "not quite",
    "pas correct",
    "pas juste",
    "n'est pas",
)
#: a clause holding one of these says the work is correct (its CLAIM words are ignored)
NO_MISTAKE: tuple[str, ...] = (
    "no mistake",
    "no mistakes",
    "no error",
    "no errors",
    "don't see any",
    "do not see any",
    "didn't find any",
    "couldn't find any",
    "can't find any",
    "nothing wrong",
    "nothing is wrong",
    "nothing went wrong",
    "is correct",
    "are correct",
    "all correct",
    "looks correct",
    "looks right",
    "is right",
    "well done",
    "checks out",
    "aucune erreur",
    "pas d'erreur",
    "pas de faute",
    "c'est correct",
    "c'est juste",
    "tout est juste",
    "bravo",
    # R-4 (a)
    "don't see a",
    "do not see a",
    "didn't find a",
    "can't find a",
    "couldn't find a",
)
#: a clause holding one of these (and no NO_MISTAKE phrase) claims a mistake
CLAIM: tuple[str, ...] = (
    "mistake",
    "error",
    "wrong",
    "incorrect",
    "should be",
    "should have",
    "forgot",
    "missing",
    "instead of",
    "erreur",
    "faute",
    "faux",
    "fausse",
    "oublié",
    "au lieu de",
)
#: A5: the clause split (after casefolding and folding curly apostrophes)
#: R-4 (a): also ``,``, ``however``, ``mais`` and ``cependant``; the delimiter is captured, so a clause ending in
#: ``?`` (a question) is known
CLAUSE_SPLIT = re.compile(r"([.!?;,\n]|\bbut\b|\bhowever\b|\bmais\b|\bcependant\b)")

LOCATED, MISSED, SAID_CORRECT, CLAIMED, NO_VERDICT, ERRORED = (
    "located",
    "missed",
    "said_correct",
    "claimed_mistake",
    "no_verdict",
    "errored",
)

# --- notes stated in the report -----------------------------------------------------------------------------------
SCAN_NOTE = (
    "the pages are phone scans at their own sizes (scan_size per page); each is resized by Pillow to the HUD "
    "screen bucket (frame_size), whereas a real share is resized on a browser canvas, so the pixels differ slightly."
)


def in_force_note(tutor: str, question: str) -> str:
    """T-0229 A2: the note naming the instructions and question in force (the run's recorded regime values)."""
    return (
        'the user instructions are the built-in "Education tutor" text, sent verbatim through the settings '
        f'path: "{tutor}"; the typed question is "{question}".'
    )


RED_NOTES_NOTE = "phone scans may include a teacher's red notes that can hint at the answer; this is not controlled."
JUDGE_NOTE = (
    "judged text = every spoken text of the turn plus the shown card text, joined with spaces; thoughts and clips "
    "are never judged. Clauses: casefolded, curly apostrophes folded, split on . ! ? ; , newline and the words 'but', "
    "'however', 'mais', 'cependant' (R-4); a clause ending in ? is a question, neither correct nor a claim. A clause with a NEGATED_CLAIM phrase is a claim; else one with a NO_MISTAKE phrase is a correct clause (its "
    "CLAIM words ignored); else one with a CLAIM word is a claim. Mistake pages: located when every judge keyword "
    "entry hits (T-0223 R-2), else missed (said_correct when the turn says correct). Correct pages: claimed_mistake, "
    "else said_correct, else no_verdict. Unclear pages: flags only. Errored, timed-out, not-run and empty turns are "
    "errored and count in their kind's denominator. n = 1 per page."
)
THINKING_NOTE = (
    "thinking = the turn's llm.commit records with segment 'thought': characters and canvases from the session "
    "log; tokens from the engine's /tokenize on the joined thought text (no generation, posted after the turn; "
    "null when /tokenize failed, 0 without a thought). TurnRecord.thought_tokens is not used."
)
SCOPE_NOTE = (
    "report-only: real student pages from a private, sha-pinned manifest, n = 1 per page; no bar, no adoption "
    "rule. The answers are recorded verbatim and judged by hand in the hand audit below."
)
HAND_AUDIT_HEADING = "## Hand audit (main session)"

#: page id -> its PNG, filled once the manifest has passed the checks (the private instance's ``page_path``)
_PAGES: dict[str, Path] = {}
#: the last P-S record the refusals produced (``probes.json`` carries it)
LAST_PS: dict[str, Any] = {}


def _page_path(page_id: str) -> Path:
    """The private T-0223 instance's ``page_path``: the manifest's PNG for ``page_id``."""
    try:
        return _PAGES[str(page_id)]
    except KeyError:
        raise KeyError(f"page {page_id} is not in the checked manifest") from None


# the three replacements, on the private instance only, once
C.TUTOR_INSTRUCTIONS = TUTOR_INSTRUCTIONS
C.QUESTION = QUESTION
C.page_path = _page_path


def _say(msg: str) -> None:
    print(f"t0226_real_attempts: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# A3: the order and the budget (pure)
# =====================================================================================================


def page_order(ids: Sequence[str]) -> list[str]:
    """A3: the page ids sorted, then shuffled once by ``random.Random(ORDER_SEED)``; n = 1 each."""
    out = sorted(str(i) for i in ids)
    random.Random(ORDER_SEED).shuffle(out)
    return out


def budget_stop(elapsed_s: float) -> bool:
    """A3: ``True`` when no further turn may start ``elapsed_s`` seconds after process start: a turn starts only
    while ``TURN_TIMEOUT_S`` + ``REPORT_MARGIN_S`` still fit inside ``RUN_BUDGET_S`` (1065 s is the last start)."""
    return elapsed_s + TURN_TIMEOUT_S + REPORT_MARGIN_S > RUN_BUDGET_S


def default_out(sha: str) -> Path:
    return LOGS_DIR / f"t0226-real-attempts-{sha[:7]}"


# =====================================================================================================
# A5: the judge (pure)
# =====================================================================================================


def _fold(text: str) -> str:
    return text.casefold().replace("’", "'").replace("‘", "'")


def clause_flags(text: str) -> tuple[bool, bool]:
    """A5's clause rule: ``(claims_mistake, says_correct)``; ``says_correct`` needs a correct clause and no claim
    clause."""
    neg = [_fold(p) for p in NEGATED_CLAIM]
    nom = [_fold(p) for p in NO_MISTAKE]
    cla = [_fold(p) for p in CLAIM]
    claim = correct = False
    parts = CLAUSE_SPLIT.split(_fold(text))
    for i in range(0, len(parts), 2):
        clause = parts[i]
        if i + 1 < len(parts) and parts[i + 1] == "?":
            continue  # R-4 (a): a question is neither a correct clause nor a claim clause
        if any(p in clause for p in neg):
            claim = True
        elif any(p in clause for p in nom):
            correct = True
        elif any(p in clause for p in cla):
            claim = True
    return claim, correct and not claim


def _kind_verdict(
    kind: str, text: str, keywords: Sequence[str]
) -> tuple[str, bool, bool, list[str]]:
    """``(verdict, claims_mistake, says_correct, matched)`` of one text that ran."""
    claims, says = clause_flags(text)
    if kind == "mistake":
        folded = text.casefold()
        matched = [str(k) for k in keywords if C.keyword_hit(str(k), folded)]
        located = bool(keywords) and len(matched) == len(keywords)
        return (LOCATED if located else MISSED), claims, says, matched
    if kind == "correct":
        return (CLAIMED if claims else SAID_CORRECT if says else NO_VERDICT), claims, says, []
    return NO_VERDICT, claims, says, []


def judge_turn(
    kind: str,
    spoken: Sequence[str],
    shown: str | None,
    judge_keywords: Sequence[str],
    status: str = "ran",
) -> dict[str, Any]:
    """A5 for one page's turn: ``verdict`` and ``spoken_only_verdict`` (the same rule over the spoken text alone),
    ``claims_mistake``, ``says_correct``, ``said_correct`` (a missed mistake page that said correct) and ``matched``
    (the keyword entries the judged text satisfies, in the page's order). A turn whose status is not ``ran`` or
    whose judged text is empty is ``errored`` on both verdicts."""
    spoken_list = [str(s) for s in spoken if s]
    spoken_text = " ".join(spoken_list)
    judged = " ".join([*spoken_list, *([str(shown)] if shown else [])])
    if status != "ran" or not judged.strip():
        claims, says = clause_flags(judged)
        return {
            "verdict": ERRORED,
            "spoken_only_verdict": ERRORED,
            "claims_mistake": claims,
            "says_correct": says,
            "said_correct": False,
            "matched": [],
        }
    verdict, claims, says, matched = _kind_verdict(kind, judged, judge_keywords)
    spoken_verdict, _c, _s, _m = _kind_verdict(kind, spoken_text, judge_keywords)
    return {
        "verdict": verdict,
        "spoken_only_verdict": spoken_verdict,
        "claims_mistake": claims,
        "says_correct": says,
        "said_correct": kind == "mistake" and verdict == MISSED and says,
        "matched": matched,
    }


# =====================================================================================================
# A6: thinking
# =====================================================================================================


def thought_measures(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A6: ``chars`` (sum of ``len``) and ``canvases`` (count) of the turn's ``llm.commit`` records with
    ``segment == "thought"`` (0 without one, never null), and ``text``, their texts joined in ``seq`` order."""
    texts = [
        str((r.get("data") or {}).get("text") or "")
        for r in sorted(records, key=lambda r: int(r.get("seq") or 0))
        if r.get("kind") == "llm.commit" and (r.get("data") or {}).get("segment") == "thought"
    ]
    return {"chars": sum(len(t) for t in texts), "canvases": len(texts), "text": "".join(texts)}


def thought_tokens(base_url: str, text: str, model: str | None = None) -> int | None:
    """A6: the engine's ``/tokenize`` count of ``text`` (no generation, no special tokens added); 0 for an empty
    text (nothing posted); ``None`` when ``/tokenize`` fails."""
    if not text:
        return 0
    if model is None:
        from local_astra.conductor.llm_client import VllmEndpoint

        model = VllmEndpoint().model
    body = {"model": model, "prompt": text, "add_special_tokens": False}
    # R-4 (b): a short timeout, so the last turn's post still fits the wrapper's 21 min
    status_, payload = T._post_json(G._root(base_url) + "/tokenize", body, timeout_s=10)
    if status_ != 200 or not isinstance(payload, Mapping):
        return None
    count = payload.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    tokens = payload.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


# =====================================================================================================
# A2: refusals (exit 2), all before any socket
# =====================================================================================================


@dataclass
class _Checked:
    """What the refusals established: the manifest's pages (by id), the order, the out-dir and the HEAD sha."""

    manifest_path: Path
    manifest_sha256: str
    pages: dict[str, dict[str, Any]]
    order: list[str]
    out: Path
    sha: str


def _merged(subject: str) -> bool:
    """Whether a merge commit whose subject starts with ``subject`` is reachable from HEAD."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _in_work_tree(directory: Path) -> bool:
    """``git -C <dir> rev-parse --is-inside-work-tree`` answers ``true``."""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
    try:
        r = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        return False
    return r.returncode == 0 and r.stdout.strip() == "true"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_problem(raw: bytes) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """A2 item 10: version 1, a non-empty page list, every page an object with a unique non-empty string id and a
    kind among ``KINDS``."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return f"the manifest is malformed JSON ({type(exc).__name__})", {}
    if not isinstance(obj, Mapping):
        return "the manifest is malformed: not a JSON object", {}
    if obj.get("version") != 1:
        return f"the manifest's version is {obj.get('version')!r}, not 1", {}
    items = obj.get("pages")
    if not isinstance(items, list) or not items:
        return "the manifest has no pages", {}
    pages: dict[str, dict[str, Any]] = {}
    for n, item in enumerate(items):
        if not isinstance(item, Mapping):
            return f"the manifest is malformed: page entry {n} is not an object", {}
        pid = item.get("id")
        if not isinstance(pid, str) or not pid:
            return f"the manifest is malformed: page entry {n} has no id", {}
        if pid in pages:
            return f"the manifest is malformed: duplicate page id {pid}", {}
        if item.get("kind") not in KINDS:
            return (
                f"the manifest is malformed: page {pid} has kind {item.get('kind')!r}, not one of {KINDS}",
                {},
            )
        pages[pid] = dict(item)
    return None, pages


def _png_problems(directory: Path, pages: Mapping[str, Mapping[str, Any]]) -> str | None:
    """A2 items 11-14, each over every page before the next item."""
    resolved: dict[str, Path] = {}
    for pid, page in pages.items():  # item 11
        rel = page.get("png")
        if not isinstance(rel, str) or not rel:
            return f"page {pid}: png is not a relative path string"
        if os.path.isabs(rel) or rel.startswith(("/", "\\")):
            return f"page {pid}: png is an absolute path, not allowed"
        # R-4 (c): a '..' is judged by where it resolves (symlinks followed), not refused outright
        target = (directory / rel).resolve()
        if not _inside(target, directory):
            return f"page {pid}: png resolves outside the manifest directory"
        resolved[pid] = target
    for pid, page in pages.items():  # item 12
        target = resolved[pid]
        if not target.is_file():
            return f"page {pid}: its png is missing (not a regular file)"
        got = _sha256_bytes(target.read_bytes())
        if got != page.get("sha256"):
            return f"page {pid}: its png's sha256 differs from its manifest entry"
    for pid, page in pages.items():  # item 13
        if page.get("kind") != "mistake":
            continue
        kws = page.get("judge_keywords")
        if not isinstance(kws, list) or not kws or not all(isinstance(k, str) and k for k in kws):
            return f"page {pid}: a mistake page needs a non-empty judge_keywords list of non-empty strings"
        for k in kws:
            if any(not alt for alt in k.split(C.KEYWORD_ALTERNATIVES)):
                return f"page {pid}: judge_keywords entry {k!r} has an empty alternative"
    for pid, page in pages.items():  # item 14
        if page.get("kind") in ("correct", "unclear") and not isinstance(
            page.get("judge_keywords"), list
        ):
            return f"page {pid}: judge_keywords is not a list"
    _PAGES.clear()
    _PAGES.update(resolved)
    return None


def _white_page(size: tuple[int, int]) -> tuple[Path, Path]:
    """P-S's synthetic white page at ``size`` in a fresh temporary directory: ``(dir, png)``."""
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="t0226-ps-"))
    png = tmp / "white.png"
    Image.new("RGB", size, (255, 255, 255)).save(png, format="PNG")
    return tmp, png


def _scan_size(png: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(png) as im:
        return int(im.size[0]), int(im.size[1])


def probe_ps(size: tuple[int, int]) -> dict[str, Any]:
    """A2's last item: T-0223's CPU settings probe (``C.probe_ps``, the instance's tutor text) on a synthetic white
    page at ``size`` (never a private page); its temporary directory is removed afterwards."""
    tmp: Path | None = None
    try:
        tmp, png = _white_page(size)
        ps = dict(C.probe_ps(png))
    except Exception as exc:
        ps = {"passed": False, "error": repr(exc)}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    ps["page"] = f"synthetic white page {size[0]}x{size[1]}"
    return ps


def _refusal(args: argparse.Namespace) -> tuple[str | None, _Checked | None]:
    """A2: why the run is refused (exit 2), checked in the task's order, all before any socket; else what was
    checked."""
    LAST_PS.clear()
    _PAGES.clear()
    try:  # 1
        G.check_loopback(str(args.base_url))
    except ValueError as exc:
        return f"--base-url is not loopback: {exc}", None
    try:  # 2
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})", None
    if dirty:
        return f"the tree is dirty:\n{dirty}", None
    out = Path(args.out_dir) if args.out_dir is not None else default_out(sha)
    if out.exists() or out.is_symlink():  # 3
        return (
            f"{out} already exists; the run happens once per SHA (move it aside only with the human)",
            None,
        )
    logs = LOGS_DIR.resolve()  # 4
    full = Path(os.path.abspath(out)).resolve()
    if not _inside(full, logs) or full == logs:
        return f"the out-dir {out} is not under {LOGS_DIR}", None
    try:
        _git("check-ignore", "-q", str(full))
    except (OSError, subprocess.CalledProcessError):
        return f"the out-dir {out} is not ignored by git (git check-ignore)", None
    for subject, what in REQUIRED_MERGES:  # 5
        if not _merged(subject):
            return (
                f"no '{subject}' merge is reachable from HEAD ({what}; a branch merge does not count)",
                None,
            )
    want = str(args.manifest_sha256)
    if not SHA256_RE.fullmatch(want):  # 6
        return "--manifest-sha256 is malformed: it must be 64 lowercase hex characters", None
    given = Path(os.path.abspath(args.manifest))
    if not given.is_file():  # 7
        return "the manifest is not a regular file (missing, or a directory)", None
    raw = given.read_bytes()
    if _sha256_bytes(raw) != want:  # 8
        return "the manifest's sha256 differs from --manifest-sha256", None
    directory = given.resolve().parent  # 9
    for d, how in ((given.parent, "as given"), (directory, "with symlinks followed")):
        if _inside(d, REPO) or _inside(d, REPO.resolve()):
            return (
                f"the manifest directory is inside the repo ({how}); it must live outside it",
                None,
            )
    if _in_work_tree(directory):
        return "the manifest directory is inside a git work tree; it must live outside any", None
    problem, pages = _manifest_problem(raw)  # 10
    if problem is not None:
        return problem, None
    problem = _png_problems(directory, pages)  # 11-14
    if problem is not None:
        return problem, None
    from local_astra.conductor import prompt_builder

    why = prompt_builder.validate_instructions(TUTOR_INSTRUCTIONS)  # 15
    if why is not None:
        return f"the tutor instructions are refused by validate_instructions: {why}", None
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]  # 16
    budget = cli.grounding_of(lean, C._off_args(), C.MODE).visual_tokens
    kw_budget = C.session_keywords(lean).get("visual_tokens")
    if budget != VISUAL_TOKENS or kw_budget != VISUAL_TOKENS:
        return (
            f"the visual-token budget in force is {budget} (Session keyword {kw_budget}), not {VISUAL_TOKENS} "
            "(cli.grounding_of, tools off)",
            None,
        )
    run_order = page_order(list(pages))  # 17
    try:
        size = _scan_size(_PAGES[run_order[0]])
    except Exception as exc:
        LAST_PS.update({"passed": False, "error": f"the first page cannot be read: {exc!r}"})
        return f"the CPU settings probe P-S failed: {LAST_PS['error']}", None
    ps = probe_ps(size)
    LAST_PS.update(ps)
    if not ps.get("passed"):
        return f"the CPU settings probe P-S failed: {json.dumps(ps, default=str)}", None
    return None, _Checked(
        manifest_path=given,
        manifest_sha256=want,
        pages=pages,
        order=run_order,
        out=out,
        sha=sha,
    )


# =====================================================================================================
# P0 (engine)
# =====================================================================================================


def probe_p0(base_url: str, model: str, frames: Mapping[str, bytes], first: str) -> dict[str, Any]:
    """A3 P0: one text and one image generate at ``max_soft_tokens`` 1120 must answer 200 (the image is the first
    ordered page's frame), the engine healthy after. Also, for the report and never gating: each page frame's
    image-token count at 1120 (``/tokenize`` with and without it) beside ``policy.expected_soft_tokens``."""
    from PIL import Image

    from local_astra.perception.policy import expected_soft_tokens

    text_status = asyncio.run(C._probe_generate(base_url, None, "t0226-probe-p0-text"))
    image_status = asyncio.run(C._probe_generate(base_url, frames[first], "t0226-probe-p0-image"))
    healthy = bool(T._healthy(base_url))
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
# The turn runner
# =====================================================================================================


@dataclass
class _Loop:
    out: Path
    base_url: str
    model: str
    order: list[str]
    run: Any
    manifest: dict[str, Any]
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


def _finish(out: Path, pid: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """T-0223's ``finish_turn`` with no attempts (its own judge sees no page; A5's judge is this file's)."""
    return dict(C.finish_turn(out, pid, 0, raw, attempts={}))


async def _run_all(loop: _Loop) -> None:
    """Every page of the order, concurrency 1. Before each: ``ERRORED_STOP`` consecutive non-clean turns, or
    ``budget_stop`` (from ``PROCESS_START_S``, probes included), stop the run; every page not run is then written
    as not run (errored). After each executed turn, its thought text goes to ``/tokenize`` (no generation)."""
    errored = 0
    for index, pid in enumerate(loop.order):
        if loop.stopped is None:
            elapsed = time.monotonic() - PROCESS_START_S
            if errored >= ERRORED_STOP:
                loop.stopped = f"{errored} consecutive non-clean turns (at turn {index})"
            elif budget_stop(elapsed):
                loop.stopped = (
                    f"the run budget of {RUN_BUDGET_S} s cannot fit a {TURN_TIMEOUT_S} s turn and the "
                    f"{REPORT_MARGIN_S} s report after {elapsed:.0f} s (at turn {index})"
                )
            if loop.stopped is not None:
                loop.manifest["stopped"] = loop.stopped
                G._write_json(loop.out / "manifest.json", loop.manifest)
                _say(f"stopping: {loop.stopped}")
        if loop.stopped is not None:
            raw: dict[str, Any] = dict(C.not_run(pid, 0, loop.stopped))
        else:
            started = time.monotonic()
            try:
                raw = dict(await C._one_turn(loop.run, pid, 0))
            except Exception as exc:
                raw = dict(C.not_run(pid, 0, f"harness: {exc!r}"))
            if raw.get("session_log"):
                records = C._session_records(loop.out, raw.get("session_log")) or []
                text = thought_measures(records)["text"]
                try:
                    raw["thought_tokens"] = thought_tokens(loop.base_url, text, loop.model)
                except Exception as exc:
                    _say(f"{pid}: /tokenize of the thought text raised {exc!r}")
                    raw["thought_tokens"] = None
            loop.turn_s.append(round(time.monotonic() - started, 1))
        trial = _finish(loop.out, pid, raw)
        trial["index"] = index
        errored = errored + 1 if trial["status"] != "ran" else 0
        G._append_jsonl(loop.out / "trials.jsonl", [trial])
        if raw.get("session_log"):
            _say(
                f"{index:02d} {pid} status={trial['status']} first_word_ms={trial['first_word_ms']} "
                f"violations={len(trial['violations'])} body_violations={len(trial['body_violations'])} "
                f"thought_tokens={raw.get('thought_tokens')} err={trial.get('exception')} "
                f"wall={loop.turn_s[-1] if loop.turn_s else None}s"
            )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    why, checked = _refusal(args)
    if why is not None or checked is None:
        _say(f"refusing: {why}")
        return EXIT_REFUSED
    ps = dict(LAST_PS)

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint
    from local_astra.config import load

    model = VllmEndpoint().model
    listing = G._get_text(str(args.base_url).rstrip("/") + "/models")
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
        return _measure(str(args.base_url), checked, model, lean, vram, ps)
    finally:
        vram.stop()


def _measure(
    base_url: str, checked: _Checked, model: str, lean: Any, vram: Any, ps: Mapping[str, Any]
) -> int:
    """Past every refusal and the engine check: the order, the manifest, the probes, the turns, the report."""
    out = checked.out
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", list(checked.order))
    frames: dict[str, bytes] = {}
    frame_sizes: dict[str, list[int]] = {}
    scan_sizes: dict[str, list[int]] = {}
    for pid in checked.order:
        jpeg, w, h, _ = C.frame_jpeg(_page_path(pid))
        frames[pid] = jpeg
        frame_sizes[pid] = [w, h]
        scan_sizes[pid] = list(_scan_size(_page_path(pid)))
    kinds = {k: sum(1 for p in checked.pages.values() if p.get("kind") == k) for k in KINDS}
    regime = {
        **C.P.regime_sources(),
        "stt_source": "scripts/locked/t0226_real_attempts.sh refuses while 127.0.0.1:8080/8081 are bound",
        "commit": checked.sha,
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
        "scan_sizes": scan_sizes,
        "frames_per_turn": 1,
        "user_instructions": TUTOR_INSTRUCTIONS,
        "user_instructions_source": "hud/src/settings/prompts.js education_tutor (T-0229)",
        "question": QUESTION,
        "pages_per_kind": kinds,
        "n": REPEATS,
        "turns": len(checked.order),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "report_margin_s": REPORT_MARGIN_S,
        "errored_stop": ERRORED_STOP,
        "manifest_sha256": checked.manifest_sha256,
        "input": "typed (UserText); FakeStt/FakeTts/fake audio, STT down; a fresh Session per turn (empty history "
        "and memory); hello -> astra.settings.set (user_instructions) -> echo rev 1 -> one astra.frame (screen) "
        "through HudServer.route into the session's BrowserFrameSource -> UserText; no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
    }
    manifest: dict[str, Any] = {
        "manifest_sha256": checked.manifest_sha256,
        "manifest_dir_name": checked.manifest_path.resolve().parent.name,
        "regime": regime,
        "session_keywords": {k: C.D._jsonable(v) for k, v in C.session_keywords(lean).items()},
        "frame_sha256": {pid: hashlib.sha256(j).hexdigest() for pid, j in frames.items()},
        "notes": [C.FRAME_NOTE, SCAN_NOTE, C.LATENCY_NOTE, JUDGE_NOTE, THINKING_NOTE],
        "stopped": None,
        "wall_s": None,
    }
    G._write_json(out / "manifest.json", manifest)
    (out / "trials.jsonl").touch()
    (out / "turns.jsonl").touch()

    probes: dict[str, Any] = {"P-S": dict(ps)}
    probe_start = time.monotonic()
    try:
        probes["P0"] = probe_p0(base_url, model, frames, checked.order[0])
    except Exception as exc:
        probes["P0"] = {"passed": False, "error": repr(exc)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES):
        manifest["stopped"] = "not run: a probe did not answer as required"
        for pid in checked.order:
            G._append_jsonl(
                out / "trials.jsonl", [_finish(out, pid, C.not_run(pid, 0, manifest["stopped"]))]
            )
        manifest["wall_s"] = round(time.monotonic() - PROCESS_START_S, 1)
        G._write_json(out / "manifest.json", manifest)
        report = write_report(out, checked.manifest_path)
        _say(f"P0 did not answer 200: the run is void; label {report['label']}")
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = C._Run(
        base_url=base_url, out=out, profile=lean, manifest=manifest, vram=vram, attempts={}
    )
    loop = _Loop(
        out=out,
        base_url=base_url,
        model=model,
        order=list(checked.order),
        run=run,
        manifest=manifest,
    )
    try:
        asyncio.run(_run_all(loop))
    except Exception as exc:
        loop.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {loop.stopped}")
        done = {str(t.get("page")) for t in G._read_jsonl(out / "trials.jsonl")}
        for pid in checked.order:
            if pid not in done:
                G._append_jsonl(
                    out / "trials.jsonl", [_finish(out, pid, C.not_run(pid, 0, loop.stopped))]
                )
    manifest["stopped"] = loop.stopped
    manifest["wall_s"] = round(time.monotonic() - PROCESS_START_S, 1)
    G._write_json(out / "manifest.json", manifest)

    report = write_report(out, checked.manifest_path)
    _say(f"label={report['label']}; {report['reading']}; written to {out}")
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if report["label"] == INCOMPLETE else EXIT_OK


# =====================================================================================================
# A7/A8: the report (from the run dir and the manifest)
# =====================================================================================================


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _oid(entry: Any) -> str:
    return str(entry[0]) if isinstance(entry, list | tuple) else str(entry)


def reading_line(counts: Mapping[str, Mapping[str, int]]) -> str:
    m, c, u = counts["mistake"], counts["correct"], counts["unclear"]
    return (
        f"{m['located']} of {m['n']} mistake pages located; {c['said_correct']} of {c['n']} correct pages said "
        f"correct, {c['claimed_mistake']} claimed a mistake; {u['n']} unclear pages reported"
    )


def build_report(out: Path, manifest_path: Path) -> dict[str, Any]:
    """``report.json`` from the run dir and the private manifest: every trial re-finished from its raw fields
    (T-0223's ``finish_turn``) and judged by A5's ``judge_turn``. ``label`` (A8: ``void`` when a probe is missing
    or failed or any executed turn has an A4 violation; ``incomplete`` when a page of the order has no trial or
    did not run cleanly, or the run stopped; else ``measured``). No bar."""
    out = Path(out)
    man_pages = {
        str(p["id"]): dict(p)
        for p in (json.loads(Path(manifest_path).read_bytes()).get("pages") or [])
    }
    run_order = [
        _oid(x) for x in (_read_json(out / "order.json", None) or page_order(list(man_pages)))
    ]
    manifest = _read_json(out / "manifest.json", {}) or {}
    probes = _read_json(out / "probes.json", {}) or {}
    p0_table = (probes.get("P0") or {}).get("image_tokens") or {}
    finished: dict[str, dict[str, Any]] = {}
    for raw in G._read_jsonl(out / "trials.jsonl"):
        pid = str(raw.get("page"))
        done = _finish(out, pid, raw)
        done["thought_tokens"] = raw.get("thought_tokens")
        finished[pid] = done
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    body_violations: list[str] = []
    latencies: list[int] = []
    warm: list[int] = []
    thinking_chars: list[int] = []
    thinking_canvases: list[int] = []
    thinking_tokens: list[int] = []
    cold_ms: int | None = None
    incomplete = bool(manifest.get("stopped"))
    statuses: dict[str, int] = {}
    for index, pid in enumerate(run_order):
        page = man_pages.get(pid) or {}
        kind = str(page.get("kind"))
        t = finished.get(pid)
        st = "missing" if t is None else str(t["status"])
        statuses[st] = statuses.get(st, 0) + 1
        if st != "ran":
            incomplete = True
        records = (C._session_records(out, t.get("session_log")) or []) if t is not None else []
        spoken = list(t["spoken"]) if t is not None else []
        shown = str(t["shown"]) if t is not None else ""
        verdict = judge_turn(kind, spoken, shown, list(page.get("judge_keywords") or []), status=st)
        think = thought_measures(records)
        ms = t["first_word_ms"] if t is not None else None
        line = (
            C.turn_line(out, t.get("session_log"), records)
            if t is not None and t.get("session_log")
            else None
        )
        extras = (line or {}).get("extras") or {}
        p0_row = p0_table.get(pid) or {}
        if t is not None:
            violations += list(t["violations"])
            body_violations += list(t["body_violations"])
        if st == "ran":
            thinking_chars.append(int(think["chars"]))
            thinking_canvases.append(int(think["canvases"]))
            tok = t.get("thought_tokens") if t is not None else None
            if isinstance(tok, int):
                thinking_tokens.append(tok)
            if isinstance(ms, int):
                latencies.append(ms)
                if index == 0:
                    cold_ms = ms
                else:
                    warm.append(ms)
        rows.append(
            {
                "id": pid,
                "index": index,
                "kind": kind,
                "status": st,
                **verdict,
                "spoken": spoken,
                "shown": shown,
                "clips": list(t["clips"]) if t is not None else [],
                "speech_refused": extras.get("speech_refused"),
                "thought_chars": int(think["chars"]),
                "thought_canvases": int(think["canvases"]),
                "thought_tokens": t.get("thought_tokens") if t is not None else None,
                "first_word_ms": ms,
                "wall_s": t.get("wall_s") if t is not None else None,
                "scan_size": ((manifest.get("regime") or {}).get("scan_sizes") or {}).get(pid),
                "frame_size": p0_row.get("size"),
                "image_tokens": p0_row.get("image_tokens"),
                "expected_soft_tokens": p0_row.get("expected_soft_tokens"),
                "llm_errors": int(t.get("llm_errors") or 0) if t is not None else 0,
                "vram_min_free_mib": t.get("vram_min_free_mib") if t is not None else None,
                "violations": (list(t["violations"]) + list(t["body_violations"]))
                if t is not None
                else [],
                "exception": t.get("exception") if t is not None else "no trial line",
            }
        )
    by_id = {r["id"]: r for r in rows}
    counts: dict[str, dict[str, int]] = {
        "mistake": {"n": 0, "located": 0, "missed": 0, "missed_said_correct": 0, "errored": 0},
        "correct": {"n": 0, "said_correct": 0, "claimed_mistake": 0, "no_verdict": 0, "errored": 0},
        "unclear": {"n": 0, "claims_mistake": 0, "says_correct": 0, "errored": 0},
    }
    for pid, page in man_pages.items():
        kind = str(page.get("kind"))
        if kind not in counts:
            continue
        row = by_id.get(pid) or {"verdict": ERRORED, "claims_mistake": False, "says_correct": False}
        c = counts[kind]
        c["n"] += 1
        if row["verdict"] == ERRORED:
            c["errored"] += 1
        elif kind == "mistake":
            c[LOCATED if row["verdict"] == LOCATED else MISSED] += 1
            c["missed_said_correct"] += int(bool(row.get("said_correct")))
        elif kind == "correct":
            c[str(row["verdict"])] += 1
        else:
            c["claims_mistake"] += int(bool(row["claims_mistake"]))
            c["says_correct"] += int(bool(row["says_correct"]))
    probes_ok = set(probes) >= set(PROBE_NAMES) and all(
        bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES
    )
    if not probes_ok or violations or body_violations:
        label = VOID
    elif incomplete:
        label = INCOMPLETE
    else:
        label = MEASURED
    vram = [
        int(t["vram_min_free_mib"])
        for t in finished.values()
        if t.get("vram_min_free_mib") is not None
    ]

    def p50(xs: list[int]) -> int | None:
        return statistics.median_low(xs) if xs else None

    return {
        "label": label,
        "stopped": manifest.get("stopped"),
        "probes": {n: bool((probes.get(n) or {}).get("passed")) for n in PROBE_NAMES},
        "violations": violations,
        "body_violations": body_violations,
        "statuses": statuses,
        "counts": counts,
        "reading": reading_line(counts),
        "pages": rows,
        "latency": {
            "p50_ms": p50(latencies),
            "max_ms": max(latencies) if latencies else None,
            "n": len(latencies),
            "cold_first_turn_ms": cold_ms,
            "warm_p50_ms": p50(warm),
            "warm_max_ms": max(warm) if warm else None,
            "warm_n": len(warm),
            "what": C.LATENCY_NOTE,
        },
        "thinking": {
            "p50_chars": p50(thinking_chars),
            "max_chars": max(thinking_chars) if thinking_chars else None,
            "p50_canvases": p50(thinking_canvases),
            "max_canvases": max(thinking_canvases) if thinking_canvases else None,
            "p50_tokens": p50(thinking_tokens),
            "max_tokens": max(thinking_tokens) if thinking_tokens else None,
            "n": len(thinking_chars),
            "n_tokens": len(thinking_tokens),
            "what": THINKING_NOTE,
        },
        "llm_errors": sum(int(t.get("llm_errors") or 0) for t in finished.values()),
        "http_non_200": sum(int(t.get("http_non_200") or 0) for t in finished.values()),
        "vram_min_free_mib": min(vram) if vram else None,
        "probe_vram_min_free_mib": manifest.get("probe_vram_min_free_mib"),
        "wall_s": manifest.get("wall_s"),
        "phrases": {
            "NEGATED_CLAIM": list(NEGATED_CLAIM),
            "NO_MISTAKE": list(NO_MISTAKE),
            "CLAIM": list(CLAIM),
        },
        "notes": {
            "scope": SCOPE_NOTE,
            "judge": JUDGE_NOTE,
            "frame": C.FRAME_NOTE,
            "scans": SCAN_NOTE,
            "in_force": in_force_note(
                str((manifest.get("regime") or {}).get("user_instructions", TUTOR_INSTRUCTIONS)),
                str((manifest.get("regime") or {}).get("question", QUESTION)),
            ),
            "red_notes": RED_NOTES_NOTE,
            "thinking": THINKING_NOTE,
            "latency": C.LATENCY_NOTE,
        },
        "regime": dict(manifest.get("regime") or {}),
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """``report.md``: the label, the reading line verbatim, the notes, the phrase lists verbatim, counts, latency,
    thinking, the per-page table, every page's spoken and shown text verbatim, the regime, and last the hand-audit
    placeholder (one line per page)."""
    lat = report.get("latency") or {}
    th = report.get("thinking") or {}
    notes = report.get("notes") or {}
    counts = report.get("counts") or {}
    lines = [
        "# T-0226 real student attempts: does Gearvis find the mistake? (report-only: no bar)",
        "",
        f"- label: **{report['label']}**",
        f"- stopped early: {_fmt(report.get('stopped'))}",
        "- probes: " + ", ".join(f"{k} {v}" for k, v in (report.get("probes") or {}).items()),
        f"- turn statuses: {_fmt(report.get('statuses'))}",
        f"- A4 violations: {_fmt(report.get('violations'))}",
        f"- body violations: {_fmt(report.get('body_violations'))}",
        f"- llm.error records: {report.get('llm_errors')}; non-200 generates: {report.get('http_non_200')}",
        f"- min free VRAM MiB over the turns: {_fmt(report.get('vram_min_free_mib'))} (probes: "
        f"{_fmt(report.get('probe_vram_min_free_mib'))})",
        f"- wall time s: {_fmt(report.get('wall_s'))}",
        "",
        f"**Reading: {report['reading']}**",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in notes.items()]
    lines += ["", "## Phrase lists (pre-registered, verbatim)", ""]
    for name, phrases in (report.get("phrases") or {}).items():
        lines.append(f"- {name}: " + " · ".join(f"`{p}`" for p in phrases))
    lines += ["", "## Counts", ""]
    for kind, c in counts.items():
        lines.append(f"- {kind}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
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
        "## Thinking",
        "",
        f"- {th.get('what')}",
        f"- turns that ran (n = {th.get('n')}): chars p50 {_fmt(th.get('p50_chars'))}, max "
        f"{_fmt(th.get('max_chars'))}; canvases p50 {_fmt(th.get('p50_canvases'))}, max "
        f"{_fmt(th.get('max_canvases'))}; tokens (n = {th.get('n_tokens')}) p50 {_fmt(th.get('p50_tokens'))}, "
        f"max {_fmt(th.get('max_tokens'))}",
        "",
        "## Per page",
        "",
        "| # | page | kind | status | verdict | spoken-only | claims | says correct | matched | thought chars / "
        "canvases / tokens | first word ms | speech refused | clips | scan | frame | image tokens / expected |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = list(report.get("pages") or [])
    for r in rows:
        lines.append(
            f"| {r['index']}{' (cold)' if r['index'] == 0 else ''} | {r['id']} | {r['kind']} | {r['status']} | "
            f"{r['verdict']}{' (said correct)' if r.get('said_correct') else ''} | {r['spoken_only_verdict']} | "
            f"{r['claims_mistake']} | {r['says_correct']} | {', '.join(r['matched']) or '-'} | "
            f"{r['thought_chars']} / {r['thought_canvases']} / {_fmt(r['thought_tokens'])} | "
            f"{_fmt(r['first_word_ms'])} | {_fmt(r['speech_refused'])} | {', '.join(r['clips']) or '-'} | "
            f"{_fmt(r['scan_size'])} | {_fmt(r['frame_size'])} | {_fmt(r['image_tokens'])} / "
            f"{_fmt(r['expected_soft_tokens'])} |"
        )
    lines.append("")
    lines += ["## Every page, verbatim", ""]
    for r in rows:
        lines += [
            f"### {r['id']} ({r['kind']}): {r['status']}, verdict {r['verdict']}, spoken-only "
            f"{r['spoken_only_verdict']}, violations {_fmt(r['violations'] or None)}, error {_fmt(r['exception'])}",
            "",
            "spoken:",
            "",
        ]
        lines += list(r["spoken"]) or ["(none)"]
        lines += ["", "shown:", ""]
        lines += list(str(r["shown"]).splitlines()) or ["(none)"]
        lines.append("")
    lines += [
        "## Regime",
        "",
        "```",
        json.dumps(report.get("regime") or {}, indent=1, default=str, ensure_ascii=False),
        "```",
        "",
        HAND_AUDIT_HEADING,
        "",
        "For each page: did the answer name the right line and fix (mistake pages); is a claimed mistake on a "
        "correct page real or invented.",
        "",
    ]
    lines += [f"- {r['id']} ({r['kind']}): " for r in rows]
    lines.append("")
    return "\n".join(lines)


def write_report(out: Path, manifest_path: Path) -> dict[str, Any]:
    """``report.json`` and ``report.md`` under ``out``, from the run dir and the manifest (``build_report``)."""
    out = Path(out)
    report = build_report(out, Path(manifest_path))
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    sys.exit(main())
