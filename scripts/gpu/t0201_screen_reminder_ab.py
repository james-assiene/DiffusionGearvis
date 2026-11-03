"""T-0201 group C -- the screen-reminder GPU A/B: does telling the model, last in its own turn, that it has the current
screenshot stop the "I don't have the current screenshot" thoughts, without costing valid steps or suite successes?
``/launch`` ONLY.

    uv run python scripts/gpu/t0201_screen_reminder_ab.py --base-url http://127.0.0.1:8001/v1 [--part 1|2|both]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the corpus pin (``P_CORPUS``, ``C_CORPUS``,
``prompt_sha``, ``corpus_findings``, ``eligible_requests``), ``no_see_v2``, the orders (``order``, ``order_part2``),
the arms (``base_prompt``, ``tip_prompt``, ``reminder_key``), the metrics (``part1_inputs``, ``part2_inputs``), the
decision (``decide``) and the report (``build_report``, ``render_markdown``) are pure; everything engine- or
browser-facing is imported inside ``main`` and the runners.

C1: imported, never copied. T-0138's harness (``scripts/gpu/t0138_suite_ab.py``, loaded read-only by path under
``T0138_MODULE_NAME``) and through it T-0134's (``F``: ``_client``, ``thought_record``), T-0132's (``P``:
``newcombe``), T-0131's (``T``) and T-0111g's (``G``: ``_git``, ``check_loopback``, the JSON writers, ``_wait_idle``).
Part 2 runs T-0138's own turn runner (``S._one_turn``) unchanged; the only seam is ``Session`` itself, which
``_one_turn`` looks up at call time: for one turn it is ``_ArmSession``, which passes HEAD's live settings T-0138
predates (the live guidance text, the page fact, the re-ask, the action outcome and repeat guard when on) and the
arm's ``screen_reminder`` over T-0138's keywords. B's helper (``local_astra.toolcall.loop.B_HELPER``) and B's reminder
table (``local_astra.conductor.session.B_TABLE``) are looked up when TIP is built; until group B merges TIP raises
``ScreenReminderUnavailable`` (and ``main`` refuses without ``Merge T-0201-b:`` anyway).

Refusals (C2; exit 2), all **before any socket** and before the out-dir is touched: a non-loopback ``--base-url``; a
dirty tree (named by its paths); an existing ``--out-dir``; no ``Merge T-0201-b:`` merge reachable from HEAD;
``Session`` without the ``screen_reminder`` keyword; a corpus log missing, a pinned seq missing or a pinned record's
``prompt_sha`` differing (each named); a corpus label no reminder key maps; a box that is not idle (a GPU compute
process other than the engine, 1-min load >= 2, port 8000 or 1420 bound); Chromium absent (only when Part 2 runs).
Then ``order.json`` (Part 1's order) and ``order_part2.json`` are written, and only then the first socket: with no
healthy engine (``engine_healthy``) the harness exits 1 having sent nothing.

Part 1 (C3, text-exact replay): every pinned record's logged prompt, every image replaced by one stand-in 1280x800
capture of the loopback ``/kettles.html`` (``capture_stand_in``, sha in the manifest; H-T201-6), sent as BASE (the
logged prompt) and TIP (B's helper with the reminder of the record's label key), n = ``N_P`` per arm per P item and
``N_C`` per arm per C item, ``order(ORDER_SEED)`` jointly shuffled, concurrency 1, the engine unseeded, default
sampling options, budget = HEAD's live visual tokens. A sample is a *no-see* when any of its commit segments matches
``no_see_v2``; *valid* when its tool commit parses (``native.parse_call``) or its answer is non-empty, and neither the
thought nor the answer trips T-0142's ``degenerate`` (answers at ``MIN_REPEATS_ANSWER``).

Part 2 (C4, live regression): the T-0138 suite S6-S15 through a real ``Session`` at HEAD's live default (read from
``cli`` with ``--no-web``: loopback suite pages), arms ``screen_reminder`` False/True, every other setting explicit and
equal, n = ``N_SUITE`` per task per arm, ``order_part2(ORDER_SEED)``. Per arm: successes (T-0138's ``succeeded``,
i.e. ``judge``), no-see steps with an image (``eligible_requests`` over the turn's session log), T-0138's safety
counters, loop ends.

Decision (C6, ``decide``): see its docstring; readings 1-6 of ``tests/unit/test_t0201_ab_harness.py`` are taken.
Engine errors in Part 1 make Part 1 INCONCLUSIVE (the arms would not be compared on equal samples).

Output (``--out-dir``, default ``logs/t0201-screen-reminder-ab-<sha7>/``): ``order.json``, ``order_part2.json``,
``manifest.json``, ``samples.jsonl`` (Part 1), ``trials.jsonl``/``steps.jsonl``/``nosee.jsonl``/``sessions/`` (Part
2), ``report.json``, ``report.md`` (every hit named, and an audit list: all hits plus 10 random non-hits per arm, seed
0). Exit codes: 0 measured; 1 no engine, a run that could not start or stopped early; 2 refused.

Regime (state it beside every number): profile lean; versions from ``services/versions.lock`` at HEAD; CL 256, 48
steps, thinking on, repair on; tools v1 ``bbfc53cc``; live guidance v3 (v2's text with ``--no-web``); HEAD's
``max_tool_steps`` (T-0138's turn policy, recorded); visual tokens 1120, ReGround off; headless Chromium 1280x800,
loopback; typed turns, fake STT/TTS/audio; engine unseeded; order and corpus seed 0; P 13 x 5 per arm, C 20 x 3 per
arm; Part 2 10 tasks x 3 per arm; clean tree, idle box.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
#: T-0138's harness, read-only (C1: imported, never copied); it imports T-0134's, T-0132's, T-0131's and T-0111g's
T0138_PATH: Path = REPO / "scripts" / "gpu" / "t0138_suite_ab.py"
T0138_MODULE_NAME = "_t0138_suite_ab_for_t0201"


def _load_t0138() -> ModuleType:
    """T-0138's harness, imported read-only by path under ``T0138_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0138_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0138_MODULE_NAME, T0138_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0138_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0138_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0138_MODULE_NAME, None)
        raise
    return module


S: ModuleType = _load_t0138()
#: T-0134's, T-0132's, T-0131's and T-0111g's harnesses, as T-0138 loaded them (the same module objects)
F: ModuleType = S.F
P: ModuleType = S.P
T: ModuleType = S.T
G: ModuleType = S.G

# --- imported, unchanged (the same objects) -------------------------------------------------------------------
newcombe = P.newcombe
wilson = P.wilson
thought_record = F.thought_record
BROWSER = S.BROWSER
VIEWPORT = S.VIEWPORT

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
ARMS = ("BASE", "TIP")
#: C4: the arm's ``Session(screen_reminder=...)``
REMINDER_ON = {"BASE": False, "TIP": True}
ORDER_SEED = 0
#: C3: per-arm repeats per P item and per C item
N_P = 5
N_C = 3
#: C4: per-arm repeats per suite task
N_SUITE = 3
#: C6: Part 1 is conclusive iff BASE's no-see count is at least this share of P samples (reading 4: inclusive)
CONCLUSIVE_SHARE = (1, 10)
#: C6: C's valid-step margin, in steps per 60 (reading 4)
C_VALID_MARGIN = 5
C_VALID_PER = 60
#: C6: C's no-see margin, in samples
C_NO_SEE_MARGIN = 2
#: C6: Part 2's success margin (of 30)
P2_SUCCESS_MARGIN = 3
#: the audit list: this many random non-hits per arm (seed ``ORDER_SEED``)
AUDIT_NON_HITS = 10
#: one Part 1 sample's bound (the request, end to end)
SAMPLE_TIMEOUT_S = 180.0
#: the stand-in page (H-T201-6)
STAND_IN_PAGE = "/kettles.html"
#: ports a session or the HUD would hold (C2: the box must be idle)
BUSY_PORTS = (8000, 1420)
LOAD_MAX = 2.0

#: group B's names, looked up when TIP is built (B owns them; this file never defines them)
B_HELPER = "with_screen_reminder"
B_TABLE = "SCREEN_REMINDERS"
B_SEPARATOR = "SCREEN_REMINDER_SEPARATOR"
B_MERGE = "Merge T-0201-b:"
B_KEYWORD = "screen_reminder"

#: the default of ``--logs-dir``
LOGS_DIR = REPO / "logs" / "sessions"

#: C3's P corpus (hand-checked by the test-author; ``tests/unit/test_t0201_ab_harness.py``'s docstring): every labelled
#: live ``tool_step`` with an image in the four logs whose own commits match ``no_see_v2``; sha = ``prompt_sha``
P_CORPUS: tuple[tuple[str, int, str], ...] = (
    (
        "20261027-230926-966483",
        8043,
        "047f639ab16744d6ccc4057c8e8da59993cef4e794b0c1084f5fffd34061a0c5",
    ),
    (
        "20261027-230926-966483",
        9377,
        "48c89284294b872a3a18109b0ccf19ce0020b1507f73362dfab60581c4b865b8",
    ),
    (
        "20261027-230926-966483",
        11655,
        "aab016fcfd23c20e5b1d9164b6196be25502dbf83a210a9eb64cf907244d58c8",
    ),
    (
        "20261028-145330-1824860",
        1055,
        "b9e45e0d7e7875cae58fb45fbc053f6bf6eced19ef36ac0ab6ba9b5afbdc7077",
    ),
    (
        "20261028-145330-1824860",
        6791,
        "099406e5538c3f90971a61dab619bceeafad72e78022d0de6f7bb61efd8ce632",
    ),
    (
        "20261029-173950-15317",
        5225,
        "004c257c0f70fb875eb6107b47c67e3f843d327b8bbd77981090ce6ffbe378d0",
    ),
    (
        "20261029-173950-15317",
        11592,
        "4bf86b537c9243ad7c5dec92ca29618c3ecd5d340044d3a3fa5de3ddfa8cf303",
    ),
    (
        "20261029-173950-15317",
        16202,
        "f8f3700dadf3ba7c6a83ff93b74ea96230125d4e936c4e0e09dd5cb8b4d6852a",
    ),
    (
        "20261029-181336-282350",
        3418,
        "81cc3105e57697fc9009a8bda59b2071c18b3be2bed44fa8388997096a69ec73",
    ),
    (
        "20261029-181336-282350",
        3957,
        "db3af900f7823bb58c1d5c0fd8612926acb8723c22e078a37a1570144e78e710",
    ),
    (
        "20261029-181336-282350",
        7383,
        "9a2c379014139e28c5ca1f2552163b9143a38a90226184a99060cd94ae6c6410",
    ),
    (
        "20261029-181336-282350",
        8945,
        "0e3b39b5cccf5e41487dd543e053d97ede76a2476962cbc29634c13f25ee4869",
    ),
    (
        "20261029-181336-282350",
        9083,
        "9777e99dcd5f283f06c59d6915b31edfd0767e43a3b5fc0f8e49410c41c9d6e0",
    ),
)
#: C3's C corpus: ``sorted(random.Random(0).sample(sorted(non-P (log, seq)), 20))`` over the 200 other eligible requests
C_CORPUS: tuple[tuple[str, int, str], ...] = (
    (
        "20261027-230926-966483",
        7949,
        "b2284111a3fc4df012aee4b5e963cdc4693bc0cbc71b6a1e679b80d6bcd67d96",
    ),
    (
        "20261027-230926-966483",
        11057,
        "e4e55ae1c9caa6e821bc762c901ac321a7b5da42fd0e5cd8690d6518b8b7a3d7",
    ),
    (
        "20261028-145330-1824860",
        682,
        "808cf0c929dfab1f053396d6386b9aeb73cf0809579eea73bcd04b7adc9254e0",
    ),
    (
        "20261028-145330-1824860",
        6991,
        "b3637e2cb138ba7f06798e3ca735e13d94b728c33a8a47e9631272b9675fe64b",
    ),
    (
        "20261028-145330-1824860",
        8503,
        "75d0c91b30b98af89a2c96c2dc277b545248da2aaf8c000dfbf19334698dd788",
    ),
    (
        "20261028-145330-1824860",
        8731,
        "57e1c2573b447ce3f02d044875db6a191567fa833594d85b3938c828ec21adc9",
    ),
    (
        "20261028-145330-1824860",
        9813,
        "ee0262368f5670a728a9cd81c7ae2753152eea2591b073413e517157330b3d1b",
    ),
    (
        "20261028-145330-1824860",
        10173,
        "b71c5bf218ca3fefceec80148c492361f04993a2d5f489af13d9e78c072eea1c",
    ),
    (
        "20261028-145330-1824860",
        11855,
        "e02dc1a4946adda4d8f15fc6304cb8cdd1d86d028fce5a172130b37e4ecea84b",
    ),
    (
        "20261028-145330-1824860",
        12581,
        "1c1c24eedf130fb562b23ce56f3eba596576bc179202d822a5d29a9bd83e0d80",
    ),
    (
        "20261028-145330-1824860",
        13518,
        "10735e5be282fc2e98a829a257dc234ce153e0ce7d0390bfec8f406de615d9cb",
    ),
    (
        "20261028-145330-1824860",
        14725,
        "ed4e7a7bfa48696e79ee47bd02e28ce36e6f45854e261f74259ec5314f9ee440",
    ),
    (
        "20261029-173950-15317",
        4528,
        "ac6d7fa584b4554eece34f599d3e8fc435499709105386ea93692be782900aaf",
    ),
    (
        "20261029-173950-15317",
        15728,
        "fc8eac2381dc1d1d3617e2cdd52010bf73bb0c933e28322265c59e8bfab084f9",
    ),
    (
        "20261029-181336-282350",
        330,
        "f03c3253d20f369a5071bef1317ac610ac41ee9c2d6239323992f66d02fde1db",
    ),
    (
        "20261029-181336-282350",
        419,
        "dd7700f4694dc9dca2599b81cfad3d911fade859a5919590479e86bca3a6f9f6",
    ),
    (
        "20261029-181336-282350",
        3616,
        "e7cc289921cb23f11fb7c6baaecc9facf033f0d30bded40b082fffb2afd98232",
    ),
    (
        "20261029-181336-282350",
        7560,
        "0ac179b58c9012df9c17e1ce422afb64e62fae160e07af5cd78e1f06da268a06",
    ),
    (
        "20261029-181336-282350",
        14718,
        "0089e289c9479f742e1e9714c727319ad8d4f9ab16dcdd0e5a976ef51ee2d98c",
    ),
    (
        "20261029-181336-282350",
        14835,
        "bee1cbd177a2e57ab5b075f63b7451c9de98803e5ef556dc96af52a459277db4",
    ),
)

HISTORY = (
    "T-0134 measured the earlier no-see proxy (``(don't|cannot) see the (updated|new) screenshot``) on S1-S5; "
    "``no_see_v2`` (C5) is wider (latest/current/new/any screenshot, the screen, the image) and is not comparable.",
)


def _say(msg: str) -> None:
    print(f"t0201_screen_reminder_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# C5: no_see_v2
# =====================================================================================================

_NEG = r"(?:don['’]?t|do\s+not|doesn['’]?t|does\s+not|cannot|can['’]?t|can\s+not)"
#: "I don't have the current screenshot", "I can't see the screen", "I don't see the latest screenshot": a negated
#: have/see whose object is the screen, a screenshot or an image (optionally latest/current/new/...); nothing else
_NO_SEE_OBJECT = re.compile(
    rf"\b{_NEG}\s+(?:have|see)\s+(?:(?:the|a|an|any)\s+)?(?:(?:latest|current|new|newest|updated|fresh|next)\s+)?"
    r"(?:screenshot|screen|image)s?\b",
    re.IGNORECASE,
)
#: "no screenshot", "no new screenshot"
_NO_SCREENSHOT = re.compile(
    r"\bno\s+(?:(?:latest|current|new|newest|updated|fresh)\s+)?screenshots?\b", re.IGNORECASE
)


def no_see_hit(text: str) -> str | None:
    """The first ``no_see_v2`` phrase in ``text``, or ``None``."""
    found = [m for m in (_NO_SEE_OBJECT.search(text), _NO_SCREENSHOT.search(text)) if m is not None]
    if not found:
        return None
    return min(found, key=lambda m: m.start()).group(0)


def no_see_v2(text: str) -> bool:
    """C5: the text claims the model has or sees no (current) screenshot, screen or image; objects are limited to
    screen/screenshot/image, so "I cannot see the price yet" and "I don't see a search box" do not match."""
    return no_see_hit(text) is not None


# =====================================================================================================
# C3: the corpus pin
# =====================================================================================================


def prompt_sha(prompt: Any) -> str:
    """Reading 1: sha256 of the canonical JSON of the logged ``data["prompt"]`` (sorted keys, compact, UTF-8)."""
    blob = json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_log(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _requests(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {
        int(r["seq"]): r
        for r in records
        if r.get("kind") == "llm.request" and isinstance(r.get("seq"), int)
    }


def corpus_findings(logs_dir: Path, corpus: Sequence[Sequence[Any]]) -> list[str]:
    """One finding per pinned ``(log, seq, sha)`` that is not there as pinned: a missing log, a missing seq (no
    ``llm.request`` at that seq) or a differing ``prompt_sha``; each names the log and the seq. Empty when all match."""
    findings: list[str] = []
    cache: dict[str, dict[int, Mapping[str, Any]] | None] = {}
    for log, seq, sha in corpus:
        log, seq = str(log), int(seq)
        if log not in cache:
            path = Path(logs_dir) / f"{log}.jsonl"
            try:
                cache[log] = _requests(_read_log(path)) if path.is_file() else None
            except (OSError, ValueError) as exc:
                findings.append(f"corpus log {log} (seq {seq}) is unreadable: {exc}")
                cache[log] = None
                continue
        reqs = cache[log]
        if reqs is None:
            findings.append(f"corpus log {log} is missing (seq {seq}) under {logs_dir}")
            continue
        record = reqs.get(seq)
        if record is None:
            findings.append(f"corpus log {log} has no llm.request at seq {seq}")
            continue
        got = prompt_sha((record.get("data") or {}).get("prompt"))
        if got != sha:
            findings.append(f"corpus log {log} seq {seq}: prompt sha {got} is not the pinned {sha}")
    return findings


def corpus_records(
    logs_dir: Path, corpus: Sequence[Sequence[Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """The pinned records' ``data``, by ``(log, seq)`` (call after ``corpus_findings`` is empty)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for log in sorted({str(x[0]) for x in corpus}):
        reqs = _requests(_read_log(Path(logs_dir) / f"{log}.jsonl"))
        for x in corpus:
            if str(x[0]) == log:
                out[(log, int(x[1]))] = dict(reqs[int(x[1])].get("data") or {})
    return out


def _image_count(prompt: Mapping[str, Any]) -> int:
    return sum(
        1
        for m in prompt.get("messages") or []
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, Mapping) and p.get("type") == "image_url"
    )


def eligible_requests(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """C3/C4's population over one session log: every ``llm.request`` with ``phase == "tool_step"``, a
    ``live_label`` with text and at least one image part; each with its own commits' text per segment (same
    ``attempt_id``, in ``seq`` order) and its first ``no_see_v2`` hit (``None`` when it has none)."""
    reqs: list[tuple[int, Mapping[str, Any]]] = []
    texts: dict[str, dict[str, str]] = {}
    for r in sorted(records, key=lambda r: int(r.get("seq") or 0)):
        data = r.get("data") or {}
        if r.get("kind") == "llm.request" and data.get("phase") == "tool_step":
            label = (data.get("live_label") or {}).get("text")
            if label and _image_count(data.get("prompt") or {}):
                reqs.append((int(r.get("seq") or 0), data))
        elif r.get("kind") == "llm.commit":
            seg = texts.setdefault(str(data.get("attempt_id")), {})
            name = str(data.get("segment"))
            seg[name] = seg.get(name, "") + str(data.get("text") or "")
    out: list[dict[str, Any]] = []
    for seq, data in reqs:
        segments = texts.get(str(data.get("attempt_id")), {})
        hit = next((h for h in (no_see_hit(t) for t in segments.values()) if h), None)
        out.append(
            {
                "seq": seq,
                "step": data.get("step"),
                "attempt_id": data.get("attempt_id"),
                "label": (data.get("live_label") or {}).get("text"),
                "hit": hit,
            }
        )
    return out


# =====================================================================================================
# C3: the arms
# =====================================================================================================


class ScreenReminderUnavailable(RuntimeError):
    """Group B's helper or reminder table is not there (T-0201 B not merged): TIP cannot be built."""


def reminder_key(label_text: str) -> str:
    """B3's key for a logged ``live_label`` text, through group B's own
    ``local_astra.conductor.session.screen_reminder_key`` (C1: imported, never copied; looked up at call time).
    ``ValueError`` for a label B maps to no key."""
    from local_astra.conductor import session as session_module

    key = session_module.screen_reminder_key(label_text)
    if key is None:
        raise ValueError(f"no screen-reminder key for the label {label_text!r}")
    return key


def _data_url(image: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


def base_prompt(logged_prompt: Mapping[str, Any], image: bytes) -> Any:
    """BASE (C3): the logged prompt as a ``Prompt``, every (redacted) image part's ``url`` the stand-in ``image``
    (its ``phash`` kept); every other byte as logged."""
    from local_astra.contracts.llm import Prompt

    tree = copy.deepcopy(dict(logged_prompt))
    url = _data_url(image)
    for message in tree.get("messages") or []:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["url"] = url
    return Prompt.model_validate(tree)


def _b_names() -> tuple[Any, Mapping[str, str]]:
    from local_astra.conductor import session as session_module
    from local_astra.toolcall import loop

    helper = getattr(loop, B_HELPER, None)
    table = getattr(session_module, B_TABLE, None)
    missing = [
        name
        for name, value in (
            (f"local_astra.toolcall.loop.{B_HELPER}", callable(helper)),
            (f"local_astra.toolcall.loop.{B_SEPARATOR}", hasattr(loop, B_SEPARATOR)),
            (f"local_astra.conductor.session.{B_TABLE}", isinstance(table, Mapping)),
        )
        if not value
    ]
    if missing:
        raise ScreenReminderUnavailable(
            f"T-0201 group B is not merged: {', '.join(missing)} missing; TIP cannot be built"
        )
    assert isinstance(table, Mapping)
    return helper, table


def tip_prompt(base: Any, key: str) -> Any:
    """TIP (C3): ``base`` through group B's append-only helper with B's reminder for ``key`` (B3). Raises
    ``ScreenReminderUnavailable`` until group B merges; ``KeyError`` for a key B's table does not have."""
    helper, table = _b_names()
    return helper(base, table[key])


# =====================================================================================================
# C3/C4: the orders
# =====================================================================================================


def order(seed: int) -> list[tuple[str, str, int, int]]:
    """Reading 2: arm-major over ``ARMS``, then ``P_CORPUS`` x ``range(N_P)`` then ``C_CORPUS`` x ``range(N_C)``,
    then one ``random.Random(seed).shuffle`` (jointly: the arms interleave). The corpora are read at call time."""
    turns = [
        (arm, str(log), int(seq), r)
        for arm in ARMS
        for items, n in ((P_CORPUS, N_P), (C_CORPUS, N_C))
        for (log, seq, _sha) in items
        for r in range(n)
    ]
    random.Random(seed).shuffle(turns)
    return turns


def order_part2(seed: int) -> list[tuple[str, str, int]]:
    """C4: every ``(arm, task, repeat)`` of ``ARMS`` x T-0138's tasks x ``range(N_SUITE)``, arm-major, one joint
    ``random.Random(seed).shuffle``."""
    turns = [(arm, t, r) for arm in ARMS for t in S.TASK_IDS for r in range(N_SUITE)]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# C6: metrics and the decision
# =====================================================================================================


def _tool_parses(text: str) -> bool:
    from local_astra.toolcall import native
    from local_astra.toolcall.score import ParseFailure

    body = text.strip()
    if body.startswith("<|tool_call>"):
        body = body[len("<|tool_call>") :]
    for candidate in (body, body + "<tool_call|>"):
        if not isinstance(native.parse_call(candidate), ParseFailure):
            return True
    return False


def sample_facts(commits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One Part 1 sample's facts from its commits (``segment``, ``text``): the text per segment, the first no-see
    hit, the T-0142 trips, and ``valid`` (a parseable tool call or a non-empty answer, and no trip)."""
    from local_astra.conductor.degeneracy import MIN_REPEATS_ANSWER, degenerate

    segments: dict[str, str] = {}
    for c in commits:
        name = str(c.get("segment"))
        segments[name] = segments.get(name, "") + str(c.get("text") or "")
    hit = next((h for h in (no_see_hit(t) for t in segments.values()) if h), None)
    thought_trip = degenerate(segments.get("thought", ""))
    answer_trip = degenerate(segments.get("answer", ""), min_repeats=MIN_REPEATS_ANSWER)
    tool = segments.get("tool")
    produced = (tool is not None and _tool_parses(tool)) or bool(segments.get("answer", "").strip())
    return {
        "segments": segments,
        "no_see": hit is not None,
        "hit": hit,
        "thought_trip": None if thought_trip is None else str(thought_trip),
        "answer_trip": None if answer_trip is None else str(answer_trip),
        "valid": produced and thought_trip is None and answer_trip is None,
    }


def part1_inputs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C6's Part 1 inputs from ``samples.jsonl``: ``n`` (per-arm P samples, the smaller arm's count over samples
    without an error), ``b``/``t`` (no-see counts on P), ``c_n``, ``c_valid`` and ``c_no_see`` per arm, and
    ``errors`` (samples with an engine error, per arm)."""
    p_keys = {(str(x[0]), int(x[1])) for x in P_CORPUS}
    ok = [s for s in samples if not s.get("error")]
    errors = {a: sum(1 for s in samples if s.get("arm") == a and s.get("error")) for a in ARMS}

    def on_p(s: Mapping[str, Any]) -> bool:
        return (str(s.get("log")), int(s.get("seq") or 0)) in p_keys

    p_n = {a: sum(1 for s in ok if s.get("arm") == a and on_p(s)) for a in ARMS}
    c_n = {a: sum(1 for s in ok if s.get("arm") == a and not on_p(s)) for a in ARMS}
    no_see = {
        a: sum(1 for s in ok if s.get("arm") == a and on_p(s) and s.get("no_see")) for a in ARMS
    }
    return {
        "n": min(p_n.values()),
        "b": no_see["BASE"],
        "t": no_see["TIP"],
        "c_n": min(c_n.values()),
        "c_valid": {
            a: sum(1 for s in ok if s.get("arm") == a and not on_p(s) and s.get("valid"))
            for a in ARMS
        },
        "c_no_see": {
            a: sum(1 for s in ok if s.get("arm") == a and not on_p(s) and s.get("no_see"))
            for a in ARMS
        },
        "errors": errors,
        "p_n": p_n,
    }


def reminder_requests(records: Sequence[Mapping[str, Any]]) -> int:
    """The positive control (reading (ii)): how many ``llm.request`` records of one session log carry
    ``screen_reminder`` (B8 logs it only when the reminder was placed)."""
    return sum(
        1
        for r in records
        if r.get("kind") == "llm.request" and (r.get("data") or {}).get(B_KEYWORD) is not None
    )


def part2_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    nosee: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """C6's Part 2 inputs: per arm the successes (T-0138's ``succeeded``: ``judge``, never on an engine error or a
    ``degenerate`` end), the no-see steps with an image, T-0138's safety counters, the loop ends, and the positive
    control ``reminder_requests``: the turns' ``llm.request`` records that logged ``screen_reminder`` (B8), summed
    from each trial's ``reminder_requests`` (``_run_part2`` counts them from the turn's session log)."""

    def mine(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
        return [r for r in rows if r.get("arm") == arm]

    def turn_steps(
        trial: Mapping[str, Any], arm_steps: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        return [
            s
            for s in arm_steps
            if s.get("task") == trial.get("task") and s.get("repeat") == trial.get("repeat")
        ]

    successes: dict[str, int] = {}
    safety: dict[str, dict[str, int]] = {}
    ends: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        tr, st = mine(trials, arm), mine(steps, arm)
        successes[arm] = sum(1 for t in tr if S.succeeded(t, turn_steps(t, st)))
        safety[arm] = dict(S.safety_counts(tr, st))
        counts: dict[str, int] = {}
        for t in tr:
            end = str(t.get("tool_loop_end"))
            counts[end] = counts.get(end, 0) + 1
        ends[arm] = counts
    return {
        "n": min(len(mine(trials, a)) for a in ARMS),
        "successes": successes,
        "no_see_steps": {a: sum(1 for r in mine(nosee, a) if r.get("hit")) for a in ARMS},
        "safety": safety,
        "loop_ends": ends,
        "reminder_requests": {
            a: sum(int(t.get("reminder_requests") or 0) for t in mine(trials, a)) for a in ARMS
        },
    }


def decide(part1: Mapping[str, Any] | None, part2: Mapping[str, Any] | None) -> dict[str, Any]:
    """C6. Part 1 (``None``: not run, INCONCLUSIVE): conclusive iff ``10 * b >= n`` (reading 4) and no engine
    error; passes iff ``2 * t <= b``, the 95 % Newcombe interval of ``b/n - t/n`` (T-0132's ``newcombe``, independent
    arms) excludes 0, C's TIP valid >= BASE valid - 5 per 60 and C's TIP no-see <= BASE + 2. Part 2 (``None``: not
    run, INCONCLUSIVE) passes iff TIP successes >= BASE - 3, TIP no-see steps <= BASE and every safety counter is 0 in
    both arms; whatever its bars say, Part 2 is INCONCLUSIVE unless its positive control holds (``reminder_requests``:
    TIP >= 1 and BASE == 0; reading (ii)). PASS iff both pass; FAIL iff Part 1 is conclusive and fails, or Part 2 fails; else INCONCLUSIVE."""
    interval: tuple[float, float] | None = None
    reasons: list[str] = []
    conclusive = False
    p1 = "INCONCLUSIVE"
    if part1 is None:
        reasons.append("Part 1 not run")
    else:
        n, b, t = int(part1["n"]), int(part1["b"]), int(part1["t"])
        if n > 0:
            interval = newcombe(b, n, t, n)
        errored = sum(int(v) for v in (part1.get("errors") or {}).values())
        num, den = CONCLUSIVE_SHARE
        if errored:
            reasons.append(f"Part 1 had {errored} engine errors")
        elif n <= 0 or den * b < num * n:
            reasons.append(f"Part 1 inconclusive: b = {b} of {n} is under 10 %")
        else:
            conclusive = True
            c_n = int(part1["c_n"])
            cv, cs = part1["c_valid"], part1["c_no_see"]
            checks = {
                "t <= b/2": 2 * t <= b,
                "Newcombe interval excludes 0": interval is not None
                and (interval[0] > 0 or interval[1] < 0),
                "C valid within 5/60": C_VALID_PER * int(cv["TIP"])
                >= C_VALID_PER * int(cv["BASE"]) - C_VALID_MARGIN * c_n,
                "C no-see within +2": int(cs["TIP"]) <= int(cs["BASE"]) + C_NO_SEE_MARGIN,
            }
            failed = [k for k, v in checks.items() if not v]
            p1 = "FAIL" if failed else "PASS"
            reasons += [f"Part 1 fails: {k}" for k in failed]
    p2 = "INCONCLUSIVE"
    if part2 is None:
        reasons.append("Part 2 not run")
    else:
        s, ns = part2["successes"], part2["no_see_steps"]
        unsafe = [
            f"{arm} {name} = {value}"
            for arm in ARMS
            for name, value in (part2["safety"].get(arm) or {}).items()
            if int(value) != 0
        ]
        checks2 = {
            "TIP successes >= BASE - 3": int(s["TIP"]) >= int(s["BASE"]) - P2_SUCCESS_MARGIN,
            "TIP no-see steps <= BASE": int(ns["TIP"]) <= int(ns["BASE"]),
            "every safety counter 0": not unsafe,
        }
        failed2 = [k for k, v in checks2.items() if not v]
        p2 = "FAIL" if failed2 else "PASS"
        reasons += [f"Part 2 fails: {k}" for k in failed2] + [f"unsafe: {u}" for u in unsafe]
        shown = part2.get("reminder_requests") or {}
        tip_shown, base_shown = int(shown.get("TIP", 0)), int(shown.get("BASE", 0))
        if not (tip_shown > 0 and base_shown == 0):
            p2 = "INCONCLUSIVE"
            reasons.append(
                f"Part 2 inconclusive: positive control failed (requests with screen_reminder: "
                f"TIP {tip_shown}, BASE {base_shown}; need TIP >= 1 and BASE 0)"
            )
    if p1 == "PASS" and p2 == "PASS":
        verdict = "PASS"
    elif (conclusive and p1 == "FAIL") or p2 == "FAIL":
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "part1": p1,
        "part2": p2,
        "conclusive": conclusive,
        "interval": None if interval is None else [interval[0], interval[1]],
        "reasons": reasons,
    }


# =====================================================================================================
# the report
# =====================================================================================================


def audit_list(
    rows: Sequence[Mapping[str, Any]], seed: int = ORDER_SEED
) -> dict[str, list[Mapping[str, Any]]]:
    """C6's hand audit: per arm every hit and ``AUDIT_NON_HITS`` random non-hits (``random.Random(seed)``)."""
    rng = random.Random(seed)
    out: dict[str, list[Mapping[str, Any]]] = {}
    for arm in ARMS:
        mine = [r for r in rows if r.get("arm") == arm]
        hits = [r for r in mine if r.get("hit")]
        non = [r for r in mine if not r.get("hit")]
        out[arm] = hits + rng.sample(non, min(AUDIT_NON_HITS, len(non)))
    return out


def build_report(
    part1: Mapping[str, Any] | None,
    part2: Mapping[str, Any] | None,
    samples: Sequence[Mapping[str, Any]],
    nosee: Sequence[Mapping[str, Any]],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: the decision, both parts' inputs, every hit named and the audit lists."""
    decision = decide(part1, part2)
    if stopped is not None and decision["verdict"] == "PASS":
        decision = {
            **decision,
            "verdict": "INCONCLUSIVE",
            "reasons": [*decision["reasons"], f"stopped: {stopped}"],
        }

    def brief(r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            k: r.get(k)
            for k in ("arm", "log", "seq", "repeat", "task", "step", "hit", "valid", "error")
        }

    return {
        "decision": decision,
        "stopped": stopped,
        "part1": None if part1 is None else dict(part1),
        "part2": None if part2 is None else dict(part2),
        "part1_hits": [brief(s) for s in samples if s.get("hit")],
        "part2_hits": [brief(r) for r in nosee if r.get("hit")],
        "part1_audit": {a: [brief(r) for r in v] for a, v in audit_list(samples).items()},
        "part2_audit": {a: [brief(r) for r in v] for a, v in audit_list(nosee).items()},
    }


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the decision and its reasons, both parts' numbers with n, every hit, the audit."""
    d = report["decision"]
    lines = ["# T-0201 C: screen-reminder A/B", ""]
    if manifest is not None:
        lines += [
            "## Regime",
            "",
            "```json",
            json.dumps(manifest.get("regime"), indent=1, sort_keys=True),
            "```",
            "",
        ]
    lines += [
        f"**Verdict: {d['verdict']}** (Part 1 {d['part1']}, Part 2 {d['part2']}; conclusive {d['conclusive']}; "
        f"Newcombe 95 % of b/n - t/n: {d['interval']})",
        "",
        *[f"- {r}" for r in d["reasons"]],
        "",
    ]
    p2 = report.get("part2")
    if p2 is not None:
        lines += [
            f"Part 2 positive control (llm.requests with screen_reminder, per arm): "
            f"{json.dumps(p2.get('reminder_requests'), sort_keys=True)}",
            "",
        ]
    for name in ("part1", "part2"):
        lines += [
            f"## {name}",
            "",
            "```json",
            json.dumps(report.get(name), indent=1, sort_keys=True),
            "```",
            "",
        ]
    for name in ("part1_hits", "part2_hits"):
        lines += [f"## {name} ({len(report[name])})", ""]
        lines += [f"- {json.dumps(h, sort_keys=True)}" for h in report[name]] + [""]
    for name in ("part1_audit", "part2_audit"):
        lines += [
            f"## {name} (all hits + {AUDIT_NON_HITS} random non-hits per arm; audit by hand)",
            "",
        ]
        for arm, rows in report[name].items():
            lines += (
                [f"### {arm}", ""] + [f"- [ ] {json.dumps(r, sort_keys=True)}" for r in rows] + [""]
            )
    return "\n".join(lines)


# =====================================================================================================
# seams (``main`` reads them at call time; the tests replace them)
# =====================================================================================================

#: the tree is read through this name (T-0111g's)
_git = G._git


def _nvidia_smi() -> str:
    """``nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`` (read-only)."""
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout


def _loadavg() -> tuple[float, float, float]:
    return os.getloadavg()


def _listening_ports() -> set[int]:
    """Every TCP port in LISTEN on this box, read from ``/proc/net/tcp{,6}`` (no socket)."""
    ports: set[int] = set()
    for name in ("tcp", "tcp6"):
        path = Path("/proc/net") / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="ascii", errors="replace").splitlines()[1:]:
            cols = line.split()
            if len(cols) > 3 and cols[3] == "0A":
                ports.add(int(cols[1].rsplit(":", 1)[1], 16))
    return ports


def _chromium_path() -> str | None:
    """The pinned Chromium the headless executor launches, or ``None`` (reads paths only)."""
    from local_astra import cli, preflight

    if cli.live_prerequisites(BROWSER) is not None:
        return None
    return preflight.pinned_executable(BROWSER)


def capture_stand_in() -> bytes:
    """H-T201-6: one 1280x800 frame of the loopback ``STAND_IN_PAGE`` through a fresh headless executor."""
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.pages.server import StartPageServer
    from local_astra.pages.suite_tasks import SUITE_DIR

    async def run() -> bytes:
        pages = StartPageServer(static_dir=SUITE_DIR)
        pages.start()
        ex = BrowserToolExecutor(
            now_ms=lambda: 0,
            viewport=VIEWPORT,
            guard=None,
            headless=True,
            end_when_closed=False,
            allowed_origins=frozenset({pages.origin}),
            echo="model",
        )
        try:
            await ex.open(pages.origin + STAND_IN_PAGE)
            frame = await ex.frame()
            if (frame.width, frame.height) != tuple(VIEWPORT):
                raise RuntimeError(
                    f"stand-in frame is {frame.width}x{frame.height}, not {VIEWPORT}"
                )
            return bytes(frame.jpeg)
        finally:
            with contextlib.suppress(Exception):
                await ex.close()
            pages.stop()

    return asyncio.run(run())


def engine_healthy(base_url: str) -> bool:
    """The first socket: the engine answers ``/health`` and serves the model (spends nothing)."""
    from local_astra.conductor.llm_client import VllmEndpoint

    if not T._healthy(base_url):
        return False
    listing = G._get_text(base_url.rstrip("/") + "/models")
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        return False
    return VllmEndpoint().model in served


async def send_request(client: Any, prompt: Any, request_id: str) -> list[dict[str, Any]]:
    """One generation request; its commits as ``segment``/``text``/``thought_open`` dicts, in order."""
    out: list[dict[str, Any]] = []
    async for c in client.generate(prompt, request_id):
        out.append({"segment": c.segment, "text": c.text, "thought_open": c.thought_open})
    return out


# =====================================================================================================
# Part 2's seam over T-0138's turn runner
# =====================================================================================================


def head_live_extras(profile: Any) -> dict[str, Any]:
    """C4: the ``Session`` keywords HEAD's live default passes that T-0138's turn runner predates, read from ``cli``
    for a loopback (``--no-web``) live session: the live guidance text, the page fact (v2/v3), the re-ask, and the
    action outcome and repeat guard when on."""
    from local_astra import cli

    ns = argparse.Namespace(no_web=True)
    version = cli.live_guidance_of(profile, ns, "live").value
    extras: dict[str, Any] = {
        "tool_guidance": cli.live_guidance_text(version, cli.web_of(profile, ns, "live")),
        "page_fact": version in ("v2", "v3"),
    }
    reask = cli.reask_of(profile, ns, "live").value
    if reask is not None:
        extras["reask_announced_calls"] = reask
    if cli.action_outcome_of(profile, ns, "live").action_outcome:
        extras["action_outcome"] = True
    if cli.repeat_guard_of(profile, ns, "live").repeat_guard:
        extras["repeat_guard"] = True
    return extras


@contextlib.contextmanager
def _arm_session(overrides: Mapping[str, Any]) -> Iterator[None]:
    """For one T-0138 turn: ``session.Session`` is a subclass that applies ``overrides`` over the keywords
    ``S._one_turn`` passes (concurrency 1, restored after)."""
    from local_astra.conductor import session as session_module

    real = session_module.Session

    class _ArmSession(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, **overrides})

    setattr(session_module, "Session", _ArmSession)  # noqa: B010
    try:
        yield
    finally:
        setattr(session_module, "Session", real)  # noqa: B010


# =====================================================================================================
# main
# =====================================================================================================


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--logs-dir", type=Path, default=None)
    ap.add_argument("--part", choices=("1", "2", "both"), default="both")
    return ap


def _merged(subject: str) -> bool:
    """Reading 5: a merge commit reachable from HEAD whose subject starts ``subject``."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


#: the engine's own process as nvidia-smi names it on the 5090 box (vLLM V1's engine core process title), beside
#: preflight's ``ENGINE_PROCESS_NAMES`` (``python3``/``python``); preflight.py is not C's to change (reading (i))
ENGINE_CORE_NAMES = ("VLLM::EngineCore",)


def _other_compute(text: str) -> list[str]:
    """``nvidia-smi --query-compute-apps`` rows whose executable is neither one of preflight's
    ``ENGINE_PROCESS_NAMES`` nor ``ENGINE_CORE_NAMES``."""
    from local_astra.preflight import ENGINE_PROCESS_NAMES

    engine = (*ENGINE_PROCESS_NAMES, *ENGINE_CORE_NAMES)
    others: list[str] = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 2 or not cols[0].isdigit():
            continue  # the header, or an empty line
        words = cols[1].split()
        if not words or Path(words[0]).name not in engine:
            others.append(line.strip())
    return others


def _refusal(args: argparse.Namespace) -> tuple[str | None, str, Path]:
    """C2's refusals in order; ``(reason, sha, out)``. Reads the tree, the logs and the box; opens no socket."""
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        return f"--base-url is not loopback: {exc}", "", Path()
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})", "", Path()
    if dirty:
        return f"the tree at {REPO} is dirty:\n{dirty}", sha, Path()
    out: Path = args.out_dir or REPO / "logs" / f"t0201-screen-reminder-ab-{sha[:7]}"
    if out.exists():
        return (
            f"{out} exists; the run happens once per SHA (move it aside only with the human)",
            sha,
            out,
        )
    if not _merged(B_MERGE):
        return f"no '{B_MERGE}' merge is reachable from HEAD (T-0201-b must merge first)", sha, out
    from local_astra.conductor import session as session_module

    if B_KEYWORD not in inspect.signature(session_module.Session.__init__).parameters:
        return f"Session has no {B_KEYWORD} keyword (T-0201 B1)", sha, out
    logs_dir: Path = args.logs_dir or LOGS_DIR
    corpus = (*P_CORPUS, *C_CORPUS)
    findings = corpus_findings(logs_dir, corpus)
    if findings:
        return "the pinned corpus is not there as pinned:\n" + "\n".join(findings), sha, out
    for (log, seq), data in corpus_records(logs_dir, corpus).items():
        label = (data.get("live_label") or {}).get("text")
        try:
            reminder_key(str(label))
        except ValueError as exc:
            return f"corpus record {log} seq {seq}: {exc}", sha, out
    try:
        smi = _nvidia_smi()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi could not be asked ({exc}); an idle box cannot be established", sha, out
    others = _other_compute(smi)
    if others:
        return "another GPU compute process is running: " + "; ".join(others), sha, out
    load1 = float(_loadavg()[0])
    if load1 >= LOAD_MAX:
        return f"the 1-min load is {load1:.2f} (must be under {LOAD_MAX})", sha, out
    busy = sorted(p for p in BUSY_PORTS if p in _listening_ports())
    if busy:
        return f"port(s) {busy} are bound (a session or the HUD is up)", sha, out
    if args.part in ("2", "both") and _chromium_path() is None:
        return f"Chromium ({BROWSER}) is not available; Part 2 needs it", sha, out
    return None, sha, out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    reason, sha, out = _refusal(args)
    if reason is not None:
        _say(f"refusing: {reason}")
        return EXIT_REFUSED
    # --- the order on disk before any socket --------------------------------------------------------------------
    out.mkdir(parents=True)
    G._write_json(out / "order.json", [list(t) for t in order(ORDER_SEED)])
    G._write_json(out / "order_part2.json", [list(t) for t in order_part2(ORDER_SEED)])
    # --- sockets from here on: loopback, unproxied ------------------------------------------------------------
    if not engine_healthy(args.base_url):
        _say(
            f"no healthy engine at {args.base_url}; nothing sent (the engine must already be up: start it before running this harness)"
        )
        return EXIT_FAILED
    return _measure(args, sha, out)


def _measure(args: argparse.Namespace, sha: str, out: Path) -> int:
    """Past every refusal and the engine check: the manifest, Part 1, Part 2, the report."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    default = S.live_default(lean)
    bad = S.live_default_findings(default)
    if bad:
        _say(f"cannot run: HEAD's live default {default} is not one T-0138's runner takes: {bad}")
        return EXIT_FAILED
    try:
        kit, detokenizer = cli.load_tool_kit(lean)
    except cli.ToolsUnavailable as exc:
        _say(f"cannot run: {exc}")
        return EXIT_FAILED
    if str(kit.tools_version) != S.V1_SHA256 or kit.repair_off is not None:
        _say(
            f"cannot run: tools {kit.tools_version} (want {S.V1_SHA256}), repair off: {kit.repair_off}"
        )
        return EXIT_FAILED
    extras = head_live_extras(lean)
    loop = cli.tool_loop_of(lean, argparse.Namespace())
    logs_dir: Path = args.logs_dir or LOGS_DIR
    stand_in = capture_stand_in() if args.part in ("1", "both") else b""
    manifest: dict[str, Any] = {
        "regime": {
            **P.regime_sources(),
            "commit": sha,
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "engine_version": G._get_text(G._root(args.base_url) + "/version"),
            "profile": lean.name,
            "canvas_length": lean.diffusion.canvas_length,
            "max_denoising_steps": lean.diffusion.max_denoising_steps,
            "thinking": lean.diffusion.thinking,
            "tools_sha256": str(kit.tools_version),
            "repair": "on",
            "live_default": dict(default),
            "head_live_extras": {
                k: (v if k != "tool_guidance" else _sha256_text(v)) for k, v in extras.items()
            },
            "max_tool_steps_head": loop.max_tool_steps,
            "max_tool_steps_t0138_policy": G.MAX_TOOL_STEPS,
            "part": args.part,
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "n_p": N_P,
            "n_c": N_C,
            "n_suite": N_SUITE,
            "stand_in_page": STAND_IN_PAGE,
            "stand_in_sha256": hashlib.sha256(stand_in).hexdigest() if stand_in else None,
            "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
            "history": list(HISTORY),
        },
        "p_corpus": [list(x) for x in P_CORPUS],
        "c_corpus": [list(x) for x in C_CORPUS],
        "logs_dir": str(logs_dir),
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    stopped: str | None = None
    part1: dict[str, Any] | None = None
    part2: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    nosee: list[dict[str, Any]] = []
    if args.part in ("1", "both"):
        (out / "samples.jsonl").touch()
        try:
            samples = asyncio.run(
                _run_part1(
                    args.base_url,
                    out,
                    logs_dir,
                    stand_in,
                    detokenizer,
                    int(default["visual_tokens"]),
                )
            )
            part1 = part1_inputs(samples)
        except Exception as exc:
            stopped = f"Part 1 raised: {exc!r}"
            _say(f"stopping: {stopped}")
    if stopped is None and args.part in ("2", "both"):
        try:
            trials, steps, nosee = _run_part2(args.base_url, out, lean, default, extras)
            part2 = part2_inputs(trials, steps, nosee)
        except Exception as exc:
            stopped = f"Part 2 raised: {exc!r}"
            _say(f"stopping: {stopped}")
    manifest["stopped"] = stopped
    G._write_json(out / "manifest.json", manifest)
    report = build_report(part1, part2, samples, nosee, stopped)
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    _say(f"verdict={report['decision']['verdict']} ({'; '.join(report['decision']['reasons'])})")
    _say(f"written to {out}")
    return EXIT_FAILED if stopped is not None else EXIT_OK


def _sha256_text(value: Any) -> Any:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else value


async def _run_part1(
    base_url: str, out: Path, logs_dir: Path, stand_in: bytes, detokenizer: Any, budget: int
) -> list[dict[str, Any]]:
    """C3: every turn of ``order(ORDER_SEED)``, one request each, concurrency 1; ``samples.jsonl`` appended per
    sample (the commits kept)."""
    from local_astra.conductor.llm_client import VllmEndpoint

    data = corpus_records(logs_dir, (*P_CORPUS, *C_CORPUS))
    client = F._client(VllmEndpoint(base_url=base_url), detokenizer, budget)
    samples: list[dict[str, Any]] = []
    try:
        for index, (arm, log, seq, repeat) in enumerate(order(ORDER_SEED)):
            record = data[(log, seq)]
            key = reminder_key(str((record.get("live_label") or {}).get("text")))
            base = base_prompt(record["prompt"], stand_in)
            prompt = base if arm == "BASE" else tip_prompt(base, key)
            idle = await G._wait_idle(base_url)
            started = time.monotonic()
            error: str | None = None
            commits: list[dict[str, Any]] = []
            try:
                commits = await asyncio.wait_for(
                    send_request(client, prompt, f"t0201-{index:03d}-{arm}-{seq}-r{repeat}"),
                    timeout=SAMPLE_TIMEOUT_S,
                )
            except Exception as exc:
                error = repr(exc)
            facts = sample_facts(commits)
            sample = {
                "index": index,
                "arm": arm,
                "log": log,
                "seq": seq,
                "repeat": repeat,
                "step": record.get("step"),
                "key": key,
                "corpus": "P" if (log, seq) in {(x[0], x[1]) for x in P_CORPUS} else "C",
                "continuation_chars": len(prompt.messages[-1].text()),
                "engine_idle_before": idle,
                "wall_s": round(time.monotonic() - started, 2),
                "error": error,
                **facts,
                "commits": commits,
            }
            samples.append(sample)
            G._append_jsonl(out / "samples.jsonl", [sample])
            _say(
                f"P1 {index:03d} {arm} {log}:{seq}/r{repeat} no_see={facts['no_see']} valid={facts['valid']} "
                f"err={error} wall={sample['wall_s']}s"
            )
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
    return samples


def _run_part2(
    base_url: str, out: Path, lean: Any, default: Mapping[str, Any], extras: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """C4: every turn of ``order_part2(ORDER_SEED)`` through T-0138's ``_one_turn`` (unchanged), ``Session`` the
    arm's (``_arm_session``); per turn the trial, its steps and its no-see rows (``eligible_requests`` over the
    turn's session log) appended."""
    profile = P.arm_profile(lean, int(default["visual_tokens"]), bool(default["reground"]), None)
    shared = {k: default[k] for k in S.DEFAULT_KEYS}
    rows = {
        arm: {
            "label": arm,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": bool(default["overview_frame"]),
            "tools_sha256": S.V1_SHA256,
            B_KEYWORD: REMINDER_ON[arm],
        }
        for arm in ARMS
    }
    (out / "sessions").mkdir(exist_ok=True)
    for name in ("trials.jsonl", "steps.jsonl", "nosee.jsonl", "turns.jsonl"):
        (out / name).touch()
    vram = T._VramSampler()
    vram.start()
    trials: list[dict[str, Any]] = []
    steps_all: list[dict[str, Any]] = []
    nosee: list[dict[str, Any]] = []
    try:
        run = S._Run(
            base_url=base_url,
            out=out,
            profile=profile,
            sentinel=G._sentinel_jpeg(),
            manifest={},
            vram=vram,
            arms=rows,
            crop=None,
        )
        errored = 0
        for index, (arm, task_id, repeat) in enumerate(order_part2(ORDER_SEED)):
            if errored >= G.ERRORED_STOP:
                raise RuntimeError(f"{errored} consecutive errored turns (at turn {index})")
            with _arm_session({**extras, B_KEYWORD: REMINDER_ON[arm]}):
                trial, steps, _regrounds, _thoughts, _line = asyncio.run(
                    S._one_turn(run, arm, task_id, repeat)
                )
            trial[B_KEYWORD] = REMINDER_ON[arm]
            steps = [{**s, "arm": arm, "task": task_id, "repeat": repeat} for s in steps]
            rows_ns: list[dict[str, Any]] = []
            shown = 0
            log_name = trial.get("session_log")
            if log_name:
                path = out / "sessions" / str(log_name)
                if path.is_file():
                    records = _read_log(path)
                    rows_ns = [
                        {**r, "arm": arm, "task": task_id, "repeat": repeat}
                        for r in eligible_requests(records)
                    ]
                    shown = reminder_requests(records)
            trial["reminder_requests"] = shown
            errored = errored + 1 if T._errored(trial, steps, []) else 0
            trials.append(trial)
            steps_all += steps
            nosee += rows_ns
            G._append_jsonl(out / "trials.jsonl", [trial])
            G._append_jsonl(out / "steps.jsonl", steps)
            G._append_jsonl(out / "nosee.jsonl", rows_ns)
            _say(
                f"P2 {index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} "
                f"no_see={sum(1 for r in rows_ns if r['hit'])} success={trial.get('success')} "
                f"end={trial.get('tool_loop_end')} err={trial.get('exception')}"
            )
    finally:
        vram.stop()
    return trials, steps_all, nosee


if __name__ == "__main__":
    sys.exit(main())
