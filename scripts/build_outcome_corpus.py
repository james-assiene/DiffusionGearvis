#!/usr/bin/env python3
"""T-0136 group A1: the outcome-label calibration corpus (docs/tasks/T-0136.md, A1-1..A1-8). CPU + headless browser.

    uv run python scripts/build_outcome_corpus.py <run-dir>... --out logs/t0136-outcome-corpus-<sha7> [--probes]
        [--tasks S1,S2,S3]
    uv run python scripts/build_outcome_corpus.py --compare <out-a> <out-b>

**What it does.** Each logged ``(run, arm, task, repeat)`` of a run dir's ``steps.jsonl`` (its executed steps, in
``step`` order; ``read_sequences``) is replayed in a fresh headless Chromium at 1280x800, DSF 1, through the real
``BrowserToolExecutor`` + ``PlaywrightDriver``, from the task's start page (T-0111g's ``TASKS`` and
``StartPageServer``). Every call refers to the executor's latest own frame, as the session's does; a call logged with
``frame_kind == "overview"`` is replayed through T-0150's capture (``page_overview``) and ``overview_frame_of``, as the
session does (T-0151 A2-2). A missing ``frame_kind`` (t0134) is a live frame (O-A1 (4)). The echo is the arm's, read
from the run dir's ``manifest.json`` (``regime.arms[].echo``), ``model`` when the manifest does not say (O-A1b (1)):
the echo changes the action string only, and t0134's arm A logged pixel echoes.

**Reproduction (A1-3).** A step is reproduced when the replay's action string equals the logged ``action`` and its
``cart_after`` (``window.__state.cart``, T-0151's read) equals the logged one when logged. A sequence stops at its first
divergence (``first_divergence``); its pairs before that are kept.

**Pairs (A1-4).** Every reproduced executed ``click_at``/``type_at`` gives one pair: ``before`` the executor's latest
own frame at dispatch (the session's ``turn.live_frame``), ``after`` the frame the action took after its settle
(``resolved["frame_after"]``), ``point`` the ``resolved`` viewport point, ``prev_point`` the previous executed pointer
point of the sequence (a scroll or key press neither sets nor clears it), the source ids, ``frame_kind``,
``scrolled_px`` and an oracle block for the record only (``window.__state.cart``, CDP layout metrics, the URL).
Frames are stored once each as ``frames/<sha256>.jpg``; distinct pairs are keyed by ``pair_id`` (byte-based, O-A1
(7)) with an ``occurrences`` count and the list of their ``sources``.

**Probes (A1-5, ``--probes``; A1c-3).** The tables of the probe pages ``PROBE_PAGES`` (``outcomes.html`` then
``outcomes_b.html`` under ``tests/fixtures/t0136/``), in that order (``probe_table``): each probe in a fresh browser on
a fresh load of ``origin + "/" + <page> + <query>`` (``probe_url``), one call in model units; the caret probe is the
click's after frame and a fresh frame ``wait_ms`` later. A probe with a ``pre`` first makes one executed
``click_at`` at ``pre`` after the load: its ``before`` is then the executor's latest frame after that click, its
``prev_point`` the pre click's resolved point, its oracle observed after it; a pre click not executed yields no pair
(a printed line names the probe). Probe records carry ``probe`` (the id) and ``probe_page`` (the file name).

**Frame groups (A1c-4).** ``frame_groups.json`` maps every frame's sha256 to its cluster id: two frames are
equivalent when they have the same size and a max full-resolution gray ``|Δ|`` <= 8 (the O-A1 (7) jitter rule),
clusters are the transitive closure, and a cluster's id is its lexicographically smallest sha256.

**Contact sheets (A1-6).** ``sheets/<pair_id>.png``: before and after, each with the click point, the previous point
and the near box (+-0.25 W x +-0.10 H) drawn, and ``|Δ|`` x4; panels at half scale, the ``|Δ|`` panel max-pooled so a
one-pixel change stays visible. For the labellers only.

**Output** (``--out``, exactly ``logs/t0136-outcome-corpus-<sha7>`` under this repository): ``manifest.json`` (the
counts ``sequences``, ``reproduced_steps``, ``truncated_sequences``, ``pairs``, per run dir and in total, and the
regime), ``pairs.jsonl``, ``divergences.jsonl`` (one line per truncated sequence), ``frame_groups.json``, ``frames/``,
``sheets/``.

**Determinism (A1-7 as ruled in O-A1 (7)).** ``--compare A B`` (read-only, no browser) prints ``compare_builds``:
whether the two builds have the same pair structure, the largest per-pixel gray ``|Δ|`` between matched frames and the
pixels over 24. One build per SHA: an existing ``--out`` is refused, so the second build of a determinism check runs
after the first is moved aside.

**Refusals (exit 2)**, all before any process or browser starts and before the out dir is made: a dirty tree; an out
dir that is not exactly ``logs/t0136-outcome-corpus-<sha7>`` (or already exists); a missing ``browser`` extra
(Playwright not importable); a run dir without ``steps.jsonl``, or whose executed steps log no
``executed_arguments`` (t0132, dropped by O-A1 (4)); with ``--probes``, a probe table holding a probe id twice (across
the pages or within one), a ``query`` that does not start with ``?``, or a ``pre`` without integer ``x``/``y``
(A1c-3). Exit 0 when the corpus was written, 1 when the browser could not
be started at all.

Loopback only (the start-page server binds 127.0.0.1; the executor's fence allows only its origin). No GPU, no
``artifacts/``. Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, n = 1 replay per sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
OUT_PREFIX = "t0136-outcome-corpus-"
T0111G_PATH = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
T0111G_MODULE_NAME = "_t0111g_live_smoke_for_t0136"
PROBE_PAGE = REPO / "tests" / "fixtures" / "t0136" / "outcomes.html"
PROBE_PAGE_B = REPO / "tests" / "fixtures" / "t0136" / "outcomes_b.html"
#: A1c-3: the probe pages in run order (read at call time)
PROBE_PAGES: tuple[Path, ...] = (PROBE_PAGE, PROBE_PAGE_B)
PAGES_DIR = REPO / "local_astra" / "pages" / "static"

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2

VIEWPORT = (1280, 800)
POINTER_TOOLS = ("click_at", "type_at")
#: A2-3's near box, definitional: +-0.25 W x +-0.10 H around the point (drawn on the sheets only)
NEAR_HALF_W_FRACTION = 0.25
NEAR_HALF_H_FRACTION = 0.10
#: O-A1 (7): the "no pixel over" level between two builds' frames
NOISE_OVER = 24
#: A1c-4 (the O-A1 (7) jitter rule): two same-size frames are equivalent at a max full-res gray |Δ| <= this
JITTER_MAX = 8
#: the bound of one read-only page read (T-0151's ``CART_READ_S``)
READ_S = 1.0
#: the bound of one replayed browser call (open, frame, execute): above the executor's own 8 s action bound
CALL_S = 20.0
STATE_JS = (
    "(() => window.__state === undefined ? null : JSON.parse(JSON.stringify(window.__state)))()"
)


def _say(msg: str) -> None:
    print(f"build_outcome_corpus: {msg}", file=sys.stderr, flush=True)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def _load_t0111g() -> ModuleType:
    """T-0111g's smoke harness, imported read-only by path (its ``TASKS``). Importing it opens no socket."""
    found = sys.modules.get(T0111G_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0111G_MODULE_NAME, T0111G_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0111G_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0111G_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0111G_MODULE_NAME, None)
        raise
    return module


# =====================================================================================================
# Pure parts: sequences, pair ids, the reproduction check
# =====================================================================================================

SeqKey = tuple[str, str | None, str, int]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def read_sequences(run_dir: Path) -> dict[SeqKey, list[dict[str, Any]]]:
    """A1-2: ``(run, arm, task, repeat) -> [record, ...]``, the logged executed step records (``status ==
    "executed"``) as logged, in ``step`` order; ``run`` is ``run_dir.name``, ``arm`` ``None`` when not logged. Keys
    are in sorted order."""
    run = Path(run_dir).name
    groups: dict[SeqKey, list[dict[str, Any]]] = {}
    for r in _read_jsonl(Path(run_dir) / "steps.jsonl"):
        if r.get("status") != "executed":
            continue
        arm = r.get("arm")
        key: SeqKey = (run, None if arm is None else str(arm), str(r["task"]), int(r["repeat"]))
        groups.setdefault(key, []).append(r)
    ordered = sorted(groups, key=lambda k: (k[0], k[1] or "", k[2], k[3]))
    return {k: sorted(groups[k], key=lambda r: int(r["step"])) for k in ordered}


def _point(p: Any) -> list[int] | None:
    return None if p is None else [int(p[0]), int(p[1])]


def pair_id(before_sha: str, after_sha: str, point: Any, prev_point: Any) -> str:
    """A1-4: ``sha256(before_sha, after_sha, point, prev_point)[:16]`` over canonical JSON (points as int lists, a
    missing ``prev_point`` as ``null``): the same for tuples and JSON lists, in every process."""
    blob = json.dumps(
        [str(before_sha), str(after_sha), _point(point), _point(prev_point)],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _reproduced(logged: Mapping[str, Any], replayed: Mapping[str, Any]) -> bool:
    if replayed.get("action") != logged.get("action"):
        return False
    return "cart_after" not in logged or replayed.get("cart_after") == logged["cart_after"]


def first_divergence(
    logged: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]]
) -> int | None:
    """A1-3: the index of the first logged step the replay did not reproduce (its action string differs, or its
    ``cart_after`` differs where one is logged, or the replay has no such step); ``None`` when all reproduced."""
    for i, step in enumerate(logged):
        if i >= len(replayed) or not _reproduced(step, replayed[i]):
            return i
    return None


# =====================================================================================================
# O-A1 (7): comparing two builds
# =====================================================================================================

_BYTE_FIELDS = ("pair_id", "before", "after")
_SOURCE_FIELDS = ("run", "arm", "task", "repeat", "step")


def _gray(data: bytes) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def _frame_file(out: Path, sha: str) -> Path:
    hits = sorted((out / "frames").glob(f"{sha}*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"frame {sha} is stored {len(hits)} times under {out / 'frames'}")
    return hits[0]


def _occurrences(out: Path) -> dict[str, list[tuple[str, str]]]:
    """Each pair record expanded to its source occurrences: canonical structure (every field but the byte-derived
    ones, ``occurrences`` and ``sources``; the source ids of that occurrence) -> its ``(before, after)`` shas."""
    found: dict[str, list[tuple[str, str]]] = {}
    for r in _read_jsonl(out / "pairs.jsonl"):
        base = {
            k: v
            for k, v in r.items()
            if k not in _BYTE_FIELDS and k not in ("occurrences", "sources")
        }
        sources = r.get("sources") or [[r.get(k) for k in _SOURCE_FIELDS]]
        for src in sources:
            item = dict(base) | dict(zip(_SOURCE_FIELDS, src, strict=True))
            key = json.dumps(item, sort_keys=True)
            found.setdefault(key, []).append((str(r["before"]), str(r["after"])))
    return found


def compare_builds(out_a: Path, out_b: Path) -> dict[str, Any]:
    """O-A1 (7): two out dirs' pairs matched on their structure (every ``pairs.jsonl`` field except ``pair_id``,
    ``before`` and ``after``, per source occurrence, so a dedupe that happened in one build only is not a structural
    difference). ``structure_equal``; ``max_abs_delta``, the largest per-pixel gray ``|Δ|`` over every matched pair's
    ``before`` and ``after``; ``pixels_over_24``, the pixels with ``|Δ| > 24``, summed; ``matched``, the matched
    occurrences."""
    import numpy as np

    occ_a, occ_b = _occurrences(Path(out_a)), _occurrences(Path(out_b))
    structure_equal = {k: len(v) for k, v in occ_a.items()} == {k: len(v) for k, v in occ_b.items()}
    cache: dict[tuple[str, str], tuple[int, int]] = {}
    worst, over, matched = 0, 0, 0
    for key in sorted(set(occ_a) & set(occ_b)):
        for (ba, aa), (bb, ab) in zip(sorted(occ_a[key]), sorted(occ_b[key]), strict=False):
            matched += 1
            for sa, sb in ((ba, bb), (aa, ab)):
                if (sa, sb) not in cache:
                    ga = _gray(_frame_file(Path(out_a), sa).read_bytes())
                    gb = _gray(_frame_file(Path(out_b), sb).read_bytes())
                    if ga.shape != gb.shape:
                        cache[(sa, sb)] = (255, int(max(ga.size, gb.size)))
                    else:
                        d = np.abs(ga - gb)
                        cache[(sa, sb)] = (int(d.max()), int((d > NOISE_OVER).sum()))
                m, n = cache[(sa, sb)]
                worst = max(worst, m)
                over += n
    return {
        "structure_equal": structure_equal,
        "max_abs_delta": worst,
        "pixels_over_24": over,
        "matched": matched,
        "only_in_a": sum(len(v) for k, v in occ_a.items() if k not in occ_b),
        "only_in_b": sum(len(v) for k, v in occ_b.items() if k not in occ_a),
    }


# =====================================================================================================
# A1c-4: frame groups (jitter clusters)
# =====================================================================================================


def _block_means(g: Any, k: int) -> Any:
    """Means of the ``k x k`` blocks of ``g`` (cropped to whole blocks; the global mean when no block fits). A mean
    over the same pixels of two frames differs by at most their max ``|Δ|``, so these only prefilter."""
    import numpy as np

    h, w = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    if h == 0 or w == 0:
        return np.asarray([float(g.mean())]) if g.size else np.zeros(1)
    return g[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def frame_groups(frames: Mapping[str, bytes]) -> dict[str, str]:
    """A1c-4: frame sha256 -> cluster id over the given encoded frames, decoded to full-resolution gray. Two frames
    are equivalent when they have the same size and a max per-pixel gray ``|Δ|`` <= ``JITTER_MAX``; clusters are the
    transitive closure; a cluster's id is its lexicographically smallest sha256. Independent of the input order.

    Exact: block means (global, 64 px, 8 px) only skip pairs whose max ``|Δ|`` is certainly over the bound; every
    merge is decided on the full frames."""
    import numpy as np

    keys = sorted(frames)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        root = k
        while parent[root] != root:
            root = parent[root]
        while parent[k] != root:
            parent[k], k = root, parent[k]
        return root

    shapes: dict[str, tuple[int, ...]] = {}
    coarse: dict[str, Any] = {}
    fine: dict[str, Any] = {}
    means: dict[str, float] = {}
    for k in keys:
        g = _gray(frames[k])
        shapes[k] = tuple(g.shape)
        fine[k] = _block_means(g, 8)
        coarse[k] = _block_means(g, 64)
        means[k] = float(g.mean()) if g.size else 0.0
    cache: dict[str, Any] = {}

    def full(k: str) -> Any:
        if k not in cache:
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[k] = _gray(frames[k])
        return cache[k]

    bound = JITTER_MAX + 1e-9
    by_shape: dict[tuple[int, ...], list[str]] = {}
    for k in keys:
        by_shape.setdefault(shapes[k], []).append(k)
    for _shape, group in sorted(by_shape.items()):
        ordered = sorted(group, key=lambda k: (means[k], k))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if means[b] - means[a] > bound:
                    break
                if find(a) == find(b):
                    continue
                if float(np.abs(coarse[a] - coarse[b]).max()) > bound:
                    continue
                if float(np.abs(fine[a] - fine[b]).max()) > bound:
                    continue
                if int(np.abs(full(a) - full(b)).max()) <= JITTER_MAX:
                    ra, rb = find(a), find(b)
                    parent[max(ra, rb)] = min(ra, rb)
    return {k: find(k) for k in keys}


# =====================================================================================================
# The corpus store
# =====================================================================================================


@dataclass
class _Corpus:
    out: Path
    pairs: dict[str, dict[str, Any]] = field(default_factory=dict)
    frames: dict[str, bytes] = field(default_factory=dict)

    def frame(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self.frames:
            self.frames[sha] = data
            path = self.out / "frames" / f"{sha}.jpg"
            if not path.exists():
                path.write_bytes(data)
        return sha

    def add(self, raw: Mapping[str, Any]) -> None:
        before, after = self.frame(raw["before_jpeg"]), self.frame(raw["after_jpeg"])
        point, prev = _point(raw["point"]), _point(raw["prev_point"])
        pid = pair_id(before, after, point, prev)
        source = [raw["run"], raw["arm"], raw["task"], raw["repeat"], raw["step"]]
        if pid in self.pairs:
            self.pairs[pid]["occurrences"] += 1
            self.pairs[pid]["sources"].append(source)
            return
        record: dict[str, Any] = {
            "pair_id": pid,
            "before": before,
            "after": after,
            "point": point,
            "prev_point": prev,
            "frame_kind": raw["frame_kind"],
            "scrolled_px": raw["scrolled_px"],
            "occurrences": 1,
            "run": raw["run"],
            "arm": raw["arm"],
            "task": raw["task"],
            "repeat": raw["repeat"],
            "step": raw["step"],
            "sources": [source],
            "oracle": dict(raw["oracle"]),
        }
        if raw.get("probe") is not None:
            record["probe"] = raw["probe"]
        if raw.get("probe_page") is not None:
            record["probe_page"] = raw["probe_page"]
        self.pairs[pid] = record


def _sheet(before: bytes, after: bytes, point: Any, prev_point: Any) -> bytes:
    """A1-6: one PNG, three panels (before with marks, after with marks, ``|Δ|`` x4), at half scale; the ``|Δ|``
    panel max-pooled 2x2 so that a one-pixel change stays visible."""
    import numpy as np
    from PIL import Image, ImageDraw

    def marked(data: bytes) -> Image.Image:
        with Image.open(io.BytesIO(data)) as im:
            img = im.convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        if point is not None:
            x, y = int(point[0]), int(point[1])
            hw, hh = round(NEAR_HALF_W_FRACTION * w), round(NEAR_HALF_H_FRACTION * h)
            draw.rectangle((x - hw, y - hh, x + hw, y + hh), outline=(255, 140, 0), width=2)
            draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=2)
            draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=2)
        if prev_point is not None:
            px, py = int(prev_point[0]), int(prev_point[1])
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(0, 90, 255), width=2)
        return img

    a, b = marked(before), marked(after)
    w, h = a.size
    ga, gb = _gray(before), _gray(after)
    if ga.shape == gb.shape:
        d = np.clip(np.abs(ga - gb) * 4, 0, 255).astype(np.uint8)
        hh, ww = (d.shape[0] // 2) * 2, (d.shape[1] // 2) * 2
        pooled = d[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).max(axis=(1, 3))
        delta = Image.fromarray(pooled, mode="L").convert("RGB")
    else:
        delta = Image.new("RGB", (w // 2, h // 2), (255, 0, 255))
    half = (max(1, w // 2), max(1, h // 2))
    panels = [a.resize(half, Image.Resampling.LANCZOS), b.resize(half, Image.Resampling.LANCZOS)]
    panels.append(delta.resize(half, Image.Resampling.NEAREST))
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 2 * 8, half[1]), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 8
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


# =====================================================================================================
# The browser replay
# =====================================================================================================


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _executor(origin: str, echo: str) -> Any:
    from local_astra.computer.executor import BrowserToolExecutor

    return BrowserToolExecutor(
        now_ms=_now_ms,
        viewport=VIEWPORT,
        guard=None,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({origin}),
        echo=echo,  # type: ignore[arg-type]
    )


async def _read_state(ex: Any) -> Any:
    try:
        return await asyncio.wait_for(ex.driver.page.evaluate(STATE_JS), timeout=READ_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        return None


def _cart(state: Any) -> list[str] | None:
    """T-0151's ``cart_after``: ``window.__state.cart`` as strings, ``None`` when the page has none."""
    if isinstance(state, Mapping) and isinstance(state.get("cart"), list):
        return [str(item) for item in state["cart"]]
    return None


async def _scroll_y(ex: Any) -> int | None:
    try:
        metrics = await asyncio.wait_for(ex.driver.layout_metrics(), timeout=READ_S)
        return round(float(metrics["scroll_y"]))
    except asyncio.CancelledError:
        raise
    except Exception:
        return None


def _url(ex: Any) -> str | None:
    try:
        return str(ex.driver.page.url)
    except Exception:
        return None


@dataclass
class _Observed:
    cart: list[str] | None
    scroll_y: int | None
    url: str | None


async def _observe(ex: Any) -> _Observed:
    return _Observed(_cart(await _read_state(ex)), await _scroll_y(ex), _url(ex))


def _oracle(before: _Observed, after: _Observed) -> dict[str, Any]:
    return {
        "cart_before": before.cart,
        "cart_after": after.cart,
        "scroll_y_before": before.scroll_y,
        "scroll_y_after": after.scroll_y,
        "url_changed": before.url != after.url,
    }


def _geometry(frame: Any) -> Any:
    from local_astra.contracts.perception import Source
    from local_astra.contracts.toolcall import FrameGeometry

    return FrameGeometry(
        phash=frame.phash, source=Source.WINDOW, width=frame.width, height=frame.height
    )


def _invocation(call_id: str, name: str, arguments: Mapping[str, Any], frame: Any) -> Any:
    from local_astra.contracts.toolcall import ToolInvocation

    return ToolInvocation(
        call_id=call_id,
        turn_id="t0136-replay",
        name=name,
        arguments=dict(arguments),
        source="draft",
        frame=frame,
    )


@dataclass
class _Replay:
    replayed: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


async def _step(
    ex: Any, i: int, record: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One logged executed step, re-executed: the replayed ``{"action", "cart_after"}`` and, for an executed
    pointer call, the raw pair (frames as bytes; ``prev_point`` filled in by the caller)."""
    name = str(record["executed_name"])
    args = dict(record.get("executed_arguments") or {})
    frame_kind = record.get("frame_kind")
    before_frame = ex.latest_frame
    seen = await _observe(ex)
    geometry = None
    if name in POINTER_TOOLS:
        if frame_kind == "overview":
            # T-0151 A2-2 as the session does it: T-0150's capture, then the executor registers the overview frame
            shot = await asyncio.wait_for(
                ex.execute(_invocation(f"replay-{i}-overview-frame", "page_overview", {}, None)),
                timeout=CALL_S,
            )
            key = shot.resolved.get("overview") if shot.status == "executed" else None
            registered = ex.overview_frame_of(key) if key is not None else None
            if registered is None:
                return {"action": f"overview frame not registered ({shot.action})"}, None
            geometry = _geometry(registered)
        elif before_frame is not None:
            geometry = _geometry(before_frame)
    out = await asyncio.wait_for(
        ex.execute(_invocation(f"replay-{i}", name, args, geometry)), timeout=CALL_S
    )
    after_seen = await _observe(ex)
    replayed = {"action": out.action, "cart_after": after_seen.cart}
    if out.status != "executed" or name not in POINTER_TOOLS or before_frame is None:
        return replayed, None
    after_frame = ex.latest_frame
    if after_frame is None or after_frame.phash != out.resolved.get("frame_after"):
        return replayed, None
    raw = {
        "before_jpeg": before_frame.jpeg,
        "after_jpeg": after_frame.jpeg,
        "point": (int(out.resolved["x_px"]), int(out.resolved["y_px"])),
        "frame_kind": frame_kind,
        "scrolled_px": out.resolved.get("scrolled_px"),
        "oracle": _oracle(seen, after_seen),
    }
    return replayed, raw


async def _replay_sequence(
    origin: str, start_page: str, echo: str, records: Sequence[Mapping[str, Any]]
) -> _Replay:
    """A1-2/A1-3: one sequence in a fresh browser from its start page; stops at the first step not reproduced."""
    result = _Replay()
    ex = _executor(origin, echo)
    try:
        await asyncio.wait_for(ex.open(origin + start_page), timeout=CALL_S)
        await asyncio.wait_for(ex.frame(), timeout=CALL_S)
        prev_point: tuple[int, int] | None = None
        for i, record in enumerate(records):
            replayed, raw = await _step(ex, i, record)
            result.replayed.append(replayed)
            if not _reproduced(record, replayed):
                break
            if raw is not None:
                raw["prev_point"] = prev_point
                raw["step"] = int(record["step"])
                result.pairs.append(raw)
                prev_point = raw["point"]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # the type only: a Playwright message carries call logs with URLs
        result.error = type(exc).__name__
    finally:
        try:
            await ex.close()
        except Exception as exc:
            result.error = result.error or f"close: {type(exc).__name__}"
    return result


def _table_of(page: Path) -> list[dict[str, Any]]:
    """A1-5: the probe table of one page (its ``<script type="application/json" id="probes">``)."""
    text = re.sub(r"<!--.*?-->", "", Path(page).read_text(encoding="utf-8"), flags=re.S)
    m = re.search(r'<script type="application/json" id="probes">(.*?)</script>', text, flags=re.S)
    if m is None:
        raise ValueError(f"no probe table in {page}")
    table = json.loads(m.group(1))
    if not isinstance(table, list) or not all(isinstance(p, Mapping) for p in table):
        raise ValueError(f"the probe table of {page} is not a list of objects")
    return [dict(p) for p in table]


def probe_table() -> list[dict[str, Any]]:
    """A1c-3: the run plan, every page's table in ``PROBE_PAGES`` order, each entry a copy of its table entry plus
    ``probe_page`` (the page's file name)."""
    return [p | {"probe_page": Path(page).name} for page in PROBE_PAGES for p in _table_of(page)]


def probe_url(origin: str, probe: Mapping[str, Any]) -> str:
    """A1c-3: ``origin + "/" + probe_page + query``; no query appends nothing (page a's URL is A1's)."""
    page = str(probe.get("probe_page") or PROBE_PAGE.name)
    return origin + "/" + page + str(probe.get("query") or "")


def probe_table_problems(table: Sequence[Mapping[str, Any]]) -> list[str]:
    """A1c-3's refusals, one line per problem, each naming the probe: a probe id held twice (across the pages or
    within one), a ``query`` that is not a string starting with ``?``, a ``pre`` that is not an object with integer
    ``x``/``y``."""
    problems: list[str] = []
    first_page: dict[Any, Any] = {}
    for p in table:
        pid, page = p.get("id"), p.get("probe_page")
        if pid in first_page:
            problems.append(
                f"probe {pid}: duplicated probe id ({first_page[pid]} and {page}); ids must be unique"
            )
        else:
            first_page[pid] = page
        if "query" in p:
            q = p["query"]
            if not (isinstance(q, str) and q.startswith("?")):
                problems.append(f"probe {pid}: its query {q!r} does not start with '?'")
        if "pre" in p:
            pre = p["pre"]
            if not (isinstance(pre, Mapping) and all(type(pre.get(k)) is int for k in ("x", "y"))):
                problems.append(f"probe {pid}: its pre {pre!r} has no integer x and y")
    return problems


async def _run_probe(origin: str, probe: Mapping[str, Any]) -> dict[str, Any] | None:
    """One probe: a fresh browser on a fresh load of its page and query, an optional ``pre`` click, then one call in
    model units (echo model); the caret probe's pair is the click's after frame and a fresh frame ``wait_ms``
    later."""
    ex = _executor(origin, "model")
    try:
        await asyncio.wait_for(ex.open(probe_url(origin, probe)), timeout=CALL_S)
        before_frame = await asyncio.wait_for(ex.frame(), timeout=CALL_S)
        prev_point: tuple[int, int] | None = None
        pre = probe.get("pre")
        if isinstance(pre, Mapping):
            pre_args = {"x": int(pre["x"]), "y": int(pre["y"])}
            pre_out = await asyncio.wait_for(
                ex.execute(
                    _invocation(
                        f"probe-{probe['id']}-pre", "click_at", pre_args, _geometry(before_frame)
                    )
                ),
                timeout=CALL_S,
            )
            if pre_out.status != "executed" or ex.latest_frame is None:
                _say(f"probe {probe['id']}: pre click {pre_out.status} ({pre_out.action}); no pair")
                return None
            prev_point = (int(pre_out.resolved["x_px"]), int(pre_out.resolved["y_px"]))
            before_frame = ex.latest_frame
        tool = str(probe["tool"])
        name = "click_at" if tool == "caret" else tool
        args: dict[str, Any] = {"x": int(probe["x"]), "y": int(probe["y"])}
        if name == "type_at":
            args |= {"text": str(probe["text"]), "press_enter": bool(probe["press_enter"])}
        seen = await _observe(ex)
        out = await asyncio.wait_for(
            ex.execute(_invocation(f"probe-{probe['id']}", name, args, _geometry(before_frame))),
            timeout=CALL_S,
        )
        if out.status != "executed":
            _say(f"probe {probe['id']}: {out.status} ({out.action})")
            return None
        after_frame = ex.latest_frame
        if tool == "caret":
            seen = await _observe(ex)
            before_frame = after_frame
            await asyncio.sleep(int(probe["wait_ms"]) / 1000)
            after_frame = await asyncio.wait_for(ex.frame(), timeout=CALL_S)
        after_seen = await _observe(ex)
        return {
            "before_jpeg": before_frame.jpeg,
            "after_jpeg": after_frame.jpeg,
            "point": (int(out.resolved["x_px"]), int(out.resolved["y_px"])),
            "prev_point": prev_point,
            "frame_kind": None,
            "scrolled_px": out.resolved.get("scrolled_px"),
            "oracle": _oracle(seen, after_seen),
            "run": "probes",
            "arm": None,
            "task": str(probe["id"]),
            "repeat": 0,
            "step": 1,
            "probe": str(probe["id"]),
            "probe_page": str(probe.get("probe_page") or PROBE_PAGE.name),
        }
    finally:
        await ex.close()


# =====================================================================================================
# The run
# =====================================================================================================


def _arm_echoes(run_dir: Path) -> dict[str, str]:
    """The echo of each arm as the run's manifest records it (``regime.arms[]``); empty when it does not."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    arms = (manifest.get("regime") or {}).get("arms") or manifest.get("arms") or []
    out: dict[str, str] = {}
    for arm in arms if isinstance(arms, list) else []:
        if isinstance(arm, Mapping) and arm.get("echo") in ("pixels", "model"):
            out[str(arm.get("label"))] = str(arm["echo"])
    return out


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@dataclass
class _Plan:
    run_dirs: list[Path]
    out: Path
    sha: str
    probes: bool
    tasks: frozenset[str] | None


async def _build(plan: _Plan) -> int:
    from local_astra.pages.server import StartPageServer

    g = _load_t0111g()
    starts = {t.task_id: t.start_page for t in g.TASKS}
    corpus = _Corpus(plan.out)
    runs: dict[str, dict[str, Any]] = {}
    divergences: list[dict[str, Any]] = []
    totals = {"sequences": 0, "executed_steps": 0, "reproduced_steps": 0, "truncated_sequences": 0}
    started = time.monotonic()
    pages = StartPageServer()
    pages.start()
    browser_ok = False
    try:
        for run_dir in plan.run_dirs:
            echoes = _arm_echoes(run_dir)
            stats = dict.fromkeys(totals, 0) | {"errors": 0}
            for key, records in read_sequences(run_dir).items():
                run, arm, task, repeat = key
                if plan.tasks is not None and task not in plan.tasks:
                    continue
                echo = echoes.get(str(arm), "model")
                result = await _replay_sequence(pages.origin, starts[task], echo, records)
                browser_ok = browser_ok or result.error is None or bool(result.replayed)
                cut = first_divergence(records, result.replayed)
                kept = len(records) if cut is None else cut
                stats["sequences"] += 1
                stats["executed_steps"] += len(records)
                stats["reproduced_steps"] += kept
                if cut is not None:
                    stats["truncated_sequences"] += 1
                    divergences.append(
                        {
                            "run": run,
                            "arm": arm,
                            "task": task,
                            "repeat": repeat,
                            "step": int(records[cut]["step"]),
                            "index": cut,
                            "logged": {
                                "action": records[cut].get("action"),
                                "cart_after": records[cut].get("cart_after", "not logged"),
                            },
                            "replayed": result.replayed[cut]
                            if cut < len(result.replayed)
                            else None,
                            "error": result.error,
                        }
                    )
                if result.error is not None:
                    stats["errors"] += 1
                for raw in (
                    result.pairs
                ):  # only reproduced steps carry a pair (the replay stops at a divergence)
                    corpus.add(raw | {"run": run, "arm": arm, "task": task, "repeat": repeat})
                _say(
                    f"{run} {arm}/{task}/r{repeat}: {kept}/{len(records)} reproduced"
                    + (f" (diverged at step {records[cut]['step']})" if cut is not None else "")
                    + (f" error={result.error}" if result.error else "")
                )
            stats["rate"] = (
                round(stats["reproduced_steps"] / stats["executed_steps"], 4)
                if stats["executed_steps"]
                else None
            )
            runs[run_dir.name] = stats
            for k in totals:
                totals[k] += stats[k]
    finally:
        pages.stop()
    probe_count = 0
    if plan.probes:
        fixture = StartPageServer(static_dir=PROBE_PAGE.parent)
        fixture.start()
        try:
            for probe in probe_table():
                raw = await _run_probe(fixture.origin, probe)
                browser_ok = True
                if raw is not None:
                    corpus.add(raw)
                    probe_count += 1
        finally:
            fixture.stop()
    rows = sorted(corpus.pairs.values(), key=lambda r: r["pair_id"])
    for r in rows:
        (plan.out / "sheets" / f"{r['pair_id']}.png").write_bytes(
            _sheet(
                corpus.frames[r["before"]], corpus.frames[r["after"]], r["point"], r["prev_point"]
            )
        )
    (plan.out / "pairs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    (plan.out / "divergences.jsonl").write_text(
        "".join(json.dumps(d, sort_keys=True) + "\n" for d in divergences), encoding="utf-8"
    )
    groups = frame_groups(corpus.frames)
    _write_json(plan.out / "frame_groups.json", groups)
    manifest = {
        "sequences": totals["sequences"],
        "executed_steps": totals["executed_steps"],
        "reproduced_steps": totals["reproduced_steps"],
        "truncated_sequences": totals["truncated_sequences"],
        "pairs": len(rows),
        "pair_occurrences": sum(r["occurrences"] for r in rows),
        "frames": len(corpus.frames),
        "frame_clusters": len(set(groups.values())),
        "probe_pairs": probe_count,
        "runs": runs,
        "regime": _regime(plan),
    }
    _write_json(plan.out / "manifest.json", manifest)
    _say(
        f"{totals['sequences']} sequences, {totals['reproduced_steps']}/{totals['executed_steps']} steps reproduced, "
        f"{totals['truncated_sequences']} truncated, {len(rows)} pairs, {time.monotonic() - started:.0f} s; "
        f"written to {plan.out}"
    )
    if not browser_ok and (totals["sequences"] > 0 or plan.probes):
        _say("the browser could not be started")
        return EXIT_FAILED
    return EXIT_OK


def _regime(plan: _Plan) -> dict[str, Any]:
    try:
        from importlib.metadata import version

        playwright: str | None = version("playwright")
    except Exception:
        playwright = None
    return {
        "commit": plan.sha,
        "inputs": {
            d.name: hashlib.sha256((d / "steps.jsonl").read_bytes()).hexdigest()
            for d in plan.run_dirs
        },
        "tasks": None if plan.tasks is None else sorted(plan.tasks),
        "probes": plan.probes,
        "probe_page_sha256": hashlib.sha256(PROBE_PAGE.read_bytes()).hexdigest()
        if plan.probes
        else None,
        "probe_pages_sha256": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in PROBE_PAGES
        }
        if plan.probes
        else None,
        "start_pages_sha256": _sha256_tree(PAGES_DIR),
        "browser": "Playwright Chromium, headless, a fresh browser per sequence and per probe",
        "viewport": list(VIEWPORT),
        "device_scale_factor": 1,
        "playwright": playwright,
        "echo": "the arm's, from the run's manifest (regime.arms[].echo); model when not recorded",
        "frame_kind_missing": "live (O-A1 (4))",
        "cart_check": "when logged (A1-3)",
    }


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# =====================================================================================================
# CLI
# =====================================================================================================


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="*", type=Path, help="logged A/B run dirs (steps.jsonl)")
    ap.add_argument("--out", type=Path, help="exactly logs/t0136-outcome-corpus-<sha7>")
    ap.add_argument(
        "--probes",
        action="store_true",
        help="add the probe pairs of outcomes.html, then outcomes_b.html",
    )
    ap.add_argument("--tasks", default=None, help="replay only these task ids (comma-separated)")
    ap.add_argument(
        "--compare", nargs=2, type=Path, metavar=("OUT_A", "OUT_B"), help="compare two builds"
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.compare is not None:
        a, b = args.compare
        for d in (a, b):
            if not (d / "pairs.jsonl").is_file():
                _say(f"refusing: {d} has no pairs.jsonl")
                return EXIT_REFUSED
        print(json.dumps(compare_builds(a, b), indent=1, sort_keys=True))
        return EXIT_OK
    if args.out is None:
        _say("refusing: --out logs/t0136-outcome-corpus-<sha7> is required")
        return EXIT_REFUSED
    # 1. a clean tree
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({type(exc).__name__})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty (commit or remove first):\n{dirty}")
        return EXIT_REFUSED
    # 2. the out dir: exactly logs/t0136-outcome-corpus-<sha7> under this repository, not there yet
    expected = LOGS.resolve() / f"{OUT_PREFIX}{sha[:7]}"
    out = Path(args.out).resolve()
    if out != expected:
        _say(
            f"refusing: --out must be exactly {expected} (logs/{OUT_PREFIX}<sha7> under the repository's logs/), "
            f"not {args.out}"
        )
        return EXIT_REFUSED
    if out.exists():
        _say(f"refusing: {out} exists; one build per SHA (move it aside to build again)")
        return EXIT_REFUSED
    # 3. the browser extra
    try:
        importlib.import_module("playwright.async_api")
    except ImportError:
        _say(
            "refusing: Playwright is not importable; install the browser extra (uv sync --extra browser)"
        )
        return EXIT_REFUSED
    # 4. the inputs
    for d in args.run_dirs:
        steps = Path(d) / "steps.jsonl"
        if not steps.is_file():
            _say(f"refusing: {d} has no steps.jsonl")
            return EXIT_REFUSED
        try:
            seqs = read_sequences(Path(d))
        except (ValueError, KeyError, TypeError) as exc:
            _say(f"refusing: {steps} is not readable step records ({type(exc).__name__}: {exc})")
            return EXIT_REFUSED
        if any("executed_arguments" not in r for v in seqs.values() for r in v):
            _say(
                f"refusing: {steps} logs executed steps without executed_arguments (t0132's shape, dropped by "
                "O-A1 (4))"
            )
            return EXIT_REFUSED
    # 5. the probe tables (A1c-3)
    if args.probes:
        try:
            problems = probe_table_problems(probe_table())
        except (OSError, ValueError) as exc:
            problems = [f"the probe pages are not readable ({type(exc).__name__}: {exc})"]
        if problems:
            _say("refusing: the probe table is invalid:\n" + "\n".join(problems))
            return EXIT_REFUSED
    tasks = (
        None
        if args.tasks is None
        else frozenset(t.strip() for t in str(args.tasks).split(",") if t.strip())
    )

    out.mkdir(parents=True)
    (out / "frames").mkdir()
    (out / "sheets").mkdir()
    plan = _Plan(
        run_dirs=[Path(d) for d in args.run_dirs],
        out=out,
        sha=sha,
        probes=bool(args.probes),
        tasks=tasks,
    )
    return asyncio.run(_build(plan))


if __name__ == "__main__":
    sys.exit(main())
