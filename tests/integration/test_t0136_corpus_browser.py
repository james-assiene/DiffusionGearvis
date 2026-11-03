"""T-0136 group A1 (integration + browser): the corpus builder ``scripts/build_outcome_corpus.py`` end to end through
the real ``BrowserToolExecutor`` + ``PlaywrightDriver`` and headless Chromium (docs/tasks/T-0136.md, A1-2..A1-7).

**Gate.** T-0140's/T-0151's browser modules' gate, imported (T-0111c's autouse ``_browser_layer``): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``. ``make browser`` does not list this file (the ``Makefile`` is not
A1's to own); run it as::

    ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0136_corpus_browser.py -q -m browser

The A1-3 criterion row over the real ``t0151-overview-frame-ab-4471255`` replays about 200 sequences (minutes); it
runs only with ``T0136_FULL_REPLAY=1`` as well, and only where that run dir exists (this worktree's ``logs/`` or the
main checkout's, next to the git common dir).

**Status.** Activated for group A1 (rulings O-A1; no ``pending`` markers). Until the builder exists each builder row
fails on a named assertion, never on an ImportError. The probe-page premise row is green at BASE_A1 (the page is the
test-author's). O-A1 (7): A1-7 determinism is pair structure plus frames within JPEG noise (max gray |Δ| <= 8, no
pixel over 24), with the builder's ``compare_builds`` recording the max |Δ|; O-A1 (4): t0132 dropped, t0134 read as
live with the cart check skipped.

**How a row runs the builder.** In process, ``main([<run-dir>..., "--out", <out>, ("--probes")])`` with the
builder's ``_git`` seam answering ``status`` with a clean tree and ``rev-parse`` with a fresh fake sha per run (any
other git call goes to the real repository with the fake sha replaced by ``HEAD``'s), so a row runs in a dirty
worktree and its out dir ``logs/t0136-outcome-corpus-<fake sha7>`` never names a real corpus. Each out dir is removed
after the row.

**The outputs these rows read** (the task names the contents, not the files; these are the names a row needs, and
the implementer must provide them):

=================================  ==================================================================================
``<out>/manifest.json``            integer counts ``sequences``, ``reproduced_steps``, ``truncated_sequences``,
                                   ``pairs`` (A1-3)
``<out>/pairs.jsonl``              one record per distinct pair (A1-4): ``pair_id``, ``before``/``after`` (the
                                   sha256 hex of the frame's bytes), ``point``, ``prev_point``, ``frame_kind``,
                                   ``scrolled_px``, ``occurrences``, ``oracle`` {``cart_before``, ``cart_after``,
                                   ``scroll_y_before``, ``scroll_y_after``, ``url_changed``} and the source ids
``<out>/frames/<sha256>[.ext]``    each frame once, its bytes hashing to its name (A1-4)
one PNG per distinct pair          anywhere under ``<out>``, its file name containing the ``pair_id`` (A1-6)
=================================  ==================================================================================

**Oracles, independent of the builder.** Hand-built ``steps.jsonl`` records copied from the real T-0151 run
(S2 arm B #11, the Goal's over-click: two ``click_at(423, 631)`` with carts ``["blue mug"]`` then two mugs; S3 arm B
#4; S1 arm A #0; S5 arm A #17, a scroll then a click; S5 arm B #1, a step-1 call on the overview frame). Pixel points
by hand from ``to_pixels(v, n) = round(v * (n - 1) / 999)``: ``(423, 631) -> (542, 505)`` (the logged ``click.point``
agrees), ``(155, 898) -> (198, 718)``; the overview click ``(159, 936)`` on the 1220 px deals page maps to page y
``round(936 * 1219 / 999) = 1142`` and, after the logged 420 px scroll, viewport ``(204, 722)`` (T-0151 A1-7's
mapping). The probe page's own table gives every probe's ``resolved`` point, class and target.

BASE_A1 = 6405e0d.

Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, loopback only, n = 1 per row (two builds for the
determinism row).
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.integration.test_t0140_overview_browser import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    execute,
    executor,
    inv,
    run,
)
from tests.unit.test_t0136_corpus import (
    PROBE_PAGE,
    PROBE_PAGE_B,
    S2_B11,
    S3_B4,
    TRIVIAL_KINDS,
    _box,
    _element,
    probe_table,
    probe_table_b,
    rec,
    write_run,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts" / "build_outcome_corpus.py"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "t0136"
T0151_RUN = "t0151-overview-frame-ab-4471255"
FULL_ENV = "T0136_FULL_REPLAY"
W, H = 1280, 800
ORACLE_KEYS = {"cart_before", "cart_after", "scroll_y_before", "scroll_y_after", "url_changed"}

C1 = "clicked at (423, 631)"
S1_A0 = [
    rec(
        "S1",
        "A",
        0,
        1,
        name="type_at",
        arguments={"press_enter": True, "text": "mugs", "x": 483, "y": 500},
        action="clicked at (483, 500) and typed 4 characters and pressed Enter",
        cart_after=None,
    ),
    rec(
        "S1",
        "A",
        0,
        2,
        name="click_at",
        arguments={"x": 126, "y": 376},
        action="clicked at (126, 376)",
        cart_after=[],
    ),
    rec(
        "S1",
        "A",
        0,
        3,
        name="click_at",
        arguments={"x": 425, "y": 630},
        action="clicked at (425, 630)",
        cart_after=["blue mug"],
    ),
    rec("S1", "A", 0, 4, status=None, cart_after=None),
]
S5_A17 = [
    rec(
        "S5",
        "A",
        17,
        1,
        name="scroll",
        arguments={"direction": "down"},
        action="scrolled down by 600 px",
        cart_after=[],
    ),
    rec(
        "S5",
        "A",
        17,
        2,
        name="click_at",
        arguments={"x": 155, "y": 898},
        action="clicked at (155, 898)",
        cart_after=["cotton tea towel"],
    ),
    rec("S5", "A", 17, 3, status=None, cart_after=None),
]
S5_B1 = [
    rec(
        "S5",
        "B",
        1,
        1,
        name="click_at",
        arguments={"x": 159, "y": 936},
        action="scrolled down by 420 px and clicked at (159, 936)",
        frame_kind="overview",
        scrolled_px=420,
        cart_after=["cotton tea towel"],
    ),
    rec("S5", "B", 1, 2, status=None, cart_after=None),
]


# =====================================================================================================
# Running the builder
# =====================================================================================================


@functools.cache
def builder() -> ModuleType:
    assert BUILDER.is_file(), f"T-0136 A1-1: {BUILDER.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0136_build_outcome_corpus_browser", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{BUILDER.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    for name in ("main", "_git", "pair_id", "compare_builds"):
        assert callable(getattr(module, name, None)), f"T-0136 A1: {BUILDER.name} has no {name!r}"
    return module


def _real_git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Path]]:
    """``build(run_dirs, probes=False) -> out``: one clean in-process builder run (exit 0 asserted)."""
    made: list[Path] = []

    def go(run_dirs: Sequence[Path], *, probes: bool = False) -> Path:
        b = builder()
        sha = uuid.uuid4().hex + uuid.uuid4().hex[:8]
        head = _real_git("rev-parse", "HEAD").strip()

        def git(*args: str) -> str:
            if args[:1] == ("status",):
                return ""
            if args[:1] == ("rev-parse",) and not any(
                a.startswith("--git") or a.startswith("--show") for a in args
            ):
                return (sha[:7] if any(a.startswith("--short") for a in args) else sha) + "\n"
            return _real_git(*(a.replace(sha, head).replace(sha[:7], head[:7]) for a in args))

        monkeypatch.setattr(b, "_git", git)
        (REPO / "logs").mkdir(exist_ok=True)
        out = REPO / "logs" / f"t0136-outcome-corpus-{sha[:7]}"
        made.append(out)
        argv = [*(str(d) for d in run_dirs), "--out", str(out), *(["--probes"] if probes else [])]
        rc = b.main(argv)
        assert rc == 0, f"the builder exited {rc} on {argv}"
        assert (out / "manifest.json").is_file(), f"no manifest.json under {out}"
        return out

    yield go
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


_SHARED: dict[str, Path | BaseException] = {}
_SHARED_DIRS: list[Path] = []


def _one_build(
    run_dirs: Sequence[Path], *, probes: bool, setup: Callable[[Any, Any], None] | None
) -> Path:
    """A clean in-process builder run outside a fixture (so a pending row fails in its call phase, on an assertion);
    ``setup(module, monkeypatch)`` may patch the builder first. The out dir is removed at module teardown."""
    b = builder()
    sha = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    head = _real_git("rev-parse", "HEAD").strip()

    def git(*args: str) -> str:
        if args[:1] == ("status",):
            return ""
        if args[:1] == ("rev-parse",) and not any(
            a.startswith("--git") or a.startswith("--show") for a in args
        ):
            return (sha[:7] if any(a.startswith("--short") for a in args) else sha) + "\n"
        return _real_git(*(a.replace(sha, head).replace(sha[:7], head[:7]) for a in args))

    out = REPO / "logs" / f"t0136-outcome-corpus-{sha[:7]}"
    _SHARED_DIRS.append(out)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(b, "_git", git)
        if setup is not None:
            setup(b, mp)
        (REPO / "logs").mkdir(exist_ok=True)
        argv = [*(str(d) for d in run_dirs), "--out", str(out), *(["--probes"] if probes else [])]
        rc = b.main(argv)
    assert rc == 0, f"the builder exited {rc} on {argv}"
    assert (out / "manifest.json").is_file(), f"no manifest.json under {out}"
    return out


def _shared(key: str, make: Callable[[], Path]) -> Path:
    if key not in _SHARED:
        try:
            _SHARED[key] = make()
        except (
            BaseException
        ) as exc:  # remembered, so every row sharing the build fails the same way
            _SHARED[key] = exc
    got = _SHARED[key]
    if isinstance(got, BaseException):
        raise AssertionError(
            f"the shared {key!r} build failed: {type(got).__name__}: {got}"
        ) from got
    return got


@pytest.fixture(scope="module", autouse=True)
def _shared_builds_cleanup() -> Iterator[None]:
    yield
    for d in _SHARED_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _SHARED.clear()
    _SHARED_DIRS.clear()


def _empty_run_dir() -> Path:
    d = REPO / "logs" / f"t0136-empty-run-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text("", encoding="utf-8")
    _SHARED_DIRS.append(d)
    return d


def probe_build() -> Path:
    """One ``--probes`` build with an empty run dir, shared by the probe rows of this module (about 300 fresh
    browsers once A1c runs page b too: several minutes)."""
    return _shared("probes", lambda: _one_build([_empty_run_dir()], probes=True, setup=None))


def manifest(out: Path) -> dict[str, Any]:
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for k in ("sequences", "reproduced_steps", "truncated_sequences", "pairs"):
        assert isinstance(m.get(k), int), f"manifest.json has no integer {k!r}: {m}"
    return dict(m)


def pairs(out: Path) -> list[dict[str, Any]]:
    path = out / "pairs.jsonl"
    assert path.is_file(), f"no pairs.jsonl under {out}"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [r.get("pair_id") for r in rows]
    assert len(set(ids)) == len(ids), (
        f"pairs.jsonl repeats a pair_id: {Counter(ids).most_common(3)}"
    )
    return rows


def frame_bytes(out: Path, sha: str) -> bytes:
    assert isinstance(sha, str) and len(sha) == 64, sha
    hits = sorted((out / "frames").glob(f"{sha}*"))
    assert len(hits) == 1, f"frame {sha[:12]} is stored {len(hits)} times under frames/"
    data = hits[0].read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, (
        f"frames/{hits[0].name} does not hash to its name"
    )
    return data


def gray(data: bytes) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def size_of(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


def pt(v: Any) -> tuple[int, int] | None:
    return None if v is None else (int(v[0]), int(v[1]))


def check_record(out: Path, r: Mapping[str, Any]) -> None:
    """A1-4's fields on one pair record, its frames stored and its id the builder's own ``pair_id`` of them."""
    assert set(r.get("oracle") or {}) >= ORACLE_KEYS, f"oracle block incomplete: {r.get('oracle')}"
    for k in (
        "pair_id",
        "before",
        "after",
        "point",
        "prev_point",
        "frame_kind",
        "scrolled_px",
        "occurrences",
    ):
        assert k in r, f"pair record lacks {k!r}: {sorted(r)}"
    frame_bytes(out, r["before"])
    frame_bytes(out, r["after"])
    want = builder().pair_id(r["before"], r["after"], pt(r["point"]), pt(r["prev_point"]))
    assert r["pair_id"] == want, (
        "pair_id is not pair_id(before, after, point, prev_point)",
        r["pair_id"],
        want,
    )
    assert isinstance(r["occurrences"], int) and r["occurrences"] >= 1, r["occurrences"]


def sheets(out: Path) -> list[Path]:
    return sorted(p for p in out.rglob("*.png") if p.is_file())


# =====================================================================================================
# A1-5: the probe page (premise, green at BASE_A1) and the probe script
# =====================================================================================================


def _fixture_server() -> Any:
    import http.server
    import threading

    from tests.integration.test_computer_browser_e2e import _Quiet

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(FIXTURE_DIR))
    )
    t = threading.Thread(target=srv.serve_forever, name="t0136-fixture-server", daemon=True)
    t.start()
    return srv, t


HIT_JS = "([x, y]) => { const e = document.elementFromPoint(x, y); return e ? (e.id ? '#' + e.id : e.tagName) : null; }"


async def test_probe_page_premise() -> None:
    """A1-5's fixture, through the real executor at 1280x800: the page fits the viewport (no scroll), every probe's
    pixel point lands on its declared target (a blank probe on the page container), ``window.__state`` starts empty,
    and the table the page exposes is the one in the file."""
    srv, t = _fixture_server()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/outcomes.html"
        async with executor(echo="model") as ex:
            await run(ex.open(url))
            f = await run(ex.frame())
            assert (f.width, f.height) == (W, H), (f.width, f.height)
            page = ex.driver.page
            assert await page.evaluate("document.documentElement.scrollHeight") == H
            assert await page.evaluate("window.__probes") == probe_table()
            st = await page.evaluate("window.__state")
            assert (st["cart"], st["toasts"], st["page"], st["status"]) == ([], [], None, "idle"), (
                st
            )
            for p in probe_table():
                hit = await page.evaluate(HIT_JS, [p["x_px"], p["y_px"]])
                assert hit == (p["target"] or "#probe-page"), (p["id"], hit)
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def test_probes_one_pair_per_probe_with_the_oracle(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-5/A1-4/A1-6: ``--probes`` (with an empty run dir) gives exactly one distinct pair per probe of the table,
    the caret pair included, each at its probe's ``resolved`` point with no previous point, its oracle block filled
    from the page: a badge probe's cart grows by its item and only by it, a link changes the URL and nothing else
    does, nothing scrolls. Every changing probe (near, elsewhere, page) has ``before != after``; a badge probe's
    change is in the badge (x 1060..1260, y 16..56) and nowhere near the click; a page probe's after frame is a
    different page almost everywhere. Every ``none`` click probe (A1's and the A1b follow-up's 84, O-A2) changes no
    pixel by more than 24 grey levels outside the hovered control's box, and nothing in the cart or the URL. One
    contact sheet per pair. (About 120 fresh browsers: this row takes about 1.5 minutes.)"""
    out = probe_build()
    table = probe_table()
    all_rows = pairs(out)
    # A1c-3: a build now runs page b's table too; this A1 row reads page a's records (ids are unique across pages)
    ids_a, ids_b = {p["id"] for p in table}, {p["id"] for p in probe_table_b()}
    assert {r.get("probe") for r in all_rows} <= ids_a | ids_b, "a record of no probe"
    rows = [r for r in all_rows if r.get("probe") in ids_a]
    by_point = {pt(r["point"]): r for r in rows}
    assert len(rows) == len(table) and set(by_point) == {(p["x_px"], p["y_px"]) for p in table}, (
        f"{len(rows)} pairs for {len(table)} probes; missing "
        f"{sorted({(p['x_px'], p['y_px']) for p in table} - set(by_point))}"
    )
    items = {"#add1": "red mug", "#add2": "blue mug", "#add3": "green mug", "#add4": "glass carafe"}
    for p in table:
        r = by_point[(p["x_px"], p["y_px"])]
        check_record(out, r)
        o = r["oracle"]
        assert r["prev_point"] is None, (p["id"], r["prev_point"])
        assert (o["scroll_y_before"], o["scroll_y_after"]) == (0, 0), (p["id"], o)
        assert o["url_changed"] is (p["intended"] == "page"), (p["id"], o)
        assert o["cart_before"] == [], (p["id"], o)
        want_cart = [items[p["target"]]] if p["kind"] == "badge" else []
        if p["intended"] != "page":
            assert o["cart_after"] == want_cart, (p["id"], o)
        before, after = frame_bytes(out, r["before"]), frame_bytes(out, r["after"])
        assert size_of(before) == size_of(after) == (W, H), (
            p["id"],
            size_of(before),
            size_of(after),
        )
        if p["intended"] in ("near", "elsewhere", "page"):
            assert r["before"] != r["after"], f"{p['id']}: before and after are the same frame"
        d = abs(gray(before) - gray(after)) > 24
        if p["kind"] == "badge":
            assert d[16:56, 1060:1260].sum() > 50, (p["id"], int(d[16:56, 1060:1260].sum()))
            y, x = p["y_px"], p["x_px"]
            assert d[max(0, y - 80) : y + 80, max(0, x - 320) : x + 320].sum() == 0, (
                f"{p['id']}: a change near the click; is `before` the frame at dispatch?"
            )
        if p["intended"] == "page":
            assert d.mean() > 0.5, (p["id"], float(d.mean()))
        if p["intended"] == "none" and p["tool"] == "click_at":
            # A1b (O-A2): a `none` probe changes nothing outside the hover styling of the control under the pointer
            # (the hover-only boxes from the fixture's CSS: #hover1/#hover2 460..660 x 280..316 / 340..376, #hover3's
            # line 460..800 x 400..424); blank, heading and text probes change no pixel at all.
            outside = d.copy()
            if p["kind"] == "hover":
                y0, y1, x1 = {
                    "#hover1": (280, 316, 660),
                    "#hover2": (340, 376, 660),
                    "#hover3": (400, 424, 800),
                }[p["target"]]
                outside[y0:y1, 460:x1] = False
            assert outside.sum() == 0, (
                p["id"],
                int(outside.sum()),
                "pixels with |d| > 24 outside the hover box",
            )
            assert o["cart_after"] == [] and o["url_changed"] is False, (p["id"], o)
    shots = sheets(out)
    for r in all_rows:
        named = [s for s in shots if r["pair_id"] in s.name]
        assert len(named) == 1, (r["pair_id"], [s.name for s in shots][:5])
        assert named[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", named[0]
    assert len(shots) == len(all_rows), (len(shots), len(all_rows))


# =====================================================================================================
# A1-2..A1-4: replay
# =====================================================================================================


def test_s2_replay_reproduces_the_logged_action_and_cart(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-2/A1-3/A1-4 on the Goal's S2 #11: both clicks reproduce (action and cart), giving two pairs at (542, 505):
    the first with no previous point and the cart ``[] -> ["blue mug"]``, the second with previous point (542, 505),
    the cart ``["blue mug"] -> ["blue mug", "blue mug"]`` and, as its ``before``, the first click's ``after`` (every
    call refers to the executor's latest own frame). Nothing scrolls or navigates."""
    rd = write_run(tmp_path, T0151_RUN, S2_B11)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"], m["pairs"]) == (
        1,
        2,
        0,
        2,
    ), m
    rows = pairs(out)
    assert len(rows) == 2, rows
    for r in rows:
        check_record(out, r)
        assert pt(r["point"]) == (542, 505), r["point"]
        assert r["frame_kind"] is None and r["occurrences"] == 1, r
        o = r["oracle"]
        assert (o["scroll_y_before"], o["scroll_y_after"], o["url_changed"]) == (0, 0, False), o
        text = json.dumps(r)
        assert T0151_RUN in text and "S2" in text, (
            f"the record does not name its source: {text[:400]}"
        )
    (first,) = [r for r in rows if r["prev_point"] is None]
    (second,) = [r for r in rows if r["prev_point"] is not None]
    assert pt(second["prev_point"]) == (542, 505), second["prev_point"]
    assert (first["oracle"]["cart_before"], first["oracle"]["cart_after"]) == ([], ["blue mug"])
    assert (second["oracle"]["cart_before"], second["oracle"]["cart_after"]) == (
        ["blue mug"],
        ["blue mug", "blue mug"],
    )
    assert second["before"] == first["after"], (
        "step 2's before is not step 1's after (the latest own frame)"
    )
    assert first["before"] != first["after"], "the cart badge changed but the frames are equal"


def test_divergence_truncates_and_keeps_the_earlier_pairs(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-3 (adversarial; the mutant keeps post-divergence pairs): S2 #11 with step 2's logged cart edited to one
    mug (the replay gives two) and a third click logged as reproducing. The sequence is truncated at step 2: one
    reproduced step, one truncated sequence, and only step 1's pair (no previous point) is kept."""
    s2 = [
        dict(S2_B11[0]),
        dict(S2_B11[1], cart_after=["blue mug"]),
        rec(
            "S2",
            "B",
            11,
            3,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            cart_after=["blue mug", "blue mug", "blue mug"],
        ),
    ]
    rd = write_run(tmp_path, T0151_RUN, s2)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"], m["pairs"]) == (
        1,
        1,
        1,
        1,
    ), m
    rows = pairs(out)
    assert len(rows) == 1 and rows[0]["prev_point"] is None, [
        (r["prev_point"], r["oracle"]) for r in rows
    ]
    assert rows[0]["oracle"]["cart_after"] == ["blue mug"], rows[0]["oracle"]


def test_identical_sequences_dedupe_with_occurrences(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-4: two logged turns (arms A and B, different repeats) that make the same single click from the same start
    page. ``pair_id`` is byte-based (O-A1 (7)), and Chromium's frames are not byte-stable between fresh browsers
    (measured: about 1 in 3), so the row branches on the bytes it got:
    - both turns' frames byte-equal: one distinct pair with ``occurrences == 2``, its two frames stored once each;
    - otherwise: two pairs of one occurrence each, the same structure, frames within JPEG noise (max gray |Δ| <= 8,
      none over 24), and every stored frame referenced.
    Either way the occurrences sum to 2 and no frame is stored twice (a builder that never dedupes is caught in
    every byte-equal run)."""
    one = [
        rec(
            "S2",
            "A",
            0,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            cart_after=["blue mug"],
        ),
        rec(
            "S2",
            "B",
            3,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            cart_after=["blue mug"],
        ),
    ]
    rd = write_run(tmp_path, T0151_RUN, one)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"]) == (2, 2, 0), m
    rows = pairs(out)
    for r in rows:
        check_record(out, r)
    assert sum(r["occurrences"] for r in rows) == 2, [
        (r["pair_id"], r["occurrences"]) for r in rows
    ]
    shas = {r[side] for r in rows for side in ("before", "after")}
    stored = [p for p in (out / "frames").iterdir() if p.is_file()]
    assert len(stored) == len(shas), ([p.name for p in stored], sorted(shas))
    if len(rows) == 1:
        assert rows[0]["occurrences"] == 2 and len(stored) == 2, (rows[0], [p.name for p in stored])
        return
    assert len(rows) == 2 and [r["occurrences"] for r in rows] == [1, 1], rows
    a, b = rows
    assert (a["before"], a["after"]) != (b["before"], b["after"]), (
        "two records for byte-equal frames: no dedupe"
    )
    assert (pt(a["point"]), pt(a["prev_point"]), a["frame_kind"], a["oracle"]) == (
        pt(b["point"]),
        pt(b["prev_point"]),
        b["frame_kind"],
        b["oracle"],
    ), (a, b)
    for side in ("before", "after"):
        d = abs(gray(frame_bytes(out, a[side])) - gray(frame_bytes(out, b[side])))
        assert d.max() <= NOISE_MAX and (d > NOISE_OVER).sum() == 0, (side, int(d.max()))


def test_scroll_is_replayed_and_gives_no_pair(tmp_path: Path, build: Callable[..., Path]) -> None:
    """A1-2/A1-4 on S5 arm A #17: the executed ``scroll`` is re-executed (the next click lands on the scrolled page
    and reproduces the tea towel) but yields no pair; the click's pair is at (198, 718) with no previous pointer point
    (a scroll has none) and ``scrollY`` 420 before and after (deals.html is 1220 px)."""
    rd = write_run(tmp_path, T0151_RUN, S5_A17)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"], m["pairs"]) == (
        1,
        2,
        0,
        1,
    ), m
    (r,) = pairs(out)
    check_record(out, r)
    assert pt(r["point"]) == (198, 718) and r["prev_point"] is None, (r["point"], r["prev_point"])
    o = r["oracle"]
    assert (o["cart_before"], o["cart_after"]) == ([], ["cotton tea towel"]), o
    assert (o["scroll_y_before"], o["scroll_y_after"], o["url_changed"]) == (420, 420, False), o


def test_overview_step_replayed_through_the_overview_frame(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-2: a step-1 call with ``frame_kind == "overview"`` (S5 arm B #1) is replayed through the T-0150 capture and
    ``overview_frame_of``, as the session does: it reproduces ``scrolled down by 420 px and clicked at (159, 936)``
    and the tea towel; its pair has ``frame_kind "overview"``, ``scrolled_px`` 420, the viewport point (204, 722)
    and ``scrollY`` 0 -> 420."""
    rd = write_run(tmp_path, T0151_RUN, S5_B1)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"], m["pairs"]) == (
        1,
        1,
        0,
        1,
    ), m
    (r,) = pairs(out)
    check_record(out, r)
    assert (r["frame_kind"], r["scrolled_px"], pt(r["point"])) == ("overview", 420, (204, 722)), r
    o = r["oracle"]
    assert (o["cart_before"], o["cart_after"]) == ([], ["cotton tea towel"]), o
    assert (o["scroll_y_before"], o["scroll_y_after"]) == (0, 420), o
    assert size_of(frame_bytes(out, r["after"])) == (W, H)


BYTE_FIELDS = ("pair_id", "before", "after")
NOISE_MAX = 8  # O-A1 (7): max per-pixel gray |Δ| between two builds' frames
NOISE_OVER = 24  # O-A1 (7): no pixel above this


def structure(r: Mapping[str, Any]) -> str:
    """A record's structure (O-A1 (7)): every field but the byte-derived ones, canonical JSON (source ids, step,
    point, prev_point, frame_kind, scrolled_px, occurrences and the oracle block included)."""
    return json.dumps({k: v for k, v in r.items() if k not in BYTE_FIELDS}, sort_keys=True)


def test_two_runs_give_the_same_pairs_within_jpeg_noise(
    tmp_path: Path, build: Callable[..., Path]
) -> None:
    """A1-7's determinism as ruled in O-A1 (7), on S1-S3 sequences (hand copies of t0151's S1 A #0, S2 B #11, S3 B
    #4): two builds give the same manifest counts (every step reproduces: 3 + 2 + 4 = 9 steps, 9 pairs) and the same
    set of pair structures (every field but ``pair_id``/``before``/``after``, oracle blocks included). Each matched
    pair's ``before`` and ``after`` agree between the builds within JPEG noise, decoded here independently: max
    per-pixel gray |Δ| <= 8 and no pixel over 24 (adversarial: a builder whose frames differ by a large |Δ| between
    builds, e.g. ``after`` taken before the page settled in one build, fails). The builder's own
    ``compare_builds`` records exactly that max |Δ| and no pixel over 24."""
    rd = write_run(tmp_path, T0151_RUN, [*S1_A0, *S2_B11, *S3_B4])
    out1 = build([rd])
    out2 = build([rd])
    m1, m2 = manifest(out1), manifest(out2)
    counts = ("sequences", "reproduced_steps", "truncated_sequences", "pairs")
    assert (m1["sequences"], m1["reproduced_steps"], m1["truncated_sequences"], m1["pairs"]) == (
        3,
        9,
        0,
        9,
    ), m1
    assert {k: m1[k] for k in counts} == {k: m2[k] for k in counts}, (m1, m2)
    r1 = {structure(r): r for r in pairs(out1)}
    r2 = {structure(r): r for r in pairs(out2)}
    assert len(r1) == 9 and set(r1) == set(r2), sorted(set(r1) ^ set(r2))[:4]
    worst = 0
    for key, a in r1.items():
        b = r2[key]
        for side in ("before", "after"):
            d = abs(gray(frame_bytes(out1, a[side])) - gray(frame_bytes(out2, b[side])))
            assert d.max() <= NOISE_MAX and (d > NOISE_OVER).sum() == 0, (
                f"{side} of {a['point']}/{a['prev_point']}: max |Δ| {int(d.max())}, "
                f"{int((d > NOISE_OVER).sum())} px over {NOISE_OVER} between two builds"
            )
            worst = max(worst, int(d.max()))
    rec_ = dict(builder().compare_builds(out1, out2))
    assert (rec_["structure_equal"], rec_["max_abs_delta"], rec_["pixels_over_24"]) == (
        True,
        worst,
        0,
    ), (
        rec_,
        worst,
    )


def test_t0134_shape_replays_as_live(tmp_path: Path, build: Callable[..., Path]) -> None:
    """O-A1 (4): a t0134 run (records with no ``frame_kind`` and no ``cart_after``) is replayed with the missing
    ``frame_kind`` read as a live frame and the cart check skipped: S2's two clicks reproduce on the action alone,
    the pairs have ``frame_kind`` None, and the oracle still reads the real cart (one mug, then two)."""
    rows = [
        rec(
            "S2",
            "A",
            3,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
        rec(
            "S2",
            "A",
            3,
            2,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
        rec("S2", "A", 3, 3, status=None, frame_kind="absent"),
    ]
    assert all("frame_kind" not in r and "cart_after" not in r for r in rows), (
        "premise: the t0134 shape"
    )
    rd = write_run(tmp_path, "t0134-feedback-ab-e59aa69", rows)
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"], m["pairs"]) == (
        1,
        2,
        0,
        2,
    ), m
    rows_out = pairs(out)
    for r in rows_out:
        check_record(out, r)
        assert r["frame_kind"] is None and pt(r["point"]) == (542, 505), r
    carts = sorted((r["oracle"]["cart_before"], r["oracle"]["cart_after"]) for r in rows_out)
    assert carts == [([], ["blue mug"]), (["blue mug"], ["blue mug", "blue mug"])], carts


# =====================================================================================================
# A1-3: the criterion on the real t0151 run (slow; opt-in)
# =====================================================================================================


def _t0151_dir() -> Path | None:
    here = REPO / "logs" / T0151_RUN
    if here.is_dir():
        return here
    common = Path(_real_git("rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    there = common.parent / "logs" / T0151_RUN
    return there if there.is_dir() else None


def test_t0151_reproduces_at_least_95_percent(build: Callable[..., Path]) -> None:
    """A1-3's criterion: at least 95 % of the logged executed steps of ``t0151-overview-frame-ab-4471255`` reproduce.
    The denominator is counted here from its ``steps.jsonl`` (``status == "executed"``); the sequences are its
    ``(arm, task, repeat)`` with at least one executed step. Opt-in: ``T0136_FULL_REPLAY=1`` (minutes)."""
    if os.environ.get(FULL_ENV) != "1":
        pytest.skip(f"{FULL_ENV} is not 1: the full t0151 replay takes minutes (A1-3, A1-7)")
    rd = _t0151_dir()
    if rd is None:
        pytest.skip(f"{T0151_RUN} is not under this worktree's or the main checkout's logs/")
    logged = [
        json.loads(line)
        for line in (rd / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    executed = [r for r in logged if r.get("status") == "executed"]
    seqs = {(r.get("arm"), r["task"], r["repeat"]) for r in executed}
    out = build([rd])
    m = manifest(out)
    assert m["sequences"] == len(seqs), (m, len(seqs))
    assert m["reproduced_steps"] >= 0.95 * len(executed), (m["reproduced_steps"], len(executed))


# =====================================================================================================
# Review gaps 2 and 3: the per-arm echo (O-A1b (1)) and the caret pair (A1-5)
# =====================================================================================================


def test_arm_echo_comes_from_the_run_manifest(tmp_path: Path, build: Callable[..., Path]) -> None:
    """O-A1b (1) (kills ``echo-ignore-manifest``): the run's ``manifest.json`` gives arm A ``echo: pixels`` and arm B
    ``echo: model``; arm C is not listed (default ``model``). The same click ``click_at(423, 631)`` was logged by A as
    ``clicked at (542, 505)`` (pixels, as t0134 arm A #8 logged it) and by B and C as ``clicked at (423, 631)``. All
    three reproduce: a builder that replays every arm with one echo truncates A (or B and C)."""
    rows = [
        rec(
            "S2",
            "A",
            8,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action="clicked at (542, 505)",
            frame_kind="absent",
        ),
        rec(
            "S2",
            "B",
            2,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
        rec(
            "S2",
            "C",
            5,
            1,
            name="click_at",
            arguments={"x": 423, "y": 631},
            action=C1,
            frame_kind="absent",
        ),
    ]
    rd = write_run(tmp_path, "t0134-feedback-ab-e59aa69", rows)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "regime": {
                    "arms": [{"label": "A", "echo": "pixels"}, {"label": "B", "echo": "model"}]
                }
            }
        ),
        encoding="utf-8",
    )
    out = build([rd])
    m = manifest(out)
    assert (m["sequences"], m["reproduced_steps"], m["truncated_sequences"]) == (3, 3, 0), m


def test_caret_pair_is_after_the_click_and_530_ms_later(
    tmp_path: Path, build: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1-5's caret pair (kills ``caret-no-wait`` and ``caret-pair-is-click``): with the probe table cut to the
    caret probe and every executor the builder makes wrapped by a recorder (the builder's ``probe_table`` and
    ``_executor`` seams), the caret pair's ``before`` is the executor's latest frame right after the focus click
    (not the frame before it), and its ``after`` is a fresh ``frame()`` that returned at least 500 ms after that
    click returned (the probe waits ``wait_ms`` = 530 ms). Time is the test's monotonic clock: only a lower bound."""
    import time as _time

    b = builder()
    for seam in ("probe_table", "_executor"):
        assert callable(getattr(b, seam, None)), f"the builder has no {seam!r} seam (review gap 3)"
    caret = [p for p in b.probe_table() if p.get("tool") == "caret"]
    assert len(caret) == 1, caret
    monkeypatch.setattr(b, "probe_table", lambda: [dict(caret[0])])
    log: list[tuple[str, float, str | None]] = []
    real = b._executor

    def sha(f: Any) -> str | None:
        return None if f is None else hashlib.sha256(f.jpeg).hexdigest()

    def spy(*args: Any, **kwargs: Any) -> Any:
        ex = real(*args, **kwargs)
        real_frame, real_execute = ex.frame, ex.execute

        async def frame_spy() -> Any:
            f = await real_frame()
            log.append(("frame", _time.monotonic(), sha(f)))
            return f

        async def execute_spy(i: Any) -> Any:
            out = await real_execute(i)
            log.append(("execute", _time.monotonic(), sha(ex.latest_frame)))
            return out

        ex.frame, ex.execute = frame_spy, execute_spy  # instance attributes shadow the methods
        return ex

    monkeypatch.setattr(b, "_executor", spy)
    empty = write_run(tmp_path, "empty-run", [])
    out = build([empty], probes=True)
    rows = pairs(out)
    assert len(rows) == 1 and pt(rows[0]["point"]) == (1160, 538), rows
    r = rows[0]
    execs = [e for e in log if e[0] == "execute"]
    assert len(execs) == 1, (
        f"the spy saw {len(execs)} executions (is _executor the builder's only factory?)"
    )
    _, t_click, after_click = execs[0]
    frames_before = [e for e in log if e[0] == "frame" and e[1] <= t_click]
    assert r["before"] == after_click, (
        "the caret pair's before is not the frame after the focus click"
        + (
            " (it is the frame before the click)"
            if any(f[2] == r["before"] for f in frames_before)
            else ""
        )
    )
    later = [e for e in log if e[0] == "frame" and e[2] == r["after"]]
    assert later, "the caret pair's after is not a frame() the executor returned"
    assert max(e[1] for e in later) - t_click >= 0.5, (
        f"the after frame came {max(e[1] for e in later) - t_click:.3f} s after the click; the probe waits 530 ms"
    )


# =====================================================================================================
# A1c: the second probe page through Chromium (premise rows, not pending: the page is the test-author's) and the
# builder's page-b probes (A1c-3, A1c-4, A1c-6; pending until /implement of A1c)
# =====================================================================================================
#
# Thresholds. O-A1 (7) bounds JPEG noise between two captures of the same screen at max gray |Δ| 8 with no pixel
# over 24, so "differs" is |Δ| > 8 (the A1c-4 jitter rule) and "a visible pixel change" is |Δ| > 24. An underline's
# rows are those with a pixel |Δ| > 32: measured 2026-10-25 on the 26 N-UL probes (Playwright 1.64.0, Chromium 156
# headless, 1280x800, n = 1 each), 1-3 JPEG-ringing pixels per link exceed 24 one to four rows above the underline
# inside its 8x8 block, none exceeds 32. A change box is padded by 8 px (one JPEG block) for ringing.

PAGE_B = "outcomes_b.html"
PAD = 8
RECT_JS = (
    "(sel) => { const e = document.querySelector(sel); if (!e) return null; const r = e.getBoundingClientRect();"
    " return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; }"
)
HIT_B_JS = (
    "([x, y, sel]) => { const e = document.elementFromPoint(x, y); if (!e) return 'nothing';"
    " if (sel === null) return e.tagName === 'SECTION' ? 'section' : 'el:' + (e.id || e.tagName);"
    " return e.closest(sel) ? 'ok' : 'miss:' + (e.id || e.tagName); }"
)
UL_JS = (
    "(sel) => { const e = document.querySelector(sel); const r = e.getBoundingClientRect(); const cs = getComputedStyle(e);"
    " return {w: Math.round(r.width), h: Math.round(r.height), fs: cs.fontSize, lh: cs.lineHeight,"
    " sw: e.scrollWidth, cw: e.clientWidth, sh: e.scrollHeight, ch: e.clientHeight}; }"
)


def _pad(box: Sequence[int], k: int = PAD) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, x0 - k), max(0, y0 - k), min(W, x1 + k), min(H, y1 + k)


def _mask(boxes: Sequence[Sequence[int]]) -> Any:
    import numpy as np

    m = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def _page_b_html() -> str:
    return PROBE_PAGE_B.read_text(encoding="utf-8")


def _html_box(sel: str) -> tuple[int, int, int, int]:
    return _box(_element(_page_b_html(), sel[1:])[1])


def _runs(rows: Sequence[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for r in rows:
        if out and r == out[-1][-1] + 1:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def _thin_band_problem(d: Any, box: Sequence[int], lines: int) -> str | None:
    """None when the |Δ| is an underline: the rows with a pixel |Δ| > 32 form one run of at most 2 rows per line (at
    least one run), each in the lower half of a line box of the link, and no pixel |Δ| > 24 lies outside the link's
    box grown by 2 px."""
    import numpy as np

    _x0, y0, _x1, y1 = box
    lh = (y1 - y0) // lines
    runs = _runs(sorted(set(np.nonzero(d > 32)[0].tolist())))
    if not 1 <= len(runs) <= lines:
        return f"{len(runs)} row bands for {lines} line(s): {runs}"
    for band in runs:
        if len(band) > 2:
            return f"a band of {len(band)} rows: {band}"
        if not any(
            y0 + k * lh + lh // 2 <= band[0] and band[-1] < y0 + (k + 1) * lh for k in range(lines)
        ):
            return f"band {band} is not in the lower half of a line of {box}"
    outside = (d > 24) & ~_mask([_pad(box, 2)])
    if outside.any():
        return f"{int(outside.sum())} px over 24 outside the link"
    return None


async def test_probe_page_b_premise() -> None:
    """A1c-1's fixture through the real executor at 1280x800, scene by scene: each fits the viewport (no scroll),
    nothing is focused, ``window.__state.scene`` is the query's, the table the page exposes is the file's; every
    probe's point lands on (inside) its target (a blank point on the scene itself), every ``pre`` point on its link;
    every N-UL link's rendered width, font size, line count and overflow (text longer than the box) are the table's;
    every ``changes`` rect is its element's rendered box (a focus probe's grown by 3 px)."""
    srv, t = _fixture_server()
    bad: list[str] = []
    try:
        origin = f"http://127.0.0.1:{srv.server_address[1]}"
        table = probe_table_b()
        async with executor(echo="model") as ex:
            for q in sorted({p["query"] for p in table}):
                await run(ex.open(f"{origin}/{PAGE_B}{q}"))
                f = await run(ex.frame())
                assert (f.width, f.height) == (W, H), (q, f.width, f.height)
                page = ex.driver.page
                assert await page.evaluate("document.documentElement.scrollHeight") == H, q
                assert await page.evaluate("document.activeElement === document.body"), q
                assert await page.evaluate("window.__state.scene") == q[len("?scene=") :], q
                assert await page.evaluate("window.__probes") == table, q
                for p in [p for p in table if p["query"] == q]:
                    hit = await page.evaluate(HIT_B_JS, [p["x_px"], p["y_px"], p["target"]])
                    if hit != ("section" if p["target"] is None else "ok"):
                        bad.append(f"{p['id']}: point {hit}")
                    if "pre" in p:
                        pre = p["pre"]
                        h2 = await page.evaluate(
                            HIT_B_JS, [pre["x_px"], pre["y_px"], pre["target"]]
                        )
                        if h2 != "ok":
                            bad.append(f"{p['id']}: pre point {h2}")
                    if p["kind"] == "n-ul":
                        g = await page.evaluate(UL_JS, p["target"])
                        lh = int(float(g["lh"].removesuffix("px")))
                        overflow = g["sw"] > g["cw"] if p["lines"] == 1 else g["sh"] > g["ch"]
                        if (g["w"], g["fs"], g["h"], overflow) != (
                            p["width_px"],
                            f"{p['font_px']}px",
                            lh * p["lines"],
                            True,
                        ):
                            bad.append(f"{p['id']}: link geometry {g}")
                    if "changes_el" in p:
                        r = await page.evaluate(RECT_JS, p["changes_el"])
                        if p["kind"] == "c-focus":
                            r = [r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3]
                        if r != p["changes"]:
                            bad.append(
                                f"{p['id']}: changes {p['changes']} but the element is at {r}"
                            )
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)
    assert bad == [], bad


async def _capture(origin: str, p: Mapping[str, Any]) -> dict[str, Any]:
    """One page-b probe through a fresh executor as A1c-3 describes it (load, frame, the ``pre`` click if any, the
    click), independently of the builder: the fresh, before and after frames, the outcome, the URL and the target's
    and the pre link's rendered boxes."""
    async with executor(echo="model") as ex:
        await run(ex.open(f"{origin}/{PAGE_B}{p['query']}"))
        fresh = await run(ex.frame())
        page = ex.driver.page
        tbox = await page.evaluate(RECT_JS, p["target"]) if p["target"] else None
        pbox = await page.evaluate(RECT_JS, p["pre"]["target"]) if "pre" in p else None
        before = fresh
        prev = None
        if "pre" in p:
            o = await execute(ex, inv("click_at", fresh, x=p["pre"]["x"], y=p["pre"]["y"]))
            assert o.status == "executed", (p["id"], "pre", o)
            prev = (int(o.resolved["x_px"]), int(o.resolved["y_px"]))
            before = ex.latest_frame
        out = await execute(ex, inv("click_at", before, x=p["x"], y=p["y"]))
        return {
            "fresh": fresh.jpeg,
            "before": before.jpeg,
            "after": ex.latest_frame.jpeg,
            "out": out,
            "prev": prev,
            "url": str(page.url),
            "tbox": tbox,
            "pbox": pbox,
        }


def _capture_problem(p: Mapping[str, Any], c: Mapping[str, Any]) -> str | None:
    """What A1c-2 says this probe's capture must show (Q-A1c-1), or why it does not."""

    if c["out"].status != "executed":
        return f"not executed: {c['out'].action}"
    if (int(c["out"].resolved["x_px"]), int(c["out"].resolved["y_px"])) != (p["x_px"], p["y_px"]):
        return f"resolved {c['out'].resolved}"
    d = abs(gray(c["before"]) - gray(c["after"]))
    kind, intended = p["kind"], p["intended"]
    hover = [_pad(b) for b in (c["tbox"], c["pbox"]) if b is not None]
    if p["stress"]:
        return None  # report-only (A1c-2); printed by the row
    if kind in TRIVIAL_KINDS:
        return None if int(d.max()) <= 8 else f"trivial probe changed: max |Δ| {int(d.max())}"
    if intended == "none":
        if int(d.max()) <= 8:
            return f"the after frame does not differ (max |Δ| {int(d.max())}): hover styling not captured"
        outside = (d > 24) & ~_mask(hover)
        if outside.any():
            return f"{int(outside.sum())} px over 24 outside the hovered element(s)"
        if kind == "n-ul":
            return _thin_band_problem(d, c["tbox"], p["lines"])
        if kind == "n-pre":
            f = abs(gray(c["fresh"]) - gray(c["before"]))
            x0, y0, x1, y1 = c["pbox"]
            if int(f[y0:y1, x0:x1].max()) <= 24:
                return "the pre link's hover is not in the before frame"
        return None
    if intended in ("near", "elsewhere"):
        x0, y0, x1, y1 = _pad(p["changes"])
        inside = int((d[y0:y1, x0:x1] > 24).sum())
        if inside < 10:
            return f"only {inside} px over 24 in the change rect"
        outside = (d > 24) & ~_mask([_pad(p["changes"]), *hover])
        if outside.any():
            return (
                f"{int(outside.sum())} px over 24 outside the change rect and the hovered element"
            )
        if intended == "elsewhere":
            x, y = p["x_px"], p["y_px"]
            near = _mask([(max(0, x - 320), max(0, y - 80), min(W, x + 321), min(H, y + 81))])
            stray = (d > 24) & near & ~_mask(hover)
            if stray.any():
                return f"{int(stray.sum())} px over 24 in the near box"
        return None
    # page
    if not c["url"].endswith(p["dest"]):
        return f"no navigation to {p['dest']}: {c['url']}"
    if (d > 24).mean() <= 0.5:
        return f"only {(d > 24).mean():.2f} of the frame changed"
    top = d[:64] > 24
    if kind == "p-bar" and top.any():
        return f"the 64 px bar changed ({int(top.sum())} px)"
    if kind == "p-full" and top.mean() <= 0.5:
        return "the bar area did not change on a full replacement"
    return None


async def test_probe_page_b_capture_premise() -> None:
    """Q-A1c-1 / A1c-2's premise, through the real executor and headless Chromium, every page-b probe in a fresh
    browser as A1c-3 runs it (independently of the builder):

    - gated ``none``: the after frame differs (max |Δ| > 8: not trivial) and only inside the hovered element(s)
      (target, and the ``pre`` link for N-PRE); an N-UL change is a thin band (``_thin_band_problem``); an N-PRE
      ``before`` shows the pre link's hover against the fresh load;
    - trivial: max |Δ| <= 8;
    - gated ``near``/``elsewhere``: >= 10 px over 24 in the ``changes`` rect and none outside it but in the hovered
      target; an elsewhere probe has none in the near box outside the target;
    - ``page``: the URL is the probe's ``dest``, more than half the frame changes, a bar-keeping one leaves the top
      64 px unchanged and a full replacement does not.
    Stress probes are executed and reported only. Prints one line per kind (max |Δ|, px over 24). About 176 fresh
    browsers (a few minutes)."""
    import numpy as np

    srv, t = _fixture_server()
    bad: list[str] = []
    stats: dict[str, list[tuple[int, int]]] = {}
    try:
        origin = f"http://127.0.0.1:{srv.server_address[1]}"
        for p in probe_table_b():
            c = await _capture(origin, p)
            d = abs(gray(c["before"]) - gray(c["after"]))
            stats.setdefault(p["kind"], []).append((int(d.max()), int((d > 24).sum())))
            why = _capture_problem(p, c)
            if why is not None:
                bad.append(f"{p['id']} ({p['kind']}): {why}")
            if "pre" in p and c["prev"] != (p["pre"]["x_px"], p["pre"]["y_px"]):
                bad.append(f"{p['id']}: the pre click resolved to {c['prev']}")
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)
    for k, v in sorted(stats.items()):
        mx = [a for a, _ in v]
        print(
            f"T-0136 A1c capture {k}: n={len(v)} max|Δ| {min(mx)}..{max(mx)}, median px>24 {int(np.median([b for _, b in v]))}"
        )
    assert bad == [], bad


def _records_b(out: Path) -> list[dict[str, Any]]:
    ids_b = {p["id"] for p in probe_table_b()}
    return [r for r in pairs(out) if r.get("probe") in ids_b]


def _cart_item(target: str | None) -> list[str]:
    if target is None:
        return []
    m = re.search(r'data-cart="([^"]*)"', _element(_page_b_html(), target[1:])[1])
    return [m.group(1)] if m else []


def test_probe_page_b_one_pair_per_probe() -> None:
    """A1c-3/A1c-6: a ``--probes`` build gives exactly one pair per page-b probe (after page a's, which keep
    ``probe_page`` "outcomes.html"), each with ``probe_page`` "outcomes_b.html", its probe's ``resolved`` point, its
    ``pre``'s resolved point as ``prev_point`` (None without a ``pre``) and its oracle block: nothing scrolls, the
    cart starts empty and grows by the trigger's ``data-cart`` item only on the probes that have one, the URL changes
    exactly on page probes. The manifest regime keeps ``probe_page_sha256`` (page a) and adds
    ``probe_pages_sha256`` for both pages."""
    out = probe_build()
    table = probe_table_b()
    rows = _records_b(out)
    assert rows, "no page-b record: the build ran page a only"
    by_probe: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_probe.setdefault(r["probe"], []).append(r)
    assert set(by_probe) == {p["id"] for p in table}, sorted(
        {p["id"] for p in table} - set(by_probe)
    )[:10]
    assert all(len(v) == 1 for v in by_probe.values()), [
        k for k, v in by_probe.items() if len(v) > 1
    ]
    ids_a = {p["id"] for p in probe_table()}
    assert {r.get("probe_page") for r in pairs(out) if r.get("probe") in ids_a} == {"outcomes.html"}
    for p in table:
        (r,) = by_probe[p["id"]]
        check_record(out, r)
        assert r.get("probe_page") == PAGE_B, (p["id"], r.get("probe_page"))
        assert pt(r["point"]) == (p["x_px"], p["y_px"]), (p["id"], r["point"])
        want_prev = (p["pre"]["x_px"], p["pre"]["y_px"]) if "pre" in p else None
        assert pt(r["prev_point"]) == want_prev, (p["id"], r["prev_point"])
        o = r["oracle"]
        assert (o["scroll_y_before"], o["scroll_y_after"]) == (0, 0), (p["id"], o)
        assert o["url_changed"] is (p["intended"] == "page"), (p["id"], o)
        assert o["cart_before"] == [], (p["id"], o)
        if p["intended"] != "page":
            assert o["cart_after"] == _cart_item(p["target"]), (p["id"], o)
        assert (
            size_of(frame_bytes(out, r["before"]))
            == size_of(frame_bytes(out, r["after"]))
            == (W, H)
        )
    regime = manifest(out)["regime"]
    sha = {
        n: hashlib.sha256(q.read_bytes()).hexdigest()
        for n, q in (("outcomes.html", PROBE_PAGE), (PAGE_B, PROBE_PAGE_B))
    }
    assert regime.get("probe_page_sha256") == sha["outcomes.html"], regime.get("probe_page_sha256")
    assert regime.get("probe_pages_sha256") == sha, regime.get("probe_pages_sha256")


def test_pre_probe_before_is_hovered_and_prev_point_set() -> None:
    """A1c-3/A1c-6 on every N-PRE probe of the build: ``prev_point`` is the pre click's resolved point (the hand
    mapping of ``pre.x``/``pre.y``), and ``before`` differs from the scene's fresh-load frame (the ``before`` of the
    trivial probe ``b-triv-01``, a fresh-load click on the same scene) by the pre link's hover: a pixel over 24 inside
    the link's box and none outside it (grown by 8 px).

    Adversarial (the task's mutant): a builder that takes ``before`` from the fresh load and ignores ``pre`` gives a
    ``before`` within JPEG noise of the fresh frame (no pixel over 24 in the link) and no ``prev_point``."""
    out = probe_build()
    rows = {r["probe"]: r for r in _records_b(out)}
    assert "b-triv-01" in rows, "no page-b record for the fresh-load reference b-triv-01"
    fresh = gray(frame_bytes(out, rows["b-triv-01"]["before"]))
    pres = [p for p in probe_table_b() if "pre" in p]
    for p in pres:
        assert p["id"] in rows, f"{p['id']}: no pair"
        r = rows[p["id"]]
        assert pt(r["prev_point"]) == (p["pre"]["x_px"], p["pre"]["y_px"]), (
            p["id"],
            r["prev_point"],
        )
        d = abs(gray(frame_bytes(out, r["before"])) - fresh)
        x0, y0, x1, y1 = _html_box(p["pre"]["target"])
        assert int(d[y0:y1, x0:x1].max()) > 24, (
            f"{p['id']}: before shows no hover on {p['pre']['target']}"
        )
        outside = (d > 24) & ~_mask([_pad((x0, y0, x1, y1))])
        assert not outside.any(), (
            f"{p['id']}: before differs from the fresh load outside the pre link"
        )


def test_underline_probe_is_captured_as_a_thin_band() -> None:
    """A1c-6, the landmine premise, on the builder's own frames: every N-UL pair's after frame differs from its
    before frame only in a thin band under the link (``_thin_band_problem``: one run of <= 2 rows per line with a
    pixel |Δ| > 32, in the lower half of a line box, nothing over 24 outside the link). A builder whose after frame
    is taken without the pointer's hover (or before the settle) shows no band and fails."""
    out = probe_build()
    rows = {r["probe"]: r for r in _records_b(out)}
    uls = [p for p in probe_table_b() if p["kind"] == "n-ul"]
    bad = []
    for p in uls:
        assert p["id"] in rows, f"{p['id']}: no pair"
        r = rows[p["id"]]
        d = abs(gray(frame_bytes(out, r["before"])) - gray(frame_bytes(out, r["after"])))
        why = _thin_band_problem(d, _html_box(p["target"]), p["lines"])
        if why:
            bad.append(f"{p['id']}: {why}")
    assert bad == [], bad


def _structure_a(r: Mapping[str, Any]) -> str:
    return json.dumps({k: v for k, v in r.items() if k not in BYTE_FIELDS}, sort_keys=True)


def test_page_a_pairs_unchanged_by_page_b() -> None:
    """A1c-6/A1c-3 ("outcomes.html's probes run exactly as before"): the page-a records of the both-pages build and of
    a build with ``PROBE_PAGES`` cut to page a have equal structures (every field but ``pair_id``/``before``/
    ``after``), and each matched pair's frames agree within JPEG noise (max gray |Δ| <= 8, none over 24).
    Adversarial: page b's server, URL or order perturbing page a (a different root, a page-b scene left loaded, a
    shared browser) changes a structure or a frame. A build without page-b records fails first (named)."""
    both = probe_build()
    assert _records_b(both), "no page-b record: the build ran page a only"
    a_only = _shared(
        "probes-a-only",
        lambda: _one_build(
            [_empty_run_dir()],
            probes=True,
            setup=lambda b, mp: mp.setattr(b, "PROBE_PAGES", (PROBE_PAGE,), raising=False),
        ),
    )
    assert not _records_b(a_only), "PROBE_PAGES cut to page a still ran page b"
    ids_a = {p["id"] for p in probe_table()}
    ra = {_structure_a(r): r for r in pairs(both) if r.get("probe") in ids_a}
    rb = {_structure_a(r): r for r in pairs(a_only) if r.get("probe") in ids_a}
    assert len(ra) == len(ids_a) and set(ra) == set(rb), sorted(set(ra) ^ set(rb))[:3]
    for key, x in ra.items():
        y = rb[key]
        for side in ("before", "after"):
            d = abs(gray(frame_bytes(both, x[side])) - gray(frame_bytes(a_only, y[side])))
            assert d.max() <= NOISE_MAX and (d > NOISE_OVER).sum() == 0, (
                x.get("probe"),
                side,
                int(d.max()),
            )


def _own_groups(frames: Mapping[str, bytes]) -> set[frozenset[str]]:
    """Independent A1c-4 clustering for the row: same size and max full-resolution gray |Δ| <= 8, transitive; block
    means (8x8) prefilter the candidate pairs (|mean Δ| <= max |Δ|, so no true edge is skipped)."""
    import numpy as np

    keys = sorted(frames)
    g = {k: gray(frames[k]) for k in keys}
    sig = {}
    for k in keys:
        a = g[k]
        h, w = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
        sig[k] = a[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if g[a].shape != g[b].shape or np.abs(sig[a] - sig[b]).max() > 8:
                continue
            if int(np.abs(g[a] - g[b]).max()) <= 8:
                parent[find(b)] = find(a)
    by: dict[str, set[str]] = {}
    for k in keys:
        by.setdefault(find(k), set()).add(k)
    return {frozenset(v) for v in by.values()}


def test_frame_groups_written() -> None:
    """A1c-4 on a real build: ``frame_groups.json`` maps every frame stored under ``frames/`` (by its sha256 name)
    and nothing else to a cluster id; each cluster's id is its smallest sha256; the clusters equal this row's own
    clustering of the decoded frames (adversarial, mutation 21: sha-only groups miss every jitter variant, and the
    page-b scenes repeat their fresh-load frame in byte-different captures)."""
    out = probe_build()
    path = out / "frame_groups.json"
    assert path.is_file(), "no frame_groups.json"
    groups = json.loads(path.read_text(encoding="utf-8"))
    names = {p.name.split(".")[0]: p for p in (out / "frames").iterdir() if p.is_file()}
    assert set(groups) == set(names), sorted(set(groups) ^ set(names))[:4]
    clusters: dict[str, set[str]] = {}
    for sha, gid in groups.items():
        clusters.setdefault(gid, set()).add(sha)
    for gid, members in clusters.items():
        assert gid == min(members), (gid, sorted(members)[:3])
    own = _own_groups({k: p.read_bytes() for k, p in names.items()})
    assert {frozenset(v) for v in clusters.values()} == own, (len(clusters), len(own))


def test_unexecuted_pre_yields_no_pair(capsys: pytest.CaptureFixture[str]) -> None:
    """A1c-3: when a probe's pre click is not ``executed`` (here x = 1500, refused by the executor as bad
    coordinates: an integer, so the table check passes it), the probe yields no pair and a printed line names it;
    the next probe is unaffected."""
    b_table = probe_table_b()
    pre = next(dict(p) for p in b_table if "pre" in p)
    pre = dict(pre, id="b-pre-unexecuted", pre=dict(pre["pre"], x=1500), probe_page=PAGE_B)
    good = dict(next(p for p in b_table if p["kind"] == "n-btn"), probe_page=PAGE_B)
    out = _one_build(
        [_empty_run_dir()],
        probes=True,
        setup=lambda b, mp: mp.setattr(b, "probe_table", lambda: [pre, good]),
    )
    err = capsys.readouterr().err
    rows = pairs(out)
    assert [r.get("probe") for r in rows] == [good["id"]], [r.get("probe") for r in rows]
    assert "b-pre-unexecuted" in err, err[-2000:]


def test_probe_runs_urls_and_calls() -> None:
    """A1c-3, observed through the builder's ``_executor`` seam (wrapped by a recorder): with the plan cut to two
    page-a probes then a page-b N-PRE and a page-b N-UL probe, each runs in its own executor in that order, opening
    ``/outcomes.html`` (no query: A1's URL), ``/outcomes_b.html?scene=pre`` and ``/outcomes_b.html?scene=ul1``, and
    executes exactly one ``click_at`` at the probe's model point, preceded by one at the ``pre`` point for the N-PRE
    probe."""
    b = builder()
    a2 = [dict(p, probe_page="outcomes.html") for p in probe_table()[:2]]
    bp = dict(next(p for p in probe_table_b() if "pre" in p), probe_page=PAGE_B)
    bu = dict(next(p for p in probe_table_b() if p["kind"] == "n-ul"), probe_page=PAGE_B)
    plan = [*a2, bp, bu]
    log: list[list[Any]] = []
    real = b._executor

    def spy(*args: Any, **kwargs: Any) -> Any:
        ex = real(*args, **kwargs)
        mine: list[Any] = []
        log.append(mine)
        real_open, real_execute = ex.open, ex.execute

        async def open_spy(url: str, *a: Any, **k: Any) -> Any:
            mine.append(("open", url))
            return await real_open(url, *a, **k)

        async def execute_spy(i: Any) -> Any:
            mine.append(("call", i.name, dict(i.arguments)))
            return await real_execute(i)

        ex.open, ex.execute = open_spy, execute_spy
        return ex

    def setup(mod: Any, mp: Any) -> None:
        mp.setattr(mod, "probe_table", lambda: [dict(p) for p in plan])
        mp.setattr(mod, "_executor", spy)

    _one_build([_empty_run_dir()], probes=True, setup=setup)
    assert len(log) == 4, f"{len(log)} executors for 4 probes"
    for mine, p in zip(log, plan, strict=True):
        opens = [e[1] for e in mine if e[0] == "open"]
        calls = [e[1:] for e in mine if e[0] == "call"]
        suffix = "/" + p["probe_page"] + p.get("query", "")
        assert len(opens) == 1 and opens[0].endswith(suffix) and "127.0.0.1" in opens[0], (
            p["id"],
            opens,
        )
        want = [("click_at", {"x": p["x"], "y": p["y"]})]
        if "pre" in p:
            want.insert(0, ("click_at", {"x": p["pre"]["x"], "y": p["pre"]["y"]}))
        assert [(n, {k: a[k] for k in ("x", "y")}) for n, a in calls] == want, (p["id"], calls)
