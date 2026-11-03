"""T-0208 group B, criteria B2, B8 and B10: the three shared images read as files, the two start-page roots' 404s, the
``browser-handoff`` Make target and the frozen paths (docs/tasks/T-0208.md, "Tests to write",
``tests/unit/test_t0208_handoff_static.py``).

**What these rows read.** ``local_astra/pages/handoff/`` (B2's directory, by path), the table's ``image`` and
``image_sha256`` (through ``handoff_tasks``, a named ``AssertionError`` while it is absent), Pillow's decoder, two
real loopback ``StartPageServer``s, ``make -n -s`` and ``git diff`` over literal commits. No browser.

**Oracles.** The image sha is recomputed here from the committed bytes and compared with the table's own
``image_sha256`` (the shas cannot be typed before the images exist; ``test_t0208_handoff_judge.IMAGE_SHA256`` takes the
literals once reported). Size and markers come from the bytes (SOI, EOI, a baseline/progressive SOF segment parsed by
hand) and from Pillow. Quality 85 is checked by comparing the file's quantisation tables with the tables Pillow's own
encoder writes at ``quality=85`` (libjpeg's standard tables scaled to 85; an independent encoder run, not the
implementer's script). "Inside the HUD's screen bucket" is ``policy.target_size(w, h, policy.bucket_for(SCREEN, w, h))
== (w, h)``: the HUD would send the image unscaled.

**Readings this module pins where the task is silent** (reported):

- The directory holds exactly the three JPEGs and ``MANIFEST.md``; the manifest names each image, says
  "original" and "Apache-2.0" and names ``scripts/make_t0208_shared_images.py``.
- The generator script exists, imports Pillow and no network module (``socket``, ``urllib``, ``http``, ``httpx``,
  ``requests``, ``ddgs``). Whether it reproduces the committed bytes is not pinned (B2: the bytes are the source of
  truth).
- "Neither root serves the directory (404)": a real ``StartPageServer()`` and ``StartPageServer(static_dir=
  SCENARIO_DIR)`` answer exactly 404 to each image under ``/handoff/``, at the root, under ``/pages/handoff/`` and to
  ``/handoff/MANIFEST.md``; the traversal spellings (``/../handoff/...``, ``/%2e%2e/handoff/...``) answer anything but
  200 and never the image's bytes.
- B10's TIP is a literal: BASE until group B's implementation lands; the test-author or orchestrator bumps ``TIP`` to
  B's last implementation commit, and at the merge it closes to that commit (tests.md, T-0094). Under ``tests/`` only
  this task's three B modules and ``test_guards_use_fixed_refs.py`` may change over the range. The Makefile may change
  only by lines naming ``browser-handoff`` (B8: the orchestrator's target and its ``.PHONY`` entry).

BASE = 7e52c3aa (main at branch time), TIP = 60ce31dc (B's last implementation commit, the H1 caption fix). Activated at group B's
``/implement`` (no ``pending`` marker); the B8 row stays red until the orchestrator writes the Makefile target.

Regime: CPU, files and loopback sockets only, n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import io
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit.test_start_pages import new_server, port_of, request

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
HANDOFF_DIR = PAGES / "handoff"
SCRIPT = REPO / "scripts" / "make_t0208_shared_images.py"
MODULE = "local_astra.pages.handoff_tasks"
#: B1/B2, typed from the task
IMAGES = {"H1": "h1_steam_wand.jpg", "H2": "h2_stove_box.jpg", "H3": "h3_chat.jpg"}
W, H = 1280, 800
QUALITY = 85
MAKE_BROWSER_HANDOFF = (
    "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0208_share_handoff_hud.py "
    "tests/integration/test_t0208_handoff_session.py tests/integration/test_t0211_shared_ctrl_l.py -q -m browser"
)
#: T-0208 group B's BASE: main at 7e52c3aa ("T-0208 planned: shared screen to browser handoff scenarios").
BASE = "7e52c3aa49b38347a16d50d7d90d9c7226df3ccf"
#: B's implementation tip: 60ce31dc ("T-0208 B fix: drop the two H1 captions (B2); re-pin H1 image sha"),
#: B's last implementation commit; closes to M^2 at the merge.
TIP = "60ce31dc8b3254adc67d95b8806026b9bdbd5779"
#: B10's list, written out from the task.
FROZEN = (
    "local_astra/pages/scenarios",
    "local_astra/pages/scenario_tasks.py",
    "local_astra/pages/server.py",
    "local_astra/pages/static",
    "local_astra/pages/suite",
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/contracts",
    "tests/fakes",
)
TESTS_OWNED = frozenset(
    {
        "tests/unit/test_t0208_handoff_judge.py",
        "tests/unit/test_t0208_handoff_static.py",
        "tests/integration/test_t0208_handoff_session.py",
        "tests/unit/test_guards_use_fixed_refs.py",
    }
)
#: The power range: T-0205 group B (a4ba3cae..d4ad7597) changed ``local_astra/pages/scenarios``.
POWER_BASE = "a4ba3cae01da9bb4731124e5ebb068d87140b41a"
POWER_TIP = "d4ad7597f1ee7ec2d587d37bd8977792e2dbe338"
NETWORK_MODULES = frozenset({"socket", "urllib", "http", "httpx", "requests", "ddgs", "aiohttp"})


# =====================================================================================================
# helpers
# =====================================================================================================


def ho() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0208 B1: {MODULE} does not exist yet") from exc
        raise


def table() -> dict[str, Any]:
    m = ho()
    assert hasattr(m, "HANDOFFS"), f"T-0208 B1: {MODULE} has no HANDOFFS"
    return {r.handoff_id: r for r in m.HANDOFFS}


def image_path(hid: str) -> Path:
    p = HANDOFF_DIR / IMAGES[hid]
    assert p.is_file(), f"T-0208 B2: {p.relative_to(REPO)} does not exist yet"
    return p


def sof_size(data: bytes) -> tuple[int, int, int]:
    """(marker, width, height) of the first SOF segment, walking the JPEG segments by hand."""
    i = 2
    while i + 4 <= len(data):
        assert data[i] == 0xFF, f"not a marker at {i}: {data[i]:#x}"
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return marker, w, h
        if marker == 0xDA:
            break
        i += 2 + length
    raise AssertionError("no SOF0/SOF1/SOF2 segment before the scan")


# =====================================================================================================
# B2: the images
# =====================================================================================================


@pytest.mark.parametrize("hid", tuple(IMAGES))
def test_image_sha_matches_the_table(hid: str) -> None:
    """B2: each committed image's sha256 is the table's ``image_sha256`` for that row, and the row names that file."""
    row = table()[hid]
    assert row.image == IMAGES[hid], (hid, row.image)
    data = image_path(hid).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    assert got == row.image_sha256, f"T-0208 B2 {hid}: file sha {got} != table {row.image_sha256}"


@pytest.mark.parametrize("hid", tuple(IMAGES))
def test_image_is_a_1280x800_jpeg(hid: str) -> None:
    """B2: SOI ``FFD8FF`` first, EOI ``FFD9`` last; the SOF segment says 1280x800; Pillow decodes it as an RGB JPEG
    of that size."""
    from PIL import Image

    data = image_path(hid).read_bytes()
    assert data[:3] == b"\xff\xd8\xff", data[:4]
    assert data[-2:] == b"\xff\xd9", data[-4:]
    _marker, w, h = sof_size(data)
    assert (w, h) == (W, H), f"T-0208 B2 {hid}: SOF says {w}x{h}"
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "JPEG" and im.size == (W, H) and im.mode == "RGB", (
            im.format,
            im.size,
            im.mode,
        )
        im.load()


@pytest.mark.parametrize("hid", tuple(IMAGES))
def test_image_quality_85(hid: str) -> None:
    """B2 "quality 85": the file's quantisation tables equal those Pillow writes at ``quality=85`` for an RGB image
    of the same size (an independent encoder run). Power: a quality-75 encode differs."""
    from PIL import Image

    def tables(raw: bytes) -> dict[int, list[int]]:
        with Image.open(io.BytesIO(raw)) as im:
            q = getattr(im, "quantization", None)
            assert q, "no quantisation tables"
            return {k: list(v) for k, v in q.items()}

    def encode(quality: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (W, H), (200, 120, 40)).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    want = tables(encode(QUALITY))
    assert tables(encode(75)) != want, "power: a different quality gives different tables"
    got = tables(image_path(hid).read_bytes())
    assert got == want, f"T-0208 B2 {hid}: the quantisation tables are not quality {QUALITY}'s"


def test_images_fit_the_hud_screen_bucket() -> None:
    """B2 "inside the HUD's screen bucket": the HUD would send each image unscaled (``target_size`` of the screen
    bucket for its size is its size)."""
    from local_astra.contracts.perception import Source
    from local_astra.perception import policy

    for hid in IMAGES:
        _m, w, h = sof_size(image_path(hid).read_bytes())
        bucket = policy.bucket_for(Source.SCREEN, w, h)
        assert policy.target_size(w, h, bucket) == (w, h), (hid, bucket)


def test_directory_contents_and_manifest() -> None:
    """B2: the directory holds exactly the three JPEGs and ``MANIFEST.md``; the manifest names each image, states
    original work and Apache-2.0, and names the generator script."""
    assert HANDOFF_DIR.is_dir(), f"T-0208 B2: {HANDOFF_DIR.relative_to(REPO)} does not exist yet"
    names = sorted(
        p.relative_to(HANDOFF_DIR).as_posix() for p in HANDOFF_DIR.rglob("*") if p.is_file()
    )
    assert names == sorted([*IMAGES.values(), "MANIFEST.md"]), names
    text = (HANDOFF_DIR / "MANIFEST.md").read_text(encoding="utf-8")
    for img in IMAGES.values():
        assert img in text, f"T-0208 B2: MANIFEST.md does not name {img}"
    assert re.search(r"(?i)\boriginal\b", text), "T-0208 B2: MANIFEST.md does not say original work"
    assert "Apache-2.0" in text, "T-0208 B2: MANIFEST.md does not state the licence"
    assert "scripts/make_t0208_shared_images.py" in text, "MANIFEST.md does not name the generator"


def test_generator_script_uses_pillow_and_no_network() -> None:
    """B2: ``scripts/make_t0208_shared_images.py`` exists, imports Pillow, imports no network module, and names
    ``local_astra/pages/handoff`` as where it writes."""
    assert SCRIPT.is_file(), f"T-0208 B2: {SCRIPT.relative_to(REPO)} does not exist yet"
    src = SCRIPT.read_text(encoding="utf-8")
    tops: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.add(node.module.split(".")[0])
    assert "PIL" in tops, sorted(tops)
    assert not (tops & NETWORK_MODULES), sorted(tops & NETWORK_MODULES)
    assert re.search(r"pages[\"'/, ]+handoff", src), "the script does not name pages/handoff"


def test_h1_drawing_names_no_part() -> None:
    """B2 (H1, adversarial): the only text B2 lists for H1 is the label plate "Barista Express BES870"; the broken
    part is shown by a drawing in a red circle, never named. So every ``text(`` call in the generator's ``h1()`` draws
    a string literal, the multiset of them is exactly the label plate, and none names the part or the breakage (a
    caption such as "steam wand snapped off" hands the shared arm the answer). Static on the script, because the
    committed bytes are pinned by sha and cannot be OCR'd here."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "h1"]
    assert len(fns) == 1, "T-0208 B2: the generator has no single top-level h1()"
    texts: list[str] = []
    for node in ast.walk(fns[0]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"text", "multiline_text"}
        ):
            arg = node.args[1] if len(node.args) > 1 else None
            if arg is None:
                arg = next((k.value for k in node.keywords if k.arg == "text"), None)
            assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), (
                f"T-0208 B2: h1() draws a text that is not a string literal (line {node.lineno})"
            )
            texts.append(arg.value)
    assert texts == ["Barista Express BES870"], (
        f"T-0208 B2: H1's only text is the label plate; h1() draws {texts!r}"
    )
    named = [t for t in texts if re.search(r"steam|wand|froth|nozzle|snap|broke|break", t, re.I)]
    assert named == [], f"T-0208 B2: H1 caption names the part: {named!r}"


def test_neither_root_serves_the_images() -> None:
    """B2: a real ``StartPageServer()`` and ``StartPageServer(static_dir=SCENARIO_DIR)`` answer 404 for every
    spelling of a handoff path, and a traversal spelling never answers 200 or the image bytes. Premise: the images
    exist (so the row is red before they do)."""
    from local_astra.pages.scenario_tasks import SCENARIO_DIR

    datas = {hid: image_path(hid).read_bytes() for hid in IMAGES}
    plain: list[str] = ["/handoff/", "/handoff/MANIFEST.md"]
    traversal: list[str] = []
    for img in IMAGES.values():
        plain += [f"/handoff/{img}", f"/{img}", f"/pages/handoff/{img}"]
        traversal += [f"/../handoff/{img}", f"/%2e%2e/handoff/{img}", f"/..%2fhandoff/{img}"]
    for root in (None, SCENARIO_DIR):
        srv = new_server(root)
        srv.start()
        try:
            port = port_of(srv.origin)
            for p in plain:
                status, body, _ = request(port, "GET", p)
                assert status == 404, f"T-0208 B2: root {root} answers {status} for {p}"
                assert all(d not in body for d in datas.values()), p
            for p in traversal:
                status, body, _ = request(port, "GET", p)
                assert status != 200 and all(d not in body for d in datas.values()), (
                    root,
                    p,
                    status,
                )
        finally:
            srv.stop()


# =====================================================================================================
# B8: the Make target
# =====================================================================================================


def _make_lines(target: str) -> tuple[int, list[str], str]:
    proc = subprocess.run(
        ["make", "-n", "-s", target],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = [ln.split("#", 1)[0].strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return proc.returncode, lines, proc.stderr


def test_make_browser_handoff_target() -> None:
    """B8: ``make -n -s browser-handoff`` prints exactly B8's command and the target is ``.PHONY`` (written by the
    orchestrator; red until it is)."""
    code, lines, err = _make_lines("browser-handoff")
    assert code == 0, f"T-0208 B8: no browser-handoff target: {err}"
    assert lines == [MAKE_BROWSER_HANDOFF], lines
    phony = [ln for ln in (REPO / "Makefile").read_text().splitlines() if ln.startswith(".PHONY:")]
    assert any("browser-handoff" in ln.split() for ln in phony), phony


# =====================================================================================================
# B10 (green at BASE)
# =====================================================================================================


def _makefile_at(rev: str) -> list[str]:
    proc = gr._git("show", f"{rev}:Makefile")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_frozen_paths_over_closed_range() -> None:
    """B10 (green at BASE): B10's paths are byte-identical over ``BASE..TIP`` (two literals, never the working tree),
    each exists at BASE; under ``tests/`` only B's three modules and ``test_guards_use_fixed_refs`` change; the
    Makefile changes only by lines naming ``browser-handoff``. Power: the same predicate sees T-0205 B's change to
    ``local_astra/pages/scenarios``."""
    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE[:7]} is not an ancestor of {TIP[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE, TIP, *FROZEN)
    assert changed == [], f"T-0208 B10: frozen paths differ in {BASE[:7]}..{TIP[:7]}: {changed}"
    tests_changed = [p for p in gr.changed(BASE, TIP, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0208 B10: other test files edited: {tests_changed}"
    before, after = _makefile_at(BASE), _makefile_at(TIP)
    removed = [ln for ln in before if ln not in after]
    added = [ln for ln in after if ln not in before]
    assert all(ln.startswith(".PHONY:") for ln in removed), removed
    assert all("browser-handoff" in ln for ln in added), added
    gr.require_commits(POWER_BASE, POWER_TIP)
    assert gr.changed(POWER_BASE, POWER_TIP, "local_astra/pages/scenarios"), (
        "power: the predicate cannot see a change"
    )
