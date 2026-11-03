"""T-0131 B2-B4 (pure half): ``local_astra/toolcall/reground.py`` (docs/tasks/T-0131.md, group B).

**The interface these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``ZOOM == 0.4``;
- ``OUTCOMES``: the seven B3 labels ``ok``, ``pass2_unparseable``, ``pass2_no_point``, ``pass2_other_tool``,
  ``pass2_timeout``, ``error``, ``skipped_no_point``;
- ``applies(name: str) -> bool``: ``True`` for ``click_at`` and ``type_at`` only (B1);
- ``crop_rect(frame_w, frame_h, px, py) -> (x0, y0, w, h)``: zoom 0.4 of the frame, centred on pass 1's pixel point
  ``(px, py)`` (``to_pixels`` of its 0-999 values) and clamped inside the frame;
- ``map_point(x2, y2, rect, frame_w, frame_h) -> (x, y)``: pass 2's 0-999 point in the crop mapped back to the full
  frame's 0-999;
- ``decide(name, arguments, pass2_tool, rect, frame_w, frame_h) -> Decision`` with ``.outcome`` (one of the four
  parse outcomes) and ``.arguments`` (what executes): ``pass2_tool`` is pass 2's ``tool`` commit text as the client
  yields it (``<|tool_call>call:...<tool_call|>``) or ``None`` for no call;
- ``replace_frame_image(prompt, identity, crop_jpeg) -> Prompt``: every ``ImagePart`` whose ``phash`` is
  ``identity`` becomes ``data:image/jpeg;base64,<b64(crop_jpeg)>`` (its own phash is the implementer's); every other
  part and field unchanged;
- ``crop_jpeg(frame_jpeg: bytes, rect) -> bytes``: the region as a JPEG at its native size; PIL is imported inside
  it only.

Oracles: hand-worked points (each docstring), ``tool_executor.to_pixels`` (the executor's own formula, which B4 is
stated in), and a hand-built prompt. Hypothesis runs with ``database=None`` (tests.md).

Readings (reported): B4's "within 1 px" is achievable only while one 0-999 step is at most 2 px, i.e. a side of at
most 1999 px; the property draws sides of 64-1999 px (the 1280x800 viewport is well inside). "Centred" allows the
crop's centre to be within 1 px of the point (integer halves). Corners are pinned exactly: a crop flush with the
frame's edge maps pass 2's 0 and 999 to the frame's 0 and 999, so a mapping scaled by ``W`` instead of ``W - 1``
(mutation 12) is caught.

Every row is ``pending(task="T-0131")``: at BASE the module does not exist, and the import helper turns that into
a named ``AssertionError``.

Regime: CPU, pure; Hypothesis ``database=None``, default example counts; no measurement.
"""

from __future__ import annotations

import ast
import base64
import importlib
import inspect
import io
import subprocess
import sys
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from tests.unit._live_browser_support import jpeg_bytes

Q = '<|"|>'
W, H = 1280, 800
OUTCOMES = (
    "ok",
    "pass2_unparseable",
    "pass2_no_point",
    "pass2_other_tool",
    "pass2_timeout",
    "error",
    "skipped_no_point",
)


def rg() -> Any:
    try:
        return importlib.import_module("local_astra.toolcall.reground")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "T-0131 B2: local_astra/toolcall/reground.py does not exist yet"
        ) from exc


def need(name: str) -> Any:
    mod = rg()
    assert hasattr(mod, name), f"T-0131 B2: reground.{name} does not exist yet"
    return getattr(mod, name)


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


def type_at(x: int, y: int, text: str = "mug", enter: bool = True) -> str:
    e = "true" if enter else "false"
    return f"<|tool_call>call:type_at{{press_enter:{e},text:{Q}{text}{Q},x:{x},y:{y}}}<tool_call|>"


SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
PRESS = f"<|tool_call>call:press_key{{combo:{Q}Enter{Q}}}<tool_call|>"
SEARCH = f"<|tool_call>call:google_search{{query:{Q}blue mug{Q}}}<tool_call|>"


def full_px(rect: tuple[int, int, int, int], x2: int, y2: int) -> tuple[int, int]:
    """B4's target: the crop pixel pass 2 names, in full-frame pixels."""
    x0, y0, w, h = rect
    return x0 + to_pixels(x2, w), y0 + to_pixels(y2, h)


def assert_maps(
    rect: tuple[int, int, int, int], x2: int, y2: int, got: Any, fw: int, fh: int
) -> None:
    x, y = got
    assert isinstance(x, int) and isinstance(y, int) and 0 <= x <= 999 and 0 <= y <= 999, got
    tx, ty = full_px(rect, x2, y2)
    assert abs(to_pixels(x, fw) - tx) <= 1, (
        f"B4: x {x} -> {to_pixels(x, fw)} px, target {tx} (rect {rect})"
    )
    assert abs(to_pixels(y, fh) - ty) <= 1, (
        f"B4: y {y} -> {to_pixels(y, fh)} px, target {ty} (rect {rect})"
    )


# --------------------------------------------------------------------------------------------- B1/B3: names


def test_constants_and_applies() -> None:
    """B1/B3: ``ZOOM`` is 0.4, ``OUTCOMES`` the seven labels, and only ``click_at``/``type_at`` are regrounded."""
    assert need("ZOOM") == 0.4
    assert tuple(need("OUTCOMES")) == OUTCOMES or set(need("OUTCOMES")) == set(OUTCOMES)
    applies = need("applies")
    assert applies("click_at") and applies("type_at")
    for name in ("scroll", "press_key", "google_search", "nonsense"):
        assert not applies(name), name


# --------------------------------------------------------------------------------------------- B2: the crop rect


@pytest.mark.parametrize(
    ("px", "want"),
    [
        ((640, 200), (384, 40, 512, 320)),
        ((0, 0), (0, 0, 512, 320)),
        ((1279, 799), (768, 480, 512, 320)),
        ((100, 700), (0, 480, 512, 320)),
        ((1279, 0), (768, 0, 512, 320)),
    ],
    ids=["interior", "top-left", "bottom-right", "left-bottom", "top-right"],
)
def test_crop_rect_hand_values(px: tuple[int, int], want: tuple[int, int, int, int]) -> None:
    """B2: 1280x800 gives a 512x320 crop. Pass 1's (500, 250) is (640, 200) px (``to_pixels``), so the crop is
    (640-256, 200-160) = (384, 40), within 1 px; at the corners and edges it is clamped flush inside the frame."""
    got = tuple(need("crop_rect")(W, H, px[0], px[1]))
    assert got[2:] == want[2:], f"B2: 512x320 at 1280x800; got {got}"
    if want[:2] == (384, 40):
        assert abs(got[0] - 384) <= 1 and abs(got[1] - 40) <= 1, got
    else:
        assert got[:2] == want[:2], f"B2: clamped flush; got {got}, want {want}"


@settings(database=None, deadline=None)
@given(
    fw=st.integers(64, 3840),
    fh=st.integers(64, 2160),
    fx=st.floats(0, 1),
    fy=st.floats(0, 1),
)
def test_crop_rect_clamped_and_sized(fw: int, fh: int, fx: float, fy: float) -> None:
    """B2 (property): the crop lies inside the frame, its sides are 0.4 of the frame's (within 1 px), and its
    centre is within 1 px of the point unless the crop is clamped against an edge."""
    px, py = min(fw - 1, int(fx * fw)), min(fh - 1, int(fy * fh))
    x0, y0, w, h = need("crop_rect")(fw, fh, px, py)
    assert x0 >= 0 and y0 >= 0 and x0 + w <= fw and y0 + h <= fh, (fw, fh, px, py, (x0, y0, w, h))
    assert abs(w - 0.4 * fw) <= 1 and abs(h - 0.4 * fh) <= 1, (fw, fh, w, h)
    if 0 < x0 < fw - w:
        assert abs(x0 + w / 2 - px) <= 1, (px, x0, w)
    if 0 < y0 < fh - h:
        assert abs(y0 + h / 2 - py) <= 1, (py, y0, h)
    if x0 == 0 and px > w / 2 + 1:
        raise AssertionError(f"clamped left with room to centre: {px}, {w}")
    if x0 == fw - w and px < fw - w / 2 - 1:
        raise AssertionError(f"clamped right with room to centre: {px}, {w}")


# --------------------------------------------------------------------------------------------- B4: the mapping


def test_mapping_hand_values() -> None:
    """B4 by hand: rect (384, 40, 512, 320) and pass 2's (900, 100): ``to_pixels(900, 512)`` =
    round(900·511/999) = 460, ``to_pixels(100, 320)`` = round(100·319/999) = 32, so the target is (844, 72) px;
    659 -> round(659·1279/999) = 844 and 90 -> round(90·799/999) = 72, so a mapping within 1 px exists."""
    rect = (384, 40, 512, 320)
    assert_maps(rect, 900, 100, need("map_point")(900, 100, rect, W, H), W, H)


@pytest.mark.parametrize(
    ("rect", "x2", "y2", "want"),
    [
        ((768, 480, 512, 320), 999, 999, (999, 999)),
        ((0, 0, 512, 320), 0, 0, (0, 0)),
        ((768, 0, 512, 320), 999, 0, (999, 0)),
    ],
    ids=["bottom-right", "top-left", "top-right"],
)
def test_mapping_corners_exact(
    rect: tuple[int, int, int, int], x2: int, y2: int, want: tuple[int, int]
) -> None:
    """B4 (mutation 12): a crop flush with the frame's edge maps pass 2's edge to the frame's edge exactly: the
    crop's last pixel (768 + 511 = 1279) is the frame's last, which only 999 reaches (998 -> 1278). A mapping
    scaled by ``W`` instead of ``W - 1`` gives 998."""
    assert tuple(need("map_point")(x2, y2, rect, W, H)) == want


@settings(database=None, deadline=None)
@given(
    fw=st.integers(64, 1999),
    fh=st.integers(64, 1999),
    x1=st.integers(0, 999),
    y1=st.integers(0, 999),
    x2=st.integers(0, 999),
    y2=st.integers(0, 999),
)
def test_mapping_within_one_px(fw: int, fh: int, x1: int, y1: int, x2: int, y2: int) -> None:
    """B4 (property): for every frame size (sides 64-1999 px, see the docstring), the crop of any pass-1 point and
    any pass-2 point, the mapped value is in 0-999 and ``to_pixels`` of it is within 1 px of
    ``x0 + to_pixels(x2, w)`` (and likewise for y)."""
    rect = tuple(need("crop_rect")(fw, fh, to_pixels(x1, fw), to_pixels(y1, fh)))
    assert_maps(rect, x2, y2, need("map_point")(x2, y2, rect, fw, fh), fw, fh)  # type: ignore[arg-type]


def test_mapping_interior_hand_value() -> None:
    """B4, a deterministic interior row (reviewer B finding 1 at 68cd08f: the property alone let the crop-side
    ``to_pixels(x2, w + 1)`` mutant through on most seeds). Rect (384, 40, 512, 320) at 1280x800, pass 2's
    (83, 500), by hand: ``to_pixels(83, 512)`` = round(83·511/999) = round(42.46) = 42, so the x target is
    384 + 42 = **426 px**; 333 -> round(333·1279/999) = round(426.33) = 426, so a 1-px answer exists.
    ``to_pixels(500, 320)`` = round(500·319/999) = round(159.66) = 160, so the y target is 40 + 160 = **200 px**;
    250 -> round(250·799/999) = round(199.95) = 200."""
    rect = (384, 40, 512, 320)
    x, y = need("map_point")(83, 500, rect, W, H)
    assert isinstance(x, int) and isinstance(y, int) and 0 <= x <= 999 and 0 <= y <= 999, (x, y)
    assert abs(to_pixels(x, W) - 426) <= 1, f"B4: x {x} -> {to_pixels(x, W)} px, target 426"
    assert abs(to_pixels(y, H) - 200) <= 1, f"B4: y {y} -> {to_pixels(y, H)} px, target 200"


def test_mapping_sweep_fixed_rect() -> None:
    """B4, every pass-2 coordinate at one fixed rect (reviewer B finding 1): with rect (384, 40, 512, 320) at
    1280x800, each x2 in 0..999 (y2 = 500) and each y2 in 0..999 (x2 = 500) maps within 1 px of
    ``x0 + to_pixels(x2, w)`` (resp. ``y0 + to_pixels(y2, h)``), ``to_pixels`` being the executor's own formula.
    Deterministic, so a crop-side ``w + 1``/``h + 1`` mutant fails on every run, not on a lucky draw."""
    rect = (384, 40, 512, 320)
    map_point = need("map_point")
    bad_x = []
    bad_y = []
    for v in range(1000):
        x, _ = map_point(v, 500, rect, W, H)
        if not (0 <= x <= 999 and abs(to_pixels(x, W) - (384 + to_pixels(v, 512))) <= 1):
            bad_x.append((v, x))
        _, y = map_point(500, v, rect, W, H)
        if not (0 <= y <= 999 and abs(to_pixels(y, H) - (40 + to_pixels(v, 320))) <= 1):
            bad_y.append((v, y))
    assert not bad_x, f"B4: x2 values off by more than 1 px: {len(bad_x)}, first {bad_x[:5]}"
    assert not bad_y, f"B4: y2 values off by more than 1 px: {len(bad_y)}, first {bad_y[:5]}"


def test_stop_texts_are_the_stop_specials() -> None:
    """Reviewer B at 68cd08f: ``reground._STOP_TEXTS`` (a local copy, since the module may not import ``live``) is
    exactly the literals of the pinned stop ids, ``{SPECIAL_IDS[i] for i in STOP_IDS}``
    (``toolcall/special_tokens.py``), so a stop added to or dropped from the table cannot drift past pass 2's
    parser."""
    from local_astra.toolcall.special_tokens import SPECIAL_IDS, STOP_IDS

    want = {SPECIAL_IDS[i] for i in STOP_IDS}
    assert len(want) == 3, "premise: three stop ids"
    assert set(need("_STOP_TEXTS")) == want, (sorted(need("_STOP_TEXTS")), sorted(want))


# --------------------------------------------------------------------------------------------- B3: the outcomes

RECT = (384, 40, 512, 320)
P1_CLICK = ("click_at", {"x": 500, "y": 250})
P1_TYPE = ("type_at", {"press_enter": True, "text": "mug", "x": 500, "y": 250})


@pytest.mark.parametrize(
    ("p1", "pass2", "outcome"),
    [
        (P1_CLICK, click(900, 100), "ok"),
        (P1_TYPE, type_at(900, 100, "cup", enter=False), "ok"),
        (P1_CLICK, None, "pass2_unparseable"),
        (P1_CLICK, "<|tool_call>call:click_at{x:12<tool_call|>", "pass2_unparseable"),
        (P1_CLICK, "<|tool_call>call:click_at{x:1200,y:5}<tool_call|>", "pass2_unparseable"),
        (P1_CLICK, "<|tool_call>call:open_url{url:1}<tool_call|>", "pass2_unparseable"),
        (P1_CLICK, SCROLL, "pass2_no_point"),
        (P1_CLICK, PRESS, "pass2_no_point"),
        (P1_TYPE, SEARCH, "pass2_no_point"),
        (P1_CLICK, type_at(900, 100), "pass2_other_tool"),
        (P1_TYPE, click(900, 100), "pass2_other_tool"),
    ],
    ids=[
        "ok-click",
        "ok-type",
        "none",
        "truncated",
        "out-of-range",
        "unknown-tool",
        "scroll",
        "press_key",
        "search",
        "click-then-type",
        "type-then-click",
    ],
)
def test_outcome_table(p1: tuple[str, dict[str, Any]], pass2: str | None, outcome: str) -> None:
    """B3, one row per parse outcome (timeout, error and skipped_no_point are the session's; see
    ``test_session_reground.py``): ``ok`` executes pass 1's call with only x/y replaced by the mapped point;
    every other outcome executes pass 1's arguments unchanged (mutations 5, 6)."""
    name, args = p1
    d = need("decide")(name, dict(args), pass2, RECT, W, H)
    assert d.outcome == outcome, (pass2, d.outcome)
    if outcome == "ok":
        assert set(d.arguments) == set(args)
        assert_maps(RECT, 900, 100, (d.arguments["x"], d.arguments["y"]), W, H)
    else:
        assert dict(d.arguments) == args, (
            f"B3: {outcome} executes pass 1 unchanged; got {d.arguments}"
        )


def test_ok_keeps_name_text_press_enter() -> None:
    """B3 ``ok`` (mutation 5): a ``type_at`` pass 2 with other text and ``press_enter`` still executes pass 1's
    ``text`` and ``press_enter``; only x and y move."""
    name, args = P1_TYPE
    d = need("decide")(name, dict(args), type_at(900, 100, "SECRET-PASS2", enter=False), RECT, W, H)
    assert d.outcome == "ok"
    assert d.arguments["text"] == "mug" and d.arguments["press_enter"] is True, d.arguments
    assert (d.arguments["x"], d.arguments["y"]) != (500, 250), "premise: the point moved"
    assert "SECRET-PASS2" not in repr(d), "pass 2's text is never carried"


# --------------------------------------------------------------------------------------------- B2: the prompt


def _img(url: str, phash: str) -> ImagePart:
    return ImagePart(url=url, phash=phash)


def test_replaces_only_call_frame_image() -> None:
    """B2 / R-a (mutation 8's prompt half): with two history images, the call's frame riding on the newest history
    image (no current image) and a second current image, only the part whose phash is the call's identity becomes
    the crop; every other byte of the prompt (messages, prefix_len, tools, thinking) is unchanged."""
    replace = need("replace_frame_image")
    crop = jpeg_bytes(512, 320, seed=7)
    crop_url = "data:image/jpeg;base64," + base64.b64encode(crop).decode()
    h1, h2, cur = (
        "data:image/jpeg;base64,AAAA",
        "data:image/jpeg;base64,BBBB",
        "data:image/jpeg;base64,CCCC",
    )
    prompt = Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="sys"),)),
            Message(role="user", content=(_img(h1, "h1" * 8), TextPart(text="first"))),
            Message(role="assistant", content=(TextPart(text="A shop."),)),
            Message(role="user", content=(_img(h2, "call" * 4), TextPart(text="second"))),
            Message(role="assistant", content=(TextPart(text="A cart."),)),
            Message(
                role="user", content=(_img(cur, "cur" * 5 + "x"), TextPart(text="add the blue mug"))
            ),
        ),
        prefix_len=6,
    )
    got = replace(prompt, "call" * 4, crop)
    assert got.prefix_len == prompt.prefix_len and got.thinking == prompt.thinking
    assert getattr(got, "tools", None) == getattr(prompt, "tools", None)
    assert len(got.messages) == len(prompt.messages)
    for i, (g, o) in enumerate(zip(got.messages, prompt.messages, strict=True)):
        if i != 3:
            assert g == o, f"message {i} unchanged"
    third = got.messages[3]
    assert third.role == "user" and len(third.content) == 2
    assert isinstance(third.content[0], ImagePart) and third.content[0].url == crop_url
    assert third.content[1] == TextPart(text="second")
    assert replace(prompt, "absent" * 3, crop) == prompt, (
        "no part with that identity: the prompt comes back equal"
    )


# --------------------------------------------------------------------------------------------- B2: the JPEG crop


def test_crop_jpeg_native_size_and_content() -> None:
    """B2: the crop is a JPEG of exactly w x h, cut from the frame's own bytes: its centre pixel matches the frame's
    pixel at the same absolute place (solid 64-px blocks, so JPEG loss stays far below the tolerance)."""
    from PIL import Image

    frame = jpeg_bytes(W, H, seed=3)
    rect = (384, 40, 512, 320)
    out = need("crop_jpeg")(frame, rect)
    assert isinstance(out, bytes) and out[:2] == b"\xff\xd8", "a JPEG"
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and img.size == (512, 320)
    src = Image.open(io.BytesIO(frame)).convert("RGB")
    crop_rgb = img.convert("RGB")
    # block centres of the frame (bx*64+32, by*64+32) that fall inside the rect, read at the same absolute place
    for fx, fy in ((6 * 64 + 32, 1 * 64 + 32), (10 * 64 + 32, 4 * 64 + 32)):
        a = crop_rgb.getpixel((fx - 384, fy - 40))
        b = src.getpixel((fx, fy))
        assert isinstance(a, tuple) and isinstance(b, tuple)
        assert all(abs(int(p) - int(q)) <= 12 for p, q in zip(a, b, strict=True)), (a, b)


def test_imports_pure() -> None:
    """Owned files: ``reground.py`` imports only the contracts, ``toolcall.native``/``score``/``tools``,
    ``conductor.tool_executor`` and the standard library at module level, and importing it loads no PIL (the crop
    helper imports it lazily) and no asyncio, clock or HTTP module."""
    mod = rg()
    tree = ast.parse(inspect.getsource(mod))
    allowed_local = (
        "local_astra.contracts",
        "local_astra.toolcall.native",
        "local_astra.toolcall.score",
        "local_astra.toolcall.tools",
        "local_astra.conductor.tool_executor",
    )
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("local_astra")
        ):
            # ``from local_astra.toolcall import native`` names ``local_astra.toolcall.native``
            full = [f"{node.module}.{a.name}" for a in node.names]
            ok = node.module.startswith(allowed_local) or all(
                f.startswith(allowed_local) for f in full
            )
            assert ok, f"reground.py imports {node.module}: {[a.name for a in node.names]}"
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            )
            for n in names:
                assert n is None or n.split(".")[0] not in ("PIL", "asyncio", "httpx", "time"), n
    probe = (
        "import sys, local_astra.toolcall.reground as r; "
        "bad = [m for m in ('PIL', 'asyncio', 'httpx') if m in sys.modules and m not in BEFORE]; "
        "print(bad)"
    )
    code = "import sys; BEFORE = set(sys.modules); " + probe
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "PIL" not in out.stdout, f"importing reground loads PIL: {out.stdout}"
