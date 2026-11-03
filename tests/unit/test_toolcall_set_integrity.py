"""T-0080b criterion 1: integrity of the tool-call eval set v1 (``tests/eval/toolcall/``). Unit, hermetic.

Reads only committed files (``cases.jsonl``, ``MANIFEST.json``, ``screens/``); never renders, never opens a socket.

- counts per tool and split: 120 tool cases (30 google_search, 30 click_at, 30 type_at, 15 scroll, 15 press_key)
  plus 20 no-tool; dev 40 (+8 no-tool) and test 80 (+12 no-tool), stratified 1:2 per tool;
- the MANIFEST: the cases file's sha256, the split's sha256 (recomputed from the cases *and* pinned here, so a
  silent re-split needs an edit to this file), every screen file's sha256, the generator's sha256;
- every screen is a 1280 x 800 JPEG (dimensions parsed from the SOF segment, an independent oracle), every bbox
  is an inclusive 0-999 box inside the frame;
- every ground truth has the shape ``local_astra.toolcall.score.exact`` reads, and the case's reference call is
  schema-valid under ``tools_v1.json`` and exact under it -- while each of a handful of one-field perturbations of
  it is *not* exact, so no ground truth is vacuous;
- dev and test are disjoint (cases and screens), and the HTML the screens were rendered from loads nothing from
  the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import score, tools

HERE = (
    Path(__file__).resolve().parents[1] / "eval" / "toolcall"
)  # the set stays beside its generator (ruling R-T46)
CASES = HERE / "cases.jsonl"
MANIFEST = HERE / "MANIFEST.json"

#: The split, pinned. Recomputed from cases.jsonl as sha256 of
#: ``json.dumps({"dev": sorted ids, "test": sorted ids}, sort_keys=True, separators=(",", ":"))``.
SPLIT_SHA256 = "7dfa793b70ec237abdaba68c6b3540b19ed9c6243b04a12190d7f8d33a46225c"

TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")
EXPECTED = {
    "dev": {
        "google_search": 10,
        "click_at": 10,
        "type_at": 10,
        "scroll": 5,
        "press_key": 5,
        "no_tool": 8,
    },
    "test": {
        "google_search": 20,
        "click_at": 20,
        "type_at": 20,
        "scroll": 10,
        "press_key": 10,
        "no_tool": 12,
    },
}
GT_KEYS = {
    "google_search": {"name", "keywords"},
    "click_at": {"name", "bbox"},
    "type_at": {"name", "bbox", "text", "press_enter"},
    "scroll": {"name", "direction"},
    "press_key": {"name", "combo"},
}
FRAME = (1280, 800)


def _cases() -> list[dict[str, Any]]:
    return [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]


def _manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MANIFEST.read_text())
    return data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jpeg_size(data: bytes) -> tuple[int, int]:
    """``(width, height)`` from the first SOFn segment; a hand-written JPEG walk, no imaging library."""
    assert data[:2] == b"\xff\xd8", "not a JPEG (no SOI marker)"
    i = 2
    while i < len(data):
        assert data[i] == 0xFF, f"bad marker at {i}"
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return w, h
        i += 2 + length
    raise AssertionError("no SOF segment")


TOOLSET = tools.load()


# ----------------------------------------------------------------------------------------- counts


def test_counts_per_tool_and_split() -> None:
    cases = _cases()
    got = {
        s: {
            **{t: sum(1 for c in cases if c["split"] == s and c["tool"] == t) for t in TOOLS},
            "no_tool": sum(1 for c in cases if c["split"] == s and c["kind"] == "no_tool"),
        }
        for s in ("dev", "test")
    }
    assert got == EXPECTED
    assert len(cases) == 140
    assert {c["split"] for c in cases} == {"dev", "test"}
    assert {c["kind"] for c in cases} == {"tool", "no_tool"}
    for c in cases:
        assert (c["kind"] == "tool") == (c["tool"] in TOOLS), c["id"]


def test_ids_and_requests_are_unique() -> None:
    cases = _cases()
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids), [k for k, v in Counter(ids).items() if v > 1]
    users = [c["user"] for c in cases]
    assert len(set(users)) == len(users), [k for k, v in Counter(users).items() if v > 1]
    assert all(isinstance(u, str) and u.strip() for u in users)


# --------------------------------------------------------------------------------------- manifest


def test_manifest_pins_the_cases_file_and_the_split() -> None:
    m = _manifest()
    assert m["cases_sha256"] == _sha(CASES)
    cases = _cases()
    split = {s: sorted(c["id"] for c in cases if c["split"] == s) for s in ("dev", "test")}
    recomputed = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recomputed == m["split_sha256"]
    assert recomputed == SPLIT_SHA256, (
        "the split changed: re-pin SPLIT_SHA256 only with a reviewed reason"
    )
    assert m["counts"] == EXPECTED


def test_manifest_lists_every_screen_file_with_its_sha() -> None:
    m = _manifest()
    on_disk = {p.relative_to(HERE).as_posix() for p in (HERE / "screens").rglob("*") if p.is_file()}
    assert set(m["files"]) == on_disk
    for rel, sha in m["files"].items():
        assert _sha(HERE / rel) == sha, rel
    referenced = {c["screen"] for c in _cases() if c["screen"]}
    assert referenced <= set(m["files"])


def test_manifest_pins_the_generator() -> None:
    m = _manifest()
    assert set(m["generator"]) == {"build.py", "screens.py", "cases_src.py"}
    for name, sha in m["generator"].items():
        assert _sha(HERE / name) == sha, f"{name} changed after the set was built: re-run build.py"


# ----------------------------------------------------------------------------------------- screens


def test_every_screen_is_a_1280_long_side_jpeg_of_the_declared_size() -> None:
    for c in _cases():
        if c["screen"] is None:
            assert c["screen_size"] is None, c["id"]
            continue
        data = (HERE / c["screen"]).read_bytes()
        assert jpeg_size(data) == FRAME, c["id"]
        assert tuple(c["screen_size"]) == FRAME, c["id"]
        assert max(FRAME) == 1280


def test_computer_control_cases_carry_a_screen_and_search_cases_do_not() -> None:
    for c in _cases():
        if c["tool"] in ("click_at", "type_at", "scroll", "press_key"):
            assert c["screen"] is not None, c["id"]
        if c["tool"] == "google_search":
            assert c["screen"] is None, c["id"]


def _bbox_ok(b: Any) -> bool:
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(v, int) and not isinstance(v, bool) for v in b)
        and 0 <= b[0] <= b[2] <= 999
        and 0 <= b[1] <= b[3] <= 999
    )


def test_every_bbox_is_inside_the_frame() -> None:
    n = 0
    for c in _cases():
        t = c["target"]
        if c["tool"] in ("click_at", "type_at"):
            assert t is not None, c["id"]
            assert c["gt"]["bbox"] == t["bbox"], c["id"]
        if t is not None:
            assert _bbox_ok(t["bbox"]), (c["id"], t)
            assert isinstance(t["element"], str) and t["element"], c["id"]
            n += 1
    assert n >= 60  # every click/type case plus the scroll/press_key regions


def test_click_and_type_bboxes_are_not_the_whole_frame() -> None:
    """A GT bbox covering most of the frame would make ``click_at`` exact by accident."""
    for c in _cases():
        if c["tool"] in ("click_at", "type_at"):
            x0, y0, x1, y1 = c["gt"]["bbox"]
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            assert area <= 0.10 * 1000 * 1000, (c["id"], c["gt"]["bbox"])  # a list row is ~5 %


# ------------------------------------------------------------------------------------ ground truth


@pytest.mark.parametrize("cid", [c["id"] for c in _cases() if c["kind"] == "tool"])
def test_ground_truth_shape_and_reference_call(cid: str) -> None:
    c = next(x for x in _cases() if x["id"] == cid)
    gt, ref = c["gt"], c["reference_call"]
    assert gt["name"] == c["tool"]
    assert set(gt) == GT_KEYS[c["tool"]], gt
    assert ref["name"] == c["tool"]
    assert score.schema_valid(ref, TOOLSET), ref
    assert score.exact(ref, gt), (ref, gt)
    parsed = score.parse_call(json.dumps(ref))
    assert parsed == ref


def _perturbations(c: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One-field changes of the reference call that must not be exact (each stays schema-valid)."""
    ref = c["reference_call"]
    args = dict(ref["arguments"])
    gt = c["gt"]
    out: list[dict[str, Any]] = []
    if c["tool"] == "google_search":
        for k in gt["keywords"]:
            q = " ".join(x for x in gt["keywords"] if x != k) or "something else"
            out.append({"name": "google_search", "arguments": {"query": q}})
    elif c["tool"] in ("click_at", "type_at"):
        x0, y0, x1, y1 = gt["bbox"]
        for x, y in (
            (x0 - 1, args["y"]),
            (x1 + 1, args["y"]),
            (args["x"], y0 - 1),
            (args["x"], y1 + 1),
        ):
            if 0 <= x <= 999 and 0 <= y <= 999:
                out.append({"name": c["tool"], "arguments": {**args, "x": x, "y": y}})
        if c["tool"] == "type_at":
            out.append(
                {"name": "type_at", "arguments": {**args, "press_enter": not args["press_enter"]}}
            )
            out.append({"name": "type_at", "arguments": {**args, "text": args["text"] + "x"}})
    elif c["tool"] == "scroll":
        other = {"up": "down", "down": "up", "left": "right", "right": "left"}[args["direction"]]
        out.append({"name": "scroll", "arguments": {"direction": other}})
    elif c["tool"] == "press_key":
        other = "Enter" if args["combo"] != "Enter" else "Escape"
        out.append({"name": "press_key", "arguments": {"combo": other}})
    return out


@pytest.mark.parametrize("cid", [c["id"] for c in _cases() if c["kind"] == "tool"])
def test_ground_truth_discriminates(cid: str) -> None:
    """Adversarial: a call one field away from the reference is schema-valid but not exact."""
    c = next(x for x in _cases() if x["id"] == cid)
    bad = _perturbations(c)
    assert bad, cid
    for call in bad:
        assert score.schema_valid(call, TOOLSET), call
        assert not score.exact(call, c["gt"]), (cid, call)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().casefold()


def test_search_keywords_are_grounded_in_the_request() -> None:
    """Every keyword is a substring of the (normalised) user request: the ground truth asks for nothing the user
    did not say."""
    for c in _cases():
        if c["tool"] != "google_search":
            continue
        kws = c["gt"]["keywords"]
        assert kws and all(isinstance(k, str) and k == _norm(k) for k in kws), c["id"]
        for k in kws:
            assert k in _norm(c["user"]), (c["id"], k)


def test_type_at_text_is_quoted_and_enter_is_explicit() -> None:
    for c in _cases():
        if c["tool"] != "type_at":
            continue
        gt = c["gt"]
        assert f'"{gt["text"]}"' in c["user"], c["id"]
        u = c["user"].lower()
        negated = any(w in u for w in ("without", "no enter", "don't"))
        assert any(w in u for w in ("enter", "submit", "send")), c["id"]
        assert negated is not gt["press_enter"], (c["id"], c["user"])


def test_press_key_combos_are_in_the_d1_enum() -> None:
    enum = set(TOOLSET.parameters("press_key")["properties"]["combo"]["enum"])
    for c in _cases():
        if c["tool"] == "press_key":
            assert c["gt"]["combo"] in enum, c["id"]


def test_scroll_covers_every_direction() -> None:
    dirs = Counter(c["gt"]["direction"] for c in _cases() if c["tool"] == "scroll")
    assert set(dirs) == {"up", "down", "left", "right"}


def test_no_tool_cases_carry_no_ground_truth() -> None:
    for c in _cases():
        if c["kind"] == "no_tool":
            assert c["tool"] is None and c["gt"] is None and c["reference_call"] is None, c["id"]
            assert c["target"] is None, c["id"]


# ------------------------------------------------------------------------------------------ splits


def test_dev_and_test_are_disjoint_in_cases_and_screens() -> None:
    cases = _cases()
    dev = {c["id"] for c in cases if c["split"] == "dev"}
    test = {c["id"] for c in cases if c["split"] == "test"}
    assert dev.isdisjoint(test)
    assert dev | test == {c["id"] for c in cases}
    dev_screens = {c["screen"] for c in cases if c["split"] == "dev" and c["screen"]}
    test_screens = {c["screen"] for c in cases if c["split"] == "test" and c["screen"]}
    assert dev_screens and test_screens
    assert dev_screens.isdisjoint(test_screens)


def test_screen_html_loads_nothing_from_the_network() -> None:
    """Q4: rendered offline from file:// only. The pages carry no URL scheme, no src/href to a resource and no
    ``url(...)``/``@import`` in their CSS (``href="#"`` anchors are inert)."""
    pages = sorted((HERE / "screens" / "html").glob("*.html"))
    assert len(pages) == 24
    for p in pages:
        text = p.read_text()
        assert not re.search(r"(?i)\b(?:https?|wss?|ftp|data):", text), p.name
        assert not re.search(r"(?i)\bsrc\s*=", text), p.name
        assert not re.search(r"(?i)url\s*\(|@import", text), p.name
        assert set(re.findall(r'href="([^"]*)"', text)) <= {"#"}, p.name


# =========================================================================== T-0086 criterion 3


def test_t0086_every_search_reference_query_is_plausible_and_exact() -> None:
    """T-0086 criterion 3: all 30 ``google_search`` reference queries have no ``query_problem`` and stay exact."""
    qp = getattr(score, "query_problem", None)
    assert callable(qp), "T-0086 not implemented: score.query_problem is missing"
    search = [c for c in _cases() if c["tool"] == "google_search"]
    assert len(search) == 30
    bad = {c["id"]: qp(c["reference_call"]["arguments"]["query"]) for c in search}
    assert {k: v for k, v in bad.items() if v is not None} == {}
    assert [c["id"] for c in search if score.exact(c["reference_call"], c["gt"]) is not True] == []


def test_t0086_non_search_reference_calls_are_unchanged() -> None:
    """T-0086 criterion 3: every non-search reference call scores as under 4c7dfdc (frozen oracle) -- exact -- and
    a google_search call never matches a non-search case."""
    from tests.unit import _frozen_score_4c7dfdc as old

    assert callable(getattr(score, "query_problem", None)), "T-0086 not implemented"
    other = [c for c in _cases() if c["tool"] not in (None, "google_search")]
    assert len(other) == 90
    for c in other:
        assert (
            score.exact(c["reference_call"], c["gt"])
            is old.exact(c["reference_call"], c["gt"])
            is True
        ), c["id"]
