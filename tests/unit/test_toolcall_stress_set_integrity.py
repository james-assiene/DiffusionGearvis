"""T-0099 criterion 1: integrity of the tool-call stress set (``tests/eval/toolcall_stress/``). Unit, hermetic.

Every row first requires the human's approval of the case set (H-T99a), recorded in ``MANIFEST.json``'s
``approval`` field ("H-T99a (human, 2026-10-14)"): without it every row fails with an ``AssertionError`` naming
H-T99a, so an unapproved edit of the set cannot pass. The checks themselves:

- 30 cases, criterion 1's category counts (K-in 6, K-out 6, S-extra 4, T-hard 6, C-pixel 4, U-under 4);
- every screen is a screen of a v1 dev case, none of a v1 test case; every reused target (``source_case``) is a v1
  dev case on the same screen with the same bbox;
- no user text repeats a ``tests/eval/toolcall`` case (normalised) or another stress case;
- every gt-bearing case's ``reference_call`` validates against ``tools_v1`` and is ``score.exact`` against its gt
  (U-under: under criterion 1's rescue rule), and the gt's fields are the tool's;
- **adversarial:** every K-out combo is absent from the ``press_key`` enum *read from tools_v1.json*, also
  case-folded and with whitespace removed (a secretly in-enum K-out case would let a repair "rescue" it); the
  checker is shown to catch a forged in-enum K-out case;
- C-pixel: a pixel coordinate above 999, and the gt bbox contains the hand-computed normalised point (the table
  below is the oracle, copied from the ``cases_src.py`` docstring, not computed by the generator);
- T-hard covers the six hard features, each a real property of the text; U-under says nothing about Enter;
- the manifest pins the cases file's sha256 (and the generator's), and the sha is pinned here as well, so a silent
  edit of the set needs an edit of this file.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import score, tools

ROOT = Path(__file__).resolve().parents[1] / "eval"
HERE = ROOT / "toolcall_stress"
CASES = HERE / "cases.jsonl"
MANIFEST = HERE / "MANIFEST.json"
V1_CASES = ROOT / "toolcall" / "cases.jsonl"
V1_DIR = ROOT / "toolcall"

#: The stress set's sha256, pinned (the harness pins it too, criterion 1).
CASES_SHA256 = "dea52a3763419d00f8d03cffd36c90dc7b3f4353203ee0d6e4cbd296228c79eb"

EXPECTED_COUNTS = {"K-in": 6, "K-out": 6, "S-extra": 4, "T-hard": 6, "C-pixel": 4, "U-under": 4}

#: Hand-computed (cases_src.py docstring): id -> (pixel, normalised point on 1280 x 800, bbox).
C_PIXEL_ORACLE = {
    "cpix-01": ((1024, 304), (800, 380), [740, 358, 839, 402]),
    "cpix-02": ((1152, 184), (900, 230), [189, 204, 983, 257]),
    "cpix-03": ((1088, 748), (850, 935), [267, 913, 915, 959]),
    "cpix-04": ((1120, 168), (875, 210), [94, 180, 906, 237]),
}

GT_KEYS = {
    "K-in": {"name", "combo"},
    "S-extra": {"name", "direction"},
    "T-hard": {"name", "bbox", "text", "press_enter"},
    "C-pixel": {"name", "bbox"},
    "U-under": {"name", "bbox", "text", "press_enter"},
}


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data


def _require_approval() -> None:
    approval = _manifest().get("approval")
    assert isinstance(approval, str) and "H-T99a" in approval, (
        "the stress set is not approved by the human yet (H-T99a): MANIFEST.json approval is "
        f"{approval!r}"
    )


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    return _load(CASES)


def _enum() -> list[str]:
    decl = tools.load().get("press_key")
    assert decl is not None
    enum: list[str] = decl["parameters"]["properties"]["combo"]["enum"]
    return enum


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().casefold()


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).casefold()


def _out_combo_problem(combo: Any, enum: list[str]) -> str | None:
    """Why ``combo`` is not a legitimate K-out combo, or None."""
    if not isinstance(combo, str) or not combo.strip():
        return "no out_combo"
    if combo in enum:
        return f"{combo!r} is in the press_key enum"
    if _squash(combo) in {_squash(e) for e in enum}:
        return f"{combo!r} is in the press_key enum up to case/whitespace"
    return None


def _bbox_ok(b: Any) -> bool:
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 999 for v in b)
        and b[0] <= b[2]
        and b[1] <= b[3]
    )


# --------------------------------------------------------------------------------------------- counts


def test_thirty_cases_in_criterion_ones_categories(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    assert len(cases) == 30
    assert Counter(c["category"] for c in cases) == EXPECTED_COUNTS
    assert len({c["id"] for c in cases}) == 30
    assert all(c["split"] == "dev" for c in cases)


# --------------------------------------------------------------------------------------------- screens


def test_every_screen_is_a_dev_screen_never_a_test_screen(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    v1 = _load(V1_CASES)
    dev_screens = {c["screen"] for c in v1 if c["split"] == "dev" and c["screen"]}
    test_screens = {c["screen"] for c in v1 if c["split"] == "test" and c["screen"]}
    assert dev_screens and test_screens
    for c in cases:
        assert c["screen"] in dev_screens, c["id"]
        assert c["screen"] not in test_screens, c["id"]
        assert (V1_DIR / c["screen"]).is_file(), c["id"]
        assert c["screen_size"] == [1280, 800], c["id"]


def test_reused_targets_are_dev_targets_on_the_same_screen(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    v1 = {c["id"]: c for c in _load(V1_CASES)}
    reused = [c for c in cases if c["category"] in ("T-hard", "C-pixel", "U-under")]
    assert len(reused) == 14
    for c in reused:
        src = v1[c["source_case"]]
        assert src["split"] == "dev", c["id"]
        assert src["screen"] == c["screen"], c["id"]
        assert src["target"]["bbox"] == c["target"]["bbox"] == c["gt"]["bbox"], c["id"]


# --------------------------------------------------------------------------------------------- text


def test_no_user_text_reuses_an_existing_case(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    existing = {_norm(c["user"]) for c in _load(V1_CASES)}
    mine = [_norm(c["user"]) for c in cases]
    assert len(set(mine)) == len(mine)
    assert not (set(mine) & existing)


# --------------------------------------------------------------------------------------------- gt


def test_every_gt_validates_against_tools_v1(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    ts = tools.load()
    for c in cases:
        if c["category"] == "K-out":
            assert c["gt"] is None and c["reference_call"] is None, c["id"]
            continue
        gt, ref = c["gt"], c["reference_call"]
        assert set(gt) == GT_KEYS[c["category"]], c["id"]
        assert gt["name"] == c["tool"] == ref["name"], c["id"]
        assert score.schema_valid(ref, ts), c["id"]
        if "bbox" in gt:
            assert _bbox_ok(gt["bbox"]), c["id"]
        if "combo" in gt:
            assert gt["combo"] in _enum(), c["id"]
        if "direction" in gt:
            decl = ts.get("scroll")
            assert decl is not None
            assert gt["direction"] in decl["parameters"]["properties"]["direction"]["enum"]
        if c["category"] == "U-under":
            assert gt["press_enter"] is None, c["id"]
            # the rescue rule (criterion 1): valid, inside the bbox, text equal; press_enter not scored
            for enter in (True, False):
                call = {"name": "type_at", "arguments": {**ref["arguments"], "press_enter": enter}}
                assert score.exact(call, {**gt, "press_enter": enter}), c["id"]
        else:
            assert score.exact(ref, gt), c["id"]


def test_k_in_asks_the_enum_key_in_another_spelling(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    k_in = [c for c in cases if c["category"] == "K-in"]
    assert {c["gt"]["combo"] for c in k_in} == {
        "Enter",
        "Escape",
        "Ctrl+C",
        "Space",
        "PageDown",
        "ArrowDown",
    }
    for c in k_in:
        assert c["gt"]["combo"] not in c["user"], c["id"]


def test_k_out_combos_are_absent_from_the_enum(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    enum = _enum()
    k_out = [c for c in cases if c["category"] == "K-out"]
    assert len(k_out) == 6
    for c in k_out:
        assert _out_combo_problem(c["out_combo"], enum) is None, c["id"]
        assert c["out_combo"] in c["user"], c["id"]
    assert len({_squash(c["out_combo"]) for c in k_out}) == 6


@pytest.mark.parametrize("forged", ["Ctrl+T", "ctrl+t", "PageDown", "Alt + Tab"])
def test_adversarial_an_in_enum_k_out_case_is_caught(forged: str) -> None:
    """The K-out check above is not vacuous: a forged K-out case naming an enum combo is flagged."""
    _require_approval()
    assert _out_combo_problem(forged, _enum()) is not None


def test_c_pixel_points_are_above_999_and_inside_the_gt_bbox(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    c_pixel = {c["id"]: c for c in cases if c["category"] == "C-pixel"}
    assert set(c_pixel) == set(C_PIXEL_ORACLE)
    for cid, (pixel, point, bbox) in C_PIXEL_ORACLE.items():
        c = c_pixel[cid]
        assert tuple(c["pixel"]) == pixel, cid
        assert pixel[0] > 999 or pixel[1] > 999, cid
        assert f"({pixel[0]}, {pixel[1]})" in c["user"], cid
        assert c["gt"]["bbox"] == bbox, cid
        x0, y0, x1, y1 = bbox
        assert x0 <= point[0] <= x1 and y0 <= point[1] <= y1, cid
        # the copied pixel is out of schema (maximum 999): the category's point
        assert not score.schema_valid(
            {"name": "click_at", "arguments": {"x": pixel[0], "y": pixel[1]}}, tools.load()
        ), cid


def test_t_hard_covers_the_six_hard_features(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    texts = [c["gt"]["text"] for c in cases if c["category"] == "T-hard"]
    assert len(texts) == 6

    def accent_and_emoji(t: str) -> bool:
        nfd = unicodedata.normalize("NFD", t)
        accent = any(unicodedata.category(ch) == "Mn" for ch in nfd)
        emoji = any(ord(ch) >= 0x1F000 for ch in t)
        return accent and emoji

    features = {
        "double quotes": lambda t: '"' in t,
        "backslash": lambda t: "\\" in t,
        "braces": lambda t: "{" in t and "}" in t,
        "newline": lambda t: "\n" in t,
        "accents and emoji": accent_and_emoji,
        "long": lambda t: len(t) >= 400,
    }
    for feature, has in features.items():
        assert sum(1 for t in texts if has(t)) >= 1, feature
    # every T-hard text is given verbatim in the user text (the newline case line by line)
    for c in cases:
        if c["category"] == "T-hard":
            for line in c["gt"]["text"].split("\n"):
                assert line in c["user"], c["id"]
            assert re.search(r"press(ing)? enter", c["user"], re.IGNORECASE), c["id"]


def test_u_under_says_nothing_about_enter(cases: list[dict[str, Any]]) -> None:
    _require_approval()
    u = [c for c in cases if c["category"] == "U-under"]
    assert len(u) == 4
    for c in u:
        assert not re.search(r"enter|submit|send|return", c["user"], re.IGNORECASE), c["id"]
        assert f'"{c["gt"]["text"]}"' in c["user"], c["id"]


# --------------------------------------------------------------------------------------------- manifest


def test_manifest_pins_the_cases_sha() -> None:
    _require_approval()
    raw = CASES.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    m = _manifest()
    assert m["cases_sha256"] == sha == CASES_SHA256
    assert m["n_cases"] == 30
    assert m["counts"] == EXPECTED_COUNTS
    gen = (HERE / "cases_src.py").read_bytes()
    assert m["generator"]["cases_src.py"] == hashlib.sha256(gen).hexdigest()
    assert m["source_set_sha256"] == hashlib.sha256(V1_CASES.read_bytes()).hexdigest()


def test_generator_reproduces_the_committed_file() -> None:
    _require_approval()
    from tests.eval.toolcall_stress import cases_src

    assert cases_src.render_cases(cases_src.build()) == CASES.read_bytes()
