"""T-0223 group A: the ten handwritten maths/physics attempts and their renderer.

Derived from docs/tasks/T-0223.md (the ground-truth table, criteria 1-3, rulings R-1 and R-2) and the
schema stated in ``demos/handwritten/README.md``. Every expected number is recomputed here from the
problem data (by hand formula, exact fractions, numpy's eigen-solver, or a finite-difference / Riemann
oracle), never read back from ``attempts.json``.

The renderer needs matplotlib and pillow, which are not project dependencies: it runs as a subprocess
under ``uv run --with matplotlib --with pillow`` (criterion 2) and the render rows skip with a reason
when ``uv`` is absent or cannot resolve those packages (no network, empty cache).
"""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
import subprocess
import sys
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demos" / "handwritten"
ATTEMPTS = DEMO / "attempts.json"
OUT = DEMO / "out"
FONTS = DEMO / "fonts"
RENDERER = REPO / "scripts" / "demo" / "render_handwritten.py"

ORDER = ["M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5"]
G = 9.8  # every physics problem states g = 9.8 m/s^2


def _data() -> dict[str, Any]:
    return json.loads(ATTEMPTS.read_text(encoding="utf-8"))


def _by_id() -> dict[str, dict[str, Any]]:
    return {a["id"]: a for a in _data()["attempts"]}


def _lines(att: dict[str, Any]) -> list[str]:
    return [ln["text"] for ln in att["lines"]]


def _final(att: dict[str, Any]) -> str:
    return att["lines"][-1]["text"]


def _flagged(att: dict[str, Any]) -> int:
    idx = [i for i, ln in enumerate(att["lines"]) if ln["mistake"] is True]
    assert len(idx) == 1, (att["id"], idx)
    return idx[0]


def _after_mistake(att: dict[str, Any]) -> str:
    """The flagged line and every line after it (where the consequences of the mistake live)."""
    return " ".join(_lines(att)[_flagged(att) :])


# ---------------------------------------------------------------------------------------------
# Criterion 1: the schema
# ---------------------------------------------------------------------------------------------


def test_ten_ids_in_order() -> None:
    ids = [a["id"] for a in _data()["attempts"]]
    assert sorted(ids) == sorted(ORDER), ids
    assert len(ids) == len(set(ids)) == 10


@pytest.mark.parametrize("aid", ORDER)
def test_attempt_fields(aid: str) -> None:
    att = _by_id()[aid]
    for key in ("id", "course", "title", "problem", "mistake", "correct_result"):
        assert isinstance(att.get(key), str) and att[key].strip(), (aid, key)
    assert aid[0] in "MP"
    # the task: each attempt is 6-12 lines
    assert isinstance(att["lines"], list) and 6 <= len(att["lines"]) <= 12, (aid, len(att["lines"]))
    for ln in att["lines"]:
        assert set(ln) >= {"text", "mistake"}, (aid, ln)
        assert isinstance(ln["text"], str) and ln["text"].strip(), (aid, ln)
        assert isinstance(ln["mistake"], bool), (aid, ln)  # a real flag, not a truthy string
    assert sum(ln["mistake"] for ln in att["lines"]) == 1, aid
    # the mistake is never on the final answer line: the answer must *follow* from an earlier line
    assert _flagged(att) < len(att["lines"]) - 1, aid
    assert "diagram" in att, aid


@pytest.mark.parametrize("aid", ORDER)
def test_judge_and_false_blame_keywords_shape(aid: str) -> None:
    att = _by_id()[aid]
    jk = att["judge_keywords"]
    assert isinstance(jk, list) and jk, aid
    for entry in jk:
        assert isinstance(entry, str), (aid, entry)
        alts = entry.split("|")
        # every |-alternative non-empty (an empty one would match every answer)
        assert all(a.strip() for a in alts), (aid, entry)
        for a in alts:
            # R-2: no bare "= 1" style alternative (a lone digit, with or without "=")
            assert not re.fullmatch(r"\s*=?\s*-?\d\s*", a), (aid, a)
    fb = att["false_blame_keywords"]
    assert isinstance(fb, list), aid
    assert all(isinstance(k, str) and k.strip() for k in fb), (aid, fb)


# The ground-truth mistakes of the task table, as the substring each flagged line must carry (raw
# mathtext). A flag on any other line fails.
FLAGGED_SIGNATURE = {
    "M1": [r"v = 2e^{2x}"],
    "M2": [r"\cos(x^2)"],
    "M3": [r"(x^2 - x)"],
    "M4": [r"+ (1)(1)"],
    "M5": [r"e^{2x} + C"],
    "P1": [r"N = mg"],
    "P2": [r"20 = 5t"],
    "P3": [r"3 + 6 = 9"],
    "P4": [r"{m_2}"],
    "P5": [r"L\cos 45"],
}


@pytest.mark.parametrize("aid", ORDER)
def test_flagged_line_is_the_ground_truth_mistake(aid: str) -> None:
    att = _by_id()[aid]
    text = _lines(att)[_flagged(att)]
    for sig in FLAGGED_SIGNATURE[aid]:
        assert sig in text, (aid, text)
    # no correct line carries the mistake's signature too (else the flag would be ambiguous)
    others = [t for i, t in enumerate(_lines(att)) if i != _flagged(att)]
    for sig in FLAGGED_SIGNATURE[aid]:
        assert not any(sig in t for t in others), (aid, sig)
    if aid == "M2":
        assert "2x" not in text  # the chain-rule factor really is missing on that line
    if aid == "P1":
        assert r"\cos" not in text
    if aid == "P4":
        assert "m_1 + m_2" not in text


# ---------------------------------------------------------------------------------------------
# Criterion 3: correct results recomputed, and each wrong answer follows from its one mistake
# ---------------------------------------------------------------------------------------------


def _f2(x: float) -> str:
    return f"{x:.2f}"


def test_m1_by_parts() -> None:
    att = _by_id()["M1"]

    def correct(x: float) -> float:
        return (x / 2) * math.exp(2 * x) - 0.25 * math.exp(2 * x)

    def student(x: float) -> float:
        # u = x, v = 2e^{2x} (the mistake): uv - ∫v du = 2x e^{2x} - e^{2x}
        return 2 * x * math.exp(2 * x) - math.exp(2 * x)

    h = 1e-6
    for x in (-1.0, 0.0, 0.3, 1.2):
        d = (correct(x + h) - correct(x - h)) / (2 * h)
        assert d == pytest.approx(x * math.exp(2 * x), rel=1e-6, abs=1e-6)
        ds = (student(x + h) - student(x - h)) / (2 * h)
        # the student's v is 4x too big, so the answer is 4 times the true antiderivative
        assert ds == pytest.approx(4 * x * math.exp(2 * x), rel=1e-6, abs=1e-6)
    assert att["correct_result"].replace(" ", "") == "(x/2)e^{2x}-(1/4)e^{2x}+C"
    assert "(2x - 1)" in _final(att) and "e^{2x}" in _final(att)
    assert r"2x\,e^{2x} - e^{2x}" in _after_mistake(att)


def test_m2_tangent() -> None:
    att = _by_id()["M2"]
    x0 = math.sqrt(math.pi)
    h = 1e-6
    slope = (math.sin((x0 + h) ** 2) - math.sin((x0 - h) ** 2)) / (2 * h)
    assert slope == pytest.approx(-2 * math.sqrt(math.pi), rel=1e-6)
    assert math.sin(x0**2) == pytest.approx(0, abs=1e-12)
    cr = att["correct_result"]
    assert "-2 sqrt(pi)" in cr and "2 pi" in cr  # y = -2√π x + 2π passes through (√π, 0)
    # the student's slope cos(π) = -1, so the tangent y = -(x - √π)
    assert math.cos(math.pi) == -1
    assert "m = \\cos(\\pi) = -1" in _after_mistake(att)
    assert "y = -x + \\sqrt{\\pi}" in _final(att)


def test_m3_area() -> None:
    att = _by_id()["M3"]
    # Riemann oracle on the enclosed region (|upper - lower|), n = 20000 midpoints
    n = 20000
    area = sum(abs((k + 0.5) / n - ((k + 0.5) / n) ** 2) for k in range(n)) / n
    assert area == pytest.approx(1 / 6, abs=1e-6)
    assert att["correct_result"] == "1/6"
    wrong = Fraction(1, 3) - Fraction(1, 2)  # ∫0^1 (x^2 - x) dx
    assert wrong == Fraction(-1, 6)
    assert "= -1/6" in _final(att)
    assert r"\frac{1}{3} - \frac{1}{2}" in _after_mistake(att)


def test_m4_eigenvalues() -> None:
    att = _by_id()["M4"]
    ev = sorted(np.linalg.eigvals(np.array([[2.0, 1.0], [1.0, 2.0]])).real)
    assert ev == pytest.approx([1.0, 3.0])
    assert re.search(r"λ = 1\b.*λ = 3\b", att["correct_result"]), att["correct_result"]
    # the student's polynomial (2-λ)^2 + 1 = λ^2 - 4λ + 5: discriminant 16 - 20, roots 2 ± i
    roots = sorted(np.roots([1, -4, 5]), key=lambda z: z.imag)
    assert roots[0] == pytest.approx(2 - 1j) and roots[1] == pytest.approx(2 + 1j)
    tail = _after_mistake(att)
    assert r"\lambda^2 - 4\lambda + 5" in tail and "16 - 20" in tail
    assert r"2 \pm i" in _final(att)


def test_m5_ivp() -> None:
    att = _by_id()["M5"]

    def y(x: float) -> float:
        return 3 * math.exp(2 * x)

    h = 1e-6
    for x in (0.0, 0.4, 1.0):
        assert (y(x + h) - y(x - h)) / (2 * h) == pytest.approx(2 * y(x), rel=1e-6)
    assert y(0) == 3
    assert att["correct_result"].replace(" ", "") == "y=3e^{2x}"
    # student: y = e^{2x} + C, y(0) = 3 -> C = 3 - e^0 = 2
    c = 3 - math.exp(0)
    assert c == 2
    assert "C = 2" in _after_mistake(att) and "y = e^{2x} + 2" in _final(att)

    # and the student's answer really fails the ODE (so the mistake shows in the result)
    def ys(x: float) -> float:
        return math.exp(2 * x) + 2

    assert (ys(0.5 + h) - ys(0.5 - h)) / (2 * h) != pytest.approx(2 * ys(0.5), rel=1e-3)


def test_p1_incline() -> None:
    att = _by_id()["P1"]
    m, th, mu = 2.0, math.radians(30), 0.2
    a = G * (math.sin(th) - mu * math.cos(th))
    assert _f2(a) == "3.20"
    assert "3.20" in att["correct_result"]
    # student: N = mg
    n_w = m * G
    f_w = mu * n_w
    drive = m * G * math.sin(th)
    a_w = (drive - f_w) / m
    tail = _after_mistake(att)
    for v in (n_w, f_w, drive, m * a_w):
        assert f"{v:.2f}".rstrip("0").rstrip(".") in tail, (v, tail)
    assert _f2(a_w) == "2.94" and "2.94" in _final(att)
    assert abs(a - a_w) > 0.2


def test_p2_cliff() -> None:
    att = _by_id()["P2"]
    h_, vx = 20.0, 5.0
    t = math.sqrt(2 * h_ / G)
    assert _f2(t) == "2.02" and f"{vx * t:.1f}" == "10.1"
    assert "2.02" in att["correct_result"] and "10.1" in att["correct_result"]
    # student: 20 = 5t + 4.9t^2
    disc = vx**2 + 4 * (G / 2) * h_
    assert disc == pytest.approx(417)
    t_w = (-vx + math.sqrt(disc)) / G
    tail = _after_mistake(att)
    assert f"{math.sqrt(disc):.2f}" in tail  # 20.42
    assert _f2(t_w) == "1.57" and "1.57" in _final(att)
    assert f"{vx * float(_f2(t_w)):.1f}" == "7.9" and "7.9" in _final(att)


def test_p3_resistors() -> None:
    att = _by_id()["P3"]
    r1, r2, r3, v = Fraction(6), Fraction(3), Fraction(6), Fraction(12)
    r23 = 1 / (1 / r2 + 1 / r3)
    assert r23 == 2 and r1 + r23 == 8 and v / (r1 + r23) == Fraction(3, 2)
    cr = att["correct_result"]
    assert "8 Ω" in cr and "1.5 A" in cr
    # student: r2 + r3
    req_w = r1 + r2 + r3
    i_w = v / req_w
    assert req_w == 15 and i_w == Fraction(4, 5)
    tail = _after_mistake(att)
    assert "6 + 9 = 15" in tail and "12 / 15 = 0.8" in tail
    assert f"{float(i_w * r1):.1f}" in tail  # the check line, 4.8 V
    assert "15" in _final(att) and "0.8" in _final(att)


def test_p4_atwood() -> None:
    att = _by_id()["P4"]
    m1, m2 = 3.0, 5.0
    a = (m2 - m1) * G / (m1 + m2)
    t1, t2 = m1 * (G + a), m2 * (G - a)  # two independent routes to the tension
    assert t1 == pytest.approx(t2)
    assert a == pytest.approx(G / 4) and _f2(a) == "2.45" and f"{t1:.1f}" == "36.8"
    cr = att["correct_result"]
    assert "2.45" in cr and "36.8" in cr
    # student: divides by m2 only
    a_w = (m2 - m1) * G / m2
    t_w = m1 * (G + a_w)
    tail = _after_mistake(att)
    assert _f2(a_w) == "3.92" and "3.92" in tail
    assert _f2(t_w) == "41.16" and "41.16" in tail
    assert "3.92" in _final(att) and f"{t_w:.1f}" in _final(att)


def test_p5_pendulum_at_45() -> None:
    att = _by_id()["P5"]
    assert "45°" in att["problem"] and "60" not in att["problem"]
    L, th = 1.0, math.radians(45)
    v = math.sqrt(2 * G * L * (1 - math.cos(th)))
    assert _f2(v) == "2.40" and "2.40" in att["correct_result"]
    assert "45" in att["correct_result"]
    # student: h = L cos θ
    h_w = L * math.cos(th)
    v_w = math.sqrt(2 * G * h_w)
    tail = _after_mistake(att)
    assert f"{h_w:.3f}" in tail and f"{2 * G * h_w:.2f}" in tail  # 0.707, 13.86
    assert _f2(v_w) == "3.72" and "3.72" in _final(att)
    # why 45 and not 60: at 60° the wrong height equals the right one, so the mistake would be invisible
    th60 = math.radians(60)
    assert math.cos(th60) == pytest.approx(1 - math.cos(th60))
    assert abs(v_w - v) > 1.0


# ---------------------------------------------------------------------------------------------
# R-2: judge alternatives specific to the diagnosis; page words only beside a specific entry
# ---------------------------------------------------------------------------------------------

_MATHTEXT = {
    r"\theta": "θ",
    r"\pi": "π",
    r"\sqrt": "√",
    r"\cdot": "·",
    r"\parallel": "∥",
    r"\Omega": "Ω",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\leq": "≤",
    r"\pm": "±",
    r"\int": "∫",
    r"\cos": "cos",
    r"\sin": "sin",
    r"\ln": "ln",
    r"\det": "det",
    r"\left": "",
    r"\right": "",
    r"\frac": "",
    r"\,": " ",
    r"\ ": " ",
}


def _plain(s: str) -> str:
    for k, v in _MATHTEXT.items():
        s = s.replace(k, v)
    s = re.sub(r"[${}\\]", "", s)
    return re.sub(r"\s+", " ", s)


def _printed_texts(att: dict[str, Any]) -> list[str]:
    """What the page prints: the raw JSON strings, and the same with mathtext markup stripped."""
    raw = " ".join(
        [att["course"], att["title"], f"Problem {att['id']}.", att["problem"], *_lines(att)]
    )
    return [raw.casefold(), _plain(raw).casefold()]


def _page_word_entries(att: dict[str, Any], texts: list[str]) -> tuple[list[str], list[str]]:
    """Split judge entries into (page-word entries, specific entries). An entry is a page-word entry
    when any of its alternatives is a casefold substring of the page's printed text."""
    page, specific = [], []
    for entry in att["judge_keywords"]:
        if any(alt.casefold() in t for alt in entry.split("|") for t in texts):
            page.append(entry)
        else:
            specific.append(entry)
    return page, specific


def _located(judged: str, att: dict[str, Any]) -> bool:
    """The B4 rule (pre-registered): every judge entry (any one |-alternative) and no false-blame
    phrase, casefold substring."""
    s = judged.casefold()
    ok = all(any(a.casefold() in s for a in e.split("|")) for e in att["judge_keywords"])
    return ok and not any(k.casefold() in s for k in att["false_blame_keywords"])


@pytest.mark.parametrize("aid", ORDER)
def test_r2_page_words_only_beside_a_specific_entry(aid: str) -> None:
    att = _by_id()[aid]
    page, specific = _page_word_entries(att, _printed_texts(att))
    assert specific, f"{aid}: every judge entry is satisfiable by the page's own text: {page}"


# The pages whose README names a page word as a judge entry, and the specific sibling each needs.
PAGE_WORD_PAGES = {
    "P1": ("normal", "mg cos"),
    "P2": ("horizontal", "no initial vertical"),
    "P3": ("parallel", "1/r"),
}


@pytest.mark.parametrize("aid", sorted(PAGE_WORD_PAGES))
def test_r2_named_page_words_have_their_specific_sibling(aid: str) -> None:
    att = _by_id()[aid]
    word, specific_alt = PAGE_WORD_PAGES[aid]
    page, specific = _page_word_entries(att, _printed_texts(att))
    # the page word is one |-alternative of a page-word entry (it may sit beside symbol forms such
    # as P1's "n = mg cos", which make a diagnosis without the word itself count)
    assert any(word in e.split("|") for e in page), (aid, page)
    assert any(specific_alt in e.split("|") for e in specific), (aid, specific)


@pytest.mark.parametrize("aid", ORDER)
def test_adversarial_reading_the_page_back_is_not_located(aid: str) -> None:
    """Failure mode: a judge a model passes by reading the page aloud. The page's own text (problem,
    every line, the boxed answer), as typed or with mathtext stripped, must not count as located."""
    att = _by_id()[aid]
    for t in _printed_texts(att):
        assert not _located(t, att), (aid, t[:120])


@pytest.mark.parametrize("aid", ORDER)
def test_ground_truth_diagnosis_is_located(aid: str) -> None:
    """Too-strict direction: a diagnosis worded like the pre-registered ground truth (the mistake
    sentence plus the correct result) must satisfy every judge entry."""
    att = _by_id()[aid]
    assert _located(f"{att['mistake']} {att['correct_result']}", att), aid


@pytest.mark.parametrize(
    "spoken",
    [
        "Line 3 forgot the chain rule: the derivative is 2x cos(x squared).",
        "Line 3 drops the chain-rule factor, so dy/dx should be 2x cos(x^2).",
        "You missed the CHAIN RULE on line 3, the slope is 2 root pi with a minus sign.",
    ],
)
def test_m2_chain_rule_spelled_either_way_is_located(spoken: str) -> None:
    """M2's ground truth says "chain-rule"; a spoken "chain rule" and a written "chain-rule" both
    name the diagnosis, so both must satisfy the judge."""
    assert _located(spoken, _by_id()["M2"]), spoken


def test_m2_without_the_chain_rule_is_not_located() -> None:
    """Too-wide direction: naming the factor without the rule, or the rule without the factor, fails."""
    att = _by_id()["M2"]
    assert not _located("Line 3 is wrong, the slope should be 2x cos(x^2).", att)
    assert not _located("Line 3 forgot the chain rule.", att)


# Review finding 8: per page, natural correct diagnoses (symbol form and spoken form) that must be
# located, and a plausible wrong diagnosis that must not be. Written by hand from the task's ground
# truth table, not from the keyword lists; the spoken forms are what the model's TTS-bound text looks
# like (operators in words, numbers as digits). Partly from the reviewer's diag.py.
CORRECT_SYMBOL = {
    "M1": "Line 3 is wrong: you differentiated e^{2x} instead of integrating it, so v should be "
    "(1/2)e^{2x} and the answer is (x/2)e^{2x} - (1/4)e^{2x} + C.",
    "M2": "Line 3 misses the chain rule: dy/dx = 2x cos(x^2), so the slope at √π is -2√π.",
    "M3": "Line 2 subtracts the wrong way: y = x is above y = x² on [0, 1], so A = ∫(x − x²) dx "
    "= 1/6; an area cannot be negative.",
    "M4": "Line 3 has the wrong sign: det = (2 − λ)² − (1)(1) = λ² − 4λ + 3, so λ = 1 and λ = 3.",
    "M5": "Line 5: exponentiating ln y = 2x + C gives y = e^C·e^{2x} = Ae^{2x}, a multiplicative "
    "constant, so y(0) = 3 gives y = 3e^{2x}.",
    "P1": "Line 3: on the incline N = mg cos θ = 16.97 N, not mg, so f = 3.39 N and a ≈ 3.20 m/s².",
    "P2": "Line 2 puts the horizontal 5 m/s into the vertical equation; the initial vertical "
    "velocity is 0 m/s, so 20 = 4.9t², t ≈ 2.02 s and x ≈ 10.1 m.",
    "P3": "Line 2: R2 and R3 are in parallel, so 1/R23 = 1/3 + 1/6, R23 = 2 Ω, R_eq = 8 Ω and "
    "I = 1.5 A.",
    "P4": "Line 3 divides by m2 only; divide by m1 + m2 = 8 kg: a = 19.6/8 = 2.45 m/s², T ≈ 36.8 N.",
    "P5": "Line 3: the drop is h = L(1 - cos 45°) = 0.293 m, not L cos 45°, so v = √(2gh) ≈ "
    "2.40 m/s.",
}
CORRECT_SPOKEN = {
    "M1": "The mistake is in line 3. You differentiated e to the 2x instead of integrating it. If "
    "dv is e to the 2x dx, then v is e to the 2x over 2. Then the answer is x over 2 times e to "
    "the 2x minus a quarter e to the 2x plus C.",
    "M2": "In line 3 you forgot the chain rule. The derivative of sin of x squared is 2x times cos "
    "of x squared, so the slope at root pi is minus 2 root pi.",
    "M3": "Line 2 has the subtraction backwards. On 0 to 1 the line y = x is above the parabola, "
    "so you integrate x minus x squared, and the area is 1/6. An area can't be negative.",
    "M4": "Line 3 has a sign error. The determinant is 2 minus lambda squared minus 1, not plus "
    "1. That gives lambda squared minus 4 lambda plus 3, so the eigenvalues are 1 and 3.",
    "M5": "When you exponentiate, the constant becomes a factor, not something you add. So y "
    "equals A e to the 2x, and y of 0 equals 3 gives y equals 3 e to the 2x.",
    "P1": "Line 3: on an incline the normal force isn't mg, it's mg cos theta, about 16.97 "
    "newtons. Then friction is 3.39 newtons and the acceleration is about 3.2 metres per second "
    "squared.",
    "P2": "In line 2 you used the 5 metres per second in the vertical equation, but that speed is "
    "horizontal. The initial vertical velocity is zero, so 20 equals 4.9 t squared, t is about "
    "2.02 seconds, and it lands about 10.1 metres out.",
    "P3": "Line 2 is wrong: R2 and R3 are in parallel, so you combine them with 1 over R, giving 2 "
    "ohms. The total is 8 ohms and the current is 1.5 amps.",
    "P4": "Line 3 is the mistake. The net force should be divided by the total mass, 8 "
    "kilograms, not just m2. So a is 19.6 over 8, which is 2.45 metres per second squared, and the "
    "tension is about 36.8 newtons.",
    "P5": "The drop is L minus L cos theta, about 0.29 metres, so the speed is about 2.4 metres "
    "per second, not 3.72.",
}
WRONG_DIAGNOSIS = {
    # blames a correct line (the e^{2x} bookkeeping on line 5) with the differentiation vocabulary
    "M1": "Line 5 is wrong: you differentiated incorrectly, 2 times 1/2 e^{2x} should be e^{2x}.",
    # names the chain rule but blames the point, not the missing 2x; says "sin" freely
    "M2": "Line 2 is the problem: sin(π) is not zero, and the chain rule only matters for the "
    "intercept of the tangent line.",
    # right final number, wrong reason (the limits), no upper-minus-lower
    "M3": "Line 3 is wrong: you should evaluate from 1 to 0 instead, so the area is positive 1/6.",
    # blames the discriminant, not the determinant's sign
    "M4": "Line 5 is wrong: the discriminant should be 16 + 4, so λ = 2 ± √5 and both are real.",
    # blames the initial-condition arithmetic
    "M5": "Line 6 is wrong: e^0 is 0, so C = 3 and the answer is y = e^{2x} + 3.",
    # the classic sin/cos swap, blaming the driving-force line and clearing the normal force
    "P1": "Line 5 is wrong: the component along the slope is mg cos θ, not mg sin θ; the normal "
    "force line is fine.",
    # clears the setup, blames the root choice
    "P2": "Your horizontal and vertical setup is fine; the quadratic should equal zero but you "
    "took the wrong root.",
    # puts R1 in parallel too
    "P3": "Line 3 is wrong: R1 is in parallel with the pair as well, so R_eq = 1/(1/6 + 1/9) "
    "= 3.6 Ω.",
    # blames the tension equation
    "P4": "Line 5 is wrong: for m1 the tension equation should be T + m1 g = m1 a, so T is "
    "smaller.",
    # swaps cos for sin in the drop, which gives the same number at 45°
    "P5": "Line 3 should use L sin 45° for the drop instead; it is 0.707 m either way, so "
    "v ≈ 3.72 m/s is right.",
}

_SPOKEN_SYMBOLS = [
    ("^2", " squared"),
    ("²", " squared"),
    ("√", " square root of "),
    ("θ", " theta"),
    ("π", " pi"),
    ("λ", " lambda"),
    ("μ", " mu"),
    ("Ω", " ohms"),
    ("∥", " parallel "),
    ("±", " plus or minus "),
    ("≈", " approximately "),
    ("·", " times "),
    ("∫", " integral of "),
    ("°", " degrees"),
    ("≤", " less than or equal to "),
    ("=", " equals "),
    ("+", " plus "),
    ("−", " minus "),
    (" - ", " minus "),
    ("/", " over "),
    ("^", " to the "),
]
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]


def _spoken_readback(att: dict[str, Any], words_for_numbers: bool) -> str:
    """The page read aloud: the printed text with its symbols spoken ("=" -> "equals",
    "√" -> "square root of", "²" -> "squared", "θ" -> "theta"), optionally with the small whole
    numbers as words ("= 0" -> "equals zero"), and a closing "looks right to me"."""
    s = _plain(" ".join([att["problem"], *_lines(att)]))
    for sym, word in _SPOKEN_SYMBOLS:
        s = s.replace(sym, word)
    if words_for_numbers:
        s = re.sub(
            r"(?<![\d.])\d+(?![\d.])",
            lambda m: _NUMBER_WORDS[int(m.group())] if int(m.group()) <= 20 else m.group(),
            s,
        )
    return re.sub(r"\s+", " ", s) + " Looks right to me."


@pytest.mark.parametrize("aid", ORDER)
def test_natural_symbol_diagnosis_is_located(aid: str) -> None:
    assert _located(CORRECT_SYMBOL[aid], _by_id()[aid]), (aid, CORRECT_SYMBOL[aid])


@pytest.mark.parametrize("aid", ORDER)
def test_natural_spoken_diagnosis_is_located(aid: str) -> None:
    assert _located(CORRECT_SPOKEN[aid], _by_id()[aid]), (aid, CORRECT_SPOKEN[aid])


@pytest.mark.parametrize("words", [False, True], ids=["digits", "number-words"])
@pytest.mark.parametrize("aid", ORDER)
def test_spoken_readback_of_the_page_is_not_located(aid: str, words: bool) -> None:
    """Failure mode: a model that reads the page aloud and approves it. The spoken read-back must
    not satisfy the judge."""
    att = _by_id()[aid]
    text = _spoken_readback(att, words)
    assert not _located(text, att), (aid, text[:300])


@pytest.mark.parametrize("aid", ORDER)
def test_plausible_wrong_diagnosis_is_not_located(aid: str) -> None:
    assert not _located(WRONG_DIAGNOSIS[aid], _by_id()[aid]), (aid, WRONG_DIAGNOSIS[aid])


@pytest.mark.parametrize("aid", ORDER)
def test_false_blame_does_not_hit_the_ground_truth(aid: str) -> None:
    """A diagnosis that states the ground-truth mistake and the correct result must not be failed
    as false blame."""
    att = _by_id()[aid]
    truth = f"{att['mistake']} {att['correct_result']}".casefold()
    hits = [k for k in att["false_blame_keywords"] if k.casefold() in truth]
    assert not hits, (aid, hits)


# ---------------------------------------------------------------------------------------------
# Criterion 2: the font and its licence
# ---------------------------------------------------------------------------------------------


def test_vendored_font_and_licence() -> None:
    ttfs = sorted(FONTS.glob("*.ttf")) + sorted(FONTS.glob("*.otf"))
    assert ttfs, list(FONTS.iterdir())
    for f in ttfs:
        assert f.read_bytes()[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), f
    lic = (FONTS / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in lic.upper() and "1.1" in lic
    assert (FONTS / "PatrickHand-Regular.ttf").is_file()


# ---------------------------------------------------------------------------------------------
# Criterion 2: the renderer (subprocess under uv run --with matplotlib --with pillow)
# ---------------------------------------------------------------------------------------------

_NO_RESOLVE = re.compile(
    r"(network|resolve|dns|connect|offline|timed out|Failed to fetch|Failed to download|"
    r"No solution found|not found in the package registry)",
    re.IGNORECASE,
)


def _pinned_versions() -> tuple[str, str]:
    """The matplotlib / pillow versions the committed out/ was rendered with, parsed from the
    renderer's module constants (never by importing matplotlib in the test process)."""
    src = RENDERER.read_text(encoding="utf-8")
    found = []
    for name in ("MATPLOTLIB_VERSION", "PILLOW_VERSION"):
        m = re.search(rf'^{name}\s*=\s*"([^"]+)"', src, re.MULTILINE)
        assert m, f"{name} module constant missing from {RENDERER}"
        found.append(m.group(1))
    return found[0], found[1]


def test_renderer_pins_exact_versions() -> None:
    """The byte-equality rows are only meaningful against one exact matplotlib/pillow release."""
    for v in _pinned_versions():
        assert re.fullmatch(r"\d+\.\d+\.\d+", v), v


def _uv_python(args: list[str], timeout: float = 300) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH: cannot run the renderer under `uv run --with matplotlib`")
    mpl, pil = _pinned_versions()
    cmd = [
        uv,
        "run",
        "--no-project",
        "--with",
        f"matplotlib=={mpl}",
        "--with",
        f"pillow=={pil}",
        "python",
        *args,
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and _NO_RESOLVE.search(proc.stderr) and "Traceback" not in proc.stderr:
        pytest.skip(
            f"uv could not provide matplotlib/pillow (network or cache): {proc.stderr[-300:]}"
        )
    return proc


@pytest.fixture(scope="module")
def renders(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two independent fresh renders with the default font dir and attempts file."""
    dirs = []
    for k in (1, 2):
        out = tmp_path_factory.mktemp(f"render{k}")
        proc = _uv_python([str(RENDERER), "--out", str(out)])
        assert proc.returncode == 0, proc.stderr[-2000:]
        # under the pinned versions the renderer's own version warning must stay silent
        assert "render_handwritten: warning" not in proc.stderr, proc.stderr[-2000:]
        dirs.append(out)
    return dirs[0], dirs[1]


def _png_size(path: Path) -> tuple[int, int]:
    b = path.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", path
    assert b[12:16] == b"IHDR"
    w, h = struct.unpack(">II", b[16:24])
    return w, h


def _pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", path.read_bytes()))


def test_render_writes_ten_pngs_and_a_ten_page_pdf(renders: tuple[Path, Path]) -> None:
    out, _ = renders
    for aid in ORDER:
        w, h = _png_size(out / f"{aid}.png")
        assert w >= 1600, (aid, w)
        assert h > w  # a portrait page
    assert _pdf_pages(out / "attempts.pdf") == 10


def test_render_is_deterministic(renders: tuple[Path, Path]) -> None:
    a, b = renders
    for aid in ORDER:
        assert (a / f"{aid}.png").read_bytes() == (b / f"{aid}.png").read_bytes(), aid
    # distinct pages (a renderer writing one image ten times would pass the line above)
    assert len({(a / f"{aid}.png").read_bytes() for aid in ORDER}) == 10


def test_committed_pngs_equal_a_fresh_render(renders: tuple[Path, Path]) -> None:
    out, _ = renders
    for aid in ORDER:
        committed = OUT / f"{aid}.png"
        assert committed.is_file(), committed
        assert committed.read_bytes() == (out / f"{aid}.png").read_bytes(), aid
    assert (OUT / "attempts.pdf").is_file()
    assert _pdf_pages(OUT / "attempts.pdf") == 10


def test_pdf_pages_in_order(renders: tuple[Path, Path]) -> None:
    exe = shutil.which("pdftotext")
    if exe is None:
        pytest.skip("pdftotext (poppler-utils) not installed: cannot read the PDF page order")
    out, _ = renders
    titles = _by_id()
    for page, aid in enumerate(ORDER, start=1):
        text = subprocess.run(
            [exe, "-f", str(page), "-l", str(page), str(out / "attempts.pdf"), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert f"Problem {aid}." in text, (page, aid, text[:200])
        assert titles[aid]["title"] in text, (page, aid)
        others = [o for o in ORDER if o != aid]
        assert not any(f"Problem {o}." in text for o in others), (page, aid)


_PIXEL_PROBE = r"""
import json, sys
from PIL import Image
res = {}
for p in sys.argv[1:]:
    im = Image.open(p).convert("RGB")
    w, h = im.size
    px = im.load()
    ink = 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            r, g, b = px[x, y]
            if b < 160 and b - r > 50 and b - g > 30:
                ink += 1
    ruled = 0
    for y in range(h):
        blue = 0
        for x in range(0, w, 10):
            r, g, b = px[x, y]
            if 130 < b and b - r > 40 and r > 110:
                blue += 1
        if blue > 0.6 * (w / 10):
            ruled += 1
    res[p] = {"ink": ink, "ruled_rows": ruled}
print(json.dumps(res))
"""


def test_dark_blue_ink_on_lined_paper(renders: tuple[Path, Path], tmp_path: Path) -> None:
    """Criterion 2's look: dark-blue handwriting and light-blue rules spanning the page."""
    out, _ = renders
    probe = tmp_path / "probe.py"
    probe.write_text(_PIXEL_PROBE, encoding="utf-8")
    proc = _uv_python([str(probe), *(str(out / f"{aid}.png") for aid in ORDER)])
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(proc.stdout)
    for aid in ORDER:
        r = res[str(out / f"{aid}.png")]
        assert r["ink"] > 2000, (aid, r)  # sampled every 2nd pixel: thousands of ink pixels
        assert r["ruled_rows"] >= 15, (aid, r)  # at 88 px spacing on 2200 px, ~20 rules


def test_renderer_refuses_wrong_id_set(tmp_path: Path) -> None:
    """Adversarial: an attempts file missing a page must not silently render nine pages."""
    data = _data()
    data["attempts"] = [a for a in data["attempts"] if a["id"] != "P5"]
    bad = tmp_path / "attempts.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    proc = _uv_python([str(RENDERER), "--attempts", str(bad), "--out", str(out)])
    assert proc.returncode != 0
    assert not (out / "attempts.pdf").exists()


def test_png_is_not_blank(renders: tuple[Path, Path]) -> None:
    """Cheap stdlib check that the PNG stream holds real image data, independent of Pillow."""
    out, _ = renders
    b = (out / "P5.png").read_bytes()
    chunks, i = [], 8
    while i < len(b):
        (n,) = struct.unpack(">I", b[i : i + 4])
        kind = b[i + 4 : i + 8]
        if kind == b"IDAT":
            chunks.append(b[i + 8 : i + 8 + n])
        i += 12 + n
    raw = zlib.decompress(b"".join(chunks))
    w, h = _png_size(out / "P5.png")
    assert len(raw) >= w * h * 3  # at least RGB per pixel plus filter bytes
    assert len(set(raw[:: max(1, len(raw) // 50000)])) > 16  # not a flat colour


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
