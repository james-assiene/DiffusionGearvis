"""T-0211 group B, B1-B3 (text half): live guidance v4, its two sentences and what they say
(docs/tasks/T-0211.md, "The exact v4 text", B1, B2, B3).

**What these rows pin** (from the criteria; where the task is silent the reading is stated):

- B1: ``prompt_builder.S_SHARED_GUIDANCE`` and ``prompt_builder.S_SOURCE_GUIDANCE`` equal, byte for byte, the two
  sentences typed below from the task (213 and 151 code points); each is one module-level assignment of a single
  string literal. ``LIVE_TOOL_GUIDANCE_V4 == LIVE_TOOL_GUIDANCE_V3 + " " + S_SHARED_GUIDANCE + " " +
  S_SOURCE_GUIDANCE``, ``len(V4) <= 1000`` (it is 997), assigned once at module level. **Reading, reported**: the
  task shows V4 as the expression ``V3 + " " + S_SHARED + " " + S_SOURCE`` while "Tests to write" says "single
  literal"; both shapes are accepted: one ``str`` literal, or a ``+`` tree whose leaves are only ``str`` literals and
  the names ``LIVE_TOOL_GUIDANCE_V3``, ``S_SHARED_GUIDANCE``, ``S_SOURCE_GUIDANCE`` (no call, no f-string, no other
  name), so the value is fixed at import either way.
- B1: v1 still hashes to ``768a973b…``; v2 to ``518f56a1…``; v3 to ``b20cd723…`` (literal BASE 175a29b0).
- B2 (a): every Ctrl-like token in V4 is exactly ``Ctrl+L``, a member of the parsed ``press_key`` enum; no
  ``Ctrl-L``, ``ctrl+l``, ``Control+L``, ``Cmd``/``⌘`` (power rows show the predicate refuses each).
- B2 (b): the only tool-shaped token (snake_case or ``name(``) in V4 is ``google_search``, declared in
  ``tools_v1.json``; none of ``open_url``, ``navigate``, ``goto``, ``go_to`` (word match, any case; the English verb
  "navigate" is refused too, T-0191's reading).
- B2 (c): no "address bar" / "URL bar" (nor "location bar", "omnibox": the same false claim).
- B2 (d): V4 fails T-0174 A2(f)'s forward-pointing regex.
- B2 (e): V4 holds none of ``test_live_tool_guidance.py::test_no_confirmation_or_narration_request``'s banned
  substrings (typed in ``test_t0191_guidance_v3.NARRATION_BANNED``, imported).
- B2 (f): S_SOURCE holds ``google_search``; S_SHARED holds ``shared screen`` and ``own browser``.
- B3 (text half): ``LIVE_GUIDANCE`` is read-only, exactly ``{v1, v2, v3, v4}``; v1, v2 and v3 the same objects as
  the module attributes; v4 the attribute. (``config`` and ``cli`` are ``test_t0211_cli_guidance.py``.)

``test_v1_v2_v3_unchanged`` and the predicate power rows are active (green at BASE 175a29b0). Every other row was
``pending(task="T-0211")`` until activation (2026-10-30): at BASE the three names do not exist, so each fails on a named ``AssertionError`` ("does
not exist yet").

Regime: CPU, pure, no measurement; n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from local_astra.conductor import prompt_builder as pb
from tests.unit import test_t0191_guidance_v3 as g3

#: B1, typed from the task ("The exact v4 text"), never imported
S_SHARED = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: "
    "to do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)
S_SOURCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)
#: B1: v1, v2 (T-0174) and v3 (T-0191 A1 at main 175a29b0, T-0211's BASE) sha256, literals
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
V2_SHA = "518f56a1de3faff26c7fc1b8f938c4d233e59c5379f9acc431b74b7cefc41e50"
V3_SHA = "b20cd7239f29698a5bcf8f587e9b2a2b53de91e6f021bd63e74ae5ef40e082c5"
MAX_V4 = 1000
NAMES = ("S_SHARED_GUIDANCE", "S_SOURCE_GUIDANCE", "LIVE_TOOL_GUIDANCE_V4")
#: B1: the names a V4 expression may use
V4_LEAVES = frozenset({"LIVE_TOOL_GUIDANCE_V3", "S_SHARED_GUIDANCE", "S_SOURCE_GUIDANCE"})
CTRL_LIKE_RE = re.compile(r"(?i)\b(?:ctrl|control|cmd)\s*[-+]\s*[a-z0-9]+|⌘")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attr(name: str) -> str:
    value = getattr(pb, name, None)
    assert isinstance(value, str), f"T-0211 B1: prompt_builder.{name} (a str) does not exist yet"
    return value


def v4() -> str:
    return attr("LIVE_TOOL_GUIDANCE_V4")


def source() -> str:
    return Path(pb.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- predicates


def shortcut_problems(text: str, enum: list[str]) -> list[str]:
    """B2 (a): every shortcut-like token is exactly ``Ctrl+L`` and in the enum; at least one is there."""
    out = []
    tokens = CTRL_LIKE_RE.findall(text)
    if "Ctrl+L" not in tokens:
        out.append("no exact 'Ctrl+L'")
    off = sorted({t for t in tokens if t != "Ctrl+L"})
    if off:
        out.append(f"other shortcut spellings {off}")
    if "Ctrl+L" not in enum:
        out.append("Ctrl+L is not in press_key's enum")
    return out


def tool_names(text: str) -> set[str]:
    """B2 (b): tool-shaped tokens (snake_case words and ``name(``)."""
    found = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text))
    found |= set(re.findall(r"\b([A-Za-z_]+)\(", text))
    return found


def tool_problems(text: str, declared: set[str]) -> list[str]:
    out = [
        n
        for n in g3.UNDECLARED
        if re.search(rf"(?i)(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text)
    ]
    names = tool_names(text)
    if names != {"google_search"}:
        out.append(f"tool-shaped tokens {sorted(names)} (only google_search allowed)")
    out += [n for n in names if n not in declared]
    return out


#: a v4-shaped fixture for the predicates (v3 + the task's two sentences, built here from the typed literals)
GOOD = g3.GOOD + " " + S_SHARED + " " + S_SOURCE


def test_predicates_accept_good() -> None:
    """Power premise (active): the typed sentences after a hand-written v3-like sentence pass every B2 predicate."""
    enum = g3.press_key_enum()
    assert shortcut_problems(GOOD, enum) == []
    assert tool_problems(GOOD, g3.declared_tools()) == []
    assert g3.BAR_RE.search(GOOD) is None
    assert g3.FORWARD_RE.search(GOOD) is None
    assert g3.narration_problems(GOOD) == []


@pytest.mark.parametrize(
    "bad",
    ["Ctrl-L", "ctrl+l", "Control+L", "CTRL+L", "Cmd+L", "⌘L", "Ctrl + L", "Ctrl+T"],
)
def test_shortcut_predicate_power(bad: str) -> None:
    """Power (active; B2 (a) adversarial): ``Ctrl-L``, ``ctrl+l``, ``Control+L`` and the other wrong spellings in
    S_SHARED are refused."""
    text = g3.GOOD + " " + S_SHARED.replace("Ctrl+L", bad) + " " + S_SOURCE
    assert shortcut_problems(text, g3.press_key_enum()), f"power: {bad!r} passes"


def test_other_predicates_power() -> None:
    """Power (active): an undeclared tool, a second tool name, an address bar, a forward point and a confirmation
    request are each refused."""
    declared = g3.declared_tools()
    for name in g3.UNDECLARED:
        assert tool_problems(GOOD.replace("open the page", f"call {name} on the page"), declared), (
            name
        )
    assert tool_problems(GOOD.replace("open the page", "use click_at on the page"), declared)
    assert tool_problems(GOOD.replace("google_search", "web search"), declared)
    assert g3.BAR_RE.search(GOOD.replace("start page and", "address bar and"))
    assert g3.FORWARD_RE.search(GOOD.replace("read it there", "wait for it there"))
    assert g3.narration_problems(GOOD.replace("name that page", "confirm that page"))


# =====================================================================================================
# B1: v1, v2, v3 unchanged (active)
# =====================================================================================================


def test_v1_v2_v3_unchanged() -> None:
    """B1 (active): v1, v2 and v3 hash to their BASE literals, each assigned once as one literal, and
    ``LIVE_GUIDANCE["v1"|"v2"|"v3"]`` are those objects."""
    src = source()
    for name, want in (
        ("LIVE_TOOL_GUIDANCE", V1_SHA),
        ("LIVE_TOOL_GUIDANCE_V2", V2_SHA),
        ("LIVE_TOOL_GUIDANCE_V3", V3_SHA),
    ):
        assert sha(getattr(pb, name)) == want, f"T-0211 B1: {name} changed"
        found = g3.module_assignments(src, name)
        assert len(found) == 1, f"T-0211 B1: {name} is assigned once; {len(found)}"
        node = found[0].value
        assert isinstance(node, ast.Constant) and sha(str(node.value)) == want, name
    m = pb.LIVE_GUIDANCE
    assert m["v1"] is pb.LIVE_TOOL_GUIDANCE
    assert m["v2"] is pb.LIVE_TOOL_GUIDANCE_V2
    assert m["v3"] is pb.LIVE_TOOL_GUIDANCE_V3


# =====================================================================================================
# B1: the literals
# =====================================================================================================


def test_sentences_byte_exact() -> None:
    """B1: ``S_SHARED_GUIDANCE`` and ``S_SOURCE_GUIDANCE`` equal the task's sentences byte for byte (213 and 151 code
    points), each one module-level assignment of a single string literal."""
    assert (len(S_SHARED), len(S_SOURCE)) == (213, 151), "premise: the typed sentences"
    src = source()
    for name, want in (("S_SHARED_GUIDANCE", S_SHARED), ("S_SOURCE_GUIDANCE", S_SOURCE)):
        got = attr(name)
        assert got == want, f"T-0211 B1: {name} differs from the task's sentence: {got!r}"
        found = g3.module_assignments(src, name)
        assert len(found) == 1, f"T-0211 B1: {name} is assigned once at module level; {len(found)}"
        node = found[0].value
        assert isinstance(node, ast.Constant) and node.value == want, (
            f"T-0211 B1: {name} is one string literal; got {ast.dump(node)[:200] if node else None}"
        )


def _v4_shape_ok(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Name):
        return node.id in V4_LEAVES
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _v4_shape_ok(node.left) and _v4_shape_ok(node.right)
    return False


def test_v4_is_v3_plus_two_sentences() -> None:
    """B1: ``V4 == V3 + " " + S_SHARED + " " + S_SOURCE`` (the typed sentences), ``len(V4) <= 1000``; V4 is assigned
    once at module level, as one literal or a ``+`` of literals and the three names (module docstring)."""
    value = v4()
    assert value == pb.LIVE_TOOL_GUIDANCE_V3 + " " + S_SHARED + " " + S_SOURCE, (
        f"T-0211 B1: v4 is not v3 + ' ' + S_SHARED + ' ' + S_SOURCE: {value[-400:]!r}"
    )
    assert value == pb.LIVE_TOOL_GUIDANCE_V3 + " " + attr("S_SHARED_GUIDANCE") + " " + attr(
        "S_SOURCE_GUIDANCE"
    )
    assert len(value) <= MAX_V4 and len(value) == 997, f"T-0211 B1: len(v4) = {len(value)}"
    found = g3.module_assignments(source(), "LIVE_TOOL_GUIDANCE_V4")
    assert len(found) == 1, f"T-0211 B1: LIVE_TOOL_GUIDANCE_V4 is assigned once; {len(found)}"
    node = found[0].value
    assert node is not None and _v4_shape_ok(node), (
        f"T-0211 B1: V4 is a fixed literal (or literals and the three names joined by +); {ast.dump(node)[:300]}"
    )
    for name in NAMES:
        assert len(g3.module_assignments(source(), name)) == 1, name
    assert sha(value) not in (V1_SHA, V2_SHA, V3_SHA)


def test_mapping_has_v4_readonly() -> None:
    """B3: ``LIVE_GUIDANCE`` is a read-only ``Mapping`` of exactly v1..v4; v1, v2, v3 the same objects as before, v4
    the attribute; assignment, deletion and ``update`` refused."""
    value = v4()
    m = pb.LIVE_GUIDANCE
    assert isinstance(m, Mapping) and not isinstance(m, dict), type(m)
    assert sorted(m) == ["v1", "v2", "v3", "v4"], (
        f"T-0211 B3: LIVE_GUIDANCE's keys are v1..v4; {list(m)}"
    )
    assert m["v1"] is pb.LIVE_TOOL_GUIDANCE and m["v2"] is pb.LIVE_TOOL_GUIDANCE_V2
    assert m["v3"] is pb.LIVE_TOOL_GUIDANCE_V3
    assert m["v4"] is value, "T-0211 B3: LIVE_GUIDANCE['v4'] is LIVE_TOOL_GUIDANCE_V4 itself"
    with pytest.raises(TypeError):
        m["v4"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        m["v5"] = "added"  # type: ignore[index]
    with pytest.raises(TypeError):
        del m["v4"]  # type: ignore[attr-defined]
    with pytest.raises((TypeError, AttributeError)):
        m.update({"v4": "changed"})  # type: ignore[attr-defined]
    assert sorted(m) == ["v1", "v2", "v3", "v4"]


# =====================================================================================================
# B2: what v4 says
# =====================================================================================================


def test_only_ctrl_l_token() -> None:
    """B2 (a) adversarial (a misspelled combo is a refused call): the only shortcut token in V4 is exact ``Ctrl+L``,
    a member of ``press_key``'s parsed enum."""
    problems = shortcut_problems(v4(), g3.press_key_enum())
    assert problems == [], f"T-0211 B2 (a): {problems}"


def test_only_declared_tools_named() -> None:
    """B2 (b): the only tool named in V4 is ``google_search``, declared in ``tools_v1.json``; no ``open_url``,
    ``navigate``, ``goto``, ``go_to``."""
    declared = g3.declared_tools()
    assert "google_search" in declared, "premise: google_search is declared"
    problems = tool_problems(v4(), declared)
    assert problems == [], f"T-0211 B2 (b): {problems}"


def test_no_address_bar() -> None:
    """B2 (c) adversarial (the agent browser has no address bar): V4 says nothing of an address, URL or location bar
    or an omnibox."""
    hit = g3.BAR_RE.search(v4())
    assert hit is None, f"T-0211 B2 (c): {hit.group(0)!r}"


def test_points_backwards() -> None:
    """B2 (d): no part of V4 matches T-0174 A2(f)'s forward-pointing regex (power: v1 does)."""
    assert g3.FORWARD_RE.search(pb.LIVE_TOOL_GUIDANCE), "power: v1 matches"
    hit = g3.FORWARD_RE.search(v4())
    assert hit is None, f"T-0211 B2 (d): V4 points forward ({hit.group(0)!r})"


def test_no_narration_or_confirmation() -> None:
    """B2 (e): V4 asks for no narration, confirmation or permission (T-0131 C1's banned list)."""
    found = g3.narration_problems(v4())
    assert not found, f"T-0211 B2 (e): found {found}"


def test_scope_words() -> None:
    """B2 (f): S_SOURCE holds ``google_search`` (the rule is scoped to web-search answers); S_SHARED holds
    ``shared screen`` and ``own browser``; and both are in V4 once, in that order, after v3."""
    s_shared, s_source, value = attr("S_SHARED_GUIDANCE"), attr("S_SOURCE_GUIDANCE"), v4()
    assert "google_search" in s_source, f"T-0211 B2 (f): {s_source!r}"
    for w in ("shared screen", "own browser"):
        assert w in s_shared, f"T-0211 B2 (f): S_SHARED lacks {w!r}"
    assert value.count(s_shared) == 1 and value.count(s_source) == 1
    assert len(pb.LIVE_TOOL_GUIDANCE_V3) < value.index(s_shared) < value.index(s_source)
