"""T-0191 group A, A1-A3 (text half), A7 (v1/v2 bytes) and A8: live guidance v3, the literal and what its one added
sentence says (docs/tasks/T-0191.md).

**What these rows pin** (from the criteria; where the task is silent the reading is stated):

- A1: ``prompt_builder.LIVE_TOOL_GUIDANCE_V3`` is one module-level assignment of a single ``str`` constant (implicitly
  concatenated literals are one constant), equal to ``LIVE_TOOL_GUIDANCE_V2 + " " + S``; S is one line, stripped, ends
  with ``.``, and holds no other ``". "``, ``"? "`` or ``"! "``; ``len(S) <= 240`` and ``len(V3) <= 640`` code points.
- A2 (a): S contains ``Ctrl+L`` exactly, a member of ``tools_v1.json``'s ``press_key`` ``combo`` enum (parsed here, not
  typed); every ``Ctrl``-like token in S is an exact enum member, and the set of them is ``{"Ctrl+L"}`` (Q5 at its
  recommendation, "Ctrl+L only"); no ``(?i)ctrl-|control\\+|cmd\\+|⌘`` and no lower-case ``ctrl+l``.
- A2 (b): S says to type the address into the start page's address box and press Enter. Checkable words, in order:
  ``Ctrl+L``, then ``type``, then ``(?i)\\benter\\b``; ``start page`` and ``address`` appear (reading: "the start
  page's address box" names both).
- A2 (c): S names no tool absent from ``tools_v1.json``: none of ``open_url``, ``navigate``, ``goto``, ``go_to``
  (word match, any case; **reading, reported**: the criterion lists ``navigate`` as a tool name, so the plain English
  verb is refused too), and every snake_case or ``name(`` token in S is a declared tool name.
- A2 (d): no ``address bar`` / ``URL bar`` (any case, any spacing), nor ``location bar`` or ``omnibox`` (same claim).
- A2 (e): at least one example address in S (a host-like token whose last label has two or more letters and is not a
  file suffix); every host-like token passes ``/web.html``'s ``target()`` rules as the criterion states them (a dot,
  no whitespace, scheme absent or ``https``) and is under none of the smoke's targets (``photopea.com``,
  ``wikipedia.org``, ``amazon.*``, ``duckduckgo.com``, ``example.com``).
- A2 (f): the whole of V3 fails T-0174 A2's forward-pointing regex.
- A2 (g): S asks for no narration and no confirmation: the banned list of
  ``test_live_tool_guidance.py::test_no_confirmation_or_narration_request``, typed out below.
- Implementation notes: S promises no new tab (Ctrl+L replaces the page) and names no ``Ctrl+T`` (Q5).
- A3 (text half): ``LIVE_GUIDANCE`` is read-only, exactly ``{"v1", "v2", "v3"}``, v1 and v2 the same objects as the
  module attributes. (``config`` and ``cli`` are ``test_t0191_cli_guidance.py``.)
- A7: v1 still hashes to ``768a973b…``; v2 still hashes to ``518f56a1…`` (T-0174's pin at f23bb148).
- A8: docs/02 §5 holds exactly one ``**Spec delta (T-0191)**`` paragraph naming v3, v2, ``--no-web``, ``page_fact``,
  the prefix, the default and T-0174.

``test_v1_v2_unchanged`` and the predicate power rows are green at BASE (f1caef6a). Every other row was written ahead
as pending and activated at the start of ``/implement T-0191`` A (2026-10-28): at BASE ``LIVE_TOOL_GUIDANCE_V3`` does
not exist, so each fails on a named ``AssertionError``.

Regime: CPU, pure, no measurement; n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from local_astra.conductor import prompt_builder as pb
from tests.unit import test_t0107a_docs as t7d

REPO = Path(__file__).resolve().parents[2]
TOOLS_FILE = REPO / "local_astra" / "toolcall" / "tools_v1.json"
DOC02 = REPO / "docs" / "02-architecture.md"
#: A7: v1's sha256 (T-0131 C1, T-0174 A1) and v2's (T-0174 A2b, pinned at f23bb148)
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
V2_SHA = "518f56a1de3faff26c7fc1b8f938c4d233e59c5379f9acc431b74b7cefc41e50"
MAX_S = 240
MAX_V3 = 640
#: T-0174 A2's forward-pointing regex, typed from the criterion
FORWARD_RE = re.compile(
    r"(?i)\b(next|new|upcoming|following)\s+screenshot|\bwill\s+(show|see|appear)|\bwait\b"
)
#: A2 (a): other spellings of a shortcut
OTHER_SPELLING_RE = re.compile(r"(?i)ctrl-|control\+|cmd\+|⌘")
#: every Ctrl-like token (any case, optional spaces around the plus), to compare with the enum
CTRL_TOKEN_RE = re.compile(r"(?i)\bctrl\s*\+\s*[a-z0-9]+")
#: A2 (c): the tools that do not exist in tools_v1 (the criterion's list)
UNDECLARED = ("open_url", "navigate", "goto", "go_to")
#: A2 (d)
BAR_RE = re.compile(r"(?i)\b(address|url|location)\s*bar\b|\bomnibox\b")
#: A2 (e): host-like tokens: an optional scheme, dotted labels ending in a 2+ letter label, an optional port and path
HOST_RE = re.compile(
    r"(?i)(?:\b[a-z][a-z0-9+.\-]*://)?\b(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+[a-z]{2,}\b(?::\d+)?(?:/[^\s\"'”’)]*)?"
)
FILE_SUFFIXES = frozenset({"html", "htm", "json", "js", "css", "png", "jpg", "jpeg", "txt", "py"})
SMOKE_DOMAINS = ("photopea.com", "wikipedia.org", "duckduckgo.com", "example.com")
AMAZON_RE = re.compile(r"(^|\.)amazon\.[a-z.]+$")
#: A2 (g): typed from test_live_tool_guidance.py::test_no_confirmation_or_narration_request (T-0131 C1)
NARRATION_BANNED = (
    "confirm",
    "approv",
    "wait for",
    "each action",
    "every action",
    "before each",
    "before every",
    "tell the user",
    "aloud",
    "out loud",
    "permission",
    "ask the user",
    "ask for",
    "ask before",
    "ask first",
    "check with the user",
    "narrat",
    "announce",
    "each step",
    "every step",
    "describe each",
    "describe every",
    "tell the user what",
    "say what you",
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def v3() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V3", None)
    assert isinstance(value, str), (
        "T-0191 A1: prompt_builder.LIVE_TOOL_GUIDANCE_V3 (a str) does not exist yet"
    )
    return value


def sentence() -> str:
    """S: v3 after v2 and one space (A1). A named failure when v3 is not v2 + " " + S."""
    value, v2 = v3(), pb.LIVE_TOOL_GUIDANCE_V2
    head = v2 + " "
    assert value.startswith(head), (
        f"T-0191 A1: v3 begins with v2 and one space, byte for byte; v3 is {value!r}"
    )
    s = value[len(head) :]
    assert s, "T-0191 A1: v3 adds a sentence after v2"
    return s


def module_assignments(source: str, name: str) -> list[ast.Assign | ast.AnnAssign]:
    found: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            found.append(node)
    return found


def press_key_enum() -> list[str]:
    """``press_key``'s ``combo`` enum, parsed from ``tools_v1.json``."""
    data = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("tools", data.get("functions", []))
    for item in items:
        fn = item.get("function", item) if isinstance(item, dict) else {}
        if fn.get("name") == "press_key":
            enum = fn["parameters"]["properties"]["combo"]["enum"]
            assert isinstance(enum, list) and enum, "premise: press_key's combo enum"
            return [str(x) for x in enum]
    raise AssertionError(f"premise: press_key is declared in {TOOLS_FILE}")


def declared_tools() -> set[str]:
    """Every declared tool name and parameter name in ``tools_v1.json`` (a parameter such as ``press_enter`` is
    declared too, so naming it is not naming an undeclared tool)."""
    data = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("tools", data.get("functions", []))
    names = set()
    for item in items:
        fn = item.get("function", item) if isinstance(item, dict) else {}
        if isinstance(fn.get("name"), str):
            names.add(fn["name"])
            names.update((fn.get("parameters") or {}).get("properties", {}))
    return names


# --------------------------------------------------------------------------------------------- predicates


def combo_problems(s: str, enum: list[str]) -> list[str]:
    """A2 (a) as a predicate: what is wrong with how ``s`` spells the shortcut."""
    out = []
    if "Ctrl+L" not in s:
        out.append("no exact 'Ctrl+L'")
    if OTHER_SPELLING_RE.search(s):
        out.append(f"other spelling {OTHER_SPELLING_RE.search(s).group(0)!r}")  # type: ignore[union-attr]
    if "ctrl+l" in s:
        out.append("lower-case 'ctrl+l'")
    tokens = CTRL_TOKEN_RE.findall(s)
    off = [t for t in tokens if t not in enum]
    if off:
        out.append(f"tokens not in the enum {off}")
    if set(tokens) - {"Ctrl+L"}:
        out.append(f"shortcuts other than Ctrl+L {sorted(set(tokens) - {'Ctrl+L'})} (Q5)")
    if re.search(r"(?<![A-Za-z+])⌘|\bCmd\b", s):
        out.append("a Mac shortcut")
    return out


def host_tokens(s: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?") for m in HOST_RE.finditer(s)]


def host_of(token: str) -> str:
    rest = token.split("://", 1)[1] if "://" in token else token
    return rest.split("/", 1)[0].split(":", 1)[0].lower().rstrip(".")


def target_accepts(token: str) -> bool:
    """A Python oracle of the three rules A2 (e) states for ``/web.html``'s ``target()``: has a dot, no whitespace,
    scheme absent or ``https``."""
    if re.search(r"\s", token) or "." not in token:
        return False
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):(?!\d)", token)
    return m is None or m.group(1).lower() == "https"


def is_smoke_target(token: str) -> bool:
    host = host_of(token)
    return any(host == d or host.endswith("." + d) for d in SMOKE_DOMAINS) or bool(
        AMAZON_RE.search(host)
    )


def example_problems(s: str) -> list[str]:
    tokens = host_tokens(s)
    out = []
    examples = [t for t in tokens if host_of(t).rsplit(".", 1)[-1] not in FILE_SUFFIXES]
    if not examples:
        out.append("no example address")
    for t in tokens:
        if not target_accepts(t):
            out.append(f"{t!r} is refused by web.html's target() rules")
        if is_smoke_target(t):
            out.append(f"{t!r} is a smoke target")
    return out


def steps_problems(s: str) -> list[str]:
    """A2 (b) as a predicate: Ctrl+L, then type, then Enter; names the start page and the address."""
    low = s.lower()
    out = []
    if "start page" not in low:
        out.append("start page")
    if "address" not in low:
        out.append("address")
    i_ctrl = s.find("Ctrl+L")
    m_type = re.search(r"(?i)\btype\b", s)
    m_enter = re.search(r"(?i)\benter\b", s)
    if m_type is None:
        out.append("type")
    if m_enter is None:
        out.append("enter")
    in_order = (
        i_ctrl < 0 or m_type is None or m_enter is None or i_ctrl < m_type.start() < m_enter.start()
    )
    if not in_order:
        out.append("order Ctrl+L < type < Enter")
    return out


def tool_problems(s: str, declared: set[str]) -> list[str]:
    out = [
        name
        for name in UNDECLARED
        if re.search(rf"(?i)(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", s)
    ]
    for m in re.finditer(r"\b[a-z]+(?:_[a-z]+)+\b|\b([A-Za-z_]+)\(", s):
        tok = m.group(1) or m.group(0)
        if tok not in declared:
            out.append(tok)
    return out


def narration_problems(s: str) -> list[str]:
    low = s.lower()
    return [b for b in NARRATION_BANNED if b in low]


# =====================================================================================================
# predicate power (active): each predicate refuses the shapes it exists for, and accepts a good sentence
# =====================================================================================================

#: a sentence written here to satisfy every A2 rule (not the implementer's wording; a fixture for the predicates)
GOOD = (
    "To go to a different website, press Ctrl+L to open the start page, then type the web address (like "
    "news.ycombinator.com) into its address box and press Enter."
)


def test_predicates_accept_a_good_sentence() -> None:
    """Power premise (active): the hand-written ``GOOD`` passes every predicate (so the predicates are not refusing
    everything)."""
    enum = press_key_enum()
    assert "Ctrl+L" in enum and "Ctrl+T" in enum, (
        f"premise: the enum holds Ctrl+L and Ctrl+T; {enum}"
    )
    assert combo_problems(GOOD, enum) == []
    assert steps_problems(GOOD) == []
    assert tool_problems(GOOD, declared_tools()) == []
    assert BAR_RE.search(GOOD) is None
    assert example_problems(GOOD) == [], example_problems(GOOD)
    assert FORWARD_RE.search(GOOD) is None
    assert narration_problems(GOOD) == []
    assert len(GOOD) <= MAX_S and GOOD.endswith(".")


@pytest.mark.parametrize(
    "bad",
    [
        GOOD.replace("Ctrl+L", "Ctrl-L"),
        GOOD.replace("Ctrl+L", "Control+L"),
        GOOD.replace("Ctrl+L", "ctrl+l"),
        GOOD.replace("Ctrl+L", "CTRL+L"),
        GOOD.replace("Ctrl+L", "Cmd+L"),
        GOOD.replace("Ctrl+L", "⌘L"),
        GOOD.replace("Ctrl+L", "Ctrl + L"),
        GOOD.replace("Ctrl+L", "Ctrl+T"),
        GOOD.replace("press Ctrl+L", "press Ctrl+L or Ctrl+T"),
    ],
)
def test_combo_predicate_power(bad: str) -> None:
    """Power (active, the adversarial wrong-spelling mutations): each misspelling, a non-enum token and a second
    shortcut are refused."""
    assert combo_problems(bad, press_key_enum()), f"power: {bad!r} passes the combo predicate"


@pytest.mark.parametrize(
    "token",
    [
        "en.wikipedia.org",
        "www.photopea.com",
        "https://photopea.com",
        "amazon.com",
        "www.amazon.co.uk",
        "duckduckgo.com/html/?q=x",
        "example.com",
        "http://news.ycombinator.com",
        "ftp://news.ycombinator.com",
    ],
)
def test_example_predicate_power(token: str) -> None:
    """Power (active): a smoke target or a scheme ``target()`` would not take (as A2 (e) states the rule) is
    refused when it is S's example."""
    bad = GOOD.replace("news.ycombinator.com", token)
    assert token in host_tokens(bad) or any(token.startswith(t) for t in host_tokens(bad)), (
        f"premise: the token is found: {host_tokens(bad)}"
    )
    assert example_problems(bad), f"power: {bad!r} passes the example predicate"


def test_other_predicates_power() -> None:
    """Power (active): an "address bar", an undeclared tool name (``open_url``, ``navigate``, ``goto``), a forward
    point, a confirmation request, a wrong order and a missing Enter are each refused."""
    assert BAR_RE.search(GOOD.replace("its address box", "the address bar"))
    assert BAR_RE.search(GOOD.replace("its address box", "the URL bar"))
    declared = declared_tools()
    assert "open_url" not in declared and "navigate" not in declared, (
        "premise: tools_v1 has neither"
    )
    for name in UNDECLARED:
        assert tool_problems(GOOD.replace("press Ctrl+L", f"call {name}"), declared), name
    assert tool_problems(GOOD.replace("press Ctrl+L", "use open_url("), declared)
    assert FORWARD_RE.search(
        GOOD.replace("press Enter.", "press Enter; the next screenshot shows it.")
    )
    assert narration_problems(GOOD.replace("To go", "After you confirm with the user, to go"))
    assert steps_problems(GOOD.replace("press Enter", "click Go"))
    assert steps_problems(
        "Type the web address into the start page's address box, press Enter, then press Ctrl+L."
    )
    assert example_problems(GOOD.replace(" (like news.ycombinator.com)", ""))


# =====================================================================================================
# A7: v1 and v2 unchanged (active: green at BASE)
# =====================================================================================================


def test_v1_v2_unchanged() -> None:
    """A7 (active): ``LIVE_TOOL_GUIDANCE`` still hashes to 768a973b..., ``LIVE_TOOL_GUIDANCE_V2`` to 518f56a1...,
    each assigned once at module level as a single literal, and ``LIVE_GUIDANCE["v1"]``/``["v2"]`` are those
    objects."""
    assert sha(pb.LIVE_TOOL_GUIDANCE) == V1_SHA, f"T-0191 A7: v1 changed: {pb.LIVE_TOOL_GUIDANCE!r}"
    assert sha(pb.LIVE_TOOL_GUIDANCE_V2) == V2_SHA, (
        f"T-0191 A7: v2 changed: {pb.LIVE_TOOL_GUIDANCE_V2!r}"
    )
    source = Path(pb.__file__).read_text(encoding="utf-8")
    for name, want in (("LIVE_TOOL_GUIDANCE", V1_SHA), ("LIVE_TOOL_GUIDANCE_V2", V2_SHA)):
        found = module_assignments(source, name)
        assert len(found) == 1, f"T-0191 A7: {name} is assigned once; {len(found)}"
        value = found[0].value
        assert isinstance(value, ast.Constant) and isinstance(value.value, str), name
        assert sha(value.value) == want, name
    m = pb.LIVE_GUIDANCE
    assert m["v1"] is pb.LIVE_TOOL_GUIDANCE and m["v2"] is pb.LIVE_TOOL_GUIDANCE_V2


# =====================================================================================================
# A1: the literal
# =====================================================================================================


def test_v3_is_v2_plus_one_sentence() -> None:
    """A1: one module-level assignment of a single string literal equal to the attribute; v3 == v2 + " " + S; S is
    one stripped line ending with "." and holds no other ". ", "? " or "! "; ``len(S) <= 240``,
    ``len(V3) <= 640``."""
    value = v3()
    source = Path(pb.__file__).read_text(encoding="utf-8")
    found = module_assignments(source, "LIVE_TOOL_GUIDANCE_V3")
    assert len(found) == 1, (
        f"T-0191 A1: LIVE_TOOL_GUIDANCE_V3 is assigned exactly once at module level; found {len(found)}"
    )
    node = found[0].value
    assert isinstance(node, ast.Constant) and isinstance(node.value, str), (
        "T-0191 A1: LIVE_TOOL_GUIDANCE_V3 is one string literal (not v2 + ..., not an f-string, not a call); "
        f"got {ast.dump(node)[:200] if node is not None else None}"
    )
    assert node.value == value, "T-0191 A1: the attribute is the literal's value"
    s = sentence()
    assert value == pb.LIVE_TOOL_GUIDANCE_V2 + " " + s
    assert s == s.strip() and "\n" not in s and "\r" not in s, (
        f"T-0191 A1: S is one stripped line: {s!r}"
    )
    assert s.endswith("."), f"T-0191 A1: S ends with a full stop: {s!r}"
    for sep in (". ", "? ", "! "):
        assert sep not in s, f"T-0191 A1: S is one sentence; it holds {sep!r}: {s!r}"
    assert len(s) <= MAX_S, f"T-0191 A1: S is {len(s)} code points (at most {MAX_S})"
    assert len(value) <= MAX_V3, f"T-0191 A1: v3 is {len(value)} code points (at most {MAX_V3})"
    assert sha(value) not in (V1_SHA, V2_SHA)


def test_mapping_has_v3_readonly() -> None:
    """A3: ``LIVE_GUIDANCE`` is a read-only ``Mapping`` of exactly v1, v2 and v3; v1 and v2 the same objects as
    before, v3 the attribute; assignment, deletion and ``update`` refused and nothing changed. Re-scoped by T-0211
    B3 (the key set gains v4, ``LIVE_TOOL_GUIDANCE_V4``; the junk key becomes ``"v5"``); pending for T-0211."""
    value = v3()
    v4 = getattr(pb, "LIVE_TOOL_GUIDANCE_V4", None)
    m = pb.LIVE_GUIDANCE
    assert isinstance(m, Mapping) and not isinstance(m, dict), type(m)
    assert isinstance(v4, str), "T-0211 B1: prompt_builder.LIVE_TOOL_GUIDANCE_V4 does not exist yet"
    assert sorted(m) == ["v1", "v2", "v3", "v4"] and len(m) == 4, (
        f"T-0211 B3: LIVE_GUIDANCE's keys are v1, v2, v3, v4; {list(m)}"
    )
    assert m["v1"] is pb.LIVE_TOOL_GUIDANCE and m["v2"] is pb.LIVE_TOOL_GUIDANCE_V2
    assert m["v3"] is value, "T-0191 A3: LIVE_GUIDANCE['v3'] is the attribute object"
    assert m["v4"] is v4, "T-0211 B3: LIVE_GUIDANCE['v4'] is the attribute object"
    with pytest.raises(TypeError):
        m["v3"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        m["v5"] = "added"  # type: ignore[index]
    with pytest.raises(TypeError):
        del m["v3"]  # type: ignore[attr-defined]
    with pytest.raises((TypeError, AttributeError)):
        m.update({"v3": "changed"})  # type: ignore[attr-defined]
    assert dict(m) == {
        "v1": pb.LIVE_TOOL_GUIDANCE,
        "v2": pb.LIVE_TOOL_GUIDANCE_V2,
        "v3": value,
        "v4": v4,
    }


# =====================================================================================================
# A2: what S says
# =====================================================================================================


def test_combo_token_is_in_enum() -> None:
    """A2 (a) adversarial (the wrong spelling the model would copy into a refused call): S holds ``Ctrl+L``
    exactly, a member of the parsed ``press_key`` enum; no ``Ctrl-L``, ``Control+L``, ``Cmd+L``, ``⌘``, lower-case
    ``ctrl+l``, or any Ctrl token outside the enum; Ctrl+L is the only shortcut named (Q5)."""
    s = sentence()
    enum = press_key_enum()
    assert "Ctrl+L" in enum, f"premise: Ctrl+L is in the enum {enum}"
    problems = combo_problems(s, enum)
    assert problems == [], f"T-0191 A2 (a): {problems}: {s!r}"


def test_says_type_address_and_enter() -> None:
    """A2 (b): S says, in order, Ctrl+L, type (the address into the start page's address box), Enter
    (``(?i)\\benter\\b``)."""
    s = sentence()
    problems = steps_problems(s)
    assert problems == [], f"T-0191 A2 (b): S is missing {problems}: {s!r}"


def test_no_undeclared_tool_named() -> None:
    """A2 (c): S names none of ``open_url``, ``navigate``, ``goto``, ``go_to`` (absent from tools_v1), and every
    tool-shaped token in it is declared."""
    s = sentence()
    declared = declared_tools()
    assert not set(UNDECLARED) & declared, "premise: none of the four is declared"
    problems = tool_problems(s, declared)
    assert problems == [], f"T-0191 A2 (c): S names {problems}: {s!r}"


def test_no_address_bar_claim() -> None:
    """A2 (d) adversarial (the agent browser has no address bar): S says nothing of an address, URL or location bar
    or an omnibox; v3 as a whole neither."""
    s = sentence()
    for text in (s, v3()):
        hit = BAR_RE.search(text)
        assert hit is None, f"T-0191 A2 (d): {hit.group(0)!r} in {text!r}"


def test_example_host_accepted_by_web_html() -> None:
    """A2 (e) adversarial (the smoke must not be taught its answer): S holds at least one example address; each
    host-like token passes the three ``target()`` rules (dot, no whitespace, scheme absent or https) and is under no
    smoke target."""
    s = sentence()
    problems = example_problems(s)
    assert problems == [], f"T-0191 A2 (e): {problems}; host-like tokens {host_tokens(s)}: {s!r}"


def test_v3_points_backwards() -> None:
    """A2 (f): no part of v3 matches T-0174 A2's forward-pointing regex (power: v1 does)."""
    value = v3()
    assert FORWARD_RE.search(pb.LIVE_TOOL_GUIDANCE), "power: v1 matches the regex"
    hit = FORWARD_RE.search(value)
    assert hit is None, f"T-0191 A2 (f): v3 points forward ({hit.group(0)!r}): {value!r}"


def test_no_narration_or_confirmation() -> None:
    """A2 (g): S (and v3) asks for no narration, confirmation or permission (T-0131 C1's banned list)."""
    s = sentence()
    for text in (s, v3()):
        found = narration_problems(text)
        assert not found, f"T-0191 A2 (g): found {found} in {text!r}"


def test_no_false_promise() -> None:
    """Implementation notes (S must not promise what the executor does not do): Ctrl+L replaces the current page with
    the start page, so S says nothing of a new tab or window; and it does not name Ctrl+T (Q5)."""
    s = sentence()
    low = s.lower()
    assert not re.search(r"\bnew\s+(tab|window)\b", low), f"T-0191: S promises a new tab: {s!r}"
    assert "Ctrl+T" not in s and "ctrl+t" not in low, f"T-0191 Q5: Ctrl+L only: {s!r}"


# =====================================================================================================
# A8: docs/02 §5
# =====================================================================================================


def test_docs_spec_delta() -> None:
    """A8: docs/02 §5 holds exactly one paragraph beginning ``**Spec delta (T-0191)**`` that names v3 and v2,
    ``--no-web``, ``page_fact``, the prefix property, the default (still v2) and the T-0174 amendment."""
    sec = t7d.section(DOC02.read_text(encoding="utf-8"), "## 5. ")
    paras = [p for p in t7d.paragraphs(sec) if p.lstrip().startswith("**Spec delta (T-0191)")]
    assert len(paras) == 1, f"T-0191 A8: one T-0191 spec delta in docs/02 §5; found {len(paras)}"
    p = paras[0]
    missing = [
        w
        for w in ("v3", "v2", "--no-web", "page_fact", "prefix", "default", "T-0174")
        if w not in p
    ]
    assert not missing, f"T-0191 A8: the paragraph does not name {missing}"
