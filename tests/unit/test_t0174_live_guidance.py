"""T-0174 group A, A1 and A2: the permanent v1 live guidance, the v2 literal and the read-only mapping
(docs/tasks/T-0174.md).

**What these rows pin** (from the criteria; where the task is silent the reading is stated):

- A1: ``prompt_builder.LIVE_TOOL_GUIDANCE`` hashes to ``768a973b...`` (the full sha below, typed from the T-0131
  pins), in the working tree and in ``prompt_builder.py`` at ``BASE`` and ``TIP`` (the module-level literal, read
  with ``ast`` from ``git show``); ``cli.LIVE_TOOL_GUIDANCE`` is the same object.
- A2: ``LIVE_TOOL_GUIDANCE_V2`` is one module-level assignment whose value is a single ``str`` constant in
  ``prompt_builder.py`` (implicitly concatenated literals are one constant), at most 400 code points; it is v1 with
  the first sentence replaced and, per amendment A2b, ``"the new screenshot"`` replaced by ``"the latest
  screenshot"`` in the second sentence only: it ends with ``" " + V2_REST``, where ``V2_REST`` (typed out below) is
  ``V1_REST`` (v1 after its first sentence ``"After an action, the next screenshot shows the screen as it is
  now."``) with exactly that one phrase changed; every other byte of sentences 2 and 3 is v1's.
- A2's forward-pointing regex applies to **the whole of v2** (A2b; before A2b it applied to the replacement sentence
  only, because the kept second sentence said "the new screenshot"); v2 contains no "new screenshot" (any case).
- The replacement "states that the screenshot in the current turn already shows the screen after the latest
  action": wording is the implementer's, so only what is checkable is pinned: it names the screenshot
  (``screenshot``, any case), is one line, ends with a full stop, and differs from v1's first sentence.
- "Quotes no session option's text, including the 'screen now' label" (H-T174-3): none of ``session.LABEL_NOW``,
  ``LABEL_UPDATED``, ``LABEL_UNCHANGED`` (with or without their brackets), the T-0134 extent clause stem, the
  T-0136 outcome clause stem, nor the phrase "screen now" (any case) appears in v2.
- ``LIVE_GUIDANCE`` is a ``Mapping`` equal to ``{"v1": v1, "v2": v2}`` holding the same objects, not a ``dict`` and
  refusing item assignment and deletion.

``test_v1_sha_permanent`` and ``test_v1_literal_over_range`` were green at BASE and stay green. The other rows were
written ahead as ``pending(task="T-0174")``, failed at BASE on a named ``AssertionError`` (the v2 seams did not
exist), and were activated for ``/implement T-0174`` group A (87172bda); the implementation is 31b91d7d. The A2b
rows (``test_v2_only_first_sentence_differs``, ``test_v2_points_backwards``, ``test_v2_no_new_screenshot``,
``test_v2_sentence_states_content``) were red against 31b91d7d (its v2 kept "the new screenshot"); A2b is
implemented in f23bb148 (``TIP``), where ``test_v2_sha_at_tip`` pins v2's sha.

Regime: CPU, pure, no measurement; n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import session
from tests import _git_range as gr

#: A1: the v1 guidance's sha256 (T-0131 C1; also pinned in ``test_persona_name.py``)
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
#: Group A's BASE: d486bcb9, main after Merge T-0172 B plus the T-0174 status commit (the branch point). A literal.
BASE = "d486bcb9"
#: The implementation tip: f23bb148, group A's A2b implementation commit (bumped from 31b91d7d by the test-author).
TIP = "f23bb148"
#: v2's sha256 at TIP (A2b implemented), pinned once it landed (review B, optional pin)
V2_SHA = "518f56a1de3faff26c7fc1b8f938c4d233e59c5379f9acc431b74b7cefc41e50"
#: v1's first sentence and the rest, typed out from the T-0131 literal (the rest is what v2 keeps, A2)
V1_FIRST = "After an action, the next screenshot shows the screen as it is now."
V1_REST = (
    "Before saying an action worked, check that the new screenshot shows its effect (for example, the cart count "
    "changed); if it does not, try again or say it did not work. When asked about a whole page (the cheapest item, "
    "every item, a list), scroll to see the rest of the page before answering."
)
#: A2b: what v2 keeps after its first sentence, typed out: ``V1_REST`` with "the new screenshot" -> "the latest
#: screenshot" in the verification sentence, nothing else
V2_REST = (
    "Before saying an action worked, check that the latest screenshot shows its effect (for example, the cart count "
    "changed); if it does not, try again or say it did not work. When asked about a whole page (the cheapest item, "
    "every item, a list), scroll to see the rest of the page before answering."
)
#: A2's forward-pointing regex, typed from the criterion
FORWARD_RE = re.compile(
    r"(?i)\b(next|new|upcoming|following)\s+screenshot|\bwill\s+(show|see|appear)|\bwait\b"
)
MAX_V2 = 400
PB_PATH = "local_astra/conductor/prompt_builder.py"
#: session option texts v2 must not quote (H-T174-3): T-0134's label openings, T-0134's extent clause stem,
#: T-0136's outcome clause stem
OPTION_STEMS = ("Page beyond this screen", "Compared with the previous screenshot")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def v2() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V2", None)
    assert isinstance(value, str), (
        "T-0174 A2: prompt_builder.LIVE_TOOL_GUIDANCE_V2 does not exist yet"
    )
    return value


def mapping() -> Any:
    value = getattr(pb, "LIVE_GUIDANCE", None)
    assert value is not None, "T-0174 A2: prompt_builder.LIVE_GUIDANCE does not exist yet"
    return value


def replacement(text: str) -> str:
    """v2's first sentence: v2 without its tail ``" " + V2_REST`` (A2/A2b: sentence 1 replaced; in sentence 2 only
    "the new screenshot" -> "the latest screenshot"; the rest verbatim)."""
    tail = " " + V2_REST
    assert text.endswith(tail), (
        "T-0174 A2/A2b: after its first sentence v2 must be v1's remaining sentences with only 'the new screenshot' "
        f"replaced by 'the latest screenshot', as its tail after one space; v2 is {text!r}"
    )
    return text[: -len(tail)]


def module_literal(source: str, name: str) -> tuple[str, ast.Assign | ast.AnnAssign]:
    """The value of the module-level assignment to ``name`` in ``source``; it must be a single ``str`` constant."""
    tree = ast.parse(source)
    found: list[ast.Assign | ast.AnnAssign] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            found.append(node)
    assert len(found) == 1, (
        f"T-0174 A2: exactly one module-level assignment to {name}; found {len(found)}"
    )
    node = found[0]
    value = node.value
    assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
        f"T-0174 A2: {name} is assigned a single string literal, not {ast.dump(value)[:200] if value else None}"
    )
    return value.value, node


# =====================================================================================================
# A1: v1 is permanent (active)
# =====================================================================================================


def test_v1_sha_permanent() -> None:
    """A1 (active): v1 hashes to 768a973b..., is v1's first sentence, one space and the typed rest, and
    ``cli.LIVE_TOOL_GUIDANCE`` is the same object."""
    assert sha(pb.LIVE_TOOL_GUIDANCE) == V1_SHA, (
        f"T-0174 A1: LIVE_TOOL_GUIDANCE changed: {pb.LIVE_TOOL_GUIDANCE!r}"
    )
    assert pb.LIVE_TOOL_GUIDANCE == V1_FIRST + " " + V1_REST, (
        "premise: the typed sentences are v1's"
    )
    assert cli.LIVE_TOOL_GUIDANCE is pb.LIVE_TOOL_GUIDANCE, (
        "T-0174 A1: cli.LIVE_TOOL_GUIDANCE is the prompt_builder object"
    )
    assert FORWARD_RE.search(V1_FIRST), "premise: v1's first sentence matches A2's forward regex"


def test_v1_literal_over_range() -> None:
    """A1 (active): in ``prompt_builder.py`` at BASE and at TIP the module-level ``LIVE_TOOL_GUIDANCE`` literal hashes
    to 768a973b... (two literals, never the working tree)."""
    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
    for rev in (BASE, TIP):
        proc = gr._git("show", f"{rev}:{PB_PATH}")
        assert proc.returncode == 0, proc.stderr
        text, _ = module_literal(proc.stdout, "LIVE_TOOL_GUIDANCE")
        assert sha(text) == V1_SHA, f"T-0174 A1: LIVE_TOOL_GUIDANCE at {rev} is {text!r}"


# =====================================================================================================
# A2: the v2 literal
# =====================================================================================================


def test_v2_literal_bounds() -> None:
    """A2 (reviewer: a computed v2): one module-level assignment of a single string literal, equal to the attribute,
    at most 400 code points, different from v1, and its sha is not v1's."""
    value = v2()
    source = (Path(pb.__file__)).read_text(encoding="utf-8")
    literal, _ = module_literal(source, "LIVE_TOOL_GUIDANCE_V2")
    assert literal == value, "T-0174 A2: the attribute is the literal's value"
    assert len(value) <= MAX_V2, f"T-0174 A2: v2 is {len(value)} code points (at most {MAX_V2})"
    assert value != pb.LIVE_TOOL_GUIDANCE and sha(value) != V1_SHA


def test_v2_only_first_sentence_differs() -> None:
    """A2/A2b (reviewer mutation: another word of sentence 2 changed): v2 is the replacement sentence, one space, then
    ``V2_REST`` byte for byte, which differs from v1's rest in exactly the one phrase (premise, checked word by word);
    the replacement is one non-empty line ending with a full stop, unlike v1's first sentence, and names the
    screenshot."""
    assert (
        V1_REST.count("the new screenshot") == 1
        and V1_REST.replace("the new screenshot", "the latest screenshot") == V2_REST
    ), "premise: V2_REST is V1_REST with the one A2b phrase changed"
    w1, w2 = V1_REST.split(" "), V2_REST.split(" ")
    assert len(w1) == len(w2) and [
        i for i, (a, b) in enumerate(zip(w1, w2, strict=True)) if a != b
    ] == [w1.index("new")], "premise: exactly one word differs"
    value = v2()
    first = replacement(value)
    assert first and first == first.strip(), f"T-0174 A2: replacement sentence {first!r}"
    assert "\n" not in first, f"T-0174 A2: the replacement is one line: {first!r}"
    assert first.endswith("."), (
        f"T-0174 A2: the replacement is a sentence ending in a full stop: {first!r}"
    )
    assert first != V1_FIRST, "T-0174 A2: the first sentence is replaced"
    assert "screenshot" in first.lower(), (
        f"T-0174 A2: the replacement states what the screenshot shows; it does not name it: {first!r}"
    )
    assert value == first + " " + V2_REST


def test_v2_points_backwards() -> None:
    """A2/A2b adversarial (the failure mode: the model quotes the guidance as a reason to wait): no part of v2 matches
    A2's forward-pointing regex; v1's first sentence and v1's second sentence each do (power); v2 quotes none of the
    session's option texts, the "screen now" label included (H-T174-3)."""
    value = v2()
    hit = FORWARD_RE.search(value)
    assert hit is None, f"T-0174 A2/A2b: v2 points forward ({hit.group(0)!r}): {value!r}"
    assert FORWARD_RE.search(V1_FIRST), "power: v1's first sentence matches the same regex"
    assert FORWARD_RE.search(V1_REST.split(". ")[0]), "power: v1's second sentence matches too"
    low = value.lower()
    assert "screen now" not in low, (
        f"T-0174 A2 (H-T174-3): v2 quotes the 'screen now' label: {value!r}"
    )
    for label in (session.LABEL_NOW, session.LABEL_UPDATED, session.LABEL_UNCHANGED):
        assert label.lower() not in low and label.rstrip(".").lower() not in low, (
            f"T-0174 A2 (H-T174-3): v2 quotes the live label {label!r}"
        )
    for stem in OPTION_STEMS:
        assert stem.lower() not in low, f"T-0174 A2: v2 quotes the session option text {stem!r}"


#: review B: replacement sentences that satisfied every other A2 row but say nothing of A2's content (survivors)
WEAK_FIRST = (
    "The screenshot shows the screen.",
    "The screenshot shows the screen as it is now.",
    "The screenshot in this turn shows the screen.",
    "The screenshot already shows the screen after your action.",
    "This turn's screenshot shows the screen after your action.",
)
#: the 31b91d7d sentence (the implementation's wording), which states A2's content
IMPL_FIRST = "The latest screenshot in this turn already shows the screen as it is after any action you have taken."


def content_problems(first: str) -> list[str]:
    """A2's content, as checkable words: the replacement says the screenshot of **this** turn (``this turn`` or
    ``the current turn``) **already** shows the screen **after** an action. Returns what is missing."""
    low = " ".join(first.lower().split())
    out = []
    if "screenshot" not in low:
        out.append("screenshot")
    if not re.search(r"\balready\b", low):
        out.append("already")
    if "this turn" not in low and "the current turn" not in low:
        out.append("this turn / the current turn")
    if not re.search(r"\bafter\b", low):
        out.append("after")
    elif not re.search(r"\bafter\b.*\baction\b", low):
        out.append("after ... action")
    return out


def test_content_predicate_power() -> None:
    """Review B (active): every weak sentence lacks part of A2's content; the implementation's sentence has all of
    it."""
    for weak in WEAK_FIRST:
        assert content_problems(weak), f"power: {weak!r} passes the content pin"
    assert content_problems(IMPL_FIRST) == [], content_problems(IMPL_FIRST)


def test_v2_sentence_states_content() -> None:
    """A2 (review B, the weak-v2 survivor): v2's replaced first sentence says the screenshot in this turn (or the
    current turn) already shows the screen after an action: it contains ``already``, ``this turn`` or ``the current
    turn``, and ``after`` followed later by ``action``."""
    first = replacement(v2())
    missing = content_problems(first)
    assert missing == [], (
        f"T-0174 A2: the v2 sentence does not state A2's content (missing {missing}): {first!r}"
    )


def test_v2_sha_at_tip() -> None:
    """A2/A2b (review B, pinned once landed): v2 hashes to 518f56a1... in the working tree and as the module-level
    literal of ``prompt_builder.py`` at TIP f23bb148."""
    assert sha(v2()) == V2_SHA, f"T-0174 A2: v2 changed after review: {v2()!r}"
    gr.require_commits(TIP)
    proc = gr._git("show", f"{TIP}:{PB_PATH}")
    assert proc.returncode == 0, proc.stderr
    text, _ = module_literal(proc.stdout, "LIVE_TOOL_GUIDANCE_V2")
    assert sha(text) == V2_SHA, f"T-0174 A2: LIVE_TOOL_GUIDANCE_V2 at {TIP} is {text!r}"


def test_v2_no_new_screenshot() -> None:
    """A2b: v2 contains no "new screenshot" (any case, any spacing) and says "the latest screenshot" in the
    verification sentence; v1 keeps its "the new screenshot" (A1)."""
    value = v2()
    assert re.search(r"(?i)\bnew\s+screenshot", value) is None, (
        f"T-0174 A2b: v2 still says 'new screenshot': {value!r}"
    )
    assert "check that the latest screenshot shows its effect" in value, (
        f"T-0174 A2b: v2's verification sentence says 'the latest screenshot': {value!r}"
    )
    assert "check that the new screenshot shows its effect" in pb.LIVE_TOOL_GUIDANCE


def test_mapping_readonly() -> None:
    """A2, re-scoped by T-0191 A3 (the key set gains v3) and by T-0211 B3 (it gains v4): ``LIVE_GUIDANCE`` is a
    read-only ``Mapping`` of exactly ``v1``, ``v2``, ``v3`` and ``v4`` (``v1``, ``v2`` and ``v3`` the same objects as
    the module attributes, not equal copies; ``v4`` is ``LIVE_TOOL_GUIDANCE_V4``); assignment, deletion and
    ``update`` are refused and leave it unchanged. Pending for T-0211 (at T-0211's BASE the mapping holds v1..v3)."""
    m = mapping()
    value = v2()
    v3 = getattr(pb, "LIVE_TOOL_GUIDANCE_V3", None)
    v4 = getattr(pb, "LIVE_TOOL_GUIDANCE_V4", None)
    assert isinstance(m, Mapping), f"T-0174 A2: LIVE_GUIDANCE is a Mapping, not {type(m)!r}"
    assert not isinstance(m, dict), "T-0174 A2: LIVE_GUIDANCE is read-only (a dict is writable)"
    assert isinstance(v4, str), "T-0211 B1: prompt_builder.LIVE_TOOL_GUIDANCE_V4 does not exist yet"
    assert sorted(m) == ["v1", "v2", "v3", "v4"] and len(m) == 4, (
        f"T-0211 B3: LIVE_GUIDANCE's keys are v1, v2, v3, v4; {list(m)}"
    )
    assert isinstance(v3, str), "T-0191 A1: prompt_builder.LIVE_TOOL_GUIDANCE_V3 does not exist yet"
    assert m["v1"] is pb.LIVE_TOOL_GUIDANCE and m["v2"] is value and m["v3"] is v3 and m["v4"] is v4
    with pytest.raises(TypeError):
        m["v1"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        m["v5"] = "added"  # type: ignore[index]
    with pytest.raises(TypeError):
        del m["v2"]  # type: ignore[attr-defined]
    with pytest.raises((TypeError, AttributeError)):
        m.update({"v1": "changed"})  # type: ignore[attr-defined]
    assert dict(m) == {"v1": pb.LIVE_TOOL_GUIDANCE, "v2": value, "v3": v3, "v4": v4}
    assert sha(m["v1"]) == V1_SHA
