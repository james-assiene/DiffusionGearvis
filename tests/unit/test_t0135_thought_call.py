"""T-0135 A1 and A3: the pure thought-call recovery rule ``local_astra.toolcall.thought_call.recover``.

A1, from the task file: ``recover(thought: str) -> str | None`` is stdlib-only, linear in ``len(thought)`` and never
raises. It returns ``thought[k:]``, where ``k`` is the index of the **last** ``<|tool_call>``, iff
``thought[k:].rstrip()`` ends with ``<tool_call|>``; otherwise ``None`` (no tag; any non-whitespace text after the final
``<tool_call|>``). It does not parse or validate: judging is ``live.classify``'s job.

Every expected value below is written out by hand from that sentence, or comes from the T-0132 fixture
(``tests/fixtures/t0135/thought_tails.jsonl``), or from a brute-force oracle over every index of a short string
(``_oracle``: try every ``k`` and keep the largest one where the tag starts). Nothing here imports a regex of the
rule.

The module does not exist at f3712c8 (no stub was planned), so the rows fail with an ``AssertionError`` naming it, not
with an import error at collection.

Regime: CPU, pure, deterministic; Hypothesis rows use the project profile (no seed needed for a property).
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.toolcall import live
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.toolcall.thought_call"
MODULE_FILE = REPO / "local_astra" / "toolcall" / "thought_call.py"
FIXTURE = REPO / "tests" / "fixtures" / "t0135" / "thought_tails.jsonl"
TAG = "<|tool_call>"
CLOSE = "<tool_call|>"
Q = '<|"|>'

#: T-0135's BASE for group A1: main at the T-0135 plan commit f3712c8 (the branch point of ``T-0135-thought-calls``);
#: a literal, never a moving ref.
BASE = "f3712c8af75a97c29c8a0d9c519a9285518941fe"
#: The A1 implementation tip: 6eb87ae (branch ``T-0135-a1-impl``; was f3712c8, the plan). Bumped by the test-author
#: after each implementation commit; never the working tree.
TIP_IMPL = "6eb87aeacd92c23ad42f6463444fef4b7b3ddf29"
#: A3's frozen paths (group A1's closed range).
FROZEN_A3 = (
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/router.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "tests/fakes",
)
#: Power for A3's predicate: T-0102's commit 9b283f1 changed ``toolcall/live.py`` (its parent is b81f5e6).
POWER_PARENT = "b81f5e6efa91b984aafcff121abf3dd178f937de"
POWER = "9b283f18797c6d6336f82b1a61691902d019f45a"


def recover_fn() -> Callable[[str], str | None]:
    try:
        mod = importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODULE:
            raise
        raise AssertionError(
            "T-0135 A1: local_astra/toolcall/thought_call.py does not exist yet"
        ) from None
    fn = getattr(mod, "recover", None)
    assert callable(fn), "T-0135 A1: thought_call.recover is missing"
    return fn  # type: ignore[no-any-return]


def _oracle(thought: str) -> str | None:
    """Brute force over every index: the largest k where the tag starts, then the A1 test on thought[k:]."""
    ks = [k for k in range(len(thought)) if thought[k : k + len(TAG)] == TAG]
    if not ks:
        return None
    tail = thought[max(ks) :]
    return tail if tail.rstrip()[-len(CLOSE) :] == CLOSE else None


TYPE_ADA = f"{TAG}call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},x:305,y:448}}{CLOSE}"
CLICK = f"{TAG}call:click_at{{x:155,y:900}}{CLOSE}"
PLAN = 'The email field is at approximately (305, 448).\nNow I need to fill in the email "ada@example.com".\n'


def _fixture_row(source_prefix: str, request_id: str) -> dict[str, object]:
    for line in FIXTURE.read_text().splitlines():
        row = json.loads(line)
        if row["source"].startswith(source_prefix) and row["request_id"] == request_id:
            return row  # type: ignore[no-any-return]
    raise AssertionError(f"premise: {source_prefix} {request_id} is in the fixture")


def _thought_block(row: dict[str, object]) -> str:
    commits = row["commits"]
    assert isinstance(commits, list)
    assert all(c["segment"] == "thought" and c["thought_open"] for c in commits), "premise"
    return "".join(c["text"] for c in commits)


# --------------------------------------------------------------------------- the hand table


TABLE: list[tuple[str, str, str | None]] = [
    # (case, thought, expected)
    ("tail call", PLAN + TYPE_ADA, TYPE_ADA),
    (
        "tail call, trailing spaces and newlines kept",
        PLAN + TYPE_ADA + " \n\n\t ",
        TYPE_ADA + " \n\n\t ",
    ),
    ("the whole thought is the call", TYPE_ADA, TYPE_ADA),
    ("no tag", PLAN + "I will click the email field next.", None),
    ("empty", "", None),
    ("a close but no tag", PLAN + "call:click_at{x:1,y:2}" + CLOSE, None),
    ("a tag with no close", PLAN + f"{TAG}call:click_at{{x:155,y:900}}", None),
    ("a tag with no close, trailing newline", PLAN + f"{TAG}call:click_at{{x:155,y:900}}\n", None),
    (
        "a close then prose (T-0102 spirit)",
        PLAN + CLICK + " Actually, I should scroll.",
        None,
    ),
    ("a close then a single letter", PLAN + CLICK + "\nx", None),
    ("a close then punctuation", PLAN + CLICK + ".", None),
    (
        "two tags, the last well formed at the tail: from the last",
        PLAN + CLICK + "\nThat was wrong; instead:\n" + TYPE_ADA,
        TYPE_ADA,
    ),
    (
        "an unclosed earlier tag, the last well formed: from the last",
        PLAN + f"{TAG}call:click_at{{x:1," + "\n" + TYPE_ADA,
        TYPE_ADA,
    ),
    (
        "a well-formed earlier call, the last tag unclosed: None (not the earlier one)",
        PLAN + CLICK + "\nNext:\n" + f"{TAG}call:type_at{{press_enter:true,",
        None,
    ),
    (
        "a doubled close: recovered whole",
        PLAN + CLICK + CLOSE,
        CLICK + CLOSE,
    ),
    (
        "not validated: garbage between tag and close is returned",
        PLAN + f"{TAG}not a call at all{CLOSE}",
        f"{TAG}not a call at all{CLOSE}",
    ),
    ("the tag alone then a close", f"{TAG}{CLOSE}", f"{TAG}{CLOSE}"),
]


@pytest.mark.parametrize(("case", "thought", "expected"), TABLE, ids=[c for c, _, _ in TABLE])
def test_recover_table(case: str, thought: str, expected: str | None) -> None:
    """A1, by hand: ``thought[k:]`` from the last tag iff its rstrip ends in the close; else None."""
    recover = recover_fn()
    got = recover(thought)
    assert got == expected, f"A1 {case}: recover returned {got!r}, expected {expected!r}"
    if expected is not None:
        assert thought.endswith(got), f"A1 {case}: not a suffix of the thought"  # type: ignore[arg-type]


def test_recover_doubled_close_is_classified_repairable() -> None:
    """A1 + the planner's census: the S5 ``…-210810-…`` ``t1-gen-s2`` tail, doubled ``<tool_call|><tool_call|>``, is
    recovered from its last tag (not repaired, not dropped), and ``live.classify`` calls it ``repairable``, "text after
    call"."""
    recover = recover_fn()
    row = _fixture_row("20261022-210810-", "t1-gen-s2")
    thought = _thought_block(row)
    want = f"{TAG}call:click_at{{x:155,y:900}}{CLOSE}{CLOSE}"
    assert thought.endswith(
        "I will now click the add to cart button for the Cotton tea towel.\n" + want
    ), "premise"
    got = recover(thought)
    assert got == want
    v = live.classify(got)
    # T-0196 moved this verdict: the doubled close is judged as its single-close call (was repairable)
    assert (v.status, v.name, v.arguments, v.reason) == (
        "valid",
        "click_at",
        {"x": 155, "y": 900},
        None,
    )


def test_recover_reopened_channel_mid_thought() -> None:
    """A1 on ``…-203318-…`` ``t1-gen-s3`` (the fixture, verbatim): the thought reopens ``<|channel>`` mid-thought,
    then writes the call; the call is recovered from its tag, and ``classify`` says valid."""
    recover = recover_fn()
    row = _fixture_row("20261022-203318-", "t1-gen-s3")
    thought = _thought_block(row)
    want = f"{TAG}call:type_at{{press_enter:true,text:{Q}Hello from Astra{Q},x:306,y:624}}{CLOSE}"
    assert thought.endswith("<|channel>\nI will now fill in the message." + want), "premise"
    assert recover(thought) == want
    v = live.classify(want)
    assert (v.status, v.name, v.arguments) == (
        "valid",
        "type_at",
        {"press_enter": True, "text": "Hello from Astra", "x": 306, "y": 624},
    )


def test_recover_on_every_fixture_firing_row() -> None:
    """A1 on the 29 logged firings (fixture): the recovered text is the tail the fixture recorded; on the 3 call-less
    open-thought rows it is None."""
    recover = recover_fn()
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines()]
    firing = [r for r in rows if r["class"] == "firing"]
    no_call = [r for r in rows if r["class"] == "open_no_call"]
    assert (len(firing), len(no_call)) == (29, 3), "premise: the fixture"
    for r in firing:
        assert recover(_thought_block(r)) == r["expect"]["tail"], (r["source"], r["request_id"])
    for r in no_call:
        assert recover(_thought_block(r)) is None, (r["source"], r["request_id"])


# --------------------------------------------------------------------------- linear time (adversarial)


_LINEAR_PROBE = r"""
import json, sys, time
from local_astra.toolcall.thought_call import recover
TAG, CLOSE = "<|tool_call>", "<tool_call|>"
N = 200_000
cases = {
    "tags_no_close": (TAG * (N // len(TAG)) + "x", None),
    "tag_close_pairs_then_junk": ((TAG + "call:a{" + CLOSE) * (N // 31) + " junk", None),
    "closes_and_spaces_no_tag": ((CLOSE + " " * 20) * (N // 32) + "z", None),
    "one_tag_long_space_run": (TAG + " " * N + CLOSE + " " * 1000, TAG + " " * N + CLOSE + " " * 1000),
    "many_open_calls_last_closed": (
        (TAG + "call:x{") * (N // 19) + "}" + CLOSE + "\n",
        TAG + "call:x{}" + CLOSE + "\n",
    ),
    "close_then_space_run_then_letter": (TAG + "call:x{}" + (CLOSE + " " * 1000) * (N // 1012) + "q", None),
}
out = {}
for name, (text, want) in cases.items():
    t0 = time.perf_counter()
    got = recover(text)
    out[name] = {"s": time.perf_counter() - t0, "ok": got == want, "len": len(text)}
print(json.dumps(out))
"""

#: Generous: a linear pass over 200 000 characters is milliseconds; a backtracking regex is seconds to minutes.
LINEAR_BOUND_S = 2.0


def test_recover_is_linear() -> None:
    """A1, adversarial (mutant: a backtracking regex such as ``(<\\|tool_call>.*?<tool_call\\|>)\\s*$``). Six
    200 000-character inputs built from ``<|tool_call>``/``<tool_call|>`` fragments and whitespace runs each return
    the right answer in under ``LINEAR_BOUND_S``; run in a child process with a hard timeout so a catastrophic mutant
    fails instead of hanging the suite."""
    recover_fn()  # the AssertionError when the module is absent
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LINEAR_PROBE],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError("A1: recover did not finish 6 x 200 000 chars in 60 s") from None
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    for name, r in out.items():
        assert r["len"] >= 199_000, f"premise: {name} is {r['len']} chars"
        assert r["ok"], f"A1: wrong answer on {name}"
        assert r["s"] < LINEAR_BOUND_S, (
            f"A1: {name} took {r['s']:.2f} s (> {LINEAR_BOUND_S} s): not linear"
        )


# --------------------------------------------------------------------------- never raises; the brute-force oracle

_FRAGMENTS = st.sampled_from(
    [
        TAG,
        CLOSE,
        " ",
        "\n",
        "\t",
        "x",
        "call:click_at{",
        "}",
        "{",
        Q,
        "<|channel>",
        "<channel|>",
        "<|",
        "|>",
        "<tool_call",
    ]
)
_TEXTISH = st.lists(st.one_of(_FRAGMENTS, st.text(max_size=6)), max_size=40).map("".join)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(thought=st.one_of(st.text(), _TEXTISH))
def test_never_raises(thought: str) -> None:
    """A1: over arbitrary text (and text built from tag fragments), ``recover`` never raises, returns ``str`` or
    ``None``, and agrees with the brute-force oracle (every index tried)."""
    recover = recover_fn()
    got = recover(thought)
    assert got is None or isinstance(got, str)
    assert got == _oracle(thought), f"A1 on {thought!r}: {got!r} != oracle {_oracle(thought)!r}"


# --------------------------------------------------------------------------- stdlib-only


def test_module_is_stdlib_only() -> None:
    """A1: ``thought_call.py`` imports the standard library only (no ``local_astra``, no third-party module); it
    does not parse or validate (judging is ``live.classify``'s, A6)."""
    recover_fn()
    tree = ast.parse(MODULE_FILE.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "A1: a relative import is not stdlib"
            names.add((node.module or "").split(".")[0])
    foreign = sorted(n for n in names if n != "__future__" and n not in sys.stdlib_module_names)
    assert foreign == [], f"A1: thought_call imports non-stdlib modules {foreign}"


# --------------------------------------------------------------------------- A3: byte identity over the closed range


def test_a3_frozen_paths_over_closed_range() -> None:
    """A3 (green at BASE): ``toolcall/live.py``, ``native.py``, ``router.py``, ``conductor/llm_client.py``,
    ``contracts/**`` and ``tests/fakes/**`` are byte-identical over ``BASE..TIP_IMPL`` (two literals, never the
    working tree), with no file added or removed. Power: the same predicate sees T-0102's ``live.py`` change."""
    gr.require_commits(BASE, TIP_IMPL)
    assert gr.is_ancestor(BASE, TIP_IMPL), f"{BASE[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE, "--", *FROZEN_A3).stdout.split()
    for p in FROZEN_A3:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE, TIP_IMPL, *FROZEN_A3)
    assert changed == [], f"T-0135 A3: frozen paths differ in {BASE[:7]}..{TIP_IMPL[:7]}: {changed}"
    gr.require_commits(POWER_PARENT, POWER)
    assert "local_astra/toolcall/live.py" in gr.changed(POWER_PARENT, POWER, *FROZEN_A3), "power"
