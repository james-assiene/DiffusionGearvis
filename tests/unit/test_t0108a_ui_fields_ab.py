"""T-0108a: the required-fields prompt A/B -- a fourth arm table (A0 vs K1, K2) on the T-0098 harness, 40 repeats on
the four Q12 positives and 5 elsewhere (1320 requests), the complete-card reading through the real ``UiGate`` and the
pre-registered rule H-T108a (unit; CPU; no engine).

Rows that need the T-0108a table are red on the named assertion in ``harness_108a()``
(``TABLES["T-0108a"]`` absent) or on an assertion of their own, never on an import error: the harness is loaded lazily,
by path, through T-0098's loader. The guard on the earlier tables (criterion 2) is unmarked and green before and after
the implementation.

**The assumed surface** (``scripts/gpu/t0098_native_prompt_ab.py``; the task names the behaviour, these names are the
test-author's, stated so the implementer can honour them or say why not; they extend T-0103's):

- ``TABLES["T-0108a"]`` is an ``ArmTable`` like ``TABLES["T-0103"]``: ``arms == ("A0", "K1", "K2")``,
  ``candidates == ("K1", "K2")``, ``rule == "H-T108a"``, ``a0_sha256``, ``texts`` (the arm-text builder, a callable
  of the tools) and ``repeats`` (case id -> 40 on the four positives, 5 on the other 56). Among the table's other
  values are ``True`` under a field whose name contains ``ui``, the string ``"t0108a"`` (request prefix) and the
  string ``"t0108a-ui-fields-ab"`` (out stem); the field names are free.
- The anchor and the 2000-code-point checks live in the builder: ``arm_texts(tools, table="T-0108a")`` raises
  ``ValueError`` naming the arm (as T-0098c's C5/C4). The K1/K2 pins and the repeats map are checked by ``main`` on
  what the table gives it (as T-0103's C3 pin), so a replaced ``texts`` or ``repeats`` is refused there.
- ``schedule("T-0108a")``, ``arm_order(case_index, repeat, "T-0108a")``, ``load_cases(table="T-0108a")``,
  ``arm_prompts(case, kit, table="T-0108a")`` as for T-0103.
- ``ui_reading(commits, request_id=...)`` additionally returns ``last_ok`` (bool), ``pruned_last`` (a collection of
  component ids, compared as a set) and ``complete`` (bool) -- criterion 4.
- ``build_report(run_dir)`` on a run whose manifest says ``table: "T-0108a"`` returns T-0103's per-arm keys
  (``q12_positive_shown``, ``q12_per_case`` = shown per case, ``placement_failures_q12_positive``,
  ``q12_tool_calls``, ``surfaces_no_belong``, ``surfaces_tool``, ``false_tools``, ``thought_open``, ``errored``,
  ``tools``...) plus ``q12_positive_complete`` and ``q12_positive_last_ok`` (k/160) and ``q12_per_case_complete`` /
  ``q12_per_case_last_ok`` (case id -> k/40); ``candidates[K][c]`` (bool) for K in K1, K2 and c in
  Q1 Q2 U1 U2 P3 P4 P5 P6 P7 P9 P10 P11; ``readings == {"R3": bool, "R4": bool}``; ``void_arms``; ``complete``;
  ``winner`` (``"K1"``, ``"K2"``, or ``None`` / ``"no change"``); ``outcome``, a string of one of the forms
  ``"K<i> selected..."``, ``"no change: ..."`` (every failing condition of every K named by its id, Q1 / U2 / P11
  ...) and ``"inconclusive: <reason>"``.

**Readings this module makes where the task is open** (named in the test-author report): the inconclusive reasons
are pinned only as containing ``incomplete``, ``void`` and the arm, ``R3`` or ``R4``; "a 600-line run" (criterion 6's
adversarial row, carried from T-0103) is read as two arms of five repeats (A0 and K1 only, 60 x 5 x 2), and the
three-arm run with the positives at 5 (900 lines) is pinned beside it.

**The baseline T-0108a run** (every rule row edits it; all numbers by hand): 60 cases, arms A0 K1 K2, repeats 40 on
the four positives and 5 elsewhere = 1320 lines, 440 per arm, in the schedule of *Design*.

- Tool cases: every arm speaks and calls (P1/P2 are dropped per D16, so nothing is silent). No-tool, plain and Q12
  negative cases: an answer, no call, no surface.
- Q12 positives, per case of 40 (repeats in this order): A0 20 complete (``RUN_COMPLETE``, one Text), 16 pruned
  (``RUN_PRUNED``, a Column and an unrepairable ``TextField`` (T-0108b 9(ii)): shown, the last call OK, the TextField pruned), 4
  speech only -> complete 80/160,
  last-OK 144, shown 144 (R3 holds, 144 >= 112; R4 holds, 80 <= 96; no case protected for Q2, 20 < 24). K1 36
  complete, 4 speech only -> 144/160. **K2 copies A0's positives**, so K2 fails exactly Q1 (80 < 80 + 24) and the
  outcome turns on K1 alone: "K1 selected".

Thresholds are typed from docs/tasks/T-0108a.md criterion 5, never imported: Q1 K >= A0 + 24 (of 160); Q2 protected
iff A0 complete >= 24/40, then K >= 24/40; U1 / U2 K >= A0 - 16 with the same per-case clause on last-OK / shown; P9
<= A0 + 1 of 80 and <= A0 + 2 of 200; P10 <= A0 + 8; P11 <= A0 + 10 of 175; P6 = 0; R3 A0 shown >= 112/160; R4 A0
complete <= 96/160; void >= 9 of 440.

Regime: CPU, pure, hand-built data, the real ``UiGate`` and vendored A2UI v0.9.1 schema at the working tree; no
measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.conductor.ui_gate import UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE, UI_INSTRUCTIONS
from local_astra.toolcall.detok import pinned_tokenizer_path
from local_astra.ui import A2UIValidator
from tests.integration.test_t0098_harness_smoke import no_sockets
from tests.unit._harness_reading_memo import memoize_ui_reading
from tests.unit._pre_t0155_persona import (
    PRE_T0155_A0_SHA256,
    pre_t0155_fixture,
    run_main_against_live_persona,
)
from tests.unit.test_t0098_prompt_ab import (
    A0_SHA256,
    HARNESS,
    NONE,
    REPO,
    SILENT,
    SPEAK,
    baseline,
    commit,
    sha256,
    tool_kit,
    write_run,
)
from tests.unit.test_t0098c_ui_prompt_ab import (
    ERROR_C,
    OPEN_SURFACE,
    PROSE,
    Q12_POSITIVE,
    Q12_TOOL,
    SHOWN,
    SHOWN_FAIL,
    WITH_SURFACE,
    case_rows_c,
    commits_for_c,
    harness_c,
    patch_u,
    pieces,
    write_run_c,
)
from tests.unit.test_t0103_c3_confirm import (
    NEGATIVE_SLOTS,
    NO_BELONG_SLOTS,
    PLAIN_SLOTS,
    T0098_REPORT_KEYS,
    T0098C_REPORT_KEYS,
    _fake_git,
    _field,
    _find,
    _get,
    _protected_flags,
    _t0098c_fixture_plan,
    baseline_103,
    write_run_103,
)
from tests.unit.test_t0103_c3_confirm import expected_schedule as expected_schedule_103

#: The commit T-0108a branched from (task header: ``main`` at ``ca61f08``). Criterion 2: the earlier tables' output is
#: exactly as here. A literal, never a moving ref (.claude/rules/tests.md).
BASE_SHA = "ca61f086ba143330fc6e2910203ec2be5d88077b"

TABLE = "T-0108a"

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)
RULE = "H-T108a"
ARMS = ("A0", "K1", "K2")
CANDIDATES = ("K1", "K2")
CONDS = ("Q1", "Q2", "U1", "U2", "P3", "P4", "P5", "P6", "P7", "P9", "P10", "P11")
COND_RE = re.compile(r"\b(?:Q[12]|U[12]|P\d+)\b")

# --- T-0108a *Arms*, typed out -----------------------------------------------------------------------------------
E1_ANCHOR = (
    "Copy that catalogId exactly, use the same surfaceId in every message, give the tree a component\n"
    'with id "root", and define every id you reference in children in the same fence.'
)
E1_REPLACEMENT = (
    'Copy that catalogId exactly, keep one surfaceId, give one component id "root" and define every\n'
    "id you reference in the same fence."
)
F1 = (
    'Required: TextField label; DateTimeInput value ""; ChoicePicker options [{"label":"A","value":"a"}]\n'
    'and value []; Button child (a Text id) and action {"event":{"name":"go"}}.\n'
)
F2 = "Text variant: h1-h5, caption or body; children are ids, never objects.\n"
K1_ANCHOR = "flashcard is a Card.\n"
K2_ANCHOR = (
    "There is no table, chart, maths or flashcard component: a table is a Column of Rows of Text, a\n"
    "flashcard is a Card.\n"
)
K2_LEAD = "A table is a Column of Rows of Text.\n"
U_CHARS = 1870
U_SHA256 = "4360995b013627819f20de7044bcbae75edfef1f9724356eba5f94cc2b19495b"
K1_U_CHARS = 1999
K1_U_SHA256 = "50f7d7424777260ad52e8bcdf36cce683e53eaf8c68fb0b2c08299120245b60e"
K2_U_CHARS = 1991
K2_U_SHA256 = "71765248036810fa08b556494e74008b12413bf25d7b310a5bcc5092a54fe8ea"
A0_CHARS = 2744
K1_CHARS = 2873
K1_SHA256 = "22a2ba7e9a022268fb41f847fa3cfad1e0a123f5c323cb120c63d7526fac0702"
K2_CHARS = 2865
K2_SHA256 = "23a34c3656547eebaf2f89c65cc6a0a04367f8dcd93924d3bb4b8be79ced687c"
U_MAX = 2000

# --- *Design* ----------------------------------------------------------------------------------------------------
POSITIVE_REPEATS = 40
OTHER_REPEATS = 5
N_REQUESTS = 1320
PER_ARM = 440

# --- criterion 5, typed out --------------------------------------------------------------------------------------
Q1_MARGIN = 24  # 3/20 x 160
PROTECT_MIN = 24  # of 40
U_MARGIN = 16  # 1/10 x 160
P10_MARGIN = 8
P11_MARGIN = 10
R3_MIN = 112
R4_MAX = 96
VOID_MIN = 9  # > 2 % of 440

Plan = dict[tuple[str, str, int], str]


# =====================================================================================================
# Loading
# =====================================================================================================


def harness_108a() -> ModuleType:
    """The harness (T-0098's loader), with the T-0108a table asserted. Red here until criterion 1."""
    h = harness_c()
    assert TABLE in h.TABLES, (
        "T-0108a criterion 1: the harness has no TABLES['T-0108a'] (see this module's docstring)"
    )
    assert callable(getattr(h, "schedule", None)), "T-0103's schedule(table) is gone"
    return h


@functools.cache
def _old_source() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE_SHA}:scripts/gpu/t0098_native_prompt_ab.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def old_harness() -> ModuleType:
    """The harness exactly as at ``BASE_SHA``, compiled from the git blob with the real file's ``__file__``: the frozen
    oracle for criterion 2. Skips when the commit is absent."""
    src = _old_source()
    if src is None:
        pytest.skip(f"{BASE_SHA[:7]} is not in this clone; the frozen oracle is unavailable")
    name = "_t0098_harness_at_ca61f08"
    module = ModuleType(name)
    module.__file__ = str(HARNESS)
    sys.modules[name] = module
    try:
        exec(compile(src, f"{BASE_SHA[:7]}:{HARNESS.name}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def _table_values(table: Any) -> list[tuple[str, Any]]:
    if isinstance(table, Mapping):
        return [(str(k), v) for k, v in table.items()]
    return [(f.name, getattr(table, f.name)) for f in dataclasses.fields(table)]


def expected_repeats() -> dict[str, int]:
    return {
        r["id"]: (POSITIVE_REPEATS if r["id"] in Q12_POSITIVE else OTHER_REPEATS)
        for r in case_rows_c()
    }


def expected_schedule(
    positive_repeats: int = POSITIVE_REPEATS,
) -> list[tuple[int, str, int, int, str]]:
    """(case_index, case_id, repeat, slot, arm) in request order, by hand from *Design*: repeats outermost; within a
    repeat the cases scheduled for it in index order; per (case, repeat) the three arms from offset
    (case_index + repeat) mod 3 of A0 K1 K2."""
    rows = case_rows_c()
    out: list[tuple[int, str, int, int, str]] = []
    for r in range(max(positive_repeats, OTHER_REPEATS)):
        for i, row in enumerate(rows):
            n = positive_repeats if row["id"] in Q12_POSITIVE else OTHER_REPEATS
            if r >= n:
                continue
            off = (i + r) % 3
            for slot in range(3):
                out.append((i, row["id"], r, slot, ARMS[(off + slot) % 3]))
    return out


# =====================================================================================================
# Criterion 1: the arm texts, by hand
# =====================================================================================================


def u_k1(u: str = UI_INSTRUCTIONS) -> str:
    return u.replace(E1_ANCHOR, E1_REPLACEMENT).replace(K1_ANCHOR, K1_ANCHOR + F1)


def u_k2(u: str = UI_INSTRUCTIONS) -> str:
    return u.replace(E1_ANCHOR, E1_REPLACEMENT).replace(K2_ANCHOR, K2_LEAD + F1 + F2)


def texts_by_hand(u: str = UI_INSTRUCTIONS) -> dict[str, str]:
    p, ct, ss, _, n = pieces()
    return {
        "A0": "\n".join([p, ct, ss, u, n]),
        "K1": "\n".join([p, ct, ss, u_k1(u), n]),
        "K2": "\n".join([p, ct, ss, u_k2(u), n]),
    }


#: F1's snippets as components (the literal JSON of F1, given ids), and the four as-written errors of the Goal.
F1_SNIPPETS = [
    {"id": "t", "component": "TextField", "label": "Name"},
    {"id": "d", "component": "DateTimeInput", "value": ""},
    {
        "id": "c",
        "component": "ChoicePicker",
        "options": [{"label": "A", "value": "a"}],
        "value": [],
    },
    {"id": "bt", "component": "Text", "text": "Go"},
    {"id": "b", "component": "Button", "child": "bt", "action": {"event": {"name": "go"}}},
]
GOAL_ERRORS = [
    {"id": "t", "component": "TextField", "placeholder": "Name"},
    {"id": "b", "component": "Button", "text": "Go"},
    {"id": "d", "component": "DateTimeInput"},
    {"id": "c", "component": "ChoicePicker", "options": ["Economy"], "value": []},
]


def test_t0108a_arm_texts_by_hand() -> None:
    """Criterion 1 and *Arms*: K1 and K2 built here from the literals (E1, then K1's or K2's anchor replaced once),
    compared with the task's char counts and shas (U' 1999 / 1991, texts 2873 / 2865, A0 still ``3605c012...`` at
    2744); each U' <= 2000; each K equals A0 outside U (same P, CT, SS before it and N after it); every F1 snippet has
    zero ``component_errors`` and each of the Goal's four as-written forms has at least one. The harness's
    ``arm_texts(tools, table="T-0108a")`` equals the hand build; the table holds the three arms, two candidates, rule
    H-T108a, A0's pin, the 40/5 repeats map, the UI channel on, prefix ``t0108a`` and stem ``t0108a-ui-fields-ab``; the
    60 cases are T-0098c's, and every K body differs from A0's only in ``messages[0]``."""
    # premises: the literals, by hand
    _, _, _, u, _ = pieces()
    assert len(u) == U_CHARS and sha256(u) == U_SHA256
    for anchor in (E1_ANCHOR, K1_ANCHOR, K2_ANCHOR):
        assert u.count(anchor) == 1, anchor
    assert u_k1().count(F1) == 1 and u_k2().count(F2) == 1 and K2_ANCHOR not in u_k2()
    assert len(u_k1()) == K1_U_CHARS <= U_MAX and sha256(u_k1()) == K1_U_SHA256
    assert len(u_k2()) == K2_U_CHARS <= U_MAX and sha256(u_k2()) == K2_U_SHA256
    want = texts_by_hand()
    assert (len(want["A0"]), sha256(want["A0"])) == (A0_CHARS, A0_SHA256)
    assert (len(want["K1"]), sha256(want["K1"])) == (K1_CHARS, K1_SHA256)
    assert (len(want["K2"]), sha256(want["K2"])) == (K2_CHARS, K2_SHA256)
    v = A2UIValidator()
    for c in F1_SNIPPETS:
        assert v.component_errors(c) == [], c
    for c in GOAL_ERRORS:
        assert v.component_errors(c), f"premise: {c} is invalid as written"
    assert '[{"label":"A","value":"a"}]' in F1 and '{"event":{"name":"go"}}' in F1

    h = harness_108a()
    kit = tool_kit()
    texts = dict(h.arm_texts(kit.tools, table=TABLE))
    assert texts == want, "criterion 1: the arm texts are not the hand-built ones"
    p, ct, ss, _, n = pieces()
    head, tail = "\n".join([p, ct, ss]) + "\n", "\n" + n
    for k, uk in (("K1", u_k1()), ("K2", u_k2())):
        assert texts[k].startswith(head) and texts[k].endswith(tail)
        assert texts[k][len(head) : len(texts[k]) - len(tail)] == uk, f"{k} differs outside U"
    t = h.TABLES[TABLE]
    assert tuple(_get(t, "arms")) == ARMS
    assert tuple(_get(t, "candidates")) == CANDIDATES
    assert _get(t, "rule") == RULE and _get(t, "a0_sha256") == A0_SHA256
    assert dict(_get(t, "repeats")) == expected_repeats(), "criterion 1: the 40/5 repeats map"
    values = _table_values(t)
    assert any("ui" in name.lower() and v is True for name, v in values), "criterion 1: ui_channel"
    strings = {v for _, v in values if isinstance(v, str)}
    assert {"t0108a", "t0108a-ui-fields-ab"} <= strings, f"criterion 1: prefix / stem: {strings}"

    old_cases = [c.case_id for c in old_harness().load_cases(table="T-0098c")]
    cases = h.load_cases(table=TABLE)
    assert [c.case_id for c in cases] == old_cases == [r["id"] for r in case_rows_c()]
    client = VllmClient()
    for case in cases[::7] + cases[53:]:
        prompts = h.arm_prompts(case, kit, table=TABLE)
        assert set(prompts) == set(ARMS), case.case_id
        for arm in ARMS:
            assert prompts[arm].messages[0].text() == want[arm], (case.case_id, arm)
        bodies = {a: client.request_body(pr, "fidelity") for a, pr in prompts.items()}
        assert h.check_fidelity(bodies) == [], case.case_id


# =====================================================================================================
# *Design*: the schedule
# =====================================================================================================

#: A0 first per positive case over r = 0..39, by hand: offset (i + r) mod 3 == 0 -- i = 53: r = 1, 4, ..., 37 (13);
#: i = 54: r = 0, 3, ..., 39 (14); i = 55: r = 2, ..., 38 (13); i = 56: r = 1, ..., 37 (13).
A0_FIRST = {
    "flight_booking": 13,
    "flight_booking_with_memory": 14,
    "primes_table": 13,
    "omelette_steps": 13,
}


def test_t0108a_schedule() -> None:
    """*Design*, by hand: 1320 requests = 4 x 40 x 3 + 56 x 5 x 3 with unique ids ``t0108a-r{repeat}-{case_id}-{arm}``;
    40 per arm on indices 53-56 and 5 elsewhere; r >= 5 schedules only 53-56; repeats outermost, cases in index order;
    arms from offset (i + r) mod 3 (A0 first 13, 14, 13, 13 times of 40 on the positives). A mod-2 rotation or
    A0-always-first fails the slot check; 5 repeats fails the length."""
    h = harness_108a()
    items = list(h.schedule(TABLE))
    got = [
        (
            int(_field(x, "case_index")),
            str(_field(x, "case_id")),
            int(_field(x, "repeat")),
            int(_field(x, "order")),
            str(_field(x, "arm")),
        )
        for x in items
    ]
    want = expected_schedule()
    assert len(want) == N_REQUESTS == 4 * 40 * 3 + 56 * 5 * 3, "premise"
    assert len(got) == N_REQUESTS, f"{len(got)} requests"
    assert got == want, "the schedule is not *Design*'s"
    ids = [str(_field(x, "request_id")) for x in items]
    assert len(set(ids)) == N_REQUESTS
    assert ids == [f"t0108a-r{r}-{cid}-{arm}" for _, cid, r, _, arm in want]
    per = Counter((cid, arm) for _, cid, _, _, arm in got)
    for cid, n in expected_repeats().items():
        assert per[(cid, "A0")] == per[(cid, "K1")] == per[(cid, "K2")] == n, cid
    assert {i for i, _, r, _, _ in got if r >= 5} == {53, 54, 55, 56}
    first = Counter(cid for _, cid, _, slot, arm in got if slot == 0 and arm == "A0")
    assert {cid: first[cid] for cid in Q12_POSITIVE} == A0_FIRST
    for (i, r), order in {
        (53, 0): ("K2", "A0", "K1"),  # 53 mod 3 = 2
        (54, 0): ("A0", "K1", "K2"),  # 54 mod 3 = 0
        (0, 1): ("K1", "K2", "A0"),  # 1 mod 3 = 1
        (59, 4): ("A0", "K1", "K2"),  # 63 mod 3 = 0
    }.items():
        assert tuple(h.arm_order(i, r, TABLE)) == order, (i, r)


# =====================================================================================================
# Criterion 2: the earlier tables are untouched (guard; green today and through the implementation)
# =====================================================================================================


def test_prior_tables_untouched(tmp_path: Path) -> None:
    """Criterion 2: T-0098 schedules 1590 in mod 6, T-0098c 2100 in mod 7, T-0103 880 in mod 2 (hand-built rotation
    checks, and every schedule item equal to the harness's at ca61f08); ``build_report`` and ``render_markdown`` on a
    T-0098 baseline, a T-0098c fixture and the T-0103 baseline equal the ca61f08 harness's output byte for byte, with
    the ca61f08 key sets and the T-0103 outcome "C3 confirmed"."""
    h = harness_c()
    old = old_harness()
    assert tuple(h.arm_order(52, 4)) == ("D1", "D2", "C2", "C1", "A0", "A1")  # (52 + 4) mod 6 = 2
    assert tuple(h.arm_order(53, 0, "T-0098c")) == ("C3", "C5", "C4", "A0", "D1", "D3", "D4")
    assert tuple(h.arm_order(53, 0, "T-0103")) == ("C3", "A0")  # 53 mod 2 = 1
    assert tuple(h.arm_order(54, 1, "T-0103")) == ("C3", "A0")  # 55 mod 2 = 1
    assert tuple(h.arm_order(54, 0, "T-0103")) == ("A0", "C3")

    def rows(mod: Any, table: str) -> list[tuple[Any, ...]]:
        return [
            tuple(
                _field(x, k)
                for k in ("case_index", "case_id", "repeat", "order", "arm", "request_id")
            )
            for x in mod.schedule(table)
        ]

    for table, n in (("T-0098", 1590), ("T-0098c", 2100), ("T-0103", 880)):
        now = rows(h, table)
        assert len(now) == n, (table, len(now))
        assert now == rows(old, table), f"criterion 2: {table}'s schedule moved"
    assert [x[:5] for x in rows(h, "T-0103")] == expected_schedule_103()

    memoize_ui_reading(h)  # T-0110 part A: the report building below, on these two objects only
    memoize_ui_reading(old)
    run98 = write_run(tmp_path / "t0098", baseline())
    rep98 = h.build_report(run98)
    assert set(rep98) == T0098_REPORT_KEYS and rep98["trials"] == 1590 and rep98["winner"] == "C2"
    assert json.dumps(rep98, sort_keys=True) == json.dumps(old.build_report(run98), sort_keys=True)
    assert h.render_markdown(rep98) == old.render_markdown(rep98)

    run98c = write_run_c(tmp_path / "t0098c", _t0098c_fixture_plan())
    rep98c = h.build_report(run98c)
    assert set(rep98c) == T0098C_REPORT_KEYS and rep98c["trials"] == 2100
    assert rep98c["winner"] == "C5"
    assert json.dumps(rep98c, sort_keys=True) == json.dumps(
        old.build_report(run98c), sort_keys=True
    )
    assert h.render_markdown(rep98c) == old.render_markdown(rep98c)

    run103 = write_run_103(tmp_path / "t0103", baseline_103())
    rep103 = h.build_report(run103)
    assert rep103["trials"] == 880 and str(rep103["outcome"]).startswith("C3 confirmed")
    assert "q12_positive_complete" not in rep103["arms"]["A0"], "a T-0108a key leaked into T-0103"
    assert json.dumps(rep103, sort_keys=True) == json.dumps(
        old.build_report(run103), sort_keys=True
    )
    assert h.render_markdown(rep103) == old.render_markdown(rep103)
    # reviewer (b): key order too -- the report as written, not only as a set of keys
    assert json.dumps(rep103) == json.dumps(old.build_report(run103)), (
        "criterion 2: the T-0103 report's key order moved"
    )


# =====================================================================================================
# Criterion 3: the refusals (adversarial)
# =====================================================================================================

#: (edited U, the arm the builder must name)
ANCHOR_EDITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "e1-twice": (UI_INSTRUCTIONS + "\n" + E1_ANCHOR, ("K1", "K2")),
    "e1-zero": (UI_INSTRUCTIONS.replace(E1_ANCHOR, E1_REPLACEMENT), ("K1", "K2")),
    "k1-anchor-twice": (UI_INSTRUCTIONS + "\nA flashcard is a Card.\n", ("K1",)),
    "k2-anchor-zero": (
        UI_INSTRUCTIONS.replace(
            "a table is a Column of Rows of Text, a\nflashcard",
            "a table is a Column of Rows, a\nflashcard",
        ),
        ("K2",),
    ),
}


@pytest.mark.parametrize("why", list(ANCHOR_EDITS))
def test_t0108a_anchor_refusals(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    """Criterion 3 (adversarial): with U changed so that E1's anchor occurs twice or not at all, K1's anchor twice
    (K2's still once) or K2's not at all (K1's still once), ``arm_texts(tools, table="T-0108a")`` raises ``ValueError``
    naming the arm whose anchor broke -- never a double edit or a silent no-op that sends A0's U as a candidate.
    Premise: the counts. Guard the other way: under the same patch T-0098's table still builds."""
    new_u, arms = ANCHOR_EDITS[why]
    counts = {a: new_u.count(a) for a in (E1_ANCHOR, K1_ANCHOR, K2_ANCHOR)}
    want_counts = {
        "e1-twice": (2, 1, 1),
        "e1-zero": (0, 1, 1),
        "k1-anchor-twice": (1, 2, 1),
        "k2-anchor-zero": (1, 1, 0),
    }[why]
    assert tuple(counts.values()) == want_counts, f"premise: {counts}"
    h = harness_108a()
    kit = tool_kit()
    patch_u(monkeypatch, h, new_u)
    assert set(h.arm_texts(kit.tools)) == {"A0", "A1", "D1", "D2", "C2", "C1"}
    with pytest.raises(ValueError) as caught:
        h.arm_texts(kit.tools, table=TABLE)
    assert any(a in str(caught.value) for a in arms), f"{why}: {caught.value}"


@pytest.mark.parametrize(
    ("pad", "refused"), [(1, False), (2, True)], ids=["k1-2000-ok", "k1-2001-refused"]
)
def test_t0108a_u_length_boundary(monkeypatch: pytest.MonkeyPatch, pad: int, refused: bool) -> None:
    """Criterion 3, both directions: U with ``pad`` trailing spaces makes K1's U' 1999 + pad code points. 2000 builds
    (K1 is the hand build over the padded U); 2001 is refused, naming K1."""
    new_u = UI_INSTRUCTIONS + " " * pad
    assert len(u_k1(new_u)) == K1_U_CHARS + pad
    h = harness_108a()
    kit = tool_kit()
    patch_u(monkeypatch, h, new_u)
    if refused:
        with pytest.raises(ValueError) as caught:
            h.arm_texts(kit.tools, table=TABLE)
        assert "K1" in str(caught.value)
    else:
        assert h.arm_texts(kit.tools, table=TABLE)["K1"] == texts_by_hand(new_u)["K1"]


def _replace_text(
    h: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    variant: Callable[[dict[str, str]], str],
) -> None:
    t = h.TABLES[TABLE]
    orig = t.texts

    def texts(tools: Any) -> dict[str, str]:
        out = dict(orig(tools))
        out[arm] = variant(out)
        return out

    monkeypatch.setitem(h.TABLES, TABLE, dataclasses.replace(t, texts=texts))


def _replace_repeats(
    h: ModuleType, monkeypatch: pytest.MonkeyPatch, edit: Mapping[str, int]
) -> None:
    t = h.TABLES[TABLE]
    monkeypatch.setitem(
        h.TABLES, TABLE, dataclasses.replace(t, repeats={**dict(t.repeats), **edit})
    )


REFUSALS: dict[str, tuple[str, ...]] = {
    "a0-drift-ui-instructions": ("3605c012", "A0"),
    "k1-sha-drift-trailing-space": ("K1", "22a2ba7e"),
    "k2-is-k1": ("K2", "23a34c36"),
    "k1-is-a0": ("K1", "22a2ba7e"),
    "anchor-e1-twice": ("anchor", "K1", "K2"),
    "anchor-k1-zero": ("anchor", "K1"),
    "u-prime-2001": ("2000", "2001", "K1"),
    "repeats-39-on-one-positive": ("repeat",),
    "repeats-5-on-the-positives": ("repeat",),
    "repeats-6-on-one-other-case": ("repeat",),
    "split-test": ("split",),
}


@pytest.mark.parametrize("why", [*REFUSALS, "control-unbroken"])
def test_t0108a_refusals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, why: str
) -> None:
    """Criterion 3 (adversarial): under ``--table T-0108a`` each row breaks one condition and ``main`` exits 2 with no
    socket, nothing written, and a message naming the cause -- A0 drift through ``UI_INSTRUCTIONS``; K1 plus one space;
    K2 replaced by K1's text; K1 replaced by A0's; E1's anchor twice; K1's anchor absent; a 2001-code-point K1 U'; a
    repeats map with 39 on one positive, 5 on all four (T-0098c's design), 6 on one other case; ``--split test``. The
    git reads see a clean tree (a double) and the kit is the real one. Guard the other way: the same invocation
    unbroken passes every refusal and reaches the engine (a socket attempt, refused here)."""
    h = harness_108a()
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    monkeypatch.setattr(h._T93, "_git", _fake_git(h._T93._git))
    out = tmp_path / "t0108a-out"
    argv = [
        "--base-url",
        "http://127.0.0.1:9/v1",
        "--out-dir",
        str(out),
        "--split",
        "dev",
        "--table",
        TABLE,
    ]
    if why == "a0-drift-ui-instructions":
        patch_u(monkeypatch, h, UI_INSTRUCTIONS + " Be brief.")
    elif why == "k1-sha-drift-trailing-space":
        _replace_text(h, monkeypatch, "K1", lambda t: t["K1"] + " ")
    elif why == "k2-is-k1":
        _replace_text(h, monkeypatch, "K2", lambda t: t["K1"])
    elif why == "k1-is-a0":
        _replace_text(h, monkeypatch, "K1", lambda t: t["A0"])
    elif why == "anchor-e1-twice":
        patch_u(monkeypatch, h, ANCHOR_EDITS["e1-twice"][0])
    elif why == "anchor-k1-zero":
        patch_u(monkeypatch, h, UI_INSTRUCTIONS.replace(K1_ANCHOR, "flashcard is a Card too.\n"))
    elif why == "u-prime-2001":
        patch_u(monkeypatch, h, UI_INSTRUCTIONS + "  ")
    elif why == "repeats-39-on-one-positive":
        _replace_repeats(h, monkeypatch, {"primes_table": 39})
    elif why == "repeats-5-on-the-positives":
        _replace_repeats(h, monkeypatch, dict.fromkeys(Q12_POSITIVE, 5))
    elif why == "repeats-6-on-one-other-case":
        _replace_repeats(h, monkeypatch, {"plain-03": 6})
    elif why == "split-test":
        argv[5] = "test"
    reached_engine = False
    with no_sockets() as attempts:
        try:
            rc = h.main(argv)
        except AssertionError:
            rc, reached_engine = None, True
    said = capsys.readouterr()
    text = said.out + said.err
    if why == "control-unbroken":
        assert rc != 2 and (reached_engine or attempts), (
            f"the unbroken run was refused: rc={rc}: {text[-400:]!r}"
        )
        return
    assert rc == 2, f"criterion 3 ({why}): exit {rc}: {text[-400:]!r}"
    assert attempts == [] and not reached_engine, f"criterion 3 ({why}): sockets: {attempts}"
    assert not out.exists() or not any(out.iterdir()), (
        f"criterion 3 ({why}): the refusal wrote files"
    )
    assert any(tok in text for tok in REFUSALS[why]), f"criterion 3 ({why}): {text[-400:]!r}"


# =====================================================================================================
# Criterion 4: the complete-card reading, through the real UiGate
# =====================================================================================================

CREATE = UI_EXAMPLE.split("\n")[0]
UPDATE = '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
TITLE = '{"id":"h","component":"Text","text":"Your trip","variant":"h3"}'
#: A Text and an unrepairable TextField (``variant: "huge"``; T-0108b criterion 9(ii) replaced the placeholder-only
#: one, which T-0108b repairs): valid JSON, the TextField schema-invalid.
PLACEHOLDER = (
    CREATE
    + "\n"
    + UPDATE
    + '{"id":"root","component":"Column","children":["h","f"]},'
    + TITLE
    + ","
    '{"id":"f","component":"TextField","variant":"huge","label":"From"}]}}'
)
#: The same TextField unreferenced, in a canvas missing its last ``}``: the fixer path's salvage drops it, so it is in
#: ``_repair(raw).dropped`` and absent from ``_repair(raw).messages`` (premise checked below).
SALVAGE_DROP = (
    CREATE + "\n" + UPDATE + '{"id":"root","component":"Column","children":["h"]},' + TITLE + ","
    '{"id":"f","component":"TextField","variant":"huge","label":"From"}]}'
)
#: A reference to a component that has not arrived (``later``): pruned as a forward reference, every component valid.
FORWARD_REF = (
    CREATE
    + "\n"
    + UPDATE
    + '{"id":"root","component":"Column","children":["h","later"]},'
    + TITLE
    + "]}}"
)


def _oracle(raws: list[str], turn_id: str) -> tuple[bool, set[str]]:
    """Criterion 4's definition applied here, independently: a fresh gate, every canvas in order with
    ``final=False``; the last call's surface, and the pruned set read on the last canvas through ``_repair``."""
    g = UiGate()
    last = None
    for i, raw in enumerate(raws):
        last = g.gate(raw, turn_id=turn_id, canvas_idx=i, final=False)
    if last is None:
        return False, set()
    rep = g._repair(raws[-1])
    invalid = set(
        g.validator.invalid_component_ids(list(g.validator.iter_components(rep.messages)))
    )
    return last.surface is not None, invalid | set(rep.dropped)


def _ui(*raws: str, open_: bool = False) -> list[dict[str, Any]]:
    return [
        commit("thought", "OK."),
        commit("answer", "Here it is."),
        *(commit("ui", raw, idx=i, open_=open_) for i, raw in enumerate(raws)),
    ]


#: (commits, last_ok, pruned_last, complete) -- all by hand
READING_ROWS: dict[str, tuple[list[dict[str, Any]], bool, set[str], bool]] = {
    "ui-example-complete": (_ui(UI_EXAMPLE), True, set(), True),
    "placeholder-textfield-last-ok-not-complete": (_ui(PLACEHOLDER), True, {"f"}, False),
    "adv-salvage-only-drop-not-complete": (_ui(SALVAGE_DROP), True, {"f"}, False),
    "adv-ok-then-pruned-last-not-complete": (_ui(UI_EXAMPLE, PLACEHOLDER), True, {"f"}, False),
    "pruned-then-clean-last-complete": (_ui(PLACEHOLDER, UI_EXAMPLE), True, set(), True),
    "ok-then-failing-last-not-complete": (_ui(UI_EXAMPLE, PROSE), False, set(), False),
    "forward-ref-only-complete": (_ui(FORWARD_REF), True, set(), True),
    "no-ui-commit-not-complete": (
        [commit("thought", "OK."), commit("answer", "Here are your flights.")],
        False,
        set(),
        False,
    ),
}


@pytest.mark.parametrize("row", list(READING_ROWS))
def test_complete_card_reading(row: str) -> None:
    """Criterion 4 on hand-made canvases through the real ``UiGate`` (premises: this module's own application of the
    definition agrees with the hand values, and the salvage row's TextField is in ``dropped`` only). ``complete`` =
    last call OK and nothing pruned on it; the last call, not the first nor any (OK-then-pruned is not complete,
    pruned-then-clean is); salvage drops count; forward-ref prunes do not; no ``ui`` commit is not complete."""
    commits, last_ok, pruned, complete = READING_ROWS[row]
    rid = "t0108a-r0-flight_booking-K1"
    raws = [c["text"] for c in commits if c["segment"] == "ui"]
    if raws:
        assert _oracle(raws, rid) == (last_ok, pruned), f"premise ({row}): {_oracle(raws, rid)}"
    if row.startswith("adv-salvage"):
        g = UiGate()
        rep = g._repair(SALVAGE_DROP)
        assert (
            g.validator.invalid_component_ids(list(g.validator.iter_components(rep.messages))) == []
        )
        assert tuple(rep.dropped) == ("f",), "premise: the drop is salvage's alone"
    got = harness_108a().ui_reading(commits, request_id=rid)
    assert bool(got["last_ok"]) is last_ok, (row, got["last_ok"])
    assert set(got["pruned_last"]) == pruned, (row, got["pruned_last"])
    assert bool(got["complete"]) is complete, (row, got["complete"])


# =====================================================================================================
# Criteria 5-7: hand-built T-0108a runs
# =====================================================================================================

COMPLETE, PRUNED, FAILED_LAST = SHOWN, "pruned_card", SHOWN_FAIL

#: The run fixtures' cards, kept to one or two components because gating dominates a report's cost (T-0108a fast-suite
#: budget). Premises in ``test_run_fixture_cards``: the complete card is shown with nothing pruned; the pruned card is
#: shown with exactly the TextField ``f`` pruned (unrepairable since T-0108b 9(ii), as in ``PLACEHOLDER``).
RUN_COMPLETE = CREATE + "\n" + UPDATE + '{"id":"root","component":"Text","text":"Your trip"}]}}'
RUN_PRUNED = (
    CREATE + "\n" + UPDATE + '{"id":"root","component":"Column","children":["f"]},'
    '{"id":"f","component":"TextField","variant":"huge","label":"From"}]}}'
)


def commits_108a(outcome: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if outcome == COMPLETE:
        return _ui(RUN_COMPLETE)
    if outcome == PRUNED:
        return _ui(RUN_PRUNED)
    if outcome == FAILED_LAST:
        return _ui(RUN_COMPLETE, PROSE)
    if outcome == OPEN_SURFACE:
        return [
            commit("thought", "I will build it.", open_=True),
            commit("ui", RUN_COMPLETE, open_=True),
        ]
    return commits_for_c(outcome, row)


def test_run_fixture_cards() -> None:
    """Premises of the run fixtures, on the real gate (criterion 4's definition, applied by ``_oracle``): the complete
    card is shown with nothing pruned; the pruned card is shown with ``f`` pruned; complete-then-prose ends on a failed
    call."""
    assert _oracle([RUN_COMPLETE], "p") == (True, set())
    assert _oracle([RUN_PRUNED], "p") == (True, {"f"})
    assert _oracle([RUN_COMPLETE, PROSE], "p") == (False, set())


Spec = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...]
]  # complete, pruned, failed-last per positive case


def set_pos(plan: Plan, arm: str, spec: Spec) -> None:
    """Per positive case, repeats 0.. are complete, then pruned, then failed-last, the rest speech only."""
    zeros = (0, 0, 0, 0)
    complete, pruned, failed = (*spec, zeros, zeros)[:3]
    for cid, c, p, f in zip(Q12_POSITIVE, complete, pruned or zeros, failed or zeros, strict=True):
        assert c + p + f <= POSITIVE_REPEATS
        for r in range(POSITIVE_REPEATS):
            plan[(arm, cid, r)] = (
                COMPLETE
                if r < c
                else PRUNED
                if r < c + p
                else FAILED_LAST
                if r < c + p + f
                else NONE
            )


def four(x: int) -> tuple[int, int, int, int]:
    return (x, x, x, x)


A0_BASE: Spec = (four(20), four(16), four(0))
K1_BASE: Spec = (four(36), four(0), four(0))


def baseline_108a(positive_repeats: int = POSITIVE_REPEATS) -> Plan:
    plan: Plan = {}
    for arm in ARMS:
        for row in case_rows_c():
            n = positive_repeats if row["id"] in Q12_POSITIVE else OTHER_REPEATS
            for r in range(n):
                plan[(arm, row["id"], r)] = SPEAK if row["kind"] == "tool" else NONE
    if positive_repeats == POSITIVE_REPEATS:
        set_pos(plan, "A0", A0_BASE)
        set_pos(plan, "K1", K1_BASE)
        set_pos(plan, "K2", A0_BASE)
    return plan


def write_run_108a(
    run_dir: Path,
    plan: Plan,
    *,
    positive_repeats: int = POSITIVE_REPEATS,
    drop: Iterable[tuple[str, str, int]] = (),
) -> Path:
    """trials.jsonl in the schedule of *Design* (by hand), T-0103's line shape with ``t0108a-...`` ids, and a manifest
    naming table T-0108a and rule H-T108a."""
    texts = texts_by_hand()
    rows = {r["id"]: r for r in case_rows_c()}
    skip = set(drop)
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, cid, r, slot, arm in expected_schedule(positive_repeats):
        if (arm, cid, r) in skip:
            continue
        row = rows[cid]
        outcome = plan.get((arm, cid, r), NONE if row["kind"] != "tool" else SPEAK)
        err = outcome == ERROR_C
        line = {
            "case_id": cid,
            "kind": row["kind"],
            "tool": row.get("tool"),
            "gt": row.get("gt"),
            "case_index": i,
            "arm": arm,
            "repeat": r,
            "order": slot,
            "request_id": f"t0108a-r{r}-{cid}-{arm}",
            "system_sha256": sha256(texts[arm]),
            "commits": commits_108a(outcome, row),
            "token_ids": [] if err else [2, 3, 5, 106],
            "error": "ReadTimeout('mock')" if err else None,
            "wall_ms": 640.0,
            "first_canvas_ms": None if err else 310.0,
        }
        if row["kind"] == "q12":
            line["expect_surface"] = row["expect_surface"]
        lines.append(json.dumps(line))
    (run_dir / "trials.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "task": TABLE,
        "table": TABLE,
        "rule": RULE,
        "regime": {"profile": "lean", "split": "dev", "seed": None, "concurrency": 1},
        "arms": {a: {"text": t, "sha256": sha256(t)} for a, t in texts.items()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return run_dir


def report_108a(
    tmp_path: Path,
    edit: Callable[[Plan], None] | None = None,
    *,
    positive_repeats: int = POSITIVE_REPEATS,
    drop: Iterable[tuple[str, str, int]] = (),
) -> dict[str, Any]:
    """``build_report`` on the hand-built run, built once per distinct run in this process: the cache key is the run's
    whole content (every planned outcome, the repeats and the dropped lines), so two rows share a report only when
    their trials.jsonl would be byte-identical. Each caller gets its own deep copy."""
    plan = baseline_108a(positive_repeats)
    if edit is not None:
        edit(plan)
    drop = tuple(sorted(drop))
    key = (tuple(sorted(plan.items())), positive_repeats, drop)
    if key not in _REPORTS:
        h = harness_108a()
        memoize_ui_reading(h)  # T-0110 part A: this report's harness object only
        report = h.build_report(
            write_run_108a(tmp_path / "run", plan, positive_repeats=positive_repeats, drop=drop)
        )
        assert isinstance(report, dict)
        _REPORTS[key] = report
    return copy.deepcopy(_REPORTS[key])


_REPORTS: dict[Any, dict[str, Any]] = {}


def failing(report: Mapping[str, Any], k: str) -> set[str]:
    return {c for c in CONDS if report["candidates"][k][c] is not True}


def kind_of(report: Mapping[str, Any]) -> str:
    o = str(report["outcome"])
    if o.startswith(("K1 selected", "K2 selected")):
        return "selected"
    if o.startswith("no change: "):
        return "no change"
    if o.startswith("inconclusive: "):
        return "inconclusive"
    raise AssertionError(f"criterion 5: outcome {o!r} is none of the three pre-registered forms")


def named(report: Mapping[str, Any]) -> set[str]:
    return set(COND_RE.findall(str(report["outcome"]).split(":", 1)[1]))


def _get_path(arm: Mapping[str, Any], key: str) -> Any:
    out: Any = arm
    for part in key.split("."):
        out = out[part]
    return out


def _pos(a0: Spec, k1: Spec, k2: Spec | None = None) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        set_pos(plan, "A0", a0)
        set_pos(plan, "K1", k1)
        set_pos(plan, "K2", a0 if k2 is None else k2)

    return edit


def _put(arm: str, slots: Iterable[tuple[str, int]], outcome: str) -> Callable[[Plan], None]:
    slots = list(slots)

    def edit(plan: Plan) -> None:
        for cid, r in slots:
            plan[(arm, cid, r)] = outcome

    return edit


def _seq(*edits: Callable[[Plan], None]) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        for e in edits:
            e(plan)

    return edit


def _type_silent(arm: str) -> Callable[[Plan], None]:
    return _put(
        arm, [(f"type-{i:02d}", r) for i in range(1, 11) for r in range(OTHER_REPEATS)], SILENT
    )


def _kn(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


#: The 63/160 A0 of the Evidence, by case 16, 16, 16, 15 complete, each case's last-OK 36.
A0_63: Spec = ((16, 16, 16, 15), (20, 20, 20, 21), four(0))
A0_64: Spec = (four(16), four(20), four(0))
K_90: Spec = ((23, 23, 22, 22), (13, 13, 14, 14), four(0))


def test_t0108a_denominators_ignore_the_extra_repeats(tmp_path: Path) -> None:
    """Adversarial: on the baseline every arm has *Design*'s denominators -- 440 trials; tool trials 200; false tools
    over 65; surfaces where none belongs over 80 and on tool trials over 200; Q12 positives 160, negatives 15, Q12
    calls 175; 40 per positive case. A0 complete 80, last-OK 144, shown 144 (20/36/36 per case); K1 144/144/144
    (36 per case); K2 as A0."""
    report = report_108a(tmp_path)
    want = {
        "A0": (80, 144, 144, 20, 36),
        "K1": (144, 144, 144, 36, 36),
        "K2": (80, 144, 144, 20, 36),
    }
    for arm, (c, lo, sh, pc, pl) in want.items():
        a = report["arms"][arm]
        assert a["trials"] == PER_ARM and a["errored"] == _kn(0, PER_ARM)
        assert a["route_recall"] == _kn(200, 200) and a["thought_open"] == _kn(0, 200)
        assert a["false_tools"] == _kn(0, 65)
        assert a["surfaces_no_belong"] == _kn(0, 80) and a["surfaces_tool"] == _kn(0, 200)
        assert a["q12_negative_shown"] == _kn(0, 15) and a["q12_tool_calls"] == _kn(0, 175)
        assert a["q12_positive_complete"] == _kn(c, 160)
        assert a["q12_positive_last_ok"] == _kn(lo, 160)
        assert a["q12_positive_shown"] == _kn(sh, 160)
        for cid in Q12_POSITIVE:
            assert a["q12_per_case_complete"][cid] == _kn(pc, 40), (arm, cid)
            assert a["q12_per_case_last_ok"][cid] == _kn(pl, 40), (arm, cid)
            assert a["q12_per_case"][cid] == _kn(pl, 40), (arm, cid)
    assert kind_of(report) == "selected" and report["winner"] == "K1"
    assert report["readings"] == {"R3": True, "R4": True}


def _all_k2(rows_n: int = OTHER_REPEATS) -> list[tuple[str, str, int]]:
    return [("K2", r["id"], k) for r in case_rows_c() for k in range(rows_n)]


@pytest.mark.parametrize(
    ("case", "positive_repeats", "drop", "complete"),
    [
        ("1320-lines-complete", 40, (), True),
        ("1319-lines-one-missing-incomplete", 40, (("K2", "plain-03", 4),), False),
        ("1317-lines-last-positive-repeat-missing-incomplete", 40,
         tuple((a, "primes_table", 39) for a in ARMS), False),
        ("900-lines-positives-at-5-incomplete", 5, (), False),
        ("600-lines-two-arms-at-5-incomplete", 5, tuple(_all_k2()), False),
    ],
)  # fmt: skip
def test_t0108a_completeness(
    tmp_path: Path, case: str, positive_repeats: int, drop: tuple[Any, ...], complete: bool
) -> None:
    """Adversarial: the 1320-line run is complete; a run missing one line or the last positive repeat is incomplete
    with "incomplete" in its reason (it fails nothing else); a three-arm run with the positives at 5 (900 lines) and a
    two-arm run at 5 repeats (600 lines) are incomplete, inconclusive, and select nothing (their R3 fails too, so the
    reason is not pinned)."""
    report = report_108a(tmp_path, positive_repeats=positive_repeats, drop=drop)
    assert report["complete"] is complete, case
    if complete:
        assert report["trials"] == N_REQUESTS and kind_of(report) == "selected"
        return
    assert kind_of(report) == "inconclusive", report["outcome"]
    assert report["winner"] in (None, "no change")
    if positive_repeats == POSITIVE_REPEATS:
        assert report["readings"] == {"R3": True, "R4": True}, "premise: only completeness fails"
        assert "incomplete" in str(report["outcome"]), report["outcome"]


OUTCOME_EDITS: dict[str, tuple[Callable[[Plan], None], str, str | None, str | None]] = {
    # (edit, kind, winner, reason)
    "k1-selected": (lambda plan: None, "selected", "K1", None),
    "k2-selected-k1-fails": (_pos(A0_BASE, A0_BASE, K1_BASE), "selected", "K2", None),
    "tie-k1-selected": (_pos(A0_BASE, K1_BASE, K1_BASE), "selected", "K1", None),
    "no-change-names-every-failing-condition": (
        _seq(
            _pos(A0_63, ((22, 22, 21, 21), (14, 14, 15, 15))),
            _put("K1", NO_BELONG_SLOTS[:2], WITH_SURFACE),
            _put("K1", NEGATIVE_SLOTS[:11], Q12_TOOL),
        ),
        "no change", None, None,
    ),
    "inconclusive-incomplete": (lambda plan: None, "inconclusive", None, "incomplete"),
    "inconclusive-void-a0": (_put("A0", PLAIN_SLOTS[:9], ERROR_C), "inconclusive", None, "void"),
    "inconclusive-void-k2": (_put("K2", PLAIN_SLOTS[:9], ERROR_C), "inconclusive", None, "void"),
    "inconclusive-r3-beats-a-failing-k": (
        _pos((four(20), (8, 8, 8, 7)), ((22, 22, 21, 20), (6, 6, 7, 7))), "inconclusive", None, "R3",
    ),
    "inconclusive-r4-beats-a-failing-k": (
        _pos(((25, 24, 24, 24), (11, 12, 12, 12)), ((25, 24, 24, 24), (11, 12, 12, 12))), "inconclusive", None,
        "R4",
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(OUTCOME_EDITS))
def test_t0108a_outcomes(tmp_path: Path, case: str) -> None:
    """Criterion 5's outcomes and criterion 7's ``winner``. K1 selected on the baseline (readings exactly R3 and R4,
    candidates exactly K1 and K2, table and rule named); K2 selected when K1 copies A0; a tie (both 144) goes to K1.
    No change with K1 failing Q1 (86 vs 63), P9 (2/80) and P11 (11/175) and K2 failing Q1: the outcome names K1, K2,
    Q1, P9 and P11 and no other condition. Inconclusive when one line is missing, when A0 or K2 is void (9 of 440),
    when R3 fails (A0 shown 111) with K1 also failing Q1, and when R4 fails (A0 97) with K1 equal to A0."""
    edit, kind, winner, reason = OUTCOME_EDITS[case]
    drop = (("K2", "plain-03", 4),) if case == "inconclusive-incomplete" else ()
    report = report_108a(tmp_path, edit, drop=drop)
    assert kind_of(report) == kind, report["outcome"]
    if kind == "selected":
        assert report["winner"] == winner and str(report["outcome"]).startswith(
            f"{winner} selected"
        )
    else:
        assert report["winner"] in (None, "no change"), report["winner"]
    if case == "k1-selected":
        assert report["readings"] == {"R3": True, "R4": True}
        assert set(report["candidates"]) == set(CANDIDATES)
        assert report.get("table") == TABLE and report.get("rule") == RULE
        assert failing(report, "K1") == set() and failing(report, "K2") == {"Q1"}
    elif case == "no-change-names-every-failing-condition":
        assert failing(report, "K1") == {"Q1", "P9", "P11"} and failing(report, "K2") == {"Q1"}
        assert named(report) == {"Q1", "P9", "P11"}, report["outcome"]
        assert "K1" in str(report["outcome"]) and "K2" in str(report["outcome"])
    elif case == "inconclusive-r3-beats-a-failing-k":
        assert "Q1" in failing(report, "K1") and report["readings"]["R3"] is False
    elif case == "inconclusive-r4-beats-a-failing-k":
        assert "Q1" in failing(report, "K1") and report["readings"]["R4"] is False
    if reason is not None:
        assert reason in str(report["outcome"]), report["outcome"]
    if reason == "void":
        arm = "A0" if case.endswith("a0") else "K2"
        assert list(report["void_arms"]) == [arm] and arm in str(report["outcome"])


def test_t0108a_report_extras(tmp_path: Path) -> None:
    """Criterion 7 on a fixture where A0 is complete 24, 23, 20, 20 of 40 (87/160) with pruned 12, 13, 16, 16 (each
    case last-OK 36) and K1 and K2 as the baseline K1 (36 complete per case): ``q12_positive_complete`` k/160 per
    arm; per case complete, last-OK and shown k/40; Q2's protected flags by hand (A0 complete >= 24: flight_booking
    only); pruned kinds per arm (A0: TextField in 57 trials; K1: none) and per case (TextField 12 in flight_booking);
    the thresholds as k/n strings; the speech rates labelled "reported, not gated (D16)" in ``report.md``; the outcome
    "K1 selected" (a tie at 144) in ``report.md``."""
    edit = _pos(((24, 23, 20, 20), (12, 13, 16, 16)), K1_BASE, K1_BASE)
    report = report_108a(tmp_path, edit)
    a0, k1 = report["arms"]["A0"], report["arms"]["K1"]
    assert a0["q12_positive_complete"] == _kn(87, 160) and k1["q12_positive_complete"] == _kn(
        144, 160
    )
    assert {cid: a0["q12_per_case_complete"][cid] for cid in Q12_POSITIVE} == dict(
        zip(Q12_POSITIVE, (_kn(24, 40), _kn(23, 40), _kn(20, 40), _kn(20, 40)), strict=True)
    )
    for cid in Q12_POSITIVE:
        assert a0["q12_per_case_last_ok"][cid] == _kn(36, 40) == a0["q12_per_case"][cid]
        assert k1["q12_per_case_complete"][cid] == _kn(36, 40)
    want_protected = dict(zip(Q12_POSITIVE, (True, False, False, False), strict=True))
    assert want_protected in _protected_flags(report), "criterion 7: Q2's protected flags"
    assert _find(a0, {"TextField": 57}), (
        "criterion 7: A0's pruned kinds (57 trials prune a TextField)"
    )
    assert _find(a0, {"TextField": 12}), "criterion 7: flight_booking's pruned kinds"
    assert not _find(k1, {"TextField": 57}) and not _find(k1, {"TextField": 12})
    flat = json.dumps(report)
    for kn in (
        "24/160",
        "24/40",
        "16/160",
        "1/80",
        "2/200",
        "8/160",
        "10/175",
        "112/160",
        "96/160",
    ):
        assert f'"{kn}"' in flat, f"criterion 7: the threshold {kn} is not in the report as k/n"
    assert str(report["outcome"]).startswith("K1 selected") and report["winner"] == "K1"
    md = harness_108a().render_markdown(report)
    assert str(report["outcome"]) in md
    assert "reported, not gated (D16)" in md
    assert "T-0098c UI-block prompt A/B" not in md and "T-0098 native-prompt A/B" not in md


# =====================================================================================================
# T-0155 criterion 8: the frozen harness refuses today's persona
# =====================================================================================================


def test_harness_refuses_live_persona(
    pre_t0155: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T-0155 criterion 8: with the pre-T-0155 patch undone, ``main --table T-0108a`` sees the live Gearvis A0
    (``869f8f96...``) and exits 2 with its A0-drift refusal naming both shas (on the A0 check, or as the ``; and
    the A0 system text ...`` suffix when K1/K2 cannot be built from a drifted A0), before any socket and without a
    run directory."""
    from tests.unit.test_persona_name import A0_T0155_SHA256

    out = tmp_path / "t0108a-out"
    _, rc, said, attempts = run_main_against_live_persona(
        pre_t0155, harness_108a, lambda h: h._T93, ["--table", TABLE], out, monkeypatch, capsys
    )
    assert rc == 2, f"T-0155 criterion 8: exit {rc}: {said[-400:]!r}"
    assert attempts == [], f"T-0155 criterion 8: sockets before the refusal: {attempts}"
    assert not out.exists(), "T-0155 criterion 8: the refusal wrote a run directory"
    assert f"the A0 system text has sha256 {A0_T0155_SHA256}, not {PRE_T0155_A0_SHA256}" in said, (
        f"T-0155 criterion 8: not the A0-drift refusal: {said[-400:]!r}"
    )
