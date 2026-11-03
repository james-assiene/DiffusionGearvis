"""T-0103: the C3 confirmation -- a third arm table (A0 vs C3) on the T-0098 harness, 40 repeats on the four Q12
positive cases and 5 elsewhere (880 requests), and the pre-registered rule H-T103 with P8/P10/P11 scaled by reading S
(unit; CPU; no engine).

Rows that need the T-0103 table are red on the named assertion in ``harness_103()`` (``TABLES["T-0103"]`` /
``schedule`` absent) until the table exists, never on an import error: the harness is loaded lazily, by path, through
T-0098's loader. The guards on the earlier tables pass before and after the implementation (criterion 2).

**The assumed surface** (``scripts/gpu/t0098_native_prompt_ab.py``; the task names the behaviour, these names are the
test-author's, stated so the implementer can honour them or say why not):

- ``TABLES["T-0103"]`` is an ``ArmTable`` (the existing frozen dataclass) with ``arms == ("A0", "C3")``,
  ``candidates == ("C3",)``, ``rule == "H-T103"``, ``a0_sha256``, ``texts`` (the arm-text builder) and a new field
  ``repeats``: a mapping case id -> repeats (40 on the four positives, 5 on the other 56). The refusal rows replace
  ``texts`` or ``repeats`` with ``dataclasses.replace`` and ``monkeypatch.setitem(h.TABLES, "T-0103", ...)``, so
  ``main`` must check the C3 pin and the repeats map on what the table gives it (not only inside the builder).
- ``schedule(table) -> Sequence[item]``: the run's requests in request order; an item has (as keys or attributes)
  ``case_index``, ``case_id``, ``repeat``, ``order`` (the slot in its block), ``arm`` and ``request_id``. For T-0098
  and T-0098c it is the schedule ``_run_all`` runs today (1590 / 2100 requests, request ids ``t0098-...`` /
  ``t0098c-...``).
- ``arm_order(case_index, repeat, "T-0103")``, ``arm_texts(tools, table="T-0103")``, ``load_cases(table="T-0103")``,
  ``arm_prompts(case, kit, table="T-0103")`` as for the earlier tables.
- ``build_report(run_dir)`` on a T-0103 run (manifest ``table: "T-0103"``) returns the T-0098c keys for the two arms
  (``arms[a]["q12_positive_shown"]`` and the rest, with T-0103's denominators: 160 positives, 15 negatives, 40 per
  positive case) plus ``candidates["C3"][P1..P11]``, ``readings`` holding exactly ``R1`` and ``R3``, ``void_arms``,
  ``complete``, ``winner`` (``"C3"`` iff confirmed, else ``"no change"``) and ``outcome``: a string of one of the three
  pre-registered forms ``"C3 confirmed"``, ``"C3 not confirmed: fails P..."`` (every failing P named) or
  ``"inconclusive: <reason>"`` (the reason says ``incomplete``, ``void`` with the arm, ``finding not reproduced`` or
  ``UI baseline not reproduced``).

**The baseline T-0103 run** (every rule row edits it; all numbers by hand): 60 cases, A0 and C3, repeats 40 on
``flight_booking``, ``flight_booking_with_memory``, ``primes_table``, ``omelette_steps`` and 5 elsewhere = 880 lines,
440 per arm, written in the schedule order of *Design*.

- tool cases: A0 ``type_at`` SILENT on all 50 (R1 holds, 0/50 <= 3/10); every other tool trial speaks, C3's
  ``type_at`` included (50/50, 10/10 majority cases).
- no-tool, plain and Q12 negative cases: an answer, no call, no surface.
- Q12 positives: both arms show ``UI_EXAMPLE`` at repeats 0-29 and answer in speech only at 30-39: 30/40 per case,
  120/160 (R3 holds, 120 >= 112; every case protected, 30 >= 24).
- So C3 passes P1-P11 and the outcome is "C3 confirmed".

Thresholds are typed out from docs/tasks/T-0103.md criterion 5 (reading S, ruled H-T103 (c)), never imported from the
harness: P8 total C3 >= A0 - 16 (of 160); a positive case is protected iff A0 >= 24/40, and P8 fails if a protected
case has C3 <= 23/40; P10 <= A0 + 8; P11 <= A0 + 10 (of 175); P9 verbatim (<= A0 + 1 of 80, <= A0 + 2 of 200); R1 A0
``type_at`` spoken <= 3/10; R3 A0 positive shown >= 112/160; void > 2 % of 440, i.e. >= 9.

Regime: CPU, pure, hand-built data, the real ``UiGate`` and vendored A2UI v0.9.1 schema; no measurement.
"""

from __future__ import annotations

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

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import VllmClient
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.integration.test_t0098_harness_smoke import no_sockets
from tests.unit._harness_reading_memo import memoize_ui_reading
from tests.unit._pre_t0155_persona import (
    PRE_T0155_A0_SHA256,
    pre_t0155_fixture,
    run_main_against_live_persona,
)
from tests.unit.test_t0098_prompt_ab import (
    A0_SHA256,
    ANOMALY,
    HARNESS,
    NONE,
    PLAIN_QUESTIONS,
    REPO,
    SILENT,
    SPEAK,
    baseline,
    sha256,
    tool_kit,
    write_run,
)
from tests.unit.test_t0098c_ui_prompt_ab import (
    ERROR_C,
    OPEN_SURFACE,
    PS_C,
    Q12_NEGATIVE,
    Q12_POSITIVE,
    Q12_TOOL,
    SHOWN,
    WITH_SURFACE,
    baseline_c,
    case_rows_c,
    commits_for_c,
    harness_c,
    patch_u,
    pieces,
    q12_shown,
    type_silent,
    write_run_c,
)

#: The commit T-0103 branched from (task header: ``main`` at ``3578b66``). Criterion 2: the earlier tables' output is
#: exactly as here. A literal, never a moving ref (.claude/rules/tests.md).
BASE_SHA = "3578b662aebc4ec145e6cb6a182bdb4a7991daec"

TABLE = "T-0103"

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)
RULE = "H-T103"
ARMS_103 = ("A0", "C3")
CANDIDATES_103 = ("C3",)
#: T-0103 *Arms*: C3's pin.
C3_SHA256 = "72c3b92bcff2e5924901e42475342826ffc38efefa116f66138f8aa43bad9308"
A0_CHARS = 2744
#: T-0103 *Design*.
POSITIVE_REPEATS = 40
OTHER_REPEATS = 5
N_REQUESTS = 880
PER_ARM = 440
#: T-0103 criterion 5 (reading S), typed out.
P8_TOTAL_MARGIN = 16
PROTECT_MIN = 24
P10_MARGIN = 8
P11_MARGIN = 10
R3_MIN = 112

Plan = dict[tuple[str, str, int], str]


# =====================================================================================================
# Loading
# =====================================================================================================


def harness_103() -> ModuleType:
    """The harness (T-0098's loader), with the T-0103 names asserted. Red here until criterion 1."""
    h = harness_c()
    assert TABLE in h.TABLES, (
        "T-0103 criterion 1: the harness has no TABLES['T-0103'] (see this module's docstring)"
    )
    assert callable(getattr(h, "schedule", None)), (
        "T-0103 criteria 1/4: the harness has no callable schedule(table) (see this module's docstring)"
    )
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
    """The harness exactly as at ``BASE_SHA``, compiled from the git blob with the real file's ``__file__`` (so its
    ``REPO`` and pinned paths resolve as they did): the frozen oracle for criterion 2. Skips when the commit is
    absent."""
    src = _old_source()
    if src is None:
        pytest.skip(
            f"{BASE_SHA[:7]} is not in this clone; the frozen T-0098/T-0098c oracle is unavailable"
        )
    name = "_t0098_harness_at_3578b66"
    module = ModuleType(name)
    module.__file__ = str(HARNESS)
    sys.modules[name] = module  # its dataclasses look themselves up here while they are built
    try:
        exec(compile(src, f"{BASE_SHA[:7]}:{HARNESS.name}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _get(table: Any, name: str) -> Any:
    if isinstance(table, Mapping):
        return table[name]
    return getattr(table, name)


def expected_repeats() -> dict[str, int]:
    return {
        r["id"]: (POSITIVE_REPEATS if r["id"] in Q12_POSITIVE else OTHER_REPEATS)
        for r in case_rows_c()
    }


def expected_schedule(
    positive_repeats: int = POSITIVE_REPEATS,
) -> list[tuple[int, str, int, int, str]]:
    """(case_index, case_id, repeat, slot, arm) in request order, by hand from *Design*: repeats outermost; within a
    repeat the cases scheduled for it in index order; per (case, repeat) both arms from offset (i + r) mod 2 of A0 C3."""
    rows = case_rows_c()
    out: list[tuple[int, str, int, int, str]] = []
    for r in range(max(positive_repeats, OTHER_REPEATS)):
        for i, row in enumerate(rows):
            n = positive_repeats if row["id"] in Q12_POSITIVE else OTHER_REPEATS
            if r >= n:
                continue
            first = "A0" if (i + r) % 2 == 0 else "C3"
            second = "C3" if first == "A0" else "A0"
            out.append((i, row["id"], r, 0, first))
            out.append((i, row["id"], r, 1, second))
    return out


# =====================================================================================================
# Criterion 1: the arm texts, by hand
# =====================================================================================================


def a0_c3_by_hand() -> tuple[str, str]:
    p, ct, ss, u, n = pieces()
    return "\n".join([p, ct, ss, u, n]), "\n".join([p, ct, ss, n, u])


def test_t0103_arm_texts_by_hand() -> None:
    """Criterion 1 and *Arms*: A0 = join(P, CT, SS, U, N) and C3 = join(P, CT, SS, N, U), built here from the imported
    constants, compared as strings and as the task's shas (``3605c012...``, ``72c3b92b...``, 2744 characters each). C3
    is A0's character multiset in another order. The table holds exactly the two arms, one candidate (C3), rule H-T103,
    A0's pin, and the repeats map of *Design*. C3 is byte-identical to T-0098c's C3, and for every one of the 60 cases
    the A0 and C3 prompts equal T-0098c's A0 and C3 prompts as built by the harness at 3578b66 (whose A0 a ``Session``
    builds, T-0098c criterion 4 and N10: Q12 no frame, memory hits with no ``now_ms``); the two bodies differ only in
    ``messages[0]`` (``check_fidelity`` finds nothing)."""
    h = harness_103()
    a0, c3 = a0_c3_by_hand()
    assert sha256(a0) == A0_SHA256 and sha256(c3) == C3_SHA256, "premise: the task's shas"
    assert len(a0) == len(c3) == A0_CHARS
    assert Counter(a0) == Counter(c3) and a0 != c3

    kit = tool_kit()
    texts = h.arm_texts(kit.tools, table=TABLE)
    assert dict(texts) == {"A0": a0, "C3": c3}, (
        "criterion 1: the arm texts are not the hand-built ones"
    )
    t = h.TABLES[TABLE]
    assert tuple(_get(t, "arms")) == ARMS_103
    assert tuple(_get(t, "candidates")) == CANDIDATES_103
    assert _get(t, "rule") == RULE
    assert _get(t, "a0_sha256") == A0_SHA256
    assert dict(_get(t, "repeats")) == expected_repeats(), "criterion 1: the per-case repeats map"

    old = old_harness()
    assert texts["C3"] == old.arm_texts(kit.tools, table="T-0098c")["C3"]
    old_cases = {c.case_id: c for c in old.load_cases(table="T-0098c")}
    cases = h.load_cases(table=TABLE)
    assert [c.case_id for c in cases] == [r["id"] for r in case_rows_c()]
    client = VllmClient()
    for case in cases:
        prompts = h.arm_prompts(case, kit, table=TABLE)
        assert set(prompts) == set(ARMS_103), case.case_id
        ref = old.arm_prompts(old_cases[case.case_id], kit, table="T-0098c")
        assert prompts["A0"] == ref["A0"], f"criterion 4: {case.case_id} A0 is not T-0098c's"
        assert prompts["C3"] == ref["C3"], f"criterion 4: {case.case_id} C3 is not T-0098c's"
        bodies = {a: client.request_body(p, "fidelity") for a, p in prompts.items()}
        assert h.check_fidelity(bodies) == [], case.case_id


# =====================================================================================================
# Criterion 4: the schedule
# =====================================================================================================


def test_t0103_schedule() -> None:
    """*Design*, by hand: 880 requests, unique request ids ``t0103-r{repeat}-{case_id}-{arm}``; 40 repeats per arm on
    indices 53-56 and 5 elsewhere; r >= 5 schedules only 53-56; repeats outermost, cases in index order; A0 first iff
    (case_index + repeat) is even, so each positive case has A0 first 20 times of 40; case indices equal T-0098c's
    (from the pinned files, and from the harness at 3578b66). A0-always-first fails the slot check; an expected set
    built from 5 repeats fails the length."""
    h = harness_103()
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
    assert len(want) == N_REQUESTS == 4 * 40 * 2 + 56 * 5 * 2, "premise"
    assert len(got) == N_REQUESTS, f"criterion 4: {len(got)} requests"
    assert got == want, "criterion 4: the schedule is not *Design*'s"
    ids = [str(_field(x, "request_id")) for x in items]
    assert len(set(ids)) == N_REQUESTS
    assert ids == [f"t0103-r{r}-{cid}-{arm}" for _, cid, r, _, arm in want]

    per: Counter[tuple[str, str]] = Counter((cid, arm) for _, cid, _, _, arm in got)
    for cid, n in expected_repeats().items():
        assert per[(cid, "A0")] == per[(cid, "C3")] == n, cid
    assert {i for i, _, r, _, _ in got if r >= 5} == {53, 54, 55, 56}
    first = Counter(cid for _, cid, _, slot, arm in got if slot == 0 and arm == "A0")
    for cid in Q12_POSITIVE:
        assert first[cid] == 20, (cid, first[cid])
    for i, r in ((53, 0), (54, 0), (53, 1), (0, 0), (1, 0), (59, 4)):
        want_first = "A0" if (i + r) % 2 == 0 else "C3"
        assert tuple(h.arm_order(i, r, TABLE)) == (want_first, "C3" if want_first == "A0" else "A0")
    old_ids = [c.case_id for c in old_harness().load_cases(table="T-0098c")]
    assert [c.case_id for c in h.load_cases(table=TABLE)] == old_ids
    assert {i: cid for i, cid, *_ in got} == dict(enumerate(old_ids))


# =====================================================================================================
# Criterion 2: the earlier tables are untouched
# =====================================================================================================


def _t0098c_fixture_plan() -> Plan:
    """T-0098c's baseline (tests/unit/test_t0098c_ui_prompt_ab.py: A0 Q12 positive 18/20) with C3 at 4, 4, 4, 3 shown
    (15/20: P8 fails at A0 - 2 = 16), two of C3's shown positives built inside an open thought (placement 2 > A0 + 1:
    P10 fails) and three Q12 calls on the negatives (3 > A0 + 2: P11 fails). Scaled bars (A0 - 16, + 8, + 10) would
    pass all three."""
    plan = baseline_c()
    q12_shown(
        plan,
        "C3",
        {
            "flight_booking": 4,
            "flight_booking_with_memory": 4,
            "primes_table": 4,
            "omelette_steps": 3,
        },
    )
    for cid in ("flight_booking", "primes_table"):
        plan[("C3", cid, 0)] = OPEN_SURFACE
    for cid in Q12_NEGATIVE:
        plan[("C3", cid, 0)] = Q12_TOOL
    return plan


#: The report keys of each earlier table at 3578b66 (criterion 2: build_report output exactly as there; no T-0103 key
#: such as ``outcome`` appears).
T0098_REPORT_KEYS = {
    "task", "regime", "arm_texts", "trials", "complete", "void_arms", "arms", "candidates", "readings", "winner",
    "notes",
}  # fmt: skip
T0098C_REPORT_KEYS = T0098_REPORT_KEYS | {"table", "rule"}


def test_prior_tables_untouched(tmp_path: Path) -> None:
    """Criterion 2 (guard; green today and through the implementation). Rotation mod 6 and mod 7 by hand; the
    T-0098 baseline (1590 lines) and a T-0098c fixture (2100 lines) give hand-counted, pinned reports -- T-0098: winner
    C2, R1 and R2 hold; T-0098c: A0 18/20, C3 15/20 with placement 2/20 and Q12 calls 3/35, C3 failing exactly P8, P10
    and P11 under T-0098c's unscaled bars, winner C5 -- with exactly the 3578b66 key sets; and ``build_report`` /
    ``render_markdown`` on both equal the output of the harness as at 3578b66 (the frozen oracle), byte for byte."""
    h = harness_c()
    old = old_harness()
    # rotation, by hand (the T-0098 and T-0098c modules pin the same offsets; restated so this row stands alone)
    assert tuple(h.arm_order(52, 4)) == ("D1", "D2", "C2", "C1", "A0", "A1")  # (52 + 4) mod 6 = 2
    assert tuple(h.arm_order(53, 0, "T-0098c")) == (
        "C3",
        "C5",
        "C4",
        "A0",
        "D1",
        "D3",
        "D4",
    )  # 53 mod 7 = 4
    assert len(h.load_cases()) == 53 and len(h.load_cases(table="T-0098c")) == 60

    memoize_ui_reading(h)  # T-0110 part A: the report building below, on these two objects only
    memoize_ui_reading(old)
    run98 = write_run(tmp_path / "t0098", baseline())
    rep98 = h.build_report(run98)
    assert set(rep98) == T0098_REPORT_KEYS
    assert rep98["winner"] == "C2" and rep98["readings"] == {"R1": True, "R2": True}
    assert rep98["trials"] == 1590 and rep98["complete"] is True
    assert json.dumps(rep98, sort_keys=True) == json.dumps(old.build_report(run98), sort_keys=True)
    assert h.render_markdown(rep98) == old.render_markdown(rep98)

    run98c = write_run_c(tmp_path / "t0098c", _t0098c_fixture_plan())
    rep = h.build_report(run98c)
    assert set(rep) == T0098C_REPORT_KEYS, sorted(set(rep) ^ T0098C_REPORT_KEYS)
    assert rep["trials"] == 2100 and rep["complete"] is True and rep["table"] == "T-0098c"
    a0, c3 = rep["arms"]["A0"], rep["arms"]["C3"]
    assert a0["q12_positive_shown"] == {"k": 18, "n": 20}
    assert c3["q12_positive_shown"] == {"k": 15, "n": 20}
    assert c3["placement_failures_q12_positive"] == {"k": 2, "n": 20}
    assert c3["q12_tool_calls"] == {"k": 3, "n": 35}
    assert c3["q12_per_case"]["omelette_steps"] == {"k": 3, "n": 5}
    assert [p for p in PS_C if rep["candidates"]["C3"][p] is not True] == ["P8", "P10", "P11"]
    assert rep["readings"] == {"R1": True, "R2c": True, "R3": True}
    assert rep["winner"] == "C5"
    assert json.dumps(rep, sort_keys=True) == json.dumps(old.build_report(run98c), sort_keys=True)
    assert h.render_markdown(rep) == old.render_markdown(rep)


def test_prior_table_schedules() -> None:
    """Criterion 2 through the new ``schedule``: T-0098 is 1590 requests over 5 repeats in the mod-6 rotation with ids
    ``t0098-r{r}-{case}-{arm}``; T-0098c 2100 in mod 7 with ids ``t0098c-...``; every (case, arm, repeat) once."""
    h = harness_103()
    for table, arms, n_cases in (
        ("T-0098", ("A0", "A1", "D1", "D2", "C2", "C1"), 53),
        ("T-0098c", ("A0", "D1", "D3", "D4", "C3", "C5", "C4"), 60),
    ):
        items = list(h.schedule(table))
        assert len(items) == n_cases * len(arms) * 5, (table, len(items))
        prefix = "t0098" if table == "T-0098" else "t0098c"
        ids = [h.load_cases(table=table)[i].case_id for i in range(n_cases)]
        k = 0
        for r in range(5):
            for i, cid in enumerate(ids):
                off = (i + r) % len(arms)
                for slot in range(len(arms)):
                    x = items[k]
                    arm = arms[(off + slot) % len(arms)]
                    assert (
                        int(_field(x, "case_index")),
                        str(_field(x, "case_id")),
                        int(_field(x, "repeat")),
                        int(_field(x, "order")),
                        str(_field(x, "arm")),
                        str(_field(x, "request_id")),
                    ) == (i, cid, r, slot, arm, f"{prefix}-r{r}-{cid}-{arm}"), (table, k)
                    k += 1


# =====================================================================================================
# Criterion 3: the refusals (adversarial)
# =====================================================================================================


def _fake_git(real: Callable[..., str]) -> Callable[..., str]:
    """A clean tree at a fixed commit for ``main``'s two git reads; every other git call goes to the real one."""

    def git(*args: str) -> str:
        if args[:1] == ("status",):
            return ""
        if args[:1] == ("rev-parse",):
            return "0" * 40 + "\n"
        return real(*args)

    return git


def _replace_c3(
    h: ModuleType, monkeypatch: pytest.MonkeyPatch, variant: Callable[[str, str], str]
) -> None:
    t = h.TABLES[TABLE]
    orig = t.texts

    def texts(tools: Any) -> dict[str, str]:
        out = dict(orig(tools))
        out["C3"] = variant(out["A0"], out["C3"])
        return out

    monkeypatch.setitem(h.TABLES, TABLE, dataclasses.replace(t, texts=texts))


def _replace_repeats(
    h: ModuleType, monkeypatch: pytest.MonkeyPatch, edit: Mapping[str, int]
) -> None:
    t = h.TABLES[TABLE]
    monkeypatch.setitem(
        h.TABLES, TABLE, dataclasses.replace(t, repeats={**dict(t.repeats), **edit})
    )


def _other_order(a0: str, c3: str) -> str:
    p, ct, ss, u, n = pieces()
    return "\n".join([p, ct, n, ss, u])


REFUSALS_103: dict[str, tuple[str, ...]] = {
    "a0-drift-ui-instructions": ("3605c012", "72c3b92b"),
    "a0-drift-native-tool-instructions": ("3605c012", "72c3b92b"),
    "c3-builder-returns-a0-order": ("C3", "72c3b92b"),
    "c3-another-order-same-characters": ("C3", "72c3b92b"),
    "c3-trailing-space": ("C3", "72c3b92b"),
    "repeats-39-on-one-positive": ("repeat",),
    "repeats-5-on-the-positives": ("repeat",),
    "repeats-6-on-one-other-case": ("repeat",),
    "split-test": ("split",),
}


@pytest.mark.parametrize("why", [*REFUSALS_103, "control-unbroken"])
def test_t0103_refusals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, why: str
) -> None:
    """Criterion 3 (adversarial): under ``--table T-0103`` each row breaks one condition and ``main`` exits 2 with no
    socket at all, nothing written, and a message naming the cause -- A0 drift through ``UI_INSTRUCTIONS`` or
    ``NATIVE_TOOL_INSTRUCTIONS`` patched in the harness's and the builder's view; a C3 that is A0's order, another
    permutation of the same five pieces (same characters), or the right order plus one space; a repeats map with 39 on
    one positive, 5 on all four positives (the T-0098c design), or 6 on one other case; ``--split test``. The git
    reads see a clean tree (a double; the real clean-tree path is the integration row's), and the kit is the real one
    (``cli.load_tool_kit``; skipped without the pinned tokenizer). Guard in the other direction: the same invocation,
    unbroken, passes every refusal and reaches the engine (a socket attempt, refused here), so the rows above refuse
    for their cause, not because the guard is too wide. The UI drift appends 10 characters, so T-0098c's C4 block
    (1977 + 10) stays under its 2000 limit and the refusal seen is the pin's."""
    h = harness_103()
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    monkeypatch.setattr(h._T93, "_git", _fake_git(h._T93._git))
    out = tmp_path / "t0103-out"
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
    _, _, _, u, n = pieces()
    if why == "a0-drift-ui-instructions":
        patch_u(monkeypatch, h, u + " Be brief.")
    elif why == "a0-drift-native-tool-instructions":
        new_n = n + " Keep it short."
        monkeypatch.setattr(pb, "NATIVE_TOOL_INSTRUCTIONS", new_n)
        for name, value in list(vars(h).items()):
            if isinstance(value, str) and value is n:
                monkeypatch.setattr(h, name, new_n)
    elif why == "c3-builder-returns-a0-order":
        _replace_c3(h, monkeypatch, lambda a0, c3: a0)
    elif why == "c3-another-order-same-characters":
        _replace_c3(h, monkeypatch, _other_order)
        assert Counter(_other_order("", "")) == Counter(a0_c3_by_hand()[1]), (
            "premise: same multiset"
        )
    elif why == "c3-trailing-space":
        _replace_c3(h, monkeypatch, lambda a0, c3: c3 + " ")
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
            rc, reached_engine = None, True  # no_sockets refused the first socket
    said = capsys.readouterr()
    text = said.out + said.err
    if why == "control-unbroken":
        assert rc != 2 and (reached_engine or attempts), (
            f"the unbroken T-0103 invocation was refused (too wide a guard): rc={rc}: {text[-400:]!r}"
        )
        return
    assert rc == 2, f"criterion 3 ({why}): exit {rc}: {text[-400:]!r}"
    assert attempts == [] and not reached_engine, (
        f"criterion 3 ({why}): sockets before the refusal: {attempts}"
    )
    assert not out.exists() or not any(out.iterdir()), (
        f"criterion 3 ({why}): the refusal wrote files"
    )
    assert any(token in text for token in REFUSALS_103[why]), (
        f"criterion 3 ({why}): the refusal does not name its cause {REFUSALS_103[why]}: {text[-400:]!r}"
    )


# =====================================================================================================
# Criteria 5-7: hand-built T-0103 runs
# =====================================================================================================


def rows_103() -> list[dict[str, Any]]:
    return case_rows_c()


def repeats_of(row: Mapping[str, Any], positive_repeats: int = POSITIVE_REPEATS) -> int:
    return positive_repeats if row["id"] in Q12_POSITIVE else OTHER_REPEATS


def baseline_103(positive_repeats: int = POSITIVE_REPEATS) -> Plan:
    plan: Plan = {}
    for arm in ARMS_103:
        for row in rows_103():
            for r in range(repeats_of(row, positive_repeats)):
                if row["kind"] == "tool":
                    o = SILENT if (arm == "A0" and row["tool"] == "type_at") else SPEAK
                elif row["kind"] == "q12" and row["expect_surface"]:
                    o = SHOWN if r < 30 else NONE
                else:
                    o = NONE
                plan[(arm, row["id"], r)] = o
    return plan


def write_run_103(
    run_dir: Path,
    plan: Plan,
    *,
    positive_repeats: int = POSITIVE_REPEATS,
    drop: Iterable[tuple[str, str, int]] = (),
) -> Path:
    """trials.jsonl in the schedule order of *Design* (built by hand), the T-0098c line shape with
    ``t0103-r{r}-{case}-{arm}`` request ids, and a manifest naming table T-0103 and rule H-T103."""
    a0, c3 = a0_c3_by_hand()
    texts = {"A0": a0, "C3": c3}
    rows = {r["id"]: r for r in rows_103()}
    skip = set(drop)
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, cid, r, slot, arm in expected_schedule(positive_repeats):
        if (arm, cid, r) in skip:
            continue
        row = rows[cid]
        outcome = plan[(arm, cid, r)]
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
            "request_id": f"t0103-r{r}-{cid}-{arm}",
            "system_sha256": sha256(texts[arm]),
            "commits": commits_for_c(outcome, row),
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


def report_103(
    tmp_path: Path,
    edit: Callable[[Plan], None] | None = None,
    *,
    positive_repeats: int = POSITIVE_REPEATS,
    drop: Iterable[tuple[str, str, int]] = (),
) -> dict[str, Any]:
    plan = baseline_103(positive_repeats)
    if edit is not None:
        edit(plan)
    h = harness_103()
    memoize_ui_reading(h)  # T-0110 part A: this row's harness object only
    run = write_run_103(tmp_path / "run", plan, positive_repeats=positive_repeats, drop=drop)
    report = h.build_report(run)
    assert isinstance(report, dict)
    return report


def q4(a: int, b: int, c: int, d: int) -> dict[str, int]:
    return dict(zip(Q12_POSITIVE, (a, b, c, d), strict=True))


def set_q12(plan: Plan, arm: str, per_case: Mapping[str, int], *, last: bool = False) -> None:
    """``per_case[cid]`` of the case's 40 trials show a surface (repeats 0..k-1, or the last k with ``last``); the
    rest answer in speech only."""
    for cid, k in per_case.items():
        for r in range(POSITIVE_REPEATS):
            shown = (r >= POSITIVE_REPEATS - k) if last else (r < k)
            plan[(arm, cid, r)] = SHOWN if shown else NONE


def put(plan: Plan, arm: str, slots: Iterable[tuple[str, int]], outcome: str) -> None:
    for cid, r in slots:
        plan[(arm, cid, r)] = outcome


def failing(report: Mapping[str, Any]) -> set[str]:
    return {p for p in PS_C if report["candidates"]["C3"][p] is not True}


def kind_of(report: Mapping[str, Any]) -> str:
    o = str(report["outcome"])
    if o.startswith("C3 confirmed"):
        return "confirmed"
    if o.startswith("C3 not confirmed: fails "):
        return "not confirmed"
    if o.startswith("inconclusive: "):
        return "inconclusive"
    raise AssertionError(f"criterion 5: outcome {o!r} is none of the three pre-registered forms")


def named_ps(report: Mapping[str, Any]) -> set[str]:
    o = str(report["outcome"])
    return set(re.findall(r"\bP\d+\b", o.split("fails", 1)[1]))


#: Slots used by the rows (hand-chosen; all are SHOWN, or speech-only, in the baseline as stated).
PLACEMENT_SLOTS = [
    (cid, r) for r in range(3) for cid in Q12_POSITIVE
]  # 12 shown positive trials (r < 30)
NEGATIVE_SLOTS = [(cid, r) for cid in Q12_NEGATIVE for r in range(5)]  # the 15 Q12 negative trials
NO_BELONG_SLOTS = [("plain-01", 0), ("chat-01", 0), ("capital_of_france", 0)]
TOOL_SLOTS = [("click-01", 0), ("search-01", 0), ("scroll-01", 0), ("key-01", 0)]
PLAIN_SLOTS = [(cid, r) for r in range(5) for cid, _ in PLAIN_QUESTIONS]  # 25 per arm


def _p8(a0: tuple[int, int, int, int], c3: tuple[int, int, int, int]) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        set_q12(plan, "A0", q4(*a0))
        set_q12(plan, "C3", q4(*c3))

    return edit


def _seq(*edits: Callable[[Plan], None]) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        for e in edits:
            e(plan)

    return edit


def _put(arm: str, slots: Iterable[tuple[str, int]], outcome: str) -> Callable[[Plan], None]:
    slots = list(slots)
    return lambda plan: put(plan, arm, slots, outcome)


#: (edit, failing Ps of C3, outcome kind, reason fragment for inconclusive, {arm: {report key: value}} premises)
Row = tuple[Callable[[Plan], None], set[str], str, str | None, dict[str, dict[str, Any]]]
BOUNDARY_ROWS: dict[str, Row] = {
    # P8 total, reading S: C3 >= A0 - 16 of 160; every case stays protected and >= 24 so only the total decides
    "p8-total-120-to-104-passes": (
        _p8((30, 30, 30, 30), (26, 26, 26, 26)), set(), "confirmed", None,
        {"A0": {"q12_positive_shown": {"k": 120, "n": 160}}, "C3": {"q12_positive_shown": {"k": 104, "n": 160}}},
    ),
    "p8-total-120-to-103-fails": (
        _p8((30, 30, 30, 30), (26, 26, 26, 25)), {"P8"}, "not confirmed", None,
        {"C3": {"q12_positive_shown": {"k": 103, "n": 160}}},
    ),
    # adversarial (i): A0 - 10 passes (a literal margin of 2, or 1/10 x 20, fails it)
    "p8-adv-i-a0-minus-10-passes": (
        _p8((30, 30, 30, 30), (28, 28, 27, 27)), set(), "confirmed", None,
        {"C3": {"q12_positive_shown": {"k": 110, "n": 160}}},
    ),
    # per-case clause: protected iff A0 >= 24/40; C3 <= 23 there fails even with C3's total above A0's
    "p8-protected-24-to-23-fails-total-above-a0": (
        _p8((24, 32, 32, 32), (23, 35, 35, 35)), {"P8"}, "not confirmed", None,
        {"A0": {"q12_positive_shown": {"k": 120, "n": 160}}, "C3": {"q12_positive_shown": {"k": 128, "n": 160}}},
    ),
    "p8-protected-24-to-24-passes": (_p8((24, 32, 32, 32), (24, 32, 32, 32)), set(), "confirmed", None, {}),
    "p8-unprotected-23-to-5-holds": (
        _p8((23, 32, 32, 32), (5, 40, 40, 40)), set(), "confirmed", None,
        {"A0": {"q12_positive_shown": {"k": 119, "n": 160}}, "C3": {"q12_positive_shown": {"k": 125, "n": 160}}},
    ),
    # adversarial (ii): a per-case bar at the count 3 passes this
    "p8-adv-ii-protected-30-to-10-fails": (
        _p8((30, 30, 30, 30), (10, 40, 40, 40)), {"P8"}, "not confirmed", None, {}
    ),
    # adversarial (iii): protected set from A0 (flight cases), not C3 (primes/omelette); equal totals
    "p8-adv-iii-protected-set-read-from-a0": (
        _p8((40, 40, 20, 20), (20, 20, 40, 40)), {"P8"}, "not confirmed", None, {}
    ),
    # P10, reading S: C3 placement failures on positives <= A0 + 8 (ui commits inside an open thought; still shown)
    "p10-a0-2-c3-10-passes": (
        _seq(_put("A0", PLACEMENT_SLOTS[:2], OPEN_SURFACE), _put("C3", PLACEMENT_SLOTS[:10], OPEN_SURFACE)),
        set(), "confirmed", None,
        {"A0": {"placement_failures_q12_positive": {"k": 2, "n": 160}},
         "C3": {"placement_failures_q12_positive": {"k": 10, "n": 160}, "q12_positive_shown": {"k": 120, "n": 160}}},
    ),
    "p10-a0-2-c3-11-fails": (
        _seq(_put("A0", PLACEMENT_SLOTS[:2], OPEN_SURFACE), _put("C3", PLACEMENT_SLOTS[:11], OPEN_SURFACE)),
        {"P10"}, "not confirmed", None,
        {"C3": {"placement_failures_q12_positive": {"k": 11, "n": 160}}},
    ),
    # P11, reading S: routed calls on the 175 Q12 trials <= A0 + 10 (calls on the negatives)
    "p11-0-to-10of175-passes": (
        _put("C3", NEGATIVE_SLOTS[:10], Q12_TOOL), set(), "confirmed", None,
        {"C3": {"q12_tool_calls": {"k": 10, "n": 175}}, "A0": {"q12_tool_calls": {"k": 0, "n": 175}}},
    ),
    "p11-0-to-11of175-fails": (
        _put("C3", NEGATIVE_SLOTS[:11], Q12_TOOL), {"P11"}, "not confirmed", None,
        {"C3": {"q12_tool_calls": {"k": 11, "n": 175}}},
    ),
    # P9 verbatim: <= A0 + 1 of the 80 where none belongs, <= A0 + 2 of the 200 tool trials (a scaled P9 passes 2/80)
    "p9-no-belong-0-to-1of80-passes": (
        _put("C3", NO_BELONG_SLOTS[:1], WITH_SURFACE), set(), "confirmed", None,
        {"C3": {"surfaces_no_belong": {"k": 1, "n": 80}}},
    ),
    "p9-no-belong-0-to-2of80-fails": (
        _put("C3", NO_BELONG_SLOTS[:2], WITH_SURFACE), {"P9"}, "not confirmed", None,
        {"C3": {"surfaces_no_belong": {"k": 2, "n": 80}}},
    ),
    "p9-q12-negative-counts-among-the-80": (
        _put("C3", [("capital_of_france", 0), ("spider_legs", 1)], WITH_SURFACE), {"P9"}, "not confirmed", None,
        {"C3": {"surfaces_no_belong": {"k": 2, "n": 80}}},
    ),
    "p9-tool-0-to-2of200-passes": (
        _put("C3", TOOL_SLOTS[:2], WITH_SURFACE), set(), "confirmed", None,
        {"C3": {"surfaces_tool": {"k": 2, "n": 200}}},
    ),
    "p9-tool-0-to-3of200-fails": (
        _put("C3", TOOL_SLOTS[:3], WITH_SURFACE), {"P9"}, "not confirmed", None,
        {"C3": {"surfaces_tool": {"k": 3, "n": 200}}},
    ),
    # R3 (control): A0 positive shown >= 112/160 (the count 14 holds at 111)
    "r3-112of160-holds": (_p8((28, 28, 28, 28), (30, 30, 30, 30)), set(), "confirmed", None, {}),
    "r3-111of160-fails-inconclusive-though-c3-passes-every-p": (
        _p8((28, 28, 28, 27), (30, 30, 30, 30)), set(), "inconclusive", "UI baseline not reproduced",
        {"A0": {"q12_positive_shown": {"k": 111, "n": 160}}},
    ),
    # R1 (control): A0 type_at spoken <= 3/10 of its 50 routed trials
    "r1-15of50-holds": (lambda plan: type_silent(plan, "A0", 15), set(), "confirmed", None, {}),
    "r1-16of50-fails-inconclusive": (
        lambda plan: type_silent(plan, "A0", 16), set(), "inconclusive", "finding not reproduced", {}
    ),
    # void: > 2 % of 440 errored, i.e. >= 9 (over 300, 8 would already be void); errors on plain trials
    "void-a0-8of440-not-void": (
        _put("A0", PLAIN_SLOTS[:8], ERROR_C), set(), "confirmed", None,
        {"A0": {"errored": {"k": 8, "n": 440}}},
    ),
    "void-a0-9of440-inconclusive": (
        _put("A0", PLAIN_SLOTS[:9], ERROR_C), set(), "inconclusive", "void",
        {"A0": {"errored": {"k": 9, "n": 440}}},
    ),
    "void-c3-9of440-inconclusive": (
        _put("C3", PLAIN_SLOTS[:9], ERROR_C), set(), "inconclusive", "void",
        {"C3": {"errored": {"k": 9, "n": 440}}},
    ),
    # adversarial (iv): extra-repeat Q12 trials stay out of P5's 65 and P7's tool trials; P6 counts every trial
    "iv-q12-calls-at-r5-7-stay-out-of-p5": (
        _put("C3", [("primes_table", r) for r in (5, 6, 7)], Q12_TOOL), set(), "confirmed", None,
        {"C3": {"false_tools": {"k": 0, "n": 65}, "q12_tool_calls": {"k": 3, "n": 175},
                "q12_positive_shown": {"k": 117, "n": 160}}},
    ),
    "iv-open-thoughts-at-r5-9-stay-out-of-p7": (
        _put("C3", [("flight_booking", r) for r in range(5, 10)], OPEN_SURFACE), set(), "confirmed", None,
        {"C3": {"placement_failures_q12_positive": {"k": 5, "n": 160}}},
    ),
    "p6-anomaly-on-an-extra-repeat-trial-fails": (
        _put("C3", [("omelette_steps", 39)], ANOMALY), {"P6"}, "not confirmed", None, {}
    ),
}  # fmt: skip


@pytest.mark.parametrize("row", list(BOUNDARY_ROWS))
def test_t0103_rule_boundaries(tmp_path: Path, row: str) -> None:
    """Criterion 6, every row on hand-counted trial fixtures (module docstring's baseline, one condition moved):
    C3's failing Ps are exactly the expected set (every other condition holds), the outcome has the expected form
    (a failing P is named in it), ``winner`` is ``"C3"`` iff confirmed and ``"no change"`` otherwise, and the per-arm
    numbers the row moves read as counted by hand. Adversarial rows (i)-(iv) are named in the ids."""
    edit, want_failing, want_kind, reason, premises = BOUNDARY_ROWS[row]
    report = report_103(tmp_path, edit)
    for arm, keys in premises.items():
        for key, value in keys.items():
            assert report["arms"][arm][key] == value, (row, arm, key, report["arms"][arm][key])
    assert failing(report) == want_failing, f"{row}: C3 fails {sorted(failing(report))}"
    assert kind_of(report) == want_kind, f"{row}: outcome {report['outcome']!r}"
    if want_kind == "not confirmed":
        assert named_ps(report) == want_failing, f"{row}: outcome {report['outcome']!r}"
    if reason is not None:
        assert reason in str(report["outcome"]), f"{row}: outcome {report['outcome']!r}"
    if reason == "void":
        arm = "A0" if row.startswith("void-a0") else "C3"
        assert list(report["void_arms"]) == [arm] and arm in str(report["outcome"])
    else:
        assert list(report["void_arms"]) == []
    assert report["winner"] == ("C3" if want_kind == "confirmed" else "no change")


def test_t0103_denominators_ignore_the_extra_repeats(tmp_path: Path) -> None:
    """Adversarial (iv): on the baseline the per-arm denominators are *Design*'s -- 440 trials; tool trials 200
    (``type_at`` 50; click + scroll + key spoken 100 of 100 for C3, A0 likewise); false tools over 65; surfaces where
    none belongs over 80 and on tool trials over 200; Q12 positives 160, negatives 15, Q12 calls 175; placement over
    160; 40 per positive case and 5 per negative. A0 ``type_at`` 0/50, C3 50/50."""
    report = report_103(tmp_path)
    for arm in ARMS_103:
        a = report["arms"][arm]
        assert a["trials"] == PER_ARM and a["errored"] == {"k": 0, "n": PER_ARM}
        assert a["route_recall"] == {"k": 200, "n": 200}
        assert a["tools"]["type_at"]["routed"] == {"k": 50, "n": 50}
        assert a["csk_spoken"] == {"k": 100, "n": 100}
        assert a["false_tools"] == {"k": 0, "n": 65}
        assert a["thought_open"] == {"k": 0, "n": 200}
        assert a["surfaces_no_belong"] == {"k": 0, "n": 80}
        assert a["surfaces_tool"] == {"k": 0, "n": 200}
        assert a["q12_positive_shown"] == {"k": 120, "n": 160}
        assert a["q12_negative_shown"] == {"k": 0, "n": 15}
        assert a["q12_tool_calls"] == {"k": 0, "n": 175}
        assert a["placement_failures_q12_positive"] == {"k": 0, "n": 160}
        for cid in Q12_POSITIVE:
            assert a["q12_per_case"][cid] == {"k": 30, "n": 40}, (arm, cid)
        for cid in Q12_NEGATIVE:
            assert a["q12_per_case"][cid] == {"k": 0, "n": 5}, (arm, cid)
    assert report["arms"]["A0"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert report["arms"]["C3"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}


@pytest.mark.parametrize(
    ("case", "positive_repeats", "drop", "complete"),
    [
        ("880-lines-complete", 40, (), True),
        ("600-lines-positives-at-5-incomplete", 5, (), False),
        ("879-lines-one-missing-incomplete", 40, (("C3", "plain-03", 4),), False),
        ("878-lines-one-positive-repeat-missing-incomplete", 40, (("A0", "primes_table", 39), ("C3", "primes_table", 39)), False),
    ],
)  # fmt: skip
def test_t0103_completeness(
    tmp_path: Path, case: str, positive_repeats: int, drop: tuple[Any, ...], complete: bool
) -> None:
    """Adversarial (v): the 880-line run is complete; a two-arm run with the positives at 5 repeats (60 x 5 x 2 = 600
    lines, the set an ``expected`` built from ``REPEATS = 5`` would accept) is incomplete, and so are runs missing one
    line or the last positive repeat. An incomplete run is inconclusive and selects nothing. The 600-line run also
    fails R3 (A0 shows at most 20 < 112), so its reason is not pinned; the 879- and 878-line runs fail nothing else,
    so their reason must say ``incomplete``."""
    report = report_103(tmp_path, positive_repeats=positive_repeats, drop=drop)
    assert report["complete"] is complete, case
    if complete:
        assert report["trials"] == N_REQUESTS and kind_of(report) == "confirmed"
        return
    assert kind_of(report) == "inconclusive", report["outcome"]
    assert report["winner"] == "no change"
    if positive_repeats == POSITIVE_REPEATS:
        assert report["readings"] == {"R1": True, "R3": True}, "premise: only completeness fails"
        assert "incomplete" in str(report["outcome"]), report["outcome"]


@pytest.mark.parametrize(
    "case",
    [
        "confirmed",
        "not-confirmed-names-every-failing-p",
        "inconclusive-r3-beats-a-failing-p",
        "inconclusive-r1-beats-a-failing-p",
        "inconclusive-void-a0",
        "inconclusive-void-c3",
    ],
)
def test_t0103_outcomes(tmp_path: Path, case: str) -> None:
    """Criterion 5's outcomes and criterion 7's ``winner``. Confirmed on the baseline (readings exactly R1 and R3,
    candidates exactly C3, rule H-T103, table T-0103). Not confirmed with C3 silent on ``type_at`` (P1: 0/50), at
    103/160 (P8) and with 11 Q12 calls (P11): the outcome names P1, P8 and P11 and no other. R3 or R1 failing with P8
    also failing is inconclusive, not "not confirmed". A void A0 or void C3 (9 of 440) is inconclusive. ``winner`` is
    ``"C3"`` only when confirmed. (Incomplete: ``test_t0103_completeness``.)"""
    edits: dict[str, Callable[[Plan], None]] = {
        "confirmed": lambda plan: None,
        "not-confirmed-names-every-failing-p": _seq(
            lambda plan: type_silent(plan, "C3", 0),
            _p8((30, 30, 30, 30), (26, 26, 26, 25)),
            _put("C3", NEGATIVE_SLOTS[:11], Q12_TOOL),
        ),
        "inconclusive-r3-beats-a-failing-p": _p8((28, 28, 28, 27), (24, 24, 24, 23)),
        "inconclusive-r1-beats-a-failing-p": _seq(
            lambda plan: type_silent(plan, "A0", 16), _p8((30, 30, 30, 30), (26, 26, 26, 25))
        ),
        "inconclusive-void-a0": _put("A0", PLAIN_SLOTS[:9], ERROR_C),
        "inconclusive-void-c3": _put("C3", PLAIN_SLOTS[:9], ERROR_C),
    }
    report = report_103(tmp_path, edits[case])
    if case == "confirmed":
        assert str(report["outcome"]).startswith("C3 confirmed") and kind_of(report) == "confirmed"
        assert set(report["readings"]) == {"R1", "R3"}, "criterion 1: readings R1 and R3 only"
        assert report["readings"] == {"R1": True, "R3": True}
        assert set(report["candidates"]) == {"C3"}
        assert report.get("table") == TABLE and report.get("rule") == RULE
        assert report["winner"] == "C3"
        return
    if case == "not-confirmed-names-every-failing-p":
        assert failing(report) == {"P1", "P8", "P11"}
        assert kind_of(report) == "not confirmed" and named_ps(report) == {"P1", "P8", "P11"}, (
            report["outcome"]
        )
    elif case == "inconclusive-r3-beats-a-failing-p":
        assert "P8" in failing(report) and report["readings"]["R3"] is False
        assert kind_of(report) == "inconclusive" and "UI baseline not reproduced" in str(
            report["outcome"]
        )
    elif case == "inconclusive-r1-beats-a-failing-p":
        assert "P8" in failing(report) and report["readings"]["R1"] is False
        assert kind_of(report) == "inconclusive" and "finding not reproduced" in str(
            report["outcome"]
        )
    else:
        arm = "A0" if case.endswith("a0") else "C3"
        assert list(report["void_arms"]) == [arm]
        assert (
            kind_of(report) == "inconclusive"
            and "void" in str(report["outcome"])
            and arm in str(report["outcome"])
        )
    assert report["winner"] == "no change"


def _find(value: Any, want: Any) -> bool:
    if value == want:
        return True
    if isinstance(value, Mapping):
        return any(_find(v, want) for v in value.values())
    if isinstance(value, list):
        return any(_find(v, want) for v in value)
    return False


def _protected_flags(value: Any) -> list[dict[str, bool]]:
    """Every mapping in the report that holds, for all four positive ids, a bool or a mapping with a bool under a key
    containing ``protect`` (found by name, never called)."""
    found: list[dict[str, bool]] = []
    if isinstance(value, Mapping):
        if all(cid in value for cid in Q12_POSITIVE):
            flags: dict[str, bool] = {}
            for cid in Q12_POSITIVE:
                v = value[cid]
                if isinstance(v, bool):
                    flags[cid] = v
                elif isinstance(v, Mapping):
                    hits = [x for k, x in v.items() if "protect" in str(k) and isinstance(x, bool)]
                    if len(hits) == 1:
                        flags[cid] = hits[0]
            if len(flags) == len(Q12_POSITIVE):
                found.append(flags)
        for v in value.values():
            found.extend(_protected_flags(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_protected_flags(v))
    return found


def test_t0103_report_extras(tmp_path: Path) -> None:
    """Criterion 7 on a fixture where the positives show on the *last* k repeats: A0 24, 23, 40, 40 of 40 (127/160),
    C3 30 each (120/160). By hand: per positive case k/40 for each arm; protected = A0 >= 24: flight_booking yes,
    flight_booking_with_memory no (23), primes_table and omelette_steps yes; the block split -- A0 r 0-4 shows only on
    the two 40/40 cases, 10/20, and r 5-39 117/140; C3 0/20 and 120/140 (reported, not gated: C3 passes P8 at
    120 >= 111 with every protected case >= 24). ``report.md`` is rendered from the report and carries the outcome."""

    def edit(plan: Plan) -> None:
        set_q12(plan, "A0", q4(24, 23, 40, 40), last=True)
        set_q12(plan, "C3", q4(30, 30, 30, 30), last=True)

    report = report_103(tmp_path, edit)
    a0, c3 = report["arms"]["A0"], report["arms"]["C3"]
    assert {cid: a0["q12_per_case"][cid] for cid in Q12_POSITIVE} == {
        "flight_booking": {"k": 24, "n": 40},
        "flight_booking_with_memory": {"k": 23, "n": 40},
        "primes_table": {"k": 40, "n": 40},
        "omelette_steps": {"k": 40, "n": 40},
    }
    assert all(c3["q12_per_case"][cid] == {"k": 30, "n": 40} for cid in Q12_POSITIVE)
    want_protected = {
        "flight_booking": True,
        "flight_booking_with_memory": False,
        "primes_table": True,
        "omelette_steps": True,
    }
    flags = _protected_flags(report)
    assert want_protected in flags, (
        f"criterion 7: no per-case protected flags as counted by hand: {flags}"
    )
    for k, n in ((10, 20), (117, 140), (0, 20), (120, 140)):
        assert _find(report, {"k": k, "n": n}), (
            f"criterion 7: the block split {k}/{n} is not in the report"
        )
    assert kind_of(report) == "confirmed" and report["winner"] == "C3"
    md = harness_103().render_markdown(report)
    assert str(report["outcome"]) in md
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
    """T-0155 criterion 8: with the pre-T-0155 patch undone, ``main --table T-0103`` sees the live Gearvis A0
    (``869f8f96...``) and exits 2 with its A0-drift refusal naming both shas (on the A0 check, or as the ``; and
    the A0 system text ...`` suffix when C3 cannot be built from a drifted A0), before any socket and without a run
    directory."""
    from tests.unit.test_persona_name import A0_T0155_SHA256

    out = tmp_path / "t0103-out"
    _, rc, said, attempts = run_main_against_live_persona(
        pre_t0155, harness_103, lambda h: h._T93, ["--table", TABLE], out, monkeypatch, capsys
    )
    assert rc == 2, f"T-0155 criterion 8: exit {rc}: {said[-400:]!r}"
    assert attempts == [], f"T-0155 criterion 8: sockets before the refusal: {attempts}"
    assert not out.exists(), "T-0155 criterion 8: the refusal wrote a run directory"
    assert f"the A0 system text has sha256 {A0_T0155_SHA256}, not {PRE_T0155_A0_SHA256}" in said, (
        f"T-0155 criterion 8: not the A0-drift refusal: {said[-400:]!r}"
    )
