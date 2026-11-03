"""T-0098c: the UI-block A/B -- a second arm table on the T-0098 harness, the 7 Q12 surface cases scored "surface
shown" by the real ``UiGate``, and the pre-registered rule P1-P11 / R1, R2c, R3 (unit; CPU; no engine).

Every row was pending until ``/implement T-0098c`` started (markers removed then). The harness exists (T-0098, done), so
these rows are red on the named assertion in ``harness_c()`` (the T-0098c names are absent) or on an assertion of
their own, never on an ImportError at collection (the harness is loaded lazily, by path, through T-0098's loader).

**The assumed surface** (``scripts/gpu/t0098_native_prompt_ab.py``; the task names the behaviour, not these names;
they are the test-author's and are stated here so the implementer can honour them or say why not):

- ``TABLES``: a mapping keyed ``"T-0098"`` and ``"T-0098c"``; a table exposes (as attributes or mapping keys)
  ``arms`` (table order), ``candidates`` (winner order) and ``a0_sha256``.
- The existing names take a keyword ``table=`` defaulting to ``"T-0098"`` (criterion 1: with no table every name
  behaves as at ``3ee1636``): ``load_cases(table=...)``, ``arm_texts(tools, table=...)``,
  ``arm_prompts(case, kit, table=...)``. ``main(argv)`` takes ``--table {T-0098,T-0098c}``.
- ``arm_order(case_index, repeat, table="T-0098") -> tuple[str, ...]``: the block's arm order (criterion 3).
- ``ui_reading(commits, request_id=...) -> Mapping``: one trial's surface reading (criterion 5) with keys
  ``shown``, ``placement_failure``, ``every_ui_valid``, ``gated``, ``emitted``, ``failed`` (the last three are the
  trial's ``ui.gate`` counts: ``ui`` commits gated, gate calls that returned a surface, calls that did not).
- ``build_report(run_dir)`` reads ``manifest["table"]`` (no key = T-0098). For a T-0098c run:
  ``report["candidates"][c][p]`` for ``c`` in ``C3 C5 C4`` and ``p`` in ``P1``..``P11`` plus ``"passes"``;
  ``report["readings"]["R1" | "R2c" | "R3"]`` (bool: the reading holds); ``report["winner"]`` (``"C3"``, ``"C5"``,
  ``"C4"`` or no change: ``None`` / ``"no change"``); ``report["void_arms"]``;
  ``report["arms"][a]["tools"]["type_at"]["spoken"] == {"k", "n"}`` (as T-0098);
  ``report["arms"][a]["q12_positive_shown"] == {"k", "n": 20}``, ``["q12_negative_shown"] == {"k", "n": 15}``,
  ``["q12_per_case"][case_id] == {"k": shown, "n": 5}`` and ``["ui_gate"] == {"gated", "emitted", "failed"}``.
  ``render_markdown`` carries the three reading labels of criterion 6 verbatim when a reading fails.

**The run directory these rows build by hand** is T-0098's (``tests/unit/test_t0098_prompt_ab.py``) plus
``request_id`` per line (the gate's ``turn_id``, criterion 5) and ``expect_surface`` on Q12 lines; the manifest
carries ``table: "T-0098c"`` and ``rule``. 60 cases x 7 arms x 5 repeats = 2100 lines, 300 per arm, in the rotated
order of criterion 3.

**The baseline run** (every decision row edits it; all numbers by hand):

- tool cases: A0 ``type_at`` SILENT on all 50 (routed, plan sentence in a closed thought) -> A0 ``type_at`` spoken
  0/50 (R1 holds, 0 <= 0.3); every other tool trial of every arm speaks (D1 ``type_at`` 50/50: R2c holds).
- no-tool and plain cases: an answer, no call, no surface, every arm.
- Q12 positives (``flight_booking``, ``flight_booking_with_memory``, ``primes_table``, ``omelette_steps``): A0 and the
  three candidates show a surface (``ui_instructions.UI_EXAMPLE`` in one ``ui`` commit) on every trial except
  ``flight_booking_with_memory`` and ``omelette_steps`` at repeat 4 (speech only): per case 5, 4, 5, 4 -> 18/20 (R3
  holds, 18 >= 14). D1, D3, D4 show on all 20 (so a P8 read against D1, a reviewer mutation, moves outcomes).
- Q12 negatives: speech only, every arm (0/15 shown).
- So each candidate passes P1-P11 (type 50/50, 10/10 cases; csk 100/100; recall 200/200; valid/exact 200/200; false
  tools 0/65; anomalies 0; thought_open 0; Q12 18/20 >= 16 with no case below A0's; surfaces 0 and 0; placement 0;
  Q12 calls 0) and the winner is **C3** (winner order C3, C5, C4).

Readings this module makes where the task is open (named in the test-author report): C5's anchor is
``"what you are showing."`` as it stands in ``U`` (after the wrap ``"speech says in plain words\\n  "``; the task's
one-line spelling occurs zero times in ``U``), C4's ``U`` is 1977 code points (the task says 1975), and the memory
case is faithful to the session, which passes no ``now_ms`` to ``PromptBuilder.full`` (session.py ``_turn_prompt``),
so its hits are dated ``at some earlier time``.

Regime: CPU, pure, hand-built data, the real ``UiGate`` and vendored A2UI v0.9.1 schema; no measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import ui_instructions
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, VllmClient
from local_astra.conductor.ui_gate import UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE, UI_INSTRUCTIONS
from tests.unit._harness_reading_memo import memoize_ui_reading
from tests.unit._pre_t0155_persona import (
    PRE_T0155_A0_SHA256,
    pre_t0155_fixture,
    run_main_against_live_persona,
)
from tests.unit._pre_t0155_persona import (
    PRE_T0155_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,  # T-0155 criterion 8: the harness's own persona
)
from tests.unit.test_t0098_prompt_ab import (
    A0_SHA256,
    NONE,
    PLAIN_QUESTIONS,
    REPO,
    SILENT,
    SPEAK,
    _body_without_system,
    _without_system,
    case_rows,
    commit,
    commits_for,
    expected_arm_texts,
    harness,
    native_call,
    tool_kit,
)

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

Q12_FILE = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
T0093_HARNESS = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"

#: H-T98c, in table order (the rotation indexes it) and winner order.
ARMS_C = ("A0", "D1", "D3", "D4", "C3", "C5", "C4")
CANDIDATES_C = ("C3", "C5", "C4")
DIAGNOSTICS_C = ("D1", "D3", "D4")
#: T-0098's table (H-T98a), typed out.
ARMS_98 = ("A0", "A1", "D1", "D2", "C2", "C1")
PS_C = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11")
REPEATS = 5

#: The task's numbers for U = UI_INSTRUCTIONS (T-0098c *Arms*).
U_CHARS = 1870
U_SHA12 = "4360995b0136"
U_BULLETS_AT = 1353
#: C5: the bullet's last words as they stand in U (wrapped after "plain words"); the edit adds " or doing".
C5_ANCHOR = "what you are showing."
C5_REPLACEMENT = "what you are showing or doing."
#: C4: the anchor and H-T98c (a)'s added line, verbatim (with the task's line break inside the sentence).
C4_ANCHOR = "an empty or decorative surface is worse than none.\n"
C4_ADDED = (
    "Every reply says at least one sentence aloud, after you have finished thinking, whatever else\n"
    "it contains.\n"
)
#: len(U) + len(C4_ADDED) = 1870 + 107, counted by hand: 93 + 1 + 12 + 1 = 107.
C4_U_CHARS = 1977
U_MAX = 2000

Q12_POSITIVE = ("flight_booking", "flight_booking_with_memory", "primes_table", "omelette_steps")
Q12_NEGATIVE = ("capital_of_france", "spider_legs", "cloud_joke")

PROSE = "Here is a table of the first five primes: 2, 3, 5, 7 and 11, with their squares."
#: A progressive canvas: the createSurface line of UI_EXAMPLE and one Text, no "root" yet. The real gate shows it
#: with ``final=False`` (the session's call) and refuses it with ``final=True`` (premise checked in the row).
NO_ROOT = (
    UI_EXAMPLE.split("\n")[0]
    + "\n"
    + '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
    + '{"id":"h","component":"Text","text":"Primes and squares","variant":"h3"}]}}'
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(table: Any, name: str) -> Any:
    if isinstance(table, Mapping):
        return table[name]
    return getattr(table, name)


def harness_c() -> ModuleType:
    """T-0098's harness (loaded by T-0098's loader), with the T-0098c names asserted. Red here until criterion 1."""
    h = harness()
    tables = getattr(h, "TABLES", None)
    assert isinstance(tables, Mapping) and {"T-0098", "T-0098c"} <= set(tables), (
        "T-0098c criterion 1: the harness has no TABLES registry keyed T-0098 / T-0098c (see this module's docstring)"
    )
    for attr in ("arm_order", "ui_reading"):
        assert callable(getattr(h, attr, None)), (
            f"T-0098c surface: the harness has no callable {attr!r}"
        )
    return h


# =====================================================================================================
# The arm texts, by hand (H-T98c (a))
# =====================================================================================================


def pieces() -> tuple[str, str, str, str, str]:
    return (
        DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        UI_INSTRUCTIONS,
        pb.NATIVE_TOOL_INSTRUCTIONS,
    )


def u_c5(u: str = UI_INSTRUCTIONS) -> str:
    return u.replace(C5_ANCHOR, C5_REPLACEMENT)


def u_c4(u: str = UI_INSTRUCTIONS) -> str:
    return u.replace(C4_ANCHOR, C4_ANCHOR + C4_ADDED)


def expected_arm_texts_c(u: str = UI_INSTRUCTIONS) -> dict[str, str]:
    """The seven system texts of T-0098c *Arms*, joined here by hand from the imported constants."""
    p, ct, ss, _, n = pieces()
    i = u.index("- Between the fences")
    return {
        "A0": "\n".join([p, ct, ss, u, n]),
        "D1": "\n".join([p, ct, ss, n]),
        "D3": "\n".join([p, ct, ss, u[i:], n]),
        "D4": "\n".join([p, ct, ss, u[: i - 1], n]),
        "C3": "\n".join([p, ct, ss, n, u]),
        "C5": "\n".join([p, ct, ss, u_c5(u), n]),
        "C4": "\n".join([p, ct, ss, u_c4(u), n]),
    }


def changed_span(old: str, new: str) -> tuple[str, str]:
    """The one differing span of two strings: (old part, new part) between their longest common prefix and suffix."""
    a = 0
    while a < min(len(old), len(new)) and old[a] == new[a]:
        a += 1
    b = 0
    while b < min(len(old), len(new)) - a and old[-1 - b] == new[-1 - b]:
        b += 1
    return old[a : len(old) - b], new[a : len(new) - b]


def test_t0098c_arm_texts_by_hand() -> None:
    """H-T98c (a): each of the seven arms equals the text joined here by hand from the imported constants, compared as
    strings. Premises (the task's literals): U is 1870 characters with sha12 ``4360995b0136``; ``- Between the
    fences`` starts at 1353 after a newline; A0 hashes to ``3605c012...``. C5 changes exactly one span of U (adds
    `` or doing``), C4 exactly one (adds the approved line) and its U is 1977 <= 2000 code points; C3 is A0's
    characters in another order; D3 + D4 are U's two halves around the newline at 1352."""
    h = harness_c()
    _, _, _, u, n = pieces()
    assert len(u) == U_CHARS and sha256(u)[:12] == U_SHA12
    i = u.index("- Between the fences")
    assert i == U_BULLETS_AT and u[i - 1] == "\n"
    assert u.count(C5_ANCHOR) == 1 and u.count(C4_ANCHOR) == 1
    want = expected_arm_texts_c()
    assert sha256(want["A0"]) == A0_SHA256
    assert len(set(want.values())) == 7, "premise: seven different texts"

    # C5 / C4: one span of U each, exactly the approved addition
    assert changed_span(u, u_c5()) == ("", " or doing")
    assert changed_span(u, u_c4()) == ("", C4_ADDED)
    assert len(u_c4()) == C4_U_CHARS <= U_MAX
    assert (
        "- Never mention the fence, the JSON or the component names aloud; speech says in plain words\n"
        "  what you are showing or doing."
    ) in u_c5()
    # C3: the same characters, another order; the tool paragraph is no longer last
    assert Counter(want["C3"]) == Counter(want["A0"]) and want["C3"] != want["A0"]
    assert want["C3"].endswith(u) and not want["C3"].endswith(n)
    # D3 / D4: the two halves
    assert want["D4"].count(u[: i - 1]) == 1 and "- Between the fences" not in want["D4"]
    assert "Your reply has two parts" not in want["D3"] and u[i:] in want["D3"]

    kit = tool_kit()
    texts = h.arm_texts(kit.tools, table="T-0098c")
    assert set(texts) == set(ARMS_C), f"H-T98c: the arms are {sorted(texts)}"
    for arm in ARMS_C:
        assert texts[arm] == want[arm], f"{arm}: the harness's text is not the hand-built one"

    cases = {c.case_id: c for c in h.load_cases(table="T-0098c")}
    for cid in ("type-01", "capital_of_france"):
        prompts = h.arm_prompts(cases[cid], kit, table="T-0098c")
        assert set(prompts) == set(ARMS_C)
        for arm in ARMS_C:
            first = prompts[arm].messages[0]
            assert first.role == "system" and first.text() == want[arm], f"{cid} {arm}"


def test_t0098_table_is_the_default_and_unchanged() -> None:
    """Criterion 1: with no table every name is T-0098's -- ``ARMS == ("A0","A1","D1","D2","C2","C1")``, candidates
    C2 then C1, A0's sha ``3605c012...``, the six T-0098 texts (T-0098's own hand-built ones), the 53 cases, and the
    rotation mod 6: case 52 repeat 4 starts at offset (52 + 4) mod 6 = 2, i.e. D1 D2 C2 C1 A0 A1; case 5 repeat 1 at
    offset 0. The registry holds both tables with their own arms and the same A0 sha."""
    h = harness_c()
    assert tuple(h.ARMS) == ARMS_98 and tuple(h.CANDIDATES) == ("C2", "C1")
    kit = tool_kit()
    texts = h.arm_texts(kit.tools)
    assert texts == expected_arm_texts() and sha256(texts["A0"]) == A0_SHA256
    assert h.arm_texts(kit.tools, table="T-0098") == texts
    assert [c.case_id for c in h.load_cases()] == [r["id"] for r in case_rows()]
    assert set(h.arm_prompts(h.load_cases()[0], kit)) == set(ARMS_98)
    assert tuple(h.arm_order(52, 4)) == ("D1", "D2", "C2", "C1", "A0", "A1")
    assert tuple(h.arm_order(5, 1)) == ARMS_98
    assert tuple(h.arm_order(52, 4, table="T-0098")) == ("D1", "D2", "C2", "C1", "A0", "A1")
    t98, t98c = h.TABLES["T-0098"], h.TABLES["T-0098c"]
    assert tuple(_get(t98, "arms")) == ARMS_98 and tuple(_get(t98, "candidates")) == ("C2", "C1")
    assert tuple(_get(t98c, "arms")) == ARMS_C and tuple(_get(t98c, "candidates")) == CANDIDATES_C
    assert _get(t98, "a0_sha256") == A0_SHA256 == _get(t98c, "a0_sha256")


def test_rotation_mod_7_and_case_order() -> None:
    """Criterion 3: T-0098's 53 cases at their indices, then the 7 Q12 cases in file order at 53-59 (each with its
    file's ``expect_surface``; the memory case carries its three captions). Rotation from offset
    ``(case_index + repeat) mod 7`` of A0 D1 D3 D4 C3 C5 C4, by hand: case 59 repeat 4 -> 63 mod 7 = 0 (table order);
    case 53 repeat 0 -> 4 (C3 C5 C4 A0 D1 D3 D4); case 10 repeat 3 -> 6 (C4 A0 D1 D3 D4 C3 C5). A mod-6 rotation
    gives offsets 3, 5 and 1 there."""
    h = harness_c()
    q12 = [json.loads(x) for x in Q12_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = h.load_cases(table="T-0098c")
    ids = [c.case_id for c in cases]
    assert ids == [r["id"] for r in case_rows()] + [r["id"] for r in q12]
    assert len(ids) == 60 and ids[53:] == [*Q12_POSITIVE, *Q12_NEGATIVE]
    for c, r in zip(cases[53:], q12, strict=True):
        assert c.kind == "q12" and c.text == r["request"]
        assert bool(c.expect_surface) is bool(r["expect_surface"])
        assert tuple(c.memory_captions) == tuple(r["memory_captions"])
    assert len(cases[54].memory_captions) == 3
    assert tuple(h.arm_order(59, 4, table="T-0098c")) == ARMS_C
    assert tuple(h.arm_order(53, 0, table="T-0098c")) == ("C3", "C5", "C4", "A0", "D1", "D3", "D4")
    assert tuple(h.arm_order(10, 3, table="T-0098c")) == ("C4", "A0", "D1", "D3", "D4", "C3", "C5")
    assert tuple(h.arm_order(0, 0, table="T-0098c")) == ARMS_C


# =====================================================================================================
# Criterion 2: the anchor refusals (adversarial: a doubled anchor is a refusal, not a double edit)
# =====================================================================================================

#: The whole last bullet, so that any anchor spelling inside it occurs twice when it is appended again.
LAST_BULLET = (
    "- Never mention the fence, the JSON or the component names aloud; speech says in plain words\n"
    "  what you are showing."
)


def patch_u(monkeypatch: pytest.MonkeyPatch, h: ModuleType, new_u: str) -> None:
    """Change U in the harness's view and the builder's: ``prompt_builder.UI_INSTRUCTIONS``,
    ``ui_instructions.UI_INSTRUCTIONS`` and any module-level name of the harness bound to the original string (found
    by identity, never called). The builder still joins exactly P, CT, SS, U, N, so only the anchor moves."""
    original = UI_INSTRUCTIONS
    monkeypatch.setattr(pb, "UI_INSTRUCTIONS", new_u)
    monkeypatch.setattr(ui_instructions, "UI_INSTRUCTIONS", new_u)
    for name, value in list(vars(h).items()):
        if isinstance(value, str) and value is original:
            monkeypatch.setattr(h, name, new_u)


ANCHOR_EDITS: dict[str, tuple[str, str]] = {
    "c5-twice": ("C5", UI_INSTRUCTIONS + "\n" + LAST_BULLET),
    "c5-zero": ("C5", UI_INSTRUCTIONS.replace(C5_ANCHOR, "what you show.")),
    "c4-twice": (
        "C4",
        UI_INSTRUCTIONS + "\nAgain: an empty or decorative surface is worse than none.\nEnd.",
    ),
    "c4-zero": (
        "C4",
        UI_INSTRUCTIONS.replace(
            C4_ANCHOR, "an empty or decorative surface is worse than nothing.\n"
        ),
    ),
}


@pytest.mark.parametrize("why", list(ANCHOR_EDITS))
def test_anchor_refusals(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    """Criteria 2 and 11 (adversarial): with U changed so that C5's or C4's anchor occurs twice, or not at all,
    ``arm_texts(tools, table="T-0098c")`` raises ``ValueError`` naming the arm -- a refusal, never a double edit or a
    silent no-op (which would send A0's text as a candidate). Premise: the edited U holds the anchor 2 or 0 times.
    Guard in the other direction: under the same patch T-0098's table still builds (its arms use no anchor). The
    ``main`` exit 2 with no socket for these edits is the integration smoke's row (it needs a clean tree)."""
    h = harness_c()
    arm, new_u = ANCHOR_EDITS[why]
    anchor = C5_ANCHOR if arm == "C5" else C4_ANCHOR
    assert new_u.count(anchor) == (2 if why.endswith("twice") else 0), "premise"
    kit = tool_kit()
    patch_u(monkeypatch, h, new_u)
    assert pb.PromptBuilder(
        DEFAULT_SYSTEM_PROMPT, tools=kit.tools
    ).system_message().text() == "\n".join(
        [
            DEFAULT_SYSTEM_PROMPT,
            pb.CLOSE_THE_THOUGHT,
            pb.SPOKEN_STYLE,
            new_u,
            pb.NATIVE_TOOL_INSTRUCTIONS,
        ]
    ), "premise: the builder sees the patched U"
    texts98 = h.arm_texts(kit.tools)
    assert set(texts98) == set(ARMS_98), "T-0098's table needs no anchor and still builds"
    with pytest.raises(ValueError) as caught:
        h.arm_texts(kit.tools, table="T-0098c")
    assert arm in str(caught.value), (
        f"criterion 2 ({why}): the refusal does not name {arm}: {caught.value}"
    )


@pytest.mark.parametrize(
    ("pad", "refused"), [(23, False), (24, True)], ids=["c4-2000-ok", "c4-2001-refused"]
)
def test_c4_block_length_boundary(monkeypatch: pytest.MonkeyPatch, pad: int, refused: bool) -> None:
    """Criterion 2, both directions: U padded with ``pad`` trailing spaces makes C4's U 1977 + pad code points.
    1977 + 23 = 2000 is accepted (and C4 is exactly the hand-built text over the padded U); 2001 is refused, naming
    C4."""
    h = harness_c()
    new_u = UI_INSTRUCTIONS + " " * pad
    assert len(u_c4(new_u)) == C4_U_CHARS + pad
    kit = tool_kit()
    patch_u(monkeypatch, h, new_u)
    if refused:
        with pytest.raises(ValueError) as caught:
            h.arm_texts(kit.tools, table="T-0098c")
        assert "C4" in str(caught.value)
    else:
        texts = h.arm_texts(kit.tools, table="T-0098c")
        assert texts["C4"] == expected_arm_texts_c(new_u)["C4"]


# =====================================================================================================
# Criterion 4: Q12 bodies are what a Session sends
# =====================================================================================================


def _t0093() -> ModuleType:
    """T-0093's harness loaded by path under this module's own name (read-only; importing it opens no socket)."""
    name = "_t0093_for_t0098c_unit"
    found = sys.modules.get(name)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(name, T0093_HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def q12_rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in Q12_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]


def _session_prompt_q12(
    tmp_path: Path, row: Mapping[str, Any], kit: Any
) -> tuple[Any, dict[str, Any]]:
    """One typed turn of a Q12 request through a real ``Session`` over the fakes (no frame; memory is T-0093's
    ``_CaseMemory`` over ``_memory_hits`` of a T-0093 ``Case`` built here from the file's captions); returns pass 1's
    prompt and the session's body. Shaped as T-0098's ``_session_prompt``."""
    import asyncio

    from tests.fakes import FakeStt
    from tests.fakes.llm import is_caption_prompt
    from tests.unit._tools_switch_support import one_typed_turn
    from tests.unit.test_session_loop import (
        POLICY,
        RecordingAudio,
        RecordingLlm,
        RecordingMachine,
        RecordingPerception,
        RecordingTts,
        Rig,
        ScriptedClock,
        Session,
        SessionConfig,
        answer,
        require_session,
    )

    require_session()
    t93 = _t0093()
    case = t93.Case(
        case_id=str(row["id"]),
        kind="q12",
        text=str(row["request"]),
        expect_surface=bool(row["expect_surface"]),
        memory_captions=tuple(row["memory_captions"]),
    )
    clock = ScriptedClock()
    llm = RecordingLlm([answer("Okay.")])
    llm.clock = clock
    turns = tmp_path / "turns.jsonl"
    machine = RecordingMachine(POLICY)
    stt, tts, audio = FakeStt(), RecordingTts(ms_per_word=200), RecordingAudio(clock)
    perception, memory = RecordingPerception([]), t93._CaseMemory(t93._memory_hits(case))
    session = Session(
        machine,
        llm,
        stt,
        tts,
        audio,
        perception,
        memory,
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns),
        tool_kit=kit,
    )
    rig = Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=stt,
        tts=tts,
        audio=audio,
        perception=perception,
        memory=memory,
        session=session,
        turns_path=turns,
    )

    async def go() -> None:
        await session.start()
        try:
            await one_typed_turn(rig, str(row["request"]))
        finally:
            await session.aclose()

    asyncio.run(go())
    passes = [
        (rid, p)
        for rid, p, _ in llm.generates
        if not is_caption_prompt(p) and not p.continue_final_message
    ]
    assert len(passes) == 1, f"premise: one pass-1 generate, got {len(passes)}"
    rid, prompt = passes[0]
    bodies = [b for b in llm.bodies if b["vllm_xargs"]["astra_request_id"] == rid]
    assert len(bodies) == 1
    return prompt, bodies[0]


@pytest.mark.parametrize("case_id", ["capital_of_france", "flight_booking_with_memory"])
def test_q12_bodies_and_fidelity(tmp_path: Path, case_id: str) -> None:
    """Criterion 4: the harness's A0 prompt for a Q12 case equals the pass-1 prompt a ``Session`` builds for that typed
    turn (no frame, so the ``[screen]`` no-image note; for ``flight_booking_with_memory`` the three captions as memory
    hits, exactly as T-0093's ``_CaseMemory`` provides them), and the real client's body equals the session's once
    the request id is the same. Every other arm differs from A0 only in ``messages[0]``. Guard: the memory case's A0
    carries all three captions and differs from ``flight_booking``'s A0 (same request, no hits)."""
    h = harness_c()
    kit = tool_kit()
    row = next(r for r in q12_rows() if r["id"] == case_id)
    session_prompt, session_body = _session_prompt_q12(tmp_path, row, kit)
    cases = {c.case_id: c for c in h.load_cases(table="T-0098c")}
    prompts = h.arm_prompts(cases[case_id], kit, table="T-0098c")
    a0 = prompts["A0"]
    assert a0 == session_prompt, f"criterion 4 ({case_id}): the A0 prompt is not the session's"
    rid = session_body["vllm_xargs"]["astra_request_id"]
    client = VllmClient()
    assert json.dumps(client.request_body(a0, rid), sort_keys=True) == json.dumps(
        session_body, sort_keys=True
    )
    user = a0.messages[-1]
    assert "ImagePart" not in [type(p).__name__ for p in user.content]
    assert pb.NO_IMAGE_STATEMENT in user.text(), "a Q12 case carries the [screen] note"
    if row["memory_captions"]:
        for caption in row["memory_captions"]:
            first_line = caption.split("\n")[0]
            assert first_line in user.text(), f"the memory case lacks the hit {first_line[:40]!r}"
        # the session passes no now_ms to PromptBuilder.full, so all three hits are dated "at some earlier time"
        # (pb.UNKNOWN_AGE); a harness that pins a clock reading here builds a prompt the session never sends
        assert user.text().count("recalled, not on screen, " + pb.UNKNOWN_AGE + ":") == 3
        plain = h.arm_prompts(cases["flight_booking"], kit, table="T-0098c")["A0"]
        assert plain != a0 and "Exercises and Problems in Calculus" not in plain.messages[-1].text()
    a0_body = client.request_body(a0, "r")
    for arm in ARMS_C[1:]:
        p = prompts[arm]
        assert _without_system(p) == _without_system(a0), (
            f"{case_id} {arm}: differs outside messages[0]"
        )
        assert p.messages[0] != a0.messages[0]
        body = client.request_body(p, "r")
        assert _body_without_system(body) == _body_without_system(a0_body)
        assert "seed" not in body


# =====================================================================================================
# Criteria 5 and 11: one trial's surface reading, with the real UiGate
# =====================================================================================================


def test_surface_shown_rows() -> None:
    """Criterion 5 and 11 on hand-built commit lists; the UI text is ``ui_instructions.UI_EXAMPLE`` (which the real
    ``UiGate`` accepts with zero repairs), prose inside a fence (which it cannot recover), and a progressive canvas
    without ``root`` (shown under the session's ``final=False``, refused under ``final=True``). By hand:

    1. a surface in the reply: shown, no placement failure, gated 1 / emitted 1 / failed 0;
    2. a surface built only inside an unclosed thought (a ``ui`` commit flagged ``thought_open``): **shown** (the
       session gates and publishes it whatever ``thought_open`` says) **and** a placement failure;
    3. the T-0055 defect: the fenced surface inside a closed thought, no ``ui`` commit: not shown, nothing gated, a
       placement failure (a thought commit containing ``UI_OPEN``);
    4. prose inside the fence: gated, not shown (1 / 0 / 1), every-ui-valid false;
    5. a shown canvas then a failing one: still shown (never retracted), 2 / 1 / 1, every-ui-valid false;
    6. a failing canvas then a shown one: shown (any, not first-write-wins);
    7. the root-less progressive canvas: shown (``final=False``);
    8. speech only: not shown, nothing gated; an open thought alone is not a placement failure.
    """
    h = harness_c()
    # premises on the real gate, as the session calls it
    assert UiGate().gate(UI_EXAMPLE, turn_id="p", canvas_idx=0, final=False).surface is not None
    assert UiGate().gate(PROSE, turn_id="p", canvas_idx=0, final=False).surface is None
    assert UiGate().gate(NO_ROOT, turn_id="p", canvas_idx=0, final=False).surface is not None
    assert UiGate().gate(NO_ROOT, turn_id="p", canvas_idx=0, final=True).surface is None

    fenced = "I will build the form. " + UI_OPEN + UI_EXAMPLE + UI_CLOSE
    rows: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        (
            "surface in the reply",
            [commit("thought", "OK."), commit("answer", "Here it is."), commit("ui", UI_EXAMPLE)],
            {
                "shown": True,
                "placement_failure": False,
                "every_ui_valid": True,
                "gated": 1,
                "emitted": 1,
                "failed": 0,
            },
        ),
        (
            "surface only inside an unclosed thought",
            [
                commit("thought", "I will build it.", open_=True),
                commit("ui", UI_EXAMPLE, open_=True),
            ],
            {"shown": True, "placement_failure": True, "gated": 1, "emitted": 1, "failed": 0},
        ),
        (
            "fenced surface inside a closed thought (T-0055)",
            [commit("thought", fenced), commit("answer", "I have created an interface for you.")],
            {"shown": False, "placement_failure": True, "gated": 0, "emitted": 0, "failed": 0},
        ),
        (
            "prose inside the fence",
            [commit("thought", "OK."), commit("answer", "Here it is."), commit("ui", PROSE)],
            {
                "shown": False,
                "placement_failure": False,
                "every_ui_valid": False,
                "gated": 1,
                "emitted": 0,
                "failed": 1,
            },
        ),
        (
            "shown, then a failing canvas",
            [
                commit("thought", "OK."),
                commit("answer", "Here it is."),
                commit("ui", UI_EXAMPLE, idx=0),
                commit("ui", PROSE, idx=1),
            ],
            {"shown": True, "every_ui_valid": False, "gated": 2, "emitted": 1, "failed": 1},
        ),
        (
            "a failing canvas, then a shown one",
            [commit("thought", "OK."), commit("ui", PROSE, idx=0), commit("ui", UI_EXAMPLE, idx=1)],
            {"shown": True, "gated": 2, "emitted": 1, "failed": 1},
        ),
        (
            "a progressive canvas without root",
            [commit("thought", "OK."), commit("answer", "Here it is."), commit("ui", NO_ROOT)],
            {"shown": True, "placement_failure": False, "gated": 1, "emitted": 1},
        ),
        (
            "speech only, after an open thought",
            [commit("thought", "Hmm.", open_=True), commit("answer", "Paris.")],
            {"shown": False, "placement_failure": False, "gated": 0},
        ),
    ]
    for name, commits, want in rows:
        got = h.ui_reading(commits, request_id="t0098c-r0-case-A0")
        for key, value in want.items():
            assert got[key] == value, (
                f"criterion 5/11 ({name}): {key} is {got[key]!r}, not {value!r}"
            )


# =====================================================================================================
# Criteria 6 and 7: the report and the rule, on hand-built runs
# =====================================================================================================

SHOWN, OPEN_SURFACE, FENCE_THOUGHT, PROSE_UI, SHOWN_FAIL, Q12_TOOL, WITH_SURFACE, ERROR_C = (
    "shown",
    "open_surface",
    "fence_thought",
    "prose_ui",
    "shown_fail",
    "q12_tool",
    "with_surface",
    "error_c",
)

Plan = dict[tuple[str, str, int], str]


def case_rows_c() -> list[dict[str, Any]]:
    """The 60 cases in run order: T-0098's 53 (from the pinned file and J10) then the 7 Q12 rows in file order."""
    q12 = [
        {
            "id": r["id"],
            "kind": "q12",
            "user": r["request"],
            "gt": None,
            "tool": None,
            "reference_call": None,
            "expect_surface": bool(r["expect_surface"]),
        }
        for r in q12_rows()
    ]
    return [*case_rows(), *q12]


def commits_for_c(outcome: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if outcome == SHOWN:
        return [commit("thought", "OK."), commit("answer", "Here it is."), commit("ui", UI_EXAMPLE)]
    if outcome == OPEN_SURFACE:
        return [
            commit("thought", "I will build it.", open_=True),
            commit("ui", UI_EXAMPLE, open_=True),
        ]
    if outcome == FENCE_THOUGHT:
        return [
            commit("thought", "I will build it. " + UI_OPEN + UI_EXAMPLE + UI_CLOSE),
            commit("answer", "I have created an interface for you."),
        ]
    if outcome == PROSE_UI:
        return [commit("thought", "OK."), commit("answer", "Here it is."), commit("ui", PROSE)]
    if outcome == SHOWN_FAIL:
        return [
            commit("thought", "OK."),
            commit("answer", "Here it is."),
            commit("ui", UI_EXAMPLE, idx=0),
            commit("ui", PROSE, idx=1),
        ]
    if outcome == Q12_TOOL:
        call = native_call("google_search", {"query": "something"})
        return [
            commit("thought", "OK."),
            commit("answer", "Let me look that up."),
            commit("tool", call),
        ]
    if outcome == WITH_SURFACE:
        # the trial's usual commits (SPEAK on a tool case, an answer otherwise) plus a surface before any call
        base = commits_for(SPEAK if row["kind"] == "tool" else NONE, row)
        return [*base[:2], commit("ui", UI_EXAMPLE), *base[2:]]
    if outcome == ERROR_C:
        return []
    return commits_for(outcome, row)


def baseline_c() -> Plan:
    plan: Plan = {}
    for arm in ARMS_C:
        for row in case_rows_c():
            for r in range(REPEATS):
                if row["kind"] == "tool":
                    o = SILENT if (arm == "A0" and row["tool"] == "type_at") else SPEAK
                elif row["kind"] == "q12" and row["expect_surface"]:
                    if arm in DIAGNOSTICS_C:
                        o = SHOWN
                    elif row["id"] in ("flight_booking_with_memory", "omelette_steps") and r == 4:
                        o = NONE
                    else:
                        o = SHOWN
                else:
                    o = NONE
                plan[(arm, row["id"], r)] = o
    return plan


def write_run_c(run_dir: Path, plan: Plan) -> Path:
    texts = expected_arm_texts_c()
    rows = case_rows_c()
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in range(REPEATS):
        for i, row in enumerate(rows):
            off = (i + r) % 7
            for slot in range(7):
                arm = ARMS_C[(off + slot) % 7]
                outcome = plan[(arm, row["id"], r)]
                err = outcome == ERROR_C
                line = {
                    "case_id": row["id"],
                    "kind": row["kind"],
                    "tool": row.get("tool"),
                    "gt": row.get("gt"),
                    "case_index": i,
                    "arm": arm,
                    "repeat": r,
                    "order": slot,
                    "request_id": f"t0098c-r{r}-{row['id']}-{arm}",
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
    assert len(lines) == 2100
    (run_dir / "trials.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "table": "T-0098c",
        "rule": "H-T98c",
        "regime": {
            "profile": "lean",
            "split": "dev",
            "repeats": REPEATS,
            "seed": None,
            "concurrency": 1,
            "tools": "native:bbfc53ccdd80",
        },
        "arms": {a: {"text": t, "sha256": sha256(t)} for a, t in texts.items()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return run_dir


def report_c(tmp_path: Path, edit: Callable[[Plan], None] | None = None) -> dict[str, Any]:
    plan = baseline_c()
    if edit is not None:
        edit(plan)
    h = harness_c()
    memoize_ui_reading(h)  # T-0110 part A: this row's harness object only
    report = h.build_report(write_run_c(tmp_path / "run", plan))
    assert isinstance(report, dict)
    return report


def no_change(report: Mapping[str, Any]) -> bool:
    return report["winner"] in (None, "no change")


def failing(report: Mapping[str, Any], c: str) -> list[str]:
    return [p for p in PS_C if report["candidates"][c][p] is not True]


def q12_shown(plan: Plan, arm: str, per_case: Mapping[str, int]) -> None:
    """``per_case[cid]`` of the case's 5 trials show a surface (repeats 0..k-1); the rest answer in speech only."""
    for cid, k in per_case.items():
        for r in range(REPEATS):
            plan[(arm, cid, r)] = SHOWN if r < k else NONE


def type_silent(plan: Plan, arm: str, spoken: int = 0) -> None:
    """``arm``'s first ``spoken`` ``type_at`` trials (type-01 repeat 0, 1, ... then type-02 ...) speak; the rest are
    silent (routed)."""
    left = spoken
    for i in range(1, 11):
        for r in range(REPEATS):
            plan[(arm, f"type-{i:02d}", r)] = SPEAK if left > 0 else SILENT
            left -= 1


def test_baseline_selects_c3_and_counts_surfaces(tmp_path: Path) -> None:
    """The baseline (module docstring) with two A0 trials edited to exercise the gate counts:
    ``flight_booking`` repeat 0 is a shown canvas then a failing one, ``flight_booking_with_memory`` repeat 4 (speech
    only before) is prose inside a fence. By hand, A0: Q12 positive shown 18/20 (5, 4, 5, 4 per case), negative 0/15;
    ``ui.gate`` gated 17 + 2 + 1 = 20, emitted 17 + 1 = 18, failed 1 + 1 = 2. D1 shows 20/20 and gates 20/20/0.
    Readings R1, R2c, R3 hold, every candidate passes P1-P11, the winner is C3, no arm is void, the table is
    T-0098c's (candidates C3, C5, C4 only)."""

    def edit(plan: Plan) -> None:
        plan[("A0", "flight_booking", 0)] = SHOWN_FAIL
        plan[("A0", "flight_booking_with_memory", 4)] = PROSE_UI

    report = report_c(tmp_path, edit)
    assert set(report["candidates"]) == set(CANDIDATES_C), "N7: only C3, C5, C4 are candidates"
    for c in CANDIDATES_C:
        assert failing(report, c) == [] and report["candidates"][c]["passes"] is True, (
            c,
            failing(report, c),
        )
    assert (
        report["readings"]["R1"] is True
        and report["readings"]["R2c"] is True
        and report["readings"]["R3"] is True
    )
    assert report["winner"] == "C3" and list(report["void_arms"]) == []
    a0, d1 = report["arms"]["A0"], report["arms"]["D1"]
    assert a0["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert d1["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    assert a0["q12_positive_shown"] == {"k": 18, "n": 20}
    assert a0["q12_negative_shown"] == {"k": 0, "n": 15}
    assert {cid: a0["q12_per_case"][cid] for cid in Q12_POSITIVE} == {
        "flight_booking": {"k": 5, "n": 5},
        "flight_booking_with_memory": {"k": 4, "n": 5},
        "primes_table": {"k": 5, "n": 5},
        "omelette_steps": {"k": 4, "n": 5},
    }
    assert a0["ui_gate"] == {"gated": 20, "emitted": 18, "failed": 2}
    assert d1["q12_positive_shown"] == {"k": 20, "n": 20}
    assert d1["ui_gate"] == {"gated": 20, "emitted": 20, "failed": 0}


def test_every_ui_valid_is_reported_not_gated(tmp_path: Path) -> None:
    """Criterion 5: every-ui-valid is reported, not gated. C3's 18 shown Q12 trials each add a failing second canvas
    (shown, then prose): shown stays 18/20, so P8 holds and C3 still wins; its gate counts read gated 36, emitted 18,
    failed 18. A rule that gates on every canvas being valid (or reads shown as the last canvas) selects C5."""

    def edit(plan: Plan) -> None:
        for key, o in list(plan.items()):
            if key[0] == "C3" and o == SHOWN:
                plan[key] = SHOWN_FAIL

    report = report_c(tmp_path, edit)
    assert report["arms"]["C3"]["q12_positive_shown"] == {"k": 18, "n": 20}
    assert report["arms"]["C3"]["ui_gate"] == {"gated": 36, "emitted": 18, "failed": 18}
    assert report["candidates"]["C3"]["P8"] is True and report["winner"] == "C3"


@pytest.mark.parametrize(
    ("per_case", "p8"),
    [
        (
            {
                "flight_booking": 4,
                "flight_booking_with_memory": 4,
                "primes_table": 4,
                "omelette_steps": 4,
            },
            True,
        ),
        (
            {
                "flight_booking": 4,
                "flight_booking_with_memory": 4,
                "primes_table": 4,
                "omelette_steps": 3,
            },
            False,
        ),
    ],
    ids=["16of20-passes", "15of20-fails"],
)
def test_p8_total_boundary(tmp_path: Path, per_case: dict[str, int], p8: bool) -> None:
    """P8's total, both directions, with ``type_at`` 50/50 (criterion 11: a candidate that restores speech but loses
    Q12 uptake). A0 18/20; C3 at 16/20 = A0 - 2 passes, 15/20 = A0 - 3 fails; every case stays >= 3/5 in both, so only
    the total moves. D1 shows 20/20, so a P8 read against D1 (reviewer mutation) fails the 16/20 row. Winner C3 or C5."""
    report = report_c(tmp_path, lambda plan: q12_shown(plan, "C3", per_case))
    assert report["arms"]["C3"]["q12_positive_shown"] == {"k": sum(per_case.values()), "n": 20}
    assert report["arms"]["C3"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    assert report["candidates"]["C3"]["P8"] is p8
    assert [p for p in failing(report, "C3") if p != "P8"] == []
    assert report["winner"] == ("C3" if p8 else "C5")


@pytest.mark.parametrize(
    ("a0", "c3", "p8"),
    [
        # A0 5,4,5,4 (18): C3 5,2,5,4 = 16 (total ok) but flight_booking_with_memory 4/5 -> 2/5: fails
        ((5, 4, 5, 4), (5, 2, 5, 4), False),
        # the same total with every A0-majority case >= 3/5: passes
        ((5, 4, 5, 4), (5, 3, 4, 4), True),
        # the clause's own boundary: A0 omelette exactly 3/5 is a majority case; C3 2/5 fails (total 16 >= 17 - 2)
        ((5, 4, 5, 3), (5, 4, 5, 2), False),
        # too wide: a case A0 shows in only 2/5 may fall to 0/5 (total 14 >= 16 - 2): passes
        ((5, 4, 5, 2), (5, 4, 5, 0), True),
    ],
    ids=[
        "majority-case-to-2of5-fails",
        "all-majority-cases-hold",
        "a0-at-3of5-is-majority",
        "a0-minority-may-fall",
    ],
)
def test_p8_per_case_clause(
    tmp_path: Path, a0: tuple[int, ...], c3: tuple[int, ...], p8: bool
) -> None:
    """P8's per-case clause alone (criterion 11): "no positive case A0 shows in >= 3/5 falls below 3/5". Every row
    keeps C3 within P8's total (A0 - 2), so only the clause decides; R3 holds in each (A0 >= 16 >= 14). A rule
    without the clause (reviewer mutation) passes rows 1 and 3; a clause applied to every case fails row 4."""

    def edit(plan: Plan) -> None:
        q12_shown(plan, "A0", dict(zip(Q12_POSITIVE, a0, strict=True)))
        q12_shown(plan, "C3", dict(zip(Q12_POSITIVE, c3, strict=True)))

    report = report_c(tmp_path, edit)
    assert sum(c3) >= sum(a0) - 2 and sum(a0) >= 14, "premise: the total holds"
    assert report["readings"]["R3"] is True
    assert report["candidates"]["C3"]["P8"] is p8
    assert report["winner"] == ("C3" if p8 else "C5")


NO_BELONG_SLOTS = [("plain-01", 0), ("chat-01", 0), ("capital_of_france", 0), ("plain-02", 1)]
TOOL_SLOTS = [("click-01", 0), ("search-01", 0), ("scroll-01", 0), ("key-01", 0)]


@pytest.mark.parametrize(
    ("where", "a0_n", "c3_n", "p9"),
    [
        ("no-belong", 0, 1, True),  # a surface on a plain question: 1 <= 0 + 1
        ("no-belong", 0, 2, False),  # plain-01 and chat-01: 2 > 0 + 1
        ("no-belong", 1, 2, True),  # 2 <= 1 + 1
        ("q12-negative", 0, 2, False),  # capital_of_france and spider_legs count among the 80
        ("tool", 0, 2, True),  # 2 <= 0 + 2
        ("tool", 0, 3, False),  # 3 > 0 + 2
        ("tool", 1, 3, True),  # 3 <= 1 + 2
    ],
)
def test_p9_boundaries(tmp_path: Path, where: str, a0_n: int, c3_n: int, p9: bool) -> None:
    """P9 both clauses, both directions: surfaces on the 80 trials where none belongs (3 negative Q12 + 8 no-tool + 5
    plain cases) <= A0 + 1, and on the 200 tool trials <= A0 + 2. The surfaces are added to otherwise unchanged trials
    (a tool trial still speaks and calls), so no other P moves."""
    if where == "tool":
        slots = TOOL_SLOTS
    elif where == "q12-negative":
        slots = [("capital_of_france", 0), ("spider_legs", 0), ("cloud_joke", 0)]
    else:
        slots = NO_BELONG_SLOTS

    def edit(plan: Plan) -> None:
        for cid, r in slots[:a0_n]:
            plan[("A0", cid, r)] = WITH_SURFACE
        for cid, r in slots[:c3_n]:
            plan[("C3", cid, r)] = WITH_SURFACE

    report = report_c(tmp_path, edit)
    assert report["candidates"]["C3"]["P9"] is p9
    assert [p for p in failing(report, "C3") if p != "P9"] == []
    assert report["winner"] == ("C3" if p9 else "C5")


@pytest.mark.parametrize(
    ("a0_kinds", "c3_kinds", "p10"),
    [
        ((), (OPEN_SURFACE,), True),
        ((), (OPEN_SURFACE, OPEN_SURFACE), False),
        ((), (FENCE_THOUGHT,), True),
        ((), (FENCE_THOUGHT, FENCE_THOUGHT), False),
        ((), (OPEN_SURFACE, FENCE_THOUGHT), False),
        ((OPEN_SURFACE,), (OPEN_SURFACE, FENCE_THOUGHT), True),
    ],
    ids=["1-open", "2-open", "1-fence", "2-fence", "open+fence", "a0-has-1"],
)
def test_p10_placement_boundaries(
    tmp_path: Path, a0_kinds: tuple[str, ...], c3_kinds: tuple[str, ...], p10: bool
) -> None:
    """P10: placement failures on Q12 positive trials <= A0 + 1. A failure is a ``ui`` commit flagged ``thought_open``
    (still shown) or a thought commit holding the fence (not shown). They replace shown trials ``flight_booking``
    repeat 0 and ``primes_table`` repeat 0, so C3's shown stays >= 16/20 with every case >= 4/5 and P8 holds. A
    reading that ignores ``thought_open`` ``ui`` commits (reviewer mutation) passes the "2-open" and "open+fence"
    rows."""
    slots = [("flight_booking", 0), ("primes_table", 0)]

    def edit(plan: Plan) -> None:
        for (cid, r), kind in zip(slots, a0_kinds, strict=False):
            plan[("A0", cid, r)] = kind
        for (cid, r), kind in zip(slots, c3_kinds, strict=False):
            plan[("C3", cid, r)] = kind

    report = report_c(tmp_path, edit)
    assert report["candidates"]["C3"]["P10"] is p10
    assert [p for p in failing(report, "C3") if p != "P10"] == []
    assert report["winner"] == ("C3" if p10 else "C5")


@pytest.mark.parametrize(("a0_n", "c3_n", "p11"), [(0, 2, True), (0, 3, False), (1, 3, True)])
def test_p11_and_q12_calls_stay_out_of_p5(tmp_path: Path, a0_n: int, c3_n: int, p11: bool) -> None:
    """P11: routed tool calls on the 35 Q12 trials <= A0 + 2, both directions; the calls fall on Q12 negatives
    (``capital_of_france``, ``spider_legs``, ``cloud_joke`` at repeat 0). P5 holds in every row: its 65 are the
    no-tool and plain trials only, so Q12 calls never count there. Reviewer mutation (Q12 trials counted into P5's
    65): the second row then reads 3 > 0 + 2 and fails P5 -- caught here; rows 1 and 3 read P5 true either way."""
    slots = [("capital_of_france", 0), ("spider_legs", 0), ("cloud_joke", 0)]

    def edit(plan: Plan) -> None:
        for cid, r in slots[:a0_n]:
            plan[("A0", cid, r)] = Q12_TOOL
        for cid, r in slots[:c3_n]:
            plan[("C3", cid, r)] = Q12_TOOL

    report = report_c(tmp_path, edit)
    c3 = report["candidates"]["C3"]
    assert c3["P11"] is p11 and c3["P5"] is True
    assert [p for p in failing(report, "C3") if p != "P11"] == []
    assert report["winner"] == ("C3" if p11 else "C5")


@pytest.mark.parametrize(
    ("silent", "winner"),
    [((), "C3"), (("C3",), "C5"), (("C3", "C5"), "C4"), (("C3", "C5", "C4"), None)],
    ids=["all-pass", "c3-fails", "c3-c5-fail", "none-pass"],
)
def test_winner_order_and_diagnostics_never_win(
    tmp_path: Path, silent: tuple[str, ...], winner: str | None
) -> None:
    """Winner order C3, C5, C4 by order, not by best numbers (criterion 7): the candidates in ``silent`` speak on no
    ``type_at`` trial (P1 fails: 0/50). When all three fail, D1, D3 and D4 -- perfect on every count -- are never
    selected (N7): no change, and report.md says so."""

    def edit(plan: Plan) -> None:
        for c in silent:
            type_silent(plan, c)

    report = report_c(tmp_path, edit)
    for c in silent:
        assert report["candidates"][c]["P1"] is False
    if winner is None:
        assert no_change(report), f"N7: {report['winner']!r} was selected"
        assert "no change" in harness_c().render_markdown(report)
    else:
        assert report["winner"] == winner


@pytest.mark.parametrize(
    ("spoken", "r1"), [(15, True), (16, False)], ids=["15of50-holds", "16of50-fails"]
)
def test_r1_failing_selects_nothing(tmp_path: Path, spoken: int, r1: bool) -> None:
    """R1: A0 ``type_at`` spoken <= 0.3 over its 50 routed trials. 15/50 = 0.30 holds and C3 wins; 16/50 = 0.32 fails:
    no candidate is selected though every P passes, and report.md reads "finding not reproduced"."""
    report = report_c(tmp_path, lambda plan: type_silent(plan, "A0", spoken))
    assert report["arms"]["A0"]["tools"]["type_at"]["spoken"] == {"k": spoken, "n": 50}
    assert report["readings"]["R1"] is r1
    assert report["candidates"]["C3"]["passes"] is True
    md = harness_c().render_markdown(report)
    assert ("finding not reproduced" in md) is (not r1)
    assert (report["winner"] == "C3") if r1 else no_change(report)


@pytest.mark.parametrize(
    ("a0", "r3"), [((5, 3, 3, 3), True), ((4, 3, 3, 3), False)], ids=["14of20", "13of20"]
)
def test_r3_failing_selects_nothing_even_when_every_p_passes(
    tmp_path: Path, a0: tuple[int, ...], r3: bool
) -> None:
    """R3 (criterion 11): A0 Q12 positive shown >= 14/20. A0 at 14 holds and C3 wins; at 13 the guard cannot detect a
    regression: no candidate is selected even though every candidate passes P1-P11 (each shows 18/20, every case
    >= 4/5 >= 3/5), and report.md reads "UI baseline not reproduced: the guard cannot detect a regression"."""
    report = report_c(
        tmp_path, lambda plan: q12_shown(plan, "A0", dict(zip(Q12_POSITIVE, a0, strict=True)))
    )
    assert report["arms"]["A0"]["q12_positive_shown"] == {"k": sum(a0), "n": 20}
    assert report["readings"]["R3"] is r3
    for c in CANDIDATES_C:
        assert report["candidates"][c]["passes"] is True, (c, failing(report, c))
    md = harness_c().render_markdown(report)
    assert ("UI baseline not reproduced: the guard cannot detect a regression" in md) is (not r3)
    assert (report["winner"] == "C3") if r3 else no_change(report)


@pytest.mark.parametrize(
    ("spoken", "r2c"), [(35, True), (34, False)], ids=["35of50-holds", "34of50-fails"]
)
def test_r2c_is_a_label_only(tmp_path: Path, spoken: int, r2c: bool) -> None:
    """R2c: D1 ``type_at`` spoken >= 0.7. 35/50 holds; 34/50 = 0.68 fails and report.md carries "UI-block cause not
    reproduced" -- a label only: the winner is C3 in both rows."""
    report = report_c(tmp_path, lambda plan: type_silent(plan, "D1", spoken))
    assert report["arms"]["D1"]["tools"]["type_at"]["spoken"] == {"k": spoken, "n": 50}
    assert report["readings"]["R2c"] is r2c
    assert report["winner"] == "C3"
    md = harness_c().render_markdown(report)
    assert ("UI-block cause not reproduced" in md) is (not r2c)


@pytest.mark.parametrize(
    ("arm", "errors", "winner"),
    [("A0", 6, "C3"), ("A0", 7, None), ("C3", 6, "C3"), ("C3", 7, "C5")],
    ids=["a0-6of300", "a0-7of300-void", "c3-6of300", "c3-7of300-void"],
)
def test_void_arms(tmp_path: Path, arm: str, errors: int, winner: str | None) -> None:
    """Criterion 7: an arm with > 2 % errored trials is void (300 trials per arm: 6 = 2.0 % is not, 7 = 2.33 % is);
    a void A0 selects nothing; a void C3 cannot be selected, so C5 wins. The errors fall on plain trials, so no rate
    of a P moves."""
    plain = [cid for cid, _ in PLAIN_QUESTIONS]
    slots = [(cid, r) for r in range(REPEATS) for cid in plain][:errors]

    def edit(plan: Plan) -> None:
        for cid, r in slots:
            plan[(arm, cid, r)] = ERROR_C

    report = report_c(tmp_path, edit)
    assert (arm in report["void_arms"]) is (errors == 7)
    if winner is None:
        assert no_change(report), f"a void A0 selected {report['winner']!r}"
    else:
        assert report["winner"] == winner


# =====================================================================================================
# T-0155 criterion 8: the frozen harness refuses today's persona
# =====================================================================================================


def test_harness_refuses_live_persona(
    pre_t0155: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T-0155 criterion 8: with the pre-T-0155 patch undone, ``main --table T-0098c`` sees the live Gearvis A0
    (``869f8f96...``) and exits 2 with its A0-drift refusal naming both shas, before any socket and without a run
    directory."""
    from tests.unit.test_persona_name import A0_T0155_SHA256

    out = tmp_path / "t0098c-out"
    _, rc, said, attempts = run_main_against_live_persona(
        pre_t0155, harness_c, lambda h: h._T93, ["--table", "T-0098c"], out, monkeypatch, capsys
    )
    assert rc == 2, f"T-0155 criterion 8: exit {rc}: {said[-400:]!r}"
    assert attempts == [], f"T-0155 criterion 8: sockets before the refusal: {attempts}"
    assert not out.exists(), "T-0155 criterion 8: the refusal wrote a run directory"
    assert f"the A0 system text has sha256 {A0_T0155_SHA256}, not {PRE_T0155_A0_SHA256}" in said, (
        f"T-0155 criterion 8: not the A0-drift refusal: {said[-400:]!r}"
    )
