"""T-0184 group C, the prompt: C4-C6 and the C8 negatives (docs/tasks/T-0184.md).

**C4. The v2 block** ``ui_instructions.UI_INSTRUCTIONS_V2`` (a literal) advertises, in ``ADVERTISED_ORDER_V2``, the seven v2
kinds then Text, Column, Row, Card, List and Divider; the interactive kinds are not named (H-T184-7). It interpolates
``UI_OPEN``/``UI_CLOSE``, keeps T-0055's placement rule and the "never say the fence or JSON aloud" rule, and shows one
example under the v2 id with at least one v2 kind, which passes the real gate complete with zero repairs (the renderer
half is ``tests/contract/test_t0184_example_renders.py``). Hand-off from the group A review: the block never advertises
``accessibility`` or ``weight``.

**C5. Budget.** ``len(UI_INSTRUCTIONS_V2) <= len(UI_INSTRUCTIONS)`` (1 870 code points at BASE). Tokens: v2 <= v1's 500 and
the example <= 200, in an opt-in CPU row with the cached tokenizer (rev 2ea8372; skips naming the path when the file is
absent, and needs ``tokenizers``: ``uv run --with tokenizers==0.23.2 pytest tests/unit/test_t0184_prompt_block.py``).

**C6. Prefix cache.** ``PromptBuilder(…, ui_block=…)`` defaults to v1, and the default system message is byte-identical to
BASE for tools off, tools on and both live guidances (sha256 literals measured at BASE 3cb090f1; unmarked, green at BASE).
With v2 the message differs only inside the block's span, is the same bytes under two ``PYTHONHASHSEED``s, and the block
carries no added special token and no URL.

**C8.** ``UI_INSTRUCTIONS``, ``UI_EXAMPLE`` and ``ADVERTISED_ORDER`` are pinned by content (sha256, unmarked); the file
scope (session.py, cli.py, turn_machine.py, speech_gate.py, llm_client.py, contracts/**, tests/fakes/**, hud/**) is a
closed-range row ``BASE_C..TIP_C`` whose TIP is pinned by whoever activates group C (pending until then); no network
import is added to ``ui_instructions.py`` or ``prompt_builder.py``.

**The interface these rows assume** (the task names ``UI_INSTRUCTIONS_V2``, ``ADVERTISED_ORDER_V2`` and the ``ui_block``
keyword): the example is read *from the block* (the one ``UI_OPEN … UI_CLOSE`` span), so no name for it is assumed. The
order of the seven v2 kinds inside ``ADVERTISED_ORDER_V2`` is read as the task's listing order (Stat, StatTiles,
Comparison, Bars, Steps, Source, Actions); the tail is given by C4.

Regime: CPU; the token rows use ``tokenizer.json`` of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372 (sha256
``cc8d3a0c…``), ``tokenizers`` 0.23.2, ``add_special_tokens=False``; v1 measured 500 tokens with a 144-token example
there (2026-10-28, matching the task's figures). n = 1, deterministic.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import ui_instructions
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, split_segments
from local_astra.conductor.prompt_builder import LIVE_GUIDANCE, PromptBuilder
from local_astra.conductor.ui_gate import UiGate
from local_astra.ui import A2UIValidator, parse_concatenated_json
from local_astra.ui.a2ui_schema import load_cards_v2_catalog
from tests import _git_range as gr
from tests.unit.test_t0184_gate import _tokenizer_path
from tests.unit.test_ui_instructions import (
    ERASURE_SPELLINGS,
    SPECIAL_TOKEN_SHAPE,
    SPECIAL_TOKENS,
    URL_RE,
    matches,
    unit_stating_both,
    units,
)

REPO = Path(__file__).resolve().parents[2]
SYSTEM = "You are Gearvis."
V2_ID = "gearvis:cards/v2"
V2_KINDS = ("Stat", "StatTiles", "Comparison", "Bars", "Steps", "Source", "Actions")
#: C4, retyped: the seven v2 kinds in the task's listing order, then the six basic kinds C4 names, in its order.
EXPECTED_ORDER_V2 = (*V2_KINDS, "Text", "Column", "Row", "Card", "List", "Divider")
#: Basic kinds the v2 block must not name (H-T184-7: inputs, Tabs, Modal, Slider; and Button, Icon and the media kinds,
#: which C4's list leaves out).
NOT_ADVERTISED = (
    "Button",
    "CheckBox",
    "ChoicePicker",
    "TextField",
    "DateTimeInput",
    "Slider",
    "Tabs",
    "Modal",
    "Icon",
    "Image",
    "Video",
    "AudioPlayer",
)
#: T-0044's "never name the fence in speech" spellings (tests/unit/test_ui_instructions.py), anchored on the prohibition.
SPEECH_PROHIBITION = (
    "never mention",
    "do not mention",
    "don't mention",
    "never say",
    "do not say",
    "never speak",
    "do not speak",
    "never read",
    "do not read",
    "not in speech",
    "never in speech",
    "keep it out of speech",
)

#: BASE (main 3cb090f1) measurements, 2026-10-28: v1's length and the sha256 of v1's three literals.
V1_CODE_POINTS = 1_870
V1_TOKENS = 500
EXAMPLE_TOKENS_MAX = 200
UI_INSTRUCTIONS_SHA256 = "4360995b013627819f20de7044bcbae75edfef1f9724356eba5f94cc2b19495b"
UI_EXAMPLE_SHA256 = "3e1c2e4fc062e15e1e4554814c79e0028af763b030d4016e45b4d11fa48999af"
#: sha256 of "\n".join(ADVERTISED_ORDER)
ADVERTISED_ORDER_SHA256 = "fb071cc7a33b34ed3d741fc69a5fe97ceb7066bb4ffa1b61d09ab6894a37f499"

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {"name": "click", "parameters": {"type": "object", "properties": {}}},
    },
)
#: (config name, PromptBuilder keyword arguments, sha256 of BASE's system message text with SYSTEM). Measured at BASE.
CONFIGS: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("tools-off", {}, "fa8f6affe7cc22eab874a0f2b26dd815838420b957cbed03a4793ce22e97d394"),
    (
        "tools-on",
        {"tools": TOOLS},
        "21e13b93db297066c378684169b1e07b238ac22b395896f184c13613dc699278",
    ),
    (
        "live-guidance-v1",
        {"tools": TOOLS, "tool_guidance": LIVE_GUIDANCE["v1"]},
        "e4ec6b4bc4f596f8564510d9ca6bf8d9f2fdbfe221d016842f2f98e8bf591538",
    ),
    (
        "live-guidance-v2",
        {"tools": TOOLS, "tool_guidance": LIVE_GUIDANCE["v2"]},
        "8612df390115e1287f3480442b782d53a9da325dff60933fe0f112f75237685d",
    ),
)

#: T-0184 group C's BASE for the C8 scope range: ``git merge-base main HEAD`` after the branch merged main (096610e7),
#: i.e. main at e45f39a5 ("T-0184: name the B and C test modules"). A literal. The branch point was 3cb090f1; nothing
#: under local_astra/, hud/ or tests/fakes/ changed between the two. The BASE/BEFORE pins elsewhere in this module and in
#: test_t0184_gate.py (sha256 literals, the frozen gate) stay at 3cb090f1, whose bytes they record.
BASE_C = "e45f39a5e8d22a1f635194bd92365b1118456618"
#: The implementation tip (tests.md: a scope guard compares two literals): ca1fe748, the C9a fix (round 2), after
#: 9d2409cb (first implementation) and 17a05602 (C9/C10). Re-pin if a later implementation commit lands.
TIP_C: str | None = "ca1fe748f790ac60206201fabbe87e5dc4d355f8"
#: Power ranges known to change guarded paths (group A's): T-0178 (session.py) and T-0165 WIP (hud/src/main.js).
POWER_CONDUCTOR = (
    "5f072f41d26c7195963d04bd5d6b0486e899291e",
    "52be36b07a2429d24207e3e0d0daac67627156de",
)
POWER_HUD = ("3deb54aa09d78a41a5cdf07d3dc796c3fa1168d1", "78db2a87a5f027f46b743cd2ccc5f8979d72a0e4")
C8_PATHS = (
    "local_astra/conductor/session.py",
    "local_astra/cli.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "tests/fakes",
    "hud",
)


# --- accessors: a missing name is the criterion's failure, never a collection error ----------------------------------


def block_v2() -> str:
    value = getattr(ui_instructions, "UI_INSTRUCTIONS_V2", None)
    if not isinstance(value, str):
        raise NotImplementedError(
            "C4: local_astra.conductor.ui_instructions.UI_INSTRUCTIONS_V2 (a str) does not exist"
        )
    return value


def order_v2() -> tuple[str, ...]:
    value = getattr(ui_instructions, "ADVERTISED_ORDER_V2", None)
    if value is None:
        raise NotImplementedError(
            "C4: local_astra.conductor.ui_instructions.ADVERTISED_ORDER_V2 does not exist"
        )
    assert isinstance(value, tuple), (
        f"ADVERTISED_ORDER_V2 is a tuple (its order is prefix bytes); got {type(value)}"
    )
    return value


def builder(**kwargs: Any) -> PromptBuilder:
    try:
        return PromptBuilder(SYSTEM, **kwargs)
    except TypeError as err:
        raise AssertionError(f"C6: PromptBuilder does not take {sorted(kwargs)}: {err}") from err


def example_v2() -> str:
    text = block_v2()
    assert text.count(UI_OPEN) == 1 and text.count(UI_CLOSE) == 1, (
        "C4: the block shows exactly one fenced example"
    )
    start = text.index(UI_OPEN) + len(UI_OPEN)
    end = text.index(UI_CLOSE, start)
    return text[start:end]


def _components(payload: str) -> list[dict[str, Any]]:
    return [
        c
        for m in parse_concatenated_json(payload)
        if isinstance(m, dict) and isinstance(m.get("updateComponents"), dict)
        for c in m["updateComponents"]["components"]
    ]


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])")


# --- C4: the block -------------------------------------------------------------------------------------------------


def test_c4_advertised_order_v2_is_the_seven_then_six() -> None:
    """C4: ``ADVERTISED_ORDER_V2`` is exactly the seven v2 kinds then Text, Column, Row, Card, List, Divider; every one is
    a kind of the v2 catalog."""
    assert order_v2() == EXPECTED_ORDER_V2
    assert set(EXPECTED_ORDER_V2) <= A2UIValidator(catalog=load_cards_v2_catalog()).component_kinds


def test_c4_block_lists_the_advertised_kinds_in_order_and_names_no_other() -> None:
    """C4 / H-T184-7: one unit of the block (a sentence or bullet, hard wraps joined) names the thirteen kinds in that
    order, and no dropped basic kind (inputs, Tabs, Modal, Slider, Button, Icon, media) is named anywhere."""
    text = block_v2()
    in_order = re.compile(r".*?".join(_word(k).pattern for k in EXPECTED_ORDER_V2), re.S)
    assert any(in_order.search(u) for u in units(text)), (
        "C4: no unit lists the advertised kinds in ADVERTISED_ORDER_V2"
    )
    named = [k for k in NOT_ADVERTISED if _word(k).search(text)]
    assert named == [], f"H-T184-7: the v2 block names {named}"


def test_c4_block_never_advertises_accessibility_or_weight() -> None:
    """Hand-off from the group A review: the v2 kinds inherit ``accessibility`` and ``weight`` from ``ComponentCommon``;
    the block never mentions either, and the example never writes either key."""
    text = block_v2().lower()
    assert "accessibility" not in text and not re.search(r"\bweight\b", text)
    keys = {k for c in _components(example_v2()) for k in c}
    assert not ({"accessibility", "weight"} & keys), keys


def test_c4_block_is_a_literal_interpolating_the_fence() -> None:
    """C4 / implementation note 3: ``UI_INSTRUCTIONS_V2`` is assigned once at module level as a string literal or an
    f-string whose fields include ``UI_OPEN`` and ``UI_CLOSE`` (never retyped), and ``ADVERTISED_ORDER_V2`` as a tuple of
    string literals. Read with ``ast``."""
    block_v2()
    tree = ast.parse(
        (REPO / "local_astra" / "conductor" / "ui_instructions.py").read_text(encoding="utf-8")
    )

    def assigned(name: str) -> ast.expr:
        found = [
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign | ast.AnnAssign)
            and n.value is not None
            and any(
                getattr(t, "id", None) == name
                for t in ([n.target] if isinstance(n, ast.AnnAssign) else n.targets)
            )
        ]
        assert len(found) == 1, (
            f"{name} must be assigned exactly once at module level; found {len(found)}"
        )
        return found[0]

    value = assigned("UI_INSTRUCTIONS_V2")
    assert isinstance(value, ast.JoinedStr), (
        f"UI_INSTRUCTIONS_V2 must be an f-string literal, not {type(value).__name__}"
    )
    names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
    assert {"UI_OPEN", "UI_CLOSE"} <= names, (
        f"C4: the block interpolates {sorted(names)}, not UI_OPEN and UI_CLOSE"
    )
    literal = "".join(
        p.value for p in value.values if isinstance(p, ast.Constant) and isinstance(p.value, str)
    )
    assert UI_OPEN not in literal and UI_CLOSE not in literal, (
        "C4: the fence is retyped in the literal"
    )
    order = assigned("ADVERTISED_ORDER_V2")
    assert isinstance(order, ast.Tuple) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in order.elts
    ), "ADVERTISED_ORDER_V2 is a tuple of string literals"


def test_c4_block_keeps_the_placement_and_speech_rules() -> None:
    """C4: T-0055's placement rule (placement and write-once in one unit, plus the erasure clause) and the "never say
    the fence or JSON aloud" rule survive, by T-0044/T-0055's own spelling lists."""
    text = block_v2()
    assert unit_stating_both(text) is not None, (
        "C4: T-0055's placement rule (one unit: in the reply + written once)"
    )
    assert matches(text, ERASURE_SPELLINGS), (
        "C4: T-0055's erasure clause (what is written while thinking is lost)"
    )
    low = text.lower()
    assert any(s in low for s in SPEECH_PROHIBITION), (
        "C4: the rule never to say the fence or the JSON aloud"
    )
    rule = next(u for u in units(text) if any(s in u.lower() for s in SPEECH_PROHIBITION))
    assert re.search(r"fence|json", rule, re.I), (
        f"C4: the prohibition does not name the fence or JSON: {rule!r}"
    )


def test_c4_example_uses_the_v2_id_and_a_v2_kind_and_only_advertised_kinds() -> None:
    """C4: the one example creates its surface under ``gearvis:cards/v2``, holds at least one v2 kind, and uses only
    advertised kinds."""
    messages = parse_concatenated_json(example_v2())
    creates = [m["createSurface"] for m in messages if isinstance(m, dict) and "createSurface" in m]
    assert [c.get("catalogId") for c in creates] == [V2_ID], creates
    kinds = {c.get("component") for c in _components(example_v2())}
    assert kinds & set(V2_KINDS), f"C4: the example shows no v2 kind: {sorted(kinds)}"
    assert kinds <= set(EXPECTED_ORDER_V2), (
        f"C4: the example uses unadvertised kinds {sorted(kinds - set(EXPECTED_ORDER_V2))}"
    )


@pytest.mark.parametrize("final", [False, True])
def test_c4_example_passes_the_real_gate_complete_with_zero_repairs(final: bool) -> None:
    """C4: fenced, the example is exactly one ``ui`` piece (``split_segments``), and ``UiGate()`` turns that piece into a
    surface that is complete with ``repaired == 0`` and every component the example writes."""
    pieces = split_segments(None, UI_OPEN + example_v2() + UI_CLOSE, False)
    assert [p[0] for p in pieces] == ["ui"], pieces
    result = UiGate().gate(pieces[0][1], turn_id="ex", canvas_idx=0, final=final)
    assert result.surface is not None, f"C4: no surface: {list(result.report.problems)[:3]}"
    assert result.complete and result.repaired == 0 and result.lost == ()
    emitted = {
        c["id"]
        for m in result.surface.messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    }
    assert emitted == {c["id"] for c in _components(example_v2())}


# --- C5: budget ----------------------------------------------------------------------------------------------------


def test_c5_v2_block_is_no_longer_than_v1() -> None:
    """C5: ``len(UI_INSTRUCTIONS_V2) <= len(UI_INSTRUCTIONS)``, which is 1 870 code points (BASE, pinned by C8)."""
    assert len(ui_instructions.UI_INSTRUCTIONS) == V1_CODE_POINTS
    assert len(block_v2()) <= V1_CODE_POINTS, f"C5: the v2 block is {len(block_v2())} code points"


def test_c5_token_budget_with_the_cached_tokenizer() -> None:
    """C5, opt-in (``uv run --with tokenizers==0.23.2``): v2 <= v1's 500 tokens, the example <= 200, and (C6) no id of
    an added special token in the encoded block. Skips naming the tokenizer path when the snapshot is absent."""
    block_v2()
    path = _tokenizer_path()
    tokenizers = pytest.importorskip(
        "tokenizers", reason="opt-in: run with `uv run --with tokenizers==0.23.2`"
    )
    tok = tokenizers.Tokenizer.from_file(str(path))
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    assert count(ui_instructions.UI_INSTRUCTIONS) == V1_TOKENS, (
        "premise: v1 is 500 tokens (T-0184 Regime)"
    )
    assert count(block_v2()) <= V1_TOKENS, f"C5: v2 is {count(block_v2())} tokens"
    assert count(example_v2()) <= EXAMPLE_TOKENS_MAX, (
        f"C5: the example is {count(example_v2())} tokens"
    )
    special = {k for k, v in tok.get_added_tokens_decoder().items() if v.special}
    assert len(special) == 24, "premise: the pinned tokenizer has 24 special added tokens"
    hit = special & set(tok.encode(block_v2(), add_special_tokens=False).ids)
    assert not hit, f"C6: the block encodes to special token ids {sorted(hit)}"


# --- C6: prefix cache ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "kwargs", "digest"), CONFIGS, ids=[c[0] for c in CONFIGS])
def test_c6_default_system_message_is_bases_bytes(
    name: str, kwargs: dict[str, Any], digest: str
) -> None:
    """C6 (green at BASE, unmarked): with the default block the system message is BASE's, byte for byte."""
    text = PromptBuilder(SYSTEM, **kwargs).system_message().text()
    assert hashlib.sha256(text.encode()).hexdigest() == digest, (
        f"C6: {name}'s cached system message moved"
    )


@pytest.mark.parametrize(("name", "kwargs", "digest"), CONFIGS, ids=[c[0] for c in CONFIGS])
def test_c6_ui_block_defaults_to_v1(name: str, kwargs: dict[str, Any], digest: str) -> None:
    """C6: ``ui_block=UI_INSTRUCTIONS`` given explicitly is the default, byte for byte."""
    explicit = builder(ui_block=ui_instructions.UI_INSTRUCTIONS, **kwargs).system_message().text()
    assert hashlib.sha256(explicit.encode()).hexdigest() == digest, name


@pytest.mark.parametrize(("name", "kwargs", "digest"), CONFIGS, ids=[c[0] for c in CONFIGS])
def test_c6_v2_differs_only_inside_the_blocks_span(
    name: str, kwargs: dict[str, Any], digest: str
) -> None:
    """C6: the v2 system message is the default one with the v1 block's single span replaced by the v2 block; the bytes
    before and after it are identical."""
    v1 = PromptBuilder(SYSTEM, **kwargs).system_message().text()
    v2 = builder(ui_block=block_v2(), **kwargs).system_message().text()
    old = ui_instructions.UI_INSTRUCTIONS
    assert v1.count(old) == 1
    i = v1.index(old)
    assert v2.startswith(v1[:i]), f"{name}: the bytes before the block moved"
    assert v2.endswith(v1[i + len(old) :]), f"{name}: the bytes after the block moved"
    assert v2 == v1[:i] + block_v2() + v1[i + len(old) :], f"{name}: more than the block changed"


def test_c6_v2_system_message_is_the_same_bytes_under_two_hash_seeds() -> None:
    """C6: two interpreters with ``PYTHONHASHSEED`` 0 and 1 build the v2 system message (tools on, live guidance v2) and
    print its sha256; they agree with each other and with this process."""
    code = (
        "import hashlib\n"
        "from local_astra.conductor.prompt_builder import LIVE_GUIDANCE, PromptBuilder\n"
        "from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS_V2\n"
        f"b = PromptBuilder({SYSTEM!r}, tools={TOOLS!r}, tool_guidance=LIVE_GUIDANCE['v2'], ui_block=UI_INSTRUCTIONS_V2)\n"
        "print(hashlib.sha256(b.system_message().text().encode()).hexdigest())\n"
    )
    digests = []
    for seed in ("0", "1"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed, "HOME": os.environ["HOME"]},
        )
        assert proc.returncode == 0, (
            f"C6: the v2 builder failed in a child (seed {seed}): {proc.stderr[-400:]}"
        )
        digests.append(proc.stdout.strip())
    here = (
        builder(tools=TOOLS, tool_guidance=LIVE_GUIDANCE["v2"], ui_block=block_v2())
        .system_message()
        .text()
    )
    assert digests[0] == digests[1] == hashlib.sha256(here.encode()).hexdigest(), digests


def test_c6_v2_block_has_no_special_token_and_no_url() -> None:
    """C6: no tokenizer special token (T-0055's named list and the added-token shape) and no URL at all: the v2 id is
    not a URL, so nothing URL-shaped has a reason to be in this block."""
    text = block_v2()
    assert [t for t in SPECIAL_TOKENS if t in text] == []
    assert SPECIAL_TOKEN_SHAPE.findall(text) == []
    assert URL_RE.findall(text) == [] and "://" not in text and "www." not in text.lower()


def test_c6_special_token_and_url_scans_are_not_inert() -> None:
    """The two scans above fire on a block that carries what they forbid (unmarked)."""
    for bad in ("<channel|>", "<|think|>", "<|separator|>"):
        assert [t for t in SPECIAL_TOKENS if t in bad] or SPECIAL_TOKEN_SHAPE.findall(bad), bad
    assert URL_RE.findall("see https://example.com/x for more")


# --- C8: negatives ---------------------------------------------------------------------------------------------------


def test_c8_v1_literals_are_byte_identical() -> None:
    """C8 (unmarked; content pins, meant to hold until a task changes v1 by name): ``UI_INSTRUCTIONS``, ``UI_EXAMPLE``
    and ``ADVERTISED_ORDER`` are BASE's bytes."""
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    assert h(ui_instructions.UI_INSTRUCTIONS) == UI_INSTRUCTIONS_SHA256
    assert h(ui_instructions.UI_EXAMPLE) == UI_EXAMPLE_SHA256
    assert isinstance(ui_instructions.ADVERTISED_ORDER, tuple)
    assert h("\n".join(ui_instructions.ADVERTISED_ORDER)) == ADVERTISED_ORDER_SHA256


def test_c8_group_c_changes_none_of_the_guarded_files() -> None:
    """C8 over ``BASE_C..TIP_C`` (two literals): none of session.py, cli.py, turn_machine.py, speech_gate.py,
    llm_client.py, contracts/**, tests/fakes/**, hud/** changes. Power: the same predicate sees a session.py change and
    a HUD change in known ranges. ``TIP_C`` is pinned at activation; until then the row is red by design."""
    assert TIP_C is not None, (
        "C8: pin TIP_C to group C's implementation tip (a literal) when activating"
    )
    gr.require_commits(BASE_C, TIP_C, *POWER_CONDUCTOR, *POWER_HUD)
    assert gr.is_ancestor(BASE_C, TIP_C)
    moved = gr.changed(BASE_C, TIP_C, *C8_PATHS)
    assert moved == [], f"C8: group C changed {moved}"
    assert "local_astra/conductor/session.py" in gr.changed(*POWER_CONDUCTOR, *C8_PATHS)
    assert "hud/src/main.js" in gr.changed(*POWER_HUD, *C8_PATHS)


@pytest.mark.parametrize("module", ["ui_instructions.py", "prompt_builder.py"])
def test_c8_prompt_modules_import_no_network_module(module: str) -> None:
    """C8 (unmarked): the prompt modules import nothing that can open a connection (imports only, via ``ast``)."""
    tree = ast.parse((REPO / "local_astra" / "conductor" / module).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    net = ("socket", "http", "urllib", "httpx", "requests", "aiohttp", "websockets", "ssl")
    assert sorted(n for n in names if n.split(".")[0] in net) == []
