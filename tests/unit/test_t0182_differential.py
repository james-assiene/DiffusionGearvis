"""T-0182 group B, the differentials: B8 F2 (the frozen BASE_B machine), B10 (negatives) and B11 (docs/02), under
rulings B-R4 (``TurnPolicy()`` keeps BASE's ack; the product default lives in ``cli``) and B-R5 (no ``extras.clips``;
B4's state field excluded from dumps).

- **The frozen oracle.** ``tests/unit/_frozen_turn_machine_t0182.py`` is BASE_B's (7646b8b6) ``turn_machine.py`` from
  its line 18 on; the sha below was computed by hand from ``git show 7646b8b6:local_astra/conductor/turn_machine.py |
  sed '1,17d'``. ``BASE_B_STATE_FIELDS`` (``_t0182_pins``) is re-derived here from the BASE_B blob of
  ``contracts/turn.py`` with ``ast``.
- **The goldens with filler clips on** (Tests to write; B10). After B-R4 "filler clips on" is
  ``TurnPolicy(ack_clip=True, still_working_after_ms=None)``, what ``cli`` builds in ``off`` with ``--filler-clips``;
  both golden rigs (T-0092a ``test_session_stale_playback_golden``, T-0092b ``test_session_tools_off_golden``) run with
  that policy set explicitly and must equal their fixtures directly, through each module's own comparison row (persona
  and the earlier tasks' declared extras only; B-R5: nothing for T-0182). B-R5 voids the dry_run half: both rigs are
  tools-off only, and dry_run is ``test_t0182_cli``'s BASE_B byte-identity rows.
- **The CLI-default capture** (B1/F2 at session level): the same rigs with the policy ``cli`` resolves with defaults
  (``ack_clip=False``) equal the fixtures through ``_t0182_pins.without_ack`` on both sides. T-0092a: every section.
  T-0092b: ``prompts`` and ``requests`` in full, and ``machine`` up to the TTS-error turn. Past that point the absent
  ack changes the playback run itself (``_t0182_pins`` item 3, validated on the prototype): that is a session
  consequence of B1, not an ack-only record, so it is not normalised away; ``hud`` and ``log`` are not compared there
  (reported). A companion asserts the ack really is gone (no ``play_clip`` ack in the capture).
- **Frozen paths** over ``BASE_B..TIP_B`` (two literals, never the working tree, ``.claude/rules/tests.md``):
  ``filler_policy.py``, ``contracts/commands.py``, ``prompt_builder.py``, ``toolcall/**``, ``computer/**``,
  ``audio_io.py``, ``tts_client.py``, ``tests/fixtures/session/*.json`` unchanged; ``tests/fakes/**`` additive only;
  ``contracts/turn.py`` changes only by B4's one ``MachineState`` field. ``TIP_B`` starts at BASE_B (no
  implementation yet), so these rows hold trivially until it lands. **TIP moves**: whoever commits group B
  implementation bumps ``TIP_B``; it is closed at the merge.
- **CLIP_TEXT / ClipKind / ENDING_CLIPS** (B10, B-R1): content pins, literals copied by hand from the BASE_B blobs (and
  re-checked against them when BASE_B is in the clone).
- ``test_doc02_delta_t0182`` (B8/B11) reads docs/02 at ``TIP_B``: red until the paragraph is committed and TIP bumped.
  Keywords, never wording.

Regime: CPU over the fakes, ``ScriptedClock``, thinking on, n = 1, deterministic; git reads only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import subprocess
import typing
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from tests import _git_range as gr
from tests.unit import _frozen_turn_machine_t0182 as frozen
from tests.unit import _t0182_pins as pins
from tests.unit import test_session_stale_playback_golden as gold_a
from tests.unit import test_session_tools_off_golden as gold_b
from tests.unit import test_t0107a_docs as t7d
from tests.unit.test_t0182_clip_policy import cli_default, clip_policy

#: BASE_B, a literal (``.claude/rules/tests.md``): main when group B branched.
BASE_B = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
#: The group B implementation tip. Started at BASE_B; bumped to 8f836adc (implementation afc57335 + b8d26ae7, then the
#: orchestrator's docs/03 B11 commit), closed at the merge.
TIP_B = "8f836adc43a2bd2d4b93f6e81632f6ba0e377b92"
#: main as merged into the branch at e8dc8271 (its second parent; T-0177 A, T-0183, T-0185 landed since BASE_B). Rows
#: about files main legitimately changed over BASE_B..MAIN_MERGED compare T-0182's own change, MAIN_MERGED..TIP_B.
MAIN_MERGED = "c0cb001ee5b638d88903ca6129f9baf0a87c3000"
BODY_SHA256 = "9ae6fda8307a44b205a7018e9000d873fa131d45793ee11f4ad5e91d5b27be3f"
HEADER = "**Spec delta (T-0182)"

#: B10 / Not owned: byte-identical over BASE_B..TIP_B
FROZEN_B = (
    "local_astra/reflex/filler_policy.py",
    "local_astra/contracts/commands.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/audio_io.py",
    "local_astra/conductor/tts_client.py",
    "local_astra/toolcall",
    "local_astra/computer",
    "tests/fixtures/session/t0092a_golden.json",
    "tests/fixtures/session/t0092b_tools_off_golden.json",
)
#: Power pair: T-0179 B (ebaf4248) added ``lost_thread`` to ``contracts/commands.py``.
POWER_PARENT, POWER = "ebaf4248^", "ebaf4248"

#: ``session.CLIP_TEXT`` at BASE_B, by hand from ``git show 7646b8b6:local_astra/conductor/session.py`` (lines 161-170)
BASE_CLIP_TEXT = {
    "ack": "let me look",
    "hmm": "hmm",
    "still_working": "still working on it",
    "out_of_steps": "I ran out of steps before I could finish.",
    "not_finished": "I could not finish that.",
    "lost_thread": "Sorry, I lost my train of thought. Could you ask again?",
}
#: ``ClipKind`` at BASE_B, by hand from ``contracts/commands.py`` line 100
BASE_CLIP_KIND = ("ack", "hmm", "still_working", "out_of_steps", "not_finished", "lost_thread")
#: ``session.ENDING_CLIPS`` at BASE_B (B-R1: unchanged; its one use is T-0178's ``loop_report``)
BASE_ENDING_CLIPS = frozenset({"out_of_steps", "not_finished"})
#: T-0192 B1 (re-pin before implementation): the one clip T-0192 adds, appended to ``CLIP_TEXT`` and ``ClipKind``;
#: the text typed from docs/tasks/T-0192.md B1 (ruling H-T192-2), not from the code
T0192_CLIP_TEXT = {
    "at_checkout": "I've reached the checkout, so I'll stop here. Over to you to finish the purchase."
}
T0192_CLIP_KIND = ("at_checkout",)
#: T-0192 "``session.ENDING_CLIPS`` is left free": BASE_B's set, or BASE_B's plus ``at_checkout``; nothing else
T0192_ENDING_CLIPS = (BASE_ENDING_CLIPS, BASE_ENDING_CLIPS | {"at_checkout"})


def _show(rev: str, path: str) -> str:
    proc = gr._git("show", f"{rev}:{path}")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# =====================================================================================================================
# The frozen oracle and the BASE_B field list
# =====================================================================================================================


def _body() -> str:
    text = pathlib.Path(frozen.__file__).read_text()
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    return text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"


def test_frozen_copy_is_base_b_turn_machine_verbatim() -> None:
    """F2's oracle: the copied body hashes to the hand-computed sha and, when BASE_B is in the clone, equals it."""
    body = _body()
    assert hashlib.sha256(body.encode()).hexdigest() == BODY_SHA256
    assert frozen.BASE_SHA == BASE_B
    shown = subprocess.run(
        ["git", "show", f"{BASE_B}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"base commit {BASE_B} is not in this clone")
    assert body == "".join(shown.stdout.splitlines(keepends=True)[17:]).rstrip("\n") + "\n"


def _class_fields(source: str, name: str) -> tuple[str, ...]:
    """The annotated class-level names of ``class name`` in ``source``, in order (a pydantic model's fields)."""
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(found) == 1, f"one class {name} in the blob; found {len(found)}"
    return tuple(
        n.target.id
        for n in found[0].body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    )


def test_base_b_state_fields_are_the_blobs() -> None:
    """``_t0182_pins.BASE_B_STATE_FIELDS`` (hand-written) is ``MachineState``'s field list in the BASE_B blob, read
    with ``ast``; a field dropped from or added to the hand list fails here."""
    gr.require_commits(BASE_B)
    fields = _class_fields(_show(BASE_B, "local_astra/contracts/turn.py"), "MachineState")
    assert fields == pins.BASE_B_STATE_FIELDS, (
        f"BASE_B_STATE_FIELDS differs from the blob: only in blob "
        f"{sorted(set(fields) - set(pins.BASE_B_STATE_FIELDS))}, only in the pin "
        f"{sorted(set(pins.BASE_B_STATE_FIELDS) - set(fields))}"
    )


# =====================================================================================================================
# B10: CLIP_TEXT, ClipKind and ENDING_CLIPS are unchanged (content pins)
# =====================================================================================================================


def test_clip_text_clip_kind_and_ending_clips_are_base_bs() -> None:
    """B10 (``CLIP_TEXT``, ``ClipKind`` unchanged) and B-R1 (``ENDING_CLIPS`` stays): the live values equal the
    hand-copied BASE_B literals, and, when BASE_B is in the clone, the literals equal the blobs' (read with ``ast``,
    never by importing the old tree).

    T-0192 B1 re-pin (before implementation): the live ``CLIP_TEXT`` is BASE_B's, in BASE_B's order, plus exactly
    ``at_checkout`` appended; ``ClipKind`` is BASE_B's tuple plus exactly ``"at_checkout"`` appended; ``ENDING_CLIPS``
    is BASE_B's set or BASE_B's plus ``at_checkout`` (the task leaves it free). The BASE_B literals themselves are
    unchanged and still checked against the BASE_B blobs below."""
    want_text = {**BASE_CLIP_TEXT, **T0192_CLIP_TEXT}
    assert want_text == session_mod.CLIP_TEXT, (
        "T-0182 B10 / T-0192 B1: CLIP_TEXT is BASE_B's plus exactly at_checkout"
    )
    assert list(session_mod.CLIP_TEXT) == list(want_text), (
        "T-0182 B10 / T-0192 B1: and so is its order (at_checkout appended)"
    )
    assert typing.get_args(C.ClipKind) == (*BASE_CLIP_KIND, *T0192_CLIP_KIND), (
        "T-0182 B10 / T-0192 B1: ClipKind is BASE_B's plus exactly at_checkout, appended"
    )
    assert session_mod.ENDING_CLIPS in T0192_ENDING_CLIPS, (
        "T-0182 B-R1 / T-0192: ENDING_CLIPS is BASE_B's, or BASE_B's plus at_checkout"
    )
    if not gr.has_commit(BASE_B):
        pytest.skip(f"base commit {BASE_B} is not in this clone; the literals are not re-checked")
    tree = ast.parse(_show(BASE_B, "local_astra/conductor/session.py"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in ("CLIP_TEXT", "ENDING_CLIPS")
            and node.value is not None
        ):
            values[node.target.id] = node.value
    assert ast.literal_eval(values["CLIP_TEXT"]) == BASE_CLIP_TEXT
    ending = values["ENDING_CLIPS"]
    assert isinstance(ending, ast.Call) and ast.literal_eval(ending.args[0]) == set(
        BASE_ENDING_CLIPS
    )
    commands = ast.parse(_show(BASE_B, "local_astra/contracts/commands.py"))
    kind = [
        n.value
        for n in commands.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ClipKind" for t in n.targets)
    ]
    assert len(kind) == 1 and isinstance(kind[0], ast.Subscript)
    assert ast.literal_eval(kind[0].slice) == BASE_CLIP_KIND


# =====================================================================================================================
# The goldens with filler clips on (equal to the fixtures directly) and the CLI-default capture (through without_ack)
# =====================================================================================================================


def _use(monkeypatch: pytest.MonkeyPatch, module: Any, policy: Any) -> None:
    """Run ``module``'s golden rig with ``policy`` (its ``StateMachine`` reads the module's ``POLICY`` at
    construction), and prove it reaches the machine."""
    monkeypatch.setattr(module, "POLICY", policy)
    assert module.StateMachine().policy is policy, "premise: the rig's machine runs the policy"


def _fixture(module: Any) -> dict[str, Any]:
    return json.loads(module.GOLDEN.read_text(encoding="utf-8"))


#: ``cli`` with ``--filler-clips`` in ``off`` (B2 on: ``ack_clip=True``; B4: no ``still_working`` outside live)
def filler_clips_on() -> Any:
    return clip_policy(ack_clip=True, still_working_after_ms=None)


async def test_t0092a_golden_with_filler_clips_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests to write ("the goldens with filler_clips on"), B10: T-0092a's golden session with the filler-clips-on
    policy equals the fixture in every section, through the golden module's own row (no T-0182 normalisation)."""
    _use(monkeypatch, gold_a, filler_clips_on())
    await gold_a.test_every_run_ending_in_its_own_turn_is_identical_to_the_base_sha(
        tmp_path, _fixture(gold_a)
    )


async def test_t0092b_golden_with_filler_clips_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests to write, B10: T-0092b's tools-off golden session with the filler-clips-on policy equals the fixture in
    every section, through the golden module's own row."""
    _use(monkeypatch, gold_b, filler_clips_on())
    await gold_b.test_tools_off_is_identical_to_the_base_sha(tmp_path, _fixture(gold_b))


def _a_records_as_golden(got: dict[str, Any]) -> dict[str, Any]:
    """T-0092a's own record normalisation (its criterion-6 row: T-0092b's ``tool``/``regime.tools``, T-0108's and
    T-0127's extras), applied to a capture; nothing T-0182's."""
    for record in got["records"]:
        assert record.get("tool", "<absent>") is None and record["regime"].get("tools") == "off"
        del record["tool"]
        del record["regime"]["tools"]
        for key in gold_a.T0108_EXTRAS | gold_a.T0127_EXTRAS:
            record["extras"].pop(key, None)
    return got


def _ack_plays(tree: Any) -> int:
    """How many ``play_clip`` command dumps of kind ``ack`` the JSON-like ``tree`` holds."""
    if isinstance(tree, dict):
        own = int(tree.get("type") == "play_clip" and tree.get("kind") == "ack")
        return own + sum(_ack_plays(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_ack_plays(v) for v in tree)
    return 0


async def test_t0092a_cli_default_equals_the_golden_without_the_ack(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1/F2 at session level: T-0092a's session under the policy ``cli`` resolves with defaults (``ack_clip=False``)
    plays no ack, and equals the golden in every section once ``without_ack`` is applied to both sides."""
    _use(monkeypatch, gold_a, cli_default())
    # T-0177 B-P1 (pre-authorised pin move): the published astra.vision.frame messages and their wire.out records,
    # shape checked, are the one declared difference on the observed side; the golden is never normalised.
    from tests.unit._t0177_pins import without_vision_frames

    got = without_vision_frames(
        _a_records_as_golden(gold_a.without_persona(await gold_a.capture(tmp_path)))
    )
    want = _fixture(gold_a)["sections"]
    assert _ack_plays(want["machine"]) > 0, "premise: BASE played the ack in this session"
    assert _ack_plays(got["machine"]) == 0, "T-0182 B1: no ack with the CLI-default policy"
    for section in gold_a.SECTIONS:
        assert pins.without_ack(got[section]) == pins.without_ack(want[section]), (
            f"T-0182 F2: the {section!r} section differs from the golden beyond the ack-only records"
        )
    assert got["machine"] != want["machine"], "premise: the comparison is not vacuous"


async def test_t0092b_cli_default_equals_the_golden_without_the_ack(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1/F2 at session level, T-0092b's tools-off session under the CLI-default policy: no ack; ``prompts`` and
    ``requests`` equal the golden's (the ack never reaches the model); the ``machine`` section equals the golden's
    through ``without_ack`` up to the TTS-error turn's ``llm.start`` (past it the playback run itself differs, a
    session consequence of B1, ``_t0182_pins`` item 3)."""
    _use(monkeypatch, gold_b, cli_default())
    got = gold_b.without_persona(await gold_b.capture(tmp_path))
    want = _fixture(gold_b)["sections"]
    assert _ack_plays(want["machine"]) > 0, "premise: BASE played the ack in this session"
    assert _ack_plays(got) == 0, "T-0182 B1: no ack with the CLI-default policy"
    for section in ("prompts", "requests"):
        assert got[section] == want[section], f"T-0182 B1: the {section!r} section moved"
    types = [e["event"]["type"] for e in want["machine"]]
    error = types.index("tts.error")
    starts = [
        i
        for i, e in enumerate(want["machine"][:error])
        if any(c.get("type") == "llm.start" for c in e["commands"])
    ]
    cut = starts[-1]
    assert cut > 10, "premise: the comparison covers the turns before the TTS-error turn"
    assert pins.without_ack(got["machine"][:cut]) == pins.without_ack(want["machine"][:cut]), (
        "T-0182 F2: before the TTS-error turn the machine section differs beyond the ack-only records"
    )


# =====================================================================================================================
# B10: frozen paths over BASE_B..TIP_B (two literals)
# =====================================================================================================================


def test_frozen_paths_over_base_b_tip_b() -> None:
    """B10 / Not owned: the frozen paths are byte-identical over ``BASE_B..TIP_B``; each exists at BASE_B; the
    predicate has power (it names ``commands.py`` over T-0179 B's commit)."""
    gr.require_commits(BASE_B, TIP_B)
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B[:8]} is not an ancestor of {TIP_B[:8]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_B, "--", *FROZEN_B).stdout.split()
    for path in FROZEN_B:
        assert any(x == path or x.startswith(path + "/") for x in listed), (
            f"premise: {path} exists at BASE"
        )
    changed = gr.changed(BASE_B, TIP_B, *FROZEN_B)
    assert changed == [], f"T-0182 B10: frozen paths differ in {BASE_B[:8]}..{TIP_B[:8]}: {changed}"
    gr.require_commits(POWER)
    assert "local_astra/contracts/commands.py" in gr.changed(POWER_PARENT, POWER, *FROZEN_B)


def test_fakes_are_additive_only_over_base_b_tip_b() -> None:
    """Not owned: ``tests/fakes/**`` additive only (B7's device-clock seam, if any): no fake modified, deleted or
    renamed over ``BASE_B..TIP_B``.

    Main-merge re-scope: T-0177 A legitimately modified ``tests/fakes/hud_server.py`` on main (BASE_B..MAIN_MERGED),
    so T-0182's own change is read over ``MAIN_MERGED..TIP_B`` (MAIN_MERGED is merged into TIP_B) and must modify,
    delete or rename nothing there; over the whole range only main's file may show as modified."""
    gr.require_commits(BASE_B, MAIN_MERGED, TIP_B)
    assert gr.is_ancestor(BASE_B, MAIN_MERGED) and gr.is_ancestor(MAIN_MERGED, TIP_B), (
        "premise: BASE_B -> MAIN_MERGED -> TIP_B"
    )
    touched = gr.changed(MAIN_MERGED, TIP_B, "tests/fakes", diff_filter="MDR")
    assert touched == [], f"T-0182: tests/fakes is additive only; modified or removed {touched}"
    by_main = gr.changed(BASE_B, MAIN_MERGED, "tests/fakes", diff_filter="MDR")
    assert by_main == ["tests/fakes/hud_server.py"], (
        f"premise: main's fake edits are T-0177's; {by_main}"
    )
    whole = gr.changed(BASE_B, TIP_B, "tests/fakes", diff_filter="MDR")
    assert set(whole) <= set(by_main), f"T-0182: fakes modified beyond main's {by_main}: {whole}"


def test_turn_py_gains_only_b4s_state_field() -> None:
    """B-R2: ``contracts/turn.py`` changes only by B4's one ``MachineState`` field: at TIP_B ``MachineState``'s field
    list is BASE_B's plus at most one new name (exactly one once the machine has changed), and every other top-level
    statement of the module is BASE_B's, by ``ast.dump``."""
    gr.require_commits(BASE_B, TIP_B)
    base_src = _show(BASE_B, "local_astra/contracts/turn.py")
    tip_src = _show(TIP_B, "local_astra/contracts/turn.py")
    base_fields = _class_fields(base_src, "MachineState")
    tip_fields = _class_fields(tip_src, "MachineState")
    added = [f for f in tip_fields if f not in base_fields]
    assert [f for f in tip_fields if f in base_fields] == list(base_fields), (
        "T-0182 B-R2: BASE_B's MachineState fields keep their order and none is removed"
    )
    machine_changed = bool(gr.changed(BASE_B, TIP_B, "local_astra/conductor/turn_machine.py"))
    assert len(added) == (1 if machine_changed else 0), (
        f"T-0182 B-R2: turn.py gains exactly B4's one MachineState field; added {added}"
    )

    def others(src: str) -> list[str]:
        return [
            ast.dump(n)
            for n in ast.parse(src).body
            if not (isinstance(n, ast.ClassDef) and n.name == "MachineState")
        ]

    assert others(tip_src) == others(base_src), (
        "T-0182 B-R2: turn.py changes only inside MachineState"
    )


# =====================================================================================================================
# B8 / B11: docs/02
# =====================================================================================================================


def _row(table_lines: list[str], *needles: str) -> list[str]:
    return [line for line in table_lines if all(n in line for n in needles)]


def test_doc02_delta_t0182() -> None:
    """B8 / B11 / B-R2: docs/02 §4 at TIP_B
    - row 6 (the ENDPOINTING Tick row that starts the generate) names ``ack_clip`` beside the ack decision;
    - a THINKING ``Tick`` row emits ``play_clip(still_working)`` and names ``still_working_after_ms``;
    - one paragraph starting ``**Spec delta (T-0182)`` records: ack off by default and ``filler_clips`` restoring it,
      ``still_working`` once per turn after ``still_working_after_ms`` (20 s) in live tool loops, the ending clips
      still playing after it, B-R2's stale ack flag (``clip_playing`` never cleared in a silent tool loop), and the
      ``TURN_MACHINE_SHA256`` re-pin with its reason;
    - apart from §4, docs/02 equals BASE_B's paragraph for paragraph; inside §4 every BASE_B paragraph and every table
      line that does not mention a clip is kept verbatim. Keywords, never wording.

    Main-merge re-scope: T-0177 and T-0183 edited §2.1 and §5 on main, so "outside §4 unchanged" is read against
    MAIN_MERGED's docs/02 (merged into TIP_B); MAIN_MERGED's §4 is BASE_B's verbatim (premise), so every §4 check
    still compares against BASE_B's text."""
    gr.require_commits(BASE_B, MAIN_MERGED, TIP_B)
    assert gr.is_ancestor(MAIN_MERGED, TIP_B), "premise: main is merged into TIP_B"
    base_b = _show(BASE_B, "docs/02-architecture.md")
    base = _show(MAIN_MERGED, "docs/02-architecture.md")
    assert t7d.section(base, "## 4. ") == t7d.section(base_b, "## 4. "), (
        "premise: main did not touch docs/02 §4 over BASE_B..MAIN_MERGED"
    )
    doc = _show(TIP_B, "docs/02-architecture.md")
    s4 = t7d.section(doc, "## 4. ")
    base_s4 = t7d.section(base, "## 4. ")
    table = [line for line in s4.splitlines() if line.startswith("|")]
    base_row6 = [
        line
        for line in base_s4.splitlines()
        if line.startswith("| ENDPOINTING | Tick") and "llm.start" in line
    ]
    assert len(base_row6) == 1 and "ack" in base_row6[0], "premise: BASE_B's row 6"
    row6 = [
        line for line in table if line.startswith("| ENDPOINTING | Tick") and "llm.start" in line
    ]
    assert len(row6) == 1 and "ack_clip" in row6[0], (
        f"T-0182 B11: §4 row 6 names ack_clip; rows naming llm.start and the ack: {row6}"
    )
    tick = [
        line
        for line in _row(table, "Tick", "still_working")
        if line.lstrip("| ").startswith("THINKING")
    ]
    assert len(tick) == 1 and "still_working_after_ms" in tick[0], (
        f"T-0182 B11: one THINKING Tick row plays still_working after still_working_after_ms; {tick}"
    )
    paras = [p for p in t7d.paragraphs(s4) if p.lstrip("> ").startswith(HEADER)]
    assert len(paras) == 1, f"T-0182 B11: one §4 paragraph starting {HEADER!r}; {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "ack_clip": "ack_clip" in para,
        "filler_clips": "filler_clips" in para,
        "still_working": "still_working" in para,
        "still_working_after_ms / 20 s": "still_working_after_ms" in para
        and re.search(r"\b20\s*(s\b|000)", para) is not None,
        "once per turn": "once" in low and "turn" in low,
        "live tool loop": "live" in low and "loop" in low,
        "ending clip still plays": "ending" in low,
        "B-R2 stale flag (clip_playing)": "clip_playing" in para,
        "TURN_MACHINE_SHA256": "TURN_MACHINE_SHA256" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0182 B8/B11: the docs/02 delta does not record {missing}"
    outside = [p for p in t7d.paragraphs(doc) if p not in t7d.paragraphs(s4)]
    base_outside = [p for p in t7d.paragraphs(base) if p not in t7d.paragraphs(base_s4)]
    assert outside == base_outside, "T-0182 B11: docs/02 changes only in §4"
    base_table = [line for line in base_s4.splitlines() if line.startswith("|")]
    lost = [line for line in base_table if "clip" not in line and line not in table]
    assert lost == [], f"T-0182 B11: §4 table lines without a clip changed: {lost[:2]}"
    kept = [p for p in t7d.paragraphs(base_s4) if not p.startswith("|")]
    gone = [p[:80] for p in kept if p not in t7d.paragraphs(s4)]
    assert gone == [], f"T-0182 B11: §4 prose paragraphs stay verbatim; changed {gone}"
