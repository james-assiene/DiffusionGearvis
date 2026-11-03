"""The turn-level UI metric judged on the turn's last gate call. T-0108 criteria 10 and 11.

Rulings H-T108 (a) A and (b): ``TurnRecord.extras`` gains ``ui_valid_final`` (``None`` when no
``ui`` commit was gated; else whether the *last* gated call returned a surface) and ``ui_shown``
(``None`` when nothing was gated; else whether *any* call did). ``ui_valid`` keeps its meaning
(sticky AND over gated calls). An ungated commit (non-active request) touches none of the three.
The ``ui.gate`` session-log record gains ``structural_repairs``.

Every triple below is written by hand from those definitions, in the order
``(ui_valid, ui_valid_final, ui_shown)``. The per-call outcomes the scripts rely on (the cut
header fails, the rig's canvas succeeds, at 14cde78 and by criterion 4 after) are pinned by the
setup assertions on the ``ui.gate`` records, not assumed.

Regime: scripted turns over ``tests/fakes/`` with the ``ScriptedClock``; CPU only; n = 1 per row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.protocol import AstraUiSurface
from tests.unit._hud_rig import UI_CANVAS, Recorder, answer, frame, running, thought, ui
from tests.unit._session_log_rig import make_log, of_kind, read_records

QUESTION = "what is on this screen"
OK = UI_CANVAS
#: a canvas cut inside its header: no surface (an unrepaired class, criterion 4)
CUT = '{"version":"v0.9.1","createSur'
#: the rig's canvas with one duplicated opener: structural-only damage
DUP = UI_CANVAS.replace('{"id":"a"', '{"{"id":"a"')
#: distinct texts the rows below deliver from a request that is no longer active
STALE_BAD = CUT + "fa"
STALE_OK = UI_CANVAS.replace("one half", "the other half")
STALE_REQUEST = "stale-request-t0108"
KEYS = ("ui_valid", "ui_valid_final", "ui_shown")


async def run_turn(
    tmp_path: Path, canvases: list[str], *, stale: tuple[str, ...] = ()
) -> tuple[dict[str, Any], Recorder, list[dict[str, Any]]]:
    """One voice turn whose ``ui`` commits carry ``canvases``; returns (extras, wire, ui.gate data)."""
    script = [thought("Thinking. ")]
    script += [ui(i + 1, text) for i, text in enumerate(canvases)]
    script.append(answer("There you go. "))
    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=script,
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        if stale:
            real = rig.session.submit

            def submit(event: Any) -> None:
                if isinstance(event, E.LlmCommit) and event.segment == "ui" and event.text in stale:
                    event = event.model_copy(update={"request_id": STALE_REQUEST})
                real(event)

            rig.session.submit = submit  # type: ignore[method-assign]
        await rig.ask(QUESTION)
        await rig.speak_out()
        records = rig.records()
    assert records, "the turn wrote no turns.jsonl line"
    gates = [r["data"] for r in of_kind(read_records(Path(log.path)), "ui.gate")]
    return records[-1].extras, recorder, gates


def triple(extras: dict[str, Any]) -> tuple[Any, Any, Any]:
    missing = [k for k in KEYS if k not in extras]
    assert not missing, (
        f"criterion 10: the turn line carries {list(KEYS)} on every turn; missing {missing} "
        f"(it holds {sorted(extras)})"
    )
    return tuple(extras[k] for k in KEYS)  # type: ignore[return-value]


def emitted(gates: list[dict[str, Any]]) -> list[bool]:
    return [bool(g["surface_emitted"]) for g in gates if g.get("gated")]


@pytest.mark.parametrize(
    ("name", "canvases", "calls", "want"),
    [
        ("one valid commit", [OK], [True], (True, True, True)),
        ("header cut then OK", [CUT, OK], [False, True], (False, True, True)),
        ("OK then fail", [OK, CUT], [True, False], (False, False, True)),
        # adversarial (mutation 8): the last call decides, not the first and not an AND
        ("OK, fail, OK", [OK, CUT, OK], [True, False, True], (False, True, True)),
        ("only failures", [CUT, CUT], [False, False], (False, False, False)),
        ("no ui", [], [], (None, None, None)),
    ],
)
async def test_the_three_values(
    tmp_path: Path, name: str, canvases: list[str], calls: list[bool], want: tuple[Any, ...]
) -> None:
    extras, _, gates = await run_turn(tmp_path, canvases)
    assert emitted(gates) == calls, f"setup ({name}): per-call outcomes {emitted(gates)}"
    assert triple(extras) == want, (
        f"criterion 10 ({name}): (ui_valid, ui_valid_final, ui_shown) must be {want}; the line "
        f"reads {triple(extras)}"
    )


@pytest.mark.parametrize(
    ("canvases", "stale", "want"),
    [
        # a gated success, then a stale failure: counting it would read (False, False, True)
        ([OK, STALE_BAD], (STALE_BAD,), (True, True, True)),
        # only a stale success: counting it would read (True, True, True)
        ([STALE_OK], (STALE_OK,), (None, None, None)),
    ],
    ids=["stale failure after a success", "stale success alone"],
)
async def test_a_commit_from_a_non_active_request_leaves_all_three_unchanged(
    tmp_path: Path, canvases: list[str], stale: tuple[str, ...], want: tuple[Any, ...]
) -> None:
    extras, _, gates = await run_turn(tmp_path, canvases, stale=stale)
    assert sum(1 for g in gates if g.get("gated") is False) == len(stale), (
        f"setup: the stale commit is logged as not gated; records {gates}"
    )
    assert triple(extras) == want


async def test_a_shown_partial_is_resent_final_when_the_last_call_fails(tmp_path: Path) -> None:
    """Guard (criterion 11, H-T108a): behaviour unchanged at 14cde78 and after. The surface of
    the first call is re-sent with ``final=True`` at turn end although the last call failed."""
    _, recorder, gates = await run_turn(tmp_path, [OK, CUT])
    assert emitted(gates) == [True, False]
    surfaces = recorder.of(AstraUiSurface)
    assert [s.final for s in surfaces] == [False, True], (
        f"criterion 11: one partial, then its final re-send; got {[s.final for s in surfaces]}"
    )
    assert surfaces[1].messages == surfaces[0].messages
    assert surfaces[1].canvas_idx == surfaces[0].canvas_idx == 1


async def test_the_stale_partial_pair_is_the_t0109_trigger(tmp_path: Path) -> None:
    """Criterion 11: the same turn reads ``ui_shown: true, ui_valid_final: false``."""
    extras, _, _ = await run_turn(tmp_path, [OK, CUT])
    assert (extras.get("ui_shown"), extras.get("ui_valid_final")) == (True, False)
    assert "ui_shown" in extras and "ui_valid_final" in extras


async def test_the_ui_gate_record_carries_structural_repairs(tmp_path: Path) -> None:
    """Criterion 10: one additive key on the gated ``ui.gate`` record."""
    extras, _, gates = await run_turn(tmp_path, [DUP, OK])
    gated = [g for g in gates if g.get("gated")]
    assert len(gated) == 2
    for g in gated:
        assert isinstance(g.get("structural_repairs"), int), (
            f"criterion 10: the ui.gate record gains structural_repairs; it holds {sorted(g)}"
        )
    assert gated[0]["structural_repairs"] >= 1, "the duplicated opener is a structural edit"
    assert gated[1]["structural_repairs"] == 0, "a strict-valid canvas has none"
    assert gated[0]["repairs"] >= gated[0]["structural_repairs"]
    assert triple(extras) == (True, True, True)
