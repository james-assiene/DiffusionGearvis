"""T-0127 criteria 5 and 6 (and 4's log keys): ``ui_complete_final`` on the turn line, ``complete``/``lost`` on the
``ui.gate`` record, and the one session-log note an incomplete final surface writes.

Definitions (written by hand from the task): ``ui_complete_final`` is omitted from the turn line when no ``ui`` commit
was gated (spec change at 7b9e6dd; it was ``None``), else the last gated call's ``GateResult.complete``. ``ui_valid``, ``ui_valid_final`` and ``ui_shown`` keep their T-0108
meaning. When the last call is incomplete and a surface was shown, ``_publish_final_surface`` writes exactly one
``note`` ``{"what": "final ui surface is incomplete", "root": bool, "unreachable": [...], "lost": [...]}``, ids only;
no note when complete or when nothing was shown. The final re-send itself is unchanged (H-T127-4).

Reading where the task is open: a turn whose last call returns no surface after an earlier shown one ([OK, CUT]) has
``ui_complete_final`` false; whether it also notes is not pinned here (named in the report).

Regime: scripted turns over ``tests/fakes/`` with the rig's clock; CPU only; n = 1 per row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraUiSurface
from tests.unit._hud_rig import UI_CANVAS, Recorder, answer, frame, running, thought, ui
from tests.unit._session_log_rig import make_log, of_kind, read_records

NOTE = "final ui surface is incomplete"
QUESTION = "what is on this screen"
CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"


def canvas(*components: str) -> str:
    return (
        '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}\n'
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        + ",".join(components)
        + "]}}"
    )


ROOT_A = '{"id":"root","component":"Column","children":["a"]}'
A = '{"id":"a","component":"Text","text":"one half"}'
OK = UI_CANVAS
CUT = '{"version":"v0.9.1","createSur'
#: a valid orphan: unreachable from root
ORPHAN = canvas(ROOT_A, A, '{"id":"z","component":"Text","text":"stray words"}')
#: a Text variant no rule repairs: ``b`` is lost
LOSSY = canvas(
    '{"id":"root","component":"Column","children":["a","b"]}',
    A,
    '{"id":"b","component":"Text","text":"secret heading","variant":"body2"}',
)
#: no root yet, then the whole thing
NO_ROOT = canvas(A)
FULL = canvas(ROOT_A, A)
#: strings the note must never carry (ids only)
CARD_TEXT = ("one half", "stray words", "secret heading")


async def run_turn(
    tmp_path: Path, canvases: list[str]
) -> tuple[dict[str, Any], Recorder, list[dict[str, Any]]]:
    """One voice turn whose ``ui`` commits carry ``canvases``; returns (extras, wire, session-log records)."""
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
        await rig.ask(QUESTION)
        await rig.speak_out()
        records = rig.records()
    assert records, "the turn wrote no turns.jsonl line"
    return records[-1].extras, recorder, read_records(Path(log.path))


def gates(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r["data"] for r in of_kind(log, "ui.gate") if r["data"].get("gated")]


def notes(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r["data"] for r in of_kind(log, "note") if r["data"].get("what") == NOTE]


#: the key is omitted from the turn line when nothing was gated (orchestrator ruling at 7b9e6dd)
ABSENT = "<absent>"


def complete_final(extras: dict[str, Any], *, gated: bool = True) -> Any:
    """The turn line's ``ui_complete_final``: required on a gated turn, absent (``ABSENT``) when nothing was gated."""
    if not gated:
        assert "ui_complete_final" not in extras, (
            f"criterion 5 (7b9e6dd): no gated call, so the key is omitted; it reads {extras['ui_complete_final']!r}"
        )
        return ABSENT
    assert "ui_complete_final" in extras, (
        f"criterion 5: a gated turn's line carries ui_complete_final; it holds {sorted(extras)}"
    )
    return extras["ui_complete_final"]


@pytest.mark.parametrize(
    ("name", "canvases", "want"),
    [
        ("no ui", [], ABSENT),
        ("one complete canvas", [OK], True),
        ("orphan", [ORPHAN], False),
        ("lost component", [LOSSY], False),
        ("no root, then complete", [NO_ROOT, FULL], True),
        ("complete, then no surface", [OK, CUT], False),
        ("only failures", [CUT], False),
    ],
)
async def test_ui_complete_final_per_scenario(
    tmp_path: Path, name: str, canvases: list[str], want: Any
) -> None:
    """Criterion 5: omitted when nothing was gated (7b9e6dd), else the last gated call's ``complete``."""
    extras, _, log = await run_turn(tmp_path, canvases)
    assert len(gates(log)) == len(canvases), f"setup ({name}): {len(gates(log))} gated calls"
    assert complete_final(extras, gated=bool(canvases)) is want, (
        f"{name}: ui_complete_final {extras.get('ui_complete_final')!r}"
    )


@pytest.mark.parametrize("canvases", [[ORPHAN], [LOSSY]], ids=["orphan", "lost"])
async def test_an_incomplete_surface_is_still_valid_and_shown(
    tmp_path: Path, canvases: list[str]
) -> None:
    """Criterion 5 guard (green at base and after; mutation 9): ``ui_valid``, ``ui_valid_final``, ``ui_shown`` are
    unchanged -- an incomplete surface is still emitted and still counts as valid and shown."""
    extras, recorder, _ = await run_turn(tmp_path, canvases)
    assert (extras["ui_valid"], extras["ui_valid_final"], extras["ui_shown"]) == (True, True, True)
    assert [s.final for s in recorder.of(AstraUiSurface)] == [False, True]


async def test_the_ui_gate_record_carries_complete_and_lost(tmp_path: Path) -> None:
    """Criterion 4: every gated ``ui.gate`` record carries ``complete`` (bool) and ``lost`` (list of ids)."""
    _, _, log = await run_turn(tmp_path, [LOSSY, ORPHAN])
    got = [(g.get("complete", "<absent>"), g.get("lost", "<absent>")) for g in gates(log)]
    assert got == [(False, ["b"]), (False, [])], got
    _, _, log_ok = await run_turn(tmp_path / "ok", [OK])
    assert [(g.get("complete"), g.get("lost")) for g in gates(log_ok)] == [(True, [])]


@pytest.mark.parametrize(
    ("canvases", "want"),
    [
        ([ORPHAN], {"what": NOTE, "root": True, "unreachable": ["z"], "lost": []}),
        ([LOSSY], {"what": NOTE, "root": True, "unreachable": [], "lost": ["b"]}),
    ],
    ids=["orphan", "lost"],
)
async def test_exactly_one_incomplete_final_note_ids_only(
    tmp_path: Path, canvases: list[str], want: dict[str, Any]
) -> None:
    """Criterion 6: one note, the ruled shape, ids only (no card text)."""
    _, _, log = await run_turn(tmp_path, canvases)
    got = notes(log)
    assert got == [want], got
    dumped = json.dumps(got)
    assert not [s for s in CARD_TEXT if s in dumped], (
        f"criterion 6: card text in the note: {dumped}"
    )


async def test_a_root_less_final_notes_root_false(tmp_path: Path) -> None:
    """Criterion 6: a final surface with no ``root`` notes ``root: false``."""
    _, _, log = await run_turn(tmp_path, [NO_ROOT])
    got = notes(log)
    assert len(got) == 1 and got[0]["root"] is False, got


@pytest.mark.parametrize(
    "canvases",
    [[OK], [NO_ROOT, FULL], [CUT], []],
    ids=["complete", "completed-later", "never-shown", "no-ui"],
)
async def test_no_note_when_complete_or_nothing_shown(tmp_path: Path, canvases: list[str]) -> None:
    """Criterion 6 (mutation 7): no note when the last call is complete, or when no surface was shown. Paired with
    ``ui_complete_final`` (present when gated, absent when not) so the row is red until the key is right."""
    extras, _, log = await run_turn(tmp_path, canvases)
    assert notes(log) == []
    complete_final(extras, gated=bool(canvases))


@pytest.mark.parametrize("canvases", [[ORPHAN], [LOSSY]], ids=["orphan", "lost"])
async def test_the_final_surface_on_the_wire_is_unchanged(
    tmp_path: Path, canvases: list[str]
) -> None:
    """Criterion 6 / H-T127-4 guard (green at base and after): the incomplete final is still re-sent, and its messages
    are the base gate's for that canvas: the canvas with ``surfaceId`` rewritten, ``b`` pruned for the lost case."""
    _, recorder, _ = await run_turn(tmp_path, canvases)
    partial, final = recorder.of(AstraUiSurface)
    assert final.final and final.messages == partial.messages
    sid = final.surface_id
    comps = json.loads(canvases[0].split("\n")[1])["updateComponents"]["components"]
    if canvases == [LOSSY]:
        comps = [{**comps[0], "children": ["a"]}, comps[1]]
    want = [
        {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": CATALOG}},
        {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": comps}},
    ]
    assert json.loads(json.dumps(list(final.messages))) == want
