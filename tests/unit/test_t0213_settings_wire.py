"""T-0213 group A2-read: the server says what the settings are, read-only (docs/tasks/T-0213.md A2r-1..A2r-5).

**The surface these rows assume** (the task's names; readings flagged in the test-author's report):

- A2r-1: ``contracts/protocol.py`` defines ``AstraSettingsState`` (``type = "astra.settings.state"``, server to client,
  in ``SERVER_TO_CLIENT`` and not in ``CLIENT_TO_SERVER``), ``SettingItem`` and ``CatalogEntry``. The rows build them
  through ``model_validate`` from wire-shaped dicts, never by positional constructor. ``SettingItem`` fields:
  ``key, value, source in {default, profile, flag, ui}, kind, changeable, pending, relaunch``; because the codec
  encodes with ``exclude_none``, ``pending``, ``relaunch`` and ``catalog`` must decode when absent (reading
  **W-1**). ``refused`` items are ``{key, reason}``. ``CatalogEntry`` is read only by its ``key``, ``what``,
  ``useful_when``, ``not_useful_when`` and ``evidence`` (the A1-1 text fields every reading of "the text fields"
  includes; reading **W-2**).
- A2r-3: ``HudServer.hello_reply`` yields exactly one ``astra.settings.state``, **directly after**
  ``astra.assist.state`` (reading **W-3** of "after the assist state"; the caller's brief says "assist state, then
  settings state"), with ``catalog`` set (one entry per ``settings_catalog.CATALOG`` key) and ``replay`` False; a
  session that receives no ``astra.settings.set`` publishes none outside a hello (A2-read is read-only).
  ``hello_head_with_settings`` is the one shared re-pin the hello-pinning modules read.
- A2r-2: a ``cli``-built session's state has one item per catalog key it reports (reading **W-4**: items are keyed
  by catalog key; every item key is a catalog key), each with the value and source the ``cli`` resolver gave.
- A2r-4: ``turns_log.Regime`` gains ``settings_rev: int = 0``, ``system_sha12: str = ""`` and
  ``instructions_edited: bool = False``; ``system_sha12`` is ``sha256(text.encode("utf-8")).hexdigest()[:12]`` of the
  turn's rendered system message text, i.e. ``Message.text()`` of the request's ``role == "system"`` message (reading
  **W-5**: UTF-8).

Every A2-read row is active (the last four ``pending`` rows were activated with B-read); a row that needs code which
is missing fails first on a named "does not exist yet" assertion. The R-10 and R-12 rows at the end (panel review
rulings, 2026-10-31) are active since the R-10a/R-12 change landed.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, profile ``lean``; nothing served, no socket; n = 1,
deterministic per row. No measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.contracts import protocol as P
from local_astra.contracts.codec import Malformed, decode, encode
from local_astra.contracts.protocol import CLIENT_TO_SERVER, SERVER_TO_CLIENT, AstraHello
from local_astra.settings_catalog import CATALOG
from local_astra.telemetry.turns_log import Regime, TurnRecord

REPO = Path(__file__).resolve().parents[2]
SETTINGS_STATE_TYPE = "astra.settings.state"
NEW_REGIME_FIELDS = ("settings_rev", "system_sha12", "instructions_edited")
#: A1-4's bound, typed from the criterion (``prompt_builder.MAX_USER_INSTRUCTIONS``)
INSTRUCTIONS_MAX = 1000
SOURCES = ("default", "profile", "flag", "ui")

# ------------------------------------------------------------------------------------------------- seams


def settings_state_class() -> Any:
    cls = getattr(P, "AstraSettingsState", None)
    assert cls is not None, (
        "T-0213 A2r-1: contracts/protocol.py AstraSettingsState (astra.settings.state) does not exist yet"
    )
    return cls


def require_wire() -> tuple[Any, Any, Any]:
    state = settings_state_class()
    item = getattr(P, "SettingItem", None)
    entry = getattr(P, "CatalogEntry", None)
    assert item is not None, "T-0213 A2r-1: contracts/protocol.py SettingItem does not exist yet"
    assert entry is not None, "T-0213 A2r-1: contracts/protocol.py CatalogEntry does not exist yet"
    return state, item, entry


def require_regime_fields() -> None:
    missing = [f for f in NEW_REGIME_FIELDS if f not in Regime.model_fields]
    assert not missing, (
        f"T-0213 A2r-4: telemetry/turns_log.Regime fields {missing} do not exist yet"
    )


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def system_text(prompt: Any) -> str:
    """The rendered system message text of one request: ``Message.text()`` of its one ``system`` message."""
    systems = [m for m in prompt.messages if m.role == "system"]
    assert len(systems) == 1, f"premise: one system message per request; got {len(systems)}"
    return str(systems[0].text())


def settings_states(messages: Sequence[Any]) -> list[Any]:
    return [m for m in messages if getattr(m, "type", None) == SETTINGS_STATE_TYPE]


def _type_of(message: Any) -> Any:
    """The wire type of a hello entry as a row holds it: a raw JSON frame, a decoded dict, a message, a message class
    or a type string."""
    if isinstance(message, type):
        field = getattr(message, "model_fields", {}).get("type")
        return None if field is None else field.default
    if isinstance(message, str):
        if message.startswith("{"):
            try:
                return json.loads(message).get("type")
            except ValueError:
                return None
        return message
    if isinstance(message, dict):
        return message.get("type")
    return getattr(message, "type", None)


def hello_with_settings_last(reply: Sequence[Any]) -> list[Any]:
    """R-5 named re-pin: a BASE hello reply (message classes or type strings) with A2r-3's one settings state
    appended **last**, in the reply's own kind (a class when the reply holds classes, else the type string). Unchanged
    on a tree with no ``AstraSettingsState`` (looked up by name)."""
    cls = getattr(P, "AstraSettingsState", None)
    if cls is None:
        return list(reply)
    new: Any = SETTINGS_STATE_TYPE if (reply and isinstance(reply[-1], str)) else cls
    return [*reply, new]


def hello_without_settings_state(reply: Sequence[Any]) -> list[Any]:
    """R-5 named re-pin: a whole hello reply as a row holds it (raw frames, dicts or messages) with its one
    ``astra.settings.state`` taken out, **after asserting** that there is exactly one and that it is the last message
    (R-5). What is left is what the row compares with BASE. Unchanged on a tree with no ``AstraSettingsState``."""
    if getattr(P, "AstraSettingsState", None) is None:
        return list(reply)
    types = [_type_of(m) for m in reply]
    at = [i for i, t in enumerate(types) if t == SETTINGS_STATE_TYPE]
    assert len(at) == 1, f"T-0213 A2r-3: exactly one settings state in a hello reply; got {types}"
    assert at[0] == len(types) - 1, (
        f"T-0213 A2r-3 (R-5): the settings state is the last message of the hello reply; got {types}"
    )
    return list(reply[:-1])


def item_dict(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "key": "action_outcome",
        "value": False,
        "source": "default",
        "kind": "next_turn",
        "changeable": True,
        "pending": None,
        "relaunch": None,
    }
    base.update(over)
    return base


def state_dict(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": SETTINGS_STATE_TYPE,
        "rev": 0,
        "items": [
            item_dict(),
            item_dict(
                key="browser",
                value="headed",
                source="flag",
                kind="relaunch",
                changeable=False,
                relaunch="diffusion_gearvis session --tools live --browser headless",
            ),
            item_dict(key="visual_tokens", value=1120, source="profile", kind="next_turn_cold"),
        ],
        "catalog": None,
        "system_sha12": "0123456789ab",
        "instructions": "",
        "instructions_max": INSTRUCTIONS_MAX,
        "refused": [{"key": "action_outcome", "reason": "needs live feedback"}],
        "replay": False,
    }
    base.update(over)
    return base


def make_server(tmp_path: Path) -> tuple[Any, Any]:
    from tests.unit import test_hud_hello_replay as hr

    return hr.make_server(tmp_path)


def hello(server: Any) -> list[Any]:
    connection = server.open()
    connection.receive(encode(AstraHello()))
    return list(connection.sent())


# ------------------------------------------------------------------------------------------------- A2r-1: codec


def test_codec_round_trip():
    """A2r-1: a hand-built state (every field, items of three sources and two kinds, a refusal) survives
    ``decode(encode(m)) == m``, and the wire form is a fixpoint. The absent ``catalog`` (None) decodes."""
    state, _, _ = require_wire()
    m = state.model_validate(state_dict())
    wire = encode(m)
    back = decode(wire)
    assert not isinstance(back, Malformed), f"T-0213 A2r-1: a valid state decodes; got {back}"
    assert back == m, f"T-0213 A2r-1: round trip; sent {m!r}, got {back!r}"
    assert encode(back) == wire, "T-0213 A2r-1: the wire form is a fixpoint of encode . decode"
    assert json.loads(wire)["type"] == SETTINGS_STATE_TYPE
    assert m.replay is False and m.catalog is None and m.rev == 0
    assert [i.source for i in m.items] == ["default", "flag", "profile"]
    assert m.items[1].relaunch == "diffusion_gearvis session --tools live --browser headless"


def test_the_type_is_server_to_client_only():
    """A2r-1: ``astra.settings.state`` joins ``SERVER_TO_CLIENT`` and is not a client message."""
    settings_state_class()
    assert SETTINGS_STATE_TYPE in SERVER_TO_CLIENT, sorted(SERVER_TO_CLIENT)
    assert SETTINGS_STATE_TYPE not in CLIENT_TO_SERVER, sorted(CLIENT_TO_SERVER)


@pytest.mark.parametrize(
    "where",
    ["top", "item", "refused", "source", "negative_rev", "kind_value"],
)
def test_an_unknown_field_or_bad_value_is_malformed(where):
    """A2r-1 (adversarial, the strict contract): an unknown top-level field, an unknown field in one item or in one
    refusal, a source outside {default, profile, flag, ui}, ``rev < 0`` each decode ``Malformed`` (never ``Unknown``,
    never accepted). ``kind_value`` is the control: the A1-1 kind ``next_turn_cold`` is accepted."""
    settings_state_class()
    raw = state_dict()
    if where == "top":
        raw["settings"] = {}
    elif where == "item":
        raw["items"][0]["note"] = "x"
    elif where == "refused":
        raw["refused"][0]["hint"] = "x"
    elif where == "source":
        raw["items"][0]["source"] = "cli"
    elif where == "negative_rev":
        raw["rev"] = -1
    got = decode(json.dumps(raw))
    if where == "kind_value":
        assert not isinstance(got, Malformed) and got.type == SETTINGS_STATE_TYPE, (
            f"T-0213 A2r-1: the control state decodes; got {got}"
        )
        return
    assert isinstance(got, Malformed), (
        f"T-0213 A2r-1 (contracts are extra='forbid', source is a literal, rev >= 0): {where!r} must decode "
        f"Malformed; got {got!r}"
    )


def test_every_source_literal_is_accepted():
    """A2r-1, the other direction of the source guard: each of the four sources is accepted (a guard too wide)."""
    require_wire()
    for source in SOURCES:
        raw = state_dict(items=[item_dict(source=source)])
        got = decode(json.dumps(raw))
        assert not isinstance(got, Malformed), (
            f"T-0213 A2r-1: source {source!r} is legal; got {got}"
        )
        assert got.items[0].source == source


# ------------------------------------------------------------------------------------------------- A2r-3: hello


# T-0227 re-pin (named, docs/tasks/T-0227.md Owned files): the R-9 premise of 29 rows becomes 31
async def test_hello_carries_one_settings_state_last(tmp_path):
    """A2r-3 (R-5): one ``astra.settings.state``, the **last** message of the hello reply (so no earlier message
    moves), live (``replay`` False), with the catalog: one entry per ``CATALOG`` key, its text fields equal to the
    catalog's (the single source of truth), and ``instructions_max`` equal to A1-4's 1000. R-9: one item per catalog
    row (29), in catalog order."""
    settings_state_class()
    _, server = make_server(tmp_path)
    reply = hello(server)
    types = [m.type for m in reply]
    found = [i for i, t in enumerate(types) if t == SETTINGS_STATE_TYPE]
    assert len(found) == 1, f"T-0213 A2r-3: exactly one settings state in a hello; reply {types}"
    assert found[0] == len(types) - 1, (
        f"T-0213 A2r-3 (R-5): the settings state is the last message of the hello reply; reply {types}"
    )
    assert "astra.assist.state" in types[: found[0]], types
    m = reply[found[0]]
    assert m.replay is False, f"T-0213 A2r-3: the hello's settings state is live; got {m!r}"
    assert m.catalog is not None, (
        "T-0213 A2r-1/A2r-3: the hello's settings state carries the catalog"
    )
    assert [e.key for e in m.catalog] == [e.key for e in CATALOG], (
        "T-0213 A2r-1: one catalog entry per settings_catalog.CATALOG row, in its order; got "
        f"{[e.key for e in m.catalog]}"
    )
    by_key = {e.key: e for e in CATALOG}
    for got in m.catalog:
        want = by_key[got.key]
        for name in ("what", "useful_when", "not_useful_when", "evidence"):
            assert getattr(got, name) == getattr(want, name), (
                f"T-0213 A2r-1: catalog {got.key}.{name} is the catalog module's text"
            )
    assert m.instructions_max == INSTRUCTIONS_MAX and m.instructions == "", (
        m.instructions_max,
        m.instructions,
    )
    assert m.rev == 0 and list(m.refused) == [], (m.rev, m.refused)
    assert len(CATALOG) == 31, (
        f"premise (R-9, T-0227 re-pin): the catalog has 31 rows; got {len(CATALOG)}"
    )
    assert [i.key for i in m.items] == [e.key for e in CATALOG], (
        "T-0213 R-9: the hello state has one item for every catalog row, in catalog order, so the panel never shows "
        f"'not reported'; got {[i.key for i in m.items]}"
    )
    assert all(i.source in SOURCES for i in m.items)
    wire = encode(m)
    assert encode(decode(wire)) == wire, (
        "T-0213 A2r-1: the hello's own state (with its catalog) round-trips"
    )


async def test_every_hello_carries_the_catalog_and_nothing_else_publishes_a_settings_state(
    tmp_path,
):
    """A2r-1/A2r-3: a second connection's hello carries the catalog too; a turn published through the server adds
    no settings state to the live stream (A2-read is read-only: without a set, none is published)."""
    from tests.unit import test_hud_hello_replay as hr

    settings_state_class()
    _, server = make_server(tmp_path)
    first = server.open()
    first.receive(encode(AstraHello()))
    n_hello = len(first.sent())
    hr.publish_a_turn(server)
    live = list(first.sent())[n_hello:]
    assert not settings_states(live), (
        f"T-0213 A2-read: no settings state outside a hello; got {live}"
    )
    second = settings_states(hello(server))
    assert len(second) == 1 and second[0].catalog is not None, (
        "T-0213 A2r-3: every hello reply carries one settings state with the catalog"
    )


# ------------------------------------------------------------------------------------------------- A2r-2: sources


def _cli_state(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> dict[str, Any]:
    """A real ``cli`` start over the fakes (nothing served); the hello's settings items by key."""
    from tests.unit.test_cli_t0132 import build
    from tests.unit.test_t0151_cli import LIVE_PROFILE

    settings_state_class()
    calls, _ = build(tmp_path, capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    graph = calls.graphs[-1]
    states = settings_states(list(graph.server.hello_reply()))
    assert len(states) == 1, (
        f"T-0213 A2r-3: one settings state in the cli session's hello; got {len(states)}"
    )
    return {i.key: i for i in states[0].items}


SOURCE_CASES = {
    "flag": (["--action-outcome"], "", True, "flag"),
    "profile": ([], "action_outcome = true\n", True, "profile"),
    "default": ([], "", False, "default"),
}


@pytest.mark.parametrize("case", list(SOURCE_CASES))
def test_action_outcome_source(tmp_path, capsys, monkeypatch, case):
    """A2r-2: ``--action-outcome`` gives ``flag``; the profile key ``action_outcome = true`` gives ``profile``; nothing
    gives ``default`` (``LIVE_DEFAULT_ACTION_OUTCOME``, False). The flag row and the profile row agree on the value
    and differ only by the source, so a source map hard-wired to one answer reddens two of three."""
    flags, body, value, source = SOURCE_CASES[case]
    items = _cli_state(tmp_path, capsys, monkeypatch, *flags, body=body)
    assert "action_outcome" in items, (
        f"T-0213 A2r-2: the state reports action_outcome; keys {sorted(items)}"
    )
    got = items["action_outcome"]
    assert (got.value, got.source) == (value, source), (
        f"T-0213 A2r-2 ({case}): action_outcome is ({value!r}, {source!r}); got ({got.value!r}, {got.source!r})"
    )
    assert got.changeable is True and got.pending is None, got


def test_values_and_sources_come_from_the_cli_resolvers(tmp_path, capsys, monkeypatch):
    """A2r-2 (adversarial, "a value is never re-derived"): with no flag and no profile key, two resolvers are made to
    answer what the defaults would not — ``action_outcome_of`` says ``(True, "profile")`` and ``repeat_guard_of`` says
    ``(False, "flag")`` — and the state reports exactly those. A ``cli`` that re-reads the flags or the profile itself
    reports ``(False, "default")`` and ``(True, "default")`` and fails."""
    real_ao, real_rg = cli.action_outcome_of, cli.repeat_guard_of

    def ao(profile: Any, args: Any, mode: str) -> Any:
        got = real_ao(profile, args, mode)
        return (
            dataclasses.replace(got, action_outcome=True, action_outcome_from="profile")
            if mode == "live"
            else got
        )

    def rg(profile: Any, args: Any, mode: str) -> Any:
        got = real_rg(profile, args, mode)
        return (
            dataclasses.replace(got, repeat_guard=False, repeat_guard_from="flag")
            if mode == "live"
            else got
        )

    monkeypatch.setattr(cli, "action_outcome_of", ao)
    monkeypatch.setattr(cli, "repeat_guard_of", rg)
    items = _cli_state(tmp_path, capsys, monkeypatch)
    for key, want in (("action_outcome", (True, "profile")), ("repeat_guard", (False, "flag"))):
        assert key in items, f"T-0213 A2r-2: the state reports {key}; keys {sorted(items)}"
        got = (items[key].value, items[key].source)
        assert got == want, (
            f"T-0213 A2r-2: {key} is the resolver's {want}, not re-derived; got {got}"
        )


def test_grounding_value_and_source_from_its_resolver(tmp_path, capsys, monkeypatch):
    """A2r-2: ``--visual-tokens 560`` reports ``visual_tokens`` (560, flag); the ReGround it left alone reports
    (False, default) — two values from one resolver (``grounding_of``) with two different sources."""
    items = _cli_state(tmp_path, capsys, monkeypatch, "--visual-tokens", "560")
    assert (items["visual_tokens"].value, items["visual_tokens"].source) == (560, "flag"), items[
        "visual_tokens"
    ]
    assert (items["reground"].value, items["reground"].source) == (False, "default"), items[
        "reground"
    ]


# ------------------------------------------------------------------------------------------------- A2r-4: Regime


GOLDEN = REPO / "tests" / "fixtures" / "session" / "t0092b_tools_off_golden.json"
#: a turns.jsonl line in the shape main wrote before T-0213 (regime with ``tools``, no T-0213 field), typed by hand
OLD_LINE = (
    '{"turn_id":"t9","session_id":"s","spans":{"llm.prefill":[0,0]},"events":[],"regime":{"profile":"lean",'
    '"canvas_length":256,"max_denoising_steps":48,"thinking":true,"vllm_version":"v","model":"m","stt":"s",'
    '"tts":"t","audio_path":"headset","assists":"pixels","assist_flags":{"at_spi":false,"cells":false,'
    '"page_text":false,"pdf_text":false},"git_commit":"abc","seed":null,"tools":"native:0123456789ab:live"},'
    '"input":"text","outcome":"spoken","canvases":1,"thought_tokens":0,"answer_tokens":3,"thought_closed":null,'
    '"session_log":null,"tool":null,"extras":{}}'
)


def test_regime_parses_a_line_written_before_this_change():
    """A2r-4: a line captured at a502871 (the tools-off golden's first record) and a hand-typed line of the shape
    main wrote before T-0213 both parse, with the three new fields at their defaults (0, "", False)."""
    require_regime_fields()
    captured = json.loads(GOLDEN.read_text(encoding="utf-8"))["sections"]["records"][0]
    assert not set(NEW_REGIME_FIELDS) & set(captured["regime"]), (
        "premise: the a502871 line predates T-0213"
    )
    for line in (json.dumps(captured), OLD_LINE):
        record = TurnRecord.from_line(line)
        got = tuple(getattr(record.regime, f) for f in NEW_REGIME_FIELDS)
        assert got == (0, "", False), f"T-0213 A2r-4: an old line reads the defaults; got {got}"


def _regime_keys_of_dump(regime: Regime) -> dict[str, Any]:
    """What a turn line holds for ``regime``: the record's own ``to_line`` (the writer's path), parsed back."""
    record = TurnRecord.from_line(OLD_LINE).model_copy(update={"regime": regime})
    return dict(json.loads(record.to_line())["regime"])


@pytest.mark.parametrize(
    ("update", "dumped"),
    [
        ({"settings_rev": 1, "system_sha12": "0123456789ab"}, True),
        ({"instructions_edited": True, "system_sha12": "0123456789ab"}, True),
        ({"system_sha12": "0123456789ab"}, False),
        ({}, False),
    ],
    ids=["rev_1", "instructions_edited", "rev_0_with_sha", "defaults"],
)
def test_regime_dumps_the_three_fields_only_once_something_changed(update, dumped):
    """R-8 (both directions of the guard): a ``Regime`` with ``settings_rev=1`` or ``instructions_edited`` writes all
    three fields with its values, and the line parses back to them; a rev-0, unedited ``Regime`` writes none of them
    **even when it holds a sha** (the session always knows one), so its line is the line BASE wrote. A writer that
    always dumps fails ``rev_0_with_sha``/``defaults``; one that never dumps fails ``rev_1``/``instructions_edited``;
    one keyed on ``settings_rev`` alone fails ``instructions_edited``. The old line still parses (``OLD_LINE``)."""
    require_regime_fields()
    old = TurnRecord.from_line(OLD_LINE)
    regime = Regime.model_validate({**old.regime.model_dump(), **update})
    got = _regime_keys_of_dump(regime)
    present = [f for f in NEW_REGIME_FIELDS if f in got]
    if dumped:
        assert present == list(NEW_REGIME_FIELDS), (
            f"T-0213 R-8: a changed regime ({update}) writes all three fields; it wrote {present}"
        )
        want = {
            "settings_rev": update.get("settings_rev", 0),
            "system_sha12": "0123456789ab",
            "instructions_edited": update.get("instructions_edited", False),
        }
        assert {f: got[f] for f in NEW_REGIME_FIELDS} == want, got
        back = TurnRecord.from_line(
            TurnRecord.from_line(OLD_LINE).model_copy(update={"regime": regime}).to_line()
        )
        assert tuple(getattr(back.regime, f) for f in NEW_REGIME_FIELDS) == tuple(want.values())
    else:
        assert present == [], (
            f"T-0213 R-8: a regime nobody changed ({update or 'defaults'}) writes none of the three fields; it wrote "
            f"{present}"
        )
        assert set(got) == set(json.loads(OLD_LINE)["regime"]), (
            "T-0213 R-8: the rev-0 regime writes exactly the keys BASE wrote"
        )


async def test_turn_line_carries_the_system_sha12_tools_off(tmp_path):
    """A2r-4 under R-8: a tools-off turn nobody changed the settings for writes a line with none of the three
    ``Regime`` fields (byte-identical to BASE: the golden rows below), and the session's ``system_sha12`` lives in the
    hello state instead: sha256[:12] of the system message text the fake backend was handed (every request of the
    turn, the prefill included, carries that one text)."""
    from local_astra.conductor.hud_server import HudServer
    from tests.unit import test_session_tools_off_golden as g
    from tests.unit.test_hud_hello_replay import _NullSink

    require_regime_fields()
    settings_state_class()
    async with g.golden_rig(tmp_path) as rig:
        rig.llm.script = g.SPOKEN
        await rig.ask("what is on this screen")
        await rig.speak_out()
        await rig.pump()
        lines = rig.lines()
        prompts = [p for _, p in rig.llm.prefills] + [p for _, p, _ in rig.llm.generates]
        states = settings_states(hello(HudServer(rig.session, frame_sink=_NullSink(), port=0)))
    texts = {system_text(p) for p in prompts}
    assert len(texts) == 1, f"premise: one system text per turn; got {len(texts)}"
    assert len(lines) == 1, f"premise: one turn line; got {len(lines)}"
    regime = json.loads(lines[0])["regime"]
    assert not [f for f in NEW_REGIME_FIELDS if f in regime], (
        f"T-0213 R-8: a rev-0 turn line carries none of {NEW_REGIME_FIELDS}; got {sorted(regime)}"
    )
    want = sha12(texts.pop())
    assert len(states) == 1 and states[0].system_sha12 == want, (
        f"T-0213 R-8/A2r-1: the hello state's system_sha12 is sha256(system text)[:12] = {want}; got "
        f"{[s.system_sha12 for s in states]}"
    )


async def test_turn_line_system_sha12_follows_the_system_message_tools_on(tmp_path):
    """A2r-4 under R-8 (adversarial: a constant or a once-per-process sha): a tools-on (dry_run) turn's system message
    carries the tool instructions, so its text differs from the tools-off one; the hello state's ``system_sha12``
    asked after the turn is that text's sha, and the rev-0 turn line carries none of the three fields."""
    from tests.unit._tool_turn_support import answer, tool_rig

    require_regime_fields()
    settings_state_class()
    async with tool_rig(tmp_path, script=[answer("It says welcome back.")]) as rig:
        await rig.ask("and what does it say now")
        await rig.turn_done()
        lines = rig.lines()
        prompts = [p for _, p, _ in rig.llm.generates]
        from local_astra.conductor.hud_server import HudServer
        from tests.unit.test_hud_hello_replay import _NullSink

        server = HudServer(rig.session, frame_sink=_NullSink(), port=0)
        states = settings_states(hello(server))
    texts = {system_text(p) for p in prompts}
    assert len(texts) == 1 and len(lines) == 1, (len(texts), len(lines))
    text = texts.pop()
    from local_astra.conductor.prompt_builder import NATIVE_TOOL_INSTRUCTIONS

    assert NATIVE_TOOL_INSTRUCTIONS in text, (
        "premise: the tools-on system text carries the tool instructions"
    )
    want = sha12(text)
    regime = json.loads(lines[0])["regime"]
    assert not [f for f in NEW_REGIME_FIELDS if f in regime], (
        f"T-0213 R-8: a rev-0 turn line carries none of {NEW_REGIME_FIELDS}; got {sorted(regime)}"
    )
    assert len(states) == 1 and states[0].system_sha12 == want, (
        f"T-0213 R-8/A2r-1: the state's system_sha12 is the session's system text sha {want}; got "
        f"{[s.system_sha12 for s in states]}"
    )


# ------------------------------------------------------------------------------------------------- A2r-5: negative


def _assert_no_new_regime_fields(record: dict[str, Any], where: str) -> None:
    """R-8: a rev-0 record holds none of the three ``Regime`` fields (the line BASE wrote)."""
    present = [f for f in NEW_REGIME_FIELDS if f in record["regime"]]
    assert not present, f"T-0213 R-8: {where} is a rev-0 line and carries {present}"


async def test_tools_off_golden_unchanged_at_rev_0(tmp_path):
    """A2r-5 under R-8: the T-0092b tools-off golden (captured at a502871): machine, HUD, prompts, request bodies and
    log records are identical as before (no settings state is published by a session), and every turn line equals the
    base's once the earlier declared additions (``tool``, ``regime.tools``, the T-0108/T-0127 extras) are taken out.
    The three ``Regime`` fields are **not** taken out: a rev-0 line carries none of them."""
    from tests.unit import test_session_tools_off_golden as g
    from tests.unit._t0177_pins import without_vision_frames

    require_regime_fields()
    golden = json.loads(g.GOLDEN.read_text(encoding="utf-8"))
    got = without_vision_frames(g.without_persona(await g.capture(tmp_path / "c")))
    assert not [
        m for m in got["hud"] if isinstance(m, dict) and m.get("type") == SETTINGS_STATE_TYPE
    ], "T-0213 A2-read: the session publishes no settings state without a set"
    for section in g.SECTIONS:
        assert got[section] == golden["sections"][section], (
            f"T-0213 A2r-5: the {section!r} section moved"
        )
    base = golden["sections"]["records"]
    assert len(got["records"]) == len(base)
    for i, (want, now) in enumerate(zip(base, got["records"], strict=True)):
        _assert_no_new_regime_fields(now, f"record {i}")
        stripped = {k: v for k, v in now.items() if k != "tool"}
        stripped["regime"] = {k: v for k, v in now["regime"].items() if k != "tools"}
        stripped["extras"] = {
            k: v for k, v in now["extras"].items() if k not in g.T0108_EXTRAS | g.T0127_EXTRAS
        }
        assert stripped == want, f"T-0213 A2r-5: record {i} differs by more than the declared keys"


async def test_tool_loop_golden_byte_identical_at_rev_0(tmp_path):
    """A2r-5 under R-8: the T-0114 loop-off scenario against the literal base d3bc6ce (run in a subprocess over an
    archive of the base tree, origin checked): every section — machine, HUD, prompts, requests, every session-log
    record and the turn lines — is identical to the base's with **nothing** taken out (a rev-0 turn line and its
    ``turn.record`` carry none of the three ``Regime`` fields)."""
    import subprocess
    import sys

    from tests import _git_range as gr
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._t0162c_pins import observed

    require_regime_fields()
    gr.require_commits(tsl.BASE_SHA)
    base_dir = tsl._base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", tsl._BASE_RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    base = json.loads(out.read_text())
    assert Path(base["origin"]).resolve().is_relative_to(base_dir.resolve()), base["origin"]
    now = observed(await tsl._loop_off_capture(tmp_path / "now"))
    assert now["records"], "premise: the scenario writes turn lines"
    for i, r in enumerate(now["records"]):
        _assert_no_new_regime_fields(r, f"line {i}")
    turn_records = [
        r["data"]
        for r in now["log"]
        if r["kind"] == "turn.record"
        and isinstance(r["data"], dict)
        and isinstance(r["data"].get("regime"), dict)
    ]
    for data in turn_records:
        _assert_no_new_regime_fields(data, "turn.record")
    for section in ("machine", "hud", "prompts", "requests", "log", "records"):
        assert now[section] == base["sections"][section], (
            f"T-0213 A2r-5 (R-8): with the loop off the {section!r} section differs from {tsl.BASE_SHA}"
        )


# ------------------------------------------------------------------------------------------------- A2r-1: the cap
#
# The rows below were added after the A2-read review (ten mutants survived the module); each names the mutant it
# kills. Values are typed from the criteria and the Profile field defaults, never read off the implementation.


@pytest.mark.parametrize(
    ("length", "accepted"),
    [(INSTRUCTIONS_MAX, True), (INSTRUCTIONS_MAX + 1, False)],
    ids=["at_cap", "over_cap"],
)
def test_the_instructions_cap_both_sides(length, accepted):
    """A2r-1 (both sides of the guard): ``instructions`` of exactly ``instructions_max`` code points decodes; one
    more is ``Malformed``. The text is astral (U+1F600: one code point, two UTF-16 units, four UTF-8 bytes), so a cap
    counted in bytes or UTF-16 units refuses the at-cap state. Kills ``>`` -> ``>=`` (at_cap) and a dropped check
    (over_cap)."""
    settings_state_class()
    raw = state_dict(instructions="\U0001f600" * length)
    got = decode(json.dumps(raw))
    if accepted:
        assert not isinstance(got, Malformed), (
            f"T-0213 A2r-1: {length} code points at a {INSTRUCTIONS_MAX} cap is legal; got {got}"
        )
        assert len(got.instructions) == length
    else:
        assert isinstance(got, Malformed), (
            f"T-0213 A2r-1: {length} code points over a {INSTRUCTIONS_MAX} cap is malformed; got {got!r}"
        )


@pytest.mark.parametrize("bad", [True, False, 1.0, "1"], ids=["true", "false", "float", "string"])
def test_rev_is_a_strict_int(bad):
    """A2r-1 (``rev: int >= 0``, strict like every T-0213 int): ``"rev": true`` (and false, 1.0, "1") decodes
    ``Malformed``. Kills a lax ``int`` that coerces a bool to 1."""
    settings_state_class()
    ok = decode(json.dumps(state_dict(rev=1)))
    assert not isinstance(ok, Malformed) and ok.rev == 1, f"control: rev = 1 decodes; got {ok}"
    got = decode(json.dumps(state_dict(rev=bad)))
    assert isinstance(got, Malformed), (
        f"T-0213 A2r-1: rev = {bad!r} is not an int and must decode Malformed; got {got!r}"
    )


# ------------------------------------------------------------------------------------------------- R-8: a changed turn


@pytest.mark.parametrize(
    ("change", "want_rev", "want_edited"),
    [({"rev": 1}, 1, False), ({"user_instructions": "Answer in one sentence."}, 0, True)],
    ids=["rev_1", "instructions_edited"],
)
async def test_a_changed_session_writes_the_three_fields(tmp_path, change, want_rev, want_edited):
    """A2r-4 under R-8: a tools-off session given a changed snapshot through ``init_settings`` (``rev=1``, or
    non-empty ``user_instructions``) before its turn opens writes a turn line carrying all three ``Regime`` fields:
    the rev, sha256[:12] of the system text the fake backend was handed, and whether instructions were set. Kills
    "drop the three fields from the session's regime update" (the line would carry none of them)."""
    from tests.unit import test_session_tools_off_golden as g

    require_regime_fields()
    async with g.golden_rig(tmp_path) as rig:
        session = rig.session
        session.init_settings(dataclasses.replace(session.settings, **change), {}, {})
        rig.llm.script = g.SPOKEN
        await rig.ask("what is on this screen")
        await rig.speak_out()
        await rig.pump()
        lines = rig.lines()
        prompts = [p for _, p in rig.llm.prefills] + [p for _, p, _ in rig.llm.generates]
    texts = {system_text(p) for p in prompts}
    assert len(texts) == 1 and len(lines) == 1, (len(texts), len(lines))
    regime = json.loads(lines[0])["regime"]
    present = [f for f in NEW_REGIME_FIELDS if f in regime]
    assert present == list(NEW_REGIME_FIELDS), (
        f"T-0213 R-8: a changed session ({change}) writes all three fields on its turn line; it wrote {present}"
    )
    want = {
        "settings_rev": want_rev,
        "system_sha12": sha12(texts.pop()),
        "instructions_edited": want_edited,
    }
    assert {f: regime[f] for f in NEW_REGIME_FIELDS} == want, regime


# ------------------------------------------------------------------------------------------------- A2r-2/R-9: per item


def _cli_graph(tmp_path: Path, capsys: Any, monkeypatch: Any, config: str, *flags: str) -> Any:
    """A real ``cli`` start over the fakes (nothing served) with ``config`` as the whole astra.toml and no flag but
    ``flags`` (unlike ``_cli_state``, no ``--tools``/``--browser`` is added); the built graph."""
    from tests.unit.test_cli_t0132 import FakeLlm, T, run_cli
    from tests.unit.test_cli_tools_live import Events, _out, fake_live

    settings_state_class()
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    monkeypatch.setattr(cli, "VllmClient", lambda *a, **k: FakeLlm())
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    path = tmp_path / "astra.toml"
    path.write_text(config)
    code = run_cli(T.session_argv(path, tmp_path, *flags))
    out, err = _out(capsys)
    assert code == 0 and calls.graphs, f"{flags} {config!r}: exit 0; got {code}\n{out}{err}"
    return calls.graphs[-1]


def _items_of(graph: Any) -> dict[str, Any]:
    states = settings_states(list(graph.server.hello_reply()))
    assert len(states) == 1, f"T-0213 A2r-3: one settings state in the hello; got {len(states)}"
    return {i.key: i for i in states[0].items}


def _cli_items(
    tmp_path: Path, capsys: Any, monkeypatch: Any, config: str, *flags: str
) -> dict[str, Any]:
    return _items_of(_cli_graph(tmp_path, capsys, monkeypatch, config, *flags))


#: a live profile with no ``profile =`` line (the engine's source is then ``default``)
LIVE_TOML = '[profiles.lean]\ntools = "live"\nbrowser = "headless"\n'
LIVE_ARGS = ("--tools", "live", "--browser", "headless")
BARE_TOML = "[profiles.lean]\n"


def _pair(items: dict[str, Any], key: str) -> tuple[Any, Any]:
    assert key in items, f"T-0213 R-9: the state reports {key}; keys {sorted(items)}"
    value = items[key].value
    return (list(value) if isinstance(value, (list, tuple)) else value), items[key].source


TOOLS_CASES = {
    "default": (BARE_TOML, (), ("off", "default")),
    "profile": (LIVE_TOML, (), ("live", "profile")),
    "flag": (BARE_TOML, LIVE_ARGS, ("live", "flag")),
}


@pytest.mark.parametrize("case", list(TOOLS_CASES))
def test_tools_value_and_source(tmp_path, capsys, monkeypatch, case):
    """R-9: ``tools`` is the resolved mode with its source: ``off``/``default`` with no tools key and no flag
    (``Profile.tools`` defaults to off), ``live``/``profile`` from astra.toml's ``tools = "live"``, ``live``/``flag``
    with ``--tools live``. Kills a source that reads "profile" whenever no flag was given (the default row)."""
    config, flags, want = TOOLS_CASES[case]
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    assert _pair(items, "tools") == want, (case, _pair(items, "tools"))


ENGINE_CASES = {
    "default": (LIVE_TOML, (), ("lean", "default")),
    "profile": ('profile = "lean"\n' + LIVE_TOML, (), ("lean", "profile")),
    "flag": (LIVE_TOML, ("--profile", "lean"), ("lean", "flag")),
}


@pytest.mark.parametrize("case", list(ENGINE_CASES))
def test_engine_value_and_source(tmp_path, capsys, monkeypatch, case):
    """R-9: ``engine`` is the profile the services run under: ``default`` with no ``profile =`` line and no
    ``--profile``, ``profile`` from astra.toml's ``profile = "lean"``, ``flag`` with ``--profile lean``; fixed for the
    session (``changeable`` False, with its relaunch command). Kills a source that reads only the flag (profile
    row)."""
    config, flags, want = ENGINE_CASES[case]
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    assert _pair(items, "engine") == want, (case, _pair(items, "engine"))
    assert items["engine"].changeable is False and items["engine"].relaunch, items["engine"]


def test_engine_source_of_the_shared_test_config_and_of_no_config(tmp_path, capsys, monkeypatch):
    """R-9 pin at activation: the shared test config (``_tools_switch_support.write_config``, which every
    ``_cli_state`` row starts from) writes ``profile = 'lean'`` on its first line, so ``engine`` there is
    ``("lean", "profile")``; an empty astra.toml (no ``profile =`` line, no ``[profiles]`` table, no flag) gives
    ``("lean", "default")``. ``lean`` is the default profile name (CLAUDE.md commands, ``--profile lean``). Kills a
    source pinned to ``default`` (the shared-config half) and one pinned to ``profile`` whenever a config file
    exists (the empty-file half)."""
    from tests.unit import _tools_switch_support as T

    for sub in ("premise", "shared", "empty"):
        (tmp_path / sub).mkdir()
    premise = T.write_config(tmp_path / "premise").read_text()
    assert premise.splitlines()[0] == "profile = 'lean'", (
        f"premise: the shared test config; got {premise!r}"
    )
    shared = _cli_state(tmp_path / "shared", capsys, monkeypatch)
    assert _pair(shared, "engine") == ("lean", "profile"), _pair(shared, "engine")
    empty = _cli_items(tmp_path / "empty", capsys, monkeypatch, "")
    assert _pair(empty, "engine") == ("lean", "default"), _pair(empty, "engine")


#: (config, flags, (key, value, source)); the defaults are ``Profile``'s field defaults (headed, 1280x800, web on)
BROWSER_CASES: dict[str, tuple[str, tuple[str, ...], tuple[str, Any, str]]] = {
    "browser-default": ('[profiles.lean]\ntools = "live"\n', (), ("browser", "headed", "default")),
    "browser-profile": (LIVE_TOML, (), ("browser", "headless", "profile")),
    "browser-flag": (
        '[profiles.lean]\ntools = "live"\n',
        ("--browser", "headless"),
        ("browser", "headless", "flag"),
    ),
    "viewport-default": (LIVE_TOML, (), ("browser_viewport", [1280, 800], "default")),
    "viewport-profile": (
        LIVE_TOML + "browser_viewport = [1024, 768]\n",
        (),
        ("browser_viewport", [1024, 768], "profile"),
    ),
    "web-default": (LIVE_TOML, (), ("browser_web", True, "default")),
    "web-profile": (LIVE_TOML + "browser_web = false\n", (), ("browser_web", False, "profile")),
    "web-flag": (LIVE_TOML, ("--no-web",), ("browser_web", False, "flag")),
}


@pytest.mark.parametrize("case", list(BROWSER_CASES))
def test_browser_rows_value_and_source(tmp_path, capsys, monkeypatch, case):
    """R-9: ``browser``, ``browser_viewport`` and ``browser_web`` in live report the value cli resolved and where it
    came from (the profile's own default, astra.toml, or a flag). Kills a source pinned to one answer for any of the
    three (each has a row per source it can take)."""
    config, flags, (key, value, source) = BROWSER_CASES[case]
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    assert _pair(items, key) == (value, source), (case, _pair(items, key))


#: (config, flags, {key: (value, source)}): with web on the budget is ``max_tool_steps_web`` (T-0178 A1, default
#: 20) and ``max_tool_steps`` has no value; with web off it is the other way round (default 8)
STEPS_CASES: dict[str, tuple[str, tuple[str, ...], dict[str, tuple[Any, str]]]] = {
    "web-default": (
        LIVE_TOML,
        (),
        {"max_tool_steps_web": (20, "default"), "max_tool_steps": (None, "default")},
    ),
    "web-profile": (
        LIVE_TOML + "max_tool_steps_web = 12\n",
        (),
        {"max_tool_steps_web": (12, "profile"), "max_tool_steps": (None, "default")},
    ),
    "web-flag": (
        LIVE_TOML,
        ("--max-tool-steps", "9"),
        {"max_tool_steps_web": (9, "flag"), "max_tool_steps": (None, "default")},
    ),
    "noweb-default": (
        LIVE_TOML,
        ("--no-web",),
        {"max_tool_steps": (8, "default"), "max_tool_steps_web": (None, "default")},
    ),
    "noweb-profile": (
        LIVE_TOML + "max_tool_steps = 5\n",
        ("--no-web",),
        {"max_tool_steps": (5, "profile"), "max_tool_steps_web": (None, "default")},
    ),
    "noweb-flag": (
        LIVE_TOML,
        ("--no-web", "--max-tool-steps", "9"),
        {"max_tool_steps": (9, "flag"), "max_tool_steps_web": (None, "default")},
    ),
}


@pytest.mark.parametrize("case", list(STEPS_CASES))
def test_step_budgets_value_and_source(tmp_path, capsys, monkeypatch, case):
    """A2r-2/R-9: the step budget in force is reported under the key of the budget that applies, which is decided
    by web (T-0178 A1), not by where the number came from: with web on ``--max-tool-steps 9`` is
    ``max_tool_steps_web`` (9, flag) and ``max_tool_steps`` is null. Kills a key chosen from the source (web-flag)
    and a source pinned to ``default`` (the profile rows)."""
    config, flags, want = STEPS_CASES[case]
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    got = {key: _pair(items, key) for key in want}
    assert {k: v[0] for k, v in got.items()} == {k: v[0] for k, v in want.items()}, (case, got)
    for key, (value, source) in want.items():
        if value is not None:
            assert got[key][1] == source, (case, key, got[key])


def test_context_helpers_follows_an_assist_toggle(tmp_path, capsys, monkeypatch):
    """R-9: ``context_helpers`` reads the assist state in force: (False, default) at start, (True, ui) after the
    Context helpers panel turns page text on (only the panel turns one on). Kills a source pinned to ``default``."""
    from local_astra.contracts.perception import AssistFlags
    from local_astra.contracts.protocol import AstraAssistToggle

    graph = _cli_graph(tmp_path, capsys, monkeypatch, LIVE_TOML)
    assert _pair(_items_of(graph), "context_helpers") == (False, "default")
    graph.session.apply_assist_toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
    assert _pair(_items_of(graph), "context_helpers") == (True, "ui"), _items_of(graph)[
        "context_helpers"
    ]


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        ((), (True, "default")),
        (("--wait-tool", "off"), (False, "flag")),
        (("--wait-tool", "on"), (True, "flag")),
    ],
    ids=["default", "flag-off", "flag-on"],
)
def test_wait_tool_value_and_source(tmp_path, capsys, monkeypatch, flags, want):
    """R-9: ``wait_tool`` in live is on by default (T-0202 D) and ``flag`` with ``--wait-tool``; ``flag-on`` has the
    default's value and differs only by the source. Kills a source pinned to ``default``."""
    items = _cli_items(tmp_path, capsys, monkeypatch, LIVE_TOML, *flags)
    assert _pair(items, "wait_tool") == want, _pair(items, "wait_tool")


@pytest.mark.parametrize(
    ("flags", "want"),
    [((), None), (("--reground",), 280)],
    ids=["reground-off", "reground-on"],
)
def test_reground_visual_tokens_only_with_reground_on(tmp_path, capsys, monkeypatch, flags, want):
    """A1's ruling under R-9: ``reground_visual_tokens`` is null with ReGround off (``grounding_of`` still resolves
    it to the main budget then), and 280 with ``--reground`` at the 1120 main budget (H-T133-1). Kills reporting the
    resolver's value whatever ReGround is."""
    items = _cli_items(tmp_path, capsys, monkeypatch, LIVE_TOML, *flags)
    assert _pair(items, "visual_tokens")[0] == 1120, "premise: the live main budget is 1120"
    assert _pair(items, "reground_visual_tokens")[0] == want, _pair(items, "reground_visual_tokens")


# ------------------------------------------------------------------------------------------------- changeable/relaunch


def _catalog() -> dict[str, Any]:
    return {e.key: e for e in CATALOG}


# T-0217 re-pin: R-27 (the tools row in live has no relaunch text)
def test_changeable_and_relaunch_inside_live(tmp_path, capsys, monkeypatch):
    """A2r-1/B-2 in live: every catalog row with ``v1_changeable`` is changeable with no relaunch command; every
    other row is not changeable, and each ``relaunch``-kind row carries a relaunch text (the text the panel shows
    with Copy; R-12 re-pin: its content, the flag to add, is pinned by the R-12 rows, no longer the catalog's whole
    command). Kills ``changeable`` pinned False (or True) and ``relaunch`` dropped. T-0217 re-pin (R-27): the
    ``tools`` row in live has no single flag for the other value, so its relaunch is None (kind still relaunch)."""
    items = _cli_items(tmp_path, capsys, monkeypatch, LIVE_TOML)
    catalog = _catalog()
    for key, item in items.items():
        entry = catalog[key]
        assert item.changeable is entry.v1_changeable, (key, item.changeable)
        if entry.v1_changeable:
            assert item.relaunch is None, (key, item.relaunch)
        if entry.kind == "relaunch":
            if key == "tools":  # T-0217 re-pin: R-27
                assert item.relaunch is None and item.kind == "relaunch", (key, item)
                continue
            assert item.relaunch and entry.relaunch, (key, item.relaunch)
    assert items["action_outcome"].changeable is True and items["browser"].changeable is False


@pytest.mark.parametrize("flags", [(), ("--tools", "dry-run")], ids=["off", "dry_run"])
def test_changeable_and_relaunch_outside_live(tmp_path, capsys, monkeypatch, flags):
    """A2r-1 outside live: a live-only setting (it has no value there) is not changeable, and every
    ``relaunch``-kind row is still read-only with its catalog command. Kills ``changeable = v1_changeable``
    regardless of the mode."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    catalog = _catalog()
    for key in (
        "action_outcome",
        "repeat_guard",
        "reground",
        "live_feedback",
        "screen_reminder",
        "completion_check",
    ):
        assert items[key].value is None, (
            f"premise: {key} has no value outside live; got {items[key].value!r}"
        )
        assert items[key].changeable is False, (
            f"T-0213 A2r-1: {key} does not apply outside live, so the panel cannot change it; got changeable"
        )
    for key, item in items.items():
        entry = catalog[key]
        if entry.kind == "relaunch":
            # R-12 re-pin: the text is the flag to add (pinned by the R-12 rows), no longer the catalog command
            assert item.changeable is False and item.relaunch, (key, item)
        if entry.kind == "fixed":
            assert item.changeable is False, key


# ------------------------------------------------------------------------------------------------- R-10: effective kind
#
# R-10 (amends A2r-1/B-1): each item carries its effective ``kind``: "relaunch" when it is not changeable in this
# session and a relaunch exists, "not_here" when it is not changeable and none exists, else the catalog kind. The rows
# below hand-type the expected kind per key from the ruling and the brief; none is computed from the catalog by a
# copy of the rule. Readings (reported): ``context_helpers`` keeps its catalog kind (it changes through the Context
# helpers panel, so the brief exempts it); the ``fixed`` rows (thinking, honest_ending_clips) are not pinned here.

#: the live-only rows the A2r-1 outside-live row already finds valueless outside live (hand list, same as there)
LIVE_ONLY = (
    "action_outcome",
    "repeat_guard",
    "reground",
    "live_feedback",
    "screen_reminder",
    "completion_check",
)


def test_r10_a_not_here_item_decodes():
    """R-10: ``not_here`` is a kind the wire carries (an item, not a catalog entry); it survives the codec."""
    state, _item, _entry = require_wire()
    body = state_dict(
        items=[item_dict(key="reground", value=None, kind="not_here", changeable=False)]
    )
    try:
        built = state.model_validate(body)
    except ValueError as err:  # pydantic's ValidationError is a ValueError
        raise AssertionError(f"T-0213 R-10: SettingItem refuses kind 'not_here': {err}") from err
    got = decode(encode(built))
    assert not isinstance(got, Malformed), got
    assert got.items[0].kind == "not_here", got.items[0]


@pytest.mark.parametrize(
    ("config", "flags"),
    [(BARE_TOML, LIVE_ARGS), (LIVE_TOML, ())],
    ids=["flag-live", "profile-live"],
)
def test_r10_effective_kind_inside_live(tmp_path, capsys, monkeypatch, config, flags):
    """R-10 in live (default flags otherwise): Page overview and the wait tool are fixed for the session (not
    changeable) and each has a relaunch, so their kind is ``relaunch`` (the catalog says next_turn_cold for both);
    ``context_helpers`` keeps its catalog kind ``next_turn``; every changeable item keeps its catalog kind. Kills the
    catalog kind passed through (page_overview/wait_tool) and an effective kind forced on every row (the changeable
    rows)."""
    items = _cli_items(tmp_path, capsys, monkeypatch, config, *flags)
    for key in ("page_overview", "wait_tool"):
        assert items[key].changeable is False, f"premise: {key} is fixed for a live session"
        assert items[key].kind == "relaunch", (
            f"T-0213 R-10: {key} is not changeable here and has a relaunch, so its kind is relaunch; "
            f"got {items[key].kind!r}"
        )
    assert items["context_helpers"].kind == "next_turn", items["context_helpers"]
    hand = {
        "action_outcome": "next_turn",
        "repeat_guard": "next_turn",
        "reground": "next_turn",
        "visual_tokens": "next_turn_cold",
        "live_guidance": "next_turn_cold",
        "user_instructions": "next_turn_cold",
    }
    for key, kind in hand.items():
        assert items[key].changeable is True, f"premise: {key} is changeable in live"
        assert items[key].kind == kind, (key, items[key].kind)


@pytest.mark.parametrize("flags", [(), ("--tools", "dry-run")], ids=["off", "dry_run"])
def test_r10_live_only_rows_are_not_here_outside_live(tmp_path, capsys, monkeypatch, flags):
    """R-10 outside live: a live-only row is not changeable and has no relaunch here, so its kind is ``not_here``
    (chip "doesn't apply in this mode"), never its catalog next-turn kind. Kills the catalog kind passed through
    outside live."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *flags)
    for key in LIVE_ONLY:
        assert items[key].changeable is False, f"premise: {key} is not changeable outside live"
        assert items[key].kind == "not_here", (
            f"T-0213 R-10: {key} does not apply outside live; got kind {items[key].kind!r}"
        )
    # R-10a: outside live every live-only row is not_here even when it has a command; tools carries --tools live
    for key in (
        "page_overview",
        "wait_tool",
        "browser",
        "browser_viewport",
        "browser_web",
        "max_tool_steps",
        "max_tool_steps_web",
        "action_echo",
    ):
        assert items[key].kind == "not_here", (
            f"T-0213 R-10a: {key} outside live; got {items[key].kind!r}"
        )
    assert items["tools"].relaunch == "--tools live", items["tools"].relaunch
    assert items["context_helpers"].kind == "next_turn", items["context_helpers"]
    assert items["thinking"].kind == "fixed" and items["honest_ending_clips"].kind == "fixed"


# ------------------------------------------------------------------------------------------------- R-12: flag to add
#
# R-12: the relaunch box shows only the flag(s) to add (the catalog ``flag``), so copying never drops the user's other
# flags; the engine row shows ``--profile <the profile in force>``. Reading (reported): the page shows
# ``item.relaunch`` verbatim (B-2, accepted), so the R-12 text is what the server sends as ``relaunch``. Exact for the
# switch row page_overview; a row whose flag takes a value is pinned to start with its catalog flag.


# T-0217 re-pin: criterion 1 (browser headless in force -> --browser headed)
def test_r12_relaunch_is_the_flag_to_add_inside_live(tmp_path, capsys, monkeypatch):
    """R-12 in live: Page overview's relaunch is exactly ``--page-overview``; the engine's is ``--profile lean``;
    every relaunch with a catalog flag starts with that flag; none holds ``diffusion_gearvis`` (a whole command,
    which drops the user's other flags when pasted). T-0217 re-pin (criterion 1): ``LIVE_ARGS`` starts the
    session ``--browser headless``, so the browser row's hint is ``--browser headed``."""
    items = _cli_items(tmp_path, capsys, monkeypatch, BARE_TOML, *LIVE_ARGS)
    assert items["page_overview"].relaunch == "--page-overview", items["page_overview"].relaunch
    # R-12 as amended: the catalog flag with the arguments of the server's command (a bare flag fails argparse)
    exact = {
        "wait_tool": "--wait-tool off",
        "action_echo": "--action-echo pixels",
        "max_tool_steps": "--max-tool-steps N",
        "max_tool_steps_web": "--max-tool-steps N",
        "engine": "--profile lean",
        "browser": "--browser headed",  # T-0217 re-pin: criterion 1
    }
    assert {k: items[k].relaunch for k in exact} == exact
    assert items["context_helpers"].kind == "next_turn"
    assert items["thinking"].kind == "fixed" and items["honest_ending_clips"].kind == "fixed"
    catalog = _catalog()
    problems = []
    for key, item in items.items():
        if not item.relaunch or key == "browser_viewport":  # flag "": set in astra.toml (reported)
            continue
        flag = catalog[key].flag
        if "diffusion_gearvis" in item.relaunch:
            problems.append(f"{key}: a whole command {item.relaunch!r}")
        if flag and not item.relaunch.startswith(flag):
            problems.append(f"{key}: {item.relaunch!r} does not start with the flag {flag!r}")
    assert problems == [], "T-0213 R-12:\n  " + "\n  ".join(problems)


def test_r12_engine_relaunch_is_the_profile_in_force(tmp_path, capsys, monkeypatch):
    """R-12: the engine row shows ``--profile <the profile in force>``: a session started with
    ``--profile quality-tts`` (its own profile table) shows ``--profile quality-tts``. Kills ``--profile lean``
    written as a literal."""
    config = '[profiles.quality-tts]\ntools = "live"\nbrowser = "headless"\n'
    items = _cli_items(tmp_path, capsys, monkeypatch, config, "--profile", "quality-tts")
    assert _pair(items, "engine") == ("quality-tts", "flag"), "premise: the profile in force"
    assert items["engine"].relaunch == "--profile quality-tts", items["engine"].relaunch


def test_r12_adversarial_page_overview_in_a_no_web_search_off_session(
    tmp_path, capsys, monkeypatch
):
    """R-12 adversarial: in a session started with ``--no-web --search off``, Page overview's relaunch is only
    ``--page-overview``: a command rebuilt from the catalog (``... --tools live --page-overview``) would, pasted,
    turn web and search back on. Kills any whole-command relaunch and one rebuilt from the session's flags."""
    items = _cli_items(
        tmp_path, capsys, monkeypatch, BARE_TOML, *LIVE_ARGS, "--no-web", "--search", "off"
    )
    assert _pair(items, "browser_web") == (False, "flag"), "premise: web is off by flag"
    assert items["page_overview"].relaunch == "--page-overview", items["page_overview"].relaunch
