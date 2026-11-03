"""T-0227 group A: images per turn and the screen-capture interval are settings (docs/tasks/T-0227.md A-1..A-10).

Every row is derived from the task's acceptance criteria and its "Tests to write" list; nothing is read off the
implementation (none exists at BASE 0a9882a3). **The surface these rows assume** (the task's names, plus readings
flagged in the test-author's report):

- A-1: ``config.FRAMES_PER_TURN_CHOICES``, ``DEFAULT_MAX_FRAMES_PER_TURN``, ``FRAME_INTERVALS_MS``,
  ``DEFAULT_FRAME_INTERVAL_MS``; ``Profile.max_frames_per_turn`` / ``Profile.frame_interval_ms`` (``int | None``).
  **Reading R-A1:** the task says "a ``*_refusal()`` string in the config module's style" without naming it; these
  rows look it up as ``config.max_frames_per_turn_refusal(value)`` and ``config.frame_interval_ms_refusal(value)``
  (the ``visual_tokens_refusal`` pattern: the key name plus ``_refusal``).
- A-3: ``LiveSettings.max_frames_per_turn`` / ``frame_interval_ms`` (dataclass fields, never ``None``).
- A-4: ``Session.init_settings(settings, sources, others)`` with a non-default ``max_frames_per_turn`` puts a builder
  in force whose ``budget == dataclasses.replace(Budget(), max_frames_per_turn=n)``; ``session.prompts.budget`` is
  the read seam (the name ``prompt_builder.PromptBuilder.budget`` exists at BASE).
- A-5: the set path is T-0213's (``astra.settings.set`` as raw JSON into a greeted page; the reply is an
  ``astra.settings.state``), driven through ``tests/unit/test_t0213_apply_boundary``'s rig, imported.

Rows that need the implementation were ``pending`` until activation (2026-11-02) and now fail on the missing feature. The A-8 rows
(the server keeps every frame at 250 ms; a slow cadence still answers ``get_frame``) are **green at BASE by
design**: the criterion says "there is no server-side frame rate limit and none is added" -- they are regression
guards for a no-change criterion and carry no marker.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, profile ``lean``; tools off / dry_run / live
(live kit over ``FakeLiveBrowser``, loop on); n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra import config
from local_astra.conductor.prompt_builder import Budget, PromptBuilder
from local_astra.config import Profile
from local_astra.contracts import events as E
from local_astra.contracts.llm import THINKING_TEMPLATE_KWARG, ImagePart, Prompt
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.contracts.turn import TurnState
from local_astra.perception import browser_source, policy
from local_astra.settings_catalog import CATALOG, RELAUNCH_IN_FORCE
from local_astra.telemetry.turns_log import Regime, TurnRecord

REPO = Path(__file__).resolve().parents[2]
VLLM_SH = REPO / "services" / "vllm.sh"

#: A-1, typed from the criterion
FRAMES_CHOICES = (1, 2, 3, 4)
INTERVALS = (250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000)
KEYS = ("max_frames_per_turn", "frame_interval_ms")
FLAGS = {"max_frames_per_turn": "--max-frames-per-turn", "frame_interval_ms": "--frame-interval-ms"}
DEFAULTS = {"max_frames_per_turn": 3, "frame_interval_ms": 1000}
CHOICES = {"max_frames_per_turn": FRAMES_CHOICES, "frame_interval_ms": INTERVALS}
#: the engine's image limit (``Budget.max_images``, ``services/vllm.sh``)
MAX_IMAGES = 4
MODES = ("off", "dry_run", "live")

# =====================================================================================================
# seams (named failures, never AttributeError/TypeError inside a row)
# =====================================================================================================


def cfg(name: str) -> Any:
    value = getattr(config, name, None)
    assert value is not None, f"T-0227 A-1: config.{name} does not exist yet"
    return value


def refusal(key: str) -> Any:
    fn = getattr(config, f"{key}_refusal", None)
    assert callable(fn), (
        f"T-0227 A-1 (reading R-A1): config.{key}_refusal(value) does not exist yet"
    )
    return fn


def require_live_fields() -> None:
    from local_astra.conductor.live_settings import LiveSettings

    names = {f.name for f in dataclasses.fields(LiveSettings)}
    missing = [k for k in KEYS if k not in names]
    assert not missing, f"T-0227 A-3: LiveSettings fields {missing} do not exist yet"


def with_cap(settings: Any, **changes: Any) -> Any:
    require_live_fields()
    return dataclasses.replace(settings, **changes)


def images(prompt: Prompt) -> int:
    return sum(1 for m in prompt.messages for p in m.content if isinstance(p, ImagePart))


def last_user_images(prompt: Prompt) -> list[ImagePart]:
    users = [m for m in prompt.messages if m.role == "user"]
    return [p for p in users[-1].content if isinstance(p, ImagePart)] if users else []


def entry(key: str) -> Any:
    found = [e for e in CATALOG if e.key == key]
    assert len(found) == 1, (
        f"T-0227 A-2: the catalog has one {key!r} row; keys {[e.key for e in CATALOG]}"
    )
    return found[0]


def image_limit_from_vllm_sh() -> int:
    text = VLLM_SH.read_text(encoding="utf-8")
    m = re.search(r"--limit-mm-per-prompt\s+'\{\\?\"image\\?\"\s*:\s*(\d+)", text)
    assert m is not None, (
        "premise: services/vllm.sh passes --limit-mm-per-prompt with an image count"
    )
    return int(m.group(1))


# =====================================================================================================
# A-1: constants and Profile keys
# =====================================================================================================


def test_vllm_sh_image_limit_premise() -> None:
    """Premise (green at BASE): the launcher's image limit is 4, and so is ``Budget.max_images``."""
    assert image_limit_from_vllm_sh() == MAX_IMAGES == Budget().max_images


def test_constants_parity() -> None:
    """A-1's five relations (and the values typed from the criterion)."""
    assert cfg("FRAMES_PER_TURN_CHOICES") == FRAMES_CHOICES
    assert cfg("DEFAULT_MAX_FRAMES_PER_TURN") == 3
    assert cfg("FRAME_INTERVALS_MS") == INTERVALS
    assert cfg("DEFAULT_FRAME_INTERVAL_MS") == 1000
    assert cfg("DEFAULT_MAX_FRAMES_PER_TURN") == Budget().max_frames_per_turn
    assert max(cfg("FRAMES_PER_TURN_CHOICES")) == Budget().max_images == image_limit_from_vllm_sh()
    assert cfg("DEFAULT_FRAME_INTERVAL_MS") == 1000 / policy.MAX_FPS
    ring_ms = policy.RING_BUFFER_S * 1000
    assert min(cfg("FRAME_INTERVALS_MS")) * browser_source.MAX_RING_FRAMES >= ring_ms, (
        "A-1: the frame cap never cuts the 60 s ring at the fastest interval"
    )
    assert max(cfg("FRAME_INTERVALS_MS")) < ring_ms
    for name in ("FRAMES_PER_TURN_CHOICES", "FRAME_INTERVALS_MS"):
        assert all(type(v) is int for v in cfg(name)), f"{name} holds ints"


BAD_VALUES: list[Any] = [True, 3.0, "3", 0, 5, 100, 249]
#: the interval also refuses a float or a string of an allowed value (the strictness, aimed at its own range)
BAD_INTERVAL_EXTRA: list[Any] = [1000.0, "1000", False, 251, 10001, -250]


@pytest.mark.parametrize(
    ("key", "bad"),
    [(k, v) for k in KEYS for v in BAD_VALUES]
    + [("frame_interval_ms", v) for v in BAD_INTERVAL_EXTRA]
    + [("max_frames_per_turn", v) for v in (4.0, "4", False, -1)],
    ids=lambda x: repr(x),
)
def test_profile_strictness_refuses(key: str, bad: Any) -> None:
    """A-1: ``type(v) is int and v in choices``; anything else is refused with config's own refusal text."""
    assert key in Profile.model_fields, f"T-0227 A-1: Profile.{key} does not exist yet"
    for mode in MODES:
        with pytest.raises(ValidationError) as info:
            Profile.model_validate({"name": "lean", "tools": mode, key: bad})
        assert refusal(key)(bad) in str(info.value), (
            f"A-1: {key} = {bad!r} ({mode}) is refused with config's {key}_refusal text; got {info.value}"
        )


@pytest.mark.parametrize("key", KEYS)
def test_profile_accepts_every_choice_in_every_mode(key: str) -> None:
    """A-1: every allowed value loads in off, dry_run and live; the key absent is ``None``."""
    assert key in Profile.model_fields, f"T-0227 A-1: Profile.{key} does not exist yet"
    assert Profile.model_fields[key].default is None, f"A-1: Profile.{key} defaults to None"
    for mode in MODES:
        for value in CHOICES[key]:
            got = Profile.model_validate({"name": "lean", "tools": mode, key: value})
            assert getattr(got, key) == value and type(getattr(got, key)) is int


def test_refusal_texts_name_the_key_and_the_choices() -> None:
    """A-1 (the config module's style): the refusal names the key, the value and the allowed values."""
    t = refusal("max_frames_per_turn")(7)
    assert "max_frames_per_turn" in t and "7" in t and "is refused" in t and "4" in t, t
    t = refusal("frame_interval_ms")(249)
    assert "frame_interval_ms" in t and "249" in t and "is refused" in t and "10000" in t, t


# =====================================================================================================
# A-2: catalog rows
# =====================================================================================================


def test_catalog_rows_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A-2: two rows in "Seeing the screen", directly after ``visual_tokens``, in this order, with the stated kind,
    values, default, flag and profile key; ``relaunch is None``; neither in ``RELAUNCH_IN_FORCE``; each flag is a
    real ``session`` option whose argparse ``choices`` are the config constant."""
    from tests.unit.test_t0213_catalog import _session_parser

    keys = [e.key for e in CATALOG]
    assert "max_frames_per_turn" in keys, (
        "T-0227 A-2: catalog row max_frames_per_turn does not exist yet"
    )
    i = keys.index("visual_tokens")
    assert keys[i + 1 : i + 3] == list(KEYS), f"A-2: the two rows follow visual_tokens; {keys}"
    want = {
        "max_frames_per_turn": ("Screens per turn (max images)", "next_turn_cold"),
        "frame_interval_ms": ("Screen capture interval (ms)", "next_turn"),
    }
    parser = _session_parser(monkeypatch)
    actions = {s: a for a in parser._actions for s in a.option_strings}
    for key, (label, kind) in want.items():
        e = entry(key)
        assert (e.label, e.group, e.kind, e.v1_changeable) == (
            label,
            "Seeing the screen",
            kind,
            True,
        ), e
        assert (
            e.values
            == CHOICES[key]
            == cfg(
                "FRAMES_PER_TURN_CHOICES" if key == "max_frames_per_turn" else "FRAME_INTERVALS_MS"
            )
        )
        assert e.default == DEFAULTS[key] and type(e.default) is int, e.default
        assert (e.flag, e.profile_key, e.relaunch) == (FLAGS[key], key, None), e
        assert key not in RELAUNCH_IN_FORCE
        assert FLAGS[key] in actions, f"A-4: session {FLAGS[key]} does not exist yet"
        action = actions[FLAGS[key]]
        assert tuple(action.choices or ()) == CHOICES[key], (
            f"A-4: {FLAGS[key]} uses argparse choices from the config constant; got {action.choices}"
        )
        assert action.type is int or all(type(c) is int for c in action.choices), action


def test_catalog_help_states_costs() -> None:
    """A-2: the frames row states the ceiling of 4, the per-screen cost (about 1,094 at 1120, about 275 at 280), that
    at 4 no earlier screen stays an image, and that the latency cost is unmeasured; the interval row says it is the
    shared screen and camera, not the agent browser, with the page CPU / ring / log cost and the staleness."""
    f = entry("max_frames_per_turn")
    text = " ".join((f.what, f.useful_when, f.not_useful_when, f.evidence, f.cost))
    assert re.search(r"\b4\b", text) and re.search(r"1,?094", text) and "275" in text, text
    assert re.search(r"(?i)unmeasured|not (yet )?measured", text), text
    assert re.search(r"(?i)earlier|history|previous", text), text
    assert "T-0131" in text and "T-0059" in text, text
    v = entry("frame_interval_ms")
    text = " ".join((v.what, v.useful_when, v.not_useful_when, v.evidence, v.cost))
    low = text.lower()
    assert "shared screen" in low and "camera" in low and "agent browser" in low, text
    assert "180" in text and "42" in text, text
    assert re.search(r"(?i)\bold\b|stale|older", text), text


# =====================================================================================================
# A-3: validation
# =====================================================================================================


@pytest.mark.parametrize("mode", MODES)
def test_validate_accepts_in_every_mode(mode: str) -> None:
    """A-3: both keys apply in off, dry_run and live; a change is rev + 1; a no-op set keeps the rev; a bad value is
    refused with config's text; neither field is ever None."""
    from local_astra.conductor import live_settings as ls

    require_live_fields()
    cur = ls.LiveSettings(rev=2)
    assert (cur.max_frames_per_turn, cur.frame_interval_ms) == (3, 1000)
    web = mode == "live"
    new, refusals = ls.validate(
        cur, {"max_frames_per_turn": 4, "frame_interval_ms": 250}, mode=mode, web=web
    )
    assert new is not None and not refusals, refusals
    assert (new.max_frames_per_turn, new.frame_interval_ms, new.rev) == (4, 250, 3)
    same, refusals = ls.validate(
        cur, {"max_frames_per_turn": 3, "frame_interval_ms": 1000}, mode=mode, web=web
    )
    assert same is not None and same.rev == 2 and not refusals
    for key, bad in (
        ("max_frames_per_turn", 5),
        ("frame_interval_ms", "500"),
        ("max_frames_per_turn", None),
    ):
        got, refusals = ls.validate(cur, {key: bad}, mode=mode, web=web)
        assert got is None and refusals, (key, bad)
        if bad is not None:
            assert any(
                refusal(key)(bad).split(" is refused", 1)[1] in r.reason for r in refusals
            ), refusals


# =====================================================================================================
# A-4: launch (cli)
# =====================================================================================================


def _items(
    tmp_path: Path, capsys: Any, monkeypatch: Any, config_text: str, *flags: str
) -> dict[str, Any]:
    from tests.unit.test_t0213_settings_wire import _cli_items

    return _cli_items(tmp_path, capsys, monkeypatch, config_text, *flags)


LIVE_TOML = '[profiles.lean]\ntools = "live"\nbrowser = "headless"\n'
BARE_TOML = "[profiles.lean]\n"


@pytest.mark.parametrize("mode_toml", [BARE_TOML, LIVE_TOML], ids=["off", "live"])
@pytest.mark.parametrize(
    ("case", "flags", "body", "want"),
    [
        (
            "flag",
            ("--max-frames-per-turn", "2", "--frame-interval-ms", "500"),
            "",
            ((2, "flag"), (500, "flag")),
        ),
        (
            "profile",
            (),
            "max_frames_per_turn = 4\nframe_interval_ms = 2000\n",
            ((4, "profile"), (2000, "profile")),
        ),
        ("default", (), "", ((3, "default"), (1000, "default"))),
        (
            "interval-flag-over-profile",
            ("--frame-interval-ms", "250"),
            "frame_interval_ms = 5000\n",
            ((3, "default"), (250, "flag")),
        ),
        (
            "flag-over-profile",
            ("--max-frames-per-turn", "1"),
            "max_frames_per_turn = 4\nframe_interval_ms = 5000\n",
            ((1, "flag"), (5000, "profile")),
        ),
    ],
)
def test_resolver_sources(
    tmp_path, capsys, monkeypatch, mode_toml, case, flags, body, want
) -> None:
    """A-4/A-9: flag > profile > default, for both keys, in a tools-off and a live session; the state items carry the
    value and source, ``changeable`` true and the catalog kind, in every mode."""
    missing = [k for k in KEYS if k not in Profile.model_fields]
    assert not missing, f"T-0227 A-1: Profile fields {missing} do not exist yet"
    items = _items(tmp_path, capsys, monkeypatch, mode_toml + body, *flags)
    for key, (value, source) in zip(KEYS, want, strict=True):
        assert key in items, (
            f"T-0227 A-9: the settings state has an item for {key}; keys {sorted(items)}"
        )
        got = items[key]
        assert (got.value, got.source) == (value, source), f"A-4 ({case}) {key}: {got}"
        assert got.changeable is True and got.kind == entry(key).kind and got.relaunch is None, got


@pytest.mark.parametrize(
    "bad",
    [
        ("--max-frames-per-turn", "5"),
        ("--frame-interval-ms", "1001"),
        ("--max-frames-per-turn", "0"),
    ],
)
def test_flag_outside_the_choices_exits_2(tmp_path, capsys, monkeypatch, bad) -> None:
    """A-4: argparse ``choices`` refuse a value outside the constant (exit 2); an allowed one starts (control)."""
    from tests.unit.test_cli_t0132 import T, run_cli
    from tests.unit.test_t0213_settings_wire import _cli_graph

    _cli_graph(
        tmp_path / "ok",
        capsys,
        monkeypatch,
        BARE_TOML,
        bad[0],
        "2" if "frames" in bad[0] else "2000",
    )
    path = tmp_path / "astra.toml"
    path.write_text(BARE_TOML)
    code = run_cli([*T.session_argv(path, tmp_path, *bad), "--dry-run"])
    capsys.readouterr()
    assert code == 2, f"A-4: {bad} exits 2; got {code}"


def _graph(tmp_path: Path, capsys: Any, monkeypatch: Any, config_text: str, *flags: str) -> Any:
    from tests.unit.test_t0213_settings_wire import _cli_graph

    return _cli_graph(tmp_path, capsys, monkeypatch, config_text, *flags)


@pytest.mark.parametrize("toml", [BARE_TOML, LIVE_TOML], ids=["off", "live"])
def test_cli_never_touches_turn_policy(tmp_path, capsys, monkeypatch, toml) -> None:
    """A-4: the ``TurnPolicy`` repr is equal at caps 1, 4 and no flag (the machine's cap stays 3), while the builder's
    ``Budget`` follows the flag with every other field unchanged."""
    reprs, budgets = {}, {}
    for name, flags in (
        ("none", ()),
        ("1", ("--max-frames-per-turn", "1")),
        ("4", ("--max-frames-per-turn", "4")),
    ):
        graph = _graph(tmp_path / name, capsys, monkeypatch, toml, *flags)
        reprs[name] = repr(graph.session.machine.policy)
        budgets[name] = graph.session.prompts.budget
    assert reprs["1"] == reprs["4"] == reprs["none"], reprs
    assert graph.session.machine.policy.max_frames_per_turn == 3
    assert budgets["none"] == Budget(), budgets["none"]
    assert budgets["1"] == dataclasses.replace(Budget(), max_frames_per_turn=1), budgets["1"]
    assert budgets["4"] == dataclasses.replace(Budget(), max_frames_per_turn=4), budgets["4"]


def test_init_settings_at_the_default_rebuilds_nothing(tmp_path, capsys, monkeypatch) -> None:
    """A-4: "At the default nothing is rebuilt": ``init_settings`` leaves the very builder object the constructor
    made in force when the cap is 3 (with or without the explicit flag); a cap of 2 replaces it before the first
    turn."""
    from local_astra.conductor import session as session_mod

    real = session_mod.Session.init_settings
    seen: list[tuple[Any, Any]] = []

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        before = self.prompts
        out = real(self, *a, **k)
        seen.append((before, self.prompts))
        return out

    monkeypatch.setattr(session_mod.Session, "init_settings", spy)
    for name, flags in (
        ("none", ()),
        ("3", ("--max-frames-per-turn", "3")),
        ("2", ("--max-frames-per-turn", "2")),
    ):
        seen.clear()
        _graph(tmp_path / name, capsys, monkeypatch, LIVE_TOML, *flags)
        assert len(seen) == 1, f"premise: cli calls init_settings once; {len(seen)}"
        before, after = seen[0]
        if name == "2":
            assert after is not before and after.budget.max_frames_per_turn == 2
            assert after.system_message().text() == before.system_message().text()
        else:
            assert after is before, f"A-4: at the default ({name}) nothing is rebuilt"


# =====================================================================================================
# the cap on a voice turn (tools off): A-4 init path, T-0059 arithmetic
# =====================================================================================================


def hud(ts_ms: int, tag: str) -> Frame:
    from tests.unit.test_session_loop import frame

    return frame(ts_ms, tag)


async def _five_change_turn(rig: Any) -> list[Frame]:
    """Onset (frame a), five distinct scene changes 1.2 s apart, words, end of turn; the frames in order."""
    await rig.onset()
    out: list[Frame] = []
    for tag in ("b", "c", "d", "e", "f"):
        await rig.advance(1_200, step=200)
        fr = hud(rig.clock.now_ms, tag)
        rig.perception.add(fr)
        rig.submit(
            E.SceneChanged(t_ms=rig.clock.now_ms, source=Source.SCREEN.value, phash=fr.phash)
        )
        await rig.pump()
        out.append(fr)
    await rig.words("which of these changed")
    await rig.end_of_turn()
    await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
    return out


#: T-0059 D1 by hand: the onset frame a, changes b..f; append under the cap, then replace the last slot
KEPT = {1: ["f"], 2: ["a", "f"], 3: ["a", "b", "f"], 4: ["a", "b", "c", "f"]}


@pytest.mark.parametrize("n", FRAMES_CHOICES)
async def test_cap_n_bounds_current_images(tmp_path, n) -> None:
    """A-4/A-6 ("with cap n, a turn's final user message carries at most n current-turn images"): a tools-off voice
    turn with five distinct scene changes, cap n through ``init_settings`` (as ``cli`` gives it), sends exactly the
    hand-computed frames, oldest first (T-0059 D1: under the cap append, then the last slot is live); the turn line's
    ``frames`` agrees. At n = 1 the request is byte-identical to a session whose builder is the T-0059 rollback
    (``Budget(max_frames_per_turn=1)``) built by hand."""
    from tests.unit.test_session_loop import answer, running

    onset = hud(0, "a")
    async with running(tmp_path / "cap", script=[answer("Yes. ")], frames=[onset]) as rig:
        rig.session.init_settings(with_cap(rig.session.settings, max_frames_per_turn=n), {}, {})
        assert rig.session.prompts.budget == dataclasses.replace(Budget(), max_frames_per_turn=n)
        changes = await _five_change_turn(rig)
        by_tag = {"a": onset, **{fr.phash[0]: fr for fr in changes}}
        _, request = rig.llm.gen_prompts[0]
        want = [by_tag[t].phash for t in KEPT[n]]
        assert [p.phash for p in last_user_images(request)] == want, (
            f"A-6: cap {n} keeps {KEPT[n]}; got {[p.phash for p in last_user_images(request)]}"
        )
        assert len(last_user_images(request)) <= n
        assert images(request) <= MAX_IMAGES
        line = next(r for r in rig.records() if r.turn_id == "t1")
        assert line.extras["frames"] == want and line.extras["frames_kept"] == len(want)
        mine = request.model_dump_json()
    if n == 1:
        async with running(tmp_path / "rollback", script=[answer("Yes. ")], frames=[onset]) as rig:
            old = rig.session.prompts
            rig.session.prompts = PromptBuilder(
                old.system_prompt,
                Budget(max_frames_per_turn=1),
                old.thinking,
                tools=old.tools,
                tool_guidance=old.tool_guidance,
                ui_block=old.ui_block,
                user_instructions=old.user_instructions,
            )
            await _five_change_turn(rig)
            _, rollback = rig.llm.gen_prompts[0]
        assert mine == rollback.model_dump_json(), (
            "A-6: cap 1 is the T-0059 rollback, byte for byte"
        )


# =====================================================================================================
# A-6 (adversarial, most likely failure): no request over the engine ceiling
# =====================================================================================================

PATHS = (
    "tool_loop",
    "overview_frame",
    "auto_overview",
    "reground_280",
    "reground_1120",
    "completion_check",
)
#: deals.html at 1280x800 (1.525 screens, the overview frame's range) and T-0140's 9.3-screen page
DEALS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1220.0}
TALL = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 7440.0}


def _path_kwargs(path: str) -> tuple[dict[str, Any], bool | None]:
    if path == "overview_frame":
        return {"overview_frame": True}, None
    if path == "auto_overview":
        return {"auto_overview": True}, None
    if path.startswith("reground"):
        return {"reground_visual_tokens": int(path.split("_")[1]), "visual_tokens": 1120}, True
    if path == "completion_check":
        return {"completion_check": True}, None
    return {}, None


def _every_request(llm: Any) -> list[tuple[str, str, Prompt]]:
    out = [("prefill", rid, p) for rid, p in llm.prefills]
    out += [("generate", rid, p) for rid, p, _ in llm.generates]
    # ReGround's pass 2 (``make_reground_llm`` serves it outside ``generates``): its crops count too
    out += [("pass2", rid, p) for rid, p in getattr(llm, "pass2_prompts", [])]
    return out


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("cap", [3, 4])
async def test_no_request_over_the_engine_ceiling(tmp_path, monkeypatch, cap, path) -> None:
    """A-6 (adversarial): a live session (``FakeLiveBrowser``, loop on) at cap 3 or 4, two earlier turns leaving two
    young history images, then a voice turn that sees four HUD scene changes and runs (a) a three-step tool loop,
    (b) the overview frame or the auto overview on a long page, (c) ReGround with crop 280 or 1120, or (d) the
    completion check. Every request the fakes record -- prefills, step 1, steps, overview, ReGround pass 2, the
    check, captions -- carries at most ``Budget.max_images`` (4) ``ImagePart``s. At cap 4, an overview path whose
    step 1 already holds 4 images is refused (``refused`` counted, nothing shown), never a fifth image."""
    from local_astra.computer import executor as executor_mod
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._live_browser_support import (
        FakeLiveBrowser,
        jpeg_frame,
        reground_rig,
        turn_extras,
    )
    from tests.unit._tool_turn_support import VALID_CLICK, answer, tool

    require_live_fields()
    # R-3 (review finding 2): the fake's post-action frames are uniform JPEGs, which T-0200's blank check drops, so
    # the completion check had no frame (outcome no_frame); every frame is read as not blank here
    monkeypatch.setattr(executor_mod, "blank_frame", lambda jpeg: False)
    kwargs, reground = _path_kwargs(path)
    browser = FakeLiveBrowser(
        [jpeg_frame(f"t0227-{k}") for k in range(3)],
        post_frames=[jpeg_frame(f"t0227-after-{k}") for k in range(8)],
    )
    if path in ("overview_frame", "auto_overview"):
        from tests.unit.test_t0177_shared_frame import OV

        browser.overview_image = OV
        browser.page_metrics = dict(DEALS if path == "overview_frame" else TALL)
    pass2 = None
    if reground:
        from tests.unit._live_browser_support import make_reground_llm

        pass2 = make_reground_llm(browser)  # ReGround's own client: its requests count too
        kwargs = {**kwargs, "reground_llm": pass2}
    async with reground_rig(tmp_path, browser, reground=reground, session_kwargs=kwargs) as rig:
        rig.session.init_settings(with_cap(rig.session.settings, max_frames_per_turn=cap), {}, {})
        llm = rig.llm
        llm.script_generates(
            [answer("One. ")], [answer("Two. ")], [answer(tsl.S1), tool(VALID_CLICK)]
        )
        steps = (
            [[tool(VALID_CLICK)], [tool(VALID_CLICK)], [answer(tsl.S3)]]
            if path == "tool_loop"
            else [[answer(tsl.S3)]]
        )
        llm.script_steps(*steps)
        if path == "completion_check":
            llm.script_checks([answer("The page shows it, so it is done.")])
        for question in ("what is this page", "and now"):
            await rig.ask(question)
            await rig.until(lambda: rig.state.state is TurnState.IDLE, "a history turn never ended")
            await rig.advance(500)
        await rig.onset()
        # ReGround: with HUD scene changes in the turn no pass 2 is sent (requests 0, observed at db948dd2), so its
        # paths keep the turn's own screen to the browser's and pass 2 really runs (R-3 premise below)
        for k in range(0 if reground else 4):
            await rig.advance(1_200, step=200)
            fr = hud(rig.clock.now_ms, "pqrs"[k])
            rig.perception.add(fr)
            rig.submit(
                E.SceneChanged(t_ms=rig.clock.now_ms, source=Source.SCREEN.value, phash=fr.phash)
            )
            await rig.pump()
        await rig.words("open the settings menu")
        await rig.end_of_turn()
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 3,
            "the third turn never ended",
            limit_ms=60_000,
        )
        await rig.advance(2_000)
        requests = _every_request(llm) + (_every_request(pass2) if pass2 is not None else [])
        extras3 = dict(rig.record("t3").extras)
        checks3 = [rid for rid, _ in llm.check_prompts if "-t3-" in rid]
    assert requests, "premise: requests were recorded"
    # R-3 (review finding 2): each path's own request really exists, so the ceiling is checked on it
    if path == "completion_check":
        assert checks3, (
            f"premise: t3 sent a completion-check request; {extras3.get('completion_check')}"
        )
        assert (extras3.get("completion_check") or {}).get("outcome") != "no_frame", extras3
    if reground:
        assert (extras3.get("reground") or {}).get("requests", 0) >= 1, (
            f"premise: ReGround pass 2 ran in t3; {extras3.get('reground')}"
        )
        assert any(kind == "pass2" for kind, _, _ in requests), (
            "premise: the pass-2 request is among the counted ones"
        )
    if path == "tool_loop":
        assert extras3.get("tool_steps") == 4, (
            f"premise: three executed clicks and a report; {extras3}"
        )
    if path in ("overview_frame", "auto_overview"):
        decided = extras3.get(path) or {}
        assert sum(decided.values()) >= 1, f"premise: the {path} decision was made in t3; {decided}"
    over = [(kind, rid, images(p)) for kind, rid, p in requests if images(p) > MAX_IMAGES]
    assert not over, (
        f"A-6: requests over the engine's {MAX_IMAGES}-image ceiling at cap {cap} ({path}): {over}"
    )
    t3 = [(rid, p) for kind, rid, p in requests if "-t3-" in rid and kind == "generate"]
    assert t3, f"premise: t3 sent requests; ids {[r for _, r, _ in requests]}"
    if path in ("overview_frame", "auto_overview") and cap == 4:
        extras = turn_extras(rig, "t3")
        counts = extras.get(path) or {}
        step1 = [p for rid, p in t3 if rid.endswith("-t3-gen") or "-t3-gen-" not in rid]
        if step1 and max(images(p) for p in step1) >= MAX_IMAGES:
            assert counts.get("refused", 0) >= 1 and counts.get("shown", 0) == 0, (
                f"A-6: at cap 4 with a full step 1 the {path} path is refused, not a fifth image; {counts}"
            )


# =====================================================================================================
# A-5: applying at the boundary (adversarial: a mid-turn set waits)
# =====================================================================================================


def _turn_dumps(llm: Any, turn: str, sid: str) -> list[str]:
    out = [f"prefill {rid} {p.model_dump_json()}" for rid, p in llm.prefills if f"-{turn}-" in rid]
    out += [
        f"generate {rid} {p.model_dump_json()} {o.model_dump_json()}"
        for rid, p, o in llm.generates
        if f"-{turn}-" in rid
    ]
    return [s.replace(sid, "<sid>") for s in out]


async def _run_window(tmp_path: Path, window: str, send: bool) -> tuple[list[str], Any, list[Any]]:
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit import test_t0213_apply_boundary as ab

    gate = asyncio.Event()
    async with ab._window_rig(tmp_path, window, gate) as (rig, llm, browser):
        if window != "prefill":
            llm.script_steps([ab.answer(tsl.S3)])
            if window == "completion_check":
                llm.script_checks([ab.answer(ab.CHECK_SAYS)])
        budget_before = rig.session.prompts.budget
        await ab._open_window(rig, llm, browser, window, gate)
        replies: list[Any] = []
        if send:
            mark = await ab.set_and_pump(rig, {"max_frames_per_turn": 1})
            reply = ab.last_state(rig, mark, " (mid-turn)")
            assert not reply.refused, (
                f"T-0227 A-3: max_frames_per_turn is a valid set; {reply.refused}"
            )
            it = ab.item(reply, "max_frames_per_turn")
            assert (it.value, it.pending) == (3, 1), f"A-5: mid-turn the cap is pending; {it}"
            assert rig.session.prompts.budget == budget_before, (
                "A-5: the builder is not replaced mid-turn"
            )
            replies.append(reply)
        await ab._close_window(rig, llm, window, gate)
        if send:
            after = ab.last_state(rig, 0, " (after the turn)")
            assert after.rev == 1 and ab.item(after, "max_frames_per_turn").value == 1, after
            assert rig.session.prompts.budget == dataclasses.replace(
                budget_before, max_frames_per_turn=1
            ), "A-5: the next builder is replace(old.budget, max_frames_per_turn=1)"
            assert repr(rig.session.machine.policy) == repr(rig.machine.policy)
            assert rig.session.machine.policy.max_frames_per_turn == 3, (
                "A-5: TurnPolicy is never changed"
            )
        dumps = _turn_dumps(llm, "t1", rig.session.session_id)
        # t2: a voice turn that sees three distinct HUD screens; its requests are capped at 1 after the set
        await rig.onset()
        for tag in ("x", "y", "z"):
            await rig.advance(1_200, step=200)
            fr = hud(rig.clock.now_ms, tag)
            rig.perception.add(fr)
            rig.submit(
                E.SceneChanged(t_ms=rig.clock.now_ms, source=Source.SCREEN.value, phash=fr.phash)
            )
            await rig.pump()
        rig.llm.script = [ab.answer(ab.ANSWER_2)]
        await rig.words("and what does it say now")
        await rig.end_of_turn()
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 2,
            "t2 never ended",
            limit_ms=40_000,
        )
        t2 = [p for rid, p in llm.prefills if "-t2-" in rid] + [
            p for rid, p, _ in llm.generates if "-t2-" in rid
        ]
    return dumps, t2, replies


@pytest.mark.parametrize("window", ["prefill", "between_steps", "completion_check"])
async def test_mid_turn_cap_change_waits(tmp_path, window) -> None:
    """A-5 (adversarial; T-0213 A2a-3): a set from 3 to 1 arriving after the prefill opened, between tool steps 1 and
    2, or during the completion check leaves every request of that turn byte-identical to the same turn with no set
    (a differential: prefills, generates, options); the builder changes only at the boundary; the next turn's
    requests carry at most one current-turn image."""
    require_live_fields()
    control, _, _ = await _run_window(tmp_path / "control", window, send=False)
    changed, t2, _ = await _run_window(tmp_path / "set", window, send=True)
    assert control and changed == control, (
        f"A-5: every t1 request keeps its 3-frame body byte for byte ({window}); "
        f"{len(changed)} vs {len(control)} requests"
    )
    assert t2, "premise: t2 sent requests"
    worst = max(len(last_user_images(p)) for p in t2)
    assert worst <= 1, f"A-5: the next turn's requests are capped at 1 current image; got {worst}"


async def test_apply_with_guidance_keeps_budget(tmp_path) -> None:
    """A-5: one set changes ``live_guidance`` and ``max_frames_per_turn``: the builder has both (the v2 guidance and
    cap 2). A following guidance-only set keeps cap 2; a following cap-only set keeps the guidance."""
    from local_astra.cli import guidance_table_of
    from tests.unit import test_t0213_apply_boundary as ab

    require_live_fields()
    table = guidance_table_of(False, False)
    async with ab.settings_rig(tmp_path, guidance_table=table) as (rig, _llm, _browser):
        base = rig.session.prompts.budget
        mark = await ab.set_and_pump(rig, {"live_guidance": "v2", "max_frames_per_turn": 2})
        state = ab.last_state(rig, mark)
        assert state.rev == 1 and not state.refused, state.refused
        p = rig.session.prompts
        assert p.budget == dataclasses.replace(base, max_frames_per_turn=2), p.budget
        assert p.tool_guidance == table["v2"][0]
        mark = await ab.set_and_pump(rig, {"live_guidance": "v1"}, rev=1)
        assert ab.last_state(rig, mark).rev == 2
        p = rig.session.prompts
        assert p.budget.max_frames_per_turn == 2 and p.tool_guidance == table["v1"][0], (
            p.budget,
            p.tool_guidance,
        )
        mark = await ab.set_and_pump(
            rig, {"max_frames_per_turn": 4, "user_instructions": "Be brief."}, rev=2
        )
        assert ab.last_state(rig, mark).rev == 3
        p = rig.session.prompts
        assert p.budget == dataclasses.replace(base, max_frames_per_turn=4)
        assert p.tool_guidance == table["v1"][0] and p.user_instructions == "Be brief."


async def test_interval_change_changes_only_the_snapshot_and_the_state(tmp_path) -> None:
    """A-5/A-9: a ``frame_interval_ms`` set changes the snapshot and the published state only: the builder object, its
    budget, the system message and ``TurnPolicy`` are untouched; the item reads (250, ui)."""
    from tests.unit import test_t0213_apply_boundary as ab

    require_live_fields()
    async with ab.settings_rig(tmp_path) as (rig, _llm, _browser):
        builder = rig.session.prompts
        sha = rig.session.system_sha12()
        policy_repr = repr(rig.session.machine.policy)
        mark = await ab.set_and_pump(rig, {"frame_interval_ms": 250})
        state = ab.last_state(rig, mark)
        it = ab.item(state, "frame_interval_ms")
        assert state.rev == 1 and (it.value, it.source, it.pending) == (250, "ui", None), it
        assert it.changeable is True and it.kind == "next_turn", it
        assert rig.session.settings.frame_interval_ms == 250
        assert rig.session.prompts is builder or rig.session.prompts.budget == builder.budget
        assert rig.session.system_sha12() == sha and repr(rig.session.machine.policy) == policy_repr


# =====================================================================================================
# A-7: the system message and the thinking token
# =====================================================================================================


def _kit_for(mode: str) -> Any:
    from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit

    if mode == "off":
        return None
    if mode == "dry_run":
        return "default"
    return live_kit(FakeLiveBrowser([jpeg_frame("t0227-sys")]))


@pytest.mark.parametrize("mode", MODES)
async def test_system_message_unchanged_by_both(tmp_path, mode) -> None:
    """A-7: for every value of both settings (and, in live, every guidance version), the system message bytes and
    ``system_sha12`` equal those before the set; the request body's template kwargs stay ``{thinking: True}``."""
    from local_astra.cli import guidance_table_of
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit import test_t0213_apply_boundary as ab
    from tests.unit._t0181_rig import mic_rig
    from tests.unit.test_session_loop import POLICY

    require_live_fields()
    versions = config.LIVE_GUIDANCE_VERSIONS if mode == "live" else (None,)
    async with mic_rig(
        tmp_path, kit=_kit_for(mode), policy=tsl.LOOP if mode == "live" else POLICY
    ) as rig:
        if mode == "live":
            rig.session.init_settings(
                rig.session.settings, {}, {}, guidance_table=guidance_table_of(True, True)
            )
        rev = 0
        checked = 0
        for version in versions:
            if version is not None and version != rig.session.settings.live_guidance:
                mark = await ab.set_and_pump(rig, {"live_guidance": version}, rev=rev)
                rev = ab.last_state(rig, mark).rev
            base_text = rig.session.prompts.system_message().text()
            base_sha = rig.session.system_sha12()
            for key in KEYS:
                for value in CHOICES[key]:
                    mark = await ab.set_and_pump(rig, {key: value}, rev=rev)
                    state = ab.last_state(rig, mark)
                    assert not state.refused, (
                        f"A-3: {key}={value} accepted in {mode}; {state.refused}"
                    )
                    rev = state.rev
                    assert ab.item(state, key).value == value
                    assert rig.session.prompts.system_message().text() == base_text, (
                        mode,
                        version,
                        key,
                        value,
                    )
                    assert rig.session.system_sha12() == base_sha == state.system_sha12
                    assert rig.session.prompts.thinking is True, "A-7: thinking is never touched"
                    checked += 1
        assert checked == len(versions) * (len(FRAMES_CHOICES) + len(INTERVALS))
        rig.llm.script = [ab.answer(ab.ANSWER_1)]
        await rig.type_question(ab.QUESTION_1)
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 1,
            "the turn never ended",
            limit_ms=20_000,
        )
        # R-2: the fake records ``bodies`` only for tools or continuation prompts (never in off), so every recorded
        # turn request (prefill and generate, captions aside) is rendered by the real client, as BudgetLlm does
        from local_astra.conductor.llm_client import VllmClient
        from local_astra.contracts.llm import LlmOptions
        from tests.fakes.llm import is_caption_prompt

        sent = [(rid, p, LlmOptions(max_tokens=1)) for rid, p in rig.llm.prefills]
        sent += [(rid, p, o) for rid, p, o in rig.llm.generates if not is_caption_prompt(p)]
        sent = [(rid, p, o) for rid, p, o in sent if "-t1-" in rid]
        last_text = base_text
    assert sent, "premise: the turn after the sets sent a request"
    for rid, prompt, options in sent:
        body = VllmClient().request_body(prompt, rid, options)
        assert body["chat_template_kwargs"] == {THINKING_TEMPLATE_KWARG: True}, (
            rid,
            body["chat_template_kwargs"],
        )
        assert body["messages"][0].get("role") == "system", (
            f"A-7: the system message is first ({rid})"
        )
        systems = [m for m in prompt.messages if m.role == "system"]
        assert len(systems) == 1 and systems[0].text() == last_text, (
            f"A-7: the turn after every set renders the system message it had before them ({rid})"
        )


# =====================================================================================================
# A-8: the server accepts the page's cadence (green at BASE by design: no limit exists and none is added)
# =====================================================================================================


def _posted(i: int, source: Source = Source.SCREEN) -> AstraFrame:
    jpeg = b"\xff\xd8\xff\xe0" + f"T0227-{i}|".encode() * 8
    return AstraFrame(
        source=source,
        ts_ms=0,
        width=1568,
        height=1176,
        jpeg_b64=base64.b64encode(jpeg).decode(),
        phash=hashlib.sha256(str(i).encode()).hexdigest()[:16],
        title="t",
        app="firefox",
    )


def test_server_keeps_every_frame_at_250ms() -> None:
    """A-8: frames every 250 ms for 61 s on a virtual clock through ``HudServer.route`` into a real
    ``BrowserFrameSource``: zero rejected, nothing refused, and once 60 s of frames exist the ring holds exactly 240
    (by hand: receive times in (t - 60000, t] at a 250 ms spacing)."""
    from local_astra.conductor.hud_server import HudServer

    now = [0]
    src = browser_source.BrowserFrameSource(now_ms=lambda: now[0])

    class _NoSession:
        """The route under test touches only the frame sink; anything else the server asks is a no-op."""

        def set_publisher(self, publish: Any) -> None:
            return None

    server = HudServer(_NoSession(), frame_sink=src, port=0)  # type: ignore[arg-type]
    for i, t in enumerate(range(0, 61_001, 250)):
        now[0] = t
        server.route(_posted(i), None)  # type: ignore[arg-type]
        held = len(src._rings[Source.SCREEN])
        if t >= 59_750:
            assert held == 240, f"A-8: at {t} ms the ring holds 240 frames; got {held}"
        else:
            assert held == t // 250 + 1, (t, held)
    assert server.counts["frames_rejected"] == 0 and src.rejected == 0


def test_slow_cadence_onset_frame() -> None:
    """A-8: at a 10 s cadence for 3 minutes, ``get_frame(at)`` returns the latest frame at or before ``at`` (the hand
    oracle: the frame posted at floor(at / 10000) * 10000), never ``None`` while frames arrive."""
    now = [0]
    src = browser_source.BrowserFrameSource(now_ms=lambda: now[0])
    posted: dict[int, str] = {}
    for i, t in enumerate(range(0, 180_001, 10_000)):
        now[0] = t
        msg = _posted(i)
        assert src.ingest(msg) is not None
        posted[t] = msg.phash
        for at in (t, t - 1, t - 9_999, t - 5_000):
            if at < 0 or at <= t - 60_000:
                continue
            now[0] = t + 9_999  # just before the next frame arrives
            got = asyncio.run(src.get_frame(Source.SCREEN, at_ms=at))
            assert got is not None, f"A-8: get_frame(at={at}) at a 10 s cadence is never None"
            assert got.phash == posted[(at // 10_000) * 10_000], (at, got.ts_ms)
        now[0] = t


# =====================================================================================================
# A-9: state, log and turn line
# =====================================================================================================

OLD_LINE = (
    '{"turn_id":"t9","session_id":"s","spans":{"llm.prefill":[0,0]},"events":[],"regime":{"profile":"lean",'
    '"canvas_length":256,"max_denoising_steps":48,"thinking":true,"vllm_version":"v","model":"m","stt":"s",'
    '"tts":"t","audio_path":"headset","assists":"pixels","assist_flags":{"at_spi":false,"cells":false,'
    '"page_text":false,"pdf_text":false},"git_commit":"abc","seed":null,"tools":"off"},'
    '"input":"text","outcome":"spoken","canvases":1,"thought_tokens":0,"answer_tokens":3,"thought_closed":null,'
    '"session_log":null,"tool":null,"extras":{}}'
)


def _regime(**over: Any) -> Regime:
    base = {
        "profile": "lean",
        "canvas_length": 256,
        "max_denoising_steps": 48,
        "thinking": True,
        "vllm_version": "v",
        "model": "m",
        "stt": "s",
        "tts": "t",
        "audio_path": "headset",
    }
    return Regime(**{**base, **over})


def test_regime_keys_only_when_changed() -> None:
    """A-9: ``Regime`` gains the two keys with defaults 3 and 1000; each is dumped only when it differs from its
    default (independently of the T-0213 fields); a line written before them parses with the defaults."""
    for key in KEYS:
        assert key in Regime.model_fields, f"T-0227 A-9: Regime.{key} does not exist yet"
        assert Regime.model_fields[key].default == DEFAULTS[key]
    old = TurnRecord.from_line(OLD_LINE)
    assert (old.regime.max_frames_per_turn, old.regime.frame_interval_ms) == (3, 1000)
    plain = json.loads(_regime().model_dump_json())
    assert not set(KEYS) & set(plain), plain
    explicit_default = json.loads(
        _regime(max_frames_per_turn=3, frame_interval_ms=1000).model_dump_json()
    )
    assert not set(KEYS) & set(explicit_default)
    one = json.loads(_regime(max_frames_per_turn=4).model_dump_json())
    assert one.get("max_frames_per_turn") == 4 and "frame_interval_ms" not in one, one
    two = json.loads(_regime(frame_interval_ms=250).model_dump_json())
    assert two.get("frame_interval_ms") == 250 and "max_frames_per_turn" not in two, two
    assert "settings_rev" not in one, "the T-0213 R-8 fields stay as they were"


async def test_turn_lines_carry_the_new_values_after_a_set(tmp_path) -> None:
    """A-9: a session's first turn line has neither key (byte-identical to BASE's shape); after a set of both at IDLE
    the next line carries the new values; the session log has one ``settings.change`` record with old and new."""
    from tests.unit import test_t0213_apply_boundary as ab

    require_live_fields()
    async with ab.settings_rig(tmp_path) as (rig, _llm, _browser):
        await ab.typed_turn(rig, ab.QUESTION_1, ab.ANSWER_1, 1)
        mark = await ab.set_and_pump(rig, {"max_frames_per_turn": 2, "frame_interval_ms": 5000})
        assert ab.last_state(rig, mark).rev == 1
        await ab.typed_turn(rig, ab.QUESTION_2, ab.ANSWER_2, 2)
        lines = rig.raw_lines()
    assert not set(KEYS) & set(lines[0]["regime"]), lines[0]["regime"]
    assert (
        lines[1]["regime"].get("max_frames_per_turn"),
        lines[1]["regime"].get("frame_interval_ms"),
    ) == (2, 5000)
    changes = [d for d in rig.kind("settings.change")]
    assert len(changes) == 1, changes
    got = changes[0]["changes"]
    assert got["max_frames_per_turn"] == {"old": 3, "new": 2}, got
    assert got["frame_interval_ms"] == {"old": 1000, "new": 5000}, got


@pytest.mark.parametrize("mode", MODES)
async def test_state_items_in_every_mode(tmp_path, mode) -> None:
    """A-9: the hello state of a session built without ``cli`` carries one item per new row in every mode, changeable,
    with the catalog kind and the value 3 / 1000 (source ``default``)."""
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._t0181_rig import mic_rig
    from tests.unit.test_session_loop import POLICY

    async with mic_rig(
        tmp_path, kit=_kit_for(mode), policy=tsl.LOOP if mode == "live" else POLICY
    ) as rig:
        page = rig.greet()
        states = [m for m in page.sent() if getattr(m, "type", None) == "astra.settings.state"]
    assert len(states) == 1
    items = {i.key: i for i in states[0].items}
    for key in KEYS:
        assert key in items, f"T-0227 A-9: no {key} item in {mode}; {sorted(items)}"
        it = items[key]
        assert (it.value, it.source, it.changeable, it.kind) == (
            DEFAULTS[key],
            "default",
            True,
            entry(key).kind,
        ), it


def test_argparse_premise() -> None:
    """Premise: argparse refuses a value outside ``choices`` with exit 2 (what A-4's choices row relies on)."""
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=int, choices=(1, 2))
    with pytest.raises(SystemExit) as info:
        p.parse_args(["--x", "3"])
    assert info.value.code == 2
