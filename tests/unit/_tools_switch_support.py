"""Shared helpers for the T-0092b rows: the ``tools`` switch, the startup kit, the tools-on prompt and request.

Nothing here imports a T-0092b symbol at module level: every new name is looked up inside a helper, so while it is
absent each row fails *individually* on an ``AssertionError`` naming it, and collection never errors.

**The surface these rows assume.** docs/tasks/T-0092b.md names the behaviour but not every spelling. Where it is
silent, the choice is fixed here, in one place, and listed in the test-author's report for a ruling:

- ``config.Profile.tools: Literal["off", "dry_run"] = "off"`` (criterion 1, verbatim). A refused value raises a
  ``ValueError`` (pydantic's ``ValidationError`` is one) whose text names ``docs/01 §3`` and ``T-0080d``.
- ``astra session --tools {off,dry-run}``; with no flag the profile decides (``args.tools`` is ``None`` then; the
  namespaces built here carry ``tools=None``).
- ``prompt_builder.NATIVE_TOOL_INSTRUCTIONS`` (criterion 3, verbatim) and
  ``PromptBuilder(system_prompt, budget=None, thinking=True, *, tools=None)``: ``tools`` is the native tools field
  as a tuple, or ``None`` for tools off. **Assumed name: ``tools``.**
- The injected kit (criterion 2, "Wiring"): ``session.ToolKit`` with keyword fields ``mode`` (``"dry_run"``),
  ``tools`` (the tuple), ``executor``, ``payloads``, ``calibration`` and ``repair_off`` (the reason, or ``None``
  when repair is on), handed to ``Session(..., tool_kit=kit)``. ``tool_kit=None`` (the default) is tools off.
  **Assumed names: ``ToolKit``, ``tool_kit`` and the six fields.** None contains the banned substring.
- ``cli`` reads ``services/versions.lock`` as ``cli.SERVICES_DIR / "versions.lock"`` **at call time** (the CLI's
  existing constant for ``services/``); the rows point it at a temporary copy. **Assumed seam.**
- ``cli`` calls ``live.load_payloads()`` and ``live.load_calibration()`` with no argument (criterion 2(b)/(c),
  verbatim), so the rows point ``live.DEFAULT_PAYLOADS`` / ``live.DEFAULT_CALIBRATION`` at copies.
- ``HostDetokenizer.load()`` is patched on the class to return T-0088a's subset tokenizer through
  ``HostDetokenizer(path, sha)``; the real 31 MB file is never read in the fast suite.

Regime: CPU over fakes, ``ScriptedClock``, no measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.unit import _t0197_pins
from tests.unit._detok_support import write_tokenizer

REPO = Path(__file__).resolve().parents[2]
TOOLS_V1 = REPO / "local_astra" / "toolcall" / "tools_v1.json"
PAYLOADS = REPO / "local_astra" / "toolcall" / "payloads" / "native_T4.json"
CALIBRATION = REPO / "docs" / "cd" / "calibration-bfcl.json"
VERSIONS_LOCK = REPO / "services" / "versions.lock"

#: tools_v1 sha256 (T-0080d regime "tools_v1 bbfc53cc…"; the payload manifest's ``tools_sha256``), first 12 hex.
TOOLS_SHA12 = "bbfc53ccdd80"
#: criterion 5's two non-off values, written out by hand from the criterion text.
REGIME_ON = f"native:{TOOLS_SHA12}:dry_run"
REGIME_REPAIR_OFF = f"native:{TOOLS_SHA12}:dry_run:repair_off"

KIT_KWARG = "tool_kit"
KIT_CLASS = "ToolKit"
BUILDER_TOOLS_KWARG = "tools"

#: The sha256 of the H-T84-0-approved paragraphs 2-3 of ``twopass.NATIVE_SYSTEM_PROMPT`` joined by one blank
#: line, computed once from twopass at d2e2879 (the cd-group row re-derives it from twopass itself).
NATIVE_TOOL_INSTRUCTIONS_SHA256 = "6417aabda6c4e0b1bde587bdcea506f5cf2bbb12c896ac50074e9e63722ac55c"


def tools_sha256() -> str:
    return hashlib.sha256(TOOLS_V1.read_bytes()).hexdigest()


def tools_field() -> tuple[dict[str, Any], ...]:
    """``tuple(native.tools_field(tools.load()))`` — criterion 3's value, from the merged T-0080/T-0084 code."""
    from local_astra.toolcall import native, tools

    return tuple(native.tools_field(tools.load()))


def need(module: str, name: str, why: str) -> Any:
    mod = importlib.import_module(module)
    value = getattr(mod, name, None)
    assert value is not None, f"T-0092b {why}: {module}.{name} does not exist"
    return value


def native_instructions() -> str:
    return need(
        "local_astra.conductor.prompt_builder",
        "NATIVE_TOOL_INSTRUCTIONS",
        "criterion 3 (the H11 literal)",
    )


def builder(system: str, *, tools: Any = None, **kwargs: Any) -> Any:
    """``PromptBuilder(system, ..., tools=...)``; a builder without the parameter is a named failure."""
    from local_astra.conductor.prompt_builder import PromptBuilder

    if tools is None:
        return PromptBuilder(system, **kwargs)
    params = inspect.signature(PromptBuilder.__init__).parameters
    assert BUILDER_TOOLS_KWARG in params, (
        "T-0092b criterion 3: PromptBuilder takes the session's tools field "
        f"(assumed keyword {BUILDER_TOOLS_KWARG!r}); it takes {sorted(params)}"
    )
    return PromptBuilder(system, tools=tools, **kwargs)


# --- the kit ------------------------------------------------------------------------------------


def kit_class() -> Any:
    return need("local_astra.conductor.session", KIT_CLASS, "criterion 2 (the injected kit)")


def make_kit(
    *,
    repair_off: str | None = None,
    executor: Any = None,
    with_payloads: bool = True,
    tools_version: str | None = None,
) -> Any:
    """A dry-run kit built from the real, checked payloads and calibration (repair on), or with a reason."""
    from local_astra.toolcall import live

    if executor is None:
        from tests.fakes.tool_executor import RecordingToolExecutor

        executor = RecordingToolExecutor()
    payloads = live.load_payloads() if with_payloads and repair_off is None else None
    calibration = live.load_calibration() if with_payloads and repair_off is None else None
    extra: dict[str, Any] = {}
    if tools_version is not None or KIT_VERSION_FIELD in _kit_fields():
        # T-0092c (carried from the T-0092b review, amending H-A1): the kit carries the sha of the ``ToolSet`` that
        # built ``tools``. Passed only once the field exists, so the T-0092b rows keep their kit before T-0092c;
        # ``tests/unit/test_tool_kit_version.py`` is where its absence is a named failure.
        extra[KIT_VERSION_FIELD] = tools_version if tools_version is not None else tools_sha256()
    return kit_class()(
        mode="dry_run",
        tools=tools_field(),
        executor=executor,
        payloads=payloads,
        calibration=calibration,
        repair_off=repair_off,
        **extra,
    )


#: T-0092c: the kit field that carries the tools sha (docs/tasks/T-0092c.md, "Carried from the T-0092b review").
KIT_VERSION_FIELD = "tools_version"


def _kit_fields() -> set[str]:
    import dataclasses

    cls = kit_class()
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(getattr(cls, "model_fields", {}) or inspect.signature(cls).parameters)


def kit_attr(kit: Any, name: str) -> Any:
    assert hasattr(kit, name), f"T-0092b criterion 2: the kit carries {name!r}; it is {kit!r}"
    return getattr(kit, name)


# --- a session over the fakes, with or without the kit -------------------------------------------


def session_rig(
    tmp_path: Path,
    *,
    kit: Any = None,
    script: Sequence[Any] = (),
    session_log: Any = None,
    publish: Any = None,
    prefix_tokens: int = 1000,
    ms_per_word: int = 200,
) -> Any:
    """The ``test_session_loop`` rig, built here so the session can be handed the kit and a log."""
    from tests.fakes import FakeMemory, FakeStt
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
        frame,
        require_session,
    )

    require_session()
    clock = ScriptedClock()
    machine = RecordingMachine(POLICY)
    llm = RecordingLlm(list(script), prefix_tokens=prefix_tokens)
    llm.clock = clock
    tts = RecordingTts(ms_per_word=ms_per_word)
    audio = RecordingAudio(clock)
    turns_path = tmp_path / "turns.jsonl"
    extra: dict[str, Any] = {}
    if kit is not None:
        params = inspect.signature(Session.__init__).parameters
        assert KIT_KWARG in params, (
            f"T-0092b criterion 2: the session receives the kit by injection (assumed keyword {KIT_KWARG!r}); "
            f"Session.__init__ takes {sorted(params)}"
        )
        extra[KIT_KWARG] = kit
    if session_log is not None:
        extra["session_log"] = session_log
    if publish is not None:
        extra["publish"] = publish
    session = Session(
        machine,
        llm,
        FakeStt(),
        tts,
        audio,
        RecordingPerception([frame(0, "a")]),
        FakeMemory(),
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns_path),
        **extra,
    )
    return Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=session.stt,
        tts=tts,
        audio=audio,
        perception=session.perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
    )


async def one_typed_turn(rig: Any, text: str = "what is this") -> None:
    from local_astra.contracts import events as E
    from local_astra.contracts.turn import TurnState

    rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=text))
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and rig.lines(), "the typed turn never ended"
    )


async def regime_tools_of(kit: Any, tmp_path: Path) -> str:
    """Criterion 5 through a real turn: the ``Regime.tools`` a session given ``kit`` writes."""
    from tests.unit.test_session_loop import answer

    rig = session_rig(tmp_path, kit=kit, script=[answer("Done. ")])
    await rig.session.start()
    try:
        await one_typed_turn(rig)
    finally:
        await rig.session.aclose()
    regime = rig.records()[-1].regime
    assert "tools" in type(regime).model_fields, "T-0092b criterion 5: Regime.tools exists"
    return str(regime.tools)


# --- the CLI rig ---------------------------------------------------------------------------------


class Calls:
    def __init__(self) -> None:
        self.vllm_kwargs: list[dict[str, Any]] = []
        self.session_kwargs: list[dict[str, Any]] = []
        self.detok_loads: list[
            bool
        ] = []  # one entry per load: was an event loop running on the calling thread?
        self.payload_loads = 0
        self.calibration_loads = 0
        self.executors = 0
        self.graphs: list[Any] = []

    def kit(self) -> Any:
        assert self.session_kwargs, "the CLI constructed no Session"
        return self.session_kwargs[-1].get(KIT_KWARG)


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def subset_detok(tmp_path: Path) -> Any:
    from local_astra.toolcall.detok import HostDetokenizer

    path, sha = write_tokenizer(tmp_path / "subset-tok")
    return HostDetokenizer(path, sha)


def patch_detok_load(
    monkeypatch: pytest.MonkeyPatch,
    calls: Calls,
    tmp_path: Path,
    *,
    raise_: BaseException | None = None,
) -> None:
    """``HostDetokenizer.load`` → the T-0088a subset (or ``raise_``), recording whether a loop was running."""
    from local_astra.toolcall.detok import HostDetokenizer

    det = None if raise_ is not None else subset_detok(tmp_path)

    def load(cls: Any = None) -> Any:
        calls.detok_loads.append(_loop_running())
        if raise_ is not None:
            raise raise_
        return det

    monkeypatch.setattr(HostDetokenizer, "load", classmethod(load))


def spy_loaders(monkeypatch: pytest.MonkeyPatch, calls: Calls, *, bomb: bool = False) -> None:
    """Count ``load_payloads``/``load_calibration``/``DryRunToolExecutor`` wherever the CLI reaches them.

    ``bomb``: any call is the failure (criterion 7, tools off). Patched on the defining module and, when the CLI
    holds its own name for it, on ``cli`` too (the ``CONCRETE`` pattern of test_cli_session.py).
    """
    from local_astra import cli
    from local_astra.conductor import tool_executor
    from local_astra.toolcall import live

    real_p, real_c, real_x = (
        live.load_payloads,
        live.load_calibration,
        tool_executor.DryRunToolExecutor,
    )

    def payloads(*a: Any, **k: Any) -> Any:
        assert not bomb, "T-0092b criterion 7: with tools = off, live.load_payloads is never called"
        calls.payload_loads += 1
        return real_p(*a, **k)

    def calibration(*a: Any, **k: Any) -> Any:
        assert not bomb, (
            "T-0092b criterion 7: with tools = off, live.load_calibration is never called"
        )
        calls.calibration_loads += 1
        return real_c(*a, **k)

    class Executor(real_x):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            assert not bomb, (
                "T-0092b criterion 7: with tools = off, no DryRunToolExecutor is constructed"
            )
            calls.executors += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(live, "load_payloads", payloads)
    monkeypatch.setattr(live, "load_calibration", calibration)
    monkeypatch.setattr(tool_executor, "DryRunToolExecutor", Executor)
    for name, value in (
        ("load_payloads", payloads),
        ("load_calibration", calibration),
        ("DryRunToolExecutor", Executor),
    ):
        if hasattr(cli, name):
            monkeypatch.setattr(cli, name, value)


def fake_clients(monkeypatch: pytest.MonkeyPatch, calls: Calls) -> None:
    """The real ``build_session`` over the fakes: concrete clients replaced, ``Session`` and ``VllmClient`` spied."""
    from local_astra import cli, preflight
    from local_astra.conductor import session as session_mod
    from tests.fakes import FakeLlm, FakeMemory, FakeStt, FakeTts

    def vllm(*a: Any, **k: Any) -> Any:
        calls.vllm_kwargs.append(dict(k))
        return FakeLlm()

    real_session = session_mod.Session

    class SpySession(real_session):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            calls.session_kwargs.append(dict(k))
            super().__init__(*a, **k)

    monkeypatch.setattr(cli, "VllmClient", vllm)
    monkeypatch.setattr(cli, "MoshiSttClient", lambda *a, **k: FakeStt())
    monkeypatch.setattr(cli, "PocketTtsClient", lambda *a, **k: FakeTts())
    monkeypatch.setattr(cli, "SqliteMemory", lambda *a, **k: FakeMemory())
    monkeypatch.setattr(cli, "Session", SpySession)
    monkeypatch.setattr(
        preflight, "gather_regime", lambda *a, **k: preflight.Facts(), raising=False
    )


def stop_before_serving(monkeypatch: pytest.MonkeyPatch, calls: Calls) -> None:
    """``asyncio.run(_serve(graph, stop))`` returns at once with the graph recorded: nothing is served."""
    from local_astra import cli

    async def serve(graph: Any, stop: Any = None) -> int:
        calls.graphs.append(graph)
        return 0

    monkeypatch.setattr(cli, "_serve", serve)
    # T-0197 pin move: a real start may serve the HUD page now (by default); nothing in the fast suite binds the
    # profile's fixed hud_port
    _t0197_pins.install_non_binding_hud(monkeypatch)


def write_config(
    tmp_path: Path, tools: str | None = None, *, canvas_length: int | None = None
) -> Path:
    lines = ["profile = 'lean'", "[profiles.lean]"]
    if tools is not None:
        from local_astra.config import Profile

        assert "tools" in Profile.model_fields, (
            "T-0092b criterion 1: Profile.tools is settable in astra.toml"
        )
        lines.append(f"tools = {json.dumps(tools)}")
    if canvas_length is not None:
        lines += ["[profiles.lean.diffusion]", f"canvas_length = {canvas_length}"]
    path = tmp_path / "astra.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def session_argv(config: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(config),
        "session",
        "--silent",
        "--no-mic",
        "--no-log",
        "--data-dir",
        str(tmp_path / "data"),
        # T-0197 pin move: the session serves the HUD page by default now; the shared builder opts out (only on a
        # tree that parses ``--no-serve-hud``, so a BASE differential run is untouched) and before ``extra``, so a
        # row that passes ``--serve-hud`` itself still gets it (last wins)
        *_t0197_pins.opt_out(),
        *extra,
    ]


def run_main(argv: list[str]) -> int:
    from local_astra import cli

    try:
        return int(cli.main(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


# --- fixture copies ------------------------------------------------------------------------------


def payload_copy(tmp_path: Path, edit: Callable[[dict[str, Any]], None] | None = None) -> Path:
    data = json.loads(PAYLOADS.read_text())
    if edit is not None:
        edit(data)
    out = tmp_path / "payloads" / "native_T4.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, sort_keys=True))
    return out


def calibration_copy(tmp_path: Path, *, edit: bool) -> Path:
    out = tmp_path / "calibration" / "calibration-bfcl.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CALIBRATION, out)
    if edit:
        out.write_bytes(out.read_bytes() + b"\n")
    return out


def lock_copy(tmp_path: Path, *, model_revision: str | None = None) -> Path:
    """A temporary ``services/`` directory holding a copy of ``versions.lock`` (optionally re-pinned)."""
    text = VERSIONS_LOCK.read_text()
    if model_revision is not None:
        old = 'model_revision = "2ea837236295d617ac27f8c17d61228081932c40"'
        assert old in text, "premise: versions.lock pins model_revision 2ea8372…"
        text = text.replace(old, f'model_revision = "{model_revision}"')
    services = tmp_path / "services-copy"
    services.mkdir(parents=True, exist_ok=True)
    (services / "versions.lock").write_text(text)
    return services


def point_cli_at(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payloads: Path | None = None,
    calibration: Path | None = None,
    services: Path | None = None,
) -> None:
    from local_astra import cli
    from local_astra.toolcall import live

    if payloads is not None:
        monkeypatch.setattr(live, "DEFAULT_PAYLOADS", payloads)
    if calibration is not None:
        monkeypatch.setattr(live, "DEFAULT_CALIBRATION", calibration)
    if services is not None:
        monkeypatch.setattr(cli, "SERVICES_DIR", services)


def session_namespace(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "profile": None,
        "data_dir": str(tmp_path / "data"),
        "turns": None,
        "audio_path": "speakers",
        "source": "screen",
        "sink": None,
        "silent": True,
        "mic": "none",
        "no_mic": True,
        "dry_run": False,
        "serve_hud": False,
        "no_log": True,
        "log_dir": None,
        "tools": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)
