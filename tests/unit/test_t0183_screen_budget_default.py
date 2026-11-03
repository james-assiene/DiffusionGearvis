"""T-0183 group A: 1120 visual tokens by default in every tools mode, not only ``live`` (docs/tasks/T-0183.md, A1-A5).

**The interfaces these rows pin** (the task's; readings named in the test-author report):

- ``cli.grounding_of(profile, args, mode)`` resolves the main budget as the ``--visual-tokens`` flag, else the
  profile's ``visual_tokens``, else ``config.LIVE_DEFAULT_VISUAL_TOKENS`` (1120) from ``"default"``, in ``off``,
  ``dry_run`` and ``live`` alike (A1). Its ReGround fields and its pass-2 rule are BASE's.
- ``cli`` builds ``VllmClient(..., visual_tokens=1120)`` and ``Session(..., visual_tokens=1120)`` in ``off`` and
  ``dry_run`` with no flag and no key (A2); every body that client builds carries ``"mm_processor_kwargs":
  {"max_soft_tokens": 1120}``, a text-only one included (H-T183-1: unconditional).
- The ``--dry-run`` tools line is BASE's plus `` visual_tokens=1120`` (A3a); turn lines carry ``visual_tokens: 1120``
  (A3b); no ``grounding`` line outside live (A3c); the ``--visual-tokens`` help says 1120 in every mode (A3d).

**Oracles.** Every expected value is the criterion's literal or BASE's bytes:

- A4 (ii) is the D4b pattern: the working tree's *default* ``off``/``dry_run`` session equals BASE's session run
  with ``--visual-tokens 1120``, byte for byte (stdout, stderr, exit code, client and session keywords, the wire bodies
  of every phase the client issues, turn lines and session-log records).
- A4 (i) and the overrides compare the working tree with BASE run with the *same* flags.
- ``test_reground_resolution_unchanged`` runs BASE's ``grounding_of`` itself, extracted from ``git show
  BASE:local_astra/cli.py`` and executed beside the working tree's names, over the full T-0132/T-0133 grid.

**The differential harness** is T-0131's (``tsl._base_tree`` over ``git archive BASE``, a subprocess whose
``local_astra`` is BASE's, origin checked; ``HOME`` an empty directory, as T-0174's). The rig scenarios are T-0111g's
``_capture`` (tool turns, loop turns, a tools-off turn), with every rig ``Session`` given the budget ``cli`` passed.

**Active (group A activation, 2026-10-28).** The rows that need the implementation (``test_resolver_default_every_mode``,
``test_default_equals_explicit_1120``, ``test_help_states_1120_every_mode``, ``test_docs_delta``) are red at BASE on
a named ``AssertionError`` (the default is ``None``/``"unset"`` outside live; no suffix; no delta; the help still
says "with --tools live"). The differential rows pinned to BASE (A4 i, iii, iv, v; the 280 and profile overrides;
the ReGround grid; the refusals) are green at BASE and must stay green.

**Closed range.** ``BASE`` is 7646b8b6 (main when the branch was cut). ``TIP_IMPL`` equals ``BASE`` until the
implementation lands, so the frozen-paths row holds trivially until then; whoever commits implementation bumps it,
and at the merge it closes to the last implementation commit.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps), the real ``VllmClient`` over ``httpx.MockTransport`` for the wire bodies; nothing served; n = 1,
deterministic.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra import cli, config
from local_astra.conductor.prompt_builder import Budget
from local_astra.config import Profile
from local_astra.perception import policy
from tests import _git_range as gr
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit._t0181_pins import capture_without_mic_unavailable  # T-0181 A-R13
from tests.unit._t0201_pins import capture_without_default  # T-0201 D3 re-pin
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS
from tests.unit.test_cli_tools_live import Events, argv, fake_live

#: T-0183's BASE: main at 7646b8b6 ("PROGRESS: T-0182 clip render run"), the commit the branch was cut from. A literal.
BASE = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
#: The implementation tip. Equal to BASE until group A's implementation lands (the range is then empty, so the row
#: holds trivially); bumped by whoever commits implementation, closed at the merge.
TIP_IMPL = "b50dfc83aecb6f64414884a2487d9337da0b8d12"  # group A's implementation (2026-10-28)
#: T-0172 group B's merge and its first parent: the power row (``session.py`` changed there), as T-0174's
POWER = ("1b8feee0", "2114a765")

KEY = "mm_processor_kwargs"
DEFAULT = 1120
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'

#: A4 (iv), written out from the criterion
FROZEN = (
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/memory/captioner.py",
    "local_astra/perception/policy.py",
    "local_astra/toolcall",
    "services/vllm.sh",
    "services/versions.lock",
    "local_astra/bench/thresholds.py",
    "local_astra/computer",
    "local_astra/pages",
    "hud",
)

#: case -> (mode, flags, profile body). The ``off``/``dry_run`` cases of A2/A4 (ii) and the overrides.
OTHER_CASES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "off-default": ("off", (), ""),
    "dry_run-default": ("dry_run", (), ""),
    "off-1120": ("off", ("--visual-tokens", "1120"), ""),
    "dry_run-1120": ("dry_run", ("--visual-tokens", "1120"), ""),
    "off-280": ("off", ("--visual-tokens", "280"), ""),
    "dry_run-280": ("dry_run", ("--visual-tokens", "280"), ""),
    "off-profile-560": ("off", (), "visual_tokens = 560\n"),
    "dry_run-profile-560": ("dry_run", (), "visual_tokens = 560\n"),
}
#: A4 (i): live, for a default run, ``--visual-tokens 280`` and a profile of 560
LIVE_CASES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "live-default": ("live", (), ""),
    "live-280": ("live", ("--visual-tokens", "280"), ""),
    "live-profile-560": ("live", (), "visual_tokens = 560\n"),
}
CASES = {**OTHER_CASES, **LIVE_CASES}
#: the cases whose rig records are captured (one t0111g capture per distinct budget ``cli`` passed)
RECORD_CASES = (
    "off-default",
    "dry_run-default",
    "off-1120",
    "dry_run-1120",
    "off-280",
    "dry_run-280",
)
#: the phases whose wire bodies are read, in order
PHASES = (
    "prefill",
    "generate",
    "text_only",
    "prefill_tools",
    "tool_step",
    "continuation",
    "repair",
    "caption",
)


# --------------------------------------------------------------------------------------------- the capture


def _norm(v: str, tmp: str) -> str:
    root = str(Path(cli.__file__).resolve().parents[1])
    return v.replace(tmp, "<tmp>").replace(root, "<tree>")


def _describe(v: Any, tmp: str) -> Any:
    from tests.unit import test_t0134c_live_feedback_default as t0134c

    return t0134c._describe(v, tmp)


async def _phases(client: Any, sent: list[dict[str, Any]], where: Path) -> dict[str, Any]:
    """Every phase a session's client issues, through ``client``: a prefill, a tools-off generate with a frame, a
    text-only generate (H-T183-1), a tools-on prefill and step, a continuation, a constrained repair and an idle
    caption (``IdleCaptioner`` over the same client). Each stream's fate is not read; the body each phase put on the
    wire is (``None`` when it sent none: a client with no detokeniser, tools off, sends no tools-on request)."""
    from local_astra.contracts.llm import LlmOptions
    from local_astra.contracts.memory import Episode, Source
    from local_astra.memory.captioner import IdleCaptioner
    from tests.fakes.memory import FakeMemory
    from tests.unit import _llm_id_support as S
    from tests.unit import test_llm_client_visual_tokens as VT

    bodies: dict[str, Any] = {}
    off = VT.frame_prompt()
    on = VT.frame_prompt(tools=S.TOOLS)

    def took(name: str, before: int) -> None:
        new = sent[before:]
        assert len(new) <= 1, f"premise: one request per phase; {name} sent {len(new)}"
        bodies[name] = new[0] if new else None

    async def pre(name: str, prompt: Any) -> None:
        before = len(sent)
        with contextlib.suppress(Exception):
            await client.prefill(prompt, f"r-{name}")
        took(name, before)

    async def gen(name: str, prompt: Any, **kw: Any) -> None:
        before = len(sent)
        with contextlib.suppress(Exception):
            async for _ in client.generate(prompt, f"r-{name}", **kw):
                pass
        took(name, before)

    await pre("prefill", off.prefix())
    await gen("generate", off, options=LlmOptions(max_tokens=64))
    await gen("text_only", S.make_prompt(), options=LlmOptions(max_tokens=64))
    await pre("prefill_tools", on.prefix())
    await gen("tool_step", on)
    await gen("continuation", S.continuation_prompt("I will click it."))
    await gen("repair", S.continuation_prompt("I will click it."), cd=VT.CD)
    thumb = where / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8" + b"t0183" * 16 + b"\xff\xd9")
    memory = FakeMemory()
    await memory.ingest(
        Episode(ts_ms=1, source=Source.SCREEN, thumbnail_path=str(thumb), phash="0" * 16)
    )
    before = len(sent)
    with contextlib.suppress(Exception):
        await IdleCaptioner(client, memory).run_once()  # type: ignore[arg-type]
    took("caption", before)
    await client.aclose()
    assert list(bodies) == list(PHASES), "premise: every phase was tried"
    return bodies


def sending(mode: str) -> tuple[str, ...]:
    """The phases a ``mode`` session's client sends (tools off: the client has no detokeniser, so no tools-on step,
    continuation or repair reaches the wire; observed at BASE, and compared phase by phase with BASE in any case)."""
    if mode != "off":
        return PHASES
    return ("prefill", "generate", "text_only", "prefill_tools", "caption")


def _cli_run(where: Path, mode: str, flags: Sequence[str], body: str) -> dict[str, Any]:
    """``astra session <mode> <flags>`` over the fakes: the ``--dry-run`` (exit, stdout, stderr), a real start stopped
    before serving (exit, stdout, stderr, the ``VllmClient`` and ``Session`` keywords) and the wire body of every
    phase of ``PHASES`` through the client ``cli`` built (the real ``VllmClient`` over ``httpx.MockTransport``)."""
    from local_astra.conductor import llm_client
    from local_astra.toolcall.detok import HostDetokenizer
    from tests.unit import _llm_id_support as S
    from tests.unit import test_llm_client_visual_tokens as VT

    w = str(where)
    for sub in ("dry", "build"):
        (where / sub).mkdir(parents=True, exist_ok=True)
    # T-0202 D3 re-pin: pin off (a live case appends --wait-tool off on a tree with the flag, so A4 (i) compares
    # BASE's session, which has no wait tool; off and dry_run are not live and stay as they are; a no-op at BASE)
    args = t0202.pin_off([*MODE_FLAGS[mode], *flags], body, live_prefix=False)
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        mp.setenv("DISPLAY", ":99")
        T.patch_detok_load(mp, T.Calls(), where / "dry")
        explode_clients(mp)
        fake_live(mp, Events())
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run_cli([*argv(where / "dry", *args, config_body=body), "--dry-run"])
    dry = {"code": code, "out": _norm(stdout.getvalue(), w), "err": _norm(stderr.getvalue(), w)}

    stdout, stderr = io.StringIO(), io.StringIO()
    sent: list[dict[str, Any]] = []
    clients: list[Any] = []
    calls = T.Calls()
    bodies: dict[str, Any] | None = None
    sent_at_build = 0
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        mp.setenv("DISPLAY", ":99")
        T.fake_clients(mp, calls)
        real = llm_client.VllmClient

        def vllm(*a: Any, **k: Any) -> Any:
            calls.vllm_kwargs.append(dict(k))
            client = real(*a, transport=VT.serve(sent), **k)
            clients.append(client)
            return client

        mp.setattr(cli, "VllmClient", vllm)
        T.stop_before_serving(mp, calls)
        mp.setattr(HostDetokenizer, "load", classmethod(lambda cls: S.detok()))
        fake_live(mp, Events())
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            bcode = run_cli(argv(where / "build", *args, config_body=body))
        sent_at_build = len(sent)
        if bcode == 0 and clients:
            bodies = asyncio.run(_phases(clients[0], sent, where / "build"))
    return {
        "dry": dry,
        "build": {
            "code": bcode,
            "out": _norm(stdout.getvalue(), w),
            "err": _norm(stderr.getvalue(), w),
        },
        "vllm_kwargs": [_describe(k, w) for k in calls.vllm_kwargs],
        "session_kwargs": [_describe(k, w) for k in calls.session_kwargs],
        "sent_at_build": sent_at_build,
        "bodies": bodies,
    }


def _records(tmp: Path, budget: int | None) -> dict[str, Any]:
    """T-0111g's sessions (tool turns, loop turns, a tools-off turn) with every rig ``Session`` given
    ``visual_tokens=budget`` (no keyword at all when ``None``, which is what ``cli`` passes when unset)."""
    from tests.unit import _tool_turn_support as tts
    from tests.unit import test_t0111g_differential as t0111g

    with pytest.MonkeyPatch.context() as mp:
        if budget is not None:
            real = tts.Session

            class Budgeted(real):  # type: ignore[misc, valid-type]
                def __init__(self, *a: Any, **k: Any) -> None:
                    k.setdefault("visual_tokens", budget)
                    super().__init__(*a, **k)

            mp.setattr(tts, "Session", Budgeted)
        got: dict[str, Any] = asyncio.run(t0111g._capture(tmp))
    return got


def _session_budget(run: dict[str, Any]) -> Any:
    kws = run["session_kwargs"]
    return kws[-1].get("visual_tokens") if kws else "<no session>"


def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    runs = {case: _cli_run(tmp / "cli" / case, *spec) for case, spec in CASES.items()}
    budgets = sorted({json.dumps(_session_budget(runs[c])) for c in RECORD_CASES})
    records = {b: _records(tmp / "records" / f"b{i}", json.loads(b)) for i, b in enumerate(budgets)}
    out = {"cli": runs, "records": records}
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0183_screen_budget_default as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's capture, in BASE's tree (T-0131's ``git archive`` harness through ``tsl._base_tree``)."""
    gr.require_commits(BASE)
    tmp = tmp_path_factory.mktemp("t0183_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    home = tmp / "empty_home"  # T-0163/T-0175: the child reads no real Brave key file
    home.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "HOME": str(home)},
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    # T-0177 B-P1 (pre-authorised pin move): the observed side without the published astra.vision.frame messages
    # and their wire.out records (shape checked); BASE's side is not normalised.
    # T-0181 A-R13 pin move: the observed side's Session keywords without mic_unavailable (that one key only)
    # T-0201 D3 re-pin (admit): a live run's default screen reminder line and screen_reminder=True keyword only
    # T-0207 C3 re-pin (admit): a live run's default completion check line and completion_check=True keyword only
    # (A4 (i) compares printed lines, keywords and client phase bodies; the check is a session turn behaviour)
    return t0207.capture_without_default(
        capture_without_default(
            capture_without_mic_unavailable(
                without_vision_frames(_capture(tmp_path_factory.mktemp("t0183_now")))
            )
        ),
        "t0183 now",
    )


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0183: {what} differs:\nwant {json.dumps(want)[:2000]}\ngot  {json.dumps(got)[:2000]}"
    )


def _tools_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("tools=")]


def _compare_runs(want: dict[str, Any], got: dict[str, Any], what: str) -> None:
    assert want["dry"]["code"] == 0 and want["build"]["code"] == 0, f"premise: BASE's {what} runs"
    for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "sent_at_build"):
        _same(want[key], got[key], f"{what} {key}")
    assert want["bodies"] is not None and got["bodies"] is not None, (
        f"premise: the phases ran ({what})"
    )
    for phase in PHASES:
        _same(want["bodies"][phase], got["bodies"][phase], f"{what} the {phase} body")


# --------------------------------------------------------------------------------------------- A1: the resolver

#: (flag, profile key) -> (budget, source), written out from A1: the flag, else the profile, else 1120 ("default")
SOURCES: dict[str, tuple[int | None, int | None, int, str]] = {
    "none": (None, None, DEFAULT, "default"),
    "flag-280": (280, None, 280, "flag"),
    "flag-560": (560, None, 560, "flag"),
    "flag-1120": (1120, None, 1120, "flag"),
    "profile-280": (None, 280, 280, "profile"),
    "profile-560": (None, 560, 560, "profile"),
    "profile-1120": (None, 1120, 1120, "profile"),
    "flag-280-over-profile-560": (280, 560, 280, "flag"),
    "flag-1120-over-profile-280": (1120, 280, 1120, "flag"),
}


def test_resolver_default_every_mode(tmp_path, capsys, monkeypatch):
    """A1: in ``off``, ``dry_run`` and ``live`` alike the main budget is the flag, else the profile, else 1120 from
    ``"default"`` (``SOURCES``, by hand); ReGround stays off (``"unset"`` outside live, ``"default"`` in live, as at
    BASE); the pass-2 budget follows the main budget (``reground_visual_tokens == budget`` from ``"default"``), now
    1120 outside live too. Over the whole grid of ``test_reground_resolution_unchanged`` no source is ``"unset"`` for
    the main budget.

    Adversarial (a mutant keyed on the profile's tools mode, or on ``profile.tools != "off"``): the profile's own
    ``tools`` is ``off``, ``dry_run`` or ``live`` in every row and never changes the answer; through ``cli``, a
    ``tools = "live"`` profile run with ``--tools off`` (and ``--tools dry-run``) builds ``VllmClient`` and
    ``Session`` at 1120 with no ``grounding`` keyword, and so does a plain ``off`` run."""
    assert config.LIVE_DEFAULT_VISUAL_TOKENS == DEFAULT, (
        "premise: the constant keeps its name and value"
    )
    wrong: list[str] = []
    for (case, (flag, key, budget, source)), tools, mode in itertools.product(
        SOURCES.items(), ("off", "dry_run", "live"), ("off", "dry_run", "live")
    ):
        p = Profile(name="lean", tools=tools, visual_tokens=key)  # type: ignore[arg-type]
        g = cli.grounding_of(p, argparse.Namespace(visual_tokens=flag), mode)
        got = (
            g.visual_tokens,
            g.visual_tokens_from,
            g.reground,
            g.reground_from,
            g.reground_visual_tokens,
            g.reground_visual_tokens_from,
        )
        rg_from = "default" if mode == "live" else "unset"
        want = (budget, source, False, rg_from, budget, "default")
        if got != want:
            wrong.append(f"{case} profile.tools={tools} mode={mode}: got {got}, want {want}")
    assert not wrong, "T-0183 A1: the resolver:\n" + "\n".join(wrong[:20])
    unset = [k for k, g in _grid(cli.grounding_of).items() if g["visual_tokens_from"] == "unset"]
    assert not unset, (
        f"T-0183 A1: cli never produces 'unset' for the main budget; it does at {unset[:10]}"
    )
    for override in (("--tools", "off"), ("--tools", "dry-run")):
        calls = _build(tmp_path / override[1], capsys, monkeypatch, *override, body=LIVE_PROFILE)
        kw = calls.session_kwargs[-1]
        assert calls.vllm_kwargs[0].get("visual_tokens") == DEFAULT, (
            f"T-0183 A1 (adversarial): a live profile under {override} gets 1120 from the default; "
            f"VllmClient kwargs {calls.vllm_kwargs[0]}"
        )
        assert kw.get("visual_tokens") == DEFAULT and kw.get("grounding") is None, kw
        assert not kw.get("reground", False), kw
        monkeypatch.undo()
    calls = _build(tmp_path / "plain", capsys, monkeypatch)
    assert calls.vllm_kwargs[0].get("visual_tokens") == DEFAULT, calls.vllm_kwargs[0]


def _build(tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = "") -> T.Calls:
    """The graph ``cli`` builds over the fakes (nothing served); exit 0 required."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *flags, config_body=body))
    got = capsys.readouterr()
    assert code == 0 and calls.graphs, f"{flags}: exit 0; got {code}\n{got.out}{got.err}"
    return calls


# --------------------------------------------------------------------------------------------- A1: ReGround unchanged


def _base_function(name: str) -> Callable[..., Any]:
    """BASE's ``cli.<name>``: its source read from ``git show BASE:local_astra/cli.py`` and executed among a copy of
    the working tree's ``cli`` names (``Grounding``, the constants and the helpers it calls)."""
    gr.require_commits(BASE)
    proc = gr._git("show", f"{BASE}:local_astra/cli.py")
    assert proc.returncode == 0, proc.stderr
    tree = ast.parse(proc.stdout)
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(vars(cli))
    exec(compile(module, f"<{BASE[:8]}:cli.py:{name}>", "exec"), namespace)
    fn: Callable[..., Any] = namespace[name]
    assert fn is not getattr(cli, name), "premise: BASE's function, not the working tree's"
    return fn


_VT = (None, 280, 560, 1120)
_RG = (None, True, False)


def _grid(resolver: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    """``resolver`` over the full T-0132/T-0133 grid: every profile ``Profile`` accepts over ``tools`` x
    ``visual_tokens`` x ``reground`` x ``reground_visual_tokens``, every namespace over the three flags, every mode."""
    out: dict[str, dict[str, Any]] = {}
    spaces = list(itertools.product(_VT, _RG, _VT))
    for tools, pv, pr, pc in itertools.product(("off", "dry_run", "live"), _VT, _RG, _VT):
        try:
            p = Profile(  # type: ignore[arg-type]
                name="lean", tools=tools, visual_tokens=pv, reground=pr, reground_visual_tokens=pc
            )
        except ValidationError:
            continue
        for nv, nr, nc in spaces:
            ns = argparse.Namespace(visual_tokens=nv, reground=nr, reground_visual_tokens=nc)
            for mode in ("off", "dry_run", "live"):
                g = resolver(p, ns, mode)
                key = f"p={tools},{pv},{pr},{pc}|ns={nv},{nr},{nc}|{mode}"
                out[key] = dataclasses.asdict(g)
    return out


def test_reground_resolution_unchanged():
    """A1 (green at BASE, against BASE's own resolver): over the full grid, ReGround (value and source) equals
    BASE's for every input; in ``live`` the whole ``Grounding`` does; outside live, wherever BASE's main budget was set
    (flag or profile) the pass-2 budget and its source equal BASE's. ``reground_visual_tokens_cli_refusal`` (T-0132
    B1) gives BASE's verdict for every input. (Outside live with nothing set, BASE's pass-2 budget was ``None`` from
    ``"unset"``; that cell moves with the main budget and is ``test_resolver_default_every_mode``'s.)"""
    base_resolver = _base_function("grounding_of")
    base_refusal = _base_function("reground_visual_tokens_cli_refusal")
    base_grid = _grid(base_resolver)
    now_grid = _grid(cli.grounding_of)
    assert set(now_grid) == set(base_grid) and len(now_grid) > 1000, (
        f"premise: the grid ({len(now_grid)} cells)"
    )
    assert any(g["reground"] for g in base_grid.values()), "premise: the grid reaches ReGround on"
    assert any(
        g["reground_visual_tokens_from"] in ("flag", "profile") for g in base_grid.values()
    ), "premise: the grid reaches a set pass-2 budget"
    wrong: list[str] = []
    for k, b in base_grid.items():
        n = now_grid[k]
        mode = k.rsplit("|", 1)[1]
        if (n["reground"], n["reground_from"]) != (b["reground"], b["reground_from"]):
            wrong.append(f"{k}: ReGround {n} vs BASE {b}")
        if mode == "live" and n != b:
            wrong.append(f"{k}: live Grounding {n} vs BASE {b}")
        if mode != "live" and b["visual_tokens_from"] != "unset":
            pair = ("reground_visual_tokens", "reground_visual_tokens_from")
            if tuple(n[x] for x in pair) != tuple(b[x] for x in pair):
                wrong.append(f"{k}: pass-2 budget {n} vs BASE {b}")
        verdicts = (
            cli.reground_visual_tokens_cli_refusal(cli.Grounding(**n), mode),
            base_refusal(cli.Grounding(**b), mode),
        )
        if verdicts[0] != verdicts[1]:
            wrong.append(f"{k}: B1 refusal {verdicts[0]!r} vs BASE {verdicts[1]!r}")
    assert not wrong, f"T-0183 A1: {len(wrong)} cells differ from BASE:\n" + "\n".join(wrong[:20])


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_refusals_stand(tmp_path, capsys, monkeypatch, mode):
    """A1 (green at BASE): outside live, ``--reground`` (T-0131 B9), a ``tools = "live"`` profile with ``reground =
    true`` overridden to this mode (B9), and ``--reground-visual-tokens 280`` (T-0132 B1) still exit 2 naming
    ``live``, before any graph is built. Too-wide companion: the default run of the same mode, whose pass-2 budget is
    now resolved from ``"default"``, is accepted (exit 0)."""
    spelled = {"off": "off", "dry_run": "dry-run"}[mode]
    refused = {
        "--reground": ([*MODE_FLAGS[mode], "--reground"], ""),
        "profile reground": (["--tools", spelled], 'tools = "live"\nreground = true\n'),
        "--reground-visual-tokens": ([*MODE_FLAGS[mode], "--reground-visual-tokens", "280"], ""),
    }
    for n, (what, (flags, body)) in enumerate(refused.items()):
        for extra in (["--dry-run"], []):
            where = tmp_path / f"refused-{n}-{len(extra)}"
            where.mkdir(parents=True)
            calls = T.Calls()
            T.fake_clients(monkeypatch, calls)
            T.stop_before_serving(monkeypatch, calls)
            T.patch_detok_load(monkeypatch, calls, tmp_path)
            fake_live(monkeypatch, Events())
            code = run_cli([*argv(where, *flags, config_body=body), *extra])
            got = capsys.readouterr()
            assert code == 2 and "live" in got.out + got.err, (
                f"T-0183 A1: {what} with {mode} {extra} is refused (exit 2, naming live); got {code}\n"
                f"{got.out}{got.err}"
            )
            assert not calls.graphs, f"T-0183 A1: a refused run builds no graph ({what} {extra})"
            monkeypatch.undo()
    calls = _build(tmp_path / "default", capsys, monkeypatch, *MODE_FLAGS[mode])
    assert calls.graphs, "the default run of the mode is accepted"


# --------------------------------------------------------------------------------------------- A2/A3/A4 (ii)


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_default_equals_explicit_1120(base, now, mode):
    """A4 (ii), A2, A3 (a-c), H-T183-1: with no flag and no key, the working tree's ``mode`` session equals BASE's
    session run with ``--visual-tokens 1120``, byte for byte: the ``--dry-run`` output, a real start's output, the
    ``VllmClient`` and ``Session`` keywords, the wire body of every phase (prefill, generate, a text-only generate,
    tools-on prefill and step, continuation, repair, caption), and the rig sessions' turn lines and session-log
    records. Spelled out: the tools line is ``BASE_LINES[mode]`` plus `` visual_tokens=1120``; both clients' budgets
    and the session's are 1120; every body carries ``{"max_soft_tokens": 1120}``, the text-only one included; every
    turn line carries ``visual_tokens: 1120``; no ``grounding`` line and no ``grounding`` keyword."""
    want_run, got_run = base["cli"][f"{mode}-1120"], now["cli"][f"{mode}-default"]
    assert _tools_lines(got_run["dry"]["out"]) == [BASE_LINES[mode] + " visual_tokens=1120"], (
        f"T-0183 A3a: the {mode} --dry-run tools line is BASE's plus ' visual_tokens=1120'; printed\n"
        f"{got_run['dry']['out']}"
    )
    assert "grounding:" not in got_run["dry"]["out"] + got_run["build"]["out"], (
        f"T-0183 A3c: no grounding line outside live ({mode})"
    )
    assert [k.get("visual_tokens") for k in got_run["vllm_kwargs"]] == [DEFAULT], (
        f"T-0183 A2: one VllmClient, built with visual_tokens=1120; {got_run['vllm_kwargs']}"
    )
    kw = got_run["session_kwargs"][-1]
    assert kw.get("visual_tokens") == DEFAULT and kw.get("grounding") is None, (
        f"T-0183 A2: Session(visual_tokens=1120), no grounding; {kw}"
    )
    assert got_run["bodies"] is not None, "premise: the phases ran"
    for phase in sending(mode):
        body = got_run["bodies"][phase]
        assert body is not None, f"premise: the {mode} client sent a {phase} body"
        assert body.get(KEY) == {"max_soft_tokens": DEFAULT}, (
            f"T-0183 A2/H-T183-1: the default {mode} {phase} body carries max_soft_tokens 1120; "
            f"it has {body.get(KEY)!r}"
        )
    _compare_runs(want_run, got_run, f"A4 (ii) {mode} default vs BASE --visual-tokens 1120")
    want_rec = base["records"][json.dumps(_session_budget(want_run))]
    got_rec = now["records"][json.dumps(_session_budget(got_run))]
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in want_rec[scenario]:
            _same(
                want_rec[scenario][section],
                got_rec[scenario][section],
                f"A4 (ii) {scenario} {section}",
            )
        lines = got_rec[scenario]["records"]
        assert lines and all(r["extras"].get("visual_tokens") == DEFAULT for r in lines), (
            f"T-0183 A3b: every turn line carries visual_tokens 1120 ({scenario})"
        )


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_explicit_280_restores_cost(base, now, mode):
    """A1/A2 (green at BASE; kills a mutant that defaults outside live whatever the flag says): ``--visual-tokens
    280`` in ``mode`` equals BASE's run with the same flag, byte for byte; its tools line is BASE's plus ``
    visual_tokens=280``, the client is built at 280 and every body (text-only and caption included) carries
    ``{"max_soft_tokens": 280}``, the engine's own default budget, so a 1568x590 frame costs what it did at BASE (270
    soft tokens, hand-checked in ``test_window_tripwire_unchanged``)."""
    got = now["cli"][f"{mode}-280"]
    assert _tools_lines(got["dry"]["out"]) == [BASE_LINES[mode] + " visual_tokens=280"], got["dry"]
    assert [k.get("visual_tokens") for k in got["vllm_kwargs"]] == [280], got["vllm_kwargs"]
    assert got["bodies"] is not None, "premise: the phases ran"
    for phase in sending(mode):
        body = got["bodies"][phase]
        assert body is not None and body.get(KEY) == {"max_soft_tokens": 280}, (phase, body)
    _compare_runs(base["cli"][f"{mode}-280"], got, f"{mode} --visual-tokens 280")
    want_rec = base["records"][json.dumps(_session_budget(base["cli"][f"{mode}-280"]))]
    got_rec = now["records"][json.dumps(_session_budget(got))]
    _same(want_rec, got_rec, f"{mode} --visual-tokens 280 rig records")


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_profile_overrides_default(base, now, mode):
    """A2 (green at BASE): a profile's ``visual_tokens = 560`` in ``mode`` equals BASE's run with that profile, byte
    for byte, and builds the client at 560."""
    got = now["cli"][f"{mode}-profile-560"]
    assert [k.get("visual_tokens") for k in got["vllm_kwargs"]] == [560], got["vllm_kwargs"]
    _compare_runs(base["cli"][f"{mode}-profile-560"], got, f"{mode} profile visual_tokens = 560")


@pytest.mark.parametrize("case", list(LIVE_CASES))
def test_live_unchanged_vs_base(base, now, case):
    """A4 (i) (green at BASE): every live output equals BASE's for the default, ``--visual-tokens 280`` and a profile
    of 560: the ``--dry-run`` and real-start output (the ``grounding`` line included), the client and session
    keywords (``grounding`` included), and the wire body of every phase. T-0191 C3 pin move: these live cases are
    web mode with the default live guidance, which T-0191 C1 flips from v2 to v3, so BASE's side moves only the
    guidance text (v2's bytes to v3's, ``test_t0191_cli_guidance.v2_moved_to_v3``); every other byte is BASE's."""
    from tests.unit.test_t0191_cli_guidance import v2_moved_to_v3

    want = v2_moved_to_v3(base["cli"][case], f"T-0183 A4 (i) {case}")
    _compare_runs(want, now["cli"][case], f"A4 (i) {case}")
    assert any(
        line.startswith("astra session: grounding: ")
        for line in now["cli"][case]["dry"]["out"].splitlines()
    ), "premise: the live grounding line is printed"


# --------------------------------------------------------------------------------------------- A4 (iii)


async def test_rig_session_sends_no_kwargs(tmp_path):
    """A4 (iii) (green at BASE): a ``VllmClient`` built without ``visual_tokens`` sends no ``mm_processor_kwargs`` on
    any phase (prefill, generate, text-only, tools-on prefill and step, continuation, repair, caption); the keyword
    still defaults to ``None`` on ``VllmClient`` and ``Session``; and a rig ``Session`` built without it writes no
    ``visual_tokens`` key on its turn lines (tools off and ``dry_run``). The engine default stays reachable by
    constructing the client directly."""
    import inspect

    import httpx

    from local_astra.conductor import llm_client
    from local_astra.conductor import session as session_mod
    from tests.unit import _llm_id_support as S
    from tests.unit import _tool_turn_support as tts
    from tests.unit import test_llm_client_visual_tokens as VT

    for cls in (llm_client.VllmClient, session_mod.Session):
        param = inspect.signature(cls.__init__).parameters.get("visual_tokens")
        assert param is not None and param.default is None, (
            f"T-0183 A4 (iii): {cls.__name__}(visual_tokens=None) by default"
        )
    sent: list[dict[str, Any]] = []
    client = llm_client.VllmClient(transport=VT.serve(sent), detokenizer=S.detok())
    assert isinstance(client, llm_client.VllmClient) and httpx is not None
    bodies = await _phases(client, sent, tmp_path)
    missing = [n for n, b in bodies.items() if b is None]
    assert not missing, (
        f"premise: a client with a detokeniser sends every phase; none for {missing}"
    )
    carrying = [n for n, b in bodies.items() if KEY in b]
    assert carrying == [], f"T-0183 A4 (iii): a budget-less client sent {KEY} on {carrying}"
    from local_astra.contracts import events as E
    from tests.unit._tool_turn_support import answer

    for kit in (None, "default"):
        async with tts.tool_rig(
            tmp_path / f"rig-{kit}", kit=kit, script=[answer("It is a shop. ")]
        ) as rig:
            rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="what is on the page"))
            await rig.turn_done()
            record = rig.record("t1")
        assert "visual_tokens" not in record.extras, (
            f"T-0183 A4 (iii): a Session built without visual_tokens logs none (kit={kit}); {record.extras}"
        )


# --------------------------------------------------------------------------------------------- A4 (iv), (v)


def test_frozen_paths_over_closed_range():
    """A4 (iv) (green at BASE): the frozen paths are byte-identical over ``BASE..TIP_IMPL`` (two literals, never the
    working tree), with no file added or removed, and each exists at BASE. Power: the same predicate sees T-0172
    group B's ``session.py`` change in its merge."""
    gr.require_commits(BASE, TIP_IMPL, *POWER)
    assert gr.is_ancestor(BASE, TIP_IMPL), f"{BASE[:8]} is not an ancestor of {TIP_IMPL[:8]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE, TIP_IMPL, *FROZEN, diff_filter="ADR") == []
    changed = gr.changed(BASE, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0183 A4 (iv): frozen paths differ in {BASE[:8]}..{TIP_IMPL[:8]}: {changed}"
    )
    assert "local_astra/conductor/session.py" in gr.changed(POWER[0], POWER[1], *FROZEN), "power"


#: sha256 of ``tests/unit/test_t0132_window_tripwire.py`` at BASE 7646b8b6 (computed by the test-author, 2026-10-28)
TRIPWIRE_SHA256 = "11ddf90309ec2944dffcb68d9ab5de0ea0bacdd3bf7a12243f72af741601a694"


def test_window_tripwire_unchanged():
    """A4 (v) (green at BASE): ``Budget`` keeps BASE's fields and defaults (written out), and T-0132's A6 window
    tripwire module is BASE's, byte for byte (a content pin). The evidence's figures, by hand
    (``expected_soft_tokens``: patches of 16 px, a 3x3 merge, sides floored to multiples of 48): 1568x590 is 3613.75
    patches; at 280, scale = sqrt(280 * 9 / 3613.75) = 0.8351, 1309 -> 1296 and 492.7 -> 480, so 81 * 30 / 9 = 270;
    at 1120, scale = 1.6701, 2618.7 -> 2592 and 985.4 -> 960, so 162 * 60 / 9 = 1080."""
    fields = {f.name: f.default for f in dataclasses.fields(Budget)}
    assert fields == {
        "max_prompt_tokens": 14_336,
        "max_images": 4,
        "image_age_turns": 2,
        "max_history_turns": 3,
        "max_frames_per_turn": 3,
    }, f"T-0183 A4 (v): Budget changed: {fields}"
    path = Path(__file__).with_name("test_t0132_window_tripwire.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TRIPWIRE_SHA256, (
        "T-0183 A4 (v): the T-0132 A6 window tripwire changed"
    )
    assert policy.expected_soft_tokens(1568, 590, 280) == 270
    assert policy.expected_soft_tokens(1568, 590, 1120) == 1080


# --------------------------------------------------------------------------------------------- A3d, A5


def test_help_states_1120_every_mode():
    """A3 (d): the ``--visual-tokens`` help names 1120 as the default and no longer ties it to ``--tools live``
    (BASE: "unset is 1120 with --tools live ... and otherwise sends nothing"). Keywords, never wording."""
    from tests.unit.test_t0133_live_reground_off import _help_of

    text = _help_of("--visual-tokens")
    assert "1120" in text, f"T-0183 A3d: the help names 1120; it reads: {text}"
    assert "--tools live" not in text and "sends nothing" not in text, (
        f"T-0183 A3d: the help says 1120 in every mode, not only with --tools live; it reads: {text}"
    )


HEADER = "**Spec delta (T-0183).**"
#: The docs/02 revision read. ``None`` until the merge: the working tree (the implementation's paragraph is not
#: committed yet). **At the merge** whoever merges sets it to the literal full SHA of the last T-0183 commit touching
#: ``docs/02-architecture.md`` (``git log -1 --format=%H <M^2> -- docs/02-architecture.md``), as T-0132's docs row's
#: ``DOC_REV``; then both sides are fixed revisions and later docs/02 edits cannot redden this row.
#: Set 2026-10-28: b50dfc83, group A's implementation, the last T-0183 commit touching docs/02.
DOC_REV: str | None = "b50dfc83aecb6f64414884a2487d9337da0b8d12"


def _doc02(rev: str | None) -> str:
    if rev is None:
        return (gr.REPO / "docs" / "02-architecture.md").read_text()
    gr.require_commits(rev)
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_docs_delta():
    """A5: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0183).**``, stating the default in every
    mode (1120; off, dry_run, live), what is printed (the tools line's ``visual_tokens=``) and what the turn line
    carries, that the engine default (280) is reachable only by constructing the client (``VllmClient``) directly,
    the window facts (14 336), the captioner ruling, and that B3 has no baseline so no threshold moved; removing it
    once leaves docs/02 equal to BASE's, paragraph for paragraph. Keywords, never wording. Reads ``DOC_REV`` (the
    working tree until the merge sets the literal) against BASE."""
    gr.require_commits(BASE)
    base_doc = _doc02(BASE)
    doc = _doc02(DOC_REV)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0183 A5: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "1120": "1120" in para,
        "every mode": all(m in low for m in ("off", "dry_run", "live")),
        "the tools line suffix": "visual_tokens=" in para,
        "the turn line": "turn line" in low or "turn-line" in low,
        "the engine default only by constructing the client directly": "VllmClient" in para
        and "directly" in low
        and "280" in para,
        "the window": "window" in low and ("14 336" in para or "14336" in para or "14_336" in para),
        "the captioner": "caption" in low,
        "B3 has no baseline, no threshold moved": "B3" in para
        and "baseline" in low
        and "threshold" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0183 A5: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0183 A5: docs/02 changes only by the one delta paragraph"
    )
