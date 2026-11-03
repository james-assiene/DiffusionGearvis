"""T-0132 group C: the 3-arm crop-budget A/B harness's pure parts and its locked script (unit; CPU; no engine, no
browser).

Group C, activated 2026-10-22 (until then ``pending(task="T-0132")``). Rows that need the harness are red on the named
assertion in ``harness()`` / ``api()`` (the file or a name is absent) or on the locked-script rig's fixture, never on an
import error. The harness is loaded lazily, by path, as T-0131's rows load theirs. Rows run once against the harness
and once against the eval oracle (``tests/eval/test_t0132_ab.py``): a wrong oracle is caught on CPU.

**The assumed harness surface** (``scripts/gpu/t0132_crop_budget_ab.py``; the task names the behaviour, these names
and shapes are the test-author's, stated once so the implementer can implement to them or say why not):

Imported, never copied (C1)
    T-0111g's ``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
    ``non_loopback_sockets``, ``turn_records`` and T-0131's ``INTENDED_TARGETS``, ``BOXES_JS``, ``click_score``,
    ``false_claim``, ``reground_voiced_findings``, ``probe_p1``, ``run_probes``, ``config_evidence`` are module
    attributes whose objects come from ``scripts/gpu/t0111g_live_smoke.py`` / ``scripts/gpu/t0131_grounding_ab.py``
    (an alias is fine; a ``def``, ``class`` or literal of those names is not). ``main`` reads the tree through a
    module-level ``_git`` (T-0111g's seam). Every ``VllmClient(...)`` the harness constructs is the direct argument of
    a ``LoopMeter(...)`` call (C1: "every client, the second one included, is wrapped").

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``ARMS = ((1120, False, None), (1120, True,
    1120), (1120, True, 280))`` read by ``main`` at call time; ``arm_table_sha(arms=None)`` = sha256 of
    ``json.dumps([{"visual_tokens": b, "reground": r, "reground_visual_tokens": c} ...], sort_keys=True,
    separators=(",", ":"))``; ``ARM_TABLE_SHA256`` its value for ``ARMS`` (``3943ca94…``); ``REPEATS = 20``;
    ``ORDER_SEED = 0``; ``RUN_BUDGET_S = 78 * 60`` (C6c); ``PROCESS_START_S``: ``time.monotonic()`` taken at
    import, from which the run budget is measured (C6c, "from process start").

Pure functions
    ``arm_label(visual_tokens, reground, reground_visual_tokens) -> str``: ``"1120/off"``, ``"1120/on/1120"``,
    ``"1120/on/280"``.
    ``order(seed) -> list[tuple[int, bool, int | None, str, int]]``: the 300 ``(visual_tokens, reground,
    reground_visual_tokens, task_id, repeat)`` turns, shuffled jointly.
    ``arm_profile(profile, visual_tokens, reground, reground_visual_tokens) -> Profile``: a copy with ``tools =
    "live"``, ``browser = "headless"`` and the three values set explicitly (``visual_tokens`` and ``reground`` in
    ``model_fields_set``).
    ``wilson(k, n) -> (lo, hi)`` (95 %); ``newcombe(k1, n1, k2, n2) -> (lo, hi)`` for ``k1/n1 - k2/n2`` (Newcombe's
    method 10).
    ``pass2_budget_ok(body, images, *, main, crop) -> bool``: C3's configuration-evidence rule.
    ``decide(inputs, *, p1_passed, void=False, stopped=None) -> str``: C4's label; ``inputs`` maps each arm label to
    ``{"accepted", "safety", "engine_up", "success" (of 100), "hit_rate", "pass2_done_p50_ms",
    "pass2_done_p95_ms"}``.
    ``p4_continuation() -> str`` and ``p4_prompt() -> Prompt``: P4's worst case.
    ``build_report(trials, steps, regrounds, probes, *, stopped=None) -> dict``: as T-0131's, keyed by the three arm
    labels; step latency from the step lines only (``latency``), pass 2's done latency in
    ``reground_cost.latency_ms``.

**C6 (amendments, 2026-10-22; H-T132-4/5) -- the surface added for it:**

- ``decide`` and ``decide_with_reason(inputs, *, p1_passed, void=False, stopped=None, configuration_findings=())
  -> (label, reason)``. The order (C6b): P1 failed -> ``inconclusive``; P2/P3 failed or any configuration finding ->
  ``void`` (reason ``configuration: <first finding>``); stopped -> ``incomplete``; any arm ineligible ->
  ``inconclusive`` (reason naming the arm label, the clause ``acceptance``/``safety``/``engine`` -- checked in that
  order -- and the first failing trial); then (b), then (a). Boundaries compare with a 1e-9 tolerance.
- Each arm's decision input gains ``first_failing: {"acceptance", "safety", "engine"}``, each the first trial (in
  trial-line order) failing that clause, ``"<task>#<repeat>"``, or ``None``.
- ``build_report(...)`` gains a top-level ``configuration_findings: list[str]`` (C6a's five kinds), feeding the
  decision; the per-arm ``configuration.pass2_budget_findings`` stays.
- Reground lines gain ``body_images`` (C6e); ``body_budget`` and ``body_images`` are read from the pass-2 body the
  ``LoopMeter`` scanned, never from the session's record.
- report.md's Decision section (a heading containing "Decision"): the label, a line per on arm naming its label with
  its reground request and ``ok`` counts, a line with the configuration finding count, and a line per arm naming its
  label and its eligibility (``eligible``/``ineligible``) with the failing clause.
- ``manifest["regime"]`` (C6d) gains ``model_revision`` (contains 2ea8372), ``vllm_sh_sha256`` (of
  ``services/vllm.sh``), ``vllm_sh_ref`` (806da12), ``max_num_seqs`` (3), ``prefix_caching`` (true), ``kv_cache``
  (names 4 GiB and bf16), ``chromium`` and ``playwright`` (versions, ``None`` when unknown), ``stt`` (``"STT down
  (ports 8080/8081 free)"``) and the full ``arms`` table; report.md carries the model revision and the STT sentence.
- ``_probe_window(base_url, model, detokenizer)`` is P4 (the CPU row stands it in); ``report.json`` and
  ``report.md`` exist before it is called (C6c).

Regime: CPU, pure, hand-built records and stub commands; no engine, no browser, no measurement.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import io
import itertools
import json
import socket
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_t0132_ab as oracle
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
T0131_HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0111G_HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
LOCKED = REPO / "scripts" / "locked" / "t0132_crop_budget_ab.sh"

#: a row run against the harness and against the eval oracle
IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

#: C1, typed from docs/tasks/T-0132.md
SPEC_ARMS = ((1120, False, None), (1120, True, 1120), (1120, True, 280))
#: computed 2026-10-22 by the test-author from the table above (see the docstring for the canonical JSON)
ARM_TABLE_SHA256 = "3943ca94ce9f9de83aaa285b37e8ce511cf61f69a93ac58d0cf1b28cbc978e4f"
LABELS = ("1120/off", "1120/on/1120", "1120/on/280")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
FROM_T0111G = (
    "score",
    "LoopMeter",
    "sentinel_hits",
    "url_leaks",
    "log_complete",
    "non_loopback_sockets",
    "turn_records",
)
FROM_T0131 = (
    "click_score",
    "false_claim",
    "reground_voiced_findings",
    "probe_p1",
    "run_probes",
    "config_evidence",
)
DATA_FROM_T0131 = ("TASKS", "INTENDED_TARGETS", "BOXES_JS")


def load_harness(name: str) -> ModuleType:
    assert HARNESS.is_file(), f"T-0132 C1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None, f"{HARNESS} is not importable by path"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return module


@functools.cache
def harness() -> ModuleType:
    return load_harness("_t0132_crop_budget_ab_unit")


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0132 C1: {HARNESS.name} has no {name!r} (see tests/unit/test_t0132_ab_harness.py's docstring)"
    )
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    return api(name) if which == "harness" else getattr(oracle, name)


# =====================================================================================================
# C1: imports, the arm table, the order, the arm profile
# =====================================================================================================


def test_imports_from_t0131_never_copied() -> None:
    """C1: importing the harness opens no socket; the T-0111g and T-0131 names are their objects (functions and
    classes defined in those files; data equal to T-0131's), and none is defined in the T-0132 harness by ``def``,
    ``class`` or a literal."""
    with t31.no_sockets() as attempts:
        h = load_harness("_t0132_import_check")
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    _, defined = t31._top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    copied = sorted(defined & {*FROM_T0111G, *FROM_T0131, *DATA_FROM_T0131})
    assert copied == [], f"T-0132 C1: defined in the T-0132 harness, not imported: {copied}"
    for names, origin in ((FROM_T0111G, T0111G_HARNESS), (FROM_T0131, T0131_HARNESS)):
        for name in names:
            obj = api(name)
            src = Path(inspect.getsourcefile(obj) or "").resolve()
            assert src == origin.resolve(), f"T-0132 C1: {name} comes from {src}, not {origin.name}"
    t0131 = t31.harness()
    for name in DATA_FROM_T0131:
        # structural: each by-path load of T-0131's file makes its own ``Target`` class, and a frozen dataclass's
        # ``__eq__`` is False across classes (implementer's finding at b373322); a copied-and-edited table still
        # differs here, and a copy by literal is refused by the AST check above
        assert _structure(api(name)) == _structure(getattr(t0131, name)), (
            f"T-0132 C1: {name} is not T-0131's"
        )


def _structure(value: Any) -> Any:
    """``value`` as plain data: a dataclass becomes ``(class name, its fields' structures)``; mappings and sequences
    are walked; anything else is returned as is."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple((f.name, _structure(getattr(value, f.name))) for f in dataclasses.fields(value)),
        )
    if isinstance(value, dict):
        return {k: _structure(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_structure(v) for v in value]
    return value


def test_every_client_is_metered() -> None:
    """C1 (the sentinel scan covers the second client): every ``VllmClient(...)`` call in the harness is the direct
    argument of a ``LoopMeter(...)`` call, and there are at least two (the main and the pass-2 client)."""
    api("main")
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))

    def name(func: ast.expr) -> str:
        return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")

    clients = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and name(n.func) == "VllmClient"
    ]
    wrapped = {
        id(arg)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and name(n.func) == "LoopMeter"
        for arg in [*n.args, *(k.value for k in n.keywords)]
    }
    assert len(clients) >= 2, f"T-0132 C1: the harness builds {len(clients)} VllmClient call(s)"
    bare = [c.lineno for c in clients if id(c) not in wrapped]
    assert bare == [], f"T-0132 C1: VllmClient(...) not wrapped in LoopMeter at lines {bare}"


def test_arm_table_and_order_seed0() -> None:
    """C1 (reviewer mutation 8): ``ARMS`` is the task's table and ``arm_table_sha()`` its pin (a reordered table has
    another sha); ``REPEATS == 20``; the order is 300 turns, each arm 100 times and each (arm, task) 20 times with
    repeats 0-19, the same on every call, different under another seed, and shuffled jointly: the arms are
    interleaved (a per-arm shuffle runs them in three blocks; a joint shuffle switches arm about 200 times)."""
    assert tuple(api("ARMS")) == SPEC_ARMS
    assert api("arm_table_sha")() == api("ARM_TABLE_SHA256") == ARM_TABLE_SHA256
    assert api("arm_table_sha")((SPEC_ARMS[0], SPEC_ARMS[2], SPEC_ARMS[1])) != ARM_TABLE_SHA256
    assert api("REPEATS") == 20 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 78 * 60, (
        "T-0132 C6c: RUN_BUDGET_S = 78 * 60 (was 85 * 60 before C6)"
    )
    turns = [tuple(t) for t in api("order")(0)]
    assert turns == [tuple(t) for t in api("order")(0)], "the order is deterministic"
    assert turns != [tuple(t) for t in api("order")(1)], "the seed matters"
    assert len(turns) == 300
    arms = Counter(t[:3] for t in turns)
    assert arms == dict.fromkeys(SPEC_ARMS, 100), arms
    per = Counter((t[:3], t[3]) for t in turns)
    assert set(per) == {(a, task) for a in SPEC_ARMS for task in TASK_IDS} and set(
        per.values()
    ) == {20}
    assert {t[4] for t in turns} == set(range(20))
    switches = sum(1 for a, b in itertools.pairwise(turns) if a[:3] != b[:3])
    assert switches > 100, (
        f"T-0132 C1: the arms run in blocks ({switches} switches): not a joint shuffle"
    )
    assert {t[:3] for t in turns[:30]} == set(SPEC_ARMS), "the first 30 turns hold every arm"


@pytest.mark.parametrize("arm", SPEC_ARMS, ids=LABELS)
def test_arm_profile_explicit(arm: tuple[int, bool, int | None]) -> None:
    """C1: every arm sets ``visual_tokens``, ``reground`` and ``reground_visual_tokens`` explicitly, so the live
    defaults never decide an arm; the argument is unchanged."""
    from local_astra.config import Profile

    base = Profile(name="lean")
    before = base.model_dump()
    got = api("arm_profile")(base, *arm)
    assert base.model_dump() == before, "the argument changed"
    assert got.tools == "live" and got.browser == "headless"
    assert (got.visual_tokens, got.reground, got.reground_visual_tokens) == arm
    assert {"visual_tokens", "reground"} <= set(got.model_fields_set), got.model_fields_set
    assert api("arm_label")(*arm) == oracle.arm_label(*arm)


# =====================================================================================================
# C1: refusals before any socket
# =====================================================================================================


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """The T-0131 refusal rig over this harness: a clean tree, a healthy stand-in kit, Chromium present, every
    socket, browser-side constructor and ``nvidia-smi`` refused and recorded."""
    from local_astra import cli

    h = harness()
    sockets: list[tuple[str, object]] = []

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            sockets.append((what, args[1] if len(args) > 1 else args[:1]))
            raise ConnectionRefusedError(f"T-0132 refusal rig: no {what} allowed")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    built: list[str] = []
    for module, name in t31.BROWSER_SIDE:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue

        def init(self: Any, *a: Any, _n: str = name, **k: Any) -> None:
            built.append(_n)
            raise AssertionError(f"T-0132 refusal rig: {_n} was constructed")

        monkeypatch.setattr(mod, name, type(name, (), {"__init__": init}), raising=False)
    processes: list[str] = []
    real_popen = subprocess.Popen

    def popen(args: Any, *a: Any, **k: Any) -> Any:
        text = " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if "nvidia-smi" in text:
            processes.append(text)
            raise AssertionError("T-0132 refusal rig: nvidia-smi started")
        return real_popen(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else t31.GIT + "\n"

    assert callable(getattr(h, "_git", None)), (
        "the harness has no module-level _git (T-0111g's seam)"
    )
    monkeypatch.setattr(h, "_git", git)
    for fn_name in ("score", "probe_p1"):
        g = getattr(h, fn_name).__globals__
        if g is not vars(h) and "_git" in g:
            monkeypatch.setitem(g, "_git", git)
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=t31.TOOLS_SHA12 + "0" * 52, mode="live"
    )
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile, **kw: (kit, None))
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h, sockets=sockets, built=built, processes=processes, status=status,
        out=tmp_path / "t0132-out", capsys=capsys, mp=monkeypatch,
    )  # fmt: skip


REFUSALS = ("dirty-tree", "url-not-loopback", "arm-table-reordered", "arm-table-crop-560")


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(rig: Any, case: str) -> None:
    """C1: T-0131's refusals stand (a dirty tree, a non-loopback URL) and an arm table off its pin (two arms swapped;
    a 560 crop) exits 2 before any socket, browser-side object or ``nvidia-smi``, leaving no out-dir."""
    h = rig.h
    argv = ["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)]
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
    elif case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "arm-table-reordered":
        rig.mp.setattr(h, "ARMS", (SPEC_ARMS[0], SPEC_ARMS[2], SPEC_ARMS[1]))
    elif case == "arm-table-crop-560":
        rig.mp.setattr(h, "ARMS", (*SPEC_ARMS[:2], (1120, True, 560)))
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err}"
    assert rig.sockets == [] and rig.built == [] and rig.processes == [], (
        rig.sockets,
        rig.built,
        rig.processes,
    )
    assert not rig.out.exists(), f"{case}: the refusal created the out-dir"
    assert err.strip(), f"{case}: the refusal says nothing"


def test_probes_p1_to_p3_are_t0131s() -> None:
    """C2: P1-P3 are T-0131's ``probe_p1`` and ``run_probes`` (defined in its file, not here)."""
    for name in ("probe_p1", "run_probes"):
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == T0131_HARNESS.resolve(), (name, src)


# =====================================================================================================
# C2: P4
# =====================================================================================================


def test_p4_prompt_construction() -> None:
    """C2 P4: the continuation is exactly ``MAX_CONTINUATION`` characters of ``render_call``/``render_response``
    blocks over live payloads (a ``screenshot`` key); the prompt carries 4 distinct 1280x800 JPEG images, the live
    system message (the tools field and ``LIVE_TOOL_GUIDANCE``) and continues its final message with that text."""
    from PIL import Image

    from local_astra.conductor import prompt_builder as pb
    from local_astra.conductor import session
    from local_astra.contracts.llm import ImagePart

    cont = api("p4_continuation")()
    assert isinstance(cont, str) and len(cont) == session.MAX_CONTINUATION == 14_000, len(cont)
    assert "<|tool_call>call:" in cont and "<|tool_response>response:" in cont, cont[:200]
    assert "screenshot:" in cont, "the payloads are live ones (screenshot updated/unchanged)"
    prompt = api("p4_prompt")()
    parts = [p for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]
    assert len(parts) == 4 and len({p.url for p in parts}) == 4, f"{len(parts)} images"
    for p in parts:
        img = Image.open(io.BytesIO(base64.b64decode(p.url.split(",", 1)[1])))
        assert img.format == "JPEG" and img.size == (1280, 800), (img.format, img.size)
    system = prompt.messages[0]
    assert system.role == "system" and pb.LIVE_TOOL_GUIDANCE in system.text()
    assert prompt.tools, "the live tools field is sent"
    assert prompt.continue_final_message and prompt.messages[-1].text().endswith(cont)


# =====================================================================================================
# C3: configuration evidence, intervals
# =====================================================================================================


@pytest.mark.parametrize("which", IMPLS)
def test_pass2_budget_rule(which: str) -> None:
    """C3: a pass-2 body is right when it carries the crop at one image and 1120 at two or more (reviewer mutation 4
    in the evidence); a 280 body at two images, a 1120 body at one image of a 280-crop arm and a body with no budget
    are wrong."""
    ok = impl(which, "pass2_budget_ok")

    def body(n: int | None) -> dict[str, Any]:
        return {} if n is None else {"mm_processor_kwargs": {"max_soft_tokens": n}}

    assert ok(body(280), 1, main=1120, crop=280)
    assert ok(body(1120), 2, main=1120, crop=280)
    assert ok(body(1120), 3, main=1120, crop=280)
    assert ok(body(1120), 1, main=1120, crop=1120)
    assert not ok(body(280), 2, main=1120, crop=280)
    assert not ok(body(1120), 1, main=1120, crop=280)
    assert not ok(body(None), 1, main=1120, crop=280)


#: Newcombe (1998), Stat Med 17:857 (single proportion, Wilson score) and 17:873 (difference, method 10); four
#: decimals as published
WILSON = {
    (81, 263): (0.2553, 0.3662),
    (15, 148): (0.0624, 0.1605),
    (0, 20): (0.0, 0.1611),
    (1, 29): (0.0061, 0.1718),
}
NEWCOMBE = {
    (56, 70, 48, 80): (0.0524, 0.3339),
    (9, 10, 3, 10): (0.1705, 0.8090),
    (5, 56, 0, 29): (-0.0381, 0.1926),
}


@pytest.mark.parametrize("which", IMPLS)
def test_wilson_and_newcombe_hand_values(which: str) -> None:
    """C3: Wilson 95 % and Newcombe's hybrid score difference intervals equal the published values to four
    decimals."""
    w, nc = impl(which, "wilson"), impl(which, "newcombe")
    for args, want in WILSON.items():
        got = w(*args)
        assert all(abs(a - b) < 5e-5 for a, b in zip(got, want, strict=True)), (args, got, want)
    for args, want in NEWCOMBE.items():
        got = nc(*args)
        assert all(abs(a - b) < 5e-5 for a, b in zip(got, want, strict=True)), (args, got, want)


# =====================================================================================================
# C4: the decision rule
# =====================================================================================================


def _arm(**over: Any) -> dict[str, Any]:
    base = {
        "accepted": True,
        "safety": 0,
        "engine_up": True,
        "success": 70,
        "hit_rate": 0.95,
        "pass2_done_p50_ms": 1000.0,
        "pass2_done_p95_ms": 2000.0,
        # C6b (H-T132-5): the first failing trial per clause, ``"<task>#<repeat>"`` or None
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
    }
    return base | over


def _inputs(off: dict[str, Any], on1120: dict[str, Any], on280: dict[str, Any]) -> dict[str, Any]:
    return {"1120/off": _arm(**off), "1120/on/1120": _arm(**on1120), "1120/on/280": _arm(**on280)}


OFF_50 = {"success": 50, "hit_rate": 0.90, "pass2_done_p50_ms": None, "pass2_done_p95_ms": None}
FAST = {"pass2_done_p50_ms": 800.0, "pass2_done_p95_ms": 2000.0}
#: (inputs, label) by hand from C4 (b) then (a)
DECISIONS: dict[str, tuple[dict[str, Any], str]] = {
    "crop-280-all-clauses": (_inputs(OFF_50, {}, FAST), "reground on, crop 280"),
    "crop-280-success-exactly-minus-5": (
        _inputs(OFF_50, {}, FAST | {"success": 65}),
        "reground on, crop 280",
    ),
    "crop-1120-success-minus-6": (
        _inputs(OFF_50, {}, FAST | {"success": 64}),
        "reground on, crop 1120",
    ),
    "crop-280-hit-exactly-minus-0.03": (
        _inputs(OFF_50, {"hit_rate": 0.93}, FAST | {"hit_rate": 0.90}),
        "reground on, crop 280",
    ),
    "crop-1120-hit-minus-0.031": (
        _inputs(OFF_50, {"hit_rate": 0.931}, FAST | {"hit_rate": 0.90}),
        "reground on, crop 1120",
    ),
    "crop-280-p50-exactly-0.8x": (_inputs(OFF_50, {}, FAST), "reground on, crop 280"),
    "crop-1120-p50-just-over-0.8x": (
        _inputs(OFF_50, {}, FAST | {"pass2_done_p50_ms": 801.0}),
        "reground on, crop 1120",
    ),
    "crop-280-p95-equal": (
        _inputs(OFF_50, {}, FAST | {"pass2_done_p95_ms": 2000.0}),
        "reground on, crop 280",
    ),
    "crop-1120-p95-slower": (
        _inputs(OFF_50, {}, FAST | {"pass2_done_p95_ms": 2001.0}),
        "reground on, crop 1120",
    ),
    # C6 review A: zero pass-2 requests at 280 (no latency) and a 280 hit rate of None never choose the 280 crop
    # (surviving mutant: ``a is None or ...`` in the comparisons)
    "crop-1120-280-no-pass2": (
        _inputs(OFF_50, {}, {"pass2_done_p50_ms": None, "pass2_done_p95_ms": None}),
        "reground on, crop 1120",
    ),
    "crop-1120-280-hit-none": (
        _inputs(OFF_50, {}, FAST | {"hit_rate": None}),
        "reground on, crop 1120",
    ),
    # C6 review A: an EPS-dependent boundary: 0.27 - 0.03 is 0.24000000000000002 in floating point, so lo 0.24 passes
    # only with the 1e-9 tolerance (O-5); without it (b) would choose the 1120 crop
    "crop-280-eps-boundary": (
        _inputs(
            {"success": 50, "hit_rate": 0.20, "pass2_done_p50_ms": None, "pass2_done_p95_ms": None},
            {"hit_rate": 0.27},
            FAST | {"hit_rate": 0.24},
        ),
        "reground on, crop 280",
    ),
    "on-gain-exactly-5": (
        _inputs(OFF_50 | {"success": 65}, {}, {"pass2_done_p50_ms": 900.0}),
        "reground on, crop 1120",
    ),
    "off-gain-4": (
        _inputs(OFF_50 | {"success": 66}, {}, {"pass2_done_p50_ms": 900.0}),
        "reground off",
    ),
    "on-hit-exactly-minus-0.02": (
        _inputs(OFF_50 | {"hit_rate": 0.97}, {}, {"pass2_done_p50_ms": 900.0}),
        "reground on, crop 1120",
    ),
    "off-hit-minus-0.021": (
        _inputs(OFF_50 | {"hit_rate": 0.971}, {}, {"pass2_done_p50_ms": 900.0}),
        "reground off",
    ),
    "a-judges-the-280-arm": (
        _inputs(OFF_50 | {"success": 61}, {}, FAST | {"success": 65}),
        "reground off",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """C4 (reviewer mutation 7, a margin off by one): hand tables for each label, with the boundaries at exactly
    -5/100 (passes) and -6 (fails), -0.03 (passes) and -0.031, 0.8x (passes) and 801 ms, an equal p95 (passes) and
    1 ms slower, +5/100 (passes) and +4, -0.02 (passes) and -0.021. ``a-judges-the-280-arm``: (b) picks the 280 crop
    (65 >= 70 - 5, faster), and (a) is judged on that arm (65 - 61 = 4 < 5), not on the 1120 arm (70 - 61 = 9)."""
    inputs, want = DECISIONS[row]
    assert impl(which, "decide")(inputs, p1_passed=True) == want, row


#: C6b: (arm label, the inputs' arm key) and (clause, the flag that fails it)
INELIGIBLE_ARMS = {"off": "1120/off", "on1120": "1120/on/1120", "on280": "1120/on/280"}
CLAUSES = {
    "acceptance": {"accepted": False},
    "safety": {"safety": 1},
    "engine": {"engine_up": False},
}


def _ineligible(arm_key: str, clause: str, trial: str = "S3#7") -> dict[str, Any]:
    """``crop-280-all-clauses`` with one arm failing one clause, its first failing trial named."""
    inputs = {k: dict(v) for k, v in DECISIONS["crop-280-all-clauses"][0].items()}
    inputs[arm_key] |= CLAUSES[clause] | {
        "first_failing": {c: (trial if c == clause else None) for c in CLAUSES}
    }
    return inputs


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("clause", list(CLAUSES))
@pytest.mark.parametrize("arm", list(INELIGIBLE_ARMS))
def test_ineligible_arm_is_inconclusive(which: str, arm: str, clause: str) -> None:
    """C6b (H-T132-5; supersedes O-5's (b)/(a) eligibility branches and reviewer A's "an ineligible on arm gives
    reground off"): any arm ineligible, an on arm or the control, by any clause, gives ``inconclusive``; the reason
    names the arm's label, the failing clause (``acceptance``, ``safety`` or ``engine``) and the first failing trial
    (``decide_with_reason(...) -> (label, reason)``; ``inputs[arm]["first_failing"]`` maps each clause to its first
    failing trial, ``"<task>#<repeat>"``). Companion: every arm eligible gives a ReGround label."""
    key = INELIGIBLE_ARMS[arm]
    label, reason = impl(which, "decide_with_reason")(_ineligible(key, clause), p1_passed=True)
    assert label == "inconclusive", (arm, clause, label, reason)
    assert reason is not None and key in reason and clause in reason and "S3#7" in reason, reason
    assert impl(which, "decide")(_ineligible(key, clause), p1_passed=True) == "inconclusive"
    eligible = DECISIONS["crop-280-all-clauses"][0]
    assert impl(which, "decide")(eligible, p1_passed=True) == "reground on, crop 280"


FINDING = "1120/on/280 S2#4 step 1: body_budget 1120 at 1 image, the arm's crop is 280"


@pytest.mark.parametrize("which", IMPLS)
def test_decision_run_labels(which: str) -> None:
    """C4/C6a/C6b, the label order: P1 failed -> "inconclusive"; P2/P3 failed or any configuration finding ->
    "void"; stopped -> "incomplete"; any arm ineligible -> "inconclusive"; then (b), then (a). A configuration
    finding (``decide(..., configuration_findings=[...])``) gives "void" with the reason ``configuration: <first
    finding>``, outranking "incomplete" and ineligibility but not a failed P1; a stopped run with an ineligible
    control is "incomplete" (the order, pinned)."""
    decide, why = impl(which, "decide"), impl(which, "decide_with_reason")
    inputs = DECISIONS["crop-280-all-clauses"][0]
    bad_control = _ineligible("1120/off", "acceptance")
    assert decide(inputs, p1_passed=False, void=True, stopped="budget") == "inconclusive"
    assert decide(inputs, p1_passed=True, void=True, stopped="budget") == "void"
    assert decide(inputs, p1_passed=True, stopped="RUN_BUDGET_S") == "incomplete"
    assert decide(bad_control, p1_passed=True, stopped="RUN_BUDGET_S") == "incomplete"
    for fn in (decide, why):
        assert "configuration_findings" in inspect.signature(fn).parameters, (
            f"T-0132 C6a: {which} {fn.__name__} takes no configuration_findings keyword"
        )
    label, reason = why(inputs, p1_passed=True, configuration_findings=[FINDING, "another"])
    assert label == "void" and reason == f"configuration: {FINDING}", (label, reason)
    assert (
        decide(bad_control, p1_passed=True, stopped="x", configuration_findings=[FINDING]) == "void"
    )
    assert decide(inputs, p1_passed=False, configuration_findings=[FINDING]) == "inconclusive"
    assert decide(inputs, p1_passed=True, configuration_findings=[]) == "reground on, crop 280"


# =====================================================================================================
# C3: pass-2 latency is not step latency
# =====================================================================================================


def _on280(record: dict[str, Any]) -> dict[str, Any]:
    return record | {"visual_tokens": 1120, "reground": True, "reground_visual_tokens": 280}


@pytest.mark.parametrize("which", IMPLS)
def test_pass2_latency_excluded_from_step_latency(which: str) -> None:
    """C3: a turn whose pass 2 took 9999 ms has step latency from its two step lines only (first commit 200/250,
    done 600/900) and the 9999 ms in the reground cost; the pass-2 ``images``/``budget`` distribution counts the one
    request (``images`` 1, ``budget`` 280)."""
    trials = [_on280(t31._trial("S2", 0, {"cart": ["blue mug"]}))]
    steps = [
        _on280(
            t31._step(
                "S2", 0, 1, "executed", 200, 600, call_name="click_at", executed_name="click_at"
            )
        ),
        _on280(t31._step("S2", 0, 2, None, 250, 900)),
    ]
    rows = [
        _on280(
            t31._rg(
                "S2", 0, 1, "ok", ms=9999.0, attempt_id="S20-rg-1", http_status=200,
                first_commit_ms=9000.0, done_ms=9999.0, images=1, budget=280, body_budget=280,
            )
        )
    ]  # fmt: skip
    build = api("build_report") if which == "harness" else oracle.derive
    arm = build(trials, steps, rows, t31._probes())["arms"]["1120/on/280"]
    assert arm["latency"] == {
        "first_commit_ms": {"n": 2, "p50": 200, "p95": 250},
        "done_ms": {"n": 2, "p50": 600, "p95": 900},
    }, arm["latency"]
    assert arm["reground_cost"]["latency_ms"] == {"n": 1, "p50": 9999.0, "p95": 9999.0}, arm
    assert arm["pass2"] == {"images": {"1": 1}, "budget": {"280": 1}}, arm.get("pass2")


# =====================================================================================================
# C6a/C6b/C6f through build_report (hand-built records, T-0131's builders)
# =====================================================================================================


def _rec_arm(arm: tuple[int, bool, int | None]) -> dict[str, Any]:
    b, r, c = arm
    return {"visual_tokens": b, "reground": r, "reground_visual_tokens": c}


def _clean_run() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Two S2 trials per arm, each with one executed click; every arm eligible; every body as configured: the main
    budget only, the guidance sent, each on-arm click regrounded once (``ok``) with ``images == body_images == 1``
    and ``budget == body_budget ==`` the arm's crop."""
    trials, steps, regrounds = [], [], []
    for arm in SPEC_ARMS:
        for k in range(2):
            trials.append(
                t31._trial("S2", k, {"cart": ["blue mug"]}, **_rec_arm(arm))
                | {
                    "budgets_sent": [1120],
                    "guidance_sent": True,
                    "scanned_requests": [f"generate:a{k}"],
                }
            )
            steps.append(
                t31._step(
                    "S2", k, 1, "executed", 200, 600, call_name="click_at", executed_name="click_at",
                    click=t31._click([540, 371], [t31.BLUE], True, 0.0), **_rec_arm(arm),
                )
            )  # fmt: skip
            if arm[1]:
                regrounds.append(
                    t31._rg(
                        "S2", k, 1, "ok", ms=500.0, attempt_id=f"S2{k}-{arm[2]}-rg", http_status=200,
                        first_commit_ms=100.0, done_ms=500.0, images=1, body_images=1, budget=arm[2],
                        body_budget=arm[2], **_rec_arm(arm),
                    )
                )  # fmt: skip
    return trials, steps, regrounds


def _build(which: str) -> Any:
    return api("build_report") if which == "harness" else oracle.derive


def _pick(rows: list[dict[str, Any]], label: str, repeat: int) -> dict[str, Any]:
    (row,) = [r for r in rows if oracle.arm_of(r) == label and int(r["repeat"]) == repeat]
    return row


#: C6a: one configuration fault each, applied to ``_clean_run``'s records
FAULTS = {
    "non-reground-body-not-1120": lambda t, s, g: _pick(t, "1120/off", 0).update(
        budgets_sent=[280]
    ),
    "guidance-not-sent": lambda t, s, g: _pick(t, "1120/on/1120", 1).update(guidance_sent=False),
    "sent-bodies-but-no-evidence": lambda t, s, g: _pick(t, "1120/on/280", 0).update(
        budgets_sent=None, guidance_sent=None
    ),
    "images-differ-from-body-images": lambda t, s, g: _pick(g, "1120/on/1120", 0).update(images=2),
    "budget-differs-from-body-budget": lambda t, s, g: _pick(g, "1120/on/1120", 1).update(
        budget=280
    ),
    "pass2-body-wrong-budget": lambda t, s, g: _pick(g, "1120/on/280", 1).update(
        budget=1120, body_budget=1120
    ),
    "click-without-reground-line": lambda t, s, g: g.remove(_pick(g, "1120/on/280", 0)),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", list(FAULTS))
def test_configuration_finding_voids(which: str, fault: str) -> None:
    """C6a (H-T132-4): each configuration fault makes the run ``void`` with the reason ``configuration: <finding>``
    and lists the finding in ``report["configuration_findings"]``; a trial that forwarded bodies (non-empty
    ``scanned_requests``) is always judged, so missing evidence is a finding. Companion: the clean records give no
    finding and a label that is not ``void``."""
    trials, steps, regrounds = _clean_run()
    clean = _build(which)(trials, steps, regrounds, t31._probes())
    assert clean.get("configuration_findings") == [], clean.get("configuration_findings")
    assert clean["decision"]["label"] != "void", clean["decision"]
    FAULTS[fault](trials, steps, regrounds)
    report = _build(which)(trials, steps, regrounds, t31._probes())
    findings = report.get("configuration_findings")
    assert findings, f"T-0132 C6a {fault}: no configuration finding"
    assert report["decision"]["label"] == "void", (fault, report["decision"])
    assert str(report["decision"]["reason"]).startswith("configuration: "), report["decision"]


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize(
    ("clause", "fault"),
    [
        ("acceptance", {"exception": "harness: boom"}),
        ("safety", {"sentinel_requests": ["generate:a1"]}),
        ("engine", {"engine_up_after": False}),
    ],
    ids=["acceptance", "safety", "engine"],
)
def test_ineligible_reason_names_first_failing_trial(
    which: str, clause: str, fault: dict[str, Any]
) -> None:
    """C6b from records: the control's second trial fails one clause, so the run is ``inconclusive``; the inputs
    carry ``first_failing[clause] == "S2#1"`` and the reason names ``1120/off``, the clause and ``S2#1``."""
    trials, steps, regrounds = _clean_run()
    _pick(trials, "1120/off", 1).update(fault)
    report = _build(which)(trials, steps, regrounds, t31._probes())
    inputs = report["decision"]["inputs"]["1120/off"]
    assert inputs.get("first_failing", {}).get(clause) == "S2#1", inputs
    label, reason = report["decision"]["label"], str(report["decision"].get("reason"))
    assert label == "inconclusive", report["decision"]
    assert "1120/off" in reason and clause in reason and "S2#1" in reason, reason


#: review of 4c67db4, finding 1: arm-level acceptance failures with no step error of their own; the per-trial test
#: must be the arm-level clause applied to the one trial, so each names its trial
ACCEPTANCE_ONLY_AT_ARM_LEVEL = {
    "retry-400-then-200": lambda t, s, g: _pick(s, "1120/off", 1).update(
        retries=[{"http_status": 400, "attempt_id": "S21-gen-retry"}]
    ),
    "log-incomplete": lambda t, s, g: _pick(t, "1120/off", 1).update(log_complete=False),
    "no-step-line": lambda t, s, g: s.remove(_pick(s, "1120/off", 1)),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(ACCEPTANCE_ONLY_AT_ARM_LEVEL))
def test_per_trial_clause_matches_arm_clause(which: str, case: str) -> None:
    """C6b (review of 4c67db4, finding 1): a non-200 in a step's ``retries`` (the 400/413 over-long retry), a trial
    whose ``log_complete`` is False, and a trial with no step line each make 1120/off fail acceptance, and the same
    trial is ``first_failing["acceptance"]``: the reason names ``S2#1``, never ``None``."""
    trials, steps, regrounds = _clean_run()
    ACCEPTANCE_ONLY_AT_ARM_LEVEL[case](trials, steps, regrounds)
    report = _build(which)(trials, steps, regrounds, t31._probes())
    inputs = report["decision"]["inputs"]["1120/off"]
    assert inputs["accepted"] is False, f"premise: {case} fails the arm's acceptance; {inputs}"
    assert inputs.get("first_failing", {}).get("acceptance") == "S2#1", inputs
    reason = str(report["decision"].get("reason"))
    assert report["decision"]["label"] == "inconclusive" and "S2#1" in reason, report["decision"]
    assert "None" not in reason, reason


@pytest.mark.parametrize("which", IMPLS)
def test_raised_but_not_executed_is_not_safety(which: str) -> None:
    """C6b (review of 4c67db4, finding 1): a step that raised but was not executed is not a safety failure (the arm
    counts ``executed`` and ``raised`` together), so the arm stays eligible and ``first_failing["safety"]`` is None.
    Companion: the same step executed and raised is a safety failure named ``S2#1``."""
    trials, steps, regrounds = _clean_run()
    _pick(steps, "1120/off", 1).update(
        status="refused", raised=True, executed_name=None, click=None, executed=False
    )
    report = _build(which)(trials, steps, regrounds, t31._probes())
    inputs = report["decision"]["inputs"]["1120/off"]
    assert inputs["safety"] == 0 and inputs.get("first_failing", {}).get("safety") is None, inputs
    trials, steps, regrounds = _clean_run()
    _pick(steps, "1120/off", 1).update(raised=True)
    report = _build(which)(trials, steps, regrounds, t31._probes())
    inputs = report["decision"]["inputs"]["1120/off"]
    assert inputs["safety"] > 0, f"premise: an executed, raised step is a safety failure; {inputs}"
    assert inputs.get("first_failing", {}).get("safety") == "S2#1", inputs


def _fail(rows: list[dict[str, Any]], label: str, repeat: int, clause: str) -> None:
    _pick(rows, label, repeat).update(
        {
            "acceptance": {"exception": "harness: boom"},
            "safety": {"sentinel_requests": ["generate:x"]},
            "engine": {"engine_up_after": False},
        }[clause]
    )


@pytest.mark.parametrize("which", IMPLS)
def test_ineligibility_orderings(which: str) -> None:
    """C6b orderings (review of 4c67db4: M12 arm order, M13 clause order, M14 trial order), two failures each:
    (1) 1120/off fails engine and 1120/on/280 fails acceptance -> the reason names 1120/off (arm-table order), not
    1120/on/280; (2) 1120/off's S2#0 fails safety and S2#1 fails acceptance -> the reason names acceptance and S2#1
    (clause order acceptance, safety, engine), not safety; (3) both 1120/off trials fail acceptance, with S2#1's line
    written first -> S2#1 is named (trial-line order, not repeat number)."""
    trials, steps, regrounds = _clean_run()
    _fail(trials, "1120/off", 1, "engine")
    _fail(trials, "1120/on/280", 0, "acceptance")
    reason = str(_build(which)(trials, steps, regrounds, t31._probes())["decision"]["reason"])
    assert "1120/off" in reason and "engine" in reason and "1120/on/280" not in reason, reason

    trials, steps, regrounds = _clean_run()
    _fail(trials, "1120/off", 0, "safety")
    _fail(trials, "1120/off", 1, "acceptance")
    reason = str(_build(which)(trials, steps, regrounds, t31._probes())["decision"]["reason"])
    assert "acceptance" in reason and "S2#1" in reason and "safety" not in reason, reason

    trials, steps, regrounds = _clean_run()
    _fail(trials, "1120/off", 0, "acceptance")
    _fail(trials, "1120/off", 1, "acceptance")
    first, second = _pick(trials, "1120/off", 0), _pick(trials, "1120/off", 1)
    i0, i1 = trials.index(first), trials.index(second)
    trials[i0], trials[i1] = second, first
    report = _build(which)(trials, steps, regrounds, t31._probes())
    assert report["decision"]["inputs"]["1120/off"]["first_failing"]["acceptance"] == "S2#1", (
        report["decision"]["inputs"]["1120/off"]
    )
    assert "S2#1" in str(report["decision"]["reason"]), report["decision"]


@pytest.mark.parametrize("which", IMPLS)
def test_reground_line_in_off_arm_voids(which: str) -> None:
    """C6a (review of 4c67db4, M09): a reground line in the 1120/off arm is a configuration finding, so the run is
    ``void`` (``configuration: ...``), on CPU, not only in the GPU eval."""
    trials, steps, regrounds = _clean_run()
    regrounds.append(
        t31._rg(
            "S2", 0, 1, "ok", ms=500.0, attempt_id="S20-off-rg", http_status=200, first_commit_ms=100.0,
            done_ms=500.0, images=1, body_images=1, budget=1120, body_budget=1120, **_rec_arm(SPEC_ARMS[0]),
        )
    )  # fmt: skip
    report = _build(which)(trials, steps, regrounds, t31._probes())
    assert report.get("configuration_findings"), report.get("configuration_findings")
    assert report["decision"]["label"] == "void", report["decision"]
    assert str(report["decision"]["reason"]).startswith("configuration: "), report["decision"]


@pytest.mark.parametrize("which", IMPLS)
def test_arm_configuration_pass2_finding(which: str) -> None:
    """C3/C6a (review of 4c67db4, M01): a 280-arm pass-2 body at 1120 with one image is listed in that arm's
    ``configuration.pass2_budget_findings`` as ``"S2#1 step 1"``; the clean records list none."""
    trials, steps, regrounds = _clean_run()
    clean = _build(which)(trials, steps, regrounds, t31._probes())
    assert clean["arms"]["1120/on/280"]["configuration"]["pass2_budget_findings"] == []
    _pick(regrounds, "1120/on/280", 1).update(budget=1120, body_budget=1120)
    report = _build(which)(trials, steps, regrounds, t31._probes())
    conf = report["arms"]["1120/on/280"]["configuration"]
    assert conf["pass2_budget_findings"] == ["S2#1 step 1"], conf


@pytest.mark.parametrize("which", IMPLS)
def test_timeout_counts_4000_in_decision_inputs(which: str) -> None:
    """C6f: the decision's pass-2 latency is ``reground_cost.latency_ms``, where a ``pass2_timeout`` counts as 4000
    ms: with both 280-crop requests timed out (no ``done_ms``), ``pass2_done_p50_ms == pass2_done_p95_ms ==
    4000.0``, equal to the arm's ``reground_cost.latency_ms`` p50/p95."""
    trials, steps, regrounds = _clean_run()
    for k in range(2):
        _pick(regrounds, "1120/on/280", k).update(
            outcome="pass2_timeout", done_ms=None, first_commit_ms=None, http_status=0, ms=4000.0
        )
    report = _build(which)(trials, steps, regrounds, t31._probes())
    i = report["decision"]["inputs"]["1120/on/280"]
    lat = report["arms"]["1120/on/280"]["reground_cost"]["latency_ms"]
    assert (i["pass2_done_p50_ms"], i["pass2_done_p95_ms"]) == (4000.0, 4000.0), i
    assert (lat["p50"], lat["p95"]) == (4000.0, 4000.0), lat


def _decision_section(md: str) -> list[str]:
    lines = md.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("#") and "decision" in ln.lower()]
    assert starts, f"report.md has no Decision heading:\n{md[:2000]}"
    out = []
    for ln in lines[starts[0] + 1 :]:
        if ln.startswith("#"):
            break
        out.append(ln)
    return out


def _says(line: str, word: str, n: int) -> bool:
    import re

    return bool(re.search(rf"(?i)\b{word}\w*\W{{1,3}}{n}\b|\b{n}\W{{1,3}}{word}", line))


def test_report_md_decision_context() -> None:
    """C6f: beside the label, the Decision section prints each on arm's reground request count and ``ok`` count
    (1120/on/1120: 2 requests, 2 ok; 1120/on/280: 2 requests, 1 ok), the configuration finding count (1 here) and
    each arm's eligibility with the failing clause (the control fails ``engine``)."""
    trials, steps, regrounds = _clean_run()
    _pick(regrounds, "1120/on/280", 1).update(outcome="pass2_unparseable")
    _pick(trials, "1120/off", 0).update(engine_up_after=False)
    _pick(trials, "1120/on/1120", 0).update(guidance_sent=False)
    report = api("build_report")(trials, steps, regrounds, t31._probes())
    section = _decision_section(api("render_markdown")(report))
    text = "\n".join(section)
    assert report["decision"]["label"] in text, text
    on1120 = [ln for ln in section if "1120/on/1120" in ln and "request" in ln.lower()]
    on280 = [ln for ln in section if "1120/on/280" in ln and "request" in ln.lower()]
    assert on1120 and _says(on1120[0], "request", 2) and _says(on1120[0], "ok", 2), text
    assert on280 and _says(on280[0], "request", 2) and _says(on280[0], "ok", 1), text
    conf = [ln for ln in section if "configuration" in ln.lower() and "finding" in ln.lower()]
    assert conf and _says(conf[0], "finding", 1), text
    for label in LABELS:
        mine = [ln for ln in section if label in ln and "eligib" in ln.lower()]
        assert mine, f"no eligibility line for {label}:\n{text}"
    off = [ln for ln in section if "1120/off" in ln and "eligib" in ln.lower()]
    assert any("engine" in ln for ln in off), text


# =====================================================================================================
# C6 review A finding 6: the second client wired end to end, on CPU (T-0131's rig, ~15 turns)
# =====================================================================================================

#: what the session's ``tool.reground`` record is made to claim, so a line that copies it is told from one that reads
#: the scanned body (mutant: ``body_budget`` taken from the session record)
FALSE_BUDGET = 999
STT_DOWN = "STT down (ports 8080/8081 free)"


def _cpu_rig(mp: pytest.MonkeyPatch, out: Path) -> Any:
    """T-0131's CPU rig over this harness: a clean tree, the live kit, Chromium present, the stand-in engine and
    browser (``_engine`` / ``_turn_side``), P1-P3 passing; a client built while no browser is pending (the pass-2
    client) serves the current turn's browser, so pass 2 gets an answer."""
    from local_astra import cli
    from local_astra.conductor import llm_client

    h = harness()

    def git(*args: str) -> str:
        return "" if args[:1] == ("status",) else t31.GIT + "\n"

    mp.setattr(h, "_git", git)
    for mod in (getattr(h, "T", None), getattr(h, "G", None)):
        if mod is not None and hasattr(mod, "_git"):
            mp.setattr(mod, "_git", git)
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=t31.TOOLS_SHA12 + "0" * 52, mode="live"
    )
    mp.setattr(cli, "load_tool_kit", lambda profile, **kw: (kit, None))
    mp.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    rig = types.SimpleNamespace(h=h, cli=cli, kit=kit, mp=mp, out=out, capsys=None)
    st = t31._engine(rig)
    t31._turn_side(rig, st)
    stub = llm_client.VllmClient

    def client(*a: Any, **k: Any) -> Any:
        if st.pending_browser is None and st.browsers:
            st.pending_browser = st.browsers[-1]
        return stub(*a, **k)

    mp.setattr(llm_client, "VllmClient", client)
    mp.setattr(h, "run_probes", lambda base_url, model: t31._probes())
    return rig, st


def _subset(per_arm: int) -> list[tuple[Any, ...]]:
    """The first ``per_arm`` turns of each arm, in the joint seed-0 order."""
    seen: Counter[Any] = Counter()
    out = []
    for t in harness().order(0):
        if seen[tuple(t[:3])] < per_arm:
            seen[tuple(t[:3])] += 1
            out.append(tuple(t))
    return out


@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One ``main`` run of 15 turns (5 per arm) on CPU; the session's ``tool.reground`` records claim
    ``budget == 999``; P4 is stood in and records whether the report already existed."""
    from local_astra.conductor import session as session_mod

    out = tmp_path_factory.mktemp("t0132_cpu") / "run"
    seen: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as mp:
        _rig, st = _cpu_rig(mp, out)
        h = harness()
        turns = _subset(5)
        mp.setattr(h, "order", lambda seed: list(turns))
        real_log = session_mod.Session._log

        def log(self: Any, kind: str, data: dict[str, Any], **kw: Any) -> None:
            if kind == "tool.reground" and "budget" in data:
                data = {**data, "budget": FALSE_BUDGET}
            real_log(self, kind, data, **kw)

        mp.setattr(session_mod.Session, "_log", log)

        async def p4(*a: Any, **k: Any) -> dict[str, Any]:
            seen["report_before_p4"] = (out / "report.json").is_file() and (
                out / "report.md"
            ).is_file()
            return {"window": "failed", "reason": "CPU rig: P4 stood in"}

        assert hasattr(h, "_probe_window"), "the harness has no _probe_window (P4)"
        mp.setattr(h, "_probe_window", p4)
        seen["rc"] = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
        seen["clients"] = [dict(c.kwargs) for c in st.clients]

    def jsonl(name: str) -> list[dict[str, Any]]:
        p = out / name
        return (
            [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
        )

    return {
        **seen,
        "out": out,
        "trials": jsonl("trials.jsonl"),
        "regrounds": jsonl("regrounds.jsonl"),
        "report": json.loads((out / "report.json").read_text()) if (out / "report.json").is_file() else {},
        "manifest": json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").is_file() else {},
        "md": (out / "report.md").read_text() if (out / "report.md").is_file() else "",
    }  # fmt: skip


def test_second_client_wiring_on_cpu(cpu_run: dict[str, Any]) -> None:
    """C1/C3/C6e (review A finding 6; mutants: ``meters = [meter]``, the Session given ``reground_llm=meter``, the
    crop client built at ``budget``, ``body_budget`` copied from the session record): a 280-crop trial's
    ``scanned_requests`` holds its pass-2 request (``generate:<attempt_id>``, so the second client is metered and
    scanned), and its reground line has ``body_budget == 280`` and ``body_images == 1`` read from that body while the
    session record claims 999; the 280 arm's ``configuration.pass2_budget_findings`` is empty."""
    trials, regrounds = cpu_run["trials"], cpu_run["regrounds"]
    assert len(trials) == 15, f"premise: 15 turns ran; {len(trials)} (exit {cpu_run.get('rc')})"
    mine = [g for g in regrounds if g.get("reground_visual_tokens") == 280 and g.get("attempt_id")]
    assert mine, f"premise: a 280-crop pass-2 request ran; {regrounds}"
    g = mine[0]
    (trial,) = [
        t
        for t in trials
        if t.get("reground_visual_tokens") == 280
        and t["task"] == g["task"]
        and int(t["repeat"]) == int(g["repeat"])
    ]
    assert f"generate:{g['attempt_id']}" in (trial.get("scanned_requests") or []), (
        "T-0132 C1: the pass-2 request of the second client is not in the trial's scan",
        trial.get("scanned_requests"),
    )
    assert g.get("budget") == FALSE_BUDGET, "premise: the session record's budget was replaced"
    assert g.get("body_budget") == 280 and g.get("body_images") == 1, {
        k: g.get(k) for k in ("budget", "body_budget", "images", "body_images")
    }
    conf = cpu_run["report"]["arms"]["1120/on/280"]["configuration"]
    assert conf.get("pass2_budget_findings") == [], conf


def test_configuration_finding_from_a_real_run(cpu_run: dict[str, Any]) -> None:
    """C6a end to end (mutant: ``_configuration`` never finding anything): the session records claim a budget the
    bodies did not carry (999 against 280/1120), so ``budget != body_budget`` is a configuration finding and the
    run is ``void`` with the reason ``configuration: ...``."""
    report = cpu_run["report"]
    assert report.get("configuration_findings"), report.get("configuration_findings")
    assert report["decision"]["label"] == "void", report["decision"]
    assert str(report["decision"].get("reason")).startswith("configuration: "), report["decision"]


def test_report_written_before_p4(cpu_run: dict[str, Any]) -> None:
    """C6c: ``report.json`` and ``report.md`` are written before P4 runs (and rewritten after), so a timeout during
    P4 still leaves a report."""
    assert cpu_run.get("report_before_p4") is True, cpu_run.get("report_before_p4")
    assert cpu_run["report"] and cpu_run["md"], "no report after P4"


def test_manifest_and_report_regime(cpu_run: dict[str, Any]) -> None:
    """C6d: the manifest's regime records the model revision (2ea8372), ``services/vllm.sh``'s sha256 and its ref
    806da12, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, the Chromium and Playwright versions, "STT down
    (ports 8080/8081 free)" and the full arm table; report.md carries the revision and the STT sentence."""
    import hashlib as _h

    regime = cpu_run["manifest"].get("regime") or {}
    want_keys = {
        "model_revision", "vllm_sh_sha256", "vllm_sh_ref", "max_num_seqs", "prefix_caching", "kv_cache",
        "chromium", "playwright", "stt", "arms",
    }  # fmt: skip
    assert want_keys <= set(regime), sorted(want_keys - set(regime))
    assert "2ea8372" in str(regime["model_revision"]), regime["model_revision"]
    vllm_sh = (REPO / "services" / "vllm.sh").read_bytes()
    assert regime["vllm_sh_sha256"] == _h.sha256(vllm_sh).hexdigest()
    assert "806da12" in str(regime["vllm_sh_ref"])
    assert regime["max_num_seqs"] == 3 and regime["prefix_caching"] is True
    kv = str(regime["kv_cache"])
    assert "4" in kv and "GiB" in kv and "bf16" in kv, kv
    assert regime["stt"] == STT_DOWN
    assert regime["arms"] == [
        {"visual_tokens": b, "reground": r, "reground_visual_tokens": c} for b, r, c in SPEC_ARMS
    ]
    assert "2ea8372" in cpu_run["md"] and STT_DOWN in cpu_run["md"]


def test_run_budget_from_process_start(tmp_path: Path) -> None:
    """C6c: the run budget is measured from process start (``PROCESS_START_S``), P1-P3 included: with the process
    started more than ``RUN_BUDGET_S`` ago, ``main`` runs no arm turn and reports the run stopped (``incomplete``)."""
    import time as _time

    h = harness()
    assert isinstance(getattr(h, "PROCESS_START_S", None), float), (
        "the harness has no PROCESS_START_S"
    )
    out = tmp_path / "run"
    with pytest.MonkeyPatch.context() as mp:
        _rig, st = _cpu_rig(mp, out)
        turns = _subset(1)
        mp.setattr(h, "order", lambda seed: list(turns))
        mp.setattr(h, "PROCESS_START_S", _time.monotonic() - h.RUN_BUDGET_S - 1)

        async def p4(*a: Any, **k: Any) -> dict[str, Any]:
            return {"window": "failed", "reason": "CPU rig"}

        mp.setattr(h, "_probe_window", p4)
        h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert report.get("stopped"), report.get("stopped")
    assert report["decision"]["label"] == "incomplete", report["decision"]
    assert st.browsers == [], f"an arm turn ran past the budget: {len(st.browsers)} browser(s)"


def test_locked_script_lock_and_timeout() -> None:
    """C6g/C6c: every ``uv run`` line of the locked script uses ``--locked`` (there are two: the harness and the
    eval), and the harness runs under ``timeout -k 60s 90m``."""
    assert LOCKED.is_file(), f"T-0132 C5: {LOCKED.relative_to(REPO)} does not exist yet"
    code = [
        ln.strip()
        for ln in LOCKED.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2, runs
    unlocked = [ln for ln in runs if "--locked" not in ln]
    assert unlocked == [], f"T-0132 C6g: uv run without --locked: {unlocked}"
    assert any("timeout -k 60s 90m" in ln for ln in code), "T-0132 C6c: no `timeout -k 60s 90m`"


# =====================================================================================================
# C5: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_SHORT = t31.L_SHORT
L_OUT = f"logs/t0132-crop-budget-ab-{L_SHORT}"
L_HARNESS = "scripts/gpu/t0132_crop_budget_ab.py"
L_EVAL = "tests/eval/test_t0132_ab.py"
MERGED = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0132: live default 1120 + ReGround, crop budget A/B"
OTHERS = (
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge T-0131: tools = live grounding (A-D)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0132a: something else",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge T-0132b: something else",
)


@pytest.fixture
def locked(monkeypatch: pytest.MonkeyPatch) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0132 script, harness, eval and merge."""
    assert LOCKED.is_file(), f"T-0132 C5: {LOCKED.relative_to(REPO)} does not exist yet"
    git_stub = t31.GIT_STUB.replace(t31.MERGED_LINE, MERGED)
    for other in t31.OTHER_MERGES:
        git_stub = git_stub.replace(f'echo "{other}"; ', "")
    git_stub = git_stub.replace(
        "*--merges*) ", "*--merges*) " + "".join(f'echo "{m}"; ' for m in OTHERS)
    )
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(
        t31,
        "L_PLACEHOLDERS",
        (
            L_HARNESS,
            L_EVAL,
            "scripts/gpu/t0131_grounding_ab.py",
            "scripts/gpu/t0111g_live_smoke.py",
        ),
    )
    monkeypatch.setattr(t31, "GIT_STUB", git_stub)
    monkeypatch.setattr(
        t31, "UV_STUB", t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0132_RUN_DIR")
    )
    return t31.run_locked


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(locked: Any, tmp_path: Path, case: str) -> None:
    """C5: ``REFUSED`` and exit 2 before the boot or any step: an argument, a dirty tree, an Astra port bound (STT
    must be down), running containers, ``services/vllm.sh`` changed since 806da12, this SHA's out-dir present, and no
    ``Merge T-0132`` merge reachable from HEAD (adversarial: ``Merge T-0131``, ``Merge T-0132a`` and ``Merge
    T-0132b`` are in the log)."""
    res = locked(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output)
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(locked: Any, tmp_path: Path) -> None:
    """C5, the other side of every refusal: the harness under ``timeout 90m`` with a loopback ``--base-url`` and
    ``--out-dir logs/t0132-crop-budget-ab-<sha7>``, then ``tests/eval/test_t0132_ab.py -m gpu`` with
    ``T0132_RUN_DIR`` set to it; exit 0."""
    res = locked(tmp_path)
    assert res.rc == 0, res.output
    steps = res.steps()
    runs = [c for c in steps if c.names(L_HARNESS)]
    evals = [c for c in steps if c.names(L_EVAL)]
    assert runs and evals and steps.index(runs[0]) < steps.index(evals[0]), [c.argv for c in steps]
    out = runs[0].flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), runs[0].argv
    assert (runs[0].flag("--base-url") or "").startswith(("http://127.0.0.1", "http://[::1]")), (
        runs[0].argv
    )
    assert "gpu" in (evals[0].flag("-m") or ""), evals[0].argv
    assert evals[0].env.get("T0132_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    from tests.unit import test_t0111g_smoke_harness as g_tests

    assert g_tests._timeout_of(res.calls, L_HARNESS) == 90 * 60


def test_eval_oracle_arm_table_pin() -> None:
    """Not pending: the eval's arm table and pin are this module's (the canonical JSON hashed here)."""
    import hashlib

    blob = json.dumps(
        [{"visual_tokens": b, "reground": r, "reground_visual_tokens": c} for b, r, c in SPEC_ARMS],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(blob.encode()).hexdigest() == ARM_TABLE_SHA256 == oracle.ARM_TABLE_SHA256
    assert oracle.ARMS == SPEC_ARMS and oracle.LABELS == LABELS
