"""T-0183 group B: the 280-vs-1120 screen-budget A/B harness's CPU-checkable parts (docs/tasks/T-0183.md, B1-B4).

Every harness row is ``pending(task="T-0183")``: group B runs after group A merges, and the harness does not exist
yet. ``test_p2_fixture_answer_key_matches_page`` checks only the checked-in P2 fixture and is green now. The
harness is loaded lazily, by path (as T-0131's and T-0132's harness rows load theirs), so a missing file or name is a
named ``AssertionError``, never a collection error. No row opens a socket, starts a browser or spends GPU time.

**The assumed harness surface** (``scripts/gpu/t0183_screen_budget_ab.py``; the task names the behaviour, these names
and shapes are the test-author's, stated once so the implementer can implement to them or say why not):

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``ARMS = (280, 1120)`` (each passed explicitly as
    ``--visual-tokens``; B1); ``CONDITIONS = ("c1", "c2", "c3")``; ``REPEATS = 40``; ``P2_QUESTIONS = 20``;
    ``ORDER_SEED = 0``; ``arm_table_sha(arms=None)`` = sha256 of ``json.dumps([{"visual_tokens": b} for b in arms],
    sort_keys=True, separators=(",", ":"))`` and ``ARM_TABLE_SHA256`` its value for ``ARMS`` (``ARM_TABLE_SHA256``
    below, computed by hand from that canonical JSON).

Pure functions
    ``order(seed) -> list[tuple[int, str, int]]``: the 240 ``(visual_tokens, condition, repeat)`` trials, shuffled
    jointly (arms and conditions interleaved, one shuffle).
    ``body_budget_ok(body, arm) -> bool``: the body carries ``"mm_processor_kwargs": {"max_soft_tokens": arm}``.
    ``trial_void(bodies, arm) -> bool``: a trial with no body, or with any body that fails ``body_budget_ok``, is void.
    ``build_report(trials, probes) -> dict`` and ``render_report_md(report) -> str``; ``report["regime"]`` states the
    whole regime (B3, "Regime (group B)").

``main(argv)`` (``--base-url URL --out-dir DIR``)
    Reads the tree through a module-level ``_git(*args) -> str``: ``_git("status", "--porcelain")`` for a dirty tree,
    ``_git("log", "--format=%s")`` for HEAD's ancestry subjects (a ``Merge T-0183`` subject must be among them).
    Refuses, with ``EXIT_REFUSED`` and before any socket, a dirty tree, a missing ``Merge T-0183`` ancestor and a
    non-loopback ``--base-url``. Then calls, through module-level names read at call time, ``probe_p0(base_url)``
    first (``False`` or an exception: ``EXIT_PROBE``, nothing else runs), ``run_arms(...)``, ``probe_p2(...)``, and
    ``probe_p1(...)`` last (B2: "P1 runs after the arms").

Regime: CPU, pure, hand-built bodies and stubbed probes; no engine, no browser, no measurement.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import socket
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0183_screen_budget_ab.py"
FIXTURES = REPO / "tests" / "fixtures" / "t0183"
PENDING = pytest.mark.pending(task="T-0183")

#: B1, typed from docs/tasks/T-0183.md
SPEC_ARMS = (280, 1120)
SPEC_CONDITIONS = ("c1", "c2", "c3")
#: sha256 of '[{"visual_tokens":280},{"visual_tokens":1120}]' (test-author, 2026-10-28; see the docstring)
ARM_TABLE_SHA256 = "ab9502d9d975a197ac9901c441b55a0c5ab716194b9700c1ef2063acc230c227"
KEY = "mm_processor_kwargs"
VALID_URL = "http://127.0.0.1:9/v1"
#: tests/fixtures/t0183/MANIFEST.md's digests
FIXTURE_SHA256 = {
    "names.html": "f8c3257fa5597cdcd2d8031c4fcff1232e74eaebdd1c25ac1a698b172c38f14c",
    "answer_key.json": "192b65206cc96f5bb39e8bc43f8c3b1c5fef812d8c75118923236f5bb0349d94",
}
MERGED_LOG = "PROGRESS: a later note\nMerge T-0183: 1120 visual tokens by default in every tools mode\nT-0183 A\n"


def load_harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0183 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    name = "_t0183_screen_budget_ab_unit"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a harness that fails to import is a named failure
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return module


@functools.cache
def harness() -> ModuleType:
    return load_harness()


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0183 B1: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


# --------------------------------------------------------------------------------------------- B1: the arm table


@PENDING
def test_arm_table_sha_and_order_seed0():
    """B1: the arms are 280 and 1120, the conditions c1/c2/c3, n = 40 per arm per condition, seed 0; the arm table
    hashes to the pinned sha; ``order(0)`` holds each (arm, condition) exactly 40 times with repeats 0..39, is
    deterministic, differs from ``order(1)``, and is shuffled jointly (neither grouped by arm nor by condition)."""
    assert tuple(api("ARMS")) == SPEC_ARMS
    assert tuple(api("CONDITIONS")) == SPEC_CONDITIONS
    assert api("REPEATS") == 40 and api("P2_QUESTIONS") == 20 and api("ORDER_SEED") == 0
    assert api("arm_table_sha")() == api("ARM_TABLE_SHA256") == ARM_TABLE_SHA256
    assert api("arm_table_sha")((1120, 280)) != ARM_TABLE_SHA256, "the sha is order-sensitive"
    trials = list(api("order")(0))
    assert len(trials) == 2 * 3 * 40, len(trials)
    counts = Counter((arm, cond) for arm, cond, _ in trials)
    assert counts == {(a, c): 40 for a in SPEC_ARMS for c in SPEC_CONDITIONS}, counts
    for a in SPEC_ARMS:
        for c in SPEC_CONDITIONS:
            reps = sorted(r for arm, cond, r in trials if (arm, cond) == (a, c))
            assert reps == list(range(40)), (a, c, reps[:5])
    assert list(api("order")(0)) == trials, "order(0) is deterministic"
    assert list(api("order")(1)) != trials, "the seed is used"
    arms_seq = [t[0] for t in trials]
    conds_seq = [t[1] for t in trials]
    assert arms_seq != sorted(arms_seq) and arms_seq[:120] != [arms_seq[0]] * 120, (
        "arms interleaved"
    )
    assert len({tuple(conds_seq[i : i + 40]) for i in range(0, 240, 40)}) > 1, (
        "conditions interleaved"
    )


# --------------------------------------------------------------------------------------------- B1: the budget on the wire


@PENDING
@pytest.mark.parametrize("arm", SPEC_ARMS)
def test_every_body_carries_its_arm_budget(arm):
    """B1: a body carries its arm's budget only with ``mm_processor_kwargs == {"max_soft_tokens": arm}``; a body
    without the key, with the other arm's value, or with an extra key in it does not; a trial is void when any of its
    bodies fails (or it has none), and not void when all carry it (the too-wide direction)."""
    ok = api("body_budget_ok")
    void = api("trial_void")
    other = 1120 if arm == 280 else 280
    good = {"model": "m", "messages": [], KEY: {"max_soft_tokens": arm}}
    assert ok(good, arm) is True
    assert ok({"model": "m", "messages": []}, arm) is False, "no key"
    assert ok({**good, KEY: {"max_soft_tokens": other}}, arm) is False, "the other arm's budget"
    assert ok({**good, KEY: {"max_soft_tokens": arm, "x": 1}}, arm) is False, "extra mm kwargs"
    assert void([good, good], arm) is False
    assert void([good, {"model": "m"}], arm) is True
    assert void([{**good, KEY: {"max_soft_tokens": other}}], arm) is True
    assert void([], arm) is True


# --------------------------------------------------------------------------------------------- B1: refusals


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """The harness with every socket refused and recorded, ``_git`` stubbed (a clean tree whose ancestry has a
    ``Merge T-0183`` subject; ``state`` edits it), and every probe and the arm runner stubbed and recorded."""
    h = harness()
    sockets: list[tuple[str, Any]] = []

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            sockets.append((what, args[1:] if what in ("connect", "connect_ex", "bind") else args))
            raise ConnectionRefusedError(f"T-0183 refusal rig: no {what} allowed")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    state = {"porcelain": "", "log": MERGED_LOG}

    def git(*args: str) -> str:
        if args[:1] == ("status",):
            return state["porcelain"]
        if args[:1] == ("log",):
            return state["log"]
        return ""

    assert callable(getattr(h, "_git", None)), (
        "T-0183 B1: the harness has no module-level _git(*args)"
    )
    monkeypatch.setattr(h, "_git", git)
    calls: list[str] = []
    outcome: dict[str, Any] = {"p0": True}

    def stub(name: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            calls.append(name)
            if name == "probe_p0":
                if isinstance(outcome["p0"], BaseException):
                    raise outcome["p0"]
                return outcome["p0"]
            return {} if name != "run_arms" else []

        return fn

    for name in ("probe_p0", "run_arms", "probe_p1", "probe_p2"):
        assert callable(getattr(h, name, None)), (
            f"T-0183 B2: the harness has no module-level {name}"
        )
        monkeypatch.setattr(h, name, stub(name))
    return types.SimpleNamespace(
        h=h,
        sockets=sockets,
        state=state,
        calls=calls,
        outcome=outcome,
        out=tmp_path / "out",
        capsys=capsys,
    )


REFUSALS = {
    "dirty-tree": ({"porcelain": " M local_astra/cli.py\n"}, VALID_URL),
    "no-merge-ancestor": ({"log": "T-0183 A\nPROGRESS: x\n"}, VALID_URL),
    "url-not-loopback": ({}, "http://10.0.0.5:8001/v1"),
    "url-a-name": ({}, "http://gpu-box:8001/v1"),
}


@PENDING
@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_before_any_socket(rig, case):
    """B1: a dirty tree, a missing ``Merge T-0183`` ancestor and a non-loopback endpoint (an address or a name) each
    exit 2 before any socket opens, before any probe or arm runs, and leave no out-dir; the message names the reason.
    Too-wide companion: the clean, merged, loopback run is not refused (it reaches P0)."""
    edits, url = REFUSALS[case]
    rig.state.update(edits)
    rc = rig.h.main(["--base-url", url, "--out-dir", str(rig.out)])
    err = rig.capsys.readouterr().err
    assert rc == rig.h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-1500:]}"
    assert rig.sockets == [], f"{case}: a refusal opened a socket first: {rig.sockets}"
    assert rig.calls == [], f"{case}: a refusal ran {rig.calls}"
    assert not rig.out.exists(), f"{case}: the refusal wrote the out-dir"
    word = {"dirty-tree": "dirty", "no-merge-ancestor": "T-0183"}.get(case, "loopback")
    assert word in err, f"{case}: the refusal names {word!r}: {err}"


# --------------------------------------------------------------------------------------------- B2: probe order


@PENDING
def test_p0_runs_first_and_p1_last(rig):
    """B2: with a clean, merged tree and a loopback endpoint, ``probe_p0`` is the first call and ``probe_p1`` the last,
    after ``run_arms`` (and P2)."""
    rig.h.main(["--base-url", VALID_URL, "--out-dir", str(rig.out)])
    assert rig.calls and rig.calls[0] == "probe_p0", rig.calls
    assert rig.calls[-1] == "probe_p1", rig.calls
    assert "run_arms" in rig.calls and rig.calls.index("run_arms") < rig.calls.index("probe_p1"), (
        rig.calls
    )


@PENDING
@pytest.mark.parametrize("failure", ["false", "raises"])
def test_p0_failure_exits_3(rig, failure):
    """B2: P0 failing (``False`` or an exception) voids the run: exit 3 and nothing after P0 runs."""
    rig.outcome["p0"] = False if failure == "false" else RuntimeError("HTTP 400 from the engine")
    rc = rig.h.main(["--base-url", VALID_URL, "--out-dir", str(rig.out)])
    assert rc == rig.h.EXIT_PROBE == 3, rc
    assert rig.calls == ["probe_p0"], rig.calls


# --------------------------------------------------------------------------------------------- B3: the report


def _report() -> tuple[dict[str, Any], str]:
    trials: list[dict[str, Any]] = []
    for arm in SPEC_ARMS:
        for cond in SPEC_CONDITIONS:
            for r in range(3):
                trials.append(
                    {
                        "visual_tokens": arm,
                        "condition": cond,
                        "repeat": r,
                        "void": False,
                        "commit_ms": 400 + 10 * r + (300 if arm == 1120 else 0),
                        "first_audio_ms": 1300 + 10 * r + (400 if arm == 1120 else 0),
                        "prompt_tokens": 1500 if arm == 280 else 2300,
                        "cached_tokens": 1024,
                        "query_tokens": 476 if arm == 280 else 1276,
                        "min_free_vram_gib": 4.0,
                        "llm_errors": 0,
                    }
                )
    probes = {
        "p0": {"ok": True},
        "p1": {"tokens": 6100, "limit": 16_384 - 256, "ok": True},
        "p2": {"280": {"correct": 12, "n": 20}, "1120": {"correct": 19, "n": 20}},
    }
    report = api("build_report")(trials, probes)
    return report, str(api("render_report_md")(report))


@PENDING
def test_regime_stated():
    """B3 / "Regime (group B)": the report states the whole regime: profile lean, canvas 256, 48 steps, thinking on,
    model revision 2ea8372, vLLM 0.29.0, Pocket TTS, STT down, a silent sink, typed turns, ``dry_run`` tools with the
    loop at 8, the arms, n = 40, order seed 0, concurrency 1, the engine unseeded; report.md carries the model
    revision and the arms."""
    report, md = _report()
    regime = json.dumps(report.get("regime"), sort_keys=True).lower()
    need = {
        "lean": "lean",
        "canvas 256": "256",
        "48 steps": "48",
        "model revision": "2ea8372",
        "vllm": "0.29.0",
        "pocket tts": "pocket",
        "stt down": "stt",
        "silent sink": "silent",
        "dry_run": "dry_run",
        "arms": "1120",
        "seed": "seed",
        "concurrency": "concurrency",
        "unseeded engine": "unseeded",
    }
    missing = [k for k, w in need.items() if w not in regime]
    assert not missing, f"T-0183 B3: the regime does not state {missing}: {regime[:1500]}"
    assert "2ea8372" in md and "280" in md and "1120" in md, md[:1500]


@PENDING
def test_report_never_names_b3_nor_judges_thresholds():
    """B3 (adversarial; the failure mode the task names: B3 has no baseline, so nothing is judged): neither
    report.json nor report.md names a figure "B3", neither carries a pass/fail verdict, and the harness does not
    import ``local_astra.bench.thresholds``. The 1.5 s and 2.2 s figures, if shown, are context only."""
    report, md = _report()
    text = json.dumps(report) + "\n" + md
    assert not re.search(r"\bB3\b", text), "T-0183 B3: the report names a figure B3"
    assert not re.search(r"\b(pass|fail|passed|failed)\b", text, re.IGNORECASE), (
        "T-0183 B3: the report carries a pass/fail verdict"
    )
    tree = ast.parse(HARNESS.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any("thresholds" in m for m in imported), (
        f"T-0183 B3: the harness imports {imported}"
    )


# --------------------------------------------------------------------------------------------- P2's fixture


def test_p2_fixture_answer_key_matches_page():
    """B2 P2: the checked-in page and answer key agree: 20 questions, each answer the text of the cell its question
    names (parsed from the HTML, an independent reading), every name distinct, the viewport 1568x590, and the files'
    bytes are those MANIFEST.md records (green now: the fixture is checked in ahead of group B)."""
    import hashlib

    for name, digest in FIXTURE_SHA256.items():
        got = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert got == digest, f"T-0183 P2: {name} is not the fixture MANIFEST.md records"
    key = json.loads((FIXTURES / "answer_key.json").read_text())
    html = (FIXTURES / key["page"]).read_text()
    assert key["viewport"] == [1568, 590] and len(key["questions"]) == 20
    rows = re.findall(r"<tr><th>(\d+)</th>(.*?)</tr>", html)
    grid = {int(r): re.findall(r"<td>(.*?)</td>", cells) for r, cells in rows}
    assert len(grid) == 12 and all(len(v) == 8 for v in grid.values())
    flat = [n for v in grid.values() for n in v]
    assert len(set(flat)) == len(flat) == 96
    for q in key["questions"]:
        m = re.fullmatch(r"What is the name in row (\d+), column ([A-H])\?", q["question"])
        assert m, q
        assert grid[int(m.group(1))]["ABCDEFGH".index(m.group(2))] == q["answer"], q


@PENDING
def test_harness_reads_p2_fixture():
    """B2 P2: the harness's P2 reads the checked-in fixture (its source names ``tests/fixtures/t0183``, the page and
    the answer key) and asks ``P2_QUESTIONS`` (20) questions per arm."""
    harness()
    text = HARNESS.read_text()
    assert "tests/fixtures/t0183" in text or ("fixtures" in text and "t0183" in text), (
        "T-0183 B2: the harness does not read tests/fixtures/t0183"
    )
    assert "answer_key.json" in text and "names.html" in text, "T-0183 B2: P2's page and answer key"
    assert api("P2_QUESTIONS") == 20
