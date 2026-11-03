"""T-0071a criterion 2 (the report half): ``scripts/gpu/cd_microbench.py``'s formatter refuses output without the
container digest, the commit and the regime line (CLAUDE.md: no number without its regime).

The measurement itself is GPU time (a1, ``/launch``); this module never runs it. The script is loaded by path
(``importlib.util.spec_from_file_location``), so importing it must not touch CUDA or start a measurement.

Assumed API (test-author ruling for T-0071a, 2026-10-03; to be recorded in docs/tasks/T-0071a.md by its owner):

``format_report(rows: Sequence[Mapping[str, Any]], *, container_digest: str | None, commit: str | None,
regime: str | None) -> str``
    ``rows`` are the per-(arm, bucket, component) results (keys at least ``arm``, ``bucket``, ``component``,
    ``median_ms``, ``p95_ms``; a memory row may carry ``peak_added_bytes``). Returns the report text, which contains
    the digest, the commit and the regime line verbatim and names every row's component. Raises ``ValueError``
    when:
    - ``container_digest`` is missing, empty, or not a content digest (``sha256:`` + 64 lowercase hex, optionally
      preceded by ``<repo>@`` as ``docker image inspect``'s ``RepoDigests`` prints it). The T-0072 probe's
      ``regime()`` reads it from ``ASTRA_CONTAINER_DIGEST`` and yields ``None`` when unset; T-0069's host helpers
      print ``<docker failed: ...>`` placeholders; the task files abbreviate digests with ``…``. None of those
      identifies an image.
    - ``commit`` is missing, empty, or not a full 40-hex commit id (``<git failed: ...>``, ``HEAD``).
    - ``regime`` is missing or blank.

The 64-hex digest below is a made-up value of the right shape (the pinned one is abbreviated in the task files).
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gpu" / "cd_microbench.py"

DIGEST = "sha256:" + "c2914767" + "0123456789abcdef" * 3 + "01234567"
REPO_DIGEST = "vllm/vllm-openai@" + DIGEST
COMMIT = "6c82d9e" + "0" * 33
REGIME = (
    "regime: container vllm/vllm-openai:v0.29.0, torch 2.13.0+cu130, RTX 5090, no engine; CL 256, V 262144, fp32, "
    "TF32 high, n 200 after 20 warm-up, CUDA events, seed 0, buckets 64/128/256/512"
)
ROWS: list[dict[str, Any]] = [
    {"arm": "perstep_mar", "bucket": 64, "component": "emission", "median_ms": 1.25, "p95_ms": 1.5},
    {
        "arm": "perstep_mar",
        "bucket": 64,
        "component": "marginals_pass2",
        "median_ms": 2.5,
        "p95_ms": 2.75,
    },
    {
        "arm": "perstep_mar",
        "bucket": 64,
        "component": "constrained_map",
        "median_ms": 7.15,
        "p95_ms": 7.5,
    },
    {
        "arm": "perstep_mar_sc",
        "bucket": 512,
        "component": "sc_matmul",
        "median_ms": 3.0,
        "p95_ms": 3.25,
        "peak_added_bytes": 1_685_000_000,
    },
]


def microbench() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.fail(f"T-0071a not implemented: {SCRIPT.relative_to(REPO)} does not exist")
    name = "_t0071a_cd_microbench"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    if not callable(getattr(m, "format_report", None)):
        pytest.fail("T-0071a not implemented: cd_microbench.format_report is missing")
    return m


def fmt(**over: Any) -> str:
    kw: dict[str, Any] = {"container_digest": DIGEST, "commit": COMMIT, "regime": REGIME}
    kw.update(over)
    out = microbench().format_report(ROWS, **kw)
    assert isinstance(out, str)
    return out


@pytest.mark.parametrize("digest", [DIGEST, REPO_DIGEST], ids=["bare", "repo-digest"])
def test_a_complete_report_carries_the_digest_the_commit_and_the_regime_line(digest: str) -> None:
    out = fmt(container_digest=digest)
    assert digest in out
    assert COMMIT in out
    assert any(REGIME in ln for ln in out.splitlines()), (
        "the regime line must appear on one line, verbatim"
    )
    for row in ROWS:
        assert row["component"] in out, f"row {row} missing from the report"


@pytest.mark.parametrize(
    "digest",
    [
        None,
        "",
        "   ",
        "<docker failed: Error: No such image>",
        "sha256:c2914767…",
        "sha256:c2914767",
        "None",
        "vllm/vllm-openai:v0.29.0",
        "sha256:" + "C" * 64,
    ],
    ids=[
        "none",
        "empty",
        "blank",
        "placeholder",
        "ellipsis",
        "short",
        "str-none",
        "tag-only",
        "upper-hex",
    ],
)
def test_the_formatter_refuses_a_report_without_a_container_digest(digest: str | None) -> None:
    with pytest.raises(ValueError):
        fmt(container_digest=digest)


@pytest.mark.parametrize(
    "commit",
    [None, "", "  ", "<git failed: not a git repository>", "HEAD", "6c82d9e0", "g" * 40],
    ids=["none", "empty", "blank", "placeholder", "symbolic", "short", "non-hex"],
)
def test_the_formatter_refuses_a_report_without_a_commit(commit: str | None) -> None:
    with pytest.raises(ValueError):
        fmt(commit=commit)


@pytest.mark.parametrize("regime", [None, "", " \n\t "], ids=["none", "empty", "blank"])
def test_the_formatter_refuses_a_report_without_a_regime_line(regime: str | None) -> None:
    with pytest.raises(ValueError):
        fmt(regime=regime)


# --------------------------------------------------------------------------- R5: labels (review round 1)
#
# Ruling R5: label ``whole_call`` as ``steps.prepare`` + δ* only (it is not the whole sampler call: the stock call
# is outside it), and flag a negative ``marginals_pass2_derived`` (full − pass 1 − H(q) is derived, not timed; a
# negative value means the subtraction is noise, not a cost). The formatter adds both, whatever the row's ``note``.


def _line(out: str, arm: str, component: str) -> str:
    hits = [ln for ln in out.splitlines() if arm in ln and component in ln]
    assert len(hits) == 1, f"want one report line for {arm}/{component}, got {hits}"
    return hits[0]


@pytest.mark.parametrize("arm", ["perstep_mf", "perstep_mar", "perstep_mar_sc"])
def test_a_per_step_whole_call_row_is_labelled_steps_prepare_plus_delta_star(arm: str) -> None:
    rows = [
        {"arm": arm, "bucket": 128, "component": "whole_call", "median_ms": 40.0, "p95_ms": 44.0}
    ]
    out = microbench().format_report(rows, container_digest=DIGEST, commit=COMMIT, regime=REGIME)
    assert "steps.prepare" in _line(out, arm, "whole_call"), (
        "R5: a per-step whole_call row must say it is steps.prepare + δ* only"
    )


@pytest.mark.parametrize(
    ("median", "flagged"), [(-0.4, True), (0.4, False)], ids=["negative", "positive"]
)
def test_a_negative_derived_pass_two_is_flagged_and_a_positive_one_is_not(
    median: float, flagged: bool
) -> None:
    """Both directions (a flag on every derived row would be as useless as none)."""
    rows = [
        {
            "arm": "components",
            "bucket": 64,
            "component": "marginals_pass2_derived",
            "median_ms": median,
            "p95_ms": abs(median),
        }
    ]
    out = microbench().format_report(rows, container_digest=DIGEST, commit=COMMIT, regime=REGIME)
    line = _line(out, "components", "marginals_pass2_derived")
    assert ("negative" in line.lower()) is flagged, line


# --------------------------------------------------------------------------- R4: one docker run, killed on timeout
#
# Ruling R4: the microbench probes ``docker info`` alone, runs the job once with a named container, kills it on
# timeout, records exit 1, never retries. Assumed API (test-author ruling, 2026-10-03):
#
# ``run_container(argv: list[str], *, name: str, timeout_s: float) -> tuple[int, str, str]``
#     ``argv`` is the ``docker run ...`` command line; the job runs exactly once with ``--name <name>``, directly
#     when ``docker info`` succeeds, else through ``sg docker -c``. Returns ``(rc, stdout, stderr)``. On timeout:
#     ``docker kill <name>`` (or ``docker rm -f <name>``), the partial stdout kept, ``rc == 1`` (the host's exit 1)
#     and ``stderr`` naming the timeout. A failed job is returned as is, never re-run.
#
# ``subprocess`` is replaced in the script's namespace by a double that delegates every name to the real module
# except ``run`` (recorded and scripted) and the process-spawning names (``Popen``, ``call``, ``check_call``,
# ``check_output``), which fail the row by name: no docker command reaches this box.


class _Subprocess:
    def __init__(self, script: Any) -> None:
        self.script = script
        self.calls: list[list[str]] = []

    def __getattr__(self, name: str) -> Any:
        if name in ("Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"):
            raise AssertionError(
                f"the double fakes subprocess.run only; the script reached subprocess.{name}"
            )
        return getattr(subprocess, name)

    def run(self, argv: Any, *a: Any, **k: Any) -> Any:
        argv = list(argv)
        self.calls.append(argv)
        return self.script(_docker_words(argv), argv, k)


def _docker_words(argv: list[str]) -> list[str]:
    """``sg docker -c '<cmd>'`` → the inner words; anything else as is."""
    if argv[:3] == ["sg", "docker", "-c"]:
        return shlex.split(argv[3])
    return argv


def _is_run(words: list[str]) -> bool:
    return words[:2] == ["docker", "run"]


def _is_kill(words: list[str], name: str) -> bool:
    return (
        words[:1] == ["docker"]
        and name in words
        and ("kill" in words[1:2] or (words[1:2] == ["rm"] and "-f" in words))
    )


JOB = [
    "docker",
    "run",
    "--rm",
    "--gpus",
    "all",
    "--network",
    "none",
    "img@" + DIGEST,
    "python3",
    "x.py",
]
NAME = "t0071a-microbench-6c82d9e"
PARTIAL = 'ASTRA_MICROBENCH_JSON {"bucket": 64}\n'


def _script(*, info_rc: int = 0, job: str = "ok") -> Any:
    def script(words: list[str], argv: list[str], kw: dict[str, Any]) -> Any:
        if words[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, info_rc, "", "")
        if _is_run(words):
            if job == "timeout":
                raise subprocess.TimeoutExpired(
                    argv, kw.get("timeout", 0), output=PARTIAL, stderr=""
                )
            if job == "fail":
                return subprocess.CompletedProcess(
                    argv, 125, "", "docker: Error response from daemon"
                )
            return subprocess.CompletedProcess(argv, 0, PARTIAL + "done\n", "")
        if words[:1] == ["docker"] and words[1:2] in (["kill"], ["rm"]):
            return subprocess.CompletedProcess(argv, 0, NAME + "\n", "")
        raise AssertionError(f"unexpected command {argv}")

    return script


def _run_container(monkeypatch: pytest.MonkeyPatch, script: Any) -> tuple[Any, _Subprocess]:
    m = microbench()
    fn = getattr(m, "run_container", None)
    if not callable(fn):
        pytest.fail("T-0071a R4 not implemented: cd_microbench.run_container is missing")
    fake = _Subprocess(script)
    monkeypatch.setattr(m, "subprocess", fake)
    return fn(list(JOB), name=NAME, timeout_s=5.0), fake


def _runs(fake: _Subprocess) -> list[list[str]]:
    return [w for w in map(_docker_words, fake.calls) if _is_run(w)]


@pytest.mark.parametrize("info_rc", [0, 1], ids=["docker-direct", "via-sg"])
def test_a_timed_out_job_is_killed_by_name_and_never_retried(
    monkeypatch: pytest.MonkeyPatch, info_rc: int
) -> None:
    """Mutants killed: no ``--name`` (nothing to kill), no kill on timeout (the container keeps the GPU), the old
    fall-back that re-ran the job through ``sg`` after a failure (two runs), rc 0 on timeout."""
    (rc, out, err), fake = _run_container(monkeypatch, _script(info_rc=info_rc, job="timeout"))
    runs = _runs(fake)
    assert len(runs) == 1, f"the job ran {len(runs)} times: {fake.calls}"
    i = runs[0].index("--name")
    assert runs[0][i + 1] == NAME
    words = list(map(_docker_words, fake.calls))
    after = words[words.index(runs[0]) + 1 :]
    assert any(_is_kill(w, NAME) for w in after), (
        f"no docker kill {NAME} after the timeout: {fake.calls}"
    )
    assert rc == 1
    assert "timed out" in err.lower() or "timeout" in err.lower(), err
    assert PARTIAL.strip() in out, "the partial stdout was dropped"


def test_a_failed_job_is_returned_once_never_re_run(monkeypatch: pytest.MonkeyPatch) -> None:
    (rc, _out, _err), fake = _run_container(monkeypatch, _script(job="fail"))
    assert len(_runs(fake)) == 1, fake.calls
    assert rc == 125
    assert not any(_is_kill(w, NAME) for w in map(_docker_words, fake.calls))


def test_a_clean_job_runs_once_with_its_name_and_is_not_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (rc, out, _err), fake = _run_container(monkeypatch, _script())
    runs = _runs(fake)
    assert len(runs) == 1 and NAME in runs[0]
    assert rc == 0 and "done" in out
    assert not any(_is_kill(w, NAME) for w in map(_docker_words, fake.calls))
    assert ["docker", "info"] in [w[:2] for w in map(_docker_words, fake.calls)], (
        "no docker info probe"
    )
