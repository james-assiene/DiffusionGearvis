"""B16: what the canvas tap costs (T-0011 criteria 13-14). Pure: no I/O, no engine, no GPU.

``evaluate`` decides what 30 interleaved A/B pairs mean; ``format_report`` prints them with their
regime; ``boot_config_lines`` reduces an engine boot log to the configuration lines that must not
move between the plugin-on reference boot and the tapped one. The tier-3 driver that produces the
numbers is ``scripts/gpu/b16_tap.py`` (``/launch`` only).

Statistics (criterion 13, R4):

* ``delta_pct`` -- median(B) against median(A), as a percentage of median(A). A is tap-off
  (``vllm_xargs={"astra_tap": false}``), B the default, one entry per pair in issue order.
* ``control_pct`` -- the A/A' control: the first half of the A samples against the second half,
  the same statistic, absolute. It is what the harness would call a difference between two arms
  that are in fact identical.
* ``verdict`` -- ``"fail"`` when the observed event rate exceeds 30 Hz; otherwise
  ``"not_resolvable"`` when the control is at least as wide as the delta (a wide control never
  reads as a pass); otherwise ``"fail"`` above the threshold and ``"pass"`` at or below it.
"""

from __future__ import annotations

import ast
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from local_astra.bench.thresholds import THRESHOLDS

B16_THRESHOLD_PCT: float = next(
    t.limit for t in THRESHOLDS if t.bench == "B16" and t.metric == "tap_overhead_pct"
)
#: docs/03 §7 B16: observed event rate per request
MAX_EVENT_HZ = 30.0
#: float slack for "delta exactly at the threshold" (100 * 3 / 100 is not always exactly 3.0)
_EPS = 1e-9

Verdict = Literal["pass", "fail", "not_resolvable"]


@dataclass(frozen=True)
class B16Result:
    n_pairs: int
    median_a_ms: float
    median_b_ms: float
    p95_a_ms: float
    p95_b_ms: float
    delta_pct: float
    control_pct: float
    threshold_pct: float
    observed_hz: float | None
    verdict: Verdict
    reason: str


def _p95(xs: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), 95))


def evaluate(
    a_ms: Sequence[float],
    b_ms: Sequence[float],
    *,
    threshold_pct: float = B16_THRESHOLD_PCT,
    observed_hz: float | None = None,
) -> B16Result:
    """Paired tap-off (A) / tap-on (B) end-to-end latencies -> the B16 verdict."""
    a, b = [float(x) for x in a_ms], [float(x) for x in b_ms]
    if not a or not b:
        raise ValueError("both arms need samples")
    if len(a) != len(b):
        raise ValueError(f"unpaired arms: {len(a)} tap-off vs {len(b)} tap-on samples")
    if len(a) < 2:
        raise ValueError("the A/A' control needs at least two tap-off samples")
    med_a, med_b = statistics.median(a), statistics.median(b)
    delta = 100.0 * (med_b - med_a) / med_a
    half = len(a) // 2
    first, second = statistics.median(a[:half]), statistics.median(a[half:])
    control = abs(100.0 * (second - first) / first)
    n = len(a)
    verdict: Verdict
    if observed_hz is not None and observed_hz > MAX_EVENT_HZ:
        verdict = "fail"
        reason = f"event rate {observed_hz:.1f} Hz > {MAX_EVENT_HZ:.0f} Hz per request"
    elif control >= abs(delta):
        verdict = "not_resolvable"
        reason = (
            f"not resolvable at n={n}: the A/A' control ({control:.2f} %) is at least as wide "
            f"as the delta ({delta:+.2f} %)"
        )
    elif delta > threshold_pct + _EPS:
        verdict = "fail"
        reason = f"delta {delta:+.2f} % > {threshold_pct:.1f} % (control {control:.2f} %)"
    else:
        verdict = "pass"
        reason = f"delta {delta:+.2f} % <= {threshold_pct:.1f} %, control {control:.2f} % narrower"
    return B16Result(
        n_pairs=n,
        median_a_ms=med_a,
        median_b_ms=med_b,
        p95_a_ms=_p95(a),
        p95_b_ms=_p95(b),
        delta_pct=delta,
        control_pct=control,
        threshold_pct=threshold_pct,
        observed_hz=observed_hz,
        verdict=verdict,
        reason=reason,
    )


def format_report(result: B16Result, *, regime: str) -> str:
    hz = "n/a" if result.observed_hz is None else f"{result.observed_hz:.1f} Hz"
    return "\n".join(
        (
            f"B16 tap overhead -- {result.verdict.upper()}: {result.reason}",
            f"regime: {regime}",
            f"pairs: n={result.n_pairs} (A tap off, B tap on, interleaved)",
            f"median: A {result.median_a_ms:.1f} ms, B {result.median_b_ms:.1f} ms",
            f"p95: A {result.p95_a_ms:.1f} ms, B {result.p95_b_ms:.1f} ms",
            f"delta: {result.delta_pct:+.2f} % (threshold {result.threshold_pct:.1f} %)",
            f"control (A/A', first vs second half of A): {result.control_pct:.2f} %",
            f"max observed event rate per request: {hz}",
        )
    )


# --------------------------------------------------------------------------- boot reference

#: the explicit engine arguments the plugin-on boot must keep (`.claude/rules/gpu-services.md`:
#: the four flags that keep plugin-on identical to plugin-off, plus the KV and length budget)
BOOT_ARGS = (
    "attention_backend",
    "diffusion_config",
    "max_num_seqs",
    "override_generation_config",
    "kv_cache_memory_bytes",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "hf_overrides",
)

_NON_DEFAULT = re.compile(r"non-default args: (\{.*\})\s*$")
_ENGINE_KV = re.compile(r"Initializing a V1 LLM engine .*?\bkv_cache_dtype=([^,\s]+)")
_ARCH = re.compile(r"Resolved architecture: (\S+)")
_ATTN = re.compile(r"Using AttentionBackendEnum\.(\w+) backend")
_MOE = re.compile(r"Using '([\w-]+)' NvFp4 MoE backend")
_KV = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
    r"([\d.]+)x"
)
_MARLIN = re.compile(
    r"does not have native support for FP4|no FP4 support|leveraging the Marlin kernel", re.I
)


def _found(pattern: re.Pattern[str], lines: Iterable[str]) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for ln in lines:
        m = pattern.search(ln)
        if m:
            out.add(m.groups())
    return out


def boot_config_lines(log: str) -> list[str]:
    """The configuration a boot log states, one ``key: value`` line per fact, sorted.

    Values only -- never timestamps, pids, paths or timings -- so two boots of one configuration
    give the same lines, and a fact that is missing is reported as ``<missing>`` rather than
    dropped. Items: KV dtype (engine config), attention backend, the NvFp4 MoE backend and whether
    a Marlin/no-FP4 fallback was announced, KV tokens and concurrency, the resolved architecture,
    and the explicit arguments in ``BOOT_ARGS``.
    """
    lines = log.splitlines()
    out: set[str] = set()

    def add(key: str, values: set[tuple[str, ...]]) -> None:
        if not values:
            out.add(f"{key}: <missing>")
        for v in values:
            out.add(f"{key}: {' / '.join(v)}")

    add("kv_cache_dtype (engine)", _found(_ENGINE_KV, lines))
    add("attention backend", _found(_ATTN, lines))
    add("nvfp4 moe backend", _found(_MOE, lines))
    add("kv cache tokens / max len / concurrency", _found(_KV, lines))
    add("resolved architecture", _found(_ARCH, lines))
    marlin = any(_MARLIN.search(ln) for ln in lines)
    out.add(f"fp4 marlin fallback: {'PRESENT' if marlin else 'absent'}")

    args: dict[str, object] | None = None
    for ln in lines:
        m = _NON_DEFAULT.search(ln)
        if m:
            try:
                parsed = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                args = parsed
    for key in BOOT_ARGS:
        if args is None or key not in args:
            out.add(f"arg {key}: <missing>")
        else:
            out.add(f"arg {key}: {args[key]!r}")
    return sorted(out)


BOOT_MARKER = "non-default args:"


def last_boot(log: str) -> str:
    """R16: the last boot of an append-mode engine log -- the text from the last line carrying
    vLLM's ``non-default args:`` (printed once per start, before every config line) to the end.
    A log with no boot raises ``ValueError`` (an empty slice would read as a boot missing every
    fact)."""
    start = None
    pos = 0
    for ln in log.splitlines(keepends=True):
        if BOOT_MARKER in ln:
            start = pos
        pos += len(ln)
    if start is None:
        raise ValueError(f"no boot in this log: no line carries {BOOT_MARKER!r}")
    return log[start:]


# --------------------------------------------------------------------------- R12: release lines

#: the marker the plugin writes (``astra_vllm_plugin.tap.RELEASE_LINE``; the plugin cannot be
#: imported here, so the literal is repeated and the bench round-trip row pins the two together)
TAP_RELEASE_MARKER = "canvas tap release:"
TAP_COUNTERS = ("submitted", "published", "dropped", "errors", "commit_overrun")
_FIELD = re.compile(r"\b(\w+)=(\S+)")


def parse_tap_release_lines(log: str) -> list[dict[str, str | int]]:
    """The tap's per-release counter lines in an engine log, in log order (R12).

    One dict per line: ``req_id`` (str) and the five counters (int), read only from the text after
    the marker (a log prefix can hold ``key=value`` pairs of its own). A line that carries the
    marker but no ``req_id`` or a missing or non-numeric counter raises ``ValueError`` naming the
    line: the ``commit_overrun == 0`` gate must never pass because a bad line was skipped.
    """
    rows: list[dict[str, str | int]] = []
    for ln in log.splitlines():
        i = ln.find(TAP_RELEASE_MARKER)
        if i < 0:
            continue
        fields = dict(_FIELD.findall(ln[i + len(TAP_RELEASE_MARKER) :]))
        bad = [k for k in TAP_COUNTERS if not fields.get(k, "").isdigit()]
        if "req_id" not in fields or bad:
            missing = (["req_id"] if "req_id" not in fields else []) + bad
            raise ValueError(f"malformed tap release line (bad or missing {missing}): {ln!r}")
        row: dict[str, str | int] = {"req_id": fields["req_id"]}
        row.update({k: int(fields[k]) for k in TAP_COUNTERS})
        rows.append(row)
    return rows
