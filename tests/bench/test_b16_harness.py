"""T-0011 criterion 13: the *statistics* of the B16 tap-overhead harness, on synthetic samples.

The measurement itself is tier 3 (one booted engine, `/launch`, 30 interleaved A/B pairs). What
can and must be tested here is the part that decides what the numbers mean, because that is where
a 3 % bar quietly turns into "it passed": ``local_astra.bench.b16_tap.evaluate``.

    evaluate(a_ms, b_ms, *, threshold_pct=<B16 threshold>, observed_hz=None) -> B16Result

* ``a_ms`` is the tap-off arm (``vllm_xargs={"astra_tap": false}``), ``b_ms`` the tap-on arm, one
  entry per pair, in the order they were issued (the arms are interleaved inside one boot).
* ``delta_pct`` is the A-vs-B difference of the medians as a percentage of A.
* ``control_pct`` is the **A/A′ control**: the tap-off samples split in half (first half against
  second half), the same statistic. It measures what the harness would report as a "difference"
  between two arms that are in fact identical.
* ``verdict`` is ``"pass"`` only when the control is *narrower* than the delta **and** the delta
  is within the threshold; when the control is as wide as the delta the answer is
  ``"not_resolvable"`` — never ``"pass"``.

Every expected number below is computed by hand from the sample arrays (they are constant runs,
so their medians are exact) and written into the test; nothing re-implements the statistic.

Marked ``bench`` (it is the bench harness) and ``unit`` in spirit: no I/O, no GPU, no engine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.bench.thresholds import THRESHOLDS
from tests.fakes.diffusion_sampler import attr, load_module

pytestmark = pytest.mark.bench

#: docs/03 §7 B16, as recorded in local_astra/bench/thresholds.py — read, not duplicated
B16_LIMIT = next(t.limit for t in THRESHOLDS if t.bench == "B16" and t.metric == "tap_overhead_pct")


def _evaluate(a: list[float], b: list[float], **kwargs: Any) -> Any:
    mod = load_module("local_astra.bench.b16_tap", "13")
    return attr(mod, "evaluate", "13")(a, b, **kwargs)


def test_the_threshold_is_the_one_in_thresholds_py() -> None:
    """CLAUDE.md: a bench threshold lives in one place and moves only with a task entry."""
    assert B16_LIMIT == 3.0
    a = [100.0] * 30
    result = _evaluate(a, [102.0] * 30)
    assert result.threshold_pct == B16_LIMIT


def test_paired_delta_and_control_on_clean_samples() -> None:
    """n=30 pairs, A ≡ 100 ms, B ≡ 102 ms: delta exactly 2.0 %, control exactly 0.0 %."""
    result = _evaluate([100.0] * 30, [102.0] * 30)
    assert result.n_pairs == 30
    assert result.median_a_ms == 100.0
    assert result.median_b_ms == 102.0
    assert result.delta_pct == pytest.approx(2.0)
    assert result.control_pct == pytest.approx(0.0)
    assert result.verdict == "pass"


def test_p95_is_reported_for_both_arms() -> None:
    """Criterion 13 asks for the paired median *and* p95 of end-to-end request latency.

    A = 100 ms twenty times and 200 ms once (n=21), B = 110 ms throughout. n=21 is chosen so the
    two usual conventions agree and the test does not silently pick one: nearest-rank gives the
    ``ceil(0.95 * 21) = 20``-th value and linear interpolation gives index ``0.95 * 20 = 19``
    exactly — both are 100.0 ms.
    """
    a = [100.0] * 20 + [200.0]
    result = _evaluate(a, [110.0] * 21)
    assert result.p95_a_ms == pytest.approx(100.0)
    assert result.p95_b_ms == pytest.approx(110.0)


def test_a_delta_above_the_threshold_fails() -> None:
    """A ≡ 100 ms, B ≡ 105 ms → 5.0 % > 3.0 %."""
    result = _evaluate([100.0] * 30, [105.0] * 30)
    assert result.delta_pct == pytest.approx(5.0)
    assert result.verdict == "fail"


def test_a_delta_exactly_at_the_threshold_passes() -> None:
    """The bar is "≤ 3.0 %" (docs/03 §7), so 103 ms against 100 ms is a pass, not a fail."""
    result = _evaluate([100.0] * 30, [103.0] * 30)
    assert result.delta_pct == pytest.approx(3.0)
    assert result.verdict == "pass"


def test_a_faster_tap_arm_is_not_a_failure() -> None:
    """Noise can put B below A; that is not a regression, it is an unresolvable measurement."""
    result = _evaluate([100.0] * 30, [99.0] * 30)
    assert result.delta_pct == pytest.approx(-1.0)
    assert result.verdict != "fail"


def test_a_one_percent_slowdown_under_a_four_percent_control_is_not_resolvable() -> None:
    """Criterion 13, the adversarial case: a wide control must **never** read as a pass.

    A = fifteen 100 ms samples then fifteen 104 ms samples (the engine drifted during the run):
    median(A) = (100 + 104) / 2 = 102.0, first half median 100.0, second half median 104.0, so
    the A/A′ control spread is (104 - 100) / 100 = 4.0 %. B ≡ 103.02 ms, i.e. exactly 1 % above
    median(A). A harness that only compared 1.0 % against the 3 % bar would call this a pass.
    """
    a = [100.0] * 15 + [104.0] * 15
    result = _evaluate(a, [103.02] * 30)
    assert result.median_a_ms == pytest.approx(102.0)
    assert result.delta_pct == pytest.approx(1.0, abs=1e-6)
    assert result.control_pct == pytest.approx(4.0)
    assert result.verdict == "not_resolvable", (
        f"verdict was {result.verdict!r}: a control wider than the delta cannot resolve a 3 % bar"
    )
    assert "not resolvable" in result.reason.lower() or "n=30" in result.reason


def test_a_wide_control_does_not_rescue_a_real_regression() -> None:
    """The refusal path is not an escape hatch: it is reported instead of a *pass*, not a *fail*.

    Same drifting A arm (control 4.0 %), but B ≡ 120 ms — a 17.6 % delta, far wider than the
    control, so the measurement *is* resolvable and it is a failure.
    """
    a = [100.0] * 15 + [104.0] * 15
    result = _evaluate(a, [120.0] * 30)
    assert result.control_pct == pytest.approx(4.0)
    assert result.delta_pct == pytest.approx(100 * (120.0 - 102.0) / 102.0)
    assert result.verdict == "fail"


def test_unpaired_or_empty_arms_are_refused() -> None:
    """Pairing is the whole design (one boot, interleaved); unpaired input is a harness bug."""
    with pytest.raises(ValueError):
        _evaluate([100.0] * 30, [100.0] * 29)
    with pytest.raises(ValueError):
        _evaluate([], [])


def test_an_event_rate_above_30_hz_is_a_failure() -> None:
    """docs/03 §7 B16: observed event rate ≤ 30 Hz per request, whatever the latency said."""
    ok = _evaluate([100.0] * 30, [101.0] * 30, observed_hz=29.5)
    assert ok.verdict == "pass"
    hot = _evaluate([100.0] * 30, [101.0] * 30, observed_hz=31.0)
    assert hot.verdict == "fail"
    assert "hz" in hot.reason.lower()


def test_the_report_carries_the_regime_line() -> None:
    """CLAUDE.md: a number without its regime is not comparable to anything later."""
    mod = load_module("local_astra.bench.b16_tap", "13")
    result = _evaluate([100.0] * 30, [102.0] * 30)
    text = attr(mod, "format_report", "13")(result, regime="profile lean · commit abc1234")
    assert "profile lean · commit abc1234" in text
    for token in ("delta", "control", "median", "p95"):
        assert token in text.lower(), f"the report does not mention {token}"


# --------------------------------------------------------------------------- criterion 14: boot reference

#: lines 1-48 of logs/vllm-coldboot-bd8001b.log (plugin on, no tap); provenance in its MANIFEST.md
BOOT_EXCERPT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "engine"
    / "vllm-coldboot-bd8001b.excerpt.log"
)


def _boot_config_lines(log: str) -> set[str]:
    mod = load_module("local_astra.bench.b16_tap", "14")
    got = attr(mod, "boot_config_lines", "14")(log)
    assert not isinstance(got, str), (
        "boot_config_lines returns a collection of lines, not one string"
    )
    return set(got)


def _excerpt() -> str:
    return BOOT_EXCERPT.read_text()


def test_boot_config_lines_extracts_every_item_criterion_14_names() -> None:
    """Criterion 14: kv dtype, attention backend, FlashInfer CUTLASS, KV tokens and concurrency,
    the resolved architecture and the explicit flags -- each value as the bd8001b boot logged it
    (read off the fixture by hand: ``kv_cache_dtype=bfloat16``, ``TRITON_ATTN``,
    ``'FLASHINFER_CUTLASS' NvFp4 MoE backend``, ``31,605 tokens`` / ``1.93x``,
    ``Resolved architecture: AstraDiffusionGemma``, ``kv_cache_memory_bytes 4294967296``,
    ``max_model_len 16384``, ``max_num_seqs 3``)."""
    lines = _boot_config_lines(_excerpt())
    assert lines, "nothing extracted from the reference boot log"
    text = "\n".join(sorted(lines))
    for needle in (
        "bfloat16",
        "TRITON_ATTN",
        "FLASHINFER_CUTLASS",
        "1.93",
        "AstraDiffusionGemma",
        "4294967296",
        "16384",
    ):
        assert needle in text, f"{needle!r} missing from the extracted boot config:\n{text}"
    assert "31,605" in text or "31605" in text, f"KV cache tokens missing:\n{text}"
    assert any("max_num_seqs" in ln and "3" in ln for ln in lines), f"max_num_seqs missing:\n{text}"


@pytest.mark.parametrize(
    "mutation",
    ["marlin_moe_backend", "marlin_fallback_warning", "fp8_kv_cache"],
)
def test_a_config_change_changes_the_extracted_set(mutation: str) -> None:
    """Criterion 14, the guard's "too narrow" side: a Marlin fallback (chosen as the MoE backend,
    or announced by a warning line inserted after the CUTLASS line) or a different KV dtype must
    make the set differ from the reference -- otherwise T-0069's boot comparison passes a boot
    that silently fell off FlashInfer CUTLASS."""
    ref = _excerpt()
    if mutation == "marlin_moe_backend":
        bad = ref.replace(
            "Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend", "Using 'MARLIN' NvFp4 MoE backend"
        )
    elif mutation == "marlin_fallback_warning":
        out = []
        for ln in ref.splitlines(keepends=True):
            out.append(ln)
            if "NvFp4 MoE backend" in ln:
                out.append(
                    "(EngineCore pid=326) WARNING 09-17 15:42:51 [marlin_utils_fp4.py:204] Your GPU "
                    "does not have native support for FP4 computation but FP4 quantization is being "
                    "used. Weight-only FP4 compression will be used leveraging the Marlin kernel.\n"
                )
        bad = "".join(out)
    else:
        bad = ref.replace("kv_cache_dtype=bfloat16", "kv_cache_dtype=fp8").replace(
            "'kv_cache_dtype': 'bfloat16'", "'kv_cache_dtype': 'fp8'"
        )
    assert bad != ref, "the mutation did not apply to the fixture (fixture drifted?)"
    assert _boot_config_lines(bad) != _boot_config_lines(ref), f"{mutation} left the set unchanged"


def test_timestamp_pid_and_unrelated_noise_do_not_change_the_extracted_set() -> None:
    """Criterion 14, the guard's "too wide" side: another boot of the same config has other
    timestamps, other pids, other load timings. None of that may change the set, or the reference
    comparison fails every boot."""
    ref = _excerpt()
    noisy = re.sub(r"\d\d-\d\d \d\d:\d\d:\d\d", "10-03 08:01:02", ref)
    noisy = noisy.replace("pid=326", "pid=4412").replace("APIServer pid=1)", "APIServer pid=77)")
    noisy = noisy.replace("Loading weights took 6.45 seconds", "Loading weights took 9.01 seconds")
    noisy = noisy.replace(
        "vllm_dist_880e3111f1fc462a9c8ed5e279998f97", "vllm_dist_0123456789abcdef"
    )
    assert noisy != ref
    assert _boot_config_lines(noisy) == _boot_config_lines(ref)


# --------------------------------------------------------------------------- review round 1


def test_a_downward_drift_in_the_control_is_as_wide_as_an_upward_one() -> None:
    """Criterion 13 / R4, the mirror of the 100 -> 104 row: the engine drifts *down* during the run.

    A = fifteen 104 ms then fifteen 100 ms: median(A) = 102.0, first-half median 104.0, second-half
    100.0, so the control is |100 - 104| / 104 = 3.846 % (the first half is the reference, as in the
    upward row). B = 103.02 ms, a 1.0 % delta. A control computed without the absolute value is
    -3.85 %, which is "narrower" than any delta and turns this into a pass."""
    a = [104.0] * 15 + [100.0] * 15
    result = _evaluate(a, [103.02] * 30)
    assert result.median_a_ms == pytest.approx(102.0)
    assert result.delta_pct == pytest.approx(1.0, abs=1e-6)
    assert result.control_pct == pytest.approx(100.0 * 4.0 / 104.0)
    assert result.verdict == "not_resolvable", f"verdict {result.verdict!r}: {result.reason}"


def test_a_control_exactly_as_wide_as_the_delta_is_not_resolvable() -> None:
    """R4's boundary: ``not_resolvable`` when ``control_pct >= |delta_pct|`` -- equality included.

    Binary-exact numbers so the equality is exact in float64: A = fifteen 64 ms then fifteen 80 ms
    (median 72, halves 64 and 80: control 100 * 16 / 64 = 25.0 %), B = 90 ms (delta
    100 * 18 / 72 = 25.0 %). With a strict ``>`` the 25 % delta would read as a resolvable fail."""
    a = [64.0] * 15 + [80.0] * 15
    result = _evaluate(a, [90.0] * 30)
    assert result.control_pct == 25.0 and result.delta_pct == 25.0, (
        result.control_pct,
        result.delta_pct,
    )
    assert result.verdict == "not_resolvable", f"verdict {result.verdict!r}: {result.reason}"


#: R13: the four explicit flags `.claude/rules/gpu-services.md` names, each with the value the
#: bd8001b boot logged (read off the fixture's ``non-default args`` line by hand)
FOUR_FLAGS = {
    "attention_backend": ("TRITON_ATTN",),
    "diffusion_config": ("canvas_length", "256", "max_denoising_steps", "48"),
    "max_num_seqs": ("3",),
    "override_generation_config": ("max_new_tokens", "None"),
}


def _flag_lines(lines: set[str], flag: str) -> list[str]:
    pat = re.compile(flag.replace("_", "[_-]"))
    return [ln for ln in lines if pat.search(ln)]


@pytest.mark.parametrize("flag", sorted(FOUR_FLAGS))
def test_boot_config_lines_carry_each_of_the_four_explicit_flags_with_its_value(flag: str) -> None:
    """Criterion 14 / R13: "the four explicit flags" are ``--attention-backend``,
    ``--diffusion-config``, ``--max-num-seqs`` and ``--override-generation-config``. Each must be
    in the extracted set *under its own name* with the value the reference boot logged -- the
    attention backend's ``TRITON_ATTN`` from the ``Using AttentionBackendEnum`` line does not count
    for the flag."""
    lines = _boot_config_lines(_excerpt())
    named = _flag_lines(lines, flag)
    assert named, f"--{flag.replace('_', '-')} is not in the extracted set:\n" + "\n".join(
        sorted(lines)
    )
    tokens = FOUR_FLAGS[flag]
    hit = [ln for ln in named if all(t in ln for t in tokens)]
    assert hit, f"--{flag.replace('_', '-')} extracted without its value {tokens}: {named}"
    if flag == "max_num_seqs":
        assert any(re.search(r"\b3\b", ln.split("max", 1)[1]) for ln in hit), hit


_FLAG_MUTATIONS = {
    "attention_backend": (
        "'attention_backend': 'TRITON_ATTN'",
        "'attention_backend': 'FLASH_ATTN'",
    ),
    "diffusion_config": ("'max_denoising_steps': 48}", "'max_denoising_steps': 32}"),
    "max_num_seqs": ("'max_num_seqs': 3,", "'max_num_seqs': 4,"),
    "override_generation_config": (
        "'override_generation_config': {'max_new_tokens': None}, ",
        "",
    ),
}


@pytest.mark.parametrize("flag", sorted(_FLAG_MUTATIONS))
def test_a_changed_or_dropped_explicit_flag_changes_the_extracted_set(flag: str) -> None:
    """Criterion 14 / R13, "too narrow" side per flag: a boot with a different value for one of the
    four flags (or without ``--override-generation-config`` at all) must not compare equal to the
    reference. Only the ``non-default args`` line is edited, so nothing else can carry the change."""
    ref = _excerpt()
    old, new = _FLAG_MUTATIONS[flag]
    assert ref.count(old) == 1, f"fixture drifted: {old!r} occurs {ref.count(old)} times"
    bad = ref.replace(old, new)
    assert _boot_config_lines(bad) != _boot_config_lines(ref), f"{flag}: the set did not change"


# --------------------------------------------------------------------------- R12: release lines


def _parse_tap_release_lines(log: str) -> Any:
    mod = load_module("local_astra.bench.b16_tap", "14")
    return attr(mod, "parse_tap_release_lines", "14")(log)


def test_the_driver_parses_the_tap_release_lines_the_tap_writes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R12: the tier-3 driver parses the tap's per-release counter lines out of the engine log and
    requires ``commit_overrun == 0``. Round trip with no format fixed here: the tap's own line (the
    plugin over the fakes, ``release_log_scenario``) is wrapped in a vLLM engine-log prefix,
    interleaved with the bd8001b boot excerpt, and parsed by
    ``local_astra.bench.b16_tap.parse_tap_release_lines(log) -> list[dict]`` -- one dict per
    release line, in log order, with ``req_id`` and the five counters as ints. Expected values are
    the scenario's hand-computed ones."""
    import logging

    from tests.fakes.diffusion_sampler import (
        RELEASE_LOG_EXPECTED,
        RELEASE_LOG_SUBMITTED,
        TAP_COUNTERS,
        release_log_scenario,
    )

    caplog.set_level(logging.DEBUG)
    release_log_scenario()
    tap_lines = [
        f"(EngineCore pid=326) {r.levelname} 10-02 12:00:00 [tap.py:{r.lineno}] {r.getMessage()}"
        for r in caplog.records
        # R14: the plugin logs under vLLM's `vllm` hierarchy, so vLLM's own handler writes it
        if r.name.startswith("vllm.astra_vllm_plugin")
    ]
    assert tap_lines, "the tap logged nothing at the release (R12)"
    boot = _excerpt().splitlines()
    log = "\n".join(boot[:20] + tap_lines + boot[20:]) + "\n"
    got = _parse_tap_release_lines(log)
    assert isinstance(got, list) and len(got) == 1, f"one release, one parsed line: {got!r}"
    row = got[0]
    assert row["req_id"] == "req-log"
    assert row["submitted"] in RELEASE_LOG_SUBMITTED
    assert {k: row[k] for k in RELEASE_LOG_EXPECTED} == RELEASE_LOG_EXPECTED
    assert set(TAP_COUNTERS) <= set(row)


def test_a_boot_log_without_tap_lines_parses_to_nothing() -> None:
    """R12, the "too wide" side: the reference boot (plugin on, no tap) has no release lines."""
    assert _parse_tap_release_lines(_excerpt()) == []


# --------------------------------------------------------------------------- review round 2


def _tap_release_line(caplog: pytest.LogCaptureFixture) -> str:
    """The tap's own release line for ``req-log`` (``release_log_scenario``), as the engine logs it."""
    import logging

    from tests.fakes.diffusion_sampler import release_log_scenario

    caplog.set_level(logging.DEBUG)
    release_log_scenario()
    got = [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("vllm.astra_vllm_plugin")
        and "req-log" in r.getMessage()
        and "commit_overrun" in r.getMessage()
    ]
    assert len(got) == 1, got
    return f"(EngineCore pid=326) INFO 10-02 12:00:00 [tap.py:559] {got[0]}"


def test_the_parser_reads_fields_only_after_the_marker(caplog: pytest.LogCaptureFixture) -> None:
    """The counters are read from the tap's text, never from the log prefix: a prefix that happens to
    contain ``key=value`` pairs (a path, another logger's fields) must not override them."""
    line = _tap_release_line(caplog)
    poisoned = line.replace(
        "[tap.py:559] ",
        "[tap.py:559] req_id=evil submitted=999 published=999 dropped=999 errors=999 "
        "commit_overrun=999 | ",
    )
    rows = _parse_tap_release_lines(poisoned + "\n")
    assert len(rows) == 1 and rows[0]["req_id"] == "req-log", rows
    assert rows[0]["commit_overrun"] == 0 and rows[0]["errors"] == 1, rows


@pytest.mark.parametrize("damage", ["non-numeric", "missing", "no-req-id"])
def test_a_malformed_tap_release_line_raises_instead_of_being_skipped(
    damage: str, caplog: pytest.LogCaptureFixture
) -> None:
    """R12, never silent: a line that carries the tap's release marker but a non-numeric or missing
    counter (or no ``req_id``) is a broken tap or a broken parser. The driver's ``commit_overrun == 0``
    gate must not pass because the one bad line was skipped -- ``parse_tap_release_lines`` raises
    ``ValueError`` naming the line."""
    import re

    line = _tap_release_line(caplog)
    if damage == "non-numeric":
        bad = re.sub(r"commit_overrun\s*[=:]\s*\d+", "commit_overrun=x", line)
    elif damage == "missing":
        bad = re.sub(r"\s*\bdropped\s*[=:]\s*\d+", "", line)
    else:
        bad = re.sub(r"\s*\breq_id\s*[=:]\s*\S+", "", line)
    assert bad != line, "fixture: the damage did not apply"
    log = _excerpt() + line + "\n" + bad + "\n"
    with pytest.raises(ValueError):
        _parse_tap_release_lines(log)


# R16: the tier-3 driver reads only the last boot of an append-mode logs/vllm.log.
#
# The pure function the driver slices with is ``local_astra.bench.b16_tap.last_boot(log) -> str``.
# Boundary: a boot starts at vLLM's ``non-default args:`` line (api_server, printed once per
# start, before every config line); ``last_boot`` returns the text from the last such line to the
# end. A log with no boot raises ``ValueError``.


def _last_boot(log: str) -> str:
    mod = load_module("local_astra.bench.b16_tap", "14")
    return attr(mod, "last_boot", "14")(log)  # type: ignore[no-any-return]


def _two_boots(caplog: pytest.LogCaptureFixture) -> tuple[str, str, str]:
    """Boot 1: the excerpt with ``max_num_seqs`` 4 and an old request's release line; boot 2: the
    excerpt as recorded, then ``req-log``'s release line. Returns (whole log, boot-1 marker line,
    boot-2 text)."""
    ref = _excerpt()
    boot1 = ref.replace("'max_num_seqs': 3,", "'max_num_seqs': 4,")
    assert boot1 != ref
    tap = _tap_release_line(caplog)
    old = tap.replace("req-log", "req-from-boot-1")
    boot2 = ref + tap + "\n"
    return boot1 + old + "\n" + boot2, old, boot2


def test_last_boot_returns_only_the_last_boot_of_an_appended_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R16: from two appended boots, the slice holds the second boot's config and release lines and
    nothing of the first: ``boot_config_lines`` of the slice equals the reference (max_num_seqs 3,
    not boot 1's 4), and only ``req-log``'s release line is parsed."""
    log, old, boot2 = _two_boots(caplog)
    got = _last_boot(log)
    assert log.endswith(got), "the slice is a suffix of the log"
    assert old not in got and "'max_num_seqs': 4," not in got, "boot 1 leaked into the slice"
    assert "'max_num_seqs': 3," in got and "GPU KV cache size" in got
    assert _boot_config_lines(got) == _boot_config_lines(_excerpt())
    assert [r["req_id"] for r in _parse_tap_release_lines(got)] == ["req-log"]
    assert boot2.split("non-default args:", 1)[1] in got


def test_last_boot_of_a_single_boot_keeps_all_of_it(caplog: pytest.LogCaptureFixture) -> None:
    """R16, the "too narrow" side: one boot is its own last boot -- every config fact and the
    release line survive the slice."""
    tap = _tap_release_line(caplog)
    log = _excerpt() + tap + "\n"
    got = _last_boot(log)
    assert _boot_config_lines(got) == _boot_config_lines(_excerpt())
    assert [r["req_id"] for r in _parse_tap_release_lines(got)] == ["req-log"]


def test_last_boot_of_a_log_with_no_boot_raises() -> None:
    """R16, never silent: a log with no boot (wrong file, truncated) is an error, not an empty slice
    that would compare as a boot with every fact missing."""
    with pytest.raises(ValueError):
        _last_boot("(APIServer pid=1) INFO 10-02 12:00:00 [launcher.py:1] nothing to see\n")
