"""The regime line is built from facts, never from config dressed as a measurement (criteria 19, 20).

CLAUDE.md: a number without its regime cannot be compared later — and a regime field copied out of
``astra.toml`` is worse than missing, because it looks measured. When a fact cannot be gathered the
line says the literal ``unknown`` and why.

Regime: pure; no box is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

from local_astra import cli
from local_astra.config import load
from tests.unit import _box_rig as rig
from tests.unit import _mic_rig as mic_rig

PROFILE = load(None).profiles["lean"]


def regime_line(facts) -> str:
    return rig.attr(cli, "regime_line")(PROFILE, facts)


def field(line: str, key: str) -> str:
    """The regime line is ``key=value`` tokens (CLAUDE.md: a number states its whole regime)."""
    match = re.search(rf"\b{key}=(\S+)", line)
    assert match, f"criterion 19: the regime line states {key}=<value>; it said {line!r}"
    return match.group(1)


def test_the_stt_device_and_model_id_come_from_the_measurements():
    """Criterion 19: the device from the container argv (c.5), the model id from /v1/models (c.4)."""
    line = regime_line(rig.facts(**rig.good_box(stt_argv=rig.STT_ARGV_CPU)))
    assert "cpu" in field(line, "stt_device").lower(), (
        f"criterion 19: the STT device comes from the container argv (--cpu); the line said {line!r}"
    )
    assert field(line, "served_model") == rig.PINNED_LOCK["model"], (
        f"criterion 19: …and the served model id from the engine; the line said {line!r}"
    )
    assert "kyutai-gpu" not in line, (
        "criterion 19/20: `kyutai-gpu` is the *profile's* value; the container is on CPU, so "
        f"printing it would be a config value dressed as a measurement. The line said {line!r}"
    )


def test_a_fact_that_could_not_be_gathered_prints_unknown_and_the_reason():
    """Criterion 19: ``unknown`` is a fact; a config value in its place is not."""
    line = regime_line(rig.facts())
    assert field(line, "stt_device").lower() == "unknown", (
        "criterion 19: with nothing gathered the STT device prints the literal `unknown`; the "
        f"line said {line!r}"
    )
    assert field(line, "served_model").lower() == "unknown", (
        f"criterion 19: …and so does the served model id; the line said {line!r}"
    )
    assert PROFILE.stt_backend not in line, (
        f"criterion 19: …never the profile's {PROFILE.stt_backend!r}; the line said {line!r}"
    )
    assert rig.PINNED_LOCK["model"] not in line, (
        f"criterion 19: …and never the lock's model id; the line said {line!r}"
    )
    lowered = line.lower()
    assert any(
        reason in lowered
        for reason in ("no container", "not gathered", "unreachable", "could not", "no answer")
    ), f"criterion 19: …and the reason it is unknown; the line said {line!r}"


def test_a_gpu_container_is_reported_as_gpu():
    """Criterion 19: the same argv fact, the other way round — this cannot pass by printing 'cpu'."""
    line = regime_line(rig.facts(**rig.good_box(stt_argv=rig.STT_ARGV_GPU)))
    assert "cpu" not in field(line, "stt_device").lower(), (
        f"the container has no --cpu, so the device is not CPU; the line said {line!r}"
    )


def test_astra_stt_cpu_is_never_consulted_inside_the_package():
    """Criterion 20, static: the container's argv is the fact; the env var is what somebody typed.

    ``services/stt.sh`` is the only place that reads it, and this task does not edit that file.
    """
    rig.attr(cli, "regime_line")  # the fence only means something once the line is built from facts
    offenders = []
    for path in sorted((rig.REPO / "local_astra").rglob("*.py")):
        if "ASTRA_STT_CPU" in path.read_text():
            offenders.append(str(path.relative_to(rig.REPO)))
    assert offenders == [], (
        f"criterion 20: ASTRA_STT_CPU appears nowhere under local_astra/; it appears in {offenders}"
    )
    assert "ASTRA_STT_CPU" in (rig.REPO / "services" / "stt.sh").read_text(), (
        "…and it is still the launch script's own knob (this test would be vacuous if the name "
        "had simply been dropped everywhere)"
    )


def test_the_regime_line_reads_no_service_endpoint_from_the_profile_where_a_fact_was_promised():
    """Criterion 20: no regime field is populated from ``config.Profile`` where c.19 promises a fact."""
    missing = rig.facts()
    line = regime_line(missing)
    for value in (PROFILE.stt_backend, PROFILE.tts_backend):
        assert value not in line, (
            f"criterion 20: {value!r} is a profile field standing in for a measured one; the line "
            f"said {line!r}"
        )


def test_the_session_prints_a_regime_line_built_that_way(capsys, monkeypatch):
    """Criterion 19: ``astra session`` is where this line is printed (``--dry-run`` prints it too)."""
    assert Path(rig.REPO / "local_astra" / "preflight.py").exists(), (
        "criterion 19: the facts the regime line states come from local_astra/preflight.py"
    )
    preflight = rig.module("preflight")
    nothing = rig.facts()
    monkeypatch.setattr(preflight, "gather", lambda *a, **k: nothing)
    # T-0045 ruling A16: the capture source is pinned rather than read off this box. Unpinned,
    # this row asks the runner's PipeWire graph which microphone a session would open — an answer
    # that is the webcam here and nothing in a container, i.e. a regime nobody states.
    mic_rig.pin_default_source(monkeypatch, mic_rig.BOX_MIC)
    mic_rig.env_source(monkeypatch, None)
    code = cli.main(["session", "--dry-run"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "stt=" in printed, f"the resolved graph is still printed; it printed\n{printed}"
    assert "unknown" in printed.lower(), (
        "criterion 19: with nothing gathered (no engine answering, no container inspected) the "
        f"regime line says `unknown`; it printed\n{printed}"
    )


# --- T-0045: which microphone heard the turn ----------------------------------------------------
#
# CLAUDE.md's regime rule applied to the one fact the 2026-09-15 session could not state: *which*
# capture source was open. A turn recorded with the webcam mic in a noisy room and one recorded
# off the null sink's monitor are not comparable numbers, and until this task there was no source
# at all — ``open_device(sink=…)`` with no ``source`` is an output-only stream (``stt_frames_fed:
# 0``). The seam and the node names are in ``tests/unit/_mic_rig.py``.


def run_session(argv: list[str]) -> int:
    """``cli.main`` with argparse's ``SystemExit`` turned into the exit code it carries."""
    try:
        return int(cli.main(argv))
    except SystemExit as exit_code:  # an unknown flag
        return int(exit_code.code or 0)


def regime_of(printed: str) -> str:
    """The ``regime:`` line out of what ``astra session`` printed."""
    lines = [line for line in printed.splitlines() if "regime:" in line]
    assert lines, f"criterion 19: `astra session` prints a regime line; it printed:\n{printed}"
    return lines[-1]


def test_the_regime_line_states_which_microphone_heard_the_turn(capsys, monkeypatch):
    """T-0045 criterion 2: ``mic=<node name>`` is on the regime line, not only beside ``sink=``.

    The regime line is what a number is quoted with; a latency measured through the null sink's
    monitor and one measured through the webcam mic in a room are different regimes.
    """
    preflight = rig.module("preflight")
    monkeypatch.setattr(preflight, "gather", lambda *a, **k: rig.facts())
    mic_rig.pin_default_source(monkeypatch, mic_rig.BOX_MIC)
    # Ruling A14: ASTRA_AUDIO_SOURCE outranks the graph default, so a row about what the regime
    # line says has to state whether the variable is set. Here it is not.
    mic_rig.env_source(monkeypatch, None)

    code = run_session(["session", "--dry-run", "--mic", mic_rig.RIG_MIC])
    printed = capsys.readouterr().out
    assert code == 0, f"--dry-run exits 0; it exited {code} after printing:\n{printed}"
    assert field(regime_of(printed), "mic") == mic_rig.RIG_MIC, (
        f"criterion 2: the regime line names the capture source the session opened "
        f"({mic_rig.RIG_MIC!r}); it said:\n{regime_of(printed)}"
    )


def test_the_regime_line_says_mic_none_when_no_microphone_is_open(capsys, monkeypatch):
    """T-0045 criteria 1 and 2: ``--no-mic`` reads ``mic=none`` — and names no microphone anyway.

    Adversarial half: the box's default source is stubbed to a node name that must not appear
    anywhere in the output. A regime line that prints the default source of a run that opened
    none is a measurement claim about a device that was never opened.
    """
    preflight = rig.module("preflight")
    monkeypatch.setattr(preflight, "gather", lambda *a, **k: rig.facts())
    mic_rig.pin_default_source(monkeypatch, mic_rig.BOX_MIC)
    # Ruling A14: ASTRA_AUDIO_SOURCE outranks the graph default, so a row about what the regime
    # line says has to state whether the variable is set. Here it is not.
    mic_rig.env_source(monkeypatch, None)

    code = run_session(["session", "--dry-run", "--no-mic"])
    printed = capsys.readouterr().out
    assert code == 0, f"--dry-run exits 0; it exited {code} after printing:\n{printed}"
    assert field(regime_of(printed), "mic") == "none", (
        f"criterion 2: a run with no microphone says so on the regime line; it said:\n"
        f"{regime_of(printed)}"
    )
    assert mic_rig.BOX_MIC not in printed, (
        f"criterion 2, adversarial: --no-mic opened no device, so the box's default source "
        f"{mic_rig.BOX_MIC!r} may not appear anywhere in the run's regime; it printed:\n{printed}"
    )
