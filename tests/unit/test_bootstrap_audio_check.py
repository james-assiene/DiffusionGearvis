"""``scripts/bootstrap.sh`` fails where ``astra session`` fails (criterion 21).

P0-3 in one sentence: bootstrap printed "ready" on a box where ``astra session`` exits 1, because
it checked for ``pactl`` (absent here) and never tried ``import sounddevice``. The audio verdict
must be the call the session makes.

Static assertions over the script plus ``bash -n``; nothing here runs bootstrap (it would run the
whole suite inside the suite).
"""

from __future__ import annotations

import re
import subprocess

from tests.unit import _box_rig as rig

SCRIPT = rig.REPO / "scripts" / "bootstrap.sh"


def source() -> str:
    """The script as committed — *after* criterion 21's edit.

    Every test below asserts this first: they are the negatives of criterion 21 (what the edit
    must not break), and a negative asserted against the unedited script would be green before
    the work and tell nobody anything.
    """
    text = SCRIPT.read_text()
    assert "import sounddevice" in text, (
        'criterion 21: the audio check runs `uv run python -c "import sounddevice; '
        'sounddevice.query_devices()"` — the exact call `astra session` fails on (P0-3)'
    )
    return text


def lines_matching(needle: str) -> list[str]:
    return [
        line
        for line in source().splitlines()
        if needle in line and not line.lstrip().startswith("#")
    ]


def test_the_audio_check_is_the_call_the_session_makes():
    text = source()
    assert "import sounddevice" in text and "query_devices" in text, (
        "criterion 21: the audio check runs "
        '`uv run python -c "import sounddevice; sounddevice.query_devices()"` — the exact call '
        "`astra session` fails on"
    )
    probe_lines = lines_matching("sounddevice")
    assert any("uv run" in line for line in probe_lines), (
        f"criterion 21: …through `uv run python -c`; the lines were {probe_lines}"
    )


def test_missing_portaudio_is_a_cross_that_makes_bootstrap_exit_one():
    text = source()
    assert "libportaudio2" in text, (
        "criterion 21: the fix line is `apt install libportaudio2` (printed for the human; "
        "bootstrap never runs sudo)"
    )
    index = text.index("sounddevice")
    block = text[index : index + 600]
    assert "fail" in block, (
        "criterion 21: a missing PortAudio is a ✗ and makes bootstrap exit 1 — `fail` is the "
        f"function that sets FAILED. The block after the probe was:\n{block[:400]}"
    )
    assert "libportaudio2" in block, "…with the fix on the same line as the failure"
    assert re.search(r'if \[ "\$FAILED" = 1 \]', text), (
        "criterion 21: …and FAILED still decides the exit code"
    )


def test_pactl_is_at_most_a_warning_and_never_the_audio_verdict():
    for line in lines_matching("pactl"):
        assert "fail" not in line, (
            f"criterion 21: `pactl` is absent on this box and says nothing about PortAudio; it "
            f"must never decide the audio verdict. This line does:\n{line}"
        )
    assert lines_matching("pactl"), (
        "…but the line is still there as a warning (this test would be vacuous if pactl were "
        "simply deleted; the criterion says 'at most a warning line')"
    )


def test_bootstrap_still_spends_no_gpu_time_and_runs_pytest_serially():
    text = source()
    for line in lines_matching("pytest") + lines_matching("make check"):
        assert not re.search(r"(?<![\w-])-n\s*\d", line), (
            f"T-0031: bootstrap shells out to pytest without -n; this line does not:\n{line}"
        )
    assert "hooks run and block an unapproved GPU command" in text, (
        "criterion 21, negative: the guard self-test line is unchanged"
    )
    for line in lines_matching("astra bench"):
        assert "guard_bash" in line or "echo" in line, (
            f"criterion 21, negative: bootstrap spends no GPU time; this line runs one:\n{line}"
        )
    for banned in ("--gpus", "nvidia-smi -l", "astra up"):
        assert banned not in text, f"criterion 21, negative: bootstrap starts nothing ({banned})"


def test_bootstrap_still_parses():
    source()
    done = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, f"bash -n scripts/bootstrap.sh failed:\n{done.stderr}"
