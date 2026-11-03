"""Construct ``pw_top_active.txt`` from ``pw_top_idle.txt`` and pw-top's own row format.

``pw_top_idle.txt`` is a real ``pw-top -b -n 2`` of this box (read-only, 2026-09-14, pw 1.0.5):
every node is idle there, because the box has no running graph -- ``auto_null`` is its only
sink and nothing was playing.  A *follower* row therefore cannot be recorded here at all, and
ruling A14 is precisely about the follower marker, so the active fixture is **constructed**
from the recorded one by rewriting fields at pw-top 1.0.5's own column positions:

    %s %4.1u %6.1u %6.1u %s %s %s %s  %3.1u %16.16s %s%s
    S   ID  QUANT   RATE    WAIT    BUSY   W/Q   B/Q  ERR FORMAT           NAME

The trailing ``%s%s`` is ``marker + name``; the marker is ``""`` for a driver and ``" + "``
for a follower (the literal in /usr/bin/pw-top, 1.0.5 -- ``\\x00 + \\x00`` appears once in its
rodata, immediately used by the row printer).  That is the whole point of A14: the ``+`` is on
the NAME side of the row, not in the ID column.

Deterministic, no RNG.  Regenerate with::

    uv run python tests/fixtures/pipewire/make_pw_top_fixture.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
IDLE = HERE / "pw_top_idle.txt"
ACTIVE = HERE / "pw_top_active.txt"

WEBCAM = "alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo"
# The Stream/Output/Audio node of dump_rig_ok.json (id 137, node.name astra-conductor).
OURS_ID = 137
OURS_NAME = "astra-conductor"


def row(
    state: str,
    node_id: int,
    quant: int,
    rate: int,
    *,
    fmt: str,
    name: str,
    follower: bool,
    wait: str,
    busy: str,
    wq: str,
    bq: str,
    err: int = 0,
) -> str:
    """One row, at pw-top 1.0.5's column widths (see the module docstring)."""
    marker = " + " if follower else ""
    return (
        f"{state} {node_id:4d} {quant:6d} {rate:6d}"
        f" {wait:>6}{busy:>8}{wq:>6}{bq:>6}{err:>6}"
        f" {fmt:>16.16} {marker}{name}"
    )


def idle_row(state: str, node_id: int, name: str) -> str:
    return row(
        state,
        node_id,
        0,
        0,
        fmt="",
        name=name,
        follower=False,
        wait="---",
        busy="---",
        wq="---",
        bq="---",
    )


def main() -> None:
    header = IDLE.read_text().splitlines()[0]
    # First block: the recorded idle one, verbatim.  Only the last block is reported, so a
    # parser that keeps the first one reports six idle nodes and no follower.
    first = IDLE.read_text().splitlines()[: len(IDLE.read_text().splitlines()) // 2]
    lines = list(first)
    lines.append(header)
    lines.append(idle_row("S", 29, "Dummy-Driver"))
    lines.append(idle_row("S", 30, "Freewheel-Driver"))
    lines.append(idle_row("S", 45, "Midi-Bridge"))
    lines.append(
        row(
            "S",
            52,
            1024,
            48000,
            fmt="S16LE 2 48000",
            name=WEBCAM,
            follower=False,
            wait="1.2ms",
            busy="289.5us",
            wq="0.06",
            bq="0.01",
        )
    )
    lines.append(
        row(
            "R",
            OURS_ID,
            480,
            48000,
            fmt="F32LE 1 48000",
            name=OURS_NAME,
            follower=True,
            wait="412us",
            busy="98.3us",
            wq="0.04",
            bq="0.01",
        )
    )
    # A second client's stream, also a follower of node 52 and not ours: under A14 it is
    # neither "ours" nor "the driver", so the report must not mention its 2048 frames at all.
    lines.append(
        row(
            "R",
            145,
            2048,
            48000,
            fmt="S16LE 2 48000",
            name="pw-play",
            follower=True,
            wait="1.9ms",
            busy="151us",
            wq="0.09",
            bq="0.01",
        )
    )
    lines.append(idle_row("S", 123, "auto_null"))
    ACTIVE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
