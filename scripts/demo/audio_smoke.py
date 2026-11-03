#!/usr/bin/env python3
"""First contact with the audio path, in one command (T-0035 criterion 8).

The two-minute "is the audio path real" check. It opens the device through
``local_astra.conductor.audio_device.open_device()`` -- **never** through ``sounddevice``
directly, because ``audio_device.py`` is the single audio owner in the process (T-0006
criterion 12c) and a second client means a second clock -- plays a short chirp into ``--sink``,
captures from ``--source``, and prints one block: the requested *and* negotiated output/input
latencies, ``quantum_ms`` beside its ``quantum_source``, ``underflow_frames``, ``device_xruns``,
the captured peak, the verified link peers of both halves, and the graph's sink/source list.

Beside the peak it prints the discriminator the 2026-09-14 first-contact audit had to work out by
hand, because the peak is the one number that looks fine when everything is wrong:

* ``peak == 0`` means the target is absent -- you are recording nothing;
* ``peak`` of about 0.11 to 0.16 on a silent rig means you captured the webcam mic, not the
  monitor.

Both exit non-zero: a smoke script that says "OK" on a silent capture is worse than no script.

Usage on the null-sink rig (``scripts/audio_rig.sh up``)::

    uv run python scripts/demo/audio_smoke.py --sink astra-test-sink \
        --source astra-test-sink.monitor

``--dry-run`` performs no device I/O at all (this box has no ``libportaudio2`` -- prerequisite P1)
and still prints the whole report; ``--peak FLOAT`` supplies the captured peak a dry run would
otherwise have no way to produce, so the discriminator itself is exercisable without a rig
(ruling A3). Reading the graph during a dry run is allowed: ``pw-dump`` is read-only and touches
no device; when it cannot be read the report says so.

``scripts/demo/`` is D5-ungated: this script produces no number anyone will quote. Every number it
*prints* is only meaningful with its regime, which is why the header names the sink, the source,
the block and the requested latencies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_astra.conductor.audio_device import (
    DEFAULT_INPUT_LATENCY_MS,
    DEFAULT_LATENCY_MS,
    DEVICE_RATE,
    AudioDeviceError,
    PortAudioDevice,
    open_device,
    parse_link_peers,
    parse_nodes,
)
from local_astra.conductor.audio_device import _graph as read_graph
from local_astra.conductor.audio_device import _stream_nodes as stream_node_ids
from local_astra.contracts.speech import CAPTURE_BLOCK_MS

#: What ``pw-dump`` calls the two halves, in the order the report prints them.
HALVES = ("output", "input")
#: The discriminator of criterion 8, verbatim. Printed on every run, failing or not.
DISCRIMINATOR = (
    "  peak == 0            the target is absent: you are recording nothing.",
    "  peak 0.11 to 0.16    on a silent rig you captured the webcam mic, not the monitor.",
)
#: The Logitech C925e recording a silent room, measured by the 2026-09-14 audit.
WEBCAM_FLOOR = (0.11, 0.16)
#: How long the drain wait gives the device *beyond* the chirp it just handed over before it
#: calls the device wedged (T-0042 criterion 1). Generous on purpose: the ring holds
#: ``RING_MS`` = 200 ms and PortAudio's own ``write()`` ceiling is 2 s, so a device that is
#: playing at all empties the queue long before this.
DRAIN_GRACE_S = 2.0
#: Interval between two ``queued_ms()`` reads while waiting. Short enough that the wait ends
#: within a fraction of a block, long enough that the poll is not a spin.
DRAIN_POLL_S = 0.002

EXIT_OK = 0
EXIT_DEVICE = 1
EXIT_SILENT = 2
EXIT_WEBCAM = 3


def chirp(seconds: float, rate: int = DEVICE_RATE) -> npt.NDArray[np.float32]:
    """A linear 200 Hz -> 4 kHz sweep: audible, broadband, and obvious in a recording."""
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    sweep = 200.0 * t + 0.5 * (4000.0 - 200.0) / max(seconds, 1e-9) * t * t
    return (0.25 * np.sin(2.0 * np.pi * sweep)).astype(np.float32)


def graph_nodes() -> tuple[dict[str, list[str]], str | None]:
    """The sink/source names in the graph, or empty lists and why they could not be read.

    Through ``_graph()``, so a box without ``pw-dump`` on its ``PATH`` prints one line and
    carries on: the report is the point, and a traceback is not a report (ruling A8).
    """
    try:
        return parse_nodes(read_graph("listing the graph's sinks and sources")), None
    except AudioDeviceError as exc:
        return {"sinks": [], "sources": []}, str(exc)


def link_peers() -> tuple[dict[str, frozenset[str]], str | None]:
    """The nodes this process's stream halves are linked to, as the graph sees them."""
    try:
        return parse_link_peers(read_graph("reading this process's link peers"), os.getpid()), None
    except AudioDeviceError as exc:
        return {}, str(exc)


def our_node_ids() -> set[int]:
    """The graph ids of this process's stream halves, or an empty set if the graph is mute."""
    try:
        return set(stream_node_ids(read_graph("finding our stream nodes"), os.getpid()).values())
    except AudioDeviceError:
        return set()


def pw_top_quantum(ours: set[int]) -> str:
    """The **live** graph quantum, from ``pw-top -b -n 2``'s QUANT column (ruling A13).

    ``quantum_ms`` beside it is only what this process *asked* for: ``node.latency`` in
    ``pw-dump`` is ``PIPEWIRE_LATENCY`` echoed back, and PortAudio's frame count is the block it
    handed us. ``pw-top`` is the one place the number the graph actually runs at appears, so it
    is printed here and recorded by group B. Two iterations because the first is a snapshot of
    an unstarted counter; the second block is the one that is read.

    Never raises: a box without ``pw-top`` gets a sentence, not a traceback (ruling A8).
    """
    try:
        completed = subprocess.run(
            ["pw-top", "-b", "-n", "2"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"not readable ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        return (
            f"not readable (pw-top exited {completed.returncode}: {completed.stderr.strip()[:120]})"
        )
    rows = pw_top_rows(completed.stdout)
    if not rows:
        return f"not readable (pw-top printed no rows: {completed.stdout.strip()[:120]!r})"
    shown = [
        f"{'ours' if row['id'] in ours else 'driver'} {row['quant']} frames ({quantum_ms(row)})"
        for row in rows
        if row["id"] in ours or (row["driver"] and row["quant"] != "0")
    ]
    if not shown:
        return "not readable (pw-top shows every node idle: QUANT 0, no driver is running)"
    return ", ".join(shown)


def quantum_ms(row: dict[str, Any]) -> str:
    """A pw-top row's QUANT in ms, at its own RATE -- ``1024 frames`` means nothing alone."""
    if not row["quant"].isdigit() or not row["rate"].isdigit() or int(row["rate"]) <= 0:
        return "rate unknown"
    return f"{1000.0 * int(row['quant']) / int(row['rate']):.1f} ms"


def pw_top_rows(text: str) -> list[dict[str, Any]]:
    """The rows of ``pw-top``'s **last** batch iteration: ``id``, ``quant``, ``rate``, ``name``.

    Columns are ``S ID QUANT RATE WAIT BUSY W/Q B/Q ERR FORMAT NAME``; ``FORMAT`` is empty for an
    idle node, so the fields are counted from the left and the name is taken from the right.

    **The follower marker sits on the NAME, not on the ID** (ruling A14): pw-top 1.0.5 prints a
    row as ``%s %4.1u %6.1u %6.1u %s %s %s %s  %3.1u %16.16s %s%s``, whose last two conversions
    are the marker and the node name. Reading it off the id column labels every row a driver,
    and ``pw_top_quantum`` would then report a competing client's follower stream as the quantum
    the graph runs at -- which is exactly the confusion this line exists to remove.
    """
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        if fields[1] == "ID":
            rows = []  # a header line starts a new iteration; only the last one is reported
            continue
        if not fields[1].isdigit():
            continue
        # The marker is printed immediately before the name, with or without a space between
        # them depending on how pw-top pads it.
        follower = fields[-1].startswith("+") or fields[-2] == "+"
        rows.append(
            {
                "id": int(fields[1]),
                "driver": not follower,
                "quant": fields[2],
                "rate": fields[3],
                "name": fields[-1].lstrip("+"),
            }
        )
    return rows


class Measured:
    """What one run produced. ``None`` means *not measured*, which is not the same as zero.

    A plain class rather than a ``@dataclass``: the fast suite loads this script **by path**
    without putting it in ``sys.modules``, and ``dataclasses`` resolves the string annotations
    ``from __future__ import annotations`` produces through ``sys.modules[cls.__module__]``,
    which is absent there. Criterion 9 asks that the script import cleanly; it must therefore
    not depend on how it was imported.
    """

    def __init__(
        self,
        *,
        peak: float | None = None,
        peers: dict[str, frozenset[str]] | None = None,
        peers_error: str | None = None,
        underflow_frames: int | None = None,
        device_xruns: int | None = None,
        pw_top: str = "not sampled",
    ) -> None:
        self.peak = peak
        self.peers = peers if peers is not None else {}
        self.peers_error = peers_error
        self.underflow_frames = underflow_frames
        self.device_xruns = device_xruns
        self.pw_top = pw_top


def wait_for_drain(device: PortAudioDevice, seconds: float) -> None:
    """Wait for everything handed over to leave our hands -- by *waiting*, never by writing.

    The bug this exists to remove (T-0042): the wait used to be
    ``while device.queued_ms() > 0: device.write(zeros)``, and on a device that plays in real
    time every one of those writes refilled the queue the loop was polling, so the loop never
    ended and the smoke never returned. The queue empties because the device thread takes from
    it; the only correct thing to do here is nothing, repeatedly.

    The ceiling is the chirp we just handed over plus ``DRAIN_GRACE_S``: a device consuming at
    all needs about the chirp's own length, so anything past that is a device that stopped. It
    is reported with the millisecond count still queued, because "it did not finish" and "it
    never started" look the same from the exit code alone.
    """
    deadline = time.monotonic() + seconds + DRAIN_GRACE_S
    while True:
        left = device.queued_ms()
        if left <= 0.0:
            return
        if time.monotonic() >= deadline:
            raise AudioDeviceError(
                f"queue did not drain ({left:.1f} ms left) within the chirp plus "
                f"{DRAIN_GRACE_S:.0f} s -- the device stopped taking audio"
            )
        time.sleep(DRAIN_POLL_S)


def play_and_capture(device: PortAudioDevice, seconds: float) -> Measured:
    """Play the chirp, capture alongside it, and read the graph while the stream is up.

    The peak is ``None`` -- *unjudged* -- unless a capture callback was actually installed
    (ruling A8): a playback-only run has captured nothing, and reporting 0.0 for it would fire
    the "you are recording nothing" discriminator at a run that was never recording.

    Ordering matters and is stated because the numbers are quoted: the link peers are read while
    the stream is open, the two counters are snapshotted **at the end of the chirp**, and only
    then is ``pw-top`` sampled -- its two one-second iterations run against a stream that is
    open but no longer being fed, so the underflows they cause are outside the measurement.
    """
    peak: float | None = None

    def captured(samples: npt.NDArray[np.float32], adc_ms: float, xrun: bool) -> None:
        nonlocal peak
        if samples.size:
            peak = max(peak or 0.0, float(np.max(np.abs(samples))))

    if device.source is not None or device.input_device is not None:
        peak = 0.0  # a capture half exists, so 0.0 is now a measurement, not an absence
        device.set_capture_callback(captured)
        device.start_capture()
    else:
        device.start()
    block = round(CAPTURE_BLOCK_MS * DEVICE_RATE / 1000)
    signal = chirp(seconds)
    for start in range(0, signal.size - block + 1, block):
        device.write(signal[start : start + block])
    peers, peers_error = link_peers()
    wait_for_drain(device, seconds)
    measured = Measured(
        peak=peak,
        peers=peers,
        peers_error=peers_error,
        underflow_frames=device.underflow_frames,
        device_xruns=device.device_xruns,
    )
    measured.pw_top = pw_top_quantum(our_node_ids())
    device.stop_capture()
    device.stop()
    return measured


def report(
    *,
    args: argparse.Namespace,
    device: PortAudioDevice | None,
    measured: Measured,
    nodes: dict[str, list[str]],
    nodes_error: str | None,
) -> None:
    """The one printed block. Every field is labelled; a number without its regime is noise."""
    requested = {"output": DEFAULT_LATENCY_MS, "input": DEFAULT_INPUT_LATENCY_MS}
    negotiated = {
        "output": device.latency_ms if device is not None else None,
        "input": device.input_latency_ms if device is not None else None,
    }
    wanted = {"output": args.sink, "input": args.source}
    print("audio smoke -- T-0035 criterion 8 (scripts/demo, D5-ungated)")
    print(
        f"  regime               sink={args.sink!r} source={args.source!r} "
        f"rate={DEVICE_RATE} Hz mono f32 block={CAPTURE_BLOCK_MS} ms "
        f"chirp={args.seconds} s dry_run={args.dry_run}"
    )
    for half in HALVES:
        value = negotiated[half]
        shown = "not opened (dry run)" if value is None else f"{value:.3f} ms"
        label = f"{half} latency"
        print(f"  {label:<20} requested {requested[half]:.3f} ms   negotiated {shown}")
    if device is None:
        print("  quantum_ms           not read (dry run)   quantum_source unknown")
    else:
        print(
            f"  quantum_ms           {device.quantum_ms:.3f}   "
            f"quantum_source {device.quantum_source}   "
            f"(a request or a block size, never the negotiated quantum -- ruling A13)"
        )
    # The graph block first, with its own reason if it could not be read, and the pw-top line
    # after it: both mention "graph", and a reader (or a grep) meeting the pw-top line first
    # would take *its* reason for the graph's.
    print(f"  graph sinks          {nodes['sinks']}")
    print(f"  graph sources        {nodes['sources']}")
    if nodes_error is not None:
        print(f"  graph: not readable ({nodes_error})")
    if measured.peers_error is not None:
        print(f"  graph: not readable ({measured.peers_error})")
    print(f"  graph quantum (pw-top): {measured.pw_top}")
    counters = {
        "underflow_frames": measured.underflow_frames,
        "device_xruns": measured.device_xruns,
    }
    for label, count in counters.items():
        shown = "not measured (dry run)" if count is None else str(count)
        print(f"  {label:<20} {shown}")
    peers = measured.peers
    for half in HALVES:
        actual = sorted(peers[half]) if half in peers else "no such stream half"
        label = f"{half} link peer"
        print(f"  {label:<20} requested {wanted[half]!r}   actual {actual}")
    print(
        "  placement            the output half is linked by port id through pw-link; group B "
        "confirms the pw-link CLI shape on the rig (T-0035 ruling A9)"
    )
    shown_peak = "not measured" if measured.peak is None else f"{measured.peak:.4f}"
    print(f"  captured peak        {shown_peak}")
    print("  what the peak means (the 2026-09-14 audit's discriminator):")
    for line in DISCRIMINATOR:
        print(line)


def verdict(peak: float | None) -> int:
    """Exit code for a captured peak: the two peaks that look like a rig in the dark fail."""
    if peak is None:
        print("VERDICT  capture half unjudged: no capture callback was installed, so no peak.")
        return EXIT_OK
    if peak == 0.0:
        print("VERDICT  FAIL: peak == 0 -- the target is absent: you are recording nothing.")
        return EXIT_SILENT
    if WEBCAM_FLOOR[0] <= peak <= WEBCAM_FLOOR[1]:
        print(
            f"VERDICT  FAIL: peak {peak:.4f} is the 0.11 to 0.16 room floor -- you captured the "
            f"webcam mic, not the monitor."
        )
        return EXIT_WEBCAM
    print(f"VERDICT  ok: peak {peak:.4f} is above the webcam floor and is not silence.")
    return EXIT_OK


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First contact with the audio path (T-0035).")
    parser.add_argument("--sink", default=None, help="PipeWire sink node to play into")
    parser.add_argument("--source", default=None, help="PipeWire source node to capture from")
    parser.add_argument("--seconds", type=float, default=2.0, help="chirp length in seconds")
    parser.add_argument(
        "--dry-run", action="store_true", help="no device I/O at all; still print the report"
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=None,
        help="the captured peak a dry run reports instead of capturing one (ruling A3)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    nodes, nodes_error = graph_nodes()
    device: PortAudioDevice | None = None
    measured = Measured(peak=args.peak)
    failure: str | None = None
    if args.dry_run:
        # A dry run touches nothing, and that includes pw-top: with no stream of ours in the
        # graph its QUANT would be somebody else's number, and two seconds of sampling would be
        # spent to print it. The line still appears, saying why it is empty.
        measured.pw_top = "not readable (dry run)"
    else:
        try:
            device = open_device(
                sink=args.sink,
                source=args.source,
                block_ms=CAPTURE_BLOCK_MS,
                latency_ms=DEFAULT_LATENCY_MS,
            )
            measured = play_and_capture(device, args.seconds)
        except (AudioDeviceError, OSError, RuntimeError, TimeoutError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
    report(
        args=args,
        device=device,
        measured=measured,
        nodes=nodes,
        nodes_error=nodes_error,
    )
    if failure is not None:
        print(f"VERDICT  FAIL: the device refused or failed -- {failure}")
        return EXIT_DEVICE
    return verdict(measured.peak)


if __name__ == "__main__":  # pragma: no cover - the command line
    raise SystemExit(main())
