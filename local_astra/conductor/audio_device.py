"""The device seam of the playback path (T-0006 criterion 1) and its only real backend.

This is the **single module in ``local_astra/`` allowed to name a sound library** (criterion
12c, enforced by ``tests/unit/test_audio_ownership.py``): two audio clients in one process means
two clocks, and ``played_ms`` stops meaning anything. ``PipeWireAudioIO`` talks to whatever
satisfies ``AudioDevice``; the fake device in the unit tests and ``PortAudioDevice`` here are
interchangeable.

**What the protocol demands of a backend, and why.** ``rendered_ms()`` is the position the device
reports as *rendered* -- frames that have left for the DAC -- measured **inside the stream this
module wrote**, never wall time and never the write cursor (criterion 2; the docs/02 section 4
invariant "the history never contains assistant text that was not played" is only true if this
number is honest). Two consequences a backend must respect:

* frames the device emitted because it had nothing from us (an underflow) do **not** advance it,
  so a starved stream under-reports rather than over-reports;
* the configured output latency is subtracted, because audio handed to PortAudio has not been
  heard yet.

**Backend choice (stated, as the task asks).** ``sounddevice`` 0.5.6 -- already pinned in
``uv.lock`` under the ``audio`` extra -- driving PortAudio in *callback* mode, because the
callback is where PortAudio reports ``outputBufferDacTime`` and that is the only honest source
for ``rendered_ms()``. The alternative (libpipewire through ``ctypes``) needs hand-built
``spa_pod`` parameter blobs to connect a stream: several hundred lines of binary layout code for
the same number. On this box PortAudio itself is **not installed** (``libportaudio2``,
prerequisite P1 in docs/tasks/T-0006.md -- a sysadmin step), so ``open_device()`` raises with
that name in the message; nothing here is imported at collection time and the unit and contract
suites never touch it.

**Proving the stream landed where we asked (T-0035 criteria 4-6).** A ``PIPEWIRE_NODE`` naming a
node the session manager will not honour links the stream to ``auto_null`` and plays happily into
nothing; a monitor capture of an absent node yields seconds of exactly 0.0 with no error. Both are
indistinguishable from a working rig in the dark, so when a node name is requested this module
reads the graph: ``open_device()`` refuses a name that is not in ``pw-dump`` (criterion 4) and
``start()`` polls a bounded settle window until this process's ``Stream/*/Audio`` nodes are linked
to what was asked for, raising with *requested* and *actual* otherwise (criterion 5). There is
deliberately no opt-out. When **no** node is named the plain-default path is unchanged and no
``pw-dump`` runs at all.

**The three environment variables this module reads** (it is their only reader in the tree;
T-0035 criterion 7):

* ``ASTRA_AUDIO_SOURCE`` -> ``source``: a **PipeWire node name** (e.g.
  ``astra-test-sink.monitor``), the capture side of the graph;
* ``ASTRA_AUDIO_INPUT_DEVICE`` -> ``input_device``: the **PortAudio** device-name escape hatch
  for a real ALSA capture device (e.g. ``hw:CARD=C925e``);
* ``ASTRA_AUDIO_DEVICE`` -> ``device``: the PortAudio output device name.

No asyncio: the callback and everything it touches run off the event loop (criterion 13,
``.claude/rules/conductor.md``). No sockets, no network (criterion 12d). The graph is read on the
calling thread from ``open_device()`` / ``start()`` only -- **never** from a device callback.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from local_astra.contracts.speech import CAPTURE_BLOCK_MS, CAPTURE_FORMAT

#: What a backend hands the capture path: ``(samples_f32, adc_ms, xrun)`` (T-0021 criterion 1).
CaptureCallback = Callable[[npt.NDArray[np.float32], float, bool], None]

DEVICE_RATE = CAPTURE_FORMAT.sample_rate
#: Output latency asked of the device. Criterion 1 caps it at 30 ms; the negotiated value is
#: reported in ``latency_ms`` after ``start()``.
DEFAULT_LATENCY_MS = 30.0
#: How much audio ``write()`` may hold beyond what the callback has taken. Small on purpose:
#: ``pause()`` can only silence audio it has not handed over yet (criterion 3).
RING_MS = 200.0
#: Input latency asked of the device, in ms. Reported after ``start()`` as negotiated; never
#: added to a capture timestamp (ruling A4: ``captured_ms()`` already is the ADC instant).
DEFAULT_INPUT_LATENCY_MS = 30.0
#: Longest ``write()`` waits for the callback to make room before giving up.
WRITE_TIMEOUT_S = 2.0
#: How long ``start()`` waits for the graph to link our stream halves (T-0035 criterion 5).
#: Links are made asynchronously, so this is a *poll*, not a sleep-and-hope: a fixed sleep is
#: flaky on a loaded box exactly when somebody is debugging.
SETTLE_WINDOW_S = 0.5
#: Interval between two graph reads inside the settle window.
SETTLE_POLL_S = 0.02
#: ``pw-dump`` is a local, read-only query; it has never taken more than a few ms here.
PW_DUMP_TIMEOUT_S = 5.0

#: ``node.latency`` as PipeWire writes it: ``"<quantum>/<rate>"``, e.g. ``"1024/48000"``.
_NODE_LATENCY = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
#: A monitor is not a node: no PipeWire graph contains ``<sink>.monitor`` as a node name, so a
#: monitor source is looked up -- and connected to -- under the sink's own name (ruling A5).
_MONITOR_SUFFIX = ".monitor"


class AudioDeviceError(RuntimeError):
    """Base of the errors this module raises, so callers never catch a leaked builtin."""


class PipeWireDumpError(AudioDeviceError):
    """``pw-dump`` could not be run, or what it produced is not a graph."""


class UnknownNodeError(AudioDeviceError):
    """A requested PipeWire node does not exist (T-0035 criterion 4)."""


class PlacementError(AudioDeviceError):
    """The stream did not land on the requested node (T-0035 criteria 5 and 6)."""


def node_name(name: str) -> str:
    """The node a name refers to: ``astra-test-sink.monitor`` is the node ``astra-test-sink``."""
    return name[: -len(_MONITOR_SUFFIX)] if name.endswith(_MONITOR_SUFFIX) else name


# -- the pure pw-dump parsers (T-0035 criteria 3-5) ---------------------------------------
# Pure on purpose: every graph assertion in group A is then a table-driven test over recorded
# JSON (tests/fixtures/pipewire/), which is the only way criteria 4 and 5 are testable before
# PortAudio exists on this box. ``json.loads`` hands back ``Any``; it is narrowed here, at the
# boundary, rather than spreading through the module.


def _objects(dump: object) -> list[dict[str, Any]]:
    """Narrow a decoded ``pw-dump`` to the array of objects it is supposed to be."""
    if not isinstance(dump, list):
        raise PipeWireDumpError(
            f"pw-dump output is not the JSON array of objects pw-dump emits: "
            f"got {type(dump).__name__}"
        )
    objects: list[dict[str, Any]] = []
    for entry in dump:
        if not isinstance(entry, dict):
            raise PipeWireDumpError(
                f"pw-dump output contains a {type(entry).__name__} where an object was expected"
            )
        objects.append(entry)
    return objects


def _props(obj: dict[str, Any]) -> dict[str, Any]:
    """``info.props`` of one object, or empty: ``Core``/``Metadata`` entries carry none."""
    info = obj.get("info")
    if not isinstance(info, dict):
        return {}
    props = info.get("props")
    return props if isinstance(props, dict) else {}


def _is(obj: dict[str, Any], kind: str) -> bool:
    return str(obj.get("type", "")).endswith(f":{kind}")


def parse_nodes(dump: object) -> dict[str, list[str]]:
    """``{"sinks": [...], "sources": [...]}``: the audio node names in a ``pw-dump``.

    A ``Video/Source`` is not an audio source and the driver nodes (``Dummy-Driver``,
    ``Freewheel-Driver``) have a name and no ``media.class``, so neither appears. Monitors never
    appear either -- they are ports of their sink, not nodes.
    """
    sinks: list[str] = []
    sources: list[str] = []
    for obj in _objects(dump):
        if not _is(obj, "Node"):
            continue
        props = _props(obj)
        name = props.get("node.name")
        if not isinstance(name, str):
            continue
        media = props.get("media.class")
        if media == "Audio/Sink":
            sinks.append(name)
        elif media == "Audio/Source":
            sources.append(name)
    return {"sinks": sorted(sinks), "sources": sorted(sources)}


def _client_ids(objects: list[dict[str, Any]], pid: int) -> set[int]:
    """The ``Client`` ids belonging to ``pid``.

    The pid lives on the ``Client`` object, never on a node (ruling A4): a parser that took the
    first ``Stream/*/Audio`` node it found would report some other process's stream, and the
    check of criterion 5 would verify a browser's playback on a busy box.
    """
    ids: set[int] = set()
    for obj in objects:
        if not _is(obj, "Client"):
            continue
        props = _props(obj)
        owner = props.get("application.process.id", props.get("pipewire.sec.pid"))
        ident = obj.get("id")
        if owner == pid and isinstance(ident, int):
            ids.add(ident)
    return ids


def _stream_nodes(objects: list[dict[str, Any]], pid: int) -> dict[str, int]:
    """``{"output": node_id, "input": node_id}`` for the halves ``pid`` actually has."""
    clients = _client_ids(objects, pid)
    halves: dict[str, int] = {}
    for obj in objects:
        if not _is(obj, "Node"):
            continue
        props = _props(obj)
        ident = obj.get("id")
        if props.get("client.id") not in clients or not isinstance(ident, int):
            continue
        media = props.get("media.class")
        if media == "Stream/Output/Audio":
            halves["output"] = ident
        elif media == "Stream/Input/Audio":
            halves["input"] = ident
    return halves


def parse_link_peers(dump: object, pid: int) -> dict[str, frozenset[str]]:
    """The nodes each of ``pid``'s stream halves is linked to, by half (rulings A9, A12).

    Always a ``frozenset`` of node names per half -- **every** peer, never "the last link that
    won": a stream the session manager put on ``auto_null`` and ``pw-link`` then also linked to
    the requested sink is a graph nobody asked for, and a single name cannot express it.

    An empty set means "this half exists and is linked to nothing yet" (poll again inside the
    settle window); a half the process does not have is **absent** from the mapping, so the two
    stay distinguishable at every call site.
    """
    objects = _objects(dump)
    halves = _stream_nodes(objects, pid)
    names = {obj.get("id"): _props(obj).get("node.name") for obj in objects if _is(obj, "Node")}
    found: dict[str, set[str]] = {half: set() for half in halves}
    for obj in objects:
        if not _is(obj, "Link"):
            continue
        props = _props(obj)
        out_node, in_node = props.get("link.output.node"), props.get("link.input.node")
        if "output" in halves and out_node == halves["output"]:
            peer = names.get(in_node)
            if isinstance(peer, str):
                found["output"].add(peer)
        if "input" in halves and in_node == halves["input"]:
            peer = names.get(out_node)
            if isinstance(peer, str):
                found["input"].add(peer)
    return {half: frozenset(linked) for half, linked in found.items()}


def _ports(objects: list[dict[str, Any]], node_id: int, *, direction: str) -> list[int]:
    """The ids of ``node_id``'s ports in one direction, monitor ports excluded, in graph order.

    Ids, not names: ``pw-link`` takes either, and only the id identifies one node's port
    unambiguously (ruling A9 -- the ALSA plugin gives every PortAudio client the same name).
    """
    found: list[int] = []
    for obj in objects:
        if not _is(obj, "Port"):
            continue
        props = _props(obj)
        ident = obj.get("id")
        if props.get("node.id") != node_id or props.get("port.direction") != direction:
            continue
        if props.get("port.monitor") or not isinstance(ident, int):
            continue
        found.append(ident)
    return found


def _sink_inputs(objects: list[dict[str, Any]], sink: str) -> list[int]:
    """The playback port ids of the sink node called ``sink``."""
    for obj in objects:
        if not _is(obj, "Node"):
            continue
        props = _props(obj)
        ident = obj.get("id")
        if props.get("node.name") == sink and isinstance(ident, int):
            return _ports(objects, ident, direction="in")
    return []


def parse_quantum(dump: object, pid: int) -> float | None:
    """The quantum ``pid``'s stream node **asked for**, in ms, or ``None`` if the graph is mute.

    **This is a request, not a negotiation (ruling A13).** ``node.latency`` in ``pw-dump`` is the
    ``PIPEWIRE_LATENCY`` this process set, echoed back by the node -- verbatim, even when it is
    outside what the graph will clamp to -- and ``pw-dump`` exposes no live quantum at all. What
    reading it back proves is only that the plugin carried our request into the graph, which is
    why its ``quantum_source`` is ``"request"``. The live graph quantum is a measurement:
    ``pw-top``'s QUANT column, which ``scripts/demo/audio_smoke.py`` prints and group B records.

    ``None`` (the graph does not carry the property) is not 0.0 and not the block size: a
    fabricated number would be worse than an absent one (criterion 3).
    """
    objects = _objects(dump)
    halves = _stream_nodes(objects, pid)
    for obj in objects:
        if not _is(obj, "Node") or obj.get("id") not in halves.values():
            continue
        latency = _props(obj).get("node.latency")
        if not isinstance(latency, str):
            continue
        match = _NODE_LATENCY.match(latency)
        if match is None:
            continue
        quantum, rate = int(match.group(1)), int(match.group(2))
        if rate > 0:
            return 1000.0 * quantum / rate
    return None


def _pw_dump() -> list[dict[str, Any]]:
    """Read the PipeWire graph. **The single seam in this module that shells out.**

    Everything else about the graph is a pure parser over what this returns, so the tests inject
    a recorded dump here instead of needing a graph. ``scripts/audio_loopback.py::pw_nodes()``
    shells the same command; ``local_astra/`` must not import from ``scripts/``, so this one
    ``subprocess.run`` line is deliberately duplicated (docs/tasks/T-0035.md, Owned files).

    Never called from a device callback: the callback allocates nothing and does no I/O.
    """
    completed = subprocess.run(
        ["pw-dump"],
        capture_output=True,
        text=True,
        check=False,
        timeout=PW_DUMP_TIMEOUT_S,
    )
    if completed.returncode != 0:
        raise PipeWireDumpError(
            f"pw-dump exited {completed.returncode}: {completed.stderr.strip()[:200]}"
        )
    return _objects(json.loads(completed.stdout))


def _graph(reason: str) -> list[dict[str, Any]]:
    """``_pw_dump()`` with every failure turned into this module's own named error.

    An explicit node request that cannot be verified is not honoured blindly (criterion 5): a
    missing ``pw-dump`` is a refusal, never a shrug.
    """
    try:
        return _pw_dump()
    except AudioDeviceError:
        raise
    except Exception as exc:
        raise PipeWireDumpError(
            f"pw-dump could not be read while {reason} ({type(exc).__name__}: {exc}); an "
            f"explicitly requested PipeWire node that cannot be verified is not honoured"
        ) from exc


@runtime_checkable
class AudioDevice(Protocol):
    """The Conductor-owned output device (T-0006 criterion 1)."""

    #: Output latency the device is configured with, in ms (negotiated, not requested).
    latency_ms: float
    #: Block size the writer hands over, in ms -- ``CAPTURE_BLOCK_MS``.
    block_ms: int
    #: Graph quantum in ms, ``0.0`` when the backend cannot say. PipeWire's default is
    #: 1024/48000 ~ 21 ms, and a module that assumes 10 ms while the graph runs at 21 ms
    #: mis-times every pause. A backend that can only report what it *asked for* says so beside
    #: the number: ``PortAudioDevice.quantum_source`` (T-0035 ruling A13).
    quantum_ms: float
    #: Input latency the device is configured with, in ms (negotiated). Reported, never added to
    #: a capture timestamp: ``captured_ms()`` already *is* the ADC instant (T-0021 ruling A4).
    input_latency_ms: float
    #: Frames of this stream that did not carry our audio -- the backend's own xrun counter:
    #: frames the device invented because our ring was short, **plus** the whole of any callback
    #: the backend flagged as an underflow (the two causes counted as their maximum, never
    #: summed; T-0035 criterion 1). Non-decreasing while the stream is open: ``audio_io``
    #: reads it as a delta and a reset would make that delta negative.
    #: A protocol member since T-0006 ruling A22 (owed to T-0021): a device that cannot say how
    #: many frames it invented cannot be an ``AudioDevice``, because the alternative test
    #: ``rendered >= written`` is unreachable on any device that subtracts its output latency
    #: (ruling A11) and silently under-counts when it is forced.
    underflow_frames: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def write(self, block: npt.NDArray[np.float32]) -> None:
        """One ``block_ms`` block of mono f32 at 48 kHz. May block while the device is full."""
        ...

    def rendered_ms(self) -> float:
        """Frames of *our* stream the device reports as rendered. Monotonic, non-decreasing."""
        ...

    def start_capture(self) -> None:
        """Begin capturing. Capture and playback are one device object and one clock base."""
        ...

    def stop_capture(self) -> None: ...

    def captured_ms(self) -> float:
        """Device-reported ADC position since ``start_capture()``: monotonic, non-decreasing.

        The capture twin of ``rendered_ms()`` and the timestamp oracle of T-0021 criterion 3: it
        integrates the device's own clock, so it keeps advancing through a delivery stall and
        through an xrun, and it is in the same domain as the block timestamps handed to
        ``set_capture_callback``'s callback.
        """
        ...

    def set_capture_callback(self, callback: CaptureCallback) -> None:
        """Install the delivery callback: ``callback(samples_f32, adc_ms, xrun)``.

        Called on the *device* thread, never on the event loop (T-0021 criterion 5). ``adc_ms``
        is the capture time of the buffer's **first** sample -- PortAudio's
        ``time_info.inputBufferAdcTime``, not the moment the callback ran.
        """
        ...

    def queued_ms(self) -> float:
        """Audio handed over that the device has not taken yet, in ms.

        Kept apart from ``rendered_ms()`` on purpose, and the reason both the flow control and
        the drain rule ask for it (rulings A15, A20): under output latency ``written - rendered``
        never falls below the DAC lag, so a writer pacing itself on that difference stops feeding
        the device after one buffer, and a drain rule reading it waits for a moment that never
        comes. ``queued_ms() == 0`` says "everything written has left our hands", whatever the
        lag is -- and the lag is not a constant anyone may assume (PortAudio measures it per
        callback; another client can force the graph quantum).
        """
        ...


class PortAudioDevice:
    """``AudioDevice`` on PortAudio (``sounddevice``), reaching PipeWire through its ALSA plugin.

    The ring between ``write()`` and the callback is a preallocated array with a lock: the
    callback allocates nothing and does no I/O, and the lock is held for a memory copy.
    """

    def __init__(
        self,
        *,
        sink: str | None = None,
        source: str | None = None,
        block_ms: int = CAPTURE_BLOCK_MS,
        latency_ms: float = DEFAULT_LATENCY_MS,
        input_latency_ms: float = DEFAULT_INPUT_LATENCY_MS,
        sample_rate: int = DEVICE_RATE,
        channels: int = 1,
        device: str | int | None = None,
        input_device: str | int | None = None,
    ) -> None:
        self.sink = sink
        self.source = source
        self.block_ms = block_ms
        self.latency_ms = latency_ms
        self.input_latency_ms = input_latency_ms
        #: Graph quantum in ms, or ``0.0`` meaning *unknown*. Read beside ``quantum_source``,
        #: never alone: on this backend it is a **request** in every case where it is non-zero
        #: (ruling A13), because neither ``pw-dump`` nor PortAudio reports the live quantum.
        self.quantum_ms = 0.0
        #: Where ``quantum_ms`` came from (ruling A13): ``"request"`` -- the
        #: ``PIPEWIRE_LATENCY`` we asked for, read back off our own node in the graph, which
        #: proves only that the plugin carried the request in; ``"callback"`` -- the frame count
        #: PortAudio hands the callback, which on the plain-default path is all there is; or
        #: ``"unknown"``, which always comes with ``0.0``. **Neither value is the negotiated
        #: quantum**: that is ``pw-top``'s QUANT, a group-B measurement. A number without its
        #: provenance cannot be compared later (CLAUDE.md).
        self.quantum_source = "unknown"
        #: True once a node name was requested: ``start()`` owns ``quantum_ms`` from then on and
        #: the callback must not overwrite it with the block size it was handed.
        self._quantum_from_graph = False
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.input_device = input_device
        #: Frames the stream did not carry our audio, from either cause: the frames the device
        #: had to invent because our ring was short, and -- on a callback the backend itself
        #: flagged ``output_underflow`` -- the whole callback, because a flagged callback is
        #: suspect in whole. Per callback the increment is
        #: ``max(frames - taken, frames if output_underflow else 0)``: the two causes are never
        #: added together. Monotone while the stream is open, because ``audio_io.underruns``
        #: reads it as a delta (T-0006 ruling A22, T-0035 criteria 1-2).
        self.underflow_frames = 0
        #: Callbacks the backend flagged ``output_underflow``, counted apart from our own empty
        #: ring so "we were late" and "the graph starved" stay distinguishable in a report. Not
        #: a protocol member (criterion 10c): a ``PortAudioDevice`` self-report, like
        #: ``played_ms()`` in T-0006.
        self.device_xruns = 0

        # -- the input side (T-0021 criterion 1). One duplex stream serves both directions when
        # a source is configured, so ``captured_ms()`` and ``rendered_ms()`` share a clock base.
        self._capturing = False
        #: ADC position at the end of the last callback's input buffer, on the *ADC* clock and
        #: on the same base as the ``adc_ms`` handed to the capture callback (ruling A9). Never
        #: a count of delivered frames: that clock stops at a stall, freezes across an input
        #: overflow and starts at capture time rather than at the stream's, so the gate and
        #: ``yielded_ms`` would disagree with the block timestamps by a permanent offset.
        self._adc_ms = 0.0
        self._adc_base: float | None = None
        self._capture_callback: CaptureCallback | None = None

        self._lock = threading.Lock()
        self._ring = np.zeros(int(RING_MS * sample_rate / 1000.0), dtype=np.float32)
        self._read = 0
        self._fill = 0
        self._written_frames = 0
        self._frames_out = 0
        self._dac_lag_frames = round(latency_ms * sample_rate / 1000.0)
        self._rendered_frames = 0
        self._stream: Any = None

    # -- AudioDevice ---------------------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice

        block_frames = round(self.block_ms * self.sample_rate / 1000)
        # ``latency_ms`` is a cap (criterion 1: configured output latency <= 30 ms), not a
        # target: every millisecond PortAudio holds is a millisecond ``pause()`` cannot silence
        # and a millisecond ``played_ms`` has to subtract, so ask for the least the device
        # offers, floored at one block.
        info = sounddevice.query_devices(self.device, "output")
        low_s = max(float(info["default_low_output_latency"]), block_frames / self.sample_rate)
        requested_s = min(self.latency_ms / 1000.0, low_s)
        duplex = self.source is not None or self.input_device is not None
        node, props = self._pipewire_target()
        with _PipewireEnv(latency=f"{block_frames}/{self.sample_rate}", node=node, props=props):
            if not duplex:
                self._stream = sounddevice.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=block_frames,
                    latency=requested_s,
                    device=self.device,
                    callback=self._stream_callback,
                )
            else:
                # One duplex stream, not two (T-0021 criterion 1 and its implementation note):
                # two streams mean two clocks, and a capture timestamp converted into the
                # playback clock by subtracting arrival times is what criterion 3 forbids.
                # The *source* is chosen by the PipeWire environment above, not by a PortAudio
                # device name (ruling A11): PortAudio cannot name ``astra-test-sink.monitor``,
                # and capturing a sink's monitor needs ``stream.capture.sink=true`` besides.
                # ``input_device`` stays as an escape hatch for a real ALSA capture device.
                in_info = sounddevice.query_devices(self.input_device, "input")
                low_in_s = max(
                    float(in_info["default_low_input_latency"]), block_frames / self.sample_rate
                )
                requested_in_s = min(self.input_latency_ms / 1000.0, low_in_s)
                self._stream = sounddevice.Stream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=block_frames,
                    latency=(requested_in_s, requested_s),
                    device=(self.input_device, self.device),
                    callback=self._duplex_callback,
                )
            self._stream.start()
        # The callback runs on a PortAudio thread that knows nothing about the interpreter: if
        # the process exits with the stream open, that thread calls into a finalised interpreter
        # and the process dies with a segmentation fault at shutdown.
        atexit.register(self.stop)
        negotiated = self._stream.latency
        if isinstance(negotiated, tuple):
            self.input_latency_ms = 1000.0 * float(negotiated[0])
            self.latency_ms = 1000.0 * float(negotiated[1])
        else:
            self.latency_ms = 1000.0 * float(negotiated)
        self._dac_lag_frames = round(self.latency_ms * self.sample_rate / 1000.0)
        requested = self._requested_peers()
        if not requested:
            # Nobody named a node: nothing to place, nothing to verify, no subprocess. The
            # plain-default path is exactly what it was (T-0035 criterion 4).
            return
        self._quantum_from_graph = True
        try:
            if self.sink is not None and node != self.sink:
                # ``PIPEWIRE_NODE`` is per-process and a duplex stream has two halves, so it
                # went to the input half above; the output half is placed through the graph
                # (criterion 6).
                self._place_output(self.sink)
            self._verify_placement(requested)
        except BaseException:
            # A refused placement must not leave a live stream behind: it would go on pulling
            # the callback on a PortAudio thread, into a node nobody asked for, until the
            # ``atexit`` hook ran. ``stop()`` is idempotent, so a caller's own ``stop()`` in a
            # ``finally`` still works.
            self.stop()
            raise

    def _requested_peers(self) -> dict[str, str]:
        """The node each half was asked to land on, by half (empty when none was named).

        A monitor source is the sink's own node: no graph contains a ``.monitor`` node, so the
        peer to expect for a monitor capture is the sink itself (ruling A5). ``input_device`` is
        a PortAudio device name, not a graph node, and is therefore not verified here.
        """
        peers: dict[str, str] = {}
        if self.sink is not None:
            peers["output"] = node_name(self.sink)
        if self.source is not None:
            peers["input"] = node_name(self.source)
        return peers

    def _place_output(self, sink: str) -> None:
        """Link this process's output ports to ``sink`` -- the injectable placement seam.

        ``PIPEWIRE_NODE`` is reserved for the input half (criterion 6), so when the sink and the
        source are different nodes the output half is placed here, afterwards, through the
        graph. ``pw-link`` is the mechanism on this box; the seam exists so group B can replace
        it without touching a caller, and so group A can test the policy without a graph. It is
        **not** a proof: ``_verify_placement`` reads the graph afterwards either way.

        The link is made **by port id** (ruling A9). A PortAudio stream is named by the ALSA
        plugin and that name is neither unique nor ours to choose -- two clients called
        ``ALSA plug-in [python3]`` is the normal case on this box -- so linking by
        ``node.name`` would hand our audio to whichever of them ``pw-link`` resolved first. The
        ids come from the dump this method just parsed, which is the only identifier the graph
        guarantees. Group B confirms the ``pw-link`` CLI shape on the rig; until then this path
        has been exercised against recorded graphs only.
        """
        objects = _graph(f"placing the output half on {sink!r}")
        pid = os.getpid()
        halves = _stream_nodes(objects, pid)
        node_id = halves.get("output")
        if node_id is None:
            raise PlacementError(
                f"this process (pid {pid}) has no Stream/Output/Audio node in the graph yet, so "
                f"its output half cannot be linked to {sink!r} (T-0035 criterion 6)"
            )
        ours = _ports(objects, node_id, direction="out")
        theirs = _sink_inputs(objects, sink)
        if not ours or not theirs:
            raise PlacementError(
                f"cannot link the output half to {sink!r}: this process's node {node_id} has "
                f"{len(ours)} output port(s) and {sink!r} has {len(theirs)} playback port(s)"
            )
        for index, target in enumerate(theirs):
            # One mono port feeds every channel of the sink; a stereo stream pairs up.
            source_port = ours[index % len(ours)]
            completed = subprocess.run(
                ["pw-link", str(source_port), str(target)],
                capture_output=True,
                text=True,
                check=False,
                timeout=PW_DUMP_TIMEOUT_S,
            )
            if completed.returncode != 0:
                raise PlacementError(
                    f"pw-link {source_port} -> {target} (our node {node_id} -> {sink!r}) exited "
                    f"{completed.returncode}: {completed.stderr.strip()[:200]}"
                )

    def _verify_placement(self, requested: dict[str, str]) -> None:
        """Poll the graph until every requested half is linked where it was asked, or refuse.

        Links are made asynchronously, so this is a bounded poll rather than a single read: a
        check that read the graph once would fail intermittently on the rig, which teaches
        people to pass ``--force``. There is deliberately no opt-out -- the whole point is that
        ``auto_null`` stops being silent (criterion 5, Implementation notes).

        A half is verified when its peer set is **exactly** the requested node (ruling A9). Two
        peers -- the session manager's ``auto_null`` plus the sink ``_place_output`` added -- is
        a refusal naming both, not a pass: this module does not silently unlink somebody else's
        link, and a stream fanned out across two sinks is not the regime anyone asked for.
        """
        deadline = time.monotonic() + SETTLE_WINDOW_S
        pid = os.getpid()
        while True:
            objects = _graph(f"checking that the stream landed on {sorted(requested.values())}")
            peers = parse_link_peers(objects, pid)
            landed = all(peers.get(half) == frozenset({name}) for half, name in requested.items())
            if landed:
                self._read_quantum(objects, pid)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(SETTLE_POLL_S)
        actual = ", ".join(
            f"{half}: requested {name!r}, actual "
            f"{sorted(peers[half]) if half in peers else 'no such stream half'}"
            for half, name in sorted(requested.items())
        )
        raise PlacementError(
            f"the stream did not land where it was asked within {SETTLE_WINDOW_S * 1000:.0f} ms "
            f"({actual}) -- playing into the wrong node is silence that looks like success, so "
            f"this is refused rather than reported (T-0035 criteria 5-6)"
        )

    def _read_quantum(self, objects: list[dict[str, Any]], pid: int) -> None:
        """Read our node's quantum **request** back off the graph, or declare it unknown.

        Labelled ``"request"`` and not ``"negotiated"`` on purpose (ruling A13): ``node.latency``
        is the ``PIPEWIRE_LATENCY`` we set, echoed back. The live quantum is ``pw-top``'s QUANT.
        """
        requested = parse_quantum(objects, pid)
        if requested is None:
            self.quantum_ms, self.quantum_source = 0.0, "unknown"
        else:
            self.quantum_ms, self.quantum_source = requested, "request"

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._capturing = False
        if stream is not None:
            atexit.unregister(self.stop)
            stream.stop()
            stream.close()

    def _pipewire_target(self) -> tuple[str | None, str | None]:
        """Which PipeWire node this stream connects to, and the props it needs (ruling A11).

        The capture source is named here, not through PortAudio: PortAudio's device list has no
        entry for ``astra-test-sink.monitor``, and a *monitor* capture additionally needs
        ``stream.capture.sink=true`` or the session manager silently links the stream to the
        default source -- on this box the webcam mic, which records the room and looks exactly
        like a working rig (the same trap ``scripts/audio_loopback.py::Monitor`` documents).

        **``PIPEWIRE_NODE`` is one variable and a duplex stream has two halves**, so this names
        the *source's* node when a source is configured -- the **input** half's (T-0035
        criterion 6). That is also exact for the null-sink rig, where the source is the sink's
        own monitor and both halves are the same node (criterion 9's bit-exact echo path). When
        the sink is a different node, the output half is placed afterwards by ``_place_output``
        and the result is verified against the graph by ``_verify_placement``; neither half is
        left to the session manager. The mechanism behind ``_place_output`` (``pw-link``) has
        been exercised against recorded graphs only -- group B runs it on the rig (P1).
        """
        if self.source is None:
            return self.sink, None
        node = self.source[: -len(".monitor")] if self.source.endswith(".monitor") else self.source
        monitor = self.source.endswith(".monitor") or self.source == self.sink
        return node, "{ stream.capture.sink = true }" if monitor else None

    # -- the input side (T-0021 criterion 1) ----------------------------------------------
    def start_capture(self) -> None:
        """Open the (duplex) stream if it is not open yet, and let input through.

        A device opened without a source has no input half at all: saying so is the difference
        between "capture produced nothing" and "capture was never wired" (ruling A14).
        """
        if self.source is None and self.input_device is None:
            raise RuntimeError(
                "this device was opened without a capture source, so it has no input half: "
                "open_device(sink=..., source=...) (T-0021 criterion 1, ruling A14)"
            )
        self.start()
        self._capturing = True

    def stop_capture(self) -> None:
        self._capturing = False

    def captured_ms(self) -> float:
        """The ADC clock: ``last inputBufferAdcTime + frames/rate``, one base with ``adc_ms``.

        Derived from the device's own capture timestamps and **not** from delivered frames
        (ruling A9), so it advances through a delivery stall and across an input overflow, and
        it is zero-based at the stream's first callback rather than at ``start_capture()``.
        Both properties matter to the self-speech gate: the playback span is recorded on this
        clock, and audio played before capture started would otherwise be dated in a different
        domain from every block timestamp.
        """
        with self._lock:
            return self._adc_ms

    def set_capture_callback(self, callback: CaptureCallback) -> None:
        with self._lock:
            self._capture_callback = callback

    def write(self, block: npt.NDArray[np.float32]) -> None:
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        deadline = time.monotonic() + WRITE_TIMEOUT_S
        while True:
            with self._lock:
                if self._ring.size - self._fill >= samples.size:
                    start = (self._read + self._fill) % self._ring.size
                    first = min(samples.size, self._ring.size - start)
                    self._ring[start : start + first] = samples[:first]
                    self._ring[: samples.size - first] = samples[first:]
                    self._fill += samples.size
                    self._written_frames += samples.size
                    return
            if time.monotonic() > deadline:
                raise TimeoutError("the audio device did not drain: write() waited 2 s")
            time.sleep(0.001)

    def rendered_ms(self) -> float:
        with self._lock:
            rendered = max(0, self._frames_out - self._dac_lag_frames)
            self._rendered_frames = min(max(self._rendered_frames, rendered), self._written_frames)
            return 1000.0 * self._rendered_frames / self.sample_rate

    def queued_ms(self) -> float:
        """Audio we have handed over that the callback has not taken yet.

        The flow-control seam ``PipeWireAudioIO`` uses instead of ``written - rendered``: here
        ``rendered_ms()`` already subtracts the output latency, so the difference would never
        shrink below it and the writer would stop feeding the stream after one buffer.
        """
        with self._lock:
            return 1000.0 * self._fill / self.sample_rate

    @property
    def written_ms(self) -> float:
        with self._lock:
            return 1000.0 * self._written_frames / self.sample_rate

    # -- the device thread ---------------------------------------------------------------
    def _stream_callback(
        self, outdata: npt.NDArray[np.float32], frames: int, time_info: Any, status: Any
    ) -> None:
        """PortAudio's device thread. No event loop, no allocation, no I/O (criterion 13)."""
        with self._lock:
            taken = min(frames, self._fill)
            first = min(taken, self._ring.size - self._read)
            outdata[:first, 0] = self._ring[self._read : self._read + first]
            outdata[first:taken, 0] = self._ring[: taken - first]
            outdata[taken:, 0] = 0.0
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
            self._read = (self._read + taken) % self._ring.size
            self._fill -= taken
            self._frames_out += taken
            # Two causes, counted as their maximum and never summed (criterion 1): the frames we
            # were short, and -- when the backend flagged the callback -- the whole callback,
            # which is the only way a graph-side xrun is visible at all (finding F2).
            flagged = bool(getattr(status, "output_underflow", False))
            missing = frames - taken
            if flagged:
                self.device_xruns += 1
                if frames > missing:
                    missing = frames
            self.underflow_frames += missing
            if not self._quantum_from_graph:
                # No node was requested, so no graph was read (ruling A1): the frame count is
                # all the provenance there is, and it says so.
                self.quantum_ms = 1000.0 * frames / self.sample_rate
                self.quantum_source = "callback"
            lag = float(getattr(time_info, "outputBufferDacTime", 0.0)) - float(
                getattr(time_info, "currentTime", 0.0)
            )
            if lag > 0.0:
                self._dac_lag_frames = round(lag * self.sample_rate)

    def _duplex_callback(
        self,
        indata: npt.NDArray[np.float32],
        outdata: npt.NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """The one device thread, both directions (T-0021 criteria 1 and 5).

        No asyncio, no I/O, no unbounded allocation: it copies one buffer out of PortAudio's
        memory (which PortAudio reuses the moment this returns) and hands it straight to the
        capture path's own bounded ring.
        """
        self._stream_callback(outdata, frames, time_info, status)
        with self._lock:
            handler = self._capture_callback
            capturing = self._capturing
            # ``inputBufferAdcTime`` is the capture instant of the buffer's first sample: the
            # only honest timestamp there is (criterion 3). The **stream's** first callback sets
            # the origin -- not the first callback after ``start_capture()`` -- so ``adc_ms``,
            # ``captured_ms()`` and therefore the playback spans the gate reads all share one
            # base and one rate, whatever order capture and playback were started in (A9).
            adc = float(getattr(time_info, "inputBufferAdcTime", 0.0))
            if self._adc_base is None:
                self._adc_base = adc
            adc_ms = 1000.0 * (adc - self._adc_base)
            self._adc_ms = adc_ms + 1000.0 * frames / self.sample_rate
            if not capturing:
                return
            lost = bool(getattr(status, "input_overflow", False))
        if handler is not None:
            handler(np.array(indata[:, 0], dtype=np.float32), adc_ms, lost)


class _PipewireEnv:
    """Set ``PIPEWIRE_LATENCY`` / ``PIPEWIRE_NODE`` for the stream being opened, then undo it.

    PipeWire reads both when a client connects, so they have to be in the environment across
    ``OutputStream(...)`` and nowhere else. ``PIPEWIRE_LATENCY`` is a **request**, and one this
    module cannot check: the node echoes it back verbatim in ``pw-dump`` (which is why
    ``quantum_source`` calls the value ``"request"``, ruling A13) and the live quantum the graph
    actually runs at is visible only in ``pw-top``'s QUANT column.
    """

    def __init__(self, *, latency: str, node: str | None, props: str | None = None) -> None:
        self._wanted: dict[str, str | None] = {}
        self._wanted["PIPEWIRE_LATENCY"] = latency
        self._wanted["PIPEWIRE_NODE"] = node
        #: ``stream.capture.sink = true`` when the source is a sink's monitor (ruling A11).
        self._wanted["PIPEWIRE_PROPS"] = props
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self._wanted.items():
            self._saved[key] = os.environ.get(key)
            if value is not None:
                os.environ[key] = value

    def __exit__(self, *exc: object) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def open_device(
    *,
    sink: str | None = None,
    source: str | None = None,
    block_ms: int = CAPTURE_BLOCK_MS,
    latency_ms: float = DEFAULT_LATENCY_MS,
    device: str | int | None = None,
    input_device: str | int | None = None,
) -> PortAudioDevice:
    """The real backend (T-0006 ruling A1, T-0021 ruling A1).

    ``sink``/``source`` are PipeWire node names or ids; giving a ``source`` makes the device
    duplex, which is how T-0021 criterion 1 puts capture and playback on one clock base.

    A node name that is not in the graph is refused **here**, before any stream is constructed,
    quoting the name asked for and the names that do exist (T-0035 criterion 4): falling back to
    the default device is how the 2026-09-14 audit ended up playing into ``auto_null``. When no
    node name is requested the graph is not read at all.
    """
    if latency_ms > DEFAULT_LATENCY_MS:
        raise ValueError(f"criterion 1 caps the output latency at 30 ms, got {latency_ms}")
    try:
        import sounddevice  # noqa: F401
    except OSError as exc:  # pragma: no cover - P1: this box has no libportaudio2 yet
        raise RuntimeError(
            "PortAudio is not installed (docs/tasks/T-0006.md prerequisite P1, a sysadmin step "
            f"that installs libportaudio2): {exc}"
        ) from exc
    if source is None:
        source = os.environ.get("ASTRA_AUDIO_SOURCE")
    if input_device is None:
        input_device = os.environ.get("ASTRA_AUDIO_INPUT_DEVICE")
    _refuse_unknown_nodes(sink, source)
    return PortAudioDevice(
        sink=sink,
        source=source,
        block_ms=block_ms,
        latency_ms=latency_ms,
        device=device if device is not None else os.environ.get("ASTRA_AUDIO_DEVICE"),
        input_device=input_device,
    )


def _refuse_unknown_nodes(sink: str | None, source: str | None) -> None:
    """T-0035 criterion 4: a requested node that the graph does not have is an error.

    ``.monitor`` is stripped first (ruling A5): a monitor is a port of its sink, not a node, so
    looking ``astra-test-sink.monitor`` up among the node names would refuse the only rig this
    project has.
    """
    wanted = [name for name in (sink, source) if name is not None]
    if not wanted:
        return
    nodes = parse_nodes(_graph(f"checking that {wanted} exist"))
    known = set(nodes["sinks"]) | set(nodes["sources"])
    missing = [name for name in wanted if node_name(name) not in known]
    if missing:
        raise UnknownNodeError(
            f"no such PipeWire node: {missing} -- the graph has sinks {nodes['sinks']} and "
            f"sources {nodes['sources']}. Refusing rather than falling back to the default "
            f"device, which plays into auto_null and sounds exactly like a working rig "
            f"(T-0035 criterion 4)"
        )
