"""What the two environment variables mean: T-0035 criterion 7 (and criterion 6's carve-out).

``audio_device.py`` is the only reader of either name in the tree (grep, planning time), and today
``ASTRA_AUDIO_SOURCE`` is passed to ``input_device`` — i.e. to a **PortAudio device name**. The
documented ``ASTRA_AUDIO_SOURCE=astra-test-sink.monitor`` therefore cannot work: PortAudio's device
list has no such entry, and a monitor capture additionally needs ``stream.capture.sink=true``.
Criterion 7 splits the two meanings:

* ``ASTRA_AUDIO_SOURCE`` → ``source``: a **PipeWire node name**, the thing the rig, the docs and
  every caller already believe it is;
* ``ASTRA_AUDIO_INPUT_DEVICE`` → ``input_device``: the PortAudio escape hatch, the old meaning
  under a name that says so;
* ``ASTRA_AUDIO_DEVICE`` → ``device``: unchanged.

The third row pins behaviour that must **not** change while criterion 6 edits the caller of
``_PipewireEnv``: the environment is exactly as it was found after ``start()``, including the case
where a variable was not set before (it must be *absent* again, not empty).

Regime: PortAudio absent (P1) — ``sounddevice`` is ``tests/fakes/sounddevice_stub.py``, which
records the environment libpipewire would have read at stream construction; the graph is the
recorded ``dump_rig_ok.json``; 48 kHz mono f32, block 480 frames.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from local_astra.conductor.audio_device import open_device
from tests.fakes.pipewire_dumps import MONITOR, SINK, patch_pw_dump
from tests.fakes.sounddevice_stub import PIPEWIRE_VARS, installed, last_stream

ASTRA_VARS = ("ASTRA_AUDIO_DEVICE", "ASTRA_AUDIO_SOURCE", "ASTRA_AUDIO_INPUT_DEVICE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: Any) -> None:
    for name in (*ASTRA_VARS, *PIPEWIRE_VARS):
        monkeypatch.delenv(name, raising=False)


def test_astra_audio_source_is_a_pipewire_node_name(monkeypatch: Any) -> None:
    """Adversarial, criterion 7: the documented rig setting must reach ``source``.

    ``ASTRA_AUDIO_SOURCE=astra-test-sink.monitor`` is what ``scripts/audio_rig.sh``'s users and
    the docs set. It has to become the *PipeWire* source, which makes the device duplex and puts
    ``stream.capture.sink = true`` in ``PIPEWIRE_PROPS`` (T-0021 ruling A11 — without it the
    session manager links the capture half to the default source, on this box the webcam mic,
    which looks exactly like a working rig). Today the same string lands in ``input_device``,
    where PortAudio has never heard of it.
    """
    monkeypatch.setenv("ASTRA_AUDIO_SOURCE", MONITOR)
    patch_pw_dump(monkeypatch, "dump_rig_ok")
    with installed() as stub:
        device = open_device(sink=SINK, block_ms=10)
        assert device.source == MONITOR, (
            f"ASTRA_AUDIO_SOURCE={MONITOR!r} became source={device.source!r} / "
            f"input_device={device.input_device!r}: criterion 7 makes it the PipeWire node name"
        )
        assert device.input_device is None, (
            f"input_device is {device.input_device!r}: the PortAudio escape hatch is "
            f"ASTRA_AUDIO_INPUT_DEVICE now (criterion 7)"
        )
        device.start()
        try:
            stream = last_stream(stub)
            assert stream.kind == "duplex", "a source makes the device duplex (T-0021 crit. 1)"
            assert stream.env["PIPEWIRE_NODE"] == SINK, (
                f"the stream connected with PIPEWIRE_NODE={stream.env['PIPEWIRE_NODE']!r}; a "
                f"monitor capture names the sink node itself"
            )
            props = stream.env["PIPEWIRE_PROPS"] or ""
            assert "stream.capture.sink" in props and "true" in props, (
                f"PIPEWIRE_PROPS={props!r} at construction: without stream.capture.sink=true the "
                f"capture half silently lands on the default source (ruling A11)"
            )
        finally:
            device.stop()


def test_astra_audio_input_device_is_the_portaudio_escape_hatch(monkeypatch: Any) -> None:
    """Criterion 7: the old meaning survives, under the name that says what it is.

    A real ALSA capture device (``hw:CARD=C925e``) is a PortAudio device name, not a graph node:
    it must reach ``input_device``, make the stream duplex, and set no ``PIPEWIRE_NODE`` for the
    input half at all.
    """
    monkeypatch.setenv("ASTRA_AUDIO_INPUT_DEVICE", "hw:CARD=C925e")
    with installed() as stub:
        device = open_device(block_ms=10)
        assert device.input_device == "hw:CARD=C925e", (
            f"ASTRA_AUDIO_INPUT_DEVICE reached input_device={device.input_device!r} (criterion 7)"
        )
        assert device.source is None, "a PortAudio device name is not a PipeWire node name"
        device.start()
        try:
            stream = last_stream(stub)
            assert stream.kind == "duplex"
            assert stream.kwargs["device"] == ("hw:CARD=C925e", None), (
                f"the stream was opened on device={stream.kwargs['device']!r}: the escape hatch "
                f"names the *input* half's PortAudio device"
            )
            assert stream.env["PIPEWIRE_NODE"] is None, (
                f"PIPEWIRE_NODE={stream.env['PIPEWIRE_NODE']!r} while no PipeWire node was named"
            )
        finally:
            device.stop()


def test_the_env_is_restored_after_start(monkeypatch: Any) -> None:
    """Pinned unchanged (criterion 10, and criterion 6 edits this caller): ``_PipewireEnv`` cleans up.

    Two cases in one, because they fail differently: ``PIPEWIRE_PROPS`` was **not** set before
    (it must be absent again — an empty string would be read by the next client as a request) and
    ``PIPEWIRE_NODE`` **was** set to something of the user's (it must come back with that exact
    value). The comparison is over the whole environment, so a new variable the placement code
    leaks would fail here too.
    """
    monkeypatch.setenv("PIPEWIRE_NODE", "somebody-elses-node")
    patch_pw_dump(monkeypatch, "dump_rig_ok", required=False)
    before = dict(os.environ)
    with installed():
        device = open_device(sink=SINK, source=MONITOR, block_ms=10)
        device.start()
        try:
            assert dict(os.environ) == before, (
                "the environment differs after start(): "
                f"{set(os.environ.items()) ^ set(before.items())}"
            )
        finally:
            device.stop()
    assert dict(os.environ) == before, "the environment differs after stop()"
    assert "PIPEWIRE_PROPS" not in os.environ, (
        "PIPEWIRE_PROPS was unset before start() and must be absent afterwards, not empty"
    )
