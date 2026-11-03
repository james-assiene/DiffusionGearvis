"""The microphone seams T-0045's tests assume, in one place (the way ``_box_rig.py`` does for T-0033).

The defect the task fixes: ``cli.open_audio`` calls ``open_device(sink=args.sink)`` and never
passes a ``source``, so ``PortAudioDevice`` opens an **output-only** stream (``duplex = source is
not None or input_device is not None``), ``stt_blocks()`` never yields and every live turn records
``stt_frames_fed: 0``. No fast-suite row noticed, because every capture row drives the device
object directly and never goes through the CLI.

What the tests fix (the implementer owns everything else):

``local_astra/cli.py``::

    astra session --mic <PipeWire node name>   # the capture source; `none` ≡ --no-mic
    astra session --no-mic                     # output-only, explicitly
    # --source keeps meaning the *perception* source (screen | camera)
    open_device(sink=…, source=<node name | None>)     # what the flags resolve to
    "... sink=<node|default> mic=<node name|none> ..."  # describe_graph, beside sink=
    "regime: ... mic=<node name|none> ..."              # the line every turn's numbers are read with

``local_astra/preflight.py``::

    default_source() -> str | None

the **PipeWire node name** of the box's default capture source — ``pw-dump``'s
``default.audio.source``, the name ``open_device(source=…)`` and ``pw-dump`` use — or ``None``
when the graph could not be read. It is *not* the description ``wpctl status`` prints in its
Sources column: on this box the default source is the node
``alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo`` and the description is
``Logitech Webcam C925e Analog Stereo``, and handing the latter to ``open_device`` is refused by
T-0035 criterion 4 (``no such PipeWire node``), which would make *every* default session exit 1.

``Facts.default_source: str | None`` carries it into the doctor's ``audio.source`` row
(criterion 5); it is listed in ``_box_rig.FACT_FIELDS``.

Nothing here opens a device, asks this box anything, or starts a service.
"""

from __future__ import annotations

import os

import pytest

from local_astra import cli
from tests.unit import _box_rig as box_rig

#: The null-sink rig's node names (``scripts/audio_rig.sh up``), used as flag values only.
RIG_SINK = "astra-test-sink"
RIG_MIC = f"{RIG_SINK}.monitor"
#: This box's default capture source, read off ``tests/fixtures/pipewire/dump_recorded_base.json``
#: (the ``default`` Metadata object's ``default.audio.source`` → ``{"name": …}``). One literal,
#: shared with the doctor's good-box fixture (``_box_rig.DEFAULT_SOURCE_NODE``), so the node the
#: session opens and the node the doctor's row names cannot drift apart.
BOX_MIC = box_rig.DEFAULT_SOURCE_NODE
#: What ``wpctl status`` calls the same device in its Sources column (``tests/fixtures/box/``).
BOX_MIC_DESCRIPTION = "Logitech Webcam C925e Analog Stereo"
#: A capture node whose name says nothing about what kind of device it is — the other side of
#: criterion 5's "says whether it is the webcam mic" (this box's onboard analog input).
OTHER_MIC = "alsa_input.pci-0000_0c_00.4.analog-stereo"

#: The environment variable ``audio_device.open_device`` falls back to when it is given no
#: ``source`` (``source = os.environ.get("ASTRA_AUDIO_SOURCE")``, T-0035 criterion 7). It is the
#: workaround the human was told to export on 2026-09-15, so it is plausibly set in the shell a
#: session is started from; T-0045 ruling A14 makes it an explicit, *printed* step of the
#: precedence (``--mic`` > ``--no-mic`` > ``ASTRA_AUDIO_SOURCE`` > the graph's default source)
#: instead of an invisible fallback two layers below the line that prints ``mic=``.
ENV_SOURCE = "ASTRA_AUDIO_SOURCE"
#: A node name that is neither this box's default nor the rig's, so a row can tell *which* of the
#: three a session opened by the value alone.
ENV_MIC = "alsa_input.usb-exported_by_the_human-00.analog-stereo"
#: The collector seam's name, in one place so a rename is one edit.
DEFAULT_SOURCE_SEAM = "default_source"

SEAM_MESSAGE = (
    "T-0045 criterion 1: the default microphone is read from the box through "
    "`preflight.default_source() -> str | None` — the PipeWire *node name* of the default "
    "capture source (`pw-dump`'s `default.audio.source`, the name `open_device(source=…)` "
    "takes), or None when the graph could not be read. local_astra/preflight.py has no "
    f"{DEFAULT_SOURCE_SEAM}()"
)


class Recorded(RuntimeError):
    """Raised by the recording ``open_device`` instead of opening a stream.

    A ``RuntimeError`` on purpose: it is what ``open_audio`` is documented to turn into
    ``AudioDeviceUnavailable``, so a row can record the call *and* let the CLI walk its normal
    failure path without a device, a port or a service.
    """


def record_open_device(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record every ``open_device`` call the CLI makes, wherever it reaches it, and open nothing."""

    def recorder(*args: object, **kwargs: object) -> object:
        # ``_env`` is the environment as the device layer would have seen it *at the moment of the
        # call*: ``open_device`` reads ``ASTRA_AUDIO_SOURCE`` itself when ``source is None``
        # (audio_device.py line 944, unchanged by this task per criterion 6), so "this session
        # opened no microphone" is only true if the variable is not set when the call is made.
        calls.append({"args": args, "_env": os.environ.get(ENV_SOURCE), **kwargs})
        raise Recorded("T-0045: the device call was recorded, not opened")

    calls: list[dict[str, object]] = []
    monkeypatch.setattr("local_astra.conductor.audio_device.open_device", recorder, raising=False)
    if hasattr(cli, "open_device"):
        monkeypatch.setattr(cli, "open_device", recorder, raising=False)
    return calls


def pin_default_source(monkeypatch: pytest.MonkeyPatch, node: str | None) -> list[str]:
    """Make ``preflight.default_source()`` answer ``node``, so no row asks this box anything.

    Installs the stub whether or not the seam exists yet, so a row whose subject is *something
    else* (the printed line, the ``--no-mic`` path) fails on its own assertion rather than on the
    missing seam. Returns the list the stub appends to.

    ``stub_default_source`` is the strict twin: use it where "the box was consulted" is the claim.
    """
    from local_astra import preflight

    asked: list[str] = []

    def stub() -> str | None:
        asked.append(DEFAULT_SOURCE_SEAM)
        return node

    monkeypatch.setattr(preflight, DEFAULT_SOURCE_SEAM, stub, raising=False)
    if hasattr(cli, DEFAULT_SOURCE_SEAM):
        monkeypatch.setattr(cli, DEFAULT_SOURCE_SEAM, stub, raising=False)
    return asked


def stub_default_source(monkeypatch: pytest.MonkeyPatch, node: str | None) -> list[str]:
    """``pin_default_source`` plus the assertion that the seam exists at all (criterion 1)."""
    from local_astra import preflight

    assert hasattr(preflight, DEFAULT_SOURCE_SEAM), SEAM_MESSAGE
    return pin_default_source(monkeypatch, node)


def env_source(monkeypatch: pytest.MonkeyPatch, node: str | None) -> None:
    """Set (or clear) ``ASTRA_AUDIO_SOURCE`` for one row, so no row inherits the runner's shell.

    Every row that starts a session states this: the variable changes which microphone the device
    layer opens, and the box this suite runs on is the one whose owner was told to export it.
    """
    if node is None:
        monkeypatch.delenv(ENV_SOURCE, raising=False)
    else:
        monkeypatch.setenv(ENV_SOURCE, node)
