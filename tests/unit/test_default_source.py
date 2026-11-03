"""T-0045: the microphone the box says is the default one, read from the graph and not guessed.

``astra session`` with no ``--mic`` must open the box's **default capture source** (criterion 1).
Two things have to be true of the name it resolves, and both are testable here with no device:

1. it is a **PipeWire node name** — ``pw-dump``'s ``default.audio.source``, the name
   ``open_device(source=…)`` takes and ``parse_nodes`` finds — and *not* the human-readable
   description ``wpctl status`` prints in its Sources column. On this box those are
   ``alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo`` and
   ``Logitech Webcam C925e Analog Stereo``; handing the second to ``open_device`` is refused by
   T-0035 criterion 4 (``no such PipeWire node``), so a session started the normal way would exit
   1 on every box — the opposite failure to the deaf one this task fixes, and just as loud;
2. when the graph cannot be read the answer is ``None`` (*not gathered*), never a guess and never
   an exception escaping into ``astra session``.

The expectations are hand-read from the recordings of this box in ``tests/fixtures/box/`` and
``tests/fixtures/pipewire/`` (see their MANIFESTs): the ``default`` Metadata object of
``dump_recorded_base.json`` carries ``default.audio.source -> {"name": "alsa_input.usb-046d_…"}``
and node id 52 is that name, while ``wpctl_status.txt`` marks the same device default as
``*   52. Logitech Webcam C925e Analog Stereo``.

Regime: pure. Every box command is served from those fixtures; nothing here runs ``wpctl``,
``pw-dump`` or a device, and no service is touched.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from local_astra.conductor.audio_device import parse_nodes
from tests.unit import _box_rig as rig
from tests.unit import _mic_rig as mic_rig

WPCTL = rig.fixture("wpctl_status.txt")
DUMP = (rig.REPO / "tests" / "fixtures" / "pipewire" / "dump_recorded_base.json").read_text()


class _Subprocess:
    """The ``subprocess`` module ``preflight`` sees while it is served from the recordings.

    ``run`` and ``check_output`` are the doubles. Everything else an implementation may legally
    name — the exception classes **including their base ``SubprocessError``** (T-0045 ruling A9),
    the ``PIPE``/``DEVNULL`` constants — is delegated to the real module, because a double that
    exposes only the leaf exceptions silently forbids ``except subprocess.SubprocessError:`` in
    the collector: that implementation would die of ``AttributeError`` *inside this test* instead
    of failing an assertion about the default source. Naming classes is not the same as running
    commands, so delegating them costs nothing.

    The calls that would actually run something on *this* box are refused by name, loudly: a row
    here is pure (the module docstring's regime), so reaching one of them means the fixtures are
    not being served and the answer would be this box's, not the recording's.
    """

    #: Everything in ``subprocess`` that spawns a process and is not doubled above.
    RUNS_A_COMMAND = ("Popen", "call", "check_call", "getoutput", "getstatusoutput")

    def __init__(self, run: object, check_output: object) -> None:
        self.run = run
        self.check_output = check_output

    def __getattr__(self, name: str) -> object:
        if name in self.RUNS_A_COMMAND:
            raise AssertionError(
                f"tests/unit/test_default_source.py is pure: it serves the recorded box, and "
                f"subprocess.{name} would run a command on this one. The double serves run() and "
                f"check_output(); extend it rather than letting the collector reach the box."
            )
        return getattr(subprocess, name)


def fake_box(monkeypatch: pytest.MonkeyPatch, *, answers: bool = True) -> list[list[str]]:
    """Serve ``wpctl status`` / ``pw-dump`` from the recordings; record what was asked.

    Patched at ``preflight.subprocess``, not at ``preflight._run``, so an implementation that
    shells out its own way is still served the fixtures instead of this box.
    """
    asked: list[list[str]] = []

    def run(command, *args, **kwargs):  # a subprocess.run double
        asked.append(list(command))
        if not answers:
            return subprocess.CompletedProcess(command, 1, "", "no PipeWire session")
        tool = str(command[0]).rsplit("/", 1)[-1]
        if tool == "wpctl":
            return subprocess.CompletedProcess(command, 0, WPCTL, "")
        if tool in ("pw-dump", "pw-cli"):
            return subprocess.CompletedProcess(command, 0, DUMP, "")
        return subprocess.CompletedProcess(command, 127, "", f"{tool}: not in this fixture box")

    def check_output(command, *args, **kwargs):
        done = run(command, *args, **kwargs)
        if done.returncode != 0:
            raise subprocess.CalledProcessError(done.returncode, command, done.stdout)
        return done.stdout

    double = _Subprocess(run, check_output)
    preflight = rig.module("preflight")
    monkeypatch.setattr(preflight, "subprocess", double, raising=False)
    return asked


def default_source(
    monkeypatch: pytest.MonkeyPatch, **kwargs: bool
) -> tuple[object, list[list[str]]]:
    preflight = rig.module("preflight")
    assert hasattr(preflight, mic_rig.DEFAULT_SOURCE_SEAM), mic_rig.SEAM_MESSAGE
    asked = fake_box(monkeypatch, **kwargs)
    return getattr(preflight, mic_rig.DEFAULT_SOURCE_SEAM)(), asked


def test_the_default_capture_source_is_a_node_name_not_a_wpctl_description(monkeypatch):
    """Criterion 1: the name handed to ``open_device(source=…)`` is one ``pw-dump`` knows.

    The oracle is the recorded graph itself, read here by the module that will have to accept the
    name: ``parse_nodes`` finds the node, and it does not find the description.
    """
    nodes = parse_nodes(json.loads(DUMP))
    assert mic_rig.BOX_MIC in nodes["sources"], (
        f"fixture check: the recorded graph's only audio source is {mic_rig.BOX_MIC!r}; "
        f"parse_nodes says {nodes['sources']}"
    )
    assert mic_rig.BOX_MIC_DESCRIPTION not in nodes["sources"], (
        f"fixture check: {mic_rig.BOX_MIC_DESCRIPTION!r} is a *description*, not a node — which "
        f"is why resolving the default source through `wpctl status`'s Sources column alone "
        f"makes open_device refuse it (T-0035 criterion 4)"
    )

    got, asked = default_source(monkeypatch)
    assert got == mic_rig.BOX_MIC, (
        f"criterion 1: preflight.default_source() is the PipeWire node name of the box's default "
        f"capture source ({mic_rig.BOX_MIC!r}, the `default.audio.source` of the recorded graph); "
        f"it answered {got!r} after running {asked}. A description, an id or None here is a "
        f"session that exits 1 on a box with a working microphone"
    )


def test_a_graph_that_cannot_be_read_yields_no_default_source_and_no_exception(monkeypatch):
    """Criterion 1, the other half: *not gathered* is ``None`` — never a guessed name, never a raise.

    ``astra session`` calls this on the way up; a collector that throws would turn "wpctl is not
    installed" into a traceback instead of the criterion-4 message that names what is missing.
    """
    try:
        got, asked = default_source(monkeypatch, answers=False)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"criterion 1: with no readable graph, preflight.default_source() answers None; it "
            f"raised {type(exc).__name__}: {exc}"
        ) from None
    assert got is None, (
        f"criterion 1: a box whose graph cannot be read has no known default capture source, so "
        f"the answer is None (and the CLI says so); it answered {got!r} after running {asked}"
    )


def test_the_doctor_names_the_default_capture_source_in_its_audio_source_row():
    """Criterion 5: ``astra doctor``'s ``audio.source`` row names the *default* source.

    Today the row lists the capture sources and says nothing about which one a session would
    open, which is exactly the fact the human needed on 2026-09-15.
    """
    rows = rig.by_name(
        rig.evaluate(
            **rig.good_box(
                default_source=mic_rig.BOX_MIC,
                sources=(mic_rig.BOX_MIC_DESCRIPTION,),
            )
        )
    )
    row = rows["audio.source"]
    assert row.status == "PASS", (
        f"criterion 5: a box with a default capture source passes this row; got {row}"
    )
    assert mic_rig.BOX_MIC in row.detail, (
        f"criterion 5: …and the row names the node `astra session` would open "
        f"({mic_rig.BOX_MIC!r}) — the detail read {row.detail!r}"
    )
    assert "default" in row.detail.lower(), (
        f"criterion 5: …saying that it is the default one, not just another entry in the list; "
        f"the detail read {row.detail!r}"
    )


def test_a_default_capture_source_that_was_not_read_is_a_warning_not_a_pass():
    """Ruling A17: an ungathered fact is never a PASS — and here it is a PASS with consequences.

    ``preflight``'s own discipline (``_ungathered``: "Never a PASS: unknown is not evidence") is
    broken by this row alone: a box that lists capture sources but whose *default* could not be
    read passes, and the arrival checklist's audio command — ``uv run astra session --serve-hud``
    — then exits 1 (ruling A3) on a box whose doctor was green. A human reading the green table
    and following the checklist meets a failure the doctor said nothing about, which is what the
    doctor exists to prevent.

    WARN, not FAIL: the box can still be told which microphone to use (``astra session --mic
    <node>``), so it is not a blocker — and the row has to carry that as its fix.
    """
    rows = rig.by_name(
        rig.evaluate(**rig.good_box(default_source=None, sources=(mic_rig.BOX_MIC_DESCRIPTION,)))
    )
    row = rows["audio.source"]
    assert row.status == "WARN", (
        f"ruling A17: the default capture source was not read, so `astra session` has no "
        f"microphone to open by itself and exits 1 — an ungathered fact is never a PASS; got {row}"
    )
    assert "--mic" in row.fix, (
        f"…and the row says what to do about it (name the node: astra session --mic <node>); the "
        f"fix read {row.fix!r}"
    )
    passing = rig.by_name(rig.evaluate(**rig.good_box()))["audio.source"]
    assert passing.status == "PASS", (
        f"…while a box that *did* read its default source still passes, so this row is about the "
        f"ungathered fact and not about the check being strict; got {passing}"
    )


def claim_about(detail: str, *quoted: str) -> str:
    """The row's own words: the detail with the node names and descriptions it quotes removed.

    Without this the webcam half is vacuous: the node PipeWire composed is
    ``alsa_input.usb-046d_Logitech_Webcam_C925e_…`` and the ``wpctl`` description is ``Logitech
    Webcam C925e Analog Stereo``, so "webcam appears in the detail" is true of any row that merely
    prints them. What criterion 5 asks for is a *claim* about the device, in the doctor's words.
    """
    for name in quoted:
        detail = detail.replace(name, "<node>")
    return detail


def test_the_doctor_says_when_the_default_microphone_is_the_webcams_and_when_it_cannot_tell():
    """Criterion 5: the row says *whether* the default source is the webcam mic.

    On this box the webcam is the only microphone, so "is the thing that will listen the webcam?"
    is the question the human actually has — and the answer is read off the node name PipeWire
    composed from the USB product string, never assumed. The second half is what keeps that from
    being a hard-coded phrase: a node whose name says nothing about the device gets no claim
    about it.
    """
    webcam = rig.by_name(
        rig.evaluate(
            **rig.good_box(default_source=mic_rig.BOX_MIC, sources=(mic_rig.BOX_MIC_DESCRIPTION,))
        )
    )["audio.source"]
    said = claim_about(webcam.detail, mic_rig.BOX_MIC, mic_rig.BOX_MIC_DESCRIPTION)
    assert "webcam" in said.lower(), (
        f"criterion 5: the default source {mic_rig.BOX_MIC!r} is the webcam's microphone (the "
        f"node name carries the USB product string) and the row *says* so, in its own words and "
        f"not only by quoting the node; with the names it quotes removed the detail read {said!r}"
    )
    assert mic_rig.BOX_MIC in webcam.detail, (
        f"…while still naming the node itself; the detail read {webcam.detail!r}"
    )
    anonymous = rig.by_name(
        rig.evaluate(
            **rig.good_box(
                default_source=mic_rig.OTHER_MIC, sources=("Built-in Audio Analog Stereo",)
            )
        )
    )["audio.source"]
    assert mic_rig.OTHER_MIC in anonymous.detail, (
        f"criterion 5: …the row still names the node it would open; the detail read "
        f"{anonymous.detail!r}"
    )
    assert "webcam" not in anonymous.detail.lower() and "camera" not in anonymous.detail.lower(), (
        f"criterion 5, adversarial: {mic_rig.OTHER_MIC!r} says nothing about what kind of device "
        f"it is, so the row must make no claim about one — a phrase printed for every box is not "
        f"an answer to 'is this the webcam mic?'. The detail read {anonymous.detail!r}"
    )
