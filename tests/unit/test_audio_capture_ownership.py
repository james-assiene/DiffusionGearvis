"""The capture module stays inside its fence: T-0021 criteria 5, 10 and 12(b)(c)(d)(e).

A source scan over ``local_astra/**``, deliberately blunt, and the capture-side twin of
``tests/unit/test_audio_ownership.py`` (T-0006). What it is for: a second process opening the
microphone behind the module's back (two clients, two clocks, and every capture timestamp stops
meaning anything), an AEC library sneaking in while ``Profile.aec`` still says ``"none"``, a
socket in the audio path, ``asyncio`` in the device callback, an unbounded buffer where
criterion 5 requires a bounded queue, and this module writing the turn record that belongs to
T-0007.

The scan asserts the files it polices exist and that it parsed them, so deleting
``audio_capture.py`` cannot make the module pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

from local_astra.contracts.speech import CAPTURE_BLOCK_MS

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "local_astra"
DEVICE = PKG / "conductor" / "audio_device.py"
CAPTURE = PKG / "conductor" / "audio_capture.py"
AUDIO_MODULES = (
    PKG / "conductor" / "audio_io.py",
    PKG / "conductor" / "audio_device.py",
    PKG / "conductor" / "audio_resample.py",
    PKG / "conductor" / "audio_capture.py",
)

#: The only two modules that may name a device library (T-0006 criterion 12c, widened by T-0033
#: ruling A10). ``conductor/audio_device.py`` is the audio owner: it is the module that opens a
#: stream. ``preflight.py`` is ``astra doctor``'s audio check, which has to run the *same*
#: ``import sounddevice`` + ``query_devices()`` that ``astra session`` runs -- that gap is the
#: whole point of T-0033 criterion 2 -- while opening no stream at all:
#: ``tests/unit/test_doctor_verdicts.py::test_the_audio_check_opens_no_stream`` pins that no
#: ``InputStream``/``OutputStream`` is ever constructed. It is named here rather than reaching the
#: library through ``importlib.import_module``, so this AST scan can still see every device client
#: in the package (ruling A10 refuses that dodge). Exactly these two files; no other module.
AUDIO_LIB_OWNERS = (DEVICE, PKG / "preflight.py")

DEVICE_LIBS = {
    "sounddevice",
    "_sounddevice",
    "soundfile",
    "pyaudio",
    "alsaaudio",
    "simpleaudio",
    "pulsectl",
    "pipewire",
    "ctypes",
}
SPAWNED = ("aplay", "arecord", "pw-cat", "pw-play", "pw-record", "pw-cli", "pw-loopback", "pactl")
NET_MODULES = {
    "socket",
    "socketserver",
    "ssl",
    "http",
    "urllib",
    "requests",
    "httpx",
    "websockets",
    "aiohttp",
    "zmq",
}
#: criterion 12(c): v1 ships no AEC (docs/01 R4, docs/06 R3). None of these may appear anywhere.
AEC_LIBS = {"livekit", "livekit_rtc", "webrtc", "webrtcvad", "speexdsp", "speex", "pyaec"}
AEC_STRINGS = ("module-echo-cancel", "echo-cancel", "echo_cancel")
#: criterion 10 and 12(b): the turn record, the thresholds and the session are not this module's.
FORBIDDEN_IMPORTS = {
    "local_astra.telemetry",
    "local_astra.telemetry.turns_log",
    "local_astra.bench",
    "local_astra.conductor.session",
    "local_astra.conductor.stt_client",
    "local_astra.reflex",
}
#: criterion 5: the device callback runs off the loop and allocates nothing unbounded.
CALLBACK_NAMES = ("callback", "writer", "reader", "device_thread", "_pump", "on_capture")


def python_files() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def parsed() -> dict[Path, ast.AST]:
    return {p: ast.parse(p.read_text(), filename=str(p)) for p in python_files()}


def capture_tree() -> ast.AST:
    assert CAPTURE.exists(), (
        "local_astra/conductor/audio_capture.py does not exist (T-0021 Owned files): every scan "
        "in this module would be vacuous"
    )
    return ast.parse(CAPTURE.read_text(), filename=str(CAPTURE))


def test_the_files_this_scan_polices_exist() -> None:
    """Non-vacuity, first: T-0021 owns the capture module and inherits T-0006's three."""
    missing = [str(p.relative_to(REPO)) for p in AUDIO_MODULES if not p.exists()]
    assert missing == [], f"T-0021 owned/inherited files not created: {missing}"
    assert len(python_files()) > 10, "the scan found almost nothing to scan"


def test_the_capture_module_imports_no_device_library() -> None:
    """Criterion 1 and 12(e): ``audio_capture.py`` takes an ``AudioDevice`` by injection."""
    found = sorted(imported_names(capture_tree()) & DEVICE_LIBS)
    assert found == [], (
        f"audio_capture.py imports {found}: the device library lives in audio_device.py alone, "
        f"and the unit tests must pass with neither PortAudio nor libpipewire installed "
        f"(criterion 1)"
    )


def test_only_the_device_module_names_an_audio_library_or_spawns_a_client() -> None:
    """Criterion 12(e), re-asserted for capture: one audio owner in the process."""
    assert [p.name for p in AUDIO_MODULES if not p.exists()] == [], "the scan is vacuous"
    trees = parsed()
    assert [p.name for p in AUDIO_LIB_OWNERS if not p.exists()] == [], (
        f"the fence names files that do not exist: {[str(p) for p in AUDIO_LIB_OWNERS]}"
    )
    libs = {
        str(path.relative_to(REPO)): sorted(imported_names(tree) & DEVICE_LIBS)
        for path, tree in trees.items()
        if path not in AUDIO_LIB_OWNERS and imported_names(tree) & DEVICE_LIBS
    }
    assert libs == {}, (
        f"only conductor/audio_device.py (the audio owner) and preflight.py (the doctor's device "
        f"query, which opens no stream -- T-0033 ruling A10) may import a device library; "
        f"found {libs}"
    )

    spawned: dict[str, list[str]] = {}
    for path, tree in trees.items():
        if path == DEVICE:
            continue
        hits = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(node.value.split()[:1] == [cmd] or node.value == cmd for cmd in SPAWNED)
        ]
        if hits:
            spawned[str(path.relative_to(REPO))] = hits
    assert spawned == {}, f"a second audio client is spawned from {spawned} (criterion 12e)"


def test_the_capture_path_has_no_networking() -> None:
    """Criterion 12(d): loopback-only means the audio path opens no socket at all (docs/02 §9)."""
    trees = parsed()
    offenders: dict[str, list[str]] = {}
    for path in AUDIO_MODULES:
        tree = trees.get(path)
        assert tree is not None, f"{path.relative_to(REPO)} does not exist (criterion 12d)"
        bad = sorted(imported_names(tree) & NET_MODULES)
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and "socket" in ast.unparse(node.func)
        ]
        if bad or calls:
            offenders[str(path.relative_to(REPO))] = bad + calls
    assert offenders == {}, f"networking in the audio path: {offenders}"


def docstring_nodes(tree: ast.AST) -> set[int]:
    """The ``id()`` of every docstring constant, so prose about AEC is not mistaken for wiring."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def test_no_aec_of_any_kind_arrived_with_the_capture_path() -> None:
    """Criterion 12(c): v1 ships no AEC, and the profile default says so (docs/01 R4, docs/06 R3).

    Imports are checked across the package; the ``module-echo-cancel``-style strings only in the
    audio modules, and only outside docstrings — ``config.py``'s ``Profile.aec`` literal and
    ``audio_io.py``'s prose about the deferred choice are documentation, not wiring.
    """
    capture_tree()  # non-vacuity: nothing is policed until the capture module exists
    offenders: dict[str, list[str]] = {}
    trees = parsed()
    for path, tree in trees.items():
        hits = sorted(imported_names(tree) & AEC_LIBS)
        if path in AUDIO_MODULES:
            docs = docstring_nodes(tree)
            hits += sorted(
                {
                    needle
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docs
                    for needle in AEC_STRINGS
                    if needle in node.value
                }
            )
        if hits:
            offenders[str(path.relative_to(REPO))] = hits
    assert offenders == {}, (
        f"AEC arrived with the capture path: {offenders}. v1 is headset-first with no AEC; "
        f"ERLE, double-talk and `Profile.aec != 'none'` are a later task (T-0021 Out of scope)"
    )

    from local_astra.config import Profile

    default = Profile.model_fields["aec"].default
    assert default == "none", (
        f"Profile.aec now defaults to {default!r}: criterion 12(c) keeps it at 'none' until AEC "
        f"is a task of its own"
    )


def test_the_capture_module_does_not_write_the_turn_record() -> None:
    """Criteria 10 and 12(b): the module reports numbers, ``session.py`` writes the row."""
    modules = imported_modules(capture_tree())
    offenders = sorted(
        m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS)
    )
    assert offenders == [], (
        f"audio_capture.py imports {offenders}: the `audio.capture` span row belongs to T-0007's "
        f"session, and reflex/STT work stays off this module (criterion 10)"
    )

    from local_astra.telemetry.spans import SPAN_NAMES

    assert "audio.capture" in SPAN_NAMES, (
        "criterion 10 says the span name is reported, not invented — it must already be in "
        "telemetry/spans.py, which this task does not edit"
    )


def test_the_device_callback_touches_no_asyncio_and_allocates_nothing_unbounded() -> None:
    """Criterion 5 (and .claude/rules/conductor.md): the callback is not on the event loop."""
    tree = capture_tree()
    offenders: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any(k in node.name for k in CALLBACK_NAMES):
            continue
        body = ast.unparse(node)
        hits = [w for w in ("await ", "asyncio.") if w in body]
        if isinstance(node, ast.AsyncFunctionDef):
            hits.append("async def")
        if hits:
            offenders[node.name] = hits
    assert offenders == {}, (
        f"asyncio inside the capture callback/writer: {offenders}. The device thread must never "
        f"await: a 20 ms loop pause would drop frames (criterion 5, T-0006 criterion 13)"
    )

    unbounded: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        kwargs = {kw.arg for kw in node.keywords}
        if name.endswith("deque") and "maxlen" not in kwargs and len(node.args) < 2:
            unbounded.append(ast.unparse(node))
        if name.endswith("Queue") and "maxsize" not in kwargs and not node.args:
            unbounded.append(ast.unparse(node))
    assert unbounded == [], (
        f"unbounded buffers in audio_capture.py: {unbounded}. Criterion 5 wants a bounded queue "
        f"(≥ 32 blocks) whose overflow is counted, not a list that grows until the box swaps"
    )


def protocol_members(proto: type) -> set[str]:
    """The names ``isinstance`` checks for a ``runtime_checkable`` protocol, on 3.11 and 3.12."""
    declared = set(getattr(proto, "__protocol_attrs__", set()))
    for klass in proto.__mro__:
        declared |= set(getattr(klass, "__annotations__", {}))
        declared |= {
            name
            for name, value in vars(klass).items()
            if not name.startswith("_") and callable(value)
        }
    return declared


def test_the_device_protocol_declares_the_underflow_counter_and_nothing_falls_back() -> None:
    """T-0006 ruling A22, owed to this task: the counter is a protocol member, not a ``hasattr``.

    T-0006 shipped ``audio_io.py`` reading ``underflow_frames`` through ``hasattr`` with a
    ``rendered >= written`` fallback that is dead for every device in the tree and, when forced,
    under-counts on a device whose rendered position lags the DAC (ruling A11's own reason for
    the counter). T-0021 owns the xrun accounting and both files, so the branch goes: a device
    that cannot say how many frames it invented cannot be an ``AudioDevice`` at all.
    """
    from local_astra.conductor.audio_device import AudioDevice, PortAudioDevice

    members = protocol_members(AudioDevice)
    assert "underflow_frames" in members, (
        f"AudioDevice declares {sorted(members)} and not `underflow_frames`: a backend is the "
        f"only honest source of 'frames the device emitted because we gave it nothing' "
        f"(T-0006 rulings A11 and A22), so the protocol has to ask for it"
    )

    class DeviceWithoutTheCounter:
        """Everything the protocol asks for except the counter — and so not an AudioDevice."""

        latency_ms = 20.0
        block_ms = CAPTURE_BLOCK_MS
        quantum_ms = float(CAPTURE_BLOCK_MS)

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def write(self, block: object) -> None: ...
        def rendered_ms(self) -> float:
            return 0.0

        def queued_ms(self) -> float:
            return 0.0

    assert not isinstance(DeviceWithoutTheCounter(), AudioDevice), (
        "a device with no `underflow_frames` still satisfies the AudioDevice protocol: the "
        "counter was added as documentation only, so the module is free to fall back again "
        "(T-0006 ruling A22)"
    )
    # the two devices in the tree really do conform (non-vacuity: importing PortAudioDevice
    # needs no PortAudio — only ``start()`` does, prerequisite P1)
    from tests.fakes.audio_device import FakeAudioDevice

    for device in (PortAudioDevice(), FakeAudioDevice()):
        assert isinstance(device, AudioDevice), (
            f"{type(device).__name__} no longer satisfies AudioDevice: "
            f"{sorted(m for m in members if not hasattr(device, m))} missing"
        )

    offenders: dict[str, list[str]] = {}
    for path, tree in parsed().items():
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            text = ast.unparse(node)
            if "underflow_frames" not in text:
                continue
            if name.endswith("hasattr") or (name.endswith("getattr") and len(node.args) > 2):
                hits.append(text)
        if hits:
            offenders[str(path.relative_to(REPO))] = hits
    assert offenders == {}, (
        f"`underflow_frames` is still reached through a fallback: {offenders}. Ruling A22 makes "
        f"it a protocol member and drops the branch; a `hasattr` guard here means the "
        f"rendered-position path is still reachable and it under-counts on a lagged device"
    )


def test_capture_no_longer_raises_not_implemented() -> None:
    """T-0021's goal, and the one T-0006 test that must retire with it.

    ``tests/unit/test_audio_ownership.py::test_capture_still_raises_not_implemented_for_t_0006b``
    asserts the opposite (T-0006 criterion 12a: capture is out of scope there). The two cannot
    both hold: when ``/implement T-0021`` starts, that test is deleted in the same commit that
    activates this one. Reported to the orchestrator rather than done here, because deleting it
    now would change the result of T-0006's suite while T-0006 is still open.
    """
    from local_astra.conductor.audio_io import PipeWireAudioIO

    io = object.__new__(PipeWireAudioIO)
    try:
        io.capture()
    except NotImplementedError as exc:
        raise AssertionError(
            f"capture() still raises NotImplementedError({str(exc)!r}): T-0021 is the task that "
            f"makes audio enter this system"
        ) from exc
