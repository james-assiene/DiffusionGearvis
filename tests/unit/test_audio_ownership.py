"""The audio module is the sole audio owner: T-0006 criteria 12(a)(c)(d) and 13.

A source scan, deliberately blunt, over ``local_astra/**``. It is the fence that catches a second
process quietly opening the device (two PipeWire clients means two clocks and the played number
stops meaning anything), a socket in the audio path (docs/02 §9: loopback only, and nothing in
``conductor/audio_*`` talks to the network at all), and an ``asyncio`` primitive in the device
callback (``.claude/rules/conductor.md``: the callback runs off the loop).

The scan is not vacuous: it asserts the files it is supposed to police exist and that it parsed
them, so deleting ``audio_device.py`` does not make the module pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "local_astra"
DEVICE = PKG / "conductor" / "audio_device.py"
AUDIO_MODULES = (
    PKG / "conductor" / "audio_io.py",
    PKG / "conductor" / "audio_device.py",
    PKG / "conductor" / "audio_resample.py",
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

#: Libraries that open a sound device. Only ``AUDIO_LIB_OWNERS`` may name one
#: (criterion 12c). ``ctypes`` is here because the libpipewire route loads it that way.
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
#: Commands that would spawn a second audio client behind the module's back (criterion 12c).
SPAWNED = ("aplay", "arecord", "pw-cat", "pw-play", "pw-record", "pw-cli", "pw-loopback", "pactl")
#: Networking in the audio path (criterion 12d).
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
    "msgpack",
}


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


def parsed() -> dict[Path, ast.AST]:
    return {p: ast.parse(p.read_text(), filename=str(p)) for p in python_files()}


def test_the_files_this_scan_polices_exist() -> None:
    """Non-vacuity: T-0006 owns these three modules (docs/tasks/T-0006.md Owned files)."""
    missing = [p.relative_to(REPO) for p in AUDIO_MODULES if not p.exists()]
    assert missing == [], f"T-0006 owned files not created: {missing}"
    assert len(python_files()) > 10, "the scan found almost nothing to scan"


def test_only_the_device_module_names_an_audio_library() -> None:
    """Criterion 12(c): one audio owner in the process."""
    assert [p.name for p in AUDIO_MODULES if not p.exists()] == [], (
        "the scan is vacuous until T-0006's modules exist (docs/tasks/T-0006.md Owned files)"
    )
    assert [p.name for p in AUDIO_LIB_OWNERS if not p.exists()] == [], (
        f"the fence names files that do not exist: {[str(p) for p in AUDIO_LIB_OWNERS]}"
    )
    offenders = {
        str(path.relative_to(REPO)): sorted(imported_names(tree) & DEVICE_LIBS)
        for path, tree in parsed().items()
        if path not in AUDIO_LIB_OWNERS and imported_names(tree) & DEVICE_LIBS
    }
    assert offenders == {}, (
        f"only local_astra/conductor/audio_device.py (the audio owner) and local_astra/"
        f"preflight.py (the doctor's device query, which opens no stream -- T-0033 ruling A10) "
        f"may import a device library (criterion 12c); found {offenders}"
    )


def test_the_device_module_really_is_the_backend() -> None:
    """The mirror image of the test above: an empty scan must not be able to pass."""
    assert DEVICE.exists(), "local_astra/conductor/audio_device.py (criterion 1) does not exist"
    names = imported_names(ast.parse(DEVICE.read_text()))
    assert names & DEVICE_LIBS, (
        f"audio_device.py imports no device library at all ({sorted(names)}): either the backend "
        f"moved (criterion 12c) or the module is still a stub"
    )


def test_nothing_spawns_a_second_audio_client() -> None:
    """Criterion 12(c): no ``aplay``/``arecord``/``pw-*`` from inside the package."""
    assert [p.name for p in AUDIO_MODULES if not p.exists()] == [], (
        "the scan is vacuous until T-0006's modules exist (docs/tasks/T-0006.md Owned files)"
    )
    offenders: dict[str, list[str]] = {}
    for path, tree in parsed().items():
        if path == DEVICE:
            continue
        found = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(node.value.split()[:1] == [cmd] or node.value == cmd for cmd in SPAWNED)
        ]
        if found:
            offenders[str(path.relative_to(REPO))] = found
    assert offenders == {}, f"a second audio client is spawned from {offenders} (criterion 12c)"


def test_the_audio_path_has_no_networking() -> None:
    """Criterion 12(d): the audio modules import no networking module and open no socket."""
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
    assert offenders == {}, f"networking in the audio path: {offenders} (docs/02 §9, loopback)"


def test_the_device_module_calls_no_asyncio_primitive() -> None:
    """Criterion 13: the device callback/writer runs off the event loop.

    Encoded where it is checkable: the module that owns the callback does not import or touch
    ``asyncio`` at all, and any function elsewhere in the audio path whose name says it is the
    callback or the writer thread contains no ``await`` and no ``asyncio.`` attribute.
    """
    assert DEVICE.exists(), "local_astra/conductor/audio_device.py (criterion 1) does not exist"
    device_tree = ast.parse(DEVICE.read_text())
    assert "asyncio" not in imported_names(device_tree), (
        "audio_device.py imports asyncio: the device callback must not touch the event loop "
        "(criterion 13, .claude/rules/conductor.md)"
    )
    used = {node.id for node in ast.walk(device_tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(device_tree) if isinstance(node, ast.Attribute)
    }
    assert "asyncio" not in used, "audio_device.py uses asyncio (criterion 13)"
    awaits = [ast.unparse(n) for n in ast.walk(device_tree) if isinstance(n, ast.Await)]
    assert awaits == [], f"audio_device.py awaits inside the device layer: {awaits[:3]}"

    offenders: dict[str, list[str]] = {}
    for path in AUDIO_MODULES:
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not any(k in node.name for k in ("callback", "writer", "device_thread", "_pump")):
                continue
            body = ast.unparse(node)
            hits = [w for w in ("await ", "asyncio.") if w in body]
            if isinstance(node, ast.AsyncFunctionDef):
                hits.append("async def")
            if hits:
                offenders[f"{path.name}::{node.name}"] = hits
    assert offenders == {}, f"asyncio inside the device callback/writer: {offenders} (criterion 13)"
