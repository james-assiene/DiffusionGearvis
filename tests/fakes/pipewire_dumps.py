"""Load the recorded ``pw-dump`` fixtures of T-0035 and point them at this process.

The fixtures in ``tests/fixtures/pipewire/`` were recorded while *another* process (pid 1801525)
held the stream node; ``parse_link_peers(dump, pid)`` and ``parse_quantum(dump, pid)`` are asked
about **this** process, so ``with_pid`` rewrites the recorded pid. Nothing here parses the graph:
the parsers under test must do that, and a helper that re-implemented them would make every
assertion in ``tests/unit/test_pipewire_graph.py`` vacuous. The only thing this module knows about
the dump's shape is where the pid is written (``Client.info.props["application.process.id"]`` and
``pipewire.sec.pid``), which is documented in ``tests/fixtures/pipewire/MANIFEST.md``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pipewire"

#: The pid of the client that owned the stream node when the dump was recorded (MANIFEST.md).
RECORDED_PID = 1801525

#: The names the rig fixtures use, so no test spells them twice.
SINK = "astra-test-sink"
MONITOR = "astra-test-sink.monitor"
AUTO_NULL = "auto_null"
WEBCAM = "alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo"

Dump = list[dict[str, Any]]


def load_dump(name: str, *, pid: int | None = None) -> Dump:
    """Read ``tests/fixtures/pipewire/<name>.json``; rewrite the recorded pid to ``pid``."""
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        raise AssertionError(f"{path} is missing (tests/fixtures/pipewire/MANIFEST.md)")
    dump: Dump = json.loads(path.read_text())
    return dump if pid is None else with_pid(dump, pid)


def with_pid(dump: Dump, pid: int) -> Dump:
    """A copy of ``dump`` in which the recorded client's pid is ``pid``."""
    out = copy.deepcopy(dump)
    for obj in out:
        props = obj.get("info", {}).get("props", {})
        for key in ("application.process.id", "pipewire.sec.pid"):
            if props.get(key) == RECORDED_PID:
                props[key] = pid
    return out


def peer_set(peers: Any, half: str) -> frozenset[str]:
    """The set of nodes ``half`` is linked to, out of ``parse_link_peers``' answer (ruling A9).

    A9 changed the value of each half from one name to **every** peer of that half, because one
    name cannot express the residual it is about: an output half linked to ``auto_null`` *and* to
    the sink we asked for, which a last-link-wins parser reports as a clean placement. The ruling
    leaves the container open ("a set/tuple"), so this helper accepts either and normalises; what
    it refuses is the old shape, because a bare name silently drops peers.

    ``frozenset()`` is a half that exists and is linked to nothing yet (poll again inside the
    settle window); a half the process does not have is **absent** from the mapping and raises
    here, so the two stay distinguishable at every call site.
    """
    if half not in peers:
        raise AssertionError(
            f"parse_link_peers() reported no {half!r} half at all ({peers!r}): a half the "
            f"process does not have is absent from the mapping, which is not the same as a "
            f"half that is linked to nothing"
        )
    value = peers[half]
    if value is None or isinstance(value, str):
        raise AssertionError(
            f"parse_link_peers()[{half!r}] is {value!r}: ruling A9 asks for every peer of the "
            f"half (a set or tuple of node names, empty when the half is not linked yet). A "
            f"single name cannot express an output half linked to two nodes at once, which is "
            f"the residual the ruling exists for"
        )
    if not isinstance(value, set | frozenset | tuple | list):
        raise AssertionError(
            f"parse_link_peers()[{half!r}] is a {type(value).__name__}: ruling A9 asks for a "
            f"set or tuple of node names"
        )
    bad = [peer for peer in value if not isinstance(peer, str)]
    if bad:
        raise AssertionError(f"parse_link_peers()[{half!r}] contains non-names {bad!r}")
    return frozenset(value)


def without_node_latency(dump: Dump) -> Dump:
    """A copy with every ``node.latency`` prop removed: a graph that will not say its quantum."""
    out = copy.deepcopy(dump)
    for obj in out:
        obj.get("info", {}).get("props", {}).pop("node.latency", None)
    return out


def patch_pw_dump(
    monkeypatch: Any, *dumps: Any, pid: int | None = None, required: bool = True
) -> list[float]:
    """Replace ``audio_device._pw_dump`` with a sequence of recorded answers.

    ``_pw_dump() -> list[dict]`` is the one function of T-0035 that shells out (implementation
    note: "keep it behind one small function so the tests can inject a recorded dump"); this is
    the injection point every group-A test uses, and its absence is an ``AssertionError`` that
    names it rather than a silent no-op patch.

    ``required=False`` is for the rows that pin *unchanged* behaviour and must pass before the
    seam exists (the null-sink rig path, the environment restore): they patch it if it is there
    and say nothing if it is not.

    Each element is a fixture name, an already-loaded dump, or an ``Exception`` instance to
    raise. The **last** element answers every further call, so a poll inside the settle window
    sees a graph that has settled. Returns the list of call timestamps (``time.monotonic()``).
    """
    import time

    from tests.fakes.audio_device import attr_or_fail, import_or_fail

    module = import_or_fail("local_astra.conductor.audio_device", "T-0035 criteria 3-6")
    if required:
        attr_or_fail(
            module,
            "_pw_dump",
            "T-0035: the single seam that shells out to `pw-dump`, so tests can inject a "
            "recorded dump (docs/tasks/T-0035.md, Implementation notes)",
        )
    answers = [load_dump(d, pid=pid or os.getpid()) if isinstance(d, str) else d for d in dumps]
    calls: list[float] = []

    def fake_pw_dump(*args: Any, **kwargs: Any) -> Dump:
        calls.append(time.monotonic())
        answer = answers[min(len(calls) - 1, len(answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)

    monkeypatch.setattr(module, "_pw_dump", fake_pw_dump, raising=required)
    return calls


def count_pw_dump_calls(monkeypatch: Any) -> list[float]:
    """Patch ``_pw_dump`` (if it exists) with a counter that answers with an empty graph."""
    import time

    from tests.fakes.audio_device import import_or_fail

    module = import_or_fail("local_astra.conductor.audio_device", "T-0035 criteria 3-6")
    calls: list[float] = []

    def counting(*args: Any, **kwargs: Any) -> Dump:
        calls.append(time.monotonic())
        return []

    monkeypatch.setattr(module, "_pw_dump", counting, raising=False)
    return calls
