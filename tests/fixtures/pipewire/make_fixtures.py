"""Build the four ``pw-dump`` fixtures of T-0035 from the recording in this directory.

Provenance and every edit this script makes are described in ``MANIFEST.md``. Run:

    uv run python tests/fixtures/pipewire/make_fixtures.py

Input: ``dump_recorded_base.json`` -- a real ``pw-dump`` taken on this box on 2026-09-14
(libpipewire 1.0.5) while ``pw-play`` held one ``Stream/Output/Audio`` node linked to the
``auto_null`` sink, with the bulky pure-enumeration params stripped (see the manifest).

Output: ``dump_rig_ok.json``, ``dump_auto_null.json``, ``dump_no_sink.json``,
``dump_quantum_1024.json``, ``dump_not_linked_yet.json``, ``dump_sink_differs.json``,
``dump_two_peers.json``.

The edits are deliberate and mechanical: this box has never had ``astra-test-sink`` up, and a
real recording of the sink-differs-from-source case needs PortAudio (prerequisite P1), so the
rig variants are *constructed* from the recording rather than recorded.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = HERE / "dump_recorded_base.json"

#: The pid of the recorded client that owns the stream node. Tests rewrite it to their own pid
#: with ``tests/fakes/pipewire_dumps.py::with_pid`` -- ``parse_link_peers(dump, pid)`` must find
#: *this process's* node, and this process is not pid 1801525.
RECORDED_PID = 1801525
CLIENT_ID = 134
OUT_NODE = 137
AUTO_NULL = 123
SINK_ID = 300
IN_NODE = 160

Obj = dict[str, Any]


def by_id(dump: list[Obj], oid: int) -> Obj:
    for obj in dump:
        if obj["id"] == oid:
            return obj
    raise KeyError(oid)


def props(obj: Obj) -> dict[str, Any]:
    return obj["info"]["props"]


def astra_sink(dump: list[Obj]) -> list[Obj]:
    """A second null sink named ``astra-test-sink``: node 300 with four ports, like ``auto_null``."""
    node = copy.deepcopy(by_id(dump, AUTO_NULL))
    node["id"] = SINK_ID
    props(node)["node.name"] = "astra-test-sink"
    props(node)["node.description"] = "astra-test-sink"
    props(node)["object.id"] = SINK_ID
    props(node)["object.serial"] = 6001
    out = [node]
    for new_id, old_id in ((301, 106), (302, 78), (303, 99), (304, 33)):
        port = copy.deepcopy(by_id(dump, old_id))
        port["id"] = new_id
        props(port)["node.id"] = SINK_ID
        props(port)["object.id"] = new_id
        props(port)["object.serial"] = 6000 + new_id
        out.append(port)
    return out


def input_stream(dump: list[Obj]) -> list[Obj]:
    """The capture half: a ``Stream/Input/Audio`` node of the same client, with one input port."""
    node = copy.deepcopy(by_id(dump, OUT_NODE))
    node["id"] = IN_NODE
    p = props(node)
    p["media.class"] = "Stream/Input/Audio"
    p["media.category"] = "Capture"
    p["node.name"] = "astra-conductor"
    p["object.id"] = IN_NODE
    p["object.serial"] = 6100
    p["stream.capture.sink"] = True
    p["target.object"] = "astra-test-sink"
    p.pop("media.filename", None)
    p.pop("media.format", None)
    port = copy.deepcopy(by_id(dump, 151))
    port["id"] = 161
    pp = props(port)
    pp["node.id"] = IN_NODE
    pp["object.id"] = 161
    pp["object.serial"] = 6101
    pp["port.name"] = "input_FL"
    pp["port.alias"] = "astra-conductor:input_FL"
    pp["port.direction"] = "in"
    port["info"]["direction"] = "input"
    return [node, port]


def link(oid: int, out_node: int, out_port: int, in_node: int, in_port: int) -> Obj:
    return {
        "id": oid,
        "type": "PipeWire:Interface:Link",
        "version": 3,
        "permissions": ["r", "w", "x", "m"],
        "info": {
            "output-node-id": out_node,
            "output-port-id": out_port,
            "input-node-id": in_node,
            "input-port-id": in_port,
            "change-mask": ["state", "format", "props"],
            "state": "active",
            "error": None,
            "format": {"mediaType": "audio", "mediaSubtype": "dsp", "format": "F32P"},
            "props": {
                "client.id": 34,
                "factory.id": 20,
                "link.input.node": in_node,
                "link.input.port": in_port,
                "link.output.node": out_node,
                "link.output.port": out_port,
                "object.id": oid,
                "object.serial": 7000 + oid,
            },
        },
    }


def astra_client(dump: list[Obj]) -> None:
    """Rename the recorded ``pw-cat`` client: the pid is what matters, the name is cosmetic."""
    client = by_id(dump, CLIENT_ID)
    props(client)["application.name"] = "astra-conductor"


def strip_links(dump: list[Obj]) -> list[Obj]:
    return [obj for obj in dump if not obj["type"].endswith("Link")]


def build() -> dict[str, list[Obj]]:
    base: list[Obj] = json.loads(BASE.read_text())

    # -- dump_no_sink: the graph before the rig is up. No astra-test-sink, and no stream of ours.
    no_sink = [
        obj
        for obj in copy.deepcopy(base)
        if obj["id"] not in (OUT_NODE, 151, 124, CLIENT_ID) and not obj["type"].endswith("Link")
    ]

    # -- dump_auto_null: astra-test-sink exists, our stream landed on auto_null anyway.
    auto_null = strip_links(copy.deepcopy(base))
    astra_client(auto_null)
    props(by_id(auto_null, OUT_NODE))["node.name"] = "astra-conductor"
    props(by_id(auto_null, OUT_NODE))["node.latency"] = "480/48000"
    props(by_id(auto_null, OUT_NODE))["target.object"] = "astra-test-sink"
    auto_null += astra_sink(auto_null)
    auto_null += [link(138, OUT_NODE, 151, AUTO_NULL, 106), link(143, OUT_NODE, 124, AUTO_NULL, 78)]

    # -- dump_rig_ok: the same graph with the links where they were asked for, plus the capture
    #    half linked from the sink's own monitor port.
    rig_ok = strip_links(copy.deepcopy(auto_null))
    rig_ok += input_stream(rig_ok)
    rig_ok += [
        link(138, OUT_NODE, 151, SINK_ID, 301),
        link(143, OUT_NODE, 124, SINK_ID, 302),
        link(170, SINK_ID, 303, IN_NODE, 161),
    ]

    # -- dump_sink_differs: the output half on the null sink, the input half on the webcam mic
    #    (criterion 6's sink != source case, which has no working device path today).
    sink_differs = strip_links(copy.deepcopy(rig_ok))
    sink_differs += [
        link(138, OUT_NODE, 151, SINK_ID, 301),
        link(143, OUT_NODE, 124, SINK_ID, 302),
        link(171, 52, 57, IN_NODE, 161),
    ]

    # -- dump_not_linked_yet: the same, a few milliseconds earlier (the settle window).
    not_linked = strip_links(copy.deepcopy(rig_ok))

    # -- dump_two_peers: the residual of ruling A9. The session manager linked our output half
    #    to auto_null first (it is the box's default sink), then ``pw-link`` added the requested
    #    astra-test-sink: both links are live, so the output half plays into two nodes at once and
    #    half the signal goes where nobody asked. The auto_null links come *first* in the dump,
    #    as they did in time, so a parser that keeps only the last matching link reports exactly
    #    the requested node and sees nothing wrong. The input half is the webcam, as in
    #    dump_sink_differs: this is the sink != source path, the only one on which _place_output
    #    runs at all (criterion 6).
    two_peers = strip_links(copy.deepcopy(sink_differs))
    two_peers += [
        link(139, OUT_NODE, 151, AUTO_NULL, 106),
        link(144, OUT_NODE, 124, AUTO_NULL, 78),
        link(138, OUT_NODE, 151, SINK_ID, 301),
        link(143, OUT_NODE, 124, SINK_ID, 302),
        link(171, 52, 57, IN_NODE, 161),
    ]

    # -- dump_quantum_1024: the rig with a graph quantum of 1024 frames at 48 kHz.
    quantum = copy.deepcopy(rig_ok)
    for node in (OUT_NODE, IN_NODE):
        props(by_id(quantum, node))["node.latency"] = "1024/48000"

    return {
        "dump_rig_ok.json": rig_ok,
        "dump_auto_null.json": auto_null,
        "dump_no_sink.json": no_sink,
        "dump_quantum_1024.json": quantum,
        "dump_not_linked_yet.json": not_linked,
        "dump_sink_differs.json": sink_differs,
        "dump_two_peers.json": two_peers,
    }


def main() -> None:
    for name, dump in build().items():
        (HERE / name).write_text(json.dumps(dump, indent=1) + "\n")
        print(f"{name}: {len(dump)} objects")


if __name__ == "__main__":
    main()
