# tests/fixtures/pipewire — provenance

Recorded `pw-dump` output for T-0035 (audio device first contact): the graph inputs of the pure
parsers `parse_nodes` / `parse_link_peers` / `parse_quantum` in
`local_astra/conductor/audio_device.py`, so that criteria 3–6 are testable with **no PortAudio and
no rig** (prerequisite P1 is not installed on this box).

Box: `gpu-box`, Linux 7.0.0-31-generic, `pw-cli` / libpipewire **1.0.5**,
recorded 2026-09-14 by the test-author of T-0035. Licence: recorded here, Apache-2.0 with the repo.

| file | sha256 |
|---|---|
| `dump_recorded_base.json` | `feebd0a0c4e9898a68f0aa30febd67ef70e9897f2865ecbc9466e3a8d22103fe` |
| `dump_rig_ok.json` | `691f1e4f251742cbd680d069016a0c8691b4913e0a4722ec34bfa34e62bd470f` |
| `dump_auto_null.json` | `7921d2cf79f9d3566c61f9a3e606c1153ff226b1eb0448354a3ded0c9c64ecb1` |
| `dump_no_sink.json` | `f12dc5d473b9d46a331c75c6e6c4c38ced38552319e5c60a869624c9b5a6a4d2` |
| `dump_quantum_1024.json` | `ee3270eb47bc60a6bc7d0c4dd912a2c3546ee9e9b6675a04f5a28e99cb05ac5e` |
| `dump_not_linked_yet.json` | `f67e7c6769c820b58109419c0b66daf273163065af745775dc0f7f37bed41163` |
| `dump_sink_differs.json` | `2f67b2f2dbd351c66ef4ce2b61a55136d16b15b268803b6c2a4504a32b58aa21` |
| `dump_two_peers.json` | `0a6ff6c4394b1f25a8fb06d441ea5a0aaa071d4e5f5a0c8c4366fdb9c5fb09ad` |
| `make_fixtures.py` | `d13b9f3e4bf6e9f439ebd98677bfdf42931bfc5d1a88a1587318ea8784272b61` |
| `pw_top_idle.txt` | `72281c3023988c9536f13cb511a60364a95bc3e9066e596c6a30bb41cd46ff54` |
| `pw_top_active.txt` | `8c8bd33eac05a2a0f7e40863488e391007cfd927c5d19235b74f00774056eacc` |
| `make_pw_top_fixture.py` | `c1f8750a845e68f80ffddae16a9888192a9a1990a375650dd243ff3bf49e5ffd` |

## What was recorded, and what was constructed

**Recorded, unedited in substance — `dump_recorded_base.json`.** A real `pw-dump` of this box's
graph, taken while a `pw-play` client held one `Stream/Output/Audio` node:

```
python3 -c "…6 s of silence → sil.wav…"
pw-play --target auto_null sil.wav &   # a stream to hold the graph open
pw-dump > dump_raw.json                # read-only
```

Three mechanical reductions were applied to the recording and nothing else (the file is otherwise
byte-identical in content to `pw-dump`'s output, re-indented by `json.dumps(indent=1)`):

- identifiers replaced 2026-10-25 (T-0157), string for string, longest first: the hostname
  (`application.process.host`, `host-name`) → `gpu-box`; the login user
  (`application.process.user`, `user-name`, and the user part of `core.name`
  `pipewire-<user>-<pid>`) → `user`; `application.process.machine-id` → 32 zeros. Nothing else
  changed (pids, ids, serials, the webcam serial and model are kept, H-T157-2). The seven derived
  dumps were then regenerated with `make_fixtures.py` (unchanged), and each equals the same
  substitution applied to its previous bytes; the table above is `sha256sum` of the result;

- objects of type `PipeWire:Interface:{Module,Factory,Profiler}` were dropped (they carry no
  node, port or link information; 14 KB);
- the pure-enumeration `params` entries `EnumFormat`, `PropInfo`, `EnumPortConfig`, `Format` and
  `Tag` were removed from the remaining objects (66 KB, mostly the webcam's format list).
  `info.props`, `info.params.Props`, `Latency` and `ProcessLatency` are untouched —
  `info.props["node.latency"]` is the per-node quantum **request** (`PIPEWIRE_LATENCY` echoed
  back; ruling A13 — `pw-dump` carries no live quantum anywhere), and on this box
  `params.ProcessLatency` is `[]` for every node, which is itself a fact the implementer needs.

The graph as recorded: sink `auto_null` (the box's only sink — every ALC1220 analog profile reports
`available: no`, see T-0035's human queue), source
`alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo` (the webcam mic — the trap of
criterion 8's discriminator), a `Video/Source` v4l2 node, `Midi-Bridge`, the `Dummy-Driver` and
`Freewheel-Driver` nodes (which have a `node.name` and **no** `media.class`), and the `pw-play`
stream node 137 linked to `auto_null` by two links.

**The pid.** The recorded client (id 134) that owns the stream node has
`application.process.id = 1801525`; the **node** objects carry no pid at all — only
`client.id`. That is real and load-bearing: `parse_link_peers(dump, pid)` has to resolve
pid → `Client` → `client.id` → `Stream/*/Audio` node, and a parser that looks for
`application.process.id` on the node will find nothing on a real graph. Tests rewrite the pid to
their own with `tests/fakes/pipewire_dumps.py::with_pid`.

**Constructed — the four variants**, built from the recording by `make_fixtures.py` (committed
here, deterministic, no RNG). They are constructed rather than recorded because this box has never
had the null-sink rig and the sink≠source case up: `scripts/audio_rig.sh` was not run (the
recording is read-only `pw-dump`), and the PortAudio client that would produce the real thing needs
prerequisite P1. Every edit:

- `dump_no_sink.json` — the recording with our stream node, its ports, its client and all links
  removed: the graph *before* the rig is up. `astra-test-sink` does not exist. (criterion 4)
- `dump_auto_null.json` — the reproduced failure. A second null sink node `astra-test-sink`
  (id 300, a copy of `auto_null`'s node and its four ports with fresh ids) is added, the stream
  node is renamed `astra-conductor` with `node.latency = "480/48000"` and
  `target.object = "astra-test-sink"`, and the two links still point at **`auto_null`** — which is
  exactly what the 2026-09-14 audit observed when `PIPEWIRE_NODE` named a node the graph could not
  honour. (criteria 5, 12)
- `dump_rig_ok.json` — the same graph with the two output links re-pointed at `astra-test-sink`'s
  playback ports, plus a capture half: a `Stream/Input/Audio` node (id 160, same client 134,
  `stream.capture.sink = true`) linked **from** `astra-test-sink`'s own `monitor_FL` port. This is
  the null-sink rig of T-0021 criterion 9 as the graph would show it. (criteria 5, 6, 11)
- `dump_not_linked_yet.json` — `dump_rig_ok.json` with every `Link` object removed: the same
  moment a few milliseconds earlier, which is what a poll inside the 500 ms settle window sees.
  (criterion 5)
- `dump_sink_differs.json` — `dump_rig_ok.json` with the capture half linked from the **webcam's**
  `capture_FL` port instead of the sink's monitor: the `sink != source` case of criterion 6, which
  has no working device path today. (criterion 6)
- `dump_two_peers.json` — `dump_rig_ok.json` with the input half re-pointed at the webcam (the
  `sink != source` edit of `dump_sink_differs.json`, because that is the only path on which
  `_place_output` runs) and the output half linked **twice**: two residual links to `auto_null`
  (ids 139, 144, ports 151→106 and 124→78) *followed by* the two links to `astra-test-sink`
  (ids 138, 143). Nothing else differs from `dump_sink_differs.json`. This is ruling A9's
  residual: the session manager linked the default sink first, then `pw-link` added the
  requested one, and both links are live, so half the signal goes where nobody asked. The
  auto_null links come first in the file, as they did in time, which is what makes the fixture
  adversarial — a parser that keeps the last matching link per half reports `astra-test-sink`
  and the placement check finds nothing wrong. (ruling A9, criteria 5, 6)
- `dump_quantum_1024.json` — `dump_rig_ok.json` with `node.latency = "1024/48000"` on both stream
  halves: a client that **requested** a 1024-frame block (1024/48000 s = **21.333… ms**, the size
  of PipeWire's default quantum) instead of this device's 480-frame one. Under ruling A13 the
  field is a request, so this fixture is a second request value for `parse_quantum`, not a
  negotiated graph quantum. (criterion 3)

Regenerate with:

```
uv run python tests/fixtures/pipewire/make_fixtures.py
```

## Values read off these files by hand (the oracles the unit tests assert against)

`dump_rig_ok.json`:

- `Audio/Sink` nodes: `astra-test-sink`, `auto_null`.
- `Audio/Source` nodes: `alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo`
  (the `Video/Source` node is **not** an audio source, and the two driver nodes have no
  `media.class`).
- link peers of client pid `1801525`: output half → `astra-test-sink`, input half →
  `astra-test-sink` (a monitor capture links from the sink node itself; there is no
  `astra-test-sink.monitor` *node* anywhere in a PipeWire graph).
- quantum requested by that pid (`node.latency`, ruling A13): `480/48000 s = 10.0 ms`.

`dump_auto_null.json`: link peers → output `auto_null`, no input half. `dump_no_sink.json`: no
`astra-test-sink`, no stream of ours at all. `dump_quantum_1024.json`: requested quantum
`1024/48000 s = 21.3333… ms`. `dump_not_linked_yet.json`: the stream nodes exist, neither half is
linked. `dump_sink_differs.json`: output peer `astra-test-sink`, input peer
`alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo`. `dump_two_peers.json`:
output peers `auto_null` **and** `astra-test-sink` (two links, counted by hand: 139 and 144 end on
node 123 `auto_null`, 138 and 143 on node 300 `astra-test-sink`; the two links per peer are the
FL/FR ports of one node, so the *peer* count is two, not four), input peer the webcam mic.


## `pw-top` text fixtures (ruling A14)

`pw-dump` carries no live quantum (ruling A13), so the smoke script reads it from
`pw-top -b -n 2`'s QUANT column. pw-top 1.0.5 prints one row per node as

```
%s %4.1u %6.1u %6.1u %s %s %s %s  %3.1u %16.16s %s%s
S   ID  QUANT   RATE    WAIT    BUSY   W/Q   B/Q  ERR FORMAT           NAME
```

where the trailing `%s%s` is *marker* + *name*: `""` for a driver and `" + "` for a node that
follows one. (The literal `" + "` occurs exactly once in `/usr/bin/pw-top` on this box.) The
marker is therefore on the **NAME** side of the row, which is what ruling A14 is about.

**Recorded — `pw_top_idle.txt`.** A real, read-only `pw-top -b -n 2` of this box, 2026-09-14,
pw-cli/libpipewire **1.0.5**, captured with

```
pw-top -b -n 2 > pw_top_idle.txt
```

Two iterations, six nodes each, **every node idle**: `QUANT 0`, `RATE 0`, `WAIT/BUSY/W/Q/B/Q`
all `---`, an empty FORMAT column and **no follower marker anywhere** — this box's only sink is
`auto_null` and nothing was playing, so no driver was running. That is exactly why the active
case below cannot be recorded here.

**Constructed — `pw_top_active.txt`**, by `make_pw_top_fixture.py` (committed, deterministic, no
RNG; regenerate with `uv run python tests/fixtures/pipewire/make_pw_top_fixture.py`). It is the
recording's first block verbatim, followed by a second block written at pw-top's own column
positions — the idle rows it re-emits are byte-for-byte the recorded ones apart from the `S`
state column, which is how the column widths were checked. The second block adds:

- a **driver** row: the C925e source (id 52, the real node name from the recording),
  `QUANT 1024`, `RATE 48000`, no marker → 1000·1024/48000 = **21.333… ms**;
- a **follower** row: id **137** — the `Stream/Output/Audio` node of `dump_rig_ok.json`,
  `node.name` `astra-conductor` — `QUANT 480`, `RATE 48000`, `" + "` before the name →
  1000·480/48000 = **10.0 ms**;
- a second **follower** that is *not* ours: id 145, `pw-play`, `QUANT 2048`. It belongs to
  neither "ours" nor "the driver", so the report must not mention it; a parser that reads the
  marker from the ID column calls it a driver and prints `driver 2048 frames`.

Node 56 (the v4l2 source) appears in the **first** block only, so a parser that reports anything
but the last iteration is visible.

Values read off `pw_top_active.txt` by hand (the oracles of
`tests/unit/test_pw_top_parser.py`): last-block ids `[29, 30, 45, 52, 137, 145, 123]`; node 52
driver, `quant` `"1024"`, `rate` `"48000"`; node 137 follower, `"480"`/`"48000"`, name
`astra-conductor` (the marker is not part of the name); node 145 follower, `"2048"`. From
`pw_top_idle.txt`: ids `[29, 30, 45, 56, 52, 123]`, every `quant` `"0"`, no follower.
