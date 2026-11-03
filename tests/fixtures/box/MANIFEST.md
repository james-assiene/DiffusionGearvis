# tests/fixtures/box — a recording of this box on 2026-09-14 (T-0033)

Provenance: the **5090 box** (`gpu-box`, Linux 7.0.0-31-generic, PipeWire 1.0.5,
driver/GPU per `nvidia_smi.csv`), recorded 2026-09-14 by the T-0033 test-author with the
services that were running at the time (`astra-stt` on CPU, a vLLM container started by
`services/vllm.sh`, Pocket TTS on 8081). Every file is the **verbatim** stdout of the command
named below, with the one edit called out explicitly. All commands are read-only and spend no
GPU time (`.claude/rules/gpu-services.md`: status polls are free).

These files exist so the doctor's *parsers* (`local_astra/preflight.py`) are tested against output
a real box produced, not against strings written to match the parser.

| file | command | notes |
|---|---|---|
| `wpctl_status.txt` | `wpctl status` | Verbatim. This box's live P0-3 state: the default sink is `Dummy Output` (line `* 123. Dummy Output`) and the only audio source is `Logitech Webcam C925e Analog Stereo` — i.e. no usable headset/speaker sink and a webcam-only capture list. Criterion 2's FAIL rows are derived from exactly this. |
| `nvidia_smi.csv` | `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` | Verbatim: `27433 MiB, 32607 MiB` — vLLM + STT resident. Free = 5174 MiB, above the 3 GiB sampler floor (docs/02 §7), so this fixture is the **PASS** side of criterion 6; the FAIL and the no-`nvidia-smi` sides are hand-built in the test, since this box cannot produce them without stopping the engine. |
| `ss_tlnp.txt` | `ss -tlnp` | Verbatim **plus one deliberately appended row** (the last line): `LISTEN 0 4096 0.0.0.0:8081 …`. The real box listens on `127.0.0.1` only; criterion 7's FAIL has no real recording, so the fixture carries a synthetic non-loopback row, marked here, and the test asserts that row is the one that fails. Note the third column (`Peer Address:Port` = `0.0.0.0:*`) is `0.0.0.0` on *every* row: a parser that greps the file for `0.0.0.0` instead of reading the local address fails this fixture, which is the point. |
| `docker_inspect_stt.json` | `docker inspect astra-stt` | Verbatim. `.Config.Cmd` ends in `--cpu` — the live P0-2 state (container on CPU while the `lean` profile says `kyutai-gpu`). Criterion 5 reads this argv; `ASTRA_STT_CPU` (which is what `services/stt.sh` consults) appears nowhere in it, which is why the argv is the fact. |
| `docker_inspect_vllm.json` | `docker inspect infallible_kepler` | Verbatim. The container has **no `--name`** (docker generated `infallible_kepler`) and the `vllm serve` argv carries **no `--revision`** — both carve-outs named in T-0033 (`services/vllm.sh` is owned by T-0014/T-0015), so this fixture is the WARN side of criterion 5. |
| `nvidia_smi_memory_coldboot.csv` | `nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv` | **Verbatim** lines 2-3 of `logs/vram-coldboot-73a85dd.before.txt`, recorded 2026-09-16 22:17:23+01:00 at commit `73a85dd` on the same 5090 box, immediately before the cold boot that died (PROGRESS *Runs*; `docs/tasks/T-0058.md`). Three facts the 2026-09-14 recording above cannot give: the **free** column (the doctor's engine-start row must not compute `total - used`), the recogniser resident on the GPU, and the **525 MiB gap** — `3696 + 28391 = 32087`, not the `32607` the same query calls the total, which is why vLLM's own `mem_get_info` denominator (≈ 32 082 MiB) and `nvidia-smi`'s total disagree (T-0058 criterion 8). |
| `nvidia_smi_compute_apps_coldboot.csv` | `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv` | **Verbatim** lines 4-6 of the same file, same instant: Chrome's GPU process (50 MiB, a live confound) and `moshi-server` (3 180 MiB, the recogniser whose residency made the engine's startup check fail). The engine is absent because it was not running. Note the process_name column holds a full Chrome argv: a parser that splits the row on `,` and reads field 1 is reading a truncated command line, and one that reads field 2 as the memory is reading a Chrome flag. The comma-inside-the-name case is exercised with a **synthetic** row written inline in `tests/unit/test_doctor_parsers.py` rather than appended here, so both files stay verbatim. |

Licence: this repo (Apache-2.0). No third-party content; no tokens, keys or credentials appear in
the recordings (checked at recording time).

`wpctl_status.txt`, `docker_inspect_stt.json`, `docker_inspect_vllm.json`: identifiers replaced 2026-10-25 (T-0157) —
the login user → `user` (home paths read `/home/user`, `user@…` in wpctl's client column) and the hostname → `gpu-box`;
string for string, nothing else changed, so "verbatim" above means verbatim apart from these two names.

Re-record with (read-only, no token needed):

```bash
wpctl status > tests/fixtures/box/wpctl_status.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > tests/fixtures/box/nvidia_smi.csv
ss -tlnp > tests/fixtures/box/ss_tlnp.txt   # then re-append the synthetic 0.0.0.0 row
docker inspect astra-stt > tests/fixtures/box/docker_inspect_stt.json
docker inspect "$(docker ps --filter ancestor=vllm/vllm-openai:v0.29.0 -q)" \
  > tests/fixtures/box/docker_inspect_vllm.json
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv \
  > tests/fixtures/box/nvidia_smi_memory_coldboot.csv          # T-0058; re-recording it loses
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  > tests/fixtures/box/nvidia_smi_compute_apps_coldboot.csv    # the 2026-09-16 22:17 state
```

## nvidia_smi_compute_apps_engine_up.csv

`nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`, verbatim, 2026-09-17 16:45 local, this box
(RTX 5090, driver 595.84), **with the engine up and the recogniser resident** — the reading T-0058 ruling A23 owed. Taken by the
orchestrator right after the human's `/launch` cold boot at commit bd8001b (`logs/vllm-coldboot-bd8001b.log`), after one text and one
image request. The engine's process name as the driver reports it is **`VLLM::EngineCore`** (26 936 MiB), not `python3` — the
healthy-engine matcher in `local_astra/preflight.py` (`ENGINE_PROCESS_NAMES`) does not match it and reports "not gathered"; the follow-up
pins the matcher against this file. The Chrome GPU process line is the desktop's, kept as the confound it is.
