# services/

Launch scripts and configs for the GPU/CPU services. Every script binds to loopback only and
is started through `/launch` (the Bash guard requires the token for `vllm serve`, `moshi-server`,
`astra bench`, `astra eval`). `versions.lock` pins everything not covered by `uv.lock`.

| script | service | port | notes |
|---|---|---|---|
| `vllm.sh` | DiffusionGemma NVFP4 on vLLM v0.29.0 (+ plugin) | 8001 | `--network host`; tap PUB on 8002 |
| `stt.sh` | moshi-server, `stt-1b-en_fr` | 8080 | GPU, batch_size 2 |
| `tts.sh` | Pocket TTS | 8081 | CPU, two pinned cores |

`moshi/stt-1b-en_fr.toml` is the published `config-stt-en_fr-hf.toml` with `batch_size` 2.

## How the scripts are started, and how they are stopped (T-0033)

Every script here ends in a foreground server, so `astra up` starts each one **detached** — its
own session, so Ctrl-C on the command does not reach the engine — and then asks each service its
own health endpoint until it answers or the per-service budget runs out (the engine's init is
~68 s; the speech services get a short budget).

Each start leaves two pieces of evidence:

- `logs/<service>.log` — the script's stdout and stderr, appended (gitignored);
- `<git common dir>/local_astra/run/<service>.pid` — the pid, **the argv that produced it** and
  **the process start time** (`/proc/<pid>/stat` field 22), written only after the spawn
  succeeded. The directory comes from `git rev-parse --git-common-dir`, so every worktree shares
  one state directory; nothing is written under `artifacts/`.

`astra down` stops exactly what was started: for each pid file it terminates the recorded process
and then removes the file. A service recorded as a container (today STT) is stopped through the
daemon by its recorded container id instead, as described below, because a signal to the
`docker run` front end is not guaranteed to reach the container. The engine is recorded by its
launcher pid (`vllm.sh` passes no `--name`), so it is stopped by SIGTERM to its `docker run`
front end, which its PID 1 handles.

**What decides is the start time, not the command line.** A pid the kernel has since handed to an
unrelated process always carries a different start time, so that pid file is stale: `down` kills
**nothing**, says the pid was reused, and removes the file. A script that `exec`s into another
program — `tts.sh` ends in `exec taskset …` — keeps its start time while its
`/proc/<pid>/cmdline` stops matching the argv that was recorded; that is still the process `up`
started, so `down` reports the changed argv and **stops it anyway**. A pid that no longer exists
at all is a stale file: nothing killed, file removed. Every one of those exits 0.

A record of a container (today STT, `astra-stt`) is stopped **through the daemon**, by the
container id recorded in it — never by the container name, never by listing containers (T-0158).
Signalling the `docker run` front end only reaches a container whose PID 1 handles SIGTERM, and
moshi-server does not. So `down` runs `docker stop --time 10 <recorded container id>`: docker
sends SIGTERM inside the container and, after the 10 s grace period, SIGKILL. `down` then asks
`docker inspect` of that id and removes the record only when the daemon says the container is
gone or not running. If the launcher is already gone or its pid was reused, the pid is never
signalled but the recorded container is still stopped. If the daemon cannot be asked (the client
cannot run, it times out, or the socket is denied even under `sg docker`), `down` says so, quotes
why, **keeps** the record and exits non-zero: "could not ask" is not "not there". A record whose
container id is not a full 64-character hex id is refused outright — the daemon is asked nothing
and nothing is signalled — and the record is kept, which a later `down` cannot change: check
`docker ps`, stop the container yourself if it is running, then delete the pid file `down` names.

The supervisor itself never sends `SIGKILL` to a launcher pid: the child is the `docker run`
front end, and killing it would orphan the container it proxies signals to. If a service is still
alive 12 seconds after the signal, `down` says `still running after 12 s — not removed`,
**keeps** the pid file so the next `down` can try again, and exits non-zero.

`astra status` prints the same table as `up`'s final report plus what each record is — for a
containerised service that is what the daemon says about the container id it recorded, not what
`/proc` says about the launcher pid — and starts nothing.

## What a start does when something is already there (T-0049)

A health probe answers for whatever is on the port, so it can never prove that *this* start
happened. A start is proven by the launcher's own exit status — every script here ends in `exec`
on a foreground server, so any exit inside the settle window (1 s) is a failure, whatever the
status — plus an instance start time later than a reference taken before the spawn. That is what
**verified** means below, and the `<service>.pid` record is written only after it holds. A start
that cannot be verified leaves the previous record untouched: a record written on spawn is what
destroyed the provenance of the containers that were still serving on 2026-09-16 and left
`down` unable to stop them.

So the start verb decides per service, before it spawns anything, and prints one row saying which:

| row | what it means |
|---|---|
| `started` | verified new: the launcher is alive, the endpoint answers, and the instance is later than the attempt. Recorded. |
| `already running` | this CLI's own record still names the live instance and the script's digest is unchanged, so nothing was done to it |
| `CONFLICT` | something is there that this CLI cannot place. Nothing is started, stopped or recorded; the row names the remedy |
| `FAILED` / `NOT-NEW` | the start was attempted and could not be verified. Nothing is recorded |

It exits non-zero if any row is `CONFLICT`, `FAILED` or `NOT-NEW`; a run where everything was
already running exits 0, and costs no settle window.

`already running` is the whole of what a skip claims, and it is a claim about *provenance*, not
about newness: this CLI started that instance and it is still the one live. It does **not** say
the service is new, nor that its configuration matches anything beyond the script's digest — an
unchanged digest catches a changed *script*, never a changed environment variable, and two
services started from the same bytes with different `ASTRA_*` overrides look identical here.

The refusals, and what to do about each:

- **the script changed since the start** — the recorded sha256 digest of `services/<name>.sh`
  differs from the file on disk now, so skipping would leave the old configuration running under
  the new one. Reboot it deliberately: `--restart <name>`.
- **nothing this CLI recorded is answering the port** — no record at all, or the recorded pid or
  container id is gone while the endpoint still answers. Something else is serving and `astra
  down` cannot stop it (down stops only what a record names). Stop it yourself and run the start
  again. This is the state this box was in on 2026-09-16.
- **the daemon could not be asked** — a busy docker socket, a timeout, or permission denied with
  no working `sg docker` fallback. "Could not ask" is not "not there", so this is a refusal too,
  not a skip and not a restart.

`--restart all` or `--restart vllm,stt` is the deliberate reboot, and the **only** thing that
signals a running service or removes a state file: a start that reboots a live engine mid-session
is a worse bug than the one this verb was rewritten for. Each named service is stopped through
the same identity rules `down` uses and then started and verified like any other; a stop that
reports `still running` is not followed by a start.

One limitation, printed in the row rather than hidden: `vllm.sh` passes no `--name`, so docker
generates one and this CLI cannot ask the daemon which container the engine is. Its rows say
`container=unknown` and its newness rests on the launcher pid alone. **T-0048** adds
`--name astra-vllm`, and these verbs use it the moment it exists.

There are no systemd `--user` units: nothing in this repo writes one, and a doc that says
otherwise sends the next human to `systemctl --user status` for a service that was never
registered.

## Clip assets: rendered once with Pocket TTS (T-0182)

The fixed system clips (`out_of_steps`, `not_finished`, `lost_thread`; `ack`, `hmm` and
`still_working` only when named) play from `<data_dir>/clips/<kind>.s16le`, raw 24 kHz mono
s16le. They are not in git; render them into the data dir once:

```bash
services/tts.sh &                              # Pocket TTS alone, CPU only (never `up`, which also starts vLLM)
uv run python scripts/render_clips.py          # the default set; --kinds K --force --data-dir D --profile P
```

The render needs `tts.sh` up and serving on loopback; it runs on CPU only and spends no GPU
time. It writes `<data_dir>/clips/manifest.json` with each clip's text sha, the `pocket_tts`
pin and the commit it was rendered at, and skips a clip that is still current. Re-run it after
a clip text (`CLIP_TEXT`) or the `pocket_tts` pin in `versions.lock` changes: `doctor`'s `clips`
row warns, naming the kind, until the asset is current again. A missing clip only means silence.
