---
paths:
  - "local_astra/cli.py"
  - "local_astra/supervisor.py"
  - "local_astra/service_state.py"
  - "services/README.md"
---

# Service lifecycle

- **A health probe answers for whatever is on the port, so it can never prove that a start happened.** On 2026-09-16 `astra up` against
  an already-serving box printed `started` and `ok` for all three services while two of three had failed: their launchers had exited, the
  container name was already taken, and the probe was talking to containers that had been up two hours. A start is proven by **the
  launcher's own exit status** (the CLI is its parent, and every `services/*.sh` ends in `exec` on a foreground server, so any exit inside
  the settle window is a failure) **plus an instance start time strictly after a reference taken before the spawn**. Never by health alone,
  and never by the log — a log line is the *explanation* in a failure row, never the detector, because matching known failure strings is a
  blacklist that fails open on the next failure mode.
- **"Could not ask" is not "not there."** A container query that fails — a busy daemon, a timeout, permission denied, which is the ordinary
  state on this box — looks exactly like an empty answer unless the code distinguishes them. Conflating the two reported a two-hour-old
  container as freshly started and wrote its foreign id into the state record, which is the provenance corruption the verb exists to
  prevent. A start whose newness cannot be established is not a success. The mirror is equally a lie: a launcher this invocation spawned
  that is alive and healthy means the start happened, and a daemon going quiet afterwards makes the **identity** unknown, not the start
  failed.
- **Write the state record only after the start is verified.** A record written on spawn is what destroyed the provenance of the still-running
  containers on 2026-09-16 and left `astra down` unable to stop them.
- **`up` skips what is already healthy, out loud, and refuses when the configuration has drifted.** It does not restart by default: an
  `astra up` that reboots a live engine mid-session is a worse bug than the one being fixed. `--restart` is the deliberate form and the
  **only** way `up` signals anything or removes a state file.
