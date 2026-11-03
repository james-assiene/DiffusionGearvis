---
name: launch
description: Pre-flight review of a GPU or artifact-writing command (bench, eval, vllm serve, long run) by the launch-reviewer; on LAUNCH, writes the single-use approval token the bash hook checks. Required before any such command.
disable-model-invocation: true
---

Usage: `/launch <exact command> -- <purpose and the baseline it will be compared to>`

1. Spawn `launch-reviewer` with the exact command text, the purpose, and the baseline artifact path.
2. If verdict is **LAUNCH**: write the exact approved command to `$(git rev-parse --path-format=absolute --git-common-dir)/local_astra/launch_approved` (single use; the bash hook deletes it on the matching command), copy long scripts to `scripts/locked/<name>.sh` and `chmod 444` them, and run from that copy. Record command, commit hash, regime and log path in `PROGRESS.md` under *Runs*.
3. If **DO NOT LAUNCH**: show the reason and stop.
4. Long runs: `nohup ... > logs/<name>.log 2>&1 &`, record the PID. The tree must be clean (`git status`) — the hook refuses otherwise.
