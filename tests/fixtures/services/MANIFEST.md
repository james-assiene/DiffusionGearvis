# tests/fixtures/services — the artefacts of the failed start of 2026-09-16 (T-0049)

Provenance: the **5090 box** (`gpu-box`, Linux 7.0.0-31-generic), recorded
2026-09-16 by the T-0049 test-author from the state `astra up --profile lean` left behind on
2026-09-15/16 — the run that printed `started` and `ok` for all three services while two of the
three starts had failed (PROGRESS, "Live finding 2026-09-16"; `docs/tasks/T-0049.md` Goal).
Both files are **verbatim** copies, byte for byte, of what was on disk; nothing is hand-written to
match an implementation. Reading them spends no GPU time and touches no service.

| file | recorded from | notes |
|---|---|---|
| `stt_conflict.log` | `sed -n '3115,3117p' logs/stt.log` | The **tail of the failed STT launcher**, exactly as `services/stt.sh` wrote it: the docker `Conflict. The container name "/astra-stt" is already in use by container "61f0d405…"` line, an empty line, and docker's `Run 'docker run --help' for more information` trailer. The three lines are the end of the file. **The order is load-bearing**: `supervisor.log_tail` returns the last *non-empty* line, which here is the `docker run --help` trailer and not the Conflict line — so a FAILED row built from `log_tail` alone explains nothing (T-0049 criterion 1 requires the Conflict line to be quoted). That gap is what `tests/unit/test_supervisor_start_verification.py::test_a_launcher_that_exits_while_the_port_is_healthy_is_not_a_success` pins; do not "fix" it by trimming the trailer out of this file. |
| `pid_file_t0033.json` | `.git/local_astra/run/stt.pid` | The **T-0033-shape** state file as it is on disk today: `{"pid", "argv", "start_time"}`, pid 3574545 (dead — the launcher that exited), argv the single `services/stt.sh` path, start_time 858746 clock ticks. Criterion 5's compatibility row feeds exactly this shape to `supervisor.read_pid_file`/`stop`; the pid and start time of the *live* rows are substituted for a victim this test owns, because 3574545 belongs to nobody now and may be reused by the kernel at any time. |

Licence: this repo (Apache-2.0). No tokens, keys or credentials appear in either file; the
container id and the absolute path under `/home/user` are the only identifiers, and both are
already in `docs/tasks/T-0049.md`. `pid_file_t0033.json`: identifiers replaced 2026-10-25 (T-0157) — the login user in
the argv path → `user`; nothing else changed.

Re-record with (read-only, no `/launch` token — both are file reads):

```bash
grep -n 'Conflict. The container name' logs/stt.log            # find the line number N
sed -n "$((N)),$((N+2))p" logs/stt.log > tests/fixtures/services/stt_conflict.log
cp "$(git rev-parse --git-common-dir)/local_astra/run/stt.pid" \
   tests/fixtures/services/pid_file_t0033.json
```
