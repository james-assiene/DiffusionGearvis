---
name: verify
description: Independent re-run of suites/benches for a branch on a clean tree by the verifier subagent; produces the PR evidence block. Use after review approval or before merging anything.
---

Usage: `/verify [task-id]`

Require `git status` clean. Spawn `verifier` with the task id and branch. Show its evidence block and the list of what was not run. Never merge from this skill. (This project-level `/verify` intentionally shadows Claude Code's bundled `/verify` skill.)
