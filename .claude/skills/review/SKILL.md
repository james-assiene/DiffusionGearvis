---
name: review
description: Adversarial review of the current branch's diff against its task file, by the reviewer subagent with fresh context. Use on any branch, including human-written ones.
---

Usage: `/review [task-id]`

Spawn `reviewer` with: the task id (or "no task file — review against CLAUDE.md rules and architecture contracts"), `git diff main...HEAD --stat`, and the instruction to report only correctness/contract/budget/safety findings. Present the verdict and findings verbatim. Do not fix anything in this session; open `/implement` or a new task for fixes.
