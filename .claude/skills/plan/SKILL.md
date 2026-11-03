---
name: plan
description: Turn a request into docs/tasks/<id>.md with acceptance criteria and owned files, using the planner subagent. Use before any non-trivial change.
disable-model-invocation: true
---

Usage: `/plan <short request>`

1. Pick the next task id: `ls docs/tasks | sort | tail -1` → increment (format `T-0042`).
2. Spawn the `planner` subagent with the request, the id, and the instruction to return the complete task file content. Do not paste large code into the prompt; the planner reads the repo itself.
3. Write the returned content to `docs/tasks/<id>.md` (the orchestrator writes it; the planner is read-only except for task files).
4. Show the user the acceptance criteria and owned files and ask for a yes/no before continuing. If the planner split the request into several tasks, present the order.
5. Add a line to `PROGRESS.md` under *Planned*.

Do not start implementation in this session. The next step is `/implement <id>` in a fresh session.
