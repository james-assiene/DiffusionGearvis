---
name: implement
description: Run the tests-first loop for one task: test-author writes failing tests, implementer makes them pass in a worktree, reviewer reviews both, iterate until APPROVE, then verifier. Use after /plan.
disable-model-invocation: true
---

Usage: `/implement <task-id>`

You are the orchestrator. You do not write code or tests yourself; you relay between agents and keep `PROGRESS.md` current. Keep your own context small: pass file paths, not file contents. Roles are enforced by hooks from each subagent's `agent_type`; the task is read from the `T-NNNN` branch name — no state files.

1. Read `docs/tasks/<id>.md`. Confirm no task in `PROGRESS.md` *In progress* owns any of the same files; if one does, stop.
2. Create the task branch in the main checkout: `git switch -c <id>-<slug>` (the worktree hooks parse `T-NNNN` from the branch name).
3. **Test-author**: spawn `test-author` (give it `name: test-author-<id>` so it can be resumed) with the task id. Its first job is to **activate the task's pending tests**: remove every `pytest.mark.pending(task="<id>")` marker (`uv run python scripts/check_pending.py` lists them), then extend the tests to cover every acceptance criterion. Require its report to show the activated and new tests failing for the right reason (`NotImplementedError` from a stub counts; an import error does not). If a criterion is untestable, go back to `/plan`. **Commit the tests on the task branch before step 4** — subagent worktrees only see committed files. The fast suite is red on the task branch from here until the implementer is done; that is expected.
4. **Implementer**: spawn `implementer` (`name: implementer-<id>`; it runs in its own worktree branched from HEAD, i.e. from the task branch with the tests). Its first command is `git switch -c <id>-impl` if it is not already on a `T-NNNN` branch. Require test command + output; it commits on its branch and never pushes.
5. **Reviewer**: spawn `reviewer` with the task id and the implementer's branch. It reviews both the implementation and the tests.
6. Route the verdict: `REVISE (implementer)` → `SendMessage` to `implementer-<id>` with the findings (keeps its context and worktree); `REVISE (test-author)` → same for `test-author-<id>`, then re-commit tests; `ESCALATE` → stop and show the user; `APPROVE` → step 7. After two REVISE rounds on the same finding, stop and show the user.
7. **Verifier**: spawn `verifier` with the branch. It re-runs everything on a clean tree, pushes, and opens the PR with the evidence block. On any discrepancy, return to step 6.
8. Run `uv run python scripts/check_pending.py --task <id>` (fails if any pending marker survives for this task, whatever the task file says). Update `PROGRESS.md` (*In review* with PR link). Tell the user what is in the PR and what was not run (e.g. GPU tests) in one short paragraph.

Blast radius *high* tasks (turn machine, audio path, GPU services, hooks) get a second reviewer spawn with the first reviewer's findings hidden. Note: `permissionMode: plan` on the planner is ignored when the main session runs in auto/acceptEdits mode; the planner's read-only status is still enforced by the hooks.
