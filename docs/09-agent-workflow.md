# Agent workflow for local_astra

How code gets written in this repository with Claude Code. This document is the protocol; `CLAUDE.md` is the short version the agent reads every session; `.claude/agents/`, `.claude/hooks/`, `.claude/skills/` and `.claude/rules/` implement it. The reasoning behind each element is in `docs/05-agentic-coding-review.md`; the process rules earned in `diffgemma-fa` are carried over deliberately.

## 1. Roles

Seven subagents, each with fresh context and a restricted tool set. The **planner** (read-only, plan mode) turns a request into `docs/tasks/<id>.md` with numbered acceptance criteria, owned files, the tests to write, the measurement regime and a blast radius. The **test-author** writes failing tests from that file and may edit `tests/` only. The **implementer** works in its own git worktree, may edit only the task's owned files, may not touch `tests/`, and must paste test output. The **reviewer** sees the diff, the task file and the tests, reviews both code and tests (including a quick mutation check for vacuous tests), and returns APPROVE / REVISE / ESCALATE, reporting only correctness, contract, budget and safety findings. The **verifier** re-runs everything on a clean tree, tries to refute the report, produces the PR evidence block, pushes and opens the PR; it never merges. The **launch-reviewer** reads any GPU- or artifact-writing command before it runs. The **explorer** answers codebase questions so the main session's context stays clean.

The main session is the orchestrator. It writes neither code nor tests. It relays findings between agents with `SendMessage` (keeping their context) and keeps `PROGRESS.md` current.

## 2. The loop

`/plan` → human yes/no on the acceptance criteria → new session or worktree → `/implement <id>`: test-author (tests fail for the right reason) → implementer (tests pass, evidence) → reviewer → route the verdict; after two REVISE rounds on the same finding, stop and involve the human → verifier → PR. Human review scaled to blast radius, then merge. Bugs found later go through `/triage`: failing test first, then an eval case, then a rule or hook if systemic, then the task/spec is updated so the stopping criteria include the bug (Ng's rule), then `/plan` the fix.

What the protocol is skipped for: typos, comments, renames, mechanical refactors with no behavioural change. When in doubt, it applies.

**Pending tests.** The scaffold ships acceptance tests ahead of their tasks, marked `pending(task="T-NNNN")` and run as strict xfail by `tests/conftest.py` (see docs/03 §1). In `/implement` step 3 the test-author's first move is to remove the markers of *its* task (and only its task), watch the tests fail for the right reason, then extend them; the fast suite is red on the task branch until the implementer is done, which is the point. `scripts/check_pending.py` lists the backlog; `--strict` is run by the orchestrator before the PR so no marker survives for a task in progress. Because the implementer cannot edit `tests/`, it can never "make green" by touching a marker.

**Which run is canonical (T-0031).** The serial `make fast` is the canonical fast-suite run: the verifier runs it before merge and it is the number a report quotes. The Stop hook runs the same selection in parallel for latency; it selects nothing the serial run does not.

**The prescribed red window (T-0019).** Between the test-author removing a task's markers and the implementer landing the code, that task's modules are red on `main` by design, often for hours. The Stop hook (`.claude/hooks/stop_fast_suite.sh`) therefore classifies every failing module against `docs/tasks/*.md`: a failure confined to a module named by a task whose `**Status:**` is `in progress` is *expected* — printed as `expected: T-NNNN (N failing in <module>, implementer pending)` so the red stays visible and attributed — and the turn may finish; any other failure (an unclaimed module, a `planned` task's module, a merged task's module) is a *regression*, listed first, and blocks. The rule this rests on: **the task file's status is the switch, never a marker in a test source** — so moving a task to `in progress` when its markers come off is part of activation, and `scripts/check_pending.py` warns when the two disagree in either direction.

## 3. Enforcement (hooks, not prompts)

All hooks source `_lib.sh`, which reads the role from the hook input's `agent_type` (settings hooks run inside subagents and carry it), the repo root from the hook input's `cwd` (because `$CLAUDE_PROJECT_DIR` stays at the main checkout for `isolation: worktree` subagents), the task id from the `T-NNNN` branch name, and keeps the launch token in the git common dir shared by all worktrees. Nothing is stored in tracked or untracked files under `.claude/`, so the enforcement is identical in the main checkout and in every worktree, and concurrent subagents cannot clobber each other's role. `protect_paths.sh` (PreToolUse on edits) enforces file ownership: implementer cannot edit `tests/`, `.claude/`, `CLAUDE.md`, the test plan, or files outside the task's *Owned files*, and must be on a task branch; test-author can edit `tests/` only; planner only `docs/tasks/`; reviewers/verifier/explorer are read-only except their own `.claude/agent-memory/<agent>/`; version locks are always protected. `guard_bash.sh` blocks destructive git (without false positives on branch names containing `-f`), test-silencing flags in either quote style, deleting or overwriting tests, implementer pushes, an implementer's shell writes (redirection, `tee`, `sed -i`, `cp`/`mv`/`rm`) to files outside its task's *Owned files* — read by the same parser `protect_paths.sh` uses, so a wrapped or annotated bullet parses and a "does not own" list never grants (T-0029) — and any GPU/artifact-*writing* command that lacks the single-use approval token from `/launch` or runs on a dirty tree (reads of `artifacts/` are allowed). `format_and_typecheck.sh` runs ruff and mypy after every Python edit — PostToolUse exit 2 is advisory, so the blocking mypy run is in `stop_fast_suite.sh`, which runs the fast suite (`uv run pytest tests/unit tests/contract -q -m "not gpu"`, with `-n 12 --dist loadfile` so the gate answers in ≤ 60 s; the serial ceiling is 120 s, T-0031) and package mypy when a session or an implementer that changed Python files (tracked or untracked) tries to finish. `require_evidence.sh` refuses an implementer/test-author/verifier report whose `last_assistant_message` has no pytest command and summary line. `settings.json` also denies `pip install`, `sudo`, force pushes and edits to lock files outright (`Edit(...)` rules only — `Write(...)` path rules are ignored by Claude Code), and sets `worktree.baseRef: head` so subagent worktrees start from the task branch rather than the default branch; `/implement` commits the tests before spawning the implementer because worktrees see committed files only. Per-agent `hooks:` frontmatter exists and could carry the same scripts, but the settings-level hooks with `agent_type` matching are simpler to keep consistent.

## 4. Parallelism and state

Parallel work is by worktree with disjoint owned files; the planner refuses to create overlapping tasks and the orchestrator checks `PROGRESS.md` before `/implement`. Two or three implementers at a time is the practical ceiling; agent teams are not used except for parallel review of a high-blast-radius change. Long-running state lives outside any context window: `PROGRESS.md` (Planned / In progress / In review / Done / Runs), `docs/tasks/*.md`, commits after each green task, and `turns.jsonl` artifacts with their commit hash and regime.

## 5. Measurement discipline

A reviewer reads every GPU- or artifact-writing command before it runs (`/launch`). The tree must be clean; nobody edits `local_astra/` while a measurement is running. Every reported number carries its whole regime (profile, canvas length, steps, model and template versions, headset or speakers, n, seed, commit). A number that contradicts an invariant or a proof means the measurement is wrong until shown otherwise. Two runs agreeing with the same seed is not replication. Prefer a coded, falsifiable hard stop early in a long run to a silent completion.

## 6. Context hygiene

CLAUDE.md stays under ~100 lines; anything longer goes to `.claude/rules/` (path-scoped) or a skill. One task per session; `/clear` after two failed corrections. The orchestrator passes paths, not contents. Exploration goes to the explorer subagent. Skills that have side effects (`/plan`, `/implement`, `/launch`, `/spec`) are user-invoked only.

## 7. PR template

Title: `<task-id>: <goal in one line>`. Body: link to the task file; acceptance criteria as a checklist with the test that covers each; the verifier's evidence block (commands, trimmed output, bench numbers with regime); what was not run and why; blast radius; reviewer verdict(s). Footer per the repository's attribution rules. A PR without the evidence block is not reviewable.

## 8. Task file template

```markdown
# T-0001 — <goal in one line>
## Goal
## Acceptance criteria
1. … (cites I3 / B4 / §4 invariant 2)
2. … (negative: must not change …)
## Owned files
- local_astra/conductor/speech_gate.py
- tests/unit/test_speech_gate.py
## Tests to write
- tests/unit/test_speech_gate.py::test_holds_unfinished_clause — unit — asserts …
- adversarial: …
## Implementation notes
## Regime
profile=lean canvas=256 steps=48 versions=services/versions.lock@<hash> n= seed=
## Blast radius
low | medium | high
```

## 9. Bootstrapping order for the repo

1. **Done (2026-09-12):** `CLAUDE.md`, `.claude/`, `docs/`, `PROGRESS.md`, `pyproject.toml` + `uv.lock`, the package skeleton with every planned module present as a stub (so pending tests import), the decided contracts as Pydantic models, the A2UI validator, the fakes package, task files T-0002/3/4/5/9/10/11/13, and the first suite (277 active tests, 67 pending). `make check` is green.
2. First tasks through the loop, in this order: T-0003 turn machine (its pending table and Hypothesis invariants are the spec), T-0004 speech gate, T-0013 filler policy, T-0005 LLM client + prompt builder against the fake, T-0001 fakes as replayers (needs Phase 0 recordings), then the Phase 0 bench harness (B1/B2/B7) gated by `/launch`.
3. Only then GPU work on the target box (X1–X13; T-0011 plugin milestone 0 once X3 has confirmed the sampler state field names).
