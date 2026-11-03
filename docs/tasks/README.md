# docs/tasks/

One file per task, `T-NNNN.md`, written by the planner (`/plan`) and read by every role. The
hooks parse two things: the task id from the branch name (`T-NNNN-<slug>`) and the
`## Owned files` section (glob patterns the implementer may edit on that task).

Template:

```markdown
# T-NNNN — <title>

**Status:** planned | in progress | in review | done · **Blast radius:** low | medium | high · **Depends on:** T-…

## Goal
One paragraph: what exists after this task that did not before, and which doc section it implements.

## Acceptance criteria
Numbered, testable statements. Each becomes at least one test written by the test-author
*before* implementation. Pending tests already in `tests/` for this id are listed here and
activated (marker removed) in `/implement` step 3.

## Owned files
- local_astra/<module>/**
- (one glob per line; tests/ is never owned — the test-author owns it)

## Out of scope
What the implementer must not touch or decide.

## Evidence required
Commands whose output the report must contain (fast suite, mypy, a specific bench when relevant).
```

**Structured test claims (T-0147).** While a task is `in progress`, the Stop hook excuses a red
test module as `expected:` if the task claims it structurally (`scripts/check_pending.py::strong_claims`):
a `tests/**.py` path that is the first token of a bullet under `## Owned files`, before any heading
or not-owned marker line (the same rule as the grants parser, `.claude/hooks/_lib.sh owned-globs`),
or any `tests/**.py` path under `## Tests to write`, before any not-owned marker line. The marker
phrases are "does not own", "deliberately does not touch", "ownership conflicts" and "claims no "
(any case). Put a not-owned list under its own `## Not owned` heading, introduced by "This task
does not own the following (not owned)". A reference to another task's test module inside
`## Tests to write` goes after that phrase, or is written without the `tests/` prefix.
