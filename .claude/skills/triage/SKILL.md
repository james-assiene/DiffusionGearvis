---
name: triage
description: Turn an escaped bug or a failing eval into a test, an eval case, and (if systemic) a rule or hook — Ng's "fix the spec too" loop. Use whenever a bug is found outside the test suite.
---

Usage: `/triage <bug description or failing eval id>`

1. Reproduce with a failing test first (spawn `test-author`; the test goes in the right layer — unit if the logic is pure, contract if a fake and a real service disagree, e2e if only the full loop shows it).
2. Add the case to `tests/eval/` if it is a behaviour regression (latency, duplex behaviour, answer quality).
3. Ask: could a rule or a hook have prevented it? If yes, draft one line for `.claude/rules/<area>.md` or a hook change and show the user; do not add it silently. If the bug is a spec gap, update the task file or design doc section and say which.
4. Then `/plan` the fix as a normal task. Do not fix in this session.
