---
name: spec
description: Interview the user one question at a time to refine a new component or subproject into a spec section before planning tasks. Use for anything larger than one task (a new service, a decoding subproject).
disable-model-invocation: true
---

Usage: `/spec <topic>`

Ask one question at a time, starting from what is ambiguous in the existing docs for this topic (`docs/01`–`08`); stop when acceptance criteria could be written. Then write a spec section as a new `docs/specs/<topic>.md` (goal, interfaces, acceptance criteria, non-goals, measurement regime, risks) and show it. Start a fresh session for `/plan` afterwards — do not plan in the same context as the interview.
