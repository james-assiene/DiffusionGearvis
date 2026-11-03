# tests/fixtures/prompts

Prompt text captured verbatim from the tree, for tests that need the text of a prompt **as it was
at a named commit** rather than as the code currently produces it.

## `ui_block_t0044.txt`

- **What:** `local_astra.conductor.ui_instructions.UI_INSTRUCTIONS` — the A2UI instruction block of
  the cached system prefix — as shipped by T-0044.
- **Commit:** `d12ddca15196590fdc6a531c2096ce6fa73ed63e` (`main`, 2026-09-16), the merge base of
  branch `T-0055-surface-in-reply`.
- **How:** `python -c "from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS as b;
  open('tests/fixtures/prompts/ui_block_t0044.txt','w').write(b)"`. Copied byte for byte: 1 779
  code points, 1 779 bytes (pure ASCII), **no trailing newline** — the constant has none, and the
  baseline arm of `tests/eval/test_surface_placement_live.py` sends the file's bytes unchanged.
- **sha256:** `2ec55d8df45d195f6508cedd8b6c6d9befaab402a6085bf710fb5dee91818e67`
- **Why it exists:** it is the block the model was given on 2026-09-16 when it twice built a
  complete A2UI surface *inside its reasoning channel* and then said aloud that it had created an
  interface (`logs/sessions/20260916-181929-3900425.jsonl`,
  `logs/sessions/20260916-215207-2034852.jsonl` turn `t1`). T-0055 rewrites the block; this file
  is the **baseline arm** it is measured against (criteria 13–14) and the independent oracle the
  fast suite uses to prove its wording lists are not vacuous
  (`tests/unit/test_ui_instructions.py::test_the_wording_lists_do_not_match_the_block_that_failed_twice`).
- **Never hand-edit it.** A drifted baseline makes the A/B arm worthless and silently weakens the
  anti-vacuity row: the whole point is that this text does *not* contain the placement rule. It is
  a historical record, not a copy of the current constant; it must not be regenerated when
  `ui_instructions.py` changes.
