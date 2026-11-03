# tests/fixtures/memory — provenance

Fixtures for **T-0053** ("a memory is never presented as something the assistant can see").

## `calculus_session.json` — the origin turn

* **Source:** `logs/sessions/20260916-202852-987340.jsonl`, record `seq 697`, `kind llm.request`,
  `turn_id t1` — the prompt that reached the engine on the human's desk at
  `2026-09-16T20:29:42.231230+01:00`. The three captions and the transcript
  (`"Make a flat looking UI."`) are **verbatim** from that record's `messages[1].content`; the
  record carries two text parts and **zero image parts**, which is the bug.
* **Generator:** `tests/fixtures/memory/make_calculus_session.py` (`--check` verifies in place).
* **`ts_ms`, `db_id`, `phash`, `thumbnail_name`:** the log record holds the *rendered* block only —
  there is no `memory.search` tap yet (that is T-0053 criterion 23, group C) — so these are looked
  up **by exact caption match** in the human's own store,
  `~/.local/share/local_astra/memory.sqlite` (opened read-only). The rows found were ids
  `7, 4, 11`, recorded in the generator so a rebuild that matched different rows fails loudly.
* **Known oddity, recorded rather than repaired:** those `ts_ms` values (`57858`, `1233272`,
  `13903`) are **session-relative milliseconds, not epoch** — the whole store is. That is a
  separate defect from the one T-0053 fixes (it is what makes "an hour ago" unrenderable in
  production even after group C wires the clock) and it is reported as a finding, not silently
  corrected here. No test in this task asserts a *particular* age for these three episodes: the
  origin rows use `now_ms=None`, whose rendering is criterion 5.
* **Personal data:** the captions describe a public textbook page; no `thumbnail_path` outside the
  basename is copied in, and no image bytes are included.

## `recall_corpus.json` — 40 hand-written captions + 360 filler

* **Generator:** `tests/fixtures/memory/make_recall_corpus.py` (`--check` verifies in place). The
  40 captions are invented for this task; the 360 filler captions are produced deterministically
  from the vocabulary lists in that script. No user data.
* **Constraints the generator asserts** (they are what makes the document frequencies
  hand-countable, and they are the point of the fixture):
  * every caption contains the standalone word `a`, so the rejected query `"Make a flat looking
    UI."` matches **every** episode on `a` alone at every corpus size;
  * no caption contains `make`, `flat`, `look`, `ui` in any form — so `a` is the *only* term that
    query can match;
  * `flight` occurs in exactly one caption, `glasses` in exactly one, and neither caption contains
    any other word of its query, so `"what time was that flight"` and `"where did I leave my
    glasses"` match on a **single content term**;
  * `desk` occurs in exactly three captions and those three are `small_subset`.
* **Hand-counted document frequencies** (confirmed against `fts5vocab` on this box, sqlite 3.53.1,
  2026-09-16):

  | term | small subset (n=3) | 40 | 400 |
  |------|-----|-----|-----|
  | `a` | 3/3 | 40/40 | 400/400 |
  | `desk` | 3/3 | 3/40 | 3/400 |
  | `flight` | 1/3 | 1/40 | 1/400 |
  | `glass` (stem of `glasses`) | 1/3 | 1/40 | 1/400 |

* **Measured pre-change behaviour** (`SqliteMemory` at `d12ddca`, `MemoryQuery(k=3)`, n=1 per
  query, fresh store per corpus, no second writer): `"Make a flat looking UI."` returns **3 hits at
  every size**, scores `1.6e-06…1.7e-06`, `matched=('a',)`; `desk` returns its 3 episodes at
  scores `1.1e-06` (n=3) and `4.49` (n=400) — the same three rows, four million times apart in
  score, which is why the rule must be document-frequency based and not a score floor.

## Which tests read what

* `tests/unit/test_prompt_recollection.py` — `calculus_session.json` (criterion 1).
* `tests/unit/test_memory_selectivity.py` — both files (criteria 12–14).
* `tests/contract/test_memory_recall_contract.py` — both files (criteria 12, 13, 18).
* `tests/eval/test_recollection_is_not_perception.py` — `calculus_session.json` seeds the scratch
  store for criterion 24 (`gpu`, `/launch`).
