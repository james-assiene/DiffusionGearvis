---
paths:
  - "local_astra/conductor/**"
  - "local_astra/reflex/**"
---

# Conductor and reflex lane

- `turn_machine.py` is pure. No `asyncio`, no `time`, no I/O, no logging side effects. Time is a `Tick` event. If you need "now", it arrives in the event.
- Commands are data (`local_astra/contracts/commands.py`); the machine never calls a client. Executors live in `session.py`.
- Only `Commit(segment="answer")` events from `llm_client` may feed `speech_gate`; `TtsSpeak.source` is the committed substring the §4 invariant checks, `TtsSpeak.text` the normalised rendering; thought segments and any canvas that ends inside an unclosed thought never reach TTS. The system prompt always instructs the model to close its thought before answering (multimodal prompts otherwise leave it open — report §10). The speech gate holds an unfinished clause until the next commit or `llm.done`.
- Tutoring sessions keep a compact `TutorState` (problem, current step, misconceptions, hints given, ladder level) re-injected each turn *after* the frame (query-dependent suffix); raw history is capped at three spoken turns.
- Barge-in truncates history to what was *played* (from TTS/playback timestamps), not to what was generated.
- History stores `(user_text, assistant_spoken_text)`; thinking content is never stored or replayed; images older than two turns are replaced by their caption line.
- **A subsystem that silently disables itself must publish that it did.** The session log disables itself on a write error, printed one
  line to stderr and told no one: the only publish was on the page's own click, so a page that stays connected — the normal case, a human at
  the desk — kept a ticked "Log this session" box over a session writing nothing, while a page that reconnected was told the truth. The
  server knew and said nothing, which is worse than the page-side failure the checkbox was designed against. Found 2026-09-16 by a reviewer
  filling the disk under a live session; no test noticed, because making the state claim healthy regardless of the writer's error left 182
  rows green. Any component that can turn itself off owes a published state change at the moment it does, and a row that denies it.
- Thinking **is** logged in the session log (`telemetry/session_log.py`, T-0043). The rule above is about the conversation history the model is shown and about what is spoken, not a ban on recording or showing what happened.
  - Thought canvases go into `logs/sessions/*.jsonl` verbatim.
  - Thought text never enters history, the model's prompt (any later request of the turn or of the session), `astra.spoken`, TTS, `turns.jsonl` or memory.
  - On the HUD, thought text appears in exactly two places:
    - the Thinking pane's denoising view, unchanged, which may show the per-position pieces of every real canvas, thought included (T-0159, James 2026-10-26);
    - each turn's "Model thoughts" disclosure in the conversation thread, placed right under that turn's answer and collapsed by default. It shows only the committed text of the turn's `segment == "thought"` commits, carried by `astra.thought` and by no other message type (T-0180, James 2026-10-27/28).
  - No other message type, pane or DOM region carries thought text.
  - Thought-looking text inside an `answer` commit (removed by the T-0117/T-0121/T-0124 cleaners) is not a thought, and is shown nowhere.
- Prompt layout is `[<|think|>][system][history: spoken text, aged frames → captions][<|turn>user\n<|image|>(frames of this turn, oldest first, `[screen N]` labels from the second on)] ‖ [screen state, when no image is attached][TutorState][recollections, each marked and dated][assists][transcript]` — everything up to the frame is the cached prefix warmed by the anticipatory prefill; memory hits and state come *after* the frame because they depend on the query. Thinking is a session setting, never per request. A frame whose perceptual hash equals the newest history frame still held as an image is not attached a second time (the cached image carries the pixels — dropping it would break both the prefix hit and the model's view); a changed frame is always appended (a hash already seen this turn emits nothing; at the cap the last slot is replaced) and never inserted before one already sent; image aging happens at `commit_history`, never at the next onset.
- At most one active generate request plus one anticipatory prefill plus the idle captioner per session (`max-num-seqs 3`); the captioner is aborted on speech onset and the prefill at end-of-turn. Each concurrently denoising request costs a ~2.5 GiB sampler transient that vLLM does not profile.
- Reflex-lane models (Silero, turn predictor, owner-voice check) run on CPU threads, never on the GPU, and never block the event loop for more than 20 ms (there is a test).
- Every stage emits an OpenTelemetry span with the shared `turn_id`; new stages add a span name to `telemetry/spans.py`, and the bench reader must know it.
- **A duration is never computed from two clocks.** Three live instances, so the rule is not abstract (T-0051, 2026-09-16): the
  end-of-turn backstop subtracted a word's *audio* end from the *step* clock, so 800 ms of nominal silence was about 290 ms of examined
  silence and a human mid-sentence was cut off; the turn record's `stt.end_of_turn` span pairs an audio time with a session time, so any
  duration read off it is meaningless; and `reflex.onset`/`playback.stopped` pair an audio time and a playback offset in the same field.
  The fake recogniser carried the same bug (ruling B19), which is why no test could see it. When two timestamps meet, say which clock
  each is on before subtracting, or do not subtract.
- **A recollection is marked as one wherever it is shown to the model** (T-0053, 2026-09-16). A session with no frame attached and three
  cross-session captions under a bare `[memory]` header produced "the provided screenshots of a calculus textbook" in the model's own
  thought and a spoken answer about a screen that was never shared. In the prompt a recollection and a perception were typographically
  identical; the model was told, not confused. Every recalled line carries its age and a marker, under a header that says these are
  things seen earlier and not the current screen, and a prompt with no image in it says so.
- **What is spoken is filtered where speech is produced, not only where it is classified** (T-0052, 2026-09-16). A string that is
  structurally interface data is refused by the speech gate regardless of which channel the client thought it was in — and refused by
  *conserving* it with an empty text, never by dropping it, because the turn machine polices that the segmenter's outputs rebuild its
  input and a dropped sentence made whole turns silent with no log line. Found because the client's fence flag was hardcoded false per
  chunk and Pocket TTS read 807 characters of A2UI JSON aloud.
- A parser of model text must survive an integer literal past `sys.get_int_max_str_digits()`, on every parse path including the fallback, and must never change the limit; json_repair returns such a literal as a string (T-0185).
