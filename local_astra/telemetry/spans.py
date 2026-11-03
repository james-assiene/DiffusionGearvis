"""Span names. Every stage emits one with the shared ``turn_id``; the bench reader knows this list.

Adding a stage: add it here *and* to the ``turns.jsonl`` reader (``bench/``); the unit test
checks the list is unique and the reader accepts every name.
"""

from __future__ import annotations

from typing import Literal

SpanName = Literal[
    "audio.capture",
    "reflex.onset",
    "stt.first_word",
    "stt.last_word",
    "stt.end_of_turn",
    "owner.check",
    "llm.prefill",
    "llm.first_canvas",
    "llm.done",
    "ui.gate",
    "tts.first_audio",
    "playback.first_sample",
    "playback.stopped",
    "canvas.tap",
    "tool.repair",
    "tool.execute",
    "tool.reground",
]

SPAN_NAMES: tuple[str, ...] = (
    "audio.capture",
    "reflex.onset",
    "stt.first_word",
    "stt.last_word",
    "stt.end_of_turn",
    "owner.check",
    "llm.prefill",
    "llm.first_canvas",
    "llm.done",
    "ui.gate",
    "tts.first_audio",
    "playback.first_sample",
    "playback.stopped",
    "canvas.tap",
    # T-0092c criterion 7: the repair request (``ToolRepairStart`` handled to its ``LlmDone``/``LlmError``
    # submitted) and the dry run (``ToolExecute`` handled to the outcome), both on the session clock.
    "tool.repair",
    "tool.execute",
    # T-0131 B9: ReGround's pass-2 request (sent to its outcome) in the live execute stage, on the session clock;
    # step 1's, as ``tool.execute``. Only with ``reground = true``.
    "tool.reground",
)

#: B3 "last word → first audio" (docs/01 §8, docs/03 §7) is measured from the *audio end of the
#: last user word* (``stt.last_word``), so the 500–1000 ms end-of-turn decision is inside the
#: number, not hidden. ``stt.end_of_turn`` − ``stt.last_word`` is reported as its own column.
LATENCY_START_SPAN = "stt.last_word"
LATENCY_END_SPAN = "playback.first_sample"
EOT_DECISION_SPAN = "stt.end_of_turn"

#: Prometheus histogram names (docs/02 §8).
HISTOGRAMS: tuple[str, ...] = (
    "astra_last_word_to_first_audio_ms",
    "astra_barge_in_stop_ms",
    "astra_vram_bytes",
)
