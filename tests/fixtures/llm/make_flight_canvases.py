"""Regenerate tests/fixtures/llm/flight_canvases.json from the session log (T-0052).

Usage: ``uv run python tests/fixtures/llm/make_flight_canvases.py tests/fixtures/llm/flight_canvases.json``
from the repository root. The log it reads is the human's desk session of 2026-09-16 and lives
under ``logs/`` (gitignored), so this script documents the derivation rather than reproducing it
on another box; every assertion in it is a check on the reconstruction, not on the fixture.
"""

import json
import pathlib
import sys

LOG = "logs/sessions/20260916-181929-3900425.jsonl"  # logs/ is gitignored; the log is on the box
OUT = pathlib.Path(sys.argv[1])
recs = [json.loads(line) for line in pathlib.Path(LOG).read_text().splitlines() if line]
by = {r["seq"]: r for r in recs}

thought0 = by[2151]["data"]["text"]
thought1 = by[2178]["data"]["text"]
ui1 = by[2179]["data"]["text"]
answer2 = by[2221]["data"]["text"]
u1 = by[2223]["data"]
u2 = by[2228]["data"]
gate = by[2181]["data"]

UI_OPEN = "<a2ui>"
json_tail = answer2[:750]
prose_tail = answer2[750:]
assert len(ui1) == 409 and len(answer2) == 902 and len(prose_tail) == 152
assert json_tail + prose_tail == answer2
assert "<a2ui>" not in answer2 and "</a2ui>" not in answer2
assert len(u1["text"]) == 807 and len(u1["source"]) == 810

fixture = {
    "note": (
        "Turn t6 of the human's desk session of 2026-09-16 18:19-18:21 local, reconstructed from "
        "logs/sessions/20260916-181929-3900425.jsonl records seq 2151 / 2178 / 2179 / 2221 "
        "(llm.commit), 2181 (ui.gate) and 2223 / 2228 (tts.utterance). RECONSTRUCTION RULE: the "
        "log stores *post-split* pieces, not the raw SSE deltas, so the raw content delta of "
        "canvas 1 is UI_OPEN ('<a2ui>') + the 409-character ui piece, and the raw content delta of "
        "canvas 2 is the 902-character answer piece verbatim (_strip_markup removed nothing from "
        "it: the text contains neither '<a2ui>' nor '</a2ui>'). Reasoning deltas are the thought "
        "pieces. The model never wrote a closing fence; 'json_tail_canvas2' / 'prose_tail_canvas2' "
        "are the hand-measured split of canvas 2 at the structural end of the JSON (bracket depth "
        "back to zero outside a string, trailing run of closers consumed, remainder not resuming "
        "with '{'), which is character 750 of canvas 2 and character 1159 of the fenced body. "
        "'tts_utterance_*' is what Pocket TTS actually said: the negative oracle for the voice "
        "guard (T-0052 criterion 8) - no test may ever allow that string to be spoken."
    ),
    "provenance": {
        "session_log": "logs/sessions/20260916-181929-3900425.jsonl",
        "records": [2151, 2178, 2179, 2181, 2221, 2223, 2228],
        "session_id": "970ae08ab809",
        "turn_id": "t6",
        "request_id": "t6-gen",
        "git_commit": "40957df",
        "recorded_at": "2026-09-16T18:19:29+01:00",
        "request": "a flight booking interface (voice input)",
        "licence": "Apache-2.0 (this repo); the model output is the user's own session",
    },
    "regime": {
        "profile": "lean",
        "canvas_length": 256,
        "max_denoising_steps": 48,
        "thinking": True,
        "vllm_container": "vllm/vllm-openai:v0.29.0",
        "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "stt": "kyutai/stt-1b-en_fr",
        "tts": "pocket-tts@3.1.0",
        "audio_path": "speakers",
        "assists": "pixels only (all four flags false)",
        "seed": None,
        "input": "voice",
        "n": 1,
    },
    "canvases": [
        {"canvas_idx": 0, "reasoning": thought0, "content": None},
        {"canvas_idx": 1, "reasoning": thought1, "content": UI_OPEN + ui1},
        {"canvas_idx": 2, "reasoning": None, "content": answer2},
    ],
    "recorded": {
        "canvas_chars": [len(thought0), len(thought1) + len(ui1), len(answer2)],
        "closing_fence_emitted": False,
        "ui_piece_canvas1": ui1,
        "answer_piece_canvas2": answer2,
        "json_tail_canvas2": json_tail,
        "prose_tail_canvas2": prose_tail,
        "fenced_body_chars": len(ui1) + len(json_tail),
        "ui_gate_canvas1": {
            "repairs": gate["repairs"],
            "resamples": gate["resamples"],
            "surface_emitted": gate["surface_emitted"],
            "raw_len": gate["raw_len"],
            "ok": gate["ok"],
        },
        "tts_utterance_text": u1["text"],
        "tts_utterance_source": u1["source"],
        "tts_second_utterance_text": u2["text"],
        "rendered_only_after_reassembly": ["row1", "row2", "row3", "divider", "results"],
        "rendered_from_canvas_1_alone": ["root", "header"],
        "leaf_inputs_the_schema_rejects": ["date1", "date2", "passengers", "search"],
        "ui_gate_reassembled": {
            "repairs": 22,
            "resamples": 3,
            "surface_emitted": True,
            "raw_len": 1159,
            "ok": True,
        },
        "gate_note": (
            "The five ids in 'rendered_only_after_reassembly' are the components the real UiGate renders from the reassembled 1159-character body and does NOT render from canvas 1 alone: canvas 1 alone emits exactly root and header (repairs 4, resample ('row1',)), the reassembled body emits root, header, row1, row2, row3, divider, results (repairs 22, resample ('src', 'dest', 'date1')). Two components to seven is the measured win. The four ids in 'leaf_inputs_the_schema_rejects' never render in either run: they are leaf inputs written without the properties the A2UI v0.9.1 catalog requires - date1 is literally {\"id\":\"date1\",\"component\":\"DateTimeInput\"} with no properties at all - so the cause is a MISSING REQUIRED property, not a rejected one, and they leave through the resample path rather than the render path. That is a prompt / T-0010 matter, not something T-0052 may fix: the task forbids touching ui_gate.py. Recorded separately as T-0052 ruling A7: six leaf components fail the schema and UiGate.max_resamples is 3, so the gate asked for three and silently did not ask for the other three."
        ),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, OUT.stat().st_size, "bytes")
