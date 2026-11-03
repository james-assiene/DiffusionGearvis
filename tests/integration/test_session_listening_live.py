"""T-0032 group B: one spoken question on the box (criteria 11, 13, 14).

A distinct module from T-0007's ``test_session_live.py`` so the two tasks never share a file.
Everything here needs the ``lean`` services up, a PipeWire capture *and* playback device, and a
**human to speak**, so each row skips unless it is explicitly opted into:

    ASTRA_LISTEN_RUN=<dir>   the directory a ``/launch``-reviewed rung run wrote (turns.jsonl,
                             stdout captured as ``rung.log``), produced by
                             ``session_rung.py --listen``

That split -- the run is performed under ``/launch``, the assertions are a test over its
artefacts -- is deliberate: a row that started a session itself would spend GPU time and open an
audio device on any marker-selected invocation, and CLAUDE.md puts that behind a review.

**Regime** (state it with every number, per utterance): profile ``lean``; vLLM v0.29.0 container
with ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` per ``services/versions.lock``, canvas 256,
steps 48, thinking on; STT ``kyutai/stt-1b-en_fr-candle @095e38f6`` in ``astra-moshi:0.6.4``,
GPU or ``ASTRA_STT_CPU=1`` -- **say which** (on CPU the step time is 100-108 ms per 80 ms of
audio, RTF ~ 1.3, so the end-of-turn delay grows with the utterance); Pocket TTS on CPU; headset
**or** speakers, stated, with the sink and source node names; driver 595.84; n = 3 utterances,
one speaker, the sentences quoted; git commit of the tree. **n = 3 is not B3** (50 turns,
docs/03 §7) and nothing derived here may be compared to it. This path has **no barge-in**: the
self-speech gate silences the STT tap while we speak, so SPEAKING -> PAUSED still cannot fire on
hardware (T-0034), and no row here claims otherwise.

Two facts from the T-0007a census that shape what the box can show (they change no assertion
here, and both belong in the report):

* the live server echoes a ``Marker`` **only while audio keeps arriving** -- census run 3 measured
  the echo 3.0 s after the mark on CPU STT -- so criterion 13's end-of-turn delay is dominated by
  the echo wait on that regime, and the capture pump must keep feeding after the end of turn;
* the server sends ``EndWord`` (T-0038), so until T-0038 merges a real word's ``end_ms`` equals
  its ``start_ms``: ``t_eot_ms`` on the box is the last word's *start*, which is a real
  under-estimate of the end of turn and must be said out loud beside any number here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from local_astra.telemetry.turns_log import TurnRecord

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.audio,
    pytest.mark.pending(task="T-0032"),
]

#: How many utterances the rung is driven with (criterion 11: 3 of 3).
UTTERANCES = 3


def _run_dir() -> Path:
    where = os.environ.get("ASTRA_LISTEN_RUN")
    if not where:
        pytest.skip(
            "T-0032 group B: set ASTRA_LISTEN_RUN to the directory of a /launch-reviewed "
            "'session_rung.py --listen' run (it needs the lean services, an audio device and a "
            "human to speak)"
        )
    path = Path(where)
    if not (path / "turns.jsonl").is_file():
        pytest.skip(f"{path}/turns.jsonl is missing: the rung wrote no turn")
    return path


def _records(run: Path) -> list[TurnRecord]:
    return [
        TurnRecord.from_line(line)
        for line in (run / "turns.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _log(run: Path) -> str:
    log = run / "rung.log"
    if not log.is_file():
        pytest.skip(f"{run}/rung.log is missing: the rung's stdout was not captured")
    return log.read_text()


def test_rung2_transcribes_and_answers_a_spoken_question():
    """Criterion 11: three spoken utterances, three transcripts that match, three answers.

    The transcript comparison is the rung's own line (``spoken=... transcript=...``); 3 of 3 must
    match after case folding and punctuation stripping (T-0007 criterion 16).
    """
    run = _run_dir()
    records = _records(run)
    assert len(records) == UTTERANCES, (
        f"criterion 11: one turns.jsonl line per utterance ({UTTERANCES}); got {len(records)}"
    )
    assert [r.outcome for r in records] == ["spoken"] * UTTERANCES, (
        f"...each of them answered aloud; outcomes were {[r.outcome for r in records]}"
    )
    pairs = re.findall(r"spoken=(.+?)\s+transcript=(.+)", _log(run))
    assert len(pairs) == UTTERANCES, (
        "criterion 11: the rung must print 'spoken=... transcript=...' per utterance so the match "
        f"is evidence and not a claim; found {pairs}"
    )

    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.strip().lower())

    mismatched = [(s, t) for s, t in pairs if norm(s) != norm(t)]
    assert not mismatched, (
        f"criterion 11: 3 of 3 transcripts must match; these did not: {mismatched}"
    )
    for record in records:
        assert record.regime.profile == "lean" and record.regime.git_commit, record.regime


def test_end_of_turn_delay_and_last_word_to_first_audio_are_recorded():
    """Criterion 13: both columns, per utterance, as their own numbers (docs/03 §7 B3's split).

    last-word audio time -> ``stt.end_of_turn`` submitted, and last word -> first audio. They are
    *recorded*, not thresholded: n = 3 is not a B3 result. On CPU STT the first column carries the
    ``Marker`` echo wait (3.0 s in census run 3), and the "last word" is a word whose ``end_ms``
    equals its ``start_ms`` until T-0038 lands ``EndWord``.
    """
    run = _run_dir()
    path = run / "listening.json"
    measured = json.loads(path.read_text()) if path.is_file() else None
    assert measured is not None, (
        "criterion 13: the rung must write listening.json with, per utterance, "
        "{'eot_delay_ms': ..., 'last_word_to_first_audio_ms': ...}"
    )
    assert len(measured) == UTTERANCES, measured
    for row in measured:
        for key in ("eot_delay_ms", "last_word_to_first_audio_ms"):
            assert isinstance(row.get(key), (int, float)) and row[key] >= 0, (
                f"criterion 13: {key} is a number per utterance; the row was {row}"
            )


def test_no_non_loopback_socket_while_listening():
    """Criterion 14 (docs/02 §9, docs/03 E8): one non-loopback socket fails the task.

    The rung samples ``ss -tunp`` for its own pid during the run and writes the lines it saw.
    """
    run = _run_dir()
    sockets = run / "sockets.txt"
    if not sockets.is_file():
        pytest.skip(f"{sockets} is missing: the rung did not sample the process's sockets")
    offending = [
        line
        for line in sockets.read_text().splitlines()
        if line.strip()
        and not line.startswith(("Netid", "State"))
        and "127.0.0.1" not in line
        and "[::1]" not in line
    ]
    assert not offending, f"criterion 14: non-loopback sockets while listening:\n{offending}"
