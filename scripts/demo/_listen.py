#!/usr/bin/env python3
"""Mic (or a wav) -> Kyutai STT -> one line of transcript on stdout. Helper for ask.py --listen.

Run through uv with its own deps, so ask.py needs none of its own and uv.lock stays untouched:

    uv run --with websockets --with msgpack python scripts/demo/_listen.py [--max-seconds N]
    uv run --with websockets --with msgpack python scripts/demo/_listen.py --wav some.wav

Wire (kyutai-labs/delayed-streams-modeling, scripts/stt_from_mic_rust_server.py):
  ws://host/api/asr-streaming?auth_id=<token>, msgpack both ways
  out: {"type": "Audio", "pcm": [float32, ...]} at 24 kHz, 1920 frames (80 ms) per message
  in : {"type": "Word", "text": ...} and {"type": "Step", "prs": [...]}
       prs holds pause-probability heads for 0.5 / 1.0 / 2.0 / 3.0 s of silence.

End-of-turn thresholds prs[1] (the 1 s head) at 0.6 — Unmute's number, docs/01 §8. This is a
spike's approximation of what T-0003's ENDPOINTING row does properly: the real machine also waits
for the Marker confirming every word up to t_eot has been delivered, because the server lags audio
by ~0.5 s and cutting on the probability alone truncates the last word or two. Here we drain
briefly after the pause instead, which is cruder and occasionally loses a word.

--wav needs no converter: Pocket TTS emits 24 kHz mono 16-bit, which is exactly what STT wants,
so the same bytes go straight through. That makes TTS -> STT a closed loop testable with no human.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import wave
from collections.abc import Callable
from typing import Any

import msgpack
import websockets

RATE = 24_000
BLOCK = 1920  # 80 ms
STT = "ws://127.0.0.1:8080/api/asr-streaming?auth_id=public_token"


async def drain(ws: Any, words: list[str], loop: Any) -> None:
    """Collect trailing words until they stop arriving, then stop.

    The rule is "DRAIN_S since the last WORD", not "DRAIN_S since the drain began" and not
    "recv() timed out". Both of those were wrong in practice:
      - the server emits a Step every frame, so recv() never times out and `while True` over it
        spun for ever -- this is what hung the helper after it had heard the whole sentence;
      - a fixed window from the start cut the tail off, because on CPU the last words arrive
        seconds after the pause.
    A hard cap stops a talkative server keeping us here indefinitely.
    """
    last_word = loop.time()
    hard_cap = loop.time() + DRAIN_HARD_CAP_S
    while loop.time() < hard_cap and (loop.time() - last_word) < DRAIN_S:
        try:
            m = msgpack.unpackb(await asyncio.wait_for(ws.recv(), timeout=0.25))
        except TimeoutError:
            continue
        if m.get("type") == "Word":
            words.append(m["text"])
            last_word = loop.time()
            print(f"\033[2m{m['text']} \033[0m", end="", file=sys.stderr, flush=True)


def deliver(words: list[str]) -> None:
    """Emit the transcript and exit immediately, before any teardown runs.

    Measured: once the turn ends correctly, closing down reliably blocks -- the websocket close
    handshake does not complete while the server still has audio queued, and waiting it out cost
    10+ seconds of dead air after the user had already stopped speaking. There is nothing to
    clean up that the OS will not do better: the transcript is the entire product of this process.
    """
    text = " ".join(words).strip()
    if text:
        print(text)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if text else 2)


# How long to keep reading after the pause head fires. The server lags audio by ~0.5 s, so the
# last words of an utterance are still in flight when the pause is detected. Measured: 0.6 s
# dropped the final two words of "…and which line causes it", and on CPU a 2.3 s clip produced
# nothing at all within 1.5 s — decode is slower than the audio when vLLM has the machine busy.
# The correct fix is the Marker message, which confirms every word up to t_eot was delivered —
# that is what T-0003's ENDPOINTING row waits for, and why it does not trust the probability.
DRAIN_S = 2.5
# Absolute ceiling on draining, so a chatty server cannot hold the turn open.
DRAIN_HARD_CAP_S = 12.0
# Silence appended after a --wav ends, to advance the model's clock so the tail flushes.
TAIL_SILENCE_S = 3.0
# Our own end-of-turn: how long the input must stay below SILENCE_LEVEL, once at least one word
# has been transcribed, before we stop listening. The server's prs pause heads were unreliable
# here (they did not fire on a real utterance), and waiting on them left the recorder running
# until the hard deadline. Measuring the level we already compute is cruder but it is OURS and
# it is observable. The real machine uses stt.flushed + eot_settle_ms instead (T-0003 crit. 6).
SILENCE_S = 1.2
# A FIXED silence level does not work: measured in this room, ambient noise peaks around 0.116,
# so a 0.02 floor meant the "still talking" timer was refreshed by the room itself and the pause
# was never detected -- the recorder held a finished sentence until the hard deadline.
# The threshold is therefore relative to the loudest speech seen so far, with an absolute floor
# for a genuinely quiet room. 0.25 x a 0.6 peak lands at 0.15, comfortably above that noise.
SILENCE_FRACTION = 0.25
SILENCE_FLOOR = 0.02


def _wav_reader(path: str) -> Callable[[int], bytes]:
    """Read any audio file as 24 kHz mono 16-bit, resampling through ffmpeg when needed.

    Pocket TTS already emits exactly this, so its output needs no conversion. Anything else --
    a recording, a podcast clip, a different sample rate -- goes through ffmpeg, which is what
    makes the STT path testable with real-world audio instead of only our own synthesis.
    """
    try:
        w = wave.open(path, "rb")  # noqa: SIM115 — lifetime is the read callable's, not a block's
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2):
            return lambda n: w.readframes(n // 2)
        fmt = f"{w.getframerate()} Hz / {w.getnchannels()} ch / {w.getsampwidth() * 8} bit"
        w.close()
    except wave.Error:
        fmt = "not a PCM wav"

    if not shutil.which("ffmpeg"):
        raise SystemExit(
            f"{path} is {fmt}; this helper needs {RATE} Hz mono 16-bit.\n"
            "Install ffmpeg to have it converted automatically:  sudo apt install -y ffmpeg"
        )
    print(f"\033[2m  [{fmt} -> {RATE} Hz mono via ffmpeg]\033[0m", file=sys.stderr)
    conv = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
        stdout=subprocess.PIPE,
    )
    assert conv.stdout is not None
    return conv.stdout.read


async def listen(
    url: str,
    max_seconds: float,
    pause_threshold: float,
    wav: str | None,
    like_mic: bool = False,
    words: list[str] | None = None,
) -> str:
    # The caller owns `words` so that a hang ANYWHERE in teardown still leaves it with everything
    # transcribed so far. Measured: after the turn ended correctly the helper hung in cleanup --
    # most likely the websocket close handshake, which the server does not always complete while
    # audio is still queued -- and the transcript died with it. A spike's job is to hand back what
    # it heard, not to close politely.
    if words is None:
        words = []
    proc: subprocess.Popen[bytes] | None = None
    as_file = bool(wav) and not like_mic
    if wav:
        read = _wav_reader(wav)
    else:
        proc = subprocess.Popen(
            ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        # Non-blocking: an executor thread parked in a blocking read() cannot be cancelled, and
        # asyncio.run() waits for the default executor at shutdown -- which is exactly how this
        # helper came to transcribe correctly and then hang for ever.
        os.set_blocking(proc.stdout.fileno(), False)
        fd = proc.stdout

        def read(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                chunk = fd.read(n - len(buf))
                if chunk is None:  # nothing available yet
                    break
                if chunk == b"":  # recorder gone
                    break
                buf += chunk
            return buf

    peak = 0.0
    blocks = 0
    last_loud = 0.0
    try:
        async with websockets.connect(url, max_size=None) as ws:
            src = wav if wav else "microphone"
            print(f"\033[36m▸\033[0m connected to STT, reading {src}", file=sys.stderr, flush=True)
            if not wav:
                print(
                    "\033[36m▸\033[0m speak now, then stay quiet for a moment",
                    file=sys.stderr,
                    flush=True,
                )

            async def pump() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    raw = read(BLOCK * 2)
                    if proc is not None and len(raw) < BLOCK * 2:
                        if proc.poll() is None:
                            await asyncio.sleep(0.01)  # mic still running, just no data yet
                            continue
                        return
                    if not raw or len(raw) < BLOCK * 2:
                        # The model is a DELAYED stream: words come out on a ~0.5 s lag and its
                        # clock only advances while audio arrives. A file that stops dead leaves
                        # the final words unflushed for ever — measured, a 2.3 s clip transcribed
                        # to nothing at all. A live mic never shows this because the room keeps
                        # supplying silence. So pad the end with real silence frames.
                        if wav:
                            for _ in range(int(TAIL_SILENCE_S * RATE / BLOCK)):
                                await ws.send(
                                    msgpack.packb({"type": "Audio", "pcm": [0.0] * BLOCK})
                                )
                        return
                    pcm = [
                        int.from_bytes(raw[i : i + 2], "little", signed=True) / 32768.0
                        for i in range(0, len(raw), 2)
                    ]
                    nonlocal peak, blocks, last_loud
                    block_peak = max(abs(v) for v in pcm)
                    peak = max(peak, block_peak)
                    blocks += 1
                    if block_peak >= max(SILENCE_FLOOR, peak * SILENCE_FRACTION):
                        last_loud = loop.time()
                    # The meter stops as soon as transcription starts, so it never overwrites the
                    # words: they share stderr and \r would mangle them.
                    if not as_file and not words and blocks % 12 == 0:  # ~every 1 s
                        bar = "#" * min(20, int(block_peak * 60))
                        print(
                            f"\r\033[2m  level |{bar:<20}| peak {peak:.3f}\033[0m",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    await ws.send(msgpack.packb({"type": "Audio", "pcm": pcm}))
                    if like_mic:
                        await asyncio.sleep(BLOCK / RATE)  # 80 ms, real-time pace

            loop = asyncio.get_running_loop()
            last_loud = loop.time()
            sender = asyncio.create_task(pump())
            deadline = loop.time() + max_seconds
            try:
                while loop.time() < deadline:
                    # Short recv timeout so the end-of-turn check below still runs when the server
                    # goes quiet. Waiting the full max_seconds here meant silence could never be
                    # detected without an incoming message to trigger the check.
                    try:
                        msg = msgpack.unpackb(await asyncio.wait_for(ws.recv(), timeout=0.25))
                    except TimeoutError:
                        msg = {}
                    if msg.get("type") == "Word":
                        if not words:
                            print(
                                "\n\033[36m▸\033[0m hearing: ", end="", file=sys.stderr, flush=True
                            )
                        words.append(msg["text"])
                        print(f"\033[2m{msg['text']} \033[0m", end="", file=sys.stderr, flush=True)
                    if words and not as_file and (loop.time() - last_loud) > SILENCE_S:
                        print(
                            "\n\033[36m▸\033[0m pause detected, finishing up",
                            file=sys.stderr,
                            flush=True,
                        )
                        await drain(ws, words, loop)
                        deliver(words)
                    elif msg.get("type") == "Step" and words and not as_file:
                        # File input has a known end, so the pause head is the wrong signal for
                        # it: the whole wav is sent in milliseconds and the head fires while a
                        # CPU decode is still catching up, truncating the tail. For --wav we
                        # instead wait for the sender to finish and then drain (below).
                        prs = msg.get("prs") or []
                        # Only once something has been said: prs is high during the leading
                        # silence too, so thresholding earlier ends the turn immediately.
                        if len(prs) > 1 and prs[1] > pause_threshold:
                            await drain(ws, words, loop)
                            deliver(words)
                    if as_file and sender.done():
                        # All audio delivered: drain until the server goes quiet.
                        await drain(ws, words, loop)
                        deliver(words)
                # Deadline reached with the turn never ending (silence detection did not fire,
                # e.g. a room louder than the speech). Deliver anyway rather than teardown.
                deliver(words)
            except TimeoutError:
                pass
            finally:
                # Terminate the recorder BEFORE anything else. The pump's read() runs in an
                # executor thread; cancelling the task does not interrupt a thread blocked on a
                # pipe, and asyncio.run() waits for the default executor at shutdown. Leaving it
                # to the outer finally meant the helper transcribed correctly and then hung for
                # ever without printing the transcript.
                if proc is not None:
                    proc.terminate()
                sender.cancel()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("", file=sys.stderr)
    if not words:
        if peak < 0.01:
            print(
                f"\033[31m▸ the microphone was essentially silent (peak {peak:.4f}).\033[0m\n"
                "  Check the input device is unmuted and selected:  alsamixer  /  pavucontrol\n"
                "  Test it directly:  arecord -f S16_LE -r 24000 -c 1 -d 3 /tmp/t.wav && aplay /tmp/t.wav",
                file=sys.stderr,
            )
        else:
            print(
                f"\033[31m▸ audio reached STT (peak {peak:.3f}) but no words came back.\033[0m\n"
                "  STT is on CPU and slower than real time here; try speaking for longer,\n"
                "  or raise --max-seconds.",
                file=sys.stderr,
            )
    return " ".join(words).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=STT)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--pause-threshold", type=float, default=0.6)
    p.add_argument("--wav", help="transcribe this file instead of the mic (testing)")
    p.add_argument(
        "--like-mic",
        action="store_true",
        help="feed --wav through the microphone code path at real-time pace, so end-of-turn "
        "and the pause/drain/exit path are exercised without a human",
    )
    a = p.parse_args()
    words: list[str] = []

    async def run() -> str:
        # Hard ceiling on the whole exchange, so teardown can never outlast the listening budget.
        try:
            return await asyncio.wait_for(
                listen(a.url, a.max_seconds, a.pause_threshold, a.wav, a.like_mic, words),
                timeout=a.max_seconds + 3,
            )
        except TimeoutError:
            print("\n\033[2m[cleanup timed out; using what was heard]\033[0m", file=sys.stderr)
            return " ".join(words).strip()

    try:
        text = asyncio.run(run())
    except OSError as e:
        print(f"STT unreachable at {a.url}: {e}", file=sys.stderr)
        return 1
    if not text:
        print("heard nothing", file=sys.stderr)
        return 2
    print(text)  # stdout is the contract: exactly the transcript, nothing else
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard exit: the transcript is delivered and nothing else here matters. Returning normally
    # lets asyncio's shutdown wait on a socket or executor that has already proved it can block
    # for ever, which turned a successful transcription into a hang.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
