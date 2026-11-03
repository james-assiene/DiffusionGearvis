#!/usr/bin/env python3
"""Rung 1 of the manual-testing ladder: ask DiffusionGemma about a picture, by hand.

Deliberately NOT built on local_astra/. It talks to the engine directly, so it works today,
while the conductor is still being written, and so that a defect here is a defect in the
engine or the prompt rather than in machinery under review. Swap it onto conductor.llm_client
once T-0005 has been through review — at that point this becomes a way to exercise production
code by hand, which is worth more than a second parallel path.

    uv run python scripts/demo/ask.py "what is on this screen?" --image shot.png
    uv run python scripts/demo/ask.py "explain this error" --screen
    uv run python scripts/demo/ask.py "hello, who are you?"           # text only
    uv run python scripts/demo/ask.py "read it" --image a.png --think # show the thinking

Spends GPU time on an engine somebody else already started, and produces no number anyone
will quote, so it lives under scripts/demo/ and is deliberately ungated (T-0016 ruling D5).
Harnesses that produce measurements go in scripts/gpu/, which needs a /launch token per run.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx


def step(msg: str) -> None:
    """Say what is happening, on stderr, immediately. A local pipeline has several seconds of
    dead air in it (recording, CPU transcription, denoising, synthesis) and silence is
    indistinguishable from a hang."""
    print(f"\033[36m▸\033[0m {msg}", file=sys.stderr, flush=True)


API = "http://127.0.0.1:8001"
TTS = "http://127.0.0.1:8081"

# Sentence end: .!? or a closing quote/bracket after one, followed by whitespace or end of string.
# Crude on purpose — the real one is T-0004's speech gate, which also handles abbreviations,
# decimals and unclosed brackets. This is a spike, and a wrong split here costs an odd pause,
# not a wrong answer.
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]?(?=\s|$)')


class Speaker:
    """Speaks complete sentences through Pocket TTS, one at a time, off the streaming thread.

    Mirrors the rule the real conductor must obey (docs/01, CLAUDE.md): only text the model has
    actually emitted as an ANSWER is ever voiced. Reasoning deltas never reach here, and a partial
    sentence is held until it is terminated — a diffusion canvas can still be revised, so speaking
    an unterminated clause risks voicing something the model then changes.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.q: queue.Queue[str | None] = queue.Queue()
        self.failed: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            text = self.q.get()
            if text is None:
                return
            try:
                r = httpx.post(f"{self.url}/tts", data={"text": text}, timeout=120)
                if r.status_code != 200:
                    self.failed = f"TTS said {r.status_code}: {r.text[:200]}"
                    continue
                # aplay reads the wav header itself; blocking here is what serialises playback.
                subprocess.run(["aplay", "-q", "-"], input=r.content, check=False)
            except Exception as e:  # a dead TTS must not take the answer down with it
                self.failed = f"TTS unreachable at {self.url}: {e}"

    def feed(self, sentence: str) -> None:
        if sentence.strip():
            self.q.put(sentence.strip())

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=120)


def grab_screen(max_width: int = 1920) -> Path:
    """Screenshot the desktop, downscaled. Says what it tried and what each attempt did.

    mss first: it reads the X11 framebuffer from Python, needs no system package and no session
    bus. The CLI tools are fallbacks, and they are not interchangeable — measured on this box,
    `gnome-screenshot -f PATH` flashes the screen, exits 0, and writes NOTHING, even to a fresh
    path (it delegates to GNOME Shell's D-Bus API, which denies callers outside a blessed
    session: "Screenshot is not allowed"). So a zero exit status proves nothing here and every
    attempt is verified by looking at the file.
    """
    out = Path(tempfile.mkdtemp(prefix="astra-shot-")) / "screen.png"
    tried: list[str] = []

    try:
        import mss
        import mss.tools

        with mss.MSS() as sct:
            shot = sct.grab(sct.monitors[1])
            mss.tools.to_png(shot.rgb, shot.size, output=str(out))
        if out.exists() and out.stat().st_size > 0:
            return _downscale(out, max_width)
        tried.append("mss: produced no file")
    except ImportError:
        # mss is not a project dependency (adding one would touch uv.lock, which is a locked file
        # and its own task). uv can supply it for a single subprocess instead, so --screen works
        # from a plain `uv run` without anyone having to remember a flag.
        snippet = (
            "import sys, mss, mss.tools\n"
            "with mss.MSS() as s:\n"
            "    g = s.grab(s.monitors[1])\n"
            "    mss.tools.to_png(g.rgb, g.size, output=sys.argv[1])\n"
        )
        r = subprocess.run(
            ["uv", "run", "--quiet", "--with", "mss", "python", "-c", snippet, str(out)],
            capture_output=True,
        )
        if out.exists() and out.stat().st_size > 0:
            return _downscale(out, max_width)
        tried.append(f"mss via uv: exit {r.returncode} {(r.stderr or b'').decode()[:120].strip()}")
    except Exception as e:
        tried.append(f"mss: {type(e).__name__}: {e}")

    for tool, argv in (
        ("scrot", ["scrot", "-o", str(out)]),
        ("import", ["import", "-window", "root", str(out)]),
        ("spectacle", ["spectacle", "-b", "-n", "-o", str(out)]),
        ("grim", ["grim", str(out)]),
        ("gnome-screenshot", ["gnome-screenshot", "-f", str(out)]),
    ):
        if not shutil.which(tool):
            tried.append(f"{tool}: not installed")
            continue
        r = subprocess.run(argv, capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return _downscale(out, max_width)
        err = (r.stderr or b"").decode()[:120].strip()
        tried.append(
            f"{tool}: ran (exit {r.returncode}) but wrote no file{' — ' + err if err else ''}"
        )

    raise SystemExit(
        "could not capture the screen. Attempts:\n  "
        + "\n  ".join(tried)
        + "\n\nFix with either:\n"
        "  uv run --with mss python scripts/demo/ask.py ... --screen   (no install)\n"
        "  sudo apt install -y scrot                                  (then plain uv run)\n"
        "Or pass --image with a file instead."
    )


def _downscale(path: Path, max_width: int) -> Path:
    """A 4K grab is ~1.6 MB of PNG and base64 inflates it by a third, for no extra detail:
    the vision tower rescales to its own buckets anyway. Shrinking here cuts upload time."""
    try:
        from PIL import Image

        im = Image.open(path)
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
            im.save(path)
    except Exception:
        pass  # a big screenshot still works; this is only a size optimisation
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Ask the local model about an image.")
    p.add_argument("question", nargs="?", help="what to ask (omit with --listen)")
    p.add_argument("--listen", action="store_true", help="speak the question instead (Kyutai STT)")
    p.add_argument("--listen-seconds", type=float, default=15.0, help="max recording length")
    p.add_argument("--image", type=Path, help="image file to look at")
    p.add_argument("--screen", action="store_true", help="grab the screen instead")
    p.add_argument("--think", action="store_true", help="print the thinking, not just the answer")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--speak", action="store_true", help="speak the answer via Pocket TTS (CPU)")
    p.add_argument("--tts", default=TTS, help=f"Pocket TTS server (default {TTS})")
    args = p.parse_args()

    if args.listen:
        step(f"listening (up to {args.listen_seconds:.0f}s) — speak, then pause")
        # Same trick as mss: uv supplies websockets/msgpack for one subprocess, so ask.py needs
        # no dependency of its own and uv.lock stays untouched. stdout is exactly the transcript.
        helper = Path(__file__).with_name("_listen.py")
        r = subprocess.run(
            [
                "uv",
                "run",
                "--quiet",
                "--with",
                "websockets",
                "--with",
                "msgpack",
                "python",
                str(helper),
                "--max-seconds",
                str(args.listen_seconds),
            ],
            # stdout is PIPEd because it carries the transcript. stderr is deliberately NOT
            # captured, so the helper's progress shows up live -- capture_output=True buffered it
            # until the subprocess exited, which made a working recorder look like a hung one.
            stdout=subprocess.PIPE,
            text=True,
        )
        heard = r.stdout.strip()
        if r.returncode != 0 or not heard:
            raise SystemExit(
                "nothing transcribed. Is STT up?  sg docker -c 'docker ps --filter name=astra-stt'\n"
                "Start it with: sg docker -c 'ASTRA_STT_CPU=1 bash services/stt.sh'"
            )
        step(f'transcribed: "{heard}"')
        args.question = heard
    if not args.question:
        raise SystemExit("give a question, or pass --listen to speak one")

    speaker = Speaker(args.tts) if args.speak else None

    if args.screen:
        step("capturing the screen")
    img: Path | None = grab_screen() if args.screen else args.image
    if img is not None and not img.exists():
        raise SystemExit(f"no such image: {img}")

    try:
        model = httpx.get(f"{API}/v1/models", timeout=5).json()["data"][0]["id"]
        step(f"asking {model.split('/')[-1]}")
    except Exception as e:
        raise SystemExit(f"engine not reachable on {API} — is it up?  ({e})") from e

    content: list[dict[str, Any]] = []
    if img is not None:
        b64 = base64.b64encode(img.read_bytes()).decode()
        kind = "jpeg" if img.suffix.lower() in {".jpg", ".jpeg"} else "png"
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{b64}"}}
        )
        print(
            f"\033[2m[looking at {img} ({img.stat().st_size // 1024} KiB)]\033[0m", file=sys.stderr
        )
    content.append({"type": "text", "text": args.question})

    body = {
        # Only the four fields the diffusion path accepts. temperature/seed/min_p are a hard 400.
        "model": model,
        "max_tokens": args.max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": content}],
    }

    t0 = time.monotonic()
    first: float | None = None
    in_think = False
    spoken_tail = ""
    try:
        with httpx.stream("POST", f"{API}/v1/chat/completions", json=body, timeout=300) as r:
            if r.status_code != 200:
                r.read()
                raise SystemExit(f"engine said {r.status_code}: {r.text[:400]}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta") or {}
                # the reasoning key spelling is unmeasured; accept both
                think = delta.get("reasoning") or delta.get("reasoning_content")
                answer = delta.get("content")
                if first is None and (think or answer):
                    first = time.monotonic() - t0
                if think and args.think:
                    if not in_think:
                        print("\033[2m", end="", flush=True)
                        in_think = True
                    print(think, end="", flush=True)
                if answer:
                    if in_think:
                        print("\033[0m\n", end="", flush=True)
                        in_think = False
                    print(answer, end="", flush=True)
                    if speaker is not None:
                        # Hold the tail until a sentence actually terminates: a diffusion canvas
                        # can be revised, so speaking an unfinished clause risks voicing text the
                        # model then changes. Same reason the real speech gate exists (T-0004).
                        spoken_tail += answer
                        while (m := _SENTENCE_END.search(spoken_tail)) is not None:
                            speaker.feed(spoken_tail[: m.end()])
                            spoken_tail = spoken_tail[m.end() :]
    except KeyboardInterrupt:
        print("\n\033[2m[interrupted]\033[0m", file=sys.stderr)
        return 130

    if speaker is not None:
        step("speaking the answer")
        # Whatever never terminated still gets voiced once the stream is done: at that point the
        # model has committed it and it cannot be revised.
        speaker.feed(spoken_tail)
        speaker.close()
        if speaker.failed:
            print(f"\033[31m[{speaker.failed}]\033[0m", file=sys.stderr)
            print(
                "\033[2m[start it with: uvx --index https://download.pytorch.org/whl/cpu "
                "pocket-tts serve --port 8081]\033[0m",
                file=sys.stderr,
            )
    if in_think:
        print("\033[0m", end="")
    if first is None:
        print(
            "\033[33m▸ the model returned no content.\033[0m Observed with terse instructions "
            '("say X and nothing else", "one word answer") -- the engine replies with '
            "finish_reason=stop and an empty message. Rephrase as a normal question, or use "
            "--think to see whether it reasoned and then emitted nothing.",
            file=sys.stderr,
        )
    total = time.monotonic() - t0
    ttft = f"{first * 1000:.0f} ms" if first is not None else "—"
    print(
        f"\n\033[2m[first output {ttft} · total {total:.1f} s · n=1, not a benchmark]\033[0m",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
