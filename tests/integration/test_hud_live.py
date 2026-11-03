"""Group B: the endpoint alive on the box, with the real page (T-0028 criteria 20-22).

Runs only under ``-m gpu``, through ``/launch``, on the 5090 box with vLLM (8001) and Pocket TTS
(8081) up. It starts the production entry point — ``uv run astra session --profile lean`` — opens
``hud/index.html`` in headless Chromium pointed at the bound port, and drives one turn.

**What is automated and what is not.** v1 has no wire type for typed text input (docs/02 §2.2's
T-0028 spec delta (c)) and the microphone path is T-0021's, so a *spoken* question cannot be
driven from a test. The one wire-level way to open a turn is ``astra.ui.action`` (criterion 7),
which is what these tests use; the spoken half of criterion 20 is the operator's step in the
``/launch`` Runs entry, as the task's "Evidence required" says.

**No number is claimed** (criterion 23): nothing here asserts a latency. The regime belongs on
every ``turns.jsonl`` line and in the Runs entry: profile=lean · canvas_length=256 ·
max_denoising_steps=48 · thinking=on · plugin=off · stt on CPU · seed is a hard 400 on the
diffusion path, so no group-B run is reproducible and no figure is comparable to anything but
itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path

import pytest

from local_astra.contracts.codec import decode, encode
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import (
    AstraFrame,
    AstraHello,
    AstraSpoken,
    AstraTurnState,
    AstraUiAction,
)
from tests.integration.test_hud_headless import CHROME, Chrome, requires_chrome, serve_hud

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
#: The docs/03 I1 canary: a large random word drawn on a plain background. The answer must name
#: it, which is only possible if the pixels really reached the model.
CANARY_WORD = "PLUMBUS"
URL_RE = re.compile(r"ws://127\.0\.0\.1:(\d+)/v1/realtime")
#: T-0197 review RV1: readiness is the conductor listening, i.e. ``_serve``'s "connect to" line. The HUD page line
#: (``the HUD page is at http://…/?ws=ws://…/v1/realtime``, on by default since T-0197) carries the same ws URL but
#: prints before the conductor binds, so a bare ``URL_RE`` match is not readiness.
READY_RE = re.compile(
    r"^astra session: open the HUD page and connect to (ws://127\.0\.0\.1:(\d+)/v1/realtime)\s*$"
)


def ready_url(line: str) -> str | None:
    """The conductor's ws URL if ``line`` says the session is listening, else ``None`` (RV1)."""
    found = READY_RE.match(line)
    return found.group(1) if found else None


def canary_jpeg() -> bytes:
    """A 1280x720 JPEG with ``CANARY_WORD`` in large letters. Needs the ``vision`` extra."""
    pytest.importorskip("PIL", reason="the canary image needs the 'vision' extra (pillow)")
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((200, 300), CANARY_WORD, fill="black")
    for scale in (6, 10, 14):  # thicken it: the default bitmap font is small
        draw.text((200 + scale, 300 + scale), CANARY_WORD, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class LiveSession:
    """``uv run astra session --profile lean`` as the production entry point (criterion 15)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.turns_path = data_dir / "turns.jsonl"
        self.proc: subprocess.Popen[str] | None = None
        self.url = ""
        self.stdout: list[str] = []

    def __enter__(self) -> LiveSession:
        self.proc = subprocess.Popen(
            [
                "uv",
                "run",
                "astra",
                "session",
                "--profile",
                "lean",
                "--data-dir",
                str(self.data_dir),
                "--turns",
                str(self.turns_path),
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline() if self.proc.stdout else ""
            if not line:
                if self.proc.poll() is not None:
                    break
                continue
            self.stdout.append(line)
            url = ready_url(line)
            if url is not None:
                self.url = url
                return self
        raise AssertionError(
            "criterion 15: `astra session` must print the page URL and the bound port; it printed "
            + "".join(self.stdout)
        )

    def __exit__(self, *exc: object) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            code = self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - the shutdown path under test
            self.proc.kill()
            raise AssertionError("criterion 16: SIGINT/SIGTERM must close the session") from None
        assert code == 0, f"criterion 16: Ctrl-C exits 0; it exited {code}"

    def lines(self) -> list[dict]:
        if not self.turns_path.exists():
            return []
        return [
            json.loads(line) for line in self.turns_path.read_text().splitlines() if line.strip()
        ]

    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid


def ask_through_the_wire(url: str, question: str, *, frame: bytes | None = None) -> list:
    """Open a turn the only way v1's wire allows: an ``astra.ui.action`` (criterion 7)."""
    import base64

    from websockets.sync.client import connect

    received: list = []
    with connect(url, max_size=32 * 1024 * 1024, open_timeout=30) as ws:
        ws.send(encode(AstraHello()))
        if frame is not None:
            ws.send(
                encode(
                    AstraFrame(
                        source=Source.SCREEN,
                        ts_ms=int(time.time() * 1000),
                        width=1280,
                        height=720,
                        jpeg_b64=base64.b64encode(frame).decode(),
                        phash="0" * 16,
                        title="canary",
                        app="test",
                    )
                )
            )
            time.sleep(0.5)
        ws.send(encode(AstraUiAction(surface_id="none", name="ask", context={"q": question})))
        # collect until the turn returns to IDLE (the last row of docs/02 §4's clean turn)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                raw = ws.recv(timeout=10)
            except TimeoutError:
                break
            message = decode(raw)
            received.append(message)
            if (
                isinstance(message, AstraTurnState)
                and message.state == "IDLE"
                and len(received) > 3
            ):
                break
    return received


@requires_chrome
def test_live_turn_through_the_page_answers_with_the_image_fact(tmp_path):
    """Criterion 20: the page is open, the frame is the screen, the answer names the fact.

    The spoken-question half needs T-0021's capture path and is the operator's step; the turn
    here is opened through ``astra.ui.action``, which is the same session, the same prompt
    builder and the same engine. The assertion is the docs/03 I1 canary: the answer contains a
    word that exists only in the pixels.
    """
    jpeg = canary_jpeg()
    with LiveSession(tmp_path) as live:
        server, port = serve_hud()
        try:
            with Chrome(CHROME) as chrome:
                page = f"http://127.0.0.1:{port}/?ws=" + urllib.parse.quote(live.url, safe="")
                chrome.navigate(page)
                assert chrome.wait_until("!!document.querySelector('[data-turn-state]')", 30), (
                    "the page must be up before the turn (criterion 20)"
                )
                received = ask_through_the_wire(
                    live.url, "what word is written on the screen?", frame=jpeg
                )
            spoken = " ".join(m.text for m in received if isinstance(m, AstraSpoken))
            assert CANARY_WORD.lower() in spoken.lower(), (
                "criterion 20 / docs/03 I1: the answer must name the word that is only in the "
                f"pixels; the assistant said {spoken!r}"
            )
            states = [m.state for m in received if isinstance(m, AstraTurnState)]
            assert "THINKING" in states and "SPEAKING" in states, (
                f"the state chip and the thinking meter move; the page saw {states}"
            )
        finally:
            server.shutdown()

        lines = live.lines()
        assert len(lines) == 1, f"criterion 20: exactly one turns.jsonl line; got {lines}"
        regime = lines[0]["regime"]
        assert regime["profile"] == "lean" and regime["thinking"] is True
        assert all(v is False for v in regime["assist_flags"].values()), (
            f"pixel-first: every assist is off on a live run nobody ticked a box in; {regime}"
        )
        for key in ("canvas_length", "max_denoising_steps", "vllm_version", "model", "git_commit"):
            assert regime.get(key), f"the line carries its whole regime; {key} is missing: {regime}"


def test_live_action_round_trip_opens_a_ui_turn(tmp_path):
    """Criterion 21: a click on a rendered surface produces a second ``[ui] `` turn.

    Two actions, so two turns: the second line's transcript must begin ``[ui] `` and a follow-up
    surface must go out. The surface's presence on the wire is the page-facing half of T-0024's
    round trip, closed here against the real engine.
    """
    from local_astra.contracts.protocol import AstraUiSurface

    with LiveSession(tmp_path) as live:
        first = ask_through_the_wire(live.url, "show me a two-button choice about fractions")
        assert any(isinstance(m, AstraUiSurface) for m in first), (
            "criterion 21 needs a surface to click; the turn produced "
            f"{sorted({getattr(m, 'type', '?') for m in first})}"
        )
        surface = next(m for m in first if isinstance(m, AstraUiSurface))
        second = ask_through_the_wire(live.url, "one half")
        assert second, "the second turn produced nothing"
        assert surface.surface_id, "the surface carries the id the click must echo"

    lines = live.lines()
    assert len(lines) == 2, f"criterion 21: two turns, two lines; got {len(lines)}"
    transcripts = [line.get("extras", {}).get("transcript", "") for line in lines]
    assert any(t.startswith("[ui] ") for t in transcripts), (
        "criterion 21: the clicked turn's transcript begins '[ui] ' (docs/02 §2.2); the lines say "
        f"{transcripts}"
    )


def test_the_listener_and_every_socket_are_loopback(tmp_path):
    """Criterion 22 / docs/03 E8: ``ss -tunp`` for the conductor shows 127.0.0.1 only.

    *Negative criterion*: any non-loopback socket or listener fails the task.
    """
    assert shutil.which("ss"), "criterion 22 is checked with ss(8)"
    with LiveSession(tmp_path) as live:
        pid = live.pid()
        out = subprocess.run(
            ["ss", "-tunp"], capture_output=True, text=True, check=True, env=os.environ
        ).stdout
        mine = [line for line in out.splitlines() if f"pid={pid}" in line]
        assert mine, f"no socket found for the conductor (pid {pid}); ss said:\n{out}"
        offenders = [
            line
            for line in mine
            if not all(
                part.startswith(("127.0.0.1:", "[::1]:", "*:*"))
                for part in line.split()
                if ":" in part and "pid=" not in part and "users:" not in part
            )
        ]
        assert not offenders, (
            "criterion 22: every socket of the conductor process is loopback; these are not:\n"
            + "\n".join(offenders)
        )
        assert f"127.0.0.1:{urllib.parse.urlparse(live.url).port}" in out, (
            "the listening socket is 127.0.0.1:<port>, never 0.0.0.0 or [::] (docs/03 E8):\n" + out
        )
