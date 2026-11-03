"""T-0182 A2 contract: ``scripts/render_clips.py`` through the **real** ``PocketTtsClient`` (docs/tasks/T-0182.md).

The unit module (``tests/unit/test_t0182_render_clips.py``) replaces the client seam with a scripted client. Here the
seam holds the real ``conductor.tts_client.PocketTtsClient``, constructed from the ``Services`` the script passes,
with only its transport swapped for ``tests/fakes/pocket_tts_double.py``'s ``httpx.MockTransport`` (the double answers
``POST http://127.0.0.1:8081/tts`` without a socket). So the real RIFF walk, the placeholder ``data`` size and the
80 ms chunking all sit between the served wav and the asset.

- ``fake``: the double serves a hand-built wav (``make_wav`` with a ``LIST`` chunk before ``data`` and Pocket TTS
  3.1.0's 2 000 000 000-byte placeholder size), trickled in 1 000-byte pieces. The asset must be that wav's PCM, byte
  for byte, and the request must be one form POST of the clip text verbatim to 127.0.0.1:8081/tts.
- ``real`` (``gpu``-marked; CPU service, skipped unless ``GET {tts}/health`` answers 200): the same command against
  the running Pocket TTS, into ``tmp_path``. Its output is not deterministic (tests/fixtures/audio/MANIFEST.md), so
  the shared assertions are the ones both can meet: exit 0, one ``rendered`` line, an asset that passes A2's rules
  and a ``current`` status.
- A non-loopback TTS URL is refused before any connection: the real client's constructor raises on it, the script
  exits non-zero, and nothing is written. The premise (``Services`` validates the URL) is a separate row, green at
  BASE by design.

The render rows were pending at 5c4027d0 and are activated for ``/implement T-0182`` group A. Regime: CPU; the fake arm opens no
socket; files only under ``tmp_path``; n = 1, no seed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from local_astra.conductor.tts_client import PocketTtsClient
from local_astra.config import Services
from tests.fakes.pocket_tts_double import TtsDouble, chunked, make_wav, sine_pcm
from tests.unit import test_t0182_render_clips as _render
from tests.unit.test_t0182_clips_registry import BASE_REGISTRY, api, tts_pin
from tests.unit.test_t0182_render_clips import asset, files_under, no_sockets, outcome, run

#: The unit module's fixtures, registered here by assignment (``jail`` is autouse, so the real data dir is guarded).
jail = _render.jail
script = _render.script
base_registry = _render.base_registry

pytestmark = pytest.mark.contract
#: Pocket TTS 3.1.0's measured ``data`` placeholder (``tts_client.POCKET_TTS_PLACEHOLDER_DATA_SIZE``), restated.
PLACEHOLDER = 2_000_000_000


def _served_pcm() -> bytes:
    return sine_pcm(1200, freq=330.0, amplitude=0.3)


def _install_real(script: ModuleType, monkeypatch: pytest.MonkeyPatch, transport: Any) -> list[Any]:
    made: list[Any] = []

    def factory(services: Any = None, *args: Any, **kwargs: Any) -> PocketTtsClient:
        made.append(services)
        return PocketTtsClient(services, *args, transport=transport, **kwargs)

    monkeypatch.setattr(script, "PocketTtsClient", factory)
    return made


def _skip_unless_tts_is_up() -> None:
    try:
        health = httpx.get(f"{Services().tts}/health", timeout=2.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"pocket-tts is not up ({type(exc).__name__}: {exc})")
    if health.status_code != 200:
        pytest.skip(f"pocket-tts /health answered {health.status_code}")


@pytest.mark.parametrize("backend", ["fake", pytest.param("real", marks=pytest.mark.gpu)])
def test_render_through_the_real_client(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: ``--kinds ack`` through the real ``PocketTtsClient``: one ``rendered`` line, exit 0, a valid asset whose
    status is ``current``; with the double, the asset is the served PCM byte for byte and the POST is the clip text
    verbatim to 127.0.0.1:8081/tts."""
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--kinds", "ack"]
    if backend == "fake":
        pcm = _served_pcm()
        assert struct.unpack("<h", pcm[2:4])[0] != 0, "premise: the served audio is not silent"
        wav = make_wav(
            pcm,
            data_size=PLACEHOLDER,
            extra_chunks=((b"LIST", b"INFOISFT" + b"pocket-tts 3.1.0\x00"),),
        )
        double = TtsDouble.streaming(chunked(wav, 1000))
        made = _install_real(script, monkeypatch, double.transport)
        with no_sockets() as attempts:
            code, out = run(script, argv, capsys)
        assert attempts == []
    else:
        _skip_unless_tts_is_up()
        made = _install_real(script, monkeypatch, None)
        code, out = run(script, argv, capsys)
    assert code == 0, f"A2: exit {code}\n{out}"
    assert "rendered" in outcome(out, "ack")
    assert made and all(isinstance(s, Services) for s in made)
    body = asset(data, "ack").read_bytes()
    assert body and len(body) % 2 == 0 and not body.startswith(b"RIFF")
    assert 0.3 <= len(body) / 48_000 <= 6.0, f"A2: {len(body) / 48_000:.3f} s"
    values = struct.unpack(f"<{len(body) // 2}h", body)
    assert any(values) and max(values) < 32767 and min(values) > -32768
    assert api("status")(data, "ack", BASE_REGISTRY["ack"], tts_pin()) == "current"
    if backend == "fake":
        assert body == pcm, f"A2: the asset is not the served PCM ({len(body)} vs {len(pcm)} bytes)"
        assert double.texts() == [BASE_REGISTRY["ack"]], f"A2: posted {double.texts()!r}"
        url = double.requests[0].url
        assert (url.host, url.port, url.path, double.requests[0].method) == (
            "127.0.0.1",
            8081,
            "/tts",
            "POST",
        )


def test_a_non_loopback_tts_url_is_refused_before_any_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 network: the services the script passes are rewritten to a TEST-NET address (bypassing ``Services``'
    validator with ``model_copy``) and handed to the real client, which refuses them; the script exits non-zero,
    opens no socket and writes nothing."""
    made: list[Any] = []

    def factory(services: Any = None, *args: Any, **kwargs: Any) -> PocketTtsClient:
        made.append(services)
        assert isinstance(services, Services), (
            f"A2: PocketTtsClient(profile.services); got {services!r}"
        )
        off_box = services.model_copy(update={"tts": "http://192.0.2.7:8081"})
        return PocketTtsClient(off_box, *args, **kwargs)

    monkeypatch.setattr(script, "PocketTtsClient", factory)
    data = tmp_path / "data"
    with no_sockets() as attempts:
        try:
            code, out = run(script, ["--data-dir", str(data), "--kinds", "ack"], capsys)
        except ValueError as exc:
            code, out = 1, str(exc)
    assert made, "A2: the script never constructed its client"
    assert attempts == [], f"A2: a connection was attempted to {attempts}"
    assert code != 0, f"A2: a non-loopback TTS must not render; exit {code}\n{out}"
    assert not asset(data, "ack").exists()
    assert not data.exists() or [f for f in files_under(data) if f != "clips/manifest.json"] == []


def test_services_refuses_a_non_loopback_tts_url() -> None:
    """A2's premise ("``Services`` already validates its URL"), green at BASE by design: a profile cannot name an
    off-box TTS, and the client refuses one that slipped past the model."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="non-loopback"):
        Services(tts="http://192.0.2.7:8081")
    with pytest.raises(ValueError, match="non-loopback"):
        PocketTtsClient(Services().model_copy(update={"tts": "http://192.0.2.7:8081"}))
