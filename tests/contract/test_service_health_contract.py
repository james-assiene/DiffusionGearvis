"""The health contract the doctor probes, against the fake and against the real services (c. 3).

Same assertions both sides (``.claude/rules/tests.md``): the engine answers 200 on ``/health``;
Pocket TTS answers a body containing ``healthy``; moshi-server has no health route, so the http
form of the configured ``ws://…/api/asr-streaming`` with ``auth_id=public_token`` answers a **4xx**
and specifically **not** 404 — a 404 is a live process with no ASR route, which is the silent
failure this check exists for.

``backend="fake"`` runs the stubs in ``tests/fakes/http_service.py``; ``backend="real"`` is marked
``gpu`` and runs on the box through ``/launch`` against whatever ``astra.toml`` points at. Neither
side sends a request body or touches an inference endpoint, so this spends no GPU time
(``.claude/rules/gpu-services.md``: status polls are free).
"""

from __future__ import annotations

import socket
from urllib.parse import urlsplit

import pytest

from local_astra.config import load
from tests.fakes.http_service import StubService, stt_routes, tts_routes, vllm_routes
from tests.unit import _box_rig as rig

pytestmark = [pytest.mark.contract]


def check(name: str, url: str):
    return rig.attr(rig.module("supervisor"), "check_service")(name, url)


def reachable(url: str) -> bool:
    parts = urlsplit(url)
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def endpoints(backend: str):
    """``{name: url}`` in the same shape ``config.Services`` uses, fake or real."""
    if backend == "real":
        services = load(None).profiles["lean"].services
        urls = {"vllm": services.vllm, "stt": services.stt, "tts": services.tts}
        missing = [name for name, url in urls.items() if not reachable(url)]
        if missing:
            pytest.skip(f"real backend: {missing} are not listening (start them with `astra up`)")
        yield urls
        return

    stubs = [
        StubService(vllm_routes(), name="vllm"),
        StubService(stt_routes(), name="stt"),
        StubService(tts_routes(), name="tts"),
    ]
    engine, stt, tts = stubs
    try:
        yield {
            "vllm": f"{engine.base}/v1",
            "stt": f"ws://127.0.0.1:{stt.port}/api/asr-streaming",
            "tts": tts.base,
        }
    finally:
        for stub in stubs:
            stub.close()


def test_the_engine_answers_two_hundred_on_health(endpoints):
    result = check("vllm", endpoints["vllm"])
    assert result.status_code == 200, f"criterion 3: GET /health -> 200; got {result}"
    assert result.healthy and result.reason is None, f"got {result}"


def test_the_tts_health_body_says_healthy(endpoints):
    result = check("tts", endpoints["tts"])
    assert "healthy" in result.body, (
        f"criterion 3: the TTS health body contains `healthy`; it was {result.body[:200]!r}"
    )
    assert result.healthy, f"got {result}"


def test_the_stt_route_answers_four_xx_and_not_four_oh_four(endpoints):
    result = check("stt", endpoints["stt"])
    assert result.status_code is not None, f"criterion 3: the STT route answered nothing: {result}"
    assert 400 <= result.status_code < 500, (
        "criterion 3: a plain GET on the ASR path is a failed WebSocket upgrade, i.e. a 4xx — "
        f"that is what proves the route is mounted; got {result}"
    )
    assert result.status_code != 404, (
        "criterion 3: a 404 means a live process with no ASR route (the silent failure this check "
        f"exists for); got {result}"
    )
    assert result.healthy, f"…so the service is healthy; got {result}"


def test_no_probe_touches_an_inference_endpoint(endpoints):
    """Both backends: the probe URL is a status poll, never a request with a body."""
    probe_url = rig.attr(rig.module("supervisor"), "probe_url")
    for name, url in endpoints.items():
        probed = probe_url(name, url)
        assert urlsplit(probed).path not in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/responses",
        ), f"{name} would be probed at {probed}"
        assert urlsplit(probed).hostname in ("127.0.0.1", "localhost"), (
            f"{name} would be probed at {probed}, which is not loopback"
        )
