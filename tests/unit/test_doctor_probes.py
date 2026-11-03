"""Health is the service health endpoint, not a TCP connect (criterion 3).

The stubs bind ``127.0.0.1:0`` (``tests/fakes/http_service.py``). The adversarial case is P1-6:
a socket that accepts the connection and answers 503, and a live process that answers 200 on
``/`` and 404 on the ASR path. Today ``cli.probe_endpoint`` calls both healthy — this test proves
that by *running* it, so it cannot pass by accident.

Regime: no GPU, no network off the box, no request body (which is what keeps ``astra doctor``
out of ``/launch``: ``.claude/rules/gpu-services.md``, status polls are free).
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from local_astra import cli
from local_astra.config import LOOPBACK_PREFIXES, load
from tests.fakes.http_service import StubService, stt_routes, tts_routes, vllm_routes
from tests.unit import _box_rig as rig

INFERENCE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses")


def supervisor():
    return rig.module("supervisor")


def check(name: str, url: str):
    return rig.attr(supervisor(), "check_service")(name, url)


def test_a_healthy_engine_tts_and_stt_answer_their_own_endpoint():
    with StubService(vllm_routes(), name="vllm") as engine:
        result = check("vllm", f"{engine.base}/v1")
        assert result.healthy and result.reason is None, (
            f"criterion 3: 200 on /health is healthy; got {result}"
        )
        assert result.status_code == 200
        assert "/health" in engine.paths(), (
            f"criterion 3: the engine is asked for /health; it was asked for {engine.paths()}"
        )

    with StubService(tts_routes(), name="tts") as tts:
        result = check("tts", tts.base)
        assert result.healthy, f"criterion 3: a body containing `healthy` is healthy; got {result}"
        assert "healthy" in result.body

    with StubService(stt_routes(), name="stt") as stt:
        result = check("stt", f"ws://127.0.0.1:{stt.port}/api/asr-streaming")
        assert result.healthy, (
            "criterion 3: moshi has no health route, so a 4xx on the ASR path proves the route is "
            f"mounted; got {result}"
        )
        assert 400 <= (result.status_code or 0) < 500, f"…and it was a 4xx; got {result}"
        asked = stt.paths()
        assert "/api/asr-streaming" in asked, f"criterion 3: it probes the ASR path; asked {asked}"
        auth = parse_qs(urlsplit(stt.requests[-1][1]).query)
        assert auth.get("auth_id") == ["public_token"], (
            f"criterion 3: with auth_id=public_token; the query was {stt.requests[-1][1]!r}"
        )


def test_a_socket_that_answers_503_and_an_stt_route_that_404s_are_not_healthy():
    """Criterion 3, adversarial (P1-6): the two cases a TCP connect calls healthy."""
    with StubService(vllm_routes(healthy=False), name="vllm") as engine:
        url = f"{engine.base}/v1"
        with socket.create_connection(("127.0.0.1", engine.port), timeout=2) as sock:
            assert sock.fileno() >= 0
        assert cli.probe_endpoint("vllm", url) is None, (
            "the oracle for this test: today TCP-connect probe calls this stub healthy. If it "
            "does not, the stub is not reproducing P1-6 any more and the test below is vacuous."
        )
        result = check("vllm", url)
        assert not result.healthy, f"criterion 3: 503 on /health is not healthy; got {result}"
        assert "503" in (result.reason or ""), (
            f"…and the reason names the status; it said {result.reason!r}"
        )

    with StubService(stt_routes(mounted=False), name="stt") as stt:
        url = f"ws://127.0.0.1:{stt.port}/api/asr-streaming"
        assert cli.probe_endpoint("stt", url) is None, (
            "the oracle: a live process with no ASR route passes a TCP connect today"
        )
        result = check("stt", url)
        assert not result.healthy, (
            "criterion 3: a 404 means a live process with no ASR route — the silent failure this "
            f"check exists for; got {result}"
        )
        assert "404" in (result.reason or ""), f"…named; it said {result.reason!r}"


def test_a_tts_that_is_still_starting_is_not_healthy():
    with StubService(tts_routes(healthy=False), name="tts") as tts:
        result = check("tts", tts.base)
        assert not result.healthy, (
            f"criterion 3: 200 with a body that does not say `healthy` is not healthy; got {result}"
        )


def test_a_refused_connection_is_reported_not_raised():
    port = rig.free_port()
    result = check("tts", f"http://127.0.0.1:{port}")
    assert not result.healthy and result.status_code is None, (
        f"criterion 3: nothing listening is a reason, not a traceback; got {result}"
    )
    assert str(port) in (result.reason or ""), f"…naming the endpoint; it said {result.reason!r}"


def test_every_probed_url_is_loopback_and_keeps_the_configured_path():
    """Criterion 3, negative: the probe URL is derived (purely) from the parsed ``Services`` value.

    ``probe_url`` does no I/O, so this test touches no service: which URL would be probed is a
    question about the configuration, not about the box.
    """
    services = load(None).profiles["lean"].services
    probe_url = rig.attr(supervisor(), "probe_url")
    seen = {
        name: probe_url(name, url)
        for name, url in (("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts))
    }

    for name, url in seen.items():
        assert url.startswith(LOOPBACK_PREFIXES), (
            f"criterion 3, negative: {name} would be probed at {url}, which is not loopback "
            "(docs/02 §9, CLAUDE.md)"
        )
        assert urlsplit(url).path not in INFERENCE_PATHS, f"{name} -> {url}"
    assert urlsplit(seen["vllm"]).path.endswith("/health"), f"got {seen['vllm']!r}"
    assert seen["stt"].startswith("http://127.0.0.1:8080/api/asr-streaming"), (
        "criterion 3: the STT probe URL is the http form of the configured ws:// endpoint, path "
        f"and all — never a string typed in the module; got {seen['stt']!r}"
    )
    assert parse_qs(urlsplit(seen["stt"]).query).get("auth_id") == ["public_token"], (
        f"criterion 3: …with auth_id=public_token; got {seen['stt']!r}"
    )


def test_no_probe_carries_a_body_or_touches_an_inference_endpoint():
    """Criterion 3, negative, static: this is what keeps ``astra doctor`` out of ``/launch``."""
    for module_name in ("supervisor", "preflight"):
        path = Path(rig.REPO / "local_astra" / f"{module_name}.py")
        assert path.exists(), f"T-0033: local_astra/{module_name}.py"
        source = path.read_text()
        for banned in INFERENCE_PATHS:
            assert banned not in source, (
                f"criterion 3, negative: {module_name}.py names {banned}; a request with a body to "
                "an inference endpoint is GPU time and would need a /launch token"
            )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kwargs = {kw.arg for kw in node.keywords}
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in {"Request", "urlopen", "post", "put", "request"}:
                assert "data" not in kwargs and "json" not in kwargs and "body" not in kwargs, (
                    f"criterion 3, negative: {module_name}.py sends a request body in a "
                    f"{name}(...) call (line {node.lineno}); every probe is a bare GET"
                )
            assert name != "post", (
                f"criterion 3, negative: {module_name}.py POSTs (line {node.lineno})"
            )


def test_the_stub_service_records_that_the_probe_sent_no_body():
    """The dynamic half of the previous test: a probe that carried bytes would show up here."""
    with StubService(vllm_routes(), name="vllm") as engine:
        check("vllm", f"{engine.base}/v1")
        methods = {method for method, _path, _length in engine.requests}
        lengths = {length for _method, _path, length in engine.requests}
        assert methods == {"GET"}, f"criterion 3: probes are GETs; the stub saw {methods}"
        assert lengths == {0}, f"criterion 3: with no body; the stub saw content-lengths {lengths}"
