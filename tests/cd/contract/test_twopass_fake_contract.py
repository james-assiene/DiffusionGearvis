"""Contract rows for what T-0080b adds to ``tests/cd/fakes/vllm_server.py`` (tests.md: a behaviour added to a fake
needs a real-service row). Parametrised ``backend = fake | real``; ``real`` is marked ``gpu`` and needs the
served engine (``ASTRA_CD_REAL_URL``, e.g. ``http://127.0.0.1:8001/v1``); run via /launch, never against a
dirty tree.

Pinned behaviours (the two-pass driver relies on each; T-0080b implementation notes call the vLLM side
"unverified" -- these rows are where it gets verified on the 5090 box):

1. ``GET /metrics`` (server root) exposes ``vllm:prefix_cache_queries_total`` and ``vllm:prefix_cache_hits_total``
   (the names ``scripts/gpu/x3_prefix_cache.sh`` reads on v0.29.0); one chat request raises the queries counter,
   hits never exceed queries, and a repeat of the same prompt hits the prefix cache (hits rise);
2. ``continue_final_message: true`` with ``add_generation_prompt`` true -- explicitly, or by its default -- is a 400;
3. ``continue_final_message: true`` whose last message is not the assistant's is a 400 (unverified on vLLM: if the
   real row fails here, the fake is wrong and must change with it);
4. the T4 shape (pass-1 messages + an assistant message ending in ``<cc>``, ``continue_final_message`` true,
   ``add_generation_prompt`` false, thinking on) is accepted (200).

Regime (real): engine-only boot of ``services/vllm.sh`` (prefix caching on), model per ``services/versions.lock``,
thinking on, ``max_tokens`` 16, concurrency 1, no seed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.cd.fakes.vllm_server import MODEL_ID, FakeVllmServer, Reply

QUERIES = "vllm:prefix_cache_queries_total"
HITS = "vllm:prefix_cache_hits_total"
SYSTEM = (
    "You are a helpful assistant. When an action is needed, end your answer with <cc> followed by one "
    "JSON call. Tools: google_search, click_at, type_at, scroll, press_key."
)
USER = "Look up the opening hours of the British Museum."


def _fake_script(index: int, body: Any) -> Reply:
    # each served request queries 64 prompt tokens; a repeat of an earlier prompt hits 48 of them
    return Reply(
        text="<|channel>thought\nok<channel|>Sure. <cc>",
        prefix_queries=64,
        prefix_hits=0 if index == 0 else 48,
    )


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.gpu)])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str]:
    if request.param == "real":
        url = os.environ.get("ASTRA_CD_REAL_URL")
        if not url:
            pytest.fail("backend=real needs ASTRA_CD_REAL_URL (run via /launch)")
        yield url
        return
    with FakeVllmServer(_fake_script, tmp_path) as srv:
        yield srv.url


def _root(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def _counters(url: str) -> tuple[float, float]:
    with urllib.request.urlopen(_root(url) + "/metrics", timeout=10) as resp:
        assert resp.status == 200
        text = resp.read().decode()
    out: dict[str, float] = {}
    for name in (QUERIES, HITS):
        # Prometheus text: ``name{labels} value`` or ``name value``; sum over label sets
        vals = [
            float(m.group(1))
            for m in re.finditer(
                rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE
            )
        ]
        assert vals, f"{name} is not exposed by /metrics"
        out[name] = sum(vals)
    return out[QUERIES], out[HITS]


def _messages() -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]


def _body(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": _messages(),
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 16,
    }
    body.update(extra)
    return body


def _post(url: str, body: dict[str, Any]) -> int:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def test_metrics_expose_the_prefix_cache_counters_and_a_repeat_hits(backend: str) -> None:
    q0, h0 = _counters(backend)
    assert _post(backend, _body()) == 200
    q1, h1 = _counters(backend)
    assert q1 > q0, "a served request did not raise vllm:prefix_cache_queries_total"
    assert 0 <= h1 - h0 <= q1 - q0
    assert _post(backend, _body()) == 200
    q2, h2 = _counters(backend)
    assert q2 > q1
    assert h2 > h1, "the same prompt again did not hit the prefix cache"
    assert h2 - h1 <= q2 - q1


@pytest.mark.parametrize("agp", [True, None], ids=["explicit", "default"])
def test_continue_final_message_with_add_generation_prompt_is_400(
    backend: str, agp: bool | None
) -> None:
    msgs = [*_messages(), {"role": "assistant", "content": "Sure. <cc>"}]
    extra: dict[str, Any] = {"messages": msgs, "continue_final_message": True}
    if agp is not None:
        extra["add_generation_prompt"] = agp
    assert _post(backend, _body(**extra)) == 400


def test_continue_final_message_after_a_user_message_is_400(backend: str) -> None:
    body = _body(continue_final_message=True, add_generation_prompt=False)
    assert _post(backend, body) == 400


def test_the_t4_continuation_shape_is_accepted(backend: str) -> None:
    msgs = [*_messages(), {"role": "assistant", "content": "Let me look that up. <cc>"}]
    body = _body(messages=msgs, continue_final_message=True, add_generation_prompt=False)
    assert _post(backend, body) == 200


# --------------------------------------------------------------------------- T-0081b: a pass 1 cut by its budget

#: the pinned ``generation_config.json`` stop ids (EOS, ``<turn|>``, ``<|tool_response>``)
STOP_IDS = frozenset({1, 50, 106})


def _long_script(index: int, body: Any) -> Reply:
    # a whole turn far longer than any budget below: a thought, an answer, a tag and a call, then (the fake) EOS
    return Reply(
        text="<|channel>thought\n"
        + "The user wants the opening hours of a museum in London. " * 4
        + "<channel|>Let me look that up. <cc>"
        + '{"name": "google_search", "arguments": {"query": "british museum opening hours"}}'
    )


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.gpu)])
def backend_ids(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str]:
    """Like ``backend``, with ``return_token_ids`` served by the fake (it needs the pinned tokenizer)."""
    if request.param == "real":
        url = os.environ.get("ASTRA_CD_REAL_URL")
        if not url:
            pytest.fail("backend=real needs ASTRA_CD_REAL_URL (run via /launch)")
        yield url
        return
    from tests.cd import _compile_support as cs

    with FakeVllmServer(_long_script, tmp_path, encode=cs.encode) as srv:
        yield srv.url


def _token_ids(url: str, body: dict[str, Any]) -> list[int]:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
    ids = payload["choices"][0].get("token_ids")
    assert isinstance(ids, list), f"no token_ids in {payload}"
    return [int(t) for t in ids]


def test_return_token_ids_cut_by_max_tokens_carry_no_stop(backend_ids: str) -> None:
    """T-0081b criterion 4 (a truncated draft is sent without S): a generation that runs out of ``max_tokens`` serves
    exactly ``max_tokens`` ids and no stop id among them (the thought alone is longer than 8 tokens, thinking on)."""
    ids = _token_ids(backend_ids, _body(max_tokens=8, return_token_ids=True))
    assert len(ids) == 8, ids
    assert not STOP_IDS & set(ids), ids


# --------------------------------------------------------------------------- T-0082b R10: flush vs warm-up


def test_fake_flush_prompt_is_the_drivers() -> None:
    """The fake recognises the flush prompt the drivers actually send (fake-side pin; the real engine has no
    notion of a flush, see the parametrised row below)."""
    from local_astra.eval.cd import runner
    from tests.cd.fakes.vllm_server import FLUSH_PROMPT

    assert FLUSH_PROMPT == runner.FLUSH_PROMPT


def _flush_body() -> dict[str, Any]:
    from local_astra.eval.cd import prompts, runner

    body = prompts.request_body(MODEL_ID, runner.FLUSH_PROMPT)
    body["max_tokens"] = 1
    body.pop("return_token_ids", None)
    return body


def test_fake_a_flush_is_still_a_flush(tmp_path: Path) -> None:
    """T-0082b R10, one direction: the driver's flush body (``_Run.flush``: the flush prompt, ``max_tokens`` 1, no CD
    keys) is answered as a flush -- counted in ``flushes``, never in ``requests``, never scripted."""
    calls: list[int] = []

    def script(index: int, body: Any) -> Reply:
        calls.append(index)
        return Reply(text="x")

    with FakeVllmServer(script, tmp_path) as srv:
        assert _post(srv.url, _flush_body()) == 200
        assert (len(srv.flushes), len(srv.requests), calls) == (1, 0, [])


def test_fake_a_warmup_is_not_a_flush(tmp_path: Path) -> None:
    """T-0082b R10, the other direction: a pass-1-shaped body (system + user, thinking on) with ``max_tokens`` 1 and no
    CD keys -- a warm-up -- is recorded in ``requests`` and scripted, not swallowed as a flush."""
    calls: list[int] = []

    def script(index: int, body: Any) -> Reply:
        calls.append(index)
        return Reply(text="<|channel>thought\nok<channel|>Sure.")

    with FakeVllmServer(script, tmp_path) as srv:
        assert _post(srv.url, _body(max_tokens=1)) == 200
        assert (len(srv.flushes), len(srv.requests), calls) == (0, 1, [0])


def test_a_max_tokens_1_generation_serves_one_id(backend_ids: str) -> None:
    """T-0082b R10 parity row: a warm-up-shaped request (the pass-1 prompt, ``max_tokens`` 1) is a real one-token
    generation on both backends -- exactly one served id."""
    ids = _token_ids(backend_ids, _body(max_tokens=1, return_token_ids=True))
    assert len(ids) == 1, ids


# --------------------------------------------------------------------------- T-0084c: the native format's served prompt
#
# The two-pass driver's native requests send the declarations in the request's ``tools`` field with ``tool_choice``
# ``"none"`` (QN3) and check the served prompt (criterion 6). These rows pin what the fake serves against the engine:
# ``prompt_token_ids`` at the top level when ``return_token_ids`` is set; one ``<|tool>``/``<tool|>`` pair (ids 46/47)
# per ``tools`` entry even under ``tool_choice: "none"``; a ``continue_final_message`` request ending in
# ``<|tool_call>`` ends in id 48; identical requests give identical ids; ``tools`` together with
# ``continue_final_message`` is accepted. Regime (real): as above, plus ``tools_v1.json`` (5 declarations) through
# ``native.tools_field``; vLLM v0.29.0 without ``--exclude-tools-when-tool-choice-none`` (``services/vllm.sh``).

N_TOOLS = 5  # tools_v1.json declares google_search, click_at, type_at, scroll, press_key
TOOL_OPEN_ID, TOOL_CLOSE_ID, TOOL_CALL_ID = 46, 47, 48


def _tools_field() -> list[dict[str, Any]]:
    from local_astra.toolcall import native, tools

    return native.tools_field(tools.load())


def _native_body(**extra: Any) -> dict[str, Any]:
    return _body(tools=_tools_field(), tool_choice="none", return_token_ids=True, **extra)


def _t4_native_body(**extra: Any) -> dict[str, Any]:
    body = _native_body(**extra)
    body["messages"] = [
        *_messages(),
        {"role": "assistant", "content": "Let me look that up. <|tool_call>"},
    ]
    body.update(continue_final_message=True, add_generation_prompt=False)
    return body


def _payload(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return int(resp.status), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), {}


def _prompt_ids(url: str, body: dict[str, Any]) -> list[int]:
    status, payload = _payload(url, body)
    assert status == 200, (status, payload)
    ids = payload.get("prompt_token_ids")
    assert isinstance(ids, list) and ids, f"no top-level prompt_token_ids in {sorted(payload)}"
    assert all(isinstance(t, int) for t in ids), ids[:20]
    return [int(t) for t in ids]


def test_t0084c_prompt_token_ids_are_served_at_the_top_level(backend_ids: str) -> None:
    """``return_token_ids`` serves the prompt's ids beside the choice's ``token_ids`` (vLLM v0.29.0)."""
    ids = _prompt_ids(backend_ids, _body(return_token_ids=True))
    assert len(ids) > 10, ids


def test_t0084c_tool_choice_none_still_renders_one_declaration_per_tool(backend_ids: str) -> None:
    """QN3: ``tools`` with ``tool_choice: "none"`` still renders every declaration in the prompt: 5 x id 46, 5 x id 47."""
    ids = _prompt_ids(backend_ids, _native_body())
    assert (ids.count(TOOL_OPEN_ID), ids.count(TOOL_CLOSE_ID)) == (N_TOOLS, N_TOOLS)


def test_t0084c_without_tools_no_declaration_ids_are_rendered(backend_ids: str) -> None:
    """The other direction of the row above: the cc request (no ``tools``) renders neither 46 nor 47, so a count of
    5 really comes from the field."""
    ids = _prompt_ids(backend_ids, _body(return_token_ids=True))
    assert (ids.count(TOOL_OPEN_ID), ids.count(TOOL_CLOSE_ID)) == (0, 0)


def test_t0084c_tools_with_continue_final_message_are_accepted(backend_ids: str) -> None:
    """The native T4 shape (``tools`` + an assistant prefix ending in ``<|tool_call>``, ``continue_final_message``,
    ``add_generation_prompt`` false) is a 200, and its prompt ends in id 48 with the declarations still rendered."""
    ids = _prompt_ids(backend_ids, _t4_native_body())
    assert ids[-1] == TOOL_CALL_ID, ids[-8:]
    assert ids.count(TOOL_OPEN_ID) == N_TOOLS


def test_t0084c_identical_requests_get_identical_prompt_ids(backend_ids: str) -> None:
    """Stable ids: the same messages and tools give the same prompt ids (criterion 6's prefix guard relies on it), and
    a T1/T2-shaped copy (``max_tokens`` changed) keeps them."""
    a = _prompt_ids(backend_ids, _native_body())
    b = _prompt_ids(backend_ids, _native_body())
    c = _prompt_ids(backend_ids, _native_body(max_tokens=8))
    assert a == b == c


def test_t0084c_the_fake_prompt_is_the_template_render(tmp_path: Path) -> None:
    """Fake-side pin: the served ids are the pinned template's render of the request, encoded with the pinned
    tokenizer (the oracle), with the generation prompt for a plain request and none for a continuation."""
    from tests.cd import _compile_support as cs
    from tests.cd.oracle import gemma4_template as g

    with FakeVllmServer(_long_script, tmp_path, encode=cs.encode) as srv:
        got = _prompt_ids(srv.url, _native_body())
        got_t4 = _prompt_ids(srv.url, _t4_native_body())
    want = cs.encode(
        g.render(
            _messages(), tools=_tools_field(), add_generation_prompt=True, enable_thinking=True
        )
    )
    assert got == list(want)
    full = g.render(
        [*_messages(), {"role": "assistant", "content": "Let me look that up. <|tool_call>"}],
        tools=_tools_field(),
        enable_thinking=True,
    )
    cut = full[: full.rindex("<|tool_call>") + len("<|tool_call>")]
    assert got_t4 == list(cs.encode(cut))


def test_t0084c_fake_drop_declarations_mode(tmp_path: Path) -> None:
    """Fake-only scripted mode: a server started with ``--exclude-tools-when-tool-choice-none``
    (``drop_declarations=True``) renders no declaration under ``tool_choice: "none"`` -- the case criterion 6's
    ``declarations`` guard exists for -- and still renders them under ``tool_choice: "auto"``."""
    from tests.cd import _compile_support as cs

    with FakeVllmServer(_long_script, tmp_path, encode=cs.encode, drop_declarations=True) as srv:
        none_ids = _prompt_ids(srv.url, _native_body())
        auto_ids = _prompt_ids(srv.url, {**_native_body(), "tool_choice": "auto"})
    assert none_ids.count(TOOL_OPEN_ID) == 0
    assert auto_ids.count(TOOL_OPEN_ID) == N_TOOLS


def test_t0084c_fake_reply_hooks_script_the_served_prompt(tmp_path: Path) -> None:
    """Fake-only: ``Reply.omit_prompt_token_ids`` serves none, ``Reply.drop_declarations`` renders without the
    ``tools``, ``Reply.prompt_ids_edit`` maps the rendered ids; ``prompt_ids`` keeps what was served."""
    from tests.cd import _compile_support as cs

    plan = [
        Reply(text="a", omit_prompt_token_ids=True),
        Reply(text="b", drop_declarations=True),
        Reply(text="c", prompt_ids_edit=lambda ids: ids[:-1]),
        Reply(text="d"),
    ]

    def script(index: int, body: Any) -> Reply:
        return plan[index]

    with FakeVllmServer(script, tmp_path, encode=cs.encode) as srv:
        s0, p0 = _payload(srv.url, _native_body())
        dropped = _prompt_ids(srv.url, _native_body())
        edited = _prompt_ids(srv.url, _native_body())
        plain = _prompt_ids(srv.url, _native_body())
        served = list(srv.prompt_ids)
    assert s0 == 200 and "prompt_token_ids" not in p0
    assert dropped.count(TOOL_OPEN_ID) == 0
    assert edited == plain[:-1]
    assert served == [None, dropped, edited, plain]


# --------------------------------------------------------------------------- T-0085: the template check (criterion 4)
# A ``continue_final_message`` request is rendered by vLLM v0.29.0 through ``transformers``: the final content + a
# sentinel is rendered with the pinned template and the *stripped* final content must appear in the render. The pinned
# template's ``strip_thinking`` deletes every ``<channel|>`` in assistant content, so the click-01 continuation of the
# T-0084e stop (``"I will click the Sign in button.<channel|><|tool_call>"``) is a 400; the cleaned one is accepted and
# its prompt ends in id 48. Real side: the same request against the served engine (``gpu``; no separate launch before
# the H-T85 re-run, human decision).

CLICK01_STALE = "I will click the Sign in button.<channel|><|tool_call>"
CLICK01_CLEAN = "I will click the Sign in button.<|tool_call>"
MISMATCH = "the final message does not appear in the chat"


def _t4_with(content: str, *, native: bool = True) -> dict[str, Any]:
    body = _native_body() if native else _body(return_token_ids=True)
    body["messages"] = [*_messages(), {"role": "assistant", "content": content}]
    body.update(continue_final_message=True, add_generation_prompt=False)
    return body


def _status_text(url: str, body: dict[str, Any]) -> tuple[int, str]:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return int(resp.status), resp.read().decode()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode()


def test_t0085_the_click01_continuation_is_a_400(backend_ids: str) -> None:
    """Criterion 4: the T4 body of the T-0084e stop (native: ``tools`` + the stale click-01 prefix + ``<|tool_call>``)
    is refused with 400 and vLLM's text ("... the final message does not appear in the chat ...")."""
    status, text = _status_text(backend_ids, _t4_with(CLICK01_STALE))
    assert status == 400, (status, text[:300])
    assert MISMATCH in text, text[:500]


def test_t0085_the_cleaned_click01_continuation_is_accepted_ending_in_48(backend_ids: str) -> None:
    """Criterion 4, the other direction: the cleaned content is a 200 and its served prompt ends in id 48."""
    ids = _prompt_ids(backend_ids, _t4_with(CLICK01_CLEAN))
    assert ids[-1] == TOOL_CALL_ID, ids[-8:]


@pytest.mark.parametrize(
    ("content", "status"),
    [
        ("Sure.<channel|> <cc>", 400),
        ("Sure.<|channel>thinking <cc>", 400),
        ("Sure. <cc>", 200),
        ("\nSure.\n<cc>", 200),
        ("Sure.<|think|> <cc>", 200),
        ("Pipes |> and <channel are text. <cc>", 200),
    ],
    ids=[
        "cc_stray_close",
        "cc_open",
        "cc_clean",
        "cc_leading_newline",
        "cc_think_kept",
        "cc_pieces",
    ],
)
def test_t0085_cc_continuations_are_checked_too(
    backend_ids: str, content: str, status: int
) -> None:
    """Criterion 4 applies to every ``continue_final_message`` request, cc included: what the template deletes is a
    400; what it keeps (a leading newline the template trims, a non-channel special piece, marker *pieces*) is not."""
    got, text = _status_text(backend_ids, _t4_with(content, native=False))
    assert got == status, (content, got, text[:300])
    if status == 400:
        assert MISMATCH in text, text[:500]


def test_t0085_fake_refuses_before_the_script_sees_the_request(tmp_path: Path) -> None:
    """Fake-side pin: the refused request reaches neither the script nor ``requests``; it is kept in ``rejected``; an
    accepted continuation after it is scripted as usual."""
    from tests.cd import _compile_support as cs

    seen: list[int] = []

    def script(index: int, body: Any) -> Reply:
        seen.append(index)
        return Reply(text="call:click_at{x:1,y:2}<tool_call|>")

    with FakeVllmServer(script, tmp_path, encode=cs.encode) as srv:
        s1, _ = _status_text(srv.url, _t4_with(CLICK01_STALE))
        s2, _ = _status_text(srv.url, _t4_with(CLICK01_CLEAN))
        rejected, requests = list(srv.rejected), list(srv.requests)
    assert (s1, s2) == (400, 200)
    assert seen == [0]
    assert len(rejected) == 1 and rejected[0]["messages"][-1]["content"] == CLICK01_STALE
    assert len(requests) == 1 and requests[0]["messages"][-1]["content"] == CLICK01_CLEAN


def test_t0085_fake_cuts_a_continuation_as_transformers_does(tmp_path: Path) -> None:
    """Fake-side pin of the cut (the oracle render, by hand): a content with a leading newline -- which the template
    trims, so the old ``rindex(final)`` cut could not find it -- gives the render of the trimmed content cut right
    after ``<|tool_call>``; ids end in 48."""
    from tests.cd import _compile_support as cs
    from tests.cd.oracle import gemma4_template as g

    content = "\nLet me look that up. <|tool_call>"
    with FakeVllmServer(_long_script, tmp_path, encode=cs.encode) as srv:
        got = _prompt_ids(srv.url, _t4_with(content))
    full = g.render(
        [*_messages(), {"role": "assistant", "content": "Let me look that up. <|tool_call>"}],
        tools=_tools_field(),
        enable_thinking=True,
    )
    cut = full[: full.rindex("<|tool_call>") + len("<|tool_call>")]
    assert got == list(cs.encode(cut))
    assert got[-1] == TOOL_CALL_ID
