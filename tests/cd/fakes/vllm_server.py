"""A loopback fake of the vLLM OpenAI server plus the engine log that carries ``ASTRA_CD`` records (T-0070).

Scriptable per request. It stands in for the real stack the harness talks to:

- ``POST /v1/chat/completions`` (non-streaming JSON, or SSE when ``"stream": true``), ``GET /v1/models`` and
  ``GET /health`` (200, empty body, as vLLM's); every GET path is kept in ``gets`` (T-0071b C9 probes);
- a dead engine (T-0071b C9): after a reply with ``die_after`` (or ``die()``), every later request -- GET or
  POST -- has its connection closed without a response (the client sees a dropped connection, as against a
  crashed server); POSTs that arrive dead are counted in ``posts_while_dead``;
- the response message is split the way ``--reasoning-parser gemma4`` splits it (see ``split_served``):
  the header between ``<|channel>`` and ``<channel|>`` goes to ``reasoning`` (``thought\\n`` label
  stripped, as vLLM's ``gemma4_utils.parse_thinking_output`` does), and ``content`` is the rest with the
  special pieces dropped (``skip_special_tokens``). A whole-text comparison with the committed ids can
  therefore never match -- which is the point of T-0070 criterion 5;
- ``"return_token_ids": true`` adds ``choices[0].token_ids`` (the generated ids, special pieces
  included), the only served evidence of whether an unconstrained text opened with ``<|channel>``
  (hard stop (f)'s header rate);
- for every request carrying ``astra_cd_*`` keys, one ``ASTRA_CD <json>`` line per committed canvas is
  appended to ``engine_log`` behind a vLLM-style prefix, with decoy records of *other* tags interleaved
  and multi-canvas records written in reverse ``canvas_idx`` order. Every record validates against
  T-0064's ``CdCanvasRecord`` before it is written. The ``astra_cd_*`` keys are first parsed with the
  plugin's own ``parse_cd_xargs`` (T-0064's note: a fake that consumes them uses that parser), and a
  refusal becomes an error record with the parser's reason, as the real sampler does (T-0067 crit. 7).

The real server's behaviour for the reasoning split, the record line prefix and the record lag is pinned
by ``tests/cd/contract/test_eval_fake_server_contract.py`` (rows marked ``gpu``).

T-0080b extensions (the two-pass tool-call driver; real-service rows in
``tests/cd/contract/test_twopass_fake_contract.py``):

- ``GET /metrics`` (at the server root, beside ``/health``) serves the Prometheus text vLLM serves, with the two
  prefix-cache counters ``vllm:prefix_cache_queries_total`` and ``vllm:prefix_cache_hits_total`` (labelled, as
  vLLM's are). Each scripted reply adds its ``Reply.prefix_queries`` / ``Reply.prefix_hits`` tokens to them
  when it is served (flush requests add nothing). ``metrics=False`` answers ``/metrics`` with 404 (an engine
  whose counters are unavailable: the driver must report the hit as unavailable, never guess it);
- the pass-1 and pass-2 chat requests are answered by the script like any other (the text carries the
  thought, the answer and the ``<cc>`` tag; ``vllm_xargs`` makes it constrained);
- ``continue_final_message`` is honoured the way vLLM validates it: a request that sets it together with
  ``add_generation_prompt`` (whose default is true) or whose last message is not the assistant's is refused
  with 400 before the script sees it (kept in ``rejected``, not in ``requests``).

T-0081b extensions (the draft-seeded arm of the two-pass driver; the real-service row is
``test_return_token_ids_cut_by_max_tokens_carry_no_stop`` in ``tests/cd/contract/test_twopass_fake_contract.py``):

- ``return_token_ids`` honours ``max_tokens`` as vLLM does: the served ids are cut to the first ``max_tokens``, so a
  generation that runs out of budget ends without a stop id (a truncated pass-1 draft);
- ``Reply.canvas_steps`` scripts the step counter of each canvas record (``canvas_steps[i]`` on ``canvas_idx`` i);
  ``Reply.steps`` stays the value of every canvas when it is not given. This is a value the record schema already
  carries per canvas (``CdCanvasRecord.steps``), not a new behaviour.

T-0084c extensions (the native Gemma 4 tool-call format in the two-pass driver; real-service rows in
``tests/cd/contract/test_twopass_fake_contract.py``, the ``t0084c`` section):

- ``return_token_ids`` also serves the prompt's ids at the top level (``prompt_token_ids``), as vLLM v0.29.0 does. They
  are the ids of the pinned chat template rendered the way vLLM renders a chat request (``tests/cd/oracle/
  gemma4_template.py``: the request's ``messages``, its ``tools`` field, ``chat_template_kwargs.enable_thinking``,
  ``add_generation_prompt``), encoded with the pinned tokenizer, so each ``tools`` entry renders one
  ``<|tool>``/``<tool|>`` pair (ids 46/47) in the system turn, identical requests get identical ids, and a
  ``continue_final_message`` request ends at its last message's content (``...<|tool_call>`` -> id 48), cut where
  ``transformers`` cuts it (``rindex`` of the final content). Image parts render nothing here (the real engine
  inserts image tokens; no row counts them). The ids of each scripted request are kept in ``prompt_ids``;
- ``tool_choice: "none"`` still renders the declarations (vLLM without ``--exclude-tools-when-tool-choice-none``);
  ``drop_declarations=True`` is a scripted server started *with* that flag: a ``tool_choice: "none"`` request then
  renders no declarations;
- per reply: ``Reply.omit_prompt_token_ids`` serves no ``prompt_token_ids``; ``Reply.drop_declarations`` renders this
  request without its ``tools``; ``Reply.prompt_ids_edit`` maps the rendered ids to the served ones (a served prompt
  that differs from what the request implies). These script the driver's served-prompt guards; they are not
  behaviours of the real engine.

T-0085 extension (the template check; real-service rows in ``tests/cd/contract/test_twopass_fake_contract.py``, the
``t0085`` section):

- every ``continue_final_message`` request (cc and native alike) is rendered as ``transformers`` renders it for vLLM
  v0.29.0 (``render_jinja_template``): the final message's text + ``CONTINUE_SENTINEL`` is rendered with the pinned
  template, and if the *stripped* final text does not appear in the render (the template's ``strip_thinking``
  deleted a ``<channel|>`` or dropped from a ``<|channel>``), the request is refused with 400 and the text vLLM
  serves (``TEMPLATE_MISMATCH`` + the final message), before the script sees it (kept in ``rejected``);
- ``render_prompt`` cuts a continuation where ``transformers`` cuts it: at the sentinel, right-stripped when the
  template trimmed the whitespace after it (it does: ``strip_thinking`` ends in ``trim``).

Binds 127.0.0.1 only.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import http.server
import json
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

EOS, TURN_CLOSE, TOOL_RESPONSE = 1, 106, 50
STOP_IDS = frozenset({EOS, TURN_CLOSE, TOOL_RESPONSE})
SPECIAL_PIECES = (
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|tool_response>",
    "<eos>",
    "<pad>",
)
LOG_PREFIX = "(EngineCore_DP0 pid=4242) INFO 09-30 12:00:00 [record.py:88] "
MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
#: the ``astra_cd_mode`` values the plugin parses (``xargs.MODES``), T-0064 + T-0068 + T-0076
WIRE_MODES = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax")
STEP_CAP = 48
#: T-0085: the tag ``transformers`` appends to the final message of a ``continue_final_message`` request to find the
#: cut (``render_jinja_template``'s ``continue_final_message_tag``); seen in the T-0084e engine log at 9aee66d:
#: ``...<|tool_call>CONTINUE_FINAL_MESSAGE_TAG<turn|>`` (its trailing space trimmed by the template)
CONTINUE_SENTINEL = "CONTINUE_FINAL_MESSAGE_TAG "
#: T-0085: the ``ValueError`` text ``transformers`` raises (and vLLM v0.29.0 serves as a 400) when the template
#: deletes part of the final message (verbatim from the T-0084e engine log at 9aee66d)
TEMPLATE_MISMATCH = (
    "continue_final_message is set but the final message does not appear in the chat after applying the chat "
    "template! This can happen if the chat template deletes portions of the final message. Please verify the chat "
    "template and final message in your chat to ensure they are compatible."
)


@dataclasses.dataclass
class Reply:
    """What the fake does for one request.

    ``text`` is the whole generation as the model would decode it with special pieces kept (header
    included), e.g. ``"<|channel>thought\\n<channel|>3124\\n..."``. For a constrained request the committed
    ids default to ``encode(text) + [EOS]``; ``committed_text`` makes the records carry a *different* text
    than the one served (a join mismatch).
    """

    text: str = ""
    committed_text: str | None = None
    committed_ids: Sequence[int] | None = None
    in_final: bool = True
    steps: int = 12
    error: str | None = None
    http_status: int = 200
    hang_s: float = 0.0
    omit_records: bool = False
    record_delay_s: float = 0.0
    #: E9: hold this request's records until the next request (of any kind) arrives
    hold_records: bool = False
    #: with ``hold_records``: how many later requests must arrive before the records are written
    hold_for_requests: int = 1
    #: write the last canvas's record line twice, byte-identical (a duplicated ``canvas_idx``)
    duplicate_last_canvas: bool = False
    #: number the last canvas ``canvases`` instead of ``canvases - 1`` (a gap; the ids stay complete)
    gap_before_last_canvas: bool = False
    #: completeness: write every canvas but the last at once, the last one this much later
    last_canvas_delay_s: float = 0.0
    #: serve these instead of ``split_served(text)`` (a literal ``<channel|>`` spelled with ordinary
    #: tokens survives ``skip_special_tokens``; the text-level split cannot model that)
    served_reasoning: str | None = None
    served_content: str | None = None
    canvases: int = 1
    bucket: int = 64
    mean_accepted_per_step: float = 3.0
    shadow_argmax_ids: Sequence[int] | None = None
    argmax_in_language: bool | None = None
    map_equals_argmax: bool | None = None
    #: T-0071b C9: the engine dies once this reply is sent (every later request is dropped)
    die_after: bool = False
    #: T-0080b: prompt tokens this request adds to ``vllm:prefix_cache_queries_total`` / ``_hits_total``
    prefix_queries: int = 0
    prefix_hits: int = 0
    #: T-0081b: per-canvas step counters (``canvas_steps[i]`` is canvas i's ``steps``; length ``canvases``)
    canvas_steps: Sequence[int] | None = None
    #: T-0084c: serve no ``prompt_token_ids`` for this request
    omit_prompt_token_ids: bool = False
    #: T-0084c: render this request's prompt without its ``tools`` (the declarations dropped)
    drop_declarations: bool = False
    #: T-0084c: the served ``prompt_token_ids`` = ``prompt_ids_edit(rendered ids)``
    prompt_ids_edit: Callable[[list[int]], list[int]] | None = None


Script = Callable[[int, Mapping[str, Any]], Reply]


def split_served(text: str) -> tuple[str | None, str]:
    """``(reasoning, content)`` as the gemma4 reasoning parser + ``skip_special_tokens`` serve ``text``."""
    reasoning: str | None = None
    body = text
    if "<channel|>" in text:
        head, body = text.split("<channel|>", 1)
        if "<|channel>" in head:
            head = head.split("<|channel>", 1)[1]
        head = head.strip()
        if head.startswith("thought\n") or head == "thought":
            head = head[len("thought") :].lstrip("\n")
        reasoning = head.strip()
    for piece in SPECIAL_PIECES:
        body = body.replace(piece, "")
    return reasoning, body


def user_prompt(body: Mapping[str, Any]) -> str:
    msgs = body.get("messages") or []
    return str(msgs[-1]["content"]) if msgs else ""


#: The E9 flush prompt every driver sends (``runner.FLUSH_PROMPT``, ``cd_b13.FLUSH_PROMPT``; pinned equal in
#: ``tests/cd/contract/test_twopass_fake_contract.py``)
FLUSH_PROMPT = "Reply with OK."


def is_flush(body: Mapping[str, Any]) -> bool:
    """An E9 flush: no ``astra_cd_*`` key, ``max_tokens`` 1, and exactly one message, the user's flush prompt. A
    ``max_tokens``-1 request carrying any other prompt (a T-0082b R10 warm-up: a case's pass-1 body) is not a
    flush; it is recorded and scripted like any request."""
    msgs = body.get("messages") or []
    return (
        not cd_keys(body)
        and body.get("max_tokens") == 1
        and len(msgs) == 1
        and msgs[0].get("role") == "user"
        and msgs[0].get("content") == FLUSH_PROMPT
    )


def _final_text(content: Any) -> str | None:
    """The text ``transformers`` continues: a string content, or the last part carrying ``text``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for p in reversed(content):
            if isinstance(p, dict) and "text" in p:
                return str(p["text"])
    return None


def _with_sentinel(msgs: Sequence[Any]) -> list[Any]:
    out = copy.deepcopy(list(msgs))
    content = out[-1].get("content")
    if isinstance(content, str):
        out[-1]["content"] = content + CONTINUE_SENTINEL
    elif isinstance(content, list):
        for p in reversed(content):
            if isinstance(p, dict) and "text" in p:
                p["text"] = str(p["text"]) + CONTINUE_SENTINEL
                break
    return out


def _render(body: Mapping[str, Any], msgs: Sequence[Any], with_tools: bool) -> str:
    from tests.cd.oracle import gemma4_template as g

    cont = bool(body.get("continue_final_message"))
    agp = bool(body.get("add_generation_prompt", True)) and not cont
    kwargs = body.get("chat_template_kwargs") or {}
    tools = body.get("tools") if with_tools else None
    return str(
        g.render(
            list(msgs),
            tools=tools,
            add_generation_prompt=agp,
            enable_thinking=kwargs.get("enable_thinking"),
        )
    )


def template_mismatch(body: Mapping[str, Any]) -> str | None:
    """T-0085: the 400 text vLLM (via ``transformers``) serves when a ``continue_final_message`` request's stripped
    final text does not appear in the template's render of the messages with the sentinel appended; ``None`` when
    the continuation survives the template (or the request does not continue)."""
    msgs = list(body.get("messages") or [])
    if not body.get("continue_final_message") or not msgs:
        return None
    final = _final_text(msgs[-1].get("content"))
    if final is None:
        return "continue_final_message is set but we could not find any text to continue in the final message!"
    rendered = _render(body, _with_sentinel(msgs), True)
    if final.strip() not in rendered or CONTINUE_SENTINEL.strip() not in rendered:
        return f"{TEMPLATE_MISMATCH}Final message to continue: {final}"
    return None


def render_prompt(body: Mapping[str, Any], *, with_tools: bool = True) -> str:
    """T-0084c: the prompt text of a chat request as vLLM renders it with the pinned chat template (see the module
    docstring): ``messages``, ``tools`` (unless ``with_tools`` is False), ``enable_thinking``, the generation prompt
    unless the request continues its final message. T-0085: a ``continue_final_message`` render is cut as
    ``transformers`` cuts it -- the final text + ``CONTINUE_SENTINEL`` rendered, cut at the sentinel, right-stripped
    when the template trimmed the space after it. A continuation the template changes raises ``ValueError``
    (``template_mismatch``; the server refuses it with 400 first)."""
    msgs = list(body.get("messages") or [])
    cont = bool(body.get("continue_final_message"))
    if not (cont and msgs):
        return _render(body, msgs, with_tools)
    bad = template_mismatch(body)
    if bad is not None:
        raise ValueError(bad)
    text = _render(body, _with_sentinel(msgs), with_tools)
    tag = CONTINUE_SENTINEL.strip()
    loc = text.rindex(tag)
    if text[loc : loc + len(CONTINUE_SENTINEL)] == CONTINUE_SENTINEL:
        return text[:loc]
    return text[:loc].rstrip()


def cd_keys(body: Mapping[str, Any]) -> dict[str, Any]:
    xargs = body.get("vllm_xargs") or {}
    return {k: v for k, v in xargs.items() if k.startswith("astra_cd_")}


def _payload(pid: str, reasoning: str | None, content: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": pid,
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


class FakeVllmServer:
    """Context manager: ``with FakeVllmServer(script, tmp_path) as srv: run(base_url=srv.url, ...)``."""

    def __init__(
        self,
        script: Script,
        workdir: Path,
        *,
        encode: Callable[[str], Sequence[int]] | None = None,
        parse_xargs: Callable[[Mapping[str, Any]], Any] | None = None,
        decoys: bool = True,
        metrics: bool = True,
        drop_declarations: bool = False,
    ) -> None:
        self.script = script
        #: T-0084c: a server started with ``--exclude-tools-when-tool-choice-none``
        self.drop_declarations = drop_declarations
        #: T-0084c: per scripted request (``requests`` order), the prompt ids served (``None``: none served)
        self.prompt_ids: list[list[int] | None] = []
        self.engine_log = Path(workdir) / "engine.log"
        self.engine_log.write_text("(EngineCore_DP0 pid=4242) INFO engine started\n")
        self.encode = encode
        self.parse_xargs = parse_xargs
        self.decoys = decoys
        #: T-0080b: serve ``GET /metrics`` (False: 404, the counters are unavailable)
        self.metrics = metrics
        self.prefix_queries_total = 0
        self.prefix_hits_total = 0
        #: T-0080b: requests refused with 400 (invalid ``continue_final_message``), never scripted
        self.rejected: list[dict[str, Any]] = []
        #: GET paths in arrival order (``/v1/models``, ``/health``, ...), dead or alive
        self.gets: list[str] = []
        self.dead = False
        self.posts_while_dead = 0
        self.requests: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        #: E9 flush requests (``is_flush``: the flush prompt alone, ``max_tokens`` 1): answered here, never scripted
        self.flushes: list[dict[str, Any]] = []
        self._held: list[tuple[int, list[str]]] = []
        self._lock = threading.Lock()
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._timers: list[threading.Timer] = []

    # ------------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeVllmServer:
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                with fake._lock:
                    fake.gets.append(self.path)
                if fake.dead:
                    self.close_connection = True  # no response: a dropped connection
                    return
                if self.path.rstrip("/") == "/health":
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path.rstrip("/") == "/metrics":
                    if not fake.metrics:
                        self._json(404, {"error": "not found"})
                        return
                    data = fake.metrics_text().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(data)
                elif self.path.rstrip("/").endswith("/models"):
                    self._json(
                        200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}
                    )
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if fake.dead:
                    with fake._lock:
                        fake.posts_while_dead += 1
                    self.close_connection = True  # no response: a dropped connection
                    return
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._json(404, {"error": "not found"})
                    return
                status, payload, stream = fake._handle(body)
                if status != 200:
                    self._json(status, payload)
                elif stream:
                    self._sse(payload)
                else:
                    self._json(200, payload)

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)

            def _sse(self, payload: Mapping[str, Any]) -> None:
                msg = payload["choices"][0]["message"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = []
                if msg.get("reasoning") is not None:
                    chunks.append({"reasoning": msg["reasoning"]})
                chunks.append({"content": msg["content"]})
                try:
                    for i, delta in enumerate(chunks):
                        ev = {
                            "id": payload["id"],
                            "object": "chat.completion.chunk",
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": "stop" if i == len(chunks) - 1 else None,
                                }
                            ],
                        }
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for t in self._timers:
            t.cancel()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    # ------------------------------------------------------------------ behaviour
    def die(self) -> None:
        """The engine dies now: every later request is dropped without a response (T-0071b C9)."""
        self.dead = True

    def metrics_text(self) -> str:
        """The Prometheus exposition of the two prefix-cache counters (vLLM's names and label style)."""
        labels = f'{{engine="0",model_name="{MODEL_ID}"}}'
        with self._lock:
            q, h = self.prefix_queries_total, self.prefix_hits_total
        return (
            "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{labels} 0.0\n"
            "# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.\n"
            "# TYPE vllm:prefix_cache_queries_total counter\n"
            f"vllm:prefix_cache_queries_total{labels} {float(q)}\n"
            "# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.\n"
            "# TYPE vllm:prefix_cache_hits_total counter\n"
            f"vllm:prefix_cache_hits_total{labels} {float(h)}\n"
        )

    @staticmethod
    def continuation_error(body: Mapping[str, Any]) -> str | None:
        """vLLM's validation of ``continue_final_message`` (``None`` when the request is acceptable)."""
        if not body.get("continue_final_message"):
            return None
        if body.get("add_generation_prompt", True):
            return "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        msgs = body.get("messages") or []
        if not msgs or msgs[-1].get("role") != "assistant":
            return (
                "continue_final_message is set but the final message is not an assistant message."
            )
        return template_mismatch(
            body
        )  # T-0085: what the pinned template deletes is refused, as vLLM does

    def cd_requests(self) -> list[dict[str, Any]]:
        return [cd_keys(b) for b in self.requests if cd_keys(b)]

    def cd_seeds(self) -> list[dict[str, Any] | None]:
        """T-0081: per constrained request (``cd_requests`` order), the seed it carried, decoded from the wire
        format (standard base64 of little-endian int32 ids), or ``None`` when it carried no ``astra_cd_draft``.
        Recording only: the fake's replies do not depend on the seed. Noise / start step read as sent, with the
        plugin's defaults (0.0 / 0) when absent."""
        out: list[dict[str, Any] | None] = []
        for keys in self.cd_requests():
            raw = keys.get("astra_cd_draft")
            if raw is None:
                out.append(None)
                continue
            data = base64.b64decode(raw, validate=True)
            ids = [v for (v,) in struct.iter_unpack("<i", data)]
            out.append(
                {
                    "draft": ids,
                    "draft_noise": keys.get("astra_cd_draft_noise", 0.0),
                    "draft_start_step": keys.get("astra_cd_draft_start_step", 0),
                }
            )
        return out

    def _release_held(self, *, all_: bool = False) -> None:
        """Count one arriving request against every held batch; write the batches that are due."""
        with self._lock:
            due: list[str] = []
            keep: list[tuple[int, list[str]]] = []
            for remaining, lines in self._held:
                if all_ or remaining <= 1:
                    due.extend(lines)
                else:
                    keep.append((remaining - 1, lines))
            self._held = keep
        if due:
            self._write(due)

    def _handle(self, body: dict[str, Any]) -> tuple[int, Any, bool]:
        self._release_held()  # E9: held records reach the log when the next request arrives
        if is_flush(body):
            with self._lock:
                self.flushes.append(body)
            return 200, _payload(f"flush-{len(self.flushes)}", None, "x"), bool(body.get("stream"))
        bad = self.continuation_error(body)
        if bad is not None:
            with self._lock:
                self.rejected.append(body)
            return (
                400,
                {"object": "error", "message": bad, "type": "BadRequestError", "code": 400},
                False,
            )
        with self._lock:
            index = len(self.requests)
            self.requests.append(body)
            self.prompt_ids.append(None)
        reply = self.script(index, body)
        if reply.http_status == 200:
            with self._lock:
                self.prefix_queries_total += int(reply.prefix_queries)
                self.prefix_hits_total += int(reply.prefix_hits)
        if reply.die_after:
            self.dead = True  # this reply still goes out; the next request finds the engine dead
        if reply.hang_s:
            time.sleep(reply.hang_s)
        if reply.http_status != 200:
            return (
                reply.http_status,
                {"error": {"message": "scripted failure", "code": reply.http_status}},
                False,
            )
        keys = cd_keys(body)
        text = reply.text
        if keys:
            error = reply.error
            if self.parse_xargs is not None and error is None:
                try:
                    self.parse_xargs(body.get("vllm_xargs") or {})
                except Exception as exc:  # the plugin's CdRequestError: loud, non-fatal
                    error = getattr(exc, "reason", "bad_value")
            if error is not None:
                text = ""  # the request ends on an EOS-first canvas (T-0067 crit. 7)
            if not reply.omit_records:
                lines = self._records(keys, reply, error)
                if reply.last_canvas_delay_s and len(lines) > 1:
                    last = [ln for ln in lines if f'"canvas_idx": {reply.canvases - 1}' in ln]
                    first = [ln for ln in lines if ln not in last]
                    self._write(first)
                    t = threading.Timer(reply.last_canvas_delay_s, self._write, args=(last,))
                    self._timers.append(t)
                    t.start()
                elif reply.hold_records:
                    with self._lock:
                        self._held.append((max(1, reply.hold_for_requests), lines))
                elif reply.record_delay_s:
                    t = threading.Timer(reply.record_delay_s, self._write, args=(lines,))
                    self._timers.append(t)
                    t.start()
                else:
                    self._write(lines)
        reasoning, content = split_served(text)
        if reply.served_content is not None:
            reasoning, content = reply.served_reasoning, reply.served_content
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning"] = reasoning
        payload = {
            "id": f"chatcmpl-{index}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        if body.get("return_token_ids") and self.encode is not None:
            # vLLM's ``return_token_ids``: the generated ids ride on the choice (special pieces included).
            served = [*self.encode(text), EOS] if text else [EOS]
            budget = body.get("max_tokens")
            if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
                served = served[
                    :budget
                ]  # T-0081b: a generation out of budget ends without a stop id
            payload["choices"][0]["token_ids"] = served
            if not reply.omit_prompt_token_ids:
                # T-0084c: vLLM serves the prompt's ids at the top level beside the choice's ``token_ids``
                prompt = self.prompt_token_ids(body, drop=reply.drop_declarations)
                if reply.prompt_ids_edit is not None:
                    prompt = [int(t) for t in reply.prompt_ids_edit(list(prompt))]
                payload["prompt_token_ids"] = prompt
                with self._lock:
                    self.prompt_ids[index] = list(prompt)
        return 200, payload, bool(body.get("stream"))

    def prompt_token_ids(self, body: Mapping[str, Any], *, drop: bool = False) -> list[int]:
        """T-0084c: the prompt ids the fake serves for ``body`` (before any ``Reply.prompt_ids_edit``)."""
        assert self.encode is not None, "prompt_token_ids need encode= (the HF tokenizer)"
        without = drop or (self.drop_declarations and body.get("tool_choice") == "none")
        return [int(t) for t in self.encode(render_prompt(body, with_tools=not without))]

    def _ids(self, reply: Reply, error: str | None) -> list[int]:
        if error is not None:
            return [EOS]
        if reply.committed_ids is not None:
            return list(reply.committed_ids)
        assert self.encode is not None, "a constrained reply needs encode= (the HF tokenizer)"
        text = reply.text if reply.committed_text is None else reply.committed_text
        return [*self.encode(text), EOS]

    def _records(self, keys: Mapping[str, Any], reply: Reply, error: str | None) -> list[str]:
        from local_astra.contracts.cd import CdCanvasRecord

        tag = keys.get("astra_cd_record_tag")
        mode = keys.get("astra_cd_mode")
        # the wire modes (xargs.MODES; T-0076 added perstep_mar_qargmax): a refused request's record names its
        # mode only when the mode itself parsed, as the plugin's ``_raw_tag_and_arm`` does
        if error is not None and mode not in WIRE_MODES:
            mode = None
        ids = self._ids(reply, error)
        n = max(1, reply.canvases if error is None else 1)
        cut = [len(ids) * i // n for i in range(n + 1)]
        if reply.canvas_steps is not None:
            assert len(reply.canvas_steps) >= n, "canvas_steps needs one counter per canvas"
        records = []
        for idx in range(n):
            chunk = ids[cut[idx] : cut[idx + 1]] or [EOS]
            last = idx == n - 1
            stop = next((i for i, t in enumerate(chunk) if t in STOP_IDS), None)
            is_ep = mode == "end_projection"
            rec = {
                "record_tag": tag if isinstance(tag, str) else None,
                "arm": mode,
                "canvas_idx": idx + 1 if (last and n > 1 and reply.gap_before_last_canvas) else idx,
                "committed_ids": chunk,
                "state_set_after": [3] if (reply.in_final or not last) else [5],
                "in_final": bool(reply.in_final) if last else False,
                "steps": int(reply.canvas_steps[idx])
                if reply.canvas_steps is not None
                else reply.steps,
                "bucket": reply.bucket,
                "stop_pos": stop,
                "error": error,
                "mean_accepted_per_step": None
                if (is_ep or mode is None)
                else reply.mean_accepted_per_step,
                "shadow_argmax_ids": (
                    list(reply.shadow_argmax_ids) if reply.shadow_argmax_ids is not None else chunk
                )
                if is_ep
                else None,
                "argmax_in_language": (
                    reply.argmax_in_language if reply.argmax_in_language is not None else True
                )
                if is_ep
                else None,
                "map_equals_argmax": (
                    reply.map_equals_argmax if reply.map_equals_argmax is not None else True
                )
                if is_ep
                else None,
            }
            CdCanvasRecord.model_validate(rec)  # the fake never emits an off-contract record
            records.append(rec)
        with self._lock:
            self.records.extend(records)
        lines = []
        for rec in reversed(records):  # out of canvas order on purpose
            if self.decoys and error is None:
                decoy = {
                    **rec,
                    "record_tag": f"decoy-{tag}"[:64],
                    "committed_ids": [EOS],
                    "stop_pos": 0,
                }
                lines.append(LOG_PREFIX + "ASTRA_CD " + json.dumps(decoy))
            lines.append(LOG_PREFIX + "ASTRA_CD " + json.dumps(rec))
        if reply.duplicate_last_canvas:
            dup = LOG_PREFIX + "ASTRA_CD " + json.dumps(records[-1])
            lines.append(dup)
        return lines

    def release(self) -> None:
        """Emit any held records (the engine flushing on request release, T-0067 S11)."""
        self._release_held(all_=True)

    def _write(self, lines: list[str]) -> None:
        with self._lock, self.engine_log.open("a") as fh:
            for line in lines:
                fh.write(line + "\n")
                fh.write(
                    "(EngineCore_DP0 pid=4242) INFO 09-30 12:00:00 [loggers.py:1] Engine 000: throughput\n"
                )
