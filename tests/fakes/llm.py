"""Scripted fake of ``LlmBackend``: yields the commits it was given, honours abort.

``generate`` mirrors the real client's plugin-argument boundary (T-0017): ``xargs`` is
keyword-only, recorded for assertions, and passed through ``LlmOptions.extra_body`` — the same
function ``VllmClient.request_body`` uses — so a value the engine could not carry is rejected
here exactly as it is against the real service, instead of being silently accepted by the fake
and 400-ing on the GPU box.

It also mirrors the one *failure* the engine produces for a prompt it cannot accept (T-0020):
``fail_generates(...)`` scripts an exception per turn ``generate``, and ``http_error(413)`` builds
exactly the exception ``VllmClient`` raises there — ``response.raise_for_status()``'s own
``httpx.HTTPStatusError``, not a hand-rolled look-alike. Parity with the real client is pinned by
``tests/contract/test_over_long_prompt_error.py``. Caption requests (the idle captioner's) never
consume a scripted error: the script is about the *turn's* prompt.

T-0114 (the tool-result loop's step requests, T-0111a §6 "Fake LLM"): ``script_steps(*scripts)`` gives one stream
per **step** continuation, consumed in order. A step is told apart from a repair *structurally*, by the final
assistant text of a ``continue_final_message`` prompt: a step's ends in ``<tool_response|>`` (form G), a repair's in
``<|tool_call>`` (H-T84-2). Steps never consume ``script``, ``continuation``, ``generate_errors`` or
``continuation_errors``; they have their own ``step_errors`` (``fail_steps``), their own hold
(``hold_steps``/``release_steps``) and their own record (``step_prompts``, ``step_bodies``; the bodies are the same
objects ``bodies`` holds, built by the real ``VllmClient.request_body``). A step with no script left yields nothing
and is named in ``step_overruns``. Repair continuations, and every other prompt, behave exactly as before.

T-0171 (the engine's empty "retryable" response, ``tests/fakes/vllm_sse.py`` ``engine_empty``): ``engine_empty_error``
builds the exception the **real** ``VllmClient`` raises for that response, by running it in id mode against the wire
fake (so the fake cannot drift from the client's type or message; ``tests/contract/test_engine_empty_parity.py``).
An entry of ``fail_generates`` / ``fail_steps`` / ``continuation_errors`` may now also be a callable taking the engine
request id and returning the exception (or an awaitable of it), called at the moment the error is raised; pass
``engine_empty_error`` itself to get an error naming the attempt it ends. An exception or ``None`` behaves exactly as
before. ``fail_generates_after_script`` scripts an error raised *after* a turn generate has yielded its script (the
adversarial "commits, then the error" fake; the real client cannot produce an empty response after a commit);
``fail_steps_after_script`` is the same for step requests (review A-F1).

T-0172 group B (re-scope R1, the announced-call re-ask): a re-ask is one more request under the same machine request
id, built by ``announced_call.reask_prompt``: the step's prompt as sent, then the model's words, then a user turn whose
whole text is ``REASK_NUDGE``, never continuing the final message. ``is_reask_prompt`` tells it apart *structurally*
(that last user turn), so a re-ask never consumes ``script``, ``generate_errors``, ``step_scripts`` or any other
request's script or error (before R1 a non-continuation prompt replayed ``script``, so a step-1 re-ask could not be
scripted apart from the step it re-asks). ``script_reasks(*scripts)`` gives one stream per re-ask, consumed in order;
``fail_reasks`` / ``fail_reasks_after_script`` / ``hold_reasks`` / ``release_reasks`` mirror the step ones; every
re-ask is recorded in ``reask_prompts`` and ``reask_bodies`` (built by the real ``VllmClient.request_body``, so the
wire shape is the client's, not the fake's). A re-ask with no script left yields nothing and is named in
``reask_overruns``. Nothing changes for any other prompt: no earlier caller sends a prompt ending in the nudge.

T-0179 group B (additive): ``script_generates(*scripts)`` gives one stream per *turn* generate (a request that is not
a step, a re-ask, a continuation or a caption), consumed in order; once they are used up (or when none was given, the
default) a turn generate replays ``script`` exactly as before. It exists because a session that resends the identical
step-1 request (the degenerate retry) would otherwise be served the same ``script`` again and could never be shown a
clean retry. Scripted errors (``fail_generates``) are consumed first, as before: a request that raises consumes no
script.

T-0207 group A2 (additive): a completion check is one more request under the same machine request id, built by
``completion_check.check_prompt``: the step's prompt as sent, then the model's words, then a user turn holding exactly
``(ImagePart(<fresh frame>), TextPart(CHECK_INSTRUCTION or CHECK_INSTRUCTION_REPORT))``, never continuing the final
message. ``is_check_prompt`` tells it apart *structurally* (that last user turn), so a check never consumes ``script``,
``generate_scripts``, ``generate_errors`` or any step's or re-ask's script or error. Before this a check prompt was a
"turn generate" to the fake: a step-1 check could not be scripted apart from step 1, and nothing could hold it in
flight (the barge-in and timeout rows need that). ``script_checks`` / ``fail_checks`` / ``fail_checks_after_script`` /
``hold_checks`` / ``release_checks`` mirror the re-ask ones; every check is recorded in ``check_prompts`` and
``check_bodies`` (built by the real ``VllmClient.request_body``); a check with no script left yields nothing and is
named in ``check_overruns``. Nothing changes for any other prompt: no earlier caller sends a prompt ending in that turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import httpx

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import (
    VLLM_XARGS_KEY,
    Commit,
    ImagePart,
    LlmOptions,
    Message,
    PrefillHandle,
    Prompt,
    TextPart,
)
from local_astra.memory.captioner import CAPTION_PROMPT
from local_astra.toolcall import native
from local_astra.toolcall import tools as toolset
from local_astra.toolcall.special_tokens import SPECIAL_IDS, SPECIAL_TOKENS
from tests.fakes import vllm_sse

#: T-0171: a scripted failure is an exception, or a callable that makes one from the engine request id
ErrorFactory = Callable[[str], "BaseException | Awaitable[BaseException]"]
ScriptedError = BaseException | ErrorFactory | None


async def _materialise(entry: BaseException | ErrorFactory, request_id: str) -> BaseException:
    if isinstance(entry, BaseException):
        return entry
    made = entry(request_id)
    if inspect.isawaitable(made):
        made = await made
    assert isinstance(made, BaseException), f"an error factory returned {made!r}"
    return made


#: T-0088: id of the byte-fallback piece ``<0x00>`` in the pinned tokenizer (``<0xNN>`` = this + NN); the fake's
#: ``tokenize`` spells ordinary text in those pieces, so ``HostDetokenizer.decode(fake.tokenize(t)) == t``.
BYTE_FALLBACK_BASE = 238
_SPECIAL_ID_OF = {lit: i for i, lit in SPECIAL_IDS.items()}
_SPECIALS_LONGEST_FIRST = tuple(sorted(SPECIAL_TOKENS, key=len, reverse=True))


def _refusing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"FakeLlm delegated to the real client, which tried to send {request.method} {request.url}"
        )

    return httpx.MockTransport(handler)


def fake_tokenize(text: str) -> list[int]:
    """The fake's deterministic map: each special literal is its one id, every other byte its ``<0xNN>`` id."""
    out: list[int] = []
    i = 0
    while i < len(text):
        for lit in _SPECIALS_LONGEST_FIRST:
            if text.startswith(lit, i):
                out.append(_SPECIAL_ID_OF[lit])
                i += len(lit)
                break
        else:
            out.extend(BYTE_FALLBACK_BASE + b for b in text[i].encode())
            i += 1
    return out


#: The body vLLM 0.29.0 returns when the prompt is longer than ``--max-model-len``. The wording is
#: OpenAI-compatible ``context_length_exceeded``; what the Conductor may rely on is the *status*,
#: which is why the tests assert on that and only quote this for the log line.
OVER_LONG_DETAIL = (
    "This model's maximum context length is 16384 tokens. However, you requested 20481 tokens "
    "(20481 in the messages, 0 in the completion). Please reduce the length of the messages."
)

CHAT_COMPLETIONS_URL = "http://127.0.0.1:8001/v1/chat/completions"


def http_error(
    status: int = 413,
    *,
    detail: str = OVER_LONG_DETAIL,
    url: str = CHAT_COMPLETIONS_URL,
) -> httpx.HTTPStatusError:
    """The exception the real client raises for an over-long prompt.

    Built by letting ``httpx.Response.raise_for_status`` raise it — the very call
    ``VllmClient.generate`` makes after opening the stream — so the fake cannot drift from the
    real client's exception type, message or ``.response.status_code``.
    """
    request = httpx.Request("POST", url)
    response = httpx.Response(
        status,
        json={"object": "error", "message": detail, "type": "BadRequestError", "code": status},
        request=request,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise ValueError(f"HTTP {status} is not an error status")


#: T-0171: the vocabulary-subset fixture the id-mode reader decodes with (``tests/fixtures/llm/MANIFEST.md``); an
#: empty response decodes no id, but ``VllmClient`` refuses id mode without a detokeniser
_DETOK_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "vocab_subset_cc8d3a0c.json"
)
#: a 1x1 JPEG as an inline data URL: the turn that met the drift was the session's first image-bearing generate
ENGINE_EMPTY_IMAGE = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP////////////////////////////////////////////////////////////////////////////////////8AAAsIAAEAAQEBEQD/xAAUAAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAA/AH//2Q=="


@cache
def _subset_detokenizer() -> Any:
    from local_astra.toolcall.detok import HostDetokenizer

    tok = json.loads(_DETOK_FIXTURE.read_text())["tokenizer"]
    data = json.dumps(tok, ensure_ascii=False).encode()
    path = Path(tempfile.mkdtemp(prefix="t0171-detok-")) / "tokenizer.json"
    path.write_bytes(data)
    return HostDetokenizer(path, hashlib.sha256(data).hexdigest())


def engine_empty_prompt() -> Prompt:
    """A tools-on (id mode) prompt with one image: the shape of the request the drift miss ended."""
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(
                role="user",
                content=(
                    ImagePart(url=ENGINE_EMPTY_IMAGE, phash="0" * 16),
                    TextPart(text="what is on this screen"),
                ),
            ),
        ),
        prefix_len=1,
        tools=tuple(native.tools_field(toolset.load())),
    )


async def _engine_empty_attempt(attempt_id: str, *, with_role: bool) -> BaseException | None:
    wire = vllm_sse.Wire(streams=[vllm_sse.engine_empty(with_role=with_role)])
    client = VllmClient(transport=wire.transport(), detokenizer=_subset_detokenizer())
    try:
        async for commit in client.generate(engine_empty_prompt(), attempt_id):
            raise AssertionError(f"an empty response yielded a commit: {commit!r}")
    except AssertionError:
        raise
    except Exception as exc:
        return exc
    finally:
        await client.aclose()
    return None


def engine_empty_error(
    attempt_id: str = "engine-empty-1", *, with_role: bool = True
) -> BaseException:
    """The exception the real ``VllmClient`` raises for the engine's empty "retryable" response to ``attempt_id``.

    Built by running the client, in id mode, against ``vllm_sse.engine_empty`` over ``httpx.MockTransport`` (in a
    worker thread with its own event loop, so it can be called from inside a running test loop and from a
    ``fail_generates`` entry). Whatever type the client raises is returned as is: before T-0171 that is a plain
    ``LlmWireError``. A client that raises nothing is a named ``AssertionError``.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        exc = pool.submit(
            asyncio.run, _engine_empty_attempt(attempt_id, with_role=with_role)
        ).result()
    if exc is None:
        raise AssertionError(
            f"VllmClient ended the engine's empty response to {attempt_id!r} without raising"
        )
    return exc


def is_caption_prompt(prompt: Prompt) -> bool:
    """The idle captioner's prompt, told apart from a turn's by its instruction text."""
    return CAPTION_PROMPT[:40] in " ".join(m.text() for m in prompt.messages)


#: T-0114: the close of a rendered tool response; a step continuation's final assistant text ends in it.
STEP_TAIL = "<tool_response|>"


def is_step_prompt(prompt: Prompt) -> bool:
    """A tool-loop step request (T-0111a §1 form G): a continuation whose final assistant text ends in
    ``<tool_response|>``. A repair's ends in ``<|tool_call>`` and is not a step.

    T-0178 B1: the report step's continuation is the same text followed by ``toolcall.loop.REPORT_FACT`` (after the
    last tool response, any separator), so it is a step too. Read off the module at call time: before T-0178 the
    constant does not exist and this is exactly the T-0114 rule.

    T-0201 B2: a labelled live step's continuation is the same text followed by one screen reminder (a value of
    ``conductor.session.SCREEN_REMINDERS``, after the last tool response, any separator), so it is a step too. Read
    off the module at call time: before T-0201 B the table does not exist and this is exactly the T-0178 rule (so the
    base-tree differentials see what they saw)."""
    if not getattr(prompt, "continue_final_message", False) or not prompt.messages:
        return False
    text = prompt.messages[-1].text()
    if text.endswith(STEP_TAIL):
        return True
    if STEP_TAIL not in text:
        return False
    from local_astra.toolcall import loop as _loop

    after = text.rsplit(STEP_TAIL, 1)[1].strip()
    fact = getattr(_loop, "REPORT_FACT", None)
    if isinstance(fact, str) and fact.strip() and after == fact.strip():
        return True
    return bool(after) and after in _screen_reminders()


def _screen_reminders() -> frozenset[str]:
    """T-0201 B3: the stripped values of ``conductor.session.SCREEN_REMINDERS`` (empty before group B lands)."""
    from local_astra.conductor import session as _session

    table = getattr(_session, "SCREEN_REMINDERS", None)
    if not isinstance(table, Mapping):
        return frozenset()
    return frozenset(v.strip() for v in table.values() if isinstance(v, str) and v.strip())


def is_check_prompt(prompt: Prompt) -> bool:
    """T-0207 A2: a completion check (``completion_check.check_prompt``): not a continuation, and its last message is a
    user turn whose content is exactly ``(ImagePart, TextPart(<a check instruction>))``, right after an assistant
    message. Read off the module at call time (``False`` on a tree without it, so base-tree differentials are
    unchanged)."""
    if getattr(prompt, "continue_final_message", False) or len(prompt.messages) < 2:
        return False
    try:
        from local_astra.toolcall.completion_check import (
            CHECK_INSTRUCTION,
            CHECK_INSTRUCTION_REPORT,
        )
    except ImportError:
        return False
    last, before = prompt.messages[-1], prompt.messages[-2]
    content = tuple(last.content)
    return (
        last.role == "user"
        and before.role == "assistant"
        and len(content) == 2
        and isinstance(content[0], ImagePart)
        and isinstance(content[1], TextPart)
        and content[1].text in (CHECK_INSTRUCTION, CHECK_INSTRUCTION_REPORT)
    )


def is_reask_prompt(prompt: Prompt) -> bool:
    """T-0172 (R1): an announced-call re-ask (``announced_call.reask_prompt``): not a continuation, and its last
    message is a user turn whose whole text is ``REASK_NUDGE``, right after an assistant message."""
    if getattr(prompt, "continue_final_message", False) or len(prompt.messages) < 2:
        return False
    try:
        # imported here: the base-tree differentials run this fake over older ``local_astra`` trees without the module
        from local_astra.toolcall.announced_call import REASK_NUDGE
    except ImportError:
        return False
    last, before = prompt.messages[-1], prompt.messages[-2]
    return last.role == "user" and last.text() == REASK_NUDGE and before.role == "assistant"


class FakeLlm:
    def __init__(self, script: Sequence[Commit] = (), *, prefix_tokens: int = 1000) -> None:
        self.script = list(script)
        self.prefix_tokens = prefix_tokens
        self.aborted: list[str] = []
        self.prefills: list[tuple[str, Prompt]] = []
        self.generates: list[tuple[str, Prompt, LlmOptions]] = []
        #: plugin arguments of the most recent ``generate`` (None when the caller passed none)
        self.last_xargs: Mapping[str, str] | None = None
        #: one entry per ``generate``, aligned with ``generates``
        self.xargs_log: list[Mapping[str, str] | None] = []
        #: what the real client would put under ``vllm_xargs`` for that call (request id merged in)
        self.last_extra_body: dict[str, Any] = {}
        #: one entry per *turn* ``generate``, consumed in order; ``None`` means "succeed" (T-0171: or a factory)
        self.generate_errors: list[ScriptedError] = []
        #: T-0171: one entry per *turn* ``generate``, consumed at its start, raised after its script was yielded
        self.generate_errors_after: list[ScriptedError] = []
        #: per-prefill ``usage.prompt_tokens`` (see ``prefill_counts``); empty = ``prefix_tokens``
        self.prefix_tokens_script: list[int] = []
        #: ``(request_id, prefix_tokens)`` the fake reported, one entry per prefill
        self.prefill_returns: list[tuple[str, int]] = []
        #: T-0088: the ``cd`` request of each ``generate`` (None when the caller passed none), aligned with
        #: ``generates``; ``last_cd`` is the most recent
        self.last_cd: Any = None
        self.cd_log: list[Any] = []
        #: T-0088: ``Prompt.tools`` of each ``generate`` (None for tools off), aligned with ``generates``
        self.tools_log: list[Any] = []
        #: T-0088: the body the real client would send for each ``generate``, built *by* the real
        #: ``VllmClient.request_body`` (closes by delegation: a field the client adds — ``tools``, the
        #: arm-(b) fields, the continuation flags — reaches this log without the fake naming it)
        self.bodies: list[dict[str, Any]] = []
        #: T-0088: every ``tokenize`` call's text
        self.tokenized: list[str] = []
        #: T-0092c: the script of a *continuation* request (``prompt.continue_final_message``: the H-T84-2 repair).
        #: ``None`` (the default) means continuations replay ``script`` like any other request, so every row written
        #: before T-0092c sees exactly the stream it saw before.
        self.continuation: list[Commit] | None = None
        #: T-0092c: one entry per continuation ``generate``, consumed in order (``None`` = succeed); raised where the
        #: real client raises, at the first ``__anext__``. Continuations never consume ``generate_errors``.
        self.continuation_errors: list[ScriptedError] = []
        #: T-0092c: raised by the next ``tokenize`` (after the text is recorded), then cleared
        self.tokenize_error: BaseException | None = None
        #: T-0114: one script per step continuation (``script_steps``), consumed in order; ``None`` until scripted
        self.step_scripts: list[list[Commit]] | None = None
        #: T-0114: one entry per step ``generate``, consumed in order (``None`` = succeed), raised at the first
        #: ``__anext__`` like the real client's HTTP and transport errors
        self.step_errors: list[ScriptedError] = []
        #: T-0171: one entry per step ``generate``, consumed at its start, raised after its step script was yielded
        self.step_errors_after: list[ScriptedError] = []
        #: T-0114: ``(engine id, prompt)`` of every step request, in order, and the body the real client would send
        self.step_prompts: list[tuple[str, Prompt]] = []
        self.step_bodies: list[dict[str, Any]] = []
        #: T-0114: engine ids of step requests that found no script left (they yielded nothing)
        self.step_overruns: list[str] = []
        #: T-0172 (R1): one script per announced-call re-ask, consumed in order (empty: a re-ask yields nothing)
        self.reask_scripts: list[list[Commit]] = []
        #: T-0172 (R1): one entry per re-ask ``generate``, consumed in order (``None`` = succeed), raised at the first
        #: ``__anext__``; and one raised after the re-ask's script was yielded
        self.reask_errors: list[ScriptedError] = []
        self.reask_errors_after: list[ScriptedError] = []
        #: T-0172 (R1): ``(engine id, prompt)`` of every re-ask, in order, the body the real client would send, and
        #: the engine ids of re-asks that found no script left
        self.reask_prompts: list[tuple[str, Prompt]] = []
        self.reask_bodies: list[dict[str, Any]] = []
        self.reask_overruns: list[str] = []
        #: T-0179: one script per turn generate (``script_generates``), consumed in order; empty: ``script``
        self.generate_scripts: list[list[Commit]] = []
        #: T-0207 A2: the completion check's route, mirroring the re-ask's (module docstring)
        self.check_scripts: list[list[Commit]] = []
        self.check_errors: list[ScriptedError] = []
        self.check_errors_after: list[ScriptedError] = []
        self.check_prompts: list[tuple[str, Prompt]] = []
        self.check_bodies: list[dict[str, Any]] = []
        self.check_overruns: list[str] = []
        self._check_flow = asyncio.Event()
        self._check_flow.set()
        self._reask_flow = asyncio.Event()
        self._reask_flow.set()
        self._step_flow = asyncio.Event()
        self._step_flow.set()
        self._continuation_flow = asyncio.Event()
        self._continuation_flow.set()
        self._prefill_flow = asyncio.Event()
        self._prefill_flow.set()
        self._active: set[str] = set()

    def fail_generates(self, *errors: ScriptedError) -> None:
        """Script the outcome of the next turn generates: an exception, or ``None`` to succeed.

        Raised where the real client raises — at the first ``__anext__``, i.e. when the stream is
        opened and ``raise_for_status`` runs — so a caller that never iterates never sees it.
        T-0171: an entry may be a factory ``(engine request id) -> exception`` (e.g. ``engine_empty_error``).
        """
        self.generate_errors = list(errors)

    def script_generates(self, *scripts: Sequence[Commit]) -> None:
        """T-0179: what each turn generate yields, in order (``scripts[0]`` for the next one); afterwards ``script``."""
        self.generate_scripts = [list(s) for s in scripts]

    def fail_generates_after_script(self, *errors: ScriptedError) -> None:
        """T-0171: the next turn generates yield their script, then raise (``None``: end normally). Adversarial: the
        real client never ends an id-mode response "empty" after it yielded a commit."""
        self.generate_errors_after = list(errors)

    def prefill_counts(self, *counts: int) -> None:
        """Script ``usage.prompt_tokens`` per prefill; the **last value sticks** once exhausted.

        The engine counts each prefix it is given, so one number for the whole session makes a
        per-turn reset of that number unobservable (T-0020 ruling A7): a stale over-budget count
        could trim the *next* turn's history and every test would stay green. ``prefill_returns``
        records what each prefill was actually told, so a test can state its regime instead of
        assuming how many prefills a turn issued.
        """
        self.prefix_tokens_script = list(counts)

    def hold_prefill(self) -> None:
        """Keep every later prefill in flight until ``release_prefill``.

        The anticipatory prefill is aborted at ``llm.start`` by design (docs/01 D9), so a turn
        whose prefill has not answered yet — the normal case on a short question — reports no
        count at all. That is the only situation in which a *stale* count from the previous turn
        can still be lying around, which is what ruling A7 is about.
        """
        self._prefill_flow.clear()

    def release_prefill(self) -> None:
        self._prefill_flow.set()

    # --- T-0092c: the repair's continuation request -----------------------------------------------

    def script_continuation(self, *commits: Commit) -> None:
        """What a continuation request yields (T-0088 criterion 7's continuation case: one ``tool`` commit).

        A continuation is told apart by ``prompt.continue_final_message`` -- the flag the real client turns into
        ``continue_final_message: true`` on the wire -- never by hashing or comparing the ``Prompt`` (a tools
        ``Prompt`` is unhashable, T-0088 note). Nothing in this fake hashes a ``Prompt``.
        """
        self.continuation = list(commits)

    def hold_continuation(self) -> None:
        """Keep every later continuation in flight, before its first commit, until ``release_continuation``.

        The repair timeout (R6) and a barge-in during the repair need a repair that is still live; an abort or a
        cancel while held ends it the way it ends the real stream."""
        self._continuation_flow.clear()

    def release_continuation(self) -> None:
        self._continuation_flow.set()

    # --- T-0114: the tool loop's step requests ------------------------------------------------------------------

    def script_steps(self, *scripts: Sequence[Commit]) -> None:
        """What each step continuation yields, in order: ``scripts[0]`` for the first step request, and so on."""
        self.step_scripts = [list(s) for s in scripts]

    def fail_steps(self, *errors: ScriptedError) -> None:
        """Script the outcome of the next step generates: an exception, or ``None`` to succeed."""
        self.step_errors = list(errors)

    def fail_steps_after_script(self, *errors: ScriptedError) -> None:
        """T-0171 (review A-F1): the next step generates yield their step script, then raise (``None``: end normally).
        Adversarial, as ``fail_generates_after_script``."""
        self.step_errors_after = list(errors)

    def hold_steps(self) -> None:
        """Keep every later step request in flight, before its first commit, until ``release_steps``."""
        self._step_flow.clear()

    def release_steps(self) -> None:
        self._step_flow.set()

    # --- T-0172 (R1): the announced-call re-ask --------------------------------------------------------------

    def script_reasks(self, *scripts: Sequence[Commit]) -> None:
        """What each re-ask yields, in order: ``scripts[0]`` for the first re-ask request, and so on."""
        self.reask_scripts = [list(s) for s in scripts]

    def fail_reasks(self, *errors: ScriptedError) -> None:
        """Script the outcome of the next re-asks: an exception (or a factory), or ``None`` to succeed."""
        self.reask_errors = list(errors)

    def fail_reasks_after_script(self, *errors: ScriptedError) -> None:
        """The next re-asks yield their script, then raise (``None``: end normally)."""
        self.reask_errors_after = list(errors)

    def hold_reasks(self) -> None:
        """Keep every later re-ask in flight, before its first commit, until ``release_reasks``."""
        self._reask_flow.clear()

    def release_reasks(self) -> None:
        self._reask_flow.set()

    # --- T-0207 A2: the completion check -----------------------------------------------------------------

    def script_checks(self, *scripts: Sequence[Commit]) -> None:
        """What each completion check yields, in order: ``scripts[0]`` for the first check request, and so on."""
        self.check_scripts = [list(s) for s in scripts]

    def fail_checks(self, *errors: ScriptedError) -> None:
        """Script the outcome of the next checks: an exception (or a factory), or ``None`` to succeed."""
        self.check_errors = list(errors)

    def fail_checks_after_script(self, *errors: ScriptedError) -> None:
        """The next checks yield their script, then raise (``None``: end normally)."""
        self.check_errors_after = list(errors)

    def hold_checks(self) -> None:
        """Keep every later check in flight, before its first commit, until ``release_checks``."""
        self._check_flow.clear()

    def release_checks(self) -> None:
        self._check_flow.set()

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        self.prefills.append((request_id, prefix))
        self._active.add(request_id)
        await self._prefill_flow.wait()
        tokens = self.prefix_tokens
        if self.prefix_tokens_script:
            tokens = (
                self.prefix_tokens_script.pop(0)
                if len(self.prefix_tokens_script) > 1
                else self.prefix_tokens_script[0]
            )
        self.prefill_returns.append((request_id, tokens))
        return PrefillHandle(request_id=request_id, prefix_tokens=tokens)

    async def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        self.generates.append((request_id, prompt, options or LlmOptions()))
        self.last_xargs = dict(xargs) if xargs is not None else None
        self.xargs_log.append(self.last_xargs)
        self.last_cd = cd
        self.cd_log.append(cd)
        self.tools_log.append(getattr(prompt, "tools", None))
        # Same boundary as VllmClient.request_body: plugin args plus the request id, validated
        # by the contract's JSON-scalar rule before anything is "sent".
        merged: dict[str, Any] = dict(xargs or {})
        if cd is not None:
            # T-0088 criterion 3: the typed CD keys join under vllm_xargs with their JSON types; a key in both
            # is refused before anything is sent
            typed = dict(cd.to_xargs())
            clash = sorted(set(merged) & set(typed))
            if clash:
                raise ValueError(f"vllm_xargs keys given both in xargs and by cd: {clash}")
            merged.update(typed)
        merged["astra_request_id"] = request_id
        self.last_extra_body = dict(LlmOptions.extra_body(merged)[VLLM_XARGS_KEY])
        if getattr(prompt, "tools", None) is not None or getattr(
            prompt, "continue_final_message", False
        ):
            # by delegation: exactly what the real client would put on the wire (only for prompts T-0088 adds,
            # so every earlier caller's prompt, video parts included, is untouched)
            body = VllmClient(transport=_refusing_transport()).request_body(
                prompt,
                request_id,
                options,
                {k: v for k, v in merged.items() if k != "astra_request_id"},
            )
            self.bodies.append(body)
        step = is_step_prompt(prompt)
        if step:
            self.step_prompts.append((request_id, prompt))
            self.step_bodies.append(body)
        reask = not step and is_reask_prompt(prompt)
        check = not step and not reask and is_check_prompt(prompt)
        if check:
            self.check_prompts.append((request_id, prompt))
            self.check_bodies.append(body if prompt.tools is not None else {})
        if reask:
            self.reask_prompts.append((request_id, prompt))
            # a re-ask with no tools (never sent by a correct session, B2) records an empty body, so the row that
            # compares bodies fails on its own assertion instead of a NameError here
            self.reask_bodies.append(body if prompt.tools is not None else {})
        continuation = bool(getattr(prompt, "continue_final_message", False)) and not step
        after: ScriptedError = None
        if (
            not step
            and not reask
            and not check
            and not continuation
            and self.generate_errors_after
            and not is_caption_prompt(prompt)
        ):
            after = self.generate_errors_after.pop(0)
        if step and self.step_errors_after:
            after = self.step_errors_after.pop(0)
        if reask and self.reask_errors_after:
            after = self.reask_errors_after.pop(0)
        if check and self.check_errors_after:
            after = self.check_errors_after.pop(0)
        if check:
            if self.check_errors:
                failed = self.check_errors.pop(0)
                if failed is not None:
                    raise await _materialise(failed, request_id)
        elif reask:
            if self.reask_errors:
                failed = self.reask_errors.pop(0)
                if failed is not None:
                    raise await _materialise(failed, request_id)
        elif step:
            if self.step_errors:
                failed = self.step_errors.pop(0)
                if failed is not None:
                    raise await _materialise(failed, request_id)
        elif continuation:
            if self.continuation_errors:
                failed = self.continuation_errors.pop(0)
                if failed is not None:
                    raise await _materialise(failed, request_id)
        elif self.generate_errors and not is_caption_prompt(prompt):
            scripted = self.generate_errors.pop(0)
            if scripted is not None:
                raise await _materialise(scripted, request_id)
        self._active.add(request_id)
        script = self.script
        if check:
            await self._check_flow.wait()
            if request_id not in self._active:
                return  # aborted while held: the check's script is not consumed
            if self.check_scripts:
                script = self.check_scripts.pop(0)
            else:
                self.check_overruns.append(request_id)
                script = []
        elif reask:
            await self._reask_flow.wait()
            if request_id not in self._active:
                return  # aborted while held: the re-ask's script is not consumed
            if self.reask_scripts:
                script = self.reask_scripts.pop(0)
            else:
                self.reask_overruns.append(request_id)
                script = []
        elif step:
            await self._step_flow.wait()
            if request_id not in self._active:
                return  # aborted while held: the step's script is not consumed
            if self.step_scripts:
                script = self.step_scripts.pop(0)
            else:
                self.step_overruns.append(request_id)
                script = []
        elif continuation:
            await self._continuation_flow.wait()
            if self.continuation is not None:
                script = self.continuation
        elif self.generate_scripts and not is_caption_prompt(prompt):
            script = self.generate_scripts.pop(0)  # T-0179: a turn generate's own script
        for c in script:
            if request_id not in self._active:
                return
            await asyncio.sleep(0)
            yield c.model_copy(update={"request_id": request_id})
        self._active.discard(request_id)
        if after is not None:
            raise await _materialise(after, request_id)

    async def tokenize(self, text: str) -> list[int]:
        """T-0088 criterion 4: the engine's ``/tokenize`` (``add_special_tokens: false``), as a deterministic map."""
        self.tokenized.append(text)
        if self.tokenize_error is not None:
            err, self.tokenize_error = self.tokenize_error, None
            raise err
        return fake_tokenize(text)

    @staticmethod
    def tool_commit(text: str, *, stop_id: int | None = 50, canvas_idx: int = 0) -> Commit:
        """A scriptable ``tool`` commit (T-0088 criterion 7): put it last in ``script``."""
        from pydantic import ValidationError

        try:
            return Commit(  # type: ignore[call-arg]
                request_id="?",
                canvas_idx=canvas_idx,
                segment="tool",
                text=text,
                stop_id=stop_id,  # type: ignore[arg-type]
            )
        except ValidationError as exc:
            raise AssertionError(
                f"T-0088 criterion 1: Commit cannot carry segment='tool' with stop_id yet ({exc.error_count()} errors)"
            ) from exc

    async def abort(self, request_id: str) -> None:
        # idempotent: recorded every time, harmless when already gone
        self.aborted.append(request_id)
        self._active.discard(request_id)

    @property
    def active(self) -> frozenset[str]:
        return frozenset(self._active)


# --- T-0052: replaying a *recorded* generation over the real client ------------------------------
#
# The fence-carry rows do not want a fake backend: they want the real ``VllmClient`` reading a
# stream that says exactly what the model said on 2026-09-16 (the turn that put A2UI JSON into
# the speakers). What stands in for the engine is an ``httpx.MockTransport``, so everything from
# the SSE line parser down is the shipping code. These helpers live beside ``FakeLlm`` because
# they are the other half of "what the LLM looks like to a test", and because the fixture they
# read is this task's (docs/tasks/T-0052.md, *Owned files*).

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FLIGHT_CANVASES_PATH = FIXTURES / "llm" / "flight_canvases.json"

#: The fields ``Commit`` carried before T-0052 (contracts/llm.py, T-0005). Anything else a commit
#: reports is the fence lifecycle criterion 12 asks for — read by *difference* so the rows pin the
#: distinction and leave the implementer the spelling.
BASE_COMMIT_FIELDS: frozenset[str] = frozenset(
    {"request_id", "canvas_idx", "segment", "text", "thought_open"}
)


def flight_canvases() -> dict[str, Any]:
    """The recorded turn ``t6``: canvases, the pieces the log stored, and what TTS said."""
    return json.loads(FLIGHT_CANVASES_PATH.read_text())


def sse_body(canvases: Sequence[tuple[str | None, str | None]]) -> str:
    """One OpenAI SSE chunk per committed canvas (docs/02 §2.4/§5), then the sentinel.

    ``canvases`` is ``(reasoning_delta, content_delta)`` per canvas; ``None`` omits the key, which
    is what vLLM does for a channel that produced nothing in that chunk.
    """
    lines = []
    for reasoning, content in canvases:
        delta: dict[str, str] = {}
        if reasoning is not None:
            delta["reasoning"] = reasoning
        if content is not None:
            delta["content"] = content
        lines.append(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n")
    return "".join(lines) + "data: [DONE]\n\n"


def sse_transport(
    canvases: Sequence[tuple[str | None, str | None]],
    *,
    terminate: bool = True,
) -> httpx.MockTransport:
    """A transport that replays ``canvases`` as one chat-completions stream.

    ``terminate=False`` drops the ``[DONE]`` sentinel: the stream simply ends, which is what a
    server that died mid-generation looks like to the client (criterion 7's "stream close").

    Requests to any other path raise instead of being answered: a test double that quietly serves
    a URL nobody meant to call is how a client change escapes its tests (.claude/rules/tests.md,
    "a test double closes by delegation, not by enumeration").
    """
    body = sse_body(canvases)
    if not terminate:
        body = body[: -len("data: [DONE]\n\n")]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions"), (
            f"the recorded-canvas transport only serves chat completions; got {request.url}"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    return httpx.MockTransport(handler)


def fence_lifecycle(commit: Commit) -> tuple[tuple[str, str], ...]:
    """Everything a commit says beyond T-0005's five fields, as a comparable key.

    Criterion 12 asks the ``Commit`` contract to distinguish six fence endings; it deliberately
    does not name the field or its values, so the rows compare *this* key across scenarios rather
    than asserting a spelling. Today it is ``()`` for every commit, which is exactly the finding:
    the six endings are indistinguishable.
    """
    extra = {k: v for k, v in commit.model_dump().items() if k not in BASE_COMMIT_FIELDS}
    return tuple(sorted((k, repr(v)) for k, v in extra.items()))


def carried_ui_chars(texts: Sequence[str]) -> int:
    """How many characters of fenced text one carry produced, under either emission shape.

    A client that re-emits the growing carry (canvas 2's commit carries canvas 1's text plus its
    own) and one that emits each canvas's slice must be measured the same way, or criterion 6's
    numeric bound would mean two different things: a text that starts with the previous one is
    counted only for what it added.
    """
    total = 0
    grown = ""
    for text in texts:
        if grown and text.startswith(grown):
            total += len(text) - len(grown)
        else:
            total += len(text)
        grown = text
    return total
