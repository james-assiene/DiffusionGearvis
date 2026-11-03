"""Idle captioner: DiffusionGemma writes captions when the session is idle > 10 s. Task T-0009.

Aborted on speech onset (``CaptionerAbort``), never pre-empts a live turn, one request at a time
(it is the third of ``--max-num-seqs 3``).

The API is ruling A1 in ``docs/tasks/T-0009.md``: ``IdleCaptioner(llm=…, memory=…)`` with
``run_once()`` and ``abort()``. The captioner owns no clock and no idle timer — the conductor
decides *when* the session has been idle long enough and calls ``run_once()``, and turns speech
onset into ``abort()``. That keeps every time-based decision in the turn machine.

Two properties this module exists to guarantee, both of them things a simpler captioner gets
wrong:

- **Abort means abandoned.** ``abort()`` calls ``LlmBackend.abort`` so vLLM frees the sequence
  *and* cancels the local consumer, so a backend that keeps streaming (or a canvas that was
  already in flight) cannot write a caption after the user started talking. The store is never
  touched once ``abort()`` has been called for a request.
- **Only spoken/answer text is stored.** Thought segments are dropped, not stripped later: a
  caption is search-index text, and CLAUDE.md is absolute that assistant thinking is never
  stored. The same rule is why ``thought_open`` canvases contribute nothing.
"""

from __future__ import annotations

import asyncio
import base64
from itertools import count
from pathlib import Path
from typing import Protocol

from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart
from local_astra.contracts.memory import Episode
from local_astra.contracts.protocols import LlmBackend

IDLE_BEFORE_CAPTION_MS = 10_000
CAPTION_PROMPT = (
    "Describe this screenshot in one line for later search: app, document title, what is visible. "
    "Then list up to eight distinctive words that appear on screen."
)
#: A caption is one line plus a word list; a long request would hold the third sequence slot for
#: no benefit and delay the abort that frees it.
CAPTION_MAX_TOKENS = 192


class CaptionStore(Protocol):
    """The slice of the store the captioner uses: ``MemoryClient.set_caption`` plus the pending
    query, which ruling A1 keeps on the concrete store rather than in the shared protocol."""

    async def next_uncaptioned(self) -> Episode | None: ...

    async def set_caption(self, episode_id: int, caption: str) -> None: ...


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class IdleCaptioner:
    """Captions one pending episode per call, single-flight, abortable. See the module docstring."""

    def __init__(self, llm: LlmBackend, memory: CaptionStore) -> None:
        self._llm = llm
        self._memory = memory
        self._ids = count(1)
        self._running = False
        self._request_id: str | None = None
        self._consumer: asyncio.Task[str] | None = None
        #: bumped by every ``abort()``. A run whose epoch is stale writes nothing — that is how a
        #: result produced by a backend that kept streaming after the abort is discarded.
        self._epoch = 0

    async def run_once(self) -> None:
        """Caption at most one uncaptioned episode. A no-op while a caption is already in flight."""
        if self._running:
            return  # single-flight: the captioner owns exactly one of the three sequence slots
        self._running = True
        try:
            epoch = self._epoch
            episode = await self._memory.next_uncaptioned()
            if episode is None or episode.id is None or epoch != self._epoch:
                return  # an onset during the lookup: never start a request the user pre-empted
            request_id = f"caption-{next(self._ids)}"
            self._request_id = request_id
            consumer = asyncio.create_task(self._caption(episode, request_id), name=request_id)
            self._consumer = consumer
            try:
                text = await consumer
            except asyncio.CancelledError:
                if epoch == self._epoch:
                    raise  # somebody cancelled *us*, not the abort path
                return
            if epoch != self._epoch or not text.strip():
                return
            await self._memory.set_caption(episode.id, text.strip())
        finally:
            self._request_id = None
            self._consumer = None
            self._running = False

    async def abort(self) -> None:
        """Speech onset (``CaptionerAbort``). Idempotent, and safe with nothing in flight."""
        self._epoch += 1
        # State is cleared before the await, not after it: a second `abort()` that arrives while
        # `llm.abort` is in flight (or after it raised) must find nothing to abort again.
        request_id, consumer = self._request_id, self._consumer
        self._request_id = None
        self._consumer = None
        try:
            if request_id is not None:
                await self._llm.abort(request_id)  # free the sequence within 200 ms
        finally:
            # …and stop reading, whatever the backend does next — including raise. A backend
            # whose `abort` fails must not leave a consumer running against a store the user has
            # started talking over, nor leave the captioner unable to run again; the exception
            # still propagates, once the local half of the abort is done (T-0009 A6).
            if consumer is not None and not consumer.done():
                consumer.cancel()

    async def _caption(self, episode: Episode, request_id: str) -> str:
        """Consume one generation and return the answer text; thought segments are dropped."""
        prompt = _caption_prompt(episode)
        options = LlmOptions(max_tokens=CAPTION_MAX_TOKENS)
        parts: list[str] = []
        # No explicit aclose: on abort this task is cancelled while suspended on the stream, and
        # the event loop's async-generator finalisation closes it. Freeing the *engine* slot is
        # `LlmBackend.abort`'s job, which `abort()` has already done by then.
        async for commit in self._llm.generate(prompt, request_id, options):
            if commit.segment == "answer" and not commit.thought_open:
                parts.append(commit.text)
        return "".join(parts)


def _caption_prompt(episode: Episode) -> Prompt:
    """One user turn: the thumbnail, then the instruction. Thinking stays on (session setting)."""
    thumbnail = _data_url(Path(episode.thumbnail_path).read_bytes())
    message = Message(
        role="user",
        content=(
            ImagePart(url=thumbnail, phash=episode.phash),
            TextPart(text=CAPTION_PROMPT),
        ),
    )
    return Prompt(messages=(message,), prefix_len=0)
