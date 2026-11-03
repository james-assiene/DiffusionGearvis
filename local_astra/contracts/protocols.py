"""Service interfaces (docs/02 §5). Fakes in ``tests/fakes/`` and real clients both implement these.

They are ``typing.Protocol`` classes so a fake never has to inherit from production code, and
``runtime_checkable`` so the contract test can assert conformance with ``isinstance``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

from local_astra.contracts.cd import CdRequest
from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.llm import Commit, LlmOptions, PrefillHandle, Prompt
from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.speech import SttEvent, WordTiming
from local_astra.contracts.toolcall import SearchResults, ToolInvocation, ToolOutcome


@runtime_checkable
class TtsUtterance(Protocol):
    utterance_id: str

    def audio(self) -> AsyncIterator[bytes]:
        """24 kHz mono s16le chunks; first chunk before the sentence has finished rendering."""
        ...

    def words(self) -> Sequence[WordTiming] | None:
        """Word timings when the backend has them, else None (the executor estimates)."""
        ...

    async def cancel(self) -> None:
        """Idempotent; no further chunks after it returns (≤ 100 ms)."""
        ...

    @property
    def first_audio_latency_ms(self) -> float | None: ...


@runtime_checkable
class TtsBackend(Protocol):
    async def speak(self, utterance_id: str, text: str, voice: str = "default") -> TtsUtterance: ...


@runtime_checkable
class AudioIO(Protocol):
    """Conductor-owned PipeWire capture/playback (docs/01 D7). 48 kHz, 10 ms blocks.

    The playback half is the set of seams T-0006 shipped on ``PipeWireAudioIO`` and
    ``tests/fakes/audio.py`` grew with it; T-0007 widens the protocol to match them (and nothing
    else — the capture half stays with T-0021). ``played_ms``/``played_text`` are the module's
    own answer about what reached the device: the session never recomputes them from TTS word
    timings, because that is how history acquires words nobody heard (docs/02 §4).
    """

    async def play(self, utterance: TtsUtterance, *, source: str | None = None) -> None:
        """Queue one utterance. ``source`` is the *committed* substring it renders (``""`` for a
        clip, which is never history text); it is what ``played_text`` is built from."""
        ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def flush(self) -> int:
        """Drop queued audio; return ``played_ms`` of the current utterance actually rendered."""
        ...

    def played_ms(self) -> float:
        """Audio of the current run actually rendered to the device.

        A float because the real module integrates a device clock (T-0006's ``PipeWireAudioIO``
        reports fractional milliseconds); the *wire* stays integral — ``PlaybackStopped.played_ms``
        and ``PlaybackDrained.played_ms`` in ``contracts/events.py`` are unchanged.
        """
        ...

    def mark_drain(self) -> None:
        """No further audio is coming for this run: a queue that merely empties is an underrun,
        and only a marked run may report ``playback.drained`` (docs/02 §4 spec delta 1)."""
        ...

    def playback_events(self) -> AsyncIterator[PlaybackStopped | PlaybackDrained]:
        """Terminal events of a run, with the module's own ``played_ms``/``played_text``."""
        ...

    def is_rendering(self, t_ms: float) -> bool:
        """True when the device is putting samples on the wire at ``t_ms``."""
        ...

    def capture(self) -> AsyncIterator[bytes]:
        """f32le mono 48 kHz blocks of 10 ms, echo-cancelled when AEC is on."""
        ...


@runtime_checkable
class SttClient(Protocol):
    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None: ...

    async def mark(self, t_eot_ms: int) -> int:
        """Ask the server to echo a ``Marker`` for ``t_eot_ms``; the id it is echoed with.

        The echo — and nothing else — becomes ``SttFlushedEvent`` on ``events()``, which is the
        ENDPOINTING → THINKING row's "every word up to ``t_eot_ms`` has been delivered"
        (docs/02 §4 row 6; T-0007a ruling A3 assigns this one line to T-0032). The server echoes
        only while audio keeps arriving, so the caller must go on feeding after the mark.
        """
        ...

    def events(self) -> AsyncIterator[SttEvent]: ...

    async def close(self) -> None: ...


@runtime_checkable
class LlmBackend(Protocol):
    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        """Warm the prefix cache; abortable via ``abort(request_id)``."""
        ...

    def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: CdRequest | None = None,
    ) -> AsyncIterator[Commit]:
        """Stream commits; thought/answer/ui segments already separated (and, with ``prompt.tools``, at most one
        ``tool`` commit, last; T-0088).

        ``xargs`` are per-request plugin arguments, forwarded verbatim under ``vllm_xargs`` by
        ``VllmClient.request_body`` (docs/02 §5). Keys and values are strings only — a deliberate
        narrowing of the wire (``LlmOptions.extra_body`` accepts any JSON scalar; T-0017 ruling A2)
        — and the engine's ``astra_request_id`` (what ``abort`` targets)
        always wins over a caller-supplied one. What any individual argument *means* is the
        business of the tap (T-0011) and of constrained decoding (docs/07), not of this protocol.

        ``cd`` (T-0088 criterion 3) is a typed constrained-decoding request: ``cd.to_xargs()`` joins ``xargs``
        under ``vllm_xargs`` with its JSON types (float, int, str) kept. A key given by both raises
        ``ValueError`` before anything is sent; ``astra_request_id`` still wins.
        """
        ...

    async def tokenize(self, text: str) -> list[int]:
        """The engine's ``/tokenize`` of ``text`` with ``add_special_tokens: false`` (T-0088 criterion 4).
        Never tokenised locally (H9)."""
        ...

    async def abort(self, request_id: str) -> None:
        """Idempotent; the sequence is freed within 200 ms (contract test checks /metrics)."""
        ...


@runtime_checkable
class PerceptionClient(Protocol):
    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        """Frame at (or just before) ``at_ms`` — speech onset by default — else the latest."""
        ...

    async def clip(
        self, source: Source, t0_ms: int, t1_ms: int, fps: float = 1.0
    ) -> Sequence[Frame]: ...

    def events(self) -> AsyncIterator[object]:
        """``SceneChanged`` events (contracts.events) and nothing else."""
        ...


@runtime_checkable
class MemoryClient(Protocol):
    async def ingest(self, episode: Episode) -> int: ...

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]: ...

    async def set_caption(self, episode_id: int, caption: str) -> None: ...

    async def forget_since(self, t0_ms: int) -> int: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Runs a v1 tool call (T-0091 criterion 7; T-0080d settled default 1). ``mode`` is the value of the session's
    ``tools`` switch it serves (``dry_run`` in v1; ``live`` is reserved and refused at startup)."""

    mode: str

    async def execute(self, inv: ToolInvocation) -> ToolOutcome: ...


class SearchFailed(Exception):
    """A ``SearchBackend.search`` that could not answer (T-0111b criterion 3). ``kind`` says why, and is also in
    ``str(exc)``: ``rate_limited`` (the provider refused for load), ``network`` (the connection failed),
    ``error`` (anything else, including "no results"), ``closed`` (the backend was closed)."""

    def __init__(
        self, kind: Literal["rate_limited", "network", "error", "closed"], detail: str = ""
    ) -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind: Literal["rate_limited", "network", "error", "closed"] = kind


@runtime_checkable
class SearchBackend(Protocol):
    """Answers a ``google_search`` call for the tool-result loop (T-0113; T-0111a §4). It returns every result it
    has, best first; bounding and sanitising are the renderer's (``toolcall/loop.py``).

    Failure convention (T-0111b): ``TimeoutError`` means the search timed out; ``SearchFailed(kind)`` means it
    failed for the reason ``kind`` names; any other ``Exception`` also means it failed; ``asyncio.CancelledError``
    always propagates and is never a failure. The fake is ``tests/fakes/search.py``; the real backend is
    ``local_astra/search/ddgs_backend.py``."""

    async def search(self, query: str) -> SearchResults: ...
