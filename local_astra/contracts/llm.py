"""LLM-side contracts: sampling whitelist, prompt parts, canvas commits.

Facts these encode (docs/01 D2/D3, docs/10 §9, .claude/rules/gpu-services.md):

- vLLM v0.29.0 diffusion requests accept only ``top_k``, ``top_p``, ``max_tokens`` and ``stop``
  per request; ``temperature != 1``, ``seed``, ``min_p``, ``logit_bias``, ``response_format`` and
  structured outputs return HTTP 400. Canvas length and denoising steps are engine-global
  (``--diffusion-config``), never per request.
- Plugin arguments travel in ``extra_body["vllm_xargs"]`` and surface as
  ``SamplingParams.extra_args`` inside the sampler (docs/07 §5).
- A *commit* is the unit of text the Conductor may act on: one denoised canvas, split by the
  ``gemma4`` reasoning parser into ``thought`` / ``answer`` segments, plus ``ui`` for the A2UI
  channel. ``thought_open`` is true when the canvas ends inside an unclosed thought; such text
  never reaches TTS (docs/11 §2.1).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from local_astra.contracts.base import Contract
from local_astra.contracts.perception import AssistFlags
from local_astra.toolcall.special_tokens import STOP_IDS

#: ``tool`` (T-0088): everything from the first call tag to the stop of a tools-on response, one commit at the
#: end of the request. It carries ``stop_id`` and never reaches speech (the speech gate refuses it).
Segment = Literal["thought", "answer", "ui", "tool"]

#: How the UI fence of a ``ui`` commit ended (T-0052 criterion 12 / D4). A surface that arrived in
#: pieces and was reassembled, one that was abandoned and one that never existed are three
#: different things, and every consumer downstream of the client showed the same thing for all
#: three:
#:
#: - ``"written"`` — the model wrote ``</a2ui>``;
#: - ``"inferred"`` — the client closed the fence at the structural end of the JSON because the
#:   model never wrote the marker (T-0052 criterion 5; the 2026-09-16 turn);
#: - ``"open"`` — the canvas (or the whole response) ended inside the fence; the text is published
#:   to the *shown* channel anyway (D2) and is never re-read as prose;
#: - ``"abandoned"`` — the carry hit ``UI_FENCE_MAX_CANVASES``/``UI_FENCE_MAX_CHARS``, or a second
#:   ``<a2ui>`` arrived while it was open (D3). The answer channel is live again after it.
FenceClose = Literal["written", "inferred", "open", "abandoned"]

#: Request-level sampling parameters vLLM's diffusion runner accepts. Everything else → 400.
SAMPLING_WHITELIST: frozenset[str] = frozenset({"top_k", "top_p", "max_tokens", "stop"})

#: Name of the OpenAI-compatible extra body key vLLM forwards to ``SamplingParams.extra_args``.
VLLM_XARGS_KEY = "vllm_xargs"


class LlmOptions(Contract):
    """Whitelisted per-request sampling options. Unknown fields raise at construction time."""

    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: tuple[str, ...] = ()

    @field_validator("stop")
    @classmethod
    def _stop_non_empty_strings(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not s for s in v):
            raise ValueError("stop sequences must be non-empty strings")
        if len(v) > 4:
            raise ValueError("at most 4 stop sequences")
        return v

    def to_request_fields(self) -> dict[str, Any]:
        """Fields to merge into the chat-completions body; only whitelisted, non-default ones."""
        out: dict[str, Any] = {}
        if self.top_k is not None:
            out["top_k"] = self.top_k
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.stop:
            out["stop"] = list(self.stop)
        assert set(out) <= SAMPLING_WHITELIST
        return out

    @staticmethod
    def extra_body(xargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """``extra_body`` for the OpenAI client: plugin arguments under ``vllm_xargs``.

        Values must be JSON scalars or lists thereof — vLLM only forwards simple types.
        """
        if not xargs:
            return {}
        for k, v in xargs.items():
            if not isinstance(v, str | int | float | bool | list):
                raise TypeError(f"vllm_xargs[{k!r}] must be a JSON scalar or list, got {type(v)}")
        return {VLLM_XARGS_KEY: dict(xargs)}


class TextPart(Contract):
    type: Literal["text"] = "text"
    text: str


class ImagePart(Contract):
    """OpenAI-style image part; ``url`` is a ``data:image/jpeg;base64,…`` URL for local frames."""

    type: Literal["image_url"] = "image_url"
    url: str
    #: perceptual hash of the frame, used for "unchanged frame is not re-sent" and for aging.
    phash: str | None = None


class VideoPart(Contract):
    """Clip as a list of frame data URLs (vLLM ``video`` content type; ≤ 32 frames)."""

    type: Literal["video_url"] = "video_url"
    frame_urls: tuple[str, ...]
    fps: float = 1.0


Part = TextPart | ImagePart | VideoPart
Role = Literal["system", "user", "assistant", "tool"]


class Message(Contract):
    role: Role
    content: tuple[Part, ...]

    def text(self) -> str:
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


#: chat_template_kwargs key that makes the Gemma 4 template emit ``<|think|>`` as the first
#: system-turn token. Session-level, never per request (prefix cache). Name verified in X3/T-0005.
THINKING_TEMPLATE_KWARG = "enable_thinking"


class Prompt(Contract):
    """A fully built prompt.

    ``prefix_len`` is the number of leading messages that form the *anticipatory prefill*
    prefix (docs/01 D9). The D9 boundary lies *inside* the last user turn: ``‖`` sits after the
    frame and before the query-dependent parts (TutorState, memory hits, assists, transcript).
    So the last prefix message is a ``user`` message whose content is the frame (and any text
    that does not depend on the query); ``full()`` *extends that same message* with more parts —
    it never appends a second consecutive user message (Gemma templates require alternation).
    Prefix caching then hits every block up to the frame; the last partial block misses.

    ``thinking`` is rendered by the chat template (``THINKING_TEMPLATE_KWARG``), not by a literal
    token in the system text. ``assists`` records which non-pixel assists were on for this turn.
    """

    messages: tuple[Message, ...]
    prefix_len: int = Field(ge=0)
    thinking: bool = True
    assists: AssistFlags = AssistFlags()
    #: T-0088: the native ``tools`` field (``native.tools_field`` order). Set, the request is tools-on (T-0087 arm
    #: (b)) and the client reads the stream by its token ids. Part of the prefix: q8 found the tools must be in the
    #: prefill.
    tools: tuple[dict[str, Any], ...] | None = None
    #: T-0088: continue the final assistant message (q9): it must be one ``TextPart`` of an ``assistant`` message,
    #: sent as a plain string with ``continue_final_message: true, add_generation_prompt: false``.
    continue_final_message: bool = False

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {THINKING_TEMPLATE_KWARG: self.thinking}

    def shares_prefix_with(self, other: Prompt) -> bool:
        """True when ``other`` (the final request) extends this prefill prompt per the D9 rule."""
        n = self.prefix_len
        if n == 0 or other.prefix_len != n or len(other.messages) < n:
            return False
        if self.tools != other.tools:
            return False  # the tools are rendered into the prefix (q8): different tools, different prefix
        if other.messages[: n - 1] != self.messages[: n - 1]:
            return False
        last, olast = self.messages[n - 1], other.messages[n - 1]
        return last.role == olast.role and olast.content[: len(last.content)] == last.content

    @field_validator("prefix_len")
    @classmethod
    def _prefix_within(cls, v: int, info: ValidationInfo) -> int:
        msgs = info.data.get("messages")
        if msgs is not None and v > len(msgs):
            raise ValueError("prefix_len exceeds number of messages")
        return v

    @model_validator(mode="after")
    def _continuation_shape(self) -> Prompt:
        if self.continue_final_message:
            last = self.messages[-1] if self.messages else None
            if (
                last is None
                or last.role != "assistant"
                or len(last.content) != 1
                or not isinstance(last.content[0], TextPart)
            ):
                raise ValueError(
                    "continue_final_message needs a final assistant message of exactly one TextPart"
                )
        return self

    def prefix(self) -> Prompt:
        return Prompt(
            messages=self.messages[: self.prefix_len],
            prefix_len=self.prefix_len,
            thinking=self.thinking,
            assists=self.assists,
            tools=self.tools,
        )


class Commit(Contract):
    """One committed canvas (or a piece of it after reasoning-parser split)."""

    request_id: str
    canvas_idx: int = Field(ge=0)
    segment: Segment
    text: str
    #: canvas ends inside an unclosed <|think|> … block; never voice (docs/11 §2.1)
    thought_open: bool = False
    #: how this ``ui`` commit's fence ended; ``None`` on ``thought``/``answer`` commits. Optional
    #: with a default on purpose: every existing constructor, fake and fixture stays valid
    #: (T-0052 criterion 12).
    fence_close: FenceClose | None = None
    #: this ``ui`` commit continues a fence opened in an *earlier* canvas of the same request
    fence_continued: bool = False
    #: T-0088: the request's stop id (the first id of ``STOP_IDS`` in the stream), on a ``tool`` commit only;
    #: ``None`` for a length cut and on every other segment. Strict (ruling E14): no bool, no numeric string.
    stop_id: int | None = Field(default=None, strict=True)
    #: T-0170: this ``thought`` commit's text came from an implicit thought (id mode: the response opened no
    #: ``<|channel>`` and its first channel marker was an orphan ``<channel|>``); ``True`` on a thought commit only.
    implicit_thought: bool = False

    @model_validator(mode="after")
    def _stop_id_on_tool_only(self) -> Commit:
        if self.stop_id is not None:
            if self.segment != "tool":
                raise ValueError("stop_id is set only on a tool commit")
            if self.stop_id not in STOP_IDS:
                raise ValueError(f"stop_id {self.stop_id} is not in STOP_IDS {sorted(STOP_IDS)}")
        return self

    @model_validator(mode="after")
    def _implicit_thought_on_thought_only(self) -> Commit:
        if self.implicit_thought and self.segment != "thought":
            raise ValueError("implicit_thought is set only on a thought commit")
        return self


class PrefillHandle(Contract):
    request_id: str
    prefix_tokens: int = Field(ge=0)


class LlmRequestSpec(Contract):
    """What ``LlmBackend.generate`` is asked to do; kept for telemetry and the fake's replay log."""

    request_id: str
    prompt: Prompt
    options: LlmOptions = LlmOptions()
    xargs: dict[str, Any] = Field(default_factory=dict)
