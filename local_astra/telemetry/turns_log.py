"""``turns.jsonl`` record shape — written by the Conductor, read by ``astra bench`` (docs/03 §7).

One line per turn. Numbers are meaningless without the regime, so every record carries it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from local_astra.config import DEFAULT_FRAME_INTERVAL_MS, DEFAULT_MAX_FRAMES_PER_TURN
from local_astra.contracts.base import Contract
from local_astra.contracts.perception import AssistFlags
from local_astra.telemetry.spans import SPAN_NAMES

#: T-0213 R-8: the three settings fields a turn line carries only when the settings were changed
SETTINGS_FIELDS: tuple[str, ...] = ("settings_rev", "system_sha12", "instructions_edited")
#: T-0227 A-9: the two frame settings, each dumped only when it differs from its default (``config``'s
#: ``DEFAULT_MAX_FRAMES_PER_TURN`` / ``DEFAULT_FRAME_INTERVAL_MS``)
FRAME_FIELD_DEFAULTS: dict[str, int] = {
    "max_frames_per_turn": DEFAULT_MAX_FRAMES_PER_TURN,
    "frame_interval_ms": DEFAULT_FRAME_INTERVAL_MS,
}


class Regime(Contract):
    """Everything needed to compare two numbers later (CLAUDE.md: state the regime)."""

    profile: str
    canvas_length: int = Field(ge=1)
    max_denoising_steps: int = Field(ge=1)
    thinking: bool
    vllm_version: str
    model: str
    stt: str
    tts: str
    audio_path: str = Field(description="headset | speakers+aec | loopback")
    assists: str = Field(default="pixels", description="AssistFlags.badge() at turn time")
    #: the same flags, structured, so the bench reader can filter without parsing the badge (Q9)
    assist_flags: AssistFlags = AssistFlags()
    git_commit: str = ""
    seed: int | None = None
    #: T-0092b criterion 5: the session's tool setting — ``"off"``, or ``"native:<tools sha12>:dry_run"`` with
    #: ``":repair_off"`` appended when the repair could not be armed. A line written before the field reads "off".
    tools: str = "off"
    #: T-0213 A2r-4: the settings revision the turn ran under (0: as launched). A line written before it reads 0.
    settings_rev: int = Field(default=0, ge=0)
    #: T-0213 A2r-4: sha256 of the turn's rendered system message text, first 12 hex ("" on a line written before)
    system_sha12: str = ""
    #: T-0213 A2r-4: the user's own instructions block was in the turn's system message
    instructions_edited: bool = False
    #: T-0227 A-9: the screens-per-turn cap the turn ran under; on the line only when not the default (3)
    max_frames_per_turn: int = DEFAULT_MAX_FRAMES_PER_TURN
    #: T-0227 A-9: the HUD capture interval in force at the turn (ms); on the line only when not the default (1000)
    frame_interval_ms: int = DEFAULT_FRAME_INTERVAL_MS

    @model_validator(mode="after")
    def _badge_agrees_with_the_flags(self) -> Regime:
        """A line whose badge disagrees with its flags is evidence it contradicts itself.

        ``assists`` has a plausible default ("pixels"), so a forgotten assignment would otherwise
        log a *false* regime instead of failing. Use ``with_assists``.
        """
        if self.assists != self.assist_flags.badge():
            raise ValueError(
                f"assists {self.assists!r} disagrees with assist_flags "
                f"{self.assist_flags.badge()!r}; set both via Regime.with_assists()"
            )
        return self

    @model_serializer(mode="wrap")
    def _settings_fields_only_when_changed(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """T-0213 R-8: a turn nobody changed the settings for writes the line BASE wrote, byte for byte.

        The three settings fields are dumped only when ``settings_rev > 0`` or ``instructions_edited``; the
        session's ``system_sha12`` is in the hello state instead.
        """
        data: dict[str, Any] = handler(self)
        if self.settings_rev == 0 and not self.instructions_edited:
            for key in SETTINGS_FIELDS:
                data.pop(key, None)
        for key, default in FRAME_FIELD_DEFAULTS.items():
            if getattr(self, key) == default:
                data.pop(key, None)
        return data

    def with_assists(self, flags: AssistFlags) -> Regime:
        """A copy carrying ``flags`` and their badge — the sanctioned way to set the pair.

        Both fields derive from one argument, so the log cannot drift from the ``Prompt`` the
        request was built from (criterion 8: pass ``prompt.assists`` here, never a second copy of
        the assist state, which may have moved on since the turn was frozen).
        """
        return self.model_copy(update={"assists": flags.badge(), "assist_flags": flags})


class TurnRecord(Contract):
    turn_id: str
    session_id: str
    #: span name → (start_ms, end_ms) on the session audio clock
    spans: dict[str, tuple[int, int]]
    events: tuple[tuple[int, str], ...] = ()
    regime: Regime
    #: T-0039 criterion 9: how the question arrived — spoken, typed into the HUD text box, or
    #: clicked on a rendered A2UI surface. A session mixes all three, so this is per *turn*; it
    #: is not a second copy of the regime's audio path, and a typed run's numbers are not
    #: comparable to an acoustic one's.
    input: Literal["voice", "text", "ui"] = "voice"
    outcome: str = Field(pattern="^(spoken|interrupted|error|vetoed|silent)$")
    canvases: int = Field(default=0, ge=0)
    thought_tokens: int = Field(default=0, ge=0)
    answer_tokens: int = Field(default=0, ge=0)
    thought_closed: bool | None = None
    #: T-0043 criterion 14: the absolute path of the session log this turn was recorded in, or
    #: ``None`` when logging was off for that turn. A top-level field, not an ``extras`` key: a
    #: bench reader must not have to guess where the log is. A line written before the field
    #: existed reads as "not logged", which is what it was.
    session_log: str | None = None
    #: T-0092b criterion 5: the turn's tool call, judged and executed (T-0092c writes it); ``None`` without one.
    tool: dict[str, Any] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    def unknown_spans(self) -> list[str]:
        return [s for s in self.spans if s not in SPAN_NAMES]

    def latency_ms(self, start: str, end: str) -> int | None:
        a, b = self.spans.get(start), self.spans.get(end)
        if a is None or b is None:
            return None
        return b[0] - a[1]

    def to_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), separators=(",", ":"))

    @classmethod
    def from_line(cls, line: str) -> TurnRecord:
        return cls.model_validate_json(line)
