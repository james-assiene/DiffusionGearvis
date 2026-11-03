"""HUD ↔ Conductor WebSocket protocol.

The vocabulary is the OpenAI Realtime API's (as used by Unmute, from which the Conductor's design
is derived (MIT), no code copied) plus ``astra.*`` extensions. Standard Realtime events pass through as ``RealtimeEvent``
(type + payload) because the Conductor only relays them; every ``astra.*`` message is a strict
model here and round-trips in ``tests/contract/test_protocol_codec.py``.

Adding an event: add the model, add it to ``AstraMessage`` and ``ASTRA_MESSAGE_TYPES``; the
round-trip test is parametrised over that tuple and fails if the codec cannot decode what it
encodes.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from local_astra.contracts.base import Contract
from local_astra.contracts.llm import FenceClose, Segment
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.turn import TurnState

PROTOCOL_VERSION = 1

#: T-0039 criterion 2: the cap on a typed question, in code points (a chosen bound, not a
#: measured one — docs/tasks/T-0039.md §5, ambiguity 1).
MAX_TEXT_CODE_POINTS = 2000


# --- client → server ---------------------------------------------------------------------------


class AstraFrame(Contract):
    """v1 perception source: the HUD page captures a tab/camera and posts JPEG at ≤ 1 fps."""

    type: Literal["astra.frame"] = "astra.frame"
    source: Source
    ts_ms: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    jpeg_b64: str = Field(min_length=1)
    phash: str | None = None
    title: str | None = None
    app: str | None = None


class AstraFrameText(Contract):
    """Assist payload — sent only when the corresponding checkbox is on."""

    type: Literal["astra.frame.text"] = "astra.frame.text"
    source: Source
    ts_ms: int = Field(ge=0)
    kind: Literal["page_text", "pdf_text", "cells", "at_spi"]
    text: str
    url: str | None = None


class AstraAssistToggle(Contract):
    type: Literal["astra.assist.toggle"] = "astra.assist.toggle"
    flags: AssistFlags


class AstraUiAction(Contract):
    """A2UI action from the rendered surface (button, choice, submit)."""

    type: Literal["astra.ui.action"] = "astra.ui.action"
    surface_id: str
    name: str
    context: dict[str, Any] = Field(default_factory=dict)
    data_model: dict[str, Any] | None = None


class AstraText(Contract):
    """A question typed into the HUD page (T-0039 criterion 1): text mode's only ingress.

    ``text`` is validated here rather than in the transport (criterion 2), so every client — the
    page, the fake, a later rung — obeys one rule: surrounding whitespace is stripped, what is
    left must be non-empty and at most ``MAX_TEXT_CODE_POINTS`` **code points**. A violation is a
    ``codec.decode`` → ``Malformed`` answered with ``astra.error(code="malformed")``; it is never
    truncated and never silently dropped. T-0027's 8 MiB byte limit is a different rule and is
    untouched — 2000 astral characters are 8000 bytes and are accepted.
    """

    type: Literal["astra.text"] = "astra.text"
    text: str

    @field_validator("text")
    @classmethod
    def _stripped_non_empty_and_capped(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text is empty after stripping surrounding whitespace")
        if len(v) > MAX_TEXT_CODE_POINTS:
            raise ValueError(f"text is {len(v)} code points, over the {MAX_TEXT_CODE_POINTS} cap")
        return v


class AstraLogToggle(Contract):
    """T-0043 criterion 16: "log this session", on or off — the HUD checkbox's whole egress.

    A new pair rather than a field on ``AstraAssistToggle``: the log flag is not an assist, so it
    stays out of ``AssistFlags``, out of the assist badge and out of a turn's ``Regime``, which
    says what the model could *see* and not whether we were debugging. The page carries no path:
    where a session writes is the CLI's decision (criterion 15), never the page's.
    """

    type: Literal["astra.log.toggle"] = "astra.log.toggle"
    on: bool


class AstraHello(Contract):
    type: Literal["astra.hello"] = "astra.hello"
    protocol_version: int = PROTOCOL_VERSION
    client: str = "hud"


class AstraSessionReset(Contract):
    """T-0162 A-1: the page's Restart. Ends the running turn and starts a new session: history,
    frames, tool calls and the turn journal are cleared and long-term memory is wiped. No fields."""

    type: Literal["astra.session.reset"] = "astra.session.reset"


# --- server → client ---------------------------------------------------------------------------


class AstraAssistState(Contract):
    """Echo of the effective assist flags plus the badge logged with every turn."""

    type: Literal["astra.assist.state"] = "astra.assist.state"
    flags: AssistFlags
    badge: str


class AstraTurnState(Contract):
    type: Literal["astra.turn.state"] = "astra.turn.state"
    state: TurnState
    turn_id: str
    #: for the thinking meter: ms since llm.start, None outside THINKING
    thinking_ms: int | None = None
    #: T-0037: this message is a copy of one this session already published, replayed to a
    #: (re)connecting page at its ``astra.hello`` so a reload does not land on an empty pane.
    #: A message published live always carries ``False``.
    replay: bool = False


class AstraUiSurface(Contract):
    """Validated A2UI messages for one surface (docs/12 §1.1).

    ``messages`` are A2UI v0.9.1 server→client JSON objects that passed fixer → schema →
    topology. ``canvas_idx`` lets the HUD render progressively; ``final`` marks the last update
    for this turn. ``repaired`` counts json_repair edits so the HUD can show a small badge.
    """

    type: Literal["astra.ui.surface"] = "astra.ui.surface"
    surface_id: str
    canvas_idx: int = Field(ge=0)
    final: bool = False
    messages: tuple[dict[str, Any], ...]
    repaired: int = Field(default=0, ge=0)
    #: T-0037: this message is a copy of one this session already published, replayed to a
    #: (re)connecting page at its ``astra.hello`` so a reload does not land on an empty pane.
    #: A message published live always carries ``False``.
    replay: bool = False


class AstraUiFence(Contract):
    """How this turn's ``<a2ui>`` fence ended, and how much of the reply was not spoken.

    A **type** and not a field on ``astra.ui.surface`` (T-0054 D1): the page's decoder is strict
    both ways, so an unmirrored *field* turns every surface message into ``kind: "malformed"`` and
    blanks the shown pane, while an unmirrored *type* decodes as ``kind: "unknown"`` and is
    ignored and counted. ``close`` is the ``FenceClose`` of the **last** ``ui`` commit of the turn
    and ``None`` means *no fence was ever opened* — never "it ended cleanly"; ``canvases`` is how
    many distinct canvases produced one; ``continued`` says that last one carried a fence opened
    in an earlier canvas; ``speech_refused`` is how many sentences the voice guard refused as
    interface data (T-0052 criterion 8), which rides here rather than in a second type because the
    human's question is one question — what happened to the interface this turn? (D4).

    It carries **no ``replay`` field**: ``TurnJournal`` does not journal it, so a page that
    reloads mid-turn never receives it, and a replay flag would promise a catch-up that does not
    exist (T-0054 "What stays invisible" item 1).
    """

    type: Literal["astra.ui.fence"] = "astra.ui.fence"
    turn_id: str
    canvases: int = Field(ge=0)
    close: FenceClose | None = None
    continued: bool = False
    speech_refused: int = Field(default=0, ge=0)


class AstraCanvasStep(Contract):
    """One denoising step from the plugin tap (docs/12 §2), ≤ 30 Hz per request (T-0011).

    ``canvas`` is the post-renoise canvas (still-random positions included: the flicker),
    ``argmax`` the argmax canvas, ``stable`` a hex bitmask (one bit per position, MSB first,
    zero-padded to whole bytes) of positions equal across the whole accepted-canvas history.
    One wire shape per phase (R10): a ``denoise`` frame carries all three rows and no
    ``committed_tokens``, with ``0 <= step <= max_steps`` (R15: the call that converges at the cap
    leaves ``step == max_steps``); a ``commit`` frame carries the non-empty
    ``committed_tokens`` and none of the rows (the HUD draws the closed canvas from the ids), and
    its ``step`` is the step the canvas had reached before the commit reset it, so
    ``0 <= step <= max_steps`` (R14; step 0 since T-0069 ruling O6: a CD-ended slot scheduled again
    commits its EOS canvas before any denoise step).

    ``accepted`` and ``entropy`` are optional and absent on every plugin frame: v0.29.0 exposes
    neither (both are locals of the compiled step). Never synthesised: a fallback animation is
    sent as ``astra.canvas.synthetic`` instead, so the HUD can label it.
    """

    type: Literal["astra.canvas.step"] = "astra.canvas.step"
    request_id: str
    canvas_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    canvas_length: int = Field(ge=1)
    canvas: tuple[int, ...] | None = None
    argmax: tuple[int, ...] | None = None
    stable: str | None = Field(default=None, pattern="^[0-9a-f]*$")
    phase: Literal["denoise", "commit"] = "denoise"
    committed_tokens: tuple[int, ...] = ()
    accepted: str | None = Field(default=None, pattern="^[0-9a-f]*$")
    entropy: tuple[float, ...] | None = None
    #: which channel this canvas is denoising, when the tap can tell (thought canvases are dimmed)
    segment: Segment | None = None
    text_preview: str | None = None
    #: T-0159: one text piece per position of ``argmax`` (denoise frames only), added by the
    #: conductor's relay from the frame's own ids (``HostDetokenizer.pieces``); never sent by the
    #: plugin. Byte-fallback ids stay their own ``<0xNN>`` literal, one position per piece.
    #: ``StrictStr``: an item that is not a string (bytes included) is refused, never coerced,
    #: as the HUD mirror refuses it.
    argmax_pieces: tuple[StrictStr, ...] | None = None
    #: T-0159: one text piece per id of ``committed_tokens`` (commit frames only), as above.
    committed_pieces: tuple[StrictStr, ...] | None = None

    @model_validator(mode="after")
    def _lengths_agree(self) -> AstraCanvasStep:
        n = self.canvas_length
        for name, ids in (
            ("canvas", self.canvas),
            ("argmax", self.argmax),
            ("entropy", self.entropy),
        ):
            if ids is not None and len(ids) != n:
                raise ValueError(f"{name} must have canvas_length entries")
        for name, mask in (("stable", self.stable), ("accepted", self.accepted)):
            if mask is not None and (len(mask) * 4 < n or len(mask) % 2):
                raise ValueError(f"{name} mask must cover canvas_length bits in whole bytes")
        if len(self.committed_tokens) > n:
            raise ValueError("committed_tokens cannot exceed canvas_length")
        rows = [r is not None for r in (self.canvas, self.argmax, self.stable)]
        if self.phase == "denoise":
            # R10: a denoise frame carries all three rows and no committed ids
            if not all(rows):
                raise ValueError("a denoise frame carries canvas, argmax and stable")
            if self.committed_tokens:
                raise ValueError("committed_tokens belong to a commit frame")
            # R15: the post-call step of the call that converges at the cap is max_steps
            if self.step > self.max_steps:
                raise ValueError("a denoise frame's step must be in 0..max_steps")
        else:
            # R10/R14: a commit frame carries the committed ids and none of the rows; its step is
            # the pre-reset step the canvas reached, 0 <= step <= max_steps (0: ruling O6, T-0069)
            if any(rows):
                raise ValueError("a commit frame carries none of canvas, argmax and stable")
            if not self.committed_tokens:
                raise ValueError("a commit frame carries non-empty committed_tokens")
            if not 0 <= self.step <= self.max_steps:
                raise ValueError("a commit frame's step must be in 0..max_steps")
        # T-0159 A1: pieces belong to their phase's ids and have one entry per id
        if self.phase == "denoise":
            if self.committed_pieces is not None:
                raise ValueError("committed_pieces belong to a commit frame")
            if self.argmax_pieces is not None and len(self.argmax_pieces) != n:
                raise ValueError("argmax_pieces must have canvas_length entries")
        else:
            if self.argmax_pieces is not None:
                raise ValueError("argmax_pieces belong to a denoise frame")
            if self.committed_pieces is not None and len(self.committed_pieces) != len(
                self.committed_tokens
            ):
                raise ValueError("committed_pieces must have one entry per committed token")
        return self


class AstraCanvasCommit(Contract):
    type: Literal["astra.canvas.commit"] = "astra.canvas.commit"
    request_id: str
    canvas_idx: int = Field(ge=0)
    segment: Segment
    text: str
    steps_used: int = Field(ge=0)


class AstraCanvasSynthetic(Contract):
    """Labelled fallback when the tap is unavailable (plugin not loaded)."""

    type: Literal["astra.canvas.synthetic"] = "astra.canvas.synthetic"
    request_id: str
    reason: str


class AstraTranscript(Contract):
    type: Literal["astra.transcript"] = "astra.transcript"
    turn_id: str
    text: str
    final: bool = False
    #: T-0037: this message is a copy of one this session already published, replayed to a
    #: (re)connecting page at its ``astra.hello`` so a reload does not land on an empty pane.
    #: A message published live always carries ``False``.
    replay: bool = False


class AstraSpoken(Contract):
    """Assistant text as it is handed to TTS (for the HUD transcript; struck through if unplayed)."""

    type: Literal["astra.spoken"] = "astra.spoken"
    turn_id: str
    utterance_id: str
    text: str
    #: T-0037: this message is a copy of one this session already published, replayed to a
    #: (re)connecting page at its ``astra.hello`` so a reload does not land on an empty pane.
    #: A message published live always carries ``False``.
    replay: bool = False


class PlayedUtterance(Contract):
    """How much of one published utterance actually reached the device (T-0030 criterion 1).

    ``played_text`` is a *word prefix* of that utterance's committed source (``TtsSpeak.source``,
    not the TTS-normalised ``AstraSpoken.text`` — they are different strings), or ``""`` when
    nothing of it was heard. ``played_words`` is the cut index into the word sequence, valid in
    both spellings because ``audio_io._partial_text`` refuses a partial unless the two word counts
    agree. ``complete`` means the entry covers the whole source.
    """

    utterance_id: str
    played_text: str
    played_words: int = Field(ge=0)
    complete: bool


class AstraSpokenPlayed(Contract):
    """What was heard of one playback run, published once when that run ends (docs/02 §4).

    One entry per utterance the session published as ``astra.spoken`` in the run, in publication
    order; the entries' ``played_text`` rejoin — with no separator — to the run's ``played_text``.
    The page dims and strikes the tail from ``played_words`` on; it never derives that cut from
    timing.
    """

    type: Literal["astra.spoken.played"] = "astra.spoken.played"
    turn_id: str
    played_ms: int = Field(ge=0)
    utterances: list[PlayedUtterance]
    #: T-0037: this message is a copy of one this session already published, replayed to a
    #: (re)connecting page at its ``astra.hello`` so a reload does not land on an empty pane.
    #: A message published live always carries ``False``.
    replay: bool = False


class AstraLogState(Contract):
    """T-0043 criteria 16-19: what the session log is doing, as the checkbox renders it.

    The page never infers this: its checkbox is ticked by this message and by nothing else, so
    ``on`` is required and not nullable ("no state has arrived yet" is the absence of this
    message, not a null inside one). ``path`` is the file being written, or ``None`` when there
    is none; ``error`` is the reason logging is off when it disabled itself (criterion 4). Both
    default to ``None`` because ``codec.encode`` is ``exclude_none``: a nullable field with no
    default would not survive its own round trip when it is absent. The three counters are always
    knowable when a state is published — zero is a fact, not an absence — so they are required.
    """

    type: Literal["astra.log.state"] = "astra.log.state"
    on: bool
    path: str | None = None
    records: int = Field(ge=0)
    dropped: int = Field(ge=0)
    bytes: int = Field(ge=0)
    error: str | None = None


#: T-0090 criterion 1: a proposed action's lifecycle. ``proposed → repairing → dry_run | refused |
#: cancelled`` in v1 (no execution yet, T-0080d H1); ``executed`` is reserved for the later task.
ToolCallStatus = Literal["proposed", "repairing", "dry_run", "executed", "refused", "cancelled"]
#: Where the arguments came from: the model's own draft, or the repair pass over it.
ToolCallSource = Literal["draft", "repair"]
#: T-0162 C-1: how a call ended, from the session's own data; wider than the card's ``ToolCallStatus``
#: (a live failure is a ``refused`` card with a ``failed`` outcome, a deadline a ``cancelled`` card with ``timeout``).
ToolCallOutcome = Literal["executed", "dry_run", "refused", "failed", "cancelled", "timeout"]
#: T-0162 C-1: the hosts a final message names, at most this many; the rest are counted in ``hosts_more``.
TOOL_RESULT_HOSTS = 3
#: T-0162 C-1 "host only": dot-separated labels of letters, digits, ``-`` and ``_``; no scheme, port, user, path,
#: query, fragment, space or markup. Mirrored in ``hud/src/protocol.js``.
TOOL_RESULT_HOST_PATTERN = r"^[A-Za-z0-9_-]{1,63}(\.[A-Za-z0-9_-]{1,63})*$"


class ToolCallResult(Contract):
    """T-0162 C-1: the short result on a call's final ``astra.tool.call``, filled from data, never from model
    prose (C-2). ``hosts`` are host names only (the browser's reached hosts, or a search's result hosts),
    ``hosts_more`` counts those beyond them, ``results`` a search's result count (``None``: not a search, or no
    result came back)."""

    outcome: ToolCallOutcome
    hosts: list[Annotated[StrictStr, Field(max_length=253, pattern=TOOL_RESULT_HOST_PATTERN)]] = (
        Field(default_factory=list, max_length=TOOL_RESULT_HOSTS)
    )
    hosts_more: int = Field(default=0, ge=0)
    results: int | None = Field(default=None, ge=0)


class AstraToolCall(Contract):
    """One state of one proposed action, for the HUD's action card (T-0090 criterion 1).

    A later message for the same ``call_id`` supersedes the earlier one (the page updates the
    card in place, the journal keeps the latest per ``call_id`` of the open turn). ``arguments``
    are the call's own arguments — screen coordinates on the 0–999 grid, never pixels.
    ``action`` is the server's description of what would be done, ``reason`` why a call was
    refused or cancelled, ``query_problem`` a warning about a search query (T-0080d H7) that never
    blocks the card. The three default to ``None`` because ``codec.encode`` is ``exclude_none``.
    Display-only: the page sends nothing back.
    """

    type: Literal["astra.tool.call"] = "astra.tool.call"
    turn_id: str
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    source: ToolCallSource
    action: str | None = None
    reason: str | None = None
    query_problem: str | None = None
    #: T-0162 C-1: the tool-loop step the call belongs to (1 with the loop off); ``None`` when not known
    step: int | None = Field(default=None, ge=1)
    #: T-0162 C-1: set only on the call's final message (a final ``status``), never on an earlier one
    result: ToolCallResult | None = None
    #: T-0037/T-0090 criterion 3: a copy replayed to a (re)connecting page at its
    #: ``astra.hello``. A message published live always carries ``False``.
    replay: bool = False

    @model_validator(mode="after")
    def _result_only_when_final(self) -> AstraToolCall:
        if self.result is not None and self.status in ("proposed", "repairing"):
            raise ValueError("result is set only on a final message")
        return self


class AstraSessionCleared(Contract):
    """T-0162 A-1: the server's echo of a completed reset; the page clears every panel once per
    new ``resets`` value. ``resets`` counts the resets of this session, ``ended_turn_id`` is the
    turn the reset cut (``None`` when none was open), ``memory_forgotten`` the episodes deleted."""

    type: Literal["astra.session.cleared"] = "astra.session.cleared"
    resets: int = Field(ge=1)
    ended_turn_id: str | None = None
    memory_forgotten: int = Field(ge=0)
    #: a copy at the head of a (re)connecting page's hello reply; ``False`` when published live
    replay: bool = False


#: T-0177 A1: what the model looks at in a live session, the session's choice (the "Looking at" combo)
VisionSource = Literal["browser", "shared"]
#: T-0177 A1: ``prompt_builder.Budget().max_images`` (the engine's per-request image ceiling), stated here because the
#: contracts import nothing from the conductor; ``tests/contract/test_t0177_vision_wire.py`` reads the oracle
VISION_MAX_IMAGES = 4


class VisionImage(Contract):
    """T-0177 A1: one image a request actually sent, as the panel shows it. ``identity`` is the image part's
    ``phash`` (a capture identity in live), ``kind`` what it shows. ``jpeg_b64`` is the exact bytes sent, base64; it
    is ``None`` exactly when the message would exceed ``codec.MAX_MESSAGE_BYTES``, and then ``omitted`` is
    ``"too_large"`` (both default to ``None`` because ``codec.encode`` is ``exclude_none``)."""

    identity: str
    kind: Literal["window", "screen", "camera", "overview"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    jpeg_b64: str | None = None
    omitted: Literal["too_large"] | None = None

    @model_validator(mode="after")
    def _omitted_iff_no_bytes(self) -> VisionImage:
        if (self.jpeg_b64 is None) != (self.omitted == "too_large"):
            raise ValueError('jpeg_b64 is null exactly when omitted is "too_large"')
        return self


class AstraVisionSelect(Contract):
    """T-0177 A1 (client to server): the page asks the session to look at ``source``. The session decides and
    echoes ``astra.vision.state``; the page's combo follows the echo, never its own click."""

    type: Literal["astra.vision.select"] = "astra.vision.select"
    source: VisionSource


class AstraVisionState(Contract):
    """T-0177 A1 (server to client): the session's current vision choice and the choices it offers. Published
    in live only: at every routed select (redundant ones included) and last in every hello reply."""

    type: Literal["astra.vision.state"] = "astra.vision.state"
    source: VisionSource
    choices: list[VisionSource]
    replay: bool = False


class AstraVisionFrame(Contract):
    """T-0177 A1 (server to client): the images one request actually sent (``source``: where they came from),
    the "What I see" panel's only source. ``blind`` means no image was sent, so ``images`` is then empty.
    ``history_images`` counts the earlier images the request also carried."""

    type: Literal["astra.vision.frame"] = "astra.vision.frame"
    turn_id: str
    step: int = Field(ge=1)
    source: VisionSource
    blind: bool
    images: list[VisionImage] = Field(default_factory=list, max_length=VISION_MAX_IMAGES)
    history_images: int = Field(ge=0)
    replay: bool = False

    @model_validator(mode="after")
    def _blind_has_no_images(self) -> AstraVisionFrame:
        if self.blind and self.images:
            raise ValueError("a blind frame carries no images")
        return self


#: T-0213 A2r-1: one setting's value on the wire: a bool, an int, a text, a pair of ints (the viewport) or ``None``
#: (unset: a live-only setting outside live). Strict, so ``True`` is never read as ``1`` nor ``"1"`` as a number.
SettingValue: TypeAlias = StrictBool | StrictInt | StrictStr | list[StrictInt] | None
#: T-0213 A2r-1: where a setting's current value came from (the resolvers' ``unset`` reads ``default``)
SettingSource = Literal["default", "profile", "flag", "ui"]
#: T-0213 A1-1: the four change kinds, stated here because the contracts import nothing from the catalog, and
#: R-10's ``not_here`` (an item only: not changeable in this mode, and no relaunch command changes that)
SettingKindName = Literal["next_turn", "next_turn_cold", "relaunch", "fixed", "not_here"]


class SettingItem(Contract):
    """T-0213 A2r-1: one setting as the session has it now. ``changeable`` says whether the panel may change it in
    this session; ``pending`` is a validated change waiting for the next turn boundary (always ``None`` until
    A2-apply); ``relaunch`` is the exact command that changes a setting the panel cannot (``None`` otherwise). The
    nullable fields default to ``None`` because ``codec.encode`` is ``exclude_none``."""

    key: StrictStr = Field(min_length=1)
    value: SettingValue = None
    source: SettingSource
    kind: SettingKindName
    changeable: StrictBool
    pending: SettingValue = None
    relaunch: str | None = None


class CatalogEntry(Contract):
    """T-0213 A2r-1: one ``settings_catalog.SettingEntry`` on the wire, field for field (the panel's text). Sent on
    the hello reply's ``astra.settings.state`` only."""

    key: StrictStr = Field(min_length=1)
    label: str
    group: str
    kind: SettingKindName
    v1_changeable: StrictBool
    values: list[SettingValue] = Field(default_factory=list)
    default: SettingValue = None
    default_note: str
    flag: str
    flags: list[str] = Field(default_factory=list)
    profile_key: str | None = None
    what: str
    useful_when: str
    not_useful_when: str
    evidence: str
    cost: str
    relaunch: str | None = None


class SettingRefusal(Contract):
    """T-0213 A2r-1: one refused change: the key (``"*"`` when none of the changed keys is named) and why."""

    key: StrictStr = Field(min_length=1)
    reason: str


class AstraSettingsState(Contract):
    """T-0213 A2r-1 (server to client): every setting with its value and source, the session's settings revision,
    the sha of the system message the next turn renders, and the user's instructions. ``catalog`` is the panel's
    text, on the hello reply only (``None`` on every other publish). Read-only in A2-read: no client message
    changes anything yet."""

    type: Literal["astra.settings.state"] = "astra.settings.state"
    rev: StrictInt = Field(ge=0)
    items: list[SettingItem]
    catalog: list[CatalogEntry] | None = None
    #: sha256 of the rendered system message text, first 12 hex (the session computes it; not re-checked here)
    system_sha12: str
    instructions: str
    instructions_max: int = Field(ge=0)
    refused: list[SettingRefusal] = Field(default_factory=list)
    replay: bool = False
    #: T-0213 C-2 (added in A2-apply): the assembled system message the next turn renders, as text, for the panel's
    #: read-only view. ``None`` (left off the wire, ``codec.encode`` is ``exclude_none``) on a state that has none.
    system_preview: str | None = None

    @model_validator(mode="after")
    def _instructions_within_the_cap(self) -> AstraSettingsState:
        if len(self.instructions) > self.instructions_max:
            raise ValueError(
                f"instructions are {len(self.instructions)} code points, over the {self.instructions_max} cap"
            )
        return self


class AstraSettingsSet(Contract):
    """T-0213 A2a-1 (client to server): the panel asks the session to change settings. ``rev`` is the revision the
    page last saw (any other, a negative one included, is refused as stale by the session: ``key="*"``,
    ``reason="stale"``); ``changes`` maps catalog keys to new values and is applied all or nothing. The values are
    any JSON value here on purpose: the session's ``live_settings.validate`` judges them (unknown keys, the
    ``type(v) is int`` strictness, the ``Profile`` cross-rules) and answers with its own reasons, which a decode
    error could not carry. A validated set waits for the next turn boundary; ``stop_turn`` first stops a turn in
    flight (T-0193's stop). The echo is ``astra.settings.state``."""

    type: Literal["astra.settings.set"] = "astra.settings.set"
    rev: StrictInt
    changes: dict[str, Any]
    stop_turn: StrictBool = False


#: T-0180 A6 (H-T180-5): the most thought text one ``astra.thought`` carries, in code points. The tail past it is
#: not carried (no chunking); its length goes in ``omitted_chars``. Mirrored in ``hud/src/protocol.js``.
THOUGHT_MAX_CHARS = 16000
#: T-0180 A1: why the session made the attempt (the page labels the block from it, never infers it)
ThoughtKind = Literal["step", "reask", "retry", "repair", "reground"]
#: T-0180 A1: ``open`` while the attempt runs, then exactly one final value
ThoughtEnd = Literal["open", "done", "stopped_repeating", "interrupted", "error"]


class AstraThought(Contract):
    """T-0180 A1 (server to client): the committed thought text of one attempt of a turn, for that turn's
    collapsed "Model thoughts" disclosure. ``text`` is cumulative (the in-order concatenation of the attempt's
    ``segment == "thought"`` commits, never a draft), at most ``THOUGHT_MAX_CHARS`` code points; ``omitted_chars``
    counts the tail past the cap and is non-zero only when ``text`` is at the cap. A later message for the same
    ``attempt_id`` supersedes the earlier one. The only message type that carries thought text."""

    type: Literal["astra.thought"] = "astra.thought"
    turn_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    step: int = Field(ge=1)
    kind: ThoughtKind
    text: str
    omitted_chars: int = Field(ge=0)
    end: ThoughtEnd
    #: T-0037: a copy replayed to a (re)connecting page at its ``astra.hello``; ``False`` when published live
    replay: bool = False

    @model_validator(mode="after")
    def _capped_and_omitted_only_at_the_cap(self) -> AstraThought:
        if len(self.text) > THOUGHT_MAX_CHARS:
            raise ValueError(
                f"text is {len(self.text)} code points, over the {THOUGHT_MAX_CHARS} cap"
            )
        if self.omitted_chars > 0 and len(self.text) < THOUGHT_MAX_CHARS:
            raise ValueError("omitted_chars > 0 only when text is at the cap")
        return self


class AstraMicMute(Contract):
    """T-0181 A1 (client to server): the page asks the session to mute (``True``) or unmute the microphone. The
    session decides and echoes ``astra.mic.state``; the page's control follows the echo, never its own click."""

    type: Literal["astra.mic.mute"] = "astra.mic.mute"
    muted: StrictBool


class AstraMicState(Contract):
    """T-0181 A1 (server to client): the session's microphone switch. Published at every routed
    ``astra.mic.mute`` (redundant ones included) and in every hello reply, right after ``astra.log.state``.
    ``available`` is decided once, at construction; a session with no microphone is never muted and says why in
    ``reason``, which is ``None`` exactly when the mic is available."""

    type: Literal["astra.mic.state"] = "astra.mic.state"
    muted: StrictBool
    available: StrictBool
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def _unavailable_is_unmuted_with_a_reason(self) -> AstraMicState:
        if not self.available and self.muted:
            raise ValueError("an unavailable microphone is never muted")
        if (self.reason is None) != self.available:
            raise ValueError("reason is set exactly when the microphone is unavailable")
        return self


class AstraTurnStop(Contract):
    """T-0193 A-1 (client to server): the page asks the session to stop the turn ``turn_id`` (the HUD's "Interrupt
    Gearvis" button). The session submits one ``UserStop`` and the machine decides whether it applies; the echo is
    the existing ``astra.turn.state``."""

    type: Literal["astra.turn.stop"] = "astra.turn.stop"
    turn_id: StrictStr = Field(min_length=1)


class AstraError(Contract):
    type: Literal["astra.error"] = "astra.error"
    code: str
    message: str


class RealtimeEvent(Contract):
    """Any OpenAI-Realtime-vocabulary event relayed unchanged (``type`` not starting with astra.)."""

    type: str = Field(pattern=r"^[a-z_.]+$")
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _not_astra(cls, v: str) -> str:
        if v.startswith("astra."):
            raise ValueError("astra.* messages have their own models")
        return v


AstraMessage = Annotated[
    AstraHello
    | AstraFrame
    | AstraFrameText
    | AstraAssistToggle
    | AstraUiAction
    | AstraText
    | AstraLogToggle
    | AstraSessionReset
    | AstraAssistState
    | AstraTurnState
    | AstraUiSurface
    | AstraUiFence
    | AstraCanvasStep
    | AstraCanvasCommit
    | AstraCanvasSynthetic
    | AstraTranscript
    | AstraSpoken
    | AstraSpokenPlayed
    | AstraLogState
    | AstraToolCall
    | AstraSessionCleared
    | AstraError
    | AstraVisionSelect
    | AstraVisionState
    | AstraVisionFrame
    | AstraThought
    | AstraMicMute
    | AstraMicState
    | AstraTurnStop
    | AstraSettingsState
    | AstraSettingsSet,
    Field(discriminator="type"),
]

AstraMessageAdapter: TypeAdapter[AstraMessage] = TypeAdapter(AstraMessage)

ASTRA_MESSAGE_TYPES: tuple[type[Contract], ...] = (
    AstraHello,
    AstraFrame,
    AstraFrameText,
    AstraAssistToggle,
    AstraUiAction,
    AstraText,
    AstraLogToggle,
    AstraSessionReset,
    AstraAssistState,
    AstraTurnState,
    AstraUiSurface,
    AstraUiFence,
    AstraCanvasStep,
    AstraCanvasCommit,
    AstraCanvasSynthetic,
    AstraTranscript,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraLogState,
    AstraToolCall,
    AstraSessionCleared,
    AstraError,
    AstraVisionSelect,
    AstraVisionState,
    AstraVisionFrame,
    AstraThought,
    AstraMicMute,
    AstraMicState,
    AstraTurnStop,
    AstraSettingsState,
    AstraSettingsSet,
)

CLIENT_TO_SERVER: frozenset[str] = frozenset(
    {
        "astra.hello",
        "astra.frame",
        "astra.frame.text",
        "astra.assist.toggle",
        "astra.ui.action",
        "astra.text",
        "astra.log.toggle",
        "astra.session.reset",
        "astra.vision.select",
        "astra.mic.mute",
        "astra.turn.stop",
        "astra.settings.set",
    }
)
SERVER_TO_CLIENT: frozenset[str] = frozenset(
    {
        "astra.assist.state",
        "astra.turn.state",
        "astra.ui.surface",
        "astra.ui.fence",
        "astra.canvas.step",
        "astra.canvas.commit",
        "astra.canvas.synthetic",
        "astra.transcript",
        "astra.spoken",
        "astra.spoken.played",
        "astra.log.state",
        "astra.tool.call",
        "astra.session.cleared",
        "astra.error",
        "astra.vision.state",
        "astra.vision.frame",
        "astra.thought",
        "astra.mic.state",
        "astra.settings.state",
    }
)
