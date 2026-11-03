"""``astra.toml`` profiles (docs/02 §7). Loopback-only endpoints are validated at load time."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from local_astra.contracts.base import Contract

#: T-0092b criterion 1: the session's tool setting. ``off`` is the default (H3). T-0111g: ``live`` runs the
#: browser executor on Gearvis's loopback start pages, and needs the tool loop.
ToolsMode = Literal["off", "dry_run", "live"]
TOOLS_MODES: tuple[str, ...] = ("off", "dry_run", "live")


def tools_refusal(value: object) -> str:
    """Why a ``tools`` value is refused: only ``off``, ``dry_run`` and ``live`` exist (docs/01 §3, T-0080d,
    T-0111g)."""
    return (
        f"tools = {value!r} is refused: only 'off', 'dry_run' and 'live' exist; 'live' acts in Gearvis's own "
        "sandboxed browser on its loopback pages only (docs/01 §3, T-0080d, T-0111g)"
    )


def live_without_loop_refusal() -> str:
    """Why ``tools = "live"`` with the loop off is refused (T-0111g B1): the post-action frame exists only in the
    loop."""
    return (
        "tools = 'live' with tool_loop = false is refused: live needs the tool loop, since the screen after an "
        "action reaches the model only through the loop (T-0111g)"
    )


#: T-0131 A1: the per-request visual-token budgets a profile may name (``perception.policy.SUPPORTED_SOFT_TOKENS``).
VisualTokens = Literal[280, 560, 1120]
VISUAL_TOKENS: tuple[int, ...] = (280, 560, 1120)

#: T-0132 A1 (human ruling 2026-10-22): with ``tools = live`` an unset budget is 1120. T-0133 A1 (human ruling
#: 2026-10-22): an unset ReGround is off, in live too; ``--reground`` or ``reground = true`` turns it on.
#: Resolved in ``cli`` only (planner R-f), so a ``Session`` built without them is BASE's in every mode.
LIVE_DEFAULT_VISUAL_TOKENS: int = 1120
LIVE_DEFAULT_REGROUND: bool = False
#: T-0133 A5 (H-T133-1 approved): with ReGround on in live at a resolved main budget of 1120 and no pass-2 budget
#: set, pass 2's (the crop's) budget defaults to 280. Resolved in ``cli.grounding_of`` only.
LIVE_DEFAULT_REGROUND_VISUAL_TOKENS: int = 280

#: T-0227 A-1: the images one spoken turn may show the model (the onset screen plus screens that changed while the
#: user spoke, T-0059). The maximum is the engine's image limit (``--limit-mm-per-prompt '{"image": 4}'`` in
#: ``services/vllm.sh``, = ``prompt_builder.Budget.max_images``); going past it is its own task with a bench. The
#: default equals ``Budget().max_frames_per_turn``. Valid in every tools mode.
FRAMES_PER_TURN_CHOICES: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_MAX_FRAMES_PER_TURN: int = 3
#: T-0227 A-1: the HUD's capture interval per source (shared screen and camera), in milliseconds; the default is
#: ``1000 / perception.policy.MAX_FPS``. The fastest keeps the 60 s ring inside ``MAX_RING_FRAMES``; the slowest
#: stays under the ring. The agent browser's screenshots (``LiveBrowser.frame()``) are not affected.
FRAME_INTERVALS_MS: tuple[int, ...] = (250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000)
DEFAULT_FRAME_INTERVAL_MS: int = 1000

#: T-0134c A1: the live feedback a profile may name: ``off`` (no label), ``label`` (the T-0134 "screen now" label
#: after the current frame) or ``label+extent`` (the label plus the page-extent clause).
LiveFeedbackChoice = Literal["off", "label", "label+extent"]
LIVE_FEEDBACK_CHOICES: tuple[str, ...] = ("off", "label", "label+extent")
#: T-0134c A1: the units an executed ``click_at``/``type_at`` reports its point in: the page's ``pixels`` or the
#: call's own 0-999 ``model`` units (T-0134 A2).
ActionEcho = Literal["pixels", "model"]
ACTION_ECHOES: tuple[str, ...] = ("pixels", "model")
#: T-0134c (human ruling 2026-10-23): the live default is T-0134 arm B, the label without the page extent and the
#: action echo in model units. Resolved in ``cli.feedback_of`` only, so ``Session`` and ``BrowserToolExecutor``
#: keep their own defaults.
LIVE_DEFAULT_FEEDBACK: LiveFeedbackChoice = "label"
LIVE_DEFAULT_ACTION_ECHO: ActionEcho = "model"
#: T-0135b A1 (human ruling 2026-10-23, after the T-0135 A/B at e1ae76e): live recovers a tool call left at the tail
#: of an open thought (T-0135 arm B). Resolved in ``cli.recovery_of`` only, so ``Session`` keeps its own default.
LIVE_DEFAULT_THOUGHT_CALL_RECOVERY: bool = True
#: T-0140 A5: live does not offer ``page_overview`` unless opted in (``--page-overview`` or ``page_overview = true``);
#: adopting it as a default waits on the T-0140 A/B (T-0140b). Resolved in ``cli.page_overview_of`` only.
LIVE_DEFAULT_PAGE_OVERVIEW: bool = False
#: T-0150 A6: live does not show the whole page automatically at step 1 unless opted in (``--auto-overview`` or
#: ``auto_overview = true``); adopting it as a default waits on the T-0150 A/B (T-0150b). Resolved in
#: ``cli.auto_overview_of`` only.
LIVE_DEFAULT_AUTO_OVERVIEW: bool = False
#: T-0152 A1 (human ruling 2026-10-24, after the T-0151 A/B at 4471255: S4 0->20/20, S5 15->19/20, success
#: 72->94/100, over-clicks 1->4/60 all on S1/S2, which never get the frame; the over-click guard overridden): live
#: replaces step 1's frame with the whole-page picture on pages of 1.05-2.0 screens. ``--no-overview-frame`` or
#: ``overview_frame = false`` turns it off; beside the auto overview, the page overview or ReGround the default gives
#: way (H-T152-1). Resolved in ``cli.overview_frame_of`` only, so ``Session`` keeps its own default (off).
LIVE_DEFAULT_OVERVIEW_FRAME: bool = True
#: T-0136 C-1: live does not append the measured outcome clause (what visibly changed after an executed
#: ``click_at``/``type_at``) to the "screen now" label unless opted in (``--action-outcome`` or ``action_outcome =
#: true``); adopting it as a default waits on the T-0136 group D A/B. Resolved in ``cli.action_outcome_of`` only, so
#: ``Session`` keeps its own default (off).
LIVE_DEFAULT_ACTION_OUTCOME: bool = False
#: T-0137 B-9 / T-0179 A-1: live states repeats and refuses no-progress actions (the repeat guard) by default
#: (James's ruling 2026-10-27; the T-0137 group C A/B at 38423a5 measured no change). ``--no-repeat-guard`` or
#: ``repeat_guard = false`` turns it off. Resolved in ``cli.repeat_guard_of`` only, so ``Session`` keeps its own
#: default (off).
LIVE_DEFAULT_REPEAT_GUARD: bool = True
#: T-0172 B9 (H-T172-2): a tools-on step whose answer only announces an action and makes no call is re-asked once
#: (``announced_call.announces``, ``reask_prompt``), in ``dry_run`` and ``live``. ``--announced-call-reask off`` or
#: ``announced_call_reask = false`` turns it off. Resolved in ``cli.reask_of`` only, so ``Session`` keeps its own
#: default (``None``, BASE).
DEFAULT_ANNOUNCED_CALL_REASK: bool = True
#: T-0174 A3: the live tool guidance versions (``prompt_builder.LIVE_GUIDANCE``'s keys). v1 is T-0131's text; v2
#: replaces its first sentence with one that points backwards (the default from T-0174 group D, James's ruling
#: 2026-10-27, until T-0191 C). v3 is v2 plus one sentence on going to a different website, carried in web mode only
#: (T-0191; under ``--no-web`` a v3 session carries v2's text). The default is v3 (T-0191 group C, James's ruling
#: 2026-10-28 after the navigation smoke PASS). v4 is v3 plus a shared-screen sentence and a cite-the-source sentence,
#: web mode only (T-0211; under ``--no-web`` a v4 session carries v2's text); opt-in until T-0211 C.
#: ``--live-guidance {v1,v2,v3,v4}`` or ``live_guidance = "v1"|"v2"|"v3"|"v4"`` select one; resolved in
#: ``cli.live_guidance_of`` only. Live only.
LIVE_GUIDANCE_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3", "v4")
DEFAULT_LIVE_GUIDANCE: str = "v3"

#: T-0125 criterion 2 (H-T125-5, ruled 2026-10-27): the web search backends a profile may name. ``off`` turns it
#: off; ``ddgs`` is the keyless backend (``local_astra/search/``). ``None`` (the key absent) is the
#: mode's default: on in live, off in dry-run (H-T125-8), never in off. Resolved in ``cli.search_of`` only.
SearchChoice = Literal["off", "ddgs"]
SEARCH_CHOICES: tuple[str, ...] = ("off", "ddgs")
#: T-0125 criteria 3 and 12: ``ToolKit.web_search``'s values, the backends that can be on (named here, so
#: ``conductor/session.py`` names no backend)
WebSearchBackend = Literal["ddgs"]


def search_refusal(value: object) -> str:
    """Why a ``search`` value is refused (T-0125 criterion 2)."""
    return (
        f"search = {value!r} is refused: only 'off' and 'ddgs' are allowed; leave it unset for the mode's "
        "default (on in live, off in dry-run) (T-0125)"
    )


def search_without_tools_refusal() -> str:
    """Why ``search = "ddgs"`` with ``tools = "off"`` is refused (T-0125 criterion 2): nothing would search."""
    return (
        "search = 'ddgs' with tools = 'off' is refused: web search answers the model's google_search calls, "
        "which exist only with tools on; set tools to 'live' or 'dry_run', or remove search (T-0125)"
    )


def live_feedback_refusal(value: object) -> str:
    """Why a ``live_feedback`` value is refused (T-0134c A1)."""
    return (
        f"live_feedback = {value!r} is refused: only 'off', 'label' and 'label+extent' are allowed; leave it "
        "unset for the live default 'label' (T-0134c)"
    )


def action_echo_refusal(value: object) -> str:
    """Why an ``action_echo`` value is refused (T-0134c A1)."""
    return (
        f"action_echo = {value!r} is refused: only 'pixels' and 'model' are allowed; leave it unset for the "
        "live default 'model' (T-0134c)"
    )


def feedback_without_live_refusal(key: str, tools: object) -> str:
    """Why ``live_feedback``/``action_echo`` outside ``tools = "live"`` is refused (T-0134c A1): a silent no-op."""
    return (
        f"{key} with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it would "
        "silently do nothing (T-0134c)"
    )


def thought_call_recovery_refusal(value: object) -> str:
    """Why a ``thought_call_recovery`` value is refused (T-0135b A1): only a TOML bool."""
    return (
        f"thought_call_recovery = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it "
        "unset for the live default true (T-0135b)"
    )


def thought_call_recovery_without_live_refusal(tools: object) -> str:
    """Why ``thought_call_recovery`` outside ``tools = "live"`` is refused (T-0135b A1): a silent no-op."""
    return (
        f"thought_call_recovery with tools = {tools!r} is refused: it applies only with tools = 'live', and "
        "elsewhere it would silently do nothing (T-0135b)"
    )


def announced_call_reask_refusal(value: object) -> str:
    """Why an ``announced_call_reask`` value is refused (T-0172 B9): only a TOML bool."""
    return (
        f"announced_call_reask = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it "
        "unset for the default true (T-0172)"
    )


def announced_call_reask_without_tools_refusal(tools: object) -> str:
    """Why ``announced_call_reask`` with ``tools = "off"`` is refused (T-0172 B9): a silent no-op."""
    return (
        f"announced_call_reask with tools = {tools!r} is refused: it applies only with tools = 'dry_run' or "
        "'live', and with tools off it would silently do nothing (T-0172)"
    )


def live_guidance_refusal(value: object) -> str:
    """Why a ``live_guidance`` value is refused (T-0174 A3, T-0191 A3, T-0211 B3): only the strings ``v1``, ``v2``,
    ``v3`` and ``v4``."""
    return (
        f"live_guidance = {value!r} is refused: only 'v1', 'v2', 'v3' and 'v4' are allowed; leave it unset for the "
        f"live default {DEFAULT_LIVE_GUIDANCE!r} (T-0174, T-0191, T-0211)"
    )


def live_guidance_without_live_refusal(tools: object) -> str:
    """Why ``live_guidance`` outside ``tools = "live"`` is refused (T-0174 A3): a silent no-op."""
    return (
        f"live_guidance with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it "
        "would silently do nothing (T-0174)"
    )


def page_overview_refusal(value: object) -> str:
    """Why a ``page_overview`` value is refused (T-0140 A5, O-A2 Q8): only a TOML bool."""
    return (
        f"page_overview = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it unset "
        "for the live default false (T-0140)"
    )


def page_overview_without_live_refusal(tools: object) -> str:
    """Why ``page_overview`` outside ``tools = "live"`` is refused (T-0140 A5): a silent no-op."""
    return (
        f"page_overview with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it "
        "would silently do nothing (T-0140)"
    )


def auto_overview_refusal(value: object) -> str:
    """Why an ``auto_overview`` value is refused (T-0150 A6, the ``page_overview_refusal`` pattern): only a TOML
    bool."""
    return (
        f"auto_overview = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it unset "
        "for the live default false (T-0150)"
    )


def auto_overview_without_live_refusal(tools: object) -> str:
    """Why ``auto_overview`` outside ``tools = "live"`` is refused (T-0150 A6): a silent no-op."""
    return (
        f"auto_overview with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it "
        "would silently do nothing (T-0150)"
    )


def auto_and_page_overview_refusal() -> str:
    """Why the automatic overview and the ``page_overview`` tool together are refused (T-0150 A6, H-T150-2)."""
    return (
        "auto_overview = true with page_overview = true is refused: the automatic page overview and the "
        "page_overview tool are not used together; turn one of them off (T-0150, H-T150-2)"
    )


def overview_frame_refusal(value: object) -> str:
    """Why an ``overview_frame`` value is refused (T-0151 A2-9, the ``auto_overview_refusal`` pattern): only a TOML
    bool."""
    return (
        f"overview_frame = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it unset "
        "for the live default false (T-0151)"
    )


def overview_frame_without_live_refusal(tools: object) -> str:
    """Why ``overview_frame`` outside ``tools = "live"`` is refused (T-0151 A2-9): a silent no-op."""
    return (
        f"overview_frame with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it "
        "would silently do nothing (T-0151)"
    )


def overview_frame_with_refusal(other: str) -> str:
    """Why ``overview_frame = true`` beside ``auto_overview``/``page_overview``/``reground = true`` is refused
    (T-0151 A2-1/A2-9, Q-4)."""
    return (
        f"overview_frame = true with {other} = true is refused: the overview frame replaces step 1's screen with "
        f"the whole-page picture, and {other} is not used with it; turn one of them off (T-0151)"
    )


def action_outcome_refusal(value: object) -> str:
    """Why an ``action_outcome`` value is refused (T-0136 C-1, the ``overview_frame_refusal`` pattern): only a TOML
    bool."""
    return (
        f"action_outcome = {value!r} is refused: only true and false (TOML booleans) are allowed; leave it unset "
        "for the live default false (T-0136)"
    )


def action_outcome_without_live_refusal(tools: object) -> str:
    """Why ``action_outcome`` outside ``tools = "live"`` is refused (T-0136 C-1): a silent no-op."""
    return (
        f"action_outcome with tools = {tools!r} is refused: it applies only with tools = 'live', and elsewhere it "
        "would silently do nothing (T-0136)"
    )


def action_outcome_without_feedback_refusal() -> str:
    """Why ``action_outcome = true`` with ``live_feedback = "off"`` is refused (T-0136 C-1): the outcome clause lives
    in the "screen now" label, which ``off`` removes."""
    return (
        "action_outcome = true with live_feedback = 'off' is refused: the outcome clause is part of the live "
        "feedback label, which 'off' removes; set live_feedback to 'label' or 'label+extent', or turn "
        "action_outcome off (T-0136)"
    )


def visual_tokens_refusal(value: object) -> str:
    """Why a ``visual_tokens`` value is refused (T-0131 A1): only 280, 560 and 1120, as integers."""
    return (
        f"visual_tokens = {value!r} is refused: only 280, 560 and 1120 (integers) are allowed; leave it unset "
        "for the engine default of 280 per image (T-0131)"
    )


def max_frames_per_turn_refusal(value: object) -> str:
    """Why a ``max_frames_per_turn`` value is refused (T-0227 A-1): only 1, 2, 3 and 4, as integers."""
    return (
        f"max_frames_per_turn = {value!r} is refused: only 1, 2, 3 and 4 (integers) are allowed; 4 is the "
        "engine's image limit per request; leave it unset for 3 (T-0227)"
    )


def frame_interval_ms_refusal(value: object) -> str:
    """Why a ``frame_interval_ms`` value is refused (T-0227 A-1): only the ``FRAME_INTERVALS_MS`` presets."""
    return (
        f"frame_interval_ms = {value!r} is refused: only 250, 500, 750, 1000, 1500, 2000, 3000, 5000 and 10000 "
        "(integers, milliseconds) are allowed; leave it unset for 1000 (T-0227)"
    )


def reground_without_live_refusal(tools: object) -> str:
    """Why ``reground = true`` outside ``tools = "live"`` is refused (T-0131 A1): it would be a silent no-op."""
    return (
        f"reground = true with tools = {tools!r} is refused: ReGround runs only with tools = 'live', and "
        "elsewhere it would silently do nothing (T-0131)"
    )


def reground_visual_tokens_refusal(value: object) -> str:
    """Why a ``reground_visual_tokens`` value is refused (T-0132 B1): only 280, 560 and 1120, as integers."""
    return (
        f"reground_visual_tokens = {value!r} is refused: only 280, 560 and 1120 (integers) are allowed; leave "
        "it unset for pass 2 to use the main budget (T-0132)"
    )


def reground_visual_tokens_unused_refusal(tools: object, reground: object) -> str:
    """Why ``reground_visual_tokens`` with ReGround off or outside live is refused (T-0132 B1): a silent no-op."""
    if tools != "live":
        why = f"tools = {tools!r}: ReGround's pass 2 runs only with tools = 'live'"
    else:
        why = f"reground = {reground!r}: there is no pass 2 with ReGround off"
    return f"reground_visual_tokens is refused with {why}, and the budget would silently do nothing (T-0132)"


ProfileName = Literal["lean", "cpu-stt", "quality-tts", "talker", "dual-branch"]
LOOPBACK_PREFIXES = ("http://127.0.0.1", "http://localhost", "ws://127.0.0.1", "ws://localhost")


class Services(Contract):
    vllm: str = "http://127.0.0.1:8001/v1"
    stt: str = "ws://127.0.0.1:8080/api/asr-streaming"
    tts: str = "http://127.0.0.1:8081"
    talker: str | None = None
    conductor_port: int = 8000
    hud_port: int = 1420

    @field_validator("vllm", "stt", "tts", "talker")
    @classmethod
    def _loopback(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(LOOPBACK_PREFIXES):
            raise ValueError(f"non-loopback endpoint {v!r} (docs/02 §9)")
        return v


class Diffusion(Contract):
    canvas_length: int = Field(default=256, ge=32, le=1024)
    max_denoising_steps: int = Field(default=48, ge=1, le=256)
    thinking: bool = True
    max_num_seqs: int = Field(default=3, ge=1, le=4)


class Profile(Contract):
    name: ProfileName
    stt_backend: Literal["kyutai-gpu", "nemotron-cpu", "parakeet-cpu", "e2b-hears"] = "kyutai-gpu"
    tts_backend: Literal["pocket-cpu", "kyutai-1.6b-gpu", "kokoro-cpu"] = "pocket-cpu"
    talker: bool = False
    #: docs/01 R4: headset first (no AEC); PipeWire echo-cancel for speakers; in-process APM later
    aec: Literal["none", "pipewire-echo-cancel", "livekit-apm"] = "none"
    vram_budget_gib: float = Field(default=29.0, gt=0, le=32)
    services: Services = Services()
    diffusion: Diffusion = Diffusion()
    #: T-0092b: a session setting, never changed by a request or a turn; ``astra session --tools`` overrides it.
    tools: ToolsMode = "off"
    #: T-0114/T-0123: the tool-result loop (T-0111a). ``None`` (the key absent) means on whenever the resolved
    #: tools are on and off with tools off (T-0123, H-T123-1); ``astra session --tool-loop`` and
    #: ``--no-tool-loop`` override it. An explicit ``true`` with tools off is refused at startup (H-T114-2).
    tool_loop: bool | None = None
    #: T-0114: executed calls per turn with the loop on, then one report step (H-T111a-2); ``--max-tool-steps``
    #: overrides it. ``0`` would fall back to the single execute, so it is refused at load.
    max_tool_steps: int = Field(default=8, ge=1)
    #: T-0178 A1 (H-T178-2, H-T178-6): the same budget in live with web on, where ``max_tool_steps`` does not
    #: apply; inert in every other mode. ``--max-tool-steps`` overrides both.
    max_tool_steps_web: int = Field(default=20, ge=1)
    #: T-0182 B2 (James 2026-10-28): the ``ack`` clip at a thinking turn's start and the conversational interim
    #: filler, off by default; ``astra session --filler-clips``/``--no-filler-clips`` overrides it.
    filler_clips: bool = False
    #: T-0182 B4: in live with the tool loop on, the one ``still_working`` clip plays this long after llm.start.
    still_working_after_s: int = Field(default=20, ge=5)
    #: T-0111g (H-T111c-1): ``live``'s browser is headed at the desk; ``astra session --browser`` overrides it.
    browser: Literal["headed", "headless"] = "headed"
    #: T-0111g (H-T111c-3): the agent browser's viewport, width x height in CSS px.
    browser_viewport: tuple[int, int] = (1280, 800)
    #: T-0131 A1: the per-request visual-token budget (``mm_processor_kwargs.max_soft_tokens``). ``None`` sends
    #: nothing: the engine default, 280 for this checkpoint. ``astra session --visual-tokens`` overrides it.
    visual_tokens: VisualTokens | None = None
    #: T-0131 A1: ReGround (a second, never-voiced request on a crop around the first point); live only.
    #: T-0133 A1: ``None`` (the key absent) is unset: off in every mode (``LIVE_DEFAULT_REGROUND`` is ``False``,
    #: resolved in ``cli``). An explicit ``true`` outside live is refused at load; an explicit ``false`` loads everywhere.
    reground: bool | None = None
    #: T-0132 B1: pass 2's (the crop's) budget; ``None`` is the main budget (280 in live at a 1120 main budget, T-0133). Applied only when pass 2's prompt
    #: carries exactly one image (B3). ``astra session --reground-visual-tokens`` overrides it.
    reground_visual_tokens: VisualTokens | None = None
    #: T-0134c A1: the live feedback; ``None`` is ``LIVE_DEFAULT_FEEDBACK`` in live (resolved in ``cli``). Live
    #: only: a value with another ``tools`` is refused at load. ``astra session --live-feedback`` overrides it.
    live_feedback: LiveFeedbackChoice | None = None
    #: T-0134c A1: the action echo's units; ``None`` is ``LIVE_DEFAULT_ACTION_ECHO`` in live (resolved in ``cli``).
    #: Live only, as ``live_feedback``. ``astra session --action-echo`` overrides it.
    action_echo: ActionEcho | None = None
    #: T-0135b A1: thought-call recovery; ``None`` is ``LIVE_DEFAULT_THOUGHT_CALL_RECOVERY`` in live (resolved in
    #: ``cli``). Live only: a value with another ``tools`` is refused at load. ``--thought-call-recovery`` overrides it.
    thought_call_recovery: bool | None = None
    #: T-0140 A5: offer ``page_overview`` (the ``tools_v1_overview.json`` set); ``None`` is
    #: ``LIVE_DEFAULT_PAGE_OVERVIEW`` in live (resolved in ``cli``). Live only: a value with another ``tools`` is
    #: refused at load. ``astra session --page-overview`` / ``--no-page-overview`` override it.
    page_overview: bool | None = None
    #: T-0150 A6: show the whole page at step 1 when it is taller than the screen; ``None`` is
    #: ``LIVE_DEFAULT_AUTO_OVERVIEW`` in live (resolved in ``cli``). Live only, never with ``page_overview = true``;
    #: both refused at load. ``astra session --auto-overview`` / ``--no-auto-overview`` override it.
    auto_overview: bool | None = None
    #: T-0151 A2-9: on a long page, step 1 shows only the whole-page picture and its coordinates are translated to
    #: the page; ``None`` is ``LIVE_DEFAULT_OVERVIEW_FRAME`` in live (resolved in ``cli``). Live only; ``true`` never
    #: with ``auto_overview``, ``page_overview`` or ``reground = true``; all refused at load.
    #: ``astra session --overview-frame`` / ``--no-overview-frame`` override it.
    overview_frame: bool | None = None
    #: T-0136 C-1: after an executed ``click_at``/``type_at``, the next step's "screen now" label says what visibly
    #: changed; ``None`` is ``LIVE_DEFAULT_ACTION_OUTCOME`` in live (resolved in ``cli``). Live only; ``true`` never
    #: with ``live_feedback = "off"``; both refused at load. ``astra session --action-outcome`` /
    #: ``--no-action-outcome`` override it.
    action_outcome: bool | None = None
    #: T-0137 B-9: a browser action that repeats an earlier step on the same screen is stated, and a no-progress
    #: action is refused; ``None`` is ``LIVE_DEFAULT_REPEAT_GUARD`` in live (resolved in ``cli``). Live only, refused
    #: at load elsewhere. ``astra session --repeat-guard`` / ``--no-repeat-guard`` override it.
    repeat_guard: bool | None = None
    #: T-0172 B9: re-ask once when a tools-on step only announces an action and makes no call; ``None`` is
    #: ``DEFAULT_ANNOUNCED_CALL_REASK`` in ``dry_run`` and ``live`` (resolved in ``cli``). Refused at load with
    #: ``tools = "off"``. ``astra session --announced-call-reask {on,off}`` overrides it.
    announced_call_reask: bool | None = None
    #: T-0174 A3: the live tool guidance version (``"v1"``, ``"v2"``, ``"v3"`` (T-0191) or ``"v4"`` (T-0211)); ``None`` is ``DEFAULT_LIVE_GUIDANCE`` in
    #: live (resolved in ``cli``). Live only, refused at load elsewhere. ``astra session --live-guidance`` overrides it.
    live_guidance: str | None = None
    #: T-0125 criterion 2: web search for ``google_search``; ``None`` is the mode's default (on in live, off in
    #: dry-run, resolved in ``cli``). ``"ddgs"`` with ``tools = "off"`` is refused at load.
    #: ``astra session --search {off,ddgs}`` overrides it.
    search: SearchChoice | None = None
    #: T-0160 C-1 (ruling 2): a live session's agent browser reaches public ``https`` sites through the egress
    #: proxy; ``false`` keeps it on the loopback start pages. Applies only with the resolved tools mode ``live``
    #: (elsewhere it has no effect). ``astra session --no-web`` turns it off.
    browser_web: bool = True
    #: T-0227 A-1: the most screens one spoken turn shows the model; ``None`` is ``DEFAULT_MAX_FRAMES_PER_TURN``
    #: (resolved in ``cli``). Valid in every tools mode. ``astra session --max-frames-per-turn`` overrides it.
    max_frames_per_turn: int | None = None
    #: T-0227 A-1: the HUD's capture interval in ms; ``None`` is ``DEFAULT_FRAME_INTERVAL_MS`` (resolved in
    #: ``cli``). Valid in every tools mode. ``astra session --frame-interval-ms`` overrides it.
    frame_interval_ms: int | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def _tools_known(cls, v: object) -> object:
        if v not in TOOLS_MODES:
            raise ValueError(tools_refusal(v))
        return v

    @field_validator("visual_tokens", mode="before")
    @classmethod
    def _visual_tokens_known(cls, v: object) -> object:
        # ``type(v) is int`` refuses the string "1120" and ``True`` alike (lax coercion would not).
        if v is not None and not (type(v) is int and v in VISUAL_TOKENS):
            raise ValueError(visual_tokens_refusal(v))
        return v

    @field_validator("max_frames_per_turn", mode="before")
    @classmethod
    def _max_frames_per_turn_known(cls, v: object) -> object:
        if v is not None and not (type(v) is int and v in FRAMES_PER_TURN_CHOICES):
            raise ValueError(max_frames_per_turn_refusal(v))
        return v

    @field_validator("frame_interval_ms", mode="before")
    @classmethod
    def _frame_interval_ms_known(cls, v: object) -> object:
        if v is not None and not (type(v) is int and v in FRAME_INTERVALS_MS):
            raise ValueError(frame_interval_ms_refusal(v))
        return v

    @field_validator("reground_visual_tokens", mode="before")
    @classmethod
    def _reground_visual_tokens_known(cls, v: object) -> object:
        if v is not None and not (type(v) is int and v in VISUAL_TOKENS):
            raise ValueError(reground_visual_tokens_refusal(v))
        return v

    @field_validator("live_feedback", mode="before")
    @classmethod
    def _live_feedback_known(cls, v: object) -> object:
        if v is not None and not (isinstance(v, str) and v in LIVE_FEEDBACK_CHOICES):
            raise ValueError(live_feedback_refusal(v))
        return v

    @field_validator("action_echo", mode="before")
    @classmethod
    def _action_echo_known(cls, v: object) -> object:
        if v is not None and not (isinstance(v, str) and v in ACTION_ECHOES):
            raise ValueError(action_echo_refusal(v))
        return v

    @field_validator("thought_call_recovery", mode="before")
    @classmethod
    def _thought_call_recovery_bool(cls, v: object) -> object:
        # ``type(v) is bool`` refuses "true", "off", 1 and 0 alike (lax coercion would not)
        if v is not None and type(v) is not bool:
            raise ValueError(thought_call_recovery_refusal(v))
        return v

    @model_validator(mode="after")
    def _thought_call_recovery_needs_live(self) -> Profile:
        # T-0135b A1 (the T-0134c A1 pattern): a value outside live would silently do nothing
        if self.thought_call_recovery is not None and self.tools != "live":
            raise ValueError(thought_call_recovery_without_live_refusal(self.tools))
        return self

    @field_validator("page_overview", mode="before")
    @classmethod
    def _page_overview_bool(cls, v: object) -> object:
        # T-0140 (O-A2 Q8): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(page_overview_refusal(v))
        return v

    @model_validator(mode="after")
    def _page_overview_needs_live(self) -> Profile:
        # T-0140 A5 (the T-0135b A1 pattern): a value outside live would silently do nothing
        if self.page_overview is not None and self.tools != "live":
            raise ValueError(page_overview_without_live_refusal(self.tools))
        return self

    @field_validator("auto_overview", mode="before")
    @classmethod
    def _auto_overview_bool(cls, v: object) -> object:
        # T-0150 A6 (the T-0140 pattern): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(auto_overview_refusal(v))
        return v

    @model_validator(mode="after")
    def _auto_overview_needs_live(self) -> Profile:
        # T-0150 A6: a value outside live would silently do nothing; with the page_overview tool it is refused
        if self.auto_overview is not None and self.tools != "live":
            raise ValueError(auto_overview_without_live_refusal(self.tools))
        if self.auto_overview is True and self.page_overview is True:
            raise ValueError(auto_and_page_overview_refusal())
        return self

    @field_validator("overview_frame", mode="before")
    @classmethod
    def _overview_frame_bool(cls, v: object) -> object:
        # T-0151 A2-9 (the T-0150 pattern): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(overview_frame_refusal(v))
        return v

    @model_validator(mode="after")
    def _overview_frame_needs_live(self) -> Profile:
        # T-0151 A2-9: a value outside live would silently do nothing; ``true`` is never used with the automatic
        # overview, the page_overview tool or ReGround (O-A2 C1: only ``true`` beside them is refused)
        if self.overview_frame is not None and self.tools != "live":
            raise ValueError(overview_frame_without_live_refusal(self.tools))
        if self.overview_frame is True:
            for other in ("auto_overview", "page_overview", "reground"):
                if getattr(self, other) is True:
                    raise ValueError(overview_frame_with_refusal(other))
        return self

    @field_validator("action_outcome", mode="before")
    @classmethod
    def _action_outcome_bool(cls, v: object) -> object:
        # T-0136 C-1 (the T-0151 pattern): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(action_outcome_refusal(v))
        return v

    @model_validator(mode="after")
    def _action_outcome_needs_live(self) -> Profile:
        # T-0136 C-1: a value outside live would silently do nothing; ``true`` needs the label (feedback not off)
        if self.action_outcome is not None and self.tools != "live":
            raise ValueError(action_outcome_without_live_refusal(self.tools))
        if self.action_outcome is True and self.live_feedback == "off":
            raise ValueError(action_outcome_without_feedback_refusal())
        return self

    @field_validator("repeat_guard", mode="before")
    @classmethod
    def _repeat_guard_bool(cls, v: object) -> object:
        # T-0137 B-9 (the T-0136 pattern): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(
                f"repeat_guard = {v!r} is refused: only true and false (TOML booleans) are allowed; leave it "
                "unset for the live default false (T-0137)"
            )
        return v

    @model_validator(mode="after")
    def _repeat_guard_needs_live(self) -> Profile:
        # T-0137 B-9: a value outside live would silently do nothing
        if self.repeat_guard is not None and self.tools != "live":
            raise ValueError(
                f"repeat_guard with tools = {self.tools!r} is refused: it applies only with tools = 'live', and "
                "elsewhere it would silently do nothing (T-0137)"
            )
        return self

    @field_validator("announced_call_reask", mode="before")
    @classmethod
    def _announced_call_reask_bool(cls, v: object) -> object:
        # T-0172 B9 (the T-0137 pattern): ``type(v) is bool`` refuses "true", "on", 1 and 0 alike
        if v is not None and type(v) is not bool:
            raise ValueError(announced_call_reask_refusal(v))
        return v

    @model_validator(mode="after")
    def _announced_call_reask_needs_tools(self) -> Profile:
        # T-0172 B9: with tools off there is no call to re-ask for (an absent ``tools`` key is off)
        if self.announced_call_reask is not None and self.tools == "off":
            raise ValueError(announced_call_reask_without_tools_refusal(self.tools))
        return self

    @field_validator("live_guidance", mode="before")
    @classmethod
    def _live_guidance_known(cls, v: object) -> object:
        # T-0174 A3: ``isinstance(v, str)`` refuses TOML integers and booleans as well as unknown names
        if v is not None and not (isinstance(v, str) and v in LIVE_GUIDANCE_VERSIONS):
            raise ValueError(live_guidance_refusal(v))
        return v

    @model_validator(mode="after")
    def _live_guidance_needs_live(self) -> Profile:
        # T-0174 A3: a value outside live would silently do nothing (an absent ``tools`` key is off)
        if self.live_guidance is not None and self.tools != "live":
            raise ValueError(live_guidance_without_live_refusal(self.tools))
        return self

    @field_validator("search", mode="before")
    @classmethod
    def _search_known(cls, v: object) -> object:
        # ``isinstance(v, str)`` refuses ``true`` (a TOML bool) as well as unknown names
        if v is not None and not (isinstance(v, str) and v in SEARCH_CHOICES):
            raise ValueError(search_refusal(v))
        return v

    @model_validator(mode="after")
    def _search_needs_tools(self) -> Profile:
        # T-0125 criterion 2: with tools off nothing would search. An absent ``tools`` key is left to ``cli``
        # (``--tools`` may resolve it; resolved off it exits 2), the ``reground_visual_tokens`` pattern
        if self.search == "ddgs" and self.tools == "off" and "tools" in self.model_fields_set:
            raise ValueError(search_without_tools_refusal())
        return self

    @model_validator(mode="after")
    def _feedback_needs_live(self) -> Profile:
        # T-0134c A1 (the T-0131 ``reground`` pattern): a value outside live would silently do nothing
        for key in ("live_feedback", "action_echo"):
            if getattr(self, key) is not None and self.tools != "live":
                raise ValueError(feedback_without_live_refusal(key, self.tools))
        return self

    @model_validator(mode="after")
    def _live_needs_the_loop(self) -> Profile:
        if self.tools == "live" and self.tool_loop is False:
            raise ValueError(live_without_loop_refusal())
        return self

    @model_validator(mode="after")
    def _reground_needs_live(self) -> Profile:
        if self.reground and self.tools != "live":
            raise ValueError(reground_without_live_refusal(self.tools))
        return self

    @model_validator(mode="after")
    def _reground_visual_tokens_needs_reground(self) -> Profile:
        # an explicit non-live ``tools`` refuses here; an absent ``tools`` (``astra session --tools live`` may
        # resolve it) is left to ``cli``, which exits 2 when the resolved mode is not live (T-0132 B1)
        explicit_tools = "tools" in self.model_fields_set
        if self.reground_visual_tokens is not None and (
            (explicit_tools and self.tools != "live") or self.reground is False
        ):
            raise ValueError(reground_visual_tokens_unused_refusal(self.tools, self.reground))
        return self


class Config(Contract):
    profile: str = "lean"
    profiles: dict[str, Profile]

    def active(self) -> Profile:
        return self.profiles[self.profile]


DEFAULTS: dict[str, Any] = {
    "profile": "lean",
    "profiles": {
        "lean": {"name": "lean"},
        "cpu-stt": {"name": "cpu-stt", "stt_backend": "nemotron-cpu"},
        "quality-tts": {
            "name": "quality-tts",
            "stt_backend": "nemotron-cpu",
            "tts_backend": "kyutai-1.6b-gpu",
        },
        "talker": {
            "name": "talker",
            "talker": True,
            "services": {"talker": "http://127.0.0.1:8003"},
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: Path | None = None) -> Config:
    """Defaults deep-merged with ``astra.toml``: overriding one field keeps the profile's others."""
    data: dict[str, Any] = _deep_merge(DEFAULTS, {})
    if path is not None and path.exists():
        with path.open("rb") as f:
            user = tomllib.load(f)
        data = _deep_merge(DEFAULTS, user)
    for k, v in data["profiles"].items():
        v.setdefault("name", k)
    return Config.model_validate(data)
