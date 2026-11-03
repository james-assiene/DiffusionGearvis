"""The settings catalog (T-0213 A1-1): every setting the project has measured or offered, as data.

One frozen :class:`SettingEntry` per row of the T-0213 catalog table. The HUD's Settings panel renders these
entries and the docs quote them, so the explanation of a setting lives here and nowhere else. The ``evidence``
numbers are quoted from ``PROGRESS.md`` Runs (or the task file each entry cites), under :data:`CATALOG_REGIME`
unless an entry says otherwise.

This module is data only: it imports ``local_astra.config`` (the constants every ``values`` and ``default`` must
equal) and nothing from ``conductor/``, ``computer/`` or ``search/``. Defaults that live in ``cli`` (the wait tool,
the screen reminder, the completion check, the tool-step budgets) are literals here, pinned against ``cli`` by
``tests/unit/test_t0213_catalog.py``, because importing ``cli`` would import the conductor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from local_astra.config import (
    ACTION_ECHOES,
    DEFAULT_ANNOUNCED_CALL_REASK,
    DEFAULT_FRAME_INTERVAL_MS,
    DEFAULT_LIVE_GUIDANCE,
    DEFAULT_MAX_FRAMES_PER_TURN,
    FRAME_INTERVALS_MS,
    FRAMES_PER_TURN_CHOICES,
    LIVE_DEFAULT_ACTION_ECHO,
    LIVE_DEFAULT_ACTION_OUTCOME,
    LIVE_DEFAULT_AUTO_OVERVIEW,
    LIVE_DEFAULT_FEEDBACK,
    LIVE_DEFAULT_OVERVIEW_FRAME,
    LIVE_DEFAULT_PAGE_OVERVIEW,
    LIVE_DEFAULT_REGROUND,
    LIVE_DEFAULT_REGROUND_VISUAL_TOKENS,
    LIVE_DEFAULT_REPEAT_GUARD,
    LIVE_DEFAULT_THOUGHT_CALL_RECOVERY,
    LIVE_DEFAULT_VISUAL_TOKENS,
    LIVE_FEEDBACK_CHOICES,
    LIVE_GUIDANCE_VERSIONS,
    SEARCH_CHOICES,
    TOOLS_MODES,
    VISUAL_TOKENS,
)

__all__ = [
    "CATALOG",
    "CATALOG_REGIME",
    "GROUPS",
    "KINDS",
    "NOT_A_SETTING",
    "RELAUNCH_IN_FORCE",
    "SettingEntry",
    "SettingKind",
    "by_key",
]

#: T-0217 (R-27): in live, the one-way relaunch rows whose catalog command sets one value only: when that value is
#: already in force, the session's relaunch hint is the flag that sets the other value, or ``None`` when no single flag
#: does (``--no-web`` cannot turn web back on; ``--tools live`` is in force in live). Key -> (the command's value, the
#: inverse flag text). Not part of a catalog entry, so the panel's catalog is unchanged.
RELAUNCH_IN_FORCE: Final[Mapping[str, tuple[bool | str, str | None]]] = MappingProxyType(
    {
        "search": ("off", "--search ddgs"),
        "browser": ("headless", "--browser headed"),
        "wait_tool": (False, "--wait-tool on"),
        "page_overview": (True, "--no-page-overview"),
        "action_echo": ("pixels", "--action-echo model"),
        "filler_clips": (True, "--no-filler-clips"),
        "browser_web": (False, None),
        "tools": ("live", None),
    }
)

#: The four change kinds (T-0213 A1-1): applied at the next turn boundary; the same, with one cold prefill on the
#: first turn after the change; only by relaunching; not a setting at all.
SettingKind = Literal["next_turn", "next_turn_cold", "relaunch", "fixed"]
KINDS: Final[tuple[str, ...]] = ("next_turn", "next_turn_cold", "relaunch", "fixed")

#: The panel's groups, in render order (T-0213 B-1).
GROUPS: Final[tuple[str, ...]] = (
    "Seeing the screen",
    "Acting in the browser",
    "Checking its work",
    "Guidance and prompt",
    "Search and web",
    "Speech",
    "Engine (relaunch only)",
)

#: The regime every ``evidence`` number is quoted under unless the entry says otherwise (T-0213 catalog).
CATALOG_REGIME: Final[str] = (
    "DiffusionGemma 26B-A4B NVFP4 rev 2ea8372, vLLM 0.29.0, lean, canvas 256, 48 steps, thinking on; headless "
    "Chromium 1280x800, live, typed turns, order seed 0, engine unseeded; n as given. Suite run-to-run noise is "
    "about 10 points per 100 (T-0137 C)."
)

_BOOL: Final[tuple[bool, ...]] = (False, True)


@dataclass(frozen=True)
class SettingEntry:
    """One catalog row (T-0213 A1-1).

    ``flag`` is the exact CLI spelling of the option that sets it (``""`` for a setting no flag sets), ``flags``
    every spelling of it (``--x`` and ``--no-x``). ``values`` is the tuple of allowed values (``()`` for an open
    range, such as a step count or free text). ``default`` is the resolved live default (the ``config`` constant).
    ``relaunch`` is the exact command that changes a setting the panel cannot, ``None`` when the panel can (or
    when it is not a setting).
    """

    key: str
    label: str
    group: str
    kind: SettingKind
    v1_changeable: bool
    values: tuple[object, ...]
    default: object
    default_note: str
    flag: str
    flags: tuple[str, ...]
    profile_key: str | None
    what: str
    useful_when: str
    not_useful_when: str
    evidence: str
    cost: str
    relaunch: str | None


_SESSION = "diffusion_gearvis session --tools live"

CATALOG: Final[tuple[SettingEntry, ...]] = (
    # -- Seeing the screen -----------------------------------------------------------------------------------
    SettingEntry(
        key="visual_tokens",
        label="Visual tokens per image",
        group="Seeing the screen",
        kind="next_turn_cold",
        v1_changeable=True,
        values=VISUAL_TOKENS,
        default=LIVE_DEFAULT_VISUAL_TOKENS,
        default_note="1120 in live (T-0132 A1)",
        flag="--visual-tokens",
        flags=("--visual-tokens",),
        profile_key="visual_tokens",
        what="How many tokens each screenshot is encoded into. More tokens show smaller text and finer detail.",
        useful_when="small text, dense pages, precise clicks",
        not_useful_when="fast chat about a simple screen; speed matters more than precision",
        evidence=(
            "T-0131 D, n=25 per arm: success 280 5/25, 560 10/25, 1120 15/25; hit rate 0.35 / 0.88 / 1.00"
        ),
        cost="≈275 / 524 / 1094 prompt tokens per image (T-0131 P1); history images re-prefilled once",
        relaunch=None,
    ),
    SettingEntry(
        key="max_frames_per_turn",
        label="Screens per turn (max images)",
        group="Seeing the screen",
        kind="next_turn_cold",
        v1_changeable=True,
        values=FRAMES_PER_TURN_CHOICES,
        default=DEFAULT_MAX_FRAMES_PER_TURN,
        default_note="3 (T-0059)",
        flag="--max-frames-per-turn",
        flags=("--max-frames-per-turn",),
        profile_key="max_frames_per_turn",
        what=(
            "The most screens one spoken turn shows Gearvis: the screen when you started speaking plus screens "
            "that changed while you spoke. The engine allows 4 images per request."
        ),
        useful_when="the screen changes several times while you ask, and the steps in between matter",
        not_useful_when=(
            "a mostly still screen; at 4 the turn's own screens take every image slot, so no earlier screen "
            "stays an image (each becomes its caption line)"
        ),
        evidence=(
            "computed, not measured: 4 images is the engine limit (services/vllm.sh); the latency cost is "
            "unmeasured (T-0059 D is still owed)"
        ),
        cost=(
            "≈1,094 prompt tokens per screen at 1120 visual tokens, ≈275 at 280 (T-0131 P1); a change re-prefills "
            "the history images once"
        ),
        relaunch=None,
    ),
    SettingEntry(
        key="frame_interval_ms",
        label="Screen capture interval (ms)",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=FRAME_INTERVALS_MS,
        default=DEFAULT_FRAME_INTERVAL_MS,
        default_note="1000 ms, one frame per second per source (T-0227)",
        flag="--frame-interval-ms",
        flags=("--frame-interval-ms",),
        profile_key="frame_interval_ms",
        what=(
            "How often the HUD page sends a frame of your shared screen and camera, in milliseconds. It does "
            "not change the agent browser's screenshots, one per step."
        ),
        useful_when="a fast-changing screen, where a shorter interval notices changes sooner",
        not_useful_when=(
            "a loaded box or a still screen; a longer interval means the screen Gearvis answers about can be "
            "that old"
        ),
        evidence=(
            "computed, not measured (T-0227); applies to the next page that connects when none is connected"
        ),
        cost=(
            "page CPU: ≈180 KiB to encode per frame; ≈42 MiB per source in the 60 s server ring at 250 ms; one "
            "session-log line per frame when logging is on"
        ),
        relaunch=None,
    ),
    SettingEntry(
        key="reground",
        label="ReGround",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_REGROUND,
        default_note="off (T-0133)",
        flag="--reground",
        flags=("--reground", "--no-reground"),
        profile_key="reground",
        what=(
            "Before each click or type, Gearvis asks again on a zoomed crop around the point and acts where that "
            "second answer points."
        ),
        useful_when=(
            "small or dense targets: on the three.js editor click misses fell 35/96 → 10/74 (T-0212, n = 15 "
            "per arm); turn the overview frame off first (`--reground` at launch does it)"
        ),
        not_useful_when=(
            "ordinary pages, where it adds a request per click for no gain (T-0132 C: off 65, crop 280 "
            "64/100); refused with the overview frame"
        ),
        evidence=(
            "T-0212 three.js n = 15 per arm: misses 35/96 → 10/74, success 0/15 → 1/15; T-0214 with action "
            "outcome: misses 10/64; T-0132 C n=100: off 65, crop 1120 66, crop 280 64"
        ),
        cost=(
            "one extra, never-voiced request per click; pass-2 p50 352 ms (crop 280) / 724 ms (crop 1120) "
            "(T-0132 C)"
        ),
        relaunch=None,
    ),
    SettingEntry(
        key="reground_visual_tokens",
        label="ReGround crop tokens",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=VISUAL_TOKENS,
        default=LIVE_DEFAULT_REGROUND_VISUAL_TOKENS,
        default_note="280 with ReGround on at a 1120 main budget (T-0133 A5), else the main budget",
        flag="--reground-visual-tokens",
        flags=("--reground-visual-tokens",),
        profile_key="reground_visual_tokens",
        what="The visual-token budget of ReGround's zoomed crop. It applies only with ReGround on.",
        useful_when=(
            "a smaller crop costs half the pass-2 time for about the same result: T-0132 C n=100 crop 280 "
            "64/100 vs crop 1120 66/100 (ReGround off 65/100)"
        ),
        not_useful_when="with ReGround off it does nothing, so it is refused",
        evidence="T-0132 C n=100: on crop 1120 66, on crop 280 64; pass 1 hit already 0.994",
        cost="pass-2 p50 352 ms (crop 280) / 724 ms (crop 1120)",
        relaunch=None,
    ),
    SettingEntry(
        key="live_feedback",
        label="Screen-now label",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=LIVE_FEEDBACK_CHOICES,
        default=LIVE_DEFAULT_FEEDBACK,
        default_note="label (T-0134c)",
        flag="--live-feedback",
        flags=("--live-feedback",),
        profile_key="live_feedback",
        what=(
            "A short label after the current screenshot saying it is the screen now; label+extent also says how "
            "much page lies above and below."
        ),
        useful_when="the model loses track of which screenshot is current",
        not_useful_when="rarely; `off` only to reproduce old runs",
        evidence=(
            "T-0134 B n=100: A 61, B 64, C 70 (inconclusive); calls left in thought 13 / 6 / 3. Arm mapping "
            "A=off, B=label (`config.py:70`), C=label+extent assumed (Q2)"
        ),
        cost="a few tokens per step",
        relaunch=None,
    ),
    SettingEntry(
        key="page_overview",
        label="Page overview tool",
        group="Seeing the screen",
        kind="next_turn_cold",
        v1_changeable=False,
        values=_BOOL,
        default=LIVE_DEFAULT_PAGE_OVERVIEW,
        default_note="off",
        flag="--page-overview",
        flags=("--page-overview", "--no-page-overview"),
        profile_key="page_overview",
        what="Offers the model a tool that shows one picture of the whole page for one step.",
        useful_when="only for experiments: the model never called it (T-0140 B)",
        not_useful_when="everyday use: no gain from offering it (T-0140 B: 66/100 with it, 71 without)",
        evidence="T-0140 B n=100: tool never called; 71 vs 66",
        cost="adds a tool schema",
        relaunch=f"{_SESSION} --page-overview",
    ),
    SettingEntry(
        key="auto_overview",
        label="Automatic page overview",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_AUTO_OVERVIEW,
        default_note="off",
        flag="--auto-overview",
        flags=("--auto-overview", "--no-auto-overview"),
        profile_key="auto_overview",
        what=(
            "On a page taller than the screen, each turn's first request also shows the whole page as a "
            "picture, without the model asking."
        ),
        useful_when=(
            "rarely: the default overview frame already shows long pages; T-0150 B no change overall (S4 0 → "
            "20/20 but S5 13 → 0/20); turn the overview frame off first (`--auto-overview` at launch does it)"
        ),
        not_useful_when=(
            "most pages: the model clicks the overview's points on the live screen (T-0150 B: S5 13 → 0/20)"
        ),
        evidence="T-0150 B n=100: S4 0 → 20/20 but S5 13 → 0/20; false claims 7 → 20/80",
        cost="one full-page capture per turn",
        relaunch=None,
    ),
    SettingEntry(
        key="overview_frame",
        label="Overview frame",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_OVERVIEW_FRAME,
        default_note="on, on 1.05–2.0-screen pages (T-0152)",
        flag="--overview-frame",
        flags=("--overview-frame", "--no-overview-frame"),
        profile_key="overview_frame",
        what=(
            "On a page up to two screens tall, step 1 shows only the whole page as a picture, and a click on it "
            "is translated to the page."
        ),
        useful_when="pages a little taller than the screen",
        not_useful_when=(
            "pages over two screens tall or shorter than 1.05 screens, where it does not apply; refused with "
            "ReGround"
        ),
        evidence="T-0151 B n=100: success 72 → 94; S4 0 → 20/20; over-clicks 1 → 4/60, on tasks shown no frame",
        cost="none extra",
        relaunch=None,
    ),
    SettingEntry(
        key="context_helpers",
        label="Context helpers (page text, PDF text, cells, accessibility tree)",
        group="Seeing the screen",
        kind="next_turn",
        v1_changeable=False,
        values=(),
        default=False,
        default_note="off (D12a)",
        flag="",
        flags=(),
        profile_key=None,
        what=(
            "Text assists sent beside the pixels; they are changed in the Context helpers panel, which already "
            "applies them at once."
        ),
        useful_when="text the pixels cannot carry",
        not_useful_when="screens the model reads from the pixels, which is the default (every turn logs which were on)",
        evidence="T-0162 B-6: already instant from the Context helpers panel; pixel-first rule (D12a)",
        cost="page text in the suffix",
        relaunch=None,
    ),
    # -- Acting in the browser -------------------------------------------------------------------------------
    SettingEntry(
        key="action_echo",
        label="Action echo units",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=ACTION_ECHOES,
        default=LIVE_DEFAULT_ACTION_ECHO,
        default_note="model",
        flag="--action-echo",
        flags=("--action-echo",),
        profile_key="action_echo",
        what="The units an executed click or type reports its point in: the page's pixels or the model's 0-999 units.",
        useful_when="pixels only to reproduce runs from before T-0134c (arm A)",
        not_useful_when="everyday use: model units are the units the model clicks in",
        evidence="T-0134 B: arm B (model units + label) 64/100 vs arm A (pixels, no label) 61/100",
        cost="none",
        relaunch=f"{_SESSION} --action-echo pixels",
    ),
    SettingEntry(
        key="thought_call_recovery",
        label="Thought-call recovery",
        group="Acting in the browser",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_THOUGHT_CALL_RECOVERY,
        default_note="on (T-0135b)",
        flag="--thought-call-recovery",
        flags=("--thought-call-recovery",),
        profile_key="thought_call_recovery",
        what="Executes a valid tool call the model left at the end of an unclosed thought instead of dropping it.",
        useful_when="always in live",
        not_useful_when="debugging the model's raw behaviour",
        evidence="T-0135 B n=100: success 66 → 67; S3 16 → 20/20; silent tail calls 4 → 0",
        cost="none",
        relaunch=None,
    ),
    SettingEntry(
        key="wait_tool",
        label="Wait tool",
        group="Acting in the browser",
        kind="next_turn_cold",
        v1_changeable=False,
        values=_BOOL,
        default=True,
        default_note="on (T-0202 D)",
        flag="--wait-tool",
        flags=("--wait-tool",),
        profile_key=None,
        what="Offers a wait tool: about one second without touching the page, then a fresh screenshot.",
        useful_when="slow-loading pages (T-0202 C probe 1/10 → 8/10)",
        not_useful_when="fast pages, where it is not called (T-0202 C: no wait in 30 suite turns)",
        evidence="T-0202 C: probe 1/10 → 8/10; suite n=30: 18 vs 17",
        cost="≈1 s per wait; max 3 in a row, 6 per request",
        relaunch=f"{_SESSION} --wait-tool off",
    ),
    SettingEntry(
        key="max_tool_steps",
        label="Tool steps per turn",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=(),
        default=8,
        default_note="8, web 20 (T-0178)",
        flag="--max-tool-steps",
        flags=("--max-tool-steps",),
        profile_key="max_tool_steps",
        what="How many executed tool calls one turn may make before Gearvis gives its report.",
        useful_when="raise it for long tasks that run out of steps",
        not_useful_when="short tasks: a stuck turn takes longer to give its honest ending",
        evidence="T-0178: 8, web 20 (James's ruling); T-0210 C worst-case turn budget",
        cost="a stuck turn runs longer",
        relaunch=f"{_SESSION} --max-tool-steps N",
    ),
    SettingEntry(
        key="max_tool_steps_web",
        label="Tool steps per turn (web)",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=(),
        default=20,
        default_note="20 in live with web on (T-0178)",
        flag="--max-tool-steps",
        flags=("--max-tool-steps",),
        profile_key="max_tool_steps_web",
        what="The same step budget in live with web on, where the plain budget does not apply.",
        useful_when="raise it for long tasks on real websites that run out of steps",
        not_useful_when="short tasks: a stuck turn takes longer to give its honest ending",
        evidence="T-0178: 8, web 20 (James's ruling); T-0210 C worst-case turn budget",
        cost="a stuck turn runs longer",
        relaunch=f"{_SESSION} --max-tool-steps N",
    ),
    SettingEntry(
        key="browser",
        label="Agent browser window",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=("headed", "headless"),
        default="headed",
        default_note="headed",
        flag="--browser",
        flags=("--browser",),
        profile_key="browser",
        what="Whether the agent's own browser shows a window at the desk or runs without one.",
        useful_when="headed: watching it work at the desk",
        not_useful_when="headless: a box with no display, or scripted runs",
        evidence="T-0111g (H-T111c-1): headed at the desk",
        cost="none",
        relaunch=f"{_SESSION} --browser headless",
    ),
    SettingEntry(
        key="browser_viewport",
        label="Agent browser size",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=(),
        default=(1280, 800),
        default_note="1280×800",
        flag="",
        flags=(),
        profile_key="browser_viewport",
        what="The agent browser's viewport in CSS pixels; set only in the profile.",
        useful_when="a page that needs a bigger or smaller window",
        not_useful_when="comparing with the measured runs: every A/B ran at 1280×800",
        evidence="T-0111g (H-T111c-3): 1280×800, the size every A/B ran at",
        cost="none",
        relaunch="set browser_viewport in astra.toml, then relaunch diffusion_gearvis session",
    ),
    SettingEntry(
        key="tools",
        label="Tools mode",
        group="Acting in the browser",
        kind="relaunch",
        v1_changeable=False,
        values=TOOLS_MODES,
        default="off",
        default_note="off (profile)",
        flag="--tools",
        flags=("--tools",),
        profile_key="tools",
        what="Whether Gearvis has tools: off, dry-run (calls described, none executed) or live (its own browser).",
        useful_when="live: Gearvis uses its own browser; dry-run: see the calls it would make",
        not_useful_when="off is enough for talking about your shared screen",
        evidence="T-0092b, T-0111g: off by default; live needs the tool loop",
        cost="the tool schemas are in the prefix",
        relaunch=_SESSION,
    ),
    # -- Checking its work -----------------------------------------------------------------------------------
    SettingEntry(
        key="action_outcome",
        label="Action outcome",
        group="Checking its work",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_ACTION_OUTCOME,
        default_note="off",
        flag="--action-outcome",
        flags=("--action-outcome", "--no-action-outcome"),
        profile_key="action_outcome",
        what=(
            "After a click or type, the next screen-now label says what visibly changed, measured from the "
            "before and after screenshots."
        ),
        useful_when="experiments only",
        not_useful_when=(
            "everyday use: no measured gain (T-0136 D, T-0212, T-0214); needs `live_feedback` ≠ off"
        ),
        evidence=(
            "T-0136 D n=100: 71 vs 69, false claims 23 → 23; T-0212 three.js n = 15: success 0/15, misses "
            "23/69 vs base 35/96; T-0214 with ReGround: no sign beyond ReGround alone"
        ),
        cost="one clause per step",
        relaunch=None,
    ),
    SettingEntry(
        key="repeat_guard",
        label="Repeat guard",
        group="Checking its work",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=LIVE_DEFAULT_REPEAT_GUARD,
        default_note="on (T-0179, James's ruling)",
        flag="--repeat-guard",
        flags=("--repeat-guard", "--no-repeat-guard"),
        profile_key="repeat_guard",
        what=(
            "States when an action repeats an earlier step on the same screen, and refuses a fourth no-change "
            "repeat or a third visit to one screen."
        ),
        useful_when="stops stuck loops early",
        not_useful_when="when the page really needs a repeated click",
        evidence="T-0137 C n=100: 61 → 62, no change; over-actions 11 → 8",
        cost="none",
        relaunch=None,
    ),
    SettingEntry(
        key="screen_reminder",
        label="Screen reminder",
        group="Checking its work",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=True,
        default_note="on (T-0201 D)",
        flag="--screen-reminder",
        flags=("--screen-reminder",),
        profile_key=None,
        what="One sentence after each labelled step's tool response saying the last image above is the screen now.",
        useful_when="always in live",
        not_useful_when="rarely; `off` only to reproduce runs from before T-0201 D",
        evidence='T-0201 C: "can\'t see the screen" 23/65 → 8/65; suite n=30: 19 vs 18',
        cost="one sentence per step",
        relaunch=None,
    ),
    SettingEntry(
        key="completion_check",
        label="Completion check",
        group="Checking its work",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=True,
        default_note="on (T-0207 C)",
        flag="--completion-check",
        flags=("--completion-check",),
        profile_key=None,
        what=(
            "Before a final answer that follows a browser action is voiced, the model sees the screen now and "
            "confirms or continues."
        ),
        useful_when="browser tasks that end in a claim",
        not_useful_when="pure chat (it never fires there)",
        evidence=(
            'T-0207 B: false "done" caught 50/50, true "done" kept 50/50; suite n=30 false success 5 → 2'
        ),
        cost="one extra request after a no-call answer that follows an action",
        relaunch=None,
    ),
    SettingEntry(
        key="announced_call_reask",
        label="Re-ask an announced call",
        group="Checking its work",
        kind="next_turn",
        v1_changeable=True,
        values=_BOOL,
        default=DEFAULT_ANNOUNCED_CALL_REASK,
        default_note="on (T-0172)",
        flag="--announced-call-reask",
        flags=("--announced-call-reask",),
        profile_key="announced_call_reask",
        what="When an answer only announces an action and makes no call, the model is asked once more for that call.",
        useful_when="the model says it will act and does not",
        not_useful_when="rarely; `off` only to see the model's raw behaviour",
        evidence="T-0172: James's ruling 2026-10-27; no A/B found (Q2)",
        cost="at most one extra request per turn",
        relaunch=None,
    ),
    # -- Guidance and prompt ---------------------------------------------------------------------------------
    SettingEntry(
        key="live_guidance",
        label="Screenshot guidance",
        group="Guidance and prompt",
        kind="next_turn_cold",
        v1_changeable=True,
        values=LIVE_GUIDANCE_VERSIONS,
        default=DEFAULT_LIVE_GUIDANCE,
        default_note="v3 (T-0191 C)",
        flag="--live-guidance",
        flags=("--live-guidance",),
        profile_key="live_guidance",
        what=(
            "Which version of the screenshot guidance the system message carries. Under --no-web, v3 and v4 "
            "carry v2's text."
        ),
        useful_when="v3 for web; v2 is what `--no-web` gets anyway",
        not_useful_when="v4: worse on the shared handoff (T-0211 C: 3/9 → 1/9)",
        evidence=(
            "T-0174 C web W1: v1 2/5 → v2 4/5; T-0191 B v3: navigation N1 5/5, N2 4/5; T-0211 C v4 FAIL "
            "(sourcing 0/8 both arms; worse on the shared handoff (3/9 → 1/9); typed 6/9 → 7/9)"
        ),
        cost="system-message tokens, one cold prefix",
        relaunch=None,
    ),
    SettingEntry(
        key="user_instructions",
        label="Your instructions to Gearvis",
        group="Guidance and prompt",
        kind="next_turn_cold",
        v1_changeable=True,
        values=(),
        default="",
        default_note="empty",
        flag="",
        flags=(),
        profile_key=None,
        what=(
            "Your own text, appended last to the system message under a fixed header. Safety rules are enforced "
            "by Gearvis's code, not by this text."
        ),
        useful_when="style or language preferences, domain hints",
        not_useful_when="long or conflicting text (untested against the A/Bs)",
        evidence="T-0213: unmeasured",
        cost="≤ 1000 code points in the cached prefix",
        relaunch=None,
    ),
    SettingEntry(
        key="thinking",
        label="Thinking",
        group="Guidance and prompt",
        kind="fixed",
        v1_changeable=False,
        values=(True,),
        default=True,
        default_note="on",
        flag="",
        flags=(),
        profile_key="diffusion.thinking",
        what="The model thinks before it answers. It is always on and is not offered as a switch.",
        useful_when="always on: every A/B ran with it on",
        not_useful_when="cannot be turned off: switching it per request would make every reply slow",
        evidence=(
            "T-0213 A2a-6: `<|think|>` is the first system token; toggling it misses the whole prefix cache, and "
            "every A/B ran with it on"
        ),
        cost="thinking time before each answer",
        relaunch=None,
    ),
    # -- Search and web --------------------------------------------------------------------------------------
    SettingEntry(
        key="search",
        label="Web search",
        group="Search and web",
        kind="relaunch",
        v1_changeable=False,
        values=SEARCH_CHOICES,
        default="ddgs",
        default_note="on in live (T-0125)",
        flag="--search",
        flags=("--search",),
        profile_key="search",
        what=(
            "Answers the model's google_search calls through the keyless search engines. In live, with search "
            "off the google_search tool is withdrawn (T-0215)."
        ),
        useful_when="questions that need the web",
        not_useful_when="private or offline work",
        evidence=(
            "T-0125: no quality A/B; cap 5 per question (T-0176); T-0220 search off: shared handoff used its "
            "browser 9/9, success 2/9"
        ),
        cost="queries leave the box",
        relaunch=f"{_SESSION} --search off",
    ),
    SettingEntry(
        key="browser_web",
        label="Real websites",
        group="Search and web",
        kind="relaunch",
        v1_changeable=False,
        values=_BOOL,
        default=True,
        default_note="web on",
        flag="--no-web",
        flags=("--no-web",),
        profile_key="browser_web",
        what=(
            "Whether the agent browser may open public https websites through the session's egress proxy, or "
            "stays on Gearvis's loopback start pages."
        ),
        useful_when="tasks on real websites",
        not_useful_when="the loopback demo pages (unreachable in web mode), offline work",
        evidence="T-0160 smokes",
        cost="pages can carry instructions the model follows (T-0138 S14: followed 4/20)",
        relaunch=f"{_SESSION} --no-web",
    ),
    # -- Speech ----------------------------------------------------------------------------------------------
    SettingEntry(
        key="filler_clips",
        label="Filler clips",
        group="Speech",
        kind="relaunch",
        v1_changeable=False,
        values=_BOOL,
        default=False,
        default_note="off (James 2026-10-28)",
        flag="--filler-clips",
        flags=("--filler-clips", "--no-filler-clips"),
        profile_key="filler_clips",
        what="Plays the short ack clip as a thinking turn starts, and the conversational interim filler.",
        useful_when="voice chats with long thinking",
        not_useful_when="typed use, demos",
        evidence="T-0182: off by James's ruling 2026-10-28; no A/B",
        cost="none",
        relaunch="diffusion_gearvis session --filler-clips",
    ),
    SettingEntry(
        key="honest_ending_clips",
        label="Honest-ending clips (out_of_steps, not_finished, lost_thread, at_checkout)",
        group="Speech",
        kind="fixed",
        v1_changeable=False,
        values=(True,),
        default=True,
        default_note="on",
        flag="",
        flags=(),
        profile_key=None,
        what="Fixed pre-rendered clips that say honestly why a turn ended. They are system audio, not model text.",
        useful_when="always on: they say why a turn stopped early",
        not_useful_when="cannot be turned off",
        evidence="T-0178/T-0179/T-0192",
        cost="system audio; never in history",
        relaunch=None,
    ),
    # -- Engine ----------------------------------------------------------------------------------------------
    SettingEntry(
        key="engine",
        label=(
            "Engine: canvas, steps, max_num_seqs, KV, max-model-len, --diffusion-config, STT/TTS backend, profile"
        ),
        group="Engine (relaunch only)",
        kind="relaunch",
        v1_changeable=False,
        values=(),
        default=None,
        default_note="per `services/versions.lock` and the profile",
        flag="--profile",
        flags=("--profile",),
        profile_key=None,
        what="The engine and speech services' own settings. They change only by restarting the services.",
        useful_when="a different profile or engine budget for your GPU",
        not_useful_when="mid-conversation: it restarts the services",
        evidence="T-0058: pinned in services/versions.lock; `.claude/rules/gpu-services.md`",
        cost="a restart of the services",
        relaunch="diffusion_gearvis down && diffusion_gearvis up --profile lean",
    ),
)

#: Every ``diffusion_gearvis session`` option and every ``Profile`` field that is not a catalog row, with the
#: reason (T-0213 A1-2). Options are named by their exact spelling, ``Profile`` fields by their name.
NOT_A_SETTING: Final[tuple[tuple[str, str], ...]] = (
    ("-h", "argparse's help"),
    ("--help", "argparse's help"),
    ("--data-dir", "where files go, not how Gearvis behaves"),
    ("--turns", "where turns.jsonl goes, not how Gearvis behaves"),
    ("--sink", "the audio device to play through, not output quality"),
    ("--audio-path", "a regime label for turns.jsonl, not a behaviour"),
    ("--silent", "the audio device choice (a null sink), not output quality"),
    ("--mic", "the capture device, not output quality"),
    ("--no-mic", "the capture device, not output quality"),
    ("--source", "the perception source (screen or camera), chosen at launch"),
    ("--no-log", "the session log's start state; the HUD checkbox already toggles it"),
    ("--log-dir", "where the session log goes, not how Gearvis behaves"),
    ("--dry-run", "prints the resolved graph and exits"),
    ("--tool-loop", "structural: live needs the loop; only dry-run can turn it off"),
    ("--no-tool-loop", "structural: live needs the loop; only dry-run can turn it off"),
    ("--serve-hud", "whether the HUD page is served, not how Gearvis behaves"),
    ("--no-serve-hud", "whether the HUD page is served, not how Gearvis behaves"),
    ("name", "the profile's name (see the engine row)"),
    ("stt_backend", "a service choice (see the engine row)"),
    ("tts_backend", "a service choice (see the engine row)"),
    ("talker", "a service choice (see the engine row)"),
    ("aec", "the audio path, chosen with the hardware (see the engine row)"),
    ("vram_budget_gib", "an engine budget (see the engine row)"),
    ("services", "loopback endpoints, not a behaviour"),
    ("diffusion", "engine settings (see the engine row; thinking has its own fixed row)"),
    ("tool_loop", "structural: live needs the loop; only dry-run can turn it off"),
    ("still_working_after_s", "when the still_working system clip plays, not model output"),
)


def by_key(key: str) -> SettingEntry | None:
    """The catalog entry named ``key``, or ``None``."""
    for entry in CATALOG:
        if entry.key == key:
            return entry
    return None
