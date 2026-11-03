"""T-0155: the assistant's name becomes "Gearvis" -- the persona sentence and nothing else (unit; CPU; pure).

Every expected text here is typed out from the task file (criteria 1, 2, 5) or hand-joined from the builder's own
named pieces (criterion 3); the pre-T-0155 sentences live in ``tests/unit/_pre_t0155_persona.py``. Rows that need
the implementation were ``pending(task="T-0155")`` until ``/implement T-0155`` started; the negative rows ("nothing else changes", "these stay") and
the helper's self-tests hold before and after it, and are unmarked.

``A0_T0155_SHA256`` (criterion 4) is generated, per the task's implementation note 3, from the pre-T-0155 A0 text
(``PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools).system_message().text()`` at 48a28e0, sha ``3605c012...``,
2744 characters) with ``'You are Astra,'`` replaced once by ``'You are Gearvis,'``::

    a0 = pb.PromptBuilder(session.DEFAULT_SYSTEM_PROMPT, tools=make_kit().tools).system_message().text()  # at 48a28e0
    assert sha256(a0) == "3605c012..." and len(a0) == 2744
    sha256(a0.replace("You are Astra,", "You are Gearvis,", 1))   # -> 869f8f96..., 2746 characters

``test_a0_t0155_pin_follows_from_its_generator`` re-runs that generator from the hand-joined pre-T-0155 text, and
``test_live_dry_run_text_hashes_to_a0_t0155`` recomputes the sha from the live builder.

Regime: CPU, pure, no measurement.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import session
from local_astra.conductor.session import SessionConfig
from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.eval.cd import twopass
from tests.unit import _pre_t0155_persona as P
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit._tools_switch_support import NATIVE_TOOL_INSTRUCTIONS_SHA256, make_kit

REPO = Path(__file__).resolve().parents[2]
TWOPASS = REPO / "local_astra" / "eval" / "cd" / "twopass.py"
TOOLS_V1 = REPO / "local_astra" / "toolcall" / "tools_v1.json"

#: Criterion 4: the live dry_run system text after T-0155 (generator in the module docstring; 2746 characters).
A0_T0155_SHA256 = "869f8f96cc4eaccb72a060d359a1456f26225a047b43233bc65256303bd3223e"
A0_T0155_CHARS = 2746

#: Criterion 1, verbatim.
CRITERION_1 = (
    "You are Gearvis, a local assistant that can see the user's screen and hear their voice. Answer what you can "
    "actually see; say so when you cannot."
)
#: Criterion 2, verbatim.
CRITERION_2_HEAD = (
    "You are Gearvis, a voice assistant that can see the user's screen and act on it."
)
#: Criterion 5, verbatim.
CRITERION_5 = "Gearvis's own sandboxed browser"

#: Criterion 3's unchanged pins (the task's prefixes; full values from the rows that pinned them first).
LIVE_TOOL_GUIDANCE_SHA256 = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
TOOLS_V1_SHA12 = "bbfc53ccdd80"
SPOKEN_STYLE_SHA256 = "7bb891a53a6250fd873ca915d1212be174e023c425ce8a7b2eec991d92083d6d"
#: ``session.CLIP_TEXT``, typed out (T-0013). T-0178 B6a re-pin (after f8e26760): the two ending clips, ruling H-T178-5.
#: T-0179 B-6 re-pin (after ebaf4248): ``lost_thread``, ruling H-T179-4 (typed from the task file, not the code)
#: T-0192 B1 re-pin (before implementation): ``at_checkout``, ruling H-T192-2 (typed from docs/tasks/T-0192.md B1)
CLIP_TEXT = {
    "ack": "let me look",
    "hmm": "hmm",
    "still_working": "still working on it",
    "out_of_steps": "I ran out of steps before I could finish.",
    "not_finished": "I could not finish that.",
    "lost_thread": "Sorry, I lost my train of thought. Could you ask again?",
    "at_checkout": "I've reached the checkout, so I'll stop here. Over to you to finish the purchase.",
}

MODES = ("off", "dry_run", "live")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def live_guidance_text(version: str | None = None) -> str:
    """The live guidance ``cli`` wires with no flag: ``LIVE_GUIDANCE[DEFAULT_LIVE_GUIDANCE]`` (T-0174 A4; v3 in web
    mode since T-0191 C1, v2 since T-0174 D1, v1 before), or the named version's text. It follows ``config``; the
    bytes are pinned by ``test_live_helper_is_v3`` (the literal ``LIVE_GUIDANCE["v3"]``) and tied to ``cli`` by
    ``test_live_helper_is_what_cli_builds`` (web mode)."""
    return pb.LIVE_GUIDANCE[version or config.DEFAULT_LIVE_GUIDANCE]


def pre_t0155_text(mode: str, version: str | None = None) -> str:
    """The pre-T-0155 system message of ``mode``, hand-joined from the builder's named pieces in docs/02 order
    (persona prompt, close-the-thought, spoken style, UI instructions; tools on: native tool instructions; live:
    the live guidance of ``version``, default ``cli``'s), with the pre-T-0155 persona typed out in the helper. The
    guidance version is independent of the persona (T-0174 D4 re-scope: before it the live join was v1 only)."""
    parts = [
        P.PRE_T0155_DEFAULT_SYSTEM_PROMPT,
        pb.CLOSE_THE_THOUGHT,
        pb.SPOKEN_STYLE,
        UI_INSTRUCTIONS,
    ]
    if mode in ("dry_run", "live"):
        parts.append(pb.NATIVE_TOOL_INSTRUCTIONS)
    if mode == "live":
        parts.append(live_guidance_text(version))
    return "\n".join(parts)


def built_text(mode: str, tmp_path: Path, version: str | None = None) -> str:
    """The system message the session builds in ``mode``: the session's own default prompt (``SessionConfig``'s),
    the kit's tools when tools are on, the live guidance in live only, ``cli``'s default version unless ``version``
    names one (as ``cli`` wires it, T-0131 C1, T-0174 A4/D1; ``test_live_helper_is_what_cli_builds`` checks it)."""
    prompt = SessionConfig(tmp_path).system_prompt
    tools = make_kit().tools if mode != "off" else None
    guidance = live_guidance_text(version) if mode == "live" else None
    return pb.PromptBuilder(prompt, tools=tools, tool_guidance=guidance).system_message().text()


def tools_help() -> str:
    """``astra session --help``'s ``--tools`` entry, whitespace collapsed (argparse wraps it)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as stop:
        cli.main(["session", "--help"])
    assert stop.value.code == 0, f"premise: --help exits 0, got {stop.value.code}"
    out = buf.getvalue()
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("--tools"))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("-")), len(lines)
    )
    return " ".join(" ".join(lines[start:end]).split())


# =====================================================================================================
# Criteria 1-2: the persona sentences
# =====================================================================================================


def test_default_system_prompt_is_gearvis() -> None:
    """Criterion 1: ``DEFAULT_SYSTEM_PROMPT`` is the task's text byte for byte, and the session's default is it."""
    assert session.DEFAULT_SYSTEM_PROMPT == CRITERION_1, (
        f"T-0155 criterion 1: DEFAULT_SYSTEM_PROMPT is {session.DEFAULT_SYSTEM_PROMPT!r}"
    )
    assert SessionConfig(Path("/nonexistent")).system_prompt == CRITERION_1, (
        "T-0155 criterion 1: SessionConfig's default system prompt is not criterion 1's text"
    )


def test_twopass_persona_is_gearvis() -> None:
    """Criterion 2: ``_PROMPT_HEAD`` and ``NATIVE_SYSTEM_PROMPT`` start with the Gearvis sentence and hold no
    ``Astra``; ``SYSTEM_PROMPT`` follows from ``_PROMPT_HEAD``. The source text is read too (the literal as written,
    not only the imported value)."""
    for name in ("_PROMPT_HEAD", "NATIVE_SYSTEM_PROMPT"):
        text = getattr(twopass, name)
        assert text.startswith(CRITERION_2_HEAD), (
            f"T-0155 criterion 2: twopass.{name} starts {text[:90]!r}"
        )
        assert "Astra" not in text, f"T-0155 criterion 2: twopass.{name} still says Astra"
    assert (
        twopass.SYSTEM_PROMPT.startswith(twopass._PROMPT_HEAD)
        and "Astra" not in twopass.SYSTEM_PROMPT
    ), "T-0155 criterion 2: SYSTEM_PROMPT follows from _PROMPT_HEAD"
    source = TWOPASS.read_text(encoding="utf-8")
    assert "You are Astra" not in source, (
        "T-0155 criterion 2: twopass.py's source still says 'You are Astra'"
    )
    assert source.count("You are Gearvis, a voice assistant") == 2, (
        "T-0155 criterion 2: twopass.py's source holds the Gearvis sentence twice (_PROMPT_HEAD, NATIVE_SYSTEM_PROMPT)"
    )


def test_twopass_texts_differ_from_pre_t0155_only_in_the_persona() -> None:
    """Criterion 2 (negative; holds before and after): with the one persona sentence reversed, the three twopass
    texts hash to their pre-T-0155 values (``NATIVE_SYSTEM_PROMPT`` is T-0098's ``e0b54a58...``)."""
    persona = P.live_persona()
    for name, want in (
        ("NATIVE_SYSTEM_PROMPT", P.PRE_T0155_NATIVE_SYSTEM_PROMPT_SHA256),
        ("_PROMPT_HEAD", P.PRE_T0155_PROMPT_HEAD_SHA256),
        ("SYSTEM_PROMPT", P.PRE_T0155_SYSTEM_PROMPT_SHA256),
    ):
        got = sha256(P.to_pre_t0155(getattr(twopass, name), persona=persona))
        assert got == want, (
            f"T-0155 criterion 2: twopass.{name} changed beyond the persona sentence ({got[:12]})"
        )
    assert P.PRE_T0155_NATIVE_SYSTEM_PROMPT_SHA256.startswith("e0b54a58")


def test_default_prompt_differs_from_pre_t0155_only_in_the_persona() -> None:
    """Criterion 1 (negative; holds before and after): the live prompt with its persona reversed is the pre-T-0155
    text byte for byte."""
    assert P.to_pre_t0155(session.DEFAULT_SYSTEM_PROMPT) == P.PRE_T0155_DEFAULT_SYSTEM_PROMPT


# =====================================================================================================
# Criteria 3-4: the built system message, per tools mode
# =====================================================================================================


def test_pre_t0155_hand_joins_are_the_pinned_texts() -> None:
    """Premise of criterion 3: the hand join for dry_run is T-0098's A0 (``3605c012...``, 2744 characters), and each
    mode's join holds exactly one persona sentence, Astra's."""
    a0 = pre_t0155_text("dry_run")
    assert sha256(a0) == P.PRE_T0155_A0_SHA256 and len(a0) == P.PRE_T0155_A0_CHARS
    for mode in MODES:
        text = pre_t0155_text(mode)
        assert text.count("Astra") == 1 and "Gearvis" not in text, mode


@pytest.mark.parametrize("mode", MODES)
def test_system_message_names_gearvis_once(mode: str, tmp_path: Path) -> None:
    """Criterion 3: the built system message holds ``Gearvis`` exactly once and ``Astra`` zero times, and replacing
    that single ``You are Gearvis,`` with ``You are Astra,`` gives the pre-T-0155 text byte for byte (dry_run:
    ``3605c012...``, 2744 characters; off and live hand-joined from the same constants)."""
    text = built_text(mode, tmp_path)
    assert text.count("Gearvis") == 1, (
        f"T-0155 criterion 3 ({mode}): 'Gearvis' {text.count('Gearvis')} times"
    )
    assert text.count("Astra") == 0, (
        f"T-0155 criterion 3 ({mode}): 'Astra' {text.count('Astra')} times"
    )
    assert text.count("You are Gearvis,") == 1
    back = text.replace("You are Gearvis,", "You are Astra,", 1)
    assert back == pre_t0155_text(mode), (
        f"T-0155 criterion 3 ({mode}): more than the persona changed"
    )
    if mode == "dry_run":
        assert sha256(back) == P.PRE_T0155_A0_SHA256 and len(back) == P.PRE_T0155_A0_CHARS


@pytest.mark.parametrize("mode", MODES)
def test_system_message_differs_from_pre_t0155_only_in_the_persona(
    mode: str, tmp_path: Path
) -> None:
    """Criterion 3 (negative; holds before and after): the built text with its one persona sentence reversed by the
    normaliser is the hand-joined pre-T-0155 text."""
    assert P.to_pre_t0155(built_text(mode, tmp_path)) == pre_t0155_text(mode)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_live_persona_only_per_guidance_version(version: str, tmp_path: Path) -> None:
    """Criterion 3 per live guidance version (T-0174 D4 re-scope; holds before and after the flip): with either
    version the live text names Gearvis once, and reversing that one sentence gives the hand join with the same
    version's guidance."""
    text = built_text("live", tmp_path, version)
    assert text.count("Gearvis") == 1 and text.count("Astra") == 0, version
    assert text.endswith("\n" + pb.LIVE_GUIDANCE[version])
    assert P.to_pre_t0155(text) == pre_t0155_text("live", version)


def test_live_helper_is_what_cli_builds(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    """The helper's claim "as ``cli`` wires it" (T-0174 D4): ``built_text("live")`` equals the system message of
    the live session ``cli`` builds with no flag (T-0131 C1's rig). Holds before and after the flip."""
    from tests.unit.test_cli_t0131 import build

    calls = build(tmp_path / "cli", capsys, monkeypatch, "live")
    # T-0202 D3 re-pin: cli's default live system message admitted through admit_system (the one H-T202-4 wait
    # sentence removed, required present once when the default is on); the helper stays the v3 text the persona rows
    # check, so it is not taught the sentence
    got = t0202.admit_system(
        calls.graphs[-1].session.prompts.system_message().text(), "persona live helper"
    )
    assert built_text("live", tmp_path) == got, (
        f"T-0174 D4: the persona helper's live text is not what cli builds; cli's ends {got[-160:]!r}"
    )


def test_live_helper_is_v3(tmp_path: Path) -> None:
    """T-0191 C3 re-scope of T-0174 D1/D4's ``test_live_helper_is_v2``: the live system text the persona rows check
    ends with ``LIVE_GUIDANCE["v3"]`` (the web-mode default since T-0191 C1), once, and holds no v1 guidance
    (``test_live_helper_is_what_cli_builds`` ties the helper to ``cli``). The expected text is the mapping's literal
    v3 entry, not ``config``'s default."""
    text = built_text("live", tmp_path)
    v3 = pb.LIVE_GUIDANCE["v3"]
    assert text.endswith("\n" + v3) and text.count(v3) == 1 and pb.LIVE_TOOL_GUIDANCE not in text, (
        f"T-0191 C1: the default live text ends {text[-160:]!r}"
    )


def test_a0_t0155_pin_follows_from_its_generator() -> None:
    """Criterion 4: the pin is note 3's generator over the pre-T-0155 A0 text (checked to be ``3605c012...`` first)."""
    a0 = pre_t0155_text("dry_run")
    assert sha256(a0) == P.PRE_T0155_A0_SHA256, "premise: the generator's input is T-0098's A0"
    generated = a0.replace("You are Astra,", "You are Gearvis,", 1)
    assert sha256(generated) == A0_T0155_SHA256 and len(generated) == A0_T0155_CHARS


def test_live_dry_run_text_hashes_to_a0_t0155(tmp_path: Path) -> None:
    """Criterion 4: the live builder's dry_run system text (``PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools)``,
    and the session's default through ``SessionConfig``) hashes to ``A0_T0155_SHA256``."""
    direct = (
        pb.PromptBuilder(session.DEFAULT_SYSTEM_PROMPT, tools=make_kit().tools)
        .system_message()
        .text()
    )
    assert sha256(direct) == A0_T0155_SHA256, (
        f"T-0155 criterion 4: the live A0 is {sha256(direct)[:12]}..."
    )
    assert len(direct) == A0_T0155_CHARS
    assert built_text("dry_run", tmp_path) == direct


# T-0178 B6a: re-pinned after the implementation commit f8e26760 (CLIP_TEXT gains the two ending clips)
# T-0192 B1: re-pinned before implementation (CLIP_TEXT gains ``at_checkout``, its exact text typed from the task)
def test_shared_pins_unchanged() -> None:
    """Criterion 3 (negative): ``LIVE_TOOL_GUIDANCE`` ``768a973b...``, tools_v1 ``bbfc53ccdd80``, ``SPOKEN_STYLE``
    ``7bb891a5...``, ``NATIVE_TOOL_INSTRUCTIONS`` (its pin, and its parity with twopass's paragraphs 2-3, H-T84-0),
    ``CLIP_TEXT``."""
    assert sha256(pb.LIVE_TOOL_GUIDANCE) == LIVE_TOOL_GUIDANCE_SHA256
    assert hashlib.sha256(TOOLS_V1.read_bytes()).hexdigest()[:12] == TOOLS_V1_SHA12
    assert sha256(pb.SPOKEN_STYLE) == SPOKEN_STYLE_SHA256
    assert sha256(pb.NATIVE_TOOL_INSTRUCTIONS) == NATIVE_TOOL_INSTRUCTIONS_SHA256
    paragraphs = twopass.NATIVE_SYSTEM_PROMPT.split("\n\n")
    assert len(paragraphs) == 3 and "\n\n".join(paragraphs[1:3]) == pb.NATIVE_TOOL_INSTRUCTIONS, (
        "H-T84-0: NATIVE_TOOL_INSTRUCTIONS is twopass's paragraphs 2-3"
    )
    assert session.CLIP_TEXT == CLIP_TEXT


# =====================================================================================================
# Criterion 5: what users read about the browser
# =====================================================================================================


def test_tools_help_says_gearvis() -> None:
    """Criterion 5: ``astra session --help``'s ``--tools`` entry says "Gearvis's own sandboxed browser", no Astra."""
    text = tools_help()
    assert CRITERION_5 in text, f"T-0155 criterion 5: the --tools help reads {text!r}"
    assert "Astra" not in text, f"T-0155 criterion 5: the --tools help still names Astra: {text!r}"


@pytest.mark.parametrize("value", ["on", "", 3])
def test_tools_refusal_says_gearvis(value: Any) -> None:
    """Criterion 5: ``config.tools_refusal(value)`` says "Gearvis's own sandboxed browser" and holds no Astra."""
    text = " ".join(config.tools_refusal(value).split())
    assert CRITERION_5 in text, f"T-0155 criterion 5: tools_refusal reads {text!r}"
    assert "Astra" not in text, f"T-0155 criterion 5: tools_refusal still names Astra: {text!r}"
    assert repr(value) in text, "the refusal still names the refused value"


# =====================================================================================================
# Criterion 6 (code half): what T-0156 renames, not this task
# =====================================================================================================


@pytest.mark.parametrize(
    ("argv0", "want"),
    [
        pytest.param("/opt/venv/bin/pytest", "usage: diffusion_gearvis session", id="fallback"),
        pytest.param("/opt/venv/bin/astra", "usage: astra session", id="astra"),
    ],
)
def test_out_of_scope_names_stay(argv0: str, want: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 6 (negative): the argparse prog, the ``local_astra`` import path, ``astra.*`` message types and the
    ``ASTRA_*`` environment reads are not T-0155's. The prog follows T-0156 R (ruling O-M): the invoked command's
    name, ``diffusion_gearvis`` under the fallback (any other ``sys.argv[0]``), ``astra`` when invoked as ``astra``.
    (The zero-diff half over ``hud/**`` etc. is the verifier's git check: a scope row needs the merge commit's two
    ends, which do not exist yet.)"""
    import contextlib
    import io
    import sys

    import local_astra
    from local_astra.contracts.protocol import ASTRA_MESSAGE_TYPES

    assert Path(local_astra.__file__).resolve().parent.name == "local_astra"
    monkeypatch.setattr(sys, "argv", [argv0, "session", "--help"])
    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
        cli.main(["session", "--help"])
    assert buf.getvalue().startswith(want), buf.getvalue()[:80]
    types = [str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES]
    assert types and all(t.startswith("astra.") for t in types), types
    audio = (REPO / "local_astra" / "conductor" / "audio_device.py").read_text(encoding="utf-8")
    for env in ("ASTRA_AUDIO_SOURCE", "ASTRA_AUDIO_INPUT_DEVICE", "ASTRA_AUDIO_DEVICE"):
        assert f'os.environ.get("{env}")' in audio, f"criterion 6: the {env} read moved"
    assert "ASTRA_AUDIO_SOURCE" in (REPO / "local_astra" / "cli.py").read_text(encoding="utf-8")


# =====================================================================================================
# Adversarial: a persona change is one sentence, never a global replace
# =====================================================================================================


USER_SAYS = "Is your name Astra or Gearvis? My friend Astra says You are Astra, a local assistant."


def test_user_turn_naming_either_persona_passes_through_unchanged(tmp_path: Path) -> None:
    """Adversarial (the task's failure mode is a careless global rewrite): a user turn and a history answer that say
    "Astra" and "Gearvis" reach the prompt byte for byte; only the system message carries the persona, once."""
    from local_astra.contracts.turn import HistoryTurn

    builder = pb.PromptBuilder(SessionConfig(tmp_path).system_prompt, tools=make_kit().tools)
    history = [
        HistoryTurn(
            user_text="who are you, Gearvis?",
            assistant_spoken="I am Astra's successor; call me Astra or Gearvis.",
        )
    ]
    prompt = builder.full(builder.prefix(history), USER_SAYS, None, [])
    texts = [m.text() for m in prompt.messages]
    assert texts[0] == built_text("dry_run", tmp_path), "the system message is the builder's own"
    assert any(USER_SAYS in t for t in texts[1:]), "the user turn reached the prompt unchanged"
    assert any("who are you, Gearvis?" in t for t in texts[1:])
    assert any("I am Astra's successor; call me Astra or Gearvis." in t for t in texts[1:])
    # and the normaliser touches the system message only
    dumped = prompt.model_dump(mode="json")
    normalised, seen = P.normalise_system_messages(dumped)
    assert seen == 1
    assert normalised["messages"][1:] == dumped["messages"][1:], (
        "the normaliser rewrote a non-system message"
    )


# =====================================================================================================
# The helper's own rows (zero / one / two, either world)
# =====================================================================================================


GEARVIS_TEXT = P.T0155_DEFAULT_SYSTEM_PROMPT + "\nThink step by step."
ASTRA_TEXT = P.PRE_T0155_DEFAULT_SYSTEM_PROMPT + "\nThink step by step."


def test_to_pre_t0155_reverses_exactly_one_sentence() -> None:
    assert P.to_pre_t0155(GEARVIS_TEXT, persona="Gearvis") == ASTRA_TEXT
    assert P.to_pre_t0155(ASTRA_TEXT, persona="Astra") == ASTRA_TEXT
    twopass_like = (
        "You are Gearvis, a voice assistant that can see the user's screen and act on it.\n\nRest."
    )
    assert P.to_pre_t0155(twopass_like, persona="Gearvis") == (
        "You are Astra, a voice assistant that can see the user's screen and act on it.\n\nRest."
    )
    assert P.to_t0155(ASTRA_TEXT) == GEARVIS_TEXT


@pytest.mark.parametrize(
    ("text", "persona"),
    [
        ("Think step by step.", "Gearvis"),  # zero
        (GEARVIS_TEXT + "\n" + P.T0155_DEFAULT_SYSTEM_PROMPT, "Gearvis"),  # two
        (ASTRA_TEXT, "Gearvis"),  # the other name (post-change world, implementation regressed)
        (GEARVIS_TEXT, "Astra"),  # the other name (pre-change world)
        (GEARVIS_TEXT + " " + P.PRE_T0155_DEFAULT_SYSTEM_PROMPT, "Gearvis"),  # one of each
        (
            "You are Gearvis, a cloud assistant.",
            "Gearvis",
        ),  # a marker that starts no declared sentence
        (
            "You are Gearvix, a local assistant that can see the user's screen and hear their voice.",
            "Gearvis",
        ),
    ],
    ids=[
        "zero",
        "two",
        "astra-in-gearvis-world",
        "gearvis-in-astra-world",
        "one-of-each",
        "undeclared",
        "typo",
    ],
)
def test_to_pre_t0155_raises_unless_exactly_one(text: str, persona: str) -> None:
    with pytest.raises(P.PersonaCountError):
        P.to_pre_t0155(text, persona=persona)


def test_normaliser_counts_per_system_message() -> None:
    """Criterion 7: one count per system message (two messages each with one sentence normalise; a message with two
    raises even when another has none), and only ``role == "system"`` is read."""

    def msg(role: str, text: str) -> dict[str, Any]:
        return {"role": role, "content": [{"type": "text", "text": text}]}

    tree = {
        "a": [msg("system", GEARVIS_TEXT), msg("user", "Gearvis?")],
        "b": {"m": [msg("system", GEARVIS_TEXT)]},
    }
    out, seen = P.normalise_system_messages(tree, persona="Gearvis")
    assert seen == 2
    assert (
        out["a"][0]["content"][0]["text"] == ASTRA_TEXT
        and out["b"]["m"][0]["content"][0]["text"] == ASTRA_TEXT
    )
    assert out["a"][1]["content"][0]["text"] == "Gearvis?", "a user message is never normalised"
    assert tree["a"][0]["content"][0]["text"] == GEARVIS_TEXT, "the input is not mutated"
    doubled = {"x": [msg("system", GEARVIS_TEXT + GEARVIS_TEXT), msg("system", "none")]}
    with pytest.raises(P.PersonaCountError):
        P.normalise_system_messages(doubled, persona="Gearvis")
    with pytest.raises(P.PersonaCountError):
        P.normalise_system_messages({"x": [msg("system", "no persona at all")]}, persona="Gearvis")
    as_string = {"role": "system", "content": GEARVIS_TEXT}
    assert P.normalise_system_messages(as_string, persona="Gearvis") == (
        {"role": "system", "content": ASTRA_TEXT},
        1,
    )


def test_normaliser_sees_a_real_prompt_dump() -> None:
    """The normaliser reads the shape the goldens hold: ``Prompt.model_dump`` (content parts with ``text``)."""
    prompt = Prompt(
        messages=(Message(role="system", content=(TextPart(text=GEARVIS_TEXT),)),), prefix_len=1
    )
    out, seen = P.normalise_system_messages(prompt.model_dump(mode="json"), persona="Gearvis")
    assert seen == 1 and out["messages"][0]["content"][0]["text"] == ASTRA_TEXT


def test_live_persona_is_one_of_the_two_declared_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``live_persona`` names the world; a prompt that changed beyond the persona is refused, not guessed."""
    assert P.live_persona() in P.NAMES
    monkeypatch.setattr(session, "DEFAULT_SYSTEM_PROMPT", P.T0155_DEFAULT_SYSTEM_PROMPT)
    assert P.live_persona() == "Gearvis"
    monkeypatch.setattr(session, "DEFAULT_SYSTEM_PROMPT", P.PRE_T0155_DEFAULT_SYSTEM_PROMPT)
    assert P.live_persona() == "Astra"
    monkeypatch.setattr(session, "DEFAULT_SYSTEM_PROMPT", P.T0155_DEFAULT_SYSTEM_PROMPT + " ")
    with pytest.raises(AssertionError, match="neither the pre-T-0155 text nor criterion 1"):
        P.live_persona()


def test_patch_pre_t0155_patches_and_restores() -> None:
    """Criterion 8's fixture: inside, session and twopass carry the pre-T-0155 texts (by sha); after, the live ones."""
    import types

    live = (
        session.DEFAULT_SYSTEM_PROMPT,
        twopass.NATIVE_SYSTEM_PROMPT,
        twopass._PROMPT_HEAD,
        twopass.SYSTEM_PROMPT,
    )
    loaded = types.ModuleType("_already_loaded_harness")
    loaded.DEFAULT_SYSTEM_PROMPT = session.DEFAULT_SYSTEM_PROMPT  # type: ignore[attr-defined]
    with P.patch_pre_t0155([loaded]) as handle:
        assert session.DEFAULT_SYSTEM_PROMPT == P.PRE_T0155_DEFAULT_SYSTEM_PROMPT
        assert loaded.DEFAULT_SYSTEM_PROMPT == P.PRE_T0155_DEFAULT_SYSTEM_PROMPT  # type: ignore[attr-defined]
        assert sha256(twopass.NATIVE_SYSTEM_PROMPT) == P.PRE_T0155_NATIVE_SYSTEM_PROMPT_SHA256
        assert sha256(twopass._PROMPT_HEAD) == P.PRE_T0155_PROMPT_HEAD_SHA256
        assert sha256(twopass.SYSTEM_PROMPT) == P.PRE_T0155_SYSTEM_PROMPT_SHA256
        handle.undo()
        assert live[0] == session.DEFAULT_SYSTEM_PROMPT and not handle.active
    assert live == (
        session.DEFAULT_SYSTEM_PROMPT,
        twopass.NATIVE_SYSTEM_PROMPT,
        twopass._PROMPT_HEAD,
        twopass.SYSTEM_PROMPT,
    )


def _capture_like(system_text: str) -> dict[str, Any]:
    """A hand-built capture shaped like the O-8 modules' observed side: a request with a system message (text parts)
    and two user turns naming the personas, a turn line, a raw ``system`` copy and a dry-run's printed prompt."""
    return {
        "requests": [
            {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": system_text}]},
                    {"role": "user", "content": [{"type": "text", "text": "Gearvis?"}]},
                    {"role": "user", "content": "Astra?"},
                ]
            }
        ],
        "records": [{"user_text": "Is Gearvis or Astra listening?", "assistant_spoken": "Astra?"}],
        "system": system_text,
        "dry": {"out": "tools=off\n" + system_text + "\n"},
    }


def test_observed_is_counted_never_global() -> None:
    """T-0155 O-8/O-8b (the reviewer's low finding: an uncounted global replace passed every row): ``observed``
    reverses exactly the one persona sentence of each string that carries one, and nothing else. User turns and
    turn lines saying "Gearvis?" / "Astra?" are untouched; the expected tree is typed out here, not derived by the
    helper. A stray string holding two persona sentences raises, as does a system message with none."""
    got = P.observed(_capture_like(GEARVIS_TEXT), persona="Gearvis")
    want = _capture_like(
        ASTRA_TEXT
    )  # the persona sentence reversed in all three copies of the system text
    assert want["requests"][0]["messages"][1]["content"][0]["text"] == "Gearvis?", "premise"
    assert got == want, f"O-8: observed() changed more (or less) than the persona sentence: {got!r}"
    # nothing to reverse in the pre-T-0155 world: the identity, user turns included
    assert P.observed(_capture_like(ASTRA_TEXT), persona="Astra") == _capture_like(ASTRA_TEXT)

    two = _capture_like(GEARVIS_TEXT)
    two["dry"]["out"] = (
        GEARVIS_TEXT + "\n" + GEARVIS_TEXT
    )  # a stray string with two persona markers
    with pytest.raises(P.PersonaCountError):
        P.observed(two, persona="Gearvis")
    mixed = _capture_like(GEARVIS_TEXT)
    mixed["system"] = GEARVIS_TEXT + " " + ASTRA_TEXT  # one of each name: two markers in all
    with pytest.raises(P.PersonaCountError):
        P.observed(mixed, persona="Gearvis")
    none = _capture_like(GEARVIS_TEXT)
    none["requests"][0]["messages"][0]["content"][0]["text"] = (
        "Think step by step."  # a system message, no persona
    )
    with pytest.raises(P.PersonaCountError):
        P.observed(none, persona="Gearvis")
