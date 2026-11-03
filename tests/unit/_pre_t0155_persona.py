"""T-0155: the pre-T-0155 persona ("Astra"), and the one declared difference the old pins may discount.

T-0155 changes exactly the persona sentence of three prompts ("You are Astra, ..." -> "You are Gearvis, ...") and
nothing else. Tests that pinned the old text keep their meaning through this module, never by editing a fixture or a
historical sha:

- **Golden diffs** (``test_session_stale_playback_golden``, ``test_session_tools_off_golden``,
  ``test_session_loop_compaction``) normalise the *observed* side with :func:`normalise_system_messages` before
  comparing with the unchanged fixture: every system message must hold exactly one persona sentence of the live
  persona, and that one sentence -- nothing else -- is turned back into the pre-T-0155 sentence.
- **Frozen harnesses** (``scripts/gpu/t0098_native_prompt_ab.py`` with its T-0098 / T-0098c / T-0103 / T-0108a
  tables, ``scripts/gpu/t0107b_high_stakes_ab.py``) are run with the pre-T-0155 texts patched in
  (:func:`patch_pre_t0155`): ``session.DEFAULT_SYSTEM_PROMPT`` (which every harness binds into its own namespace when
  it is loaded by path), ``twopass.NATIVE_SYSTEM_PROMPT`` / ``_PROMPT_HEAD`` / ``SYSTEM_PROMPT`` (read as attributes
  at call time), and the ``DEFAULT_SYSTEM_PROMPT`` bound name of any already-loaded module passed in.

**Design of the normaliser (works before and after the implementation).** The persona the code under test carries
is read from ``session.DEFAULT_SYSTEM_PROMPT`` by :func:`live_persona`: ``"Astra"`` when it is the pre-T-0155 text
byte for byte, ``"Gearvis"`` when it is T-0155 criterion 1's text byte for byte, an ``AssertionError`` otherwise (a
prompt that changed beyond the persona is not this task's declared difference, so every golden goes red loudly).
:func:`to_pre_t0155` then requires, in one text, **exactly one** persona marker (``"You are Astra,"`` and
``"You are Gearvis,"`` counted together), that marker being the live persona's and starting one of the two known
persona sentences; it raises :class:`PersonaCountError` on zero, on two or more, and on a marker of the other name.
Today (Astra live) the replacement is the identity, so the rows stay green on main; after T-0155 (Gearvis live) it
reverses the one sentence. In the post-change world a system message with no Gearvis sentence (zero), with two, or
still saying Astra, raises. The ``persona=`` argument lets the adversarial rows exercise the Gearvis world today.

Regime: CPU, pure, no I/O.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import pytest

PRE_T0155_NAME = "Astra"
T0155_NAME = "Gearvis"
NAMES = (PRE_T0155_NAME, T0155_NAME)

#: ``session.DEFAULT_SYSTEM_PROMPT`` before T-0155, verbatim (main at 48a28e0, ``local_astra/conductor/session.py``).
PRE_T0155_DEFAULT_SYSTEM_PROMPT = (
    "You are Astra, a local assistant that can see the user's screen and hear their voice. "
    "Answer what you can actually see; say so when you cannot."
)
#: T-0155 criterion 1, verbatim.
T0155_DEFAULT_SYSTEM_PROMPT = (
    "You are Gearvis, a local assistant that can see the user's screen and hear their voice. "
    "Answer what you can actually see; say so when you cannot."
)


def session_sentence(name: str) -> str:
    """The persona sentence of ``session.DEFAULT_SYSTEM_PROMPT`` for ``name``."""
    return f"You are {name}, a local assistant that can see the user's screen and hear their voice."


def twopass_sentence(name: str) -> str:
    """The persona sentence of ``twopass._PROMPT_HEAD`` / ``twopass.NATIVE_SYSTEM_PROMPT`` for ``name`` (criterion 2)."""
    return f"You are {name}, a voice assistant that can see the user's screen and act on it."


def marker(name: str) -> str:
    return f"You are {name},"


#: The pre-T-0155 shas of the texts the frozen harnesses and earlier rows pin (computed at 48a28e0, the base T-0155
#: branched from; the A0 and A1 ones are T-0098's own literals, ``3605c012...`` and the prefix ``e0b54a58...``).
PRE_T0155_A0_SHA256 = "3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57"
PRE_T0155_A0_CHARS = 2744
PRE_T0155_NATIVE_SYSTEM_PROMPT_SHA256 = (
    "e0b54a588619bfa8e79e5925a8da9ccda3edfd0d276dbe8a3df023d59d516298"
)
PRE_T0155_PROMPT_HEAD_SHA256 = "f7fa7b631bccac86bcdcec3164bc3e7de44aad448f926581b1e97e6be5610f16"
PRE_T0155_SYSTEM_PROMPT_SHA256 = "fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3"


class PersonaCountError(ValueError):
    """A text that does not hold exactly one persona sentence of the expected name."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def live_persona() -> str:
    """``"Astra"`` or ``"Gearvis"``: which of the two declared texts ``session.DEFAULT_SYSTEM_PROMPT`` is."""
    from local_astra.conductor import session

    text = session.DEFAULT_SYSTEM_PROMPT
    if text == PRE_T0155_DEFAULT_SYSTEM_PROMPT:
        return PRE_T0155_NAME
    if text == T0155_DEFAULT_SYSTEM_PROMPT:
        return T0155_NAME
    raise AssertionError(
        "T-0155: session.DEFAULT_SYSTEM_PROMPT is neither the pre-T-0155 text nor criterion 1's text byte for byte, "
        f"so the persona sentence is not the only difference: {text!r}"
    )


def to_pre_t0155(text: str, *, persona: str | None = None) -> str:
    """``text`` with its one persona sentence turned back into the pre-T-0155 one (module docstring).

    Raises :class:`PersonaCountError` unless ``text`` holds exactly one ``"You are <name>,"`` marker over both names,
    of ``persona`` (default: :func:`live_persona`), starting one of the two persona sentences."""
    name = live_persona() if persona is None else persona
    if name not in NAMES:
        raise ValueError(f"unknown persona {name!r}; the declared ones are {NAMES}")
    counts = {n: text.count(marker(n)) for n in NAMES}
    total = sum(counts.values())
    if total != 1 or counts[name] != 1:
        raise PersonaCountError(
            f"T-0155: a system text must hold exactly one persona sentence, of {name!r}; it holds "
            f"{counts} ({total} in all): {text[:160]!r}"
        )
    for new, old in (
        (session_sentence(name), session_sentence(PRE_T0155_NAME)),
        (twopass_sentence(name), twopass_sentence(PRE_T0155_NAME)),
    ):
        if text.count(new) == 1:
            return text.replace(new, old, 1)
    raise PersonaCountError(
        f"T-0155: the {marker(name)!r} marker does not start a declared persona sentence: {text[:160]!r}"
    )


def to_t0155(text: str) -> str:
    """The forward direction, for the adversarial rows: a pre-T-0155 text with its one Astra sentence made Gearvis
    (same counting rule as :func:`to_pre_t0155`, applied to the Astra name)."""
    to_pre_t0155(text, persona=PRE_T0155_NAME)  # the counting rule, raising on 0 or >= 2
    for sentence in (session_sentence, twopass_sentence):
        old = sentence(PRE_T0155_NAME)
        if old in text:
            return text.replace(old, sentence(T0155_NAME), 1)
    raise AssertionError("unreachable: to_pre_t0155 accepted a text with no declared sentence")


def _system_text_parts(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    content = message.get("content")
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return None


def _map_system_messages(tree: Any, fn: Any) -> tuple[Any, int]:
    """A deep copy of ``tree`` with ``fn`` applied to the text of every ``role == "system"`` message (as a whole: the
    text parts joined count as one text; the part holding the change is rewritten), and how many there were."""
    out = copy.deepcopy(tree)
    seen = 0

    def walk(node: Any) -> None:
        nonlocal seen
        if isinstance(node, dict):
            if node.get("role") == "system" and "content" in node:
                seen += 1
                content = node["content"]
                if isinstance(content, str):
                    node["content"] = fn(content)
                else:
                    parts = _system_text_parts(node) or []
                    fn(
                        "".join(p["text"] for p in parts)
                    )  # the count is per message: raises on 0 or >= 2
                    holders = [p for p in parts if marker_in_any(p["text"])]
                    if len(holders) != 1:
                        raise PersonaCountError(
                            f"T-0155: the persona marker is split over the text parts: {len(holders)} hold it"
                        )
                    holders[0]["text"] = fn(holders[0]["text"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out, seen


def marker_in_any(text: str) -> bool:
    return any(marker(n) in text for n in NAMES)


def normalise_system_messages(tree: Any, *, persona: str | None = None) -> tuple[Any, int]:
    """Criterion 7: ``tree`` (a capture: prompts, request bodies, log records) with each system message's persona
    sentence reversed by :func:`to_pre_t0155` (raising on zero or more than one per message), and the number of system
    messages seen. Only ``role == "system"`` messages are touched: a user turn that says "Astra" or "Gearvis" is left
    as it is."""
    return _map_system_messages(tree, lambda t: to_pre_t0155(t, persona=persona))


def observed(tree: Any, *, persona: str | None = None) -> Any:
    """O-8: the *observed* side of a differential / base-equality row (a capture from the working tree, compared with
    a capture of an older commit's code that carried the pre-T-0155 persona), with the one declared difference
    reversed. :func:`normalise_system_messages` checks each ``role == "system"`` message holds exactly one persona
    sentence (raising on zero or two); then every string is visited once -- the messages' text and the copies of the system text these captures also hold *outside* a
    message object -- a raw ``system`` string, a dry-run's printed prompt, a serialised request body -- each string
    on its own: a string holding a persona marker is passed through :func:`to_pre_t0155` (exactly one marker of the
    live persona, starting a declared persona sentence, or it raises). A string with no marker is never touched, so a
    user turn saying "Astra" or "Gearvis" stays as it is. The base side is never normalised."""
    normalise_system_messages(tree, persona=persona)  # the per-message count; raises on zero or two

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return to_pre_t0155(node, persona=persona) if marker_in_any(node) else node
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(tree)


def map_system_messages(tree: Any, fn: Any) -> tuple[Any, int]:
    """Apply ``fn`` to every system message's text (the adversarial rows' mutants)."""
    return _map_system_messages(tree, fn)


#: The adversarial mutation of criterion 7: one character outside the persona sentence, in every system message the
#: session builds (``DEFAULT_SYSTEM_PROMPT``'s last sentence ends "say so when you cannot.").
MUTATION = ("say so when you cannot.", "say so when you cannot!")


def as_t0155(tree: Any) -> Any:
    """``tree`` as the post-T-0155 code would produce it, from either world: each system message's one persona
    sentence is Gearvis's (``to_t0155(to_pre_t0155(text))``)."""
    return map_system_messages(tree, lambda t: to_t0155(to_pre_t0155(t)))[0]


def one_more_character(tree: Any) -> Any:
    """``as_t0155(tree)`` with one more character changed in every system message (:data:`MUTATION`)."""

    def mutate(text: str) -> str:
        post = to_t0155(to_pre_t0155(text))
        old, new = MUTATION
        assert post.count(old) == 1, (
            f"premise: the mutation site is in the system text once: {post[:120]!r}"
        )
        return post.replace(old, new, 1)

    return map_system_messages(tree, mutate)[0]


def two_personas(tree: Any) -> Any:
    """``as_t0155(tree)`` with a second Gearvis persona sentence appended to every system message."""
    return map_system_messages(
        tree, lambda t: to_t0155(to_pre_t0155(t)) + " " + session_sentence(T0155_NAME)
    )[0]


class PreT0155:
    """The handle :func:`patch_pre_t0155` yields: ``undo()`` restores the live texts (the live-persona rows)."""

    def __init__(self, mp: pytest.MonkeyPatch) -> None:
        self._mp = mp
        self.active = True

    def bind(self, module: ModuleType) -> None:
        """Patch an already-loaded module's own ``DEFAULT_SYSTEM_PROMPT`` (a harness loaded before the patch)."""
        if hasattr(module, "DEFAULT_SYSTEM_PROMPT"):
            self._mp.setattr(module, "DEFAULT_SYSTEM_PROMPT", PRE_T0155_DEFAULT_SYSTEM_PROMPT)

    def undo(self) -> None:
        self._mp.undo()
        self.active = False


@contextmanager
def patch_pre_t0155(modules: Iterable[ModuleType] = ()) -> Iterator[PreT0155]:
    """Criterion 8: the pre-T-0155 texts in session, twopass and each given module's own bound names, for the frozen
    harnesses (a harness loaded by path inside the block binds the patched ``session.DEFAULT_SYSTEM_PROMPT``)."""
    from local_astra.conductor import session
    from local_astra.eval.cd import twopass

    mp = pytest.MonkeyPatch()
    handle = PreT0155(mp)
    try:
        live = session.DEFAULT_SYSTEM_PROMPT
        mp.setattr(session, "DEFAULT_SYSTEM_PROMPT", PRE_T0155_DEFAULT_SYSTEM_PROMPT)
        # session's other bound copy: ``SessionConfig(system_prompt=DEFAULT_SYSTEM_PROMPT)``, a keyword-only default
        # evaluated at import (the rows that compare a harness's A0 with a real Session's pass-1 prompt)
        kwdefaults = session.SessionConfig.__init__.__kwdefaults__
        assert kwdefaults is not None and kwdefaults.get("system_prompt") == live, (
            "premise: SessionConfig's system_prompt default is session.DEFAULT_SYSTEM_PROMPT"
        )
        mp.setitem(kwdefaults, "system_prompt", PRE_T0155_DEFAULT_SYSTEM_PROMPT)
        for name in ("NATIVE_SYSTEM_PROMPT", "_PROMPT_HEAD", "SYSTEM_PROMPT"):
            current = getattr(twopass, name)
            # the counting rule applies to the twopass texts too: exactly one persona sentence, the live persona's
            mp.setattr(twopass, name, to_pre_t0155(current, persona=_twopass_persona(current)))
        for module in modules:
            handle.bind(module)
        yield handle
    finally:
        mp.undo()


def _twopass_persona(text: str) -> str:
    """Which persona a twopass text carries (each must carry exactly one, of either name; the persona rows say which
    one it must be)."""
    found = [n for n in NAMES if marker(n) in text]
    if len(found) != 1:
        raise PersonaCountError(
            f"T-0155: a twopass text holds the persona markers {found}: {text[:120]!r}"
        )
    return found[0]


def pre_t0155_fixture() -> Iterator[PreT0155]:
    """Body of the per-module autouse fixture
    (``pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)``)."""
    with patch_pre_t0155() as handle:
        yield handle


# =====================================================================================================
# Criterion 8, second half: the frozen harness refuses today's (Gearvis) A0, before any socket
# =====================================================================================================


@contextmanager
def no_sockets() -> Iterator[list[str]]:
    """Every outbound socket attempt (loopback included) is recorded and refused: "before any socket"."""
    import socket

    attempts: list[str] = []
    mp = pytest.MonkeyPatch()

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{what} {args!r}")
            raise AssertionError(f"T-0155 criterion 8: a {what} before the A0 refusal")

        return fn

    mp.setattr(socket.socket, "connect", refuse("socket.connect"))
    mp.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    mp.setattr(socket, "create_connection", refuse("create_connection"))
    mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    try:
        yield attempts
    finally:
        mp.undo()


def run_main_against_live_persona(
    handle: PreT0155,
    load_harness: Any,
    t93_of: Any,
    argv: list[str],
    out: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[ModuleType, int, str, list[str]]:
    """Undo the pre-T-0155 patch, check the live persona is Gearvis, load the harness *after* that (so its own bound
    ``DEFAULT_SYSTEM_PROMPT`` is the live one), and run its ``main`` on an otherwise valid invocation: a loopback URL
    nothing listens on, a fresh ``--out-dir``, a clean tree (the harness's T-0093 ``_git`` answers "clean" -- the
    refusal under test is not the dirty-tree one, and the rows must not depend on the developer's tree), and the
    real pinned dry-run kit (``cli.load_tool_kit`` returns ``make_kit()``: repair on, the pinned tool set). Every
    socket is refused and recorded. Returns (harness, exit code, stdout + stderr, socket attempts)."""
    from local_astra import cli
    from tests.unit._tools_switch_support import make_kit

    handle.undo()
    assert live_persona() == T0155_NAME, (
        "T-0155 criteria 1/8: the live session.DEFAULT_SYSTEM_PROMPT is not the Gearvis text yet "
        f"(live persona: {live_persona()!r})"
    )
    h = load_harness()
    assert h.DEFAULT_SYSTEM_PROMPT == T0155_DEFAULT_SYSTEM_PROMPT, (
        "premise: the harness loaded after the undo binds the live (Gearvis) DEFAULT_SYSTEM_PROMPT"
    )
    t93 = t93_of(h)
    head = "0" * 40

    def clean_git(*args: str) -> str:
        if args[:2] == ("status", "--porcelain"):
            return ""
        if args[:2] == ("rev-parse", "HEAD"):
            return head + "\n"
        raise AssertionError(f"premise: the harness asked git for {args!r}")

    monkeypatch.setattr(t93, "_git", clean_git)
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile: (make_kit(), object()))
    capsys.readouterr()
    with no_sockets() as attempts:
        rc = h.main(["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(out), *argv])
    said = capsys.readouterr()
    return h, rc, said.out + said.err, list(attempts)
