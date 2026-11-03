"""T-0085 criterion 2: ``router.template_content`` mirrors the pinned chat template's rendering of assistant content.

The oracle is the pinned template itself (``tests/cd/oracle/gemma4_template.py``, sha ``36e3a42e...``, rendered as
``transformers`` renders it), never the implementation. Two independent readings of the template are used:

- **the whole turn:** render ``[user, assistant s]`` and take the model turn's content: the text between the
  ``<|turn>model\\n`` the template writes for the assistant (the render of ``[user]`` plus that header is a prefix) and
  the ``<turn|>\\n`` that closes the turn (the render's suffix). This is ``strip_thinking(s)`` exactly, as the
  template emits it;
- **the sentinel:** render ``s + SENTINEL`` and cut at the sentinel, as ``transformers``' ``continue_final_message``
  does (``CONTINUE_FINAL_MESSAGE_TAG ``). The sentinel sits inside the template's ``trim``, so the cut keeps ``s``'s
  trailing whitespace -- the cut, right-stripped, is the content. When ``s`` leaves a ``<|channel>`` open in its last
  piece the template drops the sentinel too (from ``<|channel>`` to the end of the piece); the row then only checks
  that it is gone, and the whole-turn reading decides.

Hand rows: click-01 (the T-0084e stop, verbatim), a leading newline, a nested open, the T4 continuations. The
adversarial row shows that a mirror deleting only ``<channel|>`` disagrees with the oracle on the hand rows (so the
equality rows would catch it) and fails them.

Regime: CPU; model rev 2ea8372; template sha 36e3a42e...; Hypothesis seed 85020, no example database, n per row.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from local_astra.toolcall import router
from tests.cd.oracle import gemma4_template as tpl
from tests.unit._toolcall_support import CLICK01_NATIVE_TEXT

OPEN, CLOSE = "<|channel>", "<channel|>"
MODEL = "<|turn>model\n"
TURN_END = "<turn|>\n"
#: ``transformers``' ``continue_final_message`` tag (``render_jinja_template``)
SENTINEL = "CONTINUE_FINAL_MESSAGE_TAG "
USER = {"role": "user", "content": "Do it."}


def template_content() -> Callable[[str], str]:
    fn = getattr(router, "template_content", None)
    if not callable(fn):
        raise AssertionError("T-0085 not implemented: router.template_content is missing")
    return fn


def rendered_content(s: str) -> str:
    """The assistant content of ``s`` as the pinned template renders it (the whole-turn reading)."""
    head = tpl.render([USER]) + MODEL
    full = tpl.render([USER, {"role": "assistant", "content": s}])
    assert full.startswith(head), (full[: len(head) + 40], head)
    assert full.endswith(TURN_END), full[-40:]
    return full[len(head) : len(full) - len(TURN_END)]


def sentinel_cut(s: str) -> str | None:
    """The render of ``s + SENTINEL`` cut at the sentinel (``None`` when the template dropped it)."""
    head = tpl.render([USER]) + MODEL
    full = tpl.render([USER, {"role": "assistant", "content": s + SENTINEL}])
    assert full.startswith(head)
    body = full[len(head) :]
    tag = SENTINEL.strip()
    if tag not in body:
        return None
    return body[: body.rindex(tag)]


def test_premise_the_oracle_is_strip_thinking() -> None:
    """The oracle's hand values (independent of the implementation): close deleted, open dropped to the end of its
    piece, the whole trimmed."""
    assert rendered_content("a<channel|>b") == "ab"
    assert rendered_content("a<|channel>b<channel|>c") == "ac"
    assert rendered_content("a<|channel>b") == "a"
    assert rendered_content("  a \n") == "a"
    assert sentinel_cut("a ") == "a "
    assert sentinel_cut("a<|channel>b") is None


HAND = [
    # (s, the content by hand)
    (
        CLICK01_NATIVE_TEXT,
        "I will click the Sign in button.<|tool_call>call:click_at{x:488,y:517}<tool_call|>",
    ),
    (
        "I will click the Sign in button.<channel|><|tool_call>",
        "I will click the Sign in button.<|tool_call>",
    ),
    (
        "I will click the Sign in button.<|tool_call>",
        "I will click the Sign in button.<|tool_call>",
    ),
    ("\nLet me look that up. <cc>", "Let me look that up. <cc>"),
    ("\n\t Hi.\n", "Hi."),
    ("a<|channel>b<|channel>c<channel|>d", "ad"),
    ("a<|channel>b<channel|>c<|channel>d<channel|>e", "ace"),
    ("x<|channel>y", "x"),
    ("<channel|>", ""),
    ("<|channel>", ""),
    ("", ""),
    ("<|chan<channel|>nel>", "<|channel>"),
    ("Sure.<|think|><|tool_call>", "Sure.<|think|><|tool_call>"),
    ("日本<channel|> 　ok　", "日本 　ok"),
]


@pytest.mark.parametrize(("s", "want"), HAND)
def test_hand_rows_match_the_template(s: str, want: str) -> None:
    """Criterion 2 hand rows: the oracle gives the hand value, and ``template_content`` gives the oracle's."""
    assert rendered_content(s) == want, "premise: the template renders the hand value"
    assert template_content()(s) == want


_FRAGMENTS = [
    OPEN,
    CLOSE,
    "<channel",
    "|>",
    "<|chan",
    "nel>",
    "<|tool_call>",
    "<tool_call|>",
    "<cc>",
    "thought\n",
    " ",
    "\n",
    "\t",
    "　",
    "a",
    "Hi.",
    "<",
    ">",
    "|",
    "日本",
]
_texts = st.lists(st.one_of(st.sampled_from(_FRAGMENTS), st.text(max_size=3)), max_size=14).map(
    "".join
)
_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@seed(85020)
@_SETTINGS
@given(s=_texts)
def test_template_content_is_the_templates_rendering(s: str) -> None:
    """Criterion 2, n = 300: ``template_content(s)`` equals the pinned template's content for ``s`` (whole-turn
    reading), and, where the sentinel survives, the ``transformers`` cut at the sentinel right-stripped."""
    want = rendered_content(s)
    assert template_content()(s) == want, (s, want)
    cut = sentinel_cut(s)
    if cut is not None:
        assert cut.rstrip() == want, (s, cut)


def _close_only(s: str) -> str:
    """The adversarial mirror: deletes ``<channel|>`` and trims, never drops an open thought."""
    return s.replace(CLOSE, "").strip()


def _global_drop(s: str) -> str:
    """A second wrong mirror: drops from the first ``<|channel>`` to the end of the whole text (not per piece)."""
    s = s.replace(CLOSE, "\x00")
    s = s.split(OPEN)[0]
    return s.replace("\x00", "").strip()


@pytest.mark.parametrize("mirror", [_close_only, _global_drop], ids=["close_only", "global_drop"])
def test_adversarial_a_wrong_mirror_disagrees_with_the_oracle(mirror: Callable[[str], str]) -> None:
    """Adversarial: a mirror that only deletes ``<channel|>`` (or drops past the piece) disagrees with the template
    on at least one hand row, so ``test_hand_rows_match_the_template`` fails it; ``template_content`` agrees with the
    oracle on exactly those rows."""
    wrong = [s for s, want in HAND if mirror(s) != want]
    assert wrong, f"no hand row discriminates {mirror.__name__}"
    for s in wrong:
        assert template_content()(s) == rendered_content(s), s


@pytest.mark.parametrize("tag", ["<cc>", "<|tool_call>"])
def test_the_routed_click01_continuation_survives_the_template(tag: str) -> None:
    """Criterion 2 against the oracle (not ``template_content``): the routed click-01 ``spoken_prefix + TAG`` renders
    unchanged, stripped; the 9aee66d prefix (with its stray close) did not."""
    r = router.route(CLICK01_NATIVE_TEXT)
    c = r.spoken_prefix + tag
    assert rendered_content(c) == c.strip(), c
    stale = "I will click the Sign in button.<channel|>" + tag
    assert rendered_content(stale) != stale.strip()
