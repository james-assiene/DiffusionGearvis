"""The A2UI instruction block of the cached system prefix (docs/02 §2.2). Tasks T-0044, T-0055.

The Conductor has had a shown channel since T-0010 — an ``<a2ui>…</a2ui>`` fence that
``llm_client.split_segments`` routes to the ``ui`` segment, a validate-and-repair ``UiGate``, a
HUD renderer — and the model was never told it exists. It narrated that it was showing
something, emitted prose (a live probe on 2026-09-15 got an invented ``<ui>…</ui>`` tag with
markdown inside), no ``ui`` segment was ever produced and every turn line carried
``ui_valid: null``. This module is the text that fixes that, and it is joined into
``PromptBuilder.system_message()`` — the cached prefix — so it is prefilled once per session.

T-0055 then had to say *where* the surface goes. Told only that it had a second channel, the
model twice built a complete, root-bearing surface **inside its reasoning channel** — which is
discarded by design, and which the chat template's own ``strip_thinking`` macro deletes when it
re-renders history — and then said aloud that it had created an interface, while the human
looked at an empty panel. So the surface is now framed as part of the *one reply*, never as a
channel callable from wherever the model is standing, and the placement rule sits in the same
bullet as the rule the model followed to the letter inside its thought.

Three things about *how* it is written, each of which a test pins:

* **literals, never generation.** The block is not assembled from ``A2UIValidator`` at import
  time and not from a ``frozenset``: ``prompt_builder`` is pure and deterministic, the validator
  reads vendored schema files, and a generated block would silently change the cached system
  prefix — and so invalidate every prefix-cache block — whenever a vendored schema changed.
  ``ADVERTISED_COMPONENTS`` is built *from* the ordered tuple the text interpolates, never the
  other way round, because a set iterates in string-hash order and would give every session a
  different prefix (``tests/unit/test_prompt_ui_block.py`` runs two ``PYTHONHASHSEED``s).
* **the fence and the component names are checked against the code, not against this file.**
  The two markers in the block *are* ``llm_client.UI_OPEN``/``UI_CLOSE``, interpolated and never
  retyped (T-0055 criterion 5): a hand-typed copy would survive a rename of the client's fence
  and send every surface into the ``answer`` channel, which since T-0052 is spoken aloud.
  ``tests/unit/test_ui_instructions.py`` imports ``llm_client.UI_OPEN``/``UI_CLOSE`` and
  ``A2UIValidator().component_kinds`` as its oracles, so a renamed fence or a changed catalog
  turns red here instead of drifting; the same test pushes :data:`UI_EXAMPLE` through the real
  validator and ``tests/contract/test_prompt_example_reaches_a_surface.py`` through the real
  ``UiGate``. The example is therefore payload the gate is known to accept with zero repairs.
* **it costs prompt budget on every turn.** It lives inside the 14 336-token prompt budget
  (never in the answer headroom), is bounded at 2 000 code points by the fast suite, and its
  token cost is the engine's number (T-0044 criterion 5b, ≤ 600, measured under ``/launch``).

Catalog: the A2UI v0.9.1 **basic** catalog, minus ``AudioPlayer``, ``Image`` and ``Video``
(they fetch media by URL, and the HUD asserts zero non-loopback requests; the HUD's own renderer,
T-0164, renders all three as nothing). The ``edu`` catalog — ``Table``, ``Chart``, ``Math``, ``Flashcard``, … — is T-0010 C4
and does not exist, so advertising one of those would make every surface using it get pruned:
the same ``ui_valid``-shaped failure with a different cause.
"""

from __future__ import annotations

from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN

#: The 15 component kinds the block offers, in the order the text lists them: layout and text
#: first, interactive last. A tuple, not a set: this order is part of the cached prefix bytes.
ADVERTISED_ORDER: tuple[str, ...] = (
    "Text",
    "Column",
    "Row",
    "List",
    "Card",
    "Divider",
    "Icon",
    "Tabs",
    "Modal",
    "Button",
    "CheckBox",
    "ChoicePicker",
    "TextField",
    "Slider",
    "DateTimeInput",
)

#: What the tests compare against ``A2UIValidator().component_kinds`` (criterion 3).
ADVERTISED_COMPONENTS: frozenset[str] = frozenset(ADVERTISED_ORDER)

#: The one example the block shows: a two-message surface (``createSurface`` then one
#: ``updateComponents``) built from ``Column``/``Row``/``Text`` only, with ``root`` present and
#: every reference resolvable in the same payload — a forward reference here would teach exactly
#: the shape that costs a resample (T-0010 A9). The ``catalogId`` is the identifier the gate's
#: validator knows and the HUD renderer aliases; it is compared as a string and never fetched,
#: and it is the only URL-shaped thing allowed anywhere in the block. Nothing in the payload is
#: a fetchable URL (CLAUDE.md: loopback only).
#:
#: Whitespace is deliberate: no indentation, one JSON object per line, so the whole fenced
#: example is 483 characters and cannot straddle a 256-token canvas (llm_client B7/M7).
UI_EXAMPLE: str = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":'
    '"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
    '{"id":"root","component":"Column","children":["h","r1"]},'
    '{"id":"h","component":"Text","text":"Primes and squares","variant":"h3"},'
    '{"id":"r1","component":"Row","children":["a","b"]},'
    '{"id":"a","component":"Text","text":"2"},'
    '{"id":"b","component":"Text","text":"4"}]}}'
)

#: The block itself, appended to the system message after ``SPOKEN_STYLE``. Read it as English:
#: the model does. The rules it states, and why each one is here:
#:
#: * **JSON only inside the fence** — prose or markdown inside it fails the gate, and the live
#:   probe showed the model reaching for markdown the moment it was not told otherwise;
#: * **one surface per turn** — the gate keeps the first surface and drops the others (A3/A7);
#: * **never name the fence aloud** — the two channels are separate and TTS would read the tag;
#: * **the whole surface, written once, in the reply** — T-0055. The two halves are one bullet on
#:   purpose: on 2026-09-16 the model twice obeyed "open and close the fence in the same message"
#:   *inside its thought*, because nothing said where the fence goes and the nearest rule about
#:   placement (``prompt_builder.CLOSE_THE_THOUGHT``) is about the spoken answer. The model
#:   follows the most local rule about the fence, so the placement rule lives in that rule. The
#:   erasure clause is not a threat: the pinned chat template's ``strip_thinking`` macro deletes
#:   the thought span from every assistant message it re-renders. The boundary is named
#:   behaviourally ("after you have finished thinking") and never by its token — ``<channel|>``
#:   and friends are added tokens with ``special: true`` and would be encoded as control tokens
#:   inside the cached system turn (T-0055 F2, criteria 4 and 12).
UI_INSTRUCTIONS: str = f"""Your reply has two parts: the words you say and, when it helps, one small
user interface shown beside them. Use it only when seeing something helps - a list, a comparison,
steps, a card, something to click - and say the answer aloud as usual. A plain question gets
speech only; an empty or decorative surface is worse than none.
To show one, write A2UI v0.9.1 messages, one JSON object per line, inside this exact fence:
{UI_OPEN}{UI_EXAMPLE}{UI_CLOSE}
Copy that catalogId exactly, use the same surfaceId in every message, give the tree a component
with id "root", and define every id you reference in children in the same fence.
Components, and nothing else:
{", ".join(ADVERTISED_ORDER)}.
There is no table, chart, maths or flashcard component: a table is a Column of Rows of Text, a
flashcard is a Card.
- Between the fences: JSON only. No markdown, no backticks, no prose, no comments.
- One surface per turn: a handful of components, short strings.
- Write the whole surface once, in your reply, after you have finished thinking: plan it while
  you think if that helps, but anything you write while you are thinking is erased and nobody
  sees it, so open {UI_OPEN} and close {UI_CLOSE} in the reply itself.
- Never mention the fence, the JSON or the component names aloud; speech says in plain words
  what you are showing."""

# --- T-0184 C4: the v2 block (``gearvis:cards/v2``) -------------------------------------------
#
# A second block for the semantic-card catalog, selected by ``PromptBuilder(ui_block=…)``; the
# default stays :data:`UI_INSTRUCTIONS`, whose bytes do not move. The same three rules as above
# apply (literals, the fence interpolated, a bounded cost): it is no longer than v1 in code points
# and tokens (C5). It advertises the seven v2 kinds, then the six basic layout kinds; the
# interactive basic kinds are not offered (H-T184-7, the gate still accepts them), and neither
# are ``accessibility`` or ``weight``, which the v2 kinds inherit from ``ComponentCommon`` (group
# A review). It names the fence once, in the example, and states the placement rule and the
# never-aloud rule as v1 does.

#: The 13 kinds the v2 block offers, in the order its text lists them: the seven v2 kinds, then
#: the layout kinds. A tuple: this order is part of the cached prefix bytes.
ADVERTISED_ORDER_V2: tuple[str, ...] = (
    "Stat",
    "StatTiles",
    "Comparison",
    "Bars",
    "Steps",
    "Source",
    "Actions",
    "Text",
    "Column",
    "Row",
    "Card",
    "List",
    "Divider",
)

#: The v2 block's one example: a three-bar chart as a single ``Bars`` at ``root``, every value a
#: plain number and every string the card shows written once. One JSON object per line, as in v1.
UI_EXAMPLE_V2: str = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"gearvis:cards/v2"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
    '{"id":"root","component":"Bars","title":"Tallest towers","unit":"m","bars":['
    '{"label":"Burj Khalifa","value":828},'
    '{"label":"Merdeka 118","value":679},'
    '{"label":"Shanghai Tower","value":632}]}]}}'
)

UI_INSTRUCTIONS_V2: str = f"""Your reply is speech plus, when seeing something helps (figures, a comparison,
a chart, steps), one small card beside it. A plain question gets speech only.
To show one, write A2UI messages, one JSON object per line, inside this fence:
{UI_OPEN}{UI_EXAMPLE_V2}{UI_CLOSE}
Copy that catalogId, use one surfaceId and an id "root", and define every id you use.
Only these components:
{", ".join(ADVERTISED_ORDER_V2)}.
The first seven take no children; group them in a Column.
- Stat: label, value; optional detail, icon (sun, partlyCloudy, cloud, rain, storm, snow, fog,
  wind, night).
- StatTiles: tiles, each title, value; optional secondary, percent (0-100), highlight.
- Comparison: columns (2 or more), each title, optional badge; rows, each label, cells (a string,
  or text and tone ok or warn); optional highlight (a column index).
- Bars: bars, each label, value (a plain number); optional title, unit.
- Steps: steps, each title, optional detail.
- Source: text (where the facts came from).
- Actions: labels (1 to 3 short replies).
- Between the fences: JSON only, no markdown or prose. One card per turn, short strings.
- Write the whole card once, in your reply, after you have finished thinking: what you write
  while thinking is erased, so open and close the fence in the reply itself.
- Never mention the fence, the JSON or the component names aloud; speech says in plain words
  what you are showing."""
