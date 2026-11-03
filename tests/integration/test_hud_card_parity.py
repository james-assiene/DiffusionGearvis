"""T-0164 group C — the GenUI card look, measured against option A of the approved mock (C1, C3, C4,
C5), plus the card-sheet rows kept from T-0154 C1 and its C5/C6 screenshot rows.

docs/tasks/T-0164.md, group C. This module was rewritten by the T-0164 C test-author (2026-10-27).
It replaces the T-0154 group C rows that measured the vendored catalog's classes inside
``<a2ui-surface>`` shadow roots against ``mockup_20261021/cards.html`` (they skipped as "superseded by
T-0164 C" from group B's activation on). T-0164 C1 supersedes T-0154 C2 and C3's numbers by name.
Retired here, each replaced by the C row named beside it:

- ``test_the_mockup_card_renders_what_its_own_css_says`` -> ``test_c1_premise_*`` (new mock);
- ``test_the_hud_rendered_the_gated_t9_card_and_the_synthetic_card`` -> ``test_c1_premise_*``;
- ``test_c2_the_card_box_matches_the_mockup`` -> C1 "card frame";
- ``test_c2_the_heading_matches_the_mockup`` -> C1 "h3";
- ``test_c2_the_list_rows_match_the_mockup`` and
  ``test_c2_a_line_divider_sits_between_rows_and_none_above_the_first`` -> C1 "list rows";
- ``test_c2_the_card_interior_and_divider_pixels_match_the_mockup`` -> C1 probes, C1 "divider";
- ``test_c3_the_t9_card_shows_the_models_text_verbatim_in_nine_rows_and_no_checkbox`` and
  ``test_c3_no_eyebrow_number_bar_or_tick_is_painted_into_the_t9_card`` -> T-0164 C3 rows;
- ``test_c3_a_caption_renders_muted_at_the_mockups_size_and_a_divider_is_a_line`` -> C1 "caption",
  "divider";
- ``test_c3_an_unjustified_two_text_row_reads_as_key_and_value``,
  ``test_c3_only_a_row_of_exactly_two_texts_reads_as_key_and_value`` and
  ``test_c3_a_row_the_model_justified_keeps_the_models_justify`` -> C1 "key/value *" (the Row-mode
  choice itself is A's and is pinned in node by ``tests/hud/ui_render.test.mjs``);
- ``test_c3_a_checkbox_is_drawn_like_the_mockups_tick``,
  ``test_c3_a_ticked_checkbox_is_filled_with_the_accent`` and
  ``test_c3_a_ticked_and_an_unticked_checkbox_look_different`` -> the checkbox edge in C2
  (``tests/integration/test_hud_chrome.py``); option A's tick is not a C1 role;
- ``test_c3_buttons_stay_on_the_tonal_pair`` -> C1 "primary button";
- ``test_c4_every_card_text_clears_its_wcag_floor`` and
  ``test_c4_the_checkbox_edge_clears_the_control_floor`` -> T-0164 C2 rows in test_hud_chrome.py.

**Amended by T-0184 B (H-T184-5, 2026-10-28):** one frame for every card in the thread, v1
included. The "card frame" role keeps only its shadow check, and the "card interior" probe is
gone; the frame (fill, border, radius, padding) of a v1 card is pinned to the v2 frame by
``tests/integration/test_t0184_genui_v2.py``. The border-colour mutant listed below now reddens
that module's frame row instead.

**Amended by T-0184 B-H5a/B-H5b (2026-10-28, after the implementation f17952a2):** C3 ``images``
lets the card root carry exactly one ``linear-gradient(...)`` (the H-T184-5 frame) and no
``url(``; every descendant still has none (``image_problems``; its adversarial arms are injected
into the HUD page after every other measurement, ``IMAGE_MUTANTS``). The C1 "chip interior"
probe compares the chip's pixel with the frame's own pixel at that height (``FRAME_REFERENCE_PROBES``),
not with option A's ``--surface-2``. Nothing else in C1/C3 moved.

**What is compared with what.** The committed ``tests/fixtures/hud/genui_20261027/cards.html``
(the option-A column only; MANIFEST.md has its SHA-256 and how it was built) is served from its
directory by a loopback static server, theme as ``?theme=``. The HUD is served by the production
``HudStaticServer``; ``HudStub`` pushes the final canvas of each of the four payloads of the census
fixture ``tests/fixtures/a2ui/genui_census_20261027.jsonl`` (trial-omelette, desk-t8, desk-t9,
trial-flight), one surface (so one card) each, and then this module's role-probe surface (below).
Both pages in one headless Chrome, 1440x900, DPR 1, reduced motion emulated; the HUD's theme chosen
through its Settings control with the system scheme emulated the same way.

**The role-probe surface ("synthetic", turn ``t90``).** C1 names roles none of the four payloads
holds (caption, a selected chip) and C2 needs every input kind, a default-variant Button and a Modal
trigger (the B reviewers' hand-over). A hand-written canvas goes through the production ``UiGate``;
the HUD gets it as the session would push it, and the mock renders the *same* gated messages with
its own renderer (``window.A2mount``, the fixture's test hook). Its strings are this module's,
labelled as such; nothing presents them as model output.

**Readings the task leaves open, stated so a reviewer can disagree:**

- *Selectors.* The mock side is located by option A's own classes inside
  ``[data-payload=<key>] .ga-card``. The HUD side never names a class (implementation note 5: class
  names are free): it is located by ``data-a2-id`` (the model's component ids, hand-read from the
  payloads) inside ``[data-ui-surface=<id>] [data-a2ui-root]``. Two structural readings: a List's
  rows are the List element's direct children, and an input is the ``input``/``textarea`` at or
  inside its component's element.
- *Free looks* (orchestrator, 2026-10-27: "the light theme is greyer than the mock, and tabs are
  drawn as pills"). In **light**, the fills of the card frame, the nested-Card tile and the input
  are not pinned to the mock's (the mock swaps them to ``--surface``/``--surface-2``/``--surface``;
  ``card.css`` has no theme rules, ruling B (5)). There each must resolve to one of the page's own
  ``--bg``/``--surface``/``--surface-2``/``--surface-3`` (dE00 <= 2), the frame must stand off the
  pane behind it (its fill or its border dE00 >= 1, A3's border rule), and the tile must stand off
  the card (dE00 >= 2.3, a just-noticeable difference). In **dark** they are pinned to the mock.
  Tabs are not a C1 role (C1 does not list them); their contrast is C2's.
- *"line-height ratio +/- 0.02"* is line-height / font-size, as T-0154 read it.
- *Hairlines* compare each row's top border with the mock's row of the same index (width +/- 0.25
  px, colour dE00 <= 2 composited over the HUD's own backdrop when the token has alpha): that pins
  "a hairline between consecutive pairs, none above the first", and the row after a Divider, as the
  mock draws them, whatever the CSS mechanism (the mock's ``margin-top: -12px`` hack is not read).
- *Value right alignment* is measured on the value's glyphs (a Range): their right edge sits at the
  row's content-box right edge (+/- 1 px).
- *Pill radius*: the computed corner radius is at least half the element's height.
- *Shadow*: the computed ``box-shadow`` strings are equal (both come from ``--card-shadow``; the
  mock's tokens are frozen, so a later token move shows up here as a named deviation to rule on).
- *Probes* (dE00 <= 3, A3's pixel tolerance): one pixel inside the card's left padding, the tile's,
  the input's and each chip's, every element scrolled to the middle of its scroller first. In light
  the free fills' probes compare with the HUD's own computed fill (the measurement lands where the
  fill is painted) instead of the mock's pixel.
- *C3 "nothing the model did not write"*: the card's ``innerText`` (which applies CSS text
  transforms) holds exactly the non-whitespace characters of the fixture's ``expected_texts`` (the
  test-author's Python oracle, group A), minus the strings the card shows as placeholders; those are
  checked as placeholders. Any ``::before``/``::after`` whose ``content`` is not ``none``/``normal``
  fails, an empty string included (the task: "no ::before or ::after with content").
- *C4 "no card overflows its card slot"*: the card root's box lies inside its
  ``[data-card-slot]``'s box (+/- 0.5 px) and the root does not scroll sideways
  (``scrollWidth <= clientWidth + 1``). "A horizontal List scrolls inside the card": the List
  computes ``overflow-x`` auto or scroll, its ``scrollWidth`` exceeds its ``clientWidth`` (premise:
  24 items), and its box lies inside the card's. "The document never scrolls at >= 1100 px" is read
  at 1100x800 and 1440x900 with these five cards in the thread.
- *C5*: the row runs the screenshot script exactly as the T-0154 C5/C6 rows do (``--out DIR
  --compare <mockup_20261021>``) and expects ``compare-genui-{dark,light}.png`` beside the existing
  composites; the task does not say how the script finds the genui mock, so no flag is assumed.

**Activated** (T-0164 group C test-author, 2026-10-27, against the implementation 313654e6): the
rows that were red at group B's tip (here only the T-0164 C5 composites row; the C2 ones are in
``tests/integration/test_hud_chrome.py``) carried ``pending(task="T-0164")`` behind the group C
Status phrase; the markers and the gate are removed. Every other row was a guard throughout.

**State at group B's tip, and what was proven (test-author, 2026-10-27; harness deleted).** At
77094da0 (B merged; B's ``card.css`` is option A's sheet ported as is) every C1, C3 and C4 row of
this module passes in both themes (73 passed, 1 xfailed, 17 s, load 4.6): B's port already *is*
the mock for every role C1 names, light fills inside the free set. So these rows are guards for
C's tuning, not its red target; group C's red rows are the C2 floors in ``test_hud_chrome.py`` and
the C5 composites row here. The rows were proven to discriminate with a temporary harness in
``_run`` that adopted one mutant sheet into the HUD page after the cards rendered (one arm per
run, then deleted):

- ``[data-a2ui-root] { border-color: transparent !important }``: only "card frame" red, both
  themes (task: "must turn the frame row red");
- ``.ga-col > .ga-kv + .ga-kv { border-top: 0 !important }``: only "key/value hairlines" red,
  both themes (task: "a mutant that drops the hairline between key/value rows");
- ``[data-a2ui-root] .ga-chip::after { content: "\\2713" }``: only the C3 pseudo-content row red;
- ``.ga-t { overflow-wrap: normal; word-break: normal }``: only the C4 slot and long-word rows red;
- ``.ga-list.h { overflow-x: visible }``: only the C4 slot and horizontal-List rows red.

Regime (T-0164 *Regime*): Google Chrome headless (``google-chrome --version``), CDP, DPR 1,
1440x900 for parity, 1000x800, 1100x800 and 1440x900 for C4; data-theme dark and light; reduced
motion on; Playwright 1.64.0 Chromium for the C5/C6 script run; n = 1, deterministic, no seed. No
GPU.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from local_astra.contracts.protocol import AstraUiSurface
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import (
    CHAIN_JS,
    choose_theme_js,
    composite,
    painted,
    parse_css_color,
)
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_mockup_parity import (
    BORDER_MIN_DE,
    COLOUR_TOL_DE,
    FONT_SIZE_TOL_PX,
    LENGTH_TOL_PX,
    MOCKUP,
    PIXEL_TOL_DE,
    THEMES,
    VIEW,
    _media,
    _pixel,
    _playwright_chromium,
    _png_size,
    _serve_dir,
    _settle,
    decode_png,
    delta_e,
    hexc,
)
from tests.integration.test_hud_redesign import (
    UI_FIXTURE,
    _Recorder,
    _ui_commits,
    fixture_lines,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CARD_CSS = HUD / "src" / "ui" / "card.css"
UI_INDEX = HUD / "src" / "ui" / "index.js"
SCREENS = REPO / "scripts" / "demo" / "hud_screens.py"
GENUI = REPO / "tests" / "fixtures" / "hud" / "genui_20261027"
FIXTURES_MANIFEST = REPO / "tests" / "fixtures" / "hud" / "MANIFEST.md"
CENSUS = REPO / "tests" / "fixtures" / "a2ui" / "genui_census_20261027.jsonl"
GENUI_FILES = ("cards.html", "mock-dark.png", "mock-light.png")

#: C1: the four mock payloads, in the mock's order (task text).
PAYLOAD_KEYS = ("trial-omelette", "desk-t8", "desk-t9", "trial-flight")
SYNTH = "synthetic"
SYNTH_TURN = "t90"
CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

L, F, C = LENGTH_TOL_PX, FONT_SIZE_TOL_PX, COLOUR_TOL_DE
LINE_HEIGHT_RATIO_TOL = 0.02
BORDER_WIDTH_TOL = 0.25
#: dE00 a tile must stand off its card by, in the free light theme ("just noticeable").
JND = 2.3
#: C3, hand-counted in the gated payloads (component trees printed from the census fixture):
T9_ROWS = 9
FLIGHT_CHIPS = ["1", "2", "3", "4", "5"]
FLIGHT_LABELS = {
    "l1": "From:",
    "l2": "To:",
    "l3": "Departure:",
    "l4": "Return:",
    "l5": "Passengers:",
}
#: Strings the flight card shows as placeholders (f1, f2 are TextFields a Row labels; their own
#: label is the placeholder, ruling "placeholders are shown text").
PLACEHOLDERS = {"trial-flight": ["Origin city", "Destination city"]}
#: C4: a horizontal List of this many short items cannot fit a 488 px card (premise row checks).
HLIST_ITEMS = 24
LONG_WORD = "W" * 10_000
LAYOUT_VIEWS = ((1000, 800), (1100, 800), (1440, 900))

#: The role-probe canvas (labelled synthetic: this module wrote it). Root is a Column, so the mock
#: renders each child as the frame column's child in this order.
SYNTH_COMPONENTS: list[dict[str, Any]] = [
    {
        "id": "root",
        "component": "Column",
        "children": [
            "title",
            "cap",
            "body",
            "kv1",
            "kv2",
            "tf",
            "cb0",
            "cb1",
            "sl",
            "cp",
            "tabs",
            "md",
            "md2",
            "bdef",
            "bbl",
            "bpri",
            "hl",
            "word",
        ],
    },
    {"id": "title", "component": "Text", "text": "Role probe card", "variant": "h3"},
    {"id": "cap", "component": "Text", "text": "A caption this test wrote", "variant": "caption"},
    {"id": "body", "component": "Text", "text": "Body text this test wrote"},
    {"id": "kv1", "component": "Row", "children": ["k1", "v1"]},
    {"id": "k1", "component": "Text", "text": "Key one"},
    {"id": "v1", "component": "Text", "text": "Value one"},
    {"id": "kv2", "component": "Row", "children": ["k2", "v2"]},
    {"id": "k2", "component": "Text", "text": "Key two"},
    {"id": "v2", "component": "Text", "text": "Value two"},
    {"id": "tf", "component": "TextField", "label": "Field label", "value": "Typed value"},
    {"id": "cb0", "component": "CheckBox", "label": "Unticked", "value": False},
    {"id": "cb1", "component": "CheckBox", "label": "Ticked", "value": True},
    {"id": "sl", "component": "Slider", "label": "Amount", "min": 0, "max": 100, "value": 50},
    {
        "id": "cp",
        "component": "ChoicePicker",
        "label": "Size",
        "variant": "mutuallyExclusive",
        "options": [{"label": "Small", "value": "s"}, {"label": "Large", "value": "l"}],
        "value": ["l"],
    },
    {
        "id": "tabs",
        "component": "Tabs",
        "tabs": [{"title": "First tab", "child": "tab1"}, {"title": "Second tab", "child": "tab2"}],
    },
    {"id": "tab1", "component": "Text", "text": "First panel"},
    {"id": "tab2", "component": "Text", "text": "Second panel"},
    {"id": "md", "component": "Modal", "trigger": "mdt", "content": "mdc"},
    {
        "id": "mdt",
        "component": "Button",
        "variant": "default",
        "child": "mdtl",
        "action": {"event": {"name": "open"}},
    },
    {"id": "mdtl", "component": "Text", "text": "More details"},
    {"id": "mdc", "component": "Text", "text": "Modal content"},
    # ruling 2396676a (b): a Modal whose trigger is a Text, so the renderer's own disclosure
    # control is drawn (a Button trigger is drawn as that Button)
    {"id": "md2", "component": "Modal", "trigger": "mdt2", "content": "mdc2"},
    {"id": "mdt2", "component": "Text", "text": "Show more"},
    {"id": "mdc2", "component": "Text", "text": "More content"},
    {
        "id": "bdef",
        "component": "Button",
        "variant": "default",
        "child": "bdefl",
        "action": {"event": {"name": "a"}},
    },
    {"id": "bdefl", "component": "Text", "text": "Default button"},
    {
        "id": "bbl",
        "component": "Button",
        "variant": "borderless",
        "child": "bbll",
        "action": {"event": {"name": "b"}},
    },
    {"id": "bbll", "component": "Text", "text": "Borderless button"},
    {
        "id": "bpri",
        "component": "Button",
        "variant": "primary",
        "child": "bpril",
        "action": {"event": {"name": "c"}},
    },
    {"id": "bpril", "component": "Text", "text": "Primary button"},
    {
        "id": "hl",
        "component": "List",
        "direction": "horizontal",
        "children": [f"h{i}" for i in range(HLIST_ITEMS)],
    },
    *[{"id": f"h{i}", "component": "Text", "text": f"Item{i:02d}"} for i in range(HLIST_ITEMS)],
    {"id": "word", "component": "Text", "text": LONG_WORD},
]


def synthetic_surface() -> AstraUiSurface:
    raw = "\n".join(
        json.dumps(m)
        for m in (
            {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": CATALOG_ID}},
            {
                "version": "v0.9.1",
                "updateComponents": {"surfaceId": "s", "components": SYNTH_COMPONENTS},
            },
        )
    )
    result = UiGate().gate(raw, turn_id=SYNTH_TURN, canvas_idx=0, final=True)
    assert result.surface is not None, f"the gate refused the role-probe canvas: {result.report}"
    return result.surface


def census() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in CENSUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("key"):
                out[str(row["key"])] = row
    return out


def _surface_id_of(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if "createSurface" in m:
            return str(m["createSurface"]["surfaceId"])
    raise AssertionError(f"no createSurface in {messages!r:.200}")


def payload_surface(key: str) -> AstraUiSurface:
    """The payload's final canvas as the session pushes it: one surface, ``final``."""
    messages = list(census()[key]["canvases"][-1])
    return AstraUiSurface(
        surface_id=_surface_id_of(messages), canvas_idx=0, final=True, messages=tuple(messages)
    )


def surface_ids() -> dict[str, str]:
    out = {key: payload_surface(key).surface_id for key in PAYLOAD_KEYS}
    out[SYNTH] = synthetic_surface().surface_id
    return out


def hud_sel(*ids: str, inner: str | None = None) -> str:
    """A HUD selector by component id; ``inner`` matches the element itself or a descendant."""
    parts = []
    for cid in ids:
        base = f'[data-a2-id="{cid}"]'
        parts += [f"{base}:is({inner})", f"{base} :is({inner})"] if inner else [base]
    return ", ".join(parts)


# --- C1: the role table -------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """One row of the C1 role table (task: role, mock selector, HUD selector, property or probe,
    tolerance, deviation note)."""

    name: str
    key: str
    mock: str
    hud: str
    checks: tuple[str, ...]
    note: str = ""
    #: checks that are free in light (the "greyer" ruling): fills only
    light_free: tuple[str, ...] = ()


_INPUT = "input, textarea"
ROLES: tuple[Role, ...] = (
    # T-0184 H-T184-5 amends this row by name: one frame for every card, v1 included, so the
    # frame's fill, border, radius and padding are no longer option A's; they are pinned for v1 and
    # v2 cards alike by tests/integration/test_t0184_genui_v2.py ("B3 card frame"). Only the shadow,
    # --card-shadow on both pages, is still compared with the mock here.
    Role(
        "card frame",
        "desk-t9",
        "",
        "",
        ("shadow",),
        "--card-shadow (fill, border, radius and padding: T-0184 B3, H-T184-5)",
    ),
    Role("h3", "desk-t9", ".ga-h3", hud_sel("h"), ("font", "color"), "17 px / 650 / 1.3"),
    Role(
        "body text",
        "desk-t9",
        ".ga-li > .ga-t",
        hud_sel(*(f"i{i}" for i in range(1, 10))),
        ("font", "color"),
        "the nine list rows' text, --text-base, --ink",
    ),
    Role("caption", SYNTH, ".ga-caption", hud_sel("cap"), ("font", "color"), "12.5 px --muted"),
    Role(
        "key/value key",
        "desk-t8",
        ".ga-kv:not(.ga-hr + .ga-kv) > .ga-k",
        hud_sel("a1", "a2", "a3", "a4", "a5"),
        ("font", "color", "left"),
        "muted key at the row's left",
    ),
    Role(
        "key/value value",
        "desk-t8",
        ".ga-kv:not(.ga-hr + .ga-kv) > .ga-v",
        hud_sel("b1", "b2", "b3", "b4", "b5"),
        ("font", "color", "right"),
        "value weight 550, glyphs at the row's right",
    ),
    Role(
        "key/value hairlines",
        "desk-t8",
        ".ga-col > .ga-kv",
        hud_sel("r1", "r2", "r3", "r4", "r5", "r7"),
        ("hairlines",),
        "a 1 px --line between consecutive pairs, none above the first, none after the Divider",
    ),
    Role(
        "row after a Divider",
        "desk-t8",
        ".ga-hr + .ga-kv > :is(.ga-k, .ga-v)",
        hud_sel("a7", "b7"),
        ("font", "color"),
        "H-T164-7: the key in --ink 600, the value 700",
    ),
    Role(
        "list rows",
        "desk-t9",
        ".ga-list > .ga-li",
        '[data-a2-id="l"] > *',
        ("padding", "hairlines"),
        "9 px 0, a 1 px --line between rows, none above the first",
    ),
    Role(
        "nested-Card tile",
        "trial-omelette",
        ".ga-tile",
        hud_sel(*(f"step{i}" for i in range(1, 7))),
        ("fill", "radius", "padding"),
        "fill free in light (greyer; must stand off the card); --radius-btn, 11/14",
        light_free=("fill",),
    ),
    Role(
        "field label",
        "trial-flight",
        ".ga-field > .ga-flabel",
        hud_sel("l1", "l2", "l3", "l4", "l5"),
        ("font", "color"),
        "--text-sm 550 --dim, above the control",
    ),
    Role(
        "input",
        "trial-flight",
        ".ga-input",
        hud_sel("f1", "f2", "d1", "d2", inner=_INPUT),
        ("fill", "border", "radius", "padding", "font_size"),
        "fill free in light; 1 px --edge-strong, --radius-btn, 9/12, 14.5 px",
        light_free=("fill",),
    ),
    Role(
        "chip",
        "trial-flight",
        ".ga-chip",
        hud_sel("p1", inner="button"),
        ("border", "pill", "padding"),
        "1 px --edge-strong, pill, 5/13",
    ),
    Role(
        "selected chip",
        SYNTH,
        '.ga-chip[aria-pressed="true"]',
        hud_sel("cp", inner='button:is([aria-checked="true"], [aria-pressed="true"])'),
        ("fill", "color"),
        "--accent-soft fill, --accent ink",
    ),
    Role(
        "primary button",
        "trial-flight",
        ".ga-btn",
        hud_sel("btn", inner="button"),
        ("fill", "color", "pill", "padding"),
        "an omitted variant is primary (ruling A review): --tonal / --tonal-ink, pill, 9/18",
    ),
    Role(
        "divider",
        "desk-t8",
        ".ga-hr",
        hud_sel("d"),
        ("border_top",),
        "1 px --line-strong",
    ),
)
ROLE_BY_NAME = {r.name: r for r in ROLES}

#: (name, key, mock selector, HUD selector, x offset into the element, free in light)
#: The "card interior" probe moved to test_t0184_genui_v2.py by name (H-T184-5: the frame is the
#: v2 frame, a gradient in dark, for every card). The "chip interior" probe moved the same way
#: (T-0184 B-H5b): its reference is in ``FRAME_REFERENCE_PROBES``.
PROBES: tuple[tuple[str, str, str, str, float, bool], ...] = (
    ("tile interior", "trial-omelette", ".ga-tile", hud_sel("step1"), 5.0, True),
    ("input interior", "trial-flight", ".ga-input", hud_sel("f1", inner=_INPUT), 5.0, True),
    ("chip interior", "trial-flight", ".ga-chip", hud_sel("p1", inner="button"), 6.0, False),
    (
        "selected chip interior",
        SYNTH,
        '.ga-chip[aria-pressed="true"]',
        hud_sel("cp", inner='button:is([aria-checked="true"], [aria-pressed="true"])'),
        6.0,
        False,
    ),
)
#: T-0184 B-H5b: an unselected chip is transparent (the v1 chip gets no fill), so its interior pixel
#: is compared with the H-T184-5 frame at that point, not with option A's ``--surface-2``. "The
#: frame at that point" is read as the frame's own pixel at the probe's height, sampled 8 px inside
#: the card root's left edge (inside its 22-24 px padding, where only the frame paints; the dark
#: frame is a vertical gradient, so the height is what matters). Both themes; dE00 <= 3.
FRAME_REFERENCE_PROBES = frozenset({"chip interior"})
FRAME_REF_DX = 8.0
_FREE_TOKENS = ("--bg", "--surface", "--surface-2", "--surface-3")

_COMMON_JS = (
    CHAIN_JS
    + r"""
  const px = (v) => parseFloat(v) || 0;
  const SIDE = __SIDE__, SIDS = __SIDS__;
  const container = (key) => {
    if (SIDE === 'mock') return document.querySelector('[data-payload="' + key + '"] .ga-card');
    const w = document.querySelector('[data-ui-surface="' + SIDS[key] + '"]');
    return w ? w.querySelector('[data-a2ui-root]') : null;
  };
  const pick = (root, sel) => (!root ? [] : sel ? [...root.querySelectorAll(sel)] : [root]);
  const textNode = (el) => {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) if (n.textContent.trim()) return n;
    return null;
  };
  const glyphs = (el) => {
    const n = textNode(el);
    if (!n) return null;
    const range = document.createRange();
    range.selectNodeContents(n);
    const r = range.getBoundingClientRect();
    return [r.left, r.top, r.right, r.bottom];
  };
  const contentBox = (el) => {
    if (!el) return null;
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return [r.left + px(s.borderLeftWidth) + px(s.paddingLeft),
            r.right - px(s.borderRightWidth) - px(s.paddingRight)];
  };
  const look = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return {
      tag: el.tagName.toLowerCase(), x: r.x, y: r.y, w: r.width, h: r.height,
      pad: [px(s.paddingTop), px(s.paddingRight), px(s.paddingBottom), px(s.paddingLeft)],
      bw: [px(s.borderTopWidth), px(s.borderRightWidth), px(s.borderBottomWidth), px(s.borderLeftWidth)],
      border_style: s.borderTopStyle, border_color: s.borderTopColor,
      radius: px(s.borderTopLeftRadius), shadow: s.boxShadow, background: s.backgroundColor,
      color: s.color, font_size: px(s.fontSize), font_weight: parseInt(s.fontWeight, 10) || 400,
      line_height: px(s.lineHeight), text: (el.textContent || '').trim().slice(0, 60),
      chain: chainOf(el), parent_chain: chainOf(el.parentElement || el),
      glyphs: glyphs(el), parent_box: contentBox(el.parentElement),
    };
  };
"""
)


def _js(body: str, side: str, sids: dict[str, str], **subs: Any) -> str:
    out = ("(() => {" + _COMMON_JS + body + "\n})()").replace("__SIDE__", json.dumps(side))
    out = out.replace("__SIDS__", json.dumps(sids))
    for name, value in subs.items():
        out = out.replace(f"__{name}__", json.dumps(value))
    return out


ROLE_BODY = r"""
  const out = {};
  for (const [name, key, sel] of __SPECS__) {
    const root = container(key);
    out[name] = {root: !!root, matches: pick(root, sel).map(look)};
  }
  const tok = (t) => {
    const p = document.createElement('div');
    p.style.cssText = 'position:fixed;left:-50px;top:0;width:1px;height:1px;background-color:var(' + t + ')';
    document.body.appendChild(p);
    const v = getComputedStyle(p).backgroundColor;
    p.remove();
    return v;
  };
  out.__tokens = Object.fromEntries(__TOKENS__.map((t) => [t, tok(t)]));
  out.__theme = document.documentElement.getAttribute('data-theme');
  return out;
"""

PROBE_BODY = r"""
  const [key, sel, dx] = __PROBE__;
  const el = pick(container(key), sel)[0];
  if (!el) return null;
  el.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'instant'});
  const r = el.getBoundingClientRect();
  const x = Math.floor(r.left + dx), y = Math.floor(r.top + r.height / 2);
  // An older card's wrapper is `inert` (H-T154-4), and inert content is not hit-tested, so the
  // wrapper is lifted for this one synchronous hit test and put back (nothing paints between).
  const inert = el.closest('[inert]');
  if (inert) inert.inert = false;
  const hit = document.elementFromPoint(x + 0.5, y + 0.5);
  // B-H5b: the frame's own pixel at the probe's height (the card root's left padding)
  const root = container(key);
  const fx = Math.floor(root.getBoundingClientRect().left + __FRAME_DX__);
  const fhit = document.elementFromPoint(fx + 0.5, y + 0.5);
  if (inert) inert.inert = true;
  return {x, y, in_el: !!hit && (hit === el || el.contains(hit)), fill: chainOf(el),
          frame_x: fx, frame_hit_is_root: fhit === root};
"""

#: C3: what each card shows, read in the page.
C3_BODY = r"""
  const out = {};
  for (const key of Object.keys(SIDS)) {
    const root = container(key);
    if (!root) { out[key] = null; continue; }
    const all = [root, ...root.querySelectorAll('*')];
    const pseudo = [];
    for (const el of all) for (const which of ['::before', '::after']) {
      const c = getComputedStyle(el, which).content;
      if (c && c !== 'none' && c !== 'normal') pseudo.push([el.tagName.toLowerCase() + '[data-a2-id=' + (el.getAttribute('data-a2-id') || '') + ']', which, c]);
    }
    const id = (cid) => root.querySelector('[data-a2-id="' + cid + '"]');
    out[key] = {
      inner_text: root.innerText,
      placeholders: [...root.querySelectorAll('input, textarea')].map((i) => i.placeholder).filter(Boolean),
      pseudo,
      checkboxes: root.querySelectorAll('input[type=checkbox], [role=checkbox]').length,
      list_items: all.filter((e) => getComputedStyle(e).display === 'list-item').map((e) => e.tagName.toLowerCase()),
      images: all.filter((e) => getComputedStyle(e).backgroundImage !== 'none')
        .map((e) => [e === root, e.tagName.toLowerCase() + '[data-a2-id=' + (e.getAttribute('data-a2-id') || '') + ']', getComputedStyle(e).backgroundImage]),
      meters: root.querySelectorAll('progress, meter').length,
      t9_rows: id('l') ? id('l').children.length : null,
      flight_chips: id('p1') ? [...id('p1').querySelectorAll('button')].map((b) => b.textContent) : null,
      flight_labels: Object.fromEntries(['l1', 'l2', 'l3', 'l4', 'l5'].map((c) => [c, id(c) ? id(c).textContent : null])),
    };
  }
  return out;
"""

#: C4: every card against its slot, the long word, the horizontal List, the document.
LAYOUT_BODY = r"""
  const out = {cards: {}, doc: null};
  for (const key of Object.keys(SIDS)) {
    const root = container(key);
    if (!root) { out.cards[key] = null; continue; }
    const slot = root.closest('[data-card-slot]');
    const box = (e) => { if (!e) return null; const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
    const word = root.querySelector('[data-a2-id="word"]');
    const hl = root.querySelector('[data-a2-id="hl"]');
    out.cards[key] = {
      root: box(root), slot: box(slot), root_scroll: [root.scrollWidth, root.clientWidth],
      word: word ? {box: box(word), line: px(getComputedStyle(word).lineHeight) || px(getComputedStyle(word).fontSize) * 1.5} : null,
      hl: hl ? {box: box(hl), overflow_x: getComputedStyle(hl).overflowX, scroll: [hl.scrollWidth, hl.clientWidth],
                items: [...hl.children].map((li) => {
                  const g = glyphs(li), n = textNode(li);
                  // a squeezed item wraps its text (overflow-wrap: anywhere), so its glyph box
                  // shrinks with it: count the text's lines too (distinct client-rect tops)
                  let lines = 0;
                  if (n) {
                    const range = document.createRange();
                    range.selectNodeContents(n);
                    lines = new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size;
                  }
                  return {w: li.getBoundingClientRect().width, glyph_w: g ? g[2] - g[0] : null,
                          lines, text: (li.textContent || '').trim()};
                })} : null,
    };
  }
  const se = document.scrollingElement;
  out.doc = {scroll: [se.scrollWidth, se.scrollHeight], view: [window.innerWidth, window.innerHeight]};
  return out;
"""

#: C1 (kept from T-0154): the sheet the page serves, parsed in the page as the oracle, and every
#: copy of it in every surface's shadow root. ``tests/integration/test_hud_ui_surface.py`` imports
#: this name for its B5 one-page-sheet row; kept byte-identical.
SHEET_JS = r"""
(async () => {
  const resp = await fetch('/src/ui/card.css', {cache: 'no-store'});
  if (!resp.ok) return {ok: false, status: resp.status};
  const oracle = new CSSStyleSheet();
  oracle.replaceSync(await resp.text());
  const sels = (sheet) => {
    const out = [];
    const walk = (rules) => { for (const r of rules) { if (r.selectorText !== undefined) out.push(r.selectorText);
                                                      if (r.cssRules) walk(r.cssRules); } };
    try { walk(sheet.cssRules); } catch (e) { return null; }
    return out;
  };
  const want = JSON.stringify(sels(oracle));
  const copies = (root) => [...(root.adoptedStyleSheets || []), ...(root.styleSheets || [])]
    .filter((s) => JSON.stringify(sels(s)) === want).length;
  return {
    ok: true, rules: JSON.parse(want).length,
    surfaces: [...document.querySelectorAll('a2ui-surface')].map((h) => ({
      id: (h.closest('[data-ui-surface]') || {getAttribute: () => null}).getAttribute('data-ui-surface'),
      shadow: !!h.shadowRoot, copies: h.shadowRoot ? copies(h.shadowRoot) : 0,
      sheets: h.shadowRoot ? (h.shadowRoot.adoptedStyleSheets || []).length + h.shadowRoot.styleSheets.length : 0})),
  };
})()
"""


@dataclass
class Side:
    roles: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)
    pixels: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    c3: dict[str, Any] = field(default_factory=dict)


@dataclass
class Parity:
    error: str = ""
    seconds: float = 0.0
    sids: dict[str, str] = field(default_factory=dict)
    mock: dict[str, Side] = field(default_factory=dict)
    hud: dict[str, Side] = field(default_factory=dict)
    hud_theme: dict[str, Any] = field(default_factory=dict)
    rendered: bool = False
    layout: dict[str, Any] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    hud_origin: str = ""
    page_requests: list[str] = field(default_factory=list)
    #: B-H5a's adversarial arms: arm -> the HUD's C3 reading with that rule injected
    c3_mutants: dict[str, dict[str, Any]] = field(default_factory=dict)


_GRADIENT = "linear-gradient(rgb(1, 2, 3), rgb(4, 5, 6))"
_DATA_URL = 'url("data:image/gif;base64,R0lGODlhAQABAAAAACw=")'
#: T-0184 B-H5a adversarial arms, each a rule injected into the HUD page after every other
#: measurement (and after the request log is read); (rule, what the checker must report).
IMAGE_MUTANTS: dict[str, tuple[str, str]] = {
    "url on the root": (
        f"[data-a2ui-root] {{ background-image: {_DATA_URL} !important; }}",
        "the card root",
    ),
    "url beside the gradient on the root": (
        f"[data-a2ui-root] {{ background-image: {_GRADIENT}, {_DATA_URL} !important; }}",
        "the card root",
    ),
    "two gradients on the root": (
        f"[data-a2ui-root] {{ background-image: {_GRADIENT}, {_GRADIENT} !important; }}",
        "the card root",
    ),
    "gradient on a descendant": (
        f"[data-a2ui-root] > :first-child {{ background-image: {_GRADIENT} !important; }}",
        "descendant",
    ),
    "frame gradient moved onto a descendant": (
        "[data-a2ui-root] { background-image: none !important; } "
        f"[data-a2ui-root] > :first-child {{ background-image: {_GRADIENT} !important; }}",
        "descendant",
    ),
}
_INJECT_JS = (
    "(() => { const s = document.createElement('style'); s.id = 't0184-bh5a-mutant';"
    " s.textContent = __CSS__; document.head.appendChild(s); return true; })()"
)
_UNINJECT_JS = "(() => { const s = document.getElementById('t0184-bh5a-mutant'); if (s) s.remove(); return true; })()"


def _specs(side: str) -> list[list[str]]:
    return [[r.name, r.key, r.mock if side == "mock" else r.hud] for r in ROLES]


def _measure(chrome: _Recorder, side: str, sids: dict[str, str]) -> Side:
    out = Side()
    out.roles = dict(
        chrome.evaluate(_js(ROLE_BODY, side, sids, SPECS=_specs(side), TOKENS=list(_FREE_TOKENS)))
        or {}
    )
    for name, key, mock_sel, hud_sel_, dx, _free in PROBES:
        sel = mock_sel if side == "mock" else hud_sel_
        p = chrome.evaluate(
            _js(PROBE_BODY, side, sids, PROBE=[key, sel, dx], FRAME_DX=FRAME_REF_DX)
        )
        if not p:
            continue
        _settle(chrome)
        out.probes[name] = p
        out.pixels[name] = _pixel(chrome, int(p["x"]), int(p["y"]))
        if name in FRAME_REFERENCE_PROBES:
            out.pixels[name + " @frame"] = _pixel(chrome, int(p["frame_x"]), int(p["y"]))
    return out


def _run(result: Parity, chrome: _Recorder) -> None:
    sids = surface_ids()
    result.sids = sids
    synth = synthetic_surface()
    synth_messages = [dict(m) for m in synth.messages]
    mock_server, mock_port = _serve_dir(GENUI)
    try:
        chrome.reset()
        chrome._cmd("Runtime.enable")
        chrome._cmd("Network.enable")
        for theme in THEMES:
            _media(chrome, theme)
            chrome.viewport(*VIEW)
            chrome.navigate(f"http://127.0.0.1:{mock_port}/cards.html?theme={theme}")
            chrome.wait_until("window.__ready === true", 15)
            chrome.evaluate(f"window.A2mount({json.dumps(SYNTH)}, {json.dumps(synth_messages)})")
            _settle(chrome)
            side = _measure(chrome, "mock", sids)
            side.c3 = dict(chrome.evaluate(_js(C3_BODY, "mock", sids)) or {})
            result.mock[theme] = side
    finally:
        mock_server.shutdown()
        mock_server.server_close()

    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        result.hud_origin = f"http://127.0.0.1:{server.port}"
        mark = len(chrome.requests)
        url = f"{result.hud_origin}/?ws=" + urllib.parse.quote(stub.url, safe="")
        _media(chrome, "dark")
        chrome.viewport(*VIEW)
        chrome.navigate(url)
        stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
        stub.wait_for_client(10)
        chrome.wait_panes(8)
        for key in PAYLOAD_KEYS:
            stub.push(payload_surface(key))
            time.sleep(0.05)
        stub.push(synth)
        ready = (
            "(() => { const ids = " + json.dumps(list(sids.values())) + ";"
            " return ids.every((id) => { const w = document.querySelector("
            "'[data-ui-surface=\"' + id + '\"]'); return w && w.querySelector('[data-a2ui-root] "
            "[data-a2-id]'); }); })()"
        )
        result.rendered = bool(chrome.wait_until(ready, 15))
        chrome.drain(0.3)
        result.page_requests = list(chrome.requests[mark:])
        for theme in THEMES:
            result.hud_theme[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
            _media(chrome, theme)
            _settle(chrome)
            side = _measure(chrome, "hud", sids)
            side.c3 = dict(chrome.evaluate(_js(C3_BODY, "hud", sids)) or {})
            result.hud[theme] = side
        for w, h in LAYOUT_VIEWS:
            chrome.viewport(w, h)
            _settle(chrome)
            result.layout[f"{w}x{h}"] = chrome.evaluate(_js(LAYOUT_BODY, "hud", sids)) or {}
        chrome.viewport(*VIEW)
        chrome.evaluate(choose_theme_js("system"))
        chrome.drain(0.2)
        result.requests = list(chrome.requests)
        for arm, (css, _what) in IMAGE_MUTANTS.items():
            chrome.evaluate(_INJECT_JS.replace("__CSS__", json.dumps(css)))
            _settle(chrome)
            result.c3_mutants[arm] = dict(chrome.evaluate(_js(C3_BODY, "hud", sids)) or {})
            chrome.evaluate(_UNINJECT_JS)
            _settle(chrome)
    finally:
        server.close()
        stub.stop()


@pytest.fixture(scope="module")
def parity() -> Iterator[Parity]:
    """Every browser measurement, in one headless Chrome closed again before any row runs (so the
    Playwright Chromium of the C5/C6 rows never runs beside it). Never raises."""
    result = Parity()
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        result.error = "no chrome"
        yield result
        return
    started = time.monotonic()
    try:
        with _Recorder(CHROME) as chrome:
            chrome._cmd("Runtime.enable")
            chrome._cmd("Network.enable")
            _run(result, chrome)
    except Exception as exc:  # a broken session fails in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


def _ok(p: Parity) -> None:
    assert not p.error, f"the card parity session failed before any assertion ran: {p.error}"
    assert p.rendered, f"the HUD never rendered all five cards ({sorted(p.sids.values())})"


# --- static rows (no browser) ----------------------------------------------------------------


def _card_css() -> str:
    assert CARD_CSS.is_file(), f"the page's card sheet {CARD_CSS.relative_to(REPO)} does not exist"
    return re.sub(r"/\*[\s\S]*?\*/", " ", CARD_CSS.read_text(encoding="utf-8"))


#: Colour syntax that is not a token reference. ``var(--x, <literal>)`` fallbacks count too.
_COLOUR_LITERAL = re.compile(
    r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I
)
_NAMED_COLOURS = [
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "gray",
    "grey",
    "silver",
    "navy",
    "teal",
    "maroon",
    "olive",
    "lime",
    "aqua",
    "fuchsia",
]


def test_c1_the_card_sheet_paints_only_tokens():
    """T-0154 C1 / T-0164 B5 kept: no hex, functional or named colour in card.css (``transparent``,
    ``currentColor`` and ``inherit`` are not palette colours). Comments stripped first."""
    css = _card_css()
    found = [m.group(0) for m in _COLOUR_LITERAL.finditer(css)]
    for decl in re.findall(r":\s*([^;{}]+)", css):
        for word in re.findall(r"(?<![\w-])([a-zA-Z]+)(?![\w-])", decl):
            if word.lower() in _NAMED_COLOURS:
                found.append(word)
    assert found == [], f"C1: colour literals in card.css (tokens only): {found}"


def test_c1_the_card_sheet_writes_no_content_and_transforms_no_text():
    """T-0154 C1 / T-0164 B5 kept, statically: no ``content`` but ``none``/``normal``, no
    ``text-transform`` but ``none`` (comments stripped)."""
    css = _card_css()
    bad = [
        f"{prop}: {value.strip()}"
        for prop, value in re.findall(r"(?<![\w-])(content|text-transform)\s*:\s*([^;}]*)", css)
        if value.strip().lower() not in {"none", "normal"}
    ]
    assert bad == [], f"C1: card.css alters what the model wrote: {bad}"


def test_c1_the_card_sheet_loads_nothing_no_url_and_no_import():
    """B reviewers' hand-over to C (A11 for the sheet): card.css holds no ``url(`` (an image, font
    or cursor would be a fetch the page makes on the model's behalf) and no ``@import``. Comments
    are stripped, so prose saying "url(" does not trip it; a string inside a declaration would."""
    css = _card_css()
    found = re.findall(r"url\s*\(|@import\b|image-set\s*\(", css, re.I)
    assert found == [], f"card.css loads something: {found}"


def test_c1_node_can_still_import_the_ui_module_and_mount_without_a_document():
    """T-0154 C1 kept: node imports ``hud/src/ui/index.js`` and ``mount({})`` resolves unmounted."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (
        f"const m = await import({json.dumps(UI_INDEX.as_uri())});"
        "const r = await m.mount({log: () => {}});"
        "console.log(JSON.stringify(r));"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"node could not import/mount the ui module:\n{proc.stderr}"
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"mounted": False}, proc.stdout


def _manifest_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in FIXTURES_MANIFEST.read_text(encoding="utf-8").splitlines():
        m = re.search(r"genui_20261027/([A-Za-z0-9_.-]+)`?.*?\b([0-9a-f]{64})\b", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_c1_premise_the_genui_mock_is_committed_byte_for_byte_as_the_manifest_records():
    """C1 / Owned files: the option-A fixture and the two screenshots James saw, each SHA-256 equal
    to the MANIFEST row; the fixture holds the option-A sheet and no "today" column."""
    hashes = _manifest_hashes()
    assert sorted(hashes) == sorted(GENUI_FILES), hashes
    for name in GENUI_FILES:
        digest = hashlib.sha256((GENUI / name).read_bytes()).hexdigest()
        assert digest == hashes[name], f"{name} is {digest}, the MANIFEST says {hashes[name]}"
    html = (GENUI / "cards.html").read_text(encoding="utf-8")
    assert ".ga-card {" in html and "OPTION A" in html
    assert "data:image/png" not in html and "shot-light" not in html.split("<body>")[1], (
        "the fixture still embeds the today column"
    )
    for key in PAYLOAD_KEYS:
        assert f'"key": "{key}"' in html, f"the fixture does not embed the {key} payload"


def test_c1_premise_the_payloads_and_the_role_probe_canvas_are_what_the_rows_assume():
    """Premise of every browser row, hand-read from the census fixture: the four payloads' final
    canvases hold the component ids the HUD selectors name, the t9 card has 9 list rows and no
    CheckBox, the flight card 5 chips labelled 1-5 and the five labels; the production gate
    accepts the role-probe canvas whole and keeps its ids."""
    rows = census()

    def comps(key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for m in rows[key]["canvases"][-1]:
            for c in (m.get("updateComponents") or {}).get("components") or []:
                out[str(c["id"])] = c
        return out

    t9, t8, om, fl = (comps(k) for k in ("desk-t9", "desk-t8", "trial-omelette", "trial-flight"))
    assert t9["l"]["children"] == [f"i{i}" for i in range(1, T9_ROWS + 1)]
    assert "CheckBox" not in {c["component"] for c in t9.values()}
    assert t9["h"]["variant"] == "h3"
    assert [t8[f"r{i}"]["component"] for i in (1, 2, 3, 4, 5, 7)] == ["Row"] * 6
    assert t8["d"]["component"] == "Divider"
    assert t8["__gate_root_body"]["children"].index("d") == 6
    assert [om[f"step{i}"]["component"] for i in range(1, 7)] == ["Card"] * 6
    assert [o["label"] for o in fl["p1"]["options"]] == FLIGHT_CHIPS
    assert {k: fl[k]["text"] for k in FLIGHT_LABELS} == FLIGHT_LABELS
    assert fl["f1"]["component"] == "TextField" and fl["d1"]["component"] == "DateTimeInput"
    assert "variant" not in fl["btn"], "the flight button is the omitted-variant (primary) case"
    synth = synthetic_surface()
    ids = {
        str(c["id"])
        for m in synth.messages
        for c in (m.get("updateComponents") or {}).get("components") or []
    }
    assert {c["id"] for c in SYNTH_COMPONENTS} <= ids, "the gate renamed a role-probe id"
    assert synth.surface_id == f"turn-{SYNTH_TURN}"
    assert len(set(surface_ids().values())) == 5


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_c1_premise_both_pages_render_every_role(parity, theme):
    """Premise of the role table: every role is found on both pages, the same number of times
    (an implementation that renders a role twice, or not at all, is not compared silently)."""
    _ok(parity)
    m, h = parity.mock[theme].roles, parity.hud[theme].roles
    assert m.get("__theme") == theme and h.get("__theme") == theme, (
        m.get("__theme"),
        h.get("__theme"),
    )
    problems = []
    for r in ROLES:
        nm, nh = len(m[r.name]["matches"]), len(h[r.name]["matches"])
        if nm == 0 or nm != nh:
            problems.append(f"{r.name}: the mock has {nm} match(es) ({r.mock!r}), the HUD {nh}")
    assert problems == [], f"data-theme={theme}:\n  " + "\n  ".join(problems)


# --- C1 checks ---------------------------------------------------------------------------------


def _backdrop(el: dict[str, Any]) -> tuple[float, float, float]:
    return painted(list(el["chain"]))


def _colour_de(
    hud_css: str, hud_el: dict[str, Any], mock_css: str, mock_el: dict[str, Any]
) -> float:
    """Each colour over its own page's painted backdrop; an alpha mock colour (``--line``) is
    composited over the HUD's backdrop on both sides, so the line is compared as a token."""
    mock_rgba = parse_css_color(mock_css)
    got = composite(parse_css_color(hud_css), _backdrop(hud_el))
    behind = _backdrop(hud_el) if mock_rgba[3] < 1 else _backdrop(mock_el)
    return delta_e(got, composite(mock_rgba, behind))


def _free_fill(h: dict[str, Any], tokens: dict[str, str]) -> str | None:
    got = painted(list(h["chain"]))
    best = min((delta_e(got, parse_css_color(v)[:3]), t) for t, v in tokens.items() if v)
    if best[0] > C:
        return f"fill {hexc(got)} is none of {list(tokens)} (nearest {best[1]}, dE00 {best[0]:.2f})"
    return None


def _check(
    role: Role,
    check: str,
    i: int,
    h: dict[str, Any],
    m: dict[str, Any],
    theme: str,
    hud: dict[str, Any],
) -> list[str]:
    where = f"{role.name} #{i} {h.get('text', '')[:24]!r}"
    out: list[str] = []
    if check == "fill":
        if theme == "light" and "fill" in role.light_free:
            p = _free_fill(h, hud["__tokens"])
            if p:
                out.append(f"{where}: {p} (light is free, but only on the page's surfaces)")
            got = painted(list(h["chain"]))
            behind = painted(list(h["parent_chain"])[1:])
            if role.name == "card frame":
                edge = composite(parse_css_color(h["border_color"]), got) if h["bw"][0] > 0 else got
                if max(delta_e(got, behind), delta_e(edge, behind)) + 1e-9 < BORDER_MIN_DE:
                    out.append(f"{where}: the frame does not stand off the pane {hexc(behind)}")
            if role.name == "nested-Card tile" and delta_e(got, behind) + 1e-9 < JND:
                out.append(
                    f"{where}: tile {hexc(got)} is dE00 {delta_e(got, behind):.2f} from the card "
                    f"{hexc(behind)}, needs >= {JND} to read as a tile"
                )
            return out
        de = delta_e(painted(list(h["chain"])), painted(list(m["chain"])))
        if de > C:
            out.append(
                f"{where}: fill {hexc(painted(list(h['chain'])))} vs the mock's "
                f"{hexc(painted(list(m['chain'])))}, dE00 {de:.2f} > {C}"
            )
    elif check in ("border", "border_top"):
        sides = range(4) if check == "border" else range(1)
        for s in sides:
            if abs(h["bw"][s] - m["bw"][s]) > BORDER_WIDTH_TOL:
                out.append(f"{where}: border width[{s}] {h['bw'][s]} vs the mock's {m['bw'][s]}")
        if m["bw"][0] > 0:
            if h["border_style"] != m["border_style"]:
                out.append(f"{where}: border style {h['border_style']} vs {m['border_style']}")
            de = _colour_de(h["border_color"], h, m["border_color"], m)
            if de > C:
                out.append(
                    f"{where}: border {h['border_color']} vs {m['border_color']}, dE00 {de:.2f}"
                )
    elif check == "hairlines":
        if abs(h["bw"][0] - m["bw"][0]) > BORDER_WIDTH_TOL:
            out.append(
                f"{where}: top border {h['bw'][0]} px, the mock's row #{i} has {m['bw'][0]} px"
            )
        elif m["bw"][0] > 0:
            de = _colour_de(h["border_color"], h, m["border_color"], m)
            if h["border_style"] != m["border_style"] or de > C:
                out.append(
                    f"{where}: top border {h['border_style']} {h['border_color']} vs "
                    f"{m['border_style']} {m['border_color']} (dE00 {de:.2f})"
                )
    elif check == "radius":
        if abs(h["radius"] - m["radius"]) > L:
            out.append(f"{where}: radius {h['radius']} vs the mock's {m['radius']}")
    elif check == "pill":
        if h["radius"] + 0.5 < h["h"] / 2:
            out.append(f"{where}: radius {h['radius']} on a {h['h']:.1f} px tall box is not a pill")
    elif check == "shadow":
        if h["shadow"] != m["shadow"]:
            out.append(f"{where}: box-shadow {h['shadow']!r} vs the mock's {m['shadow']!r}")
    elif check == "padding":
        for s in range(4):
            if abs(h["pad"][s] - m["pad"][s]) > L:
                out.append(f"{where}: padding {h['pad']} vs the mock's {m['pad']}")
                break
    elif check in ("font", "font_size"):
        if abs(h["font_size"] - m["font_size"]) > F:
            out.append(f"{where}: font-size {h['font_size']} vs the mock's {m['font_size']}")
        if check == "font":
            if h["font_weight"] != m["font_weight"]:
                out.append(f"{where}: weight {h['font_weight']} vs the mock's {m['font_weight']}")
            rh = h["line_height"] / max(h["font_size"], 1e-9)
            rm = m["line_height"] / max(m["font_size"], 1e-9)
            if abs(rh - rm) > LINE_HEIGHT_RATIO_TOL:
                out.append(f"{where}: line-height ratio {rh:.3f} vs the mock's {rm:.3f}")
    elif check == "color":
        de = _colour_de(h["color"], h, m["color"], m)
        if de > C:
            out.append(f"{where}: colour {h['color']} vs the mock's {m['color']}, dE00 {de:.2f}")
    elif check in ("right", "left"):
        for name, el in (("HUD", h), ("mock", m)):
            g, box = el.get("glyphs"), el.get("parent_box")
            if not g or not box:
                out.append(f"{where}: the {name} side has no glyphs or row box")
                continue
            edge, want = (g[2], box[1]) if check == "right" else (g[0], box[0])
            if abs(edge - want) > L:
                out.append(f"{where}: {name} glyph {check} edge {edge:.1f} vs the row's {want:.1f}")
    else:  # pragma: no cover
        raise AssertionError(f"unknown check {check}")
    return out


#: Every role row holds at group B's tip (see the module docstring), so none carries
#: ``PENDING_T0164_C``: they are guards for group C's tuning of card.css.
_ROLE_ROWS = [pytest.param(r.name, t, id=f"{r.name}-{t}") for r in ROLES for t in THEMES]


@requires_chrome
@pytest.mark.parametrize(("role", "theme"), _ROLE_ROWS)
def test_c1_the_role_matches_option_a(parity, role, theme):
    """C1: one row of the role table per theme; the reference is measured from the committed mock
    in the same Chrome run (T-0154 P-row method). The checks and the deviation notes are in
    ``ROLES``; tolerances: lengths +/- 1 px, font size +/- 0.5 px, line-height ratio +/- 0.02,
    border width +/- 0.25 px, colours dE00 <= 2."""
    _ok(parity)
    r = ROLE_BY_NAME[role]
    hud = parity.hud[theme].roles
    hm, mm = hud[r.name]["matches"], parity.mock[theme].roles[r.name]["matches"]
    assert hm and len(hm) == len(mm), f"{role}: HUD {len(hm)} vs mock {len(mm)} matches"
    problems: list[str] = []
    for i, (h, m) in enumerate(zip(hm, mm, strict=True)):
        for check in r.checks:
            problems += _check(r, check, i, h, m, theme, hud)
    assert problems == [], f"T-0164 C1 {role} (data-theme={theme}; {r.note}):\n  " + "\n  ".join(
        problems
    )


_PROBE_ROWS = [pytest.param(p[0], t, id=f"{p[0]}-{t}") for p in PROBES for t in THEMES]


@requires_chrome
@pytest.mark.parametrize(("probe", "theme"), _PROBE_ROWS)
def test_c1_the_probe_pixel_matches_option_a(parity, probe, theme):
    """C1 "ΔE probes inside the card, the tile, an input and a chip": the HUD's pixel is dE00 <= 3
    from the mock's (A3's pixel tolerance), each landing on its own element. In light, the free
    fills (card, tile, input) are compared with the HUD's own computed fill instead. The "chip
    interior" probe is compared, in both themes, with the H-T184-5 frame at that height (T-0184
    B-H5b, ``FRAME_REFERENCE_PROBES``)."""
    _ok(parity)
    spec = next(p for p in PROBES if p[0] == probe)
    free = spec[5] and theme == "light"
    hp, mp = parity.hud[theme].probes.get(probe), parity.mock[theme].probes.get(probe)
    assert hp and mp, f"{probe}: not probed (HUD {hp}, mock {mp})"
    assert hp["in_el"] and mp["in_el"], f"{probe}: a probe missed its element: HUD {hp}, mock {mp}"
    got = parity.hud[theme].pixels[probe]
    if probe in FRAME_REFERENCE_PROBES:
        # T-0184 B-H5b: the reference is the H-T184-5 frame at that height, on the HUD itself
        assert hp["frame_hit_is_root"], (
            f"{probe}: the frame reference point x={hp['frame_x']} y={hp['y']} is not on the "
            f"card root's own padding: {hp}"
        )
        want = parity.hud[theme].pixels[probe + " @frame"]
        label = "the H-T184-5 frame at that height"
    else:
        want = painted(list(hp["fill"])) if free else parity.mock[theme].pixels[probe]
        label = "the HUD's own fill" if free else "the mock's pixel"
    de = delta_e(got, want)
    assert de <= PIXEL_TOL_DE, (
        f"T-0164 C1 probe {probe} (data-theme={theme}): {hexc(got)} vs {label} {hexc(want)}, "
        f"dE00 {de:.2f} > {PIXEL_TOL_DE}"
    )


# --- C3: style only ----------------------------------------------------------------------------


def _chars(texts: list[str]) -> list[str]:
    return sorted(ch for t in texts for ch in t if not ch.isspace())


def _layers(background_image: str) -> list[str]:
    """The comma-separated layers of a computed ``background-image`` (commas inside parentheses,
    a gradient's colour stops, do not split)."""
    out, depth, cur = [], 0, ""
    for ch in background_image:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [x for x in out if x]


def image_problems(images: list[list[Any]]) -> list[str]:
    """T-0164 C3 ``images``, amended by name by T-0184 B-H5a: the card root (the element that
    carries the H-T184-5 frame) may have exactly one ``background-image``, a
    ``linear-gradient(...)`` with no ``url(``; every descendant still needs ``none``.
    ``images`` holds ``[is_root, element, computed background-image]`` for every element of the
    card whose computed ``background-image`` is not ``none``."""
    out = []
    for is_root, where, value in images:
        if not is_root:
            out.append(f"descendant {where} has background-image {value!r} (must be none)")
            continue
        layers = _layers(value)
        low = value.lower()
        if "url(" in low:
            out.append(f"the card root {where} has a url( in background-image {value!r}")
        elif len(layers) != 1 or not layers[0].lower().startswith("linear-gradient("):
            out.append(
                f"the card root {where} has background-image {value!r}: only one "
                "linear-gradient(...) is allowed (the H-T184-5 frame)"
            )
    return out


def test_c3_premise_image_problems_reads_layers_by_hand_written_cases():
    """The B-H5a checker, on hand-written computed values (no browser): the frame gradient passes;
    a url (alone, or beside the gradient), two layers, another gradient function, and any
    descendant image fail."""
    grad = "linear-gradient(rgb(27, 32, 48), rgb(23, 26, 34))"
    assert _layers(grad) == [grad]
    assert len(_layers(grad + ", " + grad)) == 2
    assert image_problems([]) == []
    assert image_problems([[True, "div[data-a2-id=]", grad]]) == []
    for bad in (
        'url("data:image/gif;base64,R0lGODlhAQABAAAAACw=")',
        grad + ', url("x.png")',
        grad + ", " + grad,
        "radial-gradient(rgb(0, 0, 0), rgb(1, 1, 1))",
        "repeating-linear-gradient(rgb(0, 0, 0), rgb(1, 1, 1) 4px)",
    ):
        assert image_problems([[True, "div[data-a2-id=]", bad]]) != [], bad
    assert image_problems([[False, "div[data-a2-id=h]", grad]]) != []
    assert image_problems([[True, "div[data-a2-id=]", grad], [False, "span[data-a2-id=]", grad]])


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_c3_the_t9_card_has_nine_rows_and_no_checkbox(parity, theme):
    """C3 (adversarial): "the t9 card has 9 rows and 0 checkboxes"."""
    _ok(parity)
    c3 = parity.hud[theme].c3["desk-t9"]
    assert c3["t9_rows"] == T9_ROWS, f"t9: {c3['t9_rows']} list rows, the model wrote {T9_ROWS}"
    assert c3["checkboxes"] == 0, f"t9: {c3['checkboxes']} checkbox(es) the model did not write"


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_c3_the_flight_card_has_five_chips_and_the_labels_verbatim(parity, theme):
    """C3: "the flight card has 5 chips labelled 1-5 and the labels From:, To:, Departure:,
    Return: and Passengers: verbatim" (textContent, byte-equal)."""
    _ok(parity)
    c3 = parity.hud[theme].c3["trial-flight"]
    assert c3["flight_chips"] == FLIGHT_CHIPS, c3["flight_chips"]
    assert c3["flight_labels"] == FLIGHT_LABELS, c3["flight_labels"]
    assert sorted(c3["placeholders"]) == sorted(PLACEHOLDERS["trial-flight"]), c3["placeholders"]


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_c3_no_card_shows_a_character_the_model_did_not_write(parity, theme):
    """C3 (negative, H-T154-6): no eyebrow, tick, step number, bar, total or "+N more". Each of
    the four cards' rendered ``innerText`` (CSS text transforms applied) holds exactly the
    non-whitespace characters of the fixture's ``expected_texts`` (Python oracle), minus the
    placeholder strings; no list marker, progress or meter element; and (``images``, amended by
    name by T-0184 B-H5a) no background image but the card root's one ``linear-gradient(...)``
    frame, with no ``url(`` (``image_problems``)."""
    _ok(parity)
    rows = census()
    problems = []
    for key in PAYLOAD_KEYS:
        c3 = parity.hud[theme].c3[key]
        shown = [t for t in rows[key]["expected_texts"] if t not in PLACEHOLDERS.get(key, [])]
        got, want = _chars([c3["inner_text"]]), _chars(shown)
        if got != want:
            extra = sorted(set(got) - set(want)) or "(counts differ)"
            problems.append(f"{key}: rendered characters differ from the model's: extra {extra}")
        if c3["list_items"]:
            problems.append(f"{key}: list_items {c3['list_items']}")
        problems += [f"{key}: {p}" for p in image_problems(c3["images"])]
        if c3["meters"]:
            problems.append(f"{key}: {c3['meters']} progress/meter element(s)")
    assert problems == [], f"T-0164 C3 (data-theme={theme}):\n  " + "\n  ".join(problems)


@requires_chrome
@pytest.mark.parametrize("arm", list(IMAGE_MUTANTS))
def test_c3_images_adversarial_the_check_catches_a_url_on_the_root_or_an_image_below_it(
    parity, arm
):
    """T-0184 B-H5a (adversarial): with the arm's rule injected into the HUD page, the amended C3
    ``images`` check reports every one of the four payload cards, for the arm's reason (a url( or a
    second layer on the root; any background image on a descendant)."""
    _ok(parity)
    _css, what = IMAGE_MUTANTS[arm]
    c3 = parity.c3_mutants.get(arm)
    assert c3, f"the {arm!r} arm was not measured"
    missed = []
    for key in PAYLOAD_KEYS:
        assert c3.get(key), f"{arm}: card {key} not found with the rule injected"
        found = image_problems(c3[key]["images"])
        if not any(p.startswith(what) for p in found):
            missed.append(f"{key}: {found or 'no problem reported'} (images {c3[key]['images']})")
    assert missed == [], f"B-H5a arm {arm!r} was not caught:\n  " + "\n  ".join(missed)


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_c3_no_pseudo_element_with_content_inside_any_card(parity, theme):
    """C3: "No ::before or ::after with content inside [data-a2ui-root]", over all five cards
    (the role-probe card holds every input kind, so a tick or a chip check drawn in CSS shows)."""
    _ok(parity)
    found = {k: v["pseudo"] for k, v in parity.hud[theme].c3.items() if v and v["pseudo"]}
    missing = [k for k, v in parity.hud[theme].c3.items() if not v]
    assert missing == [], f"cards not found: {missing}"
    assert found == {}, f"T-0164 C3 (data-theme={theme}): generated content: {found}"


# --- C4: layout ---------------------------------------------------------------------------------


def _layout(parity: Parity, view: str) -> dict[str, Any]:
    _ok(parity)
    lay = parity.layout.get(view) or {}
    assert lay.get("cards"), f"{view}: no layout measured: {lay}"
    return lay


@requires_chrome
@pytest.mark.parametrize("view", ["1000x800", "1440x900"])
def test_c4_no_card_overflows_its_card_slot(parity, view):
    """C4: at 1000 px (stacked) and 1440 px, every card's box lies inside its card slot and the
    card does not scroll sideways (the 10 000-character word and the 24-item horizontal List are on
    the role-probe card)."""
    lay = _layout(parity, view)
    problems = []
    for key, c in lay["cards"].items():
        if c is None or c["slot"] is None:
            problems.append(f"{key}: card or slot not found ({c})")
            continue
        (rl, _rt, rr, _rb), (sl, _st, sr, _sb) = c["root"], c["slot"]
        if rl < sl - 0.5 or rr > sr + 0.5:
            problems.append(f"{key}: card x {rl:.1f}..{rr:.1f} outside its slot {sl:.1f}..{sr:.1f}")
        sw, cw = c["root_scroll"]
        if sw > cw + 1:
            problems.append(
                f"{key}: the card scrolls sideways (scrollWidth {sw} > clientWidth {cw})"
            )
    assert problems == [], f"T-0164 C4 at {view}:\n  " + "\n  ".join(problems)


@requires_chrome
@pytest.mark.parametrize("view", ["1000x800", "1440x900"])
def test_c4_a_ten_thousand_character_word_wraps(parity, view):
    """C4: the role-probe card's 10 000-character unbroken word stays inside the card and wraps
    onto many lines (its box is taller than ten line heights)."""
    c = _layout(parity, view)["cards"][SYNTH]
    word, root = c["word"], c["root"]
    assert word, "the long word was not found"
    (wl, wt, wr, wb), line = word["box"], word["line"]
    assert wl >= root[0] - 0.5 and wr <= root[2] + 0.5, (
        f"the word {wl}..{wr} leaves the card {root}"
    )
    assert wb - wt > 10 * line, f"the word is {wb - wt:.0f} px tall at line {line}: it did not wrap"


@requires_chrome
@pytest.mark.parametrize("view", ["1000x800", "1440x900"])
def test_c4_a_horizontal_list_scrolls_inside_the_card(parity, view):
    """C4: the 24-item horizontal List computes overflow-x auto/scroll, overflows its own box
    (premise) and stays inside the card's box."""
    c = _layout(parity, view)["cards"][SYNTH]
    hl, root = c["hl"], c["root"]
    assert hl, "the horizontal List was not found"
    assert hl["overflow_x"] in ("auto", "scroll"), f"overflow-x {hl['overflow_x']}"
    assert hl["scroll"][0] > hl["scroll"][1], (
        f"premise: the List does not overflow ({hl['scroll']})"
    )
    assert hl["box"][0] >= root[0] - 0.5 and hl["box"][2] <= root[2] + 0.5, (hl["box"], root)


@requires_chrome
@pytest.mark.parametrize("view", ["1000x800", "1440x900"])
def test_c4_horizontal_list_items_keep_their_natural_width(parity, view):
    """C4 / ruling 2396676a (a): a horizontal List's items keep their natural width and the list
    scrolls (the mock's one-glyph squeeze is not carried over): each of the 24 items keeps its text
    on one line (a squeezed item wraps it, ``overflow-wrap: anywhere``, so its glyph box shrinks
    with it) and is at least as wide as that text's glyphs (a Range, -0.5 px), and the List
    overflows its box. Red at 313654e6: every item 14 px wide, its text on 5 lines."""
    c = _layout(parity, view)["cards"][SYNTH]
    hl = c["hl"]
    assert hl and len(hl["items"]) == HLIST_ITEMS, f"premise: {HLIST_ITEMS} items ({hl})"
    squeezed = [
        f"{it['text']}: {it['w']:.1f} px, its text {it['glyph_w']} px on {it['lines']} line(s)"
        for it in hl["items"]
        if it["glyph_w"] is None or it["w"] + 0.5 < it["glyph_w"] or it["lines"] != 1
    ]
    assert squeezed == [], f"T-0164 C4 at {view}: squeezed items:\n  " + "\n  ".join(squeezed)
    assert hl["scroll"][0] > hl["scroll"][1], f"the List does not scroll ({hl['scroll']})"


@requires_chrome
@pytest.mark.parametrize("view", ["1100x800", "1440x900"])
def test_c4_the_document_never_scrolls_at_1100_px_and_wider(parity, view):
    """C4 / T-0154 B9 with the five cards in the thread: the document's scroll size equals the
    viewport at >= 1100 px (the thread scrolls inside its pane)."""
    doc = _layout(parity, view)["doc"]
    (sw, sh), (vw, vh) = doc["scroll"], doc["view"]
    assert sw <= vw and sh <= vh, (
        f"the document scrolls at {view}: scroll {sw}x{sh}, view {vw}x{vh}"
    )


# --- sheet origin and loopback (kept) ------------------------------------------------------------


@requires_chrome
def test_c1_the_card_sheet_is_loaded_from_the_pages_own_origin(parity):
    """T-0154 C1 / T-0164 B5 kept: the page itself requested ``/src/ui/card.css`` from the HUD's
    loopback origin, once."""
    _ok(parity)
    hits = [u for u in parity.page_requests if urllib.parse.urlsplit(u).path == "/src/ui/card.css"]
    assert len(hits) == 1, f"card.css requests: {hits}"
    assert hits[0].startswith(parity.hud_origin + "/"), (hits, parity.hud_origin)


@requires_chrome
def test_the_card_session_requests_nothing_off_loopback(parity):
    """T-0128 criterion 2 kept, for the mock page too: every request is on loopback."""
    _ok(parity)
    assert parity.requests, "the request log is empty, so this row would be vacuous"
    offenders = [
        u
        for u in parity.requests
        if urllib.parse.urlsplit(u).scheme in {"http", "https", "ws", "wss"}
        and urllib.parse.urlsplit(u).hostname not in {"127.0.0.1", "localhost", "::1"}
    ]
    assert offenders == [], offenders


# --- C5 / C6 (T-0154) and T-0164 C5: the screenshot script ---------------------------------------

#: T-0154 C5: the turns the UI fixture carries ``ui`` commits for, and the wire surfaces the
#: session fixture carries per turn (hand-counted, ``grep astra.ui.surface``).
FIXTURE_TURNS = ("t3", "t5", "t6", "t8", "t9", "t10")
WIRE_SURFACES = {"t3": 3, "t4": 3, "t5": 3, "t6": 3, "t8": 3, "t9": 3, "t10": 4}


def _fresh_gate(turn: str) -> list[dict[str, Any]]:
    """The spec's recipe for one fixture turn (T-0154 C5): one fresh UiGate, the logged turn_id
    and canvas_idx, final=False, every emitted surface, then the last once more with final=True."""
    gate = UiGate()
    out: list[AstraUiSurface] = []
    for commit in _ui_commits(turn):
        r = gate.gate(
            commit.get("text", ""),
            turn_id=turn,
            canvas_idx=int(commit.get("canvas_idx", 0)),
            final=False,
        )
        if r.surface is not None:
            out.append(r.surface)
    if out:
        out.append(out[-1].model_copy(update={"final": True}))
    return [s.model_dump(mode="json") for s in out]


def test_c5_premise_the_fixtures_hold_the_turns_and_wire_surfaces_counted_by_hand():
    """T-0154 C5 premise kept: the UI fixture's turns are the six named, each makes a surface
    through a fresh gate, and the wire fixture's surfaces per turn are the hand counts."""
    for turn in FIXTURE_TURNS:
        assert _ui_commits(turn), f"{UI_FIXTURE.name} has no ui commits for {turn}"
        assert _fresh_gate(turn), f"a fresh UiGate makes no surface from {turn}'s commits"
    counts: dict[str, int] = {}
    for line in fixture_lines():
        msg = json.loads(line)
        if msg.get("type") == "astra.ui.surface":
            turn = str(msg["surface_id"]).removeprefix("turn-")
            counts[turn] = counts.get(turn, 0) + 1
    assert counts == WIRE_SURFACES, counts


def _as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, str):
        try:
            value = json.loads(message)
        except ValueError:
            return {"__raw__": message}
        return value if isinstance(value, dict) else {"__raw__": message}
    dump = getattr(message, "model_dump", None)
    return dict(dump(mode="json")) if callable(dump) else {"__raw__": repr(message)}


def _surface_view(msg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        msg.get("surface_id"),
        msg.get("canvas_idx"),
        bool(msg.get("final")),
        json.dumps(msg.get("messages"), sort_keys=True),
    )


@dataclass
class Screens:
    error: str = ""
    out: Path | None = None
    stdout: str = ""
    pushed: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture(scope="module")
def screens(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Screens]:
    """``hud_screens.py --out DIR --compare <mockup>`` run once, in this process, with
    ``HudStub.push`` wrapped so the rows see what it pushed (the wrapper delegates)."""
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    result = Screens(out=tmp_path_factory.mktemp("t0164c-shots"))
    spec = importlib.util.spec_from_file_location("_t0164c_hud_screens", SCREENS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original = HudStub.push

    def recording(self: HudStub, message: Any, *args: Any, **kwargs: Any) -> None:
        result.pushed.append(_as_dict(message))
        original(self, message, *args, **kwargs)

    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HudStub, "push", recording)
        try:
            spec.loader.exec_module(module)
            with contextlib.redirect_stdout(buf):
                code = module.main(["--out", str(result.out), "--compare", str(MOCKUP)])
            if code:
                result.error = f"hud_screens.py returned {code}"
        except SystemExit as exc:
            result.error = f"hud_screens.py exited: SystemExit({exc.code!r})"
        except Exception as exc:
            result.error = f"hud_screens.py raised {type(exc).__name__}: {exc}"
    result.stdout = buf.getvalue()
    yield result


def test_c5_the_script_regates_every_fixture_turn_and_keeps_the_other_turns_wire_surfaces(screens):
    """T-0154 C5 kept: for each fixture turn the script pushes exactly the surfaces a fresh
    ``UiGate`` per turn makes (the final copy last) and none of the stale wire surfaces, each after
    the wire replay reached that turn's last message; t4 keeps its three wire surfaces verbatim."""
    assert not screens.error, f"C5: {screens.error}\n{screens.stdout[-2000:]}"
    pushed = screens.pushed
    assert pushed, "the script pushed nothing through HudStub.push"
    wire = [json.loads(line) for line in fixture_lines()]
    surfaces = [(i, p) for i, p in enumerate(pushed) if p.get("type") == "astra.ui.surface"]
    problems: list[str] = []
    for turn in FIXTURE_TURNS:
        sid = f"turn-{turn}"
        got = [_surface_view(p) for _, p in surfaces if p.get("surface_id") == sid]
        want = [_surface_view(s) for s in _fresh_gate(turn)]
        stale = {
            _surface_view(w)
            for w in wire
            if w.get("type") == "astra.ui.surface" and w.get("surface_id") == sid
        } - set(want)
        if got != want:
            problems.append(
                f"{turn}: pushed {[g[:3] for g in got]}, a fresh gate makes {[w[:3] for w in want]}"
            )
        if stale & set(got):
            problems.append(f"{turn}: stale wire surface(s) pushed")
        turn_lines = {
            json.dumps(w, sort_keys=True)
            for w in wire
            if w.get("turn_id") == turn and w.get("type") != "astra.ui.surface"
        }
        last_wire = max(
            (i for i, p in enumerate(pushed) if json.dumps(p, sort_keys=True) in turn_lines),
            default=-1,
        )
        if last_wire < 0:
            problems.append(f"{turn}: none of the turn's wire messages was pushed")
        first_gated = min(
            (i for i, p in surfaces if p.get("surface_id") == sid and _surface_view(p) in want),
            default=None,
        )
        if first_gated is not None and first_gated < last_wire:
            problems.append(f"{turn}: a gated surface was pushed before the replay reached its end")
    t4 = [_surface_view(p) for _, p in surfaces if p.get("surface_id") == "turn-t4"]
    t4_wire = [
        _surface_view(w)
        for w in wire
        if w.get("type") == "astra.ui.surface" and w.get("surface_id") == "turn-t4"
    ]
    if t4 != t4_wire:
        problems.append(f"t4: pushed {len(t4)} surfaces, the wire has {len(t4_wire)}")
    assert problems == [], "T-0154 C5:\n  " + "\n  ".join(problems)


def test_c6_the_script_writes_and_lists_the_card_composites(screens):
    """T-0154 C6 kept ("the existing composites are kept and regenerated", T-0164 C5)."""
    assert not screens.error, f"C6: {screens.error}\n{screens.stdout[-2000:]}"
    assert screens.out is not None
    for theme in THEMES:
        for name in (f"compare-{theme}.png", f"compare-cards-{theme}.png"):
            path = screens.out / name
            assert path.is_file() and path.stat().st_size > 0, f"C6: {name} was not written"
            assert str(path) in screens.stdout, f"C6: {name} is not listed on stdout"
        w, h = _png_size((screens.out / f"compare-cards-{theme}.png").read_bytes())
        assert 0 < h < VIEW[1] and 0 < w < 2 * VIEW[0], (
            f"C6: compare-cards-{theme}.png is {w}x{h}; two cards cropped to the card are smaller "
            f"than two {VIEW[0]}x{VIEW[1]} pages"
        )


#: T-0154 O-B3 (5), kept: every turn-t3 and turn-t8 wire surface lacks ``root`` (pre-T-0127 gate
#: output; hand-checked in the wire fixture); the other turns' wire surfaces have one.
ROOTLESS_WIRE_TURNS = ("t3", "t8")


def _has_root(surface: dict[str, Any]) -> bool:
    return any(
        comp.get("id") == "root"
        for message in surface.get("messages") or []
        for comp in (message.get("updateComponents") or {}).get("components") or []
    )


def test_c5_premise_the_rootless_wire_turns_are_t3_and_t8_and_the_gate_gives_them_a_root():
    """T-0154 C5 premise kept: the wire's t3 and t8 surfaces have no ``root``, and a fresh gate
    gives every fixture turn's surfaces one."""
    wire = [json.loads(line) for line in fixture_lines()]
    rootless = sorted(
        {
            str(w["surface_id"]).removeprefix("turn-")
            for w in wire
            if w.get("type") == "astra.ui.surface" and not _has_root(w)
        }
    )
    assert rootless == sorted(ROOTLESS_WIRE_TURNS), rootless
    for turn in FIXTURE_TURNS:
        gated = _fresh_gate(turn)
        assert gated and all(_has_root(s) for s in gated), (
            f"a fresh gate's {turn} surface lacks root"
        )


def test_c5_no_fixture_turn_is_left_on_a_rootless_surface_the_t8_placeholder(screens):
    """T-0154 C5 / O-B3 (5) kept, adversarial: the last surface the script pushes for each fixture
    turn carries a ``root`` component, so no screenshot card is an empty frame."""
    assert not screens.error, f"C5: {screens.error}\n{screens.stdout[-2000:]}"
    last: dict[str, dict[str, Any]] = {}
    for p in screens.pushed:
        if p.get("type") == "astra.ui.surface":
            last[str(p.get("surface_id"))] = p
    problems = []
    for turn in FIXTURE_TURNS:
        p = last.get(f"turn-{turn}")
        if p is None:
            problems.append(f"{turn}: no surface pushed at all")
        elif not _has_root(p):
            problems.append(f"{turn}: the last surface pushed has no root component")
    assert problems == [], "T-0154 C5:\n  " + "\n  ".join(problems)


def test_t0164_c5_the_script_writes_the_genui_composites_cropped_to_the_cards(screens):
    """T-0164 C5: ``compare-genui-{dark,light}.png`` are written and listed on stdout: the mock's
    option-A card beside the HUD's card for each of the four payloads, cropped to the cards — so
    narrower than two 1440 px pages, and decodable as a PNG with at least four card rows' worth of
    height (four payloads, each card at least 100 px tall)."""
    assert not screens.error, f"C5: {screens.error}\n{screens.stdout[-2000:]}"
    assert screens.out is not None
    for theme in THEMES:
        path = screens.out / f"compare-genui-{theme}.png"
        assert path.is_file() and path.stat().st_size > 0, f"C5: {path.name} was not written"
        assert str(path) in screens.stdout, f"C5: {path.name} is not listed on stdout"
        w, h = _png_size(path.read_bytes())
        assert 0 < w < 2 * VIEW[0] and h >= 4 * 100, f"C5: {path.name} is {w}x{h}"
        _w, _h, rows = decode_png(path.read_bytes())
        assert len(rows) == h


# decode_png and base64 stay imported for the C5 row and for test_hud_ui_surface's imports.
_ = base64
