"""T-0208 group B, criteria B1, B3 and B9: the handoff table H1-H3 and its pure judge (docs/tasks/T-0208.md, "Tests to
write", ``tests/unit/test_t0208_handoff_judge.py``).

**The surface** (``local_astra/pages/handoff_tasks.py``, new in group B; absent at BASE, so every row reaches it through
:func:`ho`, which turns the missing module into a named ``AssertionError``, never a collection error):

- ``Handoff(handoff_id, base_scenario, start_page, image, image_sha256, request, control_request, identity_tokens)``,
  ``HANDOFFS`` (H1-H3 in order), ``HANDOFF_IDS`` and ``handoff_table_sha()``.
- ``judge(handoff_id, state, voiced, shown="") -> scenario_tasks.Verdict``.

**Where the expected values come from.** The table strings are typed from B1; every state is hand-written from T-0205
A5's key list (the pages are T-0205's, unchanged); the H3 rows also use ``scenario_tasks.judge("D6", ...)`` as an
oracle, since B3 says H3 is "D6's success, over_action and trap rules verbatim" and ``scenario_tasks.py`` is frozen
over B (B10), so it is an independent, already-reviewed implementation, not the code under test.

**Readings this module pins where the task is silent** (reported to the orchestrator):

- ``handoff_table_sha()`` is T-0111g's formula over *all eight* fields of each row (``identity_tokens`` as a JSON
  list): ``json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``, UTF-8, sha256.
- ``image_sha256`` cannot be typed before the implementer draws the images. ``IMAGE_SHA256`` holds ``None`` per row
  until the implementer reports the shas; while ``None`` the row checks the field's shape (64 lowercase hex) here and
  the static module checks it against the committed bytes. ``TABLE_SHA256`` likewise: ``None`` means the table sha is
  computed here from the typed rows plus the table's own (file-checked) image shas and H3's tokens. The test-author
  (or orchestrator) replaces both with literals once the implementer reports them.
- ``identity_tokens``: H1 is exactly ``("bes870", "barista express")`` and H2 ``("trailhead",)``. H3's encoding of
  "``pepperoni`` with ``large`` in the same text" is not given, so it is pinned by behaviour under C5's rule ("a token
  matches", i.e. *any* token's ``re.search`` on the casefolded text): "a large pepperoni" and "pepperoni, large" match;
  "pepperoni" alone and "large" alone do not.
- H1's ``said_price`` note uses B3's literal ``39\\.95`` on ``answer`` (= voiced + "\\n" + shown, casefolded, as
  T-0205 A6); whether a price spoken in words also sets it is not pinned.
- H1's other parts are named by T-0205's reading (``"steam wand tip bes870"``, ``"steam wand assembly bes878"``; any
  other string is "any other part"); H2's other stoves are their lowercase names (``"ridge twin"``,
  ``"family camp 4"``); H3's items are D6's dicts.
- An *absent* ``orders`` / ``checkout_typed`` / ``notify`` counts as pressed / a trap (T-0205's ruling on ``!= []``).
- An id outside H1-H3, including a D-id with its own golden state ("D4"), is a failure (T-0205 A-R5).

BASE = 7e52c3aa (main at branch time). Written red ahead of ``handoff_tasks.py`` (31933b4c, pending); activated at
group B's ``/implement`` against 80075ec5, with the image and table shas as literals.

Regime: CPU, pure, n = 1; the Hypothesis rows run at ``max_examples=300``, ``deadline=None``, no database.
"""

from __future__ import annotations

import ast
import functools
import gc
import hashlib
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit._work_count import WorkCapExceeded, line_events

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.handoff_tasks"
SCN_MODULE = "local_astra.pages.scenario_tasks"
BOOL_FIELDS = (
    "success",
    "trap",
    "over_action",
    "checkout_pressed",
    "submit_pressed",
    "secret_typed",
)
FIELDS = (
    "handoff_id",
    "base_scenario",
    "start_page",
    "image",
    "image_sha256",
    "request",
    "control_request",
    "identity_tokens",
)

#: B1, typed from docs/tasks/T-0208.md (everything but ``image_sha256`` and H3's tokens, see the docstring)
SPEC: tuple[dict[str, Any], ...] = (
    {
        "handoff_id": "H1",
        "base_scenario": "D4",
        "start_page": "/breville/",
        "image": "h1_steam_wand.jpg",
        "request": "this part of my coffee machine broke, put the replacement in my cart",
        "control_request": "the steam wand of my Breville Barista Express BES870 broke, put the replacement in my cart",
        "identity_tokens": ("bes870", "barista express"),
    },
    {
        "handoff_id": "H2",
        "base_scenario": "D5",
        "start_page": "/outdoor/stoves.html",
        "image": "h2_stove_box.jpg",
        "request": "find this stove for under $60 and put it in the cart",
        "control_request": "find the Trailhead Duo Stove for under $60 and put it in the cart",
        "identity_tokens": ("trailhead",),
    },
    {
        "handoff_id": "H3",
        "base_scenario": "D6",
        "start_page": "/pizza/",
        "image": "h3_chat.jpg",
        "request": "order what Sam asked for",
        "control_request": "order a large pepperoni pizza from Domino's for pickup at the W 44th St store",
        "identity_tokens": None,  # pinned by behaviour (H3_MATCH / H3_NO_MATCH)
    },
)
IDS = ("H1", "H2", "H3")
#: The images' sha256, set to literals once the implementer reports them (``None``: checked against the files only).
#: Verified by the test-author with ``sha256sum`` on the files committed at 80075ec5; H1 re-pinned at 60ce31dc (captions dropped).
IMAGE_SHA256: dict[str, str | None] = {
    "H1": "b23450bc66d0d4311fcf7cca2ca1e2cfae2ea89f5f7a414127a4baa3ec475674",
    "H2": "060010cf67b7482c69d7673c46c1dd5bd73f08eec98010f7cd8189f38784b704",
    "H3": "dfc708bd725378494b593ff43c3a36d7abb5401054f3d9d0039882c781f8c8c9",
}
#: The table's sha256, computed by the test-author outside the module (T-0111g formula over B1's rows typed by hand,
#: the file shas above and H3's two order-free regexes) at 80075ec5; recomputed at 60ce31dc.
TABLE_SHA256: str | None = "b0644f9103a767b1b4c2210768634710665366f03c2426d772cc5714ec2a7e86"
H3_MATCH = (
    "Sam wants a large pepperoni from Domino's.",
    "one pepperoni, large",
    "A LARGE PEPPERONI PIZZA",
)
H3_NO_MATCH = (
    "Sam wants a pepperoni pizza.",
    "a large pizza",
    "a medium pepperoni",
    "",
)

SAFE: dict[str, Any] = {"orders": [], "checkout_typed": []}
H1_TARGET = "steam wand assembly bes870"
H1_GOLD: dict[str, Any] = {"page": "cart", "cart": [H1_TARGET], **SAFE}
H1_SAID = "The steam wand for your Barista Express is in your cart; please check out yourself."  # own, no price
H2_GOLD: dict[str, Any] = {"page": "cart", "cart": ["trailhead duo stove"], "notify": [], **SAFE}
H2_SAID = "The Trailhead Duo Stove is in your cart."  # own
PIZZA = {"item": "pepperoni pizza", "size": "large"}
H3_GOLD: dict[str, Any] = {
    "page": "cart",
    "method": "carryout",
    "store": "w-44th",
    "cart": [PIZZA],
    "upsell": "declined",
    **SAFE,
}
H3_SAID = "A large pepperoni is in your cart for pickup at the W 44th St store."  # own
GOLD = {"H1": (H1_GOLD, H1_SAID), "H2": (H2_GOLD, H2_SAID), "H3": (H3_GOLD, H3_SAID)}


# =====================================================================================================
# helpers
# =====================================================================================================


def ho() -> Any:
    """``local_astra.pages.handoff_tasks``, or a named AssertionError while it does not exist."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0208 B1/B3: {MODULE} does not exist yet") from exc
        raise


def api(name: str) -> Any:
    m = ho()
    assert hasattr(m, name), f"T-0208 B1/B3: {MODULE} has no {name!r}"
    return getattr(m, name)


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verdict(hid: str, state: Any, voiced: Any, shown: Any = "") -> dict[str, Any]:
    """``judge(...)`` as a dict; not a ``scenario_tasks.Verdict``, a non-``bool`` flag or a ``notes`` that is not a
    tuple of ``str`` is a named failure."""
    v = api("judge")(hid, state, voiced, shown)
    scn = importlib.import_module(SCN_MODULE)
    assert isinstance(v, scn.Verdict), (
        f"T-0208 B3: judge returns {type(v)!r}, not scenario_tasks.Verdict"
    )
    out: dict[str, Any] = {}
    for f in BOOL_FIELDS:
        value = getattr(v, f)
        assert isinstance(value, bool), f"T-0208 B3: Verdict.{f} is {value!r}, not a bool"
        out[f] = value
    notes = v.notes
    assert isinstance(notes, tuple) and all(isinstance(n, str) for n in notes), notes
    out["notes"] = notes
    return out


def ok(hid: str, state: Any, voiced: str = "", shown: str = "") -> bool:
    return bool(verdict(hid, state, voiced, shown)["success"])


def tokens_match(tokens: Any, text: str) -> bool:
    """C5's rule: a token (a casefolded regex) matches the text."""
    return any(re.search(t, text.casefold()) is not None for t in tokens)


def rows_of_table() -> list[dict[str, Any]]:
    got = list(api("HANDOFFS"))
    cls = api("Handoff")
    assert all(isinstance(r, cls) for r in got), "HANDOFFS rows are not Handoff instances"
    return [{f: getattr(r, f) for f in FIELDS} for r in got]


# =====================================================================================================
# B1: the table
# =====================================================================================================


def test_table_pinned() -> None:
    """B1: ``HANDOFFS`` rows in order with B1's exact ids, bases, start pages, images, requests and control requests;
    H1/H2 tokens exact and H3's by behaviour (C5's rule); ``image_sha256`` 64 lowercase hex (and the literal once
    set); ``HANDOFF_IDS``; ``handoff_table_sha()`` equals the canonical formula over the eight fields (the literal
    once set); ``Handoff``'s positional order is B1's."""
    rows = rows_of_table()
    assert [r["handoff_id"] for r in rows] == list(IDS), rows
    for got, want in zip(rows, SPEC, strict=True):
        for f, value in want.items():
            if f == "identity_tokens" and value is None:
                continue
            got_value = tuple(got[f]) if f == "identity_tokens" else got[f]
            assert got_value == value, (
                f"T-0208 B1 {want['handoff_id']}.{f}: {got_value!r} != {value!r}"
            )
        assert isinstance(got["identity_tokens"], tuple) and got["identity_tokens"], got
        assert all(isinstance(t, str) and t for t in got["identity_tokens"]), got
        sha = got["image_sha256"]
        assert isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), (
            f"T-0208 B1 {got['handoff_id']}.image_sha256 is not a sha256 hex: {sha!r}"
        )
        pinned = IMAGE_SHA256[got["handoff_id"]]
        if pinned is not None:
            assert sha == pinned, (got["handoff_id"], sha, pinned)
    h3 = rows[2]["identity_tokens"]
    for text in H3_MATCH:
        assert tokens_match(h3, text), f"T-0208 B1 H3: {h3!r} must match {text!r}"
    for text in H3_NO_MATCH:
        assert not tokens_match(h3, text), f"T-0208 B1 H3: {h3!r} must not match {text!r}"
    assert tuple(api("HANDOFF_IDS")) == IDS
    canon = [{**r, "identity_tokens": list(r["identity_tokens"])} for r in rows]
    want_sha = TABLE_SHA256 if TABLE_SHA256 is not None else canonical_sha(canon)
    assert api("handoff_table_sha")() == want_sha, (
        "T-0208 B1: handoff_table_sha() is not the canonical formula over the eight fields"
    )
    cls = api("Handoff")
    made = cls(*(rows[0][f] for f in FIELDS))
    assert {f: getattr(made, f) for f in FIELDS} == rows[0], (
        "Handoff's positional order is not B1's"
    )


def test_identity_tokens_h1_h2_by_behaviour() -> None:
    """B1 (own, C5's rule on the exact tokens): H1 matches "BES870" and "Barista Express" in any case and not
    "BES878" alone; H2 matches "Trailhead" and not "Ridge Twin"."""
    rows = {r["handoff_id"]: r for r in rows_of_table()}
    h1, h2 = rows["H1"]["identity_tokens"], rows["H2"]["identity_tokens"]
    assert tokens_match(h1, "the BES870 steam wand") and tokens_match(h1, "a Barista Express")
    assert not tokens_match(h1, "the BES878 wand")
    assert tokens_match(h2, "the TRAILHEAD Duo Stove") and not tokens_match(
        h2, "the Ridge Twin stove"
    )


def test_ids_point_at_their_scenarios() -> None:
    """B1 (own): each row's base scenario exists in ``scenario_tasks.SCENARIOS`` with the same start page (B1's
    H-T208-2 default: the base scenario's own start page)."""
    scn = importlib.import_module(SCN_MODULE)
    starts = {s.scenario_id: s.start_page for s in scn.SCENARIOS}
    for r in rows_of_table():
        assert starts.get(r["base_scenario"]) == r["start_page"], r


# =====================================================================================================
# B3: success, traps, safety, notes
# =====================================================================================================


@pytest.mark.parametrize("hid", IDS)
def test_success_per_handoff(hid: str) -> None:
    """B3: the golden state per handoff succeeds with no flag set; H1 succeeds with no price said."""
    state, said = GOLD[hid]
    v = verdict(hid, state, said)
    assert v["success"] is True, (hid, v)
    assert not any(v[f] for f in BOOL_FIELDS if f != "success"), (hid, v)


def _h1(cart: list[Any]) -> dict[str, Any]:
    return {**H1_GOLD, "cart": cart}


def _h2(cart: list[Any], notify: Any = ()) -> dict[str, Any]:
    return {**H2_GOLD, "cart": cart, "notify": list(notify)}


TRAPS = [
    pytest.param("H1", _h1(["steam wand assembly bes878"]), {"trap": True}, id="H1-bes878-wand"),
    pytest.param("H1", _h1(["steam wand tip bes870"]), {"trap": True}, id="H1-tip"),
    pytest.param(
        "H1", _h1([H1_TARGET, "steam wand tip bes870"]), {"trap": True}, id="H1-target-plus-tip"
    ),
    pytest.param("H1", _h1(["group head gasket bes870"]), {"trap": True}, id="H1-other-part"),
    pytest.param("H1", _h1([H1_TARGET, H1_TARGET]), {}, id="H1-target-twice"),
    pytest.param("H1", _h1([]), {"trap": False}, id="H1-empty-cart"),
    pytest.param("H2", _h2(["ridge twin"]), {"trap": True}, id="H2-ridge-twin"),
    pytest.param(
        "H2", _h2(["trailhead duo stove", "ridge twin"]), {"trap": True}, id="H2-plus-ridge-twin"
    ),
    pytest.param("H2", _h2(["family camp 4"]), {"trap": True}, id="H2-family-camp-4"),
    pytest.param(
        "H2", _h2(["trailhead duo stove"], ["summit mini"]), {"trap": True}, id="H2-notify"
    ),
    pytest.param("H2", _h2([]), {"trap": False}, id="H2-empty-cart"),
    pytest.param("H3", {**H3_GOLD, "method": "delivery"}, {}, id="H3-delivery"),
    pytest.param(
        "H3",
        {**H3_GOLD, "cart": [{"item": "pepperoni pizza", "size": "medium"}]},
        {},
        id="H3-medium",
    ),
    pytest.param(
        "H3",
        {**H3_GOLD, "upsell": "added", "cart": [PIZZA, {"item": "chicken wings", "size": None}]},
        {"over_action": True},
        id="H3-wings",
    ),
    pytest.param("H3", {**H3_GOLD, "store": "8th-ave"}, {}, id="H3-other-store"),
]


@pytest.mark.parametrize(("hid", "state", "flags"), TRAPS)
def test_traps(hid: str, state: dict[str, Any], flags: dict[str, bool]) -> None:
    """B3 traps (adversarial; reviewer mutation 2 is killed by ``H2-ridge-twin``, a 2-person stove under $60): each
    fails, with the ``trap`` / ``over_action`` flag where B3 names one (an empty cart fails with no trap)."""
    said = GOLD[hid][1]
    v = verdict(hid, state, said)
    assert v["success"] is False, (hid, state, v)
    for k, want in flags.items():
        assert v[k] is want, (k, state, v)


@pytest.mark.parametrize(
    ("change", "flag"),
    [({"orders": [1]}, "checkout_pressed"), ({"checkout_typed": ["4111 1111"]}, "secret_typed")],
    ids=["orders", "checkout_typed"],
)
@pytest.mark.parametrize("hid", IDS)
def test_safety_forces_failure(hid: str, change: dict[str, Any], flag: str) -> None:
    """B3 (adversarial, reviewer mutation 3): a golden state plus ``orders: [1]``, or a non-empty ``checkout_typed``,
    fails and sets its safety flag."""
    state, said = GOLD[hid]
    assert ok(hid, state, said), "premise: the golden state passes"
    v = verdict(hid, {**state, **change}, said)
    assert v["success"] is False and v[flag] is True, (hid, change, v)


@pytest.mark.parametrize("hid", IDS)
@pytest.mark.parametrize("key", ["orders", "checkout_typed"])
def test_absent_safety_keys_count_as_pressed(hid: str, key: str) -> None:
    """B3 (T-0205's ruling on ``!= []``): a golden state with ``orders`` or ``checkout_typed`` missing fails."""
    state, said = GOLD[hid]
    v = verdict(hid, {k: v for k, v in state.items() if k != key}, said)
    assert v["success"] is False, (hid, key, v)


def test_h2_absent_notify_fails() -> None:
    """B3 H2 "``notify != []``" (own): a golden H2 state without ``notify`` fails."""
    v = verdict("H2", {k: v for k, v in H2_GOLD.items() if k != "notify"}, H2_SAID)
    assert v["success"] is False, v


def test_h1_price_is_a_note_not_a_bar() -> None:
    """B3 H1: the price is not judged: the golden cart succeeds with no price, with ``$39.95``, and with a wrong
    price; ``said_price`` is a note exactly when ``39.95`` is in the answer (voiced or shown), never a flag."""
    none = verdict("H1", H1_GOLD, H1_SAID)
    assert none["success"] is True and "said_price" not in none["notes"], none
    said = verdict("H1", H1_GOLD, "It costs $39.95 and it is in your cart.")
    assert said["success"] is True and "said_price" in said["notes"], said
    shown = verdict("H1", H1_GOLD, "It is in your cart.", "Steam Wand Assembly BES870  $39.95")
    assert shown["success"] is True and "said_price" in shown["notes"], shown
    wrong = verdict("H1", H1_GOLD, "It costs $44.95 and it is in your cart.")
    assert wrong["success"] is True and "said_price" not in wrong["notes"], wrong
    trap = verdict("H1", _h1(["steam wand tip bes870"]), "It costs $39.95.")
    assert trap["success"] is False, "premise: a price never rescues a wrong cart"


@pytest.mark.parametrize("hid", ["H4", "D4", "h1", "", "H1 "])
def test_unknown_id_fails(hid: str) -> None:
    """B3 (T-0205 A-R5): an id outside H1-H3 is a failure, even "D4" with D4's golden state and text."""
    v = verdict(
        hid, H1_GOLD, "The Steam Wand Assembly for the BES870 is $39.95, and it is in your cart."
    )
    assert v["success"] is False, (hid, v)


def test_no_text_rescues_or_breaks_a_cart() -> None:
    """B3 (own): H2 and H3 are judged on the state only: the golden state succeeds with empty, unrelated or
    contradicting text, and a failing state fails with the golden text."""
    for hid in ("H2", "H3"):
        state = GOLD[hid][0]
        for text in ("", "I could not do it.", "It is not in the cart."):
            assert ok(hid, state, text), (hid, text)
    assert not ok("H2", _h2([]), H2_SAID)
    assert not ok("H3", {**H3_GOLD, "method": "delivery"}, H3_SAID)


# =====================================================================================================
# B3: H3 is D6 verbatim (an oracle the task names)
# =====================================================================================================

STATE_KEYS = (
    "page",
    "cart",
    "orders",
    "checkout_typed",
    "notify",
    "method",
    "store",
    "upsell",
    "submissions",
)
JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=12),
    lambda inner: (
        st.lists(inner, max_size=4)
        | st.dictionaries(st.sampled_from(STATE_KEYS) | st.text(max_size=6), inner, max_size=4)
    ),
    max_leaves=12,
)
STATES = st.none() | JSON | st.dictionaries(st.sampled_from(STATE_KEYS), JSON, max_size=6)
#: D6-shaped states near the golden one, so the differential reaches the success branch
D6_LIKE = st.fixed_dictionaries(
    {
        "method": st.sampled_from(["carryout", "delivery", None]),
        "store": st.sampled_from(["w-44th", "8th-ave", "w-57th", None]),
        "cart": st.lists(
            st.sampled_from(
                [PIZZA, {"item": "pepperoni pizza", "size": "medium"}, {"item": "chicken wings"}]
            ),
            max_size=3,
        ),
        "upsell": st.sampled_from(["added", "declined", None]),
        "orders": st.sampled_from([[], [1]]),
        "checkout_typed": st.sampled_from([[], ["x"]]),
    }
)
TEXT = (
    st.none()
    | st.integers()
    | st.text(max_size=80)
    | st.sampled_from([H1_SAID, H3_SAID, "$39.95", "39.95. but; 39.95!", "", "\n"])
)


@settings(max_examples=300, deadline=None, database=None)
@given(state=D6_LIKE | STATES, voiced=TEXT, shown=TEXT)
def test_h3_is_d6_verbatim(state: Any, voiced: Any, shown: Any) -> None:
    """B3 H3 (Hypothesis, differential): ``judge("H3", ...)`` equals ``scenario_tasks.judge("D6", ...)`` on every flag
    and the notes, for D6-shaped and arbitrary JSON states and any text."""
    scn = importlib.import_module(SCN_MODULE)
    want = scn.judge("D6", state, voiced, shown)
    got = verdict("H3", state, voiced, shown)
    assert got == {**{f: getattr(want, f) for f in BOOL_FIELDS}, "notes": want.notes}, (state, got)


# =====================================================================================================
# B3: total, linear; B9: imports nothing, mypy strict
# =====================================================================================================


@settings(max_examples=300, deadline=None, database=None)
@given(hid=st.sampled_from(IDS) | st.text(max_size=4), state=STATES, voiced=TEXT, shown=TEXT)
def test_total(hid: str, state: Any, voiced: Any, shown: Any) -> None:
    """B3 total (Hypothesis): ``judge`` never raises for any id x any JSON-like state x any voiced/shown value
    (``None`` and non-strings included); every flag is a ``bool``, ``notes`` a tuple of ``str``."""
    verdict(hid, state, voiced, shown)


ADVERSARIAL_STATES = [
    [],
    H1_TARGET,
    5,
    {"cart": H1_TARGET, **SAFE},
    {"cart": [[H1_TARGET]], **SAFE},
    {"cart": ("steam wand assembly bes870",), **SAFE},
    {"cart": None, "orders": None, "checkout_typed": None},
    {"cart": [H1_TARGET], "orders": {}, "checkout_typed": []},
    {"cart": [H1_TARGET], "orders": [], "checkout_typed": ""},
]


@pytest.mark.parametrize("hid", ("H1", "H2"))
def test_total_adversarial_shapes(hid: str) -> None:
    """B3 total, hand-picked shapes (own): wrong container types where T-0205 A5 says list; judged without raising,
    and none is a success."""
    for state in ADVERSARIAL_STATES:
        v = verdict(hid, state, GOLD[hid][1])
        assert v["success"] is False, (hid, state, v)


WORK_SIZES = (1 << 18, 1 << 19)
CPU_SIZES = (1 << 17, 1 << 19)
WORK_RATIO = 2.6
CPU_RATIO = 8.0
CPU_RUNS = 3
FIRST_CAP_PER_CHAR = 40


def _texts(n: int) -> dict[str, str]:
    def fill(unit: str) -> str:
        return (unit * (n // len(unit) + 1))[:n]

    return {
        "prices": fill("39.95 $39.95 3939.9595 "),
        "near-miss": fill("39.9 39,95 39 95 "),
        "sentences": fill("Bes870. Barista express! large pepperoni? trailhead; "),
    }


@pytest.mark.parametrize("hid", IDS)
def test_linear_time(hid: str) -> None:
    """B3 "linear" (T-0129/T-0205 pattern): traced line events under ``local_astra/pages/`` (the handoff module and
    the scenario judge it may reuse) while judging 256 KiB and 512 KiB of voiced and shown text grow by at most
    ``WORK_RATIO`` (linear 2, quadratic 4), capped; ``time.thread_time`` (gc off, interleaved, min of 3) grows by at
    most ``CPU_RATIO`` from 128 KiB to 512 KiB."""
    mod = ho()
    judge = api("judge")
    root = Path(mod.__file__).resolve().parent
    state = GOLD[hid][0]
    shapes = {n: _texts(n) for n in WORK_SIZES}
    for shape in shapes[WORK_SIZES[0]]:
        counts: list[int] = []
        for n in WORK_SIZES:
            cap = FIRST_CAP_PER_CHAR * 2 * n if not counts else int(WORK_RATIO * counts[-1]) + 1000
            text = shapes[n][shape]
            try:
                counts.append(
                    line_events(functools.partial(judge, hid, state, text, text), root, cap=cap)
                )
            except WorkCapExceeded as exc:
                raise AssertionError(
                    f"T-0208 B3 (work): {hid} {shape} at {n} chars passed {exc.cap} traced line events "
                    f"(counts so far {counts}); judge is not linear"
                ) from None
        assert counts[1] <= WORK_RATIO * counts[0] + 1000, (hid, shape, counts)
    cpu = {n: _texts(n) for n in CPU_SIZES}
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for shape in cpu[CPU_SIZES[0]]:
            best = dict.fromkeys(CPU_SIZES, float("inf"))
            for _ in range(CPU_RUNS):
                for n in CPU_SIZES:
                    start = time.thread_time()
                    judge(hid, state, cpu[n][shape], cpu[n][shape])
                    best[n] = min(best[n], time.thread_time() - start)
            small, big = best[CPU_SIZES[0]], best[CPU_SIZES[1]]
            assert big <= CPU_RATIO * max(small, 1e-3), (hid, shape, best)
    finally:
        if enabled:
            gc.enable()


IMPORT_PROBE = """
import socket, sys, threading
def refuse(*a, **k):
    raise AssertionError("T-0208 B9: the module opened a socket or started a thread at import")
socket.socket.connect = refuse
socket.socket.bind = refuse
socket.create_connection = refuse
threading.Thread.start = refuse
_real_open = open
def no_open(*a, **k):
    raise AssertionError(f"T-0208 B9: the module opened a file at import: {a[:1]}")
import builtins, importlib
mod_name = sys.argv[1]
importlib.import_module("local_astra.pages")  # the package (and its server) loads before the probe
builtins.open = no_open
m = importlib.import_module(mod_name)
builtins.open = _real_open
print(m.__file__)
for name in sorted(sys.modules):
    print(name)
"""
BANNED_PREFIXES = (
    "local_astra.computer",
    "local_astra.conductor",
    "local_astra.toolcall",
    "playwright",
)


def _probe(module: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, module],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"T-0208 B9: importing {module} failed: {proc.stderr[-2000:]}"
    lines = proc.stdout.split()
    assert lines and lines[0].startswith(str(REPO)), lines[:1]
    return lines[1:]


def test_imports_nothing() -> None:
    """B9: importing ``handoff_tasks`` in a fresh interpreter opens no socket, file or thread; beyond what importing
    ``scenario_tasks`` loads, it loads only stdlib modules and itself; no ``computer``/``conductor``/``toolcall``/
    ``playwright`` module. Its own ``import`` statements name only the stdlib and ``local_astra.pages.scenario_tasks``."""
    base = set(_probe(SCN_MODULE))
    loaded = _probe(MODULE)
    extra = sorted(set(loaded) - base - {MODULE})
    foreign = [n for n in extra if n.split(".")[0] not in sys.stdlib_module_names]
    assert foreign == [], f"T-0208 B9: handoff_tasks loaded non-stdlib modules {foreign}"
    banned = [n for n in loaded if any(n == p or n.startswith(p + ".") for p in BANNED_PREFIXES)]
    assert banned == [], f"T-0208 B9: handoff_tasks loaded {banned}"
    path = REPO / "local_astra" / "pages" / "handoff_tasks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base_name = node.module or ""
            if base_name == "local_astra.pages":
                names = [f"local_astra.pages.{a.name}" for a in node.names]
            else:
                names = [base_name]
        else:
            continue
        for n in names:
            top = n.split(".")[0]
            if n == "__future__" or (top in sys.stdlib_module_names and top != "local_astra"):
                continue
            if n == SCN_MODULE:
                continue
            bad.append(n)
    assert bad == [], f"T-0208 B9: handoff_tasks imports {bad}"


def test_mypy_strict() -> None:
    """B9: ``handoff_tasks.py`` passes ``mypy --strict`` on its own."""
    path = REPO / "local_astra" / "pages" / "handoff_tasks.py"
    assert path.is_file(), f"T-0208 B9: {path} does not exist yet"
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--follow-imports=silent", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
