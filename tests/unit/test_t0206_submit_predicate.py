"""T-0206 A1 (with the A5/A6 rows as hand-built facts): ``fence.irreversible_submit(facts) -> bool``
(docs/tasks/T-0206.md, "Tests to write", ``tests/unit/test_t0206_submit_predicate.py``).

A1, as written: ``True`` exactly when (a) the control is not a navigating link (``href_nav`` false), (b) the scope is
not search-like (no ``search_scope``, no *filled* field of ``type=search``, no field whose ``name``/``id`` is in
``SEARCH_NAMES``), (c) at least one personal field is ``filled`` and ``visible`` (a textarea; ``type`` email/tel; an
``autocomplete`` token in ``PERSONAL_AUTOCOMPLETE``; a normalised ``name``/``id``/``label``/``placeholder`` matching
whole-word ``name|email|e-mail|phone|tel|mobile|address|street|zip|postcode|postal|message|comments?``; never a
search-like field), and (d) the control's normalised label begins on a word boundary with a ``SEND_LABELS`` entry, or
the control is ``structural`` with an empty label. ``SEARCH_LABELS`` win over ``SEND_LABELS``.

Every expected value below is worked out by hand from that text; the comment beside each row names the clause that
decides it. None is captured from an implementation.

**Readings (the test-author's, reported, not guessed silently):**

- "``SEARCH_LABELS`` win" is pinned only where the label *begins* with a search label ("Search", "Go", "Find
  stores", "Search flights", "Show results"); a label that begins with a send label and contains a search word later
  ("Send search") is not pinned.
- "at most 6 words" is not pinned (truncation of a longer label, or a longer label not counting, are both readings;
  a "begins with" match is the same under truncation, so only a >6-word label could tell them apart).
- Punctuation *inside* a word ("Submit-request", "first_name") is not pinned (replaced by a space or removed are both
  readings); only trailing punctuation and arrows, case and runs of whitespace are.
- An ``autocomplete`` value is split on whitespace into tokens ("shipping postal-code" holds ``postal-code``); pinned.
- The adversarial ``email-search`` row follows the planner's reading (the field is ``type=text``: a ``type=search``
  field that is filled is clause (b) and would be ``False``).
- Formless sends (orchestrator ruling on reading 6, 2026-10-30, docs/tasks/T-0206.md Status): the scope of a formless
  control is its nearest ancestor holding a text-like input (``scope`` is not ``"form"``; written ``"document"`` here,
  A1's only other value), and it is a search only if that scope's *filled* inputs are all search-like; filled personal
  fields there make it a send. So an empty site-search ``q`` box in that scope does not exempt it. A form-bound send
  keeps form scope: there any field named in ``SEARCH_NAMES`` still makes (b) fail.
- Review rulings R-2/R-3 (docs/tasks/T-0206.md Status, d1dab0d1): a search-named field exempts a form only when it
  is filled (and visible), or when all filled fields are search-like; a form whose filled personal fields are all
  location fields (address, street, city, postcode, zip, country) is a locator search. The rows pinned before those
  rulings are re-pinned here (``test_search_names``, the location rows moved out of ``PERSONAL``).
- Reviewer mutation 4 ("count a ``q`` field as personal") is equivalent at the predicate: any field named ``q`` already
  fails (b). It is caught only where a search name is not in ``SEARCH_NAMES`` exactly; reported.

Regime: CPU, pure function, no I/O, n = 1 per row; Hypothesis for totality (deterministic profile of the suite).
"""

from __future__ import annotations

import ast
import gc
import inspect
import sys
import time
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.computer import fence
from tests.unit._work_count import line_events

#: A1 verbatim (H-T206-1); typed here, never imported for the oracle
SEND_LABELS = (
    "submit", "send", "send message", "send request", "send enquiry", "send inquiry",
    "request quote", "request a quote", "request a free quote", "request estimate", "request an estimate",
    "get quote", "get a quote", "get my quote", "get a free quote", "get free quote", "get estimate",
    "place order", "place my order", "complete order", "complete purchase", "confirm order", "confirm booking",
    "confirm reservation",
    "book now", "reserve now", "pay", "pay now", "buy now",
    "sign up", "register", "create account", "subscribe", "apply now", "donate",
)  # fmt: skip
SEARCH_LABELS = ("search", "find", "go", "look up", "find stores", "search flights", "show results")
SEARCH_NAMES = (
    "q", "query", "search", "s", "k", "keyword", "keywords", "field-keywords", "term", "search_query",
)  # fmt: skip
PERSONAL_AUTOCOMPLETE = (
    "name", "given-name", "family-name", "email", "tel", "tel-national", "street-address", "address-line1",
    "address-line2", "postal-code", "address-level1", "address-level2", "organization",
)  # fmt: skip
LOCATION_AUTOCOMPLETE = frozenset(
    {
        "street-address",
        "address-line1",
        "address-line2",
        "postal-code",
        "address-level1",
        "address-level2",
    }
)
LOCATION_WORDS = frozenset({"address", "street", "zip", "postcode", "postal"})
PERSONAL_WORDS = (
    "name", "email", "e-mail", "phone", "tel", "mobile", "address", "street", "zip", "postcode", "postal",
    "message", "comment", "comments",
)  # fmt: skip


def predicate() -> Any:
    fn = getattr(fence, "irreversible_submit", None)
    assert fn is not None, "T-0206 A1: fence.irreversible_submit does not exist yet"
    return fn


def check(facts: Any) -> bool:
    got = predicate()(facts)
    assert isinstance(got, bool), f"A1: irreversible_submit returns a bool, got {got!r}"
    return got


# ============================================================================================ builders


def ctl(
    label: str = "Submit",
    *,
    tag: str = "button",
    type: str = "submit",
    role: str = "",
    href_nav: bool = False,
    structural: bool | None = None,
) -> dict[str, Any]:
    """A control. ``structural`` defaults to A1's definition for a form owner: a ``button`` of type submit or none, or an
    ``input`` of type submit/image."""
    if structural is None:
        structural = (tag == "button" and type in ("submit", "")) or (
            tag == "input" and type in ("submit", "image")
        )
    return {
        "tag": tag,
        "type": type,
        "role": role,
        "label": label,
        "href_nav": href_nav,
        "structural": structural,
    }


def fld(
    *,
    tag: str = "input",
    type: str = "text",
    autocomplete: str = "",
    name: str = "",
    id: str = "",
    placeholder: str = "",
    label: str = "",
    filled: bool = True,
    visible: bool = True,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "type": type,
        "autocomplete": autocomplete,
        "name": name,
        "id": id,
        "placeholder": placeholder,
        "label": label,
        "filled": filled,
        "visible": visible,
    }


def facts(
    control: dict[str, Any],
    fields: list[dict[str, Any]],
    *,
    scope: str = "form",
    search_scope: bool = False,
) -> dict[str, Any]:
    return {"control": control, "scope": scope, "search_scope": search_scope, "fields": fields}


EMAIL = fld(type="email", name="email", label="Email", autocomplete="email")
NAME = fld(name="name", id="name", label="Name", autocomplete="name")
PHONE = fld(type="tel", name="phone", label="Phone", autocomplete="tel")
DESCRIPTION = fld(tag="textarea", type="textarea", name="description", label="Description")
QUANTITY = fld(type="number", name="quantity", label="Quantity")


def contact(label: str, *fields: dict[str, Any], **ctl_kw: Any) -> dict[str, Any]:
    """A contact form: the control ``label`` and ``fields`` (default: a filled email)."""
    return facts(ctl(label, **ctl_kw), list(fields) or [EMAIL])


# ============================================================================================ A5 / A6 rows


A5_ROWS = [
    pytest.param(
        facts(ctl("Google Search", tag="input", type="submit"), [fld(name="q", label="Search")]),
        id="google-q",  # (b) q in SEARCH_NAMES (and the label begins with no send label)
    ),
    pytest.param(
        facts(
            ctl("Go", tag="input", type="submit"),
            [fld(type="text", name="field-keywords", id="twotabsearchtextbox")],
        ),
        id="amazon-field-keywords",  # (b) field-keywords; (d) "go" is a search label
    ),
    pytest.param(
        facts(ctl("Search"), [fld(type="search", name="search", placeholder="Search Wikipedia")]),
        id="wikipedia-search",  # (b) a filled type=search and name=search
    ),
    pytest.param(
        facts(
            ctl("Search"),
            [fld(type="search", name="model", placeholder="Model number", label="Model")],
        ),
        id="d4-model-search",  # (b) a filled type=search
    ),
    pytest.param(
        facts(
            ctl("Find stores"),
            [fld(name="address", label="Street address", autocomplete="street-address")],
        ),
        id="d6-find-stores-with-street-address",  # (c) holds (address), (d): "find stores" is a search label
    ),
    pytest.param(
        facts(
            ctl("Search"),
            [
                fld(name="origin", label="From", placeholder="Origin"),
                fld(name="destination", label="To", placeholder="Destination"),
                fld(type="date", name="depart", label="Depart"),
            ],
        ),
        id="d9-flight-search",  # (c) no personal field; (d) "search" is a search label
    ),
    pytest.param(
        facts(
            ctl("Sign in"),
            [EMAIL, fld(type="password", name="password", label="Password")],
        ),
        id="sign-in",  # (d) "sign in" is no send label, and the label is not empty
    ),
    pytest.param(
        facts(ctl("Add to cart"), [QUANTITY]),
        id="add-to-cart",  # (c) quantity is not personal; (d) no send label
    ),
    pytest.param(
        facts(
            ctl("Get a free quote", tag="a", type="", href_nav=True, structural=False),
            [],
            scope="document",
        ),
        id="quote-link-nothing-filled",  # (a) a navigating link; (c) nothing filled
    ),
    pytest.param(
        facts(
            ctl("Get a free quote", tag="a", type="", href_nav=True, structural=False),
            [fld(type="email", name="newsletter-email", label="Your email")],
            scope="document",
        ),
        id="quote-link-newsletter-email-filled",  # (a) alone decides: (b)-(d) all hold
    ),
    pytest.param(
        facts(
            ctl("In stock", tag="input", type="checkbox", structural=False),
            [fld(type="checkbox", name="in-stock", label="In stock")],
        ),
        id="filter-checkbox",  # (c) none personal; (d) no send label, not structural
    ),
    pytest.param(
        facts(
            ctl("Go"),
            [fld(name="address", id="address", placeholder="en.wikipedia.org", autocomplete="off")],
        ),
        id="start-page-web-html",  # census: /web.html's "address" box is personal by name, "go" is a search label
    ),
    pytest.param(
        facts(
            ctl("Accept all", type="button", structural=False),
            [EMAIL],
            scope="document",
        ),
        id="cookie-accept-all",  # (d) "accept all" is no send label, not structural
    ),
]

A6_ROWS = [
    pytest.param(
        facts(ctl("Request quote"), [NAME, fld(type="tel", name="phone", filled=False)]),
        id="d7-request-quote-after-name",  # (c) name; (d) "request quote"
    ),
    pytest.param(
        facts(ctl("Request quote"), [fld(name="name", filled=False), DESCRIPTION]),
        id="d7-description-only",  # (c) a textarea
    ),
    pytest.param(
        facts(ctl("", type="submit", structural=True), [EMAIL]),
        id="empty-label-structural-submit",  # (d) structural with an empty label
    ),
    pytest.param(
        facts(
            ctl("Send message", tag="div", type="", role="button", structural=False),
            [fld(tag="textarea", type="textarea", name="body", label="Your message")],
            scope="document",
        ),
        id="spa-div-send-message",  # formless: document scope; (c) textarea; (d) "send message"
    ),
    pytest.param(
        facts(ctl("Request quote"), [NAME, PHONE]),
        id="enter-in-d7-phone",  # the focused read with the form's default button as the control
    ),
    pytest.param(facts(ctl("Submit"), [NAME, EMAIL]), id="space-on-focused-submit"),
    pytest.param(
        facts(ctl("Send"), [EMAIL, fld(tag="textarea", type="textarea", name="message")]),
        id="cross-origin-iframe-send",
    ),
]


@pytest.mark.parametrize("f", A5_ROWS)
def test_a5_false_positives_stay_open(f: dict[str, Any]) -> None:
    """A5: search, store locator, flight search, sign-in, add-to-cart, a CTA link, a filter, a cookie banner: False."""
    assert check(f) is False, f


@pytest.mark.parametrize("f", A6_ROWS)
def test_a6_true_positives(f: dict[str, Any]) -> None:
    """A6: the send control of a form with filled personal data: True."""
    assert check(f) is True, f


def test_rule_a_link_partner() -> None:
    """(a) is what decides the newsletter row: the same facts with ``href_nav`` false (a ``#`` or ``javascript:`` link,
    or a button) are True; a navigating link is False."""
    email = [fld(type="email", name="newsletter-email", label="Your email")]
    link = ctl("Get a free quote", tag="a", type="", structural=False)
    assert check(facts(dict(link, href_nav=False), email, scope="document")) is True
    assert check(facts(dict(link, href_nav=True), email, scope="document")) is False


# ============================================================================================ (d) labels


@pytest.mark.parametrize(
    "label",
    [
        "Request a quote →",  # trailing arrow stripped
        "SUBMIT",  # casefold
        "Send  message",  # whitespace collapsed
        "Submit request",  # begins with "submit" on a word boundary
        "  Send\n\tmessage  ",
        "Submit!",
        "Send.",
        "Get a free quote →",
        "Book now",
        "Pay",
        "Pay now",
        "Buy now",
        "Place order",
        "Confirm booking",
        "Sign up",
        "Register",
        "Create account",
        "Subscribe",
        "Apply now",
        "Donate",
        "Send enquiry",
        "Request an estimate",
    ],
)
def test_send_labels_true(label: str) -> None:
    """(d) normalisation: casefold, whitespace collapsed, trailing punctuation and arrows stripped; a filled email."""
    assert check(contact(label)) is True, label


@pytest.mark.parametrize(
    "label",
    [
        "Sender details",  # "send" not on a word boundary
        "Sending options",
        "Submitted",
        "Payment options",
        "Paypal",
        "Booking details",
        "Registered users",
        "Subscriber login",
        "Sign in",
        "Confirm",  # H-T206-1: confirm alone, apply, post, next, continue are out
        "Apply",
        "Post",
        "Next",
        "Continue",
        "Add to cart",
        "Learn more about how we send",  # does not *begin* with a send label
    ],
)
def test_not_send_labels_false(label: str) -> None:
    """(d): a non-empty label that does not begin (on a word boundary) with a send label is False, even with a filled
    email and a structural submit."""
    assert check(contact(label)) is False, label


@pytest.mark.parametrize(
    "label", ["Search", "Go", "Find", "Find stores", "Search flights", "Show results", "Look up"]
)
def test_search_labels_win(label: str) -> None:
    """``SEARCH_LABELS`` precedence: a search label on a form with a filled email (no search field) is False."""
    assert check(contact(label)) is False, label


def test_search_label_partner() -> None:
    """The discriminating partner of the precedence rows: the same form with "Submit" is True."""
    assert check(contact("Submit")) is True


def test_structural_only_when_label_empty() -> None:
    """(d) second branch: an empty label counts only on a structural control; a non-structural empty-label control
    (``type=button``) is False; a structural control with a non-send label is False."""
    assert check(facts(ctl("", type="submit", structural=True), [EMAIL])) is True
    assert check(facts(ctl("", tag="input", type="image", structural=True), [EMAIL])) is True
    assert check(facts(ctl("", type="button", structural=False), [EMAIL])) is False
    assert check(facts(ctl("Next", type="submit", structural=True), [EMAIL])) is False


# ============================================================================================ (b) search scope


def test_search_scope_flag() -> None:
    """(b): ``search_scope`` (a role=search ancestor or a <search> element) alone makes a "Submit" with a filled email
    False."""
    assert check(facts(ctl("Submit"), [EMAIL], search_scope=True)) is False
    assert check(facts(ctl("Submit"), [EMAIL], search_scope=False)) is True


@pytest.mark.parametrize("name", SEARCH_NAMES)
@pytest.mark.parametrize("attr", ["name", "id"])
def test_search_names(name: str, attr: str) -> None:
    """(b) as ruled (R-2): a visible field whose ``name`` or ``id`` is in ``SEARCH_NAMES`` exempts the form when it is
    filled; left empty beside a filled email it does not."""
    assert check(facts(ctl("Submit"), [EMAIL, fld(filled=True, **{attr: name})])) is False, (
        attr,
        name,
    )
    assert check(facts(ctl("Submit"), [EMAIL, fld(filled=False, **{attr: name})])) is True, (
        attr,
        name,
    )


def test_type_search_counts_only_filled() -> None:
    """(b): a *filled* ``type=search`` field is search-like; an empty one (named outside ``SEARCH_NAMES``) is not."""
    assert (
        check(facts(ctl("Submit"), [EMAIL, fld(type="search", name="site", filled=True)])) is False
    )
    assert (
        check(facts(ctl("Submit"), [EMAIL, fld(type="search", name="site", filled=False)])) is True
    )


def test_structural_empty_label_on_q_form_false() -> None:
    """Adversarial: a structural empty-label submit on a ``q`` form (implicit submission of a search) is False."""
    assert (
        check(facts(ctl("", type="submit", structural=True), [fld(name="q", label="Your name")]))
        is False
    )
    assert check(facts(ctl("", type="submit", structural=True), [fld(name="q"), EMAIL])) is False


def test_email_search_named_box_planner_reading() -> None:
    """Adversarial (planner's reading, accepted as written): a text box named ``email-search`` is not in
    ``SEARCH_NAMES`` (exact match), it matches the whole word ``email``, so it is personal; with "Submit": True."""
    box = fld(type="text", name="email-search", placeholder="Search by email")
    assert check(facts(ctl("Submit"), [box])) is True


def test_formless_scope_empty_search_box_does_not_exempt() -> None:
    """Ruling on reading 6 (safety): in a formless scope an *empty* site-search box (``name=q``) does not make the
    SPA's "Send message" a search; the filled textarea makes it a send: True either way."""
    msg = fld(tag="textarea", type="textarea", name="body")
    spa = ctl("Send message", tag="div", type="", role="button", structural=False)
    assert check(facts(spa, [msg], scope="document")) is True
    assert check(facts(spa, [fld(name="q", filled=False), msg], scope="document")) is True
    assert (
        check(facts(spa, [fld(type="search", name="site", filled=False), msg], scope="document"))
        is True
    )


def test_spa_quote_form_with_header_search_refused() -> None:
    """Ruling row: an SPA quote form (a div "Send", Name, Email and Message filled) whose scope also holds the page
    header's empty ``q`` box is True."""
    send = ctl("Send", tag="div", type="", role="button", structural=False)
    fields = [
        fld(name="q", placeholder="Search the site", filled=False),
        fld(name="name", label="Name"),
        fld(type="email", name="email", label="Email"),
        fld(tag="textarea", type="textarea", name="message", label="Message"),
    ]
    assert check(facts(send, fields, scope="document")) is True


def test_formless_scope_filled_search_beside_filled_email() -> None:
    """Ruling 6 partner: in a formless scope a filled ``q`` beside a filled email is not "all filled inputs search-like",
    so a div "Send" is True (the form-scope filled-search exemption must not leak to document scope)."""
    assert check(facts(ctl("Send"), [fld(name="q"), EMAIL], scope="document")) is True


def test_formless_scope_filled_search_only_is_open() -> None:
    """Ruling: a formless scope whose filled inputs are all search-like (a typed ``q``) is a search: a div "Send" there
    is False, also with empty personal fields beside it."""
    send = ctl("Send", tag="div", type="", role="button", structural=False)
    assert check(facts(send, [fld(name="q")], scope="document")) is False
    q_and_empty = [fld(name="q"), fld(type="email", name="email", filled=False)]
    assert check(facts(send, q_and_empty, scope="document")) is False


def test_form_scope_empty_search_box_does_not_exempt() -> None:
    """R-2 (extends ruling 6 to form scope): a whole-page ``<form>`` (WebForms shape) holding an empty header ``q`` and a
    quote form (Name, Email, Message filled, "Send") is True; the same page with ``q`` filled and nothing personal
    filled is False."""
    quote = [
        fld(name="q", placeholder="Search", filled=False),
        fld(name="name"),
        fld(type="email", name="email"),
        fld(tag="textarea", type="textarea", name="message"),
    ]
    assert check(facts(ctl("Send"), quote, scope="form")) is True
    searching = [dict(quote[0], filled=True)] + [dict(f, filled=False) for f in quote[1:]]
    assert check(facts(ctl("Send"), searching, scope="form")) is False


def test_hidden_search_name_does_not_exempt() -> None:
    """Reviewer gap r2: a hidden ``<input type=hidden name=q value=...>`` (filled, not visible) does not make a filled
    contact form a search."""
    hidden_q = fld(type="hidden", name="q", filled=True, visible=False)
    assert check(facts(ctl("Submit"), [hidden_q, EMAIL])) is True


def test_hash_link_send_in_personal_form() -> None:
    """Reviewer gap r12: ``<a href="#">Send</a>`` (not a navigating link) in a form with a filled email is True."""
    link = ctl("Send", tag="a", type="", structural=False, href_nav=False)
    assert check(facts(link, [EMAIL, fld(tag="textarea", type="textarea", name="message")])) is True


# ============================================================================================ (c) personal fields


PERSONAL = [
    pytest.param(fld(tag="textarea", type="textarea", name="notes"), id="textarea"),
    pytest.param(fld(type="email", name="x1"), id="type-email"),
    pytest.param(fld(type="tel", name="x2"), id="type-tel"),
    *[
        pytest.param(fld(autocomplete=tok, name="f1"), id=f"ac-{tok}")
        for tok in PERSONAL_AUTOCOMPLETE
        if tok not in LOCATION_AUTOCOMPLETE
    ],
    pytest.param(fld(autocomplete="billing email", name="f1"), id="ac-token-with-section"),
    *[pytest.param(fld(name=w), id=f"name-{w}") for w in PERSONAL_WORDS if w not in LOCATION_WORDS],
    *[pytest.param(fld(id=w), id=f"id-{w}") for w in PERSONAL_WORDS if w not in LOCATION_WORDS],
    pytest.param(fld(label="Your name"), id="label-your-name"),
    pytest.param(fld(label="Email address"), id="label-email-address"),
    pytest.param(fld(label="Phone number"), id="label-phone-number"),
    pytest.param(fld(label="MOBILE"), id="label-casefold"),
    pytest.param(fld(placeholder="Your message"), id="placeholder-message"),
    pytest.param(fld(placeholder="Any comments?"), id="placeholder-comments-q"),
    pytest.param(fld(name="full-name"), id="name-hyphen-boundary"),
]

#: R-3: location fields (address, street, city, postcode, zip, country): personal by A1's lists, but a form whose filled
#: personal fields are all location fields is a locator search. ``postal`` and the ``address-level*`` tokens (region,
#: city) are read as location (test-author reading, reported).
LOCATION = [
    *[
        pytest.param(fld(autocomplete=tok, name="f1"), id=f"ac-{tok}")
        for tok in sorted(LOCATION_AUTOCOMPLETE)
    ],
    pytest.param(fld(autocomplete="shipping postal-code", name="f1"), id="ac-token-with-section"),
    *[pytest.param(fld(name=w), id=f"name-{w}") for w in sorted(LOCATION_WORDS)],
    *[pytest.param(fld(id=w), id=f"id-{w}") for w in sorted(LOCATION_WORDS)],
    pytest.param(fld(placeholder="Postal code"), id="placeholder-postal"),
    pytest.param(fld(label="ZIP"), id="label-zip"),
    pytest.param(fld(name="city"), id="name-city"),
    pytest.param(fld(name="billing_address_country"), id="name-billing-address-country"),
]


@pytest.mark.parametrize("field", LOCATION)
def test_location_only_is_a_locator(field: dict[str, Any]) -> None:
    """R-3: a form whose only filled personal field is a location field is a locator search: "Submit" and an icon-only
    structural submit are False; adding a filled email (or name, or phone) makes it a personal send: True."""
    icon = ctl("", type="submit", structural=True)
    assert check(facts(ctl("Submit"), [field])) is False, field
    assert check(facts(icon, [field])) is False, field
    assert check(facts(icon, [field, EMAIL])) is True, field
    assert check(facts(icon, [field, NAME])) is True, field
    assert check(facts(icon, [field, PHONE])) is True, field


def test_store_locator_postcode_and_address() -> None:
    """R-3: a store locator with postcode and street address filled and an icon-only submit is False; adding a filled
    email makes it True."""
    loc = [
        fld(name="postcode", placeholder="Postcode"),
        fld(name="address", autocomplete="street-address"),
    ]
    icon = ctl("", type="submit", structural=True)
    assert check(facts(icon, loc)) is False
    assert check(facts(icon, [*loc, EMAIL])) is True


NOT_PERSONAL = [
    pytest.param(fld(name="quantity"), id="quantity"),
    pytest.param(fld(name="username"), id="username-not-whole-word"),
    pytest.param(fld(name="nickname"), id="nickname"),
    pytest.param(fld(name="telescope"), id="telescope"),
    pytest.param(fld(label="Addressee"), id="addressee"),
    pytest.param(fld(placeholder="Messages"), id="messages-plural"),
    pytest.param(fld(type="number", name="guests"), id="guests"),
    pytest.param(fld(type="date", name="date"), id="date"),
    pytest.param(fld(type="checkbox", name="agree"), id="checkbox"),
    pytest.param(fld(autocomplete="username", name="login"), id="ac-username"),
    pytest.param(fld(autocomplete="off", name="promo"), id="ac-off"),
    pytest.param(fld(autocomplete="cc-name", name="f2"), id="ac-cc-name"),
    pytest.param(fld(type="url", name="website"), id="url"),
]


@pytest.mark.parametrize("field", PERSONAL)
def test_personal_field(field: dict[str, Any]) -> None:
    """(c), field by field: this one filled and visible field makes a "Submit" True."""
    assert check(facts(ctl("Submit"), [field])) is True, field


@pytest.mark.parametrize("field", NOT_PERSONAL)
def test_not_personal_field(field: dict[str, Any]) -> None:
    """(c), field by field: this one field alone is not personal; "Submit" is False."""
    assert check(facts(ctl("Submit"), [field])) is False, field


@pytest.mark.parametrize(
    ("filled", "visible", "want"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_filled_and_visible_gate(filled: bool, visible: bool, want: bool) -> None:
    """(c): only a filled *and* visible personal field counts (a hidden filled email does not)."""
    email = fld(type="email", name="email", filled=filled, visible=visible)
    assert check(facts(ctl("Submit"), [email])) is want, (filled, visible)


def test_hidden_input_named_email_does_not_count() -> None:
    """(c): ``<input type=hidden name=email>`` (filled, not visible) beside a visible empty name field: False."""
    f = facts(
        ctl("Submit"),
        [fld(type="hidden", name="email", visible=False), fld(name="name", filled=False)],
    )
    assert check(f) is False


# ============================================================================================ MAX_FIELDS


def test_max_fields_constant() -> None:
    assert getattr(fence, "MAX_FIELDS", None) == 64, "T-0206 A1: fence.MAX_FIELDS = 64"


def test_max_fields_truncation() -> None:
    """``fields`` holds at most ``MAX_FIELDS = 64`` items: a personal field at index 63 counts, at index 64 it does not;
    a search name past index 63 is not read either."""
    filler = [QUANTITY] * 63
    assert check(facts(ctl("Submit"), [*filler, EMAIL])) is True
    assert check(facts(ctl("Submit"), [*filler, QUANTITY, EMAIL])) is False
    assert check(facts(ctl("Submit"), [EMAIL, *filler, fld(name="q")])) is True
    assert check(facts(ctl("Submit"), [EMAIL, *filler[:-1], fld(name="q")])) is False


# ============================================================================================ constants


def test_lexicon_constants() -> None:
    """A1/H-T206-1: the lexicons as written (as sets; order is not part of the contract)."""
    for name, want in (
        ("SEND_LABELS", SEND_LABELS),
        ("SEARCH_LABELS", SEARCH_LABELS),
        ("SEARCH_NAMES", SEARCH_NAMES),
        ("PERSONAL_AUTOCOMPLETE", PERSONAL_AUTOCOMPLETE),
    ):
        got = getattr(fence, name, None)
        assert got is not None, f"T-0206 A1: fence.{name} does not exist yet"
        assert set(got) == set(want), (name, sorted(set(got) ^ set(want)))


@pytest.mark.parametrize("label", SEND_LABELS)
def test_every_send_label_true(label: str) -> None:
    """(d): every ``SEND_LABELS`` entry, title-cased, on a form with a filled email: True."""
    assert check(contact(label.title())) is True, label


def test_signature_and_stdlib_only() -> None:
    """A1: ``irreversible_submit(facts)``, in a module that imports the standard library only."""
    fn = predicate()
    assert list(inspect.signature(fn).parameters) == ["facts"]
    tree = ast.parse(inspect.getsource(fence))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    non_std = sorted(r for r in roots if r not in sys.stdlib_module_names and r != "__future__")
    assert non_std == [], f"A1: fence imports non-stdlib modules {non_std}"


# ============================================================================================ totality, time


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=12), children, max_size=5)
    ),
    max_leaves=25,
)
KEYS = ("tag", "type", "role", "label", "href_nav", "structural", "autocomplete", "name", "id",
        "placeholder", "filled", "visible")  # fmt: skip
SHAPED_DICT = st.dictionaries(st.sampled_from(KEYS), JSON, max_size=12)
SHAPED = st.fixed_dictionaries(
    {},
    optional={
        "control": SHAPED_DICT | JSON,
        "scope": st.sampled_from(["form", "document", ""]) | JSON,
        "search_scope": st.booleans() | JSON,
        "fields": st.lists(SHAPED_DICT | JSON, max_size=6) | JSON,
    },
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(value=JSON | SHAPED)
def test_total_over_json_like_values(value: Any) -> None:
    """A1 (Hypothesis): for any JSON-like value, and any facts-shaped mapping whose values are any JSON-like value, the
    predicate returns a bool and never raises."""
    check(value)


@pytest.mark.parametrize("bad", [None, {}, [], "submit", 0, {"control": None}, {"fields": None}])
def test_total_degenerate(bad: Any) -> None:
    """A1: degenerate inputs are False (no control, no fields: (c) cannot hold)."""
    assert check(bad) is False


def _big(n: int) -> dict[str, Any]:
    """64 fields and a control whose strings are built to make a backtracking regex blow up: runs of near-matches,
    spaces, arrows and hyphens, about ``6 n`` characters each."""
    junk = ("send " * n) + "!" + (" " * n) + ("\u2192" * n) + ("-" * n) + "x"
    field = fld(name=junk, id=junk, label=junk, placeholder=junk, autocomplete=junk)
    return facts(ctl(junk), [field] * 64)


def test_linear_work_in_string_length() -> None:
    """A1 (linear time), counted (T-0129 rule): the traced Python lines of ``fence`` for 10x longer strings are at most
    15x (linear predicts at most 10x, quadratic 100x)."""
    root = fence.__file__
    small = line_events(lambda: check(_big(200)), root)
    large = line_events(lambda: check(_big(2000)), root, cap=max(100_000, 50 * small))
    assert large <= 15 * max(small, 1), (small, large)


def test_linear_cpu_in_string_length() -> None:
    """A1 (linear time), CPU half (C-level regex work is invisible to line counts): ``thread_time``, gc off, sizes
    interleaved, min of 5; 10x the input costs under 40x (linear 10x, quadratic 100x), and the large input is answered
    within 2 s of CPU."""
    small, large = _big(200), _big(2000)

    def cpu(f: dict[str, Any]) -> float:
        t0 = time.thread_time()
        check(f)
        return time.thread_time() - t0

    check(small)
    ts: list[float] = []
    tl: list[float] = []
    was = gc.isenabled()
    gc.disable()
    try:
        for _ in range(5):
            ts.append(cpu(small))
            tl.append(cpu(large))
    finally:
        if was:
            gc.enable()
    t_small, t_large = min(ts), min(tl)
    assert t_large < 2.0, f"A1: {t_large:.3f} s of CPU"
    assert t_large < 40 * max(t_small, 1e-3), (t_small, t_large)


def test_field_list_beyond_max_is_not_walked() -> None:
    """A1 (MAX_FIELDS, counted): 200 000 fields cost at most 2x the traced lines of 64 (only the first 64 are read)."""
    root = fence.__file__
    f64 = facts(ctl("Submit"), [QUANTITY] * 64)
    fbig = facts(ctl("Submit"), [QUANTITY] * 200_000)
    n64 = line_events(lambda: check(f64), root)
    nbig = line_events(lambda: check(fbig), root, cap=max(10_000, 4 * n64))
    assert nbig <= 2 * max(n64, 1), (n64, nbig)
