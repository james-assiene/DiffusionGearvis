"""T-0192 A1 (and the A5/A6 negatives): ``fence.checkout_url(url) -> bool`` (docs/tasks/T-0192.md).

A1, as written: ``True`` exactly when, for an ``http(s)`` URL, (a) a path segment matches ``checkout``,
``checkouts``, ``checkoutnow``, ``payment``, ``payments``, ``pay`` (case-insensitive, after one percent-decode, after
removing at most one trailing ``.html``/``.htm``/``.php``/``.aspx``/``.jsp``); or (b) a segment ``buy`` comes right
after a segment ``gp``; or (c) the host's leftmost DNS label is ``checkout``, ``pay`` or ``payments``. Query and
fragment are never read. An unparseable URL is ``False``.

Every expected value below is worked out by hand from that text (the comment beside a row says which clause decides
it); none is captured from an implementation. Rows the text leaves open are not pinned (reported): a ``%2F`` inside
a segment (decode before or after the split), an upper-case ``GP``/``BUY`` (clause (b) does not say
case-insensitive), a suffix after ``buy``, an upper-case suffix, a port out of range, ``;``-parameters in a segment, an empty segment between ``gp`` and ``buy``.

Regime: CPU, pure function, no I/O, n = 1 per row.
"""

from __future__ import annotations

import ast
import inspect
import sys
from typing import Any

import pytest

from local_astra.computer import fence

S = "https://shop.example"


def predicate() -> Any:
    fn = getattr(fence, "checkout_url", None)
    assert fn is not None, "T-0192 A1: fence.checkout_url does not exist yet"
    return fn


def check(url: str) -> bool:
    got = predicate()(url)
    assert isinstance(got, bool), f"A1: checkout_url returns a bool, got {got!r}"
    return got


TRUE = [
    # the task's true rows
    (f"{S}/checkout", "(a) checkout"),
    (f"{S}/Checkout/", "(a) case-insensitive; the trailing slash is an empty segment"),
    (f"{S}/checkout/entry/cart?proceedToCheckout=1", "(a) the first segment"),
    (f"{S}/gp/buy/spc/handlers/display.html", "(b) gp then buy"),
    (f"{S}/checkouts/abc", "(a) checkouts"),
    (f"{S}/checkout.html", "(a) one .html removed"),
    (f"{S}/%63heckout", "(a) one percent-decode: %63 is c"),
    ("https://checkout.stripe.com/c/x", "(c) leftmost label checkout"),
    ("https://pay.ebay.com/rxo", "(c) leftmost label pay"),
    (f"{S}/en-gb/payment", "(a) payment, not the first segment"),
    # the rest of A1's lists
    (f"{S}/checkoutnow", "(a) checkoutnow"),
    (f"{S}/payments/new", "(a) payments"),
    (f"{S}/x/pay", "(a) pay, last segment"),
    (f"{S}/checkout.htm", "(a) .htm"),
    (f"{S}/checkout.php", "(a) .php"),
    (f"{S}/pay.aspx", "(a) .aspx"),
    (f"{S}/payment.jsp", "(a) .jsp"),
    (f"{S}/x/checkout.html/", "(a) a middle segment with its suffix"),
    (f"{S}/PAYMENT", "(a) case-insensitive"),
    (f"{S}/checkout%2Ehtml", "(a) decode (%2E is .) then remove .html"),
    ("https://payments.example/", "(c) leftmost label payments"),
    (
        "https://checkout.example.evil/",
        "(c) fail closed: any host whose leftmost label is checkout",
    ),
    ("https://CHECKOUT.shop.example/", "(c) DNS names are case-insensitive"),
    ("https://checkout.example:8443/", "(c) a port does not change the host"),
    ("https://checkout.example./", "(c) a trailing root dot"),
    ("https://shop.example@checkout.example/", "(c) the host is after the userinfo"),
    ("http://shop.example/checkout", "http as well as https"),
    ("HTTPS://shop.example/checkout", "the scheme is case-insensitive"),
    ("https://93.184.215.14/checkout", "(a) on an IP host"),
    ("https://xn--80ak6aa92e.example/checkout", "(a) on an IDNA host"),
    (f"{S}/gp/buy", "(b) exactly gp, buy"),
    (f"{S}/x/gp/buy/", "(b) gp, buy anywhere in the path"),
]

FALSE = [
    # A5: the add-to-cart path stays open
    (f"{S}/cart", "A5"),
    (f"{S}/gp/cart/view.html", "A5"),
    (f"{S}/cart/smart-wagon", "A5"),
    (f"{S}/gp/aws/cart/add.html", "A5"),
    (f"{S}/gp/buyagain", "A5: buyagain is not buy"),
    (f"{S}/checkout-faq", "A5: a whole segment, not a prefix"),
    (f"{S}/payment-methods", "A5"),
    (f"{S}/s?k=checkout", "A5: the query is never read"),
    (
        f"{S}/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.com%2Fcheckout%2Fentry",
        "A5: the query is never read",
    ),
    (f"{S}/#checkout", "A5: the fragment is never read"),
    ("https://checkout-help.example/", "A5: leftmost label checkout-help"),
    ("https://shop.checkout.example/", "A5: checkout is not the leftmost label"),
    # A6: CAPTCHA and verification walls
    ("https://www.google.com/sorry/index?continue=https://www.google.com/search", "A6"),
    ("https://www.amazon.com/errors/validateCaptcha", "A6"),
    ("https://www.amazon.com/ap/cvf/request?arb=1", "A6"),
    ("https://www.amazon.com/ap/cvf/verify", "A6"),
    # adversarial look-alikes
    (f"{S}/checkouts-guide", "not a listed segment"),
    (f"{S}/gpbuy", "one segment, not gp then buy"),
    (f"{S}/buy", "buy alone"),
    (f"{S}/buy/gp", "(b) is ordered"),
    (f"{S}/gp/x/buy", "(b) right after"),
    (f"{S}/payroll", "pay is a whole segment"),
    (f"{S}/paypal", "pay is a whole segment"),
    (f"{S}/cart?next=/checkout", "query"),
    (f"{S}/#/checkout", "fragment"),
    (f"{S}/cart#/checkout", "fragment"),
    (f"{S}/checkout.html.html", "at most one suffix removed"),
    (f"{S}/checkout.json", "not a listed suffix"),
    (f"{S}/%2563heckout", "one decode only: %25 is %, leaving %63heckout"),
    (f"{S}/check%20out", "a space inside"),
    (f"{S}/%20checkout", "a leading space is part of the segment"),
    ("https://xn--checkout-xyz.example/", "an IDNA leftmost label that only contains checkout"),
    ("https://xn--80ak6aa92e.example/", "an IDNA host, plain path"),
    ("https://payment.example/", "(c) lists pay and payments, not payment"),
    ("https://checkoutnow.example/", "(c) does not list checkoutnow"),
    ("https://pay-later.example/", "leftmost label pay-later"),
    ("https://checkout@shop.example/", "the userinfo is not the host"),
    ("https://shop.example/a/checkoutx", "not a listed segment"),
    # not http(s), or not a URL
    ("ftp://shop.example/checkout", "not http(s)"),
    ("file:///checkout", "not http(s)"),
    ("javascript:checkout()", "not http(s)"),
    ("data:text/html,checkout", "not http(s)"),
    ("about:blank", "not http(s)"),
    ("wss://checkout.example/checkout", "not http(s)"),
    ("checkout", "no scheme"),
    ("/checkout", "no scheme"),
    ("", "empty"),
    ("http://[::1", "unparseable (an unclosed bracket)"),
]


@pytest.mark.parametrize(("url", "why"), TRUE, ids=[u for u, _ in TRUE])
def test_true_rows(url: str, why: str) -> None:
    """A1: a checkout (the clause in ``why``)."""
    assert check(url) is True, f"A1: {url!r} is a checkout ({why})"


@pytest.mark.parametrize(("url", "why"), FALSE, ids=[u for u, _ in FALSE])
def test_false_rows(url: str, why: str) -> None:
    """A1/A5/A6: not a checkout (``why``); never an exception."""
    assert check(url) is False, f"A1: {url!r} is not a checkout ({why})"


def test_long_path_bounded() -> None:
    """A1, adversarial: a 10 kB path is answered (no exception, no recursion limit), both ways."""
    long_path = "/a" * 5_000
    assert check(S + long_path) is False
    assert check(S + long_path + "/checkout") is True
    assert check(S + "/" + "%" * 10_000) is False


def test_query_and_fragment_never_read() -> None:
    """A1: the same path with any query/fragment gives the same answer (a discriminating pair each way)."""
    for tail in ("?step=checkout", "#checkout", "?a=/gp/buy#/pay", "?x=%2Fcheckout"):
        assert check(f"{S}/cart{tail}") is False, tail
        assert check(f"{S}/checkout{tail}") is True, tail


def test_signature_and_stdlib_only() -> None:
    """A1: ``checkout_url(url: str) -> bool``, in a module that imports the standard library only."""
    fn = predicate()
    params = list(inspect.signature(fn).parameters)
    assert params == ["url"], params
    tree = ast.parse(inspect.getsource(fence))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    non_std = sorted(r for r in roots if r not in sys.stdlib_module_names and r != "__future__")
    assert non_std == [], f"A1: fence imports non-stdlib modules {non_std}"


def test_no_captcha_detection_in_group_a_sources() -> None:
    """A6: no code path names or detects a CAPTCHA (fence, runtime, executor; a docstring saying "CAPTCHA" would
    trip this too: keep the word out of these three modules)."""
    from local_astra.computer import executor, runtime

    for mod in (fence, runtime, executor):
        src = inspect.getsource(mod).lower()
        assert "captcha" not in src, f"A6: {mod.__name__} names a CAPTCHA"
