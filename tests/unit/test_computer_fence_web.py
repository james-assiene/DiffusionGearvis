"""T-0160 group A, the pure half: ``fence.is_public_address``, ``fence.web_url_allowed``, ``fence.card_like`` and
``fence.secret_field`` (criteria A-1 to A-4), plus the A-7 differential that the T-0111c/T-0111g rules are unchanged.

Every expected value is an oracle table written by hand from the criteria (RFC 6890 special-purpose registries,
the WHATWG IPv4 parser's numeric forms, Luhn worked by hand on the standard test card numbers). Nothing here
re-implements the functions. Both directions of every guard are rows: a guard needs a row for being absent and a
row for being too wide (.claude/rules/tests.md).

The A-1..A-4 rows are active (activated for /implement T-0160). Only the O-A2 (3) ``card_like`` rows (a card number
joined to a date or a short number) are ``pending(task="T-0160")`` until the implementer revises ``card_like``. The
A-7 differential rows were never pending (they hold on BASE and must keep holding).
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

PENDING = pytest.mark.pending(task="T-0160")

START = "http://127.0.0.1:8765"
SECRET_KEYS = ("tag", "type", "autocomplete", "name", "id", "aria_label", "placeholder", "label")


def fence() -> Any:
    return importlib.import_module("local_astra.computer.fence")


def fn(name: str) -> Any:
    """A T-0160 fence function; its absence is an assertion (the pending contract), never an AttributeError."""
    f = getattr(fence(), name, None)
    assert f is not None, f"T-0160 A-1..A-4: fence.{name} does not exist yet"
    return f


# --- A-1: is_public_address -----------------------------------------------------------------------------------

NOT_PUBLIC = [
    # the task's table
    "127.0.0.1",
    "127.1.2.3",
    "0.0.0.0",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "100.64.0.1",
    "169.254.169.254",
    "198.18.0.1",
    "192.0.2.1",
    "203.0.113.5",
    "224.0.0.1",
    "255.255.255.255",
    "::",
    "::1",
    "fe80::1",
    "fc00::1",
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "64:ff9b::a00:1",
    "2002:7f00:1::",
    "2001:db8::1",
    # range edges and the rest of the A-1 list
    "10.255.255.255",
    "172.31.255.255",
    "192.168.255.255",
    "100.127.255.255",  # last CGNAT address (100.64/10)
    "198.19.255.255",  # last benchmarking address (198.18/15)
    "198.51.100.7",  # TEST-NET-2
    "0.1.2.3",  # "this network" 0/8
    "192.0.0.8",  # IETF protocol assignments 192.0.0/24
    "240.0.0.1",  # reserved 240/4
    "233.252.0.1",  # multicast (MCAST-TEST-NET)
    "239.255.255.250",  # multicast (SSDP)
    "fd12:3456:789a::1",  # ULA fd00::/8 inside fc00::/7
    "febf::1",  # last /16 of fe80::/10
    "fec0::1",  # deprecated site-local fec0::/10 (reviewer 2 T-2)
    "ff02::1",  # IPv6 multicast
    "::ffff:169.254.169.254",  # IPv4-mapped metadata endpoint
    "::ffff:192.168.1.1",
    "::ffff:7f00:1",  # IPv4-mapped 127.0.0.1 in hex
    "::ffff:100.64.0.1",
    "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
    "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
    "2002:a00:1::",  # 6to4 of 10.0.0.1
    "2002:c0a8:101::1",  # 6to4 of 192.168.1.1
    "2002:a9fe:a9fe::",  # 6to4 of 169.254.169.254
    # Teredo 2001:0::/32, server 65.54.227.120, client 127.0.0.1 obfuscated (XOR ffffffff -> 80ff:fffe)
    "2001:0:4136:e378:8000:63bf:80ff:fffe",
    # Teredo, client 10.0.0.1 obfuscated (-> f5ff:fffe)
    "2001:0:4136:e378:8000:63bf:f5ff:fffe",
    "::7f00:1",  # deprecated IPv4-compatible ::/96 (reserved ::/8)
]
PUBLIC = [
    "1.1.1.1",
    "93.184.215.14",
    "2606:4700:4700::1111",
    # just outside each refused range
    "8.8.8.8",
    "100.128.0.1",  # first address after CGNAT
    "198.20.0.1",  # first address after benchmarking
    "172.32.0.1",  # first address after 172.16/12
    "11.0.0.1",
    "2a00:1450:4001:82a::200e",
]
JUNK = [
    "",
    "1.2.3",
    "x",
    "1.1.1.1.1",
    "999.1.1.1",
    "1.1.1.1/32",
    "[2606:4700:4700::1111]",
    "example.com",
    "::ffff:999.0.0.1",
    "gggg::1",
]


def test_public_address_table() -> None:
    """A-1: an oracle table, both directions; unparseable input is false."""
    is_public = fn("is_public_address")
    wrongly_public = [a for a in NOT_PUBLIC if is_public(a) is not False]
    wrongly_private = [a for a in PUBLIC if is_public(a) is not True]
    junk_true = [a for a in JUNK if is_public(a) is not False]
    assert wrongly_public == [], (
        f"non-public addresses called public (guard absent): {wrongly_public}"
    )
    assert wrongly_private == [], f"public addresses refused (guard too wide): {wrongly_private}"
    assert junk_true == [], f"unparseable input not false: {junk_true}"


# --- A-2: web_url_allowed -------------------------------------------------------------------------------------

WEB_ALLOWED = [
    # (a) the start origin, exact scheme://host:port
    "http://127.0.0.1:8765/",
    "http://127.0.0.1:8765/web.html",
    "http://127.0.0.1:8765/web.html?x=1#y",
    # (b) https/wss, no userinfo, port absent or 443, a public host
    "https://example.com/",
    "https://example.com:443/",
    "https://example.com/a/b?q=1#f",
    "https://en.wikipedia.org/wiki/Mount_Everest",
    "https://www.example.co.uk/",
    "wss://example.com/socket",
    "wss://example.com:443/socket",
    "https://1.1.1.1/",
    "https://93.184.215.14:443/",
    "https://[2606:4700:4700::1111]/",
    "https://16843009/",  # single-decimal 1.1.1.1: an IP literal that is public
]
WEB_REFUSED = [
    # the start origin's host on another port, another scheme, or another spelling of loopback
    "http://127.0.0.1:8766/",
    "http://127.0.0.1/",
    "http://localhost:8765/",
    "http://[::1]:8765/",
    "https://127.0.0.1:8765/",
    "ws://127.0.0.1:8766/",
    "http://127.0.0.1:8765@evil.example/",
    # the adversarial core: numeric host forms of loopback and private addresses
    "https://127.0.0.1/",
    "https://2130706433/",
    "https://0x7f.1/",
    "https://0177.0.0.1/",
    "https://127.1/",
    "https://0x7f000001/",
    "https://017700000001/",
    "https://[::ffff:7f00:1]/",
    "https://[::ffff:127.0.0.1]/",
    "https://[::1]/",
    "https://0.0.0.0/",
    "https://0/",
    "https://10.0.0.1/",
    "https://167772161/",  # 10.0.0.1
    "https://3232235777/",  # 192.168.1.1
    "https://169.254.169.254/",
    "https://0251.0376.0251.0376/",  # 169.254.169.254 in octal
    "https://0xa9.0xfe.0xa9.0xfe/",  # 169.254.169.254 in hex
    # reviewer 2 T-2: forms that slipped past a mutant
    "https://127.0.0.1./",  # trailing dot: still the IPv4 127.0.0.1, not a two-label name
    "https://0x7f.1./",
    "https://127.0.0.1\\.evil.com/",  # a backslash host (WHATWG: a path separator; here not a name either way)
    "https://0x64.0x40.0.1/",  # 100.64.0.1 (CGNAT): hex parts whose digits are all decimal
    "https://1.1.1.256/",  # out of range: WHATWG refuses the URL
    "https://4294967296/",  # 2**32: out of range
    "https://100.64.0.1/",
    "https://[fe80::1]/",
    "https://[fc00::1]/",
    "https://[::ffff:10.0.0.1]/",
    "https://[64:ff9b::a00:1]/",
    "https://[2002:7f00:1::]/",
    # userinfo, port, scheme
    "https://user:pw@example.com/",
    "https://user@example.com/",
    "https://example.com:443@10.0.0.1/",
    "https://example.com:8443/",
    "https://example.com:80/",
    "wss://example.com:8443/",
    "http://example.com/",
    "http://93.184.215.14/",
    "ws://example.com/",
    # names
    "https://localhost/",
    "https://LOCALHOST/",
    "https://foo.localhost/",
    "https://intranet/",
    "https://router.local/",
    "https://ROUTER.LOCAL/",
    "https://db.internal/",
    "https://nas.lan/",
    "https://printer.home.arpa/",
    "https://box.localdomain/",
    "https://site.test/",
    "https://example.123/",  # numeric last label
    "https://example.0x1f/",  # 0x... last label
    # other schemes
    "file:///etc/passwd",
    "file://localhost/etc/passwd",
    "chrome://settings",
    "about:config",
    "data:text/html,<h1>hi</h1>",
    "blob:https://example.com/0b1d-uuid",
    "javascript:alert(1)",
    "about:blank",  # O-A1 (2): A-2 is "true for exactly (a)/(b)"
    "ftp://example.com/",
    # junk
    "",
    "https://",
    "https:///nohost",
    "not a url",
]


def test_web_url_allowed_table() -> None:
    """A-2 (a) and (b), both directions, with the start origin ``http://127.0.0.1:8765``."""
    allowed = fn("web_url_allowed")
    wrongly_refused = [u for u in WEB_ALLOWED if allowed(u, START) is not True]
    wrongly_allowed = [u for u in WEB_REFUSED if allowed(u, START) is not False]
    assert wrongly_refused == [], f"allowed URLs refused (guard too wide): {wrongly_refused}"
    assert wrongly_allowed == [], f"refused URLs allowed (guard absent): {wrongly_allowed}"


def test_web_url_start_origin_is_the_argument_not_a_constant() -> None:
    """A-2 (a): the exact-origin rule follows ``start_origin``; another port's start pages are not allowed."""
    allowed = fn("web_url_allowed")
    assert allowed("http://127.0.0.1:9100/web.html", "http://127.0.0.1:9100") is True
    assert allowed("http://127.0.0.1:8765/web.html", "http://127.0.0.1:9100") is False
    assert allowed("http://127.0.0.1:9100/web.html", START) is False


#: Reviewer 2 T-2: tabs, newlines, percent-encoding and IDN, pinned as the implementation at a1e58ea3 handles them
#: (the spec is silent). Tab and newline are stripped first, as WHATWG and ``urlsplit`` both do, so the host that
#: remains is judged; a stripped loopback literal stays refused. Percent-encoded, non-ASCII (raw IDN, full-width
#: digits, the ideographic full stop) hosts are refused: Chromium canonicalises them before routing, so the fence
#: never needs to allow the raw form, and refusing it closes a decoding mismatch. Punycode is an ordinary name.
WEB_ENCODING_ROWS = [
    ("https://exa\tmple.com/", True),
    ("https://example.com\t/", True),
    ("https://exa\nmple.com/", True),
    ("https://127.0.0.1\t/", False),
    ("https://12\t7.0.0.1/", False),
    ("https://ex%61mple.com/", False),
    ("https://%31%32%37.0.0.1/", False),
    ("https://127.0.0.1%2f/", False),
    ("https://bücher.example/", False),
    ("https://xn--bcher-kva.example/", True),
    ("https://１２７.0.0.1/", False),
    ("https://example。com/", False),
]


def test_web_url_tabs_percent_and_idn() -> None:
    allowed = fn("web_url_allowed")
    got = [(u, allowed(u, START)) for u, _ in WEB_ENCODING_ROWS]
    assert got == WEB_ENCODING_ROWS


def test_web_url_out_of_range_ipv4_is_false_and_does_not_raise() -> None:
    """Reviewer 2 T-2: WHATWG refuses an IPv4 host with a part out of range; the fence returns False, never raises."""
    allowed = fn("web_url_allowed")
    for url in (
        "https://1.1.1.256/",
        "https://4294967296/",
        "https://0x100000000/",
        "https://256.0.0.1/",
    ):
        assert allowed(url, START) is False, url


# --- A-3: card_like -------------------------------------------------------------------------------------------

CARD_TRUE = [
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "pay with 5555555555554444 now",
    "378282246310005",  # 15-digit Amex test number
    "4222222222222",  # 13-digit Visa test number
    "6011111111111117",  # Discover test number
    "my card is 4111 1111 1111 1111, thanks",
    # reviewer 2 T-3, boundaries (Luhn worked offline on each literal)
    "4111111111111111110",  # 19 digits, the longest candidate
    # 19 digits, Luhn-valid as a whole, and no 13-18-digit prefix or suffix is: only a 19-digit candidate finds it
    "4088995790328921840",
    # 20 digits: the 15-digit Amex test number is the only Luhn-valid 13-19 window aligned to either end (non-zero
    # padding, since leading zeros would keep every longer end window valid too)
    "54523378282246310005",
    "37828224631000515767",
    "55555555555544440000",  # 20 digits: a card of digits above 4 (doubling must subtract 9) aligned to the start
]
CARD_FALSE = [
    "4111111111111112",  # 16 digits, Luhn-invalid
    "4111 1111 1111 1112",
    "+33 6 12 34 56 78",  # phone: 11 digits
    "0612345678",  # 10-digit phone
    "2026-10-26",
    "411111111111",  # 12 digits
    "4111  1111 1111 1111",  # a double space breaks the run into 4 + 12
    "4111--1111-1111-1111",
    "",
    "hello",
    "order 12345",
    # reviewer 2 T-3, boundaries
    "411111111117",  # 12 digits and Luhn-valid: below the 13-digit floor
    "17953304135256012301",  # 20 digits, Luhn-valid as a whole, every 13-19 window aligned to either end fails
]


def test_card_like_table() -> None:
    """A-3: 13-19 digits, single space/hyphen separators, Luhn-valid; both directions."""
    card_like = fn("card_like")
    missed = [t for t in CARD_TRUE if card_like(t) is not True]
    false_hits = [t for t in CARD_FALSE if card_like(t) is not False]
    assert missed == [], f"card-like text not caught (guard absent): {missed}"
    assert false_hits == [], f"non-card text caught (guard too wide): {false_hits}"


#: O-A2 (3), superseding O-A1 (5): a card number joined to a date, a short number or extra digits is still caught.
#: Candidates are adjacent single-separator groups totalling 13-19 digits, and for a plain run over 19 digits the
#: 13-19-digit windows aligned to its start or end.
CARD_JOINED_TRUE = [
    "2026-10-26 4111 1111 1111 1111",  # a date joined by a single space to the four groups
    "12 4111111111111111",  # a short number joined to one 16-digit group
    "41111111111111110000",  # 20 digits: the card is the window aligned to the start
    "00004111111111111111",  # 20 digits: the card is the window aligned to the end
]


def test_card_like_joined_groups() -> None:
    """O-A2 (3): the new true rows; the A-3 false rows (phone, date, 12 digits, Luhn-invalid 16) stay false."""
    card_like = fn("card_like")
    missed = [t for t in CARD_JOINED_TRUE if card_like(t) is not True]
    assert missed == [], f"card number joined to other digits not caught (O-A2 (3)): {missed}"
    false_hits = [t for t in CARD_FALSE if card_like(t) is not False]
    assert false_hits == [], f"non-card text caught (guard too wide): {false_hits}"


def test_card_like_is_linear_on_long_input() -> None:
    """A-3 "linear", measured the T-0129 way: traced line events in ``fence.py`` (exact) and ``thread_time`` (min of
    3, gc off, sizes interleaved) at n and 4n. Linear predicts a ratio of 4, quadratic 16; the bar is 8 for line
    events and 10 for CPU time. Inputs are the adversarial shapes: a long run of single-spaced digits (every window a
    candidate), a hyphen run ending in a near miss, one long digit run, and two-digit groups."""
    shapes = [
        lambda n: "1 " * n,
        lambda n: "1-" * n + "x",
        lambda n: "4" * (2 * n),
        lambda n: "12 " * n + "a",
    ]
    for make in shapes:
        _assert_card_like_linear(make, 2_000, 8_000)


def test_card_like_is_linear_on_a_long_single_spaced_run() -> None:
    """A-3 "linear", reviewer 2 T-1: ``"1 " * n`` at n = 10k and 40k. At 2k/8k a C-level quadratic (a slice of the
    remaining groups per start) hides under the per-call overhead; at 40k its CPU ratio is about 16 against the bar
    of 10. Pending until the index-loop fix (I-1) lands."""
    _assert_card_like_linear(lambda n: "1 " * n, 10_000, 40_000)


def _assert_card_like_linear(make: Any, small: int, big: int) -> None:
    import gc

    from tests.unit._work_count import line_events

    card_like = fn("card_like")
    root = fence().__file__
    a, b = make(small), make(big)
    ev_a = line_events(lambda: card_like(a), root)
    ev_b = line_events(lambda: card_like(b), root)
    assert ev_b <= 8 * max(ev_a, 50), f"card_like line events {ev_a} -> {ev_b} for 4x input"
    best = {small: float("inf"), big: float("inf")}
    gc.disable()
    try:
        for _ in range(3):
            for n, text in ((small, a), (big, b)):
                t0 = time.thread_time()
                card_like(text)
                best[n] = min(best[n], time.thread_time() - t0)
    finally:
        gc.enable()
    assert best[big] <= 10 * max(best[small], 2e-3), (
        f"card_like CPU {best} for 4x input ({small} -> {big})"
    )


# --- A-4: secret_field ----------------------------------------------------------------------------------------


def attrs(**kw: str) -> dict[str, str]:
    out = dict.fromkeys(SECRET_KEYS, "")
    out["tag"] = "input"
    out["type"] = "text"
    out.update(kw)
    return out


SECRET_ROWS: list[tuple[dict[str, str], str | None]] = [
    # password
    (attrs(type="password"), "password"),
    (attrs(type="PASSWORD"), "password"),
    (attrs(autocomplete="current-password"), "password"),
    (attrs(autocomplete="new-password"), "password"),
    (attrs(autocomplete="section-login Current-Password"), "password"),
    # card: autocomplete tokens
    (attrs(autocomplete="cc-number"), "card"),
    (attrs(autocomplete="billing cc-csc"), "card"),
    (attrs(autocomplete="cc-exp"), "card"),
    (attrs(autocomplete="CC-EXP-MONTH"), "card"),
    (attrs(autocomplete="shipping cc-exp-year"), "card"),
    # card: name / id, lowercased, non-alphanumerics stripped
    (attrs(name="cardNumber"), "card"),
    (attrs(name="card-number"), "card"),
    (attrs(id="cc_number"), "card"),
    (attrs(id="cc-num"), "card"),
    (attrs(name="CardNum"), "card"),
    (attrs(name="cvv2"), "card"),
    (attrs(id="checkout_cvc"), "card"),
    (attrs(name="csc"), "card"),
    (attrs(id="security-code"), "card"),
    (attrs(name="card_expiry"), "card"),
    (attrs(id="ccExp"), "card"),
    (attrs(name="exp-date"), "card"),
    # card: aria_label / placeholder / label substrings
    (attrs(aria_label="Card number"), "card"),
    (attrs(placeholder="Security code"), "card"),
    (attrs(label="CVC"), "card"),
    (attrs(label="Enter your CVV"), "card"),
    (attrs(placeholder="1234 1234 1234 1234", aria_label="Credit CARD NUMBER"), "card"),
    # None: the task's negative rows and their neighbours
    (attrs(name="giftcard_search"), None),
    (attrs(name="postcode"), None),
    (attrs(autocomplete="cc-name"), None),
    (attrs(type="search"), None),
    (attrs(autocomplete="one-time-code"), None),  # H-T160-5
    (attrs(autocomplete="username", name="email"), None),
    (attrs(type="tel", name="phone"), None),
    (attrs(aria_label="Name on card"), None),
    (attrs(placeholder="Search"), None),
    (attrs(tag="textarea", type="", name="message"), None),
    (attrs(), None),
]


def test_secret_field_table() -> None:
    """A-4: the positive and negative rows, case-insensitive."""
    secret_field = fn("secret_field")
    wrong = [(a, want, got) for a, want in SECRET_ROWS if (got := secret_field(a)) != want]
    assert wrong == [], "secret_field misclassified (attrs, want, got): " + repr(
        [({k: v for k, v in a.items() if v}, w, g) for a, w, g in wrong]
    )


# --- totality (Hypothesis) ------------------------------------------------------------------------------------

_url_ish = st.one_of(
    st.text(max_size=60),
    st.builds(
        lambda s, h, p: f"{s}://{h}{p}",
        st.sampled_from(["http", "https", "ws", "wss", "file", "data", "HTTPS", ""]),
        st.text(max_size=30),
        st.sampled_from(["", "/", ":443/", ":0/", ":99999/", ":-1/", ":x/", "@", "/?#"]),
    ),
)
_addr_ish = st.one_of(
    st.text(max_size=40),
    st.ip_addresses().map(str),
    st.text(alphabet="0123456789abcdefx.:[]%", max_size=40),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    addr=_addr_ish,
    url=_url_ish,
    origin=st.one_of(st.just(START), st.text(max_size=30)),
    text=st.text(max_size=80),
)
def test_fence_web_total(addr: str, url: str, origin: str, text: str) -> None:
    """A-1..A-4: total over strings, never raises, and returns the declared types."""
    assert fn("is_public_address")(addr) in (True, False)
    assert fn("web_url_allowed")(url, origin) in (True, False)
    assert fn("card_like")(text) in (True, False)
    assert fn("secret_field")(dict.fromkeys(SECRET_KEYS, text)) in ("password", "card", None)


@settings(max_examples=200, deadline=None)
@given(ip=st.ip_addresses())
def test_public_address_agrees_with_independent_oracle_on_plain_ipv4(ip: Any) -> None:
    """A-1 against an independent oracle on plain IPv4 only: ``ipaddress``'s ``is_global`` minus the A-1 extras.
    (IPv6 transition forms are covered by the table, where ``is_global`` is not the spec.)"""
    import ipaddress

    if ip.version != 4:
        return
    # registry entries whose global-ness changed across Python point releases: not pinned by A-1, left out
    if any(ip in ipaddress.ip_network(n) for n in ("192.0.0.0/24", "192.88.99.0/24")):
        return

    extra_refused = [
        ipaddress.ip_network(n) for n in ("100.64.0.0/10", "198.18.0.0/15", "224.0.0.0/4")
    ]
    want = ip.is_global and not any(ip in n for n in extra_refused)
    assert fn("is_public_address")(str(ip)) is want, str(ip)


# --- A-7: the existing rules are unchanged (not pending: they hold on BASE) ------------------------------------


def test_existing_loopback_rule_unchanged() -> None:
    """A-7 differential: ``is_loopback_url`` gives every T-0111c row its BASE answer, and ``egress_record`` and the
    T-0111g exact-origin ``runtime.url_allowed`` keep theirs (hand table)."""
    from tests.unit.test_computer_fence import ALLOWED, REFUSED

    f = fence()
    assert [u for u in ALLOWED if f.is_loopback_url(u) is not True] == []
    assert [u for u in REFUSED if f.is_loopback_url(u) is not False] == []
    # the web hosts are still not loopback: web mode is a new predicate, not a widened old one
    for url in ("https://example.com/", "https://1.1.1.1/", "wss://example.com/"):
        assert f.is_loopback_url(url) is False, url
    assert f.egress_record("https://u:p@Example.COM:8443/p?q#f") == {
        "host": "example.com",
        "scheme": "https",
    }

    from local_astra.computer import runtime

    origins = frozenset({START})
    rows = [
        ("http://127.0.0.1:8765/web.html", True),
        ("http://127.0.0.1:8766/", False),
        ("http://localhost:8765/", False),
        ("ws://127.0.0.1:8765/", False),
        ("https://example.com/", False),
    ]
    assert [(u, runtime.url_allowed(u, origins)) for u, _ in rows] == rows
    assert runtime.url_allowed("http://localhost:1420/", None) is True
    assert runtime.url_allowed("https://example.com/", None) is False
