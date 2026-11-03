"""The browser executor's pure checks (T-0111c): the loopback origin rule, the egress record, the control-character
rule and the profile-path refusal. No I/O beyond resolving a path; no network, no clock.

- ``is_loopback_url``: an ``http``/``https``/``ws``/``wss`` URL whose host is exactly ``127.0.0.1``, ``localhost`` or
  ``[::1]``, with no credentials. Look-alikes (``127.0.0.1.nip.io``, ``localhost.example``), the unspecified address
  and every other scheme (``file:``, ``chrome:``, ``data:``) are refused (criterion 14, H-T111c-E).
- ``egress_record``: what the egress log keeps of a blocked attempt, ``{"host", "scheme"}`` only; never a port,
  path, query, fragment or credentials.
- ``has_control_character``: U+0000-U+001F or U+007F; Playwright's ``keyboard.type`` would turn ``\\n`` into Enter
  (criterion 6; cua ``carl\\enter``).
- ``check_profile_dir``: a ``user_data_dir`` resolving inside the user's own browser profiles is refused (criterion
  13, H-T111b "never the user's session").

Web mode (T-0160 group A), pure and total (never raise):

- ``is_public_address``: a globally routable unicast IP; embedded IPv4 (mapped, NAT64, 6to4) must itself be public,
  Teredo never is (A-1).
- ``web_url_allowed``: the start origin exactly, or ``https``/``wss`` on port 443 without userinfo to a public IP
  literal (any WHATWG IPv4 form) or a dotted, non-local, non-numeric name (A-2).
- ``card_like``: a Luhn-valid run of 13-19 digits, single spaces or hyphens allowed between them (A-3).
- ``secret_field``: ``"password"``, ``"card"`` or ``None`` for a focused element's attributes (A-4).

Checkout (T-0192 A1), pure and total:

- ``checkout_url``: an ``http(s)`` URL whose path has a checkout or payment segment (``CHECKOUT_SEGMENTS``, after one
  percent-decode, lowercased, at most one page suffix removed), a ``buy`` segment right after ``gp``, or whose host's
  leftmost label is in ``CHECKOUT_HOST_LABELS``. The query and fragment are never read.

Submit stop (T-0206 A1), pure and total:

- ``irreversible_submit``: ``True`` when a press of the described control sends a form holding filled personal data
  (a name, email, phone, address or free-text message): (a) the control is not a navigating link; (b) the scope is not
  search-like; (c) a filled, visible personal field; (d) a send label (``SEND_LABELS``, ``SEARCH_LABELS`` winning) or
  a structural empty-label submit. A scope is a search when its filled fields are all search-like; a form-bound send
  is also exempt when a filled, visible search-like field (named in ``SEARCH_NAMES`` or ``type=search``) is in its
  form, and never by an empty or hidden one (review R-2, R-5). A formless send's scope is the control's nearest
  ancestor holding a text-like input (ruling on reading 6). Location fields (address, street, city, postcode, zip,
  country) are not personal data on their own: a locator search stays open (review R-3).
  At most ``MAX_FIELDS`` fields are read; labels are compared here and never returned.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

__all__ = [
    "CHECKOUT_HOST_LABELS",
    "CHECKOUT_SEGMENTS",
    "LOOPBACK_HOSTS",
    "MAX_FIELDS",
    "NETWORK_SCHEMES",
    "PERSONAL_AUTOCOMPLETE",
    "SEARCH_LABELS",
    "SEARCH_NAMES",
    "SEND_LABELS",
    "card_like",
    "check_profile_dir",
    "checkout_url",
    "egress_record",
    "has_control_character",
    "irreversible_submit",
    "is_loopback_url",
    "is_public_address",
    "secret_field",
    "web_url_allowed",
]

#: The only hosts the browser may reach (``urlsplit`` drops the brackets of ``[::1]``).
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: Schemes that open a connection; anything else is never an allowed origin.
NETWORK_SCHEMES = frozenset({"http", "https", "ws", "wss"})
#: The user's own browser profiles, relative to ``$HOME``.
USER_PROFILE_DIRS = (".config/google-chrome", ".config/chromium", ".mozilla")


def is_loopback_url(url: str) -> bool:
    """True only for a network-scheme URL on an allowlisted loopback host with no userinfo."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # a malformed port raises ValueError
    except ValueError:
        return False
    if parts.scheme.lower() not in NETWORK_SCHEMES:
        return False
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return False
    return host is not None and host.lower() in LOOPBACK_HOSTS


def egress_record(url: str) -> dict[str, str]:
    """``{"host", "scheme"}`` of an attempted URL, lowercased; nothing else of it is kept."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return {"host": "", "scheme": ""}
    return {"host": host.lower(), "scheme": parts.scheme.lower()}


def has_control_character(text: str) -> bool:
    """True when ``text`` contains a C0 control character or DEL."""
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in text)


def check_profile_dir(path: str | os.PathLike[str] | None) -> Path | None:
    """The resolved ``user_data_dir`` (``None`` stays ``None``); ``ValueError`` when it resolves inside one of the
    user's own browser profiles. ``~``, ``..`` and symlinks are resolved before the comparison."""
    if path is None:
        return None
    resolved = Path(os.path.realpath(Path(path).expanduser()))
    home = Path(os.path.realpath(Path("~").expanduser()))
    for rel in USER_PROFILE_DIRS:
        forbidden = Path(os.path.realpath(home / rel))
        if resolved == forbidden or resolved.is_relative_to(forbidden):
            raise ValueError(
                f"user_data_dir {resolved} is inside the user's browser profile {forbidden}; the executor uses "
                "its own ephemeral profile, never the user's session (T-0111c criterion 13)"
            )
    return resolved


# --------------------------------------------------------------------------------------------- web mode (T-0160)

_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.IPv6Network("64:ff9b:1::/48")
_BROADCAST = ipaddress.IPv4Address("255.255.255.255")


def is_public_address(ip: str) -> bool:
    """True only for a globally routable unicast address (A-1); unparseable input is false."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return _public(addr)


def _public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.scope_id is not None:
            return False
        if addr.ipv4_mapped is not None:
            return _public(addr.ipv4_mapped)
        if addr.teredo is not None or addr in _NAT64_LOCAL:
            return False
        if addr in _NAT64:
            return _public(ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF))
        if addr.sixtofour is not None:
            return _public(addr.sixtofour)
        if addr.is_site_local:
            return False
    elif addr == _BROADCAST:
        return False
    return addr.is_global and not (
        addr.is_loopback
        or addr.is_unspecified
        or addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
    )


#: Names that never leave the machine or the LAN (A-2).
_LOCAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
    ".localdomain",
    ".test",
)
_LABEL = re.compile(r"[a-z0-9_-]+")
_WEB_SCHEMES = frozenset({"https", "wss"})
_ORIGIN_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _whatwg_ipv4_number(part: str) -> int | None:
    """One WHATWG IPv4 part: ``0x`` hex (may be empty), leading-``0`` octal, else decimal; ``None`` when invalid."""
    if part[:2] in ("0x", "0X"):
        digits, base = part[2:], 16
        if digits == "":
            return 0
    elif len(part) > 1 and part[0] == "0":
        digits, base = part[1:], 8
    else:
        digits, base = part, 10
    if not digits.isascii() or not digits.isalnum():
        return None
    try:
        return int(digits, base)
    except ValueError:
        return None


def _whatwg_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """The WHATWG IPv4 parser (dotted, single number, hex, octal, short forms); ``None`` on failure."""
    parts = host.split(".")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()
    if not 1 <= len(parts) <= 4 or "" in parts:
        return None
    numbers = []
    for p in parts:
        n = _whatwg_ipv4_number(p)
        if n is None:
            return None
        numbers.append(n)
    if any(n > 255 for n in numbers[:-1]) or numbers[-1] >= 256 ** (5 - len(numbers)):
        return None
    value = numbers[-1]
    for i, n in enumerate(numbers[:-1]):
        value += n * 256 ** (3 - i)
    return ipaddress.IPv4Address(value)


def _ends_in_number(host: str) -> bool:
    """WHATWG "ends in a number": the last label (one trailing empty label dropped) is digits or ``0x``-prefixed."""
    labels = host.split(".")
    if len(labels) > 1 and labels[-1] == "":
        labels.pop()
    last = labels[-1]
    return (last.isascii() and last.isdigit()) or last[:2] in ("0x", "0X")


def _web_host_allowed(host: str) -> bool:
    if ":" in host:  # urlsplit drops the brackets of an IPv6 literal
        return is_public_address(host)
    if not host.isascii() or "%" in host:
        return False  # Chromium canonicalises IDN and percent-escapes before routing; anything else is refused
    if _ends_in_number(host):
        v4 = _whatwg_ipv4(host)
        return v4 is not None and _public(v4)
    name = host[:-1] if host.endswith(".") else host
    labels = name.split(".")
    if len(labels) < 2 or not all(_LABEL.fullmatch(label) for label in labels):
        return False
    return name != "localhost" and not name.endswith(_LOCAL_SUFFIXES)


def _origin_key(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` lowercased with the scheme's default port; ``None`` when unparseable."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if host is None or scheme not in _ORIGIN_PORTS:
        return None
    return scheme, host.lower(), port if port is not None else _ORIGIN_PORTS[scheme]


def web_url_allowed(url: str, start_origin: str) -> bool:
    """True for the start origin exactly (T-0111g's rule) or a public ``https``/``wss`` URL on port 443 (A-2); nothing else, ``about:blank`` included (O-A1 (2))."""
    if is_loopback_url(url):
        key = _origin_key(url)
        return key is not None and key == _origin_key(start_origin)
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return False
    if parts.scheme.lower() not in _WEB_SCHEMES or host is None:
        return False
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return False
    if port is not None and port != 443:
        return False
    return _web_host_allowed(host.lower())


_CARD_MIN, _CARD_MAX = 13, 19


def _luhn(digits: list[int]) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _digit_runs(text: str) -> list[list[list[int]]]:
    """Runs of digit groups, one pass: a group is consecutive digits; adjacent groups of one run are joined by
    exactly one space or hyphen. Any other character, or a second separator in a row, ends the run."""
    runs: list[list[list[int]]] = []
    run: list[list[int]] = []
    group: list[int] = []
    for c in text:
        if c.isdecimal():
            group.append(int(c))
        elif c in " -" and group:
            run.append(group)
            group = []
        else:
            if group:
                run.append(group)
                group = []
            if run:
                runs.append(run)
                run = []
    if group:
        run.append(group)
    if run:
        runs.append(run)
    return runs


_LUHN_DOUBLE = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)


def _concatenation_is_card(run: list[list[int]], start: int) -> bool:
    """Whether some concatenation ``run[start:j]`` of 13-19 digits passes Luhn. Luhn is kept incrementally (plain
    and doubled sums per index parity), so each extension costs its own digits only; at most 19 digits are read."""
    plain, doubled, length = [0, 0], [0, 0], 0
    for k in range(
        start, len(run)
    ):  # an index loop: slicing run[start:] would copy the tail per start
        group = run[k]
        if length + len(group) > _CARD_MAX:
            return False
        for d in group:
            plain[length % 2] += d
            doubled[length % 2] += _LUHN_DOUBLE[d]
            length += 1
        # the rightmost digit (index length-1) is not doubled; the other parity is
        if length >= _CARD_MIN and (plain[(length - 1) % 2] + doubled[length % 2]) % 10 == 0:
            return True
    return False


def card_like(text: str) -> bool:
    """True when ``text`` holds a card-like number (A-3, O-A2 (3), superseding O-A1 (5)). The candidates are the
    digit groups split on single spaces or hyphens and every concatenation of adjacent groups totalling 13-19
    digits, so a date or phone number joined to a card number does not hide it; a group of more than 19 digits
    contributes its 13-19-digit windows aligned to its start or end. Any candidate passing Luhn is card-like. Each
    concatenation reads at most 19 digits and each long group 14 windows, so the check is linear."""
    for run in _digit_runs(text):
        for start, group in enumerate(run):
            if len(group) > _CARD_MAX:
                for n in range(_CARD_MIN, _CARD_MAX + 1):
                    if _luhn(group[:n]) or _luhn(group[-n:]):
                        return True
            elif _concatenation_is_card(run, start):
                return True
    return False


_PASSWORD_AUTOCOMPLETE = frozenset({"current-password", "new-password"})
_CARD_AUTOCOMPLETE = frozenset({"cc-number", "cc-csc", "cc-exp", "cc-exp-month", "cc-exp-year"})
_CARD_IDS = (
    "cardnumber",
    "ccnumber",
    "ccnum",
    "cardnum",
    "cvc",
    "cvv",
    "csc",
    "securitycode",
    "cardexpiry",
    "ccexp",
    "expdate",
)
_CARD_TEXTS = ("card number", "security code", "cvc", "cvv")
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_SPACES = re.compile(r"\s+")


def secret_field(attrs: Mapping[str, str]) -> Literal["password", "card"] | None:
    """``"password"``, ``"card"`` or ``None`` for a focused element's attributes (A-4), case-insensitive. A missing
    or non-string value reads as empty."""

    def get(key: str) -> str:
        value: object = attrs.get(key)
        return value.lower() if isinstance(value, str) else ""

    tokens = set(get("autocomplete").split())
    if get("type").strip() == "password" or tokens & _PASSWORD_AUTOCOMPLETE:
        return "password"
    if tokens & _CARD_AUTOCOMPLETE:
        return "card"
    for key in ("name", "id"):
        ident = _NON_ALNUM.sub("", get(key))
        if any(word in ident for word in _CARD_IDS):
            return "card"
    for key in ("aria_label", "placeholder", "label"):
        text = _SPACES.sub(" ", get(key))
        if any(phrase in text for phrase in _CARD_TEXTS):
            return "card"
    return None


#: T-0192 A1 (H-T192-1): path segments that name a checkout or payment page.
CHECKOUT_SEGMENTS = frozenset(
    {"checkout", "checkouts", "checkoutnow", "payment", "payments", "pay"}
)
#: T-0192 A1 (c): a host whose leftmost DNS label is one of these is a checkout host.
CHECKOUT_HOST_LABELS = frozenset({"checkout", "pay", "payments"})
#: T-0192 A1 (a): at most one of these is removed from the end of a segment before it is compared.
_PAGE_SUFFIXES = (".html", ".htm", ".php", ".aspx", ".jsp")


def _segment(raw: str) -> str:
    """One path segment as A1 compares it: one percent-decode, lowercased, at most one page suffix removed."""
    seg = unquote(raw).lower()
    for suffix in _PAGE_SUFFIXES:
        if seg.endswith(suffix):
            return seg[: -len(suffix)]
    return seg


def checkout_url(url: str) -> bool:
    """T-0192 A1: ``True`` when ``url`` is an ``http(s)`` URL of a checkout page (module docstring); never raises."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    if parts.scheme.lower() not in ("http", "https"):
        return False
    if host and host.split(".", 1)[0] in CHECKOUT_HOST_LABELS:
        return True
    previous = ""
    for raw in parts.path.split("/"):
        seg = _segment(raw)
        if seg in CHECKOUT_SEGMENTS or (seg == "buy" and previous == "gp"):
            return True
        previous = seg
    return False


# --------------------------------------------------------------------------------------------- submit stop (T-0206)

#: T-0206 A1: at most this many fields of a scope are read (the driver lists no more; a longer list is cut here).
MAX_FIELDS = 64
#: T-0206 A1 (d), H-T206-1: a control whose normalised label begins (on a word boundary) with one of these sends.
SEND_LABELS = frozenset(
    {
        "submit", "send", "send message", "send request", "send enquiry", "send inquiry",
        "request quote", "request a quote", "request a free quote", "request estimate", "request an estimate",
        "get quote", "get a quote", "get my quote", "get a free quote", "get free quote", "get estimate",
        "place order", "place my order", "complete order", "complete purchase", "confirm order",
        "confirm booking", "confirm reservation",
        "book now", "reserve now", "pay", "pay now", "buy now",
        "sign up", "register", "create account", "subscribe", "apply now", "donate",
    }
)  # fmt: skip
#: T-0206 A1: a label beginning with one of these is a search, whatever ``SEND_LABELS`` says.
SEARCH_LABELS = frozenset(
    {"search", "find", "go", "look up", "find stores", "search flights", "show results"}
)
#: T-0206 A1 (b): a field whose ``name`` or ``id`` is exactly one of these is a search box.
SEARCH_NAMES = frozenset(
    {
        "q",
        "query",
        "search",
        "s",
        "k",
        "keyword",
        "keywords",
        "field-keywords",
        "term",
        "search_query",
    }
)
#: T-0206 A1 (c): ``autocomplete`` tokens that name personal data.
PERSONAL_AUTOCOMPLETE = frozenset(
    {
        "name", "given-name", "family-name", "email", "tel", "tel-national", "street-address", "address-line1",
        "address-line2", "postal-code", "address-level1", "address-level2", "organization",
    }
)  # fmt: skip
#: T-0206 A1 (c): a whole word of a field's name, id, label or placeholder that names personal data. The lookarounds
#: are single characters, so a search is linear in the text.
_PERSONAL_WORD = re.compile(
    r"(?<![a-z0-9])(?:name|email|e-mail|phone|tel|mobile|address|street|zip|postcode|postal|message|comments?)"
    r"(?![a-z0-9])"
)
#: T-0206 review R-3/R-6: location fields (a locator search, not a personal send when they are all that is filled).
_LOCATION_AUTOCOMPLETE = frozenset(
    {
        "street-address",
        "address-line1",
        "address-line2",
        "postal-code",
        "address-level1",
        "address-level2",
    }
)
_LOCATION_WORDS = frozenset({"address", "street", "city", "postcode", "postal", "zip", "country"})
#: T-0206 A1 (d): words of a label compared, at most.
_LABEL_WORDS = 6
_SEND_WORDS = tuple(tuple(label.split()) for label in SEND_LABELS)
_SEARCH_WORDS = tuple(tuple(label.split()) for label in SEARCH_LABELS)


def _text(value: object) -> str:
    return value.casefold() if isinstance(value, str) else ""


def _trim(word: str) -> str:
    """``word`` without leading or trailing characters that are not letters or digits (punctuation, arrows)."""
    i, j = 0, len(word)
    while i < j and not word[i].isalnum():
        i += 1
    while j > i and not word[j - 1].isalnum():
        j -= 1
    return word[i:j]


def _label_words(value: object) -> tuple[str, ...]:
    """A1 (d): the label casefolded, split on whitespace, each word trimmed of punctuation and arrows, empty words
    dropped, at most ``_LABEL_WORDS`` kept."""
    out: list[str] = []
    for raw in _text(value).split():
        word = _trim(raw)
        if word:
            out.append(word)
            if len(out) == _LABEL_WORDS:
                break
    return tuple(out)


def _begins(words: tuple[str, ...], lexicon: tuple[tuple[str, ...], ...]) -> bool:
    return any(words[: len(entry)] == entry for entry in lexicon)


def _search_named(field: Mapping[object, object]) -> bool:
    """A1 (b): a field whose ``name`` or ``id`` is in ``SEARCH_NAMES``."""
    return any(_text(field.get(key)).strip() in SEARCH_NAMES for key in ("name", "id"))


def _search_field(field: Mapping[object, object]) -> bool:
    """A1: a search-like field: ``type=search``, or named in ``SEARCH_NAMES``."""
    return _text(field.get("type")).strip() == "search" or _search_named(field)


def _personal(field: Mapping[object, object]) -> bool:
    """A1 (c) with review R-3: a textarea, ``type`` email/tel, a personal ``autocomplete`` token or a personal whole
    word in the ``name``/``id``/``label``/``placeholder``; never a search-like field. Location evidence (an address,
    street, city, postcode, zip or country) does not count: a form whose filled personal fields are all location
    fields is a locator search."""
    if _search_field(field):
        return False
    if _text(field.get("tag")).strip() == "textarea":
        return True
    if _text(field.get("type")).strip() in ("email", "tel"):
        return True
    tokens = set(_text(field.get("autocomplete")).split()) & PERSONAL_AUTOCOMPLETE
    if tokens - _LOCATION_AUTOCOMPLETE:
        return True
    return any(
        match.group(0) not in _LOCATION_WORDS
        for key in ("name", "id", "label", "placeholder")
        for match in _PERSONAL_WORD.finditer(_text(field.get(key)))
    )


def _filled(field: Mapping[object, object]) -> bool:
    return field.get("filled") is True


def _shown_filled(field: Mapping[object, object]) -> bool:
    return field.get("filled") is True and field.get("visible") is True


def irreversible_submit(facts: Mapping[str, object]) -> bool:
    """T-0206 A1: whether pressing ``facts["control"]`` sends a form with filled personal data (module docstring).
    Total over JSON-like values: anything malformed reads as absent, and an absent control or field list is
    ``False``."""
    given: object = facts  # the type is the contract; a caller may still hand any JSON-like value
    if not isinstance(given, Mapping):
        return False
    control = given.get("control")
    raw_fields = given.get("fields")
    if not isinstance(control, Mapping) or not isinstance(raw_fields, list):
        return False
    # (a) a navigating link only navigates
    if control.get("href_nav") is True:
        return False
    fields = [f for f in raw_fields[:MAX_FIELDS] if isinstance(f, Mapping)]
    # (b) search-like scope
    if given.get("search_scope") is True:
        return False
    filled = [f for f in fields if _filled(f)]
    if filled and all(_search_field(f) for f in filled):
        return False
    # a form-bound send: a filled, visible search box exempts it (R-2: an empty one does not; R-5: a hidden one never)
    if given.get("scope") == "form" and any(_shown_filled(f) and _search_field(f) for f in fields):
        return False
    # (c) filled, visible personal data
    if not any(_shown_filled(f) and _personal(f) for f in fields):
        return False
    # (d) the control sends
    words = _label_words(control.get("label"))
    if not words:
        return control.get("structural") is True
    return not _begins(words, _SEARCH_WORDS) and _begins(words, _SEND_WORDS)
