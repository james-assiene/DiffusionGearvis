"""T-0111c criterion 14 (pure half): ``local_astra/computer/fence.py``'s loopback origin check and egress record.

Names pinned here (the task names the behaviours, not the functions): ``fence.is_loopback_url(url) -> bool`` and
``fence.egress_record(url) -> {"host": ..., "scheme": ...}``. Both directions of the guard are rows (a guard needs a
row for being too wide as well as one for being absent, .claude/rules/tests.md): the allowlist is exactly
``127.0.0.1``, ``localhost`` and ``[::1]`` over http(s); look-alike hosts, the unspecified address and non-network
schemes are refused (reviewer mutation 7: allow ``localhost.example``).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def fence() -> Any:
    try:
        return importlib.import_module("local_astra.computer.fence")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("local_astra.computer"):
            raise AssertionError(f"T-0111c: {exc.name} does not exist yet") from exc
        raise


ALLOWED = [
    "http://127.0.0.1/",
    "http://127.0.0.1:8000/tests/fixtures/browser/grid.html?x=1#y",
    "http://localhost/",
    "http://localhost:1420/index.html",
    "http://[::1]/",
    "http://[::1]:5000/a/b",
]
REFUSED = [
    "http://127.0.0.1.nip.io/",
    "http://127.0.0.1.nip.io:8000/grid.html",
    "http://localhost.example/",
    "http://localhost.example:1420/",
    "http://0.0.0.0/",
    "http://0.0.0.0:8000/",
    "file:///etc/passwd",
    "file://localhost/etc/passwd",
    "chrome://settings",
    "chrome://version/",
    "data:text/html,<h1>hi</h1>",
    "data:text/html;base64,PGgxPmhpPC9oMT4=",
    "http://198.51.100.7/",
    "http://example.invalid/",
    "http://127.0.0.1@evil.example/",
    "http://localhost:80@evil.example/x",
    "http://evil.example/127.0.0.1",
    "http://evil.example/?next=http://127.0.0.1/",
]


def test_origin_check_loopback_only() -> None:
    f = fence()
    wrong_allowed = [u for u in ALLOWED if f.is_loopback_url(u) is not True]
    wrong_refused = [u for u in REFUSED if f.is_loopback_url(u) is not False]
    assert wrong_allowed == [], f"loopback origins refused (too wide): {wrong_allowed}"
    assert wrong_refused == [], f"non-loopback origins allowed (absent): {wrong_refused}"


@pytest.mark.parametrize(
    ("url", "want"),
    [
        (
            "http://198.51.100.7/t0111c-secret-path/x?token=t0111c-secret-query",
            {"host": "198.51.100.7", "scheme": "http"},
        ),
        ("http://198.51.100.7:8080/a?b=c#d", {"host": "198.51.100.7", "scheme": "http"}),
        (
            "ws://example.invalid/t0111c-secret-path?token=q",
            {"host": "example.invalid", "scheme": "ws"},
        ),
        ("https://user:pw@example.invalid/p?q=1", {"host": "example.invalid", "scheme": "https"}),
    ],
)
def test_egress_record_is_host_and_scheme_only(url: str, want: dict[str, str]) -> None:
    """Criterion 14: an egress attempt is logged as ``{host, scheme}``: no port, path, query, fragment or
    credentials."""
    rec = fence().egress_record(url)
    assert dict(rec) == want, rec
