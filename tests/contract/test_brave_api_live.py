"""T-0163 criterion 11: the opt-in live rows for the Brave Search API backend. **They reach the network** and use
James's real key; opt-in only (``make search-live``).

``net``-marked and skipped unless ``ASTRA_LIVE_SEARCH=1`` **and** a child process running the production loader
(``brave_key.load_key()`` at the default path) prints ``configured``. The test process never opens the key file and
never holds the key (H-T163-5): every child prints only counts, hosts and status words.

- ``test_three_queries_and_a_chained_one``: under ``strace -f -qq -e trace=connect``, a child runs the three fixed
  queries through ``BraveApiSearch`` and one through ``FallbackSearch`` whose primary always raises
  ``SearchFailed("error")``; each returns at least one https result; the chained one reports
  ``last_engine == "brave_api"``; every AF_INET/AF_INET6 connect goes to loopback, a ``/etc/resolv.conf``
  nameserver on 53, or an address ``api.search.brave.com`` resolves to now.
- ``test_cancelled_real_search_exits_promptly``: a real search cancelled at once, then ``aclose``: the child exits
  within 7 s.

Report-only regime: Brave Search API v1 web search, httpx 0.28.1, client timeout 5 s, ``count=5``,
``safesearch=moderate``, three fixed queries plus one chained, n = 1 each, unseeded, proxies unset.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
ENV = "ASTRA_LIVE_SEARCH"
QUERIES = ("weather in Paris", "python 3.11 release date", "Eiffel Tower height")
CHAINED = "Mont Blanc height"
PROXY_VARS = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
API_HOST = "api.search.brave.com"
EXIT_BOUND_S = 7


def _net_mark() -> list[pytest.MarkDecorator]:
    ini = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"][
        "ini_options"
    ]
    return (
        [pytest.mark.net] if any(m.split(":")[0].strip() == "net" for m in ini["markers"]) else []
    )


pytestmark = [
    pytest.mark.contract,
    *_net_mark(),
    pytest.mark.skipif(
        os.environ.get(ENV) != "1",
        reason=f"live Brave Search API: set {ENV}=1 to run (make search-live); it uses the real key and the network",
    ),
]


def _env(base: dict[str, str]) -> dict[str, str]:
    """The children's env: ``base`` (the root conftest's ``real_home_env``, T-0175 criterion 7: ``HOME`` is the
    pre-switch home, so the production loader finds the real key; the escape fails unless ``ASTRA_LIVE_SEARCH=1``)
    without proxies."""
    env = {k: v for k, v in base.items() if k not in PROXY_VARS}
    env["PYTHONPATH"] = str(REPO)
    return env


@pytest.fixture
def live_env(real_home_env: dict[str, str]) -> dict[str, str]:
    """T-0175: every child of this module goes through the session jail's one escape (function scope: the guard
    permits the real home only while a row of this module runs)."""
    return _env(real_home_env)


_STATUS_SCRIPT = r"""
from local_astra.search import brave_key
print(brave_key.load_key().status, flush=True)
"""


@pytest.fixture
def configured(live_env: dict[str, str]) -> None:
    """Skip unless the production loader, in a child, says ``configured``. Only the status word crosses back."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _STATUS_SCRIPT],
            cwd=REPO,
            env=live_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the loader child did not return")
    if proc.returncode != 0 and "No module named 'local_astra.search.brave_key'" in proc.stderr:
        raise NotImplementedError("T-0163: local_astra.search.brave_key does not exist yet")
    words = proc.stdout.strip().splitlines()
    if words != ["configured"]:
        pytest.skip(
            f"the Brave Search API key file is not configured ({words[-1:] or proc.returncode})"
        )


def _nameservers() -> set[str]:
    try:
        text = Path("/etc/resolv.conf").read_text(encoding="utf-8")
    except OSError:
        return set()
    return {
        parts[1].split("%")[0]
        for line in text.splitlines()
        if len(parts := line.split()) >= 2 and parts[0] == "nameserver"
    }


def _resolved(hosts: frozenset[str]) -> set[str]:
    out: set[str] = set()
    for h in hosts:
        try:
            infos = socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            continue
        out.update(str(ipaddress.ip_address(i[4][0].split("%")[0])) for i in infos)
    return out


_V4 = re.compile(r'sa_family=AF_INET, sin_port=htons\((\d+)\), sin_addr=inet_addr\("([^"]+)"\)')
_V6 = re.compile(r'sa_family=AF_INET6, sin6_port=htons\((\d+)\).*?inet_pton\(AF_INET6, "([^"]+)"')


def _connects(trace: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for line in trace.splitlines():
        if "connect(" not in line:
            continue
        if (m := _V4.search(line)) or (m := _V6.search(line)):
            out.append((m.group(2), int(m.group(1))))
    return out


_QUERIES_SCRIPT = r"""
import asyncio, json, sys
from urllib.parse import urlsplit
from local_astra.contracts.protocols import SearchFailed
from local_astra.search import brave_key
from local_astra.search.brave_api import BraveApiSearch
from local_astra.search.fallback import FallbackSearch


class Refuses:
    last_engine = None

    async def search(self, query):
        raise SearchFailed("error", "No results found.")

    async def aclose(self):
        pass


def summary(r):
    return {"n": len(r.results), "https": sum(x.url.startswith("https://") for x in r.results),
            "hosts": sorted({urlsplit(x.url).hostname or "" for x in r.results})[:5]}


async def main():
    loaded = brave_key.load_key()
    out = []
    b = BraveApiSearch(loaded.key)
    for q in json.loads(sys.argv[1]):
        try:
            out.append({"query": q, **summary(await b.search(q))})
        except Exception as exc:
            out.append({"query": q, "error": type(exc).__name__, "kind": getattr(exc, "kind", None)})
    await b.aclose()
    chain = FallbackSearch(Refuses(), BraveApiSearch(loaded.key))
    try:
        r = await chain.search(sys.argv[2])
        out.append({"query": sys.argv[2], "chained": True, "engine": chain.last_engine, **summary(r)})
    except Exception as exc:
        out.append({"query": sys.argv[2], "chained": True, "error": type(exc).__name__,
                    "kind": getattr(exc, "kind", None)})
    await chain.aclose()
    print("RESULT " + json.dumps(out), flush=True)

asyncio.run(main())
"""


def test_three_queries_and_a_chained_one(
    configured: None, live_env: dict[str, str], tmp_path: Path
) -> None:
    import importlib

    strace = shutil.which("strace")
    if strace is None:
        pytest.skip("strace is not on PATH: the connect audit needs it (apt install strace)")
    api = importlib.import_module("local_astra.search.brave_api")
    assert frozenset({API_HOST}) == api.ALLOWED_HOSTS
    trace_file = tmp_path / "connect.trace"
    proc = subprocess.run(
        [
            strace,
            "-f",
            "-qq",
            "-e",
            "trace=connect",
            "-o",
            str(trace_file),
            sys.executable,
            "-c",
            _QUERIES_SCRIPT,
            json.dumps(QUERIES),
            CHAINED,
        ],
        cwd=REPO,
        env=live_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    results: list[dict[str, Any]] = json.loads(line[len("RESULT ") :])
    print("report:", results)  # counts, hosts and status words only
    _assert_connects_allowlisted(api.ALLOWED_HOSTS, trace_file)
    failed = [r for r in results if "error" in r]
    assert failed == [], f"searches that failed: {failed}"
    for r in results:
        assert r["https"] >= 1, r
    chained = [r for r in results if r.get("chained")]
    assert len(chained) == 1 and chained[0]["engine"] == "brave_api", chained


def _assert_connects_allowlisted(hosts: frozenset[str], trace_file: Path) -> None:
    allowed = _resolved(hosts)
    nameservers = _nameservers()
    stray = []
    for addr, port in _connects(trace_file.read_text(encoding="utf-8")):
        ip = ipaddress.ip_address(addr)
        if ip.is_loopback or (str(ip) in nameservers and port == 53) or str(ip) in allowed:
            continue
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None and (mapped.is_loopback or str(mapped) in allowed):
            continue
        stray.append((addr, port))
    assert stray == [], f"connects outside {sorted(hosts)} ({sorted(allowed)}): {stray}"


_CANCEL_SCRIPT = r"""
import asyncio
from local_astra.search import brave_key
from local_astra.search.brave_api import BraveApiSearch

async def main():
    b = BraveApiSearch(brave_key.load_key().key)
    t = asyncio.ensure_future(b.search("weather in Paris"))
    await asyncio.sleep(0)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    await b.aclose()

asyncio.run(main())
print("returned", flush=True)
"""


def test_cancelled_real_search_exits_promptly(configured: None, live_env: dict[str, str]) -> None:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CANCEL_SCRIPT],
            cwd=REPO,
            env=live_env,
            capture_output=True,
            text=True,
            timeout=EXIT_BOUND_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"did not exit within {EXIT_BOUND_S} s after cancel + aclose") from exc
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "returned" in proc.stdout
    assert time.monotonic() - start < EXIT_BOUND_S
