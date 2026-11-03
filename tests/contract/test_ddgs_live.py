"""T-0111b criterion 9: the live rows. **They reach the network**; opt-in only.

``net``-marked, and skipped unless ``ASTRA_LIVE_SEARCH=1`` (``make search-live``), so neither the fast suite nor
``-m gpu`` under ``/launch`` ever searches. Report-only regime: ddgs/primp at the pin, engines ``ENGINES``
(duckduckgo first, then the keyless fallbacks of ruling 4a), region wt-wt, safesearch moderate, timeout 5 s per
engine, the three fixed queries, n = 1 each, unseeded, proxies unset.

- ``test_three_queries_answer_and_connect_only_to_allowed_hosts``: a fresh interpreter runs the three queries under
  ``strace -f -qq -e trace=connect``; each answers with at least one ``http(s)`` result from any engine (the one that
  answered is printed); a query every engine refuses fails the row naming each engine's refusal (each engine is then
  asked directly, once, to say why); every AF_INET/AF_INET6
  ``connect`` goes to loopback, to a ``/etc/resolv.conf`` nameserver on port 53, or to an address one of
  ``ALLOWED_HOSTS`` resolves to now. ``primp`` is Rust, so a Python socket patch would see nothing: hence strace.
- ``test_cancelled_real_search_exits_promptly``: a real search cancelled at once, then ``aclose``, then return; the
  process exits within ``CLIENT_TIMEOUT_S + 2`` s (ddgs may join its own executor at exit; the client timeout bounds
  it).
"""

from __future__ import annotations

import importlib
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
BACKEND = "local_astra.search.ddgs_backend"
ENV = "ASTRA_LIVE_SEARCH"
QUERIES = ("weather in Paris", "python 3.11 release date", "Eiffel Tower height")
PROXY_VARS = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


def _net_mark() -> list[pytest.MarkDecorator]:
    """``net`` once pyproject registers it (T-0111b's implementer); until then ``--strict-markers`` would refuse it,
    and the env gate below keeps the rows off the network either way."""
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
        reason=f"live web search: set {ENV}=1 to run (make search-live); it reaches the search engines in ENGINES",
    ),
]


def _mod() -> Any:
    try:
        return importlib.import_module(BACKEND)
    except ModuleNotFoundError as exc:
        if exc.name in (BACKEND, "local_astra.search"):
            raise NotImplementedError(f"T-0111b: {BACKEND} does not exist yet") from exc
        raise


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env["PYTHONPATH"] = str(REPO)
    return env


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
from local_astra.search import ddgs_backend as m
from local_astra.search.ddgs_backend import DdgsSearch


def refusals(q):
    # ruling (4a): when every engine refuses, name each engine's refusal. Each engine is asked once, directly,
    # with the backend's own bounds (still under strace: these connects are audited too).
    import ddgs
    out = {}
    for name in m.ENGINES:
        try:
            rows = ddgs.DDGS(proxy=None, timeout=m.CLIENT_TIMEOUT_S).text(
                q, backend=name, max_results=m.MAX_RESULTS, region="wt-wt", safesearch="moderate")
            out[name] = f"answered {len(rows)} rows when asked directly"
        except Exception as exc:
            out[name] = repr(exc)[:300]
    return out


async def main():
    b = DdgsSearch()
    out = []
    for q in json.loads(sys.argv[1]):
        try:
            r = await b.search(q)
            engine = getattr(r, "engine", None) or getattr(b, "last_engine", None)
            out.append({"query": q, "urls": [x.url for x in r.results], "engine": engine})
        except Exception as exc:
            out.append({"query": q, "error": repr(exc)[:300], "refusals": refusals(q)})
    await b.aclose()
    print("RESULT " + json.dumps(out), flush=True)

asyncio.run(main())
"""


def test_three_queries_answer_and_connect_only_to_allowed_hosts(tmp_path: Path) -> None:
    m = _mod()
    strace = shutil.which("strace")
    if strace is None:
        pytest.skip("strace is not on PATH: the connect audit needs it (apt install strace)")
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
        ],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    results = json.loads(line[len("RESULT ") :])
    # report-only: which engine answered each query (ruling 4a: any engine in ENGINES may)
    print("engines that answered:", {r["query"]: r.get("engine") for r in results})
    _assert_connects_allowlisted(m, trace_file)
    refused = {r["query"]: r["refusals"] for r in results if "error" in r}
    assert refused == {}, (
        f"every engine in {m.ENGINES} refused these queries; each engine's refusal: {refused}"
    )
    for r in results:
        assert any(u.startswith(("http://", "https://")) for u in r["urls"]), r


def _assert_connects_allowlisted(m: Any, trace_file: Path) -> None:
    allowed = _resolved(m.ALLOWED_HOSTS)
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
    assert stray == [], (
        f"connects outside ALLOWED_HOSTS={sorted(m.ALLOWED_HOSTS)} ({sorted(allowed)}): {stray}"
    )


_CANCEL_SCRIPT = r"""
import asyncio
from local_astra.search.ddgs_backend import DdgsSearch

async def main():
    b = DdgsSearch()
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


def test_cancelled_real_search_exits_promptly() -> None:
    m = _mod()
    bound = m.CLIENT_TIMEOUT_S + 2
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CANCEL_SCRIPT],
            cwd=REPO,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=bound,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"did not exit within {bound} s after cancel + aclose") from exc
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "returned" in proc.stdout
    assert time.monotonic() - start < bound
