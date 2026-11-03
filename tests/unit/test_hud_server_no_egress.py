"""The transport stays out of the session, and the session out of the wire (criterion 14).

A blunt scan, in the shape of T-0007's ``tests/unit/test_session_no_egress.py`` (which this task
must leave green and unmodified — the criterion leans on it). Two directions:

* ``conductor/hud_server.py`` *is* the transport, so it may import ``websockets``; what it may
  not do is construct a concrete service client, name a non-loopback host, or carry a bind
  address that ``config.Services`` — whose validator refuses non-loopback (docs/02 §9) — never
  saw;
* ``session.py`` and ``executor.py`` gain a publisher and nothing else: no client library, no URL
  literal, and no ``await`` on a send inside the pump or inside a ``CommandExecutor`` handler
  (criterion 12's negative; ``.claude/rules/conductor.md``: 20 ms).

Textual on purpose: this is the fence that catches a ``websockets.connect()`` default argument
that no behavioural test over fakes would ever see.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.unit.test_session_no_egress import LIBRARY_FILES

REPO = Path(__file__).resolve().parents[2]

HUD_SERVER = "local_astra/conductor/hud_server.py"
SESSION_FILES = ("local_astra/conductor/session.py", "local_astra/conductor/executor.py")

#: Classes that *are* a service. The server is handed its frame sink and its session; it builds
#: neither, so none of these names may appear in it.
CONCRETE_CLIENTS = (
    "VllmClient",
    "MoshiSttClient",
    "PocketTtsClient",
    "PipeWireAudioIO",
    "SqliteMemory",
    "BrowserFrameSource",
    "AsyncClient",
    "ClientSession",
)

URL_RE = re.compile(r"\b(?:https?|wss?)://([^\s\"')]*)")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")
#: Wildcard binds: the exact mistake docs/03 E8 and criterion 22 are about.
WILDCARD_RE = re.compile(r"0\.0\.0\.0|\"::\"|'::'")


def source(name: str) -> str:
    path = REPO / name
    assert path.is_file(), f"T-0028 adds {name} (docs/tasks/T-0028.md, Owned files)"
    return path.read_text()


def code_only(body: str) -> str:
    """Drop docstrings — and **only** docstrings (T-0007 ruling A9).

    Prose may cite the endpoint the page points at; code may not carry a second source of truth.
    """
    tree = ast.parse(body)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            first.value.value = ""
    return ast.unparse(tree)


#: The two insertions the scans below must catch, as a synthetic module. Proving the scan on this
#: sample is what stops it from passing vacuously over a file that happens to be clean today.
BAD_SAMPLE = (
    '"""A docstring may cite ws://127.0.0.1:8000/v1/realtime."""\n'
    '_EP = "ws://10.0.0.5/"\n'
    "class Server:\n"
    "    def __init__(self, llm=VllmClient()):\n"
    '        self.host = "0.0.0.0"\n'
    "        self.port = 8000\n"
)


def url_offenders(body: str) -> list[str]:
    out: list[str] = []
    for lineno, line in enumerate(code_only(body).splitlines(), start=1):
        for match in URL_RE.finditer(line):
            if not match.group(1).startswith(LOOPBACK_HOSTS):
                out.append(f"{lineno}: {line.strip()[:90]}")
    return out


def client_offenders(body: str) -> list[str]:
    code = code_only(body)
    return [client for client in CONCRETE_CLIENTS if client in code]


def wildcard_offenders(body: str) -> list[str]:
    return [
        f"{lineno}: {line.strip()[:90]}"
        for lineno, line in enumerate(code_only(body).splitlines(), start=1)
        if WILDCARD_RE.search(line)
    ]


def awaited_sends(body: str) -> list[str]:
    """``await <something>.publish(...)`` / ``.send(...)`` — the pump waiting for the wire."""
    out: list[str] = []
    for node in ast.walk(ast.parse(body)):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        label = getattr(callee, "attr", getattr(callee, "id", ""))
        if "publish" in label or label in ("send", "send_str", "send_bytes", "write"):
            out.append(f"{node.lineno}: await …{label}(…)")
    return out


def test_the_scans_catch_the_two_insertions_they_exist_for():
    """The scans' own guard: prove them on a sample that is deliberately wrong.

    ``"ws://10.0.0.5/"`` and ``VllmClient()`` are exactly the two edits the reviewer uses to
    check that this module is not vacuous; the docstring's loopback URL must stay invisible.
    """
    assert url_offenders(BAD_SAMPLE), "a remote ws:// literal in code must be flagged"
    assert client_offenders(BAD_SAMPLE) == ["VllmClient"], client_offenders(BAD_SAMPLE)
    assert wildcard_offenders(BAD_SAMPLE), "a 0.0.0.0 bind must be flagged"
    assert not url_offenders('"""ws://127.0.0.1:8000/v1/realtime."""\nx = 1\n'), (
        "a docstring is prose, not a second source of truth"
    )
    # …and the scans then run on the real file, which this task writes.
    assert source(HUD_SERVER)


def test_the_server_names_no_client_and_no_non_loopback_host():
    """Criterion 14: the transport constructs no service and names no host it did not get."""
    body = source(HUD_SERVER)
    assert not client_offenders(body), (
        "criterion 14: hud_server.py constructs no concrete service client — the session and the "
        f"frame sink are injected: {client_offenders(body)}"
    )
    assert not url_offenders(body), (
        f"criterion 14: no non-loopback URL literal in the transport: {url_offenders(body)}"
    )
    assert not wildcard_offenders(body), (
        "criterion 1/22: the listener binds 127.0.0.1, never 0.0.0.0 and never ::: "
        f"{wildcard_offenders(body)}"
    )


def test_the_bind_address_comes_only_from_config_services():
    """Criterion 1 and 14: the port is ``config.Services.conductor_port``, not a literal 8000."""
    body = code_only(source(HUD_SERVER))
    assert "conductor_port" in body, (
        "criterion 1: the port comes from config.Services.conductor_port (whose validator is the "
        "loopback fence, docs/02 §9); hud_server.py never mentions it"
    )
    literals = [
        f"{lineno}: {line.strip()[:90]}"
        for lineno, line in enumerate(body.splitlines(), start=1)
        if re.search(r"(?<![\w.])8000(?![\w.])", line)
    ]
    assert not literals, (
        "criterion 14: a hard-coded 8000 is a second source of truth the config validator never "
        f"sees: {literals}"
    )


def test_the_session_still_imports_no_client_library_and_names_no_url():
    """Criterion 14: ``session.py``/``executor.py`` are unchanged in this respect.

    T-0007's ``tests/unit/test_session_no_egress.py`` asserts the same thing with its own
    ``LIBRARY_FILES`` list and must still pass **unmodified**; this is the re-assertion that the
    list still covers the two files T-0028 now owns, so the fence cannot be "passed" by shrinking
    it.
    """
    assert source(HUD_SERVER)
    for name in SESSION_FILES:
        assert name in LIBRARY_FILES, (
            f"criterion 14: {name} must stay in test_session_no_egress.LIBRARY_FILES "
            f"(it lists {LIBRARY_FILES})"
        )
        tree = ast.parse(source(name))
        banned = ("websockets", "httpx", "socket", "aiohttp", "requests")
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and node.module:
                offenders += [node.module] if node.module.split(".")[0] in banned else []
        assert not offenders, (
            f"criterion 14: {name} imports no transport library; it imports {offenders}. The "
            "session's egress is one injected publisher."
        )
        assert not url_offenders(source(name)), (
            f"criterion 14: no URL literal in {name}: {url_offenders(source(name))}"
        )


def test_nothing_in_the_pump_awaits_a_send():
    """Criterion 12, negative: publication is a hand-off, not an await.

    The scan is only worth anything if there *is* a publication site, so it asserts that too:
    the session must call its publisher (the name ``publish`` appears in its code) and must never
    await it.
    """
    for name in SESSION_FILES:
        body = source(name)
        offenders = awaited_sends(body)
        assert not offenders, (
            f"criterion 12: {name} awaits the wire: {offenders}. Publication is a non-blocking "
            "hand-off to a bounded per-connection queue; nothing in the pump or in a "
            "CommandExecutor handler may block the loop for more than 20 ms "
            "(.claude/rules/conductor.md)"
        )
    published = code_only(source("local_astra/conductor/session.py"))
    assert "publish" in published, (
        "the scan above must not be vacuous: the session does call its typed publisher "
        "(docs/02 §4 spec delta (b)); the word does not occur in session.py's code at all"
    )
    leftovers = [name for name in ("publish_hud", "hud_out") if name in published]
    assert not leftovers, (
        "the narrow mandate: the typed publisher *replaces* the scaffold's untyped outbox "
        f"(docs/tasks/T-0028.md, Owned files); {leftovers} is still there, so there are two "
        "egress paths and only one of them is on the wire"
    )
