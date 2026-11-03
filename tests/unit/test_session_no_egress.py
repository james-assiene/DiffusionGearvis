"""A blunt scan over T-0007's owned files: injected edges only, loopback only (criterion 1).

Same shape as ``tests/unit/test_hud_no_audio_no_egress.py``. The criterion is "injection is total,
and the module cannot reach a service by itself": every edge arrives as one of the
``contracts/protocols.py`` objects, so the session has no reason to import an HTTP or WebSocket
client, to open a socket, or to name an endpoint. The concrete clients are named in exactly one
place — ``scripts/demo/session_rung.py`` — and even there the endpoints come from
``config.Services``, whose validator already refuses anything but loopback (docs/02 §9).

Textual on purpose: this is the fence that catches an ``httpx.AsyncClient()`` default argument
that no behavioural test over fakes would ever see.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The owned files that are *library* code: no client library, no URL, no concrete client.
#: ``telemetry/session_log.py`` is T-0043's (ruling A6): the session log writes a debug file under
#: ``logs/`` and is the newest module with a reason to grow a "ship this somewhere" edge, so E8
#: fences it here rather than in a module of its own. docs/03 E8, criterion 25.
LIBRARY_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/conductor/executor.py",
    "local_astra/telemetry/turns_writer.py",
    "local_astra/telemetry/session_log.py",
    "local_astra/contracts/protocols.py",
)
#: The rung is the one place allowed to name the concrete clients (criterion 1, Owned files).
RUNG = "scripts/demo/session_rung.py"

BANNED_IMPORTS = (
    "httpx",
    "websockets",
    "sounddevice",
    "soundfile",
    "socket",
    "aiohttp",
    "requests",
)

#: Classes that *are* a service. A default argument that constructs one of these turns injection
#: into decoration: the session would still reach the network when nobody passed a fake.
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

URL_RE = re.compile(r"\b(?:https?|wss?)://")


def source(name: str) -> str:
    path = REPO / name
    assert path.is_file(), (
        f"{name} is one of the files this scan fences; the task that owns it adds it "
        "(docs/tasks/T-0007.md and docs/tasks/T-0043.md, Owned files)"
    )
    return path.read_text()


#: Words that turn a demo rung into a measurement (T-0016 ruling D5: harnesses that measure live
#: in ``scripts/gpu/`` and need a ``/launch`` token per run).
MEASUREMENT_RE = re.compile(r"\b(?:p50|p95|measured|throughput|benchmark)\b", re.IGNORECASE)


def code_only(body: str) -> str:
    """Drop comments and docstrings — and **only** docstrings.

    Ruling A9: the first version of this helper blanked every string constant, so
    ``_EP = "http://127.0.0.1:8001/v1"`` and ``Path("artifacts/rung.json")`` were invisible to the
    scans below — the two lines these scans exist to catch. Prose may cite an endpoint (a module
    docstring explaining the wire format is not a second source of truth); code may not.
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


def test_the_scan_helper_hides_docstrings_and_nothing_else():
    """Ruling A9, the scans' own guard: ``code_only`` must not blank a *code* string literal.

    The two insertions are the ones the review used to prove the scans had gone vacuous: an
    endpoint bound to a module constant, and a measurement written into ``artifacts/``. Both must
    survive ``code_only``; the docstring's URL must not.
    """
    sample = (
        '"""A module docstring that may cite ws://127.0.0.1:8000/v1/realtime."""\n'
        '_EP = "http://127.0.0.1:8001/v1"\n'
        "def go():\n"
        '    """Doc."""\n'
        '    Path("artifacts/rung.json").write_text("p50=1.2s measured")\n'
    )
    out = code_only(sample)
    assert "http://127.0.0.1:8001/v1" in out, "a code literal must stay visible to the URL scan"
    assert "artifacts/rung.json" in out, "…and to the artifacts scan"
    assert MEASUREMENT_RE.search(out), "…and to the measurement scan"
    assert "ws://127.0.0.1:8000/v1/realtime" not in out, "a docstring is prose, not a second URL"


# ``source`` asserts the file exists, so none of the scans below can pass over a file this task
# has not written yet (the vacuous-scan failure mode of ``test_hud_no_audio_no_egress.py``).


def test_no_owned_library_file_imports_a_client_library():
    """Criterion 1, negative: no ``httpx``/``websockets``/``sounddevice``/``soundfile``/``socket``."""
    offenders: list[str] = []
    for name in LIBRARY_FILES:
        tree = ast.parse(source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{name}:{node.lineno}: import {alias.name}"
                    for alias in node.names
                    if alias.name.split(".")[0] in BANNED_IMPORTS
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in BANNED_IMPORTS
            ):
                offenders.append(f"{name}:{node.lineno}: from {node.module} import …")
    assert not offenders, (
        "criterion 1: every edge is injected as a contracts/protocols.py object, so the session "
        "never imports a client library (docs/02 §5, §9; docs/03 §1.2):\n  "
        + "\n  ".join(offenders)
    )


def test_no_owned_file_carries_an_absolute_url_literal():
    """Criterion 1, negative: no ``http://``/``ws://`` literal anywhere, not even loopback.

    Endpoints come from ``config.Services`` — which is where the loopback validator lives. A
    literal in the session is a second source of truth that the validator never sees.
    """
    offenders: list[str] = []
    for name in (*LIBRARY_FILES, RUNG):
        for lineno, line in enumerate(code_only(source(name)).splitlines(), start=1):
            if URL_RE.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "criterion 1: no absolute URL literal in T-0007's files; every endpoint comes from "
        "config.Services, whose validator refuses non-loopback (docs/02 §9):\n  "
        + "\n  ".join(offenders)
    )


def test_no_default_argument_constructs_a_concrete_client():
    """Criterion 1, negative: ``def __init__(self, llm=VllmClient())`` is not injection."""
    offenders: list[str] = []
    for name in LIBRARY_FILES:
        tree = ast.parse(source(name))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
            for default in defaults:
                for inner in ast.walk(default):
                    if isinstance(inner, ast.Call):
                        callee = inner.func
                        label = getattr(callee, "attr", getattr(callee, "id", ""))
                        if label in CONCRETE_CLIENTS:
                            offenders.append(f"{name}:{node.lineno}: {node.name}(…={label}())")
    assert not offenders, (
        "criterion 1: no default argument constructs a concrete client; the six services are "
        "passed in, and the only place that names them is scripts/demo/session_rung.py:\n  "
        + "\n  ".join(offenders)
    )


def test_the_concrete_clients_are_named_only_in_the_rung():
    """Criterion 1: the library half never mentions a concrete client by name."""
    offenders: list[str] = []
    for name in LIBRARY_FILES:
        body = code_only(source(name))
        offenders += [f"{name}: {client}" for client in CONCRETE_CLIENTS if client in body]
    assert not offenders, (
        "criterion 1: the concrete clients are named only in scripts/demo/session_rung.py (and "
        "later in T-0007b):\n  " + "\n  ".join(offenders)
    )
    rung = source(RUNG)
    assert any(client in rung for client in CONCRETE_CLIENTS), (
        "the rung is what wires the real clients (criterion 15); it names none of "
        f"{CONCRETE_CLIENTS}, so this scan would pass vacuously"
    )


def test_the_rung_writes_no_artifact_and_claims_no_measurement():
    """T-0016 ruling D5 / the task's Implementation notes: ``scripts/demo/`` is ungated, so it must
    not write into ``artifacts/`` and must not claim a measurement — harnesses that measure live
    in ``scripts/gpu/`` and need a ``/launch`` token."""
    body = code_only(source(RUNG))
    offenders = [
        f"{lineno}: {line.strip()[:90]}"
        for lineno, line in enumerate(body.splitlines(), start=1)
        if "artifacts" in line or MEASUREMENT_RE.search(line)
    ]
    assert not offenders, (
        "scripts/demo/session_rung.py must not write into artifacts/ and must not claim a "
        "measurement (T-0016 ruling D5); harnesses that measure live in scripts/gpu/:\n  "
        + "\n  ".join(offenders)
    )


# --- T-0092b criterion 7: with tools = dry_run, no non-loopback socket ------------------------------
#
# Behavioural, beside the textual scans above: the dry-run startup (the CLI's kit loading over the fakes) and a
# dry-run tool turn run under a trap that refuses any connect, ``sendto``, ``create_connection`` or name lookup
# aimed anywhere but loopback. Loopback and AF_UNIX (the event loop's self-pipe) pass through, so the trap does
# not dictate how the session is written; the self-check row proves it fires.

import ipaddress  # noqa: E402
import socket  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402

from tests.unit import _tools_switch_support as T  # noqa: E402


def _loopback(host: object) -> bool:
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


@contextmanager
def loopback_only():
    attempts: list[str] = []
    mp = pytest.MonkeyPatch()
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    # T-0092c (carried from the T-0092b review): a server bound or listening anywhere but loopback is egress too.
    real_bind, real_listen = socket.socket.bind, socket.socket.listen
    real_sendto, real_cc, real_gai = (
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
    )

    def check(address: object, what: str) -> None:
        if isinstance(address, tuple) and not _loopback(address[0]):
            attempts.append(f"{what} {address!r}")
            raise AssertionError(f"T-0092b criterion 7: a non-loopback {what} to {address!r}")

    def connect(self, address):  # type: ignore[no-untyped-def]
        check(address, "connect")
        return real_connect(self, address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        check(address, "connect_ex")
        return real_connect_ex(self, address)

    def sendto(self, data, *rest):  # type: ignore[no-untyped-def]
        check(rest[-1] if rest else None, "sendto")
        return real_sendto(self, data, *rest)

    def create_connection(address, *a, **k):  # type: ignore[no-untyped-def]
        check(address, "create_connection")
        return real_cc(address, *a, **k)

    def getaddrinfo(host, *a, **k):  # type: ignore[no-untyped-def]
        check((host,), "getaddrinfo")
        return real_gai(host, *a, **k)

    def bind(self, address):  # type: ignore[no-untyped-def]
        # I13 (reviewer 2): for a bind, "" means INADDR_ANY (the wildcard, every interface), so it is never treated
        # as loopback here, whatever ``_loopback`` says for a connect
        if isinstance(address, tuple) and address[0] in ("", None):
            attempts.append(f"bind {address!r}")
            raise AssertionError(
                f"T-0092b criterion 7: a non-loopback bind to {address!r} (the wildcard)"
            )
        check(address, "bind")
        return real_bind(self, address)

    def listen(self, *a):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            # an unbound socket listens on the wildcard address; a bound one on what it was bound to
            check(self.getsockname(), "listen")
        return real_listen(self, *a)

    mp.setattr(socket.socket, "bind", bind)
    mp.setattr(socket.socket, "listen", listen)
    mp.setattr(socket.socket, "connect", connect)
    mp.setattr(socket.socket, "connect_ex", connect_ex)
    mp.setattr(socket.socket, "sendto", sendto)
    mp.setattr(socket, "create_connection", create_connection)
    mp.setattr(socket, "getaddrinfo", getaddrinfo)
    try:
        yield attempts
    finally:
        mp.undo()


def test_the_trap_fires_on_a_non_loopback_connect_and_lets_loopback_through():
    """Self-check: a green row below means "no non-loopback attempt", not "the trap was off"."""
    with loopback_only() as attempts:
        with pytest.raises(AssertionError, match="non-loopback"):
            socket.create_connection(("192.0.2.1", 80), timeout=0.01)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=1):
                pass
    assert attempts == ["create_connection ('192.0.2.1', 80)"]


async def test_a_dry_run_tool_turn_opens_no_non_loopback_socket(tmp_path):
    """Criterion 7: the dry-run session, a turn with a ``tool`` commit, under the trap."""
    from local_astra.contracts.turn import TurnState
    from tests.fakes.llm import FakeLlm
    from tests.unit.test_session_loop import answer

    kit = T.make_kit()
    script = [
        answer("I will click it. "),
        FakeLlm.tool_commit("<|tool_call>call:click_at{x:1,y:2}<tool_call|>"),
    ]
    with loopback_only() as attempts:
        rig = T.session_rig(tmp_path, kit=kit, script=script)
        await rig.session.start()
        try:
            await rig.ask("click it")
            await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        finally:
            await rig.session.aclose()
    assert not attempts and rig.lines(), attempts


def test_the_dry_run_startup_opens_no_non_loopback_socket(tmp_path, monkeypatch):
    """Criterion 7: ``astra session --tools dry-run`` over the fakes — the kit is loaded, nothing is served."""
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    with loopback_only() as attempts:
        code = T.run_main(T.session_argv(T.write_config(tmp_path), tmp_path, "--tools", "dry-run"))
    assert code == 0 and calls.detok_loads and calls.kit() is not None, (
        f"premise: the dry-run startup ran (exit {code}, loads {calls.detok_loads})"
    )
    assert not attempts, attempts


# --- T-0092c: the trap covers bind and listen; a full tool turn under it --------------------------------------


@pytest.mark.parametrize("how", ["bind", "bind-empty", "bind-empty-udp", "listen"])
def test_the_trap_fires_on_a_non_loopback_bind_or_listen(how):
    """Self-check (T-0092c, carried from the T-0092b review): binding the wildcard address, or listening on a socket
    that was never bound (the kernel would bind the wildcard), is refused before it happens; loopback passes."""
    kind = socket.SOCK_DGRAM if how.endswith("udp") else socket.SOCK_STREAM
    with loopback_only() as attempts:
        with (
            socket.socket(socket.AF_INET, kind) as sock,
            pytest.raises(AssertionError, match="non-loopback"),
        ):
            if how == "bind":
                sock.bind(("0.0.0.0", 0))
            elif how.startswith("bind-empty"):
                # I13 (reviewer 2): for bind, "" is INADDR_ANY -- every interface, the wildcard -- never loopback
                sock.bind(("", 0))
            else:
                sock.listen(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ok:
            ok.bind(("127.0.0.1", 0))
            ok.listen(1)
    assert len(attempts) == 1 and attempts[0].startswith(how.split("-")[0]), attempts


async def test_a_full_dry_run_tool_turn_with_its_repair_opens_no_non_loopback_socket(tmp_path):
    """T-0092c criterion 8: a repairable draft, its repair, the dry run and the HUD publishes, under the trap; the
    turn really executed (``dry_run`` published), so a green row is not a turn that did nothing."""
    from tests.unit import _tool_turn_support as TT

    failure: BaseException | None = None
    with loopback_only() as attempts:
        try:
            async with TT.tool_rig(
                tmp_path,
                script=[TT.answer(TT.SPOKEN), TT.tool(TT.REPAIRABLE_CLICK)],
                continuation=[TT.tool(TT.REPAIRED_CLICK, canvas_idx=0)],
                executor=TT.WatchingExecutor(inner=dry_run_executor()),
            ) as rig:
                await rig.ask("click the sign in button")
                await rig.turn_done()
        except Exception as exc:
            # the trap's refusal can stall the turn; report the attempt, not the stall
            failure = exc
    assert not attempts, (
        f"T-0092c criterion 8: non-loopback socket use during a tool turn: {attempts}"
    )
    if failure is not None:
        raise failure
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], (
        f"T-0092c: the tool turn ran to its dry run under the trap; published {rig.statuses()}"
    )


def dry_run_executor():
    from local_astra.conductor.tool_executor import DryRunToolExecutor

    return DryRunToolExecutor()
