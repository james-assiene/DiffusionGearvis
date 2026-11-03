"""``astra up`` stays free: no GPU time, no request body, nothing off loopback (T-0049 criterion 9).

The pre-check this task adds reuses ``supervisor.check_service`` — the same body-free ``GET`` on a
URL derived from ``config.Services`` that T-0033 criterion 3 pinned for ``astra doctor``. That is
what keeps ``astra up`` a *routine service start* under ``.claude/rules/gpu-services.md``: a clean
tree, no ``/launch`` token, no reviewer pass, no *Runs* entry. A request with a body to an
inference endpoint would change that classification for every start on the box, so the fence is
textual and blunt, like ``tests/unit/test_session_no_egress.py``: it catches a default argument or
a convenience helper that no behavioural test over fakes would ever see.

``docker inspect`` of a recorded id is a local socket call, not egress (docs/02 §9). The scans
below also pin criterion 6's negative as T-0158 criterion 7 restated it: the allowed daemon verbs
are ``inspect`` plus exactly one ``stop --time`` in one function (aimed at the container id of this
CLI's own record — that half is dynamic, ``tests/unit/test_t0158_container_stop.py``), and no code
path removes, kills, renames, restarts, starts or lists a container.

Every row looks the new seam up first. Without that the scans would pass over today's stub and
report a fence that is not there.

Regime: static, no GPU, no network, no services. Nothing here runs the code it reads.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from local_astra.config import LOOPBACK_PREFIXES, load
from tests.unit import _service_rig as rig

REPO = Path(__file__).resolve().parents[2]
MODULES = ("supervisor", "service_state")

#: A request with a body to one of these is GPU time (`.claude/rules/gpu-services.md`).
INFERENCE_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
    "/generate",
)

#: Neither module may import a new way of reaching the network. ``urllib`` is the one already
#: there (T-0033's probes) and is deliberately absent from this list.
BANNED_IMPORTS = ("socket", "httpx", "requests", "aiohttp", "websockets", "ssl", "ftplib")

#: ``docker`` verbs that change the world, or that establish identity by listing rather than by
#: asking about a recorded handle (``ps``). T-0158 criterion 7 takes ``stop`` out of this list and
#: into :data:`ALLOWED_VERBS` — once, in one function — and keeps every other one forbidden.
FORBIDDEN_DOCKER = ("run", "rm", "kill", "rename", "restart", "exec", "ps", "start", "pause")
#: The read verb, spelled here and **not** read from the module under test: taking it from
#: ``supervisor.DOCKER_READ_VERB`` would let a module that renamed that constant redefine
#: read-only and pass its own fence (ruling A15).
READ_VERB = "inspect"
#: T-0158 criterion 7 (spec change to T-0049 criterion 6 / ruling A15): the one mutating verb the
#: supervisor may use, on a container id read from its own state record. Spelled here for the
#: same reason as :data:`READ_VERB`.
STOP_VERB = "stop"
ALLOWED_VERBS = (READ_VERB, STOP_VERB)
#: The stop carries docker's grace flag (``--time``, criterion 1; docker 29 also spells it
#: ``--timeout``/``-t``). A stop without it is docker's own default and not the constant the
#: criterion names.
STOP_GRACE_FLAGS = ("--time", "--timeout", "-t")


def source_of(name: str) -> str:
    module = rig.module(name, "9")
    return Path(module.__file__).read_text()


def code_only(body: str) -> str:
    """Drop docstrings and nothing else (T-0007 ruling A9).

    Prose may cite an endpoint — a module docstring explaining the wire format is not a second
    source of truth — but a *code* string literal must stay visible to the scans, which is what
    blanking every constant would have destroyed.
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


def require_the_seam() -> None:
    """The scans are vacuous over a stub, so every row asserts the task's symbols exist first."""
    rig.attr(rig.module("service_state", "9"), "decide", "9")
    supervisor = rig.module("supervisor", "9")
    for name in ("start_service", "live_identity"):
        rig.attr(supervisor, name, "9")


def test_no_call_in_either_module_carries_a_request_body_or_names_an_inference_path():
    """Criterion 9: this is what keeps `astra up` out of `/launch` (T-0033 criterion 3's shape)."""
    require_the_seam()
    for name in MODULES:
        source = source_of(name)
        for banned in INFERENCE_PATHS:
            assert banned not in source, (
                f"criterion 9: {name}.py names {banned}; a request to an inference endpoint is GPU "
                "time and would make every `astra up` need a /launch token and a reviewer"
            )
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            kwargs = {kw.arg for kw in node.keywords}
            if called in {"Request", "urlopen", "post", "put", "patch", "request"}:
                assert not kwargs & {"data", "json", "body"}, (
                    f"criterion 9: {name}.py:{node.lineno} sends a request body in a {called}(…) "
                    "call; every probe is a bare GET"
                )
            assert called != "post", f"criterion 9: {name}.py:{node.lineno} POSTs"


def test_neither_module_imports_a_new_way_onto_the_network():
    """Criterion 9: "no new socket kinds" — the import list is where one would arrive."""
    require_the_seam()
    offenders: list[str] = []
    for name in MODULES:
        for node in ast.walk(ast.parse(source_of(name))):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{name}.py:{node.lineno}: import {alias.name}"
                    for alias in node.names
                    if alias.name.split(".")[0] in BANNED_IMPORTS
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in BANNED_IMPORTS
            ):
                offenders.append(f"{name}.py:{node.lineno}: from {node.module} import …")
    assert not offenders, (
        "criterion 9: all services bind to loopback and every probe is the urllib GET T-0033 "
        "already had (docs/02 §9, CLAUDE.md):\n  " + "\n  ".join(offenders)
    )


def test_no_url_is_typed_into_either_module():
    """Criterion 9: every probed URL is derived from ``config.Services``, whose validator refuses
    anything but loopback. A literal is a second source of truth that validator never sees."""
    require_the_seam()
    offenders: list[str] = []
    for name in MODULES:
        for lineno, line in enumerate(code_only(source_of(name)).splitlines(), start=1):
            if "http://" in line or "https://" in line or "ws://" in line or "wss://" in line:
                offenders.append(f"{name}.py:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "criterion 9: no absolute URL literal — probe_url() derives the probed URL from the "
        "configured one and re-validates it against config.LOOPBACK_PREFIXES:\n  "
        + "\n  ".join(offenders)
    )


def test_every_probe_target_is_loopback_and_is_not_an_inference_endpoint():
    """Criterion 9, dynamic: the URLs the pre-check would really probe, for the real profile."""
    require_the_seam()
    supervisor = rig.module("supervisor", "9")
    probe_url = rig.attr(supervisor, "probe_url", "9")
    services = load(None).profiles["lean"].services
    for name, configured in (
        ("vllm", services.vllm),
        ("stt", services.stt),
        ("tts", services.tts),
    ):
        probed = probe_url(name, configured)
        assert probed.startswith(LOOPBACK_PREFIXES), (
            f"criterion 9: {name} would be probed at {probed}, which is not loopback (docs/02 §9)"
        )
        assert urlsplit(probed).path not in INFERENCE_PATHS, (
            f"criterion 9: {name}'s pre-check probes {probed}, which costs GPU time"
        )


def module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, so a verb spelled as a constant is still read.

    Ruling A15: the previous version of this fence matched the literal text ``docker stop``, so a
    helper assembled from the module's own ``DOCKER_BIN`` and a ``_STOP_VERB`` constant walked
    straight through it and through all 29 rows. Only simple string constants are resolved — that
    is deliberately narrow, and anything the scan below cannot resolve is an offender rather than
    a pass, because a docker verb a reader cannot follow is exactly what this fence is for.
    """
    constants: dict[str, str] = {}
    for node in tree.body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else node.targets
            if isinstance(node, ast.Assign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                resolved = resolve(node.value, constants)
                if resolved is not None:
                    constants[target.id] = resolved
    return constants


#: Stands in for a piece of a string the source does not fix (a runtime value interpolated into an
#: f-string, say). It keeps the *shape* of the string readable — ``f"{DOCKER_BIN} {verb} {ref}"``
#: still shows a verb in verb position — while never matching a word the scans look for.
UNKNOWN_PIECE = "\x00"


def resolve(node: ast.expr | None, constants: dict[str, str], *, piece: bool = False) -> str | None:
    """The string this expression is, as far as the source fixes it; ``None`` when it fixes none.

    ``piece=True`` is used for the parts of a concatenation or an f-string: an unknown part becomes
    :data:`UNKNOWN_PIECE` rather than sinking the whole string, so that a command assembled at
    runtime around a constant verb — the shape ruling A15 is about — is still readable here.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else (UNKNOWN_PIECE if piece else None)
    if isinstance(node, ast.Name):
        found = constants.get(node.id)
        return found if found is not None else (UNKNOWN_PIECE if piece else None)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = resolve(node.left, constants, piece=True)
        right = resolve(node.right, constants, piece=True)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [resolve(value, constants, piece=True) for value in node.values]
        return "".join(part if part is not None else UNKNOWN_PIECE for part in parts)
    if isinstance(node, ast.FormattedValue):
        return resolve(node.value, constants, piece=True)
    return UNKNOWN_PIECE if piece else None


def docker_argvs(tree: ast.Module, constants: dict[str, str]) -> list[tuple[int, list[str | None]]]:
    """Every literal argv list/tuple whose first word is the docker client, resolved."""
    found: list[tuple[int, list[str | None]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List | ast.Tuple):
            continue
        parts = [resolve(element, constants) for element in node.elts]
        if parts and parts[0] is not None and Path(parts[0]).name == "docker":
            found.append((node.lineno, parts))
    return found


def sg_argvs(tree: ast.Module, constants: dict[str, str]) -> list[tuple[int, list[str | None]]]:
    """The ``sg docker -c '<command>'`` spelling — the group fallback, and its own way in."""
    found: list[tuple[int, list[str | None]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List | ast.Tuple):
            continue
        parts = [resolve(element, constants) for element in node.elts]
        if parts[:1] == ["sg"]:
            found.append((node.lineno, parts))
    return found


def enclosing_functions(tree: ast.Module) -> dict[int, str | None]:
    """``id(node) -> name of the innermost function that contains it`` (``None``: module level)."""
    owner: dict[int, str | None] = {}

    def visit(node: ast.AST, current: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else current
            )
            owner[id(child)] = inner
            visit(child, inner)

    visit(tree, None)
    return owner


_DOCKER_WORD = re.compile(r"(?<![\w/.-])docker\s+([\w-]+)")


def verb_fence(source: str) -> tuple[list[str], list[tuple[int, str | None]]]:
    """T-0158 criterion 7's fence over one module's source: ``(offenders, stop_sites)``.

    Asserted **positively** and with constants resolved (ruling A15): every docker command the
    module builds must have a verb in :data:`ALLOWED_VERBS` in verb position, and a command whose
    verb cannot be read off the source is an offender, not a pass. ``stop_sites`` is every argv
    (or literal ``sg`` payload) that runs ``docker stop``, with the function it is built in; a stop
    built at module level, or one without docker's grace flag, is an offender here, and *how many*
    sites there are is for the caller to judge (criterion 7: exactly one).

    Three spellings, because all three exist in this repo: an argv list built in Python, the
    ``sg docker -c`` retry whose payload is one string, and a command written as a literal.
    """
    tree = ast.parse(code_only(source))
    constants = module_string_constants(tree)
    owner = enclosing_functions(tree)
    offenders: list[str] = []
    stop_sites: list[tuple[int, str | None]] = []

    def check_stop(lineno: int, function: str | None, words: list[str | None], how: str) -> None:
        stop_sites.append((lineno, function))
        if function is None:
            offenders.append(f"line {lineno}: {how} `docker stop` is built at module level")
        if not any(
            word is not None and (word in STOP_GRACE_FLAGS or word.startswith("--time"))
            for word in words
        ):
            offenders.append(f"line {lineno}: {how} `docker stop` without its grace flag (--time)")

    for node in ast.walk(tree):
        if not isinstance(node, ast.List | ast.Tuple):
            continue
        parts = [resolve(element, constants) for element in node.elts]
        if not parts or parts[0] is None:
            continue
        function = owner.get(id(node))
        if Path(parts[0]).name == "docker":
            verb = parts[1] if len(parts) > 1 else None
            if verb not in ALLOWED_VERBS:
                offenders.append(f"line {node.lineno}: docker argv {parts} — verb is {verb!r}")
            elif verb == STOP_VERB:
                check_stop(node.lineno, function, parts, "an argv")
        elif parts[0] == "sg":
            if parts[:3] != ["sg", "docker", "-c"]:
                offenders.append(f"line {node.lineno}: sg argv {parts} is not `sg docker -c …`")
                continue
            payload = parts[3] if len(parts) > 3 else None
            # The payload is normally `shlex.join(command)` — unreadable here, and checked at its
            # own definition by the argv branch above. A payload written out is checked here.
            if payload is not None:
                for found in _DOCKER_WORD.findall(payload):
                    if found not in ALLOWED_VERBS:
                        offenders.append(
                            f"line {node.lineno}: `sg docker -c` runs `docker {found}`"
                        )
                    elif found == STOP_VERB:
                        check_stop(node.lineno, function, payload.split(), "an sg payload")

    # Review T3: a string that spells `docker stop` is a stop site too (a split shell string, a
    # `bash -c` payload), unless it lives in the one function that builds the argv stop — that is
    # where its own error messages are. With no argv stop at all, no function is exempt.
    stop_home = stop_sites[0][1] if len(stop_sites) == 1 else None
    argv_lines = {line for line, _function in stop_sites}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant | ast.JoinedStr | ast.BinOp):
            continue
        text = resolve(node, constants)
        if text is None:
            continue
        for found in _DOCKER_WORD.findall(text):
            if found in FORBIDDEN_DOCKER:
                offenders.append(f"line {node.lineno}: the string `docker {found}` is built here")
            elif found == STOP_VERB:
                function = owner.get(id(node))
                if node.lineno in argv_lines and function == stop_home:
                    continue  # the literal sg payload already counted above
                if stop_home is None or function != stop_home:
                    stop_sites.append((node.lineno, function))
                    offenders.append(
                        f"line {node.lineno}: the string `docker stop` is built outside the one "
                        f"function that stops ({stop_home!r})"
                    )
    if len(stop_sites) > 1:
        offenders.append(f"`docker stop` is built in {len(stop_sites)} places: {stop_sites}")
    return offenders, stop_sites


def supervisor_fence() -> tuple[list[str], list[tuple[int, str | None]]]:
    require_the_seam()
    source = source_of("supervisor")
    tree = ast.parse(code_only(source))
    assert any(
        Path(parts[0]).name == "docker"
        for _lineno, parts in docker_argvs(tree, module_string_constants(tree))
        if parts[0] is not None
    ), (
        "criterion 6: this fence reads docker commands off the source, so the module has to build "
        "them as literal argv lists (`[DOCKER_BIN, DOCKER_READ_VERB, ref]`). Not one was found: "
        "either nothing asks the daemon anything (criterion 6 wants `docker inspect` of the "
        "recorded id) or the argv is assembled dynamically, which no static fence can check"
    )
    return verb_fence(source)


def test_every_docker_verb_in_the_supervisor_is_inspect_or_the_one_stop():
    """T-0158 criterion 7 (spec change to T-0049 criterion 6 / ruling A15), the fence half.

    Not pending: true before T-0158 (no stop at all) and required to stay true after it. The
    allowed verbs are ``inspect`` and ``stop``; ``rm``, ``kill``, ``ps``, ``rename``, ``restart``,
    ``exec``, ``run``, ``start`` and ``pause`` stay forbidden; a verb the source does not fix is an
    offender; and there is **at most** one stop site, inside a function, carrying ``--time``.
    "Exactly one" is the next row's, pending until T-0158 adds the stop.
    """
    offenders, _stop_sites = supervisor_fence()
    assert not offenders, (
        "criterion 7: this module asks docker about a container it recorded (`inspect`) and stops "
        "one by its recorded id in exactly one place (`stop --time`). It never removes, kills, "
        "renames, restarts, starts or lists containers:\n  " + "\n  ".join(sorted(set(offenders)))
    )


def test_the_supervisor_builds_exactly_one_stop_inside_one_function():
    """T-0158 criterion 7: ``stop`` appears in exactly one argv, inside exactly one function."""
    offenders, stop_sites = supervisor_fence()
    assert len(stop_sites) == 1 and stop_sites[0][1] is not None, (
        "criterion 7: the supervisor builds `docker stop` in exactly one argv, inside one "
        f"function (the container branch of `stop`); found {stop_sites}"
    )
    assert not offenders, f"criterion 7: …and nothing else is offended: {offenders}"


# --- the fence over synthetic modules: it goes red on the mutations criterion 7 names -------------

_GOOD_MODULE = """
import shlex
DOCKER_BIN = "docker"
DOCKER_READ_VERB = "inspect"
DOCKER_STOP_VERB = "stop"

def _docker_read(ref):
    command = [DOCKER_BIN, DOCKER_READ_VERB, ref]
    return ["sg", DOCKER_BIN, "-c", shlex.join(command)]

def _docker_stop(container_id, grace):
    command = [DOCKER_BIN, DOCKER_STOP_VERB, "--time", f"{grace:g}", container_id]
    return f"{DOCKER_BIN} {DOCKER_STOP_VERB} could not be run", command
"""


def test_the_fence_accepts_one_stop_with_its_grace_in_one_function():
    """The fence's too-wide side: the shape criterion 7 allows is not an offender."""
    offenders, stop_sites = verb_fence(_GOOD_MODULE)
    assert offenders == [], offenders
    assert [function for _line, function in stop_sites] == ["_docker_stop"], stop_sites


@pytest.mark.parametrize(
    ("label", "addition"),
    [
        (
            "a second stop argv in another function",
            "\ndef _other(ref):\n    return [DOCKER_BIN, 'stop', '--time', '1', ref]\n",
        ),
        (
            "a second stop argv in one function",
            "\ndef _docker_stop2(ref):\n    a = [DOCKER_BIN, DOCKER_STOP_VERB, '--time', '1', ref]\n"
            "    b = [DOCKER_BIN, DOCKER_STOP_VERB, '--time', '1', ref]\n    return a, b\n",
        ),
        (
            "a stop in an sg payload beside the argv",
            "\ndef _sg_stop(ref):\n    return ['sg', 'docker', '-c', 'docker stop --time 1 x']\n",
        ),
        (
            "a split shell-string stop in another function (review T3)",
            "\ndef _other(ref):\n    return _docker_call('docker stop --time 1 astra-stt'.split())\n",
        ),
        (
            "a bash -c stop in another function (review T3)",
            "\ndef _other(ref):\n    return _docker_call(['bash', '-c', 'docker stop astra-stt'])\n",
        ),
        ("an rm argv", "\ndef _rm(ref):\n    return [DOCKER_BIN, 'rm', '-f', ref]\n"),
        ("a kill argv", "\ndef _kill(ref):\n    return ['docker', 'kill', ref]\n"),
        (
            "a kill verb behind a constant",
            "\n_KILL = 'kill'\ndef _kill(ref):\n    return [DOCKER_BIN, _KILL, ref]\n",
        ),
        ("an unresolvable verb", "\ndef _any(verb, ref):\n    return [DOCKER_BIN, verb, ref]\n"),
        (
            "an sg payload that removes",
            "\ndef _rm_sg(ref):\n    return ['sg', 'docker', '-c', 'docker rm astra-stt']\n",
        ),
        ("a ps listing", "\ndef _ps():\n    return [DOCKER_BIN, 'ps', '-q']\n"),
        ("a stop without --time", "\ndef _bare(ref):\n    return [DOCKER_BIN, 'stop', ref]\n"),
        ("a stop at module level", "\nSTOP_ARGV = [DOCKER_BIN, 'stop', '--time', '1', 'x']\n"),
        ("a shell string that renames", "\ndef _mv(ref):\n    return 'docker rename x y'\n"),
    ],
)
def test_the_fence_goes_red_on_each_named_mutation(label, addition):
    """Criterion 7, "Mutations that must go red": each one, added to the allowed shape."""
    offenders, _stop_sites = verb_fence(_GOOD_MODULE + addition)
    assert offenders, f"criterion 7: the fence must go red on {label}; it found nothing"
