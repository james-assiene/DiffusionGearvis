"""A static scan over T-0007a's three owned files: loopback, no resampler, no drain reflex.

Criteria 1, 2, 5, 8 and 17's negatives, in the shape of ``test_hud_no_audio_no_egress.py``:
deliberately textual, because these are the fences no behavioural test looks at — a stray
``ws://192.168.…``, an ``import scipy`` that would make ``feed`` a resampler, an
``except Exception: pass`` that turns criterion 5's five failures into silence, or
``_listen.py``'s ``DRAIN_S`` tail heuristic copied across (criterion 4's adversarial case, which
would then make ``SttFlushed`` a clock instead of a message).

Comments are excluded by construction (the scan reads the AST, not the text) and docstrings are
skipped, so prose may name a spec, a URL or the heuristic this task forbids.

Every test here starts with :func:`require_implemented`: a scan over two stubs that raise
``NotImplementedError`` would pass vacuously, which is exactly the failure mode
``.claude/rules/tests.md`` calls "asserting current behaviour".
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from local_astra.config import LOOPBACK_PREFIXES, Services

REPO = Path(__file__).resolve().parents[2]

#: ``docs/tasks/T-0007a.md`` "Owned files" — the whole blast radius of this task.
OWNED = (
    "local_astra/conductor/stt_client.py",
    "local_astra/conductor/tts_client.py",
    "local_astra/contracts/speech.py",
)
CLIENT_MODULES = OWNED[:2]

#: Everything already in ``[project].dependencies`` (pyproject.toml): "no new dependency is
#: needed and none may be added" (Implementation notes).
ALLOWED_THIRD_PARTY = frozenset(
    {"pydantic", "msgpack", "websockets", "httpx", "numpy", "jsonschema", "referencing"}
)

#: Modules that would mean the client resamples, reframes or captures audio itself — T-0021's job.
FORBIDDEN_IMPORTS = (
    "scipy",
    "resampy",
    "soxr",
    "samplerate",
    "librosa",
    "av",
    "torch",
    "sounddevice",
    "pyaudio",
    "local_astra.conductor.audio_resample",
    "local_astra.conductor.audio_capture",
)

#: The spike's heuristics (``scripts/demo/_listen.py``), which must not cross into the client.
FORBIDDEN_NAMES = (
    "DRAIN_S",
    "DRAIN_HARD_CAP_S",
    "SILENCE_S",
    "SILENCE_LEVEL",
    "SILENCE_FLOOR",
    "SILENCE_FRACTION",
    "TAIL_SILENCE_S",
    "SttEndOfTurnEvent",
)

URL_RE = re.compile(r"\b(?:https?|wss?)://")
PORT_LITERALS = frozenset({8000, 8001, 8080, 8081, 8003, 1420})


def source(relative: str) -> str:
    return (REPO / relative).read_text()


def tree(relative: str) -> ast.Module:
    return ast.parse(source(relative), filename=relative)


def require_implemented() -> None:
    """These scans mean nothing over the stubs: no ``mark``, no error classes, no behaviour."""
    from local_astra.conductor import stt_client, tts_client

    missing = [n for n in ("mark", "stats") if not hasattr(stt_client.MoshiSttClient, n)]
    missing += [n for n in ("TtsClientError", "TtsFormatError") if not hasattr(tts_client, n)]
    assert not missing, (
        f"T-0007a is not implemented ({missing} missing), so every scan in this module "
        "would pass vacuously over a stub"
    )


def string_constants(relative: str) -> list[str]:
    """Every string literal except docstrings (prose may name anything)."""
    module = tree(relative)
    docstrings = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def imported_modules(relative: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree(relative)):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


# --- criterion 1 / 17: loopback, and the endpoint comes from the config ---------------------------


def test_no_url_host_or_port_literal_beyond_the_documented_loopback_defaults():
    """Criterion 1: the only URL literal allowed is the default that mirrors ``Services``."""
    require_implemented()
    services = Services()
    allowed = {services.stt, services.tts, services.vllm, "public_token"}
    allowed |= {f"{services.stt}?auth_id=public_token", "/tts", "/health"}
    allowed |= set(LOOPBACK_PREFIXES)  # the client may carry its own loopback guard

    for relative in OWNED:
        for literal in string_constants(relative):
            if not URL_RE.search(literal):
                continue
            assert literal.startswith(LOOPBACK_PREFIXES), f"{relative}: non-loopback {literal!r}"
            assert literal in allowed, (
                f"{relative}: {literal!r} is not the documented default "
                f"(the endpoint comes from local_astra.config.Services)"
            )

    for relative in CLIENT_MODULES:  # speech.py's 8000 is ``AudioFormat.sample_rate``'s floor
        ports = [
            node.value
            for node in ast.walk(tree(relative))
            if isinstance(node, ast.Constant) and node.value in PORT_LITERALS
        ]
        assert ports == [], f"{relative}: port literal(s) {ports} — ports live in Services"


# --- criterion 2 / 17: no resampler, no new dependency ---------------------------------------------


def test_no_resampler_and_no_new_third_party_import():
    """Criterion 2's negative: ``feed`` is a consumer — T-0021 owns 48 → 24 kHz and the framing."""
    require_implemented()
    stdlib = set(sys.stdlib_module_names)
    for relative in OWNED:
        for module in imported_modules(relative):
            root = module.split(".")[0]
            assert not any(
                module == bad or module.startswith(f"{bad}.") for bad in FORBIDDEN_IMPORTS
            ), f"{relative} imports {module}: resampling/capture belongs to T-0021"
            assert root in stdlib or root == "local_astra" or root in ALLOWED_THIRD_PARTY, (
                f"{relative} imports {module}, which is not in [project].dependencies "
                "(T-0007a adds no dependency)"
            )


# --- criterion 4 / 18: no drain reflex, no endpointing ----------------------------------------------


def test_the_stt_client_carries_no_drain_heuristic_and_no_endpointing():
    """Criteria 4 and 18's negatives: ``SttFlushed`` comes from a ``Marker``, never from a clock."""
    require_implemented()
    text = source("local_astra/conductor/stt_client.py")
    module = tree("local_astra/conductor/stt_client.py")
    identifiers = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(module) if isinstance(node, ast.Attribute)}
    identifiers |= {
        target.id
        for node in ast.walk(module)
        for target in getattr(node, "targets", [])
        if isinstance(target, ast.Name)
    }
    for name in FORBIDDEN_NAMES:
        assert name not in identifiers, (
            f"stt_client.py names {name}: the drain/silence reflex is the spike's "
            "(scripts/demo/_listen.py), and endpointing is the reflex lane's (T-0007)"
        )
    assert "SttEndOfTurnEvent" not in text, "criterion 18: the client never endpoints on its own"
    floats = {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    assert 0.6 not in floats, "the pause-head threshold (prs[1] > 0.6) belongs to the reflex lane"


# --- criterion 5: failures are events, never silence -------------------------------------------------


def test_no_silently_swallowed_exception():
    """Criterion 5's negative: every failure becomes exactly one ``SttErrorEvent``."""
    require_implemented()
    for relative in CLIENT_MODULES:
        for node in ast.walk(tree(relative)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = [s for s in node.body if not isinstance(s, ast.Expr)]
            silent = all(isinstance(s, ast.Pass) for s in body) and body
            assert not silent, (
                f"{relative}:{node.lineno}: `except …: pass` — a swallowed failure is "
                "indistinguishable from a service that transcribes nothing (docs/03 I7)"
            )


# --- criterion 8: bounded buffers, no task per block --------------------------------------------------


def test_no_unbounded_queue_and_no_task_per_fed_block():
    """Criterion 8's negative: a bounded outbound buffer and one long-lived reader/sender."""
    require_implemented()
    module = tree("local_astra/conductor/stt_client.py")
    creations = 0
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.endswith("Queue"):
            assert node.args or node.keywords, (
                f"stt_client.py:{node.lineno}: {name}() has no maxsize — an unbounded queue "
                "turns a stuck peer into unbounded memory (criterion 8)"
            )
        if name.endswith("create_task"):
            creations += 1
    assert creations <= 4, f"{creations} create_task call sites: one connection, not one per block"

    for node in ast.walk(module):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "feed":
            calls = [
                ast.unparse(inner.func)
                for inner in ast.walk(node)
                if isinstance(inner, ast.Call) and ast.unparse(inner.func).endswith("create_task")
            ]
            assert calls == [], f"feed() creates a task per block: {calls}"
