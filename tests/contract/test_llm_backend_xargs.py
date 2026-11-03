"""T-0017: ``LlmBackend.generate`` takes ``xargs``; the fake and the real client both conform.

Acceptance criteria (docs/tasks/T-0017.md), one section each:

1. the protocol declares ``xargs`` — keyword-only, ``Mapping[str, str] | None``, default ``None``,
   and existing positional call sites keep working;
2. ``FakeLlm.generate`` accepts and *records* it;
3. both implementations conform (``isinstance`` lives in ``test_fakes_conform.py``; the
   signature-level half is here);
4. a caller typed as ``LlmBackend`` passing ``xargs`` passes ``mypy --strict``;
5. values are strings: a non-string is rejected statically, and a value the engine could not
   carry never reaches the wire.

The "real" side here is ``VllmClient`` over ``httpx.MockTransport`` — the real request builder and
the real SSE reader, no GPU. The against-a-live-vLLM variant belongs to T-0007's replayers
(``tests/contract/test_vllm_shim_gpu.py``), which is why these are not marked ``gpu``.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import (
    VLLM_XARGS_KEY,
    Commit,
    LlmOptions,
    Message,
    Prompt,
    TextPart,
)
from local_astra.contracts.protocols import LlmBackend
from tests.contract._xargs_callers import TAP_XARGS, no_xargs_commits, tap_commits
from tests.fakes import FakeLlm

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
CALLER = REPO / "tests" / "contract" / "_xargs_callers.py"
BAD_CALLER = REPO / "tests" / "contract" / "_xargs_caller_bad.py"

PROMPT = Prompt(messages=(Message(role="user", content=(TextPart(text="hi"),)),), prefix_len=1)

#: the two commits both backends produce below, so fake and real assert the same thing
EXPECTED = [("thought", "hmm"), ("answer", "Four.")]
SSE = (
    'data: {"choices":[{"delta":{"reasoning":"hmm"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Four."}}]}\n\n'
    "data: [DONE]\n\n"
)
SCRIPT = (
    Commit(request_id="?", canvas_idx=0, segment="thought", text="hmm", thought_open=True),
    Commit(request_id="?", canvas_idx=1, segment="answer", text="Four."),
)


# --- harness: one interface over the fake and the real client -----------------------------------


@dataclass
class Harness:
    """A backend plus "what reached the wire", so both sides carry the same assertions."""

    backend: LlmBackend
    bodies: list[dict[str, Any]] = field(default_factory=list)
    fake: FakeLlm | None = None

    def sent_xargs(self) -> list[dict[str, Any]]:
        """The ``vllm_xargs`` mappings that actually left the boundary, oldest first."""
        if self.fake is not None:
            return [dict(self.fake.last_extra_body)] if self.fake.last_extra_body else []
        return [dict(b[VLLM_XARGS_KEY]) for b in self.bodies]


def _fake_harness() -> Harness:
    fake = FakeLlm(SCRIPT)
    return Harness(backend=fake, fake=fake)


def _vllm_harness() -> Harness:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=SSE)

    return Harness(backend=VllmClient(transport=httpx.MockTransport(handler)), bodies=bodies)


@pytest.fixture(params=["fake", "vllm"])
def harness(request: pytest.FixtureRequest) -> Harness:
    return _fake_harness() if request.param == "fake" else _vllm_harness()


# --- mypy runner (criteria 4 and 5) -------------------------------------------------------------


@lru_cache(maxsize=1)
def _mypy_errors() -> tuple[tuple[str, int, str, str], ...]:
    """``(file, line, code, message)`` for every mypy error in the two caller modules.

    One subprocess for both files, cached for the module: ~1 s, well inside the fast suite.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--show-error-codes",
            "--no-color-output",
            str(CALLER),
            str(BAD_CALLER),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out: list[tuple[str, int, str, str]] = []
    for line in proc.stdout.splitlines():
        # ``path:line: error: message  [code]`` (notes and blank lines are skipped)
        parts = line.split(":", 2)
        if len(parts) < 3 or not parts[2].startswith(" error: "):
            continue
        path, lineno, rest = parts
        message = rest[len(" error: ") :]
        code = message.rsplit("[", 1)[1].rstrip("]") if message.endswith("]") else ""
        out.append((Path(path).name, int(lineno), code, message))
    # A non-zero exit with nothing parsed means mypy itself did not run (missing module, crash,
    # a changed output format). Never read that as "no type errors" — that would make the
    # criterion-4 assertion vacuously green.
    if proc.returncode != 0 and not out:
        raise AssertionError(
            f"mypy did not run: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )
    return tuple(out)


def _errors_in(filename: str) -> list[tuple[str, int, str, str]]:
    return [e for e in _mypy_errors() if e[0] == filename]


# --- criterion 1: the protocol signature --------------------------------------------------------


def test_protocol_generate_declares_keyword_only_xargs_defaulting_to_none():
    params = inspect.signature(LlmBackend.generate).parameters
    assert "xargs" in params, (
        "criterion 1: LlmBackend.generate must take an xargs keyword (T-0017); "
        f"it takes {list(params)}"
    )
    xargs = params["xargs"]
    assert xargs.kind is inspect.Parameter.KEYWORD_ONLY, "criterion 1: xargs is keyword-only"
    assert xargs.default is None, "criterion 1: xargs defaults to None"
    # the keyword goes last: prompt/request_id/options keep their positions
    # T-0088 ruling E1: widened by exactly ``cd`` (T-0088 criterion 3), keyword-only, default None
    assert list(params) == ["self", "prompt", "request_id", "options", "xargs", "cd"]
    assert params["cd"].kind is inspect.Parameter.KEYWORD_ONLY and params["cd"].default is None


def test_protocol_xargs_maps_strings_to_strings():
    """Criterion 5's static half: the declared type is what forbids ``{"n": 1}``."""
    hints = typing.get_type_hints(LlmBackend.generate)
    assert "xargs" in hints, "criterion 1: LlmBackend.generate has no xargs annotation yet"
    assert set(typing.get_args(hints["xargs"])) == {Mapping[str, str], type(None)}, (
        f"criterion 5: xargs must be Mapping[str, str] | None, got {hints['xargs']!r}"
    )


async def test_positional_call_sites_are_unaffected(harness):
    """Criterion 1: every call written before T-0017 still binds, on both implementations."""
    got = await no_xargs_commits(harness.backend, PROMPT, "r0")
    assert [(c.segment, c.text) for c in got] == EXPECTED
    assert harness.sent_xargs() == [{"astra_request_id": "r0"}]


# --- criterion 2: the fake records what it was given ---------------------------------------------


async def test_fake_records_the_xargs_of_each_generate():
    fake = FakeLlm(SCRIPT)
    assert fake.last_xargs is None, "criterion 2: nothing recorded before the first call"
    got = [c async for c in fake.generate(PROMPT, "r1", xargs={"astra_tap": "1"})]
    assert [(c.segment, c.text) for c in got] == EXPECTED
    assert fake.last_xargs == {"astra_tap": "1"}
    assert fake.xargs_log == [{"astra_tap": "1"}]


async def test_fake_does_not_remember_xargs_from_a_previous_turn():
    """Adversarial: a fake that only assigns when ``xargs`` is truthy reports last turn's tap."""
    fake = FakeLlm(SCRIPT)
    async for _ in fake.generate(PROMPT, "r1", xargs={"astra_tap": "1"}):
        pass
    async for _ in fake.generate(PROMPT, "r2"):
        pass
    assert fake.last_xargs is None, "a turn without xargs must not inherit the previous turn's"
    assert fake.xargs_log == [{"astra_tap": "1"}, None]


async def test_fake_records_a_snapshot_not_the_callers_mapping():
    """Adversarial: a caller that reuses one dict must not rewrite history."""
    caller_dict = {"astra_tap": "1"}
    fake = FakeLlm(SCRIPT)
    async for _ in fake.generate(PROMPT, "r1", xargs=caller_dict):
        pass
    caller_dict["astra_tap"] = "0"
    caller_dict["constraint"] = "length_bounded:64"
    assert fake.last_xargs == {"astra_tap": "1"}


def test_fake_refuses_xargs_as_a_fourth_positional_argument():
    """Criterion 2: same shape as the protocol — keyword-only, so no call site can drift."""
    with pytest.raises(TypeError):
        FakeLlm(SCRIPT).generate(PROMPT, "r1", LlmOptions(), {"astra_tap": "1"})


# --- criterion 3: both implementations conform ---------------------------------------------------


@pytest.mark.parametrize("impl", [FakeLlm, VllmClient], ids=["fake", "vllm"])
def test_both_implementations_accept_the_protocols_xargs(impl):
    proto = inspect.signature(LlmBackend.generate).parameters
    assert "xargs" in proto, "criterion 3: the protocol itself does not declare xargs yet (T-0017)"
    got = inspect.signature(impl.generate).parameters
    assert "xargs" in got, f"criterion 3: {impl.__name__}.generate does not accept xargs"
    assert got["xargs"].kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), "a protocol caller passes xargs by keyword; the implementation must accept that"
    assert got["xargs"].default is None
    assert set(proto) <= set(got), (
        f"criterion 3: {impl.__name__}.generate is missing {set(proto) - set(got)}"
    )


async def test_the_typed_caller_reaches_the_wire_on_both_backends(harness):
    """Criteria 3+5: the same protocol-typed function, run against fake and real client."""
    got = await tap_commits(harness.backend, PROMPT, "r7")
    assert [(c.segment, c.text) for c in got] == EXPECTED
    assert [c.request_id for c in got] == ["r7", "r7"]
    assert harness.sent_xargs() == [{"astra_tap": "1", "astra_request_id": "r7"}], (
        "well-formed xargs travel under vllm_xargs beside the request id abort() targets"
    )
    assert TAP_XARGS == {"astra_tap": "1"}, "the caller's mapping was mutated in place"


async def test_the_request_id_in_vllm_xargs_is_the_engines_not_the_callers(harness):
    """Adversarial: abort() finds a sequence by ``astra_request_id``; a caller must not spoof it.

    (docs/02 §5 + llm_client: the id under ``vllm_xargs`` is what abort targets.)
    """
    async for _ in harness.backend.generate(
        PROMPT, "r9", None, xargs={"astra_request_id": "spoofed", "astra_tap": "1"}
    ):
        pass
    assert harness.sent_xargs() == [{"astra_request_id": "r9", "astra_tap": "1"}]


def test_mypy_accepts_a_caller_typed_as_llmbackend_passing_xargs():
    """Criterion 4, the reason this is a mypy test and not a runtime one.

    ``isinstance`` against a ``runtime_checkable`` Protocol only checks that the *names* exist, so
    it cannot see a missing parameter; and both concrete classes would accept the call at runtime
    once the fake is updated. Only the type checker reads the protocol's signature, which is what
    T-0011 and the constrained-decoding work will be written against.
    """
    errors = _errors_in(CALLER.name)
    assert errors == [], f"criterion 4: mypy --strict rejects the protocol-typed caller: {errors}"


# --- criterion 5: string values only -------------------------------------------------------------


def test_mypy_rejects_non_string_xargs_values_for_being_non_string():
    """Criterion 5: both bad calls are errors, and about the *value type*, not a missing keyword.

    Before T-0017 these lines also fail — with ``call-arg`` ("Unexpected keyword argument
    xargs"), which would be the wrong reason: it says nothing about ``{"n": 1}``.
    """
    errors = _errors_in(BAD_CALLER.name)
    codes = {code for _, _, code, _ in errors}
    assert len(errors) == 2, f"criterion 5: expected one error per bad call, got {errors}"
    assert codes <= {"arg-type", "dict-item"}, (
        f"criterion 5: xargs values must be rejected for their type; mypy said {errors}"
    )
    assert "call-arg" not in codes


async def test_a_value_the_engine_cannot_carry_never_reaches_the_wire(harness):
    """Criterion 5: rejected at the boundary, not forwarded to 400 later. Same on both backends."""
    with pytest.raises((TypeError, ValueError)) as excinfo:
        async for _ in harness.backend.generate(
            PROMPT, "r5", None, xargs={"budget": {"tokens": 64}}
        ):
            pass
    assert "budget" in str(excinfo.value)
    assert harness.sent_xargs() == [], "the bad mapping must not have been sent"
