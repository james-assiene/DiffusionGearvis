"""Both ``SttClient`` implementations conform to the protocol *as declared* (T-0032 criterion 7).

``contracts/protocols.py`` gains ``async def mark(self, t_eot_ms: int) -> int`` -- the one-line
follow-on T-0007a ruling A3 assigns to this task. ``isinstance`` against a ``runtime_checkable``
Protocol sees member *names* only, so it would be just as green with a ``mark`` that took no
argument or returned a string; the T-0007 ruling A13 landmine (``played_ms -> int`` against an
implementation returning ``float``) is the precedent, and this module is
``test_audio_io_protocol_static.py`` reused for the STT half.

The check is ``mypy --strict`` over ``_stt_callers.py`` (a module that assigns both classes to
variables annotated ``SttClient`` and drives them through the protocol only) and over
``_stt_caller_bad.py`` (a client that is right about everything except ``mark``). Neither is
collected by pytest. ``--follow-imports=silent`` keeps other modules' errors out of the verdict,
so the exit status *is* the assertion.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parents[2]
CALLER = REPO / "tests" / "contract" / "_stt_callers.py"
BAD_CALLER = REPO / "tests" / "contract" / "_stt_caller_bad.py"


@lru_cache(maxsize=4)
def _mypy(path: str) -> tuple[int, str]:
    """``(returncode, stdout+stderr)`` of ``mypy --strict`` over one caller module."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--show-error-codes",
            "--no-color-output",
            "--follow-imports=silent",
            path,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_mypy_accepts_both_stt_implementations_through_the_protocol() -> None:
    """A session typed against ``SttClient`` compiles against ``MoshiSttClient`` *and* ``FakeStt``.

    A failure here is one of two things and the output says which: ``mark`` is not on the protocol
    (so ``session.py`` cannot call it under ``mypy --strict``), or one of the two clients declares
    it with a signature the protocol does not.
    """
    rc, out = _mypy(str(CALLER))
    assert rc == 0, (
        "criterion 7: MoshiSttClient and/or FakeStt do not satisfy the SttClient protocol as "
        f"declared in local_astra/contracts/protocols.py:\n{out}"
    )


def test_the_type_checker_would_have_caught_a_mark_that_drifted() -> None:
    """Vacuity guard: a client whose ``mark`` returns ``str`` must be rejected, at the assignment.

    This is the row that proves the protocol actually *declares* ``mark``: with an ``SttClient``
    that does not, ``_stt_caller_bad.py`` type-checks cleanly and this assertion fails.
    """
    rc, out = _mypy(str(BAD_CALLER))
    assert rc != 0, (
        "criterion 7: mypy accepted an SttClient whose mark() returns str — the protocol does not "
        f"declare mark, so the harness is blind:\n{out}"
    )
    assert "[assignment]" in out, (
        "expected the protocol assignment to be rejected; mypy reported something else (a crash "
        f"or an import error would also be non-zero):\n{out}"
    )
    assert f"{BAD_CALLER.name}:" in out
