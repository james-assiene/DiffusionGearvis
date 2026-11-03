"""Both ``AudioIO`` implementations conform to the protocol *as declared* (docs/02 §5).

Why a type checker and not ``isinstance``: ``contracts/protocols.py`` marks ``AudioIO``
``runtime_checkable``, and a runtime-checkable Protocol only verifies that the member *names*
exist. It cannot see a wrong return type, a missing keyword or a missing parameter. That is how
T-0007's confirmation review found the protocol promising ``played_ms() -> int`` while
``PipeWireAudioIO.played_ms`` returned ``float``, with every conformance test in
``tests/contract/test_audio_io_conformance.py`` green: only a ``mypy`` run on an unrelated caller
noticed. This module closes that hole for the two implementations the conductor is ever handed:

* the real ``local_astra.conductor.audio_io.PipeWireAudioIO`` (device injected), and
* ``tests/fakes/audio.py::FakeAudioIO``, which the whole fast suite substitutes for it -- a fake
  that does not satisfy the same declaration makes every test written against it a lie
  (``.claude/rules/tests.md``).

The check is ``mypy --strict`` over ``_audio_io_callers.py``: a module that assigns both classes
to variables annotated ``AudioIO`` and then drives them through the protocol only. It is not
collected by pytest and never imported at runtime (constructing ``PipeWireAudioIO`` starts a
writer thread). ``--follow-imports=silent`` keeps the errors of other modules out of the verdict
while still using their types, so the exit status *is* the assertion.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parents[2]
CALLER = REPO / "tests" / "contract" / "_audio_io_callers.py"
BAD_CALLER = REPO / "tests" / "contract" / "_audio_io_caller_bad.py"


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


def test_mypy_accepts_both_audio_io_implementations_through_the_protocol() -> None:
    """The criterion: a session typed against ``AudioIO`` compiles against module *and* fake.

    A failure here is one of two things and the output says which: a member the protocol does not
    declare (the session cannot call it), or a signature the implementation gets wrong (the
    ``played_ms -> int`` / ``-> float`` disagreement). Neither is visible to ``isinstance``.
    """
    rc, out = _mypy(str(CALLER))
    assert rc == 0, (
        "PipeWireAudioIO and/or FakeAudioIO do not satisfy the AudioIO protocol as declared in "
        f"local_astra/contracts/protocols.py:\n{out}"
    )


def test_the_type_checker_would_have_caught_a_drifting_implementation() -> None:
    """Vacuity guard: if mypy cannot see the protocol, the test above is green for nothing.

    ``_audio_io_caller_bad.py`` assigns a class that is missing ``capture`` and returns ``str``
    from ``flush`` -- the two shapes of drift. mypy must reject it, and reject it at the
    *assignment*, i.e. for failing the protocol, not because the module failed to import.
    """
    rc, out = _mypy(str(BAD_CALLER))
    assert rc != 0, f"mypy accepted a class that is not an AudioIO; the harness is blind:\n{out}"
    assert "[assignment]" in out, (
        "expected the protocol assignment to be rejected; mypy reported something else "
        f"(a crash or an import error would also be non-zero):\n{out}"
    )
    assert f"{BAD_CALLER.name}:" in out
