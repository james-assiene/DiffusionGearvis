"""The work counter of the linearity rows (T-0129): ``tests/unit/_work_count.py``.

The toys below live in this file, so ``ROOT`` (this file, from ``__file__``) is their root. Expected
counts are by hand from the toy's lines (one ``line`` event per executed line, and one per loop
head each time the loop re-tests), never from the counter's output. Calls are passed as
``functools.partial`` (a C callable, so no extra traced frame) unless a row says otherwise.

Regime: CPU only, deterministic (no clock is read). Python 3.11 (``sys.settrace``).
"""

from __future__ import annotations

import functools
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests.unit._work_count import ForeignTracerError, WorkCapExceeded, line_events

ROOT = Path(__file__)
TEXT = "\n".join(f"    line {k}" for k in range(200))


def toy_linear(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total


def toy_quadratic(n: int) -> int:
    total = 0
    for i in range(n):
        for j in range(n):
            total += i ^ j
    return total


def toy_calls_stdlib(text: str) -> int:
    out = textwrap.dedent(text)
    return len(out)


def toy_calls_builtin(text: str) -> int:
    out = str(text)
    return len(out)


def test_counts_are_exact_on_a_hand_counted_loop() -> None:
    """``toy_linear(n)``: ``total = 0`` once, the ``for`` head n + 1 times, the body n times and
    ``return`` once, so 2n + 3 line events."""
    for n in (0, 1, 5, 100):
        got = line_events(functools.partial(toy_linear, n), ROOT)
        assert got == 2 * n + 3, f"toy_linear({n}): {got} line events, by hand {2 * n + 3}"


def test_counts_are_deterministic() -> None:
    counts = [line_events(functools.partial(toy_linear, 1000), ROOT) for _ in range(3)]
    assert counts == [2003, 2003, 2003], f"three calls of toy_linear(1000) counted {counts}"


def test_quadratic_toy_is_seen() -> None:
    """A nested loop: L(n) = 2n² + 2n + 3 by hand, so L(2n)/L(n) is about 4 (> 3.5)."""
    small = line_events(functools.partial(toy_quadratic, 40), ROOT)
    large = line_events(functools.partial(toy_quadratic, 80), ROOT)
    assert large / small > 3.5, (
        f"quadratic toy: L(80)/L(40) = {large}/{small} = {large / small:.2f}"
    )
    linear_small = line_events(functools.partial(toy_linear, 40), ROOT)
    linear_large = line_events(functools.partial(toy_linear, 80), ROOT)
    assert linear_large / linear_small <= 2.25, (
        f"linear toy: L(80)/L(40) = {linear_large}/{linear_small}; the bar must pass a linear loop"
    )


def test_frames_outside_root_are_not_counted() -> None:
    """``textwrap.dedent`` runs dozens of Python lines on a 200-line text; under this file's root
    it adds nothing, so the toy that calls it counts the same 2 lines as the toy calling a C
    builtin. The positive control counts with ``textwrap``'s own file as root: the same call is
    then visible, so the equality is not an empty tracer."""
    via_stdlib = line_events(functools.partial(toy_calls_stdlib, TEXT), ROOT)
    via_builtin = line_events(functools.partial(toy_calls_builtin, TEXT), ROOT)
    assert via_stdlib == via_builtin == 2, (
        f"only this file's lines count: stdlib toy {via_stdlib}, builtin toy {via_builtin}, by hand 2"
    )
    assert textwrap.__file__ is not None
    inside = line_events(functools.partial(toy_calls_stdlib, TEXT), textwrap.__file__)
    assert inside > 10, f"positive control: dedent under its own root counted {inside} lines"


def test_a_directory_root_counts_every_file_below_it() -> None:
    """A directory root covers its files: this file's directory counts the toy; a sibling
    directory does not."""
    assert line_events(functools.partial(toy_linear, 5), ROOT.parent) == 13
    assert line_events(functools.partial(toy_linear, 5), ROOT.parent.parent / "fakes") == 0


def test_cap_aborts() -> None:
    """A cap below the work stops the run inside the traced code with the named error; a cap at
    the exact count does not."""
    reached: list[int] = []

    def toy_records(n: int) -> None:
        for i in range(n):
            reached.append(i)

    with pytest.raises(WorkCapExceeded) as info:
        line_events(functools.partial(toy_records, 10**6), ROOT, cap=100)
    assert info.value.cap == 100 and info.value.count == 101
    assert len(reached) < 100, f"the run went on after the cap: {len(reached)} iterations"
    assert sys.gettrace() is None, "the tracer is removed after an aborted run"
    assert line_events(functools.partial(toy_linear, 5), ROOT, cap=13) == 13


def test_cap_is_not_swallowed_by_the_code_under_test() -> None:
    """Code under test that catches ``Exception`` (or even ``BaseException``) cannot turn the cap
    into a pass: the helper re-raises it."""

    def toy_swallows(n: int) -> None:
        try:
            for _ in range(n):
                pass
        except BaseException:
            pass

    with pytest.raises(WorkCapExceeded):
        line_events(functools.partial(toy_swallows, 10**5), ROOT, cap=50)


def test_previous_tracer_is_restored() -> None:
    """After a normal run, an aborted run and a run whose function raises, ``sys.gettrace()`` is
    what it was (``None`` here: the helper refuses to run under another tracer)."""
    assert sys.gettrace() is None, "setup: no tracer before the row"
    line_events(functools.partial(toy_linear, 3), ROOT)
    assert sys.gettrace() is None
    with pytest.raises(WorkCapExceeded):
        line_events(functools.partial(toy_linear, 100), ROOT, cap=5)
    assert sys.gettrace() is None

    def toy_raises() -> None:
        raise KeyError("boom")

    with pytest.raises(KeyError):
        line_events(toy_raises, ROOT)
    assert sys.gettrace() is None


def test_refuses_under_a_foreign_tracer() -> None:
    """Adversarial: with another tracer installed (coverage, a debugger) the helper raises the
    named error -- not a silent count of 0, and not by displacing the other tracer, which must
    still be installed afterwards and still see its own events."""
    seen: list[str] = []

    def foreign(frame: Any, event: str, arg: Any) -> Any:
        if event == "call" and frame.f_code is toy_linear.__code__:
            seen.append(event)
        return None

    sys.settrace(foreign)
    try:
        with pytest.raises(ForeignTracerError, match="another tracer"):
            line_events(functools.partial(toy_linear, 3), ROOT)
        still = sys.gettrace()
        toy_linear(3)
    finally:
        sys.settrace(None)
    assert issubclass(ForeignTracerError, AssertionError)
    assert still is foreign, f"the foreign tracer was displaced: {still!r}"
    assert seen == ["call"], f"the foreign tracer stopped seeing calls: {seen}"
