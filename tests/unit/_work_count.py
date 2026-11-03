"""Deterministic work counting for the linearity rows (T-0129): traced Python ``line`` events.

:func:`line_events` runs ``fn`` under :func:`sys.settrace` and counts the ``line`` events of every
frame whose code file resolves under ``root`` (a directory, or one file). Frames elsewhere -- the
test, the stdlib, httpx, the fakes -- are not traced at all, so their work is not counted. The count
does not depend on load, clocks or the machine: it is the number of Python lines the code under
test executed, so two sizes give a work ratio that cannot be reddened by a busy box. It cannot see
work done inside C (a regex, ``str.find``, a slice copy); the CPU-time half of each row covers that.

Rules (T-0129 *Tests to write*):

- the root is passed in by the caller, taken from an imported module's ``__file__`` at runtime,
  never a literal path, so a mutation copy counts its own files;
- the previous tracer (``sys.gettrace()``) is restored in ``finally``;
- a count that passes ``cap`` raises :class:`WorkCapExceeded` from inside the traced code, so a
  quadratic mutant stops in bounded time with a named error;
- under another tracer (coverage, a debugger) the helper refuses with a named ``AssertionError``:
  it never chains to it or displaces it silently, since either would make the count wrong or
  break the other tool;
- not for Hypothesis-decorated tests (Hypothesis may trace for its own explain phase).

Python 3.11 is pinned, so ``sys.monitoring`` is not available; ``settrace`` costs about 0.1 s per
0.5 M events on the 5090 box (T-0129 *Implementation notes*).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from types import FrameType
from typing import Any

__all__ = ["ForeignTracerError", "WorkCapExceeded", "line_events"]


class WorkCapExceeded(BaseException):
    """The traced code passed its line-event cap.

    A ``BaseException`` so that an ``except Exception`` in the code under test cannot swallow it
    and keep running untraced; :func:`line_events` also re-raises it after ``fn`` returns if the
    code under test swallowed it anyway.
    """

    def __init__(self, count: int, cap: int) -> None:
        super().__init__(f"traced line events passed the cap: {count} > {cap}")
        self.count = count
        self.cap = cap


class ForeignTracerError(AssertionError):
    """Another tracer is installed; this row cannot count work under it."""


def _under(root: str) -> Callable[[str], bool]:
    """A cached ``co_filename -> bool``: does the file resolve to ``root`` or lie under it?"""
    cache: dict[str, bool] = {}
    prefix = root.rstrip(os.sep) + os.sep

    def check(filename: str) -> bool:
        hit = cache.get(filename)
        if hit is None:
            if filename.startswith("<"):  # <frozen ...>, <string>, <genexpr> sources
                hit = False
            else:
                resolved = os.path.realpath(filename)
                hit = resolved == root or resolved.startswith(prefix)
            cache[filename] = hit
        return hit

    return check


def line_events(
    fn: Callable[[], Any], root_dir: str | os.PathLike[str], cap: int | None = None
) -> int:
    """The number of ``line`` events executed by frames under ``root_dir`` while ``fn()`` runs.

    ``root_dir`` is a directory (every file below it counts) or a single file. ``cap``, when given,
    stops the run as soon as the count passes it, with :class:`WorkCapExceeded`.
    """
    previous = sys.gettrace()
    if previous is not None:
        raise ForeignTracerError(
            f"T-0129: another tracer is installed ({previous!r}: coverage, a debugger, or a "
            "Hypothesis test); this row counts traced line events and cannot count under another "
            "tracer. Run it without the tracer."
        )
    root = os.path.realpath(os.fspath(root_dir))
    assert os.path.exists(root), f"T-0129: the work-count root {root!r} does not exist"
    counted = _under(root)
    count = 0
    exceeded: WorkCapExceeded | None = None

    def local(frame: FrameType, event: str, arg: Any) -> Any:
        nonlocal count, exceeded
        if event == "line":
            count += 1
            if cap is not None and count > cap:
                exceeded = WorkCapExceeded(count, cap)
                raise exceeded
        return local

    def global_(frame: FrameType, event: str, arg: Any) -> Any:
        return local if counted(frame.f_code.co_filename) else None

    sys.settrace(global_)
    try:
        fn()
    finally:
        sys.settrace(previous)
    if exceeded is not None:
        raise exceeded  # the code under test caught it: the count is still over the cap
    return count
