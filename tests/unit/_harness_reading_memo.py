"""T-0110 part A: a per-harness-instance memo of the T-0098 harness's ``ui_reading`` (not collected).

``build_report`` on a hand-built run calls ``ui_reading`` once per trial line; the fixture runs repeat the same few
commit lists hundreds of times, and each call runs a fresh real ``UiGate`` and the vendored jsonschema (about 2.5 ms).
``ui_reading`` depends on the commit list and on its keyword arguments other than ``request_id`` (the premise, checked
on every fixture commit list by ``tests/unit/test_t0110_speed_guards.py``), so a reading computed once per distinct
``(commits, other arguments)`` is the reading every later identical call would compute.

Scope (T-0110 criterion 3):

- ``memoize_ui_reading(h)`` replaces ``ui_reading`` on the one harness module object ``h`` (``build_report`` looks it
  up as a module global), and on nothing else. The T-0098 loader and the git-blob ``old_harness()`` build a fresh module
  per call, so the memo lives exactly as long as the row's harness object: it is never process-global and never shared
  between rows.
- It is installed only by the report-building helpers (``report_c``, ``report_103``, ``report_108a``) and by the report
  building of T-0103's and T-0108a's ``test_prior_tables_untouched``. A row that calls ``h.ui_reading`` directly gets
  its own unwrapped harness and runs unmemoized.
- Every call returns a deep copy, so a caller that mutates its reading cannot change the next caller's.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from types import ModuleType
from typing import Any

#: The one argument ``ui_reading`` provably ignores (T-0110 criterion 4's premise); everything else is in the key.
IGNORED = "request_id"


class ReadingMemo:
    """Wraps one ``ui_reading``: the key is the commit list as canonical JSON plus every other argument but
    ``request_id``. ``calls`` and ``misses`` count what happened, for the guard rows."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.calls = 0
        self.misses = 0
        self._memo: dict[str, Any] = {}

    def key(self, commits: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
        rest = {k: v for k, v in kwargs.items() if k != IGNORED}
        try:
            return json.dumps([commits, list(args), rest], sort_keys=True)
        except (TypeError, ValueError):
            return None  # not canonical JSON: never memoized

    def __call__(self, commits: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        k = self.key(commits, args, kwargs)
        if k is None:
            self.misses += 1
            return self.fn(commits, *args, **kwargs)
        if k not in self._memo:
            self.misses += 1
            self._memo[k] = self.fn(commits, *args, **kwargs)
        return copy.deepcopy(self._memo[k])


def memoize_ui_reading(h: ModuleType) -> ReadingMemo:
    """Install a fresh ``ReadingMemo`` over ``h.ui_reading`` on this harness object only, and return it."""
    memo = ReadingMemo(h.ui_reading)
    h.ui_reading = memo  # type: ignore[attr-defined]
    return memo
