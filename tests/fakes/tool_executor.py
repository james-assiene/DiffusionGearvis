"""A recording ``ToolExecutor`` (T-0091 criterion 7; T-0080d default 1: "``tests/fakes`` holds a recording fake").

It records every ``ToolInvocation`` it is given and answers with a scripted ``ToolOutcome`` per ``call_id`` or, by
default, a ``dry_run`` outcome that describes nothing. It never touches a device, a socket or a file. The "real"
side of its contract row is ``DryRunToolExecutor`` (CPU, no service), so both run in the fast suite.

T-0114 (the tool-result loop runs the executor in a task, behind a session-clock timeout) adds two modes, both off by
default so every earlier row sees exactly the executor it saw before:

- **hanging** (``hang=True`` for every call, or a collection of call ids): ``execute`` waits until ``release()``.
  It is cancellable: a cancel while waiting is recorded in ``cancelled`` and re-raised, so the task ends cancelled.
  ``shrug_cancel=True`` models a misbehaving executor that records the cancel and keeps waiting, then returns its
  outcome after ``release()`` -- the only way a result can arrive *after* the session gave up on it (T-0114
  criterion 12, "a result that arrives later is noted as stale").
- **raising** (``raise_=exc``): ``execute`` records the invocation and raises ``exc``.

The waits are ``asyncio.Event`` waits only: no clock, no sleep, so a hanging call is cut by whoever cancels it (the
session's timeout on the session clock), never by wall time here.

``local_astra.contracts.toolcall`` is imported lazily, inside ``execute``, so importing this module never fails while
T-0091 is unimplemented (its rows are red on their own assertions).
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from typing import Any


class RecordingToolExecutor:
    """``mode`` is the T-0080d switch value it stands in for (``dry_run`` by default)."""

    def __init__(
        self,
        outcomes: Mapping[str, Any] | None = None,
        *,
        mode: str = "dry_run",
        hang: bool | Collection[str] = False,
        shrug_cancel: bool = False,
        raise_: BaseException | None = None,
    ) -> None:
        self.mode = mode
        self.invocations: list[Any] = []
        self._outcomes = dict(outcomes or {})
        self._hang = hang
        self.shrug_cancel = shrug_cancel
        self.raise_ = raise_
        #: T-0114: call ids whose hanging ``execute`` saw a cancel, in order (a shrugged cancel is recorded too)
        self.cancelled: list[str] = []
        #: T-0114: call ids whose ``execute`` returned an outcome, in order
        self.returned: list[str] = []
        self._released: asyncio.Event | None = None
        self._release_requested = False

    # --- T-0114: the hanging mode -----------------------------------------------------------------------------

    def hangs(self, call_id: str) -> bool:
        if isinstance(self._hang, bool):
            return self._hang
        return call_id in self._hang

    def release(self) -> None:
        """Let every hanging call (now and later) return its outcome."""
        self._release_requested = True
        if self._released is not None:
            self._released.set()

    async def _wait_released(self, call_id: str) -> None:
        if self._released is None:
            self._released = asyncio.Event()
            if self._release_requested:
                self._released.set()
        while not self._released.is_set():
            try:
                await self._released.wait()
            except asyncio.CancelledError:
                self.cancelled.append(call_id)
                if not self.shrug_cancel:
                    raise

    # --- the executor -----------------------------------------------------------------------------------------

    async def execute(self, inv: Any) -> Any:
        from local_astra.contracts.toolcall import ToolOutcome

        self.invocations.append(inv)
        if self.raise_ is not None:
            raise self.raise_
        if self.hangs(inv.call_id):
            await self._wait_released(inv.call_id)
        scripted = self._outcomes.get(inv.call_id)
        self.returned.append(inv.call_id)
        if scripted is not None:
            return scripted
        return ToolOutcome(
            call_id=inv.call_id,
            status="dry_run",
            action=f"recorded {inv.name}",
            reason=None,
            resolved={},
        )
