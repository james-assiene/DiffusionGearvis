"""Command executors: the impure half of docs/02 §4. Task T-0007.

The turn machine is pure and returns *data* (``contracts/commands.py``); this module is the one
place that turns that data into calls on the injected services. Two properties the tests pin:

* **Totality.** Every member of the ``Command`` union has a named handler in ``_TABLE``. A command
  with no handler raises ``UnknownCommand`` instead of being dropped silently — the same guard
  T-0003 added for ``EVENT_TYPES``, which found three rowless event types then.
* **Idempotence per cause.** ``llm.abort``, ``tts.cancel`` and ``playback.flush`` may be emitted by
  more than one row for the same interruption. Executing one of them twice performs the effect
  once, raises nothing, and produces no second terminal event — a duplicated ``playback.stopped``
  would give the machine a second history commit for one barge-in.

Nothing here blocks the pump: work that talks to the LLM or the TTS goes into a task
(``.claude/rules/conductor.md``: nothing in the pump may block the event loop for more than
20 ms). That is why ``playback.pause`` can be the first thing a barge-in hop does even while the
TTS client is stuck.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from local_astra.contracts import commands as C

if TYPE_CHECKING:  # pragma: no cover - typing only, and it would be a cycle at runtime
    from local_astra.conductor.session import Session

#: A handler takes the command it is registered for and performs its effect.
Handler = Callable[[Any], Awaitable[None]]


class UnknownCommand(TypeError):
    """Raised for an object that is not a member of the ``Command`` union."""


class CommandExecutor:
    """Dispatch table over ``contracts/commands.Command``; the effects live on the session."""

    def __init__(self, session: Session) -> None:
        self._s = session
        self._table: dict[type[Any], Handler] = {
            C.SnapshotFrame: self.snapshot_frame,
            C.PrefillOpen: self.prefill_open,
            C.PrefillReissue: self.prefill_reissue,
            C.PrefillAbort: self.prefill_abort,
            C.FreezeAfter: self.freeze_after,
            C.Unfreeze: self.unfreeze,
            C.CheckOwner: self.check_owner,
            C.LlmStart: self.llm_start,
            C.LlmAbort: self.llm_abort,
            C.LlmContinue: self.llm_continue,
            C.CaptionerAbort: self.captioner_abort,
            C.PlayClip: self.play_clip,
            C.StopClip: self.stop_clip,
            C.ArmTimer: self.arm_timer,
            C.CancelTimer: self.cancel_timer,
            C.TtsSpeak: self.tts_speak,
            C.TtsCancel: self.tts_cancel,
            C.PlaybackPause: self.playback_pause,
            C.PlaybackResume: self.playback_resume,
            C.PlaybackFlush: self.playback_flush,
            C.TruncateHistoryToPlayed: self.truncate_history_to_played,
            C.CommitHistory: self.commit_history,
            C.ScheduleSummary: self.schedule_summary,
            C.HudState: self.hud_state,
            C.HudError: self.hud_error,
            C.Degrade: self.degrade,
            C.ToolRepairStart: self.tool_repair_start,
            C.ToolExecute: self.tool_execute,
            C.ToolCancel: self.tool_cancel,
            C.ResetSession: self.reset_session,
        }

    # --- dispatch ---------------------------------------------------------------------------

    def handler_for(self, command: object) -> Handler:
        """The named executor for ``command``; never ``None``, never a silent drop."""
        handler = self._table.get(type(command))
        if handler is None:
            raise UnknownCommand(
                f"no executor for {type(command).__name__}: every member of the Command union "
                "needs one (docs/tasks/T-0007.md criterion 2)"
            )
        return handler

    async def execute(self, command: C.Command) -> None:
        await self.handler_for(command)(command)

    # --- perception and the anticipatory prefill ---------------------------------------------

    async def snapshot_frame(self, command: C.SnapshotFrame) -> None:
        """The frame at speech onset (``at_ms``), never ``latest``, and the prefix built from it."""
        await self._s.snapshot_frame(command.at_ms)

    async def prefill_open(self, command: C.PrefillOpen) -> None:
        await self._s.open_prefill(command.request_id, reissue=False)

    async def prefill_reissue(self, command: C.PrefillReissue) -> None:
        """Rate limited here, not in the machine: the machine is pure and owns no clock."""
        await self._s.open_prefill(command.request_id, reissue=True)

    async def prefill_abort(self, command: C.PrefillAbort) -> None:
        await self._s.abort_request(command.request_id)

    # --- endpointing ------------------------------------------------------------------------

    async def freeze_after(self, command: C.FreezeAfter) -> None:
        self._s.freeze_transcript_after(command.t_eot_ms)

    async def unfreeze(self, command: C.Unfreeze) -> None:
        self._s.freeze_transcript_after(None)

    async def check_owner(self, command: C.CheckOwner) -> None:
        """Record the request. The veto itself (``user.not_owner``) is T-0023's family; this task
        only owes the seam, and a check nobody implemented must not invent a verdict."""
        self._s.owner_checks.append((command.from_ms, command.to_ms))

    # --- the LLM ----------------------------------------------------------------------------

    async def llm_start(self, command: C.LlmStart) -> None:
        self._s.begin_generate(command.request_id, command.transcript)

    async def llm_abort(self, command: C.LlmAbort) -> None:
        await self._s.abort_request(command.request_id)

    async def llm_continue(self, command: C.LlmContinue) -> None:
        """T-0114: the next loop step's request, in its own task. Inert (one note, no other effect) with tools
        off or the loop off, where no machine row emits it."""
        if self._s.tool_kit is None or not self._s.machine.policy.tool_loop:
            self._note_inert(command)
            return
        self._s.begin_step(command)

    async def captioner_abort(self, command: C.CaptionerAbort) -> None:
        await self._s.abort_captioner()

    # --- clips ------------------------------------------------------------------------------

    async def play_clip(self, command: C.PlayClip) -> None:
        await self._s.play_clip(command.kind)

    async def stop_clip(self, command: C.StopClip) -> None:
        await self._s.stop_clip()

    # --- timers: the session's own tick, and nothing else -------------------------------------

    async def arm_timer(self, command: C.ArmTimer) -> None:
        self._s.reschedule_tick()

    async def cancel_timer(self, command: C.CancelTimer) -> None:
        self._s.reschedule_tick()

    # --- speech and playback ------------------------------------------------------------------

    async def tts_speak(self, command: C.TtsSpeak) -> None:
        """Queue one utterance. ``text`` is the normalised rendering handed to the backend,
        ``source`` the committed substring ``play`` carries; a latched ``text=""`` is skipped."""
        self._s.enqueue_speech(command.utterance_id, command.text, command.source)

    async def tts_cancel(self, command: C.TtsCancel) -> None:
        await self._s.cancel_speech()

    async def playback_pause(self, command: C.PlaybackPause) -> None:
        await self._s.playback_pause()

    async def playback_resume(self, command: C.PlaybackResume) -> None:
        await self._s.playback_resume()

    async def playback_flush(self, command: C.PlaybackFlush) -> None:
        await self._s.playback_flush()

    # --- history and telemetry ------------------------------------------------------------

    async def truncate_history_to_played(self, command: C.TruncateHistoryToPlayed) -> None:
        """Carry the *module's* answer through: nothing here recomputes what was played."""
        self._s.note_truncation(command.played_ms, command.played_text)

    async def commit_history(self, command: C.CommitHistory) -> None:
        await self._s.commit_history(user_only=command.user_only)

    async def schedule_summary(self, command: C.ScheduleSummary) -> None:
        """Memory summarisation is its own task; the session only notes that it was asked."""
        self._s.summaries_scheduled += 1

    async def hud_state(self, command: C.HudState) -> None:
        """One ``astra.turn.state`` per state change, through the session's typed publisher.

        The publisher is synchronous and never awaited: a page that stopped reading is its own
        connection's problem, never this hop's (docs/tasks/T-0028.md criterion 12).
        """
        self._s.publish_turn_state(command.state, command.turn_id)

    async def hud_error(self, command: C.HudError) -> None:
        self._s.publish_turn_error(command.message)

    async def degrade(self, command: C.Degrade) -> None:
        self._s.degrade(command.mode)

    async def reset_session(self, command: C.ResetSession) -> None:
        """T-0162 A-3: the session clears its state and wipes memory once the cut turn's line is written."""
        await self._s.reset_session(command)

    # --- tools (T-0092c criterion 5: the session with tools on; inert, one note each, with tools off)

    async def tool_repair_start(self, command: C.ToolRepairStart) -> None:
        """The repair runs in its own task; this hop only starts it (it never blocks the pump)."""
        if self._s.tool_kit is None:
            self._note_inert(command)
            return
        self._s.start_tool_repair(command)

    async def tool_execute(self, command: C.ToolExecute) -> None:
        """The dry run, awaited inline: it does no I/O (an executor that does is a later task).

        T-0114: with the loop on it runs in a task behind a session-clock timeout and answers with a
        ``ToolResult`` (``Session.start_tool_execute``); with the loop off this path is unchanged."""
        if self._s.tool_kit is None:
            self._note_inert(command)
            return
        if self._s.machine.policy.tool_loop:
            self._s.start_tool_execute(command)
            return
        await self._s.execute_tool(command)

    async def tool_cancel(self, command: C.ToolCancel) -> None:
        if self._s.tool_kit is None:
            self._note_inert(command)
            return
        self._s.cancel_tool(command)

    def _note_inert(
        self, command: C.ToolRepairStart | C.ToolExecute | C.ToolCancel | C.LlmContinue
    ) -> None:
        """Record that the machine asked for a tool effect nobody performs yet (no other effect)."""
        self._s._log("note", {"what": command.type, "inert": True, **command.model_dump()})
