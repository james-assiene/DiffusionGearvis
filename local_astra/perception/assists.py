"""Pixel-first assist state: the ingress filter and the per-turn snapshot (docs/12 §3, D12a).

The model sees pixels. DOM/page text, the PDF text layer, spreadsheet cells and the accessibility
tree are *assists*: all off until the user ticks a box, and the badge is logged with every turn so
a number measured with DOM text on can never be compared to a pixel-only one by accident.

``AssistState`` is the one authority for those flags in the Conductor:

* ``toggle`` applies ``astra.assist.toggle`` (a whole flag set, never a per-kind merge) and echoes
  the effective state;
* ``accept_frame_text`` is the *ingress* filter — text for a kind whose flag is off is counted and
  dropped, never buffered, so an eager or buggy HUD cannot put it in front of the model (defence
  in depth behind the prompt-side filter in ``conductor/prompt_builder.py``);
* ``freeze`` hands one turn an immutable snapshot, so a box ticked mid-answer changes nothing for
  the turn in flight;
* ``texts_for`` is that turn's assist payload.

Like the turn machine this module is pure: no clock, no asyncio, no I/O. Timestamps arrive as
arguments (``ts_ms`` on the message, ``at_ms`` on the read).
"""

from __future__ import annotations

from collections.abc import Mapping

from local_astra.contracts.perception import AssistFlags, FrameText
from local_astra.contracts.protocol import AstraAssistState, AstraAssistToggle, AstraFrameText
from local_astra.perception.policy import RING_BUFFER_S

#: Assist kinds in ``AssistFlags`` declaration order. That order is what ``texts_for`` returns
#: (deterministic, so the prompt suffix does not inherit arrival order) and the keys of the drop
#: counter. ``FrameText.kind`` values and ``AssistFlags`` field names are the same four strings.
ASSIST_KINDS: tuple[str, ...] = tuple(AssistFlags.model_fields)

#: Buffered assist text shares the frame ring's horizon (.claude/rules/perception-memory.md).
ASSIST_HORIZON_MS = RING_BUFFER_S * 1000


def _is_on(flags: AssistFlags, kind: str) -> bool:
    """Fail closed: a real assist kind whose flag is on, and nothing else (A10).

    ``kind`` is whatever the wire said. ``AssistFlags`` also carries methods (``badge``,
    ``any_on``) whose bound objects are truthy, and the badge spellings (``dom``, ``a11y``) are
    not field names at all — so a bare ``getattr`` would admit a method or raise on an unknown
    name. Membership first turns both into a drop.
    """
    return kind in ASSIST_KINDS and bool(getattr(flags, kind))


class AssistState:
    """All assists off until an ``astra.assist.toggle`` says otherwise (criterion 1).

    Constructed with no arguments on purpose: a config key or an environment variable that
    pre-armed an assist would make a recording dishonest without anybody ticking a box.
    """

    def __init__(self) -> None:
        self._flags = AssistFlags()
        self._snapshot: AssistFlags | None = None
        self._buffer: dict[str, list[FrameText]] = {kind: [] for kind in ASSIST_KINDS}
        self._dropped: dict[str, int] = dict.fromkeys(ASSIST_KINDS, 0)
        self._newest_ms: int | None = None

    @property
    def flags(self) -> AssistFlags:
        """The flags in force *now* — what the next ``freeze()`` will hand the next turn."""
        return self._flags

    @property
    def dropped(self) -> Mapping[str, int]:
        """Messages refused at ingress, per kind; zero for kinds never dropped (read-only copy).

        A message whose ``kind`` is not one of the four is counted under the name it claimed.
        """
        return dict(self._dropped)

    def state_message(self) -> AstraAssistState:
        """The ``astra.assist.state`` echo for the flags in force; the badge is asked of them."""
        return AstraAssistState(flags=self._flags, badge=self._flags.badge())

    def toggle(self, message: AstraAssistToggle) -> AstraAssistState:
        """Replace the flag set wholesale and purge what the now-off assists captured.

        Purging is criterion 4: filtering on read only would leave the DOM dump in the buffer, so
        ticking the box again — or any later turn whose snapshot includes the kind — would
        resurrect text the user unticked.
        """
        self._flags = message.flags
        for kind in ASSIST_KINDS:
            if not _is_on(self._flags, kind):
                self._buffer[kind].clear()
        return self.state_message()

    def accept_frame_text(self, message: AstraFrameText) -> FrameText | None:
        """Admit assist text iff its kind is on; otherwise count it and drop it unbuffered."""
        kind = message.kind
        if not _is_on(self._flags, kind):
            # ``dropped`` is keyed by plain ``str``, so a kind the wire invented is counted under
            # its own name: the counter stays honest about what was refused instead of folding it
            # into one of the four, and the four keep their zero entry.
            self._dropped[kind] = self._dropped.get(kind, 0) + 1
            return None
        text = FrameText(
            source=message.source,
            ts_ms=message.ts_ms,
            kind=kind,
            text=message.text,
            url=message.url,
        )
        self._buffer[kind].append(text)
        self._forget_older_than_the_horizon(text.ts_ms)
        return text

    def clear_buffers(self) -> None:
        """T-0162 I-2: drop every held assist text (a session reset); the flags and drop counts stay."""
        for held in self._buffer.values():
            held.clear()

    def freeze(self) -> AssistFlags:
        """Snapshot the flags for one turn (criterion 5).

        ``AssistFlags`` is a frozen contract and ``toggle`` rebinds rather than mutates, so the
        value returned here is one a later toggle cannot reach.
        """
        self._snapshot = self._flags
        return self._snapshot

    def texts_for(self, at_ms: int) -> tuple[FrameText, ...]:
        """This turn's assist payload: newest entry at or before ``at_ms``, one per on kind.

        The *frozen* snapshot decides which kinds are on, not the live flags — a box ticked while
        the answer is being spoken takes effect from the next ``freeze()``. Before any ``freeze()``
        there is no snapshot and nothing is admitted: a caller that forgot to freeze gets
        pixel-only, which is the safe default.

        The horizon is a window on *this turn's* time, not on the last ingest (A9): a DOM dump
        from ten minutes ago is not current context just because the HUD went quiet after it, so
        only ``at_ms - ASSIST_HORIZON_MS <= ts_ms <= at_ms`` is returned. The ingest-time prune
        bounds the buffer; this bounds what a turn can see.
        """
        snapshot = self._snapshot
        if snapshot is None:
            return ()
        oldest_ms = at_ms - ASSIST_HORIZON_MS
        out: list[FrameText] = []
        for kind in ASSIST_KINDS:
            if not _is_on(snapshot, kind):
                continue
            newest: FrameText | None = None
            for text in self._buffer[kind]:
                if not oldest_ms <= text.ts_ms <= at_ms:
                    continue
                if newest is None or text.ts_ms >= newest.ts_ms:
                    newest = text
            if newest is not None:
                out.append(newest)
        return tuple(out)

    def _forget_older_than_the_horizon(self, ts_ms: int) -> None:
        """Bound the buffer to the ring horizon, measured against the newest timestamp seen."""
        self._newest_ms = ts_ms if self._newest_ms is None else max(self._newest_ms, ts_ms)
        cutoff = self._newest_ms - ASSIST_HORIZON_MS
        for kind in ASSIST_KINDS:
            self._buffer[kind] = [t for t in self._buffer[kind] if t.ts_ms >= cutoff]
