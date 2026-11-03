"""T-0090 ruling U10 as a Hypothesis property: the turn journal's tool calls, by first-seen order.

docs/tasks/T-0090.md U10 replaces U7–U9 with one rule:

* the journal keeps the turn ids in the order they were first seen (newest plus the 8 before it);
* the *current* turn is the newest first-seen id; a turn is *superseded* iff seen and not current;
* a tool call for a superseded turn is published live but neither recorded nor allowed to clear
  anything;
* a tool call for the current turn is recorded, and reopens the journal if needed (this covers the
  turn a departure clear left);
* departure clears and stale non-tool messages never change the order.

The reference model below is that rule and nothing else, written from the ruling (it does not read
``hud_server``). The journal is driven directly through ``TurnJournal.record`` and, for the
departure, ``TurnJournal.clear()`` — the exact call ``HudServer._on_last_gone`` makes (T-0037
criterion 4); the departure over the real server is covered by the U10 rows in
``tests/contract/test_hud_hello_replay_wire.py``.

**Excluded, on purpose (U10, last bullet):** what a stale non-tool message does to the journal's
*content* is the deferred ``spoken.played`` follow-up. Today it reopens the old turn and wipes the
current turn's lines and cards; the follow-up may keep them. So everything of the current turn
published *before* the last stale non-tool message is "may be replayed" (an upper bound), and
everything after it is "must be replayed" (a lower bound). Everything else is exact.

Regime: CPU, no I/O; ``derandomize=True``, 300 examples, sequences of up to 40 publications over at
most 9 distinct turn ids (so the 8-turn memory bound is never the deciding factor — it is pinned
by ``test_the_last_eight_superseded_turns_are_remembered`` instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.protocol import (
    AstraSpoken,
    AstraSpokenPlayed,
    AstraTranscript,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState

TOOL_CALL_TYPE = "astra.tool.call"
STATUSES = ("proposed", "repairing", "dry_run", "executed", "refused", "cancelled")
#: U10's memory: "the newest plus the 8 before it". The property stays inside it.
MAX_TURNS = 9
CALL_IDS = 3  # per turn: far below REPLAY_MAX_TOOL_CALLS, so eviction never decides anything


def _journal() -> Any:
    from local_astra.conductor.hud_server import TurnJournal

    return TurnJournal()


def _tool_call(turn: str, cid: int, status: str, n: int) -> Any:
    from local_astra.contracts import protocol

    model = getattr(protocol, "AstraToolCall", None)
    assert model is not None, "T-0090 criterion 1: contracts/protocol.py gains AstraToolCall"
    return model(
        turn_id=turn,
        call_id=f"{turn}-c{cid}",
        name="scroll",
        # ``n`` is the publication's sequence number: it tells which message for a call_id is
        # the one replayed, without comparing anything the journal computed.
        arguments={"direction": "down", "n": n},
        status=status,
        source="draft",
    )


def _line(kind: int, turn: str, n: int) -> Any:
    """A non-tool message for ``turn``, tagged with the sequence number ``n``."""
    if kind == 0:
        return AstraTranscript(turn_id=turn, text=f"q{n}", final=True)
    if kind == 1:
        return AstraSpoken(turn_id=turn, utterance_id=f"u{n}", text=f"line {n}.")
    if kind == 2:
        return AstraSpokenPlayed(turn_id=turn, played_ms=n, utterances=[])
    return AstraTurnState(state=TurnState.IDLE, turn_id=turn)


def _tag(m: Any) -> tuple:
    """What identifies one publication in a replay: its type, turn and sequence number."""
    if m.type == TOOL_CALL_TYPE:
        return (m.type, m.turn_id, m.call_id, m.status, m.arguments.get("n"))
    if isinstance(m, AstraTranscript):
        return (m.type, m.turn_id, m.text)
    if isinstance(m, AstraSpoken):
        return (m.type, m.turn_id, m.utterance_id)
    if isinstance(m, AstraSpokenPlayed):
        return (m.type, m.turn_id, m.played_ms)
    return (m.type, getattr(m, "turn_id", None))


# --- the reference model of U10 ------------------------------------------------------------------


@dataclass
class Reference:
    """U10 over turn ids, plus what of the current turn the replay must / may hold."""

    order: list[str] = field(default_factory=list)  # first-seen order; current = order[-1]
    #: current-turn publications since the turn opened or the last departure: the upper bound
    calls: dict[str, tuple] = field(default_factory=dict)  # call_id -> latest tag, first-seen order
    lines: list[tuple] = field(default_factory=list)
    #: …and since the last stale non-tool message too: the lower bound (U10 exclusion)
    firm_calls: set[str] = field(default_factory=set)
    firm_lines: list[tuple] = field(default_factory=list)
    stale_since_open: bool = False

    @staticmethod
    def in_replay_order(tags: list[tuple]) -> list[tuple]:
        """T-0037's replay order: the transcript, the spoken lines, then the played reports."""
        rank = {"astra.transcript": 0, "astra.spoken": 1, "astra.spoken.played": 2}
        return sorted(tags, key=lambda t: rank[t[0]])  # stable: publish order within a kind

    @property
    def current(self) -> str | None:
        return self.order[-1] if self.order else None

    def superseded(self) -> list[str]:
        return self.order[:-1]

    def _open(self, turn: str) -> None:
        self.order.append(turn)
        self.calls, self.lines, self.firm_calls, self.firm_lines = {}, [], set(), []
        self.stale_since_open = False

    def depart(self) -> None:
        # T-0037 criterion 4: everything published for the turn is forgotten; U10: the order is not
        # touched, so the departed turn stays current and a call for it reopens the journal.
        self.calls, self.lines, self.firm_calls, self.firm_lines = {}, [], set(), []
        self.stale_since_open = False

    def stale(self) -> None:
        # U10: the order is unchanged. What the stale message does to the content is the deferred
        # follow-up's, so the lower bound restarts here and the upper bound is kept.
        self.firm_calls, self.firm_lines = set(), []
        self.stale_since_open = True

    def line(self, turn: str, m: Any) -> None:
        if turn not in self.order:
            self._open(turn)
        assert turn == self.current, "the generator only sends lines for the current or a new turn"
        if isinstance(m, AstraTurnState):
            return  # an IDLE state is nobody's catch-up (T-0037)
        tag = _tag(m)
        if isinstance(m, AstraTranscript):  # latest-wins
            self.lines = [t for t in self.lines if t[0] != m.type] + [tag]
            self.firm_lines = [t for t in self.firm_lines if t[0] != m.type] + [tag]
        else:
            self.lines.append(tag)
            self.firm_lines.append(tag)

    def call(self, m: Any) -> bool:
        """Returns whether U10 records it (False: a superseded turn's call, published live only)."""
        if m.turn_id not in self.order:
            self._open(m.turn_id)
        if m.turn_id != self.current:
            return False
        self.calls[m.call_id] = _tag(m)  # dict keeps first-seen order: the card keeps its place
        self.firm_calls.add(m.call_id)
        return True


# --- the generator -------------------------------------------------------------------------------

#: (op, a, b, c): op 0 open a new turn, 1 a line for the current turn, 2 a stale non-tool message
#: for a superseded turn, 3 a departure clear, 4 a call for the current turn, 5 a call for a
#: superseded turn, 6 a call for a never-seen turn.
OPS = st.lists(
    st.tuples(
        st.integers(0, 6),
        st.integers(0, 3),
        st.integers(0, CALL_IDS - 1),
        st.sampled_from(STATUSES),
    ),
    max_size=40,
)


def _show(m: Any) -> str:
    return "DEPART" if m == "DEPART" else repr(_tag(m))


@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(OPS)
def test_u10_replayed_tool_calls_are_the_current_turns_latest_per_call_id(ops):
    journal = _journal()
    ref = Reference()
    fresh = 0
    trace: list[str] = []

    def publish(m: Any) -> list:
        trace.append(_show(m))
        journal.record(m)
        return journal.messages()

    for n, (op, a, cid, status) in enumerate(ops):
        old = ref.superseded()
        if op in (0, 6) or ref.current is None:
            if len(ref.order) == MAX_TURNS:
                continue  # stay inside U10's memory bound
            fresh += 1
            turn = f"t{fresh}"
            if op == 6:
                m = _tool_call(turn, cid, status, n)
                assert ref.call(m)
            else:
                m = _line(a % 2, turn, n)  # a turn opens with a transcript or a spoken line
                ref.line(turn, m)
            publish(m)
        elif op == 1:
            m = _line(a, ref.current, n)
            ref.line(ref.current, m)
            publish(m)
        elif op == 2:
            if not old:
                continue
            m = _line(a, old[(a + cid) % len(old)], n)
            ref.stale()
            publish(m)
        elif op == 3:
            trace.append("DEPART")
            journal.clear()  # = HudServer._on_last_gone (T-0037 criterion 4)
            ref.depart()
        elif op == 4:
            m = _tool_call(ref.current, cid, status, n)
            assert ref.call(m)
            publish(m)
        else:  # op == 5: a late call for a superseded turn
            if not old:
                continue
            m = _tool_call(old[(a + cid) % len(old)], cid, status, n)
            before = [_tag(x) for x in journal.messages()]
            assert not ref.call(m)
            after = [_tag(x) for x in publish(m)]
            assert after == before, (
                f"U10: a tool call for superseded turn {m.turn_id} (current {ref.current}) is "
                "neither recorded nor allowed to clear anything, so the replay must not change.\n"
                f"before: {before}\nafter:  {after}\nsequence: {trace}"
            )

        replay = journal.messages()
        calls = [_tag(m) for m in replay if m.type == TOOL_CALL_TYPE]
        others = [_tag(m) for m in replay if m.type != TOOL_CALL_TYPE]
        lines = [t for t in others if t[1] == ref.current]
        where = f"current {ref.current}, order {ref.order}\nsequence: {trace}"

        # 1. Every replayed tool call is the current turn's, and the latest for its call_id.
        wrong_turn = [t for t in calls if t[1] != ref.current]
        assert not wrong_turn, (
            f"U10: only the current turn's tool calls are replayed; got {wrong_turn}\n{where}"
        )
        stale_versions = [t for t in calls if ref.calls.get(t[2]) != t]
        assert not stale_versions, (
            "criterion 3 / U10: each replayed call is the latest message for its call_id; "
            f"replayed {stale_versions}, latest {ref.calls}\n{where}"
        )
        # 2. Exactly the current turn's calls: all of them (in first-seen order) unless a stale
        #    message intervened, and then at least every one published since it.
        if not ref.stale_since_open:
            assert calls == list(ref.calls.values()), (
                "U10: the replayed tool calls are exactly the latest per call_id of the current "
                f"turn, in first-seen order;\nreplayed {calls}\nexpected {list(ref.calls.values())}"
                f"\n{where}"
            )
        else:
            missing = ref.firm_calls - {t[2] for t in calls}
            assert not missing, (
                "U10: a current-turn call published after the last stale message must be "
                f"replayed; missing {sorted(missing)}, replayed {calls}\n{where}"
            )
        # 3. No current-turn line is lost (beyond the excluded stale-message wipe).
        if not ref.stale_since_open:
            expected = ref.in_replay_order(ref.lines)
            assert others == expected, (
                "U10: the current turn's lines are replayed whole and alone — no superseded-turn "
                f"call removes one or brings its own back;\nreplayed {others}\nexpected {expected}"
                f"\n{where}"
            )
        else:
            assert all(t in lines for t in ref.firm_lines), (
                "U10: a current-turn line published after the last stale message must be "
                f"replayed;\nreplayed {lines}\nrequired {ref.firm_lines}\n{where}"
            )
            assert all(t in ref.lines for t in lines), (
                f"U10: nothing but the current turn's own lines; replayed {lines}\n{where}"
            )
        assert journal.truncated is False, (
            f"nothing here reaches a bound, so nothing may be declared lost\n{where}"
        )
