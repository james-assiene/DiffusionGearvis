"""T-0128c C2: the pure gates of the `/launch` denoising smoke, on hand-made evidence.

``local_astra/eval/denoise_smoke.py`` (C implementer) decides the Regime's gates G1-G7 from saved
evidence only; the GPU driver ``scripts/gpu/t0128c_denoise_smoke.py`` collects that evidence and
writes it to ``logs/t0128c-smoke-<sha>/evidence.json`` (never ``artifacts/``). These rows pin what
each gate must catch and, for every guard, one row where a too-wide guard would wrongly fire
(.claude/rules/tests.md). Every expected value is written by hand below.

**The evidence format** (test-author, 2026-10-26; the implementer and the driver follow it):

``evidence`` is one JSON object::

    {
      "turns": [Turn, ...],          # the typed turns, in the order they were typed
      "session_log": [Record, ...],  # the session log's JSONL records, verbatim, in file order
      "release_log": str,            # the engine log text covering the run (the slice after boot)
      "ss_lines": [str, ...],        # `ss -tnp` output lines, verbatim (header line allowed)
      "conductor_pid": int           # the `astra session` pid; only its sockets are gated
    }

    Turn   = {"turn_id": str,        # the session's turn id (read off astra.turn.state)
              "text": str,           # the typed prompt
              "samples": [Sample, ...]}
    Sample = {"t_ms": int,           # driver wall clock, monotonic ms
              "frames": int | None,  # [data-denoise-root] data-denoise-frames, None = attribute absent
              "draws": int | None,   # data-denoise-draws
              "digest": str | None,  # data-denoise-digest
              "label": str | None,   # the Thinking pane's "canvas k · step s/N" label text, None = none
              "malformed": int}      # the page's malformed counter (the "m" of #ignored-count)

    The page's counters are cumulative over the page's lifetime. A turn's FIRST sample is taken
    before its text is sent (the baseline: it carries the previous turn's last frame, B7), the
    rest at 20 Hz until the turn is back to IDLE.

    Record = a session-log line: {"seq", "t_ms", "wall", "session_ms", "kind", "turn_id", "data"};
    a ``wire.out`` record's ``data`` is ``{"n_bytes", "msg", "dropped", "connections"}`` and
    ``data["msg"]`` is the message as published (for ``astra.canvas.step``, A7's compact dict).

**API these rows fix:**

``relayed_frames(session_log) -> list[dict]``
    one ``{"request_id", "state", "turn_id"}`` per ``wire.out`` whose ``msg.type`` is
    ``astra.canvas.step``, in log order; ``state``/``turn_id`` are those of the most recent
    non-replay ``astra.turn.state`` ``wire.out`` before it (``"IDLE"``/``None`` before any).
    A dropped (``dropped: true``) canvas row was still relayed and is listed.
``gates(evidence) -> dict[str, list[str]]``
    keys exactly ``"G1"`` ... ``"G7"``; each value one human-readable problem per finding, ``[]`` =
    pass. Release lines are read with ``local_astra.bench.b16_tap.parse_tap_release_lines``.

    G1  per turn, frames decoded = last non-None ``frames`` minus the baseline's (None = 0) >= 5;
        a problem names the turn id.
    G2  per turn, >= 3 distinct non-None digests and >= 3 distinct non-None labels among the
        samples after the baseline, not counting the baseline's own digest/label (carried over).
    G3  every sample's ``malformed`` is 0; every relayed ``request_id`` equals the ``req_id`` of
        some release line.
    G4  no relayed ``request_id`` whose FIRST relayed frame has state ``IDLE`` (later frames of a
        request admitted in another state may arrive in IDLE: that is A4's final commit, a pass);
        a problem names the request id.
    G5  every ``ss`` line of ``pid=<conductor_pid>`` (exact pid) has loopback local and peer
        addresses (127.0.0.0/8, ``[::1]``, ``[::ffff:127.x.x.x]``); a line of another pid is
        ignored; zero lines of the conductor is a problem (no evidence is not a pass); a problem
        quotes the offending address.
    G6  every turn has >= 1 non-blank non-replay ``astra.spoken`` with its turn id; there is no
        ``astra.error`` in the log at all (``denoise_no_tap`` included).
    G7  at least one release line; every line ``errors == 0`` and ``commit_overrun == 0``
        (``dropped`` is reported, never gated). A malformed release line never passes: either a
        G7 problem or the parser's ``ValueError``.

Regime: CPU, pure Python, hand-built inputs, no I/O; n/a seed.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

MODULE = "local_astra.eval.denoise_smoke"
REPO = Path(__file__).resolve().parents[2]
#: the GPU driver; imported by path for its pure helpers only, never run (main() is not called)
HARNESS = REPO / "scripts" / "gpu" / "t0128c_denoise_smoke.py"
PID = 4242
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")

#: the real id shape the 806da12 trajectory recorded (chatcmpl-<16 hex>-<8 hex>)
RID = {
    "t1": "chatcmpl-b8d75dac2b8f054f-b19b91f7",
    "t2": "chatcmpl-0c1d2e3f40516273-a0b1c2d3",
    "t3": "chatcmpl-9f8e7d6c5b4a3921-0badf00d",
}
CAPTION_RID = "chatcmpl-1111222233334444-55556666"


def smoke() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise AssertionError(
                f"T-0128c C2 not implemented: {MODULE.replace('.', '/')}.py does not exist"
            ) from None
        raise


def fn(name: str) -> Any:
    f = getattr(smoke(), name, None)
    if f is None:
        raise AssertionError(f"T-0128c C2 not implemented: {MODULE}.{name} is missing")
    return f


# --------------------------------------------------------------------------- builders


class Log:
    """A session log under construction: records numbered in the order they are added."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = [
            {
                "seq": 0,
                "t_ms": 0,
                "wall": "2026-10-26T12:00:00+00:00",
                "session_ms": None,
                "kind": "log.started",
                "turn_id": None,
                "data": {"pid": PID, "path": "/x.jsonl"},
            }
        ]

    def add(self, kind: str, data: dict[str, Any], turn_id: str | None = None) -> None:
        n = len(self.records)
        self.records.append(
            {
                "seq": n,
                "t_ms": n * 10,
                "wall": "2026-10-26T12:00:00+00:00",
                "session_ms": None,
                "kind": kind,
                "turn_id": turn_id,
                "data": data,
            }
        )

    def wire(self, msg: dict[str, Any], *, dropped: bool = False, connections: int = 1) -> None:
        self.add(
            "wire.out",
            {"n_bytes": 100, "msg": msg, "dropped": dropped, "connections": connections},
        )

    def state(self, state: str, turn_id: str, *, replay: bool = False) -> None:
        self.wire(
            {
                "type": "astra.turn.state",
                "state": state,
                "turn_id": turn_id,
                "thinking_ms": None,
                "replay": replay,
            }
        )

    def canvas(self, rid: str, idx: int, step: int, *, phase: str = "denoise") -> None:
        # A7's compact row, exactly its ten keys
        self.wire(
            {
                "type": "astra.canvas.step",
                "request_id": rid,
                "canvas_idx": idx,
                "step": step,
                "max_steps": 48,
                "phase": phase,
                "canvas_length": 256,
                "n_committed": 256 if phase == "commit" else 0,
                "n_kept": 0 if phase == "commit" else 100,
                "n_stable": 0 if phase == "commit" else 50,
            }
        )

    def spoken(self, turn_id: str, text: str, *, replay: bool = False) -> None:
        self.wire(
            {
                "type": "astra.spoken",
                "turn_id": turn_id,
                "utterance_id": f"{turn_id}-u0",
                "text": text,
                "replay": replay,
            }
        )

    def error(self, code: str, message: str, *, connections: int = 1) -> None:
        self.wire(
            {"type": "astra.error", "code": code, "message": message}, connections=connections
        )

    def machine_step(
        self, event_type: str, turn_id: str, *, commands: list[dict[str, Any]] | None = None
    ) -> None:
        """A ``machine.step`` record as the session log writes it (shape copied from the second
        /launch's evidence at ffba98c, seq 627: the harness page's ``client.disconnected``)."""
        n = len(self.records)
        self.add(
            "machine.step",
            {
                "event": {"t_ms": n * 10, "type": event_type},
                "state_before": "IDLE",
                "state_after": "IDLE",
                "turn_seq_before": 3,
                "turn_seq_after": 3,
                "commands": commands or [],
            },
            turn_id=turn_id,
        )

    def disconnected(self, turn_id: str) -> None:
        self.machine_step(
            "client.disconnected", turn_id, commands=[{"type": "degrade", "mode": "text_only"}]
        )

    def turn_record(self, turn_id: str) -> None:
        """A ``turn.record`` record (its ``data`` trimmed to the keys a gate could read)."""
        self.add(
            "turn.record",
            {"turn_id": turn_id, "session_id": "1d3620ea56d4", "spans": {}, "events": []},
            turn_id=turn_id,
        )


def sample(
    t_ms: int, frames: int | None, digest: str | None, label: str | None, malformed: int = 0
) -> dict[str, Any]:
    return {
        "t_ms": t_ms,
        "frames": frames,
        "draws": frames,
        "digest": digest,
        "label": label,
        "malformed": malformed,
    }


def release_line(rid: str, *, errors: int = 0, overrun: int = 0, dropped: int = 0) -> str:
    return (
        "(EngineCore pid=326) INFO 10-26 12:00:02 [tap.py:563] canvas tap release: "
        f"req_id={rid} submitted=40 published={40 - dropped} dropped={dropped} "
        f"errors={errors} commit_overrun={overrun}"
    )


SS_HEADER = "State  Recv-Q Send-Q   Local Address:Port    Peer Address:Port Process"


def ss(local: str, peer: str, pid: int = PID, fd: int = 17) -> str:
    return f'ESTAB  0      0        {local}   {peer} users:(("python3",pid={pid},fd={fd}))'


def good() -> dict[str, Any]:
    """Three healthy turns: each THINKING -> 6 denoise frames + 1 commit -> SPEAKING -> IDLE.

    Page samples per turn: a baseline, then six samples whose ``frames`` grow by 2 and whose digest
    and label are new each time (turn 1's baseline is the empty page: all None).
    """
    log = Log()
    turns: list[dict[str, Any]] = []
    frames = 0
    prev_digest: str | None = None
    prev_label: str | None = None
    for k, tid in enumerate(("t1", "t2", "t3")):
        rid = RID[tid]
        log.state("LISTENING", tid)
        log.state("THINKING", tid)
        for s in range(1, 7):
            log.canvas(rid, 0, s * 3)
        log.canvas(rid, 0, 20, phase="commit")
        log.state("SPEAKING", tid)
        log.spoken(tid, f"Answer number {k + 1}.")
        log.state("IDLE", tid)
        t0 = 10_000 * (k + 1)
        samples = [sample(t0, frames or None, prev_digest, prev_label)]
        for i in range(1, 7):
            frames += 2
            prev_digest = f"{tid}d{i}"
            prev_label = f"canvas 0 · step {i * 3}/48"
            samples.append(sample(t0 + 50 * i, frames, prev_digest, prev_label))
        turns.append({"turn_id": tid, "text": f"prompt {k + 1}", "samples": samples})
    return {
        "turns": turns,
        "session_log": log.records,
        "release_log": "\n".join(
            ["INFO unrelated engine line", *(release_line(r) for r in RID.values())]
        ),
        "ss_lines": [
            SS_HEADER,
            ss("127.0.0.1:51234", "127.0.0.1:8001", fd=17),
            ss("[::ffff:127.0.0.1]:8080", "[::ffff:127.0.0.1]:40112", fd=20),
            ss("[::1]:43210", "[::1]:8081", fd=21),
            ss("127.0.0.1:40000", "127.0.0.1:8002", fd=22),
        ],
        "conductor_pid": PID,
    }


def run(evidence: dict[str, Any]) -> dict[str, list[str]]:
    out = fn("gates")(evidence)
    assert isinstance(out, dict), f"gates() must return a dict, got {type(out).__name__}"
    assert set(out) == set(GATES), f"gates() keys must be exactly {GATES}, got {sorted(out)}"
    for k, v in out.items():
        assert isinstance(v, list) and all(isinstance(p, str) for p in v), (
            f"{k} must be a list of problem strings, got {v!r}"
        )
    return out


def only(out: dict[str, list[str]], *failing: str) -> None:
    """The named gates fail and every other gate passes (a fault lands where it belongs)."""
    for g in GATES:
        if g in failing:
            assert out[g], f"{g} should report a problem; all gates: {out}"
        else:
            assert out[g] == [], f"{g} should pass; it reported {out[g]} (all gates: {out})"


def replace_log(ev: dict[str, Any], log: Log) -> dict[str, Any]:
    ev = copy.deepcopy(ev)
    ev["session_log"] = log.records
    return ev


# --------------------------------------------------------------------------- relayed_frames


def test_relayed_frames_carry_the_state_published_before_each_frame():
    """By hand: t1 THINKING then two frames, IDLE, then a third frame of the same request."""
    log = Log()
    log.state("THINKING", "t1")
    log.canvas(RID["t1"], 0, 3)
    log.state("LISTENING", "t1", replay=True)  # a replay to a reconnecting page is not a change
    log.canvas(RID["t1"], 0, 6)
    log.state("IDLE", "t1")
    log.canvas(RID["t1"], 0, 9, phase="commit")
    log.spoken("t1", "not a canvas row")
    assert fn("relayed_frames")(log.records) == [
        {"request_id": RID["t1"], "state": "THINKING", "turn_id": "t1"},
        {"request_id": RID["t1"], "state": "THINKING", "turn_id": "t1"},
        {"request_id": RID["t1"], "state": "IDLE", "turn_id": "t1"},
    ]


def test_relayed_frames_before_any_turn_state_are_idle_and_dropped_rows_still_count():
    log = Log()
    log.wire(
        {
            "type": "astra.canvas.step",
            "request_id": CAPTION_RID,
            "canvas_idx": 0,
            "step": 1,
            "max_steps": 48,
            "phase": "denoise",
            "canvas_length": 256,
            "n_committed": 0,
            "n_kept": 3,
            "n_stable": 0,
        },
        dropped=True,
    )
    assert fn("relayed_frames")(log.records) == [
        {"request_id": CAPTION_RID, "state": "IDLE", "turn_id": None}
    ]


# --------------------------------------------------------------------------- the pass case


def test_healthy_run_passes_every_gate():
    only(run(good()))


# --------------------------------------------------------------------------- G1


def test_g1_a_turn_with_zero_frames_fails_g1_and_names_the_turn():
    ev = good()
    t2 = ev["turns"][1]
    base = t2["samples"][0]
    # the page never moved during t2: every sample equals the baseline
    t2["samples"] = [dict(base, t_ms=base["t_ms"] + 50 * i) for i in range(7)]
    out = run(ev)
    only(out, "G1", "G2")  # no frame also means no new digest or label
    assert any("t2" in p for p in out["G1"]), out["G1"]
    assert not any("t1" in p or "t3" in p for p in out["G1"]), out["G1"]


def test_g1_counts_from_the_baseline_not_from_zero():
    """Adversarial: t3's cumulative counter reads 24..28 (>= 5) but only 4 frames arrived in t3."""
    ev = good()
    t3 = ev["turns"][2]
    base = t3["samples"][0]["frames"]  # 24
    assert base == 24
    for i, s in enumerate(t3["samples"][1:], start=1):
        s["frames"] = base + min(i, 4)  # 25, 26, 27, 28, 28, 28
    out = run(ev)
    assert any("t3" in p for p in out["G1"]), out
    assert out["G2"] == [], out["G2"]


def test_g1_boundary_exactly_five_frames_passes():
    """Too-wide guard row: five frames is the threshold itself (>= 5)."""
    ev = good()
    t1 = ev["turns"][0]
    # baseline None (= 0); last frames value 5
    for i, s in enumerate(t1["samples"][1:], start=1):
        s["frames"] = min(i, 5)
    # later turns' baselines follow the page counter
    later = ev["turns"][1]["samples"][0]
    assert later["frames"] == 12
    delta = 12 - 5
    for t in ev["turns"][1:]:
        for s in t["samples"]:
            s["frames"] -= delta
    only(run(ev))


# --------------------------------------------------------------------------- G2


@pytest.mark.parametrize("field", ["digest", "label"])
def test_g2_a_value_carried_over_from_the_previous_turn_is_not_counted(field):
    """Adversarial: t2 shows t1's last digest (or label), then only two new ones; the other field
    stays healthy, so only the carried-over field can make G2 fail."""
    ev = good()
    t2 = ev["turns"][1]
    carried = t2["samples"][0][field]
    assert carried == {"digest": "t1d6", "label": "canvas 0 · step 18/48"}[field]
    new = {"digest": "t2x{}", "label": "canvas 0 · step {}/48"}[field]
    for i, s in enumerate(t2["samples"][1:], start=1):
        # carried, carried, then new 3, 4, 4, 4: two values this turn produced
        s[field] = carried if i <= 2 else new.format(min(i, 4))
    out = run(ev)
    only(out, "G2")
    assert any("t2" in p for p in out["G2"]), out["G2"]


def test_g2_none_samples_before_the_first_frame_are_not_values_and_three_is_enough():
    """Too-wide guard row: t1 has two empty samples, then exactly 3 digests and 3 labels."""
    ev = good()
    t1 = ev["turns"][0]
    vals = [None, None, "a", "b", "c", "c"]
    for s, v in zip(t1["samples"][1:], vals, strict=True):
        s["digest"] = v
        s["label"] = None if v is None else f"canvas 0 · step {ord(v)}/48"
    t1["samples"][1]["frames"] = None
    t1["samples"][2]["frames"] = None
    only(run(ev))


# --------------------------------------------------------------------------- G3


def test_g3_a_malformed_page_count_fails_g3():
    ev = good()
    ev["turns"][2]["samples"][4]["malformed"] = 1
    only(run(ev), "G3")


def test_g3_a_relayed_id_absent_from_every_release_line_fails_g3():
    ev = good()
    ev["release_log"] = "\n".join(release_line(r) for r in (RID["t1"], RID["t3"]))
    out = run(ev)
    only(out, "G3")
    assert any(RID["t2"] in p for p in out["G3"]), out["G3"]


def test_g3_join_is_exact_not_a_prefix_match():
    """Adversarial: a release line whose id merely starts with the relayed id does not join."""
    ev = good()
    ev["release_log"] = "\n".join(
        [release_line(RID["t1"]), release_line(RID["t2"] + "0"), release_line(RID["t3"])]
    )
    out = run(ev)
    assert any(RID["t2"] in p for p in out["G3"]), out["G3"]


# --------------------------------------------------------------------------- G4


def test_g4_a_request_first_seen_while_idle_fails_g4_and_names_it():
    """The idle captioner leaking: a request whose first relayed frame is in the IDLE window."""
    ev = good()
    log = Log()
    log.records = copy.deepcopy(ev["session_log"])
    # after t3 is IDLE: a new request appears
    log.canvas(CAPTION_RID, 0, 2)
    log.canvas(CAPTION_RID, 0, 4)
    ev = replace_log(ev, log)
    ev["release_log"] += "\n" + release_line(CAPTION_RID)  # so only G4 is about it
    out = run(ev)
    only(out, "G4")
    assert any(CAPTION_RID in p for p in out["G4"]), out["G4"]


def test_g4_a_request_first_seen_in_idle_stays_refused_when_a_turn_opens():
    """Adversarial: the caption request's later frames land in LISTENING; first-seen decides."""
    ev = good()
    log = Log()
    log.state("IDLE", "t0")
    log.canvas(CAPTION_RID, 0, 2)
    for r in copy.deepcopy(ev["session_log"][1:]):
        log.records.append(r)
        if (
            r["data"].get("msg", {}).get("state") == "LISTENING"
            and r["data"]["msg"]["turn_id"] == "t1"
        ):
            log.canvas(CAPTION_RID, 0, 5)  # the captioner is still going while t1 opens
    for i, r in enumerate(log.records):
        r["seq"] = i
    ev = replace_log(ev, log)
    ev["release_log"] += "\n" + release_line(CAPTION_RID)
    out = run(ev)
    assert any(CAPTION_RID in p for p in out["G4"]), out["G4"]


def test_g4_a_thinking_request_whose_frames_finish_after_idle_passes():
    """Too-wide guard row (A4): first frame in THINKING, the final commit after IDLE."""
    log = Log()
    ev = good()
    for r in ev["session_log"][1:]:
        msg = r["data"].get("msg", {}) if r["kind"] == "wire.out" else {}
        is_t1_commit = msg.get("type") == "astra.canvas.step" and (
            msg.get("request_id") == RID["t1"] and msg.get("phase") == "commit"
        )
        if not is_t1_commit:
            log.records.append(copy.deepcopy(r))
        if msg.get("type") == "astra.turn.state" and msg == {
            "type": "astra.turn.state",
            "state": "IDLE",
            "turn_id": "t1",
            "thinking_ms": None,
            "replay": False,
        }:
            log.canvas(RID["t1"], 0, 20, phase="commit")  # 50 ms after IDLE, still t1's
    for i, r in enumerate(log.records):
        r["seq"] = i
    ev = replace_log(ev, log)
    relayed = fn("relayed_frames")(ev["session_log"])
    assert {"request_id": RID["t1"], "state": "IDLE", "turn_id": "t1"} in relayed, relayed
    only(run(ev))


# --------------------------------------------------------------------------- G5


def test_g5_a_non_loopback_conductor_socket_fails_g5_and_quotes_it():
    ev = good()
    ev["ss_lines"].append(ss("192.168.1.20:40000", "140.82.112.3:443", fd=30))
    out = run(ev)
    only(out, "G5")
    assert any("140.82.112.3" in p for p in out["G5"]), out["G5"]


def test_g5_a_non_loopback_local_address_fails_too():
    ev = good()
    ev["ss_lines"].append(ss("0.0.0.0:8080", "127.0.0.1:40500", fd=31))
    out = run(ev)
    only(out, "G5")
    assert any("0.0.0.0" in p for p in out["G5"]), out["G5"]


def test_g5_other_processes_and_ipv6_loopback_forms_are_not_findings():
    """Too-wide guard row: another pid (including one whose number starts with ours) is ignored;
    ``[::1]``, ``[::ffff:127.0.0.1]`` and 127.0.0.53 are loopback."""
    ev = good()
    ev["ss_lines"] += [
        ss("192.168.1.20:5555", "140.82.112.3:443", pid=999, fd=3),
        ss("192.168.1.20:5556", "140.82.112.3:443", pid=PID * 10, fd=4),  # pid=42420
        ss("127.0.0.1:41000", "127.0.0.53:53", fd=40),
    ]
    only(run(ev))


def test_g5_no_conductor_socket_at_all_is_a_problem_not_a_pass():
    ev = good()
    ev["ss_lines"] = [SS_HEADER, ss("192.168.1.20:5555", "140.82.112.3:443", pid=999)]
    only(run(ev), "G5")


# --------------------------------------------------------------------------- G6


def test_g6_a_turn_that_never_spoke_fails_g6():
    ev = good()
    log = Log()
    log.records = [
        r
        for r in copy.deepcopy(ev["session_log"])
        if not (
            r["kind"] == "wire.out"
            and r["data"]["msg"].get("type") == "astra.spoken"
            and r["data"]["msg"].get("turn_id") == "t3"
        )
    ]
    log.spoken("t3", "a replayed line is not this turn speaking", replay=True)
    out = run(replace_log(ev, log))
    only(out, "G6")
    assert any("t3" in p for p in out["G6"]), out["G6"]


def test_g6_any_astra_error_fails_g6_including_denoise_no_tap():
    ev = good()
    log = Log()
    log.records = copy.deepcopy(ev["session_log"])
    log.error("denoise_no_tap", "no tap frame arrived during this session")
    out = run(replace_log(ev, log))
    only(out, "G6")
    assert any("denoise_no_tap" in p for p in out["G6"]), out["G6"]


# --------------------------------------------------------------------------- O-C5 (2): G6 and teardown
#
# Ruling O-C5 (2), on the second /launch at ffba98c: the only ``astra.error`` was ``degraded:
# text_only``, published after turn 3 closed because the harness's own headless page disconnected
# (``machine.step`` ``client.disconnected`` -> ``degrade text_only``; the error's ``wire.out`` row
# has ``connections: 0``). G6 ignores an ``astra.error`` published after the FIRST
# ``client.disconnected`` machine step that follows the LAST turn's ``turn.record``; every other
# ``astra.error`` still fails G6. The real order at ffba98c (seqs 620-628): machine.step
# playback.drained -> wire.out astra.turn.state IDLE -> turn.record t3 -> machine.step tick ->
# machine.step client.disconnected -> wire.out astra.error degraded/text_only.
#
# Reading (stated, not guessed silently): "the last turn" is the last of ``evidence["turns"]``,
# whose ``turn.record`` is also the last ``turn.record`` in the log in every row below, so either
# reading passes them. The "reported, not gated" listing of ignored errors is not pinned here: the
# reported block lives in the driver (``scripts/gpu/t0128c_denoise_smoke.py`` ``reported()``),
# which O-C5 does not give the implementer, and O-C5 names no key.

Hook = dict[str, list[tuple[str, Any]]]


def recorded(ev: dict[str, Any], *, before: Hook | None = None, after: Hook | None = None) -> Log:
    """``ev``'s session log with a ``turn.record`` after each turn's IDLE ``wire.out`` (the real
    order), and extra rows spliced in: ``before[tid]`` just before that turn's ``turn.record``,
    ``after[tid]`` just after it. A row is ``("error", None)``, ``("disconnected", None)`` or
    ``("tick", None)``. Records are renumbered in file order."""
    before = before or {}
    after = after or {}
    log = Log()
    log.records = []

    def splice(rows: list[tuple[str, Any]], tid: str) -> None:
        for what, _ in rows:
            if what == "error":
                log.error("degraded", "text_only", connections=0)
            elif what == "disconnected":
                log.disconnected(tid)
            elif what == "tick":
                log.machine_step("tick", tid)
            else:
                raise AssertionError(what)

    for r in copy.deepcopy(ev["session_log"]):
        log.records.append(r)
        msg = r["data"].get("msg", {}) if r["kind"] == "wire.out" else {}
        if msg.get("type") == "astra.turn.state" and msg.get("state") == "IDLE":
            tid = msg["turn_id"]
            splice(before.get(tid, []), tid)
            log.turn_record(tid)
            splice(after.get(tid, []), tid)
    for i, r in enumerate(log.records):
        r["seq"], r["t_ms"] = i, i * 10
    return log


TEARDOWN = [("tick", None), ("disconnected", None), ("error", None)]


def test_o_c5_premise_turn_records_and_machine_steps_change_no_gate():
    """The spliced rows alone (records, a tick, a disconnect, no error) are a healthy run."""
    ev = good()
    out = run(replace_log(ev, recorded(ev, after={"t3": [("tick", None), ("disconnected", None)]})))
    only(out)


def test_o_c5_g6_ignores_the_harness_teardown_error():
    """The ffba98c shape: t3's turn.record, a tick, the page's client.disconnected, then
    ``astra.error degraded: text_only`` (connections 0). Before O-C5 this failed G6 with
    "astra.error degraded: text_only" (gates.json at ffba98c)."""
    ev = good()
    out = run(replace_log(ev, recorded(ev, after={"t3": TEARDOWN})))
    only(out)


def test_o_c5_g6_the_teardown_exemption_hides_no_other_g6_finding():
    """Too-wide side: the teardown exemption covers that error only; a turn that never spoke
    still fails G6."""
    ev = good()
    log = recorded(ev, after={"t3": TEARDOWN})
    log.records = [
        r
        for r in log.records
        if not (
            r["kind"] == "wire.out"
            and r["data"]["msg"].get("type") == "astra.spoken"
            and r["data"]["msg"].get("turn_id") == "t2"
        )
    ]
    out = run(replace_log(ev, log))
    only(out, "G6")
    assert any("t2" in p for p in out["G6"]), out["G6"]
    assert not any("degraded" in p for p in out["G6"]), out["G6"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # a page disconnect + degrade between turns 2 and 3 (after t2's record, before t3's):
        # "after any disconnect" or "after any turn.record" would wrongly ignore it
        ({}, {"t2": TEARDOWN}),
        # in turn 3 after its IDLE, before its turn.record, behind a disconnect
        ({"t3": [("disconnected", None), ("error", None)]}, {}),
        # turn 1, before its own record
        ({"t1": [("disconnected", None), ("error", None)]}, {"t3": [("disconnected", None)]}),
    ],
    ids=["after-t2-record", "t3-before-its-record", "t1-before-its-record"],
)
def test_o_c5_g6_the_same_error_before_the_last_turn_record_still_fails(before, after):
    ev = good()
    out = run(replace_log(ev, recorded(ev, before=before, after=after)))
    only(out, "G6")
    assert any("degraded" in p and "text_only" in p for p in out["G6"]), out["G6"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # no disconnect anywhere
        ({}, {"t3": [("tick", None), ("error", None)]}),
        # the disconnect comes after the error
        ({}, {"t3": [("error", None), ("disconnected", None)]}),
        # the only disconnect precedes the last turn.record (mid-run), so it is not the one that
        # follows it: "the first client.disconnected in the log" would wrongly ignore the error
        ({"t3": [("disconnected", None)]}, {"t3": [("error", None)]}),
    ],
    ids=["no-disconnect", "disconnect-after-the-error", "disconnect-only-before-the-record"],
)
def test_o_c5_g6_an_error_after_the_last_turn_record_but_before_any_disconnect_still_fails(
    before, after
):
    ev = good()
    out = run(replace_log(ev, recorded(ev, before=before, after=after)))
    only(out, "G6")
    assert any("degraded" in p and "text_only" in p for p in out["G6"]), out["G6"]


def test_o_c5_g6_without_the_last_turns_record_nothing_is_teardown():
    """No turn.record in the log at all (good() as it is), then a disconnect and the error: there
    is no "after the last turn's turn.record", so the error still fails G6."""
    ev = good()
    log = Log()
    log.records = copy.deepcopy(ev["session_log"])
    log.disconnected("t3")
    log.error("degraded", "text_only", connections=0)
    out = run(replace_log(ev, log))
    only(out, "G6")
    assert any("degraded" in p for p in out["G6"]), out["G6"]


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ({}, {"t3": TEARDOWN}, ["astra.error degraded: text_only"]),
        # an error G6 gates (between turns 2 and 3) is not listed beside the teardown one
        ({}, {"t2": TEARDOWN, "t3": TEARDOWN}, ["astra.error degraded: text_only"]),
        ({}, {"t3": [("error", None), ("disconnected", None)]}, []),
        ({}, {"t3": [("tick", None), ("error", None)]}, []),
        ({"t3": [("disconnected", None)]}, {"t3": [("error", None)]}, []),
        ({}, {"t2": TEARDOWN}, []),
        ({"t3": [("disconnected", None), ("error", None)]}, {}, []),
    ],
    ids=[
        "teardown",
        "teardown-plus-a-gated-earlier-error",
        "disconnect-after-the-error",
        "no-disconnect",
        "disconnect-only-before-the-record",
        "after-t2-record",
        "t3-before-its-record",
    ],
)
def test_o_c5_reported_lists_exactly_the_teardown_errors(before, after, expected):
    """O-C5 (2), key ruled by the orchestrator: ``reported(ev)["g6_teardown_errors"]`` lists each
    ``astra.error`` G6 ignored as the harness's teardown, as "astra.error <code>: <message>"; an
    error G6 still gates is never listed there (it is a G6 problem instead)."""
    ev = good()
    ev = replace_log(ev, recorded(ev, before=before, after=after))
    rep = harness().reported(ev)
    assert "g6_teardown_errors" in rep, f"reported() has no g6_teardown_errors key: {sorted(rep)}"
    assert rep["g6_teardown_errors"] == expected, rep["g6_teardown_errors"]


# --------------------------------------------------------------------------- G7


def test_g7_commit_overrun_fails_g7():
    ev = good()
    ev["release_log"] = "\n".join(
        [release_line(RID["t1"]), release_line(RID["t2"], overrun=1), release_line(RID["t3"])]
    )
    only(run(ev), "G7")


def test_g7_tap_errors_fail_g7():
    ev = good()
    ev["release_log"] = "\n".join(
        [release_line(RID["t1"], errors=2), release_line(RID["t2"]), release_line(RID["t3"])]
    )
    only(run(ev), "G7")


def test_g7_dropped_frames_are_reported_not_gated():
    """Too-wide guard row: the tap dropping frames at its HWM is not an error or an overrun."""
    ev = good()
    ev["release_log"] = "\n".join(
        [release_line(RID["t1"], dropped=7), release_line(RID["t2"]), release_line(RID["t3"])]
    )
    only(run(ev))


def test_g7_no_release_line_is_a_problem():
    ev = good()
    ev["release_log"] = "INFO nothing from the tap here"
    out = run(ev)
    assert out["G7"], out
    assert out["G3"], out  # and no relayed id can join


def test_g7_a_malformed_release_line_never_passes():
    ev = good()
    ev["release_log"] += "\nINFO [tap.py:563] canvas tap release: req_id=x submitted=1 published=1"
    try:
        out = run(ev)
    except ValueError:
        return
    assert out["G7"], f"a malformed release line was skipped and G7 passed: {out}"


# --------------------------------------------------------------------------- the driver's pure helpers
#
# ``scripts/gpu/t0128c_denoise_smoke.py`` is GPU-spending as a script; importing it by path runs
# only its module-level constants and imports (no subprocess, no socket), and these rows call
# only its pure functions. Expected values are written by hand from the Regime and the evidence
# format above.


def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0128c C2: {HARNESS.relative_to(REPO)} does not exist"
    name = "_t0128c_denoise_smoke_harness"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None, f"{HARNESS} is not importable by path"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_harness_preflight_covers_the_regime_ports_and_the_three_prompts():
    """Regime: ports 8001/8002/8080/8081 must be free; the three typed turns verbatim, in order."""
    h = harness()
    assert {8001, 8002, 8080, 8081} <= set(h.PORTS), h.PORTS
    assert tuple(h.PROMPTS) == (
        "What is 17 times 23?",
        "Explain in three sentences why the sky is blue.",
        "Give me a shopping list card for pancakes.",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("3u/2m/0r", 2), ("12u/0m/5r", 0), ("0u/51m/0r", 51), ("0", 0), ("", 0)],
)
def test_harness_malformed_of_reads_the_m_of_ignored_count(text, expected):
    """``#ignored-count`` is ``"<u>u/<m>m/<r>r"`` (hud/src/main.js); "12u" is not an m count."""
    assert harness().malformed_of(text) == expected


def test_harness_listening_ports_reads_every_address_form():
    ss_ltn = "\n".join(
        [
            "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process",
            "LISTEN 0      4096       127.0.0.1:8001       0.0.0.0:*",
            "LISTEN 0      128            [::1]:8080          [::]:*",
            "LISTEN 0      511                *:8081             *:*",
            "LISTEN 0      100          0.0.0.0:22         0.0.0.0:*",
        ]
    )
    assert harness().listening_ports(ss_ltn) == {8001, 8080, 8081, 22}


def test_harness_new_turn_id_skips_replays_idle_and_taken_ids():
    log = Log()
    log.state("THINKING", "t0", replay=True)  # a replay is not a new turn
    log.state("IDLE", "t9")  # IDLE opens nothing
    log.state("LISTENING", "t1")
    log.state("THINKING", "t1")
    log.state("LISTENING", "t2")
    h = harness()
    assert h.new_turn_id(log.records, []) == "t1"
    assert h.new_turn_id(log.records, ["t1"]) == "t2"
    assert h.new_turn_id(log.records, ["t1", "t2"]) is None


def test_harness_id_shape_is_chatcmpl_16hex_8hex():
    shape = harness().ID_SHAPE
    assert shape.match(RID["t1"])  # the 806da12 recording's real engine id
    for bad in (
        "chatcmpl-b8d75dac2b8f054f",  # the response id, no engine suffix
        "chatcmpl-b8d75dac2b8f054f-b19b91f",  # 7 hex
        "chatcmpl-B8D75DAC2B8F054F-B19B91F7",  # upper case
        "cmpl-b8d75dac2b8f054f-b19b91f7",
    ):
        assert not shape.match(bad), bad


def test_harness_reported_numbers_by_hand_on_the_healthy_run():
    """good(): each turn decodes 12 frames, its counter moving on the 6 samples 50..300 ms after
    the baseline, so 12 frames over 0.25 s = 48.0 /s; three release lines of published=40."""
    rep = harness().reported(good())
    assert [t["decoded"] for t in rep["per_turn"]] == [12, 12, 12]
    assert [t["page_frames_per_s"] for t in rep["per_turn"]] == [48.0, 48.0, 48.0]
    assert rep["page_decoded"] == 36
    assert rep["tap_published_for_relayed_ids"] == 120
    assert rep["relayed_rows"] == 21
    assert rep["relayed_request_ids"] == [RID["t1"], RID["t2"], RID["t3"]]
    assert all(rep["ids_match_chatcmpl_16hex_8hex"].values())


# --------------------------------------------------------------------------- O-C2 (group C review)


def test_o_c2_harness_ports_include_the_conductor_and_the_hud_page():
    """H7/H8: the conductor (8000) and the HUD page (1420) are pre-flighted too."""
    assert {8000, 1420} <= set(harness().PORTS), harness().PORTS


@pytest.mark.parametrize("keep", [0, 1], ids=["zero-turns", "one-of-three-turns"])
def test_o_c2_fewer_than_three_turns_is_a_problem(keep):
    """O-C2 (1): the smoke types three turns; evidence with fewer must not pass (before O-C2 every
    gate passed on ``turns == []``: no turn, no per-turn finding)."""
    ev = good()
    ev["turns"] = ev["turns"][:keep]
    out = run(ev)
    assert any(out[g] for g in GATES), f"{keep} turn(s) of 3 passed every gate: {out}"


def test_o_c2_a_turn_with_no_samples_is_a_g1_problem():
    """M6: a turn the driver recorded no page sample for is not a pass of G1."""
    ev = good()
    ev["turns"][1]["samples"] = []
    out = run(ev)
    assert any("t2" in p for p in out["G1"]), out["G1"]


def test_o_c2_a_commit_only_label_stream_fails_g2():
    """O-C2 (2): six commit labels across canvases 0-5 are six distinct strings but no denoise
    step was ever shown; G2 counts denoise-form step labels, so t2 fails it."""
    ev = good()
    for i, s in enumerate(ev["turns"][1]["samples"][1:]):
        s["label"] = f"canvas {i} · committed at step 20/48"
    out = run(ev)
    only(out, "G2")
    assert any("t2" in p for p in out["G2"]), out["G2"]


def test_o_c2_blank_spoken_text_does_not_count_for_g6():
    """M24: an ``astra.spoken`` of whitespace is not the turn speaking."""
    ev = good()
    log = Log()
    for r in copy.deepcopy(ev["session_log"]):
        msg = r["data"].get("msg", {}) if r["kind"] == "wire.out" else {}
        if msg.get("type") == "astra.spoken" and msg.get("turn_id") == "t3":
            msg["text"] = "  \n "
        log.records.append(r)
    out = run(replace_log(ev, log))
    only(out, "G6")
    assert any("t3" in p for p in out["G6"]), out["G6"]


def test_o_c2_a_request_first_seen_in_listening_passes_g4():
    """M32, too-wide side: only IDLE-first requests are refused; LISTENING is a turn being open."""
    ev = good()
    log = Log()
    log.records = []
    for r in copy.deepcopy(ev["session_log"]):
        log.records.append(r)
        msg = r["data"].get("msg", {}) if r["kind"] == "wire.out" else {}
        if msg.get("type") == "astra.turn.state" and (msg["state"], msg["turn_id"]) == (
            "LISTENING",
            "t2",
        ):
            log.canvas(CAPTION_RID, 0, 1)  # first frame while t2 is LISTENING
    for i, r in enumerate(log.records):
        r["seq"] = i
    ev = replace_log(ev, log)
    ev["release_log"] += "\n" + release_line(CAPTION_RID)
    relayed = fn("relayed_frames")(ev["session_log"])
    assert {"request_id": CAPTION_RID, "state": "LISTENING", "turn_id": "t2"} in relayed
    only(run(ev))


def test_o_c2_an_unparseable_conductor_ss_line_is_a_g5_problem():
    """M33: a conductor line whose addresses cannot be read is not silently loopback."""
    ev = good()
    ev["ss_lines"].append(f'ESTAB 0 0 garbled users:(("python3",pid={PID},fd=9))')
    out = run(ev)
    only(out, "G5")


# --------------------------------------------------------------------------- O-C3: the contract step
#
# ``contract_problem(returncode, junit_path) -> str | None`` judges the ``pytest -m gpu
# tests/contract/test_canvas_relay_wire.py --junitxml=…`` step. O-C3: it passes iff the three
# engine rows' ``[real]`` arms ran and passed, by name, and no row failed or errored; the three
# designed skips are allowed; any skip of an engine row ("no engine on …" included) fails. The
# junit documents below are hand-written in pytest's own junit shape, with the module's real names.

WIRE_CLASS = "tests.contract.test_canvas_relay_wire"
ENGINE_ROWS = (
    "test_tap_frames_reach_the_relay_and_are_published_unchanged",
    "test_the_relay_owns_one_sub_that_connects_bounded_and_closed_before_term",
    "test_a_silent_endpoint_publishes_nothing_and_costs_almost_nothing",
)
#: the designed real-arm skips, with the reasons tests/contract/test_canvas_relay_wire.py gives
DESIGNED_SKIPS = {
    "test_the_relay_reconnects_when_the_tap_restarts": (
        "an engine restart is not driven by the /launch smoke; fake tap only (T-0128c)"
    ),
    "test_only_the_step_topic_is_subscribed_and_a_malformed_multipart_is_invalid": (
        "the engine's tap cannot be made to send other topics or malformed parts"
    ),
    "test_a_start_that_fails_at_connect_closes_its_socket_and_aclose_returns": (
        "a refused connect involves no engine"
    ),
}
NO_ENGINE = "no engine on http://127.0.0.1:8001 (ConnectError); real rows run under /launch"


def junit(cases: list[tuple[str, str, str]]) -> str:
    """``cases``: (name without ``[real]``, outcome ``pass|skip|fail|error``, message)."""
    body = []
    for name, outcome, msg in cases:
        head = f'<testcase classname="{WIRE_CLASS}" name="{name}[real]" time="0.412"'
        if outcome == "pass":
            body.append(head + " />")
        elif outcome == "skip":
            body.append(
                head + f'><skipped type="pytest.skip" message="{msg}">'
                f"tests/contract/test_canvas_relay_wire.py:133: {msg}</skipped></testcase>"
            )
        elif outcome == "fail":
            body.append(
                head + f'><failure message="{msg}">AssertionError: {msg}</failure></testcase>'
            )
        else:
            body.append(head + f'><error message="{msg}">{msg}</error></testcase>')
    n = {k: sum(o == k for _, o, _ in cases) for k in ("skip", "fail", "error")}
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="{n["error"]}" failures="{n["fail"]}" '
        f'skipped="{n["skip"]}" tests="{len(cases)}" time="3.1" '
        'timestamp="2026-10-26T19:00:00.000000+00:00" hostname="box">'
        + "".join(body)
        + "</testsuite></testsuites>"
    )


def healthy_cases() -> list[tuple[str, str, str]]:
    return [(n, "pass", "") for n in ENGINE_ROWS] + [
        (n, "skip", why) for n, why in DESIGNED_SKIPS.items()
    ]


def contract(tmp_path: Path, returncode: int, cases: list[tuple[str, str, str]]) -> str | None:
    path = tmp_path / "contract.xml"
    path.write_text(junit(cases), encoding="utf-8")
    return harness().contract_problem(returncode, path)


def test_o_c3_a_healthy_contract_step_with_the_designed_skips_passes(tmp_path):
    """(a) 6 testcases: the three engine rows passed, the three designed rows skipped -> pass."""
    assert contract(tmp_path, 0, healthy_cases()) is None


def test_o_c3_no_engine_skips_are_a_problem(tmp_path):
    """(b) the engine rows skipped with "no engine on ..." (exit 0: pytest passes skips)."""
    cases = [(n, "skip", NO_ENGINE) for n in ENGINE_ROWS] + [
        (n, "skip", why) for n, why in DESIGNED_SKIPS.items()
    ]
    assert contract(tmp_path, 0, cases)


def test_o_c3_an_engine_row_skipped_for_any_reason_is_a_problem(tmp_path):
    """Adversarial: a designed-sounding reason on an engine row is still a skip of that row."""
    cases = healthy_cases()
    cases[1] = (ENGINE_ROWS[1], "skip", "a refused connect involves no engine")
    assert contract(tmp_path, 0, cases)


@pytest.mark.parametrize("returncode", [1, 0], ids=["exit-1", "junit-only"])
@pytest.mark.parametrize("outcome", ["fail", "error"])
def test_o_c3_a_failed_or_errored_row_is_a_problem(tmp_path, returncode, outcome):
    """(c) one engine row failed (or errored); the ``junit-only`` arm checks the report is read,
    not just the exit code."""
    cases = healthy_cases()
    cases[0] = (ENGINE_ROWS[0], outcome, "no frame arrived within 30 s")
    assert contract(tmp_path, returncode, cases)


@pytest.mark.parametrize("returncode", [5, 0], ids=["exit-5", "junit-only"])
def test_o_c3_zero_testcases_is_a_problem(tmp_path, returncode):
    """(d) nothing collected (pytest exits 5)."""
    assert contract(tmp_path, returncode, [])


def test_o_c3_an_engine_row_absent_from_the_report_is_a_problem(tmp_path):
    """Adversarial: "ran and passed by name". Three rows all passed and nothing skipped, but the
    silent-endpoint row is not among them (a count of passes is not the three names)."""
    cases = [
        (ENGINE_ROWS[0], "pass", ""),
        (ENGINE_ROWS[1], "pass", ""),
        ("test_some_other_row", "pass", ""),
    ]
    assert contract(tmp_path, 0, cases)
