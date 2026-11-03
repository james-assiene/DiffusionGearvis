"""The three-way start decision, taken before anything is spawned (T-0049 criteria 2, 7, 8).

``astra up`` printed ``started`` and then ``ok`` for three services on 2026-09-16 while two of the
three starts had failed, because the start plan runs unconditionally and a health probe answers
for *whatever* is on the port. D1 replaces that with a decision taken **first**, from four facts:
is it healthy, what does this CLI's state record say it started, is that handle still live, and is
the service script still the one the record was written against.

    not healthy                                             -> START
    healthy + the record names the live instance + digest unchanged -> SKIP  (exit 0, loud)
    healthy + record missing / handle gone / digest changed  -> CONFLICT (nothing is started)

Every row here is about ``service_state.decide``, which is the **pure** middle of
*observe → decide → act → verify → record*: no clock, no socket, no subprocess (criterion 2). The
observations are built by hand in ``tests/unit/_service_rig.py``; the two digest rows use the real
``services/*.sh`` bytes and an independent ``hashlib`` oracle rather than anything the
implementation computes.

Regime: fast suite, no GPU, no network, no subprocess, no clock. Nothing here starts, signals or
probes anything.
"""

from __future__ import annotations

import ast
import itertools
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from tests.unit import _service_rig as rig

REPO = Path(__file__).resolve().parents[2]
SERVICES = REPO / "services"

#: ``decide`` may not reach for any of these. Criterion 2 names the first three explicitly.
IO_TRAPDOORS = (
    (subprocess, "run"),
    (subprocess, "Popen"),
    (socket, "socket"),
    (time, "time"),
    (time, "monotonic"),
    (urllib.request, "urlopen"),
)

#: A pure decision module imports none of these. ``os`` is on the list because the environment is
#: an input to the *observation* (supervisor records it), never something ``decide`` may read.
BANNED_IMPORTS = (
    "subprocess",
    "socket",
    "time",
    "datetime",
    "urllib",
    "http",
    "httpx",
    "requests",
    "aiohttp",
    "websockets",
    "asyncio",
    "os",
)


def _trip_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every I/O and clock call in the process raise, for the duration of one row."""

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "T-0049 criterion 2: service_state.decide is pure — it reached for I/O or a clock"
        )

    for module, name in IO_TRAPDOORS:
        monkeypatch.setattr(module, name, explode)


def test_decide_is_pure_and_deterministic(monkeypatch):
    """Criterion 2: no I/O, no clock, no subprocess; and the same input always decides the same.

    Two guards, because either alone is weak: the trapdoors catch a decision that *asks* the
    world something, and the two structurally equal but distinct observations catch a decision
    that remembers (a cache keyed on identity would pass a plain "call it twice" check).
    """
    first = rig.observation("2", healthy=False)
    second = rig.observation("2", healthy=False)
    assert first is not second, "the two observations must be distinct objects for this to mean it"

    _trip_io(monkeypatch)
    one = rig.decide(first, "2")
    two = rig.decide(first, "2")
    three = rig.decide(second, "2")

    assert (one.action, one.reason) == (two.action, two.reason), (
        f"criterion 2: the same observation decides the same way twice; got {one} then {two}"
    )
    assert (one.action, one.reason) == (three.action, three.reason), (
        "criterion 2: the decision is a function of the observation's *value*, not of the object "
        f"it arrived in; got {one} for the first and {three} for an equal second"
    )


def test_the_pure_module_reaches_for_nothing_that_can_do_io():
    """Criterion 2, static: the import list is where purity is lost first."""
    service_state = rig.service_state("2")
    rig.attr(service_state, "decide", "2")  # the scan is vacuous over an empty stub
    source = Path(service_state.__file__).read_text()
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            offenders += [
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in BANNED_IMPORTS
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in BANNED_IMPORTS:
                offenders.append(f"line {node.lineno}: from {node.module} import …")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            offenders.append(f"line {node.lineno}: open(…)")
    assert not offenders, (
        "criterion 2: local_astra/service_state.py is the pure middle — observing and acting live "
        "in supervisor.py. It imports:\n  " + "\n  ".join(offenders)
    )


def test_an_unhealthy_service_is_a_start():
    """Criterion 2: the first branch of D1, and what keeps the other rows from being vacuous."""
    decision = rig.decide(
        rig.observation("2", healthy=False, health_reason="connection refused on 8080"),
        "2",
    )
    assert decision.action == "START", (
        f"criterion 2: nothing is answering, so this service is started; decide said {decision}"
    )


def test_healthy_with_matching_provenance_is_skip():
    """D1/criterion 2: healthy, ours, unchanged script -> skipped **out loud**, with the identity.

    The identity is what makes a skip honest: a row that says "already running" and names neither
    the handle nor its age is the same claim the broken ``up`` made.
    """
    record = rig.record("2", container_id="61f0d405d989", script_digest=rig.DIGEST_A)
    live = rig.identity("2", container_id="61f0d405d989", alive=True)
    decision = rig.decide(
        rig.observation("2", healthy=True, record=record, live=live, script_digest=rig.DIGEST_A),
        "2",
    )
    assert decision.action == "SKIP", (
        "D1: healthy, the state record names the live instance and the script digest is "
        f"unchanged, so this start is skipped, not run again; decide said {decision}"
    )
    assert "61f0d405d989" in decision.reason, (
        "D1: …and the row names the handle it checked (pid or container id), so the human can "
        f"tell *which* instance is being skipped; the reason was {decision.reason!r}"
    )


def test_healthy_with_a_changed_script_digest_is_conflict():
    """Criterion 7: T-0048 group B's case — the engine's script was edited since the record.

    The digests are opaque here on purpose (only their inequality decides); the row that uses the
    real ``services/vllm.sh`` bytes and a ``hashlib`` oracle is the next one. What this row pins is
    what the CONFLICT has to *say*, because a refusal whose remedy the human cannot read is the
    refusal that gets worked around with a manual ``docker stop``.
    """
    decision = rig.decide(
        rig.observation(
            "7",
            name="vllm",
            url="http://127.0.0.1:8001/v1",
            healthy=True,
            record=rig.record("7", name="vllm", script_digest=rig.DIGEST_A),
            live=rig.identity("7", alive=True),
            script_digest=rig.DIGEST_B,
        ),
        "7",
    )
    assert decision.action == "CONFLICT", (
        "criterion 7: the service is healthy but the script it was started from has changed, so "
        "this is a refusal — not a skip (which would measure the old engine under the new "
        f"configuration, T-0048's unattributable number) and not a silent restart. Got {decision}"
    )
    assert rig.DIGEST_A[:12] in decision.reason and rig.DIGEST_B[:12] in decision.reason, (
        "criterion 7: …naming both digests — the one the record was written against and the one "
        f"on disk now; the reason was {decision.reason!r}"
    )
    assert "--restart" in decision.reason and "vllm" in decision.reason, (
        f"criterion 7: …with `--restart vllm` as the remedy; the reason was {decision.reason!r}"
    )


def test_a_real_edit_to_a_copy_of_a_service_script_changes_the_decision(tmp_path):
    """Criterion 7, end to end over real bytes: edit the script, the skip refuses itself.

    Exactly T-0048 group B's move (it changes ``--kv-cache-memory-bytes``), performed on a copy so
    nothing in ``services/`` is touched. Hand-computed expectation: the same observation decides
    SKIP before the edit and CONFLICT after it, and nothing else about it changes.
    """
    copy = tmp_path / "vllm.sh"
    copy.write_bytes((SERVICES / "vllm.sh").read_bytes())
    before = rig.sha256_of(copy)

    copy.write_bytes(copy.read_bytes().replace(b"4294967296", b"2684354560"))
    after = rig.sha256_of(copy)
    assert before != after, (
        "the fixture edit must actually change the file, or this row tests nothing "
        "(T-0048 group B's edit is the KV pool in services/vllm.sh)"
    )

    record = rig.record("7", name="vllm", script_digest=before)
    live = rig.identity("7", alive=True)

    unedited = rig.decide(
        rig.observation(
            "7", name="vllm", healthy=True, record=record, live=live, script_digest=before
        ),
        "7",
    )
    assert unedited.action == "SKIP", (
        f"the control: before the edit this is an ordinary skip; got {unedited}"
    )

    drifted = rig.decide(
        rig.observation(
            "7", name="vllm", healthy=True, record=record, live=live, script_digest=after
        ),
        "7",
    )
    assert drifted.action == "CONFLICT", (
        "criterion 7: after the edit the running engine no longer matches the configuration on "
        f"disk, so `up` refuses instead of skipping or restarting; got {drifted}"
    )


def test_healthy_with_no_record_is_conflict_naming_what_answers_the_port():
    """Criterion 8: something is serving that this CLI cannot claim — a refusal, not a start."""
    url = "ws://127.0.0.1:8080/api/asr-streaming"
    decision = rig.decide(rig.observation("8", url=url, healthy=True, record=None, live=None), "8")
    assert decision.action == "CONFLICT", (
        "criterion 8: the port answers but no state record names the instance, so `up` must "
        f"neither start over it nor pretend it started it; got {decision}"
    )
    assert "8080" in decision.reason or url in decision.reason, (
        f"criterion 8: …naming what is answering the port; the reason was {decision.reason!r}"
    )
    assert "astra down" in decision.reason, (
        "criterion 8: …and saying plainly that `astra down` will not stop it (it stops only what "
        f"a record names); the reason was {decision.reason!r}"
    )


@pytest.mark.parametrize("handle", ["gone", "dead"])
def test_a_record_whose_handle_is_gone_while_the_port_answers_is_not_a_skip(handle):
    """Criterion 8, adversarial: **the shape the box was in** when the human ran the command.

    The record existed, its digest matched, and the port answered — but the recorded pids were
    dead and the containers answering were two hours older than the attempt. Today's ``up`` called
    that ``ok``; ``status`` called the same state "not running (stale pid file)". A decision that
    reads the record and the health flag but not the *liveness of the handle* reproduces exactly
    that lie, so this row exists to make it red.

    Both spellings of "gone" are covered: no live identity at all, and one that resolved to a
    handle that is no longer running.
    """
    record = rig.record("8", script_digest=rig.DIGEST_A)
    live = None if handle == "gone" else rig.identity("8", alive=False)
    decision = rig.decide(
        rig.observation("8", healthy=True, record=record, live=live, script_digest=rig.DIGEST_A),
        "8",
    )
    assert decision.action != "SKIP", (
        "criterion 8: the recorded handle is not running, so the record does not name whatever is "
        f"answering the port; calling that `already running` is the 2026-09-16 lie. Got {decision}"
    )
    assert decision.action == "CONFLICT", (
        "criterion 8: …and it is a refusal, not a START: starting a second instance over a live "
        f"one is how the docker name conflict happened in the first place. Got {decision}"
    )


def test_the_decision_is_total_and_every_action_is_reachable():
    """Criterion 2, the table's own guard: no observation crashes, and no branch is dead.

    Brute force over the cartesian product of the three facts that vary (healthy x record state x
    handle liveness = 12 observations). It is small enough to enumerate, so the oracle is the table
    in D1 rather than a re-implementation of ``decide``: the row asserts only totality,
    determinism and that all three actions occur — which is what stops the specific rows above
    from being satisfied by a constant function.
    """
    seen: dict[str, list[str]] = {action: [] for action in rig.ACTIONS}
    for healthy, record_state, handle_live in itertools.product(
        (True, False), (None, "matching", "drifted"), (True, False)
    ):
        record = None if record_state is None else rig.record("2", script_digest=rig.DIGEST_A)
        disk_digest = rig.DIGEST_B if record_state == "drifted" else rig.DIGEST_A
        live = rig.identity("2", alive=handle_live) if record is not None else None
        observation = rig.observation(
            "2",
            healthy=healthy,
            record=record,
            live=live,
            script_digest=disk_digest,
        )
        decision = rig.decide(observation, "2")
        again = rig.decide(observation, "2")
        assert decision.action == again.action, f"criterion 2: not deterministic on {observation}"
        seen[decision.action].append(f"healthy={healthy} record={record_state} live={handle_live}")

    counts = {action: len(cases) for action, cases in seen.items()}
    unreachable = sorted(action for action, cases in seen.items() if not cases)
    assert not unreachable, (
        f"criterion 2: D1 has three outcomes and every one of them must be reachable; "
        f"{unreachable} never happened over the 12 observations. Seen: {counts}"
    )
