"""A start is only a start when *this* invocation produced a new, healthy, nameable instance.

T-0049 criteria 1, 3, 4, 5, 6. On 2026-09-16 ``astra up --profile lean`` printed ``started`` and
``ok`` for all three services against a box that was already serving: the STT launcher had exited
125 with ``Conflict. The container name "/astra-stt" is already in use…`` in its log, the recorded
pids were dead, and the containers answering the probe were the originals from two hours earlier.
The probe was healthy the whole time, which is precisely why it proves nothing: **a health probe
answers for whatever is on the port**.

So the evidence a start really happened is ranked (D2) and none of it is health:

1. the launcher's own exit status — the CLI is its parent, and every ``services/*.sh`` ends in
   ``exec`` on a foreground server, so *any* exit inside the settle window is a failure;
2. liveness at the end of that window;
3. **newness** — the instance's own start time must be strictly after the attempt (D4);
4. the log, quoted in the FAILED row as the explanation, never as the detector.

Fixtures: the launcher's recorded output (``tests/fixtures/services/stt_conflict.log``) and the
T-0033-shape state file that is on this box's disk right now
(``tests/fixtures/services/pid_file_t0033.json``) — see that directory's ``MANIFEST.md``. The
container facts come from a fake ``docker`` on ``PATH`` that only ever answers ``inspect``, with a
response built from the *recorded* ``docker inspect`` of this box (``tests/fixtures/box``).

Regime: fast suite, no GPU, no network, nothing containerised actually started — every launcher
here ``exec``s a ``sleep`` and is reaped by the test that started it. The one measurement is
criterion 4's bound (a launcher that dies at ~40 ms is reported in < 250 ms wall, ``time.monotonic``,
n = 5, no seed, ``settle_s`` = 1.0): three orders of magnitude from the failure it excludes, not a
performance claim.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_astra.config import load
from tests.unit import _box_rig as box
from tests.unit import _service_rig as rig

REPO = Path(__file__).resolve().parents[2]
SERVICES = REPO / "services"
LEAN = load(None).profiles["lean"].services


def started_ago(seconds: float) -> str:
    """A ``.State.StartedAt`` that many seconds before *now*, in the daemon's RFC3339Nano spelling.

    Relative, never absolute (ruling A10/A14). These constants used to be fixed strings, one of
    them several hours in the **future** when it was written, which is the worst of both: it stops
    discriminating anything at an arbitrary hour, for a reason nobody would connect to the fixture.
    An anchor in the past only ever moves further into the past, so every row that wants a stale
    instance gets one at 03:00 as at 20:00, on any box, forever.
    """
    moment = datetime.now(UTC) - timedelta(seconds=seconds)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond * 1000:09d}Z"


def epoch_of(stamp: str) -> float:
    """The daemon's stamp as epoch seconds — the fixture reading itself, for the guards below.

    Only ever used to check that the *fixture* is in the shape a row needs; no row compares a
    timestamp this way on the implementation's behalf.
    """
    head, _, fraction = stamp.rstrip("Z").partition(".")
    moment = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return moment.timestamp() + (float(f"0.{fraction}") if fraction else 0.0)


#: The container that was really answering on 2026-09-16 (docs/tasks/T-0049.md Goal, logs/stt.log):
#: on the box it had been up two hours when the human ran ``astra up``, so that is where it sits
#: relative to every attempt made here — older than the attempt, which is the one fact that makes
#: it NOT-NEW under D4 as ruling A19 reads it.
OLD_ID = "61f0d405d9898f7e8098b9fd3f3b6837a4afa874711d7247f6e2a3ca5cc8df0c"
OLD_STARTED_AT = started_ago(2 * 60 * 60)
#: The instance a happy path produces. Its ``StartedAt`` is **not** a constant: the launcher writes
#: the moment it really ran and the fake daemon reports that (:data:`rig.SPAWN_STAMP`), because a
#: container this invocation started necessarily starts *after* the attempt reference, and nothing
#: written in a test file ahead of time can say that honestly (ruling A19(b)).
NEW_ID = "9c1be0f2a77341bd9f0e5c1a2b3d4e5f60718293a4b5c6d7e8f90123456789ab"
#: A start time for the rows about *reading* a container's identity, where newness is not at stake.
LIVE_STARTED_AT = started_ago(30)
#: How long the fake daemon takes over the pre-spawn read on the happy-path row. Not colour: it
#: puts a whole second between ``attempt_at`` and the instant the launcher stamps, so the row says
#: the same thing whether the implementation parses the daemon's nine fractional digits or
#: truncates to seconds. Sub-second margins would make this row depend on that choice, and a
#: fixture that is only sometimes true is worse than no fixture (A10's lesson, restated).
SLOW_PRE_SPAWN_READ_S = 1.2


class Probe:
    """The T-0033 probe seam, ``probe(name, url) -> None | reason``, with a fixed verdict."""

    def __init__(self, *, healthy: bool = True, reason: str = "connection refused") -> None:
        self.healthy = healthy
        self.reason = reason
        self.asked: list[tuple[str, str]] = []

    def __call__(self, name: str, url: str) -> str | None:
        self.asked.append((name, url))
        return None if self.healthy else self.reason


def start_time_of(pid: int) -> int:
    """``/proc/<pid>/stat`` field 22, read from the kernel — the oracle, not the module's copy."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rindex(")") + 1 :].split()[19])


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch, tmp_path):
    """No row here may read or write the checkout's ``logs/`` or its real run directory.

    Both defaults are module functions on ``supervisor``; ``spawn`` reads them at call time. A row
    that forgot to pass ``run_dir=`` would otherwise overwrite the state of the services running
    on this box — which is exactly the damage the 2026-09-16 run did.
    """
    supervisor = rig.supervisor("1")
    logs = tmp_path / "logs"
    runs = tmp_path / "run"
    logs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(supervisor, "log_dir", lambda: logs)
    monkeypatch.setattr(supervisor, "run_dir", lambda: runs)
    assert supervisor.run_dir() == runs and ".git" not in os.fspath(supervisor.run_dir()), (
        "the redirection has to bite: if supervisor.run_dir() were still the checkout's, every "
        "row here would overwrite the state of the services running on this box — which is the "
        "damage the 2026-09-16 run did, and it would be invisible in a green suite"
    )
    return logs, runs


@pytest.fixture
def dirs(isolate_state):
    return isolate_state


def reap_outcome(outcome: object) -> None:
    handle = getattr(outcome, "handle", None)
    pid = getattr(handle, "pid", None)
    if isinstance(pid, int):
        box.reap(None, pid)


# =================================================================================================
# Criterion 1 — the reproduction
# =================================================================================================


def test_a_launcher_that_exits_while_the_port_is_healthy_is_not_a_success(tmp_path, monkeypatch):
    """Criterion 1, adversarial: **the 2026-09-16 failure**, and it may not report success.

    The scenario is run twice, with the probe healthy and with the probe failing, and the two
    outcomes must agree: that is the only way to show the health result plays no part in deciding
    that a start happened. A pre-existing state record (the recorded T-0033-shape file) is in the
    run directory and must be byte-identical afterwards — the record was written on spawn in the
    broken version, which is what destroyed the provenance of the still-serving containers and
    left ``astra down`` unable to stop them.
    """
    results: dict[bool, object] = {}
    for probe_healthy in (True, False):
        with monkeypatch.context() as patch:
            base = tmp_path / f"healthy-{probe_healthy}"
            logs, runs = base / "logs", base / "run"
            logs.mkdir(parents=True)
            runs.mkdir(parents=True)
            (runs / "stt.pid").write_text(rig.t0033_pid_file_bytes())
            before = rig.snapshot(runs)

            docker = rig.fake_docker(
                base,
                patch,
                before=rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT),
            )
            script = rig.failing_launcher(
                base / "services" / "stt.sh",
                code=125,
                log=rig.CONFLICT_LOG,
                container="astra-stt",
            )
            probe = Probe(healthy=probe_healthy)
            outcome = rig.start(
                "1",
                "stt",
                [script],
                LEAN.stt,
                probe=probe,
                log_dir=logs,
                run_dir=runs,
                settle_s=0.5,
                budget_s=1.0,
            )
            results[probe_healthy] = outcome

            logged = (logs / "stt.log").read_text()
            assert rig.CONFLICT_QUOTE in logged, (
                "the fixture must reach the service log, or this row asserts nothing about "
                f"quoting it; logs/stt.log held:\n{logged}"
            )

            assert not rig.claims_success(outcome), (
                "criterion 1: the launcher exited 125 — no branch of this outcome may be "
                f"`ok`/`started`. {rig.describe(outcome)}"
            )
            assert str(outcome.status).upper() == "FAILED", (
                f"criterion 1: …it is FAILED. {rig.describe(outcome)}"
            )
            assert "125" in outcome.message, (
                "criterion 1: …and the row names the launcher's exit status, which is the one "
                f"piece of evidence a stale healthy service cannot fake. Row: {outcome.message!r}"
            )
            assert rig.CONFLICT_QUOTE in outcome.message, (
                "criterion 1: …and quotes the log line that explains it. Note the recorded log's "
                "last non-empty line is docker's `Run 'docker run --help'…` trailer, so a row "
                "built from `supervisor.log_tail` alone explains nothing — see the MANIFEST. "
                f"Row: {outcome.message!r}"
            )
            assert rig.snapshot(runs) == before, (
                "criterion 1: a failed start leaves the previous state record byte-identical — "
                "the record is written after verification, never on spawn (Implementation notes, "
                "the landmine). The run directory changed."
            )
            assert all(call[:1] == ("inspect",) or call[:1] == ("sg",) for call in docker.calls), (
                f"criterion 6, negative: nothing may be stopped, removed or renamed: {docker.calls}"
            )
            reap_outcome(outcome)

    healthy, unhealthy = results[True], results[False]
    assert str(healthy.status).upper() == str(unhealthy.status).upper(), (
        "criterion 1, the whole point: the health probe's answer must not appear anywhere in the "
        "decision that a start happened. With the probe healthy the outcome was "
        f"{rig.describe(healthy)}; with it failing, {rig.describe(unhealthy)}"
    )
    assert "125" in unhealthy.message, (
        "criterion 1: …and the exit status is the reason in both cases, not the probe's verdict"
    )


# =================================================================================================
# Criterion 3 — newness
# =================================================================================================


def test_a_healthy_instance_older_than_the_attempt_is_not_new(tmp_path, monkeypatch, dirs):
    """Criterion 3, adversarial: healthy before, healthy after, same instance — not a start.

    The fake daemon answers with the same container id and the same ``StartedAt`` before and after
    the launcher ran: the stale instance that today's probe certifies. A newness comparison that
    is deleted, or weakened from "strictly after the attempt" to "not before it", turns this row
    green again, which is the mutation the reviewer is asked to run (Blast radius, check 2).
    """
    logs, runs = dirs
    stale = rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=stale, after=stale)
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    before = rig.snapshot(runs)

    outcome = rig.start(
        "3",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=1.0,
    )
    try:
        assert not rig.claims_success(outcome), (
            "criterion 3: the instance answering the port started before this attempt, so nothing "
            f"was started here — NOT-NEW is a failure, never an `ok`. {rig.describe(outcome)}"
        )
        assert "NOT-NEW" in str(outcome.status).upper().replace("_", "-"), (
            f"criterion 3: …reported as NOT-NEW. {rig.describe(outcome)}"
        )
        assert OLD_ID[:12] in outcome.message or OLD_STARTED_AT[:10] in outcome.message, (
            "criterion 3: …naming the instance it found and when it started, because that is what "
            f"tells the human their `up` did nothing. Row: {outcome.message!r}"
        )
        assert rig.snapshot(runs) == before, (
            "criterion 3 + 5: a start that was not verified new writes no state record"
        )
    finally:
        reap_outcome(outcome)


def test_a_verified_start_writes_the_identity_record(tmp_path, monkeypatch, dirs):
    """Criteria 3 and 5, the control: a genuinely new instance is `started`, and it is recorded.

    Same rig as the NOT-NEW row with one fact changed — the daemon reports a different container
    id, and a ``StartedAt`` the **launcher itself stamps when it runs**, so the two rows together
    show the outcome is decided by newness and not by something constant.

    Ruling A19 is why the stamp is generated rather than written down: the primary arm is the
    instance's own start time against ``attempt_at``, and a container this invocation produced
    necessarily starts *after* that reference. A fixture that puts the new instance before the
    attempt (which the previous revision of this row did, to catch A10's expiring-fixture mistake)
    models something that cannot happen on a real happy path and would force the weaker design.
    The A10 mistake is caught instead by ``test_a_pre_spawn_read_that_failed_is_unknown_not_absent``,
    where a snapshot-only implementation has nothing to compare and stays red — a discriminator
    that does not expire and does not depend on the hour.
    """
    logs, runs = dirs
    docker = rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT),
        after=rig.container_json(container_id=NEW_ID, started_at=rig.SPAWN_STAMP),
        before_delay_s=SLOW_PRE_SPAWN_READ_S,
    )
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    attempt_after = time.time()
    outcome = rig.start(
        "5",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=2.0,
    )
    try:
        stamped = docker.spawn_started_at()
        assert epoch_of(stamped) - attempt_after >= 1.0, (
            "the fixture has to put the new instance a clear second after `attempt_at`, or this "
            "row would depend on how many fractional digits the implementation parses (see "
            f"SLOW_PRE_SPAWN_READ_S). The launcher stamped {stamped}, the attempt began no "
            f"earlier than {attempt_after}"
        )
        assert getattr(outcome, "ok", False) is True, (
            "criteria 3-5: the launcher is alive, the service is healthy and the instance on the "
            f"port started after the attempt — that is a start. {rig.describe(outcome)}"
        )
        assert str(outcome.status).lower() == "started", (
            f"…reported as started. {rig.describe(outcome)}"
        )

        files = sorted(runs.glob("*"))
        assert len(files) == 1, f"criterion 5: the start leaves exactly one state file; got {files}"
        written = json.loads(files[0].read_text())
        expected_keys = {
            "version",
            "name",
            "kind",
            "pid",
            "start_time",
            "argv",
            "script_digest",
            "container_name",
            "container_id",
            "started_at",
        }
        missing = sorted(expected_keys - set(written))
        assert not missing, (
            f"criterion 5: the versioned record carries {sorted(expected_keys)}; missing {missing} "
            f"from {written}"
        )
        assert written["name"] == "stt" and written["kind"] == "container", (
            f"criterion 5/6: this service is a container this CLI can name; got {written}"
        )
        assert written["pid"] == outcome.handle.pid, (
            f"criterion 5: the pid is the launcher this CLI started; got {written}"
        )
        assert written["start_time"] == start_time_of(outcome.handle.pid), (
            "criterion 5: …with /proc/<pid>/stat field 22 beside it, which is what makes a reused "
            f"pid detectable later (T-0033 ruling A11); got {written}"
        )
        assert tuple(written["argv"]) == (str(script),), f"got {written}"
        assert rig.sha256_of(script) in str(written["script_digest"]), (
            "criterion 5/7: …and the sha256 of the service script's bytes, which is what makes "
            f"configuration drift detectable on the next `up`; got {written}"
        )
        assert written["container_name"] == "astra-stt", f"got {written}"
        assert NEW_ID.startswith(str(written["container_id"])), (
            "criterion 5/6: …and the container id resolved immediately after the start — the new "
            f"one, not the one that was there before. Got {written}"
        )
        assert docker.inspect_calls(), "the container id has to come from `docker inspect`"
    finally:
        reap_outcome(outcome)


#: How many times the row below may re-run its attempt to get the fixture it needs. The attempt
#: reference and the launcher's own stamp have to land in the *same* wall-clock second; starting
#: just after a second boundary leaves ~0.95 s of slack for one fake `docker inspect` plus a fork,
#: so crossing into the next second means the box stalled for a second. That is rare and it is not
#: a fact about the code, so the row re-rolls rather than failing — and never weakens what it
#: asserts. Exhausting the tries is a fixture failure that says so.
SAME_SECOND_TRIES = 3


def at_the_top_of_a_second() -> float:
    """Block until just after a second boundary and return that instant (see SAME_SECOND_TRIES)."""
    time.sleep(1.0 - (time.time() % 1.0) + 0.002)
    return time.time()


def whole_second_of(stamp: str) -> int:
    """The daemon stamp with its fraction thrown away — what a seconds-resolution parse keeps."""
    return int(epoch_of(stamp))


def test_a_start_in_the_same_second_as_the_attempt_is_still_a_start(tmp_path, monkeypatch, dirs):
    """Criterion 3, adversarial (ruling A22): a *fast* start is a start, and precision decides it.

    The happy-path row above deliberately puts a clear second between ``attempt_at`` and the
    instant the launcher stamps, so that it says the same thing at any parsing precision. That is
    the right shape for a behaviour row and the wrong shape for pinning the parse: with the delay
    removed, this is what really happens on the box — the container appears a few tens of
    milliseconds after the reference, *inside the same second*, later within it.

    Microseconds: the stamp is strictly after the reference, so the start is reported. Seconds
    (``fraction = "000000"`` in :func:`supervisor.daemon_time`): both collapse onto the same
    whole second, "strictly after" is false, and a start that really happened is reported NOT-NEW
    — a false failure whose trigger is the start being quick, so it appears under load and
    disappears whenever anyone goes looking. Nothing else here differs from the row above; the
    fixture guard below is what makes the difference load-bearing rather than incidental.

    What this row does **not** pin is ``>`` against ``>=`` (ruling A23): equality would need two
    stamps identical to the microsecond, and no rig can produce that.
    """
    logs, runs = dirs
    outcome = None
    stamped = ""
    attempt_floor = 0.0
    try:
        for attempt in range(SAME_SECOND_TRIES):
            sandbox = tmp_path / f"try{attempt}"
            docker = rig.fake_docker(
                sandbox,
                monkeypatch,
                before=rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT),
                after=rig.container_json(container_id=NEW_ID, started_at=rig.SPAWN_STAMP),
            )
            script = rig.sleeper_launcher(
                sandbox / "services" / "stt.sh", container="astra-stt", switch=docker.switch
            )
            attempt_floor = at_the_top_of_a_second()
            outcome = rig.start(
                "3",
                "stt",
                [script],
                LEAN.stt,
                probe=Probe(healthy=True),
                log_dir=logs,
                run_dir=runs,
                settle_s=0.3,
                budget_s=2.0,
            )
            stamped = docker.spawn_started_at()
            # `attempt_at` is taken inside start_service, after this read and before the spawn,
            # so it lies between `attempt_floor` and the launcher's stamp. When those two share a
            # whole second, so does the reference the implementation actually compares against —
            # which is the fixture this row needs and cannot observe directly.
            if whole_second_of(stamped) == int(attempt_floor):
                break
            reap_outcome(outcome)
            outcome = None

        assert outcome is not None, (
            f"fixture: {SAME_SECOND_TRIES} attempts in a row crossed a second boundary between "
            "the attempt reference and the launcher's stamp, so this row never got the same-second "
            "start it is about. That is a stalled box, not a verdict on the code — flag the timing "
            "rather than reading anything into it"
        )
        assert epoch_of(stamped) > attempt_floor, (
            "fixture: the launcher stamps after the attempt reference by construction (the "
            f"reference is taken before the spawn); got {stamped} against {attempt_floor}"
        )
        assert getattr(outcome, "ok", False) is True, (
            "criterion 3: the container this attempt produced started at "
            f"{stamped} — inside the same wall-clock second as the attempt reference "
            f"({attempt_floor}) and later within it. That is strictly after the reference, so it "
            "is a start. Reading the daemon's stamp only to the second collapses the two onto the "
            "same instant and reports a start that really happened as NOT-NEW, which is a false "
            f"failure caused by the start being fast. {rig.describe(outcome)}"
        )
        assert str(outcome.status).lower() == "started", (
            f"criterion 3: …reported as started. {rig.describe(outcome)}"
        )
    finally:
        reap_outcome(outcome)


@pytest.mark.parametrize("failure", ["permission-denied", "busy-daemon"])
def test_a_pre_spawn_read_that_failed_is_unknown_not_absent(tmp_path, monkeypatch, dirs, failure):
    """Criterion 3, adversarial (ruling A13): "the daemon could not be asked" is not "nothing was
    there", and a start whose newness cannot be established is not a start.

    The two facts arrive in the same shape — no container came back — and they mean opposite
    things. *Absent* is evidence the thing on the port afterwards must be ours; *unknown* is no
    evidence at all. The pre-spawn read fails here in the two ways it fails on this box: permission
    denied on the socket with no working group fallback (the module's own comment calls that the
    normal state here, and the ``sg`` retry fails with it), and a daemon that does not answer
    within ``DOCKER_TIMEOUT_S``. Afterwards the probe is healthy, the launcher is alive, and the
    container answering to the script's ``--name`` is the one from **two hours before this
    attempt** — 2026-09-16's box, exactly.

    Two things are asserted and the second is the worse one. The outcome may not be a success; and
    no state record may be written naming that foreign container, because group B's ``status`` and
    ``--restart`` act on the record, so a start that adopts a stranger's container id does not
    merely misreport once — it hands ``astra down`` and ``--restart`` a handle to somebody else's
    instance. That is the provenance corruption this whole task exists to prevent (D4, ruling A13).

    Under ruling A19 this is also where a snapshot-only implementation dies, and it is the only
    place it can: on the happy path the snapshot design and the clock design agree, but here there
    is no snapshot to compare against, while the clock arm — the primary one, per D4 — settles it
    on its own, because a container from two hours ago precedes the attempt whatever else is or is
    not known. That discriminator does not expire.

    Not satisfiable by refusing every containerised start: the control row
    ``test_a_verified_start_writes_the_identity_record`` runs the same rig with a pre-spawn read
    that *succeeded* and a different container afterwards, and requires ``started`` there. Nor by
    refusing whenever the daemon is unhappy: the mirror row below requires a start whose *post*-
    spawn read fails to stay a start.
    """
    logs, runs = dirs
    assert started_ago(0) > OLD_STARTED_AT, (
        "the container that answers after the spawn has to predate the attempt for this row to "
        f"mean anything; it claims {OLD_STARTED_AT}"
    )
    stale = rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT)
    if failure == "busy-daemon":
        monkeypatch.setattr(rig.supervisor("3"), "DOCKER_TIMEOUT_S", 0.3)
        docker = rig.fake_docker(
            tmp_path,
            monkeypatch,
            before=None,
            after=stale,
            before_error="the daemon did not answer",
            before_delay_s=2.0,
        )
    else:
        docker = rig.fake_docker(
            tmp_path,
            monkeypatch,
            before=None,
            after=stale,
            before_error=(
                "permission denied while trying to connect to the Docker daemon socket "
                "(and `sg docker -c` cannot help: this login session has no docker group)"
            ),
        )
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    snapshot_before = rig.snapshot(runs)

    outcome = rig.start(
        "3",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=2.0,
    )
    try:
        assert not rig.claims_success(outcome), (
            "criterion 3 / ruling A13: the reference point could not be taken, so nothing about "
            "the container answering now can be called new — and it is in fact the instance that "
            f"was already there, up since {OLD_STARTED_AT}. A failed `docker inspect` must read "
            f"as *unknown*, never as *no container there*. {rig.describe(outcome)}"
        )
        assert str(outcome.status).upper().replace("_", "-") in {"FAILED", "NOT-NEW"}, (
            "criterion 3: …reported as one of the two failure statuses, so `up` exits non-zero "
            f"and the human is told their start produced nothing. {rig.describe(outcome)}"
        )
        message = str(outcome.message)
        assert "astra-stt" in message or OLD_ID[:12] in message, (
            "criterion 3: …naming the container it could not establish anything about, because "
            f"that is what the human has to go and look at. Row: {message!r}"
        )

        state = {path.name: path.read_text() for path in sorted(runs.rglob("*")) if path.is_file()}
        adopted = {name: text for name, text in state.items() if OLD_ID[:12] in text}
        assert not adopted, (
            "criterion 3 + 5, the worse half: a record naming a container this invocation did not "
            "start is a handle to a stranger's instance, and group B's `status` and `--restart` "
            f"act on it — that is how `astra down` lost the live containers on 2026-09-16. {adopted}"
        )
        assert rig.snapshot(runs) == snapshot_before, (
            "criterion 5: the record is written only after the start is verified, and this one "
            f"never was. The run directory now holds {sorted(state)}"
        )
        assert all(call[:1] in (("inspect",), ("sg",)) for call in docker.calls), (
            f"criterion 6, negative: nothing may be stopped, removed or renamed: {docker.calls}"
        )
    finally:
        reap_outcome(outcome)


def test_a_post_spawn_read_that_failed_does_not_turn_a_real_start_into_a_failure(
    tmp_path, monkeypatch, dirs
):
    """Criterion 3/6, the mirror of ruling A13: refusing to lie is not the same as refusing.

    The start really happened — this process's own child is alive at the end of the settle window,
    it was spawned after the attempt reference, and the service is healthy — and then the daemon
    cannot be asked which container it is. Nothing about *that* makes the start a failure: it makes
    the container's identity unknown, which is the situation criterion 6 already has an answer for
    (the engine, whose script passes no ``--name``): degrade to the launcher pid, say so in the row,
    do not guess. Newness still holds on the evidence that remains, because the launcher is this
    invocation's child and started after the reference by construction (ruling A6).

    Without this row the fix for A13 has an obvious over-correction — treat every unhappy daemon as
    a failed start — and that is its own kind of lying: it would report a service that is up and
    serving as not started, on a box where `permission denied` on the docker socket is the normal
    state. The negative control sits right below: a daemon that *answers* and says the container is
    not running is still FAILED, so this row cannot be satisfied by swallowing every docker problem.
    """
    logs, runs = dirs
    docker = rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=None,
        after=rig.container_json(container_id=NEW_ID, started_at=rig.SPAWN_STAMP),
        after_error=(
            "permission denied while trying to connect to the Docker daemon socket "
            "(and `sg docker -c` cannot help: this login session has no docker group)"
        ),
    )
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    outcome = rig.start(
        "3",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=2.0,
    )
    try:
        assert getattr(outcome, "ok", False) is True, (
            "criterion 3, the mirror of A13: the launcher this invocation spawned is alive and the "
            "service is healthy — the start happened. That the daemon could not then be asked "
            "which container it is makes the *identity* unknown, not the start failed. "
            f"{rig.describe(outcome)}"
        )
        assert str(outcome.status).lower() == "started", (
            f"…reported as started, not FAILED and not NOT-NEW. {rig.describe(outcome)}"
        )
        message = str(outcome.message)
        assert "astra-stt" in message, (
            f"criterion 6: …naming the container it was asking about. Row: {message!r}"
        )
        assert any(word in message.lower() for word in ("unknown", "could not", "cannot")), (
            "criterion 6: …and saying plainly that this one could not be established, because a "
            "start whose handle is unknown is one `astra down` will not be able to stop by id. "
            f"The limitation belongs in the row rather than papered over. Row: {message!r}"
        )

        files = sorted(runs.glob("*"))
        assert len(files) == 1, f"criterion 5: a verified start records itself; got {files}"
        written = json.loads(files[0].read_text())
        assert written["pid"] == outcome.handle.pid, (
            f"criterion 5: …as the launcher this CLI started; got {written}"
        )
        assert not written.get("container_id"), (
            "criterion 6: …with no container id, because none could be read. Recording the one "
            f"from the pre-spawn snapshot, or a guess, is how provenance gets corrupted; {written}"
        )
        assert all(call[:1] in (("inspect",), ("sg",)) for call in docker.calls), (
            f"criterion 6, negative: nothing may be stopped, removed or renamed: {docker.calls}"
        )
    finally:
        reap_outcome(outcome)


def test_a_container_the_daemon_says_is_not_running_is_still_a_failure(tmp_path, monkeypatch, dirs):
    """Criterion 3/6, the negative control for the row above: an *answer* is not a non-answer.

    Here the daemon is reachable and reports the container the script names as exited. That is
    evidence, and it says the thing this attempt started is already dead — a FAILED start, whatever
    the port says (the port is exactly what cannot be trusted). The row exists so that the fix for
    the mirror case cannot be "treat anything docker says awkwardly as fine": the two rows differ in
    one fact, whether the daemon answered.
    """
    logs, runs = dirs
    stopped = rig.container_json(container_id=NEW_ID, started_at=rig.SPAWN_STAMP, running=False)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=None, after=stopped)
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    before_state = rig.snapshot(runs)
    outcome = rig.start(
        "3",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=2.0,
    )
    try:
        assert not rig.claims_success(outcome), (
            "criterion 3: the daemon answered, and what it said is that the container this start "
            f"produced is not running — the port answering proves nothing. {rig.describe(outcome)}"
        )
        assert rig.snapshot(runs) == before_state, (
            "criterion 5: …and an unverified start records nothing"
        )
    finally:
        reap_outcome(outcome)


def test_the_new_record_is_a_pid_file_that_down_can_still_stop(tmp_path, monkeypatch, dirs):
    """Criterion 5: "replaces the bare pid file" — ``stop`` must read what a start writes.

    Without this the fix would trade one lost-provenance bug for another: `up` would record the
    engine in a shape `astra down` cannot act on, which is the state this box is in today.
    """
    logs, runs = dirs
    supervisor = rig.supervisor("5")
    script = rig.sleeper_launcher(tmp_path / "services" / "tts.sh", containerised=False)
    outcome = rig.start(
        "5",
        "tts",
        [script],
        LEAN.tts,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=1.0,
    )
    try:
        assert getattr(outcome, "ok", False) is True, rig.describe(outcome)
        files = sorted(runs.glob("*"))
        assert len(files) == 1, f"criterion 5: one state file per started service; got {files}"
        assert rig.attr(supervisor, "pid_files", "5")(runs) == (files[0],), (
            "criterion 5: …and `pid_files` still finds it, so `down` and `status` see it"
        )
        pid = outcome.handle.pid
        result = rig.attr(supervisor, "stop", "5")(files[0])
        assert box.wait_until(lambda: not box.alive(pid), timeout_s=5), (
            f"criterion 5: `stop` must be able to stop what a verified start recorded: {result}"
        )
        assert not files[0].exists(), f"…and remove the file: {result}"
    finally:
        reap_outcome(outcome)


def test_a_failed_start_does_not_overwrite_the_previous_record(tmp_path, monkeypatch, dirs):
    """Criterion 5, the other failure branch: NOT-NEW must leave a *new-format* record alone too.

    Criterion 1 covers the FAILED branch over the T-0033-shape file that is on disk today; this
    one covers the record shape this task introduces, so neither branch can regress on its own.
    """
    logs, runs = dirs
    previous = {
        "version": 1,
        "name": "stt",
        "kind": "container",
        "pid": 4242,
        "start_time": 858746,
        "argv": [str(SERVICES / "stt.sh")],
        "script_digest": rig.DIGEST_A,
        "container_name": "astra-stt",
        "container_id": OLD_ID[:12],
        "started_at": 1_758_000_000.0,
    }
    (runs / "stt.pid").write_text(json.dumps(previous, indent=2) + "\n")
    before = rig.snapshot(runs)

    stale = rig.container_json(container_id=OLD_ID, started_at=OLD_STARTED_AT)
    docker = rig.fake_docker(tmp_path, monkeypatch, before=stale, after=stale)
    script = rig.sleeper_launcher(
        tmp_path / "services" / "stt.sh", container="astra-stt", switch=docker.switch
    )
    outcome = rig.start(
        "5",
        "stt",
        [script],
        LEAN.stt,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=1.0,
    )
    try:
        assert not rig.claims_success(outcome), rig.describe(outcome)
        assert rig.snapshot(runs) == before, (
            "criterion 5: the record is written only after the start is verified, so an "
            "unverified one leaves the previous record exactly as it was"
        )
    finally:
        reap_outcome(outcome)


# =================================================================================================
# Criterion 4 — the launcher's exit status
# =================================================================================================


def test_the_handle_exposes_the_launchers_return_code(tmp_path, dirs):
    """Criterion 4 / ruling A1: ``Handle`` keeps the child, so its exit status is observable.

    ``start_new_session=True`` detaches the controlling terminal, not parenthood — ``Popen.poll()``
    would have returned 125 for the STT launcher on 2026-09-16. Today ``Handle`` drops the
    ``Popen`` and keeps the pid, so ``cli.up``'s only failure check can never fire.
    """
    logs, runs = dirs
    supervisor = rig.supervisor("4")
    spawn = rig.attr(supervisor, "spawn", "4")

    dying = rig.failing_launcher(tmp_path / "services" / "stt.sh", code=7, log=rig.CONFLICT_LOG)
    handle = spawn("stt", [str(dying)], log_dir=logs, run_dir=runs)
    try:
        returncode = rig.attr(handle, "returncode", "4")
        assert box.wait_until(lambda: returncode() is not None, timeout_s=5), (
            "criterion 4: Handle.returncode() is the child's exit status once it has exited; it "
            "stayed None for a launcher that exited"
        )
        assert returncode() == 7, (
            f"criterion 4: …and it is the status the launcher exited with; got {returncode()}"
        )
    finally:
        box.reap(None, handle.pid)

    alive = rig.sleeper_launcher(tmp_path / "services" / "tts.sh", containerised=False)
    running = spawn("tts", [str(alive)], log_dir=logs, run_dir=runs)
    try:
        assert rig.attr(running, "returncode", "4")() is None, (
            "criterion 4, negative: a launcher that is still running has no exit status yet — "
            "reporting one would make every healthy start a FAILED row"
        )
    finally:
        box.reap(None, running.pid)


def test_a_launcher_still_alive_at_the_end_of_the_window_is_not_a_failure(tmp_path, dirs):
    """Criterion 4, negative: the happy path must survive the new check.

    A non-containerised service (the Pocket TTS shape: ``exec taskset …``) whose launcher stays up
    and whose probe is healthy is a start, and its instance start time is by construction after
    the attempt.
    """
    logs, runs = dirs
    script = rig.sleeper_launcher(tmp_path / "services" / "tts.sh", containerised=False)
    outcome = rig.start(
        "4",
        "tts",
        [script],
        LEAN.tts,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=1.0,
    )
    try:
        assert getattr(outcome, "ok", False) is True, (
            "criterion 4, negative: the launcher is alive at the end of the settle window and the "
            f"service is healthy — nothing here is a failure. {rig.describe(outcome)}"
        )
        assert box.alive(outcome.handle.pid), (
            "criterion 4: …and the service it started is still running (T-0033 criterion 10: the "
            "child is in its own session and outlives the command)"
        )
    finally:
        reap_outcome(outcome)


def test_a_launcher_that_dies_early_is_reported_inside_the_window(tmp_path, monkeypatch, dirs):
    """Criterion 4 + Regime: the settle wait is a poll loop, not a flat sleep.

    Regime: ``settle_s`` = 1.0, a launcher that exits at ~40 ms, n = 5 repeats, no seed, fast
    suite, probe stubbed healthy, **a fake docker on PATH** (see below). Threshold: reported in
    < 250 ms wall (``time.monotonic``) — six times the failure's own 40 ms and four times below
    the flat sleep it excludes. Not a performance claim and not comparable across boxes.

    The fake is not decoration (rulings A9/A14). The launcher names a container, so the pre-spawn
    reference read happens; without a fake on PATH it was the **live daemon on this box** that
    answered it, inside the one row in this module that reports a number. Idle that is ~8 ms, but
    the reviewer measured two of five iterations at 5.06 s — ``DOCKER_TIMEOUT_S`` — under load,
    which failed the row for a reason that has nothing to do with the settle loop. That is
    CLAUDE.md's unattributable number, and `.claude/rules/tests.md` keeps ``tests/unit`` free of
    I/O besides. The fake answers "no such object" in a bash fork, so what is being timed is the
    poll loop.
    """
    logs, _runs = dirs
    walls: list[float] = []
    for run in range(5):
        base = tmp_path / f"early-{run}"
        runs = base / "run"
        runs.mkdir(parents=True)
        docker = rig.fake_docker(base, monkeypatch, before=None)
        script = rig.failing_launcher(
            base / "services" / "stt.sh", code=125, log=rig.CONFLICT_LOG, delay_s=0.04
        )
        started = time.monotonic()
        outcome = rig.start(
            "4",
            "stt",
            [script],
            LEAN.stt,
            probe=Probe(healthy=True),
            log_dir=logs,
            run_dir=runs,
            settle_s=1.0,
            budget_s=5.0,
        )
        walls.append(time.monotonic() - started)
        assert not rig.claims_success(outcome), rig.describe(outcome)
        assert all(call[:1] in (("inspect",), ("sg",)) for call in docker.calls), (
            "the number below is only attributable if the daemon that answered was the fake one "
            f"on PATH (rulings A9/A14); docker saw {docker.calls}"
        )
        reap_outcome(outcome)
    assert max(walls) < 0.25, (
        "criterion 4 (Regime): a launcher that dies in 40 ms is reported in 40 ms — the settle "
        f"window is polled at ~50 ms granularity, never slept through. Walls (n=5): {walls}"
    )


# =================================================================================================
# Criterion 6 — the right authority answers for liveness
# =================================================================================================


def test_container_liveness_uses_docker_inspect_of_the_recorded_id(tmp_path, monkeypatch):
    """Criterion 6: the recorded **id**, through ``docker inspect``, with the ``sg docker -c`` retry.

    Not the image (that would claim containers this CLI never started) and never ``docker ps |
    grep``. The retry shape is pinned here because it is a deliberate duplicate of
    ``preflight._docker`` (Implementation notes): the two copies must not drift silently.
    """
    docker = rig.fake_docker(
        tmp_path,
        monkeypatch,
        before=rig.container_json(container_id=NEW_ID, started_at=LIVE_STARTED_AT),
        deny_first=True,
    )
    record = rig.record("6", kind="container", container_id=NEW_ID, container_name="astra-stt")
    identity = rig.live_identity(record)

    assert identity.alive is True, (
        f"criterion 6: the daemon says the container is running; got {identity!r}. "
        f"docker saw {docker.calls}"
    )
    assert str(identity.started_at) == LIVE_STARTED_AT, (
        f"criterion 6: …and its start time is the daemon's .State.StartedAt; got {identity!r}"
    )
    assert NEW_ID.startswith(str(identity.container_id)), (
        f"criterion 6: …for the id that was recorded; got {identity!r}"
    )

    inspects = docker.inspect_calls()
    assert inspects, (
        f"criterion 6: liveness is asked of `docker inspect`; calls were {docker.calls}"
    )
    assert all(call[-1] == NEW_ID for call in inspects), (
        "criterion 6: …of the **recorded id**, never the image and never a `docker ps | grep`; "
        f"the calls were {docker.calls}"
    )
    retry = [call for call in docker.calls if call[:1] == ("sg",)]
    assert retry == [("sg", "docker", "-c", f"docker inspect {NEW_ID}")], (
        "criterion 6: a permission-denied socket is retried as `sg docker -c '<the same "
        "command>'` — the fallback services/stt.sh documents and preflight._docker implements "
        f"(shlex.join of the argv). The calls were {docker.calls}"
    )


def test_a_recorded_container_that_is_gone_is_reported_not_alive(tmp_path, monkeypatch):
    """Criterion 6/8: an unresolvable handle is a fact about provenance, not a crash."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=None)
    record = rig.record("6", kind="container", container_id=OLD_ID, container_name="astra-stt")
    identity = rig.live_identity(record)
    assert identity.alive is False, (
        "criterion 6: the daemon has never heard of this id, so the handle this CLI recorded is "
        f"gone — that is criterion 8's CONFLICT input, not an exception. Got {identity!r}. "
        f"docker saw {docker.calls}"
    )
    assert identity.note, (
        f"criterion 6: …and the Identity carries why, for the row to print; got {identity!r}"
    )


def test_process_liveness_is_read_from_proc_and_asks_docker_nothing(tmp_path, monkeypatch):
    """Criterion 6: a host process is ``/proc``'s business — the authority per kind, not per taste."""
    docker = rig.fake_docker(tmp_path, monkeypatch, before=None)
    victim = subprocess.Popen(
        ["sleep", "20"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        record = rig.record(
            "6",
            kind="process",
            pid=victim.pid,
            start_time=start_time_of(victim.pid),
            container_name=None,
            container_id=None,
        )
        identity = rig.live_identity(record)
        assert identity.alive is True and identity.pid == victim.pid, (
            f"criterion 6: the recorded pid is running; got {identity!r}"
        )
        assert identity.start_time == start_time_of(victim.pid), (
            "criterion 6: …and its start time comes from /proc/<pid>/stat field 22, which is the "
            f"identity test a reused pid fails (T-0033 ruling A11); got {identity!r}"
        )
        assert docker.calls == (), (
            f"criterion 6: a host process is never asked about through docker; it ran {docker.calls}"
        )
    finally:
        box.reap(victim)


def test_an_unnamed_container_degrades_to_process_with_the_limitation_in_the_row(
    tmp_path, monkeypatch, dirs
):
    """Criterion 6: the engine today — ``services/vllm.sh`` passes no ``--name`` (ruling A1).

    Docker generated ``exciting_maxwell`` for it, so container-identity tracking cannot work until
    T-0048 adds a name. The honest behaviour is to degrade to ``kind=process``, say so in the row,
    and let newness rest on the launcher pid — never to guess the container by image ancestry,
    which would claim containers this CLI did not start.
    """
    logs, runs = dirs
    docker = rig.fake_docker(tmp_path, monkeypatch, before=None)
    script = rig.sleeper_launcher(
        tmp_path / "services" / "vllm.sh", container=None, containerised=True
    )
    outcome = rig.start(
        "6",
        "vllm",
        [script],
        LEAN.vllm,
        probe=Probe(healthy=True),
        log_dir=logs,
        run_dir=runs,
        settle_s=0.3,
        budget_s=2.0,
    )
    try:
        assert getattr(outcome, "ok", False) is True, (
            "criterion 6: an unnameable container is a limitation, not a failure — the engine "
            f"still starts. {rig.describe(outcome)}"
        )
        assert "container=unknown" in outcome.message, (
            "criterion 6: …with the limitation printed in the row, in those words. Row: "
            f"{outcome.message!r}"
        )
        assert "--name" in outcome.message and "vllm.sh" in outcome.message, (
            "criterion 6: …naming the script that passes no --name, so the reader knows what to "
            f"fix (T-0048). Row: {outcome.message!r}"
        )
        written = json.loads(sorted(runs.glob("*"))[0].read_text())
        assert written["kind"] == "process", (
            f"criterion 6: …and the record degrades to kind=process rather than guessing; {written}"
        )
        assert all(call[:1] in (("inspect",), ("sg",)) for call in docker.calls), (
            f"criterion 6, negative: nothing is stopped, removed or renamed: {docker.calls}"
        )
    finally:
        reap_outcome(outcome)


def test_the_container_name_is_read_from_the_script_so_t0048s_name_lands_for_free(tmp_path):
    """Criterion 6 / ambiguity 3: the name comes from the script, not from a table in the code.

    Expectations are read off the real ``services/*.sh``: ``stt.sh`` runs a named container,
    ``vllm.sh`` runs an unnamed one *today*, ``tts.sh`` is a host process. The synthetic script is
    T-0048's edit: the moment ``--name astra-vllm`` lands, this code must use it with no change.
    """
    supervisor = rig.supervisor("6")
    containerised = rig.attr(supervisor, "script_is_containerised", "6")
    container_of = rig.attr(supervisor, "script_container", "6")

    assert containerised(SERVICES / "stt.sh") is True, (
        "criterion 6: services/stt.sh's foreground server is a container (it `exec`s a container "
        "run at the end)"
    )
    assert container_of(SERVICES / "stt.sh") == "astra-stt", (
        "criterion 6: …named astra-stt, which is the name in the Conflict line of 2026-09-16"
    )
    assert containerised(SERVICES / "vllm.sh") is True, (
        "criterion 6: the engine is containerised too — that it has no name is a separate fact. "
        "Note its last line is `exec $DOCKER run …` with `DOCKER=docker` set above it (so that "
        "ASTRA_DRY_RUN=1 can swap in `printf`), so a detector matching only the literal spelling "
        "reports the one service whose container this CLI cannot name as not containerised at all"
    )
    assert container_of(SERVICES / "vllm.sh") in (None, "astra-vllm"), (
        "criterion 6 / ruling A1: services/vllm.sh passes no --name today (docker generated "
        "`exciting_maxwell`), so this is None until T-0048 group B adds `--name astra-vllm` — and "
        f"then it is that, with no change here. Got {container_of(SERVICES / 'vllm.sh')!r}"
    )
    assert containerised(SERVICES / "tts.sh") is False, (
        "criterion 6: Pocket TTS runs on the host (`exec taskset …`), so there is no container to "
        "ask about"
    )
    assert container_of(SERVICES / "tts.sh") is None, "…and no name to read"

    future = rig.sleeper_launcher(tmp_path / "vllm.sh", container="astra-vllm")
    assert container_of(future) == "astra-vllm", (
        "criterion 6: the name is read from the script's own `--name`, so T-0048's edit needs no "
        "change here (ambiguity 3's recommendation)"
    )


# =================================================================================================
# Criterion 5 — the files already on disk
# =================================================================================================


def test_t0033_shape_pid_files_are_still_read(tmp_path, dirs):
    """Criterion 5, negative: the three files in ``.git/local_astra/run`` keep working.

    They are the T-0033 shape — ``{"pid", "argv", "start_time"}`` — and the recorded fixture *is*
    one of them. The three T-0033/A11 behaviours are re-run against that shape: a stale pid kills
    nothing, a reused pid is never signalled, and a service that ``exec``d is still stopped.

    The new seam is looked up first on purpose: without that, this row would pass today (it is
    describing behaviour that already exists) and say nothing about the record replacing the pid
    file. With it, the row is red until the record lands and red again if the record breaks them.
    """
    supervisor = rig.supervisor("5")
    rig.attr(supervisor, "start_service", "5")
    rig.attr(supervisor, "read_state_record", "5")
    read_pid_file = rig.attr(supervisor, "read_pid_file", "5")
    stop = rig.attr(supervisor, "stop", "5")
    _logs, runs = dirs

    recorded = json.loads(rig.PID_FILE_T0033.read_text())
    assert recorded["pid"] == 3574545, (
        "the fixture is the STT record the failed start of 2026-09-16 left behind; if it was "
        "re-recorded, the MANIFEST and this row move together"
    )

    verbatim = runs / "stt.pid"
    verbatim.write_text(rig.PID_FILE_T0033.read_text())
    assert read_pid_file(verbatim) == (
        recorded["pid"],
        tuple(recorded["argv"]),
        recorded["start_time"],
    ), "criterion 5: a T-0033-shape file still reads as (pid, argv, start_time)"
    assert rig.attr(supervisor, "read_state_record", "5")(verbatim) is None, (
        "criterion 5/8: …and it is *not* a versioned record: it carries no kind and no script "
        "digest, so it cannot certify provenance — that is exactly why this box's healthy "
        "services are criterion 8's CONFLICT and not a SKIP"
    )
    assert verbatim in rig.attr(supervisor, "pid_files", "5")(runs), (
        "criterion 5: …and `pid_files` still lists it, so `down`/`status` still see it"
    )
    verbatim.unlink()

    # (a) stale: the recorded process is gone. Kill nothing, remove the file, exit clean.
    dead = subprocess.Popen(["sleep", "20"], start_new_session=True)
    dead_pid, dead_start = dead.pid, start_time_of(dead.pid)
    dead.kill()
    dead.wait(timeout=5)
    assert not box.alive(dead_pid), "the pid is free again"
    stale_file = runs / "vllm.pid"
    stale_file.write_text(rig.t0033_pid_file_bytes(pid=dead_pid, start_time=dead_start))
    outcome = stop(stale_file)
    assert not outcome.still_running and not stale_file.exists(), (
        f"criterion 5 (T-0033 criterion 13): a stale T-0033 file is removed, nothing is killed: "
        f"{outcome}"
    )

    # (b) reused: the pid is live but belongs to somebody else — the start time says so.
    victim = subprocess.Popen(["sleep", "20"], start_new_session=True)
    try:
        reused_file = runs / "stt.pid"
        reused_file.write_text(
            rig.t0033_pid_file_bytes(pid=victim.pid, start_time=start_time_of(victim.pid) - 1)
        )
        outcome = stop(reused_file)
        assert box.alive(victim.pid) and victim.poll() is None, (
            "criterion 5 (T-0033 ruling A11): the recorded start time does not match /proc, so "
            f"this pid was reused and must never be signalled. It killed the stranger: {outcome}"
        )
        assert not reused_file.exists(), f"…and the stale file is removed: {outcome}"
    finally:
        box.reap(victim)

    # (c) exec'd: the argv no longer matches, but the start time says it is our process.
    execd = subprocess.Popen(["sleep", "20"], start_new_session=True)
    try:
        execd_file = runs / "tts.pid"
        execd_file.write_text(
            rig.t0033_pid_file_bytes(pid=execd.pid, start_time=start_time_of(execd.pid))
        )
        assert tuple(recorded["argv"]) != ("sleep", "20"), "the argvs must differ for this to bite"
        outcome = stop(execd_file)
        assert box.wait_until(lambda: not box.alive(execd.pid), timeout_s=5), (
            "criterion 5 (T-0033 ruling A11): the start time matches, so this is the process we "
            f"started and it exec'd — it must still be stopped: {outcome}"
        )
        assert any(word in outcome.message.lower() for word in ("exec", "argv")), (
            f"…and the argv mismatch is reported rather than hidden: {outcome}"
        )
    finally:
        box.reap(execd)
