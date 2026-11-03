"""``astra doctor``: the verdicts are a pure function of a box description (criteria 1, 2, 4-6, 8, 9).

Every expectation here is computed by hand from the criteria in ``docs/tasks/T-0033.md`` against a
``Facts`` the test writes out in full — never from what an implementation returns. The severity
rule is the task's: **FAIL = the first live run cannot succeed; WARN = a fact that should be
pinned but does not block.**

The seam (``preflight.Facts`` / ``Check`` / ``evaluate`` / ``gather``, and the check names) is in
``tests/unit/_box_rig.py``'s docstring.

Regime: no measurement except criterion 9's wall bound (< 5 s with every collector stubbed, n = 3,
no seed, no GPU, no network); nothing here starts a service or opens a device.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from pathlib import Path

import pytest

from local_astra import cli
from tests.unit import _box_rig as rig

REPO = rig.REPO


@pytest.fixture(autouse=True)
def _no_sounddevice_leak():
    """Undo any real ``import sounddevice`` a row in this module performed (T-0035 ruling A16).

    Some rows here run the *real* collectors — ``test_doctor_is_a_verb`` runs `astra doctor`
    against this box — and ``preflight.collect_audio()`` is by design exactly the import
    ``astra session`` makes. On a box with ``libportaudio2`` installed that import succeeds and
    leaves the library in ``sys.modules`` for every later module in the same worker, which is
    invisible here and fatal in
    ``tests/unit/test_audio_smoke_script.py::test_the_script_imports_without_portaudio``: that
    row asks whether the *smoke script* pulled in a sound library, and a residue from this
    module answers for it. Process-global state that a row creates is that row's to clean up,
    so each row starts and ends with ``sounddevice`` exactly as it found it.
    """
    present = "sounddevice" in sys.modules
    before = sys.modules.get("sounddevice")
    try:
        yield
    finally:
        if present:
            sys.modules["sounddevice"] = before
        else:
            sys.modules.pop("sounddevice", None)


def run_cli(argv: list[str], **kwargs) -> int:
    try:
        return int(cli.main(argv, **kwargs))
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    except TypeError as exc:
        raise AssertionError(
            f"T-0033: cli.main(argv, *, runner=None, probe=None) — {exc}"
        ) from None


def stub_gather(monkeypatch, box: dict) -> None:
    """``astra doctor`` must reach the box only through ``preflight.gather(profile)``."""
    preflight = rig.module("preflight")
    built = rig.facts(**box)
    rig.attr(preflight, "gather")
    monkeypatch.setattr(preflight, "gather", lambda *a, **k: built)


# --- criterion 1 + 2 + 9: this box on 2026-09-14 ------------------------------------------------

#: The four things that stop the first live run here, and the fix each line must carry.
#: 1. no PortAudio -> `astra session` exits 1 while bootstrap says ready (P0-3);
#: 2. the default sink is `Dummy Output` (nothing is audible);
#: 3. no capture source at all (nothing is heard);
#: 4. a service that answers but is not healthy (P1-6).
THIS_BOX = rig.good_box(
    sounddevice_ok=False,
    sounddevice_error="PortAudio library not found (libportaudio.so.2)",
    default_sink="Dummy Output",
    sources=(),
    service_health={"vllm": None, "stt": None, "tts": "HTTP 503 from http://127.0.0.1:8081/health"},
    stt_argv=rig.STT_ARGV_CPU,
    vllm_argv=rig.VLLM_ARGV_NO_REVISION,
)
#: Hand-computed from THIS_BOX and the criteria: four FAILs, two WARNs (the CPU-STT mismatch of
#: criterion 5 and the missing `--revision` of the carve-out), and the remaining rows PASS.
#:
#: T-0058's `vram.engine_start` is one of those PASSes, and the arithmetic is worth stating
#: because an earlier version of this file had it as a third WARN (ruling A12). This box has
#: 5 174 MiB free *with the engine resident*, but a restart releases the 24 203 MiB the engine
#: itself holds, so what the new engine has to fit into is 29 377 MiB; minus the ~512 MiB of CUDA
#: context it creates before it looks, 28 865 against the 27 591 it demands at 0.86. It fits,
#: with 1 274 MiB to spare, so the row is quiet -- which is the whole point of the flag. At the
#: 0.92 default the same box is 651 MiB short, and that is the WARN, pinned below.
THIS_BOX_FAILS = {"audio.sounddevice", "audio.sink", "audio.source", "health.tts"}
THIS_BOX_WARNS = {"container.stt_device", "container.vllm_argv"}
THIS_BOX_PASSES = set(rig.REQUIRED_CHECKS) - THIS_BOX_FAILS - THIS_BOX_WARNS


def test_this_box_on_2026_09_14_fails_and_names_all_four_blockers(monkeypatch, capsys):
    """Criteria 1, 2, 9: the exact multiset of statuses, and a fix on every FAIL line."""
    checks = rig.evaluate(**THIS_BOX)
    got = rig.statuses(checks)
    assert got["FAIL"] == THIS_BOX_FAILS, (
        f"criteria 1-3: exactly the four blockers fail on this box; got FAIL={sorted(got['FAIL'])}"
    )
    assert got["WARN"] == THIS_BOX_WARNS, f"got WARN={sorted(got['WARN'])}"
    assert got["PASS"] == THIS_BOX_PASSES, f"got PASS={sorted(got['PASS'])}"

    rows = rig.by_name(checks)
    assert "apt install libportaudio2" in rows["audio.sounddevice"].fix, (
        "criterion 2: the PortAudio fix line is the exact gap between bootstrap and "
        f"`astra session`; it said {rows['audio.sounddevice'].fix!r}"
    )
    assert "Dummy Output" in rows["audio.sink"].detail, (
        f"criterion 2: the sink line names the dummy sink; detail={rows['audio.sink'].detail!r}"
    )
    assert "503" in rows["health.tts"].detail, (
        f"criterion 3: the health line names the status; detail={rows['health.tts'].detail!r}"
    )
    for name in sorted(THIS_BOX_FAILS):
        assert rows[name].fix.strip(), f"criterion 1: every FAIL carries a fix; {name} carries none"

    stub_gather(monkeypatch, THIS_BOX)
    code = run_cli(["doctor"])
    out = capsys.readouterr()
    report = out.out + out.err
    assert code != 0, f"criterion 9: exit non-zero iff at least one FAIL; it exited {code}"
    for name in rig.REQUIRED_CHECKS:
        assert name in report, f"criterion 9: one line per check; {name} is missing from\n{report}"
    assert "FAIL" in report and "WARN" in report, (
        f"criterion 9: WARNs are still printed beside the FAILs; the report was\n{report}"
    )


def test_a_tbd_lock_line_and_a_cpu_stt_are_warnings_not_failures(monkeypatch, capsys):
    """Criteria 4 and 5: the human's open decisions do not block a run; exit stays 0."""
    box = rig.good_box(
        lock={**rig.PINNED_LOCK, "pocket_tts": "TBD", "chat_template_revision": "TBD"},
        stt_argv=rig.STT_ARGV_CPU,
        profile_stt_backend="kyutai-gpu",
        vllm_argv=rig.VLLM_ARGV_NO_REVISION,
    )
    checks = rig.evaluate(**box)
    rows = rig.by_name(checks)
    got = rig.statuses(checks)
    assert got["FAIL"] == set(), (
        "criteria 4 and 5: a TBD pin and a CPU STT stop a *number* from being comparable, not a "
        f"run; these failed: {sorted(got['FAIL'])}"
    )
    assert rows["lock.pins"].status == "WARN", f"lock.pins was {rows['lock.pins'].status}"
    lock_line = rows["lock.pins"].detail + " " + rows["lock.pins"].fix
    for missing in ("pocket_tts", "chat_template_revision"):
        assert missing in lock_line, (
            f"criterion 4: the WARN names the TBD line; {missing} is missing from {lock_line!r}"
        )
    assert "pocket-tts" in lock_line or "uvx" in lock_line, (
        f"criterion 4: …and the command that produces the value; it said {lock_line!r}"
    )

    stt = rows["container.stt_device"]
    assert stt.status == "WARN", f"criterion 5: a CPU container under kyutai-gpu is WARN, got {stt}"
    assert "1.3" in stt.detail and "RTF" in stt.detail.upper(), (
        f"criterion 5: the detail carries the measured cost (RTF ~ 1.3); it said {stt.detail!r}"
    )
    assert "80 ms" in stt.detail or "80ms" in stt.detail, (
        "CLAUDE.md: never a number without its regime — the RTF figure is per 80 ms of audio; "
        f"the detail said {stt.detail!r}"
    )
    assert "cpu" in stt.detail.lower(), "…and says which device the container actually runs on"

    vllm = rows["container.vllm_argv"]
    assert vllm.status == "WARN" and "--revision" in (vllm.detail + vllm.fix), (
        f"criterion 5: a missing --revision is a WARN naming the carve-out; got {vllm}"
    )

    stub_gather(monkeypatch, box)
    code = run_cli(["doctor"])
    report = capsys.readouterr()
    assert code == 0, (
        f"criterion 9: WARNs keep exit 0; it exited {code}. Output:\n{report.out}{report.err}"
    )
    assert "WARN" in report.out + report.err, "…and are still printed"


#: What ``collect_containers`` puts in ``Facts.container_error`` on this box when the login
#: session predates the docker group: one clause per collector that failed, joined with "; "
#: (``local_astra/preflight.py::collect_containers``). Written out here so the assertion below
#: compares against a literal, not against whatever the collector produced.
DOCKER_DENIED_TEXT = (
    "astra-stt: permission denied while trying to connect to the Docker daemon socket at "
    "unix:///var/run/docker.sock; "
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4: no running container from it"
)


def test_a_container_error_is_carried_verbatim_into_both_container_rows():
    """Criterion 5 / ruling A21: "not gathered" must say *why* it was not gathered.

    ``collect_containers`` is the only writer of ``Facts.container_error``: when ``docker inspect``
    refuses (the normal state on this box — the login session predates the docker group), both
    argv facts are ``None`` and the collector's exception text lands there. The two ``container.*``
    rows are then WARN (unknown is never evidence, and a missing argv does not stop a run), and
    each detail repeats that text **verbatim**, so a human reading the table learns it is a docker
    permission problem rather than being told a bare "not gathered".
    """
    box = rig.good_box(stt_argv=None, vllm_argv=None, container_error=DOCKER_DENIED_TEXT)
    checks = rig.evaluate(**box)
    rows = rig.by_name(checks)
    got = rig.statuses(checks)

    assert got["FAIL"] == set(), (
        "an un-inspectable container is not a blocker: the services may be up and healthy; "
        f"these failed: {sorted(got['FAIL'])}"
    )
    # Still exactly two: T-0058's `vram.engine_start` is a PASS on this description (a restart
    # releases the engine's own 24 203 MiB, so 28 865 MiB against 27 591 demanded at 0.86 —
    # ruling A12), and the container facts being ungathered does not change that.
    assert got["WARN"] == {"container.stt_device", "container.vllm_argv"}, (
        f"exactly the two rows whose facts are missing warn; got WARN={sorted(got['WARN'])}"
    )
    for name in ("container.stt_device", "container.vllm_argv"):
        assert DOCKER_DENIED_TEXT in rows[name].detail, (
            f"ruling A21: {name} must carry the collector's reason verbatim, not a bare "
            f'"not gathered"; it said {rows[name].detail!r}'
        )
        assert "sg docker" in rows[name].fix, (
            f"…and the fix names the retry that works on this box; {name} said {rows[name].fix!r}"
        )


def test_evaluate_is_pure(monkeypatch):
    """Criterion 1, negative: no subprocess, no socket, and the same answer twice."""
    preflight = rig.module("preflight")
    first = rig.evaluate(**THIS_BOX)

    import socket as socket_module
    import subprocess as subprocess_module

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "criterion 1: evaluate() is pure — it shelled out or opened a socket "
            f"({args[:1]!r}); the collectors are the only place allowed to touch the box"
        )

    monkeypatch.setattr(subprocess_module, "run", forbidden)
    monkeypatch.setattr(subprocess_module, "Popen", forbidden)
    monkeypatch.setattr(socket_module, "socket", forbidden)
    monkeypatch.setattr(socket_module, "create_connection", forbidden)
    second = rig.attr(preflight, "evaluate")(rig.facts(**THIS_BOX))
    third = rig.attr(preflight, "evaluate")(rig.facts(**THIS_BOX))

    assert [(c.name, c.status, c.detail, c.fix) for c in second] == [
        (c.name, c.status, c.detail, c.fix) for c in first
    ], "criterion 1: evaluate is a function of its facts, not of the box"
    assert [c.name for c in third] == [c.name for c in second], (
        "criterion 1: …and the rows come back in the same order every call"
    )


def test_a_box_with_every_fact_missing_still_produces_a_full_table():
    """Criterion 1, adversarial: a doctor that drops a row it could not gather hides a P1-6.

    ``Facts()`` is the box where every collector failed. Nothing may crash, every row must be
    present, and no row may claim PASS: an ungathered fact is not evidence that the box is fine.
    """
    checks = rig.evaluate()
    rows = rig.by_name(checks)
    passing = sorted(name for name, check in rows.items() if check.status == "PASS")
    assert not passing, (
        "criterion 1: a fact that was not gathered is never a PASS; these rows claimed one "
        f"with an empty Facts(): {passing}"
    )
    for name, check in rows.items():
        assert check.detail.strip(), f"{name}: the row says nothing about why it could not decide"


def test_every_collector_may_fail_and_gather_still_returns_facts(monkeypatch):
    """Criterion 1: collectors are injected and each is allowed to fail into 'not gathered'."""
    preflight = rig.module("preflight")
    collectors = [n for n in dir(preflight) if n.startswith("collect_")]
    assert collectors, (
        "criterion 1: the collectors are named collect_* so `gather` can be tested with all of "
        "them failing (tests/unit/_box_rig.py)"
    )

    def boom(*args, **kwargs):
        raise OSError("the box refused")

    for name in collectors:
        monkeypatch.setattr(preflight, name, boom)
    from local_astra.config import load

    gathered = rig.attr(preflight, "gather")(load(None).profiles["lean"])
    checks = rig.attr(preflight, "evaluate")(gathered)
    rig.by_name(checks)
    assert all(c.status != "PASS" for c in checks), (
        "criterion 1: every collector raised, so nothing is known and nothing may PASS"
    )


# --- criterion 4: the served model is the pinned model -------------------------------------------


def test_a_served_model_or_revision_that_is_not_the_pinned_one_is_a_failure():
    rows = rig.by_name(rig.evaluate(**rig.good_box(served_model_id="google/gemma-4-E2B-it")))
    served = rows["model.served"]
    assert served.status == "FAIL", f"criterion 4: a different model is FAIL; got {served}"
    assert "google/gemma-4-E2B-it" in served.detail and rig.PINNED_LOCK["model"] in served.detail, (
        f"criterion 4: the FAIL names both sides; it said {served.detail!r}"
    )

    rows = rig.by_name(rig.evaluate(**rig.good_box(hf_snapshot_revision="deadbeef" * 5)))
    revision = rows["model.revision"]
    assert revision.status == "FAIL", f"criterion 4: a different revision is FAIL; got {revision}"
    assert "deadbeef" in revision.detail and rig.PINNED_LOCK["model_revision"] in revision.detail, (
        f"criterion 4: the FAIL names both sides; it said {revision.detail!r}"
    )


# --- criterion 6: VRAM against the 3 GiB floor ---------------------------------------------------


def test_vram_below_the_three_gib_sampler_floor_is_a_failure_and_no_gpu_is_a_warning():
    """Criterion 6: free = total - used against the >= 3 GiB floor (docs/02 §7).

    Hand-computed: 30500/32607 MiB leaves 2107 MiB = 2.06 GiB, under the 3072 MiB floor -> FAIL.
    27433/32607 (this box, tests/fixtures/box/nvidia_smi.csv) leaves 5174 MiB = 5.05 GiB -> PASS.
    """
    low = rig.by_name(rig.evaluate(**rig.good_box(vram_used_mib=30500)))["vram.free"]
    assert low.status == "FAIL", f"criterion 6: 2107 MiB free is under the floor; got {low}"
    assert "30500" in low.detail and "32607" in low.detail, (
        f"criterion 6: the detail states used/total; it said {low.detail!r}"
    )

    ok = rig.by_name(rig.evaluate(**rig.good_box()))["vram.free"]
    assert ok.status == "PASS", f"criterion 6: 5174 MiB free clears the floor; got {ok}"

    laptop = rig.by_name(rig.evaluate(**rig.good_box(vram_used_mib=None, vram_total_mib=None)))[
        "vram.free"
    ]
    assert laptop.status == "WARN", (
        f"criterion 6: no nvidia-smi is a WARN ('laptop'), never a crash and never a PASS; got {laptop}"
    )


# --- criterion 8: toolchain and tree -------------------------------------------------------------


def test_a_missing_browser_fails_and_a_dirty_tree_warns_with_the_paths():
    tools = {**rig.TOOLS_PRESENT, "chromium": None}
    rows = rig.by_name(
        rig.evaluate(**rig.good_box(tools=tools, dirty_paths=("local_astra/cli.py",)))
    )
    assert rows["tool.chromium"].status == "FAIL", (
        "criterion 8: there is no live run without a page, so a missing chromium is FAIL; got "
        f"{rows['tool.chromium']}"
    )
    tree = rows["git.tree"]
    assert tree.status == "WARN", f"criterion 8: a dirty tree is a WARN, not a FAIL; got {tree}"
    assert "local_astra/cli.py" in tree.detail, (
        f"criterion 8: …naming the paths (the reason guard_bash refuses); it said {tree.detail!r}"
    )
    clean = rig.by_name(rig.evaluate(**rig.good_box()))["git.tree"]
    assert clean.status == "PASS", f"a clean tree passes; got {clean}"


# --- criterion 9: the machine-readable form, the exit code, and what doctor must not do ----------


def test_doctor_json_prints_one_object_per_check_and_nothing_else(monkeypatch, capsys):
    stub_gather(monkeypatch, THIS_BOX)
    code = run_cli(["doctor", "--json"])
    out = capsys.readouterr()
    assert code != 0, "criterion 9: --json keeps the same exit rule"
    try:
        objects = rig.json_lines(out.out)
    except ValueError as exc:
        raise AssertionError(
            f"criterion 9: --json prints one JSON object per check and nothing else on stdout; "
            f"stdout was:\n{out.out}\n({exc})"
        ) from None
    assert len(objects) == len(rig.REQUIRED_CHECKS), (
        f"criterion 9: one object per check ({len(rig.REQUIRED_CHECKS)}); got {len(objects)}"
    )
    for obj in objects:
        assert set(obj) == {"name", "status", "detail", "fix"}, (
            f"criterion 9: stable keys name/status/detail/fix; one object had {sorted(obj)}"
        )
    assert {obj["name"] for obj in objects} == set(rig.REQUIRED_CHECKS)
    assert {obj["status"] for obj in objects} <= set(rig.STATUSES)


def test_doctor_starts_nothing_writes_nothing_and_opens_no_listening_socket(
    monkeypatch, tmp_path, capsys
):
    """Criterion 9 + criterion 7, negative: read-only, and it listens nowhere."""
    import socket as socket_module

    before_lock = hashlib.sha256((REPO / "services" / "versions.lock").read_bytes()).hexdigest()
    (tmp_path / "marker").write_text("x")
    monkeypatch.chdir(tmp_path)

    def no_listen(self, *args, **kwargs):
        raise AssertionError("criterion 7, negative: `astra doctor` opens no listening socket")

    monkeypatch.setattr(socket_module.socket, "listen", no_listen, raising=True)
    stub_gather(monkeypatch, THIS_BOX)
    run_cli(["doctor"])
    capsys.readouterr()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["marker"], (
        f"criterion 9: `astra doctor` writes nothing; {tmp_path} gained "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )
    assert not (tmp_path / "artifacts").exists()
    after_lock = hashlib.sha256((REPO / "services" / "versions.lock").read_bytes()).hexdigest()
    assert after_lock == before_lock, "criterion 9: doctor never mutates services/versions.lock"


def test_doctor_completes_in_under_five_seconds_with_every_collector_stubbed(monkeypatch, capsys):
    """Criterion 9's regime: n = 3, collectors stubbed, no seed. Not a performance claim."""
    stub_gather(monkeypatch, THIS_BOX)
    walls = []
    for _ in range(3):
        started = time.monotonic()
        run_cli(["doctor"])
        walls.append(time.monotonic() - started)
        capsys.readouterr()
    assert max(walls) < 5.0, (
        f"criterion 9: `astra doctor` with stubbed collectors is < 5 s (n=3); walls were {walls}"
    )


def test_doctor_is_a_verb(capsys):
    """Criterion 9: the verb exists and is not the 'not implemented yet' stub."""
    assert Path(REPO / "local_astra" / "preflight.py").exists(), (
        "criterion 1: local_astra/preflight.py is where the verdicts live"
    )
    code = run_cli(["doctor", "--json"])
    printed = capsys.readouterr()
    assert "not implemented yet" not in printed.err, (
        f"criterion 9: `astra doctor` is real; it said {printed.err!r}"
    )
    assert code in (0, 1), f"criterion 9: doctor exits 0 or 1 (a FAIL), not {code}"


# --- criterion 2, negative: the audio check listens to nothing and plays nothing -----------------


class _ForbiddenStream:
    """Constructing one is the failure under test."""

    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "criterion 2, negative: the audio check opens no stream — it plays nothing and "
            "records nothing. `import sounddevice` + query_devices() is the whole probe."
        )


class _FakeSounddevice:
    """The three answers criterion 2 asks for, and a trap on everything that opens a device."""

    InputStream = _ForbiddenStream
    OutputStream = _ForbiddenStream
    RawInputStream = _ForbiddenStream
    RawOutputStream = _ForbiddenStream
    Stream = _ForbiddenStream
    default = None

    def __init__(self, devices):
        self._devices = devices
        self.queried = 0

    def query_devices(self, *args, **kwargs):
        self.queried += 1
        return self._devices

    def play(self, *args, **kwargs):
        raise AssertionError("criterion 2, negative: the audio check plays nothing")

    def rec(self, *args, **kwargs):
        raise AssertionError("criterion 2, negative: the audio check records nothing")


def test_the_audio_check_opens_no_stream(monkeypatch):
    """Criterion 2: `import sounddevice` + `query_devices()` — and nothing else touches a device."""
    preflight = rig.module("preflight")
    fake = _FakeSounddevice(
        [
            {"name": "Dummy Output", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Logitech Webcam C925e", "max_input_channels": 2, "max_output_channels": 0},
        ]
    )
    if hasattr(preflight, "sounddevice"):
        monkeypatch.setattr(preflight, "sounddevice", fake)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake)

    gathered = rig.attr(preflight, "collect_audio")()
    assert fake.queried >= 1, (
        "criterion 2: the check is exactly the call `astra session` makes — "
        "`import sounddevice; sounddevice.query_devices()`"
    )
    assert isinstance(gathered, dict) and gathered.get("sounddevice_ok") is True, (
        f"criterion 2: collect_audio() returns the Facts fields it gathered; it returned {gathered!r}"
    )


def test_missing_portaudio_is_a_failure_that_names_the_fix():
    """Criterion 2 (P0-3): the exact gap between 'bootstrap says ready' and `astra session` -> 1."""
    rows = rig.by_name(
        rig.evaluate(
            **rig.good_box(
                sounddevice_ok=False,
                sounddevice_error="PortAudio library not found (libportaudio.so.2)",
            )
        )
    )
    check = rows["audio.sounddevice"]
    assert check.status == "FAIL", f"criterion 2: no PortAudio is FAIL; got {check}"
    assert "apt install libportaudio2" in check.fix
    assert "PortAudio" in check.detail


def test_no_capture_source_is_a_failure():
    rows = rig.by_name(rig.evaluate(**rig.good_box(sources=())))
    assert rows["audio.source"].status == "FAIL", (
        f"criterion 2: with no capture source there is nothing to hear; got {rows['audio.source']}"
    )
    ok = rig.by_name(rig.evaluate(**rig.good_box()))["audio.source"]
    assert ok.status == "PASS", f"…and one source is enough; got {ok}"


# =================================================================================================
# T-0058 criteria 6-8: `vram.engine_start` -- can the engine still be started on this box?
#
# A different question from `vram.free`, which asks whether a *running* engine has room for the
# sampler transient. On 2026-09-16 at 22:17 the answer to this one was no, and nothing said so:
# the engine burned 21 s and died with
#
#     ValueError: Free memory on device cuda:0 (27.23/31.33 GiB) on startup is less than desired
#     GPU memory utilization (0.92, 28.83 GiB).
#
# The arithmetic, with the reading each figure comes from, is in `tests/unit/_box_rig.py` under
# "the cold boot that could not start"; the rows below recompute the verdicts from those MiB
# figures by hand. Three facts decide every row: the driver's own `memory.free`, the utilisation
# `services/vllm.sh`'s argv will carry, and whether the engine is already healthy.
# =================================================================================================

REQUIRED = "vram.engine_start"

#: vLLM takes its free-memory snapshot *after* creating its own CUDA context, which cost 509 MiB
#: on the recorded boot (28 391 -> 27 882). The row must subtract it; T-0058's implementation
#: notes say to round it up to 512 and not to tune it, so both readings are accepted here.
PREDICTED_FREE_AT_CHECK = (
    rig.COLDBOOT_FREE_MIB - rig.DOCUMENTED_INIT_OVERHEAD_MIB,  # 27 879
    rig.COLDBOOT_FREE_MIB - rig.MEASURED_INIT_OVERHEAD_MIB,  # 27 882
)


def _numbers(text: str) -> list[int]:
    """Every integer in a row's text, with thousands separators tolerated."""
    return [int(hit.replace(",", "").replace("_", "")) for hit in re.findall(r"\d[\d,_]*", text)]


def _states(text: str, value: int, *, tolerance: int = 0) -> bool:
    return any(abs(number - value) <= tolerance for number in _numbers(text))


def engine_start(**overrides):
    """The `vram.engine_start` row for a box description, with the other facts gathered."""
    return rig.by_name(rig.evaluate(**rig.good_box(**overrides)))[REQUIRED]


#: The box as `nvidia-smi` described it seconds before the failed launch: the recogniser and
#: Chrome resident, the engine not running, and `services/vllm.sh` passing no utilisation flag.
COLDBOOT_BOX = {
    "vram_free_mib": rig.COLDBOOT_FREE_MIB,
    "vram_used_mib": rig.COLDBOOT_USED_MIB,
    "vram_total_mib": rig.COLDBOOT_TOTAL_MIB,
    "vram_processes": (
        {
            "pid": 11419,
            "process_name": "/opt/google/chrome/chrome --type=gpu-process",
            "used_mib": rig.CHROME_MIB,
        },
        {"pid": 3380309, "process_name": "moshi-server", "used_mib": rig.RECOGNISER_MIB},
    ),
    "vllm_utilization": rig.VLLM_DEFAULT_UTILIZATION,
    "vllm_utilization_is_default": True,
    "service_health": {
        "vllm": "connection refused to http://127.0.0.1:8001/health",
        "stt": None,
        "tts": None,
    },
}


def test_engine_start_row_reproduces_the_2026_09_16_refusal():
    """Criterion 6: the recorded facts in, FAIL out, and the arithmetic in the detail.

    Hand-computed from the recording, every step:

    * free now (`nvidia-smi --query-gpu=memory.free`) = **28 391 MiB**;
    * minus the engine's own CUDA-context cost, 509-512 MiB, gives the free memory vLLM will
      see at its check: **27 879-27 882 MiB** (it measured 27 882);
    * the demand is 0.92 (the v0.29.0 default, because the script passes no flag) of vLLM's own
      total, ≈ 32 082 MiB, = **29 515 MiB**; the engine's own message says 28.83 GiB = 29 522;
    * deficit **≈ 1 636 MiB = 1.60 GiB**, which is the figure in the ValueError.

    FAIL rather than WARN because the engine is *not* healthy: on this box the first live run
    cannot succeed, which is T-0033's definition of a FAIL.
    """
    row = engine_start(**COLDBOOT_BOX)
    text = row.detail + " " + row.fix
    assert row.status == "FAIL", (
        "criterion 6: the recogniser holds 3 180 MiB, the engine demands 29 515 MiB free and "
        f"only ~27 880 will be there when it looks — the boot cannot succeed. Got {row}"
    )
    assert _states(row.detail, rig.COLDBOOT_FREE_MIB), (
        f"criterion 6: the detail states the free memory it read (28391 MiB); it said "
        f"{row.detail!r}"
    )
    assert any(_states(row.detail, value) for value in PREDICTED_FREE_AT_CHECK), (
        "criterion 6: …and what will still be free when the engine looks, after its own CUDA "
        f"context (28391 - ~512 = ~27880) — the whole point of the row; it said {row.detail!r}"
    )
    assert _states(row.detail, rig.ENGINE_DEMAND_AT_DEFAULT_MIB, tolerance=15), (
        "criterion 6: …the demand (0.92 x 32 082 ≈ 29 515 MiB; the engine itself said 29 522); "
        f"it said {row.detail!r}"
    )
    assert _states(row.detail, rig.ENGINE_DEFICIT_MIB, tolerance=50), (
        "criterion 6: …and the deficit, which must agree with the engine's own message "
        f"(1 640 MiB) to within 50 MiB. The detail said {row.detail!r}"
    )
    assert "1.6" in row.detail, (
        f"criterion 6: …stated in GiB as well, the unit the ValueError uses; it said {row.detail!r}"
    )
    assert "moshi-server" in text and _states(text, rig.RECOGNISER_MIB), (
        "criterion 6: …and which processes hold the memory, because that is what the human acts "
        f"on. The row said {row}"
    )
    assert "32082" in text.replace(" ", "") or "31.33" in text or "mem_get_info" in text, (
        "criterion 8: the row names the denominator it used and where it came from (vLLM's own "
        f"mem_get_info total, 31.33 GiB). It said {row.detail!r}"
    )
    lowered = text.lower()
    assert "stt" in lowered or "recogniser" in lowered or "recognizer" in lowered, (
        f"criterion 6: the fix names the recogniser as the thing holding the memory; got {row.fix!r}"
    )
    assert "order" in lowered or "first" in lowered, (
        "criterion 6: …and the remedy, which is the boot order (engine first, recogniser "
        f"second); got {row.fix!r}"
    )


def test_a_healthy_engine_that_could_not_be_restarted_is_a_warn_not_a_fail():
    """Criterion 7: the row that would have predicted the 22:17 failure a day early.

    The box before the flag landed — engine healthy, recogniser resident, the launcher passing
    no `--gpu-memory-utilization` so vLLM's 0.92 default applies. Hand-computed:

    * free with everything resident: **5 174 MiB**;
    * plus what the engine's own process holds and a restart therefore releases first:
      **24 203 MiB** -> 29 377 MiB is what the new engine has to fit into;
    * minus the ~512 MiB CUDA context it creates before it looks: **28 865 MiB**;
    * demanded at the 0.92 default: ceil(0.92 x 32 082) = **29 516 MiB**;
    * **651 MiB short** — so the engine that is answering right now could not be started again
      where it stands, and a human about to `astra up --restart vllm` is about to lose it.

    WARN, not FAIL: a running box is not broken (T-0033's severity rule). And the arithmetic
    must be *this* arithmetic — ruling A12: an earlier implementation compared the demand
    against free memory **while the engine was resident**, which is 22 GiB short on every
    healthy box, so it warned always and said something false while doing it.
    """
    row = engine_start(vllm_utilization=0.92, vllm_utilization_is_default=True)
    assert row.status == "WARN", (
        "criterion 7: at the 0.92 default this box is 651 MiB short of what a restart needs "
        f"(29 377 - 512 = 28 865 against 29 516) — running, not restartable. Got {row}"
    )
    assert _states(row.detail, 651, tolerance=5), (
        "criterion 7: the row states how short the restart would be. 651 MiB, from 29 516 "
        f"demanded against 28 865 predicted. It said {row.detail!r}"
    )
    assert _states(row.detail, 28865, tolerance=3) or _states(row.detail, 29377, tolerance=3), (
        "ruling A12: …and shows that it counted the memory the restart releases — 29 377 MiB "
        "available to a restart, 28 865 after the engine's own CUDA context. A row that compares "
        f"the 5 174 MiB free right now warns on every healthy box. It said {row.detail!r}"
    )
    text = (row.detail + " " + row.fix).lower()
    assert "restart" in text or "again" in text, (
        "criterion 7: the row says plainly that the engine is running but could not be started "
        f"again in place; it said {row}"
    )
    assert "order" in text or "first" in text, (
        f"criterion 7: …and gives the boot order as the fix; it said {row.fix!r}"
    )
    assert rig.statuses(rig.evaluate(**rig.good_box()))["FAIL"] == set(), (
        "criterion 7: a box whose engine is up and answering is not a failing box; this row must "
        "not turn `astra doctor` red on the configuration the demo runs in"
    )


def test_the_same_healthy_box_at_the_committed_value_is_a_pass():
    """Criterion 7's complement, and the point of the whole task: the flag makes it quiet.

    The same box, the same processes, the same free memory — only the launcher's value differs.
    28 865 MiB will be there for the restart; 0.86 demands 27 591; it fits with **1 274 MiB** to
    spare, so the row says nothing. A doctor that warned here anyway would be telling a human to
    fix something the task has already fixed (ruling A12).
    """
    row = engine_start()
    assert row.status == "PASS", (
        "criterion 7: 28 865 MiB available to a restart clears the 27 591 MiB demanded at 0.86, "
        f"so the row is quiet. Got {row}"
    )
    assert _states(row.detail, 27591, tolerance=2), (
        f"…and it still shows the demand it cleared; it said {row.detail!r}"
    )


def test_a_healthy_engine_that_is_not_among_the_holders_is_ungathered():
    """Ruling A12, adversarial: the row may not guess how much a restart would release.

    The engine answers its health endpoint, but no compute app on the card looks like the
    engine's process — the container's pid namespace, a driver that lists nothing, a stale
    reading. The honest answer is "not gathered": assuming it holds nothing warns on a box that
    is fine, and assuming it holds 27 GiB passes a box that is not. T-0033's discipline says an
    ungathered fact is never a PASS, and it is not a FAIL either.
    """
    row = engine_start(
        vram_processes=(
            {"pid": 3380309, "process_name": "moshi-server", "used_mib": rig.RECOGNISER_MIB},
            {"pid": 11419, "process_name": "/opt/google/chrome/chrome", "used_mib": rig.CHROME_MIB},
        )
    )
    assert row.status == "WARN", (
        "ruling A12: the engine is healthy but is not among the holders, so how much a restart "
        f"would release is unknown — and unknown is neither a PASS nor a FAIL. Got {row}"
    )
    assert "not gathered" in row.detail, (
        f"…and it says so in the module's own words (`_ungathered`); it said {row.detail!r}"
    )


def test_using_the_nvidia_smi_total_would_fail_a_bootable_box():
    """Criterion 8, adversarial 1: the two denominators differ by 525 MiB and only one is right.

    Facts: 28 391 MiB free (the recorded reading), the script at 0.86, the engine down.

    * against vLLM's own total (32 082): demand **27 591 MiB**, predicted free at the check
      27 879 — it fits, so **PASS**;
    * against `nvidia-smi`'s total (32 607): demand **28 043 MiB**, 452 MiB more, which does
      *not* fit — **FAIL** on a box that would in fact boot.

    A doctor that refuses a working configuration gets turned off, so this is not a cosmetic
    error. The verdict is the discriminator: an implementation that picks the wrong total
    reddens here, and nowhere else.
    """
    # `vllm/v1/worker/utils.py::request_memory` is `requested = ceil(total * util)`, so the two
    # demands are 28 043 and 27 591 MiB.
    assert math.ceil(0.86 * rig.COLDBOOT_TOTAL_MIB) - math.ceil(0.86 * rig.VLLM_TOTAL_MIB) == 452, (
        "fixture check: at 0.86 the two denominators differ by 452 MiB — the gap this row exists "
        "to catch"
    )
    row = engine_start(
        vram_free_mib=rig.COLDBOOT_FREE_MIB,
        vram_used_mib=rig.COLDBOOT_USED_MIB,
        # The engine is down, so it is not among the holders and releases nothing: what a start
        # has to fit into is the free reading itself (ruling A12's other side).
        vram_processes=COLDBOOT_BOX["vram_processes"],
        vllm_utilization=0.86,
        service_health={"vllm": "connection refused", "stt": None, "tts": None},
    )
    assert row.status == "PASS", (
        "criterion 8: 0.86 x 32 082 = 27 591 MiB demanded against 27 879 predicted free — this "
        "box boots. Computing the demand against nvidia-smi's 32 607 asks for 28 043 and fails "
        f"it. Got {row}"
    )
    assert _states(row.detail, 27591, tolerance=2), (
        "criterion 8: the row states the demand it computed (27 591 MiB at 0.86 of vLLM's "
        f"32 082); it said {row.detail!r}"
    )


def test_forgetting_the_engines_own_init_overhead_passes_a_box_that_dies():
    """Criterion 8, adversarial 2: the 509 MiB vLLM spends before it looks at free memory.

    Facts: 28 391 MiB free, and a script at **0.876**, which demands
    ceil(0.876 x 32 082) = **28 104 MiB**. That figure sits in the only window where the two
    implementations disagree:

    * 28 391 (raw free) > 28 104  -> a row comparing raw free says **PASS**;
    * 27 879 (free at the engine's check, after its CUDA context) < 28 104 -> the correct row
      says **FAIL**.

    This is the mutation that matters: it PASSes a box that then burns 21 s and dies, which is
    the failure the row exists to prevent, so a worthless row looks exactly like a working one
    everywhere except here. 0.876 is outside the window criterion 1 allows for the committed
    script; it is used here because it is the value that discriminates.
    """
    demand = 28104
    assert rig.COLDBOOT_FREE_MIB > demand > PREDICTED_FREE_AT_CHECK[1], (
        "fixture check: the demand must sit between the raw free reading and the free memory the "
        "engine will actually see, or this row cannot tell the two implementations apart"
    )
    row = engine_start(
        vram_free_mib=rig.COLDBOOT_FREE_MIB,
        vram_used_mib=rig.COLDBOOT_USED_MIB,
        vram_processes=COLDBOOT_BOX["vram_processes"],  # engine down: nothing to release
        vllm_utilization=0.876,
        service_health={"vllm": "connection refused", "stt": None, "tts": None},
    )
    assert row.status == "FAIL", (
        f"criterion 8: {demand} MiB demanded, but the engine's own CUDA context takes ~509 MiB "
        "first, so only ~27 879 will be free when it looks. A row that compares the raw 28 391 "
        f"calls this box fine and the boot dies anyway. Got {row}"
    )
    assert any(_states(row.detail, value) for value in PREDICTED_FREE_AT_CHECK), (
        "criterion 8: …and the row shows the subtraction rather than asserting a conclusion; it "
        f"said {row.detail!r}"
    )
    assert _states(row.detail, rig.DOCUMENTED_INIT_OVERHEAD_MIB) or _states(
        row.detail, rig.MEASURED_INIT_OVERHEAD_MIB
    ), (
        "criterion 8: …naming the overhead it subtracted (≈ 512 MiB, an estimate from n = 1 "
        f"across the 2026-09-16 boot); it said {row.detail!r}"
    )


@pytest.mark.parametrize(
    "missing",
    ["vram_free_mib", "vram_processes", "vllm_utilization", "service_health"],
)
def test_every_engine_start_fact_missing_is_ungathered_never_pass(missing: str):
    """Criterion 8: an ungathered fact is never a PASS (T-0033's discipline, `_ungathered`).

    Each fact is removed in turn from a box that is otherwise fully described. The row must say
    it could not decide — a green row here tells a human the engine will start when nobody
    asked the device.
    """
    row = engine_start(**{missing: None})
    assert row.status == "WARN", (
        f"criterion 8: with {missing} not gathered the row cannot decide, and unknown is not "
        f"evidence; got {row}"
    )
    assert "not gathered" in row.detail, (
        f"criterion 8: …and it says so in the module's own words (`_ungathered`); it said "
        f"{row.detail!r}"
    )


def test_the_flag_absent_default_is_named_with_the_version_it_belongs_to():
    """Criterion 8 / ambiguity 5: 0.92 is v0.29.0's default, not a law of nature.

    When `services/vllm.sh` passes no `--gpu-memory-utilization`, the row's arithmetic rests on
    a default that belongs to the pinned container. The next container pin may move it, so the
    row states both the figure and the version it read it from.
    """
    row = engine_start(**COLDBOOT_BOX)
    text = row.detail + " " + row.fix
    assert "0.92" in text, (
        f"criterion 8: the row names the default it assumed; it said {row.detail!r}"
    )
    assert "0.29" in text, (
        "criterion 8 / ambiguity 5: …and the vLLM version that default belongs to (v0.29.0), "
        f"because a container pin change re-derives it. It said {row.detail!r}"
    )
    passed = engine_start(
        **{**COLDBOOT_BOX, "vllm_utilization": 0.86, "vllm_utilization_is_default": False}
    )
    assert "0.86" in passed.detail, (
        "…and when the script does pass the flag, the row states the value it read rather than "
        f"a default; it said {passed.detail!r}"
    )


def test_the_free_vram_row_and_its_floor_are_unchanged():
    """Criterion 7, negative: the new row is an addition, not an edit of `vram.free`.

    `VRAM_FLOOR_MIB` is ten assumed copies of a 0.25 GiB logits tensor (T-0048 ruling A13); the
    measurement that could move it is T-0058 group C's, not this task's. Lowering it to make a
    box green would erase the only thing standing between the sampler and an OOM mid-turn.
    """
    preflight = rig.module("preflight")
    assert rig.attr(preflight, "VRAM_FLOOR_MIB") == 3 * 1024, (
        "criterion 7, negative: the 3 GiB sampler-transient floor (docs/02 §7) does not move in "
        "this task — not down to make a box pass, and not up either"
    )
    row = rig.by_name(rig.evaluate(**rig.good_box()))["vram.free"]
    assert row.status == "PASS", (
        f"criterion 7, negative: 5 174 MiB free clears the floor; got {row}"
    )
    assert "27433" in row.detail and "32607" in row.detail and "5174" in row.detail, (
        f"criterion 7, negative: …with the same used/total/free wording as before; got {row.detail!r}"
    )
    assert "3072" in row.detail, (
        f"…against the same floor, named in MiB as it always was; got {row.detail!r}"
    )


# --- criterion 8, negative: the doctor still only polls ------------------------------------------

#: Programs a collector may run, and the sub-commands that keep each one read-only. `docker ps`
#: and `docker inspect` describe; `docker run/start/stop/kill/rm` change the box. `nvidia-smi`
#: answers queries; `-l`/`-lms` make it a sampler that never returns, and `--gpu-reset` is not a
#: poll at all. `.claude/rules/gpu-services.md`: status polls are free, everything else is
#: /launch-gated — which is the whole reason `astra doctor` may run unattended.
READ_ONLY_PROGRAMS = {
    "nvidia-smi": None,
    "docker": {"ps", "inspect", "version", "info"},
    "sg": None,
    "ss": None,
    "wpctl": {"status"},
    "pw-dump": None,
    "git": {"status", "rev-parse"},
}
FORBIDDEN_SMI_FLAGS = (
    "-l",
    "--loop",
    "-lms",
    "--loop-ms",
    "--gpu-reset",
    "-r",
    "--applications-clocks",
)


def test_the_engine_start_collector_only_polls_and_starts_nothing(monkeypatch):
    """Criterion 8, negative: `astra doctor` spends no GPU time and starts or stops nothing.

    The new facts come from `nvidia-smi` queries, which are status polls. This row runs the real
    `gather()` with `subprocess.run` replaced by a recorder, then reads back **every** command
    the collectors issued — so a collector that shells `docker start` or leaves `nvidia-smi` in
    `-lms 200` sampling mode fails here rather than on the box.
    """
    preflight = rig.module("preflight")
    import subprocess as subprocess_module

    recorded: list[list[str]] = []
    canned = {
        "nvidia-smi": rig.fixture("nvidia_smi_memory_coldboot.csv"),
        "ss": rig.fixture("ss_tlnp.txt"),
        "wpctl": rig.fixture("wpctl_status.txt"),
    }

    def recorder(command, *args, **kwargs):
        recorded.append([str(part) for part in command])
        stdout = canned.get(Path(str(command[0])).name, "")
        return subprocess_module.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess_module, "run", recorder)
    # The health probes are HTTP, not subprocesses, and T-0033's own rows cover them; stubbing
    # them keeps this row about the commands and off the network.
    for collector in ("collect_health", "collect_served_model"):
        monkeypatch.setattr(preflight, collector, lambda *a, **k: {}, raising=True)
    from local_astra.config import load

    rig.attr(preflight, "gather")(load(None).profiles["lean"])

    assert recorded, "the collectors ran no command at all — the recorder is not wired in"
    for command in recorded:
        program = Path(command[0]).name
        assert program in READ_ONLY_PROGRAMS, (
            f"criterion 8, negative: `astra doctor` ran {program!r} ({command}); the collectors "
            f"may only poll: {sorted(READ_ONLY_PROGRAMS)}"
        )
        allowed = READ_ONLY_PROGRAMS[program]
        words = command[1:]
        if program == "sg":  # `sg docker -c "<command>"`: the payload is the real command
            assert words[:1] == ["docker"], f"unexpected sg invocation {command}"
            words = words[-1].split()[1:] if words else []
            allowed = READ_ONLY_PROGRAMS["docker"]
        if allowed is not None:
            # `-C <path>` and friends take their value as a separate word, which is not the
            # sub-command: `git -C /repo status` is a status.
            skip = False
            candidates = []
            for word in words:
                if skip:
                    skip = False
                    continue
                if word in ("-C", "--git-dir", "--work-tree", "-f", "--filter", "--format"):
                    skip = True
                    continue
                if not word.startswith("-"):
                    candidates.append(word)
            subcommand = candidates[0] if candidates else ""
            assert subcommand in allowed, (
                f"criterion 8, negative: `{' '.join(command)}` is not a read-only use of "
                f"{program}; allowed sub-commands are {sorted(allowed)}"
            )
        if program == "nvidia-smi":
            for flag in FORBIDDEN_SMI_FLAGS:
                assert flag not in words, (
                    f"criterion 8, negative: `{' '.join(command)}` puts nvidia-smi in "
                    f"{flag} mode; a doctor row asks one question and returns"
                )

    smi = [" ".join(command) for command in recorded if Path(command[0]).name == "nvidia-smi"]
    assert any("--query-compute-apps" in call for call in smi), (
        "criterion 6: the row names which processes hold the memory, so a collector has to ask "
        f"for them (--query-compute-apps). nvidia-smi was called: {smi}"
    )
    assert any("memory.free" in call for call in smi), (
        "criterion 8: …and the free figure is read from the driver's own memory.free column, "
        f"never derived from total - used. nvidia-smi was called: {smi}"
    )
