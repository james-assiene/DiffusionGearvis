"""The seams T-0033's tests assume, in one place (the way ``_hud_rig.py`` does for T-0028).

The implementer owns the *shape* of the modules; what the tests below fix is the public surface
they call. Written here, in one file, so the implementer can read the whole contract at once.

``local_astra/preflight.py`` — collect → describe → evaluate → render; only ``evaluate`` decides::

    Facts(**field)            # frozen dataclass, EVERY field keyword-only with default None
                              # (None means "not gathered"); the field names are FACT_FIELDS below
    Check(name, status, detail, fix)      # frozen dataclass; status in {"PASS","WARN","FAIL"}
    evaluate(facts: Facts) -> tuple[Check, ...]     # pure: no I/O, no clock, no subprocess
    gather(profile) -> Facts                        # runs the collectors; a collector that
                                                    # raises leaves its fields None
    collect_*(...)            # the collectors; discovered by name in the adversarial test
    parse_wpctl_status(text) -> dict        # keys: default_sink, sources
    parse_nvidia_smi_csv(text) -> dict      # keys: vram_used_mib, vram_total_mib
    parse_nvidia_smi_memory(text) -> dict   # T-0058; --query-gpu=memory.total,memory.used,
                                            # memory.free read BY HEADER NAME (the columns are
                                            # whatever the caller asked for, in that order);
                                            # keys: vram_total_mib, vram_used_mib, vram_free_mib
    parse_nvidia_smi_compute_apps(text) -> tuple[dict, ...]   # T-0058; keys per row: pid,
                                            # process_name, used_mib
    parse_vllm_utilization(text) -> float | None   # T-0058; the --gpu-memory-utilization value
                                            # in services/vllm.sh's executable text, or None
                                            # when the flag is absent (which is not the same as
                                            # "the script could not be read")
    parse_docker_inspect(text) -> dict      # keys: name, image, argv
    parse_ss_tlnp(text) -> tuple[dict, ...] # keys per row: address, port, process

``evaluate`` returns exactly one ``Check`` per name in ``REQUIRED_CHECKS``; the *order* is the
implementer's, but it must be the same order every call.

``local_astra/supervisor.py`` — the detached spawn and the real health probes::

    check_service(name, url, *, timeout_s=...) -> ProbeResult
        ProbeResult: .name .url .status_code (int|None) .body (str) .healthy (bool) .reason (str|None)
        .url is the URL actually probed (derived from the config value, loopback-validated);
        .reason is None iff .healthy
    probe_url(name, url) -> str            # pure: the URL that would be probed, loopback-checked
    probe(name, url) -> str | None          # = check_service(...).reason: the CLI's Probe seam
    probe_budget_s(name) -> float           # per-service probe budget (vLLM long, speech short)
    spawn(name, command, *, log_dir=None, run_dir=None) -> Handle
        detached (own session), stdout+stderr to <log_dir>/<name>.log, pid file written after the
        spawn succeeds; Handle: .name .pid .argv .log_path .pid_file
    run_dir() -> Path                       # <git common dir>/local_astra/run
    log_dir() -> Path                       # <repo>/logs
    write_pid_file(path, pid, argv, start_time) -> None      # start_time = /proc/<pid>/stat f.22
    read_pid_file(path) -> tuple[int, tuple[str, ...], int]  # (pid, argv, start_time), ruling A11

``local_astra/hud_static.py`` — the loopback static server::

    HUD_DIR: Path                                   # <repo>/hud
    page_urls(services) -> tuple[str, ...]          # the URLs a human can paste (criterion 16)
    HudStaticServer(root: Path, port: int = 0, *, poll_interval: float = 0.5)
                                                    # binds in __init__ (so a taken port raises
                                                    # OSError there), serves on a daemon thread;
                                                    # T-0110c: poll_interval is serve_forever's,
                                                    # tests pass 0.01, the CLI nothing, and
                                                    # 0/negative/nan/inf -> ValueError pre-bind
        .port .address (getsockname) .close()

``local_astra/cli.py`` — argparse-level entry points, as in ``tests/unit/test_cli_session.py``::

    cli.main(argv, *, runner=None, probe=None) -> int
        runner(command) -> Handle-like (.pid, .argv)        # supersedes T-0028's `-> int` seam
    cli.regime_line(profile, facts) -> str   # criterion 19; `key=value` tokens, at least
                                             # stt_device= and served_model=, each a measured
                                             # fact or the literal `unknown` plus the reason
    verbs: doctor [--json], up, down, status, hud, session --serve-hud
    `down`/`status` read the pid directory through ``supervisor.run_dir()`` at call time.

Nothing in this file imports the new modules at import time: they do not exist while these tests
are pending, and a collection error would take the T-0028 rows in ``test_cli_up.py`` down with it.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import signal
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOX = REPO / "tests" / "fixtures" / "box"

#: One row per check, always — a doctor that drops the row it could not gather is the failure
#: mode criterion 1's adversarial test exists for.
REQUIRED_CHECKS = (
    "audio.sounddevice",
    "audio.sink",
    "audio.source",
    "health.vllm",
    "health.stt",
    "health.tts",
    "model.served",
    "model.revision",
    "lock.pins",
    "container.stt_device",
    "container.vllm_argv",
    "vram.free",
    # T-0058 criteria 6-8: can the *engine* still start on this box? A different question
    # from vram.free, which asks whether the sampler has room while it is already running.
    "vram.engine_start",
    "net.loopback",
    "tool.uv",
    "tool.git",
    "tool.node",
    "tool.chromium",
    "git.tree",
    # T-0111g B4 (re-scope R7): the live browser executor's prerequisites; WARN at worst, since tools are off by
    # default.
    "browser.playwright",
    "browser.chromium",
    "browser.display",
    # T-0182 A3: the default-enabled clip assets; WARN at worst (a missing clip only means silence).
    "clips",
)

#: Every field of ``Facts``. All keyword, all defaulting to ``None`` = "not gathered".
FACT_FIELDS = (
    "sounddevice_ok",
    "sounddevice_error",
    "default_sink",
    #: T-0045 criterion 5: the PipeWire *node name* of the box's default capture source — the
    #: microphone `astra session` opens when no --mic says otherwise (`preflight.default_source()`).
    "default_source",
    "sources",
    "service_health",
    "served_model_id",
    "hf_snapshot_revision",
    "lock",
    "stt_argv",
    "vllm_argv",
    "profile_stt_backend",
    "vram_used_mib",
    "vram_total_mib",
    # T-0058 criteria 6-8. `nvidia-smi`'s own `memory.free` column: on this device it is NOT
    # `total - used` (the 2026-09-16 recording reads 32607 total, 3696 used, 28391 free -- a
    # 525 MiB gap), and the engine's startup check compares against the free reading, so the
    # engine-start row may not derive it by subtraction.
    "vram_free_mib",
    # The compute apps holding device memory: one dict per row of
    # `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory`, keys `pid`,
    # `process_name`, `used_mib`.
    "vram_processes",
    # The `--gpu-memory-utilization` the engine's argv will carry, read out of
    # `services/vllm.sh`: the flag's value, or vLLM's own default when the flag is absent.
    "vllm_utilization",
    # True when `vllm_utilization` is that *default* rather than a value the script passes --
    # the state of the box on 2026-09-16, and the row must say which version the default
    # belongs to (T-0058 criterion 8, ambiguity 5).
    "vllm_utilization_is_default",
    "listeners",
    "tools",
    "dirty_paths",
    # Why the container argv could not be read (T-0033 ruling A18); carried into the
    # ``container.*`` rows' detail when the argv facts are missing (ruling A21's owed row).
    "container_error",
    # T-0111g B4 (re-scope R7): ``{"playwright", "chromium", "display"}`` from ``collect_browser``, each value
    # ``str | None``; see tests/unit/test_doctor_browser.py.
    "browser",
    # T-0182 A3: kind -> ``current``/``missing``/``stale`` for the default-enabled clip kinds, from ``collect_clips``;
    # ``{}`` = none registered; see tests/unit/test_t0182_doctor_clips.py.
    "clips",
)

STATUSES = ("PASS", "WARN", "FAIL")


def module(name: str):
    """Import ``local_astra.<name>``, or raise the AssertionError that says what is missing.

    AssertionError (not ImportError) on purpose: ``tests/conftest.py`` runs a pending test as
    ``xfail(strict=True, raises=(NotImplementedError, AssertionError))``, so a missing module is
    the documented "not implemented yet" outcome and a *wrong* call signature is not.
    """
    try:
        return importlib.import_module(f"local_astra.{name}")
    except ImportError as exc:  # pragma: no cover - the pending path
        raise AssertionError(
            f"T-0033: local_astra/{name}.py does not import ({exc}). "
            f"The seam it must offer is in tests/unit/_box_rig.py's docstring."
        ) from None


def attr(mod, name: str):
    value = getattr(mod, name, None)
    assert value is not None, (
        f"T-0033: {mod.__name__}.{name} is missing "
        f"(the seam is in tests/unit/_box_rig.py's docstring)"
    )
    return value


def facts(**overrides):
    """Build a ``preflight.Facts``; an unknown or missing field is an AssertionError, not a crash."""
    preflight = module("preflight")
    factory = attr(preflight, "Facts")
    unknown = sorted(set(overrides) - set(FACT_FIELDS))
    assert not unknown, f"the test used fact fields that are not in _box_rig.FACT_FIELDS: {unknown}"
    try:
        return factory(**overrides)
    except TypeError as exc:
        raise AssertionError(
            f"T-0033 criterion 1: Facts(**fields) with every field keyword-only and defaulting "
            f"to None — {exc}. Fields: {', '.join(FACT_FIELDS)}"
        ) from None


def evaluate(**overrides):
    preflight = module("preflight")
    checks = attr(preflight, "evaluate")(facts(**overrides))
    assert isinstance(checks, tuple), "criterion 1: evaluate returns a tuple of Check"
    for check in checks:
        assert check.status in STATUSES, (
            f"{check.name}: status {check.status!r} is not PASS/WARN/FAIL"
        )
    return checks


def by_name(checks) -> dict:
    names = [c.name for c in checks]
    dupes = [n for n, k in Counter(names).items() if k > 1]
    assert not dupes, f"criterion 1: one row per check; these repeat: {dupes}"
    missing = sorted(set(REQUIRED_CHECKS) - set(names))
    extra = sorted(set(names) - set(REQUIRED_CHECKS))
    assert not missing, (
        f"criterion 1: every check has a row, always. Missing: {missing} "
        f"(the names are pinned in tests/unit/_box_rig.REQUIRED_CHECKS)"
    )
    assert not extra, f"unexpected rows {extra}; add them to _box_rig.REQUIRED_CHECKS first"
    return {c.name: c for c in checks}


def statuses(checks) -> dict:
    out: dict[str, set[str]] = {s: set() for s in STATUSES}
    for check in checks:
        out[check.status].add(check.name)
    return out


def fixture(name: str) -> str:
    return (BOX / name).read_text()


# --- the box this repo actually runs on, on 2026-09-14 -----------------------------------------

#: `services/versions.lock` with both TBDs filled in — the box as it will be once the human's
#: queue item 2 is done. Used where a test is about something other than the TBDs.
PINNED_LOCK = {
    "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "model_revision": "2ea837236295d617ac27f8c17d61228081932c40",
    "chat_template_revision": "b0a5c1d3e4f60718293a4b5c6d7e8f9012345678",
    "pocket_tts": "3.1.0",
    # T-0111g B4 (re-scope R7): ``collect_lock`` flattens ``[browser]`` of services/versions.lock; the values are
    # that file's pins (2026-10-20).
    "playwright": "1.64.0",
    "chromium": "156.0.8078.4",
}

#: T-0111g B4 (re-scope R7): a box with the browser extra, its pinned Chromium and a display -- the gathered
#: ``browser`` fact on which all three ``browser.*`` rows PASS. An ungathered fact is never a PASS, so
#: ``good_box`` carries this one to stay all-PASS.
BROWSER_PRESENT = {
    "playwright": "1.64.0",
    "chromium": "/home/someone/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
    "display": ":1",
}

#: The STT container's argv, from `tests/fixtures/box/docker_inspect_stt.json` (`--cpu` at the end).
STT_ARGV_CPU = (
    "moshi-server",
    "worker",
    "--config",
    "/config/stt-1b-en_fr.toml",
    "--addr",
    "127.0.0.1",
    "--port",
    "8080",
    "--cpu",
)
#: The same container as the `lean` profile expects it: no `--cpu`.
STT_ARGV_GPU = STT_ARGV_CPU[:-1]

#: The vLLM argv as recorded on 2026-09-14: no `--revision` (the T-0014/T-0015 carve-out).
VLLM_ARGV_NO_REVISION = (
    "vllm",
    "serve",
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "--host",
    "127.0.0.1",
    "--port",
    "8001",
    "--max-model-len",
    "16384",
    "--max-num-seqs",
    "3",
    "--kv-cache-dtype",
    "bfloat16",
    "--attention-backend",
    "TRITON_ATTN",
)
VLLM_ARGV_PINNED = (
    *VLLM_ARGV_NO_REVISION,
    "--revision",
    "2ea837236295d617ac27f8c17d61228081932c40",
)

LOOPBACK_LISTENERS = (
    {"address": "127.0.0.1", "port": 8001, "process": "python3"},
    {"address": "127.0.0.1", "port": 8080, "process": "moshi-server"},
    {"address": "127.0.0.1", "port": 8081, "process": "python"},
)

#: T-0045 criterion 5: the PipeWire **node name** of this box's default capture source — the same
#: node ``tests/unit/_mic_rig.BOX_MIC`` names (read off
#: ``tests/fixtures/pipewire/dump_recorded_base.json``'s ``default.audio.source``), kept here so
#: ``good_box`` can carry the fact and there is exactly one literal. ``wpctl status`` prints the
#: *description* of the same device in its Sources column, which is what ``sources`` below holds.
DEFAULT_SOURCE_NODE = "alsa_input.usb-046d_Logitech_Webcam_C925e_8473F8AF-02.analog-stereo"

# --- T-0058: the cold boot that could not start, in MiB -----------------------------------------
#
# Every figure is read from `logs/vram-coldboot-73a85dd.before.txt` (recorded verbatim into
# `tests/fixtures/box/nvidia_smi_*_coldboot.csv`), `logs/vram-coldboot-73a85dd.csv` and the
# ValueError in `logs/vllm-coldboot-73a85dd.log`, 2026-09-16 22:17, commit 73a85dd, clean tree.
# They are the *provenance* of the T-0058 rows, written here once so a row states the number and
# the reading it came from rather than a digit somebody liked. `docs/tasks/T-0058.md` carries the
# same table with its derivation.

#: `nvidia-smi --query-gpu=memory.total`. NOT the denominator the engine's check uses.
COLDBOOT_TOTAL_MIB = 32607
#: vLLM's own total, from `mem_get_info`, recovered from the ValueError's `27.23/31.33 GiB`.
#: 525 MiB below what `nvidia-smi` calls the total, while the two *free* readings agree exactly.
VLLM_TOTAL_MIB = 32082
COLDBOOT_USED_MIB = 3696
#: `memory.free` as `nvidia-smi` reported it seconds before the launch.
COLDBOOT_FREE_MIB = 28391
#: `init_snapshot.free_memory` as vLLM read it, *after* creating its own CUDA context.
COLDBOOT_FREE_AT_ENGINE_CHECK_MIB = 27882
#: What that context cost on this box: 28391 - 27882. n = 1, and it is an estimate.
MEASURED_INIT_OVERHEAD_MIB = COLDBOOT_FREE_MIB - COLDBOOT_FREE_AT_ENGINE_CHECK_MIB  # 509
#: The conservative rounding a guard should carry (T-0058 implementation notes: round up, do not
#: tune). Being pessimistic costs a spurious WARN; being optimistic costs a boot.
DOCUMENTED_INIT_OVERHEAD_MIB = 512
#: The engine's own steady footprint at KV 4 GiB (T-0014 / T-0048 criterion 11), device-used
#: minus Chrome. A utilisation demanding less than this stops being a guard.
ENGINE_FOOTPRINT_MIB = 27434
#: The recogniser and Chrome's GPU process, from `--query-compute-apps` at the same instant.
RECOGNISER_MIB = 3180
CHROME_MIB = 50
#: vLLM v0.29.0's default when `--gpu-memory-utilization` is absent, which is the state
#: `services/vllm.sh` was in on 2026-09-16. Version-specific (T-0058 ambiguity 5).
VLLM_DEFAULT_UTILIZATION = 0.92
#: What the engine demanded and what it was short by, from the ValueError itself (0.92,
#: 28.83 GiB against 27.23 GiB free).
ENGINE_DEMAND_AT_DEFAULT_MIB = 29522
ENGINE_DEFICIT_MIB = ENGINE_DEMAND_AT_DEFAULT_MIB - COLDBOOT_FREE_AT_ENGINE_CHECK_MIB  # 1640
#: What the engine's own process holds on the 2026-09-14 box description below: the 27 433 MiB
#: used, minus the recogniser and Chrome. A restart releases it before it asks for anything, so
#: it is part of what a restart has to fit into (ruling A12).
ENGINE_PROCESS_MIB = 27433 - RECOGNISER_MIB - CHROME_MIB  # 24203

HEALTHY = {"vllm": None, "stt": None, "tts": None}
TOOLS_PRESENT = {
    "uv": "/usr/bin/uv",
    "git": "/usr/bin/git",
    "node": "/usr/bin/node",
    "chromium": "/usr/bin/chromium",
}


def good_box(**overrides) -> dict:
    """A box on which the first live run can succeed: everything gathered, everything fine.

    Hand-computed expectation: every row PASS except the two the carve-outs make WARN
    (`container.stt_device` only if the argv disagrees with the profile, `container.vllm_argv`
    only if `--revision` is missing) — both are pinned here, so a `good_box()` is all-PASS.

    That all-PASS claim is why ``default_source`` is in here (T-0045 ruling A17): an ungathered
    fact is never a PASS (``_ungathered``'s own discipline), and a box that cannot name its
    default capture source is one where ``uv run astra session --serve-hud`` — the arrival
    checklist's audio command — exits 1. ``tests/unit/test_default_source.py`` pins both halves.
    """
    box = {
        "sounddevice_ok": True,
        "sounddevice_error": None,
        "default_sink": "Family 17h/19h HD Audio Controller Analog Stereo",
        "sources": ("Logitech Webcam C925e Analog Stereo",),
        # T-0045 ruling A17: a good box *knows* which microphone `astra session` would open. Left
        # out, this fixture described a box whose default source was never read — the state in
        # which the arrival checklist's audio command exits 1 — while claiming every row passes.
        "default_source": DEFAULT_SOURCE_NODE,
        "service_health": dict(HEALTHY),
        "served_model_id": PINNED_LOCK["model"],
        "hf_snapshot_revision": PINNED_LOCK["model_revision"],
        "lock": dict(PINNED_LOCK),
        "stt_argv": STT_ARGV_GPU,
        "vllm_argv": VLLM_ARGV_PINNED,
        "profile_stt_backend": "kyutai-gpu",
        "vram_used_mib": 27433,
        "vram_total_mib": 32607,
        # T-0058: the 2026-09-14 recording carries no `memory.free` column, so this fixture uses
        # `total - used`; the rows that turn on the 525 MiB gap between the two denominators use
        # the 2026-09-16 numbers above instead.
        #
        # 5 174 MiB free *while the engine is resident* — and the engine is 24 203 of the 27 433
        # used (the recogniser and Chrome are the rest). What a restart of the engine has to fit
        # into is therefore 5 174 + 24 203 = 29 377 MiB, not 5 174: stopping the engine releases
        # what the engine holds (T-0058 ruling A12; the first implementation compared the demand
        # against free memory the restart would release and warned on every healthy box).
        "vram_free_mib": 32607 - 27433,
        "vram_processes": (
            {"pid": 3380309, "process_name": "moshi-server", "used_mib": RECOGNISER_MIB},
            {"pid": 11419, "process_name": "/opt/google/chrome/chrome", "used_mib": CHROME_MIB},
            # The engine's own process, as the driver names it inside the container: the full
            # argv, so a row is not pinned to one way of recognising it (a matcher keying on
            # `vllm` and one keying on `python3` both find this). A holders list *without* such a
            # process is the "not gathered" case -- the doctor may not guess how much a restart
            # would release.
            {
                "pid": 2201,
                "process_name": "/usr/bin/python3 -m vllm.entrypoints.openai.api_server",
                "used_mib": ENGINE_PROCESS_MIB,
            },
        ),
        # The value `services/vllm.sh` is expected to carry after T-0058 group A. A *box
        # description*, not a reading of the script: the window the script's value must satisfy
        # is asserted from the MiB figures in tests/unit/test_vllm_launch_flags.py.
        "vllm_utilization": 0.86,
        "vllm_utilization_is_default": False,
        "listeners": LOOPBACK_LISTENERS,
        "tools": dict(TOOLS_PRESENT),
        "dirty_paths": (),
        "browser": dict(BROWSER_PRESENT),
        # T-0182 A3: gathered, with no default clip registered (BASE's registry) -> the clips row is PASS. Left out,
        # the row would be WARN "not gathered" and this box would no longer be all-PASS.
        "clips": {},
    }
    box.update(overrides)
    return box


# --- process helpers (criteria 10 and 13) -------------------------------------------------------


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not reachable for our own children
        return True
    return True


def reap(proc: subprocess.Popen | None, pid: int | None = None) -> None:
    """Kill and wait, so no test leaves a 30 s sleeper behind."""
    if proc is not None:
        with_pid = proc.pid
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):  # pragma: no cover
            proc.wait(timeout=5)
        pid = pid if pid is not None else with_pid
    if pid is not None and alive(pid):  # pragma: no cover - the orphan path
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def free_port() -> int:
    """A port nobody is listening on right now (bound then closed)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def json_lines(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
