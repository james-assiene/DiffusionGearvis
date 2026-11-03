"""``astra doctor``: collect → describe → evaluate → render (T-0033 group A).

The verdicts are a pure function of a box description. :func:`gather` runs the collectors, each
of which is allowed to fail into "not gathered" (``None``); :func:`evaluate` turns the resulting
:class:`Facts` into one :class:`Check` per row, always — a doctor that drops the row it could not
gather hides exactly the failure the row exists for. ``evaluate`` touches nothing: no clock, no
socket, no subprocess.

Severity discipline (T-0033): **FAIL = the first live run cannot succeed. WARN = a fact that
should be pinned but does not block.** An ungathered fact is never a PASS.

Nothing here spends GPU time: the health probes are ``supervisor``'s bare status ``GET``s, the
container facts come from ``docker inspect``, and the VRAM figure from an ``nvidia-smi`` query
(``.claude/rules/gpu-services.md``: status polls are free). Fix lines are printed for the human;
this module never runs them, and never runs a privileged command.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_astra.config import LOOPBACK_PREFIXES, Profile
from local_astra.supervisor import REPO_ROOT, check_service

#: docs/02 §7 and ``.claude/rules/gpu-services.md``: this much VRAM must stay free for the
#: unprofiled sampler transient.
VRAM_FLOOR_MIB = 3 * 1024

# --- T-0058: can the *engine* still be started on this box? --------------------------------------
#
# A different question from ``vram.free``, which asks whether a running engine has room for the
# sampler transient. The engine's own startup check (``vllm/v1/worker/utils.py::request_memory``)
# is ``requested = ceil(total * utilisation); if init_snapshot.free_memory < requested: raise`` --
# it happens before any weight is loaded, and on 2026-09-16 22:17 it killed the boot 21 s in:
#
#     ValueError: Free memory on device cuda:0 (27.23/31.33 GiB) on startup is less than desired
#     GPU memory utilization (0.92, 28.83 GiB).     -- logs/vllm-coldboot-73a85dd.log
#
# Both constants below are **box- and version-specific measurements**, not properties of CUDA
# (CLAUDE.md: never a number without its regime). Their regime: RTX 5090 on this box, driver
# 595.84, container vllm/vllm-openai:v0.29.0, commit 73a85dd, 2026-09-16 22:17, n = 1.

#: vLLM's own denominator, from ``torch.cuda.mem_get_info``, recovered from the ValueError's
#: ``27.23/31.33 GiB``. ``nvidia-smi`` calls the same device 32607 MiB (31.84 GiB): the totals
#: disagree by ~525 MiB while the two *free* readings agree exactly. Using nvidia-smi's total
#: over-states the demand by 452 MiB at 0.86 -- enough to refuse a box that boots.
VLLM_TOTAL_MIB = 32082

#: What vLLM spends on its own CUDA context *before* it takes the free-memory snapshot: 509 MiB
#: on the recorded boot (28391 free by nvidia-smi seconds earlier, 27882 at the engine's check),
#: rounded up to 512 and deliberately not tuned. An estimate from n = 1: being ~100 MiB
#: pessimistic costs a spurious WARN, being optimistic costs a boot.
ENGINE_INIT_OVERHEAD_MIB = 512

#: vLLM's default when ``--gpu-memory-utilization`` is absent -- the state ``services/vllm.sh``
#: was in on 2026-09-16, and the reason the engine demanded 0.92 of the device. It belongs to the
#: pinned container: when that pin moves, this and the launcher's value are re-derived.
VLLM_DEFAULT_UTILIZATION = 0.92
VLLM_DEFAULT_UTILIZATION_VERSION = "v0.29.0"

#: Said the same way by the launcher's own refusal (``services/vllm.sh``) and by this row: the
#: engine has to be started while the card is empty, and the recogniser put back afterwards.
ENGINE_BOOT_ORDER_FIX = (
    "stop the recogniser (Kyutai STT, the moshi-server container), start the engine first and "
    "the recogniser second -- that boot order is the only way the two fit on this card today; "
    "`uv run astra down` then `uv run astra up` starts them in that order"
)

#: The measured cost of running Kyutai STT on the CPU, with its whole regime (CLAUDE.md: never a
#: number without one). Printed by the ``container.stt_device`` WARN.
CPU_STT_COST = (
    "step time 100-108 ms per 80 ms of audio, RTF ~ 1.3 "
    "(stt-1b-en_fr, batch_size 2, 2026-09-14 audit of this box)"
)

#: Flags the engine container must carry whatever else changes (``.claude/rules/gpu-services.md``).
REQUIRED_VLLM_FLAGS = ("--max-num-seqs", "--kv-cache-dtype", "--attention-backend")

#: How long the engine has to answer its model listing.
MODEL_LISTING_TIMEOUT_S = 2.0

#: How long ``pw-dump`` has to print the graph. A local, read-only query; the same ceiling
#: ``conductor/audio_device.py`` gives it, where it has never taken more than a few ms.
PW_DUMP_TIMEOUT_S = 5.0

#: Where a ``TBD`` pin comes from, for the WARN's fix line (the human's queue item 2).
LOCK_PIN_COMMANDS = {
    "pocket_tts": "uvx pocket-tts --version (or read the dist-info of the installed release)",
    "chat_template_revision": "pin the Gemma 4 chat template commit beside model_revision",
    "llama_cpp": "pin the build tag when the talker profile is first exercised (X9)",
}

#: The containers the doctor inspects, by the name they are started with.
STT_CONTAINER = "astra-stt"

#: What the docker socket says when the login session predates the docker group — the normal
#: state on this box, and the reason ``services/stt.sh`` re-runs itself under ``sg docker -c``.
DOCKER_DENIED = "permission denied"

#: What to do about a box with no capture source, said once: the ``audio.source`` row prints it
#: as its fix and ``astra session`` prints it when it refuses to start deaf (T-0045 ruling A3).
NO_CAPTURE_SOURCE_FIX = (
    "plug a microphone (or a webcam with one), or start the session with --no-mic"
)

_TOOLS = ("uv", "git", "node", "chromium")
_CHROMIUM_BINARIES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


@dataclass(frozen=True)
class Check:
    """One row of the doctor's table: what was asked, what was found, what would fix it."""

    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True, kw_only=True)
class Facts:
    """A box description. Every field defaults to ``None``, which means *not gathered*."""

    sounddevice_ok: bool | None = None
    sounddevice_error: str | None = None
    default_sink: str | None = None
    #: T-0045 criterion 5: the PipeWire **node name** of the box's default capture source -- the
    #: microphone ``astra session`` opens when no ``--mic`` says otherwise. A node name, never the
    #: description ``wpctl status`` prints in its Sources column: ``open_device`` refuses that one.
    default_source: str | None = None
    sources: tuple[str, ...] | None = None
    service_health: dict[str, str | None] | None = None
    served_model_id: str | None = None
    hf_snapshot_revision: str | None = None
    lock: dict[str, str] | None = None
    stt_argv: tuple[str, ...] | None = None
    vllm_argv: tuple[str, ...] | None = None
    profile_stt_backend: str | None = None
    vram_used_mib: int | None = None
    vram_total_mib: int | None = None
    #: T-0058: ``nvidia-smi``'s own ``memory.free`` column, which on this device is **not**
    #: ``total - used`` (32607, 3696 and 28391 on 2026-09-16: a 520 MiB gap, generous in the
    #: direction that costs a boot). The engine's startup check compares against the free
    #: reading, so ``vram.engine_start`` may not derive it by subtraction.
    vram_free_mib: int | None = None
    #: One dict per row of ``--query-compute-apps=pid,process_name,used_gpu_memory``: keys
    #: ``pid``, ``process_name``, ``used_mib``. An empty tuple is a fact (nothing holds the
    #: device); ``None`` is "not gathered".
    vram_processes: tuple[dict[str, Any], ...] | None = None
    #: The ``--gpu-memory-utilization`` the engine's argv will carry, read out of
    #: ``services/vllm.sh``: the flag's value, or vLLM's own default when the flag is absent.
    vllm_utilization: float | None = None
    #: True when the figure above is that *default* rather than a value the script passes -- the
    #: state of the box on 2026-09-16, and something the row has to say out loud, because the
    #: default belongs to the pinned container version.
    vllm_utilization_is_default: bool | None = None
    listeners: tuple[dict[str, Any], ...] | None = None
    tools: dict[str, str | None] | None = None
    dirty_paths: tuple[str, ...] | None = None
    #: Why the container argv could not be read, when it could not (ruling A18). It carries the
    #: collector's exception text into the row's ``detail`` so the human is not told "not
    #: gathered" with no reason (pinned by ``tests/unit/test_doctor_verdicts.py``, ruling A21).
    container_error: str | None = None
    #: T-0111g B4: ``live``'s browser prerequisites, from ``collect_browser``: ``playwright`` (the installed
    #: distribution's version, ``None`` when not importable), ``chromium`` (the pinned executable's path, ``None``
    #: when absent) and ``display`` (``DISPLAY`` or ``WAYLAND_DISPLAY``, ``None`` when neither is set).
    browser: dict[str, str | None] | None = None
    #: T-0182 A3: each default-enabled clip kind's ``reflex.clips.status`` in the default data dir (``current``,
    #: ``missing`` or ``stale``), from ``collect_clips``. ``{}`` means no default clip is registered.
    clips: dict[str, str] | None = None


# --- the pure verdicts ---------------------------------------------------------------------------


def _ungathered(name: str, what: str, how: str) -> Check:
    """The row for a fact nobody could gather. Never a PASS: unknown is not evidence."""
    return Check(name, "WARN", f"not gathered: {what}", how)


def _audio_checks(facts: Facts) -> list[Check]:
    out: list[Check] = []
    if facts.sounddevice_ok is None:
        out.append(
            _ungathered(
                "audio.sounddevice",
                "the sounddevice import was not attempted",
                'uv run python -c "import sounddevice; sounddevice.query_devices()"',
            )
        )
    elif facts.sounddevice_ok:
        out.append(
            Check("audio.sounddevice", "PASS", "import sounddevice + query_devices() answered")
        )
    else:
        out.append(
            Check(
                "audio.sounddevice",
                "FAIL",
                f"sounddevice is unusable, so `astra session` exits 1: {facts.sounddevice_error}",
                "apt install libportaudio2 (then re-run; this is printed for you, never run here)",
            )
        )

    if facts.default_sink is None:
        out.append(
            _ungathered("audio.sink", "no default sink was read", "wpctl status (PipeWire session)")
        )
    elif "dummy" in facts.default_sink.lower():
        out.append(
            Check(
                "audio.sink",
                "FAIL",
                f"the default sink is {facts.default_sink!r}: nothing would be audible",
                "wpctl set-default <id of a real sink>, from `wpctl status`",
            )
        )
    else:
        out.append(Check("audio.sink", "PASS", f"default sink {facts.default_sink!r}"))

    if facts.sources is None:
        out.append(
            _ungathered(
                "audio.source", "no capture source list was read", "wpctl status (PipeWire session)"
            )
        )
    elif not facts.sources:
        out.append(
            Check(
                "audio.source",
                "FAIL",
                "no audio capture source: nothing would be heard",
                NO_CAPTURE_SOURCE_FIX,
            )
        )
    else:
        listed = f"capture sources: {', '.join(facts.sources)}"
        if facts.default_source is None:
            # T-0045 criterion 5 and ruling A17: the list alone is what the human had on
            # 2026-09-15, and it did not say which node a session would open. WARN, not PASS:
            # this module's own discipline is that an ungathered fact is never a PASS
            # (`_ungathered`), and here the consequence is concrete -- the arrival checklist's
            # `uv run astra session --serve-hud` exits 1 on this box (ruling A3), so a green row
            # would send a human into a failure the doctor said nothing about. Not a FAIL,
            # because the box can still be told which microphone to use.
            out.append(
                Check(
                    "audio.source",
                    "WARN",
                    f"{listed}; the default capture source was not read, so `astra session` "
                    f"has no microphone to open by itself",
                    "name one: astra session --mic <PipeWire node name> (pw-dump lists them)",
                )
            )
        else:
            out.append(
                Check(
                    "audio.source",
                    "PASS",
                    f"default capture source {facts.default_source}"
                    f"{_source_kind(facts.default_source)} -- the node `astra session` opens "
                    f"when no --mic says otherwise; {listed}",
                )
            )
    return out


def _source_kind(node: str) -> str:
    """Say when the default source is the webcam's microphone (criterion 5: the only mic here).

    Read off the node name PipeWire itself composed, which carries the USB product string; a box
    whose name says nothing gets no claim about the device.
    """
    lowered = node.lower()
    if "webcam" in lowered or "camera" in lowered:
        return " (a webcam's microphone)"
    return ""


def _health_checks(facts: Facts) -> list[Check]:
    out: list[Check] = []
    for service in ("vllm", "stt", "tts"):
        name = f"health.{service}"
        if facts.service_health is None or service not in facts.service_health:
            out.append(
                _ungathered(name, f"{service} was not probed", "uv run astra status --profile lean")
            )
            continue
        reason = facts.service_health[service]
        if reason is None:
            out.append(Check(name, "PASS", f"{service} answered its own health endpoint"))
        else:
            out.append(
                Check(
                    name,
                    "FAIL",
                    f"{service} is not healthy: {reason}",
                    f"read logs/{service}.log, then start it again (`uv run astra up`)",
                )
            )
    return out


def _model_checks(facts: Facts) -> list[Check]:
    out: list[Check] = []
    lock = facts.lock or {}
    pinned_model = lock.get("model")
    if facts.served_model_id is None or pinned_model is None:
        out.append(
            _ungathered(
                "model.served",
                "the served model id or the lock's model is missing",
                "start the engine, then re-run; the lock is services/versions.lock",
            )
        )
    elif facts.served_model_id == pinned_model:
        out.append(Check("model.served", "PASS", f"serving the pinned {pinned_model}"))
    else:
        out.append(
            Check(
                "model.served",
                "FAIL",
                f"serving {facts.served_model_id!r}, but services/versions.lock pins "
                f"{pinned_model!r}",
                "restart the engine on the pinned model (services/vllm.sh reads the lock)",
            )
        )

    pinned_revision = lock.get("model_revision")
    if facts.hf_snapshot_revision is None or pinned_revision in (None, "TBD"):
        out.append(
            _ungathered(
                "model.revision",
                "the HF cache snapshot ref or the pinned revision is missing",
                "download the pinned revision with `hf download` (see the README's One-time "
                "setup), or fill model_revision in the lock",
            )
        )
    elif facts.hf_snapshot_revision == pinned_revision:
        out.append(Check("model.revision", "PASS", f"snapshot {pinned_revision} is the pinned one"))
    else:
        out.append(
            Check(
                "model.revision",
                "FAIL",
                f"the HF cache snapshot is {facts.hf_snapshot_revision!r}, but the lock pins "
                f"{pinned_revision!r}",
                "download the pinned revision with the `hf download` commands in the README's "
                '"One-time setup" before any measurement',
            )
        )
    return out


def _lock_check(facts: Facts) -> Check:
    if facts.lock is None:
        return _ungathered(
            "lock.pins", "services/versions.lock was not read", "check services/versions.lock"
        )
    tbd = sorted(key for key, value in facts.lock.items() if value == "TBD")
    if not tbd:
        return Check("lock.pins", "PASS", f"{len(facts.lock)} pinned lines, none TBD")
    fixes = "; ".join(f"{key}: {LOCK_PIN_COMMANDS.get(key, 'pin it')}" for key in tbd)
    return Check(
        "lock.pins",
        "WARN",
        f"services/versions.lock still says TBD for: {', '.join(tbd)} — an unpinned line does "
        "not stop a run, it stops a number from being comparable",
        f"{fixes} (the lock is a human edit: hooks deny it to agents)",
    )


def _container_checks(facts: Facts) -> list[Check]:
    out: list[Check] = []
    if facts.stt_argv is None or facts.profile_stt_backend is None:
        why = facts.container_error or "the STT container argv or the profile's stt_backend"
        out.append(
            _ungathered(
                "container.stt_device",
                why,
                f"docker inspect {STT_CONTAINER} (the argv is the fact, not the environment); "
                f"if the socket refuses: sg docker -c 'docker inspect {STT_CONTAINER}'",
            )
        )
    else:
        on_cpu = "--cpu" in facts.stt_argv
        wants_cpu = facts.profile_stt_backend.endswith("-cpu")
        device = "cpu" if on_cpu else "gpu"
        if on_cpu == wants_cpu:
            out.append(
                Check(
                    "container.stt_device",
                    "PASS",
                    f"the STT container runs on {device}, which is what the profile "
                    f"({facts.profile_stt_backend}) asks for",
                )
            )
        else:
            out.append(
                Check(
                    "container.stt_device",
                    "WARN",
                    f"the STT container argv says {device} while the profile says "
                    f"{facts.profile_stt_backend}; cost of CPU STT: {CPU_STT_COST}",
                    "human queue item 1 in docs/tasks/T-0033.md: accept CPU STT and change the "
                    "profile, or free VRAM and restart the container without --cpu",
                )
            )

    if facts.vllm_argv is None:
        out.append(
            _ungathered(
                "container.vllm_argv",
                facts.container_error or "the engine container was not inspected",
                "docker inspect the engine container (docker ps) and compare to services/vllm.sh; "
                "if the socket refuses: sg docker -c 'docker ps'",
            )
        )
    else:
        missing = [flag for flag in REQUIRED_VLLM_FLAGS if flag not in facts.vllm_argv]
        if missing:
            out.append(
                Check(
                    "container.vllm_argv",
                    "WARN",
                    f"the engine argv is missing {', '.join(missing)} (services/vllm.sh is the "
                    "executable form of the flags in .claude/rules/gpu-services.md)",
                    "restart the engine through services/vllm.sh",
                )
            )
        elif "--revision" not in facts.vllm_argv:
            out.append(
                Check(
                    "container.vllm_argv",
                    "WARN",
                    "the engine argv carries no --revision, so the served weights are whatever "
                    "the cache holds",
                    'add --revision "$MODEL_REVISION" to services/vllm.sh — the carve-out: that '
                    "file belongs to T-0014/T-0015, so it is a follow-up task, not this one",
                )
            )
        else:
            out.append(
                Check(
                    "container.vllm_argv", "PASS", "the engine argv pins --revision and the flags"
                )
            )
    return out


def _vram_check(facts: Facts) -> Check:
    if facts.vram_used_mib is None or facts.vram_total_mib is None:
        return Check(
            "vram.free",
            "WARN",
            "no nvidia-smi answer (laptop?), so free VRAM is unknown",
            "on the 5090 box: nvidia-smi --query-gpu=memory.used,memory.total --format=csv",
        )
    free = facts.vram_total_mib - facts.vram_used_mib
    detail = (
        f"{facts.vram_used_mib} MiB used of {facts.vram_total_mib} MiB, {free} MiB free "
        f"against the {VRAM_FLOOR_MIB} MiB sampler-transient floor (docs/02 §7)"
    )
    if free < VRAM_FLOOR_MIB:
        return Check(
            "vram.free",
            "FAIL",
            detail,
            "stop a service holding VRAM (`uv run astra down`) or shrink the engine footprint",
        )
    return Check("vram.free", "PASS", detail)


#: How much of a process name the row prints. The parser keeps the whole argv (it is the fact),
#: but Chrome's GPU process carries ~700 characters of it and this row is a line a human reads.
_HOLDER_NAME_CHARS = 60


def _holders(processes: tuple[dict[str, Any], ...]) -> str:
    """Who holds the device right now — the part of the row a human can act on."""
    if not processes:
        return "no process holds device memory"
    named = []
    for row in processes:
        name = str(row.get("process_name", ""))
        if len(name) > _HOLDER_NAME_CHARS:
            name = name[:_HOLDER_NAME_CHARS] + "…"
        named.append(f"{name} (pid {row.get('pid')}) {row.get('used_mib')} MiB")
    return f"holding the device: {', '.join(named)}"


#: How ``nvidia-smi --query-compute-apps`` names the engine **on this box**: vLLM serves from a
#: python process inside the container and the driver reports the executable, not the argv. Read
#: off the 2026-09-14 recording, where the engine held 24 203 MiB as ``python3`` (pid 2201) while
#: the same pid listened on 127.0.0.1:8001. Box-, container- and version-specific: a driver that
#: names it otherwise makes the row say "not gathered", which is the honest answer -- matching a
#: wider pattern would let some unrelated python script stand in for the engine, and the number
#: it would contribute decides a verdict.
ENGINE_PROCESS_NAMES = ("python3", "python")


def _engine_held_mib(processes: tuple[dict[str, Any], ...]) -> int | None:
    """What the engine's own process holds, or ``None`` when no holder looks like the engine.

    Ruling A12: this is the memory a restart releases *before* the new process takes its
    free-memory snapshot, so it is the difference between "the engine is up, and a restart would
    die" (almost never true) and "the engine is up, and a restart would not fit" (the thing the
    row is for). ``None`` is never treated as zero: zero is a claim about the box, and the wrong
    one -- it is what produced the permanent WARN this ruling corrects.
    """
    for row in processes:
        # The column holds the whole command line; the executable is its first word, and the
        # driver may print it as a path (`/usr/bin/python3`). An empty name matches nothing.
        words = str(row.get("process_name", "")).split()
        name = Path(words[0]).name if words else ""
        if name in ENGINE_PROCESS_NAMES:
            held = row.get("used_mib")
            if isinstance(held, int):
                return held
    return None


def _engine_start_check(facts: Facts) -> Check:
    """``vram.engine_start``: would the engine clear its own startup check on this box?

    The arithmetic, and every reason it is not the obvious one:

    * free comes from the driver's ``memory.free`` column, never ``total - used`` (they differ by
      520 MiB here, and the difference is room that is not there);
    * the demand is ``ceil(utilisation x 32082)``, computed against **vLLM's** total, not
      ``nvidia-smi``'s 32607 -- at 0.86 that choice is 452 MiB, enough to fail a box that boots;
    * vLLM takes its snapshot *after* creating its own CUDA context, so ~512 MiB is subtracted
      from free before the comparison. Forgetting it passes a box that then dies 21 s in, which
      is the failure this row exists to prevent.

    Severity (T-0033): a box whose engine is down and cannot start is a **FAIL** -- the first live
    run cannot succeed. A box whose engine is *up* while the arithmetic says it could not be
    started again where it stands is a **WARN**: it is running, so it is not broken, but a human
    about to restart it in place is about to lose it for nothing.

    **What a healthy engine changes (ruling A12).** A restart releases the engine's own memory
    *before* the new process takes its snapshot, so on a healthy box the figure to compare is
    free **plus what the engine itself holds**, not free alone. Comparing free alone makes the
    row a permanent WARN -- a running engine holds ~27.4 GiB of a 32.6 GiB card, so the demand
    never fits beside it -- whose text ("a restart would die") is false and whose fix contradicts
    the thing the launcher's flag fixes. Measured on this card: engine up alone, 5 123 MiB free,
    and the restart boots; the honest prediction of that is 5 123 + what the engine holds. When
    health says the engine is up and no engine process is among the holders, that is a fact
    missing, and the row is ungathered rather than a guess in either direction.
    """
    name = "vram.engine_start"
    if (
        facts.vram_free_mib is None
        or facts.vram_processes is None
        or facts.vllm_utilization is None
        or facts.service_health is None
    ):
        return _ungathered(
            name,
            "the driver's free memory, the processes holding the device, the engine's "
            "utilisation or the engine's health is missing, and unknown is not evidence",
            "nvidia-smi --query-gpu=memory.free --format=csv and nvidia-smi "
            "--query-compute-apps=pid,process_name,used_gpu_memory --format=csv (both read-only), "
            "then re-run",
        )

    healthy = facts.service_health.get("vllm", "not probed") is None
    engine_held_mib = _engine_held_mib(facts.vram_processes) if healthy else None
    if healthy and engine_held_mib is None:
        return _ungathered(
            name,
            f"the engine answers its health endpoint, but no engine process is among the "
            f"processes holding the device, so what a restart would release is unknown "
            f"({_holders(facts.vram_processes)})",
            "check how nvidia-smi --query-compute-apps names the engine on this box and add it "
            f"to preflight.ENGINE_PROCESS_NAMES (it is {', '.join(ENGINE_PROCESS_NAMES)} today)",
        )

    utilisation = facts.vllm_utilization
    demand_mib = math.ceil(utilisation * VLLM_TOTAL_MIB)
    # Ruling A12: a restart gives the engine's own memory back before the new process looks, so
    # on a healthy box that memory counts as available; on a box where the engine is down there
    # is nothing to give back and free is free.
    available_mib = facts.vram_free_mib + (engine_held_mib or 0)
    predicted_free_mib = available_mib - ENGINE_INIT_OVERHEAD_MIB
    deficit_mib = demand_mib - predicted_free_mib
    if facts.vllm_utilization_is_default:
        source = (
            f"{utilisation:g} (vLLM {VLLM_DEFAULT_UTILIZATION_VERSION}'s default: "
            f"services/vllm.sh passes no --gpu-memory-utilization, which is what killed the "
            f"2026-09-16 boot; a container pin change re-derives it)"
        )
    else:
        source = f"{utilisation:g} (the value services/vllm.sh passes)"
    released = (
        ""
        if engine_held_mib is None
        else (
            f", plus the {engine_held_mib} MiB the running engine holds and gives back when it "
            f"stops, is {available_mib} MiB the restart can use"
        )
    )
    arithmetic = (
        f"{facts.vram_free_mib} MiB free now, from the driver's own memory.free column{released}; "
        f"the engine spends ~{ENGINE_INIT_OVERHEAD_MIB} MiB on its CUDA context before it looks "
        f"(measured once, 2026-09-16, logs/vram-coldboot-73a85dd.before.txt), so "
        f"~{predicted_free_mib} MiB will be free at its startup check, against {demand_mib} MiB "
        f"demanded at --gpu-memory-utilization {source} of vLLM's own {VLLM_TOTAL_MIB} MiB total "
        f"(mem_get_info, 31.33 GiB -- nvidia-smi calls the same device 32607 MiB / 31.84 GiB and "
        f"the two disagree by 525 MiB). {_holders(facts.vram_processes)}"
    )
    short = f"{deficit_mib} MiB ({deficit_mib / 1024:.2f} GiB) short"

    if predicted_free_mib >= demand_mib:
        return Check(name, "PASS", f"the engine clears its own startup check here: {arithmetic}")
    if healthy:
        return Check(
            name,
            "WARN",
            f"the engine is running, but it could not be started again where it stands: even "
            f"with its own memory given back, a restart would die at its own startup check, "
            f"{short}. {arithmetic}",
            f"do not restart the engine in place -- {ENGINE_BOOT_ORDER_FIX}",
        )
    return Check(
        name,
        "FAIL",
        f"the engine cannot start on this box: it would die at its own startup check, before "
        f"loading a weight, {short}. {arithmetic}",
        ENGINE_BOOT_ORDER_FIX,
    )


def _is_loopback(address: str) -> bool:
    host = address.split("%", 1)[0].strip("[]")
    return host.startswith("127.") or host in ("::1", "localhost")


def _loopback_check(facts: Facts) -> Check:
    if facts.listeners is None:
        return _ungathered("net.loopback", "no listening sockets were read", "ss -tlnp (read-only)")
    offenders = [row for row in facts.listeners if not _is_loopback(str(row.get("address", "")))]
    if offenders:
        named = ", ".join(
            f"{row.get('address')}:{row.get('port')}"
            + (f" ({row['process']})" if row.get("process") else "")
            for row in offenders
        )
        return Check(
            "net.loopback",
            "FAIL",
            f"listening off loopback: {named} (docs/02 §9, docs/03 E8, CLAUDE.md)",
            "bind the service to 127.0.0.1 — every services/*.sh already passes --host 127.0.0.1",
        )
    return Check(
        "net.loopback", "PASS", f"{len(facts.listeners)} listening sockets, all on loopback"
    )


def _tool_checks(facts: Facts) -> list[Check]:
    out: list[Check] = []
    fixes = {
        "uv": "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git": "apt install git",
        "node": "install node (dev only: node --test tests/hud)",
        "chromium": "apt install chromium (there is no live run without a page)",
    }
    for tool in _TOOLS:
        name = f"tool.{tool}"
        if facts.tools is None:
            out.append(_ungathered(name, f"{tool} was not looked for", f"which {tool}"))
            continue
        where = facts.tools.get(tool)
        if where:
            out.append(Check(name, "PASS", f"{tool} at {where}"))
        elif tool == "node":
            out.append(Check(name, "WARN", "node is missing (a dev prerequisite)", fixes[tool]))
        else:
            out.append(Check(name, "FAIL", f"{tool} is missing", fixes[tool]))
    return out


def _tree_check(facts: Facts) -> Check:
    if facts.dirty_paths is None:
        return _ungathered("git.tree", "the tree was not inspected", "git status --porcelain")
    if not facts.dirty_paths:
        return Check("git.tree", "PASS", "clean tree")
    return Check(
        "git.tree",
        "WARN",
        "dirty tree: " + ", ".join(facts.dirty_paths) + " — a service started against a tree an "
        "agent is editing makes every later number unattributable",
        "commit or stash before starting services (the same reason guard_bash.sh refuses)",
    )


#: T-0111g B2/B4: how to get ``live``'s browser (the P1 prerequisite of T-0111c).
BROWSER_FIX = "uv sync --all-extras && uv run playwright install chromium"


def _browser_checks(facts: Facts) -> list[Check]:
    """T-0111g B4: ``live``'s three prerequisites. WARN at worst, since tools are off by default."""
    names = ("browser.playwright", "browser.chromium", "browser.display")
    if facts.browser is None:
        return [
            _ungathered(names[0], "playwright was not looked for", BROWSER_FIX),
            _ungathered(names[1], "the pinned Chromium was not looked for", BROWSER_FIX),
            _ungathered(
                names[2], "the display was not looked for", "echo $DISPLAY $WAYLAND_DISPLAY"
            ),
        ]
    out: list[Check] = []
    installed = facts.browser.get("playwright")
    pinned = (facts.lock or {}).get("playwright")
    if installed is None:
        out.append(
            Check(
                names[0],
                "WARN",
                "playwright is not installed (only tools = live needs it)",
                BROWSER_FIX,
            )
        )
    elif pinned is None:
        out.append(
            Check(
                names[0],
                "WARN",
                f"playwright {installed} installed; services/versions.lock [browser] pins no version",
                "pin playwright under [browser] in services/versions.lock",
            )
        )
    elif installed != pinned:
        out.append(
            Check(
                names[0],
                "WARN",
                f"playwright {installed} installed; services/versions.lock [browser] pins {pinned}",
                f"{BROWSER_FIX} (uv.lock pins playwright {pinned})",
            )
        )
    else:
        out.append(Check(names[0], "PASS", f"playwright {installed}, as pinned"))
    chromium = facts.browser.get("chromium")
    if chromium:
        out.append(Check(names[1], "PASS", f"pinned Chromium at {chromium}"))
    else:
        out.append(
            Check(
                names[1],
                "WARN",
                "the pinned Chromium is not installed (only tools = live needs it): headed live launches "
                "chromium-<rev>/chrome-linux64/chrome, headless live chromium_headless_shell-<rev>/"
                "chrome-headless-shell-linux64/chrome-headless-shell, and at least one is missing",
                BROWSER_FIX,
            )
        )
    display = facts.browser.get("display")
    if display:
        out.append(Check(names[2], "PASS", f"a headed browser can open on {display}"))
    else:
        out.append(
            Check(
                names[2],
                "WARN",
                "neither DISPLAY nor WAYLAND_DISPLAY is set: a headed browser cannot open",
                "run with a display, or use astra session --tools live --browser headless",
            )
        )
    return out


#: T-0182 A3: how to get the clip assets.
CLIPS_FIX = (
    "start the TTS service (services/tts.sh) and run uv run python scripts/render_clips.py "
    "(it writes clips/ under the default data dir)"
)


def _clips_check(facts: Facts) -> Check:
    """T-0182 A3: WARN at worst, since a missing clip only means silence."""
    if facts.clips is None:
        return _ungathered("clips", "the clip assets were not looked for", CLIPS_FIX)
    if not facts.clips:
        return Check("clips", "PASS", "no default clips registered")
    bad = {kind: state for kind, state in sorted(facts.clips.items()) if state != "current"}
    if not bad:
        return Check(
            "clips", "PASS", "every default clip is current: " + ", ".join(sorted(facts.clips))
        )
    named = ", ".join(f"{kind} ({state})" for kind, state in bad.items())
    return Check(
        "clips", "WARN", f"clip assets not current: {named}; they play as silence", CLIPS_FIX
    )


def evaluate(facts: Facts) -> tuple[Check, ...]:
    """The whole table, in a fixed order, from the facts alone. Pure: no I/O, no clock."""
    checks: list[Check] = []
    checks += _audio_checks(facts)
    checks += _health_checks(facts)
    checks += _model_checks(facts)
    checks.append(_lock_check(facts))
    checks += _container_checks(facts)
    checks.append(_vram_check(facts))
    checks.append(_engine_start_check(facts))
    checks.append(_loopback_check(facts))
    checks += _tool_checks(facts)
    checks.append(_tree_check(facts))
    checks += _browser_checks(facts)
    checks.append(_clips_check(facts))
    return tuple(checks)


# --- the parsers (pure text/JSON, tested against tests/fixtures/box/) ----------------------------

_WPCTL_ENTRY = re.compile(r"^\s*[│├└─\s]*(\*?)\s*(\d+)\.\s+(.*?)\s*(?:\[[^]]*\])?\s*$")


def parse_wpctl_status(text: str) -> dict[str, Any]:
    """``wpctl status`` → the default sink and the *audio* capture sources.

    The Video section lists the webcam as a source too; a parser that reads it would call a box
    with no microphone "fine".
    """
    audio = text.split("\nVideo", 1)[0]
    start = audio.find("\nAudio")
    audio = audio[start:] if start >= 0 else audio
    section: str | None = None
    default_sink: str | None = None
    sources: list[str] = []
    for line in audio.splitlines():
        stripped = line.strip(" │├└─")
        if stripped.startswith("Sinks:"):
            section = "sinks"
            continue
        if stripped.startswith("Sources:"):
            section = "sources"
            continue
        if stripped.endswith(":") and not stripped[0].isdigit():
            section = None if not stripped.startswith("*") else section
            continue
        match = _WPCTL_ENTRY.match(line)
        if match is None or section is None:
            continue
        is_default, _id, name = match.group(1), match.group(2), match.group(3).strip()
        if not name:
            continue
        if section == "sinks" and is_default:
            default_sink = name
        elif section == "sources":
            sources.append(name)
    return {"default_sink": default_sink, "sources": tuple(sources)}


def parse_default_source(dump: object) -> str | None:
    """``pw-dump`` → the **node name** of the default capture source, or ``None``.

    The fact lives on the ``default`` Metadata object, as ``default.audio.source`` with a
    ``{"name": …}`` value: that name is what ``open_device(source=…)`` and ``pw-dump`` call a
    node. ``wpctl status``'s Sources column prints *descriptions* instead ("Logitech Webcam C925e
    Analog Stereo"), and handing one of those to ``open_device`` is refused as an unknown node
    (T-0035 criterion 4) — which would make every default session exit 1 (T-0045).

    Anything else in the graph — no metadata, another shape, no default set — is ``None``: not
    gathered, never a guess.
    """
    if not isinstance(dump, list):
        return None
    for entry in dump:
        if not isinstance(entry, dict) or not str(entry.get("type", "")).endswith(":Metadata"):
            continue
        props = entry.get("props")
        if not isinstance(props, dict) or props.get("metadata.name") != "default":
            continue
        for item in entry.get("metadata") or ():
            if not isinstance(item, dict) or item.get("key") != "default.audio.source":
                continue
            value = item.get("value")
            name = value.get("name") if isinstance(value, dict) else None
            if isinstance(name, str) and name:
                return name
    return None


def parse_nvidia_smi_csv(text: str) -> dict[str, int | None]:
    """``--query-gpu=memory.used,memory.total --format=csv`` (header included, units in cells)."""
    used: int | None = None
    total: int | None = None
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2:
            continue
        numbers = [re.match(r"^(\d+)", cell) for cell in cells[:2]]
        if not all(numbers):
            continue
        used = int(numbers[0].group(1))  # type: ignore[union-attr]
        total = int(numbers[1].group(1))  # type: ignore[union-attr]
        break
    return {"vram_used_mib": used, "vram_total_mib": total}


#: ``--query-gpu`` prints the fields in the order they were *asked for*, so the columns are read
#: by header name. A positional parser silently swaps free with total the day somebody reorders
#: the query — and reading free from the wrong column is the error that costs a boot.
_SMI_MEMORY_FIELDS = {
    "memory.total": "vram_total_mib",
    "memory.used": "vram_used_mib",
    "memory.free": "vram_free_mib",
}


def parse_nvidia_smi_memory(text: str) -> dict[str, int | None]:
    """``--query-gpu=memory.total,memory.used,memory.free --format=csv`` → the three figures.

    Read by header name (see :data:`_SMI_MEMORY_FIELDS`), and a column that was not asked for —
    or that answered in a shape other than a plain MiB integer — stays ``None`` rather than being
    derived from the others: on this device ``memory.free`` is **not** ``total - used``.
    """
    out: dict[str, int | None] = dict.fromkeys(_SMI_MEMORY_FIELDS.values())
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return out
    header = [cell.split("[")[0].strip() for cell in lines[0].split(",")]
    values = [cell.strip() for cell in lines[1].split(",")]
    for field, cell in zip(header, values, strict=False):
        key = _SMI_MEMORY_FIELDS.get(field)
        if key is None:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*MiB)?", cell)
        out[key] = int(match.group(1)) if match else None
    return out


def parse_nvidia_smi_compute_apps(text: str) -> tuple[dict[str, Any], ...]:
    """``--query-compute-apps=pid,process_name,used_gpu_memory --format=csv`` → one row each.

    The process name is whatever lies between the first comma and the last one, kept verbatim:
    ``nvidia-smi`` does not quote it and a Chrome argv routinely carries ``--disable-features=A,B``,
    so a parser that reads field 1 as the name and field 2 as the memory reports a *flag* as the
    megabytes a process holds. A line that does not start with a pid and end with a memory figure
    (the header, an ``[N/A]``) is not a compute app.
    """
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        pid_cell, comma, rest = line.partition(",")
        if not comma:
            continue
        name, comma, used_cell = rest.rpartition(",")
        if not comma:
            continue
        used = re.fullmatch(r"(\d+)(?:\s*MiB)?", used_cell.strip())
        if not pid_cell.strip().isdigit() or used is None:
            continue
        rows.append(
            {
                "pid": int(pid_cell.strip()),
                "process_name": name.strip(),
                "used_mib": int(used.group(1)),
            }
        )
    return tuple(rows)


def parse_vllm_utilization(text: str) -> float | None:
    """``services/vllm.sh`` → the ``--gpu-memory-utilization`` its argv carries, or ``None``.

    ``None`` means *the flag is absent* — the state of 2026-09-16, where vLLM's own default
    applied — which is not the same as "the script could not be read"; the caller turns the first
    into that default and the second into "not gathered". Comment lines are stripped first: a
    commented-out flag is not passed to vLLM, and reading one is how a doctor tells a human their
    box is fine when it is not.
    """
    executable = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    match = re.search(r"--gpu-memory-utilization(?:=|\s+)([0-9]*\.?[0-9]+)", executable)
    return float(match.group(1)) if match else None


def parse_docker_inspect(text: str) -> dict[str, Any]:
    """``docker inspect <container>`` → name, image and the **argv**, which is the fact.

    The argv is ``Path`` + ``Args``; when that is a shell ``-c`` command (the engine container is
    started that way) the script is tokenised too, so the flags survive into the comparison.
    """
    data = json.loads(text)
    entry = data[0] if isinstance(data, list) else data
    argv = [str(entry.get("Path", ""))] + [str(part) for part in entry.get("Args", [])]
    argv = [part for part in argv if part]
    if len(argv) >= 3 and Path(argv[0]).name in ("sh", "bash", "dash") and argv[1] == "-c":
        with contextlib.suppress(ValueError):  # an unbalanced quote leaves the raw script
            argv = argv[:2] + shlex.split(argv[2])
    config = entry.get("Config") or {}
    return {
        "name": str(entry.get("Name", "")).lstrip("/"),
        "image": str(config.get("Image") or entry.get("Image") or ""),
        "argv": tuple(argv),
    }


def parse_ss_tlnp(text: str) -> tuple[dict[str, Any], ...]:
    """``ss -tlnp`` → one row per LISTEN socket, read by **local** address.

    Every row carries ``0.0.0.0:*`` in its *peer* column, so grepping the output for ``0.0.0.0``
    calls a loopback-only box non-loopback. The local address is the third-from-last field group.
    """
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != "LISTEN":
            continue
        local = fields[3]
        address, _, port = local.rpartition(":")
        if not address or not port.isdigit():
            continue
        process = ""
        match = re.search(r'users:\(\("([^"]+)"', line)
        if match:
            process = match.group(1)
        rows.append({"address": address.strip("[]"), "port": int(port), "process": process})
    return tuple(rows)


# --- the collectors (thin; each may fail into "not gathered") ------------------------------------


def _run(command: list[str], *, timeout_s: float = 5.0) -> str:
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    if done.returncode != 0:
        raise OSError(f"{command[0]} exited {done.returncode}: {done.stderr.strip()[:200]}")
    return done.stdout


def collect_audio() -> dict[str, Any]:
    """Exactly the call ``astra session`` fails on. Opens no stream, plays and records nothing."""
    try:
        import sounddevice

        sounddevice.query_devices()
    except Exception as exc:
        return {"sounddevice_ok": False, "sounddevice_error": f"{type(exc).__name__}: {exc}"}
    return {"sounddevice_ok": True, "sounddevice_error": None}


def collect_pipewire() -> dict[str, Any]:
    return parse_wpctl_status(_run(["wpctl", "status"]))


def default_source() -> str | None:
    """The microphone ``astra session`` opens when no ``--mic`` names one (T-0045 criterion 1).

    The PipeWire node name of the box's default capture source, read from the graph — never
    guessed from a list, never the description ``wpctl status`` prints. A box whose graph cannot
    be read (no PipeWire session, no ``pw-dump``, a timeout) has no known default: the answer is
    ``None`` and the caller says so. This is on the way up of every session, so it never raises.
    """
    try:
        return parse_default_source(json.loads(_run(["pw-dump"], timeout_s=PW_DUMP_TIMEOUT_S)))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # OSError: pw-dump is missing or exited non-zero (``_run``); ValueError: what it printed
        # is not JSON; TimeoutExpired: the graph never arrived. None of these is a default source.
        return None


def collect_default_source() -> dict[str, Any]:
    return {"default_source": default_source()}


def collect_health(profile: Profile) -> dict[str, Any]:
    services = profile.services
    health: dict[str, str | None] = {}
    for name, url in (("vllm", services.vllm), ("stt", services.stt), ("tts", services.tts)):
        health[name] = check_service(name, url).reason
    return {"service_health": health}


def collect_served_model(profile: Profile) -> dict[str, Any]:
    """``GET {services.vllm}/models``: the served id, asked of the engine itself (criterion 4).

    A listing, not an inference call: no request body, so it spends nothing
    (``.claude/rules/gpu-services.md``). The URL is built from the configured endpoint and
    re-validated, never typed here.
    """
    url = profile.services.vllm.rstrip("/") + "/models"
    if not url.startswith(LOOPBACK_PREFIXES):
        raise ValueError(f"refusing to ask non-loopback {url!r} (docs/02 §9)")
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=MODEL_LISTING_TIMEOUT_S) as response:
        listing = json.loads(response.read(4096).decode("utf-8", "replace"))
    entries = listing.get("data") or []
    return {"served_model_id": str(entries[0]["id"]) if entries else None}


def collect_hf_snapshot(lock: dict[str, str] | None = None) -> dict[str, Any]:
    """The HF cache's snapshot ref for the pinned repo — the revision actually on disk."""
    pins = lock if lock is not None else collect_lock()["lock"]
    repo = pins.get("model")
    if not repo:
        raise OSError("the lock has no model to look up")
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = cache / ("models--" + repo.replace("/", "--"))
    ref = folder / "refs" / "main"
    if ref.is_file():
        return {"hf_snapshot_revision": ref.read_text().strip()}
    snapshots = sorted((folder / "snapshots").iterdir()) if (folder / "snapshots").is_dir() else []
    if not snapshots:
        raise OSError(f"no HF cache snapshot under {folder}")
    return {"hf_snapshot_revision": snapshots[-1].name}


def collect_lock(path: Path | None = None) -> dict[str, Any]:
    """``services/versions.lock``, flattened to ``{key: value}``. Read only, never written."""
    lock_path = path if path is not None else REPO_ROOT / "services" / "versions.lock"
    with lock_path.open("rb") as handle:
        data = tomllib.load(handle)
    flat: dict[str, str] = {}
    for section in data.values():
        if isinstance(section, dict):
            for key, value in section.items():
                flat[key] = str(value)
    return {"lock": flat}


def _docker(args: list[str]) -> str:
    """``docker <args>``, retried under ``sg docker -c`` when the socket says permission denied.

    The login session on this box predates the docker group, so a plain ``docker inspect`` fails
    with ``permission denied`` while the very same command under ``sg docker`` works — which is
    exactly the fallback ``services/stt.sh`` documents. Without the retry the doctor reports the
    two container rows as "not gathered" on the one box it exists to describe (ruling A18).
    """
    command = ["docker", *args]
    try:
        return _run(command)
    except OSError as exc:
        if DOCKER_DENIED not in str(exc).lower():
            raise
        quoted = shlex.join(command)
        return _run(["sg", "docker", "-c", quoted])


def collect_containers() -> dict[str, Any]:
    """The argv of the running containers — the fact the environment variable only hints at."""
    out: dict[str, Any] = {"stt_argv": None, "vllm_argv": None, "container_error": None}
    errors: list[str] = []
    try:
        out["stt_argv"] = parse_docker_inspect(_docker(["inspect", STT_CONTAINER]))["argv"]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f"{STT_CONTAINER}: {exc}")
    image = None
    try:
        image = collect_lock()["lock"].get("vllm_container")
    except (OSError, ValueError) as exc:
        errors.append(f"the lock does not say which image the engine runs: {exc}")
    if image:
        try:
            ids = _docker(["ps", "--filter", f"ancestor={image}", "-q"]).split()
            if ids:
                out["vllm_argv"] = parse_docker_inspect(_docker(["inspect", ids[0]]))["argv"]
            else:
                errors.append(f"no running container from {image}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{image}: {exc}")
    if errors:
        out["container_error"] = "; ".join(errors)
    return out


def collect_vram() -> dict[str, Any]:
    return parse_nvidia_smi_csv(
        _run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])
    )


def collect_vram_free() -> dict[str, Any]:
    """The driver's own ``memory.free`` column (T-0058), asked for on its own.

    A second status poll rather than an extra column on ``collect_vram``: that collector feeds the
    ``vram.free`` row, which computes free as ``total - used`` and over-reads by 520 MiB on this
    device. Correcting it is a later task (it changes a row this one may not touch), so the two
    figures are gathered separately and each is labelled with the reading it came from.
    """
    text = _run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv"])
    return {"vram_free_mib": parse_nvidia_smi_memory(text)["vram_free_mib"]}


def collect_vram_processes() -> dict[str, Any]:
    """Which processes hold the device — the part of an engine-start refusal a human acts on."""
    text = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]
    )
    return {"vram_processes": parse_nvidia_smi_compute_apps(text)}


def collect_vllm_utilization(path: Path | None = None) -> dict[str, Any]:
    """The startup gate the engine's argv will carry, read out of ``services/vllm.sh``.

    The script is the executable form of the flags, so it is the fact — not the lock, not a
    running container. When it passes no flag, vLLM's own default applies and the row says which
    version that default belongs to. A script that cannot be read at all raises, and both fields
    stay ``None``: "not gathered", which is never a PASS.
    """
    script = path if path is not None else REPO_ROOT / "services" / "vllm.sh"
    value = parse_vllm_utilization(script.read_text())
    if value is None:
        return {
            "vllm_utilization": VLLM_DEFAULT_UTILIZATION,
            "vllm_utilization_is_default": True,
        }
    return {"vllm_utilization": value, "vllm_utilization_is_default": False}


def collect_listeners() -> dict[str, Any]:
    return {"listeners": parse_ss_tlnp(_run(["ss", "-tlnp"]))}


def collect_tools() -> dict[str, Any]:
    tools: dict[str, str | None] = {}
    for tool in _TOOLS:
        if tool == "chromium":
            found = next(
                (shutil.which(name) for name in _CHROMIUM_BINARIES if shutil.which(name)), None
            )
        else:
            found = shutil.which(tool)
        tools[tool] = found
    return {"tools": tools}


def _playwright_package() -> Path | None:
    """Where the ``playwright`` package lives, found without importing it; ``None`` when it is not importable."""
    try:
        spec = importlib.util.find_spec("playwright")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def _pinned_executables(package: Path) -> dict[str, Path] | None:
    """The two executables Playwright launches, by mode: ``headed`` (Chrome for Testing,
    ``chromium-<rev>/chrome-linux64/chrome``) and ``headless`` (``chromium_headless_shell-<rev>/
    chrome-headless-shell-linux64/chrome-headless-shell``), each revision read off the package's ``browsers.json``,
    under the browsers directory Playwright uses: ``$PLAYWRIGHT_BROWSERS_PATH``, ``0`` meaning inside the package
    (``driver/package/.local-browsers``), else the user cache. Only paths are built; nothing is read but the JSON."""
    try:
        listing = json.loads((package / "driver" / "package" / "browsers.json").read_text())
        revisions = {
            str(b["name"]): str(b["revision"])
            for b in listing.get("browsers", [])
            if isinstance(b, dict) and "name" in b and "revision" in b
        }
        headed, headless = revisions["chromium"], revisions["chromium-headless-shell"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured == "0":
        root = package / "driver" / "package" / ".local-browsers"
    elif configured:
        root = Path(configured)
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        root = Path(cache) / "ms-playwright"
    return {
        "headed": root / f"chromium-{headed}" / "chrome-linux64" / "chrome",
        "headless": root
        / f"chromium_headless_shell-{headless}"
        / "chrome-headless-shell-linux64"
        / "chrome-headless-shell",
    }


def pinned_executable(browser: str) -> str | None:
    """T-0111g B2: the executable ``live`` launches for ``browser`` (``headed`` or ``headless``) when it is present,
    else ``None``. Reads paths only."""
    package = _playwright_package()
    found = _pinned_executables(package) if package is not None else None
    exe = found.get(browser) if found is not None else None
    return str(exe) if exe is not None and exe.is_file() else None


def _pinned_chromium(package: Path) -> str | None:
    """The doctor's ``chromium`` fact: both pinned executables named, when both are present (headed live launches
    the first, headless the second); ``None`` when either is missing."""
    found = _pinned_executables(package)
    if found is None or not all(p.is_file() for p in found.values()):
        return None
    return f"{found['headed']} and the headless shell {found['headless']}"


def collect_browser() -> dict[str, Any]:
    """T-0111g B4: ``live``'s prerequisites from the filesystem and the environment only. Playwright is never
    imported and Chromium never runs."""
    package = _playwright_package()
    version: str | None = None
    chromium: str | None = None
    if package is not None:
        try:
            version = importlib.metadata.version("playwright")
        except importlib.metadata.PackageNotFoundError:
            version = None
        chromium = _pinned_chromium(package)
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or None
    return {"browser": {"playwright": version, "chromium": chromium, "display": display}}


def collect_tree() -> dict[str, Any]:
    lines = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]).splitlines()
    return {"dirty_paths": tuple(line[3:].strip() for line in lines if line.strip())}


def collect_clips(*, filler_clips: bool = False) -> dict[str, Any]:
    """T-0182 A3: each default-enabled clip kind's status in ``memory.store.DEFAULT_DIR`` (read now), against the
    lock's ``pocket_tts`` pin; B6: ``filler_clips`` (the profile's) adds ``ack``/``hmm``. Imported lazily so
    ``doctor`` does not load the session at import."""
    from local_astra.memory import store
    from local_astra.reflex import clips

    registry = clips.registered()
    pin = clips.tts_pin()
    data_dir = store.DEFAULT_DIR
    return {
        "clips": {
            kind: clips.status(data_dir, kind, registry[kind], pin)
            for kind in sorted(clips.default_enabled(registry, filler_clips=filler_clips))
        }
    }


#: ``gather`` calls these by module attribute, so a test can make each one fail in turn.
COLLECTORS = (
    "collect_audio",
    "collect_pipewire",
    "collect_default_source",
    "collect_health",
    "collect_served_model",
    "collect_lock",
    "collect_hf_snapshot",
    "collect_containers",
    "collect_vram",
    "collect_vram_free",
    "collect_vram_processes",
    "collect_vllm_utilization",
    "collect_listeners",
    "collect_tools",
    "collect_tree",
    "collect_browser",
    "collect_clips",
)


def gather_regime(profile: Profile) -> Facts:
    """Only the facts ``cli.regime_line`` states: the container argv and the served model id.

    ``astra session`` is not a doctor — it must not inspect the whole box on the way up (ruling
    A19) — but the two things its regime line claims have to be measured or it is printing a
    guess. A collector that cannot answer leaves its field ``None``, which prints ``unknown``.
    """
    this = sys.modules[__name__]
    values: dict[str, Any] = {}
    for name in ("collect_containers", "collect_served_model"):
        collector = getattr(this, name)
        try:
            values.update(collector(profile) if name == "collect_served_model" else collector())
        except Exception:  # "not gathered" is a fact too
            continue
    known = set(Facts.__dataclass_fields__)
    return Facts(**{key: value for key, value in values.items() if key in known})


def gather(profile: Profile) -> Facts:
    """Run every collector. One that raises leaves its fields ``None`` — and its row WARN."""
    this = sys.modules[__name__]
    values: dict[str, Any] = {"profile_stt_backend": profile.stt_backend}
    for name in COLLECTORS:
        collector = getattr(this, name)
        try:
            if name in ("collect_health", "collect_served_model"):
                values.update(collector(profile))
            elif name == "collect_hf_snapshot":
                values.update(collector(values.get("lock")))
            elif (
                name == "collect_clips"
            ):  # T-0182 B6 (B-R2): the profile's filler_clips, by keyword
                values.update(collector(filler_clips=profile.filler_clips))
            else:
                values.update(collector())
        except Exception:
            continue
    known = set(Facts.__dataclass_fields__)
    return Facts(**{key: value for key, value in values.items() if key in known})


def render(checks: tuple[Check, ...]) -> list[str]:
    """One line per check: status, name, detail, and the fix when there is one."""
    width = max((len(check.name) for check in checks), default=0)
    lines = []
    for check in checks:
        line = f"{check.status:<4}  {check.name:<{width}}  {check.detail}"
        if check.fix and check.status != "PASS":
            line += f"\n      {' ' * width}  fix: {check.fix}"
        lines.append(line)
    return lines


def render_json(checks: tuple[Check, ...]) -> list[str]:
    """One JSON object per check, stable keys, nothing else."""
    return [
        json.dumps({"name": c.name, "status": c.status, "detail": c.detail, "fix": c.fix})
        for c in checks
    ]
