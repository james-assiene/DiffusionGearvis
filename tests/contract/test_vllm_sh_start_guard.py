"""``services/vllm.sh`` refuses, in about a second, a boot that would die 21 s in (T-0058 A).

On 2026-09-16 at 22:17 the engine was started on a box where the recogniser had moved onto the
GPU. The container pulled up, created a CUDA context, asked the driver how much memory was free,
found 27.23 GiB where its default demanded 28.83, and died — 21 s after launch, before weight
load, before KV reservation, with a `ValueError` at the bottom of a 155-line log. The information
needed to predict that was available before the container started: free device memory, the
utilisation the argv carries, and the list of processes holding the card.

So the launcher asks first. These rows drive that guard by putting a **fake `nvidia-smi`** first
on `PATH` (`tests/fakes/gpu_cli.py`) and reading what the script does with each answer. They are
contract rows rather than unit rows because the behaviour *is* the script's exit status and
streams, which only a subprocess can show — but nothing real starts:

* every row runs under `ASTRA_DRY_RUN=1`, where the script prints the argv it would have used
  instead of handing it to the container runtime;
* a fake `docker` is first on `PATH` in every row and writes a sentinel when invoked, so a lost
  dry-run branch is caught by the fake rather than by the 5090;
* the script is a byte-identical copy in a temporary tree, because the repository keeps no built
  plugin wheel and the launcher refuses without one.

An **agent** may not run this script from a Bash call — it is on the engine-start path and needs
a `/launch` token (`.claude/rules/gpu-services.md`, T-0016 I9). Spawning it from inside pytest
with the fakes above is a different thing, and criterion 5's sentinel row is what keeps it one.

Regime: no GPU, no container runtime, no network; n = 1 per case, no timing claimed. The
`gpu`-marked row at the bottom is the real-service half of the fake (`.claude/rules/tests.md`)
and spends nothing: a status poll.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.fakes import gpu_cli
from tests.unit import _box_rig as rig

SCRIPT = rig.REPO / "services" / "vllm.sh"

#: The readings of 2026-09-16 22:17 (`tests/unit/_box_rig.py` carries the whole table).
FREE_WITH_RECOGNISER_MIB = rig.COLDBOOT_FREE_MIB  # 28 391, and the engine still fits at 0.86
FREE_ON_A_FULL_BOX_MIB = 20000  # short at every value the window allows

#: The row that pins the engine's own CUDA context (ruling A13). 27 800 MiB free is *more* than
#: the 27 591 MiB the argv demands at 0.86 — so a guard that compares the driver's reading
#: straight against the demand proceeds — but the engine spends ~512 MiB creating its context
#: before it looks, leaving 27 288, and the boot then dies 21 s in. Every other fixture in this
#: module (28 391 and 20 000) is far enough from the demand that dropping the subtraction changes
#: nothing, which is how it stayed unpinned through 27 rows. It discriminates for any committed
#: utilisation up to 27800/32082 = 0.866, and the committed value is 0.86.
FREE_SHORT_ONLY_BY_THE_ENGINES_OWN_CONTEXT_MIB = 27800

#: What the argv may demand, in MiB, for any utilisation criterion 1 permits (0.855-0.869 of
#: vLLM's 32 082 MiB total). The refusal has to state a demand inside this band: a number outside
#: it means the script computed against the wrong denominator or the wrong flag.
DEMAND_BAND = (round(0.8551 * rig.VLLM_TOTAL_MIB), round(0.8691 * rig.VLLM_TOTAL_MIB))

#: The first words of the server invocation, used to tell "the launcher composed its argv" from
#: "the launcher refused". Not a start: under ASTRA_DRY_RUN=1 the argv is printed, never run.
ARGV_MARKER = "vllm serve"


@pytest.fixture
def launcher(tmp_path):
    """A copy of ``services/vllm.sh`` in its own tree, with a bin directory ready for fakes."""
    script = gpu_cli.launcher_tree(tmp_path / "tree")
    fakes = tmp_path / "bin"
    sentinel = tmp_path / "docker-was-invoked"
    gpu_cli.install_docker_sentinel(fakes, sentinel)
    smi_log = tmp_path / "nvidia-smi-calls.log"

    class Launcher:
        path = script
        bin_dir = fakes
        docker_sentinel = sentinel
        smi_calls = smi_log

        def run(self, *, mode="ok", free_mib=FREE_WITH_RECOGNISER_MIB, env=None):
            gpu_cli.install_nvidia_smi(fakes, mode=mode, free_mib=free_mib, log=smi_log)
            return gpu_cli.run_launcher(script, path_dirs=[fakes], env=env)

    return Launcher()


def _numbers(text: str) -> list[int]:
    return [int(hit.replace(",", "")) for hit in re.findall(r"\d[\d,]*", text)]


def test_the_launcher_parses_under_bash_n():
    """A syntax error in the engine's launcher is a FAILED row with no log to read."""
    done = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"bash -n services/vllm.sh:\n{done.stderr}"


def test_a_box_with_the_recogniser_resident_proceeds(launcher):
    """Criterion 3: 28 391 MiB free is enough at the committed value, so the script proceeds.

    This is the box of 2026-09-16 — recogniser up, 3 180 MiB gone — and the point of the whole
    task is that the engine *can* boot there once the argv stops demanding 0.92 of the device.
    At 0.86 it demands 27 591 MiB and will see ~27 879 when it looks: 288 MiB to spare. A guard
    that refused here would have replaced a 21 s failure with a permanent one.
    """
    done = launcher.run(free_mib=FREE_WITH_RECOGNISER_MIB)
    assert done.returncode == 0, (
        f"criterion 3: {FREE_WITH_RECOGNISER_MIB} MiB free clears the demand, so the launcher "
        f"proceeds. It exited {done.returncode}.\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    assert ARGV_MARKER in done.stdout, (
        f"criterion 5: with a permissive answer the argv is composed as before; stdout was\n"
        f"{done.stdout}"
    )
    assert "--gpu-memory-utilization" in done.stdout, (
        "criterion 1: …carrying the startup gate, which is the flag whose absence killed the "
        f"2026-09-16 boot. The argv was:\n{done.stdout}"
    )
    assert launcher.smi_calls.exists(), (
        "criterion 3: the guard asks the driver how much memory is free *before* the container "
        "starts; nvidia-smi was never called, so the script cannot have checked anything"
    )
    asked = launcher.smi_calls.read_text()
    assert "memory.free" in asked, (
        "criterion 3: …and it asks for memory.free. free is not total-used on this device (the "
        f"recording differs by 520 MiB). The guard asked:\n{asked}"
    )


def test_a_box_without_room_refuses_and_names_the_numbers(launcher):
    """Criterion 3: 20 000 MiB free is short at every permitted value, so the launcher refuses.

    The refusal is the whole deliverable: exit non-zero, on stderr, before the container starts,
    saying what was free, what the argv will demand, by how much it is short, who is holding the
    card, and what to do about it. A refusal that says "not enough VRAM" leaves the human exactly
    where the 21 s traceback did.
    """
    done = launcher.run(free_mib=FREE_ON_A_FULL_BOX_MIB)
    report = done.stdout + done.stderr
    assert done.returncode != 0, (
        f"criterion 3: {FREE_ON_A_FULL_BOX_MIB} MiB free is ~7 GiB short of any demand the "
        f"window allows; the launcher must refuse. It exited 0.\nstdout:\n{done.stdout}\n"
        f"stderr:\n{done.stderr}"
    )
    assert done.stderr.strip(), (
        f"criterion 3: the refusal goes to stderr (it is what `astra up` captures and quotes); "
        f"stdout was\n{done.stdout}"
    )
    assert ARGV_MARKER not in done.stdout, (
        "criterion 3: …and it happens *before* the argv is composed and handed over — a refusal "
        f"after that point is a 21 s failure with extra text. stdout was\n{done.stdout}"
    )
    numbers = _numbers(report)
    assert FREE_ON_A_FULL_BOX_MIB in numbers, (
        f"criterion 3: the refusal states the free memory it read; it said\n{report}"
    )
    low, high = DEMAND_BAND
    assert any(low <= number <= high for number in numbers), (
        f"criterion 3: …and what the argv will demand, which for a permitted utilisation is "
        f"{low}-{high} MiB (0.855-0.869 of vLLM's own 32 082 MiB total). A demand outside that "
        f"band means the wrong denominator or the wrong flag. It said\n{report}"
    )
    deficit = low - FREE_ON_A_FULL_BOX_MIB
    assert any(abs(number - deficit) <= (high - low) + 600 for number in numbers), (
        f"criterion 3: …and the deficit (about {deficit} MiB here), so the human knows how much "
        f"has to be freed rather than only that something must. It said\n{report}"
    )
    assert "moshi-server" in report, (
        "criterion 3: …and which processes hold the memory (--query-compute-apps), because that "
        f"is what the human acts on. It said\n{report}"
    )
    lowered = report.lower()
    assert "astra_stt_cpu" in lowered, (
        f"criterion 3: …and the documented fallback, ASTRA_STT_CPU=1; it said\n{report}"
    )
    assert "order" in lowered or "first" in lowered, (
        "criterion 3: …and the remedy that actually works today: stop the recogniser, boot the "
        f"engine, start the recogniser second. It said\n{report}"
    )


def test_the_refusal_lists_the_processes_it_read_from_the_driver(launcher):
    """Criterion 3 / ruling A17: the holder list is *read*, not a sentence about the recogniser.

    The refusal's fixed remedy text names the recogniser whatever the card holds — it is the
    thing to stop — so asserting that word proves nothing about the process list: deleting the
    line that prints the holders leaves it satisfied. What cannot be faked by the remedy text is
    a figure the guard could only have got from the driver. The fake reports the recogniser as
    **pid 3380309 holding 3180 MiB**, and neither number appears anywhere in the script.

    Why the list matters at all: "not enough VRAM" sends a human off to ask the driver who is
    holding the card, which is the 20 s the refusal exists to save. The 2026-09-16 boot died with
    Chrome's GPU process and the recogniser resident, and only one of those was worth stopping.
    """
    done = launcher.run(free_mib=FREE_ON_A_FULL_BOX_MIB)
    report = done.stdout + done.stderr
    assert done.returncode != 0, (
        f"the box is short, so this must be a refusal; it exited {done.returncode}\n{report}"
    )
    holder_pid, holder_mib = gpu_cli.RECOGNISER_PID, rig.RECOGNISER_MIB
    numbers = _numbers(report)
    assert holder_pid in numbers or holder_mib in numbers, (
        f"ruling A17: the refusal must carry what it read from the driver — the recogniser's pid "
        f"({holder_pid}) or the {holder_mib} MiB it holds. Naming the process in the remedy text "
        f"is the script talking about itself. It said\n{report}"
    )
    assert launcher.smi_calls.exists(), "the guard never asked the driver anything at all"
    asked = launcher.smi_calls.read_text()
    assert "--query-compute-apps" in asked, (
        "ruling A17: …and it has to ask for the compute apps to know them. The guard asked:\n"
        f"{asked}"
    )


def test_a_box_short_only_by_the_engines_own_context_still_refuses(launcher):
    """Criterion 3 / ruling A13: the 512 MiB subtraction is the guard, on the tight boxes.

    Hand-computed at the committed value: 27 800 MiB free, minus the ~512 MiB of CUDA context
    vLLM creates before it reads free memory, is **27 288 MiB** against the **27 591 MiB** the
    argv demands — 303 MiB short, so the launcher refuses. A guard that compares the driver's
    27 800 straight against 27 591 proceeds, the container starts, and 21 s later the engine
    says the same thing with a traceback. This is the box-kills-your-boot direction on the
    launcher side, and it is the only fixture in this module that can see it: at 28 391 MiB the
    subtraction is slack and at 20 000 MiB everything refuses.

    Measured direction of the mutation: deleting `- ENGINE_INIT_OVERHEAD_MIB` from the
    launcher's prediction leaves the other 27 rows green and reddens this one.
    """
    done = launcher.run(free_mib=FREE_SHORT_ONLY_BY_THE_ENGINES_OWN_CONTEXT_MIB)
    report = done.stdout + done.stderr
    assert done.returncode != 0, (
        f"ruling A13: {FREE_SHORT_ONLY_BY_THE_ENGINES_OWN_CONTEXT_MIB} MiB free looks like enough "
        "and is not: the engine's own CUDA context takes ~512 MiB before it checks, leaving "
        "27 288 against 27 591 demanded. The launcher proceeded into a boot that dies.\n"
        f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    assert ARGV_MARKER not in done.stdout, (
        f"…and it refuses before the argv is composed; stdout was\n{done.stdout}"
    )
    numbers = _numbers(report)
    assert FREE_SHORT_ONLY_BY_THE_ENGINES_OWN_CONTEXT_MIB in numbers, (
        f"ruling A13: the refusal states the free reading it started from; it said\n{report}"
    )
    low, high = DEMAND_BAND
    assert any(low <= number <= high for number in numbers), (
        f"…and the demand it compared against ({low}-{high} MiB for a permitted utilisation); it "
        f"said\n{report}"
    )


def test_the_refusal_never_reaches_the_container_runtime(launcher):
    """Criterion 5, adversarial: the check is a *pre*-flight, not an early exit from the boot.

    A fake `docker` sits first on `PATH` and records its argv in a sentinel file. After a
    refusal that file must not exist — a guard that refuses after the container has already been
    started has saved nothing, and on this box the difference is 21 s of GPU and a
    half-initialised engine process.
    """
    done = launcher.run(free_mib=FREE_ON_A_FULL_BOX_MIB)
    assert done.returncode != 0, (
        f"the box is short, so this must be a refusal for the sentinel to mean anything; it "
        f"exited {done.returncode}\n{done.stdout}{done.stderr}"
    )
    assert not launcher.docker_sentinel.exists(), (
        "criterion 5: the launcher reached the container runtime after refusing — the sentinel "
        f"records {launcher.docker_sentinel.read_text()!r}"
    )


def test_a_permissive_answer_reaches_no_container_runtime_either(launcher):
    """Criterion 5's other half: the dry run composes the argv and starts nothing.

    Without this row the sentinel above could be satisfied by a script that never reaches the
    runtime at all, and `astra up`'s classification of an engine start as a routine service
    start would rest on nothing.
    """
    done = launcher.run(free_mib=FREE_WITH_RECOGNISER_MIB)
    assert done.returncode == 0 and ARGV_MARKER in done.stdout, (
        f"the permissive case must proceed for this row to mean anything; it exited "
        f"{done.returncode}\n{done.stdout}{done.stderr}"
    )
    assert not launcher.docker_sentinel.exists(), (
        "criterion 5: ASTRA_DRY_RUN=1 prints the argv instead of handing it over; the fake was "
        f"invoked with {launcher.docker_sentinel.read_text()!r}"
    )


@pytest.mark.parametrize(
    ("mode", "what"),
    [
        ("missing", "nvidia-smi is not on PATH at all (a laptop, a CI container)"),
        ("exit1", "nvidia-smi exits 1 (NVML/driver version mismatch after an update)"),
        ("empty", "nvidia-smi prints nothing"),
        ("na", "nvidia-smi answers [N/A]"),
        ("units", "nvidia-smi answers in GiB, a unit the guard did not expect"),
    ],
)
def test_the_guard_fails_open(launcher, mode: str, what: str):
    """Criterion 4: every unknown proceeds, with one warning line. The guard is a speed bump.

    This is the most likely way for this change to do damage. The arithmetic is checkable and
    will be checked; a guard that *refuses* because the driver query answered in an unfamiliar
    shape stops the engine on every box that is not this one, and it does so at the moment
    somebody is trying to start a demo. The engine's own check remains the authority — it costs
    21 s, which is exactly the price of being wrong in this direction, and it is the right price
    to pay for an unreadable status query.

    ``set -euo pipefail`` is the trap: an `nvidia-smi` that exits 1 inside a command
    substitution takes the whole script with it unless the guard says otherwise.
    """
    done = launcher.run(mode=mode, free_mib=FREE_WITH_RECOGNISER_MIB)
    assert done.returncode == 0, (
        f"criterion 4: {what} — the guard must fail *open* and let the engine's own check "
        f"decide. It exited {done.returncode}.\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    assert ARGV_MARKER in done.stdout, (
        f"criterion 4: …and the boot proceeds; the argv was never composed.\nstdout:\n{done.stdout}"
    )
    warnings = [line for line in done.stderr.splitlines() if line.strip()]
    assert len(warnings) == 1, (
        f"criterion 4: one warning line, so a human sees that the check was skipped without the "
        f"log filling with it. stderr was:\n{done.stderr}"
    )
    assert "nvidia-smi" in warnings[0].lower() or "vram" in warnings[0].lower(), (
        f"criterion 4: …naming what could not be read; it said {warnings[0]!r}"
    )


def test_the_guard_has_no_dry_run_bypass(launcher):
    """Criterion 4, negative: a dry run that skips the guard cannot test the guard.

    Every row in this module sets ``ASTRA_DRY_RUN=1``, so if the guard were behind
    ``[ "$ASTRA_DRY_RUN" = 1 ] || check`` the refusal rows above would pass a script that never
    refuses on a real start. The refusal must fire on this path too — which is what the rows
    above observe, and what this row states outright.
    """
    done = launcher.run(free_mib=FREE_ON_A_FULL_BOX_MIB, env={"ASTRA_DRY_RUN": "1"})
    assert done.returncode != 0, (
        "criterion 4, negative: the guard has no ASTRA_DRY_RUN branch; it refused nothing on the "
        f"path every test in this module uses.\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )


def test_a_non_zero_launcher_exit_is_what_up_already_observes(launcher):
    """The composition with T-0049 criterion 4, asserted as shape only.

    ``astra up`` starts ``services/vllm.sh`` detached with stdout and stderr redirected into
    ``logs/vllm.log`` and watches the exit status inside its settle window. So a refusal needs
    exactly two properties for `up` to print a FAILED row quoting it instead of a 21 s mystery:
    a non-zero exit, and the explanation on a stream the redirect captures. Nothing is imported
    from ``cli`` or ``supervisor`` here and nothing is started: T-0049 owns those files, and this
    task's claim on them is one of shape.
    """
    done = launcher.run(free_mib=FREE_ON_A_FULL_BOX_MIB)
    assert done.returncode != 0, "a refusal exits non-zero"
    captured = done.stdout + done.stderr
    assert "free" in captured.lower(), (
        f"the explanation must be in the captured streams, not only in the exit status; the "
        f"launcher printed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )


@pytest.mark.gpu
def test_the_real_nvidia_smi_answers_in_the_shape_the_fake_imitates():
    """The real-service half of `tests/fakes/gpu_cli.py` (`.claude/rules/tests.md`).

    Two status polls, no GPU time (`.claude/rules/gpu-services.md`), asserting only the shape the
    fake and the parsers depend on: a header line naming the queried fields, values carrying
    ``MiB``, and a compute-apps table whose last column is the memory. If a driver update changes
    that, the fake becomes a lie and this row says so before the rows above pass for the wrong
    reason.
    """
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on this box")
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    header, row = memory.splitlines()[:2]
    assert header.replace(" ", "").startswith(
        "memory.total[MiB],memory.used[MiB],memory.free[MiB]"
    ), f"the fake writes this header; the real one said {header!r}"
    assert row.count(",") == 2 and "MiB" in row, f"the real row was {row!r}"
    values = _numbers(row)
    assert values[0] > values[1] and values[0] > values[2], f"total is the largest; got {values}"
    assert values[1] + values[2] <= values[0], (
        "criterion 8: used + free may be *less* than total (520 MiB unaccounted on this box on "
        f"2026-09-16) but never more; got {values}"
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    apps_header = apps.splitlines()[0].replace(" ", "")
    assert apps_header == "pid,process_name,used_gpu_memory[MiB]", (
        f"the fake and tests/fixtures/box/nvidia_smi_compute_apps_coldboot.csv write this header; "
        f"the real one said {apps.splitlines()[0]!r}"
    )
