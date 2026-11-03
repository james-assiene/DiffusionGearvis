"""T-0015 criteria 4-5: the two things in ``services/vllm.sh`` that silently break the boot.

Pure text tests -- no GPU, no docker, no import of the plugin (it only imports inside the vLLM
container). Both failures they guard against are invisible at review time and expensive at boot
time, so each assertion carries the mechanism in its message.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VLLM_SH = REPO / "services" / "vllm.sh"
PLUGIN_INIT = REPO / "plugin" / "astra_vllm_plugin" / "astra_vllm_plugin" / "__init__.py"

# Criterion 5. Long, because the failure looks like pointless strictness without it.
WHY_THE_FLAGS = """
Why these four flags are load-bearing (docs/tasks/T-0015 criterion 5):

vLLM's MODELS_CONFIG_MAP (vllm/model_executor/models/config.py) is keyed by the LITERAL
architecture string. services/vllm.sh overrides `architectures` to the plugin's class name, so
that lookup misses and DiffusionGemmaModelForBlockDiffusionConfig.verify_and_update_config()
NEVER RUNS. Today that changes nothing only because every branch it would have applied is
already falsified by one of these four explicit flags:

  --attention-backend        backend selection and the use_non_causal branch only fire when
                             attention_config.backend is None
  --diffusion-config         suppresses auto-creation of the diffusion config
  --max-num-seqs             the default would be raised to 8 (and >4 is forbidden anyway:
                             diffusion state scales seqs x canvas x 262K vocab)
  --override-generation-config
                             suppresses the max_new_tokens injection

Drop any one of them and a plugin-on boot silently diverges from a plugin-off boot -- same
model, different sampler/attention/scheduler settings, no error and no log line. That is why
this test fails on a missing flag even though the server would start happily.

If a flag genuinely has to go, the honest fix is to stop overriding `architectures` (register
the class under the stock name) or to prove the corresponding branch is inert on the pinned
vLLM and record that in docs/tasks/. Deleting the flag and this assertion together is not it.
"""


def _shell_without_comments() -> str:
    """``services/vllm.sh`` with whole-line comments removed.

    The file documents the *old* architecture name and several rejected flag spellings in its
    comments; a plain grep would happily match those and prove nothing.
    """
    lines = VLLM_SH.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _hf_overrides_architecture() -> str:
    text = _shell_without_comments()
    match = re.search(r"--hf-overrides\s+(['\"])(?P<body>.*?)\1", text)
    if match is None:
        pytest.fail(
            f"no --hf-overrides argument found in the executable part of {VLLM_SH}; the plugin "
            "is selected by overriding the model's `architectures`, so if that flag is gone the "
            "stock DiffusionGemma boots and the shim never loads"
        )
    payload = match.group("body").replace('\\"', '"')
    try:
        overrides = json.loads(payload)
    except json.JSONDecodeError as exc:
        pytest.fail(f"--hf-overrides payload is not JSON after unescaping: {payload!r} ({exc})")
    architectures = overrides.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        pytest.fail(
            f"--hf-overrides sets architectures={architectures!r}; vLLM resolves the model from "
            "this list, so it must name exactly one class -- ours"
        )
    return str(architectures[0])


def _plugin_architecture() -> str:
    """Read the ``ARCHITECTURE`` literal out of the plugin's ``__init__.py`` without importing it.

    Importing is not an option on the host: the plugin's module imports vLLM, which only exists
    inside the container.
    """
    tree = ast.parse(PLUGIN_INIT.read_text())
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "ARCHITECTURE":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                pytest.fail(
                    f"ARCHITECTURE in {PLUGIN_INIT} is not a plain string literal; it has to be "
                    "one so that services/vllm.sh can be checked against it without importing "
                    "vLLM"
                )
    pytest.fail(
        f"{PLUGIN_INIT} defines no module-level ARCHITECTURE; that constant is the single name "
        "vLLM resolves the model by and services/vllm.sh must repeat it verbatim"
    )
    raise AssertionError("unreachable")


def test_architecture_string_matches_between_launcher_and_plugin() -> None:
    """Criterion 4: the two copies of the architecture name must be byte-identical."""
    from_shell = _hf_overrides_architecture()
    from_plugin = _plugin_architecture()

    assert from_shell, "empty architecture name in services/vllm.sh"
    assert from_plugin, "empty ARCHITECTURE in the plugin"
    assert from_shell == from_plugin, (
        f"services/vllm.sh boots vLLM with architectures=[{from_shell!r}] but the plugin "
        f"registers {from_plugin!r}. These are two copies of one name and must change together: "
        "vLLM looks the model class up by this exact string, so any drift -- including case or "
        "a trailing suffix -- makes the server fail to resolve the model at boot, after the "
        "container has already started and pulled the weights. (This exact drift happened once: "
        "the launcher said AstraDiffusionGemmaForCausalLM, a class v0.29.0 does not have.)"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--attention-backend",
        "--diffusion-config",
        "--max-num-seqs",
        "--override-generation-config",
    ],
)
def test_load_bearing_flags_are_still_passed(flag: str) -> None:
    """Criterion 5: each flag that keeps a plugin-on boot equivalent to a plugin-off boot."""
    text = _shell_without_comments()
    match = re.search(rf"{re.escape(flag)}(?:=|\s+)(?P<value>\S+)", text)
    assert match is not None, (
        f"services/vllm.sh no longer passes {flag} (outside comments).\n{WHY_THE_FLAGS}"
    )
    assert match.group("value") not in ("\\", "$GEN_FLAGS", "$PLUGIN_FLAGS"), (
        f"{flag} is present but has no value in services/vllm.sh; a bare flag does not falsify "
        f"the branch it is there to falsify.\n{WHY_THE_FLAGS}"
    )


def test_architecture_override_is_only_applied_with_the_plugin() -> None:
    """The override and the plugin install are one decision, not two.

    ``--hf-overrides`` names a class that only exists once the plugin wheel is installed. If the
    override could be emitted on an ASTRA_PLUGIN=0 boot, vLLM would fail to resolve the model;
    if the plugin were installed without the override, the stock (broken) image path would run
    while the log still claimed the plugin was loaded.
    """
    text = _shell_without_comments()
    assert text.count("--hf-overrides") == 1, (
        "expected exactly one --hf-overrides in services/vllm.sh; more than one makes it "
        "ambiguous which boot paths carry the architecture override"
    )
    guard = re.search(r'if \[ "\$PLUGIN" = "1" \];.*?\nfi\n', text, re.DOTALL)
    assert guard is not None, (
        "services/vllm.sh no longer guards the plugin flags with a PLUGIN=1 branch; the "
        "architecture override and the wheel install must be enabled by the same condition"
    )
    assert "--hf-overrides" in guard.group(0), (
        "--hf-overrides is emitted outside the PLUGIN=1 branch: an ASTRA_PLUGIN=0 boot would "
        "then ask vLLM for a class that is not installed and die at model resolution"
    )


# =================================================================================================
# T-0058 criteria 1-2: the startup gate the engine could not clear on 2026-09-16.
#
# `services/vllm.sh` has never passed `--gpu-memory-utilization`, so vLLM's v0.29.0 default of
# 0.92 applied and the engine demanded 0.92 of the *whole device* be free before it loaded a
# weight. With the recogniser resident it never was, and the container died 21 s after launch:
#
#     ValueError: Free memory on device cuda:0 (27.23/31.33 GiB) on startup is less than desired
#     GPU memory utilization (0.92, 28.83 GiB).      -- logs/vllm-coldboot-73a85dd.log
#
# What the flag does and does not do, because the window below is meaningless without it: with
# `--kv-cache-memory-bytes` set, `GPUWorker.determine_available_memory` returns those bytes and
# logs that it "does not respect the gpu_memory_utilization config". So on our path this flag
# gates **only** the startup free-memory check: it sizes no arena, does not shrink the KV pool
# and gives the sampler neither more room nor less. The single thing it costs is guard strength,
# which is why the window has a floor as well as a ceiling.
#
# The window, from the MiB recorded that night (`tests/unit/_box_rig.py` carries the readings):
#
#   upper  27 882 / 32 082 = 0.8691   free at the engine's own check, over vLLM's own total.
#                                     Above this the engine cannot start with the recogniser up.
#   lower  27 434 / 32 082 = 0.8551   the engine's measured steady footprint. A demand *below*
#                                     what the engine actually uses passes boots that die later
#                                     and deeper, so the check stops protecting anything.
#
# 0.86 sits 291 MiB below the observed free and 157 MiB above the footprint. 0.85 -- the figure
# PROGRESS first recorded -- is 164 MiB below the footprint and is excluded for that reason.
# Widening this window is a `docs/tasks/T-0058.md` edit (ambiguity 1), never a test edit.
# =================================================================================================

VLLM_TOTAL_MIB = 32082  # vLLM's own denominator (mem_get_info), read from the ValueError
FREE_AT_ENGINE_CHECK_MIB = 27882  # what vLLM saw with the recogniser resident
ENGINE_FOOTPRINT_MIB = 27434  # the engine's steady footprint at KV 4 GiB (T-0014/T-0048 c11)

UTILISATION_CEILING = FREE_AT_ENGINE_CHECK_MIB / VLLM_TOTAL_MIB  # 0.86908...
UTILISATION_FLOOR = ENGINE_FOOTPRINT_MIB / VLLM_TOTAL_MIB  # 0.85512...

WHY_THE_WINDOW = (
    f"the value must be > {UTILISATION_FLOOR:.4f} (= {ENGINE_FOOTPRINT_MIB}/{VLLM_TOTAL_MIB}, the "
    f"engine's own footprint -- below it the startup check passes boots that die later) and "
    f"<= {UTILISATION_CEILING:.4f} (= {FREE_AT_ENGINE_CHECK_MIB}/{VLLM_TOTAL_MIB}, the free "
    f"memory with the recogniser resident -- above it the engine cannot start at all). "
    "docs/tasks/T-0058.md, 'The arithmetic, once, with its provenance'."
)

UTILISATION_FLAG = "--gpu-memory-utilization"


def _utilisation_in(text: str) -> float | None:
    """The ``--gpu-memory-utilization`` value in a piece of shell text, or ``None`` if absent."""
    match = re.search(rf"{re.escape(UTILISATION_FLAG)}(?:=|\s+)([0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else None


def _outside_the_window(value: float) -> str:
    """Empty when ``value`` is in the measured window, else the reason it is not."""
    if value <= UTILISATION_FLOOR:
        return (
            f"{value} demands {round(value * VLLM_TOTAL_MIB)} MiB, which is below the engine's "
            f"own {ENGINE_FOOTPRINT_MIB} MiB footprint"
        )
    if value > UTILISATION_CEILING:
        return (
            f"{value} demands {round(value * VLLM_TOTAL_MIB)} MiB, more than the "
            f"{FREE_AT_ENGINE_CHECK_MIB} MiB free with the recogniser resident"
        )
    return ""


def test_the_startup_gate_flag_is_present_and_within_the_measured_window():
    """Criterion 1: the gate is passed explicitly, at a value the box can actually clear."""
    text = _shell_without_comments()
    value = _utilisation_in(text)
    assert value is not None, (
        f"services/vllm.sh passes no {UTILISATION_FLAG} (outside comments), so vLLM's 0.92 "
        f"default applies and the engine demands {round(0.92 * VLLM_TOTAL_MIB)} MiB of the "
        f"device be free before it loads anything. With the recogniser resident only "
        f"{FREE_AT_ENGINE_CHECK_MIB} MiB is, so every start in that configuration dies 21 s in "
        f"(logs/vllm-coldboot-73a85dd.log, 2026-09-16 22:17). {WHY_THE_WINDOW}"
    )
    assert not _outside_the_window(value), (
        f"services/vllm.sh passes {UTILISATION_FLAG} {value}: {_outside_the_window(value)}. "
        f"{WHY_THE_WINDOW}"
    )


def test_the_startup_gate_is_passed_to_vllm_serve_not_to_docker():
    """Criterion 1/2: the flag belongs to the server's argv, inside the container's ``-c`` payload.

    ``docker run`` has no such option and refuses it outright, and a flag on the wrong side of
    ``--entrypoint bash "$CONTAINER" -c`` never reaches vLLM — the boot then fails the same way
    it did on 2026-09-16 while the script looks fixed.
    """
    text = _shell_without_comments()
    assert _utilisation_in(text) is not None, "the flag is absent; see the row above"
    serve = text.split("vllm serve", 1)
    assert len(serve) == 2, "services/vllm.sh no longer contains a `vllm serve` invocation"
    assert _utilisation_in(serve[1]) is not None, (
        f"{UTILISATION_FLAG} appears in services/vllm.sh but not after `vllm serve` — docker run "
        "rejects the flag, and anything before the container's `-c` payload never reaches the "
        "engine at all"
    )


def test_the_comment_block_carries_the_arithmetic_and_the_log_reference():
    """Criterion 1: the next reader must find the failure, not just the digit.

    A bare number in a launcher is the thing nobody dares touch and everybody eventually
    "simplifies". The comment has to carry the boot it fixes, the ceiling, the engine's own
    footprint as a fraction, and the sentence that a value below that footprint stops being a
    guard at all.
    """
    comments = "\n".join(
        line for line in VLLM_SH.read_text().splitlines() if line.lstrip().startswith("#")
    )
    assert "logs/vllm-coldboot-73a85dd.log" in comments, (
        "criterion 1: the comment names the log of the boot this flag exists to fix "
        "(2026-09-16 22:17), so the arithmetic can be re-derived from the recording"
    )
    for figure, why in (
        ("27.23", "the free memory the engine measured with the recogniser resident"),
        ("31.33", "vLLM's own total -- the denominator, which is NOT nvidia-smi's 31.84"),
        ("0.869", "the ceiling, 27882/32082: above it the engine cannot start at all"),
        ("0.855", "the engine's own footprint as a fraction, 27434/32082: the floor"),
    ):
        assert figure in comments, (
            f"criterion 1: the comment block is missing {figure} ({why}). The whole point of the "
            "block is that the value can be checked against the recording without this task file"
        )
    lowered = comments.lower()
    assert "footprint" in lowered and ("guard" in lowered or "protect" in lowered), (
        "criterion 1: …and the sentence that a value *below* the engine's own footprint stops "
        "being a guard — the reason 0.85 was rejected. Without it the next person reads 0.86 as "
        "a margin to be widened when a boot is tight"
    )


def test_the_kv_pool_is_unchanged_and_still_spelled_in_bytes():
    """Criterion 2: the whole diff to the argv is one flag; the pool is not touched.

    Green before the implementation and after it — that is the point of a negative criterion.
    The pool stays 4 GiB (T-0048 ruling A13, whose shrink premise the 2026-09-16 run removed),
    and it stays spelled in bytes: ``vllm.utils.argparse_utils.human_readable_int`` takes a
    single-letter suffix, so ``4GiB`` matches nothing and dies in argparse, while ``4g`` would
    quietly mean 4e9 — 7 % less cache than intended.
    """
    text = _shell_without_comments()
    match = re.search(r"--kv-cache-memory-bytes(?:=|\s+)(\S+)", text)
    assert match is not None, "services/vllm.sh no longer passes --kv-cache-memory-bytes"
    assert match.group(1) == "4294967296", (
        "criterion 2: the KV pool is byte-identical at 4294967296 (ruling A13). The utilisation "
        "flag cannot shrink the pool and is not a way to re-open that decision: with "
        "--kv-cache-memory-bytes set, vLLM skips profiling and says so in the log. It said "
        f"{match.group(1)!r}"
    )
    for banned in ("4GiB", "4G", "4g", "4096M"):
        assert f"--kv-cache-memory-bytes {banned}" not in text, (
            f"criterion 2: `--kv-cache-memory-bytes {banned}` is the argparse landmine "
            "(.claude/rules/gpu-services.md): a single-letter suffix means decimal in lowercase "
            "and binary in uppercase, and a multi-letter one is a boot-time crash"
        )


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (0.92, "vLLM v0.29.0's default — the value this task exists to escape"),
        (0.85, "below the engine's 27434 MiB footprint: the check stops protecting anything"),
        (0.87, "demands 27911 MiB, 29 MiB more than the box has free"),
        (0.90, "demands 28874 MiB, nearly a gibibyte more than the box has free"),
    ],
)
def test_a_value_outside_the_measured_window_is_rejected(value: float, why: str):
    """Criterion 1, adversarial: the window as written excludes the values that broke the box.

    The script's own text is rewritten to each bad value and re-read through the same parser and
    the same predicate the row above uses, so a window quietly widened to accommodate a future
    edit reddens here. This row is green before the implementation lands on purpose: it is what
    keeps the row above from being satisfiable by any number at all.
    """
    mutated = _with_utilisation(VLLM_SH.read_text(), value)
    parsed = _utilisation_in(
        "\n".join(line for line in mutated.splitlines() if not line.lstrip().startswith("#"))
    )
    assert parsed == value, (
        f"the mutation harness did not take: expected {value}, the mutated script parses as "
        f"{parsed!r}"
    )
    assert _outside_the_window(value), (
        f"{value} ({why}) must be outside the window, and this window accepts it. {WHY_THE_WINDOW}"
    )


def test_a_value_inside_the_window_is_accepted():
    """The other half of the row above: the window is not simply empty (0.86 is the candidate)."""
    assert not _outside_the_window(0.86), (
        "0.86 demands 27591 MiB: 291 below the 27882 MiB free with the recogniser resident, and "
        f"157 above the engine's 27434 MiB footprint. {WHY_THE_WINDOW}"
    )
    assert _utilisation_in(_with_utilisation(VLLM_SH.read_text(), 0.86)) == 0.86


def _with_utilisation(text: str, value: float) -> str:
    """The script with the flag's value rewritten — or the flag added, while it is still absent."""
    if _utilisation_in(text) is not None:
        return re.sub(
            rf"({re.escape(UTILISATION_FLAG)}(?:=|\s+))[0-9]*\.?[0-9]+", rf"\g<1>{value}", text
        )
    return text.replace("vllm serve", f"vllm serve {UTILISATION_FLAG} {value}", 1)
