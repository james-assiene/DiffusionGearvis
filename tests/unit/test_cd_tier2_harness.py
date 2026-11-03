"""T-0067 tier 2 harness guards (fast suite; no docker, no GPU).

The first real tier-2 run (2026-09-30, ``logs/t0067-tier2-295c13b.log``) lost five rows to the harness, not to the
code under test:

- the host venv's site-packages were mounted as ``/hostsite`` and appended to ``sys.path`` inside the pinned
  image; transformers (via vllm) then found the host's cp311 scipy there and could not import it under the
  image's Python 3.12. Everything the inner modules import is in the image, so nothing from the host venv goes on
  the container's path (ruling S7 as amended);
- the coupling probe indexed a map of three hard-coded classes with the five class names the host sent and died
  on ``KeyError: 'RequestState'``. It now resolves each class by dotted path and reports a missing one by name.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "tests" / "container_scripts"
HARNESS = [
    REPO / "tests" / "contract" / "test_vllm_cd_cuda_gpu.py",
    REPO / "tests" / "contract" / "test_vllm_cd_perf_gpu.py",
    REPO / "tests" / "contract" / "test_vllm_cd_coupling_gpu.py",
    REPO / "tests" / "contract" / "test_vllm_cd_perstep_sync_gpu.py",
    *sorted(SCRIPTS.glob("*.py")),
]


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _is_sys_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


@pytest.mark.parametrize("path", HARNESS, ids=lambda p: p.name)
def test_no_host_site_packages_on_the_container_path(path: Path) -> None:
    tree = ast.parse(path.read_text())
    docs = _docstring_ids(tree)
    offences = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docs
            and "hostsite" in node.value
        ):
            offences.append(f"line {node.lineno}: string mentions hostsite: {node.value[:80]!r}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_sys_path(node.func.value)
        ):
            offences.append(f"line {node.lineno}: sys.path.{node.func.attr}(...)")
    assert offences == [], (
        f"{path.relative_to(REPO)} puts host packages in the container: {offences}"
    )


def _probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_astra_cd_probe", SCRIPTS / "astra_cd_probe.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Plain:
    klass_level = 1

    def __init__(self) -> None:
        self.assigned_in_init = 0

    def later(self) -> None:
        self.assigned_later = 0

    @property
    def prop(self) -> int:
        return 0


@dataclasses.dataclass
class _Batch:  # like v0.29.0's InputBatch: fields without defaults are not in dir(cls)
    num_reqs: int
    idx_mapping_np: Any


def _importer(modules: dict[str, ModuleType]) -> Any:
    def imp(name: str) -> ModuleType:
        if name == "boom":
            raise RuntimeError("module-level failure")
        if name not in modules:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return modules[name]

    return imp


def _fake_module() -> ModuleType:
    m = ModuleType("fakevllm.states")
    m._Plain = _Plain  # type: ignore[attr-defined]
    m._Batch = _Batch  # type: ignore[attr-defined]
    m.not_a_class = 3  # type: ignore[attr-defined]
    return m


def test_probe_reports_missing_classes_by_name_instead_of_raising() -> None:
    probe = _probe()
    wanted = {
        "fakevllm.states._Plain": ["assigned_in_init"],
        "fakevllm.states.RequestState": ["draft_tokens"],  # the class the first run choked on
        "fakevllm.gone.InputBatch": ["num_reqs"],
        "boom.Thing": ["x"],
        "fakevllm.states.not_a_class": ["x"],
        "Undotted": ["x"],
    }
    attrs, missing = probe.check_reads(wanted, _importer({"fakevllm.states": _fake_module()}))
    assert attrs == {"fakevllm.states._Plain": {"assigned_in_init": True}}
    assert set(missing) == set(wanted) - {"fakevllm.states._Plain"}
    assert "no class RequestState" in missing["fakevllm.states.RequestState"]
    assert "fakevllm.gone" in missing["fakevllm.gone.InputBatch"]
    assert "RuntimeError" in missing["boom.Thing"]
    assert "no class not_a_class" in missing["fakevllm.states.not_a_class"]


def test_probe_sees_init_later_class_property_and_dataclass_attributes() -> None:
    probe = _probe()
    wanted = {
        "fakevllm.states._Plain": [
            "assigned_in_init",
            "assigned_later",
            "klass_level",
            "prop",
            "never_defined",
        ],
        "fakevllm.states._Batch": ["num_reqs", "idx_mapping_np", "cu_num_logits_np"],
    }
    attrs, missing = probe.check_reads(wanted, _importer({"fakevllm.states": _fake_module()}))
    assert missing == {}
    assert attrs["fakevllm.states._Plain"] == {
        "assigned_in_init": True,
        "assigned_later": True,
        "klass_level": True,
        "prop": True,
        "never_defined": False,
    }
    assert attrs["fakevllm.states._Batch"] == {
        "num_reqs": True,
        "idx_mapping_np": True,
        "cu_num_logits_np": False,
    }


def test_coupling_reads_are_dotted_vllm_class_paths() -> None:
    from tests.contract.test_vllm_cd_coupling_gpu import READS

    for key in READS:
        module, _, cls = key.rpartition(".")
        assert module.startswith("vllm.") and cls[:1].isupper(), key
    # the reads added after review (req_states.draft_tokens; InputBatch fields) are present
    assert READS["vllm.v1.worker.gpu.states.RequestState"] == ["draft_tokens"]
    assert {"num_draft_tokens", "idx_mapping_np", "cu_num_logits_np"} <= set(
        READS["vllm.v1.worker.gpu.input_batch.InputBatch"]
    )


# --------------------------------------------------------------------------- T-0072 perf wrapper (2026-10-01)

#: tier-2 wrappers that run pytest inside the container; each must go through T-0067's helper with an exact count
TIER2_WRAPPERS = [
    REPO / "tests" / "contract" / "test_vllm_cd_perf_gpu.py",
    REPO / "tests" / "contract" / "test_vllm_cd_perstep_gpu.py",
    REPO / "tests" / "contract" / "test_vllm_cd_perstep_sync_gpu.py",  # T-0071a criterion 1
]


def _module_ints(tree: ast.Module) -> dict[str, int]:
    out = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            out[node.targets[0].id] = node.value.value
    return out


@pytest.mark.parametrize("path", TIER2_WRAPPERS, ids=lambda p: p.name)
def test_tier2_wrappers_use_the_helper_with_exact_counts(path: Path) -> None:
    """No hand-rolled ``docker run ... pytest.main`` (exit 0 alone passed 43 skips, launch review 2026-10-01);
    every ``_pytest_in_container`` call names a positive pinned count."""
    tree = ast.parse(path.read_text())
    docs = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            assert "pytest.main" not in node.value, (
                f"{path.name} line {node.lineno} runs pytest itself"
            )
    ints = _module_ints(tree)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_pytest_in_container"
    ]
    assert calls, f"{path.name} does not route through test_vllm_cd_cuda_gpu._pytest_in_container"
    for call in calls:
        assert len(call.args) >= 2, f"{path.name} line {call.lineno}: no expected count"
        count = call.args[1]
        if isinstance(count, ast.Name):
            assert ints.get(count.id, 0) > 0, (
                f"{path.name}: {count.id} is not a pinned positive int"
            )
        else:  # a pinned per-module dict lookup (perstep) or a literal
            assert isinstance(count, (ast.Subscript, ast.Constant)), ast.dump(count)


def test_perf_wrapper_pins_counts_and_requires_the_5090() -> None:
    from tests.contract import test_vllm_cd_perf_gpu as perf

    assert perf.PERF_CUDA_PASSED == 36 + 5 + 1 + 1
    assert perf.CUT_AUTOMATA == 793
    assert perf.BUCKETS == (64, 128, 256, 512)
    src = (REPO / "tests" / "contract" / "test_vllm_cd_perf_gpu.py").read_text()
    assert '"5090" in' in src and "t0072-perf-probe-" in src and "host_tree_dirty" in src


def test_perf_cuda_module_fails_rather_than_skips_under_require_cuda() -> None:
    src = (REPO / "tests" / "cd" / "cuda" / "test_cd_perf_cuda.py").read_text()
    assert 'os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1"' in src and "pytest.fail(" in src


# --------------------------------------------------------------------------- T-0071a sync rows (2026-10-03)


def test_perstep_sync_wrapper_pins_its_count_and_the_cuda_module_fails_rather_than_skips() -> None:
    """T-0071a criterion 1 + T-0076 criterion 11: 1 liveness row + 5 per-step arms (mf, mar, mar_sc, qa, qa_sc)
    x buckets {64, 128} = 11; a skip under ``ASTRA_CD_REQUIRE_CUDA=1`` would make "0 raises" vacuous."""
    from tests.contract import test_vllm_cd_perstep_sync_gpu as sync

    assert sync.CD_PERSTEP_SYNC_PASSED == 1 + 5 * 2
    cuda_src = (REPO / "tests" / "cd" / "cuda" / "test_cd_perstep_sync_cuda.py").read_text()
    arms = buckets = None
    for node in ast.parse(cuda_src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "ARMS":
                arms = ast.literal_eval(node.value)
            elif node.targets[0].id == "BUCKETS":
                buckets = ast.literal_eval(node.value)
    assert arms == ("mf", "mar", "mar_sc", "qa", "qa_sc"), arms
    assert buckets == (64, 128), buckets
    assert 1 + len(arms) * len(buckets) == sync.CD_PERSTEP_SYNC_PASSED
    assert sync.CUDA_ENV == {"ASTRA_CD_DEVICE": "cuda", "ASTRA_CD_STOCK": "image"}
    src = (REPO / "tests" / "cd" / "cuda" / "test_cd_perstep_sync_cuda.py").read_text()
    assert 'os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1"' in src and "pytest.fail(" in src
