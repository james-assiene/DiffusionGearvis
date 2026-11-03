"""T-0011 criterion 10: the wheel contains this code, and the image has its imports.

``services/vllm.sh`` installs the plugin with ``pip install --no-deps --no-index``, so what the
plugin's ``pyproject.toml`` *declares* as a dependency is never installed: a top-level import that
the pinned image does not already provide is a boot-time ``ImportError`` inside the engine, not a
missing-wheel message. This file therefore checks two things no other test can:

1. every ``astra_vllm_plugin.*`` module in the source tree is inside the built wheel;
2. every third-party top-level import in the package — **derived with ``ast``, not hardcoded** —
   is on the list of modules the pinned image is known to have.

The build is ``uv build --wheel --offline`` into a temporary directory (no network, no
``pip install``, `.claude/rules/gpu-services.md`), which is also what ``make plugin-wheel``
runs, minus the destination.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin" / "astra_vllm_plugin"
PACKAGE = PLUGIN / "astra_vllm_plugin"

#: Third-party top-level modules present in the pinned image ``vllm/vllm-openai:v0.29.0``
#: (services/versions.lock). ``numpy``, ``msgpack`` and ``zmq`` were checked inside the image on
#: **2026-09-13** (docs/tasks/T-0011.md criterion 10); ``torch`` and ``vllm`` are what the image
#: exists to provide. Anything else must be vendored into the plugin or not used: with
#: ``pip install --no-deps --no-index`` there is no third chance.
CONTAINER_MODULES = frozenset({"numpy", "msgpack", "zmq", "torch", "vllm"})

#: modules the tap is expected to ship after T-0011 (the task's Owned files)
REQUIRED_MODULES = ("__init__", "model", "sampler", "tap", "model_state")
#: the subset T-0011 writes; ``model.py`` and ``__init__.py`` belong to T-0015 and are not ours
OWNED_MODULES = ("sampler", "tap", "model_state")


def _require_the_new_modules() -> None:
    """Every assertion below is about the package *after* T-0011; say so when it is not there."""
    missing = [m for m in REQUIRED_MODULES if not (PACKAGE / f"{m}.py").exists()]
    assert missing == [], f"T-0011 criterion 10: the plugin package does not contain {missing} yet"


def _top_level_imports(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: inside the package
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(out)],
        cwd=PLUGIN,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`uv build --wheel --offline` failed in {PLUGIN}:\n{proc.stdout}\n{proc.stderr}"
    )
    wheels = sorted(out.glob("astra_vllm_plugin-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    return wheels[0]


def test_wheel_contains_every_plugin_module(wheel: Path) -> None:
    """Criterion 10: a module that is not in the wheel does not exist inside the engine."""
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    on_disk = {p.stem for p in PACKAGE.glob("*.py")}
    missing_from_disk = [m for m in REQUIRED_MODULES if m not in on_disk]
    assert missing_from_disk == [], (
        f"T-0011 owns these modules but they do not exist yet: {missing_from_disk}"
    )
    missing = [m for m in sorted(on_disk) if f"astra_vllm_plugin/{m}.py" not in names]
    assert missing == [], (
        f"{wheel.name} is missing {missing}; check [tool.hatch.build.targets.wheel] packages"
    )


def test_third_party_imports_are_all_present_in_the_pinned_image() -> None:
    """Criterion 10: AST-derived, because a hardcoded list drifts from the code silently."""
    _require_the_new_modules()
    offenders: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        roots = _top_level_imports(path.read_text())
        third_party = {
            r
            for r in roots
            if r not in sys.stdlib_module_names and r != "astra_vllm_plugin" and r != "__future__"
        }
        extra = third_party - CONTAINER_MODULES
        if extra:
            offenders[path.name] = extra
    assert offenders == {}, (
        f"imports the pinned image does not provide (installed with --no-deps --no-index): "
        f"{offenders}; checked against {sorted(CONTAINER_MODULES)} (image inventory 2026-09-13)"
    )


def test_the_plugin_never_imports_local_astra() -> None:
    """Criterion 10, negative: the container has no ``local_astra`` on its path."""
    _require_the_new_modules()
    bad = {
        p.name: sorted(_top_level_imports(p.read_text()) & {"local_astra", "tests"})
        for p in sorted(PACKAGE.glob("*.py"))
    }
    bad = {k: v for k, v in bad.items() if v}
    assert bad == {}, f"the plugin imports first-party modules the engine cannot see: {bad}"


def test_torch_and_vllm_are_imported_lazily_not_at_module_scope() -> None:
    """The host suite imports ``tap``/``sampler``/``model_state`` with neither installed.

    Criterion 3's whole test file depends on this: a module-level ``import torch`` in the tap
    would turn every tier-1 test into a collection error on the host. Only T-0011's own modules
    are scanned -- ``model.py`` is T-0015's and imports vllm at module scope by design (it is
    only ever imported inside the container).
    """
    _require_the_new_modules()
    offenders: dict[str, set[str]] = {}
    for path in [PACKAGE / f"{m}.py" for m in OWNED_MODULES]:
        tree = ast.parse(path.read_text())
        module_level: set[str] = set()
        for node in tree.body:  # only the module scope, not function bodies
            if isinstance(node, ast.Import):
                module_level |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module_level.add(node.module.split(".")[0])
        heavy = module_level & {"torch", "vllm"}
        if heavy:
            offenders[path.name] = heavy
    assert offenders == {}, f"torch/vllm imported at module scope: {offenders}"


def test_the_plugin_declares_its_dependencies_even_though_pip_will_not_install_them() -> None:
    """Criterion 10: ``--no-deps`` means the declaration is documentation, and must still be true."""
    _require_the_new_modules()
    text = (PLUGIN / "pyproject.toml").read_text()
    used: set[str] = set()
    for path in sorted(PACKAGE.glob("*.py")):
        used |= _top_level_imports(path.read_text())
    for module, dist in (("numpy", "numpy"), ("msgpack", "msgpack"), ("zmq", "pyzmq")):
        if module in used:
            assert dist in text, (
                f"the plugin imports {module} but pyproject.toml does not declare {dist}"
            )
