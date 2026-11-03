"""T-0063 criteria 1-2: the constrained-decoding stack stays out of main's environment and imports.

torch, outlines-core and tokenizers live in the non-default ``cd`` uv dependency group (docs/tasks/T-0062.md,
human decision 1). These rows run in the fast suite, so they read files and spawn one short subprocess;
they never import torch, outlines_core, tokenizers or ``local_astra.decoding`` themselves.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[2]
CD_PACKAGES = frozenset({"torch", "outlines-core", "tokenizers"})
CD_MODULES = ("torch", "outlines_core", "tokenizers", "local_astra.decoding")
#: diffgemma-fa docs/ENV.md at 27c7d0d: the outlines-core it ran.
OUTLINES_CORE_PIN = "==0.2.14"
#: torch inside vllm/vllm-openai:v0.29.0 is 2.13.0+cu130 (docs/tasks/T-0062.md, decision 2).
TORCH_PIN = "==2.13.0"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CD_TEST_RECIPE = 'ASTRA_CD_TESTS=1 uv run --all-extras --group cd pytest tests/cd -q -m "not gpu"'


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def _cd_group() -> dict[str, Requirement]:
    groups = _pyproject().get("dependency-groups", {})
    assert "cd" in groups, (
        "pyproject.toml has no [dependency-groups] cd (T-0063 criterion 1): torch, outlines-core and "
        "tokenizers must live in a non-default uv dependency group, not an extra"
    )
    reqs = [Requirement(r) for r in groups["cd"] if isinstance(r, str)]
    return {canonicalize_name(r.name): r for r in reqs}


# --------------------------------------------------------------------------- criterion 1


def test_cd_group_contains_the_three_packages() -> None:
    group = _cd_group()
    missing = CD_PACKAGES - set(group)
    assert not missing, f"[dependency-groups] cd lacks {sorted(missing)}"


def test_outlines_core_is_pinned_to_the_version_diffgemma_fa_ran() -> None:
    assert str(_cd_group()["outlines-core"].specifier) == OUTLINES_CORE_PIN


def test_torch_is_pinned_to_the_container_version() -> None:
    assert str(_cd_group()["torch"].specifier) == TORCH_PIN


def test_torch_comes_from_an_explicit_pytorch_cpu_index() -> None:
    tool_uv = _pyproject().get("tool", {}).get("uv", {})
    source = tool_uv.get("sources", {}).get("torch")
    assert source is not None, "[tool.uv.sources] has no torch entry"
    sources = source if isinstance(source, list) else [source]
    index_names = {s.get("index") for s in sources}
    indexes = {i.get("name"): i for i in tool_uv.get("index", [])}
    matching = [indexes[n] for n in index_names if n in indexes]
    assert matching, f"torch's source index {index_names} is not declared in [[tool.uv.index]]"
    for idx in matching:
        assert idx.get("url", "").rstrip("/") == PYTORCH_CPU_INDEX, idx
        # explicit: only packages pinned to it resolve there, so nothing else silently comes from it
        assert idx.get("explicit") is True, idx


def test_makefile_cd_test_target_runs_the_cd_suite_with_the_group() -> None:
    text = (REPO / "Makefile").read_text()
    m = re.search(r"^cd-test\s*:[^\n]*$", text, re.MULTILINE)
    assert m is not None, "Makefile has no `cd-test` target"
    # the recipe is either inline after `;` or the following tab-indented lines
    head = m.group(0)
    body = [head.split(";", 1)[1]] if ";" in head else []
    for line in text[m.end() :].splitlines()[1:]:
        if not line.startswith("\t"):
            break
        body.append(line)
    recipe = " ".join(part.strip().lstrip("@") for part in body)
    recipe = recipe.split("#", 1)[0].strip()
    assert recipe == CD_TEST_RECIPE, f"cd-test recipe is {recipe!r}"


def _editable_roots(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """The project's own package in uv.lock, selected by its editable source rather than by name: T-0156 R1/R2 rename
    the project ``local-astra`` -> ``diffusion-gearvis`` (ruling O-R), and the root is the one ``source = {editable =
    "."}`` entry under either name. More than one would make "the root" ambiguous, so that is an error too."""
    roots = [p for p in lock["package"] if p.get("source") == {"editable": "."}]
    assert len(roots) <= 1, (
        f"uv.lock has {len(roots)} editable-root packages: {[p['name'] for p in roots]}"
    )
    return roots


def test_uv_lock_records_the_cd_group() -> None:
    """``uv sync --group cd`` installs the three packages: the lock resolves them under the group.

    Red until the human regenerates uv.lock after the pyproject change (agents may not edit it).
    """
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    roots = _editable_roots(lock)
    assert roots, 'uv.lock has no root package (source = {editable = "."})'
    groups = roots[0].get("dev-dependencies", {})
    assert "cd" in groups, (
        "uv.lock has no `cd` dependency group for the root package: the human regenerates uv.lock after "
        "the pyproject change (docs/tasks/T-0062.md, decision 1)"
    )
    names = {canonicalize_name(d["name"]) for d in groups["cd"]}
    assert names >= CD_PACKAGES, f"the locked cd group resolves {sorted(names)}"


# --------------------------------------------------------------------------- criterion 2


def test_cd_is_a_group_and_not_an_extra() -> None:
    project = _pyproject()["project"]
    extras = project.get("optional-dependencies", {})
    assert "cd" not in extras, (
        "`cd` must be a dependency group; `uv sync --all-extras` installs extras"
    )
    for extra, reqs in extras.items():
        names = {canonicalize_name(Requirement(r).name) for r in reqs}
        assert not names & CD_PACKAGES, f"extra {extra!r} pulls in {sorted(names & CD_PACKAGES)}"
    deps = {canonicalize_name(Requirement(r).name) for r in project.get("dependencies", [])}
    assert not deps & CD_PACKAGES, f"[project] dependencies pull in {sorted(deps & CD_PACKAGES)}"
    default_groups = _pyproject().get("tool", {}).get("uv", {}).get("default-groups", ["dev"])
    assert default_groups != "all" and "cd" not in default_groups, default_groups


def _all_extras_closure(lock: dict[str, Any], default_groups: list[str]) -> set[str]:
    """Package names ``uv sync --all-extras`` installs, read from uv.lock (markers ignored: wider)."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for pkg in lock["package"]:
        by_name.setdefault(canonicalize_name(pkg["name"]), []).append(pkg)
    roots = _editable_roots(lock)
    assert roots, 'uv.lock has no root package (source = {editable = "."})'
    root = roots[0]
    todo: list[tuple[str, tuple[str, ...]]] = []
    todo += [(d["name"], tuple(d.get("extra", ()))) for d in root.get("dependencies", [])]
    for deps in root.get("optional-dependencies", {}).values():
        todo += [(d["name"], tuple(d.get("extra", ()))) for d in deps]
    for group in default_groups:
        todo += [
            (d["name"], tuple(d.get("extra", ())))
            for d in root.get("dev-dependencies", {}).get(group, [])
        ]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    names: set[str] = set()
    while todo:
        name, extras = todo.pop()
        key = (canonicalize_name(name), extras)
        if key in seen:
            continue
        seen.add(key)
        names.add(key[0])
        for pkg in by_name.get(key[0], []):
            todo += [(d["name"], tuple(d.get("extra", ()))) for d in pkg.get("dependencies", [])]
            for extra in extras:
                todo += [
                    (d["name"], tuple(d.get("extra", ())))
                    for d in pkg.get("optional-dependencies", {}).get(extra, [])
                ]
    return names


def test_uv_sync_all_extras_installs_none_of_the_cd_packages() -> None:
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    default_groups = _pyproject().get("tool", {}).get("uv", {}).get("default-groups", ["dev"])
    assert isinstance(default_groups, list), default_groups
    closure = _all_extras_closure(lock, default_groups)
    assert len(closure) > 5, f"closure walk found only {sorted(closure)}; the lock walker is broken"
    assert (
        canonicalize_name("pydantic") in closure
    )  # probe self-check: a known core dependency is seen
    leaked = closure & CD_PACKAGES
    assert not leaked, f"`uv sync --all-extras` would install {sorted(leaked)} (T-0063 criterion 2)"


def _fast_suite_python_files() -> list[Path]:
    tests = REPO / "tests"
    files = sorted((tests / "unit").rglob("*.py")) + sorted((tests / "contract").rglob("*.py"))
    files += sorted((tests / "fakes").rglob("*.py"))  # imported by the contract rows
    files += [tests / "conftest.py"]
    return [f for f in files if "__pycache__" not in f.parts]


def _forbidden_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []

    def hit(name: str) -> bool:
        return any(name == m or name.startswith(m + ".") for m in CD_MODULES)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if hit(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if hit(node.module):
                found.append(node.module)
            elif node.module in ("local_astra",):
                found += [f"local_astra.{a.name}" for a in node.names if a.name == "decoding"]
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and hit(arg.value):
                    found.append(arg.value)
    return found


def test_the_import_scan_detects_a_forbidden_import() -> None:
    """Probe self-check: the scan below is not vacuous (each form it claims to catch is caught)."""
    src = textwrap.dedent(
        """
        import torch
        from outlines_core import Vocabulary
        from local_astra import decoding
        import importlib
        importlib.import_module("tokenizers")
        from local_astra.decoding.grammars import compile_bfcl
        import local_astra.conductor
        """
    )
    assert sorted(_forbidden_imports(ast.parse(src))) == sorted(
        [
            "torch",
            "outlines_core",
            "local_astra.decoding",
            "tokenizers",
            "local_astra.decoding.grammars",
        ]
    )


def test_nothing_in_the_fast_suite_imports_the_cd_stack() -> None:
    offenders: dict[str, list[str]] = {}
    files = _fast_suite_python_files()
    assert len(files) > 50, f"only {len(files)} files scanned; the fast-suite file walk is broken"
    for path in files:
        found = _forbidden_imports(ast.parse(path.read_text(), filename=str(path)))
        if found:
            offenders[str(path.relative_to(REPO))] = found
    assert not offenders, f"fast-suite modules import the cd stack: {offenders}"


_PROBE = textwrap.dedent(
    """
    import importlib, json, pkgutil, sys
    import local_astra, local_astra.cli, local_astra.conductor
    imported = []
    for info in pkgutil.walk_packages(local_astra.conductor.__path__, "local_astra.conductor."):
        importlib.import_module(info.name)
        imported.append(info.name)
    watch = json.loads(sys.argv[1])
    present = sorted(m for m in sys.modules if any(m == w or m.startswith(w + ".") for w in watch))
    print(json.dumps({"imported": imported, "present": present}))
    """
)


def _probe(watch: list[str]) -> dict[str, list[str]]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(watch)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )
    assert proc.returncode == 0, (
        f"importing local_astra / cli / conductor.* failed:\n{proc.stderr[-4000:]}"
    )
    result: dict[str, list[str]] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def test_importing_the_runtime_leaves_the_cd_stack_unimported() -> None:
    result = _probe(list(CD_MODULES))
    assert len(result["imported"]) >= 5, f"walked only {result['imported']}; the probe is broken"
    assert result["present"] == [], (
        f"importing local_astra, local_astra.cli and every local_astra.conductor.* module put "
        f"{result['present']} in sys.modules (T-0063 criterion 2)"
    )


def test_the_sys_modules_probe_sees_what_is_imported() -> None:
    """Probe self-check (a guard needs a row for being absent): a module the probe certainly imports is
    reported, so an empty ``present`` above means absent, not unobserved."""
    result = _probe(["local_astra.cli", "local_astra.conductor.turn_machine"])
    assert result["present"] == ["local_astra.cli", "local_astra.conductor.turn_machine"], result[
        "present"
    ]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
