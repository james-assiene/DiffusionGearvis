"""T-0077b criteria 6 (AST part) and 7 on CPU: where triton may be imported, what ``cd/kernels/`` may not do, packaging.

Rows:
- **AST, criterion 7 / ruling M1**: no module of the plugin outside ``cd/kernels/`` imports triton (live guard);
  inside ``cd/kernels/`` triton is imported only within function bodies, and ``entropy.py`` does import it there.
- **AST, criterion 6 / N2 / N6**: in ``cd/kernels/``: no ``tl.atomic_*``, ``triton.autotune``, ``triton.heuristics``;
  no write to ``triton.knobs``, ``os.environ``, ``os.putenv`` or torch/CUDA global config; no ``.item()``,
  ``.tolist()``, ``.cpu()``, ``.numpy()`` (or ``synchronize``). Launch options: ``enable_fp_fusion`` and
  ``enable_reflect_ftz`` are passed as ``False`` and never ``True`` (task "Implementation notes"; ruling: per
  launch only). A control row shows the walker flags each spelling on a synthetic source.
- **Blocked triton** (subprocess, a ``sys.meta_path`` finder that refuses ``triton*``): every
  ``astra_vllm_plugin.cd.*`` module (``cd/kernels/*`` included) imports, CPU ``constrained_marginals`` runs with
  ``kernels=True``/``False``/default and equals the branch base, and nothing even tries to import triton.
- **Triton installed** (subprocess): importing every ``cd`` module and running CPU marginals leaves ``triton`` out of
  ``sys.modules``; ``import astra_vllm_plugin.cd.kernels`` pulls in neither torch nor triton ("imports nothing heavy").
- **Packaging**: the wheel (``uv build --wheel --offline``) contains every ``cd/kernels/*.py`` (the glob of
  ``tests/unit/test_plugin_wheel.py`` is top-level only); ``plugin/astra_vllm_plugin/pyproject.toml`` is
  byte-identical to the branch base (no triton dependency); the host ``cd`` group pins ``triton==3.7.1`` and the
  installed ``triton.__version__`` is ``3.7.1``.

Regime: CPU, the ``cd`` group per ``uv.lock`` (torch 2.13.0+cpu, triton 3.7.1).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from pathlib import Path

import pytest

from tests import _git_range as gr
from tests.cd.unit import _frozen_t0077b as kb

REPO = Path(__file__).resolve().parents[3]
PKG = kb.PLUGIN_SRC / "astra_vllm_plugin"
PLUGIN_DIR = REPO / "plugin" / "astra_vllm_plugin"


def _kernel_sources() -> dict[str, str]:
    missing = [m for m in kb.KERNEL_MODULES if not (kb.KERNELS_DIR / f"{m}.py").is_file()]
    assert not missing, f"T-0077b not implemented: cd/kernels/{{{', '.join(missing)}}}.py missing"
    return {p.name: p.read_text() for p in sorted(kb.KERNELS_DIR.glob("*.py"))}


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(kb.PLUGIN_SRC)])
    env.setdefault("OMP_NUM_THREADS", "4")
    env.update(extra)
    return env


# --------------------------------------------------------------------------- the AST walkers


def _chain(node: ast.AST) -> list[str]:
    """``a.b.c`` → ``["a", "b", "c"]`` (calls and subscripts in the chain are looked through)."""
    out: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            out.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            out.append(node.id)
            break
        else:
            break
    return out[::-1]


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [] if node.level else [node.module or ""]
    if isinstance(node, ast.Call):
        name = _chain(node.func)[-1:] or [""]
        if name[0] in ("import_module", "__import__") and node.args:
            a = node.args[0]
            return [a.value] if isinstance(a, ast.Constant) and isinstance(a.value, str) else []
    return []


def triton_imports(src: str, *, outside_functions_only: bool) -> list[int]:
    """Line numbers of imports of ``triton`` (static or constant ``import_module``)."""
    lines: list[int] = []

    def visit(node: ast.AST, in_fn: bool) -> None:
        for mod in _imported(node):
            if mod.split(".")[0] == "triton" and not (outside_functions_only and in_fn):
                lines.append(getattr(node, "lineno", 0))
        for child in ast.iter_child_nodes(node):
            visit(
                child,
                in_fn or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)),
            )

    visit(ast.parse(src), False)
    return lines


FORBIDDEN_METHODS = frozenset({"item", "tolist", "cpu", "numpy", "synchronize"})
ENV_WRITERS = frozenset({"putenv", "unsetenv"})
ENV_MUTATORS = frozenset({"update", "setdefault", "pop", "popitem", "clear", "__setitem__"})


def forbidden(src: str) -> list[str]:
    """Every construct criterion 6 / N2 / N6 forbids in ``cd/kernels/`` (``"<line>: <what>"``)."""
    bad: list[str] = []
    tree = ast.parse(src)

    def flag(node: ast.AST, what: str) -> None:
        bad.append(f"{getattr(node, 'lineno', 0)}: {what}")

    def written(target: ast.AST) -> None:
        ch = _chain(target)
        if "knobs" in ch:
            flag(target, f"write to {'.'.join(ch)} (triton.knobs)")
        elif ch[:2] == ["os", "environ"] or ch[:1] == ["environ"]:
            flag(target, "write to os.environ")
        elif ch[:1] == ["torch"] and len(ch) > 1:
            flag(target, f"write to torch global {'.'.join(ch)}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("atomic_"):
                flag(node, f"atomic {node.attr}")
            if node.attr in ("autotune", "heuristics"):
                flag(node, f"triton.{node.attr}")
        if isinstance(node, ast.ImportFrom) and any(
            a.name in ("autotune", "heuristics") or a.name.startswith("atomic_") for a in node.names
        ):
            flag(node, "imports autotune/heuristics/atomic_*")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                for sub in ast.walk(t) if isinstance(t, (ast.Tuple, ast.List)) else [t]:
                    if isinstance(sub, (ast.Attribute, ast.Subscript)):
                        written(sub)
        if isinstance(node, ast.Call):
            ch = _chain(node.func)
            last = ch[-1] if ch else ""
            if isinstance(node.func, ast.Attribute) and last in FORBIDDEN_METHODS:
                flag(node, f".{last}() (host sync)")
            if last in ENV_WRITERS:
                flag(node, f"os.{last}")
            if "environ" in ch[:-1] and last in ENV_MUTATORS:
                flag(node, f"os.environ.{last}")
            if "knobs" in ch:
                flag(node, f"call on {'.'.join(ch)} (triton.knobs)")
            if ch[:1] == ["torch"] and (
                last.startswith("set_") or last == "use_deterministic_algorithms"
            ):
                flag(node, f"torch global config {'.'.join(ch)}")
            if last == "setattr" and node.args:
                tch = _chain(node.args[0])
                if "knobs" in tch or tch[:1] in (["torch"], ["os"]):
                    flag(node, f"setattr on {'.'.join(tch)}")
    return bad


def launch_option_values(src: str) -> dict[str, list[object]]:
    """Every literal value given to ``enable_fp_fusion``/``enable_reflect_ftz`` (keyword or dict entry)."""
    out: dict[str, list[object]] = {"enable_fp_fusion": [], "enable_reflect_ftz": []}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.keyword) and node.arg in out:
            v = node.value
            out[node.arg].append(v.value if isinstance(v, ast.Constant) else "<non-literal>")
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=True):
                if isinstance(k, ast.Constant) and k.value in out:
                    out[k.value].append(v.value if isinstance(v, ast.Constant) else "<non-literal>")
    return out


# --------------------------------------------------------------------------- AST rows


def test_ast_walkers_see_every_forbidden_spelling() -> None:
    """Control: each forbidden spelling is flagged on a synthetic source, a clean source is not, and a
    function-body triton import is allowed where a module-level one is not."""
    dirty = textwrap.dedent(
        """
        import os, torch
        import triton
        def f(x, p):
            import triton.language as tl
            tl.atomic_add(p, 1)
            triton.autotune(configs=[])
            triton.heuristics({})
            triton.knobs.runtime.interpret = True
            os.environ["TRITON_INTERPRET"] = "1"
            os.environ.setdefault("A", "1")
            os.putenv("A", "1")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            setattr(triton.knobs.compilation, "x", 1)
            x.item(); x.tolist(); x.cpu(); x.numpy(); torch.cuda.synchronize()
        """
    )
    got = forbidden(dirty)
    for what in (
        "atomic atomic_add",
        "triton.autotune",
        "triton.heuristics",
        "triton.knobs",
        "os.environ",
        "os.environ.setdefault",
        "os.putenv",
        "torch global torch.backends.cuda.matmul.allow_tf32",
        "torch global config torch.set_float32_matmul_precision",
        "setattr on triton.knobs.compilation",
        ".item()",
        ".tolist()",
        ".cpu()",
        ".numpy()",
        ".synchronize()",
    ):
        assert any(what in g for g in got), f"walker missed {what!r}: {got}"
    assert triton_imports(dirty, outside_functions_only=True) == [3]
    assert triton_imports(dirty, outside_functions_only=False) == [3, 5]
    clean = textwrap.dedent(
        """
        import functools, os
        @functools.cache
        def build():
            global tl
            import triton.language as tl
            return os.environ.get("TRITON_INTERPRET") == "1"
        def launch(k, x):
            k[(x.shape[0],)](x, enable_fp_fusion=False, enable_reflect_ftz=False)
        """
    )
    assert forbidden(clean) == []
    assert launch_option_values(clean) == {
        "enable_fp_fusion": [False],
        "enable_reflect_ftz": [False],
    }


def test_no_plugin_module_outside_kernels_imports_triton() -> None:
    """Criterion 7 (live guard): triton stays inside ``cd/kernels/``."""
    bad = {}
    for path in sorted(PKG.rglob("*.py")):
        if kb.KERNELS_DIR.resolve() in path.resolve().parents:
            continue
        lines = triton_imports(path.read_text(), outside_functions_only=False)
        if lines:
            bad[str(path.relative_to(PKG))] = lines
    assert bad == {}, f"triton imported outside cd/kernels/: {bad}"


def test_kernels_import_triton_only_inside_function_bodies() -> None:
    """Criterion 7 / ruling M1: no module-level triton import in ``cd/kernels/`` (not under ``try``, ``if`` or
    ``TYPE_CHECKING`` either); ``entropy.py`` imports it inside a function."""
    srcs = _kernel_sources()
    bad = {n: ln for n, s in srcs.items() if (ln := triton_imports(s, outside_functions_only=True))}
    assert bad == {}, f"module-level triton imports in cd/kernels/: {bad}"
    assert triton_imports(srcs["entropy.py"], outside_functions_only=False), (
        "cd/kernels/entropy.py never imports triton (the kernels are @triton.jit, ruling M1)"
    )


def test_kernels_have_no_atomics_autotune_global_config_or_host_sync() -> None:
    """Criterion 6 / N2 / N6 over every ``cd/kernels/*.py``."""
    bad = {n: f for n, s in _kernel_sources().items() if (f := forbidden(s))}
    assert bad == {}, f"forbidden in cd/kernels/: {json.dumps(bad, indent=1)}"


def test_kernel_launches_turn_fp_fusion_and_reflect_ftz_off() -> None:
    """Landmines (task notes): FMA contraction and libdevice FTZ are on by default; both are passed False, per
    launch, and never True."""
    vals: dict[str, list[object]] = {"enable_fp_fusion": [], "enable_reflect_ftz": []}
    for s in _kernel_sources().values():
        for k, v in launch_option_values(s).items():
            vals[k] += v
    for k, v in vals.items():
        assert v and all(x is False for x in v), f"{k} values in cd/kernels/: {v} (need False only)"


# --------------------------------------------------------------------------- subprocess rows

#: triton unavailable, every attempt recorded (``kb.TRITON_BLOCKER``; ruling K9: ``find_spec`` returns None
#: instead of raising, which torch's own probe tolerates)
_BLOCKER = kb.TRITON_BLOCKER + "\nimport importlib, json, pkgutil, sys\n"


_IMPORT_AND_RUN = textwrap.dedent(
    """
    import astra_vllm_plugin.cd as cd
    names = sorted(m.name for m in pkgutil.walk_packages(cd.__path__, "astra_vllm_plugin.cd."))
    for want in ("astra_vllm_plugin.cd.kernels", "astra_vllm_plugin.cd.kernels.backend",
                 "astra_vllm_plugin.cd.kernels.entropy"):
        if want not in names:
            names.append(want)
    imported, failed = [], {}
    for n in names:
        try:
            importlib.import_module(n)
            imported.append(n)
        except Exception as exc:
            failed[n] = f"{type(exc).__name__}: {exc}"
    after_import = list(attempts)
    from tests.cd.unit import _frozen_t0077 as fz
    from tests.cd.unit import _frozen_t0077b as kb
    inst = fz.build_inst("s91_b128_nfa")
    ref = kb.marginals_on("frozen", inst)
    diffs = {}
    for label, kw in (("default", {}), ("kernels=True", {"kernels": True}),
                      ("kernels=False", {"kernels": False})):
        try:
            diffs[label] = kb.field_diffs(ref, kb.marginals_on("live", inst, "cpu", **kw))
        except Exception as exc:
            diffs[label] = f"{type(exc).__name__}: {exc}"
    print("ASTRA_T0077B_IMPORTS " + json.dumps({
        "imported": imported, "failed": failed, "after_import": after_import,
        "attempts": list(attempts), "diffs": diffs,  # snapshot before blocker_check
        "blocked": blocker_check() if "blocker_check" in globals() else None,
        "triton_loaded": sorted(m for m in sys.modules if m.split(".")[0] == "triton"),
    }))
    """
)


def _run(code: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_T0077B_IMPORTS ")), None
    )
    assert proc.returncode == 0 and line is not None, (
        f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    )
    out: dict[str, object] = json.loads(line.split(" ", 1)[1])
    return out


def test_every_cd_module_imports_and_cpu_marginals_run_with_triton_blocked() -> None:
    """Criterion 7: no triton on the host (a meta-path blocker) → every ``cd`` module still imports and CPU
    ``constrained_marginals`` runs, equal to the branch base; nothing on the CPU path even tries triton."""
    out = _run(_BLOCKER + "import importlib, json, pkgutil, sys\n" + _IMPORT_AND_RUN)
    assert out["failed"] == {}, f"cd modules that do not import without triton: {out['failed']}"
    assert out["after_import"] == [], f"importing cd tried to import triton: {out['after_import']}"
    assert out["diffs"] == {"default": {}, "kernels=True": {}, "kernels=False": {}}, out["diffs"]
    assert out["attempts"] == [], f"the CPU path tried to import triton: {out['attempts']}"
    assert out["blocked"] == {"find_spec_none": True, "import_raises": True}, out["blocked"]


def test_cpu_marginals_with_triton_installed_leave_triton_unimported() -> None:
    """Criterion 5: triton is installed (the ``cd`` group), yet importing every ``cd`` module and running CPU
    marginals (``kernels=True`` included) never puts ``triton`` in ``sys.modules``."""
    out = _run("import importlib, json, pkgutil, sys\nattempts = []\n" + _IMPORT_AND_RUN)
    assert out["failed"] == {}, out["failed"]
    assert out["diffs"] == {"default": {}, "kernels=True": {}, "kernels=False": {}}, out["diffs"]
    assert out["triton_loaded"] == [], (
        f"triton in sys.modules after CPU work: {out['triton_loaded']}"
    )


def test_kernels_package_imports_nothing_heavy() -> None:
    """``cd/kernels/__init__.py`` imports nothing heavy: neither torch nor triton after importing the package."""
    _kernel_sources()
    code = textwrap.dedent(
        """
        import json, sys
        import astra_vllm_plugin.cd.kernels
        print("ASTRA_T0077B_IMPORTS " + json.dumps(sorted(
            m for m in sys.modules if m.split(".")[0] in ("torch", "triton", "numpy"))))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/",
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    heavy = json.loads(proc.stdout.strip().splitlines()[-1].split(" ", 1)[1])
    assert heavy == [], f"import astra_vllm_plugin.cd.kernels loaded {heavy[:10]}"


# --------------------------------------------------------------------------- packaging


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(out)],
        cwd=PLUGIN_DIR,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, f"uv build --wheel --offline failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = sorted(out.glob("astra_vllm_plugin-*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as zf:
        return set(zf.namelist())


def test_wheel_contains_every_cd_kernels_module(wheel_names: set[str]) -> None:
    """Criterion 7: what is not in the wheel does not exist inside the engine."""
    on_disk = sorted(
        p.name for p in (PLUGIN_DIR / "astra_vllm_plugin" / "cd" / "kernels").glob("*.py")
    )
    for m in kb.KERNEL_MODULES:
        assert f"{m}.py" in on_disk, f"T-0077b not implemented: cd/kernels/{m}.py"
    missing = [n for n in on_disk if f"astra_vllm_plugin/cd/kernels/{n}" not in wheel_names]
    assert missing == [], f"the wheel lacks cd/kernels/{missing}"


PLUGIN_PYPROJECT = "plugin/astra_vllm_plugin/pyproject.toml"
#: T-0077b's merge commit b53457a; BASE = ``git merge-base b53457a^1 b53457a^2`` = ``kb.FROZEN_COMMIT`` (2d90189),
#: TIP = ``b53457a^2`` (T-0094b criterion 2). The plugin pyproject has one revision in history (the 404577a
#: scaffold), so the power of the byte row is the shared helper row (ruling P6) and the content row carries the intent.
T0077B_MERGE = "b53457a"
T0077B_BASE = "2d90189"
T0077B_TIP = "ac2befb"


def _mentions_triton(text: str) -> bool:
    """Criterion 7's content predicate: the plugin pyproject names ``triton`` anywhere."""
    return "triton" in text


def test_plugin_pyproject_is_byte_identical_to_the_branch_base() -> None:
    """Criterion 7: no triton dependency for the plugin (the image provides it; ``--no-deps`` install): T-0077b left
    the plugin pyproject unchanged between its BASE and TIP (T-0094b: a closed range, never the working tree), and
    the BASE blob names no triton. The "one-byte change is reported" half is
    ``tests/unit/test_guards_use_fixed_refs.py::test_changed_reports_every_modified_deleted_and_added_file_under_its_paths_only``."""
    assert T0077B_BASE == kb.FROZEN_COMMIT
    gr.prove_range(T0077B_BASE, T0077B_TIP, T0077B_MERGE)
    assert gr.changed(T0077B_BASE, T0077B_TIP, PLUGIN_PYPROJECT) == []
    blob = subprocess.run(
        ["git", "show", f"{T0077B_BASE}:{PLUGIN_PYPROJECT}"], cwd=REPO, capture_output=True
    )
    assert blob.returncode == 0, blob.stderr
    assert not _mentions_triton(blob.stdout.decode())


def test_the_plugin_pyproject_declares_no_triton() -> None:
    """T-0094b criterion 10 (the intent, forever): the current plugin pyproject declares no ``triton`` dependency (a
    property of the file, no git reference)."""
    assert not _mentions_triton((REPO / PLUGIN_PYPROJECT).read_text())


def test_the_no_triton_check_fails_on_an_added_triton() -> None:
    """T-0094b criterion 10 (adversarial): the same check on the current text with ``"triton"`` appended fails."""
    text = (REPO / PLUGIN_PYPROJECT).read_text()
    assert _mentions_triton(text + "triton")


def test_host_cd_group_pins_triton_3_7_1() -> None:
    """Criterion 7: the host ``cd`` dependency group pins ``triton==3.7.1`` (the image's version)."""
    groups = tomllib.loads((REPO / "pyproject.toml").read_text())["dependency-groups"]
    pins = [d for d in groups["cd"] if d.replace(" ", "").lower().startswith("triton")]
    assert pins == [f"triton=={kb.TRITON_VERSION}"], pins
    others = [g for g, deps in groups.items() if g != "cd" and any("triton" in d for d in deps)]
    assert others == [], f"triton outside the cd group: {others}"


def test_host_triton_version_is_3_7_1() -> None:
    """Criterion 7: the installed host triton is the pinned one (the interpreter rows run on it)."""
    import importlib.metadata as md

    assert md.version("triton") == kb.TRITON_VERSION
    proc = subprocess.run(
        [sys.executable, "-c", "import triton; print(triton.__version__)"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == kb.TRITON_VERSION
