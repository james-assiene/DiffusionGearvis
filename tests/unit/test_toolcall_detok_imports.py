"""T-0088a criterion 4 (negative): ``toolcall/detok.py`` and ``toolcall/special_tokens.py`` are stdlib only.

A new module so that ``tests/unit/test_toolcall_imports.py`` stays unedited (its AST fence already globs every file
of ``local_astra/toolcall/``). Rows:

- a fresh interpreter imports both modules and then has none of ``tokenizers``, ``local_astra.decoding``, torch,
  numpy, outlines or outlines_core in ``sys.modules``;
- an AST row: every import in the two files is a stdlib module (``sys.stdlib_module_names``), ``__future__``, or
  the sibling ``local_astra.toolcall.special_tokens`` (the detokeniser may read the frozen table);
- the frozen table and the speech gate need no file: in a fresh interpreter whose HF cache and home point at an
  empty directory, ``SPECIAL_TOKENS`` has its 24 literals and the gate trims ``<|tool_call>``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests.unit._detok_support import DETOK, REPO, SPECIAL

FORBIDDEN = ("tokenizers", "local_astra.decoding", "torch", "numpy", "outlines", "outlines_core")
FILES = {
    DETOK: REPO / "local_astra" / "toolcall" / "detok.py",
    SPECIAL: REPO / "local_astra" / "toolcall" / "special_tokens.py",
}
ALLOWED_LOCAL = {"local_astra.toolcall.special_tokens"}


def _fresh(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_a_fresh_interpreter_imports_both_modules_without_any_forbidden_module() -> None:
    proc = _fresh(
        f"""
        import importlib, json, sys
        for m in {[DETOK, SPECIAL]!r}:
            importlib.import_module(m)
        forbidden = {list(FORBIDDEN)!r}
        loaded = sorted(n for n in sys.modules if any(n == f or n.startswith(f + ".") for f in forbidden))
        print(json.dumps(loaded))
        """
    )
    assert proc.returncode == 0, (
        f"T-0088a: the modules do not import in a fresh interpreter:\n{proc.stderr[-2000:]}"
    )
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], f"importing detok/special_tokens pulled in {loaded}"


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                mod = ".".join(["local_astra", "toolcall", *([node.module] if node.module else [])])
                if node.module is None:
                    out.extend((node.lineno, f"{mod}.{a.name}") for a in node.names)
                    continue
            else:
                mod = node.module or ""
            out.append((node.lineno, mod))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.append((node.lineno, arg.value))
                else:
                    out.append((node.lineno, "<dynamic import>"))
    return out


def test_every_import_of_both_files_is_stdlib() -> None:
    missing = [m for m, p in FILES.items() if not p.is_file()]
    assert not missing, f"T-0088a not implemented: {missing} do not exist"
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad = []
    for mod, path in FILES.items():
        for lineno, name in _imports(path):
            if name in ALLOWED_LOCAL and mod == DETOK:
                continue
            if name.split(".")[0] not in stdlib:
                bad.append(f"{path.relative_to(REPO)}:{lineno}: {name}")
    assert not bad, "non-stdlib imports:\n" + "\n".join(bad)


def test_the_tables_and_the_gate_need_no_tokenizer_file(tmp_path: Path) -> None:
    env = dict(os.environ)
    empty = tmp_path / "empty"
    empty.mkdir()
    env.update(HF_HUB_CACHE=str(empty / "hub"), HF_HOME=str(empty), HOME=str(empty))
    proc = _fresh(
        f"""
        import json
        from {SPECIAL} import SPECIAL_TOKENS
        from local_astra.conductor.speech_gate import SpeechGate
        from local_astra.contracts.llm import Commit
        g = SpeechGate()
        c = Commit(request_id="r", canvas_idx=0, segment="answer", text="I will.<|tool_call>call:x{{}}<tool_call|>")
        out = list(g.feed(c)) + list(g.close())
        print(json.dumps([len(SPECIAL_TOKENS), [s.text for s in out]]))
        """,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    n, texts = json.loads(proc.stdout.strip().splitlines()[-1])
    assert n == 24
    assert "".join(texts) == "I will."
