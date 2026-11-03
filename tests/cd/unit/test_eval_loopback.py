"""T-0070 criterion 12 (negative): the runner refuses any non-loopback base URL before opening a socket
or resolving a name, and nothing under ``local_astra/eval/cd`` or ``local_astra/bench/b13_cd.py`` imports
torch (by AST and by a subprocess ``sys.modules`` check). Regime: CPU.
"""

from __future__ import annotations

import ast
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.cd import _eval_support as es

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "local_astra" / "eval" / "cd"
B13 = REPO / "local_astra" / "bench" / "b13_cd.py"


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:8000/v1",
        "http://192.168.1.20:8000/v1",
        "http://10.0.0.5:8000/v1",
        "http://example.com/v1",
        "https://api.openai.com/v1",
        "http://127.0.0.1.nip.io:8000/v1",  # adversarial: a loopback-looking public name
        "http://localhost.evil.test:8000/v1",
    ],
)
def test_non_loopback_base_url_is_refused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    r = es.runner()
    calls: list[object] = []

    def spy(*a: object, **k: object) -> object:
        calls.append(a)
        raise AssertionError(f"network touched for {url}: {a}")

    monkeypatch.setattr(socket, "getaddrinfo", spy)
    monkeypatch.setattr(socket, "create_connection", spy)
    repo, head = es.clean_repo(tmp_path)
    with pytest.raises(ValueError):
        r.run(
            r.RunConfig(
                task="sudoku",
                cut="test",
                arm="unconstrained",
                base_url=url,
                engine_log=tmp_path / "e.log",
                out_dir=tmp_path,
                repo=repo,
                expected_commit=head,
                limit=1,
            )
        )
    assert calls == []


def test_loopback_base_url_runs(tmp_path: Path) -> None:
    """Companion: the guard is not too wide (127.0.0.1 is served)."""
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="unconstrained", limit=1)
        assert len(srv.requests) == 1
    assert res.exit_code == 0, res.stopped


def _modules() -> list[Path]:
    if not EVAL.is_dir():
        pytest.fail("T-0070 not implemented: local_astra/eval/cd does not exist")
    files = [p for p in EVAL.rglob("*.py")]
    return [*files, B13] if B13.is_file() else files


def test_no_torch_import_by_ast() -> None:
    bad = []
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [
                f"{path.relative_to(REPO)}: {n}"
                for n in names
                if n == "torch" or n.startswith("torch.")
            ]
    assert bad == []


def test_importing_the_harness_leaves_torch_out_of_sys_modules() -> None:
    mods = sorted(
        "local_astra." + ".".join(p.relative_to(REPO / "local_astra").with_suffix("").parts)
        for p in _modules()
        if p.name != "__init__.py" and "_vendor" not in p.parts
    )
    code = (
        "import importlib, sys\n"
        + "".join(f"importlib.import_module({m!r})\n" for m in mods)
        + ("print(sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')))\n")
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout[-500:]
    assert "local_astra.eval.cd.runner" in mods
