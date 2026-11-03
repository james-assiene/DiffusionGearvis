"""T-0063 criterion 3: the vendored diffgemma-fa compiler is auditable.

The expected upstream file list and sha256s are read from diffgemma-fa itself (``git archive 27c7d0d
diffgemma_fa/compile``, 2026-09-29), never from the vendored copy, so a file that drifted *and* had its
PROVENANCE hash updated to match is still caught.

PROVENANCE.md format these rows rely on (stated here because the task does not fix one): one line per
vendored file -- a markdown table row is fine -- carrying the file's path relative to ``dgfa_compile/``
(e.g. ``tasks/sudoku.py``), its 64-hex sha256, and the word ``unmodified`` or ``modified``. Every modified
file is named again below a heading containing "Deviation", with its reason.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

from tests.cd._compile_support import REPO

VENDOR = REPO / "local_astra" / "decoding" / "_vendor"
TREE = VENDOR / "dgfa_compile"
PROVENANCE = VENDOR / "PROVENANCE.md"
DGFA_COMMIT = "27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03"

#: sha256 of every file of diffgemma-fa ``diffgemma_fa/compile/`` at 27c7d0d (upstream, not the copy).
UPSTREAM_SHA256 = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "automaton.py": "66267e2b5f2e68ff37a00eb87c008d0289b760863659755a8dc5a6e4c4f9a2e7",
    "bfcl_data.py": "655fcfd8e21926a098e9c60630bd8534b28bd42b0425abb68eb97dada24a1f73",
    "classes.py": "59846f304f858eaa2c40c2e09f6b370d2561c8a90f0254fd89578b496ee0fd03",
    "lift.py": "ce1d2da0ddd911ad0f2f66b2a612c6bd07efbfa4a87db51db80f9469e3ec29cf",
    "minimize.py": "bb9a74bc9456602284ecb3a7b5ea855dec2baf6876f48d94bee5676213f3421d",
    "pipeline.py": "6c646bd0ff5a56626cb23ae964da3b6234d773ffe36b80035dca91a68b600a69",
    "schema.py": "d242d567157c2e714626d6064ae2cde93eb2358d0dd5bd0f5ffeb962e14a19ca",
    "tasks/__init__.py": "ae2eaad9ada6c6975cf49832380695393dcdcfd0aa2728afdcda20ea9b750cfd",
    "tasks/bfcl.py": "98b302c5fc73d3f5037c0fb5c263dc73c2533ddcdf262c9148d6483a456c8fef",
    "tasks/bfcl_python.py": "abbb840ccf7abb1fe103d9114801486545f4e4409879b591fbb442687fab7ad2",
    "tasks/countdown.py": "6f4fd10f8a805caa0aa03ad8dfdf61addd83809ad3e8f166ea90ca4f0a35f6d0",
    "tasks/grammars.py": "6afdab1999a65a7f3af03e99a6d124ba66a0b29d09856ed5ffbcc63263d845f5",
    "tasks/sudoku.py": "59b413d89552693e6f5cd17d9c10bfbe8003d7010fcd57a67835fdf5377388f7",
    "validate.py": "044ca20ee2d27c3978bb7d8e33853f7d5eb46909f2b4d1491beaaa47b3e7ddbc",
    "vocab.py": "ab3b96bb72a77240fb4b182fffdf137e78349abebfa468afc4a4b367c9babe13",
}

_ROW = re.compile(
    r"(?P<path>(?:tasks/)?[A-Za-z_][\w]*\.py)\b.*?\b(?P<sha>[0-9a-f]{64})\b.*?\b(?P<status>unmodified|modified)\b"
    r"|(?P<path2>(?:tasks/)?[A-Za-z_][\w]*\.py)\b.*?\b(?P<status2>unmodified|modified)\b.*?\b(?P<sha2>[0-9a-f]{64})\b"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vendored_files() -> dict[str, Path]:
    if not TREE.is_dir():
        pytest.fail(f"T-0063 not implemented: {TREE.relative_to(REPO)} does not exist")
    return {
        p.relative_to(TREE).as_posix(): p
        for p in sorted(TREE.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def _provenance_text() -> str:
    if not PROVENANCE.is_file():
        pytest.fail(f"T-0063 not implemented: {PROVENANCE.relative_to(REPO)} does not exist")
    return PROVENANCE.read_text()


def _provenance_rows() -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in _provenance_text().splitlines():
        m = _ROW.search(line)
        if m is None:
            continue
        path = m.group("path") or m.group("path2")
        sha = m.group("sha") or m.group("sha2")
        status = m.group("status") or m.group("status2")
        assert path not in rows, f"PROVENANCE lists {path} twice"
        rows[path] = (sha, status)
    return rows


def test_vendored_tree_is_exactly_the_upstream_compile_package() -> None:
    files = set(_vendored_files())
    assert files == set(UPSTREAM_SHA256), (
        f"missing from the vendored tree: {sorted(set(UPSTREAM_SHA256) - files)}; "
        f"not in diffgemma-fa compile/ at 27c7d0d: {sorted(files - set(UPSTREAM_SHA256))}"
    )


def test_provenance_names_the_commit() -> None:
    text = _provenance_text()
    assert DGFA_COMMIT[:7] in text, "PROVENANCE.md does not name diffgemma-fa commit 27c7d0d"


def test_provenance_lists_every_vendored_file_with_its_current_sha256() -> None:
    rows = _provenance_rows()
    files = _vendored_files()
    unlisted = sorted(set(files) - set(rows))
    assert not unlisted, f"PROVENANCE.md has no (path, sha256, status) line for {unlisted}"
    stale = {p: (rows[p][0], _sha(f)) for p, f in files.items() if rows[p][0] != _sha(f)}
    assert not stale, f"PROVENANCE sha256 != file on disk (recorded, actual): {stale}"


def test_unmodified_files_have_not_drifted_from_upstream() -> None:
    """The drift check of criterion 3: every file marked unmodified hashes to the 27c7d0d original."""
    rows = _provenance_rows()
    files = _vendored_files()
    unmodified = [p for p, (_, status) in rows.items() if status == "unmodified"]
    assert unmodified, (
        "PROVENANCE marks no file unmodified; vendoring should keep most files verbatim"
    )
    drift = {
        p: _sha(files[p])
        for p in unmodified
        if p not in files or _sha(files[p]) != UPSTREAM_SHA256.get(p)
    }
    assert not drift, f"marked unmodified but differ from diffgemma-fa 27c7d0d: {drift}"


def test_every_changed_file_is_declared_modified_with_a_deviation() -> None:
    rows = _provenance_rows()
    files = _vendored_files()
    changed = sorted(p for p, f in files.items() if _sha(f) != UPSTREAM_SHA256.get(p))
    undeclared = [p for p in changed if rows.get(p, ("", ""))[1] != "modified"]
    assert not undeclared, f"differ from upstream but not marked modified: {undeclared}"
    text = _provenance_text()
    m = re.search(r"^#+[^\n]*[Dd]eviation[^\n]*$", text, re.MULTILINE)
    assert m is not None, "PROVENANCE.md has no Deviations section"
    deviations = text[m.end() :]
    unexplained = [p for p in changed if p not in deviations]
    assert not unexplained, (
        f"modified files with no entry under the Deviations heading: {unexplained}"
    )


def test_the_sentencepiece_tokenizer_deviation_is_recorded() -> None:
    """vocab.py cannot stay verbatim: upstream imports the JAX ``gemma`` package for its SentencePiece
    tokenizer, which this port replaces with the HF ``tokenizer.json`` (criterion 3's named deviation)."""
    rows = _provenance_rows()
    assert rows.get("vocab.py", ("", ""))[1] == "modified", rows.get("vocab.py")
    text = _provenance_text()
    assert "tokenizer.json" in text and re.search(r"sentencepiece|SentencePiece", text), (
        "PROVENANCE does not record the SentencePiece -> HF tokenizer.json deviation"
    )


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
    return roots


@pytest.mark.parametrize("forbidden", ["diffgemma_fa", "gemma", "jax", "flax", "sentencepiece"])
def test_vendored_code_imports_nothing_from_the_jax_stack(forbidden: str) -> None:
    offenders = {
        rel: sorted(
            r for r in _imported_roots(f) if r == forbidden or r.startswith(forbidden + ".")
        )
        for rel, f in _vendored_files().items()
        if f.suffix == ".py"
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"vendored modules still import {forbidden}: {offenders}"


def test_the_vendored_package_imports_under_its_new_path() -> None:
    from tests.cd._compile_support import VENDOR_PKG, need

    for mod in (
        "automaton",
        "classes",
        "lift",
        "minimize",
        "pipeline",
        "schema",
        "validate",
        "vocab",
    ):
        need(f"{VENDOR_PKG}.{mod}")
    for mod in ("grammars", "sudoku"):
        need(f"{VENDOR_PKG}.tasks.{mod}")
