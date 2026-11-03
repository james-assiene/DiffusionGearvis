"""T-0111c criterion 17 (negative): nothing outside the new package moves, and ``live`` is still refused.

Green at BASE and must stay green on the branch (no ``pending`` marker).

- The unowned paths (conductor, config, cli, contracts, toolcall, the fakes, ``services/versions.lock``) are
  unchanged, with no file added or removed, across T-0111c's own closed merge range ``T0111C_BASE..T0111C_TIP``
  (.claude/rules/tests.md "pin both ends", T-0094): M = c34fe94, BASE = ``git merge-base M^1 M^2`` = 6d32447 (the
  branch was cut there, a descendant of the declared base 02ccff5, which is checked to be its ancestor and equal to it
  on these paths), TIP = ``M^2`` = 464b140. The working tree is never read, so a later task's legitimate edit (T-0127's
  ``conductor/ui_gate.py``) cannot redden it. Before merge the row compared ``BASE`` with the working tree; it was
  closed after the merge (T-0111c-close-guard).
  The power row runs the same predicate over a range known to change one of the paths (the human's P2 commit).
- ``tools = "live"`` was refused with BASE's message, pinned as a literal, at T-0111c's TIP (T-0111g re-scope R6:
  read from ``git show 464b140:local_astra/config.py``, never the working tree, since T-0111g makes ``live`` a mode).
- The tools sha is BASE's: the sha256 of ``tools_v1.json`` at BASE, pinned as a literal and re-derived from the blob.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from local_astra.toolcall import tools
from tests import _git_range as gr

#: T-0111c's BASE: the human's P1/P2 commit (docs/tasks/T-0111c.md, Rulings 2026-10-20).
BASE = "02ccff5"
#: BASE's parent: ``P2_PARENT..BASE`` changed ``services/versions.lock`` (the ``[browser]`` block).
P2_PARENT = "82da380"
UNOWNED = (
    "local_astra/conductor",
    "local_astra/config.py",
    "local_astra/cli.py",
    "local_astra/contracts",
    "local_astra/toolcall",
    "tests/fakes",
    "services/versions.lock",
)
LIVE_REFUSAL = (
    "tools = 'live' is refused: only 'off' and 'dry_run' exist; executing tool calls ('live') "
    "is a v1 non-goal (docs/01 §3, T-0080d)"
)
TOOLS_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"


#: T-0111c's closed merge range (tests.md "pin both ends", T-0094): M = c34fe94, BASE = ``git merge-base M^1 M^2``,
#: TIP = ``M^2``. Full SHAs, never a moving ref.
T0111C_MERGE = "c34fe94f132d1a08498a2b3a0da9a73289aa001f"
T0111C_BASE = "6d3244719a075ddf50fde2ab808498d038588c56"
T0111C_TIP = "464b140a12873c784b0d23067fe69f28b66d49dc"
#: main when this row was closed (the merge itself); ``prove_range``'s "later" premise holds reflexively. No power row
#: reads ``BASE..LATER`` here: the power row is ``P2_PARENT..BASE``.
T0111C_LATER = "c34fe94f132d1a08498a2b3a0da9a73289aa001f"


def test_unowned_files_equal_base() -> None:
    """Criterion 17: nothing under the unowned paths changed, was added or was removed across
    ``T0111C_BASE..T0111C_TIP``; the declared BASE 02ccff5 is an ancestor of that BASE and equal to it on these
    paths."""
    gr.require_commits(BASE, P2_PARENT)
    gr.prove_range(T0111C_BASE, T0111C_TIP, T0111C_MERGE, later=T0111C_LATER)
    assert gr.is_ancestor(BASE, T0111C_BASE), f"premise: {BASE} is an ancestor of {T0111C_BASE}"
    assert gr.changed(BASE, T0111C_BASE, *UNOWNED) == [], (
        f"premise: {BASE} and {T0111C_BASE} hold the same unowned files"
    )
    added_or_removed = gr.changed(T0111C_BASE, T0111C_TIP, *UNOWNED, diff_filter="ADR")
    assert added_or_removed == [], (
        f"T-0111c criterion 17: files added, removed or renamed under the unowned paths in "
        f"{T0111C_BASE[:7]}..{T0111C_TIP[:7]}: {added_or_removed}"
    )
    assert gr.changed(T0111C_BASE, T0111C_TIP, *UNOWNED) == [], (
        f"T-0111c criterion 17: unowned files differ in {T0111C_BASE[:7]}..{T0111C_TIP[:7]}"
    )


def test_the_scope_predicate_has_power() -> None:
    """The same path set over ``P2_PARENT..BASE`` reports the lock file the human changed."""
    gr.require_commits(BASE, P2_PARENT)
    assert gr.is_ancestor(P2_PARENT, BASE)
    assert gr.changed(P2_PARENT, BASE, *UNOWNED) == ["services/versions.lock"]


def config_at(sha: str, tmp_path: Path) -> ModuleType:
    """``local_astra/config.py`` as it was at ``sha`` (``git show``), loaded in isolation under a unique module name,
    never the working tree (T-0111g re-scopes R4-R6). Its only project import is ``local_astra.contracts.base``,
    which T-0111g B17 keeps byte-identical."""
    gr.require_commits(sha)
    blob = gr._git("show", f"{sha}:local_astra/config.py")
    assert blob.returncode == 0, blob.stderr
    imports = [
        line
        for line in blob.stdout.splitlines()
        if line.startswith(("from local_astra", "import local_astra"))
    ]
    assert imports == ["from local_astra.contracts.base import Contract"], imports
    path = tmp_path / f"config_at_{sha[:12]}.py"
    path.write_text(blob.stdout)
    name = f"_config_at_{sha[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves the postponed annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_live_still_refused(tmp_path: Path) -> None:
    """T-0111g re-scope R6: a historical claim -- ``live`` was refused with BASE's message at T-0111c's TIP. Re-pinned
    against ``git show 464b140:local_astra/config.py`` loaded in isolation; ``live`` is a mode from T-0111g on."""
    config = config_at(T0111C_TIP, tmp_path)
    with pytest.raises(ValidationError) as exc:
        config.Profile(name="lean", tools="live")
    assert LIVE_REFUSAL in str(exc.value), str(exc.value)


def test_tools_sha_unchanged() -> None:
    gr.require_commits(BASE)
    blob = gr._git("show", f"{BASE}:local_astra/toolcall/tools_v1.json")
    assert blob.returncode == 0, blob.stderr
    assert hashlib.sha256(blob.stdout.encode()).hexdigest() == TOOLS_SHA
    assert tools.load().version == TOOLS_SHA
