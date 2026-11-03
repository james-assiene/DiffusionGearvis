"""T-0156 group R: the ``diffusion_gearvis`` console command beside ``astra``, project name ``diffusion-gearvis``.

Every expected value here is written down by hand from the task file (R1-R3, R6), not read from the implementation:
the two script entries, the project name, the unchanged version ``0.0.1``, the eleven subcommand names of ``cli.py``
at the base, and the directory names that must keep ``local_astra``.

The ``uv.lock`` row compares two literal commits, ``ae7b7a4`` (where T-0156-r branched) and ``41f1d12`` (the
implementation tip), never the working tree: a later task that adds a dependency legitimately changes ``uv.lock``.
Its comparator keeps its power through separate rows that feed it mutants of the base lock.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, supervisor
from local_astra.memory import store
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]

#: The commit T-0156 group R branched from (main when the task file landed, 2026-10-25). Literal on purpose: a moving
#: ref (``main``) would already hold the rename once it merges and the row would pass vacuously (.claude/rules/tests.md).
T0156_BASE = "ae7b7a4"
#: The implementation tip of T-0156 group R (41f1d12, branch T-0156-r-command, 2026-10-25): the other literal end of
#: the R2 lock range, so the row neither decays when a later task edits ``uv.lock`` nor goes vacuous after the merge.
T0156_TIP = "41f1d12"

ENTRY = "local_astra.cli:main"
#: ``cli.py``'s subcommands at ``T0156_BASE`` (``sub.add_parser`` calls, in order). R3/R6: they do not change.
SUBCOMMANDS_AT_BASE = (
    "session",
    "up",
    "doctor",
    "down",
    "status",
    "hud",
    "bench",
    "eval",
    "forget",
    "fetch-models",
    "config",
)


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


# --- R1 --------------------------------------------------------------------------------------------------------


def test_r1_both_console_scripts_point_at_the_same_main() -> None:
    """R1: ``diffusion_gearvis`` is added and ``astra`` kept, both ``local_astra.cli:main``; no third script."""
    scripts = _pyproject()["project"].get("scripts", {})
    assert scripts == {"astra": ENTRY, "diffusion_gearvis": ENTRY}, scripts


def test_r1_project_name_is_diffusion_gearvis_and_the_version_is_unchanged() -> None:
    """R1/R2: ``[project] name = "diffusion-gearvis"``, version still ``0.0.1``, description not "Astra-style"."""
    project = _pyproject()["project"]
    assert project["name"] == "diffusion-gearvis", project["name"]
    assert project["version"] == "0.0.1", "R2: the rename is not a version change"
    assert "astra-style" not in project["description"].lower(), project["description"]


def test_r1_the_package_directory_stays_local_astra() -> None:
    """R1: group R moves no package (that is group B); the wheel still ships ``local_astra``. Passes at the base and
    must keep passing through group R; group B's test-author replaces it."""
    assert (REPO / "local_astra" / "cli.py").is_file()
    assert not (REPO / "diffusion_gearvis").exists(), (
        "the package move is group B (v0.1.1), not group R"
    )
    wheel = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["local_astra"], wheel


# --- R2 --------------------------------------------------------------------------------------------------------


def _pkg_key(pkg: dict[str, Any]) -> tuple[str, str, str]:
    return (pkg["name"], pkg.get("version", ""), repr(sorted(pkg.get("source", {}).items())))


def _canonical(lock: dict[str, Any]) -> dict[str, Any]:
    """The lock with its package list as a set keyed by (name, version, source): uv sorts packages by name, so the
    renamed root legitimately moves from under ``l`` to under ``d``; order is not content."""
    out = {k: v for k, v in lock.items() if k != "package"}
    out["package"] = {_pkg_key(p): p for p in lock["package"]}
    assert len(out["package"]) == len(lock["package"]), (
        "duplicate (name, version, source) in the lock"
    )
    return out


def _is_root(pkg: dict[str, Any]) -> bool:
    return bool(pkg.get("source") == {"editable": "."})


def _name_only_rename_problems(base: dict[str, Any], tip: dict[str, Any]) -> list[str]:
    """Every way ``tip`` differs from ``base`` other than the root package renamed ``local-astra`` ->
    ``diffusion-gearvis``; empty when the change is name-only.

    The expected lock is ``base`` with that one field edited, so any other change -- a version bump, a re-resolved
    marker, a dropped or altered wheel hash, an added or removed package -- is reported.
    """
    roots = [p for p in base["package"] if _is_root(p)]
    if [(p["name"], p["version"]) for p in roots] != [("local-astra", "0.0.1")]:
        return [
            f"base root is not local-astra 0.0.1: {[(p['name'], p.get('version')) for p in roots]}"
        ]
    expected = copy.deepcopy(base)
    for pkg in expected["package"]:
        if _is_root(pkg):
            pkg["name"] = "diffusion-gearvis"
    want, got = _canonical(expected), _canonical(tip)
    problems: list[str] = []
    if set(got) != set(want):
        problems.append(f"top-level keys changed: {sorted(set(got) ^ set(want))}")
    problems += [
        f"uv.lock top-level {key!r} changed"
        for key in want
        if key != "package" and key in got and got[key] != want[key]
    ]
    missing = sorted(set(want["package"]) - set(got["package"]))
    extra = sorted(set(got["package"]) - set(want["package"]))
    if missing or extra:
        problems.append(f"package set changed: missing {missing[:5]}, extra {extra[:5]}")
    changed = [
        k
        for k in want["package"]
        if k in got["package"] and got["package"][k] != want["package"][k]
    ]
    if changed:
        problems.append(f"package entries changed beyond the root name: {changed[:5]}")
    return problems


def _lock_at(rev: str) -> dict[str, Any]:
    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:uv.lock"],
        capture_output=True,
        text=True,
        check=True,
    )
    return tomllib.loads(shown.stdout)


def test_r2_uv_lock_differs_from_the_base_only_in_the_root_package_name() -> None:
    """R2, closed range ``T0156_BASE..T0156_TIP``: same package set, versions, sources, hashes, markers and top-level
    fields; only the root's name moves ``local-astra`` -> ``diffusion-gearvis``. Both ends are literals, so a later
    task that legitimately changes ``uv.lock`` cannot turn this red, and the merge cannot make it vacuous."""
    gr.require_commits(T0156_BASE, T0156_TIP)
    assert gr.is_ancestor(T0156_BASE, T0156_TIP), f"{T0156_BASE} is not an ancestor of {T0156_TIP}"
    problems = _name_only_rename_problems(_lock_at(T0156_BASE), _lock_at(T0156_TIP))
    assert not problems, problems


def _base_lock_or_skip() -> dict[str, Any]:
    gr.require_commits(T0156_BASE)
    return _lock_at(T0156_BASE)


def _renamed_and_reordered(base: dict[str, Any]) -> dict[str, Any]:
    """The base lock as a name-only relock would leave it: root renamed, package list re-sorted by the new name."""
    tip = copy.deepcopy(base)
    for pkg in tip["package"]:
        if _is_root(pkg):
            pkg["name"] = "diffusion-gearvis"
    tip["package"].sort(key=lambda p: p["name"])
    return tip


def test_r2_comparator_accepts_a_name_only_relock() -> None:
    """Power row, accepting direction: the rename plus uv's re-sort by name is not a difference (a comparator that
    compared the package list in order would refuse every real relock)."""
    base = _base_lock_or_skip()
    tip = _renamed_and_reordered(base)
    assert tip["package"] != base["package"], "the probe must actually reorder/rename"
    assert _name_only_rename_problems(base, tip) == []


def _corrupt_first_hash(lock: dict[str, Any]) -> None:
    for pkg in lock["package"]:
        for art in [pkg.get("sdist"), *pkg.get("wheels", [])]:
            if art and "hash" in art:
                art["hash"] = art["hash"][:-1] + ("0" if art["hash"][-1] != "0" else "1")
                return
    raise AssertionError("the base lock has no hashed artefact to corrupt")


def _bump_first_registry_version(lock: dict[str, Any]) -> None:
    for pkg in lock["package"]:
        if not _is_root(pkg) and "version" in pkg:
            pkg["version"] = pkg["version"] + ".post1"
            return
    raise AssertionError("no non-root package")


def _drop_last_non_root(lock: dict[str, Any]) -> None:
    idx = max(i for i, p in enumerate(lock["package"]) if not _is_root(p))
    del lock["package"][idx]


def _unrename_root(lock: dict[str, Any]) -> None:
    for pkg in lock["package"]:
        if _is_root(pkg):
            pkg["name"] = "local-astra"


def _change_requires_python(lock: dict[str, Any]) -> None:
    lock["requires-python"] = ">=3.12"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_corrupt_first_hash, id="one-hash-altered"),
        pytest.param(_bump_first_registry_version, id="one-version-bumped"),
        pytest.param(_drop_last_non_root, id="one-package-dropped"),
        pytest.param(_unrename_root, id="root-not-renamed"),
        pytest.param(_change_requires_python, id="top-level-field-changed"),
    ],
)
def test_r2_comparator_refuses_anything_beyond_the_rename(mutate: Any) -> None:
    """Power rows, refusing direction: each mutant of an otherwise name-only relock is reported. Kept because the
    closed-range row above now reads two fixed commits and so can never again see a bad lock by itself."""
    base = _base_lock_or_skip()
    tip = _renamed_and_reordered(base)
    mutate(tip)
    assert _name_only_rename_problems(base, tip), f"{mutate.__name__} went unreported"


# --- R3 --------------------------------------------------------------------------------------------------------


def _help(
    argv0: str, args: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> str:
    monkeypatch.setattr(sys, "argv", [argv0, *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args)
    assert exit_info.value.code == 0, f"{argv0} {' '.join(args)} exited {exit_info.value.code}"
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "args", [pytest.param(["--help"], id="top"), pytest.param(["doctor", "--help"], id="doctor")]
)
def test_r3_usage_line_names_diffusion_gearvis_when_invoked_as_such(
    args: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R3: invoked as ``.venv/bin/diffusion_gearvis``, the usage line says ``diffusion_gearvis`` (either ruling --
    ``prog`` from ``sys.argv[0]``'s basename or fixed ``diffusion_gearvis`` -- gives this)."""
    out = _help("/opt/venv/bin/diffusion_gearvis", args, monkeypatch, capsys)
    want = "usage: diffusion_gearvis" + (" doctor" if args[0] == "doctor" else "")
    assert out.splitlines()[0].startswith(want + " "), out.splitlines()[0]


def test_r3_r6_subcommand_names_do_not_change(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R3/R6: the rename touches the program name only; the subcommand list is the base's (passes at the base)."""
    out = _help("/opt/venv/bin/diffusion_gearvis", ["--help"], monkeypatch, capsys)
    found = re.search(r"\{([a-z,-]+)\}", out)
    assert found is not None, out
    names = tuple(found.group(1).split(","))
    # T-0182 B re-scope (B-R3, docs/tasks/T-0182.md B "render-clips"): exactly one new subcommand, ``render-clips``;
    # the task does not fix its position, so the base's eleven keep their order around it and nothing else is added
    assert names.count("render-clips") == 1, (
        f"T-0182 B: `render-clips` is a subcommand; got {names}"
    )
    assert tuple(n for n in names if n != "render-clips") == SUBCOMMANDS_AT_BASE, names


@pytest.mark.parametrize(
    ("argv0", "want_prog"),
    [
        pytest.param("/opt/venv/bin/astra", "astra", id="astra"),
        pytest.param(
            "/opt/venv/bin/diffusion_gearvis", "diffusion_gearvis", id="diffusion_gearvis"
        ),
        pytest.param("diffusion_gearvis", "diffusion_gearvis", id="bare-new-name"),
        # Fallbacks: anything that is not exactly one of the two script basenames gets the new name.
        pytest.param(
            "/opt/venv/lib/python3.11/site-packages/local_astra/__main__.py",
            "diffusion_gearvis",
            id="dash-m",
        ),
        pytest.param("/opt/venv/bin/pytest", "diffusion_gearvis", id="pytest"),
        pytest.param("/opt/venv/bin/diffusion-gearvis", "diffusion_gearvis", id="near-miss-hyphen"),
        pytest.param("/opt/venv/bin/astra2", "diffusion_gearvis", id="near-miss-astra2"),
    ],
)
def test_o_r_version_names_the_invoked_command(
    argv0: str, want_prog: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """O-R (2): ``--version`` prints ``<prog> 0.0.1`` with the same prog as the usage line -- the invoked script's
    basename for ``astra`` / ``diffusion_gearvis``, else ``diffusion_gearvis``. ``0.0.1`` is R2's unchanged version,
    written by hand. Before O-R it printed ``astra 0.0.1`` whatever the command was."""
    out = _help(argv0, ["--version"], monkeypatch, capsys)
    assert out.strip() == f"{want_prog} 0.0.1", out
    usage = _help(argv0, ["--help"], monkeypatch, capsys).splitlines()[0]
    assert usage.startswith(f"usage: {want_prog} "), usage


def _normalised(text: str) -> str:
    """Both program names to one placeholder and whitespace collapsed: argparse wraps the usage line by ``prog``'s
    length, and under the argv[0] ruling each script names itself."""
    return " ".join(re.sub(r"\b(diffusion_gearvis|astra)\b", "PROG", text).split())


def test_r3_both_console_scripts_are_installed_and_answer_alike() -> None:
    """R3: ``uv sync --all-extras`` installs both scripts beside the running interpreter, each a wrapper of
    ``local_astra.cli.main``, and ``doctor --help`` exits alike and prints the same lines through either.

    Reads the venv this test runs in: ``uv run`` syncs it from this tree's ``pyproject.toml`` first. Under a bare
    ``.venv/bin/python`` that was synced elsewhere, a red here means "this venv predates R1"; run ``uv sync``.
    """
    bindir = Path(sys.executable).parent
    outs: dict[str, subprocess.CompletedProcess[str]] = {}
    for name in ("astra", "diffusion_gearvis"):
        script = bindir / name
        assert script.is_file(), (
            f"{script} is not installed (R3: uv sync --all-extras installs both)"
        )
        assert "from local_astra.cli import main" in script.read_text(), (
            f"{script} does not wrap {ENTRY}"
        )
        outs[name] = subprocess.run(
            [str(script), "doctor", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    old, new = outs["astra"], outs["diffusion_gearvis"]
    assert old.returncode == new.returncode == 0, (
        old.returncode,
        new.returncode,
        old.stderr,
        new.stderr,
    )
    assert new.stdout.splitlines()[0].startswith("usage: diffusion_gearvis doctor "), new.stdout
    assert _normalised(new.stdout) == _normalised(old.stdout)


# --- R6 --------------------------------------------------------------------------------------------------------


def test_r6_the_hook_state_dir_still_ends_in_local_astra() -> None:
    """R6 / H-T156-1: launch tokens stay under ``<git common dir>/local_astra`` (an approval written by ``/launch``
    must be found by the hook; moving one side strands every token)."""
    text = (REPO / ".claude" / "hooks" / "_lib.sh").read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith("STATE=")]
    assert len(lines) == 1, lines
    assert lines[0].endswith('/local_astra"'), lines[0]


def test_r6_the_supervisor_run_dir_still_ends_in_local_astra_run() -> None:
    """R6: pid files stay where ``astra down`` started before the rename will look for them."""
    assert supervisor.run_dir().parts[-2:] == ("local_astra", "run"), supervisor.run_dir()


def test_r6_the_memory_data_dir_is_still_local_share_local_astra() -> None:
    """R6 / H-T156-1: the user's memory database is not orphaned by the rename."""
    assert Path.home() / ".local" / "share" / "local_astra" == store.DEFAULT_DIR, store.DEFAULT_DIR
