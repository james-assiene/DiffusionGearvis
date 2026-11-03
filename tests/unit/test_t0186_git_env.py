"""T-0186: tests never inherit a git repo selector (docs/tasks/T-0186.md).

On 2026-10-28 the fast suite ran with ``GIT_DIR`` exported at the main repo. ``test_protect_paths_hook.py``'s sandbox
ran ``git init <tmp>`` with an env of ``{**os.environ, ...}``: git re-initialised the repo ``GIT_DIR`` named, wrote
``core.worktree = <tmp>`` into its config, and the next ``git commit`` landed the sandbox tree on ``main``. These rows
pin the session-wide fix:

- criterion 1: ``tests/conftest.py`` strips git's eight repo selectors from ``os.environ`` at import, so a nested
  session and its children never see them (and the T-0175 home jail is still in force);
- criterion 2: the T-0175 audit-hook spawn guard also refuses a ``git`` child whose env carries ``GIT_DIR`` or
  ``GIT_WORK_TREE``, naming the variable, on every spawn route;
- criterion 3: ``test_protect_paths_hook.GIT_ENV`` drops the selectors itself, and the adversarial row: a nested run of
  a sandbox row with both variables aimed at a throwaway victim repo leaves the victim alone;
- criterion 5: the guard is not too wide (``-C``/``--git-dir`` on argv, a non-git child with ``GIT_DIR``, a binary whose
  name merely ends in ``git``);
- criterion 6: a reduced selection of the modules that write git repos, run with ``GIT_DIR``/``GIT_WORK_TREE`` at a
  victim, leaves the victim's and the main repo's refs, config and reflogs byte-identical.

**Safety.** Every ``GIT_DIR``/``GIT_WORK_TREE`` any row sets names a victim repo under ``tmp_path``, never this checkout
or any of its worktrees. Victims are built with an env that carries no ``GIT_*`` variable at all.

Nested rows run ``python -m pytest`` from ``REPO`` by node id. The inner row (``test_inner_*``) skips unless
``ASTRA_T0186_REPORT`` names a report file.

Regime: CPU only, no network, Python 3.11, the venv's pytest, n = 1.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _home_jail as hj

REPO = Path(__file__).resolve().parents[2]
THIS = "tests/unit/test_t0186_git_env.py"
REPORT_VAR = "ASTRA_T0186_REPORT"

#: The eight selectors, written out from criterion 1 (never read from the implementation).
SELECTORS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
)
#: The two a git child may never carry (criterion 2).
GUARDED = ("GIT_DIR", "GIT_WORK_TREE")

#: Criterion 6's reduced selection: rows of the modules that ``git init`` and commit (a sandbox fixture, a
#: ``git -C <tmp> init`` with the inherited env, a scratch repo), plus one whole module that builds a repo.
VICTIM_SELECTION = (
    "tests/unit/test_protect_paths_hook.py::test_the_bare_path_workaround_keeps_working",
    "tests/unit/test_guard_bash_hook.py::test_gpu_spending_command_is_refused_on_a_dirty_tree",
    "tests/unit/test_stop_hook.py::test_one_expected_line_per_in_progress_task",
    "tests/unit/test_t0157_release.py::test_history_free_guards_skip",
    "tests/unit/test_cd_seed_locked_script.py::test_a_dirty_tree_is_refused_before_anything_runs",
    "tests/unit/test_guards_use_fixed_refs.py"
    "::test_changed_reports_every_modified_deleted_and_added_file_under_its_paths_only",
    "tests/unit/test_t0128c_locked_guard.py",
)
SANDBOX_ROW = VICTIM_SELECTION[0]


# ------------------------------------------------------------------------------------------------- helpers


def _clean_git_env() -> dict[str, str]:
    """An env with no ``GIT_*`` variable at all, no user config and a fixed identity: for building victims only."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "t0186-victim",
            "GIT_AUTHOR_EMAIL": "t0186-victim@example.invalid",
            "GIT_COMMITTER_NAME": "t0186-victim",
            "GIT_COMMITTER_EMAIL": "t0186-victim@example.invalid",
        }
    )
    return env


def _git_clean(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, env=_clean_git_env(), timeout=60
    )
    return out.stdout.strip()


def make_victim(tmp_path: Path) -> Path:
    """A throwaway repo with one commit on branch ``victim``; asserts it is not this checkout nor under it."""
    victim = tmp_path / "victim"
    victim.mkdir()
    assert not victim.resolve().is_relative_to(REPO.resolve()), (
        f"premise: victim {victim} is under the repo"
    )
    _git_clean("init", "-q", "-b", "victim", str(victim))
    (victim / "victim.txt").write_text("victim\n")
    _git_clean("-C", str(victim), "add", "-A")
    _git_clean("-C", str(victim), "commit", "-qm", "victim")
    assert (victim / ".git").is_dir()
    return victim


def snapshot_gitdir(gitdir: Path) -> dict[str, bytes]:
    """config, HEAD, packed-refs and every file under refs/ and logs/ of ``gitdir``, by relative path."""
    out: dict[str, bytes] = {}
    for name in ("config", "HEAD", "packed-refs"):
        p = gitdir / name
        if p.is_file():
            out[name] = p.read_bytes()
    for sub in ("refs", "logs"):
        root = gitdir / sub
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    out[str(p.relative_to(gitdir))] = p.read_bytes()
    return out


def _main_common_dir() -> Path:
    raw = _git_clean("-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(raw)


def snapshot_main() -> dict[str, bytes | None]:
    """The main repo's config, the main checkout's HEAD and its reflog, and ``main`` (loose ref, reflog, packed-refs).

    Narrower than the victim's snapshot on purpose: other agents commit on their own branches while the suite runs;
    the 2026-10-28 leak wrote ``core.worktree`` into this config and committed onto the main checkout's branch."""
    common = _main_common_dir()
    head = (common / "HEAD").read_bytes()
    names = {
        "config",
        "HEAD",
        "logs/HEAD",
        "packed-refs",
        "refs/heads/main",
        "logs/refs/heads/main",
    }
    m = re.match(rb"ref: (refs/\S+)", head)
    if m:
        ref = m.group(1).decode()
        names |= {ref, f"logs/{ref}"}
    return {
        n: ((common / n).read_bytes() if (common / n).is_file() else None) for n in sorted(names)
    }


def _diff_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def _check_main_unchanged(before: dict[str, bytes | None], what: str) -> None:
    after = snapshot_main()
    changed = _diff_keys(before, after)
    assert not changed, (
        f"T-0186 ({what}): the main repo's {changed} changed during the row "
        f"(config after: {after.get('config')!r}). If main moved by a legitimate merge mid-row, rerun; otherwise "
        f"a test wrote to the real repo"
    )
    assert b"worktree" not in (after.get("config") or b""), (
        f"T-0186 ({what}): the main repo's config names a core.worktree: {after.get('config')!r}"
    )


def _check_victim_unchanged(victim: Path, before: dict[str, bytes], what: str) -> None:
    after = snapshot_gitdir(victim / ".git")
    config = after.get("config", b"")
    assert b"worktree" not in config, (
        f"T-0186 ({what}): git re-initialised the victim and wrote core.worktree into its config:\n"
        f"{config.decode(errors='replace')}"
    )
    changed = _diff_keys(before, after)
    assert not changed, (
        f"T-0186 ({what}): the victim's {changed} changed (HEAD/refs/reflog/config must not move)"
    )


def nested_pytest(
    env: dict[str, str], *nodeids: str, extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """``python -m pytest -q -p no:cacheprovider <nodeids>`` from ``REPO`` with exactly ``env``."""
    scrub = {"PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST"}
    env = {k: v for k, v in env.items() if k not in scrub and not k.startswith("PYTEST_XDIST_")}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            *extra,
            *nodeids,
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _out(proc: subprocess.CompletedProcess[str]) -> str:
    return f"exit {proc.returncode}\nstdout {proc.stdout[-3000:]}\nstderr {proc.stderr[-1500:]}"


#: The exported shapes of the nested rows. ``GIT_DIR`` alone is the 2026-10-28 incident (git init writes
#: ``core.worktree`` into the named repo and the sandbox commit lands on its ``main``); both variables is the shape
#: criterion 3 names (git init then leaves the config alone, and the sandbox's commits go to the victim instead).
EXPORTS = ("GIT_DIR", "GIT_DIR+GIT_WORK_TREE")


def _victim_env(victim: Path, export: str = "GIT_DIR+GIT_WORK_TREE") -> dict[str, str]:
    """``os.environ`` with ``GIT_DIR`` (and ``GIT_WORK_TREE``) aimed at the victim (never at the real repo)."""
    assert (victim / ".git").is_dir() and not victim.resolve().is_relative_to(REPO.resolve())
    env = {**os.environ, "GIT_DIR": str(victim / ".git")}
    if "GIT_WORK_TREE" in export:
        env["GIT_WORK_TREE"] = str(victim)
    return env


def _git_path() -> str:
    git = shutil.which("git")
    assert git, "premise: git is on PATH"
    return git


# ------------------------------------------------------------------------------------------------- inner row

INNER = pytest.mark.skipif(
    REPORT_VAR not in os.environ,
    reason=f"inner row of T-0186's nested-session row: runs only inside a nested pytest started by "
    f"test_conftest_strips_the_selectors_for_the_session_and_its_children ({REPORT_VAR} names its report file)",
)

_CHILD_PROBE = (
    "import json, os, sys; print(json.dumps(sorted(k for k in sys.argv[1:] if k in os.environ)))"
)


@INNER
def test_inner_report() -> None:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE, *SELECTORS], capture_output=True, text=True, timeout=60
    )
    data = {
        "session": sorted(k for k in SELECTORS if k in os.environ),
        "child": json.loads(child.stdout)
        if child.returncode == 0
        else {"error": child.stderr[-2000:]},
        "home": os.environ.get("HOME"),
        "marker": os.environ.get(hj.MARKER),
        "xdg": os.environ.get("XDG_CONFIG_HOME"),
    }
    Path(os.environ[REPORT_VAR]).write_text(json.dumps(data))


# ------------------------------------------------------------------------------------------------- criterion 1


def test_conftest_strips_the_selectors_for_the_session_and_its_children(tmp_path: Path) -> None:
    """Criterion 1: a nested session started with all eight selectors set sees none of them in ``os.environ``, nor
    does a child it starts with no ``env=``; the T-0175 jail is still in force there (HOME is the jail marker)."""
    victim = make_victim(tmp_path)
    gd = victim / ".git"
    env = {
        **os.environ,
        "GIT_DIR": str(gd),
        "GIT_WORK_TREE": str(victim),
        "GIT_INDEX_FILE": str(gd / "index"),
        "GIT_OBJECT_DIRECTORY": str(gd / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(gd / "objects"),
        "GIT_COMMON_DIR": str(gd),
        "GIT_NAMESPACE": "t0186",
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
    }
    assert set(SELECTORS) <= set(env)
    report = tmp_path / "report.json"
    env[REPORT_VAR] = str(report)
    before = snapshot_gitdir(gd)
    proc = nested_pytest(env, f"{THIS}::test_inner_report")
    assert proc.returncode == 0, _out(proc)
    data = json.loads(report.read_text())
    assert data["session"] == [], (
        f"T-0186 criterion 1: the nested session's os.environ still carries {data['session']}"
    )
    assert data["child"] == [], (
        f"T-0186 criterion 1: a child of the nested session inherited {data['child']}"
    )
    assert data["home"] and data["home"] == data["marker"], (
        f"T-0186 criterion 1 (beside T-0175): the nested session is not in the home jail: {data}"
    )
    assert data["xdg"] == os.path.join(data["home"], ".config"), f"T-0175 jail weakened: {data}"
    _check_victim_unchanged(victim, before, "criterion 1")


# ------------------------------------------------------------------------------------------------- criterion 2

ROUTES = (
    "popen-name",
    "popen-abspath",
    "popen-executable",
    "asyncio",
    "posix-spawn",
    "env-none",
    "os-system",
)


def _init_argv(probe: Path) -> list[str]:
    return ["git", "init", "-q", str(probe)]


async def _attempt(
    route: str, var: str, victim: Path, probe: Path, mp: pytest.MonkeyPatch
) -> BaseException | None:
    value = str(victim / ".git") if var == "GIT_DIR" else str(victim)
    env = {**os.environ, var: value}
    argv = _init_argv(probe)
    try:
        if route == "popen-name":
            subprocess.Popen(argv, env=env).wait(timeout=60)
        elif route == "popen-abspath":
            subprocess.Popen([_git_path(), *argv[1:]], env=env).wait(timeout=60)
        elif route == "popen-executable":
            subprocess.Popen(["probe-git", *argv[1:]], executable=_git_path(), env=env).wait(
                timeout=60
            )
        elif route == "asyncio":
            p = await asyncio.create_subprocess_exec(*argv, env=env)
            await asyncio.wait_for(p.wait(), 60)
        elif route == "posix-spawn":
            pid = os.posix_spawn(_git_path(), argv, env)
            os.waitpid(pid, 0)
        elif route == "env-none":
            mp.setenv(var, value)
            try:
                subprocess.Popen(argv).wait(timeout=60)
            finally:
                mp.undo()
        elif route == "os-system":
            mp.setenv(var, value)
            try:
                os.system(f"git init -q '{probe}'")
            finally:
                mp.undo()
        else:  # pragma: no cover
            raise AssertionError(route)
    except AssertionError:
        raise
    except Exception as exc:
        return exc
    return None


def _assert_git_refusal(err: BaseException | None, var: str, what: str) -> None:
    assert err is not None, (
        f"T-0186 criterion 2 ({what}): a git spawn carrying {var} was not refused"
    )
    assert isinstance(err, hj.JailRefusal) and not isinstance(err, OSError), (
        f"T-0186 criterion 2 ({what}): refused by something other than the spawn guard: {err!r}"
    )
    assert var in str(err), f"T-0186 criterion 2 ({what}): the refusal does not name {var}: {err}"


@pytest.mark.parametrize("var", GUARDED)
@pytest.mark.parametrize("route", ROUTES)
async def test_guard_refuses_a_git_child_that_names_a_repo(
    route: str, var: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 2: on every spawn route, a git child whose env carries ``var`` is refused before exec with a
    ``JailRefusal`` naming ``var``; the ``git init`` never ran (no probe dir, the victim untouched)."""
    victim = make_victim(tmp_path)
    before = snapshot_gitdir(victim / ".git")
    probe = tmp_path / "probe"
    err = await _attempt(route, var, victim, probe, monkeypatch)
    _assert_git_refusal(err, var, route)
    assert not probe.exists(), f"T-0186 criterion 2 ({route}): the refused git init ran"
    _check_victim_unchanged(victim, before, f"criterion 2, {route}")


@pytest.mark.parametrize("var", GUARDED)
def test_guard_refuses_a_git_exec(var: str, tmp_path: Path) -> None:
    """Criterion 2, the ``os.exec*`` route: refused in the process that calls it (a forked child reports it)."""
    victim = make_victim(tmp_path)
    before = snapshot_gitdir(victim / ".git")
    probe = tmp_path / "probe"
    value = str(victim / ".git") if var == "GIT_DIR" else str(victim)
    env = {**os.environ, var: value}
    git = _git_path()
    refused, other = 77, 78
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        code = other
        try:
            os.execve(git, _init_argv(probe), env)
        except hj.JailRefusal as exc:
            code = refused if var in str(exc) else 79
        except BaseException:
            code = other
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    got = os.waitstatus_to_exitcode(status)
    assert got == refused, (
        f"T-0186 criterion 2 (exec): the child's execve of git with {var} was not refused by the guard (exit {got}; "
        f"0 means git ran, 78 another error, 79 a refusal that does not name {var})"
    )
    assert not probe.exists()
    _check_victim_unchanged(victim, before, "criterion 2, exec")


# ------------------------------------------------------------------------------------------------- criterion 5

ALLOW_CASES = (
    "git-dash-C",
    "git-dir-flag-joined",
    "git-dir-flag-separate",
    "non-git-with-git-dir",
    "name-ends-in-git",
)


def test_guard_allows_what_criterion_5_allows_premise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premise of the allow rows, as its own row: the git clause is on in this process."""
    victim = make_victim(tmp_path)
    err = asyncio.run(_attempt("popen-name", "GIT_DIR", victim, tmp_path / "probe", monkeypatch))
    _assert_git_refusal(err, "GIT_DIR", "premise")


@pytest.mark.parametrize("case", ALLOW_CASES)
def test_guard_is_not_too_wide(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 5 (the over-wide direction): each case runs and does its job. Premise, in the same row: a git spawn
    carrying ``GIT_DIR`` is refused, so the clause is on (an allowed case proves nothing without it)."""
    victim = make_victim(tmp_path)
    err = asyncio.run(_attempt("popen-name", "GIT_DIR", victim, tmp_path / "probe", monkeypatch))
    _assert_git_refusal(err, "GIT_DIR", f"premise of {case}")
    before = snapshot_gitdir(victim / ".git")
    gd = str((victim / ".git").resolve())
    try:
        if case == "git-dash-C":
            out = subprocess.run(
                ["git", "-C", str(victim), "rev-parse", "--absolute-git-dir"],
                capture_output=True,
                text=True,
            )
            assert out.returncode == 0 and out.stdout.strip() == gd, out
        elif case == "git-dir-flag-joined":
            out = subprocess.run(
                ["git", f"--git-dir={gd}", "rev-parse", "--absolute-git-dir"],
                capture_output=True,
                text=True,
            )
            assert out.returncode == 0 and out.stdout.strip() == gd, out
        elif case == "git-dir-flag-separate":
            out = subprocess.run(
                [_git_path(), "--git-dir", gd, "rev-parse", "--absolute-git-dir"],
                capture_output=True,
                text=True,
            )
            assert out.returncode == 0 and out.stdout.strip() == gd, out
        elif case == "non-git-with-git-dir":
            flag = tmp_path / "ran"
            code = subprocess.Popen(
                ["/bin/sh", "-c", f'touch "{flag}"'], env=_victim_env(victim)
            ).wait(60)
            assert code == 0 and flag.exists()
        elif case == "name-ends-in-git":
            legit = tmp_path / "legit"
            legit.symlink_to(shutil.which("true") or "/bin/true")
            code = subprocess.Popen([str(legit)], env=_victim_env(victim)).wait(60)
            assert code == 0
        else:  # pragma: no cover
            raise AssertionError(case)
    except hj.JailRefusal as exc:
        raise AssertionError(f"T-0186 criterion 5 ({case}): refused: {exc}") from exc
    _check_victim_unchanged(victim, before, f"criterion 5, {case}")


# ------------------------------------------------------------------------------------------------- criterion 3


def test_protect_paths_git_env_drops_the_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 3 (belt and braces): ``GIT_ENV`` built while all eight selectors are in ``os.environ`` carries none
    of them. The module is loaded as a fresh copy under another name, so the collected module is not touched."""
    for k in SELECTORS:
        monkeypatch.setenv(k, f"/nonexistent/t0186/{k}")
    path = REPO / "tests" / "unit" / "test_protect_paths_hook.py"
    spec = importlib.util.spec_from_file_location("_t0186_protect_paths_copy", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the defining module up by name
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    leaked = sorted(k for k in SELECTORS if k in mod.GIT_ENV)
    assert not leaked, f"T-0186 criterion 3: test_protect_paths_hook.GIT_ENV carries {leaked}"
    assert mod.GIT_ENV.get("GIT_AUTHOR_NAME") == "protect-test", (
        "premise: GIT_ENV still pins the identity"
    )


@pytest.mark.parametrize("export", EXPORTS)
def test_protect_paths_sandbox_cannot_touch_a_victim(export: str, tmp_path: Path) -> None:
    """Criterion 3, the adversarial row: a nested run of a sandbox row with ``GIT_DIR`` (and ``GIT_WORK_TREE``) at a
    victim passes; the victim's config has no core.worktree and its HEAD/refs/reflog do not move; the sandbox has its
    own ``.git`` with the sandbox's one commit."""
    victim = make_victim(tmp_path)
    before = snapshot_gitdir(victim / ".git")
    head = _git_clean("-C", str(victim), "rev-parse", "HEAD")
    main_before = snapshot_main()
    basetemp = tmp_path / "bt"
    proc = nested_pytest(
        _victim_env(victim, export), SANDBOX_ROW, extra=(f"--basetemp={basetemp}",)
    )
    _check_victim_unchanged(victim, before, "criterion 3 adversarial")
    assert _git_clean("-C", str(victim), "rev-parse", "HEAD") == head
    _check_main_unchanged(main_before, "criterion 3 adversarial")
    assert proc.returncode == 0, _out(proc)
    # pytest also links ``<name>current`` to the newest numbered dir; count the real directories only
    sandboxes = sorted(p for p in basetemp.glob("*/repo") if not p.parent.is_symlink())
    assert len(sandboxes) == 1, (
        f"T-0186 criterion 3: expected one sandbox under {basetemp}: {sandboxes}"
    )
    sb = sandboxes[0]
    assert (sb / ".git").is_dir(), f"T-0186 criterion 3: the sandbox {sb} has no .git of its own"
    assert _git_clean("-C", str(sb), "rev-parse", "--absolute-git-dir") == str(
        (sb / ".git").resolve()
    )
    log = _git_clean("-C", str(sb), "log", "--format=%an %s")
    assert log == "protect-test init", f"T-0186 criterion 3: the sandbox history is {log!r}"


# ------------------------------------------------------------------------------------------------- criterion 6


@pytest.mark.parametrize("export", EXPORTS)
def test_git_writing_modules_leave_victim_and_main_byte_identical(
    export: str, tmp_path: Path
) -> None:
    """Criterion 6, at a reduced selection (``VICTIM_SELECTION``): with ``GIT_DIR`` (and ``GIT_WORK_TREE``) exported at
    a victim, the selection passes and the victim's refs, config and reflog are byte-identical before and after, and
    so are the main repo's (``snapshot_main``)."""
    victim = make_victim(tmp_path)
    before = snapshot_gitdir(victim / ".git")
    main_before = snapshot_main()
    proc = nested_pytest(
        _victim_env(victim, export), *VICTIM_SELECTION, extra=(f"--basetemp={tmp_path / 'bt'}",)
    )
    _check_victim_unchanged(victim, before, "criterion 6")
    _check_main_unchanged(main_before, "criterion 6")
    assert proc.returncode == 0, _out(proc)
    m = re.search(r"(\d+) passed", proc.stdout)
    assert m and int(m.group(1)) >= len(VICTIM_SELECTION), _out(proc)
