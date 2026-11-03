"""T-0175: test-spawned processes never see the real home, where the Brave Search API key file lives
(docs/tasks/T-0175.md).

A fresh interpreter does not load conftest, so ``brave_key_nowhere``'s in-process patch never protected a child: A6's
BASE subprocess read the real ``~/.config/diffusion_gearvis/brave_api_key``. These rows pin the session jail (``HOME``
and ``XDG_CONFIG_HOME`` point at a per-session directory from the import of ``tests/conftest.py`` on), that children
inherit it (working tree, an old BASE tree, ``uv run``), the mirror allowlist, the fail-closed spawn guard in both
directions, the ``ASTRA_LIVE_SEARCH``-only escape, and one jail per top-level session.

**Security.** No row opens, stats or lists anything under the real home. The real home appears only as a string
(``pwd.getpwuid(os.getuid()).pw_dir`` and the pre-switch ``HOME`` the jail records). Every "would read" proof uses a
tmp planted home with a fake 32-character key. A row that would start a child from this process first asserts the
jail is in force (``_jail``), so no child is ever started with the real home in ``HOME``.

Nested-session rows run ``python -m pytest`` on one inner row of this module by node id. The inner rows
(``test_inner_*``) skip, with a reason, unless ``ASTRA_T0175_REPORT`` names a report file; each spawns a status child
and writes JSON there. Only planted runs set ``ASTRA_T0175_RESOLVE=1``, which makes the inner row list the jail and
resolve the cache paths (never done against the real home).

Regime: CPU only, no network, Python 3.11, the venv's pytest and xdist, n = 1, the fake key random per row.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import re
import secrets
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests import _home_jail as hj

REPO = Path(__file__).resolve().parents[2]
THIS = "tests/unit/test_t0175_home_jail.py"
#: the first BASE that has ``local_astra/search/brave_key.py`` (A6's BASE)
BASE_WITH_LOADER = "d486bcb9"
REPORT_VAR = "ASTRA_T0175_REPORT"
RESOLVE_VAR = "ASTRA_T0175_RESOLVE"
LIVE_VAR = "ASTRA_LIVE_SEARCH"

#: The mirror allowlist, written out from the task's implementation notes (never read from the implementation).
ALLOWLIST = frozenset(
    {
        ".cache",
        ".gitconfig",
        ".config/git",
        ".config/uv",
        ".local/share/uv",
        ".docker",
        ".Xauthority",
        ".local/share/fonts",
        ".fonts",
        ".config/fontconfig",
    }
)
#: The only real directories the jail may hold: the parents of nested allowlist entries.
REAL_DIRS = frozenset({".config", ".local", ".local/share"})
#: Variables that pin a cache away from the ``~/.cache`` default rule; removed for planted runs.
CACHE_PINS = (
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "PLAYWRIGHT_BROWSERS_PATH",
)

STATUS = r"""
import json
from pathlib import Path
from local_astra.search import brave_key
k = brave_key.load_key()
print(json.dumps({"home": str(Path.home()), "status": k.status, "reason": k.reason, "origin": brave_key.__file__}))
"""
STATUS_WORD = r"""
from local_astra.search import brave_key
print(brave_key.load_key().status)
"""


# ------------------------------------------------------------------------------------------------- helpers


def _pwd_home() -> str:
    """The password-database home, as a string (never touched)."""
    return os.path.normpath(pwd.getpwuid(os.getuid()).pw_dir)


def _jail() -> str:
    """This session's jail, asserted in force *before* any child is started from this process."""
    marker = os.environ.get(hj.MARKER)
    home = os.environ.get("HOME", "")
    real = os.environ.get(hj.REAL_HOME_VAR)
    assert marker, f"T-0175 criterion 1: {hj.MARKER} is not set: the session jail is not in force"
    assert home and os.path.normpath(home) == os.path.normpath(marker), (
        f"T-0175 criterion 1: HOME {home!r} is not the jail {marker!r}"
    )
    assert os.path.normpath(home) != _pwd_home(), (
        "T-0175 criterion 1: HOME is the password-database home"
    )
    assert real and os.path.normpath(real) != os.path.normpath(home), (
        f"T-0175 criterion 1: the jail records no pre-switch home ({hj.REAL_HOME_VAR}={real!r})"
    )
    return home


def _child_json(proc: subprocess.CompletedProcess[str], what: str) -> dict[str, Any]:
    assert proc.returncode == 0 and proc.stdout.strip(), (
        f"{what}: exit {proc.returncode}\nstdout {proc.stdout[-1500:]}\nstderr {proc.stderr[-2500:]}"
    )
    data: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return data


def planted_home(tmp_path: Path) -> tuple[Path, bytes]:
    """A tmp home with a fake 32-character key at the default path (mode 0600) and the cache directories."""
    home = tmp_path / "planted"
    cfg = home / ".config" / "diffusion_gearvis"
    cfg.mkdir(parents=True)
    for d in (
        ".cache/huggingface/hub",
        ".cache/uv",
        ".cache/ms-playwright",
        ".local/share/local_astra",
    ):
        (home / d).mkdir(parents=True)
    alphabet = string.ascii_letters + string.digits
    key = ("".join(secrets.choice(alphabet) for _ in range(32)) + "\n").encode()
    path = cfg / "brave_api_key"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return home, key


def planted_env(planted: Path) -> dict[str, str]:
    """A top-level environment whose ``HOME`` is the planted home: no jail markers, no cache pins, no opt-in."""
    drop = {hj.MARKER, hj.REAL_HOME_VAR, hj.CREATOR_VAR, "XDG_CONFIG_HOME", LIVE_VAR, *CACHE_PINS}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(planted)
    env[RESOLVE_VAR] = "1"
    return env


def nested_pytest(
    env: dict[str, str], nodeid: str, report: Path
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    """``python -m pytest -q -p no:cacheprovider <nodeid>`` from ``REPO``; the inner row writes JSON to ``report``."""
    scrub = {"PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST"}
    env = {k: v for k, v in env.items() if k not in scrub and not k.startswith("PYTEST_XDIST_")}
    env[REPORT_VAR] = str(report)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{THIS}::{nodeid}"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    data = json.loads(report.read_text()) if report.exists() else None
    return proc, data


def _out(proc: subprocess.CompletedProcess[str]) -> str:
    return f"exit {proc.returncode}\nstdout {proc.stdout[-3000:]}\nstderr {proc.stderr[-1500:]}"


def _listing(root: str) -> list[list[Any]]:
    """Every entry under ``root`` (the jail) without following a symlink: ``[relpath, kind, readlink or None]``."""
    out: list[list[Any]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(dirnames + filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out.append([rel, "link", os.readlink(p)])
            elif os.path.isdir(p):
                out.append([rel, "dir", None])
            else:
                out.append([rel, "file", None])
    return out


def _check_planted_report(
    data: dict[str, Any] | None, planted: Path, key: bytes, what: str
) -> None:
    """Criteria 3, 4 and 9 over one nested planted run's report."""
    assert data is not None, f"{what}: the inner row wrote no report"
    child = data["child"]
    assert child.get("status") == "not configured" and child.get("reason") is None, (
        f"T-0175 criterion 3 ({what}): the inner child read the planted key: {child}"
    )
    home = os.path.normpath(data["home"])
    p = os.path.normpath(str(planted))
    assert home != p and not Path(home).is_relative_to(p), (
        f"T-0175 criterion 3 ({what}): the inner session's HOME is the planted home {home}"
    )
    assert data["path_home"] == data["home"] == data["marker"], (
        f"T-0175 criterion 1 ({what}): Path.home(), HOME and the marker disagree: {data}"
    )
    assert data["xdg"] == os.path.join(data["home"], ".config"), (
        f"T-0175 criterion 1 ({what}): XDG_CONFIG_HOME is {data['xdg']!r}"
    )
    forbidden = [
        p,
        os.path.join(p, ".config"),
        os.path.join(p, ".config", "diffusion_gearvis"),
        os.path.join(p, ".config", "diffusion_gearvis", "brave_api_key"),
        os.path.join(p, ".local", "share", "local_astra"),
    ]
    entries = data["jail_listing"]
    kinds = {rel: kind for rel, kind, _ in entries}
    assert kinds.get(".config") == "dir", (
        f"T-0175 criterion 1 ({what}): <jail>/.config is not a real directory: {entries}"
    )
    for rel, kind, target in entries:
        if kind == "dir":
            assert rel in REAL_DIRS, (
                f"T-0175 criterion 4 ({what}): real directory {rel} in the jail: {entries}"
            )
            continue
        assert kind == "link" and rel in ALLOWLIST, (
            f"T-0175 criterion 4 ({what}): {rel} ({kind}) is not an allowlisted symlink: {entries}"
        )
        assert target == os.path.join(p, rel), (
            f"T-0175 criterion 4 ({what}): {rel} -> {target}, not to the same path under the planted home"
        )
        t = os.path.normpath(target)
        for f in forbidden:
            assert not (f == t or f.startswith(t + os.sep)), (
                f"T-0175 criterion 4 ({what}): {rel} -> {target} reaches {f}"
            )
    for field, rel in (
        ("hf_hub", ".cache/huggingface/hub"),
        ("uv_cache", ".cache/uv"),
        ("pw_path", ".cache/ms-playwright"),
    ):
        want = os.path.realpath(os.path.join(p, rel))
        assert data.get(field) == want, (
            f"T-0175 criterion 4 ({what}): {field} resolves to {data.get(field)!r} in the jail, not {want}"
        )
    assert not os.path.lexists(data["home"]), (
        f"T-0175 criterion 9 ({what}): the inner top-level run left its jail {data['home']} behind"
    )
    assert (planted / ".config/diffusion_gearvis/brave_api_key").read_bytes() == key, (
        f"T-0175 ({what}): the planted key file changed"
    )
    for d in (
        ".cache/huggingface/hub",
        ".cache/uv",
        ".cache/ms-playwright",
        ".local/share/local_astra",
    ):
        assert (planted / d).is_dir(), (
            f"T-0175 criterion 9 ({what}): cleanup followed a symlink and removed <planted>/{d}"
        )


# ------------------------------------------------------------------------------------------------- inner rows

INNER = pytest.mark.skipif(
    REPORT_VAR not in os.environ,
    reason=f"inner row of T-0175's nested-session rows: runs only inside a nested pytest started by an outer row "
    f"of this module, selected by node id ({REPORT_VAR} names its report file)",
)


@INNER
def test_inner_report(monkeypatch: pytest.MonkeyPatch) -> None:
    home = os.environ.get("HOME", "")
    if not home or os.path.normpath(home) == _pwd_home():
        pytest.fail("T-0175 inner row: refusing to start a status child with the real home in HOME")
    child = subprocess.run(
        [sys.executable, "-c", STATUS], cwd=REPO, capture_output=True, text=True, timeout=60
    )
    data: dict[str, Any] = {
        "home": home,
        "path_home": str(Path.home()),
        "marker": os.environ.get(hj.MARKER),
        "creator": os.environ.get(hj.CREATOR_VAR),
        "xdg": os.environ.get("XDG_CONFIG_HOME"),
        "child": json.loads(child.stdout.strip().splitlines()[-1])
        if child.returncode == 0 and child.stdout.strip()
        else {"error": child.stderr[-2000:]},
    }
    if os.environ.get(RESOLVE_VAR) == "1":
        data["jail_listing"] = _listing(home)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        from local_astra.toolcall import detok

        data["hf_hub"] = os.path.realpath(detok._hf_hub_cache())
        uv = shutil.which("uv")
        if uv is not None:
            got = subprocess.run([uv, "cache", "dir"], capture_output=True, text=True, timeout=60)
            data["uv_cache"] = os.path.realpath(got.stdout.strip()) if got.returncode == 0 else None
        pw = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        m = re.search(r"Install location:\s+(\S+)", pw.stdout)
        data["pw_path"] = os.path.realpath(os.path.dirname(m.group(1))) if m else None
    Path(os.environ[REPORT_VAR]).write_text(json.dumps(data))


@INNER
def test_inner_escape(real_home_env: dict[str, str]) -> None:
    """Requests the escape as a parameter, so a refusing fixture is a setup error (the skip mark is evaluated
    before fixture setup, so the outer suite only ever skips this row)."""
    env = real_home_env
    child = subprocess.run(
        [sys.executable, "-c", STATUS_WORD],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    Path(os.environ[REPORT_VAR]).write_text(json.dumps({"status": child.stdout.strip()}))


# ------------------------------------------------------------------------------------------------- rows 1-4


def test_jail_in_force_in_process(tmp_path: Path) -> None:
    """Row 1 (criterion 1): ``Path.home()``, ``HOME`` and the marker agree; ``XDG_CONFIG_HOME`` is ``<jail>/.config``,
    a real directory with no ``diffusion_gearvis``; ``store.DEFAULT_DIR`` is under the jail; the home is not the
    password-database home. Plus: in a fresh interpreter with ``HOME=<planted>``, pytest's own config preparation
    (``_pytest.config._prepareconfig``: it loads the initial conftests and never calls ``pytest_configure``) leaves
    ``HOME`` jailed, so the switch happens at conftest import; that probe's jail is gone after it exits."""
    jail = _jail()
    assert str(Path.home()) == jail, f"Path.home() is {Path.home()}, not the jail {jail}"
    cfg = os.path.join(jail, ".config")
    assert os.environ.get("XDG_CONFIG_HOME") == cfg, (
        f"T-0175 criterion 1: XDG_CONFIG_HOME is {os.environ.get('XDG_CONFIG_HOME')!r}, not {cfg}"
    )
    assert not os.path.islink(cfg) and os.path.isdir(cfg), (
        "T-0175 criterion 1: <jail>/.config is not a real directory"
    )
    assert "diffusion_gearvis" not in os.listdir(cfg), (
        "T-0175 criterion 1: <jail>/.config holds diffusion_gearvis"
    )
    from local_astra.memory import store

    assert Path(store.DEFAULT_DIR).is_relative_to(jail), (
        f"T-0175 criterion 1: memory DEFAULT_DIR {store.DEFAULT_DIR} was computed before the jail ({jail})"
    )
    assert os.path.normpath(jail) != _pwd_home()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        assert os.environ.get(hj.CREATOR_VAR) != str(os.getpid()), (
            "T-0175 criterion 9: this xdist worker made its own jail instead of reusing the controller's"
        )

    planted, _ = planted_home(tmp_path)
    probe = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from _pytest.config import _prepareconfig\n"
        "config = _prepareconfig([sys.argv[2], '-p', 'no:cacheprovider'])\n"
        "h = os.environ.get('HOME', '')\n"
        "print(json.dumps({'home': h, 'path_home': str(Path.home()), 'marker': os.environ.get(sys.argv[1]),"
        " 'xdg': os.environ.get('XDG_CONFIG_HOME'), 'conftest': 'tests.conftest' in sys.modules}))\n"
        "config._ensure_unconfigure()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, hj.MARKER, THIS],
        cwd=REPO,
        env=planted_env(planted),
        capture_output=True,
        text=True,
        timeout=120,
    )
    got = _child_json(proc, "pytest config preparation in a fresh interpreter")
    assert got["conftest"], f"premise: preparing the config loaded tests/conftest.py: {got}"
    assert got["home"] != str(planted) and got["home"] == got["path_home"] == got["marker"], (
        f"T-0175 criterion 1: HOME is not jailed once the initial conftests are loaded: {got}"
    )
    assert got["xdg"] == os.path.join(got["home"], ".config"), got
    assert not Path(got["home"]).is_relative_to(REPO), "T-0175: the jail is under the repo"
    assert not os.path.lexists(got["home"]), (
        f"T-0175 criterion 9: the probe left its jail {got['home']} behind"
    )


def test_library_import_of_conftest_keeps_the_rig_home(tmp_path: Path) -> None:
    """Criterion 6 (a tmp HOME holding a planted key is allowed, and existing T-0163 rows rely on it): a fresh
    interpreter that imports ``tests.conftest`` as a library (CLI rigs and BASE runners import test modules, which
    import it) keeps the tmp ``HOME`` its parent chose; it is not re-jailed, and it reads the planted key."""
    planted, _ = planted_home(tmp_path)
    probe = "import tests.conftest\n" + STATUS
    env = {k: v for k, v in planted_env(planted).items() if k != RESOLVE_VAR}
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    got = _child_json(proc, "library import of tests.conftest")
    assert got["home"] == str(planted), (
        f"T-0175: a library import of tests/conftest.py moved the rig's HOME: {got}"
    )
    assert got["status"] == "configured", got


def test_child_from_working_tree_sees_jail() -> None:
    """Row 2 (criterion 2): a ``sys.executable`` child with no ``env=`` reports the jail and ``not configured``."""
    jail = _jail()
    proc = subprocess.run(
        [sys.executable, "-c", STATUS], cwd=REPO, capture_output=True, text=True, timeout=60
    )
    got = _child_json(proc, "working-tree status child")
    assert got["home"] == jail, f"T-0175 criterion 2: the child's home is {got['home']}, not {jail}"
    assert got["status"] == "not configured" and got["reason"] is None, got


def test_child_from_old_base_sees_jail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Row 3 (criterion 2): a child over ``git archive d486bcb9`` (A6's ``_base_tree``), with origin proof, reports
    the jail and ``not configured``: an old tree that knows no new variable is protected by ``HOME`` alone."""
    jail = _jail()
    gr.require_commits(BASE_WITH_LOADER)
    from tests.unit import test_session_tool_loop as tsl

    monkeypatch.setattr(tsl, "BASE_SHA", BASE_WITH_LOADER)
    base_dir = tsl._base_tree(tmp_path / "base_tree")
    proc = subprocess.run(
        [sys.executable, "-c", STATUS], cwd=base_dir, capture_output=True, text=True, timeout=60
    )
    got = _child_json(proc, f"status child over {BASE_WITH_LOADER}")
    assert Path(got["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: brave_key came from {got['origin']}, not the {BASE_WITH_LOADER} tree"
    )
    assert got["home"] == jail, f"T-0175 criterion 2: the BASE child's home is {got['home']}"
    assert got["status"] == "not configured" and got["reason"] is None, got


def test_uv_run_child_sees_jail() -> None:
    """Row 4 (criterion 2): ``uv run --no-sync --offline python`` reports the jail and exits 0 (the uv cache and
    config are reachable offline through the mirror)."""
    jail = _jail()
    uv = shutil.which("uv")
    assert uv is not None, "premise: uv is on PATH"
    proc = subprocess.run(
        [uv, "run", "--no-sync", "--offline", "python", "-c", STATUS],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    got = _child_json(proc, "uv run child")
    assert got["home"] == jail, f"T-0175 criterion 2: the uv child's home is {got['home']}"
    assert got["status"] == "not configured" and got["reason"] is None, got


# ------------------------------------------------------------------------------------------------- rows 5-8


def test_planted_key_is_read_without_the_jail(tmp_path: Path) -> None:
    """Row 5 (criterion 3a, premise): a bare child with ``HOME=<planted>`` and no pytest reads the planted key. Green
    before and after T-0175 by design: if it fails, every planted row below proves nothing."""
    planted, _ = planted_home(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", STATUS],
        cwd=REPO,
        env={"HOME": str(planted), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    got = _child_json(proc, "bare planted child")
    assert got["home"] == str(planted)
    assert got["status"] == "configured", f"premise: the planted key is not read: {got}"


def test_nested_session_with_planted_home_is_jailed(tmp_path: Path) -> None:
    """Row 6 (criteria 3b, 4, 9; adversarial against a jail that misses children or mirrors the config dir)."""
    planted, key = planted_home(tmp_path)
    proc, data = nested_pytest(planted_env(planted), "test_inner_report", tmp_path / "report.json")
    assert proc.returncode == 0 and "1 passed" in proc.stdout, _out(proc)
    _check_planted_report(data, planted, key, "fresh planted session")


def test_stale_marker_is_rejailed(tmp_path: Path) -> None:
    """Row 7 (criterion 3c): a marker naming an existing directory that is not ``HOME`` is stale: still jailed, in a
    fresh jail (not the stale directory, not the planted home). The recorded real home is a tmp string too, so a
    jail that trusted it would mirror a path that is not under the planted home."""
    planted, key = planted_home(tmp_path)
    stale = tmp_path / "stale"
    stale.mkdir()
    env = planted_env(planted)
    env[hj.MARKER] = str(stale)
    env[hj.REAL_HOME_VAR] = str(tmp_path / "stale_real")
    env[hj.CREATOR_VAR] = "1"
    proc, data = nested_pytest(env, "test_inner_report", tmp_path / "report.json")
    assert proc.returncode == 0 and "1 passed" in proc.stdout, _out(proc)
    assert data is not None and os.path.normpath(data["home"]) != os.path.normpath(str(stale)), (
        f"T-0175 criterion 3c: the stale marker was taken as the jail: {data}"
    )
    _check_planted_report(data, planted, key, "stale marker")
    assert stale.is_dir(), "the stale directory was removed"


def test_nested_run_keeps_parent_jail(tmp_path: Path) -> None:
    """Row 8 (criterion 9): a nested session with this session's environment reuses this jail, does not take
    ownership, and leaves it in place."""
    jail = _jail()
    proc, data = nested_pytest(dict(os.environ), "test_inner_report", tmp_path / "report.json")
    assert proc.returncode == 0 and "1 passed" in proc.stdout, _out(proc)
    assert data is not None and data["home"] == jail, (
        f"T-0175 criterion 9: the nested run made its own jail: {data}"
    )
    assert data["creator"] == os.environ.get(hj.CREATOR_VAR), data
    assert data["child"].get("status") == "not configured", data
    assert os.path.isdir(jail) and os.path.isdir(os.path.join(jail, ".config")), (
        f"T-0175 criterion 9: the nested run deleted the parent's jail {jail}"
    )


# ------------------------------------------------------------------------------------------------- rows 9-10

REFUSE_CASES = (
    "no-home",
    "empty-home",
    "pre-switch-home",
    "pwd-home-trailing-slash",
    "xdg-under-pwd-home",
    "asyncio-no-home",
    "posix-spawn-no-home",
    "env-none-home-is-pwd-home",
    "os-system-home-is-pwd-home",
    "unreadable-env",
    "exec-no-home",
)
#: the exit code a forked child uses when its ``os.execve`` was refused by the guard with the T-0175 HOME rule
EXEC_REFUSED = 77


class _UnreadableEnv(dict):  # type: ignore[type-arg]
    """An env the guard cannot read (``get`` raises): the guard must refuse it (fail closed), not crash."""

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("this env cannot be read")


def _touch(flag: Path) -> list[str]:
    return ["/bin/sh", "-c", f'touch "{flag}"']


def _path_only() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


async def _attempt(case: str, flag: Path, monkeypatch: pytest.MonkeyPatch) -> BaseException | None:
    """Try the refused spawn of ``case``; the exception it raised, or ``None`` if it ran."""
    argv = _touch(flag)
    try:
        if case == "no-home":
            subprocess.Popen(argv, env=_path_only()).wait(timeout=30)
        elif case == "empty-home":
            subprocess.Popen(argv, env={**_path_only(), "HOME": ""}).wait(timeout=30)
        elif case == "pre-switch-home":
            real = os.environ.get(hj.REAL_HOME_VAR)
            assert real, f"premise: the jail records the pre-switch home in {hj.REAL_HOME_VAR}"
            subprocess.Popen(argv, env={**_path_only(), "HOME": real}).wait(timeout=30)
        elif case == "pwd-home-trailing-slash":
            home = pwd.getpwuid(os.getuid()).pw_dir.rstrip("/") + "/"
            subprocess.Popen(argv, env={**_path_only(), "HOME": home}).wait(timeout=30)
        elif case == "xdg-under-pwd-home":
            xdg = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".config")
            env = {**_path_only(), "HOME": str(flag.parent), "XDG_CONFIG_HOME": xdg}
            subprocess.Popen(argv, env=env).wait(timeout=30)
        elif case == "asyncio-no-home":
            p = await asyncio.create_subprocess_exec(*argv, env=_path_only())
            await asyncio.wait_for(p.wait(), 30)
        elif case == "posix-spawn-no-home":
            pid = os.posix_spawn(argv[0], argv, _path_only())
            os.waitpid(pid, 0)
        elif case == "env-none-home-is-pwd-home":
            monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
            try:
                subprocess.Popen(argv).wait(timeout=30)
            finally:
                monkeypatch.undo()
        elif case == "os-system-home-is-pwd-home":
            monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
            try:
                os.system(f'touch "{flag}"')
            finally:
                monkeypatch.undo()
        elif case == "unreadable-env":
            env = _UnreadableEnv({**_path_only(), "HOME": str(flag.parent)})
            subprocess.Popen(argv, env=env).wait(timeout=30)
        else:  # pragma: no cover
            raise AssertionError(case)
    except AssertionError:
        raise
    except Exception as exc:
        return exc
    return None


def _assert_refusal(err: BaseException | None, case: str, rule: str) -> None:
    assert err is not None, f"T-0175 criterion 5 ({case}): the spawn was not refused"
    assert isinstance(err, hj.JailRefusal) and not isinstance(err, OSError), (
        f"T-0175 criterion 5 ({case}): refused by something other than the guard: {err!r}"
    )
    assert "T-0175" in str(err) and rule in str(err), (
        f"T-0175 criterion 5 ({case}): the refusal does not name T-0175 and the {rule} rule: {err}"
    )


@pytest.mark.parametrize("case", REFUSE_CASES)
async def test_guard_refuses(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Row 9 (criterion 5): the guard refuses before exec, with its own message; the command never ran."""
    flag = tmp_path / "ran"
    if case == "exec-no-home":
        # os.exec* (and so os.spawn*, fork + exec on POSIX) is refused in the process that calls it: fork, and let
        # the child report the refusal through its exit code
        argv = _touch(flag)
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child
            code = 78
            try:
                os.execve(argv[0], argv, _path_only())
            except hj.JailRefusal as exc:
                code = EXEC_REFUSED if "T-0175" in str(exc) and "HOME" in str(exc) else 79
            except BaseException:
                code = 78
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        got = os.waitstatus_to_exitcode(status)
        assert got == EXEC_REFUSED, (
            f"T-0175 criterion 5 ({case}): the child's execve was not refused by the guard (exit {got}; "
            f"0 means the command ran, 78 another error, 79 a refusal without the T-0175 HOME rule)"
        )
        assert not flag.exists(), f"T-0175 criterion 5 ({case}): the refused command ran"
        return
    err = await _attempt(case, flag, monkeypatch)
    rule = (
        "XDG_CONFIG_HOME"
        if case.startswith("xdg")
        else "could not be read"
        if case == "unreadable-env"
        else "HOME"
    )
    _assert_refusal(err, case, rule)
    assert not flag.exists(), f"T-0175 criterion 5 ({case}): the refused command ran"


ALLOW_CASES = (
    "env-none",
    "copy-of-environ",
    "tmp-home",
    "tmp-home-with-planted-key",
    "home-under-jail",
    "os-system",
    "docker-argv-names-the-real-home",
    "home-is-an-ancestor-of-the-real-home",
    "posix-spawn-tmp-home",
)


@pytest.mark.parametrize("case", ALLOW_CASES)
def test_guard_allows(case: str, tmp_path: Path) -> None:
    """Row 10 (criterion 6, the over-wide direction): each case runs and exits 0 under the guard. Premise, in the
    same row: a HOME-less spawn is refused, so the guard is on (an allowed case proves nothing without it)."""
    premise = tmp_path / "premise_ran"
    try:
        subprocess.Popen(_touch(premise), env=_path_only()).wait(timeout=30)
        err: BaseException | None = None
    except Exception as exc:
        err = exc
    _assert_refusal(err, "premise: HOME-less spawn", "HOME")
    assert not premise.exists()

    flag = tmp_path / "ran"
    argv = _touch(flag)
    if case == "os-system":
        assert os.system(f'touch "{flag}"') == 0
    elif case == "posix-spawn-tmp-home":
        try:
            pid = os.posix_spawn(argv[0], argv, {**_path_only(), "HOME": str(tmp_path)})
        except Exception as exc:
            raise AssertionError(f"T-0175 criterion 6 ({case}): refused: {exc!r}") from exc
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, f"T-0175 criterion 6 ({case}): exit {status}"
    else:
        env: dict[str, str] | None
        if case == "env-none":
            env = None
        elif case == "copy-of-environ":
            env = {**os.environ}
        elif case == "tmp-home":
            env = {**_path_only(), "HOME": str(tmp_path)}
        elif case == "tmp-home-with-planted-key":
            planted, _ = planted_home(tmp_path)
            env = {**_path_only(), "HOME": str(planted)}
        elif case == "home-under-jail":
            env = {**_path_only(), "HOME": os.path.join(_jail(), "sub")}
        elif case == "docker-argv-names-the-real-home":
            env = None
            argv = [*argv, "docker", "run", "-e", "HOME=/root", "-e", f"HOME={_pwd_home()}"]
        elif case == "home-is-an-ancestor-of-the-real-home":
            parent = os.path.dirname(_pwd_home())
            if parent == _pwd_home():
                pytest.skip("the password-database home is /: it has no ancestor")
            env = {**_path_only(), "HOME": parent}
        else:  # pragma: no cover
            raise AssertionError(case)
        try:
            code = subprocess.Popen(argv, env=env).wait(timeout=30)
        except Exception as exc:
            raise AssertionError(f"T-0175 criterion 6 ({case}): refused: {exc!r}") from exc
        assert code == 0, f"T-0175 criterion 6 ({case}): exit {code}"
    assert flag.exists(), f"T-0175 criterion 6 ({case}): the allowed command did not run"


# ------------------------------------------------------------------------------------------------- rows 11-12


def test_live_escape_refused_without_opt_in(tmp_path: Path) -> None:
    """Row 11 (criterion 7): requesting the escape without ``ASTRA_LIVE_SEARCH=1`` fails (an error naming T-0175
    and the variable), not a skip; no report is written."""
    planted, _ = planted_home(tmp_path)
    proc, data = nested_pytest(planted_env(planted), "test_inner_escape", tmp_path / "report.json")
    assert proc.returncode != 0, _out(proc)
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert re.search(r"\b1 error\b", summary) and "skipped" not in summary, (
        f"T-0175 criterion 7: the escape without opt-in is not one setup error: {summary!r}\n{_out(proc)}"
    )
    assert "T-0175" in proc.stdout and f"{LIVE_VAR}=1" in proc.stdout, (
        f"T-0175 criterion 7: the failure is not the escape fixture's own message\n{_out(proc)}"
    )
    assert data is None, f"the inner row ran: {data}"


def test_live_escape_reaches_planted_key_with_opt_in(tmp_path: Path) -> None:
    """Row 12 (criterion 7): with ``ASTRA_LIVE_SEARCH=1`` in a nested planted run, the escape's env reaches the
    pre-switch (planted) home and a status child says ``configured``. Only the status word crosses back."""
    planted, key = planted_home(tmp_path)
    env = planted_env(planted)
    env[LIVE_VAR] = "1"
    proc, data = nested_pytest(env, "test_inner_escape", tmp_path / "report.json")
    assert proc.returncode == 0 and "1 passed" in proc.stdout, _out(proc)
    assert data == {"status": "configured"}, (
        f"T-0175 criterion 7: the escape did not reach the pre-switch home: {data}"
    )
    assert (planted / ".config/diffusion_gearvis/brave_api_key").read_bytes() == key


# ------------------------------------------------------------------------------------------------- row 13


def test_escape_permits_the_real_home_only_while_active() -> None:
    """Criterion 7, "only while that fixture is active": in-process, no child. Before ``escape()`` the pre-switch
    home is forbidden and a spawn env naming it is refused; inside, it is permitted; after the block (also after an
    exception inside it), it is forbidden again. A sticky escape (the counter never decremented) fails the last
    two checks. The real home is a string here; nothing is spawned or opened."""
    _jail()
    real = hj.state().real_home
    env = {"PATH": "/usr/bin:/bin", "HOME": real}
    assert real in hj.forbidden_homes(), (
        "premise: the pre-switch home is forbidden outside the escape"
    )
    assert hj.refusal(env, hj.forbidden_homes()) is not None, (
        "premise: a spawn naming it is refused"
    )
    with hj.escape() as got:
        assert got == real
        assert real not in hj.forbidden_homes(), (
            "T-0175 criterion 7: the escape does not permit the home"
        )
        assert hj.refusal(env, hj.forbidden_homes()) is None
    assert real in hj.forbidden_homes(), (
        "T-0175 criterion 7: the escape is still in force after its block ended"
    )
    assert hj.refusal(env, hj.forbidden_homes()) is not None
    try:
        with hj.escape():
            raise LookupError("inside the escape")
    except LookupError:
        pass
    assert real in hj.forbidden_homes(), (
        "T-0175 criterion 7: an exception inside the escape left it in force"
    )


def test_forbidden_homes_hold_the_pwd_home_and_the_pre_switch_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 5's two homes, with distinct strings: on this box the pre-switch home usually *is* the
    password-database home, so a guard that forgot the pwd rule would pass every process row. Here the pwd home is a
    string that names nothing and differs from the recorded one; both must be forbidden. Nothing is spawned."""
    _jail()
    distinct = "/srv/people/pwd-only-t0175"
    monkeypatch.setattr(hj, "pwd_home", lambda: distinct)
    real = hj.state().real_home
    assert real != distinct
    homes = hj.forbidden_homes()
    assert distinct in homes, (
        f"T-0175 criterion 5: the password-database home is not forbidden: {homes}"
    )
    assert real in homes, f"T-0175 criterion 5: the pre-switch home is not forbidden: {homes}"
    assert hj.refusal({"PATH": "/usr/bin:/bin", "HOME": distinct + "/"}, homes) is not None


def test_a_pytest_session_without_a_jail_fails_loudly(tmp_path: Path) -> None:
    """Review F1: if the jail is not installed in a pytest process (here forced: ``loaded_by_pytest`` patched to
    ``False`` before ``pytest.main`` loads the conftest), the session fails at ``pytest_configure`` with the jail's
    message and runs no row. Planted HOME, so the would-be unjailed row could only reach the fake key."""
    planted, _ = planted_home(tmp_path)
    report = tmp_path / "report.json"
    scrub = {"PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST"}
    env = {
        k: v
        for k, v in planted_env(planted).items()
        if k not in scrub and not k.startswith("PYTEST_XDIST_")
    }
    env[REPORT_VAR] = str(report)
    probe = (
        "import sys\n"
        "import tests._home_jail as h\n"
        "h.loaded_by_pytest = lambda: False\n"
        "import pytest\n"
        "sys.exit(pytest.main([sys.argv[1], '-q', '-p', 'no:cacheprovider']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, f"{THIS}::test_inner_report"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"T-0175 F1: an unjailed pytest session ran\n{_out(proc)}"
    assert "the session jail is not installed" in out, (
        f"T-0175 F1: the session did not fail with the jail's own message\n{_out(proc)}"
    )
    assert not report.exists(), "T-0175 F1: a row ran in the unjailed session"


def test_jail_helpers_on_dict_env(tmp_path: Path) -> None:
    """Row 13: ``tests/_home_jail.py`` directly, no processes. The "real homes" are strings that name nothing."""
    assert frozenset(hj.MIRROR) == ALLOWLIST, (
        f"the mirror is not the task's allowlist: {sorted(hj.MIRROR)}"
    )
    for p in hj.MIRROR:
        for never in (".config/diffusion_gearvis", ".local/share/local_astra"):
            assert not (never == p or never.startswith(p + "/")), f"{p} reaches {never}"
    for never in (
        "",
        ".",
        ".config",
        ".local",
        ".local/share",
        ".config/diffusion_gearvis",
        ".local/share/local_astra",
    ):
        assert never not in hj.MIRROR, never

    # "already jailed" means HOME equals an existing marker, never merely a marker
    j = tmp_path / "jail"
    j.mkdir()
    fake_pwd = tmp_path / "pwdhome"
    fake_pwd.mkdir()
    real = "/srv/people/someone-else"
    base = {hj.MARKER: str(j), hj.REAL_HOME_VAR: real}
    assert hj.is_jailed({**base, "HOME": str(j)}, str(fake_pwd)) is True
    assert hj.is_jailed({**base, "HOME": str(j) + "/"}, str(fake_pwd)) is True
    assert hj.is_jailed({**base, "HOME": str(tmp_path / "planted")}, str(fake_pwd)) is False
    assert (
        hj.is_jailed(
            {**base, hj.MARKER: str(tmp_path / "gone"), "HOME": str(tmp_path / "gone")},
            str(fake_pwd),
        )
        is False
    )
    assert hj.is_jailed({"HOME": str(j)}, str(fake_pwd)) is False
    assert (
        hj.is_jailed({hj.MARKER: str(j), "HOME": str(j)}, str(fake_pwd)) is False
    )  # no recorded real home
    assert (
        hj.is_jailed({**base, hj.MARKER: str(fake_pwd), "HOME": str(fake_pwd)}, str(fake_pwd))
        is False
    )
    assert hj.is_jailed({**base, hj.REAL_HOME_VAR: str(j), "HOME": str(j)}, str(fake_pwd)) is False

    homes = ("/srv/people/alice", "/srv/alice-home")
    path = {"PATH": "/usr/bin:/bin"}
    refused: list[dict[Any, Any]] = [
        dict(path),
        {**path, "HOME": ""},
        {**path, "HOME": "/srv/people/alice"},
        {**path, "HOME": "/srv/people/alice/"},
        {**path, "HOME": "/srv//people/alice/."},
        {**path, "HOME": "/srv/alice-home"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/srv/people/alice/.config"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/srv/people/alice"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/srv/alice-home/deep/er"},
        {b"PATH": b"/usr/bin", b"HOME": b"/srv/people/alice"},
    ]
    allowed: list[dict[Any, Any]] = [
        {**path, "HOME": "/tmp/x"},
        {**path, "HOME": "/srv/people"},
        {**path, "HOME": "/srv/people/alice2"},
        {**path, "HOME": "/srv/people/alice/sub"},
        {**path, "HOME": "/tmp/jail/sub"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/tmp/x/.config"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/srv/people/alice2/.config"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": "/srv/people"},
        {**path, "HOME": "/tmp/x", "XDG_CONFIG_HOME": ""},
        {b"PATH": b"/usr/bin", b"HOME": b"/tmp/x"},
    ]
    for env in refused:
        msg = hj.refusal(env, homes)
        assert msg is not None and "T-0175" in msg, f"not refused: {env}"
    for env in allowed:
        assert hj.refusal(env, homes) is None, f"refused: {env}"
