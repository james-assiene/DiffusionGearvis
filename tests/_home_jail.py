"""T-0175: the session home jail and the fail-closed spawn guard, as plain functions (docs/tasks/T-0175.md).

A fresh interpreter does not load conftest, so an in-process patch (``brave_key_nowhere``) never protects a child: a
differential BASE run over an old tree, a CLI rig or a ``uv run`` child reads ``Path.home()`` from ``HOME``. So
``tests/conftest.py`` calls ``install()`` at import, before any sub-conftest or test module is imported:

- **The jail.** ``HOME`` and ``XDG_CONFIG_HOME`` point at a per-session directory (``tempfile.mkdtemp``, never under
  the repo) whose ``.config`` and ``.local/share`` are real, empty directories, and whose only other entries are the
  ``MIRROR`` symlinks to the same path under the pre-switch home (the caches tests rely on). Every child inherits it.
  The session is *already jailed* iff ``HOME`` equals the ``MARKER`` directory, which exists, and a pre-switch home is
  recorded (``is_jailed``); xdist workers and nested pytest runs then reuse it. Anything else (no marker, a stale one)
  makes a fresh jail whose pre-switch home is the current ``HOME``. Only a process pytest is loading conftest in is
  jailed (``loaded_by_pytest``): a fresh interpreter that imports ``tests.conftest`` as a library (a BASE runner
  importing a test module) keeps the ``HOME`` its parent gave it, which is the jail or a tmp home the rig chose.
- **The guard.** An audit hook, installed once per process, refuses any ``subprocess.Popen`` (and so
  ``asyncio.create_subprocess_exec``), ``os.posix_spawn*``, ``os.exec*``, ``os.spawn*`` or ``os.system`` whose
  effective env (the given one, or ``os.environ`` when ``None``) has no or an empty ``HOME``, a ``HOME`` equal to the
  pre-switch home or the password-database home, or an ``XDG_CONFIG_HOME`` equal to or under either (``refusal``). It
  raises ``JailRefusal`` before exec. For the home rules argv is never inspected. T-0186 adds one argv clause
  (``git_refusal``): a spawn whose argv[0] or executable basename is exactly ``git`` and whose effective env carries
  ``GIT_DIR`` or ``GIT_WORK_TREE`` is refused, naming the variable (``-C``/``--git-dir`` on argv is fine).
- **The git strip.** ``strip_git_selectors`` removes ``GIT_SELECTORS`` from ``os.environ``; ``tests/conftest.py`` calls
  it at import beside ``install()`` (T-0186 criterion 1), so no child inherits a repo selector. Accepted gaps of the git
  clause: a git started by a non-git child (``sh -c``, ``env GIT_DIR=… git``, a script) is not seen; the strip covers
  those unless a test hand-builds such an env.
- **The escape.** Inside ``escape()`` (the root conftest's ``real_home_env`` fixture, only under
  ``ASTRA_LIVE_SEARCH=1``) the pre-switch home is permitted, in this process only.

The real home is handled as a string only: ``os.path.normpath``, never ``realpath``, ``resolve`` or ``exists``; the
mirror links are made without looking at their targets (dangling links are fine), and the real home is never listed.
Accepted gaps: ``multiprocessing`` and raw ``_posixsubprocess`` spawns are not seen directly (they inherit
``os.environ``, which the next event checks); grandchildren of a non-pytest child are not guarded (they inherit the
jail unless a script scrubs its env).
"""

from __future__ import annotations

import atexit
import os
import pwd
import shlex
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The jail directory of this top-level session (the "already jailed" marker).
MARKER = "ASTRA_TEST_JAIL"
#: The pre-switch ``HOME``, recorded as a string so children know the real home without having it in ``HOME``.
REAL_HOME_VAR = "ASTRA_TEST_REAL_HOME"
#: The PID of the process that created the jail (the only one that removes it).
CREATOR_VAR = "ASTRA_TEST_JAIL_PID"

#: Relative paths mirrored into the jail as symlinks to the same path under the pre-switch home: the caches (uv,
#: Hugging Face, Playwright, fontconfig under ``.cache``), git config, uv config and pythons, docker, X auth, fonts.
#: Never ``.config`` itself, nothing on the path to ``.config/diffusion_gearvis``, never ``.local/share/local_astra``.
MIRROR: tuple[str, ...] = (
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
)
#: Real (empty) directories in the jail: the parents of the nested mirror entries.
REAL_DIRS: tuple[str, ...] = (".config", ".local", ".local/share")
JAIL_PREFIX = "astra-t0175-jail-"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The audit events the guard looks at; the fast path is one membership test on the event name.
_EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"})
#: Index of the env argument per event (``os.system`` has none: it inherits ``os.environ``).
_ENV_ARG = {"subprocess.Popen": 3, "os.posix_spawn": 2, "os.exec": 2, "os.spawn": 3}

_RULE = (
    "every process a test starts must inherit the session home jail, because the real home holds the Brave "
    "Search API key file (T-0163 criterion 10). Pass env=None, or copy os.environ and change what you need; "
    "a tmp HOME is fine. The only escape is tests/conftest.py's real_home_env, under ASTRA_LIVE_SEARCH=1"
)


#: T-0186: git's repo selectors. ``tests/conftest.py`` strips them from ``os.environ`` at import
#: (``strip_git_selectors``): with ``GIT_DIR`` exported, a sandbox's ``git init <tmp>`` re-initialises the repo it
#: names and the sandbox's commit lands on that repo's branch (2026-10-28, the stray commit on ``main``).
GIT_SELECTORS: tuple[str, ...] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
)
#: T-0186 criterion 2: the variables a ``git`` child may never carry in its env (pass ``-C``/``--git-dir`` instead).
GIT_GUARDED: tuple[str, ...] = ("GIT_DIR", "GIT_WORK_TREE")

_GIT_RULE = (
    "a test's git child must never inherit a repo selector: with GIT_DIR exported, `git init <tmp>` re-initialises "
    "the repo it names and a sandbox commit lands on that repo's branch (T-0186, 2026-10-28). Pass -C <dir> or "
    "--git-dir=<dir> on the command line instead; tests/conftest.py strips the selectors from os.environ"
)


class JailRefusal(RuntimeError):
    """The spawn guard's own refusal (never an ``OSError``, so it cannot be mistaken for a failed exec)."""


@dataclass(frozen=True)
class Jail:
    path: str
    real_home: str
    created: bool
    creator_pid: int


_STATE: Jail | None = None
_GUARD_INSTALLED = False
_ESCAPES = 0
_OWNER: object | None = None


def _norm(p: str) -> str:
    return os.path.normpath(p)


def pwd_home() -> str:
    """The password-database home, as a normalised string ("" if the uid has no entry)."""
    try:
        return _norm(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        return ""


def _get(env: Mapping[Any, Any], name: str) -> str | None:
    """``env[name]`` as a string, for a str- or bytes-keyed env; ``None`` when absent."""
    value: Any = None
    try:
        value = env.get(name)
    except TypeError:
        value = None
    if value is None:
        try:
            value = env.get(name.encode())
        except (TypeError, AttributeError):
            value = None
    if value is None:
        return None
    return os.fsdecode(value) if isinstance(value, bytes) else str(value)


def is_jailed(env: Mapping[str, str], pwd_home: str) -> bool:
    """ "Already jailed": ``HOME`` equals the marker, a pre-switch home is recorded, ``HOME`` is neither that home nor
    ``pwd_home``, and the marker directory exists. The string checks come first, so a real home is never stat'ed."""
    home = env.get("HOME") or ""
    marker = env.get(MARKER) or ""
    real = env.get(REAL_HOME_VAR) or ""
    if not (home and marker and real):
        return False
    h = _norm(home)
    if h != _norm(marker) or h == _norm(real) or (pwd_home and h == _norm(pwd_home)):
        return False
    return os.path.isdir(h)


def _under(path: str, home: str) -> bool:
    if path == home:
        return True
    if home == os.sep:
        return False
    return path.startswith(home.rstrip(os.sep) + os.sep)


def refusal(env: Mapping[Any, Any], homes: Iterable[str]) -> str | None:
    """Why the guard refuses a spawn with effective env ``env`` given the forbidden ``homes``; ``None`` to allow."""
    forbidden = {_norm(h) for h in homes if h}
    home = _get(env, "HOME")
    if not home:
        return f"T-0175 spawn guard: refused a process with {'an empty' if home == '' else 'no'} HOME; {_RULE}"
    if _norm(home) in forbidden:
        return (
            f"T-0175 spawn guard: refused a process whose HOME is the real home ({home!r}); {_RULE}"
        )
    xdg = _get(env, "XDG_CONFIG_HOME")
    if xdg:
        x = _norm(xdg)
        for h in forbidden:
            if _under(x, h):
                return (
                    f"T-0175 spawn guard: refused a process whose XDG_CONFIG_HOME ({xdg!r}) is the real home "
                    f"or under it; {_RULE}"
                )
    return None


def strip_git_selectors(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """T-0186 criterion 1: remove ``GIT_SELECTORS`` from ``environ`` (``os.environ`` by default); the names removed."""
    env = os.environ if environ is None else environ
    removed = [name for name in GIT_SELECTORS if name in env]
    for name in removed:
        del env[name]
    return removed


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return os.fsdecode(value)
    if isinstance(value, os.PathLike):
        return os.fsdecode(os.fspath(value))
    return str(value)


def _spawned_programs(event: str, args: tuple[Any, ...]) -> list[str]:
    """The program names a spawn event starts: its argv[0] and its executable path (when the event has one); for
    ``os.system`` the first word of the shell command."""
    names: list[Any] = []
    if event == "os.system":
        command = _as_str(args[0]) if args else ""
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        names = words[:1]
    else:
        path_i, argv_i = {"subprocess.Popen": (0, 1), "os.spawn": (1, 2)}.get(event, (0, 1))
        if len(args) > path_i and args[path_i] is not None:
            names.append(args[path_i])
        argv = args[argv_i] if len(args) > argv_i else None
        if isinstance(argv, (str, bytes, os.PathLike)):
            names.append(argv)
        elif argv is not None:
            first = next(iter(argv), None)
            if first is not None:
                names.append(first)
    return [_as_str(n) for n in names]


def git_refusal(programs: Iterable[str], env: Mapping[Any, Any]) -> str | None:
    """T-0186 criterion 2: why a spawn of ``programs`` with effective env ``env`` is refused; ``None`` to allow. Only a
    program whose basename is exactly ``git`` is looked at, and only ``GIT_GUARDED`` in its env."""
    if not any(os.path.basename(p) == "git" for p in programs):
        return None
    for name in GIT_GUARDED:
        value = _get(env, name)
        if value is not None:
            return f"T-0186 spawn guard: refused a git process whose env carries {name} ({value!r}); {_GIT_RULE}"
    return None


def make_jail(real_home: str) -> str:
    """A fresh jail mirroring ``MIRROR`` from ``real_home`` (links made without looking at their targets)."""
    jail = tempfile.mkdtemp(prefix=JAIL_PREFIX)
    if _under(_norm(jail), _norm(_REPO)):
        os.rmdir(jail)
        raise RuntimeError(f"T-0175: the temp dir {jail} is under the repo; set TMPDIR elsewhere")
    for d in REAL_DIRS:
        os.makedirs(os.path.join(jail, d), exist_ok=True)
    real = _norm(real_home)
    for rel in MIRROR:
        os.symlink(os.path.join(real, rel), os.path.join(jail, rel))
    return jail


_PYTEST_CONFIG_FILE = os.sep + os.path.join("_pytest", "config", "__init__.py")


def loaded_by_pytest() -> bool:
    """Whether any frame on the call stack is from ``_pytest/config/__init__.py`` (pytest's conftest loader: a
    pytest session's controller, an xdist worker, a nested run). ``False`` for a *library* import of ``tests.conftest`` in a fresh interpreter (a BASE runner or a CLI
    rig child importing a test module): that child keeps the ``HOME`` its parent chose (often a tmp home with a
    planted key), and inherits the jail from its parent's env otherwise."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.endswith(_PYTEST_CONFIG_FILE):
            return True
        frame = frame.f_back
    return False


def install(environ: MutableMapping[str, str] | None = None) -> Jail | None:
    """Put this pytest process in the session jail (reuse it when already jailed) and install the guard. Idempotent.
    A library import (``loaded_by_pytest()`` false) changes nothing and returns ``None``."""
    global _STATE
    env = os.environ if environ is None else environ
    if _STATE is None and not loaded_by_pytest():
        return None
    if _STATE is None:
        if is_jailed(env, pwd_home()):
            jail, real, created = _norm(env["HOME"]), _norm(env[REAL_HOME_VAR]), False
        else:
            real = _norm(env.get("HOME") or pwd_home() or os.sep)
            jail, created = make_jail(real), True
            env[MARKER] = jail
            env[REAL_HOME_VAR] = real
            env[CREATOR_VAR] = str(os.getpid())
        env["HOME"] = jail
        env["XDG_CONFIG_HOME"] = os.path.join(jail, ".config")
        _STATE = Jail(jail, real, created, os.getpid())
        if created:
            # a fresh interpreter that imports tests.conftest outside pytest (a BASE runner importing a test
            # module) has no pytest_unconfigure: remove its jail at exit too, never leak one
            atexit.register(_remove, _STATE)
    install_guard()
    return _STATE


def state() -> Jail:
    if _STATE is None:
        raise RuntimeError(
            "T-0175: the session jail is not installed in this pytest process (tests/conftest.py installs it at "
            "import); refusing to run tests that could reach the real home"
        )
    return _STATE


def forbidden_homes() -> set[str]:
    """The homes no spawn may name: the pre-switch home (unless an escape is active) and the password-database home."""
    out = {pwd_home()}
    if _STATE is not None:
        out.add(_STATE.real_home)
        if _ESCAPES > 0:
            out.discard(_STATE.real_home)
    out.discard("")
    return out


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event not in _EVENTS:
        return
    index = _ENV_ARG.get(event)
    env = args[index] if index is not None and len(args) > index else None
    if env is None:
        env = os.environ
    try:
        msg = refusal(env, forbidden_homes())
        if msg is None:
            msg = git_refusal(_spawned_programs(event, args), env)
    except Exception as exc:  # fail closed: an env the guard cannot read is refused
        msg = (
            f"T-0175 spawn guard: refused a process whose env could not be read ({exc!r}); {_RULE}"
        )
    if msg is not None:
        raise JailRefusal(msg)


def install_guard() -> None:
    """``sys.addaudithook`` cannot be undone, so the hook goes in once per process."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    sys.addaudithook(_audit)
    _GUARD_INSTALLED = True


@contextmanager
def escape() -> Iterator[str]:
    """Permit the pre-switch home in this process while the block runs; yields it."""
    global _ESCAPES
    real = state().real_home
    _ESCAPES += 1
    try:
        yield real
    finally:
        _ESCAPES -= 1


def claim_owner(config: object) -> None:
    """The first pytest config of the process owns the cleanup (an in-process nested ``pytest.main`` does not)."""
    global _OWNER
    if _OWNER is None:
        _OWNER = config


def cleanup(config: object) -> None:
    """At session end the creating process removes its jail; ``rmtree`` unlinks the mirror links, never following."""
    if _STATE is not None and config is _OWNER:
        _remove(_STATE)


def _remove(s: Jail) -> None:
    """Remove ``s`` if this process created it (a forked child never does)."""
    if not s.created or os.getpid() != s.creator_pid:
        return
    if not Path(s.path).name.startswith(JAIL_PREFIX) or os.path.islink(s.path):
        return
    shutil.rmtree(s.path, ignore_errors=True)
