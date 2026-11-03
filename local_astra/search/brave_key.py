"""The Brave Search API key file loader (T-0163 criterion 3). Standard library only.

The key lives in one file, ``Path.home() / ".config/diffusion_gearvis/brave_api_key"`` (``default_key_path``, computed
at call time), created by the user. ``load_key`` uses it only if it is a regular file the effective user owns with no
group or other permission bits, holding one token (``_TOKEN``) after surrounding whitespace is stripped. Every check is
on what was opened: the path is opened without following a symlink and without blocking (a FIFO), then ``fstat`` on
the descriptor decides; at most ``MAX_READ`` bytes are read.

Outcomes: no file is "not configured" and silent (``reason`` is ``None``); a refusal is "not configured" with one
closed ``reason`` (``REASONS``) and a ``message`` that may name the path, the octal mode and the uids, never a byte of
the content. A configured result holds the key in an ``ApiKey``, whose ``repr``/``str`` never show it. Nothing here
prints or logs.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "MAX_READ",
    "REASONS",
    "ApiKey",
    "KeyFile",
    "default_key_path",
    "load_key",
]

#: The refusal codes (criterion 3).
REASONS: Final = frozenset({"symlink", "not_a_file", "owner", "mode", "unreadable", "malformed"})
#: The most bytes ever read from the file.
MAX_READ: Final = 4096
#: One token, after surrounding whitespace is stripped.
_TOKEN: Final = re.compile(r"[A-Za-z0-9_-]{16,128}")
_WHITESPACE: Final = " \t\r\n"

Reason = Literal["symlink", "not_a_file", "owner", "mode", "unreadable", "malformed"]


def default_key_path() -> Path:
    """Where the key file is: under the current ``HOME``, read at each call."""
    return Path.home() / ".config/diffusion_gearvis/brave_api_key"


class ApiKey:
    """The key, held opaquely: ``repr``/``str``/``format`` never show it. ``reveal`` is for the one header it is
    sent in."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ApiKey(<redacted>)"

    __str__ = __repr__


@dataclass(frozen=True)
class KeyFile:
    """What ``load_key`` found. ``status`` is the only word the logs may say; ``reason`` a refusal code or
    ``None``; ``message`` why it was refused (empty otherwise); ``key`` the key when configured."""

    status: Literal["configured", "not configured"]
    reason: Reason | None
    message: str
    key: ApiKey | None
    path: Path


def _refused(path: Path, reason: Reason, message: str) -> KeyFile:
    return KeyFile("not configured", reason, message, None, path)


def _refused_open(path: Path, err: OSError) -> KeyFile:
    """The open failed: say why without reading anything. A permission error is refined from ``lstat`` (metadata
    only, used for the message, never to accept the file)."""
    if err.errno == errno.ELOOP:
        return _refused(path, "symlink", f"{path} is a symbolic link; it must be the file itself")
    if err.errno in (errno.ENXIO, errno.EISDIR, errno.ENODEV, errno.EOPNOTSUPP):
        return _refused(path, "not_a_file", f"{path} is not a regular file")
    if err.errno in (errno.EACCES, errno.EPERM):
        try:
            st = os.lstat(path)
        except OSError:
            st = None
        if st is not None:
            refused = _check(path, st)
            if refused is not None:
                return refused
    return _refused(
        path,
        "unreadable",
        f"{path} cannot be read ({errno.errorcode.get(err.errno or 0, 'error')})",
    )


def _check(path: Path, st: os.stat_result) -> KeyFile | None:
    """The metadata checks, in order: kind, owner, mode."""
    if stat.S_ISLNK(st.st_mode):
        return _refused(path, "symlink", f"{path} is a symbolic link; it must be the file itself")
    if not stat.S_ISREG(st.st_mode):
        return _refused(path, "not_a_file", f"{path} is not a regular file")
    euid = os.geteuid()
    if st.st_uid != euid:
        return _refused(
            path, "owner", f"{path} is owned by uid {st.st_uid}, not by you (uid {euid})"
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        return _refused(
            path, "mode", f"{path} has mode {mode:04o}; it must have no group or other permissions"
        )
    return None


def _read_bounded(fd: int) -> bytes:
    """At most ``MAX_READ`` bytes."""
    chunks: list[bytes] = []
    got = 0
    while got < MAX_READ:
        chunk = os.read(fd, MAX_READ - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def load_key(path: Path | None = None) -> KeyFile:
    """Load the key from ``path`` (``default_key_path()`` when ``None``). Never raises for anything about the file;
    see the module docstring for the outcomes."""
    if path is None:
        path = default_key_path()
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOCTTY
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        return KeyFile("not configured", None, "", None, path)
    except OSError as err:
        return _refused_open(path, err)
    try:
        st = os.fstat(fd)
        refused = _check(path, st)
        if refused is not None:
            return refused
        if st.st_size > MAX_READ:
            return _refused(
                path, "malformed", f"{path} is longer than {MAX_READ} bytes; it must hold one key"
            )
        try:
            data = _read_bounded(fd)
        except OSError:
            return _refused(path, "unreadable", f"{path} cannot be read")
    finally:
        os.close(fd)
    if len(data) >= MAX_READ:  # it grew after ``fstat``: never parse a cut read
        return _refused(
            path, "malformed", f"{path} is {MAX_READ} bytes or longer; it must hold one key"
        )
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return _refused(path, "malformed", f"{path} is not one key: it holds non-ASCII bytes")
    token = text.strip(_WHITESPACE)
    if _TOKEN.fullmatch(token) is None:
        return _refused(
            path,
            "malformed",
            f"{path} is not one key: it must hold a single token of 16 to 128 letters, digits, '_' or '-'",
        )
    return KeyFile("configured", None, "", ApiKey(token), path)
