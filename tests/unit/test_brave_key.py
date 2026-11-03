"""T-0163 criteria 3 and 4: the Brave Search API key-file loader, ``local_astra/search/brave_key.py``.

Every row uses an injected path under ``tmp_path`` (or a tmp ``HOME``) and a generated canary key; the real key file
is never named, opened or statted (criterion 10; the autouse ``brave_key_nowhere`` fixture in ``conftest.py`` points
the default path at a nonexistent tmp path for the whole session).

The surface these rows assume (the task names the behaviour, not the names; listed in the test-author's report):

- ``default_key_path() -> Path``: ``Path.home() / ".config/diffusion_gearvis/brave_api_key"``, computed at call time;
- ``load_key(path: Path | None = None)``: ``None`` means ``default_key_path()``; returns a result with
  ``status`` (``"configured"`` or ``"not configured"``, the two words the logs may say), ``reason`` (``None`` when
  configured or absent, else one refusal code of criterion 3), ``message`` (the refusal's text, may name the path,
  the octal mode and the uids, never a byte of the content) and ``key`` (``None`` unless configured; otherwise an
  opaque value that ``BraveApiSearch(key, transport=...)`` accepts).

The loaded key's value is read back only through what the backend sends (the ``X-Subscription-Token`` header over
``tests/fakes/brave_api.py``), never through an accessor, so the rows dictate no way of holding it.

Adversarial (the failure modes the task names): a check on the path instead of what was opened (``stat`` for
``lstat``/``fstat``): ``test_symlink_to_valid_file_refused`` and ``test_check_is_on_what_was_opened`` (the file is
swapped for a symlink between any check and the open, from an audit hook in a fresh interpreter); the mode mask
``0o044`` for ``0o077`` (``test_mode[0620]``, ``[0606]``, ``[0602]``, ``[0610]``, ``[0601]``); a FIFO that blocks the
open (``test_fifo_refused_without_blocking``); a whole-file read (``test_huge_file_read_is_bounded``); content in a
refusal (``test_refusal_message_has_no_content_bytes``). A guard too wide is pinned too: 0600, 0400, a 16- and a
128-character key, the whole charset, surrounding whitespace and a trailing newline are accepted.

Regime: CPU, tmp files only, no network.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import secrets
import stat
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from tests.fakes.brave_api import FakeBrave

REPO = Path(__file__).resolve().parents[2]
TASK = "T-0163"
KEY_MOD = "local_astra.search.brave_key"
API_MOD = "local_astra.search.brave_api"
CODES = frozenset({"symlink", "not_a_file", "owner", "mode", "unreadable", "malformed"})
#: letters that are not hex digits, so a path or a mode in a message cannot match a window of the canary by chance
_ALPHABET = "GHJKMNPQRSTVWXYZghjkmnpqrstvwxyz"


def canary(n: int = 32) -> str:
    """A generated key matching ``^[A-Za-z0-9_-]{16,128}$`` (never a real key)."""
    return "BSAzq" + "".join(secrets.choice(_ALPHABET) for _ in range(n - 5))


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.search"):
            raise NotImplementedError(f"{TASK}: {name} does not exist yet") from exc
        raise


def _key_mod() -> Any:
    return _import(KEY_MOD)


def write_key(path: Path, content: str | bytes, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode() if isinstance(content, str) else content
    path.write_bytes(data)
    path.chmod(mode)
    return path


def sent_key(key: Any) -> str:
    """The key as the backend sends it: one search over the fake API, the ``X-Subscription-Token`` header read
    back."""
    api = _import(API_MOD)
    fake = FakeBrave()

    async def go() -> None:
        backend = api.BraveApiSearch(key, transport=fake.transport())
        try:
            await backend.search("q")
        finally:
            await backend.aclose()

    asyncio.run(go())
    assert len(fake.requests) == 1, len(fake.requests)
    values = fake.requests[0].headers.get_list("x-subscription-token")
    assert len(values) == 1, f"one X-Subscription-Token header; got {len(values)}"
    return str(values[0])


def assert_configured(result: Any, key: str) -> None:
    assert result.status == "configured", (result.status, result.reason, result.message)
    assert result.reason is None, result.reason
    assert result.key is not None
    assert sent_key(result.key) == key, "the loaded key is the file's token"


def assert_refused(result: Any, code: str, *content: str) -> None:
    assert code in CODES
    assert result.status == "not configured", f"a refused file is not configured; {result.status!r}"
    assert result.reason == code, f"refusal code {result.reason!r}, want {code!r}"
    assert result.key is None, "a refused file yields no key"
    assert isinstance(result.message, str) and result.message, "a refusal says why"
    for text in content:
        _no_content(text, result.message, repr(result), str(result))


def _no_content(content: str, *shown: str) -> None:
    """No byte of ``content`` is shown: neither the whole nor any 8-character window of it."""
    windows = {content[i : i + 8] for i in range(max(1, len(content) - 7))} if content else set()
    for s in shown:
        assert content not in s, f"the content appears in {s[:200]!r}"
        hit = sorted(w for w in windows if len(w) == 8 and w in s)
        assert hit == [], f"part of the content appears: {hit[:3]} in {s[:200]!r}"


# --- the default path ---------------------------------------------------------------------------------------------


def test_absent_is_not_configured_and_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    m = _key_mod()
    caplog.set_level(logging.DEBUG)
    result = m.load_key(tmp_path / "no" / "such" / "brave_api_key")
    assert result.status == "not configured"
    assert result.reason is None, "absent is not a refusal"
    assert result.key is None
    out, err = capsys.readouterr()
    assert (out, err) == ("", ""), "absent is silent"
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_default_path_follows_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, brave_key_nowhere: Any
) -> None:
    """Criterion 3: ``Path.home() / ".config/diffusion_gearvis/brave_api_key"``, computed at call time (a second
    ``HOME`` moves it). The unpatched function is the conftest's ``original``; ``HOME`` is a tmp dir throughout."""
    _key_mod()
    original = brave_key_nowhere.original
    assert original is not None, "the conftest fixture found the module and kept the original"
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HOME", str(a))
    assert original() == a / ".config" / "diffusion_gearvis" / "brave_api_key"
    monkeypatch.setenv("HOME", str(b))
    assert original() == b / ".config" / "diffusion_gearvis" / "brave_api_key", (
        "computed at call time"
    )


def test_no_argument_uses_the_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_key()`` reads ``default_key_path()`` at call time (the seam the conftest fixture patches)."""
    m = _key_mod()
    key = canary()
    path = write_key(tmp_path / "k" / "brave_api_key", key + "\n")
    monkeypatch.setattr(m, "default_key_path", lambda: path)
    assert_configured(m.load_key(), key)
    monkeypatch.setattr(m, "default_key_path", lambda: tmp_path / "absent")
    assert m.load_key().status == "not configured"


# --- modes and ownership --------------------------------------------------------------------------------------------

#: mode -> None (accepted) or the refusal code; criterion 3's list plus the other single group/other bits
MODES: dict[str, int | None] = {
    "0600-ok": None,
    "0400-ok": None,
    "0640": 0o640,
    "0604": 0o604,
    "0620": 0o620,
    "0606": 0o606,
    "0644": 0o644,
    "0660": 0o660,
    "0610": 0o610,
    "0602": 0o602,
    "0601": 0o601,
}


@pytest.mark.parametrize("case", list(MODES))
def test_mode(tmp_path: Path, case: str) -> None:
    m = _key_mod()
    key = canary()
    mode = int(case.split("-")[0], 8)
    path = write_key(tmp_path / "brave_api_key", key + "\n", mode)
    assert stat.S_IMODE(path.stat().st_mode) == mode, "premise: the mode was set"
    result = m.load_key(path)
    if MODES[case] is None:
        assert_configured(result, key)
    else:
        assert_refused(result, "mode", key)


def test_owner_mismatch_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner != effective uid. The file cannot be chowned without root, so the effective uid moves instead
    (``os.geteuid``, read at call time)."""
    m = _key_mod()
    key = canary()
    path = write_key(tmp_path / "brave_api_key", key)
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)
    assert_refused(m.load_key(path), "owner", key)


def test_owner_matches_when_euid_is_the_owner(tmp_path: Path) -> None:
    """The other direction of the owner rule: the file's own uid is accepted (not a check against uid 0 or
    the real uid's group)."""
    m = _key_mod()
    key = canary()
    path = write_key(tmp_path / "brave_api_key", key)
    assert path.stat().st_uid == os.geteuid(), "premise"
    assert_configured(m.load_key(path), key)


# --- what kind of file ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["valid_0600", "dangling"])
def test_symlink_to_valid_file_refused(tmp_path: Path, target: str) -> None:
    """A symlink is refused even when it points at a valid 0600 file (no follow)."""
    m = _key_mod()
    key = canary()
    real = tmp_path / "real_key"
    if target == "valid_0600":
        write_key(real, key)
        assert_configured(m.load_key(real), key)  # premise: the target alone is accepted
    link = tmp_path / "brave_api_key"
    link.symlink_to(real)
    assert_refused(m.load_key(link), "symlink", key)


def test_directory_refused(tmp_path: Path) -> None:
    m = _key_mod()
    d = tmp_path / "brave_api_key"
    d.mkdir(mode=0o700)
    assert_refused(m.load_key(d), "not_a_file")


_FIFO_SCRIPT = r"""
import json, sys
from pathlib import Path
from local_astra.search import brave_key
r = brave_key.load_key(Path(sys.argv[1]))
print("RESULT " + json.dumps({"status": r.status, "reason": r.reason}), flush=True)
"""


def test_fifo_refused_without_blocking(tmp_path: Path) -> None:
    """A FIFO is not a regular file. Opening one for reading blocks until a writer appears, so the loader must not
    block on it (``O_NONBLOCK``, or a check before the open). A fresh interpreter, bounded at 20 s."""
    _key_mod()
    fifo = tmp_path / "brave_api_key"
    os.mkfifo(fifo, 0o600)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FIFO_SCRIPT, str(fifo)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("the loader blocked on a FIFO (no O_NONBLOCK)") from exc
    found = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert found, proc.stderr[-2000:]
    got = json.loads(found[-1][len("RESULT ") :])
    assert got == {"status": "not configured", "reason": "not_a_file"}, got


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0000 file")
def test_unreadable_refused(tmp_path: Path) -> None:
    """Mode 0000: no group or other bits (so not ``mode``), owned by the user (so not ``owner``), but it cannot be
    opened for reading."""
    m = _key_mod()
    key = canary()
    path = write_key(tmp_path / "brave_api_key", key, 0o000)
    try:
        assert_refused(m.load_key(path), "unreadable", key)
    finally:
        path.chmod(0o600)


# --- content ----------------------------------------------------------------------------------------------------------


def _malformed(case: str) -> tuple[bytes, str]:
    """(file bytes, a string that must never be shown)."""
    a, b = canary(), canary()
    return {
        "empty": (b"", ""),
        "whitespace": (b"  \n\t \r\n", ""),
        "two_lines": (f"{a}\n{b}\n".encode(), a),
        "two_tokens": (f"{a} {b}\n".encode(), a),
        "too_long": (canary(129).encode(), a),
        "too_short": (canary(15).encode(), ""),
        "non_ascii": ((a[:20] + "é" + a[20:]).encode(), a),
        "base64_chars": ((a[:20] + "+/=" + a[20:]).encode(), a),
        "nul": ((a + "\x00").encode(), a),
        "5kib": (canary(5 * 1024).encode(), ""),
        "invalid_utf8": (a.encode() + b"\xff\xfe", a),
    }[case]


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "whitespace",
        "two_lines",
        "two_tokens",
        "too_long",
        "too_short",
        "non_ascii",
        "base64_chars",
        "nul",
        "5kib",
        "invalid_utf8",
    ],
)
def test_malformed(tmp_path: Path, case: str) -> None:
    m = _key_mod()
    data, secret = _malformed(case)
    path = write_key(tmp_path / "brave_api_key", data)
    result = m.load_key(path)
    assert_refused(result, "malformed", *([secret] if secret else []))
    if case in ("too_long", "5kib", "too_short"):
        _no_content(data.decode(), result.message, repr(result))


@pytest.mark.parametrize(
    "content",
    ["{k}\n", "{k}", "  {k}  \r\n", "\n\t{k}\n\n"],
    ids=["trailing_newline", "bare", "spaces_crlf", "blank_lines_around"],
)
def test_trailing_newline_accepted(tmp_path: Path, content: str) -> None:
    """Surrounding whitespace is stripped (``echo KEY > file`` writes a newline)."""
    m = _key_mod()
    key = canary()
    path = write_key(tmp_path / "brave_api_key", content.format(k=key))
    assert_configured(m.load_key(path), key)


@pytest.mark.parametrize(
    "key",
    [
        "A" * 16,
        "z" * 128,
        "Az09_-Az09_-Az09_-",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    ],
    ids=["len16", "len128", "dash_underscore", "whole_charset"],
)
def test_token_bounds_accepted(tmp_path: Path, key: str) -> None:
    """The guard is not too narrow: the bounds of ``^[A-Za-z0-9_-]{16,128}$`` are accepted."""
    m = _key_mod()
    path = write_key(tmp_path / "brave_api_key", key + "\n")
    assert_configured(m.load_key(path), key)


def test_huge_file_read_is_bounded(tmp_path: Path) -> None:
    """At most 4 KiB is read: a 256 MiB sparse file is refused as malformed without allocating anything near its
    size (``tracemalloc`` peak under 1 MiB)."""
    m = _key_mod()
    path = tmp_path / "brave_api_key"
    with path.open("wb") as f:
        f.truncate(256 * 1024 * 1024)
    path.chmod(0o600)
    tracemalloc.start()
    try:
        result = m.load_key(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert_refused(result, "malformed")
    assert peak < 1024 * 1024, f"the loader allocated {peak} bytes reading a 256 MiB file"


# --- what was opened ---------------------------------------------------------------------------------------------------

_SWAP_SCRIPT = r"""
import json, os, sys
from pathlib import Path

target, other = Path(sys.argv[1]), Path(sys.argv[2])
state = {"swapped": False}

def hook(event, args):
    # one-shot: just before the first open of the key path, replace it with a symlink to another user's-style
    # 0644 file holding a different valid token (a check-then-open race, made deterministic)
    if event != "open" or state["swapped"] or not args or args[0] is None:
        return
    p = args[0]
    name = os.fsdecode(p) if isinstance(p, (str, bytes, os.PathLike)) else ""
    if not name.endswith("brave_api_key"):
        return
    state["swapped"] = True
    tmp_link = target.with_name("swap_link")
    os.symlink(other, tmp_link)
    os.replace(tmp_link, target)

from local_astra.search import brave_key
sys.addaudithook(hook)
r = brave_key.load_key(target)
print("RESULT " + json.dumps({"status": r.status, "reason": r.reason, "swapped": state["swapped"],
                              "holds_other": r.key is not None}), flush=True)
"""


def test_swap_after_the_open_is_judged_by_the_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 15 (reviewer A F1): the path is swapped *after* the open. The opened descriptor is a 0644 file
    holding token A; right after the open a valid 0600 file holding token B replaces the path. Every check must be
    on the descriptor (``fstat``): the result is ``not configured``/``mode``, and token A is never the key. A loader
    that re-stats the path (``stat``/``lstat``) sees the 0600 file, passes, and reads A from the descriptor."""
    m = _key_mod()
    path = write_key(tmp_path / "brave_api_key", canary_a := canary(), 0o644)
    valid = write_key(tmp_path / "valid_b", canary(), 0o600)
    real_open = os.open
    swapped: list[bool] = []

    def open_then_swap(p: Any, flags: int, *a: Any, **k: Any) -> int:
        fd = real_open(p, flags, *a, **k)
        if not swapped and os.fsdecode(p) == str(path):
            os.replace(
                valid, path
            )  # the path now names a valid 0600 file; ``fd`` is still the 0644 one
            swapped.append(True)
        return fd

    monkeypatch.setattr(m.os, "open", open_then_swap)
    result = m.load_key(path)
    monkeypatch.undo()
    assert swapped == [True], "premise: the loader opened the key path with os.open"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "premise: the path now names the valid file"
    assert result.status == "not configured" and result.reason == "mode", (
        f"judged by the path, not the descriptor: {result.status!r}/{result.reason!r}"
    )
    assert result.key is None
    _no_content(canary_a, result.message, repr(result))


def test_check_is_on_what_was_opened(tmp_path: Path) -> None:
    """Criterion 3: every check is on what was actually opened (no follow, then ``fstat`` on the descriptor). The file
    passes any check made on the path, then is swapped for a symlink to a 0644 file with another valid token at the
    moment of the open. A loader that checks the path and then opens it following links reads the other token; a
    correct one refuses (``symlink`` from the no-follow open, or ``mode`` from ``fstat``)."""
    _key_mod()
    target = write_key(tmp_path / "brave_api_key", canary())
    other = write_key(tmp_path / "other_users_key", canary(), 0o644)
    proc = subprocess.run(
        [sys.executable, "-c", _SWAP_SCRIPT, str(target), str(other)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    found = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert found, proc.stderr[-2000:]
    got = json.loads(found[-1][len("RESULT ") :])
    assert got["swapped"], f"premise: the loader opened the key path (audit 'open' event); {got}"
    assert got["status"] == "not configured" and not got["holds_other"], (
        f"the loader read a file it had not checked; {got}"
    )
    assert got["reason"] in ("symlink", "mode"), got


# --- nothing of the content is shown ----------------------------------------------------------------------------------


def _refusal_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path, str]]:
    key = canary()
    cases = [
        ("mode", write_key(tmp_path / "mode" / "brave_api_key", key, 0o644), key),
        ("malformed", write_key(tmp_path / "mal" / "brave_api_key", f"{key} {key}\n"), key),
        ("malformed", write_key(tmp_path / "mal2" / "brave_api_key", f"{key}\n{key}x\n"), key),
        ("malformed", write_key(tmp_path / "mal3" / "brave_api_key", key + "!"), key),
    ]
    real = write_key(tmp_path / "sym" / "real", key)
    link = tmp_path / "sym" / "brave_api_key"
    link.symlink_to(real)
    cases.append(("symlink", link, key))
    return cases


def test_refusal_message_has_no_content_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Criterion 3/4 (adversarial): a refusal's message, the result's ``repr``/``str``, anything printed and any log
    record at DEBUG carry no byte of the file's content, whichever check refused it."""
    m = _key_mod()
    caplog.set_level(logging.DEBUG)
    for code, path, secret in _refusal_cases(tmp_path, monkeypatch):
        result = m.load_key(path)
        assert_refused(result, code, secret)
    # the owner refusal, on a file whose content is a valid token
    key = canary()
    path = write_key(tmp_path / "owner" / "brave_api_key", key)
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)
    result = m.load_key(path)
    assert_refused(result, "owner", key)
    out, err = capsys.readouterr()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    _no_content(key, out, err, logged)


def test_configured_result_does_not_show_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 4: ``repr()``/``str()`` of the loader's result and of its key value never contain the key; nothing
    is printed or logged that does."""
    m = _key_mod()
    caplog.set_level(logging.DEBUG)
    key = canary()
    path = write_key(tmp_path / "brave_api_key", key + "\n")
    result = m.load_key(path)
    assert result.status == "configured"
    shown = [repr(result), str(result), repr(result.key), str(result.key), f"{result.key}"]
    out, err = capsys.readouterr()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    _no_content(key, *shown, out, err, logged)
    assert_configured(result, key)  # premise: it really holds the key


def test_loader_is_stdlib_only() -> None:
    """Criterion 3: ``brave_key.py`` imports the standard library only (and nothing of ``local_astra``'s transports);
    a fresh interpreter importing it loads no third-party module."""
    _key_mod()
    script = (
        "import sys\n"
        "import local_astra.search\n"  # the packages' own imports are not the loader's
        "before = set(sys.modules)\n"
        "import local_astra.search.brave_key\n"
        "new = sorted(m.split('.')[0] for m in set(sys.modules) - before)\n"
        "std = sys.stdlib_module_names\n"
        "print('THIRD', sorted({m for m in new if m not in std and m != 'local_astra'}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "THIRD []" in proc.stdout.splitlines(), proc.stdout
