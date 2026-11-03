"""T-0226 group A: the harness ``scripts/gpu/t0226_real_attempts.py`` (unit; CPU; fakes; no engine, no socket).
docs/tasks/T-0226.md A1-A10 and its "Tests to write" list.

BASE 67c6bde3 (main with the task file). Activated at ``/implement T-0226`` (rulings R-1..R-3 in the task file):
no row carries ``pending``. :func:`harness` loads the harness by path lazily and a missing file is a named
``AssertionError``, so collection never errors. Two rows are green by design: the privacy guard over the tracked
files and the eval oracle's self-check.

**Privacy.** Real student pages live in a private directory outside the repo. Nothing here reads, names or points
at it. Every manifest is synthetic: four Pillow pages generated at runtime in ``tmp_path`` at 1240x1755, 1755x1240,
1241x1754 and 875x1241, each with drawn text lines, invented ids (``pg_*``, ``zq-*``) and invented keywords. No image
or PDF is committed. Rows that need a directory inside the repo create a uniquely named one under ``REPO/logs/``
(git-ignored) and remove it afterwards.

**The harness surface these rows assume** (the test-author's reading of the task, reported to the orchestrator; the
implementer may rename only by telling the reviewer)

CLI
    ``main(argv) -> int`` with ``--manifest PATH``, ``--manifest-sha256 HEX`` (required, no default),
    ``--base-url``, ``--out-dir``; exit 2 refused (A2, one stderr line naming the item), 1 no engine / incomplete,
    3 void, 0 measured. ``default_out(sha) -> Path`` is ``REPO/logs/t0226-real-attempts-<sha7>``.
Seams (as T-0223)
    ``_git(*args)`` is the only way git runs in ``REPO`` (``status --porcelain``, ``log --merges``,
    ``rev-parse HEAD``, ``check-ignore -q <out>``); the rig replaces it in every held module and delegates
    everything it does not fake to the real git. ``git -C <manifest dir> rev-parse --is-inside-work-tree`` may run
    any way. The engine is reached only through T-0223 / T-0131 / T-0111g helpers held by name: ``G._get_text``
    (``/models``, ``/version``), ``_probe_generate`` (P0), ``T._healthy``, ``T._tokenize_count`` /
    ``T._post_json`` (``/tokenize``), ``G._wait_idle``, ``G.read_metrics``, ``T._VramSampler``; the one-turn
    function is T-0223's ``_one_turn`` on the private instance. ``PROCESS_START_S`` (module level) is the budget's
    origin.
The private T-0223 instance
    A module held by the harness whose ``__file__`` is ``scripts/gpu/t0223_handwritten_check.py`` and whose
    ``__name__`` is not T-0223's own; its ``TUTOR_INSTRUCTIONS``, ``QUESTION`` and ``page_path`` are replaced once.
Constants
    ``TUTOR_INSTRUCTIONS``, ``QUESTION``, ``ORDER_SEED = 0``, ``TURN_TIMEOUT_S = 90``, ``RUN_BUDGET_S = 1200``,
    ``ERRORED_STOP = 3``, ``VISUAL_TOKENS = 1120``, ``NEGATED_CLAIM``, ``NO_MISTAKE``, ``CLAIM`` (sequences of str).
Pure functions
    ``page_order(ids) -> list[str]``; ``budget_stop(elapsed_s) -> bool`` (True: no further turn starts);
    ``judge_turn(kind, spoken, shown, judge_keywords, status="ran") -> Mapping`` with ``verdict``,
    ``spoken_only_verdict``, ``claims_mistake``, ``says_correct``, ``said_correct`` (a missed mistake page that said
    correct) and ``matched``; ``thought_measures(records) -> Mapping`` with ``chars`` and ``canvases``;
    ``thought_tokens(base_url, text) -> int | None``; ``frame_jpeg`` (T-0223's, re-exported or on the instance).
Report
    ``write_report(out, manifest_path)`` writes ``report.json`` and ``report.md`` from the run dir and the private
    manifest, re-deriving every turn from its raw trial (T-0223's raw fields plus ``thought_tokens``). Shapes:
    ``tests/eval/test_t0226_real_attempts.py``'s docstring (``pages`` rows keyed ``id``, ``counts``, ``reading``,
    ``latency``, ``thinking``, ``phrases``, ``regime``, ``label``). Run-dir files: ``order.json`` (a list of ids),
    ``manifest.json`` (``manifest_sha256``, ``manifest_dir_name``, ``regime``, ``stopped``), ``probes.json``
    (``P-S``, ``P0`` with ``image_tokens`` by id), ``trials.jsonl``, ``turns.jsonl``, ``sessions/``.

Readings and gaps (reported, not resolved silently)

1. The "symlink placed inside the repo" row is read as a refusal, like its neighbour rows: A2 says the directory is
   checked "resolved with symlinks followed", which alone would let a link inside the repo that points outside it
   through, but the task lists the row among the refusal rows and a link in the repo would name the private path.
   Its mirror (a link outside the repo pointing into it) is refused by the resolved check.
2. The task's example "1755x1240 -> 1566x1106" is not what ``policy.target_size`` gives (1568x1106 at this BASE;
   portrait 1240x1755 -> 1106x1568). The rows assert ``policy.target_size`` as the task's rule says, plus the
   875x1241 -> 874x1240 example, which agrees.
3. Page-level refusals must name the page id (so the order rows can tell two pages apart). "Inside the repo" is
   checked as a refusal of the given path too, not only the resolved one (reading 1).
4. Verdict strings, errored, phrase-list minimum, clause split, thinking, latency, label and reading: the eval's
   readings 1-8. ``thought_chars`` is the sum of ``len`` over the thought commits.
5. The P-S page: "a synthetic white page" is checked as mostly white (mean grey > 250) at the first ordered page's
   size, outside the manifest dir and the repo, gone after the run.
6. ``logs/vllm.log`` is written by the verbatim boot block the wrapper must copy; the wrapper row allows it beside
   the two named outputs ("writes only" reads as "only these run outputs").
7. A10 (T-0223's files byte-identical) has no row here: it is a diff check for the reviewer, and a content pin would
   outlive the task.

Regime: CPU, synthetic pages and hand-built session logs in the real record shapes, n = 1 per page, no engine.
"""

from __future__ import annotations

import base64
import copy
import functools
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0226_real_attempts.py"
T0223_PATH = REPO / "scripts" / "gpu" / "t0223_handwritten_check.py"
WRAPPER = REPO / "scripts" / "locked" / "t0226_real_attempts.sh"
PROMPTS_JS = REPO / "hud" / "src" / "settings" / "prompts.js"
LOGS = REPO / "logs"
#: the task's base (main with the task file), a literal
BASE = "67c6bde3"

#: T-0229 A2: the typed question in force (James, 2026-11-02: "something like 'can you double check my work'")
QUESTION = "Can you double-check my work?"
#: T-0229 A1: the "Education tutor" built-in in force, verbatim from docs/tasks/T-0229.md A1
TUTOR = (
    "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, point to it, "
    "explain why, and ask me a guiding question before giving the fix; then give the corrected step and the "
    "correct final answer. If everything is right, say so plainly and don't invent a problem. Speak in plain "
    "words, not symbols, and keep each reply short."
)
#: the T-0226 B run's question and tutor text (2789ec90; docs/tasks/T-0226.md, T-0225), historical literals
T0226_QUESTION = "Here's my work on this page. Is it right? If there's a mistake, tell me where."
T0226_TUTOR = (
    "You are a patient tutor. When I show you my work, find the line with the mistake, point to it, and explain "
    "why it is wrong. Ask me a guiding question before giving the fix, then give the corrected step and the "
    "correct final answer. Speak in plain words, not symbols, and keep each reply short."
)
T0223_QUESTION = "Here's my attempt. Can you check it and tell me where I went wrong?"
T0223_TUTOR = (
    "You are a patient tutor. Point to the line with the mistake and explain why it is wrong, then give the "
    "corrected step."
)
HEADER = "The user's own instructions follow. Apply them unless they conflict with anything above."
SYS_BASE = "You are Gearvis, a local assistant. Keep answers short."
MANIFEST_SHA_PIN = "3a8693b94e640c3244f2a527e9cc95ac588ba8c810cd9fdb065c5b07f4d32bed"
#: the private directory's basename, built from two pieces so this file does not contain it
NEEDLE = "private_" + "attempts"
SIZES = ((1240, 1755), (1755, 1240), (1241, 1754), (875, 1241))
VALID_URL = "http://127.0.0.1:9/v1"
GIT_SHA = "7c226a0" + "e" * 33
REQUIRED_MERGES = (
    "aaaa226aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa Merge T-0226-a: the real-attempts harness",
    "bbbb223bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Merge T-0223-b: the handwritten check harness",
    "cccc213ccccccccccccccccccccccccccccccccc Merge T-0213-a2a: apply panel settings at the next idle turn boundary",
    "dddd225ddddddddddddddddddddddddddddddddd Merge T-0225: built-in instruction presets",
)
OTHER_MERGES = (
    "5237da2d4f67fbc53beaf8f5b9a3972cce1f93dd Merge T-0215: --search off in live does not offer google_search",
)
LOOKALIKE_MERGES = (
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'main' into T-0226-a",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge T-0226-ab: something else",
)
CS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":'
    '"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}'
)


# =====================================================================================================
# loading
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0226: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0226_real_attempts_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), f"T-0226: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def private() -> ModuleType:
    """The T-0223 instance the harness holds (A10, implementation notes): exactly one held module loaded from
    T-0223's file."""
    found = [
        v
        for v in vars(harness()).values()
        if isinstance(v, ModuleType)
        and getattr(v, "__file__", None)
        and Path(str(v.__file__)).resolve() == T0223_PATH.resolve()
    ]
    assert len({id(m) for m in found}) == 1, (
        f"T-0226: the harness holds {len(found)} T-0223 instances"
    )
    return found[0]


def fn(name: str) -> Any:
    """A T-0223 helper the harness re-exports, else the private instance's."""
    h = harness()
    return getattr(h, name) if hasattr(h, name) else getattr(private(), name)


@functools.cache
def _eval() -> ModuleType:
    from importlib import import_module

    return import_module("tests.eval.test_t0226_real_attempts")


def _held(h: ModuleType) -> list[ModuleType]:
    """The harness and every ``scripts/gpu`` module it holds by name, five levels down."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _ in range(5):
        nxt = []
        for mod in frontier:
            for v in vars(mod).values():
                if (
                    isinstance(v, ModuleType)
                    and id(v) not in seen
                    and "/scripts/gpu/" in str(getattr(v, "__file__", "") or "")
                ):
                    seen[id(v)] = v
                    nxt.append(v)
        frontier = nxt
    return list(seen.values())


def _patch_name(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> int:
    n = 0
    for mod in _held(h):
        if hasattr(mod, name):
            mp.setattr(mod, name, value)
            n += 1
    return n


def _patch_object(mp: pytest.MonkeyPatch, h: ModuleType, original: Any, value: Any) -> int:
    n = 0
    for mod in _held(h):
        for k, v in list(vars(mod).items()):
            if v is original:
                mp.setattr(mod, k, value)
                n += 1
    return n


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _order_oracle(ids: Sequence[str]) -> list[str]:
    """A3 by hand: sorted, then one ``random.Random(0).shuffle``."""
    out = sorted(ids)
    random.Random(0).shuffle(out)
    return out


# =====================================================================================================
# the audit hook: sockets and writes (registered once; toggled per call)
# =====================================================================================================

_AUDIT: dict[str, Any] = {"on": False, "sockets": [], "writes": []}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_PATH_EVENTS = {
    "os.mkdir": 0,
    "os.remove": 0,
    "os.rmdir": 0,
    "os.rename": 1,
    "os.symlink": 1,
    "os.link": 1,
    "os.truncate": 0,
    "shutil.copyfile": 1,
    "shutil.copytree": 1,
    "shutil.move": 1,
    "shutil.rmtree": 0,
}
#: where each event's ``dir_fd`` arguments start (``os.mkdir(path, mode, dir_fd)``, ``os.remove(path, dir_fd)``, ...)
_DIR_FD_FROM = {"os.mkdir": 2, "os.remove": 1, "os.rmdir": 1, "os.rename": 2}


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _AUDIT["on"]:
        return
    if event in ("socket.connect", "socket.getaddrinfo"):
        _AUDIT["sockets"].append((event, repr(args)[:200]))
        return
    path: Any = None
    if event == "open":
        target, mode, flags = [*list(args), None, None, None][:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        )
        if writing and isinstance(target, str | bytes | os.PathLike):
            path = target
    elif event in _PATH_EVENTS:
        fd_from = _DIR_FD_FROM.get(event)
        if fd_from is not None and any(isinstance(a, int) and a >= 0 for a in args[fd_from:]):
            return  # relative to a dir fd (rmtree internals): the shutil.rmtree event names the root
        idx = _PATH_EVENTS[event]
        path = args[idx] if len(args) > idx else None
    if path is None:
        return
    try:
        p = os.fsdecode(path)
    except TypeError:
        return
    if "__pycache__" in p or p.endswith(".pyc"):
        return
    if "/astra-t0175-jail-" in p:
        return  # the suite's HOME jail (tests/_home_jail.py) mirrors caches lazily; never the harness's doing
    _AUDIT["writes"].append((event, os.path.abspath(p)))


sys.addaudithook(_audit)

TMPROOT = Path(tempfile.gettempdir()).resolve()


def _under(p: str | Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(p)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


def _ps_roots(writes: Sequence[tuple[str, str]]) -> list[Path]:
    """Temporary directories created directly under the system temp root during the call (P-S's)."""
    return [
        Path(p)
        for e, p in writes
        if e == "os.mkdir" and Path(os.path.realpath(p)).parent == TMPROOT
    ]


def _bad_writes(
    writes: Sequence[tuple[str, str]], allowed: Sequence[Path], ps: bool
) -> list[tuple[str, str]]:
    roots = list(allowed) + (_ps_roots(writes) if ps else [])
    return [(e, p) for e, p in writes if not any(_under(p, r) for r in roots)]


# =====================================================================================================
# synthetic pages and manifests
# =====================================================================================================


@functools.cache
def _png(size: tuple[int, int], label: str) -> bytes:
    """A phone-scan-like page: off-white paper, ruled lines and drawn text lines (so its phash is not constant)."""
    from PIL import Image, ImageDraw

    w, h = size
    im = Image.new("RGB", size, (232, 228, 214))
    d = ImageDraw.Draw(im)
    for y in range(90, h, 70):
        d.line([(30, y), (w - 30, y)], fill=(160, 175, 205), width=2)
    for i, y in enumerate(range(120, min(h, 900), 140)):
        d.text((60 + 13 * i, y), f"{label} ligne {i + 1}: x + {i} = {2 * i}", fill=(25, 30, 110))
    d.rectangle([w // 5, h // 3, w // 5 + 300, h // 3 + 120], outline=(180, 30, 30), width=5)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


DEFAULT_PAGES: tuple[dict[str, Any], ...] = (
    {
        "id": "pg_m1",
        "png": "scan_m1.png",
        "size": SIZES[0],
        "kind": "mistake",
        "key": "invented key: line three drops a sign",
        "judge_keywords": ["ligne 3|line 3", "x = 4"],
    },
    {
        "id": "pg_m2",
        "png": "scan_m2.png",
        "size": SIZES[1],
        "kind": "mistake",
        "key": "invented key: the speed unit",
        "judge_keywords": ["vitesse|speed"],
    },
    {
        "id": "pg_c1",
        "png": "scan_c1.png",
        "size": SIZES[2],
        "kind": "correct",
        "key": "invented key: all fine",
        "judge_keywords": ["c1 note"],
    },
    {
        "id": "pg_u1",
        "png": "scan_u1.png",
        "size": SIZES[3],
        "kind": "unclear",
        "key": "invented key: unreadable",
        "judge_keywords": [],
    },
)
PAGE_IDS = tuple(p["id"] for p in DEFAULT_PAGES)
SIZE_OF = {p["id"]: p["size"] for p in DEFAULT_PAGES}


def _entry(obj: dict[str, Any], pid: str) -> dict[str, Any]:
    return next(p for p in obj["pages"] if p.get("id") == pid)


def write_manifest_dir(
    d: Path,
    pages: Sequence[Mapping[str, Any]],
    edits: Sequence[Callable[[dict[str, Any], Path], None]] = (),
) -> tuple[Path, str]:
    """Write the PNGs and ``manifest.json`` (A1's shape) into ``d``; return the manifest path and its sha256."""
    d.mkdir(parents=True, exist_ok=True)
    obj: dict[str, Any] = {"version": 1, "pages": []}
    for p in pages:
        data = _png(tuple(p["size"]), str(p["id"]))
        (d / str(p["png"])).write_bytes(data)
        obj["pages"].append(
            {
                "id": p["id"],
                "png": p["png"],
                "sha256": _sha(data),
                "kind": p["kind"],
                "key": p["key"],
                "judge_keywords": list(p["judge_keywords"]),
            }
        )
    for e in edits:
        e(obj, d)
    raw = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    (d / "manifest.json").write_bytes(raw)
    return d / "manifest.json", _sha(raw)


# =====================================================================================================
# the rig: refused sockets, a fake git in every held module, a synthetic manifest, repo scratch dirs
# =====================================================================================================


@pytest.fixture
def scratch() -> Iterator[list[Path]]:
    made: list[Path] = []
    yield made
    for p in made:
        if p.is_symlink():
            p.unlink()
        elif p.exists():
            shutil.rmtree(p, ignore_errors=True)


class Rig:
    """Every real socket refused and recorded; a clean tree with the four merges through the harness's ``_git``;
    a synthetic manifest directory materialised lazily (so faults compose); the out-dir a fresh, git-ignored path
    under ``REPO/logs``."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch, scratch: list[Path]) -> None:
        self.h = h = harness()
        self.mp = mp
        self.tmp = tmp_path
        self.scratch = scratch
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*OTHER_MERGES, *REQUIRED_MERGES)
        self.not_ignored = False
        self.url = VALID_URL
        self.out: Path = self.repo_path("out")
        self.pages: list[dict[str, Any]] = [copy.deepcopy(p) for p in DEFAULT_PAGES]
        self.edits: list[Callable[[dict[str, Any], Path], None]] = []
        self.posts: list[Callable[[Path], None]] = []
        self.tail = b""
        self.raw_override: bytes | None = None
        self.sha_transform: Callable[[str], str] | None = None
        self.arg_override: Callable[[Path], Path] | None = None
        self.location = "tmp"
        self.manifest_dir: Path | None = None
        self.manifest_arg: Path | None = None
        self.sha: str | None = None
        rig = self

        def refuse(what: str) -> Callable[..., Any]:
            def f(*a: Any, **k: Any) -> Any:
                rig.sockets.append(what)
                raise ConnectionRefusedError(f"T-0226 rig: no {what}")

            return f

        for name in ("connect", "connect_ex"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            a = [str(x) for x in args]
            if a[:1] == ["status"]:
                return rig.porcelain
            if "log" in a:
                if any("%H" in x for x in a):
                    return "\n".join(rig.merges) + "\n"
                return "\n".join(m.split(" ", 1)[1] for m in rig.merges) + "\n"
            if "check-ignore" in a and rig.not_ignored:
                raise subprocess.CalledProcessError(1, ["git", *a])
            if "rev-parse" in a and a[-1] == "HEAD":
                return (GIT_SHA[:7] if any(x.startswith("--short") for x in a) else GIT_SHA) + "\n"
            return subprocess.run(
                ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True
            ).stdout

        api("_git")
        _patch_name(mp, h, "_git", git)

    def repo_path(self, tag: str) -> Path:
        """A fresh path under ``REPO/logs`` (git-ignored), removed after the row."""
        LOGS.mkdir(exist_ok=True)
        p = LOGS / f"t0226-unit-{tag}-{uuid.uuid4().hex[:10]}"
        self.scratch.append(p)
        return p

    def materialise(self) -> None:
        loc = self.location
        if loc in ("repo", "symlink-into-repo"):
            real = self.repo_path("scans")
        elif loc == "gitinit":
            work = self.tmp / "gitwork"
            work.mkdir()
            subprocess.run(["git", "init", "-q", str(work)], check=True, capture_output=True)
            real = work / "scans"
        else:
            real = self.tmp / "scans"
        path, sha = write_manifest_dir(real, self.pages, self.edits)
        if self.raw_override is not None:
            path.write_bytes(self.raw_override)
            sha = _sha(self.raw_override)
        if self.tail:
            with path.open("ab") as fh:
                fh.write(self.tail)
        for post in self.posts:
            post(real)
        arg_dir = real
        if loc == "symlink-in-repo":
            link = self.repo_path("link")
            link.symlink_to(real, target_is_directory=True)
            arg_dir = link
        elif loc == "symlink-into-repo":
            link = self.tmp / "link"
            link.symlink_to(real, target_is_directory=True)
            arg_dir = link
        self.manifest_dir = real
        self.manifest_arg = (
            self.arg_override(arg_dir) if self.arg_override else arg_dir / "manifest.json"
        )
        self.sha = self.sha_transform(sha) if self.sha_transform else sha

    def argv(self) -> list[str]:
        if self.manifest_arg is None:
            self.materialise()
        return [
            "--manifest",
            str(self.manifest_arg),
            "--manifest-sha256",
            str(self.sha),
            "--base-url",
            self.url,
            "--out-dir",
            str(self.out),
        ]

    def main(self, argv: list[str] | None = None) -> int:
        argv = self.argv() if argv is None else argv
        _AUDIT["sockets"] = []
        _AUDIT["writes"] = []
        _AUDIT["on"] = True
        try:
            rc = self.h.main(argv)
        finally:
            _AUDIT["on"] = False
        return int(rc)

    def writes(self) -> list[tuple[str, str]]:
        return list(_AUDIT["writes"])


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: list[Path]) -> Rig:
    return Rig(tmp_path, monkeypatch, scratch)


# =====================================================================================================
# A2: the refusal table (item number, patterns all required in stderr, the fault)
# =====================================================================================================


def _set(attr: str, value: Any) -> Callable[[Rig], None]:
    def f(r: Rig) -> None:
        setattr(r, attr, value)

    return f


def _edit(e: Callable[[dict[str, Any], Path], None]) -> Callable[[Rig], None]:
    def f(r: Rig) -> None:
        r.edits.append(e)

    return f


def _post(p: Callable[[Path], None]) -> Callable[[Rig], None]:
    def f(r: Rig) -> None:
        r.posts.append(p)

    return f


def _no_merge(prefix: str) -> Callable[[Rig], None]:
    def f(r: Rig) -> None:
        r.merges = tuple(m for m in r.merges if prefix not in m)

    return f


def _branch_merge_only(r: Rig) -> None:
    r.merges = (*(m for m in r.merges if "Merge T-0226-a:" not in m), *LOOKALIKE_MERGES)


def _out_exists(r: Rig) -> None:
    r.out.mkdir(parents=True)
    (r.out / "keep.txt").write_text("x", encoding="utf-8")


def _out_outside_logs(r: Rig) -> None:
    r.out = r.tmp / "out"


def _out_lookalike(r: Rig) -> None:
    p = REPO / f"logs_t0226-unit-{uuid.uuid4().hex[:10]}"
    r.scratch.append(p)
    r.out = p


def _out_dotdot(r: Rig) -> None:
    name = f"t0226-unit-dotdot-{uuid.uuid4().hex[:10]}"
    r.scratch.append(REPO / name)
    r.out = LOGS / ".." / name


def _sha_tf(t: Callable[[str], str]) -> Callable[[Rig], None]:
    return _set("sha_transform", t)


def _manifest_missing(r: Rig) -> None:
    r.arg_override = lambda d: d / "absent.json"


def _manifest_is_dir(r: Rig) -> None:
    r.arg_override = lambda d: d
    r.sha_transform = lambda s: "0" * 64


def _dup_id(obj: dict[str, Any], d: Path) -> None:
    obj["pages"].append(dict(_entry(obj, "pg_u1")))


def _missing_id(obj: dict[str, Any], d: Path) -> None:
    del _entry(obj, "pg_u1")["id"]


def _bad_kind(obj: dict[str, Any], d: Path) -> None:
    _entry(obj, "pg_u1")["kind"] = "maybe"


def _png_dotdot(pid: str, name: str) -> Callable[[dict[str, Any], Path], None]:
    def e(obj: dict[str, Any], d: Path) -> None:
        entry = _entry(obj, pid)
        (d.parent / "x.png").write_bytes(
            (d / name).read_bytes()
        )  # same bytes: only the escape is wrong
        entry["png"] = "../x.png"

    return e


def _png_absolute(obj: dict[str, Any], d: Path) -> None:
    _entry(obj, "pg_m2")["png"] = str(
        (d / "scan_m2.png").resolve()
    )  # absolute, even though it is inside


def _png_symlink_out(obj: dict[str, Any], d: Path) -> None:
    outside = d.parent / "outside_m2.png"
    outside.write_bytes((d / "scan_m2.png").read_bytes())
    (d / "scan_m2.png").unlink()
    (d / "scan_m2.png").symlink_to(outside)


def _unlink(name: str) -> Callable[[Path], None]:
    def p(d: Path) -> None:
        (d / name).unlink()

    return p


def _flip(name: str) -> Callable[[Path], None]:
    def p(d: Path) -> None:
        data = bytearray((d / name).read_bytes())
        data[len(data) // 2] ^= 0x01
        (d / name).write_bytes(bytes(data))

    return p


def _kw(pid: str, value: Any) -> Callable[[dict[str, Any], Path], None]:
    def e(obj: dict[str, Any], d: Path) -> None:
        if value is KeyError:
            del _entry(obj, pid)["judge_keywords"]
        else:
            _entry(obj, pid)["judge_keywords"] = value

    return e


def _instructions_refused(r: Rig) -> None:
    from local_astra.conductor import prompt_builder

    real = prompt_builder.validate_instructions
    tutor = api("TUTOR_INSTRUCTIONS")

    def refusing(text: str) -> str | None:
        return "T0226-RIG-REFUSAL" if text == tutor else real(text)

    r.mp.setattr(prompt_builder, "validate_instructions", refusing)
    _patch_object(r.mp, r.h, real, refusing)


def _visual_560(r: Rig) -> None:
    from local_astra import cli

    r.mp.setattr(cli, "LIVE_DEFAULT_VISUAL_TOKENS", 560)


def _ps_fails(r: Rig) -> None:
    from local_astra.conductor import prompt_builder

    real_sys = prompt_builder.PromptBuilder.system_message

    def dropped(self: Any) -> Any:
        msg = real_sys(self)
        text = msg.content[0].text
        cut = text.split("\n" + HEADER, 1)[0]
        return msg.model_copy(update={"content": (prompt_builder.TextPart(text=cut),)})

    r.mp.setattr(prompt_builder.PromptBuilder, "system_message", dropped)


_URL = r"loopback"
_DIRTY = r"scratch_t0226\.txt"
_EXISTS = r"exist|already"
_OUTDIR = r"logs|ignore"
_SHA_BAD = r"malformed|lowercase|\bhex\b"
_NOT_FILE = r"regular file|not a file|no such file|does not exist|is a directory|missing"
_SHA_DIFF = r"sha256|sha-256|digest|hash"
_IN_REPO = r"\brepo(sitory)?\b|work tree|work-tree|\bgit\b"
_ESCAPE = r"absolute|outside|escape|traversal|contains|not allowed|relative"
_MISSING = r"missing|not found|no such|does not exist|absent|not a (regular )?file"
_KEYWORDS = r"keyword"
_LIST = r"keyword|list"

#: case -> (A2 item 1..17, required stderr patterns, fault)
REFUSALS: dict[str, tuple[int, tuple[str, ...], Callable[[Rig], None]]] = {
    "url-not-loopback": (1, (_URL,), _set("url", "http://10.0.0.1:8001/v1")),
    "url-localhost-lookalike": (1, (_URL,), _set("url", "http://localhost.example:8001/v1")),
    "dirty-untracked": (2, (_DIRTY,), _set("porcelain", "?? scratch_t0226.txt\n")),
    "dirty-modified": (
        2,
        (r"session\.py",),
        _set("porcelain", " M local_astra/conductor/session.py\n"),
    ),
    "out-dir-exists": (3, (_EXISTS,), _out_exists),
    "out-dir-outside-logs": (4, (_OUTDIR,), _out_outside_logs),
    "out-dir-logs-lookalike": (4, (_OUTDIR,), _out_lookalike),
    "out-dir-dotdot": (4, (_OUTDIR,), _out_dotdot),
    "out-dir-not-ignored": (4, (_OUTDIR,), _set("not_ignored", True)),
    "no-merge-t0226-a": (5, (r"T-0226-a",), _no_merge("Merge T-0226-a:")),
    "no-merge-t0223-b": (5, (r"T-0223-b",), _no_merge("Merge T-0223-b:")),
    "no-merge-t0213-a2a": (5, (r"T-0213-a2a",), _no_merge("Merge T-0213-a2a:")),
    "no-merge-t0225": (5, (r"T-0225",), _no_merge("Merge T-0225:")),
    "branch-merge-only": (5, (r"T-0226-a",), _branch_merge_only),
    "sha-uppercase": (6, (_SHA_BAD,), _sha_tf(lambda s: s.upper())),
    "sha-63-hex": (6, (_SHA_BAD,), _sha_tf(lambda s: s[:63])),
    "sha-not-hex": (6, (_SHA_BAD,), _sha_tf(lambda s: "g" * 64)),
    "manifest-missing": (7, (_NOT_FILE,), _manifest_missing),
    "manifest-is-dir": (7, (_NOT_FILE,), _manifest_is_dir),
    "manifest-sha-newline": (8, (_SHA_DIFF,), _set("tail", b"\n")),
    "dir-in-repo": (9, (_IN_REPO,), _set("location", "repo")),
    "dir-symlink-in-repo": (9, (_IN_REPO,), _set("location", "symlink-in-repo")),
    "dir-symlink-into-repo": (9, (_IN_REPO,), _set("location", "symlink-into-repo")),
    "dir-in-git-init": (9, (_IN_REPO,), _set("location", "gitinit")),
    "manifest-version-2": (10, (r"version",), _edit(lambda o, d: o.__setitem__("version", 2))),
    "manifest-no-pages": (10, (r"page",), _edit(lambda o, d: o.__setitem__("pages", []))),
    "manifest-duplicate-id": (10, (r"duplicat|repeat|twice|pg_u1",), _edit(_dup_id)),
    "manifest-missing-id": (10, (r"\bid\b",), _edit(_missing_id)),
    "manifest-bad-kind": (10, (r"kind",), _edit(_bad_kind)),
    "manifest-not-json": (
        10,
        (r"json|malformed|pars|decode",),
        _set("raw_override", b'{"version": 1, "pages": ['),
    ),
    "png-dotdot": (11, (r"pg_m2", _ESCAPE), _edit(_png_dotdot("pg_m2", "scan_m2.png"))),
    "png-absolute": (11, (r"pg_m2", _ESCAPE), _edit(_png_absolute)),
    "png-symlink-out": (11, (r"pg_m2", _ESCAPE), _edit(_png_symlink_out)),
    "png-missing": (12, (r"pg_c1", _MISSING), _post(_unlink("scan_c1.png"))),
    "png-byte-flipped": (12, (r"pg_c1", _SHA_DIFF), _post(_flip("scan_c1.png"))),
    "mistake-empty-keywords": (13, (r"pg_m2", _KEYWORDS), _edit(_kw("pg_m2", []))),
    "mistake-empty-alternative": (
        13,
        (r"pg_m1", _KEYWORDS),
        _edit(_kw("pg_m1", ["ligne 3", "a||b"])),
    ),
    "mistake-no-keywords": (13, (r"pg_m2", _KEYWORDS), _edit(_kw("pg_m2", KeyError))),
    "mistake-empty-string": (13, (r"pg_m2", _KEYWORDS), _edit(_kw("pg_m2", ["vitesse", ""]))),
    "mistake-not-a-string": (13, (r"pg_m2", _KEYWORDS), _edit(_kw("pg_m2", ["vitesse", 3]))),
    "correct-keywords-string": (14, (r"pg_c1", _LIST), _edit(_kw("pg_c1", "c1 note"))),
    "unclear-keywords-string": (14, (r"pg_u1", _LIST), _edit(_kw("pg_u1", "x"))),
    "instructions-refused": (
        15,
        (r"T0226-RIG-REFUSAL|validate_instructions|instructions",),
        _instructions_refused,
    ),
    "visual-tokens-not-1120": (16, (r"visual",), _visual_560),
    "ps-fails": (17, (r"P-S",), _ps_fails),
}


def _assert_refused(r: Rig, rc: int, err: str, case: str, ps: bool) -> None:
    _item, pats, _f = REFUSALS[case]
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    for pat in pats:
        assert re.search(pat, err, re.IGNORECASE | re.DOTALL), (case, pat, err[-2000:])
    assert r.sockets == [], (case, r.sockets)
    assert _AUDIT["sockets"] == [], (case, _AUDIT["sockets"])
    bad = _bad_writes(r.writes(), [], ps=ps)
    assert bad == [], f"{case}: a refusal wrote {bad[:5]}"
    for root in _ps_roots(r.writes()):
        assert not root.exists(), f"{case}: the P-S temporary directory {root} was not removed"


@pytest.mark.parametrize(
    "case", list(REFUSALS), ids=[f"a{v[0]:02d}-{k}" for k, v in REFUSALS.items()]
)
def test_refusal(rig: Rig, capsys: pytest.CaptureFixture[str], case: str) -> None:
    """A2, one row per item (and its adversarial variants): exit 2 with the item's own message, before any socket
    (doubles and the audit hook), and no file written anywhere (P-S's temporary directory excepted for P-S's row,
    and removed). Adversarial: ``localhost.example`` is not loopback; an untracked file is dirty; ``logs_...``,
    ``logs/../x`` and an un-ignored path are not git-ignored dirs under ``logs/``; a branch merge or ``T-0226-ab``
    does not count; one appended newline changes the manifest's sha; the manifest dir inside the repo, through a
    link inside the repo, through a link outside pointing in, or inside a fresh ``git init``; ``../x.png`` with the
    right bytes, an absolute path inside the dir, a PNG symlinked outside; a missing PNG and one flipped byte; a
    mistake page with ``[]`` or ``"a||b"``."""
    item, _pats, apply = REFUSALS[case]
    apply(rig)
    rc = rig.main()
    err = capsys.readouterr().err
    _assert_refused(rig, rc, err, case, ps=item == 17)
    if case == "out-dir-exists":
        assert sorted(p.name for p in rig.out.iterdir()) == ["keep.txt"]


#: adjacent A2 items, each pair applied together: the earlier item's message must be the one printed
ORDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("url-not-loopback", "dirty-untracked"),
    ("dirty-untracked", "out-dir-exists"),
    ("out-dir-exists", "out-dir-outside-logs"),
    ("out-dir-outside-logs", "no-merge-t0226-a"),
    ("no-merge-t0226-a", "sha-uppercase"),
    ("sha-uppercase", "manifest-is-dir"),
    ("manifest-sha-newline", "dir-in-repo"),
    ("dir-in-repo", "manifest-duplicate-id"),
    ("manifest-duplicate-id", "png-dotdot"),
    ("png-dotdot", "png-missing"),
    ("png-missing@pg_m1", "mistake-empty-keywords"),
    ("mistake-empty-keywords", "correct-keywords-string"),
    ("correct-keywords-string", "instructions-refused"),
    ("instructions-refused", "visual-tokens-not-1120"),
    ("visual-tokens-not-1120", "ps-fails"),
)
_EXTRA: dict[str, tuple[int, tuple[str, ...], Callable[[Rig], None]]] = {
    "png-missing@pg_m1": (12, (r"pg_m1", _MISSING), _post(_unlink("scan_m1.png"))),
}


@pytest.mark.parametrize(("first", "second"), ORDER_PAIRS, ids=[f"{a}+{b}" for a, b in ORDER_PAIRS])
def test_refusal_order(
    rig: Rig, capsys: pytest.CaptureFixture[str], first: str, second: str
) -> None:
    """A2 "in this order": with two adjacent items' faults at once, the earlier item refuses (its patterns are in
    stderr). The later fault is applied first, so a fault that moves the out-dir is in place before one that
    creates it."""
    table = {**REFUSALS, **_EXTRA}
    table[second][2](rig)
    table[first][2](rig)
    rc = rig.main()
    err = capsys.readouterr().err
    assert rc == 2, f"{first}+{second}: exit {rc}\n{err[-2000:]}"
    for pat in table[first][1]:
        assert re.search(pat, err, re.IGNORECASE | re.DOTALL), (first, second, pat, err[-2000:])
    assert rig.sockets == [] and _AUDIT["sockets"] == []


NOT_REFUSED: dict[str, Callable[[Rig], None]] = {
    "at-head": lambda r: None,
    "correct-empty-keywords": _edit(_kw("pg_c1", [])),
    "unclear-empty-keywords": _edit(_kw("pg_u1", [])),
    "mistake-alternatives": _edit(_kw("pg_m1", ["ligne 3|line 3|l3", "x = 4"])),
    # R-4 (c): a ".." that stays inside the manifest dir is accepted (the escape is what A2 refuses)
    "png-dotdot-inside": _edit(lambda o, d: _dotdot_inside(o, d)),
}


def _dotdot_inside(obj: dict[str, Any], d: Path) -> None:
    (d / "sub").mkdir(exist_ok=True)
    _entry(obj, "pg_m1")["png"] = "sub/../scan_m1.png"


@pytest.mark.parametrize("case", list(NOT_REFUSED))
def test_not_refused_goes_to_the_engine(
    rig: Rig, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A2, the too-wide direction: with every merge, a well-formed manifest whose pins hold (a correct or unclear
    page with ``[]`` keywords is valid), 1120 in force and P-S passing, ``main`` refuses nothing; it asks for the
    engine and, finding none, exits 1. Nothing is written outside the out-dir and P-S's removed temporary
    directory, and nothing under the manifest directory."""
    NOT_REFUSED[case](rig)
    rc = rig.main()
    err = capsys.readouterr().err
    assert rc == 1, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"
    assert rig.manifest_dir is not None
    bad = _bad_writes(rig.writes(), [rig.out], ps=True)
    assert bad == [], bad[:5]
    assert not any(_under(p, rig.manifest_dir) for _e, p in rig.writes())


def test_manifest_sha256_is_required(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    """A1: ``--manifest-sha256`` has no default: without it the CLI refuses (argparse's exit 2) before any socket."""
    argv = rig.argv()
    i = argv.index("--manifest-sha256")
    del argv[i : i + 2]
    try:
        rc = rig.main(argv)
    except SystemExit as exc:
        rc = int(exc.code or 0)
    capsys.readouterr()
    assert rc == 2, rc
    assert rig.sockets == []


def test_default_out_dir() -> None:
    """A1: the default out-dir is ``REPO/logs/t0226-real-attempts-<sha7>``."""
    got = Path(api("default_out")(GIT_SHA))
    assert got == REPO / "logs" / f"t0226-real-attempts-{GIT_SHA[:7]}", got


# =====================================================================================================
# A3: texts, the private instance, frames, order, budget
# =====================================================================================================


def _builtin_tutor() -> str:
    """The ``education_tutor`` entry's ``text`` in ``hud/src/settings/prompts.js``: its ``+``-joined string
    literals, each decoded as a JS (JSON-compatible) double-quoted string."""
    src = PROMPTS_JS.read_text(encoding="utf-8")
    at = src.index('id: "education_tutor"')
    m = re.compile(r'text:\s*((?:"(?:[^"\\]|\\.)*"\s*\+\s*)*"(?:[^"\\]|\\.)*")').search(src, at)
    assert m is not None, "no text: literal after the education_tutor id"
    assert src.rfind("id:", at + 1, m.start()) == -1, "the text: literal belongs to another preset"
    parts = re.findall(r'"(?:[^"\\]|\\.)*"', m.group(1))
    return "".join(json.loads(p) for p in parts)


def test_tutor_text_is_the_builtin() -> None:
    """A3: ``TUTOR_INSTRUCTIONS`` is byte-equal to the T-0225 "Education tutor" built-in, and the product's
    validator accepts it."""
    from local_astra.conductor import prompt_builder

    builtin = _builtin_tutor()
    # T-0229 A1/A2: the prefix follows the new text; T-0225's "find the line with the mistake" is gone
    assert builtin.startswith(
        "You are a patient tutor. When I show you my work, check it line by line."
    ), builtin[:80]
    assert "find the line with the mistake" not in builtin, builtin
    assert builtin == TUTOR, "T-0229 A1: the built-in is the task's text, verbatim"
    got = api("TUTOR_INSTRUCTIONS")
    assert got.encode("utf-8") == builtin.encode("utf-8"), (got, builtin)
    assert prompt_builder.validate_instructions(got) is None
    assert prompt_builder.USER_INSTRUCTIONS_HEADER == HEADER, "premise: T-0213's header"


def test_question_is_neutral() -> None:
    """A3, T-0229 A2: the typed question is "Can you double-check my work?"; it is neither T-0223's, which
    presupposes a mistake, nor T-0226 B's, which names one."""
    q = api("QUESTION")
    assert q == QUESTION, q
    assert q != T0223_QUESTION and "went wrong" not in q
    assert q != T0226_QUESTION and "mistake" not in q.casefold(), q


def test_constants() -> None:
    """A3: order seed 0, 90 s per turn, a 20 min budget, 3 consecutive non-clean turns stop the run, 1120."""
    assert api("ORDER_SEED") == 0
    assert api("TURN_TIMEOUT_S") == 90
    assert api("RUN_BUDGET_S") == 20 * 60
    assert api("ERRORED_STOP") == 3
    assert api("VISUAL_TOKENS") == 1120


def _load_t0223_own_name() -> ModuleType:
    name = "t0223_handwritten_check"
    spec = importlib.util.spec_from_file_location(name, T0223_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_private_t0223_instance_is_isolated(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    """A10 and the implementation notes: after the T-0226 harness is imported and has run (refusals passed, no
    engine), T-0223 loaded by path under its own name keeps T-0223's ``TUTOR_INSTRUCTIONS``, ``QUESTION`` and
    ``page_path``; the private instance is another module object, under another name, whose ``settings_message()``
    and ``instructions_block()`` carry the tutor built-in and whose ``page_path(id)`` is the manifest's PNG."""
    from local_astra.contracts.protocol import AstraSettingsSet

    harness()
    assert "t0223_handwritten_check" not in sys.modules, (
        "T-0226 registered T-0223 under its own name"
    )
    rc = rig.main()
    capsys.readouterr()
    assert rc == 1, rc
    own = _load_t0223_own_name()
    assert own.TUTOR_INSTRUCTIONS == T0223_TUTOR
    assert own.QUESTION == T0223_QUESTION
    assert Path(own.page_path("M1")) == REPO / "demos" / "handwritten" / "out" / "M1.png"
    inst = private()
    assert inst is not own and inst.__name__ != "t0223_handwritten_check", inst.__name__
    builtin = _builtin_tutor()
    assert builtin == inst.TUTOR_INSTRUCTIONS
    assert inst.QUESTION == QUESTION
    msg = inst.settings_message()
    assert isinstance(msg, AstraSettingsSet) and msg.rev == 0, msg
    assert msg.changes == {"user_instructions": builtin}, msg.changes
    assert inst.instructions_block() == HEADER + "\n" + builtin
    assert rig.manifest_dir is not None
    for p in DEFAULT_PAGES:
        got = Path(inst.page_path(p["id"]))
        assert got.resolve() == (rig.manifest_dir / p["png"]).resolve(), (p["id"], got)


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_frames_for_scan_sizes(tmp_path: Path, size: tuple[int, int]) -> None:
    """A3: each scan size framed through T-0223's ``frame_jpeg``: a decodable JPEG at
    ``policy.target_size(bucket_for(SCREEN, w, h))`` (875x1241 -> 874x1240 by hand; the 1755x1240 example in the
    task says 1566x1106 where the policy gives 1568x1106: reading 2), even dimensions, a 16-hex phash."""
    from PIL import Image

    from local_astra.contracts.perception import Source
    from local_astra.perception import policy

    png = tmp_path / "page.png"
    png.write_bytes(_png(size, "frame"))
    jpeg, w, h, ph = fn("frame_jpeg")(png)
    want = policy.target_size(size[0], size[1], policy.bucket_for(Source.SCREEN, size[0], size[1]))
    if size == (875, 1241):
        assert want == (874, 1240), "premise: the hand example"
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG"
        assert im.size == want == (w, h), (im.size, want, (w, h))
    assert w % 2 == 0 and h % 2 == 0, (w, h)
    assert re.fullmatch(r"[0-9a-f]{16}", ph), ph


def test_ps_runs_on_a_synthetic_white_page(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A2's last item, privacy: P-S frames a synthetic white page at the first ordered page's size, never a page of
    the manifest; that page lives outside the repo and the manifest dir and is gone after the run."""
    from PIL import Image, ImageStat

    calls: list[tuple[Path, tuple[int, int], float]] = []
    real = private().frame_jpeg

    def spy(png_path: Any) -> Any:
        p = Path(png_path).resolve()
        with Image.open(p) as im:
            im.load()
            mean = float(ImageStat.Stat(im.convert("L")).mean[0])
            calls.append((p, im.size, mean))
        return real(png_path)

    assert _patch_object(monkeypatch, rig.h, real, spy) >= 1
    rc = rig.main()
    capsys.readouterr()
    assert rc == 1, rc
    assert rig.manifest_dir is not None
    synthetic = [c for c in calls if not _under(c[0], rig.manifest_dir)]
    assert synthetic, f"P-S framed no synthetic page (calls: {[str(c[0].name) for c in calls]})"
    first = _order_oracle(PAGE_IDS)[0]
    for p, size, mean in synthetic:
        assert size == SIZE_OF[first], (size, first, SIZE_OF[first])
        assert mean > 250.0, f"the P-S page is not white (mean {mean:.1f})"
        assert not _under(p, REPO), p
        assert not p.exists(), f"the P-S page {p} outlived the run"


def test_order_seed0() -> None:
    """A3: sorted ids shuffled once by ``Random(0)``: deterministic, independent of the manifest's page order, a
    permutation of all ids (n = 1 each), and the hand-computed order for a fixed 6-id manifest."""
    page_order = api("page_order")
    ids = ["zq-m1", "zq-m2", "zq-c1", "zq-c2", "zq-u1", "zq-u2"]
    want = [
        "zq-u1",
        "zq-m1",
        "zq-c2",
        "zq-c1",
        "zq-u2",
        "zq-m2",
    ]  # random.Random(0).shuffle(sorted(ids)), by hand
    got = list(page_order(ids))
    assert got == want, got
    assert list(page_order(list(reversed(ids)))) == want
    assert list(page_order(["zq-c2", "zq-u2", "zq-m1", "zq-u1", "zq-c1", "zq-m2"])) == want
    assert sorted(got) == sorted(ids) and len(set(got)) == len(ids)


@pytest.mark.parametrize(
    ("elapsed", "stop"),
    [
        (0.0, False),
        (1000.0, False),
        (1065.0, False),
        (1065.5, True),
        (1100.0, True),
        (1200.0, True),
    ],
)
def test_budget_gate(elapsed: float, stop: bool) -> None:
    """A3: no turn starts unless 90 + 45 s still fit in 1200 s: 1065 s is the last start (both directions: a gate
    without the 45 s margin lets 1100 start; one that is too wide stops 1065)."""
    assert bool(api("budget_stop")(elapsed)) is stop, (elapsed, stop)


# =====================================================================================================
# A5: the judge
# =====================================================================================================

KW_M = ["ligne 3|line 3", "x = 4"]
KW_ALT = ["a-word|b-word"]

#: case -> (kind, status, spoken, shown, keywords, expected subset of judge_turn's result)
JUDGE: dict[str, tuple[str, str, list[str], str, list[str], dict[str, Any]]] = {
    "mistake-all-hit": (
        "mistake",
        "ran",
        ["Line 3 is where it slips: it should be x = 4."],
        "",
        KW_M,
        {"verdict": "located", "spoken_only_verdict": "located"},
    ),
    "mistake-one-missing": (
        "mistake",
        "ran",
        ["Line 3 is where it slips."],
        "",
        KW_M,
        {"verdict": "missed", "said_correct": False},
    ),
    "mistake-alternative": (
        "mistake",
        "ran",
        ["Look at the b-word here."],
        "",
        KW_ALT,
        {"verdict": "located"},
    ),
    "mistake-french-alternative": (
        "mistake",
        "ran",
        ["À la ligne 3, on doit avoir x = 4."],
        "",
        KW_M,
        {"verdict": "located"},
    ),
    "mistake-card-only": (
        "mistake",
        "ran",
        ["Have a look at the card."],
        "Line 3: x = 4",
        KW_M,
        {"verdict": "located", "spoken_only_verdict": "missed"},
    ),
    "mistake-split-channels": (
        "mistake",
        "ran",
        ["Line 3 slips."],
        "x = 4",
        KW_M,
        {"verdict": "located", "spoken_only_verdict": "missed"},
    ),
    "mistake-says-correct": (
        "mistake",
        "ran",
        ["Everything is correct, well done."],
        "",
        KW_M,
        {"verdict": "missed", "said_correct": True, "says_correct": True, "claims_mistake": False},
    ),
    "correct-is-correct": (
        "correct",
        "ran",
        ["Your work is correct."],
        "",
        [],
        {"verdict": "said_correct", "says_correct": True, "claims_mistake": False},
    ),
    "correct-claims": (
        "correct",
        "ran",
        ["There is a mistake in line 2."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "correct-dont-see-any": (
        "correct",
        "ran",
        ["I don't see any mistake."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "correct-nothing-is-wrong": (
        "correct",
        "ran",
        ["Nothing is wrong here."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "correct-but-split": (
        "correct",
        "ran",
        ["Line 1 is correct, but line 3 is wrong."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "correct-negated-claim": (
        "correct",
        "ran",
        ["This is not correct."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "correct-french-correct": (
        "correct",
        "ran",
        ["Aucune erreur, c'est juste."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "correct-french-claim": (
        "correct",
        "ran",
        ["Il y a une erreur à la ligne 2."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True},
    ),
    "correct-guiding-question": (
        "correct",
        "ran",
        ["What do you get when you multiply both sides by two?"],
        "",
        [],
        {"verdict": "no_verdict", "claims_mistake": False, "says_correct": False},
    ),
    "correct-curly-apostrophe": (
        "correct",
        "ran",
        ["I don’t see any mistake in your work."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "correct-card-claims": (
        "correct",
        "ran",
        ["Have a look at the card."],
        "Line 2 is wrong",
        [],
        {"verdict": "claimed_mistake", "spoken_only_verdict": "no_verdict"},
    ),
    "unclear-claims": (
        "unclear",
        "ran",
        ["Line 2 is wrong."],
        "",
        [],
        {"verdict": "no_verdict", "claims_mistake": True, "says_correct": False},
    ),
    "unclear-says-correct": (
        "unclear",
        "ran",
        ["It looks right to me."],
        "",
        [],
        {"verdict": "no_verdict", "claims_mistake": False, "says_correct": True},
    ),
    "errored-mistake": (
        "mistake",
        "errored",
        ["Line 3 is where it slips: it should be x = 4."],
        "",
        KW_M,
        {"verdict": "errored"},
    ),
    "timed-out-correct": (
        "correct",
        "timed_out",
        ["Your work is correct."],
        "",
        [],
        {"verdict": "errored"},
    ),
    "not-run-correct": ("correct", "not_run", [], "", [], {"verdict": "errored"}),
    "empty-mistake": ("mistake", "ran", [], "", KW_M, {"verdict": "errored"}),
    "empty-correct": ("correct", "ran", [], "", [], {"verdict": "errored"}),
    "errored-unclear": ("unclear", "errored", ["Line 2 is wrong."], "", [], {"verdict": "errored"}),
    # --- R-4 (a): the split also on ",", "however", "mais", "cependant"; a clause ending in "?" is neither ------
    "r4-comma-split": (
        "correct",
        "ran",
        ["Your work is correct, line 3 is wrong."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "r4-however-split": (
        "correct",
        "ran",
        ["The setup is correct however you forgot the units."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "r4-mais-split": (
        "correct",
        "ran",
        ["C'est juste mais tu as oublié le signe."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "r4-cependant-split": (
        "correct",
        "ran",
        ["Tout est juste cependant tu as oublié l'unité."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "r4-question-with-claim-word": (
        "correct",
        "ran",
        ["Is there a mistake in line 2?"],
        "",
        [],
        {"verdict": "no_verdict", "claims_mistake": False, "says_correct": False},
    ),
    "r4-question-then-claim": (  # too wide: only the question clause is neutral
        "correct",
        "ran",
        ["Is line 2 right? Line 3 is wrong."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True},
    ),
    "r4-question-then-correct": (
        "correct",
        "ran",
        ["Is there an error? No, everything is correct."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False, "says_correct": True},
    ),
    "r4-couldnt-find-a": (
        "correct",
        "ran",
        ["I couldn't find a single error."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "r4-didnt-find-a": (
        "correct",
        "ran",
        ["I didn't find a problem, nice."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "r4-cant-find-a": (
        "correct",
        "ran",
        ["I can't find a mistake anywhere."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    "r4-do-not-see-a": (
        "correct",
        "ran",
        ["I do not see a wrong step."],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False},
    ),
    # --- R-4 (c): NEGATED_CLAIM wins over NO_MISTAKE inside one clause -----------------------------------------
    "r4-negated-beats-no-mistake": (
        "correct",
        "ran",
        ["Well done on line 1 and line 2 is not right."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    "r4-negated-beats-no-mistake-fr": (
        "correct",
        "ran",
        ["Bravo pour l'effort et ce n'est pas juste."],
        "",
        [],
        {"verdict": "claimed_mistake", "claims_mistake": True, "says_correct": False},
    ),
    # --- R-4 (c): the five reviewer sentences -------------------------------------------------------------------
    "r4-reviewer-1": (
        "correct",
        "ran",
        ["Line 1 is correct, line 3 is wrong."],
        "",
        [],
        {"verdict": "claimed_mistake"},
    ),
    "r4-reviewer-2": (
        "correct",
        "ran",
        ["Are you sure the sign in line 3 is correct?"],
        "",
        [],
        {"verdict": "no_verdict", "claims_mistake": False, "says_correct": False},
    ),
    "r4-reviewer-3": (
        "correct",
        "ran",
        ["C'est correct jusqu'à la ligne 3, où il y a une erreur."],
        "",
        [],
        {"verdict": "claimed_mistake"},
    ),
    "r4-reviewer-4": (
        "correct",
        "ran",
        ["I don't see a mistake. Well done!"],
        "",
        [],
        {"verdict": "said_correct", "claims_mistake": False, "says_correct": True},
    ),
    "r4-reviewer-5": (
        "correct",
        "ran",
        ["Your first two steps are correct, however you forgot the minus sign."],
        "",
        [],
        {"verdict": "claimed_mistake"},
    ),
}
#: R-4 (a): the five phrases NO_MISTAKE gains
R4_NO_MISTAKE = ("don't see a", "do not see a", "didn't find a", "can't find a", "couldn't find a")


@pytest.mark.parametrize("case", list(JUDGE))
def test_judge_table(case: str) -> None:
    """A5, every row by hand: keywords per T-0223 R-2 over spoken + shown (a card-only hit locates, spoken-only
    misses); the clause rule (a negated claim is a claim; a no-mistake phrase makes its clause correct and hides its
    claim words, so "don't see any mistake" is correct; ``but`` splits clauses; curly apostrophes fold); French and
    English forms; a guiding question with no phrase gives no verdict; errored, timed-out, not-run and empty turns are
    errored; unclear pages give only the flags. R-4 (a): ``,``, ``however``, ``mais`` and ``cependant`` split too; a
    clause ending in ``?`` is neither correct nor a claim (and only that clause: a statement after it still counts);
    the five new no-mistake phrases. R-4 (c): a negated claim beats a no-mistake phrase in one clause; the
    reviewer's five sentences."""
    kind, st, spoken, shown, kws, want = JUDGE[case]
    got = api("judge_turn")(kind, list(spoken), shown, list(kws), status=st)
    for k, v in want.items():
        assert k in got, (case, k, dict(got))
        assert (bool(got[k]) if isinstance(v, bool) else got[k]) == v, (case, k, dict(got))


def test_judge_matched_and_thoughts_are_not_text() -> None:
    """A5: ``matched`` lists the entries the judged text satisfies, in the page's order; the judge takes only spoken
    and shown text (a thought is not an argument it could read)."""
    got = api("judge_turn")("mistake", ["it should be x = 4"], "", list(KW_M), status="ran")
    assert list(got["matched"]) == ["x = 4"], got
    assert got["verdict"] == "missed"


def test_phrase_lists_hold_the_task_examples() -> None:
    """A5: the three module constants hold every example phrase the task lists (reading 4)."""
    ev = _eval()
    for name, required in (
        ("NEGATED_CLAIM", ev.REQUIRED_NEGATED_CLAIM),
        ("NO_MISTAKE", ev.REQUIRED_NO_MISTAKE),
        ("CLAIM", ev.REQUIRED_CLAIM),
    ):
        got = list(api(name))
        assert all(isinstance(p, str) and p for p in got), name
        missing = [p for p in required if p not in got]
        assert missing == [], (name, missing)
    assert [p for p in R4_NO_MISTAKE if p not in list(api("NO_MISTAKE"))] == [], (
        "R-4 (a)'s five phrases"
    )
    assert set(R4_NO_MISTAKE) <= set(ev.REQUIRED_NO_MISTAKE), (
        "premise: the eval's minimum carries R-4 (a)"
    )


def test_eval_judge_oracle_agrees_with_the_table() -> None:
    """The eval's own judge (the oracle the run rows use), with the task's example lists, gives the hand table
    (green by design: no harness)."""
    ev = _eval()
    phrases = {
        "NEGATED_CLAIM": ev.REQUIRED_NEGATED_CLAIM,
        "NO_MISTAKE": ev.REQUIRED_NO_MISTAKE,
        "CLAIM": ev.REQUIRED_CLAIM,
    }
    for case, (kind, st, spoken, shown, kws, want) in JUDGE.items():
        got = ev.verdict_of(kind, st == "ran", spoken, shown, kws, phrases)
        for k, v in want.items():
            if k in got:
                assert got[k] == v, (case, k, got)


# =====================================================================================================
# A6: thinking
# =====================================================================================================


def _commit(seq: int, rid: str, idx: int, segment: str, text: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "t_ms": 10 * seq,
        "kind": "llm.commit",
        "turn_id": "t1",
        "data": {
            "request_id": rid,
            "canvas_idx": idx,
            "segment": segment,
            "text": text,
            "thought_open": False,
        },
    }


def test_thinking_measures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A6: thought characters and canvases count only ``segment == "thought"`` commits (two requests here; answer
    and ui commits excluded); a turn with no thought gives 0, never null. ``/tokenize`` is posted the thought text
    (no generation) and its count returned; a failing poster gives null."""
    measures = api("thought_measures")
    recs = [
        _commit(0, "r1", 0, "thought", "Line three."),
        _commit(1, "r1", 1, "thought", " Then x."),
        _commit(2, "r1", 2, "answer", "Line 3 is wrong, it should be x = 4."),
        _commit(3, "r2", 0, "thought", "Again."),
        _commit(4, "r2", 1, "ui", CS),
    ]
    got = measures(recs)
    assert (int(got["chars"]), int(got["canvases"])) == (11 + 8 + 6, 3), dict(got)
    none = measures([_commit(0, "r1", 0, "answer", "Your work is correct.")])
    assert none["chars"] == 0 and none["canvases"] == 0, dict(none)

    h = harness()
    posts: list[tuple[str, Any]] = []
    timeouts: list[Any] = []
    answer: dict[str, Any] = {"status": 200, "payload": {"count": 7, "tokens": list(range(7))}}

    def post(url: str, body: Any, *a: Any, **k: Any) -> tuple[int, Any]:
        posts.append((url, body))
        timeouts.append(k["timeout_s"] if "timeout_s" in k else (a[0] if a else None))
        return answer["status"], answer["payload"]

    assert _patch_name(monkeypatch, h, "_post_json", post) >= 1
    tokens = api("thought_tokens")
    assert tokens(VALID_URL, "Line three. Then x.") == 7
    assert posts and all(str(u).rstrip("/").endswith("/tokenize") for u, _ in posts), posts
    assert "Line three. Then x." in json.dumps(posts[-1][1]), posts[-1]
    # R-4 (b): the post is bounded at 10 s (T._post_json's default is 60 s), so the last turn fits the 21 min wrapper
    assert timeouts and all(t is not None and float(t) == 10.0 for t in timeouts), timeouts
    for status_, payload in ((0, None), (500, None), (200, {"error": "x"})):
        answer.update(status=status_, payload=payload)
        assert tokens(VALID_URL, "Line three.") is None, (status_, payload)


# =====================================================================================================
# synthetic turns and run dirs (the real record shapes)
# =====================================================================================================


def _ui(text: str) -> str:
    return (
        CS
        + "\n"
        + json.dumps(
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["t"]},
                        {"id": "t", "component": "Text", "text": text},
                    ],
                },
            }
        )
    )


def _jpeg(index: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 88), (10 * index % 250, 200, 100)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def write_turn(
    out: Path,
    pid: str,
    *,
    index: int,
    spoken: str | None,
    shown: str | None,
    thoughts: Sequence[str],
    latency: int,
    tutor: str,
    question: str = QUESTION,
    fault: str | None = None,
    thought_tokens: Any = "absent",
) -> dict[str, Any]:
    """One turn's session log and turn line (what the real ``Session`` writes) and the runner's raw trial.
    ``question``: the typed text. ``fault``: ``no-image-body`` (the generate body summary has no image), ``no-block`` (the system messages lack
    the tutor block)."""
    from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
    from local_astra.telemetry.session_log import redact

    block = HEADER + "\n" + tutor
    jpeg = _jpeg(index)
    b64 = base64.b64encode(jpeg).decode("ascii")
    url = "data:image/jpeg;base64," + b64
    digest = _sha(b64.encode("utf-8"))[:8]
    system = SYS_BASE if fault == "no-block" else SYS_BASE + "\n" + block
    name = f"turn-{pid}.jsonl"
    user_ms = 1000
    recs: list[dict[str, Any]] = []

    def add(kind: str, data: dict[str, Any], turn: Any = "t1") -> None:
        recs.append(
            {"seq": len(recs), "t_ms": 10 * len(recs), "kind": kind, "turn_id": turn, "data": data}
        )

    add("log.started", {"pid": 4242, "path": name}, None)
    add(
        "settings.change",
        {
            "rev": 1,
            "changes": {
                "user_instructions": {
                    "old": {"sha12": "", "chars": 0},
                    "new": {"chars": len(tutor)},
                }
            },
            "source": "ui",
            "turn_seq": 0,
            "applied_at": "immediate",
        },
        None,
    )
    add(
        "machine.step",
        {"event": {"type": "user.text", "t_ms": user_ms, "text": question}, "commands": []},
    )
    for phase, rid, with_text in (("prefill", "t1-prefill", False), ("generate", "t1-gen", True)):
        parts: list[Any] = [ImagePart(url=url, phash="0123456789abcdef")]
        if with_text:
            parts.append(TextPart(text=question))
        prompt = Prompt(
            messages=(
                Message(role="system", content=(TextPart(text=system),)),
                Message(role="user", content=tuple(parts)),
            ),
            prefix_len=2,
            thinking=True,
        )
        add(
            "llm.request",
            {
                "phase": phase,
                "request_id": rid,
                "attempt_id": f"{pid}-{rid}",
                "options": None,
                "xargs": None,
                "prompt": redact(prompt.model_dump(mode="json")),
            },
        )
    canvas = 0
    for t in thoughts:
        add(
            "llm.commit",
            {
                "request_id": "t1-gen",
                "canvas_idx": canvas,
                "segment": "thought",
                "text": t,
                "thought_open": False,
            },
        )
        canvas += 1
    if spoken is not None:
        add(
            "llm.commit",
            {
                "request_id": "t1-gen",
                "canvas_idx": canvas,
                "segment": "answer",
                "text": spoken,
                "thought_open": False,
            },
        )
        canvas += 1
    if shown is not None:
        add(
            "llm.commit",
            {
                "request_id": "t1-gen",
                "canvas_idx": canvas,
                "segment": "ui",
                "text": _ui(shown),
                "thought_open": False,
            },
        )
    for sentence in spoken.split("\n\n") if spoken is not None else []:
        add(
            "machine.step",
            {
                "event": {"type": "llm.commit"},
                "commands": [{"type": "tts.speak", "text": sentence, "source": sentence}],
            },
        )
    regime = {
        "profile": "lean",
        "canvas_length": 256,
        "max_denoising_steps": 48,
        "thinking": True,
        "tools": "off",
        "settings_rev": 1,
        "system_sha12": _sha(system.encode("utf-8"))[:12],
        "instructions_edited": True,
    }
    spans: dict[str, list[int]] = {}
    if spoken is not None:
        spans["tts.first_audio"] = [user_ms + latency, user_ms + latency + 300]
    line = {
        "turn_id": "t1",
        "session_id": f"s-{pid}",
        "spans": spans,
        "regime": regime,
        "input": "text",
        "outcome": "spoken" if spoken is not None else "silent",
        "session_log": str(out / "sessions" / name),
        "extras": {"speech_refused": 0},
    }
    add("turn.record", line)
    add("log.stopped", {"reason": "session closed", "records": len(recs) + 1, "dropped": 0}, None)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    (out / "sessions" / name).write_text(
        "".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8"
    )
    with (out / "turns.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    sha = _sha(jpeg)
    bodies = [
        {
            "label": f"prefill:{pid}-p",
            "phase": "prefill",
            "images": [sha],
            "max_soft_tokens": 1120,
            "tools": False,
        },
        {
            "label": f"generate:{pid}-g",
            "phase": "generate",
            "images": [] if fault == "no-image-body" else [sha],
            "max_soft_tokens": 1120,
            "tools": False,
        },
    ]
    trial: dict[str, Any] = {
        "page": pid,
        "repeat": 0,
        "session_log": name,
        "exception": None,
        "timed_out": False,
        "frame_sha256": sha,
        "stored_sha256": sha,
        "frame_digest": digest,
        "user_text_ms": user_ms,
        "settings_echo": {"rev": 1, "refused": []},
        "wall_s": 12.0 + index,
        "bodies": bodies,
        "http_non_200": 0,
        "engine_up_after": True,
        "vram_min_free_mib": 4000 - index,
    }
    if fault == "exception":
        trial["exception"] = "RuntimeError('T-0226 rig: the turn raised')"
    if fault == "timed-out":
        trial["timed_out"] = True
    if thought_tokens != "absent":
        trial["thought_tokens"] = thought_tokens
    return trial


#: the 6-page synthetic run: id -> (kind, keywords, spoken, shown, thoughts, latency ms)
SIX: dict[str, tuple[str, list[str], str | None, str | None, list[str], int]] = {
    "zq-m1": (
        "mistake",
        ["ligne 3|line 3", "x = 4"],
        "The slip is on line 3.",
        "x = 4 there",
        ["Line three.", " Then x."],
        900,
    ),
    "zq-m2": (
        "mistake",
        ["vitesse|speed"],
        "Everything is correct, well done.",
        None,
        ["the speed is off"],
        2000,
    ),
    "zq-c1": ("correct", [], "Your work is correct.", None, ["All good, I see."], 1200),
    "zq-c2": ("correct", ["zz"], "There is a mistake in line 2.", None, ["Checking line 2."], 1200),
    "zq-u1": ("unclear", [], "Line 2 is wrong.", None, ["Hmm."], 700),
    "zq-u2": ("unclear", [], "Have a look at the card.", "Tout est juste.", [], 1500),
}
SIX_ORDER = ["zq-u1", "zq-m1", "zq-c2", "zq-c1", "zq-u2", "zq-m2"]
#: by hand from SIX
SIX_VERDICT = {
    "zq-m1": ("located", "missed"),
    "zq-m2": ("missed", "missed"),
    "zq-c1": ("said_correct", "said_correct"),
    "zq-c2": ("claimed_mistake", "claimed_mistake"),
    "zq-u1": ("no_verdict", "no_verdict"),
    "zq-u2": ("no_verdict", "no_verdict"),
}
SIX_FLAGS = {"zq-u1": (True, False), "zq-u2": (False, True)}
SIX_THOUGHT = {
    "zq-m1": (19, 2),
    "zq-m2": (16, 1),
    "zq-c1": (16, 1),
    "zq-c2": (16, 1),
    "zq-u1": (4, 1),
    "zq-u2": (0, 0),
}
SIX_READING = "1 of 2 mistake pages located; 1 of 2 correct pages said correct, 1 claimed a mistake; 2 unclear pages reported"


def _six_pages() -> list[dict[str, Any]]:
    return [
        {
            "id": pid,
            "png": f"scan_{pid}.png",
            "size": SIZES[i % 4],
            "kind": v[0],
            "key": f"invented key {pid}",
            "judge_keywords": v[1],
        }
        for i, (pid, v) in enumerate(SIX.items())
    ]


def build_six(
    tmp_path: Path,
    *,
    drop: str | None = None,
    fault: tuple[str, str] | None = None,
    p0: bool = True,
    texts: tuple[str, str] | None = None,
    turn_texts: tuple[str, str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """The 6-page run dir written with the harness's writers (``G._write_json``, ``G._append_jsonl`` through the
    private T-0223 instance) and ``write_report``; returns (run dir, manifest path, report). ``texts``: the
    ``(user_instructions, question)`` the run's regime records (default: the harness's ``TUTOR_INSTRUCTIONS`` and
    ``QUESTION``, as its ``main`` records them); ``turn_texts``: what the turns carry (default: ``texts``). When
    either is given, ``write_report`` runs with the harness and its private instance holding ``turn_texts``, so the
    report is the one a harness of that text writes and only the eval can see a record that disagrees."""
    inst = private()
    tutor, question = texts if texts is not None else (api("TUTOR_INSTRUCTIONS"), api("QUESTION"))
    turn_tutor, turn_question = turn_texts if turn_texts is not None else (tutor, question)
    man, man_sha = write_manifest_dir(tmp_path / "scans6", _six_pages())
    out = tmp_path / "run"
    (out / "sessions").mkdir(parents=True)
    (out / "turns.jsonl").touch()
    (out / "trials.jsonl").touch()
    inst.G._write_json(out / "order.json", list(api("page_order")(list(SIX))))
    inst.G._write_json(
        out / "manifest.json",
        {
            "manifest_sha256": man_sha,
            "manifest_dir_name": man.parent.name,
            "regime": {
                "profile": "lean",
                "canvas_length": 256,
                "max_denoising_steps": 48,
                "thinking": True,
                "visual_tokens": 1120,
                "tools": "off",
                "search": "off",
                "order_seed": 0,
                "n": 1,
                "engine_seed": None,
                "manifest_sha256": man_sha,
                "user_instructions": tutor,
                "question": question,
            },
            "stopped": None,
        },
    )
    inst.G._write_json(
        out / "probes.json",
        {
            "P-S": {"passed": True},
            "P0": {
                "passed": p0,
                "image_tokens": {
                    pid: {
                        "size": list(SIZES[i % 4]),
                        "image_tokens": 1000 + i,
                        "expected_soft_tokens": 1100,
                    }
                    for i, pid in enumerate(SIX)
                },
            },
        },
    )
    for index, pid in enumerate(SIX_ORDER):
        if pid == drop:
            continue
        _kind, _kws, spoken, shown, thoughts, lat = SIX[pid]
        raw = write_turn(
            out,
            pid,
            index=index,
            spoken=spoken,
            shown=shown,
            thoughts=thoughts,
            latency=lat,
            tutor=turn_tutor,
            question=turn_question,
            fault=fault[1] if fault and fault[0] == pid else None,
            thought_tokens=40 + index,
        )
        inst.G._append_jsonl(out / "trials.jsonl", [raw])
    with pytest.MonkeyPatch.context() as mp:
        if texts is not None or turn_texts is not None:
            # the harness as it was when it wrote this run's report: holding the texts its turns carry (for the
            # 2789ec90 dir, T-0225's text and T-0226's question), so its own A4 check passes on them
            for module in (harness(), inst):
                mp.setattr(module, "TUTOR_INSTRUCTIONS", turn_tutor)
                mp.setattr(module, "QUESTION", turn_question)
        api("write_report")(out, man)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return out, man, report


def _rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {str(p.get("id")): p for p in report.get("pages") or []}
    return rows


def _keys(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [x for v in obj.values() for x in _keys(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _keys(v)]
    return []


def test_report_over_synthetic_run_dir(tmp_path: Path) -> None:
    """A5/A6/A7 over a clean 6-page dir (2 per kind), every number by hand (``SIX``): verdicts and spoken-only
    verdicts (a keyword on the card locates, spoken-only misses; keywords only in the thought miss and "Everything
    is correct, well done." sub-flags ``said_correct``); counts; the reading line in report.json and report.md;
    thinking chars and canvases (0 for the turn with no thought) and tokens from the trial; first-word p50/max
    (the 3rd and 4th values are equal, so p50 is unambiguous) and the cold first turn; frame size and image tokens
    from P0's table; the hand-audit section with one line per page; the phrase lists verbatim; the notes; label
    measured; no bar or threshold key."""
    out, _man, report = build_six(tmp_path)
    assert report.get("label") == "measured", (report.get("label"), report.get("violations"))
    rows = _rows(report)
    assert sorted(rows) == sorted(SIX), sorted(rows)
    for pid, (v, sv) in SIX_VERDICT.items():
        assert (rows[pid].get("verdict"), rows[pid].get("spoken_only_verdict")) == (v, sv), (
            pid,
            rows[pid],
        )
        assert rows[pid].get("kind") == SIX[pid][0]
    assert bool(rows["zq-m2"].get("said_correct")) is True
    assert sorted(rows["zq-m1"].get("matched") or []) == sorted(SIX["zq-m1"][1])
    for pid, (claims, says) in SIX_FLAGS.items():
        assert (bool(rows[pid].get("claims_mistake")), bool(rows[pid].get("says_correct"))) == (
            claims,
            says,
        ), pid
    counts = report.get("counts") or {}
    assert {
        k: int(v)
        for k, v in (counts.get("mistake") or {}).items()
        if k in ("n", "located", "missed", "missed_said_correct", "errored")
    } == {"n": 2, "located": 1, "missed": 1, "missed_said_correct": 1, "errored": 0}, counts.get(
        "mistake"
    )
    assert {
        k: int(v)
        for k, v in (counts.get("correct") or {}).items()
        if k in ("n", "said_correct", "claimed_mistake", "no_verdict", "errored")
    } == {"n": 2, "said_correct": 1, "claimed_mistake": 1, "no_verdict": 0, "errored": 0}, (
        counts.get("correct")
    )
    assert int((counts.get("unclear") or {}).get("n", -1)) == 2
    assert report.get("reading") == SIX_READING, report.get("reading")
    md = (out / "report.md").read_text(encoding="utf-8")
    assert SIX_READING in md
    for pid, (chars, canvases) in SIX_THOUGHT.items():
        assert (rows[pid].get("thought_chars"), rows[pid].get("thought_canvases")) == (
            chars,
            canvases,
        ), (pid, rows[pid])
        assert rows[pid].get("thought_tokens") == 40 + SIX_ORDER.index(pid), (pid, rows[pid])
        assert rows[pid].get("first_word_ms") == SIX[pid][5], (pid, rows[pid])
        i = list(SIX).index(pid)
        assert list(rows[pid].get("frame_size") or []) == list(SIZES[i % 4]), (
            pid,
            rows[pid].get("frame_size"),
        )
        assert rows[pid].get("image_tokens") == 1000 + i, (pid, rows[pid].get("image_tokens"))
        spoken = SIX[pid][2]
        assert spoken is not None and spoken in md, (pid, spoken)
    lat = report.get("latency") or {}
    assert (lat.get("p50_ms"), lat.get("max_ms"), lat.get("n"), lat.get("cold_first_turn_ms")) == (
        1200,
        2000,
        6,
        700,
    ), lat
    th = report.get("thinking") or {}
    assert (th.get("p50_chars"), th.get("max_chars")) == (16, 19), th
    heading = "## Hand audit (main session)"
    assert heading in md
    section = md.split(heading, 1)[1].split("\n## ", 1)[0]
    for pid in SIX:
        hits = [
            ln
            for ln in section.splitlines()
            if re.search(rf"(?<![\w-]){re.escape(pid)}(?![\w-])", ln)
        ]
        assert len(hits) == 1, (pid, hits)
    phrases = report.get("phrases") or {}
    for name in ("NEGATED_CLAIM", "NO_MISTAKE", "CLAIM"):
        assert list(phrases.get(name) or []) == list(api(name)), name
        for p in api(name):
            assert p in md, (name, p)
    assert private().FRAME_NOTE in md, "FRAME_NOTE as T-0223"
    # T-0229 A2: PRESUPPOSE_NOTE is replaced by a note naming the instructions and question in force
    assert not re.search(r"presuppos", md, re.IGNORECASE), (
        "T-0229: no presupposition note in report.md"
    )
    notes = report.get("notes") or {}
    note_texts = [str(v) for v in (notes.values() if isinstance(notes, dict) else notes)]
    assert not [n for n in note_texts if re.search(r"presuppos", n, re.IGNORECASE)], note_texts
    in_force = [n for n in note_texts if QUESTION in n and (TUTOR in n or "Education tutor" in n)]
    assert in_force, ("T-0229: a note naming the Education tutor text and the question", note_texts)
    assert all(n in md for n in in_force), "the in-force note is in report.md"
    assert re.search(r"teacher.{0,80}red|red.{0,80}teacher", md, re.IGNORECASE | re.DOTALL), (
        "the red-notes note"
    )
    bad = [k for k in _keys(report) if re.search(r"\bbar\b|threshold|adopt", k, re.IGNORECASE)]
    assert bad == [], bad
    assert not re.search(r"\b(PASS|FAIL)\b", md)
    assert not re.search(r"\b(pass|fail)", str(report.get("reading")), re.IGNORECASE)


VARIANTS: dict[str, tuple[dict[str, Any], str]] = {
    "dropped-turn": ({"drop": "zq-c1"}, "incomplete"),
    "p0-fails": ({"p0": False}, "void"),
    "body-without-image": ({"fault": ("zq-m1", "no-image-body")}, "void"),
    "system-without-tutor-block": ({"fault": ("zq-c1", "no-block")}, "void"),
    # R-4 (c): an errored mistake page (its log and keyword-bearing text are there) counts as errored, not located
    "errored-mistake-page": ({"fault": ("zq-m1", "exception")}, "incomplete"),
    # R-4 (c): a timed-out turn makes the run incomplete (not void) and its page errored
    "timed-out-turn": ({"fault": ("zq-c2", "timed-out")}, "incomplete"),
}
#: faults that make a turn non-clean without an A4 violation
_NONCLEAN = ("exception", "timed-out")


@pytest.mark.parametrize("case", list(VARIANTS))
def test_report_variants(tmp_path: Path, case: str) -> None:
    """A8 over the 6-page dir: a dropped turn is incomplete (its page errored, counted in its kind's denominator, the
    reading moves); P0 failing is void; a generate body without the image is void; a system message without the
    tutor block is void (the violation names its page). The eval's label row agrees."""
    kw, label = VARIANTS[case]
    out, man, report = build_six(tmp_path, **kw)
    assert report.get("label") == label, (case, report.get("label"), report.get("violations"))
    rows = _rows(report)
    if case == "dropped-turn":
        assert rows["zq-c1"].get("verdict") == "errored", rows.get("zq-c1")
        correct = report["counts"]["correct"]
        assert (int(correct["n"]), int(correct["errored"]), int(correct["said_correct"])) == (
            2,
            1,
            0,
        ), correct
        assert report.get("reading", "").startswith(
            "1 of 2 mistake pages located; 0 of 2 correct pages said correct"
        )
    if case == "errored-mistake-page":
        assert rows["zq-m1"].get("verdict") == "errored", rows.get("zq-m1")
        mistake = report["counts"]["mistake"]
        assert (
            int(mistake["n"]),
            int(mistake["errored"]),
            int(mistake["located"]),
            int(mistake["missed"]),
        ) == (2, 1, 0, 1), mistake
        assert report.get("reading", "").startswith("0 of 2 mistake pages located;"), report.get(
            "reading"
        )
        assert report.get("violations") in ([], None), report.get("violations")
    if case == "timed-out-turn":
        assert rows["zq-c2"].get("verdict") == "errored", rows.get("zq-c2")
        correct = report["counts"]["correct"]
        assert (int(correct["errored"]), int(correct["claimed_mistake"])) == (1, 0), correct
        assert report.get("violations") in ([], None), report.get("violations")
    if "fault" in kw and kw["fault"][1] not in _NONCLEAN:
        pid = kw["fault"][0]
        v = [
            str(x)
            for x in list(report.get("violations") or [])
            + list(report.get("body_violations") or [])
        ]
        assert v and all(pid in x for x in v), v
    ev = _eval()
    ev.test_label_re_derived(ev.load_run(out, man))


def test_every_eval_row_passes_on_a_harness_built_run_dir(tmp_path: Path) -> None:
    """A8 (binding lesson): every run row of the eval, which imports no harness, passes on the clean dir the
    harness's writers built."""
    out, man, _report = build_six(tmp_path)
    ev = _eval()
    run = ev.load_run(out, man)
    for row in ev.RUN_ROWS:
        row(run)


def test_eval_rows_pass_on_a_2789ec90_run_dir(tmp_path: Path) -> None:
    """T-0229 A3: the eval still re-derives the T-0226 B run (2789ec90): a run dir whose regime records T-0225's
    tutor text and T-0226's question, and whose turns carry them, passes every run row."""
    out, man, _report = build_six(tmp_path, texts=(T0226_TUTOR, T0226_QUESTION))
    ev = _eval()
    run = ev.load_run(out, man)
    assert ev.texts_of(run) == (T0226_TUTOR, T0226_QUESTION)
    for row in ev.RUN_ROWS:
        row(run)


def test_eval_rows_pass_on_a_t0229_run_dir(tmp_path: Path) -> None:
    """T-0229 A3: the eval re-derives a run carrying T-0229's tutor text and "Can you double-check my work?"."""
    out, man, _report = build_six(tmp_path, texts=(TUTOR, QUESTION))
    ev = _eval()
    run = ev.load_run(out, man)
    assert ev.texts_of(run) == (TUTOR, QUESTION)
    for row in ev.RUN_ROWS:
        row(run)


def test_eval_knows_exactly_the_two_pairs() -> None:
    """T-0229 A3: the eval's known pairs are T-0226 B's and T-0229's, typed from the task files, never mixed."""
    ev = _eval()
    assert tuple(ev.KNOWN_TEXTS) == ((T0226_TUTOR, T0226_QUESTION), (TUTOR, QUESTION))
    assert ev.test_tutor_text_and_question_recorded in ev.RUN_ROWS


def _drop_regime_texts(out: Path) -> None:
    for name in ("report.json", "manifest.json"):
        p = out / name
        obj = json.loads(p.read_text(encoding="utf-8"))
        for key in ("user_instructions", "question"):
            (obj.get("regime") or {}).pop(key, None)
        p.write_text(json.dumps(obj), encoding="utf-8")


#: T-0229 A3 adversarial: (regime texts, turn texts, post-edit, the row that must fail)
TEXT_FAULTS: dict[
    str, tuple[tuple[str, str], tuple[str, str], Callable[[Path], None] | None, str]
] = {
    # the regime claims the new pair, the turns still carry the old block and question
    "regime-new-turns-old": (
        (TUTOR, QUESTION),
        (T0226_TUTOR, T0226_QUESTION),
        None,
        "test_evidence_re_derived",
    ),
    # only the typed question disagrees with the record
    "question-not-typed": (
        (TUTOR, QUESTION),
        (TUTOR, T0226_QUESTION),
        None,
        "test_evidence_re_derived",
    ),
    # only the system block disagrees with the record
    "block-not-sent": (
        (TUTOR, QUESTION),
        (T0226_TUTOR, QUESTION),
        None,
        "test_evidence_re_derived",
    ),
    # a consistent run of a text no task names
    "unknown-text": (
        (TUTOR + " Be brief.", QUESTION),
        (TUTOR + " Be brief.", QUESTION),
        None,
        "test_tutor_text_and_question_recorded",
    ),
    # a consistent run of a pair that never ran: the new text with the old question
    "mixed-pair": (
        (TUTOR, T0226_QUESTION),
        (TUTOR, T0226_QUESTION),
        None,
        "test_tutor_text_and_question_recorded",
    ),
    # the regime records neither
    "not-recorded": (
        (TUTOR, QUESTION),
        (TUTOR, QUESTION),
        _drop_regime_texts,
        "test_tutor_text_and_question_recorded",
    ),
}


@pytest.mark.parametrize("fault", list(TEXT_FAULTS))
def test_eval_refuses_a_tutor_text_or_question_it_cannot_tie_to_the_record(
    tmp_path: Path, fault: str
) -> None:
    """T-0229 A3 (adversarial): accepting the run's recorded text must not become accepting anything: turns that
    do not carry the recorded text or question, an unknown text, a mixed pair, or no record fail the named row."""
    texts, turn_texts, edit, row = TEXT_FAULTS[fault]
    out, man, _report = build_six(tmp_path, texts=texts, turn_texts=turn_texts)
    if edit is not None:
        edit(out)
    ev = _eval()
    with pytest.raises(AssertionError):
        getattr(ev, row)(ev.load_run(out, man))


def _edit_report(out: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    p = out / "report.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    edit(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _page_row(obj: dict[str, Any], pid: str) -> dict[str, Any]:
    return next(r for r in obj["pages"] if r.get("id") == pid)


def _manifest_sha_off(out: Path) -> None:
    p = out / "manifest.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["manifest_sha256"] = "0" * 64
    p.write_text(json.dumps(obj), encoding="utf-8")


EVAL_FAULTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "verdict-flipped": (
        lambda o: _edit_report(
            o, lambda r: _page_row(r, "zq-m2").__setitem__("verdict", "located")
        ),
        "test_verdicts_re_derived",
    ),
    "count-bumped": (
        lambda o: _edit_report(o, lambda r: r["counts"]["mistake"].__setitem__("located", 2)),
        "test_counts_re_derived",
    ),
    "reading-off-by-one": (
        lambda o: _edit_report(
            o,
            lambda r: r.__setitem__(
                "reading", SIX_READING.replace("1 of 2 mistake", "2 of 2 mistake")
            ),
        ),
        "test_reading_re_derived",
    ),
    "latency-max": (
        lambda o: _edit_report(o, lambda r: r["latency"].__setitem__("max_ms", 2001)),
        "test_latency_re_derived",
    ),
    "thinking-chars": (
        lambda o: _edit_report(o, lambda r: _page_row(r, "zq-u2").__setitem__("thought_chars", 3)),
        "test_thinking_re_derived",
    ),
    "label": (
        lambda o: _edit_report(o, lambda r: r.__setitem__("label", "void")),
        "test_label_re_derived",
    ),
    "manifest-sha": (_manifest_sha_off, "test_manifest_sha_re_checked"),
}


@pytest.mark.parametrize("fault", list(EVAL_FAULTS))
def test_each_seeded_fault_fails_its_eval_row(tmp_path: Path, fault: str) -> None:
    """A8: a report or run manifest edited after the fact fails the eval row that re-derives that number (green
    before the edit)."""
    out, man, _report = build_six(tmp_path)
    edit, row = EVAL_FAULTS[fault]
    ev = _eval()
    getattr(ev, row)(ev.load_run(out, man))
    edit(out)
    with pytest.raises(AssertionError):
        getattr(ev, row)(ev.load_run(out, man))


# =====================================================================================================
# A3/A8/A9 through main with the engine stubbed (no socket): the clean run, P0, budget, errored stop, writes
# =====================================================================================================

#: the rig's four pages: (spoken, shown, thoughts, latency)
RIG_TEXTS: dict[str, tuple[str, str | None, list[str], int]] = {
    "pg_m1": ("The slip is on line 3: it should be x = 4.", None, ["Line three.", " Then x."], 900),
    "pg_m2": ("Everything is correct, well done.", None, ["the speed is off"], 1300),
    "pg_c1": ("Your work is correct.", None, ["All good, I see."], 1100),
    "pg_u1": ("Line 2 is wrong.", None, ["Hmm."], 700),
}
RIG_READING = "1 of 2 mistake pages located; 1 of 1 correct pages said correct, 0 claimed a mistake; 1 unclear pages reported"


class Engine:
    """Stubs for every engine-facing helper (by name, in every held module): ``/models`` lists the model, P0
    generates answer ``p0``, ``/tokenize`` answers 42, the VRAM sampler is inert, and T-0223's ``_one_turn`` on the
    private instance writes a synthetic turn (or raises, per ``plan``)."""

    def __init__(
        self, rig: Rig, *, p0: int = 200, plan: Callable[[int, str], str] | None = None
    ) -> None:
        from local_astra.conductor.llm_client import VllmEndpoint

        self.rig = rig
        self.calls: list[str] = []
        self.posts: list[tuple[str, Any]] = []
        self.order_at_first_turn: Any = None
        model = VllmEndpoint().model
        mp, h = rig.mp, rig.h
        tutor = api("TUTOR_INSTRUCTIONS")

        def get_text(url: str, *a: Any, **k: Any) -> str | None:
            if str(url).rstrip("/").endswith("/models"):
                return json.dumps({"data": [{"id": model}]})
            return "0.29.0-stub"

        async def generate(*a: Any, **k: Any) -> int:
            return p0

        async def idle(*a: Any, **k: Any) -> bool:
            return True

        def post(url: str, body: Any, *a: Any, **k: Any) -> tuple[int, Any]:
            self.posts.append((str(url), body))
            return 200, {"count": 42, "tokens": list(range(42))}

        class Vram:
            error = None

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def min_between(self, a: float, b: float) -> int:
                return 4096

        async def one_turn(*a: Any, **k: Any) -> dict[str, Any]:
            pid = next(x for x in a if isinstance(x, str) and x in PAGE_IDS)
            if not self.calls:
                p = rig.out / "order.json"
                self.order_at_first_turn = (
                    json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
                )
            self.calls.append(pid)
            mode = plan(len(self.calls), pid) if plan else "clean"
            if mode == "raise":
                raise RuntimeError(f"T-0226 rig: turn {len(self.calls)} raised")
            spoken, shown, thoughts, lat = RIG_TEXTS[pid]
            return write_turn(
                rig.out,
                pid,
                index=len(self.calls),
                spoken=spoken,
                shown=shown,
                thoughts=thoughts,
                latency=lat,
                tutor=tutor,
                question=api("QUESTION"),  # what T-0223's _one_turn types on the private instance
            )

        _patch_name(mp, h, "_get_text", get_text)
        _patch_name(mp, h, "_probe_generate", generate)
        _patch_name(mp, h, "_healthy", lambda *a, **k: True)
        _patch_name(
            mp,
            h,
            "_tokenize_count",
            lambda base, model_, budget, jpeg, *a, **k: 100 if jpeg is None else 1100,
        )
        _patch_name(mp, h, "_post_json", post)
        _patch_name(mp, h, "_wait_idle", idle)
        _patch_name(mp, h, "read_metrics", lambda *a, **k: None)
        _patch_name(mp, h, "_VramSampler", Vram)
        mp.setattr(private(), "_one_turn", one_turn)
        if getattr(h, "_one_turn", None) is getattr(private(), "_one_turn", None):
            mp.setattr(h, "_one_turn", one_turn)


def _trials(out: Path) -> list[dict[str, Any]]:
    p = out / "trials.jsonl"
    return (
        [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if p.is_file()
        else []
    )


def test_writes_only_under_out_dir(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    """A3/A6/A7/A9 through ``main`` with the engine stubbed (adversarial on privacy). The run is measured (exit 0);
    ``order.json`` holds the A3 order before the first turn; P0's token table has a row per page; ``/tokenize`` is
    posted each turn's thought text (no generation) and the count reaches the report; the regime names 1120, tools
    and search off, seed 0, n = 1 and the manifest's sha. Every write (audit hook: write-mode ``open``,
    ``os.mkdir``/``rename``/``remove``, ``shutil.copyfile``...) is under the out-dir or P-S's removed temporary
    directory, none under the manifest dir; ``manifest.json`` holds the sha and the dir's basename, never its
    absolute path or a PNG name; no file in the out-dir is a copy of a page or of the manifest. ``write_report``
    alone writes only under the out-dir. Every eval row passes on this run dir."""
    eng = Engine(rig)
    rc = rig.main()
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    out = rig.out
    assert rig.manifest_dir is not None and rig.manifest_arg is not None
    want_order = _order_oracle(PAGE_IDS)
    assert eng.calls == want_order, eng.calls
    assert eng.order_at_first_turn == want_order, eng.order_at_first_turn
    bad = _bad_writes(rig.writes(), [out], ps=True)
    assert bad == [], bad[:5]
    assert not any(_under(p, rig.manifest_dir) for _e, p in rig.writes())
    for root in _ps_roots(rig.writes()):
        assert not root.exists(), root
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "measured", (report.get("label"), report.get("violations"))
    assert report.get("reading") == RIG_READING, report.get("reading")
    probes = json.loads((out / "probes.json").read_text(encoding="utf-8"))
    assert probes["P0"]["passed"] is True
    assert sorted(probes["P0"]["image_tokens"]) == sorted(PAGE_IDS), probes["P0"].get(
        "image_tokens"
    )
    tok = [(u, b) for u, b in eng.posts if u.rstrip("/").endswith("/tokenize")]
    assert all(u.rstrip("/").endswith("/tokenize") for u, _ in eng.posts), [u for u, _ in eng.posts]
    for pid, (_s, _sh, thoughts, _l) in RIG_TEXTS.items():
        for t in thoughts:
            assert any(t in json.dumps(b) for _u, b in tok), (pid, t)
    rows = _rows(report)
    assert all(rows[p].get("thought_tokens") == 42 for p in PAGE_IDS), {
        p: rows[p].get("thought_tokens") for p in rows
    }
    reg = report.get("regime") or {}
    assert reg.get("visual_tokens") == 1120 and reg.get("tools") == "off", reg
    assert (
        reg.get("search") in ("off", False) and reg.get("order_seed") == 0 and reg.get("n") == 1
    ), reg
    manifest_text = (out / "manifest.json").read_text(encoding="utf-8")
    man = json.loads(manifest_text)
    assert man.get("manifest_sha256") == rig.sha and reg.get("manifest_sha256") == rig.sha
    assert man.get("manifest_dir_name") == rig.manifest_dir.name
    for needle in (str(rig.manifest_dir), str(rig.manifest_dir.resolve()), str(rig.manifest_arg)):
        assert needle not in manifest_text, needle
    for p in DEFAULT_PAGES:
        assert p["png"] not in manifest_text, p["png"]
    forbidden = {_sha((rig.manifest_dir / p["png"]).read_bytes()) for p in DEFAULT_PAGES}
    forbidden.add(_sha(rig.manifest_arg.read_bytes()))
    for f in out.rglob("*"):
        if f.is_file():
            assert _sha(f.read_bytes()) not in forbidden, f"{f} is a copy of a private file"
    ev = _eval()
    run = ev.load_run(out, rig.manifest_arg)
    for row in ev.RUN_ROWS:
        row(run)
    # write_report alone, under the audit hook
    (out / "report.json").unlink()
    (out / "report.md").unlink()
    _AUDIT["writes"] = []
    _AUDIT["on"] = True
    try:
        api("write_report")(out, rig.manifest_arg)
    finally:
        _AUDIT["on"] = False
    assert _bad_writes(list(_AUDIT["writes"]), [out], ps=False) == []


def test_p0_failing_is_void(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    """A3/A8: P0 answering 500 is void (exit 3); no turn runs; every page is written as not run."""
    eng = Engine(rig, p0=500)
    rc = rig.main()
    capsys.readouterr()
    assert rc == 3, rc
    assert eng.calls == []
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "void"
    assert all(_rows(report)[p].get("verdict") == "errored" for p in PAGE_IDS)


@pytest.mark.parametrize(("pattern", "calls"), [("all-raise", 3), ("two-then-ok", 4)])
def test_errored_stop(
    rig: Rig, capsys: pytest.CaptureFixture[str], pattern: str, calls: int
) -> None:
    """A3: ``ERRORED_STOP = 3`` consecutive non-clean turns stop the run and the rest are written as not run (both
    directions: every turn raising stops after 3 calls; raise, raise, clean, raise never stops). Either way a page
    did not run cleanly: incomplete, exit 1, and every page has a trial line."""

    def plan(n: int, pid: str) -> str:
        return "raise" if pattern == "all-raise" or n != 3 else "clean"

    eng = Engine(rig, plan=plan)
    rc = rig.main()
    capsys.readouterr()
    assert rc == 1, rc
    assert len(eng.calls) == calls, eng.calls
    trials = _trials(rig.out)
    assert sorted(str(t.get("page")) for t in trials) == sorted(PAGE_IDS), trials
    report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "incomplete"
    stopped = json.loads((rig.out / "manifest.json").read_text(encoding="utf-8")).get("stopped")
    if pattern == "all-raise":
        assert stopped, stopped
        last = _order_oracle(PAGE_IDS)[-1]
        t = next(x for x in trials if x.get("page") == last)
        assert not t.get("session_log"), t
    else:
        assert not stopped, stopped


@pytest.mark.parametrize(("ago", "starts"), [(1066.0, 0), (900.0, 4)])
def test_budget_stops_the_run(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ago: float,
    starts: int,
) -> None:
    """A3: the budget is measured from process start (probes included): with the process started 1066 s ago no
    turn starts and every page is written as not run (incomplete, exit 1); started 900 s ago, all four run."""
    eng = Engine(rig)
    assert _patch_name(monkeypatch, rig.h, "PROCESS_START_S", time.monotonic() - ago) >= 1
    rc = rig.main()
    capsys.readouterr()
    assert len(eng.calls) == starts, eng.calls
    assert sorted(str(t.get("page")) for t in _trials(rig.out)) == sorted(PAGE_IDS)
    if starts == 0:
        assert rc == 1, rc
        report = json.loads((rig.out / "report.json").read_text(encoding="utf-8"))
        assert report.get("label") == "incomplete"
        assert all(_rows(report)[p].get("verdict") == "errored" for p in PAGE_IDS)
    else:
        assert rc == 0, rc


# =====================================================================================================
# A9/A10 text guards and the wrapper
# =====================================================================================================


def test_no_tracked_file_names_the_private_dir() -> None:
    """A9: no tracked text file contains the private directory's name (built from two pieces here), and no tracked
    file with "t0226" in its path is an image or a PDF. Green by design; it guards every later commit."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"not a git work tree ({exc})")
    hits: list[str] = []
    for rel in filter(None, listing.split("\0")):
        path = REPO / rel
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        if NEEDLE in data.decode("utf-8", "replace"):
            hits.append(rel)
    assert hits == [], hits
    media = [
        rel
        for rel in filter(None, listing.split("\0"))
        if "t0226" in rel.lower() and Path(rel).suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
    ]
    assert media == [], media


def test_wrapper_shape() -> None:
    """Group B's wrapper, text checks: exactly one argument; the manifest sha a 64-hex literal passed through
    ``--manifest-sha256``; the four merge subjects matched through a here-string, never a pipe; ``timeout -k 30s
    21m``, ``--locked``, proxies unset; the eval only if ``report.json`` exists, with ``T0226_RUN_DIR`` and
    ``T0226_MANIFEST``; writes only ``logs/t0226-real-attempts-$SHA`` and ``logs/t0226-launch-$SHA`` (and the boot
    block's ``logs/vllm.log``, reading 6); ``bash -n`` passes; no private name."""
    assert WRAPPER.is_file(), f"T-0226: {WRAPPER.relative_to(REPO)} does not exist yet"
    text = WRAPPER.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r'"\$#"\s+-ne\s+1\b', code), "the wrapper must refuse any argument count but 1"
    assert re.search(r"MANIFEST_SHA256=['\"]?" + MANIFEST_SHA_PIN + r"\b", code), (
        "the manifest sha literal"
    )
    assert re.search(r"--manifest-sha256\s+\"?\$\{?MANIFEST_SHA256\}?\"?", code), (
        "passed through --manifest-sha256"
    )
    for subject in ("Merge T-0226-a:", "Merge T-0223-b:", "Merge T-0213-a2a:", "Merge T-0225:"):
        lines = [ln for ln in code.splitlines() if subject in ln and "grep" in ln]
        assert lines, f"no merge check for {subject}"
        assert all("<<<" in ln for ln in lines), (subject, lines)
    assert not re.search(r"\|\s*grep\b[^\n]*Merge T-", code), "a merge check through a pipe"
    assert re.search(r"timeout -k 30s 21m\b", code), "timeout -k 30s 21m"
    assert code.count("--locked") >= 2, "both uv run steps use --locked"
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert re.search(rf"-u {var}\b", code), var
    assert re.search(r'if \[ -f "\$OUT/report\.json" \]', code), (
        "the eval runs only if report.json exists"
    )
    assert "T0226_RUN_DIR=" in code and "T0226_MANIFEST=" in code
    assert re.search(r'OUT="?logs/t0226-real-attempts-\$SHA"?', code), "OUT"
    assert re.search(r'AUX="?logs/t0226-launch-\$SHA"?', code), "AUX"
    allowed = {"logs/t0226-real-attempts-$SHA", "logs/t0226-launch-$SHA", "logs/vllm.log"}
    named = set(re.findall(r"logs/[A-Za-z0-9_.$/{}-]*", code))
    assert named <= allowed, sorted(named - allowed)
    assert "tests/eval/test_t0226_real_attempts.py" in code
    assert "scripts/gpu/t0226_real_attempts.py" in code
    assert NEEDLE not in text
    assert "demos/handwritten/out" not in code, "the file list names no handwritten pages"
    r = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
