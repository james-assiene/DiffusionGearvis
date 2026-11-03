"""T-0163 criteria 1, 2, 4 and 8: the CLI wiring of the Brave Search API fallback (``cli._search_backend`` only).

Rigs (the T-0125 ones, ``tests/unit/test_cli_search.py``): a served live start over the CLI fakes whose HUD double
SIGINTs at once (``_serve`` runs its start-up and close), with a session log under ``tmp_path``. In-process rows point
the loader's default path (``brave_key.default_key_path``, the seam the conftest's ``brave_key_nowhere`` already
patches) at a tmp file; fresh-interpreter rows set ``HOME`` to a tmp dir, which moves the default path (criterion 3),
and record every ``open`` audit event on the key path. No row names, opens or stats the real key file; every key is a
generated canary.

- **No file is BASE** (criterion 1; active, green now and must stay green): the same served start in a fresh
  interpreter over ``git archive`` of BASE (3e4d45b7, the task's fork point; ``local_astra`` from the archive, origin
  checked) and over this tree, both with a tmp ``HOME`` holding no key file: stdout, stderr, the session-log records
  (normalised: tmp dir, log name, wall/t_ms/session_ms/pid/bytes) and the regime line are equal; the backend is a bare
  ``DdgsSearch`` and its label BASE's.
- **Where the key is read** (criterion 2): with a valid key file present, a start with search off, tools off, under
  ``--dry-run``, and ``load_tool_kit`` open nothing at the key path and import no Brave module (active; vacuous until
  the modules exist, hence the pending power row ``test_key_read_once_at_a_real_start``: a real live start opens it
  exactly once and imports them).
- **Configured / refused** (criterion 8; pending): the chain, the label suffix, the session-log note; the one stderr
  line and the bare ``DdgsSearch`` for a refused file, default-on and explicit ``--search ddgs`` alike (H-T163-3).

Where the criteria are ambiguous the rows take this reading (listed in the test-author's report):

- criterion 1 says a no-file start imports neither ``brave_key`` nor ``brave_api``; but with search on the loader must
  run to learn there is no file, and only ``brave_key.py`` may name the path (criterion 9). The row asserts
  ``brave_api`` and ``fallback`` are not imported and leaves ``brave_key`` free;
- "turns.jsonl and cards equal BASE's": no turn runs in a served start, and ``session.py`` is byte-identical
  (criterion 13), so the session-level equality reduces to the backend handed to it being a bare ``DdgsSearch``
  (``backend_type``);
- the fix for codes other than ``mode`` is not specified: only the line's frame and the ``mode`` fix are pinned.

Regime: CPU over fakes; no browser, no service, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import logging
import os
import secrets
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from tests import _git_range as gr
from tests.fakes.brave_api import FakeBrave
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0201_pins as t0201  # T-0201 D3 re-pin
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit._session_log_rig import read_records
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import Events, _Hud, argv, fake_live

REPO = Path(__file__).resolve().parents[2]
TASK = "T-0163"
#: T-0163's fork point (``main`` 3e4d45b7, "T-0170 done; PROGRESS"). A literal, never a moving ref.
BASE = "3e4d45b7661312ca8d01f1cc7918024e874c0429"

PROXY_ENV = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
LIVE = ("--tools", "live", "--browser", "headless")
DRY = ("--tools", "dry-run")
KEYLESS = "keyless: duckduckgo, then brave, then mojeek"
SUFFIX = "; then the Brave Search API (key configured)"
STARTUP = (
    "astra session: web search is on ({engines}): each search query the model makes is sent off this machine "
    "and logged in the session log; results are untrusted web text; --search off turns it off"
)
REFUSED_PREFIX = "astra session: the Brave Search API key file is not used: "
REFUSED_SUFFIX = "; web search stays keyless"
SEARCH_DEFAULT = "astra session: search: ddgs (default)"
SEARCH_FLAG = "astra session: search: ddgs (flag)"
_ALPHABET = "GHJKMNPQRSTVWXYZghjkmnpqrstvwxyz"
BRAVE_MODULES = (
    "local_astra.search.brave_key",
    "local_astra.search.brave_api",
    "local_astra.search.fallback",
)


def canary() -> str:
    return "BSAzq" + "".join(secrets.choice(_ALPHABET) for _ in range(27))


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in (name, "local_astra.search"):
            raise NotImplementedError(f"{TASK}: {name} does not exist yet") from exc
        raise


def key_file(directory: Path, content: str, mode: int = 0o600) -> Path:
    path = directory / ".config" / "diffusion_gearvis" / "brave_api_key"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(mode)
    return path


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")


# =====================================================================================================
# The fresh-interpreter runner (BASE and this tree)
# =====================================================================================================

_RUNNER = r"""
import contextlib, io, json, os, re, sys, tempfile
from pathlib import Path

opened = []

def _audit(event, args):
    if event != "open" or not args or args[0] is None or isinstance(args[0], int):
        return
    try:
        name = os.fsdecode(args[0])
    except TypeError:
        return
    if name.endswith("brave_api_key") or ".config/diffusion_gearvis" in name:
        opened.append(name)

sys.addaudithook(_audit)

import pytest
import local_astra
from local_astra import cli
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import Events, _Hud, argv, fake_live

spec = json.loads(sys.argv[1])
tmp = Path(tempfile.mkdtemp())
logs = tmp / "sessions"
out, err = io.StringIO(), io.StringIO()
search_type = None
code = None
with pytest.MonkeyPatch.context() as mp:
    for name in PROXY_NAMES:
        mp.delenv(name, raising=False)
    mp.setenv("DISPLAY", ":99")
    calls = T.Calls()
    events = Events()
    T.fake_clients(mp, calls)
    T.patch_detok_load(mp, calls, tmp)
    if spec["mode"] == "load_tool_kit":
        from local_astra.config import Profile
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            kit, _ = cli.load_tool_kit(Profile(name="lean"))
        code = 0
        search_type = type(kit.search).__name__
    else:
        if spec["mode"] == "serve":
            T.point_cli_at(mp, payloads=tmp / "nowhere" / "native_T4.json")
            _Hud.events = events
            mp.setattr(cli, "HudServer", _Hud)
        else:
            T.stop_before_serving(mp, calls)
        fake_live(mp, events)
        args = argv(tmp, *(["--dry-run"] if spec["mode"] == "dry" else []), *spec["flags"])
        if spec["mode"] == "serve":
            args.remove("--no-log")
            args += ["--log-dir", str(logs)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli(args)
        kit = calls.kit() if calls.session_kwargs else None  # --dry-run constructs no Session
        search_type = None if kit is None else type(getattr(kit, "search", None)).__name__

LOGNAME = re.compile(r"\d{8}-\d{6}-\d+\.jsonl")

def norm(text):
    return LOGNAME.sub("<LOG>.jsonl", text.replace(str(tmp), "<TMP>"))

records = []
for f in sorted(logs.rglob("*.jsonl")) if logs.exists() else []:
    for line in f.read_text().splitlines():
        r = json.loads(line)
        for k in ("wall", "t_ms", "session_ms"):
            r.pop(k, None)
        data = r.get("data")
        if isinstance(data, dict):
            for k in ("pid", "bytes"):
                data.pop(k, None)
        records.append(json.loads(norm(json.dumps(r, sort_keys=True))))
print("RESULT " + json.dumps({
    "code": code,
    "out": norm(out.getvalue()),
    "err": norm(err.getvalue()),
    "records": records,
    "opened": opened,
    "loaded": sorted(m for m in sys.modules if m in BRAVE),
    "search_type": search_type,
    "origin": local_astra.__file__,
}), flush=True)
""".replace("PROXY_NAMES", repr(PROXY_ENV)).replace("BRAVE", repr(BRAVE_MODULES))


def _fresh(cwd: Path, home: Path, mode: str, flags: tuple[str, ...] = ()) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_ENV}
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, json.dumps({"mode": mode, "flags": list(flags)})],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    found = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert found, (
        f"the fresh interpreter printed no result\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )
    got: dict[str, Any] = json.loads(found[-1][len("RESULT ") :])
    return got


def _base_tree(dest: Path) -> Path:
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            "--format=tar",
            BASE,
            "local_astra",
            "services",
            "docs/cd",
        ],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


#: T-0179 A-5: the flag the working tree's live start passes, the line it prints, and BASE's line in its place
GUARD_OFF_PIN = "--no-repeat-guard"
GUARD_OFF_PIN_LINE = "astra session: repeat guard: off (flag)"
GUARD_BASE_LINE = "astra session: repeat guard: off (default)"

#: the two real starts criterion 1 compares: the default-on live start, and the dry-run opt-in
STARTS = {"live_default": LIVE, "dry_run_opt_in": (*DRY, "--search", "ddgs")}


@pytest.fixture(scope="module")
def no_file_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """Each start of ``STARTS`` served at BASE and on this tree, ``HOME`` a tmp dir with no key file."""
    gr.require_commits(BASE)
    tmp = tmp_path_factory.mktemp("t0163_no_file")
    base_dir = _base_tree(tmp / "base_tree")
    home = tmp / "home"
    home.mkdir()
    runs: dict[str, dict[str, Any]] = {}
    for name, flags in STARTS.items():
        base = _fresh(base_dir, home, "serve", flags)
        assert Path(base["origin"]).resolve().is_relative_to(base_dir.resolve()), (
            f"origin proof: the BASE run imported local_astra from {base['origin']}"
        )
        # T-0179 A-5 (the T-0174 D4b rig pattern): the working tree's live start pins the repeat guard off (BASE's)
        pinned = (*flags, GUARD_OFF_PIN) if name == "live_default" else flags
        # T-0202 D3 re-pin (pin off): the live start also pins the wait tool off (BASE's; B9: the flag prints nothing)
        pinned = tuple(t0202.pin_off(list(pinned))) if name == "live_default" else pinned
        now = _fresh(REPO, home, "serve", pinned)
        assert Path(now["origin"]).resolve().is_relative_to(REPO.resolve()), now["origin"]
        assert base["code"] == 0 and now["code"] == 0, (base["err"], now["err"])
        runs[name] = {"base": base, "now": now}
    assert not (home / ".config").exists(), "premise: no key file under the tmp HOME"
    return runs


def _regime(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith("astra session: regime:")]


@pytest.mark.parametrize("start", list(STARTS))
@pytest.mark.parametrize(
    "what", ["stdout", "stderr", "session_log", "regime", "backend_type", "label"]
)
def test_no_file_is_base(no_file_runs: dict[str, Any], start: str, what: str) -> None:
    """Criterion 1 (negative): with no key file a real start is BASE's, byte for byte."""
    base, now = no_file_runs[start]["base"], no_file_runs[start]["now"]
    if what == "stdout":
        # T-0172 R9 pin move (B9): both starts are tools-on, so the default reask line sits before the regime line
        out = now["out"]
        if start == "live_default":
            # T-0179 A-5: the pin's one ``off (flag)`` line read back as BASE's ``off (default)``
            assert out.splitlines().count(GUARD_OFF_PIN_LINE) == 1, out
            out = out.replace(GUARD_OFF_PIN_LINE, GUARD_BASE_LINE)
            out = t0201.admit_default_line(out, start)  # T-0201 D3 re-pin (admit): the default line
            # T-0207 C3 re-pin (admit): exactly one default ``completion check: on (default)`` line
            out = t0207.admit_default_line(out, start)
        assert t0172.admit_reask_line(out, "start") == base["out"]
        assert (SEARCH_DEFAULT if start == "live_default" else SEARCH_FLAG) in now[
            "out"
        ].splitlines()
    elif what == "stderr":
        assert now["err"] == base["err"]
        assert STARTUP.format(engines=KEYLESS) in now["err"].splitlines(), "premise: search was on"
    elif what == "session_log":
        assert base["records"], "premise: the BASE run wrote a session log"
        assert now["records"] == base["records"]
        notes = [
            r for r in now["records"] if (r.get("data") or {}).get("what") == "web search fallback"
        ]
        assert notes == [], "no fallback note without a key file"
    elif what == "regime":
        assert len(_regime(now["out"])) == 1 and _regime(now["out"]) == _regime(base["out"])
    elif what == "backend_type":
        assert now["search_type"] == base["search_type"] == "DdgsSearch", (
            "a bare DdgsSearch, not a wrapper"
        )
    else:
        backend, label = cli._search_backend("ddgs")
        from local_astra.search.ddgs_backend import DdgsSearch

        assert type(backend) is DdgsSearch, type(backend)
        assert label == KEYLESS, label
        asyncio.run(backend.aclose())


def test_no_file_imports_neither_module(no_file_runs: dict[str, Any]) -> None:
    """Criterion 1: without a key file no Brave backend is built, so ``brave_api`` and ``fallback`` are never
    imported (``brave_key`` is left free: the loader runs to learn there is no file; see the module docstring)."""
    for start, run in no_file_runs.items():
        loaded = set(run["now"]["loaded"])
        assert not loaded & {"local_astra.search.brave_api", "local_astra.search.fallback"}, (
            start,
            sorted(loaded),
        )


def test_dry_run_output_unchanged_with_a_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 2 (negative): ``--dry-run`` never reads the key, so its output with a valid key file is the
    output without one."""
    outs = []
    for i, present in enumerate((False, True)):
        d = tmp_path / str(i)
        d.mkdir()
        if present:
            path = key_file(d / "home", canary() + "\n")
            # before the loader exists nothing can read the file (the row is vacuous until then)
            with contextlib.suppress(NotImplementedError):
                _point_default_at(monkeypatch, path)
        fake_live(monkeypatch, Events())
        code = run_cli(argv(d, "--dry-run", *LIVE))
        got = capsys.readouterr()
        assert code == 0, got.err
        outs.append((got.out.replace(str(d), "<D>"), got.err.replace(str(d), "<D>")))
    assert outs[0] == outs[1]


# =====================================================================================================
# Criterion 2: where the key is (not) read
# =====================================================================================================

#: case -> (runner mode, flags)
NOT_READ: dict[str, tuple[str, tuple[str, ...]]] = {
    "search_off": ("real", (*LIVE, "--search", "off")),
    "tools_off": ("real", ()),
    "dry_run_tools_default": ("real", DRY),
    "dry_run_flag": ("dry", LIVE),
    "dry_run_flag_explicit": ("dry", (*LIVE, "--search", "ddgs")),
    "load_tool_kit": ("load_tool_kit", ()),
}


@pytest.mark.parametrize("case", list(NOT_READ))
def test_key_not_read_when(tmp_path: Path, case: str) -> None:
    """With a valid key file at the default path (a tmp ``HOME``), nothing opens it and no Brave module is
    imported. Vacuous until the modules exist; ``test_key_read_once_at_a_real_start`` is the same harness's power
    row."""
    home = tmp_path / "home"
    key = canary()
    key_file(home, key + "\n")
    mode, flags = NOT_READ[case]
    got = _fresh(REPO, home, mode, flags)
    assert got["code"] == 0, got["err"][-2000:]
    assert got["opened"] == [], f"[{case}] the key path was opened: {got['opened']}"
    assert got["loaded"] == [], f"[{case}] Brave modules imported: {got['loaded']}"
    assert key not in got["out"] + got["err"]


def test_key_read_once_at_a_real_start(tmp_path: Path) -> None:
    """Power row for ``test_key_not_read_when`` (criterion 2): a real live start with a valid key opens the key
    file exactly once and imports the three modules; nothing it prints holds the key."""
    home = tmp_path / "home"
    key = canary()
    path = key_file(home, key + "\n")
    got = _fresh(REPO, home, "real", LIVE)
    assert got["code"] == 0, got["err"][-2000:]
    leaf = [p for p in got["opened"] if p.endswith("brave_api_key")]
    assert len(leaf) == 1, f"read once per process, at construction; opens {got['opened']}"
    assert Path(leaf[0]).name == path.name
    assert got["loaded"] == sorted(BRAVE_MODULES), got["loaded"]
    assert got["search_type"] == "FallbackSearch", got["search_type"]
    assert key not in got["out"] + got["err"]


# =====================================================================================================
# Criterion 8: configured and refused, in-process
# =====================================================================================================


def _point_default_at(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    m = _import("local_astra.search.brave_key")
    monkeypatch.setattr(m, "default_key_path", lambda: path)


def serve(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> tuple[int, str, str, T.Calls, Path]:
    """A served start with a session log under ``tmp_path/sessions``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = Events()
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json")
    _Hud.events = events
    monkeypatch.setattr(cli, "HudServer", _Hud)
    fake_live(monkeypatch, events)
    logs = tmp_path / "sessions"
    args = argv(tmp_path, *flags)
    args.remove("--no-log")
    args += ["--log-dir", str(logs)]
    code = run_cli(args)
    got = capsys.readouterr()
    return code, got.out, got.err, calls, logs


def fallback_notes(logs: Path) -> list[dict[str, Any]]:
    return [
        r["data"]
        for f in sorted(logs.rglob("*.jsonl"))
        for r in read_records(f)
        if r.get("kind") == "note" and (r.get("data") or {}).get("what") == "web search fallback"
    ]


class _Spied:
    def __init__(self) -> None:
        self.chains: list[tuple[Any, ...]] = []
        self.braves: list[Any] = []


def _spy_constructors(monkeypatch: pytest.MonkeyPatch) -> _Spied:
    """Record what ``cli._search_backend`` builds: the chain's arguments and the Brave backend's key argument (the
    classes are wrapped where ``cli`` imports them, inside its body)."""
    api = _import("local_astra.search.brave_api")
    fb = _import("local_astra.search.fallback")
    spied = _Spied()
    real_brave, real_chain = api.BraveApiSearch, fb.FallbackSearch

    class Brave(real_brave):  # type: ignore[misc, valid-type]
        def __init__(self, key: Any, *a: Any, **k: Any) -> None:
            spied.braves.append(key)
            super().__init__(key, *a, **k)

    class Chain(real_chain):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            spied.chains.append((*a, *k.values()))
            super().__init__(*a, **k)

    Brave.__name__ = Brave.__qualname__ = "BraveApiSearch"
    Chain.__name__ = Chain.__qualname__ = "FallbackSearch"
    monkeypatch.setattr(api, "BraveApiSearch", Brave)
    monkeypatch.setattr(fb, "FallbackSearch", Chain)
    return spied


def _sent_key(key_arg: Any) -> str:
    api = _import("local_astra.search.brave_api")
    fake = FakeBrave()

    async def go() -> None:
        b = api.BraveApiSearch(key_arg, transport=fake.transport())
        try:
            await b.search("q")
        finally:
            await b.aclose()

    asyncio.run(go())
    return str(fake.requests[0].headers.get("x-subscription-token"))


def test_valid_key_builds_chain_and_says_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    key = canary()
    path = key_file(tmp_path / "home", key + "\n")
    _point_default_at(monkeypatch, path)
    spied = _spy_constructors(monkeypatch)
    from local_astra.search.ddgs_backend import DdgsSearch

    # the helper itself
    backend, label = cli._search_backend("ddgs")
    try:
        assert type(backend).__name__ == "FallbackSearch", type(backend)
        assert label == KEYLESS + SUFFIX, label
        assert len(spied.chains) == 1 and len(spied.chains[0]) == 2, spied.chains
        primary, fallback = spied.chains[0]
        assert type(primary) is DdgsSearch, type(primary)
        assert type(fallback).__name__ == "BraveApiSearch", type(fallback)
        assert len(spied.braves) == 1 and _sent_key(spied.braves[0]) == key, (
            "the chain's Brave backend holds the file's key"
        )
    finally:
        asyncio.run(backend.aclose())
    # a served default-on live start
    code, out, err, calls, logs = serve(tmp_path / "run", capsys, monkeypatch, *LIVE)
    assert code == 0, f"{code}\n{out}{err}"
    assert err.splitlines().count(STARTUP.format(engines=KEYLESS + SUFFIX)) == 1, err
    assert not any(x.startswith(REFUSED_PREFIX) for x in err.splitlines()), err
    assert [x for x in out.splitlines() if x.startswith("astra session: search:")] == [
        SEARCH_DEFAULT
    ]
    kit = calls.kit()
    assert type(kit.search).__name__ == "FallbackSearch" and kit.web_search == "ddgs", kit
    assert fallback_notes(logs) == [{"what": "web search fallback", "brave_api": "configured"}]
    assert key not in out + err


#: case -> (file content builder, mode, refusal code)
def _refused_file(case: str, home: Path, key: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    if case == "mode":
        return key_file(home, key + "\n", 0o644)
    if case == "malformed":
        return key_file(home, f"{key} {key}\n")
    if case == "symlink":
        real = home / "elsewhere" / "real_key"
        real.parent.mkdir(parents=True)
        real.write_text(key + "\n")
        real.chmod(0o600)
        path = home / ".config" / "diffusion_gearvis" / "brave_api_key"
        path.parent.mkdir(parents=True)
        path.symlink_to(real)
        return path
    if case == "owner":
        path = key_file(home, key + "\n")
        euid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: euid + 1)
        return path
    raise AssertionError(case)


@pytest.mark.parametrize("start", ["default_on", "explicit"])
@pytest.mark.parametrize("case", ["mode", "owner", "symlink", "malformed"])
def test_refused_key_one_stderr_line_stays_keyless(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    start: str,
) -> None:
    """Criterion 8 (H-T163-3): a present but refused key file is one stderr line and one note, and the session
    starts keyless (exit 0, the bare ``DdgsSearch``, BASE's label), whether search is on by default or asked for
    with ``--search ddgs``."""
    _import("local_astra.search.brave_key")
    key = canary()
    path = _refused_file(case, tmp_path / "home", key, monkeypatch)
    _point_default_at(monkeypatch, path)
    flags = LIVE if start == "default_on" else (*LIVE, "--search", "ddgs")
    code, out, err, calls, logs = serve(tmp_path / "run", capsys, monkeypatch, *flags)
    assert code == 0, f"[{case}/{start}] the session still starts; {code}\n{out}{err}"
    lines = [x for x in err.splitlines() if x.startswith(REFUSED_PREFIX)]
    assert len(lines) == 1, f"exactly one line; {err}"
    assert lines[0].endswith(REFUSED_SUFFIX), lines[0]
    if case == "mode":
        assert lines[0].endswith(f"; chmod 600 {path}{REFUSED_SUFFIX}"), lines[0]
    assert err.splitlines().count(STARTUP.format(engines=KEYLESS)) == 1, (
        f"the keyless startup line, unchanged; {err}"
    )
    from local_astra.search.ddgs_backend import DdgsSearch

    kit = calls.kit()
    assert type(kit.search) is DdgsSearch and kit.web_search == "ddgs", kit
    assert fallback_notes(logs) == [
        {"what": "web search fallback", "brave_api": "not configured", "why": case}
    ]
    want_line = SEARCH_DEFAULT if start == "default_on" else SEARCH_FLAG
    assert [x for x in out.splitlines() if x.startswith("astra session: search:")] == [want_line]
    assert key not in out + err
    assert key not in "\n".join(p.read_text() for p in logs.rglob("*") if p.is_file())


def test_refused_helper_returns_the_bare_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_search_backend`` itself, with a 0644 file: the bare ``DdgsSearch`` and BASE's label (no wrapper)."""
    _import("local_astra.search.brave_key")
    path = key_file(tmp_path / "home", canary() + "\n", 0o644)
    _point_default_at(monkeypatch, path)
    from local_astra.search.ddgs_backend import DdgsSearch

    backend, label = cli._search_backend("ddgs")
    try:
        assert type(backend) is DdgsSearch and label == KEYLESS, (type(backend), label)
    finally:
        asyncio.run(backend.aclose())


def test_search_choices_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 8 (H-T163-4): no new ``--search`` value; the profile's choices and ``ToolKit.web_search``'s values
    are as at BASE."""
    import typing

    from local_astra import config

    assert config.SEARCH_CHOICES == ("off", "ddgs")
    assert typing.get_args(config.SearchChoice) == ("off", "ddgs")
    assert typing.get_args(config.WebSearchBackend) == ("ddgs",)
    for value in ("brave", "brave_api", "brave-api", "api"):
        fake_live(monkeypatch, Events())
        d = tmp_path / value
        d.mkdir()
        code = run_cli(argv(d, "--dry-run", *LIVE, "--search", value))
        err = capsys.readouterr().err
        assert code == 2 and "--search" in err, (value, code, err)


@pytest.mark.parametrize("case", ["configured", "refused_malformed"])
def test_canary_absent_from_every_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Criterion 4 (adversarial): a served start with a key file (valid, or malformed with the key inside): the key
    is in no byte of stdout, stderr, any file the run wrote (session log, turns.jsonl, data dir), any log record at
    DEBUG, or ``repr()`` of the backend the session got. Logs say only ``configured`` or ``not configured``."""
    caplog.set_level(logging.DEBUG)
    for name in ("httpx", "httpcore", "local_astra"):
        caplog.set_level(logging.DEBUG, logger=name)
    _import("local_astra.search.brave_key")
    key = canary()
    content = key + "\n" if case == "configured" else f"{key}\n{key}\n"
    path = key_file(tmp_path / "home", content)
    _point_default_at(monkeypatch, path)
    run_dir = tmp_path / "run"
    code, out, err, calls, logs = serve(run_dir, capsys, monkeypatch, *LIVE)
    assert code == 0, f"{code}\n{out}{err}"
    kit = calls.kit()
    want = "FallbackSearch" if case == "configured" else "DdgsSearch"
    assert type(kit.search).__name__ == want, "premise"
    shown = [out, err, repr(kit.search), str(kit.search), repr(kit)]
    shown += [r.getMessage() for r in caplog.records]
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    assert any(p.suffix == ".jsonl" for p in files), "premise: a session log was written"
    shown += [p.read_bytes().decode("utf-8", "replace") for p in files]
    leaked = [s[:120] for s in shown if key in s or key[5:17] in s]
    assert leaked == [], f"the key appears in: {leaked}"
    words = {n.get("brave_api") for n in fallback_notes(logs)}
    assert words == {"configured" if case == "configured" else "not configured"}, words
