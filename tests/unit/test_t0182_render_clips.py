"""T-0182 A2, A4, A5: ``scripts/render_clips.py`` behind a scripted TTS client (docs/tasks/T-0182.md, group A).

**The interface these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``scripts/render_clips.py`` is loadable by path and has ``main(argv: list[str] | None = None) -> int`` (synchronous;
  an ``argparse`` ``SystemExit`` is read as its exit code). Output lines may go to stdout or stderr.
- **The client seam:** the module-global name ``PocketTtsClient`` (imported from ``conductor.tts_client``) is what the
  script constructs, as ``PocketTtsClient(profile.services)``, and it calls ``await client.speak(uid, text)`` (voice
  ``DEFAULT_VOICE``) and iterates ``utterance.audio()``. The rows replace that name with a factory returning
  :class:`ScriptedTts`; the contract module puts the real client there, over ``tests/fakes/pocket_tts_double.py``.
- ``--data-dir`` defaults to ``memory.store.DEFAULT_DIR``. Every row passes a ``tmp_path`` data dir, and an autouse
  fixture rebinds ``DEFAULT_DIR`` in every loaded module to a ``tmp_path`` sentinel and asserts the real data dir's
  ``clips/`` is untouched.
- ``--kinds`` is given one kind per row, so no separator is pinned. Each kind's line contains the kind and one of
  ``rendered``, ``skipped (current)``, ``failed:``.
- The manifest is ``<data_dir>/clips/manifest.json``, read as JSON ``{"clips": {kind: entry}}`` (the registry module's
  docstring) and through ``clips.status``.

**Readings (reported):** "0.3–6.0 s" is inclusive (exactly 0.3 s and 6.0 s render); "full scale" is a sample of
32767 or -32768 (32766 and -32766 render); dBFS is ``20·log10(|x| / 32768)``, compared within 0.05 dB, so a 32767
reference also passes; ``tts`` in the manifest is the lock's ``pocket_tts`` value verbatim (``3.1.0``); ``git_commit``
agrees with ``git rev-parse HEAD`` in its first seven characters. "Writes nothing for that kind" is read as: no asset,
no temporary file of any name, and no manifest entry for it; an asset that existed before keeps its bytes.

Oracles: PCM built by hand (``square``, ``pulse``); dBFS and durations computed by hand in the row; the lock read with
``tomllib``; the registry definition (``get_args(ClipKind)`` ∩ ``CLIP_TEXT``) driven by ``set_registry``.

Activated for ``/implement T-0182`` group A (pending at 6d2e1672): until the script exists, the loader turns its
absence into a named ``AssertionError``. The scope guard is green at BASE by design (``TIP_A`` starts at ``BASE_A``).

Regime: CPU; no service, no socket (every render row runs under ``no_sockets``), files only under ``tmp_path``;
n = 1, no seed.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import re
import socket
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor.tts_client import DEFAULT_VOICE, TtsClientError, TtsFormatError
from local_astra.config import Services
from local_astra.memory import store
from tests import _git_range as gr
from tests.unit.test_t0182_clips_registry import (
    BASE_REGISTRY,
    ENDING_TEXTS,
    api,
    ours,
    set_registry,
    square,
    text_sha,
    tts_pin,
)

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "render_clips.py"
REAL_DEFAULT_DIR = store.DEFAULT_DIR
OUTCOMES = ("rendered", "skipped (current)", "failed:")

_names = itertools.count()


# --- PCM, by hand ---------------------------------------------------------------------------------


def samples(values: Sequence[int]) -> bytes:
    return b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)


def pulse(ms: int = 1000, *, high: int = 16384) -> bytes:
    """``high, 0, 0, 0`` repeated: peak 20·log10(high/32768), RMS 6.02 dB lower (sqrt(1/4) of the peak)."""
    return samples([high if i % 4 == 0 else 0 for i in range(ms * 24)])


def with_sample(pcm: bytes, index: int, value: int) -> bytes:
    return pcm[: 2 * index] + value.to_bytes(2, "little", signed=True) + pcm[2 * index + 2 :]


def dbfs(x: float) -> float:
    return 20 * math.log10(x / 32768)


# --- the scripted client ------------------------------------------------------------------------


class Plan:
    """What the scripted client does for one text: yield ``chunks``, then optionally raise ``error`` (``at_speak``
    raises from ``speak`` instead)."""

    def __init__(
        self,
        chunks: Sequence[bytes] = (),
        *,
        error: BaseException | None = None,
        at_speak: bool = False,
    ):
        self.chunks = list(chunks)
        self.error = error
        self.at_speak = at_speak


class ScriptedUtterance:
    def __init__(self, utterance_id: str, plan: Plan) -> None:
        self.utterance_id = utterance_id
        self.plan = plan
        self.failed = False
        self.cancelled = False

    async def audio(self) -> AsyncIterator[bytes]:
        for chunk in self.plan.chunks:
            yield chunk
        if self.plan.error is not None:
            self.failed = True
            raise self.plan.error

    def words(self) -> None:
        return None

    async def cancel(self) -> None:
        self.cancelled = True

    @property
    def first_audio_latency_ms(self) -> float | None:
        return 0.0 if self.plan.chunks else None


class ScriptedTts:
    """A ``PocketTtsClient`` stand-in: records ``(text, voice)`` per ``speak``; serves a ``Plan`` per text (a list of
    plans is served in turn). Any other attribute the script reaches for is a named AssertionError."""

    def __init__(
        self, plans: dict[str, Plan | list[Plan]] | None = None, *, default: Plan | None = None
    ) -> None:
        self.plans = dict(plans or {})
        self.default = default if default is not None else Plan([square(1000)])
        self.calls: list[tuple[str, str]] = []
        self.services: Any = None
        self.url = Services().tts
        self.closed = False

    async def speak(
        self, utterance_id: str, text: str, voice: str = DEFAULT_VOICE
    ) -> ScriptedUtterance:
        self.calls.append((text, voice))
        plan = self.plans.get(text, self.default)
        if isinstance(plan, list):
            plan = plan.pop(0) if len(plan) > 1 else plan[0]
        if plan.at_speak and plan.error is not None:
            raise plan.error
        return ScriptedUtterance(utterance_id, plan)

    async def aclose(self) -> None:
        self.closed = True

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"render_clips used PocketTtsClient.{name}, which ScriptedTts does not model"
        )

    def texts(self) -> list[str]:
        return [text for text, _voice in self.calls]


# --- fixtures and helpers -----------------------------------------------------------------------


def _clips_listing(root: Path) -> dict[str, int]:
    clips = root / "clips"
    if not clips.is_dir():
        return {}
    return {p.name: p.stat().st_mtime_ns for p in clips.iterdir()}


@pytest.fixture(autouse=True)
def jail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Rebind ``DEFAULT_DIR`` wherever a loaded module holds the real one; prove the real ``clips/`` untouched."""
    sentinel = tmp_path / "default-data-dir"
    before = _clips_listing(REAL_DEFAULT_DIR)
    for module in ours():
        if module.__dict__.get("DEFAULT_DIR") == REAL_DEFAULT_DIR:
            monkeypatch.setattr(module, "DEFAULT_DIR", sentinel)
    yield sentinel
    assert _clips_listing(REAL_DEFAULT_DIR) == before, (
        f"a T-0182 row wrote under the real data dir {REAL_DEFAULT_DIR}/clips"
    )


def load_script() -> ModuleType:
    assert SCRIPT.is_file(), "T-0182 A2: scripts/render_clips.py does not exist yet"
    name = f"_t0182_render_clips_{next(_names)}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise AssertionError(
            f"scripts/render_clips.py failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    assert callable(getattr(module, "main", None)), (
        "T-0182 A2: render_clips.py has no main(argv) -> int"
    )
    assert hasattr(module, "PocketTtsClient"), (
        "T-0182 A2: render_clips.py must construct the module-global PocketTtsClient (the client seam)"
    )
    return module


class _Missing:
    """Stands in for a script that does not load: any use raises the loader's AssertionError."""

    def __init__(self, reason: str) -> None:
        self.__dict__["_reason"] = reason
        self.__name__ = "_t0182_missing"

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(self.__dict__["_reason"])


@pytest.fixture
def script(jail: Path) -> Iterator[ModuleType]:
    """Loaded after ``jail`` (so a ``from ... import DEFAULT_DIR`` binds the sentinel), kept in ``sys.modules`` for
    the row (so ``set_registry`` reaches its ``ClipKind``)."""
    try:
        module = load_script()
    except AssertionError as exc:
        # Raised in the row's body, not its setup, so a missing script is a failed assertion (an xfail while pending).
        yield _Missing(str(exc))  # type: ignore[misc]
        return
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.fixture
def base_registry(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> None:
    """BASE's registry exactly, whatever later tasks register: ack, hmm, still_working."""
    set_registry(monkeypatch, dict(BASE_REGISTRY))


def install(script: ModuleType, monkeypatch: pytest.MonkeyPatch, tts: ScriptedTts) -> list[Any]:
    made: list[Any] = []

    def factory(services: Any = None, *args: Any, **kwargs: Any) -> ScriptedTts:
        made.append(services)
        tts.services = services
        return tts

    monkeypatch.setattr(script, "PocketTtsClient", factory)
    return made


@contextmanager
def no_sockets() -> Iterator[list[Any]]:
    attempts: list[Any] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def connect(self: socket.socket, address: Any) -> Any:
        attempts.append(address)
        raise AssertionError(f"T-0182 A2: render_clips opened a socket to {address!r}")

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(address)
        raise AssertionError(f"T-0182 A2: render_clips opened a connection to {address!r}")

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect  # type: ignore[method-assign,assignment]
    socket.create_connection = create_connection  # type: ignore[assignment]
    try:
        yield attempts
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.create_connection = real_create  # type: ignore[assignment]


def run(script: ModuleType, argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    try:
        code = script.main(argv)
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    out = capsys.readouterr()
    return int(code or 0), out.out + out.err


def outcome(output: str, kind: str) -> str:
    """The one line for ``kind``: it names the kind and exactly one outcome."""
    lines = [
        line
        for line in output.splitlines()
        if re.search(rf"(?<![A-Za-z_]){re.escape(kind)}(?![A-Za-z_])", line)
        and any(o in line for o in OUTCOMES)
    ]
    assert len(lines) == 1, f"A2: one line per kind; for {kind!r} got {lines!r} in\n{output}"
    return lines[0]


def files_under(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def manifest(data_dir: Path) -> dict[str, Any]:
    import json

    path = data_dir / "clips" / "manifest.json"
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text()).get("clips", {}))


def asset(data_dir: Path, kind: str) -> Path:
    return Path(api("clip_path")(data_dir, kind))


def head_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


# --- the default set ----------------------------------------------------------------------------


def test_default_renders_the_registered_ending_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 + A4, re-scoped by B6 (the default set gains ``PROGRESS_KINDS``): with all six kinds registered, the default
    set is the three ending kinds and ``still_working``, each rendered once from ``CLIP_TEXT`` verbatim through
    ``PocketTtsClient(profile.services)`` with ``DEFAULT_VOICE``; the asset is the served PCM byte for byte and the
    manifest entry is computed from it. ack and hmm are not written (A4; B6: fillers only with ``filler_clips``). No
    socket is opened."""
    texts = {**ENDING_TEXTS, "still_working": BASE_REGISTRY["still_working"]}
    set_registry(monkeypatch, {**BASE_REGISTRY, **ENDING_TEXTS})
    pcm = {
        "out_of_steps": pulse(1000, high=16384),  # peak -6.02, rms -12.04
        "not_finished": square(1000, high=8192),  # peak = rms = -12.04
        "lost_thread": square(2000, high=4096),  # peak = rms = -18.06
        "still_working": square(1000, high=16384),  # peak = rms = -6.02
    }
    expected_db = {
        "out_of_steps": (-6.02, -12.04),
        "not_finished": (-12.04, -12.04),
        "lost_thread": (-18.06, -18.06),
        "still_working": (-6.02, -6.02),
    }
    tts = ScriptedTts({texts[k]: Plan([v[:9600], v[9600:]]) for k, v in pcm.items()})
    made = install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets() as attempts:
        code, out = run(script, ["--data-dir", str(data)], capsys)
    assert attempts == []
    assert code == 0, f"A2: a clean render exits 0; got {code}\n{out}"
    assert made and all(isinstance(s, Services) for s in made), (
        f"A2: the client is PocketTtsClient(profile.services); it was constructed with {made!r}"
    )
    assert sorted(tts.texts()) == sorted(texts.values()), (
        f"A2/B6: each ending kind and still_working rendered once from CLIP_TEXT verbatim; spoken {tts.texts()!r}"
    )
    assert {voice for _t, voice in tts.calls} == {DEFAULT_VOICE}
    entries = manifest(data)
    status = api("status")
    for kind, body in pcm.items():
        assert "rendered" in outcome(out, kind)
        assert asset(data, kind).read_bytes() == body, (
            f"A2: {kind}'s asset is not the concatenated chunks"
        )
        got = entries.get(kind)
        assert got is not None, f"A2: the manifest has no {kind} entry: {sorted(entries)}"
        assert got["text"] == texts[kind]
        assert got["text_sha256"] == text_sha(texts[kind])
        import hashlib

        assert got["pcm_sha256"] == hashlib.sha256(body).hexdigest()
        assert got["bytes"] == len(body)
        assert got["duration_ms"] == len(body) // 48, (
            f"{kind}: {len(body)} bytes are {len(body) // 48} ms"
        )
        peak, rms = expected_db[kind]
        assert abs(got["peak_dbfs"] - peak) < 0.05, (
            f"{kind}: peak_dbfs {got['peak_dbfs']}, want {peak}"
        )
        assert abs(got["rms_dbfs"] - rms) < 0.05, f"{kind}: rms_dbfs {got['rms_dbfs']}, want {rms}"
        assert (got["sample_rate"], got["channels"], got["sample_format"]) == (24_000, 1, "s16le")
        assert got["tts"] == tts_pin(), f"A1: tts is the lock's pocket_tts pin; got {got['tts']!r}"
        assert str(got["git_commit"])[:7] == head_sha()[:7], f"git_commit {got['git_commit']!r}"
        assert str(got["rendered_at"]).strip()
        assert status(data, kind, texts[kind], tts_pin()) == "current"
    for kind in ("ack", "hmm"):
        assert not asset(data, kind).exists(), f"A4: the default set wrote {kind}"
        assert kind not in entries
    assert set(entries) == set(pcm)


def test_default_at_base_registry_writes_still_working_only_and_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 adversarial (the failure mode: "let me look" on every turn), re-scoped by B6: with only ack, hmm and
    still_working registered (BASE), the default set is ``still_working`` alone, so only its text is spoken, only its
    asset is written, ack and hmm are not, and the exit is 0."""
    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data)], capsys)
    assert code == 0, f"A2: a clean default render is not a failure; got {code}\n{out}"
    assert tts.texts() == [BASE_REGISTRY["still_working"]], (
        f"A2/A4/B6: the default spoke {tts.texts()!r}"
    )
    written = files_under(data) if data.exists() else []
    assert [f for f in written if f.endswith(".s16le") or ".tmp" in f] == [
        str(asset(data, "still_working").relative_to(data))
    ], f"A4/B6: wrote {written}"
    for kind in ("ack", "hmm"):
        assert not asset(data, kind).exists(), f"A4: the default set wrote {kind}"


def test_default_data_dir_is_memory_store_default_dir(
    jail: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: without ``--data-dir`` the assets go under ``memory.store.DEFAULT_DIR`` (here the jail's sentinel)."""
    install(script, monkeypatch, ScriptedTts())
    with no_sockets():
        code, out = run(script, ["--kinds", "ack"], capsys)
    assert code == 0, out
    assert asset(jail, "ack").is_file(), (
        f"A2: expected {asset(jail, 'ack')}; files {files_under(jail.parent)}"
    )


# --- --kinds --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ack", "hmm", "still_working"])
def test_kinds_names_a_filler_or_progress_kind_and_it_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """A2: ``--kinds ack`` (or hmm, still_working) renders exactly that kind, from its CLIP_TEXT, and nothing else."""
    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", kind], capsys)
    assert code == 0, out
    assert tts.texts() == [BASE_REGISTRY[kind]]
    assert "rendered" in outcome(out, kind)
    assert asset(data, kind).read_bytes() == square(1000)
    assert set(manifest(data)) == {kind}
    assert [f for f in files_under(data) if f.endswith(".s16le")] == [
        str(asset(data, kind).relative_to(data))
    ]


def test_kinds_bogus_is_refused_naming_the_registered_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: an unregistered kind is refused: non-zero, the refusal names it and every registered kind, nothing is
    spoken or written."""
    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "bogus"], capsys)
    assert code != 0, f"A2: --kinds bogus must be refused; exit {code}\n{out}"
    assert "bogus" in out
    for kind in BASE_REGISTRY:
        assert kind in out, (
            f"A2: the refusal names the registered kinds; {kind} missing from\n{out}"
        )
    assert tts.calls == []
    assert not data.exists() or [f for f in files_under(data) if f != "clips/manifest.json"] == []


@pytest.mark.parametrize("half", ["in ClipKind only", "in CLIP_TEXT only"])
def test_a_kind_registered_by_half_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
    half: str,
) -> None:
    """A1/A2: "registered" is ClipKind **and** CLIP_TEXT. A kind in only one of them is refused, by name (the guard
    too wide reads one of the two)."""
    if half == "in ClipKind only":
        set_registry(monkeypatch, dict(BASE_REGISTRY), kinds=(*BASE_REGISTRY, "out_of_steps"))
    else:
        set_registry(
            monkeypatch,
            {**BASE_REGISTRY, "out_of_steps": ENDING_TEXTS["out_of_steps"]},
            kinds=tuple(BASE_REGISTRY),
        )
    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "out_of_steps"], capsys)
    assert code != 0, f"A2: out_of_steps {half} is not registered; exit {code}\n{out}"
    assert tts.calls == []
    assert not asset(data, "out_of_steps").exists()


# --- idempotence ----------------------------------------------------------------------------------


def test_a_current_kind_is_skipped_and_force_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: a second run speaks nothing and prints ``skipped (current)``, leaving the asset as it was; ``--force``
    speaks again and replaces it."""
    first, second = square(1000, high=8192), square(1500, high=4096)
    tts = ScriptedTts({"let me look": [Plan([first]), Plan([second])]})
    made = install(script, monkeypatch, tts)
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--kinds", "ack"]
    with no_sockets():
        code, out = run(script, argv, capsys)
        assert code == 0 and "rendered" in outcome(out, "ack"), out
        stamp = asset(data, "ack").stat().st_mtime_ns
        constructed = len(made)
        code, out = run(script, argv, capsys)
        assert code == 0, out
        assert "skipped (current)" in outcome(out, "ack")
        assert len(tts.calls) == 1, "A2: a current kind is not re-rendered without --force"
        assert len(made) == constructed, "A2: an all-current run constructs no TTS client"
        assert (
            asset(data, "ack").read_bytes() == first
            and asset(data, "ack").stat().st_mtime_ns == stamp
        )
        code, out = run(script, [*argv, "--force"], capsys)
    assert code == 0, out
    assert "rendered" in outcome(out, "ack")
    assert len(tts.calls) == 2
    assert asset(data, "ack").read_bytes() == second
    assert manifest(data)["ack"]["bytes"] == len(second)


def test_a_text_change_makes_the_kind_stale_and_it_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: idempotence is over ``current`` only. After CLIP_TEXT changes the asset is stale, and a run without
    ``--force`` renders the new text."""
    from local_astra.conductor import session

    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--kinds", "hmm"]
    with no_sockets():
        assert run(script, argv, capsys)[0] == 0
        monkeypatch.setitem(session.CLIP_TEXT, "hmm", "hmm…")
        code, out = run(script, argv, capsys)
    assert code == 0, out
    assert "rendered" in outcome(out, "hmm")
    assert tts.texts() == ["hmm", "hmm…"]
    assert manifest(data)["hmm"]["text_sha256"] == text_sha("hmm…")


def test_rendering_one_kind_keeps_the_other_kinds_manifest_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: the manifest is merged, not replaced. After ``--kinds ack`` then ``--kinds hmm``, ack's entry is still
    there, unchanged, and ack is still ``current`` (a rewrite holding only this run's kinds would make it stale)."""
    install(script, monkeypatch, ScriptedTts())
    data = tmp_path / "data"
    with no_sockets():
        assert run(script, ["--data-dir", str(data), "--kinds", "ack"], capsys)[0] == 0
        ack_entry = manifest(data)["ack"]
        code, out = run(script, ["--data-dir", str(data), "--kinds", "hmm"], capsys)
    assert code == 0, out
    assert set(manifest(data)) == {"ack", "hmm"}, (
        f"A2: the manifest lost a kind: {sorted(manifest(data))}"
    )
    assert manifest(data)["ack"] == ack_entry
    status = api("status")
    assert status(data, "ack", BASE_REGISTRY["ack"], tts_pin()) == "current"
    assert status(data, "hmm", BASE_REGISTRY["hmm"], tts_pin()) == "current"


#: Manifests that cannot be read as a ``ClipManifest``: not JSON, and JSON of the wrong shape.
BAD_MANIFESTS = {
    "bad JSON": '{"clips": {"ack": ',
    "schema-invalid": '{"clips": {"ack": {"text": 1}}}',
}


@pytest.mark.parametrize("which", sorted(BAD_MANIFESTS))
def test_an_unreadable_manifest_is_recovered_by_rerendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
    which: str,
) -> None:
    """A1/A2: an asset beside an unreadable manifest is ``stale``, so a run without ``--force`` re-renders it, with no
    traceback, and leaves a valid manifest in which the kind is ``current``."""
    data = tmp_path / "data"
    asset(data, "ack").parent.mkdir(parents=True)
    asset(data, "ack").write_bytes(square(1000))
    (data / "clips" / "manifest.json").write_text(BAD_MANIFESTS[which])
    tts = ScriptedTts()
    install(script, monkeypatch, tts)
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "ack"], capsys)
    assert code == 0, f"A2: a {which} manifest is recoverable; exit {code}\n{out}"
    assert "Traceback" not in out, out
    assert "rendered" in outcome(out, "ack")
    assert tts.texts() == [BASE_REGISTRY["ack"]]
    assert api("status")(data, "ack", BASE_REGISTRY["ack"], tts_pin()) == "current"
    assert set(manifest(data)) == {"ack"}


# --- validation, one row per rule -----------------------------------------------------------------

BAD = {
    "empty body": Plan([]),
    "odd byte count": Plan([square(500) + b"\x01"]),
    "0.29 s, under 0.3 s": Plan([square(290)]),
    "6.01 s, over 6.0 s": Plan([square(6010)]),
    "all zeros": Plan([bytes(48_000)]),
    "a +32767 sample": Plan([with_sample(square(1000), 4321, 32767)]),
    "a -32768 sample": Plan([with_sample(square(1000), 4321, -32768)]),
    "TtsFormatError from speak": Plan(error=TtsFormatError("44100 Hz 2 ch"), at_speak=True),
    "TtsClientError from speak": Plan(
        error=TtsClientError("POST /tts answered 503", status=503), at_speak=True
    ),
    "TtsFormatError mid-stream": Plan([square(500)], error=TtsFormatError("no data chunk")),
    "TtsClientError mid-stream": Plan(
        [square(500)], error=TtsClientError("stream ended 12 bytes short")
    ),
}
GOOD_EDGES = {
    "exactly 0.3 s": square(300),
    "exactly 6.0 s": square(6000),
    "peak +32766": with_sample(square(1000), 4321, 32766),
    "peak -32766": with_sample(square(1000), 4321, -32766),
}


@pytest.mark.parametrize("rule", sorted(BAD))
def test_each_validation_rule_fails_the_kind_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
    rule: str,
) -> None:
    """A2: each rule alone fails the kind: a ``failed: <reason>`` line, a non-zero exit, no asset, no temporary file,
    no manifest entry."""
    install(script, monkeypatch, ScriptedTts({"hmm": BAD[rule]}))
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "hmm"], capsys)
    assert code != 0, f"A2: {rule} must fail the run; exit {code}\n{out}"
    assert "failed:" in outcome(out, "hmm")
    assert not asset(data, "hmm").exists(), f"A2: {rule} left an asset"
    leftovers = [
        f for f in (files_under(data) if data.exists() else []) if f != "clips/manifest.json"
    ]
    assert leftovers == [], f"A2: {rule} left {leftovers}"
    assert "hmm" not in manifest(data), f"A2: {rule} wrote a manifest entry"


@pytest.mark.parametrize("edge", sorted(GOOD_EDGES))
def test_the_validation_edges_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
    edge: str,
) -> None:
    """A2, the guard too wide: the boundaries are inside the rules (inclusive 0.3–6.0 s; 32766 is not full scale)."""
    body = GOOD_EDGES[edge]
    install(script, monkeypatch, ScriptedTts({"hmm": Plan([body])}))
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "hmm"], capsys)
    assert code == 0, f"A2: {edge} is valid; exit {code}\n{out}"
    assert asset(data, "hmm").read_bytes() == body


def test_a_failed_rerender_keeps_the_old_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: "a failure writes nothing for that kind" also holds over an existing asset: ``--force`` with a body that
    fails validation leaves the old bytes and the old manifest entry."""
    good = square(1000, high=8192)
    tts = ScriptedTts({"still working on it": [Plan([good]), Plan([bytes(48_000)])]})
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--kinds", "still_working"]
    with no_sockets():
        assert run(script, argv, capsys)[0] == 0
        before = manifest(data)["still_working"]
        code, out = run(script, [*argv, "--force"], capsys)
    assert code != 0, out
    assert "failed:" in outcome(out, "still_working")
    assert asset(data, "still_working").read_bytes() == good
    assert manifest(data)["still_working"] == before


def test_one_failing_kind_does_not_stop_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2: a failure is per kind. In the default set, lost_thread fails and the other two ending kinds (and, B6,
    still_working) render; the run exits non-zero and the manifest holds those three."""
    set_registry(monkeypatch, {**BASE_REGISTRY, **ENDING_TEXTS})
    tts = ScriptedTts(
        {ENDING_TEXTS["lost_thread"]: Plan(error=TtsClientError("connection reset"), at_speak=True)}
    )
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data)], capsys)
    assert code != 0, out
    assert "failed:" in outcome(out, "lost_thread")
    for kind in ("out_of_steps", "not_finished", "still_working"):
        assert "rendered" in outcome(out, kind)
        assert asset(data, kind).read_bytes() == square(1000)
    assert not asset(data, "lost_thread").exists()
    assert set(manifest(data)) == {"out_of_steps", "not_finished", "still_working"}


# --- atomicity ------------------------------------------------------------------------------------


def _spy_renames(
    monkeypatch: pytest.MonkeyPatch, *, fail_into: str | None = None
) -> list[tuple[str, str]]:
    """Record every ``os.replace``/``os.rename`` (``Path.replace``/``rename`` delegate to them); optionally raise for
    a destination whose name is ``fail_into``."""
    seen: list[tuple[str, str]] = []
    real_replace, real_rename = os.replace, os.rename

    def wrap(real: Any) -> Any:
        def spy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((Path(src).name, Path(dst).name))
            if fail_into is not None and Path(dst).name == fail_into:
                raise OSError(28, "No space left on device (injected)")
            return real(src, dst, *args, **kwargs)

        return spy

    monkeypatch.setattr(os, "replace", wrap(real_replace))
    monkeypatch.setattr(os, "rename", wrap(real_rename))
    return seen


def test_a_failing_rename_keeps_the_old_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 atomic: the asset is written aside and renamed; when the rename fails, the old asset is intact (a direct
    write over it would have destroyed it), the kind fails, and the run exits non-zero."""
    old, new = square(1000, high=8192), square(2000, high=2048)
    tts = ScriptedTts({"let me look": [Plan([old]), Plan([new])]})
    install(script, monkeypatch, tts)
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--kinds", "ack"]
    with no_sockets():
        assert run(script, argv, capsys)[0] == 0
        entry_before = manifest(data)["ack"]
        _spy_renames(monkeypatch, fail_into=asset(data, "ack").name)
        code, out = run(script, [*argv, "--force"], capsys)
    assert code != 0, f"A2: a failed rename is a failure; exit {code}\n{out}"
    assert asset(data, "ack").read_bytes() == old, "A2: the old asset must survive a failed rename"
    assert manifest(data)["ack"] == entry_before, (
        "A2: the manifest must not describe an asset that is not there"
    )
    leftovers = [f for f in files_under(data) if ".tmp" in f]
    assert leftovers == [], f"A2: a failed rename must not leave its temporary file: {leftovers}"


def test_assets_are_renamed_from_tmp_and_the_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 atomic: every asset arrives by a rename from a ``.tmp`` name, and the manifest is renamed into place after
    every asset."""
    set_registry(monkeypatch, {**BASE_REGISTRY, **ENDING_TEXTS})
    install(script, monkeypatch, ScriptedTts())
    data = tmp_path / "data"
    seen = _spy_renames(monkeypatch)
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data)], capsys)
    assert code == 0, out
    names = [dst for _src, dst in seen]
    assets = [asset(data, k).name for k in ENDING_TEXTS]
    for name in assets:
        into = [src for src, dst in seen if dst == name]
        assert into and all(".tmp" in src for src in into), (
            f"A2: {name} must be renamed from a .tmp; saw {seen}"
        )
    assert "manifest.json" in names, f"A2: the manifest is rewritten by a rename too; saw {seen}"
    last_manifest = max(i for i, n in enumerate(names) if n == "manifest.json")
    last_asset = max(i for i, n in enumerate(names) if n in assets)
    assert last_asset < last_manifest, (
        f"A2: the manifest is renamed after the assets; order {names}"
    )
    assert [f for f in files_under(data) if ".tmp" in f] == []


# --- the bytes ------------------------------------------------------------------------------------


def test_the_asset_is_the_concatenated_chunks_with_no_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    base_registry: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A2 output format: raw s16le, the ``audio()`` chunks concatenated, cut at odd places (mid-sample): no RIFF
    header, no resampling, gain, normalisation or trim (a pulse train with leading and trailing zeros is kept as is)."""
    body = samples([0] * 480) + pulse(1000, high=12000) + samples([0] * 480)
    cuts = [3, 4801, 4808, 30001]
    chunks = [body[a:b] for a, b in zip([0, *cuts], [*cuts, len(body)], strict=True)]
    assert b"".join(chunks) == body
    install(script, monkeypatch, ScriptedTts({"hmm": Plan(chunks)}))
    data = tmp_path / "data"
    with no_sockets():
        code, out = run(script, ["--data-dir", str(data), "--kinds", "hmm"], capsys)
    assert code == 0, out
    written = asset(data, "hmm").read_bytes()
    assert not written.startswith(b"RIFF")
    assert written == body, f"A2: {len(written)} bytes written for a {len(body)}-byte body"


# --- A5 and A4 ------------------------------------------------------------------------------------


def test_services_readme_documents_the_render_step() -> None:
    """A5: ``services/README.md`` names the command, says it needs ``tts.sh`` up, runs on CPU only, and must be re-run
    after a clip-text or pin change. Prose check, loose by design: it can only say the words are there."""
    text = (REPO / "services" / "README.md").read_text()
    assert "scripts/render_clips.py" in text, "A5: the render command is not in services/README.md"
    section = text[
        text.index("scripts/render_clips.py") - 1500 : text.index("scripts/render_clips.py") + 1500
    ]
    assert "tts.sh" in section, "A5: the render step must say tts.sh has to be up"
    assert "CPU" in section, "A5: the render step must say it runs on CPU only"
    assert re.search(r"re-?run", section, re.I), "A5: the render step must say when to re-run it"
    assert re.search(r"pin", section, re.I) and re.search(r"text", section, re.I)


#: T-0182 group A's BASE: main at f83f3b0a, the branch point of ``T-0182-a-clip-assets``. A literal.
BASE_A = "f83f3b0acf4e0594836e9627d0552979424cb26f"
#: The implementation tip: 1b07e5c2 (T-0182 A: clip registry, render script and the doctor clips row). Was BASE_A.
#: Whoever merges group A bumps it to the merged branch tip if later implementation commits land (tests.md).
TIP_A = "1b07e5c2c2c8a93951834d1ba2cd11296121ed1f"
#: A range known to change files outside group A's list (fe6a8fda: T-0174-d merged into main; cli.py, config.py).
POWER_BASE = "c440afb7bb544962c46ce653c30a38ebb9ab066b"
POWER_TIP = "fe6a8fdaa0656cbc7b0a2773b4d0b90d0b9b04d2"
#: Group A's owned files (docs/tasks/T-0182.md "Owned files", the A rows) plus ``tests/unit/_box_rig.py``, whose
#: ``REQUIRED_CHECKS``/``FACT_FIELDS`` must name the clips row at activation (reported: not yet in the owned list).
GROUP_A_FILES = frozenset(
    {
        "local_astra/reflex/clips.py",
        "scripts/render_clips.py",
        "local_astra/preflight.py",
        "services/README.md",
        "docs/tasks/T-0182.md",
        "tests/unit/test_t0182_clips_registry.py",
        "tests/unit/test_t0182_render_clips.py",
        "tests/contract/test_t0182_render_pocket_double.py",
        "tests/unit/test_t0182_doctor_clips.py",
        "tests/unit/_box_rig.py",
    }
)


def _outside_group_a(base: str, tip: str) -> list[str]:
    return [p for p in gr.changed(base, tip, ".") if p not in GROUP_A_FILES]


def test_group_a_changes_nothing_outside_its_files_over_the_closed_range() -> None:
    """A4: over ``BASE_A..TIP_A`` (two literals, never the working tree) every file outside group A's list is
    byte-identical; in particular ``session.py``, ``commands.py``, ``tts_client.py`` and ``audio_io.py`` are. Green at
    BASE (TIP_A starts there). Power: the same predicate over a range that edited ``cli.py`` reports it."""
    gr.require_commits(BASE_A, TIP_A, POWER_BASE, POWER_TIP)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:9]} is not an ancestor of {TIP_A[:9]}"
    moved = _outside_group_a(BASE_A, TIP_A)
    assert moved == [], (
        f"T-0182 A4: files outside group A changed in {BASE_A[:9]}..{TIP_A[:9]}: {moved}"
    )
    power = _outside_group_a(POWER_BASE, POWER_TIP)
    assert "local_astra/cli.py" in power, (
        f"power: the predicate must see cli.py change in the power range; {power}"
    )
