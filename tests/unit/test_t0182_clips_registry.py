"""T-0182 A1: ``local_astra/reflex/clips.py``, the clip registry (docs/tasks/T-0182.md, group A).

**The interface these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``clip_path(data_dir: Path | str, kind: str) -> Path``: the file ``Session._clip_pcm(kind)`` reads. Checked
  behaviourally: bytes placed at ``clip_path`` are what a real ``Session`` (built by ``test_session_loop.make_rig``)
  returns from ``_clip_pcm``.
- ``ENDING_KINDS``, ``PROGRESS_KINDS``, ``FILLER_KINDS``: set-likes equal to the literal sets of A1.
- ``default_enabled(registered) -> set-like``: group A's rule, ``ENDING_KINDS & registered``.
- ``ClipManifest`` (Pydantic v2): the model of ``<data_dir>/clips/manifest.json``, whose JSON is
  ``{"clips": {<kind>: <entry>}}``; an entry has the thirteen A1 fields. Reading: ``sample_rate``, ``channels`` and
  ``sample_format`` are fixed at 24000, 1 and ``"s16le"`` (any other value is a ValidationError), and every field is
  required.
- ``status(data_dir, kind, text, tts_pin) -> "current" | "missing" | "stale"``:
  ``missing`` when there is no asset at ``clip_path``; otherwise ``stale`` when the manifest is absent, has no entry for
  the kind, records a different ``text_sha256`` or a different ``tts``; otherwise ``current``. Reading:
  ``text_sha256`` is ``hashlib.sha256(text.encode("utf-8")).hexdigest()``. A ``rendered_at`` or ``git_commit``
  different from today's is *not* stale (the guard-too-wide rows).

**The registry seam the render and doctor rows use** (``set_registry`` below). "Registered" is
``get_args(ClipKind)`` intersected with ``CLIP_TEXT``'s keys (A1), computed at call time. The rows that need ending
kinds registered (BASE registers only ``ack``, ``hmm``, ``still_working``) mutate ``session.CLIP_TEXT`` in place and
rebind ``ClipKind`` in ``contracts.commands`` and in every loaded module that imported it by name. An implementation
that freezes the registered set at import time cannot be driven this way; that is pinned, and reported.

Activated for ``/implement T-0182`` group A (the rows were pending at a23a7f1b): until ``local_astra/reflex/clips.py``
exists, the import helper turns its absence into a named ``AssertionError``.

Regime: CPU, pure apart from reads and writes under ``tmp_path``; no service, no socket, n = 1, no seed.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
import json
import sys
import tomllib
import typing
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.contracts import commands as C

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "services" / "versions.lock"

ENDING = {"out_of_steps", "not_finished", "lost_thread"}
PROGRESS = {"still_working"}
FILLER = {"ack", "hmm"}
#: What BASE (f83f3b0a) registers: ``ClipKind`` and ``CLIP_TEXT`` both hold exactly these.
BASE_REGISTRY = {"ack": "let me look", "hmm": "hmm", "still_working": "still working on it"}
#: Ending-clip texts for the rows that register ending kinds (the real texts are T-0178/T-0179's to write).
ENDING_TEXTS = {
    "out_of_steps": "I ran out of steps before finishing — here’s where I got to.",
    "not_finished": "I didn’t finish that one.",
    "lost_thread": "Sorry, I lost the thread there.",
}
MANIFEST_FIELDS = {
    "text",
    "text_sha256",
    "pcm_sha256",
    "bytes",
    "duration_ms",
    "peak_dbfs",
    "rms_dbfs",
    "sample_rate",
    "channels",
    "sample_format",
    "tts",
    "rendered_at",
    "git_commit",
}


# --- helpers shared with the render and doctor modules ----------------------------------------


def clips_mod() -> ModuleType:
    try:
        return importlib.import_module("local_astra.reflex.clips")
    except ModuleNotFoundError as exc:
        if exc.name == "local_astra.reflex.clips":
            raise AssertionError(
                "T-0182 A1: local_astra/reflex/clips.py does not exist yet "
                "(the interface is in tests/unit/test_t0182_clips_registry.py's docstring)"
            ) from None
        raise


def api(name: str) -> Any:
    mod = clips_mod()
    assert hasattr(mod, name), f"T-0182 A1: local_astra.reflex.clips has no {name!r}"
    return getattr(mod, name)


def tts_pin() -> str:
    """The ``pocket_tts`` pin, read from ``services/versions.lock`` with ``tomllib`` (not through the code)."""
    with LOCK.open("rb") as handle:
        return str(tomllib.load(handle)["speech"]["pocket_tts"])


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entry(text: str, pcm: bytes, *, tts: str | None = None, **over: Any) -> dict[str, Any]:
    """One manifest entry for ``pcm`` (s16le mono 24 kHz), computed by hand. dBFS values are placeholders: no row
    here reads them."""
    out: dict[str, Any] = {
        "text": text,
        "text_sha256": text_sha(text),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "bytes": len(pcm),
        "duration_ms": len(pcm) // 2 * 1000 // 24_000,
        "peak_dbfs": -6.0,
        "rms_dbfs": -12.0,
        "sample_rate": 24_000,
        "channels": 1,
        "sample_format": "s16le",
        "tts": tts if tts is not None else tts_pin(),
        "rendered_at": "2026-10-28T12:00:00+00:00",
        "git_commit": "f83f3b0acf4e0594836e9627d0552979424cb26f",
    }
    out.update(over)
    return out


def write_manifest(data_dir: Path, entries: dict[str, dict[str, Any]]) -> Path:
    path = data_dir / "clips" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"clips": entries}))
    return path


def write_asset(data_dir: Path, kind: str, pcm: bytes) -> Path:
    """Placed where ``clip_path`` says. ``status``'s rows depend on ``clip_path`` being right, which the
    ``_clip_pcm`` row checks independently."""
    path = Path(api("clip_path")(data_dir, kind))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm)
    return path


def square(ms: int = 1000, *, high: int = 8192) -> bytes:
    """``±high`` every 24 samples (500 Hz): never zero, never full scale; peak = rms = 20·log10(high/32768)."""
    n = ms * 24
    return b"".join(
        (high if (i // 24) % 2 == 0 else -high).to_bytes(2, "little", signed=True) for i in range(n)
    )


def set_registry(
    monkeypatch: pytest.MonkeyPatch, texts: dict[str, str], *, kinds: Any = None
) -> None:
    """Make "registered" read ``kinds`` (default: ``texts``' keys) as ``ClipKind`` and ``texts`` as ``CLIP_TEXT``.

    ``CLIP_TEXT`` is mutated in place, so every module holding a reference sees it; ``ClipKind`` is a ``Literal``
    and is rebound wherever a loaded module holds the original object. Undone by ``monkeypatch``.

    ``CLIP_TEXT`` is restored whole, in its original key order (T-0182 D): per-key ``delitem`` undo re-appends the
    keys in reverse, which the order pin in ``test_t0182_differential`` catches. The restore is one entry on
    ``monkeypatch``'s own undo stack, so it unwinds in turn with the rest (also inside ``monkeypatch.context()``).
    """
    from local_astra.conductor import session

    monkeypatch.setitem(_WholeMapping(session.CLIP_TEXT), "whole", dict(texts))  # type: ignore[arg-type]
    names = tuple(kinds if kinds is not None else texts)
    original = C.ClipKind
    replacement = typing.Literal[names]  # type: ignore[valid-type]
    for module in ours():
        if module.__dict__.get("ClipKind") is original:
            monkeypatch.setattr(module, "ClipKind", replacement)


class _WholeMapping:
    """A one-key view of a dict for ``monkeypatch.setitem``: key ``"whole"`` reads a snapshot of the dict, and
    setting it replaces the dict's contents in place, in the given order. ``monkeypatch.setitem`` reads the
    snapshot before writing the replacement, and its undo writes the snapshot back: the dict as it was, keys in
    their old order, the same object (every module's reference stays valid)."""

    def __init__(self, target: dict[str, str]) -> None:
        self._target = target

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self._target) if key == "whole" else default

    def __setitem__(self, key: str, value: dict[str, str]) -> None:
        assert key == "whole"
        self._target.clear()
        self._target.update(value)


def ours() -> list[ModuleType]:
    """This repo's loaded modules (``local_astra.*``, ``tests.*``, and scripts loaded by path as ``_t0182_*``). Read
    through ``__dict__`` so no module ``__getattr__`` runs."""
    return [
        module
        for name, module in list(sys.modules.items())
        if isinstance(module, ModuleType) and name.startswith(("local_astra", "tests", "_t0182_"))
    ]


# --- clip_path ----------------------------------------------------------------------------------


def _session(tmp_path: Path) -> Any:
    from tests.unit.test_session_loop import make_rig

    return make_rig(tmp_path).session


@pytest.mark.parametrize("kind", ["ack", "hmm", "still_working", "out_of_steps", "lost_thread"])
def test_clip_path_is_the_file_session_clip_pcm_reads(tmp_path: Path, kind: str) -> None:
    """A1: bytes written at ``clip_path(data_dir, kind)`` are exactly what ``Session._clip_pcm(kind)`` returns, for a
    real session whose ``data_dir`` is ``tmp_path``; and the path depends on the kind (another kind reads nothing)."""
    clip_path = api("clip_path")
    session = _session(tmp_path)
    marker = f"T-0182 {kind} ".encode() * 7
    path = Path(clip_path(tmp_path, kind))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(marker)
    assert session._clip_pcm(kind) == marker, (
        f"A1: clip_path({tmp_path}, {kind!r}) = {path}, but Session._clip_pcm({kind!r}) did not read it"
    )
    other = "hmm" if kind != "hmm" else "ack"
    assert session._clip_pcm(other) == b"", (
        f"A1: clip_path must depend on the kind; a file for {kind!r} was read as {other!r}"
    )
    assert Path(clip_path(str(tmp_path), kind)) == path, "A1: clip_path accepts a str data_dir"


# --- the kind sets and default_enabled --------------------------------------------------------


def test_kind_sets_are_the_three_literal_sets() -> None:
    """A1: the literal sets, pairwise disjoint. T-0192 B1 re-pin (before implementation): ``ENDING_KINDS`` is T-0182's
    three plus ``at_checkout`` (played by default, doctor reports it), and nothing else; ``ENDING`` stays T-0182's
    literal, which the 128-subset rows below still use."""
    assert set(api("ENDING_KINDS")) == ENDING | {"at_checkout"}
    assert set(api("PROGRESS_KINDS")) == PROGRESS
    assert set(api("FILLER_KINDS")) == FILLER


def test_default_enabled_at_base_registry_is_still_working_only() -> None:
    """A1 adversarial, re-scoped by B6 (group B adds ``PROGRESS_KINDS`` to the default): BASE registers ack, hmm and
    still_working only, so only ``still_working`` is default-enabled (a rule that enabled every registered kind would
    say "let me look" on every turn once assets exist)."""
    assert set(api("default_enabled")(set(BASE_REGISTRY))) == {"still_working"}
    assert set(api("default_enabled")(frozenset(BASE_REGISTRY))) == {"still_working"}


def test_default_enabled_is_ending_kinds_within_registered_for_every_subset() -> None:
    """A1, re-scoped by B6: ``(ENDING_KINDS ∪ PROGRESS_KINDS) ∩ registered`` over every subset of the six kinds plus
    an unknown one (128 inputs), the switch left at its default. The expected value is read off the hand-written
    literals, not the module's constants."""
    default_enabled = api("default_enabled")
    universe = sorted(ENDING | PROGRESS | FILLER | {"bogus"})
    for r in range(len(universe) + 1):
        for subset in itertools.combinations(universe, r):
            expected = {
                k
                for k in subset
                if k in ("out_of_steps", "not_finished", "lost_thread", "still_working")
            }
            got = set(default_enabled(set(subset)))
            assert got == expected, (
                f"default_enabled({sorted(subset)}) = {sorted(got)}, want {sorted(expected)}"
            )


@pytest.mark.parametrize("filler_clips", [False, True])
def test_b6_default_enabled_follows_the_filler_clips_switch(filler_clips: bool) -> None:
    """B6: ``default_enabled(registered, filler_clips=...)`` is ``(ENDING ∪ PROGRESS ∪ (FILLER if filler_clips)) ∩
    registered`` over every subset of the six kinds plus an unknown one (128 inputs per value), with the switch passed
    by keyword; the expected sets are read off the hand-written literals. Adversarial: ``ack`` and ``hmm`` appear only
    with the switch on, ``still_working`` with it off too."""
    default_enabled = api("default_enabled")
    assert "filler_clips" in inspect.signature(default_enabled).parameters, (
        "T-0182 B6: default_enabled(registered, filler_clips=False)"
    )
    on = ENDING | PROGRESS | (FILLER if filler_clips else set())
    universe = sorted(ENDING | PROGRESS | FILLER | {"bogus"})
    for r in range(len(universe) + 1):
        for subset in itertools.combinations(universe, r):
            expected = {k for k in subset if k in on}
            got = set(default_enabled(set(subset), filler_clips=filler_clips))
            assert got == expected, (
                f"T-0182 B6: default_enabled({sorted(subset)}, filler_clips={filler_clips}) = {sorted(got)}, "
                f"want {sorted(expected)}"
            )
    every = set(universe)
    assert set(default_enabled(every, filler_clips=filler_clips)) & FILLER == (
        FILLER if filler_clips else set()
    )
    assert "still_working" in set(default_enabled(every, filler_clips=filler_clips))
    assert set(default_enabled(every)) == ENDING | PROGRESS, (
        "T-0182 B6: filler_clips defaults to False"
    )


# --- the manifest ---------------------------------------------------------------------------------


def test_manifest_round_trips_every_field(tmp_path: Path) -> None:
    """A1: ``ClipManifest`` validates the file's JSON, dumps it back, and the reload is equal, field for field."""
    manifest_cls = api("ClipManifest")
    pcm = square(500)
    raw = {
        "clips": {
            "ack": entry("let me look", pcm),
            "lost_thread": entry(ENDING_TEXTS["lost_thread"], pcm),
        }
    }
    model = manifest_cls.model_validate(raw)
    path = tmp_path / "manifest.json"
    path.write_text(model.model_dump_json())
    again = manifest_cls.model_validate_json(path.read_text())
    assert again == model, "A1: the manifest does not survive a JSON round trip"
    dumped = json.loads(path.read_text())
    assert set(dumped["clips"]) == {"ack", "lost_thread"}
    for kind, fields in dumped["clips"].items():
        missing = MANIFEST_FIELDS - set(fields)
        assert not missing, f"A1: the {kind} entry lost {sorted(missing)}"
        for name in (
            "text",
            "text_sha256",
            "pcm_sha256",
            "bytes",
            "sample_rate",
            "channels",
            "sample_format",
            "tts",
        ):
            assert fields[name] == raw["clips"][kind][name], (
                f"A1: {kind}.{name} changed in the round trip"
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_rate", 44_100), ("channels", 2), ("sample_format", "f32le")],
)
def test_manifest_refuses_another_audio_format(field: str, value: Any) -> None:
    """A1 (reading): an asset is 24 kHz mono s16le, so an entry recording anything else does not validate."""
    import pydantic

    manifest_cls = api("ClipManifest")
    bad = entry("hmm", square(400), **{field: value})
    try:
        manifest_cls.model_validate({"clips": {"hmm": bad}})
    except pydantic.ValidationError:
        return
    raise AssertionError(f"A1: an entry with {field}={value!r} validated")


@pytest.mark.parametrize("field", sorted(MANIFEST_FIELDS))
def test_manifest_requires_every_field(field: str) -> None:
    """A1: every one of the thirteen fields is held, so none may be missing."""
    import pydantic

    manifest_cls = api("ClipManifest")
    bad = entry("hmm", square(400))
    del bad[field]
    try:
        manifest_cls.model_validate({"clips": {"hmm": bad}})
    except pydantic.ValidationError:
        return
    raise AssertionError(f"A1: an entry without {field!r} validated")


# --- status ---------------------------------------------------------------------------------------


def test_status_missing_without_an_asset(tmp_path: Path) -> None:
    """A1: no file at clip_path is ``missing``, with no clips dir at all and with a manifest entry but no asset."""
    status = api("status")
    pin = tts_pin()
    assert status(tmp_path, "lost_thread", ENDING_TEXTS["lost_thread"], pin) == "missing"
    write_manifest(tmp_path, {"lost_thread": entry(ENDING_TEXTS["lost_thread"], square())})
    assert status(tmp_path, "lost_thread", ENDING_TEXTS["lost_thread"], pin) == "missing", (
        "A1: a manifest entry without its asset is still missing audio"
    )


def test_status_current_when_text_and_pin_match(tmp_path: Path) -> None:
    """A1: asset present and an entry with the same text sha and pin is ``current``."""
    status = api("status")
    pcm = square()
    write_asset(tmp_path, "out_of_steps", pcm)
    write_manifest(tmp_path, {"out_of_steps": entry(ENDING_TEXTS["out_of_steps"], pcm)})
    assert status(tmp_path, "out_of_steps", ENDING_TEXTS["out_of_steps"], tts_pin()) == "current"
    assert (
        status(str(tmp_path), "out_of_steps", ENDING_TEXTS["out_of_steps"], tts_pin()) == "current"
    )


def test_status_stale_without_a_manifest_or_an_entry(tmp_path: Path) -> None:
    """A1: an asset with no manifest is ``stale``; so is one whose manifest has entries for other kinds only."""
    status = api("status")
    pcm = square()
    write_asset(tmp_path, "not_finished", pcm)
    assert status(tmp_path, "not_finished", ENDING_TEXTS["not_finished"], tts_pin()) == "stale"
    write_manifest(tmp_path, {"lost_thread": entry(ENDING_TEXTS["lost_thread"], pcm)})
    assert status(tmp_path, "not_finished", ENDING_TEXTS["not_finished"], tts_pin()) == "stale"


def test_status_stale_on_a_text_change(tmp_path: Path) -> None:
    """A1: the clip's text changed since rendering (the entry's sha is of the old text): ``stale``. A one-character
    change is enough, and the entry's ``text`` field alone is not what decides (its sha is)."""
    status = api("status")
    pcm = square()
    old = "Sorry, I lost the thread there."
    write_asset(tmp_path, "lost_thread", pcm)
    write_manifest(tmp_path, {"lost_thread": entry(old, pcm)})
    assert status(tmp_path, "lost_thread", old, tts_pin()) == "current"
    assert status(tmp_path, "lost_thread", old.replace(".", "!"), tts_pin()) == "stale"
    write_manifest(tmp_path, {"lost_thread": entry(old, pcm, text_sha256=text_sha(old + " "))})
    assert status(tmp_path, "lost_thread", old, tts_pin()) == "stale", (
        "A1: stale is decided by text_sha256; an entry whose sha is of another text is stale"
    )


def test_status_stale_on_a_pin_change(tmp_path: Path) -> None:
    """A1: rendered under another Pocket TTS pin: ``stale``."""
    status = api("status")
    pcm = square()
    write_asset(tmp_path, "out_of_steps", pcm)
    write_manifest(
        tmp_path, {"out_of_steps": entry(ENDING_TEXTS["out_of_steps"], pcm, tts="3.0.9")}
    )
    assert status(tmp_path, "out_of_steps", ENDING_TEXTS["out_of_steps"], tts_pin()) == "stale"
    assert status(tmp_path, "out_of_steps", ENDING_TEXTS["out_of_steps"], "3.0.9") == "current"


@pytest.mark.parametrize(
    "content",
    ['{"clips": {"lost_thread": ', '{"clips": {"lost_thread": {"text": 1}}}', '{"clips": []}', ""],
    ids=["bad JSON", "schema-invalid entry", "schema-invalid clips", "empty file"],
)
def test_status_stale_when_the_manifest_is_unreadable(tmp_path: Path, content: str) -> None:
    """A1: an asset whose manifest cannot be read is ``stale`` (re-renderable), never an exception; without the asset
    it is still ``missing``."""
    status = api("status")
    write_manifest(tmp_path, {}).write_text(content)
    assert status(tmp_path, "lost_thread", ENDING_TEXTS["lost_thread"], tts_pin()) == "missing"
    write_asset(tmp_path, "lost_thread", square())
    assert status(tmp_path, "lost_thread", ENDING_TEXTS["lost_thread"], tts_pin()) == "stale"


def test_status_is_not_stale_for_other_reasons(tmp_path: Path) -> None:
    """A1, the guard too wide: ``rendered_at`` and ``git_commit`` are provenance, not staleness. An entry rendered at
    another commit, at another time, is still ``current``."""
    status = api("status")
    pcm = square()
    write_asset(tmp_path, "lost_thread", pcm)
    write_manifest(
        tmp_path,
        {
            "lost_thread": entry(
                ENDING_TEXTS["lost_thread"],
                pcm,
                rendered_at="2025-01-01T00:00:00+00:00",
                git_commit="0" * 40,
            )
        },
    )
    assert status(tmp_path, "lost_thread", ENDING_TEXTS["lost_thread"], tts_pin()) == "current"
