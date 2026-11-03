"""T-0157: release prep for publishing as DiffusionGearvis.

Every row is derived from ``docs/tasks/T-0157.md``'s acceptance criteria, never from an implementation:

- criterion 1: ``LICENSE`` is the canonical Apache-2.0 text (sha256 and size literal from the task file);
- criterion 2: ``NOTICE`` heads with the release identity and covers every item the task lists (the A2UI
  renderer bundle item and its package rows were retired by T-0164 D2 with the bundle; NOTICE now names
  no path that does not exist);
- criterion 3: the two vendored ``PROVENANCE.md`` files and ``docs/07`` carry the licence ruling;
- criteria 4-5: no owner identifier in any tracked blob (text and binary, case-insensitive), placeholders in place;
- criterion 6: the pipewire dumps regenerate byte for byte from the scrubbed base, the MANIFEST table equals the real
  hashes, the memory fixture records the store path unexpanded, the two log builders read repo-relative ``logs/``,
  verbatim fixtures carry the "identifiers replaced" MANIFEST line;
- criterion 7: ``coldboot_watch.sh`` changes directory relative to itself, the scrubbed locked scripts keep their mode;
- criteria 8-10: wording, README body, ``CONTRIBUTING.md`` and ``SECURITY.md`` present (release re-pin: James
  2026-11-01), no ``.github/``;
- criterion 11: the two history-dependent rows skip (never fail, never pass) in a ``git init`` copy with no history.

The scanner keeps no owner identifier in source (ruling O-R1 F1): the owner's username, hostname and machine-id are
read, one per line, from the untracked file named by ``ASTRA_SCRUB_NEEDLES``; without it the tracked-tree row skips
with a stated reason. Only the generic patterns (``/home/<name>`` except ``/home/user``, and the home-relative ``demos/`` clone path) live here, and
they are assembled so this file never matches its own scan.

Regime: CPU, no network, no GPU; the history-free row spawns one pytest subprocess (< 10 s).
"""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- criterion 4 (O-R1 F1)

_DEMOS = "~/" + "demos/"
_HM = "/ho" + "me/"  # adversarial home paths are assembled so this file never matches its own scan
#: Environment variable naming the untracked needles file (on the 5090 box: ``<git common dir>/local_astra/
#: scrub-needles.txt``, written by the orchestrator). One identifier per line, matched case-insensitively.
NEEDLES_ENV = "ASTRA_SCRUB_NEEDLES"
_HOME = re.compile(rb"/home/([a-z][a-z0-9_.-]*)", re.IGNORECASE)
#: ``/home/user`` is the task's placeholder (criterion 5). ``/home/u`` (tests/contract/test_protocol_codec.py,
#: tests/hud/*) and ``/home/someone`` (tests/unit/_box_rig.py, test_doctor_browser.py, test_protect_paths_hook.py) are
#: pre-existing generic placeholders in files T-0157 does not own; neither names the owner. Not covered by ruling O-I
#: (which names only the two paths below): reported to the orchestrator.
_HOME_ALLOWED = {b"user", b"u", b"someone"}
#: Ruling O-I (3), kept by O-R1: the vendored, hash-pinned file carries upstream's home path (user "ubuntu").
#: (The task file's allowlist was dropped by O-R1: it is now written with placeholders.)
_PATH_ALLOWED = {"local_astra/decoding/_vendor/dgfa_compile/bfcl_data.py"}


def read_needles(path: Path) -> list[str]:
    """Non-empty, stripped lines of the needles file (``#`` starts a comment line)."""
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def scan(blobs: Iterable[tuple[str, bytes]], needles: Iterable[str]) -> list[tuple[str, str]]:
    """(path, matched text) for every criterion-4 identifier in ``blobs``: the given needles and the home-relative demos path anywhere,
    case-insensitively, plus any ``/home/<name>`` outside the placeholders; bytes, so binary files are covered."""
    literal = [s.encode() for s in needles] + [_DEMOS.encode()]
    needle_re = re.compile(b"|".join(re.escape(s) for s in literal), re.IGNORECASE)
    hits: list[tuple[str, str]] = []
    for path, data in blobs:
        if path in _PATH_ALLOWED:
            continue
        for m in needle_re.finditer(data):
            hits.append((path, m.group(0).decode("latin-1")))
        for m in _HOME.finditer(data):
            if m.group(1).lower() not in _HOME_ALLOWED:
                hits.append((path, m.group(0).decode("latin-1")))
    return hits


def _tracked_blobs() -> Iterable[tuple[str, bytes]]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode()
        path = REPO / rel
        if path.is_file():
            yield rel, path.read_bytes()


#: Synthetic needles for the adversarial row (never the owner's): a username, a hostname, a 32-hex machine-id.
_SYNTH_USER = "zqowner9"
_SYNTH_HOST = "zqowner9-Synth-Box"
_SYNTH_MID = "0123456789abcdef0123456789abcdef"


def _optional_needles() -> list[str]:
    """The configured needles if any, else none: fixture rows still check the generic patterns and placeholders."""
    import os

    name = os.environ.get(NEEDLES_ENV)
    return read_needles(Path(name)) if name and Path(name).is_file() else []


def _needles_or_skip() -> list[str]:
    import os

    name = os.environ.get(NEEDLES_ENV)
    if not name:
        pytest.skip(
            f"{NEEDLES_ENV} is unset: no owner-identifier needles file, the owner-specific scan cannot run"
        )
    path = Path(name)
    if not path.is_file():
        pytest.skip(f"{NEEDLES_ENV}={name} does not exist: the owner-specific scan cannot run")
    needles = read_needles(path)
    assert needles, f"{name} holds no needles"
    return needles


def test_scanner_catches_mixed_case_and_binary_identifiers(tmp_path: Path) -> None:
    """Adversarial (criterion 4), synthetic needles read through the same file path as the real run: a mixed-case
    hostname, the pipewire core name, a binary blob, a home path and the machine-id in upper case are all caught; the
    placeholder ``/home/user``, the github handle and the vendored path are not."""
    needles_file = tmp_path / "needles.txt"
    needles_file.write_text(f"# synthetic\n{_SYNTH_HOST}\n\n{_SYNTH_USER}\n{_SYNTH_MID}\n")
    needles = read_needles(needles_file)
    assert needles == [_SYNTH_HOST, _SYNTH_USER, _SYNTH_MID]
    hits = scan(
        [
            ("a.txt", ("host " + _SYNTH_HOST.lower()).encode()),
            ("b.json", ('"core.name": "pipewire-' + _SYNTH_USER.upper() + '-1"').encode()),
            ("c.md", b"/home/user/demos/local_astra and github james-assiene"),
            ("d.pt", b"\x80\x02" + _SYNTH_USER.capitalize().encode() + b"\xff\x00"),
            ("e.sh", ("cd " + _HM + "alice/x").encode()),
            ("f.md", ("clone at ~/" + "Demos/diffgemma-fa").encode()),
            ("g.json", _SYNTH_MID.upper().encode()),
            ("h.txt", (_HM + "username/x").encode()),
            (
                "i.py",
                (_HM + "ubuntu/x").encode(),
            ),  # O-I allows this only at the vendored path below
            (
                "local_astra/decoding/_vendor/dgfa_compile/bfcl_data.py",
                (_HM + "ubuntu/diffgemma_fa/x").encode(),
            ),
        ],
        needles,
    )
    assert sorted({p for p, _ in hits}) == [
        "a.txt",
        "b.json",
        "d.pt",
        "e.sh",
        "f.md",
        "g.json",
        "h.txt",
        "i.py",
    ], hits


def test_no_generic_identifier_paths_tracked() -> None:
    """Criterion 4, generic half (review L1): no ``/home/<name>`` outside the placeholders and no home-relative demos
    clone path in any tracked blob. Needs no needles file, so it runs on every run, the fast suite included."""
    _assert_no_hits(scan(_tracked_blobs(), []))


def test_no_owner_identifiers_tracked() -> None:
    """Criterion 4, owner half: zero hits for the owner's username, hostname and machine-id over every tracked blob
    (working-tree bytes of ``git ls-files``). Skips, naming the reason, when the needles file is not configured (a
    public clone has none); the release recipe sets it. The generic patterns are checked here too, and always by
    ``test_no_generic_identifier_paths_tracked``."""
    _assert_no_hits(scan(_tracked_blobs(), _needles_or_skip()))


def _assert_no_hits(hits: list[tuple[str, str]]) -> None:
    by_file: dict[str, int] = {}
    for path, _ in hits:
        by_file[path] = by_file.get(path, 0) + 1
    assert not hits, (
        f"{len(hits)} identifier hits in {len(by_file)} tracked files: {sorted(by_file.items())[:40]}"
    )


# --------------------------------------------------------------------------- criterion 1


def test_license_is_canonical_apache() -> None:
    assert (REPO / "LICENSE").is_file(), "no LICENSE at the repository root"
    data = (REPO / "LICENSE").read_bytes()
    assert len(data) == 11358, len(data)
    assert hashlib.sha256(data).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_pyproject_licence_field_unchanged() -> None:
    """Criterion 1 negative: ``license = { text = "Apache-2.0" }`` stays."""
    assert 'license = { text = "Apache-2.0" }' in (REPO / "pyproject.toml").read_text()


# --------------------------------------------------------------------------- criterion 2


def _notice() -> str:
    assert (REPO / "NOTICE").is_file(), "no NOTICE at the repository root"
    return (REPO / "NOTICE").read_text()


#: T-0164 D2 retired the bundled-package rows (``bundled_packages``, ``missing_packages``,
#: ``test_bundled_package_list_is_the_task_list``, ``test_missing_packages_detects_a_dropped_name``,
#: ``test_notice_names_every_bundled_package``) with ``hud/vendor/MANIFEST.md`` and NOTICE §2, which
#: held those packages and their MIT/ISC/BSD-3-Clause texts. What replaces them is
#: ``test_notice_names_no_missing_path`` and the renumbered-items row below.


def test_notice_heading_and_sections() -> None:
    """Criterion 2: the first line is the release identity, an Apache-2.0 line follows, and every listed item has
    its in-repo path (or name) in the file."""
    notice = _notice()
    lines = notice.splitlines()
    assert lines[0].strip() == "DiffusionGearvis, Copyright 2026 James Assiene", lines[:2]
    assert any("Apache" in ln for ln in lines[1:6]), lines[:6]
    # T-0164 D2: the bundle's path and the @a2ui 0.11.0 version (named by §2 only) left with §2.
    for needle in (
        "diffgemma-fa",
        "local_astra/decoding/_vendor/dgfa_compile",
        "local_astra/eval/cd/_vendor",
        "local_astra/ui/schemas/a2ui_v0_9_1",  # O-I (4)
        "v0.9.1",
        "vLLM",
        "tests/cd/oracle/vllm_gemma4_args.py",
        "tests/fixtures/vllm_v0_29_0",
        "tests/fixtures/cd/bfcl",
        "Gorilla",
        "Pocket TTS",
        "CC-BY-4.0",
        "tests/fixtures/audio/pocket_tts_sentence.wav",
        "Kyutai STT",
        "Unmute",
        "Gemma",
        "NVFP4",
        "tests/fixtures/cd/model/generation_config.json",
    ):
        assert needle in notice, needle
    assert "Copyright James Assiene" in notice or "Copyright 2026 James Assiene" in notice


#: The top-level directories a NOTICE path may start with (hand-listed from the repository layout,
#: not from NOTICE): a token ``<root>/...`` is a repository path, ``package/LICENSE`` (a path inside
#: an npm tarball) or ``creativecommons.org/...`` is not.
_REPO_ROOTS = ("docs", "hud", "local_astra", "plugin", "scripts", "services", "tests")
_PATH_TOKEN = re.compile(r"(?<![\w@./-])((?:" + "|".join(_REPO_ROOTS) + r")/[\w@./-]*)")


def notice_missing_paths(notice: str, repo: Path) -> list[str]:
    """Repository paths ``notice`` names that do not exist under ``repo`` (sentence punctuation
    after a path is not part of it)."""
    missing: list[str] = []
    for match in _PATH_TOKEN.finditer(notice):
        path = match.group(1).rstrip(".,;:")
        if not (repo / path).exists() and path not in missing:
            missing.append(path)
    return missing


def test_notice_missing_path_check_catches_a_dangling_name_and_passes_real_ones(
    tmp_path: Path,
) -> None:
    """T-0164 D2 adversarial, both directions, on a synthetic repository.

    Too wide: existing files and directories, a path ending a sentence, a tarball-internal
    ``package/LICENSE`` and a URL are not reported. Absent: a NOTICE that still names the deleted
    bundle (the D2 adversarial), or its MANIFEST inside parentheses, is.
    """
    (tmp_path / "hud" / "src").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "audio").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "audio" / "MANIFEST.md").write_text("x")
    clean = (
        "Files: hud/src/ and tests/fixtures/audio/MANIFEST.md.\n"
        "[package/LICENSE, tarball 7f27…] https://creativecommons.org/licenses/by/4.0/\n"
    )
    assert notice_missing_paths(clean, tmp_path) == []
    stale = (
        clean
        + "2. A2UI renderer bundle (hud/vendor/a2ui-v0_9.bundle.js)\n(hud/vendor/MANIFEST.md).\n"
    )
    assert notice_missing_paths(stale, tmp_path) == [
        "hud/vendor/a2ui-v0_9.bundle.js",
        "hud/vendor/MANIFEST.md",
    ]


def test_notice_names_no_missing_path() -> None:
    """T-0164 D2: NOTICE names no repository file or directory that does not exist.

    An invariant: it held before D (the bundle existed) and holds after; a NOTICE that still names
    the deleted bundle, its MANIFEST or ``hud/package-lock.json`` turns it red.
    """
    missing = notice_missing_paths(_notice(), REPO)
    assert missing == [], f"NOTICE names paths that do not exist: {missing}"


#: The NOTICE item titles after T-0164 D1 removes §2 "A2UI renderer bundle" (hand-copied from
#: NOTICE at cbce2643, items 1 and 3-8, in order).
T0164_NOTICE_ITEMS = (
    "diffgemma-fa (vendored code)",
    "A2UI v0.9.1 schema assets",
    "vLLM (verbatim excerpts, test oracles only)",
    "Berkeley Function Calling Leaderboard data (Gorilla)",
    "Kyutai Pocket TTS and Kyutai STT",
    "Unmute (Kyutai)",
    "Gemma / DiffusionGemma and NVIDIA's NVFP4 checkpoint",
)


def test_notice_drops_the_a2ui_renderer_bundle_item_and_renumbers() -> None:
    """T-0164 D1: NOTICE §2 "A2UI renderer bundle" is removed and the items are renumbered 1..7;
    nothing else in NOTICE names the bundle, its MANIFEST, its lock file or its packages' versions."""
    notice = _notice()
    items = re.findall(r"^(\d+)\. (.+?)\s*$", notice, re.M)
    items = [
        (n, t)
        for n, t in items
        if not t.startswith("Redistributions") and not t.startswith("Neither")
    ]
    assert [int(n) for n, _ in items] == list(range(1, len(T0164_NOTICE_ITEMS) + 1)), items
    assert tuple(t for _, t in items) == T0164_NOTICE_ITEMS, items
    assert notice_bundle_leftovers(notice) == []


#: What NOTICE may no longer name once §2 is gone (T-0164 D1): the item, its paths, the lock file,
#: the @a2ui 0.11.0 version and (review T2) the bundled renderer package itself.
_BUNDLE_LEFTOVERS = (
    "A2UI renderer bundle",
    "hud/vendor/",
    "a2ui-v0_9.bundle",
    "hud/package-lock.json",
    "0.11.0",
    "@a2ui/lit",
)


def notice_bundle_leftovers(notice: str) -> list[str]:
    """The ``_BUNDLE_LEFTOVERS`` strings ``notice`` still contains."""
    return [gone for gone in _BUNDLE_LEFTOVERS if gone in notice]


def test_notice_bundle_leftovers_catches_the_package_name_alone() -> None:
    """Review T2, both directions: a NOTICE naming ``@a2ui/lit`` (no version, no path) is caught; the
    A2UI v0.9.1 schema-assets item that stays (``local_astra/ui/schemas/a2ui_v0_9_1``) is not."""
    kept = "2. A2UI v0.9.1 schema assets\nFiles: local_astra/ui/schemas/a2ui_v0_9_1/\n"
    assert notice_bundle_leftovers(kept) == []
    assert notice_bundle_leftovers(kept + "The page once used @a2ui/lit.\n") == ["@a2ui/lit"]


# --------------------------------------------------------------------------- criterion 3

_PROVENANCE = (
    REPO / "local_astra" / "decoding" / "_vendor" / "PROVENANCE.md",
    REPO / "local_astra" / "eval" / "cd" / "_vendor" / "PROVENANCE.md",
)


@pytest.mark.parametrize("path", _PROVENANCE, ids=["decoding", "eval_cd"])
def test_provenance_licence_prose(path: Path) -> None:
    """Criterion 3; the negative (no row, file list or commit text changes) is ``tests/cd/unit/
    test_vendor_provenance.py`` and ``test_eval_scorer_parity.py`` staying green, unchanged."""
    text = path.read_text()
    flat = " ".join(text.split())
    assert (
        "Copyright James Assiene; licensed Apache-2.0 for this repository (ruling 2026-10-25)"
        in flat
    )
    assert "not yet licensed" not in flat
    assert _DEMOS + "diffgemma-fa" not in text
    assert "<diffgemma-fa clone>" in text
    assert "james-assiene" in text


def test_dgfa_preserved_manifest_licence() -> None:
    """Ruling O-T (2): the preserved diffgemma-fa data row carries the criterion-3 licence wording."""
    flat = " ".join(
        (REPO / "tests" / "fixtures" / "cd" / "dgfa_preserved" / "MANIFEST.md").read_text().split()
    )
    assert "not yet licensed" not in flat
    assert (
        "Copyright James Assiene; licensed Apache-2.0 for this repository (ruling 2026-10-25)"
        in flat
    )


def test_docs07_licence_open_item_updated() -> None:
    text = (REPO / "docs" / "07-constrained-decoding.md").read_text()
    assert "not yet licensed" not in text
    assert "Apache-2.0" in text


# --------------------------------------------------------------------------- criterion 8

#: Criterion 8's phrase without its leading article (orchestrator ruling: docs/01:9 reads "…fully offline, on-device
#: voice-and-vision assistant inspired by…", where "an" sits earlier in the sentence).
_PHRASE = (
    "on-device voice-and-vision assistant inspired by research demos like Google's Project Astra"
)


def _norm(text: str) -> str:
    return " ".join(text.replace("’", "'").split())


@pytest.mark.parametrize(
    "rel",
    [
        "CLAUDE.md",  # the main session made this edit on main (445734f, O-I (2))
        "README.md",
        "pyproject.toml",  # T-0156 R's description (merged); "inspired by" not required there (O-T 3)
        "docs/00-README.md",
        "docs/01-design.md",
        "local_astra/__init__.py",
    ],
)
def test_no_astra_style_wording(rel: str) -> None:
    text = (REPO / rel).read_text()
    assert not re.search(r"astra-style", text, re.IGNORECASE), rel


@pytest.mark.parametrize(
    "rel",
    [
        "CLAUDE.md",
        "README.md",
        "docs/00-README.md",
        "docs/01-design.md",
        "local_astra/__init__.py",
    ],
)
def test_inspired_by_wording_present(rel: str) -> None:
    assert _PHRASE.lower() in _norm((REPO / rel).read_text()).lower(), rel


def test_proactivity_reference_entry_stays() -> None:
    """Criterion 8 negative: docs/15's reference entry is a citation, not a claim; it stays."""
    assert "Project Astra page" in (REPO / "docs" / "15-proactivity.md").read_text()


def test_t0135_out_of_scope_line_reworded() -> None:
    text = _norm((REPO / "docs" / "tasks" / "T-0135.md").read_text())
    assert "it is plugin work on the GPU path and is out of scope here" in text
    assert "private CD notes" not in text


# --------------------------------------------------------------------------- criteria 9-10


def test_readme_body() -> None:
    text = _norm((REPO / "README.md").read_text())
    low = text.lower()
    for needle in (
        "research prototype",
        "linux",
        "rtx 5090",
        "32 gb",
        "docker",
        "not production software",
        "diffusion_gearvis",
        'uv run pytest tests/unit tests/contract -q -m "not gpu"',
        "tests/cd/fixtures/perstep_golden_",
        "pickle",
        "license",
        "notice",
        ".claude/",
        "issues are welcome",
        "pull requests may not be reviewed",
    ):
        assert needle in low, needle


def test_contributing_and_security_and_no_github_dir() -> None:
    """Criterion 10, release re-pin: James 2026-11-01 (supersedes H-T157-3 "none"): ``CONTRIBUTING.md`` and
    ``SECURITY.md`` are tracked at the root and non-empty (allowed and required), no other file is named
    CONTRIBUTING, and nothing under ``.github/`` is tracked."""
    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for name in ("CONTRIBUTING.md", "SECURITY.md"):
        assert name in files, f"{name} is not tracked at the repo root"
        assert (REPO / name).read_text(encoding="utf-8").strip(), f"{name} is empty"
    assert [f for f in files if f.startswith(".github/")] == []
    assert [f for f in files if "contributing" in f.lower() and f != "CONTRIBUTING.md"] == []


# --------------------------------------------------------------------------- criterion 6

PIPEWIRE = REPO / "tests" / "fixtures" / "pipewire"
_DERIVED = (
    "dump_rig_ok.json",
    "dump_auto_null.json",
    "dump_no_sink.json",
    "dump_quantum_1024.json",
    "dump_not_linked_yet.json",
    "dump_sink_differs.json",
    "dump_two_peers.json",
)


def test_pipewire_dumps_regenerate(tmp_path: Path) -> None:
    """``make_fixtures.py`` run on a copy of the base reproduces the seven committed dumps byte for byte, and the
    MANIFEST sha256 table equals the files' real hashes (every row)."""
    shutil.copy(PIPEWIRE / "make_fixtures.py", tmp_path / "make_fixtures.py")
    shutil.copy(PIPEWIRE / "dump_recorded_base.json", tmp_path / "dump_recorded_base.json")
    subprocess.run(
        [sys.executable, str(tmp_path / "make_fixtures.py")], check=True, capture_output=True
    )
    for name in _DERIVED:
        assert (tmp_path / name).read_bytes() == (PIPEWIRE / name).read_bytes(), name
    table = re.findall(
        r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", (PIPEWIRE / "MANIFEST.md").read_text(), re.M
    )
    assert {n for n, _ in table} >= {"dump_recorded_base.json", *_DERIVED}
    for name, digest in table:
        assert hashlib.sha256((PIPEWIRE / name).read_bytes()).hexdigest() == digest, name


def test_pipewire_base_is_scrubbed_to_the_placeholders() -> None:
    """Criteria 5-6: the base dump carries the placeholders (user, gpu-box, 32-zero machine-id) and its MANIFEST
    records the replacement as a mechanical reduction."""
    base = (PIPEWIRE / "dump_recorded_base.json").read_text()
    assert scan([("base", base.encode())], _optional_needles()) == []
    ids = re.findall(r'"application\.process\.machine-id": "([0-9a-f]+)"', base)
    assert ids and set(ids) == {"0" * 32}, set(ids)
    assert re.findall(r'"application\.process\.user": "([^"]+)"', base) and set(
        re.findall(r'"application\.process\.user": "([^"]+)"', base)
    ) == {"user"}
    assert set(re.findall(r'"application\.process\.host": "([^"]+)"', base)) == {"gpu-box"}
    assert re.findall(r'"core\.name": "pipewire-user-\d+"', base)
    manifest = (PIPEWIRE / "MANIFEST.md").read_text()
    assert "T-0157" in manifest


def test_box_fixture_uses_placeholders() -> None:
    """Criterion 5: ``user@host`` in the verbatim wpctl capture becomes ``user@gpu-box``."""
    text = (REPO / "tests" / "fixtures" / "box" / "wpctl_status.txt").read_text()
    assert "user@gpu-box" in text
    assert scan([("wpctl", text.encode())], _optional_needles()) == []


@pytest.mark.parametrize("rel", ["box/MANIFEST.md", "services/MANIFEST.md", "a2ui/MANIFEST.md"])
def test_verbatim_fixture_manifests_record_the_replacement(rel: str) -> None:
    assert (
        "identifiers replaced 2026-10-25 (T-0157)"
        in (REPO / "tests" / "fixtures" / rel).read_text()
    ), rel


def test_memory_fixture_records_store_unexpanded() -> None:
    import json

    data = json.loads(
        (REPO / "tests" / "fixtures" / "memory" / "calculus_session.json").read_text()
    )
    assert data["provenance"]["store"] == "~/.local/share/local_astra/memory.sqlite"


def _module_level_log_literal(path: Path) -> str:
    """The string literal of the module-level ``LOG = Path("...")`` assignment."""
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "LOG" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
        ):
            value = node.value.args[0].value
            assert isinstance(value, str)
            return value
    raise AssertionError(f"no LOG = Path('...') in {path}")


@pytest.mark.parametrize(
    "rel",
    [
        "tests/fixtures/t0135/build_thought_tails.py",
        "tests/fixtures/toolcall/build_t0134_t0132_answers.py",
    ],
)
def test_log_builders_read_repo_relative_logs(rel: str) -> None:
    assert _module_level_log_literal(REPO / rel).startswith("logs/"), rel


# --------------------------------------------------------------------------- criterion 7


def test_coldboot_watch_cds_relative_to_itself() -> None:
    text = (REPO / "scripts" / "demo" / "coldboot_watch.sh").read_text()
    assert 'cd "$(dirname "$0")/../.."' in text


def test_scrubbed_locked_scripts_keep_their_mode() -> None:
    """Criterion 7 negative: the index modes as they were at 41711fa (read off ``git ls-files -s`` by hand)."""
    expected = {
        "scripts/demo/coldboot_watch.sh": "100755",
        "scripts/locked/audio_acoustic.py": "100644",
        "scripts/locked/t0007a_census.sh": "100755",
        "scripts/locked/t0011_tier3.sh": "100644",
        "scripts/locked/t0039_text_mode_live.sh": "100755",
        "scripts/locked/t0048_cold_boot.sh": "100644",
    }
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-s", "--", *expected],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    got = {line.split("\t")[1]: line.split()[0] for line in out.splitlines()}
    assert got == expected


# --------------------------------------------------------------------------- criterion 11

_HISTORY_ROWS = (
    "tests/unit/test_toolcall_score_tightening.py::test_the_frozen_oracle_is_4c7dfdcs_exact",
    "tests/unit/test_guards_use_fixed_refs.py::test_require_commits_skips_naming_an_absent_sha",
)
_COPY = (
    "pyproject.toml",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/_home_jail.py",
    "tests/_git_range.py",
    "tests/unit/__init__.py",
    "tests/unit/_frozen_score_4c7dfdc.py",
    "tests/unit/_toolcall_support.py",
    "tests/unit/test_toolcall_score_tightening.py",
    "tests/unit/test_guards_use_fixed_refs.py",
)


def test_history_free_guards_skip(tmp_path: Path) -> None:
    """Criterion 11: in a one-commit ``git init`` copy (no 4c7dfdc, no 6a8600b), both rows skip naming the absent
    commit; neither fails and neither passes. A skip can only come from the copy: the real repository has both
    commits, so a run that imported the real tree would pass instead."""
    for rel in _COPY:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "copy"], check=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-n",
            "0",
            *_HISTORY_ROWS,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr
    # pytest -q's final line is "<counts> in <t>s"; it must be exactly two skips: no pass, no failure, no error.
    summary = [ln for ln in out.splitlines() if re.search(r" in \d+(\.\d+)?s\b", ln)]
    assert summary and re.fullmatch(r"=*\s*2 skipped in \d+(\.\d+)?s\b.*", summary[-1].strip()), (
        out[-3000:]
    )
    assert proc.returncode == 0, out[-3000:]
    skips = [ln for ln in out.splitlines() if ln.startswith("SKIPPED")]
    assert len(skips) == 2, skips
    assert any("4c7dfdc" in ln for ln in skips) and any("6a8600b" in ln for ln in skips), skips
