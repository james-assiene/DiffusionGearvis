"""A blunt grep over ``hud/``: loopback only, and never a byte of audio (T-0008 criteria 12, 13).

Criterion 13: "the HUD never touches audio" — the Conductor owns PipeWire capture and playback
(docs/02 ADR-013, docs/01 D7). Criterion 12: all services bind to loopback only (CLAUDE.md,
docs/02 §9); the page must work with no network route, so no CDN import and no build step.

Deliberately single-purpose and textual: it is the fence that catches a stray ``<audio>`` or a
``https://cdn...`` import that no behavioural test would ever look at. Comments are stripped
first, so a URL in an explanatory comment is allowed (the criterion says "outside comments").
Only the page's *code* files are scanned (.html/.js/.css/.json); ``hud/README.md`` is prose and
may link to specs.

**T-0164 D2** (docs/tasks/T-0164.md, group D; amends T-0024 criterion 10): the vendored A2UI
bundle is deleted. Its two rows (the bundle is there; the bundle has no network verb) *became*
"``hud/vendor/`` does not exist and nothing under ``hud/`` references it". The URL scan now covers
all of ``hud/**`` with no vendor and no ``package-lock.json`` exemption. The audio patterns keep
covering the whole tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HUD = REPO / "hud"
CODE_SUFFIXES = {".html", ".js", ".mjs", ".css", ".json", ".map"}

#: The shell T-0008 owns (``docs/tasks/T-0008.md`` Owned files). Listed so this scan can never
#: pass vacuously over an empty directory.
REQUIRED = (
    "index.html",
    "styles.css",
    "src/main.js",
    "src/protocol.js",
    "src/ws.js",
    "src/capture.js",
    "src/frames.js",
    "src/phash.js",
    "src/state.js",
    "src/panels.js",
    "src/panels/transcript.js",
    "src/panels/spoken.js",
    "src/panels/thinking.js",
)

#: T-0024 criterion 10: the A2UI renderer's own files, so neither scan below can pass vacuously
#: over a directory the panel never landed in. Activated for ``/implement T-0024``.
UI_REQUIRED = (
    "src/ui/index.js",
    "src/ui/bridge.js",
    "src/ui/action.js",
    "src/ui/a2ui.js",
)

#: T-0024 criterion 10: the network *verbs* a panel must not reach for (the panels open no
#: transport of their own). Paired with the behavioural CDP assertion in
#: ``tests/integration/test_hud_ui_surface.py`` (zero non-loopback requests over a whole
#: render-and-click session), which also catches a URL built at runtime from fragments.
NETWORK_VERBS = (
    (re.compile(r"\bfetch\s*\("), "fetch("),
    (re.compile(r"\bXMLHttpRequest\b"), "XMLHttpRequest"),
    (re.compile(r"\bnew\s+WebSocket\s*\("), "new WebSocket("),
    (re.compile(r"\bimportScripts\s*\("), "importScripts("),
    (re.compile(r"\bsendBeacon\s*\("), "navigator.sendBeacon("),
    (re.compile(r"\bimport\s*\(\s*[\"\'`]https?://"), "dynamic import() of an absolute URL"),
    (re.compile(r"\bEventSource\b"), "EventSource"),
)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}

#: Ruling A7: the A2UI catalog *identifier* our gate emits. Compared as a string and never
#: fetched; the guarantee of that is behavioural (zero non-loopback requests over a whole CDP
#: session, ``tests/integration/test_hud_ui_surface.py``), not textual. Allowed in
#: ``hud/src/ui/a2ui.js`` and nowhere else, and nothing else is allowed there.
GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
#: The deleted vendor's ``v0_9`` twin (T-0164 D1 removes ``RENDERER_CATALOG_ID``): admitted
#: nowhere, and named here only so the rows below can say it is gone.
VENDOR_CATALOG_ID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
CATALOG_ID_FILE = "src/ui/a2ui.js"

#: T-0164 amendment 2026-10-27: the SVG namespace *identifier* the Icon builder passes to
#: ``createElementNS``. An XML namespace name, compared as a string and never fetched. Admitted as
#: this exact quoted literal in ``hud/src/ui/render.js`` only; any other URL there, and this string
#: in any other file, still fails.
SVG_NS = "http://www.w3.org/2000/svg"
SVG_NS_FILE = "src/ui/render.js"
URL_RE = re.compile(r"\b(?:https?|wss?)://([^/\s\"'`)>\\]+)")
LINE_COMMENT_RE = re.compile(r"(?<!:)//[^\n]*")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

AUDIO_PATTERNS = (
    (re.compile(r"\bAudioContext\b"), "AudioContext — the Conductor owns audio (docs/02 ADR-013)"),
    (re.compile(r"<audio\b", re.IGNORECASE), "an <audio> element"),
    (re.compile(r"\baudio\s*:\s*true\b"), "a capture request with audio: true"),
    (re.compile(r"\bAudioWorklet\b"), "AudioWorklet"),
    (re.compile(r"\bMediaRecorder\b"), "MediaRecorder"),
    (re.compile(r"getAudioTracks\s*\("), "getAudioTracks() — the page never holds an audio track"),
    (re.compile(r"\bwebkitAudioContext\b"), "webkitAudioContext"),
)


def code_files() -> list[Path]:
    """Every code file under ``hud/`` (the audio scan's scope); no directory is skipped
    (``hud/node_modules`` must not exist: ``test_the_npm_manifest_is_bare_…``)."""
    return sorted(p for p in HUD.rglob("*") if p.is_file() and p.suffix in CODE_SUFFIXES)


def strip_comments(text: str, suffix: str) -> str:
    text = HTML_COMMENT_RE.sub(" ", text)
    if suffix in {".js", ".mjs", ".css", ".html"}:
        text = BLOCK_COMMENT_RE.sub(" ", text)
        text = LINE_COMMENT_RE.sub(" ", text)
    return text


@pytest.mark.parametrize("relative", [*REQUIRED, *UI_REQUIRED])
def test_the_page_the_scan_is_supposed_to_cover_exists(relative):
    """Without this the greps below would pass over an empty ``hud/`` (T-0008 Owned files).

    The ``src/ui/*.js`` rows are T-0024 criterion 10's amendment: the fences below must cover the
    renderer's own files, so the scan is only meaningful once they exist.
    """
    assert (HUD / relative).is_file(), f"T-0008 adds hud/{relative}"


def test_no_file_under_hud_touches_audio():
    """Criterion 13, negative: no AudioContext, no <audio>, no audio track requested."""
    offenders: list[str] = []
    for path in code_files():
        body = strip_comments(path.read_text(errors="replace"), path.suffix)
        for pattern, what in AUDIO_PATTERNS:
            if pattern.search(body):
                offenders.append(f"{path.relative_to(REPO)}: {what}")
    assert not offenders, (
        "criterion 13: the HUD never touches audio (docs/01 D7, docs/02 ADR-013):\n"
        + "\n".join(offenders)
    )


def test_display_capture_is_requested_without_audio():
    """Criterion 13: ``getDisplayMedia`` is called with ``audio: false``."""
    body = "\n".join(
        strip_comments(p.read_text(errors="replace"), p.suffix)
        for p in code_files()
        if p.suffix in {".js", ".mjs"}
    )
    calls = [m.start() for m in re.finditer(r"getDisplayMedia\s*\(", body)]
    assert calls, (
        "T-0008 criterion 6/13: the page captures with getDisplayMedia (called with audio: false); "
        "no call found under hud/"
    )
    for start in calls:
        window = body[start : start + 300]
        assert re.search(r"\baudio\s*:\s*false\b", window), (
            "criterion 13: getDisplayMedia must be called with audio: false, got: "
            f"{window.splitlines()[0][:120]!r}"
        )


def test_mute_stops_the_tracks_and_does_not_merely_disable_them():
    """Criterion 6, the named negative, as far as a text scan can reach it.

    ``.claude/rules/perception-memory.md``: "global mute stops capture at the pipeline level, not
    at the consumer". ``track.enabled = false`` leaves the browser capture session and its sharing
    indicator alive, so it is the wrong implementation even though it stops frames reaching us.
    The *behavioural* half of this criterion needs an injected stream provider and is not covered
    by any automated test in this task — see the test-author's report and criterion 19.
    """
    capture = HUD / "src" / "capture.js"
    assert capture.is_file(), "T-0008 adds hud/src/capture.js"
    body = strip_comments(capture.read_text(), ".js")
    assert re.search(r"\.stop\s*\(\s*\)", body), (
        "criterion 6: mute calls stop() on every track of every stream (docs/01 §3); no stop() "
        "call found in hud/src/capture.js"
    )
    disabled = re.search(r"\benabled\s*=\s*(false|true)\b", body)
    assert disabled is None, (
        "criterion 6 negative: setting track.enabled leaves the capture session and the browser's "
        f"sharing indicator alive, so it does not satisfy mute-at-the-pipeline; found {disabled}"
    )


def url_offenders(relative: str, text: str, suffix: str) -> list[str]:
    """The non-loopback absolute URLs in one file under ``hud/`` (``relative`` to ``hud/``).

    No file is exempt (T-0164 D2); the only admissions are the gate's catalog id in ``a2ui.js``
    (ruling A7) and the SVG namespace literal in ``render.js`` (amendment 2026-10-27).
    """
    body = strip_comments(text, suffix)
    found: list[str] = []
    for match in URL_RE.finditer(body):
        host = match.group(1).split("@")[-1].rsplit(":", 1)
        hostname = host[0] if len(host) == 1 or host[-1].isdigit() else match.group(1)
        if hostname in LOOPBACK_HOSTS:
            continue
        allowed = relative == CATALOG_ID_FILE and body.startswith(GATE_CATALOG_ID, match.start())
        # the exact quoted literal: a quote before and the same quote right after the identifier
        start, end = match.start(), match.start() + len(SVG_NS)
        quote = body[start - 1 : start]
        allowed = allowed or (
            relative == SVG_NS_FILE
            and quote in {'"', "'", "`"}
            and body[start:end] == SVG_NS
            and body[end : end + 1] == quote
        )
        if not allowed:
            found.append(match.group(0))
    return found


def test_the_svg_namespace_literal_is_admitted_in_render_js():
    """T-0164 amendment: the guard is not too strict for the one identifier it admits."""
    for q in ('"', "'", "`"):
        code = f"const SVG = {q}{SVG_NS}{q};\ndoc.createElementNS(SVG, {q}svg{q});\n"
        assert url_offenders(SVG_NS_FILE, code, ".js") == [], q


def test_any_other_url_in_render_js_still_fails_the_egress_scan():
    """T-0164 amendment, negative: render.js is not exempt; only the exact identifier is."""
    for url in (
        "https://example.com/x.png",
        "http://www.w3.org/1999/xlink",
        "https://www.w3.org/2000/svg",
        "http://www.w3.org/2000/svg/../evil.js",
        "http://www.w3.org/2000/svgs",
        "http://www.w3.org/2000/svg#x",
        "http://www.w3.org/2000/",
    ):
        code = f'const NS = "{SVG_NS}";\nconst u = "{url}";\n'
        # one offender, the second literal (the scan reports scheme://host, URL_RE stops at "/")
        assert url_offenders(SVG_NS_FILE, code, ".js") == [
            url.split("/", 3)[0] + "//" + url.split("/")[2]
        ], url
    # the identifier without its quotes (e.g. spliced into a longer template) is not the literal
    assert url_offenders(SVG_NS_FILE, f"const u = `${{a}}{SVG_NS}/x`;\n", ".js") != []


def test_the_svg_namespace_literal_fails_the_egress_scan_in_any_other_file():
    """T-0164 amendment, negative: the admission is for render.js only."""
    for other in (
        "src/ui/engine.js",
        "src/ui/index.js",
        "src/main.js",
        "index.html",
        "src/ui/render.mjs",
    ):
        code = f'const SVG = "{SVG_NS}";\n'
        assert url_offenders(other, code, ".js") == ["http://www.w3.org"], other


# --- T-0164 D2: the URL scan covers all of hud/** (no vendor, no package-lock exemption) ---------


def whole_tree_url_offenders(hud: Path) -> list[str]:
    """T-0164 D2: every non-loopback URL literal in every file under ``hud`` but ``*.md`` prose.

    No directory is skipped and no file is exempt; the only admissions are the gate's catalog id
    in ``src/ui/a2ui.js`` (ruling A7, without the vendor's ``v0_9`` twin that D1 deletes) and the
    SVG namespace literal in ``src/ui/render.js`` (amendment 2026-10-27). Any suffix is read: a
    re-added bundle need not be ``.js`` to be fetched.
    """
    found: list[str] = []
    for path in sorted(p for p in hud.rglob("*") if p.is_file() and p.suffix != ".md"):
        relative = path.relative_to(hud).as_posix()
        text = path.read_text(errors="replace")
        for url in url_offenders(relative, text, path.suffix):
            found.append(f"hud/{relative}: {url}")
    return found


def _tiny_hud(root: Path) -> Path:
    """A synthetic, post-D page tree: the shell, the two constants and a renderer; no vendor."""
    hud = root / "hud"
    (hud / "src" / "ui").mkdir(parents=True)
    (hud / "index.html").write_text('<script type="module" src="./src/main.js"></script>\n')
    (hud / "src" / "main.js").write_text('import { mount } from "./ui/index.js";\n')
    (hud / "src" / "ui" / "index.js").write_text('import { createEngine } from "./engine.js";\n')
    (hud / "src" / "ui" / "a2ui.js").write_text(
        f'export const A2UI_SPEC_VERSION = "v0.9.1";\n'
        f'export const GATE_CATALOG_ID = "{GATE_CATALOG_ID}";\n'
    )
    (hud / "src" / "ui" / "render.js").write_text(f'const SVG = "{SVG_NS}";\n')
    (hud / "README.md").write_text("Spec: https://a2ui.org/specification/v0_9_1/ (prose).\n")
    return hud


def test_the_whole_tree_url_scan_passes_a_clean_tree_and_catches_a_readded_vendor_file(tmp_path):
    """T-0164 D2 adversarial, both directions, on a synthetic tree (the real tree is the next row).

    Too wide: the post-D page, its gate id, its SVG namespace and README prose are not offenders.
    Absent: a re-added ``hud/vendor/x.js`` carrying a CDN URL, a ``package-lock.json`` with the npm
    registry, the vendor's ``v0_9`` catalog id back in ``a2ui.js``, and a non-code suffix are.
    """
    hud = _tiny_hud(tmp_path)
    assert whole_tree_url_offenders(hud) == []
    (hud / "vendor").mkdir()
    (hud / "vendor" / "x.js").write_text('import "https://cdn.example.com/lit.js";\n')
    (hud / "package-lock.json").write_text('{"resolved": "https://registry.npmjs.org/lit"}\n')
    (hud / "src" / "ui" / "a2ui.js").write_text(f'export const R = "{VENDOR_CATALOG_ID}";\n')
    (hud / "src" / "ui" / "fonts.txt").write_text("https://fonts.example.net/a.woff2\n")
    assert whole_tree_url_offenders(hud) == [
        "hud/package-lock.json: https://registry.npmjs.org",
        "hud/src/ui/a2ui.js: https://a2ui.org",
        "hud/src/ui/fonts.txt: https://fonts.example.net",
        "hud/vendor/x.js: https://cdn.example.com",
    ]


def test_no_absolute_url_anywhere_under_hud_points_anywhere_but_loopback():
    """Criterion 12, negative, widened by T-0164 D2: no CDN, no external host, works with no
    network route, over all of ``hud/**``. With the bundle deleted there is no ``hud/vendor/**``
    to exempt and no ``package-lock.json`` (D1), so nothing is. (Replaces the T-0024 row that
    scanned only the page shell and ``hud/src/**``.)
    """
    offenders = whole_tree_url_offenders(HUD)
    assert not offenders, (
        "T-0164 D2 / T-0008 criterion 12: no file under hud/ (prose *.md aside) carries an absolute "
        "http(s)/ws(s) literal pointing anywhere but loopback; the vendor and package-lock "
        "exemptions are gone with the vendor (D1):\n" + "\n".join(offenders)
    )


def test_the_page_has_no_build_step_and_no_vendored_dependency_tree():
    """Criterion 12, negative: no node_modules, no bundler output (T-0008 ships buildless ESM)."""
    assert not (HUD / "node_modules").exists(), (
        "criterion 12: no node_modules — T-0008 is buildless plain ES modules; npm arrives with "
        "T-0010 when it needs @a2ui/lit (docs/tasks/T-0008.md Out of scope)"
    )


def test_the_readme_no_longer_promises_a_build_step():
    """T-0008 Out of scope, named spec change: "Build tooling arrives with T-0008 (vite, pinned)"."""
    readme = HUD / "README.md"
    assert readme.is_file(), "hud/README.md is T-0008's file"
    assert "vite" not in readme.read_text().lower(), (
        "T-0008 ships no build step and the implementer updates the hud/README.md line that "
        "promises vite (docs/tasks/T-0008.md Out of scope)"
    )


# --- T-0164 D1/D2: what T-0024's two vendor rows became --------------------------------------------

#: A *reference* to the deleted vendor: a ``vendor/`` path segment, the bundle's file name, or the
#: vendoring script. Read as raw text, comments included, because "nothing under hud/ references
#: it" is a statement about the files, not the code: prose naming the path trips it, and that is
#: intended (rules/tests.md: say so in the message). The bare word "vendor"/"vendored" in prose
#: (engine.js's parity notes, bridge.js's header) is not a reference and does not trip it.
#: Also the vendoring step's own words (review T1): ``npm ci`` / ``npm install`` and the package name
#: ``@a2ui/lit`` (D1: hud/README.md no longer mentions the bundle, npm or the vendoring step).
VENDOR_REFERENCE_RE = re.compile(
    r"(?<![\w.-])vendor/|a2ui-v0_9\.bundle|hud_vendor\.sh|\bnpm\s+(?:ci|install|i)\b|@a2ui/lit\b"
)


def vendor_findings(hud: Path) -> list[str]:
    """T-0164 D2: ``hud/vendor/`` exists, or a file under ``hud`` names it (path:line)."""
    found: list[str] = []
    vendor = hud / "vendor"
    if vendor.exists():
        listing = sorted(p.relative_to(hud).as_posix() for p in vendor.rglob("*")) or ["(empty)"]
        found.append(f"hud/vendor/ exists: {', '.join(listing)}")
    for path in sorted(p for p in hud.rglob("*") if p.is_file()):
        text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if VENDOR_REFERENCE_RE.search(line):
                found.append(
                    f"hud/{path.relative_to(hud).as_posix()}:{lineno}: {line.strip()[:120]}"
                )
    return found


def test_the_vendor_check_passes_a_clean_tree_and_catches_every_way_back(tmp_path):
    """T-0164 D2 adversarial, both directions, on a synthetic tree.

    Too wide: the post-D page, and prose using the words "vendor"/"vendored bundle" without a path
    (as engine.js and bridge.js do), are clean. Absent: a re-added ``hud/vendor/x.js``, an empty
    ``hud/vendor/``, an ``index.js`` importing ``../../vendor/``, the bundle's file name, and a
    README naming the vendoring script each produce a finding.
    """
    hud = _tiny_hud(tmp_path)
    (hud / "src" / "ui" / "engine.js").write_text(
        "// It mimics the vendored MessageProcessor; the vendor refuses __proto__.\n"
        "// No DOM and no vendor bundle here; myvendor/ is not ours.\n"
    )
    assert vendor_findings(hud) == []

    (hud / "vendor").mkdir()
    assert vendor_findings(hud) == ["hud/vendor/ exists: (empty)"]
    (hud / "vendor" / "x.js").write_text("export const x = 1;\n")
    assert vendor_findings(hud) == ["hud/vendor/ exists: vendor/x.js"]
    (hud / "vendor" / "x.js").unlink()
    (hud / "vendor").rmdir()

    for relative, text in (
        ("src/ui/index.js", 'const m = await import("../../vendor/x.js");\n'),
        ("src/ui/a2ui.js", 'export const B = new URL("./a2ui-v0_9.bundle.js", import.meta.url);\n'),
        ("README.md", "Rebuild with scripts/hud_vendor.sh.\n"),
        ("README.md", "Run `npm ci` in hud/ first.\n"),
        ("README.md", "The renderer is @a2ui/lit 0.11.0.\n"),
    ):
        before = (hud / relative).read_text()
        (hud / relative).write_text(before + text)
        assert vendor_findings(hud) == [
            f"hud/{relative}:{len(before.splitlines()) + 1}: {text.strip()}"
        ], relative
        (hud / relative).write_text(before)
    assert vendor_findings(hud) == []


def test_there_is_no_vendor_directory_and_nothing_references_it():
    """T-0164 D1/D2: what T-0024's ``test_the_vendored_bundle_is_there_so_the_vendor_scans_are_not_vacuous``
    and ``test_the_vendored_bundle_contains_no_network_verb`` became once the bundle was deleted."""
    findings = vendor_findings(HUD)
    assert not findings, (
        "T-0164 D1/D2: hud/vendor/ is deleted and no file under hud/ names it, its bundle or "
        "scripts/hud_vendor.sh (raw text, comments and prose included: a README or comment naming "
        "the path trips this too, by design):\n" + "\n".join(findings)
    )


#: T-0164 D1 amendment 2026-10-27 (012c297d): ``hud/package.json`` stays as exactly this, so node 18
#: loads ``hud/src/**/*.js`` as ES modules in place. It declares no dependency.
HUD_PACKAGE_JSON = {"type": "module"}
LOCK_FILE_NAMES = ("package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml")


def npm_findings(repo: Path) -> list[str]:
    """T-0164 D1/D3: what is left of the npm vendoring step under ``repo``."""
    import json

    found: list[str] = []
    hud = repo / "hud"
    manifest = hud / "package.json"
    if not manifest.is_file():
        found.append('hud/package.json is missing (it stays as {"type": "module"})')
    else:
        try:
            value = json.loads(manifest.read_text())
        except ValueError as exc:
            value = f"<not JSON: {exc}>"
        if value != HUD_PACKAGE_JSON:
            found.append(f"hud/package.json is {value!r}, not exactly {HUD_PACKAGE_JSON!r}")
    found += [
        f"{p.relative_to(repo).as_posix()} (a lock file)"
        for p in sorted(hud.rglob("*"))
        if p.name in LOCK_FILE_NAMES
    ]
    found += [
        f"{p.relative_to(repo).as_posix()} (node_modules)"
        for p in sorted(hud.rglob("node_modules"))
    ]
    if (repo / "scripts" / "hud_vendor.sh").exists():
        found.append("scripts/hud_vendor.sh (the vendoring script)")
    return found


def test_the_npm_check_admits_only_the_bare_module_manifest(tmp_path):
    """T-0164 D1 amendment, both directions: ``{"type": "module"}`` (any whitespace) is clean;
    a dependency, a devDependency, any other key, a lock file, ``node_modules``, the vendoring
    script, or no manifest at all is a finding."""
    hud = _tiny_hud(tmp_path)
    (hud / "package.json").write_text('{ "type": "module" }\n')
    assert npm_findings(tmp_path) == []
    for body in (
        '{"type": "module", "dependencies": {"@a2ui/lit": "0.11.0"}}',
        '{"type": "module", "devDependencies": {"esbuild": "0.25.10"}}',
        '{"type": "module", "private": true}',
        '{"type": "commonjs"}',
        "{}",
    ):
        (hud / "package.json").write_text(body)
        assert len(npm_findings(tmp_path)) == 1, body
    (hud / "package.json").write_text('{"type": "module"}')
    (hud / "package-lock.json").write_text("{}")
    (hud / "node_modules" / "lit").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "hud_vendor.sh").write_text("npm ci\n")
    assert npm_findings(tmp_path) == [
        "hud/package-lock.json (a lock file)",
        "hud/node_modules (node_modules)",
        "scripts/hud_vendor.sh (the vendoring script)",
    ]
    (hud / "package.json").unlink()
    assert npm_findings(tmp_path)[0].startswith("hud/package.json is missing")


def test_the_npm_manifest_is_bare_and_the_lock_and_vendoring_script_are_gone():
    """T-0164 D1 (amended 2026-10-27) and D3: ``hud/package.json`` is exactly ``{"type": "module"}``,
    there is no lock file and no ``node_modules`` under ``hud/``, and no ``scripts/hud_vendor.sh``:
    no runtime dependency and no build step (T-0008 criterion 12)."""
    findings = npm_findings(REPO)
    assert not findings, "T-0164 D1/D3:\n" + "\n".join(findings)


#: T-0164 D1: what ``hud/src/ui/a2ui.js`` keeps.
A2UI_JS_EXPORTS = {"A2UI_SPEC_VERSION": "v0.9.1", "GATE_CATALOG_ID": GATE_CATALOG_ID}


#: T-0184 B1 (amends the row below by name): a2ui.js gains the v2 catalog id, H-T184-4.
CARDS_V2_CATALOG_ID = "gearvis:cards/v2"
A2UI_JS_EXPORTS_T0184 = {**A2UI_JS_EXPORTS, "CARDS_V2_CATALOG_ID": CARDS_V2_CATALOG_ID}


def _a2ui_js_exports() -> tuple[str, list[str]]:
    body = strip_comments((HUD / CATALOG_ID_FILE).read_text(), ".js")
    return body, re.findall(
        r"\bexport\s+(?:async\s+)?(?:const|let|var|function\*?|class)\s+(\w+)", body
    )


def test_a2ui_js_keeps_only_the_two_constants():
    """T-0164 D1: the loader, ``BUNDLE_URL``, ``RENDERER_CATALOG_ID`` and ``createEngine`` go;
    ``A2UI_SPEC_VERSION`` and ``GATE_CATALOG_ID`` stay with their values (hand-copied from
    ``local_astra/ui`` and the gate's catalog id, never read from the JS).

    Amended by T-0184 B1: the export set may also hold ``CARDS_V2_CATALOG_ID`` (its exact set and
    value are pinned by the T-0184 row below); nothing else."""
    body, exports = _a2ui_js_exports()
    assert set(A2UI_JS_EXPORTS) <= set(exports) <= set(A2UI_JS_EXPORTS_T0184), (
        f"T-0164 D1 / T-0184 B1: a2ui.js exports {exports}"
    )
    assert len(exports) == len(set(exports)), exports
    assert not re.search(r"\bexport\s*(?:\{|\*|default\b)", body), "T-0164 D1: no other export form"
    assert not re.search(r"\bimport\b", body), "T-0164 D1: a2ui.js loads nothing"
    for name, value in A2UI_JS_EXPORTS.items():
        assert re.search(rf'\bexport\s+const\s+{name}\s*=\s*"{re.escape(value)}"\s*;', body), name
    assert VENDOR_CATALOG_ID not in body, "T-0164 D1: RENDERER_CATALOG_ID's v0_9 id goes"


def test_t0184_b1_a2ui_js_exports_exactly_the_three_constants():
    """T-0184 B1: ``a2ui.js`` exports ``A2UI_SPEC_VERSION``, ``GATE_CATALOG_ID`` and
    ``CARDS_V2_CATALOG_ID = "gearvis:cards/v2"`` (values hand-copied from the task), nothing else."""
    body, exports = _a2ui_js_exports()
    assert sorted(exports) == sorted(A2UI_JS_EXPORTS_T0184), f"T-0184 B1: a2ui.js exports {exports}"
    for name, value in A2UI_JS_EXPORTS_T0184.items():
        assert re.search(rf'\bexport\s+const\s+{name}\s*=\s*"{re.escape(value)}"\s*;', body), name


def _v2_id_strays(hud: Path) -> list[str]:
    """Every file under ``hud`` but ``src/ui/a2ui.js`` whose text names the v2 catalog id. Strings
    are *not* stripped (an id spelled in a string literal is exactly what this forbids); comments
    are, so prose may mention it."""
    out: list[str] = []
    for path in sorted(p for p in hud.rglob("*") if p.is_file() and p.suffix in CODE_SUFFIXES):
        relative = path.relative_to(hud).as_posix()
        if relative == CATALOG_ID_FILE:
            continue
        if CARDS_V2_CATALOG_ID in strip_comments(path.read_text(errors="replace"), path.suffix):
            out.append(f"hud/{relative}")
    return out


def test_t0184_the_v2_id_location_scan_catches_a_stray_and_passes_a_clean_tree(tmp_path):
    """The T-0184 B1 location guard on a synthetic tree, both directions (not pending: it tests the
    scanner): the id in a2ui.js and in a comment elsewhere is fine; in another file's code it is not."""
    hud = _tiny_hud(tmp_path)
    (hud / "src" / "ui" / "a2ui.js").write_text(
        f'export const CARDS_V2_CATALOG_ID = "{CARDS_V2_CATALOG_ID}";\n'
    )
    (hud / "src" / "ui" / "engine.js").write_text(f"// renders {CARDS_V2_CATALOG_ID} too\n")
    assert _v2_id_strays(hud) == []
    (hud / "src" / "ui" / "index.js").write_text(
        f'createEngine({{ ids: ["{CARDS_V2_CATALOG_ID}"] }});\n'
    )
    assert _v2_id_strays(hud) == ["hud/src/ui/index.js"]


def test_t0184_b1_the_v2_catalog_id_is_named_in_a2ui_js_and_nowhere_else_under_hud():
    """T-0184 B1 / T-0164 A7: ``a2ui.js`` remains the only HUD file that names a catalog id; the
    v2 id is there (premise) and in no other file's code (index.js imports the constant)."""
    body, _ = _a2ui_js_exports()
    assert CARDS_V2_CATALOG_ID in body, "T-0184 B1: a2ui.js names the v2 id"
    assert _v2_id_strays(HUD) == []


def test_the_renderer_panel_never_opens_a_socket_of_its_own():
    """Criterion 3/10: the panel speaks through ``host.send`` on T-0025's seam, nothing else."""
    ui_dir = HUD / "src" / "ui"
    files = sorted(p for p in ui_dir.rglob("*.js")) if ui_dir.is_dir() else []
    assert files, "T-0024 adds hud/src/ui/*.js"
    offenders: list[str] = []
    for path in files:
        body = strip_comments(path.read_text(errors="replace"), path.suffix)
        for pattern, what in NETWORK_VERBS:
            if pattern.search(body):
                offenders.append(f"{path.relative_to(REPO)}: {what}")
    assert not offenders, (
        "criterion 3: the panel consumes T-0025's seam (host.subscribe / host.send) and opens no "
        "transport of its own; criterion 5: an unknown catalogId is an error line, never a fetch "
        "of that URL:\n" + "\n".join(offenders)
    )


def test_the_catalog_ids_live_in_exactly_one_file_and_are_never_fetched():
    """Ruling A7, the other half: the allowlist is a *location* rule, not a blanket pass.

    The gate's ``a2ui.org`` id (and the deleted vendor's ``v0_9`` twin) anywhere but
    ``hud/src/ui/a2ui.js`` is an ordinary non-loopback literal again — and in that file it may
    only appear as a string, which the network-verb scan over ``hud/src/ui/**`` above enforces.
    T-0164 D2: every code file under ``hud/`` is read, with no vendor or package-lock exemption.
    """
    strays: list[str] = []
    for path in code_files():
        relative = path.relative_to(HUD).as_posix()
        if relative == CATALOG_ID_FILE:
            continue
        body = strip_comments(path.read_text(errors="replace"), path.suffix)
        strays += [
            f"{path.relative_to(REPO)}: {catalog_id}"
            for catalog_id in (GATE_CATALOG_ID, VENDOR_CATALOG_ID)
            if catalog_id in body
        ]
    assert not strays, (
        "ruling A7: the A2UI catalog ids are allowed in hud/src/ui/a2ui.js only — the alias is one "
        "registration in one place, not a URL the rest of the page may carry around:\n"
        + "\n".join(strays)
    )


# --- T-0039 criterion 22: the same fences over hud/src/ask/ --------------------------------------
#
# The audio and URL scans above already walk every code file under ``hud/``, so the ask panel is
# covered by them the moment it lands. What is added here is the *non-vacuity* half (the files
# exist, so the scans are really looking at something) and the panel-scoped network-verb scan
# T-0024 introduced for ``hud/src/ui/**``: the panel speaks through ``host.send`` and opens no
# transport of its own (criterion 10: it uses only the panel host).

ASK_REQUIRED = ("src/ask/index.js", "src/ask/form.js")


@pytest.mark.parametrize("relative", ASK_REQUIRED)
def test_the_ask_panel_files_the_scan_is_supposed_to_cover_exist(relative):
    """Without these the greps below pass over an empty directory (T-0039 §3 Owned files)."""
    assert (HUD / relative).is_file(), f"T-0039 criterion 10 adds hud/{relative}"


def test_the_ask_panel_never_opens_a_transport_of_its_own():
    """Criterion 22: no fetch, no second WebSocket, no ``<audio>`` — one socket, the host's."""
    ask_dir = HUD / "src" / "ask"
    files = sorted(p for p in ask_dir.rglob("*.js")) if ask_dir.is_dir() else []
    assert files, "T-0039 criterion 10 adds hud/src/ask/*.js"
    offenders: list[str] = []
    for path in files:
        body = strip_comments(path.read_text(errors="replace"), path.suffix)
        for pattern, what in (*NETWORK_VERBS, *AUDIO_PATTERNS):
            if pattern.search(body):
                offenders.append(f"{path.relative_to(REPO)}: {what}")
    assert not offenders, (
        "criterion 22: the ask panel consumes the panel host (host.send / host.subscribe) and "
        "adds no network call and no audio element — text mode is typed in and read, never "
        "heard on the page (docs/01 D7, docs/02 ADR-013):\n" + "\n".join(offenders)
    )


# --- T-0043 criterion 25: the same fences over hud/src/log/ --------------------------------------
#
# The audio and URL scans above already walk every code file under ``hud/``, so the log panel is
# covered by them the moment it lands. What is added here is the *non-vacuity* half (the files
# exist, so the scans are really looking at something) and the panel-scoped network-verb scan
# T-0024 introduced: the panel speaks through ``host.send`` and opens no transport of its own.
# Criterion 25 names this module: "the page gains no ``fetch`` and no ``<audio>``
# (``tests/unit/test_hud_no_audio_no_egress.py`` extended over ``hud/src/log/``)".

LOG_REQUIRED = ("src/log/index.js", "src/log/gate.js")


@pytest.mark.parametrize("relative", LOG_REQUIRED)
def test_the_log_panel_files_the_scan_is_supposed_to_cover_exist(relative):
    """Without these the scan below passes over an empty directory (T-0043 §3 Owned files)."""
    assert (HUD / relative).is_file(), f"T-0043 criterion 19 adds hud/{relative}"


def test_the_log_panel_never_opens_a_transport_of_its_own():
    """Criterion 25: no fetch, no second WebSocket, no ``<audio>`` — one socket, the host's.

    A debug panel is exactly where a "just POST the log somewhere" would be written, and
    criterion 25 is the fence: zero non-loopback sockets and zero outbound calls are added by this
    task (docs/03 E8, docs/02 §9). The log holds every question and answer verbatim.
    """
    log_dir = HUD / "src" / "log"
    files = sorted(p for p in log_dir.rglob("*.js")) if log_dir.is_dir() else []
    assert files, "T-0043 criterion 19 adds hud/src/log/*.js"
    offenders: list[str] = []
    for path in files:
        body = strip_comments(path.read_text(errors="replace"), path.suffix)
        for pattern, what in (*NETWORK_VERBS, *AUDIO_PATTERNS):
            if pattern.search(body):
                offenders.append(f"{path.relative_to(REPO)}: {what}")
    assert not offenders, (
        "criterion 25: the log panel consumes the panel host (host.send / host.subscribe) and "
        "adds no network call and no audio element — the session log never leaves this box "
        "(docs/02 §9):\n" + "\n".join(offenders)
    )


# --- T-0054 group B: the fence panel -------------------------------------------------------------
#
# Criterion 17 names this module: "``hud/src/ui/fence.js`` and the new code in ``index.js`` add no
# ``fetch``, no ``<audio>``, no non-loopback URL and no client→server message". The scans above
# already cover ``hud/**``; what is added here is the two things that stop them passing
# *vacuously* over a panel that never landed, and the panel's own egress statement.

FENCE_REQUIRED = ("src/ui/fence.js",)


@pytest.mark.parametrize("relative", FENCE_REQUIRED)
def test_the_fence_panel_file_the_scan_is_supposed_to_cover_exists(relative):
    """Without this the scans above pass over a file that does not exist (T-0054 Owned files)."""
    assert (HUD / relative).is_file(), (
        f"T-0054 criterion 14 adds hud/{relative} — describeFence(message), DOM-free and "
        "clock-free (docs/tasks/T-0054.md, Owned files, group B)"
    )


def test_the_fence_panel_never_opens_a_transport_of_its_own_and_sends_nothing():
    """Criterion 17: no fetch, no second socket, no ``<audio>``, and **no egress at all**.

    The fence panel is unusual among the page's panels in having *nothing* to send: every other
    one declares a client→server whitelist, and this one's whitelist is empty. So the row is
    stated as an absence: a ``send(`` in this module would be the first message the page emits
    about a state the server told it about, which is the shape of a panel that has started
    deriving instead of echoing (D5).
    """
    path = HUD / "src" / "ui" / "fence.js"
    assert path.is_file(), "T-0054 criterion 14 adds hud/src/ui/fence.js"
    text = strip_comments(path.read_text(), ".js")
    offenders = [name for pattern, name in NETWORK_VERBS if pattern.search(text)]
    assert not offenders, (
        f"criterion 17: hud/src/ui/fence.js reaches for {offenders}; the page has one socket and "
        "it is the host's (docs/03 §9 E8, docs/02 §9)"
    )
    for banned in ("send(", "Audio(", "<audio", "new Audio"):
        assert banned not in text, (
            f"criterion 17: hud/src/ui/fence.js contains {banned!r}. The panel echoes confirmed "
            "server state and sends nothing; it plays nothing (T-0008 criterion 10)."
        )
