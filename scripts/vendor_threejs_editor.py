"""Vendor the three.js r179 editor into ``local_astra/pages/threejs/`` (T-0210 B0).

Usage (offline; the upstream tree is untrusted data, read only as bytes)::

    uv run python -I scripts/vendor_threejs_editor.py ~/threejs-r179

What it writes, and only under ``local_astra/pages/threejs/``:

- ``LICENSE`` verbatim;
- ``editor/**`` with a suffix ``pages/server.py`` serves, minus ``sw.js`` (B-R3: ``images/icon.xcf`` and
  ``docs/*.md`` are not vendored);
- ``build/three.module.js`` and ``build/three.core.js``;
- from ``examples/**`` exactly the static import closure of the vendored editor (classic ``<script src>`` of
  ``index.html`` and every static ``import``/``export ... from``, resolved through the importmap). Dynamic
  ``import()`` is not followed: the loaders and exporters behind File > Import/Export are not vendored;
- the named patches P1-P6 and P4b (``PATCHES``), each anchored on text that must occur exactly once upstream;
- ``PROVENANCE.md`` with one row per file.

The files that are ours (``editor/astra_state.js``, ``editor/astra_seeds.js``, ``astra_stubs/*.js``) are kept as
they are and hashed as ``ours``; every other file under the root is removed and rewritten.

Nothing from the upstream tree is executed, imported or handed to ``git``: the commit is read from ``.git/HEAD`` /
``packed-refs`` as text; symlinks are refused, and so is any path whose resolved location is outside the resolved
upstream root (B-R15). The file set is the static import closure (plus ``build/three.module.js``,
``build/three.core.js``); B10's zero-404 sweep over the copy is the evidence that it is enough (B-R16).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "local_astra" / "pages" / "threejs"
TAG = "r179"
COMMIT = "54ac263593c81b669ca9a089491ddd9e240427d2"
FETCHED = "2026-10-30"
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js", ".json", ".svg", ".png", ".ico"})
BUILD = ("build/three.module.js", "build/three.core.js")
OURS = ("editor/astra_state.js", "editor/astra_seeds.js")
STUB_DIR = "astra_stubs"

# --------------------------------------------------------------------------------------------- patches

_INDEX = "editor/index.html"
_HELP = "editor/js/Menubar.Help.js"
_TERN = "editor/js/libs/codemirror/addon/tern.js"

#: (file, patch id, upstream text, replacement). Each upstream text must occur exactly once.
PATCHES: tuple[tuple[str, str, str, str], ...] = (
    (
        _INDEX,
        "P1",
        '"three-gpu-pathtracer": "https://cdn.jsdelivr.net/npm/three-gpu-pathtracer@0.0.23/build/index.module.js",\n'
        '\t\t\t\t\t"three-mesh-bvh": "https://cdn.jsdelivr.net/npm/three-mesh-bvh@0.7.4/build/index.module.js"\n',
        '"three-gpu-pathtracer": "../astra_stubs/three-gpu-pathtracer.js"\n',
    ),
    (
        _INDEX,
        "P2",
        "\t\t\t// ServiceWorker\n\n"
        "\t\t\tif ( 'serviceWorker' in navigator ) {\n\n"
        "\t\t\t\ttry {\n\n"
        "\t\t\t\t\tnavigator.serviceWorker.register( 'sw.js' );\n\n"
        "\t\t\t\t} catch ( error ) {\n\n"
        "\t\t\t\t}\n\n"
        "\t\t\t}\n\n",
        "",
    ),
    (
        _INDEX,
        "P3",
        '\t\t<link rel="shortcut icon" href="../files/favicon_white.ico" media="(prefers-color-scheme: dark)"/>\n'
        '\t\t<link rel="shortcut icon" href="../files/favicon.ico" media="(prefers-color-scheme: light)" />\n',
        '\t\t<link rel="icon" href="images/icon.png">\n',
    ),
    (
        _INDEX,
        "P3",
        '\t\t<script src="https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.6/dist/ffmpeg.min.js"></script>\n\n',
        "",
    ),
    (
        _INDEX,
        "P5",
        "\t\t</script>\n\t</body>\n</html>",
        '\t\t</script>\n\t\t<script type="module" src="astra_state.js"></script>\n\t</body>\n</html>',
    ),
    (
        _HELP,
        "P4",
        "\t\twindow.open( 'https://github.com/mrdoob/three.js/tree/master/editor', '_blank' );\n",
        "\t\t// T-0210 P4: opens nothing\n",
    ),
    (
        _HELP,
        "P4",
        "\t\twindow.open( 'https://threejs.org', '_blank' );\n",
        "\t\t// T-0210 P4: opens nothing\n",
    ),
    (
        _HELP,
        "P4",
        "\t\twindow.open( 'https://github.com/mrdoob/three.js/wiki/Editor-Manual', '_blank' );\n",
        "\t\t// T-0210 P4: opens nothing\n",
    ),
    (
        _TERN,
        "P4b",
        '          child.href = data.url;\n          child.target = "_blank";\n',
        "          // T-0210 P4b: no href, so [docs] opens nothing\n",
    ),
    (
        _TERN,
        "P4b",
        "        if (!data.file && data.url) { window.open(data.url); return; }\n",
        "        if (!data.file && data.url) { return; } // T-0210 P4b: opens nothing\n",
    ),
    (
        _INDEX,
        "P6",
        "\t\t\t\t\tif ( isLoadingFromHash ) return;\n",
        "\t\t\t\t\tif ( isLoadingFromHash ) return;\n"
        "\t\t\t\t\tif ( new URLSearchParams( window.location.search ).has( 'seed' ) ) return; // T-0210 P6\n",
    ),
)

PATCH_TEXT = """\
- **P1** (H-T210-3): the off-origin importmap entry `three-gpu-pathtracer` (cdn.jsdelivr.net) points to the inert
  local stub `astra_stubs/three-gpu-pathtracer.js` (exports `WebGLPathTracer`, every method a no-op, so View >
  Realistic shading renders nothing new). The off-origin entry `three-mesh-bvh` (cdn.jsdelivr.net) is removed: no
  editor module imports it at r179 (upstream maps it for the path tracer's own bundle), so it needs no stub.
- **P2**: the service-worker registration (`navigator.serviceWorker.register( 'sw.js' )`) is removed from
  `editor/index.html`, and `editor/sw.js` is not vendored.
- **P3**: the off-origin `<script>` of ffmpeg (`@ffmpeg/ffmpeg@0.11.6/dist/ffmpeg.min.js`, cdn.jsdelivr.net; only
  Project > Video uses it) is removed; the two `../files/favicon_white.ico` / `../files/favicon.ico` favicon links
  (outside `editor/`, absent from the sparse tree) are replaced by one local `<link rel="icon" href="images/icon.png">`
  so Chromium requests no `/favicon.ico` (B-R7's intent; B-R7's literal `data:,` is a `scheme:` URL, which B1's
  "no absolute URL attribute" row refuses).
- **P4**: the three live `window.open( 'https://...' )` calls of Help (Source code, About, Manual) open nothing
  (`editor/js/Menubar.Help.js`). The version string `r179` at the right of the menubar is plain text, not a link.
- **P4b** (B-R14): the two off-origin link sites of the Script editor's tern addon
  (`editor/js/libs/codemirror/addon/tern.js`) open nothing: the type tooltip's `[docs]` anchor gets no `href` and no
  `target`, and jump-to-definition no longer calls `window.open(data.url)` for a definition with a URL and no file.
- **P5**: one added `<script type="module" src="astra_state.js"></script>` after the editor's inline module.
- **P6** (B-R13): in `editor/index.html` the callback of `editor.storage.get` returns before the IndexedDB autosave
  restore (`editor.fromJSON( state )`) and the saved selection (`editor.config.getKey( 'selected' )`) when the URL
  carries `?seed=`, so the seed the state shim applies is the whole scene on every load, with nothing selected.

Not patched, recorded:

- `editor/js/Sidebar.Project.App.js` writes a `https://threejs.org/editor/#file=` link into the *published app* it
  downloads (Project > Publish); the editor page never requests it.
- `examples/jsm/webxr/XRControllerModelFactory.js` has the off-origin `DEFAULT_PROFILES_PATH`
  (`cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/dist/profiles`); it is fetched only in an XR session (the
  editor's XR button), which headless Chromium on this copy never starts.
- `editor/js/Sidebar.Project.Image.js` calls `window.open( '', '_blank', ... )` to show a rendered image
  (Project > Image > Render): a blank same-process window, no URL. `editor/js/Sidebar.Project.Video.js` has the same
  call, behind Project > Video, which needs the removed ffmpeg script (P3).
"""

#: B2 at the pin, read from the vendored files and checked in headless Chromium 156 (loopback, T-0210 B).
LAYOUT_TEXT = """\
Each B2 claim, checked at r179 (files read; behaviour in headless Chromium 156, 1280x800, on loopback):

- Top menubar File, Edit, Add, View, Help (`Menubar.js`), with autosave (a checkbox) and the version `r179` on the
  right (`Menubar.Status.js`, class `menu right`): **true**.
- Add has hover submenus Mesh, Light, Camera (`Menubar.Add.js`, rows of class `option submenu-title`, shown on
  `mouseover`): **true**. The Light option that adds a point light is labelled "Point"; the object it adds is named
  "PointLight" (B-R2).
- A viewport `<canvas>` without `tabindex`, with a transform toolbar (`Toolbar.js`: translate / rotate / scale /
  local space) and a view-helper gizmo (`Viewport.ViewHelper.js`): **true**.
- A right sidebar with Scene / Project / Settings tabs, the outliner under Scene (`Sidebar.js`, `Sidebar.Scene.js`),
  and Properties tabs Object / Geometry / Material / Script (`Sidebar.Properties.js`): **true**.
- Object tab: Name (`UIInput`), Position / Rotation (degrees) / Scale (`UINumber`), Intensity for lights
  (`Sidebar.Object.js`): **true**. `UINumber` and `UIInput` are plain `<input>` elements (`libs/ui.js`); `UINumber`
  blurs on Enter and commits on `change`; a click without drag focuses it (`onMouseUp`: distance < 2 focuses):
  **true**.
- `window.editor` exists (set by the inline module of `index.html`): **true**.
- Delete uses `confirm()`: **false** at r179 (B-R1). Edit > Delete and the Delete/Backspace shortcut remove the
  selected object with no `confirm()` (`Menubar.Edit.js`, the document `keydown` handler in `Sidebar.Settings.Shortcuts.js`);
  Ctrl+Z undoes it. File > New > Empty (and the example projects, and Open) do call `confirm()`: **true**.
- A new box's Material tab shows `MeshStandardMaterial` with Metalness and Roughness (`Menubar.Add.js`,
  `Sidebar.Material.js`): **true**.
"""

# --------------------------------------------------------------------------------------------- reading


def upstream_commit(up: Path) -> str:
    """The clone's commit, read as text (git is never run in the untrusted tree)."""

    def text(p: Path) -> str:
        if not inside(up, p):
            raise SystemExit(f"refused: {p} resolves outside the upstream tree")
        return p.read_text(encoding="utf-8")

    head = text(up / ".git" / "HEAD").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:].strip()
    loose = up / ".git" / ref
    if inside(up, loose) and loose.is_file():
        return text(loose).strip()
    for line in text(up / ".git" / "packed-refs").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise SystemExit(f"cannot read the upstream commit from {up}/.git")


def inside(up: Path, p: Path) -> bool:
    """Whether ``p``'s resolved location is under the resolved upstream root ``up`` (B-R15: a symlinked directory
    anywhere on the path is caught, not only a symlinked leaf)."""
    return p.resolve().is_relative_to(up.resolve())


def read_bytes(up: Path, rel: str) -> bytes:
    p = up / rel
    if not inside(up, p):
        raise SystemExit(f"refused: {rel} resolves outside the upstream tree")
    if p.is_symlink() or not p.is_file():
        raise SystemExit(f"refused: {rel} is not a regular file in the upstream tree")
    return p.read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


IMPORTMAP = re.compile(r'(?is)<script\b[^>]*type\s*=\s*["\']importmap["\'][^>]*>(.*?)</script\s*>')
CLASSIC = re.compile(r'(?is)<script\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']')
IMPORT = re.compile(
    r"""(?:^|[;\s}])(?:import|export)\s*(?:[\w*{}\s,$]+?\s*from\s*)?["']([^"']+)["']""", re.M
)
COMMENT = re.compile(r"/\*.*?\*/|(?<![:\\'\"])//[^\n]*", re.S)
ABSOLUTE = re.compile(r"(?i)^\s*(?:[a-z][a-z0-9+.-]*:|//)")


def resolve(spec: str, base: str, imap: dict[str, str]) -> str | None:
    """A repo-relative path (``examples/jsm/x.js``), or ``None`` for off-origin or unmapped."""
    if ABSOLUTE.match(spec):
        return None
    if spec.startswith(("./", "../", "/")):
        return posixpath.normpath(posixpath.join(posixpath.dirname(base), spec)).lstrip("/")
    best = None
    for k in imap:
        if (spec == k or (k.endswith("/") and spec.startswith(k))) and (
            best is None or len(k) > len(best)
        ):
            best = k
    if best is None:
        return None
    target = imap[best] + spec[len(best) :] if best.endswith("/") else imap[best]
    if ABSOLUTE.match(target):
        return None
    return posixpath.normpath(posixpath.join("editor", target))


def examples_closure(up: Path, editor_js: dict[str, bytes], index: str) -> set[str]:
    """``examples/**`` files reached statically from the vendored editor (its classic scripts and its modules)."""
    m = IMPORTMAP.search(index)
    imap = {str(k): str(v) for k, v in json.loads(m.group(1))["imports"].items()} if m else {}
    want: set[str] = set()
    todo: list[tuple[str, str]] = []
    for src in CLASSIC.findall(index):
        r = resolve(src if src.startswith((".", "/")) else "./" + src, _INDEX, imap)
        if r is not None and r.startswith("examples/"):
            want.add(r)
    for rel, data in editor_js.items():
        todo.append((rel, data.decode("utf-8", errors="replace")))
    seen: set[str] = set()
    while todo:
        origin, text = todo.pop()
        for spec in IMPORT.findall(COMMENT.sub(" ", text)):
            r = resolve(spec, origin, imap)
            if r is None or not r.startswith("examples/") or r in seen:
                continue
            seen.add(r)
            want.add(r)
            todo.append((r, read_bytes(up, r).decode("utf-8", errors="replace")))
    return want


# --------------------------------------------------------------------------------------------- writing


def apply_patches(rel: str, data: bytes) -> tuple[bytes, list[str]]:
    if all(f != rel for f, _p, _o, _n in PATCHES):
        return data, []
    text = data.decode("utf-8")
    ids: list[str] = []
    for f, pid, old, new in PATCHES:
        if f != rel:
            continue
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"patch {pid} on {rel}: its upstream text occurs {n} times, not once")
        text = text.replace(old, new)
        ids.append(pid)
    return text.encode("utf-8"), ids


def collect(up: Path) -> dict[str, bytes]:
    """``{relative path: upstream bytes}`` of every file to vendor."""
    files: dict[str, bytes] = {"LICENSE": read_bytes(up, "LICENSE")}
    if not inside(up, up / "editor"):
        raise SystemExit("refused: editor/ resolves outside the upstream tree")
    for p in sorted((up / "editor").rglob("*")):
        if p.is_symlink() or not inside(up, p):
            raise SystemExit(f"refused: a symlink in the upstream tree: {p}")
        if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES or p.name == "sw.js":
            continue
        rel = p.relative_to(up).as_posix()
        files[rel] = read_bytes(up, rel)
    for rel in BUILD:
        files[rel] = read_bytes(up, rel)
    editor_js = {r: d for r, d in files.items() if r.startswith("editor/") and r.endswith(".js")}
    for rel in sorted(examples_closure(up, editor_js, files[_INDEX].decode("utf-8"))):
        files[rel] = read_bytes(up, rel)
    return files


def ours_files() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for rel in OURS:
        p = OUT / rel
        if not p.is_file():
            raise SystemExit(f"missing our file {p} (write it before vendoring)")
        out[rel] = p.read_bytes()
    stubs = OUT / STUB_DIR
    for p in sorted(stubs.glob("*.js")) if stubs.is_dir() else []:
        out[p.relative_to(OUT).as_posix()] = p.read_bytes()
    for name in ("three-gpu-pathtracer.js",):
        if f"{STUB_DIR}/{name}" not in out:
            raise SystemExit(f"missing stub {STUB_DIR}/{name}")
    return out


def provenance(rows: list[tuple[str, str, str, str]], licence: bytes, size: int) -> str:
    copyright_line = next(
        (ln.strip() for ln in licence.decode("utf-8").splitlines() if ln.startswith("Copyright")),
        "",
    )
    table = "\n".join(f"| `{p}` | {v} | {s} | {u} |" for p, v, s, u in rows)
    return f"""# three.js editor: provenance (T-0210 B0)

- **Source:** the three.js editor, tag **{TAG}**, upstream commit **`{COMMIT}`**
  (github.com/mrdoob/three.js; sparse clone of `editor`, `build`, `examples/jsm` and `LICENSE`).
- **Fetched:** {FETCHED}, by James (H-T210-6 alternative 2).
- **Licence:** MIT, `{copyright_line}` (`LICENSE`, verbatim). Our files (`ours` below) are Apache-2.0.
- **Vendored by:** `scripts/vendor_threejs_editor.py` (offline, `python -I`; the upstream tree is read as data only).
- **Contents:** `LICENSE`; `editor/**` with a served suffix minus `sw.js` (B-R3: `images/icon.xcf` and `docs/*.md` are
  not vendored); `build/three.module.js`, `build/three.core.js`; from `examples/**` exactly the static import closure
  of the editor (B-R16: B10's zero-404 sweep is the evidence that it is enough; the loaders and exporters behind
  File > Import/Export are dynamic `import()`s, not vendored, and 404 only on those over-actions); our state shim,
  seeds and stubs.
- **Size:** {size} bytes in total (B-R4 cap 12 x 10^6).

## Patches

{PATCH_TEXT}
## B2 layout at the pin

{LAYOUT_TEXT}
## Files

One row per file under this directory but this one: path, sha256 as vendored, `unmodified|patched|ours`, sha256
upstream at `{COMMIT[:8]}` (`—` for ours). `tests/unit/test_t0210_threejs_static.py` recomputes them.

| file | sha256 (vendored) | status | sha256 (upstream) |
|---|---|---|---|
{table}
"""


def vendor(up: Path) -> None:
    commit = upstream_commit(up)
    if commit != COMMIT:
        raise SystemExit(f"{up} is at {commit}, not the pinned {COMMIT}")
    upstream = collect(up)
    ours = ours_files()
    written: dict[str, bytes] = {}
    rows: list[tuple[str, str, str, str]] = []
    patched_ids: set[str] = set()
    for rel, data in upstream.items():
        out, ids = apply_patches(rel, data)
        patched_ids.update(ids)
        written[rel] = out
        status = "patched" if out != data else "unmodified"
        rows.append((rel, sha256(out), status, sha256(data)))
    missing = {pid for _f, pid, _o, _n in PATCHES} - patched_ids
    if missing:
        raise SystemExit(f"patches not applied: {sorted(missing)}")
    for rel, data in ours.items():
        rows.append((rel, sha256(data), "ours", "—"))
    rows.sort()
    # clear everything but our files, then write
    if OUT.exists():
        for p in sorted(OUT.rglob("*"), reverse=True):
            rel = p.relative_to(OUT).as_posix()
            if p.is_file() or p.is_symlink():
                if rel not in ours:
                    p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    for rel, data in written.items():
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    size = sum(len(d) for d in written.values()) + sum(len(d) for d in ours.values())
    text = provenance(rows, upstream["LICENSE"], 0)
    # the size line counts PROVENANCE.md itself; one fixed point step settles its digit count
    for _ in range(3):
        total = size + len(text.encode("utf-8"))
        text = provenance(rows, upstream["LICENSE"], total)
    (OUT / "PROVENANCE.md").write_text(text, encoding="utf-8")
    n_ex = sum(1 for r in written if r.startswith("examples/"))
    print(f"vendored {len(written)} upstream files ({n_ex} from examples/), {len(ours)} ours")
    print(f"root size: {size + len(text.encode('utf-8'))} bytes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("upstream", type=Path, help="the r179 tree (untrusted; read only)")
    args = ap.parse_args(argv)
    up = args.upstream.expanduser().resolve()
    if not (up / "editor" / "index.html").is_file():
        print(f"{up} holds no editor/index.html", file=sys.stderr)
        return 2
    vendor(up)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
