# three.js editor: provenance (T-0210 B0)

- **Source:** the three.js editor, tag **r179**, upstream commit **`54ac263593c81b669ca9a089491ddd9e240427d2`**
  (github.com/mrdoob/three.js; sparse clone of `editor`, `build`, `examples/jsm` and `LICENSE`).
- **Fetched:** 2026-10-30, by James (H-T210-6 alternative 2).
- **Licence:** MIT, `Copyright © 2010-2025 three.js authors` (`LICENSE`, verbatim). Our files (`ours` below) are Apache-2.0.
- **Vendored by:** `scripts/vendor_threejs_editor.py` (offline, `python -I`; the upstream tree is read as data only).
- **Contents:** `LICENSE`; `editor/**` with a served suffix minus `sw.js` (B-R3: `images/icon.xcf` and `docs/*.md` are
  not vendored); `build/three.module.js`, `build/three.core.js`; from `examples/**` exactly the static import closure
  of the editor (B-R16: B10's zero-404 sweep is the evidence that it is enough; the loaders and exporters behind
  File > Import/Export are dynamic `import()`s, not vendored, and 404 only on those over-actions); our state shim,
  seeds and stubs.
- **Size:** 5331836 bytes in total (B-R4 cap 12 x 10^6).

## Patches

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

## B2 layout at the pin

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

## Files

One row per file under this directory but this one: path, sha256 as vendored, `unmodified|patched|ours`, sha256
upstream at `54ac2635` (`—` for ours). `tests/unit/test_t0210_threejs_static.py` recomputes them.

| file | sha256 (vendored) | status | sha256 (upstream) |
|---|---|---|---|
| `LICENSE` | bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601 | unmodified | bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601 |
| `astra_stubs/three-gpu-pathtracer.js` | 5a6f4da0dc3c77c28b405e1fe24f367d2fa6f1824d17cfc7770419be6e122238 | ours | — |
| `build/three.core.js` | 3cd45beb4da9ce682bb7f5894dafd8a146379aa4b6b5aa3acd9ef5d47fdaaf92 | unmodified | 3cd45beb4da9ce682bb7f5894dafd8a146379aa4b6b5aa3acd9ef5d47fdaaf92 |
| `build/three.module.js` | ce6be0cb5ead0027e1f6094dd82ad43bba0886c03324a3b21fd9a33ac93fc2b4 | unmodified | ce6be0cb5ead0027e1f6094dd82ad43bba0886c03324a3b21fd9a33ac93fc2b4 |
| `editor/.eslintrc.json` | 0f2efc7247106794283acbe6afd2909c098370b0a11eabc199812b489cb74fd3 | unmodified | 0f2efc7247106794283acbe6afd2909c098370b0a11eabc199812b489cb74fd3 |
| `editor/astra_seeds.js` | b93d22ed997aaa308d40c51803c113ed8867a5c6b9a3000ed68ac7bb9247997c | ours | — |
| `editor/astra_state.js` | 51d2554fa89f49b345e5be5255ca7199f713998126a2e56f8521b3c800ea9321 | ours | — |
| `editor/css/main.css` | bfdda8d358873253c9ddfc5097edfed5c207b211914e09f77148ec739f2bf844 | unmodified | bfdda8d358873253c9ddfc5097edfed5c207b211914e09f77148ec739f2bf844 |
| `editor/examples/arkanoid.app.json` | 86d8752e13a9b45e024e3132c54a12bb0d0d8d860569f30c7ab56855645edab0 | unmodified | 86d8752e13a9b45e024e3132c54a12bb0d0d8d860569f30c7ab56855645edab0 |
| `editor/examples/camera.app.json` | be8794c481e8c4f4958c51858a95c487de230504162799a1e95fddda88e2fd26 | unmodified | be8794c481e8c4f4958c51858a95c487de230504162799a1e95fddda88e2fd26 |
| `editor/examples/particles.app.json` | 126fe938349cd23e0b9ac09c237b92bca6e00c467d9f8c0bddec79dbdbb9645b | unmodified | 126fe938349cd23e0b9ac09c237b92bca6e00c467d9f8c0bddec79dbdbb9645b |
| `editor/examples/pong.app.json` | 38077ef271d15c42ca046c4ba3536193d99850b25eec1be3a45a58db1d2a43ce | unmodified | 38077ef271d15c42ca046c4ba3536193d99850b25eec1be3a45a58db1d2a43ce |
| `editor/examples/shaders.app.json` | aa3b6c4b5838f593875fac272a7b373b7a9e212b354923af4fc2c0dd158e21f7 | unmodified | aa3b6c4b5838f593875fac272a7b373b7a9e212b354923af4fc2c0dd158e21f7 |
| `editor/images/icon.png` | cc687c7cf053dc7de107f756513fc56490cef091ac007b3a59532661317b1a42 | unmodified | cc687c7cf053dc7de107f756513fc56490cef091ac007b3a59532661317b1a42 |
| `editor/images/rotate.svg` | a277ca507781d73f86ad530ddd4d4cd9705e62bfd11f3a8a6d4e53515122b81e | unmodified | a277ca507781d73f86ad530ddd4d4cd9705e62bfd11f3a8a6d4e53515122b81e |
| `editor/images/scale.svg` | d25ce50c6ae9e9fbcb39d0d8ffda68043dba38c8947a0e012af35d062803c72d | unmodified | d25ce50c6ae9e9fbcb39d0d8ffda68043dba38c8947a0e012af35d062803c72d |
| `editor/images/translate.svg` | d930c96a43fd37ccbab3929dc10647c6b94c2e54bd643bb6abde0fc0a598f1b2 | unmodified | d930c96a43fd37ccbab3929dc10647c6b94c2e54bd643bb6abde0fc0a598f1b2 |
| `editor/index.html` | a29c7f820cdbc842fc0393a58577a1ef6e73e9ee9ffd4e7678e9fb7e1f2e58c5 | patched | 9f55109b87af061a2151a252c9c4c8a5e37658d7cb3b565cc1c9818ca4582fdf |
| `editor/js/Command.js` | bf433ed17d9a0e6cae32b27f37baa31c0c8ade2812447ecc948c9dd78440b921 | unmodified | bf433ed17d9a0e6cae32b27f37baa31c0c8ade2812447ecc948c9dd78440b921 |
| `editor/js/Config.js` | bcc29ef30372d9ff4b89ab0227e226967d3e44801ce0557ec0e1db508439c40e | unmodified | bcc29ef30372d9ff4b89ab0227e226967d3e44801ce0557ec0e1db508439c40e |
| `editor/js/Editor.js` | b4cd907b21a3dce2fd1df939fbece97ebab4132b16316d704ee38470672b424b | unmodified | b4cd907b21a3dce2fd1df939fbece97ebab4132b16316d704ee38470672b424b |
| `editor/js/EditorControls.js` | 59fb930afde290ffaed2ea86a701b7fcd6aaea54156012385dec1e588af98567 | unmodified | 59fb930afde290ffaed2ea86a701b7fcd6aaea54156012385dec1e588af98567 |
| `editor/js/History.js` | 13671b7af827bc01327b7b9c64573ffbea9ccc34ec8153f42bcf78760b0fd3fe | unmodified | 13671b7af827bc01327b7b9c64573ffbea9ccc34ec8153f42bcf78760b0fd3fe |
| `editor/js/Loader.js` | b1f8c2b1342be51e4b4d285c4567895a0a26dbccc458abaeb9d272d37f98c576 | unmodified | b1f8c2b1342be51e4b4d285c4567895a0a26dbccc458abaeb9d272d37f98c576 |
| `editor/js/LoaderUtils.js` | fff49dcaafa3f83bd528a4c2e3a33ad55c1d56ca54358d15091c9e48d7d05796 | unmodified | fff49dcaafa3f83bd528a4c2e3a33ad55c1d56ca54358d15091c9e48d7d05796 |
| `editor/js/Menubar.Add.js` | 9de2f15a4472cd882d396f2533d8fbf773649066b28b952a4426dbb75614f746 | unmodified | 9de2f15a4472cd882d396f2533d8fbf773649066b28b952a4426dbb75614f746 |
| `editor/js/Menubar.Edit.js` | 4073a29a5d65c8ae71f890c2449c9518fdf64ff8a4dea8c479c71d4fef7f31a8 | unmodified | 4073a29a5d65c8ae71f890c2449c9518fdf64ff8a4dea8c479c71d4fef7f31a8 |
| `editor/js/Menubar.File.js` | 926ad2fe028e59c99b4701598ef27c70c439c5a808a925aa90d9cd91230f0f61 | unmodified | 926ad2fe028e59c99b4701598ef27c70c439c5a808a925aa90d9cd91230f0f61 |
| `editor/js/Menubar.Help.js` | d84677975e01b8a74520242238e2c34ab6afc2b0590edfc012566c173540b5c2 | patched | 30fb716a73c3be7d33e353b03abc385fd0fa17d6ca845c97578ceb33452705ed |
| `editor/js/Menubar.Status.js` | ceaceb7138b34052b7087d0b29afbed3ec6fc886763bf799c907868822f7395a | unmodified | ceaceb7138b34052b7087d0b29afbed3ec6fc886763bf799c907868822f7395a |
| `editor/js/Menubar.View.js` | e57118214cf24e12e227963f36621168cd12a8397792bb5fbe81b6bbb297dd8e | unmodified | e57118214cf24e12e227963f36621168cd12a8397792bb5fbe81b6bbb297dd8e |
| `editor/js/Menubar.js` | 77e778d211cf6a31abe3cdfb42c40526b16f0e90e0672c90a7c58131f48aade3 | unmodified | 77e778d211cf6a31abe3cdfb42c40526b16f0e90e0672c90a7c58131f48aade3 |
| `editor/js/Player.js` | dd63eda0b1ce388f2a57f455789eb8681b55cfeb38c8257339685ee21c75d478 | unmodified | dd63eda0b1ce388f2a57f455789eb8681b55cfeb38c8257339685ee21c75d478 |
| `editor/js/Resizer.js` | 9af4ead0d6e7b513ac1654d6c5d91c57f6a01fdf790f11bc58ad019c27c13047 | unmodified | 9af4ead0d6e7b513ac1654d6c5d91c57f6a01fdf790f11bc58ad019c27c13047 |
| `editor/js/Script.js` | 5e37837d0bcf5014feeee612c7ddc4495c50d5cd742e1a6385bb3ebee61e7aa0 | unmodified | 5e37837d0bcf5014feeee612c7ddc4495c50d5cd742e1a6385bb3ebee61e7aa0 |
| `editor/js/Selector.js` | d9e0a44efe66f69d5feb4e2f4504cbd913a62ebbf4f843cb29b12d6591735bb8 | unmodified | d9e0a44efe66f69d5feb4e2f4504cbd913a62ebbf4f843cb29b12d6591735bb8 |
| `editor/js/Sidebar.Geometry.BoxGeometry.js` | 36fcab719c13bb42914410584e557efb11fbea2d9d22abb6a1c2b924d1cad674 | unmodified | 36fcab719c13bb42914410584e557efb11fbea2d9d22abb6a1c2b924d1cad674 |
| `editor/js/Sidebar.Geometry.BufferGeometry.js` | e36b3089869eee1709c11b5df5c639ca581b03e64afbfe1b7f011d0bc59579b1 | unmodified | e36b3089869eee1709c11b5df5c639ca581b03e64afbfe1b7f011d0bc59579b1 |
| `editor/js/Sidebar.Geometry.CapsuleGeometry.js` | 68d955dc46138001c236c4eee048a57ecfe20b9f309e49bb65671fdb29d62d58 | unmodified | 68d955dc46138001c236c4eee048a57ecfe20b9f309e49bb65671fdb29d62d58 |
| `editor/js/Sidebar.Geometry.CircleGeometry.js` | 0dfea94fd83b221abf425ad292e1089d8d34c87589c2fbb11c13f357dc9e48a7 | unmodified | 0dfea94fd83b221abf425ad292e1089d8d34c87589c2fbb11c13f357dc9e48a7 |
| `editor/js/Sidebar.Geometry.CylinderGeometry.js` | afbe9fa117a880e4fdaffa503975ce2c5d39062aba77842b77d5b1540a737b1a | unmodified | afbe9fa117a880e4fdaffa503975ce2c5d39062aba77842b77d5b1540a737b1a |
| `editor/js/Sidebar.Geometry.DodecahedronGeometry.js` | 5757adbac747d296aef04f02983b33739d082122afd7be27ec066d170ec8fcdb | unmodified | 5757adbac747d296aef04f02983b33739d082122afd7be27ec066d170ec8fcdb |
| `editor/js/Sidebar.Geometry.ExtrudeGeometry.js` | 660357cd21ea55844f42da4c1823ac1c824435443bf30d4efd28a3a2ff34570c | unmodified | 660357cd21ea55844f42da4c1823ac1c824435443bf30d4efd28a3a2ff34570c |
| `editor/js/Sidebar.Geometry.IcosahedronGeometry.js` | 229117e42290c144c41a7ef98ad61b3cc16a4b0f785a4da8307da1fc52924dc4 | unmodified | 229117e42290c144c41a7ef98ad61b3cc16a4b0f785a4da8307da1fc52924dc4 |
| `editor/js/Sidebar.Geometry.LatheGeometry.js` | e9c1d5a0bbeff6e6c8d65146e2ed1d13880a92dc30593df830631bac5b6cf8ab | unmodified | e9c1d5a0bbeff6e6c8d65146e2ed1d13880a92dc30593df830631bac5b6cf8ab |
| `editor/js/Sidebar.Geometry.Modifiers.js` | 7fc5c75fd0c4c2c6e4add139f720f9dec936b712e9c41439ca483201e1e87edd | unmodified | 7fc5c75fd0c4c2c6e4add139f720f9dec936b712e9c41439ca483201e1e87edd |
| `editor/js/Sidebar.Geometry.OctahedronGeometry.js` | 7cc8e2a983bee58908550e0e150ab6d9d1c87ee62771fe4888ef00fdb7d370c4 | unmodified | 7cc8e2a983bee58908550e0e150ab6d9d1c87ee62771fe4888ef00fdb7d370c4 |
| `editor/js/Sidebar.Geometry.PlaneGeometry.js` | 4c21801ae98160b1b7f9ffcda26e34fbb21ea2e917a57da0646b56cc92e8978d | unmodified | 4c21801ae98160b1b7f9ffcda26e34fbb21ea2e917a57da0646b56cc92e8978d |
| `editor/js/Sidebar.Geometry.RingGeometry.js` | e467f046f144ea0858b3a1df7b101358a0e643fce0bfc949ffbb42c6302b0058 | unmodified | e467f046f144ea0858b3a1df7b101358a0e643fce0bfc949ffbb42c6302b0058 |
| `editor/js/Sidebar.Geometry.ShapeGeometry.js` | 0839f005d8d2ff2004d825a405c3db0892f33f250df2a0a015c9d73b16d33084 | unmodified | 0839f005d8d2ff2004d825a405c3db0892f33f250df2a0a015c9d73b16d33084 |
| `editor/js/Sidebar.Geometry.SphereGeometry.js` | a542beeb9dfe01afd4acfd703e3ef5e86d5d3afa5357a0975a167c5b1ee479d5 | unmodified | a542beeb9dfe01afd4acfd703e3ef5e86d5d3afa5357a0975a167c5b1ee479d5 |
| `editor/js/Sidebar.Geometry.TetrahedronGeometry.js` | 3c28c738df7461e691159dab645b9146c1ded09106efa3b5aeee94363e8f7be6 | unmodified | 3c28c738df7461e691159dab645b9146c1ded09106efa3b5aeee94363e8f7be6 |
| `editor/js/Sidebar.Geometry.TorusGeometry.js` | 920a7c1255dd670b6cf63cf2ca595a34c777acc68795ac5b1d2381ebb3198eee | unmodified | 920a7c1255dd670b6cf63cf2ca595a34c777acc68795ac5b1d2381ebb3198eee |
| `editor/js/Sidebar.Geometry.TorusKnotGeometry.js` | 21ec79ed0fa1810c386f9a37fbd609fc13a60add9b0fa6e53c7fe275c0e9b053 | unmodified | 21ec79ed0fa1810c386f9a37fbd609fc13a60add9b0fa6e53c7fe275c0e9b053 |
| `editor/js/Sidebar.Geometry.TubeGeometry.js` | 70bdfa49dcbb23268f01af77218b6e64074765f93c4ccd1053154a080e2da716 | unmodified | 70bdfa49dcbb23268f01af77218b6e64074765f93c4ccd1053154a080e2da716 |
| `editor/js/Sidebar.Geometry.js` | 82e9368340154c383c73156cf77a2e923785172e58a7248058ced5ea18c5cf37 | unmodified | 82e9368340154c383c73156cf77a2e923785172e58a7248058ced5ea18c5cf37 |
| `editor/js/Sidebar.Material.BooleanProperty.js` | 53ab07f857e63835be84ad41796cf623a76011a719e967715ff5e88bb104dc7c | unmodified | 53ab07f857e63835be84ad41796cf623a76011a719e967715ff5e88bb104dc7c |
| `editor/js/Sidebar.Material.ColorProperty.js` | ecf7686b2c599bb4cc23a8f30ba860abe022160010a8f580a36101c7b4b1f564 | unmodified | ecf7686b2c599bb4cc23a8f30ba860abe022160010a8f580a36101c7b4b1f564 |
| `editor/js/Sidebar.Material.ConstantProperty.js` | dbf3f6dc154ede00ed6f85821fd415638b5ff719f861325d19dac273c666a0fe | unmodified | dbf3f6dc154ede00ed6f85821fd415638b5ff719f861325d19dac273c666a0fe |
| `editor/js/Sidebar.Material.MapProperty.js` | 83dba32c3a47eae3d3112a76af352a5954b1f7592ac9df8bd31f21afdd883f69 | unmodified | 83dba32c3a47eae3d3112a76af352a5954b1f7592ac9df8bd31f21afdd883f69 |
| `editor/js/Sidebar.Material.NumberProperty.js` | 32ae67e15463ec75a021630f6bf4133d444b0902ffebf0157ae2eca8dfceb6ac | unmodified | 32ae67e15463ec75a021630f6bf4133d444b0902ffebf0157ae2eca8dfceb6ac |
| `editor/js/Sidebar.Material.Program.js` | 438906ce458aad4b10cdb773560c762055ceb20f95bb241866f2ddef7d9db521 | unmodified | 438906ce458aad4b10cdb773560c762055ceb20f95bb241866f2ddef7d9db521 |
| `editor/js/Sidebar.Material.RangeValueProperty.js` | 9e1f45bbeff704cfcd7ff9a63b1a7934473175c53a28aa2c8abf1c3cd68de16d | unmodified | 9e1f45bbeff704cfcd7ff9a63b1a7934473175c53a28aa2c8abf1c3cd68de16d |
| `editor/js/Sidebar.Material.js` | 0b4f033bf8ca1dc18b3a2297e96a8d72a42640f71a9c64d9219f896000267642 | unmodified | 0b4f033bf8ca1dc18b3a2297e96a8d72a42640f71a9c64d9219f896000267642 |
| `editor/js/Sidebar.Object.Animation.js` | c849cbdfd363d516b6ceb978da965cfb2ec892a3372120f578a39b89dae7f400 | unmodified | c849cbdfd363d516b6ceb978da965cfb2ec892a3372120f578a39b89dae7f400 |
| `editor/js/Sidebar.Object.js` | c85e0b63e476b3ea3c42b48d6ef4473a3b341cd511e82ce6f51718eb8164a985 | unmodified | c85e0b63e476b3ea3c42b48d6ef4473a3b341cd511e82ce6f51718eb8164a985 |
| `editor/js/Sidebar.Project.App.js` | 31186801783a76e531fe70a2f3f0cab1c6806e6cdb8a2663151eda5a9a6678e0 | unmodified | 31186801783a76e531fe70a2f3f0cab1c6806e6cdb8a2663151eda5a9a6678e0 |
| `editor/js/Sidebar.Project.Image.js` | eab499e6138b50a236bb09c72f77f41ebf41ca417f2022d440ea85a6bcb3b770 | unmodified | eab499e6138b50a236bb09c72f77f41ebf41ca417f2022d440ea85a6bcb3b770 |
| `editor/js/Sidebar.Project.Materials.js` | b6ee7e2d23e0524aadd65fadbe8477075e45bdd2274009977e3c58bec10f8745 | unmodified | b6ee7e2d23e0524aadd65fadbe8477075e45bdd2274009977e3c58bec10f8745 |
| `editor/js/Sidebar.Project.Renderer.js` | 23c281f13bde7d7a29a681db6311fabb1eee52954d2a0df13a50dab50c257e0c | unmodified | 23c281f13bde7d7a29a681db6311fabb1eee52954d2a0df13a50dab50c257e0c |
| `editor/js/Sidebar.Project.Video.js` | 2ef2daaa7dda3127f58bf62225d9b13b84a6ec703333b9e0da5feb1f46db6843 | unmodified | 2ef2daaa7dda3127f58bf62225d9b13b84a6ec703333b9e0da5feb1f46db6843 |
| `editor/js/Sidebar.Project.js` | 34e98cd62958319ff641dc5d0c1a7aeb2d7dde442f9713ae9c2a89de2c49cef0 | unmodified | 34e98cd62958319ff641dc5d0c1a7aeb2d7dde442f9713ae9c2a89de2c49cef0 |
| `editor/js/Sidebar.Properties.js` | b423bde0346347c685cdf6e9abb6efb9a6ac90d2241549310768c25a8ee71f97 | unmodified | b423bde0346347c685cdf6e9abb6efb9a6ac90d2241549310768c25a8ee71f97 |
| `editor/js/Sidebar.Scene.js` | 6311d0685943a776f0ff7921169586b27a8f2c9ab5ddb801592b4ac3625b305c | unmodified | 6311d0685943a776f0ff7921169586b27a8f2c9ab5ddb801592b4ac3625b305c |
| `editor/js/Sidebar.Script.js` | b7193c06678fa64db5c71ba01f3cc3b8affb61e5eff5134ab4c5b5e0c0fbce13 | unmodified | b7193c06678fa64db5c71ba01f3cc3b8affb61e5eff5134ab4c5b5e0c0fbce13 |
| `editor/js/Sidebar.Settings.History.js` | 5d501370a086c7e0c5bf231ba71869fe3e6bb2f528ec7ebbabda23f5c4d1d972 | unmodified | 5d501370a086c7e0c5bf231ba71869fe3e6bb2f528ec7ebbabda23f5c4d1d972 |
| `editor/js/Sidebar.Settings.Shortcuts.js` | c554c7e08f54b5c53bc852b3267d15f8d30ad5a7cab169a5d1a55ca5d5642385 | unmodified | c554c7e08f54b5c53bc852b3267d15f8d30ad5a7cab169a5d1a55ca5d5642385 |
| `editor/js/Sidebar.Settings.js` | d7b406a92a663afa6e4b6b2f79df964a94b356d31c7ac997119626469bb65f57 | unmodified | d7b406a92a663afa6e4b6b2f79df964a94b356d31c7ac997119626469bb65f57 |
| `editor/js/Sidebar.js` | d5988f9df4ab30d01625ef2af260952b63916f70aaeeaf52e9033ed0bf33f9f2 | unmodified | d5988f9df4ab30d01625ef2af260952b63916f70aaeeaf52e9033ed0bf33f9f2 |
| `editor/js/Storage.js` | df0d4757404c7c346c1ef3b254cfeb5706502d2c49e2980e728ef5b7d8956eb8 | unmodified | df0d4757404c7c346c1ef3b254cfeb5706502d2c49e2980e728ef5b7d8956eb8 |
| `editor/js/Strings.js` | dcb4c87b87dce41c7a6033554720a977f023875a6a97d97c62bd18b0545b08dc | unmodified | dcb4c87b87dce41c7a6033554720a977f023875a6a97d97c62bd18b0545b08dc |
| `editor/js/Toolbar.js` | 679872874b06580fbc7aee21e3fee157c5f40b82104000d43eafbc928ab44f7d | unmodified | 679872874b06580fbc7aee21e3fee157c5f40b82104000d43eafbc928ab44f7d |
| `editor/js/Viewport.Controls.js` | 3322db37ba67d9e568142506caecbcb0229b8e63ad0f4523ca7004a3bd64f9fe | unmodified | 3322db37ba67d9e568142506caecbcb0229b8e63ad0f4523ca7004a3bd64f9fe |
| `editor/js/Viewport.Info.js` | 401e8c5f8d6aa30a349dd6e6e189ee9fc84249299fa0062134bc3c90e8de76df | unmodified | 401e8c5f8d6aa30a349dd6e6e189ee9fc84249299fa0062134bc3c90e8de76df |
| `editor/js/Viewport.Pathtracer.js` | fb6db5372b2ddff88e86289aeb3c27c76ac0dfca14edd58acd923ecfd6b83965 | unmodified | fb6db5372b2ddff88e86289aeb3c27c76ac0dfca14edd58acd923ecfd6b83965 |
| `editor/js/Viewport.ViewHelper.js` | c79bf98897f3b27d9bb850879c2f50178d3f23a5a6b2d262380f0d426de5fa22 | unmodified | c79bf98897f3b27d9bb850879c2f50178d3f23a5a6b2d262380f0d426de5fa22 |
| `editor/js/Viewport.XR.js` | 7d733eb1c542972161cfe1d7fb0514ac6c48c2ff625a11ba83348c53a2844462 | unmodified | 7d733eb1c542972161cfe1d7fb0514ac6c48c2ff625a11ba83348c53a2844462 |
| `editor/js/Viewport.js` | 61021bde8eaf84f4534a3cb08d33f933e778d40248e240906f78aa955cc1a6e0 | unmodified | 61021bde8eaf84f4534a3cb08d33f933e778d40248e240906f78aa955cc1a6e0 |
| `editor/js/commands/AddObjectCommand.js` | b84c4e5fa1731216b55889c7ab1cf23d011e8910bf3bba015eb7ee64dc7df733 | unmodified | b84c4e5fa1731216b55889c7ab1cf23d011e8910bf3bba015eb7ee64dc7df733 |
| `editor/js/commands/AddScriptCommand.js` | 0994dc0f1f40f4aa4750450a5879e5711f1b9efedbff458692e16e594ea85333 | unmodified | 0994dc0f1f40f4aa4750450a5879e5711f1b9efedbff458692e16e594ea85333 |
| `editor/js/commands/Commands.js` | a8d44969ee0e0d759b415d73e0379f483a52af4e0c8076770917a87a0f20498c | unmodified | a8d44969ee0e0d759b415d73e0379f483a52af4e0c8076770917a87a0f20498c |
| `editor/js/commands/MoveObjectCommand.js` | eefffbbeb7bc5ac892aad2d2d451eb199ad68277c67f902e909c5a6a0e32ff73 | unmodified | eefffbbeb7bc5ac892aad2d2d451eb199ad68277c67f902e909c5a6a0e32ff73 |
| `editor/js/commands/MultiCmdsCommand.js` | 6eacfdde05f8353169a8d9cb0ee97af8bad9354c1becf3b6d7279d724d9f8051 | unmodified | 6eacfdde05f8353169a8d9cb0ee97af8bad9354c1becf3b6d7279d724d9f8051 |
| `editor/js/commands/RemoveObjectCommand.js` | a34c834f42995353aa7015138f9553f507e959d911c3ae8569fad94c7d5d76d9 | unmodified | a34c834f42995353aa7015138f9553f507e959d911c3ae8569fad94c7d5d76d9 |
| `editor/js/commands/RemoveScriptCommand.js` | 0c7c5333c2b0659f751bb0efa219474cfb231db509460012d2469d8022085928 | unmodified | 0c7c5333c2b0659f751bb0efa219474cfb231db509460012d2469d8022085928 |
| `editor/js/commands/SetColorCommand.js` | 62e736e745a6afde24956764b629e40f7ad4602d7c1ce2e7c696dcf7f614011e | unmodified | 62e736e745a6afde24956764b629e40f7ad4602d7c1ce2e7c696dcf7f614011e |
| `editor/js/commands/SetGeometryCommand.js` | 706590f7e00ac128288ec6b9dbb74724f4b958f7a8816a48cebf2e07632be676 | unmodified | 706590f7e00ac128288ec6b9dbb74724f4b958f7a8816a48cebf2e07632be676 |
| `editor/js/commands/SetGeometryValueCommand.js` | 69d8547ba1fd38e81c14a0b9bf9e804095e7fd17d920636bf0ed57758436c9e5 | unmodified | 69d8547ba1fd38e81c14a0b9bf9e804095e7fd17d920636bf0ed57758436c9e5 |
| `editor/js/commands/SetMaterialColorCommand.js` | d0fe7f05dd4775e9343d8e59b4e8d512ce203e6c155737caa8feb00435fbb18b | unmodified | d0fe7f05dd4775e9343d8e59b4e8d512ce203e6c155737caa8feb00435fbb18b |
| `editor/js/commands/SetMaterialCommand.js` | 77fc3541d14f0cff5555509a60efd0cec1e39cb29300a4167afccd2d8558036f | unmodified | 77fc3541d14f0cff5555509a60efd0cec1e39cb29300a4167afccd2d8558036f |
| `editor/js/commands/SetMaterialMapCommand.js` | 5b996bceefafc1eec38d360d5ce9dde4845d5dece4b716f7c459a05b48f99045 | unmodified | 5b996bceefafc1eec38d360d5ce9dde4845d5dece4b716f7c459a05b48f99045 |
| `editor/js/commands/SetMaterialRangeCommand.js` | 63757cede6167e71aaf6e2470d85aa29aa6abd272b7be10ffca0e7d27cea0dc4 | unmodified | 63757cede6167e71aaf6e2470d85aa29aa6abd272b7be10ffca0e7d27cea0dc4 |
| `editor/js/commands/SetMaterialValueCommand.js` | 4755d152cb200b9d19064ab75d77f15c80993fa240a2d7a5bc5dabf4fdb6e517 | unmodified | 4755d152cb200b9d19064ab75d77f15c80993fa240a2d7a5bc5dabf4fdb6e517 |
| `editor/js/commands/SetMaterialVectorCommand.js` | 041e73c64e6bf2727b7428fb11f294e8642c2782863fd7e45dfcbe7cb0f5d8b7 | unmodified | 041e73c64e6bf2727b7428fb11f294e8642c2782863fd7e45dfcbe7cb0f5d8b7 |
| `editor/js/commands/SetPositionCommand.js` | 2c5812855e0109d09c724f1c63444604161c8e8fe1e690b88aa61b7c4e0ee2be | unmodified | 2c5812855e0109d09c724f1c63444604161c8e8fe1e690b88aa61b7c4e0ee2be |
| `editor/js/commands/SetRotationCommand.js` | 530bf35ef3b3c037e6b4ba37f62643332bb0e841f6cc159e563d50f5cf5ec710 | unmodified | 530bf35ef3b3c037e6b4ba37f62643332bb0e841f6cc159e563d50f5cf5ec710 |
| `editor/js/commands/SetScaleCommand.js` | 7263ce59626fa91d2279477be821f4c5a13522471a77b80269cf9b8acd108476 | unmodified | 7263ce59626fa91d2279477be821f4c5a13522471a77b80269cf9b8acd108476 |
| `editor/js/commands/SetSceneCommand.js` | 8f9988bd1f07576021e4d78b805e78b845328a858474842e3c1f5d0d96a75fef | unmodified | 8f9988bd1f07576021e4d78b805e78b845328a858474842e3c1f5d0d96a75fef |
| `editor/js/commands/SetScriptValueCommand.js` | bdc8c511454cea5a6b4dd7658c5e065f74cc29bae213ec8981078c29cc8fa6ae | unmodified | bdc8c511454cea5a6b4dd7658c5e065f74cc29bae213ec8981078c29cc8fa6ae |
| `editor/js/commands/SetShadowValueCommand.js` | cb5678e169aeb865050175ff4425df722082c8ef73d2185a74756198336422ff | unmodified | cb5678e169aeb865050175ff4425df722082c8ef73d2185a74756198336422ff |
| `editor/js/commands/SetUuidCommand.js` | 3ac89671213e6fc232000e1d47caa1d53821645cc40331e83ee34fd66c35ff34 | unmodified | 3ac89671213e6fc232000e1d47caa1d53821645cc40331e83ee34fd66c35ff34 |
| `editor/js/commands/SetValueCommand.js` | 8ff548c4e083a848c5a72a8b02749c01bb95b281b4c2a2731823e2995b0294d0 | unmodified | 8ff548c4e083a848c5a72a8b02749c01bb95b281b4c2a2731823e2995b0294d0 |
| `editor/js/libs/acorn/acorn.js` | 5adfb18b15c5eb96a51e7e49fdc888ce5a99dd18b305f87a244aeaea4d5aa629 | unmodified | 5adfb18b15c5eb96a51e7e49fdc888ce5a99dd18b305f87a244aeaea4d5aa629 |
| `editor/js/libs/acorn/acorn_loose.js` | f33b1b1e8ba02c7f78656bb66f7ee70d702cd1c94195511430866f41edb9d02d | unmodified | f33b1b1e8ba02c7f78656bb66f7ee70d702cd1c94195511430866f41edb9d02d |
| `editor/js/libs/acorn/walk.js` | 6f6175a2a24ca586bc81438df7035bb3863ad82694833ee449a164f99c059e72 | unmodified | 6f6175a2a24ca586bc81438df7035bb3863ad82694833ee449a164f99c059e72 |
| `editor/js/libs/app.js` | a5c9349fe1f25c98c123758ff33863db807d8e51722c39d22dcf4e313e9a606b | unmodified | a5c9349fe1f25c98c123758ff33863db807d8e51722c39d22dcf4e313e9a606b |
| `editor/js/libs/app/index.html` | 0b36a3a11eba7ad76c0154a56a03ea5a10eedd0df02b3666c25c9db870b1953a | unmodified | 0b36a3a11eba7ad76c0154a56a03ea5a10eedd0df02b3666c25c9db870b1953a |
| `editor/js/libs/codemirror/addon/dialog.css` | 5df690d771f12118348562dd2a90060433ae2edeccd0908abcaa4480b1e3e468 | unmodified | 5df690d771f12118348562dd2a90060433ae2edeccd0908abcaa4480b1e3e468 |
| `editor/js/libs/codemirror/addon/dialog.js` | f823db8445973fc7be842dbe5278211e8971e7ab9ffd8ce9c403c68171fbe31d | unmodified | f823db8445973fc7be842dbe5278211e8971e7ab9ffd8ce9c403c68171fbe31d |
| `editor/js/libs/codemirror/addon/show-hint.css` | f4f8c8d48c155dcbf8e04dfb29b0d500974460d08204adfe0dda3be7bb3a2fc8 | unmodified | f4f8c8d48c155dcbf8e04dfb29b0d500974460d08204adfe0dda3be7bb3a2fc8 |
| `editor/js/libs/codemirror/addon/show-hint.js` | be880d5b21820a163127e5adfe008582fad272c413bd260ea019157726799d32 | unmodified | be880d5b21820a163127e5adfe008582fad272c413bd260ea019157726799d32 |
| `editor/js/libs/codemirror/addon/tern.css` | 107542ab314a1db5eb9ee84f7f2543f9fcf7e03e472a35edef2c5fd84fa308f7 | unmodified | 107542ab314a1db5eb9ee84f7f2543f9fcf7e03e472a35edef2c5fd84fa308f7 |
| `editor/js/libs/codemirror/addon/tern.js` | 1700edc0fea7057affbb6b5277d88b3dfc70018d7e35915b6280aeb972c0e6a1 | patched | bb690874b56764c6c65b13e35d78a5d97cfee5d3e5814622449bb5d3cc5498aa |
| `editor/js/libs/codemirror/codemirror.css` | eb494ea972d2661ef86f7f6ac656dd6786d721e49c9c1b46e1eb967e4b6f9bf3 | unmodified | eb494ea972d2661ef86f7f6ac656dd6786d721e49c9c1b46e1eb967e4b6f9bf3 |
| `editor/js/libs/codemirror/codemirror.js` | d03e9fa93fd33399bb99b98e2956cd2e799a56775663796a42b74af4ee1119b0 | unmodified | d03e9fa93fd33399bb99b98e2956cd2e799a56775663796a42b74af4ee1119b0 |
| `editor/js/libs/codemirror/mode/glsl.js` | 4183b8ea409ea5b52b2f22121d2e95db3bdc8bd23020e975ab10a7c055ac3d55 | unmodified | 4183b8ea409ea5b52b2f22121d2e95db3bdc8bd23020e975ab10a7c055ac3d55 |
| `editor/js/libs/codemirror/mode/javascript.js` | ea968e9ea578b68526edd3c113391c6558886ec5942cedb67305fb5cb9456bdf | unmodified | ea968e9ea578b68526edd3c113391c6558886ec5942cedb67305fb5cb9456bdf |
| `editor/js/libs/codemirror/theme/monokai.css` | 2463dc6fd92019a0c7ca2aaa01d03116b280fa7c6ae01bf21df7c107d9b683e8 | unmodified | 2463dc6fd92019a0c7ca2aaa01d03116b280fa7c6ae01bf21df7c107d9b683e8 |
| `editor/js/libs/esprima.js` | 0d3811557704a646fd2012b3c3a463881997dd24e3a0049647862fbf38a489a3 | unmodified | 0d3811557704a646fd2012b3c3a463881997dd24e3a0049647862fbf38a489a3 |
| `editor/js/libs/jsonlint.js` | ce94081be12d9294917590c5cbfab3204141d293573ed1baeb0ad587132509dd | unmodified | ce94081be12d9294917590c5cbfab3204141d293573ed1baeb0ad587132509dd |
| `editor/js/libs/signals.min.js` | ee17ae5ef0aa629eaa4a728ba672f3fe59706eb8ae640e5cb3ede7ffe5da6e60 | unmodified | ee17ae5ef0aa629eaa4a728ba672f3fe59706eb8ae640e5cb3ede7ffe5da6e60 |
| `editor/js/libs/tern-threejs/threejs.js` | 222c3663b4a5196afd33026ea6a3f612c7f438e45e871b74d257a03317f3131b | unmodified | 222c3663b4a5196afd33026ea6a3f612c7f438e45e871b74d257a03317f3131b |
| `editor/js/libs/ternjs/comment.js` | c8d1b69fad77e17ff7b4779f3a68ba82069558c3f675aa31a4b10351598d188d | unmodified | c8d1b69fad77e17ff7b4779f3a68ba82069558c3f675aa31a4b10351598d188d |
| `editor/js/libs/ternjs/def.js` | 267ecef7d1754d54898e65db32d4f9353303f5f9d8bfa1e39e37e8ebf325c142 | unmodified | 267ecef7d1754d54898e65db32d4f9353303f5f9d8bfa1e39e37e8ebf325c142 |
| `editor/js/libs/ternjs/doc_comment.js` | 16b0ae32d53c149330c16408423e30611dca1ae149ad525e153643ae989723b8 | unmodified | 16b0ae32d53c149330c16408423e30611dca1ae149ad525e153643ae989723b8 |
| `editor/js/libs/ternjs/infer.js` | 507ac3b413203d0970ca12dfcee3ee906bdce0d3ffb5c2bd7b6b66964ce133ef | unmodified | 507ac3b413203d0970ca12dfcee3ee906bdce0d3ffb5c2bd7b6b66964ce133ef |
| `editor/js/libs/ternjs/polyfill.js` | 901346fec4db7bdda8f7bfea57292ec16b43b299fbfd6de73290f2b14a4fc1a7 | unmodified | 901346fec4db7bdda8f7bfea57292ec16b43b299fbfd6de73290f2b14a4fc1a7 |
| `editor/js/libs/ternjs/signal.js` | 758cee05964007c5294c14226105806888182d25e39fefbd923357f12e691032 | unmodified | 758cee05964007c5294c14226105806888182d25e39fefbd923357f12e691032 |
| `editor/js/libs/ternjs/tern.js` | acc6634f628164b829b962e6ab0f2491fd09eda6ed184dbe4d1aa7836ba038bb | unmodified | acc6634f628164b829b962e6ab0f2491fd09eda6ed184dbe4d1aa7836ba038bb |
| `editor/js/libs/ui.js` | ecc365b6073b130b61f70294278bd6b0f2f49b0c24533b6e233beb27ce7c5c59 | unmodified | ecc365b6073b130b61f70294278bd6b0f2f49b0c24533b6e233beb27ce7c5c59 |
| `editor/js/libs/ui.three.js` | cb8b4eb1a0b3e57a10411a4e4e4f3e234c47a7cfd9199c147f2f7f8b322b31bc | unmodified | cb8b4eb1a0b3e57a10411a4e4e4f3e234c47a7cfd9199c147f2f7f8b322b31bc |
| `editor/manifest.json` | 4abc25026942d6c54c8ddd03a28d72075d3071f572d2d2bb8c9f88f6d8d3ab66 | unmodified | 4abc25026942d6c54c8ddd03a28d72075d3071f572d2d2bb8c9f88f6d8d3ab66 |
| `examples/jsm/controls/TransformControls.js` | 621fb8894c0f748c2be2a6ff8a6bdd72c18a22b24ddfe21d506514544df5c84e | unmodified | 621fb8894c0f748c2be2a6ff8a6bdd72c18a22b24ddfe21d506514544df5c84e |
| `examples/jsm/environments/RoomEnvironment.js` | c20f0b4677f6128a138d7152b85cbef9091f4f45cdfa05adca04d40c1697c7ae | unmodified | c20f0b4677f6128a138d7152b85cbef9091f4f45cdfa05adca04d40c1697c7ae |
| `examples/jsm/helpers/VertexNormalsHelper.js` | 666a1219d95010e4d5d290f08774af2c327b7bcd368ebdec29a111805c8b7ecc | unmodified | 666a1219d95010e4d5d290f08774af2c327b7bcd368ebdec29a111805c8b7ecc |
| `examples/jsm/helpers/ViewHelper.js` | 7c5d31edf44b1d5b7fb56b4ac7256a4fc62c445ccc964569ff072a39b002d32f | unmodified | 7c5d31edf44b1d5b7fb56b4ac7256a4fc62c445ccc964569ff072a39b002d32f |
| `examples/jsm/interactive/HTMLMesh.js` | 30d8b009f09d065f44d964a53a78ddc239f81ca204bb457b6f0777db59eeef15 | unmodified | 30d8b009f09d065f44d964a53a78ddc239f81ca204bb457b6f0777db59eeef15 |
| `examples/jsm/interactive/InteractiveGroup.js` | e475297351472f48052dbec790611abe34b12f8ebf82dda5fa118197851da43a | unmodified | e475297351472f48052dbec790611abe34b12f8ebf82dda5fa118197851da43a |
| `examples/jsm/libs/draco/draco_encoder.js` | d5c653caf242556ddff9370e86979b8b5c6131bc196be388698a85dc2e1bb59b | unmodified | d5c653caf242556ddff9370e86979b8b5c6131bc196be388698a85dc2e1bb59b |
| `examples/jsm/libs/fflate.module.js` | 209a4412eb48ce609edb4391992a792ffcc3983d30ee7e2b0b89a8c470f3cd8a | unmodified | 209a4412eb48ce609edb4391992a792ffcc3983d30ee7e2b0b89a8c470f3cd8a |
| `examples/jsm/libs/mikktspace.module.js` | a27a8e3c71d2d47eab93c57e84d9190925df42cdbd820e07be15c8edc789c7ca | unmodified | a27a8e3c71d2d47eab93c57e84d9190925df42cdbd820e07be15c8edc789c7ca |
| `examples/jsm/libs/motion-controllers.module.js` | 4bf534b98dbb0f3e90b1332ac4d49c7d7747253768970abc7e4133b7700b75b1 | unmodified | 4bf534b98dbb0f3e90b1332ac4d49c7d7747253768970abc7e4133b7700b75b1 |
| `examples/jsm/loaders/GLTFLoader.js` | caba6c51cfd8c7d5313bd7705a54b76bc0a7199d9822ecc497c5311eaffe8e5e | unmodified | caba6c51cfd8c7d5313bd7705a54b76bc0a7199d9822ecc497c5311eaffe8e5e |
| `examples/jsm/loaders/TGALoader.js` | 7560c445146ed4432a340e312672155d86cf54ad3fc7f5571da12cba9512b5c8 | unmodified | 7560c445146ed4432a340e312672155d86cf54ad3fc7f5571da12cba9512b5c8 |
| `examples/jsm/postprocessing/Pass.js` | 444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5 | unmodified | 444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5 |
| `examples/jsm/utils/BufferGeometryUtils.js` | cbcfe1864abedcc0122cb893373918fe14469491717a34ca6efcc38551805765 | unmodified | cbcfe1864abedcc0122cb893373918fe14469491717a34ca6efcc38551805765 |
| `examples/jsm/utils/SkeletonUtils.js` | 0761b1e003917b215d25dd81439d854b36034b5a57ce84c2cfe3b02428d7b253 | unmodified | 0761b1e003917b215d25dd81439d854b36034b5a57ce84c2cfe3b02428d7b253 |
| `examples/jsm/webxr/XRControllerModelFactory.js` | 657235f1d96605b49a14d58667aadaeb649851b78cf2c8b51de85fed33dfaca8 | unmodified | 657235f1d96605b49a14d58667aadaeb649851b78cf2c8b51de85fed33dfaca8 |
