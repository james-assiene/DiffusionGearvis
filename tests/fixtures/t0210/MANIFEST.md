# tests/fixtures/t0210 — T-0210 A browser fixture

Hand-written by the T-0210 test-author on 2026-10-30 from `docs/tasks/T-0210.md` A6, for
`tests/integration/test_t0210_body_focus_browser.py`. No recorded data, no third-party content. Licence: the
repository's.

Served by `StartPageServer(static_dir=tests/fixtures/t0210)` on loopback as the web-mode start origin (deny-all
resolver and connector), at `/canvas.html`.

| element | id | box (CSS px, viewport 1280x800) | A6 case |
|---|---|---|---|
| text `<input>` (value `seed-value`), first and focused on load | `field` | 20,16 300x28 (+ borders) | (c); every case starts here |
| `<canvas>` 600x400, no `tabindex` (cannot take focus: a click leaves `body` focused) | `plain` | 20,64 600x400 | (a), (e) |
| `<canvas tabindex="0">` (takes focus; reads `unreadable`) | `tab` | 660,64 300x200 | (b) |
| plain `<div>` (not focusable) | `div` | 660,284 300x180 | (a) variant |
| `srcdoc` iframe whose document is put in `designMode` on load (`window.__richReady`) | `rich` | 20,488 600x200 | (d) |

`window.__keys` collects `event.key` of every `keydown` that reaches the top `document` (a listener on `document`, as
a WebGL app's shortcut handler would be). The canvases are painted with fixed stripes so no frame is blank. No timers,
no `Date`, no `Math.random`, no network.
