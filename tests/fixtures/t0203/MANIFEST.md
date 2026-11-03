# tests/fixtures/t0203 — T-0203 delayed-navigation pages

Hand-written for T-0203 (docs/tasks/T-0203.md) by the test-author, 2026-10-29. No third-party content; licence as the
repository (Apache-2.0). Served on 127.0.0.1 only, by the server in
`tests/integration/test_t0203_follow_driver_browser.py` (`DelayHandler`), which answers `/results.html` after
`?delay=` ms (default 600), or holds it until the row releases gate `?key=`, then per `?mode=`
(`download`, `close`, `204`, or the page), and `/redirect?to=PATH` with a 302 (`/redirect` without `to`: a 302 to
`/results.html` with the same query). `mode=stream` (group B, R0) streams `/results.html`: the status line, headers and
`stream_head.html` after `?ttfb=` ms (default 0) or once gate `key` opens, then `stream_tail.html` `delay` ms later in
the same response.

Colours and geometry are chosen by hand; the rows read expected pixels from these numbers, never from a capture.

| file | what | colours / geometry (CSS px, 1280x800) |
|---|---|---|
| `search.html` | a GET form to `/results.html` with one text box; Enter submits natively. Copies its own `delay`, `key`, `mode`, `ttfb` query parameters into hidden inputs; `to=PATH` replaces the action; `js=N` replaces the native submit with a `location.assign` N ms after Enter (reading R1). | background rgb(30, 60, 200); box white, 600x60 at (340, 100), centre (640, 130) |
| `results.html` | the results page | whole viewport rgb(230, 120, 20) |
| `link.html` | the whole viewport is one link, to `/results.html` or to `?to=PATH` | rgb(200, 200, 0) |
| `noop_enter.html` | a form whose box refuses Enter in `keydown` (`preventDefault`): nothing submits | background rgb(0, 150, 60); box as `search.html`'s |
| `stream_head.html` | group B: the first chunk of the streamed results response (no closing tags) | page rgb(245, 245, 245); header band rgb(20, 20, 20), y 0..200, white text |
| `stream_tail.html` | group B: the second chunk: the results block and the end of the document | results rgb(230, 120, 20), y 200..800 |

Every page sets `window.__doc` to a random token at load (a "same document" premise check).
