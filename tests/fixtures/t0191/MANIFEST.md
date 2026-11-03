# tests/fixtures/t0191

Hand-written by the T-0191 test-author on 2026-10-28 for `tests/integration/test_t0191_navigate_browser.py` (T-0191
A6). No generator, no recording, no third-party content; licence as the repository (Apache-2.0).

The module serves these files from its own loopback TLS server, chosen by the request's `Host` header, behind the
web-mode egress proxy with an injected resolver and connector (the T-0160 B-7 / T-0174 B2-3 rig):

| Host | Path | File | What it is |
|---|---|---|---|
| `shop.example` | `/` | `shop.html` | a shop with its own autofocused search box (`#q`, CSS box x 100..499, y 100..139; a form that GETs `/search.html?q=`) |
| `shop.example` | `/unload.html` | `shop_unload.html` | the same shop with a `beforeunload` handler (armed always; `window.__armed === true`) |
| `shop.example` | `/search.html` | `search.html` | the shop's results page (`window.__query` is the `q` parameter) |
| `nav.example` | `/` | `nav.html` | the other website the user asks for |

Every page sets `window.__doc` to a random token at load, so a row can tell one document from the next.
