# tests/fixtures/browser/web — T-0160 B-7 fixture pages

Hand-written by the T-0160 group B test-author (2026-10-27); no generator, no third-party content; licence as the
repository. Served by `tests/integration/test_computer_web_browser.py` from a loopback TLS server (a self-signed
certificate made per run with `openssl` in the pytest temp dir, never in the repo) that answers by `Host` header.
Chromium reaches it only through the session's `EgressProxy`: an injected resolver maps the fixture names to
public-looking addresses and an injected connector pipes the checked address to the loopback TLS server. No page
here is ever fetched from the real network.

| file | served as | purpose (B-7 row) |
|---|---|---|
| `index.html` | `https://shop.example/` | (a) the navigation target ("Shop fixture") |
| `login.html` | `https://shop.example/login.html` | (b) a username field and a `type=password` field |
| `checkout.html` | `https://shop.example/checkout.html`, and `https://shop.example/card_form.html` (T-0192 R-A4: the B-7 (c) rows use this non-checkout alias, so the card guard, not the checkout guard, decides them) | (c) an `autocomplete=cc-number` field and a cross-origin iframe |
| `card_iframe.html` | `https://pay.example/card_iframe.html` | (c) a card field (`name=cardnumber`) inside the cross-origin iframe |
| `flip.html` | `https://shop.example/flip.html` | (c) a text field that switches to `type=password` on focus |
| `shadow_open.html` | `https://shop.example/shadow_open.html` | (b) a password field inside an open shadow root |
| `shadow_closed.html` | `https://shop.example/shadow_closed.html` | (b)/B-3 (d) a password field inside a closed shadow root (unreadable: FOCUS_UNKNOWN) |
| `shadow_nested.html` | `https://shop.example/shadow_nested.html` | (c) a card field two open shadow roots deep |
| `shadow_iframe.html` | `https://shop.example/shadow_iframe.html` | (b) an iframe inside an open shadow root, holding a password field |
| `editable.html` | `https://shop.example/editable.html` | B-3 (d): a contenteditable div (types) and a plain `div[tabindex]` (FOCUS_UNKNOWN) |
| `two_frames.html` | `https://shop.example/two_frames.html` | (c) two iframes, the card field in the second |
| `plain_iframe.html` | `https://shop.example/plain_iframe.html` | an ordinary text field, framed by `two_frames.html` and `frames_a.html` |
| `label_card.html` | `https://shop.example/label_card.html` | (c) a card field identified by its `<label>` only |
| `pw_inner.html` | `https://shop.example/pw_inner.html` | a password field, framed by `object_pw.html`, `embed_pw.html`, `frames_b.html` |
| `object_pw.html` | `https://shop.example/object_pw.html` | (b) a password field inside `<object type="text/html">` |
| `embed_pw.html` | `https://shop.example/embed_pw.html` | (b) a password field inside `<embed type="text/html">` |
| `frames_a.html`, `frames_b.html` | `https://shop.example/frames_{a,b}.html` | (b) stale frames: an iframe page, then another iframe page (password) in one browser |
| `download.html` | `https://shop.example/download.html` | (f) a download link to `/file.bin` (served as an attachment) |
| `upload.html` | `https://shop.example/upload.html` | (f) a file input |
| `beacon.html` | `https://shop.example/beacon.html?c=<port>` | (e)/(h) background phone-home to a loopback canary and a private-resolving name |
| `subresources.html` | `https://shop.example/subresources.html?c=<port>` | (e) fetch, img, iframe, WebSocket and `location=` to a second loopback port |

Server-side endpoints (in the test module, not files): `/redirect?to=<url>` answers `302` to `<url>` (row (d));
`/file.bin` answers a unique payload with `Content-Disposition: attachment` (row (f)).

`copy.html` (T-0160 C amendment, reviewer A; hand-written, same licence): a button that, on click, copies a marker
with `document.execCommand('copy')` and writes a second marker with `navigator.clipboard.writeText`; outcomes in
`window.__state`. Served from `shop.example` (web mode) and from the TLS server's own loopback origin (the
`web=False` partner).

Name map of the injected resolver: `shop.example` → 93.184.215.14, `pay.example` → 93.184.215.15 (both public by
`fence.is_public_address`), `private.example` → 10.0.0.5 (private; the proxy must refuse it before any connect).

## T-0192 A — stop at checkout (`tests/integration/test_t0192_checkout_browser.py`)

Hand-written by the T-0192 group A test-author (2026-10-29); no generator, no third-party content; licence as the
repository. Served at nested paths by `T0192_ROUTES` in `tests/integration/test_computer_web_browser.py` (the
rig's fixture server; `<token>` is `T0192_TOKEN`, the A8 marker that must never leave the executor).

| file | served as | purpose (row) |
|---|---|---|
| `t0192_product.html` | `https://shop.example/t0192/dp/kettle` | (a) "Add to cart" link to `/cart.html` |
| `t0192_cart.html` | `https://shop.example/cart.html` | (a)-(e): a click counter, links to `/checkout/entry`, `/checkout/entry2`, a `target=_blank` `/checkout/<token>/pay`, `/checkout-faq.html`, and a `history.pushState('/checkout')` button |
| `t0192_signin.html` | `https://shop.example/signin.html?openid.return_to=…%2Fcheckout…` | (b) the sign-in wall the checkout entry redirects to |
| `t0192_checkout.html` | `https://shop.example/checkout/<token>/pay` | (b2)-(h) the checkout page; counts clicks and input |
| `t0192_checkout_faq.html` | `https://shop.example/checkout-faq.html` | (e) a look-alike path that is not a checkout |
| `t0192_framed.html` | `https://shop.example/t0192/framed.html` | (f) an ordinary page framing a cross-origin checkout-shaped iframe |
| `t0192_frame.html` | `https://pay.example/checkout/<token>/frame` | (f) that iframe's document |
| `t0192_cart_form.html` | `https://shop.example/t0192/cart-form.html` | (b3) a cart form (one text input) that POSTs to `/gp/cart/submit` on Enter |

Server-side redirects (in `T0192_ROUTES`, not files): `/checkout/entry` answers `302` to
`/signin.html?openid.return_to=https%3A%2F%2Fshop.example%2Fcheckout%2Fentry` (row (b), the demo's shape: the
checkout is only a redirect hop); `/checkout/entry2` answers `302` to `/checkout/<token>/pay` (row (b2));
`/gp/cart/submit` (the form POST; the server reads the body first) answers `302` to
`/checkout/entry?proceedToCheckout=1` (row (b3), Amazon's chain: the checkout is only the middle hop).
