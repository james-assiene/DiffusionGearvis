# tests/fixtures/t0206 — T-0206 A browser fixtures

Hand-written by the T-0206 test-author on 2026-10-30, for `tests/integration/test_t0206_submit_browser.py`. No
recorded data, no third-party content; the "Google-", "Amazon-" and "Wikipedia-shaped" pages copy only the *shape*
(field names, button labels) of those sites' search forms, written from the task file. Licence: the repository's.

Served by that module's own loopback TLS server (self-signed, generated per run) posing as `https://shop.example`
and `https://pay.example` behind the session's real egress proxy (T-0160 rig: injected resolver and connector, no
network). Every name is served flat at `/<name>`; `POST /sent` and `GET /results` answer `sent.html` / `results.html`
and are recorded, so a row can tell whether a form was sent.

| file | shape | A1 verdict of its send control |
|---|---|---|
| google.html | `q` + input submit "Google Search" | False (b) |
| amazon.html | `field-keywords` + input submit "Go" | False (b), (d) search label |
| wikipedia.html | role=search, type=search `search` + "Search" | False (b) |
| stores.html | T-0205 D6 shape: street address + "Find stores" | False (d) search label |
| flights.html | T-0205 D9 shape: origin, destination + "Search" | False (c), (d) |
| signin.html | email + "Sign in" | False (d) |
| cart.html | quantity + "Add to cart" | False (c), (d) |
| cta.html | `<a href="/quote.html">Get a free quote</a>` + a newsletter email | False (a) |
| filters.html | a filter checkbox, a cookie banner "Accept all" | False (d) |
| contact.html | name, email, phone, message, hidden `ref` (value token), "Send message" | True once a personal field is filled |
| quote.html | T-0205 D7 shape: name, phone, description textarea, "Request quote" | True once filled |
| empty_label.html | email + an empty-label `<button>` (structural) | True once email is filled |
| spa.html | formless textarea + `div[role=button]` "Send message" | True once the textarea is filled |
| spa_quote.html | formless quote "form" (name, email, message, div "Send") directly under ``<body>`` beside a header ``q`` box (ruling on reading 6) | True once filled; the empty ``q`` does not exempt it |
| shadow.html | the same `div[role=button]` inside an open shadow root | True once the textarea is filled |
| framed.html + iframe_contact.html | shop.example framing a pay.example (cross-origin) contact form, "Send" | True once filled |
| overlay.html | a transparent div over "Send" that calls `requestSubmit()` | known gap: hit-testing sees the div, the click goes through (pinned) |
| token.html | a pre-filled value and a button label holding a unique token (A8) | True |
| cost.html | one form and a paragraph to click (A9) | the paragraph: no control |
| sent.html, results.html | landing pages of a sent form / a search | — |
| slotted.html | R-1: `<x-btn>Send message</x-btn>`, shadow `<button><slot>` calling `requestSubmit()`, email + message | True once filled (label, padding, Space on the inner button) |
| webforms.html | R-2: whole-page `<form>` with an empty header `q`, name/email/message, "Send" | True once filled; `q` filled and nothing personal: False |
| locator.html | R-3: postcode, street address, an (empty) email, icon-only submit | location only: False; email filled: True |
| select.html | R-4: two `<select>`s at their default options, "Submit" | a select counts as filled only once changed |
| header_footer.html | r4: formless header `q` + div "Submit", a footer newsletter email | False once `q` is filled |
| hashlink.html | r12: `<a href="#">Send</a>` (calls `requestSubmit()`) in a form with email + message | True once filled |
| framed_offset.html | r8: the pay.example iframe contact form offset (420, 260) from the corner | True once filled |
| nobutton.html | r7: one email field, no button (implicit submission on Enter) | Enter withheld once filled |
| hidden_q.html | r2: a hidden `q` with a value beside an email, "Submit" | True once the email is filled |
| slotted_span.html | R-1 variant: `<x-btn><span>Send</span></x-btn>` (the hit lands on the slotted span) | True once filled |
| select_noattr.html | R-4 variant: `phone_type` (no `selected`: the first option is the default) and a `multiple` `phone_times` with nothing selected | a select counts as filled only once changed |
