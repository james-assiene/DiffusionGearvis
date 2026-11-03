"""T-0230: the start page shows no example address, and the Browser specialist built-in goes as far as checkout
(docs/tasks/T-0230.md).

**Why.** Desk session 2026-11-02: Gearvis read ``/web.html``'s greyed placeholder ``en.wikipedia.org`` as text already
typed into the box, in every browser turn, and once typed it and clicked Go. The model sees pixels, so a placeholder
that looks like an address *is* an address to it. The failure mode these rows aim at: any address-shaped string left
anywhere the page shows at rest (placeholder, label, notice, an error paragraph pre-filled in the markup).

**Rows.**

- criterion 1, static: the address input's placeholder is exactly ``Type a web address`` and no ``placeholder`` on the
  page looks like a host; no ``[a-z0-9-]+\\.(org|com|net)`` string appears in ``web.html`` once the error hint is
  removed; the hint reads ``Type a web address, such as example.com`` and appears only in the script (never in the
  markup at rest); the old ``en.wikipedia.org`` hint is gone.
- criterion 1, "every other behaviour unchanged": ``web.html`` normalised (HTML and whole-line ``//`` comments dropped,
  whitespace runs collapsed) hashes to ``EXPECTED_SHA256``, which is BASE's ``web.html`` normalised the same way after
  exactly the spec's two substitutions (the placeholder value and the hint value). A premise row recomputes that
  literal from BASE so the literal is not taken on trust. A later task that edits ``web.html`` re-pins it.
- criterion 2: the verbatim ``browser_specialist`` text passes ``validate_instructions`` and is at most 1000 code points
  (premise rows on the spec's text: green now). The text in ``hud/src/settings/prompts.js`` is pinned by
  ``tests/hud/settings_prompts.test.mjs`` (node imports the module); the rows here check the JS source holds the new
  sentences and not the old one, so the fast suite sees it too.
- criterion 3: nothing under ``local_astra/conductor`` (which holds the live guidance texts), ``local_astra/toolcall``
  or ``local_astra/pages`` other than ``static/web.html`` differs from BASE, tracked or untracked.

The browser rows (rendered page, ``nodot``) live in ``tests/integration/test_computer_web_browser.py``
(``test_t0230_web_html_shows_no_address``, and the re-pinned no-dot row of ``test_web_html_address_rules``).

BASE ``eb58ea00`` (main with the task file). The scope row compares the closed range ``BASE..TIP`` with TIP
``7136788f``, T-0230's tip (``tests/_git_range.py``).
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from local_astra.conductor.prompt_builder import validate_instructions
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
WEB_HTML = REPO / "local_astra" / "pages" / "static" / "web.html"
PROMPTS_JS = REPO / "hud" / "src" / "settings" / "prompts.js"
#: main with the task file, the commit T-0230 branched from
BASE = "eb58ea00"
#: T-0230's tip on its branch: the implementation (8cf9837d), main merged in (R-3), and the R-3 guard re-pins
TIP = "7136788f"

PLACEHOLDER = "Type a web address"
HINT = "Type a web address, such as example.com"
OLD_PLACEHOLDER = "en.wikipedia.org"
OLD_HINT = "Type a web address like en.wikipedia.org"

#: the task's "no string matching ``[a-z0-9-]+\.(org|com|net)``", case-insensitive (pixels have no case rule)
TLD_ADDRESS = re.compile(r"[a-z0-9-]+\.(org|com|net)", re.IGNORECASE)
#: a host-shaped placeholder: two or more dot-separated labels, any TLD (``shop.example``, ``192.168.0.1``)
HOST_SHAPE = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+", re.IGNORECASE)

#: criterion 2, verbatim from docs/tasks/T-0230.md
BROWSER_SPECIALIST = (
    "When I ask you to do something on a website, do it yourself in your own browser, one step at a time, and "
    "check the screen after each step. Do not narrate every click. When you are done, tell me in one sentence "
    "what you did and what you see. If you are stuck, say what is blocking you. When I ask you to buy or order "
    "something, do every step you can: find it, add it to the cart and go to checkout. Stop at sign-in, payment "
    "or the final order button and tell me what is left for me to do. Never type passwords or card numbers, and "
    "ask me before sending, posting or deleting anything."
)
OLD_BROWSER_SENTENCE = "Always ask me before buying, sending, posting or deleting anything."

#: sha256 of ``normalise(BASE web.html with placeholder="en.wikipedia.org" -> placeholder="Type a web address" and
#: "Type a web address like en.wikipedia.org" -> "Type a web address, such as example.com")``; recomputed from BASE by
#: ``test_premise_expected_hash_is_base_with_the_two_substitutions``
EXPECTED_SHA256 = "b7528c6213786dbe3d461ad2234f73b9d2db2c2c2abd5f72100fc969d2e633c4"


def normalise(html: str) -> str:
    """Drop HTML comments and whole-line ``//`` comments, collapse whitespace runs to one space."""
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    lines = [ln for ln in html.splitlines() if not ln.strip().startswith("//")]
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def t0230_substitute(base: str) -> str:
    """A pre-T-0230 ``web.html`` with exactly T-0230's two substitutions (A1), each required to occur once: the
    placeholder ``en.wikipedia.org`` -> ``Type a web address`` and the hint "... like en.wikipedia.org" -> "..., such
    as example.com". Used by the older byte-identity guards on web.html (T-0160 C, T-0169 C4; T-0230 R-3)."""
    old_p, new_p = f'placeholder="{OLD_PLACEHOLDER}"', f'placeholder="{PLACEHOLDER}"'
    old_h, new_h = f'"{OLD_HINT}"', f'"{HINT}"'
    assert base.count(old_p) == 1 and base.count(old_h) == 1, (
        "premise: the base web.html has the old placeholder and the old hint once each"
    )
    return base.replace(old_p, new_p).replace(old_h, new_h)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Attrs(HTMLParser):
    """Every element's attributes, and the markup's text outside ``<script>``/``<style>``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.text.append(data)


def parsed() -> _Attrs:
    p = _Attrs()
    p.feed(WEB_HTML.read_text(encoding="utf-8"))
    p.close()
    return p


def address_input(p: _Attrs) -> dict[str, str | None]:
    found = [a for t, a in p.elements if t == "input" and a.get("id") == "address"]
    assert len(found) == 1, f"premise: web.html has one #address input, got {len(found)}"
    return found[0]


# ==================================================================================================================
# criterion 1: no example address at rest; the new hint
# ==================================================================================================================


def test_address_placeholder_is_type_a_web_address() -> None:
    """A1: ``placeholder="Type a web address"`` on the address input."""
    got = address_input(parsed()).get("placeholder")
    assert got == PLACEHOLDER, (
        f"T-0230 A1: the address input's placeholder is {got!r}, want {PLACEHOLDER!r}"
    )


def test_no_placeholder_looks_like_a_host() -> None:
    """A1 (adversarial): no ``placeholder`` on the page (any element) has a host shape, any TLD."""
    bad = [
        (t, a.get("placeholder"))
        for t, a in parsed().elements
        if a.get("placeholder") is not None and HOST_SHAPE.search(a.get("placeholder") or "")
    ]
    assert bad == [], f"T-0230 A1: a host-shaped placeholder the model reads as typed text: {bad}"


def test_no_address_outside_the_error_hint() -> None:
    """A1: once every occurrence of the error hint is removed, no ``[a-z0-9-]+.(org|com|net)`` string is left in
    ``web.html``'s markup, attributes, notice or script code. Scoped to code: HTML comments and whole-line ``//``
    comments are dropped first (BASE's script has a comment citing ``"example.com:443"`` as a port example; it never
    renders, and criterion 1 asks for nothing else to change, so the row does not force rewording it)."""
    rest = normalise(WEB_HTML.read_text(encoding="utf-8")).replace(HINT, "")
    hits = sorted({m.group(0) for m in TLD_ADDRESS.finditer(rest)})
    assert hits == [], f"T-0230 A1: example addresses outside the hint: {hits}"


def test_the_hint_text() -> None:
    """A1: the error hint reads "Type a web address, such as example.com"; T-0160's "like en.wikipedia.org" is gone."""
    src = WEB_HTML.read_text(encoding="utf-8")
    assert OLD_HINT not in src, "T-0230 A1: the old hint naming en.wikipedia.org is still there"
    assert OLD_PLACEHOLDER not in src, "T-0230 A1: en.wikipedia.org is still on the page"
    assert f'"{HINT}"' in src, f"T-0230 A1: the hint string {HINT!r} is not in web.html's script"


def test_the_hint_is_not_shown_at_rest() -> None:
    """A1 "the hint appears only in an error": the hint is not in the markup's text (outside ``<script>``), nor in any
    attribute, and the ``#error`` paragraph is empty in the markup. A guard, green at BASE: it reddens an
    implementation that shows the new hint (or example.com) at rest instead of only in an error."""
    p = parsed()
    text = " ".join(p.text)
    assert HINT not in text and "example.com" not in text, "T-0230 A1: the hint is visible at rest"
    attrs = [v for _, a in p.elements for v in a.values() if v]
    assert not [v for v in attrs if "example.com" in v], "T-0230 A1: example.com in an attribute"
    src = WEB_HTML.read_text(encoding="utf-8")
    m = re.search(r'<p[^>]*\bid="error"[^>]*>(.*?)</p>', src, re.DOTALL)
    assert m is not None, "premise: web.html has a #error paragraph"
    assert m.group(1).strip() == "", f"T-0230 A1: #error is pre-filled: {m.group(1)!r}"


def test_everything_else_is_unchanged() -> None:
    """A1 "every other behaviour of the page (T-0160 B-5) is unchanged": ``web.html``, normalised, is BASE's with
    exactly the two substitutions (schemes, https upgrade, refusals, ``__state``, autofocus, notice: byte for byte
    modulo comments and whitespace)."""
    got = sha(normalise(WEB_HTML.read_text(encoding="utf-8")))
    assert got == EXPECTED_SHA256, (
        "T-0230 A1: web.html differs from BASE by more (or less) than the placeholder and the hint "
        f"(normalised sha256 {got}, want {EXPECTED_SHA256}); comments and whitespace are ignored"
    )


def test_premise_expected_hash_is_base_with_the_two_substitutions() -> None:
    """Premise (green now): ``EXPECTED_SHA256`` is BASE's ``web.html`` with exactly the spec's two substitutions,
    each made once, normalised."""
    gr.require_commits(BASE, why="T-0230's BASE")
    proc = gr._git("show", f"{BASE}:local_astra/pages/static/web.html")
    assert proc.returncode == 0, proc.stderr
    base = proc.stdout
    want = t0230_substitute(base)
    assert sha(normalise(want)) == EXPECTED_SHA256
    assert sha(normalise(base)) != EXPECTED_SHA256, "premise: the hash tells BASE apart"


# ==================================================================================================================
# criterion 2: the Browser specialist built-in
# ==================================================================================================================


def test_premise_spec_text_is_a_valid_instructions_block() -> None:
    """Premise (green now): the task's text passes ``validate_instructions`` and is at most 1000 code points."""
    assert validate_instructions(BROWSER_SPECIALIST) is None, validate_instructions(
        BROWSER_SPECIALIST
    )
    assert len(BROWSER_SPECIALIST) <= 1000, len(BROWSER_SPECIALIST)


def _js_strings(src: str) -> str:
    """The concatenation of every double-quoted JS string literal joined by ``+`` (the built-ins' shape), so a text
    split across lines reads whole."""
    return re.sub(r'"\s*\+\s*"', "", src)


def test_prompts_js_holds_the_new_browser_specialist_text() -> None:
    """A2: ``prompts.js`` holds the task's text (as ``"..." + "..."`` pieces joined) and not T-0225's last sentence.
    The node module pins the exported value verbatim; this row lets the fast suite see it."""
    src = _js_strings(PROMPTS_JS.read_text(encoding="utf-8"))
    assert OLD_BROWSER_SENTENCE not in src, (
        "T-0230 A2: T-0225's ask-before-buying sentence is still a built-in"
    )
    assert f'"{BROWSER_SPECIALIST}"' in src, (
        "T-0230 A2: prompts.js lacks the browser_specialist text verbatim"
    )


# ==================================================================================================================
# criterion 3: nothing else moves
# ==================================================================================================================

PROTECTED = ("local_astra/conductor", "local_astra/toolcall", "local_astra/pages")
ALLOWED = {"local_astra/pages/static/web.html"}


def test_scope_conductor_toolcall_and_other_pages_unchanged() -> None:
    """A3, closed range ``BASE..TIP`` (``tests/_git_range.py``; .claude/rules/tests.md): between ``eb58ea00`` and
    T-0230's tip ``7136788f`` the only path under ``conductor/`` (the live guidance texts live in
    ``prompt_builder.py``), ``toolcall/`` or ``pages/`` that changed is ``pages/static/web.html``. Exact equality: a
    committed change to any other protected path reddens it, and so does a tip where web.html did not change (the
    power half: the predicate must see the one change the task made). Later tasks' edits on main are outside the
    range, so they cannot redden it; an uncommitted working-tree edit is outside it too (review the diff for that)."""
    gr.require_commits(BASE, TIP, why="T-0230's BASE and TIP")
    assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
    moved = gr.changed(BASE, TIP, *PROTECTED)
    assert moved == sorted(ALLOWED), f"T-0230 A3: protected paths changed in {BASE}..{TIP}: {moved}"


@pytest.mark.parametrize("path", sorted(ALLOWED))
def test_scope_premise_allowed_paths_exist(path: str) -> None:
    """Premise: the allowed path is real (a typo would make the scope row wider than the task)."""
    assert (REPO / path).is_file(), path
