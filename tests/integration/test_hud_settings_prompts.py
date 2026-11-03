"""T-0225: the prompt library under "Your instructions to Gearvis", in the real page (criteria 1-5).

``docs/tasks/T-0225.md`` criteria 1-5 and *Tests to write* "test_hud_settings_prompts.py". The instructions box
(T-0213 C, merged) and the server half (``astra.settings.set``, the apply at the next turn) are real here; what these
rows pin is the library: four built-ins and the user's saved prompts filling the box, saved in the page's own
``localStorage``.

**Rig**: the T-0213 C rig of ``tests/integration/test_hud_settings_instructions.py`` (``settings_page``: a real
``Session`` and ``HudServer(port=0)`` over the fakes, tools off, fake LLM replaying ``"Fine. "``; the page over
loopback http; headless Chrome over CDP with the request log and ``init_js_full``, which records every message the
page sends). One Chrome profile per row, so ``localStorage`` lives exactly as long as the row; a reload is a CDP
navigation to the page's own ``location.href`` in the same tab. The init script also replaces ``window.confirm``,
``window.prompt`` and ``window.alert`` with recorders (criterion 2: confirmations are in-page). No number here is a
latency claim; every wait is a generous ceiling; n = 1 per row. Runs only with ``ASTRA_BROWSER_TESTS=1``.

**Oracles** (never the page's code): the task's built-in texts, typed here verbatim; ``validate_instructions`` and
``USER_INSTRUCTIONS_HEADER`` (A1, merged); the system message of each fake LLM request; the server's own
``astra.settings.state``; ``localStorage`` read back through ``JSON.parse``.

**DOM readings** (hook names relayed by the orchestrator; where the criteria leave the shape open, the reading
is named here and in the report):

- the picker is a ``<select data-settings-prompts>`` in the panel; an option names its prompt by its text: the text
  is the name, or the name followed by a non-word marker (e.g. ``"Education tutor (built-in)"``); a built-in is
  "marked built-in" when its option text or its ``<optgroup>`` label contains "built-in" (any case) and no user
  prompt's is; choosing = setting the select's value and dispatching ``input`` and ``change`` (what a user's pick
  fires);
- "Save as..." is ``[data-settings-prompt-save]`` with the name in ``[data-settings-prompt-name]``; if the name field
  is not visible the rows click Save once to reveal it, type, and click Save again;
- the in-page confirmation is ``[data-settings-prompt-confirm]`` while visible; its confirming button is the one whose
  text is not a refusal (``Cancel``/``No``/``Keep``), its refusing button is the one that is;
- Delete is ``[data-settings-prompt-delete]``; the note is ``[data-settings-prompts-note]``.

**Activated** (T-0225, the implementation at dbaec300): the browser rows no longer carry the pending marker, and
``tests/hud/settings_prompts.test.mjs`` (a strict xfail only while this module holds the T-0225 marker string) runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.prompt_builder import (
    INSTRUCTIONS_FORBIDDEN,
    MAX_USER_INSTRUCTIONS,
    USER_INSTRUCTIONS_HEADER,
    validate_instructions,
)
from local_astra.contracts.turn import TurnState
from tests.integration.test_hud_rehearsal import Live, Page
from tests.integration.test_hud_settings import (
    PAGE_WAIT_S,
    SET_TYPE,
    browser,
    init_js_full,
    rec,
    send_raw_set,
    sent_msgs,
    settings_page,
    settle,
    wire_states,
)
from tests.integration.test_hud_settings_instructions import (
    box,
    clear_and_type,
    opened,
    press,
    preview_shown,
    textarea_is,
)

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
#: the commit T-0225 branched from (main at the plan commit); criterion 5's byte-identity guard compares to it
BASE = "377bc592"

KEY = "gearvis.prompts.v1"
NOTE = "Saved prompts are unavailable in this browser"

#: the task's built-ins, verbatim
#: T-0229 A1: the tutor checks the work rather than presupposing a mistake (T-0225's text is historical)
TUTOR = (
    "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, point to it, "
    "explain why, and ask me a guiding question before giving the fix; then give the corrected step and the "
    "correct final answer. If everything is right, say so plainly and don't invent a problem. Speak in plain "
    "words, not symbols, and keep each reply short."
)
GAMING = (
    "You are my gaming companion. I share my game screen with you. Answer out loud in one or two short "
    "sentences, like a friend sitting next to me. Read what is on the screen before you answer. When I ask what "
    "to do, give one clear plan and the reason. Do not use your own browser. Use google_search only for game "
    "facts you are not sure about, and say where you found them."
)
#: T-0230 A2: the Browser specialist goes as far as checkout (T-0225's text is historical)
BROWSER_TEXT = (
    "When I ask you to do something on a website, do it yourself in your own browser, one step at a time, and "
    "check the screen after each step. Do not narrate every click. When you are done, tell me in one sentence "
    "what you did and what you see. If you are stuck, say what is blocking you. When I ask you to buy or order "
    "something, do every step you can: find it, add it to the cart and go to checkout. Stop at sign-in, payment "
    "or the final order button and tell me what is left for me to do. Never type passwords or card numbers, and "
    "ask me before sending, posting or deleting anything."
)
BUILTINS = [
    ("Default", ""),
    ("Education tutor", TUTOR),
    ("Gaming companion", GAMING),
    ("Browser specialist", BROWSER_TEXT),
]
BUILTIN_NAMES = [n for n, _ in BUILTINS]

CANARY = "ZQX-T0225 answer in British English and keep it short."
OTHER = "ZQX-T0225 a second text for the same name."
EVIL_NAME = '<img src=x onerror="window.__t0225_pwned=1">'
ASTRAL = "\U0001f600"
TOO_LONG = "a" * (MAX_USER_INSTRUCTIONS - 1) + ASTRAL * 2
Q1 = "what is on this screen"
Q2 = "and what about now"

PANEL = "[data-settings-panel]"
PICKER = PANEL + " [data-settings-prompts]"
NAME = PANEL + " [data-settings-prompt-name]"
SAVE = PANEL + " [data-settings-prompt-save]"
DELETE = PANEL + " [data-settings-prompt-delete]"
CONFIRM = PANEL + " [data-settings-prompt-confirm]"
PROMPTS_NOTE = PANEL + " [data-settings-prompts-note]"

#: criterion 2: the page asks in its own DOM. These record any native dialog the page opens.
DIALOGS_JS = r"""
(() => {
  const d = {confirm: 0, prompt: 0, alert: 0};
  Object.defineProperty(window, '__t0225_dialogs', {value: d, configurable: true});
  window.confirm = () => { d.confirm += 1; return true; };
  window.prompt = () => { d.prompt += 1; return null; };
  window.alert = () => { d.alert += 1; };
})();
"""

#: criterion 3: a corrupt store (written before any page script runs, on every load of this row)
CORRUPT_JS = (
    "(() => { try { localStorage.setItem(" + json.dumps(KEY) + ", '{nope'); } catch (_) {} })();"
)
#: criterion 3: a blocked store: reading window.localStorage itself throws (as in a locked-down profile)
BLOCKED_JS = r"""
(() => {
  Object.defineProperty(window, 'localStorage', {configurable: true,
    get() { throw new DOMException('The operation is insecure.', 'SecurityError'); }});
})();
"""


#: criterion 7: Export goes "through a blob link, entirely in the page". This records every Blob turned into an
#: object URL and every anchor click (its ``download`` name and ``href``), so the row reads the file the page offered
#: without depending on Chrome's download directory. The click still proceeds.
DOWNLOADS_JS = r"""
(() => {
  const d = {blobs: {}, clicks: []};
  Object.defineProperty(window, '__t0225_downloads', {value: d, configurable: true});
  const make = URL.createObjectURL;
  URL.createObjectURL = function (obj) { const u = make.call(URL, obj); d.blobs[u] = obj; return u; };
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    d.clicks.push({download: this.getAttribute('download'), href: this.href});
    return click.call(this);
  };
  window.addEventListener('click', (e) => {
    const a = e.target && e.target.closest && e.target.closest('a[download]');
    if (a) d.clicks.push({download: a.getAttribute('download'), href: a.href});
  }, true);
})();
"""


def script(*extra: str) -> str:
    return init_js_full() + "\n" + DIALOGS_JS + "\n" + DOWNLOADS_JS + "\n" + "\n".join(extra)


def block(text: str) -> str:
    return USER_INSTRUCTIONS_HEADER + "\n" + text


def last_generate_system(live: Live) -> str:
    assert live.llm.gen_prompts, "premise: a turn generate was sent"
    _rid, prompt = live.llm.gen_prompts[-1]
    systems = [m.text() for m in prompt.messages if m.role == "system"]
    assert len(systems) == 1, systems
    return systems[0]


async def turn(live: Live, question: str, n_lines: int) -> None:
    await live.ask(question)
    await live.until(
        lambda: live.session.state.state is TurnState.IDLE and len(live.turn_lines()) >= n_lines,
        f"premise: the question {question!r} was answered and its turn line written",
    )


# --- page probes --------------------------------------------------------------------------------------------------

MATCH_JS = """const named = (opt, name) => { const t = (opt.textContent || '').trim();
  return t === name || (t.startsWith(name) && !/\\w/.test(t.charAt(name.length))); };"""

PICKER_JS = (
    "(() => { const s = document.querySelector("
    + json.dumps(PICKER)
    + """); if (!s) return {present: false};
  return {present: true, tag: s.tagName.toLowerCase(), value: s.value, options: [...s.querySelectorAll('option')].map((o) => ({
    text: (o.textContent || '').trim(), value: o.value, selected: o.selected,
    group: o.parentElement && o.parentElement.tagName === 'OPTGROUP' ? o.parentElement.label : null}))}; })()"""
)


async def picker(page: Page) -> dict[str, Any]:
    got = await page.evaluate(PICKER_JS)
    assert isinstance(got, dict) and got.get("present"), (
        f"criterion 1: the Prompt picker ({PICKER}) is in the panel: {got!r}"
    )
    return got


def option_names(probe: dict[str, Any], names: list[str]) -> list[str | None]:
    """For each option, the first of ``names`` it names (the reading in the module docstring), else None."""
    out: list[str | None] = []
    for opt in probe["options"]:
        t = opt["text"]
        hit = None
        for n in sorted(names, key=len, reverse=True):
            if t == n or (t.startswith(n) and not re.match(r"\w", t[len(n) : len(n) + 1] or " ")):
                hit = n
                break
        out.append(hit)
    return out


def marked_builtin(opt: dict[str, Any]) -> bool:
    return "built-in" in opt["text"].lower() or "built-in" in (opt["group"] or "").lower()


async def choose(page: Page, name: str) -> None:
    outcome = await page.evaluate(
        "(() => { "
        + MATCH_JS
        + " const s = document.querySelector("
        + json.dumps(PICKER)
        + "); if (!s) return 'no picker';"
        " const o = [...s.querySelectorAll('option')].filter((e) => named(e, "
        + json.dumps(name)
        + ")); if (o.length !== 1) return o.length + ' options'; if (o[0].disabled) return 'disabled';"
        " s.value = o[0].value; o[0].selected = true;"
        " s.dispatchEvent(new Event('input', {bubbles: true})); s.dispatchEvent(new Event('change', {bubbles: true}));"
        " return 'chosen'; })()"
    )
    assert outcome == "chosen", f"criterion 1: one option named {name!r} in the picker: {outcome!r}"


def visible(selector: str) -> str:
    return (
        "(() => { const e = document.querySelector("
        + json.dumps(selector)
        + "); return Boolean(e) && !e.hidden && e.checkVisibility(); })()"
    )


async def click(page: Page, selector: str) -> str:
    return str(
        await page.evaluate(
            "(() => { const e = document.querySelector("
            + json.dumps(selector)
            + "); if (!e) return 'absent'; if (e.disabled) return 'disabled';"
            " if (!e.checkVisibility()) return 'hidden'; e.click(); return 'clicked'; })()"
        )
    )


async def type_name(page: Page, name: str) -> None:
    if not await page.evaluate(visible(NAME)):
        outcome = await click(page, SAVE)
        assert outcome == "clicked", f"criterion 2: Save as... reveals the name field: {outcome!r}"
        await settle(page, visible(NAME))
    focused = await page.evaluate(
        "(() => { const e = document.querySelector("
        + json.dumps(NAME)
        + "); if (!e) return 'absent';"
        " e.focus(); e.select && e.select(); return document.activeElement === e ? 'ok' : 'not focused'; })()"
    )
    assert focused == "ok", f"criterion 2: the name field ({NAME}) takes the focus: {focused!r}"
    await asyncio.to_thread(page.chrome._cmd, "Input.insertText", text=name)


async def save_as(page: Page, name: str) -> None:
    """Type ``name`` and press Save (the box's current text is what is saved)."""
    await type_name(page, name)
    outcome = await click(page, SAVE)
    assert outcome == "clicked", f"criterion 2: Save as... is pressable: {outcome!r}"


async def answer_confirm(page: Page, yes: bool) -> str:
    return str(
        await page.evaluate(
            "(() => { const c = document.querySelector("
            + json.dumps(CONFIRM)
            + "); if (!c || c.hidden || !c.checkVisibility()) return 'no confirm';"
            " const bs = [...c.querySelectorAll('button')].filter((b) => !b.disabled && b.checkVisibility());"
            " const no = (b) => /^(cancel|no|keep)\\b/i.test(b.textContent.trim());"
            " const pick = bs.filter((b) => " + ("!no(b)" if yes else "no(b)") + ");"
            " if (pick.length !== 1) return pick.length + ' buttons: ' + bs.map((b) => b.textContent.trim()).join('|');"
            " pick[0].click(); return 'clicked'; })()"
        )
    )


async def confirm_shown(page: Page) -> bool:
    return bool(await page.evaluate(visible(CONFIRM)))


async def storage(page: Page) -> Any:
    return await page.evaluate(
        "(() => { try { const r = localStorage.getItem(" + json.dumps(KEY) + ");"
        " return r === null ? null : JSON.parse(r); } catch (e) { return {__error__: String(e)}; } })()"
    )


async def dialogs(page: Page) -> dict[str, int]:
    got = await page.evaluate("window.__t0225_dialogs || null")
    assert isinstance(got, dict), f"rig: the dialog recorder is missing: {got!r}"
    return got


async def reload(page: Page) -> None:
    """Navigate the tab to its own address and wait for the new document's socket and panel."""
    href = await page.evaluate("(() => { window.__t0225_old = 1; return location.href; })()")
    await page.navigate(str(href))
    await page.until(
        "(() => !window.__t0225_old && document.readyState === 'complete')()",
        "the reloaded document",
        timeout_s=PAGE_WAIT_S,
    )
    await page.connected()
    await opened(page)


async def settled_store(page: Page, want: Any) -> Any:
    await settle(
        page,
        "(() => { try { const r = localStorage.getItem(" + json.dumps(KEY) + ");"
        " return JSON.stringify(r === null ? null : JSON.parse(r)) === "
        + json.dumps(json.dumps(want, separators=(",", ":")))
        + ";"
        " } catch (_) { return false; } })()",
    )
    return await storage(page)


def counter_ints(view: dict[str, Any]) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", view["counter"] or "")]


# --- criterion 1 --------------------------------------------------------------------------------------------------


@browser
async def test_the_picker_lists_the_four_builtins_in_order_and_choosing_fills_the_box_sending_nothing(
    tmp_path,
):
    """Criterion 1 / Tests to write: the picker lists Default, Education tutor, Gaming companion, Browser specialist
    first, in that order, each marked built-in, and nothing else on a fresh profile; choosing each fills the textarea
    with the task's text verbatim (each one ``validate_instructions`` accepts), the counter follows (code points of
    the text against 1000), and nothing is sent (no ``astra.settings.set``, no new message type). The note is not
    shown when the store is fine (the guard is not too wide)."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        before = await rec(page)
        probe = await picker(page)
        filled: list[tuple[str, str, list[int]]] = []
        for name, text in [*BUILTINS[1:], BUILTINS[0]]:
            await choose(page, name)
            await settle(page, textarea_is(text))
            view = await box(page)
            filled.append((name, view["value"], counter_ints(view)))
        await asyncio.sleep(0.5)  # nothing-sent window, started after the last choice settled
        sets = await sent_msgs(page)
        after = await rec(page)
        note = await page.evaluate(
            "(() => { const n = document.querySelector(" + json.dumps(PROMPTS_NOTE) + ");"
            " return n && !n.hidden && n.checkVisibility() ? n.textContent : ''; })()"
        )
    assert probe["tag"] == "select", probe
    assert option_names(probe, BUILTIN_NAMES)[:4] == BUILTIN_NAMES, (
        f"criterion 1: the four built-ins first, in order: {[o['text'] for o in probe['options']]}"
    )
    assert len([o for o in probe["options"] if o["text"]]) >= 4
    named = [
        o for o, n in zip(probe["options"], option_names(probe, BUILTIN_NAMES), strict=True) if n
    ]
    assert len(named) == 4, f"each built-in listed once: {[o['text'] for o in probe['options']]}"
    assert all(marked_builtin(o) for o in named), f"criterion 1: each is marked built-in: {named}"
    for name, value, ints in filled:
        want = dict(BUILTINS)[name]
        assert value == want, (
            f"criterion 1: choosing {name!r} fills the box with its text: {value[:80]!r}"
        )
        assert validate_instructions(value) is None, (
            f"{name}: the built-in passes validate_instructions"
        )
        assert ints == [len(want), MAX_USER_INSTRUCTIONS], (
            f"{name}: the counter follows the choice: {ints}"
        )
    assert sets == [], f"criterion 1: choosing sends nothing until Apply; got {sets}"
    new = after["sent"][len(before["sent"]) :]
    assert not [t for t in new if t != "astra.frame"], (
        f"criterion 5: no message for choosing; sent {new}"
    )
    assert NOTE not in (note or ""), f"the note is shown only when the store fails: {note!r}"


@browser
async def test_apply_after_choosing_the_tutor_sends_one_set_and_the_next_request_ends_with_it(
    tmp_path,
):
    """Criterion 1 / Tests to write: after a turn on BASE, choose "Education tutor", press Apply: exactly one
    ``astra.settings.set`` with ``changes == {user_instructions: TUTOR}``; the server applies it (rev 1); the next
    turn's system message ends with ``USER_INSTRUCTIONS_HEADER + "\\n" + TUTOR`` and starts with BASE's."""
    async with settings_page(tmp_path, block=False, script=script()) as (live, page, _c):
        await turn(live, Q1, 1)
        base = last_generate_system(live)
        await opened(page)
        await choose(page, "Education tutor")
        await settle(page, textarea_is(TUTOR))
        await asyncio.sleep(0.3)
        chosen = await sent_msgs(page)
        await press(page, "Apply")
        await live.until(
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "Apply's set applied",
            timeout_s=PAGE_WAIT_S,
        )
        state = [s for s in wire_states(live) if s.rev == 1][-1]
        await asyncio.sleep(0.3)
        sets = await sent_msgs(page)
        await turn(live, Q2, 2)
        after = last_generate_system(live)
    assert chosen == [], f"nothing is sent until Apply: {chosen}"
    assert len(sets) == 1 and sets[0]["changes"] == {"user_instructions": TUTOR}, sets
    assert state.instructions == TUTOR and not state.refused, (state.instructions, state.refused)
    assert after.endswith(block(TUTOR)), after[-300:]
    assert after.startswith(base)


@browser
async def test_a_chosen_prompt_is_a_draft_that_a_later_server_state_does_not_overwrite(tmp_path):
    """Criterion 1 ("the box's current state (draft...) behaves exactly as in T-0213 C"), adversarial: a chosen
    prompt is a draft like typed text. After choosing "Gaming companion", a set from elsewhere (written straight onto
    the socket) applies CANARY at rev 1; once the page holds rev 1, the textarea still holds GAMING (a page that fills
    the box without marking it a draft loses the choice to the next state)."""
    async with settings_page(tmp_path, block=False, script=script()) as (live, page, _c):
        await opened(page)
        await choose(page, "Gaming companion")
        await settle(page, textarea_is(GAMING))
        await send_raw_set(
            page, {"type": SET_TYPE, "rev": 0, "changes": {"user_instructions": CANARY}}
        )
        await live.until(
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "premise: the raw set applied",
            timeout_s=PAGE_WAIT_S,
        )
        state = [s for s in wire_states(live) if s.rev == 1][-1]
        await page.until(
            preview_shown(state.system_preview or ""),
            "premise: the page holds rev 1",
            timeout_s=PAGE_WAIT_S,
        )
        await asyncio.sleep(0.3)
        view = await box(page)
    assert state.instructions == CANARY, "premise"
    assert view["value"] == GAMING, f"the chosen text is a draft and stays: {view['value'][:80]!r}"


# --- criterion 2 / 3 ----------------------------------------------------------------------------------------------


@browser
async def test_save_as_mine_is_stored_listed_after_a_reload_and_fills_the_box(tmp_path):
    """Criteria 2-3 / Tests to write: type CANARY, Save as "Mine": ``localStorage["gearvis.prompts.v1"]`` is the JSON
    list ``[{name: "Mine", text: CANARY}]``; saving sends nothing to the server and opens no native dialog. After a
    reload "Mine" is listed after the four built-ins and choosing it fills the box with CANARY."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        before = await rec(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, "Mine")
        stored = await settled_store(page, [{"name": "Mine", "text": CANARY}])
        await asyncio.sleep(0.3)
        after = await rec(page)
        sets = await sent_msgs(page)
        natives = await dialogs(page)
        await reload(page)
        probe = await picker(page)
        await choose(page, "Mine")
        await settle(page, textarea_is(CANARY))
        view = await box(page)
    assert stored == [{"name": "Mine", "text": CANARY}], (
        f"criterion 3: one key, a JSON list of {{name, text}}: {stored!r}"
    )
    assert sets == [], f"criterion 5: saving sends no set: {sets}"
    new = after["sent"][len(before["sent"]) :]
    assert not [t for t in new if t != "astra.frame"], f"criterion 5: no new client message: {new}"
    assert natives == {"confirm": 0, "prompt": 0, "alert": 0}, (
        f"criterion 2: no native dialog: {natives}"
    )
    hits = option_names(probe, [*BUILTIN_NAMES, "Mine"])
    assert hits[:4] == BUILTIN_NAMES and hits.count("Mine") == 1, [
        o["text"] for o in probe["options"]
    ]
    assert hits.index("Mine") > 3, "criterion 1: user prompts follow the built-ins"
    mine = probe["options"][hits.index("Mine")]
    assert not marked_builtin(mine), f"a user prompt is not marked built-in: {mine}"
    assert view["value"] == CANARY, view["value"]


@browser
async def test_user_prompts_are_listed_alphabetically_after_the_builtins(tmp_path):
    """Criterion 1: saved in the order Gamma, Alpha, Beta, the picker lists Alpha, Beta, Gamma after the four
    built-ins (names chosen so code-unit and case-insensitive order agree)."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        for name in ("Gamma", "Alpha", "Beta"):
            await clear_and_type(page, f"text of {name}")
            await settle(page, textarea_is(f"text of {name}"))
            await save_as(page, name)
            await settle(
                page,
                "(() => { try { return (JSON.parse(localStorage.getItem("
                + json.dumps(KEY)
                + ")) || [])"
                ".some((p) => p.name === "
                + json.dumps(name)
                + "); } catch (_) { return false; } })()",
            )
        await reload(page)
        probe = await picker(page)
    hits = [h for h in option_names(probe, [*BUILTIN_NAMES, "Alpha", "Beta", "Gamma"]) if h]
    assert hits == [*BUILTIN_NAMES, "Alpha", "Beta", "Gamma"], [o["text"] for o in probe["options"]]


@browser
async def test_saving_over_an_existing_name_asks_in_page_and_replaces_only_on_yes(tmp_path):
    """Criterion 2: with "Mine" = CANARY saved, saving OTHER as "Mine" shows the in-page confirmation and changes
    nothing yet; refusing keeps CANARY; saving again and confirming replaces it (one entry, OTHER). No native
    ``window.confirm``."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, "Mine")
        await settled_store(page, [{"name": "Mine", "text": CANARY}])
        await clear_and_type(page, OTHER)
        await settle(page, textarea_is(OTHER))
        await save_as(page, "Mine")
        await settle(page, visible(CONFIRM))
        asked = await confirm_shown(page)
        held = await storage(page)
        refused = await answer_confirm(page, yes=False)
        await asyncio.sleep(0.3)
        kept = await storage(page)
        await save_as(page, "Mine")
        await settle(page, visible(CONFIRM))
        accepted = await answer_confirm(page, yes=True)
        replaced = await settled_store(page, [{"name": "Mine", "text": OTHER}])
        natives = await dialogs(page)
    assert asked, "criterion 2: replacing a saved name asks first, in the page"
    assert held == [{"name": "Mine", "text": CANARY}], (
        f"nothing is replaced before the answer: {held!r}"
    )
    assert refused == "clicked", f"the confirmation can be refused: {refused}"
    assert kept == [{"name": "Mine", "text": CANARY}], f"refusing keeps the old text: {kept!r}"
    assert accepted == "clicked", accepted
    assert replaced == [{"name": "Mine", "text": OTHER}], (
        f"confirming replaces, not duplicates: {replaced!r}"
    )
    assert natives["confirm"] == 0, f"criterion 2: no window.confirm: {natives}"


@browser
async def test_delete_mine_asks_in_page_and_it_is_gone_after_a_reload(tmp_path):
    """Criterion 2 / Tests to write: with "Mine" saved and chosen, Delete shows the in-page confirmation and removes
    nothing yet; confirming empties the stored list; after a reload "Mine" is not listed and the built-ins are."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, "Mine")
        await settled_store(page, [{"name": "Mine", "text": CANARY}])
        await reload(page)
        await choose(page, "Mine")
        pressed = await click(page, DELETE)
        await settle(page, visible(CONFIRM))
        asked = await confirm_shown(page)
        held = await storage(page)
        accepted = await answer_confirm(page, yes=True)
        gone = await settled_store(page, [])
        natives = await dialogs(page)
        await reload(page)
        probe = await picker(page)
    assert pressed == "clicked", f"Delete is pressable on a user prompt: {pressed}"
    assert asked, "criterion 2: Delete asks first, in the page"
    assert held == [{"name": "Mine", "text": CANARY}], (
        f"nothing removed before the answer: {held!r}"
    )
    assert accepted == "clicked", accepted
    assert gone in ([], None), f"confirming removes it: {gone!r}"
    assert natives["confirm"] == 0, natives
    hits = option_names(probe, [*BUILTIN_NAMES, "Mine"])
    assert "Mine" not in hits and hits[:4] == BUILTIN_NAMES, [o["text"] for o in probe["options"]]


@browser
async def test_builtins_cannot_be_deleted_or_overwritten(tmp_path):
    """Criterion 2 / Tests to write: for each built-in chosen, Delete is disabled or does nothing (if a confirmation
    shows anyway, confirming it is followed through), and after a reload all four are still listed with their own
    texts. Saving CANARY under "Education tutor" (confirming if asked) stores no entry of that name, and choosing
    "Education tutor" still fills TUTOR."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        for name in BUILTIN_NAMES:
            await choose(page, name)
            await asyncio.sleep(0.1)
            if await click(page, DELETE) == "clicked" and await confirm_shown(page):
                await answer_confirm(page, yes=True)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, "Education tutor")
        await asyncio.sleep(0.3)
        if await confirm_shown(page):
            await answer_confirm(page, yes=True)
        await asyncio.sleep(0.3)
        stored = await storage(page)
        await reload(page)
        probe = await picker(page)
        await choose(page, "Education tutor")
        await settle(page, textarea_is(TUTOR))
        view = await box(page)
    names = [p.get("name") for p in (stored or []) if isinstance(p, dict)]
    assert not set(names) & set(BUILTIN_NAMES), (
        f"criterion 2: built-in names cannot be overwritten: {stored!r}"
    )
    hits = option_names(probe, BUILTIN_NAMES)
    assert [h for h in hits if h] == BUILTIN_NAMES, (
        f"all four built-ins survive: {[o['text'] for o in probe['options']]}"
    )
    assert view["value"] == TUTOR, f"the built-in keeps its own text: {view['value'][:80]!r}"


@browser
async def test_a_text_over_the_limit_cannot_be_saved(tmp_path):
    """Criterion 3: a text of exactly 1000 code points (998 ``a`` + two U+1F600: 1002 UTF-16 units) saves as "AtCap"
    (the limit is not too strict, and the library is there); a 1001-code-point text (999 ``a`` + two U+1F600) cannot
    be saved: the store holds no entry for "Long" (confirming if asked); the counter shows it over (``data-over``)."""
    at_cap = "a" * (MAX_USER_INSTRUCTIONS - 2) + ASTRAL * 2
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, at_cap)
        await settle(page, textarea_is(at_cap))
        await save_as(page, "AtCap")
        first = await settled_store(page, [{"name": "AtCap", "text": at_cap}])
        await clear_and_type(page, TOO_LONG)
        await settle(page, textarea_is(TOO_LONG))
        with contextlib.suppress(AssertionError):  # a disabled Save is a legal way to refuse
            await save_as(page, "Long")
        await asyncio.sleep(0.3)
        if await confirm_shown(page):
            await answer_confirm(page, yes=True)
        await asyncio.sleep(0.3)
        stored = await storage(page)
        view = await box(page)
    assert validate_instructions(TOO_LONG) and validate_instructions(at_cap) is None, (
        "premise: 1001 vs 1000"
    )
    assert first == [{"name": "AtCap", "text": at_cap}], (
        f"a text at the limit saves: {str(first)[:200]}"
    )
    names = [p.get("name") for p in (stored or []) if isinstance(p, dict)]
    assert "Long" not in names, (
        f"criterion 3: a text over the limit is not saved: {str(stored)[:200]}"
    )
    assert view["over"] is True, "the counter shows it over"


@browser
@pytest.mark.parametrize("broken", [CORRUPT_JS, BLOCKED_JS], ids=["corrupt-json", "blocked-store"])
async def test_a_corrupt_or_blocked_store_leaves_the_builtins_working_with_one_note(
    tmp_path, broken
):
    """Criterion 3 (adversarial): with the store holding ``{nope`` or ``window.localStorage`` throwing, the panel
    still shows the four built-ins, choosing "Education tutor" fills TUTOR and Apply sends it, and exactly one
    visible note reads "Saved prompts are unavailable in this browser"."""
    async with settings_page(tmp_path, block=False, script=script(broken)) as (live, page, _c):
        await opened(page)
        probe = await picker(page)
        await choose(page, "Education tutor")
        await settle(page, textarea_is(TUTOR))
        await press(page, "Apply")
        await live.until(
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "Apply's set applied",
            timeout_s=PAGE_WAIT_S,
        )
        notes = await page.evaluate(
            "(() => [...document.querySelectorAll(" + json.dumps(PANEL + " *") + ")]"
            ".filter((e) => e.checkVisibility() && [...e.childNodes].some((c) => c.nodeType === 3 && c.textContent.includes("
            + json.dumps(NOTE)
            + "))).map((e) => e.textContent))()"
        )
        tagged = await page.evaluate(
            "(() => { const n = document.querySelector("
            + json.dumps(PROMPTS_NOTE)
            + "); return n ? n.textContent : null; })()"
        )
        sets = await sent_msgs(page)
    assert [h for h in option_names(probe, BUILTIN_NAMES) if h] == BUILTIN_NAMES, probe
    assert len(sets) == 1 and sets[0]["changes"] == {"user_instructions": TUTOR}, sets
    assert len(notes) == 1, f"criterion 3: one note: {notes}"
    assert NOTE in (tagged or ""), f"the note is the [data-settings-prompts-note]: {tagged!r}"


# --- criterion 4 --------------------------------------------------------------------------------------------------


@browser
async def test_a_name_with_markup_renders_as_text(tmp_path):
    """Criterion 4 / Tests to write (adversarial): saving under ``<img src=x onerror=...>`` lists an option whose text
    holds the name literally, before and after a reload; the panel has no ``<img>`` and the handler never ran."""
    probe_js = (
        "(() => ({imgs: document.querySelectorAll(" + json.dumps(PANEL + " img") + ").length,"
        " pwned: window.__t0225_pwned === undefined ? null : String(window.__t0225_pwned)}))()"
    )
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, EVIL_NAME)
        stored = await settled_store(page, [{"name": EVIL_NAME, "text": CANARY}])
        await asyncio.sleep(0.3)
        first = await picker(page)
        dom1 = await page.evaluate(probe_js)
        await reload(page)
        second = await picker(page)
        await choose(page, EVIL_NAME)
        await settle(page, textarea_is(CANARY))
        dom2 = await page.evaluate(probe_js)
    assert stored == [{"name": EVIL_NAME, "text": CANARY}], f"stored verbatim: {stored!r}"
    for probe in (first, second):
        assert any(o["text"].startswith(EVIL_NAME) for o in probe["options"]), [
            o["text"] for o in probe["options"]
        ]
    assert dom1 == {"imgs": 0, "pwned": None} and dom2 == {"imgs": 0, "pwned": None}, (dom1, dom2)


@browser
async def test_keys_in_the_name_field_do_not_mute(tmp_path):
    """Criterion 4 / Tests to write (adversarial): with the name field focused, real key presses ``m``, ``u``, ``t``,
    ``e`` land in it and send no ``astra.mic.mute`` and no ``astra.settings.set``; the capture keeps running."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await page.until(
            "!document.querySelector('#mute-button').disabled", "premise: the capture runs"
        )
        await opened(page)
        await type_name(page, "")
        await page.evaluate(
            "(() => { window.__t0225_keys = []; window.addEventListener('keydown',"
            " (e) => window.__t0225_keys.push(e.key), true); return 1; })()"
        )
        before = await rec(page)
        for ch in "mute":
            await asyncio.to_thread(page.chrome.key, ch, text=ch)
        await settle(
            page,
            "(() => { const e = document.querySelector("
            + json.dumps(NAME)
            + "); return Boolean(e) && e.value === 'mute'; })()",
        )
        await asyncio.sleep(1.0)
        value = await page.evaluate(
            "(() => { const e = document.querySelector("
            + json.dumps(NAME)
            + "); return e ? e.value : null; })()"
        )
        keys = await page.evaluate("window.__t0225_keys || []")
        stop_enabled = await page.evaluate("!document.querySelector('#mute-button').disabled")
        after = await rec(page)
        sets = await sent_msgs(page)
    assert "m" in (keys or []), f"premise: the m keydown reached the page: {keys}"
    assert value == "mute", f"the keys landed in the name field: {value!r}"
    new = after["sent"][len(before["sent"]) :]
    assert "astra.mic.mute" not in new, (
        f"criterion 4: m in the name field does not mute; sent {new}"
    )
    assert sets == [], sets
    assert stop_enabled is True, "the capture keeps running"


# --- criterion 7: Export / Import ---------------------------------------------------------------------------------

EXPORT = PANEL + " [data-settings-prompts-export]"
IMPORT = PANEL + " [data-settings-prompts-import]"

EXPORTED_JS = """(async () => {
  const d = window.__t0225_downloads;
  if (!d) return {error: 'no recorder'};
  const c = d.clicks.filter((x) => x.download !== null);
  if (!c.length) return {clicks: d.clicks.length, error: 'no download link was clicked'};
  const last = c[c.length - 1];
  const blob = d.blobs[last.href];
  return {download: last.download, href_scheme: last.href.split(':')[0], n: c.length,
          type: blob ? blob.type : null, text: blob ? await blob.text() : null};
})()"""


def import_js(text: str, name: str = "gearvis-prompts.json") -> str:
    """Put ``text`` into the import file input as the user's chosen file (a ``DataTransfer`` file list, the in-page
    equivalent of a file chooser) and fire ``change``. Returns what the input looked like."""
    return (
        "(() => { const i = document.querySelector("
        + json.dumps(IMPORT)
        + "); if (!i) return {error: 'no import input'};"
        " if (i.tagName !== 'INPUT' || i.type !== 'file') return {error: 'not a file input: ' + i.outerHTML.slice(0, 120)};"
        " const dt = new DataTransfer(); dt.items.add(new File(["
        + json.dumps(text)
        + "], "
        + json.dumps(name)
        + ", {type: 'application/json'})); i.files = dt.files;"
        " i.dispatchEvent(new Event('input', {bubbles: true})); i.dispatchEvent(new Event('change', {bubbles: true}));"
        " return {accept: i.getAttribute('accept') || ''}; })()"
    )


def counted(text: str, n: int, word: str) -> bool:
    """``text`` says ``n`` next to ``word``: "1 added", "1 prompt added" or "added: 1"."""
    return bool(
        re.search(rf"(?<!\d){n}\s+(?:\w+\s+)?{word}\b|\b{word}\W{{0,3}}{n}(?!\d)", text, re.I)
    )


def panel_text_js() -> str:
    return "(() => (document.querySelector(" + json.dumps(PANEL) + ") || {}).textContent || '')()"


@browser
async def test_export_downloads_a_json_file_that_round_trips_through_import_on_a_fresh_page(
    tmp_path,
):
    """Criterion 7 / Tests to write: with "Mine" = CANARY and "Beta" = OTHER saved, Export offers
    ``gearvis-prompts.json`` through a ``blob:`` link whose content is ``{"version": 1, "prompts": [...]}``, the two
    user prompts only. On a fresh page (a new profile and origin) holding "Mine" = "old text", importing that file
    asks once in the page (one existing name), changes nothing before the answer, and after Yes the store holds Beta =
    OTHER and Mine = CANARY; both are listed and "Mine" fills the box with CANARY; a summary line reports the counts
    (1 added, 1 replaced, 0 skipped). No native dialog, nothing sent to the server."""
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    async with settings_page(first_dir, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        for name, text in (("Mine", CANARY), ("Beta", OTHER)):
            await clear_and_type(page, text)
            await settle(page, textarea_is(text))
            await save_as(page, name)
            await settle(
                page,
                "(() => { try { return (JSON.parse(localStorage.getItem("
                + json.dumps(KEY)
                + ")) || [])"
                ".some((p) => p.name === "
                + json.dumps(name)
                + "); } catch (_) { return false; } })()",
            )
        pressed = await click(page, EXPORT)
        await settle(
            page,
            "(() => Boolean(window.__t0225_downloads && window.__t0225_downloads.clicks.length))()",
        )
        exported = await page.evaluate(EXPORTED_JS)
        await asyncio.sleep(
            0.3
        )  # a popover closed by the link's own click shows within this window
        still_open = await page.evaluate(
            "(() => { const p = document.querySelector('[data-settings-popover]'); return Boolean(p) && !p.hidden; })()"
        )
    assert pressed == "clicked", f"criterion 7: Export is pressable: {pressed}"
    assert still_open is True, (
        "Export keeps the Settings popover open (the download link does not close it)"
    )
    assert isinstance(exported, dict) and not exported.get("error"), (
        f"criterion 7: a download was offered: {exported}"
    )
    assert exported["download"] == "gearvis-prompts.json", exported
    assert exported["href_scheme"] == "blob", f"through a blob link, in the page: {exported}"
    body = json.loads(exported["text"])
    assert body["version"] == 1 and set(body) == {"version", "prompts"}, body
    assert sorted(body["prompts"], key=lambda p: p["name"]) == [
        {"name": "Beta", "text": OTHER},
        {"name": "Mine", "text": CANARY},
    ], f"criterion 7: the user's prompts only, as {{name, text}}: {body}"

    async with settings_page(second_dir, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, "old text")
        await settle(page, textarea_is("old text"))
        await save_as(page, "Mine")
        await settled_store(page, [{"name": "Mine", "text": "old text"}])
        before = await rec(page)
        shape = await page.evaluate(import_js(exported["text"]))
        await settle(page, visible(CONFIRM))
        asked = await confirm_shown(page)
        held = await storage(page)
        accepted = await answer_confirm(page, yes=True)
        want = [{"name": "Beta", "text": OTHER}, {"name": "Mine", "text": CANARY}]
        await settle(
            page,
            "(() => { try { const l = JSON.parse(localStorage.getItem("
            + json.dumps(KEY)
            + ")) || [];"
            " return l.length === 2 && l.some((p) => p.name === 'Beta'); } catch (_) { return false; } })()",
        )
        merged = await storage(page)
        await asyncio.sleep(0.3)
        again = await confirm_shown(page)
        summary = await page.evaluate(panel_text_js())
        natives = await dialogs(page)
        after = await rec(page)
        sets = await sent_msgs(page)
        await reload(page)
        probe = await picker(page)
        await choose(page, "Mine")
        await settle(page, textarea_is(CANARY))
        view = await box(page)
    assert isinstance(shape, dict) and not shape.get("error"), (
        f"criterion 7: an import file input: {shape}"
    )
    assert "application/json" in shape["accept"] and ".json" in shape["accept"], shape
    assert asked, (
        "criterion 7: an imported name that exists is replaced only after an in-page confirmation"
    )
    assert held == [{"name": "Mine", "text": "old text"}], (
        f"nothing changes before the answer: {held!r}"
    )
    assert accepted == "clicked", accepted
    assert sorted(merged or [], key=lambda p: p["name"]) == want, f"merged: {merged!r}"
    assert not again, "one confirmation for the whole import"
    for count, word in ((1, "added"), (1, "replaced"), (0, "skipped")):
        assert counted(summary, count, word), (
            f"criterion 7: a summary line says {count} {word}: {summary[-400:]!r}"
        )
    assert natives == {"confirm": 0, "prompt": 0, "alert": 0}, natives
    new = after["sent"][len(before["sent"]) :]
    assert not [t for t in new if t != "astra.frame"] and sets == [], (
        f"criterion 5: import sends nothing: {new}"
    )
    hits = option_names(probe, [*BUILTIN_NAMES, "Beta", "Mine"])
    assert [h for h in hits if h] == [*BUILTIN_NAMES, "Beta", "Mine"], [
        o["text"] for o in probe["options"]
    ]
    assert view["value"] == CANARY


INVALID_FILES = {
    "invalid-json": "{nope",
    "wrong-version": json.dumps({"version": 2, "prompts": [{"name": "Evil", "text": "x"}]}),
    "oversized": json.dumps(
        {
            "version": 1,
            "prompts": [{"name": "Evil", "text": "x"}, {"name": "Pad", "text": "p" * (300 * 1024)}],
        }
    ),
}


@browser
@pytest.mark.parametrize("kind", sorted(INVALID_FILES))
async def test_an_invalid_import_file_changes_nothing_and_shows_its_reason(tmp_path, kind):
    """Criterion 7 (adversarial): with "Mine" = CANARY saved, importing invalid JSON, a version-2 file, or a file over
    256 KB changes nothing (the store byte for byte), asks nothing, and shows a reason in
    ``[data-settings-prompts-note]``; "Evil" is never listed."""
    async with settings_page(tmp_path, block=False, script=script()) as (_live, page, _c):
        await opened(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await save_as(page, "Mine")
        await settled_store(page, [{"name": "Mine", "text": CANARY}])
        raw_before = await page.evaluate("localStorage.getItem(" + json.dumps(KEY) + ")")
        shape = await page.evaluate(import_js(INVALID_FILES[kind]))
        await settle(
            page,
            "(() => { const n = document.querySelector(" + json.dumps(PROMPTS_NOTE) + ");"
            " return Boolean(n) && n.checkVisibility() && n.textContent.trim().length > 0; })()",
        )
        await asyncio.sleep(0.3)
        note = await page.evaluate(
            "(() => { const n = document.querySelector("
            + json.dumps(PROMPTS_NOTE)
            + "); return n ? n.textContent : null; })()"
        )
        raw_after = await page.evaluate("localStorage.getItem(" + json.dumps(KEY) + ")")
        asked = await confirm_shown(page)
        probe = await picker(page)
    assert isinstance(shape, dict) and not shape.get("error"), shape
    assert raw_after == raw_before, (
        f"criterion 7: an invalid file changes nothing: {str(raw_after)[:200]}"
    )
    assert not asked, "nothing to confirm"
    assert (note or "").strip(), f"criterion 7: its reason is shown inline: {note!r}"
    assert NOTE not in (note or ""), "the reason is the file's, not the store-unavailable note"
    assert not any(o["text"].startswith("Evil") for o in probe["options"]), [
        o["text"] for o in probe["options"]
    ]


# --- criterion 5 --------------------------------------------------------------------------------------------------


@browser
async def test_the_prompt_library_requests_nothing_off_loopback(tmp_path):
    """Criterion 5 / Tests to write: over a load, choosing, Save as, a reload, Delete with its confirmation and an
    Apply, every request the page makes is to loopback (``http``/``ws`` on 127.0.0.1, localhost or [::1];
    ``data:``/``blob:``)."""
    async with settings_page(tmp_path, block=False, script=script()) as (live, page, chrome):
        await opened(page)
        await choose(page, "Browser specialist")
        await settle(page, textarea_is(BROWSER_TEXT))
        await save_as(page, "Mine")
        await settled_store(page, [{"name": "Mine", "text": BROWSER_TEXT}])
        await reload(page)
        await choose(page, "Mine")
        await click(page, DELETE)
        await settle(page, visible(CONFIRM))
        await answer_confirm(page, yes=True)
        await choose(page, "Education tutor")
        await press(page, "Apply")
        await live.until(
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "premise: applied",
            timeout_s=PAGE_WAIT_S,
        )
        await asyncio.to_thread(chrome.drain, 1.0)
        requests = list(chrome.requests)
    assert requests, "the CDP request log is empty, so this row would be vacuous"
    offenders = [
        u
        for u in requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert not offenders, f"zero non-loopback requests; got {offenders}"
    assert not [
        u for u in requests if u.split("://", 1)[0] not in {"http", "ws", "data", "blob"}
    ], requests


#: criterion 5: T-0225's own commits leave these byte-identical (the library only fills the box)
FROZEN = ["hud/src/protocol.js", "hud/src/ws.js", "hud/vendor", "local_astra"]
#: T-0225's implementation commit (its parent is BASE). Both ends are literals (.claude/rules/tests.md, fixed refs):
#: later tasks may change local_astra/ (T-0224 did, after BASE), so the guard reads this task's range only. A later
#: T-0225 fix commit moves IMPL to that commit, in the same change.
IMPL = "dbaec300"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )


def test_t0225s_own_commits_leave_the_wire_the_server_and_the_vendored_code_byte_identical():
    """Criterion 5 (negative): over T-0225's range BASE..IMPL (377bc592..dbaec300), ``hud/src/protocol.js``,
    ``hud/src/ws.js``, ``hud/vendor/**`` and ``local_astra/**`` have no change. Premise: the range is not empty and
    holds the task's page work (``hud/src/settings/prompts.js``), so the row is not vacuous."""
    for ref in (BASE, IMPL):
        if _git("cat-file", "-e", ref + "^{commit}").returncode != 0:
            pytest.skip(f"{ref} is not in this checkout (a git archive copy)")
    touched = _git("diff", "--name-only", BASE, IMPL)
    assert touched.returncode == 0, touched.stderr
    assert "hud/src/settings/prompts.js" in touched.stdout.split(), (
        f"premise: {BASE}..{IMPL} is T-0225's implementation: {touched.stdout.split()}"
    )
    frozen = _git("diff", "--name-only", BASE, IMPL, "--", *FROZEN)
    assert frozen.returncode == 0, frozen.stderr
    changed = frozen.stdout.split()
    assert changed == [], f"criterion 5: T-0225 must leave these byte-identical: {changed}"


def test_the_pages_forbidden_list_is_validate_instructions_list():
    """Criterion 7 (imported texts carry "no special-token strings", the rule of ``validate_instructions``), parity:
    ``hud/src/settings/prompts.js``' ``FORBIDDEN_TEXT``, read through node, is the same set as
    ``prompt_builder.INSTRUCTIONS_FORBIDDEN``. A token added to the server's list without the page's (or the
    reverse) fails here."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed (tests/unit/test_hud_js.py names the prerequisite)")
    module = (REPO / "hud" / "src" / "settings" / "prompts.js").as_uri()
    out = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            f"import {{ FORBIDDEN_TEXT }} from {json.dumps(module)};"
            " process.stdout.write(JSON.stringify(FORBIDDEN_TEXT));",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert out.returncode == 0, (
        f"node could not import prompts.js' FORBIDDEN_TEXT: {out.stderr[-400:]}"
    )
    page = json.loads(out.stdout)
    assert len(page) == len(set(page)), f"no repeats in the page's list: {page}"
    assert set(page) == set(INSTRUCTIONS_FORBIDDEN), (
        f"page only: {sorted(set(page) - set(INSTRUCTIONS_FORBIDDEN))}; "
        f"server only: {sorted(set(INSTRUCTIONS_FORBIDDEN) - set(page))}"
    )
