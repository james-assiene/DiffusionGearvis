"""T-0213 group C: "Your instructions to Gearvis" and the read-only system message, in the real page (C-1 .. C-4).

``docs/tasks/T-0213.md`` C-1 to C-4 and *Tests to write* "test_hud_settings_instructions.py". The server half
(``astra.settings.set``, the apply at the IDLE boundary, ``validate_instructions``' refusals, ``system_preview`` and
``instructions_max`` in ``astra.settings.state``) is merged (A1, A2-read, A2-apply at 96a4d27b) and real here; what
these rows pin is the page.

**Rig**: ``tests/integration/test_hud_settings.py``'s ``settings_page`` (``test_hud_rehearsal.Live``: a real
``Session`` and ``HudServer(port=0)`` over the fakes of ``tests/unit/_hud_rig.py``, tools off, ``SessionClock`` wall
time, fake LLM replaying ``"Fine. "``), the page over loopback http with ``?source=canvas``, headless Chrome over
CDP with the request log and an init script that records every message the page sends (``init_js_full``). Turns are
spoken through ``Live.ask``. Text is typed with CDP ``Input.insertText`` into the focused textarea (a user's typing or
paste; the page's own ``input`` handlers run). No number here is a latency claim; every wait is a generous ceiling;
n = 1 per row. Runs only with ``ASTRA_BROWSER_TESTS=1``.

**Oracles** (never the page's code): the system message of each fake LLM request (``Message.text()``, as the real
client renders it); ``prompt_builder.USER_INSTRUCTIONS_HEADER`` and ``validate_instructions`` (A1, merged and tested
on their own) for the block and the refusal reasons; the server's own ``astra.settings.state`` on the wire;
``hashlib`` for the sha12.

**DOM readings** (the criteria name the behaviour; the hook names are the C implementer's, relayed by the
orchestrator, used only where the criteria leave the choice open):

- the editor lives in ``[data-settings-instructions]`` (inside the ``user_instructions`` row or beside it under
  "Guidance and prompt"): its ``<textarea>``, a live counter, and two ``<button>``s reading "Apply" and "Reset to
  default"; the label "Your instructions to Gearvis" (C-1) is shown with it;
- the counter is the element of the box (not the textarea, not a button) whose text holds two integers, in order the
  code points typed and ``instructions_max``; it carries ``data-over`` when over the cap;
- the refusal reason is shown inside the box or the ``user_instructions`` row (asserted as a substring: the page may
  prefix it, e.g. "Not applied: ");
- the disclosure is a ``<details>`` whose ``<summary>`` reads "Show the full system message"; one element in it has
  ``textContent`` equal to ``system_preview`` (a ``pre[data-settings-system-preview]``), and nothing in it is editable;
  the C-2 note is shown in the panel verbatim.

**Activated** (T-0213 group C): the rows that need the C page no longer carry the pending marker; the rig
self-check holds at 96a4d27b. ``tests/hud/settings_state_wire.test.mjs``'s group C gate row is a strict xfail only
while this module carries the T-0213 pending marker string, so it is active now.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import pytest

from local_astra import settings_catalog as sc
from local_astra.conductor.prompt_builder import (
    MAX_USER_INSTRUCTIONS,
    USER_INSTRUCTIONS_HEADER,
    validate_instructions,
)
from local_astra.contracts import protocol
from local_astra.contracts.codec import encode
from local_astra.contracts.turn import TurnState
from tests.integration.test_hud_rehearsal import Live, Page
from tests.integration.test_hud_settings import (
    OPEN_JS,
    PAGE_WAIT_S,
    SET_TYPE,
    browser,
    hello_state,
    init_js_full,
    open_panel,
    post_synthetic,
    rec,
    send_raw_set,
    sent_msgs,
    settings_page,
    settle,
    synthetic_state,
    wire_states,
)

pytestmark = [pytest.mark.integration]

#: C-1, verbatim
LABEL = "Your instructions to Gearvis"
APPLY = "Apply"
RESET = "Reset to default"
#: C-2, verbatim
SUMMARY = "Show the full system message"
SAFETY_NOTE = (
    "Safety rules (no passwords or card numbers, ask before buying or sending, the web fence) are enforced by "
    "Gearvis's code, not by this text."
)
#: a canary no other text in the session holds
CANARY = "ZQX-T0213-C answer in British English and keep it short."
#: C-2 adversarial: markup in the user's own text must reach the preview as text
EVIL = 'ZQX-T0213-C <img src=x onerror="window.__t0213c_pwned=1"> be brief'
#: C-3: 1001 code points, 1003 UTF-16 units (two astral characters)
ASTRAL = "\U0001f600"
TOO_LONG = "a" * (MAX_USER_INSTRUCTIONS - 1) + ASTRAL * 2
#: C-3: a special-token string (A1-4)
SPECIAL = "be brief <|think|> always"
Q1 = "what is on this screen"
Q2 = "and what about now"
Q3 = "one more question please"

BOX = "[data-settings-panel] [data-settings-instructions]"


def block(text: str) -> str:
    """A1-4: the user's block, the last part of the system message."""
    return USER_INSTRUCTIONS_HEADER + "\n" + text


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def system_texts(live: Live) -> list[str]:
    """The system message of every non-caption turn request (prefills and generates), in order."""
    out = []
    for _rid, prompt in [*live.llm.prefills, *live.llm.gen_prompts]:
        systems = [m.text() for m in prompt.messages if m.role == "system"]
        assert len(systems) == 1, f"premise: one system message per request; got {len(systems)}"
        out.append(systems[0])
    return out


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


BOX_JS = """(() => {
  const box = document.querySelector(__BOX__);
  const row = document.querySelector('[data-settings-panel] [data-setting-key="user_instructions"]');
  if (!box) return {present: false, row_text: row ? row.textContent : null};
  const ta = box.querySelector('textarea');
  const buttons = [...box.querySelectorAll('button')].map((b) => ({text: b.textContent.trim(), disabled: b.disabled}));
  const leaves = [...box.querySelectorAll('*')].filter((e) => !e.matches('textarea, button')
    && ![...e.children].some((c) => !c.matches('textarea, button') && /\\d/.test(c.textContent || ''))
    && /\\d+\\D+\\d+/.test(e.textContent || ''));
  const counter = leaves.find((e) => e.hasAttribute('data-over')) || leaves[0] || null;
  return {
    present: true,
    value: ta ? ta.value : null,
    textarea: ta ? {disabled: ta.disabled, readonly: ta.readOnly} : null,
    focused: ta ? document.activeElement === ta : false,
    buttons,
    counter: counter ? counter.textContent : null,
    over: counter ? counter.hasAttribute('data-over') && counter.getAttribute('data-over') !== 'false' : null,
    box_text: box.textContent,
    row_text: row ? row.textContent : null,
    panel_text: (document.querySelector('[data-settings-panel]') || {}).textContent || '',
  };
})()""".replace("__BOX__", json.dumps(BOX))


async def box(page: Page) -> dict[str, Any]:
    got = await page.evaluate(BOX_JS)
    assert isinstance(got, dict), f"the instructions probe threw: {got!r}"
    assert got.get("present"), (
        f"C-1: the instructions editor ({BOX}) is not in the panel; the user_instructions row reads "
        f"{got.get('row_text')!r}"
    )
    return got


async def focus_textarea(page: Page) -> None:
    outcome = await page.evaluate(
        "(() => { const t = document.querySelector("
        + json.dumps(BOX + " textarea")
        + "); if (!t) return 'no textarea'; if (t.disabled) return 'disabled'; t.focus();"
        " return document.activeElement === t ? 'ok' : 'not focused'; })()"
    )
    assert outcome == "ok", f"C-1: the instructions textarea takes the focus: {outcome!r}"


async def clear_and_type(page: Page, text: str) -> None:
    """Select everything in the textarea and type ``text`` over it (CDP ``Input.insertText``)."""
    await focus_textarea(page)
    await page.evaluate(
        "(() => { const t = document.querySelector(" + json.dumps(BOX + " textarea") + ");"
        " t.select(); return 1; })()"
    )
    if text:
        await asyncio.to_thread(page.chrome._cmd, "Input.insertText", text=text)
    else:
        await asyncio.to_thread(page.chrome.key, "Backspace", code=8)


async def press(page: Page, label: str) -> None:
    outcome = await page.evaluate(
        "(() => { const b = [...document.querySelectorAll(" + json.dumps(BOX + " button") + ")]"
        ".filter((e) => e.textContent.trim() === " + json.dumps(label) + ");"
        " if (b.length !== 1) return b.length + ' buttons'; if (b[0].disabled) return 'disabled';"
        " b[0].click(); return 'clicked'; })()"
    )
    assert outcome == "clicked", f"C-1: one enabled {label!r} button in the editor: {outcome!r}"


async def until_rev(live: Live, rev: int, what: str) -> Any:
    await live.until(
        lambda: any(s.rev == rev for s in wire_states(live)), what, timeout_s=PAGE_WAIT_S
    )
    return [s for s in wire_states(live) if s.rev == rev][-1]


async def opened(page: Page) -> None:
    await open_panel(page, len(sc.CATALOG))


def textarea_is(text: str) -> str:
    """T-0218 B condition: the instructions textarea holds ``text``."""
    return (
        "(() => { const t = document.querySelector("
        + json.dumps(BOX + " textarea")
        + "); return Boolean(t) && t.value === "
        + json.dumps(text)
        + "; })()"
    )


def counter_reads(text: str, n: int) -> str:
    """T-0218 B condition: the textarea holds ``text`` and the editor shows ``n`` as a whole number (the counter
    followed the input); the row's own assertion then reads the counter exactly."""
    return (
        "(() => { const b = document.querySelector("
        + json.dumps(BOX)
        + "); const t = b && b.querySelector('textarea'); return Boolean(t) && t.value === "
        + json.dumps(text)
        + " && /(^|\\D)"
        + str(n)
        + "(\\D|$)/.test(b.textContent || ''); })()"
    )


def reason_shown(reason: str) -> str:
    """T-0218 B condition: ``reason`` is in the editor's or its row's text (where C-3 reads it)."""
    return (
        "(() => { const b = document.querySelector("
        + json.dumps(BOX)
        + "); const r = document.querySelector('[data-settings-panel] [data-setting-key=\"user_instructions\"]');"
        " return ((b ? b.textContent : '') + (r ? r.textContent : '')).includes("
        + json.dumps(reason)
        + "); })()"
    )


def preview_shown(preview: str) -> str:
    """T-0218 B condition: one element of the "Show the full system message" disclosure reads ``preview`` exactly
    (read without opening the disclosure)."""
    return (
        "(() => { const p = document.querySelector('[data-settings-panel]'); if (!p) return false;"
        " return [...p.querySelectorAll('details')].some((d) => { const s = d.querySelector('summary');"
        " return Boolean(s) && s.textContent.trim() === "
        + json.dumps(SUMMARY)
        + " && [...d.querySelectorAll('*')].some((e) => !e.matches('summary') && e.textContent === "
        + json.dumps(preview)
        + "); }); })()"
    )


def popover_hidden() -> str:
    return "(() => { const p = document.querySelector('[data-settings-popover]'); return Boolean(p) && p.hidden; })()"


# --- rig ----------------------------------------------------------------------------------------------------------


@browser
async def test_rig_a_raw_instructions_set_reaches_the_next_request(tmp_path):
    """Rig self-check (holds at 96a4d27b): an ``astra.settings.set {rev: 0, changes: {user_instructions: CANARY}}``
    written straight onto the page's socket of this tools-off rig is applied (rev 1, ``instructions`` the text) and
    the next turn's system message ends with the block. So the C rows fail on the page, not on the rig."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        await turn(live, Q1, 1)
        base = last_generate_system(live)
        await send_raw_set(
            page, {"type": SET_TYPE, "rev": 0, "changes": {"user_instructions": CANARY}}
        )
        state = await until_rev(live, 1, "rig: the instructions set applied")
        await turn(live, Q2, 2)
        after = last_generate_system(live)
    assert state.instructions == CANARY
    assert CANARY not in base, "premise: turn 1 is on BASE"
    assert after.endswith(block(CANARY)) and after.startswith(base), after[-300:]


# --- C-1 ----------------------------------------------------------------------------------------------------------


@browser
async def test_c1_apply_sends_the_text_and_the_next_request_ends_with_the_block(tmp_path):
    """C-1 / Tests to write: after a turn on BASE, typing CANARY and pressing Apply sends exactly one
    ``astra.settings.set`` with the rev in force (0) and ``changes == {user_instructions: CANARY}``; the server applies
    it (rev 1); the next turn's system message ends with ``USER_INSTRUCTIONS_HEADER + "\\n" + CANARY`` and starts with
    the BASE system message (the block is appended last, A1-4). The editor is labelled "Your instructions to
    Gearvis"."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        await turn(live, Q1, 1)
        base = last_generate_system(live)
        await opened(page)
        first = await box(page)
        await clear_and_type(page, CANARY)
        await settle(page, textarea_is(CANARY))
        await asyncio.sleep(0.3)  # nothing-sent window (C-4), started once the text is in the box
        typed = await sent_msgs(page)
        await press(page, APPLY)
        state = await until_rev(live, 1, "C-1: Apply's set was applied (rev 1)")
        await asyncio.sleep(0.3)  # nothing-more-sent window, started after the settled rev 1
        sets = await sent_msgs(page)
        await turn(live, Q2, 2)
        after = last_generate_system(live)
    assert LABEL in (first["row_text"] or "") + first["box_text"], (
        f"C-1: the editor is labelled {LABEL!r}"
    )
    assert typed == [], f"C-4: typing alone sends no set; got {typed}"
    assert len(sets) == 1, f"C-1: one Apply, one astra.settings.set; got {sets}"
    assert sets[0]["rev"] == 0 and sets[0]["changes"] == {"user_instructions": CANARY}, sets[0]
    assert not sets[0].get("stop_turn", False), sets[0]
    assert state.instructions == CANARY and not state.refused, (state.instructions, state.refused)
    assert CANARY not in base, "premise: turn 1 is on BASE"
    assert after.endswith(block(CANARY)), (
        f"C-1: the next request ends with the block: {after[-300:]!r}"
    )
    assert after.startswith(base), "A1-4: everything before the block is BASE's system message"


@browser
async def test_c1_reset_to_default_restores_the_base_sha(tmp_path):
    """C-1 / Tests to write: after CANARY is applied, "Reset to default" sends exactly one more set with
    ``changes == {user_instructions: ""}`` and the rev in force (1); the server's state (rev 2) carries the hello's
    ``system_sha12`` again, which is the sha12 of turn 1's own system message, and the next turn's system message is
    turn 1's byte for byte; the textarea is empty."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        hello = await hello_state(live)
        await turn(live, Q1, 1)
        base = last_generate_system(live)
        await opened(page)
        await clear_and_type(page, CANARY)
        await press(page, APPLY)
        applied = await until_rev(live, 1, "premise: CANARY applied (rev 1)")
        # T-0218 B: rev 1 is on the wire before the page has read it; Reset pressed before then goes out with rev 0
        # (stale). Wait on the page holding rev 1: C-2's disclosure shows rev 1's system_preview.
        await page.until(
            preview_shown(applied.system_preview or ""),
            "premise: the rev 1 state reached the page (the disclosure shows its system_preview)",
            timeout_s=PAGE_WAIT_S,
        )
        await press(page, RESET)
        reset = await until_rev(live, 2, "C-1: Reset to default's set was applied (rev 2)")
        await settle(page, textarea_is(""))
        # nothing-more-sent window, started after rev 2 and the box followed it
        await asyncio.sleep(0.3)
        sets = await sent_msgs(page)
        after_reset = await box(page)
        await turn(live, Q3, 2)
        after = last_generate_system(live)
    assert hello.system_sha12 == sha12(base), "premise: the hello's sha is BASE's (the oracle)"
    assert applied.system_sha12 != hello.system_sha12, "premise: the block moved the sha"
    assert len(sets) == 2, sets
    assert sets[1]["changes"] == {"user_instructions": ""} and sets[1]["rev"] == 1, sets[1]
    assert reset.system_sha12 == hello.system_sha12 and reset.instructions == "", (
        reset.system_sha12,
        reset.instructions,
    )
    assert after == base, "C-1: after Reset the system message is BASE's, byte for byte"
    assert after_reset["value"] == "", f"the textarea follows the reset: {after_reset['value']!r}"


@browser
async def test_c1_the_counter_counts_code_points_against_instructions_max(tmp_path):
    """C-1 (synthetic state, ``instructions_max = 40``, so a page with 1000 hard-coded fails): the counter reads the
    code points typed and the state's ``instructions_max``, live on each input: "ab" + two U+1F600 + "c" is 5 code
    points (7 UTF-16 units) and reads [5, 40]; 36 more U+1F600 make 41, over the cap: [41, 40] and ``data-over``
    set; at 5 it is not over. Nothing is sent while typing (C-4)."""
    raw, body = synthetic_state()
    body = {**body, "instructions_max": 40}
    raw = encode(protocol.AstraSettingsState.model_validate(body))
    first = "ab" + ASTRAL * 2 + "c"
    async with settings_page(tmp_path, block=True, script=init_js_full(block=True)) as (
        _live,
        page,
        _c,
    ):
        await post_synthetic(page, raw)
        await opened(page)
        await clear_and_type(page, first)
        await settle(page, counter_reads(first, 5))
        at5 = await box(page)
        await asyncio.to_thread(page.chrome._cmd, "Input.insertText", text=ASTRAL * 36)
        await settle(page, counter_reads(first + ASTRAL * 36, 41))
        # nothing-sent window (C-4), started once the counter followed the last input
        await asyncio.sleep(0.3)
        at41 = await box(page)
        sets = await sent_msgs(page)
    assert len(first) == 5 and len(first.encode("utf-16-le")) // 2 == 7, "premise"
    assert at5["value"] == first, at5["value"]
    ints5 = [int(n) for n in re.findall(r"\d+", at5["counter"] or "")]
    ints41 = [int(n) for n in re.findall(r"\d+", at41["counter"] or "")]
    assert ints5 == [5, 40], (
        f"C-1: the counter counts code points against instructions_max: {at5['counter']!r}"
    )
    assert ints41 == [41, 40], f"C-1: the counter is live: {at41['counter']!r}"
    assert at41["over"] is True and at5["over"] is False, (at5["over"], at41["over"])
    assert sets == [], f"C-4: typing sends nothing; got {sets}"


# --- C-3 ----------------------------------------------------------------------------------------------------------


@browser
@pytest.mark.parametrize("text", [TOO_LONG, SPECIAL], ids=["1001-code-points", "special-token"])
async def test_c3_a_refused_text_shows_the_servers_reason_inline_and_stays_in_the_box(
    tmp_path, text
):
    """C-3 / Tests to write: a 1001-code-point text (999 ``a`` and two U+1F600: 1003 UTF-16 units, so a page that
    counts units or truncates at a ``maxlength`` fails) and a special-token text are sent on Apply as typed (the
    server judges, A1-4); the session refuses (rev stays 0; ``refused`` names ``user_instructions`` with
    ``validate_instructions``' own reason); the page shows that reason inside the editor or its row, and the textarea
    still holds the whole text. Nothing changes: the next turn's system message is BASE's."""
    reason = validate_instructions(text)
    assert reason, f"premise: A1-4 refuses {text[:40]!r}"
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        await opened(page)
        await clear_and_type(page, text)
        await press(page, APPLY)
        await live.until(
            lambda: any(s.refused for s in wire_states(live)),
            "C-3: the session answered the set with a refusal",
            timeout_s=PAGE_WAIT_S,
        )
        refused = [s for s in wire_states(live) if s.refused][-1]
        await settle(page, reason_shown(reason))
        # nothing-more-sent window, started after the refusal and its shown reason
        await asyncio.sleep(0.5)
        shown = await box(page)
        sets = await sent_msgs(page)
        await turn(live, Q1, 1)
        after = last_generate_system(live)
    assert len(sets) == 1 and sets[0]["changes"] == {"user_instructions": text}, (
        f"C-3: the page sends the text as typed: {[(s.get('changes') or {}) for s in sets][:1]!r}"[
            :400
        ]
    )
    pairs = [(r.key, r.reason) for r in refused.refused]
    assert pairs == [("user_instructions", reason)], pairs
    assert refused.rev == 0 and refused.instructions == "", (refused.rev, refused.instructions)
    where = shown["box_text"] + (shown["row_text"] or "")
    assert reason in where, (
        f"C-3: the server's reason {reason!r} is shown inline; the editor reads {where[-400:]!r}"
    )
    assert shown["value"] == text, "C-3: the refused text stays in the box, whole"
    assert USER_INSTRUCTIONS_HEADER not in after, "a refused set changes nothing"


# --- C-4 ----------------------------------------------------------------------------------------------------------


@browser
async def test_c4_m_in_the_textarea_does_not_mute_and_unapplied_text_sends_nothing(tmp_path):
    """C-4 (adversarial): with the textarea focused, real key presses ``m``, ``u``, ``t``, ``e`` land in it (the
    keydown reached the page) and send no ``astra.mic.mute`` and no ``astra.settings.set``; the capture keeps
    running (Stop sharing stays enabled, frames keep leaving). Closing the popover with the text unapplied sends no
    set either; the next turn's system message carries no block."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        await page.until(
            "!document.querySelector('#mute-button').disabled",
            "premise: the synthetic capture runs (Stop sharing enabled)",
        )
        await opened(page)
        await focus_textarea(page)
        await page.evaluate(
            "(() => { window.__t0213c_keys = []; window.addEventListener('keydown',"
            " (e) => window.__t0213c_keys.push(e.key), true); return 1; })()"
        )
        before = await rec(page)
        for ch in "mute":
            await asyncio.to_thread(page.chrome.key, ch, text=ch)
        await settle(page, textarea_is("mute"))
        # nothing-sent window (no mute, no set), started once the keys landed
        await asyncio.sleep(1.0)
        typed = await box(page)
        keys = await page.evaluate("window.__t0213c_keys || []")
        stop_enabled = await page.evaluate("!document.querySelector('#mute-button').disabled")
        await asyncio.to_thread(page.chrome.key, "Escape", code=27)
        await settle(page, popover_hidden())
        # nothing-sent window (closing sends no set), started once the popover closed
        await asyncio.sleep(0.8)
        closed = await page.evaluate("document.querySelector('[data-settings-popover]').hidden")
        after = await rec(page)
        sets = await sent_msgs(page)
        await turn(live, Q1, 1)
        system = last_generate_system(live)
    assert "m" in (keys or []), f"premise: the m keydown reached the page: {keys}"
    assert typed["value"] == "mute", f"C-4: the keys landed in the textarea: {typed['value']!r}"
    new = after["sent"][len(before["sent"]) :]
    assert "astra.mic.mute" not in new, f"C-4: m in the textarea does not mute; sent {new}"
    assert not [t for t in new if str(t).startswith("astra.settings")], f"C-4: sent {new}"
    assert sets == [], sets
    assert stop_enabled is True, "C-4: the capture keeps running"
    assert new.count("astra.frame") >= 1, f"the capture kept sending frames: {new}"
    assert closed is True, "premise: Escape closed the popover"
    assert USER_INSTRUCTIONS_HEADER not in system, "C-4: unapplied text changes nothing"


# --- C-2 ----------------------------------------------------------------------------------------------------------

DISCLOSURE_JS = """(() => {
  const panel = document.querySelector('[data-settings-panel]');
  if (!panel) return {found: 0};
  const ds = [...panel.querySelectorAll('details')].filter((d) => {
    const s = d.querySelector('summary'); return s && s.textContent.trim() === __SUMMARY__; });
  if (ds.length !== 1) return {found: ds.length};
  const d = ds[0];
  d.open = true;
  const want = __WANT__;
  const hits = [...d.querySelectorAll('*')].filter((e) => !e.matches('summary') && e.textContent === want);
  const editable = [...d.querySelectorAll('input, select, textarea, [contenteditable]')]
    .filter((e) => e.isContentEditable || ((e.matches('input, textarea') && !e.readOnly && !e.disabled)) || e.matches('select:not([disabled])'))
    .map((e) => e.outerHTML.slice(0, 80));
  return {
    found: 1,
    hits: hits.map((e) => ({tag: e.tagName.toLowerCase(), editable: e.isContentEditable})),
    editable,
    imgs: d.querySelectorAll('img').length,
    pwned: window.__t0213c_pwned === undefined ? null : String(window.__t0213c_pwned),
    panel_text: panel.textContent,
  };
})()"""


def disclosure_js(want: str) -> str:
    return DISCLOSURE_JS.replace("__SUMMARY__", json.dumps(SUMMARY)).replace(
        "__WANT__", json.dumps(want)
    )


@browser
async def test_c2_the_full_system_message_is_shown_read_only_as_text_with_the_safety_note(tmp_path):
    """C-2: one disclosure "Show the full system message" in the panel shows the hello state's ``system_preview``
    (the persona, the guidance, the UI block, the tool instructions: the server's assembled message) as the exact
    ``textContent`` of one element, nothing in it editable; the panel shows the C-2 safety note verbatim. Adversarial:
    after instructions holding ``<img src=x onerror=...>`` are applied, the disclosure shows the new
    ``system_preview`` (ending with the block, markup as text): no ``<img>``, the handler never ran."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, _c):
        preview = (await hello_state(live)).system_preview
        assert preview, "premise: A2-apply's state carries system_preview"
        await opened(page)
        base = await page.evaluate(disclosure_js(preview))
        assert validate_instructions(EVIL) is None, "premise: the markup text is legal instructions"
        await clear_and_type(page, EVIL)
        await press(page, APPLY)
        state = await until_rev(live, 1, "premise: the markup text applied (rev 1)")
        new_preview = state.system_preview
        await page.until(
            f"(() => {{ const r = {disclosure_js(new_preview or '')}; return r.found === 1 && r.hits.length >= 1; }})()",
            "C-2: the disclosure follows the server's new system_preview",
            timeout_s=PAGE_WAIT_S,
        )
        evil = await page.evaluate(disclosure_js(new_preview or ""))
    assert base.get("found") == 1, f"C-2: one {SUMMARY!r} disclosure in the panel: {base}"
    assert base["hits"], "C-2: an element of the disclosure reads system_preview exactly"
    assert all(not h["editable"] for h in base["hits"]) and base["editable"] == [], base
    assert SAFETY_NOTE in base["panel_text"], "C-2: the safety note is shown verbatim"
    assert new_preview and new_preview.endswith(block(EVIL)), (
        "premise: the server's preview carries the block"
    )
    assert evil["hits"] and evil["imgs"] == 0 and evil["pwned"] is None, evil


# --- negative: loopback only ---------------------------------------------------------------------------------------


@browser
async def test_the_instructions_editor_requests_nothing_off_loopback(tmp_path):
    """B-4 / C negative: over a page load, typing markup instructions, Apply, the disclosure opened and Reset, every
    request the page makes is to loopback (``http``/``ws`` on 127.0.0.1, localhost or [::1]; ``data:``/``blob:``)."""
    async with settings_page(tmp_path, block=False, script=init_js_full()) as (live, page, chrome):
        await opened(page)
        await clear_and_type(page, EVIL)
        await press(page, APPLY)
        await until_rev(live, 1, "premise: applied")
        await page.evaluate(disclosure_js(""))
        await press(page, RESET)
        await until_rev(live, 2, "premise: reset")
        await page.evaluate(OPEN_JS)
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
