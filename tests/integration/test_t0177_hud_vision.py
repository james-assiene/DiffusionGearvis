"""The vision selector in the HUD, in headless Chrome (docs/tasks/T-0177.md group C, C1-C5).

*Tests to write*, Group C (Chrome): no select is sent before a state arrives; the combo follows the echo,
not the click; a Share click sends ``select shared`` and ``?source=canvas`` sends nothing; Mute sends
``select browser``; adversarial: the local capture draws a red canvas while the stub pushes a blue frame,
and ``#preview`` stays blue across ten capture ticks; the caption; a blind frame hides the image; no requests
off loopback. Plus the rest of C2's stop list (the reset's stop, the track's ``ended``), C3's empty
sentences and reset, C4's thumbnails, ``+N`` and Live badge, both themes. T-0181 B3 (H-T181-4, James
2026-10-28) removed the ``m`` shortcut: S10 and C-R2 (c)'s ``m`` half now pin "``m`` does nothing"
(pending T-0181 until group B activates).

Regime: the box's Chrome (or Playwright's pinned Chromium) headless over CDP, the T-0162 rig shape
(``tests/integration/test_hud_tools_pane.py``: one fixture drives the page through every phase and records
probes; each test asserts on one), 1440x900, the stub Conductor ``tests/fakes/hud_server.HudStub`` (no
Session: group B's published frame is not merged, so every ``astra.vision.state``/``astra.vision.frame`` is
built by the fake's ``vision_state``/``vision_frame`` from A1's real models and pushed). A "Share" is a real
mouse click on ``#share-button`` with ``getDisplayMedia`` replaced by an init script that returns a real
``canvas.captureStream`` painted red (so ``capture.synthetic`` is false, as for a human's Share; the T-0154
redesign rig's technique). No measured number.

**The DOM contract this module pins** (the task fixes behaviour; these hooks are the test-author's names,
which the implementer provides):

=================================================  ===========================================================
``main > .pane.thumb select[data-vision-source]``  the combo (C1); absent or not rendered before a state
its accessible label (``<label>`` or aria-label)   contains "Looking at"
``option[value=browser|shared]``                   "Gearvis's browser"; "Shared screen" / "Shared camera"
``#preview``                                       the newest image of the newest ``astra.vision.frame``
``main > .pane.thumb img[data-vision-thumb]``      one visible thumbnail per earlier image of that frame
``main > .pane.thumb [data-empty]``                the visible empty sentence(s) of the pane
``main > .pane.thumb [data-capture-caption]``      the caption (C4)
``main > .pane.thumb .live``                       the Live badge (visible or not)
=================================================  ===========================================================

Readings (stated, not guessed silently): "``#preview`` shows an image" means it is rendered with a decoded
picture (visible, ``complete``, ``naturalWidth > 0``; the probe reads its centre pixel); "shows no image"
means no such picture, so an empty letterbox may stay on screen (``#preview[hidden]`` still computes
``display: block`` today). "its value is always the last echo" is read as: after the user
changes the combo and before any echo, the select's ``value`` reads the last echoed source again (within
1 s). "the newest large" is read within one frame (the last image, B7's order). A select "once" per Share is
pinned as exactly one ``astra.vision.select`` per Share click (the per-step delta below).

Group B: every row here is driven by the fake's builders; an end-to-end check that the panel shows the bytes a
real Session sent needs Merge T-0177-b (its ``test_t0177_vision_frame_publish.py`` pins the server side).
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub, vision_frame, vision_image, vision_state
from tests.integration.test_hud_chrome import _contrast_report, choose_theme_js, measure_js
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_redesign import _panels_line, _Recorder

pytestmark = pytest.mark.integration

LIVE_EMPTY = "Nothing sent yet: Gearvis looks when you ask"
BLIND_EMPTY = "No image was sent for this question"
#: The What I see pane's empty line in hud/index.html at c0cb001e (T-0154's existing text, C3: "other
#: modes show the existing T-0154 empty text").
NONLIVE_EMPTY = "no frame yet — press Share"
SEP = " · "

BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
YELLOW = (255, 220, 0)


def _jpeg(rgb: tuple[int, int, int], w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _is(rgb: Any, name: str) -> bool:
    if not isinstance(rgb, list) or len(rgb) < 3:
        return False
    r, g, b = (int(x) for x in rgb[:3])
    return {
        "blue": b > 170 and r < 70 and g < 70,
        "red": r > 170 and g < 70 and b < 70,
        "green": g > 140 and r < 70 and b < 70,
        "yellow": r > 170 and g > 150 and b < 90,
    }[name]


#: The local capture: a real MediaStream from a canvas painted red (a small dark mark moves along the top
#: edge so successive frames differ; the centre pixel is always red). Every stream's tracks are kept so a
#: row can fire the browser's own ``ended`` (the "Stop sharing" bar) on them.
INIT_JS = """
(() => {
  const rec = {calls: 0, streams: []};
  Object.defineProperty(window, '__t0177', {value: rec, configurable: true});
  const fake = () => {
    rec.calls += 1;
    const c = document.createElement('canvas');
    c.width = 640; c.height = 400;
    const g = c.getContext('2d');
    let n = 0;
    const paint = () => {
      g.fillStyle = 'rgb(255,0,0)'; g.fillRect(0, 0, 640, 400);
      g.fillStyle = 'rgb(90,0,0)'; g.fillRect((n * 23) % 600, 0, 40, 20);
      n += 1;
    };
    paint();
    setInterval(paint, 100);
    const stream = c.captureStream(5);
    rec.streams.push(stream);
    // C-R2: `rec.mode` 'reject' makes the picker fail (a cancelled or refused picker); 'hold' keeps the
    // picker pending until the row calls rec.release() (a stop that lands while a capture is starting).
    if (rec.mode === 'reject') {
      for (const t of stream.getTracks()) t.stop();
      return Promise.reject(new DOMException('Permission denied (T-0177 C-R2 row)', 'NotAllowedError'));
    }
    if (rec.mode === 'hold') return new Promise((ok) => { rec.release = () => { ok(stream); return true; }; });
    return Promise.resolve(stream);
  };
  rec.states = () => {
    const s = rec.streams[rec.streams.length - 1];
    return s ? s.getTracks().map((t) => t.readyState) : null;
  };
  rec.end = () => {
    const s = rec.streams[rec.streams.length - 1];
    if (!s) return false;
    for (const t of s.getVideoTracks()) {
      const ev = new Event('ended');
      t.dispatchEvent(ev);
    }
    return true;
  };
  if (window.MediaDevices) {
    MediaDevices.prototype.getDisplayMedia = function () { return fake(); };
    MediaDevices.prototype.getUserMedia = function () { return fake(); };
  }
})();
"""

#: Phase U only: every ``video.play()`` resolves PLAY_DELAY_MS late, so the ``?source=canvas`` capture
#: that main.js starts at load comes up after the session's state has arrived.
PLAY_DELAY_MS = 5000
HOLD_PLAY_JS = """
(() => {
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    const r = play.call(this);
    return new Promise((ok, bad) => setTimeout(() => Promise.resolve(r).then(ok, bad), __D__));
  };
})();
""".replace("__D__", str(PLAY_DELAY_MS))

PROBE_JS = """
(() => {
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  const pane = document.querySelector('main > .pane.thumb');
  const sel = pane ? pane.querySelector('select[data-vision-source]') : null;
  const labelText = sel ? ([...(sel.labels || [])].map((l) => l.textContent).join(' ') + ' '
                           + (sel.getAttribute('aria-label') || '')) : '';
  const prev = document.getElementById('preview');
  const colour = (im) => {
    if (!vis(im) || !im.complete || !im.naturalWidth) return null;
    const c = document.createElement('canvas');
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(c.width >> 1, c.height >> 1, 1, 1).data;
    return [d[0], d[1], d[2]];
  };
  const thumbs = pane ? [...pane.querySelectorAll('img[data-vision-thumb]')].filter(vis) : [];
  const cap = pane ? pane.querySelector('[data-capture-caption]') : null;
  const live = pane ? pane.querySelector('.live') : null;
  const mute = document.getElementById('mute-button');
  return {
    combo: !!sel, combo_visible: vis(sel), value: sel ? sel.value : null,
    combo_label: labelText.replace(/\\s+/g, ' ').trim(),
    options: sel ? [...sel.options].map((o) => ({value: o.value, text: o.textContent.replace(/\\s+/g, ' ').trim(),
                                                 disabled: o.disabled})) : [],
    preview_visible: vis(prev), preview_rgb: colour(prev),
    preview_natural: prev ? [prev.naturalWidth, prev.naturalHeight] : null,
    preview_fit: prev ? getComputedStyle(prev).objectFit : null,
    preview_src_attr: prev ? prev.getAttribute('src') : null,
    preview_broken: !!prev && vis(prev) && prev.hasAttribute('src') && prev.getAttribute('src') !== ''
                    && prev.complete && prev.naturalWidth === 0,
    thumb_srcs: pane ? [...pane.querySelectorAll('img[data-vision-thumb]')].map((t) => t.getAttribute('src')) : [],
    errors: [...document.querySelectorAll('#error-strip .error-line')].filter(vis)
              .map((e) => e.dataset.errorCode || ''),
    share_text: (document.getElementById('share-button') || {}).textContent || '',
    thumbs: thumbs.map((t) => colour(t)),
    empties: pane ? [...pane.querySelectorAll('[data-empty]')].filter(vis)
                      .map((e) => e.textContent.replace(/\\s+/g, ' ').trim()) : [],
    caption: cap ? cap.textContent.replace(/\\s+/g, ' ').trim() : null,
    badge: vis(live),
    vision_live_attr: !!pane && pane.hasAttribute('data-vision-live'),
    capturing: !!mute && !mute.disabled,
    frames_sent: Number((document.getElementById('frames-sent') || {dataset: {}}).dataset.framesSent || 0),
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""

COMBO = "main > .pane.thumb select[data-vision-source]"

CHANGE_JS = (
    "(() => { const s = document.querySelector(" + json.dumps(COMBO) + "); if (!s) return false;"
    " s.value = __V__; s.dispatchEvent(new Event('input', {bubbles: true}));"
    " s.dispatchEvent(new Event('change', {bubbles: true})); return true; })()"
)

CAPTURING_JS = (
    "!document.getElementById('mute-button').disabled && "
    "/^\\d+x\\d+$/.test((document.getElementById('capture-size').textContent || '').trim())"
)
STOPPED_JS = "document.getElementById('mute-button').disabled"
#: T-0181 B3's `m` rows: a window-level capturing keydown listener (it runs before any page handler, so a
#: handler that stops propagation cannot hide the key) records every key that reached the page; it proves
#: the `m` was delivered, so "`m` does nothing" is not vacuous.
KEYS_SEEN_JS = (
    "(() => { window.__t0181_keys = []; if (!window.__t0181_keys_on) { window.__t0181_keys_on = true;"
    " window.addEventListener('keydown', (e) => window.__t0181_keys.push(e.key), true); } return true; })()"
)

TOO_LARGE_EMPTY = "The image sent was too large to show here"

#: C-R3: how many astra.turn.state and how many denoise astra.canvas.step messages arrive while the
#: two-image frame is shown (each, so 2 * R3_N + 1 non-vision messages in all).
R3_N = 10
_R3_LEN = 8


def _turn_state(turn_id: str, state: str) -> dict[str, Any]:
    return {"type": "astra.turn.state", "state": state, "turn_id": turn_id, "thinking_ms": 0}


def _denoise_step(i: int) -> dict[str, Any]:
    """One denoise ``astra.canvas.step`` (the wire shape of contracts/protocol.py AstraCanvasStep)."""
    return {
        "type": "astra.canvas.step",
        "request_id": "req-r3",
        "canvas_idx": 0,
        "step": i + 1,
        "max_steps": R3_N + 2,
        "canvas_length": _R3_LEN,
        "canvas": [100 + i + k for k in range(_R3_LEN)],
        "argmax": [200 + i + k for k in range(_R3_LEN)],
        "stable": "ff",
        "phase": "denoise",
    }


#: C-R3: remember the pane's thumbnail <img> nodes, #preview and the combo's <option> nodes, and count every
#: mutation under the pane (child lists, attributes, text) from now on.
R3_WATCH_JS = """
(() => {
  const pane = document.querySelector('main > .pane.thumb');
  const sel = pane.querySelector('select[data-vision-source]');
  const rec = {
    records: [],
    thumbs: [...pane.querySelectorAll('img[data-vision-thumb]')],
    preview: document.getElementById('preview'),
    options: sel ? [...sel.options] : [],
    optionText: sel ? [...sel.options].map((o) => o.textContent) : [],
  };
  const name = (n) => n.nodeType === 3 ? '#text<' + (n.parentNode ? (n.parentNode.className || n.parentNode.nodeName) : '?') + '>'
                    : (n.id ? '#' + n.id : (n.className || n.nodeName));
  rec.note = (list) => { for (const m of list) rec.records.push(m.type + ':' + name(m.target) + (m.attributeName ? '@' + m.attributeName : '')); };
  rec.obs = new MutationObserver(rec.note);
  rec.obs.observe(pane, {subtree: true, childList: true, attributes: true, characterData: true});
  Object.defineProperty(window, '__t0177r3', {value: rec, configurable: true});
  return rec.thumbs.length;
})()
"""

R3_READ_JS = """
(() => {
  const rec = window.__t0177r3;
  if (!rec) return {installed: false};
  rec.note(rec.obs.takeRecords());
  const pane = document.querySelector('main > .pane.thumb');
  const sel = pane.querySelector('select[data-vision-source]');
  const thumbs = [...pane.querySelectorAll('img[data-vision-thumb]')];
  const options = sel ? [...sel.options] : [];
  return {
    installed: true,
    thumbs_before: rec.thumbs.length,
    thumbs_now: thumbs.length,
    same_thumb_nodes: thumbs.length === rec.thumbs.length && thumbs.every((t, i) => t === rec.thumbs[i])
                      && rec.thumbs.every((t) => t.isConnected),
    same_preview_node: document.getElementById('preview') === rec.preview,
    same_option_nodes: options.length === rec.options.length && options.every((o, i) => o === rec.options[i]),
    same_option_text: options.map((o) => o.textContent).join('|') === rec.optionText.join('|'),
    mutations: rec.records.length,
    mutation_kinds: [...new Set(rec.records)].slice(0, 12),
  };
})()
"""

#: Premise of C-R3: the non-vision messages reached the page (the chip shows the last turn state, the
#: denoising view labelled a frame).
R3_OTHERS_PREMISE_JS = """
(() => ({
  chip: (document.getElementById('turn-chip') || {}).textContent || '',
  denoise: ((document.querySelector('[data-denoise-label]') || {}).textContent || '').trim(),
}))()
"""

#: The contrast rows of C4's text, both themes (T-0154 A7 moved here from test_hud_chrome.py: that
#: module's page has no session, so it can no longer show a caption).
CONTRAST = [
    ("vision caption", "main > .pane.thumb [data-capture-caption]", None),
    ("vision combo", COMBO, None),
]


@dataclass
class Run:
    error: str | None = None
    probes: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, list[str]] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    contrast: dict[str, Any] = field(default_factory=dict)
    url_types: list[str] = field(default_factory=list)
    all_types: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _click(chrome: Any, selector: str) -> bool:
    """A real mouse click (Input.dispatchMouseEvent), not a synthesised DOM event."""
    got = chrome.evaluate(
        "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
        " if (!e) return null; const r = e.getBoundingClientRect();"
        " return [r.left + r.width / 2, r.top + r.height / 2]; })()"
    )
    if not isinstance(got, list):
        return False
    x, y = float(got[0]), float(got[1])
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {"type": kind, "x": x, "y": y}
        if kind != "mouseMoved":
            params.update(button="left", clickCount=1)
        chrome._cmd("Input.dispatchMouseEvent", **params)
    return True


def _mounted(chrome: Any, timeout_s: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and _panels_line(chrome.console[chrome._mark :]) is None:
        chrome.drain(0.1)
    return _panels_line(chrome.console[chrome._mark :]) is not None


def _drive(result: Run, chrome: Any, stub: HudStub, port: int) -> None:
    ws = urllib.parse.quote(stub.url, safe="")
    base = f"http://127.0.0.1:{port}/"
    probes = result.probes

    def selects() -> list[str]:
        return [str(getattr(m.source, "value", m.source)) for m in stub.vision_selects()]

    def probe(name: str) -> dict[str, Any]:
        chrome.evaluate(
            "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )
        got = chrome.evaluate(PROBE_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}
        return probes[name]

    def step(name: str, action: Any, *, expect: int, settle: float = 0.8) -> None:
        """Run ``action`` and record the selects it caused: wait for ``expect`` new ones, then ``settle``
        seconds more so an extra one (a second select for one Share) is seen too."""
        before = len(selects())
        action()
        if expect:
            stub.wait_for(lambda s: len(selects()) >= before + expect, 6)
        chrome.drain(settle)
        result.deltas[name] = selects()[before:]

    def push(msg: Any) -> None:
        stub.push(msg)
        chrome.drain(0.2)

    def share() -> None:
        _click(chrome, "#share-button")
        chrome.wait_until(CAPTURING_JS, 10)

    def mute() -> None:
        _click(chrome, "#mute-button")
        chrome.wait_until(STOPPED_JS, 5)

    def wait_preview(colour: str, timeout_s: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _is(chrome.evaluate(PROBE_JS).get("preview_rgb"), colour):
                return True
            time.sleep(0.1)
        return False

    def wait_shown_none(timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if chrome.evaluate(PROBE_JS).get("preview_rgb") is None:
                return True
            time.sleep(0.1)
        return False

    blue = vision_image(
        _jpeg(BLUE, 320, 200), identity="hud:blue", kind="screen", width=320, height=200
    )

    # --- phase S: a page with no ?source (Share goes through the page's real screen provider) ----------
    chrome.add_init_script(INIT_JS)
    chrome._cmd(
        "Emulation.setDeviceMetricsOverride",
        width=1440,
        height=900,
        deviceScaleFactor=1,
        mobile=False,
    )
    hellos = stub.received_types().count("astra.hello")
    chrome.navigate(f"{base}?ws={ws}")
    if not _mounted(chrome):
        raise RuntimeError("the page never logged 'panels mounted:'")
    if not stub.wait_for(lambda s: s.received_types().count("astra.hello") > hellos, 15):
        raise RuntimeError(f"no astra.hello; the stub saw {stub.received_types()!r}")
    stub.wait_for_client(10)
    probe("idle")

    # S1: a Share before any state (not live): nothing is selected, and the local capture is not shown.
    step("share_before_state", share, expect=0, settle=1.5)
    probe("share_before_state")
    step("mute_before_state", mute, expect=0, settle=0.5)

    # S2: the first state. The combo appears with the echoed value; shared is disabled (no capture).
    push(vision_state("browser"))
    chrome.wait_until(f"!!document.querySelector({json.dumps(COMBO)})", 5)
    probe("state_browser_idle")

    # S3: a Share click in live: exactly one select shared; the combo stays on the echo.
    step("share_live", share, expect=1)
    probe("after_share_click")

    # S4: the echo moves the combo.
    push(vision_state("shared"))
    chrome.wait_until(
        f"(document.querySelector({json.dumps(COMBO)}) || {{}}).value === 'shared'", 5
    )
    probe("echo_shared")

    # S5: the user picks browser in the combo: one select browser; the value is the last echo again.
    step(
        "combo_change",
        lambda: chrome.evaluate(CHANGE_JS.replace("__V__", json.dumps("browser"))),
        expect=1,
        settle=1.0,
    )
    probe("after_combo_change")

    # S6 (adversarial): the local capture paints red; the session's frame is blue. Ten capture ticks.
    push(vision_frame("t1", 1, source="shared", images=[blue]))
    wait_preview("blue")
    before_sent = int(chrome.evaluate(PROBE_JS).get("frames_sent") or 0)
    for _ in range(10):
        time.sleep(0.3)
        got = chrome.evaluate(PROBE_JS)
        result.samples.append(
            {k: got.get(k) for k in ("preview_visible", "preview_rgb", "frames_sent", "capturing")}
            if isinstance(got, dict)
            else {"__probe__": got}
        )
    probes["ticks"] = {
        "frames_before": before_sent,
        "frames_after": int(chrome.evaluate(PROBE_JS).get("frames_sent") or 0),
    }
    for theme in ("dark", "light"):
        chrome.evaluate(choose_theme_js(theme))
        time.sleep(0.3)
        probe(f"blue_{theme}")
        result.contrast[theme] = chrome.evaluate(measure_js(CONTRAST)) or {}
    chrome.evaluate(choose_theme_js("dark"))

    # S7: a frame with two images (an overview, then the browser frame) and history.
    overview = vision_image(
        _jpeg(GREEN, 200, 400), identity="ov:1", kind="overview", width=200, height=400
    )
    window = vision_image(
        _jpeg(YELLOW, 400, 250), identity="capture:2", kind="window", width=400, height=250
    )
    push(vision_frame("t1", 2, source="browser", images=[overview, window], history_images=2))
    wait_preview("yellow")
    probe("two_images")

    # C-R3: with the two-image frame shown, other messages (turn state, denoise steps) leave the pane alone:
    # the same <img> nodes, and no mutation at all under the pane (the combo's options included).
    chrome.evaluate(R3_WATCH_JS)
    r3_before = chrome.evaluate(R3_OTHERS_PREMISE_JS)
    for i in range(R3_N):
        stub.push(json.dumps(_turn_state("t1", "THINKING" if i % 2 == 0 else "LISTENING")))
        stub.push(json.dumps(_denoise_step(i)))
    stub.push(json.dumps(_turn_state("t1", "SPEAKING")))
    chrome.drain(1.5)
    probes["r3_others"] = {
        **(chrome.evaluate(R3_READ_JS) or {}),
        "premise_before": r3_before,
        "premise_after": chrome.evaluate(R3_OTHERS_PREMISE_JS),
    }
    # …while an astra.vision.frame still re-renders it.
    swapped = [
        vision_image(
            _jpeg(BLUE, 300, 200), identity="capture:3", kind="window", width=300, height=200
        ),
        vision_image(
            _jpeg(GREEN, 400, 250), identity="capture:4", kind="window", width=400, height=250
        ),
    ]
    push(vision_frame("t1", 3, source="browser", images=swapped))
    wait_preview("green")
    probes["r3_frame"] = {**(chrome.evaluate(R3_READ_JS) or {}), **probe("r3_frame_probe")}
    chrome.evaluate("window.__t0177r3 && window.__t0177r3.obs.disconnect()")

    # C-R1: a frame whose newest image the server omitted (jpeg_b64 null, omitted "too_large").
    push(
        vision_frame(
            "t4",
            1,
            source="browser",
            images=[
                vision_image(
                    _jpeg(GREEN, 200, 120),
                    identity="capture:5",
                    kind="window",
                    width=200,
                    height=120,
                ),
                vision_image(None, identity="capture:6", kind="window", width=640, height=480),
            ],
        )
    )
    wait_shown_none()
    chrome.drain(0.3)
    probe("too_large")

    # S8: a blind frame.
    push(vision_frame("t2", 1, source="shared", blind=True, images=[]))
    wait_shown_none()
    probe("blind")

    # S9: Mute while shared is echoed selects browser.
    step("mute_shared", mute, expect=1)
    # S10 (re-scoped by T-0181 B3, H-T181-4: the `m` shortcut is removed): `m` on the body while a Share
    # runs and shared is echoed does nothing: no select, the capture keeps running. A capturing keydown
    # listener proves the key reached the page. Then the capture is stopped by the button (only if `m` did
    # not stop it, as BASE's `m` does) so S11 starts from the same stopped state either way.
    step("share_before_m", share, expect=1)
    chrome.evaluate(KEYS_SEEN_JS)
    chrome.evaluate("document.activeElement && document.activeElement.blur(); 1")
    frames_before_m = len(stub.frames())
    step("key_m", lambda: chrome.key("m", text="m", code=77), expect=0, settle=1.2)
    stub.wait_for(lambda s: len(s.frames()) >= frames_before_m + 2, 3)
    probes["key_m"] = {
        **probe("key_m_probe"),
        "keys_seen": chrome.evaluate("window.__t0181_keys || []"),
        "frames_after": len(stub.frames()) - frames_before_m,
    }
    if chrome.evaluate(f"!({STOPPED_JS})"):
        step("stop_after_m", mute, expect=1)
    # S11: the reset's stop selects browser and clears the panel; the choice (combo) survives.
    step("share_before_reset", share, expect=1)
    push(vision_frame("t3", 1, source="shared", images=[blue]))
    wait_preview("blue")
    probe("before_reset")
    step(
        "reset",
        lambda: (
            push(
                '{"type":"astra.session.cleared","resets":1,"ended_turn_id":null,"memory_forgotten":0}'
            ),
            chrome.wait_until(STOPPED_JS, 5),
        ),
        expect=1,
    )
    probe("after_reset")
    # S12: the track's `ended` (the browser's own Stop sharing) selects browser.
    step("share_before_ended", share, expect=1)
    step("track_ended", lambda: chrome.evaluate("window.__t0177.end()"), expect=1)

    # C-R2 (a): a Share while a capture runs and shared is echoed (the stub never echoes, so the echo is
    # still S4's shared): exactly one select shared, no browser in between.
    step("share_before_reshare", share, expect=1)
    calls0 = int(chrome.evaluate("window.__t0177.calls") or 0)
    step("reshare", lambda: _click(chrome, "#share-button"), expect=1, settle=1.2)
    probes["reshare"] = {
        **probe("reshare_probe"),
        "picker_calls": int(chrome.evaluate("window.__t0177.calls") or 0) - calls0,
    }

    # C-R2 (b): a re-Share whose picker fails: the old capture is gone, so exactly select browser, and the
    # existing error line.
    chrome.evaluate("window.__t0177.mode = 'reject'")
    calls0 = int(chrome.evaluate("window.__t0177.calls") or 0)
    step(
        "reshare_rejected",
        lambda: (_click(chrome, "#share-button"), chrome.wait_until(STOPPED_JS, 5)),
        expect=1,
        settle=1.2,
    )
    probes["reshare_rejected"] = {
        **probe("reshare_rejected_probe"),
        "picker_calls": int(chrome.evaluate("window.__t0177.calls") or 0) - calls0,
    }
    chrome.evaluate("window.__t0177.mode = null")

    # C-R2 (c): a stop (a reset) that lands while the picker is still pending. The echo is browser first, so
    # the stop itself has nothing to select (C2) and every select seen is for the new stream. T-0181 B3
    # (H-T181-4) re-scopes the `m` half: `m` while the picker is pending is no stop at all (no shortcut), so
    # after the release the capture runs and one select shared is sent. Stop sharing is disabled while the
    # picker is pending (no capture yet), so no button stop can replace it; the row records that.
    push(vision_state("browser"))
    for name, stop in (
        ("pending_m", lambda: (chrome.evaluate(KEYS_SEEN_JS), chrome.key("m", text="m", code=77))),
        (
            "pending_reset",
            lambda: push(
                '{"type":"astra.session.cleared","resets":2,"ended_turn_id":null,"memory_forgotten":0}'
            ),
        ),
    ):
        chrome.evaluate("window.__t0177.mode = 'hold'")
        calls0 = int(chrome.evaluate("window.__t0177.calls") or 0)
        sel0 = len(selects())
        _click(chrome, "#share-button")
        chrome.wait_until(f"window.__t0177.calls > {calls0}", 5)
        stop_disabled_while_pending = bool(chrome.evaluate(STOPPED_JS))
        stop()
        chrome.drain(0.5)
        keys_seen = chrome.evaluate("window.__t0181_keys || []") if name == "pending_m" else None
        types0 = len(stub.received_types())
        released = chrome.evaluate("window.__t0177.release()")
        chrome.evaluate("window.__t0177.mode = null")
        chrome.drain(1.5)
        probes[name] = {
            **probe(f"{name}_probe"),
            "picker_calls": int(chrome.evaluate("window.__t0177.calls") or 0) - calls0,
            "released": released,
            "tracks": chrome.evaluate("window.__t0177.states()"),
            "sent_after_release": stub.received_types()[types0:],
            "stop_disabled_while_pending": stop_disabled_while_pending,
            "keys_seen": keys_seen,
        }
        result.deltas[name] = selects()[sel0:]
        if name == "pending_m" and chrome.evaluate(f"!({STOPPED_JS})"):
            # B: the `m` stopped nothing, so the released capture runs; stop it by the button (browser is
            # echoed, so no select) so the reset half starts from BASE's stopped state.
            mute()
            chrome.drain(0.5)

    # C-R3 (a capture stop re-renders the pane): browser is echoed (C-R2 (c) above), a Share starts a
    # capture and a shared frame is shown with the Live badge; then Mute. With browser echoed the stop
    # sends no select and the stub echoes nothing, so no astra.vision.* message arrives after the stop:
    # only the stop itself can disable the shared option and drop data-vision-live.
    step("share_before_quiet_mute", share, expect=1)
    push(vision_frame("t5", 1, source="shared", images=[blue]))
    wait_preview("blue")
    probe("before_quiet_mute")
    types0 = len(stub.received_types())
    step("quiet_mute", mute, expect=0, settle=0.8)
    probes["quiet_mute"] = {
        **probe("quiet_mute_probe"),
        "sent_after_stop": stub.received_types()[types0:],
    }

    # --- phase U: ?source=canvas, the existing live rigs' shape ---------------------------------------
    chrome.drain(0.2)
    url_from = len(stub.received_types())
    sel_from = len(selects())
    hellos = stub.received_types().count("astra.hello")
    chrome.add_init_script(HOLD_PLAY_JS)
    chrome.navigate(f"{base}?source=canvas&ws={ws}")
    if not _mounted(chrome):
        raise RuntimeError("phase U: the page never logged 'panels mounted:'")
    stub.wait_for(lambda s: s.received_types().count("astra.hello") > hellos, 15)
    stub.wait_for_client(10)
    # The state arrives while the URL capture is still starting (its video.play() is held for
    # PLAY_DELAY_MS), so the page is live when that capture comes up: a page that treated a URL start as a
    # Share would select shared here. Without the hold, the capture is up long before the socket opens and
    # the row could not tell the two apart.
    push(vision_state("browser"))
    chrome.wait_until(f"!!document.querySelector({json.dumps(COMBO)})", 5)
    probes["url_live_before_capture"] = {"capturing": bool(chrome.evaluate(CAPTURING_JS))}
    chrome.wait_until(CAPTURING_JS, PLAY_DELAY_MS / 1000 + 5)
    frames0 = len(stub.frames())
    stub.wait_for(lambda s: len(s.frames()) >= frames0 + 2, 6)
    chrome.drain(1.0)
    probe("url_state")
    push(
        vision_frame(
            "t1",
            1,
            source="browser",
            images=[
                vision_image(
                    _jpeg(BLUE, 320, 200),
                    identity="capture:1",
                    kind="window",
                    width=320,
                    height=200,
                )
            ],
        )
    )
    wait_preview("blue")
    chrome.drain(0.5)
    probe("url_frame")
    result.deltas["url_load"] = selects()[sel_from:]
    result.url_types = stub.received_types()[url_from:]
    result.all_types = stub.received_types()


@pytest.fixture(scope="module")
def run() -> Any:
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    started = time.monotonic()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        with _Recorder(CHROME) as chrome:
            chrome._cmd("Runtime.enable")
            chrome._cmd("Network.enable")
            try:
                _drive(result, chrome, stub, server.port)
            finally:
                chrome.drain(0.3)
                result.requests = list(chrome.requests)
                result.exceptions = list(chrome.exceptions)
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
        stub.stop()
    result.seconds = time.monotonic() - started
    print(f"[T-0177 C] headless session wall time {result.seconds:.1f} s")
    yield result


def _p(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, f"the T-0177 C headless session failed: {run.error}"
    got = run.probes.get(name)
    assert isinstance(got, dict) and "__probe__" not in got, (
        f"probe {name!r} missing or broken: {got!r}"
    )
    return got


def _norm(s: Any) -> str:
    return " ".join(str(s or "").replace("’", "'").replace("‘", "'").split())


# --- C1 / C5: before a state ----------------------------------------------------------------------------


@requires_chrome
def test_c1_no_combo_and_no_select_before_a_state(run):
    """C1: the combo is rendered only after an ``astra.vision.state`` arrives (never outside live); C2/C5: a
    Share and a Mute before any state send no select."""
    for name in ("idle", "share_before_state"):
        p = _p(run, name)
        assert not p["combo_visible"], (
            f"C1: a combo is visible before any astra.vision.state ({name}): {p['options']}"
        )
    assert _p(run, "share_before_state")["capturing"], (
        "premise: the Share started the (stubbed) capture"
    )
    assert run.deltas.get("share_before_state") == [], (
        f"C5: a select before any state: {run.deltas.get('share_before_state')}"
    )
    assert run.deltas.get("mute_before_state") == [], (
        f"C5: a select before any state: {run.deltas.get('mute_before_state')}"
    )


@requires_chrome
@pytest.mark.parametrize(
    ("name", "empty"),
    [("share_before_state", NONLIVE_EMPTY), ("url_state", LIVE_EMPTY)],
    ids=["share-not-live", "url-live"],
)
def test_c3_the_local_capture_is_never_shown_before_a_frame(run, name, empty):
    """C3: ``encode`` no longer touches ``#preview``: a running capture with no ``astra.vision.frame`` shows no
    image. The pane says T-0154's existing empty text outside live (a Share before any state), and
    "Nothing sent yet: Gearvis looks when you ask" in live (the URL page after its state)."""
    p = _p(run, name)
    assert p["capturing"], f"premise ({name}): a capture runs"
    assert p["preview_rgb"] is None, (
        f"C3 ({name}): #preview shows the local capture ({p['preview_rgb']})"
    )
    assert p["empties"] == [empty], (
        f"C3 ({name}): the empty text is {p['empties']}, not [{empty!r}]"
    )
    assert p["caption"] == "", f"C4 ({name}): no frame, empty caption; got {p['caption']!r}"


# --- C1: the combo follows the echo ------------------------------------------------------------------------


@requires_chrome
def test_c1_the_combo_appears_on_the_state_labelled_with_its_options(run):
    p = _p(run, "state_browser_idle")
    assert p["combo_visible"], (
        "C1: the combo is shown once a state arrives (in the What I see pane)"
    )
    assert "Looking at" in p["combo_label"], (
        f"C1: labelled 'Looking at'; label text {p['combo_label']!r}"
    )
    assert p["value"] == "browser", f"C1: its value is the echo: {p['value']!r}"
    got = [(o["value"], _norm(o["text"])) for o in p["options"]]
    assert got == [("browser", "Gearvis's browser"), ("shared", "Shared screen")], (
        f"C1: the options: {got}"
    )
    shared = next(o for o in p["options"] if o["value"] == "shared")
    assert shared["disabled"], "C1: shared is disabled with no capture running (Mute was pressed)"
    assert p["empties"] == [LIVE_EMPTY], (
        f"C3: live with no frame says {LIVE_EMPTY!r}: {p['empties']}"
    )


@requires_chrome
def test_c2_a_share_click_in_live_sends_select_shared_once(run):
    assert run.deltas.get("share_live") == ["shared"], (
        f"C2 (H-T177-1): one Share click in live sends exactly one select shared; got {run.deltas.get('share_live')}"
    )
    p = _p(run, "after_share_click")
    assert p["value"] == "browser", (
        f"C1: no echo yet, so the combo still reads the last echo 'browser': {p['value']!r}"
    )
    shared = next((o for o in p["options"] if o["value"] == "shared"), None)
    assert shared and not shared["disabled"], (
        f"C1: shared is enabled while a capture runs: {p['options']}"
    )


@requires_chrome
def test_c1_the_combo_follows_the_echo_not_the_click(run):
    assert _p(run, "echo_shared")["value"] == "shared", "C1: the echo moves the combo"
    assert run.deltas.get("combo_change") == ["browser"], (
        f"C1: changing the combo sends astra.vision.select browser; got {run.deltas.get('combo_change')}"
    )
    after = _p(run, "after_combo_change")
    assert after["value"] == "shared", (
        f"C1: its value is always the last echo; with no echo after the change it reads 'shared', not {after['value']!r}"
    )


# --- C3: the panel shows the session's frame ---------------------------------------------------------------


@requires_chrome
def test_c3_adversarial_the_preview_stays_blue_across_ten_capture_ticks(run):
    """The local capture paints red; the stub pushed one blue ``astra.vision.frame``. Ten samples 300 ms apart
    (the page's capture tick is 250 ms) all show blue, while the capture kept encoding (frames sent grew)."""
    assert run.error is None, run.error
    assert len(run.samples) == 10, f"premise: ten samples, got {len(run.samples)}"
    ticks = run.probes.get("ticks") or {}
    assert ticks.get("frames_after", 0) >= ticks.get("frames_before", 0) + 2, (
        f"premise: the local capture kept sending during the window (frames sent {ticks}); otherwise a page that "
        "still fed #preview from encode would not have had the chance to overwrite it"
    )
    bad = [
        (i, s)
        for i, s in enumerate(run.samples)
        if not (s.get("preview_visible") and _is(s.get("preview_rgb"), "blue"))
    ]
    assert bad == [], (
        f"C3: #preview left the session's blue frame (red = the local capture) at samples {bad}"
    )


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_c4_the_caption_of_a_shared_frame_and_the_live_badge(run, theme):
    p = _p(run, f"blue_{theme}")
    assert p["theme"] == theme, f"premise: data-theme reads {p['theme']!r}"
    want = f"Shared screen{SEP}320x200{SEP}turn t1 step 1"
    assert _norm(p["caption"]) == want, f"C4: the caption is {p['caption']!r}, not {want!r}"
    assert p["preview_fit"] == "contain", (
        f"T-0154 A6 (kept on the session's frame): object-fit {p['preview_fit']!r}"
    )
    assert p["badge"], (
        "C4: the Live badge shows while the newest image is shared and a capture runs"
    )
    assert p["empties"] == [], f"C3: no empty sentence while an image is shown: {p['empties']}"


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_c4_the_caption_and_the_combo_clear_their_contrast_floor(run, theme):
    """T-0154 A7's floor for the caption (moved here with C4) and the new combo: 4.5:1, 3.0:1 large."""
    assert run.error is None, run.error
    missing, vacuous, failures, _excluded = _contrast_report(run.contrast.get(theme, {}))
    assert missing == [], f"C1/C4 ({theme}): no element matches {missing}"
    assert vacuous == [], f"({theme}) vacuous selectors: {vacuous}"
    assert failures == [], f"T-0154 A7 ({theme}): text below its WCAG 2.1 floor: {failures}"


@requires_chrome
def test_c3_c4_newest_large_earlier_thumbnail_history_and_no_badge_on_a_browser_image(run):
    p = _p(run, "two_images")
    assert _is(p["preview_rgb"], "yellow"), (
        f"C3: the newest (last) image is large: {p['preview_rgb']}"
    )
    assert len(p["thumbs"]) == 1 and _is(p["thumbs"][0], "green"), (
        f"C3: the earlier image is one thumbnail: {p['thumbs']}"
    )
    want = f"Gearvis's browser{SEP}400x250{SEP}turn t1 step 2{SEP}+2 earlier images"
    assert _norm(p["caption"]) == want, f"C4: caption {p['caption']!r}, not {want!r}"
    assert not p["badge"], (
        "C4: a browser image, although a capture runs and shared is chosen: no Live badge"
    )


@requires_chrome
def test_c3_a_blind_frame_hides_the_image(run):
    p = _p(run, "blind")
    assert p["preview_rgb"] is None, f"C3: a blind frame shows no image ({p['preview_rgb']})"
    assert p["thumbs"] == [], f"C3: no thumbnails on a blind frame: {p['thumbs']}"
    assert BLIND_EMPTY in p["empties"], f"C3: the pane says {BLIND_EMPTY!r}: {p['empties']}"
    assert "400x250" not in (p["caption"] or ""), (
        f"C4: the caption still describes the previous image: {p['caption']!r}"
    )
    assert not p["badge"], "C4: no image, no Live badge"


@requires_chrome
def test_c3_a_reset_clears_the_panel_and_the_choice_survives(run):
    before = _p(run, "before_reset")
    assert _is(before["preview_rgb"], "blue"), "premise: the panel showed a frame before the reset"
    p = _p(run, "after_reset")
    assert p["preview_rgb"] is None, f"C3: a reset clears the panel ({p['preview_rgb']})"
    assert p["caption"] == "", f"C4: no frame, empty caption: {p['caption']!r}"
    assert p["empties"] == [LIVE_EMPTY], f"C3: live with no frame: {p['empties']}"
    assert p["value"] == "shared", (
        f"B9/T-0162 D-3: the choice survives the reset; the combo reads {p['value']!r}"
    )


# --- C2: stopping the capture ------------------------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize(
    ("before", "stop"),
    [
        ("echo_shared", "mute_shared"),
        ("share_before_reset", "reset"),
        ("share_before_ended", "track_ended"),
    ],
    ids=["mute", "reset", "track-ended"],
)
def test_c2_stopping_the_capture_while_shared_is_echoed_selects_browser(run, before, stop):
    assert run.error is None, run.error
    if before != "echo_shared":
        assert run.deltas.get(before) == ["shared"], (
            f"premise: the Share before {stop} selected shared: {run.deltas.get(before)}"
        )
    assert run.deltas.get(stop) == ["browser"], (
        f"C2: {stop} with shared echoed sends exactly select browser; got {run.deltas.get(stop)}"
    )


@requires_chrome
def test_s10_m_while_shared_is_echoed_is_no_stop_no_select_and_the_capture_runs(run):
    """S10, re-scoped by T-0181 B3 (H-T181-4, James 2026-10-28: the ``m`` shortcut for Stop sharing is
    removed; "No keydown stops the capture"). C2's stop list loses ``m``: ``m`` on the body while a Share
    runs and shared is echoed sends no select, the capture keeps running (Stop sharing enabled, Share reads
    "Sharing ...", frames keep arriving at the stub), and the key did reach the page."""
    assert run.error is None, run.error
    assert run.deltas.get("share_before_m") == ["shared"], (
        f"premise: the Share before m selected shared: {run.deltas.get('share_before_m')}"
    )
    p = run.probes.get("key_m") or {}
    assert "m" in (p.get("keys_seen") or []), (
        f"premise: the m keydown reached the page (window capture listener): {p.get('keys_seen')}"
    )
    assert run.deltas.get("key_m") == [], (
        f"B3: m is no shortcut, so it selects nothing; got {run.deltas.get('key_m')}"
    )
    assert p.get("capturing") is True and str(p.get("share_text", "")).strip().startswith(
        "Sharing"
    ), (
        f"B3: m stopped the capture (no keydown stops it): capturing={p.get('capturing')}, "
        f"share={p.get('share_text')!r}"
    )
    assert int(p.get("frames_after") or 0) >= 2, (
        f"B3: the capture kept sending after m (frames at the stub grew by {p.get('frames_after')})"
    )


@requires_chrome
def test_c2_adversarial_a_url_capture_never_auto_switches(run):
    """The existing live HUD rigs' shape: ``?source=canvas`` starts a capture at load, the session echoes a state
    before that capture is up (``HOLD_PLAY_JS``), a frame is shown: no select at all, and the page sends only
    hello and frames."""
    held = run.probes.get("url_live_before_capture") or {}
    assert held.get("capturing") is False, (
        f"premise: the state arrived before the URL capture came up (else this row cannot catch a URL start "
        f"treated as a Share): {held}"
    )
    p = _p(run, "url_state")
    assert p["combo_visible"] and p["value"] == "browser", (
        f"premise: the state arrived and the combo shows it: {p}"
    )
    assert p["capturing"], "premise: the URL capture runs"
    assert run.deltas.get("url_load") == [], (
        f"C2: ?source=canvas never auto-switches; sent {run.deltas.get('url_load')}"
    )
    assert set(run.url_types) <= {"astra.hello", "astra.frame"}, (
        f"C2/C5: the URL page sent {sorted(set(run.url_types))}"
    )


@requires_chrome
def test_c3_c4_a_browser_frame_on_a_url_capture_page(run):
    p = _p(run, "url_frame")
    assert _is(p["preview_rgb"], "blue"), f"C3: frames show in every mode: {p['preview_rgb']}"
    assert _norm(p["caption"]) == f"Gearvis's browser{SEP}320x200{SEP}turn t1 step 1", (
        f"C4: {p['caption']!r}"
    )
    assert p["capturing"] and not p["badge"], (
        "C4: a browser image with a capture running shows no Live badge"
    )


# --- Group C review rulings (docs/tasks/T-0177.md "## Group C review rulings") ---------------------------


@requires_chrome
def test_cr1_a_too_large_newest_image_shows_no_broken_picture_and_says_so(run):
    """C-R1: the newest image of the frame has ``jpeg_b64: null`` (``omitted: "too_large"``). ``#preview``
    shows no picture and no broken image (never a ``data:`` URL built from null), no thumbnail has a null
    ``src``, the pane says "The image sent was too large to show here", and the caption describes the image
    that was sent (640x480, not the 200x120 thumbnail nor the previous frame's 400x250)."""
    p = _p(run, "too_large")
    assert p["preview_rgb"] is None, f"C-R1: #preview shows a picture ({p['preview_rgb']})"
    assert not p["preview_broken"], (
        f"C-R1: #preview is a visible broken image; src={str(p['preview_src_attr'])[:80]!r}"
    )
    src = p["preview_src_attr"]
    # The wire may carry the omitted bytes as null or drop the key, so a URL built from either reads
    # "…base64,null", "…base64,undefined" or "…base64," (mutant evidence: "data:image/jpeg;base64,undefined").
    assert src is None or str(src).rstrip().split(",", 1)[-1] not in {"", "null", "undefined"}, (
        f"C-R1: #preview's src is a data: URL built from null: {str(src)[:80]!r}"
    )
    assert all(
        isinstance(s, str) and s.startswith("data:image/jpeg;base64,") and not s.endswith("null")
        for s in p["thumb_srcs"]
    ), f"C-R1: a thumbnail with a null src: {[str(s)[:40] for s in p['thumb_srcs']]}"
    assert len(p["thumbs"]) == 1 and _is(p["thumbs"][0], "green"), (
        f"C-R1: the earlier image (with bytes) is the one visible thumbnail: {p['thumbs']}"
    )
    assert p["empties"] == [TOO_LARGE_EMPTY], (
        f"C-R1: the pane says {TOO_LARGE_EMPTY!r}: {p['empties']}"
    )
    want = f"Gearvis's browser{SEP}640x480{SEP}turn t4 step 1"
    assert _norm(p["caption"]) == want, (
        f"C-R1: the caption still describes the image that was sent: {p['caption']!r}, not {want!r}"
    )


@requires_chrome
def test_cr2a_a_share_while_a_capture_runs_and_shared_is_echoed_selects_shared_only(run):
    """C-R2: the re-Share replaces a running capture; the page sends exactly ``["shared"]`` (no ``browser``
    for the capture it replaced)."""
    assert run.error is None, run.error
    assert run.deltas.get("share_before_reshare") == ["shared"], (
        f"premise: the first Share selected shared: {run.deltas.get('share_before_reshare')}"
    )
    r = run.probes.get("reshare") or {}
    assert r.get("picker_calls") == 1 and r.get("capturing"), (
        f"premise: the re-Share opened the picker once and a capture runs: {r}"
    )
    assert run.deltas.get("reshare") == ["shared"], (
        f"C-R2: one Share, one select: a re-Share while shared is echoed sends exactly ['shared']; "
        f"got {run.deltas.get('reshare')}"
    )


@requires_chrome
def test_cr2b_a_failed_re_share_selects_browser_once_and_shows_the_error_line(run):
    """C-R2: the re-Share's picker rejects: the old capture is gone, so exactly ``["browser"]``, and the
    existing ``capture_failed`` error line."""
    assert run.error is None, run.error
    r = run.probes.get("reshare_rejected") or {}
    assert r.get("picker_calls") == 1, (
        f"premise: the re-Share opened the (rejecting) picker once: {r}"
    )
    assert r.get("capturing") is False, (
        f"C-R2: the old capture is gone after the failed re-Share: {r}"
    )
    assert run.deltas.get("reshare_rejected") == ["browser"], (
        f"C-R2: a re-Share whose picker fails sends exactly ['browser']; got {run.deltas.get('reshare_rejected')}"
    )
    assert "capture_failed" in (r.get("errors") or []), (
        f"C-R2: the existing error line shows: {r.get('errors')}"
    )


@requires_chrome
def test_cr2c_m_while_the_picker_is_pending_does_nothing_and_the_released_capture_runs(run):
    """C-R2 (c), the ``m`` half re-scoped by T-0181 B3 (H-T181-4: no keydown stops the capture). ``m``
    pressed while ``getDisplayMedia`` is pending is no stop: when the picker resolves, the capture runs
    (tracks live, Stop sharing enabled, Share reads "Sharing ...") and exactly one select shared is sent for
    it (browser is echoed). The key did reach the page."""
    assert run.error is None, run.error
    r = run.probes.get("pending_m") or {}
    assert r.get("picker_calls") == 1 and r.get("released") is True, (
        f"premise: the Share opened the held picker once and the row released it: {r}"
    )
    assert "m" in (r.get("keys_seen") or []), (
        f"premise: the m keydown reached the page while the picker was pending: {r.get('keys_seen')}"
    )
    tracks = r.get("tracks")
    assert isinstance(tracks, list) and tracks and all(t == "live" for t in tracks), (
        f"B3: m while the picker was pending ended the new stream (m is no stop); readyState {tracks}"
    )
    assert r.get("capturing") is True and str(r.get("share_text", "")).strip().startswith(
        "Sharing"
    ), (
        f"B3: the released capture is not running: capturing={r.get('capturing')}, "
        f"share={r.get('share_text')!r}"
    )
    assert run.deltas.get("pending_m") == ["shared"], (
        f"B3: after the release exactly one select shared is sent (browser echoed); got "
        f"{run.deltas.get('pending_m')}"
    )
    sent = r.get("sent_after_release") or []
    assert sent.count("astra.vision.select") == 1 and "astra.frame" in sent, (
        f"B3: after the release the page sends one select and its frames: {sent}"
    )


@requires_chrome
def test_cr2c_stop_sharing_is_disabled_while_the_picker_is_pending(run):
    """Why the ``m`` half of C-R2 (c) has no button twin (T-0181 B, Owned files): while the picker is pending
    there is no capture, so Stop sharing is disabled and no button stop can land mid-picker."""
    assert run.error is None, run.error
    for name in ("pending_m", "pending_reset"):
        r = run.probes.get(name) or {}
        assert r.get("stop_disabled_while_pending") is True, (
            f"({name}) Stop sharing was enabled while the picker was pending: {r}"
        )


@requires_chrome
@pytest.mark.parametrize("name", ["pending_reset"], ids=["reset"])
def test_cr2c_a_stop_while_the_picker_is_pending_stops_the_new_stream_and_sends_nothing(run, name):
    """C-R2: a stop (a reset; T-0181 B3 took ``m`` out of the stop list, see the row above) lands while
    ``getDisplayMedia`` is still pending. When the picker then resolves, the page stops that stream (every
    track ``ended``: no orphaned capture, no sharing bar), shows no capture, and sends nothing for it (no
    select, no frame). The echo is ``browser`` throughout, so the stop itself has nothing to select (C2)."""
    assert run.error is None, run.error
    r = run.probes.get(name) or {}
    assert r.get("picker_calls") == 1 and r.get("released") is True, (
        f"premise ({name}): the Share opened the held picker once and the row released it: {r}"
    )
    tracks = r.get("tracks")
    assert isinstance(tracks, list) and tracks, (
        f"premise ({name}): the picker's stream has tracks: {tracks}"
    )
    assert all(t == "ended" for t in tracks), (
        f"C-R2 ({name}): the stream that arrived after the stop is still running (orphaned capture, the "
        f"sharing bar stays up); track readyState {tracks}"
    )
    assert r.get("capturing") is False and not str(r.get("share_text", "")).startswith("Sharing"), (
        f"C-R2 ({name}): the page shows a capture after the stop: capturing={r.get('capturing')}, "
        f"share={r.get('share_text')!r}"
    )
    assert run.deltas.get(name) == [], (
        f"C-R2 ({name}): a select was sent for the stopped stream: {run.deltas.get(name)}"
    )
    sent = [
        t for t in r.get("sent_after_release") or [] if t in {"astra.frame", "astra.vision.select"}
    ]
    assert sent == [], f"C-R2 ({name}): the page sent {sent} for the stopped stream"


@requires_chrome
def test_cr3_other_messages_leave_the_thumbnails_and_the_combo_untouched(run):
    """C-R3: with the two-image frame shown, R3_N ``astra.turn.state`` and R3_N denoise ``astra.canvas.step``
    messages (plus one more turn state) arrive. The thumbnail ``<img>`` elements are the same nodes, the
    combo's options are the same nodes with the same text, and a MutationObserver over the whole pane
    records nothing."""
    assert run.error is None, run.error
    r = run.probes.get("r3_others") or {}
    assert r.get("installed"), f"premise: the observer was installed: {r}"
    assert r.get("thumbs_before") == 1, (
        f"premise: one thumbnail was shown when the watch began: {r}"
    )
    after = r.get("premise_after") or {}
    assert "SPEAKING" in str(after.get("chip", "")).upper(), (
        f"premise: the turn states reached the page (the chip reads the last one): {after}"
    )
    assert after.get("denoise"), f"premise: the denoise steps reached the page (label set): {after}"
    assert r.get("same_thumb_nodes"), f"C-R3: the thumbnails' <img> elements were replaced: {r}"
    assert r.get("same_preview_node"), f"C-R3: #preview was replaced: {r}"
    assert r.get("same_option_nodes") and r.get("same_option_text"), (
        f"C-R3: the combo's options were replaced or rewritten: {r}"
    )
    assert r.get("mutations") == 0, (
        f"C-R3: the pane re-rendered on non-vision messages: {r.get('mutations')} mutation records, "
        f"e.g. {r.get('mutation_kinds')}"
    )


@requires_chrome
def test_cr3_an_astra_vision_frame_still_re_renders_the_pane(run):
    """C-R3's other side (a guard too wide would freeze the pane): a new two-image frame replaces the large
    image and the thumbnail."""
    assert run.error is None, run.error
    r = run.probes.get("r3_frame") or {}
    assert r.get("installed"), f"premise: the observer was installed: {r}"
    assert _is(r.get("preview_rgb"), "green"), (
        f"C3: the new frame's newest image is large: {r.get('preview_rgb')}"
    )
    assert len(r.get("thumbs") or []) == 1 and _is(r["thumbs"][0], "blue"), (
        f"C3: the new frame's earlier image is the thumbnail: {r.get('thumbs')}"
    )
    assert (r.get("mutations") or 0) > 0, (
        f"the observer saw the frame's re-render (else it is blind): {r}"
    )
    assert _norm(r.get("caption")) == f"Gearvis's browser{SEP}400x250{SEP}turn t1 step 3", (
        f"C4: {r.get('caption')!r}"
    )


def _shared_option(p: dict[str, Any]) -> dict[str, Any] | None:
    return next((o for o in p.get("options") or [] if o.get("value") == "shared"), None)


@requires_chrome
def test_cr3_a_mute_with_no_vision_message_after_it_re_renders_the_pane(run):
    """C-R3: a capture stop re-renders the pane. Browser is echoed, a Share runs and the newest image is
    from the shared screen (Live badge, shared enabled); Mute stops the capture. Nothing is selected
    (browser is echoed, C2) and no ``astra.vision.*`` message arrives afterwards, so only the stop can move
    the pane: the combo's shared option is disabled (no capture, C1) and the pane loses
    ``data-vision-live`` (C4: the badge needs a running capture)."""
    assert run.error is None, run.error
    assert run.deltas.get("share_before_quiet_mute") == ["shared"], (
        f"premise: the Share in live selected shared: {run.deltas.get('share_before_quiet_mute')}"
    )
    before = _p(run, "before_quiet_mute")
    assert before["value"] == "browser", f"premise: browser is echoed: {before['value']!r}"
    assert before["capturing"] and _is(before["preview_rgb"], "blue"), (
        f"premise: a capture runs and the shared frame is shown: {before}"
    )
    shared = _shared_option(before)
    assert shared and not shared["disabled"], (
        f"premise: shared is enabled while the capture runs: {before['options']}"
    )
    assert before["vision_live_attr"] and before["badge"], (
        "premise: the pane carries data-vision-live and shows the Live badge before the stop"
    )
    after = run.probes.get("quiet_mute") or {}
    assert after.get("capturing") is False, f"premise: Mute stopped the capture: {after}"
    assert run.deltas.get("quiet_mute") == [], (
        f"C2: a stop with browser echoed selects nothing: {run.deltas.get('quiet_mute')}"
    )
    assert "astra.vision.select" not in (after.get("sent_after_stop") or []), (
        f"premise: nothing vision-related was sent after the stop: {after.get('sent_after_stop')}"
    )
    shared = _shared_option(after)
    assert shared and shared["disabled"], (
        f"C-R3/C1: after the stop the combo's shared option is still enabled (the stop did not re-render "
        f"the pane): {after.get('options')}"
    )
    assert not after.get("vision_live_attr"), (
        "C-R3/C4: after the stop the pane still carries data-vision-live (the stop did not re-render it)"
    )
    assert not after.get("badge"), "C4: no capture, no Live badge"


@requires_chrome
def test_cr3_a_reset_s_stop_re_renders_the_combo(run):
    """C-R3: a reset's stop re-renders the pane too. The reset message itself is handled while the capture
    still runs (it reaches the pane before the stop, C2), so only the stop's render disables shared."""
    before = _p(run, "before_reset")
    shared = _shared_option(before)
    assert before["capturing"] and shared and not shared["disabled"], (
        f"premise: a capture ran and shared was enabled before the reset: {before.get('options')}"
    )
    p = _p(run, "after_reset")
    assert not p["capturing"], "premise: the reset stopped the capture"
    shared = _shared_option(p)
    assert shared and shared["disabled"], (
        f"C-R3/C1: after the reset's stop the combo's shared option is still enabled: {p['options']}"
    )
    assert not p["vision_live_attr"], (
        "C-R3/C4: after the reset's stop the pane carries data-vision-live"
    )


# --- C5: nothing else ----------------------------------------------------------------------------------------


@requires_chrome
def test_c5_the_only_new_message_is_astra_vision_select(run):
    assert run.error is None, run.error
    assert set(run.all_types) <= {"astra.hello", "astra.frame", "astra.vision.select"}, (
        f"C5: the page sent {sorted(set(run.all_types))}"
    )
    assert run.exceptions == [], f"the page threw: {run.exceptions[:3]}"


@requires_chrome
def test_c5_no_request_leaves_loopback(run):
    assert run.error is None, run.error
    assert run.requests, "the CDP network log is empty, so this row would be vacuous"
    off = [
        u
        for u in run.requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert off == [], f"C5: zero non-loopback requests; got {off}"


@requires_chrome
def test_the_session_fits_the_module_budget(run):
    """The module's wall time stays under 60 s (the task's per-module budget for the browser layer)."""
    assert run.error is None, run.error
    assert run.seconds < 60.0, f"the T-0177 C headless session took {run.seconds:.1f} s"


def test_the_jpeg_oracle_reads_its_own_colours():
    """Plain (no browser): the colour classifier the rows use, on Pillow's own decode of the fixtures."""
    from PIL import Image

    for rgb, name in ((BLUE, "blue"), (GREEN, "green"), (YELLOW, "yellow"), ((255, 0, 0), "red")):
        px = Image.open(io.BytesIO(_jpeg(rgb, 64, 48))).convert("RGB").getpixel((32, 24))
        assert _is(list(px), name), f"{name}: {px}"
        others = {"blue", "green", "yellow", "red"} - {name}
        assert not any(_is(list(px), o) for o in others), (
            f"{name} also classifies as another colour: {px}"
        )
    assert base64.b64decode(base64.b64encode(_jpeg(BLUE, 4, 4)))[:2] == b"\xff\xd8"
