"""T-0128c C1: the real denoising view end to end on CPU (fake tap -> relay -> HudServer -> page).

A fake tap replays the committed real trajectory (``tests/fixtures/canvas/trajectory_806da12.msgpack``,
51 frames of one request, 5 canvases, recorded at ~25 Hz on the 5090 at 806da12; provenance in
``tests/fixtures/canvas/MANIFEST.md``) with the plugin's own framing: the plugin's
``StepPublisher`` itself (``load_plugin_module("tap")``), so topic and msgpack encoding are the
engine's, on an ``ipc://`` endpoint in a fresh temp directory (never 8002). The real relay
(``local_astra.conductor.canvas_relay``) subscribes and publishes through the real ``HudServer``
of a real ``Session`` over the fakes (``tests/integration/test_hud_rehearsal.Live``, whose wire tap
records every published message), and the real ``hud/`` page in headless Chrome draws them.

**What C1 asserts** (docs/tasks/T-0128c.md C1), every expected value from the fixture or the wire:

* the page decodes >= 90 % of the canvas frames the relay published (``data-denoise-frames``
  against the ``astra.canvas.step`` count on the server's wire);
* ``data-denoise-digest`` takes >= 5 distinct values;
* every label the Thinking pane shows is one of the (canvas_idx, step) pairs the tap sent, in
  B4's two forms (``canvas k · step s/N`` for a denoise frame, ``canvas k · committed at step
  s/N`` for a commit frame) -- the kind must match the sent frame's phase too.

Supporting rows, derived from the criteria they cite: before the first frame the root has no
element child and no label is shown (B5); every relayed frame equals, field for field, a sent frame
(A2, observed end to end); all five commits reach the server (A3); the page's malformed counter
stays 0 (B1: every real frame decodes).

**The relay seam** (group A as merged, ruling O-A)::

    relay = CanvasRelay(publish=..., clock=..., state=..., session_log=None, endpoint=...)
    await relay.start()
    await relay.aclose()

**Page readiness:** frames are pushed only after all six panes have mounted
(``main > .pane``, ``ALL_PANES`` of ``test_hud_redesign``); the group B implementer saw the first
3-5 frames lost when a page was fed after only three.

``state`` is a zero-argument callable returning an ``AstraTurnState`` (cli.py passes
``session.turn_state_message``); here it says THINKING throughout, because admission by turn state
is A4's matter and is pinned in group A's unit rows. The session itself stays IDLE.

**Recording on the page.** A ``MutationObserver`` installed on the Thinking pane before the
first frame records every digest and every label as they change, so nothing depends on polling
fast enough to catch a 40 ms frame. Labels are matched in the pane's text and in its
``aria-label`` attributes (B3 states the label there too); the markup is group B's.

Regime: CPU, no GPU, no model; Chrome 147 headless; real ``SessionClock``; replay at 25 Hz
(the recorded rate, 24.6 Hz); n = 1; budget ~10 s. No number here is a latency claim.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import msgpack
import pytest

from local_astra.contracts.protocol import AstraCanvasStep, AstraTurnState
from local_astra.contracts.turn import TurnState
from tests.fakes.diffusion_sampler import load_plugin_module
from tests.integration.test_hud_headless import requires_chrome, serve_hud
from tests.integration.test_hud_redesign import ALL_PANES
from tests.integration.test_hud_rehearsal import Live, open_page, page_url

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "canvas" / "trajectory_806da12.msgpack"
RELAY_MODULE = "local_astra.conductor.canvas_relay"
REPLAY_HZ = 25.0
#: let the SUB's connect complete before the first frame (ZeroMQ's slow joiner); frames sent
#: earlier would be lost by PUB/SUB design, and C1 compares against what the relay published
SUBSCRIBE_SETTLE_S = 1.0
SETTLE_TIMEOUT_S = 10.0

LABEL_RE = r"canvas (\d+) · (committed at step|step) (\d+)/(\d+)"

RECORDER_JS = (
    r"""
(() => {
  const root = document.querySelector('[data-denoise-root]');
  if (!root) return {error: 'no [data-denoise-root]'};
  const pane = root.closest('.pane-thinking') || root.parentElement || document.body;
  const LABEL = new RegExp("""
    + json.dumps(LABEL_RE)
    + r""", 'g');
  const labelEl = document.querySelector('[data-denoise-label]');
  const rec = {
    digests: [],
    labels: {},
    labelSeq: [],
    before: {
      elementChildren: root.children.length,
      frames: root.getAttribute('data-denoise-frames'),
    },
  };
  window.__t0128c = rec;
  const scan = () => {
    const d = root.getAttribute('data-denoise-digest');
    if (d !== null && rec.digests[rec.digests.length - 1] !== d) rec.digests.push(d);
    const l = labelEl ? (labelEl.textContent || '') : '';
    if (l && rec.labelSeq[rec.labelSeq.length - 1] !== l) rec.labelSeq.push(l);
    const texts = [pane.innerText || ''];
    for (const el of pane.querySelectorAll('[aria-label]')) texts.push(el.getAttribute('aria-label') || '');
    if (pane.hasAttribute('aria-label')) texts.push(pane.getAttribute('aria-label'));
    for (const t of texts) for (const m of t.matchAll(LABEL)) rec.labels[m[0]] = true;
  };
  scan();
  rec.before.labels = Object.keys(rec.labels);
  new MutationObserver(scan).observe(pane, {
    subtree: true, attributes: true, childList: true, characterData: true,
  });
  if (labelEl && !pane.contains(labelEl)) {
    new MutationObserver(scan).observe(labelEl, {subtree: true, childList: true, characterData: true});
  }
  return {ok: true, before: rec.before, hasLabel: !!labelEl};
})()
"""
)

READ_JS = r"""
(() => {
  const root = document.querySelector('[data-denoise-root]');
  const rec = window.__t0128c;
  const ign = (document.querySelector('#ignored-count') || {}).textContent || '';
  const m = /(\d+)u\/(\d+)m\/(\d+)r/.exec(ign);
  return {
    frames: root.getAttribute('data-denoise-frames'),
    draws: root.getAttribute('data-denoise-draws'),
    digest: root.getAttribute('data-denoise-digest'),
    digests: rec ? rec.digests : null,
    labels: rec ? Object.keys(rec.labels) : null,
    labelSeq: rec ? rec.labelSeq : null,
    label: (document.querySelector('[data-denoise-label]') || {}).textContent || null,
    ignored: ign,
    malformed: m ? Number(m[2]) : null,
  };
})()
"""


def relay_class() -> Any:
    try:
        mod = importlib.import_module(RELAY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == RELAY_MODULE:
            raise AssertionError(
                "T-0128c group A not merged: local_astra/conductor/canvas_relay.py does not exist"
            ) from None
        raise
    cls = getattr(mod, "CanvasRelay", None)
    if cls is None:
        raise AssertionError(f"T-0128c group A: {RELAY_MODULE}.CanvasRelay is missing")
    return cls


def fixture_frames() -> list[dict[str, Any]]:
    if not FIXTURE.exists():
        raise AssertionError(
            f"T-0128c group A not merged: the trajectory fixture {FIXTURE.relative_to(REPO)} "
            "is missing (copied from artifacts/canvas/20261003T003342-lean-cl256-s48-806da12)"
        )
    data = msgpack.unpackb(FIXTURE.read_bytes(), raw=False)
    assert isinstance(data, dict) and {"regime", "request", "frames"} <= set(data), (
        "the fixture is a msgpack dict {regime, request, frames}"
    )
    frames = data["frames"]
    assert isinstance(frames, list) and len(frames) == data["request"]["frames"] == 51, (
        "the 806da12 trajectory has 51 frames"
    )
    # the fixture is the real 806da12 trajectory: one request, 5 commits (MANIFEST), each valid
    for f in frames:
        AstraCanvasStep.model_validate(f)
    assert len({f["request_id"] for f in frames}) == 1, "expected one request in the fixture"
    assert sum(f["phase"] == "commit" for f in frames) == 5, "expected 5 commit frames"
    return frames


def expected_label(f: dict[str, Any]) -> str:
    """B4, verbatim from the frame (H-T128c-5: the wire's 0-based canvas_idx)."""
    if f["phase"] == "commit":
        return f"canvas {f['canvas_idx']} · committed at step {f['step']}/{f['max_steps']}"
    return f"canvas {f['canvas_idx']} · step {f['step']}/{f['max_steps']}"


def as_dict(message: Any) -> dict[str, Any] | None:
    """A published canvas message as a plain dict (the relay may hand over a model or a dict)."""
    if isinstance(message, AstraCanvasStep):
        return message.model_dump(mode="json", exclude_none=True)
    if isinstance(message, dict) and message.get("type") == "astra.canvas.step":
        return AstraCanvasStep.model_validate(message).model_dump(mode="json", exclude_none=True)
    return None


def canonical(f: dict[str, Any]) -> dict[str, Any]:
    return AstraCanvasStep.model_validate(f).model_dump(mode="json", exclude_none=True)


def stub_pieces_of(d: dict[str, Any]) -> dict[str, list[str]]:
    """The T-0159 fields the stub source (``#<id>``) gives a frame: its own ids, its own phase."""
    from tests.fakes.canvas_tap import stub_piece

    if d.get("phase") == "commit":
        return {"committed_pieces": [stub_piece(i) for i in d["committed_tokens"]]}
    return {"argmax_pieces": [stub_piece(i) for i in d["argmax"]]}


def assert_sent_unchanged(d: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    """T-0128c A2 end to end, as amended by T-0159 A3: ``d`` without the piece fields is one of
    the sent frames, and the piece fields it carries (if any) are the stub's over its own ids."""
    from tests.fakes.canvas_tap import PIECE_FIELDS, without_pieces

    assert without_pieces(d) in sent, f"A2: the relay published a frame the tap never sent: {d}"
    carried = {k: list(d[k]) for k in PIECE_FIELDS if d.get(k) is not None}
    if carried:
        assert carried == stub_pieces_of(d), f"T-0159 A3: pieces not of the frame's own ids: {d}"


#: T-0159: installed before the page's own scripts run (``Page.addScriptToEvaluateOnNewDocument``):
#: every text frame the page's WebSocket receives that names ``astra.canvas.step`` is kept, so the
#: row can hand each to the page's own decoder afterwards. Nothing else on the page changes.
WS_TAP_JS = r"""
(() => {
  const Native = window.WebSocket;
  window.__t0159raw = [];
  class Tapped extends Native {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', (e) => {
        if (typeof e.data === 'string' && e.data.includes('"astra.canvas.step"')) window.__t0159raw.push(e.data);
      });
    }
  }
  window.WebSocket = Tapped;
})();
"""

#: The page's own decoder (``hud/src/protocol.js``, the module ``main.js`` imports) over every kept
#: frame: what the page decoded, field for field.
PAGE_DECODE_JS = r"""
(async () => {
  const { decode } = await import(new URL('src/protocol.js', document.baseURI).href);
  return (window.__t0159raw || []).map((raw) => {
    const d = decode(raw);
    return d.kind === 'astra' ? { kind: 'astra', msg: d.msg } : { kind: d.kind };
  });
})()
"""


def stub_pieces_or_nothing(cls: Any, source: Any) -> dict[str, Any]:
    """T-0159: the relay decodes with ``source`` when it declares the seam (so ``start`` never
    loads the real tokenizer here); BASE gets nothing and adds no pieces."""
    from tests.fakes.canvas_tap import optional_piece_kwargs

    return optional_piece_kwargs(cls, source)


async def replay(
    tmp_path: Path,
    schedule: list[tuple[dict[str, Any], float]],
    *,
    relay_kwargs: dict[str, Any] | None = None,
    tap_ws: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    """Send ``schedule`` (frame, seconds to sleep after it) through the fake tap -> the real relay
    -> HudServer -> the real page; return the settled page state (``READ_JS``) and the server's
    wire (every published message).

    T-0159: ``relay_kwargs`` default to a ``StubPieces`` source when the relay has the seam.
    ``tap_ws`` installs ``WS_TAP_JS`` and reloads the page before the frames, and adds the page's
    own decode of every canvas frame it received as ``seen["page_decoded"]``."""
    from tests.fakes.canvas_tap import StubPieces

    cls = relay_class()
    if relay_kwargs is None:
        relay_kwargs = stub_pieces_or_nothing(cls, StubPieces())
    tap = load_plugin_module("tap", "T-0128c C1")

    ipc_dir = tempfile.mkdtemp(prefix="t0128c-")  # short: ipc paths are limited to ~107 bytes
    endpoint = f"ipc://{ipc_dir}/tap.sock"
    publisher = tap.StepPublisher(endpoint)

    live = Live(tmp_path, [], ms_per_word=400)
    await live.start()
    hud, hud_port = serve_hud()
    relay = None
    try:

        def thinking() -> AstraTurnState:
            return AstraTurnState(state=TurnState.THINKING, turn_id="t-e2e")

        async with open_page(page_url(hud_port, live.server.url)) as page:
            if tap_ws:
                await asyncio.to_thread(
                    page.chrome._cmd, "Page.addScriptToEvaluateOnNewDocument", source=WS_TAP_JS
                )
                await page.navigate(page_url(hud_port, live.server.url))
                await page.until(
                    "Array.isArray(window.__t0159raw)", "the WebSocket tap is installed"
                )
                await page.connected()
            await page.until(
                f"document.querySelectorAll('main > .pane').length >= {ALL_PANES}",
                f"all {ALL_PANES} panes mounted",
            )
            installed = await page.evaluate(RECORDER_JS)
            assert isinstance(installed, dict) and installed.get("ok"), installed
            before = installed["before"]
            # B5: nothing is drawn before the first real frame
            assert before["elementChildren"] == 0, (
                f"B5: [data-denoise-root] has {before['elementChildren']} element children "
                "before any frame was sent"
            )
            assert before["labels"] == [], f"B5: a label is shown before any frame: {before}"

            relay = cls(
                publish=live.server.publish,
                clock=live.clock,
                state=thinking,
                session_log=live.session.session_log,
                endpoint=endpoint,
                **relay_kwargs,
            )
            await relay.start()
            publisher._socket()  # bind now, so the relay's connect can complete before frame 1
            await asyncio.sleep(SUBSCRIBE_SETTLE_S)

            for f, pause_s in schedule:
                assert publisher.send(f), "the fake tap's PUB dropped a frame at its HWM"
                await asyncio.sleep(pause_s)

            # wait for the page to settle: its frame counter stops moving
            deadline = time.monotonic() + SETTLE_TIMEOUT_S
            last: Any = None
            stable_since = time.monotonic()
            seen: dict[str, Any] = {}
            while time.monotonic() < deadline:
                seen = await page.evaluate(READ_JS)
                if seen.get("frames") != last:
                    last, stable_since = seen.get("frames"), time.monotonic()
                elif time.monotonic() - stable_since > 1.0:
                    break
                await asyncio.sleep(0.1)
            if tap_ws:
                seen["page_decoded"] = await page.evaluate(PAGE_DECODE_JS)
    finally:
        if relay is not None:
            await relay.aclose()
        publisher.close()
        hud.shutdown()
        await live.aclose()
        shutil.rmtree(ipc_dir, ignore_errors=True)
    return seen, list(live.wire)


@requires_chrome
async def test_a_replayed_real_trajectory_reaches_the_page_through_the_real_relay(tmp_path):
    frames = fixture_frames()
    seen, wire = await replay(tmp_path, [(f, 1.0 / REPLAY_HZ) for f in frames])

    published = [d for d in (as_dict(m) for m in wire) if d is not None]
    sent = [canonical(f) for f in frames]

    # A2 end to end: every relayed frame is one the tap sent, unchanged (except the T-0159 piece
    # fields, which are the stub's); none invented
    for d in published:
        assert_sent_unchanged(d, sent)
    assert len(published) >= 25, (
        f"only {len(published)} of {len(frames)} sent frames were published; the relay "
        "coalesces at <= 30 Hz per request and the replay ran at 25 Hz"
    )
    commits = [d for d in published if d["phase"] == "commit"]
    assert len(commits) == 5, f"A3: commits are never coalesced away; published {len(commits)}/5"

    # C1 (a): the page decodes >= 90 % of what the relay published
    assert seen.get("frames") is not None, f"the page never set data-denoise-frames: {seen}"
    decoded = int(seen["frames"])
    need = math.ceil(0.9 * len(published))
    assert decoded >= need, (
        f"C1: the page decoded {decoded} of {len(published)} published frames (need >= {need}); "
        f"page state {seen}"
    )
    assert seen.get("malformed") == 0, (
        f"B1: a real tap frame decoded as malformed on the page (#ignored-count {seen.get('ignored')!r})"
    )

    # C1 (b): >= 5 distinct digests
    digests = seen.get("digests") or []
    assert len(set(digests)) >= 5, f"C1: only {len(set(digests))} distinct digests: {digests}"

    # C1 (c): every shown label is one of the (canvas_idx, step) pairs sent, kind included
    labels = seen.get("labels") or []
    allowed = {expected_label(f) for f in frames}
    assert labels, "C1: the Thinking pane never showed a 'canvas k · step s/N' label"
    stray = sorted(set(labels) - allowed)
    assert not stray, f"C1: labels shown that no sent frame carries: {stray}"


# --------------------------------------------------------------------------- O-C5: prefill, then generate

#: two engine ids of the real shape (chatcmpl-<16 hex>-<8 hex>); the prefill's is invented, the
#: generate's is the 806da12 recording's own
PREFILL_ID = "chatcmpl-00000000000000aa-0000000a"
PREFILL_FRAMES = 3
#: the generate request's first frame follows the prefill's last by about this much (O-C5)
TAKEOVER_GAP_S = 0.050


def fnv1a32(text: str) -> str:
    """FNV-1a 32-bit over the UTF-8 bytes, 8 lowercase hex digits (O-B (5)); an oracle written from
    the definition (offset basis 0x811c9dc5, prime 0x01000193, xor then multiply), pinned below on
    the published vector and on group B's hand-computed one (O-B3 F4)."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def test_fnv1a_oracle_matches_the_pinned_vectors():
    assert fnv1a32("a") == "e40c292c"
    assert fnv1a32("foobar") == "bf9cf968"
    assert fnv1a32("stable,kept,noise") == "914acc9c"  # tests/hud/denoise.test.mjs O-B3 F4


@requires_chrome
async def test_o_c5_a_prefill_then_generate_pair_ends_on_the_generate_requests_last_frame(tmp_path):
    """O-C5 (1), the live defect replayed on CPU: every turn opens with the anticipatory prefill,
    which denoises a few steps and is aborted when the generate request starts ~150 ms later; with
    the old 1500 ms focus window the generate's canvases (landing within 0.4-1 s) were never drawn.

    Here: the first 3 frames of the 806da12 trajectory under ``PREFILL_ID``, then, 50 ms later, the
    whole trajectory (51 frames, 25 Hz) under its own recorded id. B6 as ruled: the newest request
    takes the view at once. So the page's final label and digest are the generate request's last
    frame (``canvas 4 · committed at step 20/48``, all 256 positions committed), and the generate's
    own frames are drawn from its start: its frames 3..18 (canvas 0 from step 8 and all of canvas
    1, ~0.1-0.75 s after the takeover) carry labels the prefill never sent, and the label returns to
    the trajectory's first label after the prefill's third. Under the old window none of frames
    3..36 could be drawn (they arrive within 1.5 s of the prefill's last frame); the final label
    alone would not tell, since the 2 s generate outlasts the window.

    Regime: CPU, Chrome 147 headless, real relay and HudServer, 25 Hz, n = 1."""
    frames = fixture_frames()
    gen_id = frames[0]["request_id"]
    assert gen_id != PREFILL_ID
    prefill = [{**f, "request_id": PREFILL_ID} for f in frames[:PREFILL_FRAMES]]
    schedule = [(f, 1.0 / REPLAY_HZ) for f in prefill[:-1]]
    schedule.append((prefill[-1], TAKEOVER_GAP_S))
    schedule += [(f, 1.0 / REPLAY_HZ) for f in frames]
    seen, wire = await replay(tmp_path, schedule)

    published = [d for d in (as_dict(m) for m in wire) if d is not None]
    sent = [canonical(f) for f, _ in schedule]
    for d in published:
        assert_sent_unchanged(d, sent)
    pub_prefill = [d for d in published if d["request_id"] == PREFILL_ID]
    pub_gen = [d for d in published if d["request_id"] == gen_id]
    assert len(pub_prefill) == PREFILL_FRAMES, (
        f"premise: the relay published {len(pub_prefill)} of the {PREFILL_FRAMES} prefill frames"
    )
    from tests.fakes.canvas_tap import without_pieces

    assert pub_gen and without_pieces(pub_gen[-1]) == canonical(frames[-1]), (
        "premise: the generate request's last frame (its final commit) reached the server"
    )
    assert seen.get("malformed") == 0, f"B1: malformed on the page: {seen.get('ignored')!r}"

    last = frames[-1]
    assert last["phase"] == "commit" and len(last["committed_tokens"]) == last["canvas_length"]
    want_label = expected_label(last)
    assert want_label == "canvas 4 · committed at step 20/48"
    # B2: a commit frame of CL 256 with 256 committed tokens is 256 x "committed"
    want_digest = fnv1a32(",".join(["committed"] * last["canvas_length"]))
    assert seen.get("label") == want_label, (
        f"O-C5: the page's final label is {seen.get('label')!r}, not the generate request's last "
        f"frame {want_label!r} (prefill's last: {expected_label(prefill[-1])!r}); page {seen}"
    )
    assert seen.get("digest") == want_digest, (
        f"O-C5: the page's final digest {seen.get('digest')!r} is not the generate request's last "
        f"frame's ({want_digest}); page {seen}"
    )

    seq = seen.get("labelSeq") or []
    prefill_labels = [expected_label(f) for f in prefill]
    assert seq[:PREFILL_FRAMES] == prefill_labels, (
        f"premise/O-C5: the prefill's own frames are drawn while it is the newest: {seq[:6]}"
    )
    after = seq[PREFILL_FRAMES:]
    assert after and after[0] == expected_label(frames[0]), (
        "O-C5: the generate request takes the view at once -- the label after the prefill's third "
        f"frame should be the generate's first ({expected_label(frames[0])!r}); sequence {seq[:12]}"
    )
    early = {expected_label(f) for f in frames[PREFILL_FRAMES:19]}
    drawn_early = early & set(after)
    assert len(drawn_early) >= 12, (
        f"O-C5: only {len(drawn_early)} of the generate's {len(early)} frames 3..18 (the first "
        f"~0.75 s after the takeover) were shown: {sorted(drawn_early)}; sequence {seq[:30]}"
    )


# --------------------------------------------------------------------------- T-0159 A: stub pieces


@requires_chrome
async def test_every_frame_the_page_decodes_carries_the_stub_pieces_of_its_own_ids(tmp_path):
    """T-0159 A3 + A6 end to end: the recorded trajectory at 25 Hz through the real relay with a
    stub piece source (``id -> "#<id>"``, through the frozen seam), the real ``HudServer`` and the
    real page. The page's WebSocket frames are kept (``WS_TAP_JS``) and run through the page's own
    decoder (``hud/src/protocol.js``):

    * every canvas frame the page received decodes (``kind == "astra"``), and there are >= 90 % as
      many as the relay published;
    * each decoded denoise frame carries ``argmax_pieces`` = ``#<id>`` over its own ``argmax`` (CL
      entries) and no ``committed_pieces``; each commit frame carries ``committed_pieces`` over its
      own ``committed_tokens`` and no ``argmax_pieces``;
    * the digest is unchanged by the pieces: the final digest is the last frame's (a CL 256 commit
      of 256 ids: 256 x "committed", B2) and there are >= 5 distinct digests, as in C1.

    Regime: CPU, Chrome 147 headless, real ``SessionClock``, 25 Hz, n = 1."""
    from tests.fakes.canvas_tap import StubPieces, piece_kwargs

    frames = fixture_frames()
    cls = relay_class()
    seen, wire = await replay(
        tmp_path,
        [(f, 1.0 / REPLAY_HZ) for f in frames],
        relay_kwargs=piece_kwargs(cls, piece_source=StubPieces()),
        tap_ws=True,
    )
    published = [d for d in (as_dict(m) for m in wire) if d is not None]
    sent = [canonical(f) for f in frames]
    for d in published:
        assert_sent_unchanged(d, sent)
        assert any(d.get(k) is not None for k in ("argmax_pieces", "committed_pieces")), (
            f"T-0159 A3: the relay published {d['canvas_idx']}/{d['step']} without pieces"
        )
    decoded = seen.get("page_decoded")
    assert isinstance(decoded, list), f"the page decode did not run: {decoded!r}"
    refused = [d for d in decoded if d.get("kind") != "astra"]
    assert not refused, (
        f"A6: the page refused {len(refused)} canvas frames with pieces: {refused[:2]}"
    )
    msgs = [d["msg"] for d in decoded]
    assert len(msgs) >= math.ceil(0.9 * len(published)), (
        f"the page received {len(msgs)} of {len(published)} published canvas frames"
    )
    for m in msgs:
        want = stub_pieces_of(m)
        if m.get("phase") == "commit":
            assert m.get("argmax_pieces") is None, f"A1: a commit frame with argmax_pieces: {m}"
            assert list(m.get("committed_pieces") or []) == want["committed_pieces"], (
                f"A3/A6: commit {m['canvas_idx']}/{m['step']} decoded with {m.get('committed_pieces')}"
            )
        else:
            assert m.get("committed_pieces") is None, (
                f"A1: a denoise frame with committed_pieces: {m}"
            )
            assert list(m.get("argmax_pieces") or []) == want["argmax_pieces"], (
                f"A3/A6: denoise {m['canvas_idx']}/{m['step']} decoded with {m.get('argmax_pieces')}"
            )
            assert len(want["argmax_pieces"]) == m["canvas_length"]
    last = frames[-1]
    assert last["phase"] == "commit" and len(last["committed_tokens"]) == last["canvas_length"]
    assert seen.get("digest") == fnv1a32(",".join(["committed"] * last["canvas_length"])), (
        f"A8/B-5: the final digest moved with pieces on the wire: {seen.get('digest')}"
    )
    assert len(set(seen.get("digests") or [])) >= 5, seen.get("digests")
    assert seen.get("malformed") == 0, f"A6: malformed on the page: {seen.get('ignored')!r}"
