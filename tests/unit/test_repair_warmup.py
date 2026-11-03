"""T-0101 criteria 1-3 and 5: one warm-up repair request per payload automaton, through the real request shape.

**The assumed surface** (``local_astra/toolcall/warmup.py``; the task says "names may vary", these rows pin them):

- ``async def warm_repair(llm, kit, *, total_budget_s=30.0) -> WarmReport``
- ``WarmReport.per_tool``: a mapping tool name -> entry, one entry per tool of ``kit.payloads.tools``; an entry has
  ``status`` (``"ok"`` | ``"failed: <reason>"`` | ``"skipped"`` / ``"skipped: <why>"``), ``wall_ms`` (number),
  ``commits`` (commits delivered, int) and ``verdict`` (the ``live.repair_verdict`` status of the output's ``tool``
  commit, a string, or ``None`` when there was none).
- ``WarmReport.total_ms``: the whole warm-up's wall time in ms.

The module does not exist when these rows are written, so it is imported lazily (``_warmup()``) and every row fails
on that named assertion until it lands -- never on an ImportError at collection.

Every expected value comes from the task file, ``live.repair_request`` (the oracle for the ``cd`` fields),
``live.classify`` (the oracle for "a valid native call"), the real payload file and ``tests/fakes/llm.py``'s
``fake_tokenize`` (the oracle for "the engine's ids of that text").

Regime: CPU, pure asyncio over ``FakeLlm``; real payloads (``native_T4.json``) and calibration; no engine.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import importlib
import re
import struct
import time
from collections.abc import AsyncIterator, Mapping
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor.session import ToolKit
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.llm import Commit, ImagePart, LlmOptions, Prompt, VideoPart
from local_astra.toolcall import live, native
from local_astra.toolcall import tools as toolset
from tests.fakes.llm import FakeLlm, fake_tokenize, http_error

TAG_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
OPEN = native.TOOL_CALL_OPEN
#: a valid native press_key call (classify: valid) -- what every scripted warm-up output says unless a row says else
VALID_OUT = 'call:press_key{combo:<|"|>Enter<|"|>}<tool_call|>'
#: x = 5000 is outside 0-999 (T-0100 H-T99e): repair_verdict refuses it
INVALID_OUT = "call:click_at{x:5000,y:5}<tool_call|>"


def _warmup() -> ModuleType:
    try:
        module = importlib.import_module("local_astra.toolcall.warmup")
    except ModuleNotFoundError as exc:
        # an AssertionError, not pytest.fail: the pending convention tolerates only that or NotImplementedError
        raise AssertionError(
            f"T-0101: local_astra/toolcall/warmup.py does not exist yet ({exc})"
        ) from exc
    assert callable(getattr(module, "warm_repair", None)), "T-0101: warmup.warm_repair is missing"
    return module


def _kit(payloads: live.Payloads | None = None) -> ToolKit:
    tool_set = toolset.load()
    return ToolKit(
        mode="dry_run",
        tools=tuple(native.tools_field(tool_set)),
        tools_version=tool_set.version,
        executor=DryRunToolExecutor(),
        payloads=payloads if payloads is not None else live.load_payloads(),
        calibration=live.load_calibration(),
        repair_off=None,
    )


def _llm(out: str = VALID_OUT, cls: type[FakeLlm] = FakeLlm) -> FakeLlm:
    llm = cls()
    llm.script_continuation(
        Commit(request_id="?", canvas_idx=0, segment="thought", text="ok"),
        FakeLlm.tool_commit(out, stop_id=50, canvas_idx=1),
    )
    return llm


def _run(llm: Any, kit: ToolKit, **kw: Any) -> Any:
    return asyncio.run(_warmup().warm_repair(llm, kit, **kw))


def _sha_to_tool(payloads: live.Payloads) -> dict[str, str]:
    return {p.automaton_sha256: name for name, p in payloads.tools.items()}


def _decode(draft: str) -> list[int]:
    raw = base64.b64decode(draft)
    return list(struct.unpack(f"<{len(raw) // 4}i", raw))


def _entry(report: Any, tool: str) -> Any:
    return report.per_tool[tool]


class _Tracking(FakeLlm):
    """Counts generators alive at once (sequential = never two) and whether each was consumed to its end."""

    def __init__(self) -> None:
        super().__init__()
        self.alive = 0
        self.max_alive = 0
        self.finished: list[str] = []

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        self.alive += 1
        self.max_alive = max(self.max_alive, self.alive)
        try:
            async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
                yield c
            self.finished.append(request_id)
        finally:
            self.alive -= 1


# =====================================================================================================
# Criterion 1: one request per payload automaton
# =====================================================================================================


def _variant_twins() -> live.Payloads:
    """The real payloads with google_search given scroll's n_states and bucket: the two now share every size a
    kernel-variant guess could key on (S, bucket, np2, S mod 16), and still have different automata."""
    real = live.load_payloads()
    scroll = real.tools["scroll"]
    tools = dict(real.tools)
    tools["google_search"] = dataclasses.replace(
        real.tools["google_search"], n_states=scroll.n_states, bucket=scroll.bucket
    )
    return live.Payloads(tools=tools, manifest=real.manifest)


@pytest.mark.parametrize("fixture", ["real", "variant_twins"])
def test_one_request_per_automaton_sha(fixture: str) -> None:
    """Adversarial (the most likely failure): dedup by bucket or by kernel variant sends fewer than five.

    The real payloads already share buckets: scroll and google_search are both bucket 64, press_key and type_at both
    bucket (np2) 128 -- asserted here as the fixture's premise. ``variant_twins`` makes scroll and google_search
    identical in size too. Either way: five generates, one per ``automaton_sha256``, in tool-name order, one at a
    time, each consumed to its end."""
    payloads = live.load_payloads() if fixture == "real" else _variant_twins()
    t = payloads.tools
    assert t["scroll"].bucket == t["google_search"].bucket == 64, "fixture premise: a shared bucket"
    assert t["press_key"].bucket == t["type_at"].bucket == 128, "fixture premise: a shared np2"
    if fixture == "variant_twins":
        assert (t["scroll"].n_states, t["scroll"].bucket) == (
            t["google_search"].n_states,
            t["google_search"].bucket,
        )
    llm = _llm(cls=_Tracking)
    assert isinstance(llm, _Tracking)
    _run(llm, _kit(payloads))
    sent = [cd.automaton_sha256 for cd in llm.cd_log]
    expected = [t[name].automaton_sha256 for name in sorted(t)]
    assert len(expected) == 5 and len(set(expected)) == 5
    assert sent == expected, (
        "criterion 1: one request per payload automaton, sorted by tool name, none merged; sent "
        f"{[_sha_to_tool(payloads).get(s, s[:12]) for s in sent]}"
    )
    assert llm.max_alive == 1, (
        f"criterion 1: sequential, but {llm.max_alive} generates were alive at once"
    )
    assert len(llm.finished) == 5, "criterion 2: every warm-up's commits are consumed to the end"


def _shared_sha() -> live.Payloads:
    """The real payloads with google_search given scroll's payload: two tools, one automaton (one sha)."""
    real = live.load_payloads()
    tools = dict(real.tools)
    tools["google_search"] = real.tools["scroll"]
    return live.Payloads(tools=tools, manifest=real.manifest)


@pytest.mark.parametrize("outcome", ["ok", "failed"])
def test_shared_sha_warmed_once_both_reported(outcome: str) -> None:
    """Ruling W2: one request per distinct ``automaton_sha256``; two tools sharing one are warmed once and both are
    reported with that request's outcome. Name order click_at, google_search, press_key, scroll, type_at gives four
    distinct shas, the shared one first met at google_search (the 2nd request); ``failed`` answers it HTTP 500."""
    payloads = _shared_sha()
    t = payloads.tools
    assert t["google_search"].automaton_sha256 == t["scroll"].automaton_sha256, "fixture premise"
    llm = _llm()
    if outcome == "failed":
        llm.continuation_errors = [None, http_error(500), None, None]
    report = _run(llm, _kit(payloads))
    sent = [cd.automaton_sha256 for cd in llm.cd_log]
    expected = [
        t[n].automaton_sha256 for n in ("click_at", "google_search", "press_key", "type_at")
    ]
    assert sent == expected, (
        "W2: one request per distinct sha, in tool-name order of first appearance"
    )
    shared = (_entry(report, "google_search"), _entry(report, "scroll"))
    for e in shared:
        if outcome == "ok":
            assert (e.status, e.commits, e.verdict) == ("ok", 2, "valid"), e.status
        else:
            assert e.status.startswith("failed: ") and "500" in e.status, e.status
    assert shared[0].status == shared[1].status, "W2: both tools carry the one request's outcome"
    for name in ("click_at", "press_key", "type_at"):
        assert _entry(report, name).status == "ok", name


def test_cd_matches_repair_request_except_draft_and_tag() -> None:
    """Each request's ``cd`` equals ``live.repair_request(name, ids, 50, tag, payloads, calibration)`` in every field
    except ``draft`` and ``record_tag`` (the oracle is the session's own builder)."""
    kit = _kit()
    assert kit.payloads is not None and kit.calibration is not None
    llm = _llm()
    _run(llm, kit)
    tool_of = _sha_to_tool(kit.payloads)
    assert len(llm.cd_log) == 5
    for cd in llm.cd_log:
        assert cd is not None, "criterion 1: a warm-up request without cd is not the repair path"
        name = tool_of[cd.automaton_sha256]
        oracle = live.repair_request(name, [238], 50, "oracle", kit.payloads, kit.calibration)
        skip = {"draft", "record_tag"}
        assert cd.model_dump(exclude=skip) == oracle.model_dump(exclude=skip), name


def test_options_prompt_and_tools() -> None:
    """Options are ``live.REPAIR_OPTIONS``; the prompt is ``live.repair_prompt`` of a fixed minimal pass-1 prompt (the
    same for every tool; no image or video: no screen, memory or user data), carries the kit's ``tools`` field and is
    a continuation ending in ``<|tool_call>``."""
    kit = _kit()
    llm = _llm()
    _run(llm, kit)
    assert len(llm.generates) == 5
    pass1s = []
    for _rid, prompt, options in llm.generates:
        assert options == live.REPAIR_OPTIONS, options
        assert prompt.tools == kit.tools, (
            "criterion 1: the prompt carries the session's tools field"
        )
        assert prompt.continue_final_message is True
        last = prompt.messages[-1]
        assert last.role == "assistant"
        text = last.text()
        assert text.endswith(OPEN), text[-40:]
        pass1 = Prompt(
            messages=prompt.messages[:-1],
            prefix_len=prompt.prefix_len,
            thinking=prompt.thinking,
            assists=prompt.assists,
            tools=prompt.tools,
        )
        assert live.repair_prompt(pass1, text[: -len(OPEN)]) == prompt, (
            "criterion 1: the prompt is live.repair_prompt(pass1, spoken_prefix)"
        )
        for m in prompt.messages:
            assert not any(isinstance(p, ImagePart | VideoPart) for p in m.content), (
                "criterion 1: no screen or user data in the warm-up prompt"
            )
        pass1s.append(pass1)
    assert all(p == pass1s[0] for p in pass1s), (
        "criterion 1: one fixed pass-1 prompt for every tool"
    )


def test_tap_off_and_ids() -> None:
    """Criterion 2: ``xargs == {"astra_tap": "off"}`` on every request, request ids start ``warmup-`` (distinct), and
    the ``record_tag`` starts ``warmup:``, at most 64 chars of ``[A-Za-z0-9_.:-]``; nothing is left running."""
    llm = _llm()
    _run(llm, _kit())
    assert len(llm.generates) == 5
    assert llm.xargs_log == [{"astra_tap": "off"}] * 5, llm.xargs_log
    ids = [rid for rid, _, _ in llm.generates]
    assert all(rid.startswith("warmup-") for rid in ids), ids
    assert len(set(ids)) == 5, ids
    for cd in llm.cd_log:
        assert cd.record_tag.startswith("warmup:"), cd.record_tag
        assert TAG_RE.fullmatch(cd.record_tag), cd.record_tag
    for body in llm.bodies:
        assert body["vllm_xargs"].get("astra_tap") == "off", "the tap opt-out reaches the wire"
    assert llm.active == frozenset(), f"warm-up requests left running: {sorted(llm.active)}"


def test_canonical_drafts_are_valid() -> None:
    """Criterion 1: the draft is the engine's ids of a fixed call for that tool, which ``live.classify`` judges
    ``valid`` (under T-0100: in-range coordinates, an in-list combo) and names that tool; the seed is those ids plus
    stop id 50. A payload with no canonical draft is ``skipped: no canonical draft`` and is not sent."""
    real = live.load_payloads()
    extra_automaton = "zz-not-a-v1-tool"
    extra = live.Payload(
        automaton=extra_automaton,
        automaton_sha256=hashlib.sha256(extra_automaton.encode("ascii")).hexdigest(),
        n_states=3,
        bucket=32,
    )
    payloads = live.Payloads(tools={**real.tools, "zz_tool": extra}, manifest=real.manifest)
    llm = _llm()
    report = _run(llm, _kit(payloads))
    tool_of = _sha_to_tool(payloads)
    sent = [tool_of[cd.automaton_sha256] for cd in llm.cd_log]
    assert sent == sorted(real.tools), sent
    for cd in llm.cd_log:
        name = tool_of[cd.automaton_sha256]
        ids = _decode(cd.draft)
        assert ids[-1] == 50, f"{name}: the seed ends in stop id 50"
        texts = [t for t in llm.tokenized if fake_tokenize(t) == ids[:-1]]
        assert texts, f"{name}: the draft is not llm.tokenize of any text the warm-up tokenised"
        v = live.classify(texts[0])
        assert (v.status, v.name) == ("valid", name), (name, texts[0], v)
    assert _entry(report, "zz_tool").status == "skipped: no canonical draft"


# =====================================================================================================
# Criterion 3: never stops, never raises, never turns repair off
# =====================================================================================================


@pytest.mark.parametrize("how", ["http_500_and_raise", "tokenize_raises"])
def test_failure_isolated(how: str) -> None:
    """A request that raises or ends 5xx is ``failed: <reason>``; the next tool is still tried; ``warm_repair`` does
    not raise. Sorted order is click_at, google_search, press_key, scroll, type_at."""
    llm = _llm()
    if how == "http_500_and_raise":
        llm.continuation_errors = [None, http_error(500), None, RuntimeError("boom"), None]
    else:
        llm.tokenize_error = RuntimeError("tokenize down")
    report = _run(llm, _kit())
    statuses = {t: _entry(report, t).status for t in sorted(live.load_payloads().tools)}
    if how == "http_500_and_raise":
        assert len(llm.generates) == 5, "criterion 3: every tool is still tried"
        assert (
            statuses["google_search"].startswith("failed: ") and "500" in statuses["google_search"]
        )
        assert statuses["scroll"].startswith("failed: "), statuses
        assert [t for t, s in statuses.items() if s == "ok"] == ["click_at", "press_key", "type_at"]
    else:
        failed = [t for t, s in statuses.items() if s.startswith("failed: ")]
        assert len(failed) == 1, statuses
        assert sum(s == "ok" for s in statuses.values()) == 4, statuses


def test_overrun_is_failed_aborted_and_bounded() -> None:
    """Ruling W1: a request's deadline is the *remaining total budget*. Every request here hangs, so the first one
    (click_at) holds the whole 0.3 s cap, is aborted and recorded failed, and the other four are ``skipped`` without
    being sent: exactly one generate. (A cap/n share would fail click_at early and then send, and fail, the rest.)"""
    llm = _llm()
    llm.hold_continuation()
    t0 = time.perf_counter()
    report = _run(llm, _kit(), total_budget_s=0.3)
    wall = time.perf_counter() - t0
    llm.release_continuation()
    assert wall < 0.3 + 1.5, f"criterion 3: the 0.3 s cap took {wall:.2f} s"
    sent = [rid for rid, _, _ in llm.generates]
    assert len(sent) == 1, (
        f"W1: only the first request is sent before the cap is spent; sent {sent}"
    )
    assert sent[0] in llm.aborted, (sent, llm.aborted)
    first = _entry(report, "click_at")
    assert first.status.startswith("failed"), f"the hung first request is failed: {first.status}"
    assert first.wall_ms >= 0.3 * 1000 * 0.9, (
        f"W1: click_at held the remaining budget, not a share ({first.wall_ms} ms)"
    )
    for t in ("google_search", "press_key", "scroll", "type_at"):
        assert _entry(report, t).status == "skipped", (t, _entry(report, t).status)


class _SlowFirstRepair(FakeLlm):
    """The first generate waits ``delay_s`` before its stream starts (a slow but finishing first use)."""

    delay_s = 0.5

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        self.started = getattr(self, "started", 0) + 1
        if (
            self.started == 1
        ):  # counted on entry: a request cut during the delay still counts as the first
            await asyncio.sleep(self.delay_s)
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c


def test_slow_first_request_within_remaining_budget_is_ok() -> None:
    """Ruling W1, the other direction: the first request takes 0.5 s of a 1.0 s cap -- longer than a cap/5 share
    (0.2 s), shorter than the remaining budget -- and comes back ``ok``; the other four, instant, are ``ok`` too."""
    llm = _llm(cls=_SlowFirstRepair)
    report = _run(llm, _kit(), total_budget_s=1.0)
    first = _entry(report, "click_at")
    assert first.status == "ok", (
        f"W1: a request inside the remaining budget is not cut: {first.status}"
    )
    assert len(llm.generates) == 5, [rid for rid, _, _ in llm.generates]
    assert first.wall_ms >= 500 * 0.9, first.wall_ms
    for t in sorted(live.load_payloads().tools):
        assert _entry(report, t).status == "ok", (t, _entry(report, t).status)
    assert llm.aborted == [], llm.aborted


class _SlowTokenize(FakeLlm):
    """The first tokenize blocks the loop for 0.5 s: time no per-request timeout can take back."""

    async def tokenize(self, text: str) -> list[int]:
        if not self.tokenized:
            time.sleep(0.5)
        return await super().tokenize(text)


def test_total_budget_skips_rest() -> None:
    """When the cap is spent the rest are ``skipped`` and not sent: the first tool's tokenize alone spends 0.5 s of a
    0.2 s cap, whatever share each request is given."""
    llm = _llm(cls=_SlowTokenize)
    report = _run(llm, _kit(), total_budget_s=0.2)
    tool_of = _sha_to_tool(live.load_payloads())
    sent = {tool_of[cd.automaton_sha256] for cd in llm.cd_log}
    rest = sorted(tool_of.values())[1:]
    for t in rest:
        assert _entry(report, t).status.startswith("skipped"), (t, _entry(report, t).status)
        assert t not in sent, f"{t} was sent after the cap was spent"
    assert report.total_ms >= 500, report.total_ms


def test_cancel_propagates() -> None:
    """``warm_repair`` never raises except ``CancelledError``: cancelling it is not swallowed."""
    llm = _llm()
    llm.hold_continuation()

    async def drive() -> None:
        task = asyncio.ensure_future(_warmup().warm_repair(llm, _kit()))
        await asyncio.sleep(0.05)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(drive())


def test_kit_unchanged() -> None:
    """The kit object is unchanged by a warm-up whose every request fails: same payloads, calibration and tools
    objects, ``repair_off`` still ``None``."""
    kit = _kit()
    before = (kit.payloads, kit.calibration, kit.tools, kit.repair_off, kit.mode, kit.executor)
    llm = _llm()
    llm.continuation_errors = [http_error(500)] * 5
    _run(llm, kit)
    after = (kit.payloads, kit.calibration, kit.tools, kit.repair_off, kit.mode, kit.executor)
    assert all(a is b for a, b in zip(before, after, strict=True)), (
        "criterion 3: the kit was changed"
    )
    assert kit.repair_off is None


# =====================================================================================================
# Criterion 5: report, never gate
# =====================================================================================================


@pytest.mark.parametrize(("out", "verdict"), [(INVALID_OUT, "refused"), (VALID_OUT, "valid")])
def test_invalid_output_is_reported_not_failed(out: str, verdict: str) -> None:
    """The output's ``live.repair_verdict`` status is reported; an invalid output is still ``ok``. Each entry carries
    its wall ms and the commits delivered (2 scripted per request); the total is at least the sum of the parts."""
    assert live.repair_verdict(out).status == verdict, "fixture premise"
    llm = _llm(out)
    report = _run(llm, _kit())
    parts = 0.0
    for t in sorted(live.load_payloads().tools):
        e = _entry(report, t)
        assert e.status == "ok", (t, e.status)
        assert e.verdict == verdict, (t, e.verdict)
        assert e.commits == 2, (t, e.commits)
        assert e.wall_ms >= 0
        parts += e.wall_ms
    assert report.total_ms >= parts - 1.0, (report.total_ms, parts)
