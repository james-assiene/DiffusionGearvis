"""The fakes implement the service protocols and behave like the contract says (docs/02 §5).

``backend == "real"`` variants are the same assertions against real services; they are wired
in T-0001/T-0007 (skipped with a task reference until then, per tests rule on xfail/skip).
"""

from __future__ import annotations

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.conductor.stt_client import MoshiSttClient
from local_astra.conductor.tts_client import PocketTtsClient
from local_astra.contracts.llm import Commit, LlmOptions, Message, Prompt, TextPart
from local_astra.contracts.memory import Episode, MemoryQuery
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocols import (
    AudioIO,
    LlmBackend,
    MemoryClient,
    PerceptionClient,
    SttClient,
    TtsBackend,
    TtsUtterance,
)
from local_astra.contracts.speech import SttEndOfTurnEvent
from tests.fakes import FakeAudioIO, FakeLlm, FakeMemory, FakePerception, FakeStt, FakeTts


def _real_not_wired(backend: str) -> None:
    if backend == "real":
        pytest.skip(
            "real backend wiring is T-0001/T-0007 (recorded traffic replayers + real clients)"
        )


def test_the_real_speech_clients_satisfy_the_speech_protocols():
    """T-0007a criterion 1: the two *real* service clients are an ``SttClient``/``TtsBackend``.

    The ``VllmClient`` precedent above, for the other two services. ``isinstance`` against a
    ``runtime_checkable`` Protocol is a *shallow* check — it sees method names, not parameters —
    so it is exactly the check that would have caught protocol drift (a renamed ``feed``, a
    ``speak`` that became ``say``) before T-0007 wired the services. Constructing either opens
    nothing: the moshi connection is lazy (criterion 1) and the TTS client holds no session.
    """
    stt = MoshiSttClient()
    tts = PocketTtsClient()
    assert isinstance(stt, SttClient)
    assert isinstance(tts, TtsBackend)
    assert stt.url.startswith("ws://127.0.0.1")  # docs/02 §9, loopback only
    assert tts.url.startswith("http://127.0.0.1")


PROMPT = Prompt(messages=(Message(role="user", content=(TextPart(text="hi"),)),), prefix_len=1)


def test_fakes_satisfy_the_runtime_checkable_protocols(clock):
    assert isinstance(FakeLlm(), LlmBackend)
    assert isinstance(FakeTts(), TtsBackend)
    assert isinstance(FakeAudioIO(clock), AudioIO)
    assert isinstance(FakeStt(), SttClient)
    assert isinstance(FakePerception(), PerceptionClient)
    assert isinstance(FakeMemory(), MemoryClient)


def test_the_real_llm_client_satisfies_the_llm_backend_protocol():
    """T-0017 criterion 3: the *real* client is an ``LlmBackend`` too.

    Nothing asserted this before; a protocol that drifted from ``VllmClient`` (a renamed method,
    a signature the Conductor calls with a keyword the client does not take) would have surfaced
    only when T-0007 wired the real service. ``isinstance`` against a ``runtime_checkable``
    Protocol is a *shallow* check — it sees method names, not parameters — so the signature-level
    conformance of both implementations lives in ``test_llm_backend_xargs.py``.
    """
    client = VllmClient()  # default loopback endpoint; no connection is opened by construction
    assert isinstance(client, LlmBackend)
    assert client.endpoint.base_url.startswith("http://127.0.0.1")  # docs/02 §9, loopback only


async def test_llm_streams_scripted_commits_and_abort_is_idempotent(backend):
    _real_not_wired(backend)
    script = [
        Commit(
            request_id="?", canvas_idx=0, segment="thought", text="let me see", thought_open=False
        ),
        Commit(request_id="?", canvas_idx=0, segment="answer", text="Four."),
    ]
    llm = FakeLlm(script)
    got = [c async for c in llm.generate(PROMPT, "r1", LlmOptions(top_k=40))]
    assert [c.segment for c in got] == ["thought", "answer"]
    assert all(c.request_id == "r1" for c in got)
    assert llm.generates[0][2] == LlmOptions(top_k=40)
    await llm.abort("r1")
    await llm.abort("r1")  # second call is a no-op, never an error
    assert llm.aborted == ["r1", "r1"] and "r1" not in llm.active


async def test_llm_abort_mid_stream_stops_further_commits(backend):
    _real_not_wired(backend)
    script = [
        Commit(request_id="?", canvas_idx=i, segment="answer", text=f"c{i}") for i in range(5)
    ]
    llm = FakeLlm(script)
    seen = []
    async for c in llm.generate(PROMPT, "r2"):
        seen.append(c.canvas_idx)
        if c.canvas_idx == 1:
            await llm.abort("r2")
    assert seen == [0, 1]


async def test_prefill_returns_a_handle_that_can_be_aborted(backend):
    _real_not_wired(backend)
    llm = FakeLlm(prefix_tokens=1234)
    h = await llm.prefill(PROMPT.prefix(), "p1")
    assert h.request_id == "p1" and h.prefix_tokens == 1234 and "p1" in llm.active
    await llm.abort("p1")
    assert "p1" not in llm.active


async def test_tts_utterance_streams_audio_words_and_cancels(backend):
    _real_not_wired(backend)
    tts = FakeTts(ms_per_word=300)
    u = await tts.speak("u1", "one two three")
    assert isinstance(u, TtsUtterance)
    chunks = [c async for c in u.audio()]
    assert sum(len(c) for c in chunks) == 900 * 48  # 900 ms of 24 kHz s16le mono
    words = u.words()
    assert words is not None and [w.text for w in words] == ["one", "two", "three"]
    assert words[2].start_ms == 600
    assert u.first_audio_latency_ms == 0.0
    u2 = await tts.speak("u2", "a b c d e f")
    await u2.cancel()
    assert [c async for c in u2.audio()] == []


async def test_audio_played_ms_follows_the_virtual_clock_through_pause_and_flush(clock, backend):
    _real_not_wired(backend)
    audio = FakeAudioIO(clock)
    u = await FakeTts(ms_per_word=500).speak("u", "one two three four")  # 2000 ms
    await audio.play(u)
    clock.advance(700)
    await audio.pause()
    clock.advance(1000)  # paused time does not count
    assert audio.played_ms() == 700
    await audio.resume()
    clock.advance(300)
    assert audio.played_ms() == 1000
    assert await audio.flush() == 1000
    assert audio.flushes == [1000]
    clock.advance(5000)
    assert audio.played_ms() == 0  # nothing playing


async def test_stt_words_have_audio_timestamps_and_events_replay(backend):
    _real_not_wired(backend)
    words = FakeStt.words("what is this", start_ms=1000)
    stt = FakeStt([*words, SttEndOfTurnEvent(t_eot_ms=1750)])
    events = [e async for e in stt.events()]
    assert [e.type for e in events] == ["word", "word", "word", "end_of_turn"]
    assert (events[0].start_ms, events[2].end_ms) == (1000, 1750)
    await stt.close()
    assert [e async for e in stt.events()] == []


def _frame(ts: int, source: Source = Source.SCREEN) -> Frame:
    return Frame(source=source, ts_ms=ts, width=2, height=2, jpeg=b"\xff\xd8", phash=f"{ts:016x}")


async def test_perception_returns_the_frame_at_speech_onset_not_the_latest(backend):
    _real_not_wired(backend)
    p = FakePerception([_frame(0), _frame(1000), _frame(2000), _frame(1500, Source.CAMERA)])
    f = await p.get_frame(Source.SCREEN, at_ms=1400)
    assert f is not None and f.ts_ms == 1000
    latest = await p.get_frame(Source.SCREEN)
    assert latest is not None and latest.ts_ms == 2000
    assert await p.get_frame(Source.WINDOW) is None
    clip = await p.clip(Source.SCREEN, 0, 2000, fps=1.0)
    assert [f.ts_ms for f in clip] == [0, 1000, 2000]


async def test_memory_search_filters_by_time_and_ranks_by_overlap_then_recency(backend):
    _real_not_wired(backend)
    m = FakeMemory()
    ids = [
        await m.ingest(
            Episode(ts_ms=t, source=Source.SCREEN, thumbnail_path=f"{t}.jpg", phash="0" * 16)
        )
        for t in (1000, 2000, 3000)
    ]
    await m.set_caption(ids[0], "budget spreadsheet totals")
    await m.set_caption(ids[1], "chapter on fractions")
    await m.set_caption(ids[2], "fractions worksheet totals")
    hits = await m.search(MemoryQuery(text="fractions totals", k=3))
    assert [h.episode.id for h in hits] == [
        ids[2],
        ids[1],
        ids[0],
    ]  # 2 matches first; ties by recency
    hits = await m.search(MemoryQuery(text="fractions", t0_ms=0, t1_ms=2500))
    assert [h.episode.id for h in hits] == [ids[1]]
    assert await m.forget_since(2000) == 2
    assert await m.search(MemoryQuery(text="fractions")) == []


def test_the_stt_protocol_declares_mark_and_both_clients_satisfy_it():
    """T-0032 criterion 7: ``SttClient`` grows ``mark``; the real client and the fake have it.

    ``MoshiSttClient.mark`` has existed since T-0007a (the ``Marker`` round trip) but no protocol
    declared it, so the session could not call it under ``mypy --strict`` and no fake was obliged
    to have it. The signature-level check is ``test_stt_protocol_static.py``; this is the runtime
    half that keeps ``isinstance(FakeStt(), SttClient)`` honest about the wider protocol.
    """
    assert hasattr(SttClient, "mark"), (
        "criterion 7: local_astra/contracts/protocols.py must add "
        "'async def mark(self, t_eot_ms: int) -> int' to SttClient (T-0007a ruling A3)"
    )
    for impl in (MoshiSttClient, FakeStt):
        assert hasattr(impl, "mark"), f"{impl.__name__} must implement SttClient.mark"
    assert isinstance(FakeStt(), SttClient)


# --- T-0088: the protocol widening (cd, tokenize) and the tools-on wire fake --------------------------


def test_the_llm_protocol_declares_tokenize_and_the_cd_keyword() -> None:
    """T-0088 criteria 3 and 4: ``LlmBackend`` gains ``tokenize`` and ``generate(..., *, cd=None)``; the fake and
    the real client both have them. Checked on the class dicts, not with ``hasattr`` on an instance."""
    import inspect

    assert hasattr(LlmBackend, "tokenize"), "criterion 4: LlmBackend.tokenize is not declared"
    cd = inspect.signature(LlmBackend.generate).parameters.get("cd")
    assert cd is not None and cd.kind is inspect.Parameter.KEYWORD_ONLY and cd.default is None, (
        "criterion 3: LlmBackend.generate(..., *, cd=None)"
    )
    for impl in (FakeLlm, VllmClient):
        assert "tokenize" in vars(impl), f"{impl.__name__} must implement tokenize"
        assert "cd" in inspect.signature(impl.generate).parameters, (
            f"{impl.__name__}.generate must take cd"
        )
    assert isinstance(FakeLlm(), LlmBackend)
    assert isinstance(VllmClient(), LlmBackend)


async def test_the_fake_yields_a_scripted_tool_commit_with_its_stop_id() -> None:
    """T-0088 (fake): a scriptable ``tool`` commit carries ``stop_id`` through, with the request id."""
    llm = FakeLlm(
        [
            Commit(request_id="?", canvas_idx=0, segment="answer", text="I will."),
            FakeLlm.tool_commit("<|tool_call>call:x{}<tool_call|>", stop_id=50),
        ]
    )
    got = [c async for c in llm.generate(PROMPT, "r3")]
    assert [(c.segment, c.request_id) for c in got] == [("answer", "r3"), ("tool", "r3")]
    assert got[1].stop_id == 50


async def test_the_fake_passes_the_tools_field_through_to_its_request_log() -> None:
    from tests.unit import _llm_id_support as S

    llm = FakeLlm([Commit(request_id="?", canvas_idx=0, segment="answer", text="ok")])
    async for _ in llm.generate(S.tools_prompt(), "r4"):
        pass
    async for _ in llm.generate(PROMPT, "r5"):
        pass
    assert llm.tools_log == [S.TOOLS, None]
    (body,) = llm.bodies
    assert body["tools"] == list(S.TOOLS) and body["tool_choice"] == "none"
    assert body == VllmClient().request_body(S.tools_prompt(), "r4"), (
        "the body comes from the real builder"
    )


def test_the_vllm_sse_fake_reproduces_every_recorded_stream_key_for_key() -> None:
    """The wire fake's conformance row: built from each recorded stream's ids (one canvas, as recorded), its
    chunks equal T-0087's recorded chunks key for key and in key order — ids, delta keys, ``finish_reason``,
    ``stop_reason``, ``system_fingerprint`` — except ``id``, ``created`` and the prompt fields."""
    import json

    from tests.fakes import vllm_sse as V
    from tests.unit import _llm_id_support as S

    def strip(chunk: dict) -> dict:
        return {
            k: v
            for k, v in chunk.items()
            if k not in ("id", "created", "prompt_token_ids", "prompt_text")
        }

    rows = S.recorded()
    assert len(rows) == 24
    for row in rows:
        rec = S.recorded_chunks(row)
        content_chunks = [c for c in rec if "token_ids" in c["choices"][0]]
        built = V.build_chunks(
            V.StreamSpec(chunks=[c["choices"][0]["token_ids"] for c in content_chunks]), S.detok()
        )
        assert [json.dumps(strip(c)) for c in built] == [json.dumps(strip(c)) for c in rec], (
            f"{row['arm']}/{row['case']}"
        )
        assert set(rec[0]) == set(built[0]), "the role chunk has the recorded top-level keys"


# --- T-0091: the ToolExecutor protocol, the recording fake and the dry-run executor ----------------------
#
# The "real" side of this contract is ``DryRunToolExecutor``: a pure CPU class with no service behind it, so both
# implementations run in the fast suite (no ``gpu`` arm). Every T-0091 name is reached inside the row through
# ``tests/unit/_toolcall_live_support.py``, so the rows are red on named assertions until the code exists.


def _tool_executor(impl: str):  # type: ignore[no-untyped-def]
    from tests.fakes.tool_executor import RecordingToolExecutor
    from tests.unit import _toolcall_live_support as TL

    return RecordingToolExecutor() if impl == "fake" else TL.dry_run()


@pytest.mark.parametrize("impl", ["fake", "dry_run"])
def test_both_tool_executors_satisfy_the_runtime_checkable_protocol(impl: str) -> None:
    """T-0091 criterion 7: ``ToolExecutor`` is ``runtime_checkable`` (``mode``, ``async execute``); the recording
    fake and ``DryRunToolExecutor`` are both instances, and both are in ``dry_run`` mode."""
    import inspect

    from tests.unit import _toolcall_live_support as TL

    proto = TL.protocol()
    ex = _tool_executor(impl)
    assert isinstance(ex, proto)
    assert inspect.iscoroutinefunction(ex.execute)
    assert ex.mode == "dry_run"


@pytest.mark.parametrize("impl", ["fake", "dry_run"])
async def test_both_tool_executors_answer_a_tool_outcome_for_the_same_call(impl: str) -> None:
    """T-0091 criterion 7, same assertions for both: ``execute`` returns a ``ToolOutcome`` with the invocation's
    ``call_id`` and a dry-run status; a framed ``scroll`` is ``dry_run``."""
    from tests.unit import _toolcall_live_support as TL

    outcome_cls = TL.need(TL.contracts(), "ToolOutcome", "criterion 7 (contracts)")
    ex = _tool_executor(impl)
    inv = TL.invocation("scroll", {"direction": "down"}, frame=TL.geometry(1280, 720), call_id="k1")
    out = await ex.execute(inv)
    assert isinstance(out, outcome_cls)
    assert (out.call_id, out.status) == ("k1", "dry_run")


async def test_the_recording_fake_records_and_replays_a_scripted_outcome() -> None:
    from tests.fakes.tool_executor import RecordingToolExecutor
    from tests.unit import _toolcall_live_support as TL

    outcome_cls = TL.need(TL.contracts(), "ToolOutcome", "criterion 7 (contracts)")
    scripted = outcome_cls(
        call_id="k2", status="refused", action="", reason="no frame", resolved={}
    )
    ex = RecordingToolExecutor({"k2": scripted})
    a = TL.invocation("click_at", {"x": 1, "y": 2}, call_id="k2")
    b = TL.invocation("scroll", {"direction": "up"}, call_id="k3")
    assert await ex.execute(a) == scripted
    assert (await ex.execute(b)).status == "dry_run"
    assert ex.invocations == [a, b]
