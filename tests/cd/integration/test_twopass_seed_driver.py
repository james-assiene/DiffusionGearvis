"""T-0081b criteria 1, 3-6, 9 and 10: the ``perstep_mar+seed`` pass-2 cell of the two-pass driver against the loopback
fake vLLM server (CPU).

The fake is ``tests/cd/fakes/vllm_server.py`` (T-0081b: ``return_token_ids`` honours ``max_tokens``, and
``Reply.canvas_steps`` scripts a step counter per canvas record; ``Reply.error`` scripts a record's reason code).
Seeds are read off the wire with ``FakeVllmServer.cd_seeds`` semantics (standard base64 of little-endian int32 ids)
and compared with the seed the task specifies, built here per case by hand::

    seed(thought) = (H(thought) + encode(T(thought) + draft_call_text) + S)[:256]

with ``encode`` the pinned tokenizer, H(T1) pass 1's ids through the first ``101`` (or the empty header
``100 45518 107 101`` when pass 1 has no thought), H(T2) the empty header, H(T4) nothing, T = ``"<cc>"`` for T1/T2 and
``""`` for T4, and S the first stop id of pass 1 (the fake ends every finished generation with EOS = 1).

The seeded run (one per module) scripts the dev split's first 13 cases (search-01..10, click-01..03) under the six
cells ``T1|T2|T4 / named / perstep_mar`` and ``... / perstep_mar+seed`` at the frozen ``calibration-bfcl.json``
knobs. Every case answers with its exact call after ``<cc>`` and a closed thought, except:

- ``search-02``: no thought at all (T1's H falls back to the empty header, ``seed_source`` ``t1_empty_header``);
- ``search-03``: an unknown argument (``lang``); ``click-01``: ``x`` 1500 (both parse, both out of schema);
- ``search-04``: keys without spaces and prose after the call (``... Done.``);
- ``search-05``: prose after ``<cc>``; ``search-06``: the native ``<|tool_call>`` tag (both: ``draft_name`` None, the
  name picker, two requests, no seed);
- ``search-07``: its T1 outputs (seeded and not) commit two canvases with step counters 30 and 12;
- ``click-02``: a thought so long that pass 1 runs out of its 1024-token budget inside the call (no stop id: the seed
  has no S; its T1 seed is pass 1's first 256 ids).

Hence 13 routed cases, 2 through the picker, 11 seeded per seed cell. Pass-2 requests: 11 x 6 + 2 x 2 x 6 = 90, plus
13 pass-1 requests. A seeded request's (default) step counter is 27, so its executed steps are 27 - 24 = 3; an
unseeded one's is 7.

Regime: CPU; fake server; pinned tokenizer (model rev 2ea8372); ``tools_v1.json`` pinned; JSON whitespace; CL 256;
seed constants noise 0.0 / start step 24 (T-0081 tier 2, dad1382, ``mar_n0_s24``).
"""

from __future__ import annotations

import base64
import collections
import dataclasses
import functools
import json
import socket
import struct
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _twopass_support as ts
from tests.cd.fakes.vllm_server import EOS, Reply, cd_keys

SEED = "perstep_mar+seed"
BASE = "perstep_mar"
BASE_CELLS = tuple(f"{t}/named/{BASE}" for t in ts.THOUGHTS)
SEED_CELLS = tuple(f"{t}/named/{SEED}" for t in ts.THOUGHTS)
CELLS = BASE_CELLS + SEED_CELLS
K0 = 24
NOISE = 0.0
SEED_STEPS = 27  # a seeded request's step counter: 3 executed after k0 24
CL = 256
HEADER = [100, 45518, 107, 101]
DRAFT_KEYS = ("astra_cd_draft", "astra_cd_draft_noise", "astra_cd_draft_start_step")
TAG_KEY = "astra_cd_record_tag"
LIMIT = 13
CASES = [c["id"] for c in ts.split("dev")][:LIMIT]
PICKER = {"search-05": "google_search", "search-06": "google_search"}
NO_THOUGHT = "search-02"
TWO_CANVAS = "search-07"
TRUNCATED = "click-02"
SEEDED_CASES = [c for c in CASES if c not in PICKER]
OUT_OF_SCHEMA = {
    "search-03": '{"name": "google_search", "arguments": {"query": "framework laptop 16", "lang": "en"}}',
    "click-01": '{"name": "click_at", "arguments": {"x": 1500, "y": 499}}',
}
RAW = {"search-04": '{"name":"google_search","arguments":{"query":"banana bread egg"}} Done.'}
SEED_SOURCES = {"T2": "t2_header", "T4": "t4_call"}


# --------------------------------------------------------------------------- the scripted model


def encode(text: str) -> list[int]:
    from tests.cd import _compile_support as cs

    return [int(t) for t in cs.encode(text)]


def _thought(case: str) -> str:
    return f"<|channel>thought\nThe user asked for {case}.<channel|>"


@functools.cache
def truncated_text(case: str = TRUNCATED) -> str:
    """``case``'s pass 1 (click-02 by default): a long thought, then the call, so that its first 1024 ids end inside
    the call (after the tool name). Built by growing the thought until the call starts 4..12 ids before the budget."""
    name, args = ts.gt_call(ts.by_id()[case])
    call = ts.call_text(name, args)
    unit = "Let me look at the screen again and find the right link. "
    for n in range(1, 400):
        head = f"<|channel>thought\n{unit * n}<channel|>Clicking it. <cc>"
        start = len(encode(head))
        if 1024 - 12 <= start <= 1024 - 4:
            text = head + call
            assert len(encode(text)) > 1024 + 2, "premise: the call runs past the budget"
            return text
        if start > 1024:
            break
    raise AssertionError("could not place the budget inside the call")


def pass1_texts() -> dict[str, str]:
    def gt(case: str) -> str:
        return ts.call_text(*ts.gt_call(ts.by_id()[case]))

    out = {
        NO_THOUGHT: f"On it ({NO_THOUGHT}). <cc>" + gt(NO_THOUGHT),
        "search-05": _thought("search-05") + "Searching now. <cc>I will search for it.",
        "search-06": _thought("search-06")
        + "Searching now. <|tool_call>"
        + '{"name": "google_search", "arguments": {"query": "artemis"}}',
        TRUNCATED: truncated_text(),
    }
    for case, draft in {**OUT_OF_SCHEMA, **RAW}.items():
        out[case] = _thought(case) + ts.Scenario.spoken(case) + "<cc>" + draft
    return out


@dataclasses.dataclass
class SeedScenario(ts.Scenario):
    """``ts.Scenario`` plus overrides keyed ``(case, shape, seeded)`` (a seeded and an unseeded cell share a shape),
    and a seeded request's default step counter ``SEED_STEPS``."""

    seed_override: dict[tuple[str, str, bool], Reply] = dataclasses.field(default_factory=dict)

    def __call__(self, index: int, body: Mapping[str, Any]) -> Reply:
        keys = cd_keys(body)
        if not keys:
            return super().__call__(index, body)
        seeded = "astra_cd_draft" in keys
        case = ts.case_of(body)
        shape = ts.thought_of_body(body)
        with self._lock:
            ov = self.seed_override.pop((case or "", shape, seeded), None)
        if ov is not None:
            scope, thought = ts.payload_index().get(str(keys.get("astra_cd_automaton")), ("?", "?"))
            with self._lock:
                self.sent.append(ts.Sent(index, "p2", case, scope, thought, shape, body, ov))
            return ov
        reply = super().__call__(index, body)
        if seeded and reply.steps == ts.P2_STEPS:
            reply.steps = SEED_STEPS
        return reply


def p2_text(case: str, scope: str, thought: str) -> str:
    return ts.Scenario().pass2_text(case, scope, thought)


def seeded_scenario() -> SeedScenario:
    sc = SeedScenario(pass1=pass1_texts())
    t1 = p2_text(TWO_CANVAS, "google_search", "T1")
    sc.seed_override[(TWO_CANVAS, "T1", True)] = Reply(text=t1, canvases=2, canvas_steps=[30, 12])
    sc.seed_override[(TWO_CANVAS, "T1", False)] = Reply(text=t1, canvases=2, canvas_steps=[30, 12])
    return sc


# --------------------------------------------------------------------------- expected values (by hand, per case)


def served_pass1(case: str) -> list[int]:
    """What the fake serves as pass 1's ``token_ids``: the text's ids then EOS, cut to the 1024 budget."""
    text = pass1_texts().get(case) or ts.Scenario().pass1_text(case)
    return ([*encode(text), EOS])[:1024]


def draft_text(case: str) -> str:
    """The router's draft (text after the tag) of pass 1's host-decoded answer (up to the first stop)."""
    from local_astra.eval.cd import replay
    from local_astra.toolcall import router

    ids = served_pass1(case)
    head = ids[: ids.index(EOS)] if EOS in ids else ids
    return router.route(replay.detokenize(head)).draft_call_text


def expected_seed(case: str, thought: str) -> list[int]:
    ids = served_pass1(case)
    s = [EOS] if EOS in ids else []
    d = draft_text(case).rstrip()  # R2: trailing whitespace trimmed
    if thought == "T4":
        return (encode(d) + s)[:CL]
    if thought == "T2" or case == NO_THOUGHT:
        return (HEADER + encode("<cc>" + d) + s)[:CL]
    return (ids[: ids.index(101) + 1] + encode("<cc>" + d) + s)[:CL]


def expected_source(case: str, thought: str) -> str:
    if thought == "T1":
        return "t1_empty_header" if case == NO_THOUGHT else "t1_thought"
    return SEED_SOURCES[thought]


def wire(ids: Sequence[int]) -> str:
    """The documented wire format of ``astra_cd_draft`` (T-0081 criterion 1): standard base64 of little-endian int32."""
    return base64.b64encode(struct.pack(f"<{len(ids)}i", *ids)).decode("ascii")


def unwire(raw: str) -> list[int]:
    data = base64.b64decode(raw, validate=True)
    return [v for (v,) in struct.iter_unpack("<i", data)]


# --------------------------------------------------------------------------- running


def arm_of(s: ts.Sent) -> str:
    keys = cd_keys(s.body)
    return SEED if "astra_cd_draft" in keys else str(keys.get("astra_cd_mode"))


def cell_of(s: ts.Sent) -> str:
    return f"{s.shape}/named/{arm_of(s)}"


class Run:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.result: Any = None
        self.scenario: SeedScenario | None = None
        self.srv: Any = None
        self.connects: list[Any] = []
        self._by_tag: dict[str, str] | None = None

    def get(self) -> Run:
        if self.error is not None:
            raise self.error
        return self

    @property
    def sent(self) -> list[ts.Sent]:
        assert self.scenario is not None
        return sorted(self.scenario.sent, key=lambda s: s.index)

    def p1(self, case: str) -> ts.Sent:
        got = [s for s in self.sent if s.kind == "p1" and s.case == case]
        assert len(got) == 1, (case, len(got))
        return got[0]

    def cell(self, s: ts.Sent) -> str:
        """The cell a request was sent for: by its record tag, through the rows' ``record_tags`` (a picker request
        of the seed cell carries no draft key, so its keys alone cannot tell it from the non-seed cell's)."""
        if self._by_tag is None:
            self._by_tag = {t: r["cell"] for r in self.rows(2) for t in r.get("record_tags") or []}
        by_tag = self._by_tag
        tag = cd_keys(s.body).get(TAG_KEY)
        return str(by_tag[tag]) if tag in by_tag else cell_of(s)

    def p2(self, case: str, cell: str | None = None) -> list[ts.Sent]:
        return [
            s
            for s in self.sent
            if s.kind == "p2" and s.case == case and (cell is None or self.cell(s) == cell)
        ]

    def rows(self, which: int) -> list[dict[str, Any]]:
        return ts.pass_rows(self.result.artifact, which)

    def row(self, case: str, cell: str) -> dict[str, Any]:
        got = [r for r in self.rows(2) if r["case"] == case and r["cell"] == cell]
        assert len(got) == 1, (case, cell, len(got))
        return got[0]


def _execute(
    tmp: Path, scenario: SeedScenario, cells: Sequence[str], limit: int, **cfg: Any
) -> Run:
    run = Run()
    run.scenario = scenario
    real_connect = socket.socket.connect

    def recording_connect(sock: socket.socket, address: Any) -> Any:
        run.connects.append(address)
        return real_connect(sock, address)

    try:
        ts.twopass()
    except AssertionError as exc:
        run.error = exc
        return run
    with ts.fake(scenario, tmp) as srv:
        run.srv = srv
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", recording_connect)
            try:
                run.result = ts.run_driver(srv, tmp, cells=list(cells), limit=limit, **cfg)
            except BaseException as exc:  # surfaced in every row that reads the run
                run.error = exc
    return run


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    yield _execute(tmp_path_factory.mktemp("t0081b_seeded"), seeded_scenario(), CELLS, LIMIT)


@pytest.fixture(scope="module")
def unseeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """The same scripted scenario under the three non-seed cells only (criterion 10's "without the seed cell")."""
    yield _execute(tmp_path_factory.mktemp("t0081b_unseeded"), seeded_scenario(), BASE_CELLS, LIMIT)


# =========================================================================== the seeded run


def test_the_seeded_run_completes_with_exit_0(seeded: Run) -> None:
    r = seeded.get()
    assert r.result.exit_code == 0, (r.result.stopped, ts.stopped_marker(r.result.artifact))
    assert r.srv.rejected == []
    kinds = collections.Counter(s.kind for s in r.sent)
    assert kinds == {"p1": LIMIT, "p2": 90}, kinds
    rows = r.rows(2)
    assert collections.Counter((x["case"], x["cell"]) for x in rows) == collections.Counter(
        (c, cell) for c in CASES for cell in CELLS
    )
    for x in rows:
        assert x["error"] is None and x["cs"] is True, (x["case"], x["cell"], x["error"])


def test_the_truncated_pass_1_premise(seeded: Run) -> None:
    """Fixture premise for the no-S rows: click-02's pass 1 is served as exactly 1024 ids with no stop id, and its
    draft (cut inside the call) still names ``click_at`` -- routed, not picked."""
    r = seeded.get()
    p1 = {x["case"]: x for x in r.rows(1)}[TRUNCATED]
    assert len(p1["token_ids"]) == 1024 and not {1, 50, 106} & set(p1["token_ids"])
    assert p1["route"]["draft_name"] == "click_at", p1["route"]
    assert p1["draft_call_text"] == draft_text(TRUNCATED)


# --------------------------------------------------------------------------- criterion 1: keys


@pytest.mark.parametrize("thought", ts.THOUGHTS)
def test_seed_cell_keys_are_the_base_cells_plus_the_three_draft_keys(
    seeded: Run, thought: str
) -> None:
    """Criterion 1: per seeded case, the seed cell's CD keys equal ``perstep_mar``'s for the same thought -- the same
    automaton, mode, sha and frozen knobs; only the record tag differs -- plus ``astra_cd_draft``,
    ``astra_cd_draft_noise`` = 0.0 and ``astra_cd_draft_start_step`` = 24. The base cell sends no draft key."""
    r = seeded.get()
    for case in SEEDED_CASES:
        (base,) = r.p2(case, f"{thought}/named/{BASE}")
        (seed,) = r.p2(case, f"{thought}/named/{SEED}")
        bk, sk = cd_keys(base.body), cd_keys(seed.body)
        assert not set(DRAFT_KEYS) & set(bk), (case, sorted(bk))
        assert set(DRAFT_KEYS) <= set(sk), (case, sorted(sk))
        strip = {TAG_KEY, *DRAFT_KEYS}
        assert {k: v for k, v in sk.items() if k not in strip} == {
            k: v for k, v in bk.items() if k != TAG_KEY
        }, case
        assert sk["astra_cd_mode"] == BASE
        assert sk[TAG_KEY] != bk[TAG_KEY]
        assert sk["astra_cd_draft_noise"] == NOISE and isinstance(sk["astra_cd_draft_noise"], float)
        start = sk["astra_cd_draft_start_step"]
        assert start == K0 and isinstance(start, int) and not isinstance(start, bool), start


def test_the_knobs_are_perstep_mars_frozen_file(seeded: Run) -> None:
    """Criterion 1: the seed cell's knobs come from ``load_frozen(calibration-bfcl.json, task="bfcl",
    arm="perstep_mar")``."""
    from local_astra.eval.cd import calibration

    values, _sha = calibration.load_frozen(ts.CALIBRATION[BASE], task="bfcl", arm=BASE)
    r = seeded.get()
    sent = [s for s in r.sent if s.kind == "p2" and arm_of(s) == SEED]
    assert sent
    for s in sent:
        keys = cd_keys(s.body)
        for k, v in values.items():
            assert float(keys[f"astra_cd_{k}"]) == float(v), (s.case, k)


@pytest.mark.parametrize("thought", ts.THOUGHTS)
def test_the_draft_on_the_wire_is_the_specified_seed(seeded: Run, thought: str) -> None:
    """Criteria 1-2 end to end: the ``astra_cd_draft`` of every seeded request is the canonical wire form (standard
    base64, little-endian int32) of exactly the specified seed for its case and thought -- including H(T1) from pass
    1 (search-02: the empty-header fallback), raw out-of-schema drafts, and click-02's truncated pass 1."""
    r = seeded.get()
    for case in SEEDED_CASES:
        (seed,) = r.p2(case, f"{thought}/named/{SEED}")
        raw = cd_keys(seed.body)["astra_cd_draft"]
        want = expected_seed(case, thought)
        assert unwire(raw) == want, (case, thought, unwire(raw)[:12], want[:12])
        assert raw == wire(want), (case, thought)


def test_the_regime_line_records_the_seed_constants(seeded: Run) -> None:
    """Criterion 1: noise 0.0 and start step 24 are recorded in the regime line (key names are the implementer's;
    the row looks for one mapping with a ``*noise*`` key at 0.0 and a ``*start*step*`` key at 24)."""
    r = seeded.get()
    regime = ts.artifact_lines(r.result.artifact)[0]

    def found(obj: Any) -> bool:
        if isinstance(obj, Mapping):
            noise = [v for k, v in obj.items() if "noise" in str(k).lower()]
            start = [
                v for k, v in obj.items() if "start" in str(k).lower() and "step" in str(k).lower()
            ]
            hit = any(isinstance(v, float) and v == NOISE for v in noise) and any(
                isinstance(v, int) and not isinstance(v, bool) and v == K0 for v in start
            )
            return hit or any(found(v) for v in obj.values())
        if isinstance(obj, list):
            return any(found(v) for v in obj)
        return False

    assert found(regime), json.dumps(regime)[:2000]


# --------------------------------------------------------------------------- criterion 3: rows and the picker


def test_picker_cases_send_no_draft_in_either_request(seeded: Run) -> None:
    """Criterion 3: ``draft_name`` None (prose after the tag; the native ``<|tool_call>`` tag) -> the name picker,
    two requests per seed cell, and neither carries any ``astra_cd_draft*`` key; the row records ``seed`` null and
    ``seed_reason`` ``"picker"``."""
    r = seeded.get()
    for case, tool in PICKER.items():
        p1 = {x["case"]: x for x in r.rows(1)}[case]
        assert p1["route"]["kind"] == "toolcall" and p1["route"]["draft_name"] is None, p1["route"]
        for thought in ts.THOUGHTS:
            sent = [s for s in r.p2(case) if s.shape == thought]
            assert [s.scope for s in sent] == ["name", tool] * 2, (case, thought)
            for s in sent:
                assert not [k for k in cd_keys(s.body) if k.startswith("astra_cd_draft")], (
                    case,
                    thought,
                    sorted(cd_keys(s.body)),
                )
            row = r.row(case, f"{thought}/named/{SEED}")
            assert row["requests"] == 2
            assert "seed" in row and row["seed"] is None, row.get("seed")
            assert row.get("seed_reason") == "picker", row.get("seed_reason")


@pytest.mark.parametrize("thought", ts.THOUGHTS)
def test_every_routed_case_with_a_draft_name_is_seeded_and_its_row_says_how(
    seeded: Run, thought: str
) -> None:
    """Criterion 3: every routed case with a v1 ``draft_name`` is seeded (11 of 13 here, schema-valid or not,
    truncated or not), in one request; its row records ``seed_ids`` (the ids sent), ``seed_len``, ``seed_source``
    (``t1_thought`` / ``t1_empty_header`` / ``t2_header`` / ``t4_call``), ``seed_noise`` 0.0 and ``seed_start_step``
    24."""
    r = seeded.get()
    cell = f"{thought}/named/{SEED}"
    for case in SEEDED_CASES:
        sent = r.p2(case, cell)
        assert len(sent) == 1, (case, [s.scope for s in sent])
        row = r.row(case, cell)
        want = expected_seed(case, thought)
        assert row.get("seed_ids") == want, (case, thought)
        assert row.get("seed_ids") == unwire(cd_keys(sent[0].body)["astra_cd_draft"])
        assert row.get("seed_len") == len(want), (case, row.get("seed_len"))
        assert row.get("seed_source") == expected_source(case, thought), (
            case,
            row.get("seed_source"),
        )
        assert row.get("seed_noise") == NOISE and row.get("seed_start_step") == K0, case


def test_the_draft_is_stored_once_as_ids_not_as_base64(seeded: Run) -> None:
    """Implementation note (``arms.row_knobs`` would copy the base64 draft into ``knobs``): the wire string of a
    seeded request appears nowhere in its row; the ids are stored once, as ``seed_ids``."""
    r = seeded.get()
    for case in SEEDED_CASES:
        for thought in ts.THOUGHTS:
            (s,) = r.p2(case, f"{thought}/named/{SEED}")
            raw = cd_keys(s.body)["astra_cd_draft"]
            assert raw not in json.dumps(r.row(case, f"{thought}/named/{SEED}")), (case, thought)


# --------------------------------------------------------------------------- criterion 4: no host repair


@pytest.mark.parametrize("case", [*OUT_OF_SCHEMA, *RAW])
def test_out_of_schema_and_odd_drafts_are_sent_raw(seeded: Run, case: str) -> None:
    """Criterion 4 (negative): a draft that parses but is out of schema (x 1500; an unknown argument), and one with
    keys unspaced and prose after it, are sent as ``encode`` of their raw text -- T4 exactly ``encode(draft) + [EOS]``
    -- never clamped, re-ordered, re-serialised or trimmed."""
    r = seeded.get()
    draft = {**OUT_OF_SCHEMA, **RAW}[case]
    assert draft_text(case) == draft, "premise: the router's draft is the scripted text"
    p1 = {x["case"]: x for x in r.rows(1)}[case]
    assert p1["draft_schema_valid"] is False, p1.get("draft_parse")
    (t4,) = r.p2(case, f"T4/named/{SEED}")
    assert unwire(cd_keys(t4.body)["astra_cd_draft"]) == [*encode(draft), EOS]
    (t2,) = r.p2(case, f"T2/named/{SEED}")
    assert unwire(cd_keys(t2.body)["astra_cd_draft"]) == HEADER + encode("<cc>" + draft) + [EOS]


def test_a_truncated_draft_is_sent_without_s_and_never_completed(seeded: Run) -> None:
    """Criterion 4: click-02's pass 1 ran out of budget inside the call. Its T4 seed is ``encode`` of the truncated
    draft and nothing else (no stop id, nothing closed); T2 the header + ``encode("<cc>" + draft)``; T1 is pass 1's
    first 256 ids verbatim (its thought is longer than the canvas)."""
    r = seeded.get()
    draft = draft_text(TRUNCATED).rstrip()  # R2
    seeds = {
        t: unwire(cd_keys(r.p2(TRUNCATED, f"{t}/named/{SEED}")[0].body)["astra_cd_draft"])
        for t in ts.THOUGHTS
    }
    assert seeds["T4"] == encode(draft), seeds["T4"]
    assert seeds["T2"] == HEADER + encode("<cc>" + draft)
    assert seeds["T1"] == served_pass1(TRUNCATED)[:CL]
    for t, ids in seeds.items():
        assert not {1, 50, 106} & set(ids), t


# --------------------------------------------------------------------------- criterion 5: step accounting


def test_seeded_rows_report_steps_after_the_start_step_and_unseeded_rows_unchanged(
    seeded: Run,
) -> None:
    """Criterion 5: a seeded request's counter 27 is 3 executed steps (k0 24) in ``steps`` and ``steps_per_request``;
    an unseeded request's 7 stays 7 (k0 0)."""
    r = seeded.get()
    for case in SEEDED_CASES:
        for thought in ts.THOUGHTS:
            if (case, thought) == (TWO_CANVAS, "T1"):
                continue
            s = r.row(case, f"{thought}/named/{SEED}")
            b = r.row(case, f"{thought}/named/{BASE}")
            assert s["steps"] == [SEED_STEPS - K0] and s["steps_per_request"] == [[3]], (
                case,
                thought,
                s["steps"],
            )
            assert b["steps"] == [ts.P2_STEPS] and b["steps_per_request"] == [[ts.P2_STEPS]], (
                case,
                thought,
                b["steps"],
            )


def test_a_two_canvas_seeded_row_subtracts_k0_from_the_first_canvas_only(seeded: Run) -> None:
    """Criterion 5, the ``executed_steps`` fix: search-07's T1 commits two canvases with counters 30 and 12. Seeded:
    ``[6, 12]`` (only the first canvas was seeded; subtracting from both would give ``[6, -12]``). Unseeded:
    ``[30, 12]``. The second canvas's counter 12 <= 24 is no ``seed_accounting`` stop (canvas 0 only)."""
    r = seeded.get()
    s = r.row(TWO_CANVAS, f"T1/named/{SEED}")
    b = r.row(TWO_CANVAS, f"T1/named/{BASE}")
    assert s["steps"] == [6, 12] and s["steps_per_request"] == [[6, 12]], s["steps"]
    assert b["steps"] == [30, 12] and b["steps_per_request"] == [[30, 12]], b["steps"]


# --------------------------------------------------------------------------- criterion 10: bit-identity and scope


def _norm_body(body: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(body))
    (out.get("vllm_xargs") or {}).pop(TAG_KEY, None)
    return dict(out)


VOLATILE_ROW_KEYS = {
    "wall_ms",
    "wall_ms_per_request",
    "prefix_hit_tokens",
    "prefix_query_tokens",
    "record_tags",
    "flush_requests",
}


def _norm_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in VOLATILE_ROW_KEYS and not k.startswith("seed")}
    knobs = dict(out.get("knobs") or {})
    knobs.pop("record_tag", None)
    out["knobs"] = knobs
    return out


def test_non_seed_cells_are_identical_with_and_without_the_seed_cells(
    seeded: Run, unseeded: Run
) -> None:
    """Criterion 10 (negative): on the same scripted scenario, adding the three seed cells leaves every non-seed
    request byte-identical (apart from the record tag, which carries the run id and the cell index), every pass-1
    body identical, the non-seed request count identical (45), and every non-seed artifact row identical apart from
    per-request timing, prefix-cache deltas, tags and fields this task adds (``seed*``)."""
    a, b = unseeded.get(), seeded.get()
    assert a.result.exit_code == 0, a.result.stopped
    for case in CASES:
        assert a.p1(case).body == b.p1(case).body, case
        for cell in BASE_CELLS:
            sa = [_norm_body(s.body) for s in a.p2(case, cell)]
            sb = [_norm_body(s.body) for s in b.p2(case, cell)]
            assert sa and sa == sb, (case, cell)
            assert _norm_row(a.row(case, cell)) == _norm_row(b.row(case, cell)), (case, cell)
    na = sum(1 for s in a.sent if s.kind == "p2")
    nb = sum(1 for s in b.sent if s.kind == "p2" and b.cell(s) in BASE_CELLS)
    assert na == nb == 45, (na, nb)
    assert not [s for s in a.sent if s.kind == "p2" and set(DRAFT_KEYS) & set(cd_keys(s.body))]


def test_seeded_bodies_keep_the_thought_shapes(seeded: Run) -> None:
    """Criterion 10: T1/T2 seeded bodies still equal pass 1's except ``vllm_xargs`` and ``max_tokens`` (1024 / 256);
    a T4 seeded body equals the T4 non-seed body except ``vllm_xargs`` (the prompt ends at ``<cc>``)."""
    r = seeded.get()
    for case in SEEDED_CASES:
        p1 = r.p1(case).body
        for thought in ts.THOUGHTS:
            (s,) = r.p2(case, f"{thought}/named/{SEED}")
            if thought == "T4":
                (base,) = r.p2(case, f"T4/named/{BASE}")
                strip = ("vllm_xargs",)
                assert {k: v for k, v in s.body.items() if k not in strip} == {
                    k: v for k, v in base.body.items() if k not in strip
                }, case
                continue
            assert {k: v for k, v in s.body.items() if k not in ("vllm_xargs", "max_tokens")} == {
                k: v for k, v in p1.items() if k != "max_tokens"
            }, (case, thought)
            assert s.body["max_tokens"] == (1024 if thought == "T1" else 256)


def test_every_connection_stays_on_loopback(seeded: Run) -> None:
    r = seeded.get()
    assert r.connects, "fixture: no connection was recorded"
    assert {a[0] for a in r.connects if isinstance(a, tuple)} <= {"127.0.0.1", "::1", "localhost"}


#: T-0081b's merge commit 0dc9134; BASE = ``git merge-base 0dc9134^1 0dc9134^2``, TIP = ``0dc9134^2`` (T-0094
#: criterion 2). Over this closed range no later-task exemption is needed (T-0082a, T-0084a/b, T-0086 came later).
T0081B_MERGE = "0dc9134"
T0081B_BASE = "2269299"
T0081B_TIP = "a33eb01"
UNCHANGED = [
    *(
        f"local_astra/eval/cd/{m}.py"
        for m in ("runner", "arms", "calibration", "cuts", "hardstop", "replay", "records")
    ),
    "local_astra/contracts",
    "local_astra/decoding",
    "local_astra/toolcall",
    "plugin",
    "scripts/locked/t0080_toolcall.sh",
    "local_astra/bench/thresholds.py",
]


def test_files_outside_the_task_are_unchanged() -> None:
    """Criterion 10 (negative, scope) over T-0081b's closed range (T-0094): nothing under the paths the task leaves
    alone differs between BASE and TIP. The untracked-file half of the old guard is retired: it only meant something
    while the branch was open."""
    gr.prove_range(T0081B_BASE, T0081B_TIP, T0081B_MERGE)
    changed = gr.changed(T0081B_BASE, T0081B_TIP, *UNCHANGED)
    assert changed == [], changed


def test_the_t0081b_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the predicate reports ``contracts/llm.py`` (T-0088)."""
    gr.require_commits(T0081B_BASE, gr.LATER)
    assert "local_astra/contracts/llm.py" in gr.changed(T0081B_BASE, gr.LATER, *UNCHANGED)


# =========================================================================== refusals (criterion 1)


def _refused(tmp_path: Path, **overrides: Any) -> None:
    """``run`` raises ``ValueError`` before any socket opens, and writes no artifact."""
    tp = ts.twopass()
    repo, head = ts.es.clean_repo(tmp_path)
    out = tmp_path / "refused-artifacts"
    cfg: dict[str, Any] = {
        "stage": "dev",
        "cells": [f"T4/named/{SEED}"],
        "base_url": "http://127.0.0.1:9/v1",
        "engine_log": tmp_path / "engine.log",
        "out_dir": out,
        "repo": repo,
        "expected_commit": head,
        "calibration": dict(ts.CALIBRATION),
        "set_dir": ts.SET_DIR,
        "limit": 1,
    }
    cfg.update(overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", lambda *a: pytest.fail(f"socket opened: {a}"))
        with pytest.raises(ValueError):
            tp.run(tp.TwoPassConfig(**cfg))
    assert not (out.exists() and list(out.glob("*.jsonl"))), "a refused run wrote an artifact"


def _accepted(tmp_path: Path, cells: Sequence[str], **cfg: Any) -> Run:
    """The positive control of a refusal row: the neighbouring configuration runs (1 case) to exit 0."""
    run = _execute(tmp_path / "accepted", SeedScenario(), cells, 1, **cfg).get()
    assert run.result.exit_code == 0, run.result.stopped
    return run


@pytest.mark.parametrize("thought", ts.THOUGHTS)
def test_a_union_seed_cell_is_refused_before_any_socket(tmp_path: Path, thought: str) -> None:
    """Criterion 1: ``<T>/union/perstep_mar+seed`` raises ``ValueError`` before any socket. Control (so the row cannot
    pass by refusing the seed arm everywhere): ``<T>/named/perstep_mar+seed`` runs and is seeded."""
    (tmp_path / "accepted").mkdir()
    run = _accepted(tmp_path, [f"{thought}/named/{SEED}"])
    assert [arm_of(s) for s in run.sent if s.kind == "p2"] == [SEED]
    _refused(tmp_path, cells=[f"{thought}/union/{SEED}"])


def test_a_calibration_entry_keyed_by_the_seed_arm_is_refused(tmp_path: Path) -> None:
    """Criterion 1: the seed arm has no calibration of its own; an entry keyed ``perstep_mar+seed`` is refused, even
    beside the correct ``perstep_mar`` entry. Control: the same cell with only the ``perstep_mar`` entry runs."""
    (tmp_path / "accepted").mkdir()
    _accepted(tmp_path, [f"T4/named/{SEED}"], calibration={BASE: ts.CALIBRATION[BASE]})
    _refused(tmp_path, calibration={BASE: ts.CALIBRATION[BASE], SEED: ts.CALIBRATION[BASE]})


@pytest.mark.parametrize("which", ["qargmax_file", "missing"])
def test_the_seed_cell_needs_perstep_mars_own_frozen_file(tmp_path: Path, which: str) -> None:
    """Criterion 1: the knobs come from ``load_frozen(<perstep_mar file>, arm="perstep_mar")``: another arm's frozen
    file under ``perstep_mar`` is refused, and so is no ``perstep_mar`` file at all (the seed arm has no other
    source; here the qargmax file is offered under its own key only). Control: the right file runs."""
    (tmp_path / "accepted").mkdir()
    _accepted(tmp_path, [f"T4/named/{SEED}"], calibration={BASE: ts.CALIBRATION[BASE]})
    calib = (
        {BASE: ts.QARGMAX_FILE}
        if which == "qargmax_file"
        else {"perstep_mar_qargmax": ts.QARGMAX_FILE}
    )
    _refused(tmp_path, calibration=calib)


def test_the_named_seed_cell_is_accepted_on_its_own(tmp_path: Path) -> None:
    """The other side of the refusals (a guard too wide): ``T4/named/perstep_mar+seed`` alone, with only the
    ``perstep_mar`` file, runs to exit 0 (2 cases: 2 + 2 requests, both seeded)."""
    sc = SeedScenario()
    run = _execute(tmp_path, sc, [f"T4/named/{SEED}"], 2).get()
    assert run.result.exit_code == 0, run.result.stopped
    p2 = [s for s in run.sent if s.kind == "p2"]
    assert len(p2) == 2 and all(arm_of(s) == SEED for s in p2)


# =========================================================================== stops (criterion 6)

SMALL_CELLS = [f"T4/named/{BASE}", f"T4/named/{SEED}"]


def _small(tmp_path: Path, sc: SeedScenario, limit: int = 2, **cfg: Any) -> Run:
    return _execute(tmp_path, sc, SMALL_CELLS, limit, **cfg).get()


def _trailer(run: Run) -> Mapping[str, Any]:
    lines = ts.artifact_lines(run.result.artifact)
    trailers = [x for x in lines if x.get("kind") == "trailer"]
    assert len(trailers) == 1, trailers
    return trailers[0]


def _stopped_at(run: Run, case: str, arm: str, needle: str | None) -> None:
    """Exit 1, ``stopped`` in the trailer (containing ``needle``), and nothing sent after ``case``'s ``arm`` request."""
    assert run.result.exit_code == 1, (run.result.exit_code, run.result.stopped)
    stopped = str(_trailer(run).get("stopped") or "")
    assert stopped, "no stopped in the trailer"
    if needle is not None:
        assert needle in stopped, stopped
    last = run.sent[-1]
    assert (last.case, last.kind, arm_of(last)) == (case, "p2", arm), (
        "requests continued after the stop",
        (last.case, last.kind, cell_of(last)),
    )
    assert len(run.srv.requests) == len(run.sent)


def _seeded_t4(case: str, **kw: Any) -> Reply:
    return Reply(text=p2_text(case, ts.gt_call(ts.by_id()[case])[0], "T4"), **kw)


def test_seed_accounting_stops_at_steps_equal_to_k0(tmp_path: Path) -> None:
    """Criterion 6a: search-02's seeded request commits canvas 0 with counter 24 = k0 (no executed step) -> stop
    ``seed_accounting``: exit 1, trailer, nothing after it."""
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", True)] = _seeded_t4("search-02", steps=K0)
    run = _small(tmp_path, sc)
    _stopped_at(run, "search-02", SEED, "seed_accounting")


@pytest.mark.parametrize("steps", [K0 - 5, 1])
def test_seed_accounting_stops_below_k0_too(tmp_path: Path, steps: int) -> None:
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", True)] = _seeded_t4("search-02", steps=steps)
    run = _small(tmp_path, sc)
    _stopped_at(run, "search-02", SEED, "seed_accounting")


def test_seed_accounting_does_not_stop_one_step_above_k0_or_unseeded(tmp_path: Path) -> None:
    """The other side of 6a (too wide): a seeded counter of 25 runs (1 executed step); an unseeded request at 24, or
    at the default 7, is no stop either (k0 is 0 there). 2 cases x 2 cells, exit 0."""
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", True)] = _seeded_t4("search-02", steps=K0 + 1)
    sc.seed_override[("search-01", "T4", False)] = _seeded_t4("search-01", steps=K0)
    run = _small(tmp_path, sc)
    assert run.result.exit_code == 0, run.result.stopped
    assert run.row("search-02", f"T4/named/{SEED}")["steps"] == [1]
    assert run.row("search-01", f"T4/named/{BASE}")["steps"] == [K0]
    assert len(run.srv.requests) == 2 + 4


def test_seed_accounting_stop_through_the_cli_still_writes_the_report(tmp_path: Path) -> None:
    """Criterion 6 (T-0080b criterion 4 pattern): through ``scripts/gpu/cd_toolcall.py`` the 6a stop is exit 1 and
    ``report-dev.json`` / ``.md`` are still written from the artifact, naming the stop."""
    cli = _cli()
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", True)] = _seeded_t4("search-02", steps=K0)
    out = tmp_path / "evidence"
    with ts.fake(sc, tmp_path) as srv:
        rc = cli.main(_cli_args(srv, tmp_path, out, "--cell", f"T4/named/{SEED}"))
    assert rc == 1, rc
    rep = out / "report-dev.json"
    assert rep.is_file() and (out / "report-dev.md").is_file(), sorted(
        p.name for p in out.iterdir()
    )
    assert "seed_accounting" in rep.read_text()


DRAFT_REASONS = [
    "draft_bad_encoding",
    "draft_bad_length",
    "draft_id_out_of_vocab",
    "draft_noise_out_of_range",
    "draft_start_step_out_of_range",
]


@pytest.mark.parametrize("code", [*DRAFT_REASONS, "missing_key"])
def test_a_draft_refusal_on_a_seeded_request_stops_the_run(tmp_path: Path, code: str) -> None:
    """Criterion 6b: a seeded request whose record carries a ``draft_*`` refusal reason, or ``missing_key``, stops the
    run (exit 1, trailer); it is never a row error the run carries on past."""
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", True)] = Reply(error=code)
    run = _small(tmp_path, sc)
    _stopped_at(run, "search-02", SEED, None)


@pytest.mark.parametrize("code", DRAFT_REASONS)
def test_a_draft_reason_on_an_unseeded_request_is_not_a_new_stop(tmp_path: Path, code: str) -> None:
    """The other side of 6b (``LOUD_ERRORS`` is not widened): the same reason on an *unseeded* request is recorded
    as the row's error and the run goes on (2 cases x 2 cells), exit 0."""
    sc = SeedScenario()
    sc.seed_override[("search-02", "T4", False)] = Reply(error=code)
    run = _small(tmp_path, sc)
    assert run.result.exit_code == 0, run.result.stopped
    assert len(run.srv.requests) == 2 + 4
    errs = [(x["case"], x["cell"], x["error"]) for x in run.rows(2) if x["error"]]
    assert errs == [("search-02", f"T4/named/{BASE}", code)], errs


@pytest.mark.parametrize("bad", ["out_of_vocab", "empty"])
def test_a_seed_the_contract_refuses_stops_and_is_never_sent_unseeded(
    tmp_path: Path, bad: str
) -> None:
    """Criterion 6c: a seed ``CdRequest`` refuses (an id >= the 262 144 vocabulary; an empty seed) stops the run as
    ``seed_invalid`` and the cell's request is never sent -- not unseeded, not at all. ``twopass.seed_ids`` is
    wrapped to produce the bad seed (delegating to the real function for the rest). search-01's order is
    (non-seed, seed): one pass-1 and one non-seed request reach the server, then the stop."""
    tp = ts.twopass()
    real = getattr(tp, "seed_ids", None)
    assert real is not None, "T-0081b not implemented: twopass.seed_ids is missing"

    def bad_seed(*a: Any, **kw: Any) -> list[int]:
        ids = [int(t) for t in real(*a, **kw)]
        return [*ids, 262144] if bad == "out_of_vocab" else []

    sc = SeedScenario()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tp, "seed_ids", bad_seed)
        run = _small(tmp_path, sc, limit=1)
    assert run.result.exit_code == 1, (run.result.exit_code, run.result.stopped)
    assert "seed_invalid" in str(_trailer(run).get("stopped")), _trailer(run)
    p2 = [s for s in run.sent if s.kind == "p2"]
    assert [arm_of(s) for s in p2] == [BASE], [cell_of(s) for s in p2]
    assert sum(1 for s in p2 if s.shape == "T4") == 1


# =========================================================================== criterion 7: the stage-2 order


def test_stage2_cells_run_in_the_williams_order_of_four(tmp_path: Path) -> None:
    """Criterion 7: the four stage-2 cells of a T4 choice -- end_projection, perstep_mar, perstep_mar_qargmax,
    perstep_mar+seed -- run per routed case in R-W ``williams_orders(4)``, hand-computed as rows
    ``0 1 3 2 / 1 2 0 3 / 2 3 1 0 / 3 0 2 1`` cycled by routed case (5 cases: the fifth repeats the first)."""
    cells = [f"T4/named/{a}" for a in ("end_projection", BASE, "perstep_mar_qargmax", SEED)]
    calib = {BASE: ts.CALIBRATION[BASE], "perstep_mar_qargmax": ts.QARGMAX_FILE}
    run = _execute(tmp_path, SeedScenario(), cells, 5, calibration=calib).get()
    assert run.result.exit_code == 0, run.result.stopped
    rows = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    for i, case in enumerate(CASES[:5]):
        got = [run.cell(s) for s in run.p2(case)]
        assert got == [cells[j] for j in rows[i % 4]], (case, got)


# =========================================================================== criterion 9: the report

REPORT_CELLS = [f"T4/named/{BASE}", f"T4/named/{SEED}"]


def report_scenario() -> SeedScenario:
    """search-01..06 under REPORT_CELLS. Valid drafts: search-01, -03 copied by the seeded pass 2; search-02's seeded
    pass 2 commits another (valid) query -> not a copy. search-04's draft has an unknown argument (invalid; the
    seeded pass 2 commits the exact call -> not a copy). search-05: prose after ``<cc>`` (picker, no seed).
    search-06: no tag (not routed)."""
    sc = SeedScenario(
        pass1={
            "search-04": _thought("search-04")
            + "Searching. <cc>"
            + '{"name": "google_search", "arguments": {"query": "banana bread egg", "lang": "en"}}',
            "search-05": _thought("search-05") + "Searching now. <cc>I will search for it.",
            "search-06": _thought("search-06") + "Artemis is a NASA programme.",
        }
    )
    other = ts.call_text("google_search", {"query": "british museum hours"})
    sc.seed_override[("search-02", "T4", True)] = Reply(text=other, steps=SEED_STEPS)
    return sc


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Run, Any]]:
    tmp = tmp_path_factory.mktemp("t0081b_report")
    run = _execute(tmp, report_scenario(), REPORT_CELLS, 6)
    if run.error is not None:
        yield run, None
        return
    copy = tmp / "copy" / Path(run.result.artifact).name
    copy.parent.mkdir()
    copy.write_bytes(Path(run.result.artifact).read_bytes())
    yield run, copy


def test_report_seed_coverage_and_copy_rate(reported: tuple[Run, Any]) -> None:
    """Criterion 9, from the artifact alone (a copy, read after the server closed). For the seed cell:
    coverage 4 seeded of 5 routed, no-seed {picker: 1}; copy rate (committed ids equal ``seed_ids`` through the seed's
    first stop id) on schema-valid drafts 2/3, on invalid drafts 0/1 (pooled it would read 2/4). The non-seed cell has
    no seed entry.

    Pinned names (test-author proposal; the task does not spell them out): ``report(...)["seed"][<cell>]`` with
    ``routed``, ``seeded``, ``no_seed`` (reason -> count) and ``copy`` -> ``valid_draft`` / ``invalid_draft``, each a
    rate mapping with ``k`` and ``n`` like every other rate in the report."""
    run, copy = reported
    run.get()
    assert run.result.exit_code == 0, run.result.stopped
    p1 = {x["case"]: x for x in run.rows(1)}
    assert [
        p1[c]["draft_schema_valid"] for c in ("search-01", "search-02", "search-03", "search-04")
    ] == [
        True,
        True,
        True,
        False,
    ]
    tp = ts.twopass()
    rep = tp.report([copy])
    seed = rep.get("seed")
    assert isinstance(seed, Mapping), sorted(rep)
    assert set(seed) == {f"T4/named/{SEED}"}, sorted(seed)
    s = seed[f"T4/named/{SEED}"]
    assert (s["routed"], s["seeded"]) == (5, 4), s
    assert dict(s["no_seed"]) == {"picker": 1}, s["no_seed"]
    v, i = s["copy"]["valid_draft"], s["copy"]["invalid_draft"]
    assert (v["k"], v["n"]) == (2, 3), v
    assert (i["k"], i["n"]) == (0, 1), i


def test_render_markdown_shows_seed_coverage_and_copy_rate(reported: tuple[Run, Any]) -> None:
    """Criterion 9: the markdown carries the seed cell's coverage (4/5), the no-seed reason (picker) and both copy
    rates (2/3 valid, 0/1 invalid) on lines naming the cell."""
    run, copy = reported
    run.get()
    tp = ts.twopass()
    md = tp.render_markdown(tp.report([copy]))
    lines = [ln for ln in md.splitlines() if f"T4/named/{SEED}" in ln]
    text = "\n".join(lines)
    for needle in ("4/5", "picker", "2/3", "0/1"):
        assert needle in text, (needle, text[-1500:])


def test_the_mcnemar_family_grows_with_the_seed_cell(reported: tuple[Run, Any]) -> None:
    """Criterion 9: the seed cell joins the McNemar family: 2 cells -> 4 CD policies vs ``draft_only`` + 1 pair = 5,
    with the seed cell's comparisons present and Bonferroni over 5."""
    run, copy = reported
    run.get()
    rep = ts.twopass().report([copy])
    assert rep["mcnemar_family"] == 5, rep["mcnemar_family"]
    keys = set(rep["mcnemar_exact"])
    assert f"cd_always:T4/named/{SEED} vs draft_only" in keys, sorted(keys)
    assert f"cd_on_invalid:T4/named/{SEED} vs draft_only" in keys
    assert f"cd_always:T4/named/{BASE} vs cd_always:T4/named/{SEED}" in keys
    for v in rep["mcnemar_exact"].values():
        a = v["all"]
        assert a["p_bonferroni"] == pytest.approx(min(1.0, a["p"] * 5)), v


# --------------------------------------------------------------------------- the CLI (loaded by path)

CLI = ts.REPO / "scripts" / "gpu" / "cd_toolcall.py"


def _cli() -> Any:
    import importlib.util
    import sys

    ts.twopass()
    name = "_t0081b_cd_toolcall_cli"
    spec = importlib.util.spec_from_file_location(name, CLI)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    return m


def _cli_args(srv: Any, tmp_path: Path, out: Path, *extra: str) -> list[str]:
    repo, head = ts.es.clean_repo(tmp_path)
    return [
        "--stage",
        "dev",
        "--base-url",
        srv.url,
        "--engine-log",
        str(srv.engine_log),
        "--out-dir",
        str(out),
        "--repo",
        str(repo),
        "--expected-commit",
        head,
        "--calibration",
        f"perstep_mar={ts.CALIBRATION['perstep_mar']}",
        "--limit",
        "2",
        *extra,
    ]


def test_the_cli_refuses_a_seed_calibration_key_with_exit_2(tmp_path: Path) -> None:
    """Criterion 1 through the CLI: ``--calibration perstep_mar+seed=<file>`` is exit 2 and no request is sent.
    Control (so the row cannot pass by refusing the seed cell itself): the same arguments without that entry run
    (4 requests, not a refusal)."""
    cli = _cli()
    ctl = tmp_path / "control"
    ctl.mkdir()
    with ts.fake(SeedScenario(), ctl) as srv:
        rc = cli.main(_cli_args(srv, ctl, ctl / "evidence", "--cell", f"T4/named/{SEED}"))
        # dev stage: the run completes; exit 1 only because a seed cell is no stage-1 candidate (no selection)
        assert rc != 2 and len(srv.requests) == 4, (rc, len(srv.requests))
    out = tmp_path / "evidence"
    with ts.fake(SeedScenario(), tmp_path) as srv:
        rc = cli.main(
            _cli_args(
                srv,
                tmp_path,
                out,
                "--cell",
                f"T4/named/{SEED}",
                "--calibration",
                f"{SEED}={ts.CALIBRATION[BASE]}",
            )
        )
        assert rc == 2, rc
        assert srv.requests == [], len(srv.requests)


# --------------------------------------------------------------------------- review follow-ups (fbd8668)


def _gt_text(case: str) -> str:
    return ts.call_text(*ts.gt_call(ts.by_id()[case]))


def _padded(ids: Sequence[int]) -> list[int]:
    """A whole committed canvas as the real records carry it: the ids, then EOS (1) padding to CL."""
    out = list(ids)
    assert len(out) <= CL
    return out + [EOS] * (CL - len(out))


TRUNC_REPORT = "search-03"


def padded_scenario() -> SeedScenario:
    """search-01..04 under REPORT_CELLS, every seeded pass 2 committing a whole 256-id canvas (padded with 1):

    - search-01 (valid draft): the commit is the seed ``encode(call) + [1]`` then padding -> a copy;
    - search-02 (valid draft): the commit is ``encode(call)`` then ``<turn|>`` (106) then padding -- equal to the seed
      before the stop, but the seed's stop is EOS (1), so through the stop it differs -> not a copy;
    - search-03 (truncated pass 1, so the draft does not parse: an invalid draft): the seed has no stop (encode of
      the cut draft); the commit is the whole call, EOS, padding, which starts with the seed -> a copy;
    - search-04 (valid draft): the default unpadded commit ``encode(call) + [1]`` -> a copy.

    Copy rate: valid 2/3, invalid 1/1. A whole-list comparison would read valid 1/3 (only search-04) and invalid 0/1;
    a comparison that stops before the stop id would count search-02 (valid 3/3)."""
    sc = SeedScenario(pass1={TRUNC_REPORT: truncated_text(TRUNC_REPORT)})
    for case, tail in (("search-01", [EOS]), ("search-02", [106])):
        text = _gt_text(case)
        sc.seed_override[(case, "T4", True)] = Reply(
            text=text, committed_ids=_padded([*encode(text), *tail]), steps=SEED_STEPS
        )
    text = _gt_text(TRUNC_REPORT)
    sc.seed_override[(TRUNC_REPORT, "T4", True)] = Reply(
        text=text, committed_ids=_padded([*encode(text), EOS]), steps=SEED_STEPS
    )
    return sc


def test_copy_rate_reads_padded_canvases_through_the_seeds_first_stop(tmp_path: Path) -> None:
    """Review F (medium): real ``ASTRA_CD`` records hold the whole canvas (the stop, then 1-padding), so the copy
    test is "equal through the seed's first stop id" (T-0081 ``is_copy``), never whole-list equality. Hand counts in
    ``padded_scenario``: valid 2/3, invalid 1/1."""
    run = _execute(tmp_path, padded_scenario(), REPORT_CELLS, 4).get()
    assert run.result.exit_code == 0, run.result.stopped
    p1 = {x["case"]: x for x in run.rows(1)}
    assert [p1[c]["draft_schema_valid"] for c in ("search-01", "search-02", "search-04")] == [
        True
    ] * 3
    assert p1[TRUNC_REPORT]["draft_schema_valid"] is False
    rows = {x["case"]: x for x in run.rows(2) if x["cell"] == f"T4/named/{SEED}"}
    for case in ("search-01", "search-02", TRUNC_REPORT):
        assert len(rows[case]["committed_ids"][0]) == CL, (case, "premise: a padded canvas")
    seed3 = rows[TRUNC_REPORT]["seed_ids"]
    assert not {1, 50, 106} & set(seed3), "premise: a seed without a stop"
    assert rows[TRUNC_REPORT]["committed_ids"][0][: len(seed3)] == seed3, (
        "premise: the commit extends the seed"
    )
    s = ts.twopass().report([run.result.artifact])["seed"][f"T4/named/{SEED}"]
    v, i = s["copy"]["valid_draft"], s["copy"]["invalid_draft"]
    assert (v["k"], v["n"]) == (2, 3), v
    assert (i["k"], i["n"]) == (1, 1), i


def test_a_stopped_run_counts_unrun_seed_rows_as_missing(tmp_path: Path) -> None:
    """Ruling R1: a routed case with no row for the seed cell (the run stopped first) counts under
    ``no_seed["missing"]``, so seeded + no-seed = routed. search-01's order is (non-seed, seed); its non-seed request
    ends on a loud ``z0`` -> stop before the seed cell runs: routed 1, seeded 0, no_seed {missing: 1}."""
    sc = SeedScenario()
    sc.seed_override[("search-01", "T4", False)] = Reply(error="z0")
    run = _execute(tmp_path, sc, REPORT_CELLS, 2).get()
    assert run.result.exit_code == 1, run.result.stopped
    assert [x["cell"] for x in run.rows(2)] == [f"T4/named/{BASE}"], "premise: no seed-cell row"
    s = ts.twopass().report([run.result.artifact])["seed"][f"T4/named/{SEED}"]
    assert (s["routed"], s["seeded"]) == (1, 0), s
    assert dict(s["no_seed"]) == {"missing": 1}, s["no_seed"]


def test_regime_seed_draft_only_when_a_seed_cell_runs(seeded: Run, unseeded: Run) -> None:
    """Review F (low): ``regime["seed_draft"]`` is present, with ``draft_noise`` 0.0 and ``draft_start_step`` 24,
    when a seed cell runs, and absent when none does."""
    with_seed = ts.artifact_lines(seeded.get().result.artifact)[0]
    without = ts.artifact_lines(unseeded.get().result.artifact)[0]
    assert "seed_draft" not in without, without.get("seed_draft")
    sd = with_seed.get("seed_draft")
    assert isinstance(sd, Mapping), sorted(with_seed)
    assert sd["draft_noise"] == NOISE and isinstance(sd["draft_noise"], float), sd
    assert sd["draft_start_step"] == K0 and not isinstance(sd["draft_start_step"], bool), sd


# --------------------------------------------------------------------------- ruling R2: the trim in rows and report

#: case -> (text before the call, text after it) in the draft; search-03: nothing around the call
TRIM = {
    "search-01": ("", "\n"),
    "search-02": ("", " \n"),
    "search-04": (" ", "\n"),
    "search-05": (" ", ""),
}
TRIM_PICKER = "search-06"  # prose after <cc>: routed, name picker, no seed
TRIMMED_CHARS = {"search-01": 1, "search-02": 2, "search-03": 0, "search-04": 1, "search-05": 0}


def trim_scenario() -> SeedScenario:
    """search-01..06 under REPORT_CELLS. Drafts: ``call\\n``, ``call \\n``, ``call``, `` call\\n`` and `` call``
    (leading spaces are kept, never trimmed), then a picker case. The seeded pass 2 of the leading-space cases commits
    the seed itself (`` call`` then EOS; T4's grammar admits leading whitespace), so every seeded case is a copy."""
    pass1 = {}
    for c, (lead, tail) in TRIM.items():
        pass1[c] = ts.Scenario().pass1_text(c).replace("<cc>", "<cc>" + lead, 1) + tail
    pass1[TRIM_PICKER] = _thought(TRIM_PICKER) + "Searching now. <cc>I will search for it."
    sc = SeedScenario(pass1=pass1)
    for c, (lead, _tail) in TRIM.items():
        if lead:
            sc.seed_override[(c, "T4", True)] = Reply(text=lead + _gt_text(c), steps=SEED_STEPS)
    return sc


@pytest.fixture(scope="module")
def trimmed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    yield _execute(tmp_path_factory.mktemp("t0081b_trim"), trim_scenario(), REPORT_CELLS, 6)


def test_seeded_rows_record_seed_trimmed(trimmed: Run) -> None:
    """R2: ``seed_trimmed`` counts only *trailing* whitespace: ``call\\n`` 1, ``call \\n`` 2, ``call`` 0,
    `` call\\n`` 1 and `` call`` 0 (a leading space is kept and is not counted; ``len - len(strip())`` would read 2
    and 1). The seed sent is the call with its leading text, trimmed at the end, then EOS."""
    r = trimmed.get()
    assert r.result.exit_code == 0, r.result.stopped
    p1 = {x["case"]: x for x in r.rows(1)}
    for case, n in TRIMMED_CHARS.items():
        lead, tail = TRIM.get(case, ("", ""))
        assert p1[case]["draft_call_text"] == lead + _gt_text(case) + tail, (
            "premise: the router's draft"
        )
        (s,) = r.p2(case, f"T4/named/{SEED}")
        assert unwire(cd_keys(s.body)["astra_cd_draft"]) == [*encode(lead + _gt_text(case)), EOS], (
            case
        )
        row = r.row(case, f"T4/named/{SEED}")
        assert row.get("seed_trimmed") == n, (case, row.get("seed_trimmed"))
    assert r.row(TRIM_PICKER, f"T4/named/{SEED}").get("seed_reason") == "picker"


def test_report_and_markdown_give_the_trimmed_rate(trimmed: Run) -> None:
    """R2: ``trimmed`` is the share of *seeded* rows with ``seed_trimmed > 0``: search-01, -02 and -04 of the five
    seeded cases = 3/5 (over the six routed cases, picker included, it would read 3/6). Every seeded case is a copy
    (5/5 on valid drafts). The markdown shows 3/5 on a line naming the cell and the word ``trim``."""
    r = trimmed.get()
    tp = ts.twopass()
    rep = tp.report([r.result.artifact])
    s = rep["seed"][f"T4/named/{SEED}"]
    assert (s["routed"], s["seeded"]) == (6, 5), s
    assert (s["trimmed"]["k"], s["trimmed"]["n"]) == (3, 5), s.get("trimmed")
    v = s["copy"]["valid_draft"]
    assert (v["k"], v["n"]) == (5, 5), v
    md = tp.render_markdown(rep)
    lines = [ln for ln in md.splitlines() if f"T4/named/{SEED}" in ln and "trim" in ln.lower()]
    assert any("3/5" in ln for ln in lines), lines
