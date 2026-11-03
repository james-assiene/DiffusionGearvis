"""T-0084c: the two-pass driver runs the native Gemma 4 tool-call format beside ``<cc>`` (criteria 1-4 and 6-11).

CPU; the loopback fake of ``tests/cd/fakes/vllm_server.py`` (T-0084c: ``prompt_token_ids`` rendered by the pinned
chat template, one 46/47 pair per ``tools`` entry, a continued ``<|tool_call>`` ending in 48, scripted guard
failures); the pinned tokenizer and the real ``compile_toolcall`` automata (``call_format="json"`` and ``"native"``);
frozen ``docs/cd/calibration-bfcl.json``; CL 256, seed noise 0 / start 24 / R2. Expected values come from the
criteria, from the template oracle (``tests/cd/oracle/gemma4_template.py``: every scripted native call is the
template's own rendering) or from hand counts; the cc-only bit-identity row compares against a golden captured from
``main`` at 9737368 on this same scenario (``tests/cd/fixtures/t0084c_cc_golden.json``, MANIFEST.md).

The scenario (``CASES``, an 11-case dev subset in this order, so case index parity is known):

- ``search-02 click-01 type-01 scroll-01 key-01``: both formats answer with the case's exact call (cc: ``<cc>`` + JSON;
  native: ``<|tool_call>`` + the template call + ``<|tool_response>``), spoken prefix ``"On it (<id>). "``;
- ``chat-01``: neither format calls a tool;
- ``search-04`` (native picker): the native draft names ``web_search`` (not v1) -> ``native.draft_name`` None -> the
  native name picker, then ``google_search``;
- ``click-04`` (``wrong_tag``, QN7 as amended by H-T84-0 / T-0084d criterion 7): the native pass 1 answers with
  ``<cc>`` + JSON -> called, ``draft_parse`` ``"wrong_tag"``, invalid; then, like a native draft without a name, the
  native name picker and one constrained native pass 2 for the picked tool (``click_at``) per native cell;
- ``search-05`` (adversarial): the cc pass 1 answers with ``<|tool_call>call:google_search{...}`` -> the cc picker
  exactly as on ``main`` (JSON automata, no ``tools``);
- ``scroll-02``: only native routes (cc answers without a tag) -> cc cells skipped;
- ``key-02``: only cc routes (native answers without a call) -> native cells skipped.

Hand counts for the mixed run (3 cc + 3 native cells): pass 1 = 11 x 2 = 22; cc pass 2 = 9 routed x 3 + 3 picker
extras = 30; native pass 2 = 9 routed x 3 + 6 picker extras (search-04 and, since H-T84-0, click-04) = 33; total 85
requests. (T-0084c's count was 8 x 3 + 3 = 27 native, 79 in all: ``wrong_tag`` sent no pass 2 then.)

API pins this module makes where the task is silent (reported to the orchestrator): native cells are
``native/<T>/<scope>/<arm>``; ``twopass.NATIVE_SYSTEM_PROMPT`` is the task's wording exactly (no trailing newline);
rows carry ``format`` ``"cc"``/``"native"``; stops name ``prompt_guard: <reason>`` in ``stopped``; the regime holds
sha256 of ``NATIVE_SYSTEM_PROMPT`` and of ``json.dumps(tools_field, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)``; ``derive`` takes ``draft_only:native``; native seed sources reuse T-0081b's names.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import socket
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _twopass_support as ts
from tests.cd.fakes.vllm_server import Reply, cd_keys

CASES = [
    "search-02",
    "click-01",
    "type-01",
    "scroll-01",
    "key-01",
    "chat-01",
    "search-04",
    "click-04",
    "search-05",
    "scroll-02",
    "key-02",
]
NORMAL = CASES[:5]
NONE = "chat-01"
NATIVE_PICKER = "search-04"
WRONG_TAG = "click-04"
CC_NATIVE_TAG = "search-05"
NATIVE_ONLY = "scroll-02"
CC_ONLY = "key-02"
CC_CELLS = tuple(ts.CELLS)  # T1/T2/T4 named perstep_mar
NATIVE_CELLS = tuple(ts.native_cell(c) for c in CC_CELLS)
MIXED = CC_CELLS + NATIVE_CELLS
CC_ROUTED = [c for c in CASES if c not in (NONE, NATIVE_ONLY)]
#: T-0084d criterion 7 (H-T84-0): the wrong_tag case now sends native pass 2 (through the native picker)
NATIVE_ROUTED = [c for c in CASES if c not in (NONE, CC_ONLY)]
NATIVE_PICKED = (NATIVE_PICKER, WRONG_TAG)
N_REQUESTS = 85
#: R-W Williams rows for k = 6, written out by hand from T-0080b's definition (row 0 = 0, 1, k-1, 2, k-2, ...;
#: row i adds i mod k); k even -> 6 rows.
WILLIAMS_6 = [
    [0, 1, 5, 2, 4, 3],
    [1, 2, 0, 3, 5, 4],
    [2, 3, 1, 4, 0, 5],
    [3, 4, 2, 5, 1, 0],
    [4, 5, 3, 0, 2, 1],
    [5, 0, 4, 1, 3, 2],
]
#: per case, the routed count it is indexed by (cases routed in at least one format; chat-01 routes in neither)
WILLIAMS_INDEX = {
    "search-02": 0,
    "click-01": 1,
    "type-01": 2,
    "scroll-01": 3,
    "key-01": 4,
    "search-04": 5,
    "click-04": 0,
    "search-05": 1,
    "scroll-02": 2,
    "key-02": 3,
}
OPEN, CLOSE, RESPONSE = ts.TOOL_CALL_OPEN, ts.TOOL_CALL_CLOSE, ts.TOOL_RESPONSE
SEED = "perstep_mar+seed"
SEEDED_STEPS = 27  # a seeded request's step counter (27 - 24 = 3 executed), as in the T-0081b rows
GOLDEN = ts.REPO / "tests" / "cd" / "fixtures" / "t0084c_cc_golden.json"
#: row fields that differ between two runs of the same scenario (timings, run ids inside tags)
VOLATILE = {"wall_ms", "wall_ms_per_request", "record_tags"}


# --------------------------------------------------------------------------- the scripted model


@dataclasses.dataclass
class Logged:
    index: int
    fmt: str  # "cc" | "native" (the request's shape: ``tools`` present -> native)
    kind: str  # "p1" | "p2"
    case: str | None
    automaton: str | None  # pass 2: "json" | "native" | "?"
    scope: str | None
    thought: str | None
    shape: str | None
    body: Mapping[str, Any]
    reply: Reply


class NativeScenario(ts.Scenario):
    """``ts.Scenario`` for cc requests, plus native pass-1 answers and native pass-2 outputs, all from the template
    oracle. ``edits[(case, fmt, kind)]`` (kind ``p1``/``T1``/``T2``/``T4``) replaces fields of the first matching
    reply (then is consumed); ``edited`` records the request index each edit hit."""

    def __init__(self) -> None:
        super().__init__()
        self.native_pass1: dict[str, str] = {}
        self.edits: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.edited: dict[tuple[str, str, str], int] = {}
        self.log: list[Logged] = []
        self._nlock = threading.Lock()

    def native_pass1_text(self, cid: str) -> str:
        if cid in self.native_pass1:
            return self.native_pass1[cid]
        case = ts.by_id()[cid]
        head = f"<|channel>thought\nThe user asked for {cid}.<channel|>"
        if case["kind"] != "tool":
            return head + f"Here is my answer to {cid}."
        name, args = ts.gt_call(case)
        return head + self.spoken(cid) + ts.native_call(name, args) + RESPONSE

    def native_pass2_text(self, cid: str, scope: str, thought: str) -> str:
        head = {
            "T1": f"<|channel>thought\nRe-thinking {cid}.<channel|>",
            "T2": ts.EMPTY_HEADER,
            "T4": "",
        }[thought]
        lead = "" if thought == "T4" else OPEN
        name, args = self.call_for(cid)
        if scope == "name":
            return head + lead + f"call:{name}{{"
        return head + lead + ts.native_call(name, args)[len(OPEN) :]

    def _edit(self, index: int, case: str | None, fmt: str, kind: str, reply: Reply) -> Reply:
        key = (case or "", fmt, kind)
        with self._nlock:
            fields = self.edits.pop(key, None)
            if fields is not None:
                self.edited[key] = index
        return dataclasses.replace(reply, **fields) if fields is not None else reply

    def __call__(self, index: int, body: Mapping[str, Any]) -> Reply:
        case = ts.case_of(body)
        keys = cd_keys(body)
        fmt = "native" if "tools" in body else "cc"
        if not keys:
            if fmt == "cc":
                reply = super().__call__(index, body)
            else:
                reply = Reply(
                    text=self.native_pass1_text(case) if case else "",
                    prefix_queries=5000,
                    prefix_hits=77,
                )
            reply = self._edit(index, case, fmt, "p1", reply)
            entry = Logged(index, fmt, "p1", case, None, None, None, None, body, reply)
        else:
            auto, scope, thought = ts.payload_format(keys)
            shape = ts.thought_of_body(body)
            if auto == "native" and case is not None:
                reply = Reply(
                    text=self.native_pass2_text(case, scope, thought),
                    steps=ts.P2_STEPS,
                    bucket=int(ts.compiled_native(scope, thought).n_states_bucket),
                    prefix_queries=1000 + index,
                    prefix_hits=900 + index,
                )
            elif auto == "native":
                reply = Reply(text="", prefix_queries=1000 + index, prefix_hits=900 + index)
            else:
                reply = super().__call__(index, body)
            if "astra_cd_draft" in keys and reply.steps == ts.P2_STEPS:
                reply = dataclasses.replace(reply, steps=SEEDED_STEPS)
            reply = self._edit(index, case, fmt, shape, reply)
            entry = Logged(index, fmt, "p2", case, auto, scope, thought, shape, body, reply)
        with self._nlock:
            self.log.append(entry)
        return reply


def scenario() -> NativeScenario:
    sc = NativeScenario()
    head = "<|channel>thought\nThinking.<channel|>"
    # cc side
    sc.pass1[NATIVE_ONLY] = head + "I cannot scroll from here."
    sc.pass1[CC_NATIVE_TAG] = (
        head
        + sc.spoken(CC_NATIVE_TAG)
        + ts.native_call("google_search", {"query": "artemis"})
        + RESPONSE
    )
    # native side
    sc.native_pass1[NATIVE_PICKER] = (
        head
        + sc.spoken(NATIVE_PICKER)
        + OPEN
        + 'call:web_search{query:<|"|>banana bread egg<|"|>}'
        + CLOSE
        + RESPONSE
    )
    name, args = ts.gt_call(ts.by_id()[WRONG_TAG])
    sc.native_pass1[WRONG_TAG] = head + sc.spoken(WRONG_TAG) + ts.CC + ts.call_text(name, args)
    sc.native_pass1[CC_ONLY] = head + "No key needs pressing."
    return sc


# --------------------------------------------------------------------------- runs


@dataclasses.dataclass
class Run:
    result: Any
    requests: list[dict[str, Any]]
    prompt_ids: list[list[int] | None]
    log: list[Logged]
    seeds: list[Any]
    connects: list[Any]
    srv_url: str
    edited: dict[tuple[str, str, str], int]
    refusal: ValueError | None = None
    flushes: list[Any] = dataclasses.field(default_factory=list)
    rejected: list[Any] = dataclasses.field(default_factory=list)

    def lines(self) -> list[dict[str, Any]]:
        return ts.artifact_lines(self.result.artifact)

    def p1(self, case: str, fmt: str) -> dict[str, Any]:
        rows = [
            x
            for x in self.lines()[1:]
            if x.get("pass") == 1 and x.get("case") == case and x.get("format") == fmt
        ]
        assert len(rows) == 1, (case, fmt, len(rows))
        return rows[0]

    def p2(self, case: str, cell: str) -> dict[str, Any]:
        rows = [
            x
            for x in self.lines()[1:]
            if x.get("pass") == 2 and x.get("case") == case and x.get("cell") == cell
        ]
        assert len(rows) == 1, (case, cell, len(rows))
        return rows[0]

    def sent(self, case: str, fmt: str, kind: str) -> list[Logged]:
        return [x for x in self.log if x.case == case and x.fmt == fmt and x.kind == kind]


def _tp() -> Any:
    tp = ts.twopass()
    if not hasattr(tp, "NATIVE_SYSTEM_PROMPT"):
        raise AssertionError("T-0084c not implemented: twopass.NATIVE_SYSTEM_PROMPT is missing")
    return tp


def run_scenario(
    tmp: Path,
    cells: Sequence[str],
    *,
    ids: Sequence[str] = CASES,
    sc: NativeScenario | None = None,
    fake_kw: Mapping[str, Any] | None = None,
    check: bool = True,
    refuse_ok: bool = False,
    **overrides: Any,
) -> Run:
    """One driver run over the subset ``ids``. ``check`` asserts the T-0084c surface exists first (``_tp``); a
    ``ValueError`` from ``run`` is an assertion unless ``refuse_ok`` (then it is kept in ``refusal``)."""
    tp = _tp() if check else ts.twopass()
    sc = sc or scenario()
    tmp.mkdir(parents=True, exist_ok=True)
    set_dir = ts.subset_set_dir(tmp / "set", ids)
    connects: list[Any] = []
    real_connect = socket.socket.connect

    def recording(sock: socket.socket, address: Any) -> Any:
        connects.append(address)
        return real_connect(sock, address)

    result: Any = None
    refusal: ValueError | None = None
    with ts.fake(sc, tmp, **dict(fake_kw or {})) as srv:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", recording)
            try:
                result = ts.run_driver(srv, tmp, cells=list(cells), set_dir=set_dir, **overrides)
            except ValueError as exc:
                refusal = exc
        out = Run(
            result,
            list(srv.requests),
            list(srv.prompt_ids),
            sorted(sc.log, key=lambda x: x.index),
            srv.cd_seeds(),
            connects,
            srv.url,
            dict(sc.edited),
            refusal,
            list(srv.flushes),
            list(srv.rejected),
        )
    assert tp is not None
    if out.refusal is not None and refuse_ok is False:
        raise AssertionError(f"the run was refused: {out.refusal}")
    return out


class Lazy:
    def __init__(self, fn: Callable[[], Run]) -> None:
        self.fn = fn
        self.value: Run | None = None
        self.error: BaseException | None = None

    def get(self) -> Run:
        if self.value is None and self.error is None:
            try:
                self.value = self.fn()
            except BaseException as exc:  # re-raised in every row that reads it
                self.error = exc
        if self.error is not None:
            raise self.error
        assert self.value is not None
        return self.value


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Lazy]:
    """The mixed run (3 cc + 3 native cells) over the scenario; built inside the first row that needs it."""

    def build() -> Run:
        r = run_scenario(tmp_path_factory.mktemp("t0084c_mixed"), MIXED)
        assert r.refusal is None, f"the mixed run was refused: {r.refusal}"
        assert r.result.exit_code == 0, (r.result.stopped, ts.stopped_marker(r.result.artifact))
        return r

    yield Lazy(build)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Lazy]:
    """A native seed run: ``native/<T>/named/perstep_mar+seed`` for T1/T2/T4 over two direct cases, the picker and (T-0084d
    criterion 7) the wrong_tag case."""

    def build() -> Run:
        cells = [ts.native_cell(f"{t}/named/{SEED}") for t in ts.THOUGHTS]
        r = run_scenario(
            tmp_path_factory.mktemp("t0084c_seeded"),
            cells,
            ids=["search-02", "type-01", NATIVE_PICKER, WRONG_TAG],
        )
        assert r.refusal is None, f"the seeded run was refused: {r.refusal}"
        assert r.result.exit_code == 0, (r.result.stopped, ts.stopped_marker(r.result.artifact))
        return r

    yield Lazy(build)


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def _cell_of(x: Logged) -> str:
    base = f"{x.shape}/named/perstep_mar"
    return ts.native_cell(base) if x.automaton == "native" else base


def _gt(cid: str) -> dict[str, Any]:
    name, args = ts.gt_call(ts.by_id()[cid])
    return {"name": name, "arguments": args}


def _encode(text: str) -> list[int]:
    from local_astra.decoding import vocab

    return [int(t) for t in vocab.encode(text)]


# =========================================================================== premises of the oracle


def test_premises_of_the_native_scenario() -> None:
    """The scripted texts are what the rows assume: the markers are single ids 48/49/50, every native pass-2 output is
    in its native grammar's language, the picker draft names no v1 tool and the router tags what it should. Green
    before the implementation (it checks the fixture, not the driver)."""
    from local_astra.eval.cd import replay
    from local_astra.toolcall import native, router

    assert _encode(OPEN) == [48] and _encode(CLOSE) == [49] and _encode(RESPONSE) == [50]
    sc = scenario()
    for cid in NORMAL:
        name, _args = ts.gt_call(ts.by_id()[cid])
        for thought in ts.THOUGHTS:
            for scope in (name, "name"):
                ids = [*_encode(sc.native_pass2_text(cid, scope, thought)), 1]
                assert replay.accepts(ts.compiled_native(scope, thought), ids), (
                    cid,
                    scope,
                    thought,
                )
        r = router.route(sc.native_pass1_text(cid).removesuffix(RESPONSE))
        assert (r.kind, r.tag) == ("toolcall", "native")
    picker = router.route(sc.native_pass1_text(NATIVE_PICKER).removesuffix(RESPONSE))
    assert picker.tag == "native" and native.draft_name(picker.draft_call_text) is None
    assert router.route(sc.native_pass1_text(WRONG_TAG)).tag == "cc"
    assert router.route(sc.pass1_text(CC_NATIVE_TAG).removesuffix(RESPONSE)).tag == "native"
    assert router.route(sc.native_pass1_text(CC_ONLY)).kind == "none"
    assert router.route(sc.pass1_text(NATIVE_ONLY)).kind == "none"


# =========================================================================== criterion 1: cells


def test_the_mixed_run_completes_and_sends_the_hand_counted_requests(mixed: Lazy) -> None:
    """Criterion 1: native cells beside cc cells in one run; 22 pass-1 + 30 cc + 33 native pass-2 requests (T-0084d
    criterion 7: the wrong_tag case's picker + pass 2 add 6 to T-0084c's 27)."""
    r = mixed.get()
    assert len(r.requests) == N_REQUESTS, len(r.requests)
    assert len([x for x in r.log if x.kind == "p1"]) == 2 * len(CASES)


@pytest.mark.parametrize(
    "bad",
    [
        "xml/T1/named/perstep_mar",
        "json/T1/named/perstep_mar",
        "native/T5/named/perstep_mar",
        "native/T1/global/perstep_mar",
        "native/T1/named/bogus",
        "native/T1/union/perstep_mar+seed",
        "native/T1/named",
    ],
)
def test_an_unknown_format_or_a_bad_native_cell_raises_before_any_socket(
    tmp_path: Path, bad: str
) -> None:
    """Criterion 1: an unknown format (and a malformed native cell; the seed arm with ``union``) raises
    ``ValueError`` before any socket: zero requests, zero connections to the server."""
    r = run_scenario(tmp_path, [ts.CELLS[0], bad], ids=["search-02"], refuse_ok=True)
    assert r.refusal is not None, f"{bad!r} was accepted (exit {r.result.exit_code})"
    assert r.requests == []
    port = int(r.srv_url.rsplit(":", 1)[1].split("/")[0])
    assert [a for a in r.connects if isinstance(a, tuple) and a[1:2] == (port,)] == []


@pytest.mark.parametrize(
    "cell",
    [
        "native/T1/named/perstep_mar",
        "native/T2/named/end_projection",
        "native/T4/named/perstep_mar_qargmax",
        "native/T4/named/perstep_mar+seed",
        "native/T2/union/perstep_mar",
    ],
)
def test_native_cells_with_every_cc_arm_are_accepted(tmp_path: Path, cell: str) -> None:
    """Criterion 1, the other direction: ``native/<T>/<named|union>/<arm>`` with the cc arms is accepted (a run over
    a case that routes in neither format, so no pass 2 is needed)."""
    calib = {**ts.CALIBRATION, "perstep_mar_qargmax": ts.QARGMAX_FILE}
    r = run_scenario(tmp_path, [cell], ids=[NONE], calibration=calib, refuse_ok=True)
    assert r.refusal is None, f"{cell!r} refused: {r.refusal}"
    assert r.result.exit_code == 0, r.result.stopped


def test_a_native_only_run_sends_no_cc_pass_1(tmp_path: Path) -> None:
    """Criterion 1: a run sends a cc pass 1 iff it has a cc cell -- a native-only run sends one pass 1 per case,
    every one native (``tools`` present)."""
    r = run_scenario(tmp_path, list(NATIVE_CELLS), ids=["search-02", NONE])
    assert r.result.exit_code == 0, r.result.stopped
    p1 = [x for x in r.log if x.kind == "p1"]
    assert [x.fmt for x in p1] == ["native", "native"]
    assert [x.case for x in p1] == ["search-02", NONE]


def test_a_cc_only_run_sends_no_native_request(tmp_path: Path) -> None:
    """Criterion 1: a run sends a native pass 1 iff it has a native cell -- no request of a cc-only run carries
    ``tools``, ``tool_choice`` or a native automaton. Green before the implementation by construction."""
    r = run_scenario(tmp_path, list(CC_CELLS), ids=["search-02", CC_NATIVE_TAG], check=False)
    assert r.result.exit_code == 0, r.result.stopped
    assert not any("tools" in b or "tool_choice" in b for b in r.requests)
    assert {x.automaton for x in r.log if x.kind == "p2"} == {"json"}


# =========================================================================== criterion 2: the native pass-1 body


def test_native_system_prompt_is_the_tasks_wording() -> None:
    """Criterion 2: ``NATIVE_SYSTEM_PROMPT`` is the proposed wording (the template appends the declarations, so the
    prompt carries no declaration and no ``<cc>``); ``SYSTEM_PROMPT`` keeps its sha ``fa5dace3...``."""
    tp = _tp()
    assert tp.NATIVE_SYSTEM_PROMPT == ts.NATIVE_SYSTEM_PROMPT_TEXT
    assert _sha(tp.SYSTEM_PROMPT).startswith("fa5dace3"), _sha(tp.SYSTEM_PROMPT)


def test_the_native_pass_1_body_is_the_cc_body_with_the_native_system_and_tools(
    mixed: Lazy,
) -> None:
    """Criterion 2: same user content, ``system`` = ``NATIVE_SYSTEM_PROMPT``, ``tools`` = ``native.tools_field(tools)``,
    ``tool_choice`` = ``"none"``, thinking on, ``max_tokens`` 1024, ``return_token_ids``; nothing else differs from the
    cc pass-1 body of the same case."""
    from local_astra.toolcall import native

    r = mixed.get()
    field = native.tools_field(ts.toolset())
    for cid in CASES:
        (cc,) = r.sent(cid, "cc", "p1")
        (nat,) = r.sent(cid, "native", "p1")
        want = copy.deepcopy(dict(cc.body))
        want["messages"][0] = {"role": "system", "content": ts.NATIVE_SYSTEM_PROMPT_TEXT}
        want["tools"] = field
        want["tool_choice"] = "none"
        assert dict(nat.body) == want, cid
        assert nat.body["chat_template_kwargs"] == {"enable_thinking": True}
        assert (nat.body["max_tokens"], nat.body["return_token_ids"]) == (1024, True)
        assert "vllm_xargs" not in nat.body


def test_the_regime_records_the_native_prompt_and_tools_shas(mixed: Lazy) -> None:
    """Criterion 2: the regime records sha256 of ``NATIVE_SYSTEM_PROMPT`` and of the canonical JSON of the ``tools``
    field (pinned here: ``sort_keys``, separators ``(",", ":")``, UTF-8)."""
    from local_astra.toolcall import native

    regime = mixed.get().lines()[0]
    canon = json.dumps(
        native.tools_field(ts.toolset()), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    values = set()

    def walk(x: Any) -> None:
        if isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            values.add(x)

    walk(regime)
    assert _sha(ts.NATIVE_SYSTEM_PROMPT_TEXT) in values
    assert _sha(canon) in values


# =========================================================================== criterion 3: native pass-2 bodies


def _without(body: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in keys}


def test_native_t1_t2_bodies_are_the_native_pass_1_body_plus_xargs(mixed: Lazy) -> None:
    """Criterion 3: T1/T2 = a deep copy of the native pass-1 body plus ``vllm_xargs`` and ``max_tokens`` (1024 / 256);
    ``tools`` is kept."""
    r = mixed.get()
    for cid in NATIVE_ROUTED:
        (p1,) = r.sent(cid, "native", "p1")
        for x in r.sent(cid, "native", "p2"):
            if x.shape == "T4":
                continue
            assert x.body["max_tokens"] == {"T1": 1024, "T2": 256}[x.shape]
            assert _without(x.body, "vllm_xargs", "max_tokens") == _without(p1.body, "max_tokens")
            assert x.body["tools"] == p1.body["tools"]


def test_native_t4_bodies_continue_the_spoken_prefix_and_the_tool_call_marker(mixed: Lazy) -> None:
    """Criterion 3: T4 = the native pass-1 body + ``{"role": "assistant", "content": spoken + "<|tool_call>"}``,
    ``continue_final_message`` true, ``add_generation_prompt`` false, ``max_tokens`` 256, ``tools`` kept."""
    r = mixed.get()
    for cid in NATIVE_ROUTED:
        (p1,) = r.sent(cid, "native", "p1")
        t4 = [x for x in r.sent(cid, "native", "p2") if x.shape == "T4"]
        assert t4, cid
        for x in t4:
            want = copy.deepcopy(dict(p1.body))
            want["messages"].append(
                {"role": "assistant", "content": ts.Scenario.spoken(cid) + OPEN}
            )
            want.update(continue_final_message=True, add_generation_prompt=False, max_tokens=256)
            assert _without(x.body, "vllm_xargs") == want, cid


def test_native_cells_use_the_native_grammar_of_the_drafts_tool(mixed: Lazy) -> None:
    """Criterion 3: every native pass-2 request carries ``compile_toolcall(..., call_format="native")`` for the tool
    ``native.draft_name`` reads (the case's tool), at the body's thought; one request per cell."""
    r = mixed.get()
    for cid in [c for c in NATIVE_ROUTED if c not in NATIVE_PICKED]:
        reqs = r.sent(cid, "native", "p2")
        assert len(reqs) == 3, cid
        tool = ts.by_id()[cid]["tool"]
        for x in reqs:
            assert (x.automaton, x.scope, x.thought) == ("native", tool, x.shape), cid


def test_a_native_draft_naming_no_v1_tool_goes_through_the_native_picker(mixed: Lazy) -> None:
    """Criterion 3: ``native.draft_name`` None -> one request under the native ``name`` grammar, the picked name read
    with ``native.draft_name`` from its call text (``call:google_search{``), then that tool's native grammar."""
    r = mixed.get()
    for cell in NATIVE_CELLS:
        thought = cell.split("/")[1]
        reqs = [x for x in r.sent(NATIVE_PICKER, "native", "p2") if x.shape == thought]
        assert [(x.automaton, x.scope) for x in reqs] == [
            ("native", "name"),
            ("native", "google_search"),
        ]
        row = r.p2(NATIVE_PICKER, cell)
        assert (row["requests"], row["scopes"]) == (2, ["name", "google_search"])
        assert (row["parse"], row["schema_valid"], row["exact"]) == ("ok", True, True)
        assert row["format"] == "native"


# =========================================================================== criterion 4: scoring


def test_native_pass_1_drafts_are_parsed_natively_and_scored_unchanged(mixed: Lazy) -> None:
    """Criterion 4: a native draft is parsed by ``native.parse_call``; the template call of the ground truth is
    ``ok``, schema-valid and exact."""
    r = mixed.get()
    for cid in NORMAL:
        row = r.p1(cid, "native")
        assert row["route"]["tag"] == "native"
        assert (row["draft_parse"], row["draft_call"]) == ("ok", _gt(cid)), cid
        assert (row["draft_schema_valid"], row["draft_exact"]) == (True, True), cid


def test_native_pass_2_outputs_are_parsed_natively_and_scored_unchanged(mixed: Lazy) -> None:
    """Criterion 4: the pass-2 call text is the answer channel minus closed thoughts and a leading ``<|tool_call>``;
    the native parse of each scripted output is the ground-truth call, schema-valid and exact, CS 1."""
    r = mixed.get()
    for cid in NORMAL:
        for cell in NATIVE_CELLS:
            row = r.p2(cid, cell)
            assert (row["parse"], row["call"]) == ("ok", _gt(cid)), (cid, cell)
            assert (row["schema_valid"], row["exact"], row["cs"]) == (True, True, True)
            assert row["error"] is None


def test_wrong_tag_counts_as_called_invalid_and_is_retried_through_the_native_picker(
    mixed: Lazy,
) -> None:
    """Criterion 4 / QN7, amended by H-T84-0 (T-0084d criterion 7; this row replaced T-0084c's "sends no pass 2"): a
    native-format pass 1 answering with ``<cc>`` is called, ``draft_parse`` ``wrong_tag``, not schema-valid, not exact
    -- and is retried: per native cell, the native name picker, then one constrained native pass 2 under the picked
    tool's native grammar (``click_at``, the scripted pick), exactly as a native draft without a name. The cc cells of
    that case still run. ``draft_only:native`` keeps it called, invalid, not exact; ``cd_on_invalid:native/<cell>``
    scores it from pass 2 (the scripted output is the exact call)."""
    r = mixed.get()
    row = r.p1(WRONG_TAG, "native")
    assert row["route"]["kind"] == "toolcall" and row["route"]["tag"] == "cc"
    assert (row["draft_parse"], row["draft_schema_valid"], row["draft_exact"]) == (
        "wrong_tag",
        False,
        False,
    )
    for cell in NATIVE_CELLS:
        thought = cell.split("/")[1]
        reqs = [x for x in r.sent(WRONG_TAG, "native", "p2") if x.shape == thought]
        assert [(x.automaton, x.scope) for x in reqs] == [
            ("native", "name"),
            ("native", "click_at"),
        ], (
            cell,
            [(x.automaton, x.scope) for x in reqs],
        )
        assert all("tools" in x.body for x in reqs)
        q = r.p2(WRONG_TAG, cell)
        assert (q["format"], q["requests"]) == ("native", 2)
        assert (q["parse"], q["call"], q["schema_valid"], q["exact"]) == (
            "ok",
            _gt(WRONG_TAG),
            True,
            True,
        )
    assert len(r.sent(WRONG_TAG, "cc", "p2")) == 3
    derived = _tp().derive(r.lines()[1:], "draft_only:native")
    (d,) = [x for x in derived if x["case"] == WRONG_TAG]
    assert (d["called"], d["schema_valid"], d["exact"]) == (True, False, False)
    for cell in NATIVE_CELLS:
        derived = _tp().derive(r.lines()[1:], f"cd_on_invalid:{cell}")
        (d,) = [x for x in derived if x["case"] == WRONG_TAG]
        assert (d["called"], d["schema_valid"], d["exact"], d["pass2_sent"]) == (
            True,
            True,
            True,
            True,
        ), cell


def test_adversarial_a_cc_pass_1_answering_natively_goes_through_the_cc_picker(mixed: Lazy) -> None:
    """Criterion 4, adversarial: in a cc row a ``<|tool_call>`` route keeps today's handling -- the JSON scorer's parse
    reason (never ``wrong_tag`` nor a native parse), the cc name picker under the JSON grammars, no ``tools``."""
    from local_astra.toolcall import router, score

    r = mixed.get()
    row = r.p1(CC_NATIVE_TAG, "cc")
    assert row["route"]["tag"] == "native" and row["route"]["draft_name"] is None
    draft = router.route(row["text"]).draft_call_text
    want = score.parse_call(draft)
    assert isinstance(want, score.ParseFailure), "premise: the JSON scorer refuses a native call"
    assert row["draft_parse"] == want.reason
    assert row["draft_parse"] != "wrong_tag"
    for cell in CC_CELLS:
        thought = cell.split("/")[0]
        reqs = [x for x in r.sent(CC_NATIVE_TAG, "cc", "p2") if x.shape == thought]
        assert [(x.automaton, x.scope) for x in reqs] == [
            ("json", "name"),
            ("json", "google_search"),
        ]
        assert not any("tools" in x.body for x in reqs)
        assert r.p2(CC_NATIVE_TAG, cell)["format"] == "cc"


def test_a_header_name_holding_48_does_not_move_the_native_call(tmp_path: Path) -> None:
    """T-0084a review note: native T2's header name admits 48; the call is the one after 101 (closed thoughts are
    removed first), so ``<|channel><|tool_call>\\n<channel|><|tool_call>call:...`` still scores the call."""
    sc = scenario()
    call = ts.native_call("google_search", {"query": "british museum"})
    sc.edits[("search-02", "native", "T2")] = {"text": "<|channel>" + OPEN + "\n<channel|>" + call}
    cell = ts.native_cell("T2/named/perstep_mar")
    r = run_scenario(tmp_path, [cell], ids=["search-02"], sc=sc)
    assert r.result.exit_code == 0, r.result.stopped
    assert ("search-02", "native", "T2") in r.edited
    row = r.p2("search-02", cell)
    assert (row["parse"], row["call"]) == ("ok", _gt("search-02"))
    assert row["exact"] is True


# =========================================================================== criterion 5: seeds in the driver


def _native_seed(cid: str, thought: str) -> list[int]:
    """The hand-built native seed of a scenario case (H, T = ``<|tool_call>``/``""``, the raw draft, S = 50)."""
    text = scenario().native_pass1_text(cid)
    ids = [*_encode(text), 1]  # what the fake serves: encode(text) + EOS
    i = text.index(OPEN)
    draft = text[i + len(OPEN) : text.index(RESPONSE)]
    head = {
        "T1": ids[: ids.index(101) + 1],
        "T2": [100, 45518, 107, 101],
        "T4": [],
    }[thought]
    lead = {"T1": OPEN, "T2": OPEN, "T4": ""}[thought]
    return (head + _encode(lead + draft.rstrip()) + [50])[:256]


def test_native_seeded_requests_carry_the_native_seed(seeded: Lazy) -> None:
    """Criterion 5 in the driver: a native seed cell sends the hand-built native seed on its request; the picker case
    sends none."""
    r = seeded.get()
    for cid in ("search-02", "type-01"):
        for thought in ts.THOUGHTS:
            (x,) = [y for y in r.sent(cid, "native", "p2") if y.shape == thought]
            keys = cd_keys(x.body)
            assert "astra_cd_draft" in keys, (cid, thought)
            from local_astra.eval.cd import replay

            seed = _native_seed(cid, thought)
            assert replay.accepts(ts.compiled_native(x.scope, thought), seed), "premise"
            got = r.seeds[[y.index for y in r.log if cd_keys(y.body)].index(x.index)]
            assert got is not None and got["draft"] == seed, (cid, thought)
    assert not any(
        "astra_cd_draft" in cd_keys(x.body) for x in r.sent(NATIVE_PICKER, "native", "p2")
    )


def test_native_seeded_rows_record_the_t0081b_seed_fields(seeded: Lazy) -> None:
    """Criterion 7: native seeded rows record the T-0081b seed fields (``seed_ids`` the hand-built seed); the picker
    row records ``seed: null`` / ``seed_reason: "picker"``."""
    r = seeded.get()
    sources = {"T1": "t1_thought", "T2": "t2_header", "T4": "t4_call"}
    for cid in ("search-02", "type-01"):
        for thought in ts.THOUGHTS:
            row = r.p2(cid, ts.native_cell(f"{thought}/named/{SEED}"))
            seed = _native_seed(cid, thought)
            assert row["format"] == "native"
            assert (row["seed_ids"], row["seed_len"]) == (seed, len(seed))
            assert (row["seed_source"], row["seed_trimmed"]) == (sources[thought], 0)
            assert (row["seed_noise"], row["seed_start_step"]) == (0.0, 24)
            assert row["steps"][0] == SEEDED_STEPS - 24
    for thought in ts.THOUGHTS:
        row = r.p2(NATIVE_PICKER, ts.native_cell(f"{thought}/named/{SEED}"))
        assert (row["seed"], row["seed_reason"]) == (None, "picker")


def test_a_wrong_tag_case_on_a_seed_cell_is_picked_and_not_seeded(seeded: Lazy) -> None:
    """T-0084d criterion 7: on a native seed cell the wrong_tag case goes through the picker, so it carries no seed:
    ``seed: null`` / ``seed_reason: "picker"``, no ``astra_cd_draft`` on its requests, and its pass 2 is sent (two
    requests, the picker then ``click_at``)."""
    r = seeded.get()
    for thought in ts.THOUGHTS:
        row = r.p2(WRONG_TAG, ts.native_cell(f"{thought}/named/{SEED}"))
        assert (row["seed"], row["seed_reason"], row["requests"]) == (None, "picker", 2), thought
    sent = r.sent(WRONG_TAG, "native", "p2")
    assert len(sent) == 6
    assert not any("astra_cd_draft" in cd_keys(x.body) for x in sent)


# =========================================================================== criterion 6: served-prompt guards


GUARD_CELLS = [
    "T1/named/perstep_mar",
    "T4/named/perstep_mar",
    *(ts.native_cell(f"{t}/named/perstep_mar") for t in ts.THOUGHTS),
]


def _guard_run(
    tmp: Path, edits: Mapping[tuple[str, str, str], Mapping[str, Any]], **fake_kw: Any
) -> Run:
    sc = scenario()
    sc.edits.update({k: dict(v) for k, v in edits.items()})
    return run_scenario(tmp, GUARD_CELLS, ids=["search-02"], sc=sc, fake_kw=fake_kw)


def _assert_stopped(r: Run, reason: str, at: int) -> None:
    """A stop: exit 1, ``stopped`` names the guard, the trailer carries it (report written), and the request at
    index ``at`` is the last one sent."""
    assert r.result.exit_code == 1, (r.result.exit_code, r.result.stopped)
    assert f"prompt_guard: {reason}" in str(r.result.stopped), r.result.stopped
    trailer = [x for x in r.lines() if x.get("kind") == "trailer"]
    assert trailer and f"prompt_guard: {reason}" in str(trailer[-1].get("stopped"))
    assert len(r.requests) == at + 1, f"{len(r.requests) - at - 1} requests sent after the stop"


def _drop_one(token: int) -> Callable[[list[int]], list[int]]:
    def edit(ids: list[int]) -> list[int]:
        i = ids.index(token)
        return ids[:i] + ids[i + 1 :]

    return edit


def _change_middle(ids: list[int]) -> list[int]:
    out = list(ids)
    k = len(out) // 2
    out[k] = 236770 if out[k] != 236770 else 236771
    return out


KEY_P1 = ("search-02", "native", "p1")


@pytest.mark.parametrize(
    ("name", "key", "edit", "reason"),
    [
        ("p1_missing", KEY_P1, {"omit_prompt_token_ids": True}, "missing"),
        (
            "t1_missing",
            ("search-02", "native", "T1"),
            {"omit_prompt_token_ids": True},
            "missing",
        ),
        ("p1_declarations_dropped", KEY_P1, {"drop_declarations": True}, "declarations"),
        ("p1_one_46_missing", KEY_P1, {"prompt_ids_edit": _drop_one(46)}, "declarations"),
        ("p1_one_47_missing", KEY_P1, {"prompt_ids_edit": _drop_one(47)}, "declarations"),
        (
            "t1_prefix",
            ("search-02", "native", "T1"),
            {"prompt_ids_edit": _change_middle},
            "prefix",
        ),
        (
            "t2_prefix",
            ("search-02", "native", "T2"),
            {"prompt_ids_edit": _change_middle},
            "prefix",
        ),
        (
            "t4_continuation",
            ("search-02", "native", "T4"),
            {"prompt_ids_edit": lambda ids: ids[:-1]},
            "continuation",
        ),
    ],
)
def test_a_served_prompt_guard_failure_is_a_stop(
    tmp_path: Path, name: str, key: tuple[str, str, str], edit: Mapping[str, Any], reason: str
) -> None:
    """Criterion 6, one row per guard: the failing native request is the last one sent; exit 1, ``stopped`` set,
    the report written."""
    r = _guard_run(tmp_path, {key: edit})
    assert key in r.edited, f"fixture: the {key} request was never sent"
    _assert_stopped(r, reason, r.edited[key])


def test_a_server_dropping_the_declarations_stops_at_the_first_native_pass_1(
    tmp_path: Path,
) -> None:
    """Criterion 6: a server started with ``--exclude-tools-when-tool-choice-none`` (the fake's
    ``drop_declarations``) -> ``prompt_guard: declarations`` at the native pass 1 (index 1: cc goes first on case 0)."""
    r = _guard_run(tmp_path, {}, drop_declarations=True)
    _assert_stopped(r, "declarations", 1)


@pytest.mark.parametrize(
    ("name", "key", "edit"),
    [
        ("cc_p1_missing", ("search-02", "cc", "p1"), {"omit_prompt_token_ids": True}),
        ("cc_t1_prefix", ("search-02", "cc", "T1"), {"prompt_ids_edit": _change_middle}),
        ("cc_t4_not_48", ("search-02", "cc", "T4"), {"prompt_ids_edit": lambda ids: ids[:-1]}),
        ("native_t4_middle", ("search-02", "native", "T4"), {"prompt_ids_edit": _change_middle}),
    ],
)
def test_guards_are_not_too_wide(
    tmp_path: Path, name: str, key: tuple[str, str, str], edit: Mapping[str, Any]
) -> None:
    """Criterion 6, the other direction: cc requests are never guarded (a missing prompt, a changed T1 prompt, a T4
    prompt not ending in 48 all pass), and a native T4 prompt differing from pass 1's (it always does) but ending in
    48 passes: the run completes with exit 0."""
    r = _guard_run(tmp_path, {key: edit})
    assert key in r.edited, f"fixture: the {key} request was never sent"
    assert r.result.exit_code == 0, (name, r.result.stopped)
    assert len(r.requests) == 2 + len(GUARD_CELLS)


def test_the_prefix_guard_compares_with_the_served_pass_1_ids(tmp_path: Path) -> None:
    """Criterion 6: a native pass 1 whose served prompt has one id appended keeps 5 x 46 / 5 x 47 (no
    ``declarations`` stop), and its T1/T2 requests, rendered from the same body, no longer equal it -> ``prefix`` at
    the first native T1/T2 request, not at pass 1."""
    r = _guard_run(tmp_path, {KEY_P1: {"prompt_ids_edit": lambda ids: [*ids, 107]}})
    first = min(x.index for x in r.log if x.fmt == "native" and x.shape in ("T1", "T2"))
    _assert_stopped(r, "prefix", first)


def test_guarded_runs_pass_when_nothing_is_wrong(tmp_path: Path) -> None:
    """The unedited guard scenario completes (the positive control of the rows above)."""
    r = _guard_run(tmp_path, {})
    assert r.result.exit_code == 0, r.result.stopped
    assert len(r.requests) == 2 + len(GUARD_CELLS)


# =========================================================================== criterion 7: rows


def test_every_row_records_its_format(mixed: Lazy) -> None:
    """Criterion 7: every pass-1 and pass-2 row has ``format``; one pass-1 row per (case, format); a pass-2 row's
    format is its cell's."""
    r = mixed.get()
    rows = [x for x in r.lines()[1:] if x.get("pass") in (1, 2)]
    assert all(x.get("format") in ("cc", "native") for x in rows), [x.get("format") for x in rows]
    p1 = sorted((x["case"], x["format"]) for x in rows if x["pass"] == 1)
    assert p1 == sorted((c, f) for c in CASES for f in ("cc", "native"))
    for x in rows:
        if x["pass"] == 2:
            assert x["format"] == ("native" if x["cell"].startswith("native/") else "cc")


def test_native_pass_1_rows_record_the_prompt_and_answer_fields(mixed: Lazy) -> None:
    """Criterion 7: native pass-1 rows record ``prompt_len`` (the served prompt's length), ``text_after_close`` (""),
    ``spoken_prefix_chars`` (counted by hand), ``open_thought_call`` (false) and ``draft_close`` ("")."""
    r = mixed.get()
    for cid in NORMAL:
        row = r.p1(cid, "native")
        (sent,) = r.sent(cid, "native", "p1")
        served = r.prompt_ids[sent.index]
        assert served is not None
        assert row["prompt_len"] == len(served)
        assert row["text_after_close"] == ""
        assert row["spoken_prefix_chars"] == len(f"On it ({cid}). ")
        assert not row["open_thought_call"]
        assert row["draft_close"] == ""


def test_native_rows_with_text_after_the_close_and_an_open_thought_call(tmp_path: Path) -> None:
    """Criterion 7: ``text_after_close`` holds the answer text after ``<tool_call|>`` (before the stop), and
    ``open_thought_call`` is set when the call is written inside a thought that never closes (no route)."""
    sc = scenario()
    sc.native_pass1["search-02"] = (
        "<|channel>thought\nok<channel|>Searching. "
        + ts.native_call("google_search", {"query": "british museum"})
        + "Done."
        + RESPONSE
    )
    sc.native_pass1[NONE] = "<|channel>thought\nI will " + ts.native_call(
        "google_search", {"query": "x"}
    )
    r = run_scenario(
        tmp_path, [ts.native_cell("T4/named/perstep_mar")], ids=["search-02", NONE], sc=sc
    )
    assert r.result.exit_code == 0, r.result.stopped
    row = r.p1("search-02", "native")
    assert row["text_after_close"] == "Done."
    assert row["spoken_prefix_chars"] == len("Searching. ")
    assert not row["open_thought_call"]
    open_row = r.p1(NONE, "native")
    assert open_row["route"]["kind"] == "none"
    assert open_row["open_thought_call"]


# =========================================================================== criterion 8: derive is format-aware


def _rows(*, with_format: bool = True) -> list[dict[str, Any]]:
    """Hand-made rows for one tool case A: the cc draft valid and exact, the native draft invalid; one pass-2 row
    per format at T4 (cc: valid, not exact; native: valid, exact)."""
    rows: list[dict[str, Any]] = [
        {
            "pass": 1,
            "format": "cc",
            "case": "A",
            "kind": "tool",
            "tool": "click_at",
            "route": {"kind": "toolcall", "tag": "cc"},
            "wall_ms": 100.0,
            "draft_parse": "ok",
            "draft_call": {"name": "click_at", "arguments": {"x": 1, "y": 2}},
            "draft_schema_valid": True,
            "draft_exact": True,
        },
        {
            "pass": 1,
            "format": "native",
            "case": "A",
            "kind": "tool",
            "tool": "click_at",
            "route": {"kind": "toolcall", "tag": "native"},
            "wall_ms": 200.0,
            "draft_parse": "no close",
            "draft_call": None,
            "draft_schema_valid": False,
            "draft_exact": False,
        },
        {
            "pass": 2,
            "format": "cc",
            "case": "A",
            "cell": "T4/named/perstep_mar",
            "parse": "ok",
            "call": {"name": "click_at", "arguments": {"x": 9, "y": 9}},
            "schema_valid": True,
            "exact": False,
            "cs": True,
            "wall_ms": 10.0,
            "steps": [5],
        },
        {
            "pass": 2,
            "format": "native",
            "case": "A",
            "cell": "native/T4/named/perstep_mar",
            "parse": "ok",
            "call": {"name": "click_at", "arguments": {"x": 1, "y": 2}},
            "schema_valid": True,
            "exact": True,
            "cs": True,
            "wall_ms": 20.0,
            "steps": [3],
        },
    ]
    if not with_format:
        for x in rows:
            if x["format"] == "cc":
                del x["format"]
    return rows


#: policy -> (schema_valid, exact, e2e_wall_ms, pass2_sent), by hand from ``_rows``
DERIVED = {
    "draft_only": (True, True, 100.0, False),
    "draft_only:native": (False, False, 200.0, False),
    "cd_on_invalid:T4/named/perstep_mar": (True, True, 100.0, False),
    "cd_on_invalid:native/T4/named/perstep_mar": (True, True, 220.0, True),
    "cd_always:T4/named/perstep_mar": (True, False, 110.0, True),
    "cd_always:native/T4/named/perstep_mar": (True, True, 220.0, True),
}


@pytest.mark.parametrize("with_format", [True, False], ids=["format", "pre_t0084c_cc_rows"])
@pytest.mark.parametrize("policy", list(DERIVED))
def test_derive_pairs_each_policy_with_its_own_formats_pass_1(
    policy: str, with_format: bool
) -> None:
    """Criterion 8: ``draft_only`` uses the cc pass-1 row, ``draft_only:native`` the native one; ``cd_*:<cell>`` pairs
    a cell with its own format's pass-1 row; one derived row per case. Criterion 7: cc rows without ``format`` (older
    artifacts) read as cc."""
    _tp()
    out = ts.twopass().derive(_rows(with_format=with_format), policy)
    assert len(out) == 1, out
    (d,) = out
    got = (d["schema_valid"], d["exact"], d["e2e_wall_ms"], d["pass2_sent"])
    assert got == DERIVED[policy], (policy, got)


def _stored(stage: str) -> tuple[Path, Path]:
    common = subprocess.run(
        ["git", "-C", str(ts.REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    d = Path(common).parent / "logs" / f"t0080-toolcall-{stage}-0dc9134"
    arts = sorted(d.glob(f"*-toolcall-{stage}.jsonl"))
    assert len(arts) == 1, f"T-0080c's stored {stage} artifact is not at {d}"
    return arts[0], d / f"report-{stage}.json"


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_derive_and_select_reproduce_t0080cs_stored_tables(stage: str) -> None:
    """Criterion 8: on T-0080c's stored artifacts (rows without ``format``) the report's policy tables and selection
    equal the recorded ones (``report-<stage>.json`` at 0dc9134: stage 1 chose ``T1/named/perstep_mar``, stage 2
    ``cd_on_invalid:T1/named/perstep_mar+seed``). Green before the implementation by design: it guards the change."""
    art, stored = _stored(stage)
    want = json.loads(stored.read_text())
    got = ts.twopass().report([art])
    assert got["policies"] == want["policies"]
    assert got["selection"] == want["selection"]
    choice = {"dev": "T1/named/perstep_mar", "test": "cd_on_invalid:T1/named/perstep_mar+seed"}
    assert got["selection"]["choice"] == choice[stage]


# =========================================================================== criterion 9: order


def test_pass_1_format_order_alternates_by_case_index(mixed: Lazy) -> None:
    """Criterion 9: cc first on even case indexes, native first on odd ones; both pass 1s precede the case's pass 2."""
    r = mixed.get()
    for ci, cid in enumerate(CASES):
        reqs = [x for x in r.log if x.case == cid]
        p1 = [x.fmt for x in reqs if x.kind == "p1"]
        assert p1 == (["cc", "native"] if ci % 2 == 0 else ["native", "cc"]), (ci, cid, p1)
        kinds = [x.kind for x in reqs]
        assert kinds == sorted(kinds), (cid, kinds)  # "p1" < "p2"
    assert [x.case for x in r.log if x.kind == "p1"][::2] == CASES


def test_pass_2_order_is_the_williams_row_over_all_cells_skipping_unrouted_formats(
    mixed: Lazy,
) -> None:
    """Criterion 9: the pass-2 cells of a case follow the R-W Williams row (k = 6, written out by hand) over the
    full cell list, indexed by the count of cases routed in at least one format; cells of a format that did not
    route are skipped (since T-0084d criterion 7 a ``wrong_tag`` case routes natively through the picker)."""
    r = mixed.get()
    for cid, row in WILLIAMS_INDEX.items():
        want = [MIXED[i] for i in WILLIAMS_6[row]]
        if cid not in CC_ROUTED:
            want = [c for c in want if c.startswith("native/")]
        if cid not in NATIVE_ROUTED:
            want = [c for c in want if not c.startswith("native/")]
        got: list[str] = []
        for x in r.log:
            if x.case == cid and x.kind == "p2":
                cell = _cell_of(x)
                if not got or got[-1] != cell:
                    got.append(cell)
        assert got == want, (cid, got, want)
    assert not [x for x in r.log if x.case == NONE and x.kind == "p2"]


# =========================================================================== criterion 11: negative


def _norm_body(body: Mapping[str, Any]) -> str:
    b = copy.deepcopy(dict(body))
    if isinstance(b.get("vllm_xargs"), dict) and "astra_cd_record_tag" in b["vllm_xargs"]:
        b["vllm_xargs"]["astra_cd_record_tag"] = "<tag>"
    return json.dumps(b, sort_keys=True, ensure_ascii=False)


def norm_rows(lines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in lines[1:]:
        if row.get("pass") not in (1, 2):
            continue
        x = {k: v for k, v in row.items() if k not in VOLATILE}
        if isinstance(x.get("knobs"), dict) and "record_tag" in x["knobs"]:
            x["knobs"] = {**x["knobs"], "record_tag": "<tag>"}
        out.append(x)
    return out


def golden_of(r: Run) -> dict[str, Any]:
    """The cc-only run's observable: request count, normalised body shas in order, normalised rows."""
    return {
        "requests": len(r.requests),
        "body_sha256": [_sha(_norm_body(b)) for b in r.requests],
        "rows": norm_rows(r.lines()),
    }


def test_a_cc_only_run_is_bit_identical_to_main(tmp_path: Path) -> None:
    """Criterion 11 (golden): the cc-only run of this scenario equals ``main`` at 9737368 -- request count, every
    request body (sha of the normalised body), every row field by field -- apart from the added ``format: "cc"``,
    which every row must carry."""
    r = run_scenario(tmp_path, list(CC_CELLS), check=False)
    assert r.result.exit_code == 0, r.result.stopped
    want = json.loads(GOLDEN.read_text())
    got = golden_of(r)
    assert got["requests"] == want["requests"]
    assert got["body_sha256"] == want["body_sha256"]
    assert len(got["rows"]) == len(want["rows"])
    missing = [(x.get("pass"), x.get("case")) for x in got["rows"] if x.get("format") != "cc"]
    assert missing == [], f"rows without format 'cc': {missing}"
    for g, w in zip(got["rows"], want["rows"], strict=True):
        g = {k: v for k, v in g.items() if k != "format"}
        assert sorted(g) == sorted(w), (w.get("pass"), w.get("case"), set(g) ^ set(w))
        for k in w:
            assert g[k] == w[k], (w.get("pass"), w.get("case"), w.get("cell"), k)


def test_the_native_run_stays_on_loopback(mixed: Lazy) -> None:
    """Criterion 11: every connection of the mixed run goes to the loopback fake."""
    r = mixed.get()
    assert r.connects
    assert {a[0] for a in r.connects if isinstance(a, tuple)} <= {"127.0.0.1", "::1", "localhost"}


UNCHANGED = (
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/decoding",
    "local_astra/conductor",
    "local_astra/contracts",
    "plugin",
    *(
        f"local_astra/eval/cd/{m}.py"
        for m in ("runner", "arms", "calibration", "cuts", "hardstop", "replay", "records")
    ),
    "local_astra/bench/thresholds.py",
    "scripts/locked",
)
#: the one file under ``local_astra/`` T-0084c owns
OWNED = {"local_astra/eval/cd/twopass.py"}
#: T-0084c's merge commit 9e84f21; BASE = ``git merge-base 9e84f21^1 9e84f21^2``, TIP = ``9e84f21^2`` (T-0094 criterion
#: 2). Over this closed range no later-task exemption is needed: T-0084d, T-0085 and T-0086 came later.
T0084C_MERGE = "9e84f21"
T0084C_BASE = "9737368"
T0084C_TIP = "8ccbcc4"


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ts.REPO), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _t84c_changes(base: str, tip: str) -> tuple[list[str], list[str]]:
    """Criterion 11's two halves over ``base..tip``: changes under ``UNCHANGED``; changes under ``local_astra/``."""
    return gr.changed(base, tip, *UNCHANGED), gr.changed(base, tip, "local_astra")


def test_t0084c_changes_nothing_outside_twopass() -> None:
    """Criterion 11 (scope) over T-0084c's closed range (T-0094): nothing under ``UNCHANGED`` differs between BASE and
    TIP and the only changed file under ``local_astra/`` is ``twopass.py``. The untracked-file half of the old guard
    is retired: it only meant something while the branch was open."""
    gr.prove_range(T0084C_BASE, T0084C_TIP, T0084C_MERGE)
    changed, la = _t84c_changes(T0084C_BASE, T0084C_TIP)
    assert changed == [], changed
    assert set(la) <= OWNED, sorted(set(la) - OWNED)


def test_t0084c_s_range_changes_exactly_its_owned_file() -> None:
    """T-0094 criterion 5: the range reports ``eval/cd/twopass.py`` as the only ``local_astra`` change, and it is
    the owned file."""
    gr.require_commits(T0084C_BASE, T0084C_TIP)
    _changed, la = _t84c_changes(T0084C_BASE, T0084C_TIP)
    assert la == ["local_astra/eval/cd/twopass.py"], la
    assert set(la) <= OWNED


def test_the_t0084c_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the ``UNCHANGED`` half reports the router (T-0085) and
    the ``local_astra`` half a file outside ``OWNED`` (``executor.py``, T-0089/T-0092)."""
    gr.require_commits(T0084C_BASE, gr.LATER)
    changed, la = _t84c_changes(T0084C_BASE, gr.LATER)
    assert "local_astra/toolcall/router.py" in changed, changed
    assert "local_astra/conductor/executor.py" in set(la) - OWNED, la


# =========================================================================== rulings at activation (orchestrator)


def _wrong_tag_p1() -> dict[str, Any]:
    return {
        "pass": 1,
        "format": "native",
        "case": "A",
        "kind": "tool",
        "tool": "click_at",
        "route": {"kind": "toolcall", "tag": "cc"},
        "wall_ms": 200.0,
        "draft_parse": "wrong_tag",
        "draft_call": None,
        "draft_schema_valid": False,
        "draft_exact": False,
    }


def test_wrong_tag_under_cd_on_invalid_derives_from_pass_2() -> None:
    """Ruling C8 as amended by H-T84-0 (T-0084d criterion 7; this row replaced C8's "no pass 2, not missing"): a native
    pass 1 routed with ``<cc>`` (``wrong_tag``) is an invalid draft, so ``cd_on_invalid:native/T1/named/end_projection``
    takes the cell's pass-2 row: schema-valid and exact here, pass 2 sent, e2e = 200 + 300. ``draft_only:native`` keeps
    the draft's outcome (called, invalid, not exact). Without a pass-2 row the case is ``missing`` like any other
    invalid draft whose pass 2 did not arrive."""
    _tp()
    cell = "native/T1/named/end_projection"
    p2 = {
        "pass": 2,
        "format": "native",
        "case": "A",
        "cell": cell,
        "parse": "ok",
        "call": {"name": "click_at", "arguments": {"x": 1, "y": 2}},
        "schema_valid": True,
        "exact": True,
        "cs": True,
        "error": None,
        "wall_ms": 300.0,
        "steps": [7],
        "requests": 2,
    }
    (d,) = ts.twopass().derive([_wrong_tag_p1(), p2], f"cd_on_invalid:{cell}")
    got = (d["called"], d["schema_valid"], d["exact"], d["pass2_sent"], d["missing"])
    assert got == (True, True, True, True, False), got
    assert d["e2e_wall_ms"] == 500.0
    (d,) = ts.twopass().derive([_wrong_tag_p1(), p2], "draft_only:native")
    assert (d["called"], d["schema_valid"], d["exact"], d["pass2_sent"]) == (
        True,
        False,
        False,
        False,
    )
    (d,) = ts.twopass().derive([_wrong_tag_p1()], f"cd_on_invalid:{cell}")
    assert (d["schema_valid"], d["missing"]) == (False, True)


def test_a_native_pass_1_routed_genui_is_not_wrong_tag(tmp_path: Path) -> None:
    """Ruling (C10): a native-format pass 1 answering with ``<genui>`` routes ``genui`` (not ``wrong_tag``), sends no
    native pass 2, and ``draft_only:native`` counts it as not called."""
    sc = scenario()
    sc.native_pass1["search-02"] = (
        '<|channel>thought\nok<channel|>Here is a card. <genui>{"surface": "card"}'
    )
    r = run_scenario(tmp_path, list(NATIVE_CELLS), ids=["search-02"], sc=sc)
    assert r.result.exit_code == 0, r.result.stopped
    row = r.p1("search-02", "native")
    assert (row["route"]["kind"], row["route"]["tag"]) == ("genui", "genui")
    assert row["draft_parse"] != "wrong_tag"
    assert r.sent("search-02", "native", "p2") == []
    (d,) = ts.twopass().derive(r.lines()[1:], "draft_only:native")
    assert (d["called"], d["exact"]) == (False, False)


# =========================================================================== review gaps (2026-10-10, rulings C13/C14)


@pytest.mark.parametrize(
    ("name", "key", "edit", "reason"),
    [
        ("t1_missing", ("search-02", "native", "T1"), {"omit_prompt_token_ids": True}, "missing"),
        ("t1_prefix", ("search-02", "native", "T1"), {"prompt_ids_edit": _change_middle}, "prefix"),
        ("t2_prefix", ("search-02", "native", "T2"), {"prompt_ids_edit": _change_middle}, "prefix"),
        (
            "t4_continuation",
            ("search-02", "native", "T4"),
            {"prompt_ids_edit": lambda ids: ids[:-1]},
            "continuation",
        ),
    ],
)
def test_a_pass_2_guard_stops_before_the_records_are_awaited(
    tmp_path: Path, name: str, key: tuple[str, str, str], edit: Mapping[str, Any], reason: str
) -> None:
    """Review gap 1: the guard runs on the response, before ``await_records`` -- with the failing request's records
    held back (``hold_records``), a guard placed after the wait would flush (E9) before stopping. No flush is sent."""
    r = _guard_run(tmp_path, {key: {**edit, "hold_records": True}})
    assert key in r.edited, f"fixture: the {key} request was never sent"
    _assert_stopped(r, reason, r.edited[key])
    assert r.flushes == [], f"{len(r.flushes)} flush(es) sent before the guard stopped the run"


def test_a_t4_prompt_containing_48_but_not_ending_in_it_is_a_continuation_stop(
    tmp_path: Path,
) -> None:
    """Review gap 2: the served T4 prompt ``ids + [106]`` holds 48 but does not end in it -> ``continuation``."""
    key = ("search-02", "native", "T4")
    r = _guard_run(tmp_path, {key: {"prompt_ids_edit": lambda ids: [*ids, 106]}})
    assert key in r.edited
    _assert_stopped(r, "continuation", r.edited[key])


def test_the_williams_index_counts_cases_routed_in_either_format(tmp_path: Path) -> None:
    """Review gap 3: ``scroll-02`` routes only natively and comes first, so it is routed case 0 and ``search-02``
    takes Williams row 1 (``[1, 2, 0, 3, 5, 4]``); counting cc-routed cases only would give ``search-02`` row 0."""
    r = run_scenario(tmp_path, list(MIXED), ids=[NATIVE_ONLY, "search-02"])
    assert r.result.exit_code == 0, r.result.stopped
    for cid, row in ((NATIVE_ONLY, 0), ("search-02", 1)):
        want = [MIXED[i] for i in WILLIAMS_6[row]]
        if cid == NATIVE_ONLY:
            want = [c for c in want if c.startswith("native/")]
        got: list[str] = []
        for x in r.log:
            if x.case == cid and x.kind == "p2":
                cell = _cell_of(x)
                if not got or got[-1] != cell:
                    got.append(cell)
        assert got == want, (cid, got, want)


def test_c6_c13_native_pass_1_field_edges(tmp_path: Path) -> None:
    """Review gap 5 (C6, C13): ``text_after_close`` keeps whitespace as written; an unclosed thought without a call has
    ``open_thought_call`` false; a ``wrong_tag`` draft holding ``<tool_call|>`` has ``text_after_close`` None; a
    ``wrong_tag`` row keeps the router's ``draft_close`` ``"</cc>"``."""
    sc = scenario()
    head = "<|channel>thought\nok<channel|>"
    sc.native_pass1["search-02"] = (
        head
        + "Searching. "
        + ts.native_call("google_search", {"query": "british museum"})
        + " Done. \n"
        + RESPONSE
    )
    sc.native_pass1[NONE] = "<|channel>thought\nI am still thinking about it"
    click = ts.call_text("click_at", {"x": 500, "y": 499})
    sc.native_pass1[WRONG_TAG] = head + "Clicking. " + ts.CC + click + CLOSE + "x"
    sc.native_pass1["click-01"] = head + "Clicking. " + ts.CC + click + "</cc>"
    r = run_scenario(
        tmp_path,
        [ts.native_cell("T4/named/perstep_mar")],
        ids=["search-02", NONE, WRONG_TAG, "click-01"],
        sc=sc,
    )
    assert r.result.exit_code == 0, r.result.stopped
    assert r.p1("search-02", "native")["text_after_close"] == " Done. \n"
    none_row = r.p1(NONE, "native")
    assert none_row["route"]["kind"] == "none" and none_row["open_thought_call"] is False
    wt = r.p1(WRONG_TAG, "native")
    assert (wt["draft_parse"], wt["text_after_close"]) == ("wrong_tag", None)
    closed = r.p1("click-01", "native")
    assert (closed["draft_parse"], closed["draft_close"]) == ("wrong_tag", "</cc>")


def test_a_cc_union_cell_uses_the_json_union_grammar(tmp_path: Path) -> None:
    """Review gap 5: a cc ``union`` cell's request carries the JSON union automaton, not the native one."""
    from tests.cd import _wire_support as ws

    r = run_scenario(tmp_path, ["T4/union/perstep_mar"], ids=["search-02"])
    cd = [x for x in r.log if x.kind == "p2"]
    assert cd, "no union request was sent"
    payload = cd_keys(cd[0].body)["astra_cd_automaton"]
    json_union = ws.encode(ts.compiled("union", "T4"))
    native_union = ws.encode(ts.compiled_native("union", "T4"))
    assert json_union != native_union, "premise"
    assert payload == json_union


# =========================================================================== T-0085: the continuation agrees with the template
#
# docs/tasks/T-0085.md criteria 3 and 6 on the loopback fake, whose continuation check now renders as ``transformers``
# does (a T4 the pinned template changes is a 400 there, as on vLLM). The click-01 text is the T-0084e stop's native
# pass 1, verbatim. API pins where the task is silent (reported): the stop string is ``continuation_guard: <reason>``
# with reason ``template`` or ``marker``; when both hold (a channel marker in the spoken part is both), the reason is
# ``template`` -- criterion 3 lists (a) first.

from tests.unit._toolcall_support import (  # noqa: E402
    CLICK01_CC_TEXT,
    CLICK01_NATIVE_TEXT,
    CLICK01_SPOKEN,
)

T85_NATIVE_T4 = ts.native_cell("T4/named/perstep_mar")
T85_CLEAN = CLICK01_SPOKEN + OPEN


def _t4_sent(r: Run, fmt: str = "native") -> list[Logged]:
    return [x for x in r.log if x.fmt == fmt and x.shape == "T4"]


def _stopped_by_guard(r: Run, reason: str) -> None:
    """Criterion 3: exit 1, ``stopped`` names ``continuation_guard: <reason>``, the trailer carries it (the report is
    written), and the fake refused nothing (the guard stops before a 400)."""
    assert r.result is not None
    assert r.result.exit_code == 1, (r.result.exit_code, r.result.stopped)
    assert f"continuation_guard: {reason}" in str(r.result.stopped), r.result.stopped
    trailer = [x for x in r.lines() if x.get("kind") == "trailer"]
    assert trailer and f"continuation_guard: {reason}" in str(trailer[-1].get("stopped"))
    assert r.rejected == [], "a refused continuation reached the server (a 400, not the guard)"


def _mutant_route(mp: pytest.MonkeyPatch) -> None:
    """A router that keeps the stray close before the tag (the 9aee66d behaviour on click-01)."""
    from local_astra.toolcall import router

    real = router.route

    def mutant(text: str, **kw: Any) -> Any:
        r = real(text, **kw)
        if r.kind == "toolcall" and "<channel|>" + router.TAGS[r.tag] in text:
            return dataclasses.replace(r, spoken_prefix=r.spoken_prefix + "<channel|>")
        return r

    mp.setattr(router, "route", mutant)


def test_t0085_the_click01_mixed_run_completes_and_continues_clean(tmp_path: Path) -> None:
    """Criterion 6: a mixed run (3 cc + 3 native cells) whose click-01 pass 1s are the T-0084e texts verbatim exits
    0; every native T4 request continues ``"I will click the Sign in button.<|tool_call>"``, its served prompt ends
    in 48, and the fake refused nothing."""
    sc = scenario()
    sc.native_pass1["click-01"] = CLICK01_NATIVE_TEXT + RESPONSE
    sc.pass1["click-01"] = CLICK01_CC_TEXT
    r = run_scenario(tmp_path, MIXED, ids=["search-02", "click-01"], sc=sc)
    assert r.result.exit_code == 0, (r.result.stopped, r.rejected[:1])
    assert r.rejected == []
    t4 = [x for x in _t4_sent(r) if x.case == "click-01"]
    assert len(t4) == 1, len(t4)
    for x in t4:
        assert x.body["messages"][-1] == {"role": "assistant", "content": T85_CLEAN}
        served = r.prompt_ids[x.index]
        assert served and served[-1] == 48, served[-6:] if served else served
    (cc_t4,) = [x for x in _t4_sent(r, "cc") if x.case == "click-01"]
    assert cc_t4.body["messages"][-1]["content"] == "I will click the Sign in button.\n<cc>"


def test_t0085_a_mutant_router_keeping_the_marker_stops_at_the_guard_not_a_400(
    tmp_path: Path,
) -> None:
    """Criteria 3(a) and 6: with a router that keeps the stray close, the native T4 is never sent; the run stops
    with ``continuation_guard: template`` (exit 1, report written), and no 400 happened."""
    sc = scenario()
    sc.native_pass1["click-01"] = CLICK01_NATIVE_TEXT + RESPONSE
    with pytest.MonkeyPatch.context() as mp:
        _mutant_route(mp)
        r = run_scenario(tmp_path, [T85_NATIVE_T4], ids=["click-01"], sc=sc)
    _stopped_by_guard(r, "template")
    assert _t4_sent(r) == [], "the guarded T4 was sent"
    assert len(r.requests) == 1  # the native pass 1 only


#: criterion 3(b): structural markers that are not stop ids (the pass-1 decode ends at 1, 50 and 106), each in the
#: spoken part, the template leaving it in place -- so only (b) holds; plus a marker made by removing a stray close.
T85_MARKER_SPOKEN = {
    "turn_open": "Sure.<|turn> ",
    "tool_call_close": "Sure.<tool_call|> ",
    "tool_open": "Sure.<|tool> ",
    "tool_close": "Sure.<tool|> ",
    "think": "Sure.<|think|> ",
    "bos": "Sure.<bos> ",
    "spliced_by_removal": "Sure.<tool_<channel|>call|> ",
}


@pytest.mark.parametrize("name", list(T85_MARKER_SPOKEN))
def test_t0085_a_structural_marker_in_the_spoken_part_stops_before_t4(
    tmp_path: Path, name: str
) -> None:
    """Criterion 3(b): a native pass 1 whose spoken part holds a structural marker -> the T4 request is not sent;
    ``continuation_guard: marker``, exit 1, report written."""
    sc = scenario()
    name_, args = ts.gt_call(ts.by_id()["click-01"])
    sc.native_pass1["click-01"] = (
        "<|channel>thought\nok<channel|>"
        + T85_MARKER_SPOKEN[name]
        + ts.native_call(name_, args)
        + RESPONSE
    )
    r = run_scenario(tmp_path, [T85_NATIVE_T4], ids=["click-01"], sc=sc)
    _stopped_by_guard(r, "marker")
    assert _t4_sent(r) == [], "the guarded T4 was sent"


@pytest.mark.parametrize("path", ["picker", "wrong_tag"])
def test_t0085_the_picker_t4_request_is_guarded(tmp_path: Path, path: str) -> None:
    """Criterion 3: "before any T4 request (routed or picker)" -- a native draft naming no v1 tool (picker) and a
    ``<cc>`` answer to the native pass 1 (``wrong_tag`` -> picker) with a marker in the spoken part send no T4 request."""
    sc = scenario()
    spoken = "Searching.<|think|> "
    if path == "picker":
        tail = OPEN + 'call:web_search{query:<|"|>banana<|"|>}' + CLOSE + RESPONSE
    else:
        name_, args = ts.gt_call(ts.by_id()["search-02"])
        tail = ts.CC + ts.call_text(name_, args)
    sc.native_pass1["search-02"] = "<|channel>thought\nok<channel|>" + spoken + tail
    r = run_scenario(tmp_path, [T85_NATIVE_T4], ids=["search-02"], sc=sc)
    _stopped_by_guard(r, "marker")
    assert _t4_sent(r) == []
    assert len(r.requests) == 1


def test_t0085_the_guard_is_not_too_wide(tmp_path: Path) -> None:
    """Criterion 3, the other direction: (1) the marker spoken part with only T1/T2 cells (no T4 continuation) runs
    to exit 0; (2) a spoken part with marker *pieces* and edge whitespace on a T4 cell runs to exit 0."""
    sc = scenario()
    name_, args = ts.gt_call(ts.by_id()["click-01"])
    call = ts.native_call(name_, args) + RESPONSE
    sc.native_pass1["click-01"] = "<|channel>thought\nok<channel|>Sure.<|think|> " + call
    cells = [ts.native_cell("T1/named/perstep_mar"), ts.native_cell("T2/named/perstep_mar")]
    r = run_scenario(tmp_path / "t12", cells, ids=["click-01"], sc=sc)
    assert r.result.exit_code == 0, r.result.stopped
    sc2 = scenario()
    sc2.native_pass1["click-01"] = (
        "<|channel>thought\nok<channel|>\n Pipes |> and <channel or <|chan are text. \n" + call
    )
    r2 = run_scenario(tmp_path / "t4", [T85_NATIVE_T4], ids=["click-01"], sc=sc2)
    assert r2.result.exit_code == 0, r2.result.stopped
    assert len(_t4_sent(r2)) == 1 and r2.rejected == []


# --------------------------------------------------------------------------- T-0085 criterion 8 (negative; green by design)

T85_OWNED = {"local_astra/toolcall/router.py", "local_astra/eval/cd/twopass.py"}
#: T-0085's merge commit 794161d; BASE = ``git merge-base 794161d^1 794161d^2``, TIP = ``794161d^2`` (T-0094 ruling L5,
#: R5). Not the declared fork point 3133fac (T-0094b moved the other T-0085 rows, which compared it with the live file,
#: to this BASE and TIP; 3133fac..b7f38ef leaves twopass.py byte-identical). Over this closed range no T-0086
#: exemption is needed: T-0086 came later.
T0085_MERGE = "794161d"
T0085_BASE = "b7f38ef"
T0085_TIP = "9063661"
T85_UNCHANGED = (
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/decoding",
    "local_astra/conductor",
    "local_astra/contracts",
    "plugin",
    "local_astra/bench/thresholds.py",
    "scripts/locked",
)


def _t85_changes(base: str, tip: str) -> tuple[list[str], list[str]]:
    """Criterion 8's two halves over ``base..tip``: changes under ``T85_UNCHANGED``; changes under ``local_astra/``."""
    return gr.changed(base, tip, *T85_UNCHANGED), gr.changed(base, tip, "local_astra")


def test_t0085_changes_nothing_outside_router_and_twopass() -> None:
    """Criterion 8 over T-0085's closed range (T-0094 R5): nothing under ``T85_UNCHANGED`` differs between BASE and TIP
    and the only changed files under ``local_astra/`` are the router and the driver. The untracked-file half of the
    old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0085_BASE, T0085_TIP, T0085_MERGE)
    changed, la = _t85_changes(T0085_BASE, T0085_TIP)
    assert changed == [], changed
    assert set(la) <= T85_OWNED, sorted(set(la) - T85_OWNED)


def test_the_t0085_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the ``T85_UNCHANGED`` half reports
    ``conductor/executor.py`` (T-0089/T-0092), and so does the ``local_astra`` half outside ``T85_OWNED``."""
    gr.require_commits(T0085_BASE, gr.LATER)
    changed, la = _t85_changes(T0085_BASE, gr.LATER)
    assert "local_astra/conductor/executor.py" in changed, changed
    assert "local_astra/conductor/executor.py" in set(la) - T85_OWNED, la


def _twopass_at(rev: str) -> Any:
    """``twopass.py`` as it is at ``rev``, imported under a private name (T-0094b: replaces ``_fork_twopass``, which
    loaded only the fork point)."""
    from tests.cd import _native_artifacts as na

    return na.main_twopass(rev)


#: Criterion 8's names: functions compared by source (AST dump) and module strings compared by value.
T85_RULE_FUNCS = ("select", "seed_ids", "_seed_parts")
T85_RULE_STRINGS = ("FORMAT_RULE", "SYSTEM_PROMPT", "NATIVE_SYSTEM_PROMPT")
#: The power pair (T-0094b ruling P4): T-0084c's merge 9e84f21 (BASE = merge-base of its parents, TIP = ``^2``), whose
#: BASE blob has no ``NATIVE_SYSTEM_PROMPT`` (T-0084c added it).
T0084C_MERGE = "9e84f21"
T0084C_BASE = "9737368"
T0084C_TIP = "8ccbcc4"


def _defs(src: str) -> dict[str, str]:
    import ast

    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def _t85_rule_diffs(base: str, tip: str) -> list[str]:
    """Criterion 8's predicate over two blobs of ``twopass.py``: the names among ``T85_RULE_FUNCS`` and
    ``T85_RULE_STRINGS`` that differ between ``base`` and ``tip`` (absent on either side counts as a difference)."""
    rel = "local_astra/eval/cd/twopass.py"
    do, dn = _defs(_git("show", f"{base}:{rel}")), _defs(_git("show", f"{tip}:{rel}"))
    old, new = _twopass_at(base), _twopass_at(tip)
    out = [fn for fn in T85_RULE_FUNCS if fn not in do or dn.get(fn) != do[fn]]
    out += [
        a
        for a in T85_RULE_STRINGS
        if not (hasattr(old, a) and hasattr(new, a)) or getattr(new, a) != getattr(old, a)
    ]
    return out


def test_t0085_select_rules_prompts_and_seed_ids_are_unchanged() -> None:
    """Criterion 8: ``select``, ``select_format``'s rule text, both system prompts and ``seed_ids`` are equal between
    T-0085's own BASE and TIP blobs (source text of the functions; the prompt and rule strings; T-0094b: never BASE
    against the live file). The ``fa5dace3`` prefix is a hand value on the current module and stays."""
    gr.prove_range(T0085_BASE, T0085_TIP, T0085_MERGE)
    base_rev, tip_rev = T0085_BASE, T0085_TIP
    rel = "local_astra/eval/cd/twopass.py"
    assert _git("show", f"{base_rev}:{rel}") != _git("show", f"{tip_rev}:{rel}"), (
        "premise: two blobs"
    )
    assert _t85_rule_diffs(base_rev, tip_rev) == []
    new = _tp()
    assert _sha(new.SYSTEM_PROMPT).startswith("fa5dace3")


def test_the_t0085_rules_predicate_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over T-0084c's own pair the same predicate names ``NATIVE_SYSTEM_PROMPT``
    (absent at T-0084c's BASE); a predicate that is always empty fails."""
    gr.prove_range(T0084C_BASE, T0084C_TIP, T0084C_MERGE)
    assert gr.is_ancestor(T0084C_MERGE, T0085_BASE)
    assert "NATIVE_SYSTEM_PROMPT" in _t85_rule_diffs(T0084C_BASE, T0084C_TIP)


def test_the_t0085_rules_row_compares_two_different_blobs() -> None:
    """T-0094b too-strict twin: with the allowance emptied (every top-level function compared) the predicate over
    T-0085's own pair reports its owned change, ``continuation_guard`` (T-0085 criterion 3); a pair reduced to TIP vs
    TIP fails here."""
    gr.prove_range(T0085_BASE, T0085_TIP, T0085_MERGE)
    rel = "local_astra/eval/cd/twopass.py"
    do = _defs(_git("show", f"{T0085_BASE}:{rel}"))
    dn = _defs(_git("show", f"{T0085_TIP}:{rel}"))
    differing = {fn for fn in do.keys() | dn.keys() if do.get(fn) != dn.get(fn)}
    assert "continuation_guard" in differing, sorted(differing)
    assert not differing & set(T85_RULE_FUNCS)


#: stored artifacts (main checkout's ``logs/``) and their sha256 prefixes from the task's regime
T85_STORED = {
    "logs/t0080-toolcall-dev-0dc9134": "b0fd71cc",
    "logs/t0080-toolcall-test-0dc9134": "1c69a9c2",
    "logs/t0082-replay-dev-48313a9": "e5148afd",
    "logs/t0082-replay-test-48313a9": "41628bdb",
    "logs/t0084-native-dev-9aee66d": "31f9fd7f",
}


@pytest.mark.parametrize("rel", list(T85_STORED))
def test_t0085_stored_artifacts_are_byte_unchanged(rel: str) -> None:
    """Criterion 8: every stored artifact, the 9aee66d partial included, keeps its sha256."""
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    (art,) = sorted((Path(common).parent / rel).glob("*-toolcall-*.jsonl"))
    assert _sha(art.read_bytes()).startswith(T85_STORED[rel]), art


# =========================================================================== T-0085 criterion 3(b) and ruling K8: every special token

from tests.unit._toolcall_support import STRUCTURAL_MARKERS  # noqa: E402

CHANNEL_MARKERS = ("<|channel>", "<channel|>")


def _special_tokens() -> list[str]:
    """K8: the pinned tokenizer's added tokens with ``special: true`` (read from ``tokenizer.json``, not a hand list)."""
    from tests.cd import _compile_support as cs

    added = json.loads(cs.hf_tokenizer().to_str())["added_tokens"]
    return sorted({str(a["content"]) for a in added if a.get("special")})


def _guard() -> Any:
    fn = getattr(_tp(), "continuation_guard", None)
    if not callable(fn):
        raise AssertionError(
            "T-0085 not implemented: twopass.continuation_guard(continuation, tag) is missing"
        )
    return fn


def test_t0085_k8_premise_the_named_markers_are_special_tokens() -> None:
    """Premise: the 13 markers of criterion 3(b) are all special tokens of the pinned tokenizer, and the tokenizer
    has more (media tokens among them), so a hand list cannot be the whole guard."""
    sp = set(_special_tokens())
    assert set(STRUCTURAL_MARKERS) <= sp, set(STRUCTURAL_MARKERS) - sp
    assert "<|image|>" in sp and len(sp) > len(STRUCTURAL_MARKERS)


@pytest.mark.parametrize("tag", [ts.CC, OPEN], ids=["cc", "native"])
@pytest.mark.parametrize("marker", sorted(set(STRUCTURAL_MARKERS)))
def test_t0085_each_named_marker_in_the_spoken_part_is_refused(marker: str, tag: str) -> None:
    """Criterion 3(b) by name, both tags: ``"Sure." + marker + " " + tag`` -> ``marker`` (``template`` for the two
    channel markers, K1: the template rendering changes them first)."""
    want = "template" if marker in CHANNEL_MARKERS else "marker"
    assert _guard()(f"Sure.{marker} {tag}", tag) == want


@pytest.mark.parametrize("tag", [ts.CC, OPEN], ids=["cc", "native"])
@pytest.mark.parametrize("token", [t for t in _special_tokens() if t not in STRUCTURAL_MARKERS])
def test_t0085_k8_every_other_special_token_is_refused(token: str, tag: str) -> None:
    """K8: every special token outside the named 13 (``<|image|>``, ``<pad>``, ``<mask>``, ...), derived from the
    tokenizer, in the spoken part -> ``marker``."""
    assert _guard()(f"Sure.{token} {tag}", tag) == "marker"


@pytest.mark.parametrize("tag", [ts.CC, OPEN], ids=["cc", "native"])
@pytest.mark.parametrize(
    "text",
    [
        "Sure. ",
        "\n Sure.\t",
        "An <image> of a cat. ",
        "The <|image tag and image|> text. ",
        "Use <tool_call> or <think> or <channel> or <turn>. ",
        "Pipes |> and <|chan, nel> and <pad or mask>. ",
        "",
    ],
)
def test_t0085_k8_the_guard_is_not_too_wide(text: str, tag: str) -> None:
    """Criterion 3 / K8, the other direction: ordinary text with special-token look-alikes that are not special tokens
    passes (``None``)."""
    assert _guard()(text + tag, tag) is None


def test_t0085_k8_an_image_token_in_the_spoken_part_stops_before_t4(tmp_path: Path) -> None:
    """K8 end to end: a native pass 1 with ``<|image|>`` in its spoken part -> no T4 sent, ``continuation_guard:
    marker``."""
    sc = scenario()
    name_, args = ts.gt_call(ts.by_id()["click-01"])
    sc.native_pass1["click-01"] = (
        "<|channel>thought\nok<channel|>Sure.<|image|> " + ts.native_call(name_, args) + RESPONSE
    )
    r = run_scenario(tmp_path, [T85_NATIVE_T4], ids=["click-01"], sc=sc)
    _stopped_by_guard(r, "marker")
    assert _t4_sent(r) == []


def test_t0085_k8_image_like_text_does_not_stop(tmp_path: Path) -> None:
    """K8, the other direction end to end: ``An <image> of it.`` (not a special token) runs to exit 0 with one T4."""
    sc = scenario()
    name_, args = ts.gt_call(ts.by_id()["click-01"])
    sc.native_pass1["click-01"] = (
        "<|channel>thought\nok<channel|>An <image> of it. " + ts.native_call(name_, args) + RESPONSE
    )
    r = run_scenario(tmp_path, [T85_NATIVE_T4], ids=["click-01"], sc=sc)
    assert r.result.exit_code == 0, r.result.stopped
    assert len(_t4_sent(r)) == 1 and r.rejected == []
