"""T-0080b criteria 2-4 and 7: the two-pass driver ``local_astra.eval.cd.twopass`` against the loopback fake vLLM server.

CPU; the fake of ``tests/cd/fakes/vllm_server.py`` (extended by T-0080b: scripted pass-1 answers, ``/metrics``
prefix-cache counters, 400 on an invalid continuation); the eval set ``tests/eval/toolcall/`` (dev split: 40 tool +
8 no-tool cases); cells ``T1|T2|T4 / named / perstep_mar`` (the stage-1 cells of the K-D2 ruling) at the frozen
``docs/cd/calibration-bfcl.json`` knobs. The API, request shapes and row fields the rows read are listed in
``tests/cd/_twopass_support.py``.

The full dev run (one per module) scripts these answers, so every count below is worked out by hand:

- every tool case answers with its exact call after ``<cc>`` and the spoken prefix ``"On it (<id>). "``, except
  ``search-01`` (answers without a tag: a missed route, no pass 2), ``click-02`` (the draft after ``<cc>`` is prose:
  ``draft_name`` None -> name picker), ``type-02`` (the draft names ``navigate``, not a v1 tool -> name picker) and
  ``click-03`` (draft names ``click_at`` with ``x`` 1500: schema-invalid, scope still ``click_at``);
- no-tool cases answer without a tag, except ``chat-02`` (a false tag: ``<cc>`` + a google_search call -> pass 2,
  routing is on the answer, never the label), ``chat-03`` (``<cc>`` only inside a closed thought -> none) and
  ``chat-04`` (a thought that never closes and contains ``<cc>``; the served ``content`` still shows the tag, so a
  driver that routes on served fields instead of the decoded ids sends a pass 2 here).

Hence 40 routed cases (39 tool + chat-02); per cell 38 one-request cases + 2 picker cases x 2 = 42 pass-2 requests;
3 cells -> 126; with 48 pass-1 requests the run sends exactly 174 chat requests.
"""

from __future__ import annotations

import collections
import hashlib
import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _twopass_support as ts
from tests.cd.fakes.vllm_server import Reply, cd_keys

DEV_IDS = [c["id"] for c in ts.split("dev")]
TOOL_IDS = [c["id"] for c in ts.split("dev") if c["kind"] == "tool"]
NO_PASS2 = [
    "search-01",
    "chat-01",
    "chat-03",
    "chat-04",
    "chat-05",
    "chat-06",
    "chat-07",
    "chat-08",
]
PICKER = {"click-02": "click_at", "type-02": "type_at"}
INVALID_DRAFTS = {"click-02", "type-02", "click-03"}
ROUTED = [c for c in DEV_IDS if c not in NO_PASS2]
PASS1_N, PASS2_N = 48, 126

SPECIAL_PASS1 = {
    "search-01": "<|channel>thought\nI know this.<channel|>The weather in Lisbon tomorrow looks mild.",
    "click-02": "<|channel>thought\nThe link.<channel|>Opening the link. <cc>I will click the link.",
    "type-02": (
        "<|channel>thought\nNavigate?<channel|>Searching the shop. "
        '<cc>{"name": "navigate", "arguments": {"url": "shop"}}'
    ),
    "click-03": (
        "<|channel>thought\nThe button.<channel|>Adding it now. "
        '<cc>{"name": "click_at", "arguments": {"x": 1500, "y": 380}}'
    ),
    "chat-02": (
        "<|channel>thought\n17*23.<channel|>That is 391. "
        '<cc>{"name": "google_search", "arguments": {"query": "17 times 23"}}'
    ),
    "chat-03": (
        '<|channel>thought\nMaybe <cc>{"name": "google_search", "arguments": {"query": "happy"}} -- no.'
        "<channel|>Glad, cheerful and joyful."
    ),
    "chat-04": '<|channel>thought\nI could <cc>{"name": "google_search", "arguments": {"query": "cat',
}
SPOKEN = {
    "click-02": "Opening the link. ",
    "type-02": "Searching the shop. ",
    "click-03": "Adding it now. ",
    "chat-02": "That is 391. ",
}


def spoken(case_id: str) -> str:
    return SPOKEN.get(case_id, ts.Scenario.spoken(case_id))


def full_scenario() -> ts.Scenario:
    return ts.Scenario(
        pass1=dict(SPECIAL_PASS1),
        draft_tool={"chat-02": ("google_search", {"query": "17 times 23"})},
    )


class FullRun:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.result: Any = None
        self.scenario: ts.Scenario | None = None
        self.srv: Any = None
        self.connects: list[Any] = []

    def get(self) -> FullRun:
        if self.error is not None:
            raise self.error
        return self

    @property
    def sent(self) -> list[ts.Sent]:
        assert self.scenario is not None
        return sorted(self.scenario.sent, key=lambda s: s.index)

    def p1(self, case_id: str) -> ts.Sent:
        got = [s for s in self.sent if s.kind == "p1" and s.case == case_id]
        assert len(got) == 1, f"{case_id}: {len(got)} pass-1 requests"
        return got[0]

    def p2(self, case_id: str) -> list[ts.Sent]:
        return [s for s in self.sent if s.kind == "p2" and s.case == case_id]

    def rows(self, which: int) -> list[dict[str, Any]]:
        return ts.pass_rows(self.result.artifact, which)


@pytest.fixture(scope="module")
def full(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FullRun]:
    """One full dev run, shared by the rows that only read it. A missing driver is stored and re-raised inside
    each row (an assertion in the call phase, not a fixture error)."""
    fr = FullRun()
    tmp = tmp_path_factory.mktemp("twopass_full")
    fr.scenario = full_scenario()
    real_connect = socket.socket.connect

    def recording_connect(sock: socket.socket, address: Any) -> Any:
        fr.connects.append(address)
        return real_connect(sock, address)

    try:
        ts.twopass()
    except AssertionError as exc:
        fr.error = exc
        yield fr
        return
    with ts.fake(fr.scenario, tmp) as srv:
        fr.srv = srv
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", recording_connect)
            try:
                fr.result = ts.run_driver(srv, tmp)
            except BaseException as exc:  # surfaced in every row that reads the run
                fr.error = exc
        yield fr


# =========================================================================== the full dev run


def test_the_run_completes_with_exit_0(full: FullRun) -> None:
    fr = full.get()
    assert fr.result.exit_code == 0, (fr.result.stopped, ts.stopped_marker(fr.result.artifact))
    assert fr.srv.rejected == [], "the driver sent an invalid continuation (400)"


def test_pass1_is_sent_once_per_case_of_the_stage(full: FullRun) -> None:
    """Criterion 3: pass 1 once per case, for exactly the dev split (no test-split case, nothing unmatched)."""
    fr = full.get()
    p1 = [s.case for s in fr.sent if s.kind == "p1"]
    assert None not in p1, "a pass-1 request carries no case text"
    assert collections.Counter(p1) == collections.Counter(DEV_IDS)


def test_pass1_request_regime(full: FullRun) -> None:
    """Criterion 2: one fixed system prompt (tool instructions, the v1 declarations as text, the ``<cc>``
    convention) for every case, thinking on, ``max_tokens`` 1024, generated ids returned, no ``tools`` field, no
    ``astra_cd_*`` key; one image for a case with a screen, none without; the system prompt's and the tools file's
    sha256 in the regime line."""
    fr = full.get()
    systems = set()
    for cid in DEV_IDS:
        body = fr.p1(cid).body
        assert body.get("chat_template_kwargs", {}).get("enable_thinking") is True, cid
        assert body.get("max_tokens") == 1024, cid
        assert body.get("return_token_ids") is True, cid
        assert "tools" not in body and "tool_choice" not in body, cid
        assert not cd_keys(body), cid
        msgs = body["messages"]
        assert msgs[0]["role"] == "system" and isinstance(msgs[0]["content"], str), cid
        systems.add(msgs[0]["content"])
        user = [m for m in msgs if m["role"] == "user"]
        assert len(user) == 1, cid
        want_images = 1 if ts.by_id()[cid]["screen"] else 0
        assert len(ts.image_parts(user[0])) == want_images, cid
    assert len(systems) == 1, f"{len(systems)} different system prompts"
    system = systems.pop()
    assert "<cc>" in system
    for tool in ts.TOOLS:
        assert tool in system, f"the system prompt does not declare {tool}"
    regime = ts.artifact_lines(fr.result.artifact)[0]
    text = str(regime)
    assert hashlib.sha256(system.encode()).hexdigest() in text, (
        "system prompt sha256 not in the regime"
    )
    assert ts.toolset().version in text, "tools_v1.json sha256 not in the regime"


def test_no_pass2_without_a_tag_in_the_answer(full: FullRun) -> None:
    """Routing is on the model's answer only: a tool case answered without a tag, no-tool answers, a tag only
    inside a closed thought and an unclosed thought containing ``<cc>`` send no pass 2; a no-tool case whose
    answer carries ``<cc>`` (chat-02) does, in every cell (it counts as a false tag)."""
    fr = full.get()
    for cid in NO_PASS2:
        assert fr.p2(cid) == [], f"{cid} got a pass 2"
    assert {s.shape for s in fr.p2("chat-02")} == {"T1", "T2", "T4"}
    kinds = {r["case"]: ts.route_kind(r) for r in fr.rows(1)}
    assert kinds == {c: ("none" if c in NO_PASS2 else "toolcall") for c in DEV_IDS}


def test_exactly_the_expected_requests_are_sent(full: FullRun) -> None:
    """48 pass-1 + 126 pass-2 requests (hand count in the module docstring): no extra request for anything,
    including the derived policies."""
    fr = full.get()
    kinds = collections.Counter(s.kind for s in fr.sent)
    assert kinds == {"p1": PASS1_N, "p2": PASS2_N}
    assert len(fr.srv.requests) == PASS1_N + PASS2_N


def test_pass2_arms_are_paired_on_the_same_pass1(full: FullRun) -> None:
    """Criterion 3: every routed case gets every cell, and all of its pass-2 requests are sent after its own
    pass 1 and before the next case's pass 1 (the arms share one pass-1 output)."""
    fr = full.get()
    sent = fr.sent
    p1_at = {s.case: i for i, s in enumerate(sent) if s.kind == "p1"}
    order = sorted(p1_at, key=lambda c: p1_at[c])
    for k, cid in enumerate(order):
        lo = p1_at[cid]
        hi = p1_at[order[k + 1]] if k + 1 < len(order) else len(sent)
        between = sent[lo + 1 : hi]
        assert all(s.kind == "p2" and s.case == cid for s in between), cid
        if cid in ROUTED:
            assert sorted(ts.cells_in_order(between)) == sorted(ts.CELLS), (
                cid,
                ts.cells_in_order(between),
            )


def test_pass2_order_is_williams_balanced_by_routed_count(full: FullRun) -> None:
    """Ruling R-W (T-0011 R17 refined; T-0071a lesson: a cyclic rotation fixes each cell's predecessor, so any
    carry-over lands on one cell). For the 3 dev cells the order follows a Williams design: 2k = 6 sequences in which
    each cell takes each position equally often and every cell immediately precedes every other equally often,
    cycled by the routed-case count (not the case position: chat-02 is routed case 39 at position 41).

    Hand values: the first 6 routed cases carry 6 distinct orders (for k = 3 that is every permutation), each cell
    twice in each position, each ordered adjacent pair twice; order i equals order i - 6 for all 40 routed cases; over
    the first 36 (six cycles) each ordered adjacent pair occurs 12 times."""
    fr = full.get()
    seqs = [ts.cells_in_order(fr.p2(cid)) for cid in ROUTED]
    assert len(seqs) == 40 and all(sorted(q) == sorted(ts.CELLS) for q in seqs), seqs
    first = seqs[:6]
    assert len({tuple(q) for q in first}) == 6, first
    for pos in range(3):
        assert collections.Counter(q[pos] for q in first) == {c: 2 for c in ts.CELLS}, (pos, first)

    def adjacent(qs: list[list[str]]) -> collections.Counter[tuple[str, str]]:
        return collections.Counter((q[i], q[i + 1]) for q in qs for i in range(len(q) - 1))

    pairs = [(a, b) for a in ts.CELLS for b in ts.CELLS if a != b]
    assert adjacent(first) == {pr: 2 for pr in pairs}, adjacent(first)
    for i in range(6, len(seqs)):
        assert seqs[i] == seqs[i - 6], (i, ROUTED[i], seqs[i], seqs[i - 6])
    assert adjacent(seqs[:36]) == {pr: 12 for pr in pairs}


def test_t4_body_is_pass1_messages_plus_the_spoken_prefix_ending_in_cc(full: FullRun) -> None:
    """Criterion 3 / umbrella T4: pass-1 messages + ``{"role": "assistant", "content": spoken_prefix + "<cc>"}``
    (the committed answer text before the tag, no thought), ``continue_final_message`` true,
    ``add_generation_prompt`` false, ``max_tokens`` 256; every other field as in pass 1."""
    fr = full.get()
    n = 0
    for cid in ROUTED:
        p1 = fr.p1(cid).body
        for s in fr.p2(cid):
            if s.shape != "T4":
                continue
            n += 1
            b = s.body
            assert b["messages"][:-1] == p1["messages"], cid
            assert b["messages"][-1] == {"role": "assistant", "content": spoken(cid) + "<cc>"}, (
                cid,
                b["messages"][-1],
            )
            assert (
                b.get("continue_final_message") is True and b.get("add_generation_prompt") is False
            )
            assert b.get("max_tokens") == 256, cid
            skip = {
                "messages",
                "vllm_xargs",
                "max_tokens",
                "continue_final_message",
                "add_generation_prompt",
            }
            assert {k: v for k, v in b.items() if k not in skip} == {
                k: v for k, v in p1.items() if k not in skip
            }, cid
    assert n == 42, f"{n} T4 requests"


def test_t1_t2_bodies_equal_pass1_but_for_xargs_and_max_tokens(full: FullRun) -> None:
    """Criterion 3: T1/T2 send pass 1's body unchanged (same messages, images, system prompt, thinking) except
    ``vllm_xargs`` and ``max_tokens`` (T1 1024, T2 256) -- the prefix cache is warm by construction."""
    fr = full.get()
    seen = collections.Counter()
    for cid in ROUTED:
        p1 = fr.p1(cid).body
        for s in fr.p2(cid):
            if s.shape not in ("T1", "T2"):
                continue
            seen[s.shape] += 1
            b = s.body
            assert {k: v for k, v in b.items() if k not in ("vllm_xargs", "max_tokens")} == {
                k: v for k, v in p1.items() if k != "max_tokens"
            }, (cid, s.shape)
            assert b["max_tokens"] == (1024 if s.shape == "T1" else 256)
            assert "continue_final_message" not in b or b["continue_final_message"] is False
    assert seen == {"T1": 42, "T2": 42}


def test_named_scope_sends_the_drafts_tool_grammar_never_union(full: FullRun) -> None:
    """K-D2: the pass-2 automaton is the grammar of the tool the draft names (click-03's schema-invalid draft
    still names ``click_at``), under the thought variant of the request's shape; no union automaton is sent."""
    fr = full.get()
    for s in fr.sent:
        if s.kind != "p2":
            continue
        assert s.scope != "?", (
            f"{s.case}: an automaton that is no live (scope, thought) grammar (union?)"
        )
        assert s.thought == s.shape, (s.case, s.thought, s.shape)
        assert cd_keys(s.body).get("astra_cd_mode") == ts.ARM
    for cid in ROUTED:
        if cid in PICKER:
            continue
        want = ts.Scenario().call_for(cid)[0] if cid != "chat-02" else "google_search"
        assert {s.scope for s in fr.p2(cid)} == {want}, cid
        assert len(fr.p2(cid)) == 3, cid


def test_name_picker_runs_when_the_draft_names_no_v1_tool(full: FullRun) -> None:
    """K-D2: ``draft_name`` None (prose after the tag; a tool outside v1) -> per cell two requests: the name picker
    (``scope="name"``), then the picked tool's grammar under the same thought, with the same body otherwise; the
    pass-2 row records ``requests`` 2 (1 elsewhere)."""
    fr = full.get()
    for cid, tool in PICKER.items():
        sent = fr.p2(cid)
        assert len(sent) == 6, (cid, [(s.scope, s.shape) for s in sent])
        by_shape: dict[str, list[ts.Sent]] = collections.defaultdict(list)
        for s in sent:
            by_shape[s.shape or "?"].append(s)
        for shape, pair in by_shape.items():
            assert [s.scope for s in pair] == ["name", tool], (cid, shape)
            a, b = (dict(x.body) for x in pair)
            a.pop("vllm_xargs"), b.pop("vllm_xargs")
            assert a == b, (cid, shape)
    rows = {(r["case"], r["cell"]): r for r in fr.rows(2)}
    for (cid, cell), r in rows.items():
        assert r["requests"] == (2 if cid in PICKER else 1), (cid, cell)


def test_pass2_rows_one_per_routed_case_and_cell(full: FullRun) -> None:
    """Row contents: one pass-2 row per (routed case, cell); CS 1 by host replay, no error, the scripted exact
    call scored exact and schema-valid on every tool case."""
    fr = full.get()
    rows = fr.rows(2)
    keys = collections.Counter((r["case"], r["cell"]) for r in rows)
    assert keys == collections.Counter((c, cell) for c in ROUTED for cell in ts.CELLS)
    for r in rows:
        assert r["cs"] is True and r["error"] is None, r
        assert r["schema_valid"] is True, r
        if r["case"] in TOOL_IDS:
            assert r["exact"] is True, r


def test_prefix_cache_hit_tokens_are_the_requests_own(full: FullRun) -> None:
    """Criterion 3 / implementation note: ``/metrics`` read before and after each pass-2 request; the row's hit
    tokens are that request's delta (the fake adds 900 + request index; every pass 1 adds 77, so a delta taken
    around the wrong window, or a cumulative value, differs)."""
    fr = full.get()
    sent_at = {(s.case, f"{s.shape}/named/{ts.ARM}"): s for s in fr.sent if s.kind == "p2"}
    checked = 0
    for r in fr.rows(2):
        if r["case"] in PICKER:
            continue
        s = sent_at[(r["case"], r["cell"])]
        assert r["prefix_hit_tokens"] == 900 + s.index, (
            r["case"],
            r["cell"],
            r["prefix_hit_tokens"],
        )
        checked += 1
    assert checked == 38 * 3


def test_derived_policies_are_offline_and_follow_their_definitions(full: FullRun) -> None:
    """Criterion 3: ``draft_only`` (pass 1's draft parsed and scored) and ``cd_on_invalid:<cell>`` (the draft if
    schema-valid, else that cell's pass-2 result; latency adds pass 2 only on invalid drafts) are computed from the
    artifact rows with no request (count asserted before and after).

    Hand values: drafts are exact on the 36 routed tool cases with a valid draft; invalid drafts are click-02 (prose),
    type-02 (``navigate``) and click-03 (x 1500); search-01 made no call. cd_on_invalid repairs the three with the
    scripted exact pass-2 call: 39 exact."""
    fr = full.get()
    tp = ts.twopass()
    rows = [x for x in ts.artifact_lines(fr.result.artifact)[1:] if x.get("pass") in (1, 2)]
    before = (len(fr.srv.requests), len(fr.srv.flushes), list(fr.srv.gets))
    cell = f"T4/named/{ts.ARM}"
    draft = {r["case"]: r for r in tp.derive(rows, "draft_only")}
    repaired = {r["case"]: r for r in tp.derive(rows, f"cd_on_invalid:{cell}")}
    assert (len(fr.srv.requests), len(fr.srv.flushes), list(fr.srv.gets)) == before
    assert set(draft) == set(DEV_IDS) and set(repaired) == set(DEV_IDS)

    p1_wall = {r["case"]: float(r["wall_ms"]) for r in fr.rows(1)}
    p2_wall = {r["case"]: float(r["wall_ms"]) for r in fr.rows(2) if r["cell"] == cell}
    for cid in DEV_IDS:
        d, c = draft[cid], repaired[cid]
        assert d["tool_case"] is (cid in TOOL_IDS) and c["tool_case"] is (cid in TOOL_IDS), cid
        assert d["called"] is (cid in ROUTED), cid
        if cid in ROUTED:
            assert d["schema_valid"] is (cid not in INVALID_DRAFTS), cid
        if cid in TOOL_IDS:
            assert d["exact"] is (cid in ROUTED and cid not in INVALID_DRAFTS), cid
            assert c["exact"] is (cid in ROUTED), cid
        assert d["e2e_wall_ms"] == pytest.approx(p1_wall[cid]), cid
        if cid in INVALID_DRAFTS:
            assert c["e2e_wall_ms"] == pytest.approx(p1_wall[cid] + p2_wall[cid]), cid
        else:
            assert c["e2e_wall_ms"] == pytest.approx(p1_wall[cid]), cid
    assert sum(bool(draft[c]["exact"]) for c in TOOL_IDS) == 36
    assert sum(bool(repaired[c]["exact"]) for c in TOOL_IDS) == 39


def test_every_connection_stays_on_loopback(full: FullRun) -> None:
    """Criterion 7: no request leaves loopback (``google_search`` is never executed)."""
    fr = full.get()
    assert fr.connects, "fixture: no connection was recorded"
    hosts = {a[0] for a in fr.connects if isinstance(a, tuple)}
    assert hosts <= {"127.0.0.1", "::1", "localhost"}, hosts


# =========================================================================== stops (small runs)


def _small(
    tmp_path: Path,
    scenario: ts.Scenario,
    limit: int = 3,
    cfg: dict[str, Any] | None = None,
    **kw: Any,
) -> tuple[Any, Any, ts.Scenario]:
    with ts.fake(scenario, tmp_path, **kw) as srv:
        result = ts.run_driver(srv, tmp_path, limit=limit, **(cfg or {}))
        return result, srv, scenario


def _last_post_is(scenario: ts.Scenario, srv: Any, case: str, shape: str | None) -> None:
    sent = sorted(scenario.sent, key=lambda s: s.index)
    assert sent, "nothing was sent"
    last = sent[-1]
    assert (last.case, last.shape) == (case, shape), (
        f"requests continued after the stop: last was {(last.case, last.kind, last.shape)}"
    )
    assert len(srv.requests) == len(sent)


@pytest.mark.parametrize("where", ["pass1", "pass2"])
def test_an_http_error_is_an_engine_stop(tmp_path: Path, where: str) -> None:
    """Criterion 4 (T-0071b C9/C12): an ``http_error`` row stops the run: report written, exit 1, no later request.
    Dev cases search-01..03; the error is search-02's pass 1, or its T2 pass 2."""
    sc = ts.Scenario()
    if where == "pass1":
        sc.pass1_reply["search-02"] = Reply(http_status=500)
        shape = None
    else:
        sc.pass2[("search-02", "T2")] = Reply(http_status=500)
        shape = "T2"
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code == 1, result.stopped
    assert Path(result.artifact).is_file()
    assert ts.stopped_marker(result.artifact), "the artifact does not say the run stopped"
    _last_post_is(sc, srv, "search-02", shape)


def test_a_cs_violation_stops_the_run(tmp_path: Path) -> None:
    """Criterion 4: a constrained, terminated row the host replay rejects (CS != 1) stops the run, whatever the
    plugin says. search-02's T4 output is a google_search call with an undeclared argument, then EOS."""
    sc = ts.Scenario()
    sc.pass2[("search-02", "T4")] = Reply(
        text='{"name": "google_search", "arguments": {"q": "british museum"}}', steps=ts.P2_STEPS
    )
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code != 0
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T4")


def _p2_reply(case: str, scope: str, thought: str, **kw: Any) -> Reply:
    """The in-language pass-2 output the scenario would send, with ``kw`` changed."""
    text = ts.Scenario().pass2_text(case, scope, thought)
    return Reply(text=text, steps=ts.P2_STEPS, **kw)


def test_a_join_error_stops_the_run(tmp_path: Path) -> None:
    """Implementer-added stop (review of 528d385), T-0070 (c) pattern: a constrained request whose ``ASTRA_CD``
    records never arrive (after the E9 flushes) is a ``join_error`` and stops the run; nothing is sent after it.
    Short waits keep the row fast; the stop does not depend on them."""
    sc = ts.Scenario()
    sc.pass2[("search-02", "T1")] = _p2_reply("search-02", "google_search", "T1", omit_records=True)
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc, cfg={"record_wait_s": 0.2, "flush_wait_s": 0.1})
    assert result.exit_code != 0, result.stopped
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T1")


def test_served_ids_that_are_not_the_committed_ids_stop_the_run(tmp_path: Path) -> None:
    """Implementer-added stop: the served ``return_token_ids`` must equal the ids the records committed (up to the
    stop). Both texts here are in the language and terminated (replay accepts, ``in_final`` true), so only the
    mismatch can stop the run: served query "british museum", committed query "british museum hours"."""
    sc = ts.Scenario()
    other = ts.Scenario(
        draft_tool={"search-02": ("google_search", {"query": "british museum hours"})}
    )
    sc.pass2[("search-02", "T2")] = _p2_reply(
        "search-02",
        "google_search",
        "T2",
        committed_text=other.pass2_text("search-02", "google_search", "T2"),
    )
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code != 0, result.stopped
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T2")


def test_replay_accepting_what_the_plugin_calls_not_final_stops_the_run(tmp_path: Path) -> None:
    """T-0070 (a), first half, on its own: an in-language terminated output (replay accepts) whose last record says
    ``in_final`` false. The CS = 1 checks pass, so only the replay/plugin disagreement can stop the run."""
    sc = ts.Scenario()
    sc.pass2[("search-02", "T4")] = _p2_reply("search-02", "google_search", "T4", in_final=False)
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code != 0, result.stopped
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T4")


def test_a_terminated_output_the_replay_rejects_stops_even_when_the_plugin_agrees(
    tmp_path: Path,
) -> None:
    """T-0070 (a), second half, on its own: replay rejects a terminated output and the plugin agrees
    (``in_final`` false), so there is no disagreement to trip; only CS != 1 can stop the run."""
    sc = ts.Scenario()
    sc.pass2[("search-02", "T4")] = Reply(
        text='{"name": "google_search", "arguments": {"q": "british museum"}}',
        steps=ts.P2_STEPS,
        in_final=False,
    )
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code != 0, result.stopped
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T4")


@pytest.mark.parametrize("code", ["z0", "empty_delta", "infeasible", "internal"])
def test_a_loud_cd_error_stops_the_run(tmp_path: Path, code: str) -> None:
    sc = ts.Scenario()
    sc.pass2[("search-02", "T1")] = Reply(error=code)
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code != 0
    assert ts.stopped_marker(result.artifact)
    _last_post_is(sc, srv, "search-02", "T1")


def test_a_quiet_refusal_does_not_stop_the_run(tmp_path: Path) -> None:
    """The other side of the stop guard: ``bucket_too_large`` is not one of the loud errors of criterion 4; the
    row records it and the run goes on (3 cases: 3 pass-1 + 9 pass-2 requests), exit 0."""
    sc = ts.Scenario()
    sc.pass2[("search-02", "T1")] = Reply(error="bucket_too_large")
    ts.twopass()
    result, srv, _ = _small(tmp_path, sc)
    assert result.exit_code == 0, result.stopped
    assert len(srv.requests) == 3 + 9
    errs = [r for r in ts.pass_rows(result.artifact, 2) if r["error"]]
    assert [(r["case"], r["cell"], r["error"]) for r in errs] == [
        ("search-02", f"T1/named/{ts.ARM}", "bucket_too_large")
    ]


def test_prefix_hit_is_unavailable_without_counters_never_guessed(tmp_path: Path) -> None:
    """Implementation note: with no ``/metrics`` counters the hit is reported as unavailable (None), and the run
    still completes."""
    ts.twopass()
    result, _, _ = _small(tmp_path, ts.Scenario(), limit=2, metrics=False)
    assert result.exit_code == 0, result.stopped
    rows = ts.pass_rows(result.artifact, 2)
    assert len(rows) == 6
    assert all(r["prefix_hit_tokens"] is None for r in rows), [r["prefix_hit_tokens"] for r in rows]


def test_bucket_histogram_is_printed_before_the_first_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 4: a bucket histogram per (scope, thought) is printed on the CPU before the first request (the
    output is read from inside the fake when the first request arrives)."""
    seen: list[str] = []
    sc = ts.Scenario(on_first=lambda: seen.append(capsys.readouterr().out))
    ts.twopass()
    _small(tmp_path, sc, limit=1)
    assert seen, "no request was sent"
    out = seen[0].lower()
    assert "bucket" in out, out[-2000:]
    for thought in ts.THOUGHTS:
        assert thought.lower() in out, f"no histogram line for {thought}"


# =========================================================================== refusals


def test_a_non_loopback_base_url_is_refused_before_any_socket(tmp_path: Path) -> None:
    tp = ts.twopass()
    repo, head = ts.es.clean_repo(tmp_path)
    cfg = tp.TwoPassConfig(
        stage="dev",
        cells=list(ts.CELLS),
        base_url="http://10.1.2.3:8001/v1",
        engine_log=tmp_path / "engine.log",
        out_dir=tmp_path / "artifacts",
        repo=repo,
        expected_commit=head,
        calibration=dict(ts.CALIBRATION),
        set_dir=ts.SET_DIR,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", lambda *a: pytest.fail(f"socket opened: {a}"))
        with pytest.raises(ValueError):
            tp.run(cfg)


def _refused_without_requests(tmp_path: Path, **overrides: Any) -> None:
    sc = ts.Scenario()
    ts.twopass()
    with ts.fake(sc, tmp_path) as srv:
        try:
            result = ts.run_driver(srv, tmp_path, limit=2, **overrides)
        except ValueError:
            result = None
        if result is not None:
            assert result.exit_code != 0, "the refused configuration ran"
        assert srv.requests == [], f"{len(srv.requests)} requests sent by a refused run"


def test_a_dirty_tree_is_refused(tmp_path: Path) -> None:
    repo, head = ts.es.clean_repo(tmp_path)
    (repo / "README").write_text("dirty\n")
    _refused_without_requests(tmp_path, repo=repo, expected_commit=head)


def test_a_frozen_file_of_another_arm_is_refused(tmp_path: Path) -> None:
    """Criterion 4: ``perstep_mar`` given ``perstep_mar_qargmax``'s frozen file (not its calibration source)."""
    _refused_without_requests(tmp_path, calibration={"perstep_mar": ts.QARGMAX_FILE})


def test_the_correct_frozen_file_is_accepted(tmp_path: Path) -> None:
    """The other side of the frozen-file guard: the arm's own source file runs (2 cases, 2 + 6 requests)."""
    ts.twopass()
    result, srv, _ = _small(tmp_path, ts.Scenario(), limit=2)
    assert result.exit_code == 0, result.stopped
    assert len(srv.requests) == 2 + 6


# =========================================================================== negative criterion 7

#: T-0080b's merge commit bb9fc8a; BASE = ``git merge-base bb9fc8a^1 bb9fc8a^2``, TIP = ``bb9fc8a^2`` (T-0094
#: criterion 2). Over this closed range no later-task exemption is needed (T-0081's plugin files came later).
T0080B_MERGE = "bb9fc8a"
T0080B_BASE = "6753797"
T0080B_TIP = "7010da1"
UNCHANGED = [
    "local_astra/eval/cd/runner.py",
    "local_astra/eval/cd/arms.py",
    "local_astra/eval/cd/calibration.py",
    "local_astra/eval/cd/cuts.py",
    "local_astra/eval/cd/hardstop.py",
]
#: criterion 7's second fence
PLUGIN_AND_CONDUCTOR = ("plugin", "local_astra/conductor")


@pytest.mark.parametrize("rel", UNCHANGED)
def test_read_only_eval_modules_are_unchanged(rel: str) -> None:
    """Criterion 7 (negative): the driver reuses these read-only; over T-0080b's closed range (T-0094) ``rel`` does
    not differ between BASE and TIP."""
    gr.prove_range(T0080B_BASE, T0080B_TIP, T0080B_MERGE)
    assert gr.changed(T0080B_BASE, T0080B_TIP, rel) == [], (
        f"{rel} changed in {T0080B_BASE}..{T0080B_TIP}"
    )


def test_plugin_and_conductor_are_unchanged() -> None:
    """Criterion 7 (negative): nothing under ``plugin/**`` or ``local_astra/conductor/**`` differs over T-0080b's
    closed range (T-0094). The untracked-file half of the old guard is retired: it only meant something while the
    branch was open."""
    gr.prove_range(T0080B_BASE, T0080B_TIP, T0080B_MERGE)
    changed = gr.changed(T0080B_BASE, T0080B_TIP, *PLUGIN_AND_CONDUCTOR)
    assert changed == [], changed


def test_the_t0080b_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the plugin/conductor predicate reports ``executor.py``
    (T-0089/T-0092). The five eval modules have no later change in that range, so their row's power is the shared
    ``changed`` predicate's (shown here and in ``tests/unit/test_guards_use_fixed_refs.py``)."""
    gr.require_commits(T0080B_BASE, gr.LATER)
    got = gr.changed(T0080B_BASE, gr.LATER, *PLUGIN_AND_CONDUCTOR)
    assert "local_astra/conductor/executor.py" in got, got
    assert gr.changed(T0080B_BASE, gr.LATER, *UNCHANGED) == []


# =========================================================================== crashes after the artifact is open

CLI = ts.REPO / "scripts" / "gpu" / "cd_toolcall.py"


def _cli() -> Any:
    """``scripts/gpu/cd_toolcall.py`` loaded by path (importing it must not start a run)."""
    import importlib.util
    import sys

    ts.twopass()
    assert CLI.is_file(), f"T-0080b not implemented: {CLI.relative_to(ts.REPO)} does not exist"
    name = "_t0080b_cd_toolcall_cli"
    spec = importlib.util.spec_from_file_location(name, CLI)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    return m


def _fail_on_second_route(mp: pytest.MonkeyPatch) -> list[str]:
    """Make the router raise on its second call (search-02's pass-1 answer): an unexpected exception inside the
    run, after the artifact is open and search-01 is done. Delegates every other call to the real router."""
    from local_astra.toolcall import router

    real = router.route
    calls: list[str] = []

    def route(*a: Any, **kw: Any) -> Any:
        calls.append(str(a[0]) if a else "")
        if len(calls) == 2:
            raise RuntimeError("injected failure inside the run (T-0080b crash row)")
        return real(*a, **kw)

    mp.setattr(router, "route", route)
    return calls


def test_a_crash_after_the_artifact_is_open_is_a_stop_with_a_trailer(tmp_path: Path) -> None:
    """Reviewer advisory (528d385), implemented in ab306e0: an exception inside ``run`` once the artifact is open
    ends the run like a stop -- exit 1 (not 2, which is for pre-request refusals), a trailer whose ``stopped``
    starts with ``crash:`` and which carries the traceback -- and nothing is sent after the failure. search-01's
    rows, written before it, stay in the artifact."""
    sc = ts.Scenario()
    ts.twopass()
    with pytest.MonkeyPatch.context() as mp:
        calls = _fail_on_second_route(mp)
        result, srv, _ = _small(tmp_path, sc)
    assert len(calls) >= 2, "fixture: the injected failure was never reached"
    assert result.exit_code == 1, (result.exit_code, result.stopped)
    lines = ts.artifact_lines(result.artifact)
    trailers = [x for x in lines if x.get("kind") == "trailer"]
    assert len(trailers) == 1, trailers
    assert str(trailers[0].get("stopped", "")).startswith("crash:"), trailers[0].get("stopped")
    assert "injected failure inside the run" in str(trailers[0].get("traceback", "")), trailers[0]
    assert "Traceback" in str(trailers[0].get("traceback", ""))
    assert {r["case"] for r in ts.pass_rows(result.artifact, 2)} == {"search-01"}
    _last_post_is(sc, srv, "search-02", None)


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
        "3",
        *extra,
    ]


def test_the_cli_still_writes_its_report_after_a_crash(tmp_path: Path) -> None:
    """The same crash through ``scripts/gpu/cd_toolcall.py``: exit 1, and the report (``report-dev.json`` and
    ``.md``) is still written from the artifact; no selection is written for a stopped dev run."""
    cli = _cli()
    sc = ts.Scenario()
    out = tmp_path / "evidence"
    with ts.fake(sc, tmp_path) as srv, pytest.MonkeyPatch.context() as mp:
        _fail_on_second_route(mp)
        rc = cli.main(_cli_args(srv, tmp_path, out))
    assert rc == 1, rc
    assert list(out.glob("*.jsonl")), "no artifact"
    rep = out / "report-dev.json"
    assert rep.is_file() and (out / "report-dev.md").is_file(), sorted(
        p.name for p in out.iterdir()
    )
    assert "crash:" in rep.read_text()
    assert not (out / "selection-dev.json").exists()


def test_a_pre_artifact_refusal_still_exits_2_with_no_artifact(tmp_path: Path) -> None:
    """The other side: a configuration refused before the artifact opens (an unknown thought in a cell) is exit 2
    from the CLI and a ``ValueError`` from ``run``, with no artifact and no request."""
    cli = _cli()
    tp = ts.twopass()
    sc = ts.Scenario()
    out = tmp_path / "evidence"
    with ts.fake(sc, tmp_path) as srv:
        rc = cli.main(_cli_args(srv, tmp_path, out, "--cell", "T9/named/perstep_mar"))
        assert rc == 2, rc
        assert not (out.exists() and list(out.glob("*.jsonl"))), "an artifact was written"
        direct = tmp_path / "direct"
        direct.mkdir()
        with pytest.raises(ValueError):
            ts.run_driver(srv, direct, cells=["T9/named/perstep_mar"])
        assert not list((direct / "artifacts").glob("*.jsonl"))
        assert srv.requests == []
    assert tp is not None


# =========================================================================== T-0082a: a draft closed by </cc>
#
# Small runs (dev cases search-01..03, the three T-0080b cells, then the three seed cells). Each case answers with
# its exact call followed by ``</cc>`` (the T-0080c shape: ``...}}</cc>`` + stop); the comparison run answers the
# same without the tag. Criterion 9's "identical to main" is checked as "identical to the tag-free run": the
# tag-free scenario is what main already handles (the full-run rows above pin main's behaviour on it, unedited).

CLOSE_IDS = DEV_IDS[:3]
SEED_CELLS_T82 = tuple(f"{t}/named/perstep_mar+seed" for t in ts.THOUGHTS)
#: Fields that differ between any two runs (timing, the run id inside record tags), never compared.
VOLATILE = {"wall_ms", "wall_ms_per_request", "record_tags"}
#: Pass-1 fields the scripted ``</cc>`` legitimately changes (the served text and ids) or adds (``draft_close``).
P1_CHANGED = {"text", "token_ids", "draft_close"}


def _close_scenario(close: str) -> ts.Scenario:
    sc = ts.Scenario()
    for cid in CLOSE_IDS:
        sc.pass1[cid] = ts.Scenario().pass1_text(cid) + close
    return sc


def _norm_row(row: Any, skip: set[str]) -> dict[str, Any]:
    """A row without ``skip`` and with the record tag inside ``knobs`` blanked (it carries the run id)."""
    out = {k: v for k, v in row.items() if k not in skip}
    if isinstance(out.get("knobs"), dict) and "record_tag" in out["knobs"]:
        out["knobs"] = {**out["knobs"], "record_tag": "<tag>"}
    return out


def _norm_body(body: Any) -> Any:
    """A request body with the record tag (it carries the run's random id) blanked."""
    import copy

    b = copy.deepcopy(dict(body))
    if isinstance(b.get("vllm_xargs"), dict) and "astra_cd_record_tag" in b["vllm_xargs"]:
        b["vllm_xargs"]["astra_cd_record_tag"] = "<tag>"
    return b


class _ClosePair:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.runs: dict[str, tuple[Any, Any, ts.Scenario]] = {}

    def get(self) -> _ClosePair:
        if self.error is not None:
            raise self.error
        return self


@pytest.fixture(scope="module")
def close_pair(tmp_path_factory: pytest.TempPathFactory) -> _ClosePair:
    """Two small runs on the same three cases: answers closed by ``</cc>`` and the tag-free answers."""
    pair = _ClosePair()
    try:
        ts.twopass()
        for name, close in (("close", "</cc>"), ("plain", "")):
            tmp = tmp_path_factory.mktemp(f"t0082a_{name}")
            sc = _close_scenario(close)
            with ts.fake(sc, tmp) as srv:
                result = ts.run_driver(srv, tmp, limit=3)
                pair.runs[name] = (result, list(srv.requests), sc)
    except BaseException as exc:  # surfaced in every row that reads the pair
        pair.error = exc
    return pair


def test_a_draft_closed_by_cc_is_stripped_scored_and_kept_by_cd_on_invalid(
    close_pair: _ClosePair,
) -> None:
    """T-0082a criteria 1, 9: a ``...}}</cc>`` + stop draft is routed with ``draft_close == "</cc>"`` and the bare
    call as ``draft_call_text``; it is scored on the stripped draft (parse ok, schema-valid, exact), so
    ``cd_on_invalid`` keeps the draft and sends no pass 2 (``pass2_sent`` False); the tag-free run records
    ``draft_close == ""``."""
    pair = close_pair.get()
    tp = ts.twopass()
    result, _, _ = pair.runs["close"]
    assert result.exit_code == 0, result.stopped
    rows = ts.pass_rows(result.artifact, 1)
    assert [r["case"] for r in rows] == CLOSE_IDS
    for r in rows:
        call = ts.call_text(*ts.gt_call(ts.by_id()[r["case"]]))
        assert r.get("draft_close") == "</cc>", (r["case"], r.get("draft_close"))
        assert r["draft_call_text"] == call, r["case"]
        assert (r["draft_parse"], r["draft_schema_valid"], r["draft_exact"]) == (
            "ok",
            True,
            True,
        ), r
    all_rows = [x for x in ts.artifact_lines(result.artifact)[1:] if x.get("pass") in (1, 2)]
    for cell in ts.CELLS:
        for d in tp.derive(all_rows, f"cd_on_invalid:{cell}"):
            assert (d["pass2_sent"], d["schema_valid"], d["exact"]) == (False, True, True), (
                cell,
                d,
            )
    plain, _, _ = pair.runs["plain"]
    assert [r.get("draft_close") for r in ts.pass_rows(plain.artifact, 1)] == ["", "", ""]


def test_cd_always_bodies_are_byte_equal_to_the_tag_free_case(close_pair: _ClosePair) -> None:
    """T-0082a criterion 9 (negative): every request body -- pass 1 and each non-seed cell's pass 2, in order --
    is the same with and without the closing tag (record tag blanked: it carries the run id); 12 requests each."""
    pair = close_pair.get()
    close_reqs = pair.runs["close"][1]
    plain_reqs = pair.runs["plain"][1]
    assert len(close_reqs) == len(plain_reqs) == 3 + 3 * 3, (len(close_reqs), len(plain_reqs))
    for i, (a, b) in enumerate(zip(close_reqs, plain_reqs, strict=True)):
        ja = json.dumps(_norm_body(a), sort_keys=True, ensure_ascii=False)
        jb = json.dumps(_norm_body(b), sort_keys=True, ensure_ascii=False)
        assert ja == jb, f"request {i} differs"


def test_rows_are_identical_to_the_tag_free_case_but_for_the_close(close_pair: _ClosePair) -> None:
    """T-0082a criterion 9 (negative): pass-1 rows equal the tag-free run's apart from the served text/ids and
    ``draft_close``; pass-2 rows (non-seed cells) are equal apart from timing and record tags; counts equal."""
    pair = close_pair.get()
    a = pair.runs["close"][0].artifact
    b = pair.runs["plain"][0].artifact
    for which, skip in ((1, VOLATILE | P1_CHANGED), (2, VOLATILE)):
        ra, rb = ts.pass_rows(a, which), ts.pass_rows(b, which)
        assert len(ra) == len(rb), which
        for x, y in zip(ra, rb, strict=True):
            assert _norm_row(x, skip) == _norm_row(y, skip), (which, x.get("case"), x.get("cell"))
            if which == 1:
                assert set(x) == set(y), set(x) ^ set(y)
                assert "draft_close" in x


def test_the_seed_cells_seed_ids_decode_without_the_close(tmp_path: Path) -> None:
    """T-0082a criterion 8 through the driver: on the ``</cc>``-closed answers, every seed cell's ``seed_ids``
    (row and wire) is the hand-built seed of the bare call -- H + ``encode(T + call)`` + the fake's EOS -- and
    decodes without ``</cc>``."""
    from local_astra.decoding import vocab
    from local_astra.eval.cd import replay

    tp = ts.twopass()

    class _Seeded(ts.Scenario):
        def __call__(self, index: int, body: Any) -> Reply:
            reply = super().__call__(index, body)
            if "astra_cd_draft" in cd_keys(body) and reply.steps == ts.P2_STEPS:
                reply.steps = 27  # a seeded request executes 27 - 24 = 3 steps (T-0081b)
            return reply

    sc = _Seeded(pass1=dict(_close_scenario("</cc>").pass1))
    with ts.fake(sc, tmp_path) as srv:
        result = ts.run_driver(srv, tmp_path, limit=3, cells=list(SEED_CELLS_T82))
        wire = [s["draft"] for s in srv.cd_seeds() if s is not None]
    assert result.exit_code == 0, result.stopped
    p1 = {r["case"]: r for r in ts.pass_rows(result.artifact, 1)}
    rows = ts.pass_rows(result.artifact, 2)
    assert len(rows) == 9 and len(wire) == 9
    for r in rows:
        case, thought = r["case"], r["cell"].split("/")[0]
        ids = [int(t) for t in p1[case]["token_ids"]]
        call = ts.call_text(*ts.gt_call(ts.by_id()[case]))
        head = {"T1": ids[: ids.index(101) + 1], "T2": [100, 45518, 107, 101], "T4": []}[thought]
        tag = "" if thought == "T4" else "<cc>"
        want = [*head, *(int(t) for t in vocab.encode(tag + call)), 1]
        assert r.get("seed_ids") == want, (case, thought)
        assert "</cc>" not in replay.detokenize(r["seed_ids"]), (case, thought)
        assert want in wire, (case, thought)
    assert tp is not None


# =========================================================================== T-0085: the cc T4 continuation (criterion 3)
# The cc twin of the native rows in test_twopass_native_driver.py. The fake refuses (400) any continuation the pinned
# template changes, cc included. Stop string pinned as ``continuation_guard: <reason>`` (see the native module).

T85_CELL = "T4/named/perstep_mar"


def _t85_run(
    tmp_path: Path, text: str, cells: list[str] | None = None
) -> tuple[Any, Any, ts.Scenario]:
    sc = ts.Scenario()
    sc.pass1["search-02"] = "<|channel>thought\nok<channel|>" + text
    set_dir = ts.subset_set_dir(tmp_path / "set", ["search-02"])
    with ts.fake(sc, tmp_path) as srv:
        result = ts.run_driver(srv, tmp_path, cells=cells or [T85_CELL], set_dir=set_dir)
        return result, srv, sc


def _t85_call() -> str:
    name, args = ts.gt_call(ts.by_id()["search-02"])
    return ts.call_text(name, args)


def _t85_stop(result: Any, srv: Any, sc: ts.Scenario, reason: str) -> None:
    assert result.exit_code == 1, (result.exit_code, result.stopped)
    assert f"continuation_guard: {reason}" in str(result.stopped), result.stopped
    assert f"continuation_guard: {reason}" in str(ts.stopped_marker(result.artifact))
    assert srv.rejected == [], "a 400, not the guard"
    assert [s for s in sc.sent if s.shape == "T4"] == [], "the guarded T4 was sent"


def test_t0085_a_stray_close_in_a_cc_answer_gives_a_clean_t4_body(tmp_path: Path) -> None:
    """Criteria 1 and 6 (cc): ``Sure.<channel|> <cc>{...}`` -> the T4 continues ``"Sure. <cc>"``, exit 0, no 400."""
    result, srv, sc = _t85_run(tmp_path, "Sure.<channel|> <cc>" + _t85_call())
    assert result.exit_code == 0, result.stopped
    assert srv.rejected == []
    (t4,) = [s for s in sc.sent if s.shape == "T4"]
    assert t4.body["messages"][-1] == {"role": "assistant", "content": "Sure. <cc>"}


def test_t0085_cc_mutant_router_stops_at_the_guard(tmp_path: Path) -> None:
    """Criterion 3(a), cc: a router keeping the stray close -> no T4 sent, ``continuation_guard: template``."""
    from local_astra.toolcall import router

    real = router.route

    def mutant(text: str, **kw: Any) -> Any:
        import dataclasses

        r = real(text, **kw)
        if r.kind == "toolcall" and "<channel|>" + router.TAGS[r.tag] in text:
            return dataclasses.replace(r, spoken_prefix=r.spoken_prefix + "<channel|>")
        return r

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(router, "route", mutant)
        result, srv, sc = _t85_run(tmp_path, "Sure.<channel|><cc>" + _t85_call())
    _t85_stop(result, srv, sc, "template")


@pytest.mark.parametrize("marker", ["<|think|>", "<tool_call|>", "<|turn>", "<bos>"])
def test_t0085_cc_structural_marker_stops_before_t4(tmp_path: Path, marker: str) -> None:
    """Criterion 3(b), cc: a marker in the spoken part -> no T4 sent, ``continuation_guard: marker``."""
    result, srv, sc = _t85_run(tmp_path, f"Sure.{marker} <cc>" + _t85_call())
    _t85_stop(result, srv, sc, "marker")


def test_t0085_cc_guard_is_not_too_wide(tmp_path: Path) -> None:
    """Criterion 3, the other direction (cc): the marker answer on a T1 cell (no continuation) completes."""
    result, _srv, _sc = _t85_run(
        tmp_path, "Sure.<|think|> <cc>" + _t85_call(), cells=["T1/named/perstep_mar"]
    )
    assert result.exit_code == 0, result.stopped


def test_t0085_k3_a_spliced_native_tag_in_a_cc_route_stops_before_t4(tmp_path: Path) -> None:
    """Criterion 3(b) with K3: ``<|tool_<channel|>call> then <cc>{...}`` routes ``cc`` with ``<|tool_call>`` spoken
    (the splice is no tag) -> the cc T4 is not sent; ``continuation_guard: marker``."""
    result, srv, sc = _t85_run(tmp_path, "<|tool_<channel|>call> then <cc>" + _t85_call())
    _t85_stop(result, srv, sc, "marker")
