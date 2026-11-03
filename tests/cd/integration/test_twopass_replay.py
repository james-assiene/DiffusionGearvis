"""T-0082b criteria 1-6: pass-2 replay -- ``twopass.run`` with ``pass1_from``/``pass1_sha256`` sends only the pass-2
requests of the given cells against the stored pass-1 rows of an earlier artifact, re-scored with ``rescore``.

CPU; the loopback fake vLLM server of ``tests/cd/fakes/vllm_server.py``; the dev split of ``tests/eval/toolcall/``,
first 6 cases (search-01..06); the six cells ``T1|T2|T4 / named / perstep_mar`` and ``... / perstep_mar+seed`` at the
frozen ``docs/cd/calibration-bfcl.json`` knobs.

**The oracle is the live code path at this commit**, not the replay: one live run (``LIVE``) on the scenario below
records every request body and row. The replay's source artifact (``P``) is that live artifact edited back into the
shape a pre-T-0082a driver wrote (as the pinned T-0080c artifacts are): the two ``</cc>``-closed drafts keep the
*unstripped* ``draft_call_text``, no ``draft_close``, and the scores of an unparseable draft; every pass-1 row gets
distinctive ``wall_ms`` / prefix values (so "kept from the artifact" is distinguishable from "measured again"); and
the regime's ``commit``/``run_id``/``started_utc`` are changed (fields outside criterion 2's list must not refuse).

Scenario (pass 1 of the live run; the replay sends none):

- search-01: answered without a tag -> route none, no pass 2;
- search-02: the exact call + ``</cc>`` (+ stop); search-03: the exact call + ``</cc>\\n`` -> stripped by the router;
- search-05: prose after ``<cc>`` -> ``draft_name`` None -> the name picker (2 requests per cell, never seeded);
- search-04, search-06: the exact call.

Hand counts: 5 routed cases (search-02..06); per cell 4 one-request cases + 1 picker case x 2 = 6 pass-2 requests;
6 cells -> 36. The live run sends 6 + 36 = 42; the replay exactly 36 + 5 warm-ups (R10: one ``max_tokens`` 1 copy of
each routed case's pass-1 body, before its first pass-2 request) + flushes (the fake joins every record, so the
trailer's ``flush_requests`` is 0 in these rows; the count reads it rather than assuming it).

Williams order (ruling R-W) for k = 6, worked by hand: row 0 = 0, 1, 5, 2, 4, 3; row i adds i mod 6. Routed case j
(search-02 = 0 ... search-06 = 4) takes row j.

API pins where the task is silent (reported): a refusal is ``ValueError`` from ``run`` or a result with exit code 2,
and in both cases zero requests and zero connections to the server; the replay takes ``limit`` as the live run did.

Regime: CPU, fake server, pinned tokenizer (model rev 2ea8372), ``tools_v1.json`` pinned, CL 256, 48 steps, seed
constants noise 0.0 / start 24, R2 trim. No GPU, no network (loopback only).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import socket
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.cd import _eval_support as es
from tests.cd import _twopass_support as ts
from tests.cd.fakes.vllm_server import Reply, cd_keys

SEED = "perstep_mar+seed"
BASE_CELLS = tuple(f"{t}/named/perstep_mar" for t in ts.THOUGHTS)
SEED_CELLS = tuple(f"{t}/named/{SEED}" for t in ts.THOUGHTS)
CELLS = BASE_CELLS + SEED_CELLS
LIMIT = 6
CASES = [c["id"] for c in ts.split("dev")][:LIMIT]
NONE_ROUTED = "search-01"
CLOSED = {"search-02": "</cc>", "search-03": "</cc>\n"}
PICKER = "search-05"
ROUTED = [c for c in CASES if c != NONE_ROUTED]
PASS2_N = 36  # hand count, module docstring
#: Williams rows for k = 6 (hand-worked, docstring); routed case j uses row j
WILLIAMS6 = [
    [0, 1, 5, 2, 4, 3],
    [1, 2, 0, 3, 5, 4],
    [2, 3, 1, 4, 0, 5],
    [3, 4, 2, 5, 1, 0],
    [4, 5, 3, 0, 2, 1],
]
SEEDED_STEPS = 27  # a seeded request's step counter (27 - 24 = 3 executed), as in the T-0081b rows
#: Pass-2 row fields that differ between any two runs (timing, run id, the fake's per-index prefix counters)
VOLATILE2 = {
    "wall_ms",
    "wall_ms_per_request",
    "record_tags",
    "prefix_hit_tokens",
    "prefix_query_tokens",
}
P1_KEPT = ("wall_ms", "prefix_hit_tokens", "prefix_query_tokens")
CLI = ts.REPO / "scripts" / "gpu" / "cd_toolcall.py"


# --------------------------------------------------------------------------- scenario


class Scenario(ts.Scenario):
    """``ts.Scenario`` with the seeded requests' step counter at 27 (the T-0081b convention)."""

    def __call__(self, index: int, body: Mapping[str, Any]) -> Reply:
        reply = super().__call__(index, body)
        if "astra_cd_draft" in cd_keys(body) and reply.steps == ts.P2_STEPS:
            reply.steps = SEEDED_STEPS
        return reply


def scenario() -> Scenario:
    sc = Scenario()
    plain = ts.Scenario()
    sc.pass1[NONE_ROUTED] = (
        "<|channel>thought\nI know this.<channel|>The weather in Lisbon tomorrow looks mild."
    )
    for cid, close in CLOSED.items():
        sc.pass1[cid] = plain.pass1_text(cid) + close
    sc.pass1[PICKER] = (
        "<|channel>thought\nSearch.<channel|>Looking it up. <cc>I will search for it."
    )
    return sc


# --------------------------------------------------------------------------- the code under test


def _tp() -> Any:
    tp = ts.twopass()
    names = {f.name for f in dataclasses.fields(tp.TwoPassConfig)}
    missing = [n for n in ("pass1_from", "pass1_sha256") if n not in names]
    if missing:
        raise AssertionError(f"T-0082b not implemented: TwoPassConfig has no {missing}")
    return tp


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm_body(body: Mapping[str, Any]) -> str:
    b = copy.deepcopy(dict(body))
    if isinstance(b.get("vllm_xargs"), dict) and "astra_cd_record_tag" in b["vllm_xargs"]:
        b["vllm_xargs"]["astra_cd_record_tag"] = "<tag>"
    return json.dumps(b, sort_keys=True, ensure_ascii=False)


def _norm_row(row: Mapping[str, Any], skip: set[str]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in skip}
    if isinstance(out.get("knobs"), dict) and "record_tag" in out["knobs"]:
        out["knobs"] = {**out["knobs"], "record_tag": "<tag>"}
    return out


def _pass1_shaped(body: Mapping[str, Any]) -> bool:
    """A request with pass 1's shape: no CD keys and a case's user text (a flush carries neither)."""
    return not cd_keys(body) and ts.case_of(body) is not None


def _warmup_cases(requests: list[dict[str, Any]]) -> list[str | None]:
    """R10: the cases of the pass-1-shaped requests, asserting each is a warm-up (``max_tokens`` 1)."""
    shaped = [b for b in requests if _pass1_shaped(b)]
    assert all(b.get("max_tokens") == 1 for b in shaped), (
        "a pass-1-shaped request that is not a max_tokens-1 warm-up",
        [(ts.case_of(b), b.get("max_tokens")) for b in shaped],
    )
    return [ts.case_of(b) for b in shaped]


def _trailer(path: Path) -> dict[str, Any]:
    lines = [x for x in ts.artifact_lines(path) if x.get("kind") == "trailer"]
    assert lines, "no trailer line"
    return lines[-1]


# --------------------------------------------------------------------------- source artifact


def _to_pre_t0082a(live: Path, dest: Path) -> Path:
    """``live`` rewritten as a pre-T-0082a driver would have stored it (module docstring)."""
    lines = ts.artifact_lines(live)
    out = []
    for i, line in enumerate(lines):
        line = copy.deepcopy(line)
        if i == 0:
            line["commit"] = "0123456789abcdef0123456789abcdef01234567"
            line["expected_commit"] = line["commit"]
            line["run_id"] = "feedfacecafe"
            line["started_utc"] = "2026-10-01T00:00:00+00:00"
        elif line.get("pass") == 1:
            k = CASES.index(line["case"])
            line["wall_ms"] = 1000.0 + 7.25 * k
            line["prefix_hit_tokens"] = 4242 + k
            line["prefix_query_tokens"] = 9000 + k
            if line["case"] in CLOSED:
                text = line["text"]
                raw = text[text.index(ts.CC) + len(ts.CC) :]
                assert raw.endswith(CLOSED[line["case"]]), raw
                line["draft_call_text"] = raw  # the pre-strip router's draft (T-0080a criterion 2)
                line.pop("draft_close", None)
                line.update(
                    draft_parse="not JSON",
                    draft_call=None,
                    draft_schema_valid=False,
                    draft_exact=False,
                )
        out.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    return dest


def _edit(src: Path, dest: Path, fn: Callable[[list[dict[str, Any]]], None]) -> Path:
    lines = ts.artifact_lines(src)
    fn(lines)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return dest


def _p1(lines: list[dict[str, Any]], case: str) -> dict[str, Any]:
    return next(x for x in lines if x.get("pass") == 1 and x.get("case") == case)


# --------------------------------------------------------------------------- runs


@dataclasses.dataclass
class Run:
    result: Any  # None when ``run`` raised ``ValueError`` (``refusal`` holds it)
    requests: list[dict[str, Any]]
    sent: list[ts.Sent]
    connects: list[Any]
    srv_url: str
    seeds: list[Any]
    refusal: ValueError | None = None


def _replay(
    tmp: Path,
    source: Path,
    *,
    sha: str | None = None,
    sc: Scenario | None = None,
    cells: tuple[str, ...] = CELLS,
    **overrides: Any,
) -> Run:
    tp = _tp()
    sc = sc or scenario()
    tmp.mkdir(parents=True, exist_ok=True)
    connects: list[Any] = []
    real_connect = socket.socket.connect

    def recording(sock: socket.socket, address: Any) -> Any:
        connects.append(address)
        return real_connect(sock, address)

    result: Any = None
    refusal: ValueError | None = None
    with ts.fake(sc, tmp) as srv:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", recording)
            try:
                result = ts.run_driver(
                    srv,
                    tmp,
                    cells=list(cells),
                    limit=LIMIT,
                    pass1_from=source,
                    pass1_sha256=_sha(source) if sha is None else sha,
                    **overrides,
                )
            except ValueError as exc:
                refusal = exc
        reqs = list(srv.requests)
        seeds = srv.cd_seeds()
        url = srv.url
    assert tp is not None
    run = Run(result, reqs, sorted(sc.sent, key=lambda s: s.index), connects, url, seeds, refusal)
    return run


class Pair:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.live: Run | None = None
        self.source: Path | None = None
        self.replay: Run | None = None

    def get(self) -> Pair:
        if self.error is not None:
            raise self.error
        return self


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pair]:
    """The live run (the oracle), the pre-T-0082a source artifact built from it, and one replay of that artifact.
    A missing feature is stored and re-raised inside each row (an assertion in the call phase)."""
    p = Pair()
    try:
        ts.twopass()
        tmp = tmp_path_factory.mktemp("t0082b_live")
        sc = scenario()
        with ts.fake(sc, tmp) as srv:
            live = ts.run_driver(srv, tmp, cells=list(CELLS), limit=LIMIT)
            p.live = Run(
                live,
                list(srv.requests),
                sorted(sc.sent, key=lambda s: s.index),
                [],
                srv.url,
                srv.cd_seeds(),
            )
        assert live.exit_code == 0, ("the live oracle run failed", live.stopped)
        assert len(p.live.requests) == LIMIT + PASS2_N, len(p.live.requests)
        p.source = _to_pre_t0082a(
            Path(live.artifact), tmp_path_factory.mktemp("t0082b_src") / "src.jsonl"
        )
        p.replay = _replay(tmp_path_factory.mktemp("t0082b_replay"), p.source)
        assert p.replay.refusal is None, f"the valid replay was refused: {p.replay.refusal}"
    except BaseException as exc:  # surfaced in every row that reads the pair
        p.error = exc
    yield p


def _source(pair: Pair) -> Path:
    """The source artifact (inside the row, so a missing feature fails the row, never the fixture)."""
    p = pair.get()
    assert p.source is not None
    return p.source


# =========================================================================== criterion 3: no pass 1


def test_the_replay_completes_with_exit_0(pair: Pair) -> None:
    r = pair.get().replay
    assert r is not None
    assert r.result.exit_code == 0, (r.result.stopped, ts.stopped_marker(r.result.artifact))


def test_no_pass1_shaped_request_but_one_warmup_per_routed_case(pair: Pair) -> None:
    """Criterion 3 as amended by R10: the only pass-1-shaped requests (no CD keys and a case's text) are the
    warm-ups -- exactly one per routed case, in case order, none for the no-tool search-01 -- and each equals the
    live run's pass-1 body of that case with ``max_tokens`` 1 (oracle: the live request, not the replay's code)."""
    p = pair.get()
    r = p.replay
    assert r is not None and p.live is not None
    assert _warmup_cases(r.requests) == ROUTED
    live_p1 = {ts.case_of(b): b for b in p.live.requests if _pass1_shaped(b)}
    for b in (b for b in r.requests if _pass1_shaped(b)):
        want = copy.deepcopy(dict(live_p1[ts.case_of(b)]))
        want["max_tokens"] = 1
        assert _norm_body(b) == _norm_body(want), ts.case_of(b)


def test_each_warmup_precedes_its_cases_first_pass2_request(pair: Pair) -> None:
    """R10: per routed case the wire order is the warm-up, then that case's pass-2 requests (flushes ignored); the
    warm-up comes before the case's first pass-2 request and after the previous case's last."""
    r = pair.get().replay
    assert r is not None
    seq = [(s.kind, s.case) for s in r.sent if s.case is not None]
    want: list[tuple[str, str | None]] = []
    for cid in ROUTED:
        n = sum(1 for s in r.sent if s.kind == "p2" and s.case == cid)
        want += [("p1", cid)] + [("p2", cid)] * n
    assert seq == want, seq


def test_warmup_rows_are_logged_and_counted_in_the_trailer(pair: Pair) -> None:
    """R10: one ``{"pass": "warmup", "case", "wall_ms", prefix counters}`` row per routed case, in case order and
    before that case's pass-2 rows; the trailer counts them (a trailer key naming warm-ups equals 5)."""
    r = pair.get().replay
    assert r is not None
    lines = ts.artifact_lines(r.result.artifact)
    warm = [x for x in lines if x.get("pass") == "warmup"]
    assert [x.get("case") for x in warm] == ROUTED
    for x in warm:
        assert {"case", "wall_ms", "prefix_hit_tokens", "prefix_query_tokens"} <= set(x), x
        first_p2 = next(
            i for i, y in enumerate(lines) if y.get("pass") == 2 and y["case"] == x["case"]
        )
        assert lines.index(x) < first_p2, x["case"]
    trailer = _trailer(r.result.artifact)
    counts = {k: v for k, v in trailer.items() if "warm" in k.lower()}
    assert len(ROUTED) in counts.values(), trailer


def test_warmups_are_excluded_from_derive_and_the_report_tables(pair: Pair) -> None:
    """R10: ``derive`` over every row (warm-ups included) gives one row per pass-1 case (6), and the report's
    policy tables are the same with the warm-up rows deleted from the artifact."""
    p = pair.get()
    tp = ts.twopass()
    assert p.replay is not None
    lines = ts.artifact_lines(p.replay.result.artifact)
    rows = [x for x in lines[1:] if x.get("pass") in (1, 2, "warmup")]
    assert any(x.get("pass") == "warmup" for x in rows), "no warm-up rows"
    assert [d["case"] for d in tp.derive(rows, "draft_only")] == CASES
    stripped = Path(p.replay.result.artifact).with_name("no-warmup.jsonl")
    stripped.write_text("".join(json.dumps(x) + "\n" for x in lines if x.get("pass") != "warmup"))
    a = tp.report([p.replay.result.artifact])
    b = tp.report([stripped])
    assert a["policies"] == b["policies"]
    assert a["selection"] == b["selection"]


def test_the_report_shows_pass2_walls_by_position(pair: Pair) -> None:
    """R10: the report shows pass-2 walls by position within a case (the first cell after the warm-up vs later): a
    report key and a markdown line mentioning "position". (Loose pin: the task names no field.)"""
    p = pair.get()
    tp = ts.twopass()
    assert p.replay is not None
    rep_ = tp.report([p.replay.result.artifact])
    assert any("position" in k.lower() for k in rep_), sorted(rep_)
    md = tp.render_markdown(rep_)
    assert any("position" in ln.lower() and "wall" in ln.lower() for ln in md.splitlines())


def test_the_request_count_is_the_pass2_hand_count(pair: Pair) -> None:
    """Criterion 3 as amended by R10: requests = sum over cells of the routed cases' pass-2 requests (36, hand count
    incl. the picker's second requests) + one warm-up per routed case (5) + flushes (the trailer's count)."""
    r = pair.get().replay
    assert r is not None
    flushes = int(_trailer(r.result.artifact).get("flush_requests") or 0)
    assert len(r.requests) == PASS2_N + len(ROUTED) + flushes, (len(r.requests), flushes)
    with_keys = [b for b in r.requests if cd_keys(b)]
    assert len(with_keys) == PASS2_N
    per_case = {c: sum(1 for s in r.sent if s.kind == "p2" and s.case == c) for c in CASES}
    assert per_case == {
        "search-01": 0,
        "search-02": 6,
        "search-03": 6,
        "search-04": 6,
        "search-05": 12,
        "search-06": 6,
    }


def test_pass1_rows_are_the_rescored_stored_rows_with_replayed_from(pair: Pair) -> None:
    """Criterion 3: each pass-1 row is the stored row re-scored under the current router -- which is the live
    run's row at this commit (oracle) -- with ``wall_ms`` and the prefix fields of the *stored* row (not new
    measurements) and ``replayed_from`` = the source sha. Nothing else differs."""
    source = _source(pair)
    p = pair.get()
    assert p.live is not None and p.replay is not None
    sha = _sha(source)
    live = {x["case"]: x for x in ts.pass_rows(p.live.result.artifact, 1)}
    stored = {x["case"]: x for x in ts.pass_rows(source, 1)}
    got = ts.pass_rows(p.replay.result.artifact, 1)
    assert [x["case"] for x in got] == CASES
    for row in got:
        cid = row["case"]
        want = {**live[cid], **{k: stored[cid][k] for k in P1_KEPT}, "replayed_from": sha}
        assert row == want, (cid, {k for k in set(row) | set(want) if row.get(k) != want.get(k)})


def test_the_stale_closed_drafts_are_rescored(pair: Pair) -> None:
    """Criterion 3 via T-0082a: the stored (pre-strip) drafts of search-02/03 end in ``</cc>`` and are scored
    unparseable; in the replay they are the bare call, ``draft_close`` the removed suffix, valid and exact."""
    source = _source(pair)
    p = pair.get()
    assert p.replay is not None
    stored = {x["case"]: x for x in ts.pass_rows(source, 1)}
    got = {x["case"]: x for x in ts.pass_rows(p.replay.result.artifact, 1)}
    for cid, close in CLOSED.items():
        assert stored[cid]["draft_call_text"].endswith(close), "fixture: the source is not stale"
        call = ts.call_text(*ts.gt_call(ts.by_id()[cid]))
        assert got[cid]["draft_call_text"] == call, cid
        assert got[cid]["draft_close"] == close, cid
        assert (
            got[cid]["draft_parse"],
            got[cid]["draft_schema_valid"],
            got[cid]["draft_exact"],
        ) == (
            "ok",
            True,
            True,
        ), cid


# =========================================================================== criterion 4: same requests as live


def test_every_pass2_body_equals_the_live_runs(pair: Pair) -> None:
    """Criterion 4: the replay's request sequence equals the live run's pass-2 requests at this commit, body by
    body and in order (record tag blanked: it carries the run id), for T1, T2 and T4 -- seeds on the wire
    included."""
    p = pair.get()
    assert p.live is not None and p.replay is not None
    live = [_norm_body(b) for b in p.live.requests if cd_keys(b)]
    got = [_norm_body(b) for b in p.replay.requests if cd_keys(b)]
    assert len(live) == len(got) == PASS2_N, (len(live), len(got))
    for i, (a, b) in enumerate(zip(live, got, strict=True)):
        assert a == b, f"pass-2 request {i} differs from the live run's"
    shapes = {ts.thought_of_body(b) for b in p.replay.requests if cd_keys(b)}
    assert shapes == {"T1", "T2", "T4"}


def test_t4_bodies_continue_the_rerouted_spoken_prefix(pair: Pair) -> None:
    """Criterion 4 (T4 by hand): a T4 body's last message is the assistant ``spoken_prefix + "<cc>"`` of the
    case's stored answer, ``continue_final_message`` true, ``add_generation_prompt`` false, ``max_tokens`` 256."""
    p = pair.get()
    assert p.replay is not None
    t4 = [s for s in p.replay.sent if s.kind == "p2" and s.shape == "T4"]
    assert {s.case for s in t4} == set(ROUTED)
    for s in t4:
        spoken = "Looking it up. " if s.case == PICKER else ts.Scenario.spoken(str(s.case))
        last = s.body["messages"][-1]
        assert last == {"role": "assistant", "content": spoken + ts.CC}, s.case
        assert s.body["continue_final_message"] is True and s.body["add_generation_prompt"] is False
        assert s.body["max_tokens"] == 256


def test_pass2_rows_equal_the_live_runs(pair: Pair) -> None:
    """Criterion 4: pass-2 rows (scores, scopes, seeds, steps, CS) equal the live run's apart from timing, record
    tags and the fake's per-index prefix counters."""
    p = pair.get()
    assert p.live is not None and p.replay is not None
    live = ts.pass_rows(p.live.result.artifact, 2)
    got = ts.pass_rows(p.replay.result.artifact, 2)
    assert len(live) == len(got) == len(ROUTED) * len(CELLS)
    for a, b in zip(live, got, strict=True):
        assert _norm_row(b, VOLATILE2) == _norm_row(a, VOLATILE2), (a["case"], a["cell"])


def test_cell_order_follows_williams_by_routed_count(pair: Pair) -> None:
    """Criterion 4 / ruling R-W: routed case j takes Williams row j (k = 6, hand-worked rows)."""
    p = pair.get()
    assert p.replay is not None
    rows = ts.pass_rows(p.replay.result.artifact, 2)
    for j, cid in enumerate(ROUTED):
        order = [r["cell"] for r in rows if r["case"] == cid]
        assert order == [CELLS[i] for i in WILLIAMS6[j]], (cid, order)


def test_seeds_come_from_stored_ids_and_the_rerouted_draft(pair: Pair) -> None:
    """Criterion 4: a seed row's ``seed_ids`` = H + encode(T + re-routed draft) + S built by hand from the
    *stored* ``token_ids`` (H(T1) = ids through the first 101, H(T2) the empty header, H(T4) nothing; S the
    fake's EOS 1). For the ``}</cc>`` drafts the seed decodes without ``</cc>`` -- the stored, stale draft would
    have put it there. The picker case is not seeded."""
    source = _source(pair)
    from local_astra.decoding import vocab
    from local_astra.eval.cd import replay as host_replay

    p = pair.get()
    assert p.replay is not None
    stored = {x["case"]: x for x in ts.pass_rows(source, 1)}
    rows = [r for r in ts.pass_rows(p.replay.result.artifact, 2) if r["arm"] == SEED]
    assert len(rows) == len(ROUTED) * 3
    for r in rows:
        cid, thought = r["case"], r["thought"]
        if cid == PICKER:
            assert r.get("seed") is None and r.get("seed_reason") == "picker", r
            continue
        ids = [int(t) for t in stored[cid]["token_ids"]]
        call = ts.call_text(*ts.gt_call(ts.by_id()[cid]))
        head = {"T1": ids[: ids.index(101) + 1], "T2": [100, 45518, 107, 101], "T4": []}[thought]
        tag = "" if thought == "T4" else ts.CC
        want = [*head, *(int(t) for t in vocab.encode(tag + call)), 1]
        assert r["seed_ids"] == want, (cid, thought)
        if cid in CLOSED:
            assert "</cc>" not in host_replay.detokenize(r["seed_ids"]), (cid, thought)
    wire = [s["draft"] for s in p.replay.seeds if s is not None]
    assert len(wire) == (len(ROUTED) - 1) * 3


def test_every_connection_stays_on_loopback(pair: Pair) -> None:
    """Criterion 6: loopback only."""
    r = pair.get().replay
    assert r is not None
    assert r.connects, "fixture: no connection was recorded"
    assert {a[0] for a in r.connects if isinstance(a, tuple)} <= {"127.0.0.1", "::1", "localhost"}


# =========================================================================== criterion 5: downstream unchanged


def test_derive_and_report_run_on_the_replay_artifact(pair: Pair) -> None:
    """Criterion 5: ``derive`` and ``report`` read the replay artifact as any other. Hand values: ``draft_only``
    schema-valid per case = search-01 None (no call), 02/03/04/06 True (02/03 after the strip), 05 False (prose);
    the stripped-close line is 2 of 5 routed."""
    p = pair.get()
    tp = ts.twopass()
    assert p.replay is not None
    rows = [x for x in ts.artifact_lines(p.replay.result.artifact)[1:] if x.get("pass") in (1, 2)]
    got = {d["case"]: d["schema_valid"] for d in tp.derive(rows, "draft_only")}
    assert got == {
        "search-01": None,
        "search-02": True,
        "search-03": True,
        "search-04": True,
        "search-05": False,
        "search-06": True,
    }
    rep = tp.report([p.replay.result.artifact])
    assert (rep["stripped_close"][0]["k"], rep["stripped_close"][0]["n"]) == (2, 5)
    assert rep["selection"].get("rule") == "stage1"


def test_the_regime_line_names_the_replayed_artifact_and_its_sha(pair: Pair) -> None:
    """Criterion 5: the replay's regime line names the source artifact and its sha256, and the rendered report has
    a line naming the source (sha or path) that says the pass-1 walls come from that artifact (it mentions
    "wall")."""
    source = _source(pair)
    p = pair.get()
    tp = ts.twopass()
    assert p.replay is not None
    sha = _sha(source)
    regime = ts.artifact_lines(p.replay.result.artifact)[0]
    blob = json.dumps(regime, ensure_ascii=False)
    assert sha in blob, "the regime line does not carry the source sha256"
    assert source.name in blob, "the regime line does not name the source artifact"
    md = tp.render_markdown(tp.report([p.replay.result.artifact]))
    lines = [ln for ln in md.splitlines() if (sha[:12] in ln or source.name in ln)]
    assert lines, "the report does not name the replayed artifact"
    assert any("wall" in ln.lower() for ln in lines), lines


def test_a_live_artifacts_regime_has_no_replay_source(pair: Pair) -> None:
    """Criterion 6 (negative): the live run at this commit names no replay source and its pass-1 rows carry no
    ``replayed_from`` (the other side of criterion 5's row)."""
    p = pair.get()
    assert p.live is not None and p.replay is not None
    assert all("replayed_from" not in x for x in ts.pass_rows(p.live.result.artifact, 1))
    live_regime = ts.artifact_lines(p.live.result.artifact)[0]
    assert all(live_regime.get(k) is None for k in ("pass1_from", "pass1_sha256"))
    live_lines = ts.artifact_lines(p.live.result.artifact)
    assert [x for x in live_lines if x.get("pass") == "warmup"] == [], "a live run sent warm-ups"
    assert all(b.get("max_tokens") != 1 for b in p.live.requests if _pass1_shaped(b))


# =========================================================================== criterion 2: refusals


def _refused(run: Callable[[], Run]) -> None:
    """A refusal: ``ValueError`` from ``run`` or exit 2; zero requests and zero connections to the server either
    way (criterion 2: "before any socket")."""
    r = run()
    if r.refusal is None:
        assert r.result.exit_code == 2, (r.result.exit_code, r.result.stopped)
    assert r.requests == [], f"{len(r.requests)} requests sent before the refusal"
    port = int(r.srv_url.rsplit(":", 1)[1].split("/")[0])
    to_srv = [a for a in r.connects if isinstance(a, tuple) and len(a) > 1 and a[1] == port]
    assert to_srv == [], f"connections to the server opened before the refusal: {to_srv}"


def _regime_edit(path: list[str], value: Any) -> Callable[[list[dict[str, Any]]], None]:
    def fn(lines: list[dict[str, Any]]) -> None:
        d = lines[0]
        for k in path[:-1]:
            d = d[k]
        assert path[-1] in d, f"fixture: the source regime has no {'.'.join(path)}"
        d[path[-1]] = value

    return fn


REGIME_FIELDS: dict[str, tuple[list[str], Any]] = {
    "system_prompt_sha256": (["system_prompt_sha256"], "0" * 40),
    "tools_sha256": (["tools_sha256"], "0" * 40),
    "cases_sha256": (["cases_sha256"], "0" * 40),
    "versions_lock_sha256": (["versions", "versions_lock_sha256"], "0" * 40),
    "model_revision": (["versions", "model_revision"], "0" * 40),
    "tokenizer_sha256": (["versions", "tokenizer_sha256"], "0" * 40),
    # R11
    "pass1_max_tokens": (["pass1_max_tokens"], 512),
    "enable_thinking": (["enable_thinking"], False),
}


@pytest.mark.parametrize("field", list(REGIME_FIELDS))
def test_a_regime_field_differing_from_the_current_one_is_refused(
    tmp_path: Path, pair: Pair, field: str
) -> None:
    """Criterion 2: the artifact's system-prompt / tools / cases sha, ``versions_lock_sha256``, ``model_revision``
    or ``tokenizer_sha256`` -- and (R11) ``pass1_max_tokens`` or ``enable_thinking`` -- differs from the current one
    -> refused, nothing sent. (Its sha is the edited
    file's, so only the regime field differs.)"""
    source = _source(pair)
    _tp()
    path, value = REGIME_FIELDS[field]
    bad = _edit(source, tmp_path / "src" / "a.jsonl", _regime_edit(path, value))
    _refused(lambda: _replay(tmp_path / "run", bad))


def test_adversarial_an_artifact_from_a_different_calibration_sha_is_refused(
    tmp_path: Path, pair: Pair
) -> None:
    """Criterion 2, adversarial (task file): the artifact's frozen calibration of a cell's arm (``perstep_mar``,
    also the seed cells' base arm) has another sha256 -> refused, nothing sent."""
    source = _source(pair)
    _tp()
    bad = _edit(
        source,
        tmp_path / "src" / "a.jsonl",
        _regime_edit(["calibration", "perstep_mar", "sha256"], "f" * 64),
    )
    _refused(lambda: _replay(tmp_path / "run", bad))


def test_a_calibration_of_an_arm_outside_the_cells_does_not_refuse(
    tmp_path: Path, pair: Pair
) -> None:
    """Criterion 2, the other side (a guard too wide): only the cells' arms are compared. A stored calibration for
    ``perstep_mar_qargmax`` -- not an arm of these cells -- with any sha does not refuse; the replay runs."""
    source = _source(pair)
    _tp()

    def add(lines: list[dict[str, Any]]) -> None:
        lines[0]["calibration"]["perstep_mar_qargmax"] = {
            "path": "elsewhere.json",
            "sha256": "e" * 64,
        }

    ok = _edit(source, tmp_path / "src" / "a.jsonl", add)
    r = _replay(tmp_path / "run", ok, cells=BASE_CELLS)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 0, r.result.stopped
    flushes = int(_trailer(r.result.artifact).get("flush_requests") or 0)
    # per base cell 4 x 1 + picker 2 = 6; 3 cells; + one warm-up per routed case (R10)
    assert len(r.requests) == 3 * 6 + len(ROUTED) + flushes


def _stage1_shaped(lines: list[dict[str, Any]]) -> None:
    """The source as a stage-1 run: cells T1/T2/T4 x perstep_mar only, its seed rows and seed regime gone; its
    calibration names only perstep_mar (as the module's source already does)."""
    lines[0]["cells"] = list(BASE_CELLS)
    lines[0].pop("seed_draft", None)
    assert set(lines[0]["calibration"]) == {"perstep_mar"}, lines[0]["calibration"]
    lines[:] = [x for x in lines if not (x.get("pass") == 2 and x.get("arm") == SEED)]


CLI_CALIBRATION = {**ts.CALIBRATION, "perstep_mar_qargmax": ts.QARGMAX_FILE}


def test_r9_an_absent_calibration_entry_is_accepted_and_recorded(
    tmp_path: Path, pair: Pair
) -> None:
    """R9: a stage-1-shaped source replayed under ``REPLAY_CELLS`` (its regime has no ``perstep_mar_qargmax``
    calibration) is accepted: exit 0, per cell 4 x 1 + picker 2 = 6 pass-2 requests, 4 cells = 24, + 5 warm-ups;
    the regime lists ``calibration_absent_in_source: ["perstep_mar_qargmax"]`` (end_projection needs none; the seed
    arm uses perstep_mar's, which is present)."""
    source = _source(pair)
    _tp()
    src = _edit(source, tmp_path / "src" / "a.jsonl", _stage1_shaped)
    r = _replay(tmp_path / "run", src, cells=REPLAY_CELLS, calibration=CLI_CALIBRATION)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 0, r.result.stopped
    flushes = int(_trailer(r.result.artifact).get("flush_requests") or 0)
    assert len(r.requests) == 4 * 6 + len(ROUTED) + flushes
    regime = ts.artifact_lines(r.result.artifact)[0]
    assert regime.get("calibration_absent_in_source") == ["perstep_mar_qargmax"], regime.get(
        "calibration_absent_in_source"
    )


def test_r9_a_present_calibration_entry_with_another_sha_is_still_refused(
    tmp_path: Path, pair: Pair
) -> None:
    """R9, the other direction: the same stage-1-shaped source whose regime *has* a ``perstep_mar_qargmax`` entry
    with a different sha256 -> refused under ``REPLAY_CELLS``, nothing sent."""
    source = _source(pair)
    _tp()

    def edit(lines: list[dict[str, Any]]) -> None:
        _stage1_shaped(lines)
        lines[0]["calibration"]["perstep_mar_qargmax"] = {"path": "x.json", "sha256": "e" * 64}

    bad = _edit(source, tmp_path / "src" / "a.jsonl", edit)
    _refused(
        lambda: _replay(tmp_path / "run", bad, cells=REPLAY_CELLS, calibration=CLI_CALIBRATION)
    )


def test_r11_recorded_only_regime_keys_do_not_refuse(tmp_path: Path, pair: Pair) -> None:
    """R11, the other side: ``pass2_max_tokens``, ``whitespace`` and ``max_states`` differing in the source are
    recorded, not refused: the replay runs (exit 0)."""
    source = _source(pair)
    _tp()

    def edit(lines: list[dict[str, Any]]) -> None:
        for k in ("pass2_max_tokens", "whitespace", "max_states"):
            assert k in lines[0], f"fixture: no {k} in the source regime"
        lines[0]["pass2_max_tokens"] = {"T1": 512, "T2": 128, "T4": 128}
        lines[0]["whitespace"] = "pretty"
        lines[0]["max_states"] = 64

    ok = _edit(source, tmp_path / "src" / "a.jsonl", edit)
    r = _replay(tmp_path / "run", ok, cells=BASE_CELLS)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 0, r.result.stopped


def test_adversarial_text_edited_to_remove_cc_with_ids_unchanged_is_refused(
    tmp_path: Path, pair: Pair
) -> None:
    """Criterion 2 via T-0082a criterion 10, adversarial (task file): search-02's stored ``text`` has its ``</cc>``
    removed by hand while ``token_ids`` still decode to the closed answer -> refused, nothing sent."""
    source = _source(pair)
    _tp()

    def tamper(lines: list[dict[str, Any]]) -> None:
        row = _p1(lines, "search-02")
        assert row["text"].endswith("</cc>"), "fixture"
        row["text"] = row["text"][: -len("</cc>")]
        row["draft_call_text"] = row["draft_call_text"][: -len("</cc>")]

    bad = _edit(source, tmp_path / "src" / "a.jsonl", tamper)
    _refused(lambda: _replay(tmp_path / "run", bad))


def _set_trailer_stopped(lines: list[dict[str, Any]]) -> None:
    trailers = [x for x in lines if x.get("kind") == "trailer"]
    assert trailers, "fixture: no trailer"
    trailers[-1]["stopped"] = "engine: http_error (case search-04, pass 1)"


def _set_regime_stopped(lines: list[dict[str, Any]]) -> None:
    lines[0]["stopped"] = "h: dirty tree in /x: a.py"


def _p1_error(lines: list[dict[str, Any]]) -> None:
    row = _p1(lines, "search-01")
    for k in ("token_ids", "text", "draft_call_text", "draft_close"):
        row.pop(k, None)
    row["error"] = "http_error"


def _changed_draft_name(lines: list[dict[str, Any]]) -> None:
    _p1(lines, "search-04")["route"]["draft_name"] = "click_at"


def _unknown_case(lines: list[dict[str, Any]]) -> None:
    _p1(lines, "search-06")["case"] = "search-99"


def _no_trailer(lines: list[dict[str, Any]]) -> None:
    assert any(x.get("kind") == "trailer" for x in lines), "fixture"
    lines[:] = [x for x in lines if x.get("kind") != "trailer"]


def _cases_out_of_order(lines: list[dict[str, Any]]) -> None:
    """search-02's and search-03's case blocks (pass-1 row and its pass-2 rows) swapped."""
    blocks: dict[str, list[dict[str, Any]]] = {}
    head = [lines[0]]
    tail = [x for x in lines[1:] if x.get("pass") not in (1, 2)]
    order: list[str] = []
    for x in lines[1:]:
        if x.get("pass") in (1, 2):
            if x["case"] not in blocks:
                order.append(x["case"])
            blocks.setdefault(x["case"], []).append(x)
    i, j = order.index("search-02"), order.index("search-03")
    order[i], order[j] = order[j], order[i]
    lines[:] = head + [x for c in order for x in blocks[c]] + tail


def _cases_subset(lines: list[dict[str, Any]]) -> None:
    """search-06 (pass 1 and its pass-2 rows) dropped: a subset of the stage's first ``limit`` cases."""
    lines[:] = [x for x in lines if x.get("case") != "search-06"]


ARTIFACT_REFUSALS: dict[str, Callable[[list[dict[str, Any]]], None]] = {
    # R12
    "no_trailer": _no_trailer,
    "cases_out_of_order": _cases_out_of_order,
    "cases_subset": _cases_subset,
    "stopped_in_trailer": _set_trailer_stopped,
    "stopped_in_regime": _set_regime_stopped,
    "pass1_row_with_error": _p1_error,
    "rescore_changed_draft_name": _changed_draft_name,
    "rescore_unknown_case": _unknown_case,
}


@pytest.mark.parametrize("case", list(ARTIFACT_REFUSALS))
def test_an_unusable_artifact_is_refused(tmp_path: Path, pair: Pair, case: str) -> None:
    """Criterion 2: ``stopped`` set (trailer or regime), a pass-1 row with an ``error``, and ``rescore``'s
    refusals (a re-route that changes ``draft_name``; a case id not in the cases) -> refused, nothing sent. The
    unedited source replays (the module fixture), so each row's one edit is the cause."""
    source = _source(pair)
    _tp()
    bad = _edit(source, tmp_path / "src" / "a.jsonl", ARTIFACT_REFUSALS[case])
    _refused(lambda: _replay(tmp_path / "run", bad))


def test_an_artifact_sha_mismatch_is_refused(tmp_path: Path, pair: Pair) -> None:
    """Criterion 2: ``pass1_sha256`` != the artifact's sha256 -> refused, nothing sent (the right sha replays: the
    module fixture)."""
    source = _source(pair)
    _tp()
    wrong = hashlib.sha256(source.read_bytes() + b"\n").hexdigest()
    _refused(lambda: _replay(tmp_path / "run", source, sha=wrong))


def test_a_stage_mismatch_is_refused(tmp_path: Path, pair: Pair) -> None:
    """Criterion 2: a dev artifact replayed as ``stage="test"`` -> refused, nothing sent (``stage="dev"``
    replays: the module fixture)."""
    source = _source(pair)
    _tp()
    _refused(lambda: _replay(tmp_path / "run", source, stage="test"))


def test_a_dirty_tree_is_refused(tmp_path: Path, pair: Pair) -> None:
    """Criterion 2 (``hardstop.check_tree``): an untracked file in the repo -> refused, nothing sent."""
    source = _source(pair)
    _tp()
    (tmp_path / "r").mkdir()
    repo, head = es.clean_repo(tmp_path / "r")
    (repo / "stray.py").write_text("x = 1\n")
    _refused(lambda: _replay(tmp_path / "run", source, repo=repo, expected_commit=head))


def test_a_commit_mismatch_is_refused(tmp_path: Path, pair: Pair) -> None:
    """Criterion 2 (``hardstop.check_tree``): HEAD != ``expected_commit`` -> refused, nothing sent."""
    source = _source(pair)
    _tp()
    (tmp_path / "r").mkdir()
    repo, _head = es.clean_repo(tmp_path / "r")
    _refused(lambda: _replay(tmp_path / "run", source, repo=repo, expected_commit="1" * 40))


# =========================================================================== criterion 4: the stops apply


def _last_cd_request_is(r: Run, case: str, shape: str) -> None:
    p2 = [s for s in r.sent if s.kind == "p2"]
    assert p2, "nothing was sent"
    assert (p2[-1].case, p2[-1].shape) == (case, shape), "requests continued after the stop"


def _p2(case: str, scope: str, thought: str, **kw: Any) -> Reply:
    return Reply(text=ts.Scenario().pass2_text(case, scope, thought), steps=ts.P2_STEPS, **kw)


STOPS: dict[str, tuple[tuple[str, str], Callable[[], Reply], dict[str, Any]]] = {
    "engine_http_error": (("search-04", "T2"), lambda: Reply(http_status=500), {}),
    "loud_z0": (("search-04", "T1"), lambda: Reply(error="z0"), {}),
    "cs_violation": (
        ("search-04", "T4"),
        lambda: Reply(text='{"name": "google_search", "arguments": {"q": "x"}}', steps=ts.P2_STEPS),
        {},
    ),
    "join_error": (
        ("search-04", "T1"),
        lambda: _p2("search-04", "google_search", "T1", omit_records=True),
        {"record_wait_s": 0.2, "flush_wait_s": 0.1},
    ),
}


@pytest.mark.parametrize("case", list(STOPS))
def test_the_live_stops_apply_to_a_replay(tmp_path: Path, pair: Pair, case: str) -> None:
    """Criterion 4: the T-0080b stops (engine, CS, loud error, ``join_error``) apply unchanged: exit 1, a
    ``stopped`` trailer, nothing sent after the failing request; still no pass-1 request."""
    source = _source(pair)
    _tp()
    (cid, shape), reply, kw = STOPS[case]
    sc = scenario()
    sc.pass2[(cid, shape)] = reply()
    r = _replay(tmp_path / "run", source, sc=sc, cells=BASE_CELLS, **kw)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 1, (r.result.exit_code, r.result.stopped)
    assert _trailer(r.result.artifact).get("stopped"), "no stopped in the trailer"
    _last_cd_request_is(r, cid, shape)
    warm = _warmup_cases(r.requests)  # R10: only max_tokens-1 warm-ups, at most one per routed case
    assert len(warm) == len(set(warm)) and set(warm) <= set(ROUTED), warm


def test_the_seed_accounting_stop_applies_to_a_replay(tmp_path: Path, pair: Pair) -> None:
    """Criterion 4 (T-0081b ``seed_*``): search-02's seeded T4 request commits canvas 0 at step 24 = the start step
    -> ``seed_accounting`` stop, exit 1, nothing after it."""
    source = _source(pair)
    _tp()
    sc = scenario()
    sc.pass2[("search-02", "T4")] = Reply(
        text=ts.Scenario().pass2_text("search-02", "google_search", "T4"), steps=24
    )
    r = _replay(tmp_path / "run", source, sc=sc, cells=SEED_CELLS)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 1, (r.result.exit_code, r.result.stopped)
    assert "seed_accounting" in str(_trailer(r.result.artifact).get("stopped") or "")
    _last_cd_request_is(r, "search-02", "T4")


# =========================================================================== criterion 1: the CLI


def _cli() -> Any:
    ts.twopass()
    name = "_t0082b_cd_toolcall_cli"
    spec = importlib.util.spec_from_file_location(name, CLI)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    opts = {o for a in m._parser()._actions for o in a.option_strings}
    missing = [o for o in ("--pass1-from", "--pass1-sha256") if o not in opts]
    if missing:
        raise AssertionError(f"T-0082b not implemented: cd_toolcall.py has no {missing}")
    return m


#: Ruling R2: the CLI's replay default (T-0082c's cells), written out here independently of the CLI
REPLAY_CELLS = (
    "T1/named/end_projection",
    "T1/named/perstep_mar",
    "T1/named/perstep_mar_qargmax",
    "T1/named/perstep_mar+seed",
)


def _cli_args(
    srv: Any,
    tmp: Path,
    source: Path,
    sha: str,
    stage: str,
    cells: tuple[str, ...],
    limit: int = 3,
) -> list[str]:
    repo, head = es.clean_repo(tmp)
    args = [
        "--stage",
        stage,
        "--pass1-from",
        str(source),
        "--pass1-sha256",
        sha,
        "--base-url",
        srv.url,
        "--engine-log",
        str(srv.engine_log),
        "--out-dir",
        str(tmp / "out"),
        "--repo",
        str(repo),
        "--expected-commit",
        head,
        "--limit",
        str(limit),
    ]
    for c in cells:
        args += ["--cell", c]
    return args


@pytest.fixture(scope="module")
def split_test_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A live test-split artifact (search-11..13, exact calls) under the R2 replay cells and the frozen
    ``perstep_mar`` and ``perstep_mar_qargmax`` calibrations: 3 + 12 requests."""
    tmp = tmp_path_factory.mktemp("t0082b_testsrc")
    ts.twopass()
    with ts.fake(scenario(), tmp) as srv:
        res = ts.run_driver(
            srv,
            tmp,
            stage="test",
            cells=list(REPLAY_CELLS),
            limit=3,
            calibration={**ts.CALIBRATION, "perstep_mar_qargmax": ts.QARGMAX_FILE},
        )
    assert res.exit_code == 0, res.stopped
    return Path(res.artifact)


def _selection_files(out: Path) -> list[Path]:
    return sorted(p for p in out.rglob("*") if "selection" in p.name)


def test_cli_replay_cells_constant_is_r2s() -> None:
    """Ruling R2: ``cd_toolcall.REPLAY_CELLS`` is T1/named x the four stage-2 arms, in this order."""
    cli = _cli()
    assert tuple(getattr(cli, "REPLAY_CELLS", ())) == REPLAY_CELLS


def test_cli_replay_without_cell_uses_replay_cells_and_writes_no_selection(
    tmp_path: Path, split_test_source: Path
) -> None:
    """Rulings R2 and R6: ``--stage test --pass1-from --pass1-sha256`` with no ``--cell`` replays the R2 cells (the
    artifact's ``cells`` equal them; 3 cases x 4 cells = 12 pass-2 requests, no pass 1), exit 0, and no selection
    file is written."""
    cli = _cli()
    sha = _sha(split_test_source)
    with ts.fake(scenario(), tmp_path) as srv:
        rc = cli.main(_cli_args(srv, tmp_path, split_test_source, sha, "test", ()))
        reqs = list(srv.requests)
    assert rc == 0
    assert _warmup_cases(reqs) == ["search-11", "search-12", "search-13"]  # R10
    assert len([b for b in reqs if cd_keys(b)]) == 12
    out = tmp_path / "out"
    arts = sorted(out.glob("*-toolcall-test.jsonl"))
    assert len(arts) == 1, arts
    assert list(ts.artifact_lines(arts[0])[0]["cells"]) == list(REPLAY_CELLS)
    assert _selection_files(out) == []


def test_cli_dev_replay_writes_no_selection(tmp_path: Path, pair: Pair) -> None:
    """Ruling R6: a replay at ``--stage dev`` (the module's dev source, explicit base cells) runs no stage-1
    selection: exit 0 and no ``selection-dev.json`` or other selection file in the out-dir."""
    source = _source(pair)
    cli = _cli()
    with ts.fake(scenario(), tmp_path) as srv:
        rc = cli.main(
            _cli_args(srv, tmp_path, source, _sha(source), "dev", BASE_CELLS, limit=LIMIT)
        )
        reqs = list(srv.requests)
    assert rc == 0
    assert _warmup_cases(reqs) == ROUTED  # R10
    assert (tmp_path / "out" / "report-dev.json").is_file()
    assert _selection_files(tmp_path / "out") == []


def test_cli_replays_a_test_artifact_without_a_selection(
    tmp_path: Path, split_test_source: Path
) -> None:
    """Criterion 1 and R2/R6: ``cd_toolcall.py --stage test --pass1-from P --pass1-sha256 H --cell ...`` needs no
    ``--selection``, exits 0; the explicit cells (the three ``perstep_mar`` cells) override ``REPLAY_CELLS`` (the
    artifact's ``cells`` are exactly them); only the 9 pass-2 requests (3 cases x 3 cells) are sent;
    ``report-test.{json,md}`` and pass-1 rows with ``replayed_from`` are written; no selection file."""
    cli = _cli()
    sha = _sha(split_test_source)
    with ts.fake(scenario(), tmp_path) as srv:
        rc = cli.main(_cli_args(srv, tmp_path, split_test_source, sha, "test", BASE_CELLS))
        reqs = list(srv.requests)
    assert rc == 0
    assert _warmup_cases(reqs) == ["search-11", "search-12", "search-13"]  # R10
    assert len([b for b in reqs if cd_keys(b)]) == 9
    out = tmp_path / "out"
    assert (out / "report-test.json").is_file() and (out / "report-test.md").is_file()
    arts = sorted(out.glob("*-toolcall-test.jsonl"))
    assert len(arts) == 1, arts
    assert {r.get("replayed_from") for r in ts.pass_rows(arts[0], 1)} == {sha}
    assert list(ts.artifact_lines(arts[0])[0]["cells"]) == list(BASE_CELLS)
    assert _selection_files(out) == []


def test_cli_refuses_a_sha_mismatch_with_exit_2_and_nothing_sent(
    tmp_path: Path, split_test_source: Path
) -> None:
    """Criterion 2 through the CLI: a wrong ``--pass1-sha256`` -> exit 2, no request."""
    cli = _cli()
    wrong = "0" * 64
    with ts.fake(scenario(), tmp_path) as srv:
        rc = cli.main(_cli_args(srv, tmp_path, split_test_source, wrong, "test", BASE_CELLS))
        reqs = list(srv.requests)
    assert rc == 2
    assert reqs == []


# =========================================================================== review F2/F3 (R10)


def test_a_warmup_http_error_is_an_engine_stop(tmp_path: Path, pair: Pair) -> None:
    """R10 + criterion 4 (review F2): search-04's warm-up gets HTTP 500 -> exit 1; the trailer's ``stopped`` names
    the warm-up and the case; the warm-ups of search-02/03 and their pass-2 requests went out, and nothing at all
    (no pass-2 request, no later warm-up) is sent after the failing warm-up."""
    source = _source(pair)
    _tp()
    sc = scenario()
    sc.pass1_reply["search-04"] = Reply(http_status=500)
    r = _replay(tmp_path / "run", source, sc=sc, cells=BASE_CELLS)
    assert r.refusal is None, f"refused: {r.refusal}"
    assert r.result.exit_code == 1, (r.result.exit_code, r.result.stopped)
    stopped = str(_trailer(r.result.artifact).get("stopped") or "")
    assert "warm" in stopped.lower() and "search-04" in stopped, stopped
    seq = [(s.kind, s.case) for s in r.sent if s.case is not None]
    assert seq[-1] == ("p1", "search-04"), seq[-3:]
    assert ("p2", "search-04") not in seq and _warmup_cases(r.requests) == ROUTED[:3]


class _SlowFirst(Scenario):
    """The first pass-2 request of each case (the first cell after its warm-up) hangs ``FIRST_HANG_S``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def __call__(self, index: int, body: Mapping[str, Any]) -> Reply:
        reply = super().__call__(index, body)
        case = ts.case_of(body)
        if cd_keys(body) and case is not None and case not in self.seen:
            self.seen.add(case)
            reply.hang_s = FIRST_HANG_S
        return reply


FIRST_HANG_S = 0.6


def test_pass2_walls_by_position_put_the_first_cell_after_the_warmup_under_first(
    tmp_path: Path, pair: Pair
) -> None:
    """R10 (review F3): the fake delays the first pass-2 request of each routed case by 0.6 s. The report's
    ``pass2_wall_ms_by_position``: ``first`` has one row per routed case (5) and every one of them carries the
    delay (p50 >= 600 ms); ``later`` has the other 2 x 5 = 10 rows (3 base cells) and its median is below 600 ms."""
    source = _source(pair)
    tp = _tp()
    sc = scenario()
    slow = _SlowFirst()
    slow.pass1 = sc.pass1
    r = _replay(tmp_path / "run", source, sc=slow, cells=BASE_CELLS)
    assert r.refusal is None and r.result.exit_code == 0, (r.refusal, r.result.stopped)
    pos = tp.report([r.result.artifact])["pass2_wall_ms_by_position"]
    assert (pos["first"]["n"], pos["later"]["n"]) == (len(ROUTED), 2 * len(ROUTED)), pos
    assert pos["first"]["p50"] >= FIRST_HANG_S * 1000, pos
    assert pos["later"]["p50"] < FIRST_HANG_S * 1000, pos


# =========================================================================== T-0084c criterion 10: native rows are refused


def _native_p1(lines: list[dict[str, Any]]) -> None:
    _p1(lines, "search-04")["format"] = "native"


def _native_p2(lines: list[dict[str, Any]]) -> None:
    row = next(x for x in lines if x.get("pass") == 2 and x.get("case") == "search-04")
    row["format"] = "native"
    row["cell"] = "native/" + row["cell"]


@pytest.mark.parametrize("edit", [_native_p1, _native_p2], ids=["pass1", "pass2"])
def test_t0084c_a_source_containing_a_native_row_is_refused(
    tmp_path: Path, pair: Pair, edit: Callable[[list[dict[str, Any]]], None]
) -> None:
    """T-0084c criterion 10: ``pass1_from`` refuses an artifact containing a native row (``format: "native"``) with
    ``ValueError`` before any socket -- zero requests, zero connections. The edited row is otherwise the stored one,
    so nothing else (case order, rescore, regime) can be the reason; the message names the format."""
    source = _source(pair)
    _tp()
    bad = _edit(source, tmp_path / "src" / "native.jsonl", edit)
    r = _replay(tmp_path / "run", bad)
    assert r.refusal is not None, f"a source with a native row was accepted: {r.result.exit_code}"
    assert "native" in str(r.refusal), r.refusal
    _refused(lambda: r)


def test_t0084c_a_source_whose_rows_say_format_cc_is_still_replayed(
    tmp_path: Path, pair: Pair
) -> None:
    """T-0084c criterion 10, the other direction: rows that carry ``format: "cc"`` (an artifact written after
    T-0084c) are replayed as before -- exit 0, the hand-counted requests. Green before the implementation by design:
    it catches a refusal keyed on the mere presence of ``format``."""

    def all_cc(lines: list[dict[str, Any]]) -> None:
        for x in lines:
            if x.get("pass") in (1, 2):
                x["format"] = "cc"

    source = _source(pair)
    ok = _edit(source, tmp_path / "src" / "cc.jsonl", all_cc)
    r = _replay(tmp_path / "run", ok)
    assert r.refusal is None, r.refusal
    assert r.result.exit_code == 0, r.result.stopped
    assert len(r.requests) == len(pair.get().replay.requests)  # type: ignore[union-attr]


def test_t0084c_c14_a_replay_with_a_native_cell_is_refused(tmp_path: Path, pair: Pair) -> None:
    """Ruling C14 (criterion 10): a replay whose cells include a native cell is refused (``ValueError``, before any
    socket) -- it would mix a live native pass 1 with replayed cc rows."""
    source = _source(pair)
    r = _replay(tmp_path / "run", source, cells=(*CELLS, "native/T1/named/perstep_mar"))
    assert r.refusal is not None, f"accepted: exit {r.result.exit_code}"
    assert "native" in str(r.refusal), r.refusal
    _refused(lambda: r)
