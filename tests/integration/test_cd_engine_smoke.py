"""T-0069 criteria 4-9, re-checked independently from the tier-3 smoke's saved evidence (``-m gpu``).

Run after ``/launch`` 2 with ``ASTRA_T0069_EVIDENCE=logs/t0069-smoke-<sha>`` (and ``uv run --group cd``: the replay
compiles grammars). Unset: the rows skip (nothing to check on this box). **Set but the directory or any file below
missing: the rows fail** -- a smoke whose evidence is gone has not been shown.

This module never reads the driver's verdicts (``report.txt`` is only required to exist). It re-derives every check
from the raw evidence with its own calls: T-0070 ``replay.compile_automaton``/``accepts`` and ``records`` (engine-log
join on ``record_tag``), T-0011 ``b16_tap.last_boot``/``boot_config_lines``/``parse_tap_release_lines``, the plugin's
own ``parse_cd_xargs`` for the host-predicted codes, and the contract's ``AstraMessageAdapter`` for frames. It does not
import ``local_astra/eval/cd/smoke.py``.

Evidence layout (fixed here; the driver ``scripts/gpu/cd_smoke.py`` writes it):

``requests.jsonl``
    one JSON object per HTTP request, in send order: ``name`` (below), ``task`` (``sudoku``/``bfcl``/``literal``/null),
    ``dataset_id`` (dev-cut id or null), ``arm`` (T-0070 arm name; ``unconstrained`` for none), ``record_tag``
    (null when unconstrained), ``max_tokens``, ``xargs`` (exactly the ``vllm_xargs`` sent, automaton included, or
    null), ``status`` (HTTP status), ``response`` (the JSON body, or null), and for criterion 6 ``literal_ids``.
    Names: ``c4:<sudoku|bfcl>:<end_projection|perstep_mf|perstep_mar|perstep_mar_sc>``; ``c5:G``, ``c5:X``,
    ``c5:next``, ``c5:health`` (``status`` is ``/health``'s); ``c6:perstep_mar``, ``c6:end_projection``;
    ``c7:near``, ``c7:over``; ``c8:quiet``; ``c9:header:<0..19>``.
``frames.jsonl``
    every ``astra.canvas.step`` frame the tap published during the run, decoded, in arrival order.
``engine.log``
    the engine log from the start of the run's boot (its ``non-default args:`` line) to the end of the run.
``c8-window.log``
    criterion 8's log window: from before ``c8:quiet`` was sent until after its flush request returned.
``report.txt``
    the driver's report (written before its exit code).

The boot reference is ``logs/t0011-tier3-806da12/boot-reference.txt`` beside the evidence directory.
Criterion 9's first bullet (the two ``backend=real`` rows of ``tests/cd/contract/test_eval_fake_server_contract.py``)
is its own command on the same boot; this module checks ``choices[0].token_ids`` in the evidence.
Regime: as the run's (``docs/tasks/T-0069.md`` Regime, tier 3); printed per row from ``report.txt``'s first line.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

ENV = "ASTRA_T0069_EVIDENCE"
REPO = Path(__file__).resolve().parents[2]
FILES = ("requests.jsonl", "frames.jsonl", "engine.log", "c8-window.log", "report.txt")
TASKS = ("sudoku", "bfcl")
ARMS = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc")
MODE = {
    "end_projection": "end_projection",
    "perstep_mf": "perstep_mf",
    "perstep_mar": "perstep_mar",
    "perstep_mar_sc": "perstep_mar",
}
STOPS = frozenset({1, 50, 106})
EOS = 1
CL = 256
CAP = 1 << 20  # T-0062 decision 4
HEADER_ID = 100
ENGINE_ID = re.compile(r"^(?P<resp>.+)-[0-9a-f]{8}$")


# --------------------------------------------------------------------------- evidence


class Evidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: dict[str, dict[str, Any]] = {}
        for line in (root / "requests.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                self.requests[row["name"]] = row
        self.frames: list[dict[str, Any]] = [
            json.loads(x) for x in (root / "frames.jsonl").read_text().splitlines() if x.strip()
        ]
        self.log = (root / "engine.log").read_text()
        self.window = (root / "c8-window.log").read_text()
        self.report = (root / "report.txt").read_text()
        m = re.fullmatch(r"t0069-smoke-([0-9a-f]{7,40})", root.name)
        self.sha = m.group(1) if m else None

    def req(self, name: str) -> dict[str, Any]:
        if name not in self.requests:
            pytest.fail(f"evidence has no request {name!r} (have {sorted(self.requests)[:12]}...)")
        return self.requests[name]

    def frames_of(self, name: str) -> list[dict[str, Any]]:
        resp = (self.req(name).get("response") or {}).get("id")
        assert resp, f"{name}: no response id"
        out = []
        for f in self.frames:
            m = ENGINE_ID.match(str(f.get("request_id", "")))
            if m and m.group("resp") == resp:
                out.append(f)
        return out

    def frame_index(self, name: str) -> list[int]:
        mine = {id(f) for f in self.frames_of(name)}
        return [i for i, f in enumerate(self.frames) if id(f) in mine]

    def records(self, tag: str) -> list[Any]:
        from local_astra.eval.cd import records as rec

        log = rec.EngineLog(self.root / "engine.log", from_start=True)
        log.poll()
        joined = rec.join(log.records(tag))
        assert not joined.duplicate_idx, f"{tag}: duplicated canvas_idx {joined.duplicate_idx}"
        return list(joined.records)


@pytest.fixture(scope="module")
def ev() -> Evidence:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(f"{ENV} not set: no tier-3 evidence to re-check on this box")
    root = Path(raw)
    if not root.is_absolute():
        root = REPO / root
    if not root.is_dir():
        pytest.fail(f"{ENV}={raw} but {root} does not exist: the smoke's evidence is missing")
    missing = [f for f in FILES if not (root / f).is_file()]
    if missing:
        pytest.fail(f"{ENV}={raw} but evidence files are missing: {missing}")
    e = Evidence(root)
    if not e.report.strip():
        pytest.fail("report.txt is empty: the driver must write its report before its exit code")
    return e


@pytest.fixture(autouse=True)
def _regime(request: pytest.FixtureRequest) -> None:
    if os.environ.get(ENV):
        e = request.getfixturevalue("ev")
        print(f"T-0069 tier-3 re-check [{request.node.name}] evidence {e.root.name}: "
              f"{e.report.splitlines()[0] if e.report else '<no report>'}")  # fmt: skip


def _ok(row: Mapping[str, Any]) -> dict[str, Any]:
    assert row.get("status") == 200, f"{row['name']}: HTTP {row.get('status')}"
    body = row.get("response")
    assert isinstance(body, dict), f"{row['name']}: no response body"
    return body


def _content(body: Mapping[str, Any]) -> str | None:
    return body["choices"][0]["message"].get("content")


def _frame_problems(frames: Sequence[Mapping[str, Any]]) -> list[str]:
    from local_astra.contracts.protocol import AstraCanvasStep, AstraMessageAdapter

    bad = []
    for f in frames:
        try:
            m = AstraMessageAdapter.validate_python(dict(f))
        except Exception as exc:
            bad.append(f"invalid frame {f.get('request_id')}#{f.get('canvas_idx')}: {exc}")
            continue
        if not isinstance(m, AstraCanvasStep):
            bad.append(f"not a canvas step: {f.get('type')}")
        elif m.accepted is not None or m.entropy is not None:
            bad.append(f"accepted/entropy filled on {m.request_id}#{m.canvas_idx}")
        elif m.phase == "denoise" and not 0 <= m.step <= m.max_steps:
            bad.append(f"denoise step {m.step} outside 0..{m.max_steps} (R15)")
        elif m.phase == "commit" and not 0 <= m.step <= m.max_steps:
            bad.append(f"commit step {m.step} outside 0..{m.max_steps} (R14, ruling O6)")
    return bad


def _commits(frames: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((f for f in frames if f.get("phase") == "commit"), key=lambda f: f["canvas_idx"])


def _ids(recs: Sequence[Any]) -> list[int]:
    return [int(t) for r in recs for t in r.committed_ids]


def _host_code(xargs: Mapping[str, Any]) -> str | None:
    """The host's own ``parse_cd_xargs`` (the plugin's parser, stdlib + numpy): the code it predicts."""
    from tests.cd import _wire_support as ws

    mod = ws.plugin("xargs")
    try:
        mod.parse_cd_xargs(dict(xargs))
    except mod.CdRequestError as exc:
        return str(exc.reason)
    return None


def _release_rows(e: Evidence, name: str) -> list[dict[str, Any]]:
    from local_astra.bench.b16_tap import last_boot, parse_tap_release_lines

    resp = e.req(name)["response"]["id"]
    out = []
    for row in parse_tap_release_lines(last_boot(e.log)):
        m = ENGINE_ID.match(str(row["req_id"]))
        if m and m.group("resp") == resp:
            out.append(row)
    return out


def _case(task: str, dataset_id: str) -> Any:
    from local_astra.eval.cd import cuts

    for c in cuts.load(task, "dev"):
        if c.id == dataset_id:
            return c
    pytest.fail(f"{dataset_id!r} is not in the {task} dev cut (the test cuts stay untouched)")


# --------------------------------------------------------------------------- the frames of the whole run


def test_every_tap_frame_of_the_run_validates(ev: Evidence) -> None:
    """Criterion 4: R10 shapes, R14/R15 step ranges, ``accepted``/``entropy`` None, on every frame."""
    assert ev.frames, "no tap frame was recorded: the tap was off or the subscriber missed the run"
    bad = _frame_problems(ev.frames)
    assert bad == [], f"{len(bad)} invalid frames:\n" + "\n".join(bad[:20])


# --------------------------------------------------------------------------- criterion 4


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("task", TASKS)
def test_c4_each_arm_runs_end_to_end(ev: Evidence, task: str, arm: str) -> None:
    name = f"c4:{task}:{arm}"
    row = ev.req(name)
    body = _ok(row)
    tag = row["record_tag"]
    assert tag and row["xargs"] and row["xargs"].get("astra_cd_record_tag") == tag, row["xargs"]
    recs = ev.records(tag)
    # (a) tag echoed (by the join) and the arm named
    assert recs, f"{name}: no ASTRA_CD record for tag {tag}: xargs did not reach extra_args"
    assert {r.arm for r in recs} == {MODE[arm]}, f"{name}: records name {[r.arm for r in recs]}"
    # (b) contiguous, closed by a stop or an error; error null on all eight
    assert [r.canvas_idx for r in recs] == list(range(len(recs))), [r.canvas_idx for r in recs]
    assert recs[-1].stop_pos is not None or recs[-1].error is not None, recs[-1]
    assert [r.error for r in recs] == [None] * len(recs), (
        f"{name}: errors {[r.error for r in recs]}"
    )
    # (c) host replay with this module's own compile; the compiled automaton is the one that was sent
    from local_astra.decoding.artifact import encode_automaton
    from local_astra.eval.cd import replay

    automaton = replay.compile_automaton(task, _case(task, row["dataset_id"]), whitespace="json")
    sent = row["xargs"]["astra_cd_automaton"]
    assert encode_automaton(automaton) == sent, (
        f"{name}: the automaton sent is not the record's grammar"
    )
    accepted = replay.accepts(automaton, _ids(recs))
    assert accepted, f"{name}: host replay rejects the committed ids"
    assert accepted == recs[-1].in_final, (
        f"{name}: replay {accepted} vs plugin in_final {recs[-1].in_final}"
    )
    # (d) a release line with commit_overrun == 0
    rel = _release_rows(ev, name)
    assert rel and all(r["commit_overrun"] == 0 for r in rel), f"{name}: release lines {rel}"
    # (e) commit frames: one per record, committed_tokens == committed_ids
    commits = _commits(ev.frames_of(name))
    assert [c["canvas_idx"] for c in commits] == [r.canvas_idx for r in recs], (
        f"{name}: commit frames at {[c['canvas_idx'] for c in commits]}, records {len(recs)}"
    )
    assert [list(c["committed_tokens"]) for c in commits] == [list(r.committed_ids) for r in recs]
    assert body["choices"][0]["message"] is not None


def test_c4_reports_preempted_records(ev: Evidence) -> None:
    """T-0067 S10: the number of ``preempted`` records is reported (not a gate); it must be derivable."""
    n = 0
    for task in TASKS:
        for arm in ARMS:
            n += sum(
                r.error == "preempted" for r in ev.records(ev.req(f"c4:{task}:{arm}")["record_tag"])
            )
    print(f"T-0069 c4 preempted records: {n} of 8 requests")
    assert "preempted" in ev.report, "the driver's report does not state the preempted count"


# --------------------------------------------------------------------------- criterion 5


def test_c5_x_is_loud_with_the_host_predicted_code(ev: Evidence) -> None:
    row = ev.req("c5:X")
    body = _ok(row)
    assert not _content(body), f"X returned content {_content(body)!r}"
    payload = row["xargs"]["astra_cd_automaton"]
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    assert row["xargs"]["astra_cd_automaton_sha256"] == digest, "X's sha was not recomputed"
    predicted = _host_code(row["xargs"])
    assert predicted in ("bad_encoding", "structure"), f"host parse predicts {predicted!r}"
    recs = ev.records(row["record_tag"])
    assert len(recs) == 1 and recs[0].canvas_idx == 0, recs
    assert list(recs[0].committed_ids)[:1] == [EOS], recs[0].committed_ids
    assert recs[0].error == predicted, f"X's record says {recs[0].error}, host predicts {predicted}"


def test_c5_x_was_sent_while_g_was_denoising(ev: Evidence) -> None:
    g_idx = ev.frame_index("c5:G")
    x_idx = ev.frame_index("c5:X")
    g_first_denoise = next(i for i in g_idx if ev.frames[i].get("phase") == "denoise")
    assert x_idx, "X published no frame"
    assert g_first_denoise < x_idx[0] < g_idx[-1], (
        f"G frames {g_idx[0]}..{g_idx[-1]} (first denoise {g_first_denoise}), X's first {x_idx[0]}: "
        "the two requests did not overlap"
    )


def test_c5_g_completes_and_the_engine_keeps_serving(ev: Evidence) -> None:
    from local_astra.eval.cd import replay

    row = ev.req("c5:G")
    _ok(row)
    recs = ev.records(row["record_tag"])
    assert recs and [r.error for r in recs] == [None] * len(recs), recs
    assert [r.canvas_idx for r in recs] == list(range(len(recs)))
    automaton = replay.compile_automaton(row["task"], _case(row["task"], row["dataset_id"]))
    assert replay.accepts(automaton, _ids(recs)), "G is not accepted by host replay"
    nxt = _ok(ev.req("c5:next"))
    assert nxt["choices"][0]["message"] is not None
    assert ev.req("c5:next")["xargs"] in (None, {}), "the next request must be unconstrained"
    assert ev.req("c5:health")["status"] == 200


# --------------------------------------------------------------------------- criterion 6


def _shortest_accepting_path(automaton: Any) -> int:
    """BFS over the decoded wire automaton: tokens from a start state to a final state. The compiler's stop
    augmentation makes ACC the only final state, reached through a stop edge, so the stop id is counted: a
    literal of n ids gives n + 1 (verified 2026-10-03 on a hand-built 270-id chain: 271, ``d[start]`` 271)."""
    import numpy as np

    starts = set(np.flatnonzero(automaton.start_vector).tolist())
    finals = set(np.flatnonzero(automaton.is_final).tolist())
    adj: dict[int, set[int]] = collections.defaultdict(set)
    for s, d in zip(automaton.edge_src.tolist(), automaton.edge_dst.tolist(), strict=True):
        adj[s].add(d)
    dist = {s: 0 for s in starts}
    q = collections.deque(starts)
    while q:
        s = q.popleft()
        if s in finals:
            return dist[s]
        for d in adj[s]:
            if d not in dist:
                dist[d] = dist[s] + 1
                q.append(d)
    raise AssertionError("no accepting path")


@pytest.mark.parametrize("arm", ["perstep_mar", "end_projection"])
def test_c6_a_bucket_512_literal_closes_exactly_two_canvases(ev: Evidence, arm: str) -> None:
    from tests.cd import _wire_support as ws

    row = ev.req(f"c6:{arm}")
    body = _ok(row)
    lit = [int(t) for t in row["literal_ids"]]
    assert 260 <= len(lit) <= 290, len(lit)
    spec = ws.plugin("xargs").parse_cd_xargs(dict(row["xargs"]))
    assert spec.mode == MODE[arm] and spec.automaton.n_states_bucket == 512, (
        spec.mode,
        spec.automaton.n_states_bucket,
    )
    shortest = _shortest_accepting_path(spec.automaton)
    # literal_ids is recorded without the stop; the shortest accepted output is literal + stop. It is longer than
    # one canvas (CL 256), so no accepted output fits in canvas 0 and the request must span two canvases.
    assert shortest == len(lit) + 1, (
        f"shortest accepted length {shortest}, literal {len(lit)} + stop"
    )
    assert shortest > CL, f"shortest accepted length {shortest} fits in one canvas of {CL}"
    assert row["max_tokens"] == 300
    recs = ev.records(row["record_tag"])
    assert [r.canvas_idx for r in recs] == [0, 1], f"canvases {[r.canvas_idx for r in recs]}"
    assert [r.error for r in recs] == [None, None], (
        f"errors {[r.error for r in recs]} (internal = OOM)"
    )
    assert {r.bucket for r in recs} == {512}
    ids = _ids(recs)
    assert ids[: len(lit)] == lit and len(ids) > len(lit) and ids[len(lit)] in STOPS, (
        "replay of the literal: the committed ids are not the literal followed by a stop id"
    )
    assert recs[-1].in_final is True
    assert body["usage"]["completion_tokens"] <= 300, body["usage"]


# --------------------------------------------------------------------------- criterion 7


def test_c7_the_payload_arrives_intact_at_the_cap(ev: Evidence) -> None:
    near, over = ev.req("c7:near"), ev.req("c7:over")
    for row, length, want in (
        (near, CAP - 512, "bad_encoding"),
        (over, CAP + 1, "payload_too_large"),
    ):
        payload = row["xargs"]["astra_cd_automaton"]
        assert len(payload) == length and payload.isascii(), (row["name"], len(payload))
        assert row["status"] == 200, (
            f"{row['name']}: HTTP {row['status']}: the API server refused the body; inline base64 at 1 MiB is "
            "not viable -- this goes to the human (ruling O3), never a silent retry"
        )
        recs = ev.records(row["record_tag"])
        assert len(recs) == 1, f"{row['name']}: {len(recs)} records"
        predicted = _host_code(row["xargs"])
        assert recs[0].error == predicted == want, (
            f"{row['name']}: record {recs[0].error}, host {predicted}, criterion 7 wants {want}"
        )
    digest = hashlib.sha256(near["xargs"]["astra_cd_automaton"].encode("ascii")).hexdigest()
    assert near["xargs"]["astra_cd_automaton_sha256"] == digest, (
        "the near-cap request was not sent with the correct sha: its code would prove nothing"
    )


# --------------------------------------------------------------------------- criterion 8


def test_c8_the_boot_equals_the_806da12_reference(ev: Evidence) -> None:
    from local_astra.bench.b16_tap import boot_config_lines, last_boot

    ref_path = ev.root.parent / "t0011-tier3-806da12" / "boot-reference.txt"
    assert ref_path.is_file(), f"{ref_path} missing: criterion 8's reference"
    ref = [ln for ln in ref_path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert boot_config_lines(last_boot(ev.log)) == sorted(ref)


def test_c8_services_vllm_sh_is_unchanged_since_806da12(ev: Evidence) -> None:
    assert ev.sha, f"evidence directory {ev.root.name} does not name its commit"
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", f"806da12..{ev.sha}", "--", "services/vllm.sh"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", f"services/vllm.sh changed since 806da12:\n{proc.stdout}"


def test_c8_an_unconstrained_request_writes_no_cd_line_and_commits_by_canvas(ev: Evidence) -> None:
    row = ev.req("c8:quiet")
    body = _ok(row)
    assert row["xargs"] in (None, {}), "the quiet request must carry no xargs"
    resp = body["id"]
    assert re.search(rf"canvas tap release: req_id={re.escape(resp)}-[0-9a-f]{{8}}\b", ev.window), (
        "the window does not contain the quiet request's release line: it does not cover the request"
    )
    cd_lines = [ln for ln in ev.window.splitlines() if "ASTRA_CD " in ln or "astra_cd " in ln]
    assert cd_lines == [], f"CD lines in the unconstrained window: {cd_lines[:3]}"
    frames = ev.frames_of("c8:quiet")
    assert _frame_problems(frames) == []
    want = math.ceil(body["usage"]["completion_tokens"] / CL)
    assert len(_commits(frames)) == want, f"{len(_commits(frames))} commit frames, want {want}"


# --------------------------------------------------------------------------- criterion 9


def test_c9_twenty_unconstrained_dev_records_carry_token_ids(ev: Evidence) -> None:
    from local_astra.eval.cd import cuts

    rows = [ev.req(f"c9:header:{k}") for k in range(20)]
    by_task = collections.Counter(r["task"] for r in rows)
    assert by_task == {"sudoku": 10, "bfcl": 10}, by_task
    for r in rows:
        dev = {c.id for c in cuts.load(r["task"], "dev")}
        test = {c.id for c in cuts.load(r["task"], "test")}
        assert r["dataset_id"] in dev and r["dataset_id"] not in test, r["dataset_id"]
        assert r["xargs"] in (None, {}), "header records are unconstrained"
    first8 = []
    for r in rows:
        ids = _ok(r)["choices"][0].get("token_ids")
        assert isinstance(ids, list) and ids, (
            f"{r['name']}: choices[0].token_ids absent (T-0070 E5)"
        )
        first8.append(ids[:8])
    rate = sum(ids[0] == HEADER_ID for ids in first8) / len(first8)
    print(f"T-0069 c9 header-prefix rate {rate:.2f} (n=20); first 8 ids: {first8}")


def test_c9_constrained_token_ids_are_the_committed_ids_up_to_the_stop(ev: Evidence) -> None:
    included: dict[str, bool] = {}
    for task in TASKS:
        for arm in ARMS:
            name = f"c4:{task}:{arm}"
            row = ev.req(name)
            ids = _ok(row)["choices"][0].get("token_ids")
            assert isinstance(ids, list), f"{name}: no token_ids"
            committed = _ids(ev.records(row["record_tag"]))
            cut = next((i for i, t in enumerate(committed) if t in STOPS), len(committed))
            if ids == committed[:cut]:
                included[name] = False
            elif ids == committed[: cut + 1]:
                included[name] = True
            else:
                pytest.fail(f"{name}: token_ids {ids[:12]}... != committed ids up to the stop")
    print(f"T-0069 c9 stop id included in token_ids: {included}")
