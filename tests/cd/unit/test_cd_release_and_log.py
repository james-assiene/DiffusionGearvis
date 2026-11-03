"""T-0067 rulings S10-S12 (review, 2026-09-30).

- S10: a request re-added with a req_id CD has already seen (preemption / resume) is not restarted from A_0: it
  ends within ≤ 2 calls with an EOS-first commit and one error record ``preempted``; no second canvas-0 stream.
- S11: releasing a request flushes its records: the last canvas record of a finished or refused request is in the
  log with no further ``__call__``.
- S12: every log line that starts with ``ASTRA_CD `` is a JSON record (the refusal warning uses another prefix).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from tests.cd.fakes.diffusion_sampler import Cfg, World
from tests.cd.unit._sampler_support import G1, Grammar, Records, cd_xargs
from tests.cd.unit._scenario import canvas_logits

EOS = 1


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0067-rel-g1")


def _run_until_commit(w: World, req: str, fn: Any, limit: int = 8) -> tuple[int, list[int]]:
    for _ in range(limit):
        s = w.decode(fn)
        if s.committed(req) is not None:
            return s.call, s.committed(req)  # type: ignore[return-value]
    raise AssertionError(f"{req} did not commit within {limit} calls")


def test_s10_preempted_request_ends_loudly(g1: Grammar, caplog: Any, capsys: Any) -> None:
    records = Records(caplog, capsys)
    fn = canvas_logits({"c": 11, "u": 12}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    xargs = cd_xargs(g1, "s10")
    w.add("c", xargs=xargs)
    w.add("u")
    w.prefill()
    _run_until_commit(w, "c", fn)
    w.decode(fn)
    w.remove("c")  # preempted
    w.add("c", xargs=xargs)  # resumed: same req_id, same options
    w.prefill()
    commits = []
    first_decode = w.calls
    for _ in range(4):
        s = w.decode(fn)
        if s.committed("c") is not None:
            commits.append((s.call, s.committed("c")))
    assert commits, "the resumed request never committed"
    call, ids = commits[0]
    assert ids[0] == EOS, f"resumed request restarted instead of ending: committed {ids}"
    assert call <= first_decode + 1, "EOS commit later than 2 calls after the resume"
    mine = records.tagged("s10")
    errs = [r for r in mine if r.get("error") is not None]
    assert [r["error"] for r in errs] == ["preempted"], mine
    zeros = [r for r in mine if r.get("canvas_idx") == 0 and r.get("error") is None]
    assert len(zeros) == 1, f"a second canvas-0 stream after the resume: {mine}"


@pytest.mark.parametrize("kind", ["finished", "refused"])
def test_s11_release_flushes_the_last_record(
    kind: str, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    fn = canvas_logits({"c": 21}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    xargs = cd_xargs(g1, f"s11-{kind}")
    if kind == "refused":
        xargs = {**xargs, "astra_cd_entropy_bound": -1.0}
    w.add("c", xargs=xargs)
    w.prefill()
    _call, ids = _run_until_commit(w, "c", fn)
    w.remove("c")  # the engine finishes the request; no further sampler call follows
    mine = records.tagged(f"s11-{kind}")
    assert any(r.get("committed_ids") == ids for r in mine) or (
        kind == "refused" and any(r.get("error") == "bad_value" for r in mine)
    ), f"the last record is not in the log after release: {mine}"


def test_s12_only_json_records_use_the_prefix(g1: Grammar, caplog: Any, capsys: Any) -> None:
    caplog.set_level(logging.DEBUG)
    fn = canvas_logits({"c": 31, "d": 32}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    w.add(
        "c", xargs={**cd_xargs(g1, "s12-bad"), "astra_cd_entropy_bound": -1.0}
    )  # refused: a warning
    w.add("d", xargs=cd_xargs(g1, "s12-ok"), top_k=5)  # refused: unsupported option
    w.prefill()
    for _ in range(4):
        w.decode(fn)
    cap = capsys.readouterr()
    lines = [r.getMessage() for r in caplog.records] + (cap.out + cap.err).splitlines()
    prefixed = [ln for msg in lines for ln in msg.splitlines() if ln.startswith("ASTRA_CD ")]
    assert prefixed, "fixture: no ASTRA_CD line at all"
    bad = []
    for ln in prefixed:
        try:
            ok = isinstance(json.loads(ln[len("ASTRA_CD ") :]), dict)
        except ValueError:
            ok = False
        if not ok:
            bad.append(ln)
    assert bad == [], f"non-JSON lines carry the record prefix: {bad[:3]}"


@pytest.mark.parametrize(
    "extra", [3, 0], ids=["prefill_longer_than_prompt", "prefill_equals_prompt"]
)
def test_s10_prefill_hint_on_a_fresh_request_id(
    extra: int, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    """Ruling S10/S14: request ids get a random suffix on resume, so CD may never have seen the id. A new
    request whose ``new_req_data.prefill_token_ids`` is longer than its prompt (the V2 scheduler re-sends
    ``_all_token_ids``) ends as ``preempted``: EOS-first commit within ≤ 2 calls of its first decode, exactly
    one error record. With the prefill equal to the prompt it is an ordinary constrained request: no error
    record, first commit not EOS-first, one canvas-0 record."""
    records = Records(caplog, capsys)
    cfg = Cfg(CL=4, stability_threshold=1)
    fn = canvas_logits({"r": 11, "u": 12}, period=3)
    w = World(cfg, cd=True)
    tag = f"s10-hint-{extra}"
    prompt = [2] * cfg.prompt_len
    w.add("u")
    w.add("r", xargs=cd_xargs(g1, tag), prefill_token_ids=prompt + [40, 41, 42][:extra])
    w.prefill()
    first_decode = w.calls
    commits = []
    for _ in range(4):
        s = w.decode(fn)
        if s.committed("r") is not None:
            commits.append((s.call, s.committed("r")))
    assert commits, "the request never committed"
    call, ids = commits[0]
    mine = records.tagged(tag)
    errs = [r["error"] for r in mine if r.get("error") is not None]
    if extra:
        assert ids[0] == EOS, f"resumed (prefill > prompt) request was restarted: committed {ids}"
        assert call <= first_decode + 1, "EOS commit later than 2 calls after the first decode"
        assert errs == ["preempted"], mine
    else:
        assert errs == [], f"an ordinary request (prefill == prompt) was ended: {mine}"
        assert ids[0] != EOS, f"an ordinary request committed EOS first: {ids}"
        zeros = [r for r in mine if r.get("canvas_idx") == 0]
        assert len(zeros) == 1, mine
