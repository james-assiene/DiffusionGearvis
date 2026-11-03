"""T-0070 criterion 4 (and T-0062 decision 11): what each arm puts on the wire, and that the knobs are
recorded in every row.

``unconstrained`` sends no ``astra_cd_*`` key. The four constrained arms send a T-0064 ``CdRequest``
(checked here by the plugin's own ``parse_cd_xargs``, which the fake applies to every request):
end_projection -> (end_projection, model), perstep_mf -> (perstep_mf, model), perstep_mar ->
(perstep_mar, model), perstep_mar_sc -> (perstep_mar, constrained); the Mar arms carry the frozen
calibration's ``entropy_bound`` and ``confidence_threshold``. The automaton sent is the compiled grammar
of that very record (payload equality with an independent compile; T-0064: equal automata give equal
strings). BFCL constrained arms compile with standard JSON whitespace; the pretty-whitespace parity arm
compiles ``BFCL_BASELINE_OPTIONS`` and runs only the records with |S| <= 512 under it (T-0063's report:
14/130 above 512). Regime: CPU, fake server.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd import _wire_support as ws
from tests.cd.fakes.vllm_server import Reply, user_prompt

EXPECTED_MODE = {
    "end_projection": ("end_projection", "model"),
    "perstep_mf": ("perstep_mf", "model"),
    "perstep_mar": ("perstep_mar", "model"),
    "perstep_mar_sc": ("perstep_mar", "constrained"),
    # T-0076 criterion 8: the settling arm and its SC twin (constrained SC)
    "perstep_mar_qargmax": ("perstep_mar_qargmax", "model"),
    "perstep_mar_qargmax_sc": ("perstep_mar_qargmax", "constrained"),
}
#: the arm whose frozen calibration file each calibrated arm runs on (T-0076 criterion 8, Q3: an SC twin takes
#: its base arm's file)
CALIB_SOURCE = {
    "perstep_mar": "perstep_mar",
    "perstep_mar_sc": "perstep_mar",
    "perstep_mar_qargmax": "perstep_mar_qargmax",
    "perstep_mar_qargmax_sc": "perstep_mar_qargmax",
}
#: T-0063 implementation report: the BFCL test-cut records above 512 states under BFCL_BASELINE_OPTIONS.
PRETTY_ABOVE_512 = {
    "live_simple_44-18-0",
    "live_simple_45-18-1",
    "live_simple_48-21-0",
    "live_simple_49-21-1",
    "live_simple_51-23-0",
    "live_simple_52-23-1",
    "live_simple_70-34-0",
    "live_simple_71-35-0",
    "live_simple_106-63-0",
    "live_simple_110-67-0",
    "live_simple_111-67-1",
    "live_simple_112-68-0",
    "live_simple_117-73-0",
    "live_simple_122-78-0",
}


def test_unconstrained_sends_no_cd_key(tmp_path: Path) -> None:
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="unconstrained", limit=4)
        bodies = list(srv.requests)
    assert res.exit_code == 0, res.stopped
    assert len(bodies) == 4
    for b in bodies:
        assert not [k for k in (b.get("vllm_xargs") or {}) if k.startswith("astra_cd_")]
        assert not [k for k in b if k.startswith("astra_cd_")]


@pytest.mark.parametrize("arm", sorted(EXPECTED_MODE))
def test_constrained_arm_sends_cd_request_with_recorded_knobs(tmp_path: Path, arm: str) -> None:
    cs.warm_sudoku("test")
    calib = {"entropy_bound": 0.03, "confidence_threshold": 5e-4}
    extra = (
        {"calibration": es.freeze_calibration(tmp_path, arm=CALIB_SOURCE[arm], **calib)}
        if arm.startswith("perstep_mar")
        else {}
    )
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=4, **extra)
        sent = srv.cd_requests()
        bodies = list(srv.requests)
    assert res.exit_code == 0, res.stopped
    assert len(sent) == 4
    mode, sc = EXPECTED_MODE[arm]
    ids = es.prompt_to_id("sudoku", "test")
    by_id = es.sudoku_by_id("test")
    tags = set()
    for body, keys in zip(bodies, sent, strict=True):
        ws.parse(body["vllm_xargs"])  # the plugin accepts it (a refusal would raise CdRequestError)
        assert keys["astra_cd_mode"] == mode
        assert keys.get("astra_cd_self_conditioning", "model") == sc
        if arm.startswith("perstep_mar"):
            assert keys["astra_cd_entropy_bound"] == pytest.approx(calib["entropy_bound"])
            assert keys["astra_cd_confidence_threshold"] == pytest.approx(
                calib["confidence_threshold"]
            )
        payload = keys["astra_cd_automaton"]
        assert (
            keys["astra_cd_automaton_sha256"] == hashlib.sha256(payload.encode("ascii")).hexdigest()
        )
        puzzle, _ = by_id[ids[user_prompt(body)]]
        assert payload == ws.encode(cs.compiled_sudoku(puzzle)), (
            "the automaton is not this record's grammar"
        )
        tags.add(keys["astra_cd_record_tag"])
    assert len(tags) == 4, "record_tag must be unique per request (it is the join key)"
    _, rows, _ = es.read_artifact(res.artifact)
    assert len(rows) == 4
    for row, keys in zip(rows, sent, strict=True):
        knobs = row["knobs"]
        for k, v in keys.items():
            if k == "astra_cd_automaton":
                continue
            name = k[len("astra_cd_") :]
            assert knobs[name] == v, (name, knobs)


def test_mar_sc_differs_from_mar_only_in_self_conditioning(tmp_path: Path) -> None:
    cs.warm_sudoku("test")
    calib = es.freeze_calibration(tmp_path)
    sent = {}
    for arm in ("perstep_mar", "perstep_mar_sc"):
        d = tmp_path / arm
        d.mkdir()
        with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), d) as srv:
            res = es.run(srv, d, task="sudoku", cut="test", arm=arm, limit=2, calibration=calib)
            sent[arm] = srv.cd_requests()
        assert res.exit_code == 0, res.stopped
    drop = {"astra_cd_record_tag", "astra_cd_self_conditioning"}
    a = [{k: v for k, v in x.items() if k not in drop} for x in sent["perstep_mar"]]
    b = [{k: v for k, v in x.items() if k not in drop} for x in sent["perstep_mar_sc"]]
    assert a == b


def test_arm_specs_have_the_qargmax_arms_needing_calibration() -> None:
    """T-0076 criterion 8: both new arms are in ``ARM_SPECS`` with the wire mode/SC above and need calibration;
    the pre-existing five are unchanged."""
    arms = es.need("local_astra.eval.cd.arms")
    specs = arms.ARM_SPECS
    for arm in ("perstep_mar_qargmax", "perstep_mar_qargmax_sc"):
        assert arm in specs, f"{arm} missing from ARM_SPECS {sorted(specs)}"
        s = specs[arm]
        assert (s.mode, s.self_conditioning) == EXPECTED_MODE[arm], s
        assert s.needs_calibration is True, s
    old = {
        "unconstrained": (None, None, False),
        "end_projection": ("end_projection", "model", False),
        "perstep_mf": ("perstep_mf", "model", False),
        "perstep_mar": ("perstep_mar", "model", True),
        "perstep_mar_sc": ("perstep_mar", "constrained", True),
    }
    for arm, want in old.items():
        s = specs[arm]
        assert (s.mode, s.self_conditioning, s.needs_calibration) == want, (arm, s)


def test_qargmax_sc_differs_from_qargmax_only_in_self_conditioning(tmp_path: Path) -> None:
    """Q3: the SC twin reuses the base arm's frozen knobs; on the wire only the self-conditioning differs."""
    cs.warm_sudoku("test")
    calib = es.freeze_calibration(tmp_path, arm="perstep_mar_qargmax")
    sent = {}
    for arm in ("perstep_mar_qargmax", "perstep_mar_qargmax_sc"):
        d = tmp_path / arm
        d.mkdir()
        with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), d) as srv:
            res = es.run(srv, d, task="sudoku", cut="test", arm=arm, limit=2, calibration=calib)
            sent[arm] = srv.cd_requests()
        assert res.exit_code == 0, res.stopped
    assert [x["astra_cd_self_conditioning"] for x in sent["perstep_mar_qargmax_sc"]] == [
        "constrained"
    ] * 2
    assert [x.get("astra_cd_self_conditioning", "model") for x in sent["perstep_mar_qargmax"]] == [
        "model"
    ] * 2
    drop = {"astra_cd_record_tag", "astra_cd_self_conditioning"}
    a = [{k: v for k, v in x.items() if k not in drop} for x in sent["perstep_mar_qargmax"]]
    b = [{k: v for k, v in x.items() if k not in drop} for x in sent["perstep_mar_qargmax_sc"]]
    assert a == b


def test_bfcl_constrained_arms_use_standard_json_whitespace(tmp_path: Path) -> None:
    """T-0062 decision 11: BFCL_BASELINE_OPTIONS with whitespace overridden to "json"."""
    ids = es.cuts().load("bfcl", "test")[:2]
    resolve = es.BfclResolver("test")
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        return Reply(text=preserved[resolve(body)])  # RFC 8259 admits the pretty separators

    with es.fake(script, tmp_path) as srv:
        res = es.run(srv, tmp_path, task="bfcl", cut="test", arm="end_projection", limit=2)
        bodies = list(srv.requests)
    assert res.exit_code == 0, res.stopped
    assert len(bodies) == 2
    for body, rid in zip(bodies, resolve.resolved, strict=True):
        want = ws.encode(cs.grammars().compile_bfcl(cs.bfcl_function(rid), whitespace="json"))
        assert body["vllm_xargs"]["astra_cd_automaton"] == want, rid
        pretty = ws.encode(cs.compiled_bfcl(rid))
        assert body["vllm_xargs"]["astra_cd_automaton"] != pretty
    regime, _, _ = es.read_artifact(res.artifact)
    assert regime["whitespace"] == "json"
    assert [r.id for r in ids] == resolve.resolved


def test_pretty_parity_arm_runs_only_records_at_most_512_states(tmp_path: Path) -> None:
    """The JAX-link arm: pretty whitespace, restricted to |S| <= 512 under it (116 of 130), labelled.

    Requests are identified by their automaton, not by prompt: three pairs of test-cut records share a
    prompt with different functions (``BfclResolver``)."""
    test_ids = cs.bfcl_test_cut_ids()
    cs.warm_bfcl(test_ids)
    resolve = es.BfclResolver("test")
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        return Reply(text=preserved[resolve(body)])

    with es.fake(script, tmp_path) as srv:
        res = es.run(
            srv, tmp_path, task="bfcl", cut="test", arm="end_projection", whitespace="pretty"
        )
        payloads = [b["vllm_xargs"]["astra_cd_automaton"] for b in srv.requests]
    seen = list(resolve.resolved)
    assert res.exit_code == 0, res.stopped
    assert len(seen) == 116 and len(set(seen)) == 116, "each eligible record exactly once"
    assert set(seen) == set(test_ids) - PRETTY_ABOVE_512
    collided = {"live_simple_32-9-0", "live_simple_33-10-0", "live_simple_36-13-0"}
    for rid, payload in zip(seen, payloads, strict=True):
        if rid in collided or rid == seen[0]:
            assert payload == ws.encode(cs.compiled_bfcl(rid)), rid
    regime, rows, _ = es.read_artifact(res.artifact)
    assert regime["whitespace"] == "pretty"
    assert regime["n"] == 116
    assert sorted(r["id"] for r in rows) == sorted(seen)
