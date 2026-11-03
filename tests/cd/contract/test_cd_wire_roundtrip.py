"""T-0064 criteria 2 and 5: the wire round trip over T-0063's real automata, and the payload size report.

Path under test (criterion 2): compiled automaton -> ``artifact.encode_automaton`` -> ``CdRequest(...)
.to_xargs()`` -> ``json.dumps``/``json.loads`` -> plugin ``parse_cd_xargs``. The oracle is the compiled
automaton itself: every array must come back with the same dtype, shape and values.

Inputs (regime: CPU; T-0063's compiler at this tree's commit; tokenizer.json sha256 cc8d3a0c...bfe0f; model
rev 2ea8372...; diffgemma-fa 27c7d0d):
- BFCL-Live simple, test cut (n = 130) and dev cut (n = 100), compiled with ``BFCL_BASELINE_OPTIONS``
  (the **pretty**-whitespace parity baseline). The standard-JSON-whitespace variant of decision 11 is not
  covered here: it has no warm compile cache and is a strictly smaller grammar family.
- Sudoku 4x4 test (n = 250, seed 0) and dev (n = 100, seed 1).
- Hand-built NFAs whose outgoing label sets from one state overlap without being identical (one of them
  stored as a complement class).

Automata whose bucket exceeds 512 (14 of the BFCL test cut, 1 of dev, per T-0063's report) cannot travel:
the plugin supports buckets up to 512 (T-0062 decision 5), so their row asserts the ``bucket_too_large``
refusal (or ``payload_too_large`` if the payload itself is over the cap) instead of equality.

This is a CPU wire contract with no service behind it, so there is no fake/real ``backend`` pair; the
end-to-end check that ``vllm_xargs`` reaches ``extra_args`` on the engine is T-0069 criterion 3.
"""

from __future__ import annotations

import statistics

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _wire_support as ws

pytestmark = pytest.mark.contract

CUTS = ("bfcl_test", "bfcl_dev", "sudoku_test", "sudoku_dev")


def _automata(cut: str) -> list[tuple[str, object]]:
    if cut.startswith("bfcl"):
        ids = cs.bfcl_test_cut_ids() if cut == "bfcl_test" else cs.bfcl_dev_cut_ids()
        cs.warm_bfcl(ids)
        return [(rid, cs.compiled_bfcl(rid)) for rid in ids]
    which = cut.split("_")[1]
    cs.warm_sudoku(which)
    return [(rid, cs.compiled_sudoku(p)) for rid, p, _ in cs.sudoku_cut(which)]


def _check_one(rid: str, a: object, mode: str) -> tuple[str | None, int]:
    """Return (failure or None, payload length)."""
    payload = ws.encode(a)
    n = len(payload)
    if a.n_states_bucket > 512 or n > ws.MIB:
        want = ws.R_SIZE if n > ws.MIB else ws.R_BUCKET
        x = ws.raw_xargs(payload, mode=mode) if n > ws.MIB else ws.request_xargs(payload, mode=mode)
        try:
            ws.expect_error(x, want)
        except AssertionError as exc:
            return f"{rid} (bucket {a.n_states_bucket}, {n} B): {exc}", n
        return None, n
    spec = ws.parse(ws.request_xargs(payload, mode=mode, record_tag="rt." + rid[:40]))
    if spec is None:
        return f"{rid}: parse returned None for a constrained request", n
    bad = ws.array_mismatches(a, spec.automaton)
    return (f"{rid}: {bad}" if bad else None), n


_MODES = ("end_projection", "perstep_mf", "perstep_mar")


@pytest.mark.parametrize("cut", CUTS)
def test_roundtrip_every_automaton_of_the_cut(cut):
    autos = _automata(cut)
    expected_n = {"bfcl_test": 130, "bfcl_dev": 100, "sudoku_test": 250, "sudoku_dev": 100}[cut]
    assert len(autos) == expected_n, f"{cut}: {len(autos)} automata, expected {expected_n}"
    failures = []
    for i, (rid, a) in enumerate(autos):
        fail, _ = _check_one(rid, a, _MODES[i % 3])
        if fail:
            failures.append(fail)
    assert not failures, f"{len(failures)}/{len(autos)} failed:\n" + "\n".join(failures[:30])


@pytest.mark.parametrize("builder", ["nfa_overlapping", "nfa_wide_overlap"])
@pytest.mark.parametrize("mode", _MODES)
def test_roundtrip_hand_built_nfa_with_overlapping_edges(builder, mode):
    a = getattr(ws, builder)()
    spec = ws.parse(ws.request_xargs(ws.encode(a), mode=mode))
    assert spec.mode == mode
    assert not ws.array_mismatches(a, spec.automaton)


def test_roundtrip_echoes_the_request_knobs():
    a = ws.nfa_overlapping()
    x = ws.request_xargs(
        ws.encode(a),
        mode="perstep_mar",
        self_conditioning="constrained",
        record_tag="bfcl:live_simple_3-2-0.A",
        entropy_bound=0.25,
        confidence_threshold=0.75,
        max_bucket=16,
    )
    spec = ws.parse(x)
    assert (spec.mode, spec.self_conditioning, spec.record_tag) == (
        "perstep_mar",
        "constrained",
        "bfcl:live_simple_3-2-0.A",
    )
    assert (spec.entropy_bound, spec.confidence_threshold, spec.max_bucket) == (0.25, 0.75, 16)


# --------------------------------------------------------------------------- T-0076 criterion 1

#: the new mode; the cut rows above keep ``_MODES[i % 3]`` so every pre-existing case decodes unchanged
QARGMAX = "perstep_mar_qargmax"


@pytest.mark.parametrize("builder", ["nfa_overlapping", "nfa_wide_overlap"])
@pytest.mark.parametrize("sc", ["model", "constrained"])
def test_qargmax_roundtrips_pydantic_to_plugin(builder, sc):
    """T-0076 criterion 1: ``CdRequest(mode="perstep_mar_qargmax", ...)`` -> ``to_xargs`` -> JSON -> plugin
    ``parse_cd_xargs`` gives back the mode, the self-conditioning and the automaton unchanged."""
    a = getattr(ws, builder)()
    spec = ws.parse(ws.request_xargs(ws.encode(a), mode=QARGMAX, self_conditioning=sc))
    assert (spec.mode, spec.self_conditioning) == (QARGMAX, sc)
    assert not ws.array_mismatches(a, spec.automaton)


def test_qargmax_roundtrip_echoes_the_overrides():
    """P13 for the new mode: ``entropy_bound`` / ``confidence_threshold`` travel like ``perstep_mar``'s."""
    a = ws.nfa_overlapping()
    x = ws.request_xargs(
        ws.encode(a),
        mode=QARGMAX,
        self_conditioning="constrained",
        record_tag="sudoku:dev-7.qa",
        entropy_bound=0.3,
        confidence_threshold=0.6,
        max_bucket=16,
    )
    assert x["astra_cd_mode"] == QARGMAX, x["astra_cd_mode"]
    spec = ws.parse(x)
    assert (spec.mode, spec.self_conditioning, spec.record_tag) == (
        QARGMAX,
        "constrained",
        "sudoku:dev-7.qa",
    )
    assert (spec.entropy_bound, spec.confidence_threshold, spec.max_bucket) == (0.3, 0.6, 16)


def test_qargmax_sudoku_dev_cut_roundtrips():
    """The new mode over T-0063's real Sudoku dev automata (n = 100): every array comes back."""
    autos = _automata("sudoku_dev")
    failures = [f for rid, a in autos if (f := _check_one(rid, a, QARGMAX)[0])]
    assert not failures, f"{len(failures)}/{len(autos)} failed:\n" + "\n".join(failures[:30])


def test_the_bucket_512_edge_roundtrips():
    a = ws.with_bucket(512)
    spec = ws.parse(ws.request_xargs(ws.encode(a), max_bucket=512))
    assert not ws.array_mismatches(a, spec.automaton)


def test_payload_size_report(capsys):
    """Criterion 5: p50 and max of the encoded payload per cut, printed with the regime; every automaton
    the plugin can take (bucket <= 512) must fit the 1 MiB cap, or it cannot be sent at all."""
    lines = []
    too_big = []
    for cut in CUTS:
        sizes = []
        for rid, a in _automata(cut):
            n = len(ws.encode(a))
            sizes.append(n)
            if a.n_states_bucket <= 512 and n > ws.MIB:
                too_big.append(f"{cut}/{rid}: bucket {a.n_states_bucket}, {n} B")
        lines.append(
            f"{cut}: n={len(sizes)} p50={statistics.median(sizes):.0f} B max={max(sizes)} B "
            f"(cap {ws.MIB} B)"
        )
    with capsys.disabled():
        print(
            "\n[T-0064 criterion 5] encoded payload bytes; BFCL = BFCL_BASELINE_OPTIONS (pretty ws), "
            "Sudoku 4x4; CPU; tokenizer cc8d3a0c; model rev 2ea8372; dgfa 27c7d0d"
        )
        for line in lines:
            print("  " + line)
    assert not too_big, "automata within the bucket limit but over the payload cap:\n" + "\n".join(
        too_big
    )


# --------------------------------------------------------------------------- T-0081 criterion 1: the draft keys


@pytest.mark.parametrize("builder", ["nfa_overlapping", "nfa_wide_overlap"])
@pytest.mark.parametrize("mode", [*_MODES, QARGMAX])
def test_seeded_request_roundtrips_with_its_automaton(builder, mode):
    """A seeded request travels pydantic -> JSON -> plugin with the automaton arrays and the seed unchanged."""
    from tests.cd.unit import _seed_support as sd

    a = getattr(ws, builder)()
    ids = [(i * 104_729) % ws.VOCAB_SIZE for i in range(256)]
    x = sd.request_xargs(
        ws.encode(a), mode=mode, draft=sd.b64(ids), draft_noise=0.5, draft_start_step=24
    )
    spec = sd.parse_ok(x)
    assert spec.mode == mode
    assert not ws.array_mismatches(a, spec.automaton)
    assert sd.spec_seed(spec) == (ids, 0.5, 24)


def test_unseeded_sudoku_dev_cut_decodes_with_no_seed():
    """Every pre-existing wire case decodes to the same spec, now with no seed (T-0063's Sudoku dev automata)."""
    from tests.cd.unit import _seed_support as sd

    autos = _automata("sudoku_dev")
    bad = []
    for rid, a in autos:
        x = ws.request_xargs(ws.encode(a), mode="perstep_mar", record_tag="rt." + rid[:40])
        assert not [k for k in x if k.startswith("astra_cd_draft")], x.keys()
        spec = ws.parse(x)
        if ws.array_mismatches(a, spec.automaton) or sd.spec_seed(spec) != (None, 0.0, 0):
            bad.append(rid)
    assert bad == [], bad[:10]
