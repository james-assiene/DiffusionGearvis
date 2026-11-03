"""T-0064 criteria 1, 3 and 4 on the plugin side: ``parse_cd_xargs`` refuses every malformed, corrupt or
misspelt request with a typed ``CdRequestError`` carrying a stable reason code, never a partial object and
never a silent "unconstrained".

API and reason codes: ruling W1 in ``tests/cd/_wire_support.py``. Corrupt payloads are crafted through
``artifact.unpack_payload``/``pack_payload`` (the raw codec) so the rows do not depend on the byte format;
every crafted family has a control row proving the unmutated repack parses, so a row goes red for the
mutation and not for the repack.
"""

from __future__ import annotations

import functools
import json
import random

import numpy as np
import pytest

from tests.cd import _compile_support as cs
from tests.cd import _wire_support as ws
from tests.cd._wire_support import (
    KEYS,
    MIB,
    R_BUCKET,
    R_ENCODING,
    R_MISSING_ARRAY,
    R_MISSING_KEY,
    R_NO_MODE,
    R_SHA,
    R_SIZE,
    R_STRUCTURE,
    R_TOKENIZER,
    R_UNKNOWN_KEY,
    R_VALUE,
    R_VERSION,
    R_VOCAB,
    REASONS,
    expect_error,
    parse,
    raw_xargs,
    sha,
)


@functools.cache
def _sudoku(i: int) -> object:
    _, puzzle, _ = cs.sudoku_cut("test")[i]
    return cs.compiled_sudoku(puzzle)


@functools.cache
def _payload(i: int = 0) -> str:
    return ws.encode(_sudoku(i))


@functools.cache
def _nfa_payload() -> str:
    return ws.encode(ws.nfa_overlapping())


def _crafted(mutate, base: str | None = None) -> str:
    """Unpack ``base`` (default: the Sudoku DFA payload), apply ``mutate(meta, arrays)`` in place, repack
    (no validation)."""
    art = ws.artifact()
    meta, arrays = art.unpack_payload(_payload() if base is None else base)
    meta = dict(meta)
    arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    mutate(meta, arrays)
    return art.pack_payload(meta, arrays)


# --------------------------------------------------------------------------- criterion 4: key handling


@pytest.mark.parametrize("xargs", [None, {}, {"astra_tap": True}, {"astra_tap": 1, "other": "x"}])
def test_no_cd_keys_is_unconstrained(xargs):
    assert parse(xargs) is None


def test_foreign_keys_are_ignored_beside_a_valid_request():
    x = raw_xargs(_payload(), mode="end_projection")
    spec = parse({**x, "astra_tap": True, "astra_other": 3, "vllm_whatever": "y"})
    assert spec is not None
    assert spec.mode == "end_projection"
    assert not ws.array_mismatches(_sudoku(0), spec.automaton)


def test_cd_keys_without_mode_are_refused():
    x = raw_xargs(_payload())
    del x[KEYS["mode"]]
    expect_error(x, R_NO_MODE)


@pytest.mark.parametrize("only", [k for k in KEYS if k != "mode"])
def test_a_lone_cd_key_without_mode_is_refused(only):
    full = raw_xargs(_payload(), entropy_bound=0.5, confidence_threshold=0.9, max_bucket=64)
    expect_error({KEYS[only]: full[KEYS[only]], "astra_tap": True}, R_NO_MODE)


def _suffix_typos(key: str) -> list[str]:
    """Every single-character typo in the part after ``astra_cd_``: deletion, adjacent transposition,
    substitution, plus a case change and a trailing space. A typo inside the prefix makes a non-CD key,
    which criterion 4 says is ignored (see the report's ambiguity note)."""
    prefix, suffix = "astra_cd_", key[len("astra_cd_") :]
    out = set()
    for i in range(len(suffix)):
        out.add(suffix[:i] + suffix[i + 1 :])
        sub = "x" if suffix[i] != "x" else "y"
        out.add(suffix[:i] + sub + suffix[i + 1 :])
        if i + 1 < len(suffix) and suffix[i] != suffix[i + 1]:
            out.add(suffix[:i] + suffix[i + 1] + suffix[i] + suffix[i + 2 :])
    out.add(suffix[0].upper() + suffix[1:])
    out.add(suffix + " ")
    valid = set(KEYS.values())
    return sorted(k for k in (prefix + s for s in out if s) if k not in valid)


_TYPOS = [(field, typo) for field, key in KEYS.items() for typo in _suffix_typos(key)]


def _full() -> dict:
    return raw_xargs(_payload(), entropy_bound=0.5, confidence_threshold=0.9, max_bucket=64)


def test_the_full_key_set_parses():
    """Control for the typo rows: the unmistyped dict is accepted and every knob is echoed."""
    spec = parse(_full())
    assert (spec.mode, spec.self_conditioning, spec.record_tag) == (
        "perstep_mar",
        "model",
        "t0064-row",
    )
    assert (spec.entropy_bound, spec.confidence_threshold, spec.max_bucket) == (0.5, 0.9, 64)


def test_mdoe_is_refused_and_named():
    x = _full()
    x["astra_cd_mdoe"] = x.pop(KEYS["mode"])
    err = expect_error(x, R_UNKNOWN_KEY)
    assert "astra_cd_mdoe" in str(err)


@pytest.mark.parametrize(("field", "typo"), _TYPOS, ids=[t for _, t in _TYPOS])
def test_typo_replacing_a_key_is_refused(field, typo):
    x = _full()
    x[typo] = x.pop(KEYS[field])
    err = expect_error(x, R_UNKNOWN_KEY)
    assert typo in str(err), f"the error does not name the offending key: {err}"


@pytest.mark.parametrize(("field", "typo"), _TYPOS, ids=[t for _, t in _TYPOS])
def test_typo_beside_the_valid_key_is_refused(field, typo):
    x = _full()
    x[typo] = x[KEYS[field]]
    expect_error(x, R_UNKNOWN_KEY)


@pytest.mark.parametrize("field", ["automaton", "automaton_sha256", "record_tag"])
def test_missing_required_key_is_refused(field):
    x = raw_xargs(_payload())
    del x[KEYS[field]]
    expect_error(x, R_MISSING_KEY)


def test_optional_knobs_absent_are_none():
    spec = parse(raw_xargs(_payload()))
    assert (spec.entropy_bound, spec.confidence_threshold, spec.max_bucket) == (None, None, None)


# --------------------------------------------------------------------------- criterion 1 mirrored: values


@pytest.mark.parametrize(
    ("mode", "sc"),
    [
        ("end_projection", "model"),
        ("perstep_mf", "model"),
        ("perstep_mf", "constrained"),
        ("perstep_mar", "model"),
        ("perstep_mar", "constrained"),
        ("perstep_mar_qargmax", "model"),  # T-0076 criterion 1
        ("perstep_mar_qargmax", "constrained"),  # T-0076 criterion 1 (the SC twin's wire form)
    ],
)
def test_valid_mode_and_self_conditioning(mode, sc):
    spec = parse(raw_xargs(_payload(), mode=mode, self_conditioning=sc))
    assert (spec.mode, spec.self_conditioning) == (mode, sc)


@pytest.mark.parametrize(
    "fields",
    [
        {"mode": "perstep_foo"},
        {"mode": "PERSTEP_MAR"},
        # T-0076: the SC twin is an eval arm name, not a wire mode; near-misses of the new mode are refused
        {"mode": "perstep_mar_qargmax_sc"},
        {"mode": "perstep_qargmax"},
        {"mode": "PERSTEP_MAR_QARGMAX"},
        {"mode": "perstep_mar_qargmax "},
        {"mode": 1},
        {"mode": "end_projection", "self_conditioning": "constrained"},
        {"self_conditioning": "none"},
        {"entropy_bound": 0},
        {"entropy_bound": -0.1},
        {"entropy_bound": "0.5"},
        {"confidence_threshold": 0.0},
        {"confidence_threshold": -1},
        {"max_bucket": 48},
        {"max_bucket": 8},
        {"max_bucket": 1024},
        {"max_bucket": "64"},
        {"record_tag": ""},
        {"record_tag": "a" * 65},
        {"record_tag": "a b"},
        {"record_tag": "tag/1"},
        {"record_tag": "tag\n"},
        {"record_tag": 7},
    ],
    ids=lambda f: json.dumps(f),
)
def test_bad_values_are_refused(fields):
    base = {"mode": "perstep_mar", "self_conditioning": "model", "record_tag": "t0064-row"}
    base.update({k: v for k, v in fields.items() if k in base})
    extra = {k: v for k, v in fields.items() if k not in base}
    expect_error(raw_xargs(_payload(), **base, **extra), R_VALUE)


def test_qargmax_mode_takes_the_per_step_overrides():
    """T-0076 criterion 1 (P13 for the new mode): ``perstep_mar_qargmax`` with ``entropy_bound`` /
    ``confidence_threshold`` and constrained SC parses and echoes every knob."""
    spec = parse(
        raw_xargs(
            _payload(),
            mode="perstep_mar_qargmax",
            self_conditioning="constrained",
            entropy_bound=0.25,
            confidence_threshold=0.75,
            max_bucket=64,
        )
    )
    assert (spec.mode, spec.self_conditioning) == ("perstep_mar_qargmax", "constrained")
    assert (spec.entropy_bound, spec.confidence_threshold, spec.max_bucket) == (0.25, 0.75, 64)


def test_qargmax_is_in_the_plugin_mode_lists():
    """T-0076 criterion 1: the mode is in ``xargs.MODES``, ``requests.SERVED_MODES`` and ``steps.PERSTEP_MODES``,
    and every pre-existing mode is still there (the lists grow, nothing is renamed)."""
    old = ("end_projection", "perstep_mf", "perstep_mar")
    xargs_modes = tuple(ws.plugin("xargs").MODES)
    served = tuple(ws.plugin("requests").SERVED_MODES)
    perstep = tuple(ws.plugin("steps").PERSTEP_MODES)
    assert "perstep_mar_qargmax" in xargs_modes and set(old) <= set(xargs_modes), xargs_modes
    assert "perstep_mar_qargmax" in served and set(old) <= set(served), served
    assert "perstep_mar_qargmax" in perstep and {"perstep_mf", "perstep_mar"} <= set(perstep), (
        perstep
    )
    assert "end_projection" not in perstep, perstep


@pytest.mark.parametrize("tag", ["a", "A" * 64, "A-z_0.9:x-", "bfcl:live_simple_0-0-0.r1"])
def test_record_tag_edge_values_are_accepted(tag):
    assert parse(raw_xargs(_payload(), record_tag=tag)).record_tag == tag


# --------------------------------------------------------------------------- criterion 3: integrity


def test_sha_mismatch():
    p = _payload()
    expect_error(raw_xargs(p, digest="0" * 64), R_SHA)
    expect_error(raw_xargs(p, digest=sha(p)[::-1]), R_SHA)


def test_valid_sha_of_a_different_automaton_is_refused_cold():
    """Adversarial: the sha is a real content address -- of another Sudoku automaton with the same
    shape -- so only the integrity check distinguishes the two."""
    a, b = _payload(0), _payload(1)
    assert a != b, "fixture precondition: two different puzzles give different payloads"
    expect_error(raw_xargs(b, digest=sha(a)), R_SHA)


def test_valid_sha_of_a_different_automaton_is_refused_after_that_sha_was_cached():
    """Adversarial, aimed at 'cache decoded automata by sha': once A is cached under sha(A), payload B
    sent with sha(A) must be refused, not answered from the cache; B under its own sha decodes as B."""
    a, b = _payload(0), _payload(1)
    assert not ws.array_mismatches(_sudoku(0), parse(raw_xargs(a)).automaton)
    expect_error(raw_xargs(b, digest=sha(a)), R_SHA)
    expect_error(raw_xargs(a[: len(a) // 2], digest=sha(a)), R_SHA)
    spec_b = parse(raw_xargs(b))
    assert not ws.array_mismatches(_sudoku(1), spec_b.automaton), "B decoded as something else"
    # and the refusal is stable on a repeat
    expect_error(raw_xargs(b, digest=sha(a)), R_SHA)


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda p: p[:-1], id="truncated-by-one"),
        pytest.param(lambda p: p[: (len(p) // 2) // 4 * 4], id="truncated-on-a-quad"),
        pytest.param(lambda p: p[:100] + "!" + p[101:], id="invalid-char"),
        pytest.param(lambda p: p[:100] + " " + p[100:], id="embedded-space"),
        pytest.param(lambda p: "QUJDRA==", id="valid-base64-not-a-container"),
        pytest.param(lambda p: "", id="empty"),
    ],
)
def test_truncated_or_invalid_base64(corrupt):
    bad = corrupt(_payload())
    expect_error(raw_xargs(bad), R_ENCODING)


def test_repack_control_parses():
    """Control: the codec round trip itself is accepted, so the mutation rows below test the mutation."""
    spec = parse(raw_xargs(_crafted(lambda m, a: None)))
    assert not ws.array_mismatches(_sudoku(0), spec.automaton)


@pytest.mark.parametrize("delta", [1, 999])
def test_unknown_format_version(delta):
    def mutate(meta, arrays):
        meta["format_version"] = meta["format_version"] + delta

    expect_error(raw_xargs(_crafted(mutate)), R_VERSION)


@pytest.mark.parametrize("vocab", [262_143, 262_145, 256_000])
def test_vocab_size_mismatch(vocab):
    def mutate(meta, arrays):
        meta["vocab_size"] = vocab

    expect_error(raw_xargs(_crafted(mutate)), R_VOCAB)


@pytest.mark.parametrize("tok", ["0" * 64, "", ws.TOKENIZER_SHA256.upper()[:-1] + "0"])
def test_tokenizer_hash_mismatch(tok):
    def mutate(meta, arrays):
        meta["tokenizer_hash"] = tok

    expect_error(raw_xargs(_crafted(mutate)), R_TOKENIZER)


def test_tokenizer_hash_from_the_encoder_is_checked():
    """The real encoder path: an automaton stamped with another tokenizer never decodes."""
    a = ws.replace_meta(ws.nfa_overlapping(), tokenizer_hash="f" * 64)
    expect_error(raw_xargs(ws.encode(a)), R_TOKENIZER)


def _array_names() -> list[str]:
    return sorted(ws.ARRAY_NAMES)


@pytest.mark.parametrize("name", _array_names())
def test_missing_array(name):
    art = ws.artifact()
    _, arrays = art.unpack_payload(_payload())
    if name not in arrays:
        pytest.fail(
            f"the payload does not carry {name!r} (carries {sorted(arrays)}); criterion 2 needs every "
            "compiled array on the wire"
        )

    def mutate(meta, arrs):
        del arrs[name]

    expect_error(raw_xargs(_crafted(mutate)), R_MISSING_ARRAY)


def _dup_edge(meta, a):
    a["edge_src"] = np.append(a["edge_src"], a["edge_src"][0])
    a["edge_dst"] = np.append(a["edge_dst"], a["edge_dst"][0])
    a["edge_class"] = np.append(a["edge_class"], a["edge_class"][0])
    if "class_id" in a:
        a["class_id"] = np.append(a["class_id"], a["class_id"][0])
    if "n_edges" in meta:
        meta["n_edges"] += 1


def _set(name, idx, value):
    def mutate(meta, a):
        a[name][idx] = value(meta, a) if callable(value) else value

    return mutate


_STRUCTURAL = {
    "edge-dst-equals-bucket": _set("edge_dst", 0, lambda m, a: a["start_vector"].shape[0]),
    "edge-src-negative": _set("edge_src", 0, -1),
    "edge-class-past-the-classes": _set("edge_class", 0, lambda m, a: a["sum_is_neg"].shape[0]),
    "token-id-past-the-vocabulary": _set("sum_indices", 0, ws.VOCAB_SIZE),
    "indptr-end-past-indices": _set("sum_indptr", -1, lambda m, a: a["sum_indices"].shape[0] + 1),
    "start-vector-longer-than-bucket": lambda m, a: a.__setitem__(
        "start_vector", np.append(a["start_vector"], False)
    ),
    "is-final-shorter-than-bucket": lambda m, a: a.__setitem__("is_final", a["is_final"][:-1]),
    "edge-arrays-of-different-lengths": lambda m, a: a.__setitem__("edge_dst", a["edge_dst"][:-1]),
    "meta-bucket-disagrees-with-arrays": lambda m, a: m.__setitem__(
        "n_states_bucket", m["n_states_bucket"] * 2
    ),
}


@pytest.mark.parametrize("mutation", sorted(_STRUCTURAL))
def test_structural_invariant_violation(mutation):
    expect_error(raw_xargs(_crafted(_STRUCTURAL[mutation])), R_STRUCTURE)


def test_meta_of_one_automaton_with_arrays_of_another_is_refused():
    """Adversarial, the other reading of 'valid sha over a different automaton's arrays': the sha is
    valid over the payload, but the payload's arrays belong to a different automaton than its meta."""
    art = ws.artifact()
    meta_a, _ = art.unpack_payload(_payload())
    _, arrays_b = art.unpack_payload(ws.encode(ws.nfa_overlapping()))
    expect_error(raw_xargs(art.pack_payload(dict(meta_a), dict(arrays_b))), R_STRUCTURE)


# --------------------------------------------------------------------------- bucket limit (adversarial)


def test_bucket_exactly_at_max_bucket_is_accepted_and_one_ladder_step_below_refused():
    a = _sudoku(0)
    assert a.n_states_bucket == 64, (
        "fixture precondition: Sudoku 4x4 compiles to bucket 64 (T-0063)"
    )
    spec = parse(raw_xargs(_payload(), max_bucket=64))
    assert not ws.array_mismatches(a, spec.automaton)
    expect_error(raw_xargs(_payload(), max_bucket=32), R_BUCKET)
    expect_error(raw_xargs(_payload(), max_bucket=16), R_BUCKET)


def test_bucket_512_accepted_and_1024_refused_without_max_bucket():
    at = ws.with_bucket(512)
    above = ws.chain(at.n_states)  # `at.n_states` counts ACC, so this has one more real state
    assert above.n_states_bucket == 1024, "fixture precondition: one more state crosses to 1024"
    p_at, p_above = ws.encode(at), ws.encode(above)
    assert not ws.array_mismatches(at, parse(raw_xargs(p_at)).automaton)
    assert not ws.array_mismatches(at, parse(raw_xargs(p_at, max_bucket=512)).automaton)
    expect_error(raw_xargs(p_at, max_bucket=256), R_BUCKET)
    expect_error(raw_xargs(p_above), R_BUCKET)


# --------------------------------------------------------------------------- size boundary (criterion 5)


def test_payload_exactly_at_1_mib_is_not_refused_for_size():
    """The guard's too-wide direction: 1 MiB of base64 that decodes to nothing is refused for its
    content, not for its size."""
    p = "A" * MIB
    expect_error(raw_xargs(p), R_ENCODING)


@pytest.mark.parametrize("n", [MIB + 1, MIB + 4, 8 * MIB])
def test_payload_over_1_mib_is_refused_for_size(n):
    expect_error(raw_xargs("A" * n), R_SIZE)


# --------------------------------------------------------------------------- never raise anything else


def test_random_corruption_raises_only_cd_request_error():
    """300 seeded corruptions (n=300, seed 64): character flips of the base64 text and in-place integer
    perturbations of the arrays, each with a recomputed sha. The parser either refuses with a known
    reason or returns an automaton whose indices are all in range (independent range oracle); any other
    exception would escape into the engine thread (T-0067)."""
    err_type = ws.plugin("xargs").CdRequestError
    art = ws.artifact()
    rng = random.Random(64)
    base = _payload()
    meta0, arrays0 = art.unpack_payload(base)
    int_arrays = sorted(k for k, v in arrays0.items() if np.asarray(v).dtype.kind in "iu")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    other = []
    for i in range(300):
        if i % 2 == 0:
            pos = rng.randrange(len(base))
            bad = base[:pos] + rng.choice(alphabet) + base[pos + 1 :]
        else:
            arrays = {k: np.array(v, copy=True) for k, v in arrays0.items()}
            name = rng.choice(int_arrays)
            if arrays[name].size == 0:
                continue
            lo = 0 if arrays[name].dtype.kind == "u" else -3
            arrays[name].flat[rng.randrange(arrays[name].size)] = rng.randint(lo, VOCAB_BOUND)
            bad = art.pack_payload(dict(meta0), arrays)
        try:
            spec = parse(raw_xargs(bad))
        except err_type as err:
            if err.reason not in REASONS:
                other.append(f"#{i}: unknown reason {err.reason!r}")
            continue
        except Exception as exc:
            other.append(f"#{i}: {type(exc).__name__}: {exc}")
            continue
        au = spec.automaton
        s, c = au.n_states_bucket, au.sum_is_neg.shape[0]
        ok = (
            au.edge_src.min() >= 0
            and au.edge_dst.min() >= 0
            and au.edge_src.max() < s
            and au.edge_dst.max() < s
            and au.edge_class.min() >= 0
            and au.edge_class.max() < c
            and (au.sum_indices.size == 0 or au.sum_indices.max() < ws.VOCAB_SIZE)
            and au.start_vector.shape == (s,)
            and au.is_final.shape == (s,)
        )
        if not ok:
            other.append(f"#{i}: accepted an automaton with out-of-range indices")
    assert not other, "\n".join(other[:20])


VOCAB_BOUND = 300_000


# --------------------------------------------------------------------------- review 2026-09-29 (findings 5, 6; W5, W8)
#
# The duplicate-(src, dst) row lives on the NFA: on the Sudoku DFA the is_dfa check refused it for the wrong
# reason. Each hardening clause gets one crafted row whose mutation touches that clause only (the NFA and
# Sudoku repack controls above/below prove the unmutated repack parses).


def test_nfa_repack_control_parses():
    spec = parse(raw_xargs(_crafted(lambda m, a: None, base=_nfa_payload())))
    assert not ws.array_mismatches(ws.nfa_overlapping(), spec.automaton)


def test_duplicate_src_dst_pair_on_an_nfa():
    """W2: is_dfa is False here, so only the one-edge-per-pair invariant can refuse it."""
    expect_error(raw_xargs(_crafted(_dup_edge, base=_nfa_payload())), R_STRUCTURE)


def test_is_dfa_over_claim_is_refused():
    """W5: an NFA whose flag claims DFA would license the cheap exists-draw (vendored AUDIT-D4)."""
    a = ws.replace_meta(ws.nfa_overlapping(), is_dfa=True)
    expect_error(raw_xargs(ws.encode(a)), R_STRUCTURE)


def _real_state_with_finite_d(meta, a):
    n = meta["n_states"]
    finite = [s for s in range(n) if int(a["d"][s]) < (1 << 24)]
    assert finite, "fixture precondition: some real state reaches an accepting state"
    return finite[0]


def _d_off_by_one(meta, a):
    a["d"][_real_state_with_finite_d(meta, a)] += 1


def _d_padding_zero(meta, a):
    a["d"][meta["n_states"]] = 0


def _positive_class_pair(a, min_len: int) -> int:
    """A class stored positively in both tables with identical rows of at least ``min_len`` ids."""
    for c in range(a["sum_is_neg"].shape[0]):
        if a["sum_is_neg"][c] or a["max_is_neg"][c]:
            continue
        s = a["sum_indices"][a["sum_indptr"][c] : a["sum_indptr"][c + 1]]
        m = a["max_indices"][a["max_indptr"][c] : a["max_indptr"][c + 1]]
        if s.size >= min_len and np.array_equal(s, m) and int(s[-1]) + 1 < ws.VOCAB_SIZE:
            return c
    raise AssertionError("fixture precondition: no positively stored class to mutate")


def _max_table_other_set(meta, a):
    """Same size, same polarity, different member set in the MAX table only."""
    c = _positive_class_pair(a, 1)
    a["max_indices"][int(a["max_indptr"][c + 1]) - 1] += 1


def _row_unsorted_in_both_tables(meta, a):
    """Swap two ids of one class in BOTH tables: the sets still agree, only row order breaks."""
    c = _positive_class_pair(a, 2)
    for p in ("sum", "max"):
        lo = int(a[f"{p}_indptr"][c])
        a[f"{p}_indices"][[lo, lo + 1]] = a[f"{p}_indices"][[lo + 1, lo]]


def _padding_final(meta, a):
    a["is_final"][meta["n_states"]] = True


def _padding_start(meta, a):
    a["start_vector"][meta["n_states"]] = True


def _unused_class(meta, a):
    """W8: one more (empty, positive) class that no edge uses; sizes and both tables stay consistent."""
    for p in ("sum", "max"):
        a[f"{p}_is_neg"] = np.append(a[f"{p}_is_neg"], False)
        a[f"{p}_indptr"] = np.append(a[f"{p}_indptr"], a[f"{p}_indptr"][-1])
    a["class_size"] = np.append(a["class_size"], np.int32(0))
    if "n_classes" in meta:
        meta["n_classes"] += 1


_HARDENING = {
    "d-off-by-one-on-a-real-state": _d_off_by_one,
    "d-zero-on-a-padding-state": _d_padding_zero,
    "sum-and-max-tables-disagree": _max_table_other_set,
    "csr-row-not-strictly-increasing": _row_unsorted_in_both_tables,
    "padding-state-final": _padding_final,
    "padding-state-start": _padding_start,
    "class-no-edge-uses": _unused_class,
}


@pytest.mark.parametrize("clause", sorted(_HARDENING))
def test_hardening_clause(clause):
    expect_error(raw_xargs(_crafted(_HARDENING[clause])), R_STRUCTURE)


def test_unknown_extra_array_is_refused():
    def mutate(meta, a):
        a["extra"] = np.zeros(3, dtype=np.int32)

    x = raw_xargs(_crafted(mutate))
    err_type = ws.plugin("xargs").CdRequestError
    with pytest.raises(err_type) as info:
        parse(x)
    assert info.value.reason in (R_STRUCTURE, R_ENCODING), info.value.reason


def test_decoded_arrays_are_read_only_and_the_cache_stays_intact():
    """A decoded automaton is cached by sha and shared across requests: no caller may write into it."""
    p = _payload()
    au = parse(raw_xargs(p)).automaton
    writable = [n for n in ws.ARRAY_NAMES if getattr(au, n).flags.writeable]
    assert not writable, f"writable decoded arrays: {writable}"
    with pytest.raises(ValueError):
        au.d[0] = 7
    again = parse(raw_xargs(p, record_tag="second")).automaton
    assert not ws.array_mismatches(_sudoku(0), again)


# ---- container-level rows (format version 1: base64 of a ZIP of meta.json + <name>.npy, artifact.py)

HUGE = 560 << 20  # far above any bucket-512 automaton, yet it deflates to < 1 MiB of base64


def _entries(payload: str) -> list[tuple[str, bytes]]:
    import base64
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as zf:
        return [(i.filename, zf.read(i)) for i in zf.infolist()]


def _zip(entries, *, stream: dict | None = None) -> bytes:
    """A deflated ZIP of ``entries``; ``stream[name] = (head_bytes, n_zero_or_space_bytes, fill)``
    writes a large entry in chunks without holding it in memory."""
    import io
    import warnings
    import zipfile

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate names are deliberate in one row
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for name, data in entries:
                if stream and name in stream:
                    head, n, fill = stream[name]
                    big = zipfile.ZipInfo(name)
                    big.compress_type = zipfile.ZIP_DEFLATED
                    with zf.open(big, "w") as f:
                        f.write(head)
                        chunk = fill * (8 << 20)
                        left = n
                        while left:
                            k = min(left, len(chunk))
                            f.write(chunk[:k])
                            left -= k
                else:
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(info, data)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def _npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    import io

    f = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        f, {"descr": descr, "fortran_order": False, "shape": shape}
    )
    return f.getvalue()


def _timed_parse_error(payload: str, reason: str, bound_s: float = 0.5) -> float:
    import time

    assert len(payload) <= MIB, f"fixture precondition: crafted payload {len(payload)} B > 1 MiB"
    x = raw_xargs(payload)
    t0 = time.perf_counter()
    expect_error(x, reason)
    dt = time.perf_counter() - t0
    assert dt < bound_s, f"refusal took {dt:.3f} s (bound {bound_s} s): the entry was inflated"
    return dt


def test_zip_control_parses():
    """Control: re-zipping the entries by hand (this helper) yields a payload the plugin accepts."""
    spec = parse(raw_xargs(_b64(_zip(_entries(_payload())))))
    assert not ws.array_mismatches(_sudoku(0), spec.automaton)


def test_duplicate_archive_entries_are_refused():
    """d.npy twice, with different contents: which one a reader keeps is undefined, so refuse."""
    import io

    entries = _entries(_payload())
    other = np.lib.format.read_array(io.BytesIO(dict(entries)["d.npy"])).copy()
    other[0] += 1
    f = io.BytesIO()
    np.lib.format.write_array(f, other)
    expect_error(raw_xargs(_b64(_zip([*entries, ("d.npy", f.getvalue())]))), R_ENCODING)


def test_npy_header_declaring_a_huge_shape_is_refused_without_allocating():
    """W8: d.npy declares shape (10**12,) (4 TB of int32) over a few bytes of data. The decoder must
    refuse before allocating; a MemoryError escaping would kill the engine thread."""
    entries = []
    for name, data in _entries(_payload()):
        if name == "d.npy":
            body = data[len(data) - 4 * 64 :]  # the real 64 int32 values
            data = _npy_header("<i4", (10**12,)) + body
        entries.append((name, data))
    _timed_parse_error(_b64(_zip(entries)), R_ENCODING)


@functools.cache
def _honest_huge_payload() -> str:
    """d.npy honestly declares and holds HUGE bytes of zeros (sizes consistent)."""
    entries = _entries(_payload())
    head = _npy_header("<i4", (HUGE // 4,))
    return _b64(_zip(entries, stream={"d.npy": (head, HUGE, b"\0")}))


def test_archive_inflating_past_the_decompression_cap_is_refused_fast():
    """An honestly declared 560 MiB entry: refused as bad_encoding by the decompression cap. Without
    the cap it would inflate and then fail d's shape check as ``structure``."""
    _timed_parse_error(_honest_huge_payload(), R_ENCODING)


def _lying_meta_payload(*, crc_of_prefix: bool) -> str:
    """meta.json followed by HUGE bytes of JSON whitespace, with the declared uncompressed size patched
    down to the real meta's length (and, optionally, the CRC patched to the prefix's CRC, so a reader
    that stops at the declared size sees a perfectly consistent entry)."""
    import io
    import struct
    import zipfile
    import zlib

    entries = _entries(_payload())
    meta = dict(entries)["meta.json"]
    raw = bytearray(_zip(entries, stream={"meta.json": (meta, HUGE, b" ")}))
    with zipfile.ZipFile(io.BytesIO(bytes(raw))) as zf:
        info = zf.getinfo("meta.json")
        start_dir = zf.start_dir
    crc = zlib.crc32(meta) if crc_of_prefix else None
    off = info.header_offset
    assert raw[off : off + 4] == b"PK\x03\x04"
    struct.pack_into("<I", raw, off + 22, len(meta))
    if crc is not None:
        struct.pack_into("<I", raw, off + 14, crc)
    pos = start_dir
    patched = False
    while raw[pos : pos + 4] == b"PK\x01\x02":
        nlen, elen, clen = struct.unpack_from("<HHH", raw, pos + 28)
        if bytes(raw[pos + 46 : pos + 46 + nlen]) == b"meta.json":
            struct.pack_into("<I", raw, pos + 24, len(meta))
            if crc is not None:
                struct.pack_into("<I", raw, pos + 16, crc)
            patched = True
        pos += 46 + nlen + elen + clen
    assert patched, "fixture precondition: central directory entry for meta.json not found"
    return _b64(bytes(raw))


@pytest.mark.parametrize("crc_of_prefix", [False, True], ids=["crc-of-whole", "crc-of-prefix"])
def test_entry_whose_declared_size_lies_is_refused_fast(crc_of_prefix):
    """W8: the deflate stream holds meta.json plus 560 MiB of spaces but declares only meta.json's
    length. ``crc-of-whole``: the CRC betrays it. ``crc-of-prefix``: the CRC matches the declared
    prefix, so only a reader that notices unconsumed compressed data can tell."""
    _timed_parse_error(_lying_meta_payload(crc_of_prefix=crc_of_prefix), R_ENCODING)
