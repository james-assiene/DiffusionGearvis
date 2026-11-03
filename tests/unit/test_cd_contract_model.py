"""T-0064 criteria 1 and 6: the ``CdRequest`` and ``CdCanvasRecord`` contracts (fast suite, pydantic only).

These rows never import numpy-heavy or torch code: ``local_astra.contracts.cd`` is imported inside a fixture
so its absence reads "T-0064 not implemented", and the plugin mirror ``cd/record_schema.py`` is read by AST,
never imported. Key naming (``"astra_cd_" + field``, unset optionals omitted), the 1 MiB cap on the encoded
string and the sha convention are ruling W1 (``tests/cd/_wire_support.py``, restated here because this
module must not import from ``tests/cd``).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from local_astra.contracts.llm import VLLM_XARGS_KEY, LlmOptions, LlmRequestSpec

REPO = Path(__file__).resolve().parents[2]
RECORD_SCHEMA = (
    REPO / "plugin" / "astra_vllm_plugin" / "astra_vllm_plugin" / "cd" / "record_schema.py"
)
MIB = 1 << 20
PAYLOAD = "UEsDBBQAAAAIAAAAIQA="  # any ASCII string: the contract does not decode the automaton
RECORD_FIELDS = frozenset(
    {
        "record_tag",
        "arm",
        "canvas_idx",
        "committed_ids",
        "state_set_after",
        "in_final",
        "steps",
        "bucket",
        "stop_pos",
        "error",
        "mean_accepted_per_step",
        "shadow_argmax_ids",
        "argmax_in_language",
        "map_equals_argmax",
    }
)
END_PROJECTION_ONLY = ("shadow_argmax_ids", "argmax_in_language", "map_equals_argmax")


class _Lazy:
    """``local_astra.contracts.cd``, imported on first attribute access *inside the test body*, so its
    absence is a FAILED row reading "T-0064 not implemented" rather than a fixture ERROR."""

    def __getattr__(self, name: str) -> object:
        try:
            mod = importlib.import_module("local_astra.contracts.cd")
        except ModuleNotFoundError as exc:
            if exc.name == "local_astra.contracts.cd":
                pytest.fail(
                    f"T-0064 not implemented: local_astra.contracts.cd does not exist ({exc})"
                )
            raise
        return getattr(mod, name)


@pytest.fixture(scope="module")
def cd() -> ModuleType:
    return _Lazy()  # type: ignore[return-value]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def _req(cd: ModuleType, **kw: object) -> object:
    fields: dict[str, object] = {
        "mode": "perstep_mar",
        "self_conditioning": "model",
        "automaton": PAYLOAD,
        "automaton_sha256": _sha(PAYLOAD),
        "record_tag": "t0064.row",
    }
    fields.update(kw)
    return cd.CdRequest(**fields)


# --------------------------------------------------------------------------- criterion 1: CdRequest


@pytest.mark.parametrize(
    ("mode", "sc"),
    [
        ("end_projection", "model"),
        ("perstep_mf", "model"),
        ("perstep_mf", "constrained"),
        ("perstep_mar", "model"),
        ("perstep_mar", "constrained"),
        ("perstep_mar_qargmax", "model"),  # T-0076 criterion 1
        ("perstep_mar_qargmax", "constrained"),
    ],
)
def test_valid_mode_and_self_conditioning(cd, mode, sc):
    r = _req(cd, mode=mode, self_conditioning=sc)
    assert (r.mode, r.self_conditioning) == (mode, sc)


def test_constrained_self_conditioning_needs_a_perstep_mode(cd):
    with pytest.raises(ValidationError):
        _req(cd, mode="end_projection", self_conditioning="constrained")


@pytest.mark.parametrize(
    "kw",
    [
        {"mode": "perstep"},
        {"mode": "unconstrained"},
        {"mode": "perstep_mar_sc"},
        {"mode": "perstep_mar_qargmax_sc"},  # T-0076: an eval arm name, not a wire mode
        {"mode": "perstep_qargmax"},
        {"self_conditioning": "none"},
        {"entropy_bound": 0.0},
        {"entropy_bound": -1e-9},
        {"entropy_bound": math.nan},
        {"confidence_threshold": 0.0},
        {"confidence_threshold": -0.5},
        {"confidence_threshold": math.nan},
        {"max_bucket": 8},
        {"max_bucket": 48},
        {"max_bucket": 100},
        {"max_bucket": 1024},
        {"max_bucket": 0},
        {"record_tag": ""},
        {"record_tag": "a" * 65},
        {"record_tag": "a b"},
        {"record_tag": "tag/1"},
        {"record_tag": "tag\n"},
        {"record_tag": "taég"},
        {"record_tag": 'x"}'},
    ],
    ids=lambda kw: json.dumps(kw),
)
def test_invalid_fields_are_refused(cd, kw):
    with pytest.raises(ValidationError):
        _req(cd, **kw)


@pytest.mark.parametrize("bucket", [16, 32, 64, 128, 256, 512])
def test_every_ladder_bucket_is_a_valid_max_bucket(cd, bucket):
    assert _req(cd, max_bucket=bucket).max_bucket == bucket


@pytest.mark.parametrize("tag", ["a", "Z" * 64, "A-z_0.9:x-", "bfcl:live_simple_3-2-0.A"])
def test_record_tag_edge_values(cd, tag):
    assert _req(cd, record_tag=tag).record_tag == tag


@pytest.mark.parametrize("missing", ["mode", "automaton", "automaton_sha256", "record_tag"])
def test_required_fields(cd, missing):
    fields = {
        "mode": "perstep_mar",
        "self_conditioning": "model",
        "automaton": PAYLOAD,
        "automaton_sha256": _sha(PAYLOAD),
        "record_tag": "t0064.row",
    }
    del fields[missing]
    with pytest.raises(ValidationError):
        cd.CdRequest(**fields)


def test_positive_bounds_accepted(cd):
    r = _req(cd, entropy_bound=1e-9, confidence_threshold=0.5)
    assert (r.entropy_bound, r.confidence_threshold) == (1e-9, 0.5)


def test_payload_exactly_1_mib_accepted_and_one_more_refused(cd):
    at, over = "A" * MIB, "A" * (MIB + 1)
    assert len(_req(cd, automaton=at, automaton_sha256=_sha(at)).automaton) == MIB
    with pytest.raises(ValidationError):
        _req(cd, automaton=over, automaton_sha256=_sha(over))


def test_to_xargs_full(cd):
    """Hand-computed expectation: ``"astra_cd_" + field`` for every field that is set."""
    r = _req(
        cd,
        mode="perstep_mf",
        self_conditioning="constrained",
        record_tag="bfcl:7",
        entropy_bound=0.5,
        confidence_threshold=0.9,
        max_bucket=128,
    )
    assert r.to_xargs() == {
        "astra_cd_mode": "perstep_mf",
        "astra_cd_self_conditioning": "constrained",
        "astra_cd_automaton": PAYLOAD,
        "astra_cd_automaton_sha256": _sha(PAYLOAD),
        "astra_cd_record_tag": "bfcl:7",
        "astra_cd_entropy_bound": 0.5,
        "astra_cd_confidence_threshold": 0.9,
        "astra_cd_max_bucket": 128,
    }


def test_to_xargs_qargmax_with_overrides(cd):
    """T-0076 criterion 1: the new mode, constrained SC and the P13 overrides, hand-computed."""
    r = _req(
        cd,
        mode="perstep_mar_qargmax",
        self_conditioning="constrained",
        record_tag="sudoku:3",
        entropy_bound=0.2,
        confidence_threshold=0.4,
    )
    assert r.to_xargs() == {
        "astra_cd_mode": "perstep_mar_qargmax",
        "astra_cd_self_conditioning": "constrained",
        "astra_cd_automaton": PAYLOAD,
        "astra_cd_automaton_sha256": _sha(PAYLOAD),
        "astra_cd_record_tag": "sudoku:3",
        "astra_cd_entropy_bound": 0.2,
        "astra_cd_confidence_threshold": 0.4,
    }


def test_to_xargs_omits_unset_optionals(cd):
    x = _req(cd, mode="end_projection").to_xargs()
    assert x == {
        "astra_cd_mode": "end_projection",
        "astra_cd_self_conditioning": "model",
        "astra_cd_automaton": PAYLOAD,
        "astra_cd_automaton_sha256": _sha(PAYLOAD),
        "astra_cd_record_tag": "t0064.row",
    }


@pytest.mark.parametrize("full", [False, True])
def test_to_xargs_passes_the_existing_vllm_xargs_validator(cd, full):
    kw = {"entropy_bound": 0.5, "confidence_threshold": 0.9, "max_bucket": 64} if full else {}
    x = _req(cd, **kw).to_xargs()
    assert all(k.startswith("astra_cd_") for k in x)
    assert all(type(v) in (str, int, float) for v in x.values()), {k: type(v) for k, v in x.items()}
    assert json.loads(json.dumps(x)) == x
    body = LlmOptions.extra_body({**x, "astra_tap": True})
    assert body == {VLLM_XARGS_KEY: {**x, "astra_tap": True}}
    spec = LlmRequestSpec.model_validate(
        {"request_id": "r", "prompt": {"messages": [], "prefix_len": 0}, "xargs": x}
    )
    assert spec.xargs == x


def test_contracts_cd_is_light(cd):
    """Importing the contract never pulls the compiler, torch or the plugin (fast-suite isolation)."""
    # an absent module fails as "not implemented" here, not as a subprocess error below
    assert cd.CdRequest is not None
    code = (
        "import sys, json; import local_astra.contracts.cd; "
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch','outlines_core','tokenizers','astra_vllm_plugin') "
        "or m.startswith('local_astra.decoding'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


# --------------------------------------------------------------------------- criterion 6: CdCanvasRecord


def test_record_field_set_is_exactly_the_criterion(cd):
    assert set(cd.CdCanvasRecord.model_fields) == RECORD_FIELDS


def _mirror_fields() -> set[str]:
    if not RECORD_SCHEMA.is_file():
        pytest.fail(f"T-0064 not implemented: {RECORD_SCHEMA.relative_to(REPO)} is missing")
    tree = ast.parse(RECORD_SCHEMA.read_text())
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "RECORD_FIELDS" for t in targets):
            value = node.value
            if (
                isinstance(value, ast.Call) and len(value.args) == 1
            ):  # frozenset({...}) / tuple([...])
                value = value.args[0]
            names = ast.literal_eval(value)
            assert all(isinstance(n, str) for n in names), names
            assert len(set(names)) == len(names), f"duplicate names in RECORD_FIELDS: {names}"
            return set(names)
    pytest.fail("cd/record_schema.py defines no module-level RECORD_FIELDS literal")


def test_plugin_mirror_equals_the_model(cd):
    mirror = _mirror_fields()
    model = set(cd.CdCanvasRecord.model_fields)
    assert mirror == model, f"only in mirror {mirror - model}; only in model {model - mirror}"


def _record(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "record_tag": "bfcl:live_simple_0-0-0.A",
        "arm": "perstep_mar",
        "canvas_idx": 0,
        "committed_ids": [100, 45518, 107, 101, 123, 1],
        "state_set_after": [31],
        "in_final": True,
        "steps": 17,
        "bucket": 64,
        "stop_pos": 5,
        "error": None,
        "mean_accepted_per_step": 3.25,
        "shadow_argmax_ids": None,
        "argmax_in_language": None,
        "map_equals_argmax": None,
    }
    base.update(kw)
    return base


def test_a_perstep_record_as_emitted_json_validates(cd):
    rec = cd.CdCanvasRecord.model_validate_json(json.dumps(_record()))
    assert rec.committed_ids[-1] == 1 and rec.in_final is True and rec.error is None


def test_an_end_projection_record_carries_the_shadow_fields(cd):
    raw = _record(
        arm="end_projection",
        mean_accepted_per_step=None,
        shadow_argmax_ids=[100, 45518, 107, 101, 124, 1],
        argmax_in_language=False,
        map_equals_argmax=False,
    )
    rec = cd.CdCanvasRecord.model_validate_json(json.dumps(raw))
    assert list(rec.shadow_argmax_ids) == [100, 45518, 107, 101, 124, 1]
    assert rec.argmax_in_language is False and rec.map_equals_argmax is False


def test_a_qargmax_record_validates(cd):
    """T-0076 criterion 7: records name ``arm = "perstep_mar_qargmax"``, with ``mean_accepted_per_step`` and
    ``steps`` filled and the end_projection fields null; the schema is unchanged (the field set test above)."""
    rec = cd.CdCanvasRecord.model_validate_json(json.dumps(_record(arm="perstep_mar_qargmax")))
    assert rec.arm == "perstep_mar_qargmax" and rec.mean_accepted_per_step == 3.25
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(arm="perstep_mar_qargmax_sc"))


@pytest.mark.parametrize("arm", ["perstep_mf", "perstep_mar", "perstep_mar_qargmax"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("shadow_argmax_ids", [1]), ("argmax_in_language", True), ("map_equals_argmax", False)],
)
def test_end_projection_fields_are_null_for_other_arms(cd, arm, field, value):
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(arm=arm, **{field: value}))


def test_an_error_record_validates(cd):
    raw = _record(
        committed_ids=[1],
        state_set_after=[],
        in_final=False,
        steps=0,
        stop_pos=0,
        error="sha_mismatch",
        mean_accepted_per_step=None,
    )
    assert cd.CdCanvasRecord.model_validate(raw).error == "sha_mismatch"


@pytest.mark.parametrize(
    "kw",
    [
        {"arm": "not_an_arm"},
        {"canvas_idx": -1},
        {"committed_ids": [5, -1]},
        {"committed_ids": [262_144]},
        {"record_tag": "a b"},
        {"record_tag": ""},
        {"stop_position": 3},
    ],
    ids=lambda kw: json.dumps(kw),
)
def test_invalid_records_are_refused(cd, kw):
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(**kw))


@pytest.mark.parametrize("missing", ["record_tag", "arm", "canvas_idx"])
def test_record_required_fields(cd, missing):
    raw = _record()
    del raw[missing]
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(raw)


# --------------------------------------------------------------------------- review 2026-09-29 (W5, W8, findings 6-7)


def test_self_conditioning_defaults_to_model(cd):
    """W5."""
    r = cd.CdRequest(
        mode="perstep_mar", automaton=PAYLOAD, automaton_sha256=_sha(PAYLOAD), record_tag="t"
    )
    assert r.self_conditioning == "model"
    assert r.to_xargs()["astra_cd_self_conditioning"] == "model"


@pytest.mark.parametrize(
    "digest",
    [
        _sha("a different automaton"),  # well-formed, but not this payload's
        _sha(PAYLOAD).upper(),
        _sha(PAYLOAD)[:-1],
        _sha(PAYLOAD) + "0",
    ],
    ids=["other-payload", "uppercase", "63-chars", "65-chars"],
)
def test_sha_must_be_the_payload_sha_in_lowercase_hex(cd, digest):
    """W1/W5: the contract refuses a request whose sha is not the payload's content address."""
    with pytest.raises(ValidationError):
        _req(cd, automaton_sha256=digest)


def test_mean_accepted_per_step_is_null_for_end_projection(cd):
    """W5."""
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(arm="end_projection", mean_accepted_per_step=1.0))


def test_stop_pos_indexes_into_committed_ids(cd):
    ids = [100, 45518, 107, 101, 123, 1]
    assert cd.CdCanvasRecord.model_validate(_record(committed_ids=ids, stop_pos=5)).stop_pos == 5
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(committed_ids=ids, stop_pos=6))


def test_state_set_after_lies_inside_the_bucket(cd):
    assert cd.CdCanvasRecord.model_validate(_record(bucket=64, state_set_after=[0, 63]))
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(bucket=64, state_set_after=[64]))


def _error_record(**kw: object) -> dict[str, object]:
    base = _record(
        committed_ids=[1],
        state_set_after=[],
        in_final=False,
        steps=0,
        stop_pos=0,
        error="missing_mode",
        mean_accepted_per_step=None,
    )
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "nulls",
    [{"arm": None}, {"record_tag": None}, {"arm": None, "record_tag": None}],
    ids=["arm", "record_tag", "both"],
)
def test_w8_error_record_may_have_null_arm_and_tag(cd, nulls):
    """W8: a request refused as missing_mode / unknown_key / bad tag has no valid arm or tag, and its
    error record must still be representable."""
    rec = cd.CdCanvasRecord.model_validate_json(json.dumps(_error_record(**nulls)))
    assert rec.error == "missing_mode"
    for k in nulls:
        assert getattr(rec, k) is None


@pytest.mark.parametrize(
    "nulls",
    [{"arm": None}, {"record_tag": None}, {"arm": None, "record_tag": None}],
    ids=["arm", "record_tag", "both"],
)
def test_w8_null_arm_or_tag_without_an_error_is_refused(cd, nulls):
    with pytest.raises(ValidationError):
        cd.CdCanvasRecord.model_validate(_record(**nulls))


def _module_str_constant(path: Path, name: str) -> str:
    if not path.is_file():
        pytest.fail(f"T-0064 not implemented: {path.relative_to(REPO)} is missing")
    for node in ast.parse(path.read_text()).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            pytest.fail(f"{path.relative_to(REPO)}: {name} is not a string literal")
    pytest.fail(f"{path.relative_to(REPO)} defines no module-level {name}")


def test_plugin_tokenizer_pin_equals_the_host_pin():
    """Finding 7: the container cannot import local_astra, so the plugin restates the tokenizer.json
    sha256; the two literals must agree (read by AST, nothing imported)."""
    plugin_pin = _module_str_constant(RECORD_SCHEMA.parent / "automaton.py", "TOKENIZER_SHA256")
    host_pin = _module_str_constant(
        REPO / "local_astra" / "decoding" / "vocab.py", "TOKENIZER_SHA256"
    )
    assert plugin_pin == host_pin
    assert re.fullmatch(r"[0-9a-f]{64}", host_pin)


# --------------------------------------------------------------------------- T-0081 criterion 1: the draft fields
#
# Ruling SD1 (tests/cd/unit/_seed_support.py, restated here because this module must not import from tests/cd):
# ``draft`` = standard base64 of little-endian int32 ids (non-empty), ``draft_noise`` in [0, 1), ``draft_start_step``
# >= 0; noise or start step given without a draft is a ValidationError; ``to_xargs`` emits no ``astra_cd_draft*`` key
# without a draft. Every refusal row also builds its valid control, so a contract that refuses the fields outright
# (the pre-T-0081 ``extra="forbid"``) fails the row instead of passing it.


def _b64(ids: list[int]) -> str:
    import base64
    import struct

    return base64.b64encode(struct.pack(f"<{len(ids)}i", *ids)).decode("ascii")


DRAFT = _b64([1000, 1003, 262_143, 0])


def _ok(cd: ModuleType, **kw: object) -> object:
    try:
        return _req(cd, **kw)
    except ValidationError as exc:
        raise AssertionError(
            f"T-0081: a valid seeded CdRequest was refused: {exc.errors()[:2]}"
        ) from None


@pytest.mark.parametrize(
    "kw",
    [
        {"draft": DRAFT},
        {"draft": DRAFT, "draft_noise": 0.0},
        {"draft": DRAFT, "draft_noise": 0.999},
        {"draft": DRAFT, "draft_start_step": 0},
        {"draft": DRAFT, "draft_noise": 0.5, "draft_start_step": 47},
        {"draft": _b64([5])},
    ],
    ids=lambda kw: json.dumps(kw)[:60],
)
def test_t0081_valid_draft_fields(cd, kw):
    r = _ok(cd, **kw)
    for k, v in kw.items():
        assert getattr(r, k) == v, (k, getattr(r, k), v)


@pytest.mark.parametrize(
    "kw",
    [
        {"draft": DRAFT, "draft_noise": -0.1},
        {"draft": DRAFT, "draft_noise": 1.0},
        {"draft": DRAFT, "draft_noise": math.nan},
        {"draft": DRAFT, "draft_noise": math.inf},
        {"draft": DRAFT, "draft_start_step": -1},
        {"draft": DRAFT, "draft_start_step": 1.5},
        {"draft": ""},
        {"draft": "!!!!"},
        {"draft": DRAFT[:-1]},
        {"draft_noise": 0.5},
        {"draft_noise": 0.0},
        {"draft_start_step": 3},
        {"draft_start_step": 0},
    ],
    ids=lambda kw: json.dumps(kw)[:60],
)
def test_t0081_invalid_draft_fields_are_refused(cd, kw):
    _ok(cd, draft=DRAFT, draft_noise=0.25, draft_start_step=1)  # control: the fields exist
    with pytest.raises(ValidationError):
        _req(cd, **kw)


def test_t0081_to_xargs_with_every_draft_field(cd):
    """Hand-computed: the three keys beside the base ones, values verbatim."""
    r = _ok(cd, mode="end_projection", draft=DRAFT, draft_noise=0.25, draft_start_step=24)
    assert r.to_xargs() == {
        "astra_cd_mode": "end_projection",
        "astra_cd_self_conditioning": "model",
        "astra_cd_automaton": PAYLOAD,
        "astra_cd_automaton_sha256": _sha(PAYLOAD),
        "astra_cd_record_tag": "t0064.row",
        "astra_cd_draft": DRAFT,
        "astra_cd_draft_noise": 0.25,
        "astra_cd_draft_start_step": 24,
    }


def test_t0081_to_xargs_with_draft_only_and_without_draft(cd):
    x = _ok(cd, draft=DRAFT).to_xargs()
    assert x["astra_cd_draft"] == DRAFT
    assert all(v is not None for v in x.values()), x
    assert set(x) - {"astra_cd_draft_noise", "astra_cd_draft_start_step"} == {
        "astra_cd_mode",
        "astra_cd_self_conditioning",
        "astra_cd_automaton",
        "astra_cd_automaton_sha256",
        "astra_cd_record_tag",
        "astra_cd_draft",
    }
    assert x.get("astra_cd_draft_noise", 0.0) == 0.0 and x.get("astra_cd_draft_start_step", 0) == 0
    assert not [k for k in _req(cd).to_xargs() if k.startswith("astra_cd_draft")]


def test_t0081_seeded_xargs_pass_the_vllm_xargs_validator(cd):
    x = _ok(cd, draft=_b64(list(range(256))), draft_noise=0.5, draft_start_step=3).to_xargs()
    assert all(type(v) in (str, int, float) for v in x.values())
    assert json.loads(json.dumps(x)) == x
    assert LlmOptions.extra_body(x) == {VLLM_XARGS_KEY: x}
