"""T-0081 criterion 1: the draft keys on the wire -- parse, refuse with stable reasons, round trip, old cases unchanged.

API and reason codes: rulings SD1-SD3 in ``tests/cd/unit/_seed_support.py``. Expected ids are hand-written; the
encoding helper ``b64`` is the criterion's own definition (standard base64 of little-endian int32), and every
parsed value is compared with the ids that went in, never with another parse.

Refusals are checked twice: at the parser (``CdRequestError.reason``) for the context-free ones, and end to end
through the T-0067 CPU stand-in for all of them (EOS-first commit within two decode calls, exactly one error record
carrying the reason, the first canvas never the draft) -- including the two only admission can see (more than CL
ids, ``draft_start_step >= max_denoising_steps``).

The fake vLLM server rows: ``FakeVllmServer.cd_seeds()`` (test-author extension) reports each constrained request's
decoded seed, and a ~1.4 KB draft (256 ids) built by ``CdRequest.to_xargs()`` is accepted (``backend=real`` is the
tier-2 check that the v0.29.0 API server passes it; ``gpu``).

Regime: CPU torch, vendored v0.29.0 stock step, V = 262 144, CL 8, ``max_denoising_steps`` 6, ST 2.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, World
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit import _seed_support as sd
from tests.cd.unit._sampler_support import LIVE, Records

CL = 8
MAX_STEPS = 6
VOCAB = 262_144


def _cfg() -> Cfg:
    return Cfg(CL=CL, max_denoising_steps=MAX_STEPS, stability_threshold=1)


def _base(tag: str = "t0081-wire", mode: str = "end_projection") -> dict[str, Any]:
    g = ps.loop_grammar(LIVE, "t0081-wire-loop")
    sc = "model"
    return ws.raw_xargs(g.payload, mode=mode, self_conditioning=sc, record_tag=tag)


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return _base()


# --------------------------------------------------------------------------- parse: accepted values


def test_no_draft_key_gives_no_seed(base: dict[str, Any]) -> None:
    ids, noise, start = sd.spec_seed(sd.parse_ok(base))
    assert (ids, noise, start) == (None, 0.0, 0)


def test_draft_alone_defaults_noise_0_and_start_0(base: dict[str, Any]) -> None:
    ids, noise, start = sd.spec_seed(sd.parse_ok(sd.with_seed(base, [1000, 1003, 262_143, 0])))
    assert ids == [1000, 1003, 262_143, 0]
    assert (noise, start) == (0.0, 0)
    assert type(noise) is float and type(start) is int


@pytest.mark.parametrize(("noise", "start"), [(0.0, 0), (0.5, 3), (0.999, 5), (0.25, 24)])
def test_draft_noise_and_start_step_are_parsed(
    base: dict[str, Any], noise: float, start: int
) -> None:
    """``draft_start_step`` is bounded by ``max_denoising_steps`` only at admission (SD3): 24 parses."""
    ids, got_noise, got_start = sd.spec_seed(
        sd.parse_ok(sd.with_seed(base, sd.DRAFT8, noise=noise, start=start))
    )
    assert ids == list(sd.DRAFT8)
    assert (got_noise, got_start) == (noise, start)


def test_hand_encoded_draft_decodes_to_its_ids(base: dict[str, Any]) -> None:
    """Independent of ``b64``: id 1000 = 0x000003E8 is bytes e8 03 00 00 (little endian); id 262 143 = 0x0003FFFF."""
    raw = bytes([0xE8, 0x03, 0x00, 0x00, 0xFF, 0xFF, 0x03, 0x00])
    payload = base64.b64encode(raw).decode("ascii")
    assert payload == "6AMAAP//AwA=", payload
    ids, _, _ = sd.spec_seed(sd.parse_ok(sd.with_seed(base, raw_draft=payload)))
    assert ids == [1000, 262_143]


def test_a_256_id_draft_parses(base: dict[str, Any]) -> None:
    """CL 256 is the production canvas: 1 368 base64 characters, the ~1.4 KB of the task's check."""
    ids = [(i * 7919) % VOCAB for i in range(256)]
    x = sd.with_seed(base, ids)
    assert len(x[sd.DRAFT_KEYS["draft"]]) == 1368
    got, _, _ = sd.spec_seed(sd.parse_ok(x))
    assert got == ids


# --------------------------------------------------------------------------- parse: refusals (context-free)


def _urlsafe_only_payload() -> tuple[str, str]:
    """A draft whose standard encoding contains '+' or '/', and its URL-safe spelling ('-' / '_')."""
    for t in range(1000, VOCAB):
        s = sd.b64([t, t + 1])
        if "+" in s or "/" in s:
            return s, s.replace("+", "-").replace("/", "_")
    raise AssertionError("fixture: no id below V encodes with '+' or '/'")


_STD, _URLSAFE = _urlsafe_only_payload()

ENCODING_CASES = {
    "not_base64": "!!!!",
    "bad_padding": sd.b64([1000])[:-1],
    "urlsafe_alphabet": _URLSAFE,
    "embedded_newline": sd.b64([1000, 1001])[:4] + "\n" + sd.b64([1000, 1001])[4:],
    "six_bytes": base64.b64encode(b"\x01\x00\x00\x00\x02\x00").decode("ascii"),
    "one_byte": base64.b64encode(b"\x01").decode("ascii"),
}


@pytest.mark.parametrize("case", sorted(ENCODING_CASES))
def test_non_base64_or_ragged_draft_is_refused(base: dict[str, Any], case: str) -> None:
    ws.expect_error(sd.with_seed(base, raw_draft=ENCODING_CASES[case]), sd.R_DRAFT_ENCODING)


def test_the_standard_spelling_of_the_urlsafe_case_parses(base: dict[str, Any]) -> None:
    """Control for ``urlsafe_alphabet``: the same ids in the standard alphabet are accepted."""
    ids, _, _ = sd.spec_seed(sd.parse_ok(sd.with_seed(base, raw_draft=_STD)))
    assert ids is not None and len(ids) == 2


def test_empty_draft_is_refused_as_length(base: dict[str, Any]) -> None:
    ws.expect_error(sd.with_seed(base, raw_draft=""), sd.R_DRAFT_LENGTH)


@pytest.mark.parametrize(
    "bad", [[VOCAB], [1000, VOCAB + 5], [(1 << 31) - 1], [-1], [1000, -262_144]]
)
def test_out_of_vocab_ids_are_refused(base: dict[str, Any], bad: list[int]) -> None:
    ws.expect_error(sd.with_seed(base, bad), sd.R_DRAFT_VOCAB)


@pytest.mark.parametrize("noise", [-0.1, -0.0001, 1.0, 1.5, math.nan, math.inf, -math.inf])
def test_noise_outside_0_1_is_refused(base: dict[str, Any], noise: float) -> None:
    ws.expect_error(sd.with_seed(base, sd.DRAFT8, noise=noise), sd.R_DRAFT_NOISE)


@pytest.mark.parametrize("start", [-1, -100])
def test_negative_start_step_is_refused(base: dict[str, Any], start: int) -> None:
    ws.expect_error(sd.with_seed(base, sd.DRAFT8, start=start), sd.R_DRAFT_START)


@pytest.mark.parametrize(
    ("key", "value", "allowed"),
    [
        ("draft", [1000, 1001], (sd.R_DRAFT_ENCODING, ws.R_VALUE)),
        ("draft", 1000, (sd.R_DRAFT_ENCODING, ws.R_VALUE)),
        ("draft_noise", "0.5", (sd.R_DRAFT_NOISE, ws.R_VALUE)),
        ("draft_noise", None, (sd.R_DRAFT_NOISE, ws.R_VALUE)),
        ("draft_start_step", 1.5, (sd.R_DRAFT_START, ws.R_VALUE)),
        ("draft_start_step", "2", (sd.R_DRAFT_START, ws.R_VALUE)),
        ("draft_start_step", True, (sd.R_DRAFT_START, ws.R_VALUE)),
    ],
    ids=lambda v: json.dumps(v) if not isinstance(v, tuple) else "allowed",
)
def test_wrong_json_types_are_refused(
    base: dict[str, Any], key: str, value: Any, allowed: tuple[str, ...]
) -> None:
    x = sd.with_seed(base, sd.DRAFT8)
    x[sd.DRAFT_KEYS[key]] = value
    err_type = ws.plugin("xargs").CdRequestError
    with pytest.raises(err_type) as info:
        ws.parse(x)
    assert info.value.reason in allowed, (info.value.reason, allowed)


@pytest.mark.parametrize(
    "extra",
    [
        {"astra_cd_draft_noise": 0.5},
        {"astra_cd_draft_noise": 0.0},
        {"astra_cd_draft_start_step": 3},
        {"astra_cd_draft_start_step": 0},
        {"astra_cd_draft_noise": 0.5, "astra_cd_draft_start_step": 3},
    ],
    ids=lambda d: json.dumps(d),
)
def test_noise_or_start_without_draft_is_a_missing_key(
    base: dict[str, Any], extra: dict[str, Any]
) -> None:
    ws.expect_error({**base, **extra}, ws.R_MISSING_KEY)


def test_the_new_reasons_are_stable_members_of_reasons() -> None:
    reasons = ws.plugin("xargs").REASONS
    assert set(reasons) >= sd.SEED_REASONS, sorted(sd.SEED_REASONS - set(reasons))
    for r in sd.SEED_REASONS:
        assert r == r.lower() and len(r) <= 64 and r.replace("_", "").isalnum(), r


def test_a_draft_key_without_mode_is_missing_mode_never_unconstrained() -> None:
    """Seeding an unconstrained request is out of scope: the keys alone are a refusal (W1), not ``None``."""
    ws.expect_error({sd.DRAFT_KEYS["draft"]: sd.b64(sd.DRAFT8)}, ws.R_NO_MODE)


def test_a_misspelt_draft_key_is_unknown_never_ignored(base: dict[str, Any]) -> None:
    assert sd.spec_seed(sd.parse_ok(sd.with_seed(base, sd.DRAFT8, noise=0.5)))[1] == 0.5  # control
    ws.expect_error(
        {**sd.with_seed(base, sd.DRAFT8), "astra_cd_draft_nosie": 0.5}, ws.R_UNKNOWN_KEY
    )


# --------------------------------------------------------------------------- W1 check order


def test_order_unknown_key_before_a_bad_draft(base: dict[str, Any]) -> None:
    ws.expect_error(
        sd.with_seed(base, raw_draft="!!!!"), sd.R_DRAFT_ENCODING
    )  # control: the draft alone
    ws.expect_error({**sd.with_seed(base, raw_draft="!!!!"), "astra_cd_bogus": 1}, ws.R_UNKNOWN_KEY)


def test_order_missing_mode_before_a_bad_draft(base: dict[str, Any]) -> None:
    x = sd.with_seed(base, raw_draft="!!!!")
    del x[ws.KEYS["mode"]]
    ws.expect_error(x, ws.R_NO_MODE)


def test_order_missing_required_key_before_a_bad_draft(base: dict[str, Any]) -> None:
    x = sd.with_seed(base, [VOCAB])
    del x[ws.KEYS["record_tag"]]
    ws.expect_error(x, ws.R_MISSING_KEY)


@pytest.mark.parametrize(
    ("seed_kw", "reason"),
    [
        ({"raw_draft": "!!!!"}, sd.R_DRAFT_ENCODING),
        ({"ids": [VOCAB]}, sd.R_DRAFT_VOCAB),
        ({"ids": list(sd.DRAFT8), "noise": 1.0}, sd.R_DRAFT_NOISE),
    ],
)
def test_order_a_bad_draft_before_the_sha_and_the_payload_cap(
    base: dict[str, Any], seed_kw: dict[str, Any], reason: str
) -> None:
    """Draft values are W1 "bad values": they are reported before the 1 MiB cap and the sha (no decode is spent on
    a request that will be refused anyway)."""
    bad_sha = {**sd.with_seed(base, **seed_kw), ws.KEYS["automaton_sha256"]: "0" * 64}
    ws.expect_error(bad_sha, reason)
    over = "A" * (ws.MIB + 1)
    too_big = sd.with_seed(
        {**base, ws.KEYS["automaton"]: over, ws.KEYS["automaton_sha256"]: ws.sha(over)}, **seed_kw
    )
    ws.expect_error(too_big, reason)


def test_a_valid_draft_does_not_mask_the_sha_check(base: dict[str, Any]) -> None:
    ws.expect_error(
        {**sd.with_seed(base, sd.DRAFT8, noise=0.5), ws.KEYS["automaton_sha256"]: "0" * 64},
        ws.R_SHA,
    )


# --------------------------------------------------------------------------- round trip and old cases


@pytest.mark.contract
@pytest.mark.parametrize(
    "mode", ["end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax"]
)
def test_pydantic_to_plugin_round_trip(mode: str) -> None:
    g = ps.loop_grammar(LIVE, "t0081-rt")
    ids = [1000, 1, 262_143, 0, 50, 106, 1007]
    x = sd.request_xargs(
        g.payload,
        mode=mode,
        record_tag="t0081.rt",
        draft=sd.b64(ids),
        draft_noise=0.25,
        draft_start_step=4,
    )
    assert x[sd.DRAFT_KEYS["draft"]] == sd.b64(ids)
    assert x[sd.DRAFT_KEYS["draft_noise"]] == 0.25 and x[sd.DRAFT_KEYS["draft_start_step"]] == 4
    spec = sd.parse_ok(x)
    assert spec.mode == mode
    assert sd.spec_seed(spec) == (ids, 0.25, 4)


@pytest.mark.contract
def test_round_trip_draft_only_reads_the_defaults() -> None:
    g = ps.loop_grammar(LIVE, "t0081-rt")
    x = sd.request_xargs(
        g.payload, mode="perstep_mar", record_tag="t0081.rt0", draft=sd.b64(sd.DRAFT5)
    )
    assert x[sd.DRAFT_KEYS["draft"]] == sd.b64(sd.DRAFT5)
    assert all(v is not None for v in x.values()), x  # SD1: unset optionals are omitted, never null
    assert sd.spec_seed(sd.parse_ok(x)) == (list(sd.DRAFT5), 0.0, 0)


@pytest.mark.contract
@pytest.mark.parametrize(
    "mode", ["end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax"]
)
def test_pre_existing_request_has_no_draft_key_and_no_seed(mode: str) -> None:
    """Every pre-existing wire case (no draft field) emits no ``astra_cd_draft*`` key and decodes to a spec with no
    seed; its other fields are unchanged."""
    g = ps.loop_grammar(LIVE, "t0081-rt")
    x = (
        ws.request_xargs(
            g.payload,
            mode=mode,
            record_tag="t0081.old",
            entropy_bound=0.5,
            confidence_threshold=0.25,
        )
        if mode != "end_projection"
        else ws.request_xargs(g.payload, mode=mode, record_tag="t0081.old")
    )
    assert not [k for k in x if k.startswith("astra_cd_draft")], x
    spec = ws.parse(x)
    assert spec.mode == mode and spec.record_tag == "t0081.old"
    assert sd.spec_seed(spec) == (None, 0.0, 0)


# --------------------------------------------------------------------------- end to end: every refusal is loud


def _refusal_cases() -> dict[str, tuple[dict[str, Any], str]]:
    return {
        "encoding": ({"raw_draft": "!!!!"}, sd.R_DRAFT_ENCODING),
        "empty": ({"raw_draft": ""}, sd.R_DRAFT_LENGTH),
        "longer_than_cl": ({"ids": [*sd.DRAFT8, 1000]}, sd.R_DRAFT_LENGTH),
        "out_of_vocab": ({"ids": [1000, VOCAB]}, sd.R_DRAFT_VOCAB),
        "noise_1": ({"ids": list(sd.DRAFT8), "noise": 1.0}, sd.R_DRAFT_NOISE),
        "start_at_max": ({"ids": list(sd.DRAFT8), "start": MAX_STEPS}, sd.R_DRAFT_START),
        "start_past_max": ({"ids": list(sd.DRAFT8), "start": MAX_STEPS + 40}, sd.R_DRAFT_START),
        "noise_without_draft": ({"noise": 0.5}, ws.R_MISSING_KEY),
    }


REFUSALS = _refusal_cases()


@pytest.mark.parametrize("mode", ["end_projection", "perstep_mar"])
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_refused_seed_ends_loudly_with_an_eos_canvas(
    case: str, mode: str, caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    seed_kw, reason = REFUSALS[case]
    tag = f"t0081-ref-{case}-{mode}"[:64]
    w = World(_cfg(), cd=True, seed=3)
    w.add("c", xargs=sd.with_seed(_base(tag, mode), **seed_kw))
    w.add("u")
    pre = w.prefill()
    ids = seed_kw.get("ids")
    if ids is not None and len(ids) <= CL:
        assert sd.row(pre, "canvas", "c")[: len(ids)] != list(ids), "a refused request was seeded"
    logits = sd.fixed_logits([{t: 3.0 for t in LIVE}])
    first = None
    for _ in range(3):
        s = w.decode(logits)
        got = s.committed("c")
        if got is not None:
            first = (s.call, got)
            break
    assert first is not None, f"{case}: the refused request never committed"
    assert first[1][0] == 1 and set(first[1]) == {1}, f"{case}: not an all-EOS canvas: {first[1]}"
    assert first[0] <= pre.call + 2, f"{case}: EOS commit later than two decode calls"
    w.remove("c")
    mine = records.tagged(tag)
    errs = [r.get("error") for r in mine if r.get("error") is not None]
    assert errs == [reason], f"{case}: error records {errs}, expected exactly [{reason!r}]"
    assert not [r for r in mine if r.get("error") is None], f"{case}: a non-error record: {mine}"


@pytest.mark.parametrize(
    "seed_kw",
    [
        {"ids": list(sd.DRAFT8)},
        {"ids": list(sd.DRAFT8), "start": MAX_STEPS - 1},
        {"ids": [262_143] * CL},
    ],
    ids=["len_eq_cl", "start_max_minus_1", "top_id"],
)
def test_boundary_values_are_served_seeded(
    seed_kw: dict[str, Any], caplog: Any, capsys: Any
) -> None:
    """The guard is not too wide: CL ids, start step ``max - 1`` and id ``V - 1`` are accepted and seeded."""
    records = Records(caplog, capsys)
    tag = "t0081-boundary"
    w = World(_cfg(), cd=True, seed=3)
    w.add("c", xargs=sd.with_seed(_base(tag), **seed_kw))
    pre = w.prefill()
    assert sd.row(pre, "canvas", "c") == list(seed_kw["ids"])
    assert sd.row(pre, "step", "c") == [seed_kw.get("start", 0)]
    w.decode(sd.fixed_logits([{t: 3.0 for t in LIVE}]))
    w.remove("c")
    assert not [r for r in records.tagged(tag) if r.get("error") is not None]


# --------------------------------------------------------------------------- the fake server records the seed


def test_fake_server_reports_the_decoded_seed_of_each_request(tmp_path: Path) -> None:
    from local_astra.contracts.cd import CdRequest
    from tests.cd.fakes.vllm_server import FakeVllmServer, Reply

    g = ps.loop_grammar(LIVE, "t0081-fake")
    reqs = [
        sd.cd_request(
            mode="perstep_mar",
            automaton=g.payload,
            automaton_sha256=ws.sha(g.payload),
            record_tag="t0081.fake.seeded",
            draft=sd.b64(sd.DRAFT8),
            draft_noise=0.5,
            draft_start_step=2,
        ),
        CdRequest(
            mode="end_projection",
            automaton=g.payload,
            automaton_sha256=ws.sha(g.payload),
            record_tag="t0081.fake.plain",
        ),
    ]
    script = lambda i, body: Reply(committed_ids=[1000, 1])  # noqa: E731
    with FakeVllmServer(script, tmp_path, parse_xargs=ws.parse) as srv:
        for r in reqs:
            _post(srv.url, _body(r.to_xargs()))
        seeds = getattr(srv, "cd_seeds", None)
        assert callable(seeds), (
            "tests/cd/fakes/vllm_server.py: FakeVllmServer.cd_seeds() is missing"
        )
        got = seeds()
        recs = list(srv.records)
    assert got == [
        {"draft": list(sd.DRAFT8), "draft_noise": 0.5, "draft_start_step": 2},
        None,
    ], got
    errs = {r["record_tag"]: r["error"] for r in recs}
    assert errs == {"t0081.fake.seeded": None, "t0081.fake.plain": None}, errs


# --------------------------------------------------------------------------- fake | real: a 1.4 KB draft is served


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    import urllib.request

    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def _body(xargs: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "messages": [{"role": "user", "content": "Reply with the digits 1 2 3 4."}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
    }
    if xargs:
        body["vllm_xargs"] = xargs
    return body


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.gpu)])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    if request.param == "real":
        url, log = os.environ.get("ASTRA_CD_REAL_URL"), os.environ.get("ASTRA_CD_ENGINE_LOG")
        if not url or not log:
            pytest.fail(
                "backend=real needs ASTRA_CD_REAL_URL and ASTRA_CD_ENGINE_LOG (run via /launch)"
            )
        yield url, Path(log)
        return
    from tests.cd.fakes.vllm_server import FakeVllmServer, Reply

    with FakeVllmServer(
        lambda i, b: Reply(committed_ids=[1]), tmp_path, parse_xargs=ws.parse
    ) as srv:
        yield srv.url, srv.engine_log


@pytest.mark.contract
def test_a_256_id_draft_passes_the_server_and_is_not_refused(backend: tuple[str, Path]) -> None:
    """The task's unverified check: a ``vllm_xargs`` string of ~1.4 KB reaches the plugin and is accepted (the record
    of the request carries ``error = null``). The loop language over the whole vocabulary admits any draft."""

    url, log = backend
    g = ps.full_vocab_grammar("t0081-1k4")
    ids = [int(x) for x in np.random.default_rng(81).integers(2, VOCAB, 256)]
    tag = f"t0081-1k4-{time.time_ns()}"
    req = sd.cd_request(
        mode="perstep_mar",
        automaton=g.payload,
        automaton_sha256=ws.sha(g.payload),
        record_tag=tag,
        draft=sd.b64(ids),
        draft_noise=0.25,
    )
    assert len(req.to_xargs()[sd.DRAFT_KEYS["draft"]]) == 1368
    _post(url, _body(req.to_xargs()))
    deadline = time.monotonic() + 5.0
    found: list[dict[str, Any]] = []
    while time.monotonic() < deadline and not found:
        for line in log.read_text().splitlines():
            if "ASTRA_CD " in line:
                rec = json.loads(line.split("ASTRA_CD ", 1)[1])
                if rec.get("record_tag") == tag:
                    found.append(rec)
        time.sleep(0.05)
    assert found, f"no ASTRA_CD record with record_tag={tag}"
    assert all(r["error"] is None for r in found), found
