"""T-0069 criteria 4-9: every check of ``local_astra/eval/cd/smoke.py`` has a fire row and a benign row.

``smoke.py`` is the pure, torch-free half of the tier-3 smoke: it decides criteria 4-9 over saved evidence
(responses, tap frames, the engine-log slice). The driver (``scripts/gpu/cd_smoke.py``) and the independent
re-check (``tests/integration/test_cd_engine_smoke.py``) both read the same evidence; this module pins what
each check must catch, with every expected value written by hand.

API these rows fix (test-author, 2026-10-03; becomes the ruling when recorded in docs/tasks/T-0069.md).
Every ``*_problems`` function returns ``list[str]``: one human-readable problem per finding, ``[]`` = pass.
Records are ``CdCanvasRecord`` JSON dicts (as parsed from ``ASTRA_CD <json>`` lines); frames are
``astra.canvas.step`` wire dicts; release rows are ``local_astra.bench.b16_tap.parse_tap_release_lines`` dicts.

``belongs(engine_request_id, response_id) -> bool``
    the join key (Implementation notes): True iff ``engine_request_id == response_id + "-" + 8 lowercase hex``.
``frames_for(frames, response_id) -> list[dict]``
    the frames whose ``request_id`` belongs to ``response_id``, in input order.
``frame_problems(frames)``
    every frame validates as ``AstraCanvasStep`` (R10 shape per phase), denoise ``0 <= step <= max_steps``
    (R15), commit ``0 <= step <= max_steps`` (R14 as amended by ruling O6), ``accepted`` and ``entropy`` None.
``record_problems(records, *, record_tag, arm)``
    criterion 4(a)(b) + "error is null": every record echoes ``record_tag`` and names the arm's mode
    (``arm`` is the T-0070 arm name; ``perstep_mar_sc`` records say ``perstep_mar``, ruling W5); ``canvas_idx``
    is 0..n-1 with no gap and no duplicate; the final record carries ``stop_pos`` or ``error``; no record has
    a non-null ``error``; an empty list is a problem.
``replay_problems(records, accepts)``
    criterion 4(c): ``accepts(ids) -> bool`` is the caller's host replay (T-0070 ``replay.accepts`` bound to
    the request's automaton); ``ids`` is the concatenation of ``committed_ids`` in ``canvas_idx`` order. A
    rejection is a problem, and so is replay disagreeing with the final record's ``in_final``.
``release_problems(release_rows, response_id)``
    criterion 4(d): at least one release line of this request (by ``belongs``), and ``commit_overrun == 0`` on
    every one of them.
``commit_frame_problems(frames, records)``
    criterion 4(e): the request's commit frames equal its records in number and, matched on ``canvas_idx``,
    ``committed_tokens == committed_ids``.
``commit_count_problems(frames, completion_tokens)``
    criterion 8: the number of commit frames equals ``ceil(completion_tokens / 256)`` (CL 256).
``preempted_count(records) -> int``
    T-0067 S10, reported: the records whose ``error == "preempted"``.
``loud_problems(content, records, *, predicted)``
    criterion 5's X: empty content (``""`` or ``None``); exactly one record, ``canvas_idx`` 0, first committed id
    1 (EOS), ``error == predicted``, and ``predicted`` is ``bad_encoding`` or ``structure``.
``serving_problems(*, next_status, health_status)``
    criterion 5: the next unconstrained request returned 200 and ``/health`` returned 200.
``multicanvas_problems(records, *, literal_ids, completion_tokens, max_tokens=300)``
    criterion 6: the literal has 260-290 ids (so its minimum accepted length, literal + stop, is > 256); exactly
    two records, ``canvas_idx`` 0 and 1, bucket 512; the committed ids are the literal followed by a stop id
    (replay of a literal grammar); ``completion_tokens <= max_tokens``; any ``error`` (``internal`` = an S9-caught
    OOM included) is a problem.
``transport_problems(near, over, *, cap)``
    criterion 7: ``near``/``over`` are ``{"length", "status", "records", "predicted"}``. ``near["length"] == cap -
    512`` and ``over["length"] == cap + 1``; any HTTP status other than 200 is a problem (4xx/413 goes to the
    human, ruling O3); each has one record whose ``error`` equals ``predicted``; near's code is
    ``bad_encoding`` (``sha_mismatch`` there is a fail, never proof of transport) and over's is
    ``payload_too_large``.
``boot_problems(log, reference)``
    criterion 8 / R16: ``boot_config_lines(last_boot(log))`` equals ``reference`` minus its ``#`` and blank lines.
    A log with no boot is a problem, not an exception.
``vllm_sh_problems(diff)``
    criterion 8: ``git diff 806da12..HEAD -- services/vllm.sh`` must be empty; anything else is a refusal.
``quiet_window_problems(window)``
    criterion 8: the unconstrained request's log window holds no ``ASTRA_CD `` and no ``astra_cd `` line.
``header_report(rows) -> dict``
    criterion 9, reported not gated: ``rows`` are ``{"task", "dataset_id", "token_ids"}``; returns ``n``,
    ``first8`` (per row, ``token_ids[:8]``), ``header_prefix`` (rows with ``token_ids[0] == 100``), ``rate``
    (``header_prefix / n``) and ``problems``: not 20 rows, not 10 Sudoku + 10 BFCL, or a row without
    ``token_ids`` (E5's field absent). The rate itself is never a problem.
``token_ids_problems(token_ids, records) -> (list[str], bool | None)``
    criterion 9: ``token_ids`` equals the concatenated ``committed_ids`` (``canvas_idx`` order) up to the first
    stop id, either without or with that stop id; the second value says whether the stop id was included
    (``None`` when ``token_ids`` is absent, which is a problem).

Regime: CPU, pure Python, hand-built inputs; the boot rows use ``tests/fixtures/engine/
vllm-coldboot-bd8001b.excerpt.log`` and the 806da12 boot reference written out below by hand.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.contracts.cd import CdCanvasRecord
from local_astra.contracts.protocol import AstraMessageAdapter

REPO = Path(__file__).resolve().parents[3]
EXCERPT = REPO / "tests" / "fixtures" / "engine" / "vllm-coldboot-bd8001b.excerpt.log"
CAP = 1 << 20  # T-0062 decision 4
RESP = "chatcmpl-a55225d4582a56ef"
ENGINE = RESP + "-9ca09c90"
OTHER = "chatcmpl-a55225d4582a56ef0-1a2b3c4d"  # another response sharing RESP as a string prefix
STOPS = (1, 50, 106)

#: ``logs/t0011-tier3-806da12/boot-reference.txt`` (untracked; read 2026-10-03), copied by hand.
REFERENCE_806DA12 = """\
# boot_config_lines(logs/vllm.log) at 806da1217a41c58cb5728a9fbbc4ed6f4b0b0366
# compared with tests/fixtures/engine/vllm-coldboot-bd8001b.excerpt.log (bd8001b, plugin on, no tap)
arg attention_backend: 'TRITON_ATTN'
arg diffusion_config: {'canvas_length': 256, 'max_denoising_steps': 48}
arg enable_prefix_caching: True
arg hf_overrides: {'architectures': ['AstraDiffusionGemma']}
arg kv_cache_dtype: 'bfloat16'
arg kv_cache_memory_bytes: 4294967296
arg max_model_len: 16384
arg max_num_batched_tokens: 4096
arg max_num_seqs: 3
arg override_generation_config: {'max_new_tokens': None}
attention backend: TRITON_ATTN
fp4 marlin fallback: absent
kv cache tokens / max len / concurrency: 31,605 / 16,384 / 1.93
kv_cache_dtype (engine): bfloat16
nvfp4 moe backend: FLASHINFER_CUTLASS
resolved architecture: AstraDiffusionGemma
"""


def smoke() -> ModuleType:
    try:
        return importlib.import_module("local_astra.eval.cd.smoke")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("local_astra.eval.cd.smoke"):
            pytest.fail(
                f"T-0069 not implemented: local_astra/eval/cd/smoke.py does not exist ({exc})"
            )
        raise


def fn(name: str) -> Any:
    f = getattr(smoke(), name, None)
    if f is None:
        pytest.fail(f"T-0069 not implemented: local_astra.eval.cd.smoke.{name} is missing")
    return f


# --------------------------------------------------------------------------- builders (on-contract by construction)


def rec(
    idx: int,
    ids: Sequence[int],
    *,
    tag: str = "t69-c4-sudoku-mar",
    arm: str = "perstep_mar",
    error: str | None = None,
    in_final: bool = False,
    bucket: int | None = 64,
) -> dict[str, Any]:
    stop = next((i for i, t in enumerate(ids) if t in STOPS), None)
    out = {
        "record_tag": tag,
        "arm": arm,
        "canvas_idx": idx,
        "committed_ids": list(ids),
        "state_set_after": [] if error else [3],
        "in_final": in_final,
        "steps": 0 if error else 12,
        "bucket": bucket,
        "stop_pos": stop,
        "error": error,
        "mean_accepted_per_step": None if (error or arm == "end_projection") else 2.5,
        "shadow_argmax_ids": None,
        "argmax_in_language": None,
        "map_equals_argmax": None,
    }
    CdCanvasRecord.model_validate(out)  # fixture: every record here is on-contract
    return out


CL = 8


def denoise(
    idx: int, step: int, *, rid: str = ENGINE, max_steps: int = 48, **extra: Any
) -> dict[str, Any]:
    f: dict[str, Any] = {
        "type": "astra.canvas.step",
        "request_id": rid,
        "canvas_idx": idx,
        "step": step,
        "max_steps": max_steps,
        "canvas_length": CL,
        "canvas": [7] * CL,
        "argmax": [7] * CL,
        "stable": "f0",
        "phase": "denoise",
    }
    f.update(extra)
    return f


def commit(
    idx: int, tokens: Sequence[int], *, step: int = 12, rid: str = ENGINE, **extra: Any
) -> dict[str, Any]:
    f: dict[str, Any] = {
        "type": "astra.canvas.step",
        "request_id": rid,
        "canvas_idx": idx,
        "step": step,
        "max_steps": 48,
        "canvas_length": CL,
        "phase": "commit",
        "committed_tokens": list(tokens),
    }
    f.update(extra)
    return f


def test_the_builders_are_on_contract() -> None:
    """Fixture self-test: the benign frames validate through the real adapter, so a fire row below fires
    because of the one thing it changes, not because the builder was off-contract."""
    for f in (denoise(0, 0), denoise(0, 48), commit(0, [5, 6, 1]), commit(1, [1], step=48)):
        AstraMessageAdapter.validate_python(f)


# --------------------------------------------------------------------------- join key


def test_belongs_accepts_the_response_id_plus_eight_hex() -> None:
    assert fn("belongs")(ENGINE, RESP) is True
    assert fn("belongs")("chatcmpl-a80dd3bd8d190012-92b2aafc", "chatcmpl-a80dd3bd8d190012") is True


@pytest.mark.parametrize(
    "engine_id",
    [
        OTHER,  # a longer response id that starts with RESP: never joined by prefix alone
        RESP,  # no suffix
        RESP + "-9ca09c9",  # 7 hex
        RESP + "-9ca09c900",  # 9 hex
        RESP + "-9CA09C90",  # upper case is not what vLLM writes
        RESP + "-9ca09c9g",  # not hex
        RESP + "_9ca09c90",  # wrong separator
    ],
)
def test_belongs_refuses_everything_else(engine_id: str) -> None:
    assert fn("belongs")(engine_id, RESP) is False


def test_frames_for_keeps_only_this_requests_frames_in_order() -> None:
    a, b, c = denoise(0, 1), denoise(0, 1, rid=OTHER), commit(0, [5, 1], step=2)
    assert fn("frames_for")([a, b, c], RESP) == [a, c]
    assert fn("frames_for")([b], RESP) == []


# --------------------------------------------------------------------------- frame_problems


def test_valid_frames_have_no_problems_including_the_step_cap_edges() -> None:
    """R15/R14 edges are benign: a denoise frame at step 0 and at max_steps, a commit at 0 (ruling O6), 1 and
    max_steps."""
    frames = [
        denoise(0, 0),
        denoise(0, 48),
        commit(0, [5, 6, 1], step=48),
        commit(1, [9], step=1),
        commit(2, [1, 1], step=0),
    ]
    assert fn("frame_problems")(frames) == []


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(denoise(0, 3, accepted="ff"), id="accepted-filled"),
        pytest.param(denoise(0, 3, entropy=[0.0] * CL), id="entropy-filled"),
        pytest.param(denoise(0, 49), id="denoise-step-above-max"),
        pytest.param(commit(0, [5, 1], step=-1), id="commit-step-negative"),
        pytest.param(commit(0, [5, 1], step=49), id="commit-step-above-max"),
        pytest.param(denoise(0, 3, committed_tokens=[5]), id="denoise-with-committed-tokens"),
        pytest.param(commit(0, [5, 1], canvas=[7] * CL), id="commit-carrying-a-row"),
        pytest.param(commit(0, []), id="commit-without-tokens"),
        pytest.param(denoise(0, 3, canvas=[7] * (CL - 1)), id="canvas-wrong-length"),
    ],
)
def test_each_invalid_frame_is_a_problem(bad: dict[str, Any]) -> None:
    good = [denoise(0, 1), commit(0, [5, 1])]
    assert fn("frame_problems")(good) == []
    assert fn("frame_problems")([*good, bad]) != []


# --------------------------------------------------------------------------- record_problems (4a, 4b, error null)


def _two() -> list[dict[str, Any]]:
    return [rec(0, [1000, 1001, 1002]), rec(1, [1003, 1, 1], in_final=True)]


def test_records_that_echo_tag_and_arm_and_close_on_a_stop_pass() -> None:
    assert fn("record_problems")(_two(), record_tag="t69-c4-sudoku-mar", arm="perstep_mar") == []


def test_perstep_mar_sc_records_name_their_mode() -> None:
    """Ruling W5: the record names the mode, so an SC request's records say ``perstep_mar``."""
    recs = _two()
    assert fn("record_problems")(recs, record_tag="t69-c4-sudoku-mar", arm="perstep_mar_sc") == []
    assert fn("record_problems")(recs, record_tag="t69-c4-sudoku-mar", arm="end_projection") != []


def _two_qa() -> list[dict[str, Any]]:
    return [
        rec(0, [1000, 1001, 1002], tag="t76-qa", arm="perstep_mar_qargmax"),
        rec(1, [1003, 1, 1], tag="t76-qa", arm="perstep_mar_qargmax", in_final=True),
    ]


@pytest.mark.parametrize("arm", ["perstep_mar_qargmax", "perstep_mar_qargmax_sc"])
def test_qargmax_records_name_the_qargmax_mode(arm: str) -> None:
    """T-0076 criteria 7/8 (``smoke._ARM_MODE``): both new arms' records name ``perstep_mar_qargmax`` (the SC
    twin's too, as W5 has Mar+SC's say ``perstep_mar``)."""
    assert fn("record_problems")(_two_qa(), record_tag="t76-qa", arm=arm) == []


@pytest.mark.parametrize("arm", ["perstep_mar_qargmax", "perstep_mar_qargmax_sc"])
def test_mar_records_are_a_problem_for_the_qargmax_arms(arm: str) -> None:
    """The other direction: a qargmax run whose records say ``perstep_mar`` (the request went out as Mar)
    is caught, and Mar's arm does not accept qargmax records."""
    mar = [rec(0, [1000], tag="t76-qa"), rec(1, [1, 1], tag="t76-qa", in_final=True)]
    assert fn("record_problems")(mar, record_tag="t76-qa", arm=arm) != []
    assert fn("record_problems")(_two_qa(), record_tag="t76-qa", arm="perstep_mar") != []


def test_a_final_record_closed_by_an_error_satisfies_4b_but_the_error_is_still_a_problem() -> None:
    recs = [rec(0, [1000]), rec(1, [1, 1], error="z0")]
    problems = fn("record_problems")(recs, record_tag="t69-c4-sudoku-mar", arm="perstep_mar")
    assert len(problems) == 1 and "z0" in problems[0], problems


@pytest.mark.parametrize(
    "recs",
    [
        pytest.param([rec(0, [1000]), rec(2, [1001, 1])], id="gap-in-canvas_idx"),
        pytest.param([rec(1, [1000]), rec(2, [1001, 1])], id="does-not-start-at-0"),
        pytest.param([rec(0, [1000]), rec(0, [1000]), rec(1, [1])], id="duplicate-canvas_idx"),
        pytest.param([rec(0, [1000]), rec(1, [1001, 1002])], id="final-record-without-stop"),
        pytest.param([rec(0, [1000]), rec(1, [1, 1], tag="t69-other")], id="tag-not-echoed"),
        pytest.param([rec(0, [1000], arm="perstep_mf"), rec(1, [1])], id="arm-not-named"),
        pytest.param([], id="no-records"),
    ],
)
def test_each_record_defect_is_a_problem(recs: list[dict[str, Any]]) -> None:
    assert fn("record_problems")(recs, record_tag="t69-c4-sudoku-mar", arm="perstep_mar") != []


# --------------------------------------------------------------------------- replay_problems (4c)


def _oracle(want: Sequence[int]) -> Any:
    seen: list[list[int]] = []

    def accepts(ids: Sequence[int]) -> bool:
        seen.append(list(ids))
        return list(ids) == list(want)

    accepts.seen = seen  # type: ignore[attr-defined]
    return accepts


def test_replay_accepted_and_agreeing_with_in_final_passes_in_canvas_order() -> None:
    """The replay sees the canvases concatenated in ``canvas_idx`` order even when they arrive out of order."""
    acc = _oracle([1000, 1001, 1002, 1003, 1, 1])
    recs = list(reversed(_two()))
    assert fn("replay_problems")(recs, acc) == []
    assert acc.seen and acc.seen[-1] == [1000, 1001, 1002, 1003, 1, 1], acc.seen


def test_replay_rejection_is_a_problem() -> None:
    recs = [rec(0, [1000, 1001, 1002]), rec(1, [1003, 1, 1], in_final=False)]
    assert fn("replay_problems")(recs, _oracle([9])) != []


def test_replay_disagreeing_with_in_final_is_a_problem() -> None:
    recs = [rec(0, [1000, 1001, 1002]), rec(1, [1003, 1, 1], in_final=False)]
    assert fn("replay_problems")(recs, _oracle([1000, 1001, 1002, 1003, 1, 1])) != []


# --------------------------------------------------------------------------- release_problems (4d)


def _release(rid: str, overrun: int = 0) -> dict[str, Any]:
    return {
        "req_id": rid,
        "submitted": 9,
        "published": 9,
        "dropped": 0,
        "errors": 0,
        "commit_overrun": overrun,
    }


def test_one_release_line_with_no_overrun_passes() -> None:
    rows = [_release("chatcmpl-0000000000000000-00000000", 3), _release(ENGINE)]
    assert fn("release_problems")(rows, RESP) == []


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([_release(ENGINE, 1)], id="commit-overrun"),
        pytest.param([], id="no-release-line"),
        pytest.param([_release(OTHER)], id="only-a-prefix-sharing-request"),
        pytest.param([_release(ENGINE), _release(ENGINE, 1)], id="a-second-line-with-overrun"),
    ],
)
def test_release_defects_are_problems(rows: list[dict[str, Any]]) -> None:
    assert fn("release_problems")(rows, RESP) != []


# --------------------------------------------------------------------------- commit_frame_problems (4e)


def test_commit_frames_equal_to_the_records_pass_with_denoise_frames_between() -> None:
    frames = [denoise(0, 1), commit(0, [1000, 1001, 1002]), denoise(1, 1), commit(1, [1003, 1, 1])]
    assert fn("commit_frame_problems")(frames, _two()) == []


def test_a_commit_frame_differing_from_its_record_at_one_position_is_a_problem() -> None:
    frames = [commit(0, [1000, 1001, 1002]), commit(1, [1003, 1, 50])]  # last id: 50, record says 1
    assert fn("commit_frame_problems")(frames, _two()) != []


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param([commit(0, [1000, 1001, 1002])], id="one-frame-two-records"),
        pytest.param(
            [commit(0, [1000, 1001, 1002]), commit(1, [1003, 1, 1]), commit(2, [1])],
            id="three-frames-two-records",
        ),
        pytest.param(
            [commit(1, [1000, 1001, 1002]), commit(0, [1003, 1, 1])],
            id="tokens-on-the-wrong-canvas",
        ),
    ],
)
def test_commit_frame_count_or_canvas_mismatch_is_a_problem(frames: list[dict[str, Any]]) -> None:
    assert fn("commit_frame_problems")(frames, _two()) != []


# --------------------------------------------------------------------------- commit_count_problems (criterion 8)


@pytest.mark.parametrize(
    ("n_commits", "completion"), [(1, 1), (1, 256), (2, 257), (2, 300), (3, 513)]
)
def test_commit_frames_matching_ceil_completion_over_256_pass(
    n_commits: int, completion: int
) -> None:
    frames = [denoise(0, 0)] + [commit(i, [5]) for i in range(n_commits)]
    assert fn("commit_count_problems")(frames, completion) == []


@pytest.mark.parametrize(("n_commits", "completion"), [(2, 256), (1, 257), (2, 513), (0, 1)])
def test_commit_frames_not_matching_ceil_completion_over_256_are_problems(
    n_commits: int, completion: int
) -> None:
    frames = [denoise(0, 0)] + [commit(i, [5]) for i in range(n_commits)]
    assert fn("commit_count_problems")(frames, completion) != []


# --------------------------------------------------------------------------- preempted_count (S10, reported)


def test_preempted_count_counts_only_preempted_records() -> None:
    recs = [
        rec(0, [1], error="preempted"),
        rec(0, [1000], tag="t69-b"),
        rec(0, [1], tag="t69-c", error="z0"),
        rec(0, [1], tag="t69-d", error="preempted"),
    ]
    assert fn("preempted_count")(recs) == 2
    assert fn("preempted_count")([rec(0, [1000])]) == 0


# --------------------------------------------------------------------------- loud_problems (criterion 5's X)


def _loud(code: str = "bad_encoding", ids: Sequence[int] = (1, 1, 1)) -> dict[str, Any]:
    return rec(0, ids, tag="t69-c5-x", arm="end_projection", error=code, bucket=None)


@pytest.mark.parametrize("content", ["", None])
@pytest.mark.parametrize("code", ["bad_encoding", "structure"])
def test_a_loud_request_with_the_predicted_code_passes(content: str | None, code: str) -> None:
    assert fn("loud_problems")(content, [_loud(code)], predicted=code) == []


@pytest.mark.parametrize(
    ("content", "recs", "predicted"),
    [
        pytest.param("", [_loud(ids=(2, 1, 1))], "bad_encoding", id="eos-less-loud-record"),
        pytest.param("", [_loud("structure")], "bad_encoding", id="code-not-the-predicted-one"),
        pytest.param("", [_loud(), _loud()], "bad_encoding", id="two-records"),
        pytest.param("", [], "bad_encoding", id="no-record"),
        pytest.param(
            "",
            [{**_loud(), "canvas_idx": 1}],
            "bad_encoding",
            id="record-not-canvas-0",
        ),
        pytest.param("hello", [_loud()], "bad_encoding", id="non-empty-content"),
        pytest.param(
            "", [_loud("sha_mismatch")], "sha_mismatch", id="sha-not-recomputed-so-not-this-row"
        ),
    ],
)
def test_loud_defects_are_problems(
    content: str | None, recs: list[dict[str, Any]], predicted: str
) -> None:
    assert fn("loud_problems")(content, recs, predicted=predicted) != []


# --------------------------------------------------------------------------- serving_problems (criterion 5)


def test_the_engine_still_serving_passes() -> None:
    assert fn("serving_problems")(next_status=200, health_status=200) == []


@pytest.mark.parametrize(("nxt", "health"), [(500, 200), (200, 503), (0, 0)])
def test_the_engine_not_serving_is_a_problem(nxt: int, health: int) -> None:
    assert fn("serving_problems")(next_status=nxt, health_status=health) != []


# --------------------------------------------------------------------------- multicanvas_problems (criterion 6)


def _literal(n: int = 270) -> list[int]:
    return [2000 + i for i in range(n)]


def _mc(lit: list[int], *, tail: Sequence[int] = (1,), **kw: Any) -> list[dict[str, Any]]:
    return [
        rec(0, lit[:256], tag="t69-c6-mar", bucket=512),
        rec(1, [*lit[256:], *tail], tag="t69-c6-mar", bucket=512, in_final=True, **kw),
    ]


def test_a_literal_closed_in_two_canvases_passes() -> None:
    lit = _literal(270)
    got = fn("multicanvas_problems")(_mc(lit), literal_ids=lit, completion_tokens=271)
    assert got == []


@pytest.mark.parametrize("n", [260, 290])
def test_the_literal_length_edges_pass(n: int) -> None:
    lit = _literal(n)
    assert fn("multicanvas_problems")(_mc(lit), literal_ids=lit, completion_tokens=n + 1) == []


def test_emitted_exactly_300_passes() -> None:
    """The budget edge is inclusive (criterion 6: ``emitted <= 300``); kills a ``>`` -> ``>=`` mutant."""
    lit = _literal(270)
    assert fn("multicanvas_problems")(_mc(lit), literal_ids=lit, completion_tokens=300) == []


def test_the_smoke_constants_match_the_harness() -> None:
    """Drift row: smoke.py repeats T-0070's stop set and header id; a dropped or changed id must read here."""
    from local_astra.eval.cd import replay

    assert smoke().STOP_IDS == replay.STOP_IDS == frozenset({1, 50, 106})
    assert smoke().HEADER_ID == replay.CHANNEL_OPEN_ID == 100


def _three(lit: list[int]) -> list[dict[str, Any]]:
    recs = _mc(lit)
    return [*recs, rec(2, [1], tag="t69-c6-mar", bucket=512)]


@pytest.mark.parametrize(
    ("recs_of", "n", "completion"),
    [
        pytest.param(lambda lit: _mc(lit)[:1], 270, 256, id="one-canvas"),
        pytest.param(_three, 270, 271, id="three-canvases"),
        pytest.param(
            lambda lit: [{**r, "bucket": 256} for r in _mc(lit)], 270, 271, id="bucket-not-512"
        ),
        pytest.param(lambda lit: _mc(lit), 259, 260, id="literal-too-short"),
        pytest.param(lambda lit: _mc(lit), 291, 292, id="literal-too-long"),
        pytest.param(lambda lit: _mc(lit), 270, 301, id="emitted-above-300"),
        pytest.param(lambda lit: _mc(lit, tail=(1,), error="internal"), 270, 271, id="internal"),
        pytest.param(lambda lit: _mc(lit, tail=()), 270, 270, id="no-stop-after-the-literal"),
        pytest.param(
            lambda lit: [_mc(lit)[0], {**_mc(lit)[1], "committed_ids": [lit[256], 9, 1]}],
            270,
            259,
            id="literal-not-reproduced",
        ),
        pytest.param(
            lambda lit: [
                _mc(lit)[0],
                {**_mc(lit)[1], "committed_ids": [*lit[256:260], 9, *lit[261:], 1]},
            ],
            270,
            271,
            id="one-literal-id-differs-stop-still-at-len",
        ),
    ],
)
def test_multicanvas_defects_are_problems(recs_of: Any, n: int, completion: int) -> None:
    lit = _literal(n)
    recs = recs_of(lit)
    assert fn("multicanvas_problems")(recs, literal_ids=lit, completion_tokens=completion) != []


# --------------------------------------------------------------------------- transport_problems (criterion 7)


def _case(
    length: int, code: str | None, *, status: int = 200, predicted: str | None = None
) -> dict[str, Any]:
    recs = [] if code is None else [rec(0, [1, 1], tag="t69-c7", error=code, bucket=None)]
    return {
        "length": length,
        "status": status,
        "records": recs,
        "predicted": code if predicted is None else predicted,
    }


def _near(**kw: Any) -> dict[str, Any]:
    kw.setdefault("code", "bad_encoding")
    code = kw.pop("code")
    return _case(kw.pop("length", CAP - 512), code, **kw)


def _over(**kw: Any) -> dict[str, Any]:
    kw.setdefault("code", "payload_too_large")
    code = kw.pop("code")
    return _case(kw.pop("length", CAP + 1), code, **kw)


def test_the_payload_arriving_intact_at_the_cap_passes() -> None:
    assert fn("transport_problems")(_near(), _over(), cap=CAP) == []


@pytest.mark.parametrize(
    ("near", "over"),
    [
        pytest.param(
            _near(code="sha_mismatch", predicted="bad_encoding"), _over(), id="near-sha-mismatch"
        ),
        pytest.param(
            _near(code="sha_mismatch"), _over(), id="near-sha-mismatch-even-when-predicted"
        ),
        pytest.param(_near(status=413, code=None), _over(), id="near-413"),
        pytest.param(_near(), _over(status=413, code=None), id="over-413"),
        pytest.param(_near(), _over(status=400, code=None), id="over-400"),
        pytest.param(_near(length=CAP - 511), _over(), id="near-length-off-by-one"),
        pytest.param(_near(), _over(length=CAP), id="over-length-at-the-cap"),
        pytest.param(_near(), _over(code="bad_encoding"), id="over-decoded-before-the-size-check"),
        pytest.param(_near(predicted="structure"), _over(), id="host-predicts-another-code"),
        pytest.param(_near(code=None), _over(), id="near-no-record"),
    ],
)
def test_transport_defects_are_problems(near: dict[str, Any], over: dict[str, Any]) -> None:
    assert fn("transport_problems")(near, over, cap=CAP) != []


# --------------------------------------------------------------------------- boot_problems (criterion 8, R16)


def _excerpt() -> str:
    return EXCERPT.read_text()


def _other_boot() -> str:
    """The same boot with another attention backend: a different configuration, same shape."""
    return _excerpt().replace("TRITON_ATTN", "FLASH_ATTN")


def test_the_boot_matching_the_reference_passes() -> None:
    assert fn("boot_problems")(_excerpt(), REFERENCE_806DA12) == []


def test_only_the_last_boot_is_compared() -> None:
    """R16 in the benign direction: an earlier, different boot in the same append-mode log is ignored."""
    assert fn("boot_problems")(_other_boot() + _excerpt(), REFERENCE_806DA12) == []


def test_a_slice_from_the_wrong_boot_is_a_problem() -> None:
    """R16 in the firing direction: the reference matches an earlier boot, not the last one."""
    assert fn("boot_problems")(_excerpt() + _other_boot(), REFERENCE_806DA12) != []


def test_a_log_with_no_boot_is_a_problem_not_an_exception() -> None:
    assert fn("boot_problems")("INFO nothing here\n", REFERENCE_806DA12) != []


def test_a_reference_line_missing_from_the_boot_is_a_problem() -> None:
    log = "\n".join(ln for ln in _excerpt().splitlines() if "NvFp4 MoE backend" not in ln) + "\n"
    assert fn("boot_problems")(log, REFERENCE_806DA12) != []


# --------------------------------------------------------------------------- vllm_sh_problems (criterion 8)


def test_an_empty_vllm_sh_diff_passes() -> None:
    assert fn("vllm_sh_problems")("") == []


def test_a_vllm_sh_diff_is_a_refusal() -> None:
    diff = "diff --git a/services/vllm.sh b/services/vllm.sh\n-  --max-num-seqs 3\n+  --max-num-seqs 4\n"
    assert fn("vllm_sh_problems")(diff) != []


# --------------------------------------------------------------------------- quiet_window_problems (criterion 8)

_QUIET = (
    "(APIServer pid=1) INFO 10-03 12:00:01 [loggers.py:123] Engine 000: Avg prompt throughput: 51.2\n"
    "(EngineCore pid=326) INFO 10-03 12:00:02 [tap.py:700] canvas tap release: "
    "req_id=chatcmpl-a55225d4582a56ef-9ca09c90 submitted=9 published=9 dropped=0 errors=0 "
    "commit_overrun=0\n"
    '(APIServer pid=1) INFO:     127.0.0.1:50312 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
)


def test_a_window_without_cd_lines_passes() -> None:
    assert fn("quiet_window_problems")(_QUIET) == []


@pytest.mark.parametrize(
    "line",
    [
        '(EngineCore pid=326) INFO 10-03 12:00:02 [record.py:99] ASTRA_CD {"record_tag":"x"}\n',
        "(EngineCore pid=326) WARNING 10-03 12:00:02 [sampler.py:180] astra_cd request refused "
        "(tag=x): sha_mismatch; it ends with an EOS canvas\n",
    ],
    ids=["record-line", "non-record-cd-line"],
)
def test_a_cd_line_in_the_window_is_a_problem(line: str) -> None:
    assert fn("quiet_window_problems")(_QUIET + line) != []


# --------------------------------------------------------------------------- header_report (criterion 9, reported)


def _rows(n_sudoku: int = 10, n_bfcl: int = 10, header: int = 3) -> list[dict[str, Any]]:
    rows = []
    for i in range(n_sudoku + n_bfcl):
        task = "sudoku" if i < n_sudoku else "bfcl"
        ids = [100, 45, 9, 8, 7, 6, 5, 4, 3] if i < header else [1000 + i, 2, 3, 4, 5, 6, 7, 8, 9]
        rows.append({"task": task, "dataset_id": f"{task}-{i}", "token_ids": ids})
    return rows


def test_header_report_counts_by_hand() -> None:
    rows = _rows(header=3)
    got = fn("header_report")(rows)
    assert got["n"] == 20
    assert got["header_prefix"] == 3
    assert got["rate"] == pytest.approx(0.15)
    assert got["first8"][0] == [100, 45, 9, 8, 7, 6, 5, 4]
    assert got["first8"][5] == [1005, 2, 3, 4, 5, 6, 7, 8]
    assert got["problems"] == []


@pytest.mark.parametrize("header", [0, 20])
def test_any_header_rate_is_reported_never_a_problem(header: int) -> None:
    """Criterion 9 / T-0070 8(f): the rate is reported, not a gate -- 0 % and 100 % are both benign."""
    got = fn("header_report")(_rows(header=header))
    assert got["problems"] == [] and got["header_prefix"] == header


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(_rows(9, 10), id="19-rows"),
        pytest.param(_rows(11, 9), id="11-sudoku-9-bfcl"),
        pytest.param(
            [*_rows()[:19], {"task": "bfcl", "dataset_id": "bfcl-19", "token_ids": None}],
            id="token_ids-absent",
        ),
    ],
)
def test_header_report_flags_the_sample_not_the_rate(rows: list[dict[str, Any]]) -> None:
    assert fn("header_report")(rows)["problems"] != []


# --------------------------------------------------------------------------- token_ids_problems (criterion 9)


def test_token_ids_up_to_the_first_stop_pass_and_say_whether_the_stop_is_included() -> None:
    recs = [rec(1, [7, 50, 3]), rec(0, [5, 6])]  # out of order: concatenation is by canvas_idx
    assert fn("token_ids_problems")([5, 6, 7], recs) == ([], False)
    assert fn("token_ids_problems")([5, 6, 7, 50], recs) == ([], True)


@pytest.mark.parametrize(
    "token_ids",
    [
        pytest.param([5, 6, 8], id="one-id-differs"),
        pytest.param([5, 6], id="short"),
        pytest.param([5, 6, 7, 50, 3], id="past-the-stop"),
        pytest.param([5, 6, 7, 1], id="another-stop-id"),
    ],
)
def test_token_ids_not_equal_to_the_committed_prefix_are_problems(token_ids: list[int]) -> None:
    recs = [rec(0, [5, 6]), rec(1, [7, 50, 3])]
    problems, _ = fn("token_ids_problems")(token_ids, recs)
    assert problems != []


def test_absent_token_ids_are_a_problem() -> None:
    problems, included = fn("token_ids_problems")(None, [rec(0, [5, 1])])
    assert problems != [] and included is None
