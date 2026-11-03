"""T-0091: precompiled native T4 payloads, the classifier, the pure repair builder and the dry-run executor (unit, CPU).

Every expected value is from the task text, from hand arithmetic, or from the stored T-0084e/T-0084f artifacts
(``tests/fixtures/toolcall/native_t4_eval.json``: what the eval actually sent for ``native/T4/named/perstep_mar+seed``).
The cd-group twin, ``tests/cd/unit/test_toolcall_live_parity.py``, proves the same surfaces against ``twopass`` and
``decoding`` on every stored row. API assumptions: ``tests/unit/_toolcall_live_support.py``.

Criteria covered here: 1 (the file, the load check, stale refusal), 2, 3, 4, 5 (hand rows, the amendment's stop ids),
6, 7 (contracts, protocol, the coordinate mapping at 0/999/500 for widths 1280 and 1279, ``no frame``), 8 (scope pins
against the literal base ``bd2c124``). The no-socket/no-process/no-file rows are in ``test_tool_executor_dry_run.py``;
the AST fence is in ``test_toolcall_imports.py``.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart
from local_astra.toolcall import native, tools
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests import _git_range as gr
from tests.unit import _toolcall_live_support as S
from tests.unit._toolcall_live_support import CLOSE, OPEN, V1, s
from tests.unit._toolcall_live_support import native as call

CLICK01_SPOKEN = "I will click the Sign in button."
#: T-0100's three reasons, written out from the task's Definitions (never read from ``live``)
NOT_IN_ENUM = "combo not in enum"
NO_COMBO = "no combo"
OUT_OF_RANGE = "coordinate out of range"
H99_REASONS = (NOT_IN_ENUM, NO_COMBO, OUT_OF_RANGE)
#: T-0102's reason (H-T101c), pinned by value in docs/tasks/T-0102.md "Definitions" (never read from ``live``)
H101C = "model text after call"
STORED_CLICK01_SPOKEN = CLICK01_SPOKEN + "<channel|>"  # what the 9aee66d router kept (T-0085)


# =========================================================================== criterion 1: the payload file


def test_c1_the_payload_file_holds_one_self_consistent_payload_per_v1_tool() -> None:
    """Criterion 1: one entry per v1 tool, each with ``automaton`` (ASCII), ``automaton_sha256`` = sha256 of it,
    ``n_states`` and ``bucket``."""
    entries = S.tool_entries(S.payload_doc())
    assert sorted(entries) == sorted(V1), sorted(entries)
    for name, e in entries.items():
        assert {"automaton", "automaton_sha256", "n_states", "bucket"} <= set(e), (name, sorted(e))
        assert isinstance(e["automaton"], str) and e["automaton"].isascii(), name
        assert e["automaton_sha256"] == S.sha256_ascii(e["automaton"]), name


def test_c1_each_payload_is_the_one_the_eval_sent() -> None:
    """Criterion 1 (payload identity, fast-suite half): each tool's ``automaton_sha256``, ``n_states`` and ``bucket``
    equal what the stored T-0084e/T-0084f runs sent for ``native/T4/named/perstep_mar+seed`` (identical on all 120
    rows): click_at 29/32, google_search 41/64, scroll 39/64, type_at 66/128, press_key 80/128 (T-0084a's 29-80)."""
    want = S.eval_fixture()["tools"]
    entries = S.tool_entries(S.payload_doc())
    got = {
        n: {k: entries[n][k] for k in ("automaton_sha256", "n_states", "bucket")} for n in entries
    }
    assert got == want


def test_c1_every_bucket_is_at_most_128_the_live_gate() -> None:
    entries = S.tool_entries(S.payload_doc())
    assert entries, "criterion 1: no payloads"
    assert {n: e["bucket"] for n, e in entries.items() if not e["bucket"] <= 128} == {}


def test_c1_the_manifest_names_its_regime() -> None:
    """Criterion 1: the manifest holds the tools_v1 sha, the tokenizer sha, the model revision, the compiler version and
    the CL (values from the stored regime; the layout is the implementer's)."""
    reg = S.eval_fixture()["regime"]
    vals = S.scalars(S.payload_doc())
    strs = [v for v in vals if isinstance(v, str)]
    assert tools.load().version == reg["tools_sha256"]  # premise: tools_v1 is the evaluated one
    assert reg["tools_sha256"] in strs, "the manifest lacks the tools_v1 sha"
    assert reg["tokenizer_sha256"] in strs, "the manifest lacks the tokenizer sha"
    assert reg["compiler_version"] in strs, "the manifest lacks the compiler version"
    rev = reg["model_revision"]
    assert any(len(v) >= 7 and rev.startswith(v) for v in strs), (
        "the manifest lacks the model revision"
    )
    assert 256 in [v for v in vals if isinstance(v, int) and not isinstance(v, bool)], "no CL 256"


def test_c1_load_payloads_accepts_the_shipped_file() -> None:
    assert S.fn("load_payloads", "criterion 1 (load check)")() is not None


def test_c1_payload_stale_is_an_exception_class() -> None:
    stale = S.fn("PayloadStale", "criterion 1 (PayloadStale)")
    assert isinstance(stale, type) and issubclass(stale, Exception)


def _write(tmp_path: Path, doc: Any, name: str = "native_T4.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_c1_guard_width_an_identical_copy_elsewhere_loads(tmp_path: Path) -> None:
    """Guard too wide: the check refuses on a mismatch, not on a path or a re-serialisation of identical content."""
    load = S.fn("load_payloads", "criterion 1 (load check)")
    assert load(_write(tmp_path, S.payload_doc())) is not None


def _replace_everywhere(node: Any, old: str, new: str) -> Any:
    if isinstance(node, dict):
        return {k: _replace_everywhere(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_everywhere(v, old, new) for v in node]
    return new if node == old else node


def _stale_variants() -> dict[str, Any]:
    doc = S.payload_doc()
    version = tools.load().version
    out: dict[str, Any] = {}
    out["tools_sha"] = _replace_everywhere(doc, version, "0" * 64)
    assert out["tools_sha"] != doc, "premise: the tools sha appears in the file"
    for what in ("automaton", "automaton_sha256", "missing_tool", "bucket_256"):
        d = copy.deepcopy(doc)
        entries = S.tool_entries(d)
        # mutate the dict inside ``d`` that holds click_at's payload
        for _t, node in S.walk(d):
            if isinstance(node, dict) and node.get("automaton") == entries["click_at"]["automaton"]:
                if what == "automaton":
                    a = node["automaton"]
                    node["automaton"] = a[:-1] + ("A" if a[-1] != "A" else "B")
                elif what == "automaton_sha256":
                    node["automaton_sha256"] = "f" * 64
                elif what == "bucket_256":
                    node["bucket"] = 256
                break
        if what == "missing_tool":
            d = _drop_tool(d, "scroll")
        out[what] = d
    return out


def _drop_tool(doc: Any, name: str) -> Any:
    if isinstance(doc, dict):
        out = {}
        for k, v in doc.items():
            if k == name and isinstance(v, dict) and "automaton" in v:
                continue
            out[k] = _drop_tool(v, name)
        return out
    if isinstance(doc, list):
        return [
            _drop_tool(v, name)
            for v in doc
            if not (isinstance(v, dict) and "automaton" in v and v.get("name") == name)
        ]
    return doc


@pytest.mark.parametrize(
    "variant", ["tools_sha", "automaton", "automaton_sha256", "missing_tool", "bucket_256"]
)
def test_c1_a_stale_or_off_gate_payload_file_raises_payload_stale(
    tmp_path: Path, variant: str
) -> None:
    """Criterion 1: a tools sha other than ``tools.load().version``, an automaton whose sha256 is not its recorded
    ``automaton_sha256`` (either side edited), a v1 tool without a payload, or a bucket above 128 -> ``PayloadStale``
    (T-0092 disables repair on exactly that exception, so no other type will do)."""
    load = S.fn("load_payloads", "criterion 1 (load check)")
    stale = S.fn("PayloadStale", "criterion 1 (PayloadStale)")
    doc = _stale_variants()[variant]
    if variant == "missing_tool":
        assert "scroll" not in S.tool_entries(doc), "premise: scroll removed"
    with pytest.raises(stale):
        load(_write(tmp_path, doc))


# =========================================================================== criterion 2: classify

#: (tool_text, status, name, arguments or ..., reason or ...); ``...`` = not pinned by the task.
CLASSIFY_ROWS: dict[str, tuple[str, str, str | None, Any, Any]] = {
    "click_valid": (
        call("click_at", [("x", "488"), ("y", "517")]),
        "valid",
        "click_at",
        {"x": 488, "y": 517},
        ...,
    ),
    "type_valid": (
        call("type_at", [("press_enter", "true"), ("text", s("hello")), ("x", "10"), ("y", "20")]),
        "valid",
        "type_at",
        {"press_enter": True, "text": "hello", "x": 10, "y": 20},
        ...,
    ),
    "scroll_valid": (
        call("scroll", [("direction", s("down"))]),
        "valid",
        "scroll",
        {"direction": "down"},
        ...,
    ),
    "key_valid": (
        call("press_key", [("combo", s("Ctrl+Z"))]),
        "valid",
        "press_key",
        {"combo": "Ctrl+Z"},
        ...,
    ),
    "search_valid": (
        call("google_search", [("query", s("weather in Lisbon"))]),
        "valid",
        "google_search",
        {"query": "weather in Lisbon"},
        ...,
    ),
    # criterion 2 inputs: a trailing stop text, if any, is removed (a no-op on the live path, amendment)
    "valid_then_tool_response": (
        call("scroll", [("direction", s("up"))]) + "<|tool_response>",
        "valid",
        "scroll",
        {"direction": "up"},
        ...,
    ),
    "valid_then_turn_close": (
        call("scroll", [("direction", s("up"))]) + "<turn|>",
        "valid",
        "scroll",
        {"direction": "up"},
        ...,
    ),
    "valid_then_eos": (
        call("scroll", [("direction", s("up"))]) + "<eos>",
        "valid",
        "scroll",
        {"direction": "up"},
        ...,
    ),
    # T-0100 (H-T99e) rewrote this row: until 0756893 it pinned "x > 999 parses, schema-invalid -> repairable"; an
    # out-of-range coordinate is now refused at detection, name and arguments kept
    "click_x_1200": (
        call("click_at", [("x", "1200"), ("y", "5")]),
        "refused",
        "click_at",
        {"x": 1200, "y": 5},
        "coordinate out of range",
    ),
    # adversarial: parses, schema-invalid (an extra key, coordinates in range) -> repairable (not refused)
    "click_extra_key": (
        call("click_at", [("x", "12"), ("y", "5"), ("button", s("left"))]),
        "repairable",
        "click_at",
        {"x": 12, "y": 5, "button": "left"},
        "schema invalid",
    ),
    # adversarial: the string delimiters missing -> not valid
    "search_no_delims": (
        OPEN + "call:google_search{query:weather in Lisbon}" + CLOSE,
        "repairable",
        "google_search",
        ...,
        ...,
    ),
    "search_one_delim": (
        OPEN + 'call:google_search{query:<|"|>weather in Lisbon}' + CLOSE,
        "repairable",
        "google_search",
        ...,
        ...,
    ),
    "search_json_quotes": (
        OPEN + 'call:google_search{query:"weather"}' + CLOSE,
        "repairable",
        "google_search",
        ...,
        ...,
    ),
    "length_cut_no_close": (OPEN + "call:click_at{x:488,y:5", "repairable", "click_at", ..., ...),
    # T-0102 (H-T101c) premise rewrite: until 5c5177a this row was ``... + " ok"``; prose after a closed call is now
    # refused (``test_h101c_closed_call_then_prose_refused``), so the row keeps "text after call, repairable" with a
    # remainder that holds no model text (a stray channel marker)
    "text_after_call": (
        call("click_at", [("x", "1"), ("y", "2")]) + "<channel|>",
        "repairable",
        "click_at",
        ...,
        ...,
    ),
    "type_missing_required": (
        call("type_at", [("text", s("hi")), ("x", "1"), ("y", "2")]),
        "repairable",
        "type_at",
        ...,
        ...,
    ),
    "scroll_bad_enum": (
        call("scroll", [("direction", s("sideways"))]),
        "repairable",
        "scroll",
        ...,
        ...,
    ),
    # adversarial: a parsed call naming no v1 tool -> refused("no tool name")
    "navigate": (
        call("navigate", [("url", s("https://example.com"))]),
        "refused",
        ...,
        ...,
        "no tool name",
    ),
    "wrong_case_name": (
        call("Click_At", [("x", "1"), ("y", "2")]),
        "refused",
        ...,
        ...,
        "no tool name",
    ),
    # amendment: a tool commit may start with <cc> or <genui>; refused with the parse reason (H6)
    "cc_json_call": (
        '<cc>{"name": "click_at", "arguments": {"x": 1, "y": 2}}',
        "refused",
        ...,
        ...,
        "no call prefix",
    ),
    "genui_payload": (
        '<genui>{"name": "click_at", "arguments": {"x": 1, "y": 2}}',
        "refused",
        ...,
        ...,
        "no call prefix",
    ),
    "tag_only": (OPEN, "refused", ..., ..., "empty"),
    "no_call_prefix": (OPEN + "click_at{x:1,y:2}" + CLOSE, "refused", ..., ..., "no call prefix"),
    "nameless_unparsed": (OPEN + "call:{x:1}", "refused", ..., ..., "no close"),
}


@pytest.mark.parametrize("row", list(CLASSIFY_ROWS))
def test_c2_classify_hand_rows(row: str) -> None:
    """Criterion 2: valid iff ``native.parse_call`` succeeds and ``score.schema_valid`` holds; repairable iff not
    valid and ``native.draft_name`` is a v1 name; refused otherwise, with the parse reason or ``"no tool name"``. A
    leading ``<|tool_call>`` is removed; a ``<cc>``/``<genui>`` call is refused (amendment, H6)."""
    text, status, name, arguments, reason = CLASSIFY_ROWS[row]
    v = S.fn("classify", "criterion 2 (classify)")(text)
    assert v.status == status, (row, v.status, v.reason)
    if name is not ...:
        assert v.name == name, (row, v.name)
    if arguments is not ...:
        assert v.arguments == arguments, (row, v.arguments)
    if reason is not ...:
        assert v.reason == reason, (row, v.reason)


def test_c2_premise_the_hand_rows_mean_what_they_say() -> None:
    """Self-check of the table against the reused parser (the rule's inputs, not the rule): every repairable and
    refused row really fails ``native.parse_call`` or the schema, and the reason column is the parser's own -- except
    ``"no tool name"``, ``"schema invalid"`` and T-0100's three reasons, which are the judge's, not the parser's."""
    ts = tools.load()
    from local_astra.toolcall import score

    for row, (text, status, _n, _a, reason) in CLASSIFY_ROWS.items():
        body = text.removeprefix(OPEN)
        for stop in ("<|tool_response>", "<turn|>", "<eos>"):
            body = body.removesuffix(stop)
        parsed = native.parse_call(body)
        ok = not isinstance(parsed, score.ParseFailure) and score.schema_valid(parsed, ts)
        assert ok == (status == "valid"), row
        if reason is not ... and reason not in ("no tool name", "schema invalid", *H99_REASONS):
            assert isinstance(parsed, score.ParseFailure) and parsed.reason == reason, row


# =========================================================================== criterion 3: query_problem


@pytest.mark.parametrize(
    ("text", "status", "problem"),
    [
        (call("google_search", [("query", s("weather in Lisbon tomorrow"))]), "valid", None),
        (
            call("google_search", [("query", s("I will search for the weather in Lisbon"))]),
            "valid",
            "narration",
        ),
        (call("google_search", [("query", s("click_at the blue button"))]), "valid", "tool syntax"),
        (call("click_at", [("x", "1"), ("y", "2")]), "valid", None),
        (
            call(
                "type_at",
                [
                    ("press_enter", "false"),
                    ("text", s("I'll search for cats")),
                    ("x", "1"),
                    ("y", "2"),
                ],
            ),
            "valid",
            None,
        ),
    ],
    ids=["plain", "narration", "tool_syntax", "click_has_none", "type_text_is_not_a_query"],
)
def test_c3_query_problem_is_google_search_only_and_log_only(
    text: str, status: str, problem: str | None
) -> None:
    """Criterion 3 (H7): ``query_problem`` is computed for google_search only (a type_at text that would read as
    narration gets ``None``) and never changes ``status`` (a narrating query stays ``valid``)."""
    v = S.fn("classify", "criterion 2 (classify)")(text)
    assert (v.status, v.query_problem) == (status, problem)


def test_c3_a_narrating_query_never_turns_a_repairable_draft_into_anything_else() -> None:
    """Criterion 3: a schema-invalid google_search (an extra key) with a narrating query is ``repairable``, as without
    the narration."""
    text = call("google_search", [("extra", "1"), ("query", s("I will search for cats"))])
    assert S.fn("classify", "criterion 2 (classify)")(text).status == "repairable"


# =========================================================================== criterion 4: repair_prompt

TOOLS_FIELD = tuple(native.tools_field(tools.load()))


def _pass1(*, thinking: bool = True) -> Prompt:
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(
                role="user",
                content=(
                    ImagePart(url="data:image/jpeg;base64,AAAA", phash="ph0"),
                    TextPart(text="Sign in"),
                ),
            ),
        ),
        prefix_len=2,
        thinking=thinking,
        tools=TOOLS_FIELD,
    )


@pytest.mark.parametrize("spoken", [CLICK01_SPOKEN, "", "Sure. ", " Sure.", "Done.\n"])
@pytest.mark.parametrize("thinking", [True, False])
def test_c4_repair_prompt_continues_pass_one_with_the_spoken_prefix_and_the_tag(
    spoken: str, thinking: bool
) -> None:
    """Criterion 4: pass 1's messages plus ``assistant(spoken_prefix + "<|tool_call>")`` as one text part,
    ``continue_final_message=True``, the same tools and the same thinking (a hardcoded thinking fails the False arm)."""
    p1 = _pass1(thinking=thinking)
    got = S.fn("repair_prompt", "criterion 4 (repair_prompt)")(p1, spoken)
    assert isinstance(got, Prompt), (spoken, got)
    assert got.messages == (
        *p1.messages,
        Message(role="assistant", content=(TextPart(text=spoken + OPEN),)),
    )
    assert got.continue_final_message is True
    assert got.tools == p1.tools and got.thinking == p1.thinking


def _refusal_reason(got: Any) -> str:
    assert not isinstance(got, Prompt), "criterion 4: the continuation was not refused"
    reason = getattr(got, "reason", None)
    assert isinstance(reason, str), f"criterion 4: a Refusal carries a reason, got {got!r}"
    return reason.removeprefix("continuation_guard: ")


_CHANNEL = ("<|channel>", "<channel|>")


@pytest.mark.parametrize("lit", SPECIAL_TOKENS)
def test_c4_every_special_literal_in_the_spoken_part_is_refused(lit: str) -> None:
    """Criterion 4: a ``SPECIAL_TOKENS`` literal in the spoken part refuses the continuation: ``template`` for the two
    channel markers (the pinned template's ``strip_thinking`` changes the content, and ``template`` wins when both
    hold), ``marker`` for the other 22 (``<|image|>``, ``<|"|>``, ``<tool_call|>`` among them; T-0085 K8)."""
    got = S.fn("repair_prompt", "criterion 4 (repair_prompt)")(_pass1(), "Sure." + lit + " ")
    assert _refusal_reason(got) == ("template" if lit in _CHANNEL else "marker")


@pytest.mark.parametrize(
    ("spoken", "reason"),
    [
        (STORED_CLICK01_SPOKEN, "template"),  # T-0085's click-01 as the 9aee66d router stored it
        ("Sure.<|tool_<channel|>call> ", "template"),  # the K3 splice, as written
        ("Sure.<|tool_call> ", "marker"),  # the K3 splice after the marker is gone
        ("Look.<|image|>", "marker"),  # K8
    ],
    ids=["click01_stored", "k3_splice_raw", "k3_spliced_tag", "image"],
)
def test_c4_the_t0085_rows(spoken: str, reason: str) -> None:
    got = S.fn("repair_prompt", "criterion 4 (repair_prompt)")(_pass1(), spoken)
    assert _refusal_reason(got) == reason


@pytest.mark.parametrize(
    "spoken",
    [
        "a < b and x |> y.",
        "Press <| then |> twice.",
        "Use <tool> tags.",
        "<cc> is fine?",
        "<|to ol_call>",
        "<channel",
    ],
)
def test_c4_guard_width_prose_that_merely_looks_like_markup_is_not_refused(spoken: str) -> None:
    """Guard too wide: nothing but a whole special literal (or a template change) refuses."""
    got = S.fn("repair_prompt", "criterion 4 (repair_prompt)")(_pass1(), spoken)
    assert isinstance(got, Prompt), (spoken, got)


# =========================================================================== criterion 5: repair_request


def _req(name: str, ids: list[int], stop: Any, tag: str = "t0091.r1") -> Any:
    return S.fn("repair_request", "criterion 5 (repair_request)")(
        name, ids, stop, tag, S.payloads(), S.calibration()
    )


@pytest.mark.parametrize("name", V1)
def test_c5_the_eval_seeded_request_for_each_tool_as_stored(name: str) -> None:
    """Criterion 5 against what the eval sent (stored dev rows, one per tool): with ``draft_ids`` = the stored seed
    without its stop and ``stop_id`` = 50 (the first stop of the pass-1 ids), the request's draft ids equal the stored
    ``seed_ids``, and every non-draft key equals the stored knobs (mode ``perstep_mar``, self-conditioning ``model``,
    the tool's automaton sha, entropy bound 1.0, confidence threshold 0.005) -- no key more, no key fewer -- plus
    ``draft_noise`` 0.0 and ``draft_start_step`` 24 (the regime's ``seed_draft``)."""
    row = S.eval_fixture()["seeded"][name]
    reg = S.eval_fixture()["regime"]["seed_draft"]
    seed = row["seed_ids"]
    assert row["stop_id"] == 50 and seed[-1] == 50, "premise: the stored seed ends in its stop"
    req = _req(name, seed[:-1], 50, "t0091.r1")
    assert S.draft_ids_of(req) == seed
    x = req.to_xargs()
    rest = {
        k: v
        for k, v in x.items()
        if k
        not in (
            "astra_cd_automaton",
            "astra_cd_draft",
            "astra_cd_draft_noise",
            "astra_cd_draft_start_step",
            "astra_cd_record_tag",
        )
    }
    assert rest == {"astra_cd_" + k: v for k, v in row["knobs"].items()}
    assert x["astra_cd_record_tag"] == "t0091.r1"
    assert (
        (x["astra_cd_draft_noise"], x["astra_cd_draft_start_step"])
        == (
            reg["draft_noise"],
            reg["draft_start_step"],
        )
        == (0.0, 24)
    )
    assert S.sha256_ascii(x["astra_cd_automaton"]) == row["knobs"]["automaton_sha256"]


def test_c5_the_automaton_is_the_payload_files() -> None:
    entries = S.tool_entries(S.payload_doc())
    req = _req("click_at", [6639, 236787], 50)
    assert req.automaton == entries["click_at"]["automaton"]
    assert req.automaton_sha256 == entries["click_at"]["automaton_sha256"]


def test_c5_the_knobs_are_perstep_mar_not_qargmax() -> None:
    """Adversarial (task: the ``perstep_mar_qargmax`` knobs mutant): entropy bound 1.0 (qargmax's file says 0.3),
    confidence threshold 0.005, mode ``perstep_mar``, self-conditioning ``model``."""
    req = _req("scroll", [1, 2, 3], 50)
    assert (req.mode, req.self_conditioning, req.entropy_bound, req.confidence_threshold) == (
        "perstep_mar",
        "model",
        1.0,
        0.005,
    )


@pytest.mark.parametrize(
    ("stop", "tail"),
    [(50, [50]), (106, [106]), (1, [1]), (None, [])],
    ids=["50", "106", "1", "None"],
)
def test_c5_the_seed_is_the_draft_ids_then_the_stop(stop: int | None, tail: list[int]) -> None:
    """Criterion 5 + amendment: ``seed = (draft_ids + ([stop_id] if stop_id is not None else []))[:256]``; None is a
    length cut (no stop in the seed). Adversarial: the 'drop the stop' mutant fails the 50/106/1 rows."""
    ids = [6639, 236787, 4513, 236779, 502]
    assert S.draft_ids_of(_req("click_at", ids, stop)) == ids + tail


@pytest.mark.parametrize("bad", [0, 2, 49, 107, -1, 262143, "50"])
def test_c5_any_other_stop_id_raises_value_error(bad: Any) -> None:
    """Amendment: ``stop_id`` in {1, 106, 50} or None; anything else raises ``ValueError``."""
    with pytest.raises(ValueError):
        _req("click_at", [6639, 236787], bad)


@pytest.mark.parametrize(
    ("n", "stop", "want_len", "ends_in_stop"),
    [(255, 50, 256, True), (256, 50, 256, False), (300, None, 256, False), (300, 106, 256, False)],
)
def test_c5_the_seed_is_cut_to_the_canvas_after_the_stop_is_added(
    n: int, stop: int | None, want_len: int, ends_in_stop: bool
) -> None:
    """Criterion 5: the cut to 256 comes after the stop is appended (255 ids keep their stop, 256 lose it)."""
    ids = [1000 + i for i in range(n)]
    got = S.draft_ids_of(_req("type_at", ids, stop))
    assert len(got) == want_len
    assert got == (ids + ([stop] if stop is not None else []))[:256]
    assert (got[-1] == stop) is ends_in_stop


def test_c5_the_draft_ids_are_used_verbatim_including_trailing_whitespace_ids() -> None:
    """Criterion 5 (purity; the 'rstrip after encoding' mutant as far as ``repair_request`` can see it): the ids the
    session got from ``/tokenize`` are the seed's head unchanged -- trailing ``\\n`` (107) and space (236743) ids are not
    stripped, and nothing is re-tokenised (arbitrary ids pass through)."""
    for ids in ([6639, 236787, 107], [6639, 236787, 236743], [7, 7, 7]):
        assert S.draft_ids_of(_req("press_key", ids, 50)) == [*ids, 50]


def test_c5_the_draft_ids_list_is_not_mutated() -> None:
    ids = [6639, 236787]
    _req("click_at", ids, 50)
    assert ids == [6639, 236787]


def test_c5_a_name_without_a_payload_is_refused() -> None:
    """Criterion 5 (the payload is the named tool's): a non-v1 name has no payload and raises (the task does not
    name the type; ``KeyError`` or ``ValueError``)."""
    with pytest.raises((KeyError, ValueError)):
        _req("navigate", [1, 2], 50)


def test_c5_the_shipped_calibration_file_is_accepted() -> None:
    load = S.fn("load_calibration", "criterion 5 (calibration, sha 4084026c… checked)")
    assert load(S.CALIBRATION) is not None


def _bad_calibration(tmp_path: Path, variant: str) -> Path:
    if variant == "qargmax":
        return S.QARGMAX_CALIBRATION  # a frozen file with another sha (entropy bound 0.3)
    if variant == "edited":
        p = tmp_path / "calibration-bfcl.json"
        p.write_bytes(S.CALIBRATION.read_bytes() + b"\n")
        return p
    if variant == "missing":
        return tmp_path / "no-such-calibration-bfcl.json"
    d = tmp_path / "calibration-bfcl.json"  # unreadable as a file: a directory
    d.mkdir()
    return d


@pytest.mark.parametrize("variant", ["qargmax", "edited", "missing", "unreadable"])
def test_c5_a_bad_calibration_file_raises_payload_stale(tmp_path: Path, variant: str) -> None:
    """Criterion 5 + ruling F10: a calibration file whose sha is not ``4084026c…`` (the qargmax file, a byte-edited
    copy), a missing file or an unreadable one raises ``PayloadStale`` -- not ``ValueError``/``OSError`` -- so that
    T-0092 turns repair off and publishes it instead of crashing the session. Any other exception fails the row on
    a named assertion."""
    load = S.fn("load_calibration", "criterion 5 (calibration, sha 4084026c… checked)")
    stale = S.fn("PayloadStale", "criterion 1 (PayloadStale)")
    path = _bad_calibration(tmp_path, variant)
    try:
        load(path)
    except Exception as exc:  # the type is the assertion
        assert isinstance(exc, stale), (
            f"F10: load_calibration({variant}) raised {type(exc).__name__}: {exc}, not PayloadStale"
        )
    else:
        raise AssertionError(f"F10: load_calibration({variant}) accepted a bad calibration file")


def test_c5_the_repair_options_are_max_tokens_256() -> None:
    assert S.fn("REPAIR_OPTIONS", "criterion 5 (repair options)") == LlmOptions(max_tokens=256)


def test_c5_live_never_tokenizes() -> None:
    """Criterion 5 (purity): ``live.py`` calls no ``tokenize`` and reaches no tokenizer (the session tokenises through
    ``LlmBackend.tokenize``). Code only: names and attributes, not prose."""
    path = S.REPO / "local_astra" / "toolcall" / "live.py"
    assert path.is_file(), "T-0091: local_astra/toolcall/live.py does not exist yet"
    tree = ast.parse(path.read_text())
    hits = [
        n.lineno
        for n in ast.walk(tree)
        if (isinstance(n, ast.Name) and n.id == "tokenize")
        or (isinstance(n, ast.Attribute) and n.attr == "tokenize")
    ]
    assert hits == [], f"live.py references tokenize at lines {hits}"


# =========================================================================== criterion 6: repair_verdict

_CALL = "call:click_at{x:10,y:20}" + CLOSE


@pytest.mark.parametrize(
    ("text", "status"),
    [
        (OPEN + _CALL, "valid"),  # T-0088 E4: a continuation's tool commit starts with the tag
        (_CALL, "valid"),  # the amendment: removing the tag is a no-op when it is absent
        (OPEN + "<|channel>thought\nok<channel|>" + _CALL, "valid"),  # a closed thought is removed
        ("<|channel>thought\nok<channel|>" + OPEN + _CALL, "valid"),
        (OPEN + "<|channel>thought\nnever closed", "refused"),  # an unclosed thought leaves nothing
        (
            call("click_at", [("x", "1200"), ("y", "5")]),
            "refused",
        ),  # repairable -> refused, never a second repair
        (OPEN + "call:click_at{x:488,y:5", "refused"),
        (call("navigate", [("url", s("x"))]), "refused"),
    ],
    ids=[
        "tagged",
        "untagged",
        "thought_after_tag",
        "thought_before_tag",
        "unclosed",
        "schema",
        "cut",
        "nameless",
    ],
)
def test_c6_repair_verdict(text: str, status: str) -> None:
    """Criterion 6: the repair output is classified with closed thoughts removed; a ``repairable`` result becomes
    ``refused``."""
    v = S.fn("repair_verdict", "criterion 6 (repair_verdict)")(text)
    assert v.status == status, (text, v.status, v.reason)
    if status == "valid":
        assert (v.name, v.arguments) == ("click_at", {"x": 10, "y": 20})


# =========================================================================== criterion 7: contracts and protocol


def test_c7_the_contracts_have_the_named_fields() -> None:
    c = S.contracts()
    fields = {
        "FrameGeometry": {"phash", "source", "width", "height"},
        "ToolInvocation": {"call_id", "turn_id", "name", "arguments", "source", "frame"},
        "ToolOutcome": {"call_id", "status", "action", "reason", "resolved"},
    }
    for cls, want in fields.items():
        model = S.need(c, cls, "criterion 7 (contracts)")
        assert set(model.model_fields) == want, (cls, sorted(model.model_fields))


@pytest.mark.parametrize("status", ["dry_run", "executed", "refused", "failed"])
def test_c7_tool_outcome_admits_the_four_statuses(status: str) -> None:
    outcome = S.need(S.contracts(), "ToolOutcome", "criterion 7 (contracts)")
    o = outcome(call_id="c1", status=status, action="a", reason=None, resolved={})
    assert o.status == status


def test_c7_tool_outcome_refuses_another_status() -> None:
    outcome = S.need(S.contracts(), "ToolOutcome", "criterion 7 (contracts)")
    with pytest.raises(ValidationError):
        outcome(call_id="c1", status="live", action="a", reason=None, resolved={})


def test_c7_the_tool_executor_protocol_is_runtime_checkable_with_mode_and_async_execute() -> None:
    proto = S.protocol()
    assert getattr(proto, "_is_runtime_protocol", False), "ToolExecutor must be runtime_checkable"
    assert "mode" in vars(proto) or "mode" in getattr(proto, "__annotations__", {}), (
        "ToolExecutor declares `mode`"
    )
    assert inspect.iscoroutinefunction(proto.execute)
    ex = S.dry_run()
    assert isinstance(ex, proto)
    assert inspect.iscoroutinefunction(ex.execute)


def test_c7_the_dry_run_executor_mode_is_dry_run() -> None:
    assert S.dry_run().mode == "dry_run"


# =========================================================================== criterion 7: the dry run

#: ``x_px = round(x * (width - 1) / 999)`` by hand (Fraction): 998*1279/999 = 1277.72 -> 1278; 500*1279/999 = 640.14
#: -> 640; 500*1278/999 = 639.64 -> 640; 998*1278/999 = 1276.72 -> 1277. The 998 rows separate a ``x*w/1000`` mutant
#: (1277 / 1276) and a truncating one; 500 at width 1279 separates truncation (639).
X_ROWS = [
    (1280, 0, 0),
    (1280, 999, 1279),
    (1280, 500, 640),
    (1280, 998, 1278),
    (1279, 0, 0),
    (1279, 999, 1278),
    (1279, 500, 640),
    (1279, 998, 1277),
]
#: the same for y: 500*719/999 = 359.86 -> 360; 500*718/999 = 359.36 -> 359 (``y*h/1000`` gives 360);
#: 998*719/999 = 718.28 -> 718 (``y*h/1000`` gives 719).
Y_ROWS = [
    (720, 0, 0),
    (720, 999, 719),
    (720, 500, 360),
    (720, 998, 718),
    (719, 500, 359),
    (719, 999, 718),
]


@pytest.mark.parametrize(("width", "x", "x_px"), X_ROWS)
@pytest.mark.parametrize("name", ["click_at", "type_at"])
def test_c7_x_maps_to_frame_pixels(name: str, width: int, x: int, x_px: int) -> None:
    args: dict[str, Any] = {"x": x, "y": 0}
    if name == "type_at":
        args |= {"text": "hi", "press_enter": False}
    out = S.run(S.dry_run().execute(S.invocation(name, args, frame=S.geometry(width, 720))))
    assert out.status == "dry_run", (out.status, out.reason)
    assert (out.resolved["x_px"], out.resolved["y_px"]) == (x_px, 0)


@pytest.mark.parametrize(("height", "y", "y_px"), Y_ROWS)
def test_c7_y_maps_to_frame_pixels(height: int, y: int, y_px: int) -> None:
    out = S.run(
        S.dry_run().execute(
            S.invocation("click_at", {"x": 0, "y": y}, frame=S.geometry(1280, height))
        )
    )
    assert out.resolved["y_px"] == y_px


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("click_at", {"x": 1, "y": 2}),
        ("type_at", {"x": 1, "y": 2, "text": "hi", "press_enter": True}),
    ],
)
def test_c7_click_and_type_without_a_frame_are_refused_no_frame(
    name: str, args: dict[str, Any]
) -> None:
    out = S.run(S.dry_run().execute(S.invocation(name, args, frame=None, call_id="c9")))
    assert (out.call_id, out.status, out.reason) == ("c9", "refused", "no frame")


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("google_search", {"query": "weather in Lisbon"}),
        ("scroll", {"direction": "down"}),
        ("press_key", {"combo": "Ctrl+Z"}),
    ],
)
def test_c7_guard_width_frameless_tools_are_not_refused_for_want_of_a_frame(
    name: str, args: dict[str, Any]
) -> None:
    """Guard too wide: ``no frame`` is for click and type only."""
    out = S.run(S.dry_run().execute(S.invocation(name, args, frame=None)))
    assert out.status == "dry_run", (out.status, out.reason)


def test_c7_google_search_would_search_with_no_url() -> None:
    """Criterion 7 (H2): ``action == "would search for: <query>"``; no URL anywhere in the outcome."""
    q = "weather in Lisbon tomorrow"
    out = S.run(
        S.dry_run().execute(
            S.invocation("google_search", {"query": q}, frame=S.geometry(1280, 720))
        )
    )
    assert (out.status, out.action) == ("dry_run", "would search for: " + q)
    dumped = json.dumps(out.model_dump(mode="json")).lower()
    assert not any(u in dumped for u in ("http", "www.", "google.com", "://")), dumped


@pytest.mark.parametrize(
    ("name", "args", "word"),
    [("scroll", {"direction": "left"}, "left"), ("press_key", {"combo": "Alt+Tab"}, "Alt+Tab")],
)
def test_c7_scroll_and_press_key_are_described(name: str, args: dict[str, Any], word: str) -> None:
    out = S.run(S.dry_run().execute(S.invocation(name, args)))
    assert out.status == "dry_run" and word in out.action, out.action


_V1_CALLS = st.one_of(
    st.builds(
        lambda x, y: ("click_at", {"x": x, "y": y}), st.integers(0, 999), st.integers(0, 999)
    ),
    st.builds(
        lambda x, y, t, e: ("type_at", {"x": x, "y": y, "text": t, "press_enter": e}),
        st.integers(0, 999),
        st.integers(0, 999),
        st.text(min_size=1, max_size=20),
        st.booleans(),
    ),
    st.builds(lambda q: ("google_search", {"query": q}), st.text(min_size=1, max_size=40)),
    st.builds(
        lambda d: ("scroll", {"direction": d}), st.sampled_from(["up", "down", "left", "right"])
    ),
    st.builds(
        lambda c: ("press_key", {"combo": c}), st.sampled_from(["Enter", "Ctrl+C", "Alt+Tab"])
    ),
)


@settings(max_examples=150, deadline=None)
@given(
    call_=_V1_CALLS,
    frame=st.one_of(st.none(), st.tuples(st.integers(1, 4000), st.integers(1, 4000))),
    call_id=st.text(alphabet="abc123", min_size=1, max_size=6),
)
def test_c7_status_is_always_dry_run_or_refused_and_pixels_stay_in_the_frame(
    call_: tuple[str, dict[str, Any]], frame: tuple[int, int] | None, call_id: str
) -> None:
    """Criterion 7 (Hypothesis, 150 examples, every schema-valid v1 call, with and without a frame): the status is
    ``dry_run`` or ``refused``, never ``executed``/``failed``; the call id comes back; a resolved pixel lies inside
    the frame (0 <= x_px < width)."""
    name, args = call_
    geo = S.geometry(*frame) if frame else None
    out = S.run(S.dry_run().execute(S.invocation(name, args, frame=geo, call_id=call_id)))
    assert out.status in ("dry_run", "refused") and out.call_id == call_id
    if name in ("click_at", "type_at") and frame:
        assert out.status == "dry_run"
        assert 0 <= out.resolved["x_px"] < frame[0] and 0 <= out.resolved["y_px"] < frame[1]


# =========================================================================== criterion 8: scope, against bd2c124


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=S.REPO, capture_output=True, text=True)


def _base_or_skip() -> str:
    if _git("cat-file", "-e", S.BASE_SHA + "^{commit}").returncode != 0:
        pytest.skip(f"the T-0091 base {S.BASE_SHA} is not in this clone")
    return S.BASE_SHA


UNCHANGED = (
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/tools.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/detok.py",
    "local_astra/toolcall/special_tokens.py",
    "local_astra/eval/cd/twopass.py",
    "local_astra/decoding",
    "plugin",
)
#: T-0091's merge commit 2244cd4; BASE = ``git merge-base 2244cd4^1 2244cd4^2`` (= the literal base ``S.BASE_SHA``,
#: bd2c124), TIP = ``2244cd4^2`` (T-0094b criterion 2). The rows below compare these two commits, never the working
#: tree; the uncommitted-changes half of the old guard is retired (it only meant something while the branch was open).
T0091_MERGE = "2244cd4"
T0091_BASE = "bd2c124"
T0091_TIP = "19df2f2"
#: Power pair of the scope row (T-0094b ruling P2): T-0088a's fbc442f (parent 5b416a6) added ``detok.py``.
SCOPE_PRIOR_PARENT = "5b416a6"
SCOPE_PRIOR = "fbc442f"
#: Power pair of the protocols row: T-0088's 1351df4 (parent 681c77f) changed the ``LlmBackend`` protocol.
PROTOCOLS_PRIOR_PARENT = "681c77f"
PROTOCOLS_PRIOR = "1351df4"
PROTOCOLS = "local_astra/contracts/protocols.py"
#: The allowance of the owned file ``contracts/protocols.py``: the one new class T-0091 may add.
PROTOCOLS_ALLOWED_NEW = frozenset({"ToolExecutor"})


def test_c8_the_unchanged_files_are_unchanged_since_the_base() -> None:
    """Criterion 8 (+ amendment): router, native, score, tools, tools_v1, detok, special_tokens, twopass, decoding and
    the plugin do not differ between T-0091's BASE (bd2c124, the literal base, never ``main``) and its TIP."""
    assert T0091_BASE == S.BASE_SHA
    gr.prove_range(T0091_BASE, T0091_TIP, T0091_MERGE)
    assert gr.changed(T0091_BASE, T0091_TIP, *UNCHANGED) == []


def test_the_c8_scope_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over fbc442f's parent and fbc442f the same predicate over ``UNCHANGED``
    reports ``detok.py``, reached through the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(SCOPE_PRIOR_PARENT, SCOPE_PRIOR)
    assert gr.is_ancestor(SCOPE_PRIOR, T0091_BASE)
    detok = next(p for p in UNCHANGED if p.endswith("/detok.py"))
    assert detok in gr.changed(SCOPE_PRIOR_PARENT, SCOPE_PRIOR, *UNCHANGED)


def test_c8_premise_the_diff_helper_sees_a_change() -> None:
    """Guard absent: the helper reports a file that did change (``llm_client.py`` between T-0088's base 9669258 and
    bd2c124), so an empty list above means unchanged, not unobserved."""
    _base_or_skip()
    if _git("cat-file", "-e", "9669258^{commit}").returncode != 0:
        pytest.skip("9669258 is not in this clone")
    out = _git("diff", "--name-only", "9669258", S.BASE_SHA, "--", "local_astra/conductor")
    assert "local_astra/conductor/llm_client.py" in out.stdout.split()


def _classes(src: str) -> dict[str, str]:
    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.ClassDef)}


def _protocol_changes(
    base: str, tip: str, allowed_new: frozenset[str] = PROTOCOLS_ALLOWED_NEW
) -> tuple[list[str], list[str]]:
    """The owned-file predicate over two blobs of ``protocols.py``: (classes at ``base`` whose AST differs or that are
    gone at ``tip``, classes new at ``tip`` outside ``allowed_new``)."""
    old = _classes(_git_blob(base, PROTOCOLS))
    new = _classes(_git_blob(tip, PROTOCOLS))
    return sorted(k for k, v in old.items() if new.get(k) != v), sorted(
        set(new) - set(old) - allowed_new
    )


def _git_blob(sha: str, rel: str) -> str:
    out = _git("show", f"{sha}:{rel}")
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_c8_protocols_changes_only_by_the_tool_executor_protocol() -> None:
    """Owned files: ``contracts/protocols.py`` (the ``ToolExecutor`` protocol only). Between T-0091's BASE and TIP blobs
    (T-0094b, never the live file) every base class is AST-identical, and the only new class is ``ToolExecutor``."""
    gr.prove_range(T0091_BASE, T0091_TIP, T0091_MERGE)
    base_rev, tip_rev = T0091_BASE, T0091_TIP
    assert _git_blob(base_rev, PROTOCOLS) != _git_blob(tip_rev, PROTOCOLS), "premise: two blobs"
    changed, extra = _protocol_changes(base_rev, tip_rev)
    assert changed == [], "a base protocol changed"
    assert extra == [], extra
    new = _classes(_git_blob(T0091_TIP, PROTOCOLS))
    assert set(new) >= PROTOCOLS_ALLOWED_NEW, "the allowed protocol was never added"


def test_the_c8_protocols_allowance_is_not_too_strict() -> None:
    """T-0094b criterion 6, too strict: with the allowance emptied the predicate over T-0091's own pair reports exactly
    ``ToolExecutor``; a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(T0091_BASE, T0091_TIP, T0091_MERGE)
    assert _protocol_changes(T0091_BASE, T0091_TIP, frozenset()) == ([], ["ToolExecutor"])


def test_the_c8_protocols_predicate_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over 1351df4's parent and 1351df4 (T-0088) the same predicate reports the
    changed base protocol ``LlmBackend``; a predicate that is always empty fails here."""
    gr.require_commits(PROTOCOLS_PRIOR_PARENT, PROTOCOLS_PRIOR)
    assert gr.is_ancestor(PROTOCOLS_PRIOR, T0091_BASE)
    assert _protocol_changes(PROTOCOLS_PRIOR_PARENT, PROTOCOLS_PRIOR) == (["LlmBackend"], [])


# =========================================================================== T-0100: refuse instead of repair
#
# H-T99d (press_key whose combo does not map into the list), H-T99e (click_at/type_at coordinate outside 0-999), and
# the rulings Q1-Q4 (docs/tasks/T-0100.md). Every expected (status, reason) below is written out by hand from the task's
# Definitions; ``tools_v1.json`` is read only for the 24 list entries. The "unchanged" rows compare with the judge as it
# was at T-0100's literal base ``0756893`` (loaded from git, never the working tree), the oracle criteria 7 and 8 name.
# Rows that held before T-0100 (aliases, in-range repairs, other tools) are guards; the rest needed the new rules
# (activated when ``/implement T-0100`` started). The ``new`` column of a row table records which.

Q = '<|"|>'
T0100_BASE = "0756893"
STRESS_CASES = S.REPO / "tests" / "eval" / "toolcall_stress" / "cases.jsonl"
#: the a794a98 pass-1 tool commits (``logs/t0099-repair-live-a794a98/sessions``), verbatim
A794_KOUT02 = '<|tool_call>call:press_key{combo:<|"|>Ctrl+S<|"|>}<tool_call|>'
A794_KOUT05 = '<|tool_call>call:press_key{combo:<|"|>Ctrl+F<|"|>}<tool_call|>'
A794_CPIX03 = "<|tool_call>call:click_at{x:1088,y:748}<tool_call|>"
A794_CPIX04 = "<|tool_call>call:click_at{x:1120,y:168}<tool_call|>"


def key(combo_rendered: str) -> str:
    """``press_key`` with ``combo`` rendered as given (``s(...)`` for a string)."""
    return call("press_key", [("combo", combo_rendered)])


def click(x: str, y: str) -> str:
    return call("click_at", [("x", x), ("y", y)])


def _classify(text: str) -> Any:
    return S.fn("classify", "T-0100 (classify)")(text)


def _verdict(text: str) -> tuple[str, str | None]:
    v = _classify(text)
    return v.status, v.reason


@functools.cache
def _live_at(sha: str) -> ModuleType:
    """``local_astra/toolcall/live.py`` as committed at the literal ``sha``, executed as its own module (its imports
    are the current ``native``/``score``/``tools``, which neither T-0100 nor T-0102 owns)."""
    gr.require_commits(sha)
    proc = _git("show", f"{sha}:local_astra/toolcall/live.py")
    assert proc.returncode == 0, proc.stderr
    name = f"_live_at_{sha}"
    mod = ModuleType(name)
    mod.__file__ = str(S.REPO / "local_astra" / "toolcall" / "live.py")
    sys.modules[name] = mod
    try:
        exec(compile(proc.stdout, f"{sha}:local_astra/toolcall/live.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _live_at_base() -> ModuleType:
    """T-0100's oracle: the judge at its literal base ``0756893``."""
    return _live_at(T0100_BASE)


def _k_out_combos() -> list[str]:
    rows = [
        json.loads(x) for x in STRESS_CASES.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    return [r["out_combo"] for r in rows if r["category"] == "K-out"]


def test_t0100_premise_the_k_out_combos_are_the_six_the_task_names() -> None:
    """Premise of criterion 1: the stress set's K-out ``out_combo``s are F5, Ctrl+S, Ctrl+Shift+T, Alt+F4, Ctrl+F and
    Ctrl+Plus (the task's list), and none is a list entry."""
    assert sorted(_k_out_combos()) == sorted(
        ["F5", "Ctrl+S", "Ctrl+Shift+T", "Alt+F4", "Ctrl+F", "Ctrl+Plus"]
    )
    decl = tools.load().get("press_key")
    assert decl is not None
    assert not set(_k_out_combos()) & set(decl["parameters"]["properties"]["combo"]["enum"])


# --------------------------------------------------------------------------- criterion 1 (H-T99d)

OUT_OF_LIST_ROWS = {
    **{
        f"kout:{c}": (key(s(c)), c)
        for c in ["F5", "Ctrl+S", "Ctrl+Shift+T", "Alt+F4", "Ctrl+F", "Ctrl+Plus"]
    },
    "a794:kout-02:0": (A794_KOUT02, "Ctrl+S"),
    "a794:kout-05:0": (A794_KOUT05, "Ctrl+F"),
    "a794:kout-02:1": (A794_KOUT02, "Ctrl+S"),
    "a794:kout-02:2": (A794_KOUT02, "Ctrl+S"),
    "a794:kout-05:2": (A794_KOUT05, "Ctrl+F"),
    "cmd": (key(s("Cmd+C")), "Cmd+C"),
    "control_shift_t": (key(s("Control+Shift+T")), "Control+Shift+T"),
}


@pytest.mark.parametrize("row", list(OUT_OF_LIST_ROWS))
def test_h99d_out_of_enum_refused(row: str) -> None:
    """Criterion 1: a parsed ``press_key`` whose combo does not map into the list (after the alias table) is
    ``refused`` "combo not in enum", name and arguments kept."""
    text, combo = OUT_OF_LIST_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name, v.arguments) == (
        "refused",
        NOT_IN_ENUM,
        "press_key",
        {"combo": combo},
    ), (row, v)


def test_t0100_the_k_out_rows_read_the_stress_file() -> None:
    """The six ``kout:`` rows above are the stress file's ``out_combo``s, not a hand copy that could drift."""
    assert sorted(c for r, (_t, c) in OUT_OF_LIST_ROWS.items() if r.startswith("kout:")) == sorted(
        _k_out_combos()
    )


# --------------------------------------------------------------------------- criterion 2 (Q1, adversarial)

ALIAS_SPELLINGS = [
    "Esc",
    "Return",
    "Del",
    "Up",
    "down",
    "Control+C",
    "control + c",
    "ctrl+c",
    "Ctrl + C",
    "CTRL+C",
    "alt + tab",
    "pagedown",
    "Ctrl+Return",
    # whole parts only: "pageup" is PageUp squashed, never "pagearrowup" (aliases-as-substrings mutant)
    "pageup",
    "Page Up",
    "ESCAPE",
    "left",
    "Right",
]


def _list_entries() -> list[str]:
    decl = tools.load().get("press_key")
    assert decl is not None
    enum: list[str] = decl["parameters"]["properties"]["combo"]["enum"]
    return enum


@pytest.mark.parametrize("combo", ALIAS_SPELLINGS)
def test_h99d_aliases_and_spellings_stay_repairable(combo: str) -> None:
    """Criterion 2 (Q1): a spelling whose squashed, alias-mapped form is a list entry stays ``repairable`` "schema
    invalid" (the repair writes the exact string). Guard, green today; kills "exact match only" and "refuse every
    invalid press_key"."""
    assert combo not in _list_entries(), "premise: not an exact list entry"
    assert _verdict(key(s(combo))) == ("repairable", "schema invalid"), combo


def test_h99d_every_exact_list_entry_is_valid() -> None:
    """Criterion 2: the 24 exact entries (``PageUp`` included) are ``valid`` with their own string."""
    entries = _list_entries()
    assert len(entries) == 24 and "PageUp" in entries and "Ctrl+C" in entries
    bad = {
        c: (v.status, v.arguments)
        for c in entries
        if ((v := _classify(key(s(c)))).status, v.arguments) != ("valid", {"combo": c})
    }
    assert bad == {}


@pytest.mark.parametrize("combo", ["Cmd+C", "cmd+c", "Cmd+V", "Command+C", "Shift+Up", "Ctrl+Del"])
def test_h99d_cmd_is_not_an_alias(combo: str) -> None:
    """Q1: ``cmd`` (and ``command``) is not mapped, so Cmd+C is not Ctrl+C; ``Shift+Up`` maps to ``shift+arrowup`` and
    ``Ctrl+Del`` to ``ctrl+delete``, neither in the list -> refused "combo not in enum" (the "cmd mapped" mutant)."""
    v = _classify(key(s(combo)))
    assert (v.status, v.reason, v.name, v.arguments) == (
        "refused",
        NOT_IN_ENUM,
        "press_key",
        {"combo": combo},
    ), v


# --------------------------------------------------------------------------- criterion 3 (Q2)

NO_COMBO_ROWS = {
    "empty_object": OPEN + "call:press_key{}" + CLOSE,
    "unterminated": OPEN + f"call:press_key{{combo:{Q}Ctrl+",
    "cut_bare_value": OPEN + "call:press_key{combo:Ctr",
    "cut_after_colon": OPEN + "call:press_key{combo:",
    "cut_after_brace": OPEN + "call:press_key{",
    "extra_key_only": call("press_key", [("key", s("Ctrl+C"))]),
    "combo_only_inside_a_string": call("press_key", [("note", s("combo:Enter"))]),
}


@pytest.mark.parametrize("row", list(NO_COMBO_ROWS))
def test_h99d_no_recoverable_combo_refused(row: str) -> None:
    """Criterion 3 (Q2): no readable combo -- no key, an unterminated string, a bare value cut by the length limit,
    another key only (even one holding a list entry), ``combo:`` only inside a string value -> ``refused`` "no combo",
    name ``press_key`` (the "missing combo left repairable" mutant fails ``empty_object``)."""
    v = _classify(NO_COMBO_ROWS[row])
    assert (v.status, v.reason, v.name) == ("refused", NO_COMBO, "press_key"), (row, v)


MALFORMED_ROWS: dict[str, tuple[str, str, str, bool]] = {
    # (text, status, reason, new: needs T-0100)
    "no_close_out_of_list": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+S{Q}}}",
        "refused",
        NOT_IN_ENUM,
        True,
    ),
    "no_close_alias": (
        OPEN + f"call:press_key{{combo:{Q}Esc{Q}}}",
        "repairable",
        "no close",
        False,
    ),
    "text_after_out_of_list": (key(s("Ctrl+S")) + " ok", "refused", NOT_IN_ENUM, True),
    # T-0102 premise rewrite (was ``+ " ok"``): a remainder without model text keeps today's verdict
    "text_after_in_list": (
        key(s("Ctrl+C")) + "<channel|>",
        "repairable",
        "text after call",
        False,
    ),
    # first occurrence of each key only
    "duplicate_first_out": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+S{Q},combo:{Q}Ctrl+C{Q}}}" + CLOSE,
        "refused",
        NOT_IN_ENUM,
        True,
    ),
    "duplicate_first_in": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+C{Q},combo:{Q}F5{Q}}}" + CLOSE,
        "repairable",
        "duplicate key",
        False,
    ),
    # first call only
    "two_calls_first_in": (key(s("Ctrl+C")) + key(s("F5")), "repairable", "multiple calls", False),
    "two_calls_first_out": (key(s("F5")) + key(s("Ctrl+C")), "refused", NOT_IN_ENUM, True),
    # closed string values are skipped, never searched for keys (string-unaware regex mutant)
    "key_inside_a_string_skipped": (  # T-0102 premise rewrite: was ``+ CLOSE + " ok"``
        OPEN + f"call:press_key{{why:{Q},combo:F5{Q},combo:{Q}Ctrl+C{Q}}}" + CLOSE + "<channel|>",
        "repairable",
        "text after call",
        False,
    ),
    "in_list_extra_key": (
        call("press_key", [("combo", s("Ctrl+C")), ("extra", "1")]),
        "repairable",
        "schema invalid",
        False,
    ),
}


@pytest.mark.parametrize(
    "row",
    list(MALFORMED_ROWS),
)
def test_h99d_malformed_goes_by_the_recovered_combo(row: str) -> None:
    """Criterion 3: a draft that does not parse goes by the combo the value scan reads (first occurrence, first call,
    strings skipped); one that maps into the list keeps today's verdict and reason."""
    text, status, reason, _new = MALFORMED_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == (status, reason, "press_key"), (row, v)


# --------------------------------------------------------------------------- criterion 4 (Q3)

NON_STRING_ROWS: dict[str, tuple[str, Any]] = {
    # parsed: (text, arguments["combo"]) -- arguments kept
    "number": (key("5"), 5),
    "list_out": (key(f"[{Q}Ctrl+S{Q}]"), ["Ctrl+S"]),
    "list_of_an_entry": (key(f"[{Q}Ctrl+C{Q}]"), ["Ctrl+C"]),
    "null": (key("null"), None),
    "true": (key("true"), True),
    "object": (key(f"{{k:{Q}Enter{Q}}}"), {"k": "Enter"}),
}
#: bare tokens do not parse ("bare value"); arguments not pinned
BARE_ROWS = {
    "bare_out": OPEN + "call:press_key{combo:Ctrl+S}" + CLOSE,
    "bare_entry": OPEN + "call:press_key{combo:Enter}" + CLOSE,
}


@pytest.mark.parametrize("row", list(NON_STRING_ROWS))
def test_h99d_non_string_combo_refused(row: str) -> None:
    """Criterion 4 (Q3): a number, a list (even of a list entry), null, a boolean or an object is not a string ->
    ``refused`` "combo not in enum", arguments kept."""
    text, value = NON_STRING_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name, v.arguments) == (
        "refused",
        NOT_IN_ENUM,
        "press_key",
        {"combo": value},
    ), (row, v)


@pytest.mark.parametrize("row", list(BARE_ROWS))
def test_h99d_a_bare_combo_is_refused_even_when_it_spells_an_entry(row: str) -> None:
    """Criterion 4: a bare token is raw text, not a string -- ``combo:Ctrl+S}`` and ``combo:Enter}`` both refused."""
    assert _verdict(BARE_ROWS[row]) == ("refused", NOT_IN_ENUM)


# --------------------------------------------------------------------------- criterion 5 (H-T99e)

OUT_OF_RANGE_ROWS: dict[str, tuple[str, str, Any]] = {
    # (text, name, arguments; None when the draft does not parse -- ruling V2)
    "a794_cpix03": (A794_CPIX03, "click_at", {"x": 1088, "y": 748}),
    "a794_cpix04": (A794_CPIX04, "click_at", {"x": 1120, "y": 168}),
    "x_1000": (click("1000", "5"), "click_at", {"x": 1000, "y": 5}),
    "y_1000": (click("5", "1000"), "click_at", {"x": 5, "y": 1000}),
    "x_minus_1": (click("-1", "5"), "click_at", {"x": -1, "y": 5}),
    "y_minus_5": (click("5", "-5"), "click_at", {"x": 5, "y": -5}),
    "x_12_5": (click("12.5", "5"), "click_at", {"x": 12.5, "y": 5}),
    "x_1e3": (click("1e3", "5"), "click_at", {"x": 1000.0, "y": 5}),
    "x_string": (click(s("500"), "5"), "click_at", {"x": "500", "y": 5}),
    # bool is not an integer (as jsonschema counts): the "bool counted as integer" mutant
    "x_true": (click("true", "5"), "click_at", {"x": True, "y": 5}),
    "type_x_1200_else_valid": (
        call(
            "type_at", [("press_enter", "true"), ("text", s("hello")), ("x", "1200"), ("y", "20")]
        ),
        "type_at",
        {"press_enter": True, "text": "hello", "x": 1200, "y": 20},
    ),
    "no_close_x_1200": (
        OPEN + "call:click_at{x:1200,y:5}",
        "click_at",
        None,
    ),  # V2: unparsed -> arguments None
    "text_after_x_1200": (
        click("1200", "5") + " ok",
        "click_at",
        None,
    ),  # V2: unparsed -> arguments None
    "bare_raw_x": (
        OPEN + "call:click_at{x:abc,y:5}" + CLOSE,
        "click_at",
        None,
    ),  # V2: unparsed -> arguments None
    "cut_after_an_out_of_range_x": (
        OPEN + "call:click_at{x:1088,y:7",
        "click_at",
        None,
    ),  # V2: unparsed -> arguments None
}


@pytest.mark.parametrize("row", list(OUT_OF_RANGE_ROWS))
def test_h99e_out_of_range_refused(row: str) -> None:
    """Criterion 5 (H-T99e): a read-back x or y that is not an integer in 0-999 -> ``refused`` "coordinate out of range"
    (> 999, negative, non-integer float, string, bool, bare raw text), parsed or read by the value scan; name kept,
    and the parsed arguments when the draft parses."""
    text, name, arguments = OUT_OF_RANGE_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == ("refused", OUT_OF_RANGE, name), (row, v)
    assert v.arguments == arguments, (row, v.arguments)


# --------------------------------------------------------------------------- criterion 6 (edges, adversarial)

BOUNDARY_ROWS: dict[str, tuple[str, str, str | None, bool]] = {
    # (text, status, reason, new: needs T-0100) -- 999 is in range ("> 999" mutant), 1000 is not (">= 1000" off by one)
    "x_999_y_0": (click("999", "0"), "valid", None, False),
    "x_0_y_999": (click("0", "999"), "valid", None, False),
    "x_500_0": (click("500.0", "5"), "valid", None, False),
    "x_1000_y_0": (click("1000", "0"), "refused", OUT_OF_RANGE, True),
    "x_0_y_1000": (click("0", "1000"), "refused", OUT_OF_RANGE, True),
    "type_x_999": (
        call("type_at", [("press_enter", "false"), ("text", s("hi")), ("x", "999"), ("y", "999")]),
        "valid",
        None,
        False,
    ),
    "type_y_1000": (
        call("type_at", [("press_enter", "false"), ("text", s("hi")), ("x", "999"), ("y", "1000")]),
        "refused",
        OUT_OF_RANGE,
        True,
    ),
}


@pytest.mark.parametrize(
    "row",
    list(BOUNDARY_ROWS),
)
def test_h99e_boundaries_999_ok_1000_refused(row: str) -> None:
    text, status, reason, _new = BOUNDARY_ROWS[row]
    assert _verdict(text) == (status, reason), row


IN_RANGE_INVALID_ROWS: dict[str, tuple[str, str, str]] = {
    "extra_key": (
        call("click_at", [("x", "12"), ("y", "5"), ("button", s("left"))]),
        "click_at",
        "schema invalid",
    ),
    "type_missing_press_enter": (
        call("type_at", [("text", s("hi")), ("x", "1"), ("y", "2")]),
        "type_at",
        "schema invalid",
    ),
    # T-0102 premise rewrite (was ``+ " ok"``)
    "text_after_call": (click("1", "2") + "<channel|>", "click_at", "text after call"),
    # a missing coordinate does not trigger H-T99e (today's verdict stands)
    "x_missing": (OPEN + "call:click_at{y:5}" + CLOSE, "click_at", "schema invalid"),
    "y_missing": (OPEN + "call:click_at{x:12}" + CLOSE, "click_at", "schema invalid"),
    "type_unterminated_text": (
        OPEN + f"call:type_at{{x:500,y:36,text:{Q}usb-c",
        "type_at",
        "unterminated string",
    ),
    # first call only: the second call's x:1200 is never read
    "two_calls_first_in_range": (
        click("5", "5") + click("1200", "5"),
        "click_at",
        "multiple calls",
    ),
    # H-T99d is press_key's: a click with a stray out-of-list combo key is an ordinary schema repair
    "click_with_a_combo_key": (
        call("click_at", [("x", "5"), ("y", "5"), ("combo", s("F5"))]),
        "click_at",
        "schema invalid",
    ),
    # H-T99e is click/type's: a press_key in the list with a stray x:1200 is an ordinary schema repair
    "key_with_an_x_key": (
        call("press_key", [("combo", s("Enter")), ("x", "1200")]),
        "press_key",
        "schema invalid",
    ),
}


@pytest.mark.parametrize("row", list(IN_RANGE_INVALID_ROWS))
def test_h99e_in_range_invalid_stays_repairable(row: str) -> None:
    """Criterion 6: in-range click/type drafts invalid for another reason stay ``repairable`` with today's reason;
    the rules never cross tools. Guard, green today."""
    text, name, reason = IN_RANGE_INVALID_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == ("repairable", reason, name), (row, v)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            call(
                "type_at",
                [
                    ("text", s("go to x:1200")),
                    ("x", "10"),
                    ("y", "10"),
                    ("press_enter", "false"),
                    ("extra", "1"),
                ],
            ),
            "schema invalid",
        ),
        # does not parse (text after the call): the value scan must skip the string ", x:1200". T-0102 premise
        # rewrite: both rows were ``+ CLOSE + " ok"``; the remainder is now a stray channel marker (no model text)
        (
            OPEN
            + f"call:type_at{{text:{Q}, x:1200{Q},x:10,y:10,press_enter:false}}"
            + CLOSE
            + "<channel|>",
            "text after call",
        ),
        (
            OPEN + f"call:click_at{{note:{Q},y:1000{Q},x:10,y:10}}" + CLOSE + "<channel|>",
            "text after call",
        ),
    ],
    ids=["parsed_task_row", "unparsed_type", "unparsed_click"],
)
def test_h99e_string_contents_never_scanned(text: str, reason: str) -> None:
    """Criterion 6: keys inside closed string values are never read (the string-unaware regex mutant reads
    ``x:1200``/``y:1000`` first and refuses). Guard, green today."""
    assert _verdict(text) == ("repairable", reason)


@pytest.mark.parametrize(
    "text",
    [
        OPEN + "call:click_at{x:488,y:5",  # the task's row
        OPEN
        + "call:click_at{x:488,y:1000",  # reading the cut-off token would refuse (it may be 10003 or 100)
        OPEN + "call:click_at{x:4880",
        OPEN + "call:click_at{x:-",
    ],
    ids=["y_5", "y_1000", "x_4880", "x_minus"],
)
def test_h99e_length_cut_last_token_unread(text: str) -> None:
    """Criterion 6: a bare token running into the end of the body is a length cut and gives no value, so the
    coordinate is unreadable and the verdict stays ``repairable`` "no close". Guard, green today."""
    assert _verdict(text) == ("repairable", "no close")


# --------------------------------------------------------------------------- rulings V2-V5 (2026-10-15)

UNPARSED_REFUSALS = {
    "no_close_out_of_list": (MALFORMED_ROWS["no_close_out_of_list"][0], "press_key", NOT_IN_ENUM),
    "text_after_out_of_list": (
        MALFORMED_ROWS["text_after_out_of_list"][0],
        "press_key",
        NOT_IN_ENUM,
    ),
    "duplicate_first_out": (MALFORMED_ROWS["duplicate_first_out"][0], "press_key", NOT_IN_ENUM),
    "two_calls_first_out": (MALFORMED_ROWS["two_calls_first_out"][0], "press_key", NOT_IN_ENUM),
    "bare_out": (BARE_ROWS["bare_out"], "press_key", NOT_IN_ENUM),
    "unterminated": (NO_COMBO_ROWS["unterminated"], "press_key", NO_COMBO),
    "cut_bare_value": (NO_COMBO_ROWS["cut_bare_value"], "press_key", NO_COMBO),
    "no_close_x_1200": (OPEN + "call:click_at{x:1200,y:5}", "click_at", OUT_OF_RANGE),
}


@pytest.mark.parametrize("row", list(UNPARSED_REFUSALS))
def test_v2_an_unparsed_refusal_reports_no_arguments(row: str) -> None:
    """Ruling V2: for a draft that does not parse, the refused verdict keeps name and reason and its ``arguments`` is
    ``None`` -- the scanned values decide, they are never reported as arguments."""
    text, name, reason = UNPARSED_REFUSALS[row]
    v = _classify(text)
    assert (v.status, v.name, v.arguments, v.reason) == ("refused", name, None, reason), (row, v)


RULING_ROWS: dict[str, tuple[str, str, str, str]] = {
    # (text, status, reason, name)
    # V3: a JSON-quoted bare token is raw text, not a string
    "v3_json_quoted_combo": (
        OPEN + 'call:press_key{combo:"Ctrl+C"}' + CLOSE,
        "refused",
        NOT_IN_ENUM,
        "press_key",
    ),
    "v3_json_quoted_x": (
        OPEN + 'call:click_at{x:"500",y:5}' + CLOSE,
        "refused",
        OUT_OF_RANGE,
        "click_at",
    ),
    # V4: only top-level keys are read; nested objects and lists are never searched. T-0102 premise rewrite: the three
    # repairable V4 rows were ``+ CLOSE + " ok"``; the remainder is now a stray channel marker (no model text)
    "v4_nested_x_unread": (
        OPEN + "call:click_at{pos:{x:1200},x:5,y:5}" + CLOSE + "<channel|>",
        "repairable",
        "text after call",
        "click_at",
    ),
    "v4_nested_list_x_unread": (
        OPEN + "call:click_at{pts:[{x:1200}],x:5,y:5}" + CLOSE + "<channel|>",
        "repairable",
        "text after call",
        "click_at",
    ),
    "v4_nested_combo_unread": (
        OPEN
        + f"call:press_key{{opts:{{combo:{Q}F5{Q}}},combo:{Q}Ctrl+C{Q}}}"
        + CLOSE
        + "<channel|>",
        "repairable",
        "text after call",
        "press_key",
    ),
    "v4_only_nested_combo": (
        OPEN + f"call:press_key{{opts:{{combo:{Q}Ctrl+C{Q}}}}}" + CLOSE + " ok",
        "refused",
        NO_COMBO,
        "press_key",
    ),
    # V5: an empty bare token gives no value
    "v5_empty_x": (
        OPEN + "call:click_at{x:,y:5}" + CLOSE,
        "repairable",
        "malformed arguments",
        "click_at",
    ),
    "v5_empty_combo": (OPEN + "call:press_key{combo:}" + CLOSE, "refused", NO_COMBO, "press_key"),
}


@pytest.mark.parametrize("row", list(RULING_ROWS))
def test_v3_v4_v5_rulings(row: str) -> None:
    """Rulings V3 (bare literal = true/false/null/number only), V4 (top-level keys only) and V5 (an empty bare
    token is no value), on drafts that do not parse."""
    text, status, reason, name = RULING_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == (status, reason, name), (row, v)


# --------------------------------------------------------------------------- reviewer A's unpinned behaviours (89d8321)
#
# Each row names the mutant it kills (checked in memory against ``live.py`` at 9f61ce6 by the test-author): ``>= 999``
# and ``> 0`` on the bounds (every earlier 999/0 row was schema-valid, so the rule never ran), no ``depth == 1`` check
# (V4, a nested key after a comma), an unterminated string not ending the scan, no depth-0 stop (V7) and no call-tag
# stop. The reviewer's three rows marked "as given" pin the verdict but kill no mutant on their own (the token after
# the trap runs into the end of the body, or the trap sits at depth 0 where the key check already fails); the
# ``*_killer`` rows beside them add a terminator so the mutant reads a value.

SCAN_EDGE_ROWS: dict[str, tuple[str, str, str]] = {
    # (text, status, reason)
    "bound_click_999_0_extra_key": (
        OPEN + f"call:click_at{{x:999,y:0,button:{Q}left{Q}}}" + CLOSE,
        "repairable",
        "schema invalid",
    ),
    "bound_type_0_999_no_press_enter": (
        call("type_at", [("text", s("hi")), ("x", "0"), ("y", "999")]),
        "repairable",
        "schema invalid",
    ),
    # T-0102 premise rewrite: the next two rows were ``+ " ok"``
    "bound_unparsed_999_0": (click("999", "0") + "<channel|>", "repairable", "text after call"),
    "v4_nested_x_after_a_comma": (
        OPEN + "call:click_at{pos:{a:1,x:1200},x:5,y:5}" + CLOSE + "<channel|>",
        "repairable",
        "text after call",
    ),
    "v4_nested_combo_after_a_comma": (
        OPEN + f"call:press_key{{opts:{{a:1,combo:{Q}Ctrl+C{Q}}}}}" + CLOSE + " ok",
        "refused",
        NO_COMBO,
    ),
    "unterminated_type_as_given": (
        OPEN + f"call:type_at{{text:{Q}hello, x:1200",
        "repairable",
        "unterminated string",
    ),
    "unterminated_type_killer": (
        OPEN + f"call:type_at{{text:{Q}hello, x:1200,y:5",
        "repairable",
        "unterminated string",
    ),
    "unterminated_key_as_given": (
        OPEN + f"call:press_key{{note:{Q}a, combo:{Q}F5",
        "refused",
        NO_COMBO,
    ),
    "unterminated_key_killer": (
        OPEN + f"call:press_key{{note:{Q}a, combo:Ctrl+C,x",
        "refused",
        NO_COMBO,
    ),
}
#: T-0102 premise change: V7's depth-0 stop and the call-tag stop. Until 5c5177a these three rows sat in
#: ``SCAN_EDGE_ROWS`` as ``repairable`` ("no close", "no close", "malformed arguments"). What follows the call's end
#: (``,x:1200``, ``{a:1,x:1200,z``, ``,y:1200}``) is model text by the T-0102 Definitions (it holds letters and digits),
#: and no benign remainder can carry the ``x:1200`` the T-0100 mutants would read. So the rows now expect the T-0102
#: refusal (pending until ``/implement T-0102``), which still kills the T-0100 mutants -- they refuse first with
#: "coordinate out of range" (T-0100 precedence) -- and ``test_t0100_scan_stops_hold_either_side_of_t0102`` keeps that
#: kill live in the meantime.
SCAN_STOP_ROWS: dict[str, str] = {
    "v7_depth0_as_given": OPEN + "call:click_at{y:5},x:1200",
    "v7_depth0_killer": OPEN + "call:click_at{y:5}{a:1,x:1200,z",
    "call_tag_stop": OPEN + "call:click_at{x:5" + CLOSE + ",y:1200}",
}


@pytest.mark.parametrize("row", list(SCAN_EDGE_ROWS))
def test_t0100_scan_edges_reviewer_a(row: str) -> None:
    """Reviewer A (T-0100, at 89d8321): boundaries on not-valid drafts, V4 nested keys after a comma, the unterminated
    string, V7's depth-0 stop and the call-tag stop -- each pinned by hand from the Definitions and V2-V8."""
    text, status, reason = SCAN_EDGE_ROWS[row]
    assert _verdict(text) == (status, reason), row


@pytest.mark.parametrize("row", list(SCAN_STOP_ROWS))
def test_t0100_scan_stops_after_t0102(row: str) -> None:
    """The three scan-stop rows of reviewer A (T-0100) under T-0102: the scan stops at depth 0 / at the call tag (so
    ``x:1200``/``y:1200`` is never read and T-0100 does not refuse), and what follows is model text -> ``refused``
    "model text after call", name ``click_at``, arguments None."""
    v = _classify(SCAN_STOP_ROWS[row])
    assert (v.status, v.reason, v.name, v.arguments) == ("refused", H101C, "click_at", None), (
        row,
        v,
    )


@pytest.mark.parametrize("row", list(SCAN_STOP_ROWS))
def test_t0100_scan_stops_hold_either_side_of_t0102(row: str) -> None:
    """Guard (green before and after T-0102): whatever T-0102 does to these drafts, the T-0100 scan never reads the
    coordinate after the call's end, so the verdict is never "coordinate out of range" and the name is kept. Kills the
    no-depth-0-stop and no-call-tag-stop mutants of T-0100 while the row above is pending."""
    v = _classify(SCAN_STOP_ROWS[row])
    assert v.reason != OUT_OF_RANGE and v.name == "click_at", (row, v)
    assert v.status in ("repairable", "refused"), (row, v)


# --------------------------------------------------------------------------- criterion 7 (other tools, negative)

OTHER_TOOL_ROWS: dict[str, tuple[str, str, str | None]] = {
    # (text, status, reason), the reasons by hand from native.parse_call's rules
    "scroll_bad_enum": (
        call("scroll", [("direction", s("sideways"))]),
        "repairable",
        "schema invalid",
    ),
    "scroll_x_1200": (
        call("scroll", [("direction", s("down")), ("x", "1200")]),
        "repairable",
        "schema invalid",
    ),
    "scroll_combo_f5": (
        call("scroll", [("combo", s("F5")), ("direction", s("down"))]),
        "repairable",
        "schema invalid",
    ),
    "scroll_bare": (OPEN + "call:scroll{direction:sideways}" + CLOSE, "repairable", "bare value"),
    "search_x_1200": (
        call("google_search", [("query", s("cats")), ("x", "1200")]),
        "repairable",
        "schema invalid",
    ),
    "search_combo_f5": (
        call("google_search", [("combo", s("F5")), ("query", s("cats"))]),
        "repairable",
        "schema invalid",
    ),
    "search_no_delims": (
        OPEN + "call:google_search{query:weather in Lisbon}" + CLOSE,
        "repairable",
        "bare value",
    ),
    "search_one_delim": (
        OPEN + f"call:google_search{{query:{Q}weather in Lisbon}}" + CLOSE,
        "repairable",
        "unterminated string",
    ),
    "search_json_quotes": (
        OPEN + 'call:google_search{query:"weather"}' + CLOSE,
        "repairable",
        "bare value",
    ),
    "search_cut": (OPEN + "call:google_search{query:", "repairable", "no close"),
    "navigate": (call("navigate", [("url", s("https://example.com"))]), "refused", "no tool name"),
    "navigate_x_1200": (call("navigate", [("x", "1200")]), "refused", "no tool name"),
    "nameless": (OPEN + "call:{combo:F5}" + CLOSE, "refused", "bad name"),
    "valid_scroll": (call("scroll", [("direction", s("up"))]), "valid", None),
    "valid_search": (call("google_search", [("query", s("x:1200 combo:F5"))]), "valid", None),
    "valid_key": (key(s("Alt+Tab")), "valid", None),
    "valid_click": (click("488", "517"), "valid", None),
    "valid_type": (
        call("type_at", [("press_enter", "true"), ("text", s("x:1200")), ("x", "10"), ("y", "20")]),
        "valid",
        None,
    ),
}


@pytest.mark.parametrize("row", list(OTHER_TOOL_ROWS))
def test_other_tools_unchanged(row: str) -> None:
    """Criterion 7: every scroll/google_search row, every row naming none of the three tools, and the valid drafts of
    all five tools keep status and reason (by hand), and the whole verdict equals the 0756893 judge's. Guard."""
    text, status, reason = OTHER_TOOL_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason) == (status, reason), (row, v)
    assert _as_tuple(v) == _as_tuple(_live_at_base().classify(text)), row


def _as_tuple(v: Any) -> tuple[Any, ...]:
    return (v.status, v.name, v.arguments, v.reason, v.query_problem)


#: T-0196's literal base (docs/tasks/T-0196.md "Definitions": BASE is live.py at 611068bb).
T0196_BASE = "611068bb"


def t0196_doubled_split(body: str) -> tuple[str, str] | None:
    """T-0196's doubled-close shape (its Definitions), by an independent string-aware walk: ``(C, T)`` where ``C``
    ends at the first ``<tool_call|>`` outside every ``<|"|>`` string and ``T`` is non-empty, only closes and
    whitespace, with at least one close; else None. Never imports the implementation."""
    i = 0
    while True:
        close = body.find(CLOSE, i)
        quote = body.find(Q, i)
        if close >= 0 and (quote < 0 or close < quote):
            break
        if quote < 0:
            return None  # no close outside a string
        end = body.find(Q, quote + len(Q))
        if end < 0:
            return None  # an unterminated string: every later close is string content
        i = end + len(Q)
    c, t = body[: close + len(CLOSE)], body[close + len(CLOSE) :]
    if not t or CLOSE not in t or t.replace(CLOSE, "").strip() != "":
        return None
    return c, t


def t0196_single_close(text: str) -> str | None:
    """``OPEN + C`` for a doubled-close ``text`` that starts with ``OPEN`` (no stop text at its end), else None."""
    assert text.startswith(OPEN), text
    split = t0196_doubled_split(text[len(OPEN) :])
    return None if split is None else OPEN + split[0]


def t0196_delta(text: str, new: Any, tool_set: Any = None) -> bool:
    """T-0196's admitted change of ``classify`` (criteria 1-2): the 611068bb verdict on ``text`` is not ``valid``,
    ``text`` has the doubled-close shape, and ``new`` is the 611068bb verdict on the single-close text, which is
    ``valid``. ``repair_verdict`` admits no T-0196 change (criterion 3)."""
    base = _live_at(T0196_BASE)
    single = t0196_single_close(text)
    if single is None or base.classify(text, tool_set=tool_set).status == "valid":
        return False
    want = base.classify(single, tool_set=tool_set)
    return want.status == "valid" and _as_tuple(new) == _as_tuple(want)


def test_the_base_judge_is_the_one_before_t0100() -> None:
    """Premise of the base oracle: at 0756893 an out-of-list press_key and an out-of-range click were ``repairable``
    "schema invalid" (the T-0099 coercions), so comparing with it is not comparing the module with itself."""
    base = _live_at_base()
    assert _as_tuple(base.classify(A794_KOUT02))[::3] == ("repairable", "schema invalid")
    assert _as_tuple(base.classify(A794_CPIX03))[::3] == ("repairable", "schema invalid")


_FRAGMENTS = [
    "x:",
    "y:",
    "combo:",
    "direction:",
    "query:",
    "text:",
    "press_enter:",
    "1200",
    "5",
    "999",
    "1000",
    "-1",
    "12.5",
    "true",
    "null",
    Q,
    "Ctrl+S",
    "Ctrl+C",
    "Control+C",
    "Esc",
    "down",
    "F5",
    ",",
    "}",
    "{",
    "[",
    "]",
    " ",
    CLOSE,
    OPEN,
    " ok",
]
_NAMES = ["scroll", "google_search", "navigate", "Click_At", "click_at", "press_key", "type_at", ""]


@settings(max_examples=400, deadline=None)
@given(name=st.sampled_from(_NAMES), parts=st.lists(st.sampled_from(_FRAGMENTS), max_size=10))
def test_t0100_only_repairable_drafts_of_the_three_tools_can_change(
    name: str, parts: list[str]
) -> None:
    """Criteria 7 and 8 as a property (Hypothesis, 400 examples over call fragments): against the 0756893 judge,
    ``classify`` and ``repair_verdict`` differ only where the old ``classify`` said ``repairable``, the draft names
    press_key/click_at/type_at, and the new verdict is ``refused`` with one of the three reasons; for scroll,
    google_search and any other name the verdicts are identical; ``repair_verdict``'s status never changes.

    T-0102 widening (docs/tasks/T-0102.md "How to handle them"): the allowed new reason set is T-0100's three plus
    "model text after call", which (unlike the three) may refuse a draft of any of the five v1 tools. The T-0100
    reasons stay confined to the three tools. Green before and after T-0102: it is a weaker guard, so it cannot be a
    strict-xfail pending row; T-0102's own direction is ``test_h101c_only_repairable_changes``.

    T-0196 widening (docs/tasks/T-0196.md, Owned files; the T-0102 precedent): ``classify`` may also differ by exactly
    the T-0196 delta (``t0196_delta``: a doubled-close text whose single-close text 611068bb judges ``valid`` gets that
    verdict). ``repair_verdict`` admits no T-0196 change. Green before and after T-0196; T-0196's own direction is
    ``test_t0196_doubled_close.test_property_only_the_delta_moves``."""
    text = OPEN + "call:" + name + "{" + "".join(parts)
    base = _live_at_base()
    new, old = _classify(text), base.classify(text)
    if _as_tuple(new) != _as_tuple(old) and not t0196_delta(text, new):
        assert old.status == "repairable" and new.status == "refused", (text, old, new)
        assert new.reason in (*H99_REASONS, H101C), (text, new)
        if new.reason in H99_REASONS:
            assert name in ("press_key", "click_at", "type_at"), (text, old, new)
        else:
            assert name in V1 and new.arguments is None, (text, old, new)
        assert new.name == name, (text, new)
    rv_new = S.fn("repair_verdict", "criterion 6 (repair_verdict)")(text)
    rv_old = base.repair_verdict(text)
    assert rv_new.status == rv_old.status, (text, rv_old, rv_new)
    if rv_new.reason != rv_old.reason:
        assert rv_new.reason in (*H99_REASONS, H101C), (text, rv_old, rv_new)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=60))
def test_t0100_classify_is_total_on_any_string(text: str) -> None:
    """Implementation notes: the scan never raises on a ``str``; ``classify`` stays total with a known status."""
    for t in (text, OPEN + "call:press_key{" + text, OPEN + "call:click_at{" + text):
        assert _classify(t).status in ("valid", "repairable", "refused")


# --------------------------------------------------------------------------- criterion 8 (repair_verdict)

_ALL_T0100_TEXTS = [
    *(t for t, *_ in CLASSIFY_ROWS.values()),
    *(t for t, _c in OUT_OF_LIST_ROWS.values()),
    *(key(s(c)) for c in ALIAS_SPELLINGS),
    *NO_COMBO_ROWS.values(),
    *(t for t, *_ in MALFORMED_ROWS.values()),
    *(t for t, _v in NON_STRING_ROWS.values()),
    *BARE_ROWS.values(),
    *(t for t, *_ in OUT_OF_RANGE_ROWS.values()),
    *(t for t, *_ in BOUNDARY_ROWS.values()),
    *(t for t, *_ in IN_RANGE_INVALID_ROWS.values()),
    *(t for t, *_ in OTHER_TOOL_ROWS.values()),
    *(t for t, *_ in RULING_ROWS.values()),
    *(t for t, *_ in SCAN_EDGE_ROWS.values()),
    *SCAN_STOP_ROWS.values(),
]


def test_repair_verdict_statuses_unchanged() -> None:
    """Criterion 8 (negative): on every T-0100 row (bare, and behind a closed thought) ``repair_verdict``'s status is
    the 0756893 judge's; only the reason may differ, and then it is one of the three new reasons -- or, since T-0102
    (widened per its "How to handle them"), "model text after call". Guard, green before and after T-0102."""
    rv = S.fn("repair_verdict", "criterion 6 (repair_verdict)")
    base = _live_at_base()
    bad = []
    for text in _ALL_T0100_TEXTS:
        for t in (text, "<|channel>thought\nok<channel|>" + text):
            new, old = rv(t), base.repair_verdict(t)
            if new.status != old.status or (
                new.reason != old.reason and new.reason not in (*H99_REASONS, H101C)
            ):
                bad.append((t, (old.status, old.reason), (new.status, new.reason)))
    assert bad == []
    assert len(_ALL_T0100_TEXTS) > 100, "premise: the rows above are all here"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (A794_KOUT02, NOT_IN_ENUM),
        (A794_CPIX03, OUT_OF_RANGE),
        (OPEN + "call:press_key{}" + CLOSE, NO_COMBO),
        ("<|channel>thought\nok<channel|>" + key(s("F5")), NOT_IN_ENUM),
    ],
    ids=["ctrl_s", "x_1088", "no_combo", "after_a_thought"],
)
def test_repair_verdict_carries_the_h99_reasons(text: str, reason: str) -> None:
    """Definitions, "Where": the rules live in ``_judge``, which ``repair_verdict`` shares -- a repair that writes an
    out-of-list key or an out-of-range point is ``refused`` with the same reason (still never a second repair)."""
    v = S.fn("repair_verdict", "criterion 6 (repair_verdict)")(text)
    assert (v.status, v.reason) == ("refused", reason), v


# --------------------------------------------------------------------------- the stress set's own oracle


def test_out_of_enum_agrees_with_the_stress_set() -> None:
    """Across the 24 entries, the K-out ``out_combo``s and the forged ``["Ctrl+T", "ctrl+t", "PageDown", "Alt + Tab"]``,
    ``classify`` refuses a ``press_key`` exactly where the stress set's ``_out_combo_problem`` is None (its K-out
    check); the alias table only adds acceptances, so no combo the stress set calls "in the enum" is refused."""
    from tests.unit.test_toolcall_stress_set_integrity import _enum, _out_combo_problem

    enum = _enum()
    combos = [*enum, *_k_out_combos(), "Ctrl+T", "ctrl+t", "PageDown", "Alt + Tab"]
    got = {c: _classify(key(s(c))).status == "refused" for c in combos}
    want = {c: _out_combo_problem(c, enum) is None for c in combos}
    assert got == want
    assert sum(want.values()) == 6, "premise: exactly the six K-out combos are out of the enum"
    for c in ALIAS_SPELLINGS:  # the alias table only adds acceptances
        if _out_combo_problem(c, enum) is not None:
            assert _classify(key(s(c))).status != "refused", c


# --------------------------------------------------------------------------- the surface (criteria 12, Definitions)


def test_the_three_reasons_are_exported_module_constants() -> None:
    """Definitions, "Verdict shape": the three reasons are module constants exported from ``live`` (names are the
    implementer's; the values are pinned here so swapping two fails)."""
    mod = S.live()
    exported = {
        getattr(mod, n)
        for n in getattr(mod, "__all__", ())
        if isinstance(getattr(mod, n, None), str)
    }
    assert set(H99_REASONS) <= exported, sorted(exported)
    assert len(set(H99_REASONS)) == 3


def test_classify_stays_a_plain_module_function() -> None:
    """Criterion 12: the T-0099 harness checks ``live.classify.__qualname__ == "classify"`` and patches it by
    attribute. Guard."""
    f = S.fn("classify", "criterion 2 (classify)")
    assert (
        inspect.isfunction(f) and f.__qualname__ == "classify" and f.__module__ == S.live().__name__
    )


# =========================================================================== T-0102: refuse a retracted call (H-T101c)
#
# docs/tasks/T-0102.md, criteria 1-7 and 11 (the ``classify`` half), rulings Q1-Q5 (2026-10-15). Every expected
# (status, reason, name) is written out by hand from the task's Definitions: the call's end E (a string-aware depth walk
# from ``call:NAME{``), the remainder R = body[E:], "second call" iff ``R.lstrip()`` starts with ``<|tool_call>``, and
# "more model text" iff R, with every ``SPECIAL_TOKENS`` literal replaced by a space, holds a character with
# ``str.isalnum()``. Today's reasons on the rows that must *not* change are the parser's, as the task's Definitions
# ("Reasons, mapped to positions") state them. The "unchanged" property compares with ``live.py`` at T-0102's literal
# base ``5c5177a`` (``live.py`` is identical at the branch point ``3b70b0e``), loaded from git, never the working tree.
#
# Rows that hold before T-0102 are guards (unmarked: a strict-xfail pending row that already passes would XPASS);
# rows that need the new rule are ``pending(task="T-0102")``.

#: T-0102's literal base: the commit the planner surveyed (``live.py`` unchanged to ``3b70b0e``).
T0102_BASE = "5c5177a"
#: df379e4 kout-03 rep 2, pass 1's ``tool`` commit (stop id 1), verbatim from
#: ``t0101-launch/logs/t0101-warm-rerun-df379e4/sessions/20261015-201518-858015.jsonl`` (seq 37, ``t1-gen``). The
#: answer committed before it, voiced, was "I will press Ctrl Shift T to reopen the last closed tab.\n".
DF379E4_KOUT03_PASS1 = (
    '<|tool_call>call:press_key{combo:<|"|>Ctrl+T<|"|>} Wait, the tool `press_key` only takes one key or specific '
    "combo from a list. It doesn't support Ctrl+Shift+T directly. Let me re-check the list: `Enter`, `Escape`, `Tab`, "
    "`Backspace`, `Delete`, `Space`, `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`, `Home`, `End`, `PageUp`, "
    "`PageDown`, `Ctrl+A`, `Ctrl+C`, `Ctrl+V`, `Ctrl+X`, `Ctrl+Z`, `Ctrl+L`, `Ctrl+T`, `Ctrl+W`, `Alt+Tab`, "
    "`Ctrl+Enter`.\nI cannot perform Ctrl+Shift+T with the available tools. I should inform the user.\n\nActually, "
    'looking at the prompt, "Reopen the tab I just closed with Ctrl+Shift+T."\nIf I can\'t do it, I should say so.\n'
    "Wait, I should check if I can simulate it. No.\nI'll just state I can't perform that action.<channel|>I cannot "
    "perform that specific shortcut directly."
)
#: a794a98 sx-01 ("Scroll down three pages"), pass 1's ``tool`` commit (stop id 50), verbatim and identical in both
#: sessions of ``t0099-launch/logs/t0099-repair-live-a794a98/sessions``: 20261015-095257-3853950.jsonl and
#: 20261015-095429-3853950.jsonl.
A794_SX01_PASS1 = {
    "a794:sx-01:095257": (
        '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|><|tool_call>call:scroll{direction:<|"|>down'
        '<|"|>}<tool_call|><|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
    ),
    "a794:sx-01:095429": (
        '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|><|tool_call>call:scroll{direction:<|"|>down'
        '<|"|>}<tool_call|><|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
    ),
}
#: ``press_key{combo:Ctrl+T}`` with its object closed and no close tag
KEY_T = OPEN + f"call:press_key{{combo:{Q}Ctrl+T{Q}}}"


def _rv(text: str) -> Any:
    return S.fn("repair_verdict", "T-0102 (repair_verdict)")(text)


def test_h101c_premise_the_verbatim_texts() -> None:
    """Premise of criteria 1 and 5: the embedded literals are the logged commits' shape -- the retraction closes its
    object right after ``Ctrl+T`` and ends with the model's own words after a stray ``<channel|>``; each sx-01 commit is
    three identical closed scroll calls. Today (5c5177a) the first is ``repairable`` "no close", the others
    ``repairable`` "multiple calls" (the Evidence section's re-judging)."""
    assert DF379E4_KOUT03_PASS1.startswith(KEY_T + " Wait, the tool `press_key`")
    assert DF379E4_KOUT03_PASS1.endswith(
        "<channel|>I cannot perform that specific shortcut directly."
    )
    assert DF379E4_KOUT03_PASS1.count(OPEN) == 1 and CLOSE not in DF379E4_KOUT03_PASS1
    one_scroll = OPEN + f"call:scroll{{direction:{Q}down{Q}}}" + CLOSE
    assert set(A794_SX01_PASS1.values()) == {one_scroll * 3}
    base = _live_at(T0102_BASE)
    assert _as_tuple(base.classify(DF379E4_KOUT03_PASS1))[::3] == ("repairable", "no close")
    for text in A794_SX01_PASS1.values():
        assert _as_tuple(base.classify(text))[::3] == ("repairable", "multiple calls")


# --------------------------------------------------------------------------- criterion 1: the retraction


RETRACTION_ROWS = {
    "df379e4_kout03_verbatim": DF379E4_KOUT03_PASS1,
    "wait_no": KEY_T + " Wait, no.",
    # a stray delimiter after the brace: today "unterminated string", but the object closed before it
    "stray_delimiter_after_the_brace": KEY_T + f" Wait {Q} tool",
}


@pytest.mark.parametrize("row", list(RETRACTION_ROWS))
def test_h101c_retraction_refused(row: str) -> None:
    """Criterion 1 (H-T101c): a press_key whose object closed, followed by the model's own text, is ``refused`` "model
    text after call" at detection -- name ``press_key``, arguments None (the draft does not parse). ``Ctrl+T`` is a
    list entry, so T-0100 does not refuse it; today it is ``repairable`` and the repair ran Ctrl+T."""
    v = _classify(RETRACTION_ROWS[row])
    assert (v.status, v.reason, v.name, v.arguments) == ("refused", H101C, "press_key", None), (
        row,
        v,
    )


@pytest.mark.parametrize("row", list(RETRACTION_ROWS))
def test_h101c_repair_verdict_carries_the_h101c_reason(row: str) -> None:
    """Criterion 7, ``repair_verdict``: the rule lives in ``_judge`` (Definitions, "Where"); a repair output shaped like
    the retraction is ``refused`` as before, now with "model text after call" instead of the parse reason."""
    v = _rv(RETRACTION_ROWS[row])
    assert (v.status, v.reason, v.name) == ("refused", H101C, "press_key"), (row, v)


# --------------------------------------------------------------------------- criterion 2 (Q1: closed calls too)

#: (text, name), each ``refused`` "model text after call" with arguments None. Hand derivation of E and R per row.
CLOSED_THEN_PROSE_ROWS: dict[str, tuple[str, str]] = {
    # E past ``}<tool_call|>``; R = " Wait, no."
    "key_closed_then_wait": (KEY_T + CLOSE + " Wait, no.", "press_key"),
    # the task's row; R = " ok"
    "click_closed_then_ok": (click("1", "2") + " ok", "click_at"),
    # Q1 holds for all five tools (criterion 7: the rule is not T-0100's three)
    "scroll_closed_then_ok": (call("scroll", [("direction", s("down"))]) + " ok", "scroll"),
    "search_brace_then_prose": (
        OPEN + f"call:google_search{{query:{Q}cats{Q}}} Wait, no.",
        "google_search",
    ),
    "type_closed_then_prose": (
        call("type_at", [("press_enter", "true"), ("text", s("hi")), ("x", "1"), ("y", "2")])
        + " I cannot.",
        "type_at",
    ),
    # letters *or digits* (Definitions, "More model text")
    "digits_only": (KEY_T + CLOSE + " 5", "press_key"),
    # "malformed arguments" refused: the object closed and prose follows before a later close tag
    "prose_before_a_later_close": (KEY_T + " Wait" + CLOSE, "press_key"),
    # "unterminated string" refused: the object closed before the stray delimiter
    "stray_delimiter_then_close": (KEY_T + f" Wait {Q}" + CLOSE, "press_key"),
    # a close tag at depth 1 ends the call (E just after it); R = " Wait, no"
    "close_tag_inside_the_object": (OPEN + "call:click_at{x:5" + CLOSE + " Wait, no", "click_at"),
    # prose first, then a second call: R.lstrip() starts with "ok", so it is not a second call (Q2 does not cover it)
    "prose_then_a_second_call": (KEY_T + CLOSE + " ok" + KEY_T + CLOSE, "press_key"),
    # a nested object closes at depth 2, the call at depth 1; R = " Wait"
    "nested_object_then_prose": (OPEN + "call:click_at{pos:{x:1},x:5,y:5} Wait", "click_at"),
    # the model's words after a closed thought marker and a newline
    "channel_close_then_words": (KEY_T + "<channel|>\nI cannot.", "press_key"),
}


@pytest.mark.parametrize("row", list(CLOSED_THEN_PROSE_ROWS))
def test_h101c_closed_call_then_prose_refused(row: str) -> None:
    """Criterion 2 (Q1 ruled: closed calls followed by prose are refused too) and the reason-to-position map of the
    Definitions: whatever the parser's reason ("text after call", "no close", "malformed arguments", "unterminated
    string"), a call whose end E is followed by model text is ``refused`` "model text after call", name kept,
    arguments None."""
    text, name = CLOSED_THEN_PROSE_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name, v.arguments) == ("refused", H101C, name, None), (row, v)


# --------------------------------------------------------------------------- criterion 3: clean cuts (adversarial)

#: (text, today's reason, name); each stays ``repairable`` with that reason. The walk reaches the end with no E.
CLEAN_CUT_ROWS: dict[str, tuple[str, str, str]] = {
    "click_cut": (OPEN + "call:click_at{x:488,y:5", "no close", "click_at"),
    # the object closed, nothing after it: R = "" / "\n"
    "key_closed_nothing_after": (KEY_T, "no close", "press_key"),
    "key_closed_newline": (KEY_T + "\n", "no close", "press_key"),
    "search_cut_in_a_string": (
        OPEN + f"call:google_search{{query:{Q}weather",
        "unterminated string",
        "google_search",
    ),
    # D_UNCLOSED (hard-03)
    "d_unclosed_hard03": (
        OPEN + f"call:type_at{{x:500,y:36,text:{Q}usb-c",
        "unterminated string",
        "type_at",
    ),
    # Q4 (ruled): prose inside a call whose brace never closes stays repairable
    "q4_prose_inside_an_unclosed_brace": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+T{Q} Wait I cannot",
        "no close",
        "press_key",
    ),
    # a cut inside a string that holds a brace and prose: the brace is not the end (cut inside a string is clean)
    "cut_inside_a_string_with_a_brace": (
        OPEN + f"call:type_at{{x:5,y:5,text:{Q}a}} Wait, no",
        "unterminated string",
        "type_at",
    ),
    # depth: a nested object closed at depth 2 is not the call's end ("stop at the first }" mutant: R = ",y:5")
    "nested_object_then_cut": (OPEN + "call:click_at{pos:{x:1},y:5", "no close", "click_at"),
    "nested_list_then_cut": (OPEN + "call:click_at{pts:[1,2],y:5", "no close", "click_at"),
}


@pytest.mark.parametrize("row", list(CLEAN_CUT_ROWS))
def test_h101c_clean_cut_stays_repairable(row: str) -> None:
    """Criterion 3 (adversarial, the guard-too-wide direction): a draft cut off mid-call with nothing after it stays
    ``repairable`` with today's reason -- including the Q4 row (prose inside a brace that never closes) and the depth
    rows. Kills "refuse on the 'no close' reason instead of position" and "stop at the first ``}``". Guard, green
    before and after T-0102."""
    text, reason, name = CLEAN_CUT_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == ("repairable", reason, name), (row, v)
    assert _as_tuple(v) == _as_tuple(_live_at(T0102_BASE).classify(text)), row


# --------------------------------------------------------------------------- criterion 4: benign remainders (adversarial)

#: (text, today's reason, name); each stays ``repairable``: R holds only whitespace, special literals or punctuation.
BENIGN_ROWS: dict[str, tuple[str, str, str]] = {
    "brace_channel_close_newline": (KEY_T + "<channel|>\n", "no close", "press_key"),
    "close_then_channel_close": (KEY_T + CLOSE + "<channel|>", "text after call", "press_key"),
    "double_brace": (OPEN + "call:click_at{x:1,y:2}}" + CLOSE, "malformed arguments", "click_at"),
    "double_comma_typo": (
        OPEN + "call:click_at{x:1,,y:2}" + CLOSE,
        "malformed arguments",
        "click_at",
    ),
    "close_then_dot": (KEY_T + CLOSE + ".", "text after call", "press_key"),
    "close_then_bracket": (KEY_T + CLOSE + "]", "text after call", "press_key"),
    "brace_space_dot": (KEY_T + " .", "no close", "press_key"),
    # ``<turn|>`` mid-remainder (not a trailing stop text, so not dropped): a special literal, benign
    "close_then_turn_close_inside": (KEY_T + CLOSE + "\n<turn|>\n", "text after call", "press_key"),
    "close_then_delimiter": (KEY_T + CLOSE + Q, "text after call", "press_key"),
    "close_then_channel_open": (KEY_T + CLOSE + "<|channel>", "text after call", "press_key"),
    # non-alphanumeric punctuation outside ASCII is not text either (``"—".isalnum()`` is False)
    "close_then_em_dash": (KEY_T + CLOSE + " — ", "text after call", "press_key"),
    # depth: a nested object; the "first }" mutant takes R = ",y:5}<tool_call|><channel|>" (digits) and refuses
    "nested_object_then_benign": (
        OPEN + "call:click_at{pos:{x:1},y:5}" + CLOSE + "<channel|>",
        "text after call",
        "click_at",
    ),
}


@pytest.mark.parametrize("row", list(BENIGN_ROWS))
def test_h101c_benign_remainder_stays_repairable(row: str) -> None:
    """Criterion 4 (adversarial): remainders of whitespace, newlines, channel markers, ``<turn|>``, a stray ``}``, ``]``
    or ``.`` keep today's ``repairable`` verdict and reason. Kills "any non-whitespace instead of letters/digits" (the
    ``}}<tool_call|>`` and ``<channel|>`` rows) and "special literals not removed first". Guard, green before and
    after T-0102."""
    text, reason, name = BENIGN_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == ("repairable", reason, name), (row, v)
    assert _as_tuple(v) == _as_tuple(_live_at(T0102_BASE).classify(text)), row


@pytest.mark.parametrize("tail", ["  \n", "\n", " ", "\t\n "])
def test_h101c_whitespace_after_the_close_stays_valid(tail: str) -> None:
    """Criterion 4: trailing whitespace after ``<tool_call|>`` stays ``valid`` with its arguments. Guard."""
    v = _classify(KEY_T + CLOSE + tail)
    assert (v.status, v.name, v.arguments) == ("valid", "press_key", {"combo": "Ctrl+T"}), v


_BENIGN_TAILS = [
    "",
    " ",
    "\n",
    "\t",
    "<channel|>",
    "<|channel>",
    Q,
    "}",
    "]",
    ".",
    "<turn|>",
    "<eos>",
]
_CLOSED_CALLS = [
    KEY_T,
    OPEN + "call:click_at{x:1,y:2}",
    OPEN + f"call:scroll{{direction:{Q}down{Q}}}",
    OPEN + f"call:google_search{{query:{Q}a }} b{Q}}}",
    OPEN + f"call:type_at{{press_enter:true,text:{Q}hi{Q},x:1,y:2}}",
    OPEN + "call:click_at{pos:{x:1},y:2}",
]


@settings(max_examples=200, deadline=None)
@given(
    head=st.sampled_from(_CLOSED_CALLS),
    close=st.booleans(),
    tail=st.lists(st.sampled_from(_BENIGN_TAILS), max_size=6),
)
def test_h101c_benign_tails_never_change_a_verdict(head: str, close: bool, tail: list[str]) -> None:
    """Criterion 4 as a property (Hypothesis, 200 examples; the guard-too-wide direction): a closed call of any of the
    five tools, with or without its close tag, followed by any run of benign pieces (whitespace, special literals,
    ``}``, ``]``, ``.``) has exactly the 5c5177a verdict. Guard, green before and after T-0102."""
    text = head + (CLOSE if close else "") + "".join(tail)
    assert _as_tuple(_classify(text)) == _as_tuple(_live_at(T0102_BASE).classify(text)), text
    assert _rv(text).status == _live_at(T0102_BASE).repair_verdict(text).status, text


# --------------------------------------------------------------------------- criterion 4: a brace inside a string


STRING_BRACE_TYPE = OPEN + f"call:type_at{{text:{Q}a}} Wait{Q},x:5,y:5,press_enter:false"


@pytest.mark.parametrize(
    ("text", "status", "reason"),
    [
        (STRING_BRACE_TYPE, "repairable", "no close"),
        (STRING_BRACE_TYPE + "}" + CLOSE, "valid", None),
        (STRING_BRACE_TYPE + "}" + CLOSE + "<channel|>", "repairable", "text after call"),
    ],
    ids=["no_close", "closed_valid", "closed_benign"],
)
def test_h101c_brace_inside_string_is_not_the_end(
    text: str, status: str, reason: str | None
) -> None:
    """Criterion 4 (adversarial): the ``}`` and the word ``Wait`` inside ``<|"|>a} Wait<|"|>`` are string contents,
    so the call has not ended there -- the draft stays ``repairable`` "no close", and once closed it is ``valid`` with
    the string intact. Kills "walk not string-aware" (which would see ``} Wait`` and refuse). Guard."""
    v = _classify(text)
    assert (v.status, v.reason, v.name) == (status, reason, "type_at"), v
    if status == "valid":
        assert v.arguments == {"text": "a} Wait", "x": 5, "y": 5, "press_enter": False}


def test_h101c_the_walk_finds_the_end_past_a_string_brace() -> None:
    """Criterion 4's companion: past the string, the real closing brace is the end, and prose after it is refused --
    so the string-aware walk is not merely "never refuse when a string holds a brace"."""
    v = _classify(STRING_BRACE_TYPE + "} Wait, no.")
    assert (v.status, v.reason, v.name, v.arguments) == ("refused", H101C, "type_at", None), v


# --------------------------------------------------------------------------- criterion 5: a second call (Q2)

SECOND_CALL_ROWS: dict[str, tuple[str, str, str]] = {
    **{k: (t, "multiple calls", "scroll") for k, t in A794_SX01_PASS1.items()},
    # the task's row: the second call follows the brace directly (no close tag)
    "brace_then_second_call": (
        KEY_T + OPEN + f"call:press_key{{combo:{Q}Escape{Q}}}" + CLOSE,
        "malformed arguments",
        "press_key",
    ),
    # R.lstrip(): whitespace before the second tag
    "brace_space_second_call": (
        KEY_T + " " + OPEN + f"call:press_key{{combo:{Q}Escape{Q}}}" + CLOSE,
        "malformed arguments",
        "press_key",
    ),
    "closed_newline_second_call": (
        KEY_T + CLOSE + "\n" + KEY_T + CLOSE,
        "multiple calls",
        "press_key",
    ),
    # a call tag inside the object sets E at its own index; R starts with the tag
    "tag_inside_the_object": (
        OPEN + "call:click_at{x:5" + OPEN + "call:click_at{x:5,y:5}" + CLOSE,
        "malformed arguments",
        "click_at",
    ),
}


@pytest.mark.parametrize("row", list(SECOND_CALL_ROWS))
def test_h101c_second_call_keeps_todays_verdict(row: str) -> None:
    """Criterion 5 (negative, Q2 ruled): a remainder that starts (after whitespace) with ``<|tool_call>`` is a second
    call, not model text -- both a794a98 sx-01 commits verbatim stay ``repairable`` "multiple calls" (the
    said-three/did-one mismatch is a later ruling), and the brace-then-call row stays "malformed arguments". Kills
    "second call treated as prose". Guard, green before and after T-0102."""
    text, reason, name = SECOND_CALL_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name) == ("repairable", reason, name), (row, v)
    assert _as_tuple(v) == _as_tuple(_live_at(T0102_BASE).classify(text)), row


# --------------------------------------------------------------------------- criterion 6: T-0100 first (negative)

PRECEDENCE_ROWS: dict[str, tuple[str, str, str]] = {
    "ctrl_s_then_wait": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+S{Q}}} Wait",
        NOT_IN_ENUM,
        "press_key",
    ),
    "ctrl_s_closed_then_wait": (key(s("Ctrl+S")) + " Wait, no.", NOT_IN_ENUM, "press_key"),
    "click_x1200_then_text": (
        OPEN + "call:click_at{x:1200,y:5} Wait, no.",
        OUT_OF_RANGE,
        "click_at",
    ),
    "click_x1200_closed_then_text": (click("1200", "5") + " Wait, no.", OUT_OF_RANGE, "click_at"),
    "empty_key_then_wait": (OPEN + "call:press_key{} Wait", NO_COMBO, "press_key"),
    "type_y_minus_then_text": (
        OPEN + f"call:type_at{{text:{Q}hi{Q},x:5,y:-5,press_enter:true}} I cannot.",
        OUT_OF_RANGE,
        "type_at",
    ),
}


@pytest.mark.parametrize("row", list(PRECEDENCE_ROWS))
def test_h101c_t0100_reasons_take_precedence(row: str) -> None:
    """Criterion 6 (negative): a draft T-0100 refuses keeps its T-0100 reason even when model text follows the call
    (the T-0102 rule applies only to a draft whose verdict would otherwise be repairable). Kills "rule before T-0100"
    (reasons swapped). Guard, green before and after T-0102."""
    text, reason, name = PRECEDENCE_ROWS[row]
    v = _classify(text)
    assert (v.status, v.reason, v.name, v.arguments) == ("refused", reason, name, None), (row, v)


def test_h101c_the_t0100_constants_keep_their_values() -> None:
    """Criterion 6: ``NO_COMBO``, ``COMBO_NOT_IN_ENUM`` and ``COORDINATE_OUT_OF_RANGE`` (named by the task) keep their
    values. Guard."""
    mod = S.live()
    assert (mod.NO_COMBO, mod.COMBO_NOT_IN_ENUM, mod.COORDINATE_OUT_OF_RANGE) == (
        NO_COMBO,
        NOT_IN_ENUM,
        OUT_OF_RANGE,
    )


# --------------------------------------------------------------------------- criterion 7: only where H-T101c applies

_H101C_FRAGMENTS = [
    *_FRAGMENTS,
    " Wait",
    " no.",
    "I cannot",
    "\n",
    "<channel|>",
    "<|channel>",
    "<turn|>",
    ".",
    " 5",
    "x",
]
_H101C_ARGS = [
    "",
    "x:1,y:2",
    f"combo:{Q}Ctrl+T{Q}",
    f"combo:{Q}Control+C{Q}",
    f"direction:{Q}down{Q}",
    f"query:{Q}a }} b{Q}",
    f"text:{Q}hi{Q},x:1,y:2",
    "pos:{x:1},y:5",
    "pts:[1,2],y:5",
    "x:1,y:2,button:1",
]
_PROSE_WORDS = ["Wait", "ok", "I", "no", "5", "x", "Actually"]
_SEPARATORS = ["", " ", "\n", "<channel|>", " .", "\n\n"]


@settings(max_examples=300, deadline=None)
@given(
    name=st.sampled_from(_NAMES),
    parts=st.lists(st.sampled_from(_H101C_FRAGMENTS), max_size=10),
    tool=st.sampled_from(V1),
    args=st.sampled_from(_H101C_ARGS),
    close=st.booleans(),
    sep=st.sampled_from(_SEPARATORS),
    word=st.sampled_from(_PROSE_WORDS),
    tail=st.lists(st.sampled_from(_H101C_FRAGMENTS), max_size=4),
)
def test_h101c_only_repairable_changes(
    name: str,
    parts: list[str],
    tool: str,
    args: str,
    close: bool,
    sep: str,
    word: str,
    tail: list[str],
) -> None:
    """Criterion 7 as a property (Hypothesis, 300 examples), against ``live.py`` at 5c5177a.

    (a) Over call fragments with prose and marker tails: ``classify`` differs only where the old verdict was
    ``repairable`` and the new one is ``refused`` "model text after call" with the same name and arguments None; a
    ``valid`` verdict never changes; ``repair_verdict``'s status never changes and its reason may only become the new
    one. (b) The other direction, by construction (no walk in the test): a call of any of the five tools whose argument
    object is balanced and whose strings are closed, optionally closed with ``<tool_call|>``, then a separator and a
    word with a letter or digit, is ``refused`` "model text after call" wherever 5c5177a said ``repairable`` -- and
    keeps the old verdict where T-0100 refused it.

    T-0196 widening (as ``test_t0100_only_repairable_drafts_of_the_three_tools_can_change``): in (a), ``classify`` may
    also differ by exactly the T-0196 delta (``t0196_delta``); ``repair_verdict`` admits none. (b) is unchanged: a
    word follows the call, so the text never has the doubled-close shape."""
    base = _live_at(T0102_BASE)
    text = OPEN + "call:" + name + "{" + "".join(parts)
    new, old = _classify(text), base.classify(text)
    if _as_tuple(new) != _as_tuple(old) and not t0196_delta(text, new):
        assert old.status == "repairable" and new.status == "refused", (text, old, new)
        assert (new.reason, new.name, new.arguments) == (H101C, old.name, None), (text, old, new)
    rv_new, rv_old = _rv(text), base.repair_verdict(text)
    assert rv_new.status == rv_old.status, (text, rv_old, rv_new)
    if rv_new.reason != rv_old.reason:
        assert rv_new.reason == H101C, (text, rv_old, rv_new)

    built = OPEN + "call:" + tool + "{" + args + "}" + (CLOSE if close else "") + sep + word
    built += "".join(tail)
    old_b, new_b = base.classify(built), _classify(built)
    assert old_b.status != "valid", ("premise: prose after the call never parses", built)
    if old_b.status == "repairable":
        assert (new_b.status, new_b.reason, new_b.name, new_b.arguments) == (
            "refused",
            H101C,
            tool,
            None,
        ), (built, old_b, new_b)
    else:
        assert _as_tuple(new_b) == _as_tuple(old_b), (built, old_b, new_b)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=60))
def test_h101c_classify_total(text: str) -> None:
    """Implementation notes: the end-of-call walk is linear and never raises on a ``str``; ``classify`` and
    ``repair_verdict`` stay total with a known status, whatever follows a closed call or an open one. Guard."""
    for t in (
        text,
        KEY_T + text,
        KEY_T + CLOSE + text,
        OPEN + "call:type_at{text:" + Q + text,
        OPEN + "call:click_at{pos:{" + text,
    ):
        assert _classify(t).status in ("valid", "repairable", "refused"), t
        assert _rv(t).status in ("valid", "refused"), t


# --------------------------------------------------------------------------- the surface (Definitions, Q3, criterion 11)


def test_h101c_reason_is_an_exported_module_constant() -> None:
    """Definitions, "Verdict": one new module constant with the pinned value "model text after call", named in
    ``live.__all__`` (the name is the implementer's), distinct from T-0100's three."""
    mod = S.live()
    exported = {
        n: getattr(mod, n)
        for n in getattr(mod, "__all__", ())
        if isinstance(getattr(mod, n, None), str)
    }
    assert H101C in exported.values(), sorted(exported)
    assert H101C not in H99_REASONS


def test_h101c_classify_keeps_one_argument() -> None:
    """Q3 (stop id not used): ``classify`` takes the tool text and nothing about the commit -- ``Commit.stop_id`` is
    no input. Guard. Re-pinned by T-0140 (O-A1-9; A2 requires ``classify(tool_text, tool_set=None)``): exactly two
    parameters, ``tool_text`` (positional, no default) and the optional ``tool_set`` defaulting to ``None``; no
    ``*args``/``**kwargs`` and no third parameter, so no stop id (or any other commit field) can be passed in."""
    P = inspect.Parameter
    params = list(inspect.signature(S.fn("classify", "T-0102 (classify)")).parameters.values())
    assert [p.name for p in params] == ["tool_text", "tool_set"], params
    text, tool_set = params
    assert text.kind is P.POSITIONAL_OR_KEYWORD and text.default is P.empty, text
    assert tool_set.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY), tool_set
    assert tool_set.default is None, tool_set
