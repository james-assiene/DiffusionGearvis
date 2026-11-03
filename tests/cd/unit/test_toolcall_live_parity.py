"""T-0091 parities (cd group): the payloads, the classifier, the continuation guard and the repair request against
what ``twopass`` and ``decoding`` sent and scored on the stored T-0084e/T-0084f artifacts.

Read-only inputs: the two stored native artifacts in the main checkout's ``logs/`` (shas checked), the pinned
tokenizer through ``decoding.vocab``, the frozen calibration ``docs/cd/calibration-bfcl.json`` through
``eval.cd.calibration.load_frozen``. The oracle is always the eval's own code path:

- **payload identity** (criterion 1): ``twopass._Run.automaton`` (called with a duck-typed ``self`` carrying the eval's
  default ``whitespace``), then ``artifact.encode_automaton`` -- byte for byte;
- **classify** (criterion 2): the stored ``draft_parse``/``draft_schema_valid`` of every native pass-1 row, then
  ``twopass.score_call`` plus the eval's named-scope choice (``native.draft_name``) on perturbed drafts (length cuts,
  a dropped string delimiter, an out-of-range coordinate, a renamed tool);
- **repair_verdict** (criterion 6): ``twopass.score_call(twopass.call_text_of(text, "native"))`` on every stored
  native T4 pass-2 output;
- **guard** (criterion 4): ``twopass.continuation_guard(spoken + "<|tool_call>", "<|tool_call>")`` on every stored
  spoken prefix, T-0085's rows, every special literal, and a Hypothesis alphabet;
- **xargs** (criterion 5 + amendment): ``twopass.seeded_xargs(arms.cd_xargs("perstep_mar", payload, tag, knobs),
  twopass.seed_ids("T4", ids, draft, fmt="native"))`` key for key, for the row's own first stop, for ``None`` (the
  ids cut before their stop) and for 106 (the stop replaced); plus the stored pass-2 ``seed_ids`` and ``knobs``.

Mutants these rows kill (task): dropping the stop from the seed (the first-stop and 106 rows), the
``perstep_mar_qargmax`` knobs (entropy bound 0.3 != 1.0 on every row). The 'rstrip after encoding' mutant cannot be
expressed inside ``repair_request``, which takes ids: see the fast row
``test_c5_the_draft_ids_are_used_verbatim_including_trailing_whitespace_ids`` and the trailing-whitespace row here.

API assumptions: ``tests/unit/_toolcall_live_support.py``. The build script is assumed to take ``--out PATH``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.decoding import artifact, grammars, vocab
from local_astra.eval.cd import arms, calibration, twopass
from local_astra.toolcall import native, router, tools
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.cd import _native_artifacts as na
from tests.unit import _toolcall_live_support as S

OPEN = twopass.TOOL_CALL_OPEN
STOPS = frozenset({1, 106, 50})
ARTIFACTS = {
    "t0084e-dev": (
        "logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl",
        "06a71fc7413828ba5ff2843accf0811b5ce8303014ebf0177e99904060bd5c7f",
    ),
    "t0084f-test": (
        "logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl",
        "14728634eee2a8621e4bf3fbaf3c84c52ccc224e7c6b0383c82a4dcac0595101",
    ),
}
SEED_CELL = "native/T4/named/perstep_mar+seed"


@functools.cache
def _rows(key: str) -> tuple[dict[str, Any], ...]:
    rel, sha = ARTIFACTS[key]
    path = na._main_checkout() / rel
    assert path.is_file(), f"the stored {key} artifact is not at {path}"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha, (
        f"{key}: sha mismatch (read-only artifact edited?)"
    )
    return tuple(json.loads(x) for x in raw.decode().splitlines() if x.strip())


def _native_pass1() -> list[tuple[str, dict[str, Any]]]:
    return [
        (k, r)
        for k in ARTIFACTS
        for r in _rows(k)
        if r.get("pass") == 1 and r.get("format") == "native" and r.get("error") is None
    ]


def _routed() -> list[tuple[str, dict[str, Any]]]:
    out = [
        (k, r)
        for k, r in _native_pass1()
        if r["route"]["kind"] == "toolcall" and r["route"]["tag"] == "native"
    ]
    assert len(out) == 120, f"premise: 40 dev + 80 test routed native drafts, got {len(out)}"
    return out


@functools.cache
def _toolset() -> Any:
    return tools.load()


@functools.cache
def _eval_payload(name: str) -> tuple[str, int, int]:
    """``(encoded payload, n_states, bucket)`` exactly as ``twopass._Run.request`` builds it for a native T4 named
    cell: ``self.automaton(scope, "T4", "native")`` with the run config's default whitespace."""
    ws = twopass.TwoPassConfig.__dataclass_fields__["whitespace"].default
    fake_self = SimpleNamespace(tools=_toolset(), cfg=SimpleNamespace(whitespace=ws))
    a = twopass._Run.automaton(fake_self, name, "T4", "native")  # type: ignore[arg-type]
    return artifact.encode_automaton(a), int(a.n_states), int(a.n_states_bucket)


@functools.cache
def _knobs() -> dict[str, float]:
    values, sha = calibration.load_frozen(S.CALIBRATION, task="bfcl", arm="perstep_mar")
    assert sha == S.CALIBRATION_SHA256
    return values


def _first_stop(ids: Sequence[int]) -> int | None:
    return next((int(t) for t in ids if int(t) in STOPS), None)


def _until_stop(ids: Sequence[int]) -> list[int]:
    out: list[int] = []
    for t in ids:
        if int(t) in STOPS:
            break
        out.append(int(t))
    return out


# =========================================================================== criterion 1: payload identity


@pytest.mark.parametrize("name", S.V1)
def test_c1_each_payload_is_byte_identical_to_what_twopass_sends(name: str) -> None:
    payload, n_states, bucket = _eval_payload(name)
    e = S.tool_entries(S.payload_doc())[name]
    assert e["automaton"] == payload, (
        f"{name}: the payload differs from compile_toolcall -> encode_automaton"
    )
    assert e["automaton_sha256"] == hashlib.sha256(payload.encode("ascii")).hexdigest()
    assert (e["n_states"], e["bucket"]) == (n_states, bucket)


def test_c1_the_manifest_is_this_compiler_tokenizer_model_tools_and_canvas() -> None:
    vals = S.scalars(S.payload_doc())
    strs = [v for v in vals if isinstance(v, str)]
    for want in (
        _toolset().version,
        vocab.TOKENIZER_SHA256,
        grammars.compiler_version(),
    ):
        assert want in strs, f"the manifest lacks {want!r}"
    assert any(len(v) >= 7 and vocab.MODEL_REVISION.startswith(v) for v in strs), (
        "no model revision"
    )
    assert twopass.CANVAS_LENGTH in vals


def test_c1_the_build_script_reproduces_the_committed_file(tmp_path: Path) -> None:
    """The build script is the only writer and this row is its oracle: run with ``--out`` (assumed), it writes the
    committed bytes; twice, the same bytes."""
    script = S.REPO / "scripts" / "build_toolcall_payloads.py"
    assert script.is_file(), (
        "T-0091 criterion 1: scripts/build_toolcall_payloads.py does not exist yet"
    )
    outs = []
    for i in range(2):
        out = tmp_path / f"native_T4.{i}.json"
        proc = subprocess.run(
            [sys.executable, str(script), "--out", str(out)],
            cwd=S.REPO,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        outs.append(out.read_bytes())
    assert outs[0] == outs[1], "the build is not deterministic"
    assert outs[0] == S.PAYLOADS.read_bytes(), (
        "the committed payload file is not what the script writes"
    )


# =========================================================================== criterion 2: classify


def test_c2_classify_agrees_with_every_stored_native_draft() -> None:
    """Criterion 2 parity: on all 120 routed native pass-1 rows, ``classify(<|tool_call> + draft + close)`` is
    ``valid`` iff the row's ``draft_parse == "ok"`` and ``draft_schema_valid``; a valid verdict's name and arguments
    are the row's ``draft_call``. (All 120 stored drafts are valid -- H-T84-2's caveat -- hence the perturbed row.)"""
    classify = S.fn("classify", "criterion 2 (classify)")
    bad = []
    for key, r in _routed():
        v = classify(OPEN + r["draft_call_text"] + r["draft_close"])
        want_valid = r["draft_parse"] == "ok" and r["draft_schema_valid"] is True
        if (v.status == "valid") != want_valid:
            bad.append((key, r["case"], v.status, r["draft_parse"]))
        elif want_valid and (v.name, v.arguments) != (
            r["draft_call"]["name"],
            r["draft_call"]["arguments"],
        ):
            bad.append((key, r["case"], v.name, v.arguments))
    assert bad == []


def _perturbations(draft: str) -> list[str]:
    out = [draft[:k] for k in range(0, len(draft), 3)]
    out.append(draft.replace(native.STRING_DELIM, "", 1))
    out.append(draft.replace(native.STRING_DELIM, ""))
    out.append(draft.replace("x:", "x:1", 1))
    out.append(draft.replace("call:", "call:navigate_", 1))
    out.append(draft + " ok")
    return out


def _eval_status(draft: str) -> tuple[str, str | None]:
    """The eval's verdict on a draft: ``score_call`` decides valid; else its named scope (``native.draft_name``, as
    ``_Run.pass2`` picks it) decides repairable; else refused, with the parse reason or "no tool name"."""
    parse, _call, valid, _ex = twopass.score_call(draft, _toolset(), None, "native")
    if parse == "ok" and valid:
        return "valid", None
    if native.draft_name(draft) is not None:
        return "repairable", None
    return "refused", (parse if parse != "ok" else "no tool name")


# --------------------------------------------------------------------------- the T-0100 delta (H-T99d, H-T99e)
#
# The eval measures the model and keeps T-0091's rule; H-T99d/H-T99e are live policy (T-0100 implementation notes), so
# the oracle carries the delta itself. Written from docs/tasks/T-0100.md "Definitions" alone -- the value scan, the
# alias table and the two rules -- independently of ``live.py`` (a character walk here, whatever the implementer
# chose there).

T0100_NOT_IN_ENUM = "combo not in enum"
T0100_NO_COMBO = "no combo"
T0100_OUT_OF_RANGE = "coordinate out of range"
_T0100_ALIASES = {
    "esc": "escape",
    "return": "enter",
    "control": "ctrl",
    "del": "delete",
    "up": "arrowup",
    "down": "arrowdown",
    "left": "arrowleft",
    "right": "arrowright",
}
_HEAD = re.compile(r"\A\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*")
_BARE = re.compile(r"[^,{}\[\]\s<]*")
_NUM = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class _Raw(str):
    """A bare token that is not a JSON literal: raw text, never a string value."""


def _bare_value(tok: str) -> Any:
    if tok in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[tok]
    if _NUM.fullmatch(tok):
        return json.loads(tok)
    return _Raw(tok)


def _scan(draft: str) -> dict[str, Any]:
    """The value scan of a draft that does not parse: first occurrence of each top-level key (depth 1 of the call's
    braces, ruling V4) in key position (after ``{`` or ``,``); closed strings are values and never searched; an
    unterminated string, a call tag outside a string, or the call's closing brace ends the scan; a bare token that
    runs into the end of the text, or an empty one (V5), gives no value."""
    m = _HEAD.match(draft)
    if m is None:
        return {}
    delim, n = native.STRING_DELIM, len(draft)
    out: dict[str, Any] = {}
    i, key_position, depth = m.end(), True, 1
    while i < n:
        if draft.startswith(delim, i):
            end = draft.find(delim, i + len(delim))
            if end < 0:
                break
            i, key_position = end + len(delim), False
            continue
        if draft.startswith(native.TOOL_CALL_CLOSE, i) or draft.startswith(OPEN, i):
            break
        ch = draft[i]
        if ch in "{[":
            i, depth, key_position = i + 1, depth + 1, False
            continue
        if ch in "}]":
            i, depth, key_position = i + 1, depth - 1, False
            if depth <= 0:
                break
            continue
        if ch == ",":
            i, key_position = i + 1, True
            continue
        if ch.isspace():
            i += 1
            continue
        k = _KEY.match(draft, i) if key_position and depth == 1 else None
        if k is None:
            i, key_position = i + 1, False
            continue
        name = draft[i : k.end()].split(":", 1)[0].strip()
        j = k.end()
        if draft.startswith(delim, j):
            end = draft.find(delim, j + len(delim))
            if end < 0:
                break  # unterminated: no value, and the scan ends
            out.setdefault(name, draft[j + len(delim) : end])
            i, key_position = end + len(delim), False
            continue
        tok = _BARE.match(draft, j)
        assert tok is not None
        if tok.end() >= n:
            break  # a length cut: the last token is unread
        if tok.group():
            out.setdefault(name, _bare_value(tok.group()))
        i, key_position = max(tok.end(), j), False
    return out


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _normalised(combo: str) -> str:
    return "+".join(_T0100_ALIASES.get(p, p) for p in _squash(combo).split("+"))


@functools.cache
def _squashed_entries() -> frozenset[str]:
    decl = _toolset().get("press_key")
    assert decl is not None
    return frozenset(_squash(e) for e in decl["parameters"]["properties"]["combo"]["enum"])


def _in_range(v: Any) -> bool:
    integral = (isinstance(v, int) and not isinstance(v, bool)) or (
        isinstance(v, float) and v.is_integer()
    )
    return integral and 0 <= v <= 999


def _t0100_status(draft: str) -> tuple[str, str | None]:
    """``_eval_status`` plus the T-0100 delta: a not-valid draft naming press_key, click_at or type_at is refused by
    H-T99d/H-T99e on its read-back values (the parsed arguments, or the scan's)."""
    status, reason = _eval_status(draft)
    if status == "valid":
        return status, reason
    parse, call, _valid, _ex = twopass.score_call(draft, _toolset(), None, "native")
    if parse == "ok":
        name, values = call["name"], dict(call["arguments"])
    else:
        name, values = native.draft_name(draft), _scan(draft)
    if name == "press_key":
        if "combo" not in values:
            return "refused", T0100_NO_COMBO
        combo = values["combo"]
        if type(combo) is not str or _normalised(combo) not in _squashed_entries():
            return "refused", T0100_NOT_IN_ENUM
    elif name in ("click_at", "type_at"):
        if any(k in values and not _in_range(values[k]) for k in ("x", "y")):
            return "refused", T0100_OUT_OF_RANGE
    return status, reason


# --------------------------------------------------------------------------- the T-0102 delta (H-T101c)
#
# Written from docs/tasks/T-0102.md "Definitions" alone, independently of ``live.py`` (no implementation of the rule
# existed when this was written): the call's end E by a string-aware depth walk, the remainder R = draft[E:], "second
# call" iff ``R.lstrip()`` starts with ``<|tool_call>``, "model text" iff R with every special literal replaced by a
# space holds an ``isalnum`` character. Applied only where the T-0100 oracle says ``repairable``.

T0102_MODEL_TEXT_AFTER_CALL = "model text after call"


def _call_end(draft: str) -> int | None:
    """E per the Definitions, or None (a clean cut: the walk reaches the end, or a string never closes)."""
    m = _HEAD.match(draft)
    if m is None:
        return None
    delim, close, n = native.STRING_DELIM, native.TOOL_CALL_CLOSE, len(draft)
    depth, i = 1, m.end()
    while i < n:
        if draft.startswith(delim, i):
            j = draft.find(delim, i + len(delim))
            if j < 0:
                return None  # a cut inside a string is clean
            i = j + len(delim)
            continue
        if draft.startswith(close, i):
            return i + len(close)  # a close tag at depth >= 1
        if draft.startswith(OPEN, i):
            return i  # a call tag: E at its own index
        if draft[i] in "{[":
            depth += 1
        elif draft[i] in "}]":
            depth -= 1
            if depth == 0:
                e = i + 1
                k = e
                while k < n and draft[k].isspace():
                    k += 1
                return k + len(close) if draft.startswith(close, k) else e
        i += 1
    return None


def _holds_model_text(rest: str) -> bool:
    for lit in SPECIAL_TOKENS:
        rest = rest.replace(lit, " ")
    return any(ch.isalnum() for ch in rest)


def _h101c_applies(draft: str) -> bool:
    """The T-0102 position test alone (whether the verdict would otherwise be repairable is the caller's)."""
    e = _call_end(draft)
    if e is None:
        return False
    rest = draft[e:]
    return not rest.lstrip().startswith(OPEN) and _holds_model_text(rest)


def _with_t0102(status: tuple[str, str | None], draft: str) -> tuple[str, str | None]:
    if status[0] == "repairable" and _h101c_applies(draft):
        return "refused", T0102_MODEL_TEXT_AFTER_CALL
    return status


def _t0102_status(draft: str) -> tuple[str, str | None]:
    """``_t0100_status`` plus the T-0102 delta: a draft T-0100 leaves ``repairable`` whose call has ended and is
    followed by model text is refused "model text after call"."""
    return _with_t0102(_t0100_status(draft), draft)


def _check_against(
    oracle: Any, names: frozenset[str] | None, skip: Any = None
) -> tuple[dict[str, int], list[Any]]:
    classify = S.fn("classify", "criterion 2 (classify)")
    seen: dict[str, int] = {"valid": 0, "repairable": 0, "refused": 0}
    bad = []
    for _key, r in _routed():
        for d in _perturbations(r["draft_call_text"]):
            if names is not None and native.draft_name(d) in names:
                continue
            if skip is not None and skip(d):
                continue
            status, reason = oracle(d)
            seen[status] += 1
            v = classify(OPEN + d)
            if v.status != status or (reason is not None and v.reason != reason):
                bad.append((d, status, reason, v.status, v.reason))
    return seen, bad


def test_c2_classify_agrees_with_the_eval_on_perturbed_stored_drafts() -> None:
    """T-0091 criterion 2 parity with T-0100's delta: on every perturbation of the 120 stored drafts (length cuts, a
    dropped delimiter, ``x:`` -> ``x:1`` which pushes a coordinate past 999, a renamed tool, text after the call),
    ``classify`` agrees with the eval's verdict plus H-T99d/H-T99e, reasons included where the oracle names one.

    T-0102 premise change: the ``draft + " ok"`` perturbation is model text after a call, which T-0102 refuses. This
    row now skips the perturbations the T-0102 delta changes (so it is green before and after T-0102 and keeps pinning
    T-0100's parity on the rest); the whole set, delta included, is
    ``test_c2_classify_agrees_with_the_eval_plus_t0100_and_t0102``."""
    seen, bad = _check_against(
        _t0100_status, None, skip=lambda d: _t0102_status(d) != _t0100_status(d)
    )
    assert all(seen.values()), f"premise: every status is exercised, {seen}"
    assert bad == [], f"{len(bad)} disagreements, first: {bad[:3]}"


def test_c2_classify_agrees_with_the_eval_plus_t0100_and_t0102() -> None:
    """T-0102 criteria 1-7 parity: on every perturbation of the 120 stored drafts, ``classify`` agrees with the eval's
    verdict plus the T-0100 delta plus the independent T-0102 delta, reasons included where the oracle names one."""
    seen, bad = _check_against(_t0102_status, None)
    assert all(seen.values()), f"premise: every status is exercised, {seen}"
    assert bad == [], f"{len(bad)} disagreements, first: {bad[:3]}"


def test_c2_the_t0102_delta_is_exercised_by_the_perturbations() -> None:
    """Premise of the row above (oracle only, no ``live``): the T-0102 delta changes some perturbed verdicts, every
    change is T-0100's ``repairable`` -> ``refused`` "model text after call", and the ``draft + " ok"`` perturbation
    is among them for every stored draft whose ``+ " ok"`` T-0100 leaves repairable -- otherwise the pending row would
    be the T-0100 row again."""
    changed = 0
    ok_changed = ok_repairable = 0
    for _key, r in _routed():
        draft = r["draft_call_text"]
        for d in _perturbations(draft):
            new, old = _t0102_status(d), _t0100_status(d)
            if new != old:
                assert old[0] == "repairable", (d, old, new)
                assert new == ("refused", T0102_MODEL_TEXT_AFTER_CALL), (d, old, new)
                changed += 1
        if _t0100_status(draft + " ok")[0] == "repairable":
            ok_repairable += 1
            ok_changed += _t0102_status(draft + " ok") != _t0100_status(draft + " ok")
    assert changed > 0 and ok_repairable > 0, (changed, ok_repairable)
    assert ok_changed == ok_repairable, (ok_changed, ok_repairable)


def test_c2_the_t0100_delta_is_exercised_by_the_perturbations() -> None:
    """Premise of the row above: the delta changes some perturbed verdicts (refusing each of the three reasons) and
    leaves the rest alone -- otherwise the pending row would be the T-0091 row again."""
    reasons: dict[str, int] = {}
    for _key, r in _routed():
        for d in _perturbations(r["draft_call_text"]):
            new, old = _t0100_status(d), _eval_status(d)
            if new != old:
                assert old[0] == "repairable" and new[0] == "refused", (d, old, new)
                reasons[str(new[1])] = reasons.get(str(new[1]), 0) + 1
    assert set(reasons) == {T0100_NOT_IN_ENUM, T0100_NO_COMBO, T0100_OUT_OF_RANGE}, reasons


def test_c2_t0100_leaves_the_other_tools_with_the_eval() -> None:
    """T-0100 criterion 7 parity (negative, green before and after): perturbations whose draft names no
    press_key/click_at/type_at -- scroll, google_search, the renamed and the nameless ones -- keep the eval's T-0091
    verdict exactly.

    T-0102 premise change: T-0102 is not confined to T-0100's three tools (criterion 7), so a scroll or google_search
    ``draft + " ok"`` is refused after T-0102. This row skips the perturbations the T-0102 delta changes (green
    before and after); ``test_c2_t0102_gives_the_other_tools_only_the_h101c_delta`` covers the rest."""
    three = frozenset({"press_key", "click_at", "type_at"})
    seen, bad = _check_against(
        _eval_status, three, skip=lambda d: _with_t0102(_eval_status(d), d) != _eval_status(d)
    )
    assert seen["repairable"] and seen["refused"] and seen["valid"], seen
    assert bad == [], f"{len(bad)} disagreements, first: {bad[:3]}"


def test_c2_t0102_gives_the_other_tools_only_the_h101c_delta() -> None:
    """T-0102 criterion 7 parity: perturbations naming none of T-0100's three tools get the eval's verdict plus the
    T-0102 delta only (scroll/google_search ``+ " ok"`` refused "model text after call"), never a T-0100 reason."""
    three = frozenset({"press_key", "click_at", "type_at"})
    seen, bad = _check_against(lambda d: _with_t0102(_eval_status(d), d), three)
    assert seen["repairable"] and seen["refused"] and seen["valid"], seen
    assert bad == [], f"{len(bad)} disagreements, first: {bad[:3]}"


@pytest.mark.parametrize(
    ("draft", "want"),
    [
        ('call:press_key{combo:<|"|>Ctrl+S<|"|>}<tool_call|>', ("refused", T0100_NOT_IN_ENUM)),
        ('call:press_key{combo:<|"|>control + c<|"|>}<tool_call|>', ("repairable", None)),
        ('call:press_key{combo:<|"|>pageup<|"|>}<tool_call|>', ("repairable", None)),
        ('call:press_key{combo:<|"|>Cmd+C<|"|>}<tool_call|>', ("refused", T0100_NOT_IN_ENUM)),
        ("call:press_key{}<tool_call|>", ("refused", T0100_NO_COMBO)),
        ('call:press_key{combo:<|"|>Ctrl+', ("refused", T0100_NO_COMBO)),
        ("call:press_key{combo:Enter}<tool_call|>", ("refused", T0100_NOT_IN_ENUM)),
        ('call:press_key{combo:<|"|>Esc<|"|>}', ("repairable", None)),
        ("call:click_at{x:1088,y:748}<tool_call|>", ("refused", T0100_OUT_OF_RANGE)),
        ("call:click_at{x:999,y:0}<tool_call|>", ("valid", None)),
        ("call:click_at{x:true,y:0}<tool_call|>", ("refused", T0100_OUT_OF_RANGE)),
        ("call:click_at{x:488,y:1000", ("repairable", None)),
        ('call:click_at{note:<|"|>,y:1000<|"|>,x:10,y:10}<tool_call|> ok', ("repairable", None)),
        ("call:scroll{direction:sideways,x:1200}<tool_call|>", ("repairable", None)),
        # rulings V3-V5
        ('call:press_key{combo:"Ctrl+C"}<tool_call|>', ("refused", T0100_NOT_IN_ENUM)),
        ('call:click_at{x:"500",y:5}<tool_call|>', ("refused", T0100_OUT_OF_RANGE)),
        ("call:click_at{pos:{x:1200},x:5,y:5}<tool_call|> ok", ("repairable", None)),
        (
            'call:press_key{opts:{combo:<|"|>Ctrl+C<|"|>}}<tool_call|> ok',
            ("refused", T0100_NO_COMBO),
        ),
        ("call:click_at{x:,y:5}<tool_call|>", ("repairable", None)),
        ("call:press_key{combo:}<tool_call|>", ("refused", T0100_NO_COMBO)),
        # reviewer A (89d8321): bounds on not-valid drafts, V4 after a comma, unterminated, V7 depth 0, the call tag
        ('call:click_at{x:999,y:0,button:<|"|>left<|"|>}<tool_call|>', ("repairable", None)),
        ('call:type_at{text:<|"|>hi<|"|>,x:0,y:999}<tool_call|>', ("repairable", None)),
        ("call:click_at{x:999,y:0}<tool_call|> ok", ("repairable", None)),
        ("call:click_at{pos:{a:1,x:1200},x:5,y:5}<tool_call|> ok", ("repairable", None)),
        (
            'call:press_key{opts:{a:1,combo:<|"|>Ctrl+C<|"|>}}<tool_call|> ok',
            ("refused", T0100_NO_COMBO),
        ),
        ('call:type_at{text:<|"|>hello, x:1200', ("repairable", None)),
        ('call:type_at{text:<|"|>hello, x:1200,y:5', ("repairable", None)),
        ('call:press_key{note:<|"|>a, combo:<|"|>F5', ("refused", T0100_NO_COMBO)),
        ('call:press_key{note:<|"|>a, combo:Ctrl+C,x', ("refused", T0100_NO_COMBO)),
        ("call:click_at{y:5},x:1200", ("repairable", None)),
        ("call:click_at{y:5}{a:1,x:1200,z", ("repairable", None)),
        ("call:click_at{x:5<tool_call|>,y:1200}", ("repairable", None)),
    ],
)
def test_c2_the_t0100_oracle_on_the_tasks_hand_rows(
    draft: str, want: tuple[str, str | None]
) -> None:
    """The oracle itself, against rows the task states verbatim (criteria 1-7), so a wrong oracle cannot make the
    pending parity row agree with a wrong implementation."""
    assert _t0100_status(draft) == want


_K = 'call:press_key{combo:<|"|>Ctrl+T<|"|>}'
_H = ("refused", T0102_MODEL_TEXT_AFTER_CALL)
_R = ("repairable", None)


@pytest.mark.parametrize(
    ("draft", "want"),
    [
        # criterion 1: the retraction, and the stray delimiter after the brace
        (_K + " Wait, no.", _H),
        (_K + ' Wait <|"|> tool', _H),
        (
            _K
            + " Wait, the tool `press_key` only takes one key.<channel|>I cannot perform that specific shortcut "
            "directly.",
            _H,
        ),
        # criterion 2 (Q1): closed calls followed by prose, any of the five tools
        (_K + "<tool_call|> Wait, no.", _H),
        ("call:click_at{x:1,y:2}<tool_call|> ok", _H),
        ('call:scroll{direction:<|"|>down<|"|>}<tool_call|> ok', _H),
        (_K + "<tool_call|> 5", _H),
        (_K + " Wait<tool_call|>", _H),
        ("call:click_at{x:5<tool_call|> Wait, no", _H),
        # criterion 3: clean cuts (Q4 included)
        ("call:click_at{x:488,y:5", _R),
        (_K, _R),
        (_K + "\n", _R),
        ('call:google_search{query:<|"|>weather', _R),
        ('call:type_at{x:500,y:36,text:<|"|>usb-c', _R),
        ('call:press_key{combo:<|"|>Ctrl+T<|"|> Wait I cannot', _R),
        ("call:click_at{pos:{x:1},y:5", _R),
        # criterion 4: benign remainders, a brace inside a string
        (_K + "<channel|>\n", _R),
        (_K + "<tool_call|><channel|>", _R),
        ("call:click_at{x:1,y:2}}<tool_call|>", _R),
        ("call:click_at{x:1,,y:2}<tool_call|>", _R),
        (_K + "<tool_call|>.", _R),
        ("call:click_at{pos:{x:1},y:5}<tool_call|><channel|>", _R),
        ('call:type_at{text:<|"|>a} Wait<|"|>,x:5,y:5,press_enter:false', _R),
        ('call:type_at{text:<|"|>a} Wait<|"|>,x:5,y:5,press_enter:false} Wait, no.', _H),
        # criterion 5 (Q2): a second call keeps today's verdict
        (
            'call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
            + '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>' * 2,
            _R,
        ),
        (_K + '<|tool_call>call:press_key{combo:<|"|>Escape<|"|>}<tool_call|>', _R),
        (_K + "<tool_call|> ok" + _K + "<tool_call|>", _H),
        # criterion 6: T-0100 first
        ('call:press_key{combo:<|"|>Ctrl+S<|"|>} Wait', ("refused", T0100_NOT_IN_ENUM)),
        ("call:click_at{x:1200,y:5} Wait, no.", ("refused", T0100_OUT_OF_RANGE)),
        ("call:press_key{} Wait", ("refused", T0100_NO_COMBO)),
        # the T-0100 hand rows built with "+ ok" (premise change): refused now, T-0100's own refusals kept
        ('call:click_at{note:<|"|>,y:1000<|"|>,x:10,y:10}<tool_call|> ok', _H),
        ("call:click_at{pos:{x:1200},x:5,y:5}<tool_call|> ok", _H),
        (
            'call:press_key{opts:{combo:<|"|>Ctrl+C<|"|>}}<tool_call|> ok',
            ("refused", T0100_NO_COMBO),
        ),
        ("call:click_at{x:999,y:0}<tool_call|> ok", _H),
        ("call:click_at{pos:{a:1,x:1200},x:5,y:5}<tool_call|> ok", _H),
        # a text-level length cut inside the close literal (``<too``): the Definitions replace whole special literals
        # only, so R = "<too" holds letters -> refused. Not live-reachable (the close is one token id); 398 of the
        # stored-draft length cuts have this shape. Flagged to the reviewer as an ambiguity, pinned as written.
        ("call:click_at{x:1,y:2}<too", _H),
        # V7 depth 0 and the call tag: what follows the end is model text
        ("call:click_at{y:5},x:1200", _H),
        ("call:click_at{y:5}{a:1,x:1200,z", _H),
        ("call:click_at{x:5<tool_call|>,y:1200}", _H),
    ],
)
def test_c2_the_t0102_oracle_on_the_tasks_hand_rows(
    draft: str, want: tuple[str, str | None]
) -> None:
    """The T-0102 oracle itself, against the task's rows (criteria 1-6, by hand from the Definitions), so a wrong
    oracle cannot make the pending parity row agree with a wrong implementation."""
    assert _t0102_status(draft) == want


# =========================================================================== criterion 6: repair_verdict


def test_c6_repair_verdict_agrees_with_the_eval_scoring_of_every_stored_native_t4_output() -> None:
    """Criterion 6 parity: on every stored native T4 pass-2 output (4 cells x 120), ``repair_verdict(text)`` is
    ``valid`` iff ``score_call(call_text_of(text, "native"))`` parses and validates; otherwise ``refused`` (never
    ``repairable``)."""
    rv = S.fn("repair_verdict", "criterion 6 (repair_verdict)")
    rows = [
        r
        for k in ARTIFACTS
        for r in _rows(k)
        if r.get("pass") == 2
        and str(r.get("cell", "")).startswith("native/T4/")
        and r.get("error") is None
    ]
    assert len(rows) >= 400, len(rows)
    bad = []
    for r in rows:
        parse, call, valid, _ex = twopass.score_call(
            twopass.call_text_of(r["text"], "native"), _toolset(), None, "native"
        )
        v = rv(r["text"])
        want = "valid" if parse == "ok" and valid else "refused"
        if v.status != want or (
            want == "valid" and (v.name, v.arguments) != (call["name"], call["arguments"])
        ):
            bad.append((r["case"], r["cell"], want, v.status))
    assert bad == []


# =========================================================================== criterion 4: the continuation guard


def _pass1() -> Prompt:
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text=twopass.NATIVE_SYSTEM_PROMPT),)),
            Message(role="user", content=(TextPart(text="Sign in"),)),
        ),
        prefix_len=2,
        tools=tuple(native.tools_field(_toolset())),
    )


def _live_guard(spoken: str) -> str | None:
    got = S.fn("repair_prompt", "criterion 4 (repair_prompt)")(_pass1(), spoken)
    if isinstance(got, Prompt):
        assert got.messages[-1].content == (TextPart(text=spoken + OPEN),)
        return None
    return str(got.reason).removeprefix("continuation_guard: ")


def _spoken_cases() -> list[str]:
    from tests.unit._toolcall_support import CLICK01_SPOKEN, CLICK01_STORED_SPOKEN

    stored = sorted({r["route"]["spoken_prefix"] for _k, r in _routed()})
    hand = [
        CLICK01_SPOKEN,
        CLICK01_STORED_SPOKEN,
        "Sure.<|tool_<channel|>call> ",
        "Sure.<|tool_call> ",
        "Look.<|image|>",
        "a < b and x |> y.",
        "<cc> is fine?",
        " Sure.",
        "",
    ]
    return stored + hand + ["Sure." + lit + " " for lit in SPECIAL_TOKENS]


def test_c4_the_guard_agrees_with_twopass_on_stored_and_t0085_rows() -> None:
    cases = _spoken_cases()
    got = {c: _live_guard(c) for c in cases}
    want = {c: twopass.continuation_guard(c + OPEN, OPEN) for c in cases}
    assert {c for c in cases if want[c] is None} and {c for c in cases if want[c] is not None}, (
        "premise"
    )
    assert {c: (got[c], want[c]) for c in cases if got[c] != want[c]} == {}


_PIECES = [
    *SPECIAL_TOKENS,
    "<|",
    "|>",
    "<|to",
    "ol_call>",
    "<channel",
    "chan",
    "Sure.",
    " ",
    "\n",
    "a",
    "<",
    ">",
    "|",
]


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(_PIECES), max_size=8))
def test_c4_the_guard_agrees_with_twopass_on_spliced_fragments(parts: list[str]) -> None:
    """Criterion 4 parity, Hypothesis (300 examples over special literals, their fragments and prose)."""
    spoken = "".join(parts)
    assert _live_guard(spoken) == twopass.continuation_guard(spoken + OPEN, OPEN), repr(spoken)


# =========================================================================== criterion 5: the repair request

_DRAFT_KEYS = ("astra_cd_draft", "astra_cd_draft_noise", "astra_cd_draft_start_step")


def _twopass_xargs(name: str, ids: Sequence[int], draft: str, tag: str) -> dict[str, Any]:
    payload, _n, _b = _eval_payload(name)
    return twopass.seeded_xargs(
        arms.cd_xargs("perstep_mar", payload, tag, _knobs()),
        twopass.seed_ids("T4", list(ids), draft, fmt="native"),
    )


def _live_xargs(name: str, draft: str, stop: int | None, tag: str) -> dict[str, Any]:
    req = S.fn("repair_request", "criterion 5 (repair_request)")(
        name, vocab.encode(draft.rstrip()), stop, tag, S.payloads(), S.calibration()
    )
    return dict(req.to_xargs())


def _named_rows() -> list[tuple[str, dict[str, Any], str]]:
    out = [(k, r, native.draft_name(r["draft_call_text"])) for k, r in _routed()]
    named = [(k, r, n) for k, r, n in out if n is not None]
    assert len(named) == 120, f"premise: every stored native draft names a v1 tool, {len(named)}"
    return named  # type: ignore[return-value]


@pytest.mark.parametrize("variant", ["first_stop", "none", "106"])
def test_c5_repair_request_xargs_equal_twopass_key_for_key(variant: str) -> None:
    """Criterion 5 + amendment: for every stored native pass-1 row with a v1 draft name, ``repair_request(...)
    .to_xargs()`` equals the eval's seeded xargs key for key (``record_tag`` compared separately), with ``draft_ids =
    vocab.encode(draft.rstrip())`` and ``stop_id`` = the row's first stop (``first_stop``, natural 50 and 1), None
    with the ids cut before their stop (``none``), or 106 with the stop replaced (``106``)."""
    stops_seen: set[int | None] = set()
    bad = []
    for key, r, name in _named_rows():
        ids = [int(t) for t in r["token_ids"]]
        draft = r["draft_call_text"]
        if variant == "first_stop":
            stop, ids_v = _first_stop(ids), ids
        elif variant == "none":
            stop, ids_v = None, _until_stop(ids)
        else:
            stop, ids_v = 106, [*_until_stop(ids), 106]
        stops_seen.add(stop)
        tag = f"t0091.{key}.{r['case']}"
        want = _twopass_xargs(name, ids_v, draft, tag)
        got = _live_xargs(name, draft, stop, tag)
        if {k: v for k, v in got.items() if k != "astra_cd_record_tag"} != {
            k: v for k, v in want.items() if k != "astra_cd_record_tag"
        } or got.get("astra_cd_record_tag") != tag:
            bad.append((key, r["case"], sorted(set(got.items()) ^ set(want.items()))[:4]))
    assert bad == [], f"{len(bad)} rows differ, first: {bad[:2]}"
    if variant == "first_stop":
        assert {50, 1} <= stops_seen, f"premise: natural stops 50 and 1 both occur, {stops_seen}"


def test_c5_the_seeded_request_matches_the_stored_pass2_rows() -> None:
    """Criterion 5 against what was sent: for every stored ``native/T4/named/perstep_mar+seed`` row, the live request
    (built from the same pass-1 row) has the stored ``seed_ids`` as its draft and the stored ``knobs`` (minus the tag)
    as its other keys."""
    p1 = {(k, r["case"]): r for k, r in _routed()}
    bad = []
    n = 0
    for key in ARTIFACTS:
        for q in _rows(key):
            if q.get("pass") != 2 or q.get("cell") != SEED_CELL or q.get("error") is not None:
                continue
            n += 1
            r = p1[(key, q["case"])]
            (name,) = q["scopes"]
            req = S.fn("repair_request", "criterion 5 (repair_request)")(
                name,
                vocab.encode(r["draft_call_text"].rstrip()),
                _first_stop(r["token_ids"]),
                "t0091.stored",
                S.payloads(),
                S.calibration(),
            )
            x = req.to_xargs()
            rest = {
                k[len("astra_cd_") :]: v
                for k, v in x.items()
                if k not in (*_DRAFT_KEYS, "astra_cd_automaton", "astra_cd_record_tag")
            }
            stored = {k: v for k, v in q["knobs"].items() if k != "record_tag"}
            if S.draft_ids_of(req) != q["seed_ids"] or rest != stored:
                bad.append((key, q["case"]))
    assert n == 120 and bad == [], (n, bad[:3])


def test_c5_a_draft_with_trailing_whitespace_is_rstripped_before_encoding_as_the_eval_does() -> (
    None
):
    """The session's half of R2 (T-0092: ``tokenize(draft.rstrip())``): with ``draft_ids = encode(draft.rstrip())``
    the request equals the eval's seed for a draft carrying trailing whitespace, while the ids of the unstripped
    draft differ (so the stripping is load-bearing)."""
    _k, r, name = _named_rows()[0]
    draft = r["draft_call_text"] + "\n  "
    assert vocab.encode(draft) != vocab.encode(draft.rstrip()), "premise"
    ids = [int(t) for t in r["token_ids"]]
    want = _twopass_xargs(name, ids, draft, "t0091.ws")
    got = _live_xargs(name, draft, _first_stop(ids), "t0091.ws")
    assert got == want


def test_c5_the_knobs_are_the_perstep_mar_calibration_not_qargmax() -> None:
    """Premise of the qargmax mutant: the two frozen files differ in the entropy bound, and the live request carries
    perstep_mar's."""
    q, _sha = calibration.load_frozen(S.QARGMAX_CALIBRATION, task="bfcl", arm="perstep_mar_qargmax")
    assert q["entropy_bound"] != _knobs()["entropy_bound"]
    x = _live_xargs("click_at", "call:click_at{x:1,y:2}<tool_call|>", 50, "t0091.k")
    assert (x["astra_cd_entropy_bound"], x["astra_cd_confidence_threshold"]) == (
        _knobs()["entropy_bound"],
        _knobs()["confidence_threshold"],
    )


def test_premise_the_stored_regime_is_the_tasks() -> None:
    reg = _rows("t0084e-dev")[0]
    assert reg["tools_sha256"] == _toolset().version
    assert reg["versions"]["tokenizer_sha256"] == vocab.TOKENIZER_SHA256
    assert reg["versions"]["compiler_version"] == grammars.compiler_version()
    assert reg["calibration"]["perstep_mar"]["sha256"] == S.CALIBRATION_SHA256
    assert router.TAGS["native"] == OPEN
    assert isinstance(_rows("t0084f-test")[0], Mapping)
