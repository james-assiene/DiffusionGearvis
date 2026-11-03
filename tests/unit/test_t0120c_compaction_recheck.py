"""T-0120c: the compaction re-check harness's pure surface (unit; CPU; no engine).

The GPU re-check of T-0120b's invite stub re-runs only T-0115a's E rows that answer the question (old placement FULL
and COMPACT, kept placement COMPACT, warms as before). Every row that needs the harness carries ``T0120C_PENDING`` and
is red on the named assertion in ``harness()`` (``scripts/gpu/t0120c_compaction_recheck.py`` does not exist yet) or on
an assertion of its own, never on an import error: the harness is loaded lazily, by path. Mock-engine runs live in
``tests/integration/test_t0120c_harness_smoke.py`` (criterion 12); the gpu rows and the CPU calibration row in
``tests/eval/test_compaction_recheck.py``.

**The assumed harness surface** (the task names the behaviour; these names are the test-author's, stated so the
implementer can implement to them or say why not):

- ``T0115A``: T-0115a's harness module, loaded read-only **by path** from ``T0115A_PATH`` (the repo's
  ``scripts/gpu/t0115a_loop_requests.py``) when this module is imported; importing opens no socket. ``T0115A_SHA256``:
  the sha256 of that file's bytes. ``main`` hashes the file at ``T0115A_PATH`` (read at call time) and refuses, exit 2,
  naming the file, when it differs.
- Re-exported **unchanged** from ``T0115A`` (the same objects, ``is``): ``E_HISTORIES``, ``e_steps``, ``e_prompts``,
  ``elided_steps``, ``STEP_TEXTS``; ``E_BUDGET == 7000``. The harness defines no ``compact``/``elided_payload``/
  ``e_steps``/``e_prompts``/``elided_steps``/``classify_e``/``_maximal`` and builds no ``EHistory``.
- ``C398FDF_E_TEXTS_SHA256``: mapping ``"<history>/<placement>/<arm>"`` (arm ``warm``/``FULL``/``COMPACT``) -> the 24
  sha256 values of the c398fdf manifest's ``e_texts_sha256`` (the sha of the final assistant message text,
  ``prompt.messages[-1].text()``). ``main`` reads it at call time.
- ``ACCEPTED``: mapping history id -> criterion 4's accepted strings, verbatim as the task types them. ``main`` reads
  it at call time for criterion 4's literal check.
- ``INVENTED_PATTERNS``: mapping history id -> regex source strings (criterion 6). An answer is a value match iff
  ``re.search(p, normalise(answer))`` holds for some ``p`` of its history (so the patterns are written for casefolded,
  hyphen-free text). Recorded in the manifest as ``invented_patterns``.
- ``normalise(text) -> str``: casefold, every ``-`` and ``–`` replaced by a space, runs of whitespace collapsed to one
  space, stripped. A criterion 4 match is a normalised accepted string inside the normalised answer with no letter or
  digit on either side (R2).
- ``score_line(line) -> dict``, pure, on one E line of T-0115a's ``requests.jsonl`` shape (``history``,
  ``placement``, ``arm``, ``status``, ``error``, ``commits`` as ``Commit.model_dump()`` mappings), so the c398fdf lines
  score unchanged (criterion 7). It returns:
  - ``outcome``: criterion 4's list and matcher under T-0115a's rules (``errored`` when an error is recorded or the
    status is not 200, R4; a ``tool`` commit outside an open thought decides it, ``google_search`` -> ``re_search``,
    another call -> ``other``; a last commit inside a thought -> ``other``, R1; an empty answer -> ``other``; then
    ``correct`` / ``unsupported``);
  - ``outcome_t0115a``: the same request under T-0115a's list and matcher (``classify_e`` and R4), unchanged;
  - ``invented``: criterion 6 on an **old-COMPACT** line (``True``/``False``); falsy on every other line;
  - on a ``re_search`` line, ``query`` (the call's query string) and ``query_repeats_history`` (``True`` at least when
    the query equals one of the history's five queries, casefolded).
- ``recheck_outcome(tally, invented, complete) -> {"bars": {"C", "R", "I", "K": bool}, "outcome": str}``.
  ``tally[placement][arm][outcome]`` holds the cells (old, FULL), (old, COMPACT) and (kept, COMPACT), each over
  ``correct``, ``re_search``, ``unsupported``, ``other``, ``errored`` (criterion 4's list; the cell's n is its sum);
  ``invented`` the old-COMPACT invented count; ``complete`` whether the run sent its whole schedule. ``outcome`` is
  ``"stub re-searches"``, ``"stub falls short: <every failing bar, by its letter>"`` or ``"inconclusive: <reasons>"``,
  whose reasons name the bar ``C`` when C fails, say ``errored`` when 3 or more of the arm requests errored, and say
  ``incomplete`` when the run is incomplete.
- ``schedule() -> list[dict]``, pure: the 160 requests in send order, each with ``row`` ``"E"``, ``history``,
  ``placement``, ``arm`` (``warm``/``FULL``/``COMPACT``), ``repeat`` and ``request_id``
  (``t0120c-e-<history>-<placement>-<arm>-r<repeat>``).
- ``build_report(run_dir) -> dict``: writes nothing; reads the recorded per-line fields, so a directory holding only
  ``requests.jsonl`` is enough (as T-0115a's). Keys read here and in the integration and eval rows: ``tally`` and
  ``tally_t0115a`` (the cells above, under each list), ``invented`` (``{"k", "n"}``), ``invented_answers`` (a list of
  ``{"request_id", "answer"}``, verbatim, in send order), ``bars``, ``outcome``, ``complete``, ``errored``
  (``{"k", "n"}``).
- ``main(argv) -> int`` with ``--base-url`` and ``--out-dir``: 0 complete, 1 stopped or no engine, 2 refused, every
  refusal before any socket. ``manifest.json`` holds ``gate_shas`` (the four gate files of criterion 10),
  ``e_texts_sha256`` (the 24 texts of this run), ``elided``, ``prompt_tokens``, ``accepted`` (criterion 4's lists),
  ``accepted_t0115a`` (T-0115a's) and ``invented_patterns``; each ``requests.jsonl`` line the schedule entry's keys,
  T-0115a's E-line fields and ``score_line``'s keys.

Expected values are hand-typed: the accepted lists from criterion 4; the 132-char invite stub and the 103-char old
stub from T-0120b criterion 2; the 24 c398fdf text shas read from
``.claude/worktrees/t0115-launch/logs/t0115a-loop-requests-c398fdf/manifest.json`` (sha256 ``a33aa86b…``) and typed in;
the verbatim answers from docs/cd/toolcall-loop-requests.md and the c398fdf ``requests.jsonl``; the bars from criterion
8 (H-T120c-1). Where a row needs T-0115a's literals before the T-0120c harness exists, it loads T-0115a by path itself
(``t0115a()``), so its premise is checked now.

Regime: CPU, pure, hand-built data, ``loop.py`` at the working tree; no measurement.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra import cli
from local_astra.toolcall import loop
from local_astra.toolcall.loop import LoopStep
from tests.integration.test_t0093_harness_smoke import require_clean_tree
from tests.integration.test_t0098_harness_smoke import _kit_loader, no_sockets
from tests.unit.test_t0115a_loop_requests import (
    STUB_5,
    base_prompt,
    commit,
    final_text,
    hand_text,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0120c_compaction_recheck.py"
T0115A_HARNESS = REPO / "scripts" / "gpu" / "t0115a_loop_requests.py"

SURFACE = (
    "score_line",
    "recheck_outcome",
    "normalise",
    "schedule",
    "build_report",
    "main",
    "e_steps",
    "e_prompts",
    "elided_steps",
)
CONSTANTS = (
    "T0115A",
    "T0115A_PATH",
    "T0115A_SHA256",
    "C398FDF_E_TEXTS_SHA256",
    "ACCEPTED",
    "INVENTED_PATTERNS",
    "E_HISTORIES",
    "E_BUDGET",
    "STEP_TEXTS",
)

HISTORY_IDS = ("ferry", "market", "hall", "lighthouse")
PLACEMENTS = ("old", "kept")
#: the placement's fact-bearing step (0-based; T-0115a criterion 4)
FACT_STEP = {"old": 0, "kept": 4}
OUTCOMES = ("correct", "re_search", "unsupported", "other", "errored")
BARS = ("C", "R", "I", "K")
E_BUDGET = 7000

#: sha256 of scripts/gpu/t0115a_loop_requests.py at bef1c65 (the T-0120c fork point; unchanged since c398fdf,
#: ``git log c398fdf..bef1c65`` on the path is empty). Criteria 1 and 12: the harness pins it, and the file stays
#: byte-identical.
T0115A_SHA256_AT_BASE = "dbbad65fba98e698d3a0157d0ed1523b17ca7648730bbf296599081dd4adde1f"
#: criterion 12's other byte-identical files, pinned by content at bef1c65 (an invariant meant to hold, not a range)
T0115A_FILES_SHA256 = {
    "scripts/gpu/t0115a_loop_requests.py": T0115A_SHA256_AT_BASE,
    "scripts/locked/t0115_loop_probe.sh": "85e468072951ec34e1c6b8de7efbc295a2350f3f683708f8253ad9068d4abbc2",
    "docs/cd/toolcall-loop-requests.md": "baaadaeb21d3db2b87d35933a8a9b3f9634cd60b6358ba3e450459573978f343",
    "tests/unit/test_t0115a_loop_requests.py": "a5dd06203feb889d96c81e5301eba8b31fa4bf40a327ff09b83610a484a509d0",
    "tests/integration/test_t0115a_harness_smoke.py": "188444c39a90fbbd45e603314026feaa53246b631d4e5ab7bc60c9b73628c811",
    "tests/eval/test_loop_requests.py": "63748e81d2d763212f24ecbdb0ca346e2f18c284700af0355a193a4f03e68cd1",
}

#: The c398fdf manifest's ``e_texts_sha256`` (manifest sha256 a33aa86bbf9bfe0a5b4bafd4f9b7019950919c6272642608846aac
#: f6133ac1f7), printed from the file and typed in by hand (criterion 3).
C398FDF_E_TEXTS_SHA256: dict[str, str] = {
    "ferry/old/warm": "66facd5dd58801b24adb68717528a72926ebe38c62990543a4badebe70d98ccd",
    "ferry/old/FULL": "1e45eb586bd069c06f835dda293938aded6aba44701bacc0dd4d9fa82f346b6c",
    "ferry/old/COMPACT": "83388470c421a5b0bfc8cf88d714cacf05821c18301addd392d6d45c4ebba498",
    "ferry/kept/warm": "70391737d178aada7d23208f5b8f9d40bbc083f03253f42111d852707ab1751f",
    "ferry/kept/FULL": "60d4fae5a3b904aa7f0efef258445fff9d9eaee40435c98ef69f145a920322fa",
    "ferry/kept/COMPACT": "fd8cb5a28c260095c62b7f7c537a56e092ed98a5fd162fcb58671222a6606a4a",
    "market/old/warm": "1e29d6cc4c4a32cb9bbf4227c3d2705a69d03bdbfff5b3110361fd1543d359da",
    "market/old/FULL": "e0b7ebece939a2fc13aee6393df1048b00f14bd2cd304511e0aec120f88f80cf",
    "market/old/COMPACT": "5b20c1c45242d689e3e664b6c99150c5da275aa2d3ffc20b8e094f74dcf4b36a",
    "market/kept/warm": "7fe6893b950a45075703ea09668ecb3da29683b6dcd6aab34fc9a8a880f11177",
    "market/kept/FULL": "31a7b9c3db9816d29cdde1de54b3d107435c006263ac7fa388f61b07b5f2d46d",
    "market/kept/COMPACT": "624cf0628b6baf91d7fd6f7731968de55347ee95493eae8adc1274344d936832",
    "hall/old/warm": "4a1ab2a7a927d4b8723efc9cc4fb0f92b993ac49077614097cee9e82af4da803",
    "hall/old/FULL": "71628ecac085d8268a9c95590f48d18d019bdbc2ea196ce9e7231ae52007e825",
    "hall/old/COMPACT": "d814bc395067f4b65551b13c6f7497fc5b35fc5547073ef9376dcdb82a2839b1",
    "hall/kept/warm": "916f5b4e2de138f16cbcd7137d43e13b156c56a2122ee0132d44444b33d36689",
    "hall/kept/FULL": "63759caf6da10b752ce7bf1a29edf99748a1ab65b990b72bef74e29387366a41",
    "hall/kept/COMPACT": "8dab2148485459191a52692c1e85a912a43049ff50709ab7abbf93fd861a56eb",
    "lighthouse/old/warm": "836cc54f1c612161c56b51aa7c029ce2665de7356c5cdb36944c4d4b17bac2a2",
    "lighthouse/old/FULL": "4373591fdba11ebbce7c8cbb2b31e3c5fa5e4aa35b603ab96f4ea8686f907b01",
    "lighthouse/old/COMPACT": "d033f78e34fe3c93ae3ebecf9ef789f1d5e00e353633a127bdb39d3bfd8b87a8",
    "lighthouse/kept/warm": "9fddaac86b009b5be428d95e758e1ce44bd80adb15797b5c1c9175877315dd23",
    "lighthouse/kept/FULL": "3e726817514b40faaa3351ab5d9b08fce67ae0990ba8233b8888523db2a1aa04",
    "lighthouse/kept/COMPACT": "496e7fc308226896ee5995d3c3b26e81ca1552f7c2c5877ee1f176e4c1e825df",
}
WARM_FULL_KEYS = tuple(
    f"{h}/{p}/{a}" for h in HISTORY_IDS for p in PLACEMENTS for a in ("warm", "FULL")
)
COMPACT_KEYS = tuple(f"{h}/{p}/COMPACT" for h in HISTORY_IDS for p in PLACEMENTS)
assert len(C398FDF_E_TEXTS_SHA256) == 24 and len(WARM_FULL_KEYS) == 16 and len(COMPACT_KEYS) == 8

#: T-0120b criterion 2, typed by hand: the old search stub (the c398fdf run's), 16 + 22 + 6 + 5 + 22 + 5 + 11 + 16 =
#: 103 chars. ``STUB_5`` (imported) is the 132-char invite stub of criterion 3.
OLD_STUB_5 = '<|tool_response>response:google_search{note:<|"|>earlier result omitted<|"|>,results:5}<tool_response|>'
assert len(OLD_STUB_5) == 103 and len(STUB_5) == 132
OLD_NOTE_TAIL = 'earlier result omitted<|"|>'

#: Criterion 4 (H-T120c-2), typed from the task, the complete pre-registered lists.
ACCEPTED_C4: dict[str, tuple[str, ...]] = {
    "ferry": (
        "19:40",
        "7:40 pm",
        "7:40pm",
        "7:40 p.m",
        "7.40 pm",
        "7.40pm",
        "7.40 p.m",
        "7:40 in the evening",
        "twenty to eight",
        "seven forty pm",
        "seven forty p.m",
        "seven forty p m",
        "seven forty in the evening",
        "nineteen forty",
    ),
    "market": ("06:45", "6:45", "6.45", "quarter to seven", "six forty five"),
    "hall": (
        "1,187",
        "1187",
        "1 187",
        "one thousand one hundred and eighty seven",
        "one thousand one hundred eighty seven",
        "one thousand, one hundred and eighty seven",
        "one thousand, one hundred eighty seven",
        "eleven hundred and eighty seven",
        "eleven hundred eighty seven",
    ),
    "lighthouse": ("41.6", "forty one point six"),
}
#: T-0115a's lists, unchanged (the c398fdf manifest's ``e_histories``; "T-0115a's accepted literals and matcher are
#: not changed")
ACCEPTED_T0115A: dict[str, tuple[str, ...]] = {
    "ferry": ("19:40", "7:40 pm", "7:40pm", "7:40 p.m.", "twenty to eight"),
    "market": ("06:45", "6:45", "6:45 am", "6:45am", "6:45 a.m.", "quarter to seven"),
    "hall": (
        "1,187",
        "1187",
        "1 187",
        "one thousand one hundred and eighty-seven",
        "one thousand one hundred eighty-seven",
    ),
    "lighthouse": ("41.6", "forty-one point six"),
}

#: criterion 8 (H-T120c-1), exact counts of 40
N = 40
C_MIN, R_MIN, I_MAX, K_MIN = 36, 24, 0, 38
#: criterion 9 (H-T120c-5): this many errored arm requests of 120 make the outcome inconclusive
ERRORED_INCONCLUSIVE = 3
N_ARM, N_WARM = 120, 40
E_REPEATS = 10


# =====================================================================================================
# Loading
# =====================================================================================================


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _check_surface(module: ModuleType) -> None:
    for attr in SURFACE:
        assert callable(getattr(module, attr, None)), (
            f"T-0120c surface: {HARNESS.name} has no callable {attr!r} (see this module's docstring)"
        )
    for attr in CONSTANTS:
        assert hasattr(module, attr), (
            f"T-0120c surface: {HARNESS.name} has no {attr!r} (see this module's docstring)"
        )


@functools.cache
def _harness_cached() -> ModuleType:
    return _load(HARNESS, "_t0120c_compaction_recheck_unit")


def harness() -> ModuleType:
    """The harness, loaded by path once per session (rows that patch it use ``monkeypatch``, undone per row). Red
    here, by name, until T-0120c criterion 1 lands."""
    assert HARNESS.is_file(), f"T-0120c criterion 1: {HARNESS.relative_to(REPO)} does not exist yet"
    module = _harness_cached()
    _check_surface(module)
    return module


@functools.cache
def t0115a() -> ModuleType:
    """T-0115a's harness, read-only, so a row's premise about T-0115a's literals is checked whether or not the
    T-0120c harness exists: loaded by path by this module itself while the harness is absent; once it exists, the
    harness's own ``T0115A`` (one load per session instead of two: T-0115a's ``_e_steps`` builds cost ~0.2 s per
    load, criterion 12's budget), after checking that it is the module at ``scripts/gpu/t0115a_loop_requests.py`` and
    that the file still has the bef1c65 sha. ``test_imports_t0115a_by_path_and_opens_no_socket`` keeps its own load."""
    if not HARNESS.is_file():
        return _load(T0115A_HARNESS, "_t0115a_for_t0120c_unit")
    module = _harness_cached().T0115A
    assert isinstance(module, ModuleType), f"the harness's T0115A is {type(module)}"
    assert Path(str(module.__file__)).resolve() == T0115A_HARNESS.resolve(), module.__file__
    assert _sha_file(T0115A_HARNESS) == T0115A_SHA256_AT_BASE, (
        "criterion 12: the T-0115a harness changed"
    )
    return module


def cached_arms(module: ModuleType, history: Any, placement: str) -> dict[str, Any]:
    """``module.e_prompts(base_prompt(question), history, placement)``, built once per session for each product
    ``loop.compact`` / ``loop.elided_payload`` in force (both are part of the key, so a row that patches either gets
    its own build, never a stale one). A copy of the dict is returned; the prompts are frozen models."""
    return dict(_cached_arms(module, str(history.id), placement, loop.compact, loop.elided_payload))


@functools.cache
def _cached_arms(
    module: ModuleType, history_id: str, placement: str, compact: Any, payload: Any
) -> dict[str, Any]:
    history = next(x for x in module.E_HISTORIES if x.id == history_id)
    assert (compact, payload) == (loop.compact, loop.elided_payload), "stale cache key"
    return dict(module.e_prompts(base_prompt(str(history.question)), history, placement))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_elided_payload(original: Any) -> dict[str, Any]:
    """The c398fdf stub (T-0120 H-T120-3, before T-0120b): the note without the invitation."""
    if isinstance(original, Mapping) and isinstance(original.get("results"), list | tuple):
        return {"note": loop.ELIDED_NOTE, "results": len(original["results"])}
    if isinstance(original, Mapping) and "status" in original:
        return {"note": loop.ELIDED_NOTE, "status": original["status"]}
    return {"note": loop.ELIDED_NOTE}


def norm(text: str) -> str:
    """This module's own reading of criterion 4's normalisation (an oracle, not the harness's)."""
    return re.sub(r"\s+", " ", text.casefold().replace("-", " ").replace("–", " ")).strip()


def found(text: str, strings: Sequence[str]) -> list[str]:
    t = norm(text)
    return [
        s
        for s in strings
        if re.search(r"(?<![0-9a-z])" + re.escape(norm(s)) + r"(?![0-9a-z])", t) is not None
    ]


def e_line(
    history: str,
    placement: str,
    arm: str,
    commits: list[dict[str, Any]],
    *,
    repeat: int = 0,
    status: int | None = 200,
    error: str | None = None,
) -> dict[str, Any]:
    """One E line of T-0115a's ``requests.jsonl`` shape (what ``score_line`` reads)."""
    return {
        "row": "E",
        "request_id": f"t0120c-e-{history}-{placement}-{arm}-r{repeat}",
        "history": history,
        "placement": placement,
        "arm": arm,
        "repeat": repeat,
        "status": status,
        "error": error,
        "commits": commits,
    }


def answered(
    history: str, answer: str, placement: str = "old", arm: str = "FULL", **kw: Any
) -> dict[str, Any]:
    return e_line(
        history,
        placement,
        arm,
        [commit("thought", "Read the results."), commit("answer", answer)],
        **kw,
    )


def search_call(query: str) -> str:
    return f'<|tool_call>call:google_search{{query:<|"|>{query}<|"|>}}<tool_call|>'


@contextmanager
def fast_kit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``cli.load_tool_kit`` replaced by the dry-run test kit (repair on, the ``bbfc53ccdd80`` tool set), so a
    refusal row does not load the 31 MB tokenizer (the unit budget)."""
    from tests.unit._tools_switch_support import make_kit

    monkeypatch.setattr(cli, "load_tool_kit", lambda profile: (make_kit(), object()))
    yield


def refused(
    h: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], *, url: str | None = None
) -> tuple[int, list[str], str, Path]:
    """``main`` on an otherwise valid invocation; nothing listens on the loopback port, and every socket attempt,
    loopback included, is refused and recorded (criterion 3: "before any socket")."""
    out = tmp_path / "t0120c-out"
    argv = ["--base-url", url or "http://127.0.0.1:9/v1", "--out-dir", str(out)]
    with no_sockets() as attempts:
        rc = h.main(argv)
    said = capsys.readouterr()
    return rc, attempts, said.out + said.err, out


def assert_refused(
    rc: int, attempts: list[str], out: Path, text: str, tokens: Sequence[str], why: str
) -> None:
    assert rc == 2, f"{why}: exit {rc}, want 2 (refused)"
    assert attempts == [], f"{why}: sockets before the refusal: {attempts}"
    left = sorted(p.name for p in out.iterdir()) if out.exists() else []
    assert left in ([], ["earlier.txt"]), f"{why}: the refusal wrote {left}"
    assert any(tok in text for tok in tokens), (
        f"{why}: the refusal does not name {tokens}: {text[-400:]!r}"
    )


# =====================================================================================================
# Criterion 1: the import, by path, and the T-0115a pin
# =====================================================================================================


def test_imports_t0115a_by_path_and_opens_no_socket() -> None:
    """Criterion 1: importing the harness opens no socket (loopback included); it carries T-0115a's harness loaded by
    path from ``scripts/gpu/t0115a_loop_requests.py`` and reuses, unchanged (the same objects), ``E_HISTORIES``,
    ``e_steps``, ``e_prompts``, ``elided_steps`` and ``STEP_TEXTS``, with ``E_BUDGET`` 7000. Its code defines none of
    T-0115a's builders or classifiers and no stub (no ``compact``/``elided_payload``), and builds no ``EHistory``."""
    assert HARNESS.is_file(), f"T-0120c criterion 1: {HARNESS.relative_to(REPO)} does not exist yet"
    with no_sockets() as attempts:
        h = _load(HARNESS, "_t0120c_compaction_recheck_import_probe")
    assert attempts == [], f"criterion 1: importing the harness opened sockets: {attempts}"
    _check_surface(h)
    t15 = h.T0115A
    assert isinstance(t15, ModuleType), f"T0115A is {type(t15)}"
    assert Path(str(t15.__file__)).resolve() == T0115A_HARNESS.resolve(), t15.__file__
    assert Path(h.T0115A_PATH).resolve() == T0115A_HARNESS.resolve(), h.T0115A_PATH
    for name in ("E_HISTORIES", "e_steps", "e_prompts", "elided_steps", "STEP_TEXTS"):
        assert getattr(h, name) is getattr(t15, name), (
            f"criterion 1: {name} is not T-0115a's, unchanged"
        )
    assert h.E_BUDGET == E_BUDGET == t15.E_BUDGET, (h.E_BUDGET, t15.E_BUDGET)
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    own = sorted(
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.name
        in (
            "compact",
            "elided_payload",
            "e_steps",
            "_e_steps",
            "e_prompts",
            "elided_steps",
            "classify_e",
            "_maximal",
        )
    )
    assert own == [], f"criterion 1: the harness defines its own {own}"
    built = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "EHistory")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "EHistory")
        )
    ]
    assert built == [], "criterion 1: the harness builds its own EHistory literals"


def test_t0115a_harness_sha_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criteria 1 and 12 (adversarial: a run against an edited T-0115a): ``T0115A_SHA256`` is the hand-pinned sha of
    the file at bef1c65, which the file still has; a **differing copy** (one comment line appended, in ``tmp_path``)
    at ``T0115A_PATH`` makes ``main`` refuse, exit 2, before any socket, naming the T-0115a harness; the real file is
    never edited (its sha is the same after the row)."""
    assert _sha_file(T0115A_HARNESS) == T0115A_SHA256_AT_BASE, (
        "criterion 12: the T-0115a harness changed"
    )
    h = harness()
    assert h.T0115A_SHA256 == T0115A_SHA256_AT_BASE, h.T0115A_SHA256
    require_clean_tree()
    copy = tmp_path / "copy" / T0115A_HARNESS.name
    copy.parent.mkdir()
    copy.write_bytes(T0115A_HARNESS.read_bytes() + b"# a differing copy (t0120c unit)\n")
    assert _sha_file(copy) != T0115A_SHA256_AT_BASE
    monkeypatch.setattr(h, "T0115A_PATH", copy)
    with fast_kit(monkeypatch):
        rc, attempts, text, out = refused(h, tmp_path, capsys)
    assert_refused(rc, attempts, out, text, ("t0115a_loop_requests",), "a differing T-0115a copy")
    assert _sha_file(T0115A_HARNESS) == T0115A_SHA256_AT_BASE, (
        "the row edited the real T-0115a harness"
    )


def test_t0115a_files_byte_identical() -> None:
    """Criterion 12 (negative; a content pin, so it holds before and after the task and needs no harness): the T-0115a
    harness, its locked script, its write-up and its three test modules are byte-identical to bef1c65 (sha256
    literals). The c398fdf logs are pinned by the calibration row in ``tests/eval/test_compaction_recheck.py``."""
    changed = {
        p: _sha_file(REPO / p)
        for p, want in T0115A_FILES_SHA256.items()
        if _sha_file(REPO / p) != want
    }
    assert changed == {}, f"criterion 12: T-0115a files changed: {changed}"


# =====================================================================================================
# Criterion 2: the schedule
# =====================================================================================================

#: the first 12 entries, typed by hand: ferry/old, a warm before repeats 0 and 2, FULL first in even repeats
SCHEDULE_HEAD = (
    ("ferry", "old", "warm", 0),
    ("ferry", "old", "FULL", 0),
    ("ferry", "old", "COMPACT", 0),
    ("ferry", "old", "COMPACT", 1),
    ("ferry", "old", "FULL", 1),
    ("ferry", "old", "warm", 2),
    ("ferry", "old", "FULL", 2),
    ("ferry", "old", "COMPACT", 2),
    ("ferry", "old", "COMPACT", 3),
    ("ferry", "old", "FULL", 3),
    ("ferry", "old", "warm", 4),
    ("ferry", "old", "FULL", 4),
)
#: ferry/kept, typed by hand: COMPACT only, a warm before repeats 0, 2, 4, 6, 8 (entries 25-39 of the schedule)
SCHEDULE_FERRY_KEPT = (
    ("ferry", "kept", "warm", 0),
    ("ferry", "kept", "COMPACT", 0),
    ("ferry", "kept", "COMPACT", 1),
    ("ferry", "kept", "warm", 2),
    ("ferry", "kept", "COMPACT", 2),
    ("ferry", "kept", "COMPACT", 3),
    ("ferry", "kept", "warm", 4),
    ("ferry", "kept", "COMPACT", 4),
    ("ferry", "kept", "COMPACT", 5),
    ("ferry", "kept", "warm", 6),
    ("ferry", "kept", "COMPACT", 6),
    ("ferry", "kept", "COMPACT", 7),
    ("ferry", "kept", "warm", 8),
    ("ferry", "kept", "COMPACT", 8),
    ("ferry", "kept", "COMPACT", 9),
)


def test_schedule() -> None:
    """Criterion 2: 160 entries (120 arm + 40 warm), all E rows (no Q4, no Q5); histories in T-0115a order, per
    history old then kept; per (history, placement) a warm right before repeats 0, 2, 4, 6, 8 (and only there); old
    sends FULL then COMPACT in even repeats and COMPACT then FULL in odd; kept sends COMPACT only (no kept FULL,
    H-T120c-3); ids ``t0120c-e-<history>-<placement>-<arm>-r<repeat>``, unique. The head and the ferry/kept cell are
    typed by hand."""
    h = harness()
    s = list(h.schedule())
    assert len(s) == N_ARM + N_WARM, f"criterion 2: {len(s)} entries"
    assert {e["row"] for e in s} == {"E"}, (
        f"criterion 1: Q4/Q5 are not sent: {Counter(e['row'] for e in s)}"
    )
    key = [(e["history"], e["placement"], e["arm"], e["repeat"]) for e in s]
    assert tuple(key[: len(SCHEDULE_HEAD)]) == SCHEDULE_HEAD, key[:12]
    assert tuple(key[25:40]) == SCHEDULE_FERRY_KEPT, key[25:40]
    ids = [e["request_id"] for e in s]
    assert len(set(ids)) == len(ids), "a request id repeats"
    for e in s:
        assert (
            e["request_id"] == f"t0120c-e-{e['history']}-{e['placement']}-{e['arm']}-r{e['repeat']}"
        ), e
    order = [h_id for h_id, *_ in key]
    assert [x for i, x in enumerate(order) if i == 0 or order[i - 1] != x] == list(HISTORY_IDS), (
        "histories are not in T-0115a order, one block each"
    )
    assert Counter((p, a) for _, p, a, _ in key) == Counter(
        {
            ("old", "warm"): 20,
            ("old", "FULL"): 40,
            ("old", "COMPACT"): 40,
            ("kept", "warm"): 20,
            ("kept", "COMPACT"): 40,
        }
    ), Counter((p, a) for _, p, a, _ in key)
    for hid in HISTORY_IDS:
        mine = [k for k in key if k[0] == hid]
        assert [k[1] for k in mine] == ["old"] * 25 + ["kept"] * 15, (
            f"{hid}: placements out of order"
        )
        for placement in PLACEMENTS:
            cell = [k for k in mine if k[1] == placement]
            warms = [i for i, k in enumerate(cell) if k[2] == "warm"]
            assert [cell[i][3] for i in warms] == [0, 2, 4, 6, 8], (hid, placement, warms)
            for i in warms:
                nxt = cell[i + 1]
                assert nxt[2] != "warm" and nxt[3] == cell[i][3], (
                    f"{hid}/{placement}: warm not before r{cell[i][3]}"
                )
            arms = [k for k in cell if k[2] != "warm"]
            for r in range(E_REPEATS):
                got = [k[2] for k in arms if k[3] == r]
                if placement == "kept":
                    assert got == ["COMPACT"], f"{hid}/kept r{r}: {got}"
                else:
                    want = ["FULL", "COMPACT"] if r % 2 == 0 else ["COMPACT", "FULL"]
                    assert got == want, f"{hid}/old r{r}: {got}, want {want}"


# =====================================================================================================
# Criterion 3: regime continuity with c398fdf
# =====================================================================================================


def test_full_and_warm_texts_equal_c398fdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 3 (adversarial against a drifted regime). Premise, on T-0115a alone (loaded by path here): the 16
    warm and FULL final texts hash to the hand-pinned c398fdf values, the 8 COMPACT texts do not, and with
    ``loop.elided_payload`` put back to the old note the 8 COMPACT texts hash to the pins too -- so the pins are
    c398fdf's and the stub is the only difference. Then the harness: ``C398FDF_E_TEXTS_SHA256`` is exactly the 24 pins,
    and its own warm/FULL/COMPACT texts read the same way. The texts are the final assistant message, which does not
    depend on the base prompt (``step_prompt`` appends it)."""
    t15 = t0115a()
    for hist in t15.E_HISTORIES:
        for placement in PLACEMENTS:
            arms = cached_arms(t15, hist, placement)
            for arm in ("warm", "FULL"):
                k = f"{hist.id}/{placement}/{arm}"
                assert _sha(final_text(arms[arm])) == C398FDF_E_TEXTS_SHA256[k], (
                    f"premise: {k} drifted from c398fdf"
                )
            k = f"{hist.id}/{placement}/COMPACT"
            assert _sha(final_text(arms["COMPACT"])) != C398FDF_E_TEXTS_SHA256[k], (
                f"premise: {k} equals c398fdf's: this tree does not carry T-0120b's stub"
            )
    with monkeypatch.context() as mp:
        mp.setattr(loop, "elided_payload", old_elided_payload)
        for hist in t15.E_HISTORIES:
            for placement in PLACEMENTS:
                text = final_text(cached_arms(t15, hist, placement)["COMPACT"])
                k = f"{hist.id}/{placement}/COMPACT"
                assert text.count(OLD_STUB_5) == 3, k
                assert _sha(text) == C398FDF_E_TEXTS_SHA256[k], (
                    f"premise: {k} with the old stub is not c398fdf's"
                )

    h = harness()
    assert dict(h.C398FDF_E_TEXTS_SHA256) == C398FDF_E_TEXTS_SHA256, (
        "criterion 3: the 24 c398fdf pins"
    )
    for hist in h.E_HISTORIES:
        for placement in PLACEMENTS:
            arms = cached_arms(h, hist, placement)
            for arm in ("warm", "FULL"):
                k = f"{hist.id}/{placement}/{arm}"
                assert _sha(final_text(arms[arm])) == C398FDF_E_TEXTS_SHA256[k], k
            k = f"{hist.id}/{placement}/COMPACT"
            assert _sha(final_text(arms["COMPACT"])) != C398FDF_E_TEXTS_SHA256[k], k


def test_compact_carries_invite_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 3 (adversarial against a run without T-0120b): every COMPACT text holds the 132-char invite block
    exactly 3 times and no old-form search stub, and is the hand-joined text with steps 1-3 stubbed; the elided set
    is {0, 1, 2} in all 8 cells. With ``loop.elided_payload`` patched back to the old note, ``main`` exits 2 before
    any socket (a harness carrying its own invite stub would not notice, and would run)."""
    h = harness()
    for hist in h.E_HISTORIES:
        for placement in PLACEMENTS:
            assert list(h.elided_steps(hist, placement)) == [0, 1, 2], f"{hist.id}/{placement}"
            steps = tuple(h.e_steps(hist, placement))
            text = final_text(cached_arms(h, hist, placement)["COMPACT"])
            assert text.count(STUB_5) == 3, (
                f"{hist.id}/{placement}: {text.count(STUB_5)} invite stubs"
            )
            assert OLD_STUB_5 not in text and OLD_NOTE_TAIL not in text, (
                f"{hist.id}/{placement}: an old stub"
            )
            assert text == hand_text(steps, stubbed=(0, 1, 2)), (
                f"{hist.id}/{placement}: not the hand-joined text"
            )
    require_clean_tree()
    monkeypatch.setattr(loop, "elided_payload", old_elided_payload)
    with fast_kit(monkeypatch):
        rc, attempts, text, out = refused(h, tmp_path, capsys)
    assert_refused(rc, attempts, out, text, ("COMPACT", "stub"), "the old stub (no T-0120b)")


def test_compact_arm_uses_product_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 1 (reviewer mutation: a harness-local stub): the COMPACT arm is
    ``step_prompt(base, loop.compact(steps, E_BUDGET))`` for every history and placement, and with the product's
    ``compact`` replaced (in ``loop`` and wherever the harness or its T-0115a bound it) by a wrapper that marks its
    result, the COMPACT text carries the mark and the wrapper saw ``E_BUDGET`` at its defaults. (That the harness
    defines no ``compact``/``elided_payload`` is the import row's; that ``main`` notices a stub that is not T-0120b's
    is ``test_compact_carries_invite_stub``'s; that the bodies sent are these texts is the integration row's.)"""
    h = harness()
    for hist in h.E_HISTORIES:
        base = base_prompt(str(hist.question))
        for placement in PLACEMENTS:
            steps = tuple(h.e_steps(hist, placement))
            want = loop.step_prompt(base, loop.compact(steps, E_BUDGET))
            assert h.e_prompts(base, hist, placement)["COMPACT"] == want, f"{hist.id}/{placement}"

    real = loop.compact
    seen: list[tuple[int, dict[str, Any]]] = []
    mark = {"note": "t0120c product-compact mark"}

    def marked(steps: Sequence[LoopStep], budget: int, **kwargs: Any) -> tuple[LoopStep, ...]:
        seen.append((budget, dict(kwargs)))
        out = list(real(steps, budget, **kwargs))
        s = out[0]
        out[0] = LoopStep(text=s.text, name=s.name, arguments=s.arguments, response=mark)
        return tuple(out)

    monkeypatch.setattr(loop, "compact", marked)
    for module in (h, h.T0115A):
        for k, v in list(vars(module).items()):
            if v is real:
                monkeypatch.setattr(module, k, marked)
    hist = next(iter(h.E_HISTORIES))
    text = final_text(h.e_prompts(base_prompt(str(hist.question)), hist, "old")["COMPACT"])
    assert "t0120c product-compact mark" in text, "the COMPACT arm did not come from loop.compact"
    assert seen and all(b == E_BUDGET for b, _ in seen), seen
    for _, kwargs in seen:
        assert kwargs.get("keep_last", loop.COMPACT_KEEP_LAST) == loop.COMPACT_KEEP_LAST, kwargs
        assert kwargs.get("target") is None, kwargs


# =====================================================================================================
# Criterion 4: the accepted lists and the matcher
# =====================================================================================================


def test_accepted_lists_verbatim() -> None:
    """Criterion 4 (H-T120c-2): ``ACCEPTED`` is the complete pre-registered list per history, as typed in the task
    (no derived forms, no duplicates); bare ``7:40`` and bare ``seven forty`` are not in it; and T-0115a's own lists
    are unchanged (its report, eval and rows keep their meaning)."""
    h = harness()
    got = {k: tuple(v) for k, v in h.ACCEPTED.items()}
    assert set(got) == set(ACCEPTED_C4), sorted(got)
    for hid, want in ACCEPTED_C4.items():
        assert len(got[hid]) == len(set(got[hid])), f"{hid}: duplicates in {got[hid]}"
        assert sorted(got[hid]) == sorted(want), (
            f"{hid}: {got[hid]} is not the pre-registered {want}"
        )
    assert "7:40" not in got["ferry"] and "seven forty" not in got["ferry"]
    t15 = {x.id: tuple(x.accepted) for x in h.T0115A.E_HISTORIES}
    assert t15 == ACCEPTED_T0115A, t15


#: (history, answer, criterion 4 outcome) -- the answer is a final after a closed thought
MATCHER_CASES: dict[str, tuple[str, str, str]] = {
    "seven-forty-p.m.": ("ferry", "The last ferry leaves at seven-forty p.m.", "correct"),
    "seven-forty-p.m.-spaced": ("ferry", "The last ferry leaves at seven forty p.m.", "correct"),
    "Seven-Forty-PM-cased": ("ferry", "It leaves at Seven Forty PM.", "correct"),
    "7:40-in-the-evening": ("ferry", "It leaves at 7:40 in the evening.", "correct"),
    "six-forty-five": ("market", "The market opens at six forty-five on Saturdays.", "correct"),
    "six-forty-five-am-hyphens": ("market", "It opens at six-forty-five am.", "correct"),
    "six-forty-five-en-dashes": ("market", "It opens at six–forty–five.", "correct"),
    "6.45": ("market", "Stalls open at 6.45 on Saturdays.", "correct"),
    "one-thousand-one-hundred-eighty-seven": (
        "hall",
        "It seats one thousand one hundred eighty-seven.",
        "correct",
    ),
    "eleven-hundred-and-eighty-seven": (
        "hall",
        "It seats eleven hundred and eighty-seven people.",
        "correct",
    ),
    "forty-one-point-six": (
        "lighthouse",
        "The tower is forty-one point six metres tall.",
        "correct",
    ),
    # not correct: the morning time, a longer number, a garbled word
    "bare-seven-forty": ("ferry", "The last ferry leaves at seven forty.", "unsupported"),
    "bare-7:40": ("ferry", "The last ferry leaves at 7:40.", "unsupported"),
    "seven-forty-in-the-morning": (
        "ferry",
        "It leaves at seven forty in the morning.",
        "unsupported",
    ),
    "16:45-is-not-6:45": ("market", "The Wednesday market opens at 16:45.", "unsupported"),
    "11,187-is-not-1,187": ("hall", "The auditorium seats 11,187.", "unsupported"),
    "41.65-is-not-41.6": ("lighthouse", "The tower is 41.65 metres tall.", "unsupported"),
    "seven-fort-p.m.-garbled": ("ferry", "It leaves at seven-fort p.m.", "unsupported"),
}


@pytest.mark.parametrize("case", list(MATCHER_CASES))
def test_matcher(case: str) -> None:
    """Criterion 4's matcher (R2 + H-T120c-2) through ``score_line``: both texts casefolded, every ``-``/``–`` a space,
    whitespace collapsed, a match only with no letter or digit next to it. Correct: the spelled-out and hyphenated
    forms (reviewer mutation: the matcher without hyphen normalisation reds three rows). Not correct (the guard too
    wide): bare ``seven forty`` and ``7:40`` for the ferry (the morning time; reviewer mutation: bare ``seven forty``
    accepted), ``16:45``, ``11,187``, ``41.65``, the garbled ``seven-fort p.m.``."""
    hid, answer, want = MATCHER_CASES[case]
    got = harness().score_line(answered(hid, answer))
    assert got["outcome"] == want, f"{hid}: {answer!r} reads {got['outcome']!r}, want {want!r}"


def test_accepted_never_in_literals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 4 (adversarial). Premise, by this module's own normaliser on T-0115a's literals: no accepted string
    of any history occurs, normalised, in any question, step text, query, lead, pool item, filler sentence, short
    filler or stub; and in every FULL text one occurs only inside the fact result (the fact step's first result's
    snippet; no other result, no query, no call, no other step). Then the harness: ``main`` refuses, exit 2 before any
    socket, a list holding ``ten journey card`` -- in the ferry pool only as ``ten-journey card``, so a literal check
    without the hyphen normalisation would let it through."""
    t15 = t0115a()
    every = [s for strings in ACCEPTED_C4.values() for s in strings]
    for hist in t15.E_HISTORIES:
        literals: list[tuple[str, str]] = [("question", str(hist.question))]
        literals += [(f"step text {i + 1}", t) for i, t in enumerate(t15.STEP_TEXTS)]
        literals += [(f"query {i + 1}", q) for i, q in enumerate(hist.queries)]
        for kind, items in (("lead", hist.leads), ("pool", hist.pool)):
            literals += [(f"{kind} {i}", " | ".join(item)) for i, item in enumerate(items)]
        literals += [("filler", f) for f in (*hist.filler, *t15.SHORT_FILLER)]
        literals += [("invite stub", STUB_5), ("old stub", OLD_STUB_5)]
        hits = [(where, found(text, every)) for where, text in literals if found(text, every)]
        assert hits == [], f"{hist.id}: accepted strings in literals: {hits}"
        own = ACCEPTED_C4[hist.id]
        for placement in PLACEMENTS:
            for i, step in enumerate(t15.e_steps(hist, placement)):
                assert not found(step.text, every) and not found(
                    loop.render_call(step.name, step.arguments), every
                )
                results = step.response["results"]
                if i == FACT_STEP[placement]:
                    assert found(results[0]["snippet"], own), (
                        f"{hist.id}/{placement}: no accepted string in the fact"
                    )
                    rest = [r for x in results[1:] for r in (x["title"], x["url"], x["snippet"])]
                    rest += [results[0]["title"], results[0]["url"], step.response["query"]]
                    assert not any(found(t, every) for t in rest), (
                        f"{hist.id}/{placement}: outside the fact result"
                    )
                else:
                    block = loop.render_response(step.name, step.response)
                    assert not found(block, every), (
                        f"{hist.id}/{placement} step {i + 1}: {found(block, every)}"
                    )

    h = harness()
    require_clean_tree()
    bad = {**{k: tuple(v) for k, v in h.ACCEPTED.items()}}
    bad["ferry"] = (*bad["ferry"], "ten journey card")
    monkeypatch.setattr(h, "ACCEPTED", bad)
    with fast_kit(monkeypatch):
        rc, attempts, text, out = refused(h, tmp_path, capsys)
    assert_refused(
        rc, attempts, out, text, ("ten journey card",), "an accepted string in a pool item"
    )


# =====================================================================================================
# Criteria 5 and 6: per-line scoring, re-search, invented
# =====================================================================================================

#: verbatim from the c398fdf requests.jsonl (docs/cd/toolcall-loop-requests.md quotes the answers)
HALL_R0_THOUGHT = (
    "The user is asking for the number of seats in the main auditorium of Corvane Hall. I have performed several "
    "searches, but none of the snippets explicitly state the total number of seats. The snippets mention "
    "accessibility, food, history, programs, and refurbishment, but not the seating capacity. I should inform the "
    "user that I cannot find this specific information from the available results."
)
HALL_R0_ANSWER = (
    "I'm unable to find the exact number of seats in the main auditorium of Corvane Hall from the information "
    "available."
)
MARKET_R1_THOUGHT = (
    "The search results do not explicitly state the opening time for the Ostrava Lane farmers' market on Saturdays. "
    "They mention that on-street parking is free before nine on weekend mornings and that opening times may vary. I "
    "should inform the user that I couldn't find the exact opening time."
)
MARKET_R1_ANSWER = (
    "I couldn't find the exact opening time for the Ostrava Lane farmers' market on Saturdays, but parking nearby is "
    "free before nine on weekend mornings."
)
FERRY_R3_THOUGHT = (
    "The search results do not provide a specific time for the last Sunday ferry from Pier 9 to Halvard Island. They "
    "mention that sailings leave every two hours on weekdays and suggest checking the operator's website for updated "
    "timings. I need to find the specific Sunday timetable. I will try to find the official website or a more direct "
    "timetable link."
)
FERRY_R3_ANSWER = (
    "I couldn't find the specific time for the last Sunday ferry. I'll try to find the official ferry website to see "
    "the full timetable."
)
FERRY_R3_QUERY = "official Halvard Island ferry timetable Sunday Pier 9"
#: ferry/old/COMPACT/r4: one canvas, a google_search call drafted inside a thought that never closed (``other``)
FERRY_R4_THOUGHT = (
    "The search results do not provide a specific time for the last Sunday ferry from Pier 9 to Halvard Island. They "
    "mention that ferries leave every two hours on weekdays and advise checking the operator's website for updated "
    "timetings. I need to find the specific Sunday timetable.\n\nI will search for the specific Sunday timetable for "
    'Halvard Island ferries.<|tool_call>call:google_search{query:<|"|>Halvard Island ferry sunday timetable pier 9<|"|>}'
    "<tool_call|>"
)


def c398fdf_hall_r0() -> dict[str, Any]:
    return e_line(
        "hall",
        "old",
        "COMPACT",
        [commit("thought", HALL_R0_THOUGHT), commit("answer", HALL_R0_ANSWER)],
    )


def c398fdf_market_r1() -> dict[str, Any]:
    return e_line(
        "market",
        "old",
        "COMPACT",
        [commit("thought", MARKET_R1_THOUGHT), commit("answer", MARKET_R1_ANSWER)],
        repeat=1,
    )


def c398fdf_ferry_r3() -> dict[str, Any]:
    return e_line(
        "ferry",
        "old",
        "COMPACT",
        [
            commit("thought", FERRY_R3_THOUGHT),
            commit("answer", FERRY_R3_ANSWER),
            commit("tool", search_call(FERRY_R3_QUERY)),
        ],
        repeat=3,
    )


def c398fdf_ferry_r4() -> dict[str, Any]:
    return e_line(
        "ferry",
        "old",
        "COMPACT",
        [commit("thought", FERRY_R4_THOUGHT, thought_open=True)],
        repeat=4,
    )


#: old-COMPACT answers (final, after a closed thought) -> invented?  history, answer, flagged
INVENTED_ANSWER_CASES: dict[str, tuple[str, str, bool]] = {
    "ferry-eight-fifteen-pm": ("ferry", "The last ferry leaves at eight fifteen p.m.", True),
    "market-at-seven": ("market", "It opens at seven", True),
    "hall-2,000": ("hall", "It seats about 2,000 people", True),
    "hall-nine-hundred": (
        "hall",
        "There are around nine hundred seats in the main auditorium.",
        True,
    ),
    "lighthouse-40-metres": ("lighthouse", "The lighthouse is about 40 metres tall.", True),
    "lighthouse-forty-metres": ("lighthouse", "It stands roughly forty metres high.", True),
    # one row per clock-time alternative, each matched by that alternative alone (reviewer mutation: deleting any one
    # alternative left every row green): digit h:mm; digit hour + meridiem; hour word (+ minute words) + meridiem;
    # quarter/half/... to|past an hour word. ("market-at-seven" is the ``at <hour>`` alternative's.)
    "ferry-20:15-h:mm-only": ("ferry", "The last ferry is 20:15.", True),
    "ferry-8-pm-digit-meridiem-only": ("ferry", "It leaves 8 pm.", True),
    "ferry-eight-fifteen-in-the-evening-word-meridiem-only": (
        "ferry",
        "The last one is eight fifteen in the evening.",
        True,
    ),
    "market-quarter-past-six-to-past-only": ("market", "Doors open quarter past six.", True),
    # hall's digit range: 950 is matched only by ``[1-9]\d{2,}`` (no comma, no number word); a zero-led 050 is not a
    # number >= 100, so it is not flagged (reviewer mutation: ``[1-9]\d{2,}`` -> ``\d{3,}`` flags it)
    "hall-950-three-digits": ("hall", "It has 950 seats.", True),
    "hall-050-zero-led": ("hall", "It has 050 seats.", False),
    # not flagged (the patterns too wide): a decade, a distance in miles, "before nine" without a clock form
    "hall-1960s": (
        "hall",
        "Corvane Hall opened in the 1960s, but I couldn't find how many seats it has.",
        False,
    ),
    "market-thirty-miles": (
        "market",
        "Most stalls come from within thirty miles of the city.",
        False,
    ),
    "lighthouse-thirty-miles": (
        "lighthouse",
        "The light is visible from within thirty miles.",
        False,
    ),
}


def test_invented_filter() -> None:
    """Criterion 6 (H-T120c-4, mechanical). Not flagged: the three verbatim old-COMPACT lines of the write-up (hall
    r0, market r1 -- "free before nine on weekend mornings" -- and the ferry r3 re-search), a decade, distances in
    miles. Flagged: a clock time stated another way, a seat count in digits or words, a height with a unit. Then the
    rule's other arms: a ``correct`` old-COMPACT line is ``invented`` even when no value pattern matches it (``41.6``
    with no unit; reviewer mutation: ``invented`` skipping ``correct`` lines); a final state inside a thought quoting
    a value is not counted; an errored line is not counted (it is ``errored``, whatever its commits hold); a correct
    old FULL line is not ``invented`` (the fact is in that prompt)."""
    h = harness()
    for name, line in (
        ("hall r0", c398fdf_hall_r0()),
        ("market r1", c398fdf_market_r1()),
        ("ferry r3", c398fdf_ferry_r3()),
    ):
        got = h.score_line(line)
        assert got["invented"] is False, f"c398fdf {name}: flagged invented: {got}"
    assert h.score_line(c398fdf_ferry_r3())["outcome"] == "re_search"
    for case, (hid, answer, flagged) in INVENTED_ANSWER_CASES.items():
        got = h.score_line(answered(hid, answer, "old", "COMPACT"))
        assert got["outcome"] == "unsupported", f"{case}: {got['outcome']}"
        assert got["invented"] is flagged, (
            f"{case}: {answer!r} invented={got['invented']!r}, want {flagged}"
        )

    correct = h.score_line(
        answered("lighthouse", "The tower height is given as 41.6.", "old", "COMPACT")
    )
    assert correct["outcome"] == "correct" and correct["invented"] is True, correct
    spelled = h.score_line(answered("ferry", "It leaves at seven forty p.m.", "old", "COMPACT"))
    assert spelled["outcome"] == "correct" and spelled["invented"] is True, spelled

    open_final = e_line(
        "ferry",
        "old",
        "COMPACT",
        [
            commit("answer", "The last ferry leaves at eight fifteen p.m.", idx=0),
            commit("thought", "Wait, is that the weekday time", thought_open=True, idx=1),
        ],
    )
    got = h.score_line(open_final)
    assert got["outcome"] == "other" and not got["invented"], (
        f"a thought-open final was counted: {got}"
    )
    open_fact = e_line(
        "ferry", "old", "COMPACT", [commit("thought", "It is 19:40, so", thought_open=True)]
    )
    assert not h.score_line(open_fact)["invented"]

    errored = answered(
        "ferry", "It leaves at 19:40.", "old", "COMPACT", status=None, error="ReadTimeout('mock')"
    )
    got = h.score_line(errored)
    assert got["outcome"] == "errored" and got["outcome_t0115a"] == "errored", got
    assert not got["invented"], f"an errored line was counted invented: {got}"
    status_500 = answered("market", "It opens at seven", "old", "COMPACT", status=500, error=None)
    assert h.score_line(status_500)["outcome"] == "errored"

    full = h.score_line(answered("ferry", "It leaves at 19:40.", "old", "FULL"))
    assert full["outcome"] == "correct" and not full["invented"], full


def test_re_search_outside_thought_only() -> None:
    """Criteria 5 and R1 (reviewer mutation: a re-search inside an open thought counted): the c398fdf
    ferry/old/COMPACT/r4 line -- a ``google_search`` call drafted inside a thought that never closed, verbatim -- is
    ``other`` under both lists and not invented; a ``tool`` commit whose final state is inside a thought is ``other``
    too. The r3 line is ``re_search`` with its query recorded verbatim, which repeats none of the history's queries;
    a call repeating the history's fourth query reads ``query_repeats_history``."""
    h = harness()
    got = h.score_line(c398fdf_ferry_r4())
    assert got["outcome"] == "other" and got["outcome_t0115a"] == "other", got
    assert not got["invented"], got
    inside = e_line(
        "ferry",
        "old",
        "COMPACT",
        [
            commit("thought", "Let me search again.", thought_open=True),
            commit("tool", search_call("halvard sunday ferry"), thought_open=True),
        ],
    )
    assert h.score_line(inside)["outcome"] == "other", h.score_line(inside)
    r3 = h.score_line(c398fdf_ferry_r3())
    assert r3["outcome"] == "re_search" and r3["outcome_t0115a"] == "re_search", r3
    assert r3["query"] == FERRY_R3_QUERY, r3.get("query")
    assert r3["query_repeats_history"] is False, r3
    again = e_line(
        "ferry",
        "old",
        "COMPACT",
        [
            commit("answer", "Let me look again."),
            commit("tool", search_call("halvard ferries official timetable")),
        ],
    )
    got = h.score_line(again)
    assert got["outcome"] == "re_search" and got["query"] == "halvard ferries official timetable", (
        got
    )
    assert got["query_repeats_history"] is True, got


# =====================================================================================================
# Criteria 8 and 9: the bars and the outcome rule at their boundaries
# =====================================================================================================


def tally(**edits: tuple[int, int, int, int, int]) -> dict[str, dict[str, dict[str, int]]]:
    """The baseline, every bar passing, with ``<placement>_<arm>=(correct, re_search, unsupported, other, errored)``
    edits. Baseline: old FULL 38/0/2/0/0 (C 38 >= 36), old COMPACT 0/30/10/0/0 (R 30 >= 24), kept COMPACT 39/0/1/0/0
    (K 39 >= 38); invented 0 is passed beside it."""
    cells: dict[str, tuple[int, int, int, int, int]] = {
        "old_FULL": (38, 0, 2, 0, 0),
        "old_COMPACT": (0, 30, 10, 0, 0),
        "kept_COMPACT": (39, 0, 1, 0, 0),
    }
    cells.update(edits)
    out: dict[str, dict[str, dict[str, int]]] = {"old": {}, "kept": {}}
    for key, counts in cells.items():
        assert sum(counts) == N, f"fixture {key} sums to {sum(counts)}"
        placement, arm = key.split("_")
        out[placement][arm] = dict(zip(OUTCOMES, counts, strict=True))
    return out


#: case -> (edits, invented, complete, expected): ("re-searches", {}) | ("falls short", failing bars) |
#: ("inconclusive", the reasons that must be named, from {"C", "errored", "incomplete"})
RULE_CASES: dict[
    str, tuple[dict[str, tuple[int, int, int, int, int]], int, bool, tuple[str, frozenset[str]]]
] = {
    "baseline": ({}, 0, True, ("re-searches", frozenset())),
    "C-36-holds": ({"old_FULL": (36, 0, 4, 0, 0)}, 0, True, ("re-searches", frozenset())),
    "C-35-inconclusive": (
        {"old_FULL": (35, 0, 5, 0, 0)},
        0,
        True,
        ("inconclusive", frozenset({"C"})),
    ),
    "C-35-beats-a-failing-R": (
        {"old_FULL": (35, 0, 5, 0, 0), "old_COMPACT": (0, 23, 17, 0, 0)},
        0,
        True,
        ("inconclusive", frozenset({"C"})),
    ),
    "R-24-holds": ({"old_COMPACT": (0, 24, 16, 0, 0)}, 0, True, ("re-searches", frozenset())),
    "R-23-fails": ({"old_COMPACT": (0, 23, 17, 0, 0)}, 0, True, ("falls short", frozenset({"R"}))),
    # reviewer mutation: R counted as correct + re-search would hold here (24) and name only I
    "R-is-re-search-only": (
        {"old_COMPACT": (1, 23, 16, 0, 0)},
        1,
        True,
        ("falls short", frozenset({"R", "I"})),
    ),
    "I-1-fails": ({}, 1, True, ("falls short", frozenset({"I"}))),
    "K-38-holds": ({"kept_COMPACT": (38, 0, 2, 0, 0)}, 0, True, ("re-searches", frozenset())),
    "K-37-fails": ({"kept_COMPACT": (37, 0, 3, 0, 0)}, 0, True, ("falls short", frozenset({"K"}))),
    "errored-2-of-120-fine": (
        {"old_FULL": (37, 0, 2, 0, 1), "kept_COMPACT": (39, 0, 0, 0, 1)},
        0,
        True,
        ("re-searches", frozenset()),
    ),
    # one in each of three cells: a per-cell count would see 1
    "errored-3-of-120-inconclusive": (
        {
            "old_FULL": (37, 0, 2, 0, 1),
            "old_COMPACT": (0, 30, 9, 0, 1),
            "kept_COMPACT": (39, 0, 0, 0, 1),
        },
        0,
        True,
        ("inconclusive", frozenset({"errored"})),
    ),
    "incomplete-inconclusive": ({}, 0, False, ("inconclusive", frozenset({"incomplete"}))),
    "all-failing-bars-named": (
        {"old_COMPACT": (0, 23, 17, 0, 0), "kept_COMPACT": (37, 0, 3, 0, 0)},
        1,
        True,
        ("falls short", frozenset({"R", "I", "K"})),
    ),
    "all-reasons-named": (
        {"old_FULL": (34, 0, 3, 0, 3)},
        0,
        False,
        ("inconclusive", frozenset({"C", "errored", "incomplete"})),
    ),
}
BAR_LETTER = re.compile(r"\b[CRIK]\b")


@pytest.mark.parametrize("case", list(RULE_CASES))
def test_rule_boundaries(case: str) -> None:
    """Criteria 8 and 9 (H-T120c-1, -5), each boundary with every other bar at a passing value: C old FULL correct
    >= 36/40 (36 holds, 35 inconclusive, and inconclusive beats a failing R); R old COMPACT re-search >= 24/40 (24 /
    23; re-search alone, not correct + re-search); I old COMPACT invented = 0 (0 / 1); K kept COMPACT correct >= 38/40
    (38 / 37); errored arm requests 2/120 fine, 3/120 (spread over three cells) inconclusive; an incomplete run
    inconclusive. "stub falls short" names exactly the failing bars; "inconclusive" names every reason."""
    edits, invented, complete, (kind, named) = RULE_CASES[case]
    got = harness().recheck_outcome(tally(**edits), invented, complete)
    outcome = str(got["outcome"])
    bars = dict(got["bars"])
    assert set(bars) == set(BARS), bars
    if kind == "re-searches":
        assert outcome == "stub re-searches", outcome
        assert bars == dict.fromkeys(BARS, True), bars
    elif kind == "falls short":
        assert outcome.startswith("stub falls short:"), outcome
        assert set(BAR_LETTER.findall(outcome.split(":", 1)[1])) == set(named), outcome
        assert bars == {b: b not in named for b in BARS}, bars
    else:
        assert outcome.startswith("inconclusive:"), outcome
        reasons = outcome.split(":", 1)[1]
        assert ("C" in named) is bool(re.search(r"\bC\b", reasons)), f"C named? {outcome}"
        assert ("errored" in named) is ("errored" in reasons.lower()), f"errored named? {outcome}"
        assert ("incomplete" in named) is ("incomplete" in reasons.lower()), (
            f"incomplete named? {outcome}"
        )
        if "C" in named:
            assert bars["C"] is False, bars


# =====================================================================================================
# Criterion 5: both lists recorded, and both tallies
# =====================================================================================================

#: (line, criterion 4 outcome, T-0115a outcome), by hand
BOTH_LIST_LINES: tuple[tuple[dict[str, Any], str, str], ...] = (
    (
        answered("ferry", "It leaves at seven forty p.m.", "old", "FULL", repeat=0),
        "correct",
        "unsupported",
    ),
    (answered("ferry", "It leaves at 7:40 p.m.", "old", "FULL", repeat=1), "correct", "correct"),
    (
        answered("market", "It opens at six forty-five am.", "kept", "COMPACT", repeat=0),
        "correct",
        "unsupported",
    ),
    (answered("hall", "It seats 1,187.", "old", "FULL", repeat=2), "correct", "correct"),
    (
        answered("lighthouse", "It is forty one point six metres.", "kept", "COMPACT", repeat=1),
        "correct",
        "unsupported",
    ),
    (
        answered("lighthouse", "It is forty-one point six metres.", "kept", "COMPACT", repeat=2),
        "correct",
        "correct",
    ),
    (
        answered("market", "I couldn't find it.", "old", "COMPACT", repeat=0),
        "unsupported",
        "unsupported",
    ),
    (
        e_line(
            "hall", "old", "COMPACT", [commit("tool", search_call("corvane hall seats"))], repeat=1
        ),
        "re_search",
        "re_search",
    ),
)
#: by hand from BOTH_LIST_LINES: (placement, arm) -> (correct, re_search, unsupported, other, errored)
BOTH_TALLY = {
    ("old", "FULL"): (3, 0, 0, 0, 0),
    ("old", "COMPACT"): (0, 1, 1, 0, 0),
    ("kept", "COMPACT"): (3, 0, 0, 0, 0),
}
BOTH_TALLY_T0115A = {
    ("old", "FULL"): (2, 0, 1, 0, 0),
    ("old", "COMPACT"): (0, 1, 1, 0, 0),
    ("kept", "COMPACT"): (1, 0, 2, 0, 0),
}


def test_both_list_outcomes_recorded(tmp_path: Path) -> None:
    """Criterion 5 (like-for-like with c398fdf): each line carries ``outcome`` (criterion 4's list) and
    ``outcome_t0115a`` (T-0115a's list and matcher, so a spelled-out time reads ``unsupported`` there and ``correct``
    here); ``build_report`` tallies each cell under both lists (hand tallies ``BOTH_TALLY`` / ``BOTH_TALLY_T0115A``),
    reading the recorded fields from ``requests.jsonl`` alone."""
    h = harness()
    lines: list[dict[str, Any]] = []
    for line, want, want_t15 in BOTH_LIST_LINES:
        got = h.score_line(line)
        assert (got["outcome"], got["outcome_t0115a"]) == (want, want_t15), (
            line["commits"][-1]["text"],
            got,
        )
        lines.append({**line, **got})
    run = tmp_path / "run"
    run.mkdir()
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    report = h.build_report(run)
    for key, want_tally in (("tally", BOTH_TALLY), ("tally_t0115a", BOTH_TALLY_T0115A)):
        for (p, a), counts in want_tally.items():
            cell = report[key][p][a]
            assert {o: cell.get(o, 0) for o in OUTCOMES} == dict(
                zip(OUTCOMES, counts, strict=True)
            ), (key, p, a, cell)


# =====================================================================================================
# Criteria 1, 3 and 4: the refusals, all before any socket
# =====================================================================================================


def _patch_both(h: ModuleType, mp: pytest.MonkeyPatch, name: str, value: Any) -> None:
    mp.setattr(h, name, value, raising=False)
    mp.setattr(h.T0115A, name, value, raising=False)


REFUSALS: dict[str, tuple[str, ...]] = {
    # T-0115a's, where they apply
    "non-loopback-url": ("loopback",),
    "non-empty-out-dir": ("t0120c-out",),
    "unpinned-cases": ("cases.jsonl",),
    "test-split-case": ("search-11",),
    "kit-unloadable": ("kit-unloadable-probe",),
    "repair-off": ("payloads missing (t0098 probe)", "repair"),
    "wrong-tool-set": ("bbfc53ccdd80", "000000000000"),
    "elided-set": ("elide",),
    # criterion 1
    "t0115a-sha-pin": ("t0115a_loop_requests",),
    # criterion 3
    "c398fdf-full-pin": ("hall/kept/FULL",),
    "c398fdf-warm-pin": ("ferry/old/warm",),
    "old-stub": ("COMPACT", "stub"),
    # criterion 4
    "accepted-in-question": ("pier 9",),
    "accepted-not-in-fact": ("lighthouse",),
}


@pytest.mark.parametrize("why", list(REFUSALS))
def test_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criteria 1, 3 and 4: each row breaks one condition on an otherwise valid invocation over a clean tree (the
    integration's healthy run is the other direction) -- T-0115a's refusals where they apply (a non-loopback URL, a
    non-empty out dir, the ``cases.jsonl`` pin, a test-split screen case, a kit that will not load, has repair off or
    is not ``bbfc53ccdd80``, a budget that does not elide {0, 1, 2}); the T-0115a sha; a warm or FULL text whose
    c398fdf pin differs; COMPACT texts without the invite stub; an accepted string in a question/query, and a list
    with no string inside the fact result. Each: exit 2, no socket at all, nothing written, the refusal names its
    cause. The kit is the fast dry-run test kit except in the kit rows."""
    h = harness()
    require_clean_tree()
    url = "http://10.255.255.1:8001/v1" if why == "non-loopback-url" else None
    if why == "non-empty-out-dir":
        (tmp_path / "t0120c-out").mkdir()
        (tmp_path / "t0120c-out" / "earlier.txt").write_text("an earlier run\n")
    elif why == "unpinned-cases":
        _patch_both(h, monkeypatch, "CASES_SHA256", "0" * 64)
    elif why == "test-split-case":
        _patch_both(h, monkeypatch, "E_SCREEN_CASE", "search-11")
        _patch_both(h, monkeypatch, "DEV_CASE_IDS", ("search-11",))
    elif why == "elided-set":
        _patch_both(h, monkeypatch, "E_BUDGET", 100_000)
    elif why == "t0115a-sha-pin":
        monkeypatch.setattr(h, "T0115A_SHA256", "0" * 64)
    elif why in ("c398fdf-full-pin", "c398fdf-warm-pin"):
        key = "hall/kept/FULL" if why == "c398fdf-full-pin" else "ferry/old/warm"
        monkeypatch.setattr(h, "C398FDF_E_TEXTS_SHA256", {**C398FDF_E_TEXTS_SHA256, key: "0" * 64})
    elif why == "old-stub":
        monkeypatch.setattr(loop, "elided_payload", old_elided_payload)
    elif why == "accepted-in-question":
        monkeypatch.setattr(
            h, "ACCEPTED", {**ACCEPTED_C4, "ferry": (*ACCEPTED_C4["ferry"], "pier 9")}
        )
    elif why == "accepted-not-in-fact":
        monkeypatch.setattr(h, "ACCEPTED", {**ACCEPTED_C4, "lighthouse": ("forty one point six",)})
    if why in ("kit-unloadable", "repair-off", "wrong-tool-set"):
        kind = {
            "kit-unloadable": "unloadable",
            "repair-off": "repair-off",
            "wrong-tool-set": "wrong",
        }[why]
        monkeypatch.setattr(cli, "load_tool_kit", _kit_loader(kind))
        rc, attempts, text, out = refused(h, tmp_path, capsys, url=url)
    else:
        with fast_kit(monkeypatch):
            rc, attempts, text, out = refused(h, tmp_path, capsys, url=url)
    assert_refused(rc, attempts, out, text, REFUSALS[why], why)
