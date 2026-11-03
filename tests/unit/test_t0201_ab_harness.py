"""T-0201 group C: the screen-reminder GPU A/B harness's pure half and its refusals (unit; CPU; no engine, no
browser, no socket). docs/tasks/T-0201.md C1-C6, "Tests to write" (C).

BASE_C 0128816e (main, ``T-0201 planned``). Rows that need the harness (``scripts/gpu/t0201_screen_reminder_ab.py``) are
red on a named assertion in ``harness()``/``api()`` ("does not exist yet" / "has no ..."), never on an import error:
every harness symbol is looked up inside the row. The CPU rows cover nothing that runs on the GPU. Every expected value
is computed by hand (comments say how) or by a scratch script of the test-author's (an independent Wilson by the
quadratic, sha256 of canonical JSON, ``random.Random(0).sample``), never read off an implementation (none exists).

**The P corpus, hand-checked (C3).** Every ``llm.request`` with ``phase == "tool_step"``, a ``live_label`` with text
and at least one ``image_url`` part in the four logs (213 of them: 36 + 87 + 14 + 76), whose own ``llm.commit`` texts
(same ``attempt_id``, joined in ``seq`` order per segment) match ``no_see_v2``. Each of the 13 was read by hand; the
quoted phrase is the first hit.

=========================  =====  ====  ==========================================================================
log                        seq    step  hit (hand-checked)
=========================  =====  ====  ==========================================================================
20261027-230926-966483     8043   6     "I don't have the latest screenshot in this turn's input yet"
20261027-230926-966483     9377   4     "I don't have the screenshot after the scroll yet"
20261027-230926-966483     11655  8     "I don't have the new screenshot"
20261028-145330-1824860    1055   7     "I don't have the new screenshot. I need to wait for the screenshot"
20261028-145330-1824860    6791   4     "I don't have a new screenshot yet"
20261029-173950-15317      5225   2     "I don't have the screenshot of the results yet"
20261029-173950-15317      11592  3     "I don't have a new screenshot" / "I cannot see the screen right now" (black frame)
20261029-173950-15317      16202  7     "I don't see the latest screenshot in the prompt" (Bug 1, spoke "cannot see")
20261029-181336-282350     3418   9     "I don't have the current screenshot"
20261029-181336-282350     3957   15    "I don't have the current screenshot" (Bug 1)
20261029-181336-282350     7383   5     "I don't have the new screenshot after the scroll"
20261029-181336-282350     8945   2     "I don't have the new screenshot yet"
20261029-181336-282350     9083   3     "I can't see the screen content yet" (white frame)
=========================  =====  ====  ==========================================================================

**C** is ``sorted(random.Random(0).sample(sorted(non-P (log, seq)), 20))`` over the other 200 (reading 3). Findings of
the hand check, reported, not resolved here: (a) 11592 and 9083 are no-see phrasings about a blank (black/white) frame,
not a missing one -- they match C5's rows and are kept; (b) 3616 (181336, "the screenshot for the last turn is not
provided") is a no-see claim C5's ``no_see_v2`` does not cover, and it falls in the seed-0 C sample; 8043, 11655 and
3418 also say "not provided" but are in P through another phrase.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not).

Imported, never copied (C1)
    T-0132's ``newcombe`` (through T-0134's harness, as T-0138's does); ``VllmClient`` from
    ``local_astra.conductor.llm_client``; B's helper ``with_screen_reminder`` and ``SCREEN_REMINDER_SEPARATOR`` from
    ``local_astra.toolcall.loop``; B's label-to-key mapper ``screen_reminder_key`` (and ``SCREEN_REMINDERS``) from
    ``local_astra.conductor.session``. The harness's ``reminder_key`` may wrap B's mapper (a ``None`` from B becomes
    the harness's ``ValueError``) but never re-derives a key from the label constants itself.

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``ARMS = ("BASE", "TIP")``; ``ORDER_SEED = 0``; ``N_P = 5``;
    ``N_C = 3``; ``P_CORPUS`` and ``C_CORPUS``: tuples of ``(log stem, seq, sha256)``; ``LOGS_DIR`` (the default of
    ``--logs-dir``, ``<repo>/logs/sessions``).

Pure functions
    ``no_see_v2(text) -> bool`` (C5); ``prompt_sha(prompt) -> str`` = sha256 of ``json.dumps(prompt, sort_keys=True,
    separators=(",", ":"), ensure_ascii=False).encode("utf-8")`` over the logged ``data["prompt"]`` (reading 1);
    ``corpus_findings(logs_dir, corpus) -> list[str]`` (empty when every pinned record is there with its sha; one finding
    per missing log, missing seq or differing sha, naming the log and seq); ``order(seed) -> list[(arm, log, seq,
    repeat)]`` (reading 2); ``decide(part1, part2) -> {"verdict", "part1", "part2", "conclusive", "interval"}`` (C6;
    shapes in ``_p1``/``_p2``); ``base_prompt(logged_prompt, image) -> Prompt`` and ``tip_prompt(base, key) -> Prompt``
    (C3's two arms; TIP through B's helper).

Seams ``main`` reads at call time
    ``_git(*args) -> str`` (T-0111g's); ``_nvidia_smi() -> str`` (the text of ``nvidia-smi
    --query-compute-apps=pid,process_name,used_gpu_memory --format=csv``; a row whose executable is not one of
    ``preflight.ENGINE_PROCESS_NAMES`` is another compute process); ``_loadavg() -> (1, 5, 15 min)``;
    ``_listening_ports() -> set[int]``; ``_chromium_path() -> str | None``; ``capture_stand_in() -> bytes`` (the
    1280x800 frame of ``/kettles.html``); ``engine_healthy(base_url) -> bool`` (the first socket after the refusals);
    ``send_request(...)`` (every generation request). ``P_CORPUS``, ``C_CORPUS``. ``main(argv)`` takes ``--base-url``,
    ``--out-dir``, ``--logs-dir`` and ``--part {1,2,both}`` (default both).

Refusals (exit 2, before any socket, the out-dir left as it was): C2's list, rows in ``REFUSALS``.

**Readings** (reported to the orchestrator)
    1. "sha256 of the logged prompt tree" is read as ``prompt_sha`` above (canonical JSON of ``data["prompt"]``).
    2. The order is arm-major over ``ARMS``, then ``P_CORPUS`` x ``range(N_P)`` then ``C_CORPUS`` x ``range(N_C)``,
       then one ``random.Random(seed).shuffle``: 2 x (13 x 5 + 20 x 3) = 250. ``order.json`` is that list as JSON,
       on disk before ``engine_healthy`` is called (and so before any request).
    3. The C sample construction (C3 says only "seed 0, pinned").
    4. "b >= 10 % of P samples": P samples = per-arm samples on P (``n``), inclusive (``10 * b >= n``: 7 of 70 is
       conclusive). "C TIP valid-step rate >= BASE - 5/60" with ``c_n`` =
       60 is ``tip >= base - 5``. The Newcombe interval is T-0132's ``newcombe(b, n, t, n)`` (independent arms);
       "excludes 0" is ``lo > 0 or hi < 0``.
    5. The merge check reads subjects through ``_git`` (any call with ``"log"`` in its args returns the merge subjects
       here); a subject starting ``Merge T-0201-b:`` counts, ``Merge T-0201-a:`` does not.
    6. Chromium is required only when Part 2 runs (``--part 1`` with Chromium absent is not refused).

Regime: CPU, pure, hand-built records; no engine, no browser; n = 1.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import itertools
import json
import random
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0201_screen_reminder_ab.py"
T0132_HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
SESSION_LOGS = REPO / "logs" / "sessions"
#: the main checkout's logs (gitignored, so a worktree has none of its own)
MAIN_LOGS = (
    Path(*REPO.parts[: REPO.parts.index(".claude")]) / "logs" / "sessions"
    if ".claude" in REPO.parts
    else SESSION_LOGS
)

LOGS = (
    "20261027-230926-966483",
    "20261028-145330-1824860",
    "20261029-173950-15317",
    "20261029-181336-282350",
)
#: C3's P corpus, hand-checked (table in the docstring); sha256 by the test-author's scratch script (reading 1)
P_CORPUS = (
    (
        "20261027-230926-966483",
        8043,
        "047f639ab16744d6ccc4057c8e8da59993cef4e794b0c1084f5fffd34061a0c5",
    ),
    (
        "20261027-230926-966483",
        9377,
        "48c89284294b872a3a18109b0ccf19ce0020b1507f73362dfab60581c4b865b8",
    ),
    (
        "20261027-230926-966483",
        11655,
        "aab016fcfd23c20e5b1d9164b6196be25502dbf83a210a9eb64cf907244d58c8",
    ),
    (
        "20261028-145330-1824860",
        1055,
        "b9e45e0d7e7875cae58fb45fbc053f6bf6eced19ef36ac0ab6ba9b5afbdc7077",
    ),
    (
        "20261028-145330-1824860",
        6791,
        "099406e5538c3f90971a61dab619bceeafad72e78022d0de6f7bb61efd8ce632",
    ),
    (
        "20261029-173950-15317",
        5225,
        "004c257c0f70fb875eb6107b47c67e3f843d327b8bbd77981090ce6ffbe378d0",
    ),
    (
        "20261029-173950-15317",
        11592,
        "4bf86b537c9243ad7c5dec92ca29618c3ecd5d340044d3a3fa5de3ddfa8cf303",
    ),
    (
        "20261029-173950-15317",
        16202,
        "f8f3700dadf3ba7c6a83ff93b74ea96230125d4e936c4e0e09dd5cb8b4d6852a",
    ),
    (
        "20261029-181336-282350",
        3418,
        "81cc3105e57697fc9009a8bda59b2071c18b3be2bed44fa8388997096a69ec73",
    ),
    (
        "20261029-181336-282350",
        3957,
        "db3af900f7823bb58c1d5c0fd8612926acb8723c22e078a37a1570144e78e710",
    ),
    (
        "20261029-181336-282350",
        7383,
        "9a2c379014139e28c5ca1f2552163b9143a38a90226184a99060cd94ae6c6410",
    ),
    (
        "20261029-181336-282350",
        8945,
        "0e3b39b5cccf5e41487dd543e053d97ede76a2476962cbc29634c13f25ee4869",
    ),
    (
        "20261029-181336-282350",
        9083,
        "9777e99dcd5f283f06c59d6915b31edfd0767e43a3b5fc0f8e49410c41c9d6e0",
    ),
)
#: C3's C corpus: reading 3 over the 200 non-P requests, by the test-author's scratch script
C_CORPUS = (
    (
        "20261027-230926-966483",
        7949,
        "b2284111a3fc4df012aee4b5e963cdc4693bc0cbc71b6a1e679b80d6bcd67d96",
    ),
    (
        "20261027-230926-966483",
        11057,
        "e4e55ae1c9caa6e821bc762c901ac321a7b5da42fd0e5cd8690d6518b8b7a3d7",
    ),
    (
        "20261028-145330-1824860",
        682,
        "808cf0c929dfab1f053396d6386b9aeb73cf0809579eea73bcd04b7adc9254e0",
    ),
    (
        "20261028-145330-1824860",
        6991,
        "b3637e2cb138ba7f06798e3ca735e13d94b728c33a8a47e9631272b9675fe64b",
    ),
    (
        "20261028-145330-1824860",
        8503,
        "75d0c91b30b98af89a2c96c2dc277b545248da2aaf8c000dfbf19334698dd788",
    ),
    (
        "20261028-145330-1824860",
        8731,
        "57e1c2573b447ce3f02d044875db6a191567fa833594d85b3938c828ec21adc9",
    ),
    (
        "20261028-145330-1824860",
        9813,
        "ee0262368f5670a728a9cd81c7ae2753152eea2591b073413e517157330b3d1b",
    ),
    (
        "20261028-145330-1824860",
        10173,
        "b71c5bf218ca3fefceec80148c492361f04993a2d5f489af13d9e78c072eea1c",
    ),
    (
        "20261028-145330-1824860",
        11855,
        "e02dc1a4946adda4d8f15fc6304cb8cdd1d86d028fce5a172130b37e4ecea84b",
    ),
    (
        "20261028-145330-1824860",
        12581,
        "1c1c24eedf130fb562b23ce56f3eba596576bc179202d822a5d29a9bd83e0d80",
    ),
    (
        "20261028-145330-1824860",
        13518,
        "10735e5be282fc2e98a829a257dc234ce153e0ce7d0390bfec8f406de615d9cb",
    ),
    (
        "20261028-145330-1824860",
        14725,
        "ed4e7a7bfa48696e79ee47bd02e28ce36e6f45854e261f74259ec5314f9ee440",
    ),
    (
        "20261029-173950-15317",
        4528,
        "ac6d7fa584b4554eece34f599d3e8fc435499709105386ea93692be782900aaf",
    ),
    (
        "20261029-173950-15317",
        15728,
        "fc8eac2381dc1d1d3617e2cdd52010bf73bb0c933e28322265c59e8bfab084f9",
    ),
    (
        "20261029-181336-282350",
        330,
        "f03c3253d20f369a5071bef1317ac610ac41ee9c2d6239323992f66d02fde1db",
    ),
    (
        "20261029-181336-282350",
        419,
        "dd7700f4694dc9dca2599b81cfad3d911fade859a5919590479e86bca3a6f9f6",
    ),
    (
        "20261029-181336-282350",
        3616,
        "e7cc289921cb23f11fb7c6baaecc9facf033f0d30bded40b082fffb2afd98232",
    ),
    (
        "20261029-181336-282350",
        7560,
        "0ac179b58c9012df9c17e1ce422afb64e62fae160e07af5cd78e1f06da268a06",
    ),
    (
        "20261029-181336-282350",
        14718,
        "0089e289c9479f742e1e9714c727319ad8d4f9ab16dcdd0e5a976ef51ee2d98c",
    ),
    (
        "20261029-181336-282350",
        14835,
        "bee1cbd177a2e57ab5b075f63b7451c9de98803e5ef556dc96af52a459277db4",
    ),
)
#: H-T201-1's wording, typed from docs/tasks/T-0201.md (never imported)
REMINDER_UPDATED = (
    "[You have the current screenshot: the last image in the user's message above is the browser screen now, "
    "after your last action.]"
)


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0201 C1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0201_screen_reminder_ab_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), f"T-0201: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def _sha(prompt: Any) -> str:
    """Reading 1, written out (the definition, not the harness's code)."""
    blob = json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _logs_dir() -> Path:
    for d in (SESSION_LOGS, MAIN_LOGS):
        if all((d / f"{log}.jsonl").is_file() for log in LOGS):
            return d
    return SESSION_LOGS


def _need_logs() -> Path:
    d = _logs_dir()
    absent = [log for log in LOGS if not (d / f"{log}.jsonl").is_file()]
    if absent:
        pytest.skip(
            f"T-0201 C3: session log(s) absent (gitignored, local only): {', '.join(absent)}"
        )
    return d


# =====================================================================================================
# C1: import, constants
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """C1: importing the harness opens no socket (connect, bind or name lookup, loopback included); it has a ``main``;
    ``newcombe`` is T-0132's object, imported, not copied."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    src = Path(inspect.getsourcefile(api("newcombe")) or "").resolve()
    assert src == T0132_HARNESS.resolve(), (
        f"T-0201 C1: newcombe comes from {src}, not T-0132's harness"
    )


def test_constants() -> None:
    """C2/C3: exits, arms, seed, per-item repeats."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED")) == (0, 1, 2)
    assert tuple(api("ARMS")) == ("BASE", "TIP")
    assert api("ORDER_SEED") == 0
    assert (api("N_P"), api("N_C")) == (5, 3), "C3: n = 5/arm per P item, 3/arm per C item"


def test_corpus_pinned() -> None:
    """C3: the harness pins exactly the hand-checked P corpus (16202 and 3957 among them) and the seed-0 C sample, each
    as ``(log, seq, sha256)``; C has 20 items, none in P."""
    p = tuple(tuple(x) for x in api("P_CORPUS"))
    c = tuple(tuple(x) for x in api("C_CORPUS"))
    assert p == P_CORPUS, p
    assert c == C_CORPUS, c
    assert ("20261029-173950-15317", 16202) in {(x[0], x[1]) for x in p}
    assert ("20261029-181336-282350", 3957) in {(x[0], x[1]) for x in p}
    assert len(c) == 20 and not {(x[0], x[1]) for x in c} & {(x[0], x[1]) for x in p}


# =====================================================================================================
# C5: no_see_v2
# =====================================================================================================

#: C5's rows, verbatim
C5_MATCH = (
    "I don't have the latest screenshot",
    "I don't have the current screenshot",
    "I don't have the new screenshot",
    "no screenshot",
    "I cannot see the screen right now",
    "can't see the screen",
)
C5_REFUSE = (
    "I can see the screenshot",
    "I don't see a search box",
    "the user cannot see it",
    "I cannot see the price yet",
    "I can't see the search results yet, I will scroll",
)
#: from the hand check: phrasings in P's commits, and near misses in non-P commits (log, seq in the comment)
CORPUS_MATCH = (
    "Actually, I don't have a new screenshot yet.",  # 6791
    "Wait, I don't see the latest screenshot in the prompt.",  # 16202
    "Since I don't have the current screenshot, I will tell the user.",  # 3957
)
CORPUS_REFUSE = (
    "Since I don't see an address bar in the screenshot, I'll use the search tool",  # 15670
    "Wait, none of these images clearly show a crosswalk.",  # 7741
    'the prompt says "and now ?" and the screenshot is blank.',  # 4528
    '"One Piece (Main Story)" is not visible in the current screenshot.',  # 11057
)


@pytest.mark.parametrize("text", C5_MATCH + CORPUS_MATCH)
def test_no_see_v2_matches(text: str) -> None:
    """C5: each pinned no-see phrasing matches, also embedded in a sentence and in another case."""
    fn = api("no_see_v2")
    assert fn(text) is True, text
    assert fn(f"Thinking. {text}. Then I go on.") is True
    assert fn(text.upper()) is True, "C5: case-insensitive"


@pytest.mark.parametrize("text", C5_REFUSE + CORPUS_REFUSE)
def test_no_see_v2_refuses(text: str) -> None:
    """C5 (objects limited to screen/screenshot/image): each refused row does not match (reviewer mutation: a
    ``no_see_v2`` too wide matches "I cannot see the price yet")."""
    assert api("no_see_v2")(text) is False, text


def test_corpus_is_exactly_the_matching_set() -> None:
    """C3 on the four logs (skipped when absent): over every labelled ``tool_step`` request with an image, ``no_see_v2``
    on its own commits selects exactly the hand-checked P corpus; every C item is a non-match from the same set."""
    d = _need_logs()
    fn = api("no_see_v2")
    eligible: set[tuple[str, int]] = set()
    matched: set[tuple[str, int]] = set()
    for log in LOGS:
        reqs: dict[str, int] = {}
        texts: dict[str, dict[str, str]] = {}
        for line in (d / f"{log}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            data = r.get("data") or {}
            if r.get("kind") == "llm.request" and data.get("phase") == "tool_step":
                label = (data.get("live_label") or {}).get("text")
                images = sum(
                    1
                    for m in data["prompt"]["messages"]
                    if isinstance(m["content"], list)
                    for p in m["content"]
                    if p.get("type") == "image_url"
                )
                if label and images:
                    reqs[data["attempt_id"]] = r["seq"]
            elif r.get("kind") == "llm.commit":
                seg = texts.setdefault(data.get("attempt_id"), {})
                seg[data.get("segment")] = seg.get(data.get("segment"), "") + (
                    data.get("text") or ""
                )
        for aid, seq in reqs.items():
            eligible.add((log, seq))
            if any(fn(t) for t in texts.get(aid, {}).values()):
                matched.add((log, seq))
    assert len(eligible) == 213, len(eligible)
    assert matched == {(x[0], x[1]) for x in P_CORPUS}, sorted(
        matched ^ {(x[0], x[1]) for x in P_CORPUS}
    )
    assert {(x[0], x[1]) for x in C_CORPUS} <= eligible - matched


# =====================================================================================================
# C3: the corpus pin
# =====================================================================================================

#: a hand-built prompt and its reading-1 sha (scratch: hashlib over the canonical JSON written out)
TINY_PROMPT = {
    "messages": [{"role": "user", "content": [{"type": "text", "text": "é"}]}],
    "continue_final_message": False,
}
TINY_SHA = "f34d84b7f09f5071cc111283de13b81bacb1951f1b7e9ef297e217fd6aede509"


def test_prompt_sha_definition() -> None:
    """Reading 1: ``prompt_sha`` is sha256 of the canonical JSON (sorted keys, no spaces, UTF-8 unescaped); key order
    does not move it, a changed byte does."""
    fn = api("prompt_sha")
    assert _sha(TINY_PROMPT) == TINY_SHA, "premise: the hand pin"
    assert fn(TINY_PROMPT) == TINY_SHA
    assert fn(dict(reversed(list(TINY_PROMPT.items())))) == TINY_SHA
    moved = {
        **TINY_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "e"}]}],
    }
    assert fn(moved) != TINY_SHA


def test_corpus_shas_match_the_logs() -> None:
    """C3 (skipped when a log is absent): every pinned record's prompt hashes to its pin, through the harness and through
    reading 1 written out; ``corpus_findings`` over the real logs is empty."""
    d = _need_logs()
    want = {(log, seq): sha for log, seq, sha in P_CORPUS + C_CORPUS}
    got: dict[tuple[str, int], str] = {}
    for log in LOGS:
        for line in (d / f"{log}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") == "llm.request" and (log, r.get("seq")) in want:
                got[(log, r["seq"])] = _sha(r["data"]["prompt"])
                assert api("prompt_sha")(r["data"]["prompt"]) == want[(log, r["seq"])]
    assert got == want
    assert list(api("corpus_findings")(d, P_CORPUS + C_CORPUS)) == []


def _record(seq: int, text: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "llm.request",
        "data": {
            "phase": "tool_step",
            "step": 2,
            "attempt_id": f"a-{seq}",
            "live_label": {
                "text": "[This screenshot is the browser screen now, after your last action.]"
            },
            "prompt": {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "url": {"redacted": "image"},
                                "phash": "capture:00",
                            },
                            {
                                "type": "text",
                                "text": "[This screenshot is the browser screen now, after your last action.]",
                            },
                            {"type": "text", "text": "find the kettle"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": text}]},
                ],
                "prefix_len": 2,
                "continue_final_message": True,
            },
        },
    }


FAKE_LOG = "20990101-000000-1"


def _write_fake_log(d: Path, records: list[dict[str, Any]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    noise = {"seq": 1, "kind": "note", "data": {}}
    (d / f"{FAKE_LOG}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [noise, *records]) + "\n", encoding="utf-8"
    )


def test_corpus_findings_name_each_problem(tmp_path: Path) -> None:
    """C2/C3: a pinned record present with its sha gives nothing; a differing sha, an absent seq and an absent log each
    give a finding naming the log and the seq."""
    fn = api("corpus_findings")
    r10, r20 = _record(10, "call 1"), _record(20, "call 2")
    _write_fake_log(tmp_path, [r10, r20])
    ok = ((FAKE_LOG, 10, _sha(r10["data"]["prompt"])), (FAKE_LOG, 20, _sha(r20["data"]["prompt"])))
    assert list(fn(tmp_path, ok)) == []
    bad_sha = (ok[0], (FAKE_LOG, 20, _sha(r10["data"]["prompt"])))
    found = list(fn(tmp_path, bad_sha))
    assert len(found) == 1 and FAKE_LOG in found[0] and "20" in found[0], found
    found = list(fn(tmp_path, (*ok, (FAKE_LOG, 30, "0" * 64))))
    assert len(found) == 1 and "30" in found[0], found
    found = list(fn(tmp_path, (*ok, ("20990101-000000-2", 10, "0" * 64))))
    assert len(found) == 1 and "20990101-000000-2" in found[0], found


# =====================================================================================================
# C3: the order
# =====================================================================================================


def _reading2(seed: int, p: Any, c: Any) -> list[tuple[str, str, int, int]]:
    """Reading 2's construction, written out from the reading (the spec, not the harness's code)."""
    turns = [
        (arm, log, seq, r)
        for arm in ("BASE", "TIP")
        for items, n in ((p, 5), (c, 3))
        for (log, seq, _sha_) in items
        for r in range(n)
    ]
    random.Random(seed).shuffle(turns)
    return turns


def test_order_deterministic_seed0() -> None:
    """C3: 250 turns, each (arm, P item) 5 times, each (arm, C item) 3 times; deterministic for seed 0, another seed
    differs; the arms interleave (a per-arm shuffle would run them in blocks); reading 2's construction."""
    order = api("order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)]
    assert turns != [tuple(t) for t in order(1)]
    assert len(turns) == 2 * (13 * 5 + 20 * 3) == 250
    per = Counter((t[0], t[1], t[2]) for t in turns)
    for log, seq, _ in P_CORPUS:
        assert per[("BASE", log, seq)] == per[("TIP", log, seq)] == 5
    for log, seq, _ in C_CORPUS:
        assert per[("BASE", log, seq)] == per[("TIP", log, seq)] == 3
    switches = sum(1 for x, y in itertools.pairwise(turns) if x[0] != y[0])
    assert switches > 80, f"the arms run in blocks ({switches} switches)"
    assert turns == _reading2(0, P_CORPUS, C_CORPUS)


# =====================================================================================================
# C6: decide
# =====================================================================================================


def _p1(n: int, b: int, t: int, **over: Any) -> dict[str, Any]:
    """Part 1's inputs: ``n`` samples per arm on P, the no-see counts, and C's (60 = 20 x 3 per arm) valid-step and
    no-see counts (passing by default)."""
    return {
        "n": n,
        "b": b,
        "t": t,
        "c_n": 60,
        "c_valid": {"BASE": 58, "TIP": 58},
        "c_no_see": {"BASE": 0, "TIP": 0},
        **over,
    }


SAFE = {"over_clicks": 0, "url_leaks": 0, "false_claims": 0}


def _p2(**over: Any) -> dict[str, Any]:
    """Part 2's inputs: 30 = 10 tasks x 3 per arm; passing by default."""
    return {
        "n": 30,
        "successes": {"BASE": 25, "TIP": 25},
        "no_see_steps": {"BASE": 1, "TIP": 1},
        "safety": {"BASE": dict(SAFE), "TIP": dict(SAFE)},
        # the positive control (review of 2026-10-29): llm.requests that logged ``screen_reminder``, per arm
        "reminder_requests": {"BASE": 0, "TIP": 60},
        **over,
    }


#: (id, part1, part2, want part1, want verdict); hand-derived from C6 (intervals from the scratch Newcombe, below)
DECIDE = [
    # b = 6 of 65 is 9.2 % < 10 %: Part 1 INCONCLUSIVE although t = 0 and the interval (0.0178, 0.187) excludes 0;
    # a 5 % threshold would make it PASS
    ("b-just-under-10pct", _p1(65, 6, 0), _p2(), "INCONCLUSIVE", "INCONCLUSIVE"),
    # b = 7 of 70 is exactly 10 % (the bar is >=, not >): conclusive; interval (0.0274, 0.192)
    ("b-exactly-10pct", _p1(70, 7, 0), _p2(), "PASS", "PASS"),
    # t = b/2 exactly (10 = 20/2) passes; interval (0.0084, 0.292) excludes 0
    ("t-equals-half-b", _p1(65, 20, 10), _p2(), "PASS", "PASS"),
    # t = 11 > 21/2 fails though the interval (0.0053, 0.294) excludes 0
    ("t-just-over-half-b", _p1(65, 21, 11), _p2(), "FAIL", "FAIL"),
    # t = b/2 but the interval (-0.0290, 0.214) holds 0: the Newcombe clause fails Part 1
    ("newcombe-holds-0", _p1(65, 12, 6), _p2(), "FAIL", "FAIL"),
    # C: TIP valid 53 = 58 - 5 passes; 52 fails
    ("c-valid-at-margin", _p1(65, 20, 10, c_valid={"BASE": 58, "TIP": 53}), _p2(), "PASS", "PASS"),
    (
        "c-valid-below-margin",
        _p1(65, 20, 10, c_valid={"BASE": 58, "TIP": 52}),
        _p2(),
        "FAIL",
        "FAIL",
    ),
    # C: TIP no-see 2 = 0 + 2 passes; 3 fails
    ("c-no-see-at-margin", _p1(65, 20, 10, c_no_see={"BASE": 0, "TIP": 2}), _p2(), "PASS", "PASS"),
    (
        "c-no-see-over-margin",
        _p1(65, 20, 10, c_no_see={"BASE": 0, "TIP": 3}),
        _p2(),
        "FAIL",
        "FAIL",
    ),
    # Part 2: TIP successes 22 = 25 - 3 passes; 21 fails (and so the verdict, Part 1 passing)
    (
        "p2-success-at-margin",
        _p1(65, 20, 10),
        _p2(successes={"BASE": 25, "TIP": 22}),
        "PASS",
        "PASS",
    ),
    (
        "p2-success-below-margin",
        _p1(65, 20, 10),
        _p2(successes={"BASE": 25, "TIP": 21}),
        "PASS",
        "FAIL",
    ),
    # Part 2: TIP no-see steps 2 > BASE 1 fails
    (
        "p2-no-see-over-base",
        _p1(65, 20, 10),
        _p2(no_see_steps={"BASE": 1, "TIP": 2}),
        "PASS",
        "FAIL",
    ),
    # Part 2: one safety counter non-zero in either arm fails
    (
        "p2-safety-tip",
        _p1(65, 20, 10),
        _p2(safety={"BASE": dict(SAFE), "TIP": {**SAFE, "url_leaks": 1}}),
        "PASS",
        "FAIL",
    ),
    (
        "p2-safety-base",
        _p1(65, 20, 10),
        _p2(safety={"BASE": {**SAFE, "over_clicks": 1}, "TIP": dict(SAFE)}),
        "PASS",
        "FAIL",
    ),
    # Part 1 INCONCLUSIVE and Part 2 failing: FAIL (Part 2 alone decides FAIL)
    (
        "inconclusive-p1-failing-p2",
        _p1(65, 4, 0),
        _p2(successes={"BASE": 25, "TIP": 20}),
        "INCONCLUSIVE",
        "FAIL",
    ),
    # b = 0: INCONCLUSIVE, never a division error
    ("b-zero", _p1(65, 0, 0), _p2(), "INCONCLUSIVE", "INCONCLUSIVE"),
    # the Part 2 positive control: TIP's turns logged no screen_reminder -> the arms were not what they say: Part 2
    # INCONCLUSIVE (so the verdict, Part 1 passing); BASE logging one -> likewise
    (
        "p2-reminder-absent-in-tip",
        _p1(65, 20, 10),
        _p2(reminder_requests={"BASE": 0, "TIP": 0}),
        "PASS",
        "INCONCLUSIVE",
    ),
    (
        "p2-reminder-present-in-base",
        _p1(65, 20, 10),
        _p2(reminder_requests={"BASE": 1, "TIP": 60}),
        "PASS",
        "INCONCLUSIVE",
    ),
    # ... and it overrides a failing bar: a TIP arm that never carried the reminder cannot fail the reminder
    (
        "p2-failing-bar-but-reminder-absent",
        _p1(65, 20, 10),
        _p2(successes={"BASE": 25, "TIP": 10}, reminder_requests={"BASE": 0, "TIP": 0}),
        "PASS",
        "INCONCLUSIVE",
    ),
]


@pytest.mark.parametrize("case, part1, part2, want_p1, want", DECIDE, ids=[c[0] for c in DECIDE])
def test_decide_edges(
    case: str, part1: dict[str, Any], part2: dict[str, Any], want_p1: str, want: str
) -> None:
    """C6: the conclusive threshold (10 %, exact), t <= b/2, the Newcombe interval, C's two margins, Part 2's bars, and
    the PASS/FAIL/INCONCLUSIVE combination."""
    got = api("decide")(part1, part2)
    assert got["part1"] == want_p1, (case, got)
    assert got["verdict"] == want, (case, got)
    assert got["conclusive"] is (want_p1 != "INCONCLUSIVE"), (case, got)
    want_p2 = "FAIL" if case in P2_FAILS else "INCONCLUSIVE" if case in P2_NOT_SHOWN else "PASS"
    assert got["part2"] == want_p2, (case, got)


#: the DECIDE rows whose Part 2 inputs fail a bar (every other row's Part 2 passes)
P2_FAILS = {
    "p2-success-below-margin",
    "p2-no-see-over-base",
    "p2-safety-tip",
    "p2-safety-base",
    "inconclusive-p1-failing-p2",
}
#: the DECIDE rows whose Part 2 positive control fails (Part 2 INCONCLUSIVE whatever its bars say)
P2_NOT_SHOWN = {
    "p2-reminder-absent-in-tip",
    "p2-reminder-present-in-base",
    "p2-failing-bar-but-reminder-absent",
}


#: (b, n, t) -> Newcombe's 95 % interval of b/n - t/n, by the test-author's scratch script (Wilson bounds from the
#: quadratic (p - k/n)^2 = z^2 p (1 - p) / n, z = 1.959963984540054)
NEWCOMBE = {
    (20, 65, 10): (0.008408265270846205, 0.2920579642068729),
    (12, 65, 6): (-0.029034396103015267, 0.2136995759033004),
    (7, 70, 0): (0.027350333994379916, 0.19232914848567398),
}


@pytest.mark.parametrize("key", list(NEWCOMBE))
def test_decide_reports_newcombe_interval(key: tuple[int, int, int]) -> None:
    """C6: the reported interval is Newcombe's 95 % interval of (b - t) on P, hand values to 1e-9."""
    b, n, t = key
    lo, hi = api("decide")(_p1(n, b, t), _p2())["interval"]
    assert lo == pytest.approx(NEWCOMBE[key][0], abs=1e-9)
    assert hi == pytest.approx(NEWCOMBE[key][1], abs=1e-9)


# =====================================================================================================
# C2: refusals before any socket; C3: order.json written first; no healthy engine -> exit 1
# =====================================================================================================

ENGINE_ONLY = "pid, process_name, used_gpu_memory [MiB]\n2201, python3, 24203 MiB\n"
#: the engine as nvidia-smi names it on the 5090 box (vLLM V1's engine core process title; review of 2026-10-29)
REAL_ENGINE_ONLY = "pid, process_name, used_gpu_memory [MiB]\n2201, VLLM::EngineCore, 24203 MiB\n"
MERGED = (
    "Merge T-0200: flat frame retake",
    "Merge T-0201-b: screen reminder (inert)",
    "Merge T-0201-a: HUD",
)


def _session_with(session_mod: Any, name: str) -> Any:
    """``Session`` whose signature gains a keyword ``name`` (B merged), behaviour unchanged."""

    class New(session_mod.Session):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop(name, None)
            super().__init__(*args, **kwargs)

    sig = inspect.signature(session_mod.Session.__init__)
    extra = inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool)
    params = [*sig.parameters.values()]
    if name not in sig.parameters:
        params.append(extra)
    New.__init__.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    New.__signature__ = sig.replace(parameters=params[1:])  # type: ignore[attr-defined]
    return New


def _session_without(session_mod: Any, name: str) -> Any:
    class Old(session_mod.Session):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    sig = inspect.signature(session_mod.Session.__init__)
    kept = [p for p in sig.parameters.values() if p.name != name]
    Old.__init__.__signature__ = sig.replace(parameters=kept)  # type: ignore[attr-defined]
    Old.__signature__ = sig.replace(parameters=kept[1:])  # type: ignore[attr-defined]
    return Old


class Rig:
    """Every fake passing (clean tree, ``Merge T-0201-b:`` present, ``Session`` with ``screen_reminder``, a two-record
    corpus in a tmp logs dir, the engine alone on the GPU, load 1.99, only 8001 bound, Chromium present); every real
    socket refused and recorded; ``engine_healthy`` and ``send_request`` recorded; ``capture_stand_in`` a stub."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra.conductor import session as session_mod

        self.h = h = harness()
        self.mp = mp
        self.sockets: list[str] = []
        self.calls: list[str] = []
        self.porcelain = ""
        self.subjects = list(MERGED)
        self.smi: str | BaseException = ENGINE_ONLY
        self.load = (1.99, 1.0, 1.0)
        self.ports = {8001}
        self.chromium: str | None = "/opt/chromium/chrome"
        self.healthy = False
        self.order_on_disk_at_health: bool | None = None
        self.out = tmp_path / "out"
        self.logs = tmp_path / "logs"
        self.r10, self.r20 = _record(10, "call 1"), _record(20, "call 2")
        _write_fake_log(self.logs, [self.r10, self.r20])
        mp.setattr(h, "P_CORPUS", ((FAKE_LOG, 10, _sha(self.r10["data"]["prompt"])),))
        mp.setattr(h, "C_CORPUS", ((FAKE_LOG, 20, _sha(self.r20["data"]["prompt"])),))
        mp.setattr(session_mod, "Session", _session_with(session_mod, "screen_reminder"))

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0201 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            self.calls.append("git " + " ".join(args))
            if args[:1] == ("status",):
                return self.porcelain
            if "log" in args:
                return "\n".join(self.subjects) + "\n"
            return t31.GIT + "\n"

        def health(*a: Any, **k: Any) -> bool:
            self.calls.append("engine_healthy")
            self.order_on_disk_at_health = (self.out / "order.json").is_file()
            return self.healthy

        def send(*a: Any, **k: Any) -> Any:
            self.calls.append("send_request")
            raise AssertionError("T-0201 rig: a request was sent")

        for seam in (
            "_git",
            "_nvidia_smi",
            "_loadavg",
            "_listening_ports",
            "_chromium_path",
            "capture_stand_in",
            "engine_healthy",
            "send_request",
        ):
            assert callable(getattr(h, seam, None)), (
                f"T-0201: the harness has no module-level seam {seam!r}"
            )
        mp.setattr(h, "_git", git)
        mp.setattr(h, "_nvidia_smi", self._smi)
        mp.setattr(h, "_loadavg", lambda *a, **k: self.load)
        mp.setattr(h, "_listening_ports", lambda *a, **k: set(self.ports))
        mp.setattr(h, "_chromium_path", lambda *a, **k: self.chromium)
        mp.setattr(h, "capture_stand_in", lambda *a, **k: b"\xff\xd8stand-in")
        mp.setattr(h, "engine_healthy", health)
        mp.setattr(h, "send_request", send)

    def _smi(self, *a: Any, **k: Any) -> str:
        if isinstance(self.smi, BaseException):
            raise self.smi
        return str(self.smi)

    def argv(self, *extra: str) -> list[str]:
        return [
            "--base-url",
            t31.VALID_URL,
            "--out-dir",
            str(self.out),
            "--logs-dir",
            str(self.logs),
            *extra,
        ]


REFUSALS = (
    "url-not-loopback",
    "dirty-tree",
    "out-dir-exists",
    "no-merge-T-0201-b",
    "session-without-screen-reminder",
    "corpus-log-missing",
    "corpus-sha-differs",
    "gpu-other-compute-process",
    "load-at-2",
    "port-8000-bound",
    "port-1420-bound",
    "chromium-absent",
    # review of 2026-10-29: the real engine name beside another compute app still refuses; nvidia-smi failing
    # refuses (fail closed); a corpus label B maps to no key refuses
    "gpu-moshi-beside-real-engine",
    "gpu-chrome-beside-real-engine",
    "nvidia-smi-fails",
    "nvidia-smi-missing",
    "corpus-label-no-key",
)
#: what the refusal must name (regex over stderr, case-insensitive)
REFUSAL_NAMES = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "no-merge-T-0201-b": r"T-0201-b",
    "session-without-screen-reminder": r"screen_reminder",
    "corpus-log-missing": FAKE_LOG,
    "corpus-sha-differs": rf"{FAKE_LOG}.*\b20\b|\b20\b.*{FAKE_LOG}",
    "gpu-other-compute-process": r"blender|4242",
    "load-at-2": r"load",
    "port-8000-bound": r"8000",
    "port-1420-bound": r"1420",
    "chromium-absent": r"chromium",
    "gpu-moshi-beside-real-engine": r"moshi",
    "gpu-chrome-beside-real-engine": r"chrome",
    "nvidia-smi-fails": r"nvidia-smi",
    "nvidia-smi-missing": r"nvidia-smi",
    "corpus-label-no-key": rf"{FAKE_LOG}.*\b10\b|\b10\b.*{FAKE_LOG}",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """C2 (reviewer mutation: a refusal after a socket opens): each refusal exits 2 with a stderr line naming it, before
    any socket and before ``engine_healthy``/``send_request``, leaving the out-dir as it was."""
    import re

    from local_astra.conductor import session as session_mod

    rig = Rig(tmp_path, monkeypatch)
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-T-0201-b":
        rig.subjects = ["Merge T-0201-a: HUD", "Merge T-0200: flat frame retake"]
    elif case == "session-without-screen-reminder":
        monkeypatch.setattr(
            session_mod, "Session", _session_without(session_mod, "screen_reminder")
        )
    elif case == "corpus-log-missing":
        (rig.logs / f"{FAKE_LOG}.jsonl").unlink()
    elif case == "corpus-sha-differs":
        _write_fake_log(rig.logs, [rig.r10, _record(20, "call 2 ")])
    elif case == "gpu-other-compute-process":
        rig.smi = ENGINE_ONLY + "4242, /opt/blender/blender, 900 MiB\n"
    elif case == "load-at-2":
        rig.load = (2.0, 0.5, 0.5)
    elif case == "port-8000-bound":
        rig.ports |= {8000}
    elif case == "port-1420-bound":
        rig.ports |= {1420}
    elif case == "chromium-absent":
        rig.chromium = None
    elif case == "gpu-moshi-beside-real-engine":
        rig.smi = REAL_ENGINE_ONLY + "4243, /usr/local/bin/moshi-server, 3100 MiB\n"
    elif case == "gpu-chrome-beside-real-engine":
        rig.smi = REAL_ENGINE_ONLY + "4244, /opt/google/chrome/chrome --type=gpu-process, 210 MiB\n"
    elif case == "nvidia-smi-fails":
        rig.smi = subprocess.CalledProcessError(9, ["nvidia-smi"], "", "NVIDIA-SMI has failed")
    elif case == "nvidia-smi-missing":
        rig.smi = FileNotFoundError(2, "No such file or directory", "nvidia-smi")
    elif case == "corpus-label-no-key":
        # B's screen_reminder_key maps "[" + LABEL_NOW + " junk]" to None (only nothing or the extent clause may
        # follow the opening); the label is outside the prompt, so the pinned sha still matches
        bad = json.loads(json.dumps(rig.r10))
        bad["data"]["live_label"]["text"] = "[" + session_mod.LABEL_NOW + " junk]"
        assert session_mod.screen_reminder_key(bad["data"]["live_label"]["text"]) is None, "premise"
        _write_fake_log(rig.logs, [bad, rig.r20])
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = rig.h.main(argv)
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    assert "engine_healthy" not in rig.calls and "send_request" not in rig.calls, (case, rig.calls)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in REFUSAL_NAMES:
        assert re.search(REFUSAL_NAMES[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


def test_rig_passes_every_refusal_then_no_engine_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2 (the rig's premise: each refusal row above differs from a passing run in one fake only): with every fake
    passing and no healthy engine, ``main`` asks ``engine_healthy`` once and exits 1, nothing sent, no real socket."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.calls.count("engine_healthy") == 1 and "send_request" not in rig.calls, rig.calls
    assert rig.sockets == [], rig.sockets


def test_part1_only_does_not_need_chromium(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2 "Chromium absent (Part 2)" (reading 6): with ``--part 1`` an absent Chromium is not refused; the run goes on
    to the engine check (exit 1 here, nothing sent)."""
    rig = Rig(tmp_path, monkeypatch)
    rig.chromium = None
    rc = rig.h.main(rig.argv("--part", "1"))
    err = capsys.readouterr().err
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert "engine_healthy" in rig.calls and "send_request" not in rig.calls, rig.calls


def test_order_json_written_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C3 (reviewer mutation: no order file): ``order.json`` is on disk when ``engine_healthy`` is asked (before any
    request), and holds ``order(0)`` over the corpus the run uses."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    capsys.readouterr()
    assert rc == 1
    assert rig.order_on_disk_at_health is True, (
        "order.json is not written before the engine is asked"
    )
    written = json.loads((rig.out / "order.json").read_text(encoding="utf-8"))
    assert [tuple(t) for t in written] == [tuple(t) for t in rig.h.order(0)]
    assert [tuple(t) for t in written] == _reading2(0, rig.h.P_CORPUS, rig.h.C_CORPUS)
    assert len(written) == 2 * (5 + 3)


# =====================================================================================================
# Adversarial: TIP and BASE differ only in the suffix, through B's helper
# =====================================================================================================


def test_tip_and_base_bodies_differ_only_in_the_suffix() -> None:
    """C3 (adversarial): for one corpus-shaped item (16202's shape: image, label, utterance, a continuation of tool
    calls and responses), the BASE and TIP bodies built through ``VllmClient.request_body`` are equal except
    ``messages[-1]["content"]``, which is BASE's + ``SCREEN_REMINDER_SEPARATOR`` + the ``updated`` reminder (H-T201-1's
    wording); the stand-in image is the same in both arms."""
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.toolcall import loop

    # Merge T-0201-b is on main (46b0b4b3): B's names must be there; this row never skips.
    assert hasattr(loop, "SCREEN_REMINDER_SEPARATOR"), (
        "T-0201 B: loop.SCREEN_REMINDER_SEPARATOR missing"
    )
    assert callable(getattr(loop, "with_screen_reminder", None)), (
        "T-0201 B: loop.with_screen_reminder missing"
    )

    logged = _record(
        10,
        '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
        '<|tool_response>response:scroll{action:<|"|>scrolled down<|"|>}<tool_response|>',
    )["data"]["prompt"]
    image = b"\xff\xd8stand-in"
    base = api("base_prompt")(logged, image)
    tip = api("tip_prompt")(base, "updated")
    client = VllmClient()
    b = client.request_body(base, "r")
    t = client.request_body(tip, "r")
    assert loop.SCREEN_REMINDER_SEPARATOR == "\n\n", "H-T201-2"
    assert {k: v for k, v in b.items() if k != "messages"} == {
        k: v for k, v in t.items() if k != "messages"
    }
    assert b["messages"][:-1] == t["messages"][:-1]
    assert t["messages"][-1]["role"] == b["messages"][-1]["role"] == "assistant"
    assert t["messages"][-1]["content"] == b["messages"][-1]["content"] + "\n\n" + REMINDER_UPDATED
    assert b["messages"][-1]["content"].endswith("<tool_response|>"), (
        "premise: BASE ends at the last tool response"
    )


# =====================================================================================================
# C1: the harness maps a label to its reminder key through B's ``screen_reminder_key``, never its own copy
# =====================================================================================================

#: the session's label constants a key mapper has to read; a harness function that reads one is a copied mapper
_LABEL_CONSTANTS = frozenset(
    {"LABEL_NOW", "LABEL_UPDATED", "LABEL_UNCHANGED", "SHARED_LABEL_SCREEN", "SHARED_LABEL_CAMERA"}
)


def test_harness_reminder_key_uses_b_screen_reminder_key_not_a_copy() -> None:
    """C1 ("imports earlier harnesses, never copies") applied to B's key mapper: the harness's ``reminder_key`` goes
    through ``local_astra.conductor.session.screen_reminder_key``. Three independent checks: (1) source: the harness
    names ``screen_reminder_key`` and no harness function reads a session label constant (``LABEL_NOW`` ...
    ``SHARED_LABEL_CAMERA``), which a copied mapper must; (2) call: with B's function replaced by a spy (on the session
    module and, if the harness bound the name at import, on the harness too), ``reminder_key`` calls the spy and
    returns its key; (3) behaviour: a label B maps to ``None`` (``"[" + LABEL_NOW + " junk]"``: B accepts after the
    opening only nothing or the extent clause) has no key in the harness either (``ValueError``, the harness's own
    contract for an unmapped label) -- a prefix-matching copy returns ``"now"``."""
    import ast

    from local_astra.conductor import session as session_module

    src = HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name)
            names.add(node.name)
    assert "screen_reminder_key" in names, (
        "C1: the harness never names B's local_astra.conductor.session.screen_reminder_key"
    )
    copied = sorted(
        f"{fn.name} reads {sorted(_LABEL_CONSTANTS & {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(fn) if isinstance(n, (ast.Name, ast.Attribute))})}"
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _LABEL_CONSTANTS
        & {
            n.attr if isinstance(n, ast.Attribute) else n.id
            for n in ast.walk(fn)
            if isinstance(n, (ast.Name, ast.Attribute))
        }
    )
    assert not copied, (
        f"C1: the harness re-derives reminder keys from the label constants (a copy of B3): {copied}"
    )

    h = harness()
    reminder_key = api("reminder_key")
    label = "[" + session_module.LABEL_UPDATED + "]"
    calls: list[str] = []
    original = session_module.screen_reminder_key

    def spy(text: str) -> str | None:
        calls.append(text)
        return (
            "unchanged"  # deliberately not what the label says: the harness must return B's answer
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session_module, "screen_reminder_key", spy)
        if getattr(h, "screen_reminder_key", None) is original:
            mp.setattr(h, "screen_reminder_key", spy)
        got = reminder_key(label)
    assert calls, "C1: the harness's reminder_key did not call B's screen_reminder_key"
    assert got == "unchanged", f"C1: reminder_key returned {got!r}, not B's answer 'unchanged'"

    junk = "[" + session_module.LABEL_NOW + " junk]"
    assert original(junk) is None, "premise: B maps this label to no key"
    with pytest.raises(ValueError):
        reminder_key(junk)


# =====================================================================================================
# Review of 2026-10-29 (REQUEST CHANGES): rows 1-8
# =====================================================================================================


def test_real_engine_name_is_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C2 "another GPU compute process" (review row 1): on the 5090 box nvidia-smi lists the engine as
    ``VLLM::EngineCore``. With it alone on the GPU (every other fake passing) the idle-box check is passed: no refusal,
    the run reaches the engine check (exit 1 here, nothing sent). The rows ``gpu-*-beside-real-engine`` above keep any
    other compute app refused."""
    rig = Rig(tmp_path, monkeypatch)
    rig.smi = REAL_ENGINE_ONLY
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, (
        f"the engine's own process ({REAL_ENGINE_ONLY.splitlines()[1]!r}) was refused: exit {rc}\n{err}"
    )
    assert rig.calls.count("engine_healthy") == 1 and "send_request" not in rig.calls, rig.calls


# --- row 2: the Part 2 arms, positive control -------------------------------------------------------------------

#: hand-chosen stand-ins for ``head_live_extras`` (the keywords HEAD's live default adds over T-0138's runner)
EXTRAS = {"tool_guidance": "GUIDE-v3-standin", "page_fact": True, "reask_announced_calls": 2}
#: a stand-in for HEAD's live default: ``S.DEFAULT_KEYS`` plus the overview frame
DEFAULT = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": "label",
    "echo": "model",
    "recover_thought_calls": True,
    "overview_frame": False,
}
SUITE_IDS = ("S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15")


class _FakeVram:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _session_log(arm: str, reminders_in: set[str], with_hit: bool) -> list[dict[str, Any]]:
    """One fake turn's session log: an unlabelled step-1 request (no image label, never a reminder) and two labelled
    image steps that logged ``screen_reminder: "updated"`` when ``arm`` is in ``reminders_in``; the second step's
    thought says it has no screenshot when ``with_hit``."""
    first = _record(1, "")
    first["data"]["step"] = 1
    first["data"].pop("live_label")
    records: list[dict[str, Any]] = [first]
    for seq in (2, 3):
        r = _record(seq, "call:scroll{}")
        if arm in reminders_in:
            r["data"]["screen_reminder"] = "updated"
        records.append(r)
    text = "I don't have the new screenshot yet." if with_hit else "Scrolling to the kettles."
    records.append(
        {
            "seq": 4,
            "kind": "llm.commit",
            "data": {"attempt_id": "a-3", "segment": "thought", "text": text},
        }
    )
    return records


def _run_part2_rig(
    tmp_path: Path, mp: pytest.MonkeyPatch, reminders_in: set[str]
) -> tuple[Any, list[tuple[str, dict[str, Any]]], tuple[Any, Any, Any]]:
    """``_run_part2`` over a fake T-0138 turn runner: each turn builds ``session.Session`` (a recording class, so the
    harness's arm seam subclasses it) with T-0138-shaped keywords and writes a session log; ``S.succeeded`` and
    ``S.safety_counts`` are stubs (every turn a success, every counter 0). No browser, engine or socket."""
    from local_astra.conductor import session as session_mod

    h = harness()
    built: list[tuple[str, dict[str, Any]]] = []
    current: dict[str, str] = {}

    class Recording:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.append((current["arm"], dict(kwargs)))

    async def one_turn(run: Any, arm: str, task_id: str, repeat: int) -> Any:
        from local_astra.conductor.session import Session  # call time, as T-0138's runner does

        current["arm"] = arm
        Session(tools="live", visual_tokens=1120, tool_guidance="T0138-guidance")
        name = f"{arm}-{task_id}-{repeat}.jsonl"
        log = _session_log(arm, reminders_in, with_hit=(arm == "BASE" and repeat == 0))
        (Path(run.out) / "sessions" / name).write_text(
            "\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8"
        )
        trial = {
            "arm": arm,
            "task": task_id,
            "repeat": repeat,
            "session_log": name,
            "success": True,
            "tool_loop_end": "answer",
        }
        return trial, [], [], None, {}

    mp.setattr(session_mod, "Session", Recording)
    mp.setattr(h.S, "_one_turn", one_turn)
    mp.setattr(h.S, "succeeded", lambda t, st: bool(t.get("success")))
    mp.setattr(h.S, "safety_counts", lambda tr, st: dict(SAFE))
    mp.setattr(h.T, "_VramSampler", _FakeVram)
    mp.setattr(h.P, "arm_profile", lambda *a, **k: "profile-standin")
    out = tmp_path / "out"
    out.mkdir()
    got = h._run_part2(t31.VALID_URL, out, object(), dict(DEFAULT), dict(EXTRAS))
    return h, built, got


def test_part2_arms_built_with_reminder_and_head_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4 (review row 2, kills "``_arm_session`` applies no override"): every Part 2 turn's ``Session`` is built with
    HEAD's live extras over T-0138's keywords, ``screen_reminder=True`` in TIP and ``False`` in BASE; T-0138's other
    keywords kept; 30 turns per arm (10 tasks x 3)."""
    _h, built, _got = _run_part2_rig(tmp_path, monkeypatch, {"TIP"})
    assert Counter(arm for arm, _ in built) == {"BASE": 30, "TIP": 30}
    for arm, kwargs in built:
        assert kwargs.get("screen_reminder") is (arm == "TIP"), (arm, kwargs)
        for k, v in EXTRAS.items():
            assert kwargs.get(k) == v, f"{arm}: {k} = {kwargs.get(k)!r}, not HEAD's {v!r}"
        assert kwargs.get("tools") == "live" and kwargs.get("visual_tokens") == 1120, (arm, kwargs)


@pytest.mark.parametrize(
    "case, reminders_in, want_counts, want_p2",
    [
        # TIP's 30 turns x 2 labelled steps = 60 requests logged the reminder, BASE's none: the control holds
        ("shown", {"TIP"}, {"BASE": 0, "TIP": 60}, "PASS"),
        # neither arm logged one (the reminder never reached a request): Part 2 cannot speak to it
        ("absent", set(), {"BASE": 0, "TIP": 0}, "INCONCLUSIVE"),
        # both did (BASE was not BASE)
        ("in-base", {"BASE", "TIP"}, {"BASE": 60, "TIP": 60}, "INCONCLUSIVE"),
    ],
)
def test_part2_positive_control_counts_reminders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    reminders_in: set[str],
    want_counts: dict[str, int],
    want_p2: str,
) -> None:
    """C4/C6 (review row 2): Part 2 counts per arm the turns' ``llm.request`` records that logged ``screen_reminder``
    (B8); unless TIP's count is >= 1 and BASE's is 0, Part 2 is INCONCLUSIVE (never PASS, never FAIL), and the counts
    are in the report (``part2_inputs(...)["reminder_requests"]``, the test-author's name, carried into
    ``report.json``'s ``part2``). Every other Part 2 input passes here (equal successes, no hits in TIP, counters 0)."""
    h, _built, (trials, steps, nosee) = _run_part2_rig(tmp_path, monkeypatch, reminders_in)
    p2 = h.part2_inputs(trials, steps, nosee)
    assert "reminder_requests" in p2, (
        f"{case}: Part 2's inputs carry no reminder counts: {sorted(p2)}"
    )
    assert dict(p2["reminder_requests"]) == want_counts, (case, p2["reminder_requests"])
    report = h.build_report(None, p2, [], nosee)
    assert dict(report["part2"]["reminder_requests"]) == want_counts, case
    assert report["decision"]["part2"] == want_p2, (case, report["decision"])
    assert report["decision"]["verdict"] != "PASS"  # Part 1 not run


# --- row 3: part2_inputs no-see ------------------------------------------------------------------------------------


def test_part2_inputs_no_see_steps_per_arm() -> None:
    """C4 (review row 3, kills "``no_see_steps`` = 0"): the no-see steps per arm are the rows with a hit: TIP 2,
    BASE 1 (a row without a hit, and one of the other arm, never count)."""
    nosee = [
        {"arm": "TIP", "hit": "I don't have the new screenshot"},
        {"arm": "TIP", "hit": "I cannot see the screen right now"},
        {"arm": "TIP", "hit": None},
        {"arm": "BASE", "hit": "I don't have the current screenshot"},
        {"arm": "BASE", "hit": None},
    ]
    p2 = api("part2_inputs")([], [], nosee)
    assert dict(p2["no_see_steps"]) == {"BASE": 1, "TIP": 2}


def test_part2_no_see_hits_through_run_part2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4 (review row 3, end to end): BASE's repeat-0 turns (10 of them) each logged one no-see thought on a labelled
    image step; TIP's none: ``no_see_steps`` = BASE 10, TIP 0."""
    h, _built, (trials, steps, nosee) = _run_part2_rig(tmp_path, monkeypatch, {"TIP"})
    assert dict(h.part2_inputs(trials, steps, nosee)["no_see_steps"]) == {"BASE": 10, "TIP": 0}


# --- row 4: eligible_requests --------------------------------------------------------------------------------------


def test_eligible_requests_needs_label_image_and_tool_step() -> None:
    """C3/C4 (review row 4, kills "without the image condition" and "without the label condition"): of five
    ``llm.request`` records only those with ``phase == "tool_step"``, a label with text and an image part are eligible
    (seq 10 and 14); 11 has no image, 12 no label, 13 an empty label, 15 another phase. The hits come from each
    request's own commits joined per segment in ``seq`` order (14's answer is split over two commits)."""
    good = _record(10, "x")
    no_image = _record(11, "x")
    user = no_image["data"]["prompt"]["messages"][1]
    user["content"] = [p for p in user["content"] if p["type"] != "image_url"]
    no_label = _record(12, "x")
    no_label["data"].pop("live_label")
    empty_label = _record(13, "x")
    empty_label["data"]["live_label"] = {"text": ""}
    split = _record(14, "x")
    report = _record(15, "x")
    report["data"]["phase"] = "report"

    def commit(seq: int, aid: str, seg: str, text: str) -> dict[str, Any]:
        return {
            "seq": seq,
            "kind": "llm.commit",
            "data": {"attempt_id": aid, "segment": seg, "text": text},
        }

    records = [
        good,
        no_image,
        no_label,
        empty_label,
        split,
        report,
        commit(21, "a-10", "thought", "Scrolling now."),
        commit(23, "a-14", "answer", "current screenshot."),
        commit(22, "a-14", "answer", "Sorry, I don't have the "),
        *(
            commit(30 + s, f"a-{s}", "thought", "I don't have the new screenshot")
            for s in (11, 12, 13, 15)
        ),
    ]
    rows = api("eligible_requests")(records)
    assert [r["seq"] for r in rows] == [10, 14]
    assert rows[0]["hit"] is None
    assert rows[1]["hit"] is not None and "current screenshot" in rows[1]["hit"], rows[1]


def test_eligible_requests_on_the_four_logs_is_the_corpus() -> None:
    """C3 (review row 4, through the harness's own scan; skipped when the logs are absent): ``eligible_requests``
    over the four logs gives the 213 eligible requests, and its hits are exactly the hand-checked P corpus."""
    d = _need_logs()
    fn = api("eligible_requests")
    read = api("_read_log")
    eligible: set[tuple[str, int]] = set()
    hits: set[tuple[str, int]] = set()
    for log in LOGS:
        for r in fn(read(d / f"{log}.jsonl")):
            eligible.add((log, int(r["seq"])))
            if r["hit"]:
                hits.add((log, int(r["seq"])))
    assert len(eligible) == 213, len(eligible)
    assert hits == {(x[0], x[1]) for x in P_CORPUS}, sorted(hits ^ {(x[0], x[1]) for x in P_CORPUS})


# --- row 5: sample_facts -------------------------------------------------------------------------------------------

#: a parseable native call (``native.parse_call`` of it plus ``<tool_call|>`` is a dict; checked by hand)
CALL_OK = "call:click_at{x:100,y:200}"
#: cut inside its arguments: no parse with or without the closing tag
CALL_BAD = "call:click_at{x:"
#: "the" 10 times back to back: the T-0142 thought rule (a 1-token unit >= 8 times) trips
THOUGHT_TRIP = "the " * 10


def _commits(**segments: str) -> list[dict[str, Any]]:
    return [{"segment": k, "text": v, "thought_open": k == "thought"} for k, v in segments.items()]


SAMPLE_FACTS = [
    # (id, commits, want valid, want no_see)
    ("call-clean", _commits(thought="Click the kettle.", tool=CALL_OK), True, False),
    ("call-thought-trip", _commits(thought=THOUGHT_TRIP, tool=CALL_OK), False, False),
    ("call-unparseable", _commits(thought="Click the kettle.", tool=CALL_BAD), False, False),
    ("answer-only", _commits(thought="Done.", answer="The kettle costs 30 pounds."), True, False),
    # "the" 7 times in the answer: the T-0142b answer rule (>= 6) trips
    ("answer-trip", _commits(answer="the " * 7), False, False),
    ("nothing", _commits(thought="Hmm."), False, False),
    (
        "hit-in-answer-only",
        _commits(thought="Fine.", answer="I don't have the new screenshot yet."),
        True,
        True,
    ),
    (
        "hit-in-tool-only",
        _commits(tool=CALL_OK + " I cannot see the screen right now"),
        False,
        True,
    ),
    (
        "hit-in-thought",
        _commits(thought="I don't have the current screenshot.", tool=CALL_OK),
        True,
        True,
    ),
]


@pytest.mark.parametrize(
    "case, commits, want_valid, want_no_see", SAMPLE_FACTS, ids=[c[0] for c in SAMPLE_FACTS]
)
def test_sample_facts(
    case: str, commits: list[dict[str, Any]], want_valid: bool, want_no_see: bool
) -> None:
    """C3/C5 (review row 5, kills "valid ignores the thought trip", "valid ignores ``parse_call``", "the hit is
    searched only in the thought"): valid = a parseable tool call or a non-empty answer, and no T-0142 trip; the
    no-see hit is searched in every segment."""
    facts = api("sample_facts")(commits)
    assert facts["valid"] is want_valid, (case, facts)
    assert facts["no_see"] is want_no_see, (case, facts)


# --- row 6: engine errors ------------------------------------------------------------------------------------------


def _samples(errored: dict[str, int]) -> list[dict[str, Any]]:
    """250 samples shaped as Part 1 logs them: per arm 13 P items x 5 and 20 C items x 3; BASE no-see on its first
    20 P samples, TIP on none; every C sample valid. The first ``errored[arm]`` C samples of an arm carry an engine
    error instead (no commits: not valid, no hit)."""
    out: list[dict[str, Any]] = []
    h = harness()
    for arm in ("BASE", "TIP"):
        k = 0
        for log, seq, _ in h.P_CORPUS:
            for r in range(5):
                out.append(
                    {
                        "arm": arm,
                        "log": log,
                        "seq": seq,
                        "repeat": r,
                        "error": None,
                        "no_see": arm == "BASE" and k < 20,
                        "valid": True,
                    }
                )
                k += 1
        e = errored.get(arm, 0)
        for log, seq, _ in h.C_CORPUS:
            for r in range(3):
                bad = e > 0
                e -= 1 if bad else 0
                out.append(
                    {
                        "arm": arm,
                        "log": log,
                        "seq": seq,
                        "repeat": r,
                        "error": "ReadTimeout('engine')" if bad else None,
                        "no_see": False,
                        "valid": not bad,
                    }
                )
    return out


@pytest.mark.parametrize(
    "case, errored, want_errors, want_p1",
    [
        # b = 20, t = 0 of 65 (conclusive, 2t <= b, interval excludes 0), C valid 60 = 60: PASS
        ("none", {}, {"BASE": 0, "TIP": 0}, "PASS"),
        # one TIP engine error: INCONCLUSIVE (with it ignored TIP's C valid is 59 of 59, which would still PASS)
        ("tip-only", {"TIP": 1}, {"BASE": 0, "TIP": 1}, "INCONCLUSIVE"),
        ("base-only", {"BASE": 1}, {"BASE": 1, "TIP": 0}, "INCONCLUSIVE"),
    ],
)
def test_part1_engine_error_in_either_arm_is_inconclusive(
    case: str, errored: dict[str, int], want_errors: dict[str, int], want_p1: str
) -> None:
    """C6 (review row 6, the orchestrator's reading: any Part 1 engine error in either arm makes Part 1 INCONCLUSIVE;
    kills "errors ignored" and "errors counted for BASE only")."""
    p1 = api("part1_inputs")(_samples(errored))
    assert dict(p1["errors"]) == want_errors, (case, p1)
    got = api("decide")(p1, _p2())
    assert got["part1"] == want_p1, (case, got)
    assert got["conclusive"] is (want_p1 != "INCONCLUSIVE"), (case, got)


# --- row 8: order_part2 -------------------------------------------------------------------------------------------


def test_order_part2_pinned() -> None:
    """C4 (review row 8): Part 2's order is every (arm, task, repeat) of BASE/TIP x S6..S15 x range(3), arm-major,
    one joint ``random.Random(0).shuffle``; 60 turns, deterministic, another seed differs, the arms interleave."""
    h = harness()
    assert tuple(h.S.TASK_IDS) == SUITE_IDS
    want = [(arm, t, r) for arm in ("BASE", "TIP") for t in SUITE_IDS for r in range(3)]
    random.Random(0).shuffle(want)
    got = [tuple(x) for x in h.order_part2(0)]
    assert got == want
    assert got != [tuple(x) for x in h.order_part2(1)]
    assert sum(1 for x, y in itertools.pairwise(got) if x[0] != y[0]) > 15


def test_order_part2_json_written_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C4 (review row 8): ``order_part2.json`` is on disk before the engine is asked and holds the pinned order."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    capsys.readouterr()
    assert rc == 1
    want = [(arm, t, r) for arm in ("BASE", "TIP") for t in SUITE_IDS for r in range(3)]
    random.Random(0).shuffle(want)
    written = json.loads((rig.out / "order_part2.json").read_text(encoding="utf-8"))
    assert [tuple(x) for x in written] == want
