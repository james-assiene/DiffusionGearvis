"""T-0174 group A, A7: the W2-0 step-2 eval case, text only (docs/tasks/T-0174.md).

The fixture ``tests/fixtures/t0174/w2_0_step2.json`` (provenance, sha256 and the extraction script in
``tests/fixtures/t0174/MANIFEST.md``) holds the T-0160 web smoke's W2-0 step-2 request (session-log seq 53, image
parts replaced by ``"<image omitted>"``), the step-1 tool call and response, and the step-2 first-attempt thought.
The rows:

- (a) C2's ``DOUBT_RE``, as amended 2026-10-27 (group A review B: "latest"/"current" and optional quotes around the
  adjective in the first two alternatives; typed below, ``test_doubt_re_is_c2s`` checks it against the task file's
  text), matches the logged thought; ``test_doubt_re_amended_examples`` pins the amendment's positives and its
  narrow third alternative. Recalibrated on ``t0136-outcome-ab-97c444a/thoughts.jsonl`` (step >= 2 first attempts,
  per trial): A 30/100, B 26/100 (the pre-amendment regex: 29 and 24);
- (b) the logged system text equals the system text ``cli`` builds today for a live session with v1: with no flag
  and with ``--live-guidance v1``;
- (c) building with ``--live-guidance v2`` changes that text only in the guidance span: it is the logged text with
  its trailing ``LIVE_TOOL_GUIDANCE`` replaced by ``LIVE_GUIDANCE["v2"]``;
- (d) A2's forward-pointing regex matches nothing in v2 (amendment A2b: the whole of v2; before A2b only the
  replacement sentence), and matches both v1 phrases the model quoted back in the logged thought: v1's first
  sentence and "check that the new screenshot shows its effect".
  The flag-v1 (b), (c) and (d) rows were written ahead as pending and activated for group A (87172bda); (d) was red
  against 31b91d7d and is green from f23bb148 (A2b implemented).

CPU only: the ``gpu`` marker gates selection (every tests/eval row is deselected without ``-m gpu``, T-0046), as the
T-0142b, T-0170 and T-0172 evals. Run: ``uv run pytest tests/eval/test_t0174_w2_0_case.py -m gpu -rxX``.

Regime of the logged case: T-0160 web smoke at abcf17d, lean, canvas 256, 48 steps, thinking on, NVFP4 rev 2ea8372,
vLLM 0.29.0, tools native live, loop 8, guidance v1, n = 1, engine unseeded. The rows themselves: CPU over fakes,
deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from tests.unit.test_cli_t0131 import build as build_mode

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "t0174" / "w2_0_step2.json"
MANIFEST = REPO / "tests" / "fixtures" / "t0174" / "MANIFEST.md"
TASK = REPO / "docs" / "tasks" / "T-0174.md"
FIXTURE_SHA = "7aa1f9e3748c96c9200edcdc6ddcf51a505a93a4a737915e610822279db883ff"
SOURCE_SHA = "9b870f46a49f66b5a78864eb4df78263bc1664b4256dc64997e00173b53e2724"
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'

#: C2's pre-registered metric regex, exactly (U+2019 written as an escape)
DOUBT_TEXT = (
    "(?i)(?:do not|don't|don’t|cannot|can't|can’t|haven't|have not) (?:yet )?(?:see|seen|have|got) "
    "(?:the |a |any )?[\"“”']?(?:new|next|updated|resulting|latest|current)[\"“”']? screenshot"
    "|wait(?:ing)? for (?:the |a )?[\"“”']?(?:new|next|updated|latest|current)[\"“”']? screenshot"
    "|(?:new|next|updated) screenshot (?:yet|has not|hasn't|is not|isn't)"
)
DOUBT_RE = re.compile(DOUBT_TEXT)
#: A2's forward-pointing regex, typed from the criterion
FORWARD_RE = re.compile(
    r"(?i)\b(next|new|upcoming|following)\s+screenshot|\bwill\s+(show|see|appear)|\bwait\b"
)
V1_FIRST = "After an action, the next screenshot shows the screen as it is now."
#: v1's verification clause, as the model quoted it in the W2-0 thought (A2b)
V1_CHECK = "check that the new screenshot shows its effect"


def case() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def logged_system() -> str:
    msg = case()["request"]["prompt"]["messages"][0]
    assert msg["role"] == "system"
    text: str = msg["content"][0]["text"]
    return text


def cli_system(tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str) -> str:
    calls = build_mode(tmp_path, capsys, monkeypatch, "off", *flags, body=LIVE_PROFILE)
    text: str = calls.graphs[-1].session.prompts.system_message().text()
    return text


def v2() -> str:
    m = getattr(pb, "LIVE_GUIDANCE", None)
    assert m is not None, "T-0174 A2: prompt_builder.LIVE_GUIDANCE does not exist yet"
    return str(m["v2"])


# --------------------------------------------------------------------------------------------- premises (active)


def test_fixture_provenance() -> None:
    """Active: the fixture's bytes are the ones the MANIFEST records; it names its source and the source's sha; it is
    the step-2 ``tool_step`` request with no image bytes; the step-1 response is the logged ``type_at`` response and
    the request's final assistant message carries the same call and action."""
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert (
        FIXTURE_SHA in manifest and SOURCE_SHA in manifest and "20261027-160402-470048" in manifest
    )
    c = case()
    assert c["provenance"]["source_sha256"] == SOURCE_SHA and c["provenance"]["request_seq"] == 53
    req = c["request"]
    assert req["phase"] == "tool_step" and req["step"] == 2 and req["request_id"] == "t1-gen-s2"
    assert c["step2_first_attempt"]["attempt_id"] == req["attempt_id"]
    text = raw.decode("utf-8")
    assert "data:image" not in text and "base64" not in text and '"image_url"' not in text
    images = [p for m in req["prompt"]["messages"] for p in m["content"] if isinstance(p, str)]
    assert images == ["<image omitted>"], images
    resp = c["step1_tool_response"]
    assert resp.startswith("<|tool_response>response:type_at{") and "executed" in resp
    final = req["prompt"]["messages"][-1]
    assert final["role"] == "assistant" and req["prompt"]["continue_final_message"] is True
    assert c["step1_tool_call"]["action"] in final["content"][0]["text"]


def test_doubt_re_is_c2s() -> None:
    """Active: the typed ``DOUBT_RE`` is C2's backticked text in the task file, character for character."""
    task = TASK.read_text(encoding="utf-8")
    found = re.findall(r"(?m)^`(\(\?i\)\(\?:do not.*)`$", task)
    assert found == [DOUBT_TEXT], found


# --------------------------------------------------------------------------------------------- (a), (b) active


#: C2 amendment (group A review B): positives the amended regex must count, the reviewer's three plus the curly-quote
#: and "waiting for the latest" forms; and a content description it must not count (the third alternative is narrow)
DOUBT_POSITIVE = (
    "I don't have the latest screenshot yet",
    'Wait, I don\'t have the "new" screenshot in the prompt.',
    "I don't have the current screenshot.",
    "I don’t see the “updated” screenshot here.",
    "I am waiting for the latest screenshot.",
)
DOUBT_NEGATIVE = (
    "the latest screenshot is not showing the cart",
    "The latest screenshot shows the cart count changed to 2.",
    "Looking at the current screenshot, the button is at the bottom.",
)


def test_doubt_re_amended_examples() -> None:
    """C2 amendment: the amended regex counts a doubt echoed in v2's words ("latest", "current") and quoted
    adjectives; it does not count a description of the latest screenshot's content. Power: C2's pre-amendment regex
    misses the first three positives (it had no "latest"/"current" and no quotes)."""
    old = re.compile(
        "(?i)(?:do not|don't|don’t|cannot|can't|can’t|haven't|have not) (?:yet )?(?:see|seen|have|got) "
        "(?:the |a |any )?(?:new|next|updated|resulting) screenshot|wait(?:ing)? for (?:the |a )?(?:new|next|updated) "
        "screenshot|(?:new|next|updated) screenshot (?:yet|has not|hasn't|is not|isn't)"
    )
    for text in DOUBT_POSITIVE:
        assert DOUBT_RE.search(text), f"C2: the amended DOUBT_RE misses {text!r}"
    for text in DOUBT_NEGATIVE:
        hit = DOUBT_RE.search(text)
        assert hit is None, (
            f"C2: the amended DOUBT_RE counts a content description {text!r} ({hit.group(0)!r})"
        )
    for text in DOUBT_POSITIVE[:3]:
        assert old.search(text) is None, f"power: the pre-amendment regex already matched {text!r}"


def test_a_doubt_matches_logged_thought() -> None:
    """A7 (a): ``DOUBT_RE`` matches the logged step-2 first-attempt thought ("I don't have the new screenshot yet");
    it does not match the step's spoken answer (power)."""
    c = case()
    hit = DOUBT_RE.search(c["step2_first_attempt"]["thought"])
    assert hit is not None, "A7 (a): C2's regex does not match the logged thought"
    assert hit.group(0).lower() == "don't have the new screenshot", hit.group(0)
    assert DOUBT_RE.search(c["step2_first_attempt"]["answer"]) is None


def test_b_logged_system_is_cli_v1_default(tmp_path, capsys, monkeypatch) -> None:
    """A7 (b), active: the logged system text is the one ``cli`` builds today for a default live session, and it ends
    with v1 (768a973b...)."""
    logged = logged_system()
    built = cli_system(tmp_path, capsys, monkeypatch)
    assert built == logged, "A7 (b): cli's live system text is not the logged one"
    assert logged.endswith("\n" + pb.LIVE_TOOL_GUIDANCE)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == V1_SHA


# --------------------------------------------------------------------------------------------- the v2 rows


def test_b_logged_system_is_cli_flag_v1(tmp_path, capsys, monkeypatch) -> None:
    """A7 (b): ``--live-guidance v1`` builds the logged system text too."""
    built = cli_system(tmp_path, capsys, monkeypatch, "--live-guidance", "v1")
    assert built == logged_system()


def test_c_v2_changes_only_guidance_span(tmp_path, capsys, monkeypatch) -> None:
    """A7 (c): ``--live-guidance v2`` builds the logged system text with its v1 span replaced by v2, and nothing else
    changed."""
    new = v2()
    logged = logged_system()
    built = cli_system(tmp_path, capsys, monkeypatch, "--live-guidance", "v2")
    v1 = pb.LIVE_TOOL_GUIDANCE
    assert logged.count(v1) == 1, "premise: v1 appears once in the logged text"
    i = logged.index(v1)
    assert built == logged[:i] + new + logged[i + len(v1) :], (
        "A7 (c): the v2 system text differs from the logged one outside the guidance span"
    )
    assert built != logged


def test_d_forward_regex() -> None:
    """A7 (d) with A2b: A2's forward regex matches both v1 sentences the model quoted back in the logged thought (the
    first sentence and "check that the new screenshot shows its effect") and matches nothing in v2."""
    new = v2()
    thought = case()["step2_first_attempt"]["thought"]
    assert pb.LIVE_TOOL_GUIDANCE.startswith(V1_FIRST + " ")
    assert FORWARD_RE.search(V1_FIRST), "A7 (d): v1's first sentence matches the forward regex"
    assert V1_FIRST in thought, "premise: the model quoted v1's first sentence"
    assert V1_CHECK in pb.LIVE_TOOL_GUIDANCE and V1_CHECK in thought, (
        "premise: the model quoted v1's verification clause"
    )
    assert FORWARD_RE.search(V1_CHECK), "A7 (d): v1's verification clause matches the forward regex"
    hit = FORWARD_RE.search(new)
    assert hit is None, f"A7 (d)/A2b: v2 matches the forward regex ({hit.group(0)!r}): {new!r}"
