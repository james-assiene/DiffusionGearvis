"""T-0098c: the harness's ``main --table T-0098c`` end to end against a mock engine on loopback, every T-0098c
refusal of criterion 2, and ``--table T-0098`` on the same mock (CPU only; ``integration``, ``slow``).
Pending until ``/implement T-0098c`` started (marker removed then).

The harness is loaded lazily by path and ``main(argv)`` is called in-process (running anything under the GPU
directory from a shell is gated; this run spends no GPU time). T-0098's smoke helpers are imported read-only:
``Engine`` (subclassed here), ``no_sockets``, ``serve`` (refusal rows only: the engine must never be contacted),
``_kit_loader``, ``REFUSALS``, and T-0093's ``Encoder``-backed ``native_call`` / ``require_clean_tree``.

**The mock engine** answers by the request's ``messages[0]`` (matched against the seven T-0098c texts and the six
T-0098 texts, all built by hand in the unit modules; an unknown system text is recorded and fails the row) and its
user text (a Q12 request first -- the memory variant of "Make a flight booking UI." is told by its captions -- then
T-0098's dev and plain texts):

- a ``type_at`` case under **A0, C3** (and T-0098's C2): SILENT -- the plan inside a closed thought, then the call;
- every other tool case: SPEAKS -- a closed thought, ``I will do that.``, ``<channel|>``, the call;
- a Q12 positive: a closed thought, ``Here it is.``, then ``<a2ui>`` + ``ui_instructions.UI_EXAMPLE`` + ``</a2ui>``;
- Q12 negatives, no-tool and plain cases: a closed thought and ``Okay.``.

**Expected for the T-0098c run, by hand:** 2100 requests in the rotated order of criterion 3; A0 ``type_at`` 0/50
(R1 holds), D1 50/50 (R2c holds), A0 Q12 positive shown 20/20 (R3 holds) and negative 0/15, every arm alike on Q12;
``ui.gate`` per arm gated 20, emitted 20, failed 0; C3 fails P1 only (0/50); C5 passes P1-P11; the winner is **C5**.
**For the T-0098 run on the same mock:** 1590 requests; A0 and C2 silent on ``type_at``, so C2 fails P1 and C1 wins.

Regime: CPU, mock engine on loopback, real ``PromptBuilder``, ``VllmClient`` (id mode, the pinned detokeniser) and
``UiGate``; no measurement -- the numbers describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections import Counter
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN
from local_astra.conductor.ui_instructions import UI_EXAMPLE, UI_INSTRUCTIONS
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.integration.test_t0093_harness_smoke import native_call, require_clean_tree
from tests.integration.test_t0098_harness_smoke import (
    REFUSALS,
    STOP_TOOL,
    STOP_TURN,
    Engine,
    _has_key,
    _kit_loader,
    _outside_system,
    _sha,
    _text_of,
    no_sockets,
    serve,
)
from tests.unit._pre_t0155_persona import pre_t0155_fixture
from tests.unit.test_t0098_prompt_ab import A0_SHA256, expected_arm_texts
from tests.unit.test_t0098c_ui_prompt_ab import (
    ANCHOR_EDITS,
    ARMS_C,
    CANDIDATES_C,
    PS_C,
    Q12_FILE,
    Q12_NEGATIVE,
    Q12_POSITIVE,
    REPEATS,
    case_rows_c,
    expected_arm_texts_c,
    harness_c,
    patch_u,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

SILENT_ARMS = frozenset({"A0", "C3", "C2"})
MEMORY_MARK = "Exercises and Problems in Calculus"


class EngineC(Engine):
    """T-0098's mock engine, taught the T-0098c arms and the Q12 requests."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_of = {text: arm for arm, text in expected_arm_texts().items()}
        self.arm_of.update({text: arm for arm, text in expected_arm_texts_c().items()})
        self.q12 = [
            json.loads(x) for x in Q12_FILE.read_text(encoding="utf-8").splitlines() if x.strip()
        ]

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        system = _text_of(messages[0].get("content")) if messages else ""
        arm = self.arm_of.get(system)
        user = _text_of(messages[-1].get("content")) if messages else ""
        row: dict[str, Any] | None = None
        q12 = [r for r in self.q12 if r["request"] in user]
        if q12:
            with_memory = MEMORY_MARK in user
            row = next(r for r in q12 if bool(r["memory_captions"]) is with_memory)
            row = {"id": row["id"], "kind": "q12", "expect_surface": bool(row["expect_surface"])}
        else:
            row = next((r for text, r in self.by_text if text in user), None)
        if arm is None or row is None:
            self.unknown.append(
                f"arm={arm} case={row and row['id']} system_sha={_sha(system)[:12]}"
            )
        thought = "<|channel>thought\nOK.<channel|>"
        if row is not None and row["kind"] == "tool":
            call = native_call(row["reference_call"])
            if arm in SILENT_ARMS and row["tool"] == "type_at":
                text = "<|channel>thought\nI will use type_at with press_enter.<channel|>" + call
            else:
                text = thought + "I will do that.<channel|>" + call
            ids = [*self.encode(text), STOP_TOOL]
        elif row is not None and row["kind"] == "q12" and row["expect_surface"]:
            ids = [
                *self.encode(thought + "Here it is." + UI_OPEN + UI_EXAMPLE + UI_CLOSE),
                STOP_TURN,
            ]
        else:
            ids = [*self.encode(thought + "Okay."), STOP_TURN]
        return {
            "body": body,
            "arm": arm,
            "case_id": row["id"] if row else None,
            "system": system,
            "user": user,
            "ids": ids,
        }


@contextlib.contextmanager
def serve_c() -> Iterator[tuple[EngineC, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = EngineC()
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0098c-mock-engine", daemon=True)
    thread.start()
    try:
        yield eng, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _loopback_only() -> Any:
    from tests.unit.test_session_no_egress import loopback_only

    return loopback_only()


# =====================================================================================================
# The healthy runs
# =====================================================================================================


def test_main_t0098c_runs_end_to_end_against_a_mock_engine(tmp_path: Path) -> None:
    """Criteria 1, 3-7 on one complete ``--table T-0098c`` run (2100 requests): exit 0; pass 1 only, tools on, no seed,
    no test-split case; each (case, repeat) block sends the seven arms from offset ``(case_index + repeat) mod 7`` and
    its bodies differ only in ``messages[0]``; Q12 requests carry the ``[screen]`` note, and the memory case its
    captions; ``trials.jsonl`` mirrors the requests line for line with a request id; Q12 positive lines hold a ``ui``
    commit; the manifest records table T-0098c, a rule and the seven hand-built texts; ``report.json`` is
    ``build_report(run_dir)`` and reads the script's decision (module docstring: winner C5)."""
    h = harness_c()
    require_clean_tree()
    out = tmp_path / "t0098c-ab"
    with serve_c() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", "T-0098c"]
        )
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    chats = eng.chats
    assert len(chats) == 60 * 7 * REPEATS == 2100, f"criterion 3: {len(chats)} chat requests"
    for c in chats:
        b = c["body"]
        assert b.get("max_tokens") != 1 and not b.get("continue_final_message"), "pass 1 only"
        assert not _has_key(b, "seed"), "criterion 3: a seed field was sent"
        assert b.get("tool_choice") == "none" and b.get("tools"), "all tools on"
        assert not any(t in c["user"] for t in eng.test_texts), (
            "criterion 10: a test-split case was sent"
        )
        if c["case_id"] in (*Q12_POSITIVE, *Q12_NEGATIVE):
            assert pb.NO_IMAGE_STATEMENT in c["user"], (
                f"criterion 4: {c['case_id']} lacks the [screen] note"
            )
            assert (MEMORY_MARK in c["user"]) is (c["case_id"] == "flight_booking_with_memory")
    assert {_sha(c["system"]) for c in chats if c["arm"] == "A0"} == {A0_SHA256}

    order = [r["id"] for r in case_rows_c()]
    assert len(order) == 60
    seen: Counter[str] = Counter()
    for k in range(0, len(chats), 7):
        block = chats[k : k + 7]
        cid = block[0]["case_id"]
        assert {b["case_id"] for b in block} == {cid}, f"block {k // 7} mixes cases"
        repeat = seen[cid]
        seen[cid] += 1
        off = (order.index(cid) + repeat) % 7
        assert [b["arm"] for b in block] == [ARMS_C[(off + j) % 7] for j in range(7)], (
            f"criterion 3: {cid} r{repeat}"
        )
        ref = _outside_system(block[0]["body"])
        for b in block:
            assert _outside_system(b["body"]) == ref, (
                f"criterion 4: {cid} {b['arm']} differs outside messages[0]"
            )
    assert seen == Counter({cid: REPEATS for cid in order})
    assert [c["case_id"] for c in chats[: 7 * 60 : 7]] == order, "repeat 0 runs the cases in order"

    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == 2100
    for k, (t, c) in enumerate(zip(trials, chats, strict=True)):
        assert (t["case_id"], t["arm"]) == (c["case_id"], c["arm"]), f"line {k} is not request {k}"
        assert t["order"] == k % 7 and t.get("request_id"), (k, t["order"])
        assert t["token_ids"] == c["ids"]
        if t["case_id"] in Q12_POSITIVE:
            assert [m["text"] for m in t["commits"] if m["segment"] == "ui"] == [UI_EXAMPLE], (
                f"line {k}: the mock's fence did not arrive as one ui commit"
            )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest.get("table") == "T-0098c" and manifest.get("rule"), (
        "criterion 1: the manifest's table and rule"
    )
    texts = expected_arm_texts_c()
    got = {a: (v["text"], v["sha256"]) for a, v in manifest["arms"].items()}
    assert got == {a: (t, _sha(t)) for a, t in texts.items()}

    stored = json.loads((out / "report.json").read_text())
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "criterion 6: report.json is build_report(run_dir)"
    )
    assert (out / "report.md").is_file()
    assert [stored["readings"][r] for r in ("R1", "R2c", "R3")] == [True, True, True]
    assert list(stored["void_arms"]) == []
    arms = stored["arms"]
    assert arms["A0"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert arms["C3"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert arms["D1"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    for arm in ARMS_C:
        assert arms[arm]["q12_positive_shown"] == {"k": 20, "n": 20}, arm
        assert arms[arm]["q12_negative_shown"] == {"k": 0, "n": 15}, arm
        assert arms[arm]["ui_gate"] == {"gated": 20, "emitted": 20, "failed": 0}, arm
    assert set(stored["candidates"]) == set(CANDIDATES_C)
    assert [p for p in PS_C if stored["candidates"]["C3"][p] is not True] == ["P1"]
    assert [p for p in PS_C if stored["candidates"]["C5"][p] is not True] == []
    assert stored["winner"] == "C5"


def test_main_t0098_table_on_the_same_mock_still_writes_1590(tmp_path: Path) -> None:
    """Criterion 1: ``--table T-0098`` on the same mock runs T-0098's design unchanged -- 1590 requests in the mod-6
    rotation over the 53 cases, no Q12 case sent, the manifest's table T-0098, and T-0098's decision for this script
    (C2 silent on ``type_at`` fails P1; C1 wins)."""
    h = harness_c()
    require_clean_tree()
    out = tmp_path / "t0098-ab"
    with serve_c() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", "T-0098"]
        )
    assert attempts == [] and eng.unknown == [] and rc == 0, (attempts, eng.unknown[:3], rc)
    assert len(eng.chats) == 1590
    assert not {c["case_id"] for c in eng.chats} & {*Q12_POSITIVE, *Q12_NEGATIVE}
    assert {c["arm"] for c in eng.chats} == {"A0", "A1", "D1", "D2", "C2", "C1"}
    trials = [x for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == 1590
    assert json.loads((out / "manifest.json").read_text()).get("table") == "T-0098"
    stored = json.loads((out / "report.json").read_text())
    assert set(stored["candidates"]) == {"C2", "C1"} and stored["winner"] == "C1"


# =====================================================================================================
# Criterion 2: the T-0098c refusals, all before any socket
# =====================================================================================================


def _rebind(monkeypatch: pytest.MonkeyPatch, modules: list[Any], old: Any, new: Any) -> int:
    """Rebind every module-level name equal to ``old`` (a pin or a path, found by value, never called)."""
    hits = 0
    for module in modules:
        for name, value in list(vars(module).items()):
            if isinstance(value, str | Path) and type(value) is type(old) and value == old:
                monkeypatch.setattr(module, name, new)
                hits += 1
    return hits


Q12_SHA = "e152810b0c779a7eeeb3a5d84947d173cc8415400f36fa1cf7f007749158d6d8"
C_REFUSALS = {
    **{f"t0098-{k}": v for k, v in REFUSALS.items()},
    "q12-sha": "surface_placement_cases",
    "q12-not-4-plus-3": "surface_placement_cases",
    **{f"anchor-{k}": ANCHOR_EDITS[k][0] for k in ANCHOR_EDITS},
    "c4-over-2000": "C4",
}


@pytest.mark.parametrize("why", list(C_REFUSALS))
def test_main_t0098c_refuses_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criterion 2: under ``--table T-0098c`` every T-0098 refusal still holds, and so do the new ones -- a Q12 file
    whose sha is not ``e152810b...``; a Q12 file pinned by its own sha but holding 4 positive + 2 negative (its pin and
    path rebound by value in the harness and in T-0093's module); a C5 or C4 anchor occurring twice or not at all; C4's
    U over 2000 code points (U padded by 24 spaces: 1977 + 24 = 2001). Each: exit 2, the engine never contacted, no
    socket at all, nothing written, and the refusal names its cause. The anchor rows change U, which moves A0's text
    too; the refusal reported must be the anchor's (it names the arm), i.e. the arm texts are checked before A0's
    sha, as T-0098's ``main`` builds them before comparing it."""
    h = harness_c()
    require_clean_tree()
    out = tmp_path / "t0098c-out"
    probe = Path(h.REPO) / "tests" / "integration" / "_t0098c_dirty_probe.txt"
    base = why.removeprefix("t0098-")
    with serve() as (eng, base_url):
        argv = [
            "--base-url",
            base_url,
            "--out-dir",
            str(out),
            "--split",
            "dev",
            "--table",
            "T-0098c",
        ]
        if base == "non-loopback-url":
            argv[1] = "http://10.255.255.1:8001/v1"
        elif base == "non-empty-out-dir":
            out.mkdir()
            (out / "earlier.txt").write_text("an earlier run\n")
        elif base == "unpinned-cases":
            monkeypatch.setattr(h, "CASES_SHA256", "0" * 64)
        elif base == "test-split":
            argv[5] = "test"
        elif base in ("kit-unloadable", "repair-off", "wrong-tool-set"):
            kind = {
                "kit-unloadable": "unloadable",
                "repair-off": "repair-off",
                "wrong-tool-set": "wrong",
            }[base]
            monkeypatch.setattr(cli, "load_tool_kit", _kit_loader(kind))
        elif base == "a0-sha-drift":
            monkeypatch.setattr(pb, "SPOKEN_STYLE", pb.SPOKEN_STYLE + " Keep it short.")
        elif why == "q12-sha":
            hits = _rebind(monkeypatch, [h, h._T93], Q12_SHA, "0" * 64)
            assert hits >= 1, (
                "premise: the Q12 pin is a module-level constant of the harness or T-0093's module"
            )
        elif why == "q12-not-4-plus-3":
            rows = [x for x in Q12_FILE.read_text(encoding="utf-8").splitlines() if x.strip()][:6]
            fake = tmp_path / "surface_placement_cases.jsonl"
            fake.write_text("\n".join(rows) + "\n", encoding="utf-8")
            assert [json.loads(r)["expect_surface"] for r in rows].count(False) == 2, (
                "premise: 4 + 2"
            )
            fake_sha = hashlib.sha256(fake.read_bytes()).hexdigest()
            assert _rebind(monkeypatch, [h, h._T93], Q12_FILE, fake) >= 1, "premise: the Q12 path"
            assert _rebind(monkeypatch, [h, h._T93], Q12_SHA, fake_sha) >= 1, "premise: the Q12 pin"
        elif why.startswith("anchor-"):
            patch_u(monkeypatch, h, ANCHOR_EDITS[why.removeprefix("anchor-")][1])
        elif why == "c4-over-2000":
            patch_u(monkeypatch, h, UI_INSTRUCTIONS + " " * 24)
        try:
            if base == "dirty-tree":
                probe.write_text("dirty\n")
            with no_sockets() as attempts:
                rc = h.main(argv)
        finally:
            probe.unlink(missing_ok=True)
    said = capsys.readouterr()
    assert rc == 2, f"criterion 2 ({why}): exit {rc}"
    assert attempts == [], f"criterion 2 ({why}): sockets before the refusal: {attempts}"
    assert eng.requests == [], f"criterion 2 ({why}): the engine was contacted: {eng.requests}"
    left = sorted(p.name for p in out.iterdir()) if out.exists() else []
    assert left in ([], ["earlier.txt"]), f"criterion 2 ({why}): the refusal wrote {left}"
    token = (
        C_REFUSALS[why]
        .replace("_t0098_dirty_probe", "_t0098c_dirty_probe")
        .replace("t0098-out", "t0098c-out")
    )
    text = said.out + said.err
    ok = token in text or (base == "wrong-tool-set" and "000000000000" in text)
    if why == "q12-not-4-plus-3":
        ok = ok or "Q12" in text
    assert ok, (
        f"criterion 2 ({why}): the refusal does not name its cause ({token!r}): {text[-300:]!r}"
    )
