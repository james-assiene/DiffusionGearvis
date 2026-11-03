"""T-0103: the harness's ``main --table T-0103`` end to end against a mock engine on loopback, every criterion-3
refusal, and ``--table T-0098c`` / ``--table T-0098`` on the same mock (CPU only; ``integration``, ``slow``).

Rows that need the T-0103 table are red on the named assertion in ``harness_103()`` until the table exists; the run
of the earlier tables on this mock passes before and after the implementation (criterion 2).

The harness is loaded lazily by path and ``main(argv)`` is called in-process (running anything under the GPU
directory from a shell is gated; this run spends no GPU time). The T-0098 / T-0098c smoke helpers are imported
read-only: ``EngineC`` (subclassed here), ``no_sockets``, ``serve`` (refusal rows: the engine must never be contacted),
``_kit_loader``, ``REFUSALS``, ``_rebind``, T-0093's ``native_call`` and ``require_clean_tree``.

**The mock engine** places each request by its ``messages[0]`` (the T-0098 and T-0098c texts built by hand in the
unit modules; T-0103's A0 and C3 are T-0098c's) and its user text, and reads the repeat from the request id the client
sends (``vllm_xargs.astra_request_id``, ``...-r{repeat}-...``):

- a ``type_at`` case under **A0 only**: SILENT (the plan inside a closed thought, then the call); every other tool
  trial, every arm: SPEAKS;
- a Q12 positive: a closed thought, ``Here it is.``, ``<a2ui>`` + ``UI_EXAMPLE`` + ``</a2ui>`` -- except **C3 on
  primes_table at repeats 5-14**, which answers in speech only;
- Q12 negatives, no-tool and plain cases: a closed thought and ``Okay.``.

**Expected for the T-0103 run, by hand:** 880 requests in the schedule of *Design*; A0 ``type_at`` 0/50 (R1 holds), C3
50/50; Q12 positive shown A0 160/160 (R3 holds), C3 150/160 (primes_table 30/40); negatives 0/15; the block split for
C3: r 0-4 20/20, r 5-39 130/140. P8: 150 >= 160 - 16 = 144 and primes_table 30 >= 24, so it passes (a literal margin of
2 fails it); every other P passes: **"C3 confirmed"**, winner C3.
**On the same mock, T-0098c:** 2100 requests, rotation mod 7, ids ``t0098c-...``; only A0 is silent and every arm
shows on all 20 positives (repeats 0-4 only), so C3 passes P1-P11 and wins; the report has T-0098c's keys (no
``outcome``). **T-0098:** 1590 requests, mod 6, ids ``t0098-...``; C2 passes and wins.

Regime: CPU, mock engine on loopback, real ``PromptBuilder``, ``VllmClient`` (id mode, the pinned detokeniser) and
``UiGate``; no measurement -- the numbers describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import re
import threading
from collections.abc import Iterator, Mapping
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN
from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.toolcall.detok import pinned_tokenizer_path
from tests.integration.test_t0093_harness_smoke import native_call, require_clean_tree
from tests.integration.test_t0098_harness_smoke import (
    REFUSALS,
    STOP_TOOL,
    STOP_TURN,
    _has_key,
    _kit_loader,
    _outside_system,
    _sha,
    _text_of,
    no_sockets,
    serve,
)
from tests.integration.test_t0098c_harness_smoke import MEMORY_MARK, Q12_SHA, EngineC, _rebind
from tests.unit._pre_t0155_persona import pre_t0155_fixture
from tests.unit.test_t0098_prompt_ab import A0_SHA256
from tests.unit.test_t0098c_ui_prompt_ab import Q12_FILE, Q12_NEGATIVE, Q12_POSITIVE, harness_c
from tests.unit.test_t0103_c3_confirm import (
    C3_SHA256,
    N_REQUESTS,
    PS_C,
    RULE,
    TABLE,
    a0_c3_by_hand,
    expected_repeats,
    expected_schedule,
    harness_103,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

REPEAT_IN_ID = re.compile(r"-r(\d+)-")
C3_SPEECH_ONLY = ("primes_table", range(5, 15))


class Engine103(EngineC):
    """T-0098c's mock engine with T-0103's script (module docstring)."""

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        system = _text_of(messages[0].get("content")) if messages else ""
        arm = self.arm_of.get(system)
        user = _text_of(messages[-1].get("content")) if messages else ""
        rid = str((body.get("vllm_xargs") or {}).get("astra_request_id") or "")
        found = REPEAT_IN_ID.search(rid)
        repeat = int(found.group(1)) if found else None
        row: dict[str, Any] | None
        q12 = [r for r in self.q12 if r["request"] in user]
        if q12:
            with_memory = MEMORY_MARK in user
            hit = next(r for r in q12 if bool(r["memory_captions"]) is with_memory)
            row = {"id": hit["id"], "kind": "q12", "expect_surface": bool(hit["expect_surface"])}
        else:
            row = next((r for text, r in self.by_text if text in user), None)
        if arm is None or row is None:
            self.unknown.append(
                f"arm={arm} case={row and row['id']} system_sha={_sha(system)[:12]}"
            )
        thought = "<|channel>thought\nOK.<channel|>"
        if row is not None and row["kind"] == "tool":
            call = native_call(row["reference_call"])
            if arm == "A0" and row["tool"] == "type_at":
                text = "<|channel>thought\nI will use type_at with press_enter.<channel|>" + call
            else:
                text = thought + "I will do that.<channel|>" + call
            ids = [*self.encode(text), STOP_TOOL]
        elif row is not None and row["kind"] == "q12" and row["expect_surface"]:
            speech_only = (
                arm == "C3" and row["id"] == C3_SPEECH_ONLY[0] and repeat in C3_SPEECH_ONLY[1]
            )
            if speech_only:
                ids = [*self.encode(thought + "Okay."), STOP_TURN]
            else:
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
            "repeat": repeat,
            "system": system,
            "user": user,
            "ids": ids,
        }


@contextlib.contextmanager
def serve_103() -> Iterator[tuple[Engine103, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine103()
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0103-mock-engine", daemon=True)
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


def _find(value: Any, want: Any) -> bool:
    if value == want:
        return True
    if isinstance(value, Mapping):
        return any(_find(v, want) for v in value.values())
    if isinstance(value, list):
        return any(_find(v, want) for v in value)
    return False


# =====================================================================================================
# The healthy T-0103 run
# =====================================================================================================


def test_main_t0103_runs_880_in_schedule_order(tmp_path: Path) -> None:
    """Criteria 1, 4, 5 and 7 on one complete ``--table T-0103`` run: exit 0; 880 requests, pass 1, tools on, no seed,
    no test-split case, in exactly the schedule of *Design* (repeats outermost, scheduled cases in index order, A0
    first iff (case_index + repeat) is even); each (case, repeat) pair differs only in ``messages[0]``; Q12 requests
    carry the ``[screen]`` note and the memory case its captions; ``trials.jsonl`` mirrors the requests with ids
    ``t0103-r{r}-{case}-{arm}``; mock ``ui`` canvases arrive as one ``ui`` commit and are shown; the manifest records
    table, rule, the two texts and shas, the repeats map, the scaled thresholds as k/n strings and a ``mod 2``
    rotation; ``report.json`` is ``build_report(run_dir)`` and reads the module docstring's numbers and outcome."""
    h = harness_103()
    require_clean_tree()
    out = tmp_path / "t0103-c3"
    with serve_103() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", TABLE]
        )
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    chats = eng.chats
    want = expected_schedule()
    assert len(chats) == N_REQUESTS, f"criterion 4: {len(chats)} chat requests"
    assert [(c["case_id"], c["repeat"], c["arm"]) for c in chats] == [
        (cid, r, arm) for _, cid, r, _, arm in want
    ], "criterion 4: the requests are not in the schedule of *Design*"
    for c in chats:
        b = c["body"]
        assert b.get("max_tokens") != 1 and not b.get("continue_final_message"), "pass 1 only"
        assert not _has_key(b, "seed"), "a seed field was sent"
        assert b.get("tool_choice") == "none" and b.get("tools"), "all tools on"
        assert not any(t in c["user"] for t in eng.test_texts), "a test-split case was sent"
        if c["case_id"] in (*Q12_POSITIVE, *Q12_NEGATIVE):
            assert pb.NO_IMAGE_STATEMENT in c["user"], c["case_id"]
            assert (MEMORY_MARK in c["user"]) is (c["case_id"] == "flight_booking_with_memory")
    for k in range(0, len(chats), 2):
        a, b = chats[k], chats[k + 1]
        assert (a["case_id"], a["repeat"]) == (b["case_id"], b["repeat"]) and {
            a["arm"],
            b["arm"],
        } == {
            "A0",
            "C3",
        }
        assert _outside_system(a["body"]) == _outside_system(b["body"]), (
            f"criterion 4: {a['case_id']} r{a['repeat']} differs outside messages[0]"
        )
    assert {_sha(c["system"]) for c in chats if c["arm"] == "A0"} == {A0_SHA256}
    assert {_sha(c["system"]) for c in chats if c["arm"] == "C3"} == {C3_SHA256}

    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == N_REQUESTS
    for k, (t, c) in enumerate(zip(trials, chats, strict=True)):
        _, cid, r, slot, arm = want[k]
        assert (t["case_id"], t["arm"], t["repeat"], t["order"]) == (cid, arm, r, slot), k
        assert t["request_id"] == f"t0103-r{r}-{cid}-{arm}"
        assert t["token_ids"] == c["ids"]
        if cid in Q12_POSITIVE and not (
            arm == "C3" and cid == C3_SPEECH_ONLY[0] and r in C3_SPEECH_ONLY[1]
        ):
            assert [m["text"] for m in t["commits"] if m["segment"] == "ui"] == [UI_EXAMPLE], k

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest.get("table") == TABLE and manifest.get("rule") == RULE
    a0, c3 = a0_c3_by_hand()
    got = {a: (v["text"], v["sha256"]) for a, v in manifest["arms"].items()}
    assert got == {"A0": (a0, A0_SHA256), "C3": (c3, C3_SHA256)}
    assert _find(manifest, expected_repeats()), "criterion 1: the manifest lacks the repeats map"
    flat = json.dumps(manifest)
    for kn in ("16/160", "24/40", "8/160", "10/175", "112/160"):
        assert f'"{kn}"' in flat, f"criterion 1: the manifest lacks the scaled threshold {kn}"
    assert "mod 2" in flat and "mod 7" not in flat and "mod 6" not in flat, "criterion 1: rotation"

    stored = json.loads((out / "report.json").read_text())
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "report.json is build_report(run_dir)"
    )
    assert (out / "report.md").is_file()
    assert stored["complete"] is True and list(stored["void_arms"]) == []
    assert stored["readings"] == {"R1": True, "R3": True}
    arms = stored["arms"]
    assert arms["A0"]["tools"]["type_at"]["spoken"] == {"k": 0, "n": 50}
    assert arms["C3"]["tools"]["type_at"]["spoken"] == {"k": 50, "n": 50}
    assert arms["A0"]["q12_positive_shown"] == {"k": 160, "n": 160}
    assert arms["C3"]["q12_positive_shown"] == {"k": 150, "n": 160}
    assert arms["C3"]["q12_per_case"]["primes_table"] == {"k": 30, "n": 40}
    for arm in ("A0", "C3"):
        assert arms[arm]["q12_negative_shown"] == {"k": 0, "n": 15}, arm
    assert _find(stored, {"k": 130, "n": 140}), "criterion 7: C3's r 5-39 block split"
    assert [p for p in PS_C if stored["candidates"]["C3"][p] is not True] == []
    assert str(stored["outcome"]).startswith("C3 confirmed") and stored["winner"] == "C3"
    assert str(stored["outcome"]) in (out / "report.md").read_text()


# =====================================================================================================
# Criterion 2: the earlier tables on the same mock (guard; green today)
# =====================================================================================================


@pytest.mark.parametrize(
    ("table", "n", "arms", "prefix", "mod", "winner"),
    [
        ("T-0098c", 2100, ("A0", "D1", "D3", "D4", "C3", "C5", "C4"), "t0098c", "mod 7", "C3"),
        ("T-0098", 1590, ("A0", "A1", "D1", "D2", "C2", "C1"), "t0098", "mod 6", "C2"),
    ],
)
def test_earlier_tables_on_the_same_mock(
    tmp_path: Path, table: str, n: int, arms: tuple[str, ...], prefix: str, mod: str, winner: str
) -> None:
    """Criterion 2: the same mock under ``--table T-0098c`` still writes 2100 lines and under ``--table T-0098`` 1590,
    5 repeats each, in the rotation of their own table, with their own request-id prefix, the manifest's rotation
    string (``mod 7`` / ``mod 6``) and their own winner (only A0 is silent: C3 under T-0098c, C2 under T-0098); the
    report keeps its table's keys (no T-0103 ``outcome``)."""
    h = harness_c()
    require_clean_tree()
    out = tmp_path / f"{prefix}-ab"
    with serve_103() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", table]
        )
    assert attempts == [] and eng.unknown == [] and rc == 0, (attempts, eng.unknown[:3], rc)
    assert len(eng.chats) == n
    order = [c.case_id for c in h.load_cases(table=table)]
    k = 0
    for r in range(5):
        for i, cid in enumerate(order):
            off = (i + r) % len(arms)
            for slot in range(len(arms)):
                c = eng.chats[k]
                assert (c["case_id"], c["arm"], c["repeat"]) == (
                    cid,
                    arms[(off + slot) % len(arms)],
                    r,
                ), (
                    table,
                    k,
                )
                k += 1
    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == n
    assert all(str(t["request_id"]).startswith(f"{prefix}-r") for t in trials)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest.get("table") == table
    assert mod in manifest["regime"]["rotation"] and manifest["regime"]["repeats"] == 5
    stored = json.loads((out / "report.json").read_text())
    assert "outcome" not in stored
    assert stored["winner"] == winner


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =====================================================================================================
# Criterion 3: the T-0103 refusals, all before any socket
# =====================================================================================================

C3_REFUSALS: dict[str, tuple[str, ...]] = {
    **{f"t0098-{k}": (v,) for k, v in REFUSALS.items()},
    "q12-sha": ("surface_placement_cases",),
    "q12-not-4-plus-3": ("surface_placement_cases", "Q12"),
    "c3-not-the-join": ("C3", "72c3b92b"),
    "repeats-39-on-one-positive": ("repeat",),
}


@pytest.mark.parametrize("why", list(C3_REFUSALS))
def test_main_t0103_refuses_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criterion 3: under ``--table T-0103`` every T-0098c refusal still holds -- loopback, dirty tree, non-empty
    out-dir, ``--split test``, the ``cases.jsonl`` pin, the tool kit (unloadable, repair off, wrong set), A0's sha (one
    sentence added to ``SPOKEN_STYLE``), the Q12 pin and its 4 + 3 shape -- and so do T-0103's: a C3 text that is
    not exactly join(P, CT, SS, N, U) (here A0's order, from a replaced builder), and a repeats map with 39 on
    ``primes_table``. Each: exit 2, the engine never contacted, no socket at all, nothing written, the refusal names
    its cause. Over a real clean tree."""
    h = harness_103()
    require_clean_tree()
    out = tmp_path / "t0103-out"
    probe = Path(h.REPO) / "tests" / "integration" / "_t0103_dirty_probe.txt"
    base = why.removeprefix("t0098-")
    with serve() as (eng, base_url):
        argv = ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", TABLE]
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
            assert _rebind(monkeypatch, [h, h._T93], Q12_SHA, "0" * 64) >= 1, "premise: the Q12 pin"
        elif why == "q12-not-4-plus-3":
            rows = [x for x in Q12_FILE.read_text(encoding="utf-8").splitlines() if x.strip()][:6]
            fake = tmp_path / "surface_placement_cases.jsonl"
            fake.write_text("\n".join(rows) + "\n", encoding="utf-8")
            fake_sha = _sha_bytes(fake.read_bytes())
            assert _rebind(monkeypatch, [h, h._T93], Q12_FILE, fake) >= 1, "premise: the Q12 path"
            assert _rebind(monkeypatch, [h, h._T93], Q12_SHA, fake_sha) >= 1, "premise: the Q12 pin"
        elif why == "c3-not-the-join":
            t = h.TABLES[TABLE]
            orig = t.texts

            def texts(tools: Any) -> dict[str, str]:
                got = dict(orig(tools))
                got["C3"] = got["A0"]
                return got

            monkeypatch.setitem(h.TABLES, TABLE, dataclasses.replace(t, texts=texts))
        elif why == "repeats-39-on-one-positive":
            t = h.TABLES[TABLE]
            monkeypatch.setitem(
                h.TABLES,
                TABLE,
                dataclasses.replace(t, repeats={**dict(t.repeats), "primes_table": 39}),
            )
        try:
            if base == "dirty-tree":
                probe.write_text("dirty\n")
            with no_sockets() as attempts:
                rc = h.main(argv)
        finally:
            probe.unlink(missing_ok=True)
    said = capsys.readouterr()
    assert rc == 2, f"criterion 3 ({why}): exit {rc}"
    assert attempts == [], f"criterion 3 ({why}): sockets before the refusal: {attempts}"
    assert eng.requests == [], f"criterion 3 ({why}): the engine was contacted: {eng.requests}"
    left = sorted(p.name for p in out.iterdir()) if out.exists() else []
    assert left in ([], ["earlier.txt"]), f"criterion 3 ({why}): the refusal wrote {left}"
    tokens = tuple(
        tok.replace("_t0098_dirty_probe", "_t0103_dirty_probe").replace("t0098-out", "t0103-out")
        for tok in C3_REFUSALS[why]
    )
    if base == "a0-sha-drift":
        tokens = (*tokens, "72c3b92b")
    if base == "wrong-tool-set":
        tokens = (*tokens, "000000000000")
    text = said.out + said.err
    assert any(tok in text for tok in tokens), (
        f"criterion 3 ({why}): the refusal does not name its cause {tokens}: {text[-300:]!r}"
    )
