"""T-0108a: the harness's ``main --table T-0108a`` end to end against a mock engine on loopback, the criterion-3
refusals, and the three earlier tables on the same mock (CPU only; ``integration``, ``slow``).

Rows that need the T-0108a table are red on the named assertion in ``harness_108a()``
until the table exists; the earlier tables' rows (criterion 2) are unmarked and green before and after.

The harness is loaded lazily by path and ``main(argv)`` is called in-process. T-0103's mock (``Engine103``,
``serve_103``) and the T-0098 / T-0098c smoke helpers are imported read-only.

**The mock engine** places each request by its ``messages[0]`` (the three T-0108a texts built by hand in the unit
module) and its user text, and reads the repeat from the request id:

- every tool trial, every arm: speaks, then calls;
- a Q12 positive: a closed thought, ``Here it is.``, then a fenced canvas -- **A0**: ``PLACEHOLDER`` (a Text and a
  TextField with ``variant: "huge"``, unrepairable since T-0108b 9(ii): shown, last call OK, not complete); **K1**: ``UI_EXAMPLE`` except ``primes_table`` at
  repeats 5-14, which gets ``PLACEHOLDER``; **K2**: ``UI_EXAMPLE``;
- Q12 negatives, no-tool and plain cases: a closed thought and ``Okay.``.

**Expected, by hand:** 1320 requests in the schedule of *Design*; complete A0 0/160, K1 150/160 (primes_table 30/40),
K2 160/160; last-OK and shown 160/160 for every arm; A0 prunes a TextField in 160 trials, K1 in 10. R3 holds (160 >=
112), R4 holds (0 <= 96); Q1 holds for both (150 and 160 >= 0 + 24); no case is protected for Q2; U1, U2 and every P
hold; both pass and K2's count is higher: **"K2 selected"**, winner K2.

Regime: CPU, mock engine on loopback, real ``PromptBuilder``, ``VllmClient`` (id mode, the pinned detokeniser) and
``UiGate``; no measurement -- the numbers describe the mock, never the model.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
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
from tests.integration.test_t0098c_harness_smoke import MEMORY_MARK, Q12_SHA, _rebind
from tests.integration.test_t0103_harness_smoke import REPEAT_IN_ID, Engine103, serve_103
from tests.unit._pre_t0155_persona import pre_t0155_fixture
from tests.unit.test_t0098_prompt_ab import A0_SHA256
from tests.unit.test_t0098c_ui_prompt_ab import (
    Q12_FILE,
    Q12_NEGATIVE,
    Q12_POSITIVE,
    harness_c,
    patch_u,
    pieces,
)
from tests.unit.test_t0108a_ui_fields_ab import (
    ANCHOR_EDITS,
    ARMS,
    K1_SHA256,
    K2_SHA256,
    N_REQUESTS,
    PLACEHOLDER,
    RULE,
    TABLE,
    expected_repeats,
    expected_schedule,
    harness_108a,
    old_harness,
    texts_by_hand,
    u_k1,
    u_k2,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: T-0155 criterion 8: every row here tests the frozen harness with the persona it had at its merge commit.
pre_t0155 = pytest.fixture(autouse=True, name="pre_t0155")(pre_t0155_fixture)

K1_PRUNED = ("primes_table", range(5, 15))
GATE_FILES = (
    "local_astra/conductor/ui_gate.py",
    "local_astra/ui/bracket_repair.py",
    "local_astra/ui/a2ui_schema.py",
)
REPO = Path(__file__).resolve().parents[2]


class Engine108a(Engine103):
    """T-0103's mock engine taught the T-0108a arms, with this module's script."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_of.update({text: arm for arm, text in texts_by_hand().items()})

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
            ids = [*self.encode(thought + "I will do that.<channel|>" + call), STOP_TOOL]
        elif row is not None and row["kind"] == "q12" and row["expect_surface"]:
            canvas = UI_EXAMPLE
            if arm == "A0" or (
                arm == "K1" and row["id"] == K1_PRUNED[0] and repeat in K1_PRUNED[1]
            ):
                canvas = PLACEHOLDER
            ids = [*self.encode(thought + "Here it is." + UI_OPEN + canvas + UI_CLOSE), STOP_TURN]
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
def serve_108a() -> Iterator[tuple[Engine108a, str]]:
    if not pinned_tokenizer_path().is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {pinned_tokenizer_path()}")
    eng = Engine108a()
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    thread = threading.Thread(target=server.serve_forever, name="t0108a-mock-engine", daemon=True)
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


def _mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for v in value.values():
            yield from _mappings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _mappings(v)


def _has_token_counts(manifest: Any, want: Mapping[str, int]) -> bool:
    """A mapping keyed by the three arms whose values are the counts, or whose per-arm mappings hold the count under a
    key containing ``token``."""
    for m in _mappings(manifest):
        if not all(a in m for a in want):
            continue
        ok = True
        for a, n in want.items():
            v = m[a]
            if v == n:
                continue
            if isinstance(v, Mapping) and any(
                x == n for k, x in v.items() if "token" in str(k).lower()
            ):
                continue
            ok = False
        if ok:
            return True
    return False


def _has_gate_shas(manifest: Any) -> bool:
    want = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in GATE_FILES}
    for m in _mappings(manifest):
        got = {p: v for k, v in m.items() for p in GATE_FILES if str(k).endswith(p)}
        if got == want:
            return True
    return False


# =====================================================================================================
# The healthy T-0108a run
# =====================================================================================================


def test_main_t0108a_runs_1320_in_schedule_order(tmp_path: Path) -> None:
    """Criteria 1, 4, 5 and 7 on one complete ``--table T-0108a`` run: exit 0; 1320 requests, pass 1, tools on, no
    seed, no test-split case, in exactly the schedule of *Design*; each (case, repeat) triple differs only in
    ``messages[0]``; ``trials.jsonl`` mirrors the requests with ids ``t0108a-r{r}-{case}-{arm}``; the manifest records
    table, rule, the three texts and shas, the repeats map, U' token counts (as the mock's ``/tokenize`` counts them),
    the three gate-file shas of this tree, the thresholds as k/n strings and a ``mod 3`` rotation; ``report.json`` is
    ``build_report(run_dir)`` and reads the module docstring's numbers and outcome (UI_EXAMPLE canvases complete, the
    label-less TextField canvases not)."""
    h = harness_108a()
    require_clean_tree()
    out = tmp_path / "t0108a-ui-fields-ab"
    with serve_108a() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", TABLE]
        )
    assert attempts == [], f"a socket left loopback: {attempts}"
    assert eng.unknown == [], f"requests the mock could not place: {eng.unknown[:5]}"
    assert rc == 0, f"a complete run exits 0; got {rc}"

    chats = eng.chats
    want = expected_schedule()
    assert len(chats) == N_REQUESTS, f"{len(chats)} chat requests"
    assert [(c["case_id"], c["repeat"], c["arm"]) for c in chats] == [
        (cid, r, arm) for _, cid, r, _, arm in want
    ], "the requests are not in the schedule of *Design*"
    for c in chats:
        b = c["body"]
        assert b.get("max_tokens") != 1 and not b.get("continue_final_message"), "pass 1 only"
        assert not _has_key(b, "seed"), "a seed field was sent"
        assert b.get("tool_choice") == "none" and b.get("tools"), "all tools on"
        assert not any(t in c["user"] for t in eng.test_texts), "a test-split case was sent"
        if c["case_id"] in (*Q12_POSITIVE, *Q12_NEGATIVE):
            assert pb.NO_IMAGE_STATEMENT in c["user"], c["case_id"]
    for k in range(0, len(chats), 3):
        block = chats[k : k + 3]
        assert len({(c["case_id"], c["repeat"]) for c in block}) == 1
        assert {c["arm"] for c in block} == set(ARMS)
        assert (
            _outside_system(block[0]["body"])
            == _outside_system(block[1]["body"])
            == _outside_system(block[2]["body"])
        ), f"{block[0]['case_id']} r{block[0]['repeat']} differs outside messages[0]"
    shas = {"A0": A0_SHA256, "K1": K1_SHA256, "K2": K2_SHA256}
    for arm in ARMS:
        assert {_sha(c["system"]) for c in chats if c["arm"] == arm} == {shas[arm]}, arm

    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == N_REQUESTS
    for k, (t, c) in enumerate(zip(trials, chats, strict=True)):
        _, cid, r, slot, arm = want[k]
        assert (t["case_id"], t["arm"], t["repeat"], t["order"]) == (cid, arm, r, slot), k
        assert t["request_id"] == f"t0108a-r{r}-{cid}-{arm}"
        assert t["token_ids"] == c["ids"]
        if cid in Q12_POSITIVE:
            placeholder = arm == "A0" or (arm == "K1" and cid == K1_PRUNED[0] and r in K1_PRUNED[1])
            want_ui = PLACEHOLDER if placeholder else UI_EXAMPLE
            assert [m["text"] for m in t["commits"] if m["segment"] == "ui"] == [want_ui], k

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest.get("table") == TABLE and manifest.get("rule") == RULE
    texts = texts_by_hand()
    got = {a: (v["text"], v["sha256"]) for a, v in manifest["arms"].items()}
    assert got == {a: (texts[a], shas[a]) for a in ARMS}
    assert _find(manifest, expected_repeats()), "criterion 1: the manifest lacks the repeats map"

    u_primes = {"A0": pieces()[3], "K1": u_k1(), "K2": u_k2()}
    counts = {a: len(eng.encode(u)) for a, u in u_primes.items()}
    assert ("POST", "/tokenize") in eng.requests, (
        "the U' token counts were not taken from /tokenize"
    )
    assert _has_token_counts(manifest, counts), (
        f"criterion 1: the manifest lacks the U' token counts {counts}"
    )
    assert _has_gate_shas(manifest), "criterion 1: the manifest lacks this tree's gate-file shas"
    flat = json.dumps(manifest)
    for kn in (
        "24/160",
        "24/40",
        "16/160",
        "1/80",
        "2/200",
        "8/160",
        "10/175",
        "112/160",
        "96/160",
    ):
        assert f'"{kn}"' in flat, f"criterion 1: the manifest lacks the threshold {kn}"
    assert "mod 3" in flat and not any(m in flat for m in ("mod 2", "mod 6", "mod 7")), "rotation"

    stored = json.loads((out / "report.json").read_text())
    assert json.loads(json.dumps(h.build_report(out))) == stored, (
        "report.json is build_report(run_dir)"
    )
    assert (out / "report.md").is_file()
    assert stored["complete"] is True and list(stored["void_arms"]) == []
    assert stored["readings"] == {"R3": True, "R4": True}
    arms = stored["arms"]
    assert arms["A0"]["q12_positive_complete"] == {"k": 0, "n": 160}
    assert arms["K1"]["q12_positive_complete"] == {"k": 150, "n": 160}
    assert arms["K2"]["q12_positive_complete"] == {"k": 160, "n": 160}
    assert arms["K1"]["q12_per_case_complete"]["primes_table"] == {"k": 30, "n": 40}
    for arm in ARMS:
        assert arms[arm]["q12_positive_last_ok"] == {"k": 160, "n": 160}, arm
        assert arms[arm]["q12_positive_shown"] == {"k": 160, "n": 160}, arm
        assert arms[arm]["q12_negative_shown"] == {"k": 0, "n": 15}, arm
    assert _find(arms["A0"], {"TextField": 160}), "A0's pruned kinds"
    assert _find(arms["K1"], {"TextField": 10}), "K1's pruned kinds"
    for k in ("K1", "K2"):
        assert [
            c for c, v in stored["candidates"][k].items() if v is not True and c[:1] in "QUP"
        ] == [], k
    assert str(stored["outcome"]).startswith("K2 selected") and stored["winner"] == "K2"
    assert str(stored["outcome"]) in (out / "report.md").read_text()


# =====================================================================================================
# Criterion 2: the earlier tables on the same mock (guard; green today)
# =====================================================================================================


@pytest.mark.parametrize(
    ("table", "n", "prefix", "mod"),
    [
        ("T-0098c", 2100, "t0098c", "mod 7"),
        ("T-0098", 1590, "t0098", "mod 6"),
        ("T-0103", 880, "t0103", "mod 2"),
    ],
)
def test_earlier_tables_on_the_same_mock(
    tmp_path: Path, table: str, n: int, prefix: str, mod: str
) -> None:
    """Criterion 2: on T-0103's mock, ``--table T-0098c`` still writes 2100 lines, ``T-0098`` 1590 and ``T-0103`` 880,
    with their own request-id prefix and rotation string, and no T-0108a report key (``q12_positive_complete``); the
    T-0103 run is still "C3 confirmed"."""
    h = harness_c()
    require_clean_tree()
    out = tmp_path / f"{prefix}-out"
    with serve_103() as (eng, base_url), _loopback_only() as attempts:
        rc = h.main(
            ["--base-url", base_url, "--out-dir", str(out), "--split", "dev", "--table", table]
        )
    assert attempts == [] and eng.unknown == [] and rc == 0, (attempts, eng.unknown[:3], rc)
    assert len(eng.chats) == n
    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    assert len(trials) == n
    assert all(str(t["request_id"]).startswith(f"{prefix}-r") for t in trials)
    manifest = json.loads((out / "manifest.json").read_text())
    assert mod in json.dumps(manifest) and "mod 3" not in json.dumps(manifest)
    stored = json.loads((out / "report.json").read_text())
    assert "q12_positive_complete" not in json.dumps(stored)
    if table == "T-0103":
        assert str(stored["outcome"]).startswith("C3 confirmed") and stored["winner"] == "C3"
    else:
        assert "outcome" not in stored


def _paths(
    value: Any, words: tuple[str, ...], at: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    """Every key path in a manifest whose last key contains one of ``words`` (found in the written JSON, not in the
    harness)."""
    out: dict[tuple[str, ...], Any] = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            path = (*at, str(k))
            if any(w in str(k).lower() for w in words):
                out[path] = v
            out.update(_paths(v, words, path))
    return out


def test_t0103_manifest_reading_and_thresholds_as_at_ca61f08(tmp_path: Path) -> None:
    """Criterion 2 (reviewer (a)): on the same mock, ``--table T-0103`` under this branch's harness and under the
    harness as at ca61f08 writes the same manifest ``reading`` string(s) and the same thresholds: every key path of the
    ca61f08 manifest whose key names a reading or a threshold holds the same value now. Premise: the ca61f08 manifest
    has at least one of each (so the row cannot pass on an empty set)."""
    require_clean_tree()
    manifests = {}
    for name, h in (("now", harness_c()), ("ca61f08", old_harness())):
        out = tmp_path / f"t0103-{name}"
        with serve_103() as (eng, base_url), _loopback_only() as attempts:
            rc = h.main(
                [
                    "--base-url",
                    base_url,
                    "--out-dir",
                    str(out),
                    "--split",
                    "dev",
                    "--table",
                    "T-0103",
                ]
            )
        assert attempts == [] and eng.unknown == [] and rc == 0, (
            name,
            attempts,
            eng.unknown[:3],
            rc,
        )
        manifests[name] = json.loads((out / "manifest.json").read_text())
    old = _paths(manifests["ca61f08"], ("reading", "threshold"))
    assert any("reading" in p[-1].lower() for p in old), (
        f"premise: no reading key at ca61f08: {sorted(old)}"
    )
    assert any("threshold" in p[-1].lower() for p in old), (
        f"premise: no threshold key at ca61f08: {sorted(old)}"
    )
    now = _paths(manifests["now"], ("reading", "threshold"))
    for path, value in old.items():
        assert now.get(path) == value, (
            f"criterion 2: T-0103 manifest {'/'.join(path)} moved: {now.get(path)!r}"
        )


# =====================================================================================================
# Criterion 3: the T-0108a refusals, all before any socket
# =====================================================================================================

K_REFUSALS: dict[str, tuple[str, ...]] = {
    **{f"t0098-{k}": (v,) for k, v in REFUSALS.items()},
    "q12-sha": ("surface_placement_cases",),
    "q12-not-4-plus-3": ("surface_placement_cases", "Q12"),
    "k1-not-the-edit": ("K1", "22a2ba7e"),
    "anchor-e1-twice": ("anchor", "K1", "K2"),
    "repeats-39-on-one-positive": ("repeat",),
}


@pytest.mark.parametrize("why", list(K_REFUSALS))
def test_main_t0108a_refuses_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], why: str
) -> None:
    """Criterion 3: under ``--table T-0108a`` every T-0103 refusal still holds -- loopback, dirty tree, non-empty
    out-dir, ``--split test``, the ``cases.jsonl`` pin, the tool kit (unloadable, repair off, wrong set), A0's sha
    (one sentence added to ``SPOKEN_STYLE``), the Q12 pin and its 4 + 3 shape, the repeats map (39 on
    ``primes_table``) -- and so do T-0108a's: K1 replaced by A0's text, and E1's anchor twice in U. Each: exit 2, the
    engine never contacted, no socket at all, nothing written, the refusal names its cause. Over a real clean tree."""
    h = harness_108a()
    require_clean_tree()
    out = tmp_path / "t0108a-out"
    probe = Path(h.REPO) / "tests" / "integration" / "_t0108a_dirty_probe.txt"
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
            fake_sha = hashlib.sha256(fake.read_bytes()).hexdigest()
            assert _rebind(monkeypatch, [h, h._T93], Q12_FILE, fake) >= 1, "premise: the Q12 path"
            assert _rebind(monkeypatch, [h, h._T93], Q12_SHA, fake_sha) >= 1, "premise: the Q12 pin"
        elif why == "k1-not-the-edit":
            t = h.TABLES[TABLE]
            orig = t.texts

            def texts(tools: Any) -> dict[str, str]:
                got = dict(orig(tools))
                got["K1"] = got["A0"]
                return got

            monkeypatch.setitem(h.TABLES, TABLE, dataclasses.replace(t, texts=texts))
        elif why == "anchor-e1-twice":
            patch_u(monkeypatch, h, ANCHOR_EDITS["e1-twice"][0])
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
        tok.replace("_t0098_dirty_probe", "_t0108a_dirty_probe").replace("t0098-out", "t0108a-out")
        for tok in K_REFUSALS[why]
    )
    if base == "a0-sha-drift":
        tokens = (*tokens, "22a2ba7e", "23a34c36")
    if base == "wrong-tool-set":
        tokens = (*tokens, "000000000000")
    text = said.out + said.err
    assert any(tok in text for tok in tokens), f"criterion 3 ({why}): {tokens}: {text[-300:]!r}"
