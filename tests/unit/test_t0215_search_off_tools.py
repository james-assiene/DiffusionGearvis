"""T-0215 (tools side): a live session with no search backend declares the live tool set without ``google_search``
(docs/tasks/T-0215.md criteria 1-3, 6, 7).

Every row builds through the real ``cli`` composition root over the CLI fakes (``test_cli_search.real``: the spy
executor and fake start pages of ``test_cli_tools_live.fake_live``, the T-0088a subset detokeniser, nothing served),
so the kit is the one a real ``astra session --tools live`` start would hand its ``Session``. The three ways a live
session has no search backend are the ``--search off`` flag, a profile's ``search = "off"`` and the default-on
search degrading (``cli._search_backend`` raising ``ImportError``, which ``search_start`` reports as the extra
missing).

Oracles (none computed by the code under test):

- the on-sets are the four shipped files ``tools_v1{,_overview}{,_wait}.json``, read with ``tools.load`` and pinned
  by their sha256 literals (``ON_SHAS``, from ``sha256sum`` at aeaf81a5);
- "minus ``google_search``" is the on-set's declarations with that one name filtered out, compared declaration by
  declaration as JSON text with keys in file order (so a reordered or edited declaration fails);
- the version oracle: when the kit's set comes from a file (the R-p route; T-0215's four ``*_nosearch.json``
  files), its version is that file's sha256 and the file parses to the kit's declarations. A set derived in
  ``tools.py`` (the other route the task allows) is checked only for being a sha256 hex distinct from every on-set
  and from the other three no-search sets (reported to the caller as the one place the criterion is not fully
  testable without the implementer's canonical serialisation).

The adversarial row (criterion 3, the T-0208 C failure mode: every shared step 1 was ``google_search``, answered
"not connected") runs a live ``Session`` (``feedback_rig``) whose kit carries the cli-built no-search tools, sha and
judge set, a fake LLM committing ``call:google_search{query:<|"|>weather<|"|>}<tool_call|>``, and asserts the call
never executes. It runs twice: with no backend on the kit (the cli's own search-off kit) and with a recording
``FakeSearch`` attached, so a call that slipped past the judge would show up as a query.

Rows that failed at aeaf81a5 carried ``pending(task="T-0215")`` until activation (T-0215 implement); the negative rows (criteria 6, 7) are active and pass
at aeaf81a5.

Regime: CPU over fakes, virtual clock in the session row, no browser, no network, n = 1, deterministic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.toolcall import live, loop, native, tools
from tests.fakes.search import FakeSearch
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame, live_kit
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_cli_search import PROXY_ENV, real
from tests.unit.test_session_live import card
from tests.unit.test_t0140_session import kit_set_field

LIVE = ("--tools", "live", "--browser", "headless")
TOOLCALL = Path(tools.__file__).resolve().parent
Q = '<|"|>'
SEARCH_CALL = f"<|tool_call>call:google_search{{query:{Q}weather{Q}}}<tool_call|>"

#: (page_overview, wait_tool) -> the shipped on-set file and its sha256 (sha256sum at aeaf81a5)
ON_FILES = {
    (False, False): "tools_v1.json",
    (True, False): "tools_v1_overview.json",
    (False, True): "tools_v1_wait.json",
    (True, True): "tools_v1_overview_wait.json",
}
ON_SHAS = {
    (False, False): "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707",
    (True, False): "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc",
    (False, True): "05de281e7bd2a07d1bd1d79248530d158b48402c4bd30e333d0d428590bf754d",
    (True, True): "72c3d84fda0180fb65be0fb7a121ecef5dc299bc141df47fddbd6176b72128ac",
}
COMBOS = list(ON_FILES)
COMBO_IDS = ["v1", "overview", "wait", "overview+wait"]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")


def combo_flags(overview: bool, wait: bool) -> tuple[str, ...]:
    return (*(("--page-overview",) if overview else ()), "--wait-tool", "on" if wait else "off")


def on_set(overview: bool, wait: bool) -> tools.ToolSet:
    s = tools.load(TOOLCALL / ON_FILES[(overview, wait)])
    assert s.version == ON_SHAS[(overview, wait)], "premise: the shipped on-set is aeaf81a5's"
    assert s.names[0] == "google_search", "premise: every on-set declares google_search first"
    return s


def kit_set(kit: Any) -> Any:
    return getattr(kit, kit_set_field())


def decl_text(d: Any) -> str:
    """A declaration as JSON text in its own key order: an edited, re-keyed or reordered declaration differs."""
    return json.dumps(d, ensure_ascii=False)


def build_kit(tmp_path: Path, capsys: Any, *flags: str, body: str = "") -> tuple[Any, str, str]:
    """A real live start through ``cli`` (nothing served) in its own MonkeyPatch scope; the kit, stdout, stderr."""
    with pytest.MonkeyPatch.context() as mp:
        code, out, err, calls, _ = real(tmp_path, capsys, mp, *flags, body=body)
        assert code == 0, f"{flags} {body!r}: exit {code}\n{out}{err}"
        kit = calls.kit()
    assert kit is not None and kit.mode == "live", f"premise: a live kit; {kit}"
    return kit, out, err


def assert_no_search_set(kit: Any, overview: bool, wait: bool, how: str) -> None:
    base = on_set(overview, wait)
    want_decls = [d for d in base.declarations if d["name"] != "google_search"]
    want_names = tuple(d["name"] for d in want_decls)
    js = kit_set(kit)
    assert js is not None, (
        f"T-0215 criterion 2 [{how}]: the kit carries the judge's (no-search) set; None is v1"
    )
    assert js.names == want_names, (
        f"T-0215 criterion 1 [{how}]: names are the on-set's minus google_search in v1's order; got {js.names}, "
        f"want {want_names}"
    )
    assert [decl_text(d) for d in js.declarations] == [decl_text(d) for d in want_decls], (
        f"T-0215 criterion 1 [{how}]: each remaining declaration is the on-set's, unchanged"
    )
    assert kit.tools == tuple(native.tools_field(js)), (
        f"T-0215 criterion 2 [{how}]: kit.tools is built from the kit's own set"
    )
    names = [t["function"]["name"] for t in kit.tools]
    assert "google_search" not in names and tuple(names) == want_names, (
        f"T-0215 criterion 1 [{how}]: the declared tools field; {names}"
    )
    assert kit.tools_version == js.version, (
        f"T-0215 criterion 2 [{how}]: tools_version {kit.tools_version} is the set's {js.version}"
    )
    assert kit.tools_version not in ON_SHAS.values(), (
        f"T-0215 criterion 2 [{how}]: tools_version is an on-set's sha ({kit.tools_version})"
    )


# =====================================================================================================
# Criterion 1: the no-search set, all four combinations
# =====================================================================================================


@pytest.mark.parametrize("overview,wait", COMBOS, ids=COMBO_IDS)
def test_live_search_off_kit_has_no_google_search(
    tmp_path, capsys, overview: bool, wait: bool
) -> None:
    """Criterion 1: ``--search off`` in live gives the set it would otherwise have, minus ``google_search``; the
    other declarations in v1's order, each unchanged."""
    kit, _out, _err = build_kit(
        tmp_path, capsys, *LIVE, "--search", "off", *combo_flags(overview, wait)
    )
    assert kit.search is None, "premise: search is off"
    assert_no_search_set(kit, overview, wait, "flag")


def test_profile_search_off_also_withdraws(tmp_path, capsys) -> None:
    """Criterion 1: a profile's ``search = "off"`` (no flag) is the same as the flag."""
    kit, out, _err = build_kit(tmp_path, capsys, *LIVE, body='search = "off"\n')
    assert "astra session: search: off (profile)" in out.splitlines(), "premise: off by the profile"
    assert_no_search_set(kit, False, True, "profile")


@pytest.mark.parametrize("overview", [False, True], ids=["wait", "overview+wait"])
def test_degrade_path_also_withdraws(tmp_path, capsys, monkeypatch, overview: bool) -> None:
    """Criterion 1: a default-on search whose ``_search_backend`` raises ``ImportError`` starts with search off and
    the no-search kit (the degrade path of ``session_cmd`` / ``build_session``)."""

    def missing(name: str) -> Any:
        raise ImportError("No module named 'ddgs' (T-0215 rig)")

    monkeypatch.setattr(cli, "_search_backend", missing)
    flags = ("--page-overview",) if overview else ()
    kit, out, _err = build_kit(tmp_path, capsys, *LIVE, *flags)
    assert kit.search is None and getattr(kit, "web_search", "missing") is None, "premise: degraded"
    assert [x for x in out.splitlines() if x.startswith("astra session: search: off (default")], (
        f"premise: the degrade's search line; {out}"
    )
    assert_no_search_set(kit, overview, True, "degrade")


# =====================================================================================================
# Criterion 2: the version is the declared set's sha; the judge never accepts google_search
# =====================================================================================================


def test_search_off_version_is_the_declared_sets_sha(tmp_path, capsys) -> None:
    """Criterion 2: ``kit.tools_version == kit.tool_set.version``, it differs from the on-set's, and it is the
    sha256 of the bytes the set was built from (R-p: the file's; four distinct shas, one per combination)."""
    shas: dict[tuple[bool, bool], str] = {}
    for i, (overview, wait) in enumerate(COMBOS):
        kit, _o, _e = build_kit(
            tmp_path / f"c{i}", capsys, *LIVE, "--search", "off", *combo_flags(overview, wait)
        )
        js = kit_set(kit)
        assert js is not None, (
            f"T-0215 criterion 2: the judge's set is on the kit [{overview}, {wait}]"
        )
        assert kit.tools_version == js.version != ON_SHAS[(overview, wait)], (
            f"T-0215 criterion 2 [{overview}, {wait}]: {kit.tools_version} / {js.version}"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", js.version), js.version
        path = Path(js.path)
        if path.is_file() and path.resolve() not in {
            (TOOLCALL / f).resolve() for f in ON_FILES.values()
        }:
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == js.version, (
                f"T-0215 criterion 2: the version is the sha256 of {path.name}'s bytes"
            )
            assert [decl_text(d) for d in json.loads(data)] == [
                decl_text(d) for d in js.declarations
            ], f"T-0215 criterion 2: {path.name} holds exactly the declared set"
        shas[(overview, wait)] = js.version
        # a second build gives the same bytes and the same sha (a canonical source, not a per-run one)
        again, _o, _e = build_kit(
            tmp_path / f"d{i}", capsys, *LIVE, "--search", "off", *combo_flags(overview, wait)
        )
        assert again.tools_version == kit.tools_version and again.tools == kit.tools
    assert len(set(shas.values())) == 4, f"T-0215 criterion 2: one sha per no-search set; {shas}"


@pytest.mark.parametrize("overview,wait", COMBOS, ids=COMBO_IDS)
def test_judge_never_accepts_google_search(tmp_path, capsys, overview: bool, wait: bool) -> None:
    """Criterion 2: ``live.classify(<a well-formed google_search commit>, tool_set=kit.tool_set)`` is never
    ``valid``; a click on the same set still is (power: the judge is not refusing everything)."""
    kit, _o, _e = build_kit(
        tmp_path, capsys, *LIVE, "--search", "off", *combo_flags(overview, wait)
    )
    js = kit_set(kit)
    assert live.classify(SEARCH_CALL, tool_set=on_set(overview, wait)).status == "valid", (
        "premise: the on-set accepts the call"
    )
    got = live.classify(SEARCH_CALL, tool_set=js)
    assert got.status != "valid", (
        f"T-0215 criterion 2: the no-search judge accepted google_search; {got}"
    )
    assert live.classify(VALID_CLICK, tool_set=js).status == "valid", (
        "power: the no-search judge still accepts a click"
    )


#: a google_search commit cut off inside its query (a ``repairable`` draft against any set declaring the name)
TRUNCATED_SEARCH = f"<|tool_call>call:google_search{{query:{Q}wea"


@pytest.mark.parametrize("draft", ["well_formed", "truncated"])
@pytest.mark.parametrize("overview,wait", COMBOS, ids=COMBO_IDS)
def test_judge_refuses_google_search_as_not_declared(
    tmp_path, capsys, overview: bool, wait: bool, draft: str
) -> None:
    """Criterion 3 / R-1 (review F1): against each no-search set the judge itself refuses a ``google_search``
    commit, well-formed or truncated, with ``live.NOT_DECLARED``: never ``repairable`` (which would send a repair
    request and leave the refusal to ``repair_verdict``). Premise: the on-set calls the same texts valid and
    repairable."""
    text = SEARCH_CALL if draft == "well_formed" else TRUNCATED_SEARCH
    kit, _o, _e = build_kit(
        tmp_path, capsys, *LIVE, "--search", "off", *combo_flags(overview, wait)
    )
    want_on = "valid" if draft == "well_formed" else "repairable"
    assert live.classify(text, tool_set=on_set(overview, wait)).status == want_on, (
        f"premise: the on-set calls the {draft} draft {want_on}"
    )
    got = live.classify(text, tool_set=kit_set(kit))
    assert (got.status, got.reason) == ("refused", live.NOT_DECLARED), (
        f"T-0215 R-1 [{draft}]: the no-search judge's verdict {got}"
    )


# =====================================================================================================
# Criterion 3 (adversarial): a google_search commit in a withdrawn session never executes
# =====================================================================================================


def session_kit(browser: Any, cli_kit: Any, search: Any) -> Any:
    """``live_kit(browser)`` carrying the cli-built kit's tools, sha and judge set (and ``search`` when given)."""
    return dataclasses.replace(
        live_kit(browser),
        tools=cli_kit.tools,
        tools_version=cli_kit.tools_version,
        search=search,
        **{kit_set_field(): kit_set(cli_kit)},
    )


@pytest.fixture
def cli_off_kit(tmp_path: Path, capsys: Any) -> Any:
    """The cli-built ``--search off`` live kit, built outside the event loop (``cli`` runs ``asyncio.run``)."""
    kit, _o, _e = build_kit(tmp_path / "cli", capsys, *LIVE, "--search", "off")
    return kit


@pytest.mark.parametrize("backend", ["none", "fake_attached"])
async def test_google_search_call_refused_when_withdrawn(
    tmp_path, cli_off_kit: Any, backend: str
) -> None:
    """Criterion 3: in a session with the cli's ``--search off`` kit, a committed ``google_search`` is never
    executed: zero backend queries, no request body carrying ``SEARCH_NOT_CONNECTED``, no ``tool.result`` for
    it, and its ``tool.call`` records and card end ``refused`` (never ``executed``/``dry_run``)."""
    cli_kit = cli_off_kit
    search = FakeSearch()
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 6)]
    )
    kit = session_kit(browser, cli_kit, search if backend == "fake_attached" else None)
    async with feedback_rig(
        tmp_path / "s", browser, script=[answer(tsl.S1), tool(SEARCH_CALL)], kit=kit
    ) as rig:
        rig.llm.script_steps([answer(tsl.S3)], [answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0215: the turn never ended", limit_ms=40_000)
        await rig.advance(300)
        bodies = list(rig.llm.bodies)
        c = card(rig, "t1-call")
    # T-0215 R-3: the session log is read after the rig exits (it is flushed on close)
    records = [r for r in rig.kind("tool.call") if r.get("call_id") == "t1-call"]
    results = [r for r in rig.kind("tool.result") if r.get("call_id") == "t1-call"]
    assert search.queries == [], (
        f"T-0215 criterion 3 [{backend}]: the search backend was called: {search.queries}"
    )
    leaked = [i for i, b in enumerate(bodies) if loop.SEARCH_NOT_CONNECTED in json.dumps(b)]
    assert leaked == [], (
        f"T-0215 criterion 3 [{backend}]: SEARCH_NOT_CONNECTED reached request(s) {leaked}"
    )
    assert len(records) == 1, (
        f"T-0215 R-1 [{backend}]: exactly one tool.call record (no proposed/repairing ones); {records}"
    )
    assert records[0].get("reason") == live.NOT_DECLARED, (
        f"T-0215 R-1 [{backend}]: refused by the judge as not declared; {records[0]}"
    )
    assert all(r.get("status") != "repairing" and r.get("source") != "repair" for r in records), (
        records
    )
    assert len(bodies) == 1, (
        f"T-0215 R-1 [{backend}]: only the pass-1 request was sent (no repair, no step 2); {len(bodies)} bodies"
    )
    statuses = [r.get("status") for r in records]
    assert "executed" not in statuses and "dry_run" not in statuses and statuses[-1] == "refused", (
        f"T-0215 criterion 3 [{backend}]: the call ends refused like any call outside the set; {statuses}"
    )
    assert results == [], (
        f"T-0215 criterion 3 [{backend}]: no tool.result for the refused call; {results}"
    )
    assert c.status == "refused", f"T-0215 criterion 3 [{backend}]: the card; {c}"


# =====================================================================================================
# Criterion 6 (negative): live with search on is main's, byte for byte
# =====================================================================================================


@pytest.mark.parametrize("overview,wait", COMBOS, ids=COMBO_IDS)
def test_search_on_kit_is_main_s(tmp_path, capsys, monkeypatch, overview: bool, wait: bool) -> None:
    """Criterion 6: with search on (the live default), ``load_tool_kit`` is called with exactly today's keywords
    (``live_browser``, and ``page_overview``/``wait_tool`` only when on), and ``kit.tools``/``tools_version``/the
    judge's set are the on-set's (v1: ``None``)."""
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_load = cli.load_tool_kit

    def spy(*a: Any, **k: Any) -> Any:
        seen.append((a, dict(k)))
        return real_load(*a, **k)

    monkeypatch.setattr(cli, "load_tool_kit", spy)
    kit, out, _err = build_kit(tmp_path, capsys, *LIVE, *combo_flags(overview, wait))
    assert "astra session: search: ddgs (default)" in out.splitlines(), "premise: search on"
    assert len(seen) == 1, f"one load_tool_kit call; {seen}"
    args, kwargs = seen[0]
    assert len(args) == 1, f"T-0215 criterion 6: positional arguments are (profile,) only; {args}"
    want_keys = (
        {"live_browser"}
        | ({"page_overview"} if overview else set())
        | ({"wait_tool"} if wait else set())
    )
    assert set(kwargs) == want_keys, f"T-0215 criterion 6: load_tool_kit keywords {sorted(kwargs)}"
    assert all(kwargs[k] is True for k in want_keys - {"live_browser"}), kwargs
    s = on_set(overview, wait)
    assert (
        kit.tools == tuple(native.tools_field(s)) and kit.tools_version == ON_SHAS[(overview, wait)]
    ), f"T-0215 criterion 6: the search-on kit is main's ({kit.tools_version})"
    js = kit_set(kit)
    if (overview, wait) == (False, False):
        assert js is None, f"T-0215 criterion 6: v1's kit has no judge set (main's); {js!r}"
    else:
        assert js is not None and js.version == ON_SHAS[(overview, wait)], js


# =====================================================================================================
# Criterion 7 (negative): dry-run is unchanged, with search off or on
# =====================================================================================================


@pytest.mark.parametrize("search", ["unset", "off", "ddgs"])
def test_dry_run_unchanged_with_search_off(tmp_path, capsys, monkeypatch, search: str) -> None:
    """Criterion 7: ``--tools dry-run`` (search unset, ``--search off`` or ``--search ddgs``) loads ``load_tool_kit``
    with the profile alone and gets v1: its tools, its sha (bbfc53cc...), no judge set, repair on with v1's
    payloads."""
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_load = cli.load_tool_kit

    def spy(*a: Any, **k: Any) -> Any:
        seen.append((a, dict(k)))
        return real_load(*a, **k)

    monkeypatch.setattr(cli, "load_tool_kit", spy)
    flags = ("--tools", "dry-run", *(() if search == "unset" else ("--search", search)))
    with pytest.MonkeyPatch.context() as mp:
        code, out, err, calls, _ = real(tmp_path, capsys, mp, *flags)
        assert code == 0, f"{flags}: exit {code}\n{out}{err}"
        kit = calls.kit()
    assert [(len(a), k) for a, k in seen] == [(1, {})], (
        f"T-0215 criterion 7: load_tool_kit(profile); {seen}"
    )
    assert (
        kit.mode == "dry_run"
        and kit.tools == T.tools_field()
        and kit.tools_version == ON_SHAS[(False, False)]
    )
    assert kit_set(kit) is None, "T-0215 criterion 7: dry-run's kit is v1's (no judge set)"
    assert kit.repair_off is None and kit.payloads is not None, (
        "T-0215 criterion 7: repair on, v1 payloads"
    )
    assert kit.payloads == live.load_payloads(), "T-0215 criterion 7: payloads stay keyed to v1"


def test_search_off_live_keeps_v1_payloads(tmp_path, capsys) -> None:
    """Criterion 7: the no-search live kit's repair still reads v1's payloads and calibration (``load_payloads`` and
    ``load_calibration`` unchanged), as the search-on wait kit does today. Pending because the premise (a no-search
    set) holds only after T-0215."""
    kit, _o, _e = build_kit(tmp_path, capsys, *LIVE, "--search", "off")
    assert "google_search" not in kit_set(kit).names, "premise: the no-search kit"
    assert kit.repair_off is None, f"T-0215 criterion 7: repair stays on; {kit.repair_off}"
    assert kit.payloads == live.load_payloads() and kit.calibration == live.load_calibration()
