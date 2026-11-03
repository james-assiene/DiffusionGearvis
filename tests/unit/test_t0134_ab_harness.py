"""T-0134 group B: the 3-arm live-feedback A/B harness's pure parts and its locked script (unit; CPU; no engine, no
browser). docs/tasks/T-0134.md B1-B8.

Group B, activated 2026-10-22 (until then pending on the ``harness`` parameter only). Rows that need the harness are red on the named
assertion in ``harness()``/``api()`` (the file or a name is absent), never on an import error. Rows run once against
the harness and once against the eval oracle (``tests/eval/test_t0134_ab.py``; **not** pending, green now): a wrong
oracle is caught on CPU.

**The assumed harness surface** (``scripts/gpu/t0134_feedback_ab.py``; the task names the behaviour, these names and
shapes are the test-author's, stated once so the implementer can implement to them or say why not):

Imported, never copied (B1)
    T-0111g's ``TASKS``, ``score``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
    ``non_loopback_sockets``, ``turn_records``; T-0131's ``INTENDED_TARGETS``, ``BOXES_JS``, ``click_score``,
    ``false_claim``, ``probe_p1``, ``run_probes``, ``config_evidence``; T-0132's ``wilson`` and ``newcombe`` (both
    exported at 2ccff41). Module attributes whose objects come from those files (a ``def``/``class``/literal of those
    names is refused). ``main`` reads the tree through a module-level ``_git`` (T-0111g's seam).

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``ARMS`` = the B1 table as 5-tuples ``(label,
    visual_tokens, reground, live_feedback, echo)``, arm A ``("A", 1120, False, None, "pixels")`` (reading of B1's
    4-tuple: arm A uses echo "pixels");``arm_table_sha(arms=None)`` = sha256 of ``json.dumps([{"label",
    "visual_tokens", "reground", "live_feedback": None | {"screen_label", "page_extent"}, "echo"} ...],
    sort_keys=True, separators=(",", ":"))``; ``ARM_TABLE_SHA256`` its value (``7a87f427…``); ``REPEATS = 20``;
    ``ORDER_SEED = 0``; ``RUN_BUDGET_S = 78 * 60``; ``S4_PRICE_FORMS`` (B3's list, at least).

Pure functions (each also in the eval oracle, same name and shape)
    ``order(seed) -> list[(arm, task, repeat)]`` (300, shuffled jointly); ``score_v2(task, state, voiced) -> bool``;
    ``over_click(task, state) -> bool``; ``no_see(thought) -> bool``; ``coord_mismatch(thought) -> bool``;
    ``silent_call_in_thought(commits) -> bool`` over one request's ``llm.commit`` data (``segment``, ``text``,
    ``thought_open``), T-0135 A1/A2's structural rule (O-10), counted per turn (a turn counts once if any request
    fires);
    ``configuration_findings(trials, steps, regrounds) -> list[str]`` over the record fields named in the eval's
    docstring (``arm``, ``budgets_sent``, ``guidance_sent``, step ``has_image``/``label``/``label_count``/
    ``extent_known``/``executed_arguments``/``action``); ``p5_ok(p5) -> bool`` over ``{"scroll0", "scrolled":
    {"above_px", "below_px", "viewport_px"}, "hostile_true"}``; ``decide(inputs, *, p1_passed, void=False,
    stopped=None, configuration_findings=())`` and ``decide_with_reason(...) -> (label, reason)``, ``inputs`` keyed
    ``"A"``/``"B"``/``"C"``, each ``{"accepted", "safety", "engine_up", "success", "s4", "false_claims",
    "false_claims_non_s4", "over_clicks", "first_failing"}``. Labels: ``"feedback + extent"``, ``"feedback only"``,
    ``"no change"``, ``"inconclusive"``, ``"void"``, ``"incomplete"``; the C-vs-A fallback's reason is
    ``"via C vs A"``.

Readings (reported): B3 does not say whether ``4.75`` must stand alone; these rows read it as T-0111g's S4 rule did
(``$14.75`` does not name ``4.75``). The D-4 regex accepts a mismatch phrase within 200 characters of the quoted
``clicked at (n, n)``.

Regime: CPU, pure, hand-built records and the T-0132 fixture (``tests/fixtures/toolcall/t0134_t0132_answers.json``);
no engine, no browser, no measurement.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import itertools
import json
import socket
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_t0134_ab as oracle
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0134_feedback_ab.py"
T0111G_HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
T0131_HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0132_HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0134_feedback_ab.sh"
FIXTURE = REPO / "tests" / "fixtures" / "toolcall" / "t0134_t0132_answers.json"

IMPLS = [
    pytest.param("harness"),
    pytest.param("eval-oracle"),
]

ARM_TABLE_SHA256 = "7a87f4278b5811e7996e9d554f3fe3c191e049a5471cc50609732dfeb532b352"
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
FROM = {
    T0111G_HARNESS: (
        "score",
        "LoopMeter",
        "sentinel_hits",
        "url_leaks",
        "log_complete",
        "non_loopback_sockets",
        "turn_records",
    ),
    T0131_HARNESS: ("click_score", "false_claim", "probe_p1", "run_probes", "config_evidence"),
    T0132_HARNESS: ("wilson", "newcombe"),
}
DATA = ("TASKS", "INTENDED_TARGETS", "BOXES_JS")


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0134 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0134_feedback_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0134: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    return api(name) if which == "harness" else getattr(oracle, name)


# =====================================================================================================
# B1: imports, the arm table, the order
# =====================================================================================================


def test_imports_never_copied() -> None:
    """B1: importing the harness opens no socket; the T-0111g, T-0131 and T-0132 names are those files' objects,
    and none is defined in the T-0134 harness."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    _, defined = t31._top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    names = {n for ns in FROM.values() for n in ns} | set(DATA)
    assert sorted(defined & names) == [], (
        f"T-0134 B1: defined, not imported: {sorted(defined & names)}"
    )
    for origin, ns in FROM.items():
        for name in ns:
            src = Path(inspect.getsourcefile(api(name)) or "").resolve()
            assert src == origin.resolve(), f"T-0134 B1: {name} comes from {src}, not {origin.name}"


def _canon(arms: Any) -> list[dict[str, Any]]:
    out = []
    for label, vt, rg, lf, echo in arms:
        feedback = (
            None if lf is None else {"screen_label": lf.screen_label, "page_extent": lf.page_extent}
        )
        out.append(
            {
                "label": label,
                "visual_tokens": vt,
                "reground": rg,
                "live_feedback": feedback,
                "echo": echo,
            }
        )
    return out


def test_arm_table_and_sha() -> None:
    """B1: ``ARMS`` is the task's table (arm A pixels and no feedback; B echo + label; C echo + label + extent; 1120
    and ReGround off everywhere), pinned by ``arm_table_sha`` (a reordered table has another sha)."""
    arms = list(api("ARMS"))
    assert _canon(arms) == list(oracle.ARM_TABLE), _canon(arms)
    assert api("arm_table_sha")() == api("ARM_TABLE_SHA256") == ARM_TABLE_SHA256
    assert api("arm_table_sha")((arms[0], arms[2], arms[1])) != ARM_TABLE_SHA256
    assert api("REPEATS") == 20 and api("ORDER_SEED") == 0 and api("RUN_BUDGET_S") == 78 * 60
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )


def test_oracle_arm_table_pin() -> None:
    """Not pending: the eval's arm table hashes to the pin computed here from the canonical JSON."""
    assert oracle.arm_table_sha() == ARM_TABLE_SHA256


@pytest.mark.parametrize("which", IMPLS)
def test_order_300_joint_seed0(which: str) -> None:
    """B1 (reviewer mutation 8, shuffling per arm): 300 turns, each arm 100 times, each (arm, task) 20 times with
    repeats 0-19; deterministic; seed-dependent; the arms interleave (a per-arm shuffle runs them in blocks)."""
    order = impl(which, "order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)] and turns != [tuple(t) for t in order(1)]
    assert len(turns) == 300
    assert Counter(t[0] for t in turns) == {"A": 100, "B": 100, "C": 100}
    per = Counter((t[0], t[1]) for t in turns)
    assert set(per) == {(a, k) for a in "ABC" for k in TASK_IDS} and set(per.values()) == {20}
    assert {t[2] for t in turns} == set(range(20))
    switches = sum(1 for a, b in itertools.pairwise(turns) if a[0] != b[0])
    assert switches > 100, f"the arms run in blocks ({switches} switches)"
    assert {t[0] for t in turns[:30]} == {"A", "B", "C"}


def test_harness_order_equals_oracle() -> None:
    """B1/B8: the harness's seed-0 order is the oracle's (the eval checks the run against it)."""
    assert [tuple(t) for t in api("order")(0)] == oracle.order(0)


# =====================================================================================================
# B1: refusals before any socket; the locked script
# =====================================================================================================


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """Every socket refused and recorded; a clean tree through the harness's ``_git``."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0134 refusal rig: no {what}")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else t31.GIT + "\n"

    assert callable(getattr(h, "_git", None)), (
        "the harness has no module-level _git (T-0111g's seam)"
    )
    monkeypatch.setattr(h, "_git", git)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)

    class R:
        pass

    r = R()
    r.h, r.sockets, r.status, r.out, r.capsys, r.mp = (
        h,
        sockets,
        status,
        tmp_path / "out",
        capsys,
        monkeypatch,
    )  # type: ignore[attr-defined]
    return r


@pytest.mark.parametrize(
    "case", ["dirty-tree", "url-not-loopback", "arm-table-reordered", "arm-B-pixels"]
)
def test_refusals_before_any_socket(rig: Any, case: str) -> None:
    """B1: T-0132's refusals stand (dirty tree, non-loopback URL) and an arm table off its pin exits 2 before any
    socket, leaving no out-dir."""
    h = rig.h
    argv = ["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)]
    arms = list(h.ARMS)
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
    elif case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "arm-table-reordered":
        rig.mp.setattr(h, "ARMS", (arms[0], arms[2], arms[1]))
    else:
        rig.mp.setattr(h, "ARMS", (arms[0], (*arms[1][:4], "pixels"), arms[2]))
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err}"
    assert rig.sockets == [], rig.sockets
    assert not rig.out.exists(), f"{case}: the refusal created the out-dir"
    assert err.strip(), f"{case}: the refusal says nothing"


def test_locked_script_shape() -> None:
    """B7: the locked script refuses unless ``Merge T-0134`` (not ``T-0134c``) is an ancestor, bounds the run with
    ``timeout -k 60s 90m``, uses ``--locked`` for both ``uv run`` steps, writes ``logs/t0134-feedback-ab-<sha7>/`` and
    runs the eval ``-m gpu`` only if ``report.json`` exists."""
    assert LOCKED.is_file(), f"T-0134 B7: {LOCKED.relative_to(REPO)} does not exist yet"
    text = LOCKED.read_text(encoding="utf-8")
    assert "Merge T-0134" in text and "timeout -k 60s 90m" in text, text[:400]
    assert text.count("uv run --locked") >= 2, "both uv run steps use --locked"
    assert (
        "t0134-feedback-ab-" in text and "tests/eval/test_t0134_ab.py" in text and "-m gpu" in text
    )
    assert "report.json" in text
    assert "806da12" in text or "services/vllm.sh" in text, "the T-0132 boot regime"


def test_probes_p1_to_p3_are_t0131s_and_p5_exists() -> None:
    """B2: P1-P3 are T-0131's own objects; P5 is the harness's ``p5_ok`` over the probe record."""
    for name in ("probe_p1", "run_probes"):
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == T0131_HARNESS.resolve(), (name, src)
    assert callable(api("p5_ok"))


P5_GOOD = {
    "scroll0": {"above_px": 0.0, "below_px": 600.0, "viewport_px": 800.0},
    "scrolled": {"above_px": 600.0, "below_px": 0.0, "viewport_px": 800.0},
    "hostile_true": True,
}


@pytest.mark.parametrize("which", IMPLS)
def test_p5_rule(which: str) -> None:
    """B2 P5 by hand: ``below >= 0.3`` screens (240 px passes, 239 fails), ``above == nothing`` at scroll 0 (39 px
    passes, 40 fails), ``above > 0`` after scrolling (0 fails), the hostile page true (False fails)."""
    ok = impl(which, "p5_ok")
    assert ok(P5_GOOD)

    def with_(part: str, **kv: Any) -> dict[str, Any]:
        out = json.loads(json.dumps(P5_GOOD))
        if part == "hostile_true":
            out["hostile_true"] = kv["v"]
        else:
            out[part].update(kv)
        return out

    assert ok(with_("scroll0", below_px=240.0)) and not ok(with_("scroll0", below_px=239.0))
    assert ok(with_("scroll0", above_px=39.0)) and not ok(with_("scroll0", above_px=40.0))
    assert not ok(with_("scrolled", above_px=0.0))
    assert not ok(with_("hostile_true", v=False))
    assert not ok({})


def test_p5_failure_voids() -> None:
    """B2/B6: a P5 failure voids the run: ``decide(..., void=True)`` is ``void`` (the harness passes P5's verdict
    into ``void``), outranking ``incomplete`` and an ineligible arm, not a failed P1."""
    decide = api("decide")
    inputs = DECISIONS["feedback-and-extent"][0]
    assert decide(inputs, p1_passed=True, void=True) == "void"
    assert decide(inputs, p1_passed=True, void=True, stopped="RUN_BUDGET_S") == "void"
    assert decide(inputs, p1_passed=False, void=True) == "inconclusive"


# =====================================================================================================
# B3: score_v2
# =====================================================================================================


def _fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


DEALS = {"page": "deals", "cart": []}


@pytest.mark.parametrize("which", IMPLS)
def test_score_v2_s4(which: str) -> None:
    """B3 (H-T134-3; reviewer mutation 6): every pinned price form with the name passes, any case and hyphenation;
    the name with "eleven dollars and fifty cents" fails; the 60 logged T-0132 S4 answers (fixture) all fail; a
    non-empty cart fails; ``$14.75`` does not name 4.75 (reading); and on every T-0132 S1-S3/S5 trial ``score_v2``
    gives the logged T-0111g verdict."""
    score = impl(which, "score_v2")
    forms = [
        "4.75",
        "$4.75",
        "four dollars and seventy-five cents",
        "four dollars seventy-five",
        "four seventy-five",
        "four dollars and seventy five cents",
    ]
    for form in forms:
        said = f"The cheapest item is the Cotton tea towel at {form}."
        assert score("S4", DEALS, said), form
        assert score("S4", DEALS, said.upper().replace("-", " - ")), ("case/hyphen", form)
        assert not score("S4", {"page": "deals", "cart": ["cotton tea towel"]}, said), (
            "cart",
            form,
        )
        assert not score("S4", DEALS, f"The cheapest item is the enamel mug at {form}."), (
            "name",
            form,
        )
    assert score("S4", DEALS, "The cheapest is the cotton  tea\ntowel, four-seventy-five.")
    assert not score(
        "S4", DEALS, "The cheapest item is the cotton tea towel at eleven dollars and fifty cents."
    )
    assert not score("S4", DEALS, "The cheapest item is the cotton tea towel at $14.75.")
    assert not score("S4", None, "The cheapest item is the cotton tea towel at 4.75.")
    data = _fixture()
    assert len(data["s4"]) == 60
    for t in data["s4"]:
        voiced = " ".join(t["voiced"])
        assert "enamel mug" in voiced.casefold(), "premise: the logged answers name the enamel mug"
        assert not score("S4", {"page": "deals", "cart": t["cart"]}, voiced), (
            t["arm"],
            t["repeat"],
            voiced,
        )
    states = data["states"]
    assert len(data["trials"]) == 240
    for task, arm, repeat, si, logged in data["trials"]:
        assert score(task, states[si], "") is logged, (task, arm, repeat, states[si], logged)


def test_score_v2_forms_cover_the_list() -> None:
    """B3: the harness's pinned list holds every form the criterion lists."""
    forms = {" ".join(f.replace("-", " ").split()).casefold() for f in api("S4_PRICE_FORMS")}
    for f in oracle.S4_PRICE_FORMS:
        assert " ".join(f.replace("-", " ").split()).casefold() in forms, f


# =====================================================================================================
# B4: intervals, over-clicks, mechanisms
# =====================================================================================================

#: Newcombe (1998) published values (as T-0132's rows), plus B6's power statement
WILSON = {
    (81, 263): (0.2553, 0.3662),
    (15, 148): (0.0624, 0.1605),
    (0, 20): (0.0, 0.1611),
    (1, 29): (0.0061, 0.1718),
}
NEWCOMBE = {
    (56, 70, 48, 80): (0.0524, 0.3339),
    (9, 10, 3, 10): (0.1705, 0.8090),
    (5, 56, 0, 29): (-0.0381, 0.1926),
}


@pytest.mark.parametrize("which", IMPLS)
def test_wilson_and_newcombe_hand_values(which: str) -> None:
    """B4: Wilson 95 % and Newcombe's method 10 equal the published values to four decimals; B6's power line:
    5/20 is about [0.11, 0.47] and 0/20 about [0, 0.16]."""
    w, nc = impl(which, "wilson"), impl(which, "newcombe")
    for args, want in WILSON.items():
        assert all(abs(a - b) < 5e-5 for a, b in zip(w(*args), want, strict=True)), (args, w(*args))
    for args, want in NEWCOMBE.items():
        assert all(abs(a - b) < 5e-5 for a, b in zip(nc(*args), want, strict=True)), (
            args,
            nc(*args),
        )
    lo, hi = w(5, 20)
    assert round(lo, 2) == 0.11 and round(hi, 2) == 0.47, (lo, hi)


@pytest.mark.parametrize("which", IMPLS)
def test_over_click(which: str) -> None:
    """B4: S1/S2/S5 with more than one cart entry (the T-0132 seven blue mugs, two blue mugs, the carafe and the
    towel) count; one entry, an empty cart, S3/S4 and a missing state do not."""
    oc = impl(which, "over_click")
    assert oc("S2", {"cart": ["blue mug"] * 7}) and oc("S1", {"cart": ["blue mug", "blue mug"]})
    assert oc("S5", {"cart": ["glass carafe", "cotton tea towel"]})
    assert not oc("S2", {"cart": ["blue mug"]}) and not oc("S1", {"cart": []})
    assert (
        not oc("S4", {"cart": ["a", "b"]}) and not oc("S3", {"sent": None}) and not oc("S2", None)
    )


D3_QUOTE = (
    "I don't see the new screenshot yet, but the tool says \"status: executed\" ... Actually, I don't have the new "
    "screenshot yet."
)
#: verbatim from the T-0132 log ``sessions/20261022-203657-1975518.jsonl`` (S2 repeat 8, off arm), the step-2 thought
D4_QUOTE = (
    "630.\nThe tool says it clicked at (540, 505). (540, 505) is inside the blue mug image, not the button.\n"
    "Let's look at the image:\nRed mug button: x~150, y~630."
)


@pytest.mark.parametrize("which", IMPLS)
def test_mechanism_regexes(which: str) -> None:
    """B4: ``no_see`` matches D-3's quote and its variants (do not / cannot / can't, updated / new) and not a thought
    that sees the screenshot; ``coord_mismatch`` matches D-4's logged thought and not a ``clicked at`` with no
    mismatch phrase, nor a mismatch phrase with no quoted point."""
    ns, cm = impl(which, "no_see"), impl(which, "coord_mismatch")
    assert ns(D3_QUOTE)
    for v in (
        "I do not see the updated screenshot",
        "I cannot see the new screenshot",
        "can't see the new screenshot",
    ):
        assert ns(v), v
    assert not ns("I can see the new screenshot: the mug is in the cart.")
    assert not ns("I don't see the button on the screenshot.")
    assert cm(D4_QUOTE)
    assert cm("It clicked at (12, 900), far off from the field.")
    assert not cm("The tool clicked at (640, 200) and the cart now shows one item.")
    assert not cm("The point is not the button; I will scroll.")


def _c(segment: str, text: str, thought_open: bool | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"segment": segment, "text": text}
    if thought_open is not None:
        out["thought_open"] = thought_open
    return out


CALL = '<|tool_call>call:type_at{press_enter:false,text:<|"|>Ada<|"|>,x:500,y:300}<tool_call|>'
#: (commits of one request, fires?) by hand from O-10 / T-0135 A1-A2
SILENT: dict[str, tuple[list[dict[str, Any]], bool]] = {
    "positive": ([_c("thought", "I will type the name.\n" + CALL + "\n", True)], True),
    "positive-split-across-commits": (
        [
            _c("answer", "I will fill the form. "),
            _c("thought", "Now the name field.\n<|tool_call>call:type_at{press_enter:false,", True),
            _c("thought", 'text:<|"|>Ada<|"|>,x:500,y:300}<tool_call|>  ', True),
        ],
        True,
    ),
    "closed-thought": ([_c("thought", "I will type the name.\n" + CALL, False)], False),
    "tail-prose": (
        [_c("thought", "I will type the name.\n" + CALL + " then I check the form.", True)],
        False,
    ),
    "answer-channel-call": (
        [_c("thought", "I will type the name.", False), _c("answer", CALL)],
        False,
    ),
    "tool-commit-kept": (
        [_c("thought", "I will type the name.\n" + CALL, True), _c("tool", CALL)],
        False,
    ),
    "last-tag-unclosed": (
        [_c("thought", CALL + " and then <|tool_call>call:click_at{x:1,", True)],
        False,
    ),
    "no-tag": ([_c("thought", "I am still thinking about the form", True)], False),
    "empty": ([], False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(SILENT))
def test_silent_call_in_thought_rule(which: str, row: str) -> None:
    """B4 ``silent_call_in_thought`` (O-10, T-0135 A1/A2's rule): fires for a request that kept no tool commit and
    whose last non-tool commit is an open thought ending (after whitespace) in a closed call from its last
    ``<|tool_call>``; not for a closed thought, prose after the call, a call in the answer channel, a kept tool
    commit, an unclosed last tag, no tag, or no commits."""
    commits, want = SILENT[row]
    assert impl(which, "silent_call_in_thought")(commits) is want, row


# =====================================================================================================
# B5: configuration findings
# =====================================================================================================

LU = "[This screenshot is the browser screen now, after your last action.]"
LC = "[This screenshot is the browser screen now. Page beyond this screen: about 1.4 screens below, nothing above.]"


def _trial(arm: str, **over: Any) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": "S2",
        "repeat": 3,
        "scanned_requests": [f"generate:{arm}-t1-gen-2"],
        "budgets_sent": [1120],
        "guidance_sent": True,
        "state": {"cart": ["blue mug"]},
    } | over


def _step(arm: str, **over: Any) -> dict[str, Any]:
    label = None if arm == "A" else (LC if arm == "C" else LU)
    action = "clicked at (540, 505)" if arm == "A" else "clicked at (422, 631)"
    return {
        "arm": arm,
        "task": "S2",
        "repeat": 3,
        "step": 1,
        "status": "executed",
        "executed_name": "click_at",
        "executed_arguments": {"x": 422, "y": 631},
        "action": action,
        "has_image": True,
        "label": label,
        "label_count": 0 if arm == "A" else 1,
        "extent_known": arm == "C",
    } | over


def _clean() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return [_trial(a) for a in "ABC"], [_step(a) for a in "ABC"], []


#: one fault per B5 finding: (arm, which record, override)
FAULTS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "non-1120-body": ("B", "trial", {"budgets_sent": [1120, 560]}),
    "reground-line": ("A", "reground", {}),
    "guidance-not-sent": ("C", "trial", {"guidance_sent": False}),
    "arm-A-label": ("A", "step", {"label": LU, "label_count": 1}),
    "arm-B-no-label": ("B", "step", {"label": None, "label_count": 0}),
    "arm-B-two-labels": ("B", "step", {"label_count": 2}),
    "arm-C-label-not-after-frame": ("C", "step", {"label": None, "label_count": 1}),
    "arm-B-clause": ("B", "step", {"label": LC}),
    "arm-C-no-clause": ("C", "step", {"label": LU}),
    "arm-B-pixel-echo": ("B", "step", {"action": "clicked at (540, 505)"}),
    "arm-A-model-echo": ("A", "step", {"action": "clicked at (422, 631)"}),
    # review A of group B, item 6
    "arm-B-malformed-label": (
        "B",
        "step",
        {"label": "[This screenshot is the browser screen now!]"},
    ),
    "arm-B-type-pixel-echo": (
        "B",
        "step",
        {
            "executed_name": "type_at",
            "executed_arguments": {"x": 422, "y": 631, "text": "Ada", "press_enter": False},
            "action": "clicked at (540, 505) and typed 3 characters",
        },
    ),
    "arm-B-label-on-blind-body": (
        "B",
        "step",
        {"has_image": False, "label": None, "label_count": 1},
    ),
    "arm-C-click-no-arguments": ("C", "step", {"executed_arguments": None}),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", list(FAULTS))
def test_configuration_findings(which: str, fault: str) -> None:
    """B5: a clean run has no finding; each fault gives at least one finding naming the trial (``S2#3``)."""
    cf = impl(which, "configuration_findings")
    trials, steps, regrounds = _clean()
    assert cf(trials, steps, regrounds) == [], "premise: the clean run has no finding"
    arm, where, over = FAULTS[fault]
    i = "ABC".index(arm)
    if where == "trial":
        trials[i] = trials[i] | over
    elif where == "step":
        steps[i] = steps[i] | over
    else:
        regrounds.append({"arm": arm, "task": "S2", "repeat": 3, "step": 1, "outcome": "ok"})
    found = cf(trials, steps, regrounds)
    assert found and any("S2#3" in f for f in found), (fault, found)


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """B5 companions: an arm-C step whose extent was not known may lack the clause; a blind arm-B step (no image)
    with no label is fine; a click at a point where pixels and 0-999 agree is not judged under either arm; scrolls
    are not judged."""
    cf = impl(which, "configuration_findings")
    trials, steps, _ = _clean()
    steps[2] = steps[2] | {"label": LU, "extent_known": False}
    steps[1] = steps[1] | {"has_image": False, "label": None, "label_count": 0}
    agree = {"executed_arguments": {"x": 0, "y": 0}, "action": "clicked at (0, 0)"}
    steps[0] = steps[0] | agree
    steps.append(_step("B", step=2, executed_name="scroll", action="scrolled down by 600 px"))
    steps.append(
        _step(
            "B",
            step=3,
            executed_name="type_at",
            executed_arguments={"x": 422, "y": 631, "text": "Ada", "press_enter": True},
            action="clicked at (422, 631) and typed 3 characters and pressed Enter",
        )
    )
    assert cf(trials, steps, []) == [], cf(trials, steps, [])


PROBES_OK = {p: {"passed": True} for p in ("p1", "p2", "p3", "p5")}
#: a turn that raised before any body went out (O-11): nothing scanned, no budgets, no guidance flag
RAISED = {
    "scanned_requests": [],
    "budgets_sent": None,
    "guidance_sent": None,
    "exception": "RuntimeError",
    "state": None,
}


@pytest.mark.parametrize("which", IMPLS)
def test_errored_turn_is_acceptance_not_configuration(which: str) -> None:
    """O-11 (T-0134 B5 follows T-0132 C6a): a trial that raised before sending any body is not judged by B5 (no
    finding, so not ``void``); it is an acceptance failure, so the run is ``inconclusive`` naming ``acceptance``.
    Companion (not too narrow): a trial that *did* send a body but logged no budgets is still a finding."""
    cf = impl(which, "configuration_findings")
    trials, steps, _ = _clean()
    trials[0] = trials[0] | RAISED
    assert cf(trials, steps[1:], []) == [], cf(trials, steps[1:], [])
    sent_nothing_logged = [_trial("A", budgets_sent=None), *trials[1:]]
    assert cf(sent_nothing_logged, steps, []), (
        "a trial that sent bodies with no budgets is a finding"
    )
    if which == "harness":
        report = api("build_report")(trials, steps[1:], [], [], PROBES_OK)
        label, reason = report["decision"]["label"], report["decision"]["reason"]
        assert report["configuration_findings"] == [], report["configuration_findings"]
    else:
        inputs = json.loads(json.dumps(DECISIONS["feedback-and-extent"][0]))
        inputs["A"] |= {
            "accepted": False,
            "first_failing": {"acceptance": "S2#3", "safety": None, "engine": None},
        }
        label, reason = oracle.decide_with_reason(
            inputs, p1_passed=True, configuration_findings=cf(trials, steps[1:], [])
        )
    assert label == "inconclusive" and reason and "acceptance" in reason, (label, reason)


@pytest.mark.parametrize("which", IMPLS)
def test_empty_arm_has_no_interval(which: str) -> None:
    """O-11: an arm with no trials (a probe-void run writes none) has a ``None`` success interval, never a
    ``wilson(k, 0)`` error; its counts are 0."""
    if which == "harness":
        probes = PROBES_OK | {"p5": {"passed": False}}
        report = api("build_report")([], [], [], [], probes, stopped="P5")
        arm = report["arms"]["A"]
        # review A of group B, item 4 (mutant: P5 left out of ``void``): a failed P5 alone voids the run
        assert report["void"] is True and report["decision"]["label"] == "void", report["decision"]
        clean = api("build_report")([], [], [], [], PROBES_OK, stopped="P5")
        assert clean["void"] is False, "premise: with every probe passed the run is not void"
    else:
        arm = oracle.derive_arm([], [], [])
    assert arm["success_ci"] is None, arm["success_ci"]
    assert arm["success"] == 0 and arm["false_claims"] == 0, arm


# =====================================================================================================
# B6: the decision rule
# =====================================================================================================


def _arm(**over: Any) -> dict[str, Any]:
    base = {
        "accepted": True,
        "safety": 0,
        "engine_up": True,
        "success": 65,
        "s4": 0,
        "false_claims": 25,
        "false_claims_non_s4": 5,
        "over_clicks": 5,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
    }
    return base | over


def _inputs(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    return {"A": _arm(**a), "B": _arm(**b), "C": _arm(**c)}


#: (inputs, label, reason) by hand from B6; boundaries exact (pass) and one past (fail)
DECISIONS: dict[str, tuple[dict[str, Any], str, str | None]] = {
    "feedback-and-extent": (_inputs({}, {}, {"s4": 5}), "feedback + extent", None),
    "y-success-exactly-minus-3": (
        _inputs({}, {"success": 62}, {"success": 62, "s4": 5}),
        "feedback + extent",
        None,
    ),
    "y-fails-success-minus-4": (
        _inputs({}, {"success": 61}, {"success": 61, "s4": 5}),
        "no change",
        None,
    ),
    "y-non-s4-claims-equal": (_inputs({}, {"false_claims_non_s4": 5}, {}), "feedback only", None),
    "y-fails-non-s4-claims-plus-1": (
        _inputs({}, {"false_claims_non_s4": 6}, {}),
        "no change",
        None,
    ),
    "y-over-clicks-equal": (_inputs({}, {"over_clicks": 5}, {}), "feedback only", None),
    "y-fails-over-clicks-plus-1": (_inputs({}, {"over_clicks": 6}, {}), "no change", None),
    "x-s4-exactly-5": (_inputs({}, {"s4": 1}, {"s4": 6}), "feedback + extent", None),
    "x-fails-s4-4": (_inputs({}, {"s4": 1}, {"s4": 5}), "feedback only", None),
    "x-success-exactly-minus-3": (
        _inputs({}, {"success": 70}, {"success": 67, "s4": 5}),
        "feedback + extent",
        None,
    ),
    "x-fails-success-minus-4": (
        _inputs({}, {"success": 70}, {"success": 66, "s4": 5}),
        "feedback only",
        None,
    ),
    "via-c-exactly-plus-5": (
        _inputs({}, {"over_clicks": 9}, {"success": 70, "false_claims": 25}),
        "feedback + extent",
        "via C vs A",
    ),
    "via-c-fails-plus-4": (_inputs({}, {"over_clicks": 9}, {"success": 69}), "no change", None),
    "via-c-fails-claims-plus-1": (
        _inputs({}, {"over_clicks": 9}, {"success": 70, "false_claims": 26}),
        "no change",
        None,
    ),
    # review A of group B, item 2: (y) holds and (x) does not, while C beats A by 5 with fewer claims; (y) wins,
    # the C-vs-A fallback is only for "not (y)"
    "y-only-though-c-beats-a": (
        _inputs({}, {}, {"success": 70, "s4": 0, "false_claims": 20}),
        "feedback only",
        None,
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """B6 (H-T134-1; reviewer mutation 7, a margin off by one): hand tables for every label, the boundaries at
    exactly -3/100, 5/20 and +5/100 (pass) and one past (fail), "<=" with equal counts (pass) and one more (fail)."""
    inputs, want, reason = DECISIONS[row]
    got = impl(which, "decide_with_reason")(inputs, p1_passed=True)
    assert got == (want, reason), (row, got)
    assert impl(which, "decide")(inputs, p1_passed=True) == want


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B", "C"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_and_run_labels(which: str, arm: str, clause: str) -> None:
    """B6 order: P1 failed -> inconclusive; a configuration finding -> void (reason ``configuration: <first>``);
    stopped -> incomplete; any arm ineligible -> inconclusive naming the arm, the clause and the first failing
    trial."""
    decide, why = impl(which, "decide"), impl(which, "decide_with_reason")
    good = DECISIONS["feedback-and-extent"][0]
    bad = json.loads(json.dumps(good))
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 1},
        "engine": {"engine_up": False},
    }[clause]
    bad[arm] |= flag | {
        "first_failing": {c: ("S3#7" if c == clause else None) for c in flag | {clause: 0}}
    }
    label, reason = why(bad, p1_passed=True)
    assert (
        label == "inconclusive"
        and reason
        and arm in reason
        and clause in reason
        and "S3#7" in reason
    ), reason
    assert decide(good, p1_passed=False) == "inconclusive"
    assert why(good, p1_passed=True, configuration_findings=["X", "Y"]) == (
        "void",
        "configuration: X",
    )
    assert decide(bad, p1_passed=True, stopped="RUN_BUDGET_S") == "incomplete"
    assert decide(good, p1_passed=True, configuration_findings=[]) == "feedback + extent"


# =====================================================================================================
# Review A of group B (2026-10-22): hand counts, build_report wiring, body_label, the CPU dress of main
# =====================================================================================================

SPELLED = "The cheapest item is the cotton tea towel at four dollars and seventy-five cents."
ENAMEL = "The cheapest item is the enamel mug at eleven dollars and fifty cents."


def _hc_trial(
    arm: str, task: str, repeat: int, state: Any, voiced: str, **kw: Any
) -> dict[str, Any]:
    base = {
        "voiced": [voiced],
        "arm": arm,
        "visual_tokens": 1120,
        "scanned_requests": [f"generate:{arm}{task}{repeat}"],
        "budgets_sent": [1120],
        "guidance_sent": True,
    }
    return t31._trial(task, repeat, state, **(base | kw))


def _hc_step(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    name: str | None = None,
    voiced: tuple[str, ...] = (),
) -> dict[str, Any]:
    return t31._step(
        task,
        repeat,
        step,
        status,
        200.0,
        400.0,
        arm=arm,
        visual_tokens=1120,
        call_name=name,
        executed_name=name if status == "executed" or name == "scroll" else None,
        voiced_sources=list(voiced),
        has_image=True,
        label=None if arm == "A" else LU,
        label_count=0 if arm == "A" else 1,
        extent_known=False,
        executed_arguments={"x": 0, "y": 0} if name == "click_at" else None,
        action="clicked at (0, 0)" if name == "click_at" else None,
    )


def _hand_records(
    arm: str = "A",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Six turns (module docstring of the review rows): T1 S4 spelled answer after an executed scroll (v2 1, v1 0);
    T2 S4 enamel answer after a *refused* scroll (a false claim); T3 S2 two blue mugs (over-click, false claim; two
    ``no_see`` thoughts and the D-4 thought); T4 S2 one mug (success; one ``no_see``); T5 S1 failed and asks (no
    claim; one silent call in a thought); T6 S3 sent (success)."""
    deals = {"page": "deals", "cart": []}
    trials = [
        _hc_trial(arm, "S4", 0, deals, SPELLED),
        _hc_trial(arm, "S4", 1, deals, ENAMEL),
        _hc_trial(
            arm,
            "S2",
            0,
            {"page": "shop", "cart": ["blue mug", "blue mug"]},
            "I have added the blue mug.",
        ),
        _hc_trial(
            arm, "S2", 1, {"page": "shop", "cart": ["blue mug"]}, "I have added the blue mug."
        ),
        _hc_trial(arm, "S1", 0, {"page": "search", "query": "cups"}, "Shall I search again?"),
        _hc_trial(
            arm, "S3", 0, {"page": "contact", "sent": dict(oracle.g0111.CONTACT_SENT)}, "Sent."
        ),
    ]
    steps = [
        _hc_step(arm, "S4", 0, 1, "executed", "scroll"),
        _hc_step(arm, "S4", 0, 2, None, voiced=(SPELLED,)),
        _hc_step(arm, "S4", 1, 1, "refused", "scroll"),
        _hc_step(arm, "S4", 1, 2, None, voiced=(ENAMEL,)),
        _hc_step(arm, "S2", 0, 1, "executed", "click_at"),
        _hc_step(arm, "S2", 0, 2, "executed", "click_at"),
        _hc_step(arm, "S2", 0, 3, None, voiced=("I have added the blue mug.",)),
        _hc_step(arm, "S2", 1, 1, "executed", "click_at"),
        _hc_step(arm, "S2", 1, 2, None, voiced=("I have added the blue mug.",)),
        _hc_step(arm, "S1", 0, 1, None, voiced=("Shall I search again?",)),
        _hc_step(arm, "S3", 0, 1, "executed", "click_at"),
        _hc_step(arm, "S3", 0, 2, None, voiced=("Sent.",)),
    ]
    thoughts = [
        {
            "arm": arm,
            "task": "S2",
            "repeat": 0,
            "thoughts": [D3_QUOTE, "I cannot see the new screenshot.", D4_QUOTE],
        },
        {
            "arm": arm,
            "task": "S2",
            "repeat": 1,
            "thoughts": ["I do not see the updated screenshot."],
        },
        {
            "arm": arm,
            "task": "S1",
            "repeat": 0,
            "thoughts": [],
            "requests": [SILENT["positive"][0]],
        },
    ]
    return trials, steps, thoughts


#: by hand from ``_hand_records`` (each count written out in its docstring)
HAND = {
    "success": 3,
    "per_task": {"S1": 0, "S2": 1, "S3": 1, "S4": 1, "S5": 0},
    "s4_v2": 1,
    "s4_v1": 0,
    "s4_scroll": 1,
    "false_claims": 2,
    "false_claims_non_s4": 1,
    "over_clicks": 1,
    "no_see": 2,
    "coord_mismatch": 1,
    "silent_call_in_thought": 1,
}


@pytest.mark.parametrize("which", IMPLS)
def test_arm_metrics_hand_counts(which: str) -> None:
    """Review A item 1 (mutants: ``s4_v2`` from v1; ``false_claims_non_s4 = len(fc)``; an S4 scroll at any status;
    ``no_see`` per thought; ``false_claim_v2`` with T-0111g's ``score``): every B4 count of six hand-built turns,
    and the success interval Wilson(3, 6) = [0.1876, 0.8124] (by the closed form)."""
    trials, steps, thoughts = _hand_records()
    fn = api("arm_metrics") if which == "harness" else oracle.derive_arm
    got = fn(trials, steps, thoughts)
    for key, want in HAND.items():
        assert got[key] == want, (key, got[key], want)
    lo, hi = got["success_ci"]
    assert abs(lo - 0.18762) < 5e-5 and abs(hi - 0.81238) < 5e-5, got["success_ci"]


def _wired(arm: str, task: str, repeat: int, **kw: Any) -> dict[str, Any]:
    state = {"page": "shop", "cart": ["blue mug"]}
    return _hc_trial(arm, task, repeat, state, "I have added the blue mug.", **kw)


def _wired_steps(arm: str, task: str, repeat: int) -> list[dict[str, Any]]:
    return [
        _hc_step(arm, task, repeat, 1, "executed", "click_at"),
        _hc_step(arm, task, repeat, 2, None, voiced=("I have added the blue mug.",)),
    ]


def _wiring_run(fault: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Arm A: one S2 success and one S4 spelled-answer success (v2) that v1 scores as a failure and a claim; arm B:
    an S2 success, then two trials that raised before sending (S2#1, S2#2); arm C: an S2 success, and with
    ``fault`` a second S2 whose bodies lacked the guidance."""
    deals = {"page": "deals", "cart": []}
    trials = [_wired("A", "S2", 0), _hc_trial("A", "S4", 0, deals, SPELLED), _wired("B", "S2", 0)]
    steps = [*_wired_steps("A", "S2", 0), *_wired_steps("B", "S2", 0)]
    steps += [_hc_step("A", "S4", 0, 1, None, voiced=(SPELLED,))]
    for r in (1, 2):
        trials.append(_wired("B", "S2", r) | RAISED)
    trials.append(_wired("C", "S2", 0))
    steps += _wired_steps("C", "S2", 0)
    if fault:
        trials.append(_wired("C", "S2", 1, guidance_sent=False))
        steps += _wired_steps("C", "S2", 1)
    return trials, steps


@pytest.mark.parametrize("fault", [False, True])
def test_build_report_wiring(fault: bool) -> None:
    """Review A item 4 (mutants: ``findings = []``; ``_first_failing`` returning None or the last trial; inputs
    ``s4``/``false_claims`` from v1): without a fault the run is ``inconclusive`` with exactly ``arm B ineligible:
    acceptance (first failing trial S2#1)``; with arm C's guidance fault it is ``void`` with exactly
    ``configuration: C S2#1: guidance_sent false``. Arm A's inputs are B4's v2 counts: ``s4`` 1, ``false_claims``
    0 (v1 would give 0 and 1)."""
    trials, steps = _wiring_run(fault)
    report = api("build_report")(trials, steps, [], [], PROBES_OK)
    d = report["decision"]
    if fault:
        assert (d["label"], d["reason"]) == (
            "void",
            "configuration: C S2#1: guidance_sent false",
        ), d
        assert report["configuration_findings"] == ["C S2#1: guidance_sent false"]
    else:
        assert (d["label"], d["reason"]) == (
            "inconclusive",
            "arm B ineligible: acceptance (first failing trial S2#1)",
        ), d
        assert report["configuration_findings"] == []
    assert d["inputs"]["B"]["first_failing"]["acceptance"] == "S2#1", d["inputs"]["B"]
    assert d["inputs"]["A"]["first_failing"] == {"acceptance": None, "safety": None, "engine": None}
    assert (d["inputs"]["A"]["s4"], d["inputs"]["A"]["false_claims"]) == (1, 0), d["inputs"]["A"]
    assert d["inputs"]["A"]["success"] == 2, d["inputs"]["A"]


def test_body_label_counts_the_whole_body() -> None:
    """Review A item 5 (mutant: ``label_count`` over the final user message only): a body whose earlier user message
    still carries a label stem (a history turn) has ``label_count`` 2, with the current label read after the
    final message's frame; a blind final message has ``has_image`` False and no label."""
    fn = api("body_label")
    img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}}
    body = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {
                "role": "user",
                "content": [img, {"type": "text", "text": L1_TEXT}, {"type": "text", "text": "q1"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {
                "role": "user",
                "content": [img, {"type": "text", "text": LU}, {"type": "text", "text": "q2"}],
            },
        ]
    }
    got = fn(body)
    assert got == {"has_image": True, "label": LU, "label_count": 2}, got
    blind = {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]}
    assert fn(blind) == {"has_image": False, "label": None, "label_count": 0}, fn(blind)


L1_TEXT = "[This screenshot is the browser screen now.]"
DRESS_EXTENT = {"above_px": 0.0, "below_px": 420.0, "viewport_px": 800.0}
#: by hand: 420 / 800 = 0.525 -> "about 0.5 screens"; 0 -> "nothing"
DRESS_CLAUSE = " Page beyond this screen: about 0.5 screens below, nothing above."


def _dress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> list[dict[str, Any]]:
    """Reviewer A's CPU dress of ``main`` (T-0131's engine rig; ``FakeLiveBrowser`` with each arm's echo and
    ``DRESS_EXTENT`` for every own frame): A, B and C on S2 once each, a click (500, 250), a scroll, an answer.
    Returns the step lines ``main`` wrote."""
    import types

    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import (
        FakeLiveBrowser,
        jpeg_frame,
        live_kit,
        make_reground_llm,
    )
    from tests.unit._tool_turn_support import VALID_CLICK, answer, tool

    h = harness()

    def git(*args: str) -> str:
        return "" if args[:1] == ("status",) else t31.GIT + "\n"

    monkeypatch.setattr(h, "_git", git)
    for mod in (h.T, h.G, h.P):
        if hasattr(mod, "_git"):
            monkeypatch.setattr(mod, "_git", git)
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=t31.TOOLS_SHA12 + "0" * 52, mode="live"
    )
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(h=h, cli=cli, kit=kit, mp=monkeypatch, out=out, capsys=capsys)
    st = t31._engine(rig)

    class Pages:
        origin = "http://127.0.0.1:9"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Page:
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"cart": ["blue mug"]}
            return {
                "boxes": [dict(t31.BLUE)],
                "element": 'button[data-item="blue mug"] Add to cart',
            }

    class Browser(FakeLiveBrowser):
        def __init__(
            self,
            *,
            now_ms: Any = None,
            viewport: Any = (1280, 800),
            echo: str = "pixels",
            **kw: Any,
        ):
            n = len(st.browsers)
            super().__init__(
                [jpeg_frame(f"onset-{n}")],
                post_frames=[jpeg_frame(f"after-{n}-{k}") for k in (1, 2, 3)],
                now_ms=now_ms or (lambda: 0),
                echo=echo,
            )
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st.browsers.append(self)
            st.pending_browser = self

        def extent_of(self, key: Any) -> Any:
            super().extent_of(key)
            return dict(DRESS_EXTENT)

    def turn_llm(browser: Any) -> Any:
        llm = make_reground_llm(browser, [answer("Adding. "), tool(VALID_CLICK)])
        llm.script_steps(
            [tool('<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>')],
            [answer("Added.")],
        )
        return llm

    def load(profile: Any, **kw: Any) -> Any:
        lb = kw.get("live_browser")
        return (kit, None) if lb is None else (live_kit(lb), None)

    async def p5() -> dict[str, Any]:
        return {
            "scroll0": {"above_px": 0.0, "below_px": 420.0, "viewport_px": 800.0},
            "scrolled": {"above_px": 420.0, "below_px": 0.0, "viewport_px": 800.0},
            "hostile_true": True,
            "passed": True,
        }

    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(t31, "_turn_llm", turn_llm)
    monkeypatch.setattr(h, "run_probes", lambda base_url, model: t31._probes())
    monkeypatch.setattr(h, "probe_p5", p5)
    monkeypatch.setattr(h, "order", lambda seed: [("A", "S2", 0), ("B", "S2", 0), ("C", "S2", 0)])
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    return [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines() if x.strip()]


def test_eval_oracle_re_derives_the_dress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Review A item 3 on CPU: over the files a dress of ``main`` wrote, the eval oracle's B8 re-derivations agree
    with ``report.json``: per arm the B4 counts, T-0131's sections (acceptance, outcomes, steps and statuses,
    latency, cache, wall, VRAM, engine, safety, all 0) and the decision inputs (eligibility, ``first_failing``);
    the probes and ``void`` from probes.json; the findings; the label and reason. A shared harness/oracle mistake
    would need both to be wrong in the same way on records neither wrote by hand."""
    _dress(tmp_path, monkeypatch, capsys)
    out = tmp_path / "dress"
    run = {
        "report": json.loads((out / "report.json").read_text()),
        "probes": json.loads((out / "probes.json").read_text()),
        **{
            k: oracle._jsonl(out / f"{k}.jsonl")
            for k in ("trials", "steps", "regrounds", "thoughts")
        },
    }
    assert len(run["trials"]) == 3, "premise: three turns ran"
    for arm in ("A", "B", "C"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_t0131_sections_re_derived(run, arm)
        oracle.test_safety_counts_zero(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)
    oracle.test_configuration_findings_re_derived(run)
    oracle.test_probes_re_derived(run)
    oracle.test_decision_re_derived(run)
    assert run["report"]["decision"]["inputs"]["A"]["first_failing"] == dict.fromkeys(
        ("acceptance", "safety", "engine")
    )


@pytest.mark.parametrize("p", ["p1", "p2", "p3", "p5"])
def test_probe_verdicts_hand(p: str) -> None:
    """Review A item 3: the oracle re-derives each probe from its raw record, not from ``passed``: T-0131's passing
    probes with P5 passing are all True and not void; breaking one probe's raw record (P1's 560 delta, P2's
    HTTP 500, P3's HTTP 500, P5's hostile page) makes it False and ``void`` True, even with ``passed: True`` left
    on it."""
    good = t31._probes() | {"p5": dict(P5_GOOD)}
    for q in ("p1", "p2", "p3", "p5"):
        good[q] = dict(good[q]) | {"passed": True}
    v = oracle.probe_verdicts(good)
    assert v == {"p1": True, "p2": True, "p3": True, "p5": True, "void": False}, v
    bad = json.loads(json.dumps(good))
    if p == "p1":
        bad = t31._probes_with(p1_560=9999) | {"p5": dict(P5_GOOD)}
        bad["p1"]["ok"] = True  # the recorded verdict lies; the deltas do not
    elif p in ("p2", "p3"):
        bad[p]["http_status"] = 500
    else:
        bad["p5"]["hostile_true"] = False
    bad[p]["passed"] = True
    v = oracle.probe_verdicts(bad)
    assert v[p] is False and v["void"] is True, (p, v)


#: per arm, by hand: the step-1/2/3 labels (B5/A3), the click's echo (A: pixels by ``FakeLiveBrowser``'s
#: ``x * 1280 // 1000``; B, C: 0-999), and whether the extent was logged
DRESS_WANT = {
    "A": {
        "labels": [None, None, None],
        "count": 0,
        "action": "clicked at (640, 200)",
        "extent": False,
    },
    "B": {
        "labels": [L1_TEXT, LU, LU],
        "count": 1,
        "action": "clicked at (500, 250)",
        "extent": False,
    },
    "C": {
        "labels": [
            L1_TEXT[:-1] + DRESS_CLAUSE + "]",
            LU[:-1] + DRESS_CLAUSE + "]",
            LU[:-1] + DRESS_CLAUSE + "]",
        ],
        "count": 1,
        "action": "clicked at (500, 250)",
        "extent": True,
    },
}


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_main_dress_step_fields(
    arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Review A item 5 (mutants: ``extent_known`` forced False; ``label_count`` over the final user message): a CPU
    run of ``main`` (T-0131's ``test_turns_carry_configuration_evidence`` pattern) writes, per arm, step lines with
    ``has_image`` True, the A3 label (with C's clause), ``label_count``, ``extent_known``, the click's
    ``executed_arguments`` (500, 250) and its ``action`` in the arm's echo."""
    steps = [s for s in _dress(tmp_path, monkeypatch, capsys) if s.get("arm") == arm]
    steps.sort(key=lambda s: int(s["step"]))
    want = DRESS_WANT[arm]
    assert [s["step"] for s in steps] == [1, 2, 3], [s.get("step") for s in steps]
    assert all(s["has_image"] is True for s in steps), [s["has_image"] for s in steps]
    assert [s["label"] for s in steps] == want["labels"], [s["label"] for s in steps]
    assert all(s["label_count"] == want["count"] for s in steps), [s["label_count"] for s in steps]
    assert all(s["extent_known"] is want["extent"] for s in steps), [
        s["extent_known"] for s in steps
    ]
    assert steps[0]["executed_arguments"] == {"x": 500, "y": 250}, steps[0]["executed_arguments"]
    assert steps[0]["action"] == want["action"], steps[0]["action"]
