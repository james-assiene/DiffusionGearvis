"""T-0137 B-OFF and B-10: with the repeat guard off (the default) everything is BASE_B's; the frozen paths are
byte-identical over the closed range; docs/02 §5 gains the ``**Spec delta (T-0137).**`` paragraph
(docs/tasks/T-0137.md).

- **The differential** runs ``_capture`` below in-process and in a subprocess over ``git archive BASE_B`` (T-0114's
  ``_base_tree``, origin checked). Sections:
  - T-0134's harness, imported (``test_t0134_differential._capture``: off/``dry_run`` tool, loop and toolless turns,
    the live loop, a live ReGround turn and the executor's action strings; request bodies, turn lines, session-log
    records and HUD messages);
  - ``cli`` itself (T-0152's ``_cli_run``, imported) for the live default and each of ``--reground``,
    ``--page-overview``, ``--auto-overview`` and ``--no-overview-frame``, under ``--dry-run`` and at a real start:
    exit code, stdout/stderr, every ``Session`` and executor keyword; the new ``repeat guard`` line is the only line
    dropped before comparing (B-9 "every other line ... BASE's");
  - a rig session built with exactly the keywords ``cli`` passed (the default, ``--no-overview-frame`` and
    ``--auto-overview``; T-0152's reading Rd-1 for the other two), driven through **four identical clicks on an
    unchanging screen** (the script the guard would stop) on a fitting page: every section T-0121's
    ``_with_history`` captures, the wire bodies and the browser invocations;
  - T-0134c's other modes (off and ``dry_run`` through ``cli``).
  Green at BASE_B and must stay green.
- **Byte identity** over ``BASE_B..TIP_B`` (two literals). ``TIP_B`` starts at BASE_B and is bumped by the test-author
  or the orchestrator to the implementation tip before review.
- ``test_doc02_delta_t0137`` reads docs/02 at ``TIP_B``: red until the paragraph lands and ``TIP_B`` moves.

**BASE_B note.** Re-pinned at activation to 4382224 (main with ``Merge T-0136 group C`` merged into this branch); it
was d4adf4f while the rows were written ahead. Activated for ``/implement T-0137`` group B: ``test_doc02_delta_t0137``
is red until the paragraph lands and ``TIP_B`` moves; the other three are green at BASE_B.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps), 1280x800 frames, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from tests import _git_range as gr
from tests.unit import _t0125_pins as t0125
from tests.unit import _t0160_pins as t0160
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0181_pins as t0181
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0134_differential as t0134
from tests.unit import test_t0134c_live_feedback_default as t0134c
from tests.unit import test_t0152_overview_frame_default as t0152
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins

#: T-0137 group B's BASE: 4382224, the merge of main (T-0136 A2, B and C landed, df658ba) into
#: ``T-0137-b-repeat-guard-session``, re-pinned at activation (was d4adf4f). A literal.
BASE_B = "4382224"
#: The implementation tip: 486edba, group B's last implementation commit (bumped from BASE_B by the test-author).
TIP_B = "486edba"
HEADER = "**Spec delta (T-0137).**"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
LINE_PREFIX = "astra session: repeat guard: "

#: B-OFF's byte-identical list, written out from the criterion
FROZEN = (
    "local_astra/toolcall",
    "local_astra/computer",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/degeneracy.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/progress.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/perception",
    "local_astra/pages",
    "hud",
    "services",
    "scripts",
)

#: the live cases through ``cli``: (flags, profile body after T-0151's ``LIVE_PROFILE``)
CLI_CASES: dict[str, tuple[list[str], str]] = {
    "default": ([], ""),
    "reground": (["--reground"], ""),
    "page-overview": (["--page-overview"], ""),
    "auto-overview": (["--auto-overview"], ""),
    "no-overview-frame": (["--no-overview-frame"], ""),
}
RIG_CASES = ("default", "no-overview-frame", "auto-overview")


def _same(want: Any, got: Any, what: str) -> None:
    assert got == want, (
        f"T-0137 B-OFF: {what} differs from BASE_B {BASE_B}:\nbase {json.dumps(want)[:1500]}\n"
        f"now  {json.dumps(got)[:1500]}"
    )


def _drop_line(run: dict[str, Any]) -> dict[str, Any]:
    out = dict(run)
    for k in ("out", "err"):
        out[k] = "\n".join(x for x in str(run[k]).splitlines() if not x.startswith(LINE_PREFIX))
    return out


async def _repeat_turn(tmp: Path, kwargs: dict[str, Any], echo: str) -> dict[str, Any]:
    """Four identical clicks on an unchanging screen (each post-action capture is the onset capture again), then a
    report, on a rig session built with ``kwargs``; T-0152's ``_rig_turn`` sections."""
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_session_live import pass1_body
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0121_differential import _with_history
    from tests.unit.test_t0134_label import all_bodies

    tmp.mkdir(parents=True, exist_ok=True)
    f0 = live_frame("t0137-onset")
    browser = FakeLiveBrowser([f0], post_frames=[f0.model_copy() for _ in range(4)], echo=echo)
    click = "<|tool_call>call:click_at{x:500,y:250}<tool_call|>"
    async with feedback_rig(
        tmp,
        browser,
        script=[answer("Step 1 text."), tool(click)],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps(
            *([answer(f"Step {k} text."), tool(click)] for k in (2, 3, 4)), [answer("Done. ")]
        )
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 4, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
        bodies = all_bodies(rig)
        pass1 = pass1_body(rig)
    out = _with_history(rig, tmp, session_id, log_path)
    text = json.dumps({"bodies": bodies, "pass1": pass1}).replace(session_id, "<session>")
    out.update(json.loads(text.replace(str(tmp), "<tmp>")))
    out["invocations"] = [[i.name, i.call_id, dict(i.arguments)] for i in browser.invocations]
    # the model's calls only (the default overview frame's capture is the session's own ``*-overview-frame``)
    out["executes"] = sum(1 for c in browser.calls if c[0] == "execute" and "-call" in str(c[1]))
    return out


async def _rigs(tmp: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in RIG_CASES:
        run = runs[case]
        if run["code"] != 0 or not run["_raw_kwargs"]:
            out[case] = None
            continue
        kw, echo = t0152.rig_kwargs(run)
        out[case] = await _repeat_turn(tmp / case, kw, echo)
    return out


def _capture(tmp: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"cli": {}}
    out["t0134"] = asyncio.run(t0134._capture(tmp / "t0134"))
    runs: dict[str, dict[str, Any]] = {}
    for case, (flags, body) in CLI_CASES.items():
        # T-0174 D4b + T-0179 A-5: the working tree's live session pins v1 and the guard off (BASE_B's); a no-op in the BASE run
        dry = t0152._cli_run(
            tmp / "cli" / case / "dry",
            flags,
            body,
            dry=True,
            pin_v1=True,
            pin_guard_off=True,
            pin_reminder_off=True,  # T-0201 D3 re-pin: the reminder pinned off (BASE_B's requests)
            pin_wait_off=True,  # T-0202 D3 re-pin: the wait tool pinned off (BASE_B's kit and guidance)
            pin_check_off=True,  # T-0207 C3 re-pin: the completion check pinned off (BASE_B's turns)
        )
        start = t0152._cli_run(
            tmp / "cli" / case / "start",
            flags,
            body,
            dry=False,
            pin_v1=True,
            pin_guard_off=True,
            pin_reminder_off=True,  # T-0201 D3 re-pin
            pin_wait_off=True,  # T-0202 D3 re-pin
            pin_check_off=True,  # T-0207 C3 re-pin
        )
        runs[case] = start
        out["cli"][case] = {"dry": t0152._public(dry), "start": t0152._public(start)}
    out["rigs"] = asyncio.run(_rigs(tmp / "rig", runs))
    out["other"] = t0134c._other_modes(tmp / "other")
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0137_differential as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_B)
    tmp = tmp_path_factory.mktemp("t0137_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_B)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    # T-0155 O-8: the persona sentence only, on the observed side
    return observed(_capture(tmp_path_factory.mktemp("t0137_now")))


# T-0125 pin move: the default live search line, startup line, kit
# T-0160 ruling 2: web on by default in live (egress=web, executor web=True)
def test_off_equals_base(base: dict[str, Any], now: dict[str, Any]) -> None:
    """B-OFF (reviewer mutation 11; an always-on guard): T-0134's sessions in every mode, ``cli`` for the default and
    each listed flag (``--dry-run`` and a real start, every keyword, every line but the new one), the rig sessions
    built from ``cli``'s keywords through four identical clicks (all four executed at BASE), and the other modes all
    equal BASE_B's; no ``Session`` gets ``repeat_guard``, no turn line or record gains a guard key."""
    b, n = base["t0134"], now["t0134"]
    from tests.unit import test_t0107a_differential as t0107a

    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            t0134._same(
                b["t0111g"][scenario][section],
                n["t0111g"][scenario][section],
                f"{scenario} {section}",
            )
    for section in t0107a.SECTIONS:
        t0134._same(b["live"][section], n["live"][section], f"live {section}")
    for section in ("log", "records", "bodies", "pass2"):
        t0134._same(b["live_reground"][section], n["live_reground"][section], f"reground {section}")
    t0134._same(b["executor"], n["executor"], "executor actions")
    for case in CLI_CASES:
        for kind in ("dry", "start"):
            want, got = base["cli"][case][kind], now["cli"][case][kind]
            assert got["code"] == want["code"] == 0, (case, kind, got["err"][-800:])
            got = t0172.admit_reask_run(
                got, kind
            )  # T-0172 R12 pin move: the default reask line and keyword
            got = t0125.admit_live_run(got, kind)  # T-0125 pin move
            got = t0160.admit_live_run(
                got, kind
            )  # T-0160 pin move (ruling 2: web on by default in live)
            got = t0181.run_without_mic_unavailable(got)  # T-0181 A-R13 pin move
            _same(_drop_line(want), _drop_line(got), f"cli {case} {kind}")
            for kw in got["session_kwargs"]:
                assert "repeat_guard" not in kw, f"T-0137 B-9: off passes no keyword ({case}); {kw}"
    for case in RIG_CASES:
        want, got = base["rigs"][case], now["rigs"][case]
        assert want is not None and want["executes"] == 4, (
            f"premise: BASE executes all four clicks ({case})"
        )
        assert got is not None
        got = t0172.admit_reask_turn_key(
            got
        )  # T-0172 R12 pin move (B7): cli-built rigs carry the default key
        for key in want:
            _same(want[key], got[key], f"rig {case} {key}")
    for case, *_ in t0134c.OTHER:
        for key in ("dry", "build", "vllm_kwargs", "session_kwargs", "executors", "system", "body"):
            got_other = t0172.admit_reask_other(
                case, key, now["other"][case][key]
            )  # T-0172 R12 (B9)
            if key == "session_kwargs":
                got_other = t0125.kit_without_web_search(got_other)  # T-0125 pin move
                got_other = t0181.without_mic_unavailable(got_other)  # T-0181 A-R13 pin move
            _same(base["other"][case][key], got_other, f"other {case} {key}")
    flat = json.dumps(now)
    assert (
        '"repeat_guard"' not in flat
        and '"no_progress"' not in flat
        and "repeat_of_step" not in flat
    )


def test_frozen_shas() -> None:
    """B-OFF: ``LIVE_TOOL_GUIDANCE`` 768a973b..., ``tools_v1.json`` bbfc53cc... and ``tools_v1_overview.json``
    8048bdb7..., in the working tree and at both ends of the range."""
    gr.require_commits(BASE_B, TIP_B)
    assert hashlib.sha256(pb.LIVE_TOOL_GUIDANCE.encode()).hexdigest() == t0134.GUIDANCE_SHA
    root = Path(__file__).resolve().parents[2] / "local_astra/toolcall"
    for name, sha in (("tools_v1.json", V1_SHA), ("tools_v1_overview.json", OV_SHA)):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        for rev in (BASE_B, TIP_B):
            blob = gr._git("show", f"{rev}:local_astra/toolcall/{name}")
            assert blob.returncode == 0 and hashlib.sha256(blob.stdout.encode()).hexdigest() == sha


def test_frozen_paths_over_closed_range() -> None:
    """B-OFF: the frozen paths are byte-identical over ``BASE_B..TIP_B`` (two literals, never the working tree), and
    each exists at BASE_B. Power: the same predicate sees T-0111g's ``versions.lock`` change."""
    gr.require_commits(BASE_B, TIP_B)
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B} is not an ancestor of {TIP_B}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_B, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {BASE_B}"
        )
    changed = gr.changed(BASE_B, TIP_B, *FROZEN)
    assert changed == [], f"T-0137 B-OFF: frozen paths differ in {BASE_B}..{TIP_B}: {changed}"
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, "services"), "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0137() -> None:
    """B-10: docs/02 §5 at ``TIP_B`` holds exactly one paragraph starting ``**Spec delta (T-0137).**`` with the
    keywords ``repeat_of_step``, ``--repeat-guard``, ``MAX_NO_CHANGE_REPEATS`` and ``T-0136`` (plus, the
    test-author's reading of B-10's list, ``MAX_SCREEN_VISITS``, both reasons verbatim, "budget", "spoken" and "25");
    removing it leaves BASE_B's docs/02 paragraph for paragraph."""
    gr.require_commits(BASE_B, TIP_B)
    assert TIP_B != BASE_B, "T-0137 B-10: TIP_B is not bumped to the implementation tip yet"
    doc, base_doc = _doc02_at(TIP_B), _doc02_at(BASE_B)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0137 B-10: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    flat = " ".join(para.replace("`", "").split())
    need = [
        "repeat_of_step",
        "--repeat-guard",
        "MAX_NO_CHANGE_REPEATS",
        "MAX_SCREEN_VISITS",
        "T-0136",
        "the same action did not change this screen the last two times; it was not done again",
        "this screen has come back for the third time in this request; the action was not done",
        "25",
    ]
    missing = [k for k in need if k not in flat]
    low = flat.lower()
    missing += [k for k in ("budget", "spoken") if k not in low]
    assert not missing, f"T-0137 B-10: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0137 B-10: docs/02 changes only by the one paragraph"
    )
