"""T-0132 group A: with ``tools = live``, an unset budget becomes 1120 and an unset ReGround becomes on; the ``grounding``
line, the turn-line key and the docs delta (docs/tasks/T-0132.md, A1-A4, A7).

**The interfaces these rows pin** (the task's, plus the readings named in the test-author report):

- ``cli`` builds ``VllmClient(..., visual_tokens=<resolved>)`` and ``Session(..., visual_tokens=<resolved>,
  reground=<resolved>, grounding=<dict>)`` in live (A3, A4c). The resolver's name is the implementer's (the task says
  "e.g. ``grounding_of``"), so no row calls it: the sources are read where the task prints and logs them.
- ``grounding`` (A4c) is a dict with exactly the keys ``visual_tokens``, ``visual_tokens_from``, ``reground``,
  ``reground_from``, ``reground_visual_tokens``, ``reground_visual_tokens_from``. **Readings:** ``reground`` is a
  ``bool``; a source is one of ``"flag"``, ``"profile"``, ``"default"`` (``"unset"`` is never a live value: every live
  value resolves). With ReGround off the value of ``reground_visual_tokens`` is not pinned (the task omits it from the
  printed line and says nothing about the dict). ``cli`` passes no ``grounding`` (absent or ``None``) in ``off`` and
  ``dry_run``.
- The ``grounding`` line (A4a), live only, literal: ``astra session: grounding: visual_tokens=<N> (<src>),
  reground=on (<src>), reground_visual_tokens=<N> (<src>)``; with ReGround off ``astra session: grounding:
  visual_tokens=<N> (<src>), reground=off (<src>)``. After the tools line under ``--dry-run``; before the regime line
  at a real start.
- A source given explicitly is reported as given, even when its value equals the default (``--visual-tokens 1120`` is
  ``flag``; reviewer mutation 9).

The rigs are T-0131's and T-0111g's (``test_cli_t0131``, ``test_cli_tools_live``, ``_tools_switch_support``,
``_live_browser_support``), imported, not copied.

Activated 2026-10-22 (group A; until then ``pending(task="T-0132")``): the rows fail at BASE a2d9ac6 on a named
``AssertionError`` (no default, no
``grounding`` keyword, no line). ``test_tools_line_function_unchanged`` is a byte-identity row: green at BASE, no
marker.

**T-0133 re-scope R1** (docs/tasks/T-0133.md; human ruling 2026-10-22: live defaults to 1120 with ReGround *off*;
H-T133-1 approved): an unset ReGround in live now resolves ``False`` from ``"default"``, so ``DEFAULT_GROUNDING``,
the default rows, ``OVERRIDES``, ``LINES`` and the tools-line row assert that. ``reground_visual_tokens`` is dropped
from the compared default subset (O-1: unpinned with ReGround off). With ``--reground`` at 1120 the crop defaults to
280 and a second client is built, so the main budget is read off the first client (``vllm_kwargs[0]``), never
``vllm_kwargs[-1]``. Red against main until T-0133 is implemented, by design.

Regime: CPU over fakes, ``ScriptedClock`` for the turn rows, thinking on, 200 ms per word; nothing served; n = 1.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.contracts import events as E
from tests import _git_range as gr
from tests.unit import _t0160_pins as t0160
from tests.unit import _tools_switch_support as T
from tests.unit import test_t0107a_docs as t7d
from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, reground_rig
from tests.unit._t0202_pins import admitted_tools_lines as tools_lines  # T-0202 D3 re-pin
from tests.unit._tool_turn_support import answer
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, build, dry_run
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live

PREFIX = "astra session: grounding: "
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
GROUNDING_KEYS = frozenset(
    {
        "visual_tokens",
        "visual_tokens_from",
        "reground",
        "reground_from",
        "reground_visual_tokens",
        "reground_visual_tokens_from",
    }
)
#: A4c with no flag and no profile key, written out from A2 (T-0133 R1: ReGround off from the default; the pass-2
#: budget is not compared, O-1)
DEFAULT_GROUNDING = {
    "visual_tokens": 1120,
    "visual_tokens_from": "default",
    "reground": False,
    "reground_from": "default",
}


def grounding_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(PREFIX)]


def grounding_kw(calls: T.Calls) -> dict[str, Any]:
    """The ``grounding`` keyword ``cli`` passed to ``Session`` (A4c); its absence is a named failure."""
    got = calls.session_kwargs[-1].get("grounding")
    assert isinstance(got, dict), (
        "T-0132 A4c: cli passes Session(..., grounding=<dict>) in live; it passed "
        f"{got!r} (session kwargs {sorted(calls.session_kwargs[-1])})"
    )
    assert set(got) == GROUNDING_KEYS, f"T-0132 A4c: the grounding keys; got {sorted(got)}"
    return got


def build_out(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> tuple[T.Calls, str, int]:
    """A real start through ``cli`` over the fakes (nothing served): the recorded calls, stdout and the exit code."""
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *flags, config_body=body))
    out, err = _out(capsys)
    return calls, out + err, code


# --------------------------------------------------------------------------------------------- A2/A3: the default


def test_live_defaults_1120_reground_off(tmp_path, capsys, monkeypatch):
    """A2/A3 (T-0133 R1): live with no flag and no profile key builds ``VllmClient(visual_tokens=1120)`` and
    ``Session(visual_tokens=1120)`` with ``reground`` falsy, and the session's ``grounding`` says the budget and
    ReGround (off) came from the default."""
    calls = build(tmp_path, capsys, monkeypatch, "live")
    assert calls.vllm_kwargs[0].get("visual_tokens") == 1120, (
        f"T-0132 A3: the live client is built with visual_tokens=1120 by default; kwargs {calls.vllm_kwargs[0]}"
    )
    kw = calls.session_kwargs[-1]
    assert kw.get("visual_tokens") == 1120, f"T-0132 A3: Session(visual_tokens=1120); {kw}"
    assert not kw.get("reground", False), f"T-0133 R1: Session's reground is falsy by default; {kw}"
    g = grounding_kw(calls)
    assert {k: g[k] for k in DEFAULT_GROUNDING} == DEFAULT_GROUNDING, g


#: (flags, profile body, expected client budget, expected reground, the grounding subset). T-0133 R1: an unset
#: ReGround is now off from the default.
OVERRIDES = {
    "flag-280": (
        ["--visual-tokens", "280"],
        "",
        280,
        False,
        {"visual_tokens_from": "flag", "reground_from": "default"},
    ),
    "flag-560": (
        ["--visual-tokens", "560"],
        "",
        560,
        False,
        {"visual_tokens_from": "flag", "reground_from": "default"},
    ),
    "flag-1120-equal-to-default": (
        ["--visual-tokens", "1120"],
        "",
        1120,
        False,
        {"visual_tokens_from": "flag", "reground_from": "default"},
    ),
    "no-reground": (
        ["--no-reground"],
        "",
        1120,
        False,
        {"visual_tokens_from": "default", "reground_from": "flag"},
    ),
    "reground-flag": (["--reground"], "", 1120, True, {"reground_from": "flag"}),
    "profile-reground-true": (
        [],
        'tools = "live"\nreground = true\n',
        1120,
        True,
        {"visual_tokens_from": "default", "reground_from": "profile"},
    ),
    "profile-560": (
        [],
        "visual_tokens = 560\n",
        560,
        False,
        {"visual_tokens_from": "profile", "reground_from": "default"},
    ),
    "profile-reground-false": (
        [],
        "reground = false\n",
        1120,
        False,
        {"visual_tokens_from": "default", "reground_from": "profile"},
    ),
    "flag-over-profile": (
        ["--visual-tokens", "280"],
        "visual_tokens = 560\n",
        280,
        False,
        {"visual_tokens_from": "flag", "reground_from": "default"},
    ),
}


@pytest.mark.parametrize("case", list(OVERRIDES))
def test_flag_and_profile_override_default(tmp_path, capsys, monkeypatch, case):
    """A2/A3 (reviewer mutation 9): ``--visual-tokens 280``/``560``, ``--no-reground``, profile ``visual_tokens = 560``
    and profile ``reground = false`` override the default, in the client, the session and the ``grounding`` sources;
    a flag equal to the default (``--visual-tokens 1120``) is still ``flag``, never ``default``.

    T-0133 R1: an unset ReGround is off (``default``); ``--reground`` and profile ``reground = true`` turn it on
    (``flag``, ``profile``). The main budget is read off the first client and the session's ``llm`` (at 1120 with
    ReGround on a second, crop client follows it), never ``vllm_kwargs[-1]``."""
    flags, body, budget, reground, sources = OVERRIDES[case]
    calls = build(tmp_path, capsys, monkeypatch, "live", *flags, body=body)
    assert calls.vllm_kwargs[0].get("visual_tokens") == budget, (case, calls.vllm_kwargs[0])
    kw = calls.session_kwargs[-1]
    assert kw.get("visual_tokens") == budget, (case, kw)
    assert bool(kw.get("reground", False)) is reground, (case, kw)
    g = grounding_kw(calls)
    assert g["visual_tokens"] == budget and g["reground"] is reground, (case, g)
    got = {k: g[k] for k in sources}
    assert got == sources, f"T-0132 A2 {case}: the sources; got {got}, want {sources}"


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
def test_other_modes_unchanged(tmp_path, capsys, monkeypatch, override):
    """A2 (adversarial; reviewer mutations 1 and 2: the default keyed on ``profile.tools``, or applied in
    ``dry_run``): a live profile run with ``--tools off`` or ``--tools dry-run`` gets no budget, no ReGround, no
    ``grounding`` and exit 0, with and without ``--dry-run`` (no grounding line, BASE's tools line); plain ``off`` and
    ``dry_run`` give ``visual_tokens is None``. Premise: the same live profile with no override gets the default, so
    the resolved mode is what removes it.

    T-0183 re-scope (A1/A2/A3, human ruling 2026-10-28: 1120 by default in every tools mode): the overridden live
    profile and plain ``off``/``dry_run`` now get the budget 1120 (client and session), still no ReGround and no
    ``grounding`` (keyword or line); the tools line is BASE's plus `` visual_tokens=1120``. The premise now also
    requires the live profile alone to carry ``grounding``, which is what the override removes."""
    calls = build(tmp_path / "premise", capsys, monkeypatch, "off", body=LIVE_PROFILE)
    assert calls.vllm_kwargs[-1].get("visual_tokens") == 1120, (
        "premise (T-0132 A2): the live profile alone gets the 1120 default"
    )
    assert calls.session_kwargs[-1].get("grounding") is not None, (
        "premise (T-0132 A4c): the live profile alone passes grounding"
    )
    monkeypatch.undo()
    calls, out, code = build_out(
        tmp_path / "over", capsys, monkeypatch, *override, body=LIVE_PROFILE
    )
    assert code == 0, f"T-0132 A2: a live profile under {override} exits 0; got {code}\n{out}"
    assert calls.vllm_kwargs[-1].get("visual_tokens") == 1120, calls.vllm_kwargs[-1]
    kw = calls.session_kwargs[-1]
    assert kw.get("visual_tokens") == 1120 and not kw.get("reground", False), kw
    assert kw.get("grounding") is None, f"T-0132 A4c: cli passes grounding in live only; {kw}"
    assert grounding_lines(out) == [], out
    monkeypatch.undo()
    mode = "off" if override[1] == "off" else "dry_run"
    printed = dry_run(tmp_path / "dry", capsys, monkeypatch, "off", *override, body=LIVE_PROFILE)
    assert tools_lines(printed) == [BASE_LINES[mode] + " visual_tokens=1120"], printed
    assert "grounding:" not in printed, printed
    for plain in ("off", "dry_run"):
        monkeypatch.undo()
        calls = build(tmp_path / f"plain-{plain}", capsys, monkeypatch, plain)
        assert calls.vllm_kwargs[-1].get("visual_tokens") == 1120, (plain, calls.vllm_kwargs[-1])
        assert calls.session_kwargs[-1].get("visual_tokens") == 1120
        assert calls.session_kwargs[-1].get("grounding") is None


# --------------------------------------------------------------------------------------------- A4a/b: saying it

#: (flags, profile body, the literal grounding line) -- written out from A4a (T-0133 R1: ReGround off by default;
#: ``--reground`` at 1120 gets the 280 crop from the default, H-T133-1)
LINES = {
    "default": (
        [],
        "",
        "visual_tokens=1120 (default), reground=off (default)",
    ),
    "flag": (
        ["--visual-tokens", "560"],
        "",
        "visual_tokens=560 (flag), reground=off (default)",
    ),
    "flag-reground": (
        ["--reground"],
        "",
        "visual_tokens=1120 (default), reground=on (flag), reground_visual_tokens=280 (default)",
    ),
    "profile": (
        [],
        "visual_tokens = 280\n",
        "visual_tokens=280 (profile), reground=off (default)",
    ),
    "off-flag": (["--no-reground"], "", "visual_tokens=1120 (default), reground=off (flag)"),
    "off-profile": (
        ["--visual-tokens", "560"],
        "reground = false\n",
        "visual_tokens=560 (flag), reground=off (profile)",
    ),
}


@pytest.mark.parametrize("case", list(LINES))
def test_grounding_line_exact(tmp_path, capsys, monkeypatch, case):
    """A4a: the live ``--dry-run`` prints exactly one ``grounding`` line, after the tools line, with the literal
    text for the default, flag and profile sources and for ReGround off; the same line is printed at a real start,
    before the regime line. No ``grounding:`` line is printed in ``off`` or ``dry_run``."""
    flags, body, rest = LINES[case]
    want = PREFIX + rest
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live", *flags, body=body)
    assert grounding_lines(out) == [want], f"T-0132 A4a {case}: printed\n{out}"
    lines = out.splitlines()
    (tools_at,) = [i for i, line in enumerate(lines) if line.startswith("tools=live ")]
    assert lines.index(want) > tools_at, (
        f"T-0132 A4a: the grounding line follows the tools line\n{out}"
    )
    monkeypatch.undo()
    _, started, code = build_out(
        tmp_path / "start", capsys, monkeypatch, *MODE_FLAGS["live"], *flags, body=body
    )
    assert code == 0, started
    lines = started.splitlines()
    assert grounding_lines(started) == [want], f"T-0132 A4a {case} at a real start:\n{started}"
    regime = [i for i, line in enumerate(lines) if line.startswith("astra session: regime:")]
    assert regime and lines.index(want) < regime[0], (
        f"T-0132 A4a: the grounding line comes before the regime line at a real start\n{started}"
    )
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        other = dry_run(tmp_path / f"other-{mode}", capsys, monkeypatch, mode)
        assert "grounding:" not in other, f"T-0132 A4a: no grounding line with {mode}\n{other}"


def test_tools_line_function_unchanged():
    """A4b/A5 (byte identity, green at BASE; reviewer mutation 3): ``cli.tools_line`` keeps BASE's signature and
    returns BASE's strings; the resolved suffixes are added outside it."""
    from local_astra.toolcall import tools

    params = list(inspect.signature(cli.tools_line).parameters)
    assert params == ["kit", "loop_max", "live_line"], params
    kit = T.make_kit()
    assert kit.tools_version == tools.load().version, "premise: the kit is the native tool set"
    live = cli.LiveLine(browser="headless", viewport=(1280, 800))
    assert cli.tools_line(kit, 8, live) == BASE_LINES["live"]
    assert cli.tools_line(kit, 8) == BASE_LINES["dry_run"]
    assert cli.tools_line(None) == BASE_LINES["off"]


# T-0160 ruling 2: web on by default in live (egress=web)
def test_dry_run_live_tools_line_gains_resolved_suffixes(tmp_path, capsys, monkeypatch):
    """A4b (T-0133 R1): the default live ``--dry-run`` tools line is BASE's plus `` visual_tokens=1120``; with
    ``--no-reground`` the same; with ``--visual-tokens 280`` BASE's plus `` visual_tokens=280``; with ``--reground``
    BASE's plus `` visual_tokens=1120 reground=on``."""
    base = t0160.web_line(BASE_LINES["live"])  # T-0160 pin move (C-2)
    got = dry_run(tmp_path, capsys, monkeypatch, "live")
    assert tools_lines(got) == [base + " visual_tokens=1120"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--no-reground")
    assert tools_lines(got) == [base + " visual_tokens=1120"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--visual-tokens", "280")
    assert tools_lines(got) == [base + " visual_tokens=280"], got
    got = dry_run(tmp_path, capsys, monkeypatch, "live", "--reground")
    assert tools_lines(got) == [base + " visual_tokens=1120 reground=on"], got


# --------------------------------------------------------------------------------------------- A4c: the turn line


async def _rig_turn(tmp_path: Path, **session_kwargs: Any) -> dict[str, Any]:
    """One toolless live turn on a rig-built session with ``reground=True`` and ``session_kwargs``."""
    browser = FakeLiveBrowser([jpeg_frame("onset")])
    async with reground_rig(
        tmp_path, browser, script=[answer("It is a shop. ")], session_kwargs=session_kwargs
    ) as rig:
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="what is on the page"))
        await rig.turn_done()
        return dict(rig.record("t1").extras)


def test_turn_line_grounding_key(tmp_path, capsys, monkeypatch):
    """A4c: ``cli`` hands a live session the ``grounding`` dict (from a flag run: the sources are not all
    ``default``), and a session given it writes it on the turn line as given, beside T-0131's ``visual_tokens`` and
    ``reground`` keys, unchanged in meaning and shape. A rig-built session without it has no ``grounding`` key."""
    calls = build(tmp_path / "cli", capsys, monkeypatch, "live", "--visual-tokens", "560")
    given = grounding_kw(calls)
    assert given["visual_tokens"] == 560 and given["visual_tokens_from"] == "flag", given
    from local_astra.conductor import session as session_mod

    # the real class: ``cli.Session`` is the rig's spy here, whose ``__init__(*a, **k)`` hides the keywords
    params = inspect.signature(session_mod.Session.__init__).parameters
    assert "grounding" in params, (
        "T-0132 A4c: Session(..., grounding: Mapping[str, Any] | None = None)"
    )
    monkeypatch.undo()
    extras = asyncio.run(_rig_turn(tmp_path / "with", visual_tokens=560, grounding=dict(given)))
    assert extras.get("grounding") == given, (
        f"T-0132 A4c: the turn line carries the grounding dict as given; extras {extras}"
    )
    assert extras.get("visual_tokens") == 560, extras
    assert set(extras.get("reground", {})) == {"requests", "ok", "fallback", "skipped", "ms"}, (
        extras
    )
    plain = asyncio.run(_rig_turn(tmp_path / "without", visual_tokens=560))
    assert "grounding" not in plain, f"T-0132 A4c: no grounding key without the keyword; {plain}"
    assert plain.get("visual_tokens") == 560 and "reground" in plain, plain


# --------------------------------------------------------------------------------------------- A3: on the wire


def test_body_carries_1120_by_default(tmp_path, capsys, monkeypatch):
    """A3: the client a default live session holds is the real ``VllmClient`` over
    ``httpx.MockTransport``; its prefill, tools-off generate, tools-on prefill and step, continuation and CD repair
    bodies each carry ``"mm_processor_kwargs": {"max_soft_tokens": 1120}`` (read off the wire)."""
    pytest.importorskip("httpx")
    from local_astra.conductor import llm_client
    from local_astra.contracts.llm import LlmOptions
    from local_astra.toolcall.detok import HostDetokenizer
    from tests.unit import _llm_id_support as S
    from tests.unit import test_llm_client_visual_tokens as V

    sent: list[dict[str, Any]] = []
    made: list[Any] = []
    real = llm_client.VllmClient

    def vllm(*a: Any, **k: Any) -> Any:
        client = real(*a, transport=V.serve(sent), **k)
        made.append(client)
        return client

    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    monkeypatch.setattr(cli, "VllmClient", vllm)
    T.stop_before_serving(monkeypatch, calls)
    monkeypatch.setattr(HostDetokenizer, "load", classmethod(lambda cls: S.detok()))
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *MODE_FLAGS["live"]))
    out, err = _out(capsys)
    assert code == 0 and calls.graphs, f"{code}\n{out}{err}"
    client = calls.graphs[-1].session.llm
    assert made and client is made[0], "premise: the session holds the client cli built"

    async def phases() -> list[str]:
        names: list[str] = []
        off = V.frame_prompt()
        on = V.frame_prompt(tools=S.TOOLS)

        async def gen(name: str, prompt: Any, **kw: Any) -> None:
            names.append(name)
            try:
                async for _ in client.generate(prompt, f"r-{name}", **kw):
                    pass
            except Exception:  # the body is what is read; the stream's fate is not this row's
                pass

        names.append("prefill")
        await client.prefill(off.prefix(), "r-prefill")
        await gen("generate", off, options=LlmOptions(max_tokens=64))
        names.append("prefill_tools")
        await client.prefill(on.prefix(), "r-prefill_tools")
        await gen("tool_step", on)
        await gen("continuation", S.continuation_prompt("I will click it."))
        await gen("repair", S.continuation_prompt("I will click it."), cd=V.CD)
        await client.aclose()
        return names

    names = asyncio.run(phases())
    assert len(sent) == len(names), f"premise: one body per phase; {len(sent)} for {names}"
    for name, body in zip(names, sent, strict=True):
        assert body.get(V.KEY) == {"max_soft_tokens": 1120}, (
            f"T-0132 A3: the default live {name} body carries max_soft_tokens 1120; it has {body.get(V.KEY)!r}"
        )


# --------------------------------------------------------------------------------------------- A7: docs/02

#: T-0132's BASE (docs/tasks/T-0132.md, "from main at a2d9ac6"); a literal, never a moving ref
DOC_BASE_SHA = "a2d9ac660d6de3146006b15cdeae50d45d97c8ba"
#: Post-merge conversion (T-0132 merged at b3f5613): docs/02 as the implementation left it, read at a literal. 5e6b78e is
#: group B's docs clause, the last T-0132 commit to touch docs/02; ``git diff 5e6b78e b3f5613 -- docs/02-architecture.md``
#: is empty. Never the working tree, so later tasks' docs/02 edits cannot redden this row.
DOC_REV = "5e6b78e28fd0a9a6b20bd4d563579307dd86a1f1"
HEADER = "**Spec delta (T-0132"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0132():
    """A7: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0132``, naming the live default (1120,
    ReGround) and its precedence (flag, profile, default), the ``grounding`` line and turn-line key, the pass-2 budget
    rule (``reground_visual_tokens``, one image) and A6's findings (``max_prompt_tokens``, ``MAX_CONTINUATION``);
    otherwise docs/02 equals BASE a2d9ac6's, paragraph for paragraph. Keywords, never wording. Both sides are fixed
    revisions: BASE a2d9ac6 and ``DOC_REV`` 5e6b78e (post-merge conversion, as the T-0131 docs row's ``DOC_REV``)."""
    gr.require_commits(DOC_BASE_SHA, DOC_REV)
    base_doc = _doc02_at(DOC_BASE_SHA)
    doc = _doc02_at(DOC_REV)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0132 A7: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "1120": "1120" in para,
        "ReGround": "reground" in low,
        "precedence": all(w in low for w in ("flag", "profile", "default")),
        "grounding line/key": "grounding" in low,
        "turn line": "turn line" in low or "turn-line" in low,
        "reground_visual_tokens": "reground_visual_tokens" in para,
        "one image": "one image" in low or "single image" in low or "one `imagepart`" in low,
        "max_prompt_tokens": "max_prompt_tokens" in para,
        "MAX_CONTINUATION": "MAX_CONTINUATION" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0132 A7: the docs/02 delta does not record {missing}"
    # remove the delta exactly once, by value (the two splits make different string objects); a duplicate of it
    # anywhere, or any other edited paragraph, still fails the equality below
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0132 A7: docs/02 changes only by the one delta paragraph"
    )
