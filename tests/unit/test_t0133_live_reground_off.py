"""T-0133: live mode defaults to ReGround off (1120 kept), and the pass-2 (crop) budget defaults to 280 when ReGround
is turned on at a resolved 1120 main budget (docs/tasks/T-0133.md, A1-A8; H-T133-1 approved, so A5-approved only).

**The interfaces these rows pin** (the task's; readings in the test-author report):

- ``config.LIVE_DEFAULT_REGROUND is False``, ``config.LIVE_DEFAULT_VISUAL_TOKENS == 1120`` and
  ``config.LIVE_DEFAULT_REGROUND_VISUAL_TOKENS == 280`` (A1); ``Profile``'s fields equal BASE c29a8d9's.
- ``cli`` resolves ReGround in live as flag, else profile, else ``False`` from ``"default"`` (A2). With ReGround on,
  no crop budget set and a resolved main budget of exactly 1120, the crop resolves to 280 from ``"default"``; in every
  other case it follows the main budget (A5-approved, ruling R-i). The second client is T-0132 B2's: built only when
  the two budgets differ, after the main client.
- A crop budget set (flag or profile) with ReGround off in live exits 2 naming ReGround and builds no graph (A6,
  ruling R-h).
- ``Session`` (frozen) is unchanged: these rows read the keywords ``cli`` passes and, for the wire, what a real
  ``VllmClient`` sends.

The rigs are T-0131's, T-0132's and T-0111g's (``test_cli_t0131``, ``test_cli_t0132`` with its ``Built`` spy,
``test_cli_tools_live``, ``test_t0132_live_default``, ``test_session_reground``), imported, not copied.

Activated at write time (single group, no ``pending`` marker): the behaviour rows fail at BASE c29a8d9 on a named
``AssertionError``. ``test_frozen_paths_over_closed_range`` is green at BASE (``TIP_IMPL`` is bumped after each
implementation commit). In the parametrized adversarial rows the cases whose expected value equals BASE's (an explicit
crop budget, a main budget of 280 or 560, an explicit ``--no-reground``/``reground = false``) are A8's identity
cases: green at BASE by design, there to redden mutants 2-4.

Regime: CPU over fakes and ``httpx.MockTransport``; ``ScriptedClock`` for the click turn; thinking on; nothing served;
n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests import _git_range as gr
from tests.unit import _t0160_pins as t0160
from tests.unit import _tools_switch_support as T
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit._t0202_pins import admitted_tools_lines as tools_lines  # T-0202 D3 re-pin
from tests.unit.test_cli_session import CLI, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _names, _toml, build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_session_reground import (
    OK_PASS2,
    P1,
    inv_of,
    one_click_turn,
    reground_requests,
    regrounds,
)
from tests.unit.test_t0132_live_default import (
    PREFIX,
    build_out,
    grounding_kw,
    grounding_lines,
)

FLAG = "--reground-visual-tokens"
LIVE = MODE_FLAGS["live"]
#: T-0133's BASE (docs/tasks/T-0133.md, "from main at c29a8d9"); a literal, never a moving ref
BASE_SHA = "c29a8d92d7859323e4e45c75eb744cc53639de97"
#: The implementation tip: 2720e42 (branch ``T-0133-live-reground-off-impl``; was 5c46f35, the plan). Bumped by the
#: test-author after each implementation commit; never the working tree.
TIP_IMPL = "2720e42c26178ca9e99b0c5d22ffe8b42fd8b89b"
HEADER = "**Spec delta (T-0133"
#: docs/02 as the implementation left it, read at a literal: 2720e42 is the T-0133 implementation commit, the only
#: one to touch docs/02 (``git diff 2720e42 a4f5676 -- docs/02-architecture.md`` is empty). Never the working tree,
#: so later tasks' docs/02 edits cannot redden this row.
DOC_REV = "2720e42c26178ca9e99b0c5d22ffe8b42fd8b89b"
#: A4a, written out from the criterion
DEFAULT_LINE = PREFIX + "visual_tokens=1120 (default), reground=off (default)"
#: A2/A4c with no flag and no profile key; ``reground_visual_tokens`` is not pinned with ReGround off (O-1)
DEFAULT_SUBSET = {
    "visual_tokens": 1120,
    "visual_tokens_from": "default",
    "reground": False,
    "reground_from": "default",
}


def main_client(calls: T.Calls, built: Any) -> tuple[dict[str, Any], Any]:
    """The session's main client (its keywords and object): ``built.made[0]``, proven to be ``session.llm``."""
    assert built.made, "premise: cli built a VllmClient"
    _, kw, client = built.made[0]
    assert calls.graphs[-1].session.llm is client, (
        "premise (T-0133 landmine): the first VllmClient cli builds is the session's main client"
    )
    return kw, client


# --------------------------------------------------------------------------------------------- A1: constants


def _base_config_module(tmp_path: Path) -> Any:
    """``local_astra/config.py`` as it was at BASE c29a8d9, imported under a private name (its only local import,
    ``local_astra.contracts.base``, is a frozen path)."""
    gr.require_commits(BASE_SHA)
    proc = gr._git("show", f"{BASE_SHA}:local_astra/config.py")
    assert proc.returncode == 0, proc.stderr
    path = tmp_path / "t0133_base_config.py"
    path.write_text(proc.stdout)
    spec = importlib.util.spec_from_file_location("t0133_base_config", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t0133_base_config"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("t0133_base_config", None)
    return mod


def _fields(model: Any, module_name: str) -> dict[str, tuple[str, str, bool]]:
    out: dict[str, tuple[str, str, bool]] = {}
    for name, f in model.model_fields.items():
        ann = repr(f.annotation).replace(module_name + ".", "local_astra.config.")
        out[name] = (ann, repr(f.default), f.is_required())
    return out


def pop_max_tool_steps_web(
    now: dict[str, tuple[str, str, bool]],
) -> dict[str, tuple[str, str, bool]]:
    """T-0178 A7 pin move (A1): ``now`` (``_fields`` of this tree's ``Profile``) must hold exactly
    ``max_tool_steps_web: int = 20`` (not required); it is checked and subtracted, nothing else is admitted."""
    got = now.get("max_tool_steps_web")
    assert got == ("<class 'int'>", "20", False), (
        f"T-0178 A1: Profile.max_tool_steps_web: int = Field(default=20, ge=1); got {got!r}"
    )
    return {k: v for k, v in now.items() if k != "max_tool_steps_web"}


#: T-0182 B pin move (B-R1/B-R3): the two Profile fields group B adds, exactly
CLIP_POLICY_FIELDS: dict[str, tuple[str, str, bool]] = {
    "filler_clips": ("<class 'bool'>", "False", False),
    "still_working_after_s": ("<class 'int'>", "20", False),
}


def pop_clip_policy_fields(
    now: dict[str, tuple[str, str, bool]],
) -> dict[str, tuple[str, str, bool]]:
    """T-0182 B pin move: ``now`` (``_fields`` of this tree's ``Profile``) must hold exactly
    ``filler_clips: bool = False`` and ``still_working_after_s: int = 20`` (neither required); both are checked and
    subtracted, nothing else is admitted."""
    got = {k: now.get(k) for k in CLIP_POLICY_FIELDS}
    assert got == CLIP_POLICY_FIELDS, (
        f"T-0182 B: Profile.filler_clips: bool = False and Profile.still_working_after_s: int = 20; got {got!r}"
    )
    return {k: v for k, v in now.items() if k not in CLIP_POLICY_FIELDS}


#: T-0227 R-2 re-pin (docs/tasks/T-0227.md A-1): the two frame settings, ``int | None = None`` each (typed from A-1)
FRAME_SETTING_FIELDS: dict[str, tuple[str, str, bool]] = {
    "max_frames_per_turn": ("int | None", "None", False),
    "frame_interval_ms": ("int | None", "None", False),
}


def pop_frame_setting_fields(
    now: dict[str, tuple[str, str, bool]],
) -> dict[str, tuple[str, str, bool]]:
    """T-0227 R-2 re-pin: ``now`` (``_fields`` of this tree's ``Profile``) must hold exactly
    ``max_frames_per_turn: int | None = None`` and ``frame_interval_ms: int | None = None`` (neither required); both
    are checked and subtracted, nothing else is admitted."""
    got = {k: now.get(k) for k in FRAME_SETTING_FIELDS}
    assert got == FRAME_SETTING_FIELDS, (
        f"T-0227 A-1: Profile.max_frames_per_turn / frame_interval_ms: int | None = None; got {got!r}"
    )
    return {k: v for k, v in now.items() if k not in FRAME_SETTING_FIELDS}


def test_constants(tmp_path):
    """A1: ``LIVE_DEFAULT_REGROUND`` is ``False``, ``LIVE_DEFAULT_VISUAL_TOKENS`` stays 1120,
    ``LIVE_DEFAULT_REGROUND_VISUAL_TOKENS`` is the int 280; ``Profile``'s field set, types and defaults equal BASE
    c29a8d9's (``reground: bool | None = None`` among them), and the comment above ``Profile.reground`` no longer
    says "on in live"."""
    assert config.LIVE_DEFAULT_VISUAL_TOKENS == 1120, (
        "T-0133 A1: the live budget default stays 1120"
    )
    assert config.LIVE_DEFAULT_REGROUND is False, (
        f"T-0133 A1: LIVE_DEFAULT_REGROUND is False; it is {config.LIVE_DEFAULT_REGROUND!r}"
    )
    crop = getattr(config, "LIVE_DEFAULT_REGROUND_VISUAL_TOKENS", None)
    assert type(crop) is int and crop == 280, (
        f"T-0133 A1 (H-T133-1 approved): LIVE_DEFAULT_REGROUND_VISUAL_TOKENS: int = 280; got {crop!r}"
    )
    base = _base_config_module(tmp_path)
    # T-0134c re-scope R3 (orchestrator ruling 2026-10-23): T-0134c A1 adds ``live_feedback`` and ``action_echo``;
    # they are subtracted here, every other field stays exact (T-0134c's own row pins the two). T-0135b re-scope R3
    # (docs/tasks/T-0135b.md A1): ``thought_call_recovery`` is subtracted the same way (T-0135b's own row pins it)
    # T-0140 re-pin (O-A2): A5 adds ``page_overview``; it must be present and is subtracted, nothing else
    assert "page_overview" in Profile.model_fields, (
        "T-0140 A5: Profile.page_overview does not exist yet"
    )
    now = {
        k: v
        for k, v in _fields(Profile, "local_astra.config").items()
        # T-0150 re-pin R3 (docs/tasks/T-0150.md Q-2, A6): ``auto_overview`` is subtracted too (green before and after)
        if k
        not in (
            "live_feedback",
            "action_echo",
            "thought_call_recovery",
            "page_overview",
            "auto_overview",
            # T-0151 re-pin R3 (docs/tasks/T-0151.md Q-2, A2-9): ``overview_frame`` too (green before and after)
            "overview_frame",
            # T-0136 re-pin (O-C1, docs/tasks/T-0136.md C-1, Q-2): ``action_outcome`` too (green before and after)
            "action_outcome",
            "repeat_guard",  # T-0137 re-pin (O-B2): ``repeat_guard`` too (green before and after)
            "search",  # T-0125 pin move (Pins, amendment 2): ``search`` too; checked below
            # T-0160 re-pin (C-1): ``browser_web`` too (green before and after; pinned by test_cli_tools_web.py)
            "browser_web",
        )
    }
    assert Profile.model_fields["search"].default is None, (
        "T-0125 pin move: Profile.search defaults to None"
    )
    from tests.unit import _t0172_pins as t0172

    # T-0172 R4 pin move (B9): exactly announced_call_reask: bool | None = None is added, checked and subtracted
    now = t0172.pop_reask_field(now)
    # T-0174 R-A4 pin move (A3): exactly live_guidance: str | None = None is added, checked and subtracted
    now = t0172.pop_live_guidance_field(now)
    now = pop_max_tool_steps_web(now)  # T-0178 A7 pin move
    now = pop_clip_policy_fields(now)  # T-0182 B pin move
    now = pop_frame_setting_fields(now)  # T-0227 R-2 re-pin
    assert now == _fields(base.Profile, "t0133_base_config"), (
        "T-0133 A1: Profile's field set, types and defaults are unchanged from BASE c29a8d9"
    )
    assert Profile.model_fields["reground"].default is None
    src = Path(config.__file__).read_text().splitlines()
    (at,) = [i for i, line in enumerate(src) if line.strip().startswith("reground: bool | None")]
    comment: list[str] = []
    i = at - 1
    while i >= 0 and src[i].strip().startswith("#"):
        comment.insert(0, src[i].strip())
        i -= 1
    text = " ".join(comment)
    assert comment, "premise: Profile.reground has a comment above it"
    assert "on in live" not in text, (
        f"T-0133 A1: the Profile.reground comment still says 'on in live': {text}"
    )


# --------------------------------------------------------------------------------------------- A2/A3/A4c: the default


def test_live_default_reground_off(tmp_path, capsys, monkeypatch):
    """A2/A3/A4c: ``--tools live`` with no grounding flag and no grounding profile key builds exactly one
    ``VllmClient(visual_tokens=1120)`` and ``Session(..., visual_tokens=1120)`` with ``reground`` falsy and no
    ``reground_llm`` or ``reground_visual_tokens`` keyword; the ``grounding`` dict says 1120 and ReGround off, both
    from the default."""
    calls, built = build(tmp_path, capsys, monkeypatch)
    assert len(built.made) == 1, (
        f"T-0133 A3: one VllmClient by default; built {[k for _, k, _ in built.made]}"
    )
    kw_client, _ = main_client(calls, built)
    assert kw_client.get("visual_tokens") == 1120, kw_client
    kw = calls.session_kwargs[-1]
    assert kw.get("visual_tokens") == 1120, kw
    assert not kw.get("reground", False), (
        f"T-0133 A2/A3: live ReGround defaults off; Session got reground={kw.get('reground')!r}"
    )
    extra = sorted(k for k in ("reground_llm", "reground_visual_tokens") if k in kw)
    assert extra == [], f"T-0133 A3: no {extra} keyword without ReGround (T-0132 B2 refuses them)"
    g = grounding_kw(calls)
    got = {k: g[k] for k in DEFAULT_SUBSET}
    assert got == DEFAULT_SUBSET, f"T-0133 A4c: the default grounding; got {got}"


def test_default_click_sends_no_reground(tmp_path, capsys, monkeypatch):
    """A3: on a default live session's click no ``tool.reground`` record, no ``reground`` request phase and no
    pass-2 request is made, and pass 1's point executes. The session is built with exactly the ``reground`` keyword
    ``cli`` passes by default (absent: no keyword)."""
    calls, _ = build(tmp_path / "cli", capsys, monkeypatch)
    kw = calls.session_kwargs[-1]
    monkeypatch.undo()
    reground = kw.get("reground")  # absent: the rig passes no keyword either
    rig, browser, extras = asyncio.run(
        one_click_turn(tmp_path / "turn", [OK_PASS2], reground=reground)
    )
    assert rig.llm.pass2_prompts == [], (
        f"T-0133 A3: a default live click sent pass 2 (cli passed reground={reground!r})"
    )
    assert regrounds(rig) == [] and reground_requests(rig) == [], "T-0133 A3: no tool.reground"
    assert "reground" not in extras, (
        f"T-0131 B10: no reground turn-line key with ReGround off; {extras}"
    )
    assert dict(inv_of(browser, "t1-call").arguments) == P1, "pass 1's point executes"


# T-0160 ruling 2: web on by default in live (egress=web)
def test_default_grounding_and_tools_lines(tmp_path, capsys, monkeypatch):
    """A4a/A4b: the default live ``--dry-run`` prints exactly the A4a ``grounding`` line after the tools line, and the
    tools line is BASE's plus `` visual_tokens=1120`` (no `` reground=on``); with ``--reground`` it is BASE's plus
    `` visual_tokens=1120 reground=on``. At a real start the A4a line comes before the regime line."""
    base = t0160.web_line(BASE_LINES["live"])  # T-0160 pin move (C-2)
    out = dry_run(tmp_path / "dry", capsys, monkeypatch, "live")
    assert tools_lines(out) == [base + " visual_tokens=1120"], f"T-0133 A4b: printed\n{out}"
    assert grounding_lines(out) == [DEFAULT_LINE], f"T-0133 A4a: printed\n{out}"
    lines = out.splitlines()
    (tools_at,) = [i for i, line in enumerate(lines) if line.startswith("tools=live ")]
    assert lines.index(DEFAULT_LINE) > tools_at, out
    out = dry_run(tmp_path / "dry-on", capsys, monkeypatch, "live", "--reground")
    assert tools_lines(out) == [base + " visual_tokens=1120 reground=on"], (
        f"T-0133 A4b: --reground; printed\n{out}"
    )
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "start", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    lines = started.splitlines()
    assert grounding_lines(started) == [DEFAULT_LINE], f"T-0133 A4a at a real start:\n{started}"
    regime = [i for i, line in enumerate(lines) if line.startswith("astra session: regime:")]
    assert regime and lines.index(DEFAULT_LINE) < regime[0], started


# --------------------------------------------------------------------------------------------- A2/A5: opting in

#: (flags, profile body, expected reground_from)
OPT_IN = {
    "flag": (["--reground"], "", "flag"),
    "profile-true": ([], 'tools = "live"\nreground = true\n', "profile"),
    "flag-over-profile-false": (["--reground"], "reground = false\n", "flag"),
}


@pytest.mark.parametrize("case", list(OPT_IN))
def test_reground_opt_in_sources(tmp_path, capsys, monkeypatch, case):
    """A2 (with A5-approved): ``--reground``, profile ``reground = true`` and ``--reground`` over profile
    ``reground = false`` turn ReGround on from ``flag``, ``profile`` and ``flag``; the main client stays 1120 and
    the crop is 280 from the default (main budget 1120, no crop budget set)."""
    flags, body, src = OPT_IN[case]
    calls, built = build(tmp_path, capsys, monkeypatch, *flags, body=body)
    kw_client, _ = main_client(calls, built)
    assert kw_client.get("visual_tokens") == 1120, kw_client
    kw = calls.session_kwargs[-1]
    assert kw.get("reground") is True, f"T-0133 A2 {case}: ReGround on; {kw}"
    g = grounding_kw(calls)
    assert (g["reground"], g["reground_from"]) == (True, src), f"T-0133 A2 {case}: {g}"
    assert (g["reground_visual_tokens"], g["reground_visual_tokens_from"]) == (280, "default"), (
        f"T-0133 A5 {case}: the crop defaults to 280 at a 1120 main budget; {g}"
    )
    assert kw.get("reground_visual_tokens") == 280, kw


#: (flags, profile body, main budget, crop, crop source, clients) -- the A5 table, written out from the criterion
SCOPE = {
    "reground": (["--reground"], "", 1120, 280, "default", 2),
    "flag-1120-reground": (["--visual-tokens", "1120", "--reground"], "", 1120, 280, "default", 2),
    "flag-560-reground": (["--visual-tokens", "560", "--reground"], "", 560, 560, "default", 1),
    "flag-280-reground": (["--visual-tokens", "280", "--reground"], "", 280, 280, "default", 1),
    "profile-1120-reground": (
        [],
        'tools = "live"\nvisual_tokens = 1120\nreground = true\n',
        1120,
        280,
        "default",
        2,
    ),
    "crop-flag-1120": (["--reground", FLAG, "1120"], "", 1120, 1120, "flag", 1),
    "crop-profile-1120": (
        [],
        'tools = "live"\nreground = true\nreground_visual_tokens = 1120\n',
        1120,
        1120,
        "profile",
        1,
    ),
}


@pytest.mark.parametrize("case", list(SCOPE))
def test_crop_default_scope(tmp_path, capsys, monkeypatch, case):
    """A5-approved (adversarial; mutants: the 280 default at any main budget, and the default keyed on
    ``visual_tokens_from == "default"`` instead of the resolved 1120): the crop is 280 from the default exactly when
    live, ReGround on, no crop budget set and the resolved main budget is 1120 (from the default, a flag or the
    profile); at 560 or 280 it follows the main budget; an explicit 1120 crop is the opt-out. Each case checks the
    client count, every client's budget, the session's ``reground_llm``/``reground_visual_tokens`` and the crop's
    source; two clients share the endpoint and the detokeniser object."""
    flags, body, main, crop, src, n = SCOPE[case]
    calls, built = build(tmp_path, capsys, monkeypatch, *flags, body=body)
    budgets = [k.get("visual_tokens") for _, k, _ in built.made]
    want = [main] if n == 1 else [main, crop]
    assert budgets == want, f"T-0133 A5 {case}: clients' budgets {budgets}, want {want}"
    kw_main, c_main = main_client(calls, built)
    kw = calls.session_kwargs[-1]
    assert kw.get("visual_tokens") == main and kw.get("reground") is True, kw
    assert kw.get("reground_visual_tokens") == crop, f"T-0133 A5 {case}: {kw}"
    if n == 1:
        assert kw.get("reground_llm") is c_main, f"T-0133 A5 {case}: the main client is pass 2's"
    else:
        a1, a2 = built.made[0][0], built.made[1][0]
        kw_crop, c_crop = built.made[1][1], built.made[1][2]
        assert kw.get("reground_llm") is c_crop, (
            f"T-0133 A5 {case} (mutation 8): reground_llm is the {crop} client, not the main one"
        )
        assert kw_crop.get("detokenizer") is kw_main.get("detokenizer") is not None
        ep1 = a1[0] if a1 else kw_main.get("endpoint")
        ep2 = a2[0] if a2 else kw_crop.get("endpoint")
        assert ep1 == ep2 and ep1 is not None, (ep1, ep2)
    g = grounding_kw(calls)
    got = (g["reground_visual_tokens"], g["reground_visual_tokens_from"])
    assert got == (crop, src), (
        f"T-0133 A5 {case}: the crop and its source {got}, want {(crop, src)}"
    )


#: (flags, profile body, expected reground_from)
OFF = {
    "default": ([], "", "default"),
    "no-reground": (["--no-reground"], "", "flag"),
    "profile-false": ([], "reground = false\n", "profile"),
}


@pytest.mark.parametrize("case", list(OFF))
def test_no_crop_client_when_reground_off(tmp_path, capsys, monkeypatch, case):
    """Adversarial (mutation 4: a second client built from the crop default with ReGround off, leaking a
    connection pool): with no flag, ``--no-reground`` or profile ``reground = false`` exactly one ``VllmClient``
    (1120) is constructed and ReGround is off from the expected source."""
    flags, body, src = OFF[case]
    calls, built = build(tmp_path, capsys, monkeypatch, *flags, body=body)
    assert [k.get("visual_tokens") for _, k, _ in built.made] == [1120], (
        f"T-0133 {case}: exactly one VllmClient(1120); built {[k for _, k, _ in built.made]}"
    )
    g = grounding_kw(calls)
    assert (g["reground"], g["reground_from"]) == (False, src), f"T-0133 A2 {case}: {g}"
    kw = calls.session_kwargs[-1]
    assert not kw.get("reground", False) and "reground_llm" not in kw, kw


def test_crop_client_body_carries_280(tmp_path, capsys, monkeypatch):
    """A5-approved on the wire: in a cli-built ``--reground`` live session over real ``VllmClient`` s on
    ``httpx.MockTransport``, a generate through ``session.reground_llm`` sends ``{"max_soft_tokens": 280}`` and one
    through ``session.llm`` sends 1120 (cli passes the budget to the crop client and the client sends it)."""
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
    code = run_cli(argv(tmp_path, *LIVE, "--reground"))
    out, err = _out(capsys)
    assert code == 0 and calls.graphs, f"{code}\n{out}{err}"
    session = calls.graphs[-1].session
    assert len(made) == 2, (
        f"T-0133 A5: --reground alone builds a second (crop) client; built {len(made)}"
    )
    assert session.llm is made[0] and session.reground_llm is made[1], (
        "T-0133 A5: the session holds the main client as llm and the crop client as reground_llm"
    )

    async def one(client: Any, rid: str) -> None:
        try:
            async for _ in client.generate(
                V.frame_prompt(), rid, options=LlmOptions(max_tokens=64)
            ):
                pass
        except Exception:  # the body is what is read; the stream's fate is not this row's
            pass

    async def both() -> None:
        await one(session.reground_llm, "r-crop")
        await one(session.llm, "r-main")
        for c in made:
            await c.aclose()

    asyncio.run(both())
    assert len(sent) == 2, f"premise: one body per generate; {len(sent)}"
    assert sent[0].get(V.KEY) == {"max_soft_tokens": 280}, (
        f"T-0133 A5: the crop client's body carries 280; it has {sent[0].get(V.KEY)!r}"
    )
    assert sent[1].get(V.KEY) == {"max_soft_tokens": 1120}, (
        f"T-0133 A3: the main client's body carries 1120; it has {sent[1].get(V.KEY)!r}"
    )


# --------------------------------------------------------------------------------------------- A6: refused

#: (flags, profile body)
CROP_WITHOUT_REGROUND = {
    "flag": ([*LIVE, FLAG, "280"], ""),
    "profile": ([], 'tools = "live"\nbrowser = "headless"\nreground_visual_tokens = 280\n'),
}


@pytest.mark.parametrize("case", list(CROP_WITHOUT_REGROUND))
def test_crop_budget_without_reground_refused(tmp_path, capsys, monkeypatch, case):
    """A6 (R-h; mutation 5: a crop budget implicitly turning ReGround on): in live, ``--reground-visual-tokens 280``
    alone, and a ``tools = "live"`` profile with ``reground_visual_tokens = 280`` and no ``reground`` key (which still
    loads, O-7), each exit 2 naming ReGround and build no graph, with ``--dry-run`` and without."""
    flags, body = CROP_WITHOUT_REGROUND[case]
    if body:
        got = load(_toml(tmp_path, body)).active()
        assert got.reground_visual_tokens == 280 and got.reground is None, (  # type: ignore[attr-defined]
            "premise (O-7): the live profile with a crop budget and no reground key loads"
        )
    for extra in (["--dry-run"], []):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *flags, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and _names("reground", out + err), (
            f"T-0133 A6 {case} {extra}: exit 2 naming ReGround; got {code}\n{out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, f"{case}: a refused run built a graph"
        monkeypatch.undo()


# --------------------------------------------------------------------------------------------- A7: docs and help


def _help_of(flag: str) -> str:
    """The ``help=`` text of ``add_argument(flag, ...)`` in cli.py, read statically (string constants joined)."""
    for node in ast.walk(ast.parse(CLI.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if flag not in names:
            continue
        for k in node.keywords:
            if k.arg == "help":
                parts = [n.value for n in ast.walk(k.value) if isinstance(n, ast.Constant)]
                return "".join(str(p) for p in parts)
    raise AssertionError(f"premise: cli.py has add_argument({flag!r}, help=...)")


def test_help_texts_state_new_defaults():
    """A7: the ``--reground`` help says the live default is off (and no longer "else on"); the
    ``--reground-visual-tokens`` help names the 280 default and the 1120 main budget it is bound to. Keywords, never
    wording."""
    rg = _help_of("--reground")
    assert re.search(r"\boff\b", rg) and "else on" not in rg, (
        f"T-0133 A7: the --reground help states the off default; it reads: {rg}"
    )
    crop = _help_of(FLAG)
    assert "280" in crop and "1120" in crop, (
        f"T-0133 A7: the {FLAG} help states the 280 default at 1120; it reads: {crop}"
    )


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0133():
    """A7: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0133``, naming
    ``LIVE_DEFAULT_REGROUND``, off, ``--reground``, that it supersedes T-0132's clause, and (A5-approved)
    ``LIVE_DEFAULT_REGROUND_VISUAL_TOKENS``, 280 and the one-image rule; removing it once leaves docs/02 equal to BASE
    c29a8d9's paragraph for paragraph (so the T-0132 delta stays verbatim). Keywords, never wording. Both sides are
    fixed revisions: BASE c29a8d9 and ``DOC_REV`` 2720e42 (converted before the merge, as the T-0132 docs row's
    ``DOC_REV``), never the working tree."""
    gr.require_commits(BASE_SHA, DOC_REV)
    base_doc = _doc02_at(BASE_SHA)
    doc = _doc02_at(DOC_REV)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0133 A7: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    para = paras[0]
    low = para.lower()
    need = {
        "LIVE_DEFAULT_REGROUND": re.search(r"LIVE_DEFAULT_REGROUND(?![A-Z_])", para) is not None,
        "off": re.search(r"\boff\b", low) is not None,
        "--reground": re.search(r"--reground(?![-\w])", para) is not None,
        "supersede": "supersede" in low,
        "T-0132": "T-0132" in para,
        "LIVE_DEFAULT_REGROUND_VISUAL_TOKENS": "LIVE_DEFAULT_REGROUND_VISUAL_TOKENS" in para,
        "280": "280" in para,
        "one image": "one image" in low or "single image" in low or "one `imagepart`" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0133 A7: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0133 A7: docs/02 changes only by the one delta paragraph (the T-0132 delta stays verbatim)"
    )


# --------------------------------------------------------------------------------------------- A8: frozen paths

#: A8's list, written out from the criterion
FROZEN = (
    "local_astra/conductor/session.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/perception/policy.py",
    "local_astra/preflight.py",
    "services",
    "scripts/gpu",
    "scripts/locked",
    "local_astra/computer",
    "local_astra/pages",
    "hud",
)


def test_frozen_paths_over_closed_range():
    """A8 (green at BASE): A8's paths are byte-identical over ``BASE_SHA..TIP_IMPL`` (two literals, never the
    working tree), with no file added or removed. Power: the same predicate sees T-0111g's ``versions.lock``
    change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE_SHA, TIP_IMPL, *FROZEN, diff_filter="ADR") == []
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0133 A8: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert "services/versions.lock" in gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN), "power"
