"""T-0132 group B (cli and config side): ``Profile.reground_visual_tokens``, ``--reground-visual-tokens``, the second
client and the ``grounding`` line's pass-2 item (docs/tasks/T-0132.md, B1, B2).

**The interfaces these rows pin** (the task's; readings in the test-author report):

- ``Profile.reground_visual_tokens: Literal[280, 560, 1120] | None = None``; junk refused at load naming 280, 560 and
  1120; refused at load with ``tools != "live"`` (naming ``live``) and with an explicit ``reground = false`` (naming
  ``reground``).
- ``astra session --reground-visual-tokens {280,560,1120}``; anything else exits 2. ``cli`` exits 2 when the resolved
  value is set while the resolved ReGround is off (the message names ``reground``) or the mode is not live (the
  message names ``live``), before any graph is built, with ``--dry-run`` and without.
- ``cli`` passes ``Session(..., reground_llm=<client>, reground_visual_tokens=<N>)`` in live with ReGround on: the
  main client itself when the two budgets are equal (one ``VllmClient`` built); otherwise a second
  ``VllmClient(<same endpoint>, detokenizer=<the same object>, visual_tokens=<pass-2 budget>)``.
- The ``grounding`` line's third item reads ``reground_visual_tokens=<N> (<flag|profile|default>)``.

Group B, activated 2026-10-22 (until then ``pending(task="T-0132")``); at BASE a2d9ac6 each fails on a named ``AssertionError`` (the flag, the
field or the keyword is missing; a missing flag is checked before any exit code is read, because argparse's own
exit 2 would otherwise satisfy the refusal rows).

**T-0133 re-scope R3** (docs/tasks/T-0133.md; human ruling 2026-10-22: live defaults to ReGround *off*; H-T133-1
approved: with ReGround on, no crop budget set and a resolved 1120 main budget the crop defaults to 280): every row
that needs ReGround on now turns it on explicitly (``--reground`` or ``reground = true``); the equal-budget
``default`` case moves to a 560 main budget (where the crop follows it), and ``default-crop-280`` pins the new
default. Red against main until T-0133 is implemented, by design.

Regime: CPU over fakes; nothing served; no measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra import cli
from local_astra.config import Profile, load
from tests.fakes import FakeLlm
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import CLI, explode_clients, flags_of, run_cli
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live

FLAG = "--reground-visual-tokens"
PREFIX = "astra session: grounding: "
ALLOWED = (280, 560, 1120)


def require_flag() -> None:
    assert FLAG in flags_of(CLI), f"T-0132 B1: astra session {FLAG} does not exist yet"


def require_field() -> None:
    assert "reground_visual_tokens" in Profile.model_fields, (
        "T-0132 B1: Profile.reground_visual_tokens does not exist yet"
    )


def _toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\n' + body)
    return p


class Built:
    """Every ``VllmClient`` ``cli`` constructs: positional args, keywords and the object returned."""

    def __init__(self) -> None:
        self.made: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []


def build(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> tuple[T.Calls, Built]:
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    built = Built()

    def vllm(*a: Any, **k: Any) -> Any:
        client = FakeLlm()
        built.made.append((a, dict(k), client))
        calls.vllm_kwargs.append(dict(k))
        return client

    monkeypatch.setattr(cli, "VllmClient", vllm)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *MODE_FLAGS["live"], *flags, config_body=body))
    out, err = _out(capsys)
    assert code == 0 and calls.graphs, f"{flags} {body!r}: exit 0; got {code}\n{out}{err}"
    return calls, built


# --------------------------------------------------------------------------------------------- B1: the flag


@pytest.mark.parametrize("value", ALLOWED)
def test_flag_values_accepted(tmp_path, capsys, monkeypatch, value):
    """B1: each allowed value is accepted in live with ReGround on (by flag, ``--reground``; T-0133 R3) and printed
    as ``(flag)``."""
    require_flag()
    out = dry_run(tmp_path, capsys, monkeypatch, "live", "--reground", FLAG, str(value))
    lines = [line for line in out.splitlines() if line.startswith(PREFIX)]
    assert len(lines) == 1 and lines[0].endswith(f"reground_visual_tokens={value} (flag)"), out


@pytest.mark.parametrize("junk", ["70", "0", "1121", "many", "140"])
def test_flag_junk_exits_2(tmp_path, capsys, monkeypatch, junk):
    """B1: anything but 280, 560 or 1120 exits 2."""
    require_flag()
    T.patch_detok_load(monkeypatch, T.Calls(), tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, *MODE_FLAGS["live"], FLAG, junk), "--dry-run"])
    _out(capsys)
    assert code == 2, f"T-0132 B1: {FLAG} {junk} exits 2; got {code}"


#: (case, flags, profile body, a word the refusal names)
REFUSED = {
    "no-reground-flag": (
        ["--tools", "live", "--browser", "headless", "--no-reground"],
        "",
        "reground",
    ),
    "no-reground-profile": (
        ["--tools", "live", "--browser", "headless"],
        "reground = false\n",
        "reground",
    ),
    "off": ([], "", "live"),
    "dry_run": (["--tools", "dry-run"], "", "live"),
    "live-profile-tools-off": (["--tools", "off"], 'tools = "live"\n', "live"),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_flag_refused_when_reground_off_or_not_live(tmp_path, capsys, monkeypatch, case):
    """B1 (a silent no-op is the failure): the resolved pass-2 budget set while the resolved ReGround is off, or the
    mode is not live, exits 2 with a message naming why, with ``--dry-run`` and without, and builds no graph.
    Too-wide companion: live with ReGround on accepts it (``test_flag_values_accepted``)."""
    require_flag()
    flags, body, why = REFUSED[case]
    for extra in (["--dry-run"], []):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *flags, FLAG, "280", config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and why in (out + err).lower(), (
            f"T-0132 B1 {case} {extra}: exit 2 naming {why!r}; got {code}\n{out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, f"{case}: a refused run built a graph"
        monkeypatch.undo()


# --------------------------------------------------------------------------------------------- B1: the profile key


def test_profile_key_load_refusals(tmp_path):
    """B1: ``reground_visual_tokens`` accepts 280/560/1120 in live (ReGround unset or true); junk is refused at load
    naming the three values; it is refused with ``tools`` other than live (naming ``live``) and with an explicit
    ``reground = false`` (naming ``reground``)."""
    require_field()
    for v in ALLOWED:
        for extra in ("", "reground = true\n"):
            got = load(_toml(tmp_path, f'tools = "live"\n{extra}reground_visual_tokens = {v}\n'))
            assert got.active().reground_visual_tokens == v, (v, extra)  # type: ignore[attr-defined]
    for junk in ("70", "0", "1121", '"280"'):
        with pytest.raises(ValueError) as exc:
            load(_toml(tmp_path, f'tools = "live"\nreground_visual_tokens = {junk}\n'))
        assert all(str(v) in str(exc.value) for v in ALLOWED), (junk, str(exc.value))
    for tools in ("off", "dry_run"):
        with pytest.raises(ValueError) as exc:
            load(_toml(tmp_path, f'tools = "{tools}"\nreground_visual_tokens = 280\n'))
        assert "live" in str(exc.value), str(exc.value)
    with pytest.raises(ValueError) as exc:
        load(_toml(tmp_path, 'tools = "live"\nreground = false\nreground_visual_tokens = 280\n'))
    assert "reground" in str(exc.value).replace("reground_visual_tokens", ""), str(exc.value)
    with pytest.raises(ValidationError):
        Profile(name="lean", tools="off", reground_visual_tokens=280)  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------------- B2: the clients


@pytest.mark.parametrize(
    "flags",
    [
        ["--reground", "--visual-tokens", "560"],
        ["--reground", FLAG, "1120"],
        ["--reground", "--visual-tokens", "560", FLAG, "560"],
    ],
    ids=["default", "flag-equal", "both-560"],
)
def test_equal_budgets_pass_same_client(tmp_path, capsys, monkeypatch, flags):
    """B2 (reviewer mutation 6): with equal budgets ``cli`` builds one ``VllmClient`` and passes it as
    ``reground_llm`` (the session's own client) with ``reground_visual_tokens`` equal to the main budget.

    T-0133 R3: ReGround is turned on by flag in every case; the ``default`` case (the crop follows the main budget)
    is at a 560 main budget, since at 1120 the crop now defaults to 280 (H-T133-1)."""
    if flags:
        require_flag()
    calls, built = build(tmp_path, capsys, monkeypatch, *flags)
    kw = calls.session_kwargs[-1]
    assert "reground_llm" in kw, (
        f"T-0132 B2: Session(..., reground_llm=...) is passed; {sorted(kw)}"
    )
    assert len(built.made) == 1, (
        f"T-0132 B2: one client when the budgets are equal; {len(built.made)}"
    )
    assert kw["reground_llm"] is calls.graphs[-1].session.llm is built.made[0][2]
    assert kw.get("reground_visual_tokens") == kw.get("visual_tokens"), kw


@pytest.mark.parametrize(
    ("flags", "body", "main", "crop"),
    [
        (["--reground", FLAG, "280"], "", 1120, 280),
        (["--reground", "--visual-tokens", "560", FLAG, "1120"], "", 560, 1120),
        ([], 'tools = "live"\nreground = true\nreground_visual_tokens = 280\n', 1120, 280),
        (["--reground"], "", 1120, 280),
    ],
    ids=["flag-280", "flag-1120-over-560", "profile-280", "default-crop-280"],
)
def test_unequal_budgets_build_second_client(
    tmp_path, capsys, monkeypatch, flags, body, main, crop
):
    """B2: unequal budgets build a second ``VllmClient`` with the same endpoint, the same detokeniser object and the
    pass-2 budget; it is the session's ``reground_llm``, not its main client.

    T-0133 R3: ReGround is turned on explicitly in every case (``--reground``, or ``reground = true`` in the
    profile); ``default-crop-280`` is ``--reground`` alone, whose crop defaults to 280 at 1120 (H-T133-1)."""
    require_flag()
    calls, built = build(tmp_path, capsys, monkeypatch, *flags, body=body)
    assert len(built.made) == 2, f"T-0132 B2: a second client; built {len(built.made)}"
    (a1, k1, c1), (a2, k2, c2) = built.made
    assert k1.get("visual_tokens") == main and k2.get("visual_tokens") == crop, (k1, k2)
    assert k2.get("detokenizer") is k1.get("detokenizer") and k1.get("detokenizer") is not None
    ep1 = a1[0] if a1 else k1.get("endpoint")
    ep2 = a2[0] if a2 else k2.get("endpoint")
    assert ep1 == ep2 and ep1 is not None, (ep1, ep2)
    kw = calls.session_kwargs[-1]
    assert calls.graphs[-1].session.llm is c1, "premise: the main client is the session's llm"
    assert kw.get("reground_llm") is c2 and kw.get("reground_visual_tokens") == crop, kw


# --------------------------------------------------------------------------------------------- A4a: the line


@pytest.mark.parametrize(
    ("flags", "body", "rest"),
    [
        (
            ["--reground", FLAG, "280"],
            "",
            "visual_tokens=1120 (default), reground=on (flag), reground_visual_tokens=280 (flag)",
        ),
        (
            ["--reground"],
            "reground_visual_tokens = 560\n",
            "visual_tokens=1120 (default), reground=on (flag), reground_visual_tokens=560 (profile)",
        ),
    ],
    ids=["flag", "profile"],
)
def test_grounding_line_shows_pass2_source(tmp_path, capsys, monkeypatch, flags, body, rest):
    """A4a/B1: the ``grounding`` line names the pass-2 budget's source, ``flag`` or ``profile``. T-0133 R3:
    ReGround is turned on by ``--reground``."""
    require_flag()
    out = dry_run(tmp_path, capsys, monkeypatch, "live", *flags, body=body)
    assert [line for line in out.splitlines() if line.startswith(PREFIX)] == [PREFIX + rest], out


# --------------------------------------------------------------------------------------------- review A (group B)


def _names(word: str, text: str) -> bool:
    """``word`` in ``text``, case-insensitive, after removing the key's own spellings (``reground`` is a substring of
    ``reground_visual_tokens``, so a message naming only the key does not name ReGround)."""
    low = text.lower().replace("--reground-visual-tokens", "").replace("reground_visual_tokens", "")
    return word in low


#: (case, flags, a word the refusal names) for a pass-2 budget set in the *profile* (no tools key: load leaves the
#: mode to cli)
PROFILE_REFUSED = {
    "no-tools": ([], "live"),
    "live-no-reground": (["--tools", "live", "--browser", "headless", "--no-reground"], "reground"),
    # T-0133 R3: live with ReGround off by default (no flag, no profile reground key)
    "live-default-off": (["--tools", "live", "--browser", "headless"], "reground"),
}


@pytest.mark.parametrize("case", list(PROFILE_REFUSED))
def test_profile_pass2_budget_refused_by_cli(tmp_path, capsys, monkeypatch, case):
    """B1 (review A, gap 1; mutant: the cli refusal checks only a ``flag`` source): a profile with
    ``reground_visual_tokens = 280`` and no ``tools`` key loads, and ``cli`` refuses it with exit 2, with
    ``--dry-run`` and without, building no graph: run with no ``--tools`` (the resolved mode is off) the message
    names ``live``; with ``--tools live --no-reground`` it names ReGround. T-0133 R3 (``live-default-off``): with
    ``--tools live`` alone ReGround is off by default, so it names ReGround too."""
    require_field()
    body = "reground_visual_tokens = 280\n"
    assert load(_toml(tmp_path, body)).active().reground_visual_tokens == 280, (  # type: ignore[attr-defined]
        "premise: with no tools key the profile loads (the mode is resolved by cli)"
    )
    flags, why = PROFILE_REFUSED[case]
    for extra in (["--dry-run"], []):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *flags, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and _names(why, out + err), (
            f"T-0132 B1 {case} {extra}: exit 2 naming {why!r}; got {code}\n{out}{err}"
        )
        assert not calls.graphs and not calls.session_kwargs, f"{case}: a refused run built a graph"
        monkeypatch.undo()


def test_flag_beats_profile_for_pass2_budget(tmp_path, capsys, monkeypatch):
    """B1 (review A, gap 2; mutant: the flag and profile arms swapped in the resolver): profile
    ``reground_visual_tokens = 560`` with ``--reground-visual-tokens 280`` resolves to 280 from the flag, in the
    ``grounding`` line and in the second client ``cli`` builds. T-0133 R3: ReGround is turned on by ``--reground``
    in both halves."""
    require_flag()
    body = "reground_visual_tokens = 560\n"
    out = dry_run(
        tmp_path / "dry", capsys, monkeypatch, "live", "--reground", FLAG, "280", body=body
    )
    lines = [line for line in out.splitlines() if line.startswith(PREFIX)]
    assert len(lines) == 1 and lines[0].endswith("reground_visual_tokens=280 (flag)"), out
    monkeypatch.undo()
    calls, built = build(
        tmp_path / "build", capsys, monkeypatch, "--reground", FLAG, "280", body=body
    )
    assert len(built.made) == 2, f"a second client; built {len(built.made)}"
    assert built.made[1][1].get("visual_tokens") == 280, built.made[1][1]
    kw = calls.session_kwargs[-1]
    assert kw.get("reground_visual_tokens") == 280 and kw.get("reground_llm") is built.made[1][2], (
        kw
    )
