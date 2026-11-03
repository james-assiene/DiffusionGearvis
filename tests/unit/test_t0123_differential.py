"""T-0123 criterion 5: the tools-off session starts exactly as it did at ``68d5566`` (a differential, not a golden).

Spec (docs/tasks/T-0123.md, Goal and criterion 5): with tools off the loop setting is inert and startup is
byte-identical to ``68d5566``, except for the one refusal text of criterion 4(d) (an explicit ``--max-tool-steps``
with tools off, which used to advise ``--tool-loop``, itself refused there). For every tools-off argv x profile row
of the truth table these equal the base fixture: the exit code, the full ``--dry-run`` stdout (``tools=off``
included), the ``TurnPolicy`` handed to ``TurnMachine``, ``kit is None``, and the stderr of refused rows.

Reading (in the test-author's report): "startup is byte-identical" (the Goal) is read as covering the startup stdout
and stderr of *every* row, started or refused, not only the stderr of refused rows; criterion 5's list is a subset
of that. The 4(d) exemption is the stderr only, and only of the rows where criterion 4(a)/(b) do not also apply:
an explicit loop request with tools off keeps its 68d5566 message ("message unchanged", 4(a)/(b)), with or without
``--max-tool-steps``.

The oracle is ``tests/fixtures/cli/tool_loop_default_base_68d5566/base.json``, generated on CPU by ``emit()`` below,
run from a detached worktree at ``68d5566`` (MANIFEST.md says how). ``observe_*`` is the one code path both the
generator and the rows run, so the fixture and the comparison cannot drift apart.

Regime: CPU over the CLI rig's fakes (``tests/unit/_tools_switch_support.py``); profile ``lean`` from a temporary
``astra.toml``; thinking on; no GPU, no network; no measurement.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tool_loop import _config, _regime_tools

#: The literal commit T-0123 branched its base fixture from (docs/tasks/T-0123.md criterion 5); never a moving ref.
BASE = "68d556645448c79f7f4d32295a7413717231edb2"
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cli"
    / "tool_loop_default_base_68d5566"
    / "base.json"
)
TMP = "<tmp>"

# --- the tools-off half of the truth table's axes (docs/tasks/T-0123.md "Tests to write"), written out by hand ---

#: tools axis, tools-off values only: profile keys, argv
TOOLS_OFF_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "p-off": ({}, ()),
    "f-off": ({"tools": "dry_run"}, ("--tools", "off")),
}
LOOP_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "unset": ({}, ()),
    "p-true": ({"tool_loop": True}, ()),
    "p-false": ({"tool_loop": False}, ()),
    "f-on": ({}, ("--tool-loop",)),
    "f-off": ({}, ("--no-tool-loop",)),
    "p-false+f-on": ({"tool_loop": False}, ("--tool-loop",)),
    "p-true+f-off": ({"tool_loop": True}, ("--no-tool-loop",)),
}
STEPS_AXIS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "none": ({}, ()),
    "f3": ({}, ("--max-tool-steps", "3")),
    "p5": ({"max_tool_steps": 5}, ()),
}
ROWS = [f"{t}/{lp}/{s}" for t in TOOLS_OFF_AXIS for lp in LOOP_AXIS for s in STEPS_AXIS]

#: Criterion 4(d)'s rows, the only tools-off stderr that may differ from 68d5566: an explicit ``--max-tool-steps``
#: with tools off and no explicit loop request (an explicit request is 4(a)/(b), whose message is unchanged).
STDERR_EXEMPT_4D = frozenset(
    {
        "p-off/unset/f3",
        "p-off/p-false/f3",
        "p-off/f-off/f3",
        "p-off/p-true+f-off/f3",
        "f-off/unset/f3",
        "f-off/p-false/f3",
        "f-off/f-off/f3",
        "f-off/p-true+f-off/f3",
    }
)

#: the text fields of a tools-off observation, stored once each in the fixture's ``strings``
INTERNED = ("startup_stdout", "startup_stderr", "policy", "dry_run_stdout", "dry_run_stderr")

#: criterion 6's two tools-on label rows: the default ``--tools dry-run`` start, repair on and repair off
LABEL_KINDS = ("repair_on", "repair_off")


# --- the one observation path (generator and rows) ------------------------------------------------------------------


def _compose(row: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    tools, loop, steps = row.split("/")
    cfg: dict[str, Any] = {}
    argv: tuple[str, ...] = ()
    for axis, key in ((TOOLS_OFF_AXIS, tools), (LOOP_AXIS, loop), (STEPS_AXIS, steps)):
        keys, flags = axis[key]
        cfg.update(keys)
        argv += flags
    return cfg, argv


def _norm(text: str, tmp: Path) -> str:
    return text.replace(str(tmp), TMP)


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_cli(argv)
    return code, out.getvalue(), err.getvalue()


def _startup(tmp: Path, config: Path, argv: tuple[str, ...]) -> tuple[int, str, str, T.Calls]:
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        T.fake_clients(mp, calls)
        T.stop_before_serving(mp, calls)
        T.patch_detok_load(mp, calls, tmp)
        code, out, err = _run(T.session_argv(config, tmp, *argv))
    return code, out, err, calls


def _dry_run(tmp: Path, config: Path, argv: tuple[str, ...]) -> tuple[int, str, str]:
    with pytest.MonkeyPatch.context() as mp:
        explode_clients(mp)
        T.patch_detok_load(mp, T.Calls(), tmp)
        return _run([*T.session_argv(config, tmp), "--dry-run", *argv])


def observe_tools_off_row(row: str, tmp: Path, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """Everything criterion 5 compares for one tools-off row, with the temporary directory written ``<tmp>``.
    ``extra`` is appended to the row's argv (the generator passes none; T-0183's re-scope passes the budget flag)."""
    cfg, argv = _compose(row)
    argv += extra
    config = _config(tmp, **cfg)
    code, out, err, calls = _startup(tmp, config, argv)
    started = bool(calls.graphs)
    policy = repr(calls.graphs[-1].session.machine.policy) if started else None
    kit_is_none = (calls.kit() is None) if calls.session_kwargs else None
    d_code, d_out, d_err = _dry_run(tmp, config, argv)
    return {
        "startup_code": code,
        "startup_stdout": _norm(out, tmp),
        "startup_stderr": _norm(err, tmp),
        "policy": policy,
        "kit_is_none": kit_is_none,
        "dry_run_code": d_code,
        "dry_run_stdout": _norm(d_out, tmp),
        "dry_run_stderr": _norm(d_err, tmp),
    }


def _point_repair_off(mp: pytest.MonkeyPatch, tmp: Path, kind: str) -> None:
    if kind == "repair_off":
        # test_cli_session's repair-off row: a missing payload file turns the repair off with a reason naming it
        T.point_cli_at(mp, payloads=tmp / "nowhere" / "native_T4.json")


def observe_label(kind: str, tmp: Path, *flags: str) -> dict[str, Any]:
    """Criterion 6 for a tools-on start: the policy, the ``Regime.tools`` a turn on that policy and kit writes,
    and the ``--dry-run`` tools line."""
    import asyncio

    config = _config(tmp)
    argv = ("--tools", "dry-run", *flags)
    calls = T.Calls()
    with pytest.MonkeyPatch.context() as mp:
        _point_repair_off(mp, tmp, kind)
        T.fake_clients(mp, calls)
        T.stop_before_serving(mp, calls)
        T.patch_detok_load(mp, calls, tmp)
        code, out, err = _run(T.session_argv(config, tmp, *argv))
        assert code == 0 and calls.graphs, (
            f"premise: the tools-on start runs (exit {code}):\n{out}{err}"
        )
        policy = calls.graphs[-1].session.machine.policy
        kit = calls.kit()
        rig_dir = tmp / f"rig-{kind}-{len(flags)}"
        rig_dir.mkdir(parents=True, exist_ok=True)
        regime = asyncio.run(_regime_tools(rig_dir, policy, kit))
    with pytest.MonkeyPatch.context() as mp:
        _point_repair_off(mp, tmp, kind)
        explode_clients(mp)
        T.patch_detok_load(mp, T.Calls(), tmp)
        d_code, d_out, d_err = _run([*T.session_argv(config, tmp), "--dry-run", *argv])
    assert d_code == 0, f"premise: the tools-on dry run exits 0 (exit {d_code}):\n{d_out}{d_err}"
    lines = [line for line in (d_out + d_err).splitlines() if line.startswith("tools=")]
    assert len(lines) == 1, f"one tools line; the dry run printed:\n{d_out}{d_err}"
    return {"policy": repr(policy), "regime_tools": regime, "tools_line": _norm(lines[0], tmp)}


def _blob(path: str) -> str:
    return subprocess.run(
        ["git", "hash-object", path], capture_output=True, text=True, check=True
    ).stdout.strip()


def emit() -> None:
    """Print the fixture JSON (run from a detached worktree at ``BASE``; MANIFEST.md has the command)."""
    import sys

    from local_astra import cli, config

    print(f"origin: cli={cli.__file__} config={config.__file__}", file=sys.stderr)
    strings: list[str] = []
    rows: dict[str, Any] = {}
    for row in ROWS:
        with tempfile.TemporaryDirectory() as tmp:
            seen = observe_tools_off_row(row, Path(tmp))
        for key in INTERNED:
            if isinstance(seen[key], str):
                if seen[key] not in strings:
                    strings.append(seen[key])
                seen[key] = strings.index(seen[key])
        rows[row] = seen
    labels: dict[str, Any] = {}
    for kind in LABEL_KINDS:
        with tempfile.TemporaryDirectory() as tmp:
            labels[kind] = observe_label(kind, Path(tmp))
    provenance = {
        "base": BASE,
        "local_astra/cli.py": _blob(cli.__file__),
        "local_astra/config.py": _blob(config.__file__),
    }
    print("{")
    print(f' "provenance": {json.dumps(provenance, sort_keys=True)},')
    print(' "labels": {')
    print(
        ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in labels.items())
    )
    print(" },")
    print(' "strings": [')
    print(",\n".join(f"  {json.dumps(s)}" for s in strings))
    print(" ],")
    print(' "tools_off": {')
    print(
        ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in rows.items())
    )
    print(" }")
    print("}")


def base_fixture() -> dict[str, Any]:
    """The fixture with every interned string resolved (the file stores each distinct text once, in
    ``strings``, and a row's text field as its index; ``None`` stays ``None``)."""
    assert FIXTURE.is_file(), f"T-0123 criterion 5: the base fixture {FIXTURE} exists"
    data: dict[str, Any] = json.loads(FIXTURE.read_text())
    strings = data["strings"]
    for seen in data["tools_off"].values():
        for key in INTERNED:
            if isinstance(seen[key], int):
                seen[key] = strings[seen[key]]
    return data


# --- the rows -------------------------------------------------------------------------------------------------------


def test_the_fixture_was_generated_from_the_base_cli_and_config():
    """Provenance: the fixture's recorded blobs (``git hash-object`` of the files the generator imported) are the
    base commit's ``local_astra/cli.py`` and ``config.py``. Skips, naming the SHA, when the commit is absent."""
    gr.require_commits(BASE)
    provenance = base_fixture()["provenance"]
    assert provenance["base"] == BASE, provenance
    for path in ("local_astra/cli.py", "local_astra/config.py"):
        blob = gr._git("rev-parse", f"{BASE}:{path}").stdout.strip()
        assert blob and provenance[path] == blob, (path, provenance[path], blob)


def test_the_fixture_covers_every_tools_off_row_and_both_labels():
    """The fixture is the whole tools-off half of the table (2 tools x 7 loop x 3 steps = 42 rows) and both
    label kinds; a row missing from it would make its comparison below vacuous."""
    data = base_fixture()
    assert len(ROWS) == 42
    assert sorted(data["tools_off"]) == sorted(ROWS)
    assert sorted(data["labels"]) == sorted(LABEL_KINDS)


def test_the_4d_exemption_is_exactly_the_rows_whose_base_refusal_did_not_name_tools_dry_run():
    """The exemption is not too wide: every exempt row was refused at 68d5566 (exit 2) with a message that did
    not name ``--tools dry-run`` (it advised ``--tool-loop``), and every other refused tools-off row's base message
    did name it (4(a)/(b), unchanged)."""
    rows = base_fixture()["tools_off"]
    for row, base in rows.items():
        if row in STDERR_EXEMPT_4D:
            assert base["startup_code"] == 2 and base["dry_run_code"] == 2, (row, base)
            assert "--tools dry-run" not in base["startup_stderr"], (row, base["startup_stderr"])
        elif base["startup_code"] == 2:
            assert "--tools dry-run" in base["startup_stderr"], (row, base["startup_stderr"])


#: T-0183 re-scope (docs/tasks/T-0183.md H-T183-5; 68d5566 predates ``--visual-tokens``): the working-tree rig passes
#: ``--visual-tokens 1120`` (BASE's bytes then hold before and after T-0183 makes 1120 the default outside live), and
#: the expectation is the fixture's bytes with one literal, positioned edit, written out from T-0183 A3a: in a dry run
#: that exits 0, the one ``tools=off`` line, where it stands, gains `` visual_tokens=1120``. Nothing else may differ.
VT1120 = ("--visual-tokens", "1120")
T0183_FROM = "tools=off"
T0183_TO = "tools=off visual_tokens=1120"


def t0183_expected(base: dict[str, Any], row: str) -> dict[str, Any]:
    """``base`` (68d5566's observation) with T-0183's one positioned edit, and only that."""
    want = dict(base)
    if base["dry_run_code"] == 0:
        lines = base["dry_run_stdout"].split("\n")
        at = [i for i, line in enumerate(lines) if line == T0183_FROM]
        assert len(at) == 1, f"premise: {row}'s 68d5566 dry run prints one {T0183_FROM!r} line"
        lines[at[0]] = T0183_TO
        want["dry_run_stdout"] = "\n".join(lines)
    return want


def test_t0183_edit_reaches_every_started_row():
    """T-0183 re-scope (premise of the edit): every tools-off row whose 68d5566 dry run exits 0 (16 of the 42:
    2 tools x 4 loop settings that start x 2 step settings that start) gets exactly the one edit, and no refused row
    does."""
    rows = base_fixture()["tools_off"]
    edited = sorted(r for r, b in rows.items() if t0183_expected(b, r) != b)
    started = sorted(r for r, b in rows.items() if b["dry_run_code"] == 0)
    assert edited == started and len(started) == 16, (edited, started)


@pytest.mark.parametrize("row", ROWS)
def test_tools_off_startup_equals_68d5566(tmp_path, row):
    """Criterion 5: exit codes, startup output, the policy repr (every field), ``kit is None``, the full
    ``--dry-run`` stdout and the stderr equal the base fixture; only the 4(d) rows' stderr is exempt.

    T-0183 re-scope (H-T183-5): the rig passes ``--visual-tokens 1120`` and the expectation is the fixture with
    ``t0183_expected``'s one positioned edit to the dry-run stdout.

    T-0182 group B (B-R5, the CLI-built policy moves): a started row's policy is BASE's with the two clip fields
    appended as ``cli`` now builds them in a tools-off start, ``ack_clip=False`` (filler clips off by default) and
    ``still_working_after_ms=None`` (not live). Reading: the new fields are declared after BASE's last field."""
    base = t0183_expected(base_fixture()["tools_off"][row], row)  # a copy (``dict(base)``)
    if base["policy"] is not None:
        assert base["policy"].endswith(")"), base["policy"]
        base["policy"] = base["policy"][:-1] + ", ack_clip=False, still_working_after_ms=None)"
    got = observe_tools_off_row(row, tmp_path, VT1120)
    keys = [
        "startup_code",
        "startup_stdout",
        "policy",
        "kit_is_none",
        "dry_run_code",
        "dry_run_stdout",
    ]
    if row not in STDERR_EXEMPT_4D:
        keys += ["startup_stderr", "dry_run_stderr"]
    diff = {k: {"68d5566": base[k], "now": got[k]} for k in keys if got[k] != base[k]}
    assert not diff, f"T-0123 criterion 5: {row} differs from 68d5566: {json.dumps(diff, indent=1)}"


if __name__ == "__main__":
    emit()
