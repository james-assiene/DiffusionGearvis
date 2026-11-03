"""T-0182 group B through ``cli``: the ``filler_clips`` resolution (B2), ``still_working_after_ms`` per mode (B4), the
startup line (B2, ruling B-R2), defaults byte-identical to BASE_B (B2 negative), ``render-clips`` (B6) and the doctor
channel (B6, ruling B-R2) (docs/tasks/T-0182.md, group B and its rulings B-R1..B-R4).

**Oracles, by hand from the criteria:**

- B2: ``--filler-clips``/``--no-filler-clips`` beats ``Profile.filler_clips`` (when the profile sets it), which beats
  the default (off). ``FILLER_WANT`` is the 3×3 table (flag none/on/off × profile unset/true/false) written out.
- B2/B-R1: "on" is ``TurnPolicy(ack_clip=True)`` on the machine ``cli`` builds; "off" is ``ack_clip=False``.
  B-R4: ``TurnPolicy()`` keeps BASE's ``ack_clip=True``, so the product default (off) is decided here, by ``cli``; the
  rows that read ``policy["ack_clip"] is False`` with defaults are the ones that kill "``cli`` leaves the contract
  default in place".
- B2/B-R2: ``astra session: filler clips: on (<flag|profile>)`` is printed once, to stdout, exactly when the
  resolution is on, in every mode, ``--dry-run`` included; never when off.
- B4: ``still_working_after_ms = Profile.still_working_after_s * 1000`` (default 20, ``ge=5``) only with ``tools =
  live`` and the loop on; ``None`` in off, ``dry_run`` (loop on or off). ``SW_WANT`` is the table.
- B2 negative: with defaults, ``--dry-run`` output and a real start's stdout/stderr (and exit code, and every
  ``Session`` keyword, ruling B-R1 "one switch") equal BASE_B's (7646b8b6), in off, ``dry_run``, live ``--no-web`` and
  live with web. BASE_B runs ``_capture`` below in a subprocess over ``git archive BASE_B`` (T-0114's ``_base_tree``,
  origin checked). The machine's policy is compared on BASE_B's fields; its two new fields are asserted separately.
- B6: ``diffusion_gearvis render-clips [--data-dir D] [--kinds …] [--force]`` gives the same exit code, stdout,
  stderr, files and manifest (``rendered_at`` aside) as ``scripts/render_clips.py``'s ``main`` with the same
  arguments, behind the same scripted TTS; ``doctor`` reaches ``collect_clips(filler_clips=<profile's>)`` through
  ``preflight.gather`` (B-R2: a keyword argument, default False).

Every resolution cell opens with a named assertion on the new ``Profile`` field, so a missing field is an
``AssertionError`` naming B2/B4, never a config-load error inside the rig. Activated for ``/implement T-0182`` group
B (the rows were pending through e1ca276f); the BASE_B differential was never pending (green at BASE_B by
construction).

Regime: CPU over the CLI rig's fakes (``tests/unit/_tools_switch_support.py``, ``test_cli_tools_live.fake_live``), no
browser, nothing served, profile ``lean`` from a temporary ``astra.toml``; the render rows under ``no_sockets`` with
``ScriptedTts``; n = 1, deterministic.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra import preflight
from local_astra.config import Profile
from tests import _git_range as gr
from tests.unit import _t0197_pins as t0197  # T-0197 pin move
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _t0207_pins as t0207  # T-0207 C3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit._t0181_pins import run_without_mic_unavailable  # T-0181 A-R13
from tests.unit._t0201_pins import capture_without_default  # T-0201 D3 re-pin
from tests.unit.test_cli_session import explode_clients, run_cli
from tests.unit.test_cli_tools_live import Events, argv, fake_live

#: BASE_B, a literal (``.claude/rules/tests.md``): main when group B branched
BASE_B = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
LINE_PREFIX = "astra session: filler clips: "

#: mode -> extra argv (live runs headless; ``--no-web`` keeps the loopback start pages only)
MODES: dict[str, list[str]] = {
    "off": [],
    "dry_run": ["--tools", "dry-run"],
    "dry_run-loop-off": ["--tools", "dry-run", "--no-tool-loop"],
    "live-no-web": ["--tools", "live", "--browser", "headless", "--no-web"],
    "live-web": ["--tools", "live", "--browser", "headless"],
}
#: the four modes the BASE_B differential captures (B2 negative)
BASE_MODES = ("off", "dry_run", "live-no-web", "live-web")

#: flag case -> argv
FLAGS: dict[str, list[str]] = {
    "none": [],
    "on": ["--filler-clips"],
    "off": ["--no-filler-clips"],
}
#: profile case -> astra.toml body appended under ``[profiles.lean]``
BODIES: dict[str, str] = {
    "unset": "",
    "true": "filler_clips = true\n",
    "false": "filler_clips = false\n",
}
#: (flag, profile) -> (on, source), by hand from B2: the flag, else the profile when it sets the key, else off
FILLER_WANT: dict[tuple[str, str], tuple[bool, str | None]] = {
    ("none", "unset"): (False, None),
    ("none", "true"): (True, "profile"),
    ("none", "false"): (False, None),
    ("on", "unset"): (True, "flag"),
    ("on", "true"): (True, "flag"),
    ("on", "false"): (True, "flag"),  # the flag beats the profile
    ("off", "unset"): (False, None),
    ("off", "true"): (False, None),  # adversarial: the flag beats an on profile
    ("off", "false"): (False, None),
}

#: still_working_after_s body -> ms in a live loop (B4: default 20 s)
SW_BODIES: dict[str, tuple[str, int]] = {
    "default": ("", 20_000),
    "s7": ("still_working_after_s = 7\n", 7_000),
}
#: mode -> whether still_working_after_ms is set there (B4: live with the loop on, only)
SW_ON: dict[str, bool] = {
    "off": False,
    "dry_run": False,  # adversarial: tools on and the loop on, but not live
    "dry_run-loop-off": False,
    "live-no-web": True,
    "live-web": True,
}


def require_fields() -> None:
    """B2/B4's ``Profile`` fields, as a named assertion (never a config-load error inside a rig)."""
    missing = [
        f for f in ("filler_clips", "still_working_after_s") if f not in Profile.model_fields
    ]
    assert not missing, (
        f"T-0182 B2/B4: Profile gains {missing} (filler_clips=False, still_working_after_s=20)"
    )


# --- rigs ----------------------------------------------------------------------------------------------------------


def _run(tmp: Path, flags: list[str], body: str, *, dry: bool) -> dict[str, Any]:
    """``astra session <flags>`` over the fakes, ``--dry-run`` or a real start stopped before serving: exit code,
    stdout, stderr (tmp and tree roots normalised), each ``Session``'s keywords described, and the built machine's
    policy (a dict of its dataclass fields, or None when nothing was built)."""
    from tests.unit import test_t0134c_live_feedback_default as t0134c

    tmp.mkdir(parents=True, exist_ok=True)
    w = str(tmp)
    calls = T.Calls()
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISPLAY", ":99")
        mp.delenv("ASTRA_AUDIO_SOURCE", raising=False)
        T.patch_detok_load(mp, calls, tmp)
        if dry:
            explode_clients(mp)
        else:
            T.fake_clients(mp, calls)
            T.stop_before_serving(mp, calls)
        fake_live(mp, Events())
        args = argv(tmp, *flags, config_body=body)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_cli([*args, "--dry-run"] if dry else args)
    policy = None
    if calls.graphs:
        policy = dataclasses.asdict(calls.graphs[-1].session.machine.policy)
    return {
        "code": code,
        "out": t0134c._norm(out.getvalue(), w),
        "err": t0134c._norm(err.getvalue(), w),
        "graphs": len(calls.graphs),
        "session_kwargs": [t0134c._describe(k, w) for k in calls.session_kwargs],
        "policy": json.loads(json.dumps(policy, default=str)),
    }


def _lines(text: str) -> list[str]:
    return [x for x in text.splitlines() if x.startswith(LINE_PREFIX)]


def _want_line(source: str | None) -> list[str]:
    return [] if source is None else [f"{LINE_PREFIX}on ({source})"]


# --- B2: the 3×3 resolution -----------------------------------------------------------------------------------------


CELLS = sorted(FILLER_WANT)


def test_the_filler_table_is_the_whole_product() -> None:
    assert set(FILLER_WANT) == {(f, p) for f in FLAGS for p in BODIES}


@pytest.mark.parametrize(("flag", "body"), CELLS, ids=[f"flag-{f}/profile-{p}" for f, p in CELLS])
def test_filler_clips_resolution(tmp_path: Path, flag: str, body: str) -> None:
    """B2 (and B-R1, B-R2): the hand table ``FILLER_WANT`` through a real start (the machine's
    ``TurnPolicy.ack_clip``, the startup line once on stdout, none on stderr) and ``--dry-run`` (the same line once on
    stdout), tools off."""
    require_fields()
    on, source = FILLER_WANT[(flag, body)]
    argv_ = FLAGS[flag]
    start = _run(tmp_path / "start", argv_, BODIES[body], dry=False)
    assert start["code"] == 0 and start["graphs"] == 1, (start["code"], start["err"][-1500:])
    assert start["policy"]["ack_clip"] is on, (
        f"T-0182 B2: flag {flag} / profile {body} builds TurnPolicy(ack_clip={start['policy']['ack_clip']}), "
        f"want {on}"
    )
    assert _lines(start["out"]) == _want_line(source), (
        f"T-0182 B2: flag {flag} / profile {body}: startup lines {_lines(start['out'])}\n{start['out']}"
    )
    assert _lines(start["err"]) == [], start["err"]
    dry = _run(tmp_path / "dry", argv_, BODIES[body], dry=True)
    assert dry["code"] == 0, dry["err"][-1500:]
    assert _lines(dry["out"]) == _want_line(source), (
        f"T-0182 B-R2: the line prints under --dry-run too; flag {flag} / profile {body}:\n{dry['out']}"
    )
    assert _lines(dry["err"]) == [], dry["err"]


@pytest.mark.parametrize("mode", sorted(MODES))
def test_the_startup_line_in_every_mode(tmp_path: Path, mode: str) -> None:
    """B-R2: with ``--filler-clips`` the line ``... on (flag)`` prints once to stdout in every mode, real start and
    ``--dry-run``; the machine has ``ack_clip=True``. Without it (defaults) no such line in any mode."""
    require_fields()
    for dry in (False, True):
        on = _run(tmp_path / f"on-{dry}", [*MODES[mode], *FLAGS["on"]], "", dry=dry)
        assert on["code"] == 0, (mode, dry, on["err"][-1500:])
        assert _lines(on["out"]) == [f"{LINE_PREFIX}on (flag)"], (mode, dry, on["out"])
        if not dry:
            assert on["policy"]["ack_clip"] is True, (mode, on["policy"])
        off = _run(tmp_path / f"off-{dry}", MODES[mode], "", dry=dry)
        assert off["code"] == 0, (mode, dry, off["err"][-1500:])
        assert _lines(off["out"] + off["err"]) == [], (mode, dry, off["out"])
        if not dry:
            assert off["policy"]["ack_clip"] is False, (mode, off["policy"])


# --- B4: still_working_after_ms per mode ---------------------------------------------------------------------------


SW_CELLS = [(m, b) for m in sorted(MODES) for b in sorted(SW_BODIES)]


@pytest.mark.parametrize(("mode", "body"), SW_CELLS, ids=[f"{m}/{b}" for m, b in SW_CELLS])
def test_still_working_after_ms_by_mode(tmp_path: Path, mode: str, body: str) -> None:
    """B4: ``still_working_after_ms`` is ``still_working_after_s * 1000`` (20 000 by default, 7 000 with the key at 7)
    in live with the loop on (web or not), and None in off and ``dry_run`` whatever the key says; ``ack_clip`` stays
    False (the two switches are independent)."""
    require_fields()
    text, ms = SW_BODIES[body]
    run = _run(tmp_path / "start", MODES[mode], text, dry=False)
    assert run["code"] == 0 and run["graphs"] == 1, (mode, run["err"][-1500:])
    want = ms if SW_ON[mode] else None
    got = run["policy"]["still_working_after_ms"]
    assert got == want, f"T-0182 B4: {mode}/{body} builds still_working_after_ms={got}, want {want}"
    assert run["policy"]["ack_clip"] is False, run["policy"]


def test_still_working_after_s_is_validated() -> None:
    """B4: ``Profile.still_working_after_s: int = 20, ge=5``: 4 and 0 raise at load, 5 and 30 load."""
    import pydantic

    require_fields()
    assert Profile.model_fields["still_working_after_s"].default == 20
    assert Profile.model_fields["filler_clips"].default is False
    assert Profile(name="lean").still_working_after_s == 20  # type: ignore[attr-defined]
    for bad in (4, 0, -1):
        with pytest.raises(pydantic.ValidationError):
            Profile(name="lean", still_working_after_s=bad)
    for ok in (5, 30):
        assert Profile(name="lean", still_working_after_s=ok).still_working_after_s == ok  # type: ignore[attr-defined]


# --- B2 negative: defaults byte-identical to BASE_B -----------------------------------------------------------------


def _capture(tmp: Path) -> dict[str, Any]:
    """Defaults in each ``BASE_MODES`` mode: ``--dry-run`` and a real start (``_run``)."""
    out: dict[str, Any] = {}
    for mode in BASE_MODES:
        # T-0202 D3 re-pin: pin off (a live mode appends --wait-tool off on a tree with the flag, so the session is
        # BASE_B's, which has no wait tool; off and dry_run are not live and stay as they are; a no-op at BASE_B)
        flags = t0202.pin_off(MODES[mode], live_prefix=False)
        out[mode] = {
            "dry": _run(tmp / mode / "dry", flags, "", dry=True),
            "start": _run(tmp / mode / "start", flags, "", dry=False),
        }
    return json.loads(json.dumps(out, default=str))


_RUNNER = """
import json, pathlib, sys
import local_astra
from tests.unit import test_t0182_cli as m
out = pathlib.Path(sys.argv[1])
sections = m._capture(out.parent / "run")
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""

#: B4's field: the one ``TurnPolicy`` field the defaults set in a live loop
NEW_FIELDS = ("ack_clip", "still_working_after_ms")


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.unit import test_session_tool_loop as tsl

    gr.require_commits(BASE_B)
    tmp = tmp_path_factory.mktemp("t0182_cli_base")
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


#: T-0183 re-scope (merged from main c0cb001e; docs/tasks/T-0183.md A3a): outside live the default ``--dry-run``
#: tools line is BASE's plus `` visual_tokens=1120``; live is unchanged (A4(i)). Written out by hand.
T0183_TOOLS_SUFFIX = " visual_tokens=1120"
T0183_MODES = ("off", "dry_run")


def t0183_out(text: str, mode: str) -> str:
    """BASE_B's stdout with T-0183 A3a's one positioned edit in ``off``/``dry_run`` (the one ``tools=`` line gains
    the suffix), and only that; live text is returned unchanged."""
    if mode not in T0183_MODES:
        return text
    lines = text.split("\n")
    at = [i for i, line in enumerate(lines) if line.startswith("tools=")]
    assert len(at) == 1, f"premise: BASE_B's {mode} output has one tools line:\n{text}"
    lines[at[0]] += T0183_TOOLS_SUFFIX
    return "\n".join(lines)


def t0183_session_kwargs(kwargs: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """BASE_B's described ``Session`` keywords with T-0183 A2's one edit in ``off``/``dry_run``: ``visual_tokens``
    goes from BASE's ``None`` to 1120 (``Session`` gains no keyword); live is returned unchanged."""
    if mode not in T0183_MODES:
        return kwargs
    out = []
    for kw in kwargs:
        assert kw.get("visual_tokens") is None, (
            f"premise: BASE_B's {mode} Session had no budget: {kw}"
        )
        out.append({**kw, "visual_tokens": 1120})
    return out


#: T-0191 C3: the modes whose BASE_B expectation the default flip moves (web-mode live: v2's guidance -> v3's)
C_MOVES_MODES = ("live-web",)


@pytest.mark.parametrize(
    "mode",
    [m for m in BASE_MODES],
)
def test_defaults_byte_identical_to_base_b(tmp_path: Path, base: dict[str, Any], mode: str) -> None:
    """B2 negative (and B-R1's one switch): with defaults, ``--dry-run`` output and a real start's exit code, stdout,
    stderr and ``Session`` keywords equal BASE_B's; the built ``TurnPolicy`` equals BASE_B's on BASE_B's fields. Green
    at BASE_B and after; the new fields' values are B1/B4's (checked only where the tree has them). T-0183 re-scope
    (main c0cb001e): in ``off``/``dry_run`` BASE_B's side carries T-0183's two literal edits (``t0183_out``,
    ``t0183_session_kwargs``); live is compared as captured."""
    # T-0197 pin move: these are the *defaults*, so the run keeps the HUD default (no ``--no-serve-hud`` pin) and
    # admits exactly its one stdout line at criterion 2's position (``_t0197_pins.admit_hud_line``)
    hud_moved = t0197.tree_has_opt_out()
    with t0197.hud_default():
        now = _capture(tmp_path)[mode]
    for kind in ("dry", "start"):
        want, got = dict(base[mode][kind]), now[kind]
        if hud_moved:
            got = t0197.admit_hud_line(got, kind)
        if mode in C_MOVES_MODES:
            # T-0191 C3 pin move: only the guidance text, v2's bytes to v3's; every other byte is BASE_B's
            from tests.unit.test_t0191_cli_guidance import v2_moved_to_v3

            if kind == "start":
                want = v2_moved_to_v3(want, f"T-0182 B2 {mode} {kind}")
        got = run_without_mic_unavailable(
            got
        )  # T-0181 A-R13 pin move: only the mic_unavailable keyword
        # T-0201 D3 re-pin (admit): a live run's default screen reminder line and screen_reminder=True keyword only
        got = capture_without_default(got, f"T-0182 B2 {mode} {kind}")
        # T-0207 C3 re-pin (admit): a live run's default completion check line and completion_check=True keyword only
        got = t0207.capture_without_default(got, f"T-0182 B2 {mode} {kind}")
        if (
            kind == "dry"
        ):  # T-0183 re-scope (A3a: the dry-run tools line; a real start prints no tools line)
            want["out"] = t0183_out(want["out"], mode)
        want["session_kwargs"] = t0183_session_kwargs(
            want["session_kwargs"], mode
        )  # T-0183 re-scope (A2)
        assert got["code"] == want["code"] == 0, (mode, kind, got["err"][-1500:])
        for key in ("out", "err", "graphs", "session_kwargs"):
            assert got[key] == want[key], (
                f"T-0182 B2: {mode} {kind} {key} differs from BASE_B {BASE_B[:8]}:\n"
                f"base {json.dumps(want[key])[:2000]}\nnow  {json.dumps(got[key])[:2000]}"
            )
    want_p, got_p = base[mode]["start"]["policy"], now["start"]["policy"]
    assert want_p is not None and got_p is not None
    assert not any(f in want_p for f in NEW_FIELDS), (
        "premise: BASE_B's TurnPolicy has neither B1/B4 field"
    )
    assert {k: got_p[k] for k in want_p} == want_p, (
        f"T-0182 B2: {mode}'s TurnPolicy differs from BASE_B's on BASE_B's fields: {got_p} vs {want_p}"
    )
    extra = set(got_p) - set(want_p)
    assert extra <= set(NEW_FIELDS), (
        f"T-0182 B1/B4: TurnPolicy gains only {NEW_FIELDS}; got {sorted(extra)}"
    )
    if extra:
        assert got_p["ack_clip"] is False
        assert got_p["still_working_after_ms"] == (20_000 if mode.startswith("live") else None)


# --- B6: render-clips delegates ------------------------------------------------------------------------------------


def _config(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    return T.write_config(tmp)


def _patch_tts(mp: pytest.MonkeyPatch, tts: Any) -> None:
    """``ScriptedTts`` wherever the script can reach ``PocketTtsClient``: the source module (a fresh load binds it)
    and every loaded module that holds the real class (a cached load)."""
    from local_astra.conductor import tts_client
    from tests.unit.test_t0182_clips_registry import ours

    real = tts_client.PocketTtsClient

    def factory(services: Any = None, *a: Any, **k: Any) -> Any:
        tts.services = services
        return tts

    mp.setattr(tts_client, "PocketTtsClient", factory)
    held = [m for m in ours() if m.__dict__.get("PocketTtsClient") is real]
    held += [
        m
        for m in list(sys.modules.values())
        if isinstance(m, ModuleType)
        and m not in held
        and m.__dict__.get("PocketTtsClient") is real
        and str(m.__dict__.get("__file__", "")).endswith("render_clips.py")
    ]
    for module in held:
        mp.setattr(module, "PocketTtsClient", factory)


def _tree(data: Path) -> dict[str, Any]:
    """Every file under ``data`` (bytes as hex sha), the manifest parsed with ``rendered_at`` dropped."""
    import hashlib

    out: dict[str, Any] = {}
    for p in sorted(data.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(data))
        if p.name == "manifest.json":
            m = json.loads(p.read_text())
            for entry in m.get("clips", {}).values():
                entry.pop("rendered_at", None)
            out[rel] = m
        else:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


#: render-clips argv cases (after ``--data-dir D``), run in sequence on one data dir
RENDER_SEQUENCE: list[tuple[str, list[str]]] = [
    ("bogus", ["--kinds", "bogus"]),  # refused, exit 2, nothing written
    ("default", []),  # the default set: ending kinds and still_working (B6)
    ("again", []),  # every kind skipped (current)
    ("force", ["--force"]),
    ("ack", ["--kinds", "ack"]),  # a filler kind only when named
]


def _render_run(
    tmp: Path, capsys: pytest.CaptureFixture[str], *, through_cli: bool
) -> list[dict[str, Any]]:
    from tests.unit.test_t0182_render_clips import ScriptedTts, load_script, no_sockets

    data = tmp / "data"
    config = _config(tmp)
    script = load_script()
    results: list[dict[str, Any]] = []
    try:
        for case, extra in RENDER_SEQUENCE:
            tts = ScriptedTts()
            with pytest.MonkeyPatch.context() as mp, no_sockets():
                _patch_tts(mp, tts)
                rest = ["--data-dir", str(data), *extra]
                capsys.readouterr()
                if through_cli:
                    code = run_cli(["--config", str(config), "render-clips", *rest])
                else:
                    try:
                        code = int(script.main(["--config", str(config), *rest]))
                    except SystemExit as exc:
                        code = int(exc.code or 0)
                got = capsys.readouterr()
            results.append(
                {
                    "case": case,
                    "code": code,
                    "out": got.out.replace(str(tmp), "<tmp>"),
                    "err": got.err.replace(str(tmp), "<tmp>"),
                    "texts": tts.texts(),
                    "files": _tree(data),
                }
            )
    finally:
        sys.modules.pop(script.__name__, None)
    return results


def test_render_clips_subcommand_delegates_to_the_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B6: ``diffusion_gearvis render-clips`` is a thin wrapper over ``render_clips.main``: over the same sequence
    (a refused kind, the default set, a re-run, ``--force``, ``--kinds ack``) on a fresh data dir each, the exit codes,
    stdout, stderr, the texts sent to the TTS and the files written (the manifest without ``rendered_at``) are the
    script's. Premise rows: the script refuses ``bogus`` (exit 2) and its default set renders ``still_working``."""
    assert _has_render_clips(), "T-0182 B6: cli has no render-clips subcommand"
    want = _render_run(tmp_path / "script", capsys, through_cli=False)
    assert want[0]["code"] == 2 and "bogus" in want[0]["err"] + want[0]["out"], want[0]
    assert "still working on it" in want[1]["texts"], (
        f"premise (B6): the default set renders still_working; {want[1]}"
    )
    got = _render_run(tmp_path / "cli", capsys, through_cli=True)
    for w, g in zip(want, got, strict=True):
        assert g == w, (
            f"T-0182 B6: render-clips {w['case']} differs from scripts/render_clips.py:\n"
            f"script {json.dumps(w)[:2000]}\ncli    {json.dumps(g)[:2000]}"
        )


def _has_render_clips() -> bool:
    """``render-clips --help`` exits 0 when the subcommand exists; argparse refuses an unknown one with 2."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_cli(["render-clips", "--help"]) == 0


# --- B6: doctor passes profile.filler_clips --------------------------------------------------------------------------


def _recording_collectors(mp: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every ``preflight.COLLECTORS`` entry replaced by a stub returning ``{}``; ``collect_clips`` records its
    keyword arguments (B-R2: ``filler_clips`` as a keyword, default False) and returns no clips."""
    seen: list[dict[str, Any]] = []
    for name in preflight.COLLECTORS:
        mp.setattr(preflight, name, lambda *a, **k: {})

    def collect_clips(**kw: Any) -> dict[str, Any]:
        seen.append(dict(kw))
        return {"clips": {}}

    mp.setattr(preflight, "collect_clips", collect_clips)
    return seen


@pytest.mark.parametrize("value", [False, True])
def test_gather_passes_filler_clips_to_collect_clips(value: bool) -> None:
    """B6/B-R2: ``preflight.gather(profile)`` calls ``collect_clips(filler_clips=profile.filler_clips)``."""
    require_fields()
    with pytest.MonkeyPatch.context() as mp:
        seen = _recording_collectors(mp)
        preflight.gather(Profile(name="lean", filler_clips=value))
    assert seen == [{"filler_clips": value}], f"T-0182 B6: gather called collect_clips with {seen}"


@pytest.mark.parametrize(("body", "value"), [("", False), ("filler_clips = true\n", True)])
def test_doctor_passes_the_profiles_filler_clips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str, value: bool
) -> None:
    """B6: ``doctor`` reads the active profile's ``filler_clips`` and hands it to ``collect_clips`` (unset: False)."""
    require_fields()
    config = _config(tmp_path)
    config.write_text(config.read_text() + body)
    with pytest.MonkeyPatch.context() as mp:
        seen = _recording_collectors(mp)
        run_cli(["--config", str(config), "doctor"])
    capsys.readouterr()
    assert seen == [{"filler_clips": value}], f"T-0182 B6: doctor called collect_clips with {seen}"
