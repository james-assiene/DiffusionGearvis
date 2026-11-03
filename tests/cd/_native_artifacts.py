"""Hand-built two-pass artifacts for the T-0084d rows (format comparison, per-format report, CLI). Not a test module.

Torch-free and import-light: ``tests/unit/test_cd_toolcall_cli.py`` (fast suite) imports it too. Every row is
written by hand from the artifact contract of ``docs/tasks/T-0080b.md`` / ``T-0084c.md`` (criterion 7): a regime line
first (``stage``, ``cells``, ``commit``), then ``"pass": 1`` rows (one per case and format, carrying ``format``) and
``"pass": 2`` rows (one per case and cell), then a trailer. Nothing here calls the module under test.

Two shapes:

- ``full_rows``: every tool case routed with its format's own tag in both formats, every draft invalid (so
  ``cd_on_invalid`` sends pass 2 exactly where ``cd_always`` does, and ``draft_only``/``draft_only:native`` are
  ineligible at schema 0), every pass 2 valid, CS-true and exact, 5 steps, pass-1 wall 1000 ms, pass-2 wall 500 ms
  unless ``walls`` overrides a cell. The pre-registered choice is then decided by the overridden walls alone, which
  makes it hand-computable (see the users' docstrings).
- ``mixed_rows``: the 3a artifact (see its docstring for every hand-computed number).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("end_projection", "perstep_mar", "perstep_mar_qargmax", "perstep_mar+seed")
COMMIT = "abc1234abc1234abc1234abc1234abc1234abc1"


def cc_cells(thought: str = "T1") -> list[str]:
    return [f"{thought}/named/{a}" for a in ARMS]


def native_cells(*thoughts: str) -> list[str]:
    return [f"native/{t}/named/{a}" for t in thoughts for a in ARMS]


DEV_CELLS = [*cc_cells("T1"), *native_cells("T1", "T2", "T4")]


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "arguments": dict(args)}


def p1(
    case: str,
    fmt: str,
    *,
    kind: str = "tool",
    tool: str | None = "click_at",
    routed: bool = True,
    tag: str | None = None,
    valid: bool | None = False,
    exact: bool = False,
    draft_call: Mapping[str, Any] | None = None,
    parse: str | None = None,
    wall: float = 1000.0,
    close: str = "",
    spoken: str = "I will do it. ",
    text_after_close: str | None = None,
    open_thought_call: bool = False,
    prompt_len: int | None = None,
) -> dict[str, Any]:
    """One pass-1 row. ``routed`` False gives route kind ``none`` (no draft fields)."""
    if routed:
        route = {
            "kind": "toolcall",
            "tag": tag or ("native" if fmt == "native" else "cc"),
            "draft_name": "click_at",
            "spoken_prefix": spoken,
        }
        draft = {
            "draft_parse": parse if parse is not None else ("ok" if valid else "bad json"),
            "draft_call": dict(draft_call) if draft_call is not None else None,
            "draft_schema_valid": valid,
            "draft_exact": exact,
            "draft_call_text": "x",
        }
    else:
        route = {"kind": "none", "tag": None, "draft_name": None, "spoken_prefix": ""}
        draft = {
            "draft_parse": None,
            "draft_call": None,
            "draft_schema_valid": None,
            "draft_exact": False,
            "draft_call_text": "",
        }
    row: dict[str, Any] = {
        "pass": 1,
        "format": fmt,
        "case": case,
        "kind": kind,
        "tool": tool if kind == "tool" else None,
        "error": None,
        "wall_ms": float(wall),
        "prefix_hit_tokens": 0,
        "prefix_query_tokens": 1000,
        "route": route,
        "draft_close": close,
        **draft,
    }
    if fmt == "native":
        row.update(
            text_after_close=text_after_close,
            spoken_prefix_chars=len(route["spoken_prefix"]),
            open_thought_call=open_thought_call,
            prompt_len=prompt_len if prompt_len is not None else 1000,
        )
    return row


def p2(
    case: str,
    cell: str,
    *,
    valid: bool = True,
    exact: bool = True,
    out_call: Mapping[str, Any] | None = None,
    wall: float = 500.0,
    steps: int = 5,
    cs: bool = True,
    seed_ids: Sequence[int] | None = None,
    committed: Sequence[int] | None = None,
    seed_trimmed: int = 0,
    seed_reason: str | None = None,
    requests: int = 1,
) -> dict[str, Any]:
    fmt = "native" if cell.startswith("native/") else "cc"
    parts = cell.split("/")[-3:]
    row: dict[str, Any] = {
        "pass": 2,
        "format": fmt,
        "case": case,
        "cell": cell,
        "thought": parts[0],
        "scope": parts[1],
        "arm": parts[2],
        "requests": requests,
        "parse": "ok" if valid else "bad json",
        "call": dict(out_call)
        if out_call is not None
        else (call("click_at", x=1, y=1) if valid else None),
        "schema_valid": valid,
        "exact": exact,
        "cs": cs,
        "error": None,
        "wall_ms": float(wall),
        "steps": [steps],
        "close_101_pos": None,
        "prefix_hit_tokens": 900,
        "prefix_query_tokens": 1000,
    }
    if seed_reason is not None:
        row.update(seed=None, seed_reason=seed_reason)
    if seed_ids is not None:
        row.update(
            seed_ids=list(seed_ids),
            seed_trimmed=seed_trimmed,
            committed_ids=[list(committed or [])],
        )
    return row


def regime(stage: str, cells: Sequence[str], commit: str = COMMIT) -> dict[str, Any]:
    return {
        "kind": "regime",
        "stage": stage,
        "cells": list(cells),
        "commit": commit,
        "expected_commit": commit,
        "n_cases": None,
        "system_prompt_sha256": "0" * 64,
        "tools_sha256": "1" * 64,
        "cases_sha256": "2" * 64,
        "calibration": {},
        "pass1_max_tokens": 1024,
        "pass2_max_tokens": {"T1": 1024, "T2": 256, "T4": 256},
        "versions": {},
    }


def write_artifact(
    path: Path, stage: str, cells: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [regime(stage, cells), *rows, {"kind": "trailer", "stopped": None}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_rows(
    cells: Sequence[str],
    *,
    tool_cases: int = 4,
    no_tool: int = 1,
    walls: Mapping[str, float] | None = None,
    exact_cases: Mapping[str, set[str]] | None = None,
    exact_default: bool = True,
    wrong_tag: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """See the module docstring. Formats present = the formats of ``cells``. ``exact_cases[cell]`` lists the tool
    cases whose pass 2 is exact for that cell (others: ``exact_default``). ``wrong_tag`` cases answer the native pass 1
    with ``<cc>`` (``draft_parse`` ``wrong_tag``) and, since H-T84-0 (T-0084d criterion 7), still get their native
    pass-2 rows through the picker (``requests`` 2; seed cells ``seed_reason`` "picker")."""
    walls = dict(walls or {})
    exact_cases = dict(exact_cases or {})
    fmts = sorted({"native" if c.startswith("native/") else "cc" for c in cells})
    rows: list[dict[str, Any]] = []
    for i in range(tool_cases):
        cid = f"t{i}"
        for f in fmts:
            if f == "native" and cid in wrong_tag:
                rows.append(p1(cid, f, tag="cc", valid=False, parse="wrong_tag"))
            else:
                rows.append(p1(cid, f, valid=False, parse="bad json"))
        for c in cells:
            ex = cid in exact_cases[c] if c in exact_cases else exact_default
            picked = c.startswith("native/") and cid in wrong_tag
            rows.append(
                p2(
                    cid,
                    c,
                    wall=walls.get(c, 500.0),
                    exact=ex,
                    requests=2 if picked else 1,
                    seed_reason="picker" if picked and c.endswith("+seed") else None,
                )
            )
    for i in range(no_tool):
        for f in fmts:
            rows.append(p1(f"n{i}", f, kind="no_tool", routed=False))
    return rows


# --------------------------------------------------------------------------- the 3a mixed artifact

MIXED_CELLS = [
    "T1/named/perstep_mar",
    "T1/named/perstep_mar+seed",
    "native/T4/named/perstep_mar",
    "native/T4/named/perstep_mar+seed",
]
G_A = call("click_at", x=10, y=10)
G_B = call("click_at", x=20, y=20)
Z = call("click_at", x=21, y=21)
W_A = call("click_at", x=11, y=11)
W = call("click_at", x=30, y=30)
W2 = call("click_at", x=40, y=40)
SEED = [7, 8, 9]
COPY = [7, 8, 9, 10]
NO_COPY = [7, 8, 5, 10]


def mixed_rows(*, wrong_tag_case: bool = False) -> list[dict[str, Any]]:
    """The 3a artifact, dev stage, cells ``MIXED_CELLS``. Tool cases A, B, C (tool ``click_at``), no-tool N1, N2; per
    case the cc pass-1 row, then the native one (so a last-write-wins ``p1_by_case`` keeps the native draft), then the
    pass-2 rows.

    ======  =================================================  ===================================================
    case    cc pass 1                                          native pass 1
    ======  =================================================  ===================================================
    A       routed cc, draft G_A valid exact, wall 100         routed native, invalid ("missing close"), wall 1000,
                                                               spoken "I'll click.", prompt_len 1000
    B       routed cc, invalid ("trailing text"), close        routed native, draft G_B valid exact, wall 1100,
            "</cc>", wall 200                                  spoken "   ", text_after_close " ok", prompt 1010
    C       none, wall 300                                     routed native, draft W valid not exact, wall 1200,
                                                               spoken "Typing.", prompt 1020
    N1      none, wall 400                                     routed native (a false tag), invalid ("bad value"),
                                                               wall 1300, spoken "Sure.", prompt 1030
    N2      none, wall 500                                     none, open_thought_call True, wall 1400, prompt 1040
    ======  =================================================  ===================================================

    Pass 2 (all CS-true, valid):

    - ``T1/named/perstep_mar``: A -> G_A (exact), B -> Z (not exact).
    - ``T1/named/perstep_mar+seed``: A seeded, a copy; B seeded, not a copy, trimmed.
    - ``native/T4/named/perstep_mar``: A -> W_A (not exact), B -> G_B (exact), C -> W (not exact), N1 -> W2.
    - ``native/T4/named/perstep_mar+seed``: A copy, B copy, C not, N1 not; all seeded.

    ``wrong_tag_case`` adds tool case D: cc none (wall 350); native routed with the ``cc`` tag, ``draft_parse``
    ``wrong_tag``, invalid, close "", spoken "", wall 1250; since H-T84-0 (T-0084d criterion 7) it gets native pass 2
    through the picker: ``native/T4/named/perstep_mar`` -> W (valid, not exact, ``requests`` 2), and the seed cell a
    row with no seed (``seed_reason`` "picker").
    """
    rows: list[dict[str, Any]] = []

    def native(case: str, **kw: Any) -> dict[str, Any]:
        return p1(case, "native", **kw)

    # A
    rows += [
        p1("A", "cc", valid=True, exact=True, draft_call=G_A, wall=100),
        native(
            "A",
            valid=False,
            parse="missing close",
            wall=1000,
            spoken="I'll click.",
            prompt_len=1000,
        ),
        p2("A", "T1/named/perstep_mar", out_call=G_A, exact=True),
        p2("A", "T1/named/perstep_mar+seed", out_call=G_A, seed_ids=SEED, committed=COPY),
        p2("A", "native/T4/named/perstep_mar", out_call=W_A, exact=False),
        p2(
            "A",
            "native/T4/named/perstep_mar+seed",
            out_call=W_A,
            exact=False,
            seed_ids=SEED,
            committed=COPY,
        ),
    ]
    # B
    rows += [
        p1("B", "cc", valid=False, parse="trailing text", close="</cc>", wall=200),
        native(
            "B",
            valid=True,
            exact=True,
            draft_call=G_B,
            wall=1100,
            spoken="   ",
            text_after_close=" ok",
            prompt_len=1010,
        ),
        p2("B", "T1/named/perstep_mar", out_call=Z, exact=False),
        p2(
            "B",
            "T1/named/perstep_mar+seed",
            out_call=Z,
            exact=False,
            seed_ids=SEED,
            committed=NO_COPY,
            seed_trimmed=1,
        ),
        p2("B", "native/T4/named/perstep_mar", out_call=G_B, exact=True),
        p2("B", "native/T4/named/perstep_mar+seed", out_call=G_B, seed_ids=SEED, committed=COPY),
    ]
    # C
    rows += [
        p1("C", "cc", routed=False, wall=300),
        native(
            "C", valid=True, exact=False, draft_call=W, wall=1200, spoken="Typing.", prompt_len=1020
        ),
        p2("C", "native/T4/named/perstep_mar", out_call=W, exact=False),
        p2(
            "C",
            "native/T4/named/perstep_mar+seed",
            out_call=W,
            exact=False,
            seed_ids=SEED,
            committed=NO_COPY,
        ),
    ]
    if wrong_tag_case:
        rows += [
            p1("D", "cc", routed=False, wall=350),
            native(
                "D", tag="cc", valid=False, parse="wrong_tag", wall=1250, spoken="", prompt_len=1050
            ),
            p2("D", "native/T4/named/perstep_mar", out_call=W, exact=False, requests=2),
            p2(
                "D",
                "native/T4/named/perstep_mar+seed",
                out_call=W,
                exact=False,
                requests=2,
                seed_reason="picker",
            ),
        ]
    # N1
    rows += [
        p1("N1", "cc", kind="no_tool", routed=False, wall=400),
        native(
            "N1",
            kind="no_tool",
            valid=False,
            parse="bad value",
            wall=1300,
            spoken="Sure.",
            prompt_len=1030,
        ),
        p2("N1", "native/T4/named/perstep_mar", out_call=W2, exact=False),
        p2(
            "N1",
            "native/T4/named/perstep_mar+seed",
            out_call=W2,
            exact=False,
            seed_ids=SEED,
            committed=NO_COPY,
        ),
    ]
    # N2
    rows += [
        p1("N2", "cc", kind="no_tool", routed=False, wall=500),
        native(
            "N2", kind="no_tool", routed=False, wall=1400, open_thought_call=True, prompt_len=1040
        ),
    ]
    return rows


# --------------------------------------------------------------------------- main's module and the stored artifacts

#: T-0084d's fork point (main after T-0084c merged); "unchanged" and "byte-identical" rows compare against it. It is
#: also T-0084d's BASE: ``git merge-base 2e89e87^1 2e89e87^2`` of its merge commit 2e89e87; TIP = ``2e89e87^2``
#: (T-0094b criterion 3: the differential rows compare T-0084d's own two blobs, never BASE with the live file).
BRANCH_BASE = "9e84f21"
T0084D_MERGE = "2e89e87"
T0084D_BASE = BRANCH_BASE
T0084D_TIP = "9abdb3b"
REPO = Path(__file__).resolve().parents[2]


def main_twopass(rev: str = BRANCH_BASE) -> Any:
    """``local_astra/eval/cd/twopass.py`` as it is at ``rev`` (``BRANCH_BASE`` by default), imported under a private
    name (its sibling imports resolve to this checkout, which T-0084d does not change outside twopass.py). T-0094b:
    ``rev`` lets a differential row load the owning task's TIP blob beside its BASE blob."""
    import importlib.util
    import subprocess
    import sys

    name = "_t0084d_main_twopass" if rev == BRANCH_BASE else f"_twopass_at_{rev}"
    if name in sys.modules:
        return sys.modules[name]
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:local_astra/eval/cd/twopass.py"],
        capture_output=True,
        check=True,
    ).stdout
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    m = importlib.util.module_from_spec(spec)
    m.__file__ = f"<{rev}:local_astra/eval/cd/twopass.py>"
    sys.modules[name] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def _main_checkout() -> Path:
    import subprocess

    common = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent


#: (directory under the main checkout, artifact glob): T-0080c's stage 1/2 (logs/, inputs of T-0082c) and T-0082c's
#: replays (sha-checked copies in the main checkout's logs/, ruling D7 of docs/tasks/T-0084d.md)
STORED = {
    "t0080c-dev": ("logs/t0080-toolcall-dev-0dc9134", "*-toolcall-dev.jsonl"),
    "t0080c-test": ("logs/t0080-toolcall-test-0dc9134", "*-toolcall-test.jsonl"),
    "t0082c-dev": (
        "logs/t0082-replay-dev-48313a9",
        "*-toolcall-dev.jsonl",
    ),
    "t0082c-test": (
        "logs/t0082-replay-test-48313a9",
        "*-toolcall-test.jsonl",
    ),
}


def stored(key: str) -> Path:
    rel, pattern = STORED[key]
    d = _main_checkout() / rel
    arts = sorted(d.glob(pattern))
    assert len(arts) == 1, (
        f"the stored {key} artifact is not at {d} (fixture: a cc-only artifact of record)"
    )
    return arts[0]
