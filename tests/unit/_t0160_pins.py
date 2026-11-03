"""T-0160 group C pin moves (docs/tasks/T-0160.md C-1 named spec change, "T-0160 ruling 2: web on by default in
live"): what web mode on by default in live adds to an existing BASE-differential row, admitted exactly and nothing
else.

- ``web_line(base_live)``: BASE's live ``--dry-run`` tools line with its one ``egress=loopback`` word read as
  ``egress=web`` (C-2). Every other byte of the line is BASE's.
- ``admit_web_line(out)``: a live ``--dry-run`` stdout. Exactly one ``tools=live`` line, and it must say
  ``egress=web`` (never ``egress=loopback``); it is put back to BASE's word and the stdout returned. Any other
  difference still fails the row that compares the result with BASE.
- ``admit_web_executor(kwargs)``: the described ``BrowserToolExecutor`` keywords of a default live real start. Each
  must carry ``web is True`` and ``start_origin`` equal to its one allowed origin (C-1: ``_start_live`` passes
  ``web=…`` and ``start_origin=pages.origin``); both are dropped, nothing else is.
- ``admit_live_run(run, kind)``: both of the above over a ``t0152._cli_run``-shaped capture (``dry``: the tools line;
  ``start``: the executor keywords). These rigs stop before serving, so their stderr must not hold the C-3 notice
  (it prints once the start page is open, after the T-0125 startup line's pattern; ``test_cli_tools_web.py``).

``Profile.browser_web`` is admitted the T-0137 way instead (subtracted when present, green before and after); its
type and default are pinned exactly by ``test_cli_tools_web.py``.

Each helper fails (``AssertionError``) when the change it admits is absent, so a moved row is red until group C lands.

T-0178 (amendment A8): web mode's budget becomes 20. Once the tree under test has ``Profile.max_tool_steps_web``
(``web_budget_present``), ``web_line`` also reads BASE's ``loop=on(8)`` as ``loop=on(20)`` and ``admit_web_line``
(hence ``admit_live_run``) strictly puts exactly one ``loop=on(20)`` back to ``loop=on(8)`` (``admit_web_loop``); a
duplicated, missing or other-valued loop word fails. Before group A they behave exactly as above. The positive pin
that web mode says 20 is ``test_t0178_budget.py`` (pending), so these helpers never decide that A landed.
"""

from __future__ import annotations

from typing import Any

EGRESS_LOOPBACK = " egress=loopback "
EGRESS_WEB = " egress=web "
TOOLS_LIVE = "tools=live "
#: C-3's clauses; a rig stopped before serving prints none of them
NOTICE_MARK = "can open public websites"


#: T-0178 A1/A3 (amendment A8): web mode's default budget is 20, BASE's (and every other mode's) 8
LOOP_BASE = " loop=on(8) "
LOOP_WEB = " loop=on(20) "


def web_budget_present() -> bool:
    """T-0178 A8: the "selector present" detection (the T-0174 D4b pattern): the ``local_astra`` under test has
    ``Profile.max_tool_steps_web``. Before group A every helper below behaves exactly as at BASE."""
    from local_astra.config import Profile

    return "max_tool_steps_web" in Profile.model_fields


def web_loop_line(line: str) -> str:
    """T-0178 A3: a BASE web-live tools line as group A prints it: its one ``loop=on(8)`` read as ``loop=on(20)``,
    only once ``Profile.max_tool_steps_web`` exists (else unchanged)."""
    if not web_budget_present():
        return line
    assert line.count(LOOP_BASE) == 1 and " loop=on(" not in line.replace(LOOP_BASE, " ", 1), (
        f"premise: a BASE live tools line with exactly one loop=on(8); {line!r}"
    )
    return line.replace(LOOP_BASE, LOOP_WEB)


def admit_web_loop(out: str) -> str:
    """T-0178 A8: a default web-live ``--dry-run`` stdout with its tools line's ``loop=on(20)`` put back to BASE's
    ``loop=on(8)``. Strict: exactly one live tools line, exactly one ``loop=on(`` on it, and it must be ``20``; a
    duplicated, missing or other-valued line fails. Inert (returns ``out``) while ``Profile.max_tool_steps_web`` is
    absent, so BASE-tree runs are untouched."""
    if not web_budget_present():
        return out
    lines = out.splitlines()
    at = [i for i, x in enumerate(lines) if x.startswith(TOOLS_LIVE)]
    assert len(at) == 1, f"T-0178 A8: one live tools line under --dry-run; found {len(at)}:\n{out}"
    line = lines[at[0]]
    assert line.count(" loop=on(") == 1 and line.count(LOOP_WEB) == 1, (
        f"T-0178 A3/A8: a default web-live --dry-run line says loop=on(20), once; {line!r}"
    )
    lines[at[0]] = line.replace(LOOP_WEB, LOOP_BASE)
    return "\n".join(lines) + ("\n" if out.endswith("\n") else "")


def web_line(base_live: str) -> str:
    assert base_live.startswith(TOOLS_LIVE) and base_live.count(EGRESS_LOOPBACK) == 1, (
        f"premise: a BASE live tools line with one egress=loopback; {base_live!r}"
    )
    return web_loop_line(base_live.replace(EGRESS_LOOPBACK, EGRESS_WEB))  # T-0178 A3 (A8)


def admit_web_line(out: str) -> str:
    out = admit_web_loop(out)  # T-0178 A8: loop=on(20) -> loop=on(8), strictly, once group A exists
    lines = out.splitlines()
    at = [i for i, x in enumerate(lines) if x.startswith(TOOLS_LIVE)]
    assert len(at) == 1, f"T-0160 C-2: one live tools line under --dry-run; found {len(at)}:\n{out}"
    line = lines[at[0]]
    assert line.count(EGRESS_WEB) == 1 and EGRESS_LOOPBACK not in line, (
        f"T-0160 C-1/C-2 (ruling 2): a default live --dry-run line says egress=web; {line!r}"
    )
    lines[at[0]] = line.replace(EGRESS_WEB, EGRESS_LOOPBACK)
    return "\n".join(lines) + ("\n" if out.endswith("\n") else "")


def admit_web_executor(kwargs: list[Any]) -> list[Any]:
    assert kwargs, "premise: a real live start builds the executor"
    out = []
    for kw in kwargs:
        assert isinstance(kw, dict), kw
        origins = kw.get("allowed_origins")
        assert kw.get("web") is True, (
            f"T-0160 C-1 (ruling 2): a default live start builds the executor with web=True; {kw}"
        )
        assert (
            isinstance(origins, list) and len(origins) == 1 and kw.get("start_origin") == origins[0]
        ), f"T-0160 C-1: start_origin is the start-page origin (the one allowed origin); {kw}"
        out.append({k: v for k, v in kw.items() if k not in ("web", "start_origin")})
    return out


def admit_live_run(run: dict[str, Any], kind: str) -> dict[str, Any]:
    got = dict(run)
    assert NOTICE_MARK not in str(run["err"]), (
        f"T-0160 C-3: the web notice prints only once the session has started; this {kind} rig never serves"
    )
    if kind == "dry":
        got["out"] = admit_web_line(str(run["out"]))
    else:
        got["executor_kwargs"] = admit_web_executor(list(run["executor_kwargs"]))
    return got
