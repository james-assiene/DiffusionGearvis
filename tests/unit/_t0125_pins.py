"""T-0125 pin moves (docs/tasks/T-0125.md, *Pins*, amendment 2026-10-27): what default-on live search adds to an
existing BASE-differential row, admitted exactly and nothing else.

- ``admit_live_run(run, kind)``: a live CLI capture (``t0152._cli_run``'s dict). The one added stdout line
  ``astra session: search: ddgs (default)`` must appear exactly once at criterion 3's position (``dry``: after every
  tools-block line, i.e. the last line with no ``--serve-hud``; ``start``: immediately before the regime line) and is
  removed. At a real start (``start``) also: no stderr startup line (amendment 2: it follows ``_serve``), and each
  described ``tool_kit`` must carry ``search == "<DdgsSearch>"`` and ``web_search == "ddgs"``, which are put back to
  BASE's ``search: None`` with ``web_search`` dropped. Any other difference still fails the row.
- ``kit_without_web_search(kwargs)``: described ``Session`` keywords outside a search-on session; each described
  ``tool_kit`` must carry ``web_search`` with the value ``None`` (the new field, criterion 3), which is dropped.

Each helper fails (``AssertionError``) when the change it admits is absent, so a moved row is red until T-0125 lands.
"""

from __future__ import annotations

from typing import Any

SEARCH_DEFAULT = "astra session: search: ddgs (default)"
REGIME_PREFIX = "astra session: regime:"
STARTUP = (
    "astra session: web search is on (keyless: duckduckgo, then brave, then mojeek): each search query the model "
    "makes is sent off this machine and logged in the session log; results are untrusted web text; --search off "
    "turns it off"
)


def _remove_line(text: str, line: str, where: str) -> tuple[list[str], int]:
    lines = text.splitlines()
    found = [i for i, x in enumerate(lines) if x == line]
    assert len(found) == 1, (
        f"T-0125 pin move: exactly one {line[:60]!r}... in {where}; found {len(found)}"
    )
    i = found[0]
    return lines[:i] + lines[i + 1 :], i


def _rejoin(text: str, lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if text.endswith("\n") and lines else "")


def admit_search_line(out: str, kind: str) -> str:
    lines, i = _remove_line(out, SEARCH_DEFAULT, f"{kind} stdout")
    if kind == "dry":
        assert i == len(lines), (
            f"T-0125 criterion 3: under --dry-run the search line follows every tools-block line (last); at {i} of "
            f"{len(lines) + 1}"
        )
    else:
        assert i < len(lines) and lines[i].startswith(REGIME_PREFIX), (
            f"T-0125 criterion 3: at a real start the search line comes immediately before the regime line; "
            f"next {lines[i] if i < len(lines) else None!r}"
        )
    return _rejoin(out, lines)


def admit_startup_line(err: str) -> str:
    lines, _ = _remove_line(err, STARTUP, "start stderr")
    return _rejoin(err, lines)


def _kits(kwargs: list[Any]) -> list[Any]:
    return [kw.get("tool_kit") if isinstance(kw, dict) else None for kw in kwargs]


def admit_live_kit(kwargs: list[Any]) -> list[Any]:
    out = []
    for kw in kwargs:
        kit = kw.get("tool_kit") if isinstance(kw, dict) else None
        if isinstance(kit, dict):
            assert kit.get("web_search") == "ddgs" and kit.get("search") == "<DdgsSearch>", (
                f"T-0125 criterion 3: a default live kit carries search=DdgsSearch and web_search='ddgs'; "
                f"{ {k: kit.get(k) for k in ('search', 'web_search')} }"
            )
            kit = {k: v for k, v in kit.items() if k != "web_search"}
            kit["search"] = None
            kw = {**kw, "tool_kit": kit}
        out.append(kw)
    return out


def kit_without_web_search(kwargs: list[Any]) -> list[Any]:
    out = []
    for kw in kwargs:
        kit = kw.get("tool_kit") if isinstance(kw, dict) else None
        if isinstance(kit, dict):
            assert "web_search" in kit and kit["web_search"] is None, (
                f"T-0125 criterion 3: ToolKit gains web_search (None with search off); kit keys {sorted(kit)}"
            )
            kw = {**kw, "tool_kit": {k: v for k, v in kit.items() if k != "web_search"}}
        out.append(kw)
    return out


def admit_live_run(run: dict[str, Any], kind: str) -> dict[str, Any]:
    got = dict(run)
    got["out"] = admit_search_line(str(run["out"]), kind)
    if kind == "start":
        # amendment 2: the startup line prints only after ``_serve`` has started; these rigs stop before serving,
        # so their stderr gains nothing (and is then compared with BASE's as before)
        assert STARTUP not in str(run["err"]), (
            "T-0125 amendment 2: a start stopped before serving prints no startup line"
        )
        got["session_kwargs"] = admit_live_kit(list(run["session_kwargs"]))
    return got
