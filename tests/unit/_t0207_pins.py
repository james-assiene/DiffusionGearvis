"""T-0207 group C pin moves (docs/tasks/T-0207.md C3, R-C3; test-author, C).

Not a test module. Group C turns the completion check on by default in live: ``cli`` passes
``completion_check=True`` to every default live ``Session`` and prints one start line ``astra session: completion
check: on (default)`` (C1; R-C1: in the live block, after the screen reminder line, before ``search:``). Existing rows
that pinned the old default (no keyword, no line) move in one of two ways, named per row in the C3 census:

- **Pin off** (rig and differential rows: a ``cli``-built live session driven through steps and compared with an
  older BASE, byte for byte; under the default they gain a ``completion check outcome: no_frame`` note): the working
  tree's live run appends ``--completion-check off`` (``CHECK_OFF_PIN``) unless the case names the flag itself or is
  not live, so the session is BASE's; the one ``off (flag)`` line it prints is removed (required present exactly
  once) and a ``completion_check: False`` keyword is set aside (C1 allows ``False`` or no keyword for off).
- **Admit the default** (rows that compare only printed lines and ``Session`` keywords): the observed side drops
  exactly the default line (at most once per printed text, and only in a live run) and a ``completion_check: True``
  keyword of a live ``Session`` (its described ``tool_kit`` has ``mode == "live"``); a truthy keyword outside live is
  kept, so those rows still see a leak.

On a tree without the flag (a BASE run in a subprocess, or before group C's implementation) ``tree_has_flag()`` is
false and every helper is the identity (the pin appends nothing), so the rows stay green there.

``tree_has_flag`` reads the *imported* ``cli``'s code objects (a function that calls ``add_argument`` with the flag
as a constant), not the file, so it answers for the module actually loaded.
"""

from __future__ import annotations

import functools
import re
import types
from collections.abc import Mapping
from typing import Any

KW = "completion_check"
FLAG = "--completion-check"
CHECK_OFF_PIN = [FLAG, "off"]
#: C1's start line, by value and source
DEFAULT_LINE = re.compile(r"^astra session: completion check: on \(default\)$")
OFF_FLAG_LINE = re.compile(r"^astra session: completion check: off \(flag\)$")


def _code_defines_flag(code: types.CodeType) -> bool:
    if "add_argument" in code.co_names and FLAG in code.co_consts:
        return True
    return any(isinstance(c, types.CodeType) and _code_defines_flag(c) for c in code.co_consts)


@functools.lru_cache(maxsize=4)
def _module_defines_flag(module_id: int) -> bool:
    from local_astra import cli

    return any(
        isinstance(v, types.FunctionType)
        and v.__module__ == cli.__name__
        and _code_defines_flag(v.__code__)
        for v in vars(cli).values()
    )


def tree_has_flag() -> bool:
    """Whether the imported ``local_astra.cli`` defines ``--completion-check``."""
    from local_astra import cli

    return _module_defines_flag(id(cli))


def names_check(flags: list[str], body: str = "") -> bool:
    """A case that sets the completion check itself is left as it is (the pin would override it)."""
    return any(f.startswith(FLAG) for f in flags) or KW in body


def _is_live(flags: list[str], live_prefix: bool) -> bool:
    """Whether a run is live: its last ``--tools`` value, else the prefix's (``live_prefix``)."""
    value = None
    for i, f in enumerate(flags):
        if f == "--tools" and i + 1 < len(flags):
            value = flags[i + 1]
        elif f.startswith("--tools="):
            value = f.split("=", 1)[1]
    return value == "live" if value is not None else live_prefix


def pins(flags: list[str], body: str = "", *, live_prefix: bool = True) -> bool:
    """Whether ``pin_off`` appends the flag for this run."""
    return tree_has_flag() and not names_check(flags, body) and _is_live(flags, live_prefix)


def pin_off(flags: list[str], body: str = "", *, live_prefix: bool = True) -> list[str]:
    """Pin off: ``flags`` with ``--completion-check off`` appended for a live run that does not name the flag."""
    if pins(flags, body, live_prefix=live_prefix):
        return [*flags, *CHECK_OFF_PIN]
    return list(flags)


def _drop_lines(text: str, pattern: re.Pattern[str], where: str, *, exactly: int | None) -> str:
    lines = text.splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if pattern.match(x.rstrip("\n"))]
    if exactly is not None:
        assert len(found) == exactly, (
            f"T-0207 C3 ({where}): {pattern.pattern!r} printed {exactly} time(s); found at lines {found}"
        )
    else:
        assert len(found) <= 1, (
            f"T-0207 C1 ({where}): the completion check line printed twice; {found}"
        )
    return "".join(x for i, x in enumerate(lines) if i not in found)


def drop_off_pin_line(text: str, where: str) -> str:
    """Pin off: ``text`` without the one ``off (flag)`` line the pin prints (exactly one required)."""
    return _drop_lines(text, OFF_FLAG_LINE, where, exactly=1)


def section_without_off_pin(section: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Pin off over one T-0134c ``_other_modes`` section of a run the pin applied to (``pins(...)`` was true): the
    one ``off (flag)`` line out of the ``dry`` and ``build`` outputs (required present exactly once in each that
    exited 0) and a ``completion_check: False`` keyword set aside; every other key as it is (T-0207 C3 re-pin)."""
    out = dict(section)
    for key in ("dry", "build"):
        run = dict(out[key])
        if run.get("code") == 0:
            run["out"] = drop_off_pin_line(run["out"], f"{where}/{key}")
        out[key] = run
    out["session_kwargs"] = kwargs_without_off(out["session_kwargs"])
    return out


def kwargs_without_off(kwargs: Any) -> Any:
    """Pin off: a ``Session`` keyword mapping (or list of them) without a ``completion_check: False`` key; a ``True``
    (the pin failed) is kept, so the row reddens."""
    if isinstance(kwargs, Mapping):
        return {k: v for k, v in kwargs.items() if not (k == KW and v is False)}
    if isinstance(kwargs, (list, tuple)):
        return type(kwargs)(kwargs_without_off(k) for k in kwargs)
    return kwargs


def _is_live_kwargs(kw: Mapping[str, Any]) -> bool:
    kit = kw.get("tool_kit")
    mode = kit.get("mode") if isinstance(kit, Mapping) else getattr(kit, "mode", None)
    return mode == "live"


def kwargs_without_default(kwargs: Any) -> Any:
    """Admit: a ``Session`` keyword mapping (or list of them) without ``completion_check: True`` where that
    ``Session`` is live; unchanged on a tree without the flag."""
    if not tree_has_flag():
        return kwargs
    if isinstance(kwargs, Mapping):
        if kwargs.get(KW) is True and _is_live_kwargs(kwargs):
            return {k: v for k, v in kwargs.items() if k != KW}
        return dict(kwargs)
    if isinstance(kwargs, (list, tuple)):
        return type(kwargs)(kwargs_without_default(k) for k in kwargs)
    return kwargs


def _live_text(text: str) -> bool:
    """A printed run is live if it says so: the tools line (``--dry-run``) or the grounding line (both)."""
    return any(
        x.startswith("tools=live") or x.startswith("astra session: grounding:")
        for x in text.splitlines()
    )


def text_without_default(text: str, where: str) -> str:
    """Admit: printed text without the default line (at most once; only removed from a live run's text)."""
    if not tree_has_flag() or not _live_text(text):
        return text
    return _drop_lines(text, DEFAULT_LINE, where, exactly=None)


def admit_default_line(text: str, where: str) -> str:
    """Admit, for a row that knows its run is a default live run: exactly one default line removed."""
    if not tree_has_flag():
        return text
    return _drop_lines(text, DEFAULT_LINE, where, exactly=1)


def lines_without_default(lines: list[str], where: str) -> list[str]:
    """Admit over a list of printed lines (a live run's): the default line removed (at most once)."""
    if not tree_has_flag():
        return list(lines)
    found = [i for i, x in enumerate(lines) if DEFAULT_LINE.match(x)]
    assert len(found) <= 1, f"T-0207 C1 ({where}): the completion check line printed twice; {found}"
    return [x for i, x in enumerate(lines) if i not in found]


def capture_without_default(capture: Any, where: str = "capture") -> Any:
    """Admit over a whole captured section tree (nested mappings and lists): every ``session_kwargs`` entry through
    ``kwargs_without_default``, every string under an ``out`` key through ``text_without_default``; nothing else."""
    if isinstance(capture, Mapping):
        out: dict[str, Any] = {}
        for k, v in capture.items():
            if k == "session_kwargs":
                out[k] = kwargs_without_default(v)
            elif k == "out" and isinstance(v, str):
                out[k] = text_without_default(v, where)
            else:
                out[k] = capture_without_default(v, f"{where}/{k}")
        return out
    if isinstance(capture, list):
        return [capture_without_default(v, f"{where}[{i}]") for i, v in enumerate(capture)]
    return capture
