"""T-0201 group D pin moves (docs/tasks/T-0201.md D3; test-author, D).

Not a test module. Group D turns the screen reminder on by default in live: ``cli`` passes ``screen_reminder=True``
to every default live ``Session`` and prints one start line ``astra session: screen reminder: [screen_reminder=]on
(default)`` (R-D2 of ``test_t0201_cli_default.py``). Existing rows that compare a live ``cli`` build with an older BASE
move in one of two ways, named per row in the D3 census:

- **Pin off** (rows that drive a ``cli``-built live session through labelled steps and compare request bytes, the
  T-0174 D4b / T-0179 A-5 pattern): the working tree's run appends ``--screen-reminder off`` (``REMINDER_OFF_PIN``)
  unless the case names the flag itself, so the session is BASE's; the one ``off (flag)`` line it prints is removed
  (required present exactly once) and a ``screen_reminder: False`` keyword is set aside (D1 allows ``False`` or no
  keyword for off). Used through ``test_t0152_overview_frame_default._cli_run(..., pin_reminder_off=True)``.
- **Admit the default** (rows that compare only printed lines and ``Session`` keywords, never a step request): the
  observed side drops exactly the default line (at most once per printed text, and only in a live run) and a
  ``screen_reminder: True`` keyword of a live ``Session`` (its described ``tool_kit`` has ``mode == "live"``); a truthy
  keyword outside live is kept, so those rows still see a leak.

On a tree without the flag (a BASE run in a subprocess, or before group D's implementation) ``tree_has_flag()`` is
false and every helper is the identity, so the rows stay green at BASE.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

KW = "screen_reminder"
FLAG = "--screen-reminder"
REMINDER_OFF_PIN = [FLAG, "off"]
#: R-D2's start line, by value and source
DEFAULT_LINE = re.compile(r"^astra session: screen reminder: (?:screen_reminder=)?on \(default\)$")
OFF_FLAG_LINE = re.compile(r"^astra session: screen reminder: (?:screen_reminder=)?off \(flag\)$")


def tree_has_flag() -> bool:
    """Whether the imported ``local_astra.cli`` (the tree under test) defines ``--screen-reminder`` (read statically
    from that file, so a BASE subprocess, which imports BASE's ``cli``, answers for BASE)."""
    from local_astra import cli

    return _defines_flag(str(cli.__file__))


@functools.lru_cache(maxsize=4)
def _defines_flag(path: str) -> bool:
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and any(isinstance(a, ast.Constant) and a.value == FLAG for a in node.args)
        ):
            return True
    return False


def names_reminder(flags: list[str], body: str) -> bool:
    """A case that sets the reminder itself is left as it is (the pin would override it)."""
    return any("screen-reminder" in f for f in flags) or KW in body


def _drop_lines(text: str, pattern: re.Pattern[str], where: str, *, exactly: int | None) -> str:
    lines = text.splitlines(keepends=True)
    found = [i for i, x in enumerate(lines) if pattern.match(x.rstrip("\n"))]
    if exactly is not None:
        assert len(found) == exactly, (
            f"T-0201 D3 ({where}): {pattern.pattern!r} printed {exactly} time(s); found at lines {found}"
        )
    else:
        assert len(found) <= 1, (
            f"T-0201 D1 ({where}): the screen reminder line printed twice; {found}"
        )
    return "".join(x for i, x in enumerate(lines) if i not in found)


def drop_off_pin_line(text: str, where: str) -> str:
    """Pin off: ``text`` without the one ``off (flag)`` line the pin prints (exactly one required)."""
    return _drop_lines(text, OFF_FLAG_LINE, where, exactly=1)


def kwargs_without_off(kwargs: Any) -> Any:
    """Pin off: a ``Session`` keyword mapping (or list of them) without a ``screen_reminder: False`` key; a ``True``
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
    """Admit: a ``Session`` keyword mapping (or list of them) without ``screen_reminder: True`` where that ``Session``
    is live; unchanged on a tree without the flag."""
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
