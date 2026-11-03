#!/usr/bin/env python3
"""Generate the README's "Options at a glance" tables from the settings catalog (T-0221).

Usage: ``uv run python scripts/gen_readme_options.py [--check]``

:func:`render` returns the text between ``<!-- options:begin -->`` and ``<!-- options:end -->`` (markers
excluded): one table per catalog group, in ``GROUPS`` order, one row per entry in ``CATALOG`` order. :func:`main`
rewrites that block of ``README.md`` in place; with ``--check`` it writes nothing and exits 1 if the block is stale.
The catalog (``local_astra/settings_catalog.py``) is the only source of the text; edit it there, then re-run this.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from local_astra.settings_catalog import CATALOG, CATALOG_REGIME, GROUPS, SettingEntry

BEGIN = "<!-- options:begin -->"
END = "<!-- options:end -->"
README = Path(__file__).resolve().parents[1] / "README.md"
COMMAND = "uv run python scripts/gen_readme_options.py"

_HEADER = "| Setting | Default | Useful when | Not useful when | Changes |\n|---|---|---|---|---|\n"


def _cell(text: str) -> str:
    """One table cell: a single line, with ``|`` escaped so it cannot split the row."""
    return " ".join(text.split()).replace("|", "\\|")


def _changes(entry: SettingEntry) -> str:
    """How the setting changes: live from the panel, or a restart with the flag that sets it."""
    if entry.kind == "fixed":
        return "fixed: not a setting"
    if entry.v1_changeable:
        return (
            f"live from the panel (at launch: `{entry.flag}`)"
            if entry.flag
            else "live from the panel"
        )
    if entry.relaunch is not None:
        if entry.flag:
            return f"needs a restart: `{entry.flag}`"
        if entry.profile_key:
            return f"needs a restart: `{entry.profile_key}` in astra.toml"
        return "needs a restart"
    return "live from its own checkboxes"


def _row(entry: SettingEntry) -> str:
    cells = (
        entry.label,
        entry.default_note,
        entry.useful_when,
        entry.not_useful_when,
        _changes(entry),
    )
    return "| " + " | ".join(_cell(c) for c in cells) + " |\n"


def render(catalog: Sequence[SettingEntry]) -> str:
    """The README block between the two markers, markers excluded (it starts and ends with a newline): the
    catalog's regime (:data:`CATALOG_REGIME`), then one table per group."""
    parts: list[str] = [
        "\n",
        f"The numbers were measured under this regime: {CATALOG_REGIME} Some runs differ; the panel's "
        "evidence names them.\n\n",
    ]
    for group in GROUPS:
        rows = [e for e in catalog if e.group == group]
        if not rows:
            continue
        parts.append(f"**{group}**\n\n")
        parts.append(_HEADER)
        parts.extend(_row(e) for e in rows)
        parts.append("\n")
    return "".join(parts)


def _splice(text: str, block: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"README.md: the markers {BEGIN} and {END} are missing or out of order")
    return text[: start + len(BEGIN)] + block + text[end:]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    text = README.read_text(encoding="utf-8")
    new = _splice(text, render(CATALOG))
    if check:
        if new != text:
            print(f"README.md options table is stale; run: {COMMAND}", file=sys.stderr)
            return 1
        return 0
    if new != text:
        README.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
