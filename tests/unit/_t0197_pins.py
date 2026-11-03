"""T-0197 pin moves (docs/tasks/T-0197.md, *Pin moves*): the session serves the HUD page by default.

Three helpers, so the ~119 existing rows that start a session keep asserting what they asserted at their BASE:

- ``opt_out()``: the argv the shared builder (``_tools_switch_support.session_argv``) inserts, ``["--no-serve-hud"]``,
  **only on a tree whose ``session`` parser accepts that flag** (the T-0174 ``V1_PIN`` precedent: a BASE tree before
  T-0197, run in a subprocess by a differential, does not, and is run without it). The probe is behavioural: the
  ``session --help`` text of the ``cli`` imported in *this* process names ``--no-serve-hud``. Inside
  ``with hud_default():`` it is ``[]`` so a row can exercise the real default.
- ``NonBindingHud`` / ``install_non_binding_hud(mp)``: a stand-in for ``hud_static.HudStaticServer`` that binds
  nothing. ``_tools_switch_support.stop_before_serving`` installs it, so no fast-suite real start binds the profile's
  fixed ``hud_port`` (1420) once the default is on. Every construction is recorded on the class.
- ``admit_hud_line(run, kind)``: a "defaults byte-identical to BASE" capture taken at the real default (no opt-out).
  The one added stdout line must be present exactly once at criterion 2's position and is removed: ``dry``: the last
  stdout line is ``astra session: the HUD page would be served at <url>``; ``start``: the first line after the
  ``astra session: regime:`` line is ``astra session: the HUD page is at <url>``. Any other difference still fails
  the row. On a tree without the opt-out (BASE) it is never called.

Each admit fails (``AssertionError``) when the change it admits is absent, so a moved row is red until T-0197 lands.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

NO_SERVE_HUD = "--no-serve-hud"
SERVE_HUD = "--serve-hud"
IS_AT = "astra session: the HUD page is at "
WOULD_BE = "astra session: the HUD page would be served at "
REGIME_PREFIX = "astra session: regime:"

_probe: dict[str, bool] = {}
_default_depth = [0]


def tree_has_opt_out() -> bool:
    """Whether the ``cli`` imported here parses ``--no-serve-hud`` (``session --help`` names it). Cached per file."""
    from local_astra import cli

    key = str(getattr(cli, "__file__", ""))
    if key not in _probe:
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            contextlib.suppress(SystemExit),
        ):
            cli.main(["session", "--help"])
        _probe[key] = NO_SERVE_HUD in out.getvalue()
    return _probe[key]


@contextlib.contextmanager
def hud_default() -> Iterator[None]:
    """Inside, ``opt_out()`` is empty: the shared argv builder leaves the HUD default alone."""
    _default_depth[0] += 1
    try:
        yield
    finally:
        _default_depth[0] -= 1


def opt_out() -> list[str]:
    if _default_depth[0] or not tree_has_opt_out():
        return []
    return [NO_SERVE_HUD]


class NonBindingHud:
    """``HudStaticServer(root, port)`` that binds nothing; constructions and closes are recorded on the class."""

    built: ClassVar[list[tuple[Any, int]]] = []
    closed: ClassVar[list[int]] = []

    def __init__(self, root: Any, port: int, *a: Any, **k: Any) -> None:
        type(self).built.append((root, int(port)))
        self.port = int(port)
        self.address = ("127.0.0.1", self.port)

    def serve_forever(self) -> None:
        return None

    def close(self) -> None:
        type(self).closed.append(self.port)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"T-0197 pin move: NonBindingHud stands in for HudStaticServer and has no {name!r}"
        )


def install_non_binding_hud(mp: pytest.MonkeyPatch) -> None:
    from local_astra import hud_static

    NonBindingHud.built = []
    NonBindingHud.closed = []
    mp.setattr(hud_static, "HudStaticServer", NonBindingHud)


def _rejoin(text: str, lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if text.endswith("\n") and lines else "")


def admit_hud_out(out: str, kind: str) -> str:
    """``out`` without the one HUD line T-0197 adds at its position (criterion 2); exactly one is required."""
    prefix = WOULD_BE if kind == "dry" else IS_AT
    lines = out.splitlines()
    found = [i for i, x in enumerate(lines) if x.startswith(prefix)]
    assert len(found) == 1, (
        f"T-0197 pin move ({kind}): exactly one {prefix!r}... line by default; found {len(found)}\n{out}"
    )
    i = found[0]
    url = lines[i][len(prefix) :]
    assert url.startswith("http://127.0.0.1:") and "?ws=ws://127.0.0.1:" in url, (
        f"T-0197 criterion 1: the line carries the full page URL; {lines[i]!r}"
    )
    if kind == "dry":
        assert i == len(lines) - 1, (
            f"T-0197 criterion 2: under --dry-run the would-be-served line is the last stdout line; at {i} of "
            f"{len(lines)}"
        )
    else:
        assert i > 0 and lines[i - 1].startswith(REGIME_PREFIX), (
            f"T-0197 criterion 2: at a real start the HUD line is the first line after the regime line; previous "
            f"{lines[i - 1] if i else None!r}"
        )
    return _rejoin(out, lines[:i] + lines[i + 1 :])


def admit_hud_line(run: dict[str, Any], kind: str) -> dict[str, Any]:
    """A captured run (``{"code", "out", "err", ...}``) with the one HUD stdout line admitted and removed."""
    moved = dict(run)
    moved["out"] = admit_hud_out(run["out"], kind)
    return moved
