"""T-0172 group B pin moves (docs/tasks/T-0172.md B9, B7; re-scopes R2-R12): what the default-on announced-call
re-ask adds to an existing BASE-differential row, admitted exactly and nothing else.

With defaults, in both tools-on modes (``dry_run``, ``live``), T-0172 adds:

- one startup line, exactly ``REASK_LINE``, the last tools-on line: under ``--dry-run`` no ``astra session:`` line
  follows it; at a real start the regime line follows it immediately (``admit_reask_line``);
- one ``Session`` keyword, ``reask_announced_calls=True`` (``admit_reask_kwargs``);
- on every turn line of such a session that sends no re-ask, ``extras["announced_call_reask"] ==
  {"on": True, "asked": 0, "outcome": None}`` (``admit_reask_turn_key``);
- one ``Profile`` field, ``announced_call_reask: bool | None = None`` (``pop_reask_field``).

Tools off adds none of them (``assert_no_reask``). Each helper fails (``AssertionError``) when the change it admits is
absent or differs in any byte, so a moved row cannot pass by stripping something else, and a row that sees the change
where T-0172 does not make it still fails.
"""

from __future__ import annotations

import json
from typing import Any

REASK_PREFIX = "astra session: reask: "
REASK_LINE = REASK_PREFIX + "announced_call_reask=on (default)"
SESSION_PREFIX = "astra session: "
REGIME_PREFIX = "astra session: regime:"
KWARG = "reask_announced_calls"
TURN_KEY = "announced_call_reask"
DEFAULT_TURN = {"on": True, "asked": 0, "outcome": None}
FIELD = "announced_call_reask"


def _rejoin(text: str, lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if text.endswith("\n") and lines else "")


def admit_reask_line(out: str, kind: str) -> str:
    """``out`` without the one default reask line. ``kind``: ``"dry"`` (``--dry-run``: no ``astra session:`` line
    after it) or ``"start"`` (the regime line right after it)."""
    lines = out.splitlines()
    found = [i for i, x in enumerate(lines) if x.startswith(REASK_PREFIX)]
    assert len(found) == 1 and lines[found[0]] == REASK_LINE, (
        f"T-0172 B9 pin move: exactly one {REASK_LINE!r} in {kind} stdout; found "
        f"{[lines[i] for i in found]}"
    )
    i = found[0]
    after = lines[i + 1 :]
    if kind == "dry":
        assert not [x for x in after if x.startswith(SESSION_PREFIX)], (
            f"T-0172 B9: under --dry-run the reask line is the last tools-on line; after it {after}"
        )
    else:
        assert after and after[0].startswith(REGIME_PREFIX), (
            f"T-0172 B9: at a real start the regime line follows the reask line; next {after[:1]}"
        )
    return _rejoin(out, lines[:i] + after)


def assert_no_reask(out: str) -> None:
    assert REASK_PREFIX not in out, f"T-0172 B9: tools off prints no reask line:\n{out}"


def admit_reask_kwargs(kwargs: list[Any]) -> list[Any]:
    """Every described ``Session`` keyword mapping carries ``reask_announced_calls=True``; it is dropped."""
    assert kwargs, "premise: a build made a Session"
    out = []
    for kw in kwargs:
        assert isinstance(kw, dict) and kw.get(KWARG) is True, (
            f"T-0172 B9 pin move: a default tools-on build passes {KWARG}=True; got {kw.get(KWARG, 'absent') if isinstance(kw, dict) else kw!r}"
        )
        out.append({k: v for k, v in kw.items() if k != KWARG})
    return out


def assert_no_reask_kwargs(kwargs: list[Any]) -> None:
    for kw in kwargs:
        assert not (isinstance(kw, dict) and KWARG in kw), (
            f"T-0172 B9: tools off passes no {KWARG}; {kw}"
        )


def admit_reask_run(run: dict[str, Any], kind: str) -> dict[str, Any]:
    """A tools-on ``cli`` run capture (``out``, ``session_kwargs``): the line and the keyword admitted."""
    got = dict(run)
    got["out"] = admit_reask_line(str(run["out"]), kind)
    if run.get("session_kwargs"):
        got["session_kwargs"] = admit_reask_kwargs(list(run["session_kwargs"]))
    return got


def admit_reask_other(case: str, key: str, value: Any) -> Any:
    """One section of ``t0134c._other_modes`` (cases ``off``, ``dry_run``, ``dry_run-no-loop``, ``live-profile-off``,
    ``live-profile-dry_run``): a case whose resolved mode is ``dry_run`` has the line under ``--dry-run`` (``dry``)
    and at a build (``build``), and the keyword (``session_kwargs``); an ``off`` case has none. Other keys pass."""
    on = "dry_run" in case
    if key in ("dry", "build"):
        if not on:
            assert_no_reask(str(value["out"]))
            return value
        return {
            **value,
            "out": admit_reask_line(str(value["out"]), "dry" if key == "dry" else "start"),
        }
    if key == "session_kwargs":
        if not on:
            assert_no_reask_kwargs(list(value))
            return value
        return admit_reask_kwargs(list(value))
    return value


def admit_reask_turn_key(tree: Any) -> Any:
    """``tree`` without every ``extras["announced_call_reask"]`` equal to ``DEFAULT_TURN``; at least one must be
    there, and the key anywhere else, with any other value, or inside a string, fails."""
    found: list[str] = []

    def walk(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == TURN_KEY:
                    assert path and path[-1] == "extras" and v == DEFAULT_TURN, (
                        f"T-0172 B7 pin move: {TURN_KEY} only in turn-line extras, equal to {DEFAULT_TURN}; at "
                        f"{'/'.join(path)}: {v!r}"
                    )
                    found.append("/".join(path))
                    continue
                out[k] = walk(v, (*path, str(k)))
            return out
        if isinstance(node, list):
            return [walk(v, (*path, str(i))) for i, v in enumerate(node)]
        if isinstance(node, str):
            assert TURN_KEY not in node, (
                f"T-0172 B7 pin move: {TURN_KEY} inside a string at {'/'.join(path)}"
            )
        return node

    out = walk(tree, ())
    assert found, (
        "T-0172 B7 pin move: a default tools-on session writes announced_call_reask on its turn lines"
    )
    return out


def pop_reask_field(fields: dict[str, Any]) -> dict[str, Any]:
    """``t0133._fields``-shaped field descriptions without ``announced_call_reask``, which must be ``bool | None``
    defaulting to ``None`` (not required)."""
    assert FIELD in fields, f"T-0172 B9 pin move: Profile.{FIELD} exists; fields {sorted(fields)}"
    desc = fields[FIELD]
    assert json.dumps(desc) == json.dumps(["bool | None", "None", False]), (
        f"T-0172 B9: Profile.{FIELD} is bool | None = None; got {desc!r}"
    )
    return {k: v for k, v in fields.items() if k != FIELD}


#: T-0174 A3 (R-A1..R-A5): the one Profile field group A adds
LIVE_GUIDANCE_FIELD = "live_guidance"


def pop_live_guidance_field(fields: dict[str, Any]) -> dict[str, Any]:
    """T-0174 R-A1..R-A5 pin move (the ``pop_reask_field`` pattern): ``t0133._fields``-shaped field descriptions
    without ``live_guidance``, which must be ``str | None`` defaulting to ``None`` (not required)."""
    name = LIVE_GUIDANCE_FIELD
    assert name in fields, f"T-0174 A3 pin move: Profile.{name} exists; fields {sorted(fields)}"
    desc = fields[name]
    assert json.dumps(desc) == json.dumps(["str | None", "None", False]), (
        f"T-0174 A3: Profile.{name} is str | None = None; got {desc!r}"
    )
    return {k: v for k, v in fields.items() if k != name}
