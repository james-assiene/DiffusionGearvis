"""T-0202 group D pin moves (docs/tasks/T-0202.md D3; test-author, D).

Not a test module. Group D turns the wait tool on by default in live: a default live ``cli`` build gets the wait set
(``tools_v1_wait.json``, or ``tools_v1_overview_wait.json`` with ``--page-overview``), ``Session(tool_guidance=...)``
gains `` <H-T202-4 sentence>``, and the ``--dry-run`` tools line and the real start's regime line gain the token
`` wait=on(default)``. Existing rows that pinned the old default (off) move in one of two ways, named per row in the
D3 census:

- **Pin off** (rows that compare a ``cli``-built live session with an older BASE, byte for byte): the working tree's
  live run appends ``--wait-tool off`` (``WAIT_OFF_PIN``) unless the case names the flag itself or is not live, so
  the session is BASE's. ``--wait-tool off`` prints nothing and passes no keyword (B9), so nothing is set aside.
- **Admit the default** (rows that pin a default live tools line or guidance text, not a BASE differential): the
  observed side drops exactly the token `` wait=on(default)`` and reads the wait set's sha12 back as the v1 (or
  overview) set's, and the guidance's one trailing `` <sentence>`` is removed. Each admit requires what it removes
  to be there exactly once, so a row still sees the default being absent.

On a tree where the live default is off (BASE, or before group D's implementation) ``default_on()`` is false: the
admit helpers are the identity, and the pin-off still appends the flag (B's flag exists since c9ecd1db's ancestor
e8c2449d; ``tree_has_flag()`` guards older trees).
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
from pathlib import Path

FLAG = "--wait-tool"
WAIT_OFF_PIN = [FLAG, "off"]
DEFAULT_TOKEN = "wait=on(default)"
#: H-T202-4, typed from the task
SENTENCE = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)


def _toolcall() -> Path:
    from local_astra.toolcall import tools

    return Path(tools.__file__).resolve().parent


def _sha12(name: str) -> str:
    return hashlib.sha256((_toolcall() / name).read_bytes()).hexdigest()[:12]


def default_live_sha(page_overview: bool = False) -> str:
    """The full sha256 of the set a default live kit comes from: v1 (or overview) with the default off, the wait (or
    overview+wait) set with it on (T-0202 D1, B8's table)."""
    name = "tools_v1_overview" if page_overview else "tools_v1"
    if default_on():
        name += "_wait"
    return hashlib.sha256((_toolcall() / f"{name}.json").read_bytes()).hexdigest()


def tree_has_flag() -> bool:
    """Whether the imported ``local_astra.cli`` defines ``--wait-tool`` (read statically from that file)."""
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


def default_on() -> bool:
    """Whether the tree under test turns the wait tool on in a live session with no ``--wait-tool`` flag (B's pure
    resolver ``cli.wait_tool_of`` asked with an empty namespace)."""
    from local_astra import cli

    resolver = getattr(cli, "wait_tool_of", None)
    return bool(resolver is not None and resolver(argparse.Namespace(), "live"))


def names_wait(flags: list[str], body: str = "") -> bool:
    """A case that sets the wait tool itself is left as it is (the pin would override it)."""
    return any(f.startswith(FLAG) for f in flags) or "wait_tool" in body


def _is_live(flags: list[str], live_prefix: bool) -> bool:
    """Whether a run is live: its last ``--tools`` value, else the prefix's (``live_prefix``)."""
    value = None
    for i, f in enumerate(flags):
        if f == "--tools" and i + 1 < len(flags):
            value = flags[i + 1]
        elif f.startswith("--tools="):
            value = f.split("=", 1)[1]
    return value == "live" if value is not None else live_prefix


def pin_off(flags: list[str], body: str = "", *, live_prefix: bool = True) -> list[str]:
    """Pin off: ``flags`` with ``--wait-tool off`` appended for a live run that does not name the flag (T-0202 D3)."""
    if tree_has_flag() and not names_wait(flags, body) and _is_live(flags, live_prefix):
        return [*flags, *WAIT_OFF_PIN]
    return list(flags)


def admit_line(line: str, where: str = "") -> str:
    """Admit: one printed line without the `` wait=on(default)`` token (required once when the line carries any
    ``wait=``) and with the wait sets' sha12 read back as v1's / overview's. Identity when the default is off."""
    if not default_on():
        return line
    parts = line.split(" ")
    hits = [p for p in parts if p.startswith("wait=")]
    if not hits:
        return line
    assert hits == [DEFAULT_TOKEN], (
        f"T-0202 D3 ({where}): one {DEFAULT_TOKEN} token; got {hits} in {line!r}"
    )
    line = " ".join(p for p in parts if p != DEFAULT_TOKEN)
    return line.replace(
        _sha12("tools_v1_overview_wait.json"), _sha12("tools_v1_overview.json")
    ).replace(_sha12("tools_v1_wait.json"), _sha12("tools_v1.json"))


def admitted_tools_lines(out: str, where: str = "") -> list[str]:
    """Admit: ``test_cli_t0131.tools_lines`` with ``admit_line`` applied to each ``tools=`` line."""
    return [admit_line(x, where) for x in out.splitlines() if x.startswith("tools=")]


def admit_text(text: str, where: str = "") -> str:
    """Admit over printed text: ``admit_line`` on every line; at most one line carries the token."""
    if not default_on():
        return text
    lines = text.split("\n")
    assert sum(1 for x in lines if DEFAULT_TOKEN in x.split(" ")) <= 1, (
        f"T-0202 D3 ({where}): {DEFAULT_TOKEN} printed more than once\n{text}"
    )
    return "\n".join(admit_line(x, where) for x in lines)


def admit_guidance(text: str | None, where: str = "") -> str | None:
    """Admit: a live default's ``tool_guidance`` (or a system message) without its one trailing `` <sentence>``
    (required present exactly once). Identity when the default is off or ``text`` is ``None``."""
    if not default_on() or text is None:
        return text
    suffix = " " + SENTENCE
    assert text.count(SENTENCE) == 1, (
        f"T-0202 D3 ({where}): a default live guidance carries the H-T202-4 sentence once; got {text[-300:]!r}"
    )
    assert text.endswith(suffix), (
        f"T-0202 D3 ({where}): the sentence is the suffix; got {text[-300:]!r}"
    )
    return text[: -len(suffix)]


def admit_system(text: str, where: str = "") -> str:
    """Admit inside a system message: the one `` <sentence>`` removed wherever it sits (required exactly once)."""
    if not default_on():
        return text
    assert text.count(" " + SENTENCE) == 1, (
        f"T-0202 D3 ({where}): a default live system message carries the sentence once"
    )
    return text.replace(" " + SENTENCE, "", 1)
