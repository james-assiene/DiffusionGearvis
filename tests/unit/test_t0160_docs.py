"""T-0160 group C, C-8 to C-10: the docs (docs/tasks/T-0160.md).

C-10 rows (docs/02 §5 spec delta and §9 sentence, the docs/01 §3 status sentence, the docs/14 §7 item 3 status
line) are the implementer's; C-8 (CLAUDE.md) and C-9 (README.md) are orchestrator-applied after T-0125 and T-0160 C
merge, so those rows stay ``pending`` after group C is activated (the task's *Tests to write*). The expected words
are written out from the criteria. Each row reads one section, so a sentence elsewhere does not satisfy it.

**Readings:** "docs/14 §7 item 3" is read as the ``## Adoption plan`` section (no numbered §7 heading exists at
BASE); docs/01 §3 is the section whose heading starts ``## 3.``; docs/02 §5 and §9 are ``## 5.`` and ``## 9.``.

Regime: CPU, static text.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

#: C-8/C-9 (README, CLAUDE.md) were orchestrator-applied at 3ea5af61; their rows are active since then

CLAUDE_SENTENCE = (
    "**Scoped exception, web (T-0160, James 2026-10-26):** in a live session (unless `--no-web`), the agent "
    "browser's own page traffic may reach public `https` hosts on port 443, only through the session's egress proxy "
    "(`local_astra/computer/egress_proxy.py`, bound to 127.0.0.1). The proxy resolves each host once and refuses "
    "loopback (other than the start-page origin), private, link-local, CGNAT, multicast, reserved and documentation "
    "addresses, every other port, and plain `http`. No service binds beyond loopback. `off`, `dry_run`, live with "
    "`--no-web`, and the fast suite never touch the network through this path."
)
CLAUDE_COMMAND = (
    "uv run diffusion_gearvis session --tools live   # real https websites by default through the egress proxy "
    "(--no-web for loopback only); fresh profile, no passwords or card numbers typed, no downloads/uploads, Chromium "
    "sandbox required (T-0160)"
)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _section(path: Path, heading: re.Pattern[str]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, x in enumerate(lines) if heading.match(x)]
    assert len(starts) == 1, (
        f"premise: one {heading.pattern!r} heading in {path.name}; found {len(starts)}"
    )
    i = starts[0]
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[i:j])


def test_docs02_section5_spec_delta() -> None:
    """C-10: docs/02 §5 holds a "Spec delta (T-0160; supersedes H-T111c-E for web mode)" paragraph naming the default
    and ``--no-web``, the proxy rules and route split, the guard and its ``failed``, the keys, logging, the sandbox,
    and the unchanged confirmation."""
    sec = _section(DOCS / "02-architecture.md", re.compile(r"^## 5\. "))
    at = sec.find("Spec delta (T-0160; supersedes H-T111c-E for web mode)")
    assert at >= 0, "T-0160 C-10: the §5 spec delta paragraph is absent"
    para = _collapse(sec[at:].split("\n\n", 1)[0] if "\n\n" in sec[at:] else sec[at:])
    for word in (
        "--no-web",
        "egress proxy",
        "443",
        "continue_",
        "failed",
        "Ctrl+T",
        "Ctrl+L",
        "Ctrl+V",
        "egress_log",
        "sandbox",
        "ask-first and honour-no apply unchanged; no new confirmation guard",
    ):
        assert word in para, f"T-0160 C-10: the §5 paragraph does not mention {word!r}"


def _t0160_delta() -> str:
    """docs/02 §5 from the T-0160 spec delta to the next spec delta (or the section end), whitespace-collapsed."""
    sec = _section(DOCS / "02-architecture.md", re.compile(r"^## 5\. "))
    at = sec.find("Spec delta (T-0160")
    assert at >= 0, "T-0160 C-10: the §5 spec delta is absent"
    rest = sec[at + 1 :]
    nxt = rest.find("Spec delta (")
    return _collapse(sec[at : at + 1 + nxt] if nxt >= 0 else sec[at:])


def test_docs02_section5_amendment_residuals() -> None:
    """Amendment (reviewer A): the §5 delta names the residual of plain-text password or PIN boxes (typed into),
    states that clipboard writes are blocked in web mode, and says ``--no-web`` "launches exactly as T-0111g's"."""
    delta = _t0160_delta()
    low = delta.casefold()
    assert "pin" in low.split() or re.search(r"\bpin\b", low), "amendment: the PIN residual"
    assert re.search(r"plain[- ]text", low) and "password" in low, (
        "amendment: plain-text password/PIN boxes"
    )
    assert re.search(
        r"clipboard[^.]*\bblocked\b|\bblocked\b[^.]*clipboard|cannot write the clipboard", low
    ), "amendment: clipboard writes are blocked (or: a page cannot write the clipboard)"
    assert "launches exactly as T-0111g's" in delta, "amendment: 'launches exactly as T-0111g's'"


def test_docs02_section9_names_search() -> None:
    """Amendment (reviewer A): §9 names T-0125 web search as the other path off the box."""
    sec = _collapse(_section(DOCS / "02-architecture.md", re.compile(r"^## 9\. ")))
    at = sec.find("T-0160")
    assert at >= 0, "premise: the §9 web sentence"
    window = sec[max(0, at - 600) : at + 900]
    assert "T-0125" in window and "search" in window.casefold(), (
        f"amendment: §9 names T-0125 search; {window!r}"
    )


def test_docs02_section9_web_exception() -> None:
    """C-10: docs/02 §9 (security and privacy) gets the web exception sentence."""
    sec = _collapse(_section(DOCS / "02-architecture.md", re.compile(r"^## 9\. ")))
    assert "T-0160" in sec and "--no-web" in sec and "egress proxy" in sec, (
        "T-0160 C-10: §9 states the web exception (T-0160, the egress proxy, --no-web)"
    )


def test_docs01_section3_status_sentence() -> None:
    """C-10: docs/01 §3 gains one status sentence for T-0160 (the pinned sentences stay verbatim:
    ``test_t0111g_docs.py::test_pinned_sentences_still_verbatim``, unedited)."""
    sec = _section(DOCS / "01-design.md", re.compile(r"^## 3\. "))
    assert sec.count("T-0160") >= 1 and "--no-web" in sec, (
        "T-0160 C-10: the docs/01 §3 status sentence"
    )


def test_docs14_item3_status_line() -> None:
    """C-10: docs/14's adoption plan records the ruling (web on by default) and the stated-risk decision."""
    sec = _collapse(_section(DOCS / "14-computer-use-harness.md", re.compile(r"^## Adoption plan")))
    assert "T-0160" in sec, "T-0160 C-10: the docs/14 status line"
    assert re.search(r"T-0160[^.]*\b(?:stated|risk)\b", sec), (
        "the status line records the stated-risk decision"
    )


def test_makefile_web_targets_outside_check() -> None:
    """Owned files (group C, Makefile): ``browser-web`` runs the B-7 module and ``browser-web-network`` the C-5 module
    with ``GEARVIS_WEB_NETWORK=1``; neither is reachable from ``check`` (C-5: never in ``make check``)."""
    text = (REPO / "Makefile").read_text(encoding="utf-8")

    def recipe(target: str) -> str:
        m = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n?)*)", text, re.M)
        assert m, f"T-0160: no Makefile target {target}"
        return m.group(0)

    assert "tests/integration/test_computer_web_browser.py" in recipe("browser-web")
    net = recipe("browser-web-network")
    assert (
        "tests/integration/test_computer_web_network.py" in net and "GEARVIS_WEB_NETWORK=1" in net
    ), net
    check = recipe("check")
    assert "browser-web" not in check, f"C-5: neither web target is in check; {check}"


def test_claude_md_web_exception_and_commands() -> None:
    """C-8 (orchestrator): the scoped exception sentence follows the search one; the commands list gains the web line;
    every existing "loopback start pages only" line gets "(with `--no-web`)". The orchestrator ruled (2026-10-27)
    that CLAUDE.md has one such line, not the spec's two: at least one, each carrying it."""
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert CLAUDE_SENTENCE in text, "T-0160 C-8: the web exception sentence"
    search = text.find("**Scoped exception, web search")
    assert 0 <= search < text.find(CLAUDE_SENTENCE), "T-0160 C-8: after the search exception"
    assert CLAUDE_COMMAND in text, "T-0160 C-8: the commands line"
    loop = [x for x in text.splitlines() if "loopback start pages only" in x]
    assert len(loop) >= 1 and all("(with `--no-web`)" in x for x in loop), loop


def test_readme_real_websites_section() -> None:
    """C-9 (orchestrator): the can/cannot text, the ``--no-web`` options row, and the "Real websites" section with the
    injection figure, the guards, "do not sign in", and the sandbox subsection with the AppArmor commands."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    flat = _collapse(text)
    assert re.search(r"^\|\s*`--no-web`", text, re.M), (
        "T-0160 C-9: an options-table row for --no-web"
    )
    assert re.search(r"^#+ Real websites\s*$", text, re.M), "T-0160 C-9: a 'Real websites' section"
    assert re.search(r"^#+ Real websites: the browser sandbox\s*$", text, re.M), (
        "the sandbox subsection"
    )
    for words in (
        "on our injection test page the model followed planted instructions in 4 of 20 turns (T-0138 S14)",
        "do not sign in to anything in its window",
        "apparmor_parser -r",
        "public https websites",
    ):
        assert words.casefold() in flat.casefold(), f"T-0160 C-9: README lacks {words!r}"
