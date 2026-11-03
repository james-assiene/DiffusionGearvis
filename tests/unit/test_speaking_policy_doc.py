"""T-0104: the speaking policy for computer use (H-T104, docs/01 D16), as doc anchors.

Every row reads repo files only (no git, no network, no file is ever written). Each check is a pure function
of a string, scoped to the paragraph, clause, section or line the criterion names, so a keyword that survives
elsewhere in a document cannot satisfy it. Every adversarial row runs the *same* function on a mutated string;
every guard has a row showing it accepts the approved text (too strict) and rows showing it rejects the
mutations criterion 9 names (too wide).

Oracles:
- The D16 text is criterion 1's blockquote in ``docs/tasks/T-0104.md`` (approved verbatim, H-T104 (a)).
- Criterion 5's file list is written out by hand below and cross-checked against the parsed criterion.
- The safety and prompt shas, and the pre-T-0104 shas of every owned document, were computed once from the
  tree at 2b08f12 (identical to ``main`` at 30c52f3 for every file hashed here); they are byte-identity
  guards (criteria 6-8), not captured behaviour.

Activated at the start of /implement T-0104 (markers removed); the guards (safety text, prompt texts, thresholds,
and every parser self-test) are unmarked and pass at 2b08f12.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "docs/tasks/T-0104.md"
DOC01 = ROOT / "docs/01-design.md"
DOC02 = ROOT / "docs/02-architecture.md"
DOC03 = ROOT / "docs/03-test-plan.md"
T0080D = ROOT / "docs/tasks/T-0080d.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
THRESHOLDS = ROOT / "local_astra/bench/thresholds.py"
TWOPASS = ROOT / "local_astra/eval/cd/twopass.py"

DATE = "2026-10-16"
SECTION3_SENTENCE = (
    "Executing tool calls for real additionally requires the confirmation gate of D16 (T-0107)."
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_ws(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------------------------------------------
# Recorded shas (computed once from the tree at 2b08f12; see the module docstring).
# --------------------------------------------------------------------------------------------------------------

#: Criterion 8: docs/02 §4's P1-P9 lines (``- **P1.**`` .. the ``- **P9.**`` line), joined by "\n".
P1_P9_SHA256 = "8cd81728ed66ba6f4b2be7cff82538a78837252d72c5e5abb31d9bd255aa418f"
#: Criterion 8: T-0080d's H4, H5 and H8 items (each from its ``- **Hn.`` line to the next ``- **H`` line).
T0080D_H_SHA256 = {
    "H4": "195f346bd3cdec8a34c6e6c8260a2b378a98e784ec49134170a9e23e51af0305",
    "H5": "9a05aa2401407fb4980810a3fa0883f7e2286f8ddfe540a26e3abc76e005a25f",
    "H8": "646c176ea122dc0a787cc0721e961fc91ec1e1d156fa77febf5a41a38bd8224f",
}
#: Criterion 8: T-0080d's settled default 3 (``3. **Speech.**`` up to ``4. **``).
T0080D_DEFAULT3_SHA256 = "affdf48c8839ef02b482577845d9cb9dd32f9d1e3c2880b120590ce9c49ba6f4"
#: Criterion 8: CLAUDE.md's bullet that holds "Never voice text the LLM has not committed…".
CLAUDE_NEVER_VOICE_SHA256 = "e017e61471783919153115cd03291fe8b2d733450ed5341992331f9e491e7b11"
#: Criterion 6: ``prompt_builder.SPOKEN_STYLE``.
SPOKEN_STYLE_SHA256 = "7bb891a53a6250fd873ca915d1212be174e023c425ce8a7b2eec991d92083d6d"
#: Criterion 6: the source lines of ``twopass.NATIVE_SYSTEM_PROMPT = ( … )`` (read as text: the fast suite is
#: cd-free, so twopass is not imported here).
#: T-0155 criterion 9 re-pin. Generator (implementation note 3): this module's own ``twopass_prompt_src`` over
#: twopass.py with T-0155's two persona sentences changed (a scratch copy of 48a28e0 with "You are Astra, a voice"
#: -> "You are Gearvis, a voice", twice), i.e.
#: ``sha(twopass_prompt_src(<that copy's local_astra/eval/cd/twopass.py>))``; the pre-T-0155 value is kept below.
TWOPASS_NATIVE_SYSTEM_PROMPT_SRC_SHA256 = (
    "87f4612d3f8450fe8b3e67ce5e95a164ffe055fe5d8d72640112c927402d9f11"
)
#: The same literal's source sha before T-0155 (2b08f12 .. 48a28e0), against which the source with its persona
#: sentence reversed is still compared.
TWOPASS_NATIVE_SYSTEM_PROMPT_SRC_SHA256_PRE_T0155 = (
    "403e115431f1c8507d6960d91f4df6bf43bf59f29da73d9b8fa57a91faab1ab6"
)
#: Criterion 6: ``local_astra/bench/thresholds.py``, whole file.
THRESHOLDS_SHA256 = "0b35393b4e989012b793b1cf0b2807e5b91f551fcb879c5b576efc91553538d6"

#: Criterion 5's nine files, written out by hand from the criterion.
CRITERION5_FILES = (
    "docs/tasks/T-0080d.md",
    "docs/tasks/T-0093.md",
    "docs/tasks/T-0098.md",
    "docs/tasks/T-0098c.md",
    "docs/tasks/T-0103.md",
    "docs/cd/toolcall-live.md",
    "docs/cd/toolcall-native.md",
    "docs/cd/toolcall-native-prompt-ab.md",
    "docs/cd/toolcall-ui-prompt-ab.md",
)


# --------------------------------------------------------------------------------------------------------------
# Parsers (pure; each takes a string).
# --------------------------------------------------------------------------------------------------------------


def approved_d16() -> str:
    """Criterion 1's blockquote in the task file: the D16 paragraph as the human approved it."""
    lines = [ln.strip() for ln in read(TASK).splitlines() if ln.strip().startswith("> **D16.")]
    assert len(lines) == 1, "T-0104 criterion 1 holds exactly one D16 blockquote"
    return lines[0][2:]


def d16_span(doc01: str) -> str:
    """docs/01 from ``**D16.`` (at a line start) to the next ``**D`` paragraph or ``## `` heading."""
    starts = [m.start() for m in re.finditer(r"(?m)^\*\*D16\.", doc01)]
    assert len(starts) == 1, f"docs/01 must hold exactly one D16 paragraph, found {len(starts)}"
    start = starts[0]
    nxt = re.compile(r"(?m)^(\*\*D\d|## )").search(doc01, start + 1)
    return doc01[start : nxt.start() if nxt else len(doc01)].strip()


def d16_moments(span: str) -> tuple[str, str, str, str, str]:
    """Split D16 into its numbered moments (1)-(4) and the rest (the routine/supersedes/live sentences).

    Moment 4 ends at the end of its own sentence; the rest is everything after it."""
    pos = []
    for k in (1, 2, 3, 4):
        m = re.search(rf"\({k}\) ", span)
        assert m is not None, f"D16 lacks moment ({k})"
        pos.append(m.start())
    assert pos == sorted(pos), "D16's moments (1)-(4) are out of order"
    end4 = span.find(". ", pos[3])
    end4 = len(span) if end4 < 0 else end4 + 1
    return (
        span[pos[0] : pos[1]],
        span[pos[1] : pos[2]],
        span[pos[2] : pos[3]],
        span[pos[3] : end4],
        span[end4:],
    )


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=\.)\s+(?=[A-Z(*`])", text) if s.strip()]


def check_five_moments(span: str) -> None:
    c1, c2, c3, c4, rest = d16_moments(span)
    for needle in (
        "*Start:*",
        "one short sentence per request",
        "the HUD action card is the acknowledgement",
        "optional",
    ):
        assert needle in c1, f"D16 moment (1) lacks {needle!r}"
    for needle in (
        "*Before anything that cannot be undone*",
        "placing an order",
        "paying or transferring money",
        "sending, posting or publishing",
        "deleting beyond undo",
        "discarding unsaved work",
        "creating an account or a booking",
        "changing a password or a security setting",
        "asks",
        "explicit yes",
        "later user turn",
        "the request's own words are not that yes",
        "treats it as irreversible",
    ):
        assert needle in c2, f"D16 moment (2) (irreversible) lacks {needle!r}"
    for needle in ("*When stuck or unsure*", "asks the user to choose", "it does not guess"):
        assert needle in c3, f"D16 moment (3) lacks {needle!r}"
    for needle in ("*At the end*", "reports the outcome", "whether anything irreversible happened"):
        assert needle in c4, f"D16 moment (4) lacks {needle!r}"
    rest = rest.strip()
    routine = [s for s in sentences(rest) if s.startswith("Routine steps")]
    assert len(routine) == 1, "D16 has one sentence starting 'Routine steps'"
    assert re.search(r"\bare not narrated\b", routine[0]), "D16: routine steps are not narrated"
    assert "HUD" in routine[0], "D16: routine progress is shown on the HUD"
    sup = [s for s in sentences(rest) if "supersedes" in s]
    assert len(sup) == 1, "D16 has one 'supersedes' sentence"
    for needle in (
        "for routine steps",
        "a spoken sentence before every tool call",
        "T-0093 criterion 3",
    ):
        assert needle in sup[0], f"D16's supersedes sentence lacks {needle!r}"
    assert re.search(r"reported, never gated, and has no lower bound", rest), (
        "D16: prefix rate is report-only"
    )
    live = [s for s in sentences(rest) if "`tools = live`" in s]
    assert len(live) == 1 and "confirmation gate (T-0107)" in live[0], (
        "D16: live requires T-0107's gate"
    )


_VOICE = re.compile(r"\bvoic", re.IGNORECASE)
_NEGATOR = re.compile(r"\b(no|never|not|nothing|none)\b", re.IGNORECASE)
_PERMISSIVE = re.compile(
    r"\b(may|can|could|should|will|shall|might|is allowed to|are allowed to|is permitted to|are permitted to|"
    r"permits?|allows?)\b[^.;:]*\bvoic",
    re.IGNORECASE,
)


def clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[.;:](?:\s|$)", text) if c.strip()]


def check_speech_invariants(span: str) -> None:
    assert "T-0080d H4" in span, "D16 must name T-0080d H4"
    assert "docs/02 §4 P6" in span, "D16 must name docs/02 §4 P6"
    assert "history holds only what was spoken (H8)" in span, "D16 must keep H8"
    prohibitions = [
        c
        for c in clauses(span)
        if re.search(r"\b(never voiced|ever voiced)\b", c)
        and _NEGATOR.search(c)
        and "text built from a call" in c
        and "uncommitted text" in c
        and "thought" in c
    ]
    assert prohibitions, (
        "D16 must say no uncommitted text, thought or text built from a call is ever voiced"
    )
    for c in clauses(span):
        if _VOICE.search(c):
            assert not _PERMISSIVE.search(c), f"D16 clause permits voicing: {c.strip()!r}"
            assert _NEGATOR.search(c), (
                f"D16 clause about voicing is not a prohibition: {c.strip()!r}"
            )


def section(doc: str, heading_prefix: str) -> str:
    """From the line starting with ``heading_prefix`` to the next ``## `` heading."""
    m = re.search(rf"(?m)^{re.escape(heading_prefix)}", doc)
    assert m is not None, f"no heading {heading_prefix!r}"
    nxt = re.compile(r"(?m)^## ").search(doc, m.end())
    return doc[m.start() : nxt.start() if nxt else len(doc)]


def check_section3(doc01: str) -> None:
    s3 = section(doc01, "## 3. ")
    paras = [p for p in re.split(r"\n\s*\n", s3) if "computer use" in p]
    assert len(paras) == 1, "docs/01 §3 has one computer-use paragraph"
    assert SECTION3_SENTENCE in paras[0], (
        "docs/01 §3's computer-use clause lacks the confirmation-gate sentence"
    )


def doc02_delta_block(doc02: str) -> tuple[int, int, list[str]]:
    """The T-0104 spec-delta block of docs/02 as (first line index, end index, lines): its header line and the
    bullet or indented lines that follow it, blank lines inside the block allowed."""
    lines = doc02.splitlines()
    heads = [i for i, ln in enumerate(lines) if "Spec delta (T-0104, H-T104)" in ln]
    assert len(heads) == 1, (
        f"docs/02 must hold exactly one 'Spec delta (T-0104, H-T104)' header, found {len(heads)}"
    )
    i = heads[0]
    j = i + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() == "":
            k = j
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            if k < len(lines) and (lines[k].startswith("- ") or lines[k].startswith(" ")):
                j = k
                continue
            break
        if j > i and not (ln.startswith("- ") or ln.startswith(" ")):
            break
        j += 1
    return i, j, lines


def doc02_units(block: str) -> list[str]:
    """Sentences of the delta block (a bullet line or a prose sentence; a ';' does not split)."""
    return [u for u in re.split(r"\n|(?<=\.)\s+(?=[A-Z\x60(*])", block) if u.strip()]


def check_doc02_delta(doc02: str) -> None:
    s5 = section(doc02, "## 5.")
    assert "Spec delta (T-0104, H-T104)" in s5, "the T-0104 delta is in docs/02 §5"
    assert s5.find("Spec delta (T-0091") < s5.find("Spec delta (T-0104"), (
        "the T-0104 delta follows T-0091's"
    )
    i, j, lines = doc02_delta_block(doc02)
    block = "\n".join(lines[i:j])
    assert "D16" in block, "the delta says tool-turn speech follows docs/01 D16"
    sp = [c for c in doc02_units(block) if "tool.call.spoken_prefix" in c]
    assert sp and any(re.search(r"\bnot a target\b", c) for c in sp), (
        "spoken_prefix is logged, not a target"
    )
    q = [c for c in doc02_units(block) if "P9" in c]
    assert q and any("no `tool` commit" in c or "toolless" in c for c in q), (
        "questions are toolless turns, P9"
    )
    inv = [c for c in doc02_units(block) if "P1–P9" in c and "R1–R10" in c]
    assert inv and any("unchanged" in c for c in inv), (
        "the delta says P1–P9 and R1–R10 are unchanged"
    )
    lv = [c for c in doc02_units(block) if "`live`" in c and "T-0107" in c]
    assert lv and any("refused" in c for c in lv), "`live` stays refused until T-0107"
    assert re.search(r"never weaken", block), (
        "T-0107's rows may add properties but never weaken any"
    )


def q13_span(doc03: str) -> str:
    m = re.search(r"(?m)^(?:[-*] )?\**Q13\b", doc03)
    assert m is not None, "docs/03 has no Q13"
    nxt = re.compile(r"(?m)^((?:[-*] )?\**Q1[4-9]\b|## )").search(doc03, m.end())
    return doc03[m.start() : nxt.start() if nxt else len(doc03)]


def check_q13(doc03: str) -> None:
    s8 = section(doc03, "## 8. ")
    span = q13_span(doc03)
    assert span in s8, "Q13 is in docs/03 §8"
    assert s8.find("Q12 ") < s8.find(span), "Q13 follows Q12"
    labels = [f"Q13{c}" for c in "abcdefg"]
    pos = [span.find(lab) for lab in labels]
    assert all(p >= 0 for p in pos), (
        f"Q13 lacks {[lab for lab, p in zip(labels, pos, strict=False) if p < 0]}"
    )
    assert pos == sorted(pos), "Q13a–Q13g are out of order"
    items = {
        lab: span[p : (pos[k + 1] if k + 1 < len(pos) else len(span))]
        for k, (lab, p) in enumerate(zip(labels, pos, strict=False))
    }
    assert "commit_call" in items["Q13a"] and "T-0106" in items["Q13a"], (
        "Q13a: commit_call; a gate only by T-0106"
    )
    assert re.search(r"\byes\b", items["Q13b"]) and re.search(r"\bno\b", items["Q13b"]), (
        "Q13b: yes and no"
    )
    assert re.search(r"ambiguous", items["Q13c"]), "Q13c: ask rate on ambiguous cases"
    assert re.search(r"routine", items["Q13d"]) and re.search(r"recall", items["Q13d"]), (
        "Q13d: over-asking"
    )
    assert re.search(r"spoken.prefix", items["Q13e"]) and "distribution" in items["Q13e"], (
        "Q13e: distribution"
    )
    for lab in ("Q13f", "Q13g"):
        assert "multi-step loop" in items[lab], f"{lab} needs the multi-step loop"
    assert re.search(r"voiced anomalies = 0", span) and "J17" in span and "P6" in span, (
        "Q13's safety row"
    )
    assert re.search(r"(?i)regime", span), "Q13 carries a regime line"
    assert not re.search(r"[≥≤]\s*\d", span), "Q13 is definitions only: no threshold"


def delta_lines(text: str) -> list[int]:
    return [
        i for i, ln in enumerate(text.splitlines()) if "H-T104" in ln and "D16" in ln and DATE in ln
    ]


def _first(lines: list[str], pred, start: int = 0) -> int:  # type: ignore[no-untyped-def]
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    return len(lines)


def _heading(prefix: str):  # type: ignore[no-untyped-def]
    return lambda ln: ln.startswith(prefix)


def _next_h2(ln: str) -> bool:
    return ln.startswith("## ")


def delta_window(rel: str, lines: list[str]) -> tuple[int, int]:
    """The (lo, hi) line window criterion 5 places the delta line in: lo < index < hi."""
    if rel == "docs/tasks/T-0080d.md":
        return -1, len(lines)
    if rel == "docs/tasks/T-0093.md":
        ac = _first(lines, _heading("## Acceptance criteria"))
        lo = _first(lines, _heading("3. "), ac)
        return lo, _first(lines, lambda ln: ln.startswith("4. ") or ln.startswith("## "), lo + 1)
    if rel == "docs/tasks/T-0098.md":
        lo = _first(lines, _heading("## Result and human ruling"))
        return lo, _first(lines, _next_h2, lo + 1)
    if rel == "docs/tasks/T-0098c.md":
        lo = _first(lines, _heading("## Follow-on: T-0098d"))
        return lo, _first(lines, _next_h2, lo + 1)
    if rel == "docs/tasks/T-0103.md":
        lo = _first(lines, _heading("## Rulings"))
        return lo, _first(lines, _next_h2, lo + 1)
    if rel == "docs/cd/toolcall-live.md":
        fu = _first(lines, _heading("## Follow-ups"))
        lo = _first(lines, _heading("- Native prompt tuning for speech"), fu)
        return lo, _first(lines, lambda ln: ln.startswith("- ") or ln.startswith("#"), lo + 1)
    if rel == "docs/cd/toolcall-native.md":
        lo = _first(lines, lambda ln: "**Product finding (Q6).**" in ln)
        return lo, _first(lines, _next_h2, lo + 1)
    if rel in ("docs/cd/toolcall-native-prompt-ab.md", "docs/cd/toolcall-ui-prompt-ab.md"):
        lo = _first(lines, _heading("**Status"))
        return lo, _first(lines, _next_h2, lo + 1)
    raise AssertionError(f"no placement rule for {rel}")


def check_delta(rel: str, text: str) -> None:
    found = delta_lines(text)
    assert len(found) == 1, (
        f"{rel}: expected exactly one H-T104/D16/{DATE} line, found {len(found)}"
    )
    lines = text.splitlines()
    i = found[0]
    lo, hi = delta_window(rel, lines)
    assert lo < len(lines), f"{rel}: the anchor criterion 5 names is missing"
    assert lo < i < hi, (
        f"{rel}: the delta line (line {i + 1}) is not where criterion 5 puts it ({lo + 1}..{hi})"
    )
    line = lines[i]
    if rel == "docs/tasks/T-0080d.md":
        assert "Status (T-0104, 2026-10-16)" in line and "H4" in line, (
            "T-0080d: a Status (T-0104) line naming H4"
        )
    else:
        assert "unchanged" in line, (
            f"{rel}: the delta line says results and safety rows are unchanged"
        )
    if rel == "docs/tasks/T-0098c.md":
        assert "7/10" in line and "withdraw" in line, (
            "T-0098c: the T-0098d `type_at` >= 7/10 gate is withdrawn"
        )


def strip_t0104(rel: str, text: str) -> str:
    """Remove exactly what T-0104 adds to ``rel`` (as criteria 1, 2 and 5 describe it)."""
    if rel == "docs/01-design.md":
        span = d16_span(text)
        text = text.replace(span, "", 1)
        assert text.count(" " + SECTION3_SENTENCE) == 1, "docs/01 §3 sentence missing"
        text = text.replace(" " + SECTION3_SENTENCE, "", 1)
        return text
    if rel == "docs/02-architecture.md":
        i, j, lines = doc02_delta_block(text)
        return "\n".join(lines[:i] + lines[j:])
    found = delta_lines(text)
    assert len(found) == 1, f"{rel}: expected exactly one delta line"
    lines = text.splitlines()
    return "\n".join(lines[: found[0]] + lines[found[0] + 1 :])


def rebaseline_rows(task: str) -> list[list[str]]:
    s = task[task.index("### Re-baseline table") :]
    rows = []
    for ln in s.splitlines()[1:]:
        if ln.startswith("|"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif rows:
            break
    assert rows[0][:4] == ["where", "row", "before", "under D16"], "re-baseline table header"
    return rows[2:]


def row_file(where: str) -> str:
    m = re.search(r"`(docs/cd/[\w.-]+\.md)`", where)
    if m:
        return m.group(1)
    m = re.search(r"T-\d{4}[a-z]?", where)
    assert m, f"no file in {where!r}"
    return f"docs/tasks/{m.group(0)}.md"


def row_exempt(under: str) -> bool:
    u = under.replace("**", "").strip()
    return u.startswith("unchanged") or u == "no delta"


def criterion5_files(task: str) -> list[str]:
    a = task.index("5. **Delta lines.**")
    b = task.index("6. ***Negative (product).***")
    out = []
    for name in re.findall(r"`([\w./-]+\.md)`", task[a:b]):
        out.append(f"docs/tasks/{name}" if name.startswith("T-") else name)
    return out


def p1_p9_block(doc02: str) -> str:
    lines = doc02.splitlines()
    a = [i for i, ln in enumerate(lines) if ln.startswith("- **P1.**")]
    b = [i for i, ln in enumerate(lines) if ln.startswith("- **P9.**")]
    assert len(a) == 1 and len(b) == 1 and a[0] < b[0], "docs/02 §4 holds one P1..P9 block"
    return "\n".join(lines[a[0] : b[0] + 1])


def h_item(t0080d: str, h: str) -> str:
    lines = t0080d.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(f"- **{h}.")]
    assert len(starts) == 1, f"T-0080d holds one {h} item"
    end = _first(
        lines,
        lambda ln: ln.startswith("- **H") or ln.startswith("#") or not ln.strip(),
        starts[0] + 1,
    )
    return "\n".join(lines[starts[0] : end])


def default3(t0080d: str) -> str:
    lines = t0080d.splitlines()
    a = _first(lines, _heading("3. **Speech.**"))
    b = _first(lines, _heading("4. **"), a + 1)
    assert a < b < len(lines), "T-0080d settled default 3"
    return "\n".join(lines[a:b])


def claude_never_voice(claude: str) -> str:
    hits = [ln for ln in claude.splitlines() if "Never voice text the LLM has not committed" in ln]
    assert len(hits) == 1, "CLAUDE.md holds one 'Never voice text…' line"
    return hits[0]


def twopass_prompt_src(src: str) -> str:
    lines = src.splitlines()
    a = _first(lines, _heading("NATIVE_SYSTEM_PROMPT = ("))
    b = _first(lines, lambda ln: ln == ")", a + 1)
    assert a < b < len(lines), "twopass.NATIVE_SYSTEM_PROMPT literal"
    return "\n".join(lines[a : b + 1])


# --------------------------------------------------------------------------------------------------------------
# Fixtures built from the approved text (no file is mutated).
# --------------------------------------------------------------------------------------------------------------


def doc01_with(d16: str, section3_extra: str = " " + SECTION3_SENTENCE) -> str:
    """docs/01 as T-0104 would leave it, built from the file at HEAD and the given D16 text: D16 after D15."""
    doc = read(DOC01)
    if "**D16." in doc:  # after implementation: rebuild from the pre-T-0104 text
        doc = strip_t0104("docs/01-design.md", doc)
    s3 = section(doc, "## 3. ")
    para = next(p for p in re.split(r"\n\s*\n", s3) if "computer use" in p)
    doc = doc.replace(para, para.rstrip() + section3_extra, 1)
    d15 = re.search(r"(?m)^\*\*D15\..*$", doc)
    assert d15 is not None
    return doc[: d15.end()] + "\n\n" + d16 + doc[d15.end() :]


def _irreversible_clause_removed(d16: str) -> str:
    a = d16.index("(2) ")
    b = d16.index("(3) ")
    return d16[:a] + d16[b:]


# --------------------------------------------------------------------------------------------------------------
# Criterion 1: D16.
# --------------------------------------------------------------------------------------------------------------


def test_d16_paragraph_carries_the_five_moments() -> None:
    check_five_moments(d16_span(read(DOC01)))


def test_d16_is_the_approved_text_verbatim_after_d15() -> None:
    doc = read(DOC01)
    span = d16_span(doc)
    assert norm_ws(span) == norm_ws(approved_d16()), (
        "criterion 1: D16 verbatim (H-T104 (a) approved as drafted)"
    )
    assert doc.index("**D15.") < doc.index("**D16."), "criterion 1: D16 goes after D15"
    # Orchestrator's ruling (task file at 3db7b56): the last D-paragraph of §5b, after D13, before ``## 6``.
    i16 = doc.index("**D16.")
    assert doc.index("**D13.") < i16 < doc.index("## 6. Key design decisions"), (
        "D16 sits after D13, before §6"
    )
    between = doc[i16 + len(span) : doc.index("## 6. Key design decisions")]
    assert not re.search(r"(?m)^\*\*D\d", between), "D16 is the last D-paragraph of §5b"


def test_five_moments_parser_accepts_the_approved_text() -> None:
    check_five_moments(d16_span(doc01_with(approved_d16())))


def test_five_moments_fails_without_the_irreversible_clause_while_irreversible_survives() -> None:
    """Criterion 9, first mutation: D16's clause (2) deleted; 'irreversible' survives in §3 and in D16's (4)."""
    mutated = _irreversible_clause_removed(approved_d16())
    doc = doc01_with(
        mutated,
        section3_extra=" " + SECTION3_SENTENCE + " An irreversible step is confirmed first.",
    )
    assert "irreversible" in section(doc, "## 3. ")
    assert "irreversible" in d16_span(doc)
    with pytest.raises(AssertionError):
        check_five_moments(d16_span(doc))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("explicit yes in a later user turn", "explicit yes"),
        ("explicit yes in a later user turn", "yes in this turn"),
        ("; the request's own words are not that yes in v1", ""),
        ("; when unsure whether a step can be undone, it treats it as irreversible", ""),
        ("and asks the user to choose", "and picks one"),
        ("; it does not guess", ""),
        ("are not narrated", "are narrated"),
        ("This supersedes", "This keeps"),
        ("(4) *At the end* it reports the outcome", "(4) *At the end* it may stop"),
        ("(1) *Start:*", "*Start:*"),
    ],
)
def test_five_moments_fails_on_each_clause_mutation(old: str, new: str) -> None:
    d16 = approved_d16()
    assert old in d16, f"fixture drift: {old!r} not in the approved D16"
    with pytest.raises(AssertionError):
        check_five_moments(d16_span(doc01_with(d16.replace(old, new, 1))))


def test_five_moments_fails_when_the_irreversible_clause_moves_out_of_d16() -> None:
    """Scope: clause (2) moved verbatim into another paragraph does not count."""
    d16 = approved_d16()
    a, b = d16.index("(2) "), d16.index("(3) ")
    doc = doc01_with(d16[:a] + d16[b:]).replace("\n\n**D13.", "\n\n" + d16[a:b] + "\n\n**D13.", 1)
    assert d16[a:b] in doc
    with pytest.raises(AssertionError):
        check_five_moments(d16_span(doc))


def test_five_moments_fails_when_clause_2_keeps_its_label_and_the_word_but_loses_the_gate() -> None:
    """Criterion 9, first mutation shaped like the ruling: the "(2)" label and "irreversible" survive, and the
    clause's body moves to D13, so only the confirmation gate (explicit yes, later user turn) is gone from D16."""
    d16 = approved_d16()
    a, b = d16.index("(2) "), d16.index("(3) ")
    hollow = d16[:a] + "(2) *Before anything irreversible* it takes care. " + d16[b:]
    doc = doc01_with(hollow).replace("\n\n**D13.", "\n\n" + d16[a + 4 : b] + "\n\n**D13.", 1)
    span = d16_span(doc)
    assert "(2) " in span and "irreversible" in span and "explicit yes" in doc
    with pytest.raises(AssertionError, match=r"moment \(2\)"):
        check_five_moments(span)


def test_d16_keeps_the_speech_invariants() -> None:
    check_speech_invariants(d16_span(read(DOC01)))


def test_speech_invariants_parser_accepts_the_approved_text() -> None:
    check_speech_invariants(approved_d16())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t + " When the user asks, the assistant may voice the call.",
        lambda t: t.replace(
            "history holds only",
            "the assistant can voice text built from a call, and history holds only",
        ),
        lambda t: t.replace("no uncommitted text, thought", "uncommitted text, thought").replace(
            "is ever voiced", "is voiced"
        ),
        lambda t: t.replace("is ever voiced", "is never hidden, and it may be voiced"),
        lambda t: (
            t + " It never hesitates; a confirmation is allowed to voice the call's arguments."
        ),
        lambda t: t.replace("T-0080d H4; ", ""),
        lambda t: t.replace(" or text built from a call", ""),
    ],
    ids=[
        "may-voice-the-call",
        "can-voice",
        "negation-dropped",
        "never-plus-may",
        "allowed-to",
        "no-H4",
        "no-call-text",
    ],
)
def test_speech_invariants_fail_on_a_permissive_or_weakened_span(mutate) -> None:  # type: ignore[no-untyped-def]
    mutated = mutate(approved_d16())
    assert mutated != approved_d16()
    with pytest.raises(AssertionError):
        check_speech_invariants(mutated)


def test_section3_requires_the_confirmation_gate_for_live() -> None:
    check_section3(read(DOC01))


def test_section3_parser_scope() -> None:
    check_section3(doc01_with(approved_d16()))
    elsewhere = doc01_with(approved_d16(), section3_extra="")
    elsewhere = elsewhere.replace(
        "## 6. Key design decisions", "## 6. Key design decisions\n\n" + SECTION3_SENTENCE, 1
    )
    assert SECTION3_SENTENCE in elsewhere
    with pytest.raises(AssertionError):
        check_section3(elsewhere)
    other_para = doc01_with(approved_d16(), section3_extra="").replace(
        "\n\n## 4.", "\n\nA second paragraph of §3. " + SECTION3_SENTENCE + "\n\n## 4.", 1
    )
    # The sentence in a second §3 paragraph (not the computer-use clause) does not count.
    assert other_para.count(SECTION3_SENTENCE) == 1
    with pytest.raises(AssertionError):
        check_section3(other_para)


# --------------------------------------------------------------------------------------------------------------
# Criterion 2: docs/02 §5 spec delta.
# --------------------------------------------------------------------------------------------------------------


def test_doc02_spec_delta_paragraph() -> None:
    check_doc02_delta(read(DOC02))


# --------------------------------------------------------------------------------------------------------------
# Criterion 3: docs/03 Q13 (orchestrator-written).
# --------------------------------------------------------------------------------------------------------------


def test_doc03_q13_definitions_only() -> None:
    check_q13(read(DOC03))


#: A docs/02 delta written from criterion 2's five bullets (their own words), for the parser's self-tests.
_DOC02_DELTA = (
    "**Spec delta (T-0104, H-T104).** Tool-turn speech follows docs/01 D16.\n"
    "- `tool.call.spoken_prefix` stays a logged field; it is not a target.\n"
    "- A confirmation or clarifying question carries no `tool` commit, so the machine handles it as a toolless "
    "turn (P9 unchanged).\n"
    "- The §4 invariants P1–P9 and R1–R10 are unchanged.\n"
    "- `live` stays refused until T-0107's rows exist; those rows may add properties but never weaken any."
)


def _doc02_with(delta: str) -> str:
    doc = read(DOC02)
    if "Spec delta (T-0104, H-T104)" in doc:
        doc = strip_t0104("docs/02-architecture.md", doc)
    return doc.replace("\n## 6. ", "\n" + delta + "\n\n## 6. ", 1)


def test_doc02_parser_accepts_criterion_2_and_rejects_each_missing_bullet() -> None:
    check_doc02_delta(_doc02_with(_DOC02_DELTA))
    for old, new in [
        ("; it is not a target", "; it is a target"),
        ("(P9 unchanged)", "(see the machine)"),
        ("are unchanged", "are revised"),
        ("stays refused", "is allowed"),
        ("but never weaken any", "and may relax some"),
        ("follows docs/01 D16", "follows the prompt"),
    ]:
        assert old in _DOC02_DELTA
        with pytest.raises(AssertionError):
            check_doc02_delta(_doc02_with(_DOC02_DELTA.replace(old, new, 1)))
    # Scope: the same paragraph outside §5 (here, in §6) does not count.
    misplaced = read(DOC02).replace("\n## 7. ", "\n" + _DOC02_DELTA + "\n\n## 7. ", 1)
    assert "Spec delta (T-0104, H-T104)" in misplaced
    with pytest.raises(AssertionError):
        check_doc02_delta(misplaced)


#: A Q13 written from criterion 3's bullets (their own words), for the parser's self-tests. The orchestrator
#: writes the real one; the parser keys on the criterion's terms, not on a layout.
_Q13 = (
    "Q13 *Speaking while acting* (T-0104, docs/01 D16; definitions only, no threshold until T-0106). Regime per "
    "run: profile, canvas length, steps, thinking on, model versions, n, seed.\n"
    "- **Q13a** unconfirmed irreversible proposals: a routed call matching the case's `commit_call` on the first "
    "turn of an irreversible case. Target 0; a hard gate only by T-0106's ruling.\n"
    '- **Q13b** confirmation follow-through: after "yes" the expected call is routed; after "no" no call is '
    "routed.\n"
    "- **Q13c** ask rate on ambiguous cases.\n"
    "- **Q13d** over-asking: ask rate and route recall on routine cases.\n"
    "- **Q13e** routine narration: spoken-prefix length in words and sentences on routine act turns, as a "
    "distribution.\n"
    "- **Q13f** opening acknowledgement per multi-step request (needs the multi-step loop).\n"
    "- **Q13g** end-of-task report (needs the multi-step loop and tool results).\n"
    "- Safety row: voiced anomalies = 0, as J17/P6."
)


def _doc03_without_q13() -> str:
    """docs/03 with the real Q13 block (if any) removed, so the self-tests do not depend on it."""
    doc = read(DOC03)
    if re.search(r"(?m)^(?:[-*] )?\**Q13\b", doc):
        doc = doc.replace(q13_span(doc), "", 1)
    assert "Q13" not in doc, "docs/03 still mentions Q13 after stripping its block"
    return doc


def _doc03_with(q13: str) -> str:
    doc = _doc03_without_q13()
    q12 = next(ln for ln in doc.splitlines() if ln.startswith("Q12 "))
    return doc.replace(q12, q12 + "\n" + q13, 1)


def test_q13_parser_accepts_criterion_3_and_rejects_each_mutation() -> None:
    check_q13(_doc03_with(_Q13))
    for old, new in [
        ("`commit_call`", "call"),
        ('after "no" no call', "after a refusal nothing"),
        ("ambiguous cases", "all cases"),
        ("route recall on routine", "route rate on all"),
        ("as a distribution", "as a mean"),
        ("(needs the multi-step loop and tool results)", "(now)"),
        ("voiced anomalies = 0", "voiced anomalies reported"),
        ("Regime per run", "Per run"),
        ("Target 0;", "Target ≤ 2 per 50;"),
        ("- **Q13c** ask rate on ambiguous cases.\n", ""),
    ]:
        assert old in _Q13, old
        with pytest.raises(AssertionError):
            check_q13(_doc03_with(_Q13.replace(old, new, 1)))
    # Scope: Q13 before Q12 does not count.
    early = _doc03_without_q13().replace("\nQ11 ", "\n" + _Q13 + "\nQ11 ", 1)
    assert _Q13 in early
    with pytest.raises(AssertionError):
        check_q13(early)


# --------------------------------------------------------------------------------------------------------------
# Criteria 4 and 5: the re-baseline table and the delta lines.
# --------------------------------------------------------------------------------------------------------------


def test_criterion5_lists_the_nine_files() -> None:
    assert sorted(criterion5_files(read(TASK))) == sorted(CRITERION5_FILES)


def test_rebaseline_rows_name_criterion5_files() -> None:
    rows = rebaseline_rows(read(TASK))
    live = [r for r in rows if not row_exempt(r[3])]
    assert len(rows) == 10 and len(live) == 7, (len(rows), len(live))
    assert {row_file(r[0]) for r in live} <= set(CRITERION5_FILES)


def test_every_rebaselined_row_has_its_delta() -> None:
    for r in rebaseline_rows(read(TASK)):
        if not row_exempt(r[3]):
            assert delta_lines(read(ROOT / row_file(r[0]))), (
                f"re-baseline row {r[0]!r}: no delta line"
            )
    for rel in CRITERION5_FILES:
        check_delta(rel, read(ROOT / rel))


def _delta_line(rel: str) -> str:
    if rel == "docs/tasks/T-0080d.md":
        return f"**Status (T-0104, {DATE}):** settled default 3 and H4 agree with D16 (H-T104)."
    extra = (
        " withdraws the T-0098d `type_at` ≥ 7/10 gate;" if rel == "docs/tasks/T-0098c.md" else ""
    )
    return f"Delta ({DATE}, H-T104, docs/01 D16):{extra} supersedes the routine-step target; results and safety unchanged."


def _with_delta(rel: str, text: str) -> str:
    """The file with one synthetic delta line placed inside criterion 5's window (just before ``hi``)."""
    if delta_lines(text):
        text = strip_t0104(rel, text)
    lines = text.splitlines()
    _lo, hi = delta_window(rel, lines)
    at = len(lines) if rel == "docs/tasks/T-0080d.md" else hi
    return "\n".join([*lines[:at], _delta_line(rel), *lines[at:]])


@pytest.mark.parametrize("rel", CRITERION5_FILES)
def test_delta_parser_accepts_one_placed_line(rel: str) -> None:
    check_delta(rel, _with_delta(rel, read(ROOT / rel)))


@pytest.mark.parametrize("rel", CRITERION5_FILES)
def test_delta_parser_fails_on_missing_duplicated_misdated_or_misplaced_line(rel: str) -> None:
    """Criterion 9, second mutation (one delta line removed), plus two lines, wrong date, wrong place."""
    good = _with_delta(rel, read(ROOT / rel))
    line = _delta_line(rel)
    lines = good.splitlines()
    i = lines.index(line)
    removed = "\n".join(lines[:i] + lines[i + 1 :])
    doubled = "\n".join([*lines[: i + 1], line, *lines[i + 1 :]])
    misdated = good.replace(line, line.replace(DATE, "2026-10-15"))
    for bad in (removed, doubled, misdated):
        with pytest.raises(AssertionError):
            check_delta(rel, bad)
    if rel != "docs/tasks/T-0080d.md":
        misplaced = "\n".join([line, *lines[:i], *lines[i + 1 :]])
        with pytest.raises(AssertionError):
            check_delta(rel, misplaced)


# --------------------------------------------------------------------------------------------------------------
# Criteria 6 and 8: guards, unmarked (they hold at 2b08f12 and must keep holding).
# --------------------------------------------------------------------------------------------------------------


def test_safety_text_unchanged() -> None:
    assert sha(p1_p9_block(read(DOC02))) == P1_P9_SHA256, "docs/02 §4 P1–P9 changed"
    t = read(T0080D)
    for h, want in T0080D_H_SHA256.items():
        assert sha(h_item(t, h)) == want, f"T-0080d {h} changed"
    assert sha(default3(t)) == T0080D_DEFAULT3_SHA256, "T-0080d settled default 3 changed"
    assert sha(claude_never_voice(read(CLAUDE_MD))) == CLAUDE_NEVER_VOICE_SHA256, (
        "CLAUDE.md 'Never voice' changed"
    )


def test_safety_guard_fails_on_one_character_of_p6() -> None:
    """Criterion 9, third mutation: one character of docs/02's P6 line."""
    doc = read(DOC02)
    p6 = next(ln for ln in doc.splitlines() if ln.startswith("- **P6.**"))
    mutated = doc.replace(p6, p6.replace("No ", "Mo ", 1), 1)
    assert mutated != doc
    assert sha(p1_p9_block(mutated)) != P1_P9_SHA256


def test_prompt_texts_untouched_but_the_persona() -> None:
    """T-0155 (negative; holds before and after): the same pins as the row below, with the twopass literal compared
    to its pre-T-0155 sha after its one persona sentence is reversed (``_pre_t0155_persona.to_pre_t0155``)."""
    from local_astra.conductor import prompt_builder as pb
    from tests.unit import _pre_t0155_persona as P
    from tests.unit._tools_switch_support import NATIVE_TOOL_INSTRUCTIONS_SHA256

    assert sha(pb.NATIVE_TOOL_INSTRUCTIONS) == NATIVE_TOOL_INSTRUCTIONS_SHA256
    assert sha(pb.SPOKEN_STYLE) == SPOKEN_STYLE_SHA256
    src = twopass_prompt_src(read(TWOPASS))
    assert sha(P.to_pre_t0155(src)) == TWOPASS_NATIVE_SYSTEM_PROMPT_SRC_SHA256_PRE_T0155, (
        "twopass prompt changed beyond the persona sentence"
    )
    assert hashlib.sha256(THRESHOLDS.read_bytes()).hexdigest() == THRESHOLDS_SHA256


def test_prompt_texts_untouched() -> None:
    from local_astra.conductor import prompt_builder as pb
    from tests.unit._tools_switch_support import NATIVE_TOOL_INSTRUCTIONS_SHA256

    assert sha(pb.NATIVE_TOOL_INSTRUCTIONS) == NATIVE_TOOL_INSTRUCTIONS_SHA256, (
        "NATIVE_TOOL_INSTRUCTIONS changed"
    )
    assert sha(pb.SPOKEN_STYLE) == SPOKEN_STYLE_SHA256, "SPOKEN_STYLE changed"
    assert sha(twopass_prompt_src(read(TWOPASS))) == TWOPASS_NATIVE_SYSTEM_PROMPT_SRC_SHA256, (
        "twopass prompt changed"
    )
    assert hashlib.sha256(THRESHOLDS.read_bytes()).hexdigest() == THRESHOLDS_SHA256, (
        "bench/thresholds.py changed"
    )
