"""Build the T-0172 A4 census labels from the session logs on this box (read-only; docs/tasks/T-0172.md, A4).

Usage: ``uv run python tests/fixtures/t0172/build_labels.py <out.jsonl>``

**Independent of the rule.** This file imports the standard library only: never
``local_astra.toolcall.announced_call`` (A4: "labels made without importing ``announced_call``"), and nothing else
from ``local_astra``. ``tests/unit/test_t0172_announced_rule.py`` checks that by reading this file's imports.

**The corpus.** Every ``logs/**/sessions/*.jsonl`` under the logs root (this checkout's ``logs/``, else the main
checkout's, found through the git common dir) whose file name's timestamp (its first 15 characters,
``YYYYMMDD-HHMMSS``) is at most ``CUTOFF``: the logs on this box when the labels were made (2026-10-27, main
3e4d45b7). Per ``llm.request`` record whose ``phase`` is ``generate`` or ``tool_step`` (one request per record; no
request id repeats within a file in this corpus):

- ``tools_on``: its logged ``prompt.tools`` is a non-empty list;
- ``tool_commit``: an ``llm.commit`` with its ``attempt_id`` has ``segment == "tool"``;
- ``done_tool``: the ``tool`` of the ``machine.step`` record whose event is ``{"type": "llm.done", "request_id":
  <its request id>}`` (the ``llm.done`` log record itself carries only ``request_id`` and ``commits`` in every file
  of this corpus, so "``llm.done`` has ``tool: null``" is read from the machine's ``LlmDone`` event); ``MISSING``
  when the request has no such event (an aborted request);
- ``answer``: its ``answer`` commits' texts, in log order, joined with no separator.

A **no-call answer** has no tool commit, a ``done_tool`` that is ``None`` (present and null) and a non-empty
``answer.strip()``. It is a **candidate** when it is tools-on and the broad, deliberately over-inclusive ``INTENT``
filter below finds first-person future intent anywhere in it ("I will", "I'll", "let me", "going to", "I need to",
"I should", "next, I", ...). Every candidate is labelled by hand (``POSITIVE`` / ``NEGATIVE`` below); every other
no-call answer is a negative by definition (A4).

**The label** (the test-author's reading of the Goal, "a final no-call answer only announces an action"):
*positive* iff the answer's last statement is the assistant saying, in the first person, that it will now take an
action that needs a tool (search, click, scroll, add, look/find/check on the page, go back, refresh, deselect, ...),
nothing follows it, and the answer neither asks the user anything, nor makes the action conditional on the user,
nor states the task's result after it. Size and form are *not* part of the label: a positive the A1 grammar cannot
reach (a comma-joined clause, a 270-character leaked thought) stays a positive and costs the rule recall.

**Ambiguous cases** (also in the test-author's report):

- ``"I will add the cotton tea towel to the cart.\\nI'm adding the cotton tea towel to the cart."``: **negative**.
  Its last statement is a present-progressive claim ("I'm adding"), not future intent; answers that are only
  such claims ("I am adding the blue mug to your cart." x3, "I am looking for the cast-iron skillet on the page." x4,
  ...) hold no future intent and are negatives by A4's definition, so labelling this one positive would make the
  label depend on the earlier sentence alone. (The re-ask would plausibly help all of them: a blind spot of A1,
  reported, not labelled around.)
- ``"I found a cast-iron griddle, I will add that to your cart."`` and ``"Since I cannot see the cast-iron skillet
  yet, I will scroll down to find it."``: **positive**. One sentence each, the intent after a comma; the A1 grammar
  (lead-in from a closed list, then the subject) does not start there.
- The 270-character leaked thought ending "I will now check if it's visible and clickable.": **positive** (its last
  statement announces an action); over the 240-character bound.
- ``"I will add the blue mug to your cart.<click_at{x:425,y:631}"``: **positive**. A malformed call spelled as
  text is not a call (A5 pins the same shape with ``call:click_at{...}`` as a hit).
- ``"I will check if the blue mug was added to the cart."`` (and the tea-towel twin): **positive**. "if the" is not a
  conditional addressed to the user, and checking the page is a screen action.
- Leaked thoughts ending "Since I don't have the new screenshot yet, I'll wait.": **negative** (waiting is not an
  action); ending in a result sentence ("The Eiffel Tower is ...", "I have added ..."): **negative**.
- ``"Wait, let me check the cart. the. the. ..."`` (the T-0142b collapse, 1,103 characters): **negative**; its last
  statement is degenerate repetition, not an announcement.

**Private chats.** Files directly under ``logs/sessions/`` are ordinary voice chats. They are read and counted and
never copied: the output holds no row, text or file name from them (their only trace is the per-directory count of
``logs/sessions`` and the ``private_*`` counts). In this corpus none of them holds a candidate; the builder exits 1
if one ever does (it could not be labelled here without copying it).

**Cross-checks before anything is written** (exit 1 otherwise): 2,313 files (126 private), 6,681
``generate``/``tool_step`` requests, 1,987 tools-on no-call answers (11 private), 45 tools-off no-call answers (all
private), 116 candidates (83 distinct texts), each labelled exactly once: 108 positive, 8 negative; both smoke
records (b372eda, seq 904 and 1047) positive. The planner's census (74 + 32 = 106) is a cross-check, not the
expected value: it counted rule hits, these are hand labels.

Output: line 1 ``{"provenance": {...counts, cutoff, dirs: {dir: files}}}``; then one row per candidate, in (path,
seq) order: ``source`` (path under the logs root), ``seq`` (of the ``llm.request`` record), ``request_id``,
``attempt_id``, ``phase``, ``step`` (1 for ``generate``), ``answer``, ``label``, ``why``, ``smoke``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CUTOFF = "20261027-123442"
PHASES = ("generate", "tool_step")
PRIVATE_DIR = "sessions"
MISSING = object()
SMOKE = "t0125-search-smoke-b372eda/phase1/sessions/20261027-112444-1064873.jsonl"
#: the two smoke records by the seq of their answer commit (the task's 904 and 1047) and their request ids
SMOKE_RECORDS = {("t2-gen-s7", 904), ("t3-gen-s2", 1047)}

#: Broad candidate filter (over-inclusive on purpose; never the A1 grammar).
INTENT = re.compile(
    r"\b(?:i\s*will|i['’]ll|let\s+me|let['’]s|i\s+am\s+going\s+to|i['’]m\s+going\s+to|i\s+shall|going\s+to"
    r"|i['’]m\s+about\s+to|i\s+need\s+to|i\s+am\s+now|i['’]m\s+now|i\s+should|i\s+must|i\s+want\s+to"
    r"|next,?\s+i|now\s+i|i\s+would\s+like\s+to|i\s+plan\s+to|i['’]d\s+like\s+to)\b",
    re.IGNORECASE,
)

#: Hand labels, positive: the exact joined answer texts (each read by the test-author in context).
POSITIVE = frozenset(
    {
        "I will click on seat F14.",
        "I will look for a booking button.",
        "I will add the blue mug to the cart.",
        "I will add the blue mug to your cart.",
        "I will search for example.com on Google.",
        "I will scroll up to find the booking form.",
        "I will scroll down to find the M6 hex nuts.",
        "I will scroll up to find the booking button.",
        "I will add the cotton tea towel to your cart.",
        "I will search for example.com heading on Google.",
        "I will try adding the blue mug to the cart again.",
        "I will try to add the blue mug to the cart again.",
        "I will scroll down to find the cast-iron skillet.",
        "I will look for the cast-iron skillet on the page.",
        "I will check if the blue mug was added to the cart.",
        "I will click the add to cart button for the blue mug.",
        "I will search for the current weather in Paris again.",
        "I will try to add the blue mug to the cart once more.",
        "I will scroll down to see if there is a booking form.",
        "I will search for the latest stable version of Python.",
        "I will add the M6 hex nuts, pack of 50, to your order.",
        "I will find the booking interface to enter the details.",
        "I will scroll down again to find the Leith locker option.",
        "I found a cast-iron griddle, I will add that to your cart.",
        "I will scroll down to find the Leith locker pickup option.",
        "I will look for a booking form to start a new reservation.",
        "I will add the blue mug to your cart.<click_at{x:425,y:631}",
        "I will scroll down to find the Locker pickup, Leith option.",
        "I will find the booking form to enter the requested details.",
        "Actually, F14 is already selected. I will deselect it for you.",
        "I will look for the booking section to start a new reservation.",
        "I will scroll the list down again to find Locker pickup, Leith.",
        "I will check if the cotton tea towel has been added to your cart.",
        "I will add another pack of M6 hex nuts, pack of 50 to your order.",
        "I will try clicking the add to cart button for the blue mug again.",
        "I will click the add to cart button for the walnut desk organiser.",
        "I will add the cotton tea towel to the cart.call:click_at{x:156,y:898}",
        "I will add the cotton tea towel to the cart as it is the cheapest item.",
        "I will add the cotton tea towel to your cart as it is the cheapest item.",
        "I will look for the cotton tea towel on the screen and add it to your cart.",
        "I will scroll down to find the add to cart button for the cast-iron skillet.",
        "Since I cannot see the cast-iron skillet yet, I will scroll down to find it.",
        "I'm having trouble adding the blue mug to the cart. I will try one more time.",
        "I don't see a booking form on this page. I will try to find the booking section.",
        "I don't see the contact form on this screen yet. I will try scrolling down to find it.",
        "I don't see the booking form on the screen. I will try to find a button to book a table.",
        "I have scrolled down to find the items. I will now add the cotton tea towel to your cart.",
        "I have scrolled down to find the cheapest item. I will now add the tea towel to your cart.",
        "I cannot see the booking form on the screen. I will try to find it by clicking on the page.",
        "I don't see a yellow mug on the screen. I will scroll down to see if it is listed further down.",
        "I cannot see the current screen after the scroll. I will try to find the cotton tea towel again.",
        "I do not see a yellow mug on the page right now. I will scroll down to see if it is further down.",
        "The cheapest item is the enamel mug at eleven dollars and fifty cents. I will add it to your cart.",
        "I don't see a booking form on this page. I will search for the booking page of The Larder Kitchen.",
        "I cannot see Locker pickup, Leith in the current list. I will try to scroll down again to find it.",
        "I cannot find the Locker pickup, Leith option in the list. I will try scrolling down one more time.",
        "I have scrolled down to see all the items. I will now find the cheapest one and add it to your cart.",
        "I don't see the booking form on this screen. I will search for the booking page for The Larder Kitchen.",
        "I do not see Locker pickup, Leith in the current list. I will scroll down once more to see if it appears.",
        "I see that a booking has already been sent. I will try to refresh the page to find the booking form again.",
        "The cotton tea towel is the cheapest item at four dollars and seventy-five cents. I will add it to your cart.",
        "I will add the cotton tea towel to the cart as it is the cheapest item at four dollars and seventy-five cents.",
        "I see a booking was already sent for 7:00 pm. I will try to go back to the booking page to change the details.",
        "I cannot see Locker pickup, Leith in the current list. I will try to scroll the list again to see if it appears.",
        "I noticed a booking has already been sent for 7:00 pm. I will try to find the booking form to update the details.",
        "The booking has already been sent for two people at 7:00 pm. I will go back to the homepage to start a new booking.",
        "Actually, the cotton tea towel is the cheapest item at four dollars and seventy-five cents. I will add that to the cart.",
        "I cannot find Locker pickup, Leith in the current list. I will try to scroll down once more to see if more options appear.",
        "Actually, I will add the cheapest item, which is the cotton tea towel for four dollars and seventy-five cents, to your cart.",
        "The booking has already been sent for two people at 7:00 pm. I will try to find the booking page again to change the details.",
        "The booking for Grace Hopper has already been sent for two people at 7:00 pm. I will try to find a way to make a new booking for you.",
        "I can't see a booking form on the screen. It looks like a booking has already been sent. I will scroll up to see if I can find the form.",
        "Actually, it looks like a booking has already been sent for seven o'clock. I'll scroll down to find the booking form to make the correct request.",
        "I see that a booking has already been sent for two people at seven:00 pm. I will find the booking form to make the new reservation for four guests at seven:thirty0 pm.",
        "The user wants to add the cheapest item to the cart. Based on the initial view, the Cotton tea towel is the cheapest at $4.75. I tried to click it but it didn't seem to work, so I scrolled down to find it again. I will now check if it's visible and clickable.",
    }
)

#: Hand labels, negative: sha256 of the exact joined answer text -> why (texts are in the output rows).
NEGATIVE = {
    "d5d77cfd26e387d0e0ab156b5193ef9abbf71af982b3fbcdc7ab8cfbef8127e0": (
        "ends in a present-progressive claim (\"I'm adding ...\"), not future intent; the task's near-miss"
    ),
    "8f82753a5e5e55f4d8bf7e3b65d7c43b438d54e5166d24b93d872d1cacf240aa": (
        'asks the user ("Please let me know what items ..."); the task\'s near-miss'
    ),
    "cf4b40c8b98ae378485eab51e9e947fac0234783c129983c09188bad6ecef99f": (
        'states the result after the intent ("The cheapest item is the enamel mug ...")'
    ),
    "16d855bf5d7f79eaadaec8d215fa93fd14a379873deee5cacc93624bf0a7786e": (
        'states the result after the intent ("...list.The cheapest item is ..."); the task\'s near-miss'
    ),
    "ece8f05aa845fbfc5315b86ef2f154692c98818a7fba111b4d55f5f730c1f799": (
        'leaked thought ending in the final answer ("The Eiffel Tower is three hundred and thirty meters tall ...")'
    ),
    "b878d46e9a1c8d31d4bd172159febf8c48fee976c10271abf1dc58c14dd47d81": (
        'leaked thought ending in a completion claim ("I have added the cotton tea towel ...")'
    ),
    "22a9f68901e1f5a6f20a089865239140ece3ac53652e9784ecfa07f85984a7c0": (
        'leaked thought ending "I\'ll wait": waiting is not an action'
    ),
    "96c42ff3da6ce5225e23dcfd1d9f6aa149480bef6e3e127a859b60df0670b641": (
        "degenerate repetition (the T-0142b collapse); its last statement is not an announcement"
    ),
}

#: Notes on the positives the A1 grammar is not expected to reach (reported, never a reason to relabel).
POSITIVE_NOTES = {
    "I found a cast-iron griddle, I will add that to your cart.": "intent after a comma (one sentence)",
    "Since I cannot see the cast-iron skillet yet, I will scroll down to find it.": (
        "intent after a comma (one sentence)"
    ),
    (
        "The user wants to add the cheapest item to the cart. Based on the initial view, the Cotton tea towel is the "
        "cheapest at $4.75. I tried to click it but it didn't seem to work, so I scrolled down to find it again. "
        "I will now check if it's visible and clickable."
    ): "a leaked thought over the 240-character bound, ending in an announcement",
}

EXPECTED = {
    "files": 2313,
    "private_files": 126,
    "requests": 6681,
    "tools_on_no_call_answers": 1987,
    "private_tools_on_no_call_answers": 11,
    "tools_off_no_call_answers": 45,
    "private_tools_off_no_call_answers": 45,
    "candidates": 116,
    "distinct_candidates": 83,
    "positives": 108,
    "negatives": 8,
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def logs_roots() -> list[Path]:
    """This checkout's ``logs/``, then the main checkout's (the git common dir's parent)."""
    here = Path(__file__).resolve()
    repo = here.parents[3]
    out = [repo / "logs"]
    proc = subprocess.run(
        ["git", "-C", str(here.parent), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        main_logs = Path(proc.stdout.strip()).parent / "logs"
        if main_logs not in out:
            out.append(main_logs)
    return out


def session_files(root: Path) -> list[Path]:
    """The corpus under ``root``: ``**/sessions/*.jsonl`` with a name timestamp at most ``CUTOFF``, sorted."""
    return sorted(p for p in root.glob("**/sessions/*.jsonl") if p.name[:15] <= CUTOFF)


def is_private(rel: str) -> bool:
    return rel.split("/", 1)[0] == PRIVATE_DIR


def requests_of(path: Path) -> list[dict[str, Any]]:
    """Every ``generate``/``tool_step`` request of a session file, in log order (module docstring)."""
    reqs: list[dict[str, Any]] = []
    by_attempt: dict[str, dict[str, Any]] = {}
    done: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        data = rec.get("data") or {}
        if kind == "llm.request" and data.get("phase") in PHASES:
            prompt = data.get("prompt") or {}
            tools = prompt.get("tools") if isinstance(prompt, dict) else None
            req: dict[str, Any] = {
                "seq": rec.get("seq"),
                "request_id": data.get("request_id"),
                "attempt_id": data.get("attempt_id"),
                "phase": data.get("phase"),
                "step": 1 if data.get("phase") == "generate" else data.get("step"),
                "tools_on": isinstance(tools, list) and len(tools) > 0,
                "tool_commit": False,
                "answers": [],
                "answer_seqs": [],
            }
            reqs.append(req)
            by_attempt[str(data.get("attempt_id"))] = req
        elif kind == "llm.commit":
            owner = by_attempt.get(str(data.get("attempt_id")))
            if owner is not None:
                if data.get("segment") == "tool":
                    owner["tool_commit"] = True
                elif data.get("segment") == "answer":
                    owner["answers"].append(str(data.get("text", "")))
                    owner["answer_seqs"].append(rec.get("seq"))
        elif kind == "machine.step":
            event = data.get("event") or {}
            if isinstance(event, dict) and event.get("type") == "llm.done":
                done[str(event.get("request_id"))] = event.get("tool")
    for req in reqs:
        req["done_tool"] = done.get(str(req["request_id"]), MISSING)
        req["answer"] = "".join(req["answers"])
    return reqs


def is_no_call_answer(req: dict[str, Any]) -> bool:
    return not req["tool_commit"] and req["done_tool"] is None and bool(str(req["answer"]).strip())


def census(root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    dirs: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    unlabelled: list[str] = []
    for path in session_files(root):
        rel = path.relative_to(root).as_posix()
        private = is_private(rel)
        dirs[rel.rsplit("/", 1)[0]] += 1
        counts["files"] += 1
        counts["private_files"] += private
        for req in requests_of(path):
            counts["requests"] += 1
            if not is_no_call_answer(req):
                continue
            if not req["tools_on"]:
                counts["tools_off_no_call_answers"] += 1
                counts["private_tools_off_no_call_answers"] += private
                continue
            counts["tools_on_no_call_answers"] += 1
            counts["private_tools_on_no_call_answers"] += private
            answer = str(req["answer"])
            if not INTENT.search(answer):
                continue
            if private:
                counts["private_candidates"] += 1
                continue  # never copied (module docstring); main() refuses to write
            if answer in POSITIVE:
                label, why = True, POSITIVE_NOTES.get(answer, "announces an action, no call")
            elif sha(answer) in NEGATIVE:
                label, why = False, NEGATIVE[sha(answer)]
            else:
                unlabelled.append(f"{rel} seq {req['seq']}: {answer[:100]!r}")
                continue
            smoke = rel.endswith(SMOKE) and any(
                (req["request_id"], s) in SMOKE_RECORDS for s in req["answer_seqs"]
            )
            rows.append(
                {
                    "source": rel,
                    "seq": req["seq"],
                    "request_id": req["request_id"],
                    "attempt_id": req["attempt_id"],
                    "phase": req["phase"],
                    "step": req["step"],
                    "answer": answer,
                    "label": label,
                    "why": why,
                    "smoke": smoke,
                }
            )
    counts["candidates"] = len(rows) + len(unlabelled)
    counts["distinct_candidates"] = len({r["answer"] for r in rows})
    counts["positives"] = sum(r["label"] for r in rows)
    counts["negatives"] = sum(not r["label"] for r in rows)
    return {
        "counts": dict(counts),
        "dirs": dict(sorted(dirs.items())),
        "rows": rows,
        "unlabelled": unlabelled,
    }


def check(doc: dict[str, Any]) -> list[str]:
    problems = [f"unlabelled candidate: {u}" for u in doc["unlabelled"]]
    if doc["counts"].get("private_candidates", 0):
        problems.append(
            f"{doc['counts']['private_candidates']} candidates in logs/sessions/ (private): cannot be labelled here"
        )
    for key, want in EXPECTED.items():
        got = doc["counts"].get(key, 0)
        if got != want:
            problems.append(f"{key}: {got}, expected {want}")
    used_pos = {r["answer"] for r in doc["rows"] if r["label"]}
    used_neg = {sha(r["answer"]) for r in doc["rows"] if not r["label"]}
    problems += [f"positive label never used: {t!r}" for t in sorted(POSITIVE - used_pos)]
    problems += [f"negative label never used: {h}" for h in sorted(set(NEGATIVE) - used_neg)]
    smoke = sorted((r["request_id"], r["label"]) for r in doc["rows"] if r["smoke"])
    if smoke != [("t2-gen-s7", True), ("t3-gen-s2", True)]:
        problems.append(f"smoke records: {smoke}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: uv run python tests/fixtures/t0172/build_labels.py <out.jsonl>", file=sys.stderr
        )
        return 2
    root = next((r for r in logs_roots() if r.is_dir()), None)
    if root is None:
        print(f"build_labels: no logs/ (tried {[str(r) for r in logs_roots()]})", file=sys.stderr)
        return 1
    doc = census(root)
    problems = check(doc)
    if problems:
        print("build_labels: refusing to write:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    provenance = {
        "task": "T-0172 A4",
        "built": "2026-10-27",
        "base": "3e4d45b7",
        "cutoff": CUTOFF,
        "intent_filter": INTENT.pattern,
        **doc["counts"],
        "dirs": doc["dirs"],
    }
    lines = [json.dumps({"provenance": provenance}, ensure_ascii=False)]
    lines += [json.dumps(r, ensure_ascii=False) for r in doc["rows"]]
    Path(argv[0]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"build_labels: wrote {len(doc['rows'])} rows to {argv[0]}: {doc['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
