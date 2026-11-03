"""T-0055 criterion 12 (re-owed from T-0044 criterion 5b): the block's size as the engine counts.

Read-only against the running engine's /tokenize (no generation, no sampling, no canvas): posts the
block text, prints the token count, the token ids and the regime. The ceiling is 600 tokens; the
fast suite only pins 2000 code points, which is a structural bound and not a token claim (ruling
A9). ``/v1/tokenize`` 404s on this server, so the endpoint is the server root (T-0044 ruling A2).

Two things T-0055 added, so that one command produces the whole of criterion 12:

* **the special-token check.** ``<|think|>``, ``<|channel>``, ``<channel|>``, ``<|turn>``,
  ``<turn|>`` are added tokens with ``special: true`` (ids 98, 100, 101, 105, 106 of 24). The fast
  suite forbids their *spellings* in the block; this is the same check at the layer where it
  matters — one of them in the block is a control token inside the cached system turn.
  ``add_special_tokens`` is off so the count is the block's own, with no BOS added by the server:
  an id that comes back special therefore came from the text.
* **``--baseline <path>``**, which tokenises the T-0044 block
  (``tests/fixtures/prompts/ui_block_t0044.txt``, the text that failed twice on 2026-09-16) in the
  same run, so the delta this task costs every turn is a number and not an estimate.

Usage::

    uv run python scripts/demo/t0044_token_count.py
    uv run python scripts/demo/t0044_token_count.py --baseline tests/fixtures/prompts/ui_block_t0044.txt

Needs an engine somebody else has already started (``uv run astra up --profile lean``); state the
profile and the versions.lock revisions beside the numbers it prints.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

from local_astra.conductor.ui_instructions import UI_INSTRUCTIONS
from local_astra.config import load

CEILING = 600

#: The added tokens this repo has recorded by id (docs/07 §Native tokens, docs/08, T-0055 F2):
#: ``<|think|>`` 98, ``<|channel>`` 100, ``<channel|>`` 101, ``<|turn>`` 105, ``<turn|>`` 106.
KNOWN_SPECIAL_IDS = (98, 100, 101, 105, 106)

#: The *shape* of an added token, for the 24 this file does not name by id. Nothing in an English
#: instruction block is spelled ``<|…>`` or ``<…|>``; the A2UI fence carries no pipe.
SPECIAL_TOKEN_SHAPE = re.compile(r"<\|[^\s<>]{0,24}\|?>|<[^\s<>]{0,24}\|>")


def tokenize(root: str, text: str) -> tuple[int, list[int], list[str]]:
    """``(count, ids, token strings)`` from the server root's /tokenize. No generation."""
    body = json.dumps(
        {"prompt": text, "add_special_tokens": False, "return_token_strs": True}
    ).encode()
    request = urllib.request.Request(
        f"{root}/tokenize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as answer:
        payload = json.loads(answer.read())
    ids = list(payload.get("tokens", []))
    strings = list(payload.get("token_strs") or [])
    return int(payload.get("count", len(ids))), ids, strings


def specials(ids: list[int], strings: list[str]) -> list[str]:
    """Every token of the text that is an added special token, by id or by shape."""
    found = [f"id {i} at position {n}" for n, i in enumerate(ids) if i in KNOWN_SPECIAL_IDS]
    found += [
        f"{s!r} at position {n}" for n, s in enumerate(strings) if SPECIAL_TOKEN_SHAPE.search(s)
    ]
    return found


def report(label: str, root: str, text: str) -> int:
    count, ids, strings = tokenize(root, text)
    found = specials(ids, strings)
    print(f"[{label}]")
    print(f"  code points   {len(text)}")
    print(f"  tokens        {count}")
    print(f"  token ids     {ids}")
    print(f"  special       {found or 'none'}")
    print(f"  verdict       {'PASS' if count <= CEILING and not found else 'FAIL'}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="T-0055 criterion 12")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="a second block text to tokenise, for the delta (the T-0044 fixture)",
    )
    args = parser.parse_args()

    services = load(None).profiles["lean"].services
    root = services.vllm.rsplit("/v1", 1)[0]
    print(f"endpoint      {root}/tokenize (add_special_tokens=False)")
    print(f"ceiling       {CEILING} tokens (T-0055 criterion 12, re-owed from T-0044 5b)")
    print("regime        profile lean; engine and model revisions from services/versions.lock")

    count = report("block as merged", root, UI_INSTRUCTIONS)
    if args.baseline is not None:
        was = report(f"baseline {args.baseline}", root, args.baseline.read_text(encoding="utf-8"))
        print(f"[delta]       {count - was:+d} tokens on every turn")


if __name__ == "__main__":
    main()
