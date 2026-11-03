"""Write ``t0092b_tools_off_golden.json``: T-0092b criterion 6's golden, captured at the base SHA only.

Usage, from the repository root, on a tree whose T-0092b-owned files are byte-identical to a502871's::

    uv run python tests/fixtures/session/capture_t0092b_golden.py

It refuses to write otherwise, so the golden can never be regenerated from an implementation. The scenario,
capture and normalisation are the test module's own (``tests/unit/test_session_tools_off_golden.py``), imported.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tests.unit.test_session_tools_off_golden import (  # noqa: E402
    BASE_BLOBS,
    BASE_SHA,
    GOLDEN,
    capture,
)


def main() -> int:
    for path, blob in BASE_BLOBS.items():
        got = subprocess.run(
            ["git", "hash-object", path], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        if got != blob:
            print(f"refusing: {path} is {got}, not {BASE_SHA}'s {blob}", file=sys.stderr)
            return 2
    with tempfile.TemporaryDirectory() as tmp:
        sections = asyncio.run(capture(Path(tmp)))
    GOLDEN.write_text(
        json.dumps(
            {"base_sha": BASE_SHA, "blobs": BASE_BLOBS, "sections": sections},
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {GOLDEN.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
