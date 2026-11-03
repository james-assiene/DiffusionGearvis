"""Write ``t0092a_golden.json``: criterion 6's golden for T-0092a, captured at the base SHA only.

Usage, from the repository root, on a tree whose ``local_astra/conductor/session.py`` is
byte-identical to the base commit's::

    uv run python tests/fixtures/session/capture_t0092a_golden.py

It refuses to write otherwise, so the golden can never be regenerated from an implementation. The
scenario, the capture and the normalisation are the test module's own
(``tests/unit/test_session_stale_playback_golden.py``), imported, not copied.
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

from tests.unit.test_session_stale_playback_golden import (  # noqa: E402
    BASE_SESSION_BLOB,
    BASE_SHA,
    GOLDEN,
    capture,
)


def main() -> int:
    blob = subprocess.run(
        ["git", "hash-object", "local_astra/conductor/session.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if blob != BASE_SESSION_BLOB:
        print(
            f"refusing: local_astra/conductor/session.py is {blob}, not {BASE_SHA[:7]}'s "
            f"{BASE_SESSION_BLOB}; the golden is captured at the base SHA only",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        sections = asyncio.run(capture(Path(tmp)))
    GOLDEN.write_text(
        json.dumps(
            {"base_sha": BASE_SHA, "session_py_blob": BASE_SESSION_BLOB, "sections": sections},
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
