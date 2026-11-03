"""Parse pytest's final summary line and demand an exact outcome (T-0067 tier 2: no vacuous pass).

A container pytest run that exits 0 proves little: every row may have skipped, been deselected or xfailed. The
host rows in ``tests/contract/test_vllm_cd_cuda_gpu.py`` therefore read the inner run's summary line
(``==== 17 passed, 2 warnings in 41.20s ====`` or, under ``-q``, ``17 passed in 41.20s``) and require exactly
the expected number of passes and nothing else except warnings. Pure string parsing; pinned by
``tests/unit/test_cd_tier2_summary_parsing.py``.
"""

from __future__ import annotations

import re

#: the summary line: optional ``=`` bars, a body, `` in <secs>s``, optional `` (h:mm:ss)``
_LINE = re.compile(r"^=*\s*(?P<body>\S.*?) in \d+(?:\.\d+)?s(?: \([0-9:]+\))?\s*=*$")
_PART = re.compile(r"^(?P<n>\d+) (?P<what>[a-z]+)$")
#: pytest pluralises only these two ("1 error", "1 warning")
_PLURAL = {"error": "errors", "warning": "warnings"}
#: the only outcome besides ``passed`` a strict run may report
ALLOWED_EXTRA = frozenset({"warnings"})


def parse_summary(output: str) -> dict[str, int]:
    """Counts from the **last** pytest summary line in ``output``; ``{}`` for "no tests ran".

    Raises ``AssertionError`` when there is no summary line (pytest crashed, or did not run).
    """
    for raw in reversed(output.splitlines()):
        m = _LINE.match(raw.strip())
        if m is None:
            continue
        body = m.group("body")
        if body == "no tests ran":
            return {}
        counts: dict[str, int] = {}
        for part in body.split(", "):
            pm = _PART.match(part)
            if pm is None:
                break
            what = _PLURAL.get(pm.group("what"), pm.group("what"))
            counts[what] = counts.get(what, 0) + int(pm.group("n"))
        else:
            return counts
    raise AssertionError("no pytest summary line in the output (did pytest run to completion?)")


def require_exact(output: str, expected_passed: int) -> dict[str, int]:
    """Assert the last summary reports exactly ``expected_passed`` passed and nothing but warnings besides."""
    counts = parse_summary(output)
    passed = counts.get("passed", 0)
    others = {k: v for k, v in counts.items() if k != "passed" and k not in ALLOWED_EXTRA}
    assert passed == expected_passed and not others, (
        f"expected exactly {expected_passed} passed and no skipped/xfailed/xpassed/deselected/failed/errors; "
        f"the inner pytest summary says {counts or 'no tests ran'}"
    )
    return counts
