"""T-0079 criterion 5 -- the four frozen calibrations published under ``docs/cd/`` (CPU, reads the repo).

Kept outside ``tests/cd`` on purpose: it needs only ``local_astra.eval.cd.calibration`` (json + hashlib, no
torch), so it runs in the fast suite and keeps guarding ``docs/cd`` after T-0079; and ``tests/cd`` is collected
only with ``ASTRA_CD_TESTS=1``, where ``scripts/check_pending.py`` cannot see a pending marker.

Expected values: BFCL knobs from H2's text (docs/tasks/T-0071b.md, "H2 -- human decision": perstep_mar eb 1.0 /
ct 0.005, perstep_mar_qargmax eb 0.3 / ct 0.005); Sudoku sha256 and knobs from T-0079 criterion 5 (pre-registered
picks kept, both eb 0.3 / ct 0.005). Provenance shas: the recorded b2 campaign (9e1479b9...) and the
pre-registered b2 frozen files it supersedes (d70f5888... / 42c689a3...), from T-0079 criteria 2 and 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_astra.eval.cd import calibration

REPO = Path(__file__).resolve().parents[2]
DOCS_CD = REPO / "docs" / "cd"

SHA_SOURCE_CAMPAIGN = "9e1479b939efe847c9d310eeb1c4f848e466693e085529602e5ddcb5e1f9d5a8"
SHA_OLD_MAR = "d70f588801f85338caa2b70a4cf25949a11546337ee63c4cfc5ac17cbf6424a7"
SHA_OLD_QARGMAX = "42c689a3223d20bcd2e7ab8bf12ae568bce74dcfbcb20a6833955767743313c3"
SHA_SUDOKU = "19acb78058815fa018391aea3c12fc0bbc86aec631e6d17293c489680040c69a"
SHA_SUDOKU_QARGMAX = "ae89b553115762991c1805278be6d97efa68d0250cc8160deb8b4703f28d7b55"

EXPECTED = {
    ("bfcl", "perstep_mar"): {"entropy_bound": 1.0, "confidence_threshold": 0.005},
    ("bfcl", "perstep_mar_qargmax"): {"entropy_bound": 0.3, "confidence_threshold": 0.005},
    ("sudoku", "perstep_mar"): {"entropy_bound": 0.3, "confidence_threshold": 0.005},
    ("sudoku", "perstep_mar_qargmax"): {"entropy_bound": 0.3, "confidence_threshold": 0.005},
}


def test_published_docs_cd_files() -> None:
    """The four published files load for their task and arm (and the SC twin via CALIBRATION_SOURCE); BFCL
    carries H2's knobs re-selected from the recorded campaign; Sudoku is byte-identical to the b1 evidence."""
    for (task, arm), knobs in EXPECTED.items():
        path = calibration.default_path(DOCS_CD, task, arm)
        assert path.is_file(), f"{path.relative_to(REPO)} is not published yet (T-0079 criterion 5)"
        got, sha = calibration.load_frozen(path, task=task, arm=arm)
        assert got == knobs, (path, got)
        twin, _ = calibration.load_frozen(path, task=task, arm=f"{arm}_sc")
        assert twin == knobs
        other = "perstep_mar_qargmax" if arm == "perstep_mar" else "perstep_mar"
        with pytest.raises(ValueError):
            calibration.load_frozen(path, task=task, arm=other)
        if task == "sudoku":
            assert sha == (SHA_SUDOKU if arm == "perstep_mar" else SHA_SUDOKU_QARGMAX), path
        else:
            data = json.loads(path.read_bytes())
            assert data["source_campaign"]["sha256"] == SHA_SOURCE_CAMPAIGN
            assert data["superseded"]["sha256"] == (
                SHA_OLD_MAR if arm == "perstep_mar" else SHA_OLD_QARGMAX
            )
            assert data["decision"] == {
                "by": "human",
                "date": "2026-10-08",
                "ref": "docs/tasks/T-0071b.md#H2",
            }
