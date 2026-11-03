"""Build ``native_t4_eval.json`` from the stored T-0084e (dev) and T-0084f (test) artifacts (read-only, shas checked).

What it extracts, for the T-0091 fast-suite rows (the cd group re-derives all of it from the artifacts directly):

- per v1 tool, the ``native/T4/named/perstep_mar+seed`` request the eval sent: ``automaton_sha256``, ``n_states`` and
  ``bucket`` (identical on every row of both artifacts, which the script checks);
- one seeded row per tool (the first case of the dev artifact that names it): the pass-1 ``draft_call_text``, the
  first stop id of its pass-1 ``token_ids``, the stored ``seed_ids`` and the stored ``knobs`` without ``record_tag``;
- the regime's ``seed_draft`` block, ``tools_sha256`` and ``versions`` keys the payload manifest repeats.

Run from anywhere: ``python tests/fixtures/toolcall/build_native_t4_eval.py`` (stdlib only). The artifacts live in the
main checkout's ``logs/`` (untracked), found through ``git rev-parse --git-common-dir``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "native_t4_eval.json"
CELL = "native/T4/named/perstep_mar+seed"
STOP_IDS = (1, 106, 50)
ARTIFACTS = {
    "t0084e-dev": (
        "logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl",
        "06a71fc7413828ba5ff2843accf0811b5ce8303014ebf0177e99904060bd5c7f",
    ),
    "t0084f-test": (
        "logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl",
        "14728634eee2a8621e4bf3fbaf3c84c52ccc224e7c6b0383c82a4dcac0595101",
    ),
}


def main_checkout() -> Path:
    common = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent


def main() -> None:
    root = main_checkout()
    tools: dict[str, dict] = {}
    seeded: dict[str, dict] = {}
    regime_keep: dict = {}
    for key, (rel, sha) in ARTIFACTS.items():
        raw = (root / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha, (key, "sha mismatch")
        rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        regime = rows[0]
        keep = {
            "tools_sha256": regime["tools_sha256"],
            "seed_draft": regime["seed_draft"],
            "compiler_version": regime["versions"]["compiler_version"],
            "tokenizer_sha256": regime["versions"]["tokenizer_sha256"],
            "model_revision": regime["versions"]["model_revision"],
            "whitespace": regime["whitespace"],
            "perstep_mar_calibration_sha256": regime["calibration"]["perstep_mar"]["sha256"],
        }
        assert not regime_keep or regime_keep == keep, (key, "regimes differ")
        regime_keep = keep
        p1 = {r["case"]: r for r in rows if r.get("pass") == 1 and r.get("format") == "native"}
        for r in rows:
            if r.get("pass") != 2 or r.get("cell") != CELL:
                continue
            (name,) = r["scopes"]
            knobs = {k: v for k, v in r["knobs"].items() if k != "record_tag"}
            entry = {
                "automaton_sha256": knobs["automaton_sha256"],
                "n_states": r["n_states"][0],
                "bucket": r["bucket"],
            }
            assert tools.setdefault(name, entry) == entry, (key, name, "differs across rows")
            if key == "t0084e-dev" and name not in seeded:
                q = p1[r["case"]]
                ids = q["token_ids"]
                stop = next((t for t in ids if t in STOP_IDS), None)
                seeded[name] = {
                    "artifact": key,
                    "case": r["case"],
                    "draft_call_text": q["draft_call_text"],
                    "stop_id": stop,
                    "seed_ids": r["seed_ids"],
                    "knobs": knobs,
                }
    doc = {
        "provenance": {
            "artifacts": {k: {"path": v[0], "sha256": v[1]} for k, v in ARTIFACTS.items()},
            "cell": CELL,
            "generator": "tests/fixtures/toolcall/build_native_t4_eval.py",
        },
        "regime": regime_keep,
        "tools": dict(sorted(tools.items())),
        "seeded": dict(sorted(seeded.items())),
    }
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(OUT, len(tools), len(seeded))


if __name__ == "__main__":
    main()
