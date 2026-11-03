"""Extract the T-0134 fixture from logs/t0132-crop-budget-ab-2ccff41/trials.jsonl (read-only)."""

import collections
import json
import sys
from pathlib import Path

LOG = Path("logs/t0132-crop-budget-ab-2ccff41")
rows = [
    json.loads(line) for line in (LOG / "trials.jsonl").read_text().splitlines() if line.strip()
]


def arm(r):
    if not r["reground"]:
        return f"{r['visual_tokens']}/off"
    return f"{r['visual_tokens']}/on/{r['reground_visual_tokens']}"


s4 = []
other = []
states = []
index = {}
for r in rows:
    if r["task"] == "S4":
        s4.append(
            {
                "arm": arm(r),
                "repeat": r["repeat"],
                "cart": (r["state"] or {}).get("cart"),
                "voiced": r["voiced"],
                "success": r["success"],
            }
        )
    else:
        key = json.dumps(r["state"], sort_keys=True)
        if key not in index:
            index[key] = len(states)
            states.append(r["state"])
        other.append([r["task"], arm(r), r["repeat"], index[key], r["success"]])

s4.sort(key=lambda d: (d["arm"], d["repeat"]))
other.sort()
print("s4", len(s4), "other", len(other), "states", len(states), file=sys.stderr)
print(collections.Counter(d["arm"] for d in s4), file=sys.stderr)
print(collections.Counter((o[0], o[4]) for o in other), file=sys.stderr)
out = {
    "source": "logs/t0132-crop-budget-ab-2ccff41/trials.jsonl (T-0132 run at 2ccff41, 2026-10-22)",
    "s4": s4,
    "states": states,
    "trials": other,
}
FIELDS = {
    "s4": "arm, repeat, cart (state.cart), voiced (the voiced sources), success (T-0111g score as logged)",
    "trials": "[task, arm, repeat, index into states, success (T-0111g score as logged)]",
}


def block(name: str, rows: list, sort: bool) -> list[str]:
    body = [
        "  " + json.dumps(r, sort_keys=sort) + ("," if i < len(rows) - 1 else "")
        for i, r in enumerate(rows)
    ]
    return [f' "{name}": [', *body]


lines = [
    "{",
    ' "source": ' + json.dumps(out["source"]) + ",",
    ' "fields": ' + json.dumps(FIELDS) + ",",
]
lines += [*block("s4", s4, True), " ],", *block("states", states, True), " ],"]
lines += [*block("trials", other, False), " ]", "}"]
# one record per line, so a review diff shows each trial
Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
