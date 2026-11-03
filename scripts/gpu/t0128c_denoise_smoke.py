"""T-0128c C2 -- the `/launch` smoke of the real denoising view. ``/launch`` ONLY.

    uv run --all-extras python scripts/gpu/t0128c_denoise_smoke.py [--out-dir logs/t0128c-smoke-<sha7>]

GPU-spending by definition (scripts/gpu/, T-0016): it boots the lean profile and sends three typed turns
through a real ``diffusion_gearvis session``. Run only through ``scripts/locked/t0128c_smoke.sh``.
**Importing this file opens no socket and starts nothing**: Playwright is imported inside ``main``.

Refusals (exit 2), in this order, before any service is started or any socket opened:

1. a dirty tree (the dirty paths are named);
2. an existing out-dir (one run per SHA);
3. an Astra port already bound (8000/1420/8001/8002/8080/8081, read from ``ss -ltn``);
4. ``HEAD:plugin`` differing from the tree the task branched from (``PLUGIN_TREE``, A10). The wheel itself
   is rebuilt from that tree by the locked script into an emptied ``dist/``; no wheel sha is compared
   (O-C (2): hatchling reproducibility is unproven).

Run (docs/tasks/T-0128c.md *Regime*): ``diffusion_gearvis up --profile lean`` (the lean plan as it is:
engine, STT, TTS — O-C (1), the regime records STT up; logs under ``logs/``), ``pytest -m gpu tests/contract/test_canvas_relay_wire.py`` (a failure stops the
run before the session), ``diffusion_gearvis session --silent --serve-hud`` with its session log under the
out-dir, headless Chromium (Playwright) on the HUD page at 1440x900, the three typed turns ``PROMPTS`` each
waited to IDLE, ``[data-denoise-root]``'s ``data-denoise-*`` counters, ``[data-denoise-label]`` and the
page's malformed counter polled at 20 Hz, a Thinking-pane screenshot every second while a turn is open (the one
nearest the turn's midpoint is kept), ``ss -tnp`` beside each of those shots, at each turn's end and
after the last turn, then the engine log slice written since the boot. The session,
the browser and the services are stopped in ``finally`` (``diffusion_gearvis down``).

Output (never ``artifacts/``): ``evidence.json`` (the format of tests/unit/test_denoise_smoke_checks.py),
``gates.json``, ``regime.json``, ``report.md``, ``thinking-<turn>.png``, ``session/*.jsonl``,
``session.out``, ``contract.xml``. The gates are ``local_astra.eval.denoise_smoke.gates``. A step that
fails or raises once the out-dir exists (boot, contract rows, session start, the page, a turn that never
returns to IDLE, a Playwright timeout, an interrupt) still writes a partial ``evidence.json`` with
``failed_step`` naming it and ``step_problems`` saying why, before the cleanup in ``finally`` (O-C (5)).

Exit: 0 every gate passed and no step failed; 1 a gate failed, a step failed (the gpu contract step
fails unless its three engine rows, ``ENGINE_ROWS``, ran and passed by name and no row failed or
errored; the other real arms' designed skips are allowed, O-C3) or ``down`` exited non-zero; 2 refused.

Regime (state it beside every number): profile lean, canvas length 256, 48 steps, thinking on,
``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372, ``vllm/vllm-openai:v0.29.0`` (services/versions.lock),
driver from ``nvidia-smi``, tap 30 Hz, n = 3 typed turns, unseeded, ``--silent`` audio, headless Chromium
1440x900, load average and commit sha recorded in ``regime.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_astra.bench.b16_tap import parse_tap_release_lines  # noqa: E402
from local_astra.eval.denoise_smoke import (  # noqa: E402
    EXPECTED_TURNS,
    GATES,
    gates,
    relayed_frames,
    teardown_errors,
)

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2

PROMPTS = (
    "What is 17 times 23?",
    "Explain in three sentences why the sky is blue.",
    "Give me a shopping list card for pancakes.",
)

#: conductor 8000, HUD 1420, engine 8001, tap 8002, STT 8080, TTS 8081
PORTS = (8000, 1420, 8001, 8002, 8080, 8081)

#: ``git rev-parse <commit>:plugin`` at 9b8f5c1 (the T-0128c task file, the base every group branched
#: from) and at 17feb30: A10 says plugin/ is byte-identical to the branch base.
PLUGIN_TREE = "158d62acf9f6a8ba1a331005401e0178990d1af4"

PAGE_URL = "http://127.0.0.1:1420/?ws=ws://127.0.0.1:8000/v1/realtime"
POLL_S = 0.05  # 20 Hz
SHOT_EVERY_S = 1.0
TURN_TIMEOUT_S = 240.0
SESSION_BOOT_S = 180.0
RELEASE_SETTLE_S = 3.0
ID_SHAPE = re.compile(r"^chatcmpl-[0-9a-f]{16}-[0-9a-f]{8}$")

_PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")

#: what the page shows, read in one evaluate (``None`` = attribute absent / no label text)
SAMPLE_JS = """() => {
  const root = document.querySelector('[data-denoise-root]');
  const num = (k) => {
    const v = root ? root.getAttribute(k) : null;
    return v === null || v === '' ? null : Number(v);
  };
  const lab = document.querySelector('[data-denoise-label]');
  const text = lab ? lab.textContent.trim() : '';
  const ign = document.getElementById('ignored-count');
  const chip = document.getElementById('turn-chip');
  return {
    frames: num('data-denoise-frames'),
    draws: num('data-denoise-draws'),
    digest: root ? root.getAttribute('data-denoise-digest') || null : null,
    label: text === '' ? null : text,
    ignored: ign ? ign.textContent : '',
    state: chip ? chip.getAttribute('data-turn-state') : null,
  };
}"""


def _say(msg: str) -> None:
    print(f"[t0128c {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _env() -> dict[str, str]:
    """The environment of every child: no proxy variables (loopback only)."""
    return {k: v for k, v in os.environ.items() if k not in _PROXY_VARS}


def _run(cmd: Sequence[str], *, check: bool = True, **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), check=check, text=True, capture_output=True, env=_env(), **kw)


def _git(*args: str) -> str:
    return _run(["git", "-C", str(REPO), *args]).stdout.strip()


def _bin(name: str) -> str:
    """A console script of this venv, run directly so its pid is the Python process's own."""
    path = Path(sys.executable).parent / name
    if not path.exists():
        raise SystemExit(f"REFUSED: {path} is absent (uv sync --all-extras)")
    return str(path)


# --------------------------------------------------------------------------- pure helpers


def malformed_of(ignored_text: str) -> int:
    """The ``m`` of the page's ``#ignored-count`` (``"Xu/Ym/Zr"``; ``"0"`` before the first render)."""
    m = re.search(r"(\d+)m\b", ignored_text)
    return int(m.group(1)) if m else 0


def listening_ports(ss_ltn: str) -> set[int]:
    """The local ports of ``ss -ltn`` lines (any address)."""
    ports: set[int] = set()
    for ln in ss_ltn.splitlines()[1:]:
        cols = ln.split()
        if len(cols) >= 4 and ":" in cols[3]:
            tail = cols[3].rsplit(":", 1)[1]
            if tail.isdigit():
                ports.add(int(tail))
    return ports


#: O-C3: the gpu contract rows that exercise the engine's tap; each must run and pass by name. The
#: other real arms of tests/contract/test_canvas_relay_wire.py skip by design (engine restart, other
#: topics / malformed parts, refused connect) and their skips are allowed.
ENGINE_ROWS = (
    "test_tap_frames_reach_the_relay_and_are_published_unchanged[real]",
    "test_the_relay_owns_one_sub_that_connects_bounded_and_closed_before_term[real]",
    "test_a_silent_endpoint_publishes_nothing_and_costs_almost_nothing[real]",
)


def junit_outcomes(xml_text: str) -> dict[str, str]:
    """Each junit ``testcase``'s outcome by name: ``passed``, ``skipped``, ``failed`` or ``error``."""
    root = ElementTree.fromstring(xml_text)
    out: dict[str, str] = {}
    for case in root.iter("testcase"):
        tags = {child.tag for child in case}
        if "error" in tags:
            outcome = "error"
        elif "failure" in tags:
            outcome = "failed"
        elif "skipped" in tags:
            outcome = "skipped"
        else:
            outcome = "passed"
        name = str(case.get("name", ""))
        # a name seen twice keeps its worst outcome (error > failed > skipped > passed)
        rank = ("passed", "skipped", "failed", "error")
        if name not in out or rank.index(outcome) > rank.index(out[name]):
            out[name] = outcome
    return out


def contract_problem(returncode: int, junit: Path) -> str | None:
    """The gpu contract step's verdict (O-C3): it passes iff every ``ENGINE_ROWS`` row ran and passed
    by name and no row failed or errored. Designed skips of the other real arms are allowed; a skip
    of an engine row (e.g. "no engine on ...") is a failure, as are a non-zero exit and a missing or
    unreadable junit report."""
    if returncode != 0:
        return f"the gpu contract rows exited {returncode}"
    if not junit.exists():
        return f"the gpu contract rows wrote no junit report ({junit})"
    try:
        outcomes = junit_outcomes(junit.read_text(encoding="utf-8"))
    except ElementTree.ParseError as exc:
        return f"the gpu contract junit report is unreadable: {exc}"
    broken = sorted(n for n, o in outcomes.items() if o in ("failed", "error"))
    if broken:
        return f"gpu contract rows failed or errored: {broken}"
    missing = [n for n in ENGINE_ROWS if outcomes.get(n) != "passed"]
    if missing:
        seen = {n: outcomes.get(n, "absent") for n in missing}
        return f"gpu contract engine rows did not run and pass: {seen}"
    return None


def new_turn_id(session_log: Sequence[Mapping[str, Any]], taken: Sequence[str]) -> str | None:
    """The first turn id of a non-replay, non-IDLE ``astra.turn.state`` not already assigned."""
    for rec in session_log:
        if rec.get("kind") != "wire.out":
            continue
        msg = (rec.get("data") or {}).get("msg") or {}
        if msg.get("type") != "astra.turn.state" or msg.get("replay"):
            continue
        tid = msg.get("turn_id")
        if msg.get("state") != "IDLE" and tid is not None and tid not in taken:
            return str(tid)
    return None


def reported(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Reported, not gated: page frames/s per turn, tap published vs page-decoded, id shapes,
    and the astra.error rows of the harness's own teardown (O-C5 (2))."""
    rows = parse_tap_release_lines(str(evidence.get("release_log") or ""))
    relayed = relayed_frames(evidence.get("session_log") or [])
    ids = list(dict.fromkeys(str(f["request_id"]) for f in relayed))
    published = sum(int(r["published"]) for r in rows if r["req_id"] in set(ids))
    dropped = sum(int(r["dropped"]) for r in rows)
    per_turn = []
    decoded_total = 0
    for turn in evidence.get("turns") or []:
        samples = turn.get("samples") or []
        base = (samples[0].get("frames") if samples else None) or 0
        last = next((s["frames"] for s in reversed(samples) if s.get("frames") is not None), 0)
        decoded = int(last) - int(base)
        decoded_total += decoded
        moving = [
            s["t_ms"]
            for i, s in enumerate(samples[1:], 1)
            if (s.get("frames") or 0) != (samples[i - 1].get("frames") or 0)
        ]
        span_s = (moving[-1] - moving[0]) / 1000 if len(moving) >= 2 else None
        per_turn.append(
            {
                "turn_id": turn.get("turn_id"),
                "decoded": decoded,
                "samples": len(samples),
                "page_frames_per_s": round(decoded / span_s, 2) if span_s else None,
            }
        )
    notes = [
        (r.get("data") or {}).get("denoise")
        for r in evidence.get("session_log") or []
        if r.get("kind") == "note" and isinstance((r.get("data") or {}).get("denoise"), dict)
    ]
    return {
        "per_turn": per_turn,
        "tap_published_for_relayed_ids": published,
        "tap_dropped_all_ids": dropped,
        "relayed_rows": len(relayed),
        "page_decoded": decoded_total,
        "relayed_request_ids": ids,
        "ids_match_chatcmpl_16hex_8hex": {i: bool(ID_SHAPE.match(i)) for i in ids},
        "denoise_notes": notes,
        # O-C5 (2): astra.error after the harness page's disconnect that follows the last turn
        "g6_teardown_errors": teardown_errors(
            evidence.get("turns") or [], evidence.get("session_log") or []
        ),
    }


def render_markdown(
    sha: str, regime: Mapping[str, Any], out: Mapping[str, list[str]], rep: Mapping[str, Any]
) -> str:
    lines = [f"# T-0128c C2 denoising smoke, commit {sha}", "", "## Regime", ""]
    lines += [f"- {k}: {v}" for k, v in regime.items()]
    lines += ["", "## Gates", "", "| gate | verdict | problems |", "|---|---|---|"]
    for g in GATES:
        probs = out.get(g, [])
        lines.append(f"| {g} | {'PASS' if not probs else 'FAIL'} | {'; '.join(probs) or '-'} |")
    lines += ["", "## Reported, not gated", "", "```json", json.dumps(rep, indent=2), "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- refusals


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr, flush=True)
    return EXIT_REFUSED


def _plugin_refusal() -> str | None:
    tree = _git("rev-parse", "HEAD:plugin")
    if tree != PLUGIN_TREE:
        return f"plugin/ tree {tree} is not the branch base's {PLUGIN_TREE} (A10: plugin unchanged)"
    return None


def preflight(out_dir: Path) -> int | None:
    if len(PROMPTS) != EXPECTED_TURNS:
        return _refuse(f"{len(PROMPTS)} prompts but the gates expect {EXPECTED_TURNS} turns")
    dirty = _git("status", "--porcelain")
    if dirty:
        return _refuse(f"dirty tree: {' '.join(dirty.splitlines())}")
    if out_dir.exists():
        return _refuse(f"{out_dir} exists; the run happens once per SHA")
    busy = sorted(listening_ports(_run(["ss", "-ltn"]).stdout) & set(PORTS))
    if busy:
        return _refuse(f"Astra ports already bound: {busy} (a fresh boot is needed)")
    reason = _plugin_refusal()
    if reason:
        return _refuse(reason)
    return None


# --------------------------------------------------------------------------- the run


def _read_jsonl(path: Path, problems: list[str] | None = None) -> list[dict[str, Any]]:
    """The records of a JSONL file the session may still be writing (O-C2 (4)).

    A torn last line (the writer mid-record) is dropped; an unparseable line before it is skipped
    and, when ``problems`` is given, recorded there. Nothing here raises before evidence is written.
    """
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    lines = [
        ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()
    ]
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as exc:
            if i != len(lines) - 1 and problems is not None:
                problems.append(f"{path}: unparseable line {i + 1} skipped ({exc})")
            continue
        if isinstance(rec, dict):
            out.append(rec)
        elif problems is not None:
            problems.append(f"{path}: line {i + 1} is not a JSON object, skipped")
    return out


def _session_log_path(session_out: Path, log_dir: Path) -> Path | None:
    text = session_out.read_text(encoding="utf-8", errors="replace") if session_out.exists() else ""
    m = re.search(r"logging this session to (\S+)", text)
    if m:
        return Path(m.group(1))
    found = sorted(log_dir.glob("*.jsonl"))
    return found[-1] if found else None


def _regime(sha: str) -> dict[str, Any]:
    smi = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], check=False
    ).stdout.strip()
    lock = (REPO / "services" / "versions.lock").read_text(encoding="utf-8")
    pins = [
        ln.strip()
        for ln in lock.splitlines()
        if re.match(r"\s*(image|model|revision|vllm|tap_endpoint|tap_hz)\b", ln)
    ]
    return {
        "commit": sha,
        "profile": "lean (canvas 256, 48 steps, thinking on)",
        "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372 (services/versions.lock)",
        "engine": "vllm/vllm-openai:v0.29.0",
        "versions_lock_pins": pins,
        "gpu_driver": smi,
        "tap": "30 Hz per request, tcp://127.0.0.1:8002",
        "turns": f"n = {len(PROMPTS)} typed turns, unseeded",
        "audio": "--silent (null device, TTS runs unheard)",
        "services": "diffusion_gearvis up --profile lean: engine, STT up (O-C (1)), TTS",
        "browser": "headless Chromium (Playwright), 1440x900, prefers-reduced-motion no-preference",
        "load_average": " ".join(f"{x:.2f}" for x in os.getloadavg()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _wait_session(proc: subprocess.Popen[bytes], session_out: Path) -> str | None:
    deadline = time.monotonic() + SESSION_BOOT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f"the session exited {proc.returncode} before binding; see {session_out}"
        if {8000, 1420} <= listening_ports(_run(["ss", "-ltn"]).stdout):
            return None
        time.sleep(0.5)
    return f"the session did not bind 8000 and 1420 within {SESSION_BOOT_S:g} s"


def _sample(page: Any, t0: float) -> dict[str, Any]:
    raw = page.evaluate(SAMPLE_JS)
    return {
        "t_ms": int((time.monotonic() - t0) * 1000),
        "frames": None if raw["frames"] is None else int(raw["frames"]),
        "draws": None if raw["draws"] is None else int(raw["draws"]),
        "digest": raw["digest"],
        "label": raw["label"],
        "malformed": malformed_of(str(raw["ignored"])),
        "_state": raw["state"],
    }


class StepFailed(Exception):
    """A step of the run failed; its message is the problem the partial evidence records."""


@dataclass
class _Run:
    """What the run has collected so far: written as (partial) evidence however the run ends."""

    out_dir: Path
    step: str = "preflight"
    turns: list[dict[str, Any]] = field(default_factory=list)
    ss_lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    failed_step: str | None = None
    session_log: Path | None = None
    session: subprocess.Popen[bytes] | None = None

    def fail(self, problem: str) -> None:
        if self.failed_step is None:
            self.failed_step = self.step
        self.problems.append(f"{self.step}: {problem}")

    def ss(self) -> None:
        self.ss_lines.extend(_run(["ss", "-tnp"], check=False).stdout.splitlines())


def _one_turn(page: Any, text: str, t0: float, run: _Run, k: int) -> None:
    """Baseline, type, poll at 20 Hz to IDLE; keep the Thinking-pane shot nearest the midpoint.

    The turn's record is appended before the first sample, so a timeout leaves what was sampled."""
    samples: list[dict[str, Any]] = []
    run.turns.append({"turn_id": None, "text": text, "samples": samples})
    run.step = f"turn {k + 1}: wait for IDLE before typing"
    page.wait_for_function(
        "() => document.getElementById('turn-chip').getAttribute('data-turn-state') === 'IDLE'",
        timeout=TURN_TIMEOUT_S * 1000,
    )
    samples.append(_sample(page, t0))
    run.step = f"turn {k + 1}: type the prompt"
    box = page.locator("[data-ask-input]")
    box.fill(text)
    box.press("Enter")
    run.step = f"turn {k + 1}: poll the page until IDLE"
    started = time.monotonic()
    opened = False
    shots: list[tuple[float, bytes]] = []
    last_shot = 0.0
    try:
        while True:
            time.sleep(POLL_S)
            s = _sample(page, t0)
            samples.append(s)
            now = time.monotonic()
            if s["_state"] != "IDLE":
                opened = True
                if now - last_shot >= SHOT_EVERY_S:
                    last_shot = now
                    shots.append((now, page.locator("section.pane-thinking").screenshot()))
                    run.ss()  # O-C2 (3): the conductor's sockets mid-turn, beside each shot
            elif opened:
                break
            if now - started > TURN_TIMEOUT_S:
                raise StepFailed(f"the page did not return to IDLE within {TURN_TIMEOUT_S:g} s")
    finally:
        ended = time.monotonic()
        if shots:
            mid = (started + ended) / 2
            at, png = min(shots, key=lambda sh: abs(sh[0] - mid))
            (run.out_dir / f"thinking-turn{k + 1}.png").write_bytes(png)
            _say(
                f"turn {k + 1}: thinking-pane shot at {at - started:.1f} s of {ended - started:.1f} s"
            )
        run.ss()


def _assign_turn_ids(run: _Run) -> None:
    """Each turn's id: the first new non-IDLE turn id in the session log, in typing order (O-C (3))."""
    log = _read_jsonl(run.session_log) if run.session_log else []
    taken: list[str] = []
    for turn in run.turns:
        tid = new_turn_id(log, taken)
        if tid is not None:
            taken.append(tid)
        turn["turn_id"] = tid


def _run_turns(run: _Run) -> None:
    from playwright.sync_api import sync_playwright

    t0 = time.monotonic()
    run.step = "browser: launch headless Chromium"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-proxy-server"])
        try:
            run.step = "page: open the HUD and wait for the connection"
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(PAGE_URL)
            page.wait_for_function(
                "() => document.getElementById('conn-status').textContent === 'connected'",
                timeout=30_000,
            )
            page.wait_for_selector("[data-ask-input]:not([disabled])", timeout=30_000)
            for k, text in enumerate(PROMPTS):
                _say(f"turn {k + 1}: {text!r}")
                _one_turn(page, text, t0, run, k)
                _say(f"turn {k + 1}: {len(run.turns[-1]['samples'])} samples")
        finally:
            browser.close()


def _steps(run: _Run, cli: str) -> None:
    """Boot, the gpu contract rows, the session, the page and the turns; a failure raises."""
    out_dir = run.out_dir
    run.step = "boot: diffusion_gearvis up --profile lean"
    _say(run.step)
    up = subprocess.run([cli, "up", "--profile", "lean"], env=_env(), check=False)
    if up.returncode != 0:
        raise StepFailed(f"diffusion_gearvis up exited {up.returncode}")

    run.step = "contract: pytest -m gpu tests/contract/test_canvas_relay_wire.py"
    _say(run.step)
    contract = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/contract/test_canvas_relay_wire.py",
            "-q",
            "-m",
            "gpu",
            "-rA",
            f"--junitxml={out_dir / 'contract.xml'}",
        ],
        cwd=REPO,
        env=_env(),
        check=False,
    )
    problem = contract_problem(contract.returncode, out_dir / "contract.xml")
    if problem:
        raise StepFailed(f"{problem}; no session runs")

    run.step = "session: diffusion_gearvis session --silent --serve-hud"
    _say(run.step)
    session_out = out_dir / "session.out"
    session_dir = out_dir / "session"
    with session_out.open("wb") as fh:
        run.session = subprocess.Popen(
            [
                cli,
                "session",
                "--silent",
                "--serve-hud",
                "--log-dir",
                str(session_dir),
                "--data-dir",
                str(out_dir / "data"),
            ],
            cwd=REPO,
            env=_env(),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    reason = _wait_session(run.session, session_out)
    if reason:
        raise StepFailed(reason)
    run.session_log = _session_log_path(session_out, session_dir)
    listeners = [
        ln
        for ln in _run(["ss", "-ltn"]).stdout.splitlines()[1:]
        if re.search(r":(8000|1420)\b", ln)
    ]
    wide = [ln for ln in listeners if not re.search(r"\s127\.0\.0\.1:(8000|1420)\s", ln)]
    if wide:
        raise StepFailed(f"non-loopback conductor/HUD listener: {wide}")

    _run_turns(run)
    run.step = "after the turns"
    run.ss()
    time.sleep(RELEASE_SETTLE_S)  # the last request's tap release line


def _finish(run: _Run, sha: str, regime: Mapping[str, Any], engine_log: Path, offset: int) -> int:
    """Write evidence.json (partial when a step failed), gates.json and report.md; the exit code."""
    _assign_turn_ids(run)
    for turn in run.turns:
        for s in turn["samples"]:
            s.pop("_state", None)
    if run.session_log is None and run.session is not None:
        run.session_log = _session_log_path(run.out_dir / "session.out", run.out_dir / "session")
    records = _read_jsonl(run.session_log, run.problems) if run.session_log else []
    started = next((r for r in records if r.get("kind") == "log.started"), None)
    spawned = run.session.pid if run.session is not None else 0
    pid = int((started or {}).get("data", {}).get("pid", spawned))
    if run.session is not None and pid != spawned:
        run.problems.append(f"session log pid {pid} != spawned pid {spawned}")
    release_log = ""
    if engine_log.exists():
        with engine_log.open("rb") as fh:
            fh.seek(offset)
            release_log = fh.read().decode("utf-8", errors="replace")
    evidence = {
        "turns": run.turns,
        "session_log": records,
        "release_log": release_log,
        "ss_lines": run.ss_lines,
        "conductor_pid": pid,
        "failed_step": run.failed_step,
        "step_problems": run.problems,
    }
    (run.out_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=1) + "\n", encoding="utf-8"
    )
    if run.failed_step is not None:
        _say(f"FAIL at step {run.failed_step!r}: partial evidence in {run.out_dir}")
    try:
        verdict = gates(evidence)
    except ValueError as exc:  # a malformed tap release line never passes G7
        verdict = {g: [] for g in GATES}
        verdict["G7"] = [str(exc)]
    rep: dict[str, Any] = {}
    with contextlib.suppress(ValueError):  # the same malformed line: already a G7 problem
        rep = reported(evidence)
    rep["failed_step"] = run.failed_step
    rep["step_problems"] = run.problems
    (run.out_dir / "gates.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (run.out_dir / "report.md").write_text(
        render_markdown(sha, regime, verdict, rep), encoding="utf-8"
    )
    for g in GATES:
        _say(f"{g}: {'PASS' if not verdict[g] else 'FAIL ' + '; '.join(verdict[g])}")
    _say(f"report: {run.out_dir / 'report.md'}")
    failed = any(verdict[g] for g in GATES) or bool(run.problems)
    return EXIT_FAILED if failed else EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default logs/t0128c-smoke-<sha7>/ (never artifacts/)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sha = _git("rev-parse", "--short=7", "HEAD")
    out_dir: Path = (args.out_dir or REPO / "logs" / f"t0128c-smoke-{sha}").resolve()
    if "artifacts" in out_dir.parts:
        return _refuse(f"{out_dir} is under artifacts/; the smoke writes under logs/ only")
    refused = preflight(out_dir)
    if refused is not None:
        return refused
    out_dir.mkdir(parents=True)
    regime = _regime(sha)
    (out_dir / "regime.json").write_text(json.dumps(regime, indent=2) + "\n", encoding="utf-8")

    from local_astra.supervisor import log_dir

    engine_log = log_dir() / "vllm.log"
    offset = engine_log.stat().st_size if engine_log.exists() else 0
    cli = _bin("diffusion_gearvis")
    run = _Run(out_dir)
    rc = EXIT_FAILED
    try:
        try:
            _steps(run, cli)
        except StepFailed as exc:
            run.fail(str(exc))
        except (Exception, KeyboardInterrupt) as exc:  # a Playwright timeout, an interrupt, ...
            run.fail(f"{type(exc).__name__}: {exc}")
        # O-C (5): evidence (partial when a step failed) is written before any cleanup below
        rc = _finish(run, sha, regime, engine_log, offset)
    finally:
        down_rc = _cleanup(run, cli)
    if down_rc != 0:
        # O-C2 (5): services possibly left running make the run a failure whatever the gates said
        _say(f"FAIL: diffusion_gearvis down exited {down_rc}")
        rc = EXIT_FAILED
    _say(f"harness exit {rc}")
    return rc


def _cleanup(run: _Run, cli: str) -> int:
    """Stop the session, then everything ``up`` started; ``down``'s exit code."""
    session = run.session
    if session is not None and session.poll() is None:
        session.send_signal(2)  # SIGINT: the session's own clean stop (T-0040)
        with contextlib.suppress(subprocess.TimeoutExpired):
            session.wait(timeout=30)
        if session.poll() is None:
            session.kill()
            session.wait()
    down = subprocess.run([cli, "down"], env=_env(), check=False)
    _say(f"diffusion_gearvis down exited {down.returncode}")
    if shutil.which("nvidia-smi"):
        mem = _run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
            check=False,
        ).stdout.strip()
        _say(f"device memory at the end: {mem}")
    return down.returncode


if __name__ == "__main__":
    sys.exit(main())
