"""T-0223 group B: the GPU check harness ``scripts/gpu/t0223_handwritten_check.py`` (unit; CPU; fakes; no engine, no
socket). docs/tasks/T-0223.md "Group B — the GPU check" B1-B7 and its "Group B tests" list.

BASE e27c90bb (main; group B spec), rulings R-3..R-6 at af674ce5. :func:`harness` loads the harness by path lazily
and a missing file is a named ``AssertionError``. Revision 2 (after the group B review, 28 mutants, 22 survived):
every row passes on the implementer's harness at T-0223-b-impl f1e074ba, so none carries ``pending``. Group A's
assets (``demos/handwritten/attempts.json``, ``out/<id>.png``) are not used here: every row
builds a small synthetic ``attempts.json`` (group A's shape: ``{"attempts": [{id, course, title, problem, lines,
mistake, correct_result, judge_keywords, false_blame_keywords, diagram}]}``, ruling R-2's ``|`` alternatives) and
ten portrait PNGs in ``tmp_path`` and injects them with their sha256s.

**The harness surface these rows assume** (the orchestrator's names of 2026-11-01; the signatures are the
test-author's reading, reported to the orchestrator)

Constants
    ``TUTOR_INSTRUCTIONS``, ``QUESTION`` (criterion 5's texts), ``ORDER_SEED = 0``, ``TURN_TIMEOUT_S = 90``,
    ``RUN_BUDGET_S = 840`` (R-4: 14 min); ``PAGE_IDS`` is not required. Injectable pins (the rig monkeypatches them on the
    harness module): ``ATTEMPTS_PATH`` (Path of ``attempts.json``), ``PAGES_DIR`` (directory of ``<id>.png``),
    ``ATTEMPTS_SHA256`` (hex), ``PAGE_SHA256`` (``{id: hex}``). The harness reads all four at call time.
Seams
    ``_git(*args) -> str`` (the only way it runs git: ``status --porcelain``, ``log`` for the merge subjects).
Functions
    ``judge(text_spoken, text_shown, page) -> (located, spoken_only_located, matched)`` (``page`` an attempt dict;
    ``matched`` the ``judge_keywords`` entries satisfied by the judged text); ``frame_message(png_path, now_ms) ->
    AstraFrame``; ``order(seed) -> [(page_id, repeat)]``; ``budget_stop(elapsed_s) -> bool`` (True: no further turn
    starts); ``finish_turn(out, page_id, repeat, trial) -> dict`` (the runner's raw trial, below, plus ``spoken``,
    ``shown``, ``clips``, ``located``, ``spoken_only_located``, ``matched``, ``first_word_ms``, ``violations``);
    ``write_report(out) -> Any`` writing ``report.json`` and ``report.md`` from the run dir (``order.json``,
    ``manifest.json``, ``probes.json``, ``trials.jsonl``, ``turns.jsonl``, ``sessions/``) in the shapes
    ``tests/eval/test_t0223_handwritten_check.py``'s docstring lists; ``build_report(out) -> dict`` the same
    report without writing; ``main(argv) -> int`` with ``--base-url`` and ``--out-dir``.
The runner's raw trial (what one turn hands ``finish_turn``)
    ``page``, ``repeat``, ``session_log`` (file name under ``out/sessions``, ``None`` when not run),
    ``exception``, ``timed_out``, ``frame_sha256``, ``stored_sha256``, ``frame_digest`` (the logged form:
    sha256 of the base64 text, 8 hex), ``user_text_ms``, ``settings_echo`` (``{rev, refused}``), ``wall_s``;
    and (R-3, R-5) ``bodies`` (the meter's body summaries: ``label``, ``phase``, ``images`` as sha256 of each image,
    ``max_soft_tokens``, ``tools``), ``http_non_200``, ``engine_up_after``. The report adds ``body_violations``
    and ``latency.cold_first_turn_ms``.
Revision 2 also uses ``_Run`` (fields filtered to the dataclass's), ``_one_turn(run, page_id, repeat)`` (the dress
row), ``_run_all(run)`` (the errored stop), ``wait_settings(page, timeout_s=)`` and T-0208's ``F._client``,
``G._wait_idle``, ``T._healthy``, ``G.read_metrics`` (stubbed in the dress row).

Not called directly (reported): ``evidence(...)`` (reached through ``finish_turn``/``write_report``),
``_refusal(args)`` (every refusal goes through ``main``).

Equivalent mutants (reported, not killable by a spec-backed row): ``if not datas`` (no ``llm.request``) is also
caught by the system-sha12 clause; the judge's ``bool(text.strip())`` is covered by ``bool(keywords)`` and the
substring rule; ``set(probes) == set(PROBE_NAMES)`` is covered by the per-name ``passed`` lookup for a missing
probe; ``phash(resized)`` versus ``phash(rgb)`` (the spec asks only for a 16-hex phash).

Readings and gaps (reported to the orchestrator, not resolved silently)

1. "1120 visual tokens not in force" is mutated as ``cli.LIVE_DEFAULT_VISUAL_TOKENS = 560`` (what
   ``cli.grounding_of`` resolves for a session with no flag and no profile key); a harness that hard-codes 1120
   without reading cli's resolution is not caught by any other row.
2. "Branch merges don't count" is read as a lookalike subject (``Merge branch 'T-0223-a' ...``,
   ``Merge T-0223-ab: ...``), as in T-0220; a first-parent rule is not required.
3. B3 (c)'s ``max_soft_tokens == 1120`` lives in the HTTP body, not in the session log: R-5's body summaries in the
   trial carry it (``body-mst``), and the dress row checks the real ``request_body``.
4. Spoken text is the ``tts.speak`` ``source`` strings (the eval's reading 1); latency's p50 is the low median and
   the synthetic run has an odd count so the reading does not matter here.
5. ``test_session_matches_cli_tools_off`` inspects the ``Session`` P-S builds inside ``main`` (the only Session
   ``main`` builds on CPU): it assumes the turns and P-S share one Session builder.

Regime: CPU, pure, hand-built session logs in the real record shapes (``session_log.redact`` over real ``Prompt``
dumps), synthetic 1700x2200 PNGs, n = 1, no engine.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import importlib.util
import inspect
import io
import json
import random
import re
import socket
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0223_handwritten_check.py"
EVAL = REPO / "tests" / "eval" / "test_t0223_handwritten_check.py"
#: the task's base (main with the group B spec), a literal
BASE = "e27c90bb"

PAGE_IDS = ("M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5")
QUESTION = "Here's my attempt. Can you check it and tell me where I went wrong?"
TUTOR = (
    "You are a patient tutor. Point to the line with the mistake and explain why it is wrong, then give the "
    "corrected step."
)
HEADER = "The user's own instructions follow. Apply them unless they conflict with anything above."
BLOCK = HEADER + "\n" + TUTOR
SYS_BASE = (
    "You are Gearvis, a local assistant that can see the user's screen and hear their voice. Answer what you can "
    "actually see; say so when you cannot.\nKeep answers short."
)
VALID_URL = "http://127.0.0.1:9/v1"
GIT_SHA = "1a2b3c4" + "d" * 33
MERGE_A2A = "f0b8606fb25c332dfe08a47ee22a5bec0ae3cc67 Merge T-0213-a2a: apply panel settings at the next idle turn boundary"
MERGE_A = "a223a223a223a223a223a223a223a223a223a223 Merge T-0223-a: ten handwritten attempts and their renderer"
OTHERS = (
    "5237da2d4f67fbc53beaf8f5b9a3972cce1f93dd Merge T-0215: --search off in live does not offer google_search",
    "520223f6aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa Merge release-test-fixes: the suite is green in the squashed release tree",
)
LOOKALIKES_A = (
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0223-a' into main",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge T-0223-ab: something else",
    # names the subject but does not start with it (a revert of the merge, merged back)
    "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge branch 'revert' (Revert \"Merge T-0223-a: ten pages\")",
)
CS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":'
    '"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}'
)


# =====================================================================================================
# loading
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0223 B: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0223_handwritten_check_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), (
        f"T-0223 B: {HARNESS.name} has no {name!r} (see this module's docstring)"
    )
    return getattr(h, name)


@functools.cache
def _eval() -> ModuleType:
    from importlib import import_module

    return import_module("tests.eval.test_t0223_handwritten_check")


def _held(h: ModuleType) -> list[ModuleType]:
    """The harness and every ``scripts/gpu`` module it holds by name, four levels down."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _ in range(4):
        nxt = []
        for mod in frontier:
            for v in vars(mod).values():
                if (
                    isinstance(v, ModuleType)
                    and id(v) not in seen
                    and "/scripts/gpu/" in str(getattr(v, "__file__", "") or "")
                ):
                    seen[id(v)] = v
                    nxt.append(v)
        frontier = nxt
    return list(seen.values())


def _patch_name(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    for mod in _held(h):
        if hasattr(mod, name):
            mp.setattr(mod, name, value)


def _patch_object(mp: pytest.MonkeyPatch, h: ModuleType, original: Any, value: Any) -> int:
    n = 0
    for mod in _held(h):
        for k, v in list(vars(mod).items()):
            if v is original:
                mp.setattr(mod, k, value)
                n += 1
    return n


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =====================================================================================================
# synthetic group A assets
# =====================================================================================================


def _kw(p: str) -> list[str]:
    """Two entries per page; the first has an R-2 alternative."""
    q = p.lower()
    return [f"{q}word|{q}alt", f"{q}fix"]


def _fb(p: str) -> list[str]:
    return [f"{p.lower()} first line is wrong"]


def _attempts() -> dict[str, Any]:
    return {
        "task": "T-0223",
        "notes": "synthetic (T-0223 B unit rig)",
        "attempts": [
            {
                "id": p,
                "course": "MATH 101" if p[0] == "M" else "PHYS 101",
                "title": f"synthetic {p}",
                "problem": f"Problem {p}.",
                "lines": [
                    {"text": f"{p} line 1", "mistake": False},
                    {"text": f"{p} line 2", "mistake": True},
                    {"text": f"{p} line 3", "mistake": False},
                ],
                "mistake": f"Line 2 of {p} is wrong.",
                "correct_result": f"{p} result",
                "judge_keywords": _kw(p),
                "false_blame_keywords": _fb(p),
                "diagram": None,
            }
            for p in PAGE_IDS
        ],
    }


def _png(index: int, size: tuple[int, int] = (1700, 2200)) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, (250, 250, 245))
    d = ImageDraw.Draw(im)
    for y in range(120, size[1], 64):
        d.line([(0, y), (size[0], y)], fill=(170, 190, 220), width=2)
    d.rectangle([100 + 40 * index, 300, 600 + 40 * index, 500], outline=(20, 30, 120), width=6)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class Assets:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.attempts_path = root / "attempts.json"
        self.pages_dir = root / "out"
        self.pages_dir.mkdir(exist_ok=True)
        self.write_attempts(_attempts())
        self.page_sha: dict[str, str] = {}
        for i, p in enumerate(PAGE_IDS):
            data = _png(i)
            (self.pages_dir / f"{p}.png").write_bytes(data)
            self.page_sha[p] = _sha(data)

    def write_attempts(self, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
        self.attempts_path.write_bytes(raw)
        self.attempts_sha = _sha(raw)

    def inject(self, mp: pytest.MonkeyPatch, h: ModuleType) -> None:
        for name, value in (
            ("ATTEMPTS_PATH", self.attempts_path),
            ("PAGES_DIR", self.pages_dir),
            ("ATTEMPTS_SHA256", self.attempts_sha),
            ("PAGE_SHA256", dict(self.page_sha)),
        ):
            api(name)  # a named assertion when the pin is absent
            mp.setattr(h, name, value)


# =====================================================================================================
# socket spy: an audit hook (registered once; it cannot be removed) and refusing doubles
# =====================================================================================================

_AUDIT: dict[str, Any] = {"on": False, "events": []}


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _AUDIT["on"] and event in ("socket.connect", "socket.getaddrinfo"):
        _AUDIT["events"].append((event, repr(args)[:200]))


sys.addaudithook(_audit)


class Rig:
    """Every real socket refused and recorded (doubles plus the audit hook); a clean tree with both merges through
    the harness's ``_git``; the synthetic assets injected; idle-box seams where they exist."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        self.h = h = harness()
        self.mp = mp
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*OTHERS, MERGE_A2A, MERGE_A)
        self.out = tmp_path / "out"
        self.assets = Assets(tmp_path / "assets")
        self.assets.inject(mp, h)
        rig = self

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                rig.sockets.append(what)
                raise ConnectionRefusedError(f"T-0223 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            if args[:1] == ("status",):
                return rig.porcelain
            if "log" in args:
                if any("%H" in a for a in args):
                    return "\n".join(rig.merges) + "\n"
                return "\n".join(m.split(" ", 1)[1] for m in rig.merges) + "\n"
            if args[:1] in (
                ("hash-object",),
                ("show",),
                ("cat-file",),
                ("ls-files",),
                ("ls-tree",),
            ):
                return subprocess.run(
                    ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
                ).stdout
            return GIT_SHA + "\n"

        api("_git")
        _patch_name(mp, h, "_git", git)
        for seam, value in (
            ("_nvidia_smi", lambda *a, **k: "pid, process_name, used_gpu_memory [MiB]\n"),
            ("_loadavg", lambda *a, **k: (0.5, 0.5, 0.5)),
            ("_listening_ports", lambda *a, **k: {8001}),
        ):
            _patch_name(mp, h, seam, value)

    def argv(self, url: str = VALID_URL) -> list[str]:
        return ["--base-url", url, "--out-dir", str(self.out)]

    def main(self, argv: list[str]) -> int:
        _AUDIT["events"] = []
        _AUDIT["on"] = True
        try:
            rc = self.h.main(argv)
        finally:
            _AUDIT["on"] = False
        return int(rc)


# =====================================================================================================
# B1: refusals before any socket, through main
# =====================================================================================================

REFUSALS: dict[str, str] = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "url-localhost-lookalike": r"loopback|localhost\.example",
    "dirty-tree": r"session\.py",
    "dirty-tree-untracked": r"scratch_t0223\.txt",
    "out-dir-exists": r"exist",
    "no-merge-t0223-a": r"T-0223-a",
    "no-merge-t0213-a2a": r"T-0213-a2a",
    "branch-merge-only": r"T-0223-a",
    "attempts-off-pin": r"attempts\.json|attempts",
    "page-off-pin": r"P3",
    "page-empty-judge-keywords": r"M2",
    "page-no-false-blame-list": r"P4",
    "page-false-blame-a-string": r"P4",
    "instructions-refused": r"T0223-RIG-REFUSAL|validate_instructions|instructions",
    "visual-tokens-not-1120": r"1120|visual",
    "ps-fails": r"P-S",
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_before_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """B1: each refusal exits 2 with its own stderr line, before any socket (no ``socket.connect`` through the
    doubles nor through the audit hook), leaving an existing out-dir as it was: a non-loopback ``--base-url``
    (``http://localhost.example:8001/v1`` too: a name is not loopback); a dirty tree (an untracked file is dirty); an
    existing out-dir; no ``Merge T-0223-a:`` or no ``Merge T-0213-a2a:`` (a branch merge or a lookalike subject does
    not count); ``attempts.json`` or a page off its pinned sha256; a page with empty ``judge_keywords`` or without a
    ``false_blame_keywords`` list (pins re-made so only the schema is wrong); ``validate_instructions`` refusing
    ``TUTOR_INSTRUCTIONS``; 1120 not in force; P-S failing (the system message drops the user's block)."""
    from local_astra import cli
    from local_astra.conductor import prompt_builder

    rig = Rig(tmp_path, monkeypatch)
    h, assets = rig.h, rig.assets
    argv = rig.argv()
    if case == "url-not-loopback":
        argv = rig.argv("http://10.0.0.1:8001/v1")
    elif case == "url-localhost-lookalike":
        argv = rig.argv("http://localhost.example:8001/v1")
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "dirty-tree-untracked":
        rig.porcelain = "?? scratch_t0223.txt\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-t0223-a":
        rig.merges = tuple(m for m in rig.merges if m != MERGE_A)
    elif case == "no-merge-t0213-a2a":
        rig.merges = tuple(m for m in rig.merges if m != MERGE_A2A)
    elif case == "branch-merge-only":
        rig.merges = (*(m for m in rig.merges if m != MERGE_A), *LOOKALIKES_A)
    elif case == "attempts-off-pin":
        monkeypatch.setattr(h, "ATTEMPTS_SHA256", "0" * 64)
    elif case == "page-off-pin":
        (assets.pages_dir / "P3.png").write_bytes(_png(42))
    elif case in (
        "page-empty-judge-keywords",
        "page-no-false-blame-list",
        "page-false-blame-a-string",
    ):
        obj = _attempts()
        for a in obj["attempts"]:
            if case == "page-empty-judge-keywords" and a["id"] == "M2":
                a["judge_keywords"] = []
            if case == "page-no-false-blame-list" and a["id"] == "P4":
                del a["false_blame_keywords"]
            if case == "page-false-blame-a-string" and a["id"] == "P4":
                a["false_blame_keywords"] = "p4 first line is wrong"  # a phrase, not a list
        assets.write_attempts(obj)
        monkeypatch.setattr(h, "ATTEMPTS_SHA256", assets.attempts_sha)
    elif case == "instructions-refused":
        real = prompt_builder.validate_instructions

        def refusing(text: str) -> str | None:
            return "T0223-RIG-REFUSAL" if text == TUTOR else real(text)

        monkeypatch.setattr(prompt_builder, "validate_instructions", refusing)
        _patch_object(monkeypatch, h, real, refusing)
    elif case == "visual-tokens-not-1120":
        monkeypatch.setattr(cli, "LIVE_DEFAULT_VISUAL_TOKENS", 560)
    elif case == "ps-fails":
        real_sys = prompt_builder.PromptBuilder.system_message

        def dropped(self: Any) -> Any:
            msg = real_sys(self)
            text = msg.content[0].text
            cut = text.split("\n" + HEADER, 1)[0]
            return msg.model_copy(update={"content": (prompt_builder.TextPart(text=cut),)})

        monkeypatch.setattr(prompt_builder.PromptBuilder, "system_message", dropped)
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = rig.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    assert _AUDIT["events"] == [], (case, _AUDIT["events"])
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    if before is not None:
        assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert re.search(REFUSALS[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B1, the too-wide direction: with every merge, the pins intact, the tutor text valid, 1120 in force and P-S
    passing on CPU, ``main`` refuses nothing; it asks for the engine and, finding none, exits 1."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


# =====================================================================================================
# B2: constants, the frame, the session, order, budget
# =====================================================================================================


def test_constants() -> None:
    """B2/criterion 5: the typed texts and numbers; the tutor text is a valid instructions block."""
    from local_astra.conductor import prompt_builder

    assert api("QUESTION") == QUESTION
    assert api("TUTOR_INSTRUCTIONS") == TUTOR
    assert prompt_builder.validate_instructions(TUTOR) is None
    assert prompt_builder.USER_INSTRUCTIONS_HEADER == HEADER, "premise: T-0213's header"
    assert api("ORDER_SEED") == 0
    assert api("TURN_TIMEOUT_S") == 90
    assert api("RUN_BUDGET_S") == 14 * 60  # R-4 (amends B2's 11 min)


def test_frame_message_is_the_hud_shape(tmp_path: Path) -> None:
    """B2: one ``astra.frame`` from the screen: the PNG resized to ``policy.target_size(bucket_for(SCREEN, w, h))``,
    a JPEG at ``policy.JPEG_QUALITY`` (its quantisation tables are a quality-85 encoder's, not Pillow's default 75),
    ``width``/``height`` the JPEG's own, ``ts_ms`` the clock's, a 16-hex phash, no title or app (pixel-first); it
    round-trips through the HUD codec."""
    from PIL import Image

    from local_astra.contracts.codec import decode, encode
    from local_astra.contracts.perception import Source
    from local_astra.contracts.protocol import AstraFrame
    from local_astra.perception import policy

    png = tmp_path / "M1.png"
    png.write_bytes(_png(0))
    frame = api("frame_message")(png, 4321)
    assert isinstance(frame, AstraFrame), type(frame)
    want = policy.target_size(1700, 2200, policy.bucket_for(Source.SCREEN, 1700, 2200))
    assert want[0] < 1700 and want[1] < 2200, "premise: the page is downscaled"
    jpeg = base64.b64decode(frame.jpeg_b64)
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG"
        assert im.size == want, (im.size, want)
        tables = im.quantization
    ref = io.BytesIO()
    Image.new("RGB", want, (128, 128, 128)).save(ref, format="JPEG", quality=policy.JPEG_QUALITY)
    with Image.open(io.BytesIO(ref.getvalue())) as rim:
        assert tables == rim.quantization, "the JPEG is not at policy.JPEG_QUALITY"
    assert (frame.width, frame.height) == want
    assert frame.source == Source.SCREEN and frame.ts_ms == 4321
    assert isinstance(frame.phash, str) and re.fullmatch(r"[0-9a-f]{16}", frame.phash), frame.phash
    assert frame.title is None and frame.app is None
    assert decode(encode(frame)) == frame


def test_session_matches_cli_tools_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B2: the Session ``main`` builds (P-S, reading 5) is cli's ``--tools off`` session at 1120: no tool kit, no
    tool guidance, ``visual_tokens == 1120``, no live-only keyword, no ``search`` keyword; the system prompt is the
    default (the instructions arrive through the settings path, never ``SessionConfig.system_prompt``)."""
    from local_astra.conductor import session as session_mod

    seen: list[dict[str, Any]] = []
    real_init = session_mod.Session.__init__

    def spy(self: Any, *a: Any, **k: Any) -> None:
        seen.append(dict(k, _args=a))
        real_init(self, *a, **k)

    monkeypatch.setattr(session_mod.Session, "__init__", spy)
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.main(rig.argv())
    capsys.readouterr()
    assert rc == 1, rc
    assert seen, "main built no Session on CPU (P-S)"
    live_only = {
        "live_feedback",
        "recover_thought_calls",
        "auto_overview",
        "overview_frame",
        "action_outcome",
        "repeat_guard",
        "screen_reminder",
        "completion_check",
        "page_fact",
        "reground",
        "reground_llm",
    }
    for k in seen:
        assert k.get("tool_kit") is None, k.get("tool_kit")
        assert k.get("tool_guidance") is None
        assert k.get("visual_tokens") == 1120, k.get("visual_tokens")
        assert not (live_only & {n for n, v in k.items() if v}), sorted(live_only & set(k))
        assert "search" not in k
        config = k.get("config")
        if config is not None:
            assert config.system_prompt == session_mod.DEFAULT_SYSTEM_PROMPT


def _order0() -> list[tuple[str, int]]:
    """B2's order by hand: the page-major list shuffled once by ``random.Random(0)``."""
    turns = [(p, r) for p in PAGE_IDS for r in range(3)]
    random.Random(0).shuffle(turns)
    return turns


def test_order_seed0() -> None:
    """B2: ``order(0)`` is the hand oracle (30 turns, each once, every page 3 times) and differs from seed 1."""
    got = [tuple(x) for x in api("order")(0)]
    assert got == _order0(), got
    assert len(set(got)) == 30 and all(sum(1 for p, _ in got if p == q) == 3 for q in PAGE_IDS)
    assert [tuple(x) for x in api("order")(1)] != got


@pytest.mark.parametrize(
    ("elapsed", "stop"),
    [(0.0, False), (600.0, False), (705.0, False), (705.5, True), (760.0, True), (840.0, True)],
)
def test_budget_gate(elapsed: float, stop: bool) -> None:
    """B2 as amended by R-4: no turn starts unless ``TURN_TIMEOUT_S + 45`` s still fit in ``RUN_BUDGET_S`` = 840 s:
    840 - 135 = 705 s is the last start (both directions: a gate that is absent, or one without the 45 s report
    margin, lets 705.5 and 760 start; one that is too wide stops 705)."""
    assert bool(api("budget_stop")(elapsed)) is stop, (elapsed, stop)


# =====================================================================================================
# B4: the judge
# =====================================================================================================

PAGE_A = {
    "id": "P1",
    "judge_keywords": ["cos", "normal|perpendicular"],
    "false_blame_keywords": ["sin θ is wrong"],
}
PAGE_FOLD = {"id": "X1", "judge_keywords": ["strasse"], "false_blame_keywords": []}
PAGE_WS = {"id": "X2", "judge_keywords": ["cos θ"], "false_blame_keywords": []}
PAGE_UPPER = {
    "id": "X3",
    "judge_keywords": ["COS", "Normal|PERPENDICULAR"],
    "false_blame_keywords": [],
}
PAGE_NONE: dict[str, Any] = {"id": "X4", "judge_keywords": [], "false_blame_keywords": []}

#: case -> (page, spoken, shown, located, spoken-only located, matched entries)
JUDGE: dict[str, tuple[dict[str, Any], str, str, bool, bool, list[str]]] = {
    "all-keywords": (
        PAGE_A,
        "Line 3: the normal force needs cos θ.",
        "",
        True,
        True,
        ["cos", "normal|perpendicular"],
    ),
    "second-alternative": (
        PAGE_A,
        "The perpendicular force is mg cos θ.",
        "",
        True,
        True,
        ["cos", "normal|perpendicular"],
    ),
    "one-missing": (PAGE_A, "Line 3 needs cos θ.", "", False, False, ["cos"]),
    "casefold": (
        PAGE_A,
        "LINE 3: THE NORMAL FORCE NEEDS COS Θ.",
        "",
        True,
        True,
        ["cos", "normal|perpendicular"],
    ),
    "casefold-not-lower": (PAGE_FOLD, "Turn left at the STRAßE.", "", True, True, ["strasse"]),
    "false-blame": (
        PAGE_A,
        "The normal force needs cos θ, and SIN Θ IS WRONG too.",
        "",
        False,
        False,
        ["cos", "normal|perpendicular"],
    ),
    "empty": (PAGE_A, "", "", False, False, []),
    "shown-only": (
        PAGE_A,
        "Have a look at the card.",
        "Normal force: N = mg cos θ",
        True,
        False,
        ["cos", "normal|perpendicular"],
    ),
    "split-across-channels": (
        PAGE_A,
        "You need cos θ here.",
        "The normal force",
        True,
        False,
        ["cos", "normal|perpendicular"],
    ),
    "no-other-normalisation": (PAGE_WS, "It should be cos  θ.", "", False, False, []),
    "joined-with-a-space": (
        PAGE_A,
        "normal force with co",
        "s θ",
        False,
        False,
        ["normal|perpendicular"],
    ),
    # R-2: the alternative is casefolded too (a keyword written in capitals still matches lower-case speech)
    "uppercase-alternative": (
        PAGE_UPPER,
        "the perpendicular force needs cos θ.",
        "",
        True,
        True,
        ["COS", "Normal|PERPENDICULAR"],
    ),
    # a page without judge keywords is never located, whatever is said (and nothing said is not located either)
    "no-keywords-empty": (PAGE_NONE, "", "", False, False, []),
    "no-keywords-said": (PAGE_NONE, "Line 2 is wrong.", "", False, False, []),
}


@pytest.mark.parametrize("case", list(JUDGE))
def test_judge_table(case: str) -> None:
    """B4 (R-1, R-2), by hand: every entry (any ``|`` alternative) a casefold substring of spoken + shown joined with
    a space, and no false-blame phrase -> located; one entry missing, a false blame, or nothing said -> not;
    ``casefold`` not ``lower`` (``ß``); no other normalisation (a doubled space misses); a keyword only on the card
    -> located and not spoken-only located; the join is a space, so a keyword cut across the two channels misses."""
    page, spoken, shown, loc, spk, matched = JUDGE[case]
    got = api("judge")(spoken, shown, page)
    assert isinstance(got, tuple) and len(got) == 3, got
    assert (bool(got[0]), bool(got[1])) == (loc, spk), (case, got)
    assert sorted(got[2]) == sorted(matched), (case, got[2])


def test_judge_reads_the_page_it_is_given() -> None:
    """B4: keywords come from the page argument (a judge with built-in keywords for real ids is not the pinned
    file's): the same answer is located for a page whose keywords it contains and not for one whose it does not."""
    judge = api("judge")
    text = "zzqx-one and zzqx-two"
    yes = {"id": "P1", "judge_keywords": ["zzqx-one", "zzqx-two"], "false_blame_keywords": []}
    no = {"id": "P1", "judge_keywords": ["zzqx-one", "zzqx-three"], "false_blame_keywords": []}
    assert bool(judge(text, "", yes)[0]) is True
    assert bool(judge(text, "", no)[0]) is False


# =====================================================================================================
# B3/B5/B6: a synthetic run dir through finish_turn and write_report
# =====================================================================================================


def _ui(text: str) -> str:
    return (
        CS
        + "\n"
        + json.dumps(
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["t"]},
                        {"id": "t", "component": "Text", "text": text},
                    ],
                },
            }
        )
    )


def _jpeg(index: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 88), (10 * index, 200, 100)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _texts(page: str, kind: str, r: int) -> tuple[str | None, str | None, str]:
    """(spoken, shown, thought) for one turn kind; keywords from :func:`_kw` (first alternative on even repeats, the
    second on odd ones, upper case on repeat 0)."""
    q = page.lower()
    first = f"{q}word" if r % 2 == 0 else f"{q}alt"
    fix = f"{q}fix"
    if r == 0:
        first, fix = first.upper(), fix.upper()
    thought = "The student wants a check."
    if kind == "located":
        return f"Line 2 is where it goes wrong: {first}. The fix is {fix}.", None, thought
    if kind == "miss":
        return f"Line 2 is where it goes wrong: {first}.", None, thought
    if kind == "false_blame":
        return (
            f"Line 2 is where it goes wrong: {first}. The fix is {fix}. Also, {q.upper()} FIRST LINE IS WRONG.",
            None,
            thought,
        )
    if kind == "shown_only":
        return "Have a look at the card I put up.", f"{first}: the fix is {fix}", thought
    if kind == "split":
        return f"Line 2 is where it goes wrong: {first}.", f"The fix is {fix}", thought
    if kind == "two_sentences":
        # two spoken sentences (two tts.speak commands), one keyword in each: located and spoken-only located
        return f"Line 2 is where it goes wrong: {first}.\n\nThe fix is {fix}.", None, thought
    if kind == "thought_only":
        return "That looks fine to me.", None, f"I think {first} and {fix} matter here."
    if kind == "empty":
        return None, None, thought
    raise KeyError(kind)


def _turn_files(
    out: Path, page: str, r: int, kind: str, index: int, latency: int, fault: str | None
) -> dict[str, Any]:
    """Write one turn's session log and append its turn line (what the real ``Session`` does); return the runner's
    raw trial (with the meter's body summaries, ``http_non_200`` and ``engine_up_after``). ``fault``, each one
    clause: B3 on the log: ``no-image`` (the generate body has no image), ``two-images``, ``wrong-image``,
    ``tools`` (a body declares a tool), ``late-settings`` (the set applied after the turn: the requests carry no
    block, the line has no settings fields), ``settings-after-usertext`` (an ``immediate`` change logged after the
    ``UserText`` step, nothing else wrong), ``applied-at-wrong`` (in place, ``after_turn``), ``settings-rev-0`` (the
    turn line only), ``no-llm-request``, ``two-system-texts`` (both end with the block; the line names one),
    ``stored-differs``; B3 on the bodies (R-5): ``body-mst`` (560), ``body-no-generate``, ``body-no-image``,
    ``body-two-images``, ``body-tools``; R-3: ``llm-error`` (an ``error`` record, ``what`` "llm.generate"),
    ``http-500``, ``engine-down``; and ``exception``, ``timed-out``."""
    from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
    from local_astra.telemetry.session_log import redact

    jpeg = _jpeg(index)
    b64 = base64.b64encode(jpeg).decode("ascii")
    url = "data:image/jpeg;base64," + b64
    digest = _sha(b64.encode("utf-8"))[:8]
    other = "data:image/jpeg;base64," + base64.b64encode(_jpeg(index + 50)).decode("ascii")
    late = fault == "late-settings"
    system = SYS_BASE if late else SYS_BASE + "\n" + BLOCK
    sha12 = _sha(system.encode("utf-8"))[:12]
    name = f"turn-{page}-{r}.jsonl"
    sid = f"t0223{page}{r}"
    user_ms = 1000
    recs: list[dict[str, Any]] = []

    def add(kind_: str, data: dict[str, Any], turn: Any = "t1") -> None:
        recs.append(
            {"seq": len(recs), "t_ms": 10 * len(recs), "kind": kind_, "turn_id": turn, "data": data}
        )

    change = {
        "rev": 1,
        "changes": {
            "user_instructions": {
                "old": {"sha12": "", "chars": 0},
                "new": {"sha12": _sha(TUTOR.encode())[:12], "chars": len(TUTOR)},
            }
        },
        "source": "ui",
        "turn_seq": 1 if late else 0,
        "applied_at": "after_turn" if late or fault == "applied-at-wrong" else "immediate",
        "system_sha12_old": _sha(SYS_BASE.encode())[:12],
        "system_sha12_new": _sha((SYS_BASE + "\n" + BLOCK).encode())[:12],
    }
    add("log.started", {"pid": 4242, "path": name}, None)
    after_user = late or fault == "settings-after-usertext"
    if not after_user:
        add("settings.change", change, None)
    add(
        "machine.step",
        {"event": {"type": "user.text", "t_ms": user_ms, "text": QUESTION}, "commands": []},
    )
    if fault == "settings-after-usertext":
        add("settings.change", change, None)

    def prompt(phase: str, with_text: bool) -> dict[str, Any]:
        parts: list[Any] = []
        if not (fault == "no-image" and phase == "generate"):
            parts.append(
                ImagePart(url=other if fault == "wrong-image" else url, phash="0123456789abcdef")
            )
        if fault == "two-images" and phase == "generate":
            parts.append(ImagePart(url=other, phash="fedcba9876543210"))
        if with_text:
            parts.append(TextPart(text=QUESTION))
        sys_text = system
        if fault == "two-system-texts" and phase == "prefill":
            sys_text = SYS_BASE + " Another sentence.\n" + BLOCK
        p = Prompt(
            messages=(
                Message(role="system", content=(TextPart(text=sys_text),)),
                Message(role="user", content=tuple(parts)),
            ),
            prefix_len=2,
            thinking=True,
            tools=({"type": "function", "function": {"name": "click_at"}},)
            if fault == "tools" and phase == "generate"
            else None,
        )
        return redact(p.model_dump(mode="json"))

    for phase, rid, with_text in (("prefill", "t1-prefill", False), ("generate", "t1-gen", True)):
        if fault == "no-llm-request":
            break
        add(
            "llm.request",
            {
                "phase": phase,
                "request_id": rid,
                "attempt_id": f"{sid}-{rid}",
                "options": None,
                "xargs": None,
                "prompt": prompt(phase, with_text),
            },
        )
    spoken, shown, thought = _texts(page, kind, r)
    add(
        "llm.commit",
        {
            "request_id": "t1-gen",
            "attempt_id": f"{sid}-t1-gen",
            "canvas_idx": 0,
            "segment": "thought",
            "text": thought,
            "thought_open": False,
        },
    )
    if spoken is not None:
        add(
            "llm.commit",
            {
                "request_id": "t1-gen",
                "attempt_id": f"{sid}-t1-gen",
                "canvas_idx": 1,
                "segment": "answer",
                "text": spoken,
                "thought_open": False,
            },
        )
    if shown is not None:
        add(
            "llm.commit",
            {
                "request_id": "t1-gen",
                "attempt_id": f"{sid}-t1-gen",
                "canvas_idx": 2,
                "segment": "ui",
                "text": _ui(shown),
                "thought_open": False,
            },
        )
    if fault == "llm-error":
        add(
            "error",
            {"what": "llm.generate", "request_id": "t1-gen", "error": "HTTPStatusError('500')"},
        )
    for sentence in spoken.split("\n\n") if spoken is not None else []:
        add(
            "machine.step",
            {
                "event": {"type": "llm.commit"},
                "commands": [{"type": "tts.speak", "text": sentence, "source": sentence}],
            },
        )
    if kind == "empty":
        add(
            "machine.step",
            {
                "event": {"type": "llm.done"},
                "commands": [{"type": "play_clip", "kind": "not_finished"}],
            },
        )
    if late:
        add("settings.change", change, None)
    regime: dict[str, Any] = {
        "profile": "lean",
        "canvas_length": 256,
        "max_denoising_steps": 48,
        "thinking": True,
        "vllm_version": "0.29.0",
        "model": "diffusion-gemma",
        "stt": "fake",
        "tts": "fake",
        "audio_path": "null",
        "git_commit": GIT_SHA,
        "seed": None,
        "tools": "off",
    }
    if not late:
        regime.update(
            {
                "settings_rev": 0 if fault == "settings-rev-0" else 1,
                "system_sha12": sha12,
                "instructions_edited": True,
            }
        )
    spans: dict[str, list[int]] = {}
    if spoken is not None:
        spans["tts.first_audio"] = [user_ms + latency, user_ms + latency + 300]
    line = {
        "turn_id": "t1",
        "session_id": sid,
        "spans": spans,
        "regime": regime,
        "input": "text",
        "outcome": "spoken" if spoken is not None else "silent",
        "session_log": str(out / "sessions" / name),
        "extras": {},
    }
    add("turn.record", line)
    add("log.stopped", {"reason": "session closed", "records": len(recs) + 1, "dropped": 0}, None)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    (out / "sessions" / name).write_text(
        "".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8"
    )
    with (out / "turns.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    stored = _jpeg(index + 99) if fault == "stored-differs" else jpeg
    sha = _sha(jpeg)

    def body(phase: str, rid: str) -> dict[str, Any]:
        images = [sha]
        if fault == "body-no-image" and phase == "generate":
            images = []
        if fault == "body-two-images" and phase == "generate":
            images = [sha, _sha(_jpeg(index + 50))]
        return {
            "label": f"{phase}:{sid}-{rid}",
            "phase": phase,
            "images": images,
            "max_soft_tokens": 560 if fault == "body-mst" and phase == "generate" else 1120,
            "tools": fault == "body-tools" and phase == "generate",
        }

    bodies = [body("prefill", "t1-prefill")]
    if fault != "body-no-generate":
        bodies.append(body("generate", "t1-gen"))
    return {
        "page": page,
        "repeat": r,
        "session_log": name,
        "exception": "RuntimeError('T-0223 rig: raised')" if fault == "exception" else None,
        "timed_out": fault == "timed-out",
        "frame_sha256": sha,
        "stored_sha256": _sha(stored),
        "frame_digest": digest,
        "user_text_ms": user_ms,
        "settings_echo": {"rev": 0 if late else 1, "refused": []},
        "wall_s": 15.0 + index,
        "bodies": bodies,
        "http_non_200": 1 if fault == "http-500" else 0,
        "engine_up_after": fault != "engine-down",
    }


#: the clean run, by hand: kinds per page and repeat
PLAN: dict[str, tuple[str, str, str]] = {
    "M1": ("located", "located", "miss"),  # 2/3, spoken 2
    "M2": ("located", "shown_only", "miss"),  # 2/3, spoken 1
    "M3": ("miss", "miss", "located"),  # 1/3, spoken 1
    "M4": ("false_blame", "located", "located"),  # 2/3, spoken 2
    "M5": ("thought_only", "located", "miss"),  # 1/3, spoken 1
    "P1": ("located", "located", "two_sentences"),  # 3/3, spoken 3 (r2: two spoken sentences)
    "P2": ("empty", "located", "miss"),  # 1/3, spoken 1
    "P3": ("located", "located", "false_blame"),  # 2/3, spoken 2
    "P4": ("miss", "miss", "miss"),  # 0/3, spoken 0
    "P5": ("shown_only", "split", "located"),  # 3/3, spoken 1
}
WANT_LOCATED = {
    "M1": 2,
    "M2": 2,
    "M3": 1,
    "M4": 2,
    "M5": 1,
    "P1": 3,
    "P2": 1,
    "P3": 2,
    "P4": 0,
    "P5": 3,
}
WANT_SPOKEN = {
    "M1": 2,
    "M2": 1,
    "M3": 1,
    "M4": 2,
    "M5": 1,
    "P1": 3,
    "P2": 1,
    "P3": 2,
    "P4": 0,
    "P5": 1,
}
#: pages located in >= 2 of 3: M1, M2, M4, P1, P3, P5
WANT_READING_N = 6


def _latency(index: int) -> int:
    """A distinct latency per turn index (ms), so p50 and max are unambiguous."""
    return 900 + 37 * index + (index % 7) * 11


def _manifest(assets: Assets, stopped: str | None = None) -> dict[str, Any]:
    return {
        "regime": {
            "commit": GIT_SHA,
            "profile": "lean",
            "canvas_length": 256,
            "max_denoising_steps": 48,
            "thinking": True,
            "engine_version": "vllm-0.29.0",
            "visual_tokens": 1120,
            "tools": "off",
            "search": False,
            "order_seed": 0,
            "n": 3,
            "turns": 30,
            "concurrency": 1,
            "engine_seed": None,
            "turn_timeout_s": 90,
            "run_budget_s": 840,
            "attempts_sha256": assets.attempts_sha,
            "input": "typed (UserText) T0223; fake STT/TTS",
        },
        "attempts_sha256": assets.attempts_sha,
        "page_sha256": dict(assets.page_sha),
        "stopped": stopped,
    }


def _finish(out: Path, page: str, r: int, trial: dict[str, Any]) -> dict[str, Any]:
    fn = api("finish_turn")
    assert len(inspect.signature(fn).parameters) >= 4, list(inspect.signature(fn).parameters)
    got = fn(out, page, r, dict(trial))
    assert isinstance(got, dict), type(got)
    return got


def _write_run(
    tmp_path: Path,
    mp: pytest.MonkeyPatch,
    *,
    faults: dict[tuple[str, int], str] | None = None,
    drop: tuple[str, int] | None = None,
    not_run: tuple[str, int] | None = None,
    probes: dict[str, bool | None] | None = None,
    stopped: str | None = None,
    plan: dict[str, tuple[str, str, str]] | None = None,
) -> tuple[Path, dict[str, Any], Assets, list[int]]:
    """A run dir: ``order.json``, ``manifest.json``, ``probes.json`` and every turn of ``order(0)`` written by hand
    and finished by the harness's ``finish_turn``; ``trials.jsonl`` of the finished trials; then ``write_report``.
    Returns the dir, the report, the assets and the latencies of the turns that spoke."""
    h = harness()
    assets = Assets(tmp_path / "assets")
    assets.inject(mp, h)
    out = tmp_path / "run"
    (out / "sessions").mkdir(parents=True)
    (out / "turns.jsonl").touch()
    order = _order0()
    (out / "order.json").write_text(json.dumps([list(x) for x in order]), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(_manifest(assets, stopped)), encoding="utf-8")
    pr = {"P-S": True, "P0": True, **(probes or {})}
    (out / "probes.json").write_text(
        json.dumps({k: {"passed": v} for k, v in pr.items() if v is not None}), encoding="utf-8"
    )
    plan = plan or PLAN
    trials: list[dict[str, Any]] = []
    latencies: list[int] = []
    for index, (page, r) in enumerate(order):
        if drop == (page, r):
            continue
        if not_run == (page, r):
            raw = {
                "page": page,
                "repeat": r,
                "session_log": None,
                "exception": "harness: RuntimeError('T-0223 rig: died')",
                "timed_out": False,
                "frame_sha256": None,
                "stored_sha256": None,
                "frame_digest": None,
                "user_text_ms": None,
                "settings_echo": None,
                "wall_s": 0.0,
            }
        else:
            kind = plan[page][r]
            lat = _latency(index)
            raw = _turn_files(out, page, r, kind, index, lat, (faults or {}).get((page, r)))
            if kind != "empty":
                latencies.append(lat)
        trials.append(_finish(out, page, r, raw))
    (out / "trials.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in trials), encoding="utf-8"
    )
    api("write_report")(out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return out, report, assets, latencies


def _turn(report: dict[str, Any], page: str, r: int) -> dict[str, Any]:
    rows = [
        t
        for t in report.get("turns") or []
        if (t.get("page"), int(t.get("repeat", -1))) == (page, r)
    ]
    assert len(rows) == 1, (page, r, rows)
    return rows[0]


def test_report_over_synthetic_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B4/B5/B6 over a clean 30-turn dir, every number by hand (``PLAN``): per page k/3 located and k/3 spoken-only;
    "6 of 10 pages located in ≥ 2 of 3 turns" (the line in report.json and report.md); a keyword only in the
    thought is not located (M5 r0) and only on the card is located but not spoken-only (M2 r1, P5 r0); the empty
    turn (P2 r0) is not located and has no latency; p50 (low median) and max over the 29 turns that spoke; label
    ``measured`` with no violation; every spoken text verbatim in report.md; the clip kind is listed, not judged."""
    out, report, _assets, latencies = _write_run(tmp_path, monkeypatch)
    assert report.get("violations") == [], report.get("violations")
    assert report.get("label") == "measured", report.get("label")
    pages = report.get("pages") or {}
    for p in PAGE_IDS:
        assert (
            int(pages[p]["located"]),
            int(pages[p]["spoken_only_located"]),
            int(pages[p]["n"]),
        ) == (
            WANT_LOCATED[p],
            WANT_SPOKEN[p],
            3,
        ), (p, pages[p])
    line = str(report.get("reading") or "")
    assert f"{WANT_READING_N} of 10 pages located in ≥ 2 of 3 turns" in line, line
    md = (out / "report.md").read_text(encoding="utf-8")
    assert line in md
    assert len(latencies) == 29, "premise: one silent turn"
    lat = report.get("latency") or {}
    assert (lat.get("p50_ms"), lat.get("max_ms"), lat.get("n")) == (
        statistics.median_low(latencies),
        max(latencies),
        29,
    ), lat
    m5 = _turn(report, "M5", 0)
    assert not m5.get("located") and "m5word" not in str(m5.get("spoken", "")).casefold()
    for page, r in (("M2", 1), ("P5", 0)):
        t = _turn(report, page, r)
        assert t.get("located") and not t.get("spoken_only_located"), (page, r, t)
        assert f"{page.lower()}fix" in str(t.get("shown", "")).casefold(), t.get("shown")
    p2 = _turn(report, "P2", 0)
    assert not p2.get("located")
    assert "not_finished" in json.dumps(p2), p2
    for page, r in (("P1", 1), ("M4", 0), ("P3", 2)):
        spoken, _shown, _t = _texts(page, PLAN[page][r], r)
        assert spoken is not None and spoken in md, (page, r)
    assert _turn(report, "M4", 0).get("located") is False, "false blame (casefolded) is not located"
    # two spoken sentences, one keyword in each: both are the turn's spoken text
    p1 = _turn(report, "P1", 2)
    assert p1.get("located") and p1.get("spoken_only_located"), p1
    # B5: the cold first turn is marked (order(0)'s first turn spoke)
    first = _order0()[0]
    assert PLAN[first[0]][first[1]] != "empty", "premise: the first turn spoke"
    assert lat.get("cold_first_turn_ms") == _latency(0), lat


def test_every_eval_row_passes_on_a_harness_built_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6 (binding lesson): every run row of the eval, which imports no harness, passes on the clean dir the
    harness's writers built (the harness and the oracle agree)."""
    out, _report, assets, _ = _write_run(tmp_path, monkeypatch)
    ev = _eval()
    run = ev.load_run(out, assets.attempts_path)
    for row in ev.RUN_ROWS:
        row(run)


#: B3 fault -> the turn it is seeded on
EVIDENCE_VOID = (
    "no-image",
    "two-images",
    "wrong-image",
    "tools",
    "late-settings",
    "stored-differs",
    # single clauses (F2): each alone voids
    "settings-after-usertext",
    "applied-at-wrong",
    "settings-rev-0",
    "no-llm-request",
    "two-system-texts",
    # R-5: the meter's body summaries
    "body-mst",
    "body-no-generate",
    "body-no-image",
    "body-two-images",
    "body-tools",
)


@pytest.mark.parametrize("fault", EVIDENCE_VOID)
def test_report_void_on_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """B3 (adversarial), each makes the run ``void`` with a violation naming the turn: a generate body without the
    image; one with a second image; the image of another frame; a body declaring a tool; the settings set applied
    after ``UserText`` (the requests carry no block, the turn line has ``settings_rev`` 0, the change is
    ``after_turn``: the turn is invalid and the run void); the stored frame differing from the posted one; and each
    single clause alone (an ``immediate`` change logged after the ``UserText`` step; ``applied_at`` not
    ``immediate``; the turn line's ``settings_rev`` 0; no ``llm.request``; two different system texts); and each
    body summary fault (R-5: ``max_soft_tokens`` 560, no generate body, a generate body without the image or with
    two, a body with tools). The eval agrees (its label row passes, its evidence row passes)."""
    key = ("P1", 1)
    out, report, assets, _ = _write_run(tmp_path, monkeypatch, faults={key: fault})
    assert report.get("label") == "void", (fault, report.get("label"))
    v = list(report.get("violations") or []) + list(report.get("body_violations") or [])
    assert v and all("P1/r1" in str(x) for x in v), (fault, v)
    ev = _eval()
    run = ev.load_run(out, assets.attempts_path)
    ev.test_label_re_derived(run)
    ev.test_evidence_re_derived(run)


@pytest.mark.parametrize(
    ("probe", "passed"), [("P-S", False), ("P0", False), ("P-S", None), ("P0", None)]
)
def test_report_void_on_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: str, passed: bool | None
) -> None:
    """B2/B6: a failed P-S or P0, or one missing from ``probes.json`` (``None``: not recorded at all), is ``void``
    whatever the turns say; the eval agrees."""
    out, report, assets, _ = _write_run(tmp_path, monkeypatch, probes={probe: passed})
    assert report.get("label") == "void", report.get("label")
    ev = _eval()
    ev.test_label_re_derived(ev.load_run(out, assets.attempts_path))


#: R-3 and the errored/timed-out turns: each on a turn whose text locates the mistake (P1 r1, "located")
ERRORED = ("llm-error", "http-500", "engine-down", "exception", "timed-out")


@pytest.mark.parametrize("fault", ERRORED)
def test_errored_turn_is_not_located(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """R-3: a turn with an ``llm`` error record in its log, a non-200 attempt, or ``engine_up_after`` False is
    errored; so is a turn that raised or timed out. Each has a session log and keyword-bearing spoken text, and
    each is not located (P1 drops from 3/3 to 2/3), the run is ``incomplete`` (not void: no B3 violation is
    reported for it), and the eval re-derives the same from the log and the trial line."""
    key = ("P1", 1)
    assert PLAN["P1"][1] == "located", "premise: the faulted turn's text locates the mistake"
    out, report, assets, _ = _write_run(tmp_path, monkeypatch, faults={key: fault})
    t = _turn(report, *key)
    assert not t.get("located") and not t.get("spoken_only_located"), (fault, t)
    assert int((report.get("pages") or {})["P1"]["located"]) == 2, report.get("pages")
    assert report.get("label") == "incomplete", (
        fault,
        report.get("label"),
        report.get("violations"),
    )
    ev = _eval()
    run = ev.load_run(out, assets.attempts_path)
    for row in (ev.test_label_re_derived, ev.test_verdicts_re_derived, ev.test_k_of_3_re_derived):
        row(run)
    trial = next(x for x in run["trials"] if (x["page"], int(x["repeat"])) == key)
    assert ev.status(trial, run["sessions"][trial["session_log"]]) != "ran", fault


@pytest.mark.parametrize("how", ["dropped", "not-run", "stopped"])
def test_report_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """B6: a turn of ``order.json`` with no trial line, a not-run turn (counted errored) or a stopped run is
    ``incomplete``, not void (a not-run turn carries no B3 evidence); the page keeps n = 3 and the missing turn is
    not located (P1 drops to 2/3)."""
    key = ("P1", 2)
    kw: dict[str, Any] = {}
    if how == "dropped":
        kw["drop"] = key
    elif how == "not-run":
        kw["not_run"] = key
    else:
        kw["stopped"] = "T-0223 rig: budget"
    out, report, assets, _ = _write_run(tmp_path, monkeypatch, **kw)
    assert report.get("label") == "incomplete", (how, report.get("label"), report.get("violations"))
    assert report.get("violations") == [], report.get("violations")
    p1 = (report.get("pages") or {})["P1"]
    assert int(p1["n"]) == 3
    assert int(p1["located"]) == (3 if how == "stopped" else 2), p1
    ev = _eval()
    ev.test_label_re_derived(ev.load_run(out, assets.attempts_path))


def test_report_has_no_bar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: report-only: no key anywhere in report.json names a bar, threshold or pass/fail verdict, and neither the
    reading line nor report.md says PASS or FAIL."""
    out, report, _a, _ = _write_run(tmp_path, monkeypatch)

    def keys(obj: Any) -> list[str]:
        if isinstance(obj, dict):
            return [str(k) for k in obj] + [x for v in obj.values() for x in keys(v)]
        if isinstance(obj, list):
            return [x for v in obj for x in keys(v)]
        return []

    bad = [
        k for k in keys(report) if re.search(r"bar|threshold|pass|fail|verdict", k, re.IGNORECASE)
    ]
    assert bad == [], bad
    md = (out / "report.md").read_text(encoding="utf-8")
    assert not re.search(r"\b(PASS|FAIL)\b", md), "report.md states a pass/fail"
    assert not re.search(r"\b(pass|fail)", str(report.get("reading")), re.IGNORECASE)


# =====================================================================================================
# the eval's rows catch seeded faults (they are not vacuous); the eval imports no harness
# =====================================================================================================


def _rw_report(out: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    p = out / "report.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    edit(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _flip_turn(obj: dict[str, Any]) -> None:
    for t in obj["turns"]:
        if (t["page"], int(t["repeat"])) == ("P4", 0):
            t["located"] = True


def _bump_page(obj: dict[str, Any]) -> None:
    obj["pages"]["P4"]["located"] = 1


def _reading(obj: dict[str, Any]) -> None:
    obj["reading"] = str(obj["reading"]).replace(
        f"{WANT_READING_N} of 10", f"{WANT_READING_N + 1} of 10"
    )


def _latency_max(obj: dict[str, Any]) -> None:
    obj["latency"]["max_ms"] = int(obj["latency"]["max_ms"]) + 1


def _label(obj: dict[str, Any]) -> None:
    obj["label"] = "void"


EVAL_FAULTS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "verdict-flipped": (_flip_turn, "test_verdicts_re_derived"),
    "k-bumped": (_bump_page, "test_k_of_3_re_derived"),
    "reading-off-by-one": (_reading, "test_reading_re_derived"),
    "latency-max": (_latency_max, "test_latency_re_derived"),
    "label": (_label, "test_label_re_derived"),
}


@pytest.mark.parametrize("fault", list(EVAL_FAULTS))
def test_each_seeded_fault_fails_its_eval_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """B6: a report edited after the fact fails the eval row that re-derives that number."""
    out, _report, assets, _ = _write_run(tmp_path, monkeypatch)
    edit, row = EVAL_FAULTS[fault]
    _rw_report(out, edit)
    ev = _eval()
    with pytest.raises(AssertionError):
        getattr(ev, row)(ev.load_run(out, assets.attempts_path))


def _add_llm_error(out: Path) -> None:
    """An ``llm`` error record appended to the session log of P1 r0 (a located turn the report counts)."""
    p = out / "sessions" / "turn-P1-0.jsonl"
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs.append(
        {
            "seq": len(recs),
            "t_ms": 10 * len(recs),
            "kind": "error",
            "turn_id": "t1",
            "data": {"what": "llm.generate", "request_id": "t1-gen", "error": "500"},
        }
    )
    p.write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")


def _body_560(out: Path) -> None:
    """P1 r0's recorded generate body summary rewritten to ``max_soft_tokens`` 560 in ``trials.jsonl``."""
    p = out / "trials.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    for t in rows:
        if (t["page"], int(t["repeat"])) == ("P1", 0):
            for b in t["bodies"]:
                if b["phase"] == "generate":
                    b["max_soft_tokens"] = 560
    p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


#: a run-dir file edited after the report: the eval row that must notice
FILE_FAULTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "log-llm-error": (_add_llm_error, "test_verdicts_re_derived"),
    "log-llm-error-label": (_add_llm_error, "test_label_re_derived"),
    "trial-body-560": (_body_560, "test_evidence_re_derived"),
}


@pytest.mark.parametrize("fault", list(FILE_FAULTS))
def test_eval_reads_the_logs_and_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """R-3/R-5 (F4): the eval re-derives errored turns from the session log's ``llm`` error records and re-checks the
    body summaries in ``trials.jsonl``: an error record added to a located turn's log after the report (the report
    still counts it) fails the verdict and label rows; a body rewritten to 560 fails the evidence row."""
    out, _report, assets, _ = _write_run(tmp_path, monkeypatch)
    edit, row = FILE_FAULTS[fault]
    ev = _eval()
    getattr(ev, row)(ev.load_run(out, assets.attempts_path))  # premise: green before the edit
    edit(out)
    with pytest.raises(AssertionError):
        getattr(ev, row)(ev.load_run(out, assets.attempts_path))


def _harness_run(tmp_path: Path, mp: pytest.MonkeyPatch) -> Any:
    """The harness's ``_Run`` over ``tmp_path/out`` with the synthetic assets (fields filtered to what it declares)."""
    import dataclasses

    from local_astra.config import load

    h = harness()
    assets = Assets(tmp_path / "assets")
    assets.inject(mp, h)
    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    (out / "turns.jsonl").touch()

    class _Vram:
        def min_between(self, a: float, b: float) -> None:
            return None

    known = {
        "base_url": VALID_URL,
        "out": out,
        "profile": load().profiles["lean"],
        "manifest": {},
        "vram": _Vram(),
        "attempts": {a["id"]: a for a in _attempts()["attempts"]},
    }
    cls = api("_Run")
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in known.items() if k in names})


@pytest.mark.parametrize(("pattern", "calls"), [("all-raise", 3), ("two-then-one-ok", 30)])
def test_errored_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str, calls: int
) -> None:
    """R-6 (F2): ``ERRORED_STOP = 3`` consecutive turns that did not run cleanly stop the run (both directions):
    every turn raising stops after 3 calls and the other 27 are written as not run (30 trial lines, the run
    stopped); two raising turns then a clean one, repeated, never stops (all 30 run)."""
    import asyncio

    h = harness()
    run = _harness_run(tmp_path, monkeypatch)
    calls_seen: list[tuple[str, int]] = []

    async def one_turn(run_: Any, page: str, repeat: int, *a: Any, **k: Any) -> dict[str, Any]:
        calls_seen.append((page, repeat))
        n = len(calls_seen)
        if pattern == "all-raise" or n % 3 != 0:
            raise RuntimeError(f"T-0223 rig: turn {n} raised")
        return _turn_files(run_.out, page, repeat, "located", n, 1000, None)

    monkeypatch.setattr(h, "_one_turn", one_turn)
    monkeypatch.setattr(h, "PROCESS_START_S", __import__("time").monotonic(), raising=False)
    asyncio.run(api("_run_all")(run))
    assert len(calls_seen) == calls, (pattern, len(calls_seen))
    lines = [
        x for x in (run.out / "trials.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    assert len(lines) == 30, len(lines)
    if pattern == "all-raise":
        assert isinstance(run.stopped, str) and run.stopped
    else:
        assert run.stopped is None, run.stopped


@pytest.mark.parametrize("echo", ["accepted", "refused", "rev-0-only"])
def test_wait_settings(echo: str) -> None:
    """B2: the turn waits for ``astra.settings.state`` at ``rev == 1`` with nothing refused: an accepted echo
    returns ``{rev: 1, refused: []}``; a refused one raises; an echo that never reaches rev 1 times out (raises)."""
    import asyncio

    from local_astra.contracts.protocol import AstraSettingsState, SettingRefusal

    if echo == "accepted":
        msgs = [
            AstraSettingsState.model_construct(rev=0, refused=[]),
            AstraSettingsState.model_construct(rev=1, refused=[]),
        ]
    elif echo == "refused":
        # at rev 1 (so only the refusal can stop it): rev 1 with something refused is not the echo B2 waits for
        msgs = [
            AstraSettingsState.model_construct(
                rev=1, refused=[SettingRefusal(key="user_instructions", reason="T-0223 rig")]
            )
        ]
    else:
        msgs = [AstraSettingsState.model_construct(rev=0, refused=[])]

    class _Page:
        def sent(self) -> list[Any]:
            return list(msgs)

    wait = api("wait_settings")
    if echo == "accepted":
        got = asyncio.run(wait(_Page(), timeout_s=0.5))
        assert dict(got) == {"rev": 1, "refused": []}, got
    else:
        with pytest.raises(Exception):  # noqa: B017 - a refusal or a timeout; the message is the harness's
            asyncio.run(wait(_Page(), timeout_s=0.3))


#: the dress turn's answer (two sentences) and card
DRESS_ANSWER = "Line two has the mistake: the m1word step. The fix is m1fix."
DRESS_UI = (
    CS
    + "\n"
    + json.dumps(
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": "s",
                "components": [{"id": "root", "component": "Text", "text": "Card: m1fix"}],
            },
        }
    )
)


@pytest.mark.parametrize("mode", ["clean", "engine-500"])
def test_dress_one_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """B2/B3 dress (the real ``_one_turn``, no socket): a ``FakeLlm`` whose bodies are the real
    ``VllmClient.request_body`` at 1120, ``_wait_idle``/``_healthy``/``read_metrics`` stubbed. Clean: the logged
    ``UserText`` is ``QUESTION``; the ``settings.change`` (``immediate``) precedes it; the finished turn ran with no
    B3 or body violation, located (keywords in the answer), its bodies at 1120 with the posted image. Engine 500:
    the turn is errored (R-3) and not located."""
    import asyncio

    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.contracts.llm import Commit
    from tests.fakes.llm import FakeLlm

    h = harness()
    sockets: list[str] = []

    def refuse(*a: Any, **k: Any) -> Any:
        sockets.append("connect")
        raise ConnectionRefusedError("T-0223 dress: no socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    run = _harness_run(tmp_path, monkeypatch)

    def commits(rid: str) -> list[Any]:
        return [
            Commit(
                request_id=rid, canvas_idx=0, segment="thought", text="m1word and m1fix, I think."
            ),
            Commit(request_id=rid, canvas_idx=1, segment="answer", text=DRESS_ANSWER),
            Commit(request_id=rid, canvas_idx=2, segment="ui", text=DRESS_UI),
        ]

    class Hybrid(FakeLlm):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self._real = VllmClient(VllmEndpoint(), visual_tokens=1120)
            self.visual_tokens = 1120

        def generate(
            self,
            prompt: Any,
            request_id: str,
            options: Any = None,
            xargs: Any = None,
            cd: Any = None,
            **k: Any,
        ) -> Any:
            if mode == "engine-500":
                import httpx

                req = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
                raise httpx.HTTPStatusError(
                    "500", request=req, response=httpx.Response(500, request=req)
                )
            return super().generate(prompt, request_id, options, xargs=xargs)

        def request_body(self, prompt: Any, rid: str) -> Any:
            return self._real.request_body(prompt, rid)

    async def idle(url: str) -> bool:
        return True

    monkeypatch.setattr(h.F, "_client", lambda endpoint, det, budget: Hybrid(commits("x")))
    monkeypatch.setattr(h.G, "_wait_idle", idle)
    monkeypatch.setattr(h.T, "_healthy", lambda url: True)
    monkeypatch.setattr(h.G, "read_metrics", lambda url: None)
    raw = asyncio.run(api("_one_turn")(run, "M1", 0))
    done = api("finish_turn")(run.out, "M1", 0, raw)
    assert sockets == [], sockets
    recs = [
        json.loads(x)
        for x in (run.out / "sessions" / str(raw["session_log"]))
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    users = [
        r
        for r in recs
        if r.get("kind") == "machine.step"
        and ((r.get("data") or {}).get("event") or {}).get("type") == "user.text"
    ]
    assert [u["data"]["event"]["text"] for u in users] == [QUESTION], users
    changes = [r for r in recs if r.get("kind") == "settings.change"]
    assert len(changes) == 1 and changes[0]["data"]["applied_at"] == "immediate", changes
    assert changes[0]["seq"] < users[0]["seq"], (
        "the settings set must be applied before the UserText"
    )
    if mode == "clean":
        assert done.get("status") == "ran", (done.get("status"), done.get("exception"))
        assert done.get("violations") == [] and done.get("body_violations") == [], (
            done.get("violations"),
            done.get("body_violations"),
        )
        assert done.get("located") and done.get("spoken_only_located"), done.get("spoken")
        bodies = raw.get("bodies") or []
        assert any(b.get("phase") == "generate" for b in bodies), bodies
        assert all(b.get("max_soft_tokens") == 1120 for b in bodies), bodies
    else:
        assert done.get("status") != "ran", done.get("status")
        assert not done.get("located"), done


def test_eval_judge_oracle_table() -> None:
    """The eval's own judge (the oracle the run rows use) agrees with the hand table (green by design: no
    harness)."""
    ev = _eval()
    for case, (page, spoken, shown, loc, spk, _m) in JUDGE.items():
        got = ev.judge_text([spoken] if spoken else [], [shown] if shown else [], page)
        assert got == (loc, spk), (case, got)


def test_eval_imports_no_harness() -> None:
    """The eval is the oracle (green by design): it imports nothing under ``scripts/`` or ``tests/`` and, from
    ``local_astra``, only the A2UI gate; it names no harness file outside its docstring."""
    import ast

    text = EVAL.read_text(encoding="utf-8")
    tree = ast.parse(text)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = {a.name for a in node.names}
            if mod == "local_astra.conductor.ui_gate" and names <= {"UiGate"}:
                continue
            if mod.startswith(("local_astra", "scripts", "tests")):
                bad.append((mod, sorted(names)))
        elif isinstance(node, ast.Import):
            bad += [
                (a.name, [])
                for a in node.names
                if a.name.startswith(("local_astra", "scripts", "tests"))
            ]
    assert bad == [], bad
    body = text.split('"""', 2)[2]
    assert "t0223_handwritten_check.py" not in body and "scripts/gpu" not in body
