"""T-0182 A3, A4: the doctor's ``clips`` row (docs/tasks/T-0182.md, group A).

**The seam these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``Facts.clips: dict[str, str] | None = None``: ``None`` is "not gathered"; gathered, it maps each kind in
  ``default_enabled(registered)`` to its ``clips.status`` (``current`` / ``missing`` / ``stale``). An empty dict means
  no default clip is registered.
- ``preflight.collect_clips() -> dict[str, Any]`` returns ``{"clips": {...}}`` like every ``collect_*``, reading
  ``memory.store.DEFAULT_DIR`` (looked up at call time, so a row can rebind it) and the lock's ``pocket_tts`` pin, and
  is named in ``preflight.COLLECTORS`` so ``gather`` runs it and a raise leaves ``clips`` ``None``.
- One row named ``clips``, immediately after ``browser.display``. PASS when every kind is ``current``; PASS with
  "no default clips registered" for ``{}``; WARN naming each missing or stale kind, with a fix naming
  ``services/tts.sh`` and ``uv run python scripts/render_clips.py``; WARN "not gathered" for ``None``. Never FAIL.

**Activation (reported, Q2-style as T-0111g did):** ``tests/unit/_box_rig.py`` pins an exhaustive row list (``by_name``
fails on a row not in ``REQUIRED_CHECKS``; ``test_doctor_verdicts.py::test_doctor_json_*`` asserts the name set), so the
implementation's new row reddens those rows until ``"clips"`` joins ``REQUIRED_CHECKS`` and ``FACT_FIELDS``. That edit
is made at activation; ``test_box_rig_names_the_clips_row`` holds it to account. ``_box_rig.py`` is not yet in
T-0182's owned list.

**A4's adversarial row** loads BASE's ``preflight.py`` (``git show f83f3b0a:local_astra/preflight.py``, a frozen blob)
as an oracle and asserts every row the doctor printed at BASE is unchanged on every ``tests/fixtures/box/`` recording
(parsed with BASE's own parsers) and on the all-ungathered box. It is green at BASE by design, and stays a pin after
merge because both sides of it are fixed: the blob and the fixtures.

Every other row was pending at a9dba442 and is activated for ``/implement T-0182`` group A. Regime: CPU, pure ``evaluate`` over hand-built ``Facts``; the collector
rows read and write only ``tmp_path``; no service, no socket; n = 1, no seed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra import preflight
from local_astra.config import Profile
from local_astra.memory import store
from tests import _git_range as gr
from tests.unit import _box_rig as rig
from tests.unit.test_t0182_clips_registry import (
    BASE_REGISTRY,
    ENDING_TEXTS,
    entry,
    ours,
    set_registry,
    square,
    write_asset,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[2]
#: T-0182 group A's BASE (main at f83f3b0a): the preflight.py whose rows A4 freezes. A literal.
BASE_A = "f83f3b0acf4e0594836e9627d0552979424cb26f"
FIX_TTS = "services/tts.sh"
FIX_RENDER = "uv run python scripts/render_clips.py"
KINDS = ("out_of_steps", "not_finished", "lost_thread")
STATUS_VALUES = ("current", "missing", "stale")


# --- helpers --------------------------------------------------------------------------------------


def good_facts() -> Any:
    return rig.facts(**rig.good_box())


def with_clips(facts: Any, clips: dict[str, str] | None) -> Any:
    try:
        return dataclasses.replace(facts, clips=clips)
    except TypeError as exc:
        raise AssertionError(f"T-0182 A3: preflight.Facts has no `clips` field ({exc})") from None


def clips_row(checks: tuple[Any, ...]) -> Any:
    rows = [c for c in checks if c.name == "clips"]
    assert len(rows) == 1, f"T-0182 A3: exactly one `clips` row; got {[c.name for c in checks]}"
    return rows[0]


def row_for(clips: dict[str, str] | None) -> Any:
    return clips_row(preflight.evaluate(with_clips(good_facts(), clips)))


def collect() -> dict[str, Any]:
    fn = getattr(preflight, "collect_clips", None)
    assert callable(fn), "T-0182 A3: preflight.collect_clips() is missing"
    out = fn()
    assert isinstance(out, dict) and "clips" in out, (
        f"A3: collect_clips() returns {{'clips': ...}}; got {out!r}"
    )
    return out


@pytest.fixture
def default_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``DEFAULT_DIR`` rebound to ``tmp_path`` wherever a loaded module holds the real one."""
    real = store.DEFAULT_DIR
    target = tmp_path / "default-data-dir"
    for module in ours():
        if module.__dict__.get("DEFAULT_DIR") == real:
            monkeypatch.setattr(module, "DEFAULT_DIR", target)
    return target


# --- the verdicts ---------------------------------------------------------------------------------


def test_facts_clips_defaults_to_not_gathered() -> None:
    """A3: ``Facts().clips`` is ``None`` (every field defaults to "not gathered")."""
    assert "clips" in preflight.Facts.__dataclass_fields__, "T-0182 A3: Facts has no `clips` field"
    assert preflight.Facts().clips is None


def test_every_default_kind_current_is_pass() -> None:
    """A3: PASS when every default-enabled kind is current."""
    row = row_for(dict.fromkeys(KINDS, "current"))
    assert row.status == "PASS", f"A3: all current is PASS; got {row}"


@pytest.mark.parametrize(
    "combo",
    [c for c in itertools.product(STATUS_VALUES, repeat=3) if set(c) != {"current"}],
    ids="-".join,
)
def test_a_missing_or_stale_kind_is_warn_naming_it_with_the_fix(combo: tuple[str, ...]) -> None:
    """A3: any missing or stale kind makes the row WARN (never FAIL: a missing clip only means silence); each such
    kind is named, and the fix says to start ``services/tts.sh`` and run the render script. All 26 combinations."""
    clips = dict(zip(KINDS, combo, strict=True))
    row = row_for(clips)
    assert row.status == "WARN", f"A3: {clips} is WARN; got {row}"
    for kind, value in clips.items():
        if value != "current":
            assert kind in row.detail, f"A3: the WARN names {kind} ({value}); detail {row.detail!r}"
    assert FIX_TTS in row.fix and FIX_RENDER in row.fix, (
        f"A3: the fix names both steps; fix {row.fix!r}"
    )


def test_not_gathered_is_warn() -> None:
    """A3: ``clips=None`` (the collector raised) is WARN "not gathered", never PASS and never FAIL."""
    row = row_for(None)
    assert row.status == "WARN" and "not gathered" in row.detail, f"A3: got {row}"


def test_no_default_clips_registered_is_pass() -> None:
    """A3: an empty default set (BASE's registry: ack, hmm and still_working are not default-enabled) is PASS
    "no default clips registered", not WARN (there is nothing to render) and not "not gathered"."""
    row = row_for({})
    assert row.status == "PASS", f"A3: got {row}"
    assert "no default clips registered" in row.detail, f"A3: detail {row.detail!r}"


@pytest.mark.parametrize(
    "clips", [None, {}, {"lost_thread": "missing"}, dict.fromkeys(KINDS, "current")]
)
def test_the_row_comes_right_after_the_browser_rows(clips: dict[str, str] | None) -> None:
    """A3: one ``clips`` row, immediately after ``browser.display``, on a good box and on the all-ungathered box."""
    for facts in (good_facts(), preflight.Facts()):
        names = [c.name for c in preflight.evaluate(with_clips(facts, clips))]
        assert names.count("clips") == 1, f"A3: {names}"
        assert names.index("clips") == names.index("browser.display") + 1, f"A3: order {names}"


@pytest.mark.parametrize(
    "clips", [None, {}, {"lost_thread": "stale"}, dict.fromkeys(KINDS, "current")]
)
def test_the_clips_fact_moves_no_other_row(clips: dict[str, str] | None) -> None:
    """A3/A4: the clips fact feeds the clips row only; every other row is the same whatever it holds."""

    def others(facts: Any) -> list[tuple[str, str, str, str]]:
        return [
            (c.name, c.status, c.detail, c.fix)
            for c in preflight.evaluate(facts)
            if c.name != "clips"
        ]

    for facts in (good_facts(), preflight.Facts()):
        assert others(with_clips(facts, clips)) == others(with_clips(facts, None))


# --- the collector --------------------------------------------------------------------------------


def _stub_collectors(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "collect_clips" in preflight.COLLECTORS, (
        "T-0182 A3: gather runs collect_clips (COLLECTORS)"
    )
    for name in preflight.COLLECTORS:
        monkeypatch.setattr(preflight, name, lambda *a, **k: {})


def test_a_raising_collector_leaves_the_row_not_gathered(monkeypatch: pytest.MonkeyPatch) -> None:
    """A3: ``gather`` runs ``collect_clips``; when it raises, ``Facts.clips`` stays ``None`` and the row is WARN
    "not gathered". When it answers, ``gather`` carries its answer through. Re-scoped by B-R2: both stand-ins accept
    the ``filler_clips`` keyword ``gather`` now passes (default False), so the raise is the PermissionError."""
    _stub_collectors(monkeypatch)

    def boom(*, filler_clips: bool = False) -> dict[str, Any]:
        raise PermissionError("clips dir unreadable")

    monkeypatch.setattr(preflight, "collect_clips", boom)
    facts = preflight.gather(Profile(name="lean"))
    assert facts.clips is None
    row = clips_row(preflight.evaluate(facts))
    assert row.status == "WARN" and "not gathered" in row.detail, f"A3: {row}"
    monkeypatch.setattr(
        preflight,
        "collect_clips",
        lambda *, filler_clips=False: {"clips": {"lost_thread": "missing"}},
    )
    assert preflight.gather(Profile(name="lean")).clips == {"lost_thread": "missing"}


def test_collect_clips_at_base_registry_is_still_working_only(
    default_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3 adversarial, re-scoped by B6 (``still_working`` is default-enabled): with BASE's registry, ``collect_clips``
    reports ``still_working`` alone (missing), never ``ack`` even with an ``ack`` asset sitting in the data dir (ack is
    not default-enabled). With ``ack`` and ``hmm`` alone registered, it reports no kind and the row is PASS "no
    default clips registered"."""
    set_registry(monkeypatch, dict(BASE_REGISTRY))
    write_asset(default_dir, "ack", square())
    write_manifest(default_dir, {"ack": entry("let me look", square())})
    got = collect()["clips"]
    assert got == {"still_working": "missing"}, (
        f"A3: only default-enabled kinds are reported; got {got}"
    )
    set_registry(monkeypatch, {k: BASE_REGISTRY[k] for k in ("ack", "hmm")})
    got = collect()["clips"]
    assert got == {}, f"A3: only default-enabled kinds are reported; got {got}"
    row = row_for(got)
    assert row.status == "PASS" and "no default clips registered" in row.detail


def test_collect_clips_forwards_filler_clips(
    default_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6 / the doctor channel (``collect_clips`` takes ``filler_clips``, default False): with the full registry (the
    three BASE kinds and the three ending kinds) and an empty data dir, ``collect_clips(filler_clips=True)`` reports
    ``ack`` and ``hmm`` beside the default kinds, and ``collect_clips()`` reports neither. Expected sets by hand."""
    set_registry(monkeypatch, {**BASE_REGISTRY, **ENDING_TEXTS})
    fn = getattr(preflight, "collect_clips", None)
    assert callable(fn), "T-0182 A3: preflight.collect_clips() is missing"
    default_kinds = {"still_working", "out_of_steps", "not_finished", "lost_thread"}
    on = fn(filler_clips=True)["clips"]
    assert on == {k: "missing" for k in default_kinds | {"ack", "hmm"}}, (
        f"T-0182 B6: collect_clips(filler_clips=True) reports ack and hmm too; got {on}"
    )
    off = collect()["clips"]
    assert off == {k: "missing" for k in default_kinds}, (
        f"T-0182 B6: collect_clips() reports no filler kind; got {off}"
    )


def test_collect_clips_reads_the_default_data_dir(
    default_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3: with two ending kinds registered, ``collect_clips`` reports each one's status in ``DEFAULT_DIR``: the
    rendered one current, the other missing; then a text change makes the first stale. The row follows. Re-scoped by
    B6: BASE's ``still_working`` is default-enabled too, missing until the last stage renders it."""
    from local_astra.conductor import session

    set_registry(
        monkeypatch,
        {
            **BASE_REGISTRY,
            "out_of_steps": ENDING_TEXTS["out_of_steps"],
            "lost_thread": ENDING_TEXTS["lost_thread"],
        },
    )
    pcm = square()
    write_asset(default_dir, "out_of_steps", pcm)
    write_manifest(default_dir, {"out_of_steps": entry(ENDING_TEXTS["out_of_steps"], pcm)})
    got = collect()["clips"]
    assert got == {
        "out_of_steps": "current",
        "lost_thread": "missing",
        "still_working": "missing",
    }, f"A3: got {got}"
    row = row_for(got)
    assert row.status == "WARN" and "lost_thread" in row.detail
    monkeypatch.setitem(session.CLIP_TEXT, "out_of_steps", ENDING_TEXTS["out_of_steps"] + " ")
    assert collect()["clips"] == {
        "out_of_steps": "stale",
        "lost_thread": "missing",
        "still_working": "missing",
    }
    write_asset(default_dir, "lost_thread", pcm)
    write_asset(default_dir, "still_working", pcm)
    monkeypatch.setitem(session.CLIP_TEXT, "out_of_steps", ENDING_TEXTS["out_of_steps"])
    write_manifest(
        default_dir,
        {
            "out_of_steps": entry(ENDING_TEXTS["out_of_steps"], pcm),
            "lost_thread": entry(ENDING_TEXTS["lost_thread"], pcm),
            "still_working": entry(BASE_REGISTRY["still_working"], pcm),
        },
    )
    got = collect()["clips"]
    assert got == {"out_of_steps": "current", "lost_thread": "current", "still_working": "current"}
    assert row_for(got).status == "PASS"


@pytest.mark.parametrize(
    "content",
    ['{"clips": {', '{"clips": {"out_of_steps": {"text": 1}}}'],
    ids=["bad JSON", "schema-invalid"],
)
def test_an_unreadable_manifest_names_the_stale_kinds(
    default_dir: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """A3: an unreadable manifest beside the assets makes the kinds stale; the collector does not raise, so the row is
    WARN naming both kinds, not "not gathered". Re-scoped by B6: BASE's default-enabled ``still_working`` has an asset
    beside them too, so it is stale as well."""
    set_registry(
        monkeypatch,
        {
            **BASE_REGISTRY,
            "out_of_steps": ENDING_TEXTS["out_of_steps"],
            "lost_thread": ENDING_TEXTS["lost_thread"],
        },
    )
    for kind in ("out_of_steps", "lost_thread", "still_working"):
        write_asset(default_dir, kind, square())
    write_manifest(default_dir, {}).write_text(content)
    got = collect()["clips"]
    assert got == {"out_of_steps": "stale", "lost_thread": "stale", "still_working": "stale"}, (
        f"A3: got {got}"
    )
    row = row_for(got)
    assert row.status == "WARN" and "not gathered" not in row.detail, f"A3: {row}"
    assert "out_of_steps" in row.detail and "lost_thread" in row.detail, f"A3: {row.detail!r}"


def test_box_rig_names_the_clips_row() -> None:
    """Activation: ``_box_rig.REQUIRED_CHECKS`` is exhaustive, so ``clips`` joins it (and ``FACT_FIELDS``) when the row
    lands; until then this row is red."""
    assert "clips" in rig.REQUIRED_CHECKS, "add `clips` to tests/unit/_box_rig.REQUIRED_CHECKS"
    assert "clips" in rig.FACT_FIELDS, "add `clips` to tests/unit/_box_rig.FACT_FIELDS"


# --- A4: every existing row unchanged on every box fixture ----------------------------------------


@pytest.fixture(scope="module")
def base_preflight(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    gr.require_commits(BASE_A)
    blob = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_A}:local_astra/preflight.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = tmp_path_factory.mktemp("t0182_base") / "_t0182_base_preflight.py"
    path.write_text(blob)
    spec = importlib.util.spec_from_file_location("_t0182_base_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _box_variants(base: ModuleType) -> dict[str, dict[str, Any]]:
    """Each recording in tests/fixtures/box, parsed by BASE's own parsers, laid over a good box."""
    text = rig.fixture
    parsed = {
        "wpctl_status.txt": base.parse_wpctl_status(text("wpctl_status.txt")),
        "nvidia_smi.csv": base.parse_nvidia_smi_csv(text("nvidia_smi.csv")),
        "nvidia_smi_memory_coldboot.csv": base.parse_nvidia_smi_memory(
            text("nvidia_smi_memory_coldboot.csv")
        ),
        "nvidia_smi_compute_apps_coldboot.csv": {
            "vram_processes": base.parse_nvidia_smi_compute_apps(
                text("nvidia_smi_compute_apps_coldboot.csv")
            )
        },
        "nvidia_smi_compute_apps_engine_up.csv": {
            "vram_processes": base.parse_nvidia_smi_compute_apps(
                text("nvidia_smi_compute_apps_engine_up.csv")
            )
        },
        "ss_tlnp.txt": {"listeners": base.parse_ss_tlnp(text("ss_tlnp.txt"))},
        "docker_inspect_stt.json": {
            "stt_argv": base.parse_docker_inspect(text("docker_inspect_stt.json"))["argv"]
        },
        "docker_inspect_vllm.json": {
            "vllm_argv": base.parse_docker_inspect(text("docker_inspect_vllm.json"))["argv"]
        },
    }
    recorded = sorted(p.name for p in rig.BOX.iterdir() if p.is_file() and p.name != "MANIFEST.md")
    assert sorted(parsed) == recorded, (
        f"A4: every box fixture is a variant; unmapped {set(recorded) - set(parsed)}"
    )
    fields = set(base.Facts.__dataclass_fields__)
    variants = {
        name: {**rig.good_box(), **{k: v for k, v in facts.items() if k in fields}}
        for name, facts in parsed.items()
    }
    variants["all recordings together"] = {
        **rig.good_box(),
        **{k: v for facts in parsed.values() for k, v in facts.items() if k in fields},
    }
    variants["good box"] = rig.good_box()
    variants["nothing gathered"] = {}
    return variants


def _rows(checks: tuple[Any, ...]) -> list[tuple[str, str, str, str]]:
    return [(c.name, c.status, c.detail, c.fix) for c in checks]


#: T-0222 re-pin: R-1. T-0222 A changed exactly these fix texts (docs/tasks/T-0222.md, ruling R-1): ``uv sync
#: --extra X`` uninstalls the other extras, so BROWSER_FIX says ``uv sync --all-extras``; and the model.revision
#: rows name ``hf download`` instead of ``huggingface-cli`` / the unimplemented ``astra fetch-models``. Literals,
#: not read from today's preflight.py, so the exemption cannot widen with the implementation.
T0222_BASE_BROWSER_FIX = "uv sync --extra browser && uv run playwright install chromium"
T0222_BROWSER_FIX = "uv sync --all-extras && uv run playwright install chromium"
T0222_BASE_REVISION_FIXES = (
    "huggingface-cli download the pinned revision, or fill model_revision in the lock",
    "fetch the pinned revision (`uv run astra fetch-models`) before any measurement",
)


def _t0222_exempt(
    want: list[tuple[str, str, str, str]], got: list[tuple[str, str, str, str]], where: str
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    """T-0222 re-pin: R-1. Drop only the ``fix`` field that T-0222 changed, after checking the change is the one
    T-0222 made; name, status, detail and order (and every other row's fix) still compare byte-equal."""
    if len(want) != len(got):
        return want, got  # the plain comparison reports it
    w_out, g_out = [], []
    for w, g in zip(want, got, strict=True):
        if w[0] == g[0] == "model.revision" and w[3] in T0222_BASE_REVISION_FIXES:
            assert "hf download" in g[3] and "fetch-models" not in g[3], (
                f"T-0222 re-pin: R-1: model.revision's fix names `hf download` on {where}; got {g[3]!r}"
            )
            w, g = (*w[:3], "<T-0222 fix>"), (*g[:3], "<T-0222 fix>")
        elif w[3].startswith(T0222_BASE_BROWSER_FIX):
            tail = w[3][len(T0222_BASE_BROWSER_FIX) :]
            assert g[3] == T0222_BROWSER_FIX + tail, (
                f"T-0222 re-pin: R-1: {w[0]}'s fix is BROWSER_FIX with --all-extras on {where}; got {g[3]!r}"
            )
            w, g = (*w[:3], "<T-0222 fix>"), (*g[:3], "<T-0222 fix>")
        w_out.append(w)
        g_out.append(g)
    return w_out, g_out


def test_existing_rows_are_unchanged_on_every_box_fixture(base_preflight: ModuleType) -> None:
    """A4 adversarial: every row BASE's doctor printed is printed the same today (name, status, detail, fix, order),
    on each box recording and on the all-ungathered box; the only row allowed to be new is ``clips``. Green at BASE
    by design (the oracle is BASE's frozen preflight.py). T-0222 re-pin: R-1: the one stated exemption is
    ``_t0222_exempt`` (the fix of the model.revision rows and of the BROWSER_FIX rows, each checked to be T-0222's
    new text)."""
    for name, full in _box_variants(base_preflight).items():
        # ``good_box`` carries the clips fact since activation; BASE's Facts has no such field, and the clips row is
        # outside this comparison, so both sides get the box without it.
        box = {k: v for k, v in full.items() if k != "clips"}
        want = _rows(base_preflight.evaluate(base_preflight.Facts(**box)))
        got = [r for r in _rows(preflight.evaluate(preflight.Facts(**box))) if r[0] != "clips"]
        want, got = _t0222_exempt(want, got, name)
        assert got == want, f"A4: the doctor's existing rows changed on {name}"
        assert len(want) > 20, f"premise: BASE printed {len(want)} rows on {name}"
