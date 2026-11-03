"""T-0069 criteria 1-2 (tier 1, CPU, ``make cd-test``): the served model state is Tap ∘ CD ∘ stock.

Every row that runs a sampler goes through the **served** composition: ``model.py``'s
``AstraDiffusionGemma.get_model_state_cls()`` imported over a fake ``vllm.model_executor.models.diffusion_gemma`` whose
stock model state is the CD stand-in's ``FakeModelStateBase`` (it constructs the vendored v0.29.0
``DiffusionSampler``; ``tests/cd/_tap_cd_world.py``), with the real tap (``CanvasTap`` + ``TorchTapBackend``,
recording publisher, virtual clock) and the real CD wrapper. Reference runs compose T-0011's factory by hand
(``make_model_state_cls(FakeModelStateBase, tap=...)``: Tap ∘ stock).

Rows:
- the composition expression in ``model.py`` (AST, exact), lazy imports, no try/except fallback, and a failing ``cd``
  import failing the build (no silent Tap ∘ stock);
- the object chain ``TapSampler → ConstrainedDiffusionSampler → stock DiffusionSampler`` by identity, and the MRO;
- mixed batch {constrained, unconstrained}: every frame validates (R10/R14/R15) and equals its call's rows, and each
  commit frame of the constrained request carries the CD record's ``committed_ids`` (the MAP, which differs from the
  stock argmax on at least one canvas here, so the tap provably reads what CD wrote);
- a forced-EOS (loud) request's commit frame starts with id 1;
- *negative:* no ``astra_cd_*`` key ⇒ Tap ∘ CD ∘ stock is bit-identical to Tap ∘ stock under one seed (state rows,
  ``SamplerOutput``, global RNG, frames), and the CD wrapper issues zero torch ops (T-0067 c3 through the tap);
- *negative (git, vs e760cb7):* only ``model.py`` changed under ``plugin/``, and inside it only ``get_model_state_cls``
  and the module docstring; ``AstraDiffusionGemma`` gains no member; T-0011's criterion 1/2/15 modules are unedited;
- *adversarial (R6 under composition):* a loud constrained request commits its EOS canvas and is released, and an
  unconstrained request takes its slot before the tap is drained: no frame pairs one occupant's id with the other's
  ``committed_tokens``, and no CD record leaks to the new occupant.

Regime: CPU torch (``cd`` group), vendored v0.29.0 stock step, V = 262 144, CL 4, 8 slots, ST 2, 48 max steps, always
confident (convergence by stability), seed 0 unless stated, tap ``hz`` 1e6 on a 1 s virtual clock step (every call
publishes) with the tap drained after every call unless stated; grammar G1 of ``tests/cd/unit/_sampler_support.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests import _git_range as gr
from tests.cd import _tap_cd_world as tw
from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, FakeModelStateBase
from tests.cd.fakes.diffusion_sampler import stock as stock_mod
from tests.cd.unit._sampler_support import G1, Grammar, Records, cd_xargs
from tests.cd.unit._scenario import canvas_logits
from tests.fakes.diffusion_sampler import validate_canvas_step
from tests.stock.world import frame_mismatches, snaps_differ

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
BASE = "e760cb7"  # T-0069's branch point (T-0011 merged, main merged into T-0062-perstep-cd)
#: Ruling F1 (docs/tasks/T-0076.md): the criterion-2 git fences assert T-0069's *own scope*, not a permanent freeze of
#: cd/**. They compare T-0069's commit range BASE..T0069_HEAD (branch T-0069-cd-wiring's head; its merge into
#: T-0062-perstep-cd is T0069_MERGE), so later tasks that own cd/ files (T-0076 steps.py/xargs.py, T-0077 the core
#: modules) do not redden them, and "T-0069 changed only its owned files under plugin/" stays checkable forever.
T0069_HEAD = "154614d"
T0069_MERGE = "f3e3c43"
PLUGIN = "plugin/astra_vllm_plugin/astra_vllm_plugin"
MODEL_PY = f"{PLUGIN}/model.py"
#: files under plugin/ T-0069 may change: model.py (criterion 1), tap.py (ruling O6: the commit step range only),
#: cd/budget.py and, only if a constant must be built at bind, cd/sampler.py / cd/requests.py (ruling O5)
PLUGIN_MAY_CHANGE = {
    MODEL_PY,
    f"{PLUGIN}/tap.py",
    f"{PLUGIN}/cd/budget.py",
    f"{PLUGIN}/cd/sampler.py",
    f"{PLUGIN}/cd/requests.py",
}
NOT_OWNED = [
    f"{PLUGIN}/model_state.py",
    f"{PLUGIN}/sampler.py",
    f"{PLUGIN}/__init__.py",
]


def _cd_files_at_base() -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", BASE, "--", f"{PLUGIN}/cd"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(proc.stdout.split()) or [f"<git ls-tree {BASE} failed: {proc.stderr.strip()}>"]


#: every cd/ file at e760cb7 except the three ruling O5 lets change: each must stay byte-identical
CD_FENCED = [f for f in _cd_files_at_base() if f not in PLUGIN_MAY_CHANGE]
#: T-0011 criteria 1, 2 (fast suite) and 15 (tier 1b) must pass *unedited*
T0011_TEST_FILES = [
    "tests/unit/test_plugin_sampler.py",
    "tests/fakes/diffusion_sampler.py",
    "tests/stock",
]
#: T-0015's three members plus T-0011's seam (R9)
ASTRA_MEMBERS = {
    "_enable_mm_lora",
    "_parse_and_validate_audio_input",
    "_process_audio_input",
    "get_model_state_cls",
}
EXPECTED_EXPR = (
    "make_model_state_cls(make_cd_model_state_cls("
    "DiffusionGemmaForConditionalGeneration.get_model_state_cls()), tap=tap_from_env)"
)
CFG = Cfg()  # CL 4, 8 slots, ST 2, 48 steps, always confident
MAX_STEPS = CFG.max_denoising_steps
EOS = 1


# --------------------------------------------------------------------------- git / AST helpers


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )


def _source_at(rev: str, path: str) -> str:
    proc = _git("show", f"{rev}:{path}")
    assert proc.returncode == 0, f"git show {rev}:{path}: {proc.stderr}"
    return proc.stdout


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"model.py has no class {name}")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls.name} has no method {name}")


def _class_members(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not (n.startswith("__") and n.endswith("__"))}


def _imports_in(fn: ast.FunctionDef) -> dict[str, str]:
    """name bound -> the absolute module it was imported from (``astra_vllm_plugin`` relatives resolved)."""
    out: dict[str, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # model.py is astra_vllm_plugin.model: level 1 is the package
                mod = "astra_vllm_plugin" + ("." + mod if mod else "")
            for alias in node.names:
                out[alias.asname or alias.name] = mod
    return out


def _decorator_names(fn: ast.FunctionDef) -> list[str]:
    return [ast.unparse(d) for d in fn.decorator_list]


# --------------------------------------------------------------------------- criterion 1: the expression (AST)


def test_get_model_state_cls_returns_exactly_the_composition_expression() -> None:
    """R9 / criterion 1: one cached staticmethod with lazy imports and no fallback, returning exactly
    ``make_model_state_cls(make_cd_model_state_cls(DiffusionGemmaForConditionalGeneration.get_model_state_cls()),
    tap=tap_from_env)``."""
    tree = ast.parse((REPO / MODEL_PY).read_text())
    fn = _method(_class(tree, "AstraDiffusionGemma"), "get_model_state_cls")
    decos = _decorator_names(fn)
    assert decos[:1] == ["staticmethod"] and any(
        d in ("functools.cache", "cache") for d in decos
    ), f"get_model_state_cls must stay a cached staticmethod, decorators: {decos}"
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)], (
        "no try/except in get_model_state_cls: a failing cd import must fail the boot, never fall back to "
        "Tap ∘ stock (which would serve astra_cd_* requests unconstrained, silently)"
    )
    imports = _imports_in(fn)
    want = {
        "make_model_state_cls": "astra_vllm_plugin.model_state",
        "make_cd_model_state_cls": "astra_vllm_plugin.cd.model_state",
        "tap_from_env": "astra_vllm_plugin.tap",
    }
    assert {k: imports.get(k) for k in want} == want, f"lazy imports in the method: {imports}"
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and returns[0].value is not None, ast.unparse(fn)
    got = ast.dump(returns[0].value)
    expected = ast.dump(ast.parse(EXPECTED_EXPR, mode="eval").body)
    assert got == expected, (
        f"the composition expression is `{ast.unparse(returns[0].value)}`, criterion 1 wants `{EXPECTED_EXPR}`"
    )


def test_no_plugin_module_is_imported_at_model_py_module_level() -> None:
    """The plugin's tap and CD modules stay lazy (torch-free import of ``model.py``'s module scope): the module-level
    imports are exactly those at e760cb7."""

    def top_imports(src: str) -> list[str]:
        tree = ast.parse(src)
        return sorted(
            ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        )

    assert top_imports((REPO / MODEL_PY).read_text()) == top_imports(_source_at(BASE, MODEL_PY))


def test_model_py_changes_only_inside_get_model_state_cls_and_the_module_docstring() -> None:
    """Owned files: the body of ``get_model_state_cls``, its comment and the docstring's Scope paragraph only.
    Comments are not in the AST; with the method's body and the module docstring blanked, HEAD's AST equals
    e760cb7's."""

    def blanked(src: str) -> str:
        tree = ast.parse(src)
        if tree.body and isinstance(tree.body[0], ast.Expr):
            tree.body[0] = ast.Expr(value=ast.Constant(value="<docstring>"))
        fn = _method(_class(tree, "AstraDiffusionGemma"), "get_model_state_cls")
        fn.body = [ast.Pass()]
        return ast.dump(tree)

    assert blanked((REPO / MODEL_PY).read_text()) == blanked(_source_at(BASE, MODEL_PY))


# --------------------------------------------------------------------------- criterion 2 (git fences)


def _range_changes(a: str, b: str, path: str) -> list[str]:
    """Files under ``path`` that differ between commits ``a`` and ``b`` (ruling F1: a commit range, never HEAD)."""
    proc = _git("diff", "--name-only", a, b, "--", path)
    assert proc.returncode == 0, f"git diff {a} {b}: {proc.stderr}"
    return sorted(proc.stdout.split())


def test_f1_t0069_range_is_real_history_and_the_range_check_has_power() -> None:
    """Ruling F1's preconditions: BASE is an ancestor of T0069_HEAD, which is a parent of the merge T0069_MERGE, and
    the merge is in this branch's history; and the range check is not vacuous -- over a range that does touch
    ``cd/steps.py`` (BASE..HEAD, after T-0076) it reports it, and over T-0069's range it reports ``model.py``."""
    for anc, desc in ((BASE, T0069_HEAD), (T0069_MERGE, "HEAD")):
        proc = _git("merge-base", "--is-ancestor", anc, desc)
        assert proc.returncode == 0, f"{anc} is not an ancestor of {desc}: {proc.stderr}"
    parents = _git("rev-list", "--parents", "-n", "1", T0069_MERGE).stdout.split()[1:]
    head_full = _git("rev-parse", T0069_HEAD).stdout.strip()
    assert head_full in parents, (
        f"{T0069_HEAD} is not a parent of the merge {T0069_MERGE}: {parents}"
    )
    assert MODEL_PY in _range_changes(BASE, T0069_HEAD, "plugin/"), "T-0069's range lost model.py"
    assert f"{PLUGIN}/cd/steps.py" in _range_changes(BASE, "HEAD", f"{PLUGIN}/cd"), (
        "control: BASE..HEAD should touch cd/steps.py (T-0076); the range check would be blind"
    )


def test_only_the_owned_files_changed_under_plugin_and_the_not_owned_files_are_byte_identical() -> (
    None
):
    """Criterion 2 as amended by rulings O5/O6, over T-0069's range (ruling F1): under ``plugin/`` only
    ``model.py``, ``tap.py`` and the three named ``cd/`` files differ between BASE and T0069_HEAD (a file added in
    the range fails too), and the not-owned files are byte-identical across it."""
    others = sorted(set(_range_changes(BASE, T0069_HEAD, "plugin/")) - PLUGIN_MAY_CHANGE)
    assert others == [], (
        f"files under plugin/ changed in T-0069's range {BASE}..{T0069_HEAD} beyond rulings O5/O6: {others}"
    )
    for path in NOT_OWNED:
        assert _range_changes(BASE, T0069_HEAD, path) == [], (
            f"{path} changed in T-0069's range {BASE}..{T0069_HEAD}"
        )


@pytest.mark.parametrize("path", CD_FENCED)
def test_every_other_cd_file_is_byte_identical(path: str) -> None:
    """Ruling O5: of ``cd/**`` only ``budget.py`` (and, if needed, ``sampler.py`` / ``requests.py``) may change --
    within T-0069's range BASE..T0069_HEAD (ruling F1)."""
    assert len(CD_FENCED) >= 10, f"fixture: cd/ files at {BASE}: {CD_FENCED}"
    assert _range_changes(BASE, T0069_HEAD, path) == [], (
        f"{path} changed in T-0069's range {BASE}..{T0069_HEAD}"
    )


def test_astra_diffusion_gemma_gains_no_member() -> None:
    head = _class_members(_class(ast.parse((REPO / MODEL_PY).read_text()), "AstraDiffusionGemma"))
    base = _class_members(_class(ast.parse(_source_at(BASE, MODEL_PY)), "AstraDiffusionGemma"))
    assert base == ASTRA_MEMBERS, f"fixture: e760cb7's members are {sorted(base)}"
    assert head == ASTRA_MEMBERS, f"AstraDiffusionGemma's members: {sorted(head)}"


#: Power pair of the row below (T-0094b ruling P2): T-0011's 79b7ef8 (parent f88a978) edited a module under
#: ``tests/stock``.
T0011_FILES_PRIOR_PARENT = "f88a978"
T0011_FILES_PRIOR = "79b7ef8"


def test_t0011_criteria_1_2_15_modules_are_unedited() -> None:
    """They must pass *unedited* (criterion 2); passing is their own run, unedited is this row: nothing under
    ``T0011_TEST_FILES`` differs over T-0069's closed range BASE..T0069_HEAD (T-0094b; BASE = merge-base of
    T0069_MERGE's parents, T0069_HEAD its second parent). The ``status --porcelain`` half is retired: it only meant
    something while the branch was open."""
    gr.prove_range(BASE, T0069_HEAD, T0069_MERGE)
    for path in T0011_TEST_FILES:
        changed = gr.changed(BASE, T0069_HEAD, path)
        assert changed == [], f"{path} changed in {BASE}..{T0069_HEAD}: {changed}"


def test_the_t0011_modules_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over 79b7ef8's parent and 79b7ef8 the same predicate reports a file under the
    ``tests/stock`` entry of ``T0011_TEST_FILES``; a predicate that reports nothing fails here."""
    gr.require_commits(T0011_FILES_PRIOR_PARENT, T0011_FILES_PRIOR)
    assert gr.is_ancestor(T0011_FILES_PRIOR, BASE)
    stock = next(p for p in T0011_TEST_FILES if p.endswith("/stock"))
    changed = gr.changed(T0011_FILES_PRIOR_PARENT, T0011_FILES_PRIOR, *T0011_TEST_FILES)
    assert f"{stock}/test_tap_stock.py" in changed, changed


# --------------------------------------------------------------------------- the served composition


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, tw.TapRouter]:
    return tw.served_model(monkeypatch)


def _served_world(served: tuple[Any, tw.TapRouter], **kw: Any) -> tw.World:
    astra, router = served
    factory = tw.TapFactory()
    return tw.World(CFG, astra.get_model_state_cls(), factory=factory, router=router, **kw)


def _cd_in_chain(w: tw.World) -> None:
    """Precondition of every comparison below: without CD in the chain it would be vacuous."""
    names = tw.chain_names(w.sampler)
    assert names == ["TapSampler", "ConstrainedDiffusionSampler", "DiffusionSampler"], (
        f"the served sampler chain is {names}: T-0069 not implemented (model.py still composes Tap ∘ stock)"
    )


def test_the_served_model_state_is_tap_over_cd_over_stock(
    served: tuple[Any, tw.TapRouter],
) -> None:
    """Criterion 1 (and criterion 3's MRO, on CPU): ``AstraDiffusionGemmaModelState → Cd<stock> → stock``, cached;
    ``custom_sampler`` yields ``TapSampler → ConstrainedDiffusionSampler → stock DiffusionSampler``, each inner
    the identical object, and the second element is stock's own."""
    astra, _router = served
    cls = astra.get_model_state_cls()
    assert astra.get_model_state_cls() is cls, "get_model_state_cls must be cached"
    assert cls.__name__ == "AstraDiffusionGemmaModelState", cls.__name__
    (cd_base,) = cls.__bases__
    assert cd_base.__name__ == f"Cd{FakeModelStateBase.__name__}", (
        f"the tap's base is {cd_base.__name__}, not the CD model state over the stock one"
    )
    assert cd_base.__bases__ == (FakeModelStateBase,), cd_base.__bases__
    w = _served_world(served)
    _cd_in_chain(w)
    outer, cd, inner = tw.chain(w.sampler)
    tap_sampler_cls = tw.plugin_attr("sampler", "TapSampler", "T-0011")
    cd_sampler_cls = tw.plugin_attr("cd.sampler", "ConstrainedDiffusionSampler", "T-0067")
    assert type(outer) is tap_sampler_cls and type(cd) is cd_sampler_cls
    assert type(inner) is stock_mod.DiffusionSampler
    assert inner is w.ms.constructed[-1], "the innermost sampler is the one stock constructed"
    assert len(w.ms.constructed) == 1, "the plugin constructed no sampler of its own"
    assert w.second is None, "the second element is stock's own (None for the stand-in)"


def test_a_failing_cd_import_fails_the_build_never_falls_back(
    served: tuple[Any, tw.TapRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Implementation notes, "No fallback": if ``astra_vllm_plugin.cd.model_state`` cannot be imported,
    ``get_model_state_cls()`` raises. A Tap ∘ stock fallback would ignore ``astra_cd_*`` keys and serve those
    requests unconstrained. (The other direction -- cd importable, the build succeeds -- is the row above.)"""
    astra, _router = served
    monkeypatch.setitem(sys.modules, "astra_vllm_plugin.cd.model_state", None)
    try:
        cls = astra.get_model_state_cls()
    except ImportError:
        return
    pytest.fail(
        f"get_model_state_cls() returned {cls!r} with the cd package unimportable: a silent fallback "
        "(T-0069 not implemented, or a try/except around the cd import)"
    )


# --------------------------------------------------------------------------- criterion 1: mixed batch


def _frames_match_rows(w: tw.World, req_ids: tuple[str, ...]) -> list[str]:
    """Every frame validates (R10/R14/R15) and equals the rows of the call that staged it."""
    bad: list[str] = []
    closed: dict[str, int] = defaultdict(int)
    for s in w.snaps:
        for f in s.frames:  # type: ignore[attr-defined]
            try:
                validate_canvas_step(f)
            except AssertionError as exc:
                bad.append(f"call {s.call}: {exc}")
        if s.kind != "decode":
            continue
        by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in s.frames:  # type: ignore[attr-defined]
            by_req[f["request_id"]].append(f)
        for r in s.order:
            if r not in req_ids:
                continue
            frames = by_req.get(r, [])
            if len(frames) != 1:
                bad.append(f"call {s.call} {r}: {len(frames)} frames (want 1)")
                continue
            bad += [
                f"call {s.call} {r}: {m}"
                for m in frame_mismatches(
                    frames[0], s, r, canvas_idx=closed[r], max_steps=MAX_STEPS
                )
            ]
            if s.committed(r) is not None:
                closed[r] += 1
    return bad


def test_mixed_batch_commit_frames_carry_the_cd_records_committed_ids(
    served: tuple[Any, tw.TapRouter], caplog: Any, capsys: Any
) -> None:
    """Criterion 1: {constrained (G1, end_projection), unconstrained}, 12 decode calls then both releases. Each
    commit frame of the constrained request equals its CD record's ``committed_ids``, canvas by canvas and in
    number; every frame validates and equals its call's rows; the unconstrained request has no record."""
    records = Records(caplog, capsys)
    g = Grammar(G1, "t0069-mix-g1")
    w = _served_world(served)
    _cd_in_chain(w)
    fn = canvas_logits({"c": 41, "u": 42}, period=3)
    w.add("c", xargs=cd_xargs(g, "t69-mix-c"))
    w.add("u")
    w.prefill()
    for _ in range(12):
        w.decode(fn)
    w.remove("c")
    w.remove("u")
    bad = _frames_match_rows(w, ("c", "u"))
    assert bad == [], f"{len(bad)} frame problems:\n" + "\n".join(bad[:20])
    mine = sorted(records.tagged("t69-mix-c"), key=lambda r: r["canvas_idx"])
    assert all(r.get("record_tag") == "t69-mix-c" for r in records.all()), (
        f"a record for another request: {[r.get('record_tag') for r in records.all()]}"
    )
    frames = sorted(tw.commit_frames(w.frames, "c"), key=lambda f: f["canvas_idx"])
    assert len(mine) >= 2, (
        f"fixture: the constrained request must close >= 2 canvases, records {mine}"
    )
    assert [r["error"] for r in mine] == [None] * len(mine), mine
    assert len(frames) == len(mine), f"{len(frames)} commit frames, {len(mine)} CD records"
    assert [f["canvas_idx"] for f in frames] == [r["canvas_idx"] for r in mine]
    assert [list(f["committed_tokens"]) for f in frames] == [list(r["committed_ids"]) for r in mine]
    assert any(r["map_equals_argmax"] is False for r in mine), (
        "fixture: on no canvas did the MAP differ from the stock argmax, so the frames could not tell whether "
        "the tap read CD's commit or stock's"
    )


def test_a_forced_eos_requests_commit_frame_starts_with_id_1(
    served: tuple[Any, tw.TapRouter], caplog: Any, capsys: Any
) -> None:
    """Criterion 1: a loud request (wrong ``astra_cd_automaton_sha256``) ends with an EOS-first committed canvas
    (T-0062 decision 7) within 2 calls; its commit frame starts with id 1 and equals its error record's ids."""
    records = Records(caplog, capsys)
    g = Grammar(G1, "t0069-loud-g1")
    w = _served_world(served)
    xargs = ws.raw_xargs(g.payload, digest="0" * 64, mode="end_projection", record_tag="t69-loud-x")
    w.add("x", xargs=xargs)
    w.add("u")
    w.prefill()
    fn = canvas_logits({"x": 51, "u": 52}, period=3)
    for _ in range(3):
        w.decode(fn)
    w.remove("x")
    frames = tw.commit_frames(w.frames, "x")
    assert frames, "the loud request published no commit frame within 3 calls"
    first = list(frames[0]["committed_tokens"])
    assert first[:1] == [EOS], (
        f"the loud request's commit frame starts {first[:4]}, not with EOS (1)"
    )
    mine = records.tagged("t69-loud-x")
    assert [r["error"] for r in mine] == ["sha_mismatch"], mine
    assert list(mine[0]["committed_ids"]) == first


# --------------------------------------------------------------------------- criterion 2: no CD keys


def _unconstrained_scenario(w: tw.World) -> None:
    """3 requests x >= 2 canvases, a truncated canvas (r2: 3 of CL 4), a prefill-only call, a slot reuse (r4 takes
    r1's slot), then every release. No ``astra_cd_*`` key anywhere."""
    fn = canvas_logits({"r1": 61, "r2": 62, "r3": 63, "r4": 64}, period=3)
    w.add("r1")
    w.add("r2")
    w.reqs["r2"].valid_len = 3
    w.add("r3")
    w.prefill()
    for _ in range(9):
        w.decode(fn)
    slot = w.reqs["r1"].slot
    w.remove("r1")
    assert w.add("r4") == slot, "fixture: r4 must reuse r1's slot"
    w.prefill()
    for _ in range(9):
        w.decode(fn)
    for r in list(w.reqs):
        w.remove(r)


def test_no_cd_keys_tap_cd_stock_is_bit_identical_to_tap_stock(
    served: tuple[Any, tw.TapRouter],
) -> None:
    """Criterion 2: under one seed, Tap ∘ CD ∘ stock (served) and Tap ∘ stock (T-0011's factory alone) agree bit
    for bit after every call -- state rows, ``draft_tokens``, ``SamplerOutput``, global RNG -- and publish the
    same frames, release frames included."""
    make = tw.plugin_attr("model_state", "make_model_state_cls", "T-0011")
    ref_factory = tw.TapFactory()
    ref = tw.World(CFG, make(FakeModelStateBase, tap=ref_factory), factory=ref_factory, seed=7)
    _unconstrained_scenario(ref)
    w = _served_world(served, seed=7)
    _cd_in_chain(w)
    _unconstrained_scenario(w)
    assert len(ref.snaps) == len(w.snaps)
    diffs = [(a.call, a.kind, snaps_differ(a, b)) for a, b in zip(ref.snaps, w.snaps, strict=True)]
    diffs = [d for d in diffs if d[2]]
    assert diffs == [], f"CD with no astra_cd_* key changed the engine: {diffs[:5]}"
    assert ref.frames, "the reference published nothing: the frame comparison would be vacuous"
    assert w.frames == ref.frames, "the published frames differ"
    assert w.release_frames == ref.release_frames


def _owner(stack: list[traceback.FrameSummary]) -> str:
    """Whose code issued a torch op: from the innermost frame outward, the first plugin or stock frame decides
    (torch internals and this module are skipped). CD frames sit *outside* the stock frames they delegate to,
    so a stock op is never charged to CD, nor a CD op to the tap that delegates to it."""
    for f in reversed(stack):
        name = f.filename.replace("\\", "/")
        if "/astra_vllm_plugin/cd/" in name:
            return "cd"
        if "/astra_vllm_plugin/" in name:
            return "tap"
        if "vllm_v0_29_0" in name or "/vllm/" in name:
            return "stock"
    return "other"


class _Ops:
    def __init__(self) -> None:
        self.by_owner: dict[str, list[str]] = defaultdict(list)

    def hit(self, what: str) -> None:
        self.by_owner[_owner(traceback.extract_stack()[:-2])].append(what)


class _FnMode(torch.overrides.TorchFunctionMode):
    def __init__(self, ops: _Ops) -> None:
        super().__init__()
        self.ops = ops

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.ops.hit(f"fn:{getattr(func, '__name__', func)}")
        return func(*args, **(kwargs or {}))


class _DispatchMode(TorchDispatchMode):
    def __init__(self, ops: _Ops) -> None:
        super().__init__()
        self.ops = ops

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.ops.hit(f"aten:{func}")
        return func(*args, **(kwargs or {}))


def _instrumented(w: tw.World, ops: _Ops) -> None:
    import contextlib

    @contextlib.contextmanager
    def ctx() -> Any:
        with _DispatchMode(ops), _FnMode(ops):
            yield

    w.call_ctx = ctx


def test_no_cd_keys_the_cd_wrapper_issues_zero_torch_ops_through_the_tap(
    served: tuple[Any, tw.TapRouter],
) -> None:
    """Criterion 2 (T-0067 c3 through the tap): with no ``astra_cd_*`` key, every sampler call and release of the
    scenario under ``TorchFunctionMode`` + ``TorchDispatchMode``: zero ops are CD's, while the tap's and stock's
    are seen (the instrument is live through the composition)."""
    w = _served_world(served)
    _cd_in_chain(w)
    ops = _Ops()
    _instrumented(w, ops)
    _unconstrained_scenario(w)
    assert ops.by_owner.get("stock") and ops.by_owner.get("tap"), (
        f"instrument dead: owners seen {sorted(ops.by_owner)}"
    )
    cd_ops = ops.by_owner.get("cd", [])
    assert cd_ops == [], (
        f"{len(cd_ops)} torch ops by the CD wrapper with no astra_cd_* key: {cd_ops[:10]}"
    )


def test_the_op_attribution_sees_cd_when_a_slot_is_constrained(
    served: tuple[Any, tw.TapRouter],
) -> None:
    """Liveness of the row above, through the served composition: one constrained slot makes CD issue ops."""
    g = Grammar(G1, "t0069-ops-g1")
    w = _served_world(served)
    w.add("c", xargs=cd_xargs(g, "t69-ops-c"))
    w.add("u")
    w.prefill()
    ops = _Ops()
    _instrumented(w, ops)
    fn = canvas_logits({"c": 71, "u": 72}, period=3)
    for _ in range(3):
        w.decode(fn)
    assert ops.by_owner.get("cd"), (
        f"no op attributed to CD with a constrained slot in the batch (owners {sorted(ops.by_owner)}): "
        "CD is not in the served chain, or the attribution is dead"
    )


# --------------------------------------------------------------------------- adversarial: R6 under composition


def test_r6_a_loud_request_freed_and_its_slot_retaken_before_the_drain(
    served: tuple[Any, tw.TapRouter], caplog: Any, capsys: Any
) -> None:
    """R6 under composition. ``x`` (constrained, wrong sha: loud) commits its EOS canvas and is released; ``u``
    (unconstrained) takes the same slot and commits; only then is the tap drained. Each occupant's commit frames
    carry its own committed rows (x: EOS-first, u: live ids, never EOS-first) under its own id, u's frames start at
    ``canvas_idx`` 0, every frame validates, and the CD records are exactly x's one error record."""
    records = Records(caplog, capsys)
    g = Grammar(G1, "t0069-r6-g1")
    w = _served_world(served, drain_each_call=False)
    fn = canvas_logits({"x": 81, "u": 82}, period=3)
    slot = w.add(
        "x",
        xargs=ws.raw_xargs(
            g.payload, digest="f" * 64, mode="end_projection", record_tag="t69-r6-x"
        ),
    )
    w.prefill()
    committed: dict[str, list[list[int]]] = defaultdict(list)
    for _ in range(4):
        s = w.decode(fn)
        if s.committed("x") is not None:
            committed["x"].append(s.committed("x"))
            break
    assert committed["x"], "fixture: x committed nothing in 4 calls"
    w.remove("x")
    assert w.add("u") == slot, "fixture: u must take x's slot"
    w.prefill()
    for _ in range(9):
        s = w.decode(fn)
        if s.committed("u") is not None:
            committed["u"].append(s.committed("u"))
    w.remove("u")
    assert committed["u"], "fixture: u committed nothing"
    w.drain()  # the first drain: every frame above was staged before it
    for f in w.frames:
        validate_canvas_step(f)
    got = {
        r: [list(f["committed_tokens"]) for f in tw.commit_frames(w.frames, r)] for r in ("x", "u")
    }
    assert got["x"] == committed["x"] and got["x"][0][:1] == [EOS], (
        f"x's commit frames {got['x']} (its committed rows {committed['x']}; T-0062 decision 7: EOS-first)"
    )
    assert got["u"] == committed["u"], f"u's commit frames {got['u']} != its rows {committed['u']}"
    assert all(t[:1] != [EOS] for t in got["u"]), "an EOS-first canvas published under u's id"
    u_frames = [f for f in w.frames if f["request_id"] == "u"]
    assert u_frames and u_frames[0]["canvas_idx"] == 0, "u must start at canvas_idx 0"
    assert {f["request_id"] for f in w.frames} <= {"x", "u"}
    recs = records.all()
    assert [(r["record_tag"], r["error"]) for r in recs] == [("t69-r6-x", "sha_mismatch")], (
        f"CD records {[(r.get('record_tag'), r.get('error')) for r in recs]}: exactly x's error record"
    )


# --------------------------------------------------------------------------- ruling O6: a loud end publishes its EOS commit


def _ended_on_first_call(
    served: tuple[Any, tw.TapRouter], monkeypatch: pytest.MonkeyPatch, extra_calls: int
) -> tuple[tw.World, list[tuple[int, list[int]]]]:
    """CD fails inside its pre-step on the request's first decode call (δ* raises; T-0067 S9 ends it
    ``internal`` with an EOS canvas). The request stays scheduled for ``extra_calls`` calls after its first
    commit (0: the engine's behaviour, it stops on the EOS). Returns the world and ``(pre-call step, ids)`` per
    commit of ``c``."""
    from tests.cd.unit._sampler_support import patch_everywhere

    w = _served_world(served)
    _cd_in_chain(w)
    w.add("c", xargs=cd_xargs(Grammar(G1, "t0069-o6-g1"), "t69-o6-c"))
    w.add("u")
    w.prefill()

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("injected: CD fails on its first constrained step (ruling O6)")

    assert patch_everywhere(monkeypatch, "delta_star", boom) > 0, (
        "cannot inject: no delta_star in cd.*"
    )
    fn = canvas_logits({"c": 31, "u": 32}, period=3)
    commits: list[tuple[int, list[int]]] = []
    left = None
    for _ in range(4 + extra_calls):
        s = w.decode(fn)
        ids = s.committed("c")
        if ids is not None:
            commits.append((int(s.pre_step[w.reqs["c"].slot]), ids))  # type: ignore[attr-defined]
            left = extra_calls if left is None else left
        if left is not None:
            if left == 0:
                break
            left -= 1
    w.remove("c")
    return w, commits


def test_o6_a_request_ended_by_cd_on_its_first_call_publishes_its_eos_commit(
    served: tuple[Any, tw.TapRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling O6, engine order: the request is removed after its first (EOS) commit. Its one commit frame is
    published, validates, starts with id 1 and carries the pre-reset step; the tap counts no error and no
    overrun. Kills a tap that drops or mis-steps the forced commit."""
    w, commits = _ended_on_first_call(served, monkeypatch, extra_calls=0)
    assert len(commits) == 1 and commits[0][1][:1] == [EOS], f"fixture: c's commits {commits}"
    frames = tw.commit_frames(w.frames, "c")
    for f in frames:
        validate_canvas_step(f)
    assert [(f["step"], list(f["committed_tokens"])) for f in frames] == commits
    assert w.tap is not None and (w.tap.errors, w.tap.commit_overrun) == (0, 0)


def test_o6_eos_commits_with_step_0_are_published_not_counted_as_errors(
    served: tuple[Any, tw.TapRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling O6, the tier-2 shape (91baefa: ``errors=48 commit_overrun=1128``): a slot CD has ended stays forced
    to its EOS canvas, so if it is scheduled again it commits again with the pre-reset step 0 (stock reset the
    step at the previous commit). Every such commit is published exactly once with ``step == 0`` and
    ``committed_tokens[0] == 1``; ``tap.errors == 0`` and ``commit_overrun == 0``. Red while ``encode_step``
    refuses a commit at step 0 (and the refused commits then re-count as overrun on every later call)."""
    w, commits = _ended_on_first_call(served, monkeypatch, extra_calls=3)
    assert len(commits) == 4 and any(step == 0 for step, _ in commits), f"fixture: {commits}"
    frames = tw.commit_frames(w.frames, "c")
    bad = []
    for f in frames:
        try:
            validate_canvas_step(f)
        except AssertionError as exc:
            bad.append(str(exc))
    assert bad == []
    assert w.tap is not None and w.tap.errors == 0, (
        f"tap.errors {w.tap.errors}: a step-0 commit was refused"
    )
    assert [(f["step"], list(f["committed_tokens"])) for f in frames] == commits
    assert all(ids[:1] == [EOS] for _, ids in commits)
    assert w.tap.commit_overrun == 0, f"commit_overrun {w.tap.commit_overrun}"


# --------------------------------------------------------------------------- ruling O5: CD device constants exist before a call

_O5_SCRIPT = r"""
import json, traceback, torch
from tests.cd import _tap_cd_world as tw
from tests.cd.fakes.diffusion_sampler import Cfg, FakeModelStateBase
from tests.cd.unit._sampler_support import G1, Grammar, cd_xargs
from tests.cd.unit._scenario import canvas_logits

hits = []

class FromHost(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.tensor, torch.as_tensor) and args and not isinstance(args[0], torch.Tensor):
            cd = [f for f in traceback.extract_stack() if "/astra_vllm_plugin/cd/" in f.filename.replace("\\", "/")]
            if cd:
                hits.append(f"{func.__name__}({type(args[0]).__name__}) at {cd[-1].filename.split('/')[-1]}:{cd[-1].lineno} {cd[-1].name}")
        return func(*args, **(kwargs or {}))

# liveness: a host-data tensor built in a cd/ frame is seen
ns = {"torch": torch}
exec(compile("def build():\n    return torch.as_tensor([1, 50, 106])\n", "/x/astra_vllm_plugin/cd/budget.py", "exec"), ns)
with FromHost():
    ns["build"]()
live = list(hits)
hits.clear()

make = tw.plugin_attr("model_state", "make_model_state_cls")
make_cd = tw.plugin_attr("cd.model_state", "make_cd_model_state_cls")
f = tw.TapFactory()
w = tw.World(Cfg(), make(make_cd(FakeModelStateBase), tap=f), factory=f)
w.call_ctx = FromHost  # only the sampler calls and the release: wrap and bind are free to build constants
w.add("c", xargs=cd_xargs(Grammar(G1, "t0069-o5-g1"), "t69-o5-c"))
w.add("u")
w.prefill()
fn = canvas_logits({"c": 21, "u": 22}, period=3)
for _ in range(8):
    w.decode(fn)
w.remove("c")
print("ASTRA_O5 " + json.dumps({"live": live, "hits": hits}))
"""


def test_o5_cd_builds_no_tensor_from_host_data_inside_a_sampler_call() -> None:
    """Ruling O5: every CD device constant (the stop-id tensor of ``delta_star`` first) is built at wrap or bind,
    never inside a sampler call -- on CUDA, ``torch.as_tensor(list, device=cuda)`` there is an H2D sync that
    sync-debug "error" turned into ``internal`` (tier 2, 91baefa). On CPU the spy is ``torch.tensor``/
    ``torch.as_tensor`` with non-tensor data whose stack passes through ``astra_vllm_plugin/cd/``, over the
    first constrained request's 8 decode calls and its release. Runs in a fresh interpreter so no cache warmed
    by an earlier row can hide a lazy build. Kills the lazy ``_stop_tensor`` build (red at fe08892)."""
    import json
    import os

    env = {**os.environ, "ASTRA_CD_TESTS": "1", "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-c", _O5_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_O5 ")), None)
    assert line is not None, (
        f"the O5 probe printed nothing (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
    )
    got = json.loads(line[len("ASTRA_O5 ") :])
    assert got["live"], "spy dead: a host-data tensor built in a cd/ frame was not seen"
    assert got["hits"] == [], f"CD built tensors from host data inside sampler calls: {got['hits']}"
