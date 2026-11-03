"""T-0066 criterion 7 (negative): the MAP uses no randomness; neither function touches the global RNG,
makes a host sync on the per-step path (the T-0065 spy pair: ``TorchFunctionMode`` for the Python-level
spellings, ``TorchDispatchMode`` for the ATen ops they lower to), or imports vLLM.
"""

from __future__ import annotations

import ast
import inspect
import random

import pytest

from tests.cd.oracle.enumerate import distance_to_final
from tests.cd.unit._core_support import CD_DIR, random_toy
from tests.cd.unit._draw_support import T0066_MODULES, cdmod, draw, generator, prepare, run_map
from tests.cd.unit.test_core_hygiene import _sync_spies
from tests.cd.unit.test_draw_support_and_rng import random_op_spy


@pytest.mark.parametrize("name", T0066_MODULES)
def test_no_vllm_import(name):
    path = CD_DIR / f"{name}.py"
    if not path.exists():
        pytest.fail(f"T-0066 not implemented: {path} is missing")
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] == "vllm"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "vllm":
                bad.append(node.module)
        elif isinstance(node, ast.Call) and getattr(
            node.func, "attr", getattr(node.func, "id", "")
        ) in (
            "import_module",
            "__import__",
        ):
            bad += [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and a.value.split(".")[0] == "vllm"
            ]
    assert not bad, f"{name}.py imports vllm: {bad}"


def _prep():
    import torch

    rng = random.Random(4)
    toy = random_toy(rng, V=32, S=5, nfa=True)
    d = distance_to_final(toy)
    end = [s for s in range(5) if d[s] <= 2]
    logits = torch.tensor(
        [[rng.uniform(-30, 0) for _ in range(32)] for _ in range(16)], dtype=torch.float32
    )
    return prepare(logits, toy, [0, 1], end, l_eff=12, bucket=8)


def test_map_uses_no_randomness():
    import numpy as np
    import torch

    params = inspect.signature(cdmod("mapdecode").constrained_map).parameters
    assert "generator" not in params, "the MAP takes a generator: it must be deterministic"
    prep = _prep()
    seen, Spy = random_op_spy()
    before = (
        torch.random.get_rng_state().clone(),
        np.random.get_state()[1].copy(),
        random.getstate(),
    )
    with Spy():
        a = run_map(prep)
    assert not seen, f"random ops in the MAP: {seen}"
    assert torch.equal(before[0], torch.random.get_rng_state()), "torch global RNG consumed"
    assert (before[1] == np.random.get_state()[1]).all(), "numpy global RNG consumed"
    assert before[2] == random.getstate(), "python global RNG consumed"
    torch.manual_seed(12345)  # a different global state changes nothing
    b = run_map(prep)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.score, b.score)


def test_draw_does_not_touch_the_global_rng():
    import numpy as np
    import torch

    prep = _prep()
    before = (
        torch.random.get_rng_state().clone(),
        np.random.get_state()[1].copy(),
        random.getstate(),
    )
    draw(prep, generator(1))
    assert torch.equal(before[0], torch.random.get_rng_state()), "torch global RNG consumed"
    assert (before[1] == np.random.get_state()[1]).all(), "numpy global RNG consumed"
    assert before[2] == random.getstate(), "python global RNG consumed"
    # and the global state does not influence the draw
    torch.manual_seed(0)
    x = draw(prep, generator(9))
    torch.manual_seed(1)
    y = draw(prep, generator(9))
    assert torch.equal(x.tokens, y.tokens) and torch.equal(x.states, y.states)


@pytest.mark.parametrize("which", ["draw", "map"])
def test_no_host_sync_on_the_per_step_path(which, monkeypatch):
    import torch

    prep = _prep()
    g = generator(2)
    cdmod("draw" if which == "draw" else "mapdecode")
    seen, FnSpy, AtenSpy = _sync_spies()
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: seen.append("torch.cuda.synchronize")
    )
    with FnSpy(), AtenSpy():
        res = draw(prep, g) if which == "draw" else run_map(prep)
    assert not seen, f"synchronizing calls in {which}: {sorted(set(seen))}"
    with FnSpy(), AtenSpy():
        res.tokens.tolist()
    assert seen, "spy failed to see .tolist() (vacuous pass guard)"
