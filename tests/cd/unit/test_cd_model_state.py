"""T-0067 criteria 1-2: composition with the model state, and attribute transparency.

Mirrors T-0011 criteria 1-2 (not T-0011's code: T-0011 is planned, not merged). The inner sampler is the
vendored v0.29.0 ``DiffusionSampler`` built by ``FakeModelStateBase.custom_sampler``.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from tests.cd.fakes.diffusion_sampler import (
    PLUGIN_SRC,
    Cfg,
    FakeModelStateBase,
    FakeRunnerSampler,
    cd_sampler_cls,
    make_cd_model_state_cls,
)
from tests.fixtures.vllm_v0_29_0 import diffusion_sampler as stock

CD_DIR = PLUGIN_SRC / "astra_vllm_plugin" / "cd"
NEW_MODULES = ("sampler", "requests", "record", "model_state")

#: T-0011 criterion 2's probe list (the runner feature-detects by attribute).
PROBE_NAMES = (
    "sampling_states",
    "req_states",
    "logits_mode",
    "penalties_state",
    "apply_staged_writes",
    "add_request",
    "compute_nans",
    "get_logprobs_dims",
    "diffusion_states",
    "canvas_length",
    "remove_request",
    "states",
)
INTERCEPTED = ("add_request",)
FORBIDDEN_NAMES = ("EntropyBound", "gen_config", "t_min", "t_max", "embed_weight", "shard_indices")


class _Second:
    pass


def _wrapped(cfg: Cfg | None = None) -> tuple[Any, Any, Any]:
    cls = make_cd_model_state_cls()(FakeModelStateBase)
    ms = cls(cfg or Cfg())
    got = ms.custom_sampler(FakeRunnerSampler(ms.cfg))
    return ms, got, ms.constructed[-1]


def test_factory_returns_a_subclass_of_the_base() -> None:
    cls = make_cd_model_state_cls()(FakeModelStateBase)
    assert isinstance(cls, type) and issubclass(cls, FakeModelStateBase)


def test_custom_sampler_wraps_exactly_what_super_returned() -> None:
    second = _Second()

    class Base(FakeModelStateBase):
        def custom_sampler(self, sampler: Any) -> Any:
            inner, _ = super().custom_sampler(sampler)
            return inner, second

    ms = make_cd_model_state_cls()(Base)(Cfg())
    got = ms.custom_sampler(FakeRunnerSampler(ms.cfg))
    assert isinstance(got, tuple) and len(got) == 2, (
        f"custom_sampler must return a 2-tuple, got {got!r}"
    )
    wrapper, sec = got
    assert sec is second, "the second element must be the object super returned (identity)"
    assert isinstance(wrapper, cd_sampler_cls())
    assert len(ms.constructed) == 1, "super().custom_sampler must be called exactly once"
    inner = ms.constructed[0]
    assert wrapper.diffusion_states is inner.diffusion_states
    assert wrapper.sampling_states is inner.sampling_states


def test_custom_sampler_returns_none_when_super_does() -> None:
    class Base(FakeModelStateBase):
        custom_sampler_mode = "none"

    ms = make_cd_model_state_cls()(Base)(Cfg())
    assert ms.custom_sampler(FakeRunnerSampler(ms.cfg)) is None


def test_custom_sampler_propagates_super_exception_unchanged() -> None:
    boom = ValueError("DiffusionGemma requires an EntropyBound sampler_config")

    class Base(FakeModelStateBase):
        def custom_sampler(self, sampler: Any) -> Any:
            raise boom

    ms = make_cd_model_state_cls()(Base)(Cfg())
    with pytest.raises(ValueError) as info:
        ms.custom_sampler(FakeRunnerSampler(ms.cfg))
    assert info.value is boom


def test_composes_under_a_tap_style_factory() -> None:
    """T-0011's ``make_model_state_cls(make_cd_model_state_cls(base), tap=...)`` shape: an outer factory
    that wraps what super returned sees the CD wrapper as its inner, and the second element survives."""

    class Tap:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

    def make_tap(base: type) -> type:
        class TapState(base):  # type: ignore[misc,valid-type]
            def custom_sampler(self, sampler: Any) -> Any:
                got = super().custom_sampler(sampler)
                return None if got is None else (Tap(got[0]), got[1])

        return TapState

    ms = make_tap(make_cd_model_state_cls()(FakeModelStateBase))(Cfg())
    tap, sec = ms.custom_sampler(FakeRunnerSampler(ms.cfg))
    assert isinstance(tap._inner, cd_sampler_cls()) and sec is None


def _code_names(path: Any) -> set[str]:
    """Identifiers and attribute names used in code (strings and comments stripped)."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def test_cd_code_never_constructs_diffusion_sampler_or_reads_its_inputs() -> None:
    missing = [m for m in NEW_MODULES if not (CD_DIR / f"{m}.py").exists()]
    assert missing == [], f"T-0067 not implemented: cd/{missing} missing"
    bad = {}
    for m in NEW_MODULES:
        used = _code_names(CD_DIR / f"{m}.py")
        hit = sorted(used & ({"DiffusionSampler"} | set(FORBIDDEN_NAMES)))
        if hit:
            bad[m] = hit
    assert bad == {}, f"CD code names the stock sampler's construction inputs (criterion 1): {bad}"


def test_attribute_parity_with_the_inner_sampler() -> None:
    _ms, (wrapper, _), inner = _wrapped()
    mismatched = [n for n in PROBE_NAMES if hasattr(wrapper, n) != hasattr(inner, n)]
    assert mismatched == [], f"hasattr disagrees with the inner sampler on {mismatched}"
    for name in PROBE_NAMES:
        if not hasattr(inner, name) or name in INTERCEPTED:
            continue
        got, want = getattr(wrapper, name), getattr(inner, name)
        if inspect.ismethod(want):
            assert got.__func__ is want.__func__ and got.__self__ is inner, name
        else:
            assert got is want, f"{name} must be the inner object itself"


def test_wrapper_has_no_remove_request() -> None:
    cls = cd_sampler_cls()
    assert "remove_request" not in vars(cls)
    _ms, (wrapper, _), _inner = _wrapped()
    assert not hasattr(wrapper, "remove_request")
    assert not hasattr(stock.DiffusionSampler, "remove_request"), "fixture: upstream has none"


def test_wrapper_defines_no_public_attribute_the_inner_lacks() -> None:
    _ms, (wrapper, _), inner = _wrapped()
    extra = {n for n in dir(wrapper) if not n.startswith("_")} - set(dir(inner))
    assert extra == set(), f"public attributes the inner sampler lacks: {sorted(extra)}"


def test_add_request_reaches_the_inner_sampler() -> None:
    """Interception must still delegate: the stock ``SamplingStates`` sees the request's top_p."""
    from tests.cd.fakes.diffusion_sampler import sampling_params

    _ms, (wrapper, _), inner = _wrapped()
    wrapper.add_request(3, 5, sampling_params(top_p=0.5))
    wrapper.apply_staged_writes()
    assert float(inner.sampling_states.top_p.np[3]) == 0.5
