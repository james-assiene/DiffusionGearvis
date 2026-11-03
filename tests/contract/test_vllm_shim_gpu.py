"""T-0015 criteria 1-3: the DiffusionGemma image-path shim, checked against the pinned container.

Upstream's ``DiffusionGemmaForConditionalGeneration`` does not subclass
``Gemma4ForConditionalGeneration``; it borrows methods by class-level assignment
(``diffusion_gemma.py``, "architecturally identical to Gemma4. Delegate to avoid maintaining a
duplicate copy."). That borrow is only safe while the borrowed code touches no state that
Gemma4's ``__init__`` sets and DiffusionGemma's does not -- and since vLLM commit ``95c914442``
it does (``self._enable_mm_lora``), which takes the EngineCore down on every image request.

Three tests, one per criterion:

* ``test_shim_is_shaped_as_specified``  -- the shim supplies the state and shadows nothing.
* ``test_upstream_defect_is_still_present`` -- the deletion trigger. Red here means *good* news.
* ``test_no_borrowed_method_reads_an_unresolvable_attribute`` -- the audit, re-derived from
  upstream source on every run so a vLLM bump that adds a sibling defect is caught.

All three run inside the container (the host has no ``vllm``) and are skipped, not failed, where
the image is absent. The facts come from ``tests/container_scripts/astra_shim_probe.py``; every
assertion is here.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.vllm_container import run_probe

pytestmark = pytest.mark.gpu

DONOR_MODULE = "vllm.model_executor.models.gemma4_mm"
DONOR_CLASS = "Gemma4ForConditionalGeneration"
STOCK_MODULE = "vllm.model_executor.models.diffusion_gemma"
STOCK_CLASS = "DiffusionGemmaForConditionalGeneration"

DELETE_THE_SHIM = """
READ THIS BEFORE TOUCHING THE ASSERTION.

A failure here is GOOD NEWS, not a regression. This test's only job is to prove that the
upstream vLLM defect the shim works around is STILL THERE. It fails exactly when upstream has
fixed it -- when DiffusionGemma finally owns `_enable_mm_lora` (by subclassing Gemma4, by
setting it in its own __init__, or because Gemma4 stopped requiring it).

The correct response is to DELETE, in one commit:
  - the `_enable_mm_lora` shim in plugin/astra_vllm_plugin/astra_vllm_plugin/model.py,
  - this test,
  - the shim's entry in .claude/rules/gpu-services.md,
and then to re-run the image path on the pinned container to confirm it is healthy without us.

Do NOT relax, invert or skip this assertion to get back to green: that keeps a dead workaround
alive forever, silently forking our vision path from upstream's. If the image path is broken in
a NEW way, that is a new task, not an edit to this one. Context: docs/tasks/T-0015.md.
"""


@pytest.fixture(scope="session")
def probe() -> dict[str, Any]:
    """One container run for all three criteria (starting the image is the expensive part)."""
    facts = run_probe("astra_shim_probe.py")
    if "error" in facts:
        pytest.fail(f"probe failed inside the pinned container:\n{facts['error']}")
    return facts


def test_shim_is_shaped_as_specified(probe: dict[str, Any]) -> None:
    """Criterion 1: derives from the stock class, supplies the missing state, shadows nothing."""
    shim = probe["shim"]

    base = shim["bases"][0]
    assert shim["base0_is_stock"] is True, (
        f"the shim's first base is {base['module']}.{base['qualname']}, not "
        f"{STOCK_MODULE}.{STOCK_CLASS}: it must extend the stock model so vLLM keeps every "
        "upstream behaviour we did not deliberately change"
    )
    assert (base["module"], base["qualname"]) == (STOCK_MODULE, STOCK_CLASS)

    lora = shim["enable_mm_lora"]
    assert lora["present"] is True, (
        "the shim does not provide `_enable_mm_lora`, so every image request raises "
        "AttributeError inside the model runner and kills the EngineCore (EngineDeadError), "
        "not just the request"
    )
    assert lora["defined_in_own_class_body"] is True, (
        "`_enable_mm_lora` is inherited rather than set by the shim; the whole point is that "
        "no class in the stock MRO defines it (see the criterion-2 test)"
    )
    assert lora["is_false"] is True, (
        f"`_enable_mm_lora` is {lora['repr']} ({lora['type']}), expected the False singleton: "
        "we load no multimodal LoRA, and the truthy branches in Gemma4._process_image_input "
        "reach LoRA state that nothing in this process ever sets"
    )

    image = shim["process_image_input"]
    assert image["shadowed_in_own_class_body"] is False, (
        "the shim defines its own `_process_image_input`. It must not: the shim exists to "
        "supply missing *state*, and redefining the borrowed method silently forks upstream's "
        "vision code, which then stops tracking vLLM bumps"
    )
    assert image["is_donor_function"] is True, (
        f"`_process_image_input` resolves to {image['astra']['module']}."
        f"{image['astra']['qualname']}, not to {DONOR_CLASS}'s function object"
    )
    assert (image["astra"]["module"], image["astra"]["qualname"]) == (
        DONOR_MODULE,
        f"{DONOR_CLASS}._process_image_input",
    )


def test_upstream_defect_is_still_present(probe: dict[str, Any]) -> None:
    """Criterion 2: the deletion trigger. See DELETE_THE_SHIM -- red here means upstream fixed it."""
    upstream = probe["upstream"]

    assert upstream["stock_mro_classes_defining_it"] == [], (
        f"{STOCK_CLASS} (or a class in its MRO) now defines "
        f"`{upstream['attribute']}`: {upstream['stock_mro_classes_defining_it']}.\n{DELETE_THE_SHIM}"
    )
    assert upstream["stock_subclasses_donor"] is False, (
        f"{STOCK_CLASS} now subclasses {DONOR_CLASS}, so it inherits Gemma4's __init__ and the "
        f"borrowed-method problem is gone at the root.\n{DELETE_THE_SHIM}"
    )
    assert upstream["stock_assigns_it_in_source"] is False, (
        f"{STOCK_CLASS}.__init__ now sets `{upstream['attribute']}` itself.\n{DELETE_THE_SHIM}"
    )
    assert upstream["donor_methods_reading_it_bare"], (
        f"no {DONOR_CLASS} method reads `{upstream['attribute']}` as a bare attribute any "
        "more, so a class that borrows those methods no longer needs to provide it.\n"
        f"{DELETE_THE_SHIM}"
    )
    assert upstream["donor_reads_it_via_getattr"] is False, (
        f"{DONOR_CLASS} now reads `{upstream['attribute']}` through getattr() with a default, "
        "which makes the borrow safe without us.\n"
        f"{DELETE_THE_SHIM}"
    )


def test_no_borrowed_method_reads_an_unresolvable_attribute(probe: dict[str, Any]) -> None:
    """Criterion 3: the audit, re-derived. A vLLM bump that adds a sibling defect fails here."""
    audit = probe["audit"]
    borrowed = audit["borrowed"]

    assert borrowed, (
        "no borrowed methods were found in the class body of "
        f"{STOCK_CLASS}. Either upstream changed how it reuses Gemma4's vision code -- in which "
        "case this audit is looking in the wrong place and must be rewritten before it can be "
        "trusted again -- or the parser in astra_shim_probe.py broke. An empty borrow list "
        "would make this test pass while checking nothing."
    )
    assert "_process_image_input" in borrowed, (
        f"{STOCK_CLASS} no longer borrows `_process_image_input` from {DONOR_CLASS}; the image "
        "path this shim protects has been rewritten upstream and T-0015 needs re-deriving"
    )
    assert audit["supported_modalities"], (
        "DiffusionGemmaProcessingInfo declares no supported modalities; the audit derives its "
        "dead-branch exclusions from that list, and an empty one would exclude nothing or "
        "everything depending on the branch shape"
    )
    assert audit["methods"]["_process_image_input"]["self_loads"], (
        "the audit found no `self.<attr>` reads at all in the borrowed `_process_image_input`, "
        "which cannot be true of a method that uses the vision tower: the AST walk is broken "
        "and the audit is vacuous"
    )

    check_audit(audit)


def check_audit(audit: dict[str, Any]) -> None:
    """The criterion-3 verdict on the probe's audit facts. Pure, so the host can test it.

    ``unresolved`` is computed against the *stock* class. Each entry is fine iff it is excluded
    as dead (upstream declares no modality that reaches it) or the shim supplies it as plain
    state. Anything else is either missing outright or answered by a shim stub -- and a stub on
    a live branch raises inside the model runner just as surely as a missing attribute does.
    """
    offenders: dict[str, Any] = {}
    stubs: dict[str, Any] = {}
    for name, method in sorted(audit["methods"].items()):
        label = f"{name} (borrowed from {DONOR_CLASS}.{method['donor_method']})"
        live = {
            attr: info
            for attr, info in method["unresolved"].items()
            if not info["excluded_as_dead"] and info["supplied_by_shim"] != "state"
        }
        missing = {
            attr: info["reachable_sites"]
            for attr, info in live.items()
            if info["supplied_by_shim"] is None
        }
        stubbed = {
            attr: info["reachable_modalities"]
            for attr, info in live.items()
            if info["supplied_by_shim"] is not None
        }
        if missing:
            offenders[label] = missing
        if stubbed:
            stubs[label] = stubbed

    assert not offenders, (
        "borrowed method(s) read `self.<attr>` for attributes that do not resolve on the shim. "
        "Each one is an AttributeError inside the model runner on a live request, which takes "
        "the EngineCore down (EngineDeadError), not just the request.\n\n"
        f"{_format(offenders)}\n"
        "This is the same class of defect as `_enable_mm_lora` (vLLM commit 95c914442): a "
        "borrowed method touching state that only Gemma4's __init__ sets. Fix it in "
        "plugin/astra_vllm_plugin/astra_vllm_plugin/model.py by supplying the state the shim "
        "is missing -- do not copy the method body across, that forks upstream's vision code.\n"
        f"Modalities this model declares: {audit['supported_modalities']}. Branches guarded by "
        "any other modality are excluded as unreachable; the listing above shows only the "
        "guard sets that survive that exclusion."
    )
    assert not stubs, (
        "borrowed method(s) reach, on a branch upstream now declares live, a member that only "
        "the shim's stub provides. The stubs raise NotImplementedError by design, so a request "
        "carrying that modality kills the EngineCore (EngineDeadError).\n\n"
        f"{_format_stubs(stubs)}\n"
        f"Modalities this model declares: {audit['supported_modalities']}. The stubs were only "
        "acceptable while those branches were dead. Upstream now routes this modality to "
        "DiffusionGemma: re-derive T-0015 (does DiffusionGemma have the tower now? should the "
        "borrowed implementation be used instead of the stub?) -- do not exclude the stub here."
    )


def _format(offenders: dict[str, Any]) -> str:
    lines = []
    for method, attrs in offenders.items():
        lines.append(f"  {method}:")
        for attr, guards in attrs.items():
            lines.append(f"    self.{attr}  reachable under guards {guards}")
    return "\n".join(lines)


def _format_stubs(stubs: dict[str, Any]) -> str:
    lines = []
    for method, attrs in stubs.items():
        lines.append(f"  {method}:")
        for attr, modalities in attrs.items():
            under = ", ".join(repr(m) for m in modalities) or "no modality guard (always live)"
            lines.append(f"    self.{attr}  is a shim stub, reachable under modality {under}")
    return "\n".join(lines)
