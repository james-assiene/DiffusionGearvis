"""The seven arms and what each puts on the wire (T-0070 criterion 4, T-0076 criterion 8).

``unconstrained`` sends no ``astra_cd_*`` key. The six constrained arms send a T-0064 ``CdRequest``
built with ``to_xargs()`` (ruling W4):

======================  ===================  ===================
arm                     ``mode``             ``self_conditioning``
======================  ===================  ===================
end_projection          end_projection       model
perstep_mf              perstep_mf           model
perstep_mar             perstep_mar          model
perstep_mar_sc          perstep_mar          constrained
perstep_mar_qargmax     perstep_mar_qargmax  model
perstep_mar_qargmax_sc  perstep_mar_qargmax  constrained
======================  ===================  ===================

The Mar arms (both families; an SC twin reuses its base arm's frozen file, T-0076 Q3) also carry ``entropy_bound`` and ``confidence_threshold`` from the frozen calibration
(criterion 10). The record does not encode ``_sc``; the harness knows it from its own request, joined on
``record_tag`` (T-0064 W5).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any

from local_astra.contracts.cd import CD_XARGS_PREFIX, CdRequest

__all__ = [
    "ARMS",
    "ARM_SPECS",
    "CONSTRAINED",
    "MAR_ARMS",
    "ArmSpec",
    "arm_knobs",
    "cd_xargs",
    "row_knobs",
]


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    mode: str | None
    self_conditioning: str | None
    needs_calibration: bool


ARM_SPECS: Mapping[str, ArmSpec] = {
    "unconstrained": ArmSpec("unconstrained", None, None, False),
    "end_projection": ArmSpec("end_projection", "end_projection", "model", False),
    "perstep_mf": ArmSpec("perstep_mf", "perstep_mf", "model", False),
    "perstep_mar": ArmSpec("perstep_mar", "perstep_mar", "model", True),
    "perstep_mar_sc": ArmSpec("perstep_mar_sc", "perstep_mar", "constrained", True),
    # T-0076: the settling arm (history = argmax of q) and its SC twin
    "perstep_mar_qargmax": ArmSpec("perstep_mar_qargmax", "perstep_mar_qargmax", "model", True),
    "perstep_mar_qargmax_sc": ArmSpec(
        "perstep_mar_qargmax_sc", "perstep_mar_qargmax", "constrained", True
    ),
}
ARMS = tuple(ARM_SPECS)
CONSTRAINED = tuple(a for a in ARMS if ARM_SPECS[a].mode is not None)
MAR_ARMS = tuple(a for a in ARMS if ARM_SPECS[a].needs_calibration)


def spec(arm: str) -> ArmSpec:
    try:
        return ARM_SPECS[arm]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}; one of {list(ARMS)}") from None


def arm_knobs(arm: str, mar: Mapping[str, float] | None = None) -> dict[str, Any]:
    """The arm-level knobs (no per-request tag or automaton): what the regime line records."""
    s = spec(arm)
    out: dict[str, Any] = {"mode": s.mode, "self_conditioning": s.self_conditioning}
    if s.needs_calibration:
        if mar is None:
            raise ValueError(f"{arm} needs entropy_bound and confidence_threshold (calibration)")
        out["entropy_bound"] = float(mar["entropy_bound"])
        out["confidence_threshold"] = float(mar["confidence_threshold"])
    return out


def cd_xargs(
    arm: str, payload: str, record_tag: str, mar: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """``CdRequest(...).to_xargs()`` for a constrained arm. Raises ``pydantic.ValidationError`` when the
    request is off-contract (e.g. a payload above the 1 MiB cap)."""
    s = spec(arm)
    if s.mode is None:
        raise ValueError("the unconstrained arm sends no astra_cd_* key")
    knobs = arm_knobs(arm, mar)
    req = CdRequest(
        mode=s.mode,
        self_conditioning=s.self_conditioning,
        automaton=payload,
        automaton_sha256=hashlib.sha256(payload.encode("ascii")).hexdigest(),
        record_tag=record_tag,
        entropy_bound=knobs.get("entropy_bound"),
        confidence_threshold=knobs.get("confidence_threshold"),
    )
    return req.to_xargs()


def row_knobs(xargs: Mapping[str, Any]) -> dict[str, Any]:
    """Every ``astra_cd_*`` value sent, minus the automaton itself (its sha256 stays)."""
    return {
        k[len(CD_XARGS_PREFIX) :]: v
        for k, v in xargs.items()
        if k.startswith(CD_XARGS_PREFIX) and k != CD_XARGS_PREFIX + "automaton"
    }
