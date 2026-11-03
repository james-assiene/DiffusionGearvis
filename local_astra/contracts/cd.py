"""Constrained-decoding contracts (T-0064): the ``astra_cd_*`` request and the ``ASTRA_CD`` canvas record.

Wire format: ``docs/07-constrained-decoding.md`` §8. A constrained request travels in ``vllm_xargs``
(``contracts/llm.py``) as flat ``astra_cd_<field>`` keys with JSON scalar values; the compiled automaton
is the ``automaton`` string (``local_astra.decoding.artifact.encode_automaton``) and ``automaton_sha256``
is the lowercase hex sha256 of that string's ASCII bytes. The plugin parses it with its own numpy-only
decoder (``plugin/.../cd/xargs.py``); the container has no ``local_astra``.

``CdCanvasRecord`` is the per-canvas ``ASTRA_CD`` record the plugin emits (T-0067) and the evaluation reads
(T-0068, T-0070). Its field list is mirrored by ``cd/record_schema.py::RECORD_FIELDS`` in the plugin; a
contract test keeps the two equal.

Pydantic only: importing this module never pulls numpy, torch or ``local_astra.decoding``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from local_astra.contracts.base import Contract

#: Every key this contract emits starts with this prefix (``"astra_cd_" + field``).
CD_XARGS_PREFIX = "astra_cd_"
#: The encoded automaton string is capped at 1 MiB (T-0062 decision 4, provisional).
MAX_AUTOMATON_CHARS = 1 << 20
#: Gemma 4 vocabulary size; every token id in a record is below it.
VOCAB_SIZE = 262_144
#: The |S| buckets the plugin supports (T-0062 decision 5).
BUCKET_LADDER: tuple[int, ...] = (16, 32, 64, 128, 256, 512)

CdMode = Literal["end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax"]
SelfConditioning = Literal["model", "constrained"]
Bucket = Literal[16, 32, 64, 128, 256, 512]
TokenId = Annotated[int, Field(ge=0, lt=VOCAB_SIZE)]

_RECORD_TAG = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
#: A reason code: the plugin's ``CdRequestError.reason`` or a later runtime code (T-0067).
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _check_record_tag(v: str) -> str:
    if _RECORD_TAG.fullmatch(v) is None:
        raise ValueError("record_tag must be 1-64 characters of [A-Za-z0-9_.:-]")
    return v


class CdRequest(Contract):
    """One constrained-decoding request. ``to_xargs()`` is the only way requests are built (ruling W4)."""

    mode: CdMode
    self_conditioning: SelfConditioning = "model"
    automaton: str = Field(min_length=1, max_length=MAX_AUTOMATON_CHARS)
    automaton_sha256: str
    #: Join key between a response and its ``ASTRA_CD`` records (vLLM renames requests internally).
    record_tag: str
    entropy_bound: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    confidence_threshold: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    max_bucket: Bucket | None = None
    #: T-0081: the model's own draft, standard base64 of little-endian int32 ids (non-empty); seeds the first
    #: canvas. ``draft_noise`` / ``draft_start_step`` are only valid beside a draft (ruling S2).
    draft: str | None = None
    draft_noise: float | None = Field(default=None, ge=0.0, lt=1.0, allow_inf_nan=False)
    draft_start_step: Annotated[int, Field(strict=True, ge=0)] | None = None

    @field_validator("draft")
    @classmethod
    def _draft(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            raw = base64.b64decode(v.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError):
            raise ValueError("draft must be standard base64") from None
        if base64.b64encode(raw).decode("ascii") != v:
            raise ValueError("draft must be canonical standard base64")
        if not raw:
            raise ValueError("draft must hold at least one token id")
        if len(raw) % 4:
            raise ValueError("draft must be little-endian int32 ids (a multiple of 4 bytes)")
        ids = [int.from_bytes(raw[i : i + 4], "little", signed=True) for i in range(0, len(raw), 4)]
        if any(t < 0 or t >= VOCAB_SIZE for t in ids):
            raise ValueError(f"draft ids must be in [0, {VOCAB_SIZE})")
        return v

    @field_validator("automaton")
    @classmethod
    def _ascii(cls, v: str) -> str:
        if not v.isascii():
            raise ValueError("automaton must be an ASCII string")
        return v

    @field_validator("automaton_sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if _SHA256_HEX.fullmatch(v) is None:
            raise ValueError("automaton_sha256 must be 64 lowercase hex characters")
        return v

    @field_validator("record_tag")
    @classmethod
    def _tag(cls, v: str) -> str:
        return _check_record_tag(v)

    @model_validator(mode="after")
    def _consistent(self) -> CdRequest:
        if self.self_conditioning == "constrained" and not self.mode.startswith("perstep_"):
            raise ValueError("self_conditioning='constrained' needs a perstep_* mode")
        digest = hashlib.sha256(self.automaton.encode("ascii")).hexdigest()
        if digest != self.automaton_sha256:
            raise ValueError("automaton_sha256 is not the sha256 of the automaton string")
        if self.draft is None and (
            self.draft_noise is not None or self.draft_start_step is not None
        ):
            raise ValueError("draft_noise / draft_start_step need a draft")
        return self

    def to_xargs(self) -> dict[str, Any]:
        """``astra_cd_<field>`` for every field that is set; unset optionals are omitted, never null."""
        return {
            CD_XARGS_PREFIX + name: value
            for name, value in self.model_dump().items()
            if value is not None
        }


class CdCanvasRecord(Contract):
    """One ``ASTRA_CD`` record per committed canvas of a constrained request (T-0067 emits it).

    ``arm`` is the request's mode; T-0070 derives ``perstep_mar_sc`` from its own request, joined by
    ``record_tag`` (ruling W5). The ``end_projection`` shadow fields are null for the per-step arms, and
    ``mean_accepted_per_step`` is null for ``end_projection``. Every field is required: the emitter writes
    all of them, with null where a value does not apply. ``record_tag`` and ``arm`` are null only on an
    error record whose request carried no valid tag or mode (ruling W8).
    """

    record_tag: str | None
    arm: CdMode | None
    canvas_idx: int = Field(ge=0)
    #: the canvas's committed token ids, up to its valid length
    committed_ids: tuple[TokenId, ...]
    #: automaton states alive after this canvas
    state_set_after: tuple[Annotated[int, Field(ge=0)], ...]
    in_final: bool
    steps: int = Field(ge=0)
    bucket: Bucket | None
    #: position of the first stop token in ``committed_ids``; null when the canvas has none
    stop_pos: int | None = Field(ge=0)
    #: reason code, or null on success
    error: str | None
    mean_accepted_per_step: float | None = Field(ge=0.0, allow_inf_nan=False)
    shadow_argmax_ids: tuple[TokenId, ...] | None
    argmax_in_language: bool | None
    map_equals_argmax: bool | None

    @field_validator("record_tag")
    @classmethod
    def _tag(cls, v: str | None) -> str | None:
        return None if v is None else _check_record_tag(v)

    @field_validator("error")
    @classmethod
    def _reason(cls, v: str | None) -> str | None:
        if v is not None and _REASON.fullmatch(v) is None:
            raise ValueError(
                "error must be a reason code ([a-z][a-z0-9_]*, at most 64 chars) or null"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> CdCanvasRecord:
        if self.error is None and (self.record_tag is None or self.arm is None):
            raise ValueError("record_tag and arm are null only on an error record")
        if self.arm == "end_projection":
            if self.mean_accepted_per_step is not None:
                raise ValueError("mean_accepted_per_step is null for end_projection")
        else:
            for name in ("shadow_argmax_ids", "argmax_in_language", "map_equals_argmax"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} is only set for end_projection")
        if self.stop_pos is not None and self.stop_pos >= len(self.committed_ids):
            raise ValueError("stop_pos must index into committed_ids")
        if self.bucket is not None and any(s >= self.bucket for s in self.state_set_after):
            raise ValueError("state_set_after holds a state outside the bucket")
        return self
