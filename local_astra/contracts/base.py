"""Base class shared by every contract model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Immutable, strict model: unknown fields are errors, instances are hashable."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
