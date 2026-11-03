"""Perception contracts: sources, assist flags (pixel-first), frames (docs/12 §3, docs/01 D6/D12a)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from local_astra.contracts.base import Contract


class Source(StrEnum):
    SCREEN = "screen"
    WINDOW = "window"
    CAMERA = "camera"


class AssistFlags(Contract):
    """Non-pixel assists. All off by default; the HUD checkboxes toggle them; every turn logs them.

    The model sees pixels first. These exist to *compare*, not to carry the demo (docs/12 §3).
    """

    page_text: bool = False  # DOM / page text of the shared browser tab
    pdf_text: bool = False  # PDF text layer
    cells: bool = False  # spreadsheet cells
    at_spi: bool = False  # accessibility tree

    def any_on(self) -> bool:
        return self.page_text or self.pdf_text or self.cells or self.at_spi

    def badge(self) -> str:
        """Short label logged with every turn and shown in the HUD, e.g. ``pixels`` or ``pixels+dom``."""
        on = [
            n
            for n, v in (
                ("dom", self.page_text),
                ("pdf", self.pdf_text),
                ("cells", self.cells),
                ("a11y", self.at_spi),
            )
            if v
        ]
        return "pixels" + ("+" + "+".join(on) if on else "")


class Frame(Contract):
    """One captured frame. ``jpeg`` is the encoded image after the resolution policy."""

    source: Source
    ts_ms: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    jpeg: bytes
    phash: str = Field(min_length=1)
    title: str | None = None
    app: str | None = None
    #: optional focus crop (active window / pointer region) — pixel-only mitigation for dense docs
    crop_of: str | None = Field(default=None, description="phash of the full frame this crops")


class FrameText(Contract):
    """Assist payload; only produced when the matching flag is on."""

    source: Source
    ts_ms: int = Field(ge=0)
    kind: str = Field(pattern="^(page_text|pdf_text|cells|at_spi)$")
    text: str
    url: str | None = None


class ResolutionBucket(Contract):
    name: str
    max_long_side: int = Field(ge=64)
    max_short_side: int = Field(ge=64)
    #: visual-token estimate — replaced by measurement in Phase 0 (Q2); ``measured`` says which
    est_visual_tokens: int = Field(ge=1)
    measured: bool = False
