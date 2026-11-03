"""Generative-UI channel: A2UI v0.9.1 validation (docs/12 §1)."""

from local_astra.ui.a2ui_schema import (
    A2UI_SPEC_VERSION,
    CARDS_V2_CATALOG_ID,
    MESSAGE_KINDS,
    A2UIValidator,
    SurfaceReport,
    TopologyReport,
    parse_concatenated_json,
)

__all__ = [
    "A2UI_SPEC_VERSION",
    "CARDS_V2_CATALOG_ID",
    "MESSAGE_KINDS",
    "A2UIValidator",
    "SurfaceReport",
    "TopologyReport",
    "parse_concatenated_json",
]
