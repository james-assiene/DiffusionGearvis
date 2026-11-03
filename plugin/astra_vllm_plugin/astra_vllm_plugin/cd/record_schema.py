"""The ``ASTRA_CD`` canvas record's field list (T-0064 criterion 6), stdlib only.

Mirror of ``local_astra.contracts.cd.CdCanvasRecord`` (the container has no ``local_astra``); a contract
test asserts the two field sets are equal. The emitter (T-0067) writes every field, null where a value
does not apply.
"""

from __future__ import annotations

RECORD_FIELDS = (
    "record_tag",
    "arm",
    "canvas_idx",
    "committed_ids",
    "state_set_after",
    "in_final",
    "steps",
    "bucket",
    "stop_pos",
    "error",
    "mean_accepted_per_step",
    "shadow_argmax_ids",
    "argmax_in_language",
    "map_equals_argmax",
)

#: Fields that are non-null only when ``arm == "end_projection"``.
END_PROJECTION_FIELDS = ("shadow_argmax_ids", "argmax_in_language", "map_equals_argmax")
