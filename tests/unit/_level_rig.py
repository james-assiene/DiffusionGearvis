"""The seam T-0047 group A0's tests assume, in one place (as ``_box_rig.py`` does for T-0033).

The implementer owns the *shape* of ``local_astra/reflex/level.py``; what the rows below fix is
the public surface they call, taken from ``docs/tasks/T-0047.md`` criteria 1-6::

    SILENCE_DBFS = -100.0            # criterion 1: digital silence, never -inf, never nan
    NOISE_FLOOR_WINDOW_MS = 3000     # criterion 3
    STT_FRAME_MS = 80                # the pump's frame (audio_capture.STT_BLOCK_SAMPLES = 1920
                                     # at 24 kHz): a fact of the capture path, not a new choice
    NOISE_FLOOR_PERCENTILE = 0.2     # criterion 3
    VOICED_MARGIN_DB = 9.0           # criterion 3
    LEVEL_SPAN_DB = 30.0             # criterion 3
    PUBLISH_INTERVAL_MS = 100        # criterion 4

    rms_dbfs(samples: np.ndarray) -> float
    LevelMeter().observe(rms_db: float, t_ms: float) -> ListenLevel | None
        ListenLevel carries at least .level .rms_dbfs .noise_floor_dbfs .voiced .t_ms

Every lookup goes through :func:`attr`, which raises ``AssertionError`` naming the criterion, so
a missing symbol is the documented "not implemented yet" outcome of ``tests/conftest.py``'s
pending machinery (``xfail(strict=True, raises=(NotImplementedError, AssertionError))``) and not
a collection error that would take the whole module down.

Nothing here computes a level, a floor or a dB value: the rig only looks symbols up and drives
``observe``. Expected values are hand-computed in the tests that use it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

MODULE = "local_astra.reflex.level"

#: Criterion 2, hand-computed: ceil(NOISE_FLOOR_WINDOW_MS / STT_FRAME_MS) = ceil(3000 / 80) = 38.
WINDOW_FRAMES = 38

#: Criterion 4, hand-computed: 100 ms of frame time on an 80 ms grid is every second frame.
EMISSION_GAP_MS = 160.0

#: The five fields criterion 3 puts on the record (and criterion 7 later puts on the wire).
LEVEL_FIELDS = ("level", "rms_dbfs", "noise_floor_dbfs", "voiced", "t_ms")

_MISSING = object()


def module() -> Any:
    """Import ``local_astra.reflex.level``, or say what is missing as an AssertionError."""
    try:
        return importlib.import_module(MODULE)
    except ImportError as exc:  # pragma: no cover - the pending path
        raise AssertionError(
            f"T-0047 group A0: {MODULE} does not import ({exc}); "
            f"the seam it must offer is in tests/unit/_level_rig.py's docstring"
        ) from None


def attr(name: str, criterion: str) -> Any:
    mod = module()
    value = getattr(mod, name, _MISSING)
    assert value is not _MISSING, (
        f"T-0047 {criterion}: {MODULE}.{name} is missing "
        f"(the seam is in tests/unit/_level_rig.py's docstring)"
    )
    return value


def const(name: str, criterion: str) -> Any:
    """A module-level constant, by name (criterion 3: every number is a named constant)."""
    return attr(name, criterion)


def rms_dbfs(samples: Any) -> float:
    value = attr("rms_dbfs", "criterion 1")(samples)
    assert isinstance(value, float), f"criterion 1: rms_dbfs returns a float, got {type(value)!r}"
    return value


def meter() -> Any:
    return attr("LevelMeter", "criterion 2")()


def field(emission: Any, name: str) -> Any:
    value = getattr(emission, name, _MISSING)
    assert value is not _MISSING, (
        f"T-0047 criterion 3: the record returned by LevelMeter.observe has no .{name} "
        f"(it must carry {', '.join(LEVEL_FIELDS)})"
    )
    return value


@dataclass(frozen=True)
class Emission:
    """One value returned by ``observe``, with the index and time of the frame that produced it."""

    index: int
    frame_t_ms: float
    record: Any

    @property
    def level(self) -> float:
        return float(field(self.record, "level"))

    @property
    def rms_dbfs(self) -> float:
        return float(field(self.record, "rms_dbfs"))

    @property
    def noise_floor_dbfs(self) -> float:
        return float(field(self.record, "noise_floor_dbfs"))

    @property
    def voiced(self) -> bool:
        return bool(field(self.record, "voiced"))

    @property
    def t_ms(self) -> float:
        return float(field(self.record, "t_ms"))


def frame_times(count: int, *, dt_ms: float = 80.0, t0_ms: float = 0.0) -> list[float]:
    return [t0_ms + i * dt_ms for i in range(count)]


def drive(
    values: list[float],
    *,
    dt_ms: float = 80.0,
    t0_ms: float = 0.0,
    times: list[float] | None = None,
    inst: Any | None = None,
) -> tuple[Any, list[Emission]]:
    """Feed ``values`` to a ``LevelMeter`` on the frame grid; return it and what it emitted."""
    times = times if times is not None else frame_times(len(values), dt_ms=dt_ms, t0_ms=t0_ms)
    assert len(times) == len(values), "the rig was handed mismatched values and times"
    inst = inst if inst is not None else meter()
    out: list[Emission] = []
    for i, (value, t_ms) in enumerate(zip(values, times, strict=True)):
        record = inst.observe(value, t_ms)
        if record is not None:
            out.append(Emission(index=i, frame_t_ms=t_ms, record=record))
    return inst, out


def window_of(values: list[float], index: int) -> list[float]:
    """The frames criterion 2 says are in the window when frame ``index`` is observed.

    Read off the criterion, not off the implementation: the last ``WINDOW_FRAMES`` frames of
    frame time, the current one included.
    """
    lo = max(0, index + 1 - WINDOW_FRAMES)
    return values[lo : index + 1]


def instance_attrs(obj: Any) -> dict[str, Any]:
    """Every instance attribute of ``obj``, whether it uses ``__dict__`` or ``__slots__``."""
    out: dict[str, Any] = {}
    out.update(dict(getattr(obj, "__dict__", {}) or {}))
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()) or ():
            if hasattr(obj, name):
                out[name] = getattr(obj, name)
    return out


def source() -> str:
    """The text of ``level.py`` (the purity and docstring rows read it, not its behaviour)."""
    import pathlib

    path = getattr(module(), "__file__", None)
    assert path, f"T-0047: {MODULE} has no __file__ to read"
    return pathlib.Path(path).read_text(encoding="utf-8")
