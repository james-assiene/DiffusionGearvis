"""Build a minimal valid instance of any Contract model from its field annotations.

Used by the round-trip tests so that *every* event/command/message type is exercised without
hand-writing an example per model (adding a model automatically adds a test case).
"""

from __future__ import annotations

import types
import typing
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

_SAMPLE: dict[type, Any] = {str: "x", int: 1, float: 0.5, bool: True, bytes: b"\x00\x01"}


def sample_for(annotation: Any, *, field_name: str = "") -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Annotated:
        return sample_for(args[0], field_name=field_name)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return sample_for(non_none[0], field_name=field_name)
    if origin is typing.Literal:
        return args[0]
    if origin in (tuple, list):
        inner = args[0] if args else str
        if args and len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
        val = sample_for(inner, field_name=field_name)
        return (val,) if origin is tuple else [val]
    if origin is dict:
        return {}
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return example_of(annotation)
        if issubclass(annotation, Enum):
            return next(iter(annotation))
        if annotation in _SAMPLE:
            if annotation is int and field_name == "step":
                return 0  # AstraCanvasStep: step < max_steps
            if annotation is str and field_name in ("accepted", "stable"):
                return "ff00"  # AstraCanvasStep: hex, whole bytes, >= canvas_length bits
            if annotation is str and field_name == "phash":
                return "0f0f0f0f0f0f0f0f"
            if annotation is str and field_name == "type":
                return "session.update"
            return _SAMPLE[annotation]
    raise TypeError(f"no sample for {annotation!r}")


def example_of(model: type[BaseModel]) -> BaseModel:
    kwargs: dict[str, Any] = {}
    for name, f in model.model_fields.items():
        if f.default is not PydanticUndefined or f.default_factory is not None:
            continue
        kwargs[name] = sample_for(f.annotation, field_name=name)
    if model.__name__ == "AstraCanvasStep" and kwargs.get("phase", "denoise") == "denoise":
        # T-0011 R10: a denoise frame carries all three rows, which are optional (a commit frame
        # carries none), so "required fields only" is not a valid frame by itself
        n = int(kwargs.get("canvas_length", 1))
        kwargs.setdefault("canvas", (1,) * n)
        kwargs.setdefault("argmax", (1,) * n)
        kwargs.setdefault("stable", "00" * ((n + 7) // 8))
    if model.__name__ == "AstraThought":
        # T-0180 A1: step >= 1, and omitted_chars > 0 only with text at the cap
        kwargs["step"] = 1
        kwargs["omitted_chars"] = 0
    if model.__name__ == "AstraVisionFrame":
        # T-0177 A1 (A-P3): a vision frame's tool-loop step is >= 1; ``sample_for`` gives a field named
        # ``step`` 0 for AstraCanvasStep, which is not a valid step here
        kwargs["step"] = 1
    return model(**kwargs)
