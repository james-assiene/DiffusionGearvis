"""Keep torch's tracing loggers out of pytest's captured-log sections (T-0077c tier 2 at f3c678c/a671719: 4 950
``torch.__trace`` DEBUG lines pushed the assertion text of red rows out of the wrappers' windows).

``quiet_torch_logs()`` disables ``torch.__trace`` (torch attaches pytest's capture handlers to it directly and runs it at
DEBUG, so level and ``propagate`` are not enough) and holds ``torch._dynamo``/``torch._inductor``/``torch.fx`` at
WARNING; everything is restored on exit. Our own loggers are untouched. Import-light (no torch)."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

NOISY_LOGGERS = ("torch.__trace", "torch._dynamo", "torch._inductor", "torch.fx")


@contextlib.contextmanager
def quiet_torch_logs() -> Iterator[None]:
    saved = []
    for name in NOISY_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate, lg.disabled))
        lg.setLevel(logging.WARNING)
        if name == "torch.__trace":
            lg.propagate = False
            lg.disabled = True
    try:
        yield
    finally:
        for lg, level, prop, dis in saved:
            lg.setLevel(level)
            lg.propagate = prop
            lg.disabled = dis
