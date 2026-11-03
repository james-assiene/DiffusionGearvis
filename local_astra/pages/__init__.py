"""Astra's loopback start pages (T-0111g A4-A5): the start page, the catalogue search and three demo pages, served by
``server.StartPageServer`` on its own ``127.0.0.1`` origin for the agent browser of ``tools = live``.

The pages under ``static/`` are self-contained (no external URL, web font or remote image), deterministic at
1280x800, and expose ``window.__state`` for tests and the GPU smoke. This package imports nothing from
``local_astra.computer`` or ``local_astra.conductor``.
"""

from __future__ import annotations

from local_astra.pages.server import STATIC_DIR, StartPageServer

__all__ = ["STATIC_DIR", "StartPageServer"]
