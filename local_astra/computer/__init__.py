"""Computer control (T-0111c): a live ``ToolExecutor`` over a sandboxed Playwright Chromium on loopback pages.

Importing this package never loads ``playwright`` (``runtime.py`` imports it at launch). T-0111g adds the options the
session's ``tools = live`` mode needs (``headless``, ``end_when_closed``, ``allowed_origins``); ``cli.py`` is the one
composition root that constructs the executor, and the session reaches it only through a protocol-typed seam.
"""

from __future__ import annotations

from local_astra.computer.executor import BrowserClosedError, BrowserToolExecutor, Driver

__all__ = ["BrowserClosedError", "BrowserToolExecutor", "Driver"]
