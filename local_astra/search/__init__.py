"""Web search backends behind ``contracts.protocols.SearchBackend`` (T-0111b).

This package's ``__init__`` imports nothing and names no backend, so importing ``local_astra.search`` never loads
a web-search client; only the code that builds a backend imports its module (see ``docs/tasks/T-0111b.md``).
"""
