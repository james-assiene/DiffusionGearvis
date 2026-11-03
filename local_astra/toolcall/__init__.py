"""Tool-call mode, runtime-safe half (T-0080a): v1 tool schemas, the router and the call scorer.

Pure Python plus pydantic/jsonschema, importable by the conductor. Never imports
``local_astra.decoding``, torch, numpy or outlines; the grammars live in ``local_astra.decoding.toolcall``
(the ``cd`` group). Import the submodules: ``tools``, ``router``, ``score``.
"""
