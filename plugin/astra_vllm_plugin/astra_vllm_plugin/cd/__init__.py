"""Constrained decoding inside the engine (T-0062 tree).

Keep this file empty of imports: the wire modules (``automaton``, ``xargs``, ``record_schema``) are
stdlib + numpy only and are imported on the request path, while the torch-based core (``emission``,
``semiring``, ``tree``, ``marginals``, ``budget``) is imported only by the sampler.
"""
