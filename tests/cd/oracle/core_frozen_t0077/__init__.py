"""Frozen pre-change copies of the per-step CD core at e73cfa7 (T-0077a; see ``MANIFEST.md``).

``semiring``, ``tree``, ``emission``, ``marginals``, ``mapdecode`` and ``draw`` are byte-identical to
``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py`` at e73cfa7. Their relative imports
(``from .semiring import ...``) resolve inside this package, so they import side by side with the live
``astra_vllm_plugin.cd.*`` modules and never touch them. Never edited.
"""
