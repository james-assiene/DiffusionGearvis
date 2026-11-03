"""Frozen branch-base copies of the per-step CD core at 2d90189 (T-0077b; see ``MANIFEST.md``).

``semiring``, ``tree``, ``emission``, ``marginals``, ``mapdecode`` and ``draw`` are byte-identical to
``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py`` at 2d90189 (the T-0077a merge, the base of
``T-0077b-cd-hq-kernel``). Their relative imports (``from .semiring import ...``) resolve inside this package, so
they import side by side with the live ``astra_vllm_plugin.cd.*`` modules and never touch them. Never edited.
"""
