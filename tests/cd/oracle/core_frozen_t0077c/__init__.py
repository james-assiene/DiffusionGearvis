"""Frozen branch-base copies of the per-step CD core at b53457a (T-0077c; see ``MANIFEST.md``).

``semiring``, ``tree``, ``emission``, ``marginals``, ``mapdecode``, ``draw`` and ``kernels/{__init__,backend,entropy}``
are byte-identical to ``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py`` at b53457a (the T-0077b merge, the
base of ``T-0077c-cd-logmatmul-kernel``). Their relative imports (``from .semiring import ...``, ``from .kernels import
backend``) resolve inside this package, so they import side by side with the live ``astra_vllm_plugin.cd.*`` modules
and never touch them. Never edited.
"""
