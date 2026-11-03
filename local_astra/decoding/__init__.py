"""Constrained decoding on the host (T-0062 tree): the grammar compiler and its reports.

Needs the non-default ``cd`` dependency group (torch, outlines-core, tokenizers). Nothing in the
runtime (``local_astra``, ``local_astra.cli``, ``local_astra.conductor``) imports this package, and it
imports nothing heavy at package level; import the submodules (``grammars``, ``report``, ``vocab``).
"""
