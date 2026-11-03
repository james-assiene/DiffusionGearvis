"""Triton kernels of the per-step CD core (T-0077b onwards; ``docs/tasks/T-0077.md``, ``T-0077b.md``).

Nothing heavy is imported here: neither torch nor triton. ``backend`` decides, once per call, whether a kernel
runs (a CUDA tensor, an importable triton, and the caller's ``kernels`` keyword not False) and logs the first
decision of the process; the kernel modules (``entropy``) import triton only inside function bodies (ruling M1),
so every ``cd`` module imports on a host without triton and the CPU path never touches it.
"""
