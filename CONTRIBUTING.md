# Contributing to DiffusionGearvis

DiffusionGearvis is a research prototype maintained by one person. Issues are welcome; pull requests are read, but may not be reviewed or merged quickly.

## Reporting a problem

Open an issue with:

- what you ran (the exact command and flags) and what you expected;
- what happened, with the console output;
- the output of `uv run diffusion_gearvis doctor`;
- your GPU, driver version and Linux distribution.

Check the README's "When something is wrong" table first; most setup problems are listed there. Please do not attach session logs (`logs/sessions/*.jsonl`) without reading them: they hold what you said and typed in the session, and the model's requests.

For a security problem, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Working on the code

```bash
bash scripts/bootstrap.sh                  # toolchain, uv sync --all-extras, the test suite
make check                                 # lint, types, the fast test suite (no GPU)
make fast-parallel                         # the fast suite in parallel (about 2 minutes)
```

- Python 3.11, `uv`, `ruff`, `mypy --strict` on `local_astra/`.
- The fast suite runs without a GPU or the network. Tests marked `gpu` need the RTX 5090 services; they never run by accident (`-m 'not gpu'` is the default).
- The browser suites (`make browser`, `make browser-live` and others in the Makefile) need the agent's Chromium (`uv run playwright install chromium`) and run on loopback only.
- Services bind to 127.0.0.1 only. A pull request must not add network calls; the only paths out are web search (`local_astra/search/`) and the agent's browser in web mode, both fenced by tests.
- Wire formats are in `docs/02-architecture.md` §5; changing one means changing the fake in `tests/fakes/` in the same change.
- Latency, VRAM and duplex budgets are tests (`local_astra/bench/thresholds.py`); please do not raise a threshold without saying why.

## How changes are made here

Work on this repository is done by a planner → test-author → implementer → reviewer → verifier loop of AI agents, enforced by hooks in `.claude/` and described in `docs/09-agent-workflow.md`, with one task file per change in `docs/tasks/`. You do not need to follow that process for a pull request, but a change that comes with tests written from its acceptance criteria, and that keeps `make check` green, is much easier to accept.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, the licence of this repository (see `LICENSE`).
