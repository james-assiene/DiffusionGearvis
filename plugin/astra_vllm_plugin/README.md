# astra-vllm-plugin

Out-of-tree vLLM plugin (`vllm.general_plugins` entry point) for DiffusionGemma on the pinned
`vllm/vllm-openai:v0.29.0` container. See `docs/07-constrained-decoding.md` §5 and
`docs/12-generative-ui-and-visualization.md` §2.

Milestones

0. **Passive per-step tap** (T-0011): after every denoising step, publish
   `(request_id, canvas_idx, step, argmax token ids, accepted bitmask, per-position entropy)`
   at ≤ 30 Hz per request over ZeroMQ PUB on `127.0.0.1:8002`. No change to sampling.
   Budget: < 3 % step-time overhead (B16). Entropy is computed from the same logits the sampler
   sees, masked positions use a finite sentinel (≤ −1e20, never `-inf` → NaN).
1. `length_bounded(N)` and closing-tag automata (envelope / `</ui>`), coexisting with
   unconstrained requests in the same engine (constraint selected per request via `vllm_xargs`).
2. Full FA-guided sampler port of `diffgemma-fa` (max-plus MAP first, |S| ≤ 128).

How it hooks in (verified against v0.29.0 source before the first run — X-plugin-0):
`register()` registers a *new architecture name* (`AstraDiffusionGemmaForCausalLM`) whose model
class subclasses the stock DiffusionGemma model and overrides `get_model_state_cls()` to return a
`ModelState` subclass whose `custom_sampler(sampler)` wraps the stock sampler. The server is
started with `--hf-overrides '{"architectures": ["AstraDiffusionGemmaForCausalLM"]}'`. Never add
a `vllm.logits_processors` entry point: it forces the V1 runner, which rejects diffusion.

Install into the container:

    docker run ... -v $PWD/plugin/astra_vllm_plugin:/plugin ... \
      --entrypoint bash vllm/vllm-openai:v0.29.0 -c "pip install /plugin && vllm serve ..."

`services/vllm.sh` does this and uses `--network host` so the PUB socket is reachable on the
host's loopback.
