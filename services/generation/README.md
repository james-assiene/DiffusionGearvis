# services/generation

`astra fetch-models` (T-0007) copies the model's own `generation_config.json` here and applies our
overrides (sampler constants stay as published; `max_new_tokens` is nulled on the command line so
canvases are engine-global). Until that file exists, `services/vllm.sh` lets vLLM use the model's
file directly. Never hand-write sampler constants here (docs/10 §9).
