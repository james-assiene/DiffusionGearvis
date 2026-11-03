# scripts/demo/

Interactive demos, driven by hand: they spend GPU time on services that are already running.
Deliberately **not** gated by `.claude/hooks/guard_bash.sh` (T-0016 ruling D5) because they
produce no number anyone will quote; anything that measures belongs in `scripts/gpu/` instead.
