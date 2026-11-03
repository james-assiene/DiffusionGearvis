#!/usr/bin/env bash
# Read-only watcher for the cold-boot log (launch review 2026-09-17, T-0058 group C). No GPU spend.
cd "$(dirname "$0")/../.."
L=logs/vllm-coldboot-$(git rev-parse --short HEAD).log
for i in $(seq 1 120); do
  grep -q "refusing to start the engine" "$L" 2>/dev/null && { echo "STOP: the launcher's own guard refused"; sed -n '1,20p' "$L"; exit 1; }
  grep -q "less than desired GPU memory utilization" "$L" 2>/dev/null && { echo "STOP: PREDICTION FALSIFIED - the engine's own check failed at 0.86"; grep -n "Free memory on device" "$L"; exit 1; }
  grep -q "EngineCore failed to start\|EngineDeadError\|torch.OutOfMemoryError" "$L" 2>/dev/null && { echo "STOP: engine died - this is a result"; tail -40 "$L"; exit 1; }
  grep -q "Application startup complete" "$L" 2>/dev/null && { echo "UP"; grep -n "Initial free memory\|reserved .* for KV Cache\|Resolved architecture" "$L"; exit 0; }
  sleep 2
done
echo "TIMEOUT after 240 s; tail:"; tail -20 "$L"
