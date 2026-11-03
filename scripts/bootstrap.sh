#!/usr/bin/env bash
# First run on a new machine (the 5090 box or a laptop). Idempotent. Never spends GPU time.
#   bash scripts/bootstrap.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0

echo "== toolchain"
if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version | awk '{print $2}')"; else
  fail "uv missing — install: curl -LsSf https://astral.sh/uv/install.sh | sh  (then re-run)"; fi
command -v git >/dev/null 2>&1 && ok "git $(git --version | awk '{print $3}')" || fail "git missing"
if [ "$FAILED" = 1 ]; then exit 1; fi

echo "== python environment (uv fetches a 3.11 interpreter if the box has none)"
if uv sync --all-extras -q; then ok "uv sync --all-extras"; else fail "uv sync failed"; exit 1; fi
ok "python $(uv run python -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"

echo "== git"
if [ -d .git ]; then
  ok "repository present ($(git rev-parse --short HEAD) on $(git branch --show-current))"
  if ! git config user.email >/dev/null; then
    warn "no git identity: git config user.name 'Your Name'; git config user.email you@example.com"
  fi
  # a tree copied file-by-file (no tarball) arrives without mode bits; restore the ones git recorded
  git ls-files -s | awk '$1=="100755"{print $4}' | while read -r f; do
    [ -x "$f" ] || { chmod +x "$f" && warn "restored executable bit on $f"; }
  done
  [ -z "$(git status --porcelain)" ] && ok "clean tree" || warn "dirty tree (fine on first run, but /launch needs it clean)"
else
  warn "not a git repository — run: git init -b main && git add -A && git commit -m 'scaffold'"
fi

echo "== claude code project config"
for f in .claude/settings.json .claude/hooks/_lib.sh .claude/hooks/guard_bash.sh .claude/hooks/protect_paths.sh; do
  [ -f "$f" ] && ok "$f" || fail "$f missing"
done
for f in .claude/hooks/*.sh; do [ -x "$f" ] || { chmod +x "$f" && warn "made $f executable"; }; done
# the guard must be able to parse hook JSON on this machine
if echo '{"cwd":"'"$PWD"'","agent_type":"implementer","tool_input":{"command":"uv run astra bench B3"}}' \
   | bash .claude/hooks/guard_bash.sh >/dev/null 2>&1; then
  fail "guard_bash.sh let an unapproved GPU command through"
else ok "hooks run and block an unapproved GPU command"; fi

echo "== fast suite + lint + types"
if make check >/tmp/astra_bootstrap_check.log 2>&1; then
  ok "make check ($(grep -oE '[0-9]+ passed.*' /tmp/astra_bootstrap_check.log | tail -1))"
else
  fail "make check failed — see /tmp/astra_bootstrap_check.log"; tail -20 /tmp/astra_bootstrap_check.log
fi

echo "== audio (exactly the call astra session makes)"
# P0-3: bootstrap used to print "ready" on a box where `astra session` exits 1, because it asked
# for pactl (absent here) and never tried the import the session actually makes. PortAudio is the
# verdict; pactl is a note.
if uv run python -c "import sounddevice; sounddevice.query_devices()" >/tmp/astra_bootstrap_audio.log 2>&1; then
  ok "sounddevice imports and query_devices() answers"
else
  fail "sounddevice/PortAudio unusable — astra session exits 1 here. Fix: apt install libportaudio2 (printed for you; this script never runs it)"
  tail -3 /tmp/astra_bootstrap_audio.log
fi
command -v pactl >/dev/null 2>&1 && ok "PipeWire/PulseAudio tools present" \
  || warn "no pactl (a note only: it says nothing about PortAudio, which is what the session needs)"

echo "== GPU box checks (warnings only; nothing here runs on the GPU)"
if command -v nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU: ${gpu:-?} · driver ${drv:-?}"
  case "${drv%%.*}" in ''|[0-9]|[0-9][0-9]|[0-4][0-9][0-9]|5[0-7][0-9]) warn "driver ≥ 580 required for the cu130 container (services/versions.lock)";; esac
else warn "nvidia-smi not found (fine on a laptop; the 5090 box needs driver ≥ 580)"; fi
if command -v docker >/dev/null 2>&1; then
  docker info 2>/dev/null | grep -qi nvidia && ok "docker with the NVIDIA runtime" || warn "docker present but no NVIDIA runtime visible (install nvidia-container-toolkit)"
  docker image inspect vllm/vllm-openai:v0.29.0 >/dev/null 2>&1 && ok "vllm/vllm-openai:v0.29.0 image present" \
    || warn "pull the pinned image when convenient: docker pull vllm/vllm-openai:v0.29.0 (about 31 GB, no GPU time)"
else warn "docker not found (needed for vLLM, Phase 0)"; fi

echo
if [ "$FAILED" = 1 ]; then echo "bootstrap: FAILED (see ✗ above)"; exit 1; fi
echo "bootstrap: ready. Next: the model downloads and images in the README's \"One-time setup\","
echo "then start the services with: uv run diffusion_gearvis up --profile lean (see the README)."
