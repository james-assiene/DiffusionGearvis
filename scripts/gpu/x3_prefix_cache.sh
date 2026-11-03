#!/usr/bin/env bash
# X3 — container sanity, part 2: identical image prompts and the prefix-cache hit ratio.
#
# Sends the SAME (image + text) prompt twice and reports the cache hit ratio of the SECOND
# request alone, as a delta against the counters read immediately before it. The naive
# "hits_total / queries_total over both requests" tops out near 0.5 because the first request
# is necessarily cold -- that denominator would make a perfect cache look like a failure.
#
# This is NOT B11. B11's regime (thresholds.py:26) is "prefill-then-generate" -- a conversational
# continuation. Two identical full prompts is strictly easier, and at ~300 prompt tokens the
# 16-token block rounding alone floors the ratio near 0.99, so this cannot distinguish "the cache
# is excellent" from "the cache merely functions". It is a works/doesn't-work check that borrows
# 0.9 as a convenient bar. Never cite its output as B11.
#
# Produces no artifacts/ writes. Latency is printed but is NOT a B1/B3 number: n=1, no seed
# control, warm torch.compile, and the second request is deliberately cache-advantaged.
#
# Lives under scripts/gpu/ per the convention T-0016 introduced: anything in this directory
# is GPU-spending by definition, whatever it is called, so the Bash guard demands a /launch
# token and a clean tree before it runs. It used to sit in scripts/, where it matched no
# trigger and ran unreviewed.
set -uo pipefail
API=http://127.0.0.1:8001
IMG=/tmp/x3_frame.png

fail() { printf '\033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
ok()   { printf '\033[32mok\033[0m   %s\n' "$*"; }
FAILED=0

echo "== regime (CLAUDE.md: state the whole regime beside every number)"
MODEL=$(curl -s $API/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$MODEL" ] || { echo "FAIL: /v1/models returned nothing - is the engine up on $API?"; exit 1; }
echo "   model         : $MODEL"
echo "   model_revision: 2ea837236295d617ac27f8c17d61228081932c40"
echo "   container     : vllm/vllm-openai:v0.29.0 (sha256:c291476...)"
echo "   engine booted : ${X3_ENGINE_COMMIT:-UNSET-pass X3_ENGINE_COMMIT=<sha>}   script commit: $(git rev-parse --short HEAD)"
echo "   KV            : bf16, 4 GiB pool, max_model_len 16384"
echo "   diffusion     : canvas 256, steps 48, max_num_seqs 3"
echo "   plugin        : ${X3_PLUGIN:-UNSET-pass X3_PLUGIN=ON|OFF}   generation-config: model default"
echo "   sampling      : max_tokens 64 only, n=1 (temperature/seed/min_p rejected 400 on the diffusion path; SAMPLING_WHITELIST in contracts/llm.py)"

# Falsifiable guard for the risk that matters most when the plugin overrides `architectures`:
# that the engine quietly stops taking the DIFFUSION code path and serves as a plain LM. Such a
# boot looks completely healthy, and every canvas/denoising assumption downstream would be wrong.
# sampling_params.py:1027 rejects `temperature` ONLY on the diffusion path, so a 400 here proves
# diffusion is live and a 200 proves it is not. The rejection happens before the engine, so this
# costs no GPU time and does not spend the single-use cold prefix cache.
echo "== diffusion-path guard: temperature must be REJECTED with 400"
PROBE=$(curl -s -o /tmp/x3_probe.json -w '%{http_code}' -X POST "$API/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"temperature\":0.7,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
if [ "$PROBE" = "400" ]; then
  ok "temperature rejected 400 - engine is on the diffusion path"
else
  fail "temperature returned $PROBE, expected 400 - THE ENGINE IS NOT ON THE DIFFUSION PATH; every number below is void"
fi

echo "== build a deterministic test frame (no network, no randomness)"
python3 - "$IMG" <<'PY'
import sys
from PIL import Image, ImageDraw
# A synthetic "screen": deterministic content, no RNG, so the two requests are byte-identical.
im = Image.new("RGB", (896, 896), (18, 20, 28))
d = ImageDraw.Draw(im)
for i in range(0, 896, 64):
    d.line([(0, i), (896, i)], fill=(30, 34, 46), width=1)
    d.line([(i, 0), (i, 896)], fill=(30, 34, 46), width=1)
d.rectangle([80, 80, 816, 300], fill=(36, 42, 58), outline=(90, 140, 200), width=3)
d.text((110, 120), "Traceback (most recent call last):", fill=(235, 238, 245))
d.text((110, 150), '  File "app.py", line 42, in handler', fill=(200, 205, 215))
d.text((110, 180), "    return payload['user']['id']", fill=(200, 205, 215))
d.text((110, 210), "KeyError: 'user'", fill=(240, 120, 120))
d.rectangle([80, 360, 816, 720], fill=(26, 30, 40), outline=(70, 80, 100), width=2)
for n, y in enumerate(range(400, 700, 40)):
    d.text((110, y), f"{n+1:>3}  def handler(payload):", fill=(150, 160, 180))
im.save(sys.argv[1])
print(f"   wrote {sys.argv[1]} ({im.size[0]}x{im.size[1]})")
PY

B64=$(base64 -w0 "$IMG")
PROMPT="What error is shown on this screen, and which line causes it?"

req() {  # $1 = label -> prints "ttft_ms total_ms ntok"
  local body start end
  body=$(python3 - "$B64" "$PROMPT" "$MODEL" <<'PY'
import json, sys
b64, prompt, model = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({
  # Only SAMPLING_WHITELIST fields (contracts/llm.py): top_k, top_p, max_tokens, stop.
  # temperature/seed/min_p/min_tokens/logit_bias/bad_words/allowed_token_ids are a hard 400
  # on the diffusion path -- confirmed empirically 2026-09-13, which is what the first run of
  # this script hit. Determinism is therefore not controllable here; the answers may differ.
  "model": model, "max_tokens": 64,
  "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
      {"type": "text", "text": prompt}]}]}))
PY
)
  start=$(date +%s%3N)
  resp=$(curl -s -X POST "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  end=$(date +%s%3N)
  echo "$resp" > "/tmp/x3_resp_$1.json"
  echo "$((end - start))"
}

counters() {  # prints "queries hits"
  curl -s $API/metrics | awk '
    /^vllm:prefix_cache_queries_total/ {q=$2}
    /^vllm:prefix_cache_hits_total/    {h=$2}
    END {printf "%d %d\n", q, h}'
}

echo "== request 1 (must be COLD - this is the control, and it is single-use per engine boot)"
read -r Q0 H0 <<<"$(counters)"
T1=$(req 1)
read -r Q1 H1 <<<"$(counters)"
DQ1=$((Q1-Q0)); DH1=$((H1-H0))
echo "   ${T1} ms wall · queried ${DQ1} tok, hit ${DH1} tok"
# Without a cold request 1, request 2's ratio says nothing about the cache: assert it.
if [ "$DQ1" -le 0 ]; then
  fail "request 1 queried ${DQ1} tokens - the engine served nothing; check /tmp/x3_resp_1.json"
elif [ $((DH1 * 20)) -gt "$DQ1" ]; then
  fail "request 1 was NOT cold (${DH1}/${DQ1} hit, >5%) - something warmed the cache; run is invalid"
else
  ok "request 1 cold (${DH1}/${DQ1} hit)"
fi
# Token count tells us whether the image was actually tokenized, or silently dropped.
if [ "$DQ1" -gt 0 ] && [ "$DQ1" -lt 200 ]; then
  fail "request 1 queried only ${DQ1} tok - too few for an image; it was probably dropped and only the text was tokenized"
fi

echo "== request 2 (identical prompt)"
T2=$(req 2)
read -r Q2 H2 <<<"$(counters)"
DQ=$((Q2-Q1)); DH=$((H2-H1))
echo "   ${T2} ms wall · queried ${DQ} tok, hit ${DH} tok"

echo "== result"
if [ "$DQ" -le 0 ]; then
  fail "request 2 queried ${DQ} tokens - cannot compute a ratio (denominator is zero)"
else
  RATIO=$(python3 -c "print(f'{$DH/$DQ:.4f}')")
  echo "   second-request prefix cache hit ratio: $RATIO  (${DH}/${DQ} tokens)"
  # NOT B11. B11's regime is prefill-then-generate (a conversational continuation); two identical
  # full prompts is strictly easier, and at this prompt length block rounding alone floors the
  # ratio near 0.99. This is a works/doesn't-work check, not a threshold measurement.
  python3 -c "import sys; sys.exit(0 if $DH/$DQ >= 0.9 else 1)" \
    && ok "X3-P2 cache-hit sanity >= 0.9 (NOT the B11 regime - do not cite this as B11)" \
    || fail "X3-P2 cache-hit sanity >= 0.9 (got $RATIO)"
fi

echo "== vision path (asserted, not just printed)"
python3 - <<'PY' || fail "vision path: no usable content, or the answer never mentions the on-screen error"
import json, sys
d = json.load(open('/tmp/x3_resp_1.json'))
if 'choices' not in d:
    print('   NO CHOICES:', json.dumps(d)[:400]); sys.exit(1)
m = (d.get('choices') or [{}])[0].get('message') or {}
c = m.get('content') or ''
r = m.get('reasoning_content') or ''
print('   content  :', repr(c[:400]))
print('   reasoning:', repr(r[:200]))
# The frame renders "KeyError: 'user'". If the model read the pixels it says so somewhere, but
# it may paraphrase ("a missing 'user' key", "Key Error"), so match on the normalised text and
# accept either the error name or the key. Both fields are printed above, so a FAIL here is
# adjudicable from the log without re-running -- which matters, because request 1's cold cache
# is single-use per engine boot.
t = (c + r).lower().replace(' ', '')
sys.exit(0 if ('keyerror' in t or "'user'" in t) else 1)
PY
echo "== identical answers?"
python3 -c "
import json
a=json.load(open('/tmp/x3_resp_1.json')); b=json.load(open('/tmp/x3_resp_2.json'))
ga=a.get('choices',[{}])[0].get('message',{}).get('content')
gb=b.get('choices',[{}])[0].get('message',{}).get('content')
print('   identical' if ga==gb else '   DIFFER (expected on a diffusion path; not a failure)')
"
echo "== latency (NOT a B1/B3 number: n=1, no seed, req 2 is cache-advantaged)"
echo "   req1 ${T1} ms · req2 ${T2} ms"
echo
[ "$FAILED" = 1 ] && { echo "X3 part 2: FAILED"; exit 1; }
echo "X3 part 2: passed (cache-hit sanity + vision path; the NvFp4 backend-line half of X3 was verified at boot)"
