# What the live stream carries for a native tool call (T-0087)

This page reports one run of `scripts/gpu/t0080d_native_stream_probe.py` at commit `724c0c4`. Every number here comes from
`logs/t0087-probe-724c0c4/report.json` (sha256 `7f1d2048…`), its `report.md`, and `raw.jsonl` (sha256 `f871d037…`). The regime lines
for the device and the boot come from `logs/t0087-probe-724c0c4-run.log`. Nothing was re-run to write this page.

**Result: reading B.** The run was complete, with 0 transport errors.

**Central finding.** The text of a native call (`<|tool_call>…<tool_call|>`) never reaches `content`, in any arm. That includes the
non-stream arm (d) and every arm with `skip_special_tokens: false`. Arm (a), with the default `skip_special_tokens`, never has `call:`
in its content either, so the drop is not the detokeniser stripping special tokens.

**The rule that fits.** On all 48 rows of arms (b)–(e), 40 tool and 8 no-tool, `content` equals the answer region of the host decode cut
at its first special piece. That piece is `<|tool_call>`, or a stray special immediately before it:
- **A stray `<channel|>`** comes before the call in only some tool rows: (b) 8/10, (c) 8/10, (d) 7/10 and (e) 3/10. Where it is
  present, content stops there.
- **The rows without one are cut too.** In (b)/(c) search-01 the call comes straight after the thought, so content is `''`. Arm (e)
  key-02 is cut at a stray `<tool_call|>`.
- **No-tool rows.** These have no special piece in the answer region, so content is the whole region. That is q4's 2/2.

So the stray `<channel|>` explains where content stops in some rows, but it is not the mechanism.

**The call is in the ids, and nowhere else.** Every one of the 40 tool responses in (b)–(e) contains ids 48 (`<|tool_call>`) and 49
(`<tool_call|>`), and ends in 50 (`<|tool_response>`). The call text is in none of the served fields:
- not in `content`;
- not in `reasoning` (0/40);
- not in any `tool_calls` key. The stream deltas have only `content`, `reasoning` and `role`. The arm-(d) message has only
  `annotations`, `audio`, `content`, `function_call`, `reasoning`, `refusal` and `role`.

What *does* reach content matches the host decode byte for byte.

**Example: click-01, arm (d)** (non-stream, `skip_special_tokens: false`, `return_token_ids: true`). The served `content` is:

```
I will click the Sign in button.
```

The host decode of the same response's 99 ids (`replay.detokenize`), after the thought, is:

```
<channel|>I will click the Sign in button.<channel|><|tool_call>call:click_at{x:488,y:518}<tool_call|><|tool_response>
```

- **Same text in every arm.** Arms (a), (b), (c) and (e) serve the same `content` string for click-01. Each is a separate sample.
- **Ids.** For (b), (c) and (e), each sample's own ids carry the same `<channel|><|tool_call>call:click_at{…}<tool_call|>` tail, which
  is not in its text. Arm (a) asks for no ids.

**Likely cause, not established.** The engine runs with `--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4`
(`services/vllm.sh`), and the probe sends `tools` with `tool_choice: "none"`.
- **The reasoning parser is ruled out as the place the call went.** The call never appears in `reasoning`.
- **The tool-call parser is an inference.** It is the likeliest remaining explanation: the gemma4 tool-call parser (enabled by
  `--enable-auto-tool-choice`) consumes the call region and, under `tool_choice: "none"`, emits it nowhere. The probe does not show it.
- **What would settle it:** no arm ran without `--enable-auto-tool-choice` or `--tool-call-parser`, and none ran with another
  `tool_choice`. One run with the tool parser off would settle it.

## Regime

| | |
|---|---|
| commit | `724c0c47e311302c2ab3020155e7fbdb264322e6` (clean tree, `/launch`) |
| engine | vLLM 0.29.0 (`/version`); versions.lock `5bab0570…`; `--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4` |
| model | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `2ea8372` |
| tokenizer, template | tokenizer `cc8d3a0c…`; chat template pinned in versions.lock |
| eval set | cases `460cd1c3…` (dev split); tools_v1 `bbfc53cc…` |
| device | RTX 5090, driver 595.84; gpu-mem 0.86; engine only (no STT) on a freshly booted engine (P16) |
| diffusion | canvas length (CL) 256; 48 steps; thinking on; prefix caching on |
| requests | 12 cases × 5 arms, n = 1 each; no seed (the engine accepts none); concurrency 1; `max_tokens` 1024 |
| auxiliary | q8: 3 + 3 pairs; q9: 3 × 2 continuations; q10: 10 `/tokenize` calls |
| order | q8 (with tools, then control), then arms a..e per case, then q9, then q10 |
| timing | the box was shared with CPU test runs, so the recorded `wall_ms` is **not** a latency measurement and is not reported |

The arms are applied to `twopass.native_pass1_body` (system prompt plus `tools`, and `tool_choice: "none"`):

| arm | fields |
|---|---|
| (a) | `stream: true` (today's `llm_client`) |
| (b) | `stream: true`, `skip_special_tokens: false`, `return_token_ids: true` |
| (c) | (b) plus `spaces_between_special_tokens: false` |
| (d) | (b) but `stream: false` |
| (e) | the same request as (b), as a replicate |

## Results q1–q10

Every row carries the regime above: vLLM 0.29.0, NVFP4 rev 2ea8372, RTX 5090 with driver 595.84, gpu-mem 0.86, CL 256, 48 steps,
n = 1 and no seed.

| q | what | (a) | (b) | (c) | (d) | (e) |
|---|---|---|---|---|---|---|
| q1 | HTTP 200 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| q2 | stream complete (`[DONE]`, non-null `finish_reason`, no in-band error); for (d), a 200 with a choice | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| q3 | `reasoning` equals the thought region of the arm's own ids, and is not in `content` | no ids (P7)¹ | 12/12 | 12/12 | 12/12 | 12/12 |
| q4 | `content` byte-identical to the answer region of the own ids: tool cases | no ids (P7)¹ | **0/10** | **0/10** | **0/10** | **0/10** |
| q4 | the same, no-tool cases | no ids (P7)¹ | 2/2 | 2/2 | 2/2 | 2/2 |
| q5 | `finish_reason` | `stop` 12/12 | `stop` 12/12 | `stop` 12/12 | `stop` 12/12 | `stop` 12/12 |
| q5 | `stop_reason`: tool cases / no-tool cases | 50 / `None` | 50 / 106 | 50 / 106 | 50 / 106 | 50 / 106 |
| q5 | id 50 in the ids (tool cases); `<|tool_response>` text in content (all cases) | no ids; 0/12 | 10/10; 0/12 | 10/10; 0/12 | 10/10; 0/12 | 10/10; 0/12 |
| q6 | content chunks per canvas² | ≤ 1 | ≤ 1 | ≤ 1 | – | ≤ 1 |
| q6 | special pieces split across chunks | 0 | 0 | 0 | – | 0 |
| q7 | per-chunk `token_ids` present | – | 12/12 | 12/12 | – | 12/12 |
| field | reasoning delta/message field name | `reasoning` | `reasoning` | `reasoning` | `reasoning` | `reasoning` |

1. Arm (a) carries no ids. Its comparison with arm (d)'s decode of the same case is labelled noise-confounded and decides nothing: q3
   1/12, q4 1/12.
2. Every response fits one 256-token canvas, and the stream sends a role chunk followed by **one** chunk carrying the whole canvas
   (reasoning, content, `finish_reason`, `stop_reason` and `token_ids` together). The two search cases have empty content, so they have
   zero content chunks. A zero split count therefore says nothing about splits within a canvas, and with no special piece in any
   content, q6 is vacuous here.

The q3 fallback (`q3_stripped`) also holds on all (b)–(e) rows. Replicates: arm (b)'s ids equal arm (e)'s on 0/12 cases. This is the
noise level of the unseeded sampler, and it decides nothing.

**q8, prefix cache** (the `/metrics` delta over the generate request only, after a 1-token prefill):

| | case | hits / queries | rate |
|---|---|---|---|
| tools in the prefill | click-01 | 1056 / 1070 | 0.987 |
| | type-02 | 1056 / 1082 | 0.976 |
| | key-01 | 1056 / 1071 | 0.986 |
| control: prefill without tools, generate with tools (P8: other cases) | click-05 | 768 / 1070 | 0.718 |
| | scroll-01 | 768 / 1074 | 0.715 |
| | type-05 | 768 / 1088 | 0.706 |

The control hits the system-plus-tools prefix that earlier requests cached, and misses from the image onwards. That partial hit is
expected and report-only (P8).

**q9, T4 continuation** (non-stream; assistant content `spoken_prefix + "<|tool_call>"`; `continue_final_message: true`):

| form | search-01 | search-02 | click-01 |
|---|---|---|---|
| plain string | 200, prompt ends in 48 | 200, prompt ends in 48 | 200, prompt ends in 48 |
| one `[{"type":"text"}]` part | 200, prompt ends in 48 | 200, prompt ends in 48 | 200, prompt ends in 48 |

**q10, `/tokenize` parity.**
- Request: `POST /tokenize` at the server root, with body keys `model`, `prompt` and `add_special_tokens` (false). The prompt is
  `draft.rstrip()` of `router.route` on each tool case's arm-(d) host decode.
- Result: **10/10 equal** to `vocab.encode(draft.rstrip())`, every call status 200, at lengths 16, 16, 19, 19, 38, 33, 11, 11, 15 and 15
  ids.

## Reading (criterion 6, as pre-registered)

- **Reading B.**
  - Neither (b) nor (c) qualifies: both fail q4 on all 10 tool cases (`why.arm_b` and `why.arm_c`).
  - Status, q2, q3, q6 and the completeness facts hold on both arms. Only q4 fails.
  - The failure is not dropped specials or spacing. The call region is missing from content altogether.
- **Fallback: H10.** `streamed_ids` is true, because (b), (c) and (e) carry per-chunk ids on all 12 cases. So T-0088 is replanned on the
  streamed token ids with host detokenisation (H10), before its test-author starts. This is not an escalation: the ids carry the
  complete call on every tool response.
- **Stop id: replan T-0091.**
  - The replan is procedural: under reading B no arm is chosen, and P3/P9 then give "replan".
  - It is not caused by the stop evidence. `stop_reason` was 50 (an int in {1, 106, 50}) on every tool case of every arm, (a)
    included, with `finish_reason` `stop`.
  - T-0091's replan can start from that.
- **Tools in the prefill: required.** With tools in the prefill, all three cases reach at least 0.9 (0.976–0.987). Without them, the
  control reaches 0.71.
- **Continuation content: plain string.** All three string continuations returned 200 with the prompt ending in id 48. The list form also
  passed, but is not chosen (P5).
- **Seed tokenizer: the engine's `/tokenize`.** 10/10 equal, with every call status 200.
- **Complete: true.** There were 0 transport errors.

## SSE excerpts (arm (b), anonymised)

How these were anonymised:
- `id` and `created` are replaced (`system_fingerprint` is the engine version and is kept);
- the 1070- and 798-id `prompt_token_ids` and the empty `prompt_text` are elided;
- the middle of each `token_ids` list is elided, so only its tail is shown;
- the reasoning text is shown as served.

Each stream is two data lines plus `[DONE]`.

**Tool case: click-01.** The content ends before the stray `<channel|>` (101). The call is only in the ids: `… 101, 48, …, 49, 50`.

```
data: {"id":"chatcmpl-<anon>","object":"chat.completion.chunk","created":<t>,"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":[<1070 ids>],"prompt_text":""}

data: {"id":"chatcmpl-<anon>","object":"chat.completion.chunk","created":<t>,"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","choices":[{"index":0,"delta":{"content":"I will click the Sign in button.","reasoning":"The user wants to click the \"Sign in\" button.\nI need to find the coordinates of the \"Sign in\" button.\nThe \"Sign in\" button is a blue button near the center of the screen.\nIts coordinates look approximately around x=490, y=518."},"logprobs":null,"finish_reason":"stop","stop_reason":50,"token_ids":[100,45518,107, <64 ids>, 236777,795,4845,506,9753,528,4309,236761,101,48,6639,236787,4513,236779,502,236782,236781,236787,236812,236819,236771,236764,236762,236787,236810,236770,236828,236783,49,50]}],"system_fingerprint":"vllm-0.29.0-b591a092"}

data: [DONE]
```

**No-tool case: chat-01.** The content is the whole answer region, and the stop is 106 (`<turn|>`).

```
data: {"id":"chatcmpl-<anon>","object":"chat.completion.chunk","created":<t>,"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":[<798 ids>],"prompt_text":""}

data: {"id":"chatcmpl-<anon>","object":"chat.completion.chunk","created":<t>,"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","choices":[{"index":0,"delta":{"content":"You're very welcome! Let me know if you need anything else later. Have a great day!","reasoning":"The user is saying \"Thanks, that's all for now!\", which indicates the end of the interaction. I should respond politely and not call any tools."},"logprobs":null,"finish_reason":"stop","stop_reason":106,"token_ids":[100,45518,107, <25 ids>, 124954,532,711,2246,1027,6436,236761,101,3048,236789,500,1401,8349,236888,3792,786,1281,768,611,1202,4658,1663,3209,236761,13433,496,1822,1719,236888,106]}],"system_fingerprint":"vllm-0.29.0-b591a092"}

data: [DONE]
```

## What this means for T-0088 and T-0091

- **T-0088** is replanned under H10.
  - The live client reads the per-chunk `token_ids` (`return_token_ids: true`) and splits answer from call by host detokenisation of the
    ids.
  - It cannot rely on `content`, because content will never contain the call.
  - The body sends `tools` in the prefill as well as in the generate request.
  - Whether the server flags cause the drop is an open question for that replan. A one-arm probe without `--enable-auto-tool-choice`
    would settle it; that is a `/launch` item and is not part of this task.
- **T-0091** uses the following:
  - a plain-string continuation;
  - the engine's `/tokenize` for seeds;
  - a stop-id rule replanned on the evidence above (`stop_reason` 50 on every tool case).
