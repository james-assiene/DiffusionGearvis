# tests/fixtures/t0174

## `w2_0_step2.json` (T-0174 A7, group A, text only)

- **Source:** `logs/t0160-web-smoke-abcf17d/turns/W2-0/sessions/20261027-160402-470048.jsonl` (gitignored; the T-0160
  web smoke at abcf17d, task W2 "Go to example.com and read me the page heading.", repeat 0, session
  `ba2adf1154c8`), sha256 `9b870f46a49f66b5a78864eb4df78263bc1664b4256dc64997e00173b53e2724`, 106,621 bytes.
- **Fixture:** sha256 `7aa1f9e3748c96c9200edcdc6ddcf51a505a93a4a737915e610822279db883ff`, 11,874 bytes.
- **What it holds:**
  - `request`: the `data` of record seq 53, the `llm.request` of step 2 (`phase: tool_step`, `request_id`
    `t1-gen-s2`, `attempt_id` `ba2adf1154c8-t1-gen-s2-3`, the only attempt of that request). Its one image part
    (already a redacted digest in the log, no pixels) is replaced by the string `"<image omitted>"`; nothing else
    is changed. The system message is the live v1 system text the session sent.
  - `step1_tool_call`: the `data` of record seq 49 (`tool.call`, `type_at`, `executed`).
  - `step1_tool_response`: the `response` of record seq 51 (`tool.result`, step 1), as logged. (The rendered
    response inside `request`'s final assistant message also carries `screenshot: "updated"`, added when the step-2
    request was built.)
  - `step2_first_attempt`: the attempt id, the `thought`-segment commit texts of that attempt joined in log order
    (seqs 100, 163), and its `answer`-segment text (seq 164).
- **Extraction** (stdlib only, read-only over the log, written to the session scratchpad, then copied here byte for
  byte and compared with `cmp`), run on the 5090 box on 2026-10-27 from the main checkout:

  ```
  python3 extract_w2_0.py \
      logs/t0160-web-smoke-abcf17d/turns/W2-0/sessions/20261027-160402-470048.jsonl <scratchpad>/w2_0_step2.json
  ```

  `extract_w2_0.py` (sha256 `31d806ae4cec8f7ebf33391f998c05b9de82b2abd196fa0618d0ff66a4f8b309`):

  ```python
  """Extract the T-0174 A7 W2-0 step-2 case (text only) from the T-0160 web smoke log. Stdlib only.

  Usage: python extract_w2_0.py <session.jsonl> <out.json>
  """

  import hashlib
  import json
  import sys
  from pathlib import Path

  SEQ_REQUEST = 53
  SEQ_TOOL_CALL = 49
  SEQ_TOOL_RESULT = 51


  def strip_images(obj):
      if isinstance(obj, dict):
          if obj.get("type") == "image_url":
              return "<image omitted>"
          return {k: strip_images(v) for k, v in obj.items()}
      if isinstance(obj, list):
          return [strip_images(v) for v in obj]
      return obj


  def main(src: str, out: str) -> None:
      raw = Path(src).read_bytes()
      recs = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
      by_seq = {r["seq"]: r for r in recs}
      req = by_seq[SEQ_REQUEST]
      assert (
          req["kind"] == "llm.request"
          and req["data"]["phase"] == "tool_step"
          and req["data"]["step"] == 2
      )
      attempt = req["data"]["attempt_id"]
      rid = req["data"]["request_id"]
      attempts = sorted(
          {
              r["data"]["attempt_id"]
              for r in recs
              if r["kind"] == "llm.request" and r["data"].get("request_id") == rid
          }
      )
      assert attempts == [attempt], attempts  # one attempt: the first is the only one
      commits = [
          r for r in recs if r["kind"] == "llm.commit" and r["data"].get("attempt_id") == attempt
      ]
      thought = "".join(r["data"]["text"] for r in commits if r["data"]["segment"] == "thought")
      answer = "".join(r["data"]["text"] for r in commits if r["data"]["segment"] == "answer")
      call = by_seq[SEQ_TOOL_CALL]
      result = by_seq[SEQ_TOOL_RESULT]
      assert (
          call["kind"] == "tool.call"
          and result["kind"] == "tool.result"
          and result["data"]["step"] == 1
      )
      case = {
          "provenance": {
              "source": "logs/t0160-web-smoke-abcf17d/turns/W2-0/sessions/20261027-160402-470048.jsonl",
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "request_seq": SEQ_REQUEST,
              "tool_call_seq": SEQ_TOOL_CALL,
              "tool_result_seq": SEQ_TOOL_RESULT,
              "thought_commit_seqs": [r["seq"] for r in commits if r["data"]["segment"] == "thought"],
              "answer_commit_seqs": [r["seq"] for r in commits if r["data"]["segment"] == "answer"],
              "images": 'every image_url part replaced by the string "<image omitted>" (the log held only a redacted digest)',
          },
          "request": strip_images(req["data"]),
          "step1_tool_call": call["data"],
          "step1_tool_response": result["data"]["response"],
          "step2_first_attempt": {"attempt_id": attempt, "thought": thought, "answer": answer},
      }
      text = json.dumps(case, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
      assert "base64" not in text and "data:image" not in text
      Path(out).write_text(text, encoding="utf-8")


  if __name__ == "__main__":
      main(sys.argv[1], sys.argv[2])
  ```

- **Not here:** no pixels, no image digests, nothing from `logs/sessions/`.
- **Regime of the source run:** lean, canvas 256, 48 steps, thinking on, NVFP4 rev 2ea8372, vLLM 0.29.0, tools
  native, live (web on through the egress proxy), loop 8, live guidance v1 (sha 768a973b), headless Chromium, typed
  turn, engine unseeded, n = 1.
- **Licence:** model output produced on this box from a public test page (example.com) in a smoke run;
  project-internal.
