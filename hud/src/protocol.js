// The wire protocol, mirrored from local_astra/contracts/protocol.py and codec.py (T-0008
// criteria 1 and 2). The Python side is the spec; this file follows it and never the reverse.
//
// decode() is *total*, exactly as contracts/codec.decode is: whatever arrives on the socket it
// returns one of four outcomes and never throws, so a bad message can never kill the socket
// ("the Conductor answers with astra.error and never a crash" — codec.py).
//
//   { kind: "astra",     type, msg }   known astra.* type, all fields valid  -> typed dispatch
//   { kind: "unknown",   type }        astra.* we do not know                -> ignored, counted
//   { kind: "realtime",  type, msg }   a Realtime event passed through       -> ignored, counted
//   { kind: "malformed", type?, error} bad json / bad fields / oversized     -> ignored, counted
//
// No DOM here: `node --test tests/hud` imports this module.

export const PROTOCOL_VERSION = 1; // contracts/protocol.py
export const MAX_MESSAGE_BYTES = 8388608; // contracts/codec.py: 8 * 1024 * 1024

/** The only client->server types this page may ever emit (criterion 1). The remaining
 *  client->server types in contracts/protocol.py CLIENT_TO_SERVER belong to T-0010 and T-0012. */
export const CLIENT_TYPES = ["astra.hello", "astra.frame"];

/** contracts/protocol.py SERVER_TO_CLIENT — what the page dispatches, and nothing else. */
export const SERVER_TO_CLIENT = [
  "astra.assist.state",
  "astra.turn.state",
  "astra.ui.surface",
  "astra.ui.fence",
  "astra.canvas.step",
  "astra.canvas.commit",
  "astra.canvas.synthetic",
  "astra.transcript",
  "astra.spoken",
  "astra.spoken.played",
  "astra.log.state",
  "astra.tool.call",
  "astra.session.cleared",
  "astra.error",
  "astra.vision.state",
  "astra.vision.frame",
  "astra.thought",
  "astra.mic.state",
  "astra.settings.state",
];

/** contracts/protocol.py THOUGHT_MAX_CHARS (T-0180 A6): the most thought text one astra.thought
 *  carries, in code points (not UTF-16 units). */
export const THOUGHT_MAX_CHARS = 16000;

const SOURCES = ["screen", "window", "camera"]; // contracts/perception.py Source
const TURN_STATES = [
  // contracts/turn.py TurnState
  "IDLE",
  "LISTENING",
  "ENDPOINTING",
  "THINKING",
  "SPEAKING",
  "PAUSED",
  "INTERRUPTED",
];
const SEGMENTS = ["thought", "answer", "ui"]; // contracts/llm.py Segment
const FENCE_CLOSES = ["written", "inferred", "open", "abandoned"]; // contracts/llm.py FenceClose
// contracts/protocol.py ToolCallStatus and ToolCallSource (T-0090 criterion 1)
const TOOL_CALL_STATUSES = ["proposed", "repairing", "dry_run", "executed", "refused", "cancelled"];
const TOOL_CALL_SOURCES = ["draft", "repair"];
// contracts/protocol.py ToolCallOutcome and TOOL_RESULT_HOSTS (T-0162 C-1)
const TOOL_CALL_OUTCOMES = ["executed", "dry_run", "refused", "failed", "cancelled", "timeout"];
const TOOL_RESULT_HOSTS = 3;
// TOOL_RESULT_HOST_PATTERN ("host only"; the 253 cap is the item's max_length)
const TOOL_RESULT_HOST = /^[A-Za-z0-9_-]{1,63}(\.[A-Za-z0-9_-]{1,63})*$/;
// contracts/protocol.py VisionSource, VisionImage.kind and VISION_MAX_IMAGES (T-0177 A1)
const VISION_SOURCES = ["browser", "shared"];
const VISION_IMAGE_KINDS = ["window", "screen", "camera", "overview"];
const VISION_MAX_IMAGES = 4;
// contracts/protocol.py ThoughtKind and ThoughtEnd (T-0180 A1)
const THOUGHT_KINDS = ["step", "reask", "retry", "repair", "reground"];
const THOUGHT_ENDS = ["open", "done", "stopped_repeating", "interrupted", "error"];
// contracts/protocol.py SettingSource and SettingKindName (T-0213 A2r-1)
const SETTING_SOURCES = ["default", "profile", "flag", "ui"];
const SETTING_KINDS = ["next_turn", "next_turn_cold", "relaunch", "fixed", "not_here"];

/** Code points, as Python's len() counts them: a surrogate pair is one, a lone surrogate one. */
function codePoints(s) {
  let n = 0;
  for (const _ of s) n += 1;
  return n;
}

// --- field specs -------------------------------------------------------------------------------

const str = (opts = {}) => ({ kind: "str", ...opts });
const int = (opts = {}) => ({ kind: "int", ...opts });
const bool = (opts = {}) => ({ kind: "bool", ...opts });
const num = (opts = {}) => ({ kind: "num", ...opts });
const oneOf = (values, opts = {}) => ({ kind: "enum", values, ...opts });
const list = (opts = {}) => ({ kind: "list", ...opts });
const obj = (opts = {}) => ({ kind: "obj", ...opts });
/** A nested Contract: its own field table, validated as strictly as a top-level message. */
const model = (fields, opts = {}) => ({ kind: "model", fields, ...opts });
/** contracts/protocol.py SettingValue (T-0213 A2r-1): a bool, an integer, a string or a list of integers. */
const settingValue = (opts = {}) => ({ kind: "setting", ...opts });

function fieldError(key, value, spec) {
  switch (spec.kind) {
    case "str":
      if (typeof value !== "string") return `${key}: input should be a valid string`;
      if (spec.minLength && value.length < spec.minLength) return `${key}: string too short`;
      if (spec.maxLength !== undefined && value.length > spec.maxLength) return `${key}: string too long`;
      if (spec.pattern && !spec.pattern.test(value)) return `${key}: string does not match pattern`;
      return null;
    case "int":
      if (typeof value !== "number" || !Number.isInteger(value))
        return `${key}: input should be a valid integer`;
      if (spec.min !== undefined && value < spec.min)
        return `${key}: input should be greater than or equal to ${spec.min}`;
      return null;
    case "num":
      if (typeof value !== "number" || !Number.isFinite(value))
        return `${key}: input should be a valid number`;
      return null;
    case "bool":
      return typeof value === "boolean" ? null : `${key}: input should be a valid boolean`;
    case "enum":
      return spec.values.includes(value) ? null : `${key}: input should be one of the enum values`;
    case "list": {
      if (!Array.isArray(value)) return `${key}: input should be a valid list`;
      if (spec.maxItems !== undefined && value.length > spec.maxItems)
        return `${key}: list should have at most ${spec.maxItems} items`;
      if (!spec.item) return null;
      for (const item of value) {
        const err = fieldError(key, item, spec.item);
        if (err) return err;
      }
      return null;
    }
    case "setting": {
      if (typeof value === "boolean" || typeof value === "string") return null;
      if (typeof value === "number")
        return Number.isInteger(value) ? null : `${key}: input should be a valid integer`;
      if (Array.isArray(value) && value.every((v) => typeof v === "number" && Number.isInteger(v))) return null;
      return `${key}: input should be a bool, an integer, a string or a list of integers`;
    }
    case "obj":
      return value && typeof value === "object" && !Array.isArray(value)
        ? null
        : `${key}: input should be a valid object`;
    case "model": {
      if (!value || typeof value !== "object" || Array.isArray(value))
        return `${key}: input should be a valid object`;
      // A nested Contract carries no discriminating "type", so — unlike a message — an extra
      // "type" key is an extra input here too.
      for (const k of Object.keys(value))
        if (!(k in spec.fields)) return `${key}.${k}: extra inputs are not permitted`;
      const err = validate(value, spec.fields);
      return err ? `${key}.${err}` : null;
    }
    default:
      return `${key}: unchecked field`;
  }
}

/** contracts/base.Contract is strict: unknown fields are errors, so they are here too. */
function validate(message, fields, after) {
  for (const key of Object.keys(message)) {
    if (key !== "type" && !(key in fields)) return `${key}: extra inputs are not permitted`;
  }
  for (const [key, spec] of Object.entries(fields)) {
    const value = message[key];
    if (value === undefined) {
      if (spec.required) return `${key}: field required`;
      continue;
    }
    if (value === null && spec.nullable) continue;
    const err = fieldError(key, value, spec);
    if (err) return err;
  }
  return after ? after(message) : null;
}

// contracts/protocol.py ToolCallResult (T-0162 C-1), the nested field table of astra.tool.call's `result`
const TOOL_CALL_RESULT = {
  outcome: oneOf(TOOL_CALL_OUTCOMES, { required: true }),
  hosts: list({ item: str({ pattern: TOOL_RESULT_HOST, maxLength: 253 }), maxItems: TOOL_RESULT_HOSTS }),
  hosts_more: int({ min: 0 }),
  results: int({ min: 0, nullable: true }),
};

// contracts/protocol.py VisionImage (T-0177 A1), the nested field table of astra.vision.frame's `images`
const VISION_IMAGE = {
  identity: str({ required: true }),
  kind: oneOf(VISION_IMAGE_KINDS, { required: true }),
  width: int({ min: 1, required: true }),
  height: int({ min: 1, required: true }),
  jpeg_b64: str({ nullable: true }),
  omitted: oneOf(["too_large"], { nullable: true }),
};

// contracts/protocol.py SettingItem, CatalogEntry and SettingRefusal (T-0213 A2r-1), astra.settings.state's rows
const SETTING_ITEM = {
  key: str({ minLength: 1, required: true }),
  value: settingValue({ nullable: true }),
  source: oneOf(SETTING_SOURCES, { required: true }),
  kind: oneOf(SETTING_KINDS, { required: true }),
  changeable: bool({ required: true }),
  pending: settingValue({ nullable: true }),
  relaunch: str({ nullable: true }),
};
const CATALOG_ENTRY = {
  key: str({ minLength: 1, required: true }),
  label: str({ required: true }),
  group: str({ required: true }),
  kind: oneOf(SETTING_KINDS, { required: true }),
  v1_changeable: bool({ required: true }),
  values: list({ item: settingValue({ nullable: true }) }),
  default: settingValue({ nullable: true }),
  default_note: str({ required: true }),
  flag: str({ required: true }),
  flags: list({ item: str() }),
  profile_key: str({ nullable: true }),
  what: str({ required: true }),
  useful_when: str({ required: true }),
  not_useful_when: str({ required: true }),
  evidence: str({ required: true }),
  cost: str({ required: true }),
  relaunch: str({ nullable: true }),
};
const SETTING_REFUSAL = {
  key: str({ minLength: 1, required: true }),
  reason: str({ required: true }),
};

const MODELS = {
  "astra.hello": [{ protocol_version: int({ min: 0 }), client: str() }],
  "astra.frame": [
    {
      source: oneOf(SOURCES, { required: true }),
      ts_ms: int({ min: 0, required: true }),
      width: int({ min: 1, required: true }),
      height: int({ min: 1, required: true }),
      jpeg_b64: str({ minLength: 1, required: true }),
      phash: str({ nullable: true }),
      title: str({ nullable: true }),
      app: str({ nullable: true }),
    },
  ],
  "astra.assist.state": [{ flags: obj({ required: true }), badge: str({ required: true }) }],
  "astra.turn.state": [
    {
      state: oneOf(TURN_STATES, { required: true }),
      turn_id: str({ required: true }),
      thinking_ms: int({ min: 0, nullable: true }),
      replay: bool(),
    },
  ],
  "astra.ui.surface": [
    {
      surface_id: str({ required: true }),
      canvas_idx: int({ min: 0, required: true }),
      final: bool(),
      messages: list({ item: obj(), required: true }),
      repaired: int({ min: 0 }),
      replay: bool(),
    },
  ],
  // T-0054 criterion 13: what happened to the interface this turn — whether the fence closed,
  // whether it was carried across canvases, and how many strings the voice guard refused. A new
  // *type* rather than a field on astra.ui.surface (D1): an unmirrored field makes every message
  // of that type malformed and blanks the shown pane, while an unmirrored type decodes as
  // kind:"unknown" and is ignored and counted. `close` is the four-literal enum, so a value a
  // future contract adds decodes malformed instead of falling through to whatever the panel's
  // status table would call it.
  "astra.ui.fence": [
    {
      turn_id: str({ required: true }),
      canvases: int({ min: 0, required: true }),
      close: oneOf(FENCE_CLOSES, { nullable: true }),
      continued: bool(),
      speech_refused: int({ min: 0 }),
    },
  ],
  // T-0128c B1: AstraCanvasStep as the plugin tap sends it (tap.py). One wire shape per phase: a
  // denoise frame carries canvas/argmax/stable and no committed ids; a commit frame carries the
  // non-empty committed ids and none of the rows. accepted/entropy are optional (never sent by
  // v0.29.0). `phase` defaults to "denoise" in the contract, so an absent phase is a denoise frame.
  "astra.canvas.step": [
    {
      request_id: str({ required: true }),
      canvas_idx: int({ min: 0, required: true }),
      step: int({ min: 0, required: true }),
      max_steps: int({ min: 1, required: true }),
      canvas_length: int({ min: 1, required: true }),
      canvas: list({ item: int(), nullable: true }),
      argmax: list({ item: int(), nullable: true }),
      stable: str({ pattern: /^[0-9a-f]*$/, nullable: true }),
      phase: oneOf(["denoise", "commit"]),
      committed_tokens: list({ item: int() }),
      accepted: str({ pattern: /^[0-9a-f]*$/, nullable: true }),
      entropy: list({ item: num(), nullable: true }),
      segment: oneOf(SEGMENTS, { nullable: true }),
      text_preview: str({ nullable: true }),
      // T-0159 A1/A6: one text piece per position, added by the conductor's relay; never decoded here
      argmax_pieces: list({ item: str(), nullable: true }),
      committed_pieces: list({ item: str(), nullable: true }),
    },
    (m) => {
      // AstraCanvasStep._lengths_agree, check for check and in the same order
      const n = m.canvas_length;
      const present = (v) => v !== undefined && v !== null;
      for (const name of ["canvas", "argmax", "entropy"])
        if (present(m[name]) && m[name].length !== n) return `${name} must have canvas_length entries`;
      for (const name of ["stable", "accepted"])
        if (present(m[name]) && (m[name].length * 4 < n || m[name].length % 2))
          return `${name} mask must cover canvas_length bits in whole bytes`;
      const committed = m.committed_tokens || [];
      if (committed.length > n) return "committed_tokens cannot exceed canvas_length";
      const rows = ["canvas", "argmax", "stable"].map((k) => present(m[k]));
      if ((m.phase === undefined ? "denoise" : m.phase) === "denoise") {
        if (!rows.every(Boolean)) return "a denoise frame carries canvas, argmax and stable";
        if (committed.length) return "committed_tokens belong to a commit frame";
        if (m.step > m.max_steps) return "a denoise frame's step must be in 0..max_steps";
      } else {
        if (rows.some(Boolean)) return "a commit frame carries none of canvas, argmax and stable";
        if (!committed.length) return "a commit frame carries non-empty committed_tokens";
        if (m.step > m.max_steps) return "a commit frame's step must be in 0..max_steps";
      }
      // T-0159 A1: pieces belong to their phase's ids and have one entry per id
      if ((m.phase === undefined ? "denoise" : m.phase) === "denoise") {
        if (present(m.committed_pieces)) return "committed_pieces belong to a commit frame";
        if (present(m.argmax_pieces) && m.argmax_pieces.length !== n)
          return "argmax_pieces must have canvas_length entries";
      } else {
        if (present(m.argmax_pieces)) return "argmax_pieces belong to a denoise frame";
        if (present(m.committed_pieces) && m.committed_pieces.length !== committed.length)
          return "committed_pieces must have one entry per committed token";
      }
      return null;
    },
  ],
  "astra.canvas.commit": [
    {
      request_id: str({ required: true }),
      canvas_idx: int({ min: 0, required: true }),
      segment: oneOf(SEGMENTS, { required: true }),
      text: str({ required: true }),
      steps_used: int({ min: 0, required: true }),
    },
  ],
  "astra.canvas.synthetic": [{ request_id: str({ required: true }), reason: str({ required: true }) }],
  "astra.transcript": [
    {
      turn_id: str({ required: true }),
      text: str({ required: true }),
      final: bool(),
      replay: bool(),
    },
  ],
  "astra.spoken": [
    {
      turn_id: str({ required: true }),
      utterance_id: str({ required: true }),
      text: str({ required: true }),
      replay: bool(),
    },
  ],
  "astra.spoken.played": [
    {
      turn_id: str({ required: true }),
      played_ms: int({ min: 0, required: true }),
      utterances: list({
        // contracts/protocol.py PlayedUtterance
        item: model({
          utterance_id: str({ required: true }),
          played_text: str({ required: true }),
          played_words: int({ min: 0, required: true }),
          complete: bool({ required: true }),
        }),
        required: true,
      }),
      replay: bool(),
    },
  ],
  // T-0043 criterion 20: whether this session is being written to a file, and which one.
  // `on` is required and not nullable: the page's three-state checkbox already has a state for
  // "the server has not said" (no message), so "the server said null" must not be reachable.
  // `path` and `error` are nullable because logging off has no file and a log that disabled
  // itself has a reason (criterion 4); the counters are always known, so they are required.
  "astra.log.state": [
    {
      on: bool({ required: true }),
      path: str({ nullable: true }),
      records: int({ min: 0, required: true }),
      dropped: int({ min: 0, required: true }),
      bytes: int({ min: 0, required: true }),
      error: str({ nullable: true }),
    },
  ],
  // T-0090 criterion 2: one state of one proposed action, for the action card. `arguments` is
  // the call's own object (any keys); status and source are the contract's literals, so a value a
  // future contract adds decodes malformed rather than reaching the card unlabelled.
  "astra.tool.call": [
    {
      turn_id: str({ required: true }),
      call_id: str({ required: true }),
      name: str({ required: true }),
      arguments: obj({ required: true }),
      status: oneOf(TOOL_CALL_STATUSES, { required: true }),
      source: oneOf(TOOL_CALL_SOURCES, { required: true }),
      action: str({ nullable: true }),
      reason: str({ nullable: true }),
      query_problem: str({ nullable: true }),
      // T-0162 C-1: the loop step, and the final message's short result (from data, never prose)
      step: int({ min: 1, nullable: true }),
      result: model(TOOL_CALL_RESULT, { nullable: true }),
      replay: bool(),
    },
    // AstraToolCall._result_only_when_final
    (m) =>
      m.result !== undefined && m.result !== null && (m.status === "proposed" || m.status === "repairing")
        ? "result is set only on a final message"
        : null,
  ],
  // T-0162 A-1: a completed session reset. `resets` counts this session's resets (>= 1), so the
  // page clears once per new value; `ended_turn_id` is the cut turn, null when none was open.
  "astra.session.cleared": [
    {
      resets: int({ min: 1, required: true }),
      ended_turn_id: str({ nullable: true }),
      memory_forgotten: int({ min: 0, required: true }),
      replay: bool(),
    },
  ],
  "astra.error": [{ code: str({ required: true }), message: str({ required: true }) }],
  // T-0177 A1: the vision selector. The select is the page's own egress (decoded here for parity with the
  // contract); the state is the session's echo of its choice; the frame is the images one request sent.
  "astra.vision.select": [{ source: oneOf(VISION_SOURCES, { required: true }) }],
  "astra.vision.state": [
    {
      source: oneOf(VISION_SOURCES, { required: true }),
      choices: list({ item: oneOf(VISION_SOURCES), required: true }),
      replay: bool(),
    },
  ],
  "astra.vision.frame": [
    {
      turn_id: str({ required: true }),
      step: int({ min: 1, required: true }),
      source: oneOf(VISION_SOURCES, { required: true }),
      blind: bool({ required: true }),
      images: list({ item: model(VISION_IMAGE), maxItems: VISION_MAX_IMAGES }),
      history_images: int({ min: 0, required: true }),
      replay: bool(),
    },
    (m) => {
      // VisionImage._omitted_iff_no_bytes, then AstraVisionFrame._blind_has_no_images
      for (const img of m.images || []) {
        const absent = img.jpeg_b64 === undefined || img.jpeg_b64 === null;
        if (absent !== (img.omitted === "too_large"))
          return 'images: jpeg_b64 is null exactly when omitted is "too_large"';
      }
      if (m.blind && (m.images || []).length) return "a blind frame carries no images";
      return null;
    },
  ],
  // T-0180 A1/A2: the committed thought text of one attempt, for the turn's "Model thoughts"
  // disclosure; the only message type that carries thought text. Rendered as text, never HTML.
  "astra.thought": [
    {
      turn_id: str({ minLength: 1, required: true }),
      request_id: str({ minLength: 1, required: true }),
      attempt_id: str({ minLength: 1, required: true }),
      step: int({ min: 1, required: true }),
      kind: oneOf(THOUGHT_KINDS, { required: true }),
      text: str({ required: true }),
      omitted_chars: int({ min: 0, required: true }),
      end: oneOf(THOUGHT_ENDS, { required: true }),
      replay: bool(),
    },
    (m) => {
      // AstraThought._capped_and_omitted_only_at_the_cap, in code points
      const n = codePoints(m.text);
      if (n > THOUGHT_MAX_CHARS) return `text is ${n} code points, over the ${THOUGHT_MAX_CHARS} cap`;
      if (m.omitted_chars > 0 && n < THOUGHT_MAX_CHARS) return "omitted_chars > 0 only when text is at the cap";
      return null;
    },
  ],
  // T-0181 A1: the microphone mute. The mute is the page's own egress (decoded here for parity with the
  // contract); the state is the session's echo of its switch. Both bools are strict (no coercion).
  "astra.mic.mute": [{ muted: bool({ required: true }) }],
  "astra.mic.state": [
    {
      muted: bool({ required: true }),
      available: bool({ required: true }),
      reason: str({ nullable: true }),
    },
    (m) => {
      // AstraMicState._unavailable_is_unmuted_with_a_reason
      if (!m.available && m.muted) return "an unavailable microphone is never muted";
      const noReason = m.reason === undefined || m.reason === null;
      if (noReason !== m.available) return "reason is set exactly when the microphone is unavailable";
      return null;
    },
  ],
  // T-0193 A-1: the page's stop request for one turn (the "Interrupt Gearvis" button); client to server
  // only, decoded here for parity with the contract. The echo is the existing astra.turn.state.
  "astra.turn.stop": [{ turn_id: str({ minLength: 1, required: true }) }],
  // T-0213 A2r-1: every setting with its value and source, read-only from the panel's side. `catalog` (the
  // panel's text) arrives on the hello reply only. Every string here is rendered as text, never HTML.
  "astra.settings.state": [
    {
      rev: int({ min: 0, required: true }),
      items: list({ item: model(SETTING_ITEM), required: true }),
      catalog: list({ item: model(CATALOG_ENTRY), nullable: true }),
      system_sha12: str({ required: true }),
      instructions: str({ required: true }),
      instructions_max: int({ min: 0, required: true }),
      refused: list({ item: model(SETTING_REFUSAL) }),
      replay: bool(),
      // T-0213 C-2 (added in A2-apply): the assembled system message, read-only text for the panel
      system_preview: str({ nullable: true }),
    },
    (m) => {
      // AstraSettingsState._instructions_within_the_cap, in code points
      const n = codePoints(m.instructions);
      if (n > m.instructions_max) return `instructions are ${n} code points, over the ${m.instructions_max} cap`;
      return null;
    },
  ],
  // T-0213 A2a-1: the panel's change request (client to server; decoded here for parity with the contract).
  // `changes` maps catalog keys to values the session validates (null clears one); the echo is
  // astra.settings.state. A stale rev, negative included, is the session's refusal, not a decode error.
  "astra.settings.set": [
    {
      rev: int({ required: true }),
      changes: obj({ required: true }),
      stop_turn: bool(),
    },
  ],
};

const REALTIME_TYPE = /^[a-z_.]+$/; // RealtimeEvent.type pattern

const malformed = (error, type) => (type ? { kind: "malformed", type, error } : { kind: "malformed", error });

/** UTF-8 bytes, which is what the limit is about — `raw.length` is UTF-16 units and would let a
 *  multibyte message up to ~3x the limit through (ruling R11). Only measured near the limit,
 *  because encoding a multi-megabyte string is not free. */
const ENCODER = typeof TextEncoder === "function" ? new TextEncoder() : null;

function overTheLimit(raw) {
  if (raw.length > MAX_MESSAGE_BYTES) return true; // UTF-8 bytes >= UTF-16 units, always
  if (raw.length * 3 <= MAX_MESSAGE_BYTES) return false; // ... and <= 3 bytes per unit, always
  return ENCODER ? ENCODER.encode(raw).length > MAX_MESSAGE_BYTES : false;
}

/** The astra.hello this page sends first, and only first (criterion 1). */
export function hello() {
  return { type: "astra.hello", protocol_version: PROTOCOL_VERSION, client: "hud" };
}

/** Total decode. Never throws, whatever arrives on the socket (criterion 2). */
export function decode(raw) {
  if (typeof raw !== "string") return malformed("not a text frame");
  // The size check comes before JSON.parse: an oversized message is refused, not parsed.
  if (overTheLimit(raw)) return malformed(`message exceeds ${MAX_MESSAGE_BYTES} bytes`);
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    return malformed(`invalid json: ${err && err.message}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    return malformed("top-level value must be an object");
  const type = parsed.type;
  if (typeof type !== "string" || !type) return malformed("missing or non-string 'type'");
  if (!type.startsWith("astra.")) {
    if (!REALTIME_TYPE.test(type)) return malformed("type: string does not match pattern", type);
    const payload = {};
    for (const [k, v] of Object.entries(parsed)) if (k !== "type") payload[k] = v;
    return { kind: "realtime", type, msg: { type, payload } };
  }
  const model = MODELS[type];
  if (!model) return { kind: "unknown", type };
  const error = validate(parsed, model[0], model[1]);
  if (error) return malformed(error, type);
  return { kind: "astra", type, msg: parsed };
}
