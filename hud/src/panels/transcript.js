// The user transcript (criterion 8): a partial line is replaced in place, a final one is frozen
// and the next partial starts a new line. DOM lives here, not in state.js.
//
// T-0154 B1/B3: the page no longer has a transcript pane. The conversation thread
// (hud/src/panels/thread.js) keeps one user bubble per turn and writes it with renderBubble below;
// renderTranscript stays for its callers.

export function renderTranscript(root, lines) {
  if (!root) return;
  const doc = root.ownerDocument;
  root.replaceChildren(
    ...lines.map((line) => {
      const el = doc.createElement("p");
      el.className = line.final ? "line final" : "line partial";
      el.textContent = line.text;
      return el;
    }),
  );
  if (!lines.length) {
    const empty = doc.createElement("p");
    empty.className = "line empty";
    empty.textContent = "listening…";
    root.append(empty);
  }
}

/** One turn's user bubble: the line's text (textContent only — the transcript is the user's
 *  words, never markup) and whether it is still partial. The element is the caller's and keeps
 *  its identity; a partial is replaced by its successor in place. */
export function renderBubble(el, line) {
  if (!el || !line) return;
  el.className = line.final ? "bubble final" : "bubble partial";
  el.textContent = line.text;
  // T-0181 A0-S4 / H-T181-10: a question dropped by a mic mute ends with an empty final; it shows
  // no bubble.
  el.hidden = Boolean(line.final) && line.text === "";
}
