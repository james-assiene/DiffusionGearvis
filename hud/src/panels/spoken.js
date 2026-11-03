// The spoken channel (criteria 8 and 11): utterances grouped by turn_id in arrival order, keyed
// by utterance_id, and an interrupted turn is marked.
//
// The played/unplayed rendering (T-0030 criteria 19 and 22) comes from astra.spoken.played and
// from nothing else: the words from index `played_words` on go inside one [data-unplayed]
// element, which hud/styles.css strikes through and dims. The full text is always in the DOM —
// struck, never removed (docs/02 §4: "the HUD may show unspoken text struck through"). With no
// measurement on the socket an utterance renders exactly as it did before; elapsed time and an
// INTERRUPTED turn dim nothing.
//
// T-0154 B1: the page no longer has a spoken pane; hud/src/panels/thread.js renders each turn's
// utterances in its article with renderUtteranceText below. renderSpoken stays for its tests.

export function renderSpoken(root, groups, interrupted = []) {
  if (!root) return;
  const doc = root.ownerDocument;
  root.replaceChildren(
    ...groups.map((group) => {
      const section = doc.createElement("section");
      section.className = "turn";
      section.dataset.turn = group.turn_id;
      if (interrupted.includes(group.turn_id)) {
        section.classList.add("interrupted");
        section.dataset.interrupted = "true";
        const mark = doc.createElement("span");
        mark.className = "mark";
        mark.textContent = "interrupted";
        section.append(mark);
      }
      for (const utterance of group.utterances) {
        const line = doc.createElement("p");
        line.className = "utterance";
        line.dataset.utteranceId = utterance.utterance_id;
        line.dataset.turnId = group.turn_id;
        renderUtteranceText(doc, line, utterance);
        section.append(line);
      }
      return section;
    }),
  );
}

/** The text of one utterance, cut at the measured word index (criterion 19).
 *
 *  No played state, or `complete`, means today's rendering: one text node, no extra element.
 *  Otherwise the words from `played_words` on go inside a single [data-unplayed] element, and
 *  the two pieces still concatenate to `utterance.text` character for character — the cut is an
 *  index into the *displayed* text, which is valid because audio_io refuses a partial prefix
 *  unless the source and the rendered word counts agree (T-0030 landmine 1).
 *
 *  Exported for the conversation thread (hud/src/panels/thread.js, T-0154 implementation note
 *  2), which reuses it unchanged inside each utterance's own element.
 */
export function renderUtteranceText(doc, line, utterance) {
  const text = utterance.text;
  const cut =
    typeof utterance.played_words === "number" && utterance.complete !== true
      ? wordStart(text, utterance.played_words)
      : text.length;
  if (cut >= text.length) {
    line.textContent = text; // nothing measured as unheard: today's rendering, unchanged
    return;
  }
  line.textContent = text.slice(0, cut);
  const tail = doc.createElement("span");
  tail.className = "unplayed";
  tail.dataset.unplayed = "true";
  tail.textContent = text.slice(cut);
  line.append(tail);
}

/** The index in `text` where word `n` (0-based) starts, or `text.length` if there is no such
 *  word. `n === 0` is the start of the first word, so none of the text is claimed as played. */
function wordStart(text, n) {
  const words = /\S+/g;
  let seen = 0;
  let match;
  while ((match = words.exec(text)) !== null) {
    if (seen === n) return match.index;
    seen += 1;
  }
  return text.length;
}
