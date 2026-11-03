// T-0193 B: the HUD's "Interrupt Gearvis" button, DOM-free (main.js does the wiring).
//
// The button follows only the server's astra.turn.state echo. It is hidden before the first one
// (B-1), then shown, and enabled iff the last echo is THINKING, SPEAKING or PAUSED. A click sends
// astra.turn.stop for the echoed turn and disables the button until the next accepted echo (B-2).
// A state for a turn this page has already taken that is not the current echo's turn is stale and
// moves nothing, live or replayed (B-3, state.js's chip rule). astra.session.cleared disarms it
// until a later turn's state. Turn ids restart with a new server, so the memory of seen turns is
// per socket: a dropped socket disarms the button and forgets them (connection(false)). There is
// no keyboard entry point: only the button stops a turn (B-4, H-T193-3).

/** The one message this module makes the page send (B-5). */
export const STOP_CLIENT_TYPES = ["astra.turn.stop"];

const TEXT = "Interrupt Gearvis";
const TITLE = "Interrupt Gearvis: it stops talking and stops what it is doing in the browser";
const STOPPABLE = new Set(["THINKING", "SPEAKING", "PAUSED"]);

export function createStop() {
  let shown = false; // an astra.turn.state has arrived on some socket (B-1)
  let echo = null; // { state, turn_id } of the current echo; null when disarmed
  let sent = false; // a stop went out since that echo
  let seen = new Set(); // the turn ids taken on this socket

  const armed = () => echo !== null && !sent && STOPPABLE.has(echo.state);

  return {
    /** Every decoded astra.* message; true iff it was an astra.turn.state taken as the echo. */
    accept(msg) {
      if (!msg || typeof msg !== "object") return false;
      if (msg.type === "astra.session.cleared") {
        echo = null;
        sent = false;
        return shown;
      }
      if (msg.type !== "astra.turn.state" || typeof msg.turn_id !== "string") return false;
      const id = msg.turn_id;
      if (seen.has(id) && (echo === null || echo.turn_id !== id)) return false; // stale (B-3)
      seen.add(id);
      shown = true;
      echo = { state: String(msg.state), turn_id: id };
      sent = false;
      return true;
    },

    /** The socket opened (true) or closed (false). Closing forgets the echo and the seen turns. */
    connection(open) {
      if (open) return false;
      echo = null;
      sent = false;
      seen = new Set();
      return shown;
    },

    /** null before any astra.turn.state; else what the button shows. */
    view() {
      if (!shown) return null;
      return { text: TEXT, title: TITLE, disabled: !armed() };
    },

    /** The stop a click sends, or null while disabled; a sent stop disables it until the next echo. */
    click() {
      if (!armed()) return null;
      sent = true;
      return { type: "astra.turn.stop", turn_id: echo.turn_id };
    },
  };
}
