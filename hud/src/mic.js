// T-0181 B: the HUD microphone control, DOM-free (main.js does the wiring).
//
// The control shows only the server's last astra.mic.state echo. A click asks for the opposite of
// that echo and changes nothing on its own: the view moves only when the next echo arrives (B1). A
// session with no microphone shows a disabled "No mic" with the server's reason as its title
// (H-T181-5). There is no keyboard entry point: only the button toggles the mic (B3, H-T181-4). A
// reset does not touch it: the mute survives Restart and the server re-sends the state only on a
// hello (A9).

/** The one message this module makes the page send (B5). */
export const MIC_CLIENT_TYPES = ["astra.mic.mute"];

const LIVE = {
  text: "Mute mic",
  title: "Mute the microphone: Gearvis hears nothing until you unmute; its own speech still plays and typing still works",
};
const MUTED = { text: "Unmute mic", title: "Unmute the microphone" };
const BADGE = { text: "Mic muted", title: "Gearvis can't hear you. Click Unmute mic to let it listen again." };
const NO_MIC = "No mic";

export function createMic() {
  let echoed = null; // the last astra.mic.state, or null before the first one

  return {
    /** Every decoded astra.* message; true iff it was an astra.mic.state. */
    accept(msg) {
      if (!msg || typeof msg !== "object" || msg.type !== "astra.mic.state") return false;
      echoed = { muted: msg.muted === true, available: msg.available === true, reason: msg.reason ?? null };
      return true;
    },

    /** null before any state; else what the button and the badge show. */
    view() {
      if (echoed === null) return null;
      if (!echoed.available) {
        return { text: NO_MIC, title: String(echoed.reason ?? ""), pressed: null, disabled: true, badge: null };
      }
      const copy = echoed.muted ? MUTED : LIVE;
      return {
        text: copy.text,
        title: copy.title,
        pressed: echoed.muted,
        disabled: false,
        badge: echoed.muted ? { ...BADGE } : null,
      };
    },

    /** The message a click sends (the inverse of the echo), or null with no state or no mic. */
    click() {
      if (echoed === null || !echoed.available) return null;
      return { type: "astra.mic.mute", muted: !echoed.muted };
    },
  };
}
