// The socket: where it may point (criterion 12) and how it comes back (criterion 3).
// No DOM here: `node --test tests/hud` imports this module, and WebSocket is only touched
// inside connect(), never at module scope.

export const DEFAULT_WS_URL = "ws://127.0.0.1:8000/v1/realtime";

export const BACKOFF_INITIAL_MS = 250;
export const BACKOFF_CAP_MS = 8000;

/** CLAUDE.md: all services bind to loopback only. Nothing else is reachable from this page. */
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

/**
 * The URL to connect to, from a location.search string. `?ws=` is honoured only for the three
 * loopback hosts; anything else throws and the page shows the error instead of connecting.
 */
export function resolveWsUrl(search = "") {
  const wanted = new URLSearchParams(typeof search === "string" ? search : "").get("ws");
  if (!wanted) return DEFAULT_WS_URL;
  let url;
  try {
    url = new URL(wanted);
  } catch {
    throw new Error(`refusing ?ws=${wanted}: not an absolute ws:// URL`);
  }
  if (url.protocol !== "ws:" && url.protocol !== "wss:")
    throw new Error(`refusing ?ws=${wanted}: not a WebSocket URL`);
  if (url.username || url.password)
    throw new Error(`refusing ?ws=${wanted}: credentials in the URL hide the real host`);
  if (!LOOPBACK_HOSTS.has(url.hostname))
    throw new Error(`refusing ?ws=${wanted}: ${url.hostname} is not loopback`);
  return wanted;
}

/** Bounded exponential backoff with jitter: 250 ms initial, 8 s cap, never zero. */
export function backoffDelay(attempt, random = Math.random) {
  const n = Math.min(30, Math.max(0, Math.floor(Number(attempt) || 0)));
  const base = Math.min(BACKOFF_CAP_MS, BACKOFF_INITIAL_MS * 2 ** n);
  return Math.max(1, Math.round(base * (0.5 + random() * 0.5)));
}

/**
 * One socket at a time, reconnecting with the backoff above. The seams (socketFactory, setTimer,
 * random) are injected so this is drivable without a browser.
 *
 * `send` rejects while the socket is not open — the frame pipeline turns that into a drop, which
 * is what keeps the outbound queue bounded (criterion 3).
 */
export function connect({
  url,
  onMessage = () => {},
  onStatus = () => {},
  onOpen = () => {},
  socketFactory = (u) => new globalThis.WebSocket(u),
  setTimer = (fn, ms) => globalThis.setTimeout(fn, ms),
  random = Math.random,
} = {}) {
  let socket = null;
  let attempt = 0;
  let stopped = false;

  const schedule = () => {
    if (stopped) return;
    const delay = backoffDelay(attempt, random);
    attempt += 1;
    onStatus(`reconnecting in ${delay} ms`);
    setTimer(open, delay);
  };

  function open() {
    if (stopped || socket) return; // never more than one socket object alive
    let ws;
    try {
      ws = socketFactory(url);
    } catch (err) {
      onStatus(`socket failed: ${err && err.message}`);
      schedule();
      return;
    }
    socket = ws;
    ws.onopen = () => {
      attempt = 0;
      onStatus("connected");
      onOpen(api);
    };
    ws.onmessage = (event) => onMessage(event && event.data);
    ws.onerror = () => onStatus("socket error");
    ws.onclose = () => {
      if (socket !== ws) return;
      socket = null;
      onStatus("disconnected");
      schedule();
    };
  }

  const api = {
    open,
    isOpen: () => Boolean(socket) && socket.readyState === 1,
    send(message) {
      if (!api.isOpen()) return Promise.reject(new Error("socket is not open"));
      socket.send(JSON.stringify(message));
      return Promise.resolve();
    },
    close() {
      stopped = true;
      const ws = socket;
      socket = null;
      if (ws) ws.close();
    },
  };
  return api;
}
