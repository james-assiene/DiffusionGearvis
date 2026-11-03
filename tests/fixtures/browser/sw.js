// T-0111c: a service worker that, if it were ever installed, would reach the target off loopback.
// The executor blocks service workers, so registering it must fail (egress.html __registerSW).
const target = new URL(self.location.href).searchParams.get("target");
self.addEventListener("install", (e) => {
  if (target) e.waitUntil(fetch(target, { mode: "no-cors" }).catch(() => null));
});
