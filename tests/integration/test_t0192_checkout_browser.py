"""T-0192 group A end to end: stop at checkout in web mode, with the real executor, a real headless Chromium (sandbox
on), the real ``EgressProxy`` with T-0160's injected resolver and connector, and loopback fixtures posing as
``shop.example`` / ``pay.example`` (docs/tasks/T-0192.md, A1-A8, "Tests to write", group A, browser rows (a)-(h)).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1``
(``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0192_checkout_browser.py -q -m browser``). The rig
(TLS fixture server, canaries, start pages, ``Net``, the certificate relaxation) is T-0160's, imported from
``tests/integration/test_computer_web_browser.py``; its fixture server gained the nested ``T0192_ROUTES`` (the
``t0192_*.html`` pages, ``/checkout/entry`` -> 302 -> ``/signin.html?openid.return_to=...%2Fcheckout...`` and
``/checkout/entry2`` -> 302 -> ``/checkout/<token>/pay``; ``tests/fixtures/browser/web/MANIFEST.md``).

**Readings (the test-author's, reported):**

- Row (b) as the task lists it says "next ``click_at`` refused" after the ``/checkout/entry`` -> ``/signin.html``
  hop. A2 says reading drains the store and A4's post-settle read is a read, and A5 says the
  ``/ap/signin?openid.return_to=...%2Fcheckout...`` shape is no refusal; so after that hop the next call's read
  returns only the sign-in URL and the call is *not* refused. These rows pin A2/A4/A5: (b) the arrival is
  ``checkout: True`` and the next click on the sign-in page executes; (b2) a hop that *lands* on a checkout page
  (``/checkout/entry2`` -> ``/checkout/<token>/pay``) has the next ``click_at`` refused.
- Row (g): after ``reset_page()`` from a checkout page the next action executes (B5: Restart is a way out, so the
  Restart's own navigation leaves nothing that refuses the next call).

**Regime.** Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, no real network, n = 1 per row, one
browser per row.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests.integration import test_computer_web_browser as web_rig
from tests.integration.test_computer_web_browser import (
    ADDRESS_FIELD,
    OPENED,
    T0192_TOKEN,
    Net,
    Rig,
    click,
    execute,
    frame_by_url,
    goto,
    inv,
    type_into,
    wait_url,
    web_executor,
)

#: T-0160's fixtures, re-bound here so pytest registers them in this module (``_browser_layer`` is the autouse gate)
_browser_layer = web_rig._browser_layer
rig = web_rig.rig
launches = web_rig.launches

pytestmark = [pytest.mark.integration, pytest.mark.browser]

SHOP = "https://shop.example"
CART = f"{SHOP}/cart.html"
CHECKOUT = f"{SHOP}/checkout/{T0192_TOKEN}/pay"
SIGNIN_PREFIX = f"{SHOP}/signin.html?openid.return_to="
#: A3, verbatim
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"


def checkout_reason() -> str:
    value = getattr(executor_mod, "CHECKOUT_REASON", None)
    assert value is not None, "T-0192 A3: executor.CHECKOUT_REASON does not exist yet"
    assert value == CHECKOUT_REASON, value
    return str(value)


def assert_checkout(out: ToolOutcome) -> None:
    """A4: an executed action that reached a checkout carries ``resolved["checkout"] is True`` (exactly)."""
    assert out.status == "executed", out
    assert out.resolved.get("checkout") is True, (
        f"T-0192 A4: an action that reached a checkout carries no resolved['checkout'] = True: {out.resolved}"
    )


def assert_no_checkout(out: ToolOutcome) -> None:
    """A5/A6: executed, no refusal by this guard, and no ``checkout`` key."""
    assert out.status == "executed", out
    assert out.reason != CHECKOUT_REASON, out
    assert "checkout" not in out.resolved, out.resolved


def assert_refused_here(out: ToolOutcome) -> None:
    assert (out.status, out.reason) == ("refused", checkout_reason()), out


async def on_cart(ex: Any) -> None:
    await goto(ex, CART)
    assert ex.driver.page.url == CART, f"premise: the cart loaded ({ex.driver.page.url})"


async def wait_prefix(ex: Any, prefix: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not ex.driver.page.url.startswith(prefix):
        await asyncio.sleep(0.1)
    return str(ex.driver.page.url)


async def eval_(ex: Any, js: str) -> Any:
    return await ex.driver.page.evaluate(js)


# =====================================================================================================
# A2: the real driver's seam
# =====================================================================================================


async def test_a2_main_frame_urls_drains_and_includes_redirect_hops(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """A2 / row (b)'s premise: ``PlaywrightDriver.main_frame_urls()`` is synchronous, returns every main-frame
    navigation request URL since the previous call (the ``/checkout/entry`` redirect hop included) with
    ``page.url`` last, and reading drains the store (a second read is ``page.url`` alone)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        read = getattr(type(ex.driver), "main_frame_urls", None)
        assert read is not None, "T-0192 A2: PlaywrightDriver has no main_frame_urls yet"
        await on_cart(ex)
        ex.driver.main_frame_urls()  # drain what the harness navigation recorded
        await ex.driver.page.click("#proceed")
        await wait_prefix(ex, SIGNIN_PREFIX)
        assert ex.driver.page.url.startswith(SIGNIN_PREFIX), ex.driver.page.url
        assert ("shop.example", "/checkout/entry") in rig.hits, "premise: the hop was served"
        got = ex.driver.main_frame_urls()
        assert isinstance(got, tuple) and all(isinstance(u, str) for u in got), got
        assert got[-1] == ex.driver.page.url, got
        assert any(u.startswith(f"{SHOP}/checkout/entry") for u in got), (
            f"A2: the redirect hop /checkout/entry is not among the returned URLs: {got}"
        )
        again = ex.driver.main_frame_urls()
        assert again == (ex.driver.page.url,), f"A2: reading drains the store: {again}"


# =====================================================================================================
# (a) add to cart, (e) /checkout-faq.html, (f) a cross-origin iframe: never refused, no checkout key
# =====================================================================================================


async def test_a_add_to_cart_is_open(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """(a) / A5: on a product page "Add to cart" goes to ``/cart.html``: executed, no ``checkout``; on the cart a
    click and a scroll execute too (the guard is not too wide)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"{SHOP}/t0192/dp/kettle")
        out = await click(ex, "#add")
        assert_no_checkout(out)
        assert await wait_url(ex, CART) == CART
        out = await click(ex, "#bump")
        assert_no_checkout(out)
        assert await eval_(ex, "window.__bumps") == 1
        assert_no_checkout(await execute(ex, inv("scroll", None, direction="down")))


async def test_e_checkout_faq_is_not_a_checkout(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """(e) / A5: ``/checkout-faq.html`` is reached without ``checkout`` and a click there executes."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        out = await click(ex, "#faq")
        assert_no_checkout(out)
        assert await wait_url(ex, f"{SHOP}/checkout-faq.html") == f"{SHOP}/checkout-faq.html"
        out = await click(ex, "#bump")
        assert_no_checkout(out)
        assert await eval_(ex, "window.__bumps") == 1


async def test_f_cross_origin_checkout_iframe_is_ignored(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(f) / A6: a normal page framing a cross-origin iframe at ``https://pay.example/checkout/<token>/frame`` (a
    payment or CAPTCHA widget's shape) is not refused by this guard: only main-frame URLs are read."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"{SHOP}/t0192/framed.html")
        await frame_by_url(ex, f"/checkout/{T0192_TOKEN}/frame")
        assert ("pay.example", f"/checkout/{T0192_TOKEN}/frame") in rig.hits, (
            "premise: the iframe loaded"
        )
        out = await click(ex, "#bump")
        assert_no_checkout(out)
        assert await eval_(ex, "window.__bumps") == 1
        assert_no_checkout(await execute(ex, inv("scroll", None, direction="down")))


# =====================================================================================================
# (b), (b2), (c), (d): arrival, and the refusal that follows
# =====================================================================================================


async def test_b_checkout_hop_to_signin_is_an_arrival(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(b), the demo's shape: "Proceed to checkout" goes to ``/checkout/entry``, which redirects (302) to the sign-in
    wall; the click is executed with ``checkout: True`` although only the redirect hop was a checkout. Then (A2's
    drain, A5's ``/ap/signin?...%2Fcheckout...`` row; module docstring reading) a click on the sign-in page
    executes without ``checkout``."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        out = await click(ex, "#proceed")
        assert ("shop.example", "/checkout/entry") in rig.hits, "premise: the hop was served"
        assert ex.driver.page.url.startswith(SIGNIN_PREFIX), ex.driver.page.url
        assert_checkout(out)
        nxt = await click(ex, "#bump")
        assert_no_checkout(nxt)
        assert await eval_(ex, "window.__bumps") == 1


async def test_b3_checkout_as_middle_hop_is_an_arrival(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(b), Amazon's real chain (review R-A): a cart form submitted by ``type_at`` with Enter POSTs to
    ``/gp/cart/submit`` (not a checkout), which 302s to ``/checkout/entry`` (the checkout), which 302s to the sign-in
    wall (not a checkout). The checkout is neither the first nor the last URL of the read, so a guard that looks at
    only one end of it (``urls[:1]`` or ``urls[-1:]``) misses it; A4 says any of them counts: ``checkout: True``."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        form = f"{SHOP}/t0192/cart-form.html"
        await goto(ex, form)
        assert ex.driver.page.url == form, f"premise: the cart form loaded ({ex.driver.page.url})"
        out = await type_into(ex, "#qty", "2", enter=True)
        await wait_prefix(ex, SIGNIN_PREFIX)
        assert ("shop.example", "/gp/cart/submit") in rig.hits, "premise: the form was submitted"
        assert ("shop.example", "/checkout/entry") in rig.hits, "premise: the middle hop was served"
        assert ex.driver.page.url.startswith(SIGNIN_PREFIX), ex.driver.page.url
        assert_checkout(out)


async def test_b2_hop_landing_on_checkout_then_refused(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(b) with the landing on a checkout page: "Buy now" goes through ``/checkout/entry2`` (302) to
    ``/checkout/<token>/pay``: ``checkout: True``; the next ``click_at`` on "Place your order", a ``type_at`` and a
    ``scroll`` are refused with CHECKOUT_REASON and the page received nothing."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        out = await click(ex, "#buynow")
        assert await wait_url(ex, CHECKOUT) == CHECKOUT
        assert_checkout(out)
        assert_refused_here(await click(ex, "#place"))
        assert_refused_here(await type_into(ex, "#note", "leave at the door"))
        y = await eval_(ex, "window.scrollY")
        assert_refused_here(await execute(ex, inv("scroll", None, direction="down")))
        assert_refused_here(await execute(ex, inv("press_key", None, combo="Enter")))
        assert await eval_(ex, "[window.__clicks, window.__inputs, window.scrollY]") == [0, 0, y]
        assert ex.driver.page.url == CHECKOUT


async def test_c_checkout_popup_is_an_arrival(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """(c) / A4: a link opening ``/checkout/<token>/pay`` in a new window; the settle adopts the popup and the click
    carries ``checkout: True``; the next click there is refused."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        opener = ex.driver.page
        out = await click(ex, "#popup")
        assert ex.driver.page is not opener, "premise: the settle adopted the popup"
        assert await wait_url(ex, CHECKOUT) == CHECKOUT
        assert_checkout(out)
        assert_refused_here(await click(ex, "#place"))
        assert await eval_(ex, "window.__clicks") == 0


async def test_d_push_state_to_checkout_is_an_arrival(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(d) / A4: a same-document ``history.pushState`` to ``/checkout`` (no request) is an arrival through
    ``page.url``; the next click is refused (the cart's own button receives nothing)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        out = await click(ex, "#push")
        assert ex.driver.page.url == f"{SHOP}/checkout", "premise: pushState ran"
        assert_checkout(out)
        assert_refused_here(await click(ex, "#bump"))
        assert await eval_(ex, "window.__bumps") == 0


# =====================================================================================================
# (g) the ways out: Restart and Ctrl+L / Ctrl+T
# =====================================================================================================


async def test_g_restart_from_checkout_reaches_web_html(
    rig: Rig,
    launches: list[dict[str, Any]],
) -> None:
    """(g) / A8 / B5: ``reset_page()`` from a checkout page reaches ``/web.html`` (Restart is never refused), and
    the next action there executes (module docstring reading)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, CHECKOUT)
        assert ex.driver.page.url == CHECKOUT, "premise: the checkout page loaded"
        await asyncio.wait_for(ex.reset_page(), 30)
        assert ex.driver.page.url == f"{rig.start}/web.html", ex.driver.page.url
        out = await type_into(ex, ADDRESS_FIELD, "shop.example")
        assert_no_checkout(out)


@pytest.mark.parametrize("combo", ["Ctrl+L", "Ctrl+T"])
async def test_g_start_page_keys_from_checkout(
    rig: Rig,
    launches: list[dict[str, Any]],
    combo: str,
) -> None:
    """(g) / A3: on a checkout page ``Ctrl+L``/``Ctrl+T`` execute and open ``/web.html`` (no ``checkout`` key: the
    post-settle read sees the start page), and the next action there executes."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, CHECKOUT)
        assert ex.driver.page.url == CHECKOUT, "premise: the checkout page loaded"
        out = await execute(ex, inv("press_key", None, combo=combo))
        assert (out.status, out.action) == ("executed", OPENED), out
        assert "checkout" not in out.resolved, out.resolved
        assert await wait_url(ex, f"{rig.start}/web.html") == f"{rig.start}/web.html"
        assert_no_checkout(await type_into(ex, ADDRESS_FIELD, "shop.example"))


# =====================================================================================================
# (h) A8: no URL leaves the executor
# =====================================================================================================


async def test_h_token_nowhere(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """(h) / A8: through an arrival by redirect hop, a refusal, a popup arrival and Ctrl+L, the checkout path's unique
    token appears in no outcome (``repr``: action, reason, resolved), no ``take_events()`` entry, and no
    ``egress_log`` entry."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        seen: list[Any] = []
        driver = ex.driver
        orig = driver.take_events

        def take_events() -> list[dict[str, Any]]:
            got = orig()
            seen.extend(got)
            return got

        driver.take_events = take_events
        outs: list[ToolOutcome] = []
        await on_cart(ex)
        outs.append(await click(ex, "#buynow"))
        await wait_url(ex, CHECKOUT)
        outs.append(await click(ex, "#place"))
        outs.append(await execute(ex, inv("press_key", None, combo="Ctrl+L")))
        await on_cart(ex)
        outs.append(await click(ex, "#popup"))
        await wait_url(ex, CHECKOUT)
        outs.append(await execute(ex, inv("scroll", None, direction="down")))
        assert [o.status for o in outs] == [
            "executed",
            "refused",
            "executed",
            "executed",
            "refused",
        ], outs
        assert outs[0].resolved.get("checkout") is True and outs[3].resolved.get("checkout") is True
        blob = repr(outs) + json.dumps(seen, default=str) + json.dumps(ex.egress_log)
        for needle in (T0192_TOKEN, "/checkout/", "/pay"):
            assert needle not in blob, f"A8: {needle!r} left the executor: {blob[:2000]}"
