// The cart of the shop and deals pages (T-0111g A5). It lives in this page only (nothing is stored or sent), so
// every page load starts empty. window.__state.cart is the list of added item names, lowercase, in order.
(function () {
  "use strict";
  window.__state = { page: document.body.dataset.page, cart: [] };

  function render() {
    const n = window.__state.cart.length;
    for (const el of document.querySelectorAll("[data-cart-count]")) {
      el.textContent = "Cart: " + n + (n === 1 ? " item" : " items");
    }
  }

  document.addEventListener("click", function (event) {
    const button = event.target.closest("button[data-item]");
    if (!button) return;
    window.__state.cart.push(button.dataset.item);
    render();
  });

  render();
})();
