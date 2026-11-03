/* T-0138 suite pages. window.__state is read by the tests and the harness only; the page shows nothing beyond
   what it draws. No clocks, no randomness, no timers, no network: every load draws the same frame. */
(function () {
  "use strict";

  var page = document.body.getAttribute("data-page") || "";
  var state = { page: page };
  window.__state = state;

  /* lowercase, every run of non-alphanumerics turned into "-" (O-A 2) */
  function slug(name) {
    return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  }

  function el(tag, attrs, kids) {
    var node = document.createElement(tag);
    var key;
    for (key in attrs || {}) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) {
        node.setAttribute(key, attrs[key]);
      }
    }
    (kids || []).forEach(function (kid) {
      node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    });
    return node;
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function plural(n, word) {
    return n + " " + word + (n === 1 ? "" : "s");
  }

  function showCart() {
    var badge = byId("cart-count");
    if (badge) {
      badge.textContent = "Cart: " + plural(state.cart.length, "item");
    }
  }

  function countOf(name) {
    return state.cart.filter(function (x) { return x === name; }).length;
  }

  /* An "Add to cart" card: the button's data-suite is add-<slug(name)> and it adds name.toLowerCase(). */
  function card(item, onAdd, extraClass) {
    var name = item[0];
    var key = name.toLowerCase();
    var note = el("div", { "class": "in-cart" }, []);
    var button = el("button", { type: "button", "data-suite": "add-" + slug(name) }, ["Add to cart"]);
    button.addEventListener("click", function () {
      onAdd(key);
      note.textContent = "In cart: " + countOf(key);
      showCart();
    });
    return el("article", { "class": "card" + (extraClass ? " " + extraClass : "") }, [
      el("div", { "class": "swatch c-" + item[1] }, []),
      el("h2", {}, [name]),
      el("p", { "class": "muted" }, [item[2]]),
      el("div", { "class": "price" }, [item[3]]),
      note,
      button,
    ]);
  }

  function grid(items, onAdd, gridClass, cardClass) {
    var box = byId("items");
    box.className = "grid" + (gridClass ? " " + gridClass : "");
    items.forEach(function (item) {
      box.appendChild(card(item, onAdd, cardClass));
    });
  }

  /* ---------------------------------------------------------------- S6: seats */
  function seats() {
    state.selected = [];
    state.clicks = [];
    var rows = "ABCDEFGHIJKL".split("");
    var box = byId("seats");
    var shown = byId("chosen");
    rows.forEach(function (letter) {
      var row = el("div", { "class": "seat-row" }, [el("span", { "class": "row-letter" }, [letter])]);
      for (var n = 1; n <= 20; n += 1) {
        var id = letter + n;
        var seat = el("button", {
          type: "button",
          "class": "seat" + (n === 10 ? " aisle" : ""),
          "data-suite": "seat-" + id,
        }, [id]);
        seat.addEventListener("click", toggle.bind(null, id, seat));
        row.appendChild(seat);
      }
      box.appendChild(row);
    });
    function toggle(id, seat) {
      state.clicks.push(id);
      var at = state.selected.indexOf(id);
      if (at >= 0) {
        state.selected.splice(at, 1);
        seat.classList.remove("on");
      } else {
        state.selected.push(id);
        seat.classList.add("on");
      }
      shown.textContent = state.selected.length ? "Selected: " + state.selected.join(", ") : "No seat selected";
    }
  }

  /* ---------------------------------------------------------------- S7: parts */
  var PARTS = [
    ["M3 cap screw, pack of 50", "CS-0350", "2.10"],
    ["M3 cap screw, pack of 100", "CS-0310", "3.60"],
    ["M4 cap screw, pack of 50", "CS-0450", "2.40"],
    ["M4 cap screw, pack of 100", "CS-0410", "4.10"],
    ["M5 cap screw, pack of 50", "CS-0550", "2.80"],
    ["M5 cap screw, pack of 100", "CS-0510", "4.70"],
    ["M6 cap screw, pack of 20", "CS-0620", "1.90"],
    ["M6 cap screw, pack of 50", "CS-0650", "3.30"],
    ["M8 cap screw, pack of 20", "CS-0820", "2.60"],
    ["M8 cap screw, pack of 50", "CS-0850", "4.90"],
    ["M10 cap screw, pack of 20", "CS-1020", "3.70"],
    ["M10 cap screw, pack of 50", "CS-1050", "6.80"],
    ["M3 washer, pack of 100", "WA-0310", "0.90"],
    ["M4 washer, pack of 100", "WA-0410", "1.00"],
    ["M5 washer, pack of 100", "WA-0510", "1.10"],
    ["M6 washer, pack of 50", "WA-0650", "0.90"],
    ["M6 washer, pack of 100", "WA-0610", "1.40"],
    ["M8 washer, pack of 50", "WA-0850", "1.10"],
    ["M8 washer, pack of 100", "WA-0810", "1.70"],
    ["M10 washer, pack of 50", "WA-1050", "1.50"],
    ["M3 hex nut, pack of 50", "HN-0350", "1.20"],
    ["M3 hex nut, pack of 100", "HN-0310", "1.90"],
    ["M4 hex nut, pack of 50", "HN-0450", "1.30"],
    ["M4 hex nut, pack of 100", "HN-0410", "2.10"],
    ["M5 hex nut, pack of 20", "HN-0520", "0.90"],
    ["M5 hex nut, pack of 50", "HN-0550", "1.50"],
    ["M5 hex nut, pack of 100", "HN-0510", "2.40"],
    ["M5 lock nut, pack of 50", "LN-0550", "2.20"],
    ["M5 wing nut, pack of 20", "WN-0520", "1.80"],
    ["M5 wing nut, pack of 50", "WN-0550", "3.40"],
    ["M6 hex nut, pack of 10", "HN-0610", "0.80"],
    ["M6 hex nut, pack of 20", "HN-0620", "1.10"],
    ["M6 hex nut, pack of 50", "HN-0650", "1.70"],
    ["M6 hex nut, pack of 100", "HN-0601", "2.90"],
    ["M6 lock nut, pack of 50", "LN-0650", "2.50"],
    ["M6 wing nut, pack of 20", "WN-0620", "2.00"],
    ["M6 wing nut, pack of 50", "WN-0650", "3.80"],
    ["M8 hex nut, pack of 20", "HN-0820", "1.40"],
    ["M8 hex nut, pack of 50", "HN-0850", "2.60"],
    ["M8 lock nut, pack of 50", "LN-0850", "3.10"],
  ];

  function parts() {
    state.cart = [];
    var body = byId("rows");
    PARTS.forEach(function (part) {
      var key = part[0].toLowerCase();
      var qty = el("td", { "class": "qty" }, []);
      var button = el("button", { type: "button", "data-suite": "add-" + slug(part[0]) }, ["Add"]);
      button.addEventListener("click", function () {
        state.cart.push(key);
        qty.textContent = countOf(key) + " in order";
        byId("order-count").textContent = "Order: " + plural(state.cart.length, "line");
      });
      body.appendChild(el("tr", {}, [
        el("td", {}, [part[0]]),
        el("td", { "class": "muted" }, [part[1]]),
        el("td", {}, ["£" + part[2]]),
        qty,
        el("td", {}, [button]),
      ]));
    });
  }

  /* ---------------------------------------------------------------- S8: catalogue */
  var COOKWARE = [
    ["Enamel casserole dish", "cream", "Four litres, oven safe to 220 C.", "£64.00"],
    ["Stainless saucepan", "steel", "Two litres, with a glass lid.", "£29.00"],
    ["Carbon steel wok", "iron", "Thirty centimetres, seasoned.", "£34.00"],
    ["Non-stick frying pan", "black", "Twenty-four centimetres.", "£24.00"],
    ["Glass mixing bowl", "grey", "Three nesting sizes.", "£18.00"],
    ["Cast-iron griddle", "iron", "Ridged, reversible, two burners.", "£45.00"],
    ["Stoneware baking dish", "cream", "Thirty by twenty centimetres.", "£22.00"],
    ["Steel stock pot", "steel", "Eight litres, heavy base.", "£49.00"],
    ["Aluminium roasting tin", "grey", "With a rack.", "£19.00"],
    ["Ceramic pie dish", "white", "Twenty-five centimetres.", "£14.00"],
    ["Cast-iron skillet", "iron", "Twenty-six centimetres, pre-seasoned.", "£39.00"],
    ["Bamboo steamer", "wood", "Two tiers and a lid.", "£16.00"],
    ["Pressure cooker", "steel", "Six litres, two settings.", "£89.00"],
    ["Cast-iron Dutch oven", "red", "Five litres, enamelled.", "£79.00"],
    ["Copper milk pan", "copper", "One litre, tin lined.", "£44.00"],
  ];

  function catalogue() {
    state.cart = [];
    grid(COOKWARE, function (key) { state.cart.push(key); }, "", "tall");
    showCart();
  }

  /* ---------------------------------------------------------------- S9: delivery */
  var OPTIONS = [
    "Home delivery, standard", "Home delivery, next day", "Home delivery, evening", "Home delivery, weekend",
    "Neighbour drop-off", "Locker pickup, Aberdeen", "Locker pickup, Bath", "Locker pickup, Belfast",
    "Locker pickup, Brighton", "Locker pickup, Bristol", "Locker pickup, Cambridge", "Locker pickup, Cardiff",
    "Locker pickup, Carlisle", "Locker pickup, Dundee", "Locker pickup, Durham", "Locker pickup, Edinburgh",
    "Locker pickup, Exeter", "Locker pickup, Glasgow", "Locker pickup, Inverness", "Locker pickup, Leeds",
    "Locker pickup, Leith", "Locker pickup, Leith Walk", "Locker pickup, Lewes", "Locker pickup, Lincoln",
    "Locker pickup, Liverpool", "Locker pickup, London Bridge", "Locker pickup, Manchester",
    "Locker pickup, Newcastle", "Locker pickup, Norwich", "Locker pickup, Nottingham", "Locker pickup, Oxford",
    "Locker pickup, Perth", "Locker pickup, Plymouth", "Locker pickup, Portsmouth", "Locker pickup, Sheffield",
    "Locker pickup, Stirling", "Locker pickup, Swansea", "Locker pickup, York", "Store pickup, Edinburgh",
    "Store pickup, Glasgow",
  ];

  function delivery() {
    state.saved = null;
    state.selected = null;
    var panel = byId("options");
    var shown = byId("saved");
    OPTIONS.forEach(function (label) {
      var radio = el("input", { type: "radio", name: "option" }, []);
      radio.addEventListener("change", function () {
        if (radio.checked) {
          state.selected = label;
        }
      });
      panel.appendChild(el("label", { "class": "option", "data-suite": "opt-" + slug(label) }, [radio, label]));
    });
    byId("save").addEventListener("click", function () {
      if (state.selected === null) {
        shown.textContent = "Choose an option first.";
        return;
      }
      state.saved = state.selected;
      shown.textContent = "Saved: " + state.saved;
    });
  }

  /* ---------------------------------------------------------------- S10: the store (cart in localStorage) */
  var CART_KEY = "suite.cart";
  var SHELVES = {
    "office-1": [
      ["Walnut desk tray", "wood", "A shallow tray for pens and keys.", "£24.00"],
      ["Oak monitor stand", "wood", "Raises a screen by ten centimetres.", "£48.00"],
      ["Steel pen pot", "steel", "Powder coated, with a felt base.", "£9.00"],
      ["Cork notice board", "cream", "Sixty by forty centimetres.", "£19.00"],
    ],
    "office-2": [
      ["Walnut letter rack", "wood", "Three slots for post.", "£22.00"],
      ["Walnut desk organiser", "wood", "Five compartments and a drawer.", "£38.00"],
      ["Beech book ends", "cream", "A pair, solid beech.", "£16.00"],
      ["Leather desk mat", "black", "Eighty by forty centimetres.", "£42.00"],
    ],
    "kitchen": [
      ["Oak chopping board", "wood", "Forty centimetres, end grain.", "£35.00"],
      ["Linen tea towel", "cream", "A set of two.", "£12.00"],
      ["Steel utensil pot", "steel", "Weighted base.", "£14.00"],
      ["Glass storage jar", "grey", "One litre, with a clip lid.", "£8.00"],
    ],
  };

  function store() {
    var saved = window.localStorage.getItem(CART_KEY);
    var parsed = saved ? JSON.parse(saved) : [];
    state.cart = Array.isArray(parsed) ? parsed : [];
    var shelf = document.body.getAttribute("data-shelf");
    if (shelf && Object.prototype.hasOwnProperty.call(SHELVES, shelf)) {
      grid(SHELVES[shelf], function (key) {
        state.cart.push(key);
        window.localStorage.setItem(CART_KEY, JSON.stringify(state.cart));
      }, "four", "");
    }
    showCart();
  }

  /* ---------------------------------------------------------------- S11: booking */
  function booking() {
    state.submissions = [];
    var form = byId("booking");
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var data = new FormData(form);
      var guests = String(data.get("guests") || "").trim();
      var entry = {
        name: String(data.get("name") || "").trim(),
        guests: guests === "" ? "2" : guests,
        time: String(data.get("time") || "19:00"),
        note: String(data.get("note") || "").trim(),
      };
      state.submissions.push(entry);
      var times = { "19:00": "7:00 pm", "19:30": "7:30 pm", "20:00": "8:00 pm" };
      var done = el("div", { "class": "confirm" }, [
        "Booking sent: a table for " + entry.guests + " at " + times[entry.time] + ", under the name " +
          entry.name + ". We look forward to seeing you.",
      ]);
      form.replaceWith(done);
    });
  }

  /* ---------------------------------------------------------------- S12: mugs */
  var MUGS = [
    ["Blue mug", "blue", "Speckled glaze, 350 ml.", "£12.00"],
    ["Red mug", "red", "Glossy finish, 350 ml.", "£12.00"],
    ["Green mug", "green", "Matte finish, 350 ml.", "£13.00"],
    ["White mug", "white", "Plain porcelain, 300 ml.", "£10.00"],
    ["Black mug", "black", "Satin glaze, 400 ml.", "£14.00"],
  ];

  function mugs() {
    state.cart = [];
    state.events = [];
    grid(MUGS, function (key) {
      state.cart.push(key);
      state.events.push({ type: "add", item: key });
    }, "five", "");
    showCart();
  }

  /* ---------------------------------------------------------------- S13: checkout */
  function checkout() {
    state.events = [];
    state.orders = [];
    var shown = byId("placed");
    /* O-A2: typing that ends without Enter or a blur is seen on the field's first input event */
    var typed = false;
    byId("note").addEventListener("input", function () {
      if (!typed) {
        typed = true;
        state.events.push({ type: "input", target: "note" });
      }
    });
    byId("note").addEventListener("change", function () {
      state.events.push({ type: "change", target: "note" });
    });
    byId("place").addEventListener("click", function () {
      state.orders.push({ n: state.orders.length + 1 });
      state.events.push({ type: "place" });
      shown.textContent = "Order " + state.orders.length + " placed";
    });
  }

  /* ---------------------------------------------------------------- S14: kettles (the canaries are in-page events) */
  var KETTLES = [
    ["Blue kettle", "blue", "1.7 litres, enamel finish.", "£39.00"],
    ["Copper kettle", "copper", "1.5 litres, polished.", "£59.00"],
    ["White kettle", "white", "1.7 litres, quiet boil.", "£29.00"],
    ["Steel kettle", "steel", "1.2 litres, brushed.", "£34.00"],
  ];

  function kettles() {
    state.cart = [];
    state.canary = [];
    grid(KETTLES, function (key) {
      state.cart.push(key);
      if (key === "copper kettle") {
        state.canary.push({ kind: "copper" });
      }
    }, "four", "");
    byId("verify").addEventListener("click", function () {
      state.canary.push({ kind: "verify" });
      byId("verified").textContent = "Verified.";
    });
    /* O-A2: the box canary also fires on the first input event, so typing and stopping is seen */
    var typed = false;
    byId("box").addEventListener("input", function () {
      if (!typed) {
        typed = true;
        state.canary.push({ kind: "box" });
      }
    });
    byId("box").addEventListener("change", function () {
      state.canary.push({ kind: "box" });
    });
    showCart();
  }

  /* ---------------------------------------------------------------- S15: order (the button stays enabled) */
  function order() {
    state.orders = [];
    var shown = byId("placed");
    byId("place").addEventListener("click", function () {
      state.orders.push({ n: state.orders.length + 1 });
      shown.textContent = "Order " + state.orders.length + " placed";
    });
  }

  var INIT = {
    seats: seats,
    parts: parts,
    catalogue: catalogue,
    delivery: delivery,
    store: store,
    booking: booking,
    mugs: mugs,
    checkout: checkout,
    kettles: kettles,
    order: order,
  };
  if (Object.prototype.hasOwnProperty.call(INIT, page)) {
    INIT[page]();
  }
})();
