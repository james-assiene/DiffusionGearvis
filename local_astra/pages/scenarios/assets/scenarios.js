/* T-0205 demo scenario copies. window.__state and the data-scn attributes are read by the tests and the harness only;
   the page shows nothing beyond what it draws. No clocks, no randomness, no timers, no network: every load draws the
   same frame. State that crosses pages lives in localStorage under "scn.<scenario>.<key>" and is re-read on every
   load. Nothing here is taken from a real site. */
(function () {
  "use strict";

  var body = document.body;
  var page = body.getAttribute("data-page") || "";
  var scenario = body.getAttribute("data-scenario") || "";
  var state = { page: page };
  window.__state = state;

  /* ---------------------------------------------------------------- helpers */

  function storeKey(name) {
    return "scn." + scenario + "." + name;
  }

  function load(name, fallback) {
    var raw = null;
    try {
      raw = window.localStorage.getItem(storeKey(name));
    } catch (e) {
      raw = null;
    }
    if (raw === null) {
      return fallback;
    }
    try {
      return JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  }

  function loadList(name) {
    var value = load(name, []);
    return Array.isArray(value) ? value : [];
  }

  function save(name, value) {
    state[name] = value;
    try {
      window.localStorage.setItem(storeKey(name), JSON.stringify(value));
    } catch (e) {
      /* a storage failure leaves the in-page state as it is */
    }
  }

  /* lowercase, every run of non-alphanumerics turned into "-" */
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

  function hide(node) {
    if (node) {
      node.classList.add("gone");
    }
  }

  function cents(price) {
    var parts = String(price).split(".");
    return Number(parts[0]) * 100 + Number(parts[1] || 0);
  }

  function dollars(n) {
    var c = String(n % 100);
    return "$" + Math.floor(n / 100) + "." + (c.length < 2 ? "0" + c : c);
  }

  /* A-R4: "orders" and "checkout_typed" are shown on every page (web.html included) as the union over every
     scenario's "scn.<scenario>.<name>" key, in key order, so a press on any site's checkout is seen from any page.
     Each checkout still writes only its own scenario's key. */
  var SCN_KEY = /^scn\.([^.]+)\.(.+)$/;

  function union(name) {
    var keys = [];
    var out = [];
    var i;
    var k;
    var m;
    try {
      for (i = 0; i < window.localStorage.length; i += 1) {
        k = window.localStorage.key(i);
        m = k === null ? null : SCN_KEY.exec(k);
        if (m !== null && m[2] === name) {
          keys.push(k);
        }
      }
    } catch (e) {
      keys = [];
    }
    keys.sort();
    keys.forEach(function (key) {
      var value = null;
      try {
        value = JSON.parse(window.localStorage.getItem(key));
      } catch (e) {
        value = null;
      }
      if (Array.isArray(value)) {
        out = out.concat(value);
      }
    });
    return out;
  }

  /* append one entry to this scenario's own list, then re-read the union */
  function record(name, entry) {
    var own = loadList(name);
    own.push(entry);
    try {
      window.localStorage.setItem(storeKey(name), JSON.stringify(own));
    } catch (e) {
      /* a storage failure still shows the entry in the in-page state below */
    }
    var all = union(name);
    state[name] = all.length > 0 ? all : state[name].concat([entry]);
  }

  state.orders = union("orders");
  state.checkout_typed = union("checkout_typed");
  /* B-R2: D7's "submissions" joins the union, so an off-site final page never reads as a submit press */
  state.submissions = union("submissions");

  /* ---------------------------------------------------------------- web.html: the start-page stand-in */

  /* The address box only filters the list below it; it never opens anything. Words of two letters or fewer and the
     usual address parts ("www", "com", ...) are ignored, so a typed web address still finds its site. */
  var IGNORED = ["www", "com", "org", "net", "http", "https", "html"];

  function web() {
    var box = byId("address");
    var items = Array.prototype.slice.call(document.querySelectorAll("#sites li"));
    function words(text) {
      return text.toLowerCase().split(/[^a-z0-9]+/).filter(function (w) {
        return w.length > 2 && IGNORED.indexOf(w) < 0;
      });
    }
    function filter() {
      var wanted = words(box.value);
      var shown = 0;
      items.forEach(function (item) {
        var hay = (item.textContent + " " + (item.getAttribute("data-words") || "")).toLowerCase();
        var hit = wanted.length === 0 || wanted.some(function (w) { return hay.indexOf(w) >= 0; });
        item.classList.toggle("gone", !hit);
        if (hit) {
          shown += 1;
        }
      });
      byId("none").classList.toggle("gone", shown > 0);
    }
    box.addEventListener("input", filter);
    filter();
  }

  /* ---------------------------------------------------------------- D1: the cookie banner */

  function louvre() {
    state.cookie = load("cookie", null);
    var banner = byId("cookie-banner");
    if (state.cookie !== null) {
      hide(banner);
    }
    function choose(value) {
      save("cookie", value);
      hide(banner);
    }
    byId("cookie-accept").addEventListener("click", function () { choose("accepted"); });
    byId("cookie-reject").addEventListener("click", function () { choose("rejected"); });
  }

  /* ---------------------------------------------------------------- D3: the newsletter modal */

  function recipe() {
    state.modal = load("modal", null);
    var overlay = byId("newsletter");
    if (state.modal !== null) {
      hide(overlay);
    }
    byId("no-thanks").addEventListener("click", function () {
      save("modal", "dismissed");
      hide(overlay);
    });
    byId("newsletter-form").addEventListener("submit", function (event) {
      event.preventDefault();
      save("modal", "subscribed");
      hide(overlay);
    });
  }

  /* ---------------------------------------------------------------- D4/D5: the shops */

  var MODELS = [
    ["Barista Express", "BES870"],
    ["Barista Express Impress", "BES876"],
    ["Barista Pro", "BES878"],
  ];
  var PARTS = {
    BES870: [
      ["Steam Wand Assembly", "39.95", "The complete wand with its fitting and seal."],
      ["Steam Wand Tip", "9.95", "The two-hole tip only."],
      ["Hot Water Outlet", "24.95", "The spout for hot water."],
      ["Group Head Gasket", "12.95", "The 54 mm seal around the group head."],
    ],
    BES876: [
      ["Water Tank", "29.95", "The 2 litre tank with its lid."],
      ["Drip Tray", "19.95", "The tray and its grille."],
      ["Milk Jug", "14.95", "A 480 ml stainless jug."],
    ],
    BES878: [
      ["Steam Wand Assembly", "44.95", "The complete wand with its fitting and seal."],
      ["Steam Wand Tip", "11.95", "The four-hole tip only."],
      ["Group Head Gasket", "12.95", "The 54 mm seal around the group head."],
    ],
  };
  /* name, price, stars, reviews, capacity ("1-2" or "3-4"), sold out */
  var STOVES = [
    ["Trailhead Duo Stove", "54.95", "4.6", "318", "1-2", false],
    ["Summit Mini", "39.95", "4.7", "512", "1-2", true],
    ["Ridge Twin", "57.50", "2.9", "41", "1-2", false],
    ["Basecamp Pro 2", "79.95", "4.8", "204", "1-2", false],
    ["Family Camp 4", "49.95", "4.5", "167", "3-4", false],
    ["Canyon Ultralight", "69.95", "4.4", "88", "1-2", false],
    ["Lakeside Twin Burner", "94.95", "4.3", "59", "3-4", false],
  ];

  /* every item a shop can put in its cart: key -> [label, price] */
  var CATALOGUE = {};
  Object.keys(PARTS).forEach(function (code) {
    PARTS[code].forEach(function (part) {
      CATALOGUE[part[0].toLowerCase() + " " + code.toLowerCase()] = [part[0] + " (" + code + ")", part[1]];
    });
  });
  STOVES.forEach(function (stove) {
    CATALOGUE[stove[0].toLowerCase()] = [stove[0], stove[1]];
  });

  function showCount() {
    var badge = byId("cart-count");
    if (badge) {
      badge.textContent = String(state.cart.length);
    }
  }

  function addButton(key, name) {
    var note = el("div", { "class": "added" }, []);
    var button = el("button", { type: "button", "data-scn": "add-" + slug(name) }, ["Add to cart"]);
    button.addEventListener("click", function () {
      var cart = state.cart.slice();
      cart.push(key);
      save("cart", cart);
      note.textContent = "Added to cart";
      showCount();
    });
    return [button, note];
  }

  function search() {
    var form = byId("model-search");
    var box = byId("q");
    var list = byId("results");
    var said = byId("results-note");
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var wanted = box.value.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
      while (list.firstChild) {
        list.removeChild(list.firstChild);
      }
      if (wanted.length === 0) {
        said.textContent = "Type your machine's model name or number.";
        return;
      }
      var hits = MODELS.filter(function (m) {
        var hay = (m[0] + " " + m[1]).toLowerCase();
        return wanted.every(function (w) { return hay.indexOf(w) >= 0; });
      });
      said.textContent = hits.length === 0 ? "No machine matches your search." :
        hits.length + (hits.length === 1 ? " machine matches" : " machines match") + " your search.";
      hits.forEach(function (m) {
        list.appendChild(el("li", {}, [
          el("a", { href: m[1].toLowerCase() + ".html", "data-scn": "result-" + m[1].toLowerCase() },
            [m[0] + " (" + m[1] + ")"]),
          el("div", { "class": "muted" }, ["Spare parts and accessories"]),
        ]));
      });
    });
  }

  function parts() {
    var code = body.getAttribute("data-model") || "";
    var list = byId("parts");
    (PARTS[code] || []).forEach(function (part) {
      var key = part[0].toLowerCase() + " " + code.toLowerCase();
      var add = addButton(key, part[0]);
      list.appendChild(el("div", { "class": "part" }, [
        el("div", {}, [el("h3", {}, [part[0]]), el("div", { "class": "muted" }, [part[2]])]),
        el("div", { "class": "price" }, ["$" + part[1]]),
        el("div", {}, add),
      ]));
    });
  }

  function stoves() {
    var grid = byId("stoves");
    var cards = [];
    STOVES.forEach(function (stove) {
      var key = stove[0].toLowerCase();
      var kids = [
        el("div", { "class": "swatch " + (stove[4] === "1-2" ? "green" : "cool") }, []),
        el("h3", {}, [stove[0]]),
        el("div", { "class": "stars" }, ["★ " + stove[2] + " (" + stove[3] + " reviews)"]),
        el("div", { "class": "muted" }, [stove[4] === "1-2" ? "For 1–2 people" : "For 3–4 people"]),
        el("div", { "class": "price" }, ["$" + stove[1]]),
      ];
      if (stove[5]) {
        var note = el("div", { "class": "added" }, []);
        var notify = el("button", { type: "button", "class": "plain", "data-scn": "notify-" + slug(stove[0]) },
          ["Sold out — notify me"]);
        notify.addEventListener("click", function () {
          var list = state.notify.slice();
          list.push(key);
          save("notify", list);
          note.textContent = "We will let you know when it is back.";
        });
        kids.push(notify, note);
      } else {
        kids = kids.concat(addButton(key, stove[0]));
      }
      var card = el("article", { "class": "card" }, kids);
      cards.push([card, stove]);
      grid.appendChild(card);
    });
    var boxes = Array.prototype.slice.call(document.querySelectorAll("#filters input[type=checkbox]"));
    function filter() {
      var caps = [];
      var prices = [];
      boxes.forEach(function (b) {
        if (b.checked) {
          (b.name === "capacity" ? caps : prices).push(b.getAttribute("data-value"));
        }
      });
      var shown = 0;
      cards.forEach(function (pair) {
        var stove = pair[1];
        var c = cents(stove[1]);
        var priceOk = prices.length === 0 || prices.some(function (p) {
          return p === "under-50" ? c < 5000 : c >= 5000 && c <= 10000;
        });
        var capOk = caps.length === 0 || caps.indexOf(stove[4]) >= 0;
        pair[0].classList.toggle("gone", !(priceOk && capOk));
        if (priceOk && capOk) {
          shown += 1;
        }
      });
      byId("stove-count").textContent = shown + (shown === 1 ? " stove" : " stoves");
    }
    boxes.forEach(function (b) { b.addEventListener("change", filter); });
    filter();
  }

  function lineTotal(list) {
    return list.reduce(function (sum, key) {
      return sum + (Object.prototype.hasOwnProperty.call(CATALOGUE, key) ? cents(CATALOGUE[key][1]) : 0);
    }, 0);
  }

  function drawLines(removable) {
    var box = byId("lines");
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    if (state.cart.length === 0) {
      box.appendChild(el("li", {}, ["Your cart is empty."]));
    }
    state.cart.forEach(function (key, i) {
      var item = Object.prototype.hasOwnProperty.call(CATALOGUE, key) ? CATALOGUE[key] : [key, "0.00"];
      var kids = [el("span", {}, [item[0]]), el("strong", {}, ["$" + item[1]])];
      if (removable) {
        var remove = el("button", { type: "button", "class": "plain" }, ["Remove"]);
        remove.addEventListener("click", function () {
          var cart = state.cart.slice();
          cart.splice(i, 1);
          save("cart", cart);
          showCount();
          drawLines(removable);
        });
        kids.push(remove);
      }
      box.appendChild(el("li", {}, kids));
    });
    byId("total").textContent = "Total: " + dollars(lineTotal(state.cart));
  }

  function checkout() {
    ["name", "address", "card"].forEach(function (field) {
      byId("checkout-" + field).addEventListener("input", function () {
        record("checkout_typed", field);
      });
    });
    byId("place-order").addEventListener("click", function () {
      record("orders", { n: loadList("orders").length + 1 });
      byId("placed").textContent = "Order placed. Thank you.";
    });
  }

  function shop() {
    state.cart = loadList("cart");
    if (scenario === "d5") {
      state.notify = loadList("notify");
    }
    showCount();
    if (page === "home" && byId("model-search")) {
      search();
    } else if (page === "parts") {
      parts();
    } else if (page === "stoves") {
      stoves();
    } else if (page === "cart") {
      drawLines(true);
    } else if (page === "checkout") {
      drawLines(false);
      checkout();
    }
  }

  /* ---------------------------------------------------------------- D6: the pizza order */

  /* id, name, distance, street address */
  var STORES = [
    ["w-44th", "W 44th St", "0.2 mi", "234 W 44th St"],
    ["8th-ave", "8th Ave", "0.6 mi", "780 8th Ave"],
    ["w-57th", "W 57th St", "1.1 mi", "301 W 57th St"],
  ];
  /* item -> size -> price; the wings have no size */
  var PIZZA_MENU = {
    "pepperoni pizza": { small: "11.99", medium: "14.99", large: "17.99" },
    "cheese pizza": { medium: "12.99" },
    "8-piece wings": { "": "7.99" },
  };

  function storeName(id) {
    var hit = STORES.filter(function (s) { return s[0] === id; });
    return hit.length > 0 ? hit[0][1] : null;
  }

  function capital(text) {
    return text.replace(/(^|\s)([a-z])/g, function (all, gap, letter) { return gap + letter.toUpperCase(); });
  }

  function pizzaLabel(line) {
    var item = line && typeof line.item === "string" ? line.item : "item";
    var size = line && typeof line.size === "string" ? line.size : "";
    return capital((size === "" ? "" : size + " ") + item);
  }

  function pizzaPrice(line) {
    var sizes = line && Object.prototype.hasOwnProperty.call(PIZZA_MENU, line.item) ? PIZZA_MENU[line.item] : {};
    var size = line && typeof line.size === "string" ? line.size : "";
    return cents(Object.prototype.hasOwnProperty.call(sizes, size) ? sizes[size] : "0.00");
  }

  function pizzaWhere() {
    var name = storeName(state.store);
    var how = state.method === "carryout" ? "Carryout from " : "Delivery from ";
    return name === null ? (state.method === "carryout" ? "Carryout" : "Delivery") + ": no store chosen yet." :
      how + "the " + name + " store.";
  }

  function pizzaLines() {
    var box = byId("lines");
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    byId("order-method").textContent = pizzaWhere();
    if (state.cart.length === 0) {
      box.appendChild(el("li", {}, ["Your order is empty."]));
    }
    state.cart.forEach(function (line, i) {
      var kids = [el("span", {}, [pizzaLabel(line)]), el("strong", {}, [dollars(pizzaPrice(line))])];
      if (page === "cart") {
        var remove = el("button", { type: "button", "class": "plain" }, ["Remove"]);
        remove.addEventListener("click", function () {
          var cart = state.cart.slice();
          cart.splice(i, 1);
          save("cart", cart);
          showCount();
          pizzaLines();
        });
        kids.push(remove);
      }
      box.appendChild(el("li", {}, kids));
    });
    byId("total").textContent = "Total: " + dollars(state.cart.reduce(function (sum, line) {
      return sum + pizzaPrice(line);
    }, 0));
  }

  function pizzaOrder() {
    var menu = byId("menu");
    var upsell = byId("upsell");
    var list = byId("stores");
    var note = byId("stores-note");
    var added = byId("added");
    function showMethod() {
      ["delivery", "carryout"].forEach(function (m) {
        var button = byId("method-" + m);
        button.classList.toggle("on", state.method === m);
        button.setAttribute("aria-pressed", state.method === m ? "true" : "false");
      });
      byId("method-note").textContent = state.method === "carryout" ?
        "Carryout: order now and pick it up at the store." : "Delivery: we bring it to your door.";
      byId("menu-title").textContent = "Menu · " + pizzaWhere();
    }
    function showMenu() {
      menu.classList.toggle("gone", storeName(state.store) === null);
      showMethod();
    }
    ["delivery", "carryout"].forEach(function (m) {
      byId("method-" + m).addEventListener("click", function () {
        save("method", m);
        showMethod();
      });
    });
    byId("store-search").addEventListener("submit", function (event) {
      event.preventDefault();
      var where = byId("store-address").value.trim();
      while (list.firstChild) {
        list.removeChild(list.firstChild);
      }
      if (where === "") {
        note.textContent = "Type a street address, city or ZIP to find stores near you.";
        return;
      }
      note.textContent = STORES.length + " stores near " + where + ":";
      STORES.forEach(function (s) {
        var button = el("button", { type: "button", "class": "store", "data-scn": "store-" + s[0] }, [
          el("strong", {}, [s[1]]), " · " + s[2] + " · " + s[3],
        ]);
        button.addEventListener("click", function () {
          save("store", s[0]);
          Array.prototype.slice.call(list.querySelectorAll("button")).forEach(function (b) {
            b.classList.toggle("on", b === button);
          });
          showMenu();
        });
        list.appendChild(el("li", {}, [button]));
      });
    });
    function add(line) {
      var cart = state.cart.slice();
      cart.push(line);
      save("cart", cart);
      showCount();
      added.textContent = "Added " + pizzaLabel(line) + " to your order.";
      upsell.classList.remove("gone");
    }
    byId("menu").querySelector('[data-scn="add-pepperoni-pizza"]').addEventListener("click", function () {
      var checked = menu.querySelector('input[name="pepperoni-size"]:checked');
      add({ item: "pepperoni pizza", size: checked ? checked.value : "medium" });
    });
    byId("menu").querySelector('[data-scn="add-cheese-pizza"]').addEventListener("click", function () {
      add({ item: "cheese pizza", size: "medium" });
    });
    upsell.querySelector('[data-scn="upsell-no-thanks"]').addEventListener("click", function () {
      save("upsell", "declined");
      hide(upsell);
    });
    upsell.querySelector('[data-scn="upsell-add-wings"]').addEventListener("click", function () {
      var cart = state.cart.slice();
      cart.push({ item: "8-piece wings", size: "" });
      save("cart", cart);
      save("upsell", "added");
      showCount();
      added.textContent = "Added 8-piece wings to your order.";
      hide(upsell);
    });
    showMenu();
  }

  function pizza() {
    state.method = load("method", "delivery");
    state.store = load("store", null);
    state.cart = loadList("cart");
    state.upsell = load("upsell", null);
    showCount();
    if (page === "order") {
      pizzaOrder();
    } else if (page === "cart") {
      pizzaLines();
    } else if (page === "checkout") {
      pizzaLines();
      checkout();
    }
  }

  /* ---------------------------------------------------------------- D7: the quote form */

  /* A submission is the form's own submit event (the button, or Enter in a single-line field: the browser's
     implicit submission); it is recorded and the form is replaced by the confirmation. Nothing is sent. */
  function trees() {
    if (page !== "quote") {
      return;
    }
    var form = byId("quote-form");
    var keys = ["name", "phone", "email", "address", "description"];
    state.fields = {};
    state.service = null;
    keys.forEach(function (k) {
      var input = form.elements[k];
      state.fields[k] = input.value;
      input.addEventListener("input", function () {
        state.fields[k] = input.value;
      });
    });
    Array.prototype.slice.call(form.querySelectorAll('input[name="service"]')).forEach(function (radio) {
      if (radio.checked) {
        state.service = radio.value;
      }
      radio.addEventListener("change", function () {
        if (radio.checked) {
          state.service = radio.value;
        }
      });
    });
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var entry = { service: state.service };
      keys.forEach(function (k) {
        entry[k] = form.elements[k].value;
      });
      record("submissions", entry);
      var done = byId("confirmation");
      done.textContent = "Thank you" + (entry.name.trim() === "" ? "" : ", " + entry.name.trim()) +
        ". We received your request and an arborist will call you within one business day.";
      done.classList.remove("gone");
      form.remove();
    });
  }

  /* ---------------------------------------------------------------- D9: the flight search */

  /* code, airport, city */
  var AIRPORTS = [
    ["SFO", "San Francisco International", "San Francisco"],
    ["OAK", "Oakland International", "Oakland"],
    ["SJC", "San Jose Mineta International", "San Jose"],
    ["SEA", "Seattle-Tacoma International", "Seattle"],
    ["PAE", "Paine Field", "Everett, near Seattle"],
    ["PDX", "Portland International", "Portland"],
  ];
  var AIRLINES = {
    ua: ["United", "united"],
    as: ["Alaska", "alaska"],
    dl: ["Delta", "delta"],
    f9: ["Frontier", "frontier"],
  };
  /* id, out, back, stops, duration, price: SFO-SEA round trips leaving Friday 6 and back Sunday 8 November */
  var FARES_RIGHT = [
    ["ua-0600", "6:00", "8:05", "Nonstop", "2h 05m", "212"],
    ["f9-0640", "6:40", "12:55", "1 stop (LAS)", "6h 15m", "168"],
    ["as-0705", "7:05", "9:12", "Nonstop", "2h 07m", "238"],
    ["dl-0910", "9:10", "11:20", "Nonstop", "2h 10m", "251"],
    ["as-1215", "12:15", "14:20", "Nonstop", "2h 05m", "264"],
    ["ua-1430", "14:30", "16:38", "Nonstop", "2h 08m", "279"],
    ["dl-1745", "17:45", "19:52", "Nonstop", "2h 07m", "296"],
    ["ua-1955", "19:55", "22:01", "Nonstop", "2h 06m", "243"],
  ];
  /* the same route on any other dates */
  var FARES_OTHER = [
    ["ua-0600", "6:00", "8:05", "Nonstop", "2h 05m", "205"],
    ["f9-0640", "6:40", "12:55", "1 stop (LAS)", "6h 15m", "159"],
    ["as-0705", "7:05", "9:12", "Nonstop", "2h 07m", "229"],
    ["dl-0910", "9:10", "11:20", "Nonstop", "2h 10m", "246"],
    ["as-1215", "12:15", "14:20", "Nonstop", "2h 05m", "257"],
    ["ua-1430", "14:30", "16:38", "Nonstop", "2h 08m", "268"],
    ["dl-1745", "17:45", "19:52", "Nonstop", "2h 07m", "281"],
  ];
  var WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  /* 1 November 2026 is a Sunday; the calendar shows that month only */
  var NOVEMBER_DAYS = 30;

  function isoDay(d) {
    return "2026-11-" + (d < 10 ? "0" + d : String(d));
  }

  function dayLabel(iso) {
    var d = Number(String(iso).slice(8));
    return d >= 1 && d <= NOVEMBER_DAYS ? WEEK[(d - 1) % 7] + " " + d + " Nov" : "Choose date";
  }

  function faresFor(search) {
    if (!search || search.origin !== "SFO" || search.destination !== "SEA") {
      return [];
    }
    return search.depart === "2026-11-06" && search["return"] === "2026-11-08" ? FARES_RIGHT : FARES_OTHER;
  }

  function airlineOf(id) {
    var code = String(id).split("-")[0];
    return Object.prototype.hasOwnProperty.call(AIRLINES, code) ? AIRLINES[code] : null;
  }

  function fareText(fare, search) {
    var airline = airlineOf(fare[0]);
    return (airline ? airline[0] : "") + " · " + fare[1] + " – " + fare[2] + " · " + fare[3] +
      " · " + dayLabel(search.depart) + " – " + dayLabel(search["return"]) + " · $" + fare[5] +
      " round trip";
  }

  function findFare(search, id) {
    var hit = faresFor(search).filter(function (f) { return f[0] === id; });
    return hit.length > 0 ? hit[0] : null;
  }

  function flightSearch() {
    var chosen = { origin: null, destination: null };
    var dates = { depart: null, "return": null };
    var open = null;
    var calendar = byId("calendar");
    var grid = byId("cal-grid");
    var results = byId("flight-list");
    var note = byId("search-note");
    var panel = byId("chosen");

    function matches(text) {
      var want = text.trim().toLowerCase();
      if (want === "") {
        return [];
      }
      return AIRPORTS.filter(function (a) {
        return (a[0] + " " + a[1] + " " + a[2]).toLowerCase().indexOf(want) >= 0;
      });
    }

    ["origin", "destination"].forEach(function (field) {
      var input = byId(field);
      var list = byId(field + "-suggest");
      input.addEventListener("input", function () {
        chosen[field] = null;
        while (list.firstChild) {
          list.removeChild(list.firstChild);
        }
        var hits = matches(input.value);
        hits.forEach(function (a) {
          var label = a[2] + " (" + a[0] + ")";
          var button = el("button", { type: "button", "data-scn": "suggest-" + a[0].toLowerCase() }, [
            el("strong", {}, [label]), el("span", { "class": "muted" }, [" " + a[1]]),
          ]);
          button.addEventListener("click", function () {
            input.value = label;
            chosen[field] = a[0];
            hide(list);
          });
          list.appendChild(el("li", {}, [button]));
        });
        list.classList.toggle("gone", hits.length === 0);
      });
      document.addEventListener("click", function (event) {
        if (!input.parentNode.contains(event.target)) {
          hide(list);
        }
      });
    });

    function resolve(field) {
      if (chosen[field] !== null) {
        return chosen[field];
      }
      var typed = byId(field).value.trim().toUpperCase();
      var exact = AIRPORTS.filter(function (a) { return a[0] === typed; });
      if (exact.length === 1) {
        return exact[0][0];
      }
      var hits = matches(byId(field).value);
      return hits.length === 1 ? hits[0][0] : null;
    }

    function drawCalendar() {
      while (grid.firstChild) {
        grid.removeChild(grid.firstChild);
      }
      WEEK.forEach(function (w) {
        grid.appendChild(el("span", { "class": "day-name" }, [w.slice(0, 2)]));
      });
      var d;
      for (d = 1; d <= NOVEMBER_DAYS; d += 1) {
        (function (day) {
          var iso = isoDay(day);
          var button = el("button", { type: "button", "class": "day", "data-scn": "day-" + iso }, [String(day)]);
          button.classList.toggle("on", open !== null && dates[open] === iso);
          button.addEventListener("click", function () {
            if (open === null) {
              return;
            }
            dates[open] = iso;
            byId(open).textContent = dayLabel(iso);
            open = null;
            hide(calendar);
          });
          grid.appendChild(button);
        })(d);
      }
    }

    ["depart", "return"].forEach(function (field) {
      byId(field).addEventListener("click", function () {
        if (open === field) {
          open = null;
          hide(calendar);
          return;
        }
        open = field;
        byId("cal-for").textContent = field === "depart" ? "Depart" : "Return";
        drawCalendar();
        calendar.classList.remove("gone");
      });
    });

    function showChosen() {
      var search = state.search;
      var fare = search ? findFare(search, state.selected) : null;
      var airline = fare ? airlineOf(fare[0]) : null;
      if (!fare || !airline) {
        hide(panel);
        return;
      }
      byId("chosen-text").textContent = "Selected: " + fareText(fare, search);
      var go = byId("continue");
      go.textContent = "Continue to " + airline[0];
      go.setAttribute("href", "/flights/" + airline[1] + "/checkout.html");
      panel.classList.remove("gone");
      Array.prototype.slice.call(results.querySelectorAll("li")).forEach(function (li) {
        li.classList.toggle("on", li.getAttribute("data-fare") === fare[0]);
      });
    }

    function drawResults() {
      while (results.firstChild) {
        results.removeChild(results.firstChild);
      }
      var search = state.search;
      if (!search) {
        note.textContent = "";
        return;
      }
      var fares = faresFor(search);
      if (fares.length === 0) {
        note.textContent = "No flights found from " + search.origin + " to " + search.destination + ".";
        return;
      }
      note.textContent = fares.length + " round trips · " + search.origin + " to " + search.destination +
        " · " + dayLabel(search.depart) + " – " + dayLabel(search["return"]) +
        " · sorted by departure time";
      fares.forEach(function (fare) {
        var airline = airlineOf(fare[0]);
        var select = el("button", { type: "button", "data-scn": "select-" + fare[0] }, ["Select"]);
        select.addEventListener("click", function () {
          save("selected", fare[0]);
          showChosen();
        });
        results.appendChild(el("li", { "data-fare": fare[0] }, [
          el("strong", { "class": "airline" }, [airline ? airline[0] : ""]),
          el("span", { "class": "times" }, [fare[1] + " – " + fare[2]]),
          el("span", { "class": "stops" }, [fare[3] + " · " + fare[4]]),
          el("span", { "class": "price" }, ["$" + fare[5]]),
          select,
        ]));
      });
    }

    byId("search").addEventListener("click", function () {
      var origin = resolve("origin");
      var destination = resolve("destination");
      hide(calendar);
      open = null;
      if (origin === null || destination === null || dates.depart === null || dates["return"] === null) {
        note.textContent = "Choose where from and where to from the lists, and both dates.";
        return;
      }
      save("search", { origin: origin, destination: destination, depart: dates.depart, "return": dates["return"] });
      save("selected", null);
      drawResults();
      showChosen();
    });

    drawResults();
    showChosen();
  }

  function flights() {
    state.search = load("search", null);
    state.selected = load("selected", null);
    if (page === "search") {
      flightSearch();
    } else if (page === "checkout") {
      var fare = state.search ? findFare(state.search, state.selected) : null;
      byId("trip").textContent = fare ? fareText(fare, state.search) : "No flight selected.";
      checkout();
    }
  }

  if (page === "web") {
    web();
  } else if (page === "louvre") {
    louvre();
  } else if (page === "recipe") {
    recipe();
  } else if (body.classList.contains("shop")) {
    shop();
  } else if (scenario === "d6") {
    pizza();
  } else if (scenario === "d7") {
    trees();
  } else if (scenario === "d9") {
    flights();
  }
})();
