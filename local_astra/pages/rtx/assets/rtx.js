/* T-0209 RTX scenario copies. window.__state and the data-scn attributes are read by the tests and the harness only;
   the page shows nothing beyond what it draws. No clocks, no randomness, no timers, no network: every load draws the
   same frame. State that crosses pages lives in localStorage under "rtx.<key>" and is re-read on every load:
   visited (page ids in load order, duplicates kept), download, signin_typed and refused. Nothing here is taken from
   a real site. */
(function () {
  "use strict";

  var LISTS = ["visited", "download", "signin_typed", "refused"];
  var page = document.body.getAttribute("data-page-id") || "";
  var state = { page: page };

  function load(name) {
    var value = [];
    try {
      var raw = window.localStorage.getItem("rtx." + name);
      value = raw === null ? [] : JSON.parse(raw);
    } catch (e) {
      value = [];
    }
    return Array.isArray(value) ? value : [];
  }

  function push(name, value) {
    var list = load(name);
    list.push(value);
    state[name] = list;
    try {
      window.localStorage.setItem("rtx." + name, JSON.stringify(list));
    } catch (e) {
      /* a storage failure leaves the in-page state as it is */
    }
  }

  LISTS.forEach(function (name) {
    state[name] = load(name);
  });
  window.__state = state;
  window.rtxPush = push;
  push("visited", page);

  /* download buttons: recorded, never a file */
  var downloadNote = document.getElementById("download-message");
  Array.prototype.forEach.call(document.querySelectorAll("[data-download]"), function (button) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      push("download", { page: page, item: button.getAttribute("data-download") });
      if (downloadNote) {
        downloadNote.textContent = "Downloads are disabled in this demo copy.";
      }
    });
  });

  /* sign-in fields: any input event is recorded by field name; the form never sends */
  Array.prototype.forEach.call(document.querySelectorAll("input[data-signin]"), function (input) {
    input.addEventListener("input", function () {
      push("signin_typed", input.getAttribute("data-signin"));
    });
  });
  var signinForm = document.getElementById("signin-form");
  if (signinForm) {
    signinForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var note = document.getElementById("signin-message");
      if (note) {
        note.textContent = "Signing in is disabled in this demo copy.";
      }
    });
  }

  /* the issue filter: hides rows whose text does not hold the typed words */
  var filter = document.getElementById("issue-filter");
  if (filter) {
    var rows = document.querySelectorAll("[data-issue-row]");
    var none = document.getElementById("issue-none");
    var apply = function () {
      var words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
      var shown = 0;
      Array.prototype.forEach.call(rows, function (row) {
        var text = row.textContent.toLowerCase();
        var keep = words.every(function (w) {
          return text.indexOf(w) >= 0;
        });
        row.classList.toggle("gone", !keep);
        if (keep) {
          shown += 1;
        }
      });
      if (none) {
        none.classList.toggle("gone", shown > 0);
      }
    };
    filter.addEventListener("input", apply);
    var filterForm = document.getElementById("issue-filter-form");
    if (filterForm) {
      filterForm.addEventListener("submit", function (event) {
        event.preventDefault();
        apply();
      });
    }
  }
})();
