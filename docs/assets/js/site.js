/* ===========================================================================
   EntroPy docs — navigation, theme, search (no build step, pure static)
   Each page sets window.SITE_ROOT ('.' at root, '..' in subfolders).
   =========================================================================== */
(function () {
  "use strict";

  var ROOT = window.SITE_ROOT || ".";
  var CURRENT = location.pathname.split("/").pop() || "index.html";

  var NAV = [
    {
      title: "Get Started",
      links: [
        { href: "index.html", label: "Introduction" },
        { href: "guide/installation.html", label: "Installation" },
        { href: "guide/quickstart.html", label: "Quickstart" },
        { href: "guide/concepts.html", label: "Core Concepts" }
      ]
    },
    {
      title: "Guides",
      links: [
        { href: "guide/metrics.html", label: "Metrics Catalog" },
        { href: "guide/tracing.html", label: "Tracing" },
        { href: "guide/datasets.html", label: "Datasets" },
        { href: "guide/simulation.html", label: "Simulation" },
        { href: "guide/chaos.html", label: "Chaos Engineering" },
        { href: "guide/observability.html", label: "Observability" },
        { href: "guide/adapters.html", label: "Framework Adapters" },
        { href: "guide/cli.html", label: "CLI Reference" },
        { href: "guide/reports.html", label: "Reports" },
        { href: "guide/plugins.html", label: "Plugins" }
      ]
    },
    {
      title: "Reference",
      links: [
        { href: "api.html", label: "API Reference" },
        { href: "examples.html", label: "Examples" }
      ]
    }
  ];

  /* --------------------------- theme ----------------------------------- */
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("entropy-theme", t); } catch (e) {}
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = t === "dark" ? "☀" : "☾";
  }
  var saved;
  try { saved = localStorage.getItem("entropy-theme"); } catch (e) {}
  if (!saved) saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(saved);

  /* --------------------------- header ---------------------------------- */
  var header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML =
    '<button class="icon-btn menu-toggle" id="menu-toggle" aria-label="Menu">☰</button>' +
    '<a class="brand" href="' + ROOT + '/index.html">' +
      '<img src="' + ROOT + '/assets/img/logo.svg" alt="EntroPy logo"/>' +
      '<span class="name">EntroPy</span>' +
      '<span class="tag">v0.1.0</span>' +
    '</a>' +
    '<div class="header-spacer"></div>' +
    '<div class="header-search">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
      '<input id="search-input" type="text" placeholder="Search docs…" autocomplete="off"/>' +
      '<div class="search-results" id="search-results"></div>' +
    '</div>' +
    '<button class="icon-btn" id="theme-toggle" aria-label="Toggle theme">☾</button>' +
    '<a class="icon-btn" href="https://github.com/SatyamSingh8306/entropy-ai" aria-label="GitHub" title="GitHub">↗</a>';
  document.body.prepend(header);

  document.getElementById("theme-toggle").addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme");
    applyTheme(cur === "dark" ? "light" : "dark");
  });

  /* --------------------------- sidebar --------------------------------- */
  var groups = NAV.map(function (g) {
    var links = g.links.map(function (l) {
      var active = l.href.split("/").pop() === CURRENT ? ' class="active"' : "";
      return '<a href="' + ROOT + "/" + l.href + '"' + active + ">" + l.label + "</a>";
    }).join("");
    return '<div class="group"><div class="group-title">' + g.title + "</div>" + links + "</div>";
  }).join("");

  var layoutRoot = document.getElementById("layout-root");
  if (!layoutRoot) {
    var mt = document.getElementById("menu-toggle");
    if (mt) mt.style.display = "none";
  }
  if (layoutRoot) {
    var sidebar = document.createElement("aside");
    sidebar.className = "sidebar";
    sidebar.id = "sidebar";
    sidebar.innerHTML = "<nav>" + groups + "</nav>";
    layoutRoot.prepend(sidebar);

    /* mobile menu */
    var scrim = document.createElement("div");
    scrim.className = "scrim";
    scrim.id = "scrim";
    document.body.appendChild(scrim);
    function toggleMenu(open) {
      sidebar.classList.toggle("open", open);
      scrim.classList.toggle("show", open);
    }
    document.getElementById("menu-toggle").addEventListener("click", function () {
      toggleMenu(!sidebar.classList.contains("open"));
    });
    scrim.addEventListener("click", function () { toggleMenu(false); });
  }

  /* --------------------------- search ---------------------------------- */
  var input = document.getElementById("search-input");
  var box = document.getElementById("search-results");
  var INDEX = [];

  function render(results) {
    if (!results.length) { box.innerHTML = '<div class="empty">No matches.</div>'; return; }
    box.innerHTML = results.slice(0, 12).map(function (r) {
      return '<a href="' + ROOT + "/" + r.href + '">' +
        '<div class="sr-title">' + r.title + "</div>" +
        '<div class="sr-snippet">' + (r.snippet || "") + "</div></a>";
    }).join("");
  }

  input.addEventListener("input", function () {
    var q = input.value.trim().toLowerCase();
    if (!q) { box.classList.remove("open"); return; }
    var parts = q.split(/\s+/);
    var hits = INDEX.filter(function (r) {
      var hay = (r.title + " " + (r.keywords || "") + " " + (r.body || "")).toLowerCase();
      return parts.every(function (p) { return hay.indexOf(p) !== -1; });
    });
    render(hits);
    box.classList.add("open");
  });
  input.addEventListener("focus", function () { if (input.value.trim()) box.classList.add("open"); });
  document.addEventListener("click", function (e) {
    if (!box.contains(e.target) && e.target !== input) box.classList.remove("open");
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
    if (e.key === "Escape") { box.classList.remove("open"); input.blur(); }
  });

  fetch(ROOT + "/assets/search-index.json")
    .then(function (r) { return r.ok ? r.json() : []; })
    .then(function (data) { INDEX = data || []; })
    .catch(function () { INDEX = []; });
})();

