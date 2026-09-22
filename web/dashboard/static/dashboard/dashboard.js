/* Render Plotly figures embedded as JSON, switch light/dark themes, wire up controls.
 *
 * Figures arrive in the light palette. For dark mode every colour listed in the server's
 * DARK_COLOR_MAP (stockml.viz.theme) is swapped for its dark step, so both themes come from
 * one Python module and charts never need a second round trip.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "stockml-theme";
  const root = document.documentElement;
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const DATA_KEYS = new Set(["x", "y", "z", "customdata", "text", "hovertext", "bdata"]);
  const charts = [];
  let darkMap = {};

  function storedTheme() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (e) {
      return null;
    }
  }

  function storeTheme(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      /* private mode: the toggle still works for this page view */
    }
  }

  function currentTheme() {
    const forced = root.dataset.theme;
    if (forced === "light" || forced === "dark") return forced;
    return darkQuery.matches ? "dark" : "light";
  }

  /* Deep-copy a figure fragment, swapping any colour string found in the map. */
  function recolor(value, map) {
    if (typeof value === "string") {
      return map[value.toLowerCase()] || value;
    }
    if (Array.isArray(value)) {
      return value.map(function (item) { return recolor(item, map); });
    }
    if (value && typeof value === "object") {
      const out = {};
      Object.keys(value).forEach(function (key) {
        out[key] = DATA_KEYS.has(key) ? value[key] : recolor(value[key], map);
      });
      return out;
    }
    return value;
  }

  function themed(fig) {
    if (currentTheme() !== "dark") return fig;
    return { data: recolor(fig.data, darkMap), layout: recolor(fig.layout, darkMap) };
  }

  function draw(entry) {
    const fig = themed(entry.fig);
    // Keep the user's zoom/legend state across theme switches.
    const layout = Object.assign({}, fig.layout, { autosize: true, uirevision: "keep" });
    return window.Plotly.react(entry.el, fig.data, layout, entry.config);
  }

  function render(el) {
    const source = document.getElementById(el.dataset.figure);
    if (!source) return;
    const fig = JSON.parse(source.textContent);
    delete fig.layout.width;
    const title = fig.layout.title && fig.layout.title.text;
    if (title) el.setAttribute("aria-label", title);
    const entry = {
      el: el,
      fig: fig,
      config: {
        responsive: true,
        displaylogo: false,
        displayModeBar: "hover",
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        toImageButtonOptions: { format: "png", scale: 2, filename: el.id },
      },
    };
    charts.push(entry);
    draw(entry).then(function () { el.classList.add("is-ready"); });
  }

  function paintSwatches() {
    const dark = currentTheme() === "dark";
    document.querySelectorAll("[data-swatch]").forEach(function (el) {
      const light = el.dataset.swatch;
      el.style.background = dark ? darkMap[light.toLowerCase()] || light : light;
    });
  }

  function syncToggle() {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const dark = currentTheme() === "dark";
    button.setAttribute("aria-pressed", String(dark));
    button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    button.title = button.getAttribute("aria-label");
  }

  function applyTheme() {
    syncToggle();
    paintSwatches();
    charts.forEach(draw);
  }

  function wireThemeToggle() {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    button.addEventListener("click", function () {
      const next = currentTheme() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      storeTheme(next);
      applyTheme();
    });
    darkQuery.addEventListener("change", function () {
      if (!storedTheme()) applyTheme();
    });
  }

  function wireControls() {
    document.querySelectorAll("[data-autosubmit]").forEach(function (input) {
      input.addEventListener("change", function () { input.form.requestSubmit(); });
    });
    document.querySelectorAll("input[type=range]").forEach(function (input) {
      const out = document.querySelector('output[for="' + input.id + '"]');
      if (!out) return;
      const show = function () { out.textContent = input.value + " bp"; };
      input.addEventListener("input", show);
      show();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const mapSource = document.getElementById("theme-map");
    if (mapSource) darkMap = JSON.parse(mapSource.textContent);
    wireThemeToggle();
    wireControls();
    syncToggle();
    paintSwatches();
    if (!window.Plotly) return;
    document.querySelectorAll(".chart[data-figure]").forEach(render);
  });
})();
