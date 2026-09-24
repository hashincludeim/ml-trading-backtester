/* Render Plotly figures embedded as JSON, switch light/dark themes, wire up controls.
 *
 * Figures arrive in the light palette. For dark mode every colour listed in the server's
 * DARK_COLOR_MAP (stockml.viz.theme) is swapped for its dark step, so both themes come from
 * one Python module and charts never need a second round trip. Light is the default; dark
 * applies only after the reader picks it with the header toggle.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "stockml-theme";
  const FONT_WAIT_MS = 1500;
  const root = document.documentElement;
  // Matches the CSS phone breakpoint; below it charts get the compact layout from phoneFigure().
  const narrowQuery = window.matchMedia("(max-width: 640px)");
  const EVENT_ANNOTATION = "event";  // stockml.viz.charts.EVENT_ANNOTATION
  const LEGEND_ITEM_PX = 150;  // rough width of one legend entry at 11px
  const DATA_KEYS = new Set(["x", "y", "z", "customdata", "text", "hovertext", "bdata"]);
  const charts = [];
  let darkMap = {};

  function storeTheme(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      /* private mode: the toggle still works for this page view */
    }
  }

  function currentTheme() {
    return root.dataset.theme === "dark" ? "dark" : "light";
  }

  /* Resolve once the page's text face has loaded (or after a short wait), so Plotly measures
   * titles and legends with the real font instead of the fallback it would swap out later. */
  function fontsReady() {
    if (!document.fonts) return Promise.resolve();
    const family = getComputedStyle(document.body).fontFamily;
    const loads = ["400 12px ", "600 15px "].map(function (face) {
      return document.fonts.load(face + family).catch(function () {});
    });
    const timeout = new Promise(function (resolve) { setTimeout(resolve, FONT_WAIT_MS); });
    return Promise.race([Promise.all(loads), timeout]);
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

  function plainText(html) {
    return new DOMParser().parseFromString(html || "", "text/html").body.textContent.trim();
  }

  function legendCount(fig) {
    if (fig.layout.showlegend === false) return 0;
    const n = fig.data.filter(function (t) {
      return t.showlegend !== false && t.visible !== false;
    }).length;
    return n > 1 || fig.layout.showlegend === true ? n : 0;
  }

  /* Phone layout. On a ~340px-wide chart, Plotly's title, subtitle, legend, toolbar and range
   * buttons all compete for the same strip at the top and overlap. So on narrow screens the
   * title moves into HTML above the chart (where it can wrap), the legend moves below the plot,
   * market-event labels become numbers listed under the title, and heatmap cell text is
   * dropped (tapping a cell shows its value). The figure JSON itself is left untouched.
   */
  function phoneFigure(fig, width) {
    const layout = Object.assign({}, fig.layout);
    const margin = Object.assign({}, layout.margin);
    let height = layout.height || 420;

    layout.title = { text: "" };
    const hasTopControls = (layout.updatemenus || []).length > 0 ||
      Object.keys(layout).some(function (key) {
        return key.startsWith("xaxis") && layout[key] && layout[key].rangeselector;
      }) ||
      (layout.annotations || []).some(function (a) {
        return a.yref === "paper" && a.y >= 1 && a.yanchor === "bottom";
      });
    margin.t = hasTopControls ? 40 : 16;
    margin.l = Math.min(margin.l === undefined ? 64 : margin.l, 44);
    margin.r = Math.min(margin.r === undefined ? 24 : margin.r, 12);

    const items = legendCount(fig);
    if (items) {
      const perRow = Math.max(1, Math.floor(width / LEGEND_ITEM_PX));
      const extra = Math.ceil(items / perRow) * 20 + 12;
      layout.legend = Object.assign({}, layout.legend, {
        orientation: "h", x: 0, xanchor: "left", y: 0, yref: "container", yanchor: "bottom",
      });
      margin.b = (margin.b === undefined ? 48 : margin.b) + extra;
      height += extra;
    }

    const events = [];
    if (layout.annotations) {
      layout.annotations = layout.annotations.map(function (a) {
        if (a.name !== EVENT_ANNOTATION) return a;
        events.push(a.text);
        return Object.assign({}, a, { text: String(events.length) });
      });
    }

    const data = fig.data.map(function (trace) {
      return trace.type === "heatmap" && trace.texttemplate
        ? Object.assign({}, trace, { texttemplate: "" })
        : trace;
    });

    layout.margin = margin;
    layout.height = height;
    return { data: data, layout: layout, events: events };
  }

  /* Title, subtitle and numbered event list shown above a chart on phones (hidden by CSS on
   * wider screens, where Plotly draws the title itself). */
  function chartHead(entry, events) {
    let head = entry.head;
    if (!head) {
      head = document.createElement("div");
      head.className = "chart-head";
      entry.el.parentNode.insertBefore(head, entry.el);
      entry.head = head;
    }
    const title = entry.fig.layout.title || {};
    const parts = [
      ["h3", plainText(title.text)],
      ["p", plainText(title.subtitle && title.subtitle.text)],
      ["p", events.map(function (e, i) { return (i + 1) + " " + plainText(e); }).join(" · ")],
    ];
    head.replaceChildren();
    parts.forEach(function (part, i) {
      if (!part[1]) return;
      const node = document.createElement(part[0]);
      node.textContent = part[1];
      if (i === 2) node.className = "chart-events";
      head.appendChild(node);
    });
  }

  function draw(entry) {
    const narrow = narrowQuery.matches;
    let fig = themed(entry.fig);
    if (narrow) {
      const phone = phoneFigure(fig, entry.el.clientWidth);
      chartHead(entry, phone.events);
      fig = phone;
    }
    entry.el.style.height = (fig.layout.height || 420) + "px";
    // Keep the user's zoom/legend state across theme switches.
    const layout = Object.assign({}, fig.layout, { autosize: true, uirevision: "keep" });
    // No toolbar on phones: it covers the chart, and pinch/drag zoom works without it.
    const config = Object.assign({}, entry.config, { displayModeBar: narrow ? false : "hover" });
    return window.Plotly.react(entry.el, fig.data, layout, config);
  }

  function render(el) {
    const source = document.getElementById(el.dataset.figure);
    if (!source) return;
    const fig = JSON.parse(source.textContent);
    delete fig.layout.width;
    // plotly.js sizes an autosized plot to its container, so the container needs a definite
    // height (set in draw()); otherwise it collapses to 0px and spills over the cards below.
    const title = fig.layout.title && fig.layout.title.text;
    if (title) el.setAttribute("aria-label", title);
    const entry = {
      el: el,
      fig: fig,
      config: {
        responsive: true,
        displaylogo: false,
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

  /* On phones, show only each table's key columns (data-key-cols, 1-based) with a button to
   * reveal the rest; the first column stays pinned while scrolling sideways. */
  function compactTables() {
    document.querySelectorAll(".table-wrap[data-key-cols]").forEach(function (wrap) {
      const keys = new Set(wrap.dataset.keyCols.split(",").map(Number));
      wrap.querySelectorAll("tr").forEach(function (row) {
        Array.from(row.cells).forEach(function (cell, i) {
          if (!keys.has(i + 1)) cell.classList.add("opt");
        });
      });
      const button = document.createElement("button");
      button.type = "button";
      button.className = "show-columns";
      button.setAttribute("aria-expanded", "false");
      button.textContent = "Show all columns";
      button.addEventListener("click", function () {
        const all = wrap.classList.toggle("show-all");
        button.setAttribute("aria-expanded", String(all));
        button.textContent = all ? "Show key columns" : "Show all columns";
      });
      wrap.after(button);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const mapSource = document.getElementById("theme-map");
    if (mapSource) darkMap = JSON.parse(mapSource.textContent);
    wireThemeToggle();
    wireControls();
    compactTables();
    syncToggle();
    paintSwatches();
    if (!window.Plotly) return;
    fontsReady().then(function () {
      document.querySelectorAll(".chart[data-figure]").forEach(render);
    });
    // Rotating a phone or resizing a window across the breakpoint switches chart layouts.
    narrowQuery.addEventListener("change", function () { charts.forEach(draw); });
  });
})();
