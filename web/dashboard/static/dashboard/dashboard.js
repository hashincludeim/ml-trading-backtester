/* Render Plotly figures embedded as JSON, re-theme them for dark mode, wire up controls. */
(function () {
  "use strict";

  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function isDark() {
    const forced = document.documentElement.dataset.theme;
    return forced ? forced === "dark" : darkQuery.matches;
  }

  function themeUpdate(layout) {
    const surface = cssVar("--surface");
    const update = {
      paper_bgcolor: surface,
      plot_bgcolor: surface,
      "font.color": cssVar("--text"),
      "title.font.color": cssVar("--text"),
      "legend.font.color": cssVar("--text-2"),
      "hoverlabel.bgcolor": cssVar("--surface-2"),
      "hoverlabel.font.color": cssVar("--text"),
      "hoverlabel.bordercolor": cssVar("--axis"),
    };
    Object.keys(layout).forEach(function (key) {
      if (/^[xy]axis\d*$/.test(key)) {
        update[key + ".gridcolor"] = cssVar("--grid");
        update[key + ".linecolor"] = cssVar("--axis");
        update[key + ".tickcolor"] = cssVar("--axis");
        update[key + ".tickfont.color"] = cssVar("--text-2");
        update[key + ".title.font.color"] = cssVar("--text-2");
      }
    });
    return update;
  }

  const rendered = [];

  function render(el) {
    const source = document.getElementById(el.dataset.figure);
    if (!source) return;
    const fig = JSON.parse(source.textContent);
    const layout = fig.layout || {};
    delete layout.width;
    layout.autosize = true;
    const title = layout.title && layout.title.text;
    if (title) el.setAttribute("aria-label", title);
    window.Plotly.newPlot(el, fig.data, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      toImageButtonOptions: { format: "png", scale: 2 },
    }).then(function () {
      rendered.push({ el: el, light: JSON.parse(JSON.stringify(layout)) });
      if (isDark()) window.Plotly.relayout(el, themeUpdate(layout));
    });
  }

  function retheme() {
    rendered.forEach(function (item) {
      if (isDark()) {
        window.Plotly.relayout(item.el, themeUpdate(item.light));
      } else {
        window.Plotly.relayout(item.el, {
          paper_bgcolor: item.light.paper_bgcolor || null,
          plot_bgcolor: item.light.plot_bgcolor || null,
        });
        window.Plotly.react(item.el, item.el.data, JSON.parse(JSON.stringify(item.light)));
      }
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
    wireControls();
    if (!window.Plotly) return;
    document.querySelectorAll(".chart[data-figure]").forEach(render);
    darkQuery.addEventListener("change", retheme);
  });
})();
