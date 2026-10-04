import Plotly from "plotly.js-dist-min";

const FONT = { color: "#c9d1d9", size: 10 };

export function lineChart(div, { x, series, yTitle = "", height = 150 }) {
  const traces = series.map((s) => ({
    x,
    y: s.y,
    mode: "lines",
    name: s.name,
    line: { color: s.color, width: 1.8, dash: s.dash || "solid" },
    showlegend: s.legend !== false,
  }));
  const layout = {
    margin: { l: 44, r: 8, t: 6, b: 24 },
    height,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT,
    showlegend: true,
    legend: { orientation: "h", y: 1.15, font: { size: 9 } },
    xaxis: { gridcolor: "#22303c", zerolinecolor: "#22303c", title: { text: "t (s)", font: { size: 9 } } },
    yaxis: { gridcolor: "#22303c", zerolinecolor: "#22303c", title: { text: yTitle, font: { size: 9 } } },
    shapes: [],
  };
  Plotly.react(div, traces, layout, {
    displayModeBar: false,
    responsive: true,
    staticPlot: true,
  });
}

export function setCursor(div, xc) {
  const shape = xc == null
    ? []
    : [{ type: "line", x0: xc, x1: xc, y0: 0, y1: 1, yref: "paper",
         line: { color: "#8a8f98", width: 1, dash: "dot" } }];
  Plotly.relayout(div, { shapes: shape }).catch(() => {});
}
