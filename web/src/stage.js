import Plotly from "plotly.js-dist-min";
import { colorOf, SHORT, DEFAULT_GOAL } from "./runparams.js";

function boxTrace(o) {
  const [cx, cy] = o.center || [0, 0];
  const [hx, hy] = o.half || [0.3, 0.3];
  const a = o.angle || 0;
  const h = o.height || 2.8;
  const z0 = o.z0 || 0;
  const ca = Math.cos(a), sa = Math.sin(a);
  const corners = [ [-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy] ]
    .map(([x, y]) => [cx + ca * x - sa * y, cy + sa * x + ca * y]);
  const X = [], Y = [], Z = [];
  for (const [px, py] of corners) { X.push(px); Y.push(py); Z.push(z0); }
  for (const [px, py] of corners) { X.push(px); Y.push(py); Z.push(z0 + h); }
  const faces = [
    [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
  ];
  return {
    type: "mesh3d", x: X, y: Y, z: Z,
    i: faces.map((f) => f[0]), j: faces.map((f) => f[1]), k: faces.map((f) => f[2]),
    color: "#c98a2b", opacity: 0.8, name: "obstacle", showlegend: false, hoverinfo: "skip",
  };
}

export function renderStage(div, report, selection, markerIdx = null) {
  const traces = [];
  const scene = report.scene || {};
  const goal = report.goal || scene.goal || [0, 0, 0];

  for (const name of selection) {
    const res = report.results[name];
    if (!res) continue;
    const traj = res.trajectory;
    const x = traj.map((p) => p[0]);
    const y = traj.map((p) => p[1]);
    const z = traj.map((p) => p[2]);
    traces.push({
      type: "scatter3d", mode: "lines", x, y, z,
      name: SHORT[name] || name,
      line: { color: colorOf(name), width: 4 },
      opacity: 0.85, hoverinfo: "name",
    });
    const i = markerIdx == null ? traj.length - 1 : Math.min(markerIdx, traj.length - 1);
    const p = traj[i] || [0, 0, 0];
    traces.push({
      type: "scatter3d", mode: "markers", x: [p[0]], y: [p[1]], z: [p[2]],
      marker: { color: colorOf(name), size: 4 }, showlegend: false, hoverinfo: "name",
    });
  }

  traces.push({
    type: "scatter3d", mode: "markers",
    x: [goal[0]], y: [goal[1]], z: [goal[2]],
    marker: { color: "#22c55e", size: 4, symbol: "diamond" },
    name: "GOAL", hoverinfo: "name",
  });

  // faint marker for the shipped goal when a different interactive goal is active
  if (
    Math.abs(goal[0] - DEFAULT_GOAL[0]) +
    Math.abs(goal[1] - DEFAULT_GOAL[1]) +
    Math.abs(goal[2] - DEFAULT_GOAL[2]) > 1e-6
  ) {
    traces.push({
      type: "scatter3d", mode: "markers",
      x: [DEFAULT_GOAL[0]], y: [DEFAULT_GOAL[1]], z: [DEFAULT_GOAL[2]],
      marker: { color: "#22c55e", size: 2, symbol: "circle-open" },
      opacity: 0.4, name: "shipped goal", hoverinfo: "name",
    });
  }

  const obstacles = scene.obstacles || [];
  if (obstacles.length) {
    for (const o of obstacles) {
      if (o.kind === "cylinder") {
        const h = o.height || 2.8, z0 = o.z0 || 0;
        traces.push({
          type: "scatter3d", mode: "lines",
          x: [o.center[0], o.center[0]], y: [o.center[1], o.center[1]], z: [z0, z0 + h],
          line: { color: "#c98a2b", width: 10 }, showlegend: false,
          name: "obstacle", hoverinfo: "skip",
        });
      } else {
        traces.push(boxTrace(o));
      }
    }
  } else if (scene.obstacle) {
    const h = scene.obstacle_height || 2.8;
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [scene.obstacle[0], scene.obstacle[0]],
      y: [scene.obstacle[1], scene.obstacle[1]], z: [0, h],
      line: { color: "#c98a2b", width: 10 }, showlegend: false,
      name: "obstacle", hoverinfo: "skip",
    });
  }

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9", size: 10 },
    showlegend: true,
    legend: { orientation: "h", y: 0.02, x: 0.02, font: { size: 9 } },
    scene: {
      xaxis: { title: "x", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
      yaxis: { title: "y", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
      zaxis: { title: "z", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [0, 3.2] },
      aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 1.1 },
      camera: { eye: { x: 1.5, y: -1.6, z: 0.9 } },
      bgcolor: "rgba(0,0,0,0)",
    },
  };
  Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true });
}
