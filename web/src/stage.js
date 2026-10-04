import Plotly from "plotly.js-dist-min";
import { colorOf, SHORT, DEFAULT_GOAL } from "./runparams.js";

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

  const obs = scene.obstacle;
  if (obs) {
    const h = scene.obstacle_height || 2.8;
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [obs[0], obs[0]], y: [obs[1], obs[1]], z: [0, h],
      line: { color: "#c98a2b", width: 10 }, name: "PILLAR", hoverinfo: "name",
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
