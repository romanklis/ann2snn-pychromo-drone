import Plotly from "plotly.js-dist-min";
import { colorOf, SHORT, DEFAULT_GOAL } from "./runparams.js";

// Set the initial 3-D camera only once; later updates keep the user's rotation.
let cameraInitialized = false;

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
    color: "#d9a24a", opacity: 0.28, flatshading: false,
    lighting: { ambient: 1.0, diffuse: 0.0, specular: 0.0 },
    name: "obstacle", showlegend: false, hoverinfo: "skip",
  };
}

function boxEdges(o) {
  const [cx, cy] = o.center || [0, 0];
  const [hx, hy] = o.half || [0.3, 0.3];
  const a = o.angle || 0;
  const h = o.height || 2.8;
  const z0 = o.z0 || 0;
  const ca = Math.cos(a), sa = Math.sin(a);
  const c = [ [-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy] ]
    .map(([x, y]) => [cx + ca * x - sa * y, cy + sa * x + ca * y, z0]);
  const corners = c.concat(c.map(([x, y, z]) => [x, y, z + h]));
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const X = [], Y = [], Z = [];
  for (const [i, j] of edges) {
    X.push(corners[i][0], corners[j][0], null);
    Y.push(corners[i][1], corners[j][1], null);
    Z.push(corners[i][2], corners[j][2], null);
  }
  return {
    type: "scatter3d", mode: "lines", x: X, y: Y, z: Z,
    line: { color: "#e0b25e", width: 3 }, opacity: 0.9,
    name: "obstacle edge", showlegend: false, hoverinfo: "skip",
  };
}

// ZYX rotation matrix Rz(yaw)·Ry(pitch)·Rx(roll) (matches rpy_from_R on the server).
function rotFromRpy([roll, pitch, yaw]) {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

function apply(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

function droneTraces(color, pos, rpy, scale, name) {
  const R = rotFromRpy(rpy);
  const a = 0.11 * scale;
  const rotors = [ [a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0] ];
  const X = [], Y = [], Z = [];
  const pushSegment = (v0, v1) => {
    const p0 = add3(pos, apply(R, v0));
    const p1 = add3(pos, apply(R, v1));
    X.push(p0[0], p1[0], null);
    Y.push(p0[1], p1[1], null);
    Z.push(p0[2], p1[2], null);
  };
  for (const r of rotors) pushSegment([0, 0, 0], r);           // arms
  pushSegment([0, 0, 0], [0.28 * scale, 0, 0]);                // heading
  pushSegment([0, 0, 0], [0.18 * scale, 0, 0]);                // body x
  pushSegment([0, 0, 0], [0, 0.18 * scale, 0]);                // body y
  pushSegment([0, 0, 0], [0, 0, 0.16 * scale]);                // body z

  const rx = [], ry = [], rz = [];
  for (const r of rotors) {
    const p = add3(pos, apply(R, r));
    rx.push(p[0]); ry.push(p[1]); rz.push(p[2]);
  }
  return [
    {
      type: "scatter3d", mode: "lines", x: X, y: Y, z: Z,
      line: { color, width: 4 }, name: `${SHORT[name] || name} body`,
      showlegend: false, hoverinfo: "skip",
    },
    {
      type: "scatter3d", mode: "markers", x: rx, y: ry, z: rz,
      marker: { color, size: 3, symbol: "circle" },
      name: `${SHORT[name] || name} rotors`, showlegend: false, hoverinfo: "skip",
    },
  ];
}

function scanTraces(color, pos, angles, ranges, rMax, name) {
  if (!angles || !ranges) return [];
  const X = [], Y = [], Z = [];
  const hx = [], hy = [], hz = [];
  for (let i = 0; i < angles.length; i++) {
    const r = ranges[i];
    if (!(r > 0)) continue;
    const a = angles[i];
    const p = [pos[0] + r * Math.cos(a), pos[1] + r * Math.sin(a), pos[2]];
    X.push(pos[0], p[0], null);
    Y.push(pos[1], p[1], null);
    Z.push(pos[2], p[2], null);
    if (r < 0.999 * rMax) { hx.push(p[0]); hy.push(p[1]); hz.push(p[2]); }
  }
  const out = [{
    type: "scatter3d", mode: "lines", x: X, y: Y, z: Z,
    line: { color, width: 1 }, opacity: 0.5,
    name: `scan (${SHORT[name] || name})`, showlegend: false, hoverinfo: "skip",
  }];
  if (hx.length) {
    out.push({
      type: "scatter3d", mode: "markers", x: hx, y: hy, z: hz,
      marker: { color, size: 2 }, name: "returns", showlegend: false, hoverinfo: "skip",
    });
  }
  return out;
}

export function renderStage(div, report, selection, markerIdx = null, opts = {}) {
  const traces = [];
  const scene = report.scene || {};
  const goal = report.goal || scene.goal || [0, 0, 0];
  const scale = opts.droneScale || 1;

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

    if (opts.showDrone && res.attitude && res.attitude[i]) {
      traces.push(...droneTraces(colorOf(name), p, res.attitude[i], scale, name));
    }
  }

  if (opts.showScan) {
    for (const brain of opts.scanBrains || []) {
      const res = report.results[brain];
      if (!res || !res.scan || !res.scan_angles) continue;
      const i = markerIdx == null ? res.scan.length - 1 : Math.min(markerIdx, res.scan.length - 1);
      const pos = res.trajectory[i] || [0, 0, 0];
      const rMax = Math.max(...res.scan[i].map((v) => (v == null ? 0 : v)), 1e-6);
      traces.push(...scanTraces(colorOf(brain), pos, res.scan_angles, res.scan[i], rMax, brain));
    }
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
        traces.push(boxTrace(o), boxEdges(o));
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
    // keep the user's camera/zoom across Plotly.react updates
    uirevision: "keep",
    scene: {
      xaxis: { title: "x", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
      yaxis: { title: "y", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
      zaxis: { title: "z", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [0, 3.2] },
      aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 1.1 },
      camera: cameraInitialized ? undefined : { eye: { x: 1.5, y: -1.6, z: 0.9 } },
      bgcolor: "rgba(0,0,0,0)",
    },
  };
  Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true });
  cameraInitialized = true;
}
