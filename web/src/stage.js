import Plotly from "plotly.js-dist-min";
import { colorOf, SHORT, DEFAULT_GOAL } from "./runparams.js";

// Set the initial 3-D camera only once. Per-frame updates use Plotly.restyle and
// never touch the layout, so the user's rotation can never be reset.
let cameraInitialized = false;

function droneVisible(name, opts) {
  if (!opts.showDrone) return false;
  const brains = opts.droneBrains;
  return !brains || brains.includes(name);
}

function pathVisible(name, opts) {
  if (!opts.showPath) return false;
  const brains = opts.pathBrains;
  return !brains || brains.includes(name);
}

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

function applyR(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

function droneArrays(pos, rpy, scale) {
  const R = rotFromRpy(rpy || [0, 0, 0]);
  const a = 0.11 * scale;
  const rotors = [ [a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0] ];
  const X = [], Y = [], Z = [];
  const seg = (v0, v1) => {
    const p0 = applyR(R, v0), p1 = applyR(R, v1);
    X.push(pos[0] + p0[0], pos[0] + p1[0], null);
    Y.push(pos[1] + p0[1], pos[1] + p1[1], null);
    Z.push(pos[2] + p0[2], pos[2] + p1[2], null);
  };
  for (const r of rotors) seg([0, 0, 0], r);
  seg([0, 0, 0], [0.28 * scale, 0, 0]);   // heading
  seg([0, 0, 0], [0.18 * scale, 0, 0]);   // body x
  seg([0, 0, 0], [0, 0.18 * scale, 0]);   // body y
  seg([0, 0, 0], [0, 0, 0.16 * scale]);   // body z
  const rx = [], ry = [], rz = [];
  for (const r of rotors) {
    const p = applyR(R, r);
    rx.push(pos[0] + p[0]); ry.push(pos[1] + p[1]); rz.push(pos[2] + p[2]);
  }
  return { x: X, y: Y, z: Z, rx, ry, rz };
}

function scanArrays(pos, angles, ranges, rMax) {
  const X = [], Y = [], Z = [], hx = [], hy = [], hz = [];
  if (!angles || !ranges) return { x: X, y: Y, z: Z, hx, hy, hz };
  for (let i = 0; i < angles.length; i++) {
    const r = ranges[i];
    if (!(r > 0)) continue;
    const a = angles[i];
    const px = pos[0] + r * Math.cos(a), py = pos[1] + r * Math.sin(a);
    X.push(pos[0], px, null);
    Y.push(pos[1], py, null);
    Z.push(pos[2], pos[2], null);
    if (r < 0.999 * rMax) { hx.push(px); hy.push(py); hz.push(pos[2]); }
  }
  return { x: X, y: Y, z: Z, hx, hy, hz };
}

function emptyLine(width, color) {
  return { type: "scatter3d", mode: "lines", x: [], y: [], z: [],
    line: { color, width }, showlegend: false, hoverinfo: "skip" };
}

function emptyMarkers(color, size) {
  return { type: "scatter3d", mode: "markers", x: [], y: [], z: [],
    marker: { color, size }, showlegend: false, hoverinfo: "skip" };
}

function sceneLayout(div, opts = {}) {
  const scene = {
    xaxis: { title: "x", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
    yaxis: { title: "y", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [-2.6, 2.6] },
    zaxis: { title: "z", gridcolor: "#22303c", zerolinecolor: "#22303c", range: [0, 3.2] },
    aspectmode: "manual",
    aspectratio: { x: 1, y: 1, z: 1.1 },
    bgcolor: "rgba(0,0,0,0)",
  };
  // Carry the user's current camera into a rebuild (selection/scene/goal change),
  // so toggling a controller never resets the view. Falls back to the default eye
  // on the very first build.
  let cam = null;
  try {
    const live = div && div._fullLayout && div._fullLayout.scene && div._fullLayout.scene.camera;
    if (live && live.eye && live.eye.x != null) {
      cam = {
        eye: { x: live.eye.x, y: live.eye.y, z: live.eye.z },
        center: live.center ? { x: live.center.x, y: live.center.y, z: live.center.z } : { x: 0, y: 0, z: 0 },
        up: live.up ? { x: live.up.x, y: live.up.y, z: live.up.z } : { x: 0, y: 0, z: 1 },
      };
    }
  } catch (err) {
    cam = null;
  }
  if (!cam && opts.camera && opts.camera.eye) cam = opts.camera;
  if (!cam && !cameraInitialized) cam = { eye: { x: 1.5, y: -1.6, z: 0.9 }, up: { x: 0, y: 0, z: 1 } };
  if (cam) scene.camera = cam;
  cameraInitialized = true;
  return scene;
}

// Build the full trace set once (per report/selection). Returns index bookkeeping
// used by updateStage to push per-frame data via Plotly.restyle.
export function buildStage(div, report, selection, opts = {}) {
  const scene = report.scene || {};
  const goal = report.goal || scene.goal || [0, 0, 0];
  const traces = [];
  const refs = { ctrl: {}, scan: {} };

  const obstacles = scene.obstacles || [];
  if (obstacles.length) {
    for (const o of obstacles) {
      if (o.kind === "cylinder") {
        const h = o.height || 2.8, z0 = o.z0 || 0;
        traces.push({ type: "scatter3d", mode: "lines",
          x: [o.center[0], o.center[0]], y: [o.center[1], o.center[1]], z: [z0, z0 + h],
          line: { color: "#c98a2b", width: 10 }, showlegend: false, name: "obstacle", hoverinfo: "skip" });
      } else {
        traces.push(boxTrace(o), boxEdges(o));
      }
    }
  } else if (scene.obstacle) {
    const h = scene.obstacle_height || 2.8;
    traces.push({ type: "scatter3d", mode: "lines",
      x: [scene.obstacle[0], scene.obstacle[0]], y: [scene.obstacle[1], scene.obstacle[1]], z: [0, h],
      line: { color: "#c98a2b", width: 10 }, showlegend: false, name: "obstacle", hoverinfo: "skip" });
  }

  traces.push({ type: "scatter3d", mode: "markers",
    x: [goal[0]], y: [goal[1]], z: [goal[2]],
    marker: { color: "#22c55e", size: 4, symbol: "diamond" }, name: "GOAL", hoverinfo: "name" });
  if (Math.abs(goal[0] - DEFAULT_GOAL[0]) + Math.abs(goal[1] - DEFAULT_GOAL[1]) + Math.abs(goal[2] - DEFAULT_GOAL[2]) > 1e-6) {
    traces.push({ type: "scatter3d", mode: "markers",
      x: [DEFAULT_GOAL[0]], y: [DEFAULT_GOAL[1]], z: [DEFAULT_GOAL[2]],
      marker: { color: "#22c55e", size: 2, symbol: "circle-open" },
      opacity: 0.4, name: "shipped goal", hoverinfo: "name" });
  }

  for (const name of report.controllers) {
    const res = report.results[name];
    traces.push({ type: "scatter3d", mode: "lines",
      x: res.trajectory.map((p) => p[0]), y: res.trajectory.map((p) => p[1]),
      z: res.trajectory.map((p) => p[2]),
      name: SHORT[name] || name, line: { color: colorOf(name), width: 4 },
      opacity: 0.85, hoverinfo: "name", visible: pathVisible(name, opts) });
    const last = Math.max(0, res.trajectory.length - 1);
    const pos = res.trajectory[last] || [0, 0, 0];
    const d = droneArrays(pos, res.attitude ? res.attitude[last] : [0, 0, 0], opts.droneScale || 1);
    refs.ctrl[name] = {
      marker: traces.length,
      droneLines: traces.length + 1,
      droneMarkers: traces.length + 2,
    };
    traces.push(
      { type: "scatter3d", mode: "markers", x: [pos[0]], y: [pos[1]], z: [pos[2]],
        marker: { color: colorOf(name), size: 4 }, showlegend: false, hoverinfo: "skip" },
      { type: "scatter3d", mode: "lines", x: d.x, y: d.y, z: d.z,
        line: { color: colorOf(name), width: 4 }, showlegend: false, hoverinfo: "skip",
        visible: droneVisible(name, opts) },
      { type: "scatter3d", mode: "markers", x: d.rx, y: d.ry, z: d.rz,
        marker: { color: colorOf(name), size: 3 }, showlegend: false, hoverinfo: "skip",
        visible: droneVisible(name, opts) },
    );
  }
  for (const name of report.controllers) {
    const res = report.results[name];
    if (!res.scan) continue;
    const on = !!opts.showScan && (opts.scanBrains || []).includes(name);
    let s = { x: [], y: [], z: [], hx: [], hy: [], hz: [] };
    const i = Math.max(0, res.scan.length - 1);
    if (on) {
      const pos = res.trajectory[i] || [0, 0, 0];
      const rMax = Math.max(...res.scan[i].map((v) => (v == null ? 0 : v)), 1e-6);
      s = scanArrays(pos, res.scan_angles, res.scan[i], rMax);
    }
    refs.scan[name] = { rays: traces.length, hits: traces.length + 1 };
    traces.push(
      { type: "scatter3d", mode: "lines", x: s.x, y: s.y, z: s.z,
        line: { color: colorOf(name), width: 1 }, opacity: 0.5,
        showlegend: false, hoverinfo: "skip", visible: on },
      { type: "scatter3d", mode: "markers", x: s.hx, y: s.hy, z: s.hz,
        marker: { color: colorOf(name), size: 2 }, showlegend: false, hoverinfo: "skip",
        visible: on },
    );
  }

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9", size: 10 },
    showlegend: true,
    legend: { orientation: "h", y: 0.99, yanchor: "top", x: 0.02, xanchor: "left",
              font: { size: 9 }, bgcolor: "rgba(0,0,0,0)" },
    uirevision: "keep",
    scene: sceneLayout(div, opts),
  };
  Promise.resolve(Plotly.react(div, traces, layout, { displayModeBar: false, responsive: true }))
    .then(() => updateStage(div, report, refs, opts.markerIdx == null ? null : opts.markerIdx, opts));
  refs.selection = selection;
  return refs;
}

// Per-frame update: only trace data/visibility changes; the layout (camera) is
// never touched, so rotating the scene while it plays is preserved.
export function updateStage(div, report, refs, markerIdx, opts = {}) {
  if (!refs) return;
  const indices = [];
  const X = [], Y = [], Z = [], VIS = [];
  const add = (index, xs, ys, zs, visible) => {
    indices.push(index);
    X.push(xs); Y.push(ys); Z.push(zs); VIS.push(visible);
  };

  for (const name of Object.keys(refs.ctrl)) {
    const res = report.results[name];
    if (!res) continue;
    const T = res.trajectory.length;
    const i = markerIdx == null ? T - 1 : Math.min(markerIdx, T - 1);
    const pos = res.trajectory[i] || [0, 0, 0];
    add(refs.ctrl[name].marker, [pos[0]], [pos[1]], [pos[2]], true);
    const d = droneArrays(pos, res.attitude ? res.attitude[i] : [0, 0, 0], opts.droneScale || 1);
    add(refs.ctrl[name].droneLines, d.x, d.y, d.z, droneVisible(name, opts));
    add(refs.ctrl[name].droneMarkers, d.rx, d.ry, d.rz, droneVisible(name, opts));
  }
  for (const name of Object.keys(refs.scan)) {
    const res = report.results[name];
    const on = !!opts.showScan && (opts.scanBrains || []).includes(name);
    if (on) {
      const i = markerIdx == null ? res.scan.length - 1 : Math.min(markerIdx, res.scan.length - 1);
      const pos = res.trajectory[i] || [0, 0, 0];
      const rMax = Math.max(...res.scan[i].map((v) => (v == null ? 0 : v)), 1e-6);
      const s = scanArrays(pos, res.scan_angles, res.scan[i], rMax);
      add(refs.scan[name].rays, s.x, s.y, s.z, true);
      add(refs.scan[name].hits, s.hx, s.hy, s.hz, true);
    } else {
      add(refs.scan[name].rays, [], [], [], false);
      add(refs.scan[name].hits, [], [], [], false);
    }
  }
  // One update object, one value per listed trace (Plotly restyle convention).
  Plotly.restyle(div, { x: X, y: Y, z: Z, visible: VIS }, indices);
}
