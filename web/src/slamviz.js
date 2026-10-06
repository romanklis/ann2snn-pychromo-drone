// "What the drone knows": top-down rendering of the SLAM occupancy map, with the
// ground-truth obstacle outline overlaid and replanning events marked. The
// trajectory uses the owning brain's colour so it is coherent with the 3-D view.

import { colorOf } from "./runparams.js";

const UNKNOWN = "#2a3542";
const FREE = "#0e141d";
const OCCUPIED = "#e0b25e";
const TRUTH = "rgba(34, 197, 94, 0.45)";
const REPLAN = "#f0a020";

export function slamTargets(report) {
  return (report.controllers || []).filter((n) => {
    const res = report.results[n];
    return res && res.sensor && res.map && res.map.frames && res.map.frames.length;
  });
}

function autoSlamName(report) {
  const targets = slamTargets(report);
  for (const pref of ["field_snn", "field_ann"]) {
    if (targets.includes(pref)) return pref;
  }
  for (const n of targets) {
    const last = report.results[n].map.frames[report.results[n].map.frames.length - 1];
    if (last && last.some((row) => row.some((v) => v > 0))) return n;
  }
  return targets[0] || null;
}

export function firstSlamResult(report, name) {
  const targets = slamTargets(report);
  const pick = name && targets.includes(name) ? name : autoSlamName(report);
  return pick ? report.results[pick] : null;
}

function obstaclePath(ctx, o, mapx, mapy) {
  ctx.beginPath();
  if (o.kind === "cylinder") {
    const r = o.radius || o.core_radius || 0.35;
    const [cx, cy] = o.center || [0, 0];
    ctx.arc(mapx(cx), mapy(cy), Math.abs(mapx(cx + r) - mapx(cx)), 0, Math.PI * 2);
  } else {
    const [cx, cy] = o.center || [0, 0];
    const [hx, hy] = o.half || [0.3, 0.3];
    const a = o.angle || 0;
    const ca = Math.cos(a), sa = Math.sin(a);
    const pts = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]
      .map(([x, y]) => [cx + ca * x - sa * y, cy + sa * x + ca * y]);
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(mapx(x), mapy(y)) : ctx.moveTo(mapx(x), mapy(y))));
    ctx.closePath();
  }
}

export function drawSlamMap(canvas, report, cursor, name) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, W, H);
  const res = firstSlamResult(report, name);
  if (!res) return { hasMap: false };

  const color = colorOf(res.controller);
  const [x0, x1, y0, y1] = res.map.bounds;
  const mapx = (x) => ((x - x0) / (x1 - x0)) * W;
  const mapy = (y) => H - ((y - y0) / (y1 - y0)) * H;

  const { shape, frames, times } = res.map;
  const n = shape[0];
  // use the frame's own absolute time (trim-safe) rather than cursor*dt
  const t = (res.t && res.t.length)
    ? res.t[Math.min(cursor, res.t.length - 1)]
    : cursor * report.dt;
  let fi = 0;
  for (let i = 0; i < times.length; i++) if (times[i] <= t) fi = i;
  const grid = frames[fi] || frames[0];
  const cw = W / n, ch = H / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = grid[i][j];
      ctx.fillStyle = v === 2 ? OCCUPIED : (v === 1 ? FREE : UNKNOWN);
      ctx.fillRect((i / n) * W, H - ((j + 1) / n) * H, cw + 1, ch + 1);
    }
  }

  // ground-truth obstacle outline (what the map is trying to discover)
  ctx.strokeStyle = TRUTH;
  ctx.lineWidth = 1.2;
  for (const o of report.scene.obstacles || []) {
    obstaclePath(ctx, o, mapx, mapy);
    ctx.stroke();
  }

  const traj = res.trajectory || [];
  const upto = Math.min(cursor, traj.length - 1);

  // replanning events, marked on the trajectory
  const replanSteps = (res.slam && res.slam.replan_steps) || [];
  ctx.fillStyle = REPLAN;
  for (const step of replanSteps) {
    if (step > upto || step >= traj.length) continue;
    const p = traj[step];
    ctx.beginPath();
    ctx.arc(mapx(p[0]), mapy(p[1]), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // estimated trajectory so far (brain colour)
  if (traj.length) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i <= upto; i++) {
      const p = traj[i];
      if (i) ctx.lineTo(mapx(p[0]), mapy(p[1]));
      else ctx.moveTo(mapx(p[0]), mapy(p[1]));
    }
    ctx.stroke();
    const p = traj[upto];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(mapx(p[0]), mapy(p[1]), 4, 0, Math.PI * 2);
    ctx.fill();
    // highlight a replan currently under the cursor
    if (replanSteps.some((s) => Math.abs(s - upto) <= 1)) {
      ctx.strokeStyle = REPLAN;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(mapx(p[0]), mapy(p[1]), 8, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  return { hasMap: true, result: res };
}

// Replanning timeline strip: one tick per replan, a moving cursor, and the
// active-event highlight.
export function drawSlamTimeline(canvas, report, cursor, name) {
  if (!canvas) return { hasReplans: false };
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, W, H);
  const res = firstSlamResult(report, name);
  if (!res || !res.slam) return { hasReplans: false };
  const steps = res.slam.replan_steps || [];
  const T = Math.max(1, (res.t && res.t.length ? res.t.length : 1) - 1);
  ctx.strokeStyle = "#22303c";
  ctx.beginPath();
  ctx.moveTo(0, H - 1);
  ctx.lineTo(W, H - 1);
  ctx.stroke();
  ctx.fillStyle = REPLAN;
  for (const step of steps) {
    const x = Math.round((step / T) * (W - 1));
    ctx.fillRect(x, 4, 1.5, H - 6);
  }
  const active = steps.some((s) => Math.abs(s - cursor) <= 1);
  if (active) {
    ctx.fillStyle = "rgba(240,160,32,0.22)";
    ctx.fillRect(0, 0, W, H);
  }
  const cx = Math.round((Math.min(cursor, T) / T) * (W - 1));
  ctx.strokeStyle = "#c9d1d9";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, H);
  ctx.stroke();
  return { hasReplans: steps.length > 0, active, count: steps.length };
}
