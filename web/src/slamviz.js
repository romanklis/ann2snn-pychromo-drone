// "What the drone knows": top-down rendering of the SLAM occupancy map, with the
// ground-truth obstacle outline overlaid so discovery is visible.

const UNKNOWN = "#2a3542";
const FREE = "#0e141d";
const OCCUPIED = "#e0b25e";
const TRUTH = "rgba(34, 197, 94, 0.45)";
const TRAIL = "#4f8cff";

export function firstSlamResult(report) {
  const names = report.controllers || [];
  // prefer the field arms (they actually consume the map), then any informative map
  const order = [
    ...names.filter((n) => n.startsWith("field_")),
    ...names.filter((n) => !n.startsWith("field_")),
  ];
  let fallback = null;
  for (const name of order) {
    const res = report.results[name];
    if (!res || !res.map || !res.map.frames || !res.map.frames.length) continue;
    fallback = fallback || res;
    const last = res.map.frames[res.map.frames.length - 1];
    // a non-trivial map has at least one free (1) or occupied (2) cell
    for (const row of last) {
      if (row.some((v) => v > 0)) return res;
    }
  }
  return fallback;
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

export function drawSlamMap(canvas, report, cursor) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, W, H);
  const res = firstSlamResult(report);
  if (!res) return { hasMap: false };

  const [x0, x1, y0, y1] = res.map.bounds;
  const mapx = (x) => ((x - x0) / (x1 - x0)) * W;
  const mapy = (y) => H - ((y - y0) / (y1 - y0)) * H;

  const { shape, frames, times } = res.map;
  const n = shape[0];
  const t = cursor * report.dt;
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

  // estimated trajectory so far
  const traj = res.trajectory || [];
  if (traj.length) {
    const upto = Math.min(cursor, traj.length - 1);
    ctx.strokeStyle = TRAIL;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= upto; i++) {
      const p = traj[i];
      if (i) ctx.lineTo(mapx(p[0]), mapy(p[1]));
      else ctx.moveTo(mapx(p[0]), mapy(p[1]));
    }
    ctx.stroke();
    const p = traj[upto];
    ctx.fillStyle = TRAIL;
    ctx.beginPath();
    ctx.arc(mapx(p[0]), mapy(p[1]), 4, 0, Math.PI * 2);
    ctx.fill();
  }
  return { hasMap: true, result: res };
}
