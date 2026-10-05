import { colorOf, SHORT } from "./runparams.js";

const _activeCache = new WeakMap();

// Neuron indices that fire at least once in the run (stable rows while playing).
export function activeChannels(report, name) {
  const res = report && report.results ? report.results[name] : null;
  if (!res || !res.spikes) return [];
  let per = _activeCache.get(report);
  if (!per) {
    per = {};
    _activeCache.set(report, per);
  }
  if (per[name]) return per[name];
  const sp = res.spikes;
  const N = sp[0].length;
  const out = [];
  for (let n = 0; n < N; n++) {
    for (let t = 0; t < sp.length; t++) {
      if (sp[t][n]) {
        out.push(n);
        break;
      }
    }
  }
  per[name] = out;
  return out;
}

export function activeChannelCount(report, name) {
  const act = activeChannels(report, name);
  const res = report && report.results ? report.results[name] : null;
  return act.length || (res && res.spikes ? res.spikes[0].length : 0);
}

function _subsample(list, C) {
  if (C >= list.length) return list.slice();
  const step = list.length / C;
  return Array.from({ length: C }, (_, k) => list[Math.min(list.length - 1, Math.floor(k * step))]);
}

function _band(ctx, res, name, y0, bh, w, uptoIdx, idxs, active) {
  const T = res.spikes.length;
  const C = Math.max(1, idxs.length);
  const maxT = Math.max(0, Math.min(uptoIdx == null ? T - 1 : uptoIdx, T - 1));
  ctx.fillStyle = colorOf(name);
  for (let t = 0; t <= maxT; t++) {
    const x = 1 + Math.floor((t / Math.max(1, T - 1)) * (w - 2));
    const row = res.spikes[t];
    for (let i = 0; i < C; i++) {
      if (row[idxs[i]]) {
        const y = y0 + 1 + Math.floor((i / Math.max(1, C - 1)) * (bh - 2));
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }
  ctx.fillStyle = "#c9d1d9";
  ctx.font = "10px ui-monospace, monospace";
  const tag = C < active ? `${C}/${active} active` : `${active} active`;
  ctx.fillText(`${SHORT[name] || name} · ${tag}`, 4, y0 + 11);
}

// Draw one labelled raster band per selected spiking model, showing its active
// channels (so the panel scales to the firing population).
export function drawRasters(canvas, report, names, uptoIdx, opts = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, w, h);

  const list = (names || [])
    .map((name) => ({ name, res: report && report.results ? report.results[name] : null }))
    .filter((x) => x.res && x.res.spikes);
  if (!list.length) {
    ctx.fillStyle = "#5b6b7a";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("no spiking model selected", 10, h / 2);
    return;
  }

  const channels = Math.max(1, Number(opts.channels || 200));
  const bandH = Math.floor(h / list.length);
  list.forEach((item, i) => {
    const y0 = i * bandH;
    if (i > 0) {
      ctx.strokeStyle = "#22303c";
      ctx.beginPath();
      ctx.moveTo(0, y0);
      ctx.lineTo(w, y0);
      ctx.stroke();
    }
    const N = item.res.spikes[0].length;
    const act = activeChannels(report, item.name);
    const pool = act.length ? act : Array.from({ length: N }, (_, k) => k);
    _band(ctx, item.res, item.name, y0, bandH - 1, w, uptoIdx,
      _subsample(pool, channels), act.length || N);
  });

  // shared time cursor
  if (uptoIdx != null && list[0]) {
    const T = list[0].res.spikes.length;
    const maxT = Math.max(0, Math.min(uptoIdx, T - 1));
    const x = 1 + Math.floor((maxT / Math.max(1, T - 1)) * (w - 2));
    ctx.strokeStyle = "#8a8f98";
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// Backwards-compatible single-raster helper.
export function drawRaster(canvas, res, uptoIdx) {
  drawRasters(canvas, { results: { _: res } }, res ? ["_"] : [], uptoIdx);
}
