export function drawRaster(canvas, res, uptoIdx) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, w, h);

  if (!res || !res.spikes) {
    ctx.fillStyle = "#5b6b7a";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(res ? "no spikes (select the SNN)" : "select the SNN brain", 10, h / 2);
    return;
  }
  const T = res.spikes.length;
  const N = res.spikes[0].length;
  const maxT = Math.max(0, Math.min(uptoIdx == null ? T - 1 : uptoIdx, T - 1));
  ctx.fillStyle = "#22b8a6";
  for (let t = 0; t <= maxT; t++) {
    const x = 1 + Math.floor((t / Math.max(1, T - 1)) * (w - 2));
    const row = res.spikes[t];
    for (let n = 0; n < N; n++) {
      if (row[n]) {
        const y = 1 + Math.floor((n / Math.max(1, N - 1)) * (h - 2));
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }
  if (uptoIdx != null) {
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
