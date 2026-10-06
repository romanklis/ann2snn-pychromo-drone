export const COLORS = {
  ds_guidance: "#d9912b",
  pid: "#8a8f98",
  ann: "#4f8cff",
  snn: "#22b8a6",
  field_ann: "#c77dff",
  field_snn: "#f72585",
};

export const SHORT = {
  ds_guidance: "TRAINER (DS)",
  pid: "PID",
  ann: "ANN",
  snn: "SNN",
  field_ann: "FIELD·ANN",
  field_snn: "FIELD·SNN",
};

export const DEFAULT_SELECTION = ["ds_guidance", "ann", "snn", "field_snn"];

export function colorOf(name) {
  return COLORS[name] || "#c9d1d9";
}

// -- interactive goal (shared by the hero and extended pages via the URL) ----- //
export const DEFAULT_GOAL = [0.0, 0.0, 2.5];
export const GOAL_BOUNDS = { x: [-3.0, 3.0], y: [-3.0, 3.0], z: [0.2, 3.5] };

export function readGoal() {
  const p = new URLSearchParams(window.location.search);
  const gx = p.get("gx");
  const gy = p.get("gy");
  const gz = p.get("gz");
  if (gx === null || gy === null || gz === null) return null;
  const goal = [Number(gx), Number(gy), Number(gz)];
  return goal.every(Number.isFinite) ? goal : null;
}

export function writeGoal(goal) {
  const p = new URLSearchParams(window.location.search);
  if (!goal) {
    p.delete("gx");
    p.delete("gy");
    p.delete("gz");
  } else {
    p.set("gx", String(goal[0]));
    p.set("gy", String(goal[1]));
    p.set("gz", String(goal[2]));
  }
  const query = p.toString();
  window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
}

export function goalError(goal) {
  const names = ["x", "y", "z"];
  for (let i = 0; i < 3; i++) {
    const v = Number(goal[i]);
    const [lo, hi] = GOAL_BOUNDS[names[i]];
    if (!Number.isFinite(v)) return `goal ${names[i]} must be a number`;
    if (v < lo || v > hi) return `goal ${names[i]}=${v.toFixed(2)} outside [${lo}, ${hi}]`;
  }
  return null;
}

// -- benchmark scene preset (shared via the URL) ----------------------------- //
export const DEFAULT_SCENE = "pillar";

export function readScene() {
  const name = new URLSearchParams(window.location.search).get("scene");
  return name || DEFAULT_SCENE;
}

export function writeScene(name) {
  const p = new URLSearchParams(window.location.search);
  if (!name || name === DEFAULT_SCENE) p.delete("scene");
  else p.set("scene", name);
  const query = p.toString();
  window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
}

// -- hidden trim window (seconds), e.g. ?from=1.0&to=4.0 --------------------- //
// Intentionally not exposed in the UI: it lives only in the URL and trims the
// report (paths, plots, playback) to [from, to].
export function readWindow() {
  const p = new URLSearchParams(window.location.search);
  const fromRaw = p.get("from");
  const toRaw = p.get("to");
  if (fromRaw === null && toRaw === null) return null;
  const from = fromRaw === null ? null : Number(fromRaw);
  const to = toRaw === null ? null : Number(toRaw);
  if (from !== null && (!Number.isFinite(from) || from < 0)) return null;
  if (to !== null && (!Number.isFinite(to) || to <= 0)) return null;
  if (from !== null && to !== null && to <= from) return null;
  return { from, to };
}
