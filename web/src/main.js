import { getControllers, getHealth, runBenchmark } from "./api.js";
import { DEFAULT_SELECTION, SHORT, colorOf, DEFAULT_GOAL, readGoal, writeGoal, goalError, readScene, writeScene } from "./runparams.js";
import { renderStage } from "./stage.js";
import { drawRaster } from "./raster.js";
import { lineChart, setCursor } from "./chart.js";

const state = {
  report: null,
  selection: [...DEFAULT_SELECTION],
  cursor: 0,
  playing: false,
  steps: 500,
  seed: 0,
  startTime: 0,
  goal: readGoal(),
  scene: readScene(),
  showDrone: true,
  showScan: true,
  scanMode: "all",
  sensorBrains: [],
  lastStageMs: 0,
};

const $ = (id) => document.getElementById(id);

async function boot() {
  try {
    const [health, cat] = await Promise.all([getHealth(), getControllers()]);
    renderBadges(health, cat);
    buildPicker(cat);
    buildSceneSelect(cat.scenes);
    buildVizControls(cat);
  } catch (err) {
    $("badges").textContent = `server error: ${err.message}`;
    return;
  }
  await refresh();
  $("play").addEventListener("click", togglePlay);
  renderPipeline();
  initGoal();
  requestAnimationFrame(tick);
}

function initGoal() {
  fillGoalInputs(state.goal || DEFAULT_GOAL);
  updateExtendedLink();
  updateGoalLabel();
  $("set-goal").addEventListener("click", () => {
    const goal = readGoalInputs();
    const err = goalError(goal);
    if (err) {
      setStatus(err, "bad");
      return;
    }
    withBusy("computing", () => applyGoal(goal));
  });
  $("reset-goal").addEventListener("click", () => {
    withBusy("computing", () => applyGoal(null));
  });
  for (const id of ["gx", "gy", "gz"]) {
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter") $("set-goal").click();
    });
  }
}

const parseNum = (v) => parseFloat(String(v).replace(",", "."));

function setStatus(text, kind = "") {
  const el = $("status");
  if (el) {
    el.textContent = text;
    el.className = `status ${kind}`;
  }
}

function updateGoalLabel() {
  const g = state.goal || DEFAULT_GOAL;
  const tag = state.goal ? "goal" : "goal (shipped)";
  $("goal-label").textContent = `${tag} (${g.map((v) => v.toFixed(2)).join(", ")})`;
}

function fillGoalInputs(goal) {
  $("gx").value = goal[0];
  $("gy").value = goal[1];
  $("gz").value = goal[2];
}

function updateExtendedLink() {
  $("extended-link").href = `/extended${window.location.search}`;
}

function readGoalInputs() {
  const current = state.goal || DEFAULT_GOAL;
  const read = (id, i) => {
    const raw = $(id).value;
    return raw === "" || raw === null ? current[i] : parseNum(raw);
  };
  return [read("gx", 0), read("gy", 1), read("gz", 2)];
}

async function applyGoal(goal) {
  state.goal = goal;
  writeGoal(goal);
  fillGoalInputs(goal || DEFAULT_GOAL);
  updateExtendedLink();
  updateGoalLabel();
  // Optimistic: move the goal marker immediately so there is an instant reaction
  // while the (seconds-long) re-run is in flight.
  if (state.report && Array.isArray(state.report.controllers)) {
    state.report.goal = goal || DEFAULT_GOAL;
    renderStage($("stage"), state.report, state.report.controllers, null, vizOpts());
  }
  return refresh();
}

async function withBusy(label, fn) {
  const setBtn = $("set-goal");
  const resetBtn = $("reset-goal");
  setBtn.disabled = true;
  resetBtn.disabled = true;
  setStatus(`${label}…`, "busy");
  try {
    const ok = await fn();
    if (ok === false) setStatus("failed", "bad");
    else setStatus("✓", "");
  } catch (err) {
    setStatus(`error: ${err.message}`, "bad");
  } finally {
    setBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

async function refresh() {
  state.selection = selected();
  if (!state.selection.length) {
    setStatus("no brains selected", "bad");
    return false;
  }
  let report;
  try {
    // A custom goal is farther from the start, so give it a longer horizon
    // (16 s) to settle; the shipped goal keeps the canonical 10 s episode.
    report = await runBenchmark(state.selection, {
      steps: state.goal ? 800 : state.steps,
      seed: state.seed,
      goal: state.goal,
      scene: state.scene,
    });
  } catch (err) {
    $("resultbar").textContent = `benchmark failed: ${err.message}`;
    setStatus("benchmark failed", "bad");
    return false;
  }
  if (!Array.isArray(report.controllers)) {
    $("resultbar").textContent = "benchmark failed: malformed report (no controllers)";
    setStatus("benchmark failed", "bad");
    return false;
  }
  state.report = report;
  state.cursor = 0;
  renderAll();
  return true;
}

function renderBadges(health, cat) {
  const w = cat.weights || {};
  const bits = [
    `<span class="badge ${health.trained ? "ok" : "bad"}">${health.trained ? "trained" : "untrained"}</span>`,
    `<span class="badge">${cat.defaults.steps} steps · ${cat.defaults.dt}s</span>`,
    w.fingerprint ? `<span class="badge">fp ${String(w.fingerprint).slice(0, 8)}</span>` : "",
  ];
  $("badges").innerHTML = bits.join("");
}

function vizOpts() {
  let scanBrains = [];
  if (state.showScan) {
    if (state.scanMode === "all") scanBrains = [...state.sensorBrains];
    else if (state.scanMode !== "off") scanBrains = [state.scanMode];
  }
  return { showDrone: state.showDrone, showScan: state.showScan, scanBrains };
}

function redrawStage() {
  if (!state.report) return;
  const idx = state.playing ? state.cursor : null;
  renderStage($("stage"), state.report, state.report.controllers, idx, vizOpts());
}

function buildVizControls(cat) {
  state.sensorBrains = (cat.catalogue || []).filter((c) => c.recurrent).map((c) => c.name);
  const sel = $("scan-brain");
  const options = [["off", "off"], ["all", "all"],
    ...state.sensorBrains.map((n) => [n, SHORT[n] || n])];
  sel.innerHTML = options.map(([v, label]) => `<option value="${v}">${label}</option>`).join("");
  if (!["off", "all", ...state.sensorBrains].includes(state.scanMode)) state.scanMode = "all";
  sel.value = state.scanMode;
  $("show-drone").addEventListener("change", () => {
    state.showDrone = $("show-drone").checked;
    redrawStage();
  });
  $("show-scan").addEventListener("change", () => {
    state.showScan = $("show-scan").checked;
    redrawStage();
  });
  sel.addEventListener("change", () => {
    state.scanMode = sel.value;
    redrawStage();
  });
}

function buildSceneSelect(scenes) {
  const sel = $("scene");
  const names = Array.isArray(scenes) && scenes.length ? scenes : ["pillar"];
  sel.innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join("");
  if (!names.includes(state.scene)) state.scene = names[0];
  sel.value = state.scene;
  sel.addEventListener("change", () => {
    state.scene = sel.value;
    writeScene(state.scene);
    updateExtendedLink();
    withBusy("computing", () => refresh());
  });
}

function buildPicker(cat) {
  const avail = cat.available || {};
  const box = $("picker");
  box.innerHTML = "";
  for (const c of cat.catalogue) {
    const ok = !avail[c.name] || avail[c.name].available;
    const id = `pick-${c.name}`;
    const label = document.createElement("label");
    label.className = "pick" + (ok ? "" : " disabled");
    label.title = ok ? c.guide : (avail[c.name] && avail[c.name].reason) || "unavailable";
    label.innerHTML = `<input type="checkbox" id="${id}" value="${c.name}" ${state.selection.includes(c.name) && ok ? "checked" : ""} ${ok ? "" : "disabled"}/>`
      + `<span class="dot" style="background:${colorOf(c.name)}"></span>${SHORT[c.name] || c.name}`;
    label.querySelector("input").addEventListener("change", () => refresh());
    box.appendChild(label);
  }
}

function selected() {
  return [...document.querySelectorAll("#picker input:checked")].map((i) => i.value);
}

function firstResult() {
  const names = state.report.controllers || [];
  return names.length ? state.report.results[names[0]] : null;
}

function renderAll() {
  if (!state.report || !Array.isArray(state.report.controllers)) return;
  const r = state.report;
  renderStage($("stage"), r, r.controllers, null, vizOpts());
  const snnName = r.controllers.find((n) => r.results[n].spiking);
  drawRaster($("raster"), snnName ? r.results[snnName] : null, null);
  renderCharts();
  renderResultBar();
}

function renderCharts() {
  const r = state.report;
  const ref = firstResult();
  if (!ref) return;
  const x = ref.t;
  const names = r.controllers;

  lineChart($("ctrl-chart"), {
    x, yTitle: "‖u‖",
    series: names.map((n) => ({
      name: SHORT[n] || n, color: colorOf(n),
      y: r.results[n].command.map((v) => Math.hypot(v[0], v[1], v[2])),
    })),
  });
  lineChart($("track-chart"), {
    x, yTitle: "goal dist",
    series: names.map((n) => ({ name: SHORT[n] || n, color: colorOf(n), y: r.results[n].goal_dist })),
  });
  lineChart($("clear-chart"), {
    x, yTitle: "clearance",
    series: names.map((n) => ({ name: SHORT[n] || n, color: colorOf(n), y: r.results[n].clearance })),
  });
}

function renderPipeline() {
  $("pipeline").innerHTML = `
    <div class="pipe-row"><span class="node">PLANT</span><span class="arrow">→</span>
      <span class="node">POLICY π(e)</span><span class="arrow">→</span><span class="node">u (m/s²)</span></div>
    <div class="pipe-note">e = x − r (true state). The ANN→SNN weight transfer is <b>offline</b>:
      the SNN is not a second policy, it is the same connectome expressed as a spike rate.</div>`;
}

function renderResultBar() {
  const r = state.report;
  const pc = (r.stats && r.stats.per_controller) || {};
  const rows = r.controllers.map((n) => {
    const m = pc[n] || r.results[n].metrics;
    return `<div class="cell"><span class="dot" style="background:${colorOf(n)}"></span>
      <b>${SHORT[n] || n}</b><br/>mean ${fmt(m.mean_goal_dist_m)} m ·
      final ${fmt(m.final_goal_dist_m)} m · clear ${fmt(m.clearance_min_m)} ·
      hit ${m.collisions} · ${m.reached_goal ? "REACHED" : "not reached"}</div>`;
  });
  const d = r.stats && r.stats.snn_minus_ann;
  const delta = d
    ? `<div class="cell delta">Δ SNN−ANN<br/>mean ${fmt(d.mean_goal_dist_m)} m ·
       final ${fmt(d.final_goal_dist_m)} m · clear ${fmt(d.clearance_min_m)}</div>`
    : "";
  const un = Object.keys(r.unavailable || {});
  const unav = un.length ? `<div class="cell bad">unavailable: ${un.join(", ")}</div>` : "";
  $("resultbar").innerHTML = `<div class="result-title">TRANSFER EXPERIMENT</div><div class="cells">${rows.join("")}${delta}${unav}</div>`;
}

const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : "—");

function togglePlay() {
  state.playing = !state.playing;
  state.startTime = performance.now() - state.cursor * 20;
  $("play").textContent = state.playing ? "Pause" : "Play";
}

function tick(now) {
  const ref = firstResult();
  if (state.playing && ref) {
    const T = ref.t.length;
    const elapsed = (now - state.startTime) / 1000;
    state.cursor = Math.floor(elapsed / state.report.dt);
    if (state.cursor >= T) {
      state.cursor = 0;
      state.startTime = now;
    }
    paintCursor();
  }
  requestAnimationFrame(tick);
}

function paintCursor() {
  const r = state.report;
  if (!r) return;
  const ref = firstResult();
  const t = ref ? ref.t[Math.min(state.cursor, ref.t.length - 1)] : 0;
  $("clock").textContent = `t = ${t.toFixed(2)} s`;
  // Throttle the heavy 3-D/2-D redraws (~20 fps) so dragging/rotating the scene
  // stays responsive during playback.
  const now = performance.now();
  if (now - state.lastStageMs > 45) {
    renderStage($("stage"), r, r.controllers, state.cursor, vizOpts());
    const snnName = r.controllers.find((n) => r.results[n].spiking);
    drawRaster($("raster"), snnName ? r.results[snnName] : null, state.cursor);
    const cx = ref ? ref.t[Math.min(state.cursor, ref.t.length - 1)] : null;
    for (const id of ["ctrl-chart", "track-chart", "clear-chart"]) setCursor($(id), cx);
    state.lastStageMs = now;
  }
}

boot();
