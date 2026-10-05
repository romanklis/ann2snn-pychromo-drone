import { getControllers, getHealth, runBenchmark } from "./api.js";
import { SHORT, colorOf, DEFAULT_GOAL, readGoal, readScene } from "./runparams.js";
import { lineChart, setCursor } from "./chart.js";

const DASH = ["solid", "dash", "dot"];
const AXIS = ["x", "y", "z"];

const state = { report: null, selection: [], cursor: 0, steps: 500, seed: 0, full: false, goal: readGoal(), scene: readScene() };
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");

async function boot() {
  const [health, cat] = await Promise.all([getHealth(), getControllers()]);
  const avail = cat.available || {};
  state.selection = cat.catalogue
    .filter((c) => c.display !== false && (!avail[c.name] || avail[c.name].available))
    .map((c) => c.name);
  $("badges").innerHTML =
    `<span class="badge ${health.trained ? "ok" : "bad"}">${health.trained ? "trained" : "untrained"}</span>` +
    `<span class="badge">${cat.defaults.steps} steps</span>`;
  const g = state.goal || DEFAULT_GOAL;
  $("goal-label").textContent = `goal (${g.map((v) => v.toFixed(2)).join(", ")})`;
  $("scene-label").textContent = `scene ${state.scene}`;

  const box = $("picker");
  for (const c of cat.catalogue) {
    if (c.display === false) continue;      // PID is a CLI/compare baseline only
    const ok = !avail[c.name] || avail[c.name].available;
    const label = document.createElement("label");
    label.className = "pick" + (ok ? "" : " disabled");
    label.innerHTML = `<input type="checkbox" value="${c.name}" ${state.selection.includes(c.name) ? "checked" : ""} ${ok ? "" : "disabled"}/>` +
      `<span class="dot" style="background:${colorOf(c.name)}"></span>${SHORT[c.name] || c.name}`;
    label.querySelector("input").addEventListener("change", () => {
      state.selection = [...document.querySelectorAll("#picker input:checked")].map((i) => i.value);
      refresh();
    });
    box.appendChild(label);
  }

  $("full").addEventListener("click", () => {
    state.full = !state.full;
    $("full").textContent = `Full traces: ${state.full ? "on" : "off"}`;
    renderLanes();
  });
  await refresh();
}

async function refresh() {
  if (!state.selection.length) return;
  state.report = await runBenchmark(state.selection, {
    steps: state.goal ? 800 : state.steps,
    seed: state.seed,
    goal: state.goal,
    scene: state.scene,
  });
  if (!Array.isArray(state.report.controllers) || !state.report.controllers.length) {
    $("readout").textContent = "benchmark failed: malformed report (no controllers)";
    return;
  }
  state.cursor = 0;
  renderAll();
}

function T() {
  const n = state.report.controllers[0];
  return state.report.results[n].t.length;
}

function upto() {
  return state.full ? undefined : state.cursor + 1;
}

function series3(getter, names) {
  const out = [];
  for (const n of names) {
    for (let a = 0; a < 3; a++) {
      out.push({
        name: a === 0 ? SHORT[n] || n : undefined,
        legend: a === 0,
        color: colorOf(n),
        dash: DASH[a],
        y: getter(state.report.results[n], a, upto()),
      });
    }
  }
  return out;
}

function slice(arr, k) {
  return k == null ? arr : arr.slice(0, k);
}

function renderLanes() {
  const r = state.report;
  const names = r.controllers;
  const ref = r.results[names[0]];
  const x = slice(ref.t, upto());

  lineChart($("lane-pos"), {
    x, yTitle: "m",
    series: series3((res, a, k) => slice(res.trajectory, k).map((p) => p[a]), names),
  });
  lineChart($("lane-vel"), {
    x, yTitle: "m/s",
    series: series3((res, a, k) => slice(res.state, k).map((s) => s[3 + a]), names),
  });
  lineChart($("lane-att"), {
    x, yTitle: "deg",
    series: series3((res, a, k) => slice(res.attitude, k).map((v) => (v[a] * 180) / Math.PI), names),
  });
  lineChart($("lane-err"), {
    x, yTitle: "m",
    series: series3((res, a, k) => slice(res.trajectory, k).map((p) => p[a] - r.goal[a]), names),
  });
  lineChart($("lane-cmd"), {
    x, yTitle: "m/s²",
    series: series3((res, a, k) => slice(res.command, k).map((u) => u[a]), names),
  });
  lineChart($("lane-tele"), {
    x, yTitle: "g",
    series: names.map((n) => ({
      name: SHORT[n] || n, color: colorOf(n),
      y: slice(r.results[n].telemetry.g_force || [], upto()),
    })),
  });

  const cx = x.length ? x[x.length - 1] : null;
  for (const id of ["lane-pos", "lane-vel", "lane-att", "lane-err", "lane-cmd", "lane-tele"]) {
    setCursor($(id), cx);
  }
}

function renderAll() {
  renderLanes();
  renderReadout();
  renderMetrics();
  renderWeights();
}

function renderReadout() {
  const r = state.report;
  const i = state.cursor;
  const rows = r.controllers.map((n) => {
    const res = r.results[n];
    const s = res.state[Math.min(i, res.state.length - 1)];
    const u = res.command[Math.min(i, res.command.length - 1)];
    const tel = res.telemetry || {};
    const g = tel.spike_rate_hz ? tel.spike_rate_hz[Math.min(i, tel.spike_rate_hz.length - 1)] : null;
    const hn = tel.h_norm ? tel.h_norm[Math.min(i, tel.h_norm.length - 1)] : null;
    return `<tr><td><span class="dot" style="background:${colorOf(n)}"></span>${SHORT[n] || n}</td>
      <td>${s.slice(0, 3).map((v) => fmt(v, 2)).join(", ")}</td>
      <td>${Math.hypot(u[0], u[1], u[2]).toFixed(2)}</td>
      <td>${fmt(res.goal_dist[i], 2)}</td>
      <td>${fmt(res.clearance[i], 2)}</td>
      <td>${g == null ? (hn == null ? "—" : `h ${hn.toFixed(1)}`) : `${g.toFixed(0)} Hz`}</td></tr>`;
  });
  $("readout").innerHTML = `<table><thead><tr><th>brain</th><th>p (m)</th><th>‖u‖</th>
    <th>goal</th><th>clear</th><th>activity</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function renderMetrics() {
  const r = state.report;
  const rows = r.controllers.map((n) => {
    const m = r.results[n].metrics;
    return `<tr><td><span class="dot" style="background:${colorOf(n)}"></span>${SHORT[n] || n}</td>
      <td>${fmt(m.mean_goal_dist_m)}</td><td>${fmt(m.final_goal_dist_m)}</td>
      <td>${fmt(m.clearance_min_m)}</td><td>${m.collisions}</td>
      <td>${m.reached_goal ? "yes" : "no"}</td><td>${fmt(m.peak_g_force, 2)}</td></tr>`;
  });
  const d = r.stats && r.stats.snn_minus_ann;
  const delta = d ? `<tr class="delta"><td>Δ SNN−ANN</td><td>${fmt(d.mean_goal_dist_m)}</td>
      <td>${fmt(d.final_goal_dist_m)}</td><td>${fmt(d.clearance_min_m)}</td><td></td><td></td><td></td></tr>` : "";
  $("metrics").innerHTML = `<table><thead><tr><th>brain</th><th>mean goal</th><th>final goal</th>
    <th>clear min</th><th>hits</th><th>reached</th><th>peak g</th></tr></thead>
    <tbody>${rows.join("")}${delta}</tbody></table>`;
}

function renderWeights() {
  const w = state.report.weights || {};
  const rows = Object.entries(w).map(([k, v]) => `<tr><td>${k}</td><td>${String(v).slice(0, 42)}</td></tr>`);
  $("weights").innerHTML = `<table><tbody>${rows.join("")}</tbody></table>`;
}

$("scrub")?.addEventListener("input", (e) => {
  state.cursor = Number(e.target.value);
  renderLanes();
  renderReadout();
});

boot();
