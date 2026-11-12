import Plotly from "plotly.js-dist-min";
import { getRepairActivity, getRepairCycle, getRepairRun, getRepairRuns, getRepairTraces, getRepairWeights } from "./api.js";
import { lineChart } from "./chart.js";

const $ = (id) => document.getElementById(id);
const state = {
  runs: [], runId: null, run: null, cycle: null, activity: null,
  traces: null, traceCycleCount: -1, showRejected: false,
  selectedCycle: null, timer: null,
};
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function setStatus(text, bad = false) {
  $("status").textContent = text || "";
  $("status").className = "status" + (bad ? " bad" : "");
}

function queryRun() {
  return new URLSearchParams(window.location.search).get("run");
}

async function boot() {
  setStatus("loading runs…");
  const data = await getRepairRuns();
  state.runs = data.runs || [];
  buildPicker();
  const requested = queryRun();
  const first = state.runs.find((r) => r.run_id === requested) || state.runs[0];
  if (!first) {
    setStatus("no repair runs under out/repair yet", true);
    $("summary").textContent = "Run `python tools/repair.py run --proposer template` first.";
    return;
  }
  await loadRun(first.run_id);
  $("refresh").addEventListener("click", () => loadRun(state.runId, state.selectedCycle));
  $("run-picker").addEventListener("change", (e) => loadRun(e.target.value));
  $("show-rejected").addEventListener("change", (e) => {
    state.showRejected = e.target.checked;
    renderCompare();
  });
  state.timer = setInterval(() => {
    if (document.hidden || !state.runId) return;
    loadRun(state.runId, state.selectedCycle, true).catch(() => {});
  }, 7000);
}

function buildPicker() {
  const picker = $("run-picker");
  picker.innerHTML = "";
  for (const run of state.runs) {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    opt.textContent = `${run.run_id} · ${(run.scenario || {}).name || "?"}`;
    picker.appendChild(opt);
  }
}

async function loadRun(runId, keepCycle = null, quiet = false) {
  if (!runId) return;
  if (!quiet) setStatus("loading…");
  const changedRun = state.runId !== runId;
  state.runId = runId;
  $("run-picker").value = runId;
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState({}, "", url);
  state.run = await getRepairRun(runId);
  if (state.run.status === "no_shift") {
    clearPanels();
    renderSummary();
    renderProbe(state.run.probe);
    renderFailures({});
    setStatus("no shift");
    return;
  }
  state.activity = await getRepairActivity(runId).catch(() => null);
  const cycleCount = (state.run.cycles || []).length;
  if (changedRun || !state.traces || state.traceCycleCount !== cycleCount) {
    state.traces = await getRepairTraces(runId).catch(() => null);
    state.traceCycleCount = cycleCount;
  }
  renderSummary();
  renderProbe(state.run.probe);
  renderSuccess();
  renderCompare();
  renderCycleTable();
  renderActivity(keepCycle == null ? 0 : keepCycle);
  renderFinal();
  const target = keepCycle ?? (state.run.cycles.length ? state.run.cycles[state.run.cycles.length - 1].cycle : null);
  if (target != null) await selectCycle(target);
  setStatus("ready");
}

function renderSummary() {
  const r = state.run;
  if (r.status === "no_shift") {
    const suggested = r.suggested_scenario || (r.probe || {}).suggested_scenario || "—";
    $("badges").innerHTML = `<span class="badge bad">no shift</span>`;
    $("summary").innerHTML = `
      <div class="pipe-note">No scenario met the teacher/SNN criteria.</div>
      <div class="pipe-note">suggested run: <code>REPAIR_SCENARIO=${esc(suggested)}</code></div>
      <div class="pipe-note">probe: <code>probe.json</code> · <code>probe.md</code></div>`;
    return;
  }
  const scenario = r.scenario || {};
  const b = (r.baseline || {}).practice || {};
  const f = (r.final || {}).practice || {};
  const accepted = (r.cycles || []).filter((c) => c.accepted).length;
  $("badges").innerHTML =
    `<span class="badge ok">${esc(scenario.name || "?")}</span>` +
    `<span class="badge">${esc(r.proposer || "?")}${r.model ? ` · ${esc(r.model)}` : ""}</span>` +
    `<span class="badge">${(r.cycles || []).length} cycles</span>` +
    `${r.resume_count ? `<span class="badge">resumed ×${r.resume_count}</span>` : ""}`;
  $("summary").innerHTML = `
    <div class="metric-grid">
      <div class="metric"><b>${pct(b.near_success_rate ?? b.success_rate)}</b><span>baseline near-goal</span></div>
      <div class="metric"><b>${pct(f.near_success_rate ?? f.success_rate)}</b><span>final near-goal</span></div>
      <div class="metric"><b>${fmt(b.mean_final_progress, 3)}</b><span>baseline progress</span></div>
      <div class="metric"><b>${fmt(f.mean_final_progress, 3)}</b><span>final progress</span></div>
      <div class="metric"><b>${accepted}</b><span>accepted edits</span></div>
      <div class="metric"><b>${fmt(f.mean_final_goal_dist_m, 2)}</b><span>final goal dist (m)</span></div>
    </div>
    <div class="pipe-note">${esc(scenario.description || "")}</div>
    <div class="pipe-note">bundle: <code>${esc(r.bundle || "—")}</code></div>
    <div class="pipe-note">repaired: <code>${esc(r.bundle_out || (r.final || {}).bundle || "—")}</code></div>
    ${r.resume_count ? `<div class="pipe-note">resumed ${r.resume_count}×; new cycles start at ${esc(r.start_cycle || "—")}</div>` : ""}`;
}

function renderProbe(probe) {
  const scenarios = Array.isArray(probe) ? probe : (probe?.scenarios || []);
  if (!scenarios.length) {
    $("probe").innerHTML = `<div class="pipe-note">No auto-selection probe in this run.</div>`;
    return;
  }
  const rows = scenarios.map((item) => {
    const snn = item.snn || {};
    const teacher = item.teacher || {};
    const reason = (item.reasons || []).join("; ");
    return `<tr class="${item.selected ? "active" : ""}">
      <td>${esc(item.scenario)}</td>
      <td>${pct(snn.near_success_rate)}</td>
      <td>${fmt(snn.mean_final_progress, 3)}</td>
      <td>${pct(teacher.near_success_rate)}</td>
      <td>${fmt(teacher.mean_final_progress, 3)}</td>
      <td>${fmt(item.gap, 3)}</td>
      <td>${item.eligible ? '<span class="ok-text">✓</span>' : "—"}</td>
      <td>${esc(reason)}</td>
    </tr>`;
  }).join("");
  $("probe").innerHTML = `<table><thead><tr>
    <th>scenario</th><th>snn near</th><th>snn prog</th>
    <th>t near</th><th>t prog</th><th>gap</th><th>ok</th><th>reason</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function clearPanels() {
  for (const id of ["success-chart", "compare-metrics", "compare-3d", "replay-3d", "replay-chart", "weight-chart"]) {
    const el = $(id);
    if (el) {
      try { Plotly.purge(el); } catch { /* ignore */ }
      el.innerHTML = "";
    }
  }
  for (const id of ["cycle-table", "failures", "program", "transcript", "final", "weight-table", "compare-legend", "compare-note", "trace-note", "weight-note", "activity-note"]) {
    const el = $(id);
    if (el) el.innerHTML = "";
  }
  const canvas = $("activity-strip");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function renderSuccess() {
  const r = state.run;
  const cycles = r.cycles || [];
  const x = [0, ...cycles.map((c) => c.cycle)];
  const baseline = (((r.baseline || {}).practice || {}).near_success_rate ?? ((r.baseline || {}).practice || {}).success_rate) ?? 0;
  const practice = [baseline, ...cycles.map((c) => (((c.practice || {}).metrics || {}).near_success_rate ?? ((c.practice || {}).metrics || {}).success_rate ?? 0))];
  const regression = [(((r.baseline || {}).regression || {}).near_success_rate ?? ((r.baseline || {}).regression || {}).success_rate) ?? 0,
    ...cycles.map((c) => (((c.regression || {}).metrics || {}).near_success_rate ?? ((c.regression || {}).metrics || {}).success_rate ?? 0))];
  const finalSuccess = (((r.final || {}).practice || {}).near_success_rate ?? ((r.final || {}).practice || {}).success_rate);
  const traces = [
    {
      x, y: practice, mode: "lines+markers", name: "practice",
      line: { color: "#22b8a6", width: 2 },
      marker: { size: 5, color: [baseline, ...cycles.map((c) => (c.accepted ? "#22b8a6" : "#7b8a99"))] },
    },
    {
      x, y: regression, mode: "lines+markers", name: "nominal regression",
      line: { color: "#7b8a99", width: 1, dash: "dot" }, marker: { size: 3 },
    },
  ];
  if (Number.isFinite(finalSuccess)) {
    traces.push({
      x: [x[x.length - 1] + 1], y: [finalSuccess], mode: "markers", name: "final unseen",
      marker: { size: 10, color: "#e0a030", symbol: "diamond" },
    });
  }
  Plotly.react($("success-chart"), traces, {
    margin: { l: 44, r: 8, t: 6, b: 24 },
    height: 180,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9", size: 10 },
    xaxis: { title: { text: "cycle", font: { size: 9 } }, gridcolor: "#22303c" },
    yaxis: { title: { text: "success", font: { size: 9 } }, range: [0, 1], gridcolor: "#22303c" },
    legend: { orientation: "h", y: 1.15, font: { size: 9 } },
  }, { displayModeBar: false, responsive: true, staticPlot: true });
}

function renderCompare() {
  const r = state.run;
  if (!r) return;
  const cycles = r.cycles || [];
  const x = [0, ...cycles.map((c) => c.cycle)];
  const base = ((r.baseline || {}).practice || {});
  const metric = (c) => (c.practice || {}).metrics || {};
  const near = [base.near_success_rate ?? base.success_rate ?? 0,
    ...cycles.map((c) => metric(c).near_success_rate ?? metric(c).success_rate ?? null)];
  const progress = [base.mean_final_progress ?? 0,
    ...cycles.map((c) => metric(c).mean_final_progress ?? null)];
  const collisionFree = [base.collision_free_rate ?? 0,
    ...cycles.map((c) => metric(c).collision_free_rate ?? null)];
  const finalDist = [base.mean_final_goal_dist_m ?? 0,
    ...cycles.map((c) => metric(c).mean_final_goal_dist_m ?? null)];
  const acceptedColors = [base.near_success_rate != null ? "#7b8a99" : "#7b8a99",
    ...cycles.map((c) => (c.accepted ? "#22b8a6" : "#7b8a99"))];

  Plotly.react($("compare-metrics"), [
    {
      x, y: near, mode: "lines+markers", name: "near-goal",
      line: { color: "#22b8a6", width: 2 }, marker: { size: 5, color: acceptedColors },
    },
    {
      x, y: progress, mode: "lines+markers", name: "progress",
      line: { color: "#b06cff", width: 1.5 }, marker: { size: 3 },
    },
    {
      x, y: collisionFree, mode: "lines", name: "collision-free",
      line: { color: "#4aa3df", width: 1, dash: "dot" },
    },
    {
      x, y: finalDist, mode: "lines+markers", name: "final dist (m)", yaxis: "y2",
      line: { color: "#e0a030", width: 1.5, dash: "dash" }, marker: { size: 3 },
    },
  ], {
    margin: { l: 44, r: 44, t: 6, b: 24 }, height: 180,
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9", size: 10 },
    xaxis: { title: { text: "cycle", font: { size: 9 } }, gridcolor: "#22303c" },
    yaxis: {
      title: { text: "rate / progress", font: { size: 9 } }, range: [0, 1],
      gridcolor: "#22303c",
    },
    yaxis2: {
      title: { text: "m", font: { size: 9 } }, overlaying: "y", side: "right",
      rangemode: "tozero", showgrid: false,
    },
    legend: { orientation: "h", y: 1.18, font: { size: 9 } },
  }, { displayModeBar: false, responsive: true, staticPlot: true });

  const traces = state.traces;
  const selected = cycles.filter(
    (c) => c.accepted || state.showRejected || c.cycle === cycles[cycles.length - 1]?.cycle
  );
  const palette = ["#7b8a99", "#22b8a6", "#e0a030", "#b06cff", "#4aa3df", "#ff8a8a", "#9adf6b"];
  const items = [{ name: "baseline", trace: traces?.baseline, accepted: false }];
  for (const cycle of selected) {
    const trace = traces?.cycles?.[String(cycle.cycle)];
    if (trace) {
      items.push({
        name: `cycle ${cycle.cycle}${cycle.accepted ? " ✓" : ""}`,
        trace,
        accepted: !!cycle.accepted,
      });
    }
  }
  const lines = [];
  items.forEach((item, index) => {
    const trajectory = item.trace?.trajectory;
    if (!trajectory || !trajectory.length) return;
    lines.push({
      type: "scatter3d", mode: "lines", name: item.name,
      x: trajectory.map((p) => p[0]),
      y: trajectory.map((p) => p[1]),
      z: trajectory.map((p) => p[2]),
      line: {
        color: palette[index % palette.length],
        width: item.name === "baseline" ? 4 : 2.5,
        dash: item.accepted || item.name === "baseline" ? "solid" : "dot",
      },
    });
  });
  const goal = ((((r.scenario || {}).scene || {}).goal) || [0, 0, 2.5]);
  if (lines.length) {
    lines.push({
      type: "scatter3d", mode: "markers", name: "goal",
      x: [goal[0]], y: [goal[1]], z: [goal[2]],
      marker: { color: "#ff8a8a", size: 6, symbol: "diamond" },
    });
  }
  Plotly.react($("compare-3d"), lines, {
    margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: "x", gridcolor: "#22303c" },
      yaxis: { title: "y", gridcolor: "#22303c" },
      zaxis: { title: "z", gridcolor: "#22303c" },
      bgcolor: "rgba(0,0,0,0)", aspectmode: "data",
    },
    font: { color: "#c9d1d9", size: 9 },
    legend: { orientation: "h", y: 1.1, font: { size: 9 } },
    showlegend: lines.length > 0,
  }, { displayModeBar: false, responsive: true, staticPlot: true });
  const accepted = cycles.filter((c) => c.accepted).length;
  $("compare-note").textContent =
    `${selected.length} compared · ${accepted} accepted`;
}

function renderCycleTable() {
  const cycles = state.run.cycles || [];
  const base = (((state.run.baseline || {}).practice || {}).near_success_rate ?? ((state.run.baseline || {}).practice || {}).success_rate ?? 0);
  const rows = cycles.map((c) => {
    const metrics = (c.practice || {}).metrics || {};
    const success = (metrics.near_success_rate ?? metrics.success_rate);
    const progress = fmt(metrics.mean_final_progress, 3);
    const rationale = ((c.program || {}).rationale || c.note || "").slice(0, 80);
    return `<tr data-cycle="${c.cycle}" class="${state.selectedCycle === c.cycle ? "active" : ""}">
      <td>${c.cycle}</td>
      <td>${c.accepted ? '<span class="ok-text">✓</span>' : "—"}</td>
      <td>${pct(success)}</td>
      <td>${progress}</td>
      <td>${esc(rationale)}</td></tr>`;
  });
  const baseProgress = fmt(((state.run.baseline || {}).practice || {}).mean_final_progress, 3);
  $("cycle-table").innerHTML = `<table><thead><tr><th>#</th><th>ok</th><th>near</th><th>prog</th><th>edit</th></tr></thead>
    <tbody><tr data-cycle="0"><td>0</td><td>base</td><td>${pct(base)}</td><td>${baseProgress}</td><td>baseline</td></tr>${rows.join("")}</tbody></table>`;
  for (const tr of $("cycle-table").querySelectorAll("tr[data-cycle]")) {
    tr.addEventListener("click", () => selectCycle(Number(tr.dataset.cycle)));
  }
}

function renderActivity(cycle) {
  const a = state.activity;
  const canvas = $("activity-strip");
  if (!a || !a.silent) {
    $("activity-note").textContent = "no activity artifact";
    return;
  }
  const n = a.silent.length;
  const edited = new Set();
  const cyc = (state.run.cycles || []).find((c) => c.cycle === cycle);
  for (const op of ((cyc || {}).program || {}).ops || []) {
    if (op.unit != null) edited.add(Number(op.unit));
    if (op.dst != null) edited.add(Number(op.dst));
  }
  const cols = 100;
  const rows = Math.ceil(n / cols);
  const cw = canvas.width / cols;
  const ch = Math.min(cw, canvas.height / rows);
  canvas.height = Math.max(20, rows * ch);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < n; i++) {
    const x = (i % cols) * cw;
    const y = Math.floor(i / cols) * ch;
    let color = "#1b2531";
    if (a.silent[i]) color = "#22303c";
    if (a.rare[i]) color = "#e0a030";
    if (!a.spare[i]) color = "#22b8a6";
    if (a.critical[i]) color = "#c0392b";
    if (edited.has(i)) color = "#b06cff";
    ctx.fillStyle = color;
    ctx.fillRect(x + 0.5, y + 0.5, Math.max(1, cw - 1), Math.max(1, ch - 1));
  }
  $("activity-note").textContent = `${n} neurons · ${(a.silent.filter(Boolean).length)} silent`;
  $("activity-legend").innerHTML = `
    <span><i class="sw" style="background:#22303c"></i>silent</span>
    <span><i class="sw" style="background:#e0a030"></i>rare</span>
    <span><i class="sw" style="background:#22b8a6"></i>live</span>
    <span><i class="sw" style="background:#c0392b"></i>critical</span>
    <span><i class="sw" style="background:#b06cff"></i>edited</span>`;
}

async function selectCycle(cycle) {
  state.selectedCycle = Number(cycle);
  renderCycleTable();
  $("cycle-label").textContent = `cycle ${cycle}`;
  if (Number(cycle) === 0) {
    const base = state.run.baseline || {};
    $("program").innerHTML = `<div class="pipe-note">Baseline bundle; no edit at cycle 0.</div>`;
    $("transcript").innerHTML = `<div class="pipe-note">No transcript for the baseline.</div>`;
    $("weight-table").innerHTML = "";
    $("weight-note").textContent = "";
    $("trace-note").textContent = "baseline trace";
    await renderTraceFromApi(null);
    renderFailures((base.practice || {}).failure_counts || {});
    return;
  }
  const data = await getRepairCycle(state.runId, cycle);
  state.cycle = data;
  renderProgram(data);
  renderTranscript(data);
  renderFailures((((data.practice || {}).metrics || {}).failure_counts) || {});
  $("trace-note").textContent = `cycle ${cycle}`;
  renderTrace(data.trace || null);
  await renderWeights(cycle);
  renderActivity(cycle);
}

function renderProgram(c) {
  const p = c.program || {};
  const ops = p.ops || [];
  const rows = ops.map((op) => {
    const detail = op.kind === "add_edge"
      ? `${op.src}→${op.dst} (${fmt(op.magnitude, 3)})`
      : op.kind === "set_bias" || op.kind === "add_bias"
        ? fmt(op.value, 3)
        : op.values ? `[${op.values.slice(0, 6).map((v) => fmt(v, 2)).join(", ")}${op.values.length > 6 ? "…" : ""}]` : "";
    return `<tr><td>${esc(op.kind)}</td><td>${op.unit ?? ""}</td><td>${esc(detail)}</td></tr>`;
  }).join("");
  $("program").innerHTML = `
    <div class="pipe-note">${esc(p.rationale || "")}</div>
    <div class="pipe-note">validation: ${c.validation && c.validation.ok ? '<span class="ok-text">ok</span>' : '<span class="bad-text">failed</span>'}
      · changed ${c.validation ? c.validation.changed_params : "—"} params
      · ${c.validation ? c.validation.new_edges : 0} new edges
      ${c.validation && (c.validation.auto_isolated_units || []).length ? `· auto-isolated ${c.validation.auto_isolated_units.map(esc).join(", ")}` : ""}
      ${c.validation && (c.validation.pruned_units || []).length ? `· pruned ${c.validation.pruned_units.map(esc).join(", ")}` : ""}</div>
    <table><thead><tr><th>kind</th><th>unit</th><th>detail</th></tr></thead><tbody>${rows}</tbody></table>
    ${c.validation && (c.validation.issues || []).length ? `<div class="bad-text">${c.validation.issues.map(esc).join("<br>")}</div>` : ""}
    ${c.validation && (c.validation.warnings || []).length ? `<div class="warn-text">${c.validation.warnings.map(esc).join("<br>")}</div>` : ""}`;
}

function renderTranscript(c) {
  const rows = c.transcript || [];
  if (!rows.length) {
    $("transcript").innerHTML = `<div class="pipe-note">No transcript (template/random proposer).</div>`;
    return;
  }
  const html = rows.slice(-40).map((m) => {
    if (m.role === "tool") {
      const txt = JSON.stringify(m.result ?? {});
      return `<div class="turn tool"><b>tool ${esc(m.name)}</b><pre>${esc(txt.slice(0, 1200))}</pre></div>`;
    }
    const msg = m.message || {};
    const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? msg);
    const reasoning = typeof msg.reasoning_content === "string" ? msg.reasoning_content : "";
    const calls = (msg.tool_calls || []).map((t) => `${t.function?.name || "?"}(${(t.function?.arguments || "").slice(0, 180)})`).join("<br>");
    const reasoningBlock = reasoning ? `<pre class="reasoning">${esc(reasoning.slice(0, 1400))}</pre>` : "";
    return `<div class="turn"><b>${esc(m.role)}</b>${reasoningBlock}<pre>${esc(content.slice(0, 1600))}</pre>${calls ? `<pre>${esc(calls)}</pre>` : ""}</div>`;
  }).join("");
  $("transcript").innerHTML = html;
}

function renderFailures(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    $("failures").innerHTML = `<div class="pipe-note">No failures recorded.</div>`;
    return;
  }
  const max = Math.max(...entries.map(([, v]) => v)) || 1;
  $("failures").innerHTML = `<table><tbody>${entries.map(([k, v]) => `
    <tr><td>${esc(k)}</td><td><div class="bar" style="width:${Math.max(2, (v / max) * 100)}%"></div></td><td>${v}</td></tr>`).join("")}</tbody></table>`;
}

function renderTrace(trace) {
  if (!trace || !trace.trajectory || !trace.trajectory.length) {
    Plotly.react($("replay-3d"), [], {
      margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)",
      annotations: [{ text: "no trace in this cycle", showarrow: false, font: { color: "#7b8a99" } }],
    }, { displayModeBar: false, responsive: true, staticPlot: true });
    $("replay-chart").innerHTML = "";
    return;
  }
  const tr = trace.trajectory;
  const goal = (((state.run.scenario || {}).scene || {}).goal) || [0, 0, 2.5];
  const traces = [
    {
      type: "scatter3d", mode: "lines", name: "policy",
      x: tr.map((p) => p[0]), y: tr.map((p) => p[1]), z: tr.map((p) => p[2]),
      line: { color: "#22b8a6", width: 4 },
    },
    {
      type: "scatter3d", mode: "markers", name: "goal",
      x: [goal[0]], y: [goal[1]], z: [goal[2]],
      marker: { color: "#e0a030", size: 6, symbol: "diamond" },
    },
  ];
  Plotly.react($("replay-3d"), traces, {
    margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: "x", gridcolor: "#22303c" },
      yaxis: { title: "y", gridcolor: "#22303c" },
      zaxis: { title: "z", gridcolor: "#22303c" },
      bgcolor: "rgba(0,0,0,0)",
      aspectmode: "data",
    },
    font: { color: "#c9d1d9", size: 9 },
    showlegend: false,
  }, { displayModeBar: false, responsive: true, staticPlot: true });
  lineChart($("replay-chart"), {
    x: trace.t || [],
    yTitle: "m",
    series: [
      { name: "goal distance", color: "#22b8a6", y: trace.goal_dist || [] },
      { name: "clearance", color: "#e0a030", y: trace.clearance || [], dash: "dot" },
    ],
  });
}

async function renderTraceFromApi() {
  renderTrace(state.run.baseline_trace || null);
}

async function renderWeights(cycle) {
  try {
    const diff = await getRepairWeights(state.runId, cycle);
    const top = diff.top_units || [];
    $("weight-note").textContent = `${diff.n_added_edges || 0} new edges · ${(diff.changed_units || []).length} units`;
    $("weight-table").innerHTML = `<table><thead><tr><th>unit</th><th>|Δw_in|</th><th>|Δw_out|</th><th>Δbias</th></tr></thead>
      <tbody>${top.slice(0, 40).map((u) => `<tr><td>${u.unit}</td><td>${fmt(u.w_in_norm, 3)}</td>
      <td>${fmt(u.w_out_norm, 3)}</td><td>${fmt(u.bias, 3)}</td></tr>`).join("")}</tbody></table>`;
    Plotly.react($("weight-chart"), [{
      x: top.slice(0, 30).map((u) => `u${u.unit}`),
      y: top.slice(0, 30).map((u) => Math.max(u.w_in_norm, u.w_out_norm, Math.abs(u.bias))),
      type: "bar", marker: { color: "#b06cff" },
    }], {
      margin: { l: 44, r: 8, t: 6, b: 34 }, height: 140,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#c9d1d9", size: 9 }, showlegend: false,
      xaxis: { gridcolor: "#22303c" }, yaxis: { gridcolor: "#22303c", title: { text: "|Δ|", font: { size: 9 } } },
    }, { displayModeBar: false, responsive: true, staticPlot: true });
  } catch {
    $("weight-note").textContent = "no weight diff";
    $("weight-chart").innerHTML = "";
    $("weight-table").innerHTML = "";
  }
}

function renderFinal() {
  const f = state.run.final || {};
  const p = f.practice || {};
  const r = f.regression || {};
  const b = f.baseline_unseen || {};
  $("final").innerHTML = `
    <div class="metric-grid">
      <div class="metric"><b>${pct(p.near_success_rate ?? p.success_rate)}</b><span>unseen near-goal</span></div>
      <div class="metric"><b>${pct(r.near_success_rate ?? r.success_rate)}</b><span>nominal regression</span></div>
      <div class="metric"><b>${fmt(p.mean_final_progress, 3)}</b><span>unseen progress</span></div>
      <div class="metric"><b>${fmt(p.mean_final_goal_dist_m, 2)}</b><span>final goal dist</span></div>
    </div>
    <div class="pipe-note">unseen progress: baseline ${fmt(b.mean_final_progress, 3)} → final ${fmt(p.mean_final_progress, 3)}</div>
    <div class="pipe-note">bundle: <code>${esc(f.bundle || "—")}</code></div>`;
}

boot().catch((err) => setStatus(String(err), true));
