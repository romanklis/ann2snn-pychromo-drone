async function readJson(response) {
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    const hint = response.ok
      ? `invalid JSON (${text.length} bytes; first 80: ${text.slice(0, 80)})`
      : `HTTP ${response.status}`;
    throw new Error(hint);
  }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

export async function getControllers() {
  return readJson(await fetch("/api/controllers"));
}

export async function getHealth() {
  return readJson(await fetch("/api/health"));
}

export async function getRepairRuns() {
  return readJson(await fetch("/api/repair/runs"));
}

export async function getRepairRun(runId) {
  return readJson(await fetch(`/api/repair/runs/${encodeURIComponent(runId)}`));
}

export async function getRepairCycle(runId, cycle) {
  return readJson(await fetch(`/api/repair/runs/${encodeURIComponent(runId)}/cycles/${cycle}`));
}

export async function getRepairActivity(runId) {
  return readJson(await fetch(`/api/repair/runs/${encodeURIComponent(runId)}/activity`));
}

export async function getRepairTraces(runId) {
  return readJson(await fetch(`/api/repair/runs/${encodeURIComponent(runId)}/traces`));
}

export async function getRepairWeights(runId, cycle) {
  return readJson(await fetch(`/api/repair/runs/${encodeURIComponent(runId)}/weights/${cycle}`));
}

export async function runBenchmark(controllers, { steps = 500, seed = 0, goal = null, scene = null, mapSource = null, from = null, to = null } = {}) {
  const payload = { controllers, steps, seed };
  if (Array.isArray(goal) && goal.length === 3) payload.goal = goal;
  if (scene) payload.scene = scene;
  if (mapSource) payload.map_source = mapSource;
  if (from !== null && from !== undefined) payload.from = from;
  if (to !== null && to !== undefined) payload.to = to;
  const response = await fetch("/api/benchmark", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson(response);
}
