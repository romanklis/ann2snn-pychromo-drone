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

export async function runBenchmark(controllers, { steps = 500, seed = 0, goal = null } = {}) {
  const payload = { controllers, steps, seed };
  if (Array.isArray(goal) && goal.length === 3) payload.goal = goal;
  const response = await fetch("/api/benchmark", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson(response);
}
