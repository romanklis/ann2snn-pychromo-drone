# White-space repair: editing the connectome SNN

Source of truth: `src/drone6dof/repair/`, `tools/repair.py`,
`server/repair_api.py`, `web/src/repair.js`. The design follows the
weight-space-programming idea from
[J. Stranghöner et al., "Repairing Robot Policies with Weight Space Programming"](https://jstranghoener.github.io/blog/robotic-weight-space-programming/),
adapted to this repo's sparse recurrent SNN.

## 1. What it is

The repair harness treats the distilled **1000-neuron connectome SNN** as a
policy that can be corrected by writing small programs into neurons that the
network does not use. It is deliberately **gradient-free**: no optimizer, no
weight fitting, and no new demonstrations. A candidate edit is accepted only
after closed-loop evaluation on fixed practice episodes; a final score is
reported on disjoint unseen episodes.

The default contract is:

- **Target**: `snn` (the rate-coded IF transfer), not the field SNN.
- **Edits**: only spare (silent/rare) neurons, with bounded parameter and edge
  budgets.
- **Proposer**: deterministic template by default; an LLM can be enabled.
- **Evaluation**: fixed practice/regression/unseen splits, paired episodes.
- **Artifacts**: every cycle is logged; the `/repair` page replays the run.

A future gradient proposer is an ablation, not white-space programming. It is
not implemented in this build.

## 2. Why the SNN has white space

The connectome network is 1000 neurons with fan-in 40 (40,000 recurrent edges),
dense `w_in` (1000×46) and dense `w_out` (3×1000); recurrence is a fixed sparse
edge list (`src/drone6dof/connectome.py:87-99`, `:138-201`). On the deployment
sequence most neurons never spike: the transfer evaluation reports ~92% silent
neurons (`docs/limitations.md:33-37`,
`tools/snn_transfer_eval.py:110-117`).

Because `w_in` and `w_out` are dense, a spare neuron can read any of the 46 input
features and drive any action without changing topology. Only temporal
logic/latching needs recurrent edges; the DSL supports a bounded number of
appended edges while preserving Dale's law.

The base network has no bias term. A repair-only `bias` vector is added to the
SNN (zeros in the base bundle) so a spare unit can implement a threshold:

```text
i_syn = w_in @ x + bias + W_rec @ s
```

## 3. White-space audit

`repair/activity.py` runs a short episode, captures all 1000 spike vectors and
the 46-D inputs, and writes:

- `spike_counts`, `spike_frames`, `mean_rate`, `contribution`;
- `silent` = zero spikes; `rare` = active in <1% of frames;
- `spare` = silent or rare;
- `critical` = high readout contribution (protected by default);
- per-unit input/output norms, in/out degree, polarity and top features.

The audit is written to `baseline/activity.npz` and `baseline/audit.json`.

## 4. Edit DSL

`repair/edits.py` implements a small, validated DSL:

| Op | Meaning |
|---|---|
| `set_input_row` / `add_input_row` | write a spare unit's 46-D input row |
| `set_output_col` / `add_output_col` | write a spare unit's 3-D readout column |
| `set_bias` / `add_bias` | shift the unit threshold |
| `add_edge` / `set_edge` | bounded recurrent edge insertion/update |
| `silence_output` | zero the unit's old readout |
| `isolate_unit` | zero recurrent synapses touching the unit (removes interference) |

Guard rails:

- only spare units by default (`--allow-live` overrides);
- no polarity changes (Dale's law);
- finite values, bounded magnitudes, max units/edges/parameters per cycle;
- **quietness**: the edit's action delta on the fixed in-distribution demos is
  reported as a warning by default; pass `--strict-quiet` to make it a hard
  validation failure;
- **liveness**: edited units must fire on the target traces, otherwise the
  program is flagged as inert (warning);
- repurposed units are **auto-isolated**: existing recurrent synapses touching a
  unit whose input/readout/bias is rewritten are zeroed before the ops; explicit
  `add_edge`/`set_edge` ops still apply afterwards;
- rejected multi-unit programs are screened as small unit subsets (singles first),
  and the quietest hard-valid subset is evaluated; accepted subset/pruned units
  are recorded;
- repaired bundles keep the base fingerprint; `bias` and `repair_json` are stored
  separately in the npz.

## 5. Repair cycle

```text
baseline probe -> auto-select shift -> audit white space
repeat:
  diagnose -> propose -> apply -> validate -> evaluate -> accept/revert -> log
final unseen + nominal regression -> save repaired bundle + report
```

The Make targets default to `goal_shift`: a genuine goal-box shift where the
SNN still moves without collisions, unlike `goal_behind_wall`, where it starts
by going the wrong way. `REPAIR_SCENARIO=auto` still uses the max-gap probe and
may choose `goal_behind_wall`; for now prefer a named scenario such as
`goal_shift`, `nominal`, or `lidar_degraded`.

Auto-selection uses the pre-registered suite in `repair/scenarios.py`:
`goal_behind_wall`, `wall`, `slalom`, `boxes`, `goal_shift`, `lidar_degraded`,
`wind`, `derated`. It scores every scenario with

```text
score = near-goal rate + 0.5 * max(0, mean final progress)
gap   = teacher score - SNN score
```

and selects the largest eligible gap. Default floors are: teacher near-goal
>= 0.33 (at least one probe seed), teacher progress >= 0.60, teacher
collision-free >= 0.60, SNN near-goal <= 0.20, SNN progress <= 0.20, and gap
>= 0.35. The near-goal floor is deliberately low because it is a hard 0.45 m
threshold and can be noisy: the teacher can finish at 0.455–0.469 m with
progress ~0.91 and still be a valid repair reference. Progress, collision-free
rate, and the gap carry the decision. A gap of at least 0.75 short-circuits the
priority order; otherwise every scenario is scored and the largest eligible gap
wins. All floors, `--probe-episodes`, `--probe-steps`, and `--probe-strong-gap`
are configurable; probe seeds are disjoint from practice seeds.

If nothing qualifies, the run writes `probe.json` and `probe.md`/`report.md`
with per-episode evidence and reasons, and the CLI prints the table. Use the
suggested named scenario to bypass auto-selection, for example:

```bash
make repair-deepseek REPAIR_SCENARIO=goal_behind_wall REPAIR_CYCLES=4
```

The `/repair` page also renders the probe table for a failed run.

Metrics:

```text
strict success = final goal distance <= 0.30 m and min clearance >= 0
near-goal      = final goal distance <= 0.45 m and min clearance >= 0
utility        = mean final progress toward the goal
```

Strict success is the repo's `reached_goal` definition. In estimate-only mode the
shipped SNN and even the DS teacher often stop just short of 0.30 m
(`docs/limitations.md`), so repair acceptance is **monotonic on mean final
progress**: any strict increase over the current best (`--accept-epsilon`, default
`1e-6`) is accepted, subject to hard safety guards. Both strict and near-goal
rates are always reported; do not present near-goal as strict success.

The loop also records reached rate, collision-free rate, progress, time-to-goal,
command saturation, power, and a failure taxonomy (`collision`, `stall_short`,
`stall_at_goal`, `orbit`, `wrong_direction`, `overshoot`, `saturation`,
`sensor_blind`, `timeout`).

Acceptance is the gradient-descent analogue on the primary objective: accept any
strict improvement in mean final progress. Hard guards prevent a nominal or
safety regression: practice collision-free rate and nominal-regression
success/near-goal/collision-free/progress may not drop by more than the
regression margin. Quietness and liveness are warnings unless `--strict-quiet`
is set.

Practice, regression and unseen episodes use the **same 500-step horizon** by
default; `--fast` uses 300 steps for smoke tests only and must not be used to
claim an acceptance. The final report always includes the unseen result even
when the target is not met. A partial progress repair is not task success; state
that explicitly.

## 6. LLM proposer

The LLM is a proposer, not a trainer. It never runs the simulator; it sees a
compact diagnosis, the white-space audit, previous cycles, and a set of budgeted
tools:

```text
list_features, get_activity, list_spare_units, read_weights, probe_unit,
validate_program, run_practice (optionally with a candidate ops list),
submit_program
```

`validate_program` applies a candidate and returns hard issues, warnings,
quietness, liveness and auto-isolated units without spending practice rollouts.
A single `run_practice` call is capped (`--max-practice-per-call`, default 6) so
the model reserves budget to validate the final candidate. When the practice
budget is exhausted or the turn limit is near, the proposer is told to submit;
if it still does not, the last hard-valid validated program is submitted as a
fallback. The harness validates every op, executes rollouts, charges tool calls
and practice passes to the budget, and writes the full transcript to
`cycles/NNNN.transcript.json`.

The transport is OpenAI-compatible. OpenAI is the default; DeepSeek is an
explicit provider. Provider precedence is `--llm-provider` >
`REPAIR_LLM_PROVIDER` > `openai`.

OpenAI / generic OpenAI-compatible endpoint:

```bash
export REPAIR_LLM_API_KEY=...
export REPAIR_LLM_BASE_URL=https://api.openai.com/v1   # or a vLLM/Ollama proxy
export REPAIR_LLM_MODEL=gpt-4o-mini
```

DeepSeek:

```bash
export DEEPSEEK_API_KEY=...                            # preferred
# export REPAIR_LLM_API_KEY=...                        # fallback
# export DEEPSEEK_BASE_URL=https://api.deepseek.com    # or .../v1
export DEEPSEEK_MODEL=deepseek-chat                    # default; tool calling
# export DEEPSEEK_MODEL=deepseek-reasoner               # JSON-only fallback
make repair-deepseek REPAIR_CYCLES=8
```

DeepSeek `deepseek-chat` uses the same budgeted tool loop. `deepseek-reasoner`
does not support function calling, so the proposer omits tools and asks for the
final JSON program in the message content; the harness still validates it
exactly like a tool-submitted program. `deepseek-reasoner` uses a compact prompt
and an 8192-token output budget so the final JSON fits after the reasoning
trace. Its reasoning is kept in the transcript but stripped before any follow-up
API request.

`make repair*` loads `.env` from the repo root (gitignored) and exports these
variables to the Docker command. Start from `.env.example`:

```bash
cp .env.example .env
# edit .env with DEEPSEEK_API_KEY=...
make repair-deepseek REPAIR_CYCLES=8
```

Equivalent generic form:

```bash
export REPAIR_LLM_PROVIDER=deepseek
make repair-llm REPAIR_CYCLES=8
```

Use `--proposer llm`; without a key the template proposer remains the default.

## 7. CLI and Make targets

```bash
python tools/repair.py audit --scenario auto
python tools/repair.py run --scenario auto --proposer template --cycles 4
python tools/repair.py run --scenario slalom --proposer llm --cycles 8
python tools/repair.py runs
python tools/repair.py report --run-id repair-2026...
```

Docker targets:

```bash
make repair-audit
make repair REPAIR_SCENARIO=slalom REPAIR_CYCLES=4
make repair-llm REPAIR_CYCLES=8
make repair-deepseek REPAIR_CYCLES=8       # DEEPSEEK_API_KEY
make repair-runs
make repair-report RUN_ID=repair-...
make repair-test
```

Budgets are configurable (`--practice-episodes`, `--practice-steps`,
`--unseen-episodes`, `--max-units`, `--max-new-edges`,
`--max-param-change`, `--max-tool-calls`, `--max-practice-rollouts`,
`--max-practice-per-call`). Acceptance and validation controls:
`--accept-epsilon`, `--strict-quiet`, `--no-prune`, `--max-prunes`. Practice,
regression and unseen default to 500 steps; `--fast` uses 300 for smoke tests.
The defaults are sized for a CPU-only SNN; profile before raising them.

## 8. Resuming refinement

Refinement can continue from a completed run’s **final repaired bundle** and
append new cycles to the same run:

```bash
make repair-deepseek REPAIR_RESUME=repair-20261112-130248 REPAIR_CYCLES=2
```

Resume semantics:

- the saved scenario and split definitions/seeds are restored; a conflicting
  scenario is ignored in favor of the saved run;
- new cycles start at `max(existing cycle)+1`; existing cycle files are never
  overwritten;
- the original `baseline/` remains the cumulative safety guard;
- the resumed bundle becomes the new `best_score` starting point, so acceptance
  continues monotonically from the current state;
- the previous `final/`, `report.md`, and repaired bundle are copied to
  `resumes/round-<N>/` before the new final is written;
- only a completed run (`status: ok`) with a final bundle can resume. Resuming
  from an arbitrary cycle snapshot or replaying the LLM transcript is not
  supported;
- one writer at a time: do not resume the same run concurrently.

Manual warm start without history continuity is still available for a new run:

```bash
python tools/repair.py \
  --bundle weights/repair/repair-20261112-130248.npz \
  run --scenario goal_shift
```

## 9. Inspector

`/repair` reads artifacts from `DRONE6DOF_REPAIR_ROOT` (default `out/repair`) and
shows:

- run summary and baseline/final scores;
- success by cycle with accepted/rejected markers;
- cycle comparison: near-goal, progress, collision-free and final-distance
  curves, plus a 3-D trajectory overlay of the baseline, accepted cycles and the
  latest cycle (toggle rejected cycles);
- cycle table and failure taxonomy;
- 1000-neuron white-space strip (silent/rare/live/critical/edited);
- 3-D replay plus goal distance/clearance;
- weight deltas for the selected cycle;
- the program and (for LLM runs) the transcript.

API routes (read-only, no torch):

```text
GET /api/repair/runs
GET /api/repair/runs/<run_id>
GET /api/repair/runs/<run_id>/cycles/<n>
GET /api/repair/runs/<run_id>/weights/<n>
GET /api/repair/runs/<run_id>/activity
GET /api/repair/runs/<run_id>/events?since=<n>
GET /repair
```

The page polls the run artifact; it does not run the loop. Start the loop from
the CLI (or a separate service) while the dashboard serves the artifacts:

```bash
# terminal 1
make repair REPAIR_SCENARIO=auto REPAIR_CYCLES=8
# terminal 2
make dashboard            # open http://localhost:8080/repair
```

`out/` is mounted into the dashboard container, so cycles appear as they are
written. The page is read-only and needs no torch.

## 10. Files and limitations

```text
src/drone6dof/repair/
  types.py          dataclasses for budgets, edits, episodes, cycles
  scenarios.py      shift suite, deterministic episode runner, auto-select
  activity.py       per-neuron audit and spare/critical sets
  edits.py          DSL, apply/validate/diff, repaired bundle save
  proposers/        template, random, LLM adapter/tools, gradient stub
  artifacts.py      run directories, JSON/JSONL, compact traces, loaders
  loop.py           the repair loop
tools/repair.py     CLI
server/repair_api.py
web/repair.html, web/src/repair.js
tests/test_repair_core.py, server/tests/test_repair_api.py
```

Limitations:

- The SNN is a rate-coded finite-window approximation; programs are calibrated
  and validated in closed loop, not assumed to transfer from the ANN.
- The fixed sparse recurrence limits temporal programs; `bias` and bounded edge
  insertion are the escape hatch.
- Silent units are not guaranteed to be unused out of distribution; quietness,
  liveness and nominal regression are the guard rails.
- There is no formal safety guarantee. The repaired bundle is a research
  artifact; safety still comes only from command clipping and plant saturation.
- The final unseen split is finite. Report it honestly; do not claim general
  repair beyond the evaluated shifts.
