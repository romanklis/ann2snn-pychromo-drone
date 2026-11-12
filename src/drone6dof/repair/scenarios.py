"""Distribution-shift scenarios and deterministic closed-loop episode runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (
    CONTROL_LIMIT,
    DS_TEACHER_KWARGS,
    ESTIMATOR,
    GOAL_TOLERANCE,
    INIT_STATE,
    PLANT_GAIN,
    SENSOR,
    SENSOR_SUITE,
    STEPS,
    preset_scene,
)
from ..connectome import ConnectomeController
from ..control import DSGuidanceController
from ..dynamics import NumpyPlantBackend
from ..estimator import ErrorStateUKF
from ..params import DT, QuadParams
from ..sensor import SensorConfig as LidarConfig
from ..sensors import SensorConfig as SuiteConfig
from ..sensors import SensorSuite
from ..sim import Simulation
from ..task import ObstacleGoalTask
from .types import EpisodeResult, SplitResult

#: Near-goal task success for repair research.  The shipped estimate-only SNN
#: and teacher often stop short of the strict 0.30 m goal tolerance, so repair
#: acceptance uses this looser, explicit completion radius and reports strict
#: ``reached_goal`` separately.
NEAR_TOLERANCE = 0.45

__all__ = [
    "NEAR_TOLERANCE",
    "Scenario",
    "SCENARIOS",
    "make_snn_controller",
    "make_teacher_controller",
    "run_episode",
    "evaluate_bundle",
    "evaluate_teacher",
    "auto_select_shift",
    "REGISTRY",
]


@dataclass
class Scenario:
    """One frozen distribution-shift definition.

    ``disturbance`` is a deterministic function of the step index; it is not
    serialized.  ``to_dict`` records only its name for the manifest.
    """

    name: str
    scene: Any
    description: str = ""
    initial_state: Tuple[float, ...] = INIT_STATE
    gain: float = PLANT_GAIN
    damping: float = 0.0
    sensor: LidarConfig = SENSOR
    sensor_suite: SuiteConfig = SENSOR_SUITE
    steps: int = STEPS
    goal_tolerance: float = GOAL_TOLERANCE
    disturbance: Optional[Callable[[int], Optional[np.ndarray]]] = None
    disturbance_name: str = "none"
    priority: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scene": self.scene.to_dict(),
            "initial_state": [float(v) for v in self.initial_state],
            "gain": float(self.gain),
            "damping": float(self.damping),
            "sensor": self.sensor.to_dict(),
            "sensor_suite": self.sensor_suite.to_dict(),
            "steps": int(self.steps),
            "goal_tolerance": float(self.goal_tolerance),
            "disturbance": self.disturbance_name,
            "priority": int(self.priority),
        }


def _wind(k: int) -> np.ndarray:
    t = float(k) * DT
    return np.array([0.9 * np.sin(0.7 * t), 0.7 * np.cos(0.5 * t), 0.0])


def _make_registry() -> Dict[str, Scenario]:
    nominal = Scenario(
        name="nominal",
        scene=preset_scene("pillar"),
        description="shipped pillar scene, in-distribution",
        priority=5,
    )
    degraded_sensor = replace(SENSOR, r_max=2.0, range_sigma=0.08, dropout_p=0.15)
    return {
        "nominal": nominal,
        "goal_behind_wall": Scenario(
            name="goal_behind_wall",
            scene=preset_scene("wall", goal=(1.5, 0.0, 2.5)),
            description="wall between the start and the goal",
            priority=10,
        ),
        "wall": Scenario(
            name="wall",
            scene=preset_scene("wall"),
            description="single long wall across the approach",
            priority=20,
        ),
        "slalom": Scenario(
            name="slalom",
            scene=preset_scene("slalom"),
            description="three-obstacle slalom layout not seen as a structured preset",
            priority=30,
        ),
        "boxes": Scenario(
            name="boxes",
            scene=preset_scene("boxes"),
            description="two-box preset layout",
            priority=40,
        ),
        "goal_shift": Scenario(
            name="goal_shift",
            scene=preset_scene("pillar", goal=(2.9, 2.9, 3.2)),
            description="goal moved outside the distillation goal box",
            priority=50,
        ),
        "lidar_degraded": Scenario(
            name="lidar_degraded",
            scene=preset_scene("pillar"),
            description="shorter, noisier and dropout-heavy LiDAR",
            sensor=degraded_sensor,
            priority=60,
        ),
        "wind": Scenario(
            name="wind",
            scene=preset_scene("pillar"),
            description="sinusoidal horizontal disturbance",
            disturbance=_wind,
            disturbance_name="sinusoidal",
            priority=70,
        ),
        "derated": Scenario(
            name="derated",
            scene=preset_scene("pillar"),
            description="reduced rotor authority and extra damping",
            gain=0.7,
            damping=0.6,
            priority=80,
        ),
    }


REGISTRY: Dict[str, Scenario] = _make_registry()

#: ordered public suite; nominal is excluded from auto-selection
SCENARIOS: Tuple[Scenario, ...] = tuple(
    sorted(
        (s for name, s in REGISTRY.items() if name != "nominal"),
        key=lambda s: s.priority,
    )
)


def make_snn_controller(bundle: Dict[str, Any], scenario: Scenario, seed: int, kind: str = "snn"):
    sensor = replace(scenario.sensor, seed=int(seed))
    task = ObstacleGoalTask(goal_tolerance=float(scenario.goal_tolerance))
    return ConnectomeController(
        kind,
        bundle,
        dt=DT,
        action_limit=CONTROL_LIMIT,
        task=task,
        sensor=sensor,
    )


def make_teacher_controller(scenario: Scenario):
    return DSGuidanceController(
        action_limit=CONTROL_LIMIT,
        scene=scenario.scene,
        task=ObstacleGoalTask(goal_tolerance=float(scenario.goal_tolerance)),
        **DS_TEACHER_KWARGS,
    )


def _make_suite(scenario: Scenario, seed: int):
    suite_cfg = replace(scenario.sensor_suite, seed=int(seed))
    lidar_cfg = replace(scenario.sensor, seed=int(seed))
    suite = SensorSuite(
        suite_cfg,
        lidar=lidar_cfg,
        obstacles=scenario.scene.obstacles,
        dt=DT,
    )
    return suite, suite_cfg


def _make_estimator(suite_cfg: SuiteConfig) -> ErrorStateUKF:
    return ErrorStateUKF(
        params=QuadParams(),
        sensor=suite_cfg,
        dt=DT,
        rotor_tau=float(ESTIMATOR["rotor_tau"]),
        rotor_omega_max=float(ESTIMATOR["rotor_omega_max"]),
    )


class RecordingController:
    """Delegate to a controller while recording all-neuron spikes and inputs."""

    def __init__(self, inner, *, capture: bool = True) -> None:
        self.inner = inner
        self.capture = bool(capture)
        self.spikes: List[Optional[np.ndarray]] = []
        self.inputs: List[Optional[np.ndarray]] = []
        self.scans: List[Optional[np.ndarray]] = []
        self.actions: List[np.ndarray] = []

    def act(self, state, ref):
        action = self.inner.act(state, ref)
        if self.capture:
            spikes = getattr(self.inner, "last_spikes_all", None)
            features = getattr(self.inner, "last_input", None)
            scan = getattr(self.inner, "last_scan", None)
            self.spikes.append(
                None if spikes is None else np.asarray(spikes, dtype=np.uint8).copy()
            )
            self.inputs.append(
                None if features is None else np.asarray(features, dtype=np.float64).copy()
            )
            self.scans.append(
                None if scan is None else np.asarray(scan, dtype=np.float64).copy()
            )
            self.actions.append(np.asarray(action, dtype=np.float64).copy())
        return action

    def reset(self):
        self.spikes = []
        self.inputs = []
        self.scans = []
        self.actions = []
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            return reset()
        return None

    def __getattr__(self, name):
        return getattr(self.inner, name)


def classify_failure(
    *,
    scenario: Scenario,
    reached_goal: bool,
    collisions: int,
    goal_dists: np.ndarray,
    state: np.ndarray,
    scan: Optional[np.ndarray],
    command: np.ndarray,
    trajectory_length_m: float,
) -> str:
    if collisions > 0:
        return "collision"
    if reached_goal:
        return "none"
    if len(goal_dists) < 2:
        return "timeout"

    goal = scenario.scene.goal_np
    final_speed = float(np.linalg.norm(np.asarray(state[-1], dtype=np.float64)[3:6]))
    closest = float(np.min(goal_dists))
    final = float(goal_dists[-1])
    last = np.asarray(state[-max(1, len(state) // 5):], dtype=np.float64)
    goal_dir = goal[:2] - last[:, :2]
    goal_dir_n = goal_dir / (np.linalg.norm(goal_dir, axis=1, keepdims=True) + 1e-9)
    vxy = last[:, 3:5]
    along = np.sum(vxy * goal_dir_n, axis=1)
    sat_frac = float(
        np.mean(np.linalg.norm(np.asarray(command, dtype=np.float64), axis=1)
                >= 0.999 * CONTROL_LIMIT)
    ) if len(command) else 0.0

    if sat_frac > 0.5:
        return "saturation"
    if final_speed < 0.18:
        return "stall_at_goal" if closest <= 0.8 else "stall_short"
    if np.mean(along) < -0.05:
        return "wrong_direction"
    tail = goal_dists[-max(2, len(goal_dists) // 5):]
    if float(np.std(tail)) < 0.15 and trajectory_length_m > 1.5 * (closest + 1.0):
        return "orbit"
    if final < closest + 0.05:
        return "orbit"
    return "timeout"


def run_episode(
    scenario: Scenario,
    controller,
    *,
    episode_id: int,
    seed: int,
    steps: Optional[int] = None,
    capture: bool = False,
) -> EpisodeResult:
    """Run one deterministic closed-loop episode.

    The caller supplies a controller built with the per-episode LiDAR seed.  This
    function seeds the onboard suite and estimator, so two calls with the same
    scenario/seed/controller weights are reproducible.
    """
    steps = int(scenario.steps if steps is None else steps)
    suite, suite_cfg = _make_suite(scenario, seed)
    estimator = _make_estimator(suite_cfg)
    backend = NumpyPlantBackend(QuadParams(), heading_target=scenario.scene.goal_np)
    recorder = RecordingController(controller, capture=bool(capture))
    sim = Simulation(
        backend,
        recorder,
        scenario.scene,
        dt=DT,
        steps=steps,
        gain=float(scenario.gain),
        damping=float(scenario.damping),
        control_limit=CONTROL_LIMIT,
        initial_state=scenario.initial_state,
        goal_tolerance=float(scenario.goal_tolerance),
        disturbance=scenario.disturbance,
        sensors=suite,
        estimator=estimator,
        map_source="truth",
    )
    sim.run()
    metrics = sim.metrics()
    history = sim.history

    state = np.asarray(history["state"], dtype=np.float64)
    goal_dists = np.asarray(history["goal_dist"], dtype=np.float64)
    clearance = np.asarray(history["clearance"], dtype=np.float64)
    command = np.asarray(history["command"], dtype=np.float64)
    reached = bool(metrics["reached_goal"])
    collisions = int(metrics["collisions"])
    success = bool(reached and collisions == 0)
    initial_dist = float(np.linalg.norm(
        np.asarray(scenario.initial_state, dtype=np.float64)[:3]
        - scenario.scene.goal_np
    ))
    closest = float(metrics["closest_goal_dist_m"])
    final_dist = float(metrics["final_goal_dist_m"])
    near_success = bool(collisions == 0 and final_dist <= NEAR_TOLERANCE)
    progress = float((initial_dist - closest) / max(initial_dist, 1e-9))
    final_progress = float((initial_dist - final_dist) / max(initial_dist, 1e-9))
    tol = float(scenario.goal_tolerance)
    hits = np.flatnonzero(goal_dists <= tol)
    time_to_goal = float(hits[0] * DT) if len(hits) else None
    sat_frac = float(
        np.mean(np.linalg.norm(command, axis=1) >= 0.999 * CONTROL_LIMIT)
    ) if len(command) else 0.0
    powers = [
        float(t.get("power_w"))
        for t in history["telemetry"]
        if t.get("power_w") is not None and np.isfinite(t.get("power_w"))
    ]
    mean_power = float(np.mean(powers)) if powers else float("nan")

    scan = None
    first_scan = next((s for s in history["scan"] if s is not None), None)
    if first_scan is not None:
        scan = np.asarray(
            [np.asarray(first_scan) if s is None else s for s in history["scan"]],
            dtype=np.float64,
        )
    failure = classify_failure(
        scenario=scenario,
        reached_goal=reached,
        collisions=collisions,
        goal_dists=goal_dists,
        state=state,
        scan=scan,
        command=command,
        trajectory_length_m=float(metrics["trajectory_length_m"]),
    )

    trace = None
    if capture:
        spikes = recorder.spikes
        inputs = recorder.inputs
        trace = {
            "t": [float(i * DT) for i in range(len(state))],
            "trajectory": state[:, :3].tolist(),
            "state": state.tolist(),
            "command": command.tolist(),
            "goal_dist": goal_dists.tolist(),
            "clearance": clearance.tolist(),
            "scan": None if scan is None else scan.tolist(),
            "spikes": (
                [None if s is None else s.astype(np.uint8).tolist() for s in spikes]
                if spikes else None
            ),
            "inputs": (
                [None if x is None else x.tolist() for x in inputs] if inputs else None
            ),
            "actions": [a.tolist() for a in recorder.actions],
            "spike_counts": (
                None if getattr(controller, "spike_counts", None) is None
                else np.asarray(controller.spike_counts, dtype=np.int64).tolist()
            ),
            "spike_frames": (
                None if getattr(controller, "spike_frames", None) is None
                else np.asarray(controller.spike_frames, dtype=np.int64).tolist()
            ),
        }

    return EpisodeResult(
        episode_id=int(episode_id),
        seed=int(seed),
        scenario=scenario.name,
        success=success,
        reached_goal=reached,
        collision_free=collisions == 0,
        final_goal_dist_m=final_dist,
        closest_goal_dist_m=closest,
        clearance_min_m=float(metrics["clearance_min_m"]),
        collisions=collisions,
        steps=int(metrics["steps"]),
        time_to_goal_s=time_to_goal,
        progress=progress,
        final_progress=final_progress,
        command_saturation_frac=sat_frac,
        mean_power_w=mean_power,
        near_success=near_success,
        failure=failure,
        trace=trace,
    )


def evaluate_bundle(
    scenario: Scenario,
    bundle: Dict[str, Any],
    seeds: Sequence[int],
    *,
    split: str = "practice",
    steps: Optional[int] = None,
    capture: bool = False,
    kind: str = "snn",
) -> SplitResult:
    out = SplitResult(scenario=scenario.name, split=split, steps=int(scenario.steps if steps is None else steps))
    for i, seed in enumerate(seeds):
        controller = make_snn_controller(bundle, scenario, int(seed), kind=kind)
        result = run_episode(
            scenario,
            controller,
            episode_id=i,
            seed=int(seed),
            steps=steps,
            capture=bool(capture and i == 0),
        )
        out.episodes.append(result)
    return out


def evaluate_teacher(
    scenario: Scenario,
    seeds: Sequence[int],
    *,
    split: str = "probe",
    steps: Optional[int] = None,
    capture: bool = False,
) -> SplitResult:
    out = SplitResult(scenario=scenario.name, split=split, steps=int(scenario.steps if steps is None else steps))
    for i, seed in enumerate(seeds):
        controller = make_teacher_controller(scenario)
        result = run_episode(
            scenario,
            controller,
            episode_id=i,
            seed=int(seed),
            steps=steps,
            capture=bool(capture and i == 0),
        )
        out.episodes.append(result)
    return out


def _probe_score(split: SplitResult) -> float:
    return float(split.near_success_rate) + 0.5 * max(
        0.0, float(split.mean_final_progress)
    )


def _episode_probe(episodes: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    return [
        {
            "seed": int(e.seed),
            "near_success": bool(e.near_success),
            "success": bool(e.success),
            "collision_free": bool(e.collision_free),
            "final_goal_dist_m": float(e.final_goal_dist_m),
            "final_progress": float(e.final_progress),
            "failure": e.failure,
        }
        for e in episodes
    ]


def auto_select_shift(
    bundle: Dict[str, Any],
    *,
    scenarios: Optional[Sequence[Scenario]] = None,
    probe_seeds: Sequence[int] = (1000, 1001, 1002),
    probe_steps: int = 300,
    max_snn_success: float = 0.20,
    max_snn_progress: float = 0.20,
    min_teacher_success: float = 0.33,
    min_teacher_progress: float = 0.60,
    min_teacher_collision_free: float = 0.60,
    min_gap: float = 0.35,
    strong_gap: float = 0.75,
) -> Tuple[Optional[Scenario], List[Dict[str, Any]]]:
    """Select the clearest teacher-beats-SNN shift.

    The old first-match rule was brittle: with three probe episodes, one missed
    near-goal seed made the teacher 2/3 and rejected an otherwise useful shift.
    This rule scores every scenario, applies explicit teacher/SNN floors and a
    minimum gap, then picks the largest gap.  Priority only breaks ties.
    """
    pool = list(SCENARIOS if scenarios is None else scenarios)
    summaries: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, int, Scenario, Dict[str, Any]]] = []
    for scenario in pool:
        snn = evaluate_bundle(
            scenario,
            bundle,
            probe_seeds,
            split="probe",
            steps=probe_steps,
        )
        teacher = evaluate_teacher(
            scenario,
            probe_seeds,
            split="probe",
            steps=probe_steps,
        )
        snn_score = _probe_score(snn)
        teacher_score = _probe_score(teacher)
        gap = teacher_score - snn_score
        teacher_ok = (
            teacher.near_success_rate >= float(min_teacher_success)
            and teacher.mean_final_progress >= float(min_teacher_progress)
            and teacher.collision_free_rate >= float(min_teacher_collision_free)
        )
        snn_bad = (
            snn.near_success_rate <= float(max_snn_success)
            and snn.mean_final_progress <= float(max_snn_progress)
        )
        eligible = teacher_ok and snn_bad and gap >= float(min_gap)
        reasons: List[str] = []
        if not teacher_ok:
            reasons.append(
                "teacher floors: "
                f"near={teacher.near_success_rate:.2f} "
                f"progress={teacher.mean_final_progress:.2f} "
                f"collision_free={teacher.collision_free_rate:.2f}"
            )
        if not snn_bad:
            reasons.append(
                "SNN not bad enough: "
                f"near={snn.near_success_rate:.2f} "
                f"progress={snn.mean_final_progress:.2f}"
            )
        if gap < float(min_gap):
            reasons.append(f"gap {gap:.2f} < {float(min_gap):.2f}")
        summary = {
            "scenario": scenario.name,
            "description": scenario.description,
            "priority": int(scenario.priority),
            "snn": snn.metrics(),
            "teacher": teacher.metrics(),
            "snn_score": float(snn_score),
            "teacher_score": float(teacher_score),
            "gap": float(gap),
            "teacher_ok": bool(teacher_ok),
            "snn_bad": bool(snn_bad),
            "eligible": bool(eligible),
            "selected": False,
            "reasons": reasons,
            "snn_episodes": _episode_probe(snn.episodes),
            "teacher_episodes": _episode_probe(teacher.episodes),
        }
        summaries.append(summary)
        if eligible:
            candidates.append((gap, int(scenario.priority), scenario, summary))
            # Fast path: the first priority scenario with a clear gap is enough.
            # Otherwise score the whole suite and choose the largest gap.
            if gap >= float(strong_gap):
                summary["selected"] = True
                return scenario, summaries
    if not candidates:
        return None, summaries
    # Largest gap wins; lower priority value breaks ties.
    _, _, selected_scenario, selected_summary = sorted(
        candidates, key=lambda item: (-item[0], item[1])
    )[0]
    selected_summary["selected"] = True
    return selected_scenario, summaries
