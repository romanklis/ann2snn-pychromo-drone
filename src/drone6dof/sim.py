"""Closed-loop simulation: controller + dynamics backend + telemetry.

Transport-agnostic (no Chrono import); the visualization layer consumes the
same :class:`Simulation`, and the headless CLI runs it directly.
"""

from __future__ import annotations

import csv
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import (
    CONTROL_LIMIT,
    DT,
    GOAL_TOLERANCE,
    INIT_STATE,
    MAP_SOURCE_DEFAULT,
    PLANT_GAIN,
    SENSOR,
    SLAM,
    STEPS,
)
from .dynamics import DynamicsBackend
from .estimator import ErrorStateUKF
from .params import QuadParams
from .reference import goal_reference
from .scene import Scene
from .sensors import Observation, SensorConfig, SensorSuite
from .slam import OccupancyMap, SlamConfig, truth_shell
from .task import ObstacleGoalTask

#: telemetry columns always present in the CSV export, in order
_BASE_TELEMETRY = [
    "goal_dist",
    "clearance",
    "lyapunov_v",
    "lyapunov_vdot",
]

_STATE_COLUMNS = [
    "px", "py", "pz",
    "vx", "vy", "vz",
    "roll", "pitch", "yaw",
    "ux", "uy", "uz",
]


class Simulation:
    """One plant + one controller + one reference, stepped frame by frame."""

    def __init__(
        self,
        backend: DynamicsBackend,
        controller,
        scene: Scene,
        *,
        dt: float = DT,
        steps: int = STEPS,
        gain: float = PLANT_GAIN,
        damping: float = 0.0,
        control_limit: float = CONTROL_LIMIT,
        initial_state=INIT_STATE,
        goal_tolerance: float = GOAL_TOLERANCE,
        disturbance: Optional[Callable[[int], Optional[np.ndarray]]] = None,
        use_estimator: bool = True,
        estimator=None,
        sensors=None,
        map_source: str = MAP_SOURCE_DEFAULT,
    ) -> None:
        self.backend = backend
        self.controller = controller
        self.scene = scene
        self.dt = float(dt)
        self.steps = int(steps)
        self.gain = float(gain)
        self.damping = float(damping)
        self.control_limit = float(control_limit)
        self.initial_state = np.asarray(initial_state, dtype=np.float64).reshape(6)
        self.task = ObstacleGoalTask(goal_tolerance=float(goal_tolerance))
        self.reference = goal_reference(
            steps=self.steps, goal=scene.goal, dt=self.dt, meta={"scene": scene}
        )
        self.map_source = str(map_source or "truth")
        self.slam_config = SlamConfig.from_dict(SLAM)
        self.slam_map: Optional[OccupancyMap] = None
        self._truth_occ = None
        if self.map_source == "slam":
            self.slam_map = OccupancyMap(self.slam_config)
            self._truth_occ = truth_shell(scene, self.slam_map)
            # the mapper is shared with the controller through the reference meta
            self.reference.meta["map"] = self.slam_map
        self._disturbance = disturbance
        # Estimator-in-the-loop: the plant is the hidden truth; controllers only
        # ever consume the UKF estimate.
        self.use_estimator = bool(use_estimator)
        self.sensor_config = SensorConfig()
        self.sensors = sensors
        self.estimator = estimator
        if self.use_estimator:
            plant = getattr(backend, "plant", None)
            params = getattr(plant, "p", None) or QuadParams()
            if self.sensors is None:
                self.sensors = SensorSuite(
                    self.sensor_config, lidar=SENSOR,
                    obstacles=getattr(scene, "obstacles", ()), dt=self.dt,
                )
            if self.estimator is None:
                self.estimator = ErrorStateUKF(params=params, sensor=self.sensor_config, dt=self.dt)
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> dict:
        self.backend.reset(self.initial_state)
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        if self.use_estimator and self.estimator is not None:
            self.estimator.reset(self.initial_state[:3], self.initial_state[3:6])
            if self.sensors is not None:
                self.sensors.reset()
            self._est_state = self.estimator.state_view()
            self._gps_ok = True
        else:
            self._est_state = self.initial_state.copy()
            self._gps_ok = True
        if self.map_source == "slam":
            self.slam_map = OccupancyMap(self.slam_config)
            self._truth_occ = truth_shell(self.scene, self.slam_map)
            self.reference.meta["map"] = self.slam_map
        self.k = 0
        self.history: Dict[str, list] = {
            "state": [],
            "rpy": [],
            "command": [],
            "telemetry": [],
            "goal_dist": [],
            "clearance": [],
            "spikes": [],        # first 200 neurons per frame, for spiking controllers
            "scan": [],          # raw LiDAR ranges per frame, for sensor controllers
            "estimate": [],      # estimated [p, v] per frame
            "est_error": [],     # true p - estimated p
            "map_frames": [],    # dense occupancy snapshots (slam only)
            "map_times": [],
            "slam_explored": [],
            "slam_entropy": [],
            "slam_occupied": [],
        }
        self._record(np.zeros(3))
        return self.observe()

    # -- stepping ----------------------------------------------------------- #
    def step(self, action: Optional[np.ndarray] = None) -> dict:
        if action is None:
            # The controller only ever sees the estimate (or the raw initial state
            # when estimation is disabled), never the hidden truth.
            obs = Observation(state=self._est_state.copy(), gps_ok=self._gps_ok)
            action = self.controller.act(obs, self.reference.at(self.k))
        action = np.asarray(action, dtype=np.float64).reshape(3)
        dist = None
        if self._disturbance is not None:
            dist = self._disturbance(self.k)
        self.backend.step(
            action,
            dt=self.dt,
            limit=self.control_limit,
            gain=self.gain,
            damping=self.damping,
            disturbance=dist,
        )
        if self.use_estimator and self.estimator is not None and self.sensors is not None:
            plant = getattr(self.backend, "plant", None)
            params = getattr(plant, "p", None)
            p = params if params is not None else QuadParams()
            truth = {
                "p": self.backend.position,
                "v": np.asarray(self.backend.state, dtype=np.float64)[3:6],
                "R": self.backend.rotation,
                "omega": self.backend.omega,
                "omega_m": self.backend.omega_m,
                "mass": p.m,
                "C_T": p.C_T,
                "specific_force_body": self.backend.specific_force_body,
            }
            meas = self.sensors.measure(self.k + 1, truth)
            rotor_target = getattr(self.backend, "rotor_target", np.full(4, p.hover_omega))
            self._est_state = self.estimator.step(meas, rotor_target)
            self._gps_ok = bool(meas.gps_ok)
            self._update_map(getattr(meas, "scan", None), getattr(meas, "scan_angles", None))
        else:
            self._est_state = np.asarray(self.backend.state, dtype=np.float64).copy()
            self._update_map(getattr(self.controller, "last_scan", None),
                             getattr(self.controller, "scan_angles", None))
        self.k += 1
        self._record(action)
        return self.observe()

    def _update_map(self, scan, angles) -> None:
        """Fuse one LiDAR scan into the SLAM map at the estimated pose."""
        if self.slam_map is None or scan is None or angles is None:
            return
        pose = np.asarray(self._est_state, dtype=np.float64).reshape(-1)[:2]
        self.slam_map.update(pose, scan, angles, SENSOR.r_max)

    def step_n(self, n: int = 1) -> dict:
        obs = self.observe()
        for _ in range(int(n)):
            obs = self.step()
        return obs

    def run(self) -> "Simulation":
        while self.k < self.steps:
            self.step()
        return self

    # -- state -------------------------------------------------------------- #
    @property
    def state(self) -> np.ndarray:
        return self.backend.state

    @property
    def position(self) -> np.ndarray:
        return self.backend.position

    @property
    def rotation(self) -> np.ndarray:
        return self.backend.rotation

    @property
    def done(self) -> bool:
        return self.k >= self.steps

    def _record(self, command: np.ndarray) -> None:
        state = self.state
        rpy = np.asarray(
            getattr(self.backend, "attitude_rpy", np.zeros(3)), dtype=np.float64
        )
        tele = self.task.frame_telemetry(state, self.scene)
        telemetry = dict(self.backend.telemetry)
        telemetry.update(getattr(self.controller, "last_telemetry", {}) or {})
        spikes = getattr(self.controller, "last_spikes", None)
        self.history["state"].append(state.copy())
        self.history["rpy"].append(rpy.copy())
        self.history["command"].append(np.asarray(command, dtype=np.float64).copy())
        self.history["telemetry"].append(telemetry)
        self.history["goal_dist"].append(tele["goal_dist"])
        self.history["clearance"].append(tele["clearance"])
        self.history["spikes"].append(
            None if spikes is None else np.asarray(spikes, dtype=np.uint8)[:200].copy()
        )
        scans = getattr(self.controller, "last_scan", None)
        self.history["scan"].append(
            None if scans is None else np.asarray(scans, dtype=np.float32).copy()
        )
        est = np.asarray(self._est_state, dtype=np.float64).reshape(-1)
        self.history["estimate"].append(est.copy())
        self.history["est_error"].append(state[:3] - est[:3])
        if self.slam_map is not None:
            self.history["slam_explored"].append(self.slam_map.explored_frac())
            self.history["slam_entropy"].append(self.slam_map.entropy_bits())
            self.history["slam_occupied"].append(self.slam_map.occupied_cells())
            if self.k % max(1, int(self.slam_map.cfg.viz_stride)) == 0:
                self.history["map_frames"].append(self.slam_map.dense().astype(int).tolist())
                self.history["map_times"].append(float(self.k * self.dt))

    def observe(self) -> dict:
        idx = min(self.k, len(self.reference) - 1)
        tele = self.task.frame_telemetry(self.state, self.scene)
        return {
            "t": float(self.k * self.dt),
            "step": int(self.k),
            "done": bool(self.done),
            "state": self.state.astype(float),
            "estimate": np.asarray(self._est_state, dtype=float),
            "rpy": np.asarray(
                getattr(self.backend, "attitude_rpy", np.zeros(3)), dtype=float
            ),
            "reference": self.reference.at(idx).to_dict(),
            "goal_dist": tele["goal_dist"],
            "clearance": tele["clearance"],
            "telemetry": dict(self.backend.telemetry),
        }

    # -- reporting ---------------------------------------------------------- #
    def trajectory(self) -> np.ndarray:
        return np.asarray(self.history["state"], dtype=np.float64)

    def metrics(self) -> dict:
        traj = self.trajectory()
        clearances = self.scene.clearance_series(traj)
        goal = self.scene.goal_np
        goal_dists = np.linalg.norm(traj[:, :3] - goal, axis=1)
        tele = self.history["telemetry"]
        g_forces = [t.get("g_force", float("nan")) for t in tele]
        soc = [t.get("soc_pct", float("nan")) for t in tele]
        latencies = [t.get("latency_ms") for t in tele if t.get("latency_ms") is not None]
        rates = [t.get("spike_rate_hz") for t in tele if t.get("spike_rate_hz") is not None]
        step_dists = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1) if len(traj) > 1 else np.zeros(0)
        cmd = np.asarray(self.history["command"], dtype=np.float64)
        smooth = (np.linalg.norm(np.diff(cmd, axis=0), axis=1)
                  if len(cmd) > 1 else np.zeros(0))
        out = {
            "controller": getattr(self.controller, "name", "controller"),
            "steps": int(self.k),
            "clearance_min_m": float(clearances.min()) if len(clearances) else float("nan"),
            "collisions": int(np.sum(clearances < 0.0)),
            "reached_goal": bool(goal_dists[-1] <= self.task.goal_tolerance)
            if len(goal_dists)
            else False,
            "final_goal_dist_m": float(goal_dists[-1]) if len(goal_dists) else float("nan"),
            "closest_goal_dist_m": float(goal_dists.min()) if len(goal_dists) else float("nan"),
            "mean_goal_dist_m": float(goal_dists.mean()) if len(goal_dists) else float("nan"),
            "trajectory_length_m": float(step_dists.sum()) if len(step_dists) else float("nan"),
            "command_smoothness": float(smooth.mean()) if len(smooth) else float("nan"),
            "latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
            "spike_rate_hz": float(np.mean(rates)) if rates else float("nan"),
            "peak_g_force": float(np.nanmax(g_forces)) if g_forces else float("nan"),
            "soc_end_pct": float(soc[-1]) if soc else float("nan"),
        }
        if self.slam_map is not None:
            out["map_source"] = "slam"
            out["slam_replans"] = int(self.slam_map.replans)
            out["slam_explored_frac"] = (
                float(self.history["slam_explored"][-1]) if self.history["slam_explored"] else 0.0
            )
            out["slam_iou_vs_truth"] = float(self.slam_map.iou(self._truth_occ))
            out["slam_path_found_step"] = (
                int(self.slam_map.path_found_step)
                if self.slam_map.path_found_step is not None else -1
            )
        return out

    def telemetry_columns(self) -> List[str]:
        keys = set(_BASE_TELEMETRY)
        for tele in self.history["telemetry"]:
            keys.update(tele.keys())
        return _BASE_TELEMETRY + sorted(k for k in keys if k not in _BASE_TELEMETRY)

    def to_csv(self, path: str) -> str:
        """Write the per-frame state, command and telemetry to ``path``."""
        cols = _STATE_COLUMNS + self.telemetry_columns()
        n = len(self.history["state"])
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t"] + cols)
            for i in range(n):
                state = self.history["state"][i]
                rpy = self.history["rpy"][i]
                cmd = self.history["command"][i]
                tele = dict(self.history["telemetry"][i])
                tele["goal_dist"] = self.history["goal_dist"][i]
                tele["clearance"] = self.history["clearance"][i]
                row = [float(i * self.dt)]
                row.extend(float(v) for v in state)
                row.extend(float(v) for v in rpy)
                row.extend(float(v) for v in cmd)
                for c in cols[len(_STATE_COLUMNS):]:
                    v = tele.get(c, "")
                    row.append("" if v == "" else float(v))
                writer.writerow(row)
        return path

    def to_npz(self, path: str) -> str:
        tele_cols = self.telemetry_columns()
        tele_mat = np.full(
            (len(self.history["state"]), len(tele_cols)), np.nan, dtype=np.float64
        )
        for i, tele in enumerate(self.history["telemetry"]):
            for j, c in enumerate(tele_cols):
                tele_mat[i, j] = tele.get(c, np.nan)
        np.savez(
            path,
            t=np.arange(len(self.history["state"])) * self.dt,
            state=self.trajectory(),
            rpy=np.asarray(self.history["rpy"], dtype=np.float64),
            command=np.asarray(self.history["command"], dtype=np.float64),
            goal_dist=np.asarray(self.history["goal_dist"], dtype=np.float64),
            clearance=np.asarray(self.history["clearance"], dtype=np.float64),
            telemetry=tele_mat,
            telemetry_columns=np.asarray(tele_cols, dtype=object),
        )
        return path
