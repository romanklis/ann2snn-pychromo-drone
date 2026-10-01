"""Closed-loop simulation: controller + dynamics backend + telemetry.

Transport-agnostic (no Chrono import); the visualization layer consumes the
same :class:`Simulation`, and the headless CLI runs it directly.
"""

from __future__ import annotations

import csv
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import CONTROL_LIMIT, DT, GOAL_TOLERANCE, INIT_STATE, PLANT_GAIN, STEPS
from .dynamics import DynamicsBackend
from .reference import goal_reference
from .scene import Scene
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
        self._disturbance = disturbance
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> dict:
        self.backend.reset(self.initial_state)
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self.k = 0
        self.history: Dict[str, list] = {
            "state": [],
            "rpy": [],
            "command": [],
            "telemetry": [],
            "goal_dist": [],
            "clearance": [],
        }
        self._record(np.zeros(3))
        return self.observe()

    # -- stepping ----------------------------------------------------------- #
    def step(self, action: Optional[np.ndarray] = None) -> dict:
        if action is None:
            action = self.controller.act(self.state, self.reference.at(self.k))
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
        self.k += 1
        self._record(action)
        return self.observe()

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
        self.history["state"].append(state.copy())
        self.history["rpy"].append(rpy.copy())
        self.history["command"].append(np.asarray(command, dtype=np.float64).copy())
        self.history["telemetry"].append(dict(self.backend.telemetry))
        self.history["goal_dist"].append(tele["goal_dist"])
        self.history["clearance"].append(tele["clearance"])

    def observe(self) -> dict:
        idx = min(self.k, len(self.reference) - 1)
        tele = self.task.frame_telemetry(self.state, self.scene)
        return {
            "t": float(self.k * self.dt),
            "step": int(self.k),
            "done": bool(self.done),
            "state": self.state.astype(float),
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
        return {
            "controller": getattr(self.controller, "name", "controller"),
            "steps": int(self.k),
            "clearance_min_m": float(clearances.min()) if len(clearances) else float("nan"),
            "collisions": int(np.sum(clearances < 0.0)),
            "reached_goal": bool(goal_dists[-1] <= self.task.goal_tolerance)
            if len(goal_dists)
            else False,
            "final_goal_dist_m": float(goal_dists[-1]) if len(goal_dists) else float("nan"),
            "mean_goal_dist_m": float(goal_dists.mean()) if len(goal_dists) else float("nan"),
            "peak_g_force": float(np.nanmax(g_forces)) if g_forces else float("nan"),
            "soc_end_pct": float(soc[-1]) if soc else float("nan"),
        }

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
