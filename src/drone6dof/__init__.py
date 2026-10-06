"""Standalone PyChrono visualization of the ANN2SNN 6-DoF quadcopter example.

The dynamics are computed by this package in numpy; PyChrono is used only to
render the drone, scene and trajectory.  See :mod:`drone6dof.dynamics` for the
pluggable backend that leaves room to move dynamics into Chrono later.
"""

from .params import GRAVITY, DT, MAX_THRUST, DRONE_HALF, QuadParams
from .plant import Quad6DoF
from .scene import Scene, SceneSpec
from .reference import Reference, RefPoint, goal_reference
from .task import ObstacleGoalTask
from .control import DSGuidanceController, ClassicalPDController
from .connectome import ConnectomeController, ConnectomeTopology, ConnectomeANN, RateCodedConnectomeSNN
from .dynamics import DynamicsBackend, NumpyPlantBackend, ChronoDynamicsBackend
from .sim import Simulation
from .benchmark import run_benchmark, run_controller

__all__ = [
    "GRAVITY",
    "DT",
    "MAX_THRUST",
    "DRONE_HALF",
    "QuadParams",
    "Quad6DoF",
    "Scene",
    "SceneSpec",
    "Reference",
    "RefPoint",
    "goal_reference",
    "ObstacleGoalTask",
    "DSGuidanceController",
    "ClassicalPDController",
    "ConnectomeController",
    "ConnectomeTopology",
    "ConnectomeANN",
    "RateCodedConnectomeSNN",
    "DynamicsBackend",
    "NumpyPlantBackend",
    "ChronoDynamicsBackend",
    "Simulation",
    "run_benchmark",
    "run_controller",
]
