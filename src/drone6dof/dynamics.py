"""Pluggable dynamics backends.

The visualization layer only needs a body pose and telemetry; it does not care
how the motion is produced.  :class:`NumpyPlantBackend` wraps the ported
:class:`~drone6dof.plant.Quad6DoF`.  :class:`ChronoDynamicsBackend` is a stub
for a future milestone in which Chrono integrates a rigid body driven by rotor
thrust forces.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .params import QuadParams
from .plant import Quad6DoF

__all__ = ["DynamicsBackend", "NumpyPlantBackend", "ChronoDynamicsBackend"]


@runtime_checkable
class DynamicsBackend(Protocol):
    """Minimal interface consumed by :class:`drone6dof.sim.Simulation`."""

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Reset hidden state and seed the translational state."""

    def step(
        self,
        command: np.ndarray,
        *,
        dt: float,
        limit: float,
        gain: float,
        damping: float = 0.0,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance one frame and return the new ``[p, v]`` state."""

    @property
    def state(self) -> np.ndarray:
        """Current ``[p, v]`` state (6,)."""

    @property
    def position(self) -> np.ndarray:
        """Current world position (3,)."""

    @property
    def rotation(self) -> np.ndarray:
        """Current body-to-world rotation matrix (3, 3)."""

    @property
    def telemetry(self) -> dict:
        """Per-frame diagnostics (see :class:`Quad6DoF`)."""


class NumpyPlantBackend:
    """Backend that integrates the ported 6-DoF plant in numpy."""

    def __init__(
        self,
        params: Optional[QuadParams] = None,
        *,
        heading_target=None,
    ) -> None:
        self.plant = Quad6DoF(params)
        if heading_target is not None:
            self.plant.heading_target = np.asarray(heading_target, dtype=np.float64)
        self._state = np.zeros(6, dtype=np.float64)

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        self.plant.reset()
        self._state = np.asarray(initial_state, dtype=np.float64).reshape(6).copy()
        return self.state

    def step(
        self,
        command: np.ndarray,
        *,
        dt: float,
        limit: float,
        gain: float,
        damping: float = 0.0,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._state = self.plant(
            self._state,
            command,
            dt=dt,
            limit=limit,
            gain=gain,
            damping=damping,
            disturbance=disturbance,
        )
        return self.state

    # -- introspection ------------------------------------------------------ #
    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def position(self) -> np.ndarray:
        return self._state[:3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.plant.R.copy()

    @property
    def omega(self) -> np.ndarray:
        return self.plant.omega.copy()

    @property
    def omega_m(self) -> np.ndarray:
        return self.plant.omega_m.copy()

    @property
    def rotor_target(self) -> np.ndarray:
        return self.plant.last_rotor_target

    @property
    def specific_force_body(self) -> np.ndarray:
        """Body-frame specific force (ideal accelerometer reading, m/s²)."""
        return self.plant.specific_force_body

    @property
    def attitude_rpy(self) -> np.ndarray:
        return self.plant.attitude_rpy

    @property
    def telemetry(self) -> dict:
        return self.plant.last_telemetry


class ChronoDynamicsBackend:
    """Placeholder for a Chrono-integrated rigid-body backend (future work).

    Intended design: a ``ChBody`` with mass/inertia from :class:`QuadParams`, a
    custom force/torque functor that maps the four rotor thrusts (and the
    fuselage/rotor aero) onto the body, and a controller issuing acceleration
    demands that the functor closes as ``F_des = m·(u + g·ẑ)``.  The plant's
    motor/battery state machine would move into the functor.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - explicit stub
        raise NotImplementedError(
            "ChronoDynamicsBackend is not implemented in this milestone; the "
            "demo runs the numpy plant with Chrono as the visualization layer."
        )
