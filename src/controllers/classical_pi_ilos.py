"""Classical benchmark controller: Hirose Serpenoid + Integral Line-of-Sight (ILOS) + 200 Hz PI.

Used as the Phase 2 non-learning baseline demonstrating nominal sand locomotion
and permanent stall on unmodeled isotropic/polished tile patches (eta <= 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.controllers.inner_loop_pi import CapacitiveInnerLoopPI, InnerLoopPIConfig
from src.plant.dea_actuator import DEAActuatorPlant


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class ClassicalPIILOSConfig:
    """Configuration for classical serpenoid + ILOS + capacitive PI controller."""

    num_modules: int = 12
    amplitude_rad_per_m: float = 7.5
    frequency_hz: float = 1.6
    num_wavelengths: float = 1.05
    lookahead_m: float = 0.60
    ilos_gain: float = 0.15
    steer_kp: float = 3.0
    steer_bias_limit: float = 3.5
    dt: float = 0.005


class ClassicalPIILOSController:
    """Hirose serpenoid wave generator + ILOS steering + 200 Hz capacitive PI inner loop."""

    def __init__(
        self,
        dea_plant: DEAActuatorPlant,
        config: Optional[ClassicalPIILOSConfig] = None,
        pi_config: Optional[InnerLoopPIConfig] = None,
    ) -> None:
        self.config = config or ClassicalPIILOSConfig(num_modules=dea_plant.config.num_modules)
        self.inner_pi = CapacitiveInnerLoopPI(dea_plant, pi_config)
        self.y_int = 0.0
        self.time_s = 0.0

    def reset(self) -> None:
        """Reset ILOS integral cross-track error and PI inner loop."""
        self.inner_pi.reset()
        self.y_int = 0.0
        self.time_s = 0.0

    def compute_curvature_reference(
        self,
        x_cm: float,
        y_cm: float,
        theta_cm: float,
        goal_xy: np.ndarray,
        start_xy: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute desired 12-module curvature profile kappa_ref(t)."""
        cfg = self.config
        dt_val = cfg.dt if dt is None else float(dt)
        origin = np.zeros(2, dtype=np.float64) if start_xy is None else np.asarray(start_xy, dtype=np.float64)
        goal = np.asarray(goal_xy, dtype=np.float64)

        path_vec = goal - origin
        path_len = float(np.linalg.norm(path_vec))
        if path_len > 1e-6:
            path_angle = float(np.arctan2(path_vec[1], path_vec[0]))
            dx = x_cm - origin[0]
            dy = y_cm - origin[1]
            # Cross-track error perpendicular to desired straight path
            e_ct = -np.sin(path_angle) * dx + np.cos(path_angle) * dy
            denom = cfg.lookahead_m ** 2 + (e_ct + cfg.ilos_gain * self.y_int) ** 2
            self.y_int = float(np.clip(self.y_int + (cfg.lookahead_m * e_ct / denom) * dt_val, -2.0, 2.0))
            theta_los = path_angle - float(np.arctan((e_ct + cfg.ilos_gain * self.y_int) / cfg.lookahead_m))
        else:
            theta_los = float(np.arctan2(goal[1] - y_cm, goal[0] - x_cm))

        heading_err = wrap_angle(theta_los - theta_cm)
        kappa_steer = float(np.clip(cfg.steer_kp * heading_err, -cfg.steer_bias_limit, cfg.steer_bias_limit))

        # Hirose serpenoid travelling wave from head (i=0) to tail (i=N-1)
        idx = np.arange(cfg.num_modules, dtype=np.float64)
        phase_lag = 2.0 * np.pi * cfg.num_wavelengths * (idx / cfg.num_modules)
        wave = cfg.amplitude_rad_per_m * np.sin(2.0 * np.pi * cfg.frequency_hz * self.time_s - phase_lag)
        self.time_s += dt_val
        return wave + kappa_steer

    def step(
        self,
        plant_state: dict,
        capacitance_pf: np.ndarray,
        goal_xy: np.ndarray,
        start_xy: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute 24 safe DEA voltages from plant telemetry and goal waypoint."""
        dt_val = self.config.dt if dt is None else float(dt)
        kappa_ref = self.compute_curvature_reference(
            x_cm=float(plant_state["x_cm"]),
            y_cm=float(plant_state["y_cm"]),
            theta_cm=float(plant_state["theta_cm"]),
            goal_xy=goal_xy,
            start_xy=start_xy,
            dt=dt_val,
        )
        return self.inner_pi.step(kappa_ref=kappa_ref, capacitance_pf=capacitance_pf, dt=dt_val)
