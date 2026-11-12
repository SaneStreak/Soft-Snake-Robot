"""200 Hz Capacitive Self-Sensing PI Inner-Loop Controller.

Regulates segment curvature kappa toward reference kappa_ref(t) using
differential self-sensing capacitance feedback (C_L, C_R) and maps
required Maxwell stress differential linearly onto bilateral DEA voltages
[V_L, V_R] within the 450 V safety rail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.plant.dea_actuator import DEAActuatorPlant


@dataclass
class InnerLoopPIConfig:
    """Configuration for the 200 Hz capacitive PI curvature regulator."""

    kp_pa_per_rad_m: float = 3500.0
    ki_pa_per_rad_m_s: float = 18000.0
    feedforward_gain: float = 1.0
    integral_limit_rad_m_s: float = 1.5
    dt: float = 0.005


class CapacitiveInnerLoopPI:
    """Closed-loop 200 Hz PI controller transforming kappa_ref into 24 DEA strip voltages."""

    def __init__(
        self,
        dea_plant: DEAActuatorPlant,
        config: Optional[InnerLoopPIConfig] = None,
    ) -> None:
        self.dea = dea_plant
        self.config = config or InnerLoopPIConfig()
        self.num_modules = self.dea.config.num_modules
        self.integral_err = np.zeros(self.num_modules, dtype=np.float64)

        # Maximum Maxwell pressure achievable at v_rail (nominal 54 kPa at 450 V)
        h_m = self.dea.config.film_thickness_um * 1e-6
        self.p_max_pa = float(
            self.dea.config.vacuum_permittivity
            * self.dea.config.relative_permittivity
            * ((self.dea.v_rail / h_m) ** 2)
        )

    def reset(self) -> None:
        """Reset integrator state."""
        self.integral_err.fill(0.0)

    def step(
        self,
        kappa_ref: np.ndarray,
        capacitance_pf: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute safe bilateral DEA strip voltages (N, 2) from curvature target & self-sensing.

        Args:
            kappa_ref: Desired curvature per module (rad/m), shape (N,).
            capacitance_pf: Measured left/right strip capacitances (pF), shape (N, 2).
            dt: Step period in seconds (default 0.005 s).

        Returns:
            Safe clamped strip voltages [V_L, V_R] of shape (N, 2) in [0, 450] V.
        """
        dt_val = self.config.dt if dt is None else float(dt)
        kappa_meas = self.dea.estimate_curvature_from_capacitance(capacitance_pf)
        err = np.asarray(kappa_ref, dtype=np.float64) - kappa_meas

        # Integrator with anti-windup clamp
        self.integral_err = np.clip(
            self.integral_err + err * dt_val,
            -self.config.integral_limit_rad_m_s,
            self.config.integral_limit_rad_m_s,
        )

        # Feedforward equilibrium pressure + PI correction
        ff_gain = 1.0 / max(self.dea.config.stress_to_curvature_gain, 1e-12)
        dp_ff = self.config.feedforward_gain * ff_gain * np.asarray(kappa_ref, dtype=np.float64)
        dp_fb = (
            self.config.kp_pa_per_rad_m * err
            + self.config.ki_pa_per_rad_m_s * self.integral_err
        )
        dp_cmd = np.clip(dp_ff + dp_fb, -self.p_max_pa, self.p_max_pa)

        # Exact quadratic Maxwell inversion via balanced pre-bias allocation:
        # V_L^2 - V_R^2 = V_rail^2 * (dp_cmd / p_max) => p_L - p_R = dp_cmd
        frac = np.clip(dp_cmd / self.p_max_pa, -1.0, 1.0)
        v_left = self.dea.v_rail * np.sqrt(0.5 * (1.0 + frac))
        v_right = self.dea.v_rail * np.sqrt(0.5 * (1.0 - frac))
        v_cmd = np.stack([v_left, v_right], axis=-1)

        return self.dea.apply_safety_governor(v_cmd, dt=dt_val, enforce_slew=True)

