"""Dielectric Elastomer Actuator (DEA) plant model and pull-in safety governor.

Implements sub-kilovolt solid-state bilateral DEA physics grounded in Peng et al. (2026):
    p_i = epsilon_0 * epsilon_r * (V_i / h)^2
with a hard electric field rail at 45 V/um (V_max = 450 V on h = 10 um film),
20%+ below the 58 V/um pull-in dielectric breakdown threshold, plus dV/dt slew limiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import yaml


@dataclass
class DEAConfig:
    """Configuration for the 12-module bilateral (24-strip) DEA plant."""

    num_modules: int = 12
    num_strips_per_module: int = 2
    film_thickness_um: float = 10.0
    relative_permittivity: float = 3.012
    vacuum_permittivity: float = 8.854187817e-12
    v_max_v: float = 450.0
    e_max_v_per_um: float = 45.0
    e_breakdown_v_per_um: float = 58.0
    slew_rate_max_v_per_ms: float = 20.0
    stress_to_curvature_gain: float = 2.6e-4
    nominal_capacitance_pf: float = 620.0
    capacitive_strain_sensitivity: float = 0.04

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DEAConfig":
        """Load DEA configuration from plant YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        section = data.get("dea_actuator", data)
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in section.items() if k in valid_fields}
        return cls(**filtered)


class DEAActuatorPlant:
    """24-strip bilateral DEA electrostatic actuator with safety governor & self-sensing."""

    def __init__(self, config: Optional[DEAConfig] = None) -> None:
        self.config = config or DEAConfig()
        self.v_rail = min(
            self.config.v_max_v,
            self.config.e_max_v_per_um * self.config.film_thickness_um,
        )
        self.v_state = np.zeros(
            (self.config.num_modules, self.config.num_strips_per_module),
            dtype=np.float64,
        )
        self.breakdown_count = 0
        self.last_electric_field = np.zeros_like(self.v_state)

    def reset(self, initial_voltage: float = 0.0) -> None:
        """Reset actuator strip voltages and safety diagnostics."""
        clipped = float(np.clip(initial_voltage, 0.0, self.v_rail))
        self.v_state.fill(clipped)
        self.breakdown_count = 0
        self.last_electric_field = self.v_state / self.config.film_thickness_um

    def apply_safety_governor(
        self,
        v_cmd: np.ndarray,
        dt: float = 0.005,
        enforce_slew: bool = True,
    ) -> np.ndarray:
        """Clamp commanded voltages to pull-in safety rail and slew-rate limit.

        Args:
            v_cmd: Commanded voltages of shape (N, 2) or (2N,) in Volts.
            dt: Control timestep in seconds (default 0.005 s for 200 Hz).
            enforce_slew: Whether to enforce dV/dt <= 20 V/ms slew rate limit.

        Returns:
            Safe clamped voltages of shape (N, 2) in [0, 450] V.
        """
        v_arr = np.asarray(v_cmd, dtype=np.float64).reshape(
            self.config.num_modules, self.config.num_strips_per_module
        )
        # Step 1: Hard voltage & electric-field rail clamp (<= 450 V, <= 45 V/um)
        v_clamped = np.clip(v_arr, 0.0, self.v_rail)

        # Step 2: Slew-rate limiter (dV/dt <= 20 V/ms -> 100 V per 5 ms step)
        if enforce_slew:
            dt_ms = dt * 1000.0
            max_dv = self.config.slew_rate_max_v_per_ms * dt_ms
            dv = np.clip(v_clamped - self.v_state, -max_dv, max_dv)
            v_safe = np.clip(self.v_state + dv, 0.0, self.v_rail)
        else:
            v_safe = v_clamped

        self.v_state = v_safe
        self.last_electric_field = v_safe / self.config.film_thickness_um
        if np.any(self.last_electric_field >= self.config.e_breakdown_v_per_um):
            self.breakdown_count += 1
        return v_safe.copy()

    def maxwell_stress(self, voltages: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute electrostatic Maxwell stress p_i = eps_0 * eps_r * (V_i / h)^2 in Pascals."""
        v = self.v_state if voltages is None else np.asarray(voltages, dtype=np.float64)
        h_m = self.config.film_thickness_um * 1e-6
        e_field_v_per_m = v / h_m
        return (
            self.config.vacuum_permittivity
            * self.config.relative_permittivity
            * (e_field_v_per_m ** 2)
        )

    def differential_pressure(self, voltages: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute left-minus-right bilateral Maxwell pressure differential (Pa)."""
        p = self.maxwell_stress(voltages)
        return p[:, 0] - p[:, 1]

    def target_curvature_from_voltages(self, voltages: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert bilateral Maxwell stress differential to equilibrium bending curvature (rad/m)."""
        dp = self.differential_pressure(voltages)
        return self.config.stress_to_curvature_gain * dp

    def measure_capacitance(self, kappa: np.ndarray) -> np.ndarray:
        """Simulate capacitive self-sensing readout C_L, C_R (pF) from curvature kappa."""
        c0 = self.config.nominal_capacitance_pf
        k_c = self.config.capacitive_strain_sensitivity
        c_left = c0 * np.clip(1.0 + k_c * kappa, 0.2, 2.5)
        c_right = c0 * np.clip(1.0 - k_c * kappa, 0.2, 2.5)
        return np.stack([c_left, c_right], axis=-1)

    def estimate_curvature_from_capacitance(self, capacitance_pf: np.ndarray) -> np.ndarray:
        """Recover segment curvature estimate from bilateral differential capacitance."""
        c0 = self.config.nominal_capacitance_pf
        k_c = self.config.capacitive_strain_sensitivity
        cap = np.asarray(capacitance_pf, dtype=np.float64)
        return (cap[:, 0] - cap[:, 1]) / (2.0 * k_c * c0)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return safety and actuation telemetry summary."""
        p = self.maxwell_stress()
        return {
            "max_voltage_v": float(np.max(self.v_state)),
            "max_e_field_v_per_um": float(np.max(self.last_electric_field)),
            "max_maxwell_stress_kpa": float(np.max(p) / 1000.0),
            "breakdown_events": float(self.breakdown_count),
        }

