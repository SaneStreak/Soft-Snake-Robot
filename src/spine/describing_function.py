"""Describing function analysis verifying CPG steering bias linearity.

Verifies the analytical relation from Liu, Onal, & Fu (2023):
    bias(u_i) = u_i^e - u_i^f = 2 * u_i^e - 1
    bias(z_i) = [K_n / ((b - a) * K_n + 1)] * bias(u_i)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig


@dataclass
class LinearityReport:
    """Results of describing-function steering bias linearity verification."""

    bias_u: np.ndarray
    bias_z_mean: np.ndarray
    bias_psi_mean: np.ndarray
    slope_z: float
    intercept_z: float
    r_squared_z: float
    slope_psi: float
    intercept_psi: float
    r_squared_psi: float
    effective_kn: float
    theoretical_gain: float


def theoretical_describing_gain(a: float, b: float, k_n: float) -> float:
    """Compute theoretical DC gain K_n / ((b - a) * K_n + 1)."""
    return float(k_n / ((b - a) * k_n + 1.0))


def _cycle_accurate_mean(signal: np.ndarray) -> float:
    """Compute DC mean over an exact integer number of oscillation cycles."""
    centered = signal - 0.5 * (np.max(signal) + np.min(signal))
    crossings = np.where((centered[:-1] < 0.0) & (centered[1:] >= 0.0))[0]
    if len(crossings) >= 2:
        return float(np.mean(signal[crossings[0] : crossings[-1]]))
    return float(np.mean(signal))


def compute_dc_bias_sweep(
    config: Optional[MatsuokaCPGConfig] = None,
    u_e_values: Optional[np.ndarray] = None,
    warmup_s: float = 3.0,
    measure_s: float = 4.0,
    k_f: Optional[float] = None,
    module_idx: int = 0,
) -> LinearityReport:
    """Sweep symmetric tonic drive u_e and compute steady-state DC biases."""
    cfg = config or MatsuokaCPGConfig()
    k_f_val = cfg.k_f_default if k_f is None else float(k_f)
    if u_e_values is None:
        # Non-quenching complementary steering band for FOC c=0.75
        u_e_values = np.linspace(0.32, 0.68, 15)

    warmup_steps = int(round(warmup_s / cfg.dt))
    measure_steps = int(round(measure_s / cfg.dt))

    bias_u = 2.0 * u_e_values - 1.0
    bias_z_mean = np.zeros_like(u_e_values)
    bias_psi_mean = np.zeros_like(u_e_values)

    for idx, u_e in enumerate(u_e_values):
        cpg = MatsuokaCPG(cfg)
        cpg.reset()
        cpg.rollout(warmup_steps, u_e=float(u_e), k_f=k_f_val)
        traj = cpg.rollout(measure_steps, u_e=float(u_e), k_f=k_f_val)
        bias_z_mean[idx] = _cycle_accurate_mean(traj["z_diff"][:, module_idx])
        bias_psi_mean[idx] = _cycle_accurate_mean(traj["psi"][:, module_idx])

    from scipy.stats import linregress

    reg_z = linregress(bias_u, bias_z_mean)
    reg_psi = linregress(bias_u, bias_psi_mean)

    # Infer effective describing gain K_n from measured DC slope:
    # slope_z = K_n / ((b - a) * K_n + 1) => K_n = slope_z / (1 - (b - a) * slope_z)
    denom = 1.0 - (cfg.b - cfg.a) * reg_z.slope
    effective_kn = float(reg_z.slope / denom) if abs(denom) > 1e-9 else float(reg_z.slope)
    theo_gain = theoretical_describing_gain(cfg.a, cfg.b, effective_kn)

    return LinearityReport(
        bias_u=bias_u,
        bias_z_mean=bias_z_mean,
        bias_psi_mean=bias_psi_mean,
        slope_z=float(reg_z.slope),
        intercept_z=float(reg_z.intercept),
        r_squared_z=float(reg_z.rvalue ** 2),
        slope_psi=float(reg_psi.slope),
        intercept_psi=float(reg_psi.intercept),
        r_squared_psi=float(reg_psi.rvalue ** 2),
        effective_kn=effective_kn,
        theoretical_gain=theo_gain,
    )


def verify_steering_linearity(
    config: Optional[MatsuokaCPGConfig] = None,
    min_r_squared: float = 0.99,
) -> LinearityReport:
    """Verify that steady-state CPG bias scales linearly with R^2 >= min_r_squared."""
    report = compute_dc_bias_sweep(config=config)
    if report.r_squared_psi < min_r_squared:
        raise AssertionError(
            f"Steering linearity R^2={report.r_squared_psi:.4f} < {min_r_squared:.4f}"
        )
    return report
