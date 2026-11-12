"""Coupled non-linear Matsuoka Central Pattern Generator (CPG) neural ODEs.

Implements the 12-module antagonistic oscillator chain from:
Liu, Onal, & Fu (2023), "Reinforcement Learning of CPG-regulated Locomotion
Controller for a Soft Snake Robot", arXiv:2207.04899.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid mapping tonic activation to [0, 1]."""
    clipped = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class MatsuokaCPGConfig:
    """Configuration parameters for the coupled Matsuoka oscillator spine."""

    num_modules: int = 12
    dt: float = 0.005
    tau_r: float = 0.05
    tau_a: float = 0.25
    a: float = 2.5
    b: float = 2.5
    c: float = 0.75
    w_coupling: float = 0.6
    a_psi: float = 1.25
    k_f_default: float = 0.81
    k_f_min: float = 0.5
    k_f_max: float = 2.0
    alpha_clip: float = 2.0
    target_freq_hz: float = 1.6
    target_freq_tol_hz: float = 0.1

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MatsuokaCPGConfig":
        """Load Matsuoka CPG configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        section = data.get("matsuoka_cpg", data)
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in section.items() if k in valid_fields}
        return cls(**filtered)


class MatsuokaCPG:
    """12-segment coupled Matsuoka half-center oscillator ODE network.

    For each module i in {0, ..., N-1} and neuron q in {e (0), f (1)}:
        K_f * tau_r * dx_i^q/dt = -x_i^q - a * z_i^{q_bar} - b * y_i^q
                                  - sum_j W_{i, j} * z_j^q + u_i^q + c
        K_f * tau_a * dy_i^q/dt = z_i^q - y_i^q
        z_i^q = max(0, x_i^q)
        psi_i = a_psi * (z_i^e - z_i^f)
    """

    def __init__(self, config: Optional[MatsuokaCPGConfig] = None) -> None:
        self.config = config or MatsuokaCPGConfig()
        n = self.config.num_modules

        # Unidirectional coupling matrix W: row i receives from column j (i = j + 1)
        self.W = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            self.W[i + 1, i] = self.config.w_coupling

        self.x = np.zeros((n, 2), dtype=np.float64)
        self.y = np.zeros((n, 2), dtype=np.float64)
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        perturbation_scale: float = 0.05,
    ) -> np.ndarray:
        """Reset membrane potentials and adaptation states with symmetry-breaking seed.

        A small anti-symmetric perturbation across extensor/flexor pairs ensures
        reliable self-starting oscillation off the symmetric saddle equilibrium.
        """
        n = self.config.num_modules
        self.x.fill(0.0)
        self.y.fill(0.0)

        if seed is not None:
            rng = np.random.default_rng(seed)
            delta = rng.uniform(-perturbation_scale, perturbation_scale, size=n)
            # Ensure non-degenerate asymmetry on module 0
            if np.abs(delta[0]) < 1e-3:
                delta[0] = perturbation_scale
        else:
            # Deterministic travelling-phase seed from head to tail
            phases = np.linspace(0.0, np.pi, n)
            delta = perturbation_scale * np.cos(phases)

        self.x[:, 0] = delta
        self.x[:, 1] = -delta
        return self.get_output()

    @property
    def z(self) -> np.ndarray:
        """Current non-negative neural firing rates z_i^q = max(0, x_i^q)."""
        return np.maximum(0.0, self.x)

    def get_output_raw(self) -> np.ndarray:
        """Return unclipped oscillatory output a_psi * (z_e - z_f) of shape (N,)."""
        z = self.z
        return self.config.a_psi * (z[:, 0] - z[:, 1])

    def get_output(self) -> np.ndarray:
        """Return normalized CPG output psi_i in [-1, 1] of shape (N,)."""
        return np.clip(self.get_output_raw(), -1.0, 1.0)

    def _derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        k_f: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ODE time derivatives (dx/dt, dy/dt) for RK4 integration."""
        cfg = self.config
        z = np.maximum(0.0, x)
        z_opp = z[:, ::-1]
        # Contralateral inter-module inhibition yields smooth head-to-tail phase lag
        # and consistent sign-preserving steering bias across the 12-module chain.
        coupling = self.W @ z_opp

        inv_kr = 1.0 / (k_f * cfg.tau_r)
        inv_ka = 1.0 / (k_f * cfg.tau_a)

        dx = (-x - cfg.a * z_opp - cfg.b * y - coupling + u + cfg.c) * inv_kr
        dy = (z - y) * inv_ka
        return dx, dy

    def _rk4_step(self, u: np.ndarray, k_f: float, dt: float) -> None:
        """Advance coupled neural ODE state by one RK4 step."""
        x0, y0 = self.x, self.y

        k1_x, k1_y = self._derivatives(x0, y0, u, k_f)
        k2_x, k2_y = self._derivatives(x0 + 0.5 * dt * k1_x, y0 + 0.5 * dt * k1_y, u, k_f)
        k3_x, k3_y = self._derivatives(x0 + 0.5 * dt * k2_x, y0 + 0.5 * dt * k2_y, u, k_f)
        k4_x, k4_y = self._derivatives(x0 + dt * k3_x, y0 + dt * k3_y, u, k_f)

        self.x = x0 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        self.y = y0 + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)

    def map_alpha_to_tonic(self, alpha: Optional[np.ndarray | float]) -> np.ndarray:
        """Map steering activation alpha_i in [-2, 2] to complementary drives (u_e, u_f)."""
        n = self.config.num_modules
        if alpha is None:
            u_e = np.full(n, 0.5, dtype=np.float64)
        else:
            alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n,))
            alpha_clipped = np.clip(alpha_arr, -self.config.alpha_clip, self.config.alpha_clip)
            u_e = np.asarray(sigmoid(alpha_clipped), dtype=np.float64)
        u_f = 1.0 - u_e
        return np.stack([u_e, u_f], axis=-1)

    def step(
        self,
        alpha: Optional[np.ndarray | float] = None,
        k_f: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Integrate CPG by one timestep given steering bias alpha and frequency gain K_f.

        Args:
            alpha: Steering activation(s) in [-2.0, 2.0]. Scalar or shape (N,).
            k_f: Frequency scaling factor in [0.5, 2.0]. Defaults to config.k_f_default.
            dt: Integration step size (s). Defaults to config.dt (0.005 s).

        Returns:
            Normalized CPG output psi of shape (N,) in [-1, 1].
        """
        cfg = self.config
        k_f_val = cfg.k_f_default if k_f is None else float(np.clip(k_f, cfg.k_f_min, cfg.k_f_max))
        dt_val = cfg.dt if dt is None else float(dt)
        u = self.map_alpha_to_tonic(alpha)
        self._rk4_step(u, k_f_val, dt_val)
        return self.get_output()

    def step_with_tonic(
        self,
        u_e: np.ndarray | float,
        k_f: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Integrate CPG using explicit complementary extensor tonic drive u_e in [0, 1]."""
        cfg = self.config
        n = cfg.num_modules
        u_e_arr = np.clip(np.broadcast_to(np.asarray(u_e, dtype=np.float64), (n,)), 0.0, 1.0)
        u = np.stack([u_e_arr, 1.0 - u_e_arr], axis=-1)
        k_f_val = cfg.k_f_default if k_f is None else float(np.clip(k_f, cfg.k_f_min, cfg.k_f_max))
        dt_val = cfg.dt if dt is None else float(dt)
        self._rk4_step(u, k_f_val, dt_val)
        return self.get_output()

    def rollout(
        self,
        num_steps: int,
        alpha: Optional[np.ndarray | float] = None,
        k_f: Optional[float] = None,
        u_e: Optional[np.ndarray | float] = None,
    ) -> Dict[str, np.ndarray]:
        """Run a multi-step open-loop rollout and record CPG trajectories."""
        n = self.config.num_modules
        dt = self.config.dt
        psi_hist = np.zeros((num_steps, n), dtype=np.float64)
        psi_raw_hist = np.zeros((num_steps, n), dtype=np.float64)
        z_diff_hist = np.zeros((num_steps, n), dtype=np.float64)
        t_hist = np.arange(num_steps, dtype=np.float64) * dt

        for step_idx in range(num_steps):
            if u_e is not None:
                psi = self.step_with_tonic(u_e=u_e, k_f=k_f)
            else:
                psi = self.step(alpha=alpha, k_f=k_f)
            psi_hist[step_idx] = psi
            psi_raw_hist[step_idx] = self.get_output_raw()
            z = self.z
            z_diff_hist[step_idx] = z[:, 0] - z[:, 1]

        return {
            "t": t_hist,
            "psi": psi_hist,
            "psi_raw": psi_raw_hist,
            "z_diff": z_diff_hist,
        }
