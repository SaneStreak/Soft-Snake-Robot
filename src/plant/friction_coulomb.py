"""Anisotropic Coulomb ground contact solver (Hu et al., 2009).

Implements non-smooth directional friction where transverse friction mu_n
exceeds tangential friction mu_t (anisotropy ratio eta = mu_n / mu_t).
Supports spatially varying substrate patches (e.g., sand eta=3.0 with
embedded polished tile bands eta=1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import yaml


@dataclass
class SubstrateProperties:
    """Directional Coulomb friction coefficients for a ground substrate."""

    mu_t: float = 0.12
    mu_n: float = 0.36
    stiction_factor: float = 1.15
    v_stiction_mps: float = 0.015

    @property
    def eta(self) -> float:
        """Anisotropy ratio eta = mu_n / mu_t."""
        return float(self.mu_n / max(self.mu_t, 1e-9))


@dataclass
class TilePatch:
    """Rectangular or band substrate patch with altered friction anisotropy."""

    x_min: float
    x_max: float
    y_min: float = -10.0
    y_max: float = 10.0
    substrate: SubstrateProperties = field(
        default_factory=lambda: SubstrateProperties(
            mu_t=0.30, mu_n=0.30, stiction_factor=1.30, v_stiction_mps=0.020
        )
    )


@dataclass
class FrictionConfig:
    """Configuration for ground friction mechanics."""

    gravity_mps2: float = 9.81
    eta_bifurcation: float = 1.7
    sand: SubstrateProperties = field(default_factory=lambda: SubstrateProperties(0.12, 0.36, 1.15, 0.015))
    tile: SubstrateProperties = field(default_factory=lambda: SubstrateProperties(0.30, 0.30, 1.30, 0.020))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FrictionConfig":
        """Load friction configuration from plant YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        section = data.get("friction", data)
        sand_dict = section.get("sand", {})
        tile_dict = section.get("tile", {})
        return cls(
            gravity_mps2=float(section.get("gravity_mps2", 9.81)),
            eta_bifurcation=float(section.get("eta_bifurcation", 1.7)),
            sand=SubstrateProperties(**sand_dict) if sand_dict else SubstrateProperties(0.12, 0.36, 1.15, 0.015),
            tile=SubstrateProperties(**tile_dict) if tile_dict else SubstrateProperties(0.30, 0.30, 1.30, 0.020),
        )


class AnisotropicCoulombSolver:
    """Computes distributed anisotropic Coulomb friction forces & torques along the snake body."""

    def __init__(
        self,
        config: Optional[FrictionConfig] = None,
        base_substrate: Optional[SubstrateProperties] = None,
        patches: Optional[list[TilePatch]] = None,
    ) -> None:
        self.config = config or FrictionConfig()
        self.base_substrate = base_substrate or self.config.sand
        self.patches: list[TilePatch] = list(patches) if patches else []

    def set_terrain(
        self,
        base_substrate: Optional[SubstrateProperties] = None,
        patches: Optional[list[TilePatch]] = None,
    ) -> None:
        """Update base substrate properties and embedded tile patches."""
        if base_substrate is not None:
            self.base_substrate = base_substrate
        self.patches = list(patches) if patches else []

    def query_substrate(
        self,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate per-segment (mu_t, mu_n, stiction_factor, v_stiction) at (N, 2) positions."""
        n = positions.shape[0]
        mu_t = np.full(n, self.base_substrate.mu_t, dtype=np.float64)
        mu_n = np.full(n, self.base_substrate.mu_n, dtype=np.float64)
        stick = np.full(n, self.base_substrate.stiction_factor, dtype=np.float64)
        v_stick = np.full(n, self.base_substrate.v_stiction_mps, dtype=np.float64)

        for patch in self.patches:
            mask = (
                (positions[:, 0] >= patch.x_min)
                & (positions[:, 0] <= patch.x_max)
                & (positions[:, 1] >= patch.y_min)
                & (positions[:, 1] <= patch.y_max)
            )
            if np.any(mask):
                mu_t[mask] = patch.substrate.mu_t
                mu_n[mask] = patch.substrate.mu_n
                stick[mask] = patch.substrate.stiction_factor
                v_stick[mask] = patch.substrate.v_stiction_mps

        return mu_t, mu_n, stick, v_stick

    def compute_forces(
        self,
        positions: np.ndarray,
        tangents: np.ndarray,
        velocities: np.ndarray,
        segment_masses: np.ndarray | float,
        reg_eps: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """Evaluate per-segment orthogonal anisotropic Coulomb friction forces (Hu et al., 2009).

        f_i = -m_i * g * s_i * [ mu_t * sign(u_t) * t_i + mu_n * sign(u_n) * n_i ]
        regularized smoothly via tanh(u / reg_eps) with static stiction factor s_i near rest.
        """
        mu_t, mu_n, stick, v_stick = self.query_substrate(positions)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)

        speed = np.linalg.norm(velocities, axis=-1)
        u_t = np.sum(velocities * tangents, axis=-1)
        u_n = np.sum(velocities * normals, axis=-1)

        # Smooth static-to-kinetic stiction multiplier near zero velocity
        stiction_mult = 1.0 + (stick - 1.0) * np.exp(-((speed / np.maximum(v_stick, 1e-5)) ** 2))

        normal_load = np.asarray(segment_masses, dtype=np.float64) * self.config.gravity_mps2
        f_t_mag = -normal_load * stiction_mult * mu_t * np.tanh(u_t / reg_eps)
        f_n_mag = -normal_load * stiction_mult * mu_n * np.tanh(u_n / reg_eps)

        forces = f_t_mag[:, None] * tangents + f_n_mag[:, None] * normals

        return {
            "forces": forces,
            "f_t_mag": f_t_mag,
            "f_n_mag": f_n_mag,
            "mu_t": mu_t,
            "mu_n": mu_n,
            "eta": mu_n / np.maximum(mu_t, 1e-9),
        }
