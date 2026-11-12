"""Validated 1D-parameterized planar continuum soft snake simulator (Rev C).

Models a 12-module hyperelastic silicone-DEA continuum body (L=600 mm, W=4.5 mm,
H=12 mm, M=35.6 g) using piecewise-constant-curvature (PCC) backbone reconstruction,
second-order viscoelastic DEA bending dynamics, and Hu et al. (2009) anisotropic
Coulomb ground friction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import yaml

from src.plant.dea_actuator import DEAActuatorPlant, DEAConfig
from src.plant.friction_coulomb import (
    AnisotropicCoulombSolver,
    FrictionConfig,
    SubstrateProperties,
    TilePatch,
)


@dataclass
class PlantGeometryConfig:
    """Rev C physical geometry and viscoelastic bending parameters."""

    num_modules: int = 12
    body_length_m: float = 0.600
    body_width_m: float = 0.0045
    body_height_m: float = 0.012
    total_mass_kg: float = 0.0356
    max_curvature_rad_per_m: float = 14.0
    natural_freq_hz: float = 6.5
    damping_ratio: float = 0.85
    flexural_rigidity_nm2: float = 1.8e-4
    dt: float = 0.005
    substeps: int = 2

    @property
    def segment_length_m(self) -> float:
        """Arc length per continuum module (m)."""
        return self.body_length_m / self.num_modules

    @property
    def segment_mass_kg(self) -> float:
        """Mass per continuum module (kg)."""
        return self.total_mass_kg / self.num_modules

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PlantGeometryConfig":
        """Load geometry, mechanics, and simulation parameters from plant YAML."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        merged = {}
        for section in ("geometry", "mechanics", "simulation"):
            if section in data and isinstance(data[section], dict):
                merged.update(data[section])
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        return cls(**filtered)


def reconstruct_body_backbone(
    kappa: np.ndarray,
    segment_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct segment CoM-centered local coordinates and tangent angles.

    Module 0 is the head (+x direction at zero curvature) and module N-1 is the tail.
    The backbone is centered at its center of mass and oriented so the mean tangent
    angle is zero, decoupling internal symmetric undulation from global CoM rotation.

    Returns:
        r_body: Local segment midpoint positions of shape (N, 2).
        phi_rel: Local segment tangent angles relative to mean body heading, shape (N,).
    """
    n = len(kappa)
    # Integrate backward from head (module 0) toward tail (module N-1).
    # Standard Frenet-Serret sign: +kappa means d(theta)/ds = +kappa > 0 (CCW left turn),
    # so marching backward along -ds decrements local orientation by -kappa * ds.
    dphi = -kappa * segment_length
    cum_phi = np.concatenate(([0.0], np.cumsum(dphi[:-1])))
    phi_mid = cum_phi + 0.5 * dphi

    # Piecewise-constant-curvature chord factor sinc(dphi / 2)
    half_dphi = 0.5 * dphi
    chord_scale = np.where(
        np.abs(half_dphi) < 1e-6,
        1.0 - (half_dphi ** 2) / 6.0,
        np.sin(half_dphi) / np.where(np.abs(half_dphi) < 1e-6, 1.0, half_dphi),
    )
    ds = segment_length * chord_scale

    # Place head tip at origin, step backward along -tangent to tail
    step_vecs = np.stack([-ds * np.cos(phi_mid), -ds * np.sin(phi_mid)], axis=-1)
    # Segment midpoints along backbone
    joints = np.vstack(([0.0, 0.0], np.cumsum(step_vecs, axis=0)))
    r_raw = 0.5 * (joints[:-1] + joints[1:])

    # Center at CoM and zero out mean body angle
    com = np.mean(r_raw, axis=0)
    r_centered = r_raw - com
    mean_phi = float(np.mean(phi_mid))

    cos_m, sin_m = np.cos(-mean_phi), np.sin(-mean_phi)
    rot_align = np.array([[cos_m, -sin_m], [sin_m, cos_m]], dtype=np.float64)
    r_body = r_centered @ rot_align.T
    phi_rel = phi_mid - mean_phi
    return r_body, phi_rel


class PlanarSnakePlant:
    """12-module planar continuum soft snake physics simulator."""

    def __init__(
        self,
        geom_config: Optional[PlantGeometryConfig] = None,
        dea_config: Optional[DEAConfig] = None,
        friction_config: Optional[FrictionConfig] = None,
    ) -> None:
        self.geom = geom_config or PlantGeometryConfig()
        self.dea = DEAActuatorPlant(dea_config)
        self.friction = AnisotropicCoulombSolver(friction_config)

        n = self.geom.num_modules
        self.kappa = np.zeros(n, dtype=np.float64)
        self.kappa_dot = np.zeros(n, dtype=np.float64)

        self.x_cm = 0.0
        self.y_cm = 0.0
        self.theta_cm = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0

        self._prev_r_body, self._prev_phi_rel = reconstruct_body_backbone(
            self.kappa, self.geom.segment_length_m
        )
        self.time_s = 0.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PlanarSnakePlant":
        """Construct complete plant simulator from Rev C plant YAML file."""
        return cls(
            geom_config=PlantGeometryConfig.from_yaml(path),
            dea_config=DEAConfig.from_yaml(path),
            friction_config=FrictionConfig.from_yaml(path),
        )

    def reset(
        self,
        x_cm: float = 0.0,
        y_cm: float = 0.0,
        theta_cm: float = 0.0,
        kappa: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray | float]:
        """Reset snake pose, velocities, curvatures, and DEA voltages."""
        n = self.geom.num_modules
        self.dea.reset(initial_voltage=0.0)
        self.kappa = np.zeros(n, dtype=np.float64) if kappa is None else np.asarray(kappa, dtype=np.float64).copy()
        self.kappa_dot.fill(0.0)

        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)
        self.theta_cm = float(theta_cm)
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.time_s = 0.0

        self._prev_r_body, self._prev_phi_rel = reconstruct_body_backbone(
            self.kappa, self.geom.segment_length_m
        )
        return self.get_state()

    def set_terrain(
        self,
        base_substrate: Optional[SubstrateProperties] = None,
        patches: Optional[list[TilePatch]] = None,
    ) -> None:
        """Configure ground friction anisotropy and optional tile patches."""
        self.friction.set_terrain(base_substrate=base_substrate, patches=patches)

    def get_world_segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current world segment positions (N, 2) and tangent vectors (N, 2)."""
        r_body, phi_rel = reconstruct_body_backbone(self.kappa, self.geom.segment_length_m)
        c, s = np.cos(self.theta_cm), np.sin(self.theta_cm)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pos_world = np.array([self.x_cm, self.y_cm], dtype=np.float64) + r_body @ rot.T
        theta_seg = self.theta_cm + phi_rel
        tangents = np.stack([np.cos(theta_seg), np.sin(theta_seg)], axis=-1)
        return pos_world, tangents

    def get_capacitance_feedback(self) -> np.ndarray:
        """Read 24-channel self-sensing capacitances (N, 2) in pF."""
        return self.dea.measure_capacitance(self.kappa)

    def get_state(self) -> Dict[str, np.ndarray | float]:
        """Return full plant telemetry dictionary."""
        pos_world, tangents = self.get_world_segments()
        return {
            "x_cm": self.x_cm,
            "y_cm": self.y_cm,
            "theta_cm": self.theta_cm,
            "vx": self.vx,
            "vy": self.vy,
            "omega": self.omega,
            "kappa": self.kappa.copy(),
            "kappa_dot": self.kappa_dot.copy(),
            "segment_positions": pos_world,
            "segment_tangents": tangents,
            "head_x": float(pos_world[0, 0]),
            "head_y": float(pos_world[0, 1]),
            "tail_x": float(pos_world[-1, 0]),
            "tail_y": float(pos_world[-1, 1]),
            "time_s": self.time_s,
        }

    def step(
        self,
        voltages_cmd: np.ndarray,
        dt: Optional[float] = None,
        enforce_governor: bool = True,
    ) -> Dict[str, np.ndarray | float]:
        """Advance plant dynamics by one control step under commanded DEA voltages (N, 2)."""
        dt_total = self.geom.dt if dt is None else float(dt)
        substeps = max(1, int(self.geom.substeps))
        dt_sub = dt_total / substeps

        if enforce_governor:
            v_safe = self.dea.apply_safety_governor(voltages_cmd, dt=dt_total, enforce_slew=True)
        else:
            v_safe = np.asarray(voltages_cmd, dtype=np.float64).reshape(self.geom.num_modules, 2)

        kappa_eq = np.clip(
            self.dea.target_curvature_from_voltages(v_safe),
            -self.geom.max_curvature_rad_per_m,
            self.geom.max_curvature_rad_per_m,
        )

        omega_n = 2.0 * np.pi * self.geom.natural_freq_hz
        zeta = self.geom.damping_ratio
        seg_mass = self.geom.segment_mass_kg
        total_mass = self.geom.total_mass_kg

        for _ in range(substeps):
            # 1. Second-order viscoelastic DEA bending dynamics
            kappa_ddot = (
                (omega_n ** 2) * (kappa_eq - self.kappa)
                - 2.0 * zeta * omega_n * self.kappa_dot
            )
            self.kappa_dot = self.kappa_dot + kappa_ddot * dt_sub
            self.kappa = np.clip(
                self.kappa + self.kappa_dot * dt_sub,
                -self.geom.max_curvature_rad_per_m,
                self.geom.max_curvature_rad_per_m,
            )

            # 2. Backbone kinematics & internal shape velocity
            r_body, phi_rel = reconstruct_body_backbone(self.kappa, self.geom.segment_length_m)
            r_dot_body = (r_body - self._prev_r_body) / dt_sub
            self._prev_r_body = r_body
            self._prev_phi_rel = phi_rel

            c, s = np.cos(self.theta_cm), np.sin(self.theta_cm)
            rot = np.array([[c, -s], [s, c]], dtype=np.float64)

            r_world_rel = r_body @ rot.T
            pos_world = np.array([self.x_cm, self.y_cm], dtype=np.float64) + r_world_rel
            theta_seg = self.theta_cm + phi_rel
            tangents = np.stack([np.cos(theta_seg), np.sin(theta_seg)], axis=-1)

            # Segment world velocities u_i = v_cm + omega x r_i + R * r_dot_body_i
            rot_vel = np.stack([-self.omega * r_world_rel[:, 1], self.omega * r_world_rel[:, 0]], axis=-1)
            shape_vel = r_dot_body @ rot.T
            vel_world = np.array([self.vx, self.vy], dtype=np.float64) + rot_vel + shape_vel

            # 3. Anisotropic Coulomb ground friction
            fric = self.friction.compute_forces(
                positions=pos_world,
                tangents=tangents,
                velocities=vel_world,
                segment_masses=seg_mass,
            )
            forces = fric["forces"]
            f_net = np.sum(forces, axis=0)
            tau_net = float(np.sum(r_world_rel[:, 0] * forces[:, 1] - r_world_rel[:, 1] * forces[:, 0]))

            # 4. Rigid-body CoM & yaw integration
            iz = max(float(np.sum(seg_mass * np.sum(r_world_rel ** 2, axis=-1))), 1e-6)
            self.vx += (f_net[0] / total_mass) * dt_sub
            self.vy += (f_net[1] / total_mass) * dt_sub
            self.omega += (tau_net / iz) * dt_sub

            self.x_cm += self.vx * dt_sub
            self.y_cm += self.vy * dt_sub
            self.theta_cm = float(np.arctan2(np.sin(self.theta_cm + self.omega * dt_sub), np.cos(self.theta_cm + self.omega * dt_sub)))

        self.time_s += dt_total
        return self.get_state()
