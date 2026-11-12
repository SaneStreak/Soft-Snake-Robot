"""Potential-field progress, heading alignment, effort, and slip reward shaper.

Implements the exact multi-objective reward from Section 3.2:
    R_t = w_progress * (d_g(t-1) - d_g(t))
        + w_head * cos(theta_err)
        - w_effort * sum_i(psi_i^2)
        - w_slip * v_lateral^2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml


@dataclass
class RewardConfig:
    """Reward shaping weights for SnakeGymEnv."""

    w_progress: float = 120.0
    w_head: float = 0.30
    w_effort: float = 0.002
    w_slip: float = 0.05
    goal_bonus: float = 150.0
    stall_penalty: float = 0.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RewardConfig":
        """Load reward weights from ppoc_training.yaml."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        section = data.get("reward", data)
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in section.items() if k in valid_fields}
        return cls(**filtered)


class RewardShaper:
    """Computes potential-field locomotion rewards and diagnostic sub-terms."""

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        prev_dist: float,
        curr_dist: float,
        theta_err: float,
        psi: np.ndarray,
        v_lateral: float,
        reached_goal: bool = False,
    ) -> tuple[float, Dict[str, float]]:
        """Evaluate step reward and breakdown dictionary."""
        cfg = self.config
        r_progress = cfg.w_progress * (prev_dist - curr_dist)
        # Centered cosine alignment (cos(theta_err) - 1) <= 0 preserves exact cos gradient
        # while preventing stationary heading-reward farming.
        r_head = cfg.w_head * (float(np.cos(theta_err)) - 1.0)
        r_effort = -cfg.w_effort * float(np.sum(np.asarray(psi, dtype=np.float64) ** 2))
        r_slip = -cfg.w_slip * float(v_lateral ** 2)
        r_goal = cfg.goal_bonus if reached_goal else 0.0

        total = float(r_progress + r_head + r_effort + r_slip + r_goal)
        components = {
            "r_progress": float(r_progress),
            "r_head": float(r_head),
            "r_effort": float(r_effort),
            "r_slip": float(r_slip),
            "r_goal": float(r_goal),
            "r_total": total,
        }
        return total, components

