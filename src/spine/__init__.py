"""Spinal layer: coupled Matsuoka CPG neural ODEs and describing function analysis."""

from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig
from src.spine.describing_function import (
    compute_dc_bias_sweep,
    theoretical_describing_gain,
    verify_steering_linearity,
)

__all__ = [
    "MatsuokaCPG",
    "MatsuokaCPGConfig",
    "compute_dc_bias_sweep",
    "theoretical_describing_gain",
    "verify_steering_linearity",
]

