"""Tests for describing-function steering bias linearity (R^2 >= 0.99)."""

from pathlib import Path

import numpy as np

from src.spine.describing_function import (
    compute_dc_bias_sweep,
    theoretical_describing_gain,
    verify_steering_linearity,
)
from src.spine.matsuoka_ode import MatsuokaCPGConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cpg_matsuoka.yaml"


def test_steering_linearity_across_chain() -> None:
    """Assert Pearson R^2 >= 0.99 for DC output bias vs tonic input bias across segments."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    for mod_idx in (0, 5, 11):
        report = compute_dc_bias_sweep(cfg, module_idx=mod_idx)
        assert report.r_squared_psi >= 0.99, (
            f"Module {mod_idx} psi linearity R^2={report.r_squared_psi:.4f} < 0.99"
        )
        assert report.r_squared_z >= 0.99, (
            f"Module {mod_idx} z linearity R^2={report.r_squared_z:.4f} < 0.99"
        )
        assert report.slope_psi > 0.4, f"Module {mod_idx} slope too small: {report.slope_psi}"


def test_describing_function_gain_formula() -> None:
    """Verify theoretical describing gain K_n / ((b - a) * K_n + 1) matches measured slope."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    report = verify_steering_linearity(cfg, min_r_squared=0.99)
    expected_gain = theoretical_describing_gain(cfg.a, cfg.b, report.effective_kn)
    assert np.isclose(report.slope_z, expected_gain, atol=1e-6)
    # Zero bias symmetry check
    mid_idx = len(report.bias_u) // 2
    assert abs(report.bias_u[mid_idx]) < 1e-12
    assert abs(report.bias_psi_mean[mid_idx]) < 0.03

