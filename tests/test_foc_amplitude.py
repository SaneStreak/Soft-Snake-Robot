"""Tests for Free-Response Oscillation Constraint (FOC) anti-decay property."""

from pathlib import Path

import numpy as np

from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cpg_matsuoka.yaml"


def test_foc_prevents_amplitude_quenching() -> None:
    """Assert no amplitude quenching when c=0.75 across biases where c=0.0 collapses."""
    cfg_foc = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    assert cfg_foc.c == 0.75

    cfg_no_foc = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    cfg_no_foc.c = 0.0

    warmup_steps = int(round(3.0 / cfg_foc.dt))
    meas_steps = int(round(3.0 / cfg_foc.dt))

    # First verify baseline c=0.0 quenches under moderate steering bias u_e=0.65
    cpg_no_foc = MatsuokaCPG(cfg_no_foc)
    cpg_no_foc.reset()
    cpg_no_foc.rollout(warmup_steps, u_e=0.65)
    traj_no_foc = cpg_no_foc.rollout(meas_steps, u_e=0.65)
    ptp_no_foc = float(np.ptp(traj_no_foc["psi"][:, 0]))
    assert ptp_no_foc < 1e-3, f"Expected c=0.0 quenching at u_e=0.65, got ptp={ptp_no_foc}"

    # Verify c=0.75 maintains full oscillation amplitude across [0.32, 0.68]
    cpg_ref = MatsuokaCPG(cfg_foc)
    cpg_ref.reset()
    cpg_ref.rollout(warmup_steps, u_e=0.50)
    ref_ptp = float(np.min(np.ptp(cpg_ref.rollout(meas_steps, u_e=0.50)["psi"], axis=0)))

    for u_e in np.linspace(0.32, 0.68, 9):
        cpg = MatsuokaCPG(cfg_foc)
        cpg.reset()
        cpg.rollout(warmup_steps, u_e=float(u_e))
        traj = cpg.rollout(meas_steps, u_e=float(u_e))
        min_ptp = float(np.min(np.ptp(traj["psi"], axis=0)))
        assert min_ptp >= 0.85 * ref_ptp, (
            f"FOC amplitude decay at u_e={u_e:.2f}: ptp={min_ptp:.3f} < 0.85*{ref_ptp:.3f}"
        )


def test_foc_amplitude_invariance_across_frequencies() -> None:
    """Verify minimum entrainment amplitude A_0(c, omega) persists across K_f in [0.5, 2.0]."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    warmup_steps = int(round(3.0 / cfg.dt))
    meas_steps = int(round(4.0 / cfg.dt))

    amplitudes = []
    for k_f in (0.6, 0.8, 1.0, 1.3, 1.6):
        cpg = MatsuokaCPG(cfg)
        cpg.reset()
        cpg.rollout(warmup_steps, k_f=k_f)
        traj = cpg.rollout(meas_steps, k_f=k_f)
        amp = 0.5 * float(np.min(np.ptp(traj["psi"], axis=0)))
        assert amp > 0.70, f"Entrainment amplitude dropped at K_f={k_f}: {amp:.3f}"
        amplitudes.append(amp)

    assert np.ptp(amplitudes) < 0.15, f"Amplitude varied unexpectedly across K_f: {amplitudes}"

