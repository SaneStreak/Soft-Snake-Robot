"""Tests for Matsuoka CPG natural 1.6 Hz limit cycle and head-to-tail phase lock."""

from pathlib import Path

import numpy as np
import pytest

from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cpg_matsuoka.yaml"


def _dominant_frequency_hz(signal: np.ndarray, dt: float) -> float:
    centered = signal - np.mean(signal)
    crossings = np.where((centered[:-1] < 0.0) & (centered[1:] >= 0.0))[0]
    if len(crossings) >= 2:
        # Linear interpolation of upward zero-crossings for sub-step accuracy
        t_cross = []
        for idx in crossings:
            frac = -centered[idx] / max(centered[idx + 1] - centered[idx], 1e-12)
            t_cross.append((idx + frac) * dt)
        periods = np.diff(t_cross)
        return float(1.0 / np.mean(periods))
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    return float(freqs[np.argmax(spectrum)])


def test_cpg_self_starting_limit_cycle_1_6_hz() -> None:
    """Confirm self-starting stable limit cycle at 1.6 +/- 0.1 Hz across all 12 modules."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    cpg = MatsuokaCPG(cfg)
    cpg.reset()

    # Warm up 3.0 s, measure 5.0 s (0.2 Hz FFT resolution; 10 s -> 0.1 Hz resolution)
    cpg.rollout(int(round(3.0 / cfg.dt)))
    traj = cpg.rollout(int(round(10.0 / cfg.dt)))
    psi = traj["psi"]

    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) <= 1.0 + 1e-12

    freqs = [_dominant_frequency_hz(psi[:, i], cfg.dt) for i in range(cfg.num_modules)]
    for i, f in enumerate(freqs):
        assert abs(f - 1.6) <= 0.1, f"Module {i} frequency {f:.3f} Hz outside 1.6 +/- 0.1 Hz"

    ptp = np.ptp(psi, axis=0)
    assert np.all(ptp > 1.0), f"Oscillation amplitude too low: min ptp = {np.min(ptp):.3f}"


def test_cpg_head_to_tail_phase_lock() -> None:
    """Confirm unidirectional coupling produces consistent positive head-to-tail phase lag."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    cpg = MatsuokaCPG(cfg)
    cpg.reset()
    cpg.rollout(int(round(3.0 / cfg.dt)))
    traj = cpg.rollout(int(round(4.0 / cfg.dt)))
    psi = traj["psi"]

    lags_s = []
    for i in range(cfg.num_modules - 1):
        s_up = psi[:, i] - np.mean(psi[:, i])
        s_down = psi[:, i + 1] - np.mean(psi[:, i + 1])
        xcorr = np.correlate(s_down, s_up, mode="full")
        lag_steps = int(np.argmax(xcorr) - (len(s_up) - 1))
        lags_s.append(lag_steps * cfg.dt)

    lags_arr = np.asarray(lags_s)
    assert np.all(lags_arr > 0.0), f"Expected positive head-to-tail lag, got {lags_arr}"
    assert np.all(lags_arr < 0.15), f"Lag too large (non-smooth wave): {lags_arr}"
    assert np.std(lags_arr) < 0.01, f"Phase lock inconsistent across modules: {lags_arr}"


@pytest.mark.parametrize("k_f", [0.6, 0.8, 1.0, 1.3, 1.6])
def test_cpg_frequency_scaling_with_kf(k_f: float) -> None:
    """Verify inverse frequency scaling f ~ f_0 * (0.8 / K_f) across macro-options Omega."""
    cfg = MatsuokaCPGConfig.from_yaml(CONFIG_PATH)
    cpg = MatsuokaCPG(cfg)
    cpg.reset()
    cpg.rollout(int(round(3.0 / cfg.dt)), k_f=k_f)
    traj = cpg.rollout(int(round(10.0 / cfg.dt)), k_f=k_f)

    f_meas = _dominant_frequency_hz(traj["psi"][:, 0], cfg.dt)
    f_expected = 1.30 / k_f  # Natural K_f=1.0 frequency is 1.30 Hz
    assert abs(f_meas - f_expected) <= 0.12, (
        f"For K_f={k_f}, expected ~{f_expected:.2f} Hz, got {f_meas:.2f} Hz"
    )
