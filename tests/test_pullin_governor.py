"""Tests for DEA pull-in breakdown safety governor and Maxwell stress calibration."""

from pathlib import Path

import numpy as np

from src.plant.dea_actuator import DEAActuatorPlant, DEAConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "plant_rev_c.yaml"


def test_pullin_voltage_rail_and_breakdown_protection() -> None:
    """Assert 0 breakdown events and strict clamping <= 450 V (45 V/um) under spike commands."""
    cfg = DEAConfig.from_yaml(CONFIG_PATH)
    dea = DEAActuatorPlant(cfg)
    dea.reset()

    rng = np.random.default_rng(123)
    for _ in range(200):
        # Adversarial commands exceeding 58 V/um breakdown (580 V) and negative voltages
        v_adv = rng.uniform(-300.0, 1200.0, size=(cfg.num_modules, 2))
        v_safe = dea.apply_safety_governor(v_adv, dt=0.005, enforce_slew=True)
        e_field = v_safe / cfg.film_thickness_um

        assert np.all(v_safe >= 0.0)
        assert np.all(v_safe <= 450.0 + 1e-12)
        assert np.all(e_field <= 45.0 + 1e-12)
        assert np.all(e_field < 58.0)

    diag = dea.get_diagnostics()
    assert diag["breakdown_events"] == 0.0
    assert diag["max_voltage_v"] <= 450.0
    assert diag["max_e_field_v_per_um"] <= 45.0


def test_slew_rate_limiter() -> None:
    """Assert driver slew rate never exceeds 20 V/ms (100 V per 5 ms step)."""
    cfg = DEAConfig.from_yaml(CONFIG_PATH)
    dea = DEAActuatorPlant(cfg)
    dea.reset(initial_voltage=0.0)

    v_step = np.full((cfg.num_modules, 2), 450.0)
    v1 = dea.apply_safety_governor(v_step, dt=0.005, enforce_slew=True)
    assert np.allclose(v1, 100.0)

    v2 = dea.apply_safety_governor(v_step, dt=0.005, enforce_slew=True)
    assert np.allclose(v2, 200.0)


def test_nominal_maxwell_stress_54_kpa() -> None:
    """Verify nominal 450 V on 10 um film yields ~54 kPa Maxwell stress."""
    cfg = DEAConfig.from_yaml(CONFIG_PATH)
    dea = DEAActuatorPlant(cfg)
    v_nom = np.full((cfg.num_modules, 2), 450.0)
    p_kpa = dea.maxwell_stress(v_nom) / 1000.0
    assert np.allclose(p_kpa, 54.0, atol=0.1)

