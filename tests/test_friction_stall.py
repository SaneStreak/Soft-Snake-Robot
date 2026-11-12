"""Tests for Hu et al. (2009) anisotropic friction bifurcation and tile stall."""

from pathlib import Path

import numpy as np
import pytest

from src.controllers.classical_pi_ilos import ClassicalPIILOSController
from src.plant.friction_coulomb import SubstrateProperties, TilePatch
from src.plant.kinematics_1d import PlanarSnakePlant

PLANT_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "plant_rev_c.yaml"


@pytest.mark.parametrize("eta", [0.8, 1.0])
def test_uniform_isotropic_bifurcation_stall(eta: float) -> None:
    """Confirm classical open-loop/PI+ILOS gait stalls or slides backward when eta <= 1.0."""
    plant = PlanarSnakePlant.from_yaml(PLANT_CFG_PATH)
    mu_t = 0.25
    plant.set_terrain(SubstrateProperties(mu_t=mu_t, mu_n=eta * mu_t, stiction_factor=1.20))
    ctrl = ClassicalPIILOSController(plant.dea)
    plant.reset()
    ctrl.reset()

    goal = np.array([3.0, 0.0])
    for _ in range(1000):  # 5.0 s at 200 Hz
        state = plant.get_state()
        v_cmd = ctrl.step(state, plant.get_capacitance_feedback(), goal)
        plant.step(v_cmd)

    final_state = plant.get_state()
    assert final_state["x_cm"] <= 0.02, (
        f"Expected stall/backward slip for eta={eta}, got x_cm={final_state['x_cm']:.3f} m"
    )


def test_sand_propulsion_and_tile_band_stall() -> None:
    """Confirm 0.78 BL/s nominal sand speed and permanent stall on L/2=300 mm tile band."""
    # 1) Uniform sand (eta = 3.0)
    plant_sand = PlanarSnakePlant.from_yaml(PLANT_CFG_PATH)
    ctrl_sand = ClassicalPIILOSController(plant_sand.dea)
    plant_sand.reset()
    ctrl_sand.reset()

    goal = np.array([5.0, 0.0])
    steps_6s = 1200
    for _ in range(steps_6s):
        st = plant_sand.get_state()
        plant_sand.step(ctrl_sand.step(st, plant_sand.get_capacitance_feedback(), goal))

    vx_sand = float(plant_sand.get_state()["x_cm"]) / 6.0
    bl_per_s = vx_sand / plant_sand.geom.body_length_m
    assert bl_per_s >= 0.70, f"Expected >= 0.70 BL/s on sand, got {bl_per_s:.2f} BL/s"

    # 2) Embedded L/2 = 300 mm polished tile band (eta = 1.0)
    plant_tile = PlanarSnakePlant.from_yaml(PLANT_CFG_PATH)
    tile_patch = TilePatch(
        x_min=0.80,
        x_max=1.10,
        substrate=plant_tile.friction.config.tile,
    )
    plant_tile.set_terrain(patches=[tile_patch])
    ctrl_tile = ClassicalPIILOSController(plant_tile.dea)
    plant_tile.reset()
    ctrl_tile.reset()

    x_history = []
    for k in range(2400):  # 12.0 s
        st = plant_tile.get_state()
        plant_tile.step(ctrl_tile.step(st, plant_tile.get_capacitance_feedback(), goal))
        if k >= 1600:
            x_history.append(float(st["x_cm"]))

    late_vx_mm_s = ((x_history[-1] - x_history[0]) / 4.0) * 1000.0
    assert late_vx_mm_s <= 5.0, (
        f"Expected permanent classical stall on L/2 tile band, got late_vx={late_vx_mm_s:.1f} mm/s"
    )
    assert x_history[-1] < 1.10, (
        f"Classical controller unexpectedly crossed L/2 tile band: x={x_history[-1]:.3f} m"
    )

