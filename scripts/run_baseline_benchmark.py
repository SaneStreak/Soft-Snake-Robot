"""Phase 2 Verification Benchmark: Classical PI+ILOS across uniform sand and tile bands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controllers.classical_pi_ilos import ClassicalPIILOSController
from src.plant.friction_coulomb import SubstrateProperties, TilePatch
from src.plant.kinematics_1d import PlanarSnakePlant

PLANT_CFG_PATH = REPO_ROOT / "configs" / "plant_rev_c.yaml"


def evaluate_scenario(
    name: str,
    base_substrate: SubstrateProperties | None = None,
    patches: list[TilePatch] | None = None,
    duration_s: float = 12.0,
) -> dict:
    """Run classical PI+ILOS controller on a specified substrate terrain."""
    plant = PlanarSnakePlant.from_yaml(PLANT_CFG_PATH)
    if base_substrate is not None or patches is not None:
        plant.set_terrain(base_substrate=base_substrate, patches=patches)
    ctrl = ClassicalPIILOSController(plant.dea)
    plant.reset()
    ctrl.reset()

    dt = plant.geom.dt
    total_steps = int(round(duration_s / dt))
    goal = np.array([5.0, 0.0])

    x_trace = np.zeros(total_steps, dtype=np.float64)
    for k in range(total_steps):
        st = plant.get_state()
        v_cmd = ctrl.step(st, plant.get_capacitance_feedback(), goal)
        plant.step(v_cmd)
        x_trace[k] = float(st["x_cm"])

    # Steady-state velocity over first 6 s (uniform) and final 4 s (stall check)
    idx_6s = min(total_steps - 1, int(round(6.0 / dt)))
    idx_late = max(0, total_steps - int(round(4.0 / dt)))

    v_6s_mps = x_trace[idx_6s] / 6.0
    v_late_mps = (x_trace[-1] - x_trace[idx_late]) / 4.0
    bl = plant.geom.body_length_m
    dea_diag = plant.dea.get_diagnostics()

    return {
        "name": name,
        "final_x_m": float(x_trace[-1]),
        "v_6s_mm_s": float(v_6s_mps * 1000.0),
        "v_6s_bl_s": float(v_6s_mps / bl),
        "v_late_mm_s": float(v_late_mps * 1000.0),
        "max_v_v": dea_diag["max_voltage_v"],
        "breakdowns": int(dea_diag["breakdown_events"]),
    }


def main() -> None:
    plant_ref = PlanarSnakePlant.from_yaml(PLANT_CFG_PATH)
    L = plant_ref.geom.body_length_m
    tile_sub = plant_ref.friction.config.tile

    scenarios = [
        ("Uniform Sand (eta=3.0)", None, None),
        ("Uniform Tile (eta=1.0)", tile_sub, None),
        ("Sand + Tile Band L/4 (150 mm)", None, [TilePatch(0.80, 0.80 + L / 4.0, substrate=tile_sub)]),
        ("Sand + Tile Band L/2 (300 mm)", None, [TilePatch(0.80, 0.80 + L / 2.0, substrate=tile_sub)]),
        ("Sand + Tile Band L   (600 mm)", None, [TilePatch(0.80, 0.80 + L, substrate=tile_sub)]),
    ]

    print("=" * 84)
    print(f"{'Scenario':34s} | {'Final X':>8s} | {'Mean v':>10s} | {'BL/s':>6s} | {'Late v':>10s} | {'Breakdowns':>10s}")
    print("-" * 84)
    for title, sub, patches in scenarios:
        res = evaluate_scenario(title, sub, patches)
        print(
            f"{res['name']:34s} | {res['final_x_m']:6.3f} m | "
            f"{res['v_6s_mm_s']:6.1f} mm/s | {res['v_6s_bl_s']:5.2f}  | "
            f"{res['v_late_mm_s']:6.1f} mm/s | {res['breakdowns']:10d}"
        )
    print("=" * 84)


if __name__ == "__main__":
    main()

