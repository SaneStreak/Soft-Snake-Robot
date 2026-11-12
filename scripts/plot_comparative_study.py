"""Comparative Benchmark & Visualization: Classical PI+ILOS vs. Hierarchical PPOC-CPG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controllers.classical_pi_ilos import ClassicalPIILOSController
from src.envs.snake_gym_env import SnakeGymEnv
from src.plant.friction_coulomb import TilePatch
from src.plant.kinematics_1d import PlanarSnakePlant
from src.rl.option_critic_ppoc import PPOCAgent


def rollout_classical(
    tile_width_m: float | None,
    goal_xy: np.ndarray,
    duration_s: float = 10.0,
    tile_x_start: float = 0.70,
) -> dict:
    """Evaluate classical PI+ILOS benchmark controller."""
    plant = PlanarSnakePlant.from_yaml(REPO_ROOT / "configs" / "plant_rev_c.yaml")
    patches = []
    if tile_width_m is not None and tile_width_m > 0:
        patches.append(
            TilePatch(
                x_min=tile_x_start,
                x_max=tile_x_start + tile_width_m,
                substrate=plant.friction.config.tile,
            )
        )
    plant.set_terrain(patches=patches)
    ctrl = ClassicalPIILOSController(plant.dea)
    plant.reset()
    ctrl.reset()

    dt = plant.geom.dt
    steps = int(round(duration_s / dt))
    xs, ys, vxs, v_max_hist = [], [], [], []
    reached = False

    for _ in range(steps):
        st = plant.get_state()
        v_cmd = ctrl.step(st, plant.get_capacitance_feedback(), goal_xy)
        plant.step(v_cmd)
        xs.append(float(plant.x_cm))
        ys.append(float(plant.y_cm))
        vxs.append(float(plant.vx))
        v_max_hist.append(float(np.max(plant.dea.v_state)))
        if np.hypot(goal_xy[0] - plant.x_cm, goal_xy[1] - plant.y_cm) <= 0.20:
            reached = True
            break

    t = np.arange(len(xs)) * dt
    return {
        "t": t,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "vx": np.asarray(vxs),
        "v_max": np.asarray(v_max_hist),
        "final_x": float(xs[-1]),
        "mean_vx_mm_s": float((xs[-1] / max(t[-1], dt)) * 1000.0),
        "reached": reached,
        "breakdowns": int(plant.dea.breakdown_count),
    }


def rollout_ppoc(
    agent: PPOCAgent,
    tile_width_m: float | None,
    goal_xy: np.ndarray,
    duration_s: float = 10.0,
    tile_x_start: float = 0.70,
) -> dict:
    """Evaluate trained Hierarchical PPOC-CPG agent."""
    env = SnakeGymEnv(curriculum_stage="C", fixed_tile_width=tile_width_m)
    patches = []
    if tile_width_m is not None and tile_width_m > 0:
        patches.append(
            TilePatch(
                x_min=tile_x_start,
                x_max=tile_x_start + tile_width_m,
                substrate=env.plant.friction.config.tile,
            )
        )
    obs, _ = env.reset(
        seed=42,
        options={"goal_xy": goal_xy, "patches": patches, "sand_eta": 3.0},
    )

    dt_rl = env.dt * env.action_repeat
    steps = int(round(duration_s / dt_rl))
    xs, ys, vxs, kfs, v_max_hist = [], [], [], [], []
    option = None
    reached = False

    for _ in range(steps):
        alpha, option, *_ = agent.select_action(
            obs, current_option=option, deterministic=True, update_norm=False
        )
        k_f = agent.get_k_f(option)
        obs, _, term, trunc, info = env.step(alpha, k_f=k_f)
        xs.append(float(info["x_cm"]))
        ys.append(float(info["y_cm"]))
        vxs.append(float(env.plant.vx))
        kfs.append(k_f)
        v_max_hist.append(float(np.max(env.plant.dea.v_state)))
        if info["reached_goal"]:
            reached = True
        if term or trunc:
            break

    t = np.arange(len(xs)) * dt_rl
    return {
        "t": t,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "vx": np.asarray(vxs),
        "kf": np.asarray(kfs),
        "v_max": np.asarray(v_max_hist),
        "final_x": float(xs[-1]),
        "mean_vx_mm_s": float((xs[-1] / max(t[-1], dt_rl)) * 1000.0),
        "reached": reached,
        "breakdowns": int(env.plant.dea.breakdown_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Comparative Study: Classical PI+ILOS vs PPOC-CPG")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "ppoc_best.pt",
        help="Path to trained PPOC checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "comparative_study.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        fallback = REPO_ROOT / "checkpoints" / "ppoc_final.pt"
        if fallback.exists():
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"No trained PPOC checkpoint found at {args.checkpoint}. Run scripts/train_ppoc_agent.py first."
            )

    agent = PPOCAgent.load(args.checkpoint)
    L = 0.600
    goal_xy = np.array([2.2, 0.0], dtype=np.float64)
    tile_x_start = 0.70

    configs = [
        ("Uniform Sand", None),
        ("L/4 Tile (150 mm)", L / 4.0),
        ("L/2 Tile (300 mm)", L / 2.0),
        ("L Tile (600 mm)", L),
    ]

    classical_res = {}
    ppoc_res = {}

    print("=" * 80)
    print(f"{'Scenario':22s} | {'Classical X (v)':>20s} | {'PPOC-CPG X (v)':>22s} | {'Breakdowns':>10s}")
    print("-" * 80)
    for label, width in configs:
        c_r = rollout_classical(width, goal_xy, duration_s=10.0, tile_x_start=tile_x_start)
        p_r = rollout_ppoc(agent, width, goal_xy, duration_s=10.0, tile_x_start=tile_x_start)
        classical_res[label] = c_r
        ppoc_res[label] = p_r
        print(
            f"{label:22s} | {c_r['final_x']:5.2f}m ({c_r['mean_vx_mm_s']:5.1f} mm/s) | "
            f"{p_r['final_x']:5.2f}m ({p_r['mean_vx_mm_s']:5.1f} mm/s) | "
            f"{c_r['breakdowns'] + p_r['breakdowns']:10d}"
        )
    print("=" * 80)

    # Generate 4-panel figure focused on L/2 (300 mm) tile traversal + summary bars
    c_l2 = classical_res["L/2 Tile (300 mm)"]
    p_l2 = ppoc_res["L/2 Tile (300 mm)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    fig.suptitle(
        "Hierarchical CPG-Regulated PPOC vs. Classical PI+ILOS on Anisotropic Friction & L/2 Polished Tile",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: 2D Trajectory over L/2 tile
    ax = axes[0, 0]
    ax.axvspan(tile_x_start, tile_x_start + L / 2.0, color="#d9e3f0", alpha=0.8, label="Polished Tile (eta=1.0, L/2)")
    ax.plot(c_l2["x"], c_l2["y"], "--", color="#d62728", lw=2.2, label="Classical PI+ILOS (Stalled)")
    ax.plot(p_l2["x"], p_l2["y"], "-", color="#1f77b4", lw=2.4, label="PPOC-CPG (Traversed)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=180, color="#2ca02c", zorder=5, label="Target Goal")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Planar CoM Trajectory (L/2 = 300 mm Tile Band)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)

    # Panel 2: Forward velocity and macro-option K_f switching
    ax = axes[0, 1]
    ax.plot(c_l2["t"], c_l2["vx"] * 1000.0, "--", color="#d62728", lw=2.0, label="Classical v_x (mm/s)")
    ax.plot(p_l2["t"], p_l2["vx"] * 1000.0, "-", color="#1f77b4", lw=2.0, label="PPOC v_x (mm/s)")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward Velocity (mm/s)")
    ax.set_title("Velocity Retention & Macro-Option K_f Modulation")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.step(p_l2["t"], p_l2["kf"], where="post", color="#ff7f0e", lw=1.6, alpha=0.85, label="Macro Option K_f")
    ax2.set_ylabel("CPG Frequency Scale K_f", color="#ff7f0e")
    ax2.set_ylim(0.4, 1.8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Panel 3: Bar comparison of final advance across tile widths
    ax = axes[1, 0]
    labels = [c[0] for c in configs]
    x_pos = np.arange(len(labels))
    w = 0.35
    c_vals = [classical_res[k]["final_x"] for k in labels]
    p_vals = [ppoc_res[k]["final_x"] for k in labels]
    ax.bar(x_pos - w / 2, c_vals, width=w, color="#d62728", alpha=0.85, label="Classical PI+ILOS")
    ax.bar(x_pos + w / 2, p_vals, width=w, color="#1f77b4", alpha=0.85, label="PPOC-CPG")
    ax.axhline(goal_xy[0], color="#2ca02c", ls=":", lw=1.5, label="Goal Distance (2.2 m)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Final Forward Advance (m)")
    ax.set_title("Disturbance Rejection Across Tile Band Widths")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5)

    # Panel 4: DEA Electrical Safety Verification
    ax = axes[1, 1]
    ax.plot(c_l2["t"], c_l2["v_max"], "--", color="#d62728", lw=1.8, label="Classical Max |V|")
    ax.plot(p_l2["t"], p_l2["v_max"], "-", color="#1f77b4", lw=1.8, label="PPOC Max |V|")
    ax.axhline(450.0, color="#ff7f0e", ls="-.", lw=1.6, label="450 V Safety Rail (45 V/um)")
    ax.axhline(580.0, color="#8c564b", ls=":", lw=1.6, label="580 V Pull-in Breakdown")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Peak Strip Voltage (V)")
    ax.set_title("DEA Pull-In Breakdown Governor Telemetry")
    ax.set_ylim(0, 620)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.close(fig)
    print(f"Comparative study plot saved to {args.output}")


if __name__ == "__main__":
    main()
