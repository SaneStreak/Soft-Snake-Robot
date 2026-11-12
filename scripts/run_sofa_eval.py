"""Phase 5 Sim-to-Real Verification & 3D Visualization: Zero-Shot 1D -> 3D SOFA FEM Transfer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: F401
import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controllers.inner_loop_pi import CapacitiveInnerLoopPI
from src.envs.snake_gym_env import SnakeGymEnv, map_action_to_cpg_alpha
from src.envs.sofa_connector import SofaConnector, SofaIPCConfig, SofaLocalFEMBridge
from src.plant.dea_actuator import DEAActuatorPlant, DEAConfig
from src.rl.option_critic_ppoc import PPOCAgent
from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def eval_1d_speed(agent: PPOCAgent, goal_xy: np.ndarray, duration_s: float = 6.0) -> dict:
    """Measure trained PPOC speed and trajectory in the 1D planar environment."""
    env = SnakeGymEnv(curriculum_stage="A")
    obs, _ = env.reset(seed=77, options={"goal_xy": goal_xy, "sand_eta": 3.0})
    dt_rl = env.dt * env.action_repeat
    steps = int(round(duration_s / dt_rl))
    option = None

    t_hist, x_hist, y_hist, v_hist = [], [], [], []
    for k in range(steps):
        alpha, option, *_ = agent.select_action(
            obs, current_option=option, deterministic=True, update_norm=False
        )
        obs, _, term, trunc, info = env.step(alpha, k_f=agent.get_k_f(option))
        t_hist.append((k + 1) * dt_rl)
        x_hist.append(float(info["x_cm"]))
        y_hist.append(float(info["y_cm"]))
        v_hist.append(float(np.hypot(env.plant.vx, env.plant.vy)) * 1000.0)
        if term or trunc:
            break

    elapsed = env.step_count * dt_rl
    dist_traveled = float(np.hypot(info["x_cm"], info["y_cm"]))
    return {
        "dist_m": dist_traveled,
        "elapsed_s": elapsed,
        "speed_mm_s": (dist_traveled / max(elapsed, 1e-6)) * 1000.0,
        "t": np.asarray(t_hist),
        "x": np.asarray(x_hist),
        "y": np.asarray(y_hist),
        "speed_trace": np.asarray(v_hist),
    }


def eval_3d_sofa_speed(
    agent: PPOCAgent,
    goal_xy: np.ndarray,
    host: str,
    port: int,
    duration_s: float = 6.0,
) -> dict:
    """Measure zero-shot PPOC transfer speed & 3D hexahedral node trace over TCP IPC."""
    plant_cfg_path = REPO_ROOT / "configs" / "plant_rev_c.yaml"
    cpg_cfg = MatsuokaCPGConfig.from_yaml(REPO_ROOT / "configs" / "cpg_matsuoka.yaml")
    cpg = MatsuokaCPG(cpg_cfg)
    dea = DEAActuatorPlant(DEAConfig.from_yaml(plant_cfg_path))
    inner_pi = CapacitiveInnerLoopPI(dea)

    cpg.reset(seed=77)
    dea.reset()
    inner_pi.reset()
    cpg.rollout(50, k_f=cpg_cfg.k_f_default)

    conn = SofaConnector(SofaIPCConfig(host=host, port=port))
    conn.connect()
    try:
        tel = conn.reset(sand_eta=3.0, tile_patches=[])
        dt = cpg_cfg.dt
        action_repeat = 10
        dt_rl = dt * action_repeat
        steps = int(round(duration_s / dt_rl))
        option = None
        elapsed_steps = 0

        t_hist, x_hist, y_hist, v_hist = [], [], [], []
        nodes_hist, vmax_hist, k_hist = [], [], []

        for _ in range(steps):
            dx = float(goal_xy[0] - tel["x_cm"])
            dy = float(goal_xy[1] - tel["y_cm"])
            d_goal = float(np.hypot(dx, dy))
            theta_err = _wrap_angle(float(np.arctan2(dy, dx)) - float(tel["theta_cm"]))
            c, s = np.cos(float(tel["theta_cm"])), np.sin(float(tel["theta_cm"]))
            v_fwd = c * float(tel["vx"]) + s * float(tel["vy"])
            v_lat = -s * float(tel["vx"]) + c * float(tel["vy"])

            obs = np.concatenate(
                [
                    np.array([theta_err, d_goal, v_fwd, v_lat], dtype=np.float64),
                    np.asarray(tel["kappa"], dtype=np.float64),
                    np.asarray(tel["kappa_dot"], dtype=np.float64),
                ]
            ).astype(np.float32)

            alpha, option, *_ = agent.select_action(
                obs, current_option=option, deterministic=True, update_norm=False
            )
            k_f = agent.get_k_f(option)
            alpha_cpg = map_action_to_cpg_alpha(
                action=alpha,
                num_modules=cpg_cfg.num_modules,
                alpha_clip=cpg_cfg.alpha_clip,
                alpha_cpg_scale=0.38,
            )

            for _ in range(action_repeat):
                psi = cpg.step(alpha=alpha_cpg, k_f=k_f, dt=dt)
                kappa_ref = 12.5 * psi
                voltages = inner_pi.step(kappa_ref=kappa_ref, capacitance_pf=tel["capacitance_pf"], dt=dt)
                tel = conn.step(voltages, dt=dt)
                elapsed_steps += 1
                if np.hypot(goal_xy[0] - tel["x_cm"], goal_xy[1] - tel["y_cm"]) <= 0.20:
                    break

            t_hist.append(elapsed_steps * dt)
            x_hist.append(float(tel["x_cm"]))
            y_hist.append(float(tel["y_cm"]))
            v_hist.append(float(np.hypot(tel["vx"], tel["vy"])) * 1000.0)
            nodes_hist.append(np.asarray(tel["hex_nodes"], dtype=np.float64))
            vmax_hist.append(float(np.max(dea.v_state)))
            k_hist.append(np.asarray(tel["kappa"], dtype=np.float64))

            if np.hypot(goal_xy[0] - tel["x_cm"], goal_xy[1] - tel["y_cm"]) <= 0.20:
                break

        elapsed = elapsed_steps * dt
        dist_traveled = float(np.hypot(tel["x_cm"], tel["y_cm"]))
        return {
            "dist_m": dist_traveled,
            "elapsed_s": elapsed,
            "speed_mm_s": (dist_traveled / max(elapsed, 1e-6)) * 1000.0,
            "breakdowns": int(dea.breakdown_count),
            "hex_nodes_shape": np.asarray(tel["hex_nodes"]).shape,
            "t": np.asarray(t_hist),
            "x": np.asarray(x_hist),
            "y": np.asarray(y_hist),
            "speed_trace": np.asarray(v_hist),
            "nodes": np.asarray(nodes_hist),
            "vmax": np.asarray(vmax_hist),
            "kappa": np.asarray(k_hist),
        }
    finally:
        conn.close()


def _build_hex_boxes(centers: np.ndarray, width: float = 0.028, height: float = 0.036) -> list:
    """Extrude 12 centerline nodes into 3D hexahedral blocks for clear visualization."""
    n = len(centers)
    tangents = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        tangents[:-1] = centers[1:, :2] - centers[:-1, :2]
        tangents[-1] = tangents[-2]
    norms = np.maximum(np.linalg.norm(tangents, axis=-1, keepdims=True), 1e-9)
    t_hat = tangents / norms
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=-1)

    faces_all = []
    seg_len = 0.044
    for i in range(n):
        c = centers[i]
        dx = 0.5 * seg_len * t_hat[i]
        dy = 0.5 * width * n_hat[i]
        z0, z1 = 0.0, height
        p = np.zeros((8, 3), dtype=np.float64)
        p[0] = [c[0] - dx[0] - dy[0], c[1] - dx[1] - dy[1], z0]
        p[1] = [c[0] + dx[0] - dy[0], c[1] + dx[1] - dy[1], z0]
        p[2] = [c[0] + dx[0] + dy[0], c[1] + dx[1] + dy[1], z0]
        p[3] = [c[0] - dx[0] + dy[0], c[1] - dx[1] + dy[1], z0]
        p[4:8] = p[0:4]
        p[4:8, 2] = z1
        faces = [
            [p[0], p[1], p[2], p[3]],
            [p[4], p[5], p[6], p[7]],
            [p[0], p[1], p[5], p[4]],
            [p[2], p[3], p[7], p[6]],
            [p[1], p[2], p[6], p[5]],
            [p[0], p[3], p[7], p[4]],
        ]
        faces_all.append(faces)
    return faces_all


def save_sofa_visual_plot(
    res_1d: dict,
    res_3d: dict,
    goal_xy: np.ndarray,
    retention: float,
    output_path: Path,
) -> None:
    """Render 4-panel 3D FEM transfer verification figure."""
    fig = plt.figure(figsize=(13.5, 9.0), dpi=140)
    fig.suptitle(
        f"3D FEM IPC Zero-Shot Transfer Evaluation (FOC c=0.75 | Retention: {retention:.1f}%)",
        fontsize=13.5,
        fontweight="bold",
    )

    # Panel 1: 3D Hexahedral Backbone Snapshots
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    nodes = res_3d["nodes"]
    t_arr = res_3d["t"]
    snap_indices = np.linspace(0, len(nodes) - 1, 4, dtype=int)
    cmap = plt.get_cmap("viridis")

    for s_idx, k in enumerate(snap_indices):
        color = cmap(s_idx / max(len(snap_indices) - 1, 1))
        boxes = _build_hex_boxes(nodes[k])
        for b_faces in boxes:
            poly = Poly3DCollection(
                b_faces,
                facecolors=color,
                edgecolors="#222222",
                linewidths=0.3,
                alpha=0.75,
            )
            ax3d.add_collection3d(poly)
        ax3d.text(
            nodes[k, 0, 0],
            nodes[k, 0, 1],
            0.06,
            f"t={t_arr[k]:.1f}s",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    ax3d.plot(res_3d["x"], res_3d["y"], np.zeros_like(res_3d["x"]), "--", color="#1f77b4", lw=1.5, label="3D CoM Path")
    ax3d.scatter([goal_xy[0]], [goal_xy[1]], [0.0], marker="*", s=160, color="#2ca02c", label="Goal")
    ax3d.set_xlim(-0.4, max(3.0, float(np.max(res_3d["x"])) + 0.3))
    ax3d.set_ylim(-0.9, 0.9)
    ax3d.set_zlim(0.0, 0.16)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D FEM Hexahedral Body Snapshots")
    ax3d.view_init(elev=28, azim=-62)

    # Panel 2: Top-down 1D vs 3D trajectory overlay
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(res_1d["x"], res_1d["y"], "--", color="#ff7f0e", lw=2.2, label="1D Planar Baseline")
    ax2.plot(res_3d["x"], res_3d["y"], "-", color="#1f77b4", lw=2.4, label="3D FEM IPC Transfer")
    ax2.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=180, color="#2ca02c", zorder=5, label="Goal (2.5 m)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_title("1D Planar vs. 3D FEM CoM Trajectory")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    # Panel 3: Instantaneous velocity comparison
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(res_1d["t"], res_1d["speed_trace"], "--", color="#ff7f0e", lw=1.8, label=f"1D Speed (mean {res_1d['speed_mm_s']:.1f} mm/s)")
    ax3.plot(res_3d["t"], res_3d["speed_trace"], "-", color="#1f77b4", lw=2.0, label=f"3D FEM Speed (mean {res_3d['speed_mm_s']:.1f} mm/s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("CoM Speed (mm/s)")
    ax3.set_title(f"Velocity Retention Across 1D -> 3D Transfer ({retention:.1f}%)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=8.5)

    # Panel 4: DEA Electrical Safety Telemetry
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(res_3d["t"], res_3d["vmax"], "-", color="#1f77b4", lw=2.0, label="3D FEM Peak Strip Voltage")
    ax4.axhline(450.0, color="#ff7f0e", ls="-.", lw=1.6, label="450 V Safety Rail (45 V/um)")
    ax4.axhline(580.0, color="#d62728", ls=":", lw=1.6, label="580 V Pull-in Breakdown")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Voltage (V)")
    ax4.set_title(f"DEA Safety Governor During 3D IPC (Breakdowns: {res_3d['breakdowns']})")
    ax4.set_ylim(0, 620)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"3D FEM transfer plot saved to {output_path}")


def save_sofa_animation_gif(
    res_3d: dict,
    goal_xy: np.ndarray,
    output_gif: Path,
) -> None:
    """Render animated top-down & 3D slithering GIF of the 12-module soft snake."""
    nodes = res_3d["nodes"]
    t_arr = res_3d["t"]
    kappas = res_3d["kappa"]
    frame_indices = np.arange(0, len(nodes), 2)

    fig = plt.figure(figsize=(10.5, 4.8), dpi=110)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    x_max = max(3.0, float(np.max(res_3d["x"])) + 0.4)
    y_lim = max(0.8, float(np.max(np.abs(res_3d["y"]))) + 0.3)

    def update(frame_pos: int) -> None:
        k = int(frame_indices[frame_pos])
        ax3d.cla()
        ax2d.cla()

        # 3D mesh blocks
        boxes = _build_hex_boxes(nodes[k])
        norm_k = np.clip(0.5 + kappas[k] / 24.0, 0.0, 1.0)
        cmap = plt.get_cmap("coolwarm")
        for seg_i, b_faces in enumerate(boxes):
            poly = Poly3DCollection(
                b_faces,
                facecolors=cmap(norm_k[seg_i]),
                edgecolors="#222222",
                linewidths=0.35,
                alpha=0.88,
            )
            ax3d.add_collection3d(poly)
        ax3d.plot(res_3d["x"][: k + 1], res_3d["y"][: k + 1], np.zeros(k + 1), "--", color="#1f77b4", lw=1.4)
        ax3d.scatter([goal_xy[0]], [goal_xy[1]], [0.0], marker="*", s=140, color="#2ca02c")
        ax3d.set_xlim(-0.4, x_max)
        ax3d.set_ylim(-y_lim, y_lim)
        ax3d.set_zlim(0.0, 0.14)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_title(f"3D FEM Body (t = {t_arr[k]:.2f} s)")
        ax3d.view_init(elev=30, azim=-60)

        # 2D top-down backbone + head marker
        ax2d.plot(res_3d["x"][: k + 1], res_3d["y"][: k + 1], "--", color="#1f77b4", alpha=0.6, lw=1.5, label="CoM Path")
        ax2d.plot(nodes[k, :, 0], nodes[k, :, 1], "-o", color="#d62728", lw=3.0, markersize=4.5, label="12 Modules")
        ax2d.scatter([nodes[k, 0, 0]], [nodes[k, 0, 1]], color="black", s=55, zorder=6, label="Head")
        ax2d.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=160, color="#2ca02c", zorder=5, label="Goal")
        ax2d.set_xlim(-0.4, x_max)
        ax2d.set_ylim(-y_lim, y_lim)
        ax2d.set_xlabel("X Position (m)")
        ax2d.set_ylabel("Y Position (m)")
        ax2d.set_title(f"Top-Down Motion (v = {res_3d['speed_trace'][k]:.0f} mm/s)")
        ax2d.grid(True, alpha=0.3)
        ax2d.legend(loc="upper left", fontsize=8)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_gif, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"3D FEM animation GIF saved to {output_gif}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot 1D -> 3D SOFA FEM policy retention")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "ppoc_best.pt",
        help="Trained PPOC checkpoint path",
    )
    parser.add_argument("--external", action="store_true", help="Connect to an external running SOFA IPC server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=REPO_ROOT / "results" / "sofa_transfer_eval.png",
        help="Output path for 3D transfer summary plot",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=REPO_ROOT / "results" / "sofa_3d_animation.gif",
        help="Output path for 3D slithering animation GIF",
    )
    parser.add_argument("--no-animate", action="store_true", help="Skip GIF animation rendering")
    parser.add_argument("--duration", type=float, default=6.0, help="Evaluation duration in seconds")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        args.checkpoint = REPO_ROOT / "checkpoints" / "ppoc_final.pt"
    agent = PPOCAgent.load(args.checkpoint)

    goal_xy = np.array([2.5, 0.0], dtype=np.float64)
    res_1d = eval_1d_speed(agent, goal_xy=goal_xy, duration_s=args.duration)

    bridge = None
    port = args.port
    if not args.external:
        bridge = SofaLocalFEMBridge(REPO_ROOT / "configs" / "plant_rev_c.yaml", host=args.host, port=0)
        port = bridge.start()

    try:
        res_3d = eval_3d_sofa_speed(agent, goal_xy=goal_xy, host=args.host, port=port, duration_s=args.duration)
    finally:
        if bridge is not None:
            bridge.stop()

    retention = (res_3d["speed_mm_s"] / max(res_1d["speed_mm_s"], 1e-6)) * 100.0
    print("=" * 68)
    print("ZERO-SHOT 1D -> 3D SOFA FEM TRANSFER EVALUATION (FOC c=0.75)")
    print("-" * 68)
    print(f"1D Planar Baseline Speed : {res_1d['speed_mm_s']:6.1f} mm/s ({res_1d['dist_m']:.3f} m in {res_1d['elapsed_s']:.2f} s)")
    print(f"3D FEM IPC Transfer Speed: {res_3d['speed_mm_s']:6.1f} mm/s ({res_3d['dist_m']:.3f} m in {res_3d['elapsed_s']:.2f} s)")
    print(f"Velocity Retention Ratio : {retention:6.2f}% (Target >= 85.00%)")
    print(f"Hexahedral Node Telemetry: {res_3d['hex_nodes_shape']}, DEA Breakdowns: {res_3d['breakdowns']}")
    print("=" * 68)

    save_sofa_visual_plot(res_1d, res_3d, goal_xy, retention, args.output_plot)
    if not args.no_animate:
        save_sofa_animation_gif(res_3d, goal_xy, args.output_gif)

    if retention < 85.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
