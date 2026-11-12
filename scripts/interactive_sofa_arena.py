"""Interactive Multi-Region 3D SOFA Arena & Click-to-Add-Waypoints Controller.

Provides a live overhead interactive map synced over TCP IPC to SOFA v26.06.00:
  - Left-Click anywhere on the arena floor to queue a target waypoint (syncs 3D beacons in SOFA).
  - Right-Click anywhere to override and steer immediately to a new target point.
  - Switch live between Hierarchical PPOC-CPG (RL) and Classical PI+ILOS.
  - Switch between Multi-Region Arena (sand + L/2 tile trap + upper tile + rough gravel),
    Tile Corridor (L/2), and Open Sand.
"""

from __future__ import annotations

import argparse
from collections import deque
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controllers.classical_pi_ilos import ClassicalPIILOSController
from src.controllers.inner_loop_pi import CapacitiveInnerLoopPI
from src.envs.snake_gym_env import map_action_to_cpg_alpha
from src.envs.sofa_connector import SofaConnector, SofaIPCConfig, SofaLocalFEMBridge
from src.plant.dea_actuator import DEAActuatorPlant, DEAConfig
from src.rl.option_critic_ppoc import PPOCAgent
from src.spine.matsuoka_ode import MatsuokaCPG, MatsuokaCPGConfig


ARENA_BOUNDS = (-1.0, 3.8, -1.6, 1.6)  # (x_min, x_max, y_min, y_max)

ARENA_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "Multi-Region Arena": [
        {
            "label": "Tile Trap (L/2, eta=1.0)",
            "x_min": 0.70,
            "x_max": 1.00,
            "y_min": -1.8,
            "y_max": 1.8,
            "mu_t": 0.30,
            "mu_n": 0.30,
            "eta": 1.0,
        },
        {
            "label": "Upper Tile Pad (eta=1.0)",
            "x_min": 1.80,
            "x_max": 2.15,
            "y_min": 0.35,
            "y_max": 1.35,
            "mu_t": 0.30,
            "mu_n": 0.30,
            "eta": 1.0,
        },
        {
            "label": "Rough Gravel (eta=4.0)",
            "x_min": 1.55,
            "x_max": 2.45,
            "y_min": -1.45,
            "y_max": -0.45,
            "mu_t": 0.14,
            "mu_n": 0.56,
            "eta": 4.0,
        },
    ],
    "Tile Corridor (L/2)": [
        {
            "label": "Tile Trap (L/2, eta=1.0)",
            "x_min": 0.70,
            "x_max": 1.00,
            "y_min": -1.8,
            "y_max": 1.8,
            "mu_t": 0.30,
            "mu_n": 0.30,
            "eta": 1.0,
        }
    ],
    "Open Sand": [],
}


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _connect_with_retry(
    conn: SofaConnector,
    retries: int = 30,
    delay_s: float = 0.5,
) -> None:
    """Connect TCP IPC client with automatic retries while SOFA GUI starts up."""
    last_exc: Optional[Exception] = None
    for _ in range(retries):
        try:
            conn.connect()
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(delay_s)
    raise ConnectionError(
        f"Could not connect to SOFA server at {conn.config.host}:{conn.config.port}"
    ) from last_exc


class ArenaSimulationWorker(threading.Thread):
    """Background 20 Hz RL/Classical + 200 Hz CPG control thread talking to SOFA over IPC."""

    def __init__(
        self,
        agent: PPOCAgent,
        host: str,
        port: int,
        external: bool = True,
        goal_tolerance_m: float = 0.24,
        step_sleep_s: float = 0.0,
    ) -> None:
        super().__init__(daemon=True)
        self.agent = agent
        self.host = host
        self.port = port
        self.external = external
        self.goal_tolerance_m = goal_tolerance_m
        self.step_sleep_s = step_sleep_s

        self.cmd_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()

        self.plant_cfg_path = REPO_ROOT / "configs" / "plant_rev_c.yaml"
        self.cpg_cfg = MatsuokaCPGConfig.from_yaml(REPO_ROOT / "configs" / "cpg_matsuoka.yaml")
        self.cpg = MatsuokaCPG(self.cpg_cfg)
        self.dea = DEAActuatorPlant(DEAConfig.from_yaml(self.plant_cfg_path))
        self.inner_pi = CapacitiveInnerLoopPI(self.dea)
        self.classical = ClassicalPIILOSController(self.dea)

        self.preset_name = "Multi-Region Arena"
        self.controller_mode = "PPOC-CPG"
        self.active_goal: Optional[List[float]] = [2.20, 0.00]
        self.waypoints: deque[List[float]] = deque([[2.85, -0.95], [1.40, 0.90]])
        self.start_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.goals_reached = 0
        self.paused = False

        self.bridge: Optional[SofaLocalFEMBridge] = None
        self.conn: Optional[SofaConnector] = None
        self.snapshot: Dict[str, Any] = {
            "connected": False,
            "status": "Connecting to SOFA...",
            "x_cm": 0.0,
            "y_cm": 0.0,
            "theta_cm": 0.0,
            "vx_mm_s": 0.0,
            "speed_mm_s": 0.0,
            "k_f": self.cpg_cfg.k_f_default,
            "freq_hz": 1.30 / self.cpg_cfg.k_f_default,
            "max_v": 0.0,
            "nodes": [],
            "trail": [],
            "active_goal": list(self.active_goal),
            "queue": list(self.waypoints),
            "goals_reached": 0,
            "region": "SAND (eta=3.0)",
            "controller_mode": self.controller_mode,
            "preset_name": self.preset_name,
            "paused": False,
        }

    def enqueue(self, cmd: str, payload: Any = None) -> None:
        self.cmd_queue.put((cmd, payload))

    def get_snapshot(self) -> Dict[str, Any]:
        with self.state_lock:
            snap = dict(self.snapshot)
            snap["trail"] = list(self.snapshot["trail"])
            snap["queue"] = list(self.snapshot["queue"])
            snap["active_goal"] = list(self.snapshot["active_goal"]) if self.snapshot["active_goal"] else None
            return snap

    def _region_under_point(self, x: float, y: float) -> str:
        patches = ARENA_PRESETS[self.preset_name]
        for p in patches:
            if p["x_min"] <= x <= p["x_max"] and p.get("y_min", -10) <= y <= p.get("y_max", 10):
                return p["label"]
        return "SAND (eta=3.0)"

    def _reset_controllers(self) -> None:
        self.cpg.reset(seed=77)
        self.dea.reset()
        self.inner_pi.reset()
        self.classical.reset()
        self.cpg.rollout(50, k_f=self.cpg_cfg.k_f_default)

    def _sync_goal_in_sofa(self) -> None:
        if self.conn is None:
            return
        goal = self.active_goal if self.active_goal is not None else [2.2, 0.0]
        self.conn.set_goal(goal_xy=goal, queue=list(self.waypoints))

    def _handle_commands(self, trail: deque[Tuple[float, float]]) -> Tuple[bool, Optional[int], Dict[str, Any]]:
        need_reset = False
        option: Optional[int] = None
        new_tel: Optional[Dict[str, Any]] = None

        while not self.cmd_queue.empty():
            cmd, payload = self.cmd_queue.get_nowait()
            if cmd == "add_waypoint":
                pt = [float(payload[0]), float(payload[1])]
                if self.active_goal is None:
                    self.active_goal = pt
                    self.start_xy = np.array([self.snapshot["x_cm"], self.snapshot["y_cm"]], dtype=np.float64)
                else:
                    self.waypoints.append(pt)
                self._sync_goal_in_sofa()
            elif cmd == "set_immediate_goal":
                pt = [float(payload[0]), float(payload[1])]
                self.waypoints.clear()
                self.active_goal = pt
                self.start_xy = np.array([self.snapshot["x_cm"], self.snapshot["y_cm"]], dtype=np.float64)
                self.paused = False
                self._sync_goal_in_sofa()
            elif cmd == "clear_waypoints":
                self.waypoints.clear()
                self.active_goal = None
                self._sync_goal_in_sofa()
            elif cmd == "load_tour":
                self.waypoints = deque(
                    [
                        [1.45, 0.05],
                        [2.45, 0.85],
                        [2.00, -0.90],
                        [0.35, -0.60],
                        [2.25, 0.00],
                    ]
                )
                self.active_goal = self.waypoints.popleft()
                self.paused = False
                self._sync_goal_in_sofa()
            elif cmd == "set_mode":
                self.controller_mode = str(payload)
                self.classical.reset()
                self.start_xy = np.array([self.snapshot["x_cm"], self.snapshot["y_cm"]], dtype=np.float64)
                option = None
            elif cmd == "set_preset":
                if payload in ARENA_PRESETS:
                    self.preset_name = str(payload)
                    assert self.conn is not None
                    self.conn.set_patches(sand_eta=3.0, tile_patches=ARENA_PRESETS[self.preset_name])
            elif cmd == "toggle_pause":
                self.paused = not self.paused
            elif cmd == "reset":
                need_reset = True

        if need_reset and self.conn is not None:
            self._reset_controllers()
            trail.clear()
            self.start_xy = np.array([0.0, 0.0], dtype=np.float64)
            self.goals_reached = 0
            if self.active_goal is None:
                self.active_goal = [2.20, 0.00]
            new_tel = self.conn.reset(
                sand_eta=3.0,
                tile_patches=ARENA_PRESETS[self.preset_name],
                goal_xy=self.active_goal,
                queue=list(self.waypoints),
            )
            option = None

        return need_reset, option, (new_tel if new_tel is not None else {})

    def run(self) -> None:
        port = self.port
        if not self.external:
            self.bridge = SofaLocalFEMBridge(self.plant_cfg_path, host=self.host, port=0)
            port = self.bridge.start()

        self.conn = SofaConnector(SofaIPCConfig(host=self.host, port=port, timeout_s=10.0))
        try:
            _connect_with_retry(self.conn, retries=30, delay_s=0.5)
            self._reset_controllers()
            tel = self.conn.reset(
                sand_eta=3.0,
                tile_patches=ARENA_PRESETS[self.preset_name],
                goal_xy=self.active_goal,
                queue=list(self.waypoints),
            )
            option: Optional[int] = None
            trail: deque[Tuple[float, float]] = deque(maxlen=400)
            dt = self.cpg_cfg.dt
            action_repeat = 10
            k_f = self.cpg_cfg.k_f_default

            while not self.stop_event.is_set():
                did_reset, maybe_opt, reset_tel = self._handle_commands(trail)
                if did_reset and reset_tel:
                    tel = reset_tel
                    option = maybe_opt

                if self.paused or self.active_goal is None:
                    # Gentle idle hold until user unpauses or clicks a waypoint
                    voltages = np.zeros((self.cpg_cfg.num_modules, 2), dtype=np.float64)
                    tel = self.conn.step(voltages, dt=dt)
                    time.sleep(0.03)
                else:
                    goal_xy = np.array(self.active_goal, dtype=np.float64)
                    if self.controller_mode == "PPOC-CPG":
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

                        alpha, option, *_ = self.agent.select_action(
                            obs, current_option=option, deterministic=True, update_norm=False
                        )
                        k_f = self.agent.get_k_f(option)
                        alpha_cpg = map_action_to_cpg_alpha(
                            action=alpha,
                            num_modules=self.cpg_cfg.num_modules,
                            alpha_clip=self.cpg_cfg.alpha_clip,
                            alpha_cpg_scale=0.38,
                        )
                        for _ in range(action_repeat):
                            psi = self.cpg.step(alpha=alpha_cpg, k_f=k_f, dt=dt)
                            kappa_ref = 12.5 * psi
                            voltages = self.inner_pi.step(
                                kappa_ref=kappa_ref,
                                capacitance_pf=np.asarray(tel["capacitance_pf"]),
                                dt=dt,
                            )
                            tel = self.conn.step(voltages, dt=dt)
                            if self.stop_event.is_set():
                                break
                    else:
                        k_f = 1.30 / 1.60  # Classical runs fixed 1.60 Hz baseline
                        for _ in range(action_repeat):
                            voltages = self.classical.step(
                                plant_state=tel,
                                capacitance_pf=np.asarray(tel["capacitance_pf"]),
                                goal_xy=goal_xy,
                                start_xy=self.start_xy,
                                dt=dt,
                            )
                            tel = self.conn.step(voltages, dt=dt)
                            if self.stop_event.is_set():
                                break

                    dist_now = float(np.hypot(goal_xy[0] - tel["x_cm"], goal_xy[1] - tel["y_cm"]))
                    if dist_now <= self.goal_tolerance_m:
                        self.goals_reached += 1
                        self.start_xy = np.array([tel["x_cm"], tel["y_cm"]], dtype=np.float64)
                        if self.waypoints:
                            self.active_goal = self.waypoints.popleft()
                        else:
                            self.active_goal = None
                        self._sync_goal_in_sofa()

                trail.append((float(tel["x_cm"]), float(tel["y_cm"])))
                speed_mm_s = float(np.hypot(tel["vx"], tel["vy"]) * 1000.0)
                with self.state_lock:
                    self.snapshot = {
                        "connected": True,
                        "status": f"Live Connected ({'SOFA GUI' if self.external else 'Local FEM'})",
                        "x_cm": float(tel["x_cm"]),
                        "y_cm": float(tel["y_cm"]),
                        "theta_cm": float(tel["theta_cm"]),
                        "vx_mm_s": float(tel["vx"] * 1000.0),
                        "speed_mm_s": speed_mm_s,
                        "k_f": float(k_f),
                        "freq_hz": float(1.30 / max(k_f, 0.1)),
                        "max_v": float(np.max(self.dea.v_state)),
                        "nodes": np.asarray(tel["hex_nodes"], dtype=np.float64).tolist(),
                        "trail": list(trail),
                        "active_goal": list(self.active_goal) if self.active_goal else None,
                        "queue": list(self.waypoints),
                        "goals_reached": self.goals_reached,
                        "region": self._region_under_point(float(tel["x_cm"]), float(tel["y_cm"])),
                        "controller_mode": self.controller_mode,
                        "preset_name": self.preset_name,
                        "paused": self.paused,
                    }
                if self.step_sleep_s > 0.0:
                    time.sleep(self.step_sleep_s)
        except Exception as exc:
            with self.state_lock:
                self.snapshot["connected"] = False
                self.snapshot["status"] = f"Disconnected: {exc}"
        finally:
            try:
                self.conn.close()
            except OSError:
                pass
            if self.bridge is not None:
                self.bridge.stop()


def launch_tk_arena_ui(worker: ArenaSimulationWorker) -> None:
    """Open interactive Tkinter Click-to-Target Arena Radar synced to 3D SOFA."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Soft Snake Multi-Region Arena (Click to Add Waypoints)")
    root.geometry("1040x680")
    root.configure(bg="#181c24")

    x_min, x_max, y_min, y_max = ARENA_BOUNDS
    cw, ch = 980, 520
    pad = 40

    def world_to_px(wx: float, wy: float) -> Tuple[float, float]:
        px = pad + ((wx - x_min) / (x_max - x_min)) * (cw - 2 * pad)
        py = ch - pad - ((wy - y_min) / (y_max - y_min)) * (ch - 2 * pad)
        return px, py

    def px_to_world(px: float, py: float) -> Tuple[float, float]:
        wx = x_min + ((px - pad) / (cw - 2 * pad)) * (x_max - x_min)
        wy = y_min + ((ch - pad - py) / (ch - 2 * pad)) * (y_max - y_min)
        return (
            float(np.clip(wx, x_min + 0.25, x_max - 0.25)),
            float(np.clip(wy, y_min + 0.20, y_max - 0.20)),
        )

    # Top Control Strip
    bar = tk.Frame(root, bg="#222834", pady=8, padx=10)
    bar.pack(fill=tk.X)

    mode_var = tk.StringVar(value="PPOC-CPG")
    preset_var = tk.StringVar(value="Multi-Region Arena")

    tk.Label(bar, text="Controller:", fg="#e2e8f0", bg="#222834", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
    for m_label in ("PPOC-CPG", "Classical PI+ILOS"):
        tk.Radiobutton(
            bar,
            text=m_label,
            variable=mode_var,
            value=m_label,
            command=lambda m=m_label: worker.enqueue("set_mode", m),
            bg="#222834",
            fg="#93c5fd" if m_label == "PPOC-CPG" else "#fca5a5",
            selectcolor="#0f172a",
            activebackground="#222834",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=4)

    tk.Label(bar, text="   Arena:", fg="#e2e8f0", bg="#222834", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(10, 4))
    preset_menu = ttk.Combobox(
        bar, textvariable=preset_var, values=list(ARENA_PRESETS.keys()), state="readonly", width=18
    )
    preset_menu.pack(side=tk.LEFT, padx=4)
    preset_menu.bind("<<ComboboxSelected>>", lambda _e: worker.enqueue("set_preset", preset_var.get()))

    def mk_btn(txt: str, cmd: Any, color: str = "#3b82f6") -> None:
        tk.Button(
            bar, text=txt, command=cmd, bg=color, fg="white", relief=tk.FLAT,
            padx=10, pady=3, font=("Segoe UI", 9, "bold"), cursor="hand2"
        ).pack(side=tk.LEFT, padx=4)

    mk_btn("Reset Snake", lambda: worker.enqueue("reset"), "#059669")
    mk_btn("Zig-Zag Tour", lambda: worker.enqueue("load_tour"), "#6366f1")
    mk_btn("Clear Queue", lambda: worker.enqueue("clear_waypoints"), "#d97706")
    mk_btn("Pause / Run", lambda: worker.enqueue("toggle_pause"), "#475569")

    # Two-line responsive HUD Strip so text never overflows window edges
    status_var = tk.StringVar(value="Connecting to SOFA...")
    hud_var = tk.StringVar(value="Speed:   0.0 mm/s  |  K_f=0.81 (1.60 Hz)  |  Vmax=0V/450V  |  Goals=0")
    tk.Label(
        root,
        textvariable=status_var,
        fg="#38bdf8",
        bg="#181c24",
        font=("Consolas", 9, "bold"),
        pady=1,
    ).pack(fill=tk.X)
    tk.Label(
        root,
        textvariable=hud_var,
        fg="#a7f3d0",
        bg="#181c24",
        font=("Consolas", 9, "bold"),
        pady=2,
    ).pack(fill=tk.X)

    canvas = tk.Canvas(root, width=cw, height=ch, bg="#e8d5a7", highlightthickness=2, highlightbackground="#334155")
    canvas.pack(pady=4)

    legend = tk.Label(
        root,
        text="Left-Click = Queue Amber Waypoint  |  Right-Click = Instant Green Beacon Target  |  Sand (eta=3.0)  |  Slate = Slick Tile (eta=1.0)  |  Terracotta = Rough Gravel (eta=4.0)",
        fg="#cbd5e1",
        bg="#181c24",
        font=("Segoe UI", 9),
        pady=4,
    )
    legend.pack(fill=tk.X)

    canvas.bind("<Button-1>", lambda e: worker.enqueue("add_waypoint", px_to_world(e.x, e.y)))
    canvas.bind("<Button-3>", lambda e: worker.enqueue("set_immediate_goal", px_to_world(e.x, e.y)))

    def redraw() -> None:
        snap = worker.get_snapshot()
        canvas.delete("all")

        # 1. Multi-Region Substrate Patches
        for p in ARENA_PRESETS[snap["preset_name"]]:
            x0, y0 = world_to_px(max(p["x_min"], x_min), min(p.get("y_max", y_max), y_max))
            x1, y1 = world_to_px(min(p["x_max"], x_max), max(p.get("y_min", y_min), y_min))
            is_tile = p["eta"] <= 1.8
            fill_c = "#a5bbd4" if is_tile else "#c9835b"
            outline_c = "#475569" if is_tile else "#7c2d12"
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill_c, outline=outline_c, width=2, dash=(6, 4))
            canvas.create_text(
                0.5 * (x0 + x1), y0 + 14, text=p["label"], fill="#0f172a", font=("Segoe UI", 8, "bold")
            )

        # 2. Coordinate Grid
        for gx in np.arange(-0.5, 4.0, 0.5):
            px0, py0 = world_to_px(gx, y_max)
            px1, py1 = world_to_px(gx, y_min)
            canvas.create_line(px0, py0, px1, py1, fill="#cbb789", dash=(2, 4))
            canvas.create_text(px1, ch - 18, text=f"{gx:.1f}m", fill="#1e293b", font=("Consolas", 8))
        for gy in np.arange(-1.5, 2.0, 0.5):
            px0, py0 = world_to_px(x_min, gy)
            px1, py1 = world_to_px(x_max, gy)
            canvas.create_line(px0, py0, px1, py1, fill="#cbb789", dash=(2, 4))
            canvas.create_text(20, py0, text=f"{gy:+.1f}", fill="#1e293b", font=("Consolas", 8))

        # 3. Breadcrumb Trail
        trail = snap["trail"]
        if len(trail) >= 2:
            pts_flat = []
            for tx, ty in trail:
                px, py = world_to_px(tx, ty)
                pts_flat.extend([px, py])
            canvas.create_line(*pts_flat, fill="#0284c7", width=2)

        # 4. Planned Waypoint Path + Markers
        sx, sy = snap["x_cm"], snap["y_cm"]
        path_pts = [world_to_px(sx, sy)]
        if snap["active_goal"] is not None:
            gx, gy = snap["active_goal"]
            gpx, gpy = world_to_px(gx, gy)
            path_pts.append((gpx, gpy))
            for qx, qy in snap["queue"]:
                path_pts.append(world_to_px(qx, qy))
            for i in range(len(path_pts) - 1):
                canvas.create_line(
                    path_pts[i][0], path_pts[i][1], path_pts[i + 1][0], path_pts[i + 1][1],
                    fill="#16a34a" if i == 0 else "#d97706", width=2, dash=(4, 4)
                )
            # Active goal tolerance ring & green beacon
            r_px = (worker.goal_tolerance_m / (x_max - x_min)) * (cw - 2 * pad)
            canvas.create_oval(gpx - r_px, gpy - r_px, gpx + r_px, gpy + r_px, outline="#16a34a", width=2, dash=(3, 3))
            canvas.create_oval(gpx - 8, gpy - 8, gpx + 8, gpy + 8, fill="#22c55e", outline="#052e16", width=2)
            canvas.create_text(gpx, gpy - 15, text=f"GOAL ({gx:.2f},{gy:.2f})", fill="#064e3b", font=("Segoe UI", 8, "bold"))

            for idx, (qx, qy) in enumerate(snap["queue"]):
                qpx, qpy = world_to_px(qx, qy)
                canvas.create_oval(qpx - 6, qpy - 6, qpx + 6, qpy + 6, fill="#f59e0b", outline="#78350f", width=2)
                canvas.create_text(qpx, qpy - 12, text=f"W{idx+1}", fill="#78350f", font=("Segoe UI", 8, "bold"))

        # 5. Articulated 12-Module Snake Backbone
        nodes = snap["nodes"]
        if len(nodes) >= 2:
            body_flat = []
            for pt in nodes:
                bx, by = world_to_px(float(pt[0]), float(pt[1]))
                body_flat.extend([bx, by])
            canvas.create_line(*body_flat, fill="#1d4ed8", width=6, capstyle=tk.ROUND, joinstyle=tk.ROUND)
            hx, hy = world_to_px(float(nodes[0][0]), float(nodes[0][1]))
            canvas.create_oval(hx - 6, hy - 6, hx + 6, hy + 6, fill="#ef4444", outline="#ffffff", width=2)

        state_tag = "PAUSED" if snap["paused"] else snap["controller_mode"]
        status_var.set(f"[{snap['status']}]   Mode: {state_tag}   |   Zone: {snap['region']}")
        hud_var.set(
            f"Speed: {snap['speed_mm_s']:5.1f} mm/s   |   K_f={snap['k_f']:.2f} ({snap['freq_hz']:.2f} Hz)   |   "
            f"Vmax={snap['max_v']:.0f}V/450V   |   Goals Reached: {snap['goals_reached']}"
        )

        root.after(40, redraw)

    root.protocol("WM_DELETE_WINDOW", lambda: (worker.stop_event.set(), root.destroy()))
    root.after(50, redraw)
    root.mainloop()


def _run_smoke_test() -> None:
    """Headless automated check of multi-region waypoint queue & live SOFA IPC updates."""
    ckpt = REPO_ROOT / "checkpoints" / "ppoc_best.pt"
    if not ckpt.exists():
        ckpt = REPO_ROOT / "checkpoints" / "ppoc_final.pt"
    agent = PPOCAgent.load(ckpt)
    worker = ArenaSimulationWorker(agent=agent, host="127.0.0.1", port=0, external=False)
    worker.start()
    time.sleep(0.3)
    worker.enqueue("add_waypoint", (1.6, 0.5))
    worker.enqueue("set_preset", "Multi-Region Arena")
    time.sleep(1.5)
    snap = worker.get_snapshot()
    worker.stop_event.set()
    worker.join(timeout=2.0)
    assert snap["connected"], f"Expected connected arena worker, got {snap['status']}"
    assert snap["max_v"] <= 450.0 + 1e-6
    print(
        f"[Arena Smoke-Test OK] region={snap['region']}, speed={snap['speed_mm_s']:.1f} mm/s, "
        f"pos=({snap['x_cm']:.3f}, {snap['y_cm']:.3f}), max_v={snap['max_v']:.1f}V"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Multi-Region Soft Snake Arena for SOFA")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "ppoc_best.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--local", action="store_true", help="Run with embedded 3D bridge instead of external SOFA GUI")
    parser.add_argument("--smoke-test", action="store_true", help="Run headless verification without opening Tk window")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    ckpt = args.checkpoint if args.checkpoint.exists() else (REPO_ROOT / "checkpoints" / "ppoc_final.pt")
    agent = PPOCAgent.load(ckpt)
    worker = ArenaSimulationWorker(
        agent=agent,
        host=args.host,
        port=args.port,
        external=not args.local,
    )
    worker.start()
    launch_tk_arena_ui(worker)


if __name__ == "__main__":
    main()

