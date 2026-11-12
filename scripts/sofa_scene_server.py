"""SofaPython3 3D Hyperelastic FEM Scene + TCP IPC Server for SOFA v26.06.00.

Builds a 12-module 3D hexahedral silicone continuum soft snake on a planar floor
using SOFA's native HexahedronFEMForceField, unilateral ground contact, and
self-equilibrating bilateral DEA bending force couples. Streams live 24-channel
DEA voltages from SofaConnector and returns 3D nodal positions & capacitive strain.

Usage (SOFA GUI):
    Set PATH & PYTHONPATH for sofa_py312 + SOFA v26.06.00, then:
    runSofa.exe -a scripts/sofa_scene_server.py
    python scripts/run_sofa_eval.py --external --host 127.0.0.1 --port 5555
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plant.dea_actuator import DEAActuatorPlant, DEAConfig
from src.plant.friction_coulomb import FrictionConfig
from src.plant.kinematics_1d import PlantGeometryConfig

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
    import SofaRuntime
except ImportError as exc:
    Sofa = None  # type: ignore[assignment]
    SofaRuntime = None  # type: ignore[assignment]
    _SOFA_IMPORT_ERR = exc
else:
    _SOFA_IMPORT_ERR = None


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("IPC client disconnected.")
        buf.extend(chunk)
    return bytes(buf)


def _recv_packet(sock: socket.socket) -> Dict[str, Any]:
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def _send_packet(sock: socket.socket, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def _build_dea_muscle_springs(
    nx: int,
    ny: int,
    nz: int,
    l_seg0: float,
    k_act: float,
    k_spine: float,
    d_act: float,
    d_spine: float,
) -> Dict[str, Any]:
    """Precompute implicit SpringForceField topology for neutral spine & bilateral DEA strips."""
    idx1, idx2, rest_lengths, stiffnesses, dampings = [], [], [], [], []
    right_spring_idx, left_spring_idx = [], []
    right_mod_idx, left_mod_idx = [], []
    x_to_mod = np.array([min(11, (nx - 2 - xi) // 2) for xi in range(nx - 1)], dtype=int)

    for xi in range(nx - 1):
        mod_idx = int(x_to_mod[xi])
        for k in range(nz):
            base = k * ny * nx
            # 1. Neutral spine (j = 1)
            i_c0 = base + 1 * nx + xi
            idx1.append(i_c0)
            idx2.append(i_c0 + 1)
            rest_lengths.append(l_seg0)
            stiffnesses.append(k_spine / nz)
            dampings.append(d_spine / nz)

            # 2. Right DEA laminate strip (j = 0)
            right_spring_idx.append(len(idx1))
            right_mod_idx.append(mod_idx)
            i_r0 = base + 0 * nx + xi
            idx1.append(i_r0)
            idx2.append(i_r0 + 1)
            rest_lengths.append(l_seg0)
            stiffnesses.append(k_act / nz)
            dampings.append(d_act / nz)

            # 3. Left DEA laminate strip (j = ny - 1)
            left_spring_idx.append(len(idx1))
            left_mod_idx.append(mod_idx)
            i_l0 = base + (ny - 1) * nx + xi
            idx1.append(i_l0)
            idx2.append(i_l0 + 1)
            rest_lengths.append(l_seg0)
            stiffnesses.append(k_act / nz)
            dampings.append(d_act / nz)

    return {
        "indices1": idx1,
        "indices2": idx2,
        "rest_lengths": rest_lengths,
        "stiffnesses": stiffnesses,
        "dampings": dampings,
        "x_to_mod": x_to_mod,
        "right_spring_idx": np.asarray(right_spring_idx, dtype=int),
        "left_spring_idx": np.asarray(left_spring_idx, dtype=int),
        "right_mod_idx": np.asarray(right_mod_idx, dtype=int),
        "left_mod_idx": np.asarray(left_mod_idx, dtype=int),
    }


if Sofa is not None:

    class DEASoftSnakeIPCController(Sofa.Core.Controller):
        """SOFA v26.06.00 controller applying implicit DEA springs and anisotropic contact."""

        def __init__(
            self,
            root_node: Any,
            snake_node: Any,
            spring_meta: Dict[str, Any],
            nx: int = 25,
            ny: int = 3,
            nz: int = 2,
            half_w: float = 0.012,
            host: str = "127.0.0.1",
            port: int = 5555,
            ipc_enabled: bool = True,
        ) -> None:
            super().__init__(name="DEASoftSnakeIPCController")
            self.root = root_node
            self.snake = snake_node
            self.nx = nx
            self.ny = ny
            self.nz = nz
            self.half_w = float(half_w)
            self.geom = PlantGeometryConfig.from_yaml(REPO_ROOT / "configs" / "plant_rev_c.yaml")
            self.dea_cfg = DEAConfig.from_yaml(REPO_ROOT / "configs" / "plant_rev_c.yaml")
            self.dea = DEAActuatorPlant(self.dea_cfg)
            self.fric_cfg = FrictionConfig.from_yaml(REPO_ROOT / "configs" / "plant_rev_c.yaml")
            self.l_seg0 = self.geom.body_length_m / (self.nx - 1)
            self.x_to_mod = spring_meta["x_to_mod"]
            self._rest_spring_lengths = np.asarray(spring_meta["rest_lengths"], dtype=np.float64)
            self._right_spring_idx = spring_meta["right_spring_idx"]
            self._left_spring_idx = spring_meta["left_spring_idx"]
            self._right_mod_idx = spring_meta["right_mod_idx"]
            self._left_mod_idx = spring_meta["left_mod_idx"]

            self.ipc_enabled = ipc_enabled
            self.client_sock: Optional[socket.socket] = None
            self.server_sock: Optional[socket.socket] = None
            self.voltages = np.zeros((self.geom.num_modules, 2), dtype=np.float64)
            self._rest_positions: Optional[np.ndarray] = None
            self._prev_kappa = np.zeros(self.geom.num_modules, dtype=np.float64)
            self._last_step_dt = 0.005
            self._pending_substeps = 0
            self.sand_mu_t = float(self.fric_cfg.sand.mu_t)
            self.sand_mu_n = float(self.fric_cfg.sand.mu_n)
            self.tile_patches: list[Dict[str, float]] = [
                {
                    "x_min": 0.70,
                    "x_max": 1.00,
                    "y_min": -1.8,
                    "y_max": 1.8,
                    "mu_t": float(self.fric_cfg.tile.mu_t),
                    "mu_n": float(self.fric_cfg.tile.mu_n),
                }
            ]
            self._goal_xy = [2.2, 0.0]
            self._goal_queue: list[list[float]] = []
            self._beacon_base: Optional[np.ndarray] = None
            self._queue_base: Optional[np.ndarray] = None
            self._unit_quad_base: Optional[np.ndarray] = None

            if self.ipc_enabled:
                self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_sock.settimeout(0.05)
                self.server_sock.bind((host, port))
                self.server_sock.listen(1)
                print(f"[SOFA IPC] Listening on {host}:{port} for SofaConnector...", flush=True)

        @staticmethod
        def _safe_child(parent: Any, name: str) -> Any:
            for ch in parent.children:
                if ch.name.value == name:
                    return ch
            return None

        def onSimulationInitDoneEvent(self, event: Any) -> None:
            mo = self.snake.getObject("MechanicalObject")
            self._rest_positions = np.asarray(mo.position.value, dtype=np.float64).copy()

            beacon_node = self._safe_child(self.root, "TargetBeacon")
            if beacon_node is not None:
                b_ogl = beacon_node.getObject("BeaconVisual")
                self._beacon_base = np.asarray(b_ogl.position.value, dtype=np.float64).copy()
            q0_node = self._safe_child(self.root, "QueueMarker0")
            if q0_node is not None:
                q_ogl = q0_node.getObject("QueueVisual")
                self._queue_base = np.asarray(q_ogl.position.value, dtype=np.float64).copy()
            p0_node = self._safe_child(self.root, "TileVisual0")
            if p0_node is not None:
                p_ogl = p0_node.getObject("PatchVisual")
                self._unit_quad_base = np.asarray(p_ogl.position.value, dtype=np.float64).copy()

            self._sync_arena_visuals()
            self._sync_goal_visuals(self._goal_xy, self._goal_queue)

        def _sync_arena_visuals(self) -> None:
            if self._unit_quad_base is None:
                return
            tile_slots = ["TileVisual0", "TileVisual1", "TileVisual2"]
            gravel_slots = ["GravelVisual0"]
            tile_list = []
            gravel_list = []
            for p in self.tile_patches:
                eta = float(p.get("mu_n", 0.30)) / max(float(p.get("mu_t", 0.30)), 1e-6)
                if eta <= 1.8:
                    tile_list.append(p)
                else:
                    gravel_list.append(p)

            def _apply_slots(slot_names: list[str], patches: list[Dict[str, float]], z_height: float) -> None:
                for idx, name in enumerate(slot_names):
                    node = self._safe_child(self.root, name)
                    if node is None:
                        continue
                    ogl = node.getObject("PatchVisual")
                    pts = self._unit_quad_base.copy()
                    if idx < len(patches):
                        pat = patches[idx]
                        x0 = max(float(pat.get("x_min", 0.0)), -1.5)
                        x1 = min(float(pat.get("x_max", 0.0)), 4.5)
                        y0 = max(float(pat.get("y_min", -2.0)), -2.0)
                        y1 = min(float(pat.get("y_max", 2.0)), 2.0)
                        pts[:, 0] = x0 + pts[:, 0] * max(x1 - x0, 0.02)
                        pts[:, 1] = y0 + pts[:, 1] * max(y1 - y0, 0.02)
                        pts[:, 2] = z_height
                    else:
                        pts[:, 2] = -3.0
                    ogl.position.value = pts.tolist()

            _apply_slots(tile_slots, tile_list, z_height=0.0005)
            _apply_slots(gravel_slots, gravel_list, z_height=0.0007)

        def _sync_goal_visuals(
            self,
            goal_xy: list[float],
            queue: Optional[list[list[float]]] = None,
        ) -> None:
            self._goal_xy = [float(goal_xy[0]), float(goal_xy[1])]
            if queue is not None:
                self._goal_queue = [[float(pt[0]), float(pt[1])] for pt in queue]

            beacon_node = self._safe_child(self.root, "TargetBeacon")
            if beacon_node is not None and self._beacon_base is not None:
                b_ogl = beacon_node.getObject("BeaconVisual")
                pts = self._beacon_base.copy()
                pts[:, 0] += self._goal_xy[0]
                pts[:, 1] += self._goal_xy[1]
                b_ogl.position.value = pts.tolist()

            if self._queue_base is not None:
                for idx in range(8):
                    q_node = self._safe_child(self.root, f"QueueMarker{idx}")
                    if q_node is None:
                        continue
                    q_ogl = q_node.getObject("QueueVisual")
                    pts = self._queue_base.copy()
                    if idx < len(self._goal_queue):
                        pts[:, 0] += self._goal_queue[idx][0]
                        pts[:, 1] += self._goal_queue[idx][1]
                        pts[:, 2] = 0.004
                    else:
                        pts[:, 2] = -3.0
                    q_ogl.position.value = pts.tolist()

        def _extract_module_centers_and_vels(self) -> tuple[np.ndarray, np.ndarray]:
            mo = self.snake.getObject("MechanicalObject")
            pos = np.asarray(mo.position.value, dtype=np.float64)
            vel = np.asarray(mo.velocity.value, dtype=np.float64)

            # RegularGridTopology orders indices (k * ny * nx + j * nx + i) where i=0 is tail (-L/2)
            # and i=nx-1 is head (+L/2). With nx=25 (24 elements = 2 per module), module m spans
            # x-indices [24 - 2*m, 23 - 2*m, 22 - 2*m]; evaluate each module center at its midpoint slice.
            grid_pos = pos.reshape(self.nz, self.ny, self.nx, 3)
            grid_vel = vel.reshape(self.nz, self.ny, self.nx, 3)

            n_mod = self.geom.num_modules
            centers = np.zeros((n_mod, 3), dtype=np.float64)
            vels = np.zeros((n_mod, 3), dtype=np.float64)
            for m in range(n_mod):
                x_mid = (self.nx - 2) - 2 * m  # 23, 21, ..., 1
                centers[m] = grid_pos[:, :, x_mid, :].mean(axis=(0, 1))
                vels[m] = grid_vel[:, :, x_mid, :].mean(axis=(0, 1))
            return centers, vels

        def _extract_telemetry(self) -> Dict[str, Any]:
            centers, vels = self._extract_module_centers_and_vels()
            n_mod = self.geom.num_modules

            x_cm = float(np.mean(centers[:, 0]))
            y_cm = float(np.mean(centers[:, 1]))
            vx = float(np.mean(vels[:, 0]))
            vy = float(np.mean(vels[:, 1]))

            # Segment tangents pointing tail -> head
            vecs = centers[:-1, :2] - centers[1:, :2]
            seg_angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            mean_sin = float(np.mean(np.sin(seg_angles)))
            mean_cos = float(np.mean(np.cos(seg_angles)))
            theta_cm = float(np.arctan2(mean_sin, mean_cos))

            ds = self.geom.segment_length_m
            kappa = np.zeros(n_mod, dtype=np.float64)
            dtheta = np.arctan2(np.sin(np.diff(-seg_angles)), np.cos(np.diff(-seg_angles)))
            if len(dtheta) > 0:
                kappa[1:-1] = dtheta / ds
                kappa[0] = kappa[1]
                kappa[-1] = kappa[-2]
            kappa_dot = (kappa - self._prev_kappa) / max(self._last_step_dt, 1e-6)
            self._prev_kappa = kappa.copy()

            cap_pf = self.dea.measure_capacitance(kappa)

            return {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "theta_cm": theta_cm,
                "vx": vx,
                "vy": vy,
                "omega": 0.0,
                "kappa": kappa.tolist(),
                "kappa_dot": kappa_dot.tolist(),
                "capacitance_pf": cap_pf.tolist(),
                "hex_nodes": centers.tolist(),
            }

        def _apply_dea_and_friction_forces(self) -> None:
            mo = self.snake.getObject("MechanicalObject")
            sp = self.snake.getObject("MuscleSprings")
            cff = self.snake.getObject("DEABendingForce")
            pos = np.asarray(mo.position.value, dtype=np.float64)
            vel = np.asarray(mo.velocity.value, dtype=np.float64)

            # 1. Implicit bilateral DEA muscle rest-length actuation via SpringForceField
            kappa_eq = np.clip(
                self.dea.target_curvature_from_voltages(self.voltages),
                -self.geom.max_curvature_rad_per_m,
                self.geom.max_curvature_rad_per_m,
            )
            new_lengths = self._rest_spring_lengths.copy()
            new_lengths[self._right_spring_idx] = self.l_seg0 * (
                1.0 + self.half_w * kappa_eq[self._right_mod_idx]
            )
            new_lengths[self._left_spring_idx] = self.l_seg0 * (
                1.0 - self.half_w * kappa_eq[self._left_mod_idx]
            )
            sp.length.value = new_lengths.tolist()

            centers, _ = self._extract_module_centers_and_vels()
            vecs = centers[:-1, :2] - centers[1:, :2]
            seg_angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            seg_angles = np.append(seg_angles, seg_angles[-1])

            # 2. Hu et al. (2009) anisotropic Coulomb ground shear traction on substrate contact columns
            n_nodes = len(pos)
            n_plane = self.nx * self.ny
            forces = np.zeros((n_nodes, 3), dtype=np.float64)
            bottom_mask = pos[:n_plane, 2] <= 0.004
            if np.any(bottom_mask):
                b_idx = np.flatnonzero(bottom_mask)
                normal_load = (self.geom.total_mass_kg * self.fric_cfg.gravity_mps2) / len(b_idx)
                xi_nodes = np.minimum(b_idx % self.nx, self.nx - 2)
                th_nodes = seg_angles[self.x_to_mod[xi_nodes]]
                c_th = np.cos(th_nodes)
                s_th = np.sin(th_nodes)

                mu_t = np.full(len(b_idx), self.sand_mu_t, dtype=np.float64)
                mu_n = np.full(len(b_idx), self.sand_mu_n, dtype=np.float64)
                if self.tile_patches:
                    px = pos[b_idx, 0]
                    py = pos[b_idx, 1]
                    for patch in self.tile_patches:
                        in_p = (
                            (px >= patch.get("x_min", 0.0))
                            & (px <= patch.get("x_max", 0.0))
                            & (py >= patch.get("y_min", -1e6))
                            & (py <= patch.get("y_max", 1e6))
                        )
                        mu_t[in_p] = float(patch.get("mu_t", self.fric_cfg.tile.mu_t))
                        mu_n[in_p] = float(patch.get("mu_n", self.fric_cfg.tile.mu_n))

                v_xy = vel[b_idx, :2]
                vt = v_xy[:, 0] * c_th + v_xy[:, 1] * s_th
                vn = -v_xy[:, 0] * s_th + v_xy[:, 1] * c_th
                ft = -normal_load * mu_t * np.tanh(vt / 0.01)
                fn = -normal_load * mu_n * np.tanh(vn / 0.01)
                fx_col = (ft * c_th - fn * s_th) / self.nz
                fy_col = (ft * s_th + fn * c_th) / self.nz
                # Distribute planar contact shear evenly across vertical cross-section nodes
                # to eliminate coarse-mesh nz=2 shear/roll torque while preserving net ground traction.
                for k_layer in range(self.nz):
                    layer_idx = b_idx + k_layer * n_plane
                    forces[layer_idx, 0] = fx_col
                    forces[layer_idx, 1] = fy_col

            cff.forces.value = forces.tolist()

        def onAnimateBeginEvent(self, event: Any) -> None:
            if not self.ipc_enabled:
                self._apply_dea_and_friction_forces()
                return

            if self._pending_substeps > 0:
                self._apply_dea_and_friction_forces()
                return

            if self.client_sock is None and self.server_sock is not None:
                try:
                    self.client_sock, addr = self.server_sock.accept()
                    self.client_sock.settimeout(10.0)
                    print(f"[SOFA IPC] Connected to {addr}", flush=True)
                except socket.timeout:
                    self._apply_dea_and_friction_forces()
                    return

            if self.client_sock is not None:
                while True:
                    try:
                        req = _recv_packet(self.client_sock)
                    except (ConnectionError, OSError, socket.timeout):
                        self.client_sock.close()
                        self.client_sock = None
                        break

                    cmd = req.get("cmd")
                    if cmd == "close":
                        self.client_sock.close()
                        self.client_sock = None
                        break
                    if cmd == "reset":
                        self.voltages.fill(0.0)
                        self._pending_substeps = 0
                        self._prev_kappa.fill(0.0)
                        sand_eta = req.get("sand_eta")
                        self.sand_mu_t = float(self.fric_cfg.sand.mu_t)
                        self.sand_mu_n = (
                            self.sand_mu_t * float(sand_eta)
                            if sand_eta is not None
                            else float(self.fric_cfg.sand.mu_n)
                        )
                        self.tile_patches = list(req.get("tile_patches") or [])
                        self._sync_arena_visuals()
                        if "goal_xy" in req and req["goal_xy"] is not None:
                            self._sync_goal_visuals(req["goal_xy"], req.get("queue"))
                        elif "queue" in req:
                            self._sync_goal_visuals(self._goal_xy, req.get("queue"))
                        mo = self.snake.getObject("MechanicalObject")
                        if self._rest_positions is not None:
                            mo.position.value = self._rest_positions.tolist()
                            mo.velocity.value = np.zeros_like(self._rest_positions).tolist()
                        self._apply_dea_and_friction_forces()
                        _send_packet(self.client_sock, self._extract_telemetry())
                    elif cmd == "set_goal":
                        self._sync_goal_visuals(req.get("goal_xy", self._goal_xy), req.get("queue"))
                        _send_packet(self.client_sock, {"ok": True})
                    elif cmd == "set_patches":
                        sand_eta = req.get("sand_eta")
                        if sand_eta is not None:
                            self.sand_mu_t = float(self.fric_cfg.sand.mu_t)
                            self.sand_mu_n = self.sand_mu_t * float(sand_eta)
                        self.tile_patches = list(req.get("tile_patches") or [])
                        self._sync_arena_visuals()
                        _send_packet(self.client_sock, {"ok": True})
                    elif cmd == "step":
                        v_arr = np.asarray(req["voltages"], dtype=np.float64).reshape(
                            self.geom.num_modules, 2
                        )
                        self.voltages = np.clip(v_arr, 0.0, self.dea_cfg.v_max_v)
                        req_dt = float(req.get("dt", 0.005))
                        self._last_step_dt = req_dt
                        self._pending_substeps = max(1, int(round(req_dt / float(self.root.dt.value))))
                        self._apply_dea_and_friction_forces()
                        break

        def onAnimateEndEvent(self, event: Any) -> None:
            if self.ipc_enabled and self.client_sock is not None and self._pending_substeps > 0:
                self._pending_substeps -= 1
                if self._pending_substeps == 0:
                    try:
                        _send_packet(self.client_sock, self._extract_telemetry())
                    except OSError:
                        self.client_sock.close()
                        self.client_sock = None


def createScene(
    root_node: Any,
    ipc_enabled: bool = True,
    young_modulus: float = 1.0e6,
    k_act: float = 60000.0,
    k_spine: float = 90000.0,
    d_act: float = 14.0,
    d_spine: float = 20.0,
    rayleigh_stiffness: float = 0.002,
    plane_stiffness: float = 400.0,
    plane_damping: float = 4.0,
    half_w: Optional[float] = None,
    port: int = 5555,
) -> Any:
    """Build 12-module 3D hexahedral silicone soft snake scene in SOFA v26.06.00."""
    if Sofa is None:
        raise RuntimeError(
            "SofaPython3 is required to run inside SOFA. "
            f"Import error: {_SOFA_IMPORT_ERR}"
        )

    geom = PlantGeometryConfig.from_yaml(REPO_ROOT / "configs" / "plant_rev_c.yaml")
    root_node.gravity = [0.0, 0.0, -9.81]
    root_node.dt = 0.0025

    required = [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Mass",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ]
    for plugin in required:
        SofaRuntime.importPlugin(plugin)

    root_node.addObject("RequiredPlugin", pluginName=required)
    root_node.addObject("VisualStyle", displayFlags="showVisualModels showBehaviorModels")
    root_node.addObject("DefaultAnimationLoop")
    root_node.addObject(
        "InteractiveCamera",
        name="ArenaCamera",
        position=[1.1, -1.75, 1.35],
        lookAt=[1.1, 0.0, 0.0],
        fieldOfView=48,
    )

    # Multi-Region Arena Base Floor (Sand eta = 3.0)
    floor = root_node.addChild("GroundFloor")
    floor.addObject(
        "RegularGridTopology",
        name="FloorGrid",
        n=[16, 12, 1],
        min=[-1.5, -2.0, 0.0],
        max=[4.5, 2.0, 0.0],
    )
    floor.addObject("OglModel", name="FloorVisual", src="@FloorGrid", color=[0.88, 0.80, 0.62, 1.0])

    # Dynamic Visual Substrate Patches (cool silver-slate for slick tile, terracotta for rough gravel)
    for t_idx in range(3):
        pnode = root_node.addChild(f"TileVisual{t_idx}")
        pnode.addObject("RegularGridTopology", name="PatchGrid", n=[4, 4, 1], min=[0.0, 0.0, 0.0], max=[1.0, 1.0, 0.0])
        pnode.addObject("OglModel", name="PatchVisual", src="@PatchGrid", color=[0.66, 0.77, 0.88, 1.0])

    gnode = root_node.addChild("GravelVisual0")
    gnode.addObject("RegularGridTopology", name="PatchGrid", n=[4, 4, 1], min=[0.0, 0.0, 0.0], max=[1.0, 1.0, 0.0])
    gnode.addObject("OglModel", name="PatchVisual", src="@PatchGrid", color=[0.74, 0.44, 0.28, 1.0])

    # Glowing 3D Green Active Target Beacon (repositioned live via IPC 'set_goal')
    beacon = root_node.addChild("TargetBeacon")
    beacon.addObject("RegularGridTopology", name="BeaconGrid", n=[2, 2, 2], min=[-0.055, -0.055, 0.002], max=[0.055, 0.055, 0.065])
    beacon.addObject("OglModel", name="BeaconVisual", src="@BeaconGrid", color=[0.10, 0.90, 0.32, 1.0])

    # Upcoming Clicked Waypoint Markers (amber-gold pads, up to 8 queued)
    for q_idx in range(8):
        qnode = root_node.addChild(f"QueueMarker{q_idx}")
        qnode.addObject("RegularGridTopology", name="QueueGrid", n=[2, 2, 1], min=[-0.045, -0.045, 0.0], max=[0.045, 0.045, 0.0])
        qnode.addObject("OglModel", name="QueueVisual", src="@QueueGrid", color=[0.98, 0.76, 0.14, 1.0])

    # 12-module 3D Hexahedral Silicone Soft Snake Body (Rev C physical cross-section)
    nx, ny, nz = 25, 3, 2
    n_nodes = nx * ny * nz
    eff_half_w = 0.5 * geom.body_width_m if half_w is None else float(half_w)
    l_seg0 = geom.body_length_m / (nx - 1)
    spring_meta = _build_dea_muscle_springs(
        nx=nx,
        ny=ny,
        nz=nz,
        l_seg0=l_seg0,
        k_act=k_act,
        k_spine=k_spine,
        d_act=d_act,
        d_spine=d_spine,
    )

    snake = root_node.addChild("SoftSnakeFEM")
    snake.addObject("EulerImplicitSolver", rayleighStiffness=rayleigh_stiffness, rayleighMass=0.005)
    snake.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    snake.addObject(
        "RegularGridTopology",
        name="HexGrid",
        n=[nx, ny, nz],
        min=[-0.5 * geom.body_length_m, -eff_half_w, 0.001],
        max=[0.5 * geom.body_length_m, eff_half_w, geom.body_height_m + 0.001],
    )
    snake.addObject("MechanicalObject", name="MechanicalObject", template="Vec3d")
    snake.addObject("UniformMass", totalMass=geom.total_mass_kg)
    snake.addObject(
        "HexahedronFEMForceField",
        name="SiliconeFEM",
        youngModulus=young_modulus,
        poissonRatio=0.38,
        method="large",
    )
    snake.addObject(
        "SpringForceField",
        name="MuscleSprings",
        object1="@MechanicalObject",
        object2="@MechanicalObject",
        indices1=spring_meta["indices1"],
        indices2=spring_meta["indices2"],
        length=spring_meta["rest_lengths"],
        stiffness=spring_meta["stiffnesses"],
        damping=spring_meta["dampings"],
    )
    snake.addObject(
        "ConstantForceField",
        name="DEABendingForce",
        indices=list(range(n_nodes)),
        forces=[[0.0, 0.0, 0.0]] * n_nodes,
    )
    # Implicit planar belly-skate guide on all nodes' z DOF (preserves 3D x-y hex FEM compliance
    # while eliminating vertical snap-buckling / hex inversion of the high-aspect 4.5x12mm body).
    snake.addObject(
        "PartialFixedProjectiveConstraint",
        name="PlanarBellySkate",
        indices=list(range(n_nodes)),
        fixedDirections=[0, 0, 1],
    )
    # Implicit unilateral floor support at z = 0 (unconditionally stable under EulerImplicitSolver)
    snake.addObject(
        "PlaneForceField",
        name="GroundPlaneSupport",
        normal=[0.0, 0.0, 1.0],
        d=0.0,
        stiffness=plane_stiffness,
        damping=plane_damping,
    )

    vis = snake.addChild("Visual")
    vis.addObject("OglModel", name="VisualMesh", src="@../HexGrid", color=[0.18, 0.48, 0.78, 1.0])
    vis.addObject("IdentityMapping", input="@../MechanicalObject", output="@VisualMesh")

    root_node.addObject(
        DEASoftSnakeIPCController(
            root_node,
            snake,
            spring_meta=spring_meta,
            nx=nx,
            ny=ny,
            nz=nz,
            half_w=eff_half_w,
            port=port,
            ipc_enabled=ipc_enabled,
        )
    )
    return root_node


def _run_self_test(steps: int = 20) -> None:
    """Verify headless SOFA scene initialization & stepping without GUI."""
    root = Sofa.Core.Node("root")
    createScene(root, ipc_enabled=False)
    Sofa.Simulation.initRoot(root)
    ctrl = root.getObject("DEASoftSnakeIPCController")
    ctrl.voltages[:, 0] = 400.0
    ctrl.voltages[:, 1] = 100.0
    for _ in range(steps):
        Sofa.Simulation.animate(root, root.dt.value)
    tel = ctrl._extract_telemetry()
    print(
        f"[SOFA Self-Test OK] {steps} steps -> x_cm={tel['x_cm']:.4f} m, "
        f"y_cm={tel['y_cm']:.4f} m, hex_nodes={np.asarray(tel['hex_nodes']).shape}"
    )


def _run_ipc_server(max_steps: int = 100000, port: int = 5555) -> None:
    """Run headless SOFA v26.06.00 simulation loop serving TCP IPC requests."""
    root = Sofa.Core.Node("root")
    createScene(root, ipc_enabled=True, port=port)
    Sofa.Simulation.initRoot(root)
    ctrl = root.getObject("DEASoftSnakeIPCController")
    try:
        for _ in range(max_steps):
            Sofa.Simulation.animate(root, root.dt.value)
    except KeyboardInterrupt:
        pass
    finally:
        if ctrl.client_sock is not None:
            ctrl.client_sock.close()
        if ctrl.server_sock is not None:
            ctrl.server_sock.close()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _run_self_test()
    elif "--serve" in sys.argv:
        cli_port = 5555
        if "--port" in sys.argv:
            p_idx = sys.argv.index("--port")
            if p_idx + 1 < len(sys.argv):
                cli_port = int(sys.argv[p_idx + 1])
        _run_ipc_server(port=cli_port)
    else:
        print(
            "Launch inside SOFA GUI via:\n"
            "  .\\scripts\\launch_sofa_gui.ps1\n"
            "Or run headless SOFA IPC server via:\n"
            "  python scripts/sofa_scene_server.py --serve"
        )
