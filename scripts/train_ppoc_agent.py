"""Curriculum Training Entrypoint for Multi-Rate Option-Critic PPOC Agent.

Supports vectorized parallel environment workers (--num-envs 4) and 3-stage curriculum:
  Stage A: Straight-line & gentle heading tracking on uniform sand (eta = 3.0)
  Stage B: Multi-waypoint +/- 45 deg navigation on randomized sand (eta in [2.5, 3.5])
  Stage C: Sand + embedded polished tile patches of widths {L/4, L/2, L} (eta in [0.8, 1.2])
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.snake_gym_env import SnakeGymEnv
from src.plant.friction_coulomb import SubstrateProperties, TilePatch
from src.rl.option_critic_ppoc import PPOCAgent, PPOCConfig, PPOCRolloutBuffer


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _env_worker(
    conn: Any,
    config_path: str,
    seed: int,
) -> None:
    """Subprocess worker hosting an independent SnakeGymEnv instance."""
    env = SnakeGymEnv(training_config_path=config_path, curriculum_stage="A")
    obs, info = env.reset(seed=seed)
    conn.send((obs, info))
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action, k_f = data
            next_obs, rew, term, trunc, step_info = env.step(action, k_f=k_f)
            done = term or trunc
            final_x = float(step_info["x_cm"])
            reached = bool(step_info["reached_goal"])
            if done:
                next_obs, _ = env.reset()
            conn.send((next_obs, rew, done, final_x, reached))
        elif cmd == "set_stage":
            env.set_curriculum_stage(str(data))
        elif cmd == "close":
            conn.close()
            break


class ParallelSnakeEnvPool:
    """Lightweight multiprocessing pool for parallel SnakeGymEnv rollouts."""

    def __init__(self, num_envs: int, config_path: Path, base_seed: int) -> None:
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.procs = []
        self.obs = []

        for idx in range(num_envs):
            p_conn, c_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(c_conn, str(config_path), base_seed + idx * 101),
                daemon=True,
            )
            proc.start()
            c_conn.close()
            self.parent_conns.append(p_conn)
            self.procs.append(proc)

        for p_conn in self.parent_conns:
            o, _ = p_conn.recv()
            self.obs.append(o)

    def get_obs(self) -> np.ndarray:
        return np.stack(self.obs, axis=0)

    def set_stage(self, stage: str) -> None:
        for p_conn in self.parent_conns:
            p_conn.send(("set_stage", stage))

    def step(
        self,
        actions: np.ndarray,
        k_fs: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[bool]]:
        for idx, p_conn in enumerate(self.parent_conns):
            p_conn.send(("step", (actions[idx], k_fs[idx])))

        next_obs_list, rews, dones, final_xs, reached_list = [], [], [], [], []
        for idx, p_conn in enumerate(self.parent_conns):
            no, r, d, fx, rch = p_conn.recv()
            self.obs[idx] = no
            next_obs_list.append(no)
            rews.append(r)
            dones.append(d)
            final_xs.append(fx)
            reached_list.append(rch)

        return (
            np.stack(next_obs_list, axis=0),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            final_xs,
            reached_list,
        )

    def close(self) -> None:
        for p_conn in self.parent_conns:
            try:
                p_conn.send(("close", None))
            except OSError:
                pass
        for proc in self.procs:
            proc.join(timeout=2.0)


def _markov_cpg_teacher(obs: np.ndarray) -> Tuple[np.ndarray, int]:
    """Lead-compensated Markovian steering + macro-option K_f teacher for BC warm-start."""
    th_err = float(obs[0])
    d_goal = float(obs[1])
    v_fwd = float(obs[2])
    v_lat = float(obs[3])
    mean_k = float(np.mean(obs[4:16]))
    mean_kd = float(np.mean(obs[16:28]))

    # Option 1 (K_f=0.8) handles low-speed/stall recovery (v_fwd < 0.12 m/s), sharp turns
    # (|th_err| > 1.05 rad), and terminal docking without slew clipping or reverse runaway;
    # Option 0 (K_f=0.6) bursts at 2.16 Hz for fast cruise & momentum-based tile crossing.
    if v_fwd < 0.12 or abs(th_err) > 1.05:
        opt_idx = 1
    elif v_fwd < 0.25 or d_goal > 0.50:
        opt_idx = 0
    else:
        opt_idx = 1

    raw_a0 = (
        1.25 * th_err
        - 0.16 * mean_k * max(v_fwd, 0.0)
        - 0.04 * mean_k
        - 0.015 * mean_kd
        + 0.10 * v_lat
    )
    if v_fwd < 0.08 and abs(th_err) < 0.90:
        # Smoothly cancel zero-speed CPG derivative oscillation (|mean_kd| ~ 45 rad/m/s on glass)
        # and hold a continuous symmetry-breaking C-arc to escape isotropic glass pads
        stall_w = float(np.clip((0.08 - v_fwd) / 0.08, 0.0, 1.0))
        raw_a0 += stall_w * (0.012 * mean_kd + 0.52 * float(np.tanh((th_err + 0.28) / 0.20)))

    a0 = float(np.clip(raw_a0, -1.0, 1.0))
    return np.array([a0, 0.0, 0.0, 0.0], dtype=np.float32), opt_idx


def _warmstart_ppoc_bc(
    agent: PPOCAgent,
    config_path: str | Path,
    gamma: float = 0.98,
    num_episodes: int = 160,
) -> None:
    """Distill lead-compensated steering & bidirectional option switching into OptionCriticNetwork."""
    rng = np.random.default_rng(123)
    env = SnakeGymEnv(training_config_path=config_path, curriculum_stage="C")

    obs_list, act_list, opt_list, ret_list = [], [], [], []
    for ep in range(num_episodes):
        if ep % 4 == 3:
            # Start directly on an isotropic slick glass pad so opt=0 learns smooth glass-escape arcs
            env.set_curriculum_stage("C")
            angle = float(rng.uniform(-1.1, 1.1))
            dist = float(rng.uniform(0.9, 1.8))
            tile_sub = SubstrateProperties(
                mu_t=env.plant.friction.config.tile.mu_t,
                mu_n=env.plant.friction.config.tile.mu_t,
                stiction_factor=env.plant.friction.config.tile.stiction_factor,
                v_stiction_mps=env.plant.friction.config.tile.v_stiction_mps,
            )
            obs, _ = env.reset(
                seed=2000 + ep,
                options={
                    "goal_dist": dist,
                    "goal_angle": angle,
                    "sand_eta": 3.0,
                    "patches": [TilePatch(x_min=-0.45, x_max=0.40, y_min=-0.7, y_max=0.7, substrate=tile_sub)],
                },
            )
            retarget_done = True
        elif ep % 4 == 2:
            # Synthesize mid-tile sharp turns, border stalls, & slick isotropic pad escapes
            env.set_curriculum_stage("C")
            if ep % 8 == 6:
                angle = float(rng.uniform(0.65, 1.15) * rng.choice([-1.0, 1.0]))
                dist = float(rng.uniform(0.60, 0.95))
            else:
                angle = float(rng.uniform(-0.45, 0.45))
                dist = float(rng.uniform(0.75, 1.35))
            x0 = float(rng.uniform(0.20, 0.85))
            w0 = float(rng.uniform(0.30, 0.65))
            tile_sub = SubstrateProperties(
                mu_t=env.plant.friction.config.tile.mu_t,
                mu_n=env.plant.friction.config.tile.mu_t,
                stiction_factor=env.plant.friction.config.tile.stiction_factor,
                v_stiction_mps=env.plant.friction.config.tile.v_stiction_mps,
            )
            obs, _ = env.reset(
                seed=2000 + ep,
                options={
                    "goal_dist": dist,
                    "goal_angle": angle,
                    "sand_eta": 3.0,
                    "patches": [TilePatch(x_min=x0, x_max=x0 + w0, y_min=-0.8, y_max=0.8, substrate=tile_sub)],
                },
            )
            retarget_done = False
        else:
            stage = "B" if ep % 3 == 0 else "C"
            env.set_curriculum_stage(stage)
            angle = float(rng.uniform(-np.pi, np.pi))
            dist = float(rng.uniform(1.2, 3.0))
            obs, _ = env.reset(
                seed=2000 + ep,
                options={"goal_dist": dist, "goal_angle": angle},
            )
            retarget_done = True

        ep_obs, ep_act, ep_opt, ep_rew = [], [], [], []
        for step_idx in range(env.max_steps):
            act, opt_idx = _markov_cpg_teacher(obs)
            k_f = agent.get_k_f(opt_idx)
            ep_obs.append(agent.obs_norm.normalize(obs))
            ep_act.append(act)
            ep_opt.append(opt_idx)
            obs, rew, term, trunc, info = env.step(act, k_f=k_f)
            ep_rew.append(float(rew))
            if not retarget_done and (info["reached_goal"] or step_idx >= 32):
                # Trigger abrupt waypoint redirection on/near tile to supervise 0<->1 switches & escape
                retarget_done = True
                turn_delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.05, 1.90))
                new_th = float(info["theta_cm"]) + turn_delta
                new_d = float(rng.uniform(0.9, 1.8))
                env.goal_xy = np.array(
                    [float(info["x_cm"]) + new_d * np.cos(new_th), float(info["y_cm"]) + new_d * np.sin(new_th)],
                    dtype=np.float64,
                )
                obs, *_ = env._compute_obs()
            elif term or trunc:
                break

        g = 0.0
        ep_rets = np.zeros(len(ep_rew), dtype=np.float32)
        for t in reversed(range(len(ep_rew))):
            g = ep_rew[t] + gamma * g
            ep_rets[t] = g
        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        opt_list.extend(ep_opt)
        ret_list.extend(ep_rets.tolist())

    obs_t = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=agent.device)
    act_t = torch.as_tensor(np.asarray(act_list), dtype=torch.float32, device=agent.device)
    opt_t = torch.as_tensor(np.asarray(opt_list), dtype=torch.long, device=agent.device)
    ret_t = torch.as_tensor(np.asarray(ret_list), dtype=torch.float32, device=agent.device)

    bc_opt = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
    vf_scale = float(agent.obs_norm.scale[2])
    n_samples = len(obs_list)
    idx_arr = np.arange(n_samples)
    agent.network.train()
    for _ in range(110):
        np.random.shuffle(idx_arr)
        for s in range(0, n_samples, 256):
            mb = torch.as_tensor(idx_arr[s : s + 256], dtype=torch.long, device=agent.device)
            b_obs, b_act, b_opt, b_ret = obs_t[mb], act_t[mb], opt_t[mb], ret_t[mb]
            # Zero-mean per-segment harmonic augmentation preserves mean(kappa), mean(kappa_dot)
            # while making the distilled policy invariant to 1D->3D SOFA mode-shape shifts.
            b_obs_aug = b_obs.clone()
            nk = 0.18 * torch.randn_like(b_obs_aug[:, 4:16])
            nkd = 0.18 * torch.randn_like(b_obs_aug[:, 16:28])
            b_obs_aug[:, 4:16] = (
                b_obs_aug[:, 4:16]
                + (nk - nk.mean(dim=-1, keepdim=True))
                + 0.03 * torch.randn(b_obs.shape[0], 1, device=agent.device)
            )
            b_obs_aug[:, 16:28] = (
                b_obs_aug[:, 16:28]
                + (nkd - nkd.mean(dim=-1, keepdim=True))
                + 0.03 * torch.randn(b_obs.shape[0], 1, device=agent.device)
            )
            w_sample = torch.where((b_obs[:, 2] * vf_scale) < 0.15, 4.0, 1.0)

            intra_dist = agent.network.get_intra_dist(b_obs_aug, b_opt)
            master_logits = agent.network.master_head(agent.network.master_trunk(b_obs_aug))
            q_all = agent.network.get_q_values(b_obs_aug)
            betas_all = agent.network.get_termination_probs(b_obs_aug)
            opt_onehot = torch.nn.functional.one_hot(
                b_opt, num_classes=betas_all.shape[1]
            ).float()
            beta_tgt = 0.92 * (1.0 - opt_onehot) + 0.04 * opt_onehot
            q_tgt = b_ret.unsqueeze(1) * opt_onehot + (b_ret.unsqueeze(1) - 20.0) * (1.0 - opt_onehot)

            loss_act = (
                torch.nn.functional.mse_loss(intra_dist.mean, b_act, reduction="none").mean(dim=-1)
                * w_sample
            ).mean()
            loss_master = (
                torch.nn.functional.cross_entropy(master_logits, b_opt, reduction="none") * w_sample
            ).mean()
            loss_beta = (
                torch.nn.functional.binary_cross_entropy(
                    torch.clamp(betas_all, 1e-4, 1.0 - 1e-4), beta_tgt, reduction="none"
                ).mean(dim=-1)
                * w_sample
            ).mean()
            loss_q = torch.nn.functional.mse_loss(q_all, q_tgt)
            loss = 4.0 * loss_act + loss_master + loss_beta + 0.02 * loss_q
            bc_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)
            bc_opt.step()

    with torch.no_grad():
        agent.network.intra_log_std.fill_(-1.1)
    print(f"[BC Warmstart] Distilled {n_samples} transitions across {num_episodes} multi-angle/tile episodes.", flush=True)


def evaluate_policy(
    agent: PPOCAgent,
    stage: str = "C",
    fixed_tile_width: float = 0.30,
    num_episodes: int = 4,
) -> dict:
    """Run deterministic evaluation episodes across straight L/2 tile crossing and sharp turns."""
    env = SnakeGymEnv(curriculum_stage=stage, fixed_tile_width=fixed_tile_width)
    returns, successes, final_xs, mean_kfs, max_vs = [], [], [], [], []

    eval_angles = [0.0, 1.55, -1.55, -2.65]
    for ep in range(num_episodes):
        if ep == 4:
            tile_sub = SubstrateProperties(
                mu_t=env.plant.friction.config.tile.mu_t,
                mu_n=env.plant.friction.config.tile.mu_t,
                stiction_factor=env.plant.friction.config.tile.stiction_factor,
                v_stiction_mps=env.plant.friction.config.tile.v_stiction_mps,
            )
            obs, info = env.reset(
                seed=1000 + ep,
                options={
                    "goal_dist": 1.0,
                    "goal_angle": -1.45,
                    "sand_eta": 3.0,
                    "patches": [TilePatch(x_min=-0.18, x_max=0.17, y_min=-0.5, y_max=0.5, substrate=tile_sub)],
                },
            )
        else:
            ang = eval_angles[ep % len(eval_angles)]
            obs, info = env.reset(seed=1000 + ep, options={"goal_dist": 2.2, "goal_angle": ang})
        done = False
        ep_ret = 0.0
        opt = None
        kfs = []
        while not done:
            alpha, opt, *_ = agent.select_action(
                obs, current_option=opt, deterministic=True, update_norm=False
            )
            k_f = agent.get_k_f(opt)
            kfs.append(k_f)
            obs, rew, term, trunc, info = env.step(alpha, k_f=k_f)
            ep_ret += rew
            done = term or trunc
        returns.append(ep_ret)
        successes.append(float(info["reached_goal"]))
        final_xs.append(float(info["x_cm"]))
        mean_kfs.append(float(np.mean(kfs)))
        max_vs.append(float(info["max_voltage_v"]))

    return {
        "eval_return": float(np.mean(returns)),
        "eval_success": float(np.mean(successes)),
        "eval_final_x": float(np.mean(final_xs)),
        "eval_mean_kf": float(np.mean(mean_kfs)),
        "eval_max_v": float(np.max(max_vs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hierarchical PPOC Agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ppoc_training.yaml",
        help="Path to ppoc_training.yaml",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environment workers",
    )
    parser.add_argument(
        "--warmstart-bc",
        action="store_true",
        help="Warm-start OptionCriticNetwork from Markovian CPG steering before PPOC RL",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)
    curr_cfg = full_cfg.get("curriculum", {})
    stage_a_frac = float(curr_cfg.get("stage_a_fraction", 0.25))
    stage_b_frac = float(curr_cfg.get("stage_b_fraction", 0.35))

    ppoc_cfg = PPOCConfig.from_yaml(args.config)
    if args.total_timesteps is not None:
        ppoc_cfg.total_timesteps = int(args.total_timesteps)

    set_global_seeds(ppoc_cfg.seed)

    ref_env = SnakeGymEnv(training_config_path=args.config, curriculum_stage="A")
    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]

    agent = PPOCAgent(obs_dim=obs_dim, action_dim=act_dim, config=ppoc_cfg)
    ckpt_dir = REPO_ROOT / ppoc_cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_eval_score = -np.inf

    if args.warmstart_bc:
        _warmstart_ppoc_bc(agent, config_path=args.config, gamma=ppoc_cfg.gamma)
        bc_eval = evaluate_policy(agent, stage="C", fixed_tile_width=0.30, num_episodes=5)
        best_eval_score = bc_eval["eval_return"] + 250.0 * bc_eval["eval_success"]
        agent.save(ckpt_dir / "ppoc_best.pt")
        print(
            f"[BC Eval(L/2 tile)] Ret={bc_eval['eval_return']:6.1f} X={bc_eval['eval_final_x']:.2f}m "
            f"Succ={bc_eval['eval_success']*100:4.0f}% Kf={bc_eval['eval_mean_kf']:.2f} "
            f"Vmax={bc_eval['eval_max_v']:.0f}V",
            flush=True,
        )

    num_envs = max(1, int(args.num_envs))
    steps_per_env = max(64, ppoc_cfg.rollout_steps // num_envs)
    total_rollout_steps = steps_per_env * num_envs

    worker_buffers = [
        PPOCRolloutBuffer(capacity=steps_per_env, obs_dim=obs_dim, action_dim=act_dim)
        for _ in range(num_envs)
    ]
    merged_buffer = PPOCRolloutBuffer(capacity=total_rollout_steps, obs_dim=obs_dim, action_dim=act_dim)

    log_dir = REPO_ROOT / ppoc_cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    pool = ParallelSnakeEnvPool(num_envs=num_envs, config_path=args.config, base_seed=ppoc_cfg.seed)

    global_step = 0
    t_start = time.perf_counter()

    current_options: List[Optional[int]] = [None] * num_envs
    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_switches = np.zeros(num_envs, dtype=np.int64)
    ep_returns_hist: List[float] = []
    ep_success_hist: List[float] = []

    print(
        f"Starting Parallel PPOC Training | device={agent.device} | num_envs={num_envs} | "
        f"total_timesteps={ppoc_cfg.total_timesteps} | Omega={ppoc_cfg.omega_k_f}"
    )

    try:
        while global_step < ppoc_cfg.total_timesteps:
            progress = global_step / max(ppoc_cfg.total_timesteps, 1)
            if args.warmstart_bc:
                stage = "B" if progress < 0.35 else "C"
            elif progress < stage_a_frac:
                stage = "A"
            elif progress < stage_a_frac + stage_b_frac:
                stage = "B"
            else:
                stage = "C"
            pool.set_stage(stage)

            for b in worker_buffers:
                b.reset()

            for _ in range(steps_per_env):
                obs_batch = pool.get_obs()
                prev_options = list(current_options)
                actions, options, intra_lps, master_lps, state_vs, q_vals, switched = (
                    agent.select_action_batch(obs_batch, current_options, deterministic=False)
                )
                k_fs = [agent.get_k_f(opt) for opt in options]
                norm_obs = agent.obs_norm.normalize(obs_batch)

                next_obs_batch, rews, dones, final_xs, reached_list = pool.step(actions, k_fs)
                norm_next_obs = agent.obs_norm.normalize(next_obs_batch)

                for i in range(num_envs):
                    worker_buffers[i].add(
                        obs=norm_obs[i],
                        next_obs=norm_next_obs[i],
                        action=actions[i],
                        option=options[i],
                        prev_option=prev_options[i],
                        switched=float(switched[i]),
                        reward=float(rews[i]),
                        done=bool(dones[i]),
                        intra_logp=float(intra_lps[i]),
                        master_logp=float(master_lps[i]),
                        value=float(state_vs[i]),
                        q_val=float(q_vals[i]),
                    )
                    ep_returns[i] += float(rews[i])
                    ep_switches[i] += int(switched[i])
                    if dones[i]:
                        ep_returns_hist.append(float(ep_returns[i]))
                        ep_success_hist.append(float(reached_list[i]))
                        writer.add_scalar("train/episode_return", ep_returns[i], global_step)
                        writer.add_scalar("train/episode_success", float(reached_list[i]), global_step)
                        writer.add_scalar("train/episode_final_x", final_xs[i], global_step)
                        writer.add_scalar("train/option_switches", ep_switches[i], global_step)
                        ep_returns[i] = 0.0
                        ep_switches[i] = 0
                        current_options[i] = None
                    else:
                        current_options[i] = options[i]

                global_step += num_envs

            # Compute per-worker GAE and merge into single batch buffer
            with torch.no_grad():
                last_norm = agent.obs_norm.normalize(pool.get_obs())
                last_t = torch.as_tensor(last_norm, dtype=torch.float32, device=agent.device)
                last_vals = agent.network.get_state_value(last_t).cpu().numpy()

            merged_buffer.reset()
            for i in range(num_envs):
                worker_buffers[i].compute_returns_and_advantages(
                    last_value=float(last_vals[i]),
                    gamma=ppoc_cfg.gamma,
                    gae_lambda=ppoc_cfg.gae_lambda,
                )
                s = merged_buffer.ptr
                e = s + steps_per_env
                merged_buffer.obs[s:e] = worker_buffers[i].obs[:steps_per_env]
                merged_buffer.next_obs[s:e] = worker_buffers[i].next_obs[:steps_per_env]
                merged_buffer.actions[s:e] = worker_buffers[i].actions[:steps_per_env]
                merged_buffer.options[s:e] = worker_buffers[i].options[:steps_per_env]
                merged_buffer.prev_options[s:e] = worker_buffers[i].prev_options[:steps_per_env]
                merged_buffer.has_prev_option[s:e] = worker_buffers[i].has_prev_option[:steps_per_env]
                merged_buffer.switched[s:e] = worker_buffers[i].switched[:steps_per_env]
                merged_buffer.rewards[s:e] = worker_buffers[i].rewards[:steps_per_env]
                merged_buffer.dones[s:e] = worker_buffers[i].dones[:steps_per_env]
                merged_buffer.intra_logp[s:e] = worker_buffers[i].intra_logp[:steps_per_env]
                merged_buffer.master_logp[s:e] = worker_buffers[i].master_logp[:steps_per_env]
                merged_buffer.values[s:e] = worker_buffers[i].values[:steps_per_env]
                merged_buffer.q_values[s:e] = worker_buffers[i].q_values[:steps_per_env]
                merged_buffer.advantages_intra[s:e] = worker_buffers[i].advantages_intra[:steps_per_env]
                merged_buffer.advantages_master[s:e] = worker_buffers[i].advantages_master[:steps_per_env]
                merged_buffer.returns[s:e] = worker_buffers[i].returns[:steps_per_env]
                merged_buffer.ptr = e

            losses = agent.update(merged_buffer)
            for k, v in losses.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

            rollout_idx = global_step // total_rollout_steps
            if rollout_idx % 1 == 0 or global_step >= ppoc_cfg.total_timesteps:
                eval_res = evaluate_policy(agent, stage="C", fixed_tile_width=0.30, num_episodes=5)
                for k, v in eval_res.items():
                    writer.add_scalar(f"eval/{k}", v, global_step)

                elapsed = time.perf_counter() - t_start
                fps = global_step / max(elapsed, 1e-6)
                mean_ret = float(np.mean(ep_returns_hist[-15:])) if ep_returns_hist else eval_res["eval_return"]
                mean_succ = float(np.mean(ep_success_hist[-15:])) if ep_success_hist else eval_res["eval_success"]

                print(
                    f"[Step {global_step:6d}/{ppoc_cfg.total_timesteps} | Stage {stage} | {fps:4.0f} sps] "
                    f"TrainRet={mean_ret:6.1f} Succ={mean_succ*100:4.0f}% | "
                    f"Eval(L/2 tile): Ret={eval_res['eval_return']:6.1f} "
                    f"X={eval_res['eval_final_x']:.2f}m Succ={eval_res['eval_success']*100:4.0f}% "
                    f"Kf={eval_res['eval_mean_kf']:.2f} Vmax={eval_res['eval_max_v']:.0f}V",
                    flush=True,
                )

                score = eval_res["eval_return"] + 250.0 * eval_res["eval_success"]
                if score > best_eval_score:
                    best_eval_score = score
                    agent.save(ckpt_dir / "ppoc_best.pt")

        agent.save(ckpt_dir / "ppoc_final.pt")
        print(f"Training complete in {time.perf_counter() - t_start:.1f}s. Best checkpoint saved to {ckpt_dir / 'ppoc_best.pt'}")
    finally:
        pool.close()
        writer.close()


if __name__ == "__main__":
    main()
