"""Multi-rate Proximal Policy Option-Critic (PPOC) algorithm implementation.

Implements option-value learning, clipped PPO intra-option policy updates,
clipped PPO master option policy updates, and Bacon et al. (2017) termination
gradients with deliberation cost for temporal abstraction over K_f.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.rl.policy_networks import OptionCriticNetwork


@dataclass
class PPOCConfig:
    """Hyperparameters for Option-Critic PPOC training."""

    seed: int = 42
    total_timesteps: int = 81920
    rollout_steps: int = 2048
    num_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.975
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    target_kl: float = 0.025
    entropy_coef_intra: float = 0.001
    entropy_coef_master: float = 0.01
    termination_reg: float = 0.02
    lr_actor: float = 3.0e-4
    lr_critic: float = 5.0e-4
    lr_termination: float = 3.0e-4
    max_grad_norm: float = 0.5
    hidden_sizes: Tuple[int, ...] = (128, 128)
    omega_k_f: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.3, 1.6)
    log_dir: str = "runs/ppoc_soft_snake"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PPOCConfig":
        """Load PPOC hyperparameters from ppoc_training.yaml."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        ppoc_sec = dict(data.get("ppoc", {}))
        env_sec = data.get("env", {})
        if "omega_k_f" in env_sec:
            ppoc_sec["omega_k_f"] = tuple(env_sec["omega_k_f"])
        if "hidden_sizes" in ppoc_sec:
            ppoc_sec["hidden_sizes"] = tuple(ppoc_sec["hidden_sizes"])
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in ppoc_sec.items() if k in valid_fields}
        return cls(**filtered)


class FixedPhysicalObsNorm:
    """Stationary physical scale normalizer for 28-D state vectors.

    Prevents non-stationary running-mean/std shifts across curriculum stages A->B->C.
    State layout: [theta_err, d_goal, v_fwd, v_lat, kappa (12), kappa_dot (12)].
    """

    def __init__(self, obs_dim: int = 28, clip: float = 10.0) -> None:
        n_mod = (obs_dim - 4) // 2
        scales = np.concatenate(
            [
                np.array([0.5, 2.0, 0.5, 0.5], dtype=np.float64),
                np.full(n_mod, 12.0, dtype=np.float64),
                np.full(n_mod, 60.0, dtype=np.float64),
            ]
        )
        self.scale = scales
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        """No-op for stationary physical scaling."""
        return None

    def normalize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        return np.clip(arr / self.scale, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"scale": self.scale, "clip": self.clip}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "scale" in state:
            self.scale = np.asarray(state["scale"], dtype=np.float64)
            self.clip = float(state.get("clip", 10.0))


class PPOCRolloutBuffer:
    """On-policy rollout buffer storing intra-option & macro-option trajectories."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.options = np.zeros(capacity, dtype=np.int64)
        self.prev_options = np.zeros(capacity, dtype=np.int64)
        self.has_prev_option = np.zeros(capacity, dtype=np.float32)
        self.switched = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.intra_logp = np.zeros(capacity, dtype=np.float32)
        self.master_logp = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.q_values = np.zeros(capacity, dtype=np.float32)

        self.advantages_intra = np.zeros(capacity, dtype=np.float32)
        self.advantages_master = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        option: int,
        prev_option: Optional[int],
        switched: float,
        reward: float,
        done: bool,
        intra_logp: float,
        master_logp: float,
        value: float,
        q_val: float,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.options[idx] = option
        self.prev_options[idx] = option if prev_option is None else prev_option
        self.has_prev_option[idx] = 0.0 if prev_option is None else 1.0
        self.switched[idx] = switched
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.intra_logp[idx] = intra_logp
        self.master_logp[idx] = master_logp
        self.values[idx] = value
        self.q_values[idx] = q_val
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE returns and separate intra-option / master-option advantages."""
        n = self.ptr
        gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 1.0 - self.dones[t]
            next_v = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_v * non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.returns[t] = gae + self.values[t]
            self.advantages_master[t] = gae
            self.advantages_intra[t] = gae

    def reset(self) -> None:
        self.ptr = 0


class PPOCAgent:
    """Proximal Policy Option-Critic (PPOC) agent for hierarchical soft-snake control."""

    def __init__(
        self,
        obs_dim: int = 28,
        action_dim: int = 12,
        config: Optional[PPOCConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PPOCConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.omega_k_f = tuple(self.config.omega_k_f)
        self.num_options = len(self.omega_k_f)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.network = OptionCriticNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_options=self.num_options,
            hidden_sizes=self.config.hidden_sizes,
            action_scale=2.0,
        ).to(self.device)
        self.obs_norm = FixedPhysicalObsNorm(obs_dim=obs_dim)

        actor_params = (
            list(self.network.actor_trunk.parameters())
            + list(self.network.intra_mean_head.parameters())
            + [self.network.intra_log_std]
            + list(self.network.master_trunk.parameters())
            + list(self.network.master_head.parameters())
        )
        critic_params = list(self.network.critic_trunk.parameters()) + list(
            self.network.q_head.parameters()
        )
        term_params = list(self.network.term_head.parameters())

        self.opt_actor = torch.optim.Adam(actor_params, lr=self.config.lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(critic_params, lr=self.config.lr_critic, eps=1e-5)
        self.opt_term = torch.optim.Adam(term_params, lr=self.config.lr_termination, eps=1e-5)

    def select_action(
        self,
        obs: np.ndarray,
        current_option: Optional[int] = None,
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Tuple[np.ndarray, int, float, float, float, float, float, float]:
        """Select option, frequency gain K_f, and continuous steering action alpha."""
        if update_norm:
            self.obs_norm.update(obs)
        norm_obs = self.obs_norm.normalize(obs)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        alpha, option, intra_logp, master_logp, q_val, beta_val, switched = self.network.act(
            obs_t, current_option=current_option, deterministic=deterministic
        )
        with torch.no_grad():
            state_val = float(self.network.get_state_value(obs_t).item())
        return alpha, option, intra_logp, master_logp, state_val, q_val, beta_val, switched

    def get_k_f(self, option_idx: int) -> float:
        """Map discrete option index to frequency scaling factor K_f."""
        return float(self.omega_k_f[option_idx])

    @torch.no_grad()
    def select_action_batch(
        self,
        obs_batch: np.ndarray,
        current_options: list[Optional[int]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched multi-environment option & steering action selection."""
        norm_obs = self.obs_norm.normalize(obs_batch)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
        b = obs_t.shape[0]

        q_vals = self.network.get_q_values(obs_t)
        master_dist = self.network.get_master_dist(obs_t)
        betas = self.network.get_termination_probs(obs_t)
        state_vals = torch.sum(master_dist.probs * q_vals, dim=-1)

        options: list[int] = []
        switched = np.zeros(b, dtype=np.float32)
        sampled_opts = (
            torch.argmax(master_dist.probs, dim=-1) if deterministic else master_dist.sample()
        )
        for i, prev_opt in enumerate(current_options):
            if prev_opt is None:
                options.append(int(sampled_opts[i].item()))
                switched[i] = 1.0
            else:
                beta_val = float(betas[i, prev_opt].item())
                do_term = (beta_val > 0.5) if deterministic else (np.random.rand() < beta_val)
                if do_term:
                    options.append(int(sampled_opts[i].item()))
                    switched[i] = 1.0
                else:
                    options.append(int(prev_opt))

        opt_t = torch.tensor(options, dtype=torch.long, device=self.device)
        intra_dist = self.network.get_intra_dist(obs_t, opt_t)
        raw_act = intra_dist.mean if deterministic else intra_dist.sample()
        clipped_act = torch.clamp(raw_act, -2.0, 2.0)

        intra_lp = intra_dist.log_prob(clipped_act).sum(dim=-1).cpu().numpy()
        master_lp = master_dist.log_prob(opt_t).cpu().numpy()
        q_sel = q_vals[torch.arange(b, device=self.device), opt_t].cpu().numpy()
        v_np = state_vals.cpu().numpy()

        return (
            clipped_act.cpu().numpy(),
            options,
            intra_lp,
            master_lp,
            v_np,
            q_sel,
            switched,
        )

    def update(self, buffer: PPOCRolloutBuffer) -> Dict[str, float]:
        """Perform multi-epoch PPOC updates for intra-actor, master-policy, critic, and termination."""
        n = buffer.ptr
        cfg = self.config

        obs_t = torch.as_tensor(buffer.obs[:n], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(buffer.next_obs[:n], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(buffer.actions[:n], dtype=torch.float32, device=self.device)
        opt_t = torch.as_tensor(buffer.options[:n], dtype=torch.long, device=self.device)
        prev_opt_t = torch.as_tensor(buffer.prev_options[:n], dtype=torch.long, device=self.device)
        has_prev_t = torch.as_tensor(buffer.has_prev_option[:n], dtype=torch.float32, device=self.device)
        switched_t = torch.as_tensor(buffer.switched[:n], dtype=torch.float32, device=self.device)
        old_intra_logp_t = torch.as_tensor(buffer.intra_logp[:n], dtype=torch.float32, device=self.device)
        old_master_logp_t = torch.as_tensor(buffer.master_logp[:n], dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(buffer.returns[:n], dtype=torch.float32, device=self.device)

        adv_intra = torch.as_tensor(buffer.advantages_intra[:n], dtype=torch.float32, device=self.device)
        adv_master = torch.as_tensor(buffer.advantages_master[:n], dtype=torch.float32, device=self.device)
        adv_intra = (adv_intra - adv_intra.mean()) / (adv_intra.std() + 1e-8)
        adv_master = (adv_master - adv_master.mean()) / (adv_master.std() + 1e-8)

        metrics: Dict[str, List[float]] = {
            "loss_intra": [],
            "loss_master": [],
            "loss_critic": [],
            "loss_term": [],
            "entropy_intra": [],
            "entropy_master": [],
        }

        indices = np.arange(n)
        kl_exceeded = False
        for _ in range(cfg.num_epochs):
            if kl_exceeded:
                break
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start : start + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                b_obs = obs_t[mb_t]
                b_next_obs = next_obs_t[mb_t]
                b_act = act_t[mb_t]
                b_opt = opt_t[mb_t]
                b_prev_opt = prev_opt_t[mb_t]
                b_has_prev = has_prev_t[mb_t]
                b_switched = switched_t[mb_t]
                b_old_intra_lp = old_intra_logp_t[mb_t]
                b_old_master_lp = old_master_logp_t[mb_t]
                b_ret = returns_t[mb_t]
                b_adv_intra = adv_intra[mb_t]
                b_adv_master = adv_master[mb_t]

                # 1. Option-value critic loss
                q_all = self.network.get_q_values(b_obs)
                q_sel = q_all.gather(1, b_opt.unsqueeze(1)).squeeze(1)
                loss_critic = 0.5 * nn.functional.mse_loss(q_sel, b_ret)

                self.opt_critic.zero_grad()
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()

                # 2. Intra-option & master PPO losses
                intra_dist = self.network.get_intra_dist(b_obs, b_opt)
                new_intra_lp = intra_dist.log_prob(b_act).sum(dim=-1)
                ratio_intra = torch.exp(new_intra_lp - b_old_intra_lp)
                surr1_i = ratio_intra * b_adv_intra
                surr2_i = torch.clamp(ratio_intra, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv_intra
                ent_intra = intra_dist.entropy().sum(dim=-1).mean()
                loss_intra = -torch.min(surr1_i, surr2_i).mean() - cfg.entropy_coef_intra * ent_intra

                master_dist = self.network.get_master_dist(b_obs)
                new_master_lp = master_dist.log_prob(b_opt)
                ratio_master = torch.exp(new_master_lp - b_old_master_lp)
                surr1_m = ratio_master * b_adv_master
                surr2_m = torch.clamp(ratio_master, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv_master
                ent_master = master_dist.entropy().mean()
                switched_sum = torch.clamp(b_switched.sum(), min=1.0)
                loss_master = -(
                    (torch.min(surr1_m, surr2_m) * b_switched).sum() / switched_sum
                ) - cfg.entropy_coef_master * ent_master

                loss_actor_total = loss_intra + loss_master
                self.opt_actor.zero_grad()
                loss_actor_total.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.opt_actor.step()

                # 3. Termination network loss (Bacon et al., 2017) with scale-invariant Q adv
                with torch.no_grad():
                    q_obs_all = self.network.get_q_values(b_obs)
                    v_obs = self.network.get_state_value(b_obs)
                    q_prev = q_obs_all.gather(1, b_prev_opt.unsqueeze(1)).squeeze(1)
                    q_scale = torch.clamp(v_obs.abs().mean(), min=1.0)
                    term_adv = (q_prev - v_obs) / q_scale + cfg.termination_reg

                betas = self.network.get_termination_probs(b_obs)
                beta_prev = betas.gather(1, b_prev_opt.unsqueeze(1)).squeeze(1)
                loss_term = (beta_prev * term_adv * b_has_prev).sum() / torch.clamp(
                    b_has_prev.sum(), min=1.0
                )

                self.opt_term.zero_grad()
                loss_term.backward()
                nn.utils.clip_grad_norm_(self.network.term_head.parameters(), cfg.max_grad_norm)
                self.opt_term.step()

                metrics["loss_intra"].append(float(loss_intra.item()))
                metrics["loss_master"].append(float(loss_master.item()))
                metrics["loss_critic"].append(float(loss_critic.item()))
                metrics["loss_term"].append(float(loss_term.item()))
                metrics["entropy_intra"].append(float(ent_intra.item()))
                metrics["entropy_master"].append(float(ent_master.item()))

                with torch.no_grad():
                    log_ratio = new_intra_lp - b_old_intra_lp
                    approx_kl = float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item())
                if cfg.target_kl > 0 and approx_kl > 1.5 * cfg.target_kl:
                    kl_exceeded = True
                    break

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: Union[str, Path]) -> None:
        """Persist PPOC weights, config, and running observation normalizer."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "obs_norm": self.obs_norm.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            out_path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "PPOCAgent":
        """Restore trained PPOC agent from checkpoint."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(
            obs_dim=int(ckpt["obs_dim"]),
            action_dim=int(ckpt["action_dim"]),
            config=ckpt["config"],
            device=device,
        )
        agent.network.load_state_dict(ckpt["network"])
        agent.obs_norm.load_state_dict(ckpt["obs_norm"])
        agent.network.eval()
        return agent
