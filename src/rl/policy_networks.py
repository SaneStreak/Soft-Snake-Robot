"""Neural network architectures for Proximal Policy Option-Critic (PPOC).

Includes:
- OptionCriticActorCritic: Shared/modular MLP networks for
  1) Master option policy P(omega | s) over discrete K_f frequencies
  2) Intra-option continuous Gaussian actor pi_omega(alpha | s) in [-2, 2]^12
  3) Option-value critic Q_Omega(s, omega)
  4) Option termination network beta_omega(s) in [0, 1]
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> Tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.append(_ortho_init(nn.Linear(in_dim, h)))
        layers.append(nn.Tanh())
        in_dim = h
    return nn.Sequential(*layers), in_dim


class OptionCriticNetwork(nn.Module):
    """Multi-rate Option-Critic network for hierarchical CPG regulation."""

    def __init__(
        self,
        obs_dim: int = 28,
        action_dim: int = 12,
        num_options: int = 5,
        hidden_sizes: Sequence[int] = (128, 128),
        action_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.action_scale = float(action_scale)

        # Separate intra-option steering trunk from master option trunk so rare switch
        # gradients never destabilize continuous steering features.
        self.actor_trunk, actor_feat_dim = _build_mlp(obs_dim, hidden_sizes)
        self.intra_mean_head = _ortho_init(
            nn.Linear(actor_feat_dim + num_options, action_dim), gain=0.01
        )
        self.intra_log_std = nn.Parameter(torch.full((num_options, action_dim), -0.7))

        master_hidden = (64, 64)
        self.master_trunk, master_feat_dim = _build_mlp(obs_dim, master_hidden)
        self.master_head = _ortho_init(nn.Linear(master_feat_dim, num_options), gain=0.01)

        # Critic & termination trunk -> Q_Omega(s, omega) and beta_omega(s)
        self.critic_trunk, critic_feat_dim = _build_mlp(obs_dim, hidden_sizes)
        self.q_head = _ortho_init(nn.Linear(critic_feat_dim, num_options), gain=1.0)
        self.term_head = _ortho_init(nn.Linear(critic_feat_dim, num_options), gain=0.01)
        # Initialize termination logits around -1.5 (beta ~ 0.18) for responsive multi-step persistence
        nn.init.constant_(self.term_head.bias, -1.5)

    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q_Omega(s, omega) of shape (B, M)."""
        feat = self.critic_trunk(obs)
        return self.q_head(feat)

    def get_termination_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute option termination probabilities beta_omega(s) in (0, 1) of shape (B, M)."""
        feat = self.critic_trunk(obs)
        return torch.sigmoid(self.term_head(feat))

    def get_master_dist(self, obs: torch.Tensor) -> Categorical:
        """Return categorical distribution P(omega | s) over macro-options."""
        feat = self.master_trunk(obs)
        logits = self.master_head(feat)
        return Categorical(logits=logits)

    def get_state_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute expected state value V_Omega(s) = sum_omega P(omega|s) Q_Omega(s, omega)."""
        q_vals = self.get_q_values(obs)
        master_probs = self.get_master_dist(obs).probs
        return torch.sum(master_probs * q_vals, dim=-1)

    def get_intra_dist(self, obs: torch.Tensor, option: torch.Tensor) -> Normal:
        """Return Gaussian policy pi_omega(alpha | s) for selected option indices."""
        feat = self.actor_trunk(obs)
        opt_oh = torch.nn.functional.one_hot(option, num_classes=self.num_options).to(feat.dtype)
        mean = self.action_scale * torch.tanh(self.intra_mean_head(torch.cat([feat, opt_oh], dim=-1)))
        log_std = torch.clamp(self.intra_log_std[option], -3.0, 0.8)
        std = torch.exp(log_std)
        return Normal(mean, std)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        current_option: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, int, float, float, float, float, float]:
        """Select option (or evaluate termination) and continuous intra-option steering action.

        Returns:
            alpha: numpy array (action_dim,) clipped to [-action_scale, action_scale]
            option: active option index in {0, ..., M-1}
            intra_logp: scalar log pi_omega(alpha | s)
            master_logp: scalar log P(option | s)
            q_val: Q_Omega(s, option)
            beta_val: termination prob beta_prev(s) (or beta_curr(s) if no prev option)
            terminated: 1.0 if option switched/initialized else 0.0
        """
        q_vals = self.get_q_values(obs)
        master_dist = self.get_master_dist(obs)
        betas = self.get_termination_probs(obs)

        switched = 0.0
        if current_option is None:
            option_t = (
                torch.argmax(master_dist.probs, dim=-1)
                if deterministic
                else master_dist.sample()
            )
            option = int(option_t.item())
            beta_val = float(betas[0, option].item())
            switched = 1.0
        else:
            beta_val = float(betas[0, current_option].item())
            should_term = (beta_val > 0.5) if deterministic else (np.random.rand() < beta_val)
            if should_term:
                option_t = (
                    torch.argmax(master_dist.probs, dim=-1)
                    if deterministic
                    else master_dist.sample()
                )
                option = int(option_t.item())
                switched = 1.0
            else:
                option = int(current_option)
                option_t = torch.tensor([option], device=obs.device, dtype=torch.long)

        intra_dist = self.get_intra_dist(obs, option_t)
        raw_act = intra_dist.mean if deterministic else intra_dist.sample()
        clipped_act = torch.clamp(raw_act, -self.action_scale, self.action_scale)
        intra_logp = float(intra_dist.log_prob(clipped_act).sum(dim=-1).item())
        master_logp = float(master_dist.log_prob(option_t).item())
        q_val = float(q_vals[0, option].item())

        return (
            clipped_act.squeeze(0).cpu().numpy(),
            option,
            intra_logp,
            master_logp,
            q_val,
            beta_val,
            switched,
        )

