"""Integration tests for SnakeGymEnv, PPOCAgent, and 3D SOFA TCP IPC connector."""

from pathlib import Path

import torch  # noqa: F401
import numpy as np

from src.envs.snake_gym_env import SnakeGymEnv
from src.envs.sofa_connector import SofaConnector, SofaIPCConfig, SofaLocalFEMBridge
from src.rl.option_critic_ppoc import PPOCAgent, PPOCConfig, PPOCRolloutBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_snake_gym_env_api_and_safety() -> None:
    """Verify 28-D state space, grouped action space, and zero DEA breakdown events."""
    env = SnakeGymEnv(curriculum_stage="C", fixed_tile_width=0.30)
    obs, info = env.reset(seed=11)
    assert obs.shape == (28,)
    assert env.action_space.shape in ((4,), (12,))
    assert np.all(np.isfinite(obs))
    assert "d_goal" in info

    for step_idx in range(100):
        action = env.action_space.sample()
        k_f = env.omega_k_f[step_idx % len(env.omega_k_f)]
        next_obs, reward, term, trunc, step_info = env.step(action, k_f=k_f)
        assert next_obs.shape == (28,)
        assert np.all(np.isfinite(next_obs))
        assert np.isfinite(reward)
        assert step_info["max_voltage_v"] <= 450.0 + 1e-9
        assert step_info["breakdown_events"] == 0.0
        if term or trunc:
            break


def test_ppoc_agent_rollout_update_and_checkpoint(tmp_path: Path) -> None:
    """Verify PPOC multi-rate action selection, gradient update, and save/load round-trip."""
    env = SnakeGymEnv(curriculum_stage="A")
    act_dim = env.action_space.shape[0]
    cfg = PPOCConfig(rollout_steps=64, minibatch_size=32, num_epochs=2)
    agent = PPOCAgent(obs_dim=28, action_dim=act_dim, config=cfg)
    buf = PPOCRolloutBuffer(capacity=64, obs_dim=28, action_dim=act_dim)

    obs, _ = env.reset(seed=7)
    opt = None
    for _ in range(64):
        prev_opt = opt
        alpha, opt, intra_lp, master_lp, val, q_val, _, switched = agent.select_action(
            obs, current_option=opt, deterministic=False
        )
        norm_obs = agent.obs_norm.normalize(obs)
        next_obs, rew, term, trunc, _ = env.step(alpha, k_f=agent.get_k_f(opt))
        done = term or trunc
        buf.add(
            obs=norm_obs,
            next_obs=agent.obs_norm.normalize(next_obs),
            action=alpha,
            option=opt,
            prev_option=prev_opt,
            switched=switched,
            reward=rew,
            done=done,
            intra_logp=intra_lp,
            master_logp=master_lp,
            value=val,
            q_val=q_val,
        )
        obs = next_obs
        if done:
            obs, _ = env.reset()
            opt = None

    buf.compute_returns_and_advantages(last_value=0.0, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    metrics = agent.update(buf)
    assert all(np.isfinite(v) for v in metrics.values())

    ckpt_path = tmp_path / "test_ppoc.pt"
    agent.save(ckpt_path)
    restored = PPOCAgent.load(ckpt_path)
    a1, o1, *_ = agent.select_action(obs, deterministic=True, update_norm=False)
    a2, o2, *_ = restored.select_action(obs, deterministic=True, update_norm=False)
    assert o1 == o2
    assert np.allclose(a1, a2, atol=1e-6)


def test_sofa_ipc_bridge_roundtrip() -> None:
    """Verify TCP socket IPC streaming 24 DEA voltages and reading hexahedral FEM telemetry."""
    bridge = SofaLocalFEMBridge(REPO_ROOT / "configs" / "plant_rev_c.yaml", port=0)
    port = bridge.start()
    conn = SofaConnector(SofaIPCConfig(host="127.0.0.1", port=port))
    try:
        conn.connect()
        init_tel = conn.reset(sand_eta=3.0, tile_patches=[])
        assert init_tel["x_cm"] == 0.0
        assert np.asarray(init_tel["hex_nodes"]).shape == (12, 3)

        v_24 = np.full((12, 2), 300.0)
        v_24[:, 0] = 400.0
        step_tel = conn.step(v_24, dt=0.005)
        assert step_tel["kappa"].shape == (12,)
        assert step_tel["capacitance_pf"].shape == (12, 2)
        assert step_tel["hex_nodes"].shape == (12, 3)
        assert np.all(np.isfinite(step_tel["kappa"]))
    finally:
        conn.close()
        bridge.stop()

