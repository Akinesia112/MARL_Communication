#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# use EGL headless rendering before importing mujoco/gym
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np
import torch
from gymnasium_robotics import mamujoco_v1
import imageio.v2 as imageio
import torch.nn as nn

from baseline_no_comm import SimpleCritic  # not really needed here, but OK


# ---- NCA Comm + Policy (same as in training, minimal copy) ----

class NCAComm1D(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        grid_size: int = 32,
        channels: int = 16,
        n_steps: int = 2,
        dt: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.channels = channels
        self.n_steps = n_steps
        self.dt = dt

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, channels),
        )
        self.nca_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        positions = torch.linspace(0, grid_size - 1, n_agents).long()
        self.register_buffer("agent_pos", positions)

    def forward(self, all_obs, state):
        B, N, D = all_obs.shape
        encoded = self.obs_encoder(all_obs)  # [B,N,C]

        x = state
        deposit = torch.zeros_like(x)
        pos = self.agent_pos.unsqueeze(0).unsqueeze(-1).expand(B, N, self.channels)
        deposit = deposit.scatter_add(1, pos, encoded)
        x = x + deposit

        x_perm = x.permute(0, 2, 1)
        for _ in range(self.n_steps):
            dx = torch.tanh(self.nca_conv(x_perm))
            x_perm = x_perm + self.dt * dx
        x = x_perm.permute(0, 2, 1)

        pos = self.agent_pos.unsqueeze(0).unsqueeze(-1).expand(B, N, self.channels)
        comm_feats = torch.gather(x, 1, pos)  # [B,N,C]

        return comm_feats, x


class NCAPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, comm_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, comm_feat):
        from torch.distributions import Normal
        x = torch.cat([obs, comm_feat], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)


# ---- Evaluation ----

def load_nca_comm_policies(
    ckpt_path="checkpoints_nca_heat1d_comm/nca_heat1d_comm_best.pt",
    scenario="Ant",
    agent_conf="2x4",
    nca_grid_size=32,
    nca_channels=16,
    nca_steps=2,
    nca_dt=1.0,
):
    env = mamujoco_v1.parallel_env(
        scenario=scenario,
        agent_conf=agent_conf,
        agent_obsk=1,
        render_mode="rgb_array",
        max_episode_steps=500,
    )

    agents = env.agents
    n_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comm = NCAComm1D(
        obs_dim=obs_dim,
        n_agents=n_agents,
        grid_size=nca_grid_size,
        channels=nca_channels,
        n_steps=nca_steps,
        dt=nca_dt,
    ).to(device)

    policies = {
        agent: NCAPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            comm_dim=nca_channels,
            hidden_dim=256,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(ckpt_path, map_location=device)
    comm.load_state_dict(ckpt["comm"])
    for agent in agents:
        policies[agent].load_state_dict(ckpt["policies"][agent])
        policies[agent].eval()
    comm.eval()

    print(f"✓ Loaded NCA-Comm PPO model from {ckpt_path}")

    return env, agents, policies, comm, device


@torch.no_grad()
def eval_nca_comm_5min(
    ckpt_path="checkpoints_nca_heat1d_comm/nca_heat1d_comm_best.pt",
    out_dir="videos",
    out_name="nca_heat1d_comm_5min.mp4",
    scenario="Ant",
    agent_conf="2x4",
    nca_grid_size=32,
    nca_channels=16,
    nca_steps=2,
    nca_dt=1.0,
    fps=30,
    target_seconds=300,
):
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, out_name)

    env, agents, policies, comm, device = load_nca_comm_policies(
        ckpt_path=ckpt_path,
        scenario=scenario,
        agent_conf=agent_conf,
        nca_grid_size=nca_grid_size,
        nca_channels=nca_channels,
        nca_steps=nca_steps,
        nca_dt=nca_dt,
    )

    act_space = env.action_space(agents[0])
    action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
    max_action = torch.max(torch.abs(action_high))

    n_frames_target = int(target_seconds * fps)
    frames = []
    ep_idx = 0

    agent_idx = {agent: idx for idx, agent in enumerate(agents)}

    print(f"=== Recording NCA-Comm PPO for {target_seconds} seconds ≈ {n_frames_target} frames ===")

    while len(frames) < n_frames_target:
        ep_idx += 1
        obs_dict, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agents}

        print(f"--- Episode {ep_idx} ---")

        done = False
        step_in_ep = 0

        while not done and len(frames) < n_frames_target:
            all_obs_np = np.array([obs_dict[a] for a in agents], dtype=np.float32)
            all_obs = torch.from_numpy(all_obs_np).unsqueeze(0).to(device)  # [1,N,D]

            B = all_obs.size(0)
            nca_state = torch.zeros(
                B, nca_grid_size, nca_channels, device=device
            )
            comm_all, _ = comm(all_obs, nca_state)  # [1,N,C]

            actions_dict = {}
            for agent in agents:
                idx = agent_idx[agent]
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(device)
                comm_feat = comm_all[0, idx, :].unsqueeze(0)  # [1,C]
                dist = policies[agent](obs, comm_feat)
                mu = dist.mean
                action = torch.tanh(mu) * max_action
                actions_dict[agent] = action.cpu().numpy()[0]

            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            for agent in agents:
                episode_reward[agent] += float(rewards[agent])

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            step_in_ep += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            else:
                obs_dict = next_obs

        avg_rew = np.mean(list(episode_reward.values()))
        print(f"Episode {ep_idx}: avg reward={avg_rew:.2f}, frames so far={len(frames)}/{n_frames_target}")

    frames = frames[:n_frames_target]
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"✓ Saved video: {video_path}")

    env.close()


if __name__ == "__main__":
    eval_nca_comm_5min(
        ckpt_path="checkpoints_nca_heat1d_comm/nca_heat1d_comm_best.pt",
        out_dir="videos",
        out_name="nca_heat1d_comm_5min.mp4",
        scenario="Ant",
        agent_conf="2x4",
        nca_grid_size=32,
        nca_channels=16,
        nca_steps=2,
        nca_dt=1.0,
        fps=30,
        target_seconds=300,
    )
