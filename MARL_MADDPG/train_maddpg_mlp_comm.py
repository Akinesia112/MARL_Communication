#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emergent-Communication MADDPG on MAMuJoCo Ant (2x4) - MLP version (NO attention)

- Actor: CommPolicyMLP (continuous action + discrete msg head, no attention)
- Critic: CentralizedCritic Q_i(s, a_1..N)
- Comm channel:
    * msg_len: number of tokens per step
    * vocab_size: discrete vocabulary size
    * comm_tau: Gumbel temperature
    * comm_hard: straight-through one-hot or soft
- Comm cost: entropy-based penalty (encourage less noisy messages)

Logs:
    - Saves checkpoints under:    checkpoints_maddpg_comm/
    - Saves training curves to:   maddpg_comm_logs.npz
      keys: episode, avg_reward, max_reward, buffer_size, sigma, actor_loss, critic_loss
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium_robotics import mamujoco_v1


# -----------------------
# Replay Buffer
# -----------------------

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.memory = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, dones):
        """
        obs:      [n_agents, obs_dim]
        actions:  [n_agents, action_dim]
        rewards:  [n_agents]
        next_obs: [n_agents, obs_dim]
        dones:    [n_agents]
        """
        self.memory.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = np.array(obs, dtype=np.float32)            # [B, n_agents, obs_dim]
        actions = np.array(actions, dtype=np.float32)    # [B, n_agents, action_dim]
        rewards = np.array(rewards, dtype=np.float32)    # [B, n_agents]
        next_obs = np.array(next_obs, dtype=np.float32)  # [B, n_agents, obs_dim]
        dones = np.array(dones, dtype=np.float32)        # [B, n_agents]

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.memory)


# -----------------------
# Centralized Critic
# -----------------------

class CentralizedCritic(nn.Module):
    """
    Q_i(s, a_1, ..., a_N)
    s: concat of all obs
    a: concat of all actions
    """
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=256):
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs, all_actions):
        """
        all_obs:     [B, n_agents, obs_dim]
        all_actions: [B, n_agents, action_dim]
        """
        x = torch.cat(
            [all_obs.reshape(all_obs.size(0), -1),
             all_actions.reshape(all_actions.size(0), -1)],
            dim=-1,
        )
        q = self.net(x)
        return q.squeeze(-1)  # [B]


# -----------------------
# Policy: MLP + Emergent Comm head (NO attention)
# -----------------------

class CommPolicyMLP(nn.Module):
    """
    純 MLP actor:
      - obs -> hidden
      - hidden -> action mean
      - hidden -> msg_logits (for msg_len × vocab_size)
    注意：這裡沒有任何 attention，all_obs/self_index 只是為了介面相容，會被忽略。
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=256,
        msg_len=4,
        vocab_size=8,
        comm_tau=1.0,
        comm_hard=True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.msg_len = msg_len
        self.vocab_size = vocab_size
        self.comm_tau = comm_tau
        self.comm_hard = comm_hard

        # Encoder for observation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Continuous action head (Gaussian policy)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Discrete message head: produces logits for msg_len * vocab_size
        self.msg_head = nn.Linear(hidden_dim, msg_len * vocab_size)

    def _gumbel_softmax(self, logits):
        """
        logits: [B, L, V]
        return:
            probs: [B, L, V] (soft or straight-through one-hot)
        """
        # Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = torch.softmax((logits + gumbel_noise) / self.comm_tau, dim=-1)  # [B, L, V]

        if self.comm_hard:
            # Straight-through one-hot
            idx = y.argmax(dim=-1, keepdim=True)  # [B, L, 1]
            y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
            y = (y_hard - y).detach() + y

        return y

    def forward(self, obs, all_obs=None, self_index=None):
        """
        Args:
            obs: [B, obs_dim] - 自己的觀察
            all_obs, self_index: 為了和原本 attention 版介面相容，這裡不使用

        Returns:
            dist: Normal(mean, std) 動作分佈
            comm_info: dict {
                "msg_probs": [B, msg_len, vocab_size],
                "msg_logits": [B, msg_len, vocab_size]
            }
        """
        from torch.distributions import Normal

        h = self.encoder(obs)  # [B, hidden_dim]

        # Action distribution
        mean = self.mean_head(h)  # [B, action_dim]
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        # Message logits
        msg_logits = self.msg_head(h)  # [B, msg_len * vocab_size]
        msg_logits = msg_logits.view(-1, self.msg_len, self.vocab_size)  # [B, L, V]

        msg_probs = self._gumbel_softmax(msg_logits)  # [B, L, V]

        comm_info = {
            "msg_probs": msg_probs,
            "msg_logits": msg_logits,
        }

        return dist, comm_info


# -----------------------
# MADDPG + Emergent Comm (no attention)
# -----------------------

class MADDPGCommTrainer:
    def __init__(
        self,
        scenario="Ant",
        agent_conf="2x4",
        gamma=0.99,
        tau=0.01,
        actor_lr=1e-4,
        critic_lr=1e-3,
        batch_size=256,
        buffer_capacity=int(1e6),
        exploration_sigma=0.2,
        exploration_sigma_min=0.05,
        exploration_decay=1e-6,
        max_episode_steps=500,
        msg_len=4,
        vocab_size=8,
        comm_tau=1.0,
        comm_hard=True,
        lambda_comm=1e-3,  # comm cost weight
    ):
        # Env
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None,
            max_episode_steps=max_episode_steps,
        )

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.max_episode_steps = max_episode_steps

        # Obs / Action dimensions
        example_agent = self.agents[0]
        self.obs_dim = self.env.observation_space(example_agent).shape[0]
        self.action_dim = self.env.action_space(example_agent).shape[0]

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lambda_comm = lambda_comm

        # Action bounds
        act_space = self.env.action_space(example_agent)
        self.action_low = torch.tensor(act_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(act_space.high, dtype=torch.float32)
        # assume symmetric -> use max magnitude
        self.max_action = torch.max(torch.abs(self.action_high))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_low = self.action_low.to(self.device)
        self.action_high = self.action_high.to(self.device)
        self.max_action = self.max_action.to(self.device)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # Exploration noise
        self.exploration_sigma = exploration_sigma
        self.exploration_sigma_min = exploration_sigma_min
        self.exploration_decay = exploration_decay

        # Actors & Critics for each agent
        self.actors = {}
        self.actors_target = {}
        self.critics = {}
        self.critics_target = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        for agent in self.agents:
            actor = CommPolicyMLP(
                self.obs_dim,
                self.action_dim,
                hidden_dim=256,
                msg_len=msg_len,
                vocab_size=vocab_size,
                comm_tau=comm_tau,
                comm_hard=comm_hard,
            )
            actor_target = CommPolicyMLP(
                self.obs_dim,
                self.action_dim,
                hidden_dim=256,
                msg_len=msg_len,
                vocab_size=vocab_size,
                comm_tau=comm_tau,
                comm_hard=comm_hard,
            )
            actor_target.load_state_dict(actor.state_dict())

            critic = CentralizedCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_agents=self.n_agents,
                hidden_dim=256,
            )
            critic_target = CentralizedCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_agents=self.n_agents,
                hidden_dim=256,
            )
            critic_target.load_state_dict(critic.state_dict())

            actor.to(self.device)
            actor_target.to(self.device)
            critic.to(self.device)
            critic_target.to(self.device)

            self.actors[agent] = actor
            self.actors_target[agent] = actor_target
            self.critics[agent] = critic
            self.critics_target[agent] = critic_target

            self.actor_optimizers[agent] = optim.Adam(actor.parameters(), lr=actor_lr)
            self.critic_optimizers[agent] = optim.Adam(critic.parameters(), lr=critic_lr)

        # Save dir
        os.makedirs("checkpoints_maddpg_comm", exist_ok=True)

        # Logging for curves
        self.logs = {
            "episode": [],
            "avg_reward": [],
            "max_reward": [],
            "buffer_size": [],
            "sigma": [],
            "actor_loss": [],
            "critic_loss": [],
        }
        self._last_actor_loss = None
        self._last_critic_loss = None

    # --------------- utility ---------------

    def select_actions(self, obs_dict, noise=True):
        """
        obs_dict: dict[agent] -> np.array(obs_dim,)
        returns: dict[agent] -> np.array(action_dim,)
        """
        obs_all = np.array([obs_dict[a] for a in self.agents], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(self.device)  # [1, n_agents, obs_dim]

        actions_dict = {}

        for idx, agent in enumerate(self.agents):
            obs_i = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 這裡的 obs_all_tensor / self_index 只是 for 介面相容，CommPolicyMLP 不會用到
                dist, _ = self.actors[agent](obs_i, obs_all_tensor, self_index=idx)
                mu = dist.mean
                action = torch.tanh(mu) * self.max_action

            if noise:
                eps = torch.normal(
                    mean=0.0,
                    std=self.exploration_sigma,
                    size=action.shape,
                    device=self.device,
                )
                action = action + eps

            action = torch.max(torch.min(action, self.action_high), self.action_low)
            actions_dict[agent] = action.cpu().numpy()[0]

        return actions_dict

    @staticmethod
    def soft_update(target, source, tau):
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)

    # --------------- learning step ---------------

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs = torch.from_numpy(obs).float().to(self.device)           # [B, n_agents, obs_dim]
        actions = torch.from_numpy(actions).float().to(self.device)   # [B, n_agents, action_dim]
        rewards = torch.from_numpy(rewards).float().to(self.device)   # [B, n_agents]
        next_obs = torch.from_numpy(next_obs).float().to(self.device) # [B, n_agents, obs_dim]
        dones = torch.from_numpy(dones).float().to(self.device)       # [B, n_agents]

        actor_losses = []
        critic_losses = []

        for agent_idx, agent in enumerate(self.agents):
            # ---------------- Critic update ----------------
            q_current = self.critics[agent](obs, actions)  # [B]

            with torch.no_grad():
                next_actions_list = []
                for j, agent_j in enumerate(self.agents):
                    obs_j_next = next_obs[:, j, :]  # [B, obs_dim]
                    dist_j, _ = self.actors_target[agent_j](
                        obs_j_next, next_obs, self_index=j
                    )
                    mu_j = dist_j.mean
                    a_j = torch.tanh(mu_j) * self.max_action
                    next_actions_list.append(a_j)

                next_actions = torch.stack(next_actions_list, dim=1)  # [B, n_agents, action_dim]
                q_target_next = self.critics_target[agent](next_obs, next_actions)  # [B]

                r_i = rewards[:, agent_idx]  # [B]
                done_i = dones[:, agent_idx] # [B]
                y = r_i + self.gamma * (1.0 - done_i) * q_target_next  # [B]

            critic_loss = nn.MSELoss()(q_current, y)

            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.critic_optimizers[agent].step()

            critic_losses.append(critic_loss.detach().cpu().item())

            # ---------------- Actor update (with Comm cost) ----------------
            actions_pred_list = []
            msg_cost_list = []

            for j, agent_j in enumerate(self.agents):
                obs_j = obs[:, j, :]  # [B, obs_dim]

                if agent_j == agent:
                    dist_j, comm_info = self.actors[agent_j](obs_j, obs, self_index=j)
                    mu_j = dist_j.mean
                    a_j = torch.tanh(mu_j) * self.max_action

                    msg_probs = comm_info.get("msg_probs", None)  # [B, L, V]
                    if msg_probs is not None:
                        eps = 1e-8
                        msg_entropy = -(msg_probs * torch.log(msg_probs + eps)).sum(dim=-1).mean()
                        msg_cost_list.append(msg_entropy)
                else:
                    with torch.no_grad():
                        dist_j, _ = self.actors[agent_j](obs_j, obs, self_index=j)
                        mu_j = dist_j.mean
                        a_j = torch.tanh(mu_j) * self.max_action

                actions_pred_list.append(a_j)

            actions_pred = torch.stack(actions_pred_list, dim=1)  # [B, n_agents, action_dim]

            actor_loss = -self.critics[agent](obs, actions_pred).mean()

            if len(msg_cost_list) > 0:
                msg_cost = torch.stack(msg_cost_list).mean()
                actor_loss = actor_loss + self.lambda_comm * msg_cost

            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.actor_optimizers[agent].step()

            actor_losses.append(actor_loss.detach().cpu().item())

            # ---------------- Soft update ----------------
            self.soft_update(self.actors_target[agent], self.actors[agent], self.tau)
            self.soft_update(self.critics_target[agent], self.critics[agent], self.tau)

        # 探索噪音逐步下降
        self.exploration_sigma = max(
            self.exploration_sigma_min,
            self.exploration_sigma - self.exploration_decay,
        )

        self._last_actor_loss = float(np.mean(actor_losses))
        self._last_critic_loss = float(np.mean(critic_losses))

    # --------------- train loop ---------------

    def save_actors(self, suffix):
        """
        只存 Actor（CommPolicyMLP），格式：
        { agent_name: state_dict }
        """
        checkpoint = {agent: self.actors[agent].state_dict()
                      for agent in self.agents}
        path = os.path.join("checkpoints_maddpg_comm", f"maddpg_comm_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"✓ MADDPG-Comm actors saved to: {path}")

    def train(self, n_episodes=1000, updates_per_step=1):
        best_avg_reward = -float("inf")

        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agents}

            for t in range(self.max_episode_steps):
                actions_dict = self.select_actions(obs_dict, noise=True)

                obs_array = np.array([obs_dict[a] for a in self.agents], dtype=np.float32)
                actions_array = np.array([actions_dict[a] for a in self.agents], dtype=np.float32)

                next_obs_dict, rewards, terms, truncs, infos = self.env.step(actions_dict)

                rewards_array = np.array([rewards[a] for a in self.agents], dtype=np.float32)
                dones_array = np.array(
                    [bool(terms[a] or truncs[a]) for a in self.agents],
                    dtype=np.float32,
                )
                next_obs_array = np.array([next_obs_dict[a] for a in self.agents], dtype=np.float32)

                self.buffer.push(
                    obs_array, actions_array, rewards_array, next_obs_array, dones_array
                )

                for agent in self.agents:
                    episode_reward[agent] += rewards[agent]

                obs_dict = next_obs_dict

                for _ in range(updates_per_step):
                    self.learn()

                if any(terms.values()) or any(truncs.values()):
                    break

            avg_ep_reward = np.mean(list(episode_reward.values()))
            max_ep_reward = np.max(list(episode_reward.values()))

            print(
                f"[MADDPG-Comm-MLP] Episode {ep + 1}/{n_episodes} | "
                f"AvgRew={avg_ep_reward:.2f} | MaxRew={max_ep_reward:.2f} | "
                f"Buffer={len(self.buffer)} | sigma={self.exploration_sigma:.3f}"
            )

            actor_loss = (
                float(self._last_actor_loss) if self._last_actor_loss is not None else np.nan
            )
            critic_loss = (
                float(self._last_critic_loss) if self._last_critic_loss is not None else np.nan
            )

            self.logs["episode"].append(ep + 1)
            self.logs["avg_reward"].append(avg_ep_reward)
            self.logs["max_reward"].append(max_ep_reward)
            self.logs["buffer_size"].append(len(self.buffer))
            self.logs["sigma"].append(float(self.exploration_sigma))
            self.logs["actor_loss"].append(actor_loss)
            self.logs["critic_loss"].append(critic_loss)

            np.savez("maddpg_comm_logs.npz", **self.logs)

            if avg_ep_reward > best_avg_reward:
                best_avg_reward = avg_ep_reward
                self.save_actors("best")

            if (ep + 1) % 100 == 0:
                self.save_actors(f"ep{ep + 1}")

        np.savez("maddpg_comm_logs.npz", **self.logs)
        print("✓ Training logs saved to maddpg_comm_logs.npz")


# -----------------------
# main
# -----------------------

if __name__ == "__main__":
    trainer = MADDPGCommTrainer(
        scenario="Ant",
        agent_conf="2x4",
        gamma=0.99,
        tau=0.01,
        actor_lr=1e-4,
        critic_lr=1e-3,
        batch_size=256,
        buffer_capacity=int(1e6),
        exploration_sigma=0.2,
        exploration_sigma_min=0.05,
        exploration_decay=1e-6,
        max_episode_steps=500,
        msg_len=4,
        vocab_size=8,
        comm_tau=1.0,
        comm_hard=True,
        lambda_comm=1e-3,
    )

    trainer.train(n_episodes=1000, updates_per_step=1)
