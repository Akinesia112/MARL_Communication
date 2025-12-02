#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium_robotics import mamujoco_v1

from attention_comm import PolicyWithAttention


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

        obs = np.array(obs)            # [B, n_agents, obs_dim]
        actions = np.array(actions)    # [B, n_agents, action_dim]
        rewards = np.array(rewards)    # [B, n_agents]
        next_obs = np.array(next_obs)  # [B, n_agents, obs_dim]
        dones = np.array(dones)        # [B, n_agents]

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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, all_obs, all_actions):
        """
        all_obs:     [B, n_agents, obs_dim]
        all_actions: [B, n_agents, action_dim]
        """
        x = torch.cat(
            [all_obs.reshape(all_obs.size(0), -1),
             all_actions.reshape(all_actions.size(0), -1)],
            dim=-1
        )
        q = self.net(x)
        return q.squeeze(-1)  # [B]


# -----------------------
# MADDPG Trainer (Attention Actors)
# -----------------------

class MADDPGAttentionTrainer:
    def __init__(
        self,
        scenario="Ant",
        agent_conf="4x2",
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
    ):
        self.checkpoint_dir = "checkpoints_maddpg_attention_no_comm_ant_4x2"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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

        # Exploration
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
            # Actor uses PolicyWithAttention
            actor = PolicyWithAttention(self.obs_dim, self.action_dim)
            actor_target = PolicyWithAttention(self.obs_dim, self.action_dim)
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

        # Save dir (actor only, so你原本的 eval_policy.py 可以直接載入)
        os.makedirs("checkpoints_maddpg_attention_no_comm_ant_4x2", exist_ok=True)

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
            obs_i = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(self.device)  # [1, obs_dim]

            with torch.no_grad():
                dist, _ = self.actors[agent](obs_i, obs_all_tensor)
                mu = dist.mean  # [1, action_dim]
                # map to action space with tanh
                action = torch.tanh(mu) * self.max_action

            if noise:
                eps = torch.normal(
                    mean=0.0,
                    std=self.exploration_sigma,
                    size=action.shape,
                    device=self.device,
                )
                action = action + eps

            # clamp to env bounds
            action = torch.max(torch.min(action, self.action_high), self.action_low)

            actions_dict[agent] = action.cpu().numpy()[0]

        return actions_dict

    def soft_update(self, target, source, tau):
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)

    # --------------- learning step ---------------

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        count = 0

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        # to tensors
        obs = torch.from_numpy(obs).float().to(self.device)              # [B, n_agents, obs_dim]
        actions = torch.from_numpy(actions).float().to(self.device)      # [B, n_agents, action_dim]
        rewards = torch.from_numpy(rewards).float().to(self.device)      # [B, n_agents]
        next_obs = torch.from_numpy(next_obs).float().to(self.device)    # [B, n_agents, obs_dim]
        dones = torch.from_numpy(dones).float().to(self.device)          # [B, n_agents]

        # for each agent, update its critic & actor
        for agent_idx, agent in enumerate(self.agents):
            # ---- Critic update ----
            # Current Q
            q_current = self.critics[agent](obs, actions)  # [B]

            # Next actions using target actors
            with torch.no_grad():
                next_actions_list = []
                for j, agent_j in enumerate(self.agents):
                    obs_j = next_obs[:, j, :]  # [B, obs_dim]
                    dist_j, _ = self.actors_target[agent_j](obs_j, next_obs)
                    mu_j = dist_j.mean
                    a_j = torch.tanh(mu_j) * self.max_action
                    next_actions_list.append(a_j)
                next_actions = torch.stack(next_actions_list, dim=1)  # [B, n_agents, action_dim]

                q_target_next = self.critics_target[agent](next_obs, next_actions)  # [B]

                # y_i = r_i + gamma * (1 - done_i) * Q_target
                r_i = rewards[:, agent_idx]          # [B]
                done_i = dones[:, agent_idx]         # [B]
                y = r_i + self.gamma * (1.0 - done_i) * q_target_next

            critic_loss = nn.MSELoss()(q_current, y)

            total_critic_loss += critic_loss.item()
            count += 1

            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.critic_optimizers[agent].step()

            # ---- Actor update ----
            # 重新用 local actors 計算所有 agents 的動作
            actions_pred_list = []
            for j, agent_j in enumerate(self.agents):
                obs_j = obs[:, j, :]  # [B, obs_dim]
                if agent_j == agent:
                    dist_j, _ = self.actors[agent_j](obs_j, obs)
                    mu_j = dist_j.mean
                    a_j = torch.tanh(mu_j) * self.max_action
                else:
                    # 其他 agent 的 action 不反傳這個 agent 的梯度
                    with torch.no_grad():
                        dist_j, _ = self.actors[agent_j](obs_j, obs)
                        mu_j = dist_j.mean
                        a_j = torch.tanh(mu_j) * self.max_action
                actions_pred_list.append(a_j)
            actions_pred = torch.stack(actions_pred_list, dim=1)  # [B, n_agents, action_dim]

            actor_loss = -self.critics[agent](obs, actions_pred).mean()

            total_actor_loss += actor_loss.item()

            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.actor_optimizers[agent].step()

            # ---- Soft update target nets ----
            self.soft_update(self.actors_target[agent], self.actors[agent], self.tau)
            self.soft_update(self.critics_target[agent], self.critics[agent], self.tau)

        if count > 0:
            self._last_actor_loss = total_actor_loss / count
            self._last_critic_loss = total_critic_loss / count

        # 探索噪音慢慢下降
        self.exploration_sigma = max(
            self.exploration_sigma_min,
            self.exploration_sigma - self.exploration_decay
        )

    # --------------- train loop ---------------

    def save_actors(self, suffix):
        """
        只存 Actor（PolicyWithAttention），格式和 PPO 一樣：
        { agent_name: state_dict }
        這樣你原本的 eval_policy.py 直接 `model_path="checkpoints_maddpg/xxx.pt"` 就能載。
        """
        checkpoint = {agent: self.actors[agent].state_dict()
                      for agent in self.agents}
        path = os.path.join("checkpoints_maddpg_attention_no_comm_ant_4x2", f"maddpg_attention_no_comm_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"✓ MADDPG actors saved to: {path}")

    def train(self, n_episodes=1000, updates_per_step=1):
        best_avg_reward = -float("inf")

        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agents}

            for t in range(self.max_episode_steps):
                # 選 action（含噪音）
                actions_dict = self.select_actions(obs_dict, noise=True)

                # 整理儲存格式
                obs_array = np.array([obs_dict[a] for a in self.agents], dtype=np.float32)
                actions_array = np.array([actions_dict[a] for a in self.agents], dtype=np.float32)

                next_obs_dict, rewards, terms, truncs, infos = self.env.step(actions_dict)

                rewards_array = np.array([rewards[a] for a in self.agents], dtype=np.float32)
                dones_array = np.array(
                    [bool(terms[a] or truncs[a]) for a in self.agents],
                    dtype=np.float32,
                )
                next_obs_array = np.array([next_obs_dict[a] for a in self.agents], dtype=np.float32)

                # 存入 buffer
                self.buffer.push(
                    obs_array, actions_array, rewards_array, next_obs_array, dones_array
                )

                for agent in self.agents:
                    episode_reward[agent] += rewards[agent]

                obs_dict = next_obs_dict

                # 更新網路
                for _ in range(updates_per_step):
                    self.learn()

                if any(terms.values()) or any(truncs.values()):
                    break

            avg_ep_reward = np.mean(list(episode_reward.values()))
            max_ep_reward = np.max(list(episode_reward.values()))

            print(
                f"[MADDPG] Episode {ep + 1}/{n_episodes} | "
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

            np.savez("maddpg_attention_no_comm_logs_ant_4x2.npz", **self.logs)

            # 儲存最佳
            if avg_ep_reward > best_avg_reward:
                best_avg_reward = avg_ep_reward
                self.save_actors("best")

            # 週期性 checkpoint
            if (ep + 1) % 100 == 0:
                self.save_actors(f"ep{ep + 1}")

        np.savez("maddpg_attention_no_comm_logs_ant_4x2.npz", **self.logs)
        print("✓ Training logs saved to maddpg_attention_no_comm_logs_ant_4x2.npz")


# -----------------------
# main
# -----------------------

if __name__ == "__main__":
    trainer = MADDPGAttentionTrainer(
        scenario="Ant",
        agent_conf="4x2",
        gamma=0.99,
        tau=0.01,
        actor_lr=1e-4,   # 和你 PPO 的 lr 一樣
        critic_lr=1e-3,
        batch_size=256,
        buffer_capacity=int(1e6),
        exploration_sigma=0.2,
        exploration_sigma_min=0.05,
        exploration_decay=1e-6,
        max_episode_steps=500,
    )

    trainer.train(n_episodes=1000, updates_per_step=1)
