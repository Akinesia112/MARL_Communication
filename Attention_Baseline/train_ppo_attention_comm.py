#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium_robotics import mamujoco_v1

from attention_emergent_comm import CommPolicyWithAttention
from baseline_no_comm import SimpleCritic


class MAMuJoCoTrainer:
    def __init__(self, scenario="Ant", agent_conf="4x2", lr=3e-4, gamma=0.99):
        self.checkpoint_dir = "checkpoints_ppo_attention_comm_ant_4x2"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None,  # 訓練時不渲染
        )
        

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        action_dim = self.env.action_space(self.agents[0]).shape[0]

        self.gamma = gamma

        # 策略與價值網路
        self.policies = {
            agent: CommPolicyWithAttention(
                obs_dim,
                action_dim,
                hidden_dim=256,
                msg_len=4,      # 頻寬控制
                vocab_size=8,
                comm_tau=1.0,
                comm_hard=True,
            )
            for agent in self.agents
        }
        self.critics = {
            agent: SimpleCritic(obs_dim) for agent in self.agents
        }

        # 優化器
        self.policy_optimizers = {
            agent: optim.Adam(self.policies[agent].parameters(), lr=lr)
            for agent in self.agents
        }
        self.critic_optimizers = {
            agent: optim.Adam(self.critics[agent].parameters(), lr=lr)
            for agent in self.agents
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()

        # log 資料
        self.logs = {
            "iteration": [],
            "avg_reward": [],
            "max_reward": [],
            "policy_loss": [],
            "value_loss": [],
        }
        self.last_policy_loss = None
        self.last_value_loss = None

        # checkpoint 目錄
        os.makedirs("checkpoints_ppo_attention_comm_ant_4x2", exist_ok=True)

    def _move_to_device(self):
        for agent in self.agents:
            self.policies[agent].to(self.device)
            self.critics[agent].to(self.device)

    # ---------------- rollout 收集 ----------------

    def collect_rollout(self, n_steps=2048):
        """收集一個 rollout（多步）"""
        rollout = {
            agent: {
                "obs": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
                "all_obs": [],
                "dones": [],
            }
            for agent in self.agents
        }

        obs_dict, _ = self.env.reset()

        for step in range(n_steps):
            all_obs = (
                torch.FloatTensor(
                    np.array([obs_dict[a] for a in self.agents], dtype=np.float32)
                )
                .unsqueeze(0)
                .to(self.device)
            )

            actions_dict = {}

            for agent_idx, agent in enumerate(self.agents):
                obs = (
                    torch.FloatTensor(obs_dict[agent])
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    # Emergent Comm：policy 會在內部 sample message，但我們這裡只用動作
                    dist, _ = self.policies[agent](obs, all_obs)  # 若你有 self_index 版本，就改成 (..., self_index=agent_idx)
                    value = self.critics[agent](obs)

                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)

                rollout[agent]["obs"].append(obs_dict[agent])
                rollout[agent]["all_obs"].append(all_obs.cpu().numpy()[0])
                rollout[agent]["actions"].append(action.cpu().numpy()[0])
                rollout[agent]["log_probs"].append(log_prob.cpu().item())
                rollout[agent]["values"].append(value.cpu().item())

                actions_dict[agent] = action.cpu().numpy()[0]

            next_obs, rewards, terms, truncs, infos = self.env.step(actions_dict)

            for agent in self.agents:
                rollout[agent]["rewards"].append(rewards[agent])
                rollout[agent]["dones"].append(terms[agent] or truncs[agent])

            if any(terms.values()) or any(truncs.values()):
                obs_dict, _ = self.env.reset()
            else:
                obs_dict = next_obs

        return rollout

    # ---------------- GAE ----------------

    def compute_advantages(self, rollout):
        """計算優勢函數（GAE）"""
        lam = 0.95

        for agent in self.agents:
            rewards = np.array(rollout[agent]["rewards"], dtype=np.float32)
            values = np.array(rollout[agent]["values"], dtype=np.float32)
            dones = np.array(rollout[agent]["dones"], dtype=np.float32)

            advantages = np.zeros_like(rewards, dtype=np.float32)
            last_gae = 0.0

            # bootstrap
            last_obs = rollout[agent]["obs"][-1]
            with torch.no_grad():
                last_value = (
                    self.critics[agent](
                        torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                    )
                    .cpu()
                    .item()
                )

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = last_value * (1.0 - dones[t])
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * lam * (1.0 - dones[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + values

            rollout[agent]["advantages"] = advantages
            rollout[agent]["returns"] = returns

    # ---------------- PPO update ----------------

    def train_step(self, rollout, clip_epsilon=0.2, epochs=4, batch_size=64):
        """PPO 更新（支援 mini-batch）"""
        self.compute_advantages(rollout)

        all_policy_losses = []
        all_value_losses = []

        for epoch in range(epochs):
            for agent in self.agents:
                obs = np.array(rollout[agent]["obs"], dtype=np.float32)
                all_obs = np.array(rollout[agent]["all_obs"], dtype=np.float32)
                actions = np.array(rollout[agent]["actions"], dtype=np.float32)
                old_log_probs = np.array(rollout[agent]["log_probs"], dtype=np.float32)
                advantages = rollout[agent]["advantages"].astype(np.float32)
                returns = rollout[agent]["returns"].astype(np.float32)

                # normalize advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                n_samples = len(obs)
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    batch_obs = torch.FloatTensor(obs[batch_idx]).to(self.device)
                    batch_all_obs = torch.FloatTensor(all_obs[batch_idx]).to(self.device)
                    batch_actions = torch.FloatTensor(actions[batch_idx]).to(self.device)
                    batch_old_log_probs = torch.FloatTensor(old_log_probs[batch_idx]).to(
                        self.device
                    )
                    batch_advantages = torch.FloatTensor(advantages[batch_idx]).to(
                        self.device
                    )
                    batch_returns = torch.FloatTensor(returns[batch_idx]).to(
                        self.device
                    )

                    dist, _ = self.policies[agent](batch_obs, batch_all_obs)
                    new_log_probs = dist.log_prob(batch_actions).sum(-1)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                    ) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    entropy = dist.entropy().mean()
                    policy_loss = policy_loss - 0.01 * entropy

                    self.policy_optimizers[agent].zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policies[agent].parameters(), 0.5
                    )
                    self.policy_optimizers[agent].step()

                    values = self.critics[agent](batch_obs).squeeze(-1)
                    value_loss = F.mse_loss(values, batch_returns)

                    self.critic_optimizers[agent].zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[agent].parameters(), 0.5
                    )
                    self.critic_optimizers[agent].step()

                    all_policy_losses.append(policy_loss.detach().cpu().item())
                    all_value_losses.append(value_loss.detach().cpu().item())

        if len(all_policy_losses) > 0:
            self.last_policy_loss = float(np.mean(all_policy_losses))
        else:
            self.last_policy_loss = None

        if len(all_value_losses) > 0:
            self.last_value_loss = float(np.mean(all_value_losses))
        else:
            self.last_value_loss = None

    # ---------------- checkpoint ----------------

    def save_checkpoint(self, iteration):
        checkpoint = {agent: self.policies[agent].state_dict() for agent in self.agents}
        out_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pt")
        torch.save(checkpoint, out_path)
        print(f"  ✓ 模型已保存到 checkpoints_attention_comm_ant_4x2/model_iter_{iteration}.pt")

    # ---------------- 主訓練迴圈 ----------------

    def train(self, n_iterations=1000, save_freq=50):
        best_reward = -float("inf")

        for iteration in range(n_iterations):
            rollout = self.collect_rollout(n_steps=2048)
            self.train_step(rollout, epochs=4, batch_size=64)

            # reward 統計
            total_rewards = [
                np.sum(rollout[agent]["rewards"]) for agent in self.agents
            ]
            avg_reward = float(np.mean(total_rewards))
            max_reward = float(np.max(total_rewards))

            print(
                f"Iter {iteration}: Avg={avg_reward:.2f}, Max={max_reward:.2f}, "
                f"policy_loss={self.last_policy_loss}, value_loss={self.last_value_loss}"
            )

            # log 下來
            self.logs["iteration"].append(iteration + 1)
            self.logs["avg_reward"].append(avg_reward)
            self.logs["max_reward"].append(max_reward)
            self.logs["policy_loss"].append(
                self.last_policy_loss if self.last_policy_loss is not None else np.nan
            )
            self.logs["value_loss"].append(
                self.last_value_loss if self.last_value_loss is not None else np.nan
            )

            np.savez("ppo_attention_comm_logs_ant_4x2.npz", **self.logs)

            # 保存最佳
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint("best")
                print(f"  ✓ 新最佳模型！獎勵: {best_reward:.2f}")

            if (iteration + 1) % save_freq == 0:
                self.save_checkpoint(iteration + 1)

        np.savez("ppo_attention_comm_logs_ant_4x2.npz", **self.logs)
        print("✓ Training logs saved to ppo_attention_comm_logs.npz")


if __name__ == "__main__":
    trainer = MAMuJoCoTrainer(
        scenario="Ant",
        agent_conf="4x2",
        lr=1e-4,  # 降低學習率
        gamma=0.99,
    )
    trainer.train(n_iterations=1000, save_freq=50)
