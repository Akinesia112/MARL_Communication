#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gymnasium_robotics import mamujoco_v1
from baseline_no_comm import SimpleCritic
from gnn_policies import GATPolicyNoComm
import os


class NoCommPPOTrainer:
    def __init__(self, scenario="Ant", agent_conf="4x2", lr=1e-4, gamma=0.99):
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None,
        )

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        action_dim = self.env.action_space(self.agents[0]).shape[0]

        self.gamma = gamma

        # === networks: SimplePolicy / SimpleCritic (無通訊) ===
        self.policies = {
            agent: GATPolicyNoComm(obs_dim, action_dim, hidden_dim=256)
            for agent in self.agents
        }
        self.critics = {agent: SimpleCritic(obs_dim) for agent in self.agents}

        # === optimizers ===
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

        # checkpoints
        os.makedirs("checkpoints_ppo_gat_no_comm_ant_4x2", exist_ok=True)

        # logs 會存成 npz
        self.logs = {
            "iteration": [],
            "avg_reward": [],
            "max_reward": [],
            "policy_loss": [],
            "value_loss": [],
            # for compatibility with sigma/buffer plot
            "sigma": [],
            "buffer_size": [],
        }
        self.last_policy_loss = None
        self.last_value_loss = None

    def _move_to_device(self):
        for agent in self.agents:
            self.policies[agent].to(self.device)
            self.critics[agent].to(self.device)

    # ---------------- Rollout ----------------

    def collect_rollout(self, n_steps=2048):
        rollout = {
            agent: {
                "obs": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
                "dones": [],
            }
            for agent in self.agents
        }

        obs_dict, _ = self.env.reset()
        for step in range(n_steps):
            actions_dict = {}

            for agent in self.agents:
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    dist = self.policies[agent](obs)      # SimplePolicy(obs) -> Normal
                    value = self.critics[agent](obs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)

                rollout[agent]["obs"].append(obs_dict[agent])
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

    def compute_advantages(self, rollout, lam=0.95):
        for agent in self.agents:
            rewards = np.array(rollout[agent]["rewards"], dtype=np.float32)
            values = np.array(rollout[agent]["values"], dtype=np.float32)
            dones = np.array(rollout[agent]["dones"], dtype=np.float32)

            advantages = np.zeros_like(rewards, dtype=np.float32)

            # bootstrap last value
            last_obs = rollout[agent]["obs"][-1]
            with torch.no_grad():
                last_value = (
                    self.critics[agent](
                        torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                    )
                    .cpu()
                    .item()
                )

            last_gae = 0.0
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
        self.compute_advantages(rollout)

        all_policy_losses = []
        all_value_losses = []

        for epoch in range(epochs):
            for agent in self.agents:
                obs = np.array(rollout[agent]["obs"], dtype=np.float32)
                actions = np.array(rollout[agent]["actions"], dtype=np.float32)
                old_log_probs = np.array(rollout[agent]["log_probs"], dtype=np.float32)
                advantages = rollout[agent]["advantages"]
                returns = rollout[agent]["returns"]

                # normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                n_samples = len(obs)
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    batch_obs = torch.FloatTensor(obs[batch_idx]).to(self.device)
                    batch_actions = torch.FloatTensor(actions[batch_idx]).to(self.device)
                    batch_old_log_probs = torch.FloatTensor(
                        old_log_probs[batch_idx]
                    ).to(self.device)
                    batch_advantages = torch.FloatTensor(
                        advantages[batch_idx]
                    ).to(self.device)
                    batch_returns = torch.FloatTensor(
                        returns[batch_idx]
                    ).to(self.device)

                    # policy update
                    dist = self.policies[agent](batch_obs)
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

                    # value update
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

        self.last_policy_loss = float(np.mean(all_policy_losses)) if all_policy_losses else None
        self.last_value_loss = float(np.mean(all_value_losses)) if all_value_losses else None

    # ---------------- Save & Train loop ----------------

    def save_checkpoint(self, suffix):
        checkpoint = {agent: self.policies[agent].state_dict() for agent in self.agents}
        path = os.path.join("checkpoints_ppo_gat_no_comm_ant_4x2", f"ppo_nocomm_{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"✓ No-Comm PPO saved: {path}")

    def train(self, n_iterations=1000, save_freq=50):
        best_reward = -float("inf")

        for iteration in range(n_iterations):
            rollout = self.collect_rollout(n_steps=2048)
            self.train_step(rollout, epochs=4, batch_size=64)

            avg_reward = float(
                np.mean([np.sum(rollout[a]["rewards"]) for a in self.agents])
            )
            max_reward = float(
                np.max([np.sum(rollout[a]["rewards"]) for a in self.agents])
            )

            print(
                f"[PPO No-Comm] Iter {iteration}: "
                f"Avg={avg_reward:.2f}, Max={max_reward:.2f}, "
                f"policy_loss={self.last_policy_loss}, value_loss={self.last_value_loss}"
            )

            # logging
            self.logs["iteration"].append(iteration + 1)
            self.logs["avg_reward"].append(avg_reward)
            self.logs["max_reward"].append(max_reward)
            self.logs["policy_loss"].append(
                self.last_policy_loss if self.last_policy_loss is not None else np.nan
            )
            self.logs["value_loss"].append(
                self.last_value_loss if self.last_value_loss is not None else np.nan
            )
            # PPO 沒有 sigma / buffer，給個常數方便畫圖
            self.logs["sigma"].append(0.0)
            self.logs["buffer_size"].append(len(rollout[self.agents[0]]["rewards"]))

            np.savez("ppo_gat_no_comm_logs_ant_4x2.npz", **self.logs)

            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint("best")
                print(f"  ✓ 新最佳模型！獎勵: {best_reward:.2f}")

            if (iteration + 1) % save_freq == 0:
                self.save_checkpoint(f"iter{iteration + 1}")

        np.savez("ppo_gat_no_comm_logs_ant_4x2.npz", **self.logs)
        print("✓ Training logs saved to ppo_gat_no_comm_logs_ant_4x2.npz")


if __name__ == "__main__":
    trainer = NoCommPPOTrainer(
        scenario="Ant",
        agent_conf="4x2",
        lr=1e-4,
        gamma=0.99,
    )
    trainer.train(n_iterations=1000, save_freq=50)
