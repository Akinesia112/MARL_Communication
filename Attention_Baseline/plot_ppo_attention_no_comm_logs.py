#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot PPO training logs for attention (no-comm baseline).

Reads:
    ppo_attention_no_comm_logs.npz

Fields expected:
    - iteration
    - avg_reward
    - max_reward
    - policy_loss
    - value_loss

Outputs:
    - ppo_attention_no_comm_rewards.png
    - ppo_attention_no_comm_losses.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main(log_path="ppo_attention_no_comm_logs_ant_4x2.npz", out_dir="png/ppo_attention_no_comm_ant_4x2",):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found.")

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(log_path, allow_pickle=True)

    iteration = np.array(data["iteration"])
    avg_reward = np.array(data["avg_reward"])
    max_reward = np.array(data["max_reward"])
    policy_loss = np.array(data["policy_loss"])
    value_loss = np.array(data["value_loss"])

    # ----------------- Reward curves -----------------
    plt.figure(figsize=(8, 5))
    plt.plot(iteration, avg_reward, label="Avg Reward")
    # plt.plot(iteration, max_reward, label="Max Reward", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("PPO Attention (no-comm) — Rewards")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_rewards = "png/ppo_attention_no_comm_ant_4x2/ppo_attention_no_comm_rewards.png"
    plt.tight_layout()
    plt.savefig(out_rewards, dpi=200)
    print(f"✓ Saved reward plot to {out_rewards}")

    # ----------------- Loss curves -----------------
    plt.figure(figsize=(8, 5))
    plt.plot(iteration, policy_loss, label="Policy Loss")
    plt.plot(iteration, value_loss, label="Value Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("PPO Attention (no-comm) — Losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_losses = "png/ppo_attention_no_comm_ant_4x2/ppo_attention_no_comm_losses.png"
    plt.tight_layout()
    plt.savefig(out_losses, dpi=200)
    print(f"✓ Saved loss plot to {out_losses}")

    # If you want interactive display when running locally, uncomment:
    # plt.show()


if __name__ == "__main__":
    main()
