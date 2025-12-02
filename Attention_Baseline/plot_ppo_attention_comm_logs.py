#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 環境畫圖用
import matplotlib.pyplot as plt


def main(
    log_path="ppo_attention_comm_logs_ant_4x2.npz",
    out_dir="png/ppo_attention_comm_ant_4x2",
):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found. Run training first.")

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(log_path)

    iteration    = data["iteration"]
    avg_reward   = data["avg_reward"]
    max_reward   = data["max_reward"]
    policy_loss  = data["policy_loss"]
    value_loss   = data["value_loss"]

    # --- Reward curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(iteration, avg_reward, label="Avg Reward")
    #plt.plot(iteration, max_reward, label="Max Reward", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("PPO Attention Comm: Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = os.path.join(out_dir, "ppo_attention_comm_reward.png")
    plt.savefig(reward_path, dpi=200)

    # --- Loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(iteration, policy_loss, label="Policy Loss", linestyle="--")
    plt.plot(iteration, value_loss, label="Value Loss", linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("PPO Attention Comm: Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "ppo_attention_comm_loss.png")
    plt.savefig(loss_path, dpi=200)

    # --- Sigma / Buffer curve (如果 log 裡有就畫，沒有就跳過) ---
    sigma_path = None
    if "sigma" in data.files and "buffer_size" in data.files:
        sigma       = data["sigma"]
        buffer_size = data["buffer_size"]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(iteration, sigma, label="sigma")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Exploration σ", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(iteration, buffer_size, label="buffer size", color="C1")
        ax2.set_ylabel("Replay Buffer Size", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        plt.title("PPO Attention Comm: Sigma & Buffer Size")
        fig.tight_layout()
        sigma_path = os.path.join(out_dir, "ppo_attention_comm_sigma_buffer.png")
        plt.savefig(sigma_path, dpi=200)
    else:
        print("⚠ No 'sigma' / 'buffer_size' in log; skipping sigma-buffer plot.")

    print(f"✓ Saved: {reward_path}")
    print(f"✓ Saved: {loss_path}")
    if sigma_path is not None:
        print(f"✓ Saved: {sigma_path}")


if __name__ == "__main__":
    main()
