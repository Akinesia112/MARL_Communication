#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 畫圖
import matplotlib.pyplot as plt


def main(
    log_path="maddpg_comm_logs_ant_4x2.npz",
    out_dir="png/maddpg_mlp_comm_ant_4x2",
):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found. Run training first.")

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(log_path)

    episodes    = data["episode"]
    avg_reward  = data["avg_reward"]
    max_reward  = data["max_reward"]
    actor_loss  = data["actor_loss"]
    critic_loss = data["critic_loss"]
    sigma       = data["sigma"]
    buffer_size = data["buffer_size"]

    # --- Reward curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, avg_reward, label="Avg Reward")
    # plt.plot(episodes, max_reward, label="Max Reward", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MADDPG Comm (MLP): Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = os.path.join(out_dir, "reward_curve_maddpg_mlp_comm.png")
    plt.savefig(reward_path, dpi=200)

    # --- Loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, actor_loss, label="Actor Loss")
    plt.plot(episodes, critic_loss, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("MADDPG Comm (MLP): Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve_maddpg_mlp_comm.png")
    plt.savefig(loss_path, dpi=200)

    # --- Sigma / Buffer curve ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(episodes, sigma, label="sigma")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Exploration σ", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(episodes, buffer_size, label="buffer size", color="C1")
    ax2.set_ylabel("Replay Buffer Size", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    plt.title("MADDPG Comm (MLP): Sigma & Buffer Size")
    fig.tight_layout()
    sigma_path = os.path.join(out_dir, "sigma_buffer_maddpg_mlp_comm.png")
    plt.savefig(sigma_path, dpi=200)

    print(f"✓ Saved: {reward_path}")
    print(f"✓ Saved: {loss_path}")
    print(f"✓ Saved: {sigma_path}")


if __name__ == "__main__":
    main()
