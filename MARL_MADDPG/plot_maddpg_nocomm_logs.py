#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 環境畫圖用
import matplotlib.pyplot as plt


def main(
    log_path="maddpg_nocomm_logs.npz",
    out_dir="png/maddpg_no_comm",
):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found.")

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(log_path)

    episode     = data["episode"]
    avg_reward  = data["avg_reward"]
    max_reward  = data["max_reward"]
    actor_loss  = data["actor_loss"]
    critic_loss = data["critic_loss"]
    sigma       = data["sigma"]
    buffer_size = data["buffer_size"]

    # --- Reward 曲線 ---
    plt.figure()
    plt.plot(episode, avg_reward, label="Avg Reward", linestyle="--")
    # plt.plot(episode, max_reward, label="Max Reward", linestyle="-")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MADDPG w/o Communication: Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = os.path.join(out_dir, "reward_curve_maddpg_no_comm.png")
    plt.savefig(reward_path, dpi=200)

    # --- Loss 曲線 ---
    plt.figure()
    plt.plot(episode, actor_loss, label="Actor Loss")
    plt.plot(episode, critic_loss, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("MADDPG w/o Communication: Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve_maddpg_no_comm.png")
    plt.savefig(loss_path, dpi=200)

    # --- Sigma / Buffer 曲線 ---
    fig, ax1 = plt.subplots()
    ax1.plot(episode, sigma, label="sigma")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Exploration σ")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(episode, buffer_size, label="buffer size", color="C1")
    ax2.set_ylabel("Replay Buffer Size")

    plt.title("MADDPG w/o Communication: Sigma & Buffer")
    fig.tight_layout()
    sigma_path = os.path.join(out_dir, "sigma_buffer_maddpg_no_comm.png")
    plt.savefig(sigma_path, dpi=200)

    print(f"✓ Saved: {reward_path}")
    print(f"✓ Saved: {loss_path}")
    print(f"✓ Saved: {sigma_path}")


if __name__ == "__main__":
    main()
