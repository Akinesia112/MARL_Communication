#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

LOG_PATH = "maddpg_attention_comm_logs.npz"
OUT_DIR = os.path.join("png", "maddpg_attention_comm")

def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"{LOG_PATH} not found. Make sure you ran training first.")

    # 建資料夾
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(LOG_PATH)

    episodes     = data["episode"]
    avg_reward   = data["avg_reward"]
    max_reward   = data["max_reward"]
    buffer_size  = data["buffer_size"]
    sigma        = data["sigma"]
    actor_loss   = data["actor_loss"]
    critic_loss  = data["critic_loss"]

    # ---- Reward curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, avg_reward, label="Avg Reward", alpha=0.7)
    #plt.plot(episodes, max_reward, label="Max Reward", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MADDPG Attention Comm: Reward vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path_reward = os.path.join(OUT_DIR, "reward.png")
    plt.savefig(out_path_reward, dpi=200)
    print(f"✓ Saved reward plot to {out_path_reward}")

    # ---- Loss curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, actor_loss, label="Actor Loss")
    plt.plot(episodes, critic_loss, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("MADDPG Attention Comm: Loss vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path_loss = os.path.join(OUT_DIR, "loss.png")
    plt.savefig(out_path_loss, dpi=200)
    print(f"✓ Saved loss plot to {out_path_loss}")

    # ---- Exploration sigma / buffer size ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(episodes, sigma, label="sigma")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Exploration σ")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(episodes, buffer_size, label="Buffer size", color="C1")
    ax2.set_ylabel("Replay Buffer Size")
    ax2.tick_params(axis="y", labelcolor="C1")

    plt.title("MADDPG Attention Comm: Exploration & Buffer Size")
    fig.tight_layout()

    out_path_sigma = os.path.join(OUT_DIR, "sigma_buffer.png")
    plt.savefig(out_path_sigma, dpi=200)
    print(f"✓ Saved sigma/buffer plot to {out_path_sigma}")

if __name__ == "__main__":
    main()
