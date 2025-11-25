#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless server 用
import matplotlib.pyplot as plt


def main(log_path="maddpg_attention_no_comm_logs.npz"):
    data = np.load(log_path)

    episode = data["episode"]
    avg_reward = data["avg_reward"]
    max_reward = data["max_reward"]
    actor_loss = data["actor_loss"]
    critic_loss = data["critic_loss"]

    # ---- Reward curve ----
    plt.figure()
    plt.plot(episode, avg_reward, label="Avg Reward")
    plt.plot(episode, max_reward, label="Max Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MADDPG + Attention: Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curve_maddpg_attention_no_comm.png", dpi=200)

    # ---- Loss curve ----
    plt.figure()
    plt.plot(episode, actor_loss, label="Actor Loss")
    plt.plot(episode, critic_loss, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("MADDPG + Attention: Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_maddpg_attention_no_comm.png", dpi=200)

    print("✓ Saved: reward_curve_maddpg_attention_no_comm.png, loss_curve_maddpg_attention_no_comm.png")


if __name__ == "__main__":
    main()
