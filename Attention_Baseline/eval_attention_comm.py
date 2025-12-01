#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio
import numpy as np
import torch
from gymnasium_robotics import mamujoco_v1

from attention_emergent_comm import CommPolicyWithAttention


def evaluate_policy(
    model_path="checkpoints_ppo_attention_comm/model_iter_best.pt",
    n_episodes=1,
    max_episode_steps=500,
    save_video=True,
):
    os.makedirs("videos", exist_ok=True)

    # 只用 rgb_array 渲染，避免 DISPLAY 問題
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="2x4",
        agent_obsk=1,
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )

    agents = env.agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    # 建立 policy
    policies = {
        agent: CommPolicyWithAttention(
            obs_dim,
            action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        )
        for agent in agents
    }

    # 載入 checkpoint
    if model_path is not None and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        for agent in agents:
            if agent in ckpt:
                policies[agent].load_state_dict(ckpt[agent])
        print(f"✓ Loaded model from {model_path}")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    else:
        print("⚠ 找不到模型，使用隨機初始化權重")
        model_name = "random_comm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for agent in agents:
        policies[agent].to(device)
        policies[agent].eval()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agents}
        frames = []

        done = False
        step = 0

        while not done and step < max_episode_steps:
            all_obs = (
                torch.FloatTensor(
                    np.array([obs_dict[a] for a in agents], dtype=np.float32)
                )
                .unsqueeze(0)
                .to(device)
            )

            actions_dict = {}
            with torch.no_grad():
                for idx, agent in enumerate(agents):
                    obs = (
                        torch.FloatTensor(obs_dict[agent])
                        .unsqueeze(0)
                        .to(device)
                    )
                    dist, _ = policies[agent](obs, all_obs)  # 若有 self_index：(..., self_index=idx)
                    action = dist.mean  # 用 mean，比 sample 穩定
                    actions_dict[agent] = action.cpu().numpy()[0]

            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            for agent in agents:
                episode_reward[agent] += rewards[agent]

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            step += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            else:
                obs_dict = next_obs

        avg_ep_rew = sum(episode_reward.values()) / len(agents)
        print(
            f"[Eval] Episode {ep+1}/{n_episodes} | "
            f"steps={step} | avg_reward={avg_ep_rew:.2f} | rewards={episode_reward}"
        )

        if save_video and len(frames) > 0:
            out_path = f"videos/{model_name}_episode_{ep+1}.mp4"
            imageio.mimsave(out_path, frames, fps=30)
            print(f"  ✓ Video saved to {out_path}")

    env.close()


if __name__ == "__main__":
    # 你可以改成想測的 checkpoint
    evaluate_policy(
        model_path="checkpoints_attention_comm/model_iter_best.pt",
        n_episodes=1,
        max_episode_steps=500,
        save_video=True,
    )
