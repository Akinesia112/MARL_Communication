#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ✅ 必須在 import gymnasium 之前設定，避免走 X11
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np
import torch
from gymnasium_robotics import mamujoco_v1
from baseline_no_comm import SimplePolicy
import imageio.v2 as imageio


def make_env(render_mode="rgb_array", max_episode_steps=500):
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="4x2",
        agent_obsk=1,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    return env


def load_policies(env, model_path, device):
    agents = env.agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    policies = {agent: SimplePolicy(obs_dim, action_dim).to(device) for agent in agents}
    checkpoint = torch.load(model_path, map_location=device)
    for agent in agents:
        policies[agent].load_state_dict(checkpoint[agent])
        policies[agent].eval()

    print(f"✓ Loaded MADDPG No-Comm model from {model_path}")
    return policies


@torch.no_grad()
def run_single_episode(env, policies, device, max_episode_steps=500):
    """跑一個 episode，回傳 frames + reward dict"""
    obs_dict, _ = env.reset()
    agents = env.agents

    act_space = env.action_space(agents[0])
    action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
    max_action = torch.max(torch.abs(action_high))

    episode_reward = {agent: 0.0 for agent in agents}
    frames = []
    step_in_ep = 0

    while step_in_ep < max_episode_steps:
        actions_dict = {}
        for agent in agents:
            obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(device)
            dist = policies[agent](obs)
            mu = dist.mean
            action = torch.tanh(mu) * max_action
            actions_dict[agent] = action.cpu().numpy()[0]

        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

        for agent in agents:
            episode_reward[agent] += float(rewards[agent])

        frame = env.render()  # rgb_array
        if frame is not None:
            frames.append(frame)

        step_in_ep += 1

        if any(terms.values()) or any(truncs.values()):
            break

        obs_dict = next_obs

    return frames, episode_reward


def evaluate_maddpg_nocomm(
    model_path="checkpoints_maddpg_no_comm_ant_4x2/maddpg_nocomm_best.pt",
    out_dir="videos",
    out_name="maddpg_no_comm_ant_4x2.mp4",
    mode="episode",          # "episode" 或 "fixed_time"
    n_episodes=1,            # mode="episode" 時使用
    target_seconds=300,      # mode="fixed_time" 時使用 (例如 300 秒 = 5 分鐘)
    fps=30,
    max_episode_steps=500,
):
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, out_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode="rgb_array", max_episode_steps=max_episode_steps)
    policies = load_policies(env, model_path, device)

    if mode == "episode":
        # ✅ 原本行為：錄 n_episodes，每集各一支 mp4
        for ep in range(n_episodes):
            print(f"\n=== Episode {ep + 1}/{n_episodes} ===")
            frames, ep_rew = run_single_episode(
                env, policies, device, max_episode_steps=max_episode_steps
            )
            avg_rew = np.mean(list(ep_rew.values()))
            print(f"Episode {ep + 1} reward per agent: {ep_rew}")
            print(f"Episode {ep + 1} avg reward: {avg_rew:.2f}")

            if frames:
                ep_video_path = (
                    video_path if n_episodes == 1
                    else video_path.replace(".mp4", f"_ep{ep+1}.mp4")
                )
                imageio.mimsave(ep_video_path, frames, fps=fps)
                print(f"✓ Saved video: {ep_video_path}")
            else:
                print("⚠ No frames captured, video not saved.")

    elif mode == "fixed_time":
        # 錄一支長度固定的影片 (例如 5 分鐘)
        target_frames = int(target_seconds * fps)
        all_frames = []
        total_rewards = []
        ep_idx = 0

        print(f"\n=== Recording fixed {target_seconds} seconds ≈ {target_frames} frames ===")

        while len(all_frames) < target_frames:
            ep_idx += 1
            print(f"--- Rolling Episode {ep_idx} ---")
            frames, ep_rew = run_single_episode(
                env, policies, device, max_episode_steps=max_episode_steps
            )
            total_rewards.append(ep_rew)

            all_frames.extend(frames)
            print(f"  collected frames: {len(all_frames)}/{target_frames}")

        # 如果多錄了一些 frame，就裁掉
        all_frames = all_frames[:target_frames]

        if all_frames:
            imageio.mimsave(video_path, all_frames, fps=fps)
            print(f"✓ Saved fixed-length video: {video_path}")
            # 簡單印一下平均回報
            avg_rews = [np.mean(list(r.values())) for r in total_rewards]
            print(f"Avg reward over {ep_idx} episodes: {np.mean(avg_rews):.2f}")
        else:
            print("⚠ No frames captured, video not saved.")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    env.close()


if __name__ == "__main__":
    # 🔹 方案 1：錄一整個 episode
    # evaluate_maddpg_nocomm(
    #     model_path="checkpoints_maddpg_nocomm/maddpg_nocomm_best.pt",
    #     out_dir="videos",
    #     out_name="maddpg_nocomm_ep1.mp4",
    #     mode="episode",
    #     n_episodes=1,
    #     fps=30,
    #     max_episode_steps=500,
    # )

    # 🔹 方案 2：錄一支固定 5 分鐘影片（可以跨好幾個 episodes）
    evaluate_maddpg_nocomm(
        model_path="checkpoints_maddpg_no_comm_ant_4x2/maddpg_nocomm_best.pt",
        out_dir="videos",
        out_name="maddpg_no_comm_ant_4x2.mp4",
        mode="fixed_time",
        target_seconds=300,   # 5 分鐘
        fps=30,
        max_episode_steps=500,
    )
