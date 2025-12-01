#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 必須在 import gymnasium 前設定，避免走 X11
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np
import torch
from gymnasium_robotics import mamujoco_v1
from gnn_policies import GCNPolicyNoComm
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

    example_agent = agents[0]
    obs_dim = env.observation_space(example_agent).shape[0]
    action_dim = env.action_space(example_agent).shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policies = {
        agent: GCNPolicyNoComm(obs_dim, action_dim).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_gcn_no_comm_ant_4x2/ppo_gcn_no_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        policies[agent].load_state_dict(ckpt[agent])
        policies[agent].eval()

    print(f"✓ Loaded PPO-GCN No-Comm model from {model_path}")
    return policies


@torch.no_grad()
def run_single_episode(env, policies, device, max_episode_steps=500):
    """跑一個 episode，回傳 frames + reward dict"""
    obs_dict, _ = env.reset()
    agents = env.agents

    episode_reward = {agent: 0.0 for agent in agents}
    frames = []
    step_in_ep = 0

    while step_in_ep < max_episode_steps:
        actions_dict = {}
        for agent in agents:
            obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(device)
            dist = policies[agent](obs)
            # deterministic: 用 mean，比 sample 穩定一點
            action = dist.mean
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


def evaluate_ppo_nocomm(
    model_path="checkpoints_ppo_gcn_no_comm_ant_4x2/ppo_gcn_no_comm_best.pt",
    out_dir="videos",
    out_name="ppo_gcn_no_comm_ant_4x2.mp4",
    mode="fixed_time",
    target_seconds=300,   # 5 分鐘
    fps=30,
    max_episode_steps=500,
):
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, out_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode="rgb_array", max_episode_steps=max_episode_steps)
    policies = load_policies(env, model_path, device)

    if mode == "fixed_time":
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

            if len(frames) == 0:
                # safety: avoid infinite loop if render 失敗
                break

        all_frames = all_frames[:target_frames]

        if all_frames:
            imageio.mimsave(video_path, all_frames, fps=fps)
            print(f"✓ Saved fixed-length video: {video_path}")
            avg_rews = [np.mean(list(r.values())) for r in total_rewards]
            if len(avg_rews) > 0:
                print(f"Avg reward over {ep_idx} episodes: {np.mean(avg_rews):.2f}")
        else:
            print("⚠ No frames captured, video not saved.")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    env.close()


if __name__ == "__main__":
    evaluate_ppo_nocomm(
        model_path="checkpoints_ppo_gcn_no_comm_ant_4x2/ppo_gcn_no_comm_best.pt",
        out_dir="videos",
        out_name="ppo_gcn_no_comm_ant_4x2.mp4",
        mode="fixed_time",
        target_seconds=300,
        fps=30,
        max_episode_steps=500,
    )
