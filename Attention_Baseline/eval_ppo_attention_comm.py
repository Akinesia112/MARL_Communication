#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ✅ 先設定 EGL / OpenGL，避免走 X11
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np
import torch
import imageio.v2 as imageio
from gymnasium_robotics import mamujoco_v1

from attention_emergent_comm import CommPolicyWithAttention


def make_env(render_mode="rgb_array", max_episode_steps=500):
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="4x2",
        agent_obsk=1,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    # 如果支援 camera config，就設定一個上帝視角
    if hasattr(env.unwrapped, "mujoco_renderer"):
        cam_cfg = {
            "distance": 8.0,
            "azimuth": 90.0,
            "elevation": -30.0,
            "lookat": [0, 0, 0.5],
        }
        try:
            env.unwrapped.mujoco_renderer.default_cam_config = cam_cfg
        except Exception:
            pass

    return env


def load_policies(env, checkpoint_path, device):
    agents = env.agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    # 建立同設定的 policy
    policies = {
        agent: CommPolicyWithAttention(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        for agent in agents:
            policies[agent].load_state_dict(ckpt[agent])
            policies[agent].eval()
        print(f"✓ Loaded PPO Attention-Comm model from {checkpoint_path}")
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        print("⚠ Using untrained random policies.")
        model_name = "random_attention_comm"

    return policies, model_name


@torch.no_grad()
def run_single_episode(env, policies, device, max_episode_steps=500):
    """
    跑一個 episode，回傳 (frames, reward_dict)
    動作用 policy 的 mean（訓練時是 sample，如果你想完全一致可以改成 sample）
    """
    obs_dict, _ = env.reset()
    agents = env.agents

    episode_reward = {agent: 0.0 for agent in agents}
    frames = []
    step_in_ep = 0

    while step_in_ep < max_episode_steps:
        # all_obs 給注意力 / comm 用
        all_obs_np = np.array([obs_dict[a] for a in agents], dtype=np.float32)
        all_obs = torch.from_numpy(all_obs_np).unsqueeze(0).to(device)  # [1, n_agents, obs_dim]

        actions_dict = {}
        for idx, agent in enumerate(agents):
            obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(device)
            dist, _ = policies[agent](obs, all_obs)  # 你的 forward 是 (obs, all_obs)
            mu = dist.mean
            action = mu  # 不做 tanh/clip，跟訓練一樣類型輸出
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


def evaluate_ppo_attention_comm(
    checkpoint_path="checkpoints_ppo_attention_comm_ant_4x2/model_iter_best.pt",
    out_dir="videos",
    out_name="ppo_attention_comm_ant_4x2.mp4",
    target_seconds=300,   # 5 分鐘
    fps=30,
    max_episode_steps=500,
):
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, out_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode="rgb_array", max_episode_steps=max_episode_steps)
    policies, model_name = load_policies(env, checkpoint_path, device)

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
            # 保險：如果 render 出來是空的就 break，避免死 loop
            print("⚠ No frames in this episode; breaking.")
            break

    # 裁到剛好 target_frames
    all_frames = all_frames[:target_frames]

    if all_frames:
        imageio.mimsave(video_path, all_frames, fps=fps)
        print(f"✓ Saved fixed-length video: {video_path}")
        avg_rews = [np.mean(list(r.values())) for r in total_rewards]
        print(f"Avg reward over {ep_idx} episodes: {np.mean(avg_rews):.2f}")
    else:
        print("⚠ No frames captured, video not saved.")

    env.close()


if __name__ == "__main__":
    evaluate_ppo_attention_comm(
        checkpoint_path="checkpoints_ppo_attention_comm_ant_4x2/model_iter_best.pt",
        out_dir="videos",
        out_name="ppo_attention_comm_ant_4x2.mp4",
        target_seconds=300,
        fps=30,
        max_episode_steps=500,
    )
