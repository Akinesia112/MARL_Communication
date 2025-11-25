#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ---- MuJoCo / OpenGL: offscreen rendering ----
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np
import torch
import imageio.v2 as imageio

from gymnasium_robotics import mamujoco_v1
from attention_emergent_comm import CommPolicyWithAttention


def make_env(render_mode="rgb_array", max_episode_steps=5000):
    """建立 MAMuJoCo Ant 環境（max_episode_steps 調大一點，比較不容易早死）"""
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="2x4",
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

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        for agent in agents:
            policies[agent].load_state_dict(ckpt[agent])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        print("⚠ Using untrained random policies.")
        model_name = "random_comm"

    for agent in agents:
        policies[agent].to(device)
        policies[agent].eval()

    return policies, model_name


@torch.no_grad()
def run_episode_and_record(env, policies, device, model_name, episode_idx=0,
                           max_episode_steps=500, fps=30):
    """單一 episode 錄影（保留原本版本，方便 debug 用）"""
    obs_dict, _ = env.reset()
    agents = env.agents

    # action 範圍
    act_space = env.action_space(agents[0])
    action_low = torch.tensor(act_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
    max_action = torch.max(torch.abs(action_high))

    done = False
    step = 0
    episode_reward = {agent: 0.0 for agent in agents}
    frames = []

    while not done and step < max_episode_steps:
        # 準備 all_obs 給 communication 用
        all_obs_np = np.array([obs_dict[a] for a in agents], dtype=np.float32)
        all_obs = torch.from_numpy(all_obs_np).unsqueeze(0).to(device)  # [1, n_agents, obs_dim]

        actions_dict = {}

        for idx, agent in enumerate(agents):
            obs = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)  # [1, obs_dim]
            dist, _ = policies[agent](obs, all_obs, self_index=idx)
            mu = dist.mean
            action = torch.tanh(mu) * max_action
            action = torch.max(torch.min(action, action_high), action_low)
            actions_dict[agent] = action.cpu().numpy()[0]

        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

        for agent in agents:
            episode_reward[agent] += float(rewards[agent])

        # render 為 rgb_array
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        step += 1
        if any(terms.values()) or any(truncs.values()):
            done = True
        else:
            obs_dict = next_obs

    avg_rew = sum(episode_reward.values()) / len(agents)
    print(f"Episode {episode_idx}: steps={step}, reward={episode_reward}, avg={avg_rew:.2f}")

    # 存成 mp4
    if len(frames) > 0:
        os.makedirs("videos", exist_ok=True)
        out_path = os.path.join("videos", f"{model_name}_ep{episode_idx}.mp4")
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"✓ Saved video to {out_path}")
    else:
        print("⚠ No frames captured (env.render() returned None).")


@torch.no_grad()
def record_fixed_duration(env, policies, device, model_name,
                          total_seconds=300, fps=30, max_episode_steps=5000):
    """
    把多個 episode 串成一支固定時長（例如 5 分鐘）的影片。

    - total_seconds: 影片長度（秒），預設 300 s = 5 min
    - fps:           frame rate
    - max_episode_steps: 單一 episode 的最多步數（避免卡死）
    """
    os.makedirs("videos", exist_ok=True)
    out_path = os.path.join("videos", f"{model_name}_fixed_{total_seconds}s.mp4")

    # 先 reset 一次，取得 agents / action space
    obs_dict, _ = env.reset()
    agents = env.agents

    act_space = env.action_space(agents[0])
    action_low = torch.tensor(act_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
    max_action = torch.max(torch.abs(action_high))

    target_frames = int(total_seconds * fps)
    frame_count = 0
    episode_idx = 0
    step_in_ep = 0
    episode_reward = {agent: 0.0 for agent in agents}

    print(f"▶ Start recording fixed duration: {total_seconds}s, target_frames={target_frames}")

    with imageio.get_writer(out_path, fps=fps) as writer:
        while frame_count < target_frames:
            # 準備 all_obs 給 communication 用
            all_obs_np = np.array([obs_dict[a] for a in agents], dtype=np.float32)
            all_obs = torch.from_numpy(all_obs_np).unsqueeze(0).to(device)

            actions_dict = {}
            for idx, agent in enumerate(agents):
                obs = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
                dist, _ = policies[agent](obs, all_obs, self_index=idx)
                mu = dist.mean
                action = torch.tanh(mu) * max_action
                action = torch.max(torch.min(action, action_high), action_low)
                actions_dict[agent] = action.cpu().numpy()[0]

            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            for agent in agents:
                episode_reward[agent] += float(rewards[agent])

            frame = env.render()
            if frame is not None:
                writer.append_data(frame)
                frame_count += 1

            step_in_ep += 1

            done = any(terms.values()) or any(truncs.values()) or (step_in_ep >= max_episode_steps)

            if done:
                avg_rew = sum(episode_reward.values()) / len(agents)
                print(f"  Episode {episode_idx}: steps={step_in_ep}, avg_rew={avg_rew:.2f}")
                # reset 下一個 episode
                obs_dict, _ = env.reset()
                agents = env.agents
                episode_idx += 1
                step_in_ep = 0
                episode_reward = {agent: 0.0 for agent in agents}
            else:
                obs_dict = next_obs

    print(f"✓ Saved fixed-duration video to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 這裡改成你要的 checkpoint 名稱（注意：這是 MADDPG 的 emergent comm 版本路徑）
    checkpoint_path = "checkpoints_maddpg_attention_comm/maddpg_attention_comm_best.pt"

    # 評估環境：max_episode_steps 調大一點，避免太快 truncate
    env = make_env(render_mode="rgb_array", max_episode_steps=5000)
    policies, model_name = load_policies(env, checkpoint_path, device)

    # （可選）先錄一個單 episode 確認 OK
    # run_episode_and_record(env, policies, device, model_name,
    #                        episode_idx=1, max_episode_steps=500, fps=30)

    # 錄一支 5 分鐘影片
    record_fixed_duration(
        env,
        policies,
        device,
        model_name,
        total_seconds=300,   # 5 min
        fps=30,
        max_episode_steps=5000,
    )

    env.close()


if __name__ == "__main__":
    main()
