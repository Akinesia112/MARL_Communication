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
from train_maddpg_mlp_comm import CommPolicyMLP

def make_env(render_mode="rgb_array", max_episode_steps=5000):
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="4x2",
        agent_obsk=1,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

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

    # Use the communication policy, not SimplePolicy
    policies = {
        agent: CommPolicyMLP(
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
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        print("⚠ Using untrained CommPolicyMLP policies.")
        model_name = "random_maddpg_mlp_comm"

    for agent in agents:
        policies[agent].eval()

    return policies, model_name


@torch.no_grad()
def run_episode_and_record(env, policies, device, model_name, episode_idx=0,
                           max_episode_steps=500, fps=30):
    obs_dict, _ = env.reset()
    agents = env.agents

    act_space = env.action_space(agents[0])
    action_low = torch.tensor(act_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
    max_action = torch.max(torch.abs(action_high))

    done = False
    step = 0
    episode_reward = {agent: 0.0 for agent in agents}
    frames = []

    while not done and step < max_episode_steps:
        # build all-obs tensor once per step (shape [1, N, obs_dim])
        obs_all = np.array([obs_dict[a] for a in agents], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

        actions_dict = {}

        for idx, agent in enumerate(agents):
            obs = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)

            # ---- call MADDPG comm policy ----
            out = policies[agent](obs, obs_all_tensor, self_index=idx)

            # CommPolicy often returns (dist, comm_info) or (action, comm_info)
            if isinstance(out, tuple):
                pi_out, comm_info = out
            else:
                pi_out, comm_info = out, None

            # If pi_out is a distribution (PPO-style), use its mean;
            # otherwise assume it is already a tensor of actions.
            if isinstance(pi_out, torch.distributions.Distribution):
                mu = pi_out.mean               # [1, action_dim]
            else:
                mu = pi_out                    # [1, action_dim]

            # squash + clip to env bounds
            action = torch.tanh(mu) * max_action
            action = torch.max(torch.min(action, action_high), action_low)
            actions_dict[agent] = action.cpu().numpy()[0]

        # env step as before
        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)


        for agent in agents:
            episode_reward[agent] += float(rewards[agent])

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

    if len(frames) > 0:
        os.makedirs("videos", exist_ok=True)
        out_path = os.path.join("videos", f"{model_name}_ep{episode_idx}_mlp_comm_ant_4x2.mp4")
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"✓ Saved video to {out_path}")
    else:
        print("⚠ No frames captured (env.render() returned None).")


@torch.no_grad()
def record_fixed_duration(env, policies, device, model_name,
                          total_seconds=300, fps=30, max_episode_steps=5000):
    os.makedirs("videos", exist_ok=True)
    out_path = os.path.join("videos", f"{model_name}_mlp_comm_ant_4x2.mp4")

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
            obs_all = np.array([obs_dict[a] for a in agents], dtype=np.float32)
            obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

            actions_dict = {}
            for idx, agent in enumerate(agents):
                obs = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)

                out = policies[agent](obs, obs_all_tensor, self_index=idx)
                if isinstance(out, tuple):
                    pi_out, comm_info = out
                else:
                    pi_out, comm_info = out, None

                if isinstance(pi_out, torch.distributions.Distribution):
                    mu = pi_out.mean
                else:
                    mu = pi_out

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

    checkpoint_path = "checkpoints_maddpg_mlp_comm_ant_4x2/maddpg_comm_best.pt"

    env = make_env(render_mode="rgb_array", max_episode_steps=5000)
    policies, model_name = load_policies(env, checkpoint_path, device)

    # run_episode_and_record(env, policies, device, model_name,
    #                        episode_idx=1, max_episode_steps=500, fps=30)

    record_fixed_duration(
        env,
        policies,
        device,
        model_name,
        total_seconds=300,
        fps=30,
        max_episode_steps=5000,
    )

    env.close()


if __name__ == "__main__":
    main()
