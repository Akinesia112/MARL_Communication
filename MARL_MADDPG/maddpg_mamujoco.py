#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MADDPG baseline for (MA)MuJoCo locomotion "A -> B".
- Uses MultiAgentLocomotion wrapper for N agents.
- Logs Return, AUC, wall-clock, memory.
"""

import os, time, math, psutil, argparse, random, json
os.environ.setdefault("MUJOCO_GL", "egl")        # offscreen
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# (optional) helps on some servers:
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

from collections import deque
from typing import Tuple, List, Dict, Any, Optional
import imageio.v2 as imageio
import four_ants_env
from gymnasium_robotics import mamujoco_v1 as mamujoco 

try:
    import imageio_ffmpeg, os
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Gymnasium / Mujoco
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OOM prevention for large models
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def auc(values: List[float]):
    if len(values) < 2: return float(values[-1]) if values else 0.0
    s = 0.0
    for i in range(1, len(values)):
        s += 0.5 * (values[i] + values[i-1])
    return s / (len(values)-1)

def _tile_frames(frames, tile_hw=None):
    """
    frames: list of HxWxC uint8
    tile_hw: (rows, cols). If None, choose a near-square grid.
    """
    import numpy as np, math
    assert len(frames) > 0
    H, W, C = frames[0].shape
    K = len(frames)
    if tile_hw is None:
        r = int(math.floor(math.sqrt(K)))
        c = int(math.ceil(K / r))
    else:
        r, c = tile_hw
    # pad with black frames if needed
    pad = r * c - K
    if pad > 0:
        frames = frames + [np.zeros_like(frames[0]) for _ in range(pad)]
    rows = []
    for i in range(r):
        row = np.concatenate(frames[i*c:(i+1)*c], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    return grid

# ---------------------------
# Multi-Agent Env Wrappers
# ---------------------------

class MultiAgentLocomotion(gym.Env):
    """
    Fallback multi-agent wrapper that spawns N identical robots on a plane with shared goal.
    Each agent controls its own Mujoco env instance but we run them batched to reduce overhead.
    Shared team reward: sum of per-agent progress towards (goal_x, goal_y), minus control/impact costs.
    For scalability, observations are trimmed to low-dim proprio + (goal - pos) vector.
    """
    metadata = {"render_modes": []}

    def __init__(self, task_id: str, n_agents: int = 16,
                 goal=(10.0, 0.0), episode_len=1000, seed=0, device="cpu"):
        super().__init__()
        self.task_id = task_id
        self.n_agents = n_agents
        self.goal = np.array(goal, dtype=np.float32)
        self.episode_len = episode_len
        self._step_count = 0
        self.device = device

        # Create N gym envs (headless) with consistent seeds
        base_seed = seed
        self.envs = []
        for i in range(n_agents):
            render_mode = "rgb_array"   # ensure offscreen frames are available
            try:
                # 嘗試使用 gymnasium 0.29+ 的 API
                env = gym.make(task_id, render_mode=render_mode, 
                               width=640, height=480, 
                               reward_survive=0.0, 
                               reward_forward=0.0,
                               reward_control=0.0)
            except TypeError:
                # Fallback 到舊版 API
                env = gym.make(task_id, render_mode=render_mode, width=640, height=480)

            env.reset(seed=base_seed + i)
            self.envs.append(env)

        # Build obs/act spaces by peeking one env
        obs0, _ = self.envs[0].reset()
        
        # 檢查 obs 維度
        self.raw_obs_dim = obs0.shape[0]
        
        # 27   = Ant-v5 (無接觸力)
        # 105  = Ant-v5 (有接觸力, 13 qpos + 14 qvel + 78 cfrc_ext)
        # 111  = Humanoid-v5 (有接觸力)
        if self.raw_obs_dim not in (27, 105, 111):
            # 這是合理的警告
            print(f"[警告] 偵測到非預期的觀測維度: {self.raw_obs_dim}. "
                  f"平滑度懲罰索引 (16:19) 和 (19:27) 可能不正確。")
        else:
            # 105 是 Ant-v5 的標準維度之一
            print(f"[info] 偵測到 {self.raw_obs_dim}-dim 觀測空間 (Ant-v5 標準)。")
        
        self._obs_dim = self.raw_obs_dim + 2   # goal dx, dy
        self._act_dim = self.envs[0].action_space.shape[0]
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        self.single_action_space = self.envs[0].action_space
        
        # 為了計算 action jerk，儲存 N 個 agent 的最後動作
        self._last_actions = np.zeros((n_agents, self._act_dim), dtype=np.float32)

        self.observation_space = gym.spaces.Tuple([self.single_observation_space]*n_agents)
        self.action_space = gym.spaces.Tuple([self.single_action_space]*n_agents)

        # Per-agent state (prev positions) for progress reward
        self._last_xy = np.zeros((n_agents, 2), dtype=np.float32)

    def _extract_xy(self, env, obs):
        xy = np.zeros(2, dtype=np.float32)
        try:
            # 直接從 unwrap 後的 data 獲取
            un = env.unwrapped
            data = un.data
            xy[0] = float(data.qpos[0])
            xy[1] = float(data.qpos[1])
        except Exception:
            pass # 保持 (0,0)
        return xy

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        if seed is not None:
            for i,e in enumerate(self.envs):
                e.reset(seed=seed+i)
        obs_list = []
        self._step_count = 0
        self._last_actions.fill(0.0)
        for i, env in enumerate(self.envs):
            ob, _ = env.reset()
            xy = self._extract_xy(env, ob)
            self._last_xy[i] = xy
            dxdy = self.goal - xy
            dxdy_norm = np.linalg.norm(dxdy) + 1e-6
            dxdy = dxdy / dxdy_norm
            ob_aug = np.concatenate([ob, dxdy], axis=0)
            obs_list.append(ob_aug.astype(np.float32))
        return tuple(obs_list), {}

    def step(self, actions: Tuple[np.ndarray, ...]):
        total_rew = 0.0
        obs_list, rew_list, done_list = [], [], []
        info = {}
        self._step_count += 1

        for i, env in enumerate(self.envs):
            a = np.clip(actions[i], env.action_space.low, env.action_space.high)
            # ob 是來自 env.step 的 *原始* 觀測 (105-dim)
            ob, r, terminated, truncated, inf = env.step(a)
            
            xy = self._extract_xy(env, ob)
            progress = np.linalg.norm(self.goal - self._last_xy[i]) - np.linalg.norm(self.goal - xy) # reduced distance
            ctrl_cost = 0.001 * float(np.sum(np.square(a)))
                        
            # 1. Action Jerk 懲罰 (已註解)
            # action_jerk = a - self._last_actions[i]
            # jerk_cost = 1e-5 * float(np.sum(np.square(action_jerk))) 

            # 2. 軀幹穩定性懲罰 (已註解)
            # torso_ang_vel = ob[16:19] 
            # torso_stability_cost = 1e-5 * float(np.sum(np.square(torso_ang_vel))) 

            # 3. 關節速度懲罰 (已註解)
            # joint_velocities = ob[19:27]
            # joint_vel_cost = 1e-6 * float(np.sum(np.square(joint_velocities))) 
            
            progress *= 500.0        # (可調)
            
            # 最終獎勵：只剩下「前進」和「控制成本」
            step_rew = progress - ctrl_cost

            self._last_xy[i] = xy
            self._last_actions[i] = a

            dxdy = self.goal - xy
            dxdy_norm = np.linalg.norm(dxdy) + 1e-6
            dxdy = dxdy / dxdy_norm         # new: normalize
            # ob_aug 是 augmented 觀測 (107-dim)
            ob_aug = np.concatenate([ob, dxdy], axis=0)

            obs_list.append(ob_aug.astype(np.float32))
            rew_list.append(step_rew)
            done_list.append(bool(terminated or truncated))

        # Team reward = sum of per-agent rewards
        team_rew = float(np.sum(rew_list))
        total_rew += team_rew

        terminated = all(done_list)         # 只有全部失敗才終止
        truncated  = (self._step_count >= self.episode_len)

        return tuple(obs_list), team_rew, terminated, truncated, info
    
    
    def render_frame(self, tile_hw=None):
        frames = []
        for e in self.envs:
            f = e.render()
            if f is None:
                continue
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            frames.append(f)
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return _tile_frames(frames, tile_hw)
        

    def close(self):
        for e in self.envs: e.close()


# ---------------------------
# Replay Buffer
# ---------------------------

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=int(1e6), dtype=np.float32):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.dtype = dtype
        # store on CPU as numpy arrays (much smaller with float16)
        self.obs      = np.zeros((self.capacity, obs_dim), dtype=dtype)
        self.obs_next = np.zeros((self.capacity, obs_dim), dtype=dtype)
        self.act      = np.zeros((self.capacity, act_dim), dtype=dtype)
        self.rew      = np.zeros((self.capacity, 1), dtype=np.float32)   # keep scalar in fp32
        self.done     = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, o, a, r, o2, d):
        # o, a, r, o2, d are torch tensors on device; bring to CPU np
        o   = o.detach().cpu().numpy().astype(self.dtype)
        a   = a.detach().cpu().numpy().astype(self.dtype)
        r   = r.detach().cpu().numpy().astype(np.float32)
        o2  = o2.detach().cpu().numpy().astype(self.dtype)
        d   = d.detach().cpu().numpy().astype(np.float32)
        n = o.shape[0]
        idxs = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idxs]      = o
        self.act[idxs]      = a
        self.rew[idxs]      = r
        self.obs_next[idxs] = o2
        self.done[idxs]     = d
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=(batch_size,))
        # move to torch on the fly
        obs      = torch.tensor(self.obs[idx],      dtype=torch.float32, device=device)
        act      = torch.tensor(self.act[idx],      dtype=torch.float32, device=device)
        rew      = torch.tensor(self.rew[idx],      dtype=torch.float32, device=device)
        obs_next = torch.tensor(self.obs_next[idx], dtype=torch.float32, device=device)
        done     = torch.tensor(self.done[idx],     dtype=torch.float32, device=device)
        return (obs, act, rew, obs_next, done)
    
# ---------------------------
# Networks
# ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), act=nn.ReLU, out_act=None):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), act()]
            dim = h
        layers += [nn.Linear(dim, out_dim)]
        if out_act is not None:
            layers += [out_act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(256,256)):
        super().__init__()
        self.mu = MLP(obs_dim, act_dim, hidden, act=nn.ReLU, out_act=None)
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))

    def forward(self, obs):
        x = self.mu(obs)
        x = torch.tanh(x)
        # scale to bounds
        aa = (self.act_high - self.act_low) * (x + 1.0)/2.0 + self.act_low
        return aa

class CentralCriticMean(nn.Module):
    """
    Central critic that handles variable N by mean-pooling embeddings of (obs, act) across agents.
    For agent i update, we still use shared critic (MADDPG with parameter sharing).
    Input: per-sample we receive the *team* pooled context + the local (obs_i, act_i).
    """
    def __init__(self, obs_dim, act_dim, embed_dim=128, hidden=(256,256)):
        super().__init__()
        self.enc_local = MLP(obs_dim + act_dim, embed_dim, hidden=(embed_dim,))
        self.enc_team  = MLP(obs_dim + act_dim, embed_dim, hidden=(embed_dim,))
        self.q_head    = MLP(embed_dim*2, 1, hidden=hidden)

    def forward(self, obs_i, act_i, team_obs, team_act):
        # team_obs/team_act: (B, N, D)
        B, N, Do = team_obs.shape
        Da = team_act.shape[-1]
        # Local embed
        loc = self.enc_local(torch.cat([obs_i, act_i], dim=-1))
        # Team embed (mean across agents)
        team_flat = torch.cat([team_obs, team_act], dim=-1)  # (B,N,Do+Da)
        team_emb  = self.enc_team(team_flat)                 # (B,N,E)
        team_mean = team_emb.mean(dim=1)                     # (B,E)
        q = self.q_head(torch.cat([loc, team_mean], dim=-1))
        return q.squeeze(-1)

# ---------------------------
# MADDPG Agent (parameter sharing)
# ---------------------------

class MADDPG:
    def __init__(self, obs_dim, act_dim, act_low, act_high, n_agents,
                 gamma=0.99, tau=0.005, lr_actor=3e-4, lr_critic=3e-4, device="cpu"):
        self.n_agents = n_agents
        self.device = device
        self.actor = Actor(obs_dim, act_dim, act_low, act_high).to(device)
        self.actor_targ = Actor(obs_dim, act_dim, act_low, act_high).to(device)
        self.critic = CentralCriticMean(obs_dim, act_dim).to(device)
        self.critic_targ = CentralCriticMean(obs_dim, act_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q_opt  = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs, noise_std=0.1):
        a = self.actor(obs)
        if noise_std>0:
            a = a + noise_std*torch.randn_like(a)
        return torch.clamp(a, min=self.actor.act_low, max=self.actor.act_high)

    def soft_update(self, src, dst):
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch, n_agents, act_low, act_high):
        obs, act, rew, obs2, done = batch
        B = obs.shape[0]
        # Reshape to (B, N, D)
        N = n_agents
        Do = obs.shape[-1] // N
        Da = act.shape[-1] // N
        obs = obs.view(B, N, Do)
        act = act.view(B, N, Da)
        obs2 = obs2.view(B, N, Do)

        # Current Q
        # For shared update, we randomly pick an index i per-sample for the "local" role (or average over i)
        idx_i = torch.randint(0, N, (B,), device=obs.device)
        obs_i = obs[torch.arange(B, device=obs.device), idx_i, :]
        act_i = act[torch.arange(B, device=obs.device), idx_i, :]

        q = self.critic(obs_i, act_i, obs, act)  # (B,)

        with torch.no_grad():
            # Target actions for all agents
            obs2_flat = obs2.reshape(B*N, Do)
            act2_all  = self.actor_targ(obs2_flat).view(B, N, Da)
            # Target Q for the chosen local i
            obs2_i = obs2[torch.arange(B, device=obs.device), idx_i, :]
            act2_i = act2_all[torch.arange(B, device=obs.device), idx_i, :]
            q_targ = self.critic_targ(obs2_i, act2_i, obs2, act2_all)
            y = rew.squeeze(-1) + self.gamma * (1.0 - done.squeeze(-1)) * q_targ

        # Critic loss
        q_loss = nn.MSELoss()(q, y)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q_opt.step()

        # Actor loss: maximize Q(local obs, pi(obs)) under team mean context
        obs_flat = obs.reshape(B*N, Do)
        pi_all = self.actor(obs_flat).view(B, N, Da)
        pi_i = pi_all[torch.arange(B, device=obs.device), idx_i, :]
        pi_loss = - self.critic(obs_i, pi_i, obs, pi_all).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.pi_opt.step()

        # Targets
        self.soft_update(self.actor, self.actor_targ)
        self.soft_update(self.critic, self.critic_targ)

        return dict(q_loss=float(q_loss.item()), pi_loss=float(pi_loss.item()))

# ---------------------------
# Batching helpers to pack per-agent tuples into big tensors
# ---------------------------

def pack_team(obs_list: List[np.ndarray], act_list: List[np.ndarray], rew: float, obs2_list: List[np.ndarray],
              done: bool, device, dtype=torch.float32):
    """
    Return big tensors shaped for buffer: (N*?,)
    - obs, act are concatenations of all agents, reward is team scalar repeated per-agent then averaged (here: store once).
    """
    N = len(obs_list)
    obs = np.stack(obs_list, axis=0)   # (N, Do)
    act = np.stack(act_list, axis=0)   # (N, Da)
    obs2 = np.stack(obs2_list, axis=0)
    # Store as flat vectors
    obs_t  = torch.tensor(obs, dtype=dtype, device=device).reshape(1, -1)
    act_t  = torch.tensor(act, dtype=dtype, device=device).reshape(1, -1)
    obs2_t = torch.tensor(obs2, dtype=dtype, device=device).reshape(1, -1)
    rew_t  = torch.tensor([[rew]], dtype=dtype, device=device)
    done_t = torch.tensor([[1.0 if done else 0.0]], dtype=dtype, device=device)
    return obs_t, act_t, rew_t, obs2_t, done_t

# ---------------------------
# Training / Evaluation
# ---------------------------

def build_env(args, device):
 
    # --- 您的指令會直接執行這段 (Fallback) ---
    render_mode = "rgb_array" if getattr(args, "rgb_render", 0) else None
    
    env = MultiAgentLocomotion(task_id=args.task, n_agents=args.n_agents,
                               goal=(args.goal_x, args.goal_y),
                               episode_len=args.episode_len,
                               seed=args.seed, device=device)

    o, _ = env.reset(seed=args.seed)
    obs_dim = len(o[0])
    act_dim = env.action_space[0].shape[0]
    act_low = env.action_space[0].low
    act_high = env.action_space[0].high
    return env, obs_dim, act_dim, act_low, act_high, args.n_agents
    
def plot_training_curve(jsonl_path: str, out_png: str):
    steps, ep_ret, roll_ret = [], [], []
    if not os.path.isfile(jsonl_path):
        print(f"[plot] no log found at {jsonl_path}")
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                steps.append(rec.get("step", len(steps)))
                ep_ret.append(rec.get("episode_return", None))
                roll_ret.append(rec.get("rolling_return_mean", None))
            except Exception:
                pass
    if len(steps) == 0:
        print("[plot] log is empty")
        return

    plt.figure(figsize=(8,4.5))
    # Plot episode returns (sparse but raw)
    plt.plot(steps, ep_ret, label="Episode Return", linewidth=1.0)
    # Plot rolling mean return
    plt.plot(steps, roll_ret, label="Rolling Return (last 100)", linewidth=1.5)
    plt.xlabel("Environment Steps")
    plt.ylabel("Return")
    plt.title("Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[plot] saved: {out_png}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu")
    set_seed(args.seed)

    env, obs_dim, act_dim, act_low, act_high, n_agents = build_env(args, device)

    # Shared buffer stores concatenated team obs/act
    buf = ReplayBuffer(obs_dim*n_agents, act_dim*n_agents, capacity=args.replay_size)

    maddpg = MADDPG(obs_dim, act_dim, act_low, act_high, n_agents,
                    gamma=args.gamma, tau=args.tau,
                    lr_actor=args.lr_actor, lr_critic=args.lr_critic, device=device)

    print(f"[info] Actor params: {count_params(maddpg.actor):,} | Critic params: {count_params(maddpg.critic):,}")

    # Logging
    run_dir = os.path.join("runs", f"{args.task}_N{n_agents}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    ep_returns = []
    ep_lengths = []
    rolling_returns = deque(maxlen=100)
    best_auc = -1e9

    obs, _ = env.reset(seed=args.seed)
    wall_start = time.time()
    step = 0
    episode_ret = 0.0
    episode_len = 0

    # Warmup action noise
    warmup_steps = max(5000, args.batch_size*4)
    act_noise = args.exploration_std

    with open(log_path, "a", encoding="utf-8") as fout:
        while step < args.train_steps:
            # Pack obs and choose actions for all agents
            obs_np = np.stack(obs, axis=0)  # (N, Do)
            obs_t = to_t(obs_np, device)
            N = obs_t.shape[0]

            with torch.no_grad():
                if step < warmup_steps:
                    low  = maddpg.actor.act_low          # (act_dim,)
                    high = maddpg.actor.act_high         # (act_dim,)
                    # 依每一維上下界抽樣： low + (high-low) * U[0,1]
                    rand_u = torch.rand((N, maddpg.act_dim), device=obs_t.device, dtype=obs_t.dtype)
                    acts_t = low + (high - low) * rand_u
                else:
                    acts_t = maddpg.act(obs_t, noise_std=act_noise)

            acts_np = acts_t.detach().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(tuple(acts_np))
            done = terminated or truncated
            
            # Store transition (team-level)
            next_obs_np = np.stack(next_obs, axis=0)
            obs_t_flat, act_t_flat, rew_t, next_obs_t_flat, done_t = pack_team(
                [o for o in obs], [a for a in acts_np], reward,
                [no for no in next_obs],
                done, device
            )
            buf.add(obs_t_flat, act_t_flat, rew_t, next_obs_t_flat, done_t)

            episode_ret += reward
            episode_len += 1
            step += 1

            # Update
            if buf.size >= args.batch_size and step % args.update_every == 0:
                for _ in range(args.gradient_steps):                    
                    batch = buf.sample(args.batch_size, device)
                    stats = maddpg.update(batch, n_agents, act_low, act_high)

            # End of episode
            if done or (episode_len >= args.episode_len):
                ep_returns.append(episode_ret)
                ep_lengths.append(episode_len)
                rolling_returns.append(episode_ret)
                # Logging
                mem = psutil.Process(os.getpid()).memory_info().rss
                rec = {
                    "step": step,
                    "episode_return": episode_ret,
                    "episode_len": episode_len,
                    "rolling_return_mean": float(np.mean(rolling_returns)) if len(rolling_returns)>0 else episode_ret,
                    "wall_time_s": time.time() - wall_start,
                    "rss_bytes": mem
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()

                # Track best AUC
                r_hist = list(rolling_returns)
                cur_auc = auc(r_hist)
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    torch.save({
                        "actor": maddpg.actor.state_dict(),
                        "critic": maddpg.critic.state_dict(),
                        "obs_dim": int(obs_dim),
                        "act_dim": int(act_dim),
                        "n_agents": int(n_agents),
                        # store as lists to avoid numpy pickling
                        "act_low": np.asarray(act_low, dtype=np.float32).tolist(),
                        "act_high": np.asarray(act_high, dtype=np.float32).tolist(),
                    }, ckpt_path)

                # Reset
                obs, _ = env.reset()
                episode_ret = 0.0
                episode_len = 0

            # Anneal exploration
            act_noise = max(args.min_exploration_std, act_noise * args.exploration_decay)

    env.close()
    plot_training_curve(log_path, os.path.join(run_dir, "train_curve.png"))
    print(f"[done] Training finished. Logs at {log_path} | ckpt at {ckpt_path}")

def load_ckpt(path, device):
    import numpy as np
    import torch

    # 1) Try safe weights-only load (allowlist numpy reconstruct)
    try:
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
            ck = torch.load(path, map_location=device, weights_only=True)
            return ck
    except Exception:
        pass

    # 2) Fallback: explicit unsafe load (OK if you trust the file)
    # NOTE: This mimics PyTorch <2.6 behavior.
    return torch.load(path, map_location=device, weights_only=False)

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu")
    ck = load_ckpt(args.load_ckpt, device)

    act_low = torch.tensor(ck.get("act_low"), dtype=torch.float32, device=device)
    act_high = torch.tensor(ck.get("act_high"), dtype=torch.float32, device=device)

    Ns = args.generalize_N if len(args.generalize_N) > 0 else [ck["n_agents"]]
    Tasks = args.generalize_tasks if len(args.generalize_tasks) > 0 else [args.task]

    results = []
    for task in Tasks:
        for N in Ns:
            env, obs_dim, act_dim, act_low, act_high, n_agents_actual = build_env(
                argparse.Namespace(
                    # --- [已刪除] MaMuJoCo args ---
                    task=task, n_agents=N, goal_x=args.goal_x, goal_y=args.goal_y,
                    episode_len=args.episode_len, seed=args.seed,
                    # 補上 build_env 需要的 arg
                    rgb_render=1 
                ), device
            )

            actor = Actor(obs_dim, act_dim, act_low, act_high).to(device)
            try: actor.load_state_dict(ck["actor"])
            except Exception: pass
            actor.eval()

            video_dir = os.path.join(os.path.dirname(args.load_ckpt), "videos")
            os.makedirs(video_dir, exist_ok=True)
            mp4_path = os.path.join(video_dir, f"{task}_N{N}.mp4")

            writer = None
            frames_written = 0
            returns = []

            try:
                for ep in range(args.eval_episodes):
                    obs, _ = env.reset(seed=args.seed + ep)
                    ep_ret = 0.0
                    N_eff = len(obs) 

                    try:
                        _ = env.render_frame() if hasattr(env, "render_frame") else env.render()
                    except Exception:
                        pass
                    
                    for t in range(args.episode_len):
                        with torch.no_grad():
                            obs_np = np.stack(obs, axis=0)
                            obs_t = to_t(obs_np, device)
                            act_t = actor(obs_t)
                            acts_np = act_t.cpu().numpy()

                        next_obs, reward, terminated, truncated, info = env.step(tuple(acts_np))
                        ep_ret += reward

                        # --- [已刪除] Saliency calculation ---

                        frame = env.render_frame() if hasattr(env, "render_frame") else None
                        if frame is None:
                            try: frame = env.render()
                            except Exception: frame = None
                        if frame is not None:
                            if frame.dtype != np.uint8: frame = np.clip(frame, 0, 255).astype(np.uint8)
                            if writer is None:
                                writer = imageio.get_writer(mp4_path, fps=30, format="FFMPEG", codec="libx264")
                            writer.append_data(frame)
                            frames_written += 1

                        obs = next_obs
                        if terminated: break

                    returns.append(ep_ret)
            finally:
                if writer is not None: writer.close()
                if frames_written == 0 and os.path.exists(mp4_path):
                    try: os.remove(mp4_path)
                    except Exception: pass
                print(f"[video] saved: {mp4_path if frames_written>0 else '(no frames captured)'}")
                env.close()

            rec = {
                "task": task, "N": int(N_eff),
                "return_mean": float(np.mean(returns)) if returns else 0.0,
                "return_std": float(np.std(returns)) if returns else 0.0,
                "AUC": auc(returns) if returns else 0.0,
            }
            results.append(rec)
            print(json.dumps(rec))

    outdir = os.path.dirname(args.load_ckpt)
    with open(os.path.join(outdir, "eval_generalization.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] Evaluation written to {os.path.join(outdir, 'eval_generalization.json')}")


# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="Ant-v5", help="Gymnasium mujoco task id")
    # --- [已刪除] MaMuJoCo args ---
    p.add_argument("--n_agents", type=int, default=128)
    p.add_argument("--goal_x", type=float, default=10.0)
    p.add_argument("--goal_y", type=float, default=0.0)
    p.add_argument("--episode_len", type=int, default=1000)
    p.add_argument("--train_steps", type=int, default=1_000_000)
    p.add_argument("--replay_size", type=int, default=2_000_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--update_every", type=int, default=100)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--exploration_std", type=float, default=0.2)
    p.add_argument("--min_exploration_std", type=float, default=0.02)
    p.add_argument("--exploration_decay", type=float, default=0.9995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rgb_render", type=int, default=1, help="use rgb_array render mode in fallback env")
    p.add_argument("--plot_only", type=str, default="",
               help="Path to a train_log.jsonl to plot (writes train_curve.png next to it)")

    # Eval / generalization
    p.add_argument("--eval_only", type=int, default=0)
    p.add_argument("--load_ckpt", type=str, default="")
    p.add_argument("--generalize_N", type=int, nargs="*", default=[])
    p.add_argument("--generalize_tasks", type=str, nargs="*", default=[])
    p.add_argument("--eval_episodes", type=int, default=5)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.plot_only:
        log_path = args.plot_only
        out_png = os.path.join(os.path.dirname(log_path), "train_curve.png")
        plot_training_curve(log_path, out_png)
    elif not args.eval_only:
        train(args)
    else:
        evaluate(args)