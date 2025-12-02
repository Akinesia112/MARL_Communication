# gnn_policies.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------
# 1) GCN: no-comm 
# ---------------------------------------------------------
class GCNPolicyNoComm(nn.Module):
    """
    Self-only policy:
    obs -> MLP -> Gaussian(action)
    只是名字叫 GCN，方便和 comm 版本對齊。
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        obs: [B, obs_dim]
        return: Normal(mu, std)  (for PPO)
        """
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std.expand_as(mu)
        std = log_std.exp()
        return Normal(mu, std)


# ---------------------------------------------------------
# 2) GCN: comm 
#    forward(obs, all_obs, self_index) -> (dist, extra)
# ---------------------------------------------------------
class GCNPolicyComm(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 log_std_init: float = -0.5):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.nei_lin = nn.Linear(hidden_dim, hidden_dim)

        self.policy_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self,
                obs: torch.Tensor,
                all_obs: torch.Tensor,
                self_index: int) -> Tuple[Normal, Dict[str, Any]]:
        """
        obs:      [B, obs_dim]         (對應 self_index 這個 agent)
        all_obs:  [B, n_agents, obs_dim]
        """
        # encode all agents
        h_all = self.obs_encoder(all_obs)          # [B, N, H]
        h_self = h_all[:, self_index, :]           # [B, H]

        if self.n_agents > 1:
            B, N, H = h_all.shape
            mask = torch.ones(B, N, device=h_all.device, dtype=torch.bool)
            mask[:, self_index] = 0
            nei_sum = (h_all * mask.unsqueeze(-1)).sum(dim=1)              # [B, H]
            nei_cnt = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)           # [B, 1]
            h_nei = nei_sum / nei_cnt
        else:
            h_nei = torch.zeros_like(h_self)

        h = self.self_lin(h_self) + self.nei_lin(h_nei)
        h = F.relu(h)
        h = self.policy_net(h)

        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        std = log_std.exp()
        dist = Normal(mu, std)

        extra = {
            "agg_feat": h.detach(),   # 給 debug 用，eval 不一定會用到
        }
        return dist, extra


# ---------------------------------------------------------
# 3) GAT: no-comm 版本（介面= SimplePolicy）
# ---------------------------------------------------------
class GATPolicyNoComm(nn.Module):
    """
    Self-only MLP，跟 GCNPolicyNoComm 幾乎一樣，
    主因是 PPO_no_comm script 只吃 obs -> dist 介面。
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> Normal:
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std.expand_as(mu)
        std = log_std.exp()
        return Normal(mu, std)


# ---------------------------------------------------------
# 4) GAT: comm 版本（multi-head attention over agents）
#    forward(obs, all_obs, self_index) -> (dist, extra)
#    extra["att_weights"]: [B, N]
# ---------------------------------------------------------
class GATPolicyComm(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 n_heads: int = 1,
                 log_std_init: float = -0.5):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim 必須可以整除 n_heads"
        self.head_dim = hidden_dim // n_heads

        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.policy_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self,
                obs: torch.Tensor,
                all_obs: torch.Tensor,
                self_index: int) -> Tuple[Normal, Dict[str, Any]]:
        """
        obs:      [B, obs_dim]
        all_obs:  [B, N, obs_dim]
        """
        h_all = self.obs_encoder(all_obs)          # [B, N, H]
        B, N, H = h_all.shape

        Q = self.query(h_all).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, Hh, N, d]
        K = self.key(h_all).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)    # [B, Hh, N, d]
        V = self.value(h_all).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, Hh, N, d]

        # 取出 self agent 的 query
        q = Q[:, :, self_index:self_index + 1, :]          # [B, Hh, 1, d]
        # 注意力分數： q * k^T
        att_logits = (q * K).sum(-1) / math.sqrt(self.head_dim)     # [B, Hh, N]
        att_weights = torch.softmax(att_logits, dim=-1)             # [B, Hh, N]

        # context = sum_j alpha_j v_j
        # att_weights: [B, Hh, N]
        # V:           [B, Hh, N, d]
        # → 擴展一個最後維度，做逐元素乘法後在 agent 維 (dim=2) 上 sum
        context = (att_weights.unsqueeze(-1) * V).sum(dim=2)  # [B, Hh, d]
        context = context.reshape(B, -1)  # [B, H] = [B, hidden_dim]

        h_self = h_all[:, self_index, :]             # [B, H]
        h_cat = torch.cat([h_self, context], dim=-1) # [B, 2H]

        h = self.policy_net(h_cat)
        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        std = log_std.exp()
        dist = Normal(mu, std)

        # 將 multi-head weight average 成 [B, N]
        att_mean = att_weights.mean(dim=1)           # [B, N]
        extra = {
            "att_weights": att_mean.detach(),
        }
        return dist, extra

