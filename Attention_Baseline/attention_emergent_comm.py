# attention_emergent_comm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def gumbel_softmax_sample(logits, tau=1.0, hard=True):
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + gumbel) / tau, dim=-1)

    if hard:
        shape = y.size()
        _, k = y.max(-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, k.view(-1, 1), 1.0)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y


class DiscreteMessageHead(nn.Module):
    def __init__(self, feature_dim, msg_len=4, vocab_size=8, hidden_dim=128):
        super().__init__()
        self.msg_len = msg_len
        self.vocab_size = vocab_size

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, msg_len * vocab_size),
        )
        self.msg_embedding = nn.Parameter(
            torch.randn(vocab_size, feature_dim) * 0.1
        )  # [V, D]

    def forward(self, features, tau=1.0, hard=True):
        """
        features: [B, D]
        """
        B, D = features.shape
        logits = self.fc(features)                         # [B, L*V]
        logits = logits.view(B, self.msg_len, self.vocab_size)  # [B, L, V]

        msg_probs = gumbel_softmax_sample(logits, tau=tau, hard=hard)  # [B, L, V]
        msg_emb = msg_probs @ self.msg_embedding                      # [B, L, D]
        msg_vec = msg_emb.mean(dim=1)                                 # [B, D]

        with torch.no_grad():
            msg_tokens = logits.argmax(dim=-1)                        # [B, L]

        return msg_vec, msg_probs, msg_tokens


class AttentionCommChannel(nn.Module):
    def __init__(self, feature_dim, comm_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.comm_dim = comm_dim or feature_dim

        self.query = nn.Linear(feature_dim, self.comm_dim)
        self.key   = nn.Linear(feature_dim, self.comm_dim)
        self.value = nn.Linear(feature_dim, self.comm_dim)

        self.out_proj = nn.Linear(self.comm_dim, feature_dim)

    def forward(self, local_features, all_msg_vecs):
        """
        local_features: [B, D]
        all_msg_vecs:   [B, n_agents, D]
        """
        B, n_agents, D = all_msg_vecs.shape

        Q = self.query(local_features).unsqueeze(1)        # [B, 1, C]
        K = self.key(all_msg_vecs)                         # [B, n_agents, C]
        V = self.value(all_msg_vecs)                       # [B, n_agents, C]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.comm_dim ** 0.5)
        att_weights = F.softmax(scores, dim=-1)            # [B, 1, n_agents]

        comm = torch.matmul(att_weights, V).squeeze(1)     # [B, C]
        out = self.out_proj(comm)                          # [B, D]

        return out, att_weights.squeeze(1)                 # [B, n_agents]


class CommPolicyWithAttention(nn.Module):
    """
    - 每個 policy 只關心「自己」的 msg_probs / msg_tokens
    - 仍然會用所有 agents 的 message 做 attention
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=256,
        msg_len=4,
        vocab_size=8,
        comm_tau=1.0,
        comm_hard=True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.msg_len = msg_len
        self.vocab_size = vocab_size
        self.comm_tau = comm_tau
        self.comm_hard = comm_hard

        # local obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # message head
        self.msg_head = DiscreteMessageHead(
            feature_dim=hidden_dim,
            msg_len=msg_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )

        # comm channel
        self.comm_channel = AttentionCommChannel(feature_dim=hidden_dim)

        # policy head
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, all_obs, self_index: int = 0):
        """
        Args:
            obs:       [B, obs_dim]             # 自己的觀察
            all_obs:   [B, n_agents, obs_dim]   # 所有 agents 的觀察（固定順序）
            self_index: int                     # 在 all_obs 裡自己的 index
        """
        B, n_agents, obs_dim = all_obs.shape
        assert obs_dim == self.obs_dim

        # encode local obs
        local_feat = self.obs_encoder(obs)                # [B, H]

        # encode all obs
        all_obs_flat = all_obs.view(B * n_agents, obs_dim)
        all_feat_flat = self.obs_encoder(all_obs_flat)    # [B*n_agents, H]
        all_feat = all_feat_flat.view(B, n_agents, self.hidden_dim)  # [B, n_agents, H]

        # generate messages for all agents
        msg_vecs = []
        self_msg_probs = None
        self_msg_tokens = None

        for i in range(n_agents):
            feat_i = all_feat[:, i, :]   # [B, H]
            msg_vec_i, msg_probs_i, msg_tokens_i = self.msg_head(
                feat_i, tau=self.comm_tau, hard=self.comm_hard
            )
            msg_vecs.append(msg_vec_i)   # [B, H]

            if i == self_index:
                # 只記錄自己的 message 資訊，之後給 loss 用
                self_msg_probs = msg_probs_i      # [B, L, V]
                self_msg_tokens = msg_tokens_i    # [B, L]

        # stack for attention
        all_msg_vecs = torch.stack(msg_vecs, dim=1)       # [B, n_agents, H]

        # communication: attend over all messages
        comm_feat, att_weights = self.comm_channel(local_feat, all_msg_vecs)  # [B,H], [B,n_agents]

        # policy head
        combined = torch.cat([local_feat, comm_feat], dim=-1)   # [B, 2H]
        policy_features = self.policy_net(combined)             # [B, H]

        mean = self.mean_head(policy_features)                  # [B, action_dim]
        std = torch.exp(self.log_std).unsqueeze(0)              # [1, action_dim]

        dist = Normal(mean, std)

        comm_info = {
            "att_weights": att_weights,          # [B, n_agents]
            "msg_probs": self_msg_probs,         # [B, L, V] -> 只自己的
            "msg_tokens": self_msg_tokens,       # [B, L]
        }
        return dist, comm_info
