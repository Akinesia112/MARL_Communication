import torch
import torch.nn as nn
from torch.distributions import Normal
from pde_comm import (
    PDECommunication, 
    PDECommunication_NoDiffusion, 
    PDECommunication_NoReaction, 
    PDECommunication_NoPDE
)

# policy_with_pde.py
class PolicyWithPDE(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, grid_size=16):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        #self.comm = PDECommunication_XXX(...) 切換模式
        self.comm = PDECommunication_NoDiffusion( 
        #self.comm = PDECommunication_NoReaction(
        #self.comm = PDECommunication_NoPDE(nn.Module):
        #self.comm = PDECommunication(
            feature_dim=hidden_dim, 
            grid_size=8,
            n_steps=2,
            dt=0.2,
            sigma=0.8
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, all_obs, agent_idx=0):
        batch_size, n_agents, obs_dim = all_obs.shape

        local_features = self.obs_encoder(obs)
        all_features = self.obs_encoder(all_obs.view(-1, obs_dim)).view(batch_size, n_agents, -1)

        comm_features, field = self.comm(agent_idx, all_features)
        
        combined = torch.cat([local_features, comm_features], dim=-1)
        policy_features = self.policy_net(combined)
        mean = self.mean_head(policy_features)
        std = torch.exp(self.log_std)

        return Normal(mean, std), field