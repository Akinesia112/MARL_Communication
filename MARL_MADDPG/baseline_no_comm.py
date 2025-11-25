import torch
import torch.nn as nn
import numpy as np
from gymnasium_robotics import mamujoco_v1
from torch.distributions import Normal

class SimplePolicy(nn.Module):
    """無通信的簡單策略網絡"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        features = self.network(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

class SimpleCritic(nn.Module):
    """價值函數"""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        return self.network(obs)

# 訓練循環框架
def train_no_comm():
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="2x4",
        agent_obsk=1
    )
    
    # 為每個智能體創建策略和價值網絡
    agents = env.agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]
    
    policies = {agent: SimplePolicy(obs_dim, action_dim) for agent in agents}
    critics = {agent: SimpleCritic(obs_dim) for agent in agents}
    
    # TODO: 實現訓練循環
    print("無通信基線準備完成！")
    print(f"觀察維度: {obs_dim}, 動作維度: {action_dim}")

if __name__ == "__main__":
    train_no_comm()