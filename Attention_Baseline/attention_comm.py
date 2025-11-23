import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCommunication(nn.Module):
    """基於注意力的通信模塊"""
    def __init__(self, feature_dim, comm_dim, n_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.comm_dim = comm_dim
        self.n_heads = n_heads
        
        # 多頭注意力
        self.query = nn.Linear(feature_dim, comm_dim)
        self.key = nn.Linear(feature_dim, comm_dim)
        self.value = nn.Linear(feature_dim, comm_dim)
        
        self.out_proj = nn.Linear(comm_dim, feature_dim)
        
    def forward(self, local_features, all_features):
        """
        Args:
            local_features: [batch, feature_dim] - 自己的特徵
            all_features: [batch, n_agents, feature_dim] - 所有智能體的特徵
        Returns:
            aggregated_features: [batch, feature_dim]
        """
        batch_size, n_agents, _ = all_features.shape
        
        # 計算 Q, K, V
        Q = self.query(local_features).unsqueeze(1)  # [batch, 1, comm_dim]
        K = self.key(all_features)                    # [batch, n_agents, comm_dim]
        V = self.value(all_features)                  # [batch, n_agents, comm_dim]
        
        # 注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.comm_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, 1, n_agents]
        
        # 聚合信息
        comm_message = torch.matmul(attention_weights, V).squeeze(1)  # [batch, comm_dim]
        
        # 輸出投影
        output = self.out_proj(comm_message)  # [batch, feature_dim]
        
        return output, attention_weights.squeeze(1)


class PolicyWithAttention(nn.Module):
    """帶注意力通信的策略網絡"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, comm_dim=128):
        super().__init__()
        
        # 觀察編碼器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 通信模塊
        self.comm = AttentionCommunication(hidden_dim, comm_dim)
        
        # 策略頭 (結合本地特徵和通信信息)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 本地 + 通信
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs, all_obs):
        """
        Args:
            obs: [batch, obs_dim] - 自己的觀察
            all_obs: [batch, n_agents, obs_dim] - 所有智能體的觀察
        """
        from torch.distributions import Normal
        
        # 編碼本地觀察
        local_features = self.obs_encoder(obs)
        
        # 編碼所有觀察
        batch_size, n_agents, obs_dim = all_obs.shape
        all_features = self.obs_encoder(all_obs.view(-1, obs_dim))
        all_features = all_features.view(batch_size, n_agents, -1)
        
        # 通信
        comm_features, attention_weights = self.comm(local_features, all_features)
        
        # 結合本地和通信特徵
        combined = torch.cat([local_features, comm_features], dim=-1)
        policy_features = self.policy_net(combined)
        
        # 輸出動作分佈
        mean = self.mean_head(policy_features)
        std = torch.exp(self.log_std)
        
        return Normal(mean, std), attention_weights


# 測試
if __name__ == "__main__":
    batch_size = 32
    n_agents = 2
    obs_dim = 50
    action_dim = 8
    
    # 創建模型
    policy = PolicyWithAttention(obs_dim, action_dim)
    
    # 模擬數據
    obs = torch.randn(batch_size, obs_dim)
    all_obs = torch.randn(batch_size, n_agents, obs_dim)
    
    # 前向傳播
    dist, att_weights = policy(obs, all_obs)
    actions = dist.sample()
    
    print(f"動作形狀: {actions.shape}")
    print(f"注意力權重形狀: {att_weights.shape}")
    print(f"注意力權重範例:\n{att_weights[0]}")