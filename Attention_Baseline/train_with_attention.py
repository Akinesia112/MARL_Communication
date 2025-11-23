import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gymnasium_robotics import mamujoco_v1
from attention_comm import PolicyWithAttention
from baseline_no_comm import SimpleCritic
import os

class MAMuJoCoTrainer:
    def __init__(self, scenario="Ant", agent_conf="2x4", lr=3e-4, gamma=0.99):
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None  # 訓練時不渲染
        )
        
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        action_dim = self.env.action_space(self.agents[0]).shape[0]
        
        self.gamma = gamma
        
        # 創建網絡
        self.policies = {
            agent: PolicyWithAttention(obs_dim, action_dim) 
            for agent in self.agents
        }
        self.critics = {
            agent: SimpleCritic(obs_dim) 
            for agent in self.agents
        }
        
        # 優化器
        self.policy_optimizers = {
            agent: optim.Adam(self.policies[agent].parameters(), lr=lr)
            for agent in self.agents
        }
        self.critic_optimizers = {
            agent: optim.Adam(self.critics[agent].parameters(), lr=lr)
            for agent in self.agents
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
        # 創建保存目錄
        os.makedirs("checkpoints", exist_ok=True)
    
    def _move_to_device(self):
        for agent in self.agents:
            self.policies[agent].to(self.device)
            self.critics[agent].to(self.device)
    
    def collect_rollout(self, n_steps=2048):
        """收集一個回合的數據"""
        rollout = {agent: {
            'obs': [], 'actions': [], 'rewards': [], 
            'values': [], 'log_probs': [], 'all_obs': [],
            'dones': []
        } for agent in self.agents}
        
        obs_dict, _ = self.env.reset()
        
        for step in range(n_steps):
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in self.agents])
            ).unsqueeze(0).to(self.device)
            
            actions_dict = {}
            
            for agent in self.agents:
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    dist, _ = self.policies[agent](obs, all_obs)
                    value = self.critics[agent](obs)
                    
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                
                rollout[agent]['obs'].append(obs_dict[agent])
                rollout[agent]['all_obs'].append(all_obs.cpu().numpy()[0])
                rollout[agent]['actions'].append(action.cpu().numpy()[0])
                rollout[agent]['log_probs'].append(log_prob.cpu().item())
                rollout[agent]['values'].append(value.cpu().item())
                
                actions_dict[agent] = action.cpu().numpy()[0]
            
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_dict)
            
            for agent in self.agents:
                rollout[agent]['rewards'].append(rewards[agent])
                rollout[agent]['dones'].append(terms[agent] or truncs[agent])
            
            if any(terms.values()) or any(truncs.values()):
                obs_dict, _ = self.env.reset()
            else:
                obs_dict = next_obs
        
        return rollout
    
    def compute_advantages(self, rollout):
        """計算優勢函數（GAE）- 修正版"""
        for agent in self.agents:
            rewards = np.array(rollout[agent]['rewards'])
            values = np.array(rollout[agent]['values'])
            dones = np.array(rollout[agent]['dones'])
        
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            last_gae = 0
        
            # 獲取最後一個 value 用於 bootstrap
            last_obs = rollout[agent]['obs'][-1]
            last_all_obs = rollout[agent]['all_obs'][-1]
            with torch.no_grad():
                last_value = self.critics[agent](
                    torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                ).cpu().item()
        
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = last_value * (1 - dones[t])  # 修正：用 bootstrap
                else:
                    next_value = values[t + 1]
            
                delta = rewards[t] + self.gamma * next_value - values[t]
                advantages[t] = last_gae = delta + self.gamma * 0.95 * (1 - dones[t]) * last_gae
        
            returns = advantages + values
        
            rollout[agent]['advantages'] = advantages
            rollout[agent]['returns'] = returns
    
    def train_step(self, rollout, clip_epsilon=0.2, epochs=4, batch_size=64):
        """PPO 更新 - 添加 mini-batch"""
        self.compute_advantages(rollout)
    
        for epoch in range(epochs):
            for agent in self.agents:
                # 準備所有數據
                obs = np.array(rollout[agent]['obs'])
                all_obs = np.array(rollout[agent]['all_obs'])
                actions = np.array(rollout[agent]['actions'])
                old_log_probs = np.array(rollout[agent]['log_probs'])
                advantages = rollout[agent]['advantages']
                returns = rollout[agent]['returns']
            
                # 標準化優勢
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
                # Mini-batch 訓練
                n_samples = len(obs)
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
            
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                
                    # 轉換為 tensor
                    batch_obs = torch.FloatTensor(obs[batch_idx]).to(self.device)
                    batch_all_obs = torch.FloatTensor(all_obs[batch_idx]).to(self.device)
                    batch_actions = torch.FloatTensor(actions[batch_idx]).to(self.device)
                    batch_old_log_probs = torch.FloatTensor(old_log_probs[batch_idx]).to(self.device)
                    batch_advantages = torch.FloatTensor(advantages[batch_idx]).to(self.device)
                    batch_returns = torch.FloatTensor(returns[batch_idx]).to(self.device)
                
                    # 策略更新
                    dist, _ = self.policies[agent](batch_obs, batch_all_obs)
                    new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                
                    # 添加熵正則化（鼓勵探索）
                    entropy = dist.entropy().mean()
                    policy_loss = policy_loss - 0.01 * entropy
                
                    self.policy_optimizers[agent].zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policies[agent].parameters(), 0.5)
                    self.policy_optimizers[agent].step()
                
                    # 價值更新
                    values = self.critics[agent](batch_obs).squeeze()
                    value_loss = F.mse_loss(values, batch_returns)
                
                    self.critic_optimizers[agent].zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
                    self.critic_optimizers[agent].step()
    
    def save_checkpoint(self, iteration):
        """保存模型"""
        checkpoint = {
            agent: self.policies[agent].state_dict() 
            for agent in self.agents
        }
        torch.save(checkpoint, f"checkpoints/model_iter_{iteration}.pt")
    
    def train(self, n_iterations=1000, save_freq=50):
        """主訓練循環 - 添加更多監控"""
        best_reward = -float('inf')
    
        for iteration in range(n_iterations):
            rollout = self.collect_rollout(n_steps=2048)
            self.train_step(rollout, epochs=4, batch_size=64)  # 減少 epochs，增加 batch
        
            # 計算統計資訊
            avg_reward = np.mean([np.sum(rollout[agent]['rewards']) for agent in self.agents])
            max_reward = np.max([np.sum(rollout[agent]['rewards']) for agent in self.agents])
        
            print(f"Iter {iteration}: Avg={avg_reward:.2f}, Max={max_reward:.2f}")
        
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint('best')
                print(f"  ✓ 新最佳模型！獎勵: {best_reward:.2f}")
        
            if (iteration + 1) % save_freq == 0:
                self.save_checkpoint(iteration)

# 使用更小的學習率
if __name__ == "__main__":
    trainer = MAMuJoCoTrainer(
        scenario="Ant",
        agent_conf="2x4",
        lr=1e-4  # 降低學習率
    )
    trainer.train(n_iterations=1000, save_freq=50)