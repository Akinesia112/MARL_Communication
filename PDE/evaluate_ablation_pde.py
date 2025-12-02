import torch
import numpy as np
import matplotlib.pyplot as plt
from gymnasium_robotics import mamujoco_v1
from policy_with_pde import PolicyWithPDE
import os
import json

class AblationEvaluator:
    def __init__(self, scenario="Ant", agent_conf="2x4"):
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None
        )
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        action_dim = self.env.action_space(self.agents[0]).shape[0]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 4 個 ablation 版本
        self.model_configs = {
            'Full PDE': None,
            'No Diffusion': None,
            'No Reaction': None,
            'No PDE (MLP)': None
        }
        
        os.makedirs("evaluation_results", exist_ok=True)
    
    def load_models(self, checkpoint_paths):
        """載入 4 個 ablation 模型
        Args:
            checkpoint_paths: dict, e.g., {
                'Full PDE': 'checkpoints_pde/model_iter_best.pt',
                'No Diffusion': 'checkpoints_pde_nodiff/model_iter_best.pt',
                ...
            }
        """
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        action_dim = self.env.action_space(self.agents[0]).shape[0]
        
        for name, path in checkpoint_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ 找不到 {name}: {path}")
                continue
            
            print(f"載入 {name}: {path}")
            policies = {
                agent: PolicyWithPDE(obs_dim, action_dim).to(self.device)
                for agent in self.agents
            }
            
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            for agent in self.agents:
                policies[agent].load_state_dict(ckpt['policies'][agent])
                policies[agent].eval()
            
            self.model_configs[name] = policies
    
    def evaluate_model(self, policies, n_episodes=10, max_steps=200):
        """評估單一模型"""
        rewards = []
        
        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                all_obs = torch.FloatTensor(
                    np.array([obs_dict[agent] for agent in self.agents])
                ).unsqueeze(0).to(self.device)
                
                actions_dict = {}
                for agent_idx, agent in enumerate(self.agents):
                    obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        dist, _ = policies[agent](obs, all_obs, agent_idx)
                        action = dist.mean
                    
                    actions_dict[agent] = action.cpu().numpy()[0]
                
                obs_dict, reward_dict, terms, truncs, _ = self.env.step(actions_dict)
                total_reward += sum(reward_dict.values())
                
                if any(terms.values()) or any(truncs.values()):
                    break
            
            rewards.append(total_reward / self.n_agents)
        
        return np.mean(rewards), np.std(rewards)
    
    def run_ablation_study(self):
        """實驗 5: Ablation Study"""
        print("\n" + "="*60)
        print("Ablation Study: 比較 PDE 各組件的貢獻")
        print("="*60)
        
        results = {}
        
        for name, policies in self.model_configs.items():
            if policies is None:
                print(f"\n⚠️ 跳過 {name} (未載入)")
                continue
            
            print(f"\n評估 {name}...")
            mean_reward, std_reward = self.evaluate_model(policies, n_episodes=10)
            results[name] = {
                'reward_mean': mean_reward,
                'reward_std': std_reward
            }
            print(f"  {name}: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # 繪圖
        self.plot_ablation_results(results)
        
        # 保存數據
        with open('evaluation_results/ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("Ablation Study 完成！")
        print("="*60)
        
        return results
    
    def plot_ablation_results(self, results):
        """繪製 ablation 結果"""
        if not results:
            print("⚠️ 無數據可繪製")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 排序（Full PDE 放最前面）
        order = ['Full PDE', 'No Diffusion', 'No Reaction', 'No PDE (MLP)']
        names = [n for n in order if n in results]
        means = [results[n]['reward_mean'] for n in names]
        stds = [results[n]['reward_std'] for n in names]
        
        # 顏色編碼
        colors = {
            'Full PDE': '#2ecc71',        # 綠色（最好）
            'No Diffusion': '#e74c3c',    # 紅色（差）
            'No Reaction': '#f39c12',     # 橘色（中等）
            'No PDE (MLP)': '#95a5a6'     # 灰色（最差）
        }
        bar_colors = [colors.get(n, '#3498db') for n in names]
        
        # 繪製
        bars = ax.bar(range(len(names)), means, yerr=stds, 
                     capsize=5, color=bar_colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=12, rotation=15, ha='right')
        ax.set_ylabel('Average Reward', fontsize=13)
        ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # 添加數值標籤
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 10,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('evaluation_results/ablation_study.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n✓ 圖表已保存: evaluation_results/ablation_study.png")
    
    def compare_with_noise(self, noise_levels=[0.0, 0.2, 0.4]):
        """額外測試：不同噪音下的 ablation 比較"""
        print("\n[額外] Ablation + Noise Robustness...")
        
        results = {name: [] for name in self.model_configs.keys() if self.model_configs[name]}
        
        for noise in noise_levels:
            print(f"\n  Noise Level: {noise:.1f}")
            
            for name, policies in self.model_configs.items():
                if policies is None:
                    continue
                
                rewards = []
                for _ in range(5):
                    reward = self._evaluate_with_noise(policies, noise)
                    rewards.append(reward)
                
                mean_reward = np.mean(rewards)
                results[name].append(mean_reward)
                print(f"    {name}: {mean_reward:.2f}")
        
        # 繪圖
        plt.figure(figsize=(10, 6))
        for name, rewards in results.items():
            plt.plot(noise_levels, rewards, 'o-', label=name, linewidth=2, markersize=8)
        
        plt.xlabel('Noise Level', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Ablation Study: Robustness to Noise', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_results/ablation_noise.png', dpi=150)
        plt.close()
        
        print("\n✓ 噪音測試完成 → evaluation_results/ablation_noise.png")
    
    def _evaluate_with_noise(self, policies, noise_level, max_steps=200):
        """帶噪音的評估"""
        obs_dict, _ = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in self.agents])
            ).unsqueeze(0).to(self.device)
            
            if noise_level > 0:
                all_obs = all_obs + torch.randn_like(all_obs) * noise_level
            
            actions_dict = {}
            for agent_idx, agent in enumerate(self.agents):
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                if noise_level > 0:
                    obs = obs + torch.randn_like(obs) * noise_level
                
                with torch.no_grad():
                    dist, _ = policies[agent](obs, all_obs, agent_idx)
                    action = dist.mean
                
                actions_dict[agent] = action.cpu().numpy()[0]
            
            obs_dict, rewards, terms, truncs, _ = self.env.step(actions_dict)
            total_reward += sum(rewards.values())
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        return total_reward / self.n_agents


if __name__ == "__main__":
    evaluator = AblationEvaluator(scenario="Ant", agent_conf="2x4")
    
    # 載入 4 個模型（根據你的實際路徑修改）
    checkpoint_paths = {
        'Full PDE': 'checkpoints_pde/model_iter_best.pt',
        'No Diffusion': 'checkpoints_pde_nodiff/model_iter_best.pt',
        'No Reaction': 'checkpoints_pde_noreact/model_iter_best.pt',
        'No PDE (MLP)': 'checkpoints_pde_nofield/model_iter_best.pt'
    }
    
    evaluator.load_models(checkpoint_paths)
    
    # 執行 ablation study
    results = evaluator.run_ablation_study()
    
    # 額外測試：噪音環境下的比較
    evaluator.compare_with_noise()
    
    print("\n結果檔案：")
    print("  - evaluation_results/ablation_study.png")
    print("  - evaluation_results/ablation_noise.png")
    print("  - evaluation_results/ablation_results.json")