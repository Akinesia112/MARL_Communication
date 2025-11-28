import torch
import numpy as np
import matplotlib.pyplot as plt
from gymnasium_robotics import mamujoco_v1
from policy_with_pde import PolicyWithPDE
from attention_comm import PolicyWithAttention
from baseline_no_comm import SimpleCritic
import os
import json

class PDEEvaluator:
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
        
        # 載入模型
        self.pde_policies = {
            agent: PolicyWithPDE(obs_dim, action_dim).to(self.device) 
            for agent in self.agents
        }
        self.attention_policies = {
            agent: PolicyWithAttention(obs_dim, action_dim).to(self.device)
            for agent in self.agents
        }
        
        os.makedirs("evaluation_results", exist_ok=True)
    
    def load_models(self, pde_path, attention_path):
        """載入訓練好的模型"""
        print(f"載入 PDE 模型: {pde_path}")
        pde_ckpt = torch.load(pde_path, map_location=self.device, weights_only=False)
        for agent in self.agents:
            self.pde_policies[agent].load_state_dict(pde_ckpt['policies'][agent])
            self.pde_policies[agent].eval()
        
        print(f"載入 Attention 模型: {attention_path}")
        att_ckpt = torch.load(attention_path, map_location=self.device, weights_only=False)
        for agent in self.agents:
            self.attention_policies[agent].load_state_dict(att_ckpt[agent])
            self.attention_policies[agent].eval()
    
    # ========== 實驗 1: 視覺化對比 ==========
    def visualize_communication(self, n_steps=100):
        """對比 Attention weights vs PDE field"""
        print("\n[實驗 1] 視覺化通訊機制...")
        
        obs_dict, _ = self.env.reset()
        
        for step in [0, 50, 99]:
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in self.agents])
            ).unsqueeze(0).to(self.device)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Attention
            with torch.no_grad():
                obs = torch.FloatTensor(obs_dict[self.agents[0]]).unsqueeze(0).to(self.device)
                _, att_weights = self.attention_policies[self.agents[0]](obs, all_obs)
                
                if att_weights is not None:
                    # 改成 bar chart（1D → 長條圖）
                    weights = att_weights[0].cpu().numpy()
                    axes[0].bar(range(len(weights)), weights)
                    axes[0].set_title(f'Attention Weights (Step {step})')
                    axes[0].set_xlabel('Agent')
                    axes[0].set_ylabel('Weight')
                    axes[0].set_ylim([0, 1])
                else:
                    axes[0].text(0.5, 0.5, 'No Attention Model', ha='center', va='center')
                    axes[0].set_title('Attention (Not Loaded)')
            
            # PDE
            with torch.no_grad():
                _, field = self.pde_policies[self.agents[0]](obs, all_obs, agent_idx=0)
                field_energy = (field[0] ** 2).sum(dim=0).cpu().numpy()
                
                im = axes[1].imshow(field_energy, cmap='hot', origin='lower')
                axes[1].set_title(f'PDE Field Energy (Step {step})')
                plt.colorbar(im, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(f'evaluation_results/exp1_visualization_step{step}.png', dpi=150)
            plt.close()
            
            # 環境步進
            actions = {agent: self.env.action_space(agent).sample() for agent in self.agents}
            obs_dict, _, terms, truncs, _ = self.env.step(actions)
            if any(terms.values()) or any(truncs.values()):
                obs_dict, _ = self.env.reset()
        
        print("✓ 視覺化完成 → evaluation_results/exp1_*.png")
    
    # ========== 實驗 2: Hodge 分解 ==========
    def analyze_hodge_decomposition(self, n_episodes=10):
        """分析場的梯度與旋度分量"""
        print("\n[實驗 2] Hodge 分解分析...")
        
        gradient_energies = []
        curl_energies = []
        
        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset()
            
            for step in range(100):
                all_obs = torch.FloatTensor(
                    np.array([obs_dict[agent] for agent in self.agents])
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    obs = torch.FloatTensor(obs_dict[self.agents[0]]).unsqueeze(0).to(self.device)
                    _, field = self.pde_policies[self.agents[0]](obs, all_obs, agent_idx=0)
                    
                    # 簡化版 Hodge 分解
                    grad_x = field[:, :, 1:, :] - field[:, :, :-1, :]
                    grad_y = field[:, :, :, 1:] - field[:, :, :, :-1]
                    
                    gradient_energy = (grad_x ** 2).mean() + (grad_y ** 2).mean()
                    curl_energy = (field ** 2).mean() - gradient_energy
                    
                    gradient_energies.append(gradient_energy.item())
                    curl_energies.append(curl_energy.item())
                
                actions = {agent: self.env.action_space(agent).sample() for agent in self.agents}
                obs_dict, _, terms, truncs, _ = self.env.step(actions)
                if any(terms.values()) or any(truncs.values()):
                    break
        
        # 繪圖
        plt.figure(figsize=(10, 5))
        plt.plot(gradient_energies, label='Gradient (Navigation)', alpha=0.7)
        plt.plot(curl_energies, label='Curl (Coordination)', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('Hodge Decomposition: Navigation vs Coordination')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('evaluation_results/exp2_hodge_decomposition.png', dpi=150)
        plt.close()
        
        print(f"✓ Hodge 分解完成")
        print(f"  平均梯度能量: {np.mean(gradient_energies):.2f}")
        print(f"  平均旋度能量: {np.mean(curl_energies):.2f}")
    
    # ========== 實驗 5: Robustness ==========
    def test_with_noise(self, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], n_episodes=5):
        """測試不同 Noise Level下的表現"""
        print("\n[實驗 5] 噪音Robustness測試...")
        
        results = {'attention': [], 'pde': []}
        
        for noise in noise_levels:
            print(f"  Noise Level: {noise:.1f}")
            
            # 測試 Attention
            att_rewards = []
            for _ in range(n_episodes):
                reward = self._evaluate_model(self.attention_policies, noise, use_pde=False)
                att_rewards.append(reward)
            results['attention'].append(np.mean(att_rewards))
            
            # 測試 PDE
            pde_rewards = []
            for _ in range(n_episodes):
                reward = self._evaluate_model(self.pde_policies, noise, use_pde=True)
                pde_rewards.append(reward)
            results['pde'].append(np.mean(pde_rewards))
            
            print(f"    Attention: {np.mean(att_rewards):.2f}, PDE: {np.mean(pde_rewards):.2f}")
        
        # 繪圖
        plt.figure(figsize=(8, 6))
        plt.plot(noise_levels, results['attention'], 'o-', label='Attention', linewidth=2)
        plt.plot(noise_levels, results['pde'], 's-', label='PDE', linewidth=2)
        plt.xlabel('Noise Level')
        plt.ylabel('Average Reward')
        plt.title('Robustness to Observation Noise')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('evaluation_results/exp5_noise_robustness.png', dpi=150)
        plt.close()
        
        # 保存數據
        with open('evaluation_results/exp5_noise_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ 噪音測試完成 → evaluation_results/exp5_*.png")
    
    def _evaluate_model(self, policies, noise_level, use_pde=False, max_steps=200):
        """評估模型在指定噪音等級下的表現"""
        obs_dict, _ = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in self.agents])
            ).unsqueeze(0).to(self.device)
            
            # 添加噪音
            if noise_level > 0:
                all_obs = all_obs + torch.randn_like(all_obs) * noise_level
            
            actions_dict = {}
            for agent_idx, agent in enumerate(self.agents):
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                if noise_level > 0:
                    obs = obs + torch.randn_like(obs) * noise_level
                
                with torch.no_grad():
                    if use_pde:
                        dist, _ = policies[agent](obs, all_obs, agent_idx)
                    else:
                        dist, _ = policies[agent](obs, all_obs)
                    action = dist.mean  # 使用均值（更穩定）
                
                actions_dict[agent] = action.cpu().numpy()[0]
            
            obs_dict, rewards, terms, truncs, _ = self.env.step(actions_dict)
            total_reward += sum(rewards.values())
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        return total_reward / self.n_agents
    
    # ========== 實驗 6: 遮擋測試 ==========
    def test_with_occlusion(self, occlusion_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], n_episodes=5):
        """測試遮擋部分觀察的影響"""
        print("\n[實驗 6] 遮擋Robustness測試...")
        
        results = {'attention': [], 'pde': []}
        
        for ratio in occlusion_ratios:
            print(f"  遮擋比例: {ratio:.1%}")
            
            # Attention
            att_rewards = []
            for _ in range(n_episodes):
                reward = self._evaluate_with_occlusion(self.attention_policies, ratio, use_pde=False)
                att_rewards.append(reward)
            results['attention'].append(np.mean(att_rewards))
            
            # PDE
            pde_rewards = []
            for _ in range(n_episodes):
                reward = self._evaluate_with_occlusion(self.pde_policies, ratio, use_pde=True)
                pde_rewards.append(reward)
            results['pde'].append(np.mean(pde_rewards))
            
            print(f"    Attention: {np.mean(att_rewards):.2f}, PDE: {np.mean(pde_rewards):.2f}")
        
        # 繪圖
        plt.figure(figsize=(8, 6))
        plt.plot(occlusion_ratios, results['attention'], 'o-', label='Attention', linewidth=2)
        plt.plot(occlusion_ratios, results['pde'], 's-', label='PDE', linewidth=2)
        plt.xlabel('Occlusion Ratio')
        plt.ylabel('Average Reward')
        plt.title('Robustness to Observation Occlusion')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('evaluation_results/exp6_occlusion_robustness.png', dpi=150)
        plt.close()
        
        with open('evaluation_results/exp6_occlusion_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ 遮擋測試完成 → evaluation_results/exp6_*.png")
    
    def _evaluate_with_occlusion(self, policies, occlusion_ratio, use_pde=False, max_steps=200):
        """評估遮擋觀察下的表現"""
        obs_dict, _ = self.env.reset()
        total_reward = 0
        
        # 創建固定的遮擋 mask
        obs_dim = self.env.observation_space(self.agents[0]).shape[0]
        mask = torch.rand(obs_dim) > occlusion_ratio
        mask = mask.to(self.device)
        
        for step in range(max_steps):
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in self.agents])
            ).unsqueeze(0).to(self.device)
            all_obs = all_obs * mask  # 應用遮擋
            
            actions_dict = {}
            for agent_idx, agent in enumerate(self.agents):
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                obs = obs * mask  # 應用遮擋
                
                with torch.no_grad():
                    if use_pde:
                        dist, _ = policies[agent](obs, all_obs, agent_idx)
                    else:
                        dist, _ = policies[agent](obs, all_obs)
                    action = dist.mean
                
                actions_dict[agent] = action.cpu().numpy()[0]
            
            obs_dict, rewards, terms, truncs, _ = self.env.step(actions_dict)
            total_reward += sum(rewards.values())
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        return total_reward / self.n_agents
    
    # ========== 實驗 7: 場 SNR 分析 ==========
    def analyze_field_snr(self, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], n_samples=50):
        """分析 PDE 場的信噪比"""
        print("\n[實驗 7] 場 SNR 分析...")
        
        snr_results = []
        
        for noise in noise_levels:
            snrs = []
            
            obs_dict, _ = self.env.reset()
            
            for _ in range(n_samples):
                all_obs = torch.FloatTensor(
                    np.array([obs_dict[agent] for agent in self.agents])
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    obs = torch.FloatTensor(obs_dict[self.agents[0]]).unsqueeze(0).to(self.device)
                    
                    # 乾淨場
                    _, clean_field = self.pde_policies[self.agents[0]](obs, all_obs, agent_idx=0)
                    
                    # 噪音場
                    noisy_obs = all_obs + torch.randn_like(all_obs) * noise
                    _, noisy_field = self.pde_policies[self.agents[0]](obs, noisy_obs, agent_idx=0)
                    
                    # 計算 SNR
                    signal_power = (clean_field ** 2).mean()
                    noise_power = ((clean_field - noisy_field) ** 2).mean()
                    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                    snrs.append(snr.item())
                
                # 隨機步進
                actions = {agent: self.env.action_space(agent).sample() for agent in self.agents}
                obs_dict, _, terms, truncs, _ = self.env.step(actions)
                if any(terms.values()) or any(truncs.values()):
                    obs_dict, _ = self.env.reset()
            
            avg_snr = np.mean(snrs)
            snr_results.append(avg_snr)
            print(f"  噪音={noise:.1f}, 場 SNR={avg_snr:.2f} dB")
        
        # 繪圖
        plt.figure(figsize=(8, 6))
        plt.plot(noise_levels, snr_results, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Input Noise Level')
        plt.ylabel('Field SNR (dB)')
        plt.title('PDE Field Signal-to-Noise Ratio')
        plt.grid(alpha=0.3)
        plt.savefig('evaluation_results/exp7_field_snr.png', dpi=150)
        plt.close()
        
        print("✓ SNR 分析完成 → evaluation_results/exp7_*.png")
    
    def run_all_experiments(self):
        """執行所有實驗"""
        print("=" * 60)
        print("開始 PDE Communication 評估實驗")
        print("=" * 60)
        
        self.visualize_communication()
        self.analyze_hodge_decomposition()
        self.test_with_noise()
        self.test_with_occlusion()
        self.analyze_field_snr()
        
        print("\n" + "=" * 60)
        print("所有實驗完成！結果保存在 evaluation_results/")
        print("=" * 60)


if __name__ == "__main__":
    evaluator = PDEEvaluator(scenario="Ant", agent_conf="2x4")
    
    # 載入訓練好的模型
    evaluator.load_models(
        pde_path="checkpoints_pde/model_iter_best.pt",
        attention_path="checkpoints/model_iter_best.pt"
    )
    
    # 執行所有實驗
    evaluator.run_all_experiments()