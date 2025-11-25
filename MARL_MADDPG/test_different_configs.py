from gymnasium_robotics import mamujoco_v1
import numpy as np

def test_config(scenario, agent_conf, n_steps=100):
    """測試特定配置"""
    print(f"\n{'='*60}")
    print(f"測試: {scenario} with {agent_conf}")
    print(f"{'='*60}")
    
    try:
        env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,
            render_mode=None
        )
        
        observations, infos = env.reset(seed=42)
        
        print(f"智能體數量: {len(env.agents)}")
        print(f"觀察維度: {observations[env.agents[0]].shape}")
        print(f"動作維度: {env.action_space(env.agents[0]).shape}")
        
        total_reward = 0
        for step in range(n_steps):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            
            if any(terminations.values()) or any(truncations.values()):
                observations, infos = env.reset()
        
        print(f"平均獎勵: {total_reward / n_steps:.4f}")
        env.close()
        print("✓ 配置測試成功")
        
    except Exception as e:
        print(f"✗ 配置測試失敗: {e}")

if __name__ == "__main__":
    # 測試不同配置
    configs = [
        ("Ant", "2x4"),      # 2個智能體，每個4個關節
        ("Ant", "4x2"),      # 4個智能體，每個2個關節
        ("HalfCheetah", "2x3"),  # 半獵豹
        ("Hopper", "3x1"),   # 單足跳躍者
    ]
    
    for scenario, agent_conf in configs:
        test_config(scenario, agent_conf, n_steps=100)