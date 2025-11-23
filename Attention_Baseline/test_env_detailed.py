from gymnasium_robotics import mamujoco_v1
import numpy as np

def test_environment():
    """詳細測試 MaMuJoCo 環境"""
    
    # 創建環境
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="2x4",
        agent_obsk=1,
        render_mode=None  # 先不渲染，測試數據流
    )
    
    print("="*60)
    print("環境基本信息")
    print("="*60)
    
    # 重置環境
    observations, infos = env.reset(seed=42)
    
    print(f"智能體數量: {len(env.agents)}")
    print(f"智能體列表: {env.agents}")
    
    print("\n觀察空間:")
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        print(f"  {agent}:")
        print(f"    形狀: {obs_space.shape}")
        print(f"    範圍: [{obs_space.low[0]:.2f}, {obs_space.high[0]:.2f}]")
    
    print("\n動作空間:")
    for agent in env.agents:
        act_space = env.action_space(agent)
        print(f"  {agent}:")
        print(f"    形狀: {act_space.shape}")
        print(f"    範圍: [{act_space.low[0]:.2f}, {act_space.high[0]:.2f}]")
    
    print("\n初始觀察:")
    for agent in env.agents:
        obs = observations[agent]
        print(f"  {agent}:")
        print(f"    形狀: {obs.shape}")
        print(f"    前5個值: {obs[:5]}")
        print(f"    統計: mean={obs.mean():.3f}, std={obs.std():.3f}")
    
    print("\n" + "="*60)
    print("運行測試回合")
    print("="*60)
    
    episode_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0
    max_steps = 1000
    
    try:
        while step_count < max_steps:
            # 構建動作
            actions = {}
            for agent in env.agents:
                # 隨機動作
                actions[agent] = env.action_space(agent).sample()
            
            # 環境步進
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 累積獎勵
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
            
            step_count += 1
            
            # 每100步顯示一次
            if step_count % 100 == 0:
                print(f"\nStep {step_count}:")
                print(f"  獎勵: {rewards}")
                print(f"  累積獎勵: {episode_rewards}")
            
            # 檢查是否結束
            if any(terminations.values()) or any(truncations.values()):
                print(f"\n回合在第 {step_count} 步結束")
                print(f"最終累積獎勵: {episode_rewards}")
                break
    
    except Exception as e:
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\n環境已關閉")
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)
    print(f"總步數: {step_count}")
    print(f"平均每步獎勵: {sum(episode_rewards.values()) / step_count:.4f}")

if __name__ == "__main__":
    test_environment()