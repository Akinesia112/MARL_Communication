from gymnasium_robotics import mamujoco_v1
import numpy as np

# 創建一個簡單的環境
env = mamujoco_v1.parallel_env(
    scenario="Ant",
    agent_conf="2x4",  # 2個智能體，每個控制4個關節
    agent_obsk=1,      # 觀察1層鄰近關節
    render_mode="human"
)

# 重置環境
observations, infos = env.reset()

print("智能體列表:", env.agents)
print("觀察空間:", {agent: env.observation_space(agent) for agent in env.agents})
print("動作空間:", {agent: env.action_space(agent) for agent in env.agents})

# 運行幾個步驟
try:
    for step in range(100):
        # 正確的動作格式：字典，鍵是智能體ID字符串
        actions = {}
        for agent in env.agents:
            # 從動作空間採樣
            action = env.action_space(agent).sample()
            actions[agent] = action
        
        # 如果你想看動作的內容
        if step == 0:
            print("\n動作格式範例:")
            for agent, action in actions.items():
                print(f"  {agent}: {action}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 顯示獎勵
        if step % 10 == 0:
            print(f"Step {step}: Rewards = {rewards}")
        
        # 檢查是否有智能體結束
        if any(terminations.values()) or any(truncations.values()):
            print(f"Episode ended at step {step}")
            observations, infos = env.reset()
        
        env.render()

except KeyboardInterrupt:
    print("\n手動中斷")
except Exception as e:
    print(f"\n發生錯誤: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
    print("環境已關閉")