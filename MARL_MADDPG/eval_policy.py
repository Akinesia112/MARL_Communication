import torch
import numpy as np
from gymnasium_robotics import mamujoco_v1
from attention_comm import PolicyWithAttention
import imageio
import os


def evaluate_policy(model_path=None, n_episodes=3, save_video=False):
    """評估並視覺化策略"""
    
    # 根據是否保存影片選擇渲染模式
    render_mode = "rgb_array" if save_video else "human"
    
    # 創建帶渲染的環境
    env = mamujoco_v1.parallel_env(
        scenario="Ant",
        agent_conf="2x4",
        agent_obsk=1,
        max_episode_steps=500,
        render_mode=render_mode
    )
    
    # 設定固定的上帝視角相機
    if save_video and hasattr(env.unwrapped, 'mujoco_renderer'):
        # 設定相機位置和角度
        camera_config = {
            'distance': 8.0,      # 相機距離
            'azimuth': 90.0,      # 方位角（左右旋轉）
            'elevation': -30.0,   # 仰角（上下旋轉）
            'lookat': [0, 0, 0.5] # 相機看向的點
        }
        
        # 如果環境支援，設定相機
        try:
            env.unwrapped.mujoco_renderer.default_cam_config = camera_config
        except:
            print("警告: 無法設定相機配置，使用預設視角")
    
    agents = env.agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]
    
    # 載入模型
    policies = {
        agent: PolicyWithAttention(obs_dim, action_dim) 
        for agent in agents
    }
    
    if model_path:
        checkpoint = torch.load(model_path)
        for agent in agents:
            policies[agent].load_state_dict(checkpoint[agent])
        print(f"已載入模型: {model_path}")
        model_name = model_path.split('/')[-1].replace('.pt', '')
    else:
        print("使用隨機策略（未訓練）")
        model_name = "random"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for agent in agents:
        policies[agent].to(device)
        policies[agent].eval()
    
    # 創建影片保存目錄
    if save_video:
        os.makedirs("videos", exist_ok=True)
    
    # 運行回合
    for episode in range(n_episodes):
        obs_dict, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agents}
        step = 0
        done = False
        frames = []  # 儲存幀
        
        print(f"\n{'='*50}")
        print(f"回合 {episode + 1}/{n_episodes}")
        print(f"{'='*50}")
        
        while not done:
            # 準備所有觀察
            all_obs = torch.FloatTensor(
                np.array([obs_dict[agent] for agent in agents])
            ).unsqueeze(0).to(device)
            
            actions_dict = {}
            attention_weights_dict = {}
            
            for agent in agents:
                obs = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    dist, att_weights = policies[agent](obs, all_obs)
                    action = dist.mean  # 使用均值而非採樣（更穩定）
                
                actions_dict[agent] = action.cpu().numpy()[0]
                attention_weights_dict[agent] = att_weights.cpu().numpy()[0]
            
            # 環境步進
            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
            
            # 累積獎勵
            for agent in agents:
                episode_reward[agent] += rewards[agent]
            
            # 每50步顯示一次注意力權重
            if step % 50 == 0:
                print(f"\nStep {step}:")
                for agent in agents:
                    print(f"  {agent} 注意力權重: {attention_weights_dict[agent]}")
                print(f"  當前獎勵: {rewards}")
            
            step += 1
            
            # 渲染並保存幀
            frame = env.render()
            if save_video and frame is not None:
                frames.append(frame)
            
            # 檢查是否結束
            if any(terms.values()) or any(truncs.values()):
                done = True
            else:
                obs_dict = next_obs
        
        print(f"\n回合 {episode + 1} 結束:")
        print(f"  總步數: {step}")
        print(f"  總獎勵: {episode_reward}")
        print(f"  平均獎勵: {sum(episode_reward.values()) / len(agents):.2f}")
        
        # 保存影片
        if save_video and len(frames) > 0:
            video_path = f'videos/{model_name}_episode_{episode+1}.mp4'
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  ✓ 影片已保存: {video_path}")
    
    env.close()

if __name__ == "__main__":

    evaluate_policy(model_path=None, n_episodes=1, save_video=False)

    evaluate_policy(model_path="checkpoints/model_iter_449.pt", n_episodes=1, save_video=False)

    evaluate_policy(model_path="checkpoints/model_iter_best.pt", n_episodes=1, save_video=False)