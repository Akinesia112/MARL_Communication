# PDE Communication for Multi-Agent Reinforcement Learning

---

## 核心概念

使用 Reaction-Diffusion PDE 模擬真實世界的「空間型通訊機制」，使 MARL agent 能以物理方式交換訊息：
- Agent 將編碼後的特徵「釋放」到一個2D的PDE場
- 訊息在場中沿空間方向自然擴散(Reaction Diffusion)
- 其他agent可從場的局部區域「感知」擴散後的訊息
---

## 檔案架構

```
pde_comm.py           # PDE 通訊層（核心）
policy_with_pde.py    # 整合 PDE 的策略網路
train_with_pde.py     # PPO 訓練流程
```

---

## 1. `pde_comm.py` - PDE 通訊層

### 核心方程式

**Reaction-Diffusion PDE:**
```
∂u/∂t = D·∇²u + R·u·(1-tanh(u))
  ↑      ↑        ↑
時間變化  擴散項   反應項
```

| 參數          | 含義                               |
| ----------- | -------------------------------- |
| **D**       | diffusion_coef（可學習）→ 訊息擴散速度      |
| **R**       | reaction_coef（可學習）→ 訊息增強 / 非線性互動 |
| **∇²u**     | Laplacian（使用 3×3 卷積實作）           |
| **dt**      | 時間積分步長                           |
| **n_steps** | PDE 演化步數                         |




### PDECommunication

```python
class PDECommunication(nn.Module):
    def __init__(self, feature_dim, grid_size=8, n_steps=2, dt=0.2, sigma=0.8):
```
目的：將 [B, N, feature_dim] 的 agent 特徵轉成經 PDE 擴散後的通訊特徵 [B, N, feature_dim]。

### Workflow

```
1. agents_to_field
   將 N 個 agent 的 feature 放到場中對應位置（grid）

2. PDE reaction-diffusion step × n_steps
   使用 Laplacian kernel + 單層非線性反應項更新場

3. field_to_agents
   從場中 (3×3 neighborhood) 採樣通訊訊息

4. 選取 agent_idx 的通訊結果作為輸出
```

**重要函式說明：**

| 函式                        | 說明                           |
| ------------------------- | ---------------------------- |
| `agents_to_field`         | 把每個 agent feature 放到 PDE 網格上 |
| `reaction_diffusion_step` | PDE 更新核心                     |
| `field_to_agents`         | 從 PDE 場感知訊息（局部平均）            |
| `forward`                 | 完整 encode 過程                 |



---

## 2. `policy_with_pde.py` - 策略網路

### 關鍵程式碼

```python
class PolicyWithPDE(nn.Module):
    def forward(self, obs, all_obs, agent_idx=0):
        batch_size, n_agents, obs_dim = all_obs.shape
        # 1. 編碼觀察
        local_features = self.obs_encoder(obs)
        all_features = self.obs_encoder(all_obs)
        
        # 2. PDE 通訊
        comm_features, field = self.comm(agent_idx, all_features)
        
        # 3. 結合本地 + 通訊特徵
        combined = torch.cat([local_features, comm_features], dim=-1)
        
        # 4. 輸出動作分佈
        policy_features = self.policy_net(combined)
        mean = self.mean_head(policy_features)
        return Normal(mean, std), field
```

**為何需要 `agent_idx`？**
- PDE 通訊產生所有 agent 的訊息 [B, N, feat]
- 需要知道「我是誰」才能選對應的訊息，取
comm_message = all_messages[:, agent_idx]

---

## 3. `train_with_pde.py` - PPO 訓練

### 訓練流程

```
1. collect_rollout (收集 2048 steps)
   ┌────────────────────────────────────┐
   │ 每個 step:                         │
   │   for agent_idx, agent in agents:  │
   │     policy(obs, all_obs, agent_idx)│
   │     → action, log_prob, value      │
   │   env.step(actions) → rewards      │
   └────────────────────────────────────┘
          ↓
2. compute_advantages (GAE)
   ┌────────────────────────────────────┐
   │ 計算每個 agent 的：                 │
   │   advantages = δ + γλ·next_adv     │
   │   returns = advantages + values    │
   └────────────────────────────────────┘
          ↓
3. train_step (PPO 更新，4 epochs)
   ┌────────────────────────────────────┐
   │ Mini-batch (64 samples):           │
   │   Policy loss (PPO clip)           │
   │   Value loss (MSE)                 │
   │   Entropy bonus (探索)              │
   └────────────────────────────────────┘
          ↓
4. save_checkpoint (每 50 iter)
   保存 policies, critics, optimizers
```

### 關鍵設計

#### **Rollout 收集**
```python
for agent_idx, agent in enumerate(self.agents):
    dist, field = self.policies[agent](obs, all_obs, agent_idx)
    #                                                ↑
    #                                        傳入 agent 索引
```

#### **PPO 更新**
```python
# 策略更新（帶 agent_idx）
dist, _ = self.policies[agent](batch_obs, batch_all_obs, agent_idx)

# PPO clip
ratio = exp(new_log_prob - old_log_prob)
loss = -min(ratio·adv, clip(ratio, 1±ε)·adv)
```

#### **Checkpoint 管理**
```python
checkpoint = {
    ...略，反正就是checkpoint，可以接續之前的訓練
}
```
---

## 使用方式

### 開始訓練

```bash
python train_with_pde.py
```

輸出：
```
Iter 0: Avg=-123.45, Max=-100.23
Iter 1: Avg=-110.56, Max=-95.12
  ✓ 新最佳模型！獎勵: -110.56
...
```

### 從 Checkpoint 恢復

```python
# 修改 train_with_pde.py 最後一行
```

### 視覺化 PDE 場 (目前不確定實際作用為何，可能還要修改)

```python
if (iteration + 1) % save_freq == 0:
    agent = self.agents[0]
    obs = torch.FloatTensor(np.array([rollout[agent]['obs'][-1]])).to(self.device)
    all_obs = torch.FloatTensor(np.array([rollout[a]['obs'][-1] for a in self.agents])).unsqueeze(0).to(self.device)

    with torch.no_grad():
        _, field = self.policies[agent](obs, all_obs, agent_idx=0)
    # 繪製場能量
    energy = (field[0] ** 2).sum(dim=0).cpu().numpy()
    plt.imshow(energy, cmap='hot')
    plt.savefig(f'field_{iteration}.png')
    plt.close()
```

---

## 與 Attention Baseline 比較

| 特性        | Attention | PDE Communication |
| --------- | --------- | ----------------- |
| 訊息流動      | QKV     | 空間擴散              |
| 可解釋性      | 權重矩陣      | 2D 場熱圖            |
| 模型偏好      | 任意拓樸      | (更自然) 空間任務        |
| 計算量       | O(N²)     | O(HW)             |
| 距離衰減      | 無         | 天然產生              |
| Multi-hop | 需堆層       | PDE 自帶 multi-hop  |


---

### PDE 參數意義

| 參數                 | 效果          |
| ------------------ | ----------- |
| diffusion_coef ( D ) | 訊息傳遞距離      |
| reaction_coef ( R )  | 訊息成長 / 衰減   |
| dt                 | 穩定性（太大會爆炸）  |
| n_steps            | 擴散時間        |
| grid_size          | 場解析度（變大會變慢） |


---

