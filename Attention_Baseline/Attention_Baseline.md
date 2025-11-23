# MARL Communication with Attention Baseline

Multi-Agent Reinforcement Learning with Attention Baseline實作(基於MaMuJoCo環境)。

## 檔案說明

### 核心模型
- **`attention_comm.py`**: 注意力通訊模組
  - `AttentionCommunication`: 多頭注意力聚合其他智能體資訊
  - `PolicyWithAttention`: 結合本地觀察與通訊特徵的策略網路

- **`baseline_no_comm.py`**: 無通訊基線
  - `SimplePolicy`: 獨立策略網路（不通訊）
  - `SimpleCritic`: 價值函數網路

### 訓練與評估
- **`train_with_attention.py`**: PPO 訓練流程
  - 收集 rollout（2048 步）
  - GAE 優勢計算
  - Mini-batch 更新（4 epochs）
  - 自動保存最佳模型

- **`eval_policy.py`**: 策略評估與視覺化
  - 載入 checkpoint 進行測試
  - 顯示注意力權重分佈
  - 可選影片錄製（`save_video=True`）(這邊錄製視角不知道為什麼動不了，請先忽略影片錄製的功能)

### 環境測試
- **`test_env.py`**: 基本環境互動測試
- **`test_env_detailed.py`**: 詳細環境資訊檢查（觀察/動作空間、統計）
- **`test_different_configs.py`**: 測試多種場景配置（Ant 2x4、4x2、HalfCheetah 等）

## 快速開始

```bash
pip install gymnasium-robotics
pip install pettingzoo
pip install torch

python test_env.py <- 用這個指令測試

# 訓練注意力通訊模型
python train_with_attention.py

# 評估訓練結果
python eval_policy.py

# 測試環境配置
python test_different_configs.py
```

## 模型架構

```
觀察 → 編碼器 → [本地特徵 + 注意力通訊] → 策略頭 → 動作分佈
                      ↑
                所有智能體觀察
```

注意力權重反映智能體間的通訊強度，可用於解釋協作行為。

## 輸出

- **Checkpoints**: `checkpoints/model_iter_{n}.pt` 和 `model_iter_best.pt` (我目前是訓練1000個iteration，然後很明確的越訓練他的分數會越高)
- **影片**: 請忽略此功能