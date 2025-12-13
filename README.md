# Multi-Agent Reinforcement Learning Communication via Graph Learning, Attention based MADDPG, and reaction–diffusion PDE-inspired learning

Multi-agent reinforcement learning (MARL) is a key ingredient for scalable decision-making in robotics, autonomous driving, and complex control. However current communication mechanisms face critical bottlenecks: scalability, as attention- or GNN-based message passing often scales quadratically with team size, lack of interpretability, and insufficient robustness under noise. To address these issues, we propose Partial Differential Equation (PDE)-based MARL Communication, a field-theoretic framework where agents project features onto a shared spatial field that evolves via learnable reaction-diffusion dynamics. We demonstrate competitive performance on MaMuJoCo with near-linear scaling, field-level interpretability, and improved robustness to observation noise compared to attention and GNN baselines under PPO and MADDPG backbones. Our contributions are as follow: (1) a PDE-inspired communication module compatible with standard policy gradient and actor-critic algorithms; (2) a unified MaMuJoCo benchmark comparing graph-based, attention-based, and Neural-PDE communication under bandwidth constraints; and (3) empirical evidence that Neural-PDE communication can achieve competitive returns while offering field-level interpretability and improved robustness to variations in team size and communication-graph shifts.


---

````markdown
# MARL (MADDPG/PPO) on (Ma)MuJoCo

This repository contains a runnable implementation for training and evaluating **MADDPG/PPO** on multi-agent MuJoCo-style tasks (e.g., `Ant-v1`) with experiment logs and checkpoints saved for reproducibility.

Example run (your log):
- Actor params: 95,496
- Critic params: 194,561
- Train curve saved to `runs/Ant-v1_N4/train_curve.png`
- Logs at `runs/Ant-v1_N4/train_log.jsonl`
- Checkpoint at `runs/Ant-v1_N4/ckpt.pt`

---

## 1) Environment Setup

### 1.1 System dependencies (Ubuntu/Debian)
On headless servers, MuJoCo typically requires OpenGL/EGL-related libraries.

```bash
sudo apt-get update
sudo apt-get install -y \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
  mesa-utils mesa-utils-extra \
  libglfw3 libglfw3-dev \
  patchelf ffmpeg
````

> If you already have a working MuJoCo runtime on the cluster node, you can skip this step.

---

### 1.2 Create a clean Python environment (recommended)

We recommend Python **3.10** for maximum compatibility with Gymnasium + MuJoCo wheels.

#### Option A: Conda (recommended)

```bash
conda create -n marl python=3.10 -y
conda activate marl
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

#### Option B: venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

---

### 1.3 (IMPORTANT) Headless rendering / EGL (for servers)

Before running training/eval on a headless machine:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If you are on a desktop with display, you may omit these.

---

### 1.4 Quick sanity check

```bash
python -c "import mujoco, gymnasium as gym; env=gym.make('Ant-v1'); env.reset(); print('MuJoCo OK')"
```

---

## 2) Running Experiments

All experiments are driven by:

* `maddpg_mamujoco.py`

### 2.1 Train (example: Ant-v1, 4 agents)

This matches your command line and should reproduce the same output directory structure.

```bash
python maddpg_mamujoco.py --task Ant-v1 --n_agents 4 \
  --goal_x 10 --goal_y 0 \
  --episode_len 1000 --train_steps 300000 \
  --batch_size 512 --update_every 50 --gradient_steps 2 \
  --lr_actor 1e-4 --lr_critic 1e-4 \
  --exploration_std 0.30 --min_exploration_std 0.05 --exploration_decay 0.9998 \
  --seed 0 --cpu
```

Expected artifacts:

* `runs/Ant-v1_N4/train_curve.png`
* `runs/Ant-v1_N4/train_log.jsonl`
* `runs/Ant-v1_N4/ckpt.pt`

> Remove `--cpu` to use GPU (if your code supports CUDA).
> For reproducibility, keep `--seed` fixed.

---

### 2.2 Evaluate (load checkpoint)

```bash
python maddpg_mamujoco.py --eval_only 1 \
  --use_mamujoco 0 \
  --task Ant-v1 --n_agents 4 \
  --goal_x 10 --goal_y 0 --episode_len 1000 \
  --load_ckpt runs/Ant-v1_N4/ckpt.pt \
  --eval_episodes 10 --cpu
```

---

## 3) Reproducibility Notes

### 3.1 Determinism

To maximize reproducibility:

* Fix `--seed`
* Prefer `--cpu` for exact reproducibility across machines
* GPU runs can be slightly nondeterministic depending on CUDA/cuDNN/driver versions

If your script exposes deterministic flags (or you add them), recommended settings are:

* `torch.backends.cudnn.deterministic = True`
* `torch.backends.cudnn.benchmark = False`

### 3.2 Logs

`train_log.jsonl` is a JSON-lines file: one JSON object per line (step/episode metrics).
This is intended for paper plots and later aggregation.

### 3.3 Output directory convention

By default, outputs go under:

* `runs/{task}_N{n_agents}/...`

If you add new tasks or agents, keep this convention to simplify sweeping and plotting.

---

## 4) Common Troubleshooting

### 4.1 `MUJOCO_GL` / OpenGL errors on servers

Set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

and ensure `libegl1-mesa-dev` / `libgl1-mesa-dev` are installed.

### 4.2 `Ant-v1` not found

Make sure you installed Gymnasium with MuJoCo extras:

* `gymnasium[mujoco]==0.29.1` in `requirements.txt`

### 4.3 Torch installation mismatch (CUDA)

If `pip install -r requirements.txt` fails due to PyTorch wheels:

* Install PyTorch first using the official command for your CUDA version
* Then install the rest:

  ```bash
  pip install -r requirements.txt --no-deps
  pip install -r requirements.txt
  ```

(Or edit `requirements.txt` to remove `torch==...` and install torch separately.)

---

## 5) Reference Command Snippets (copy/paste)

Headless EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Train:

```bash
python maddpg_mamujoco.py --task Ant-v1 --n_agents 4 \
  --goal_x 10 --goal_y 0 \
  --episode_len 1000 --train_steps 300000 \
  --batch_size 512 --update_every 50 --gradient_steps 2 \
  --lr_actor 1e-4 --lr_critic 1e-4 \
  --exploration_std 0.30 --min_exploration_std 0.05 --exploration_decay 0.9998 \
  --seed 0 --cpu
```

Eval:

```bash
python maddpg_mamujoco.py --eval_only 1 \
  --use_mamujoco 0 --task Ant-v1 --n_agents 4 \
  --goal_x 10 --goal_y 0 --episode_len 1000 \
  --load_ckpt runs/Ant-v1_N4/ckpt.pt \
  --eval_episodes 10 --cpu
```

