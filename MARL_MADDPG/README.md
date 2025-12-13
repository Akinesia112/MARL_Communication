## README.md

````markdown
# MADDPG/PPO + Attention Emergent Communication (MaMuJoCo Ant)

This repo provides:
1) **Training**: MADDPG with attention-based emergent communication (`train_maddpg_attention_comm.py`)
2) **Plotting**: training curves from `.npz` logs (`plot_maddpg_attention_comm_logs.py`)
3) **Evaluation/Video**: load a checkpoint and record MP4 (`eval_maddpg_attention_comm.py`)

---

## 0. Environment (Headless EGL on server)

On headless servers (no display), set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3
````

Ubuntu/Debian system deps (recommended):

```bash
sudo apt-get update
sudo apt-get install -y \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
  mesa-utils mesa-utils-extra \
  libglfw3 libglfw3-dev \
  patchelf ffmpeg
```

---

Recommended: **Python 3.10**

```bash
conda create -n marl python=3.10 -y
conda activate marl
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "import gymnasium as gym; import mujoco; print('MuJoCo OK')"
python -c "from gymnasium_robotics import mamujoco_v1; env=mamujoco_v1.parallel_env(scenario='Ant', agent_conf='2x4'); env.reset(); env.close(); print('MaMuJoCo OK')"
```

---

## 1. Minimal Workflow (copy/paste)

```bash
# (1) headless (optional)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3

# (2) train
python train_maddpg_attention_comm.py

# (3) make plot script find the log
maddpg_attention_comm_logs.npz

# (4) plot
python plot_maddpg_attention_comm_logs.py

# (5) eval + record 5-min video
python eval_maddpg_attention_comm.py
```
---

## 2. Scripts Overview & Config

### 2.1 Training (`train_maddpg_attention_comm.py`)

**Important**: current code uses **hard-coded config** in `__main__`:

* Environment:

  * `scenario="Ant"`
  * `agent_conf="2x4"`
  * `agent_obsk=1`
  * `max_episode_steps=500`
* Algo:

  * `gamma=0.99`, `tau=0.01`
  * `actor_lr=1e-4`, `critic_lr=1e-3`
  * `batch_size=256`, `buffer_capacity=1e6`
  * exploration: `sigma=0.2`, `sigma_min=0.05`, `decay=1e-6`
  * comm: `lambda_comm=1e-3`
* CommPolicyWithAttention:

  * `hidden_dim=256`
  * `msg_len=4`, `vocab_size=8`
  * `comm_tau=1.0`, `comm_hard=True`
* Training loop:

  * `n_episodes=1000`
  * `updates_per_step=1`

Outputs:

* Logs: `maddpg_attention_logs.npz` (note: if you want to match plot script default name, see section 2.2)
* Checkpoints: `checkpoints_maddpg_attention_comm/maddpg_attention_comm_{best|epXXX}.pt`

Run:

```bash
python train_maddpg_attention_comm.py
```

#### (Recommended) Make the log filename consistent

`plot_maddpg_attention_comm_logs.py` expects:

* `maddpg_attention_comm_logs.npz`

But training currently saves:

* `maddpg_attention_logs.npz`

You have two options:

1. **Rename after training**:

```bash
mv maddpg_attention_logs.npz maddpg_attention_comm_logs.npz
```

2. Or edit one of the scripts to use the same filename.

---

### 2.2 Plotting (`plot_maddpg_attention_comm_logs.py`)

Default config inside script:

* `LOG_PATH = "maddpg_attention_comm_logs.npz"`
* Output dir: `png/maddpg_attention_comm/`

Run:

```bash
python plot_maddpg_attention_comm_logs.py
```

Outputs:

* `png/maddpg_attention_comm/reward.png`
* `png/maddpg_attention_comm/loss.png`
* `png/maddpg_attention_comm/sigma_buffer.png`

---

### 2.3 Evaluation + Video (`eval_maddpg_attention_comm.py`)

Default config inside script:

* checkpoint:

  * `checkpoint_path = "checkpoints_maddpg_attention_comm/maddpg_attention_comm_best.pt"`
* environment:

  * `scenario="Ant"`, `agent_conf="2x4"`, `agent_obsk=1`
  * `render_mode="rgb_array"`
  * `max_episode_steps=5000`
* recording:

  * fixed-duration recording: `total_seconds=300` (5 min), `fps=30`
  * output: `videos/{checkpoint_name}_fixed_300s.mp4`

Run:

```bash
python eval_maddpg_attention_comm.py
```

Outputs:

* `videos/maddpg_attention_comm_best_fixed_300s.mp4` (filename depends on checkpoint basename)

---

## 3. Reproducibility Tips

* Use the same Python version (3.10 recommended), same library versions (`requirements.txt`), and fixed random seeds if you add them.
* Headless servers: ensure EGL is enabled (env vars + mesa/egl libs).
* For paper reproduction, archive:

  * `requirements.txt`
  * training command / config block
  * `maddpg_attention_comm_logs.npz`
  * checkpoint `.pt`
