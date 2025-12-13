# Multi-Agent Reinforcement Learning Communication via Graph Learning, Attention based MADDPG, and reaction–diffusion PDE-inspired learning

Multi-agent reinforcement learning (MARL) is a key ingredient for scalable decision-making in robotics, autonomous driving, and complex control. However current communication mechanisms face critical bottlenecks: scalability, as attention- or GNN-based message passing often scales quadratically with team size, lack of interpretability, and insufficient robustness under noise. To address these issues, we propose Partial Differential Equation (PDE)-based MARL Communication, a field-theoretic framework where agents project features onto a shared spatial field that evolves via learnable reaction-diffusion dynamics. We demonstrate competitive performance on MaMuJoCo with near-linear scaling, field-level interpretability, and improved robustness to observation noise compared to attention and GNN baselines under PPO and MADDPG backbones. Our contributions are as follow: (1) a PDE-inspired communication module compatible with standard policy gradient and actor-critic algorithms; (2) a unified MaMuJoCo benchmark comparing graph-based, attention-based, and Neural-PDE communication under bandwidth constraints; and (3) empirical evidence that Neural-PDE communication can achieve competitive returns while offering field-level interpretability and improved robustness to variations in team size and communication-graph shifts.

# Run Code

See .md files in each folder.

Due to Open Review's upload size limitation, we put all of the checkpoints on https://github.com/Akinesia112/MARL_Communication/tree/main

