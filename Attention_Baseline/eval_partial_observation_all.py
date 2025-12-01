#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from gymnasium_robotics import mamujoco_v1


class MultiAlgoPartialObsEvaluator:
    """
    統一對 8 組 policy 做：
    - Partial Observation (noise / occlusion)
    - Attention / Comm 可視化
    - Robustness 曲線

    這個 class 不管「怎麼訓練」，只負責：
    - 環境 rollout（Ant, 4x2, partial obs）
    - 呼叫你註冊好的 act_fn
    """

    def __init__(self,
                 scenario: str = "Ant",
                 agent_conf: str = "4x2",
                 max_episode_steps: int = 300):

        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=1,          # partial obs already from env
            render_mode=None,
            max_episode_steps=max_episode_steps,
        )

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.max_episode_steps = max_episode_steps

        example_agent = self.agents[0]
        self.obs_dim = self.env.observation_space(example_agent).shape[0]
        self.action_dim = self.env.action_space(example_agent).shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # models[name] = dict(agent -> torch.nn.Module)
        self.models: Dict[str, Dict[str, torch.nn.Module]] = {}

        # act_fns[name] = callable(obs_dict, noise_level, occlusion_mask) -> actions_dict
        self.act_fns: Dict[str, Callable] = {}

        os.makedirs("png/evaluation_results", exist_ok=True)

    # ------------------------------------------------------------------
    # 註冊 / 載入 models
    # ------------------------------------------------------------------

    def register_model(self,
                       name: str,
                       policies: Dict[str, torch.nn.Module],
                       act_fn: Callable):
        """
        name:     e.g. "ppo_mlp_no_comm"
        policies: dict[agent] -> policy module (已經 load_state_dict + eval())
        act_fn:   callable(obs_dict, policies, device, noise_level, occlusion_mask) -> actions_dict
        """
        self.models[name] = policies
        self.act_fns[name] = act_fn

    # ------------------------------------------------------------------
    # 共用 rollout：給 noise / occlusion
    # ------------------------------------------------------------------

    def _rollout_one_episode(self,
                             model_name: str,
                             noise_level: float = 0.0,
                             occlusion_mask: torch.Tensor = None) -> float:
        """
        回傳：平均每個 agent 的 episode return（scalar）
        """
        assert model_name in self.models, f"{model_name} 尚未註冊"

        policies = self.models[model_name]
        act_fn = self.act_fns[model_name]

        obs_dict, _ = self.env.reset()
        total_reward = 0.0

        for t in range(self.max_episode_steps):
            # occlusion mask: [obs_dim] on device
            if occlusion_mask is not None:
                masked_obs_dict = {}
                for agent in self.agents:
                    o = torch.from_numpy(obs_dict[agent]).float().to(self.device)
                    o = o * occlusion_mask
                    masked_obs_dict[agent] = o.cpu().numpy()
            else:
                masked_obs_dict = obs_dict

            # 交給 act_fn 處理 noise + action
            actions_dict = act_fn(
                masked_obs_dict,
                policies,
                self.device,
                noise_level,
                occlusion_mask,
            )

            next_obs_dict, rewards, terms, truncs, _ = self.env.step(actions_dict)

            total_reward += sum(rewards.values())
            obs_dict = next_obs_dict

            if any(terms.values()) or any(truncs.values()):
                break

        return total_reward / self.n_agents

    # ------------------------------------------------------------------
    # Robustness 1: noise
    # ------------------------------------------------------------------

    def test_with_noise(self,
                        model_names,
                        noise_levels=None,
                        n_episodes: int = 5):
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        results = {name: [] for name in model_names}

        print("\n[Robustness] Noise 測試...")
        for noise in noise_levels:
            print(f"  Noise = {noise:.2f}")
            for name in model_names:
                ep_returns = []
                for _ in range(n_episodes):
                    ep_ret = self._rollout_one_episode(
                        model_name=name,
                        noise_level=noise,
                        occlusion_mask=None,
                    )
                    ep_returns.append(ep_ret)
                avg_ret = float(np.mean(ep_returns))
                results[name].append(avg_ret)
                print(f"    {name:30s}  avg_return = {avg_ret:.2f}")

        # 畫圖
        plt.figure(figsize=(8, 6))
        for name in model_names:
            plt.plot(noise_levels, results[name], marker="o", label=name)
        plt.xlabel("Noise Level (σ of N(0, σ))")
        plt.ylabel("Average Episode Return per Agent")
        plt.title("Robustness to Observation Noise")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("png/evaluation_results/robustness_noise_all_models.png", dpi=150)
        plt.close()

        # 存 JSON
        with open("png/evaluation_results/robustness_noise_all_models.json", "w") as f:
            json.dump({"noise_levels": noise_levels, "results": results}, f, indent=2)

        print("✓ Noise robustness 結果已存到 png/evaluation_results/")

    # ------------------------------------------------------------------
    # Robustness 2: occlusion
    # ------------------------------------------------------------------

    def test_with_occlusion(self,
                            model_names,
                            occlusion_ratios=None,
                            n_episodes: int = 5):
        if occlusion_ratios is None:
            occlusion_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        results = {name: [] for name in model_names}

        print("\n[Robustness] Occlusion 測試...")
        for ratio in occlusion_ratios:
            print(f"  Occlusion ratio = {ratio:.2f}")

            # 固定一個 mask，避免每步都在變
            mask = (torch.rand(self.obs_dim) > ratio).float().to(self.device)

            for name in model_names:
                ep_returns = []
                for _ in range(n_episodes):
                    ep_ret = self._rollout_one_episode(
                        model_name=name,
                        noise_level=0.0,
                        occlusion_mask=mask,
                    )
                    ep_returns.append(ep_ret)
                avg_ret = float(np.mean(ep_returns))
                results[name].append(avg_ret)
                print(f"    {name:30s}  avg_return = {avg_ret:.2f}")

        # 畫圖
        plt.figure(figsize=(8, 6))
        for name in model_names:
            plt.plot(occlusion_ratios, results[name], marker="o", label=name)
        plt.xlabel("Occlusion Ratio")
        plt.ylabel("Average Episode Return per Agent")
        plt.title("Robustness to Observation Occlusion")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("png/evaluation_results/robustness_occlusion_all_models.png", dpi=150)
        plt.close()

        with open("png/evaluation_results/robustness_occlusion_all_models.json", "w") as f:
            json.dump({"occlusion_ratios": occlusion_ratios, "results": results}, f, indent=2)

        print("✓ Occlusion robustness 結果已存到 png/evaluation_results/")

    # ------------------------------------------------------------------
    # 可解釋性：Attention / Comm 可視覺化
    # ------------------------------------------------------------------

    def visualize_attention_like(self,
                                 model_name: str,
                                 focus_agent_idx: int = 0,
                                 n_steps: int = 3,
                                 tag: str = ""):
        """
        假設 act_fn 在 extra_info 裡會回傳:
            extra_info["att_weights"] : [n_agents] 或 [n_heads, n_agents]
            extra_info["msg_probs"]   : [msg_len, vocab_size] (可選)
        我們只負責畫圖。
        """
        assert model_name in self.models, f"{model_name} 尚未註冊"

        policies = self.models[model_name]
        act_fn = self.act_fns[model_name]

        obs_dict, _ = self.env.reset()

        for step_id in range(n_steps):
            # 直接用完整 observation，不加 noise / occlusion
            actions_dict, extra = act_fn(
                obs_dict,
                policies,
                self.device,
                noise_level=0.0,
                occlusion_mask=None,
                return_extra=True,   # 需要你在 act_fn 支援這個 flag
            )

            # 畫 attention weight（假設是 1D, 長度 = n_agents）
            if "att_weights" in extra:
                w = extra["att_weights"]  # np.array [n_agents]
                plt.figure(figsize=(6, 4))
                plt.bar(range(len(w)), w)
                plt.ylim(0, 1)
                plt.xlabel("Agent index")
                plt.ylabel("Attention weight")
                plt.title(f"{model_name} | step={step_id} {tag}")
                plt.tight_layout()
                fname = f"png/evaluation_results/att_{model_name}_step{step_id}{tag}.png"
                plt.savefig(fname, dpi=150)
                plt.close()

            # 若有 emergent message 分布也可以畫熱圖
            if "msg_probs" in extra:
                probs = extra["msg_probs"]  # [msg_len, vocab_size]
                plt.figure(figsize=(4, 4))
                im = plt.imshow(probs, aspect="auto", origin="lower")
                plt.colorbar(im)
                plt.xlabel("Vocab index")
                plt.ylabel("Msg position")
                plt.title(f"{model_name} msg_probs | step={step_id} {tag}")
                plt.tight_layout()
                fname = f"png/evaluation_results/msg_{model_name}_step{step_id}{tag}.png"
                plt.savefig(fname, dpi=150)
                plt.close()
            
            if "field_energy" in extra:
                fe = extra["field_energy"]  # [H, W]
                plt.figure(figsize=(4, 4))
                im = plt.imshow(fe, origin="lower", aspect="auto")
                plt.colorbar(im)
                plt.title(f"{model_name} field_energy | step={step_id} {tag}")
                plt.tight_layout()
                fname = f"evaluation_results/field_{model_name}_step{step_id}{tag}.png"
                plt.savefig(fname, dpi=150)
                plt.close()

            # roll env with chosen actions
            obs_dict, _, terms, truncs, _ = self.env.step(actions_dict)
            if any(terms.values()) or any(truncs.values()):
                obs_dict, _ = self.env.reset()

        print(f"✓ {model_name} attention / msg 可視覺化完成")


# ----------------------------------------------------------------------
# 範例：如何註冊 8 組 model 的 act_fn
# （這裡只給骨架，你把各自的 policy / checkpoint load 進來即可）
# ----------------------------------------------------------------------

def make_act_fn_ppo_mlp_no_comm():
    """
    給 PPO-MLP-NoComm 使用：
    - 不用 all_obs
    - 不用 agent_idx
    - 只對 obs + noise 做 forward
    """

    def act_fn(obs_dict, policies, device, noise_level, occlusion_mask,
               return_extra=False):
        actions = {}
        for agent in policies.keys():
            o = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
            if noise_level > 0:
                o = o + torch.randn_like(o) * noise_level
            with torch.no_grad():
                dist = policies[agent](o)          # e.g. SimplePolicy(obs)->dist
                a = dist.mean                      # 用 mean 比較穩定
            actions[agent] = a.cpu().numpy()[0]
        if return_extra:
            return actions, {}
        return actions

    return act_fn


def make_act_fn_maddpg_mlp_comm():
    """
    給 MADDPG-MLP-Comm 使用：
    - CommPolicyMLP(obs, all_obs=None, self_index=None) -> (dist, comm_info)
    - 這裡 all_obs / self_index 只是 for 介面相容
    """

    def act_fn(obs_dict, policies, device, noise_level, occlusion_mask,
               return_extra=False):
        # 準備 all_obs tensor（雖然 CommPolicyMLP 不用）
        obs_all = np.array([obs_dict[a] for a in policies.keys()], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

        actions = {}
        last_comm_info = None

        for idx, agent in enumerate(policies.keys()):
            o = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
            if noise_level > 0:
                o = o + torch.randn_like(o) * noise_level
            with torch.no_grad():
                dist, comm_info = policies[agent](o, obs_all_tensor, self_index=idx)
                a = dist.mean
            actions[agent] = a.cpu().numpy()[0]
            last_comm_info = comm_info  # 只拿一個代表

        extra = {}
        if return_extra and last_comm_info is not None:
            probs = last_comm_info["msg_probs"][0].cpu().numpy()   # [L, V]
            extra["msg_probs"] = probs

        if return_extra:
            return actions, extra
        return actions

    return act_fn


def make_act_fn_attention_comm_or_nocomm(use_self_index: bool):
    """
    給 PPO-Attention / MADDPG-Attention 用：
    - PolicyWithAttention: forward -> (dist, att_weights_tensor)
    - CommPolicyWithAttention: forward -> (dist, extra_dict)
        extra_dict 內可能有: "att_weights", "msg_probs"
    """

    def act_fn(obs_dict, policies, device, noise_level, occlusion_mask,
               return_extra=False):
        obs_all = np.array([obs_dict[a] for a in policies.keys()], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

        actions = {}
        last_extra = None

        for idx, agent in enumerate(policies.keys()):
            o = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
            if noise_level > 0:
                o = o + torch.randn_like(o) * noise_level

            with torch.no_grad():
                if use_self_index:
                    dist, extra = policies[agent](o, obs_all_tensor, self_index=idx)
                else:
                    dist, extra = policies[agent](o, obs_all_tensor)
                a = dist.mean

            actions[agent] = a.cpu().numpy()[0]
            last_extra = extra  # 可能是 dict，也可能直接是 Tensor

        extra_out = {}
        if return_extra and last_extra is not None:
            # 情況 1：CommPolicyWithAttention，extra 是 dict
            if isinstance(last_extra, dict):
                if "att_weights" in last_extra:
                    w = last_extra["att_weights"]
                    # w 可能是 [1, n_agents] 或 [batch, ...]
                    if torch.is_tensor(w):
                        w = w[0].detach().cpu().numpy()
                    extra_out["att_weights"] = w
                if "msg_probs" in last_extra:
                    mp = last_extra["msg_probs"]
                    if torch.is_tensor(mp):
                        mp = mp[0].detach().cpu().numpy()
                    extra_out["msg_probs"] = mp

            # 情況 2：PolicyWithAttention，extra 直接是 att_weights Tensor
            elif torch.is_tensor(last_extra):
                w = last_extra
                if w.dim() > 1:
                    w = w[0]
                extra_out["att_weights"] = w.detach().cpu().numpy()

        if return_extra:
            return actions, extra_out
        return actions

    return act_fn

def make_act_fn_pde_comm():
    """
    給 PDE Communication 使用：
    PolicyWithPDE(obs, all_obs, agent_idx) -> (dist, field)
      - field: [B, C, H, W] 之類，可以算 energy heatmap
    extra:
      - "field_energy": [H, W]  (拿第 0 batch 做視覺化)
    """
    def act_fn(obs_dict, policies, device, noise_level, occlusion_mask,
               return_extra=False):
        agents = list(policies.keys())
        obs_all = np.array([obs_dict[a] for a in agents], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

        actions = {}
        last_field = None

        for idx, agent in enumerate(agents):
            o = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
            if noise_level > 0:
                o = o + torch.randn_like(o) * noise_level

            with torch.no_grad():
                dist, field = policies[agent](o, obs_all_tensor, agent_idx=idx)
                a = dist.mean
            actions[agent] = a.cpu().numpy()[0]
            last_field = field

        extra_out = {}
        if return_extra and last_field is not None:
            # 取 batch=0, 把 channel square sum 當 energy map
            fe = (last_field[0] ** 2).sum(dim=0).detach().cpu().numpy()
            extra_out["field_energy"] = fe

        if return_extra:
            return actions, extra_out
        return actions

    return act_fn

def make_act_fn_gnn(use_self_index: bool):
    """
    給 GCNPolicy / GATPolicy / CommGCNPolicy / CommGATPolicy 共用：
    forward(obs, all_obs, self_index) -> (dist, extra)
      extra 會包含 att_weights / msg_probs （對應前面 visualize_attention_like）
    """
    def act_fn(obs_dict, policies, device, noise_level, occlusion_mask,
               return_extra=False):
        agents = list(policies.keys())
        obs_all = np.array([obs_dict[a] for a in agents], dtype=np.float32)
        obs_all_tensor = torch.from_numpy(obs_all).unsqueeze(0).to(device)

        actions = {}
        last_extra = None

        for idx, agent in enumerate(agents):
            o = torch.from_numpy(obs_dict[agent]).float().unsqueeze(0).to(device)
            if noise_level > 0:
                o = o + torch.randn_like(o) * noise_level

            with torch.no_grad():
                if use_self_index:
                    dist, extra = policies[agent](o, obs_all_tensor, self_index=idx)
                else:
                    # GCN/GAT 也可以只用 self_index=0，如果你 forward 不需要
                    dist, extra = policies[agent](o, obs_all_tensor, self_index=idx)
                a = dist.mean
            actions[agent] = a.cpu().numpy()[0]
            last_extra = extra

        extra_out = {}
        if return_extra and last_extra is not None:
            if "att_weights" in last_extra:
                extra_out["att_weights"] = last_extra["att_weights"][0].cpu().numpy()
            if "msg_probs" in last_extra:
                extra_out["msg_probs"] = last_extra["msg_probs"][0].cpu().numpy()

        if return_extra:
            return actions, extra_out
        return actions

    return act_fn



if __name__ == "__main__":
    """
    1. 建一個 evaluator
    2. 把 8 組 policy 都 load 進來並 register
    3. 呼叫 test_with_noise / test_with_occlusion / visualize_attention_like
    """

    evaluator = MultiAlgoPartialObsEvaluator(
        scenario="Ant",
        agent_conf="4x2",
        max_episode_steps=300,
    )

    # ===== 0. 共用變數 & policy class import =====
    device = evaluator.device
    agents = evaluator.agents
    obs_dim = evaluator.obs_dim
    action_dim = evaluator.action_dim

    from baseline_no_comm import SimplePolicy
    from attention_comm import PolicyWithAttention
    from attention_emergent_comm import CommPolicyWithAttention
    from train_maddpg_mlp_comm import CommPolicyMLP
    from policy_with_pde import PolicyWithPDE
    from gnn_policies import (
        GCNPolicy,
        GATPolicy,
        CommGCNPolicy,
        CommGATPolicy,
    )


    # ============================================================
    # 1) PPO-MLP  No Communication
    #    train_ppo_no_comm.py 對應的 checkpoint
    # ============================================================
    ppo_mlp_no_comm_policies = {
        agent: SimplePolicy(obs_dim, action_dim).to(device) for agent in agents
    }

    ckpt = torch.load(
        # TODO: 如檔名不同，這裡改掉
        "checkpoints_ppo_no_comm_ant_4x2/ppo_nocomm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_mlp_no_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_mlp_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_mlp_no_comm",
        policies=ppo_mlp_no_comm_policies,
        act_fn=make_act_fn_ppo_mlp_no_comm(),   # self-observation Gaussian policy
    )

    # ============================================================
    # 2) PPO-「Comm」 (實際架構是 CommPolicyWithAttention)
    #    對應 train_ppo_comm.py → CommPolicyWithAttention
    # ============================================================
    ppo_mlp_comm_policies = {
        agent: CommPolicyWithAttention(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_comm_ant_4x2/ppo_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_mlp_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_mlp_comm_policies[agent].eval()

    # 推論介面跟 attention family 一樣，用 all_obs + (optionally) self_index
    evaluator.register_model(
        name="ppo_mlp_comm",
        policies=ppo_mlp_comm_policies,
        act_fn=make_act_fn_attention_comm_or_nocomm(use_self_index=False),
    )


    # ============================================================
    # 3) MADDPG-MLP  No Communication
    #    train_maddpg_mlp_no_comm.py
    # ============================================================
    maddpg_mlp_no_comm_policies = {
        agent: SimplePolicy(obs_dim, action_dim).to(device) for agent in agents
    }

    ckpt = torch.load(
        # 這個路徑你之前在 eval_maddpg_mlp_no_comm.py 用過，應該存在
        "checkpoints_maddpg_mlp_no_comm_ant_4x2/maddpg_nocomm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        maddpg_mlp_no_comm_policies[agent].load_state_dict(ckpt[agent])
        maddpg_mlp_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="maddpg_mlp_no_comm",
        policies=maddpg_mlp_no_comm_policies,
        act_fn=make_act_fn_ppo_mlp_no_comm(),   # 同樣是 self-only MLP Gaussian policy
    )

    # ============================================================
    # 4) MADDPG-MLP  Comm with Limitations
    #    對應 train_maddpg_mlp_comm.py → CommPolicyMLP
    # ============================================================
    maddpg_mlp_comm_policies = {
        agent: CommPolicyMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_maddpg_mlp_comm_ant_4x2/maddpg_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        maddpg_mlp_comm_policies[agent].load_state_dict(ckpt[agent])
        maddpg_mlp_comm_policies[agent].eval()

    evaluator.register_model(
        name="maddpg_mlp_comm",
        policies=maddpg_mlp_comm_policies,
        act_fn=make_act_fn_maddpg_mlp_comm(),  # 注意：用專門給 CommPolicyMLP 的 act_fn
    )

    # ============================================================
    # 5) PPO-Attention  No Communication
    #    train_ppo_attention_no_comm.py
    #    👉 注意：訓練時用的是 CommPolicyWithAttention，不是 PolicyWithAttention
    # ============================================================
    from attention_emergent_comm import CommPolicyWithAttention  # 確保有 import

    ppo_attn_no_comm_policies = {
        agent: CommPolicyWithAttention(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_attention_no_comm_ant_4x2/model_iter_ppo_attention_no_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        # ckpt 是 {agent_name: state_dict}
        ppo_attn_no_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_attn_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_attn_no_comm",
        policies=ppo_attn_no_comm_policies,
        # 不用 self_index 的版本（跟你上面寫的一樣）
        act_fn=make_act_fn_attention_comm_or_nocomm(use_self_index=False),
    )


    # ============================================================
    # 6) PPO-Attention  Comm with Limitations（Emergent Comm）
    #    train_ppo_attention_comm.py
    #    使用 CommPolicyWithAttention，支援 msg_len / vocab_size 等限制
    # ============================================================
    ppo_attn_comm_policies = {
        agent: CommPolicyWithAttention(
            obs_dim,
            action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        # TODO: 依 train_ppo_attention_comm.py 實際輸出修改
        "checkpoints_ppo_attention_comm_ant_4x2/model_iter_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_attn_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_attn_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_attn_comm",
        policies=ppo_attn_comm_policies,
        act_fn=make_act_fn_attention_comm_or_nocomm(use_self_index=False),
    )

    # ============================================================
    # 7) MADDPG-Attention  No Communication
    #    train_maddpg_attention_no_comm.py
    # ============================================================
    maddpg_attn_no_comm_policies = {
        agent: PolicyWithAttention(
            obs_dim,
            action_dim,
            hidden_dim=256,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        # 你之前 eval_maddpg_attention_no_comm_ant_4x2 時用過的路徑
        "checkpoints_maddpg_attention_no_comm_ant_4x2/maddpg_attention_no_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        maddpg_attn_no_comm_policies[agent].load_state_dict(ckpt[agent])
        maddpg_attn_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="maddpg_attn_no_comm",
        policies=maddpg_attn_no_comm_policies,
        act_fn=make_act_fn_attention_comm_or_nocomm(use_self_index=False),
    )

    # ============================================================
    # 8) MADDPG-Attention  Comm with Limitations
    #    train_maddpg_attention_comm.py
    # ============================================================
    maddpg_attn_comm_policies = {
        agent: CommPolicyWithAttention(
            obs_dim,
            action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        # TODO: 依 train_maddpg_attention_comm.py 實際輸出修改
        "checkpoints_maddpg_attention_comm_ant_4x2/maddpg_attention_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        maddpg_attn_comm_policies[agent].load_state_dict(ckpt[agent])
        maddpg_attn_comm_policies[agent].eval()

    evaluator.register_model(
        name="maddpg_attn_comm",
        policies=maddpg_attn_comm_policies,
        act_fn=make_act_fn_attention_comm_or_nocomm(use_self_index=True),
    )

    # ============================================================
    # 9) PDE Communication
    #    train_with_pde.py 產生的 checkpoint
    # ============================================================
    pde_policies = {
        agent: PolicyWithPDE(obs_dim, action_dim).to(device) for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_pde/model_iter_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        # train_with_pde.py: ckpt["policies"][agent]
        pde_policies[agent].load_state_dict(ckpt["policies"][agent])
        pde_policies[agent].eval()

    evaluator.register_model(
        name="pde_comm",
        policies=pde_policies,
        act_fn=make_act_fn_pde_comm(),
    )

    # ============================================================
    # 10) PPO-GAT No Communication
    # ============================================================
    ppo_gat_no_comm_policies = {
        agent: GATPolicy(obs_dim, action_dim, hidden_dim=256).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_gat_no_comm_ant_4x2/ppo_gat_no_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_gat_no_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_gat_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_gat_no_comm",
        policies=ppo_gat_no_comm_policies,
        act_fn=make_act_fn_gnn(use_self_index=True),
    )

    # ============================================================
    # 11) PPO-GAT Comm with Limitations
    # ============================================================
    ppo_gat_comm_policies = {
        agent: CommGATPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_gat_comm_ant_4x2/ppo_gat_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_gat_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_gat_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_gat_comm",
        policies=ppo_gat_comm_policies,
        act_fn=make_act_fn_gnn(use_self_index=True),
    )

    # ============================================================
    # 12) PPO-GCN No Communication
    # ============================================================
    ppo_gcn_no_comm_policies = {
        agent: GCNPolicy(obs_dim, action_dim, hidden_dim=256).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_gcn_no_comm_ant_4x2/ppo_gcn_no_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_gcn_no_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_gcn_no_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_gcn_no_comm",
        policies=ppo_gcn_no_comm_policies,
        act_fn=make_act_fn_gnn(use_self_index=True),
    )

    # ============================================================
    # 13) PPO-GCN Comm with Limitations
    # ============================================================
    ppo_gcn_comm_policies = {
        agent: CommGCNPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            msg_len=4,
            vocab_size=8,
            comm_tau=1.0,
            comm_hard=True,
        ).to(device)
        for agent in agents
    }

    ckpt = torch.load(
        "checkpoints_ppo_gcn_comm_ant_4x2/ppo_gcn_comm_best.pt",
        map_location=device,
        weights_only=False,
    )
    for agent in agents:
        ppo_gcn_comm_policies[agent].load_state_dict(ckpt[agent])
        ppo_gcn_comm_policies[agent].eval()

    evaluator.register_model(
        name="ppo_gcn_comm",
        policies=ppo_gcn_comm_policies,
        act_fn=make_act_fn_gnn(use_self_index=True),
    )



    # ============================================================
    # 2. Robustness 實驗（Noise + Occlusion）
    # ============================================================
    model_names = [
        "ppo_mlp_no_comm",
        "ppo_mlp_comm",
        "ppo_attn_no_comm",
        "ppo_attn_comm",
        "maddpg_mlp_no_comm",
        "maddpg_mlp_comm",
        "maddpg_attn_no_comm",
        "maddpg_attn_comm",
        "pde_comm",
        "ppo_gat_no_comm",
        "ppo_gat_comm",
        "ppo_gcn_no_comm",
        "ppo_gcn_comm",
    ]


    evaluator.test_with_noise(model_names=model_names, n_episodes=5)
    evaluator.test_with_occlusion(model_names=model_names, n_episodes=5)

    # ============================================================
    # 3. 可解釋性視覺化（只有 Attention / Comm 家族）
    #    這裡先對 full observation 畫 attention / msg 分布
    #    如果你想在 partial observation 下也畫，可以另外寫一個 wrapper
    # ============================================================
    for name in [
        "ppo_attn_no_comm",
        "ppo_attn_comm",
        "maddpg_attn_no_comm",
        "maddpg_attn_comm",
        "ppo_gat_no_comm",
        "ppo_gat_comm",
        "ppo_gcn_no_comm",
        "ppo_gcn_comm",
        "pde_comm",   # PDE 用 field_energy 分支畫
    ]:
        evaluator.visualize_attention_like(name, tag="_full_obs")
