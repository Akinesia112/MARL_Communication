# four_ants_env.py
import os
import numpy as np
import mujoco
import mujoco.viewer  # optional (headless時不會用到)
import gymnasium as gym
from gymnasium import spaces

ASSET_XML = os.path.join(os.path.dirname(__file__), "four_ants.xml")

ANT_PREFIXES = ["ant1", "ant2", "ant3", "ant4"]

def _xy_from_qpos(qpos):
    # MuJoCo free joint: 7 params (x,y,z, qw,qx,qy,qz)
    return float(qpos[0]), float(qpos[1])

def _obj_name(model, obj_type, i):
    """跨版本安全地拿名稱；MuJoCo 會回 str 或 bytes。"""
    n = mujoco.mj_id2name(model, obj_type, i)
    if n is None:
        return ""
    return n.decode() if isinstance(n, (bytes, bytearray)) else str(n)

class FourAntsMAEnv(gym.Env):
    """
    單場景四隻 Ant；4 個 agent（每人控制一隻）。
    observation[i]: proprio of ant_i (不含root自由度) + goal dxdy
    action[i]: 該隻 8 個馬達的控制量（對應 xml)
    reward: sum over i of (progress_i - 1e-3 * ||a_i||^2)
    episode ends when any ant falls too low或到達步數上限
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, episode_len=1000, goal=(10.0, 0.0), render_mode="rgb_array"):
        self.episode_len = int(episode_len)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.render_mode = render_mode

        with open(ASSET_XML, "r", encoding="utf-8") as f:
            xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        
        all_act_names = [
            _obj_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        # 依前綴找出每隻的 motor id、joint qpos 索引
        self._ant = []
        for p in ANT_PREFIXES:
            motors = [i for i, nm in enumerate(all_act_names) if nm.startswith(f"{p}/")]
            # 注意：每隻一個 freejoint + 8 hinge。freejoint 的 qpos, qvel 是 7+6 維
            # 這裡記下該 freejoint 的 qpos 起點 index（找 joint 名稱含 "/root"）
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{p}/root")
            qpos_adr = self.model.jnt_qposadr[j_id]  # 起點
            qvel_adr = self.model.jnt_dofadr[j_id]
            self._ant.append(dict(prefix=p, motors=motors, qpos_adr=qpos_adr, qvel_adr=qvel_adr))

        # 每個 agent 的 action: 8 馬達
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # 觀測：去掉 freejoint 的 root 7/6 維，只保留其餘 joint 的 qpos/qvel
        # 為簡化，我們直接拿全局 qpos/qvel，然後「抽取該隻關節對應的 8*2 值」
        # 再加上 (goal - xy) 2 維
        # 實務上可把選擇的 qpos/qvel index 列表化；這裡直接給一個保守上限
        self._per_ant_proprio_dim = 8 * 2  # each joint qpos (1) + qvel (1)
        obs_dim = self._per_ant_proprio_dim + 2
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.observation_space = spaces.Tuple([self.single_observation_space] * 4)
        self.action_space = spaces.Tuple([self.single_action_space] * 4)

        self._step_count = 0
        self._last_xy = np.zeros((4, 2), dtype=np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)

    def _get_ant_xy(self, i):
        a = self._ant[i]
        x, y = _xy_from_qpos(self.data.qpos[a["qpos_adr"]:a["qpos_adr"]+7])
        return np.array([x, y], dtype=np.float32)

    def _get_ant_proprio(self, i):
        """
        取該隻的 8 個鉸鏈關節的 (qpos,qvel)；依 actuator 名稱順序抓 joint id。
        """
        a = self._ant[i]
        qps = []
        for m_id in a["motors"]:
            joint_id = self.model.actuator_trnid[m_id][0]
            adr_q = self.model.jnt_qposadr[joint_id]
            adr_v = self.model.jnt_dofadr[joint_id]
            qpos = float(self.data.qpos[adr_q])
            qvel = float(self.data.qvel[adr_v])
            qps += [qpos, qvel]
        qps = np.array(qps, dtype=np.float32)
        # 防止維度變動（若未滿 16 則補零；一般會恰好 16）
        if qps.shape[0] < self._per_ant_proprio_dim:
            qps = np.pad(qps, (0, self._per_ant_proprio_dim - qps.shape[0]))
        return qps[:self._per_ant_proprio_dim]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # 輕微隨機初始化
        self.data.qpos[:] += 0.01 * np.random.randn(self.model.nq)
        self.data.qvel[:] += 0.01 * np.random.randn(self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        obs_list = []
        for i in range(4):
            xy = self._get_ant_xy(i)
            self._last_xy[i] = xy
            proprio = self._get_ant_proprio(i)
            dxdy = self.goal - xy
            obs_list.append(np.concatenate([proprio, dxdy], axis=0).astype(np.float32))
        return tuple(obs_list), {}

    def step(self, actions):
        self._step_count += 1
        # actions: tuple of 4 arrays (8,)
        a_all = np.zeros(self.model.nu, dtype=np.float32)
        for i in range(4):
            motors = self._ant[i]["motors"]
            a = np.asarray(actions[i], dtype=np.float32).reshape(-1)
            a = np.clip(a, -1.0, 1.0)
            for k, m_id in enumerate(motors):
                a_all[m_id] = a[k] if k < len(a) else 0.0

        self.data.ctrl[:] = a_all
        mujoco.mj_step(self.model, self.data)

        obs_list, rew_list, done_flags = [], [], []
        for i in range(4):
            xy = self._get_ant_xy(i)
            last = self._last_xy[i]
            progress = np.linalg.norm(self.goal - last) - np.linalg.norm(self.goal - xy)
            # 控制成本（只算本隻的馬達）
            motors = self._ant[i]["motors"]
            ctrl = self.data.ctrl[motors]
            ctrl_cost = 1e-3 * float(np.sum(ctrl * ctrl))
            
            # --- 新增：平滑度懲罰 (直接讀取 MuJoCo data) ---
            
            # 1. 關節速度懲罰
            v_indices = []
            for m_id in motors:
                joint_id = self.model.actuator_trnid[m_id][0]
                adr_v = self.model.jnt_dofadr[joint_id]
                v_indices.append(adr_v)
            joint_velocities = self.data.qvel[v_indices]
            joint_vel_cost = 1e-4 * float(np.sum(np.square(joint_velocities))) # 可調權重

            # 2. 軀幹穩定性懲罰 (角速度)
            qvel_adr = self._ant[i]["qvel_adr"]
            torso_ang_vel = self.data.qvel[qvel_adr+3 : qvel_adr+6] # x,y,z angular vel
            torso_stability_cost = 1e-3 * float(np.sum(np.square(torso_ang_vel))) # 可調權重
            
            r = float(progress - ctrl_cost - joint_vel_cost - torso_stability_cost)


            # 觀測
            proprio = self._get_ant_proprio(i)
            dxdy = self.goal - xy
            obs_list.append(np.concatenate([proprio, dxdy], axis=0).astype(np.float32))

            self._last_xy[i] = xy
            rew_list.append(r)

            # 簡單結束條件：身體太低 or NaN
            torso_body = f"{ANT_PREFIXES[i]}/torso"
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_body)
            z = self.data.xpos[b_id][2]
            done_flags.append(bool((z < 0.12) or not np.isfinite(r)))

        team_reward = float(np.sum(rew_list))
        terminated = any(done_flags) or (self._step_count >= self.episode_len)
        truncated = False
        info = {}

        return tuple(obs_list), team_reward, terminated, truncated, info

    def render(self):
        # 可選：視需要開 viewer
        pass

    def render_frame(self):
        H, W = 480, 640
        img = np.zeros((H, W, 3), dtype=np.uint8)
        mujoco.mjr_maxViewport(self.model.vis.global_.offwidth, self.model.vis.global_.offheight)
        # 直接用 offscreen render
        if not hasattr(self, "_ctx"):
            self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._scn = mujoco.MjvScene(self.model, maxgeom=2000)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.distance = 8.0
            self._cam.elevation = -20
            self._cam.azimuth = 110
        viewport = mujoco.MjrRect(0, 0, W, H)
        mujoco.mjv_updateScene(self.model, self.data, self._cam, None,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self._scn)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mujoco.mjr_render(viewport, self._scn, self._ctx)
        mujoco.mjr_readPixels(rgb, None, viewport, self._ctx)
        rgb = np.flipud(rgb)
        return rgb

    def close(self):
        pass


# ---- Gym 註冊 ----
import gymnasium.envs.registration as reg

if "FourAntsMA-v0" not in reg.registry:
    reg.register(
        id="FourAntsMA-v0",
        entry_point=lambda **kw: FourAntsMAEnv(**kw),
        max_episode_steps=1000
    )
