import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PDECommunication(nn.Module):
    """基於 2D Reaction-Diffusion 的空間場通訊（優化版）"""
    
    def __init__(self, feature_dim, grid_size=8, n_steps=2, dt=0.2, sigma=0.8):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.dt = dt
        self.sigma = sigma
        
        # 可學習參數
        self.diffusion_coef = nn.Parameter(torch.tensor(0.2))
        self.reaction_coef = nn.Parameter(torch.tensor(0.05))
        
        # 預計算 Laplacian kernel（重要！避免每次 forward 創建）
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        kernel = kernel.view(1, 1, 3, 3).repeat(feature_dim, 1, 1, 1)
        self.register_buffer('lap_kernel', kernel)
    
    def reaction_diffusion_step(self, field):
        """單步 PDE 更新: ∂u/∂t = D∇²u + R·u·(1-u)"""
        # 擴散項（使用預計算的 kernel）
        laplacian = F.conv2d(field, self.lap_kernel, padding=1, groups=self.feature_dim)
        diffusion = self.diffusion_coef * laplacian
        
        # 反應項（非線性）
        reaction = self.reaction_coef * field * (1 - field.tanh())
        
        # 時間積分
        field_next = field + self.dt * (diffusion + reaction)
        return field_next.clamp(-10, 10)
    
    def _compute_positions(self, n_agents):
        """計算 agent 在網格上的位置（圓形排列）"""
        cx = (self.grid_size - 1) / 2.0
        cy = (self.grid_size - 1) / 2.0
        radius = max(1.0, self.grid_size / 3.0)
        
        positions = []
        for i in range(n_agents):
            theta = 2.0 * math.pi * i / n_agents
            x = int(round(cx + radius * math.cos(theta)))
            y = int(round(cy + radius * math.sin(theta)))
            x = max(0, min(self.grid_size - 1, x))
            y = max(0, min(self.grid_size - 1, y))
            positions.append((x, y))
        return positions
    
    def agents_to_field(self, features):
        """簡化版 splatting: agent 特徵 → 場（只用中心點）"""
        batch, n_agents, feat = features.shape
        device = features.device
        field = torch.zeros(batch, feat, self.grid_size, self.grid_size, device=device)
        
        positions = self._compute_positions(n_agents)
        
        # 簡化：直接放中心點（移除 Gaussian splatting 的巢狀迴圈）
        for b in range(batch):
            for i, (px, py) in enumerate(positions):
                field[b, :, py, px] = features[b, i]
        
        return field, positions
    
    def field_to_agents(self, field, positions):
        """從場採樣 agent 位置的值（3x3 鄰域平均）"""
        batch, feat, h, w = field.shape
        n = len(positions)
        messages = torch.zeros(batch, n, feat, device=field.device)
        
        for i, (x, y) in enumerate(positions):
            # 3x3 鄰域平均
            vals = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = (x + dx) % self.grid_size
                    ny = (y + dy) % self.grid_size
                    vals.append(field[:, :, ny, nx])
            messages[:, i] = torch.stack(vals, dim=0).mean(dim=0)
        
        return messages
    
    def forward(self, agent_idx, all_features):
        """
        Args:
            agent_idx: int 或 [B] - 當前 agent 在 all_features 中的索引
            all_features: [B, n_agents, feature_dim]
        Returns:
            comm_message: [B, feature_dim] - 當前 agent 的通訊訊息
            field: [B, feature_dim, H, W] - PDE 場（用於視覺化）
        """
        batch, n_agents, feat = all_features.shape
        
        # 1. 建立空間場
        field, positions = self.agents_to_field(all_features)
        
        # 2. PDE 演化（資訊擴散）
        for _ in range(self.n_steps):
            field = self.reaction_diffusion_step(field)
        
        # 3. 採樣所有 agent 的訊息
        all_messages = self.field_to_agents(field, positions)
        
        # 4. 選擇當前 agent 的訊息（直接用索引）
        if isinstance(agent_idx, int):
            comm_message = all_messages[:, agent_idx, :]
        else:  # tensor [B]
            comm_message = all_messages[torch.arange(batch, device=field.device), agent_idx, :]
        
        return comm_message, field

class PDECommunication_NoDiffusion(PDECommunication):
    """Ablation: 只有 reaction，沒有 diffusion"""
    def reaction_diffusion_step(self, field):
        # diffusion = 0  # 移除擴散項
        reaction = self.reaction_coef * field * (1 - field.tanh())
        field_next = field + self.dt * reaction
        return field_next.clamp(-10, 10)

class PDECommunication_NoReaction(PDECommunication):
    """Ablation: 只有 diffusion，沒有 reaction"""
    def reaction_diffusion_step(self, field):
        laplacian = F.conv2d(field, self.lap_kernel, padding=1, groups=self.feature_dim)
        diffusion = self.diffusion_coef * laplacian
        # reaction = 0  # 移除反應項
        field_next = field + self.dt * diffusion
        return field_next.clamp(-10, 10)

class PDECommunication_NoPDE(nn.Module):
    """Ablation: 用 MLP 直接聚合，無 PDE"""
    def __init__(self, feature_dim, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, agent_idx, all_features):
        # 簡單平均其他 agents
        comm_message = all_features.mean(dim=1)  # [B, feat]
        return comm_message, None  # 無 field



# ============ 測試程式碼 ============
if __name__ == "__main__":
    print("測試 PDE Communication...")
    
    batch = 4
    n_agents = 2
    feat_dim = 128
    
    pde = PDECommunication(feat_dim, grid_size=8, n_steps=2)
    
    all_feat = torch.randn(batch, n_agents, feat_dim)
    
    # 測試
    msg, field = pde(agent_idx=0, all_features=all_feat)
    
    print(f"✓ 通訊訊息形狀: {msg.shape}")
    print(f"✓ 場形狀: {field.shape}")
    print(f"✓ 擴散係數: {pde.diffusion_coef.item():.4f}")
    print(f"✓ 反應係數: {pde.reaction_coef.item():.4f}")
    
    # 視覺化場能量
    try:
        import matplotlib.pyplot as plt
        energy = (field[0] ** 2).sum(dim=0).detach().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(energy, cmap='hot', origin='lower')
        plt.colorbar(label='Field Energy')
        plt.title('PDE Communication Field')
        
        # 標註 agent 位置
        positions = pde._compute_positions(n_agents)
        for i, (x, y) in enumerate(positions):
            plt.scatter(x, y, c='cyan', s=200, marker='x', linewidths=3)
            plt.text(x+0.3, y+0.3, f'A{i}', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('test_pde_field.png', dpi=100, bbox_inches='tight')
        print("✓ 場已視覺化: test_pde_field.png")
    except ImportError:
        print("(跳過視覺化，未安裝 matplotlib)")