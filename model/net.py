import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiCortexODE(nn.Module):
    """
    Bi-directional CortexODE with MLP-based cross-surface fusion.
    """

    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3):
        super(BiCortexODE, self).__init__()

        C = dim_h
        K = kernel_size
        Q = n_scale

        self.C = C
        self.K = K
        self.Q = Q

        # FC for point-wise features
        self.fc1 = nn.Linear(dim_in, C) # +1 for depth encoding

        # Fusion MLP before output
        self.fc2 = nn.Linear(C * 2, C * 4)
        self.fc3 = nn.Linear(C, C * 2) #(C * 4 配合 attention C*1 是原来MLP
        self.fc4 = nn.Linear(C * 2, dim_in)

        # local convolution (multi-scale)
        self.localconv = nn.Conv3d(Q, C, (K, K, K))
        self.localfc   = nn.Linear(C, C)

        # ----- MLP for cross-surface fusion -----
        self.cross_fusion = nn.Sequential(
            nn.Linear(C * 8, C * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(C * 2, C)
        )
    
        # cube sampling init
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1, 3)
        self.cubes   = torch.zeros([1, Q, K, K, K])


    def _initialize(self, V):
        self.x_shift = self.x_shift.to(V.device)
        self.cubes   = self.cubes.to(V.device)
        self.initialized = True


    def set_data(self, x_inner, x_outer, V):
        """
        Set MRI and cube sampling structures.
        Called once per batch.
        """
        if not self.initialized:
            self._initialize(V)

        # Volume shape
        D1, D2, D3 = V[0,0].shape
        D = max([D1, D2, D3])
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        # store point counts
        self.m_inner = x_inner.shape[1]
        self.m_outer = x_outer.shape[1]

        # cube sampling buffers
        self.neighbors_inner = self.cubes.repeat(self.m_inner, 1, 1, 1, 1)
        self.neighbors_outer = self.cubes.repeat(self.m_outer, 1, 1, 1, 1)

        # multi-scale MRI
        self.Vq = [V]
        for _ in range(1, self.Q):
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))
    
    def forward(self, t, x_inner, x_outer):
        """
        前向传播函数：计算内外两组点在时间 t 的速度场（即 dx/dt）。
        
        输入:
            t (float): 当前时间（ODE 求解器传入，本实现中未显式使用）
            x_inner (Tensor): 内表面点集，形状为 [B, N_inner, 3]
            x_outer (Tensor): 外表面点集，形状为 [B, N_outer, 3]
        
        输出:
            dx_inner (Tensor): 内表面点的速度更新，形状 [B, N_inner, 3]
            dx_outer (Tensor): 外表面点的速度更新，形状 [B, N_outer, 3]
        """

        # ----------------------------------------------------------------------
        # 1. 提取局部几何特征（通过体素/立方体采样 + 卷积）
        # ----------------------------------------------------------------------
        # 对内/外点分别在其局部邻域内采样小立方体，并通过共享的 localconv 提取局部体积特征
        vol_feat_inner = self.localconv(self.cube_sampling_inner(x_inner))  # [B, C, m_inner]
        vol_feat_outer = self.localconv(self.cube_sampling_outer(x_outer))  # [B, C, m_outer]

        # 将卷积输出展平并送入全连接层，统一到点级别特征（每个点对应一个局部特征向量）
        z_local_inner = self.localfc(vol_feat_inner.view(-1, self.m_inner, self.C))  # [B, N_inner, C]
        z_local_outer = self.localfc(vol_feat_outer.view(-1, self.m_outer, self.C))  # [B, N_outer, C]

        # ----------------------------------------------------------------------
        # 2. 提取点本身的全局坐标特征
        # ----------------------------------------------------------------------
        # 使用共享的 fc1 对原始坐标进行非线性编码（捕捉全局位置信息）
        z_point_inner = F.leaky_relu(self.fc1(x_inner), negative_slope=0.2)  # [B, N_inner, C]
        z_point_outer = F.leaky_relu(self.fc1(x_outer), negative_slope=0.2)  # [B, N_outer, C]

        # ----------------------------------------------------------------------
        # 3. 融合局部与全局特征（逐点拼接）
        # ----------------------------------------------------------------------
        f_inner = torch.cat([z_point_inner, z_local_inner], dim=2)  # [B, N_inner, 2C]
        f_outer = torch.cat([z_point_outer, z_local_outer], dim=2)  # [B, N_outer, 2C]

        # ----------------------------------------------------------------------
        # 4. 特征增强：通过共享 MLP（fc2）进一步提炼融合后的特征
        # ----------------------------------------------------------------------
        f_inner = F.leaky_relu(self.fc2(f_inner), negative_slope=0.2)
        f_outer = F.leaky_relu(self.fc2(f_outer), negative_slope=0.2)

        # ----------------------------------------------------------------------
        # 5. 跨表面交互：内/外表面特征相互融合
        #    注意：顺序不同表示“以谁为主”进行融合
        # ----------------------------------------------------------------------
        # fuse_inner: 以内表面特征为主，融合外表面信息
        fuse_inner = self.cross_fusion(torch.cat([f_inner, f_outer], dim=2))  # [B, N_inner, C']
        # fuse_outer: 以外表面特征为主，融合内表面信息
        fuse_outer = self.cross_fusion(torch.cat([f_outer, f_inner], dim=2))  # [B, N_outer, C']

        # ----------------------------------------------------------------------
        # 6. 解码为速度场（dx/dt）
        # ----------------------------------------------------------------------
        # 进一步非线性变换
        fuse_inner = F.leaky_relu(self.fc3(fuse_inner), negative_slope=0.2)
        fuse_outer = F.leaky_relu(self.fc3(fuse_outer), negative_slope=0.2)

        # 最终输出位移导数（速度）
        dx_inner = self.fc4(fuse_inner)  # [B, N_inner, 3]
        dx_outer = self.fc4(fuse_outer)  # [B, N_outer, 3]

        return dx_inner, dx_outer
    
    def compute_normalized_depth(self, x, v_white, v_pial):
        """
        Compute normalized cortical depth along the white-pial column.
        x, v_white, v_pial: [B, N, 3]
        Returns: [B, N, 1]
        """
        vec_wp = v_pial - v_white          # [B, N, 3]
        vec_xw = x - v_white               # [B, N, 3]
        proj = (vec_xw * vec_wp).sum(dim=-1, keepdim=True)      # [B, N, 1]
        norm_sq = (vec_wp * vec_wp).sum(dim=-1, keepdim=True) + 1e-8
        depth = proj / norm_sq
        return depth.clamp(0.0, 1.0)

    def trilinear_sample(self, volume, coords):
        """
        volume: [1, 1, D, H, W]
        coords: [1, N, 1, 1, 3]  (voxel coordinates, not normalized)
        return: [1, 1, N, 1, 1]
        """
        B, C, D, H, W = volume.shape
        _, N, _, _, _ = coords.shape

        # ---- flatten coords ----
        coords = coords.view(1, N, 3)   # [1,N,3]

        # coords[...,0], coords[...,1], coords[...,2] are already voxel x,y,z
        # x = coords[..., 0]
        # y = coords[..., 1]
        # z = coords[..., 2]
        # 这才是对的！！
        z = coords[..., 0]
        y = coords[..., 1]
        x = coords[..., 2]

        # floor & ceil
        x0 = x.floor().long().clamp(0, W - 1)
        x1 = (x0 + 1).clamp(0, W - 1)

        y0 = y.floor().long().clamp(0, H - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        z0 = z.floor().long().clamp(0, D - 1)
        z1 = (z0 + 1).clamp(0, D - 1)

        # weights
        wx = x - x0.float()
        wy = y - y0.float()
        wz = z - z0.float()

        vol = volume[0, 0]  # [D, H, W]

        # 8 corners
        c000 = vol[z0, y0, x0]
        c001 = vol[z0, y0, x1]
        c010 = vol[z0, y1, x0]
        c011 = vol[z0, y1, x1]
        c100 = vol[z1, y0, x0]
        c101 = vol[z1, y0, x1]
        c110 = vol[z1, y1, x0]
        c111 = vol[z1, y1, x1]

        # trilinear interpolation
        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        c = c0 * (1 - wz) + c1 * wz  # [N]

        return c.view(1, 1, N, 1, 1)

    def cube_sampling_inner(self, x):
        # x: [1, N, 3], normalized coordinates in [-1, 1]
        with torch.no_grad():
            B, N, _ = x.shape
            # 获取原始 volume 的空间尺寸（第0层未下采样）
            D_orig, H_orig, W_orig = self.Vq[0].shape[2:]  # 注意：Vq[q] 形状为 [1, C, D, H, W]
            # Step 1: 将归一化坐标 x ∈ [-1,1] 转换为体素坐标（连续，非整数）
            # 注意：grid_sample 的 align_corners=True 对应:
            #   voxel index i ∈ [0, S-1] ↔ normalized coord = -1 + 2*i/(S-1)
            # 所以逆变换为: i = (x_norm + 1) * (S - 1) / 2
            x_vox = torch.stack([
                (x[..., 0] + 1) * (W_orig - 1) / 2,   # x -> width index
                (x[..., 1] + 1) * (H_orig - 1) / 2,   # y -> height index
                (x[..., 2] + 1) * (D_orig - 1) / 2    # z -> depth index
            ], dim=-1)  # [1, N, 3]

            for q in range(self.Q):
                scale = 2 ** q
                # Step 2: 在体素空间中添加局部偏移（单位：voxel）
                # self.x_shift: [K^3, 3]，已经是 voxel 单位
                xq_vox = x_vox.unsqueeze(-2) + self.x_shift.to(x.device) * scale  # [1, N, K^3, 3]
                # Step 3: reshape 为适合 trilinear_sample 的格式
                xq_flat = xq_vox.view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)  # [1, N*K^3, 1, 1, 3]
                # Step 4: 采样 —— 注意：trilinear_sample 必须接收体素坐标！
                vq = self.trilinear_sample(self.Vq[q], xq_flat)  # [1, 1, N*K^3, 1, 1]
                # Step 5: reshape 回邻居立方体
                self.neighbors_inner[:, q] = vq[0, 0].view(N, self.K, self.K, self.K)

        return self.neighbors_inner.clone()

    def cube_sampling_outer(self, x):
        # x: [1, N, 3], normalized coordinates in [-1, 1]
        with torch.no_grad():
            B, N, _ = x.shape
            # 获取原始 volume 的空间尺寸（第0层未下采样）
            D_orig, H_orig, W_orig = self.Vq[0].shape[2:]  # 注意：Vq[q] 形状为 [1, C, D, H, W]
            # Step 1: 将归一化坐标 x ∈ [-1,1] 转换为体素坐标（连续，非整数）
            # 注意：grid_sample 的 align_corners=True 对应:
            #   voxel index i ∈ [0, S-1] ↔ normalized coord = -1 + 2*i/(S-1)
            # 所以逆变换为: i = (x_norm + 1) * (S - 1) / 2
            x_vox = torch.stack([
                (x[..., 0] + 1) * (W_orig - 1) / 2,   # x -> width index
                (x[..., 1] + 1) * (H_orig - 1) / 2,   # y -> height index
                (x[..., 2] + 1) * (D_orig - 1) / 2    # z -> depth index
            ], dim=-1)  # [1, N, 3]

            for q in range(self.Q):
                scale = 2 ** q
                # Step 2: 在体素空间中添加局部偏移（单位：voxel）
                # self.x_shift: [K^3, 3]，已经是 voxel 单位
                xq_vox = x_vox.unsqueeze(-2) + self.x_shift.to(x.device) * scale  # [1, N, K^3, 3]
                # Step 3: reshape 为适合 trilinear_sample 的格式
                xq_flat = xq_vox.view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)  # [1, N*K^3, 1, 1, 3]
                # Step 4: 采样 —— 注意：trilinear_sample 必须接收体素坐标！
                vq = self.trilinear_sample(self.Vq[q], xq_flat)  # [1, 1, N*K^3, 1, 1]
                # Step 5: reshape 回邻居立方体
                self.neighbors_outer[:, q] = vq[0, 0].view(N, self.K, self.K, self.K)

        return self.neighbors_outer.clone()