import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
# from data.dataloader import load_surf_data, load_seg_data
from data.dataloader import load_surf_data
from model.net import BiCortexODE 
from util.mesh import compute_dice

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import logging
from torchdiffeq import odeint_adjoint as odeint
from config import load_config

import os
import pandas as pd

from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes

from pytorch3d.loss import mesh_laplacian_smoothing
# from pytorch3d.ops import laplacian_smoothing


# ★★★ 双表面 ODE wrapper（必须加入） ★★★
class ODEfunc(nn.Module):
    def __init__(self, cortexode):
        super().__init__()
        self.cortexode = cortexode

    def forward(self, t, y):
        v_inner, v_outer = y
        dx_inner, dx_outer = self.cortexode(t, v_inner, v_outer)
        return dx_inner, dx_outer


def compute_gradient_magnitude(volume):
    if volume.ndim == 4:
        volume = volume.unsqueeze(1)  # [B, 1, D, H, W]
    elif volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("Input must be [B, D, H, W] or [B, 1, D, H, W]")

    device = volume.device
    dtype = volume.dtype

    # Define 3D Sobel kernel: shape [1, 1, 3, 3, 3]
    sobel_x = torch.tensor([
        [[  # in_channels=1, so one block
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            [[-2, 0, 2],
             [-4, 0, 4],
             [-2, 0, 2]],
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ]]
    ], dtype=dtype, device=device) / 32.0  # Now shape = [1, 1, 3, 3, 3]

    # Now these permutes work (input is 5D)
    sobel_y = sobel_x.permute(0, 1, 2, 4, 3)  # swap H and W → [1,1,3,3,3]
    sobel_z = sobel_x.permute(0, 1, 4, 2, 3)  # move Z to front spatial dim

    grad_x = F.conv3d(volume, sobel_x, padding=1)
    grad_y = F.conv3d(volume, sobel_y, padding=1)
    grad_z = F.conv3d(volume, sobel_z, padding=1)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
    return grad_mag

def compute_gradient_loss(surface_points, vol_grad, cortexode):
    """
    Compute a loss that encourages surface points to lie on high-gradient regions of the MRI.
    
    Args:
        surface_points (torch.Tensor): [B, N, 3], normalized coordinates in [-1, 1]
                                       where:
                                         dim 0 -> x (corresponds to W)
                                         dim 1 -> y (corresponds to H)
                                         dim 2 -> z (corresponds to D)
        vol_grad (torch.Tensor): [B, 1, D, H, W], gradient magnitude volume
        trilinear_sampler (object): must have method trilinear_sample(volume, coords)
                                    where coords is [1, N, 1, 1, 3] in voxel indices [z, y, x]
    
    Returns:
        loss_gradient (torch.Tensor): scalar, = -mean(gradient_at_surface)
    """
    B, _, D, H, W = vol_grad.shape
    assert B == 1, "Currently assumes batch size = 1 (as in your pipeline)"
    
    # Step 1: Convert normalized coords back to voxel indices
    x_w = (surface_points[..., 0] + 1) * (W - 1) / 2   # [B, N]
    y_h = (surface_points[..., 1] + 1) * (H - 1) / 2   # [B, N]
    z_d = (surface_points[..., 2] + 1) * (D - 1) / 2   # [B, N]
    
    # Step 2: Reorder to [z_d, y_h, x_w] because trilinear_sample expects [D, H, W] indexing
    coords_voxel = torch.stack([x_w, y_h, z_d], dim=-1)  # [B, N, 3]
    
    # Step 3: Reshape to [1, N, 1, 1, 3] as required by your trilinear_sample
    coords_input = coords_voxel.view(1, -1, 1, 1, 3)  # [1, N, 1, 1, 3]
    # print("coords_voxel",coords_voxel, flush=True)
    # Step 4: Sample gradient magnitude at surface points
    grad_vals = cortexode.trilinear_sample(vol_grad, coords_input)  # [1, 1, N, 1, 1]
    grad_vals = grad_vals.view(1, -1)  # [1, N]
   
    # print("grad_vals",grad_vals, flush=True)
    
    # Step 5: Loss = negative mean (to maximize gradient)
    loss_gradient = -grad_vals.mean()
    
    return loss_gradient

def compute_signal_band_loss(v_outer, v_inner, volume, cortexode, n_samples=10):
    """
    Compute average FLAIR intensity between two surfaces using trilinear sampling.
    
    Args:
        v_outer: [B, N, 3], normalized coords in [-1, 1] (outer_low)
        v_inner: [B, N, 3], normalized coords in [-1, 1] (inner_low)
        volume: [B, D, H, W], FLAIR image (not gradient!)
        cortexode: model with trilinear_sample method
        n_samples: number of interpolation points between surfaces
    
    Returns:
        loss: scalar = mean FLAIR intensity in the band
    """
    B, N, _ = v_outer.shape
    assert B == 1, "Assumes batch size = 1 as in your pipeline"
    device = v_outer.device

    total_intensity = 0.0
    total_count = 0

    # Interpolate between inner and outer
    for i in range(n_samples):
        t = i / (n_samples - 1) if n_samples > 1 else 0.5  # handle n_samples=1
        interp_points = (1 - t) * v_inner + t * v_outer   # [B, N, 3]

        # --- 以下逻辑完全复用 compute_gradient_loss ---
        _, _, D, H, W = volume.shape
        # Convert normalized coords to voxel indices
        x_w = (interp_points[..., 0] + 1) * (W - 1) / 2   # [B, N]
        y_h = (interp_points[..., 1] + 1) * (H - 1) / 2   # [B, N]
        z_d = (interp_points[..., 2] + 1) * (D - 1) / 2   # [B, N]

        coords_voxel = torch.stack([x_w, y_h, z_d], dim=-1)  # [B, N, 3]
        coords_input = coords_voxel.view(1, -1, 1, 1, 3)     # [1, N, 1, 1, 3]

        # Sample from FLAIR volume (add channel dim)
        # vol_with_ch = volume.unsqueeze(1)  # [B, 1, D, H, W]
        sampled_vals = cortexode.trilinear_sample(volume, coords_input)  # [1, 1, N, 1, 1]
        sampled_vals = sampled_vals.view(B, N)  # [B, N]

        total_intensity += sampled_vals.sum()
        total_count += B * N

    mean_intensity = total_intensity / total_count
    return mean_intensity

def compute_outer_edge_loss(v_outer, v_pial, v_inner, volume, cortexode, n_samples=5):
    """
    Encourage v_outer to be at a FLAIR intensity drop: 
        I(inner_side) > I(outer_side)
    
    Uses interpolation between surfaces instead of fixed eps.
    
    Args:
        v_outer: [B, N, 3] — current estimate of outer_low
        v_pial:  [B, N, 3] — pial surface (outside of v_outer)
        v_inner: [B, N, 3] — inner_low (inside)
        volume:  [B, D, H, W]
        cortexode: has trilinear_sample
        n_samples: >=2, number of points in interpolation (more = finer step)
    
    Returns:
        loss = mean( max(margin - (I_in - I_out), 0) )
    """
    B, N, _ = v_outer.shape
    assert B == 1 and n_samples >= 2

    # Sample just inside v_outer (toward v_inner)
    t_in = 1.0 / (n_samples - 1)  # e.g., if n=5, t=0.25 → 25% from v_outer to v_inner
    p_in = (1 - t_in) * v_outer + t_in * v_inner   # slightly inside

    # Sample just outside v_outer (toward v_pial)
    t_out = 1.0 / (n_samples - 1)
    p_out = (1 - t_out) * v_outer + t_out * v_pial  # slightly outside
    def _sample(points):
        _, _, D, H, W = volume.shape
        x_w = (points[..., 0] + 1) * (W - 1) / 2   # x → width
        y_h = (points[..., 1] + 1) * (H - 1) / 2   # y → height
        z_d = (points[..., 2] + 1) * (D - 1) / 2   # z → depth

        # ⚠️ CRITICAL: [z, y, x] order for trilinear_sample
        coords_voxel = torch.stack([x_w, y_h, z_d], dim=-1)
        coords_input = coords_voxel.view(1, -1, 1, 1, 3)

        sampled = cortexode.trilinear_sample(volume, coords_input)
        return sampled.view(B, N)
    I_in = _sample(p_in)   # intensity just inside
    I_out = _sample(p_out) # intensity just outside

    # margin = 0.1  # adjust based on your FLAIR normalization
    # diff = I_in - I_out   # want this > margin (inner higher than outer → drop-off at v_outer)
    # loss = torch.clamp(margin - diff, min=0.0).mean()
    diff = I_out - I_in  # 我希望这个 首先是正的 然后要尽可能大
    loss = -diff.mean() + 1.0 * torch.relu(-diff).mean()   # simple, effective, no hyperparams
    return loss


def compute_inner_edge_loss(v_inner, v_white, v_outer, volume, cortexode, n_samples=5):
    """
    Encourage v_inner to sit where FLAIR intensity rises from band to white matter:
        I(white_side) > I(band_side)  → diff = I_in - I_out > 0 and as large as possible.
    
    Args:
        v_inner: [B, N, 3] — current estimate of inner_low (normalized [-1,1])
        v_white: [B, N, 3] — white matter surface (inside of v_inner)
        v_outer: [B, N, 3] — outer_low (outside of v_inner)
        volume:  [B, D, H, W] — FLAIR image
        cortexode: model with trilinear_sample expecting [z, y, x] voxel coords
        n_samples: number of interpolation steps (>=2)

    Returns:
        loss: scalar, lower is better
    """
    B, N, _ = v_inner.shape
    assert B == 1 and n_samples >= 2

    t_step = 1.0 / (n_samples - 1)
    p_in  = (1 - t_step) * v_inner + t_step * v_white   # slightly inside (toward white matter)
    p_out = (1 - t_step) * v_inner + t_step * v_outer   # slightly outside (toward band/outer_low)
    def _sample(points):
        _, _, D, H, W = volume.shape
        x_w = (points[..., 0] + 1) * (W - 1) / 2   # x → width
        y_h = (points[..., 1] + 1) * (H - 1) / 2   # y → height
        z_d = (points[..., 2] + 1) * (D - 1) / 2   # z → depth

        # ⚠️ CRITICAL: [z, y, x] order for trilinear_sample
        coords_voxel = torch.stack([x_w, y_h, z_d], dim=-1)
        coords_input = coords_voxel.view(1, -1, 1, 1, 3)

        sampled = cortexode.trilinear_sample(volume, coords_input)
        return sampled.view(B, N)
    I_in  = _sample(p_in)   # intensity just inside (white side) → should be HIGH
    I_out = _sample(p_out)  # intensity just outside (band side) → should be LOW

    diff = I_in - I_out     # we want this > 0 and LARGE

    # Same loss design as outer: maximize positive diff, penalize negative
    loss = -diff.mean() + torch.relu(-diff).mean()

    return loss

def compute_ordering_loss(v_pial, v_white, v_outer, v_inner, epsilon=1e-8):
    """
    Enforce radial ordering: v_pial -> v_outer -> v_inner -> v_white.
    Specifically, ensure that along the initial radial direction (v_pial - v_white),
    the projection of v_outer is >= projection of v_inner.

    Args:
        v_pial:   [1, N, 3] — fixed pial surface
        v_white:  [1, N, 3] — fixed white matter surface
        v_outer:  [1, N, 3] — current outer_low prediction
        v_inner:  [1, N, 3] — current inner_low prediction
        epsilon:  small value to avoid division by zero

    Returns:
        loss: scalar, mean ReLU(inner_proj - outer_proj)
    """
    # All inputs: [1, N, 3]
    assert v_pial.shape == v_white.shape == v_outer.shape == v_inner.shape
    assert v_pial.shape[0] == 1

    # Compute initial radial direction: from white to pial
    radial_dir = v_pial - v_white  # [1, N, 3]
    radial_norm = torch.norm(radial_dir, dim=-1, keepdim=True)  # [1, N, 1]

    # Avoid division by zero in very thin regions
    radial_dir = radial_dir / (radial_norm + epsilon)  # [1, N, 3], unit vectors

    # Use v_white as origin for projection
    # Project each surface onto the radial direction
    def project(surface):
        # surface: [1, N, 3]
        offset = surface - v_white  # [1, N, 3]
        proj = torch.sum(offset * radial_dir, dim=-1)  # [1, N], dot product
        return proj

    r_outer = project(v_outer)  # [1, N]
    r_inner = project(v_inner)  # [1, N]

    # We require: r_outer >= r_inner
    # Violation when: r_inner > r_outer → penalty = r_inner - r_outer
    violation = r_inner - r_outer  # positive means bad

    # ReLU penalty: only penalize when violation > 0
    loss = torch.relu(violation).mean()

    return loss

def compute_Laplacian_smoothness_loss(v_outer, v_inner, faces):
    """
    Compute Laplacian smoothing loss using PyTorch3D's official loss function.
    
    Args:
        v_outer: [1, N, 3]
        v_inner: [1, N, 3]
        faces:   [F, 3] or [1, F, 3] (long tensor)
    
    Returns:
        loss: scalar
    """
    # Ensure faces has batch dimension if needed
    if faces.dim() == 2:
        faces = faces.unsqueeze(0)  # [1, F, 3]

    # Create Meshes objects
    mesh_outer = Meshes(verts=v_outer, faces=faces)
    mesh_inner = Meshes(verts=v_inner, faces=faces)

    # Compute loss (method="uniform" is default and recommended for cortical surfaces)
    loss_outer = mesh_laplacian_smoothing(mesh_outer, method="uniform")
    loss_inner = mesh_laplacian_smoothing(mesh_inner, method="uniform")

    return loss_outer + loss_inner


def train_surf(config):
    """
    Training CortexODE for cortical surface reconstruction
    using adjoint sensitivity method proposed in neural ODE
    
    For original neural ODE paper please see:
    - Chen et al. Neural ordinary differential equations. NeurIPS, 2018.
      Paper: https://arxiv.org/abs/1806.07366v5
      Code: https://github.com/rtqichen/torchdiffeq
    
    Note: using seminorm in adjoint method can accelerate the training, but it
    will cause exploding gradients for explicit methods in our experiments.

    For seminorm please see:
    - Patrick et al. Hey, that's not an ODE: Faster ODE Adjoints via Seminorms. ICML, 2021.
      Paper: https://arxiv.org/abs/2009.09457
      Code: https://github.com/patrick-kidger/FasterNeuralDiffEq

    Configurations (see config.py):
    model_dir: directory to save your checkpoints
    data_name: [5T, hcp, adni, ...]
    surf_type: [outer, wm, gm]
    surf_hemi: [lh, rh]
    """
    
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag
    
    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    
    # create log file
    logging.basicConfig(filename=model_dir+'/model_'+surf_type+'_'+data_name+'_'+surf_hemi+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')


    # --------------------------
    # read CSV to get train/valid subjects
    # --------------------------
    split_csv = os.path.join(config.datasplit_csv)
    split_df = pd.read_csv(split_csv)
    
    train_subjects = split_df[split_df["split"] == "train"]["subject_id"].tolist()
    valid_subjects = split_df[split_df["split"] == "valid"]["subject_id"].tolist()
    # # 测试代码有效性
    # valid_subjects = train_subjects

    logging.info(f"{len(train_subjects)} train subjects, {len(valid_subjects)} valid subjects")

    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = load_surf_data(config, subject_list=train_subjects)
    validset = load_surf_data(config, subject_list=valid_subjects)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initalize model ...")
    cortexode = BiCortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
    # 使用Stage-2开启这个
    # cortexode_exchange = BiCortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)

    optimizer = optim.Adam(cortexode.parameters(), lr=lr)
    # 使用Stage-2开启这个
    # optimizer = optim.Adam(list(cortexode.parameters()) + list(cortexode_exchange.parameters()), lr=lr)

    # <<< 添加学习率调度器 >>>
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # val_error 越小越好
        factor=0.5,        # 每次降为原来的一半
        patience=5,       # 连续 20 个 epoch 没下降就降 lr
        min_lr=1e-6,       # 最小学习率
        verbose=True       # 打印信息
    )

    # T = torch.Tensor([0,1]).to(device)    # integration time interval for ODE
    T = torch.linspace(0.0, 1.0, steps=11).to(device)  # 因为 1.0 / 0.1 = 10 步 → 11 个点


    # <<< 加载预训练模型权重 >>>
    pretrained_path = config.pretrained_path
    if pretrained_path is not None and pretrained_path != '':
        try:
            cortexode.load_state_dict(torch.load(pretrained_path, map_location=device))
            logging.info(f"✅ Successfully loaded pretrained model from {pretrained_path}")
        except Exception as e:
            logging.warning(f"❌ Failed to load pretrained model: {e}")
    # --------------------------
    # training
    # --------------------------
    logging.info("start training ...")
    for epoch in tqdm(range(n_epochs+1)):
        avg_loss = []

        avg_total_loss = []
        avg_mse_loss = []
        avg_depth_loss = []
        avg_traj_loss = []
        avg_gradient_loss = []
        avg_band_loss = []
        avg_edge_loss = []
        avg_order_loss = []
        avg_Laplacian_smooth_loss = []
        avg_thickness_loss = []

        avg_mse_inner_loss = []
        avg_mse_outer_loss = []
        avg_mse_i2o_loss = []   # inner-to-outer
        avg_mse_o2i_loss = []   # outer-to-inner

        for idx, data in enumerate(trainloader):
            volume_in, v_in_inner, v_in_outer, v_gt_inner, v_gt_outer, faces = data

            optimizer.zero_grad()
           
            # Move to device
            volume_in = volume_in.to(device).float()
            with torch.no_grad():
                vol_grad = compute_gradient_magnitude(volume_in)  # [B, 1, D, H, W]
            v_in_inner = v_in_inner.to(device)
            v_in_outer = v_in_outer.to(device)
            v_gt_inner = v_gt_inner.to(device)
            v_gt_outer = v_gt_outer.to(device)
            faces = faces.to(device)
            

            # 初始化模型数据
            cortexode.set_data(v_in_inner, v_in_outer, volume_in) # MLP

            # Use initial surfaces as reference for laminar depth
            v_pial_ref = v_in_outer.detach()   # [1, N, 3]
            v_white_ref = v_in_inner.detach()

            # ---------------------------- 1. 皮层内外表面 向皮层中间预测 皮层中间的低信号层内外表面 ----------------------------
            # Create ODE function with depth reference
            ode_func = ODEfunc(cortexode)
            # ★ odeint input必须是tuple
            y0 = (v_in_inner, v_in_outer)
            # ★ solve ODE
            # 调用 odeint，但不要加 [-1]，也不要立即解包
            full_solution = odeint(
                ode_func, 
                y0, 
                T, 
                method=solver, 
                options=dict(step_size=step_size)
            )        
            v_out_inner = full_solution[0][-1]
            v_out_outer = full_solution[1][-1]

            # ---------------------------- 2. 皮层内外表面互相交换 进一步增强他们的联系  ----------------------------
            USE_EXCHANGE_MODULE = False  # ← 控制开关

            if USE_EXCHANGE_MODULE:
                print("true ex", flush=True)
                # 初始化模型数据
                cortexode_exchange.set_data(v_out_inner, v_out_outer, volume_in)
                ode_func_exchange = ODEfunc(cortexode_exchange)
                y0 = (v_out_inner, v_out_outer)
                full_solution_exchange = odeint(
                    ode_func_exchange, y0, T,
                    method=solver, options=dict(step_size=step_size)
                )
                v_out_inner_to_outer = full_solution_exchange[0][-1]
                v_out_outer_to_inner = full_solution_exchange[1][-1]
                # ================================ 交换相关权重 可以调参================================
                exchange_loss_weight = 1
            else:
                # 占位符，避免变量未定义
                v_out_inner_to_outer = v_out_inner
                v_out_outer_to_inner = v_out_outer
                exchange_loss_weight = 0.0
                
            # ----------------------------
            # 1. MSE Loss (主任务)
            # ----------------------------
            loss_mse = (
                nn.MSELoss()(v_out_inner, v_gt_inner) +
                nn.MSELoss()(v_out_outer, v_gt_outer) +
                nn.MSELoss()(v_out_inner_to_outer, v_gt_outer) * exchange_loss_weight +
                nn.MSELoss()(v_out_outer_to_inner, v_gt_inner) * exchange_loss_weight
            )
            # ----------------------------
            # 2. Depth Consistency Loss (深度先验)
            # ----------------------------
            # 计算 PREDICTED 中间层在 white-pial 柱上的 depth
            depth_pred_inner = cortexode.compute_normalized_depth(v_out_inner, v_white_ref, v_pial_ref)
            depth_pred_outer = cortexode.compute_normalized_depth(v_out_outer, v_white_ref, v_pial_ref)

            # 计算 GT 中间层在 SAME white-pial 柱上的 depth（这才是目标！）
            depth_gt_inner = cortexode.compute_normalized_depth(v_gt_inner, v_white_ref, v_pial_ref)
            depth_gt_outer = cortexode.compute_normalized_depth(v_gt_outer, v_white_ref, v_pial_ref)

            # 计算第二次ODE后的中间层的depth 
            depth_pred_inner_to_outer = cortexode.compute_normalized_depth(v_out_inner_to_outer, v_white_ref, v_pial_ref)
            depth_pred_outer_to_inner = cortexode.compute_normalized_depth(v_out_outer_to_inner, v_white_ref, v_pial_ref)
            # Depth loss：让预测 depth ≈ GT depth（在同一参考系下）
            loss_depth = (
                nn.MSELoss()(depth_pred_inner, depth_gt_inner) +
                nn.MSELoss()(depth_pred_outer, depth_gt_outer) +
                nn.MSELoss()(depth_pred_inner_to_outer, depth_gt_outer) * exchange_loss_weight +
                nn.MSELoss()(depth_pred_outer_to_inner, depth_gt_inner) * exchange_loss_weight
            ) 
            # ----------------------------
            # 3. 第二次形变的路径对称
            # ----------------------------
            if USE_EXCHANGE_MODULE:
                Trajectory_outer_to_inner = torch.norm(v_out_inner_to_outer - v_out_inner, dim=-1)
                Trajectory_inner_to_outer = torch.norm(v_out_outer_to_inner - v_out_outer, dim=-1)
                loss_Trajectory_consistency = nn.MSELoss()(Trajectory_outer_to_inner, Trajectory_inner_to_outer)
            else:
                loss_Trajectory_consistency = torch.tensor(0.0, device=v_out_inner.device)
            # # ----------------------------
            # # 4. Image band Loss
            # # ----------------------------
            # 第一步之间的 低信号带
            loss_low_band = compute_signal_band_loss(
                v_out_outer, v_out_inner, volume_in, cortexode, n_samples=10
            )
            # 第一步之间的 高信号带
            loss_high_band = -1 * compute_signal_band_loss(
                v_in_outer, v_out_outer, volume_in, cortexode, n_samples=10
            )
            # 第二步预测后的低信号带
            loss_low_band_exc = compute_signal_band_loss(
                v_out_inner_to_outer, v_out_outer_to_inner, volume_in, cortexode, n_samples=10
            ) 
            # 第二步预测后的高信号带
            loss_high_band_exc = -1 * compute_signal_band_loss(
                v_in_outer, v_out_inner_to_outer, volume_in, cortexode, n_samples=10
            ) 
            # 主低信号带：原始 + 交换预测
            loss_low_total = loss_low_band + loss_low_band_exc * exchange_loss_weight 
            # 外侧高信号带：原始 + 交换预测
            loss_high_total = loss_high_band + loss_high_band_exc * exchange_loss_weight 
            # 总 band loss
            loss_band = loss_low_total + loss_high_total

            # # ----------------------------
            # # 5. Edge Loss
            # # ----------------------------
            # 1. Edge alignment loss for outer boundary (outer_low)
            loss_edge_outer = compute_outer_edge_loss(
                v_outer=v_out_outer,
                v_pial=v_in_outer,
                v_inner=v_out_inner,
                volume=volume_in,
                cortexode=cortexode,
                n_samples=10
            )
            # 2. Edge alignment loss for inner boundary (inner_low)
            loss_edge_inner = compute_inner_edge_loss(
                v_inner=v_out_inner,
                v_white=v_in_inner,
                v_outer=v_out_outer,
                volume=volume_in,
                cortexode=cortexode,
                n_samples=10
            )
            # 3. Edge alignment loss for outer boundary (outer_low) 从inner变成outer的那个表面
            loss_edge_outer_exc = compute_outer_edge_loss(
                v_outer=v_out_inner_to_outer,
                v_pial=v_in_outer,
                v_inner=v_out_outer_to_inner,
                volume=volume_in,
                cortexode=cortexode,
                n_samples=10
            )
            # 4. Edge alignment loss for inner boundary (inner_low)) 从outer变成inner的那个表面
            loss_edge_inner_exc = compute_inner_edge_loss(
                v_inner=v_out_outer_to_inner,
                v_white=v_in_inner,
                v_outer=v_out_inner_to_outer,
                volume=volume_in,
                cortexode=cortexode,
                n_samples=10
            )
            loss_edge_gradent = (loss_edge_outer + loss_edge_inner) + (loss_edge_outer_exc + loss_edge_inner_exc) * exchange_loss_weight
            # # ----------------------------
            # # 6. order Loss
            # # ----------------------------
            loss_order = compute_ordering_loss(v_pial=v_in_outer, v_white=v_in_inner, v_outer=v_out_outer, v_inner=v_out_inner)
            
            # # ----------------------------
            # # 7. Laplacian
            # # ----------------------------
            loss_Laplacian_smooth = compute_Laplacian_smoothness_loss(v_out_outer, v_out_inner, faces)

            # # ----------------------------
            # Total Loss
            # ----------------------------
            # ================================  可以调参的地方 ================================
            loss_mse_weight = 1e3
            loss_depth_weight = 0.01 * 0 # 取消 depth  模块 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss_Trajectory_weight = 100 * 0 # 取消 交换路径  模块 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss_band_weight = 0.01 * 0 # 取消 band  模块 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss_edge_weight = 0.01 * 0 # 取消 edge  模块 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss_order_weight = 100 
            loss_Laplacian_smooth_weight = 1 


            loss = loss_mse_weight * loss_mse + loss_depth_weight * loss_depth + loss_Trajectory_weight * loss_Trajectory_consistency + \
                    loss_band_weight * loss_band + loss_edge_weight * loss_edge_gradent + loss_order_weight * loss_order + \
                    loss_Laplacian_smooth_weight * loss_Laplacian_smooth
            
            # ✅ 分别记录各项 loss（用于 epoch 平均）
            avg_loss.append(loss.item())
            avg_mse_loss.append(loss_mse_weight * loss_mse.item())
            avg_depth_loss.append(loss_depth_weight * loss_depth.item())  # 这里用 smooth_loss 变量名暂存 depth loss
            avg_traj_loss.append(loss_Trajectory_weight * loss_Trajectory_consistency.item())
            avg_band_loss.append(loss_band_weight * loss_band.item())
            avg_edge_loss.append(loss_edge_weight * loss_edge_gradent.item())
            avg_order_loss.append(loss_order_weight * loss_order.item())
            avg_Laplacian_smooth_loss.append(loss_Laplacian_smooth_weight * loss_Laplacian_smooth.item())

            loss.backward()
            optimizer.step()

        # logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))

        logging.info('epoch:{}, total_loss:{:.6f}, mse_loss:{:.6f}, depth_loss:{:.6f}, traj_loss:{:.6f}, band_loss:{:.6f}, edge_loss:{:.6f}, order_loss:{:.6f}, laplacian_smooth_loss:{:.6f}'.format(
            epoch,
            np.mean(avg_loss),
            np.mean(avg_mse_loss),
            np.mean(avg_depth_loss),
            np.mean(avg_traj_loss),
            np.mean(avg_band_loss),
            np.mean(avg_edge_loss),
            np.mean(avg_order_loss),
            np.mean(avg_Laplacian_smooth_loss)
        ))

        if epoch % 20 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error = []
                for idx, data in enumerate(validloader):
                    # ★ validloader 也需要: 双surface
                    volume_in, v_in_inner, v_in_outer, v_gt_inner, v_gt_outer, faces = data

                    volume_in   = volume_in.to(device).float()
                    v_in_inner  = v_in_inner.to(device)
                    v_in_outer  = v_in_outer.to(device)
                    v_gt_inner  = v_gt_inner.to(device)
                    v_gt_outer  = v_gt_outer.to(device)
                    faces       = faces.to(device)

                    # cortexode.set_data(v_in_inner, v_in_outer, volume_in, faces=faces[0])
                    cortexode.set_data(v_in_inner, v_in_outer, volume_in) # 单纯 mlp
                    # depth encoder
                    v_pial_ref = v_in_outer.detach()
                    v_white_ref = v_in_inner.detach()
                    ode_func = ODEfunc(cortexode)
                    y0 = (v_in_inner, v_in_outer)
                    full_solution = odeint(
                        ode_func, y0, T,
                        method=solver,
                        options=dict(step_size=step_size)
                    )
                    v_out_inner = full_solution[0][-1]
                    v_out_outer = full_solution[1][-1]
                    loss_mse = (
                        nn.MSELoss()(v_out_inner, v_gt_inner) +
                        nn.MSELoss()(v_out_outer, v_gt_outer) 
                    )  

                    valid_error.append(loss_mse_weight * loss_mse.item())

                logging.info('epoch:{}, validation error:{}'.format(epoch, np.mean(valid_error)))
                logging.info('-------------------------------------')
                # <<< 更新学习率调度器 >>>
                scheduler.step(np.mean(valid_error))
                # ✅ 打印当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                logging.info('epoch:{}, current learning rate: {:.2e}'.format(epoch, current_lr))
                logging.info('-------------------------------------')
        # save model checkpoints 
        if epoch % 10 == 0:
            torch.save(cortexode.state_dict(), model_dir+'/model_'+surf_type+'_'+\
                       data_name+'_'+surf_hemi+'_'+tag+'_'+str(epoch)+'epochs.pt')
            # 保存stage-2的权重
            # torch.save(cortexode_exchange.state_dict(), model_dir+'/model_ex_'+surf_type+'_'+\
            #            data_name+'_'+surf_hemi+'_'+tag+'_'+str(epoch)+'epochs.pt')

    # save the final model
    torch.save(cortexode.state_dict(), model_dir+'/model_'+surf_type+'_'+\
               data_name+'_'+surf_hemi+'_'+tag+'.pt')
    # 保存stage-2的权重
    # torch.save(cortexode_exchange.state_dict(), model_dir+'/model_ex_'+surf_type+'_'+\
    #            data_name+'_'+surf_hemi+'_'+tag+'.pt')


if __name__ == '__main__':
    config = load_config()
    train_surf(config)
 