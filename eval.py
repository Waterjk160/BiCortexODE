import os
import nibabel as nib
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from data.preprocess import process_volume, process_surface, process_surface_inverse
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance, check_self_intersect
from util.tca import topology
from model.net import BiCortexODE
from config import load_config

import logging
import os
import pandas as pd
# initialize topology correction
topo_correct = topology()


def seg2surf(seg,
             data_name='hcp',
             sigma=0.5,
             alpha=16,
             level=0.8,
             n_smooth=2):
    """
    Extract the surface based on the segmentation.
    
    seg: input segmentation
    sigma: standard deviation of guassian blurring
    alpha: threshold for obtaining boundary of topology correction
    level: extracted surface level for Marching Cubes
    n_smooth: iteration of Laplacian smoothing
    """
    
    # ------ connected components checking ------ 
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i)\
                                    for i in range(1, nc+1)]))
    seg = (cc==cc_id).astype(np.float64)

    # ------ generate signed distance function ------ 
    sdf = -cdt(seg) + cdt(1-seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

     # ------ topology correction ------
    sdf_topo= topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:,[2,1,0]].copy()
    f_mc = f_mc.copy()
    D1,D2,D3 = sdf_topo.shape
    D = max(D1,D2,D3)
    v_mc = (2*v_mc - [D3, D2, D1]) / D   # rescale to [-1,1]
    
    # ------ bias correction ------
    # Note that this bias is introduced by FreeSurfer.
    # FreeSurfer changed the size of the input MRI, 
    # but the affine matrix of the MRI was not changed.
    # So this bias is caused by the different between 
    # the original and new affine matrix.
    if data_name == 'hcp':
        v_mc = v_mc + [0.0090, 0.0058, 0.0088]
    elif data_name == 'adni':
        v_mc = v_mc + [0.0090, 0.0000, 0.0095]
        
    # ------ mesh smoothing ------
    v_mc = torch.Tensor(v_mc).unsqueeze(0).to(device)
    f_mc = torch.LongTensor(f_mc).unsqueeze(0).to(device)
    for j in range(n_smooth):    # smooth and inflate the mesh
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    f_mc = f_mc[0].cpu().numpy()
    
    return v_mc, f_mc

# ★★★ 双表面 ODE wrapper（必须加入） ★★★
class ODEfunc(nn.Module):
    def __init__(self, cortexode):
        super().__init__()
        self.cortexode = cortexode

    def forward(self, t, y):
        v_inner, v_outer = y
        dx_inner, dx_outer = self.cortexode(t, v_inner, v_outer)
        return dx_inner, dx_outer



if __name__ == '__main__':
    
    # ------ load configuration ------
    config = load_config()
    test_type = config.test_type  # initial surface / prediction / evaluation
    data_dir = config.data_dir  # directory of datasets
    model_dir = config.model_dir  # directory of pretrained models
    init_dir = config.init_dir  # directory for saving the initial surfaces
    result_dir = config.result_dir  # directory for saving the predicted surfaces
    data_name = config.data_name  # hcp, adni, dhcp, 5t
    surf_hemi = config.surf_hemi  # lh, rh
    device = config.device
    tag = config.tag  # identity of the experiment

    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    n_inflate = config.n_inflate  # inflation iterations
    rho = config.rho # inflation scale

    # --------------------------
    # read CSV to get train/valid subjects
    # --------------------------
    split_csv = os.path.join(config.datasplit_csv)
    split_df = pd.read_csv(split_csv)
    
 
    # 这里改成了train 用于测试
    test_subjects = split_df[split_df["split"] == "test"]["subject_id"].tolist()
    print(f"{len(test_subjects)} test subjects", flush=True)


    if test_type == 'pred' or test_type == 'eval':
        # T = torch.Tensor([0,1]).to(device) # cortexode 原来自己的
        T = torch.linspace(0.0, 1.0, steps=11).to(device)  # 因为 1.0 / 0.1 = 10 步 → 11 个点
        cortexode = BiCortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
        cortexode.load_state_dict(torch.load(model_dir+'model_both_'+data_name+'_'+surf_hemi+'_'+tag+'.pt', map_location=device))
        cortexode.eval()
        # ★ 使用 ODE wrapper
        ode_func = ODEfunc(cortexode)
    # ------ start testing ------
    # 这部分需要改一下，改成跟train里一样的
    # subject_list = sorted(os.listdir(data_dir))
    subject_list = test_subjects

    if test_type == 'eval':
        assd_inner_all = []
        assd_outer_all = []
        hd_inner_all = []
        hd_outer_all = []
        sif_inner_all = []
        sif_outer_all = []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]

        # ------- load brain MRI ------- 
        if data_name == "5T":
            mri_path = os.path.join(data_dir, subid, "image", "brain_high.nii.gz")
            brain = nib.load(mri_path)
            brain_arr = brain.get_fdata()
        brain_arr = process_volume(brain_arr, data_name)
        volume_in = torch.Tensor(brain_arr).unsqueeze(0).to(device)

        # ------- input initial surface ------- 
        # ------- 加载白质表面 ------- 
        white_path = os.path.join(
            data_dir, subid, "surf", f"{surf_hemi}.white"
        )

        if not os.path.exists(white_path):
            print(f"[Warning] Missing white surface: {white_path}")
            # skipped += 1
            continue

        v_in_inner, faces = nib.freesurfer.io.read_geometry(white_path)
        if v_in_inner.size == 0 or faces.size == 0:
            print(f"[Warning] Empty geometry in white surface: {white_path}")
            # skipped += 1
            continue
        faces = faces.astype(np.int64).copy()
        v_tmp = np.ones([v_in_inner.shape[0], 4])
        v_tmp[:, :3] = v_in_inner
        v_in_inner = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]
        volume_shape = brain_arr.shape[-3:]  # 取 (D, H, W)，忽略通道维度（如果有）
        v_in_inner, faces = process_surface(v_in_inner, faces, volume_shape, data_name)

        # ------- 加载灰质表面 ------- 
        pial_path = os.path.join(
            data_dir, subid, "surf", f"{surf_hemi}.pial"
        )

        if not os.path.exists(pial_path):
            print(f"[Warning] Missing pial surface: {pial_path}")
            # skipped += 1
            continue

        v_in_outer, faces = nib.freesurfer.io.read_geometry(pial_path)
        if v_in_outer.size == 0 or faces.size == 0:
            print(f"[Warning] Empty geometry in pial surface: {pial_path}")
            # skipped += 1
            continue
        faces = faces.astype(np.int64).copy()
        v_tmp = np.ones([v_in_outer.shape[0], 4])
        v_tmp[:, :3] = v_in_outer
        v_in_outer = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]
        volume_shape = brain_arr.shape[-3:]  # 取 (D, H, W)，忽略通道维度（如果有）
        v_in_outer, faces = process_surface(v_in_outer, faces, volume_shape, data_name)
        
        # ------- predict cortical surfaces ------- 
        # 直接预测表面
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                # 原来的
                v_in_inner = torch.Tensor(v_in_inner).unsqueeze(0).to(device)
                v_in_outer = torch.Tensor(v_in_outer).unsqueeze(0).to(device)
                faces = torch.LongTensor(faces).unsqueeze(0).to(device)
                
                # 初始化模型数据
                # cortexode.set_data(v_in_inner, v_in_outer, volume_in, faces=faces[0])
                cortexode.set_data(v_in_inner, v_in_outer, volume_in) # MLP
                # ★★★ 关键修改：创建和训练时完全一致的 ODEfunc ★★★
                v_pial_ref = v_in_outer.detach()   # 使用初始 outer 作为 pial reference
                v_white_ref = v_in_inner.detach()  # 使用初始 inner 作为 white reference
                ode_func_eval = ODEfunc(cortexode)
                # ★ odeint input必须是tuple
                y0 = (v_in_inner, v_in_outer)
                # ★ solve ODE
                full_solution = odeint(
                        ode_func_eval, y0, T,
                        method=solver,
                        options=dict(step_size=step_size)
                    )
                v_out_inner = full_solution[0][-1]
                v_out_outer = full_solution[1][-1]
            # v_wm_pred = v_wm_pred[0].cpu().numpy()
            # f_wm_pred = f_in[0].cpu().numpy()
            # v_gm_pred = v_gm_pred[0].cpu().numpy()
            # f_gm_pred = f_in[0].cpu().numpy()
               
            # ------------------------- hypo inner surface -------------------------
            v_out_inner_pred = v_out_inner[0].cpu().numpy()
            faces_pred = faces[0].cpu().numpy()
            # # map the surface coordinate from [-1,1] to its original space
            # 反归一化
            v_out_inner_pred, faces_pred = process_surface_inverse(v_out_inner_pred, faces_pred, volume_shape, data_name)
            # 5T 体素坐标变成物理空间吧
            v_tmp = np.ones((v_out_inner_pred.shape[0], 4))
            v_tmp[:, :3] = v_out_inner_pred
            v_out_inner_pred = v_tmp.dot(brain.affine.T)[:, :3]
            # ------------------------- hypo outer surface -------------------------
            v_out_outer_pred = v_out_outer[0].cpu().numpy()
            faces_pred = faces[0].cpu().numpy()
            # # map the surface coordinate from [-1,1] to its original space
            # 反归一化
            v_out_outer_pred, faces_pred = process_surface_inverse(v_out_outer_pred, faces_pred, volume_shape, data_name)
            # 5T 体素坐标变成物理空间吧
            v_tmp = np.ones((v_out_outer_pred.shape[0], 4))
            v_tmp[:, :3] = v_out_outer_pred
            v_out_outer_pred = v_tmp.dot(brain.affine.T)[:, :3]

        # ------- save predictde surfaces ------- 
        if test_type == 'pred':
            ### save mesh to .obj or .stl format by Trimesh
            # mesh_wm = trimesh.Trimesh(v_wm_pred, f_wm_pred)
            # mesh_gm = trimesh.Trimesh(v_gm_pred, f_gm_pred)
            # mesh_wm.export(result_dir+'wm_'+data_name+'_'+surf_hemi+'_'+subid+'.stl')
            # mesh_gm.export(result_dir+'gm_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')

            # save the surfaces in FreeSurfer format
            # nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.white',
            #                                  v_wm_pred, f_wm_pred)
            # nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.pial',
            #                                  v_gm_pred, f_gm_pred)
            
            # hypo inner surface 输出
            nib.freesurfer.io.write_geometry(result_dir+'/'+data_name+'_'+surf_hemi+'_'+subid+'_'+tag +'.inner',
                                             v_out_inner_pred, faces_pred)
            # hypo outer surface 输出
            nib.freesurfer.io.write_geometry(result_dir+'/'+data_name+'_'+surf_hemi+'_'+subid+'_'+ tag +'.outer',
                                             v_out_outer_pred, faces_pred)
            
        # ------- load ground truth surfaces ------- 
        if test_type == 'eval':
            if data_name == 'hcp':
                v_wm_gt, f_wm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white.deformed')
                v_gm_gt, f_gm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial.deformed')
            elif data_name == 'adni':
                v_wm_gt, f_wm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
                v_gm_gt, f_gm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_left_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_right_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')

                # apply the affine transformation provided by brain MRI nifti
                v_tmp = np.ones([v_wm_gt.shape[0],4])
                v_tmp[:,:3] = v_wm_gt
                v_wm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
                v_tmp = np.ones([v_gm_gt.shape[0],4])
                v_tmp[:,:3] = v_gm_gt
                v_gm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]

            elif data_name == "5T":
                #  ------------------------- ground truth hypo inner surface  ------------------------- 
                gt_inner_surf = os.path.join(data_dir, subid, "surf", f"{surf_hemi}_low_signal_inner.white")
                if not os.path.exists(gt_inner_surf):
                    print(f"[Warning] Missing GT inner surface: {gt_inner_surf}")
                    # skipped += 1
                    continue
                v_gt_inner, faces = nib.freesurfer.io.read_geometry(gt_inner_surf)
                if v_gt_inner.size == 0 or faces.size == 0:
                    print(f"[Warning] Empty geometry in GT inner surface: {gt_inner_surf}")
                    # skipped += 1
                    continue
                faces = faces.astype(np.int64).copy()
                #  ------------------------- ground truth hypo outer surface  ------------------------- 
                gt_outer_surf = os.path.join(data_dir, subid, "surf", f"{surf_hemi}_low_signal_outer.white")
                if not os.path.exists(gt_outer_surf):
                    print(f"[Warning] Missing GT outer surface: {gt_outer_surf}")
                    # skipped += 1
                    continue
                v_gt_outer, faces = nib.freesurfer.io.read_geometry(gt_outer_surf)
                if v_gt_outer.size == 0 or faces.size == 0:
                    print(f"[Warning] Empty geometry in GT outer surface: {gt_outer_surf}")
                    # skipped += 1
                    continue
                faces = faces.astype(np.int64).copy()

        # ------- evaluation -------
        if test_type == 'eval':
            # v_wm_pred = torch.Tensor(v_wm_pred).unsqueeze(0).to(device)
            # f_wm_pred = torch.LongTensor(f_wm_pred).unsqueeze(0).to(device)
            # v_gm_pred = torch.Tensor(v_gm_pred).unsqueeze(0).to(device)
            # f_gm_pred = torch.LongTensor(f_gm_pred).unsqueeze(0).to(device)
            # # v_wm_gt = torch.Tensor(v_wm_gt).unsqueeze(0).to(devishce)
            # # f_wm_gt = torch.LongTensor(f_wm_gt.astype(np.float32)).unsqueeze(0).to(device)
            # v_gm_gt = torch.Tensor(v_gm_gt).unsqueeze(0).to(device)
            # f_gm_gt = torch.LongTensor(f_gm_gt.astype(np.float32)).unsqueeze(0).to(device)

            v_out_inner_pred = torch.Tensor(v_out_inner_pred).unsqueeze(0).to(device)
            v_out_outer_pred = torch.Tensor(v_out_outer_pred).unsqueeze(0).to(device)
            v_gt_inner = torch.Tensor(v_gt_inner).unsqueeze(0).to(device)
            v_gt_outer = torch.Tensor(v_gt_outer).unsqueeze(0).to(device)
            faces = torch.LongTensor(faces).unsqueeze(0).to(device)

            # compute ASSD and HD
            # assd_wm, hd_wm = compute_mesh_distance(v_wm_pred, v_wm_gt, f_wm_pred, f_wm_gt)
            assd_inner, hd_inner = compute_mesh_distance(v_out_inner_pred, v_gt_inner, faces, faces)
            assd_outer, hd_outer = compute_mesh_distance(v_out_outer_pred, v_gt_outer, faces, faces)
            if data_name == 'dhcp':  # the resolution is 0.7
                # assd_wm = 0.7*assd_wm
                assd_gm = 0.7*assd_gm
                # hd_wm = 0.7*hd_wm
                hd_gm = 0.7*hd_gm

            # assd_wm_all.append(assd_wm)
            
            assd_inner_all.append(assd_inner)
            assd_outer_all.append(assd_outer)
            # hd_wm_all.append(hd_wm)
            hd_inner_all.append(hd_inner)
            hd_outer_all.append(hd_outer)
            ### compute percentage of self-intersecting faces
            ### uncomment below if you have installed torch-mesh-isect
            ### https://github.com/vchoutas/torch-mesh-isect
            # sif_wm_all.append(check_self_intersect(v_wm_pred, f_wm_pred, collisions=20))
            # sif_gm_all.append(check_self_intersect(v_gm_pred, f_gm_pred, collisions=20))

            # sif_wm_all.append(0)
            sif_inner_all.append(check_self_intersect(v_out_inner_pred, faces, collisions=20))
            sif_outer_all.append(check_self_intersect(v_out_outer_pred, faces, collisions=20))
         
        # break

    # ------- report the final results ------- 
    if test_type == 'eval':
        print(f'tag: {tag}')
        print('======== hypo inner ========')
        print('assd mean:', np.mean(assd_inner_all))
        print('assd std:', np.std(assd_inner_all))
        print('hd mean:', np.mean(hd_inner_all))
        print('hd std:', np.std(hd_inner_all))
        print('sif mean:', np.mean(sif_inner_all))
        print('sif std:', np.std(sif_inner_all))
        print('======== hypo outer ========')
        print('assd mean:', np.mean(assd_outer_all))
        print('assd std:', np.std(assd_outer_all))
        print('hd mean:', np.mean(hd_outer_all))
        print('hd std:', np.std(hd_outer_all))
        print('sif mean:', np.mean(sif_outer_all))
        print('sif std:', np.std(sif_outer_all))