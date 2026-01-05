import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import trimesh
from data.preprocess import process_volume, process_surface, process_surface_inverse
# from util.mesh import laplacian_smooth, compute_normal


# ----------------------------
#  for surface reconstruction
# ----------------------------

class BrainData():
    """
    For bi-directional CortexODE.

    Inputs:
        v_in_inner : inner cortical surface vertices
        v_in_outer : outer cortical surface vertices

    Ground truth:
        v_gt_inner : hypointense-layer inner surface
        v_gt_outer : hypointense-layer outer surface

        f : faces (assumed identical for all surfaces)
    """

    def __init__(self,
                 volume_path,
                 v_in_inner, v_in_outer,
                 v_gt_inner, v_gt_outer,
                 faces):

        # Convert to tensors
        self.v_in_inner  = torch.Tensor(v_in_inner)
        self.v_in_outer  = torch.Tensor(v_in_outer)
        self.v_gt_inner  = torch.Tensor(v_gt_inner)
        self.v_gt_outer  = torch.Tensor(v_gt_outer)

        # faces 通常四个 surface 用的是同一个 mesh
        faces = faces.astype(np.int64).copy()
        self.faces = torch.LongTensor(faces)

        # for loading the MRI volume later
        self.volume_path = volume_path

        # Free memory
        v_in_inner = v_in_outer = None
        v_gt_inner = v_gt_outer = None
        faces = None
        
        
class BrainDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        brain = self.data[i]

        # ======================================
        # 1. Loads MRI volume lazily
        # ======================================
        img = nib.load(brain.volume_path)
        vol = img.get_fdata(dtype=np.float32)

        # Normalize / preprocess by dataset type
        vol = process_volume(vol, data_name='5T')

        # Convert to tensor
        vol = torch.from_numpy(vol)

        # ======================================
        # 2. Return bi-directional surfaces
        # ======================================
        return (
            vol,
            brain.v_in_inner,     # white matter surface
            brain.v_in_outer,     # pial surface
            brain.v_gt_inner,     # hypointense-layer inner surface
            brain.v_gt_outer,     # hypointense-layer outer surface
            brain.faces
        )

def load_surf_data(config, subject_list=None):
    """
    Load brain MRI and surfaces for given subjects.
    Automatically skip missing files or geometry errors.
    """

    import traceback

    data_dir = config.data_dir
    data_name = config.data_name
    surf_hemi = config.surf_hemi

    if subject_list is None:
        subject_list = sorted(os.listdir(data_dir))

    data_list = []
    skipped = 0

    for subid in tqdm(subject_list):
        try:
            # --------------------- Load MRI ---------------------
            mri_path = os.path.join(data_dir, subid, "image", "brain_high.nii.gz")
            if not os.path.exists(mri_path):
                print(f"[Warning] MRI not found: {mri_path}")
                skipped += 1
                continue

            brain = nib.load(mri_path)
            brain_arr = brain.get_fdata()
            if brain_arr is None or np.all(brain_arr == 0):
                print(f"[Warning] Empty MRI for {subid}")
                skipped += 1
                continue

            volume_shape = brain_arr.shape[:3]

            # =====================================================
            #                  Load white surface
            # =====================================================
            white_path = os.path.join(
                data_dir, subid, "surf", f"{surf_hemi}.white"
            )
            # white_path = os.path.join(
            #     data_dir, subid, "surf", f"{surf_hemi}_initial_low_signal_inner.white"
            # )
            # white_path = "/home_data/home/caoshui2024/DeepLearning_BrainMLSR/CortexODE-Bi/test/case/Fazekas2_0000049581_095449/init_surf/lh.layer_inner"

            if not os.path.exists(white_path):
                print(f"[Warning] Missing white: {white_path}")
                skipped += 1
                continue

            v_in_inner, faces = nib.freesurfer.io.read_geometry(white_path)
            if v_in_inner.size == 0:
                print(f"[Warning] Empty white surface: {white_path}")
                skipped += 1
                continue

            faces = faces.astype(np.int64).copy()

            # Convert to voxel space
            v_tmp = np.ones([v_in_inner.shape[0], 4])
            v_tmp[:, :3] = v_in_inner
            v_in_inner = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]

            # normalize surface
            v_in_inner, faces = process_surface(v_in_inner, faces, volume_shape, data_name)

            # =====================================================
            #                  Load pial surface
            # =====================================================
            pial_path = os.path.join(
                data_dir, subid, "surf", f"{surf_hemi}.pial"
            )
            # pial_path = os.path.join(
            #     data_dir, subid, "surf", f"{surf_hemi}_initial_low_signal_outer.white"
            # )
            # pial_path = "/home_data/home/caoshui2024/DeepLearning_BrainMLSR/CortexODE-Bi/test/case/Fazekas2_0000049581_095449/init_surf/lh.layer_outer"
            
            if not os.path.exists(pial_path):
                print(f"[Warning] Missing pial: {pial_path}")
                skipped += 1
                continue

            v_in_outer, faces = nib.freesurfer.io.read_geometry(pial_path)
            if v_in_outer.size == 0:
                print(f"[Warning] Empty pial surface: {pial_path}")
                skipped += 1
                continue

            faces = faces.astype(np.int64).copy()

            v_tmp = np.ones([v_in_outer.shape[0], 4])
            v_tmp[:, :3] = v_in_outer
            v_in_outer = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]

            v_in_outer, faces = process_surface(v_in_outer, faces, volume_shape, data_name)

            # =====================================================
            #                  Load hypo inner surface
            # =====================================================
            hypo_inner_path = os.path.join(
                data_dir, subid, "surf", f"{surf_hemi}_low_signal_inner.white"
            )
            if not os.path.exists(hypo_inner_path):
                print(f"[Warning] Missing hypo inner: {hypo_inner_path}")
                skipped += 1
                continue

            v_gt_inner, faces = nib.freesurfer.io.read_geometry(hypo_inner_path)
            if v_gt_inner.size == 0:
                print(f"[Warning] Empty hypo inner: {hypo_inner_path}")
                skipped += 1
                continue

            faces = faces.astype(np.int64).copy()

            v_tmp = np.ones([v_gt_inner.shape[0], 4])
            v_tmp[:, :3] = v_gt_inner
            v_gt_inner = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]

            v_gt_inner, faces = process_surface(v_gt_inner, faces, volume_shape, data_name)

            # =====================================================
            #                  Load hypo outer surface
            # =====================================================
            hypo_outer_path = os.path.join(
                data_dir, subid, "surf", f"{surf_hemi}_low_signal_outer.white"
            )
            if not os.path.exists(hypo_outer_path):
                print(f"[Warning] Missing hypo outer: {hypo_outer_path}")
                skipped += 1
                continue

            v_gt_outer, faces = nib.freesurfer.io.read_geometry(hypo_outer_path)
            if v_gt_outer.size == 0:
                print(f"[Warning] Empty hypo outer: {hypo_outer_path}")
                skipped += 1
                continue

            faces = faces.astype(np.int64).copy()

            v_tmp = np.ones([v_gt_outer.shape[0], 4])
            v_tmp[:, :3] = v_gt_outer
            v_gt_outer = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]

            v_gt_outer, faces = process_surface(v_gt_outer, faces, volume_shape, data_name)

            # =====================================================
            #                Create BrainData item
            # =====================================================
            braindata = BrainData(
                volume_path=mri_path,
                v_in_inner=v_in_inner,
                v_in_outer=v_in_outer,
                v_gt_inner=v_gt_inner,
                v_gt_outer=v_gt_outer,
                faces=faces
            )

            data_list.append(braindata)
            # 只测试一例的话
            # break 
        except Exception as e:
            print(f"[Error] Failed to load {subid}: {e}")
            traceback.print_exc(limit=1)
            skipped += 1
            continue

    print(f"✅ Loaded {len(data_list)} subjects, skipped {skipped}.")
    return BrainDataset(data_list)
