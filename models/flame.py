from __future__ import annotations

# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks


def to_tensor(array: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
    return None


def to_np(array: Any, dtype: type = np.float32) -> np.ndarray:
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct:
    def __init__(self, **kwargs: Any) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)


__dir__ = Path(__file__).parent.absolute()
FLAMEConfig = SimpleNamespace(
    flame_model_path=str(__dir__ / 'data/FLAME2020/generic_model.pkl'),
    n_shape=100,
    n_exp=50,
    n_tex=50,
    tex_type='BFM',
    tex_path=str(__dir__ / 'data/FLAME2020/FLAME_albedo_from_BFM.npz'),
    flame_lmk_embedding_path=str(__dir__ / 'data/landmark_embedding.npy'),
)


class FLAME(nn.Module):
    """
    Borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given FLAME parameters, generates a differentiable FLAME function that outputs
    a mesh and 2D/3D facial landmarks.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, : config.n_shape], shapedirs[:, :, 300 : 300 + config.n_exp]], 2
        )
        self.register_buffer('shapedirs', shapedirs)

        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))

        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing eyeball and neck rotation
        default_eyeball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyeball_pose, requires_grad=False))
        default_eyeball_pose_mat = torch.eye(3, dtype=self.dtype, requires_grad=False).view(1, 9).repeat(1, 2)
        self.register_parameter('eye_pose_mat', nn.Parameter(default_eyeball_pose_mat, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        default_neck_pose_mat = torch.eye(3, dtype=self.dtype, requires_grad=False).view(1, 9)
        self.register_parameter('neck_pose_mat', nn.Parameter(default_neck_pose_mat, requires_grad=False))

        # Static and dynamic landmark embeddings
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1'
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            'lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long()
        )
        self.register_buffer(
            'lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype),
        )
        self.register_buffer(
            'dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long()
        )
        self.register_buffer(
            'dynamic_lmk_bary_coords',
            lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype),
        )
        self.register_buffer(
            'full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long()
        )
        self.register_buffer(
            'full_lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype),
        )

        neck_kin_chain: list[torch.Tensor] = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose: torch.Tensor,
        dynamic_lmk_faces_idx: torch.Tensor,
        dynamic_lmk_b_coords: torch.Tensor,
        neck_kin_chain: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        pose2rot: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select face contour landmarks depending on head orientation."""
        batch_size = pose.shape[0]

        if pose2rot:
            aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
            rot_mats = batch_rodrigues(
                aa_pose.view(-1, 3), dtype=dtype
            ).view(batch_size, -1, 3, 3)
        else:
            rot_mats = torch.index_select(
                pose.view(batch_size, -1, 9), 1, neck_kin_chain
            ).view(batch_size, -1, 3, 3)

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def seletec_3d68(self, vertices: torch.Tensor) -> torch.Tensor:
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def forward(
        self,
        shape_params: torch.Tensor | None = None,
        expression_params: torch.Tensor | None = None,
        pose_params: torch.Tensor | None = None,
        eye_pose_params: torch.Tensor | None = None,
        pose2rot: bool = True,
        ignore_global_rot: bool = False,
        return_lm2d: bool = True,
        return_lm3d: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            shape_params: (N, n_shape)
            expression_params: (N, n_exp)
            pose_params: (N, 6) pose parameters
            eye_pose_params: (N, 6) eye pose parameters
            pose2rot: if True, convert axis-angle to rotation matrices
            ignore_global_rot: if True, zero out global rotation
            return_lm2d: whether to compute 2D landmarks
            return_lm3d: whether to compute 3D landmarks

        Returns:
            vertices: (N, V, 3)
            landmarks2d: (N, L, 3) or None
            landmarks3d: (N, L, 3) or None
        """
        batch_size = shape_params.shape[0]
        betas = torch.cat([shape_params, expression_params], dim=1)

        if pose2rot:
            if pose_params is None:
                pose_params = self.eye_pose.expand(batch_size, -1)
            if eye_pose_params is None:
                eye_pose_params = self.eye_pose.expand(batch_size, -1)
            head_pose = (
                pose_params[:, :3]
                if not ignore_global_rot
                else torch.zeros_like(pose_params[:, :3])
            )
            full_pose = torch.cat(
                [head_pose, self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params],
                dim=1,
            )
        else:
            if pose_params is None:
                pose_params = self.eye_pose_mat.expand(batch_size, -1)
            if eye_pose_params is None:
                eye_pose_params = self.eye_pose_mat.expand(batch_size, -1)
            head_pose = (
                pose_params[:, :9]
                if not ignore_global_rot
                else self.eye_pose_mat.expand(batch_size, -1)[:, :9]
            )
            full_pose = torch.cat(
                [head_pose, self.neck_pose_mat.expand(batch_size, -1), pose_params[:, 9:], eye_pose_params],
                dim=1,
            )

        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot,
            self.dtype,
        )

        landmarks2d: torch.Tensor | None = None
        if return_lm2d:
            lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
            lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                dtype=self.dtype,
                pose2rot=pose2rot,
            )
            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
            landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        landmarks3d: torch.Tensor | None = None
        if return_lm3d:
            bz = vertices.shape[0]
            landmarks3d = vertices2landmarks(
                vertices,
                self.faces_tensor,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )

        return vertices, landmarks2d, landmarks3d


class FLAMETex(nn.Module):
    """
    FLAME texture model.
    Ref: https://github.com/TimoBolkart/TF_FLAME
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            warnings.warn(f'Texture type {config.tex_type!r} not found.', stacklevel=2)
            raise NotImplementedError

        n_tex = config.n_tex
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode: torch.Tensor) -> torch.Tensor:
        """
        Args:
            texcode: (batchsize, n_tex)

        Returns:
            torch.Tensor: (bz, 3, 256, 256) texture in range [0, 1]
        """
        bs = texcode.shape[0]
        texcode = texcode[:1]

        # Use the same (first frame) texture for all frames
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :].repeat(bs, 1, 1, 1)
        return texture
