from __future__ import annotations

# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import torch
import torch.nn.functional as F


def rot_mat_to_euler(rot_mats: torch.Tensor) -> torch.Tensor:
    """Calculates the y-axis euler angle from a batch of rotation matrices.

    Args:
        rot_mats: (N, 3, 3)

    Returns:
        torch.Tensor: (N,) euler angles around y-axis
    """
    sy = torch.sqrt(
        rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
    )
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(
    vertices: torch.Tensor,
    pose: torch.Tensor,
    dynamic_lmk_faces_idx: torch.Tensor,
    dynamic_lmk_b_coords: torch.Tensor,
    neck_kin_chain: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the faces and barycentric coordinates for the dynamic landmarks.

    Args:
        vertices: (B, V, 3)
        pose: (B, J*3)
        dynamic_lmk_faces_idx: (L,) look-up table from neck rotation to faces
        dynamic_lmk_b_coords: (L, 3) look-up table from neck rotation to barycentric coords
        neck_kin_chain: joint indices forming the neck kinematic chain
        dtype: data type

    Returns:
        dyn_lmk_faces_idx: (B, L)
        dyn_lmk_b_coords: (B, L, 3)
    """
    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype
    ).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
    ).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    lmk_faces_idx: torch.Tensor,
    lmk_bary_coords: torch.Tensor,
) -> torch.Tensor:
    """Calculate landmarks by barycentric interpolation.

    Args:
        vertices: (B, V, 3)
        faces: (F, 3)
        lmk_faces_idx: (B, L) face indices for each landmark
        lmk_bary_coords: (B, L, 3) barycentric weights

    Returns:
        torch.Tensor: (B, L, 3) landmark coordinates
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += (
        torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
    )
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)
    return landmarks


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    pose2rot: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform Linear Blend Skinning with the given shape and pose parameters.

    Args:
        betas: (B, NB) shape parameters
        pose: (B, (J+1)*3) axis-angle pose parameters, or (B, (J+1)*9) rotation matrices
        v_template: (B, V, 3) template mesh vertices
        shapedirs: (V, 3, NB) shape displacement basis
        posedirs: (P, V*3) pose PCA coefficients
        J_regressor: (J, V) joint regressor
        parents: (J,) kinematic tree
        lbs_weights: (V, J+1) linear blend skinning weights
        pose2rot: if True, convert axis-angle to rotation matrices
        dtype: data type

    Returns:
        verts: (B, V, 3) deformed vertices
        joints: (B, J, 3) joint locations
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype
        ).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Calculate 3D joint locations from vertices.

    Args:
        J_regressor: (J, V)
        vertices: (B, V, 3)

    Returns:
        torch.Tensor: (B, J, 3) joint locations
    """
    return torch.einsum('bik,ji->bjk', vertices, J_regressor)


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """Calculate per-vertex displacements from blend shape coefficients.

    Args:
        betas: (B, NB) blend shape coefficients
        shape_disps: (V, 3, NB) blend shapes

    Returns:
        torch.Tensor: (B, V, 3) per-vertex displacements
    """
    return torch.einsum('bl,mkl->bmk', betas, shape_disps)


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate rotation matrices for a batch of axis-angle vectors.

    Args:
        rot_vecs: (N, 3) axis-angle vectors
        epsilon: small value for numerical stability
        dtype: data type

    Returns:
        torch.Tensor: (N, 3, 3) rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Create a batch of 4x4 transformation matrices from rotation and translation.

    Args:
        R: (B, 3, 3) rotation matrices
        t: (B, 3, 1) translation vectors

    Returns:
        torch.Tensor: (B, 4, 4)
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a batch of rigid transformations to joints.

    Args:
        rot_mats: (B, J, 3, 3) rotation matrices
        joints: (B, J, 3) rest-pose joint locations
        parents: (J,) kinematic tree
        dtype: data type

    Returns:
        posed_joints: (B, J, 3) posed joint locations
        rel_transforms: (B, J, 4, 4) relative rigid transformations
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1),
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms
