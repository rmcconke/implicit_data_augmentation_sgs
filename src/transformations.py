from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from irreps_utils import gradient_channels_to_matrix, matrix_to_gradient_channels


def unstandardize_sgs(tensor, scaling_factor):
    # Just scales by the magnitude
    return tensor * scaling_factor.to(tensor.device)

def unstandardize(task_type, input_output, tensor, scaling_factors, equiv_error=False):
    if equiv_error:
        if task_type == 'sgs':
            if input_output == 'input':
                return tensor * scaling_factors['input_mean_magnitude'].to(tensor.device)
            elif input_output == 'output':
                return tensor * scaling_factors['target_mean_magnitude'].to(tensor.device)
        elif task_type == 'filtered':
            if input_output == 'input':
                return tensor * scaling_factors['input_std'].to(tensor.device).unsqueeze(1)  # Equiv error is a difference, so we don't need to add the mean. Unsqueeze for correct broadcasting across channels.
            elif input_output == 'output':
                return tensor * scaling_factors['target_std'].to(tensor.device).unsqueeze(1)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    else:
        if task_type == 'sgs':
            if input_output == 'input':
                return tensor * scaling_factors['input_mean_magnitude'].to(tensor.device)
            elif input_output == 'output':
                return tensor * scaling_factors['target_mean_magnitude'].to(tensor.device)
        elif task_type == 'filtered':
            if input_output == 'input':
                return tensor * scaling_factors['input_std'].to(tensor.device) + scaling_factors['input_mean'].to(tensor.device)
            elif input_output == 'output':
                return tensor * scaling_factors['target_std'].to(tensor.device) + scaling_factors['target_mean'].to(tensor.device)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


def _apply_channel_rotation(tensor: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate channel vectors/tensors using rotation matrices ``R`` of shape ``[B, 3, 3]``.

    Supports scalars (C=1), vectors (C=3), symmetric rank-2 tensors (C=6),
    and full velocity gradients (C=9). Any other channel count is left unchanged.
    """
    B, C, D, H, W = tensor.shape

    if C == 1:
        return tensor

    if C == 3:
        flat = tensor.view(B, C, -1)
        return torch.bmm(R, flat).view(B, C, D, H, W)

    if C == 6:
        flat = tensor.view(B, C, -1)
        T = torch.zeros(B, 3, 3, flat.shape[-1], device=tensor.device, dtype=tensor.dtype)
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = flat[:, 0], flat[:, 1], flat[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = flat[:, 3]
        T[:, 0, 2] = T[:, 2, 0] = flat[:, 4]
        T[:, 1, 2] = T[:, 2, 1] = flat[:, 5]
        T_rot = torch.einsum("bij,bjkn,blk->biln", R, T, R)

        rotated_flat = torch.zeros_like(flat)
        rotated_flat[:, 0] = T_rot[:, 0, 0]
        rotated_flat[:, 1] = T_rot[:, 1, 1]
        rotated_flat[:, 2] = T_rot[:, 2, 2]
        rotated_flat[:, 3] = T_rot[:, 0, 1]
        rotated_flat[:, 4] = T_rot[:, 0, 2]
        rotated_flat[:, 5] = T_rot[:, 1, 2]
        return rotated_flat.view(B, C, D, H, W)

    if C == 9:
        matrix = gradient_channels_to_matrix(tensor)
        T = matrix.view(B, 3, 3, -1)
        T_rot = torch.einsum("bij,bjkn,blk->biln", R, T, R)
        return matrix_to_gradient_channels(T_rot.view(B, 3, 3, D, H, W))

    return tensor


### SO(3) rotation helpers

def _matrix_x(angle: torch.Tensor) -> torch.Tensor:
    c, s = angle.cos(), angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def _matrix_y(angle: torch.Tensor) -> torch.Tensor:
    c, s = angle.cos(), angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def euler_angles_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert Y-X-Y Euler angles ``[α, β, γ]`` to rotation matrices."""
    alpha, beta, gamma = angles[..., 0], angles[..., 1], angles[..., 2]
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return _matrix_y(alpha) @ _matrix_x(beta) @ _matrix_y(gamma)


def rand_so3_angles(num_angles: int) -> torch.Tensor:
    """Sample SO(3) rotations uniformly (Haar measure) in the Y-X-Y convention."""
    alpha = 2 * math.pi * torch.rand(num_angles)
    gamma = 2 * math.pi * torch.rand(num_angles)
    beta = torch.rand(num_angles).mul(2).sub(1).acos()
    return torch.stack([alpha, gamma, beta], dim=1)


def get_all_octahedral_angles() -> torch.Tensor:
    """Return the 24 discrete rotations of the cube in Y-X-Y convention."""
    values = []
    
    # Group 1: Rotations with middle angle = 0 (no tilt)
    # Just 4 rotations around Z axis (avoid gimbal lock by using only first Y)
    for k in range(4):
        values.append((0, 0, k * math.pi / 2))
    
    # Group 2: Tilt to +X face (middle angle = π/2)
    for k in range(4):
        values.append((0, math.pi / 2, k * math.pi / 2))
    
    # Group 3: Tilt to -X face (middle angle = 3π/2)
    for k in range(4):
        values.append((0, 3 * math.pi / 2, k * math.pi / 2))
    
    # Group 4: Flip upside down (middle angle = π)
    for k in range(4):
        values.append((0, math.pi, k * math.pi / 2))
    
    # Group 5: Tilt to +Y face (first angle = π/2, middle angle = π/2)
    for k in range(4):
        values.append((math.pi / 2, math.pi / 2, k * math.pi / 2))
    
    # Group 6: Tilt to -Y face (first angle = π/2, middle angle = 3π/2)
    for k in range(4):
        values.append((math.pi / 2, 3 * math.pi / 2, k * math.pi / 2))
    
    return torch.tensor(values)


def rotate_3d(
    X: torch.Tensor,
    angles: torch.Tensor,
    rotate_channels: bool = True,
    mode: str = "cartesian",
) -> torch.Tensor:
    """
    Rotate a tensor field using ``grid_sample`` for spatial interpolation.

    Args:
        X: Tensor shaped ``[C, D, H, W]`` or ``[B, C, D, H, W]``.
        angles: Euler angles ``[α, β, γ]`` (single or batched).
        rotate_channels: Apply the corresponding channel rotation (vectors/tensors).
        mode: ``"pairwise"`` applies each angle to the matching sample,
              ``"cartesian"`` applies all angles to every sample.
    """
    if X.ndim == 4:
        X = X.unsqueeze(0)
        squeeze_output = True
    elif X.ndim == 5:
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported tensor shape {X.shape}; expected 4D or 5D.")

    batch, channels, D, H, W = X.shape

    angles = angles if angles.ndim == 2 else angles.unsqueeze(0)
    num_angles = angles.shape[0]
    R = euler_angles_to_matrix(angles).to(X.device, X.dtype)

    if mode == "pairwise":
        if batch != num_angles:
            raise ValueError(f"Pairwise mode requires len(angles)==batch ({num_angles} vs {batch})")
        B_out = batch
    elif mode == "cartesian":
        X = X.unsqueeze(1).expand(-1, num_angles, -1, -1, -1, -1).reshape(batch * num_angles, channels, D, H, W)
        R = R.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * num_angles, 3, 3)
        B_out = batch * num_angles
    else:
        raise ValueError("mode must be 'pairwise' or 'cartesian'")

    A = torch.cat([R.transpose(-2, -1), torch.zeros(B_out, 3, 1, device=X.device, dtype=X.dtype)], dim=2)
    grid = F.affine_grid(A, size=X.shape, align_corners=False)
    rotated = F.grid_sample(X, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    if rotate_channels:
        rotated = _apply_channel_rotation(rotated, R.to(rotated.device, rotated.dtype))

    if squeeze_output:
        rotated = rotated.squeeze(0)

    return rotated

def apply_so3_augmentation(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(3) rotations (with interpolation) to inputs and targets."""
    random_angles = rand_so3_angles(inputs.shape[0]).to(inputs.device)
    rotated_inputs = rotate_3d(inputs, random_angles, rotate_channels=True, mode="pairwise")
    rotated_targets = rotate_3d(targets, random_angles, rotate_channels=True, mode="pairwise")
    return rotated_inputs, rotated_targets


### Octahedral rotation helpers


def _rotation_matrix_to_axis_ops(R_int: torch.Tensor) -> Tuple[list[int], list[bool]]:
    axis_map, flips = [], []
    for row in R_int:
        idx = torch.nonzero(row, as_tuple=False).item()
        axis_map.append(idx)
        flips.append(row[idx] < 0)
    return axis_map, flips


def _apply_discrete_rotation(
    X: torch.Tensor,
    R_float: torch.Tensor,
    *,
    rotate_channels: bool = True,
) -> torch.Tensor:
    R_int = torch.round(R_float).to(torch.int32)
    axis_map, flips = _rotation_matrix_to_axis_ops(R_int)

    spatial_dims = [2, 3, 4]
    permute_order = [0, 1] + [spatial_dims[i] for i in axis_map]
    rotated = X.permute(permute_order).contiguous()

    for i, flip_axis in enumerate(flips):
        if flip_axis:
            rotated = torch.flip(rotated, dims=(2 + i,))

    if rotate_channels:
        B = rotated.shape[0]
        R_batch = R_float.to(rotated.device, rotated.dtype).unsqueeze(0).expand(B, -1, -1)
        rotated = _apply_channel_rotation(rotated, R_batch)

    return rotated

def rotate_octahedral_exact(
    X: torch.Tensor,
    angles: torch.Tensor | None = None,
    rotate_channels: bool = True,
    mode: str = "pairwise",
) -> torch.Tensor:
    """
    Rotate tensors by octahedral group elements using exact axis permutations (no interpolation).

    Args:
        X: Tensor shaped ``[C, D, H, W]``, ``[B, C, D, H, W]``, or ``[B, C]``.
        angles: Euler angles. Required for ``"pairwise"`` mode, ignored for ``"whole_group"``.
        rotate_channels: Apply the corresponding channel rotation (vectors/tensors).
        mode: ``"pairwise"`` applies each angle to the matching sample (for augmentation),
              ``"whole_group"`` applies all 24 octahedral rotations to every sample,
              returning shape ``[N, 24, C, D, H, W]``.

    Returns:
        For ``"pairwise"``: same shape as input.
        For ``"whole_group"``: ``[N, 24, C, D, H, W]`` (or ``[N, 24, C]`` for pointwise).
    """
    squeeze_spatial = False
    squeeze_batch = False
    traceless_input = False
    C_raw = X.shape[1] if X.ndim >= 4 else (X.shape[-1] if X.ndim == 2 else X.shape[0])
    if C_raw == 5:
        traceless_input = True
        if X.ndim == 2:
            zz = -X[:, 0:1] - X[:, 1:2]
            X = torch.cat([X[:, 0:1], X[:, 1:2], zz, X[:, 2:5]], dim=1)
        elif X.ndim == 4:
            zz = -X[0:1] - X[1:2]
            X = torch.cat([X[0:1], X[1:2], zz, X[2:5]], dim=0)
        elif X.ndim == 5:
            zz = -X[:, 0:1] - X[:, 1:2]
            X = torch.cat([X[:, 0:1], X[:, 1:2], zz, X[:, 2:5]], dim=1)

    if X.ndim == 2:
        # [B, C] -> [B, C, 1, 1, 1]
        X = X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        squeeze_spatial = True
    elif X.ndim == 4:
        # [C, D, H, W] -> [1, C, D, H, W]
        X = X.unsqueeze(0)
        squeeze_batch = True
    elif X.ndim == 5:
        # [B, C, D, H, W] -> fine as-is
        pass
    else:
        raise ValueError(f"Unsupported tensor shape {X.shape}")

    batch, C, D, H, W = X.shape
    is_pointwise = (D == 1 and H == 1 and W == 1)

    if mode == "whole_group" or mode == "whole_group_flat":
        # Apply all 24 octahedral rotations to every sample
        # whole_group: Output shape [N, 24, C, D, H, W]
        # whole_group_flat: Output shape [N*24, C, D, H, W]
        all_angles = get_all_octahedral_angles().to(X.device)
        rotation_mats = euler_angles_to_matrix(all_angles).to(X.device, X.dtype)
        num_rotations = rotation_mats.shape[0]  # 24

        if is_pointwise and rotate_channels:
            # Fast path: just rotate channels, no spatial ops
            # X: (B, C, 1, 1, 1), rotation_mats: (24, 3, 3)
            # Output: (B, 24, C, 1, 1, 1)
            rotated = torch.empty(batch, num_rotations, C, 1, 1, 1, device=X.device, dtype=X.dtype)
            for j in range(num_rotations):
                R_j = rotation_mats[j].unsqueeze(0).expand(batch, -1, -1)  # (B, 3, 3)
                rotated[:, j] = _apply_channel_rotation(X, R_j)
        elif is_pointwise and not rotate_channels:
            # No rotation needed, just expand
            rotated = X.unsqueeze(1).expand(-1, num_rotations, -1, -1, -1, -1).clone()
        else:
            # Original slow loop for 3D data
            rotated = torch.empty(batch, num_rotations, C, D, H, W, device=X.device, dtype=X.dtype)
            for j in range(num_rotations):
                R_j = rotation_mats[j]
                for i in range(batch):
                    rotated[i, j] = _apply_discrete_rotation(
                        X[i:i+1], R_j, rotate_channels=rotate_channels
                    ).squeeze(0)

        if squeeze_spatial:
            rotated = rotated.squeeze(-1).squeeze(-1).squeeze(-1)

        if squeeze_batch:
            rotated = rotated.squeeze(0)

        if mode == "whole_group_flat":
            # Flatten [N, 24, ...] -> [N*24, ...]
            rotated = rotated.flatten(0, 1)

        if traceless_input:
            if rotated.ndim == 2:
                rotated = torch.cat([rotated[:, 0:1], rotated[:, 1:2], rotated[:, 3:6]], dim=1)
            elif rotated.ndim == 5:
                rotated = torch.cat([rotated[:, 0:1], rotated[:, 1:2], rotated[:, 3:6]], dim=1)
            elif rotated.ndim == 4:
                rotated = torch.cat([rotated[0:1], rotated[1:2], rotated[3:6]], dim=0)
            elif rotated.ndim == 6:  # whole_group output
                rotated = torch.cat([rotated[:, :, 0:1], rotated[:, :, 1:2], rotated[:, :, 3:6]], dim=2)

        return rotated

    elif mode == "pairwise":
        if angles is None:
            raise ValueError("angles required for pairwise mode")

        angles = angles if angles.ndim == 2 else angles.unsqueeze(0)
        rotation_mats = euler_angles_to_matrix(angles).to(X.device, X.dtype)
        num_angles = rotation_mats.shape[0]

        if num_angles != batch:
            raise ValueError(f"Pairwise mode requires len(angles)==batch ({num_angles} vs {batch})")

        if is_pointwise:
            # Fast path: skip spatial ops, just rotate channels
            if rotate_channels:
                rotated = _apply_channel_rotation(X, rotation_mats)
            else:
                rotated = X.clone()
        else:
            # Original loop for 3D data
            rotated = torch.cat(
                [
                    _apply_discrete_rotation(X[i : i + 1], rotation_mats[i], rotate_channels=rotate_channels)
                    for i in range(batch)
                ],
                dim=0,
            )

        if squeeze_spatial:
            rotated = rotated.squeeze(-1).squeeze(-1).squeeze(-1)
        if squeeze_batch:
            rotated = rotated.squeeze(0)

        if traceless_input:
            if rotated.ndim == 2:
                rotated = torch.cat([rotated[:, 0:1], rotated[:, 1:2], rotated[:, 3:6]], dim=1)
            elif rotated.ndim == 5:
                rotated = torch.cat([rotated[:, 0:1], rotated[:, 1:2], rotated[:, 3:6]], dim=1)
            elif rotated.ndim == 4:
                rotated = torch.cat([rotated[0:1], rotated[1:2], rotated[3:6]], dim=0)
            elif rotated.ndim == 6:  # whole_group output
                rotated = torch.cat([rotated[:, :, 0:1], rotated[:, :, 1:2], rotated[:, :, 3:6]], dim=2)

        return rotated

    else:
        raise ValueError("mode must be 'pairwise', 'whole_group', or 'whole_group_flat'")    
def apply_octahedral_augmentation(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply exact octahedral rotations (no interpolation) to inputs and targets."""
    octahedral_angles = get_all_octahedral_angles().to(inputs.device)
    random_angles = octahedral_angles[torch.randint(0, len(octahedral_angles), (inputs.shape[0],))]

    rotated_inputs = rotate_octahedral_exact(
        inputs,
        random_angles,
        rotate_channels=True,
        mode="pairwise",
    )
    rotated_targets = rotate_octahedral_exact(
        targets,
        random_angles,
        rotate_channels=True,
        mode="pairwise",
    )
    return rotated_inputs, rotated_targets

def precompute_octahedral_augmentations(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rotated_inputs = rotate_octahedral_exact(
        inputs,
        mode="whole_group_flat",
    )
    print(f"inputs precomputed")
    rotated_targets = rotate_octahedral_exact(
        targets,
        mode="whole_group_flat",
    )
    return rotated_inputs, rotated_targets



