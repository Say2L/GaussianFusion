from torch import Tensor
from einops import reduce
import torch
import torch, torch.nn as nn
from typing import NamedTuple
import torch.nn.functional as F

SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def dict_to_gpu(input):
    if not isinstance(input, dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        return input

    for k, v in input.items():
        input[k] = dict_to_gpu(v)
    return input

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)


def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def spherical2cartesian(anchor, pc_range, phi_activation="loop"):
    if phi_activation == "sigmoid":
        xyz = safe_sigmoid(anchor[..., :3])
    elif phi_activation == "loop":
        xy = safe_sigmoid(anchor[..., :2])
        z = torch.remainder(anchor[..., 2:3], 1.0)
        xyz = torch.cat([xy, z], dim=-1)
    else:
        raise NotImplementedError
    rrr = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    theta = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    phi = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xxx = rrr * torch.sin(theta) * torch.cos(phi)
    yyy = rrr * torch.sin(theta) * torch.sin(phi)
    zzz = rrr * torch.cos(theta)
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)

    return xyz


def cartesian(anchor, pc_range, use_sigmoid=True):
    if use_sigmoid:
        xy = safe_sigmoid(anchor[..., :2])
    else:
        xy = anchor[..., :2].clamp(min=1e-6, max=1 - 1e-6)
    xxx = xy[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xy[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    xy = torch.stack([xxx, yyy], dim=-1)

    return xy


def reverse_cartesian(xyz, pc_range, use_sigmoid=True):
    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    # zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    unitxy = torch.stack([xxx, yyy], dim=-1)
    if use_sigmoid:
        anchor = safe_inverse_sigmoid(unitxy)
    else:
        anchor = unitxy.clamp(min=1e-6, max=1 - 1e-6)
    return anchor


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class GaussianPrediction(NamedTuple):
    means: Tensor = None
    scales: Tensor = None
    rotations: Tensor = None
    semantics: Tensor = None
    opacities: Tensor = None
    original_means: Tensor = None
    delta_means: Tensor = None
    features: Tensor = None
    headings: Tensor = None
    exp_samples: int = 0


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 2

    # theta = torch.arctan2(tensor[..., 0], tensor[..., 1])
    mat = torch.zeros(
        *tensor.shape[:-1], 3, 3, dtype=tensor.dtype, device=tensor.device
    )
    cos_theta = tensor[..., 1]
    sin_theta = tensor[..., 0]
    mat[..., 0, 0], mat[..., 0, 1] = cos_theta, -sin_theta
    mat[..., 1, 0], mat[..., 1, 1] = sin_theta, cos_theta
    mat[..., 2, 2] = 1
    return mat

# def get_rotation_matrix(tensor):
#     assert tensor.shape[-1] == 4

#     tensor = F.normalize(tensor, dim=-1)
#     mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
#     mat1[..., 0, 0] = tensor[..., 0]
#     mat1[..., 0, 1] = - tensor[..., 1]
#     mat1[..., 0, 2] = - tensor[..., 2]
#     mat1[..., 0, 3] = - tensor[..., 3]
    
#     mat1[..., 1, 0] = tensor[..., 1]
#     mat1[..., 1, 1] = tensor[..., 0]
#     mat1[..., 1, 2] = - tensor[..., 3]
#     mat1[..., 1, 3] = tensor[..., 2]

#     mat1[..., 2, 0] = tensor[..., 2]
#     mat1[..., 2, 1] = tensor[..., 3]
#     mat1[..., 2, 2] = tensor[..., 0]
#     mat1[..., 2, 3] = - tensor[..., 1]

#     mat1[..., 3, 0] = tensor[..., 3]
#     mat1[..., 3, 1] = - tensor[..., 2]
#     mat1[..., 3, 2] = tensor[..., 1]
#     mat1[..., 3, 3] = tensor[..., 0]

#     mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
#     mat2[..., 0, 0] = tensor[..., 0]
#     mat2[..., 0, 1] = - tensor[..., 1]
#     mat2[..., 0, 2] = - tensor[..., 2]
#     mat2[..., 0, 3] = - tensor[..., 3]
    
#     mat2[..., 1, 0] = tensor[..., 1]
#     mat2[..., 1, 1] = tensor[..., 0]
#     mat2[..., 1, 2] = tensor[..., 3]
#     mat2[..., 1, 3] = - tensor[..., 2]

#     mat2[..., 2, 0] = tensor[..., 2]
#     mat2[..., 2, 1] = - tensor[..., 3]
#     mat2[..., 2, 2] = tensor[..., 0]
#     mat2[..., 2, 3] = tensor[..., 1]

#     mat2[..., 3, 0] = tensor[..., 3]
#     mat2[..., 3, 1] = tensor[..., 2]
#     mat2[..., 3, 2] = - tensor[..., 1]
#     mat2[..., 3, 3] = tensor[..., 0]

#     mat2 = torch.conj(mat2).transpose(-1, -2)
    
#     mat = torch.matmul(mat1, mat2)
#     return mat[..., 1:, 1:]
