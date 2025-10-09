from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
import torch.nn as nn, torch
from .utils import linear_relu_ln, GaussianPrediction, cartesian, reverse_cartesian
from .utils import safe_sigmoid
from .transformer_decoder import MLP
from ..gaussianfusion_config import GaussianFusionConfig

@MODELS.register_module()
class SparseGaussianRefinementModuleV2(BaseModule):
    def __init__(
        self,
        config: GaussianFusionConfig,
        refine_type='exp',
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = config.embed_dims
        self.exp_param_dim = 5 + config.semantic_dim
        self.semantic_dim = config.semantic_dim
        self.refine_type = refine_type

        self.ego_scale = config.ego_scale
        self.pc_range = np.array(config.pc_range)
        self.scale_range = np.array(config.scale_range)
        xy_range =self.pc_range[[0, 1, 3, 4]]
        self.register_buffer("xy_range", torch.from_numpy(xy_range), False)
        self.register_buffer("unit_xy", torch.tensor(config.unit_xy, dtype=torch.float), False)
        if self.refine_type == 'exp':
            self.layer = MLP(self.embed_dims, self.embed_dims * 4, self.exp_param_dim)
        elif self.refine_type == 'imp':
            self.layer = MLP(self.embed_dims, self.embed_dims * 4, 3)
        else:
            raise NotImplementedError

    def forward(
        self,
        gaussian: GaussianPrediction,
        **kwargs,
    ):
        if self.refine_type == 'exp':
            original_means = gaussian.means.detach()
            feature = gaussian.features
            gs_params = self.layer(feature)
            delta_xy = (2 * safe_sigmoid(gs_params[..., :2]) - 1.0) * self.unit_xy[None, None]
            xy = original_means + delta_xy
            scale = safe_sigmoid(gs_params[..., 2:4]) * \
                        (self.scale_range[1] - self.scale_range[0]) + \
                        self.scale_range[0]
            headings = gs_params[..., 4:5]
            rot = torch.cat([torch.sin(headings), torch.cos(headings)], dim=-1)
            sem = gs_params[..., 5:]

        elif self.refine_type == 'imp':
            feature = gaussian.features
            gs_params = self.layer(feature)
            xy = gs_params[..., :2] #original_means[..., :2] + delta_xy
            headings = gs_params[..., 2:] #safe_sigmoid(gs_params[..., 2:]) * 2 - 1
            rot = torch.cat([torch.sin(headings), torch.cos(headings)], dim=-1)
            scale = gaussian.scales
            sem = None
            original_means = None
        else:
            raise NotImplementedError

        new_gaussian = GaussianPrediction(
                means=xy,
                scales=scale,
                headings=headings,
                rotations=rot,
                semantics=sem,
                features=gaussian.features,
                original_means=original_means,
                exp_samples=gaussian.exp_samples
            )

        return new_gaussian