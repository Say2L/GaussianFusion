import torch, torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from torch.cuda.amp import autocast
from .transformer_decoder import MLP
from .utils import safe_inverse_sigmoid, safe_sigmoid, cartesian, reverse_cartesian, GaussianPrediction


@MODELS.register_module()
class GaussianInit(BaseModule):
    def __init__(
        self,
        embed_dims,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xy_activation="sigmoid",
        scale_activation="sigmoid",
        pc_range=[-32, -32, 0.2, 32, 32, 20.2],
        scale_range=[0.01, 3.2],
        ego_scale=[5.2, 2.3],
        random_samples_front=448,
        random_samples_back=64,
        implicit_samples=512,
        **kwargs,
    ):
        super().__init__()
        # assert implicit_samples == random_samples_back + random_samples_front
        self.embed_dims = embed_dims
        self.xy_act = xy_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.ego_scale = ego_scale

        self.random_samples_front = random_samples_front
        self.random_samples_back = random_samples_back
        self.explicit_samples = random_samples_front + random_samples_back
        self.implicit_samples = implicit_samples

        self.exp_gs_param = self.init_exp_gaussians()

        # self.exp_gs_feature = nn.Parameter(
        #     torch.zeros([self.explicit_samples, self.embed_dims]),
        #     requires_grad=feat_grad,
        # )
        # self.exp_gs_means = self.init_means()
        # self.imp_gs_means = self.init_means()

        imp_scale = torch.ones([self.implicit_samples, 2], dtype=torch.float)
        imp_scale[:, 0] = imp_scale[:, 0] * ego_scale[0]
        imp_scale[:, 1] = imp_scale[:, 1] * ego_scale[1]
        self.imp_gs_scales = nn.Parameter(imp_scale, requires_grad=False)
        self.exp_gs_init = nn.Embedding(self.explicit_samples, self.embed_dims)
        # self.imp_gs_init = nn.Embedding(self.implicit_samples, self.embed_dims)
        # self.imp_gs_mlp = MLP(self.embed_dims, self.embed_dims * 4, self.embed_dims)
    
    # def init_weights(self):
    #     if self.exp_gs_feature.requires_grad:
    #         torch.nn.init.xavier_uniform_(self.exp_gs_feature.data, gain=1)

    def init_exp_gaussians(self):
        x_front = (
            torch.rand(self.random_samples_front, 1, dtype=torch.float) * 0.5 + 0.5
        )
        y_front = torch.rand(self.random_samples_front, 1, dtype=torch.float)
        xy_front = torch.cat([x_front, y_front], dim=-1)
        x_back = torch.rand(self.random_samples_back, 1, dtype=torch.float) * 0.5
        y_back = torch.rand(self.random_samples_back, 1, dtype=torch.float) * 0.5 + 0.25
        xy_back = torch.cat([x_back, y_back], dim=-1)
        xy = torch.cat([xy_front, xy_back], dim=0)

        xy = cartesian(xy, self.pc_range, use_sigmoid=False)

        scale = torch.ones(self.explicit_samples, 2, dtype=torch.float) * 0.5
        scale = (
            self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        )

        heading = torch.zeros(self.explicit_samples, 1, dtype=torch.float)

        semantic = torch.randn(self.explicit_samples, self.semantic_dim, dtype=torch.float)
        explicit_gs_param = torch.cat([xy, scale, heading, semantic], dim=-1)
        explicit_gs_param = nn.Parameter(explicit_gs_param, False)

        return explicit_gs_param

    def init_imp_gaussians(self, ego_feature):
        imp_gs_feature = ego_feature[:, None] + self.imp_gs_init.weight[None]
        xy_heading = self.imp_gs_mlp(imp_gs_feature)
        xy, heading = xy_heading[..., :2], xy_heading[..., 2:]
        sin = torch.sin(heading)
        cos = torch.cos(heading)
        rots = torch.cat([cos, sin], dim=-1)
        scale = torch.ones_like(rots)
        scale[..., 0] = scale[..., 0] * self.ego_scale[0]
        scale[..., 1] = scale[..., 1] * self.ego_scale[1]

        imp_gs_means, imp_gs_scales, imp_gs_rotations = xy, nn.Parameter(scale, requires_grad=False), rots
        return imp_gs_means, imp_gs_scales, imp_gs_rotations, imp_gs_feature

    # def init_means(self):
    #     x_front = (
    #         torch.rand(self.random_samples_front, 1, dtype=torch.float) * 0.5 + 0.5 # [0.5, 1.0]
    #     )
    #     y_front = torch.rand(self.random_samples_front, 1, dtype=torch.float) #[0.0, 1.0]
    #     xy_front = torch.cat([x_front, y_front], dim=-1)
    #     x_back = torch.rand(self.random_samples_back, 1, dtype=torch.float) * 0.5 # [0.0, 0.5]
    #     y_back = torch.rand(self.random_samples_back, 1, dtype=torch.float) * 0.5 + 0.25 #[0.25, 0.75]
    #     xy_back = torch.cat([x_back, y_back], dim=-1)
    #     xy = torch.cat([xy_front, xy_back], dim=0)

    #     xy = cartesian(xy, self.pc_range, use_sigmoid=False)
    #     means = nn.Parameter(xy, requires_grad=False)
    #     return means
    
    def forward(self, batch_size, ego_feature, **kwargs):
        exp_gs_param = torch.tile(self.exp_gs_param[None], (batch_size, 1, 1))
        exp_gs_feature = self.exp_gs_init.weight[None].repeat(batch_size, 1, 1)
        
        exp_gaussian = GaussianPrediction(
            means=exp_gs_param[..., :2],
            scales=exp_gs_param[..., 2:4],
            headings=exp_gs_param[..., 4:5],
            semantics=exp_gs_param[5:],
            rotations=torch.cat(
                [torch.cos(exp_gs_param[..., 4:5]), 
                 torch.sin(exp_gs_param[..., 4:5])], 
                 dim=-1
            ),
            features=exp_gs_feature,
            exp_samples=self.explicit_samples
        )

        imp_feature = ego_feature[:, None] + self.exp_gs_init.weight[None]
        imp_scale = self.imp_gs_scales.repeat(batch_size, 1, 1)
        imp_gaussian = GaussianPrediction(
            features=imp_feature,
            scales=imp_scale,
            exp_samples=0
        )

        return exp_gaussian, imp_gaussian
