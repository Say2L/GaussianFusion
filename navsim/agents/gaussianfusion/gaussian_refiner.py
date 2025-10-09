import torch.nn as nn
from mmengine.registry import MODELS
from .encoder.gs_exp_refiner import GaussianExpRefiner
from .encoder.gs_imp_refiner import GaussianImpRefiner
from .encoder.refine_module_v2 import SparseGaussianRefinementModuleV2
from .encoder.transformer_decoder import MLP
from .encoder.utils import GaussianPrediction
from .gaussianfusion_config import GaussianFusionConfig

class GaussianRefiner(nn.Module):
    def __init__(self, config: GaussianFusionConfig, refiner_type='exp'):
        super().__init__()

        self.poses_num = config.num_poses
        self.refiner_type = refiner_type

        if refiner_type == 'exp':
            self.refiner = GaussianExpRefiner(config)
        elif refiner_type == 'imp':
            self.refiner = GaussianImpRefiner(config)

        # self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, 3)
        self.gs_decoder = SparseGaussianRefinementModuleV2(config, refine_type=refiner_type)

    def forward(self, image_feature, pts_feature, gaussian, exp_gaussian=None, gaussian_list=None):
        # Avoid mutable default arguments; initialize per-call list when not provided
        if gaussian_list is None:
            gaussian_list = []
        
        if self.refiner_type == 'imp':
            gaussian = self.gs_decoder(gaussian)

        gaussian = self.refiner(image_feature, pts_feature, gaussian, exp_gaussian)
        
        if self.refiner_type == 'exp':
            gaussian = self.gs_decoder(gaussian)

        gaussian_list.append(gaussian)

        return gaussian_list