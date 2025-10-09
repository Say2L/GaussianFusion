from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from mmengine.registry import MODELS
from .score_module.scorer import Scorer
from .gaussian_refiner import GaussianRefiner
from .encoder.sensor_backbone import SensorBackbone
from .gaussianfusion_config import GaussianFusionConfig

class GaussianFusionModel(nn.Module):
    def __init__(self, config: GaussianFusionConfig):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.exp_only = config.exp_only
        self._backbone = SensorBackbone(config)

        self._gaussian_init = MODELS.build(config.gaussian_init)

        self._ego_encoding = nn.Linear(11, config.tf_d_model)

        self.refiner_num = config.refiner_num
        
        exp_refiner = GaussianRefiner(config, refiner_type='exp')
        self._exp_gaussian_refiners = nn.ModuleList([exp_refiner for _ in range(self.refiner_num)])

        if not self.exp_only:
            imp_refiner = GaussianRefiner(config, refiner_type='imp')
            self._imp_gaussian_refiners = nn.ModuleList([imp_refiner for _ in range(self.refiner_num)])

        self._gaussian_decoder = MODELS.build(config.gaussian_decoder)

        self._scorer = Scorer(config)

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the GaussianFusionModel.
        
        Args:
            features: Dictionary of input features.
            targets: Dictionary of target data.
        
        Returns:
            Dictionary of output predictions and scores.
        """
        batch_size, multi_feature, exp_gaussian, imp_gaussian = self._process_inputs(features)
        exp_gaussian_list, imp_gaussian_list = self._refine_gaussians(multi_feature, exp_gaussian, imp_gaussian)
        output = {}
        output.update(self._decode_gaussians(exp_gaussian_list, targets))
        if not self.exp_only:
            output.update(self._score_proposals(imp_gaussian_list))
            output.update(self._compute_trajectory(output["proposals"], output["pred_logit"], batch_size))
        return output
    
    def _process_inputs(self, features):
        """Process input features and initialize Gaussian."""
        ego_status = features["ego_status"][:, -1]
        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"].transpose(-2, -1).contiguous()
        batch_size = ego_status.shape[0]
        
        ego_feature = self._ego_encoding(ego_status)

        multi_feature = self._backbone(camera_feature, lidar_feature, img_metas=features)
        
        exp_gaussian, imp_gaussian = self._gaussian_init(batch_size, ego_feature)

        return batch_size, multi_feature, exp_gaussian, imp_gaussian

    def _refine_gaussians(self, multi_feature, exp_gaussian, imp_gaussian):
        """Refine Gaussians"""

        exp_gaussian_list = []
        imp_gaussian_list = []

        for i, refiner in enumerate(self._exp_gaussian_refiners):
            exp_gaussian_list = refiner(multi_feature[2:6], multi_feature[6:], exp_gaussian, gaussian_list=exp_gaussian_list)
            exp_gaussian = exp_gaussian_list[-1]

        if not self.exp_only:
            for i, refiner in enumerate(self._imp_gaussian_refiners):
                imp_gaussian_list = refiner(multi_feature[2:6], multi_feature[6:], imp_gaussian, exp_gaussian, gaussian_list=imp_gaussian_list)
                imp_gaussian = imp_gaussian_list[-1]

        return exp_gaussian_list, imp_gaussian_list
    
    def _decode_gaussians(self, gaussian_list, targets):
        """Decode Gaussian features into occupancy maps."""
        occ_xy = targets["bev_occ_xy"]

        gaussian_dict = self._gaussian_decoder(
            gaussian_list,
            occ_xy=occ_xy if targets is not None else None,
            occ_label=targets["bev_occ_label"] if targets is not None else None
        )
        return {
            "bev_map_pred": gaussian_dict["pred_occ"],
            "bev_map_gt": gaussian_dict["sampled_label"],
        }
    

    def _score_proposals(self, gaussian_list):
        """Score proposals and compute agent states."""
        def get_proposal_list(gaussian_list):
            proposal_list = []
            for gaussian in gaussian_list:
                proposal_xy = gaussian.means  # b, g, 2
                # proposal_rot = gaussian.rotations[:, -imp_samples:].contiguous()  # b, g, 2
                # proposal_heading = torch.atan2(proposal_rot[..., 0], proposal_rot[..., 1]).unsqueeze(-1)  # b, g, 1
                proposal_heading = gaussian.headings  # b, g, 1
                proposals = torch.cat([proposal_xy, proposal_heading], dim=-1)  # b, g, 3
                proposals = proposals.view(proposals.shape[0], -1, self.poses_num, 3)  # b, n, p, 3
                proposal_list.append(proposals)
            return proposal_list
        
        proposal_list = get_proposal_list(gaussian_list)
        proposals = proposal_list[-1]
        proposal_feature = gaussian_list[-2].features  # b, g, c

        pred_logit = self._scorer(proposals, proposal_feature)
        return {
            "proposals": proposals,
            "proposal_list": proposal_list,
            "pred_logit": pred_logit,
        }
    
    def _compute_trajectory(self, proposals, pred_logit, batch_size):
        """Compute the final trajectory and PDM score."""
        pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
        
        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]
        return {
            "trajectory": trajectory,
            "pdm_score": pdm_score,
        }