import torch
import torch.nn as nn
from ..encoder.transformer_decoder import MLP
from ..gaussianfusion_config import GaussianFusionConfig

class Scorer(nn.Module):
    def __init__(self, config: GaussianFusionConfig):
        super().__init__()

        self.b2d=config.b2d

        self.proposal_num = config.proposal_num
        self.score_num = config.score_num

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

    def forward(self, proposals, bev_feature):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)

        return pred_logit
