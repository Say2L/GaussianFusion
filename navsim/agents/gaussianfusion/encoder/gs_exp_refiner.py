import torch.nn as nn
import numpy as np
import torch
from .encoder import BEVFormerEncoder
from .utils import GaussianPrediction
from ..gaussianfusion_config import GaussianFusionConfig
from .blocks import linear_relu_ln, gen_sineembed_for_position
from .spatial_cross_attention import MSDeformableAttention3D
from .temporal_self_attention import TemporalSelfAttention
from .decoder import CustomMSDeformableAttention
from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding

class GaussianExpRefiner(nn.Module):
    def __init__(self, config: GaussianFusionConfig):
        super().__init__()
        self.proposal_num = config.proposal_num
        self.pose_num = config.num_poses
        self.pose_dim=3

        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        self.embed_dim = d_model

        num_points = config.num_points_in_pillar * 5

        _num_levels_ = 3
        num_cams = 3

        num_layers = config.num_bev_layers

        half_length = config.half_length
        half_width = config.half_width
        rear_axle_to_center = config.rear_axle_to_center
        lidar_height=config.lidar_height

        self.pc_range = np.array(config.point_cloud_range)
        
        self.positional_encoding = nn.Sequential(
                                            *linear_relu_ln(d_model, 1, 1, d_model),
                                            nn.Linear(d_model, d_model),
                                        )
        
        self.bev_decoder = BEVFormerEncoder(
            embed_dims=d_model,
            num_layers=num_layers,
            pc_range=self.pc_range,
            num_points_in_pillar=config.num_points_in_pillar,
            num_learnable_pts=config.num_learnable_pts,
            half_length=half_length,
            half_width=half_width,
            rear_axle_to_center=rear_axle_to_center,
            lidar_height=lidar_height,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=d_model,
                        num_heads=config.tf_num_head,
                        attn_drop=config.tf_dropout,
                        proj_drop=config.tf_dropout,
                        is_global=True,
                       ),
                    dict(
                        type='SpatialCrossAttention',
                        num_cams=1,
                        pc_range=config.point_cloud_range,
                        dropout=config.tf_dropout,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=d_model,
                            num_points=num_points,
                            num_levels=3,
                            use_offset=True),
                        embed_dims=d_model,
                    ),
                    dict(
                        type='SpatialCrossAttention',
                        num_cams=num_cams,
                        pc_range=config.point_cloud_range,
                        dropout=config.tf_dropout,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=d_model,
                            num_points=num_points,
                            num_levels=_num_levels_,
                            use_offset=True),
                        embed_dims=d_model,
                    ),
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=d_model,
                    feedforward_channels=config.tf_d_ffn,
                    num_fcs=2,
                    ffn_drop=config.tf_dropout,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=d_ffn,
                ffn_dropout=config.tf_dropout,
                operation_order=('self_attn', 'norm', 'cross_attn_pts', 'norm', 'cross_attn_img', 'norm', 'ffn', 'norm')),
        )

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    def forward(self, image_feature, pts_feature, gaussian, *args, **kwargs):
        
        xy = gaussian.means  # b, g, 2
        pos_embedding = gen_sineembed_for_position(xy, hidden_dim=self.embed_dim)
        pos_embedding = self.positional_encoding(pos_embedding)

        feat_flatten_img, spatial_shapes_img, level_start_index_img, kwargs = image_feature
        feat_flatten_pts, spatial_shapes_pts, level_start_index_pts = pts_feature

        gs_feature = self.bev_decoder(
            gaussian.features,
            feat_flatten_img,
            feat_flatten_pts,
            gaussian=gaussian,
            bev_pos=pos_embedding,
            spatial_shapes_img=spatial_shapes_img,
            level_start_index_img=level_start_index_img,
            spatial_shapes_pts=spatial_shapes_pts,
            level_start_index_pts=level_start_index_pts,
            bev_h=self.proposal_num,
            bev_w=self.pose_num,
            **kwargs
        )
    
        gaussian = GaussianPrediction(
            means=gaussian.means,
            scales=gaussian.scales,
            headings=gaussian.headings,
            rotations=gaussian.rotations,
            semantics=gaussian.semantics,
            features=gs_feature,
            original_means=gaussian.original_means,
            exp_samples=gaussian.exp_samples
        )

        return gaussian