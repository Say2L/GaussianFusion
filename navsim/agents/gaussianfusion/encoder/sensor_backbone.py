import timm
import torch
import torch.nn as nn
from mmdet.models.necks.fpn import FPN
from mmdet.models.backbones import ResNet
from .grid_mask import GridMask
from ..gaussianfusion_config import GaussianFusionConfig

class SensorBackbone(nn.Module):
    def __init__(self, config: GaussianFusionConfig, num_feature_levels: int = 3):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        num_cams = 3

        self.num_cams = num_cams
        self.use_cams_embeds = True
        _dim_ = self.embed_dims

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        self.img_backbone = timm.create_model("resnet34", pretrained=True, features_only=True )
        
        self.num_outs=num_feature_levels

        self.img_neck=FPN(
            in_channels=[64,128,256,512][-self.num_outs:],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=self.num_outs,
            relu_before_extra_convs=True
        )
        self.level_embeds = nn.Parameter(torch.randn(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn([self.num_cams, self.embed_dims]))

        self.lidar_backbone = ResNet(**config.lidar_backbone)

        self.lidar_neck = FPN(
            in_channels=[64,128,256,512][-3:],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=3,
            relu_before_extra_convs=True
        )
    

    def forward(self, img, pts, len_queue=None, **kwargs):
        """
        Forward pass for processing images and point clouds.
        
        Args:
            img: Input images tensor (B, N, C, H, W).
            pts: Input point clouds tensor.
            len_queue: Optional queue length for reshaping.
            **kwargs: Additional arguments.
        
        Returns:
            Tuple of processed features and metadata.
        """
        img_feats, img_feat_flatten, img_spatial_shapes, img_level_start_index = self._process_images(img, len_queue)
        pts_feats, pts_feat_flatten, pts_spatial_shapes, pts_level_start_index = self._process_points(pts)
        
        return img_feats, pts_feats, img_feat_flatten, img_spatial_shapes, img_level_start_index, kwargs, pts_feat_flatten, pts_spatial_shapes, pts_level_start_index
    
    def _process_images(self, img, len_queue):
        """Process and flatten image features."""
        batch_size, num_cams, channels, height, width = img.size()
        img = img.reshape(batch_size * num_cams, channels, height, width)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats[-self.num_outs:])
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            bn, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(batch_size / len_queue), len_queue, int(bn / batch_size), c, h, w))
            else:
                img_feats_reshaped.append(img_feat.view(batch_size, int(bn / batch_size), c, h, w))
        
        return self._flatten_img_features(img_feats_reshaped)
    
    def _process_points(self, pts):
        """Process and flatten point cloud features."""
        pts_feats = self.lidar_backbone(pts)
        pts_feats = self.lidar_neck(pts_feats)
        
        pts_feats_reshaped = [pts_feat.unsqueeze(1) for pts_feat in pts_feats]
        
        return self._flatten_pts_features(pts_feats_reshaped)
    
    def _flatten_img_features(self, img_feats_reshaped):
        """Flatten and embed image features."""
        img_feat_flatten = []
        img_spatial_shapes = []
        for level, feat in enumerate(img_feats_reshaped):
            b, n, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cam, bs, h*w, c)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, level:level + 1, :].to(feat.dtype)
            spatial_shape_tensor = torch.as_tensor([spatial_shape], dtype=torch.long, device=feat.device)
            feat = feat.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
            img_spatial_shapes.append(spatial_shape_tensor)
            img_feat_flatten.append(feat)
        
        img_spatial_shapes = torch.cat(img_spatial_shapes, dim=0)
        img_feat_flatten = torch.cat(img_feat_flatten, dim=1)
        img_level_start_index = torch.cat((img_spatial_shapes.new_zeros((1,)), img_spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        return img_feats_reshaped, img_feat_flatten, img_spatial_shapes, img_level_start_index
    
    def _flatten_pts_features(self, pts_feats_reshaped):
        """Flatten point cloud features."""
        pts_feat_flatten = []
        pts_spatial_shapes = []
        for pts_feat in pts_feats_reshaped:
            b, n, c, h, w = pts_feat.size()
            spatial_shape = (h, w)
            pts_feat = pts_feat.flatten(3).permute(1, 0, 3, 2)  # (1, B, H*W, C)
            spatial_shape_tensor = torch.as_tensor([spatial_shape], dtype=torch.long, device=pts_feat.device)
            pts_feat = pts_feat.permute(0, 2, 1, 3)  # (1, H*W, bs, embed_dims)
            pts_spatial_shapes.append(spatial_shape_tensor)
            pts_feat_flatten.append(pts_feat)
        
        pts_spatial_shapes = torch.cat(pts_spatial_shapes, dim=0)
        pts_feat_flatten = torch.cat(pts_feat_flatten, dim=1)
        pts_level_start_index = torch.cat((pts_spatial_shapes.new_zeros((1,)), pts_spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        return pts_feats_reshaped, pts_feat_flatten, pts_spatial_shapes, pts_level_start_index