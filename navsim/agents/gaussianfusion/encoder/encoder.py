from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import torch.nn as nn
from .utils import safe_sigmoid

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, embed_dims, pc_range=None, num_points_in_pillar=4, num_learnable_pts=5, lidar_height=0, 
                 half_width=0, half_length=0,rear_axle_to_center=0, return_intermediate=False,
                 **kwargs):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.embed_dims = embed_dims

        self.num_learnable_pts = num_learnable_pts
        if num_learnable_pts > 0:
            self.tj_learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 2)
            self.gs_learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 2)
        self.half_length = half_length
        self.half_width = half_width
        self.rear_axle_to_center = rear_axle_to_center
        self.lidar_height=lidar_height

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cpu', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z#4,8,8
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W#4,8,8
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H#4,8,8
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 2, 1).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)#1,4,64,3
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_x,ref_y = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        # lidar2img: (B, num_cam, 4, 4)
        lidar2img = img_metas['lidar2img']
        num_cam = lidar2img.size(1)  # Number of cameras
        D, B, num_query = reference_points.size()[:3]  # D: depth, B: batch size, num_query: number of queries

        # reference_points: (D, B, 1, num_query, 4) -> (D, B, num_cam, num_query, 4, 1)
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        # lidar2img: (1, B, num_cam, 1, 4, 4) -> (D, B, num_cam, num_query, 4, 4)
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # reference_points_cam: (D, B, num_cam, num_query, 4, 1) -> (D, B, num_cam, num_query, 4)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)

        eps = 1e-5

        # bev_mask: (D, B, num_cam, num_query, 1)
        bev_mask = (reference_points_cam[..., 2:3] > eps)

        # reference_points_cam: (D, B, num_cam, num_query, 2)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # Normalize reference_points_cam
        reference_points_cam[..., 0] /= img_metas['img_shape'][0][0][1]  # Normalize x-coordinates
        reference_points_cam[..., 1] /= img_metas['img_shape'][0][0][0]  # Normalize y-coordinates

        # Update bev_mask: (D, B, num_cam, num_query, 1)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        # Handle NaN values in bev_mask
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # reference_points_cam: (D, B, num_cam, num_query, 2) -> (num_cam, B, num_query, D, 2)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)

        # bev_mask: (D, B, num_cam, num_query, 1) -> (num_cam, B, num_query, D)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def point_sampling_pts(self, reference_points):

        num_cam = 1
        D, B, num_query = reference_points.size()[:3]

        reference_points = reference_points.view(
            D, B, 1, num_query, 2).repeat(1, 1, num_cam, 1, 1)

        reference_points_pts = reference_points[..., :2]
        reference_points_pts[..., 0] = (reference_points_pts[..., 0] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]
        )
        reference_points_pts[..., 1] = (reference_points_pts[..., 1] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]
        )

        bev_mask = (
            (reference_points_pts[..., 0] >= 0)
            & (reference_points_pts[..., 1] >= 0)
            & (reference_points_pts[..., 0] < 1)
            & (reference_points_pts[..., 1] < 1)
        )

        reference_points_pts = reference_points_pts.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0)

        return reference_points_pts, bev_mask

    def compute_corners(self, instance_feature, boxes):
        B, P, C = instance_feature.size()
        # Calculate half dimensions
        x = boxes[..., 0]        # x-coordinate of the center
        y = boxes[..., 1]        # y-coordinate of the center
        headings= boxes[..., 2]

        half_width = torch.zeros_like(x)+self.half_width
        half_length = torch.zeros_like(x)+self.half_length

        cos_yaw = torch.cos(headings)[...,None]
        sin_yaw = torch.sin(headings)[...,None]

        x=x[...,None]+self.rear_axle_to_center * cos_yaw
        y=y[...,None]+self.rear_axle_to_center * sin_yaw

        # Compute the four corners
        corners_x = torch.stack([torch.zeros_like(half_length), half_length, half_length, -half_length, -half_length],dim=-1)
        corners_y = torch.stack([torch.zeros_like(half_length), half_width, -half_width, -half_width, half_width],dim=-1)

        
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(
                    self.tj_learnable_fc(instance_feature).reshape(
                        B, P, self.num_learnable_pts, 2
                    )
                )
                - 0.5
            )
            corners_x = torch.cat([corners_x, learnable_scale[..., 0]], dim=-1)
            corners_y = torch.cat([corners_y, learnable_scale[..., 1]], dim=-1)

        # Rotate corners by yaw
        rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
        rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

        # Translate corners to the center of the bounding box
        corners = torch.stack((rot_corners_x + x, rot_corners_y + y), dim=-1)

        return corners

    def compute_gaussian_corners(self, gs_feature, gs_xy, gs_scale, gs_rot, exp_samples=0):
        B, P, C = gs_feature.size()
        # Calculate half dimensions
        x = gs_xy[..., 0:1].detach()        # x-coordinate of the center
        y = gs_xy[..., 1:2].detach()        # y-coordinate of the center
        
        # headings= torch.arctan2(gs_rot[..., 0], gs_rot[..., 1])

        half_width = gs_scale[..., 1:2].detach() / 2
        half_length = gs_scale[..., 0:1].detach() / 2

        cos_yaw = gs_rot[..., 1:2].detach() #torch.cos(headings)[...,None]
        sin_yaw = gs_rot[..., 0:1].detach() #torch.sin(headings)[...,None]

        # Compute the four corners
        exp_corners_x = torch.cat([torch.zeros_like(x[:, :exp_samples]), torch.zeros_like(x[:, :exp_samples]), \
                                     torch.zeros_like(x[:, :exp_samples]), half_length[:, :exp_samples], \
                                     -half_length[:, :exp_samples]],dim=-1)
        exp_corners_y = torch.cat([torch.zeros_like(y[:, :exp_samples]), half_width[:, :exp_samples],\
                                     -half_width[:, :exp_samples], torch.zeros_like(y[:, :exp_samples]), \
                                     torch.zeros_like(y[:, :exp_samples])],dim=-1)

        imp_corners_x = torch.cat([torch.zeros_like(half_length[:, exp_samples:]), half_length[:, exp_samples:], \
                                     half_length[:, exp_samples:], -half_length[:, exp_samples:], \
                                     -half_length[:, exp_samples:]],dim=-1)
        imp_corners_y = torch.cat([torch.zeros_like(half_length[:, exp_samples:]), half_width[:, exp_samples:],\
                                     -half_width[:, exp_samples:], -half_width[:, exp_samples:], \
                                     half_width[:, exp_samples:]],dim=-1)

        corners_x = torch.cat([exp_corners_x, imp_corners_x], dim=1)
        corners_y = torch.cat([exp_corners_y, imp_corners_y], dim=1)

        if self.num_learnable_pts > 0 and gs_feature is not None:
            learnable_scale = (
                safe_sigmoid(
                    self.tj_learnable_fc(gs_feature).reshape(
                        B, P, self.num_learnable_pts, 2
                    )
                )
                - 0.5
            )
            corners_x = torch.cat([corners_x, learnable_scale[..., 0]], dim=-1)
            corners_y = torch.cat([corners_y, learnable_scale[..., 1]], dim=-1)

        # Rotate corners by yaw
        rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y # Note this transformation
        rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

        x[:, exp_samples:] = x[:, exp_samples:] + self.rear_axle_to_center * cos_yaw[:, exp_samples:]
        y[:, exp_samples:] = y[:, exp_samples:] + self.rear_axle_to_center * sin_yaw[:, exp_samples:]

        # Translate corners to the center of the bounding box
        corners = torch.stack((rot_corners_x + x, rot_corners_y + y), dim=-1)

        return corners

    @auto_fp16()
    def forward(self,
                query,
                feats_img,
                feats_pts,
                *args,
                gaussian=None,
                exp_gaussian=None,
                bev_pos=None,
                imp_pos=None,
                exp_pos=None,
                spatial_shapes_img=None,
                level_start_index_img=None,
                spatial_shapes_pts=None,
                level_start_index_pts=None,
                bev_h=None,
                bev_w=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        gs_xy = gaussian.means.detach()
        gs_scales = gaussian.scales
        gs_rots = gaussian.rotations
        intermediate = []

        bs = gs_xy.shape[0]
        len_bev = gs_xy.shape[1]

        with torch.autocast(device_type='cuda', enabled=False):  # Disable autocasting
            ref_2d = gs_xy.clone()
            ref_2d = (ref_2d[:, :, None, :2] + 32) / 64
            
            hybird_ref_2d = torch.cat([ref_2d, ref_2d])

            zs = torch.linspace(self.pc_range[2] - self.lidar_height, self.pc_range[5] - self.lidar_height, self.num_points_in_pillar, dtype=torch.float32,
                                device=ref_2d.device)

            zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)

            ref_pos = self.compute_gaussian_corners(
                gaussian.features, gs_xy, gs_scales, gs_rots, exp_samples=gaussian.exp_samples) # b,N,5,2
    
            zs = zs.repeat(1, 1, 5 + self.num_learnable_pts, 1)

            reference_points_cam = torch.cat([ref_pos.repeat(1, 1, self.num_points_in_pillar, 1), zs], dim=-1).permute(2, 0, 1, 3)
            reference_points_cam = torch.cat((reference_points_cam, torch.ones_like(reference_points_cam[..., :1])), -1)
            reference_points_pts = ref_pos.permute(2, 0, 1, 3)

            reference_points_img, bev_mask_img = self.point_sampling(
                reference_points_cam,  kwargs['img_metas'])
            reference_points_pts, bev_mask_pts = self.point_sampling_pts(
                reference_points_pts)

        # bev_pos = torch.cat([traj_pos, gs_pos], dim=1)  # bs, len_bev, embed_dim
        bev_query = query  # bs, len_bev, embed_dim

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                gaussian=exp_gaussian,
                feats_img=feats_img,
                feats_pts=feats_pts,
                *args,
                bev_pos=bev_pos,
                imp_pos=imp_pos,
                exp_pos=exp_pos,
                ref_2d=hybird_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes_img=spatial_shapes_img,
                level_start_index_img=level_start_index_img,
                spatial_shapes_pts=spatial_shapes_pts,
                level_start_index_pts=level_start_index_pts,
                reference_points_img=reference_points_img,
                reference_points_pts=reference_points_pts,
                mask_img=bev_mask_img,
                mask_pts=bev_mask_pts,
                **kwargs
                )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return bev_query


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 9
        # assert set(operation_order) == set(
        #     ['gs_attn', 'self_attn', 'norm', 'cross_attn_pts', 'cross_attn_img', 'ffn'])

    def forward(self,
                query,
                gaussian=None,
                feats_img=None,
                feats_pts=None,
                query_pos=None,
                key_pos=None,
                imp_pos=None,
                exp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                bev_pos=None,
                ref_2d=None,
                bev_h=None,
                bev_w=None,
                reference_points_img=None,
                mask_img=None,
                reference_points_pts=None,
                mask_pts=None,
                spatial_shapes_img=None,
                level_start_index_img=None,
                spatial_shapes_pts=None,
                level_start_index_pts=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    query_pos=bev_pos,
                    )
                # query = self.attentions[attn_index](
                #     query,
                #     prev_bev,
                #     prev_bev,
                #     identity if self.pre_norm else None,
                #     query_pos=bev_pos,
                #     key_pos=bev_pos,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=query_key_padding_mask,
                #     reference_points=ref_2d,
                #     spatial_shapes=torch.tensor(
                #         [[bev_h, bev_w]], device=query.device),
                #     level_start_index=torch.tensor([0], device=query.device),
                #     **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn_img':
                query = self.attentions[attn_index](
                    query,
                    feats_img,
                    feats_img,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points_img,
                    bev_mask=mask_img,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes_img,
                    level_start_index=level_start_index_img,
                    **kwargs)
                attn_index += 1
                identity = query

            # spaital cross attention
            elif layer == 'cross_attn_pts':
                query = self.attentions[attn_index](
                    query,
                    feats_pts,
                    feats_pts,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points_pts,
                    bev_mask=mask_pts,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes_pts,
                    level_start_index=level_start_index_pts,
                    **kwargs)
                attn_index += 1
                identity = query

            # Gaussian Cross Attention
            if layer == 'cross_attn_gs':
                query = self.attentions[attn_index](
                    query,
                    gaussian.features,
                    query_pos=imp_pos,
                    key_pos=exp_pos,
                    **kwargs)
                attn_index += 1
                identity = query
                
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
