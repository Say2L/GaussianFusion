import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from mmengine.model import BaseModule
from ..encoder.utils import get_rotation_matrix
from .libs.localagg_prob_fast import local_aggregate_prob_fast

@MODELS.register_module()
class GaussianDecoder(BaseModule):
    def __init__(
        self,
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type="nusc",
        empty_label=0,
        use_localaggprob=False,
        combine_geosem=False,
        **kwargs,
    ):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
           
        self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)

        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer(
                "empty_mean", torch.tensor(empty_args["mean"])[None, None, :]
            )
            self.register_buffer(
                "empty_scale", torch.tensor(empty_args["scale"])[None, None, :]
            )
            self.register_buffer(
                "empty_rot", torch.tensor([1.0, 0.0, 0.0, 0.0])[None, None, :]
            )
            self.register_buffer(
                "empty_sem", torch.zeros(self.num_classes)[None, None, :]
            )
            self.register_buffer("empty_opa", torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == "all":
            self.apply_loss_type = "all"
        elif "random" in apply_loss_type:
            self.apply_loss_type = "random"
            self.random_apply_loss_layers = int(apply_loss_type.split("_")[1])
        elif "fixed" in apply_loss_type:
            self.apply_loss_type = "fixed"
            self.fixed_apply_loss_layers = [
                int(item) for item in apply_loss_type.split("_")[1:]
            ]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer("zero_tensor", torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xy, gt_label, gt_mask=None):
        if gt_mask is None:
            width = gt_label.shape[1]
            gt_label = gt_label[:, width//2:, ...].flatten(1).contiguous()
            gt_xy = gt_xy[:, width//2:, ...].flatten(1, 2).contiguous()
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xy = gt_xy[gt_mask].reshape(1, -1, 2)
        z_zeros = torch.zeros_like(gt_xy[..., :1])
        gt_xyz = torch.cat([gt_xy, z_zeros], dim=-1).contiguous()
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means  # b, g, 2
        scales = gaussians.scales  # b, g, 2
        rotations = gaussians.rotations  # b, g, 2
        opacities = gaussians.semantics  # b, g, c
        origi_opa = gaussians.opacities  # b, g, 1
        means_3d = torch.cat([means, torch.zeros_like(means[..., :1])], dim=-1)
        scales_3d = torch.cat([scales, torch.ones_like(scales[..., :1])], dim=-1)
        if origi_opa is None:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        # opacities = opacities.softmax(dim=-1)
        bs, g, _ = means_3d.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means_3d.dtype, device=means_3d.device)
        S[..., 0, 0] = scales_3d[..., 0]
        S[..., 1, 1] = scales_3d[..., 1]
        S[..., 2, 2] = scales_3d[..., 2]
        R = get_rotation_matrix(rotations)  # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M).float()
        CovInv = Cov.cpu().inverse().cuda()  # b, g, 3, 3
        # epsilon = 1e-6
        # Cov_reg = Cov + epsilon * torch.eye(Cov.size(-1), device=Cov.device).expand_as(Cov)
        # CovInv = Cov_reg.inverse()
        return means_3d, origi_opa, opacities, scales_3d, CovInv

    def forward(self, gaussian_list, occ_xy, occ_label, **kwargs):
        num_decoder = len(gaussian_list)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(
                    num_decoder - 1, self.random_apply_loss_layers - 1, False
                )
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "fixed":
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        bin_logits = []
        density = []

        sampled_xyz, sampled_label = self._sampling(occ_xy, occ_label, None)

        for idx in apply_loss_layers:
            gaussians = gaussian_list[idx]

            means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(
                gaussians
            )
            bs, g = means.shape[:2]

            semantics = self.aggregator(
                sampled_xyz.float(),
                means.float(),
                origi_opa.reshape(bs, g).float(),
                opacities.float(),
                scales.float(),
                CovInv.float(),
            )  # 1, c, n
            if self.use_localaggprob:
                semantic = semantics[0]
                elevation = None
                if self.combine_geosem:
                    sem = semantic[..., 1:].softmax(-1) * semantics[1].unsqueeze(-1)  
                    empty_sem = (1 - semantics[1].unsqueeze(-1))
                    # restrain gaussian overlap in empty space and encourage gaussians move and overlap in fg space
                    geosem = torch.cat([empty_sem, sem], dim=-1)
                else:
                    geosem = semantic

                prediction.append(geosem.transpose(1, 2))
                bin_logits.append(semantics[1])
                density.append(semantics[2])
            else:
                prediction.append(semantics.transpose(1, 2))

        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)

        return {
            "pred_occ": prediction,
            "bin_logits": bin_logits,
            "density": density,
            "elevation": elevation,
            "sampled_label": sampled_label,
            "sampled_xyz": sampled_xyz,
            "final_occ": final_prediction,
            "gaussian": gaussian_list[-1],
        }