from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from navsim.agents.gaussianfusion.gaussianfusion_config import GaussianFusionConfig
from navsim.agents.gaussianfusion.modules.lovasz_softmax import lovasz_softmax

def _occ_focal_loss(pred_occ, sampled_label, occ_mask=None, eps=1e-4):
    one_hot_labels = F.one_hot(sampled_label.long(), num_classes=pred_occ.shape[1])
    one_hot_labels = one_hot_labels.permute(0, 2, 1).float()  # B, C, N
    pos_inds = one_hot_labels.eq(1).float()
    neg_inds = one_hot_labels.lt(1).float()

    pred_occ = torch.clamp(pred_occ.sigmoid_(), min=eps, max=1 - eps)
    pos_loss = torch.log(pred_occ + eps) * torch.pow(1 - pred_occ, 2) * pos_inds
    neg_loss = torch.log(1 - pred_occ + eps) * torch.pow(pred_occ, 2) * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    loss = 0
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _bev_occ_loss(pred_occ, sampled_label, occ_mask=None, lovasz_loss_weight=0.1, prev_weight=0.1):

    tot_loss = 0.0

    if occ_mask is not None:
        occ_mask = occ_mask.flatten(1)
        sampled_label = sampled_label[occ_mask][None]

    for semantics in pred_occ:
        if occ_mask is not None:
            semantics = semantics.transpose(1, 2)[occ_mask][None].transpose(
                1, 2
            )  # 1, c, n
        loss_dict = {}

        # semantics = semantics.transpose(1, 2)
        loss_dict["loss_voxel_ce"] = CE_wo_softmax(
            semantics,
            sampled_label.long(),
            ignore_index=255,
        )

        lovasz_input = semantics

        loss_dict["loss_voxel_lovasz"] = lovasz_loss_weight * lovasz_softmax(
            lovasz_input.transpose(1, 2).flatten(0, 1),
            sampled_label.flatten(),
            ignore=0,
        )

        loss = 0.0
        for k, v in loss_dict.items():
            loss = loss + v
        tot_loss = tot_loss * prev_weight + loss
    return tot_loss


def CE_wo_softmax(pred, target, class_weights=None, ignore_index=255):
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index)
    return loss


def _bev_ele_loss(pred_ele, gt_ele, ele_mask=None):
    gt_ele = gt_ele.flatten(1)
    if ele_mask is not None:
        ele_mask = ele_mask.flatten(1)
        gt_ele = gt_ele[ele_mask]
        pred_ele = pred_ele[ele_mask]

    ele_loss = F.l1_loss(pred_ele, gt_ele)

    return ele_loss
