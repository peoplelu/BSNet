from typing import Optional

import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from scipy.optimize import linear_sum_assignment
import numpy as np

def is_within_bb_torch(points, bb_min, bb_max):
    return torch.all(points >= bb_min, dim=-1) & torch.all(points <= bb_max, dim=-1)


@torch.jit.script
def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    # if (torch.isinf(loss)  | torch.isnan(loss)).sum()!=0:
    #     print('1')
    # if (torch.isinf(loss / N)  | torch.isnan(loss / N)).sum()!=0:
    #     print('1')
    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()

    if masks is not None:
        inputs = inputs * masks[None, :]
        targets = targets * masks[None, :]

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def dice_loss_multi_calsses(
    input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5, weight: Optional[float] = None
) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from
    https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon
    )

    loss = 1.0 - per_channel_dice

    return loss.mean()

def sample_topk_per_gt(pr_inds, gt_inds, cost, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # find topk matches for each gt
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = cost[:,gt_inds2].topk(k, largest=False, dim=0)
    map1 = torch.zeros(gt_inds2.max()+1).cuda().float()
    map1 = map1[None,:].scatter(-1,gt_inds2[None,:],torch.arange(gt_inds2.shape[0]).cuda().float()[None,:]).squeeze(0)

    scores_1 = cost[pr_inds,gt_inds]
    scores_2 = scores[-1][map1[gt_inds].long()]
    valid = scores_1 < scores_2
    pr_inds3 = pr_inds[valid]
    gt_inds3 = gt_inds[valid]
    # gt_inds2 = gt_inds2[:,None].repeat(1, k)
    
    # # filter to as many matches that gt has
    # pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2.transpose(0,1))])
    # gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])

    return pr_inds3, gt_inds3

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.thresholds = [0.4, 0.6]
        self.thresholds.insert(0, -float("inf"))
        self.thresholds.append(float("inf"))
        self.labels = [0, -1, 1]
        self.register_buffer("cost_weight", torch.tensor(cost_weight))
        self.k = 5

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts, idx=0):
        """
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        """
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)  # (n_q, 19)
            tgt_idx = inst.gt_labels  # (num_inst,)
            cost_class = -pred_label[:, tgt_idx]  # (n_q, num_inst)

            tgt_mask = inst.gt_spmasks  # (num_inst, N)

            cost_mask = batch_sigmoid_bce_loss(pred_mask.type(torch.double), tgt_mask.type(torch.double))
            cost_dice = batch_dice_loss(pred_mask.type(torch.double), tgt_mask.type(torch.double))
            binarized_inputs = (pred_mask >= 0.5).float()
            targets1 = (tgt_mask > 0.5).float()
            inter = binarized_inputs@targets1.transpose(0,1)
            point_num = binarized_inputs.sum(1)[:,None] + targets1.sum(1)[None,:]
            ious = inter / (point_num - inter + 1e-6)
            
            C = self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice
            # C = C.cpu()
            if idx >= 0:
                _, matches = C.min(dim=1)
                ious_min = ious[torch.arange(C.shape[0]).cuda(),matches]
                match_labels = matches.new_full(matches.size(), 0, dtype=torch.int8)
                # self.thresholds = [0.3, 0.6]
                for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
                    low_high = (ious_min >= low) & (ious_min < high)
                    match_labels[low_high] = l
                #self.set_low_quality_matches_(match_labels, C)
                #matched_labels = self._subsample_labels(matched_labels)
                all_pr_inds = torch.arange(ious.shape[0]).cuda()
                pos_pr_inds = all_pr_inds[match_labels == 1]
                pos_gt_inds = matches[pos_pr_inds]
                #pos_cost = C[pos_pr_inds , pos_gt_inds]
                pos_pr_inds, pos_gt_inds = sample_topk_per_gt(pos_pr_inds, pos_gt_inds, C, self.k)
                # 剩余queries与剩余gt做匈牙利
                sum_idx = torch.arange(ious.shape[1]).cuda()
                sum_idx1 = torch.arange(ious.shape[0]).cuda()
                extra_idx = sum_idx[~torch.isin(sum_idx, pos_gt_inds)]
                extra_idx1 = sum_idx1[~torch.isin(sum_idx1, pos_pr_inds)]
                C2 = C[extra_idx1][:,extra_idx].cpu().numpy()
                pos_pr_extra,pos_gt_extra = linear_sum_assignment(C2)
                pos_pr_extra = extra_idx1[pos_pr_extra]
                pos_gt_extra = extra_idx[pos_gt_extra]
                pos_pr_inds = torch.cat([pos_pr_inds,pos_pr_extra],dim=-1)
                pos_gt_inds = torch.cat([pos_gt_inds,pos_gt_extra],dim=-1)
                #pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
                indices.append((pos_pr_inds.cpu(), pos_gt_inds.cpu()))
            else:
                C = C.cpu()
                indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):
    def __init__(
        self,
        ignore_label=-100,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 0.5],
        cost_weight=[1.0, 1.0, 1.0],
        non_object_weight=0.1,
        num_class=18,
        uncertainty_weight=1,
    ):
        super().__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer("class_weight", class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer("loss_weight", loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label

        self.uncertainty_weight = uncertainty_weight

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_inst_info(self, batched_gt_instance, coords, batch_offsets):
        for i, gt_inst in enumerate(batched_gt_instance):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            coord = coords[start_id:end_id]  # (N, 3)
            inst_idx, point_idx = torch.nonzero(gt_inst["gt_masks"], as_tuple=True)
            inst_point = coord[point_idx]
            gt_inst["gt_center"] = torch_scatter.segment_coo(inst_point, inst_idx.cuda(), reduce="mean")

    def get_layer_loss(self, layer, aux_outputs, insts, sp_coords, sp_rgb_feats, sp_prob_labels, batch_offsets):
        loss_out = {}
        pred_labels = aux_outputs["labels"]
        pred_scores = aux_outputs["scores"]
        pred_masks = aux_outputs["masks"]
        indices = self.matcher(pred_labels, pred_masks, insts, idx = layer)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out["cls_loss"] = class_loss.item()

        # # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)

        for b, (mask, score, inst, (idx_q, idx_gt)) in enumerate(zip(pred_masks, pred_scores, insts, indices)):
            # for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue

            b_s, b_e = batch_offsets[b], batch_offsets[b + 1]
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)

            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)

            sp_prob_labels_b = sp_prob_labels[b_s:b_e]

            mask_bce_loss_ = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), reduce="none")
            mask_bce_loss_ = (mask_bce_loss_ * sp_prob_labels_b[None, :]).sum(dim=1) / (sp_prob_labels_b.sum())
            mask_bce_loss_ = mask_bce_loss_.mean()
            mask_bce_loss = mask_bce_loss + mask_bce_loss_

            mask_dice_loss = mask_dice_loss + dice_loss(pred_mask, tgt_mask.float(), masks=sp_prob_labels_b)


        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        loss_out["score_loss"] = score_loss.item()
        loss_out["mask_bce_loss"] = mask_bce_loss.item()
        loss_out["mask_dice_loss"] = mask_dice_loss.item()


        loss = (
            self.loss_weight[0] * class_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

        loss_out = {f"layer_{layer}_" + k: v for k, v in loss_out.items()}
        return loss, loss_out



    def forward(self, pred, insts):
        """
        pred_masks: List[Tensor (n, M)]
        pred_labels: (B, n, 19)
        pred_scores: (B, n, 1) or [(B, n, 1)]
        insts: List[Instance3D]
        """
        loss_out = {}
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        pred_masks = pred["masks"]

        sp_coords = pred["sp_coords"]
        sp_rgb_feats = pred["sp_rgb_feats"]
        batch_offsets = pred["batch_offsets"]
        sp_prob_labels = pred["sp_prob_labels"]

        # match
        indices = self.matcher(pred_labels, pred_masks, insts, idx = 6)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out["cls_loss"] = class_loss.item()

        # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for b, (mask, score, inst, (idx_q, idx_gt)) in enumerate(zip(pred_masks, pred_scores, insts, indices)):
            if len(inst) == 0:
                continue

            b_s, b_e = batch_offsets[b], batch_offsets[b + 1]

            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)


            sp_prob_labels_b = sp_prob_labels[b_s:b_e]

            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)

            mask_bce_loss_ = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), reduce="none")
            mask_bce_loss_ = (mask_bce_loss_ * sp_prob_labels_b[None, :]).sum(dim=1) / (sp_prob_labels_b.sum())
            mask_bce_loss_ = mask_bce_loss_.mean()
            mask_bce_loss = mask_bce_loss + mask_bce_loss_

    
            mask_dice_loss = mask_dice_loss + dice_loss(pred_mask, tgt_mask.float(), masks=sp_prob_labels_b)


        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)

        loss_out["score_loss"] = score_loss.item()
        loss_out["mask_bce_loss"] = mask_bce_loss.item()
        loss_out["mask_dice_loss"] = mask_dice_loss.item()


        loss = (
            self.loss_weight[0] * class_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )



        if "aux_outputs" in pred:
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss_i, loss_out_i = self.get_layer_loss(
                    i, aux_outputs, insts, sp_coords, sp_rgb_feats, sp_prob_labels, batch_offsets
                )
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out["loss"] = loss.item()

        return loss, loss_out
