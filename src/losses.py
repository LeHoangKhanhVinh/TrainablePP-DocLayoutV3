"""Full PP-DocLayoutV3 loss — pure-PyTorch port of PaddleDetection's ``DocLayoutV3Loss``.

This is a faithful reimplementation of
``PaddleDetection/ppdet/modeling/losses/detr_loss.py``:

* ``DETRLoss``         — class (focal / varifocal), L1, GIoU, mask, dice, aux losses.
* ``MaskDINOLoss``     — overrides the mask loss with uncertainty-based point
                         sampling, and adds CDN denoising losses.
* ``RelativeReadingOrderLoss`` — pairwise reading-order loss (GCE + locality).
* ``DocLayoutV3Loss``  — composes the above; order loss on the last decoder layer.

All of ``MaskDINOLoss`` + ``DETRLoss`` is folded into a single ``PPDocLayoutV3Loss``
class (method names kept identical to PaddleDetection for traceability).

The encoder-stage proposal is treated exactly as PaddleDetection's
``DocLayoutV3Head`` does: prepended as the first "layer" of the
``[1 + num_decoder_layers, B, Q, ...]`` stack, so it receives the same
class+bbox+giou+mask+dice auxiliary loss as the decoder layers.

No ``paddle`` imports — only PyTorch + HuggingFace helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.image_transforms import center_to_corners_format
from transformers.loss.loss_for_object_detection import generalized_box_iou, sigmoid_focal_loss

from .matcher import HungarianMatcher

__all__ = ["PPDocLayoutV3Loss", "RelativeReadingOrderLoss"]


# --------------------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------------------
def _weighted_bce_with_logits(logits, target, weight):
    """BCE-with-logits (``reduction='none'``) that is differentiable w.r.t. ``weight``.

    ``F.binary_cross_entropy_with_logits`` refuses a ``weight`` tensor with
    ``requires_grad=True``; PaddleDetection's varifocal weight *does* carry gradient
    (it contains ``sigmoid(pred_logits)``), so the BCE is spelled out here using the
    numerically-stable identity ``max(x,0) - x*t + log1p(exp(-|x|))``.
    """
    loss = logits.clamp(min=0) - logits * target + torch.log1p(torch.exp(-logits.abs()))
    return loss * weight


def varifocal_loss_with_logits(pred_logits, gt_score, label, normalizer=1.0, alpha=0.75, gamma=2.0):
    """Exact port of ``ppdet/modeling/transformers/utils.py::varifocal_loss_with_logits``."""
    pred_score = torch.sigmoid(pred_logits)
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = _weighted_bce_with_logits(pred_logits, gt_score, weight)
    return loss.mean(1).sum() / normalizer


def _dice_loss(inputs, targets, num_gts):
    """Port of ``DETRLoss._dice_loss`` (sigmoid applied internally)."""
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_gts


def _pairwise_box_iou(boxes_a, boxes_b, eps=1e-9):
    """Element-wise IoU of two ``[M, 4]`` xyxy tensors -> ``[M, 1]`` (port of ``bbox_iou``)."""
    if boxes_a.numel() == 0:
        return boxes_a.new_zeros((0, 1))
    px1, py1, px2, py2 = boxes_a.unbind(-1)
    gx1, gy1, gx2, gy2 = boxes_b.unbind(-1)
    x1 = torch.maximum(px1, gx1)
    y1 = torch.maximum(py1, gy1)
    x2 = torch.minimum(px2, gx2)
    y2 = torch.minimum(py2, gy2)
    overlap = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = ((px2 - px1) * (py2 - py1)).clamp(min=0)
    area2 = ((gx2 - gx1) * (gy2 - gy1)).clamp(min=0)
    union = area1 + area2 - overlap + eps
    return (overlap / union).unsqueeze(-1)


# --------------------------------------------------------------------------------------
# reading-order loss
# --------------------------------------------------------------------------------------
class RelativeReadingOrderLoss(nn.Module):
    """Pairwise relative reading-order loss — port of the PaddleDetection class.

    Supervises ``order_logit[i, j] > 0`` iff ground-truth ``rank[i] < rank[j]`` over the
    matched-query subset, with optional locality weighting, label smoothing and a
    Generalised-Cross-Entropy robust term. Elements with ``order < 0`` are ignored.
    """

    def __init__(
        self,
        use_upper_only: bool = True,
        k_local: int = 5,
        locality: bool = True,
        k_local_ratio: float = 0.3,
        w_gt: float = 2.0,
        label_smooth: float = 0.01,
        robust: str = "gce",
        q: float = 0.7,
    ) -> None:
        super().__init__()
        self.use_upper_only = use_upper_only
        self.k_local = k_local
        self.locality = locality
        self.k_local_ratio = k_local_ratio
        self.w_gt = w_gt
        self.label_smooth = label_smooth
        self.robust = robust
        self.q = q

    @staticmethod
    def _pair_mask(n: int, use_upper_only: bool, device) -> torch.Tensor:
        if use_upper_only:
            return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
        return ~torch.eye(n, dtype=torch.bool, device=device)

    @staticmethod
    def _valid_pair_mask(order: torch.Tensor) -> torch.Tensor:
        v = order >= 0
        return v.unsqueeze(0) & v.unsqueeze(1)

    def _gce_loss(self, logits, target, q):
        p = torch.sigmoid(logits)
        p_y = p * target + (1 - p) * (1 - target)
        return (1.0 - torch.clamp(p_y, 1e-6, 1.0).pow(q)) / q

    def forward(self, relative_logits, gt_read_order, match_indices):
        device = relative_logits.device
        total_loss_num = torch.zeros((), dtype=torch.float32, device=device)
        total_pairs = torch.zeros((), dtype=torch.float32, device=device)

        for i in range(len(gt_read_order)):
            pred_idx, gt_idx = match_indices[i]
            if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                continue
            n = pred_idx.shape[0]
            if n <= 1:
                continue

            logits = relative_logits[i][pred_idx][:, pred_idx]  # [n, n]
            order = gt_read_order[i][gt_idx].to(torch.float32)   # [n]

            valid_pair = self._valid_pair_mask(order)
            pair_mask = self._pair_mask(n, self.use_upper_only, device)
            base_mask = valid_pair & pair_mask

            pair_sum = base_mask.sum().to(torch.float32)
            if pair_sum.item() == 0:
                continue

            o1, o2 = order.unsqueeze(1), order.unsqueeze(0)
            target_full = (o1 < o2).to(torch.float32)
            order_dist = (o1 - o2).abs()
            k_local_i = min(max(self.k_local, int(n * self.k_local_ratio)), n - 1)

            weight = torch.ones(n, n, dtype=torch.float32, device=device)
            if self.locality:
                gt_local = (order_dist > 0) & (order_dist <= k_local_i)
                weight = weight + (self.w_gt - 1.0) * gt_local.to(torch.float32)

            z = logits[base_mask]
            t = target_full[base_mask]
            wm = weight[base_mask]
            wm = wm / (wm.mean() + 1e-6)

            if self.label_smooth > 0:
                eps = self.label_smooth
                t = t * (1 - eps) + 0.5 * eps

            if self.robust == "gce":
                per_pair_loss = self._gce_loss(z, t, self.q)
            else:
                per_pair_loss = F.binary_cross_entropy_with_logits(z, t, reduction="none")

            loss_main = (per_pair_loss * wm).mean()
            total_loss_num = total_loss_num + loss_main * pair_sum
            total_pairs = total_pairs + pair_sum

        return total_loss_num / (total_pairs + 1e-12)


# --------------------------------------------------------------------------------------
# main loss
# --------------------------------------------------------------------------------------
class PPDocLayoutV3Loss(nn.Module):
    """Faithful port of ``DocLayoutV3Loss`` (= ``MaskDINOLoss`` + reading order).

    ``forward`` consumes the HuggingFace ``PPDocLayoutV3`` model tensors directly and
    internally restacks them into PaddleDetection's ``[1 + num_decoder_layers, B, Q, ...]``
    layout (encoder proposal first), so the encoder stage receives the full auxiliary
    loss like every decoder layer.
    """

    DEFAULT_LOSS_CFG = {
        "loss_coeff": {"class": 4, "bbox": 5, "giou": 2, "mask": 5, "dice": 5, "order": 50},
        "use_focal_loss": True,
        "use_vfl": True,
        "vfl_iou_type": "mask",
        "aux_loss": True,
        "num_sample_points": 12544,
        "oversample_ratio": 3.0,
        "important_sample_ratio": 0.75,
    }
    DEFAULT_MATCHER_CFG = {
        "matcher_coeff": {"class": 4, "bbox": 5, "giou": 2, "mask": 5, "dice": 5},
        "use_focal_loss": True,
        "with_mask": True,
        "num_sample_points": 12544,
        "alpha": 0.25,
        "gamma": 2.0,
    }

    def __init__(
        self,
        config,
        loss_cfg: dict | None = None,
        matcher_cfg: dict | None = None,
        order_loss_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = config.num_labels

        cfg = dict(self.DEFAULT_LOSS_CFG)
        if loss_cfg:
            cfg.update(loss_cfg)
        self.loss_coeff = dict(cfg["loss_coeff"])
        self.use_focal_loss = bool(cfg["use_focal_loss"])
        self.use_vfl = bool(cfg["use_vfl"])
        self.vfl_iou_type = cfg["vfl_iou_type"]
        self.aux_loss = bool(cfg["aux_loss"])
        self.num_sample_points = int(cfg["num_sample_points"])
        self.oversample_ratio = float(cfg["oversample_ratio"])
        self.important_sample_ratio = float(cfg["important_sample_ratio"])

        self.num_oversample_points = int(self.num_sample_points * self.oversample_ratio)
        self.num_important_points = int(self.num_sample_points * self.important_sample_ratio)
        self.num_random_points = self.num_sample_points - self.num_important_points

        mcfg = dict(self.DEFAULT_MATCHER_CFG)
        if matcher_cfg:
            mcfg.update(matcher_cfg)
        mcfg.setdefault("num_sample_points", self.num_sample_points)
        self.matcher = HungarianMatcher(
            matcher_coeff=mcfg["matcher_coeff"],
            use_focal_loss=mcfg["use_focal_loss"],
            with_mask=mcfg["with_mask"],
            num_sample_points=mcfg["num_sample_points"],
            alpha=mcfg["alpha"],
            gamma=mcfg["gamma"],
        )

        if not self.use_focal_loss:
            # weight vector for the cross-entropy fallback (unused by the PP-DocLayoutV3 recipe).
            w = torch.full((self.num_classes + 1,), float(self.loss_coeff["class"]))
            w[-1] = float(self.loss_coeff.get("no_object", 0.1))
            self.register_buffer("class_weight", w, persistent=False)

        # Reading-order loss config: explicit arg wins, else taken from loss_cfg
        # (lets the YAML choose ``robust: bce`` / ``gce`` and locality params).
        if order_loss_config is None:
            order_loss_config = cfg.get("order_loss_config") or {}
        self.read_order_loss = RelativeReadingOrderLoss(**order_loss_config)

    # ---- assignment helpers --------------------------------------------------------
    @staticmethod
    def _get_index_updates(num_query_objects, gt_class, match_indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)]
        )
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx = src_idx + batch_idx * num_query_objects
        target_assign = torch.cat(
            [gt_class[i][dst] for i, (_, dst) in enumerate(match_indices)]
        )
        return src_idx, target_assign

    @staticmethod
    def _get_src_target_assign(src, target, match_indices):
        src_assign = torch.cat(
            [
                s[I] if I.numel() > 0 else s.new_zeros((0,) + s.shape[1:])
                for s, (I, _) in zip(src, match_indices)
            ]
        )
        target_assign = torch.cat(
            [
                t[J] if J.numel() > 0 else t.new_zeros((0,) + t.shape[1:])
                for t, (_, J) in zip(target, match_indices)
            ]
        )
        return src_assign, target_assign

    @staticmethod
    def _get_num_gts(gt_class, device, dtype=torch.float32):
        n = sum(int(a.shape[0]) for a in gt_class)
        num_gts = torch.as_tensor(float(n), dtype=dtype, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts)
            num_gts = num_gts / torch.distributed.get_world_size()
        return num_gts.clamp(min=1.0)

    # ---- per-component losses ------------------------------------------------------
    def _get_loss_class(self, logits, gt_class, match_indices, bg_index, num_gts, postfix="", iou_score=None):
        name = "loss_class" + postfix
        b, q = logits.shape[:2]
        target_label = torch.full((b, q), bg_index, dtype=torch.long, device=logits.device)
        num_gt = sum(int(a.shape[0]) for a in gt_class)
        index = None
        if num_gt > 0:
            index, updates = self._get_index_updates(q, gt_class, match_indices)
            target_label.view(-1)[index] = updates.reshape(-1).long()

        if self.use_focal_loss:
            target_label = F.one_hot(target_label, self.num_classes + 1)[..., :-1].to(logits.dtype)
            if iou_score is not None and self.use_vfl:
                target_score = torch.zeros(b * q, dtype=logits.dtype, device=logits.device)
                if num_gt > 0:
                    target_score[index] = iou_score.reshape(-1).to(logits.dtype)
                target_score = target_score.reshape(b, q, 1) * target_label
                loss_ = self.loss_coeff["class"] * varifocal_loss_with_logits(
                    logits, target_score, target_label, num_gts / q
                )
            else:
                loss_ = self.loss_coeff["class"] * sigmoid_focal_loss(
                    logits, target_label, num_gts / q
                )
        else:
            loss_ = F.cross_entropy(
                logits.permute(0, 2, 1), target_label, weight=self.class_weight
            )
        return {name: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts, postfix=""):
        name_bbox, name_giou = "loss_bbox" + postfix, "loss_giou" + postfix
        if sum(int(a.shape[0]) for a in gt_bbox) == 0:
            z = boxes.new_zeros(())
            return {name_bbox: z, name_giou: z.clone()}

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        loss_bbox = self.loss_coeff["bbox"] * F.l1_loss(src_bbox, target_bbox, reduction="sum") / num_gts
        giou = torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_bbox), center_to_corners_format(target_bbox)
            )
        )
        loss_giou = self.loss_coeff["giou"] * (1 - giou).sum() / num_gts
        return {name_bbox: loss_bbox, name_giou: loss_giou}

    def _get_point_coords_by_uncertainty(self, masks):
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = torch.rand(
            num_masks, 1, self.num_oversample_points, 2, device=masks.device, dtype=masks.dtype
        )
        out_mask = F.grid_sample(
            masks.unsqueeze(1), 2.0 * sample_points - 1.0, align_corners=False
        ).squeeze(1).squeeze(1)
        out_mask = -out_mask.abs()
        _, topk_ind = torch.topk(out_mask, self.num_important_points, dim=1)
        sp = sample_points.squeeze(1)
        sample_points = torch.gather(sp, 1, topk_ind.unsqueeze(-1).expand(-1, -1, 2))
        if self.num_random_points > 0:
            sample_points = torch.cat(
                [
                    sample_points,
                    torch.rand(
                        num_masks, self.num_random_points, 2,
                        device=masks.device, dtype=masks.dtype,
                    ),
                ],
                dim=1,
            )
        return sample_points

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=""):
        """MaskDINO point-sampled mask + dice loss."""
        name_mask, name_dice = "loss_mask" + postfix, "loss_dice" + postfix
        if sum(int(a.shape[0]) for a in gt_mask) == 0:
            z = masks.new_zeros(())
            return {name_mask: z, name_dice: z.clone()}

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask, match_indices)
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0  # [M, 1, P, 2]

        src_masks = F.grid_sample(
            src_masks.unsqueeze(1), sample_points, align_corners=False
        ).squeeze(1).squeeze(1)  # [M, P]
        target_masks = F.grid_sample(
            target_masks.unsqueeze(1).to(src_masks.dtype), sample_points, align_corners=False
        ).squeeze(1).squeeze(1).detach()

        loss_mask = self.loss_coeff["mask"] * F.binary_cross_entropy_with_logits(
            src_masks, target_masks, reduction="none"
        ).mean(1).sum() / num_gts
        loss_dice = self.loss_coeff["dice"] * _dice_loss(src_masks, target_masks, num_gts)
        return {name_mask: loss_mask, name_dice: loss_dice}

    # ---- mask-IoU score for varifocal (vfl_iou_type='mask') ------------------------
    def _mask_iou_score(self, masks, gt_mask, match_indices):
        src_mask, target_mask = self._get_src_target_assign(masks.detach(), gt_mask, match_indices)
        if src_mask.shape[0] == 0:
            return None
        src_mask = F.interpolate(
            src_mask.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False
        ).squeeze(0)
        target_mask = F.interpolate(
            target_mask.unsqueeze(0).to(src_mask.dtype),
            size=src_mask.shape[-2:], mode="bilinear", align_corners=False,
        ).squeeze(0)
        src_mask = (src_mask.flatten(1).sigmoid() > 0.5).to(masks.dtype)
        target_mask = (target_mask.flatten(1) > 0.5).to(masks.dtype)
        inter = (src_mask * target_mask).sum(1)
        union = src_mask.sum(1) + target_mask.sum(1) - inter
        return ((inter + 1e-2) / (union + 1e-2)).unsqueeze(-1)

    # ---- prediction / aux losses ---------------------------------------------------
    def _get_prediction_loss(
        self, boxes, logits, gt_bbox, gt_class, masks, gt_mask, num_gts,
        postfix="", match_indices=None,
    ):
        if match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask
            )

        iou_score = None
        if self.use_vfl and sum(int(a.shape[0]) for a in gt_bbox) > 0:
            if self.vfl_iou_type == "mask" and masks is not None and gt_mask is not None:
                iou_score = self._mask_iou_score(masks, gt_mask, match_indices)
            else:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices
                )
                iou_score = _pairwise_box_iou(
                    center_to_corners_format(src_bbox), center_to_corners_format(target_bbox)
                )

        loss = {}
        loss.update(
            self._get_loss_class(
                logits, gt_class, match_indices, self.num_classes, num_gts, postfix, iou_score
            )
        )
        loss.update(self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(self._get_loss_mask(masks, gt_mask, match_indices, num_gts, postfix))
        return loss

    def _get_loss_aux(
        self, boxes, logits, gt_bbox, gt_class, num_gts, masks, gt_mask,
        postfix="", dn_match_indices=None,
    ):
        loss_class, loss_bbox, loss_giou, loss_mask, loss_dice = [], [], [], [], []
        has_mask = masks is not None and gt_mask is not None
        has_gt = sum(int(a.shape[0]) for a in gt_bbox) > 0

        for i in range(len(boxes)):
            aux_boxes, aux_logits = boxes[i], logits[i]
            aux_masks = masks[i] if has_mask else None
            if dn_match_indices is not None:
                match_indices = dn_match_indices
            else:
                match_indices = self.matcher(
                    aux_boxes, aux_logits, gt_bbox, gt_class,
                    masks=aux_masks, gt_mask=gt_mask,
                )

            iou_score = None
            if self.use_vfl and has_gt:
                # NOTE: PaddleDetection uses *bbox* IoU for the auxiliary VFL score even
                # when vfl_iou_type='mask' (only the main prediction loss uses mask IoU).
                src_bbox, target_bbox = self._get_src_target_assign(
                    aux_boxes.detach(), gt_bbox, match_indices
                )
                iou_score = _pairwise_box_iou(
                    center_to_corners_format(src_bbox), center_to_corners_format(target_bbox)
                )

            loss_class.append(
                self._get_loss_class(
                    aux_logits, gt_class, match_indices, self.num_classes,
                    num_gts, postfix, iou_score,
                )["loss_class" + postfix]
            )
            lb = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices, num_gts, postfix)
            loss_bbox.append(lb["loss_bbox" + postfix])
            loss_giou.append(lb["loss_giou" + postfix])
            if has_mask:
                lm = self._get_loss_mask(aux_masks, gt_mask, match_indices, num_gts, postfix)
                loss_mask.append(lm["loss_mask" + postfix])
                loss_dice.append(lm["loss_dice" + postfix])

        loss = {
            "loss_class_aux" + postfix: sum(loss_class),
            "loss_bbox_aux" + postfix: sum(loss_bbox),
            "loss_giou_aux" + postfix: sum(loss_giou),
        }
        if has_mask:
            loss["loss_mask_aux" + postfix] = sum(loss_mask)
            loss["loss_dice_aux" + postfix] = sum(loss_dice)
        return loss

    @staticmethod
    def _get_dn_match_indices(gt_class, dn_positive_idx, dn_num_group, device):
        dn_match_indices = []
        for i in range(len(gt_class)):
            num_gt = int(gt_class[i].shape[0])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.long, device=device).tile((dn_num_group,))
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                dn_match_indices.append((empty, empty.clone()))
        return dn_match_indices

    # ---- forward -------------------------------------------------------------------
    def forward(
        self,
        intermediate_logits: torch.Tensor,            # (B, L, Q_total, C)
        intermediate_reference_points: torch.Tensor,  # (B, L, Q_total, 4)
        out_masks: torch.Tensor,                      # (B, L, Q_total, h, w)
        out_order_logits: torch.Tensor,               # (B, L, Q, Q)
        enc_topk_logits: torch.Tensor | None,         # (B, Q, C)
        enc_topk_bboxes: torch.Tensor | None,         # (B, Q, 4)
        enc_topk_masks: torch.Tensor | None,          # (B, Q, h, w) or None
        denoising_meta_values: dict | None,
        targets: list[dict],
    ):
        device = intermediate_logits.device

        gt_class = [t["class_labels"].reshape(-1, 1).long() for t in targets]
        gt_bbox = [t["boxes"] for t in targets]
        gt_mask = [t["masks"] for t in targets]
        gt_read_order = [t["order_rank"] for t in targets]

        # Split the denoising slice off the front of every per-query tensor.
        if denoising_meta_values is not None:
            dn_split = denoising_meta_values["dn_num_split"]  # [num_dn, num_q]
            dn_logits, match_logits = torch.split(intermediate_logits, dn_split, dim=2)
            dn_refp, match_refp = torch.split(intermediate_reference_points, dn_split, dim=2)
            dn_masks, match_masks = torch.split(out_masks, dn_split, dim=2)
        else:
            match_logits, match_refp, match_masks = (
                intermediate_logits, intermediate_reference_points, out_masks,
            )
            dn_logits = dn_refp = dn_masks = None

        # (B, L, ...) -> (L, B, ...)
        logits_lb = match_logits.transpose(0, 1).contiguous()
        boxes_lb = match_refp.transpose(0, 1).contiguous()
        masks_lb = match_masks.transpose(0, 1).contiguous()
        order_lb = out_order_logits.transpose(0, 1).contiguous()

        # Prepend the encoder-stage proposal as the first layer (PaddleDetection layout).
        if enc_topk_masks is not None and enc_topk_logits is not None and enc_topk_bboxes is not None:
            logits_all = torch.cat([enc_topk_logits.unsqueeze(0), logits_lb], dim=0)
            boxes_all = torch.cat([enc_topk_bboxes.unsqueeze(0), boxes_lb], dim=0)
            masks_all = torch.cat([enc_topk_masks.unsqueeze(0), masks_lb], dim=0)
        else:
            # Fallback (model did not expose encoder-stage masks): decoder layers only.
            logits_all, boxes_all, masks_all = logits_lb, boxes_lb, masks_lb

        num_gts = self._get_num_gts(gt_class, device)

        # Main prediction loss (last decoder layer).
        total_loss = self._get_prediction_loss(
            boxes_all[-1], logits_all[-1], gt_bbox, gt_class,
            masks_all[-1], gt_mask, num_gts,
        )

        # Auxiliary losses (encoder layer + all but last decoder layer).
        if self.aux_loss and len(boxes_all) > 1:
            total_loss.update(
                self._get_loss_aux(
                    boxes_all[:-1], logits_all[:-1], gt_bbox, gt_class,
                    num_gts, masks_all[:-1], gt_mask,
                )
            )

        # CDN denoising losses (inactive when config.num_denoising == 0).
        if denoising_meta_values is not None and dn_logits is not None:
            dn_positive_idx = denoising_meta_values["dn_positive_idx"]
            dn_num_group = int(denoising_meta_values["dn_num_group"])
            dn_match_indices = self._get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group, device
            )
            dn_num_gts = num_gts * max(dn_num_group, 1)

            dn_logits_lb = dn_logits.transpose(0, 1).contiguous()
            dn_boxes_lb = dn_refp.transpose(0, 1).contiguous()
            dn_masks_lb = dn_masks.transpose(0, 1).contiguous()

            total_loss.update(
                self._get_prediction_loss(
                    dn_boxes_lb[-1], dn_logits_lb[-1], gt_bbox, gt_class,
                    dn_masks_lb[-1], gt_mask, dn_num_gts,
                    postfix="_dn", match_indices=dn_match_indices,
                )
            )
            if self.aux_loss and len(dn_boxes_lb) > 1:
                total_loss.update(
                    self._get_loss_aux(
                        dn_boxes_lb[:-1], dn_logits_lb[:-1], gt_bbox, gt_class,
                        dn_num_gts, dn_masks_lb[:-1], gt_mask,
                        postfix="_dn", dn_match_indices=dn_match_indices,
                    )
                )
        else:
            total_loss.update({k + "_dn": torch.zeros((), device=device) for k in list(total_loss)})

        # Reading-order loss — last decoder layer only, matcher re-run on it.
        if out_order_logits is not None:
            match_indices = self.matcher(
                boxes_lb[-1], logits_lb[-1], gt_bbox, gt_class,
                masks=masks_lb[-1], gt_mask=gt_mask,
            )
            total_loss["loss_order"] = (
                self.read_order_loss(order_lb[-1], gt_read_order, match_indices)
                * self.loss_coeff["order"]
            )

        total = sum(total_loss.values())
        return total, total_loss
