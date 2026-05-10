"""Loss for PP-DocLayoutV3 fine-tuning (3 heads: detection, mask, reading order).

Detection loss reuses the RT-DETR style: VFL classification + L1 + GIoU on
matched query / target pairs after Hungarian assignment. Auxiliary losses are
applied to every decoder layer plus the encoder-stage proposals.

Mask and reading-order losses are PP-DocLayoutV3 specific:

* Mask: per-query 200x200 logits BMM'd from ``mask_query_embed`` and
  ``mask_feat``; supervised with focal + dice loss against rasterized polygon
  masks.
* Reading order: ``decoder_global_pointer`` outputs a ``(num_queries,
  num_queries)`` pairwise score matrix per layer. We supervise the upper
  triangle of the matched-query subset with BCE so that
  ``order_logit[i, j]`` is high iff target ``rank[i] < rank[j]`` — which
  matches the inference voting in ``_get_order_seqs``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.image_transforms import center_to_corners_format
from transformers.loss.loss_for_object_detection import (
    box_iou,
    dice_loss,
    generalized_box_iou,
    sigmoid_focal_loss,
)
from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher


def _matcher_config(config) -> SimpleNamespace:
    """Wrap a PPDocLayoutV3Config so it satisfies RTDetrHungarianMatcher's API."""
    return SimpleNamespace(
        matcher_class_cost=config.matcher_class_cost,
        matcher_bbox_cost=config.matcher_bbox_cost,
        matcher_giou_cost=config.matcher_giou_cost,
        matcher_alpha=config.matcher_alpha,
        matcher_gamma=config.matcher_gamma,
        use_focal_loss=True,  # PPDocLayoutV3 uses sigmoid scoring (focal-style).
    )


def _focal_loss_for_objdet(src_logits, target_classes, num_boxes, num_classes, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss with one-hot targets, summed over queries / N_boxes."""
    target = F.one_hot(target_classes, num_classes=num_classes + 1)[..., :-1].to(src_logits.dtype)
    return sigmoid_focal_loss(src_logits, target, num_boxes, alpha=alpha, gamma=gamma) * src_logits.shape[1]


class PPDocLayoutV3Loss(nn.Module):
    """Combined detection + mask + reading-order loss.

    The ``__call__`` signature differs from RTDetrLoss in that it takes the
    *intermediate* tensors directly from ``PPDocLayoutV3ModelOutput``, plus the
    encoder-stage proposals and (when present) ``denoising_meta_values``.
    """

    def __init__(
        self,
        config,
        weight_loss_vfl: float = 1.0,
        weight_loss_bbox: float = 5.0,
        weight_loss_giou: float = 2.0,
        weight_loss_mask: float = 1.0,
        weight_loss_dice: float = 1.0,
        weight_loss_order: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.matcher = RTDetrHungarianMatcher(_matcher_config(config))
        self.num_classes = config.num_labels
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.weights = {
            "loss_vfl": weight_loss_vfl,
            "loss_bbox": weight_loss_bbox,
            "loss_giou": weight_loss_giou,
            "loss_mask": weight_loss_mask,
            "loss_dice": weight_loss_dice,
            "loss_order": weight_loss_order,
        }

    # ---- core per-layer loss helpers --------------------------------------------------

    def _det_loss(self, logits, pred_boxes, targets, indices, num_boxes):
        idx = self._batch_query_idx(indices)

        # VFL classification
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if target_boxes.numel() == 0:
            ious = torch.zeros((0,), device=logits.device, dtype=logits.dtype)
        else:
            iou_mat, _ = box_iou(
                center_to_corners_format(src_boxes.detach()),
                center_to_corners_format(target_boxes),
            )
            ious = torch.diag(iou_mat)

        target_classes = torch.full(
            logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device
        )
        target_classes_orig = torch.cat(
            [t["class_labels"][i] for t, (_, i) in zip(targets, indices)]
        ) if any(len(t["class_labels"]) for t in targets) else torch.zeros((0,), dtype=torch.long, device=logits.device)
        target_classes[idx] = target_classes_orig

        target_one_hot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(logits.dtype)
        target_score_orig = torch.zeros_like(target_classes, dtype=logits.dtype)
        target_score_orig[idx] = ious.to(logits.dtype)
        target_score = target_score_orig.unsqueeze(-1) * target_one_hot

        pred_score = torch.sigmoid(logits.detach())
        weight = (self.alpha * pred_score.pow(self.gamma) * (1 - target_one_hot) + target_score).to(logits.dtype)
        loss_vfl = F.binary_cross_entropy_with_logits(logits, target_score, weight=weight, reduction="none")
        loss_vfl = loss_vfl.mean(1).sum() * logits.shape[1] / num_boxes

        # L1 + GIoU
        if target_boxes.numel() == 0:
            loss_bbox = logits.new_zeros(())
            loss_giou = logits.new_zeros(())
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
            loss_giou = (
                1
                - torch.diag(
                    generalized_box_iou(
                        center_to_corners_format(src_boxes), center_to_corners_format(target_boxes)
                    )
                )
            ).sum() / num_boxes

        return {"loss_vfl": loss_vfl, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _mask_loss(self, out_masks, targets, indices, num_boxes):
        if num_boxes == 0:
            zero = out_masks.new_zeros(())
            return {"loss_mask": zero, "loss_dice": zero}

        # Gather predicted masks at matched queries.
        src_masks = []
        tgt_masks = []
        for batch_i, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            src_masks.append(out_masks[batch_i, pred_idx])
            tgt_masks.append(targets[batch_i]["masks"][tgt_idx].to(out_masks))

        if not src_masks:
            zero = out_masks.new_zeros(())
            return {"loss_mask": zero, "loss_dice": zero}

        src = torch.cat(src_masks, dim=0)  # (M, hP, wP) — at FPN-output stride
        tgt = torch.cat(tgt_masks, dim=0)  # (M, hT, wT)

        # Upsample predictions to target size (typical Detr-style mask supervision).
        src_up = F.interpolate(src.unsqueeze(1), size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        src_up = src_up[:, 0].flatten(1)  # (M, hT*wT)
        tgt_flat = tgt.flatten(1)

        loss_mask = sigmoid_focal_loss(src_up, tgt_flat, num_boxes, alpha=self.alpha, gamma=self.gamma)
        loss_dice = dice_loss(src_up, tgt_flat, num_boxes)
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _order_loss(self, order_logits, targets, indices):
        # order_logits: (B, num_queries, num_queries) for one decoder layer.
        per_batch_losses = []
        for batch_i, (pred_idx, tgt_idx) in enumerate(indices):
            n = pred_idx.numel()
            if n < 2:
                continue
            ranks = targets[batch_i]["order_rank"][tgt_idx]  # (n,)
            pair_target = (ranks.unsqueeze(0) > ranks.unsqueeze(1)).to(order_logits.dtype)  # (n, n)
            pair_logits = order_logits[batch_i][pred_idx][:, pred_idx]  # (n, n)
            upper = torch.triu(torch.ones_like(pair_target, dtype=torch.bool), diagonal=1)
            if upper.any():
                per_batch_losses.append(
                    F.binary_cross_entropy_with_logits(pair_logits[upper], pair_target[upper])
                )
        if not per_batch_losses:
            return {"loss_order": order_logits.new_zeros(())}
        return {"loss_order": torch.stack(per_batch_losses).mean()}

    # ---- top-level forward ------------------------------------------------------------

    def forward(
        self,
        intermediate_logits: torch.Tensor,           # (B, L, Q_total, C)
        intermediate_reference_points: torch.Tensor, # (B, L, Q_total, 4)
        out_masks: torch.Tensor,                     # (B, L, Q_total, hP, wP)
        out_order_logits: torch.Tensor,              # (B, L, num_queries, num_queries)
        enc_topk_logits: torch.Tensor | None,        # (B, num_queries, C) or None
        enc_topk_bboxes: torch.Tensor | None,        # (B, num_queries, 4) or None
        denoising_meta_values: dict | None,
        targets: list[dict],
    ):
        device = intermediate_logits.device

        # Split denoising slice off the front of every per-query tensor.
        if denoising_meta_values is not None:
            dn_split = denoising_meta_values["dn_num_split"]  # [num_dn, num_q]
            dn_logits, match_logits = torch.split(intermediate_logits, dn_split, dim=2)
            dn_refp, match_refp = torch.split(intermediate_reference_points, dn_split, dim=2)
            dn_masks, match_masks = torch.split(out_masks, dn_split, dim=2)
        else:
            match_logits = intermediate_logits
            match_refp = intermediate_reference_points
            match_masks = out_masks
            dn_logits = dn_refp = dn_masks = None

        num_layers = match_logits.shape[1]
        last_layer_logits = match_logits[:, -1]
        last_layer_refp = match_refp[:, -1]

        # Hungarian matching uses the last decoder layer.
        indices = self.matcher(
            {"logits": last_layer_logits, "pred_boxes": last_layer_refp}, targets
        )
        num_boxes = max(sum(len(t["class_labels"]) for t in targets), 1)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = float(torch.clamp(num_boxes_t, min=1).item())

        loss_dict: dict[str, torch.Tensor] = {}

        # Last layer (main loss): det + mask + order.
        last = self._det_loss(last_layer_logits, last_layer_refp, targets, indices, num_boxes)
        last.update(self._mask_loss(match_masks[:, -1], targets, indices, num_boxes))
        last.update(self._order_loss(out_order_logits[:, -1], targets, indices))
        for k, v in last.items():
            loss_dict[k] = v

        # Auxiliary losses on intermediate decoder layers.
        for layer_i in range(num_layers - 1):
            aux_logits = match_logits[:, layer_i]
            aux_refp = match_refp[:, layer_i]
            aux_indices = self.matcher({"logits": aux_logits, "pred_boxes": aux_refp}, targets)
            aux = self._det_loss(aux_logits, aux_refp, targets, aux_indices, num_boxes)
            # Skip mask aux on layers other than the last for cost (RT-DETR convention).
            aux.update(self._order_loss(out_order_logits[:, layer_i], targets, aux_indices))
            for k, v in aux.items():
                loss_dict[f"{k}_aux_{layer_i}"] = v

        # Encoder-stage proposals get det loss only.
        if enc_topk_logits is not None and enc_topk_bboxes is not None:
            enc_indices = self.matcher(
                {"logits": enc_topk_logits, "pred_boxes": enc_topk_bboxes}, targets
            )
            enc = self._det_loss(enc_topk_logits, enc_topk_bboxes, targets, enc_indices, num_boxes)
            for k, v in enc.items():
                loss_dict[f"{k}_enc"] = v

        # Denoising auxiliaries: indices come from get_cdn_matched_indices (no Hungarian).
        if dn_logits is not None and dn_refp is not None:
            dn_indices = _get_cdn_matched_indices(denoising_meta_values, targets, device)
            dn_num_boxes = num_boxes * max(int(denoising_meta_values["dn_num_group"]), 1)
            for layer_i in range(dn_logits.shape[1]):
                dl = dn_logits[:, layer_i]
                dr = dn_refp[:, layer_i]
                dn = self._det_loss(dl, dr, targets, dn_indices, dn_num_boxes)
                for k, v in dn.items():
                    loss_dict[f"{k}_dn_{layer_i}"] = v

        # Apply weights.
        weighted = {}
        total = intermediate_logits.new_zeros(())
        for k, v in loss_dict.items():
            base = k.split("_aux_")[0].split("_dn_")[0].split("_enc")[0]
            w = self.weights.get(base, 0.0)
            wv = v * w
            weighted[k] = wv
            total = total + wv
        return total, weighted

    @staticmethod
    def _batch_query_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, b) for b, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def _get_cdn_matched_indices(dn_meta, targets, device):
    """Map CDN denoising queries to their corresponding target indices.

    Mirrors :pymeth:`RTDetrLoss.get_cdn_matched_indices` but parameterized on
    ``device`` so it works when a sample has zero targets on a non-default
    device.
    """
    dn_positive_idx = dn_meta["dn_positive_idx"]
    dn_num_group = int(dn_meta["dn_num_group"])
    matched = []
    for i, t in enumerate(targets):
        n_gt = int(t["class_labels"].numel())
        if n_gt > 0:
            gt_idx = torch.arange(n_gt, dtype=torch.long, device=device).tile(dn_num_group)
            matched.append((dn_positive_idx[i], gt_idx))
        else:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            matched.append((empty, empty))
    return matched


__all__ = ["PPDocLayoutV3Loss"]
