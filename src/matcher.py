"""Hungarian matcher for PP-DocLayoutV3 — pure-PyTorch port.

Faithful reimplementation of
``PaddleDetection/ppdet/modeling/transformers/matchers.py::HungarianMatcher``
(the variant used by ``DocLayoutV3Head``). The cost matrix combines:

* focal classification cost (``pos_cost - neg_cost``),
* L1 box cost,
* GIoU cost (``-giou``),
* — when ``with_mask`` — BCE mask cost + dice cost, both evaluated on a shared
  set of random points sampled with ``F.grid_sample`` (mirrors PaddleDetection
  exactly, including the ``num_sample_points`` normalisation).

``boxes`` / ``gt_bbox`` are in ``cxcywh`` normalised form.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from transformers.image_transforms import center_to_corners_format
from transformers.loss.loss_for_object_detection import generalized_box_iou

__all__ = ["HungarianMatcher"]


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        matcher_coeff: dict | None = None,
        use_focal_loss: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.matcher_coeff = matcher_coeff or {
            "class": 1,
            "bbox": 5,
            "giou": 2,
            "mask": 1,
            "dice": 1,
        }
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        gt_bbox: list[torch.Tensor],
        gt_class: list[torch.Tensor],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Args mirror PaddleDetection's matcher.

        boxes:  [b, query, 4]   (cxcywh, normalised)
        logits: [b, query, num_classes]
        gt_bbox:  list of [n_i, 4]
        gt_class: list of [n_i] or [n_i, 1]
        masks:    [b, query, h, w]            (optional)
        gt_mask:  list of [n_i, H, W]         (optional)

        Returns a list of length ``b`` of ``(pred_idx, gt_idx)`` LongTensor tuples.
        """
        bs, num_queries = boxes.shape[:2]
        device = boxes.device

        num_gts = [int(g.shape[0]) for g in gt_class]
        if sum(num_gts) == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return [(empty.clone(), empty.clone()) for _ in range(bs)]

        # [b*query, num_classes]
        logits = logits.detach()
        if self.use_focal_loss:
            out_prob = logits.flatten(0, 1).sigmoid()
        else:
            out_prob = logits.flatten(0, 1).softmax(-1)
        # [b*query, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        tgt_ids = torch.cat([g.reshape(-1) for g in gt_class]).long()
        tgt_bbox = torch.cat(gt_bbox, dim=0)

        # ---- classification cost --------------------------------------------------
        out_prob = out_prob[:, tgt_ids]  # [b*query, sum_n]
        if self.use_focal_loss:
            neg_cost_class = (
                (1 - self.alpha)
                * (out_prob ** self.gamma)
                * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha
                * ((1 - out_prob) ** self.gamma)
                * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # ---- L1 box cost ----------------------------------------------------------
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # ---- GIoU cost ------------------------------------------------------------
        cost_giou = -generalized_box_iou(
            center_to_corners_format(out_bbox),
            center_to_corners_format(tgt_bbox),
        )

        cost = (
            self.matcher_coeff["class"] * cost_class
            + self.matcher_coeff["bbox"] * cost_bbox
            + self.matcher_coeff["giou"] * cost_giou
        )

        # ---- mask + dice cost -----------------------------------------------------
        if self.with_mask:
            assert masks is not None and gt_mask is not None, (
                "with_mask=True requires `masks` and `gt_mask`"
            )
            # One shared random point grid per image.
            sample_points = torch.rand(
                bs, 1, self.num_sample_points, 2, device=device, dtype=out_bbox.dtype
            )
            sample_points = 2.0 * sample_points - 1.0

            out_mask = F.grid_sample(
                masks.detach(), sample_points, align_corners=False
            ).squeeze(-2)  # [b, query, num_sample_points]
            out_mask = out_mask.flatten(0, 1)  # [b*query, num_sample_points]

            tgt_mask = torch.cat(gt_mask, dim=0).unsqueeze(1).to(out_mask.dtype)  # [sum_n,1,H,W]
            tgt_points = torch.cat(
                [
                    sample_points[i].expand(n, -1, -1, -1)
                    for i, n in enumerate(num_gts)
                    if n > 0
                ],
                dim=0,
            )  # [sum_n, 1, num_sample_points, 2]
            tgt_mask = F.grid_sample(
                tgt_mask, tgt_points, align_corners=False
            ).squeeze(1).squeeze(1)  # [sum_n, num_sample_points]

            pos_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.ones_like(out_mask), reduction="none"
            )
            neg_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.zeros_like(out_mask), reduction="none"
            )
            cost_mask = pos_cost_mask @ tgt_mask.T + neg_cost_mask @ (1 - tgt_mask).T
            cost_mask = cost_mask / self.num_sample_points

            out_mask_p = out_mask.sigmoid()
            numerator = 2 * (out_mask_p @ tgt_mask.T)
            denominator = out_mask_p.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            cost = (
                cost
                + self.matcher_coeff["mask"] * cost_mask
                + self.matcher_coeff["dice"] * cost_dice
            )

        # ---- solve ----------------------------------------------------------------
        cost = cost.reshape(bs, num_queries, -1)
        cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
        cost = cost.cpu()
        per_image = cost.split(num_gts, dim=-1)

        indices = []
        for i in range(bs):
            if num_gts[i] == 0:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                indices.append((empty.clone(), empty.clone()))
                continue
            c = per_image[i][i].numpy()
            row, col = linear_sum_assignment(c)
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=device),
                    torch.as_tensor(col, dtype=torch.long, device=device),
                )
            )
        return indices
