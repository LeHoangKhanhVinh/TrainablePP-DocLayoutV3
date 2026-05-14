"""PP-DocLayoutV3 post-processing — pure-PyTorch port.

Faithful reimplementation of
``PaddleDetection/ppdet/modeling/post_process.py``:

* ``get_order``               — voting-based reading-order decode.
* ``DocLayoutV3PostProcess``  — bbox decode (cxcywh -> xyxy, scaled to the
  original image), focal-loss top-k selection, reading-order decode and
  optional mask post-processing.

Output bbox format matches PaddleDetection's ``DocLayoutV3PostProcess``:
``[label, score, x1, y1, x2, y2, order]`` per detection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["get_order", "DocLayoutV3PostProcess"]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def get_order(order_logits: torch.Tensor):
    """Decode a pairwise order-logit matrix into a per-element reading-order rank.

    Port of ``post_process.py::get_order``.

    Args:
        order_logits: ``[B, N, N]`` — ``logits[b, i, j] > 0`` means element ``i``
            comes before element ``j``.

    Returns:
        ``(order_seq, order_votes)`` — ``order_seq[b, i]`` is the 0-indexed
        position of element ``i`` in the decoded reading order; ``order_votes``
        is the (column-summed) vote score used for sorting.
    """
    b, n, _ = order_logits.shape
    order_scores = torch.sigmoid(order_logits)
    eye = torch.eye(n, dtype=order_scores.dtype, device=order_scores.device).unsqueeze(0)
    order_scores = order_scores * (1.0 - eye)
    # order_votes[i] = number of elements predicted to come before element i
    order_votes = order_scores.sum(dim=1)  # [B, N]
    order_pointers = torch.argsort(order_votes, dim=1, descending=False)
    order_seq = torch.full((b, n), -1, dtype=torch.long, device=order_logits.device)
    batch_idx = torch.arange(b, device=order_logits.device).reshape(-1, 1).expand(b, n)
    ranks = torch.arange(n, device=order_logits.device).expand(b, n)
    order_seq[batch_idx, order_pointers] = ranks
    return order_seq, order_votes


class DocLayoutV3PostProcess:
    """Decode PP-DocLayoutV3 head outputs — port of ``DocLayoutV3PostProcess``.

    Args:
        num_classes: number of object categories.
        num_top_queries: how many detections to keep per image.
        use_focal_loss: focal/sigmoid scoring (always True for PP-DocLayoutV3).
        with_mask: whether to post-process instance masks.
        mask_threshold: binarisation threshold for masks.
        resize_mask: if True, masks are bilinearly resized to the original image
            resolution; otherwise they are left at the model's mask resolution.
        use_avg_mask_score: multiply each score by its mask's average soft score.
    """

    def __init__(self, num_classes=25, num_top_queries=300, use_focal_loss=True,
                 with_mask=True, mask_threshold=0.5, resize_mask=False,
                 use_avg_mask_score=False):
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.mask_threshold = mask_threshold
        self.resize_mask = resize_mask
        self.use_avg_mask_score = use_avg_mask_score

    def _mask_postprocess(self, mask_pred, score_pred):
        mask_score = torch.sigmoid(mask_pred)
        mask_bin = (mask_score > self.mask_threshold).to(mask_score.dtype)
        if self.use_avg_mask_score:
            avg = (mask_bin * mask_score).sum(dim=(-2, -1)) / (
                mask_bin.sum(dim=(-2, -1)) + 1e-6)
            score_pred = score_pred * avg
        return mask_bin.flatten(0, 1).to(torch.int32), score_pred

    @torch.no_grad()
    def __call__(self, head_out, orig_target_sizes):
        """Decode raw head outputs.

        Args:
            head_out: ``(bboxes, logits, order_logits, masks)`` —
                ``bboxes`` ``[B, Q, 4]`` cxcywh normalised, ``logits`` ``[B, Q, C]``,
                ``order_logits`` ``[B, Q, Q]``, ``masks`` ``[B, Q, h, w]`` or None.
            orig_target_sizes: ``[B, 2]`` original ``(height, width)`` per image
                (corresponds to PaddleDetection's ``bbox_decode_type='origin'``).

        Returns:
            ``(bbox_pred, bbox_num, mask_pred)`` where ``bbox_pred`` is ``[N, 7]``
            (``label, score, x1, y1, x2, y2, order``), ``bbox_num`` is ``[B]`` and
            ``mask_pred`` is ``[N, h', w']`` int32 (or None).
        """
        bboxes, logits, order_logits, masks = head_out
        device = bboxes.device
        bs = bboxes.shape[0]

        bbox_pred = _cxcywh_to_xyxy(bboxes)

        orig = torch.as_tensor(orig_target_sizes, device=device, dtype=bbox_pred.dtype)
        if orig.dim() == 1:
            orig = orig.unsqueeze(0)
        img_h, img_w = orig.unbind(-1)
        # out_shape = (w, h, w, h) per image
        out_shape = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
        bbox_pred = bbox_pred * out_shape

        scores = torch.sigmoid(logits)
        order_seq, _ = get_order(order_logits)

        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
        labels = index % self.num_classes
        query_idx = index // self.num_classes

        bbox_pred = bbox_pred.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, 4))
        order_seq = order_seq.gather(1, query_idx)

        mask_pred = None
        if self.with_mask and masks is not None:
            h, w = masks.shape[-2:]
            masks = masks.gather(
                1, query_idx[:, :, None, None].expand(-1, -1, h, w)
            )
            if self.resize_mask:
                # one image at a time — sizes may differ across the batch
                resized = []
                for i in range(bs):
                    th, tw = int(img_h[i].item()), int(img_w[i].item())
                    resized.append(
                        F.interpolate(masks[i:i + 1], size=(th, tw),
                                      mode="bilinear", align_corners=False)[0]
                    )
                if len({m.shape for m in resized}) == 1:
                    masks = torch.stack(resized, dim=0)
                else:  # ragged batch — flatten directly
                    mask_pred = torch.cat(
                        [self._mask_postprocess(m.unsqueeze(0), scores[i:i + 1])[0]
                         for i, m in enumerate(resized)], dim=0
                    )
            if mask_pred is None:
                mask_pred, scores = self._mask_postprocess(masks, scores)

        bbox_pred = torch.cat(
            [
                labels.unsqueeze(-1).to(bbox_pred.dtype),
                scores.unsqueeze(-1),
                bbox_pred,
                order_seq.unsqueeze(-1).to(bbox_pred.dtype),
            ],
            dim=-1,
        )
        bbox_num = torch.full((bs,), self.num_top_queries, dtype=torch.int32, device=device)
        bbox_pred = bbox_pred.reshape(-1, 7)
        return bbox_pred, bbox_num, mask_pred
