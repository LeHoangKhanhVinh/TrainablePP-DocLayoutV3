"""Evaluation metrics for PP-DocLayoutV3 — pure numpy, no pycocotools.

PaddleDetection scores PP-DocLayoutV3 with ``DocLayoutV3Metric`` which is just
``COCOMetric`` (COCO bbox AP, optionally mask AP). This module provides a
self-contained equivalent:

* :class:`LayoutMetric` — COCO-style bbox detection AP
  (``mAP`` = AP@[.50:.05:.95], plus ``AP50`` / ``AP75``), computed exactly the
  COCO way (greedy per-class matching, 101-point interpolation), **plus** a
  reading-order score based on the **normalised edit distance** (NED) between
  the predicted and ground-truth order sequences over matched detections —
  reported as ``order_score = 1 - NED`` so that, like the AP metrics, bigger is
  better.

Boxes are ``xyxy``; predictions and ground truth only need to be in the *same*
coordinate space per image (IoU is scale-invariant), so callers may work in
normalised or pixel coordinates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LayoutMetric"]


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``[N,4]`` / ``[M,4]`` xyxy box sets -> ``[N,M]``."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _levenshtein(a, b) -> int:
    """Levenshtein edit distance between two sequences (substitution cost 1)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP from cumulative recall / precision arrays."""
    if recall.size == 0:
        return 0.0
    mpre = precision.copy()
    # make precision envelope monotonically non-increasing from the right
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    rec_thrs = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, rec_thrs, side="left")
    q = np.zeros(101, dtype=np.float64)
    valid = idx < mpre.size
    q[valid] = mpre[idx[valid]]
    return float(q.mean())


class LayoutMetric:
    """COCO-style bbox AP + pairwise reading-order accuracy.

    Usage::

        metric = LayoutMetric(num_classes=12)
        metric.update(preds, gts)        # per-batch, lists of per-image dicts
        results = metric.compute()       # {'mAP', 'AP50', 'AP75', 'order_acc', ...}

    ``preds`` items: ``{"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,),
    "order": (N,)}``; ``gts`` items: ``{"boxes": (M,4) xyxy, "labels": (M,),
    "order": (M,)}``. ``order`` is optional (reading-order accuracy is skipped
    if absent).
    """

    def __init__(self, num_classes: int, iou_thresholds=None, order_iou_thr: float = 0.5):
        self.num_classes = int(num_classes)
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 0.96, 0.05)
        self.iou_thrs = np.asarray(iou_thresholds, dtype=np.float64)
        self.order_iou_thr = float(order_iou_thr)
        self.reset()

    def reset(self) -> None:
        self._preds: list[dict] = []
        self._gts: list[dict] = []

    # ----------------------------------------------------------------------------------
    def update(self, preds: list[dict], gts: list[dict]) -> None:
        """Accumulate one batch of per-image predictions / ground truths."""
        assert len(preds) == len(gts), "preds and gts must align per image"
        for p, g in zip(preds, gts):
            self._preds.append({
                "boxes": np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4),
                "scores": np.asarray(p["scores"], dtype=np.float64).reshape(-1),
                "labels": np.asarray(p["labels"]).reshape(-1).astype(np.int64),
                "order": (None if p.get("order") is None
                          else np.asarray(p["order"], dtype=np.float64).reshape(-1)),
            })
            self._gts.append({
                "boxes": np.asarray(g["boxes"], dtype=np.float64).reshape(-1, 4),
                "labels": np.asarray(g["labels"]).reshape(-1).astype(np.int64),
                "order": (None if g.get("order") is None
                          else np.asarray(g["order"], dtype=np.float64).reshape(-1)),
            })

    # ----------------------------------------------------------------------------------
    def _ap_per_class(self, cls: int):
        """Return ``(ap_per_iou_thr [T], n_gt)`` for one class."""
        # gather detections of this class across images, sorted by score desc
        dets = []  # (img_idx, score, box)
        n_gt = 0
        gt_per_img: dict[int, np.ndarray] = {}
        for img_idx, g in enumerate(self._gts):
            sel = g["labels"] == cls
            gt_per_img[img_idx] = g["boxes"][sel]
            n_gt += int(sel.sum())
        for img_idx, p in enumerate(self._preds):
            sel = p["labels"] == cls
            boxes = p["boxes"][sel]
            scores = p["scores"][sel]
            for b, s in zip(boxes, scores):
                dets.append((img_idx, float(s), b))

        if n_gt == 0:
            return np.full(self.iou_thrs.size, np.nan), 0
        if len(dets) == 0:
            return np.zeros(self.iou_thrs.size, dtype=np.float64), n_gt

        dets.sort(key=lambda d: d[1], reverse=True)
        det_imgs = np.array([d[0] for d in dets])
        det_boxes = np.stack([d[2] for d in dets], axis=0)

        # precompute IoU of each detection vs its image's GT
        ious_per_det = [
            _iou_matrix(det_boxes[i:i + 1], gt_per_img[det_imgs[i]])[0]
            for i in range(len(dets))
        ]

        aps = np.zeros(self.iou_thrs.size, dtype=np.float64)
        for ti, thr in enumerate(self.iou_thrs):
            matched = {img: np.zeros(gt_per_img[img].shape[0], dtype=bool)
                       for img in gt_per_img}
            tp = np.zeros(len(dets), dtype=np.float64)
            fp = np.zeros(len(dets), dtype=np.float64)
            for di in range(len(dets)):
                ious = ious_per_det[di]
                img = det_imgs[di]
                best_iou, best_j = thr, -1
                for j in range(ious.shape[0]):
                    if matched[img][j]:
                        continue
                    if ious[j] >= best_iou:
                        best_iou, best_j = ious[j], j
                if best_j >= 0:
                    matched[img][best_j] = True
                    tp[di] = 1.0
                else:
                    fp[di] = 1.0
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            aps[ti] = _ap_from_pr(recall, precision)
        return aps, n_gt

    def _order_score(self) -> tuple[float, int]:
        """Reading-order score: ``1 - NED`` over IoU-matched detections.

        For each image, predictions are greedily matched to ground truth
        (score-ordered, class-aware, IoU >= ``order_iou_thr``). Each matched GT
        gets a unique id; the *reference* sequence is those ids ordered by GT
        reading order and the *hypothesis* sequence is the same ids ordered by
        the predicted reading order. The normalised edit distance is aggregated
        globally — ``NED = sum(edit_distance) / sum(sequence_length)`` — and
        returned as ``1 - NED`` (bigger is better).
        """
        total_ed = 0
        total_len = 0
        for p, g in zip(self._preds, self._gts):
            if p["order"] is None or g["order"] is None:
                continue
            if p["boxes"].shape[0] == 0 or g["boxes"].shape[0] == 0:
                continue
            # greedy score-ordered matching, class-aware, IoU >= order_iou_thr
            order = np.argsort(-p["scores"])
            ious = _iou_matrix(p["boxes"], g["boxes"])
            gt_taken = np.zeros(g["boxes"].shape[0], dtype=bool)
            matched = []  # (pred_order, gt_order, gt_id)
            for di in order:
                cand = ious[di].copy()
                cand[gt_taken] = -1.0
                cand[g["labels"] != p["labels"][di]] = -1.0
                j = int(np.argmax(cand))
                if cand[j] >= self.order_iou_thr:
                    gt_taken[j] = True
                    matched.append((float(p["order"][di]), float(g["order"][j]), j))
            if not matched:
                continue
            # sequences of GT ids ordered by predicted / by ground-truth order
            hyp = [gid for _, _, gid in sorted(matched, key=lambda t: t[0])]
            ref = [gid for _, _, gid in sorted(matched, key=lambda t: t[1])]
            total_ed += _levenshtein(hyp, ref)
            total_len += len(ref)
        if total_len == 0:
            return float("nan"), 0
        return 1.0 - total_ed / total_len, total_len

    # ----------------------------------------------------------------------------------
    def compute(self) -> dict:
        """Return ``{'mAP', 'AP50', 'AP75', 'order_score', 'num_images', 'per_class_AP'}``."""
        per_class_ap = np.full((self.num_classes, self.iou_thrs.size), np.nan)
        for c in range(self.num_classes):
            aps, _ = self._ap_per_class(c)
            per_class_ap[c] = aps

        valid = ~np.isnan(per_class_ap[:, 0])
        if valid.any():
            mAP = float(np.nanmean(per_class_ap[valid]))
            i50 = int(np.argmin(np.abs(self.iou_thrs - 0.5)))
            i75 = int(np.argmin(np.abs(self.iou_thrs - 0.75)))
            ap50 = float(np.nanmean(per_class_ap[valid, i50]))
            ap75 = float(np.nanmean(per_class_ap[valid, i75]))
        else:
            mAP = ap50 = ap75 = 0.0

        order_score, n_items = self._order_score()
        return {
            "mAP": mAP,
            "AP50": ap50,
            "AP75": ap75,
            "order_score": order_score,
            "order_items": n_items,
            "num_images": len(self._gts),
            "per_class_AP": {c: (float(np.nanmean(per_class_ap[c]))
                                 if valid[c] else float("nan"))
                             for c in range(self.num_classes)},
        }
