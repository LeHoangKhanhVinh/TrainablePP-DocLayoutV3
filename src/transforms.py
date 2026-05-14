"""PaddleDetection data transforms — pure numpy / cv2 / PIL ports.

Faithful reimplementations of the operators used by PP-DocLayoutV3's
``TrainReader`` / ``EvalReader`` (``PaddleDetection/ppdet/data/transform/``):

    sample transforms : Poly2MaskPack, RandomDistort, UpdateBBoxFromMask,
                        RandomExpand (+ Pad), RandomCrop
    batch transforms  : BatchRandomResize, UnpackMask, NormalizeImage,
                        NormalizeBox, BboxXYXY2XYWH
    eval transforms   : Poly2Mask, Resize, NormalizeImage

No ``paddle`` / ``pycocotools`` dependency — polygons are rasterised with
``cv2.fillPoly``.

Each operator takes/returns a ``sample`` dict with (a subset of) the keys:
``image`` (H,W,3), ``gt_bbox`` (N,4 xyxy abs), ``gt_class`` (N,1),
``gt_poly`` (list of polygons), ``gt_read_order`` (N,), ``im_shape`` [h,w],
``scale_factor`` [sy,sx], and — after Poly2MaskPack / Poly2Mask —
``gt_segm`` (masks), ``pack_indices`` / ``instance_ids`` (packed only).
"""

from __future__ import annotations

import random
from numbers import Number

import cv2
import numpy as np
from PIL import Image, ImageEnhance

__all__ = [
    "Poly2MaskPack", "Poly2Mask", "RandomDistort", "UpdateBBoxFromMask", "Pad",
    "RandomExpand", "RandomCrop", "Resize", "BatchRandomResize", "UnpackMask",
    "NormalizeImage", "NormalizeBox", "BboxXYXY2XYWH", "Permute",
    "Compose", "BatchCompose", "build_sample_transforms", "build_batch_transforms",
]

_INTERPS = [
    cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
    cv2.INTER_CUBIC, cv2.INTER_LANCZOS4,
]


def _poly2mask(poly, h: int, w: int) -> np.ndarray:
    """Rasterise one instance's polygon(s) to a binary (h, w) uint8 mask."""
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    parts = []
    for part in poly:
        pts = np.asarray(part, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            parts.append(np.round(pts).astype(np.int32))
    if parts:
        cv2.fillPoly(mask, parts, 1)
    return mask


# --------------------------------------------------------------------------------------
# polygon -> mask
# --------------------------------------------------------------------------------------
class Poly2MaskPack:
    """Pack non-overlapping instance masks into shared int16 images (perf op).

    Port of ``ppdet/data/transform/operators.py::Poly2MaskPack``.
    """

    def __init__(self, del_poly: bool = False, max_instances_per_pack: int | None = None):
        self.del_poly = del_poly
        self.max_instances_per_pack = max_instances_per_pack

    def _pack(self, gt_polys, bboxes, im_h, im_w):
        n = len(gt_polys)
        if n == 0:
            return [], np.zeros(0, np.int32), np.zeros(0, np.int32)
        im_h, im_w = int(im_h), int(im_w)
        packed_masks, packed_bboxes, packed_inst_ids = [], [], []
        pack_indices = np.zeros(n, np.int32)
        instance_ids = np.zeros(n, np.int32)

        for idx, (gt_poly, bbox) in enumerate(zip(gt_polys, bboxes)):
            curr = _poly2mask(gt_poly, im_h, im_w)
            x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2, y2 = min(im_w, int(bbox[2])), min(im_h, int(bbox[3]))
            if x2 <= x1 or y2 <= y1 or curr.sum() == 0:
                continue
            placed = False
            for pack_idx, (pm, pb, pids) in enumerate(
                zip(packed_masks, packed_bboxes, packed_inst_ids)
            ):
                if (self.max_instances_per_pack is not None
                        and len(pids) >= self.max_instances_per_pack):
                    continue
                curr_region = curr[y1:y2, x1:x2]
                packed_region = pm[y1:y2, x1:x2]
                if not np.any((curr_region > 0) & (packed_region > 0)):
                    next_id = len(pids) + 1
                    packed_region[curr_region > 0] = next_id
                    pb.append(bbox)
                    pids.append(next_id)
                    pack_indices[idx] = pack_idx
                    instance_ids[idx] = next_id
                    placed = True
                    break
            if not placed:
                new_pm = np.zeros((im_h, im_w), dtype=np.int16)
                new_pm[curr > 0] = 1
                packed_masks.append(new_pm)
                packed_bboxes.append([bbox])
                packed_inst_ids.append([1])
                pack_indices[idx] = len(packed_masks) - 1
                instance_ids[idx] = 1
        return packed_masks, pack_indices, instance_ids

    def __call__(self, sample):
        im_h, im_w = sample["im_shape"]
        packed, pack_indices, instance_ids = self._pack(
            sample["gt_poly"], sample["gt_bbox"], im_h, im_w
        )
        if packed:
            sample["gt_segm"] = np.stack(packed, axis=0).astype(np.int16)
        else:
            sample["gt_segm"] = np.zeros((0, int(im_h), int(im_w)), dtype=np.int16)
        sample["pack_indices"] = pack_indices
        sample["instance_ids"] = instance_ids
        if self.del_poly:
            sample.pop("gt_poly", None)
        return sample


class Poly2Mask:
    """Rasterise polygons to per-instance (N, H, W) uint8 masks (unpacked)."""

    def __init__(self, del_poly: bool = False):
        self.del_poly = del_poly

    def __call__(self, sample):
        im_h, im_w = sample["im_shape"]
        polys = sample.get("gt_poly", [])
        if polys:
            masks = np.stack([_poly2mask(p, im_h, im_w) for p in polys], axis=0)
        else:
            masks = np.zeros((0, int(im_h), int(im_w)), dtype=np.uint8)
        sample["gt_segm"] = masks
        if self.del_poly:
            sample.pop("gt_poly", None)
        return sample


# --------------------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------------------
class RandomDistort:
    """Random brightness / contrast / saturation / hue — port of ``RandomDistort``."""

    def __init__(self, hue=(-18, 18, 0.5), saturation=(0.5, 1.5, 0.5),
                 contrast=(0.5, 1.5, 0.5), brightness=(0.5, 1.5, 0.5),
                 random_apply=True, count=4, random_channel=False, prob=1.0):
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel
        self.prob = prob

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        arr = np.array(img.convert("HSV"))
        arr[:, :, 0] = arr[:, :, 0] + delta
        return Image.fromarray(arr, mode="HSV").convert("RGB")

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        return ImageEnhance.Color(img).enhance(np.random.uniform(low, high))

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        return ImageEnhance.Contrast(img).enhance(np.random.uniform(low, high))

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        return ImageEnhance.Brightness(img).enhance(np.random.uniform(low, high))

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample
        img = Image.fromarray(sample["image"].astype(np.uint8))
        if self.random_apply:
            funcs = [self.apply_brightness, self.apply_contrast,
                     self.apply_saturation, self.apply_hue]
            for func in np.random.permutation(funcs)[:self.count]:
                img = func(img)
            sample["image"] = np.asarray(img).astype(np.float32)
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)
        if mode:
            img = self.apply_contrast(img)
        img = self.apply_saturation(img)
        img = self.apply_hue(img)
        if not mode:
            img = self.apply_contrast(img)
        img = np.asarray(img).astype(np.float32)
        if self.random_channel and np.random.randint(0, 2):
            img = img[..., np.random.permutation(3)]
        sample["image"] = img
        return sample


# --------------------------------------------------------------------------------------
# bbox-from-mask
# --------------------------------------------------------------------------------------
def _bbox_from_mask(mask: np.ndarray):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.float32)


class UpdateBBoxFromMask:
    """Recompute bboxes from (possibly transformed) masks — port of ``UpdateBBoxFromMask``."""

    _FILTER_KEYS = ("gt_class", "gt_score", "is_crowd", "difficult", "gt_areas", "gt_read_order")

    def __init__(self, filter_empty: bool = True):
        self.filter_empty = filter_empty

    def __call__(self, sample):
        if "gt_segm" not in sample or len(sample["gt_segm"]) == 0:
            sample["gt_bbox"] = np.zeros((0, 4), np.float32)
            return sample

        is_packed = "pack_indices" in sample and "instance_ids" in sample
        new_bboxes, valid = [], []
        if is_packed:
            pack_indices = sample["pack_indices"]
            instance_ids = sample["instance_ids"]
            n = len(pack_indices)
            for i in range(n):
                inst = (sample["gt_segm"][pack_indices[i]] == instance_ids[i]).astype(np.uint8)
                bbox = _bbox_from_mask(inst)
                if bbox is not None:
                    new_bboxes.append(bbox)
                    valid.append(i)
                elif not self.filter_empty:
                    new_bboxes.append(np.zeros(4, np.float32))
                    valid.append(i)
        else:
            n = len(sample["gt_segm"])
            for i in range(n):
                inst = sample["gt_segm"][i]
                if inst.dtype != np.uint8:
                    inst = inst.astype(np.uint8)
                bbox = _bbox_from_mask(inst)
                if bbox is not None:
                    new_bboxes.append(bbox)
                    valid.append(i)
                elif not self.filter_empty:
                    new_bboxes.append(np.zeros(4, np.float32))
                    valid.append(i)

        if len(valid) > 0:
            sample["gt_bbox"] = np.asarray(new_bboxes, dtype=np.float32)
            if self.filter_empty and len(valid) < n:
                valid = np.asarray(valid)
                if is_packed:
                    sample["pack_indices"] = sample["pack_indices"][valid]
                    sample["instance_ids"] = sample["instance_ids"][valid]
                else:
                    sample["gt_segm"] = sample["gt_segm"][valid]
                for k in self._FILTER_KEYS:
                    if k in sample and len(sample[k]) > 0:
                        sample[k] = sample[k][valid]
                if "gt_poly" in sample and len(sample["gt_poly"]) > 0:
                    sample["gt_poly"] = [sample["gt_poly"][i] for i in valid]
        else:
            h, w = sample["gt_segm"].shape[1:]
            if is_packed:
                sample["gt_segm"] = np.zeros((0, h, w), np.int16)
                sample["pack_indices"] = np.zeros(0, np.int32)
                sample["instance_ids"] = np.zeros(0, np.int32)
            else:
                sample["gt_segm"] = np.zeros((0, h, w), np.uint8)
            sample["gt_bbox"] = np.zeros((0, 4), np.float32)
            for k in self._FILTER_KEYS:
                if k in sample:
                    sample[k] = sample[k][:0]
        return sample


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
class Pad:
    """Pad image / bbox / packed-or-unpacked masks — port of ``Pad`` (pad_mode -1/0/1/2)."""

    def __init__(self, size=None, size_divisor=32, pad_mode=0, offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        if isinstance(size, int):
            size = [size, size]
        assert pad_mode in (-1, 0, 1, 2)
        if pad_mode == -1:
            assert offsets is not None
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.offsets = offsets
        self.fill_value = tuple(fill_value)

    def __call__(self, sample):
        im = sample["image"]
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)
        if h == im_h and w == im_w:
            sample["image"] = im.astype(np.float32)
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_x, offset_y = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        canvas = np.ones((h, w, 3), dtype=np.float32) * np.array(self.fill_value, np.float32)
        canvas[offset_y:offset_y + im_h, offset_x:offset_x + im_w, :] = im.astype(np.float32)
        sample["image"] = canvas
        if self.pad_mode == 0:
            return sample

        if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
            sample["gt_bbox"] = sample["gt_bbox"] + np.array(
                [offset_x, offset_y] * 2, dtype=np.float32
            )
        if "gt_segm" in sample and len(sample["gt_segm"]) > 0:
            dtype = sample["gt_segm"].dtype
            masks = [
                cv2.copyMakeBorder(
                    m, offset_y, h - (offset_y + im_h), offset_x, w - (offset_x + im_w),
                    borderType=cv2.BORDER_CONSTANT, value=0,
                )
                for m in sample["gt_segm"]
            ]
            sample["gt_segm"] = np.asarray(masks, dtype=dtype)
        return sample


class RandomExpand:
    """Randomly expand the canvas — port of ``RandomExpand``."""

    def __init__(self, ratio=4.0, prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        assert ratio > 1.01
        self.ratio = ratio
        self.prob = prob
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        self.fill_value = tuple(fill_value)

    def __call__(self, sample):
        if np.random.uniform(0., 1.) < self.prob:
            return sample
        im = sample["image"]
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h, w = int(height * ratio), int(width * ratio)
        if h <= height or w <= width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        return Pad([h, w], pad_mode=-1, offsets=[x, y], fill_value=self.fill_value)(sample)


class RandomCrop:
    """Random crop of image / packed-or-unpacked masks / bboxes — port of ``RandomCrop``.

    Only the non-``is_mask_crop`` path is ported (PP-DocLayoutV3 deletes ``gt_poly``
    via ``Poly2MaskPack(del_poly=True)`` and never sets ``is_mask_crop``).
    """

    def __init__(self, aspect_ratio=(0.5, 2.0), thresholds=(0., .1, .3, .5, .7, .9),
                 scaling=(0.3, 1.0), num_attempts=50, allow_no_crop=True,
                 cover_all_box=False, ioumode="iou", prob=1.0,
                 use_box_candidates=False, wh_thr=4, ar_thr=40, area_thr=0.15):
        self.aspect_ratio = list(aspect_ratio) if aspect_ratio is not None else None
        self.thresholds = list(thresholds)
        self.scaling = list(scaling)
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.ioumode = ioumode
        self.prob = prob
        self.use_box_candidates = use_box_candidates
        self.wh_thr = wh_thr
        self.ar_thr = ar_thr
        self.area_thr = area_thr

    @staticmethod
    def _iou_matrix(a, b):
        tl = np.maximum(a[:, None, :2], b[:, :2])
        br = np.minimum(a[:, None, 2:], b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b - area_i + 1e-10)

    @staticmethod
    def _gtcropiou_matrix(a, b):
        tl = np.maximum(a[:, None, :2], b[:, :2])
        br = np.minimum(a[:, None, 2:], b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / (area_a[:, None] + 1e-10)

    def _box_candidates(self, box1, box2, eps=1e-16):
        w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
        not_cropped = (np.abs(w2 - w1) < 1.0) & (np.abs(h2 - h1) < 1.0)
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        base_valid = (w2 > self.wh_thr) & (h2 > self.wh_thr) & \
                     (w2 * h2 / (w1 * h1 + eps) > self.area_thr)
        ar_valid = not_cropped | (ar < self.ar_thr)
        return base_valid & ar_valid

    def _crop_box_center(self, box, crop):
        cropped = box.copy()
        cropped[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped[:, :2] -= crop[:2]
        cropped[:, 2:] -= crop[:2]
        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers, centers < crop[2:]).all(axis=1)
        valid = np.logical_and(valid, (cropped[:, :2] < cropped[:, 2:]).all(axis=1))
        return cropped, np.where(valid)[0]

    def _crop_box_candidates(self, box, crop):
        cropped_abs = box.copy()
        cropped_abs[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_abs[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped = cropped_abs.copy()
        cropped[:, :2] -= crop[:2]
        cropped[:, 2:] -= crop[:2]
        valid = self._box_candidates(box, cropped_abs)
        valid = valid & ((cropped[:, 2] > cropped[:, 0]) & (cropped[:, 3] > cropped[:, 1]))
        return cropped, np.where(valid)[0]

    @staticmethod
    def _crop_packed_masks(gt_segm, pack_indices, instance_ids, crop, valid_ids):
        x1, y1, x2, y2 = crop
        cropped = gt_segm[:, y1:y2, x1:x2].copy()
        new_pack = pack_indices[valid_ids].copy()
        new_inst = instance_ids[valid_ids].copy()
        valid_set = set(zip(new_pack.tolist(), new_inst.tolist()))
        for pack_idx in range(len(cropped)):
            pm = cropped[pack_idx]
            for inst_id in np.unique(pm):
                if inst_id == 0:
                    continue
                if (pack_idx, int(inst_id)) not in valid_set:
                    pm[pm == inst_id] = 0
        return cropped, new_pack, new_inst

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample
        if "gt_bbox" not in sample or len(sample["gt_bbox"]) == 0:
            return sample

        h, w = sample["image"].shape[:2]
        gt_bbox = sample["gt_bbox"]
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append("no_crop")
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == "no_crop":
                return sample
            found = False
            crop_box = valid_ids = cropped_box = None
            for _ in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    ar = np.random.uniform(max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(ar)
                    w_scale = scale * np.sqrt(ar)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h, crop_w = int(h * h_scale), int(w * w_scale)
                if self.aspect_ratio is None and (crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0):
                    continue
                if crop_h >= h or crop_w >= w or crop_h <= 0 or crop_w <= 0:
                    continue
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                cb = np.array([crop_box], dtype=np.float32)
                iou = (self._gtcropiou_matrix(gt_bbox, cb) if self.ioumode == "iof"
                       else self._iou_matrix(gt_bbox, cb))
                if iou.max() < thresh:
                    continue
                if self.cover_all_box and iou.min() < thresh:
                    continue
                if self.use_box_candidates:
                    cropped_box, valid_ids = self._crop_box_candidates(
                        gt_bbox, np.array(crop_box, np.float32))
                else:
                    cropped_box, valid_ids = self._crop_box_center(
                        gt_bbox, np.array(crop_box, np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                is_packed = "pack_indices" in sample and "instance_ids" in sample
                if "gt_segm" in sample and len(sample["gt_segm"]) > 0:
                    if is_packed:
                        (sample["gt_segm"], sample["pack_indices"],
                         sample["instance_ids"]) = self._crop_packed_masks(
                            sample["gt_segm"], sample["pack_indices"],
                            sample["instance_ids"], crop_box, valid_ids)
                    else:
                        x1, y1, x2, y2 = crop_box
                        sample["gt_segm"] = np.take(
                            sample["gt_segm"][:, y1:y2, x1:x2], valid_ids, axis=0)
                x1, y1, x2, y2 = crop_box
                sample["image"] = sample["image"][y1:y2, x1:x2, :]
                sample["gt_bbox"] = np.take(cropped_box, valid_ids, axis=0)
                sample["gt_class"] = np.take(sample["gt_class"], valid_ids, axis=0)
                for k in ("gt_score", "is_crowd", "difficult", "gt_read_order"):
                    if k in sample:
                        sample[k] = np.take(sample[k], valid_ids, axis=0)
                return sample
        return sample


class Resize:
    """Resize image / bbox / masks — port of ``Resize`` (keep_ratio supported)."""

    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = list(target_size)
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, sample):
        im = sample["image"]
        im_h, im_w = im.shape[:2]
        if self.keep_ratio:
            tmin, tmax = min(self.target_size), max(self.target_size)
            im_scale = min(tmin / min(im_h, im_w), tmax / max(im_h, im_w))
            resize_h = int(im_scale * im_h + 0.5)
            resize_w = int(im_scale * im_w + 0.5)
        else:
            resize_h, resize_w = self.target_size
        scale_x = resize_w / im_w
        scale_y = resize_h / im_h

        sample["image"] = cv2.resize(
            im, None, None, fx=scale_x, fy=scale_y, interpolation=self.interp
        ).astype(np.float32)
        sample["im_shape"] = np.asarray([resize_h, resize_w], dtype=np.float32)
        prev = sample.get("scale_factor", np.array([1., 1.], np.float32))
        sample["scale_factor"] = np.asarray(
            [prev[0] * scale_y, prev[1] * scale_x], dtype=np.float32
        )

        if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
            bbox = sample["gt_bbox"]
            bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_x, 0, resize_w)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_y, 0, resize_h)
            sample["gt_bbox"] = bbox
        if "gt_poly" in sample and len(sample["gt_poly"]) > 0:
            new_poly = []
            for inst in sample["gt_poly"]:
                parts = []
                for part in inst:
                    p = np.asarray(part, dtype=np.float32).copy()
                    p[0::2] *= scale_x
                    p[1::2] *= scale_y
                    parts.append(p.tolist())
                new_poly.append(parts)
            sample["gt_poly"] = new_poly
        if "gt_segm" in sample and len(sample["gt_segm"]) > 0:
            dtype = sample["gt_segm"].dtype
            masks = [
                cv2.resize(m, None, None, fx=scale_x, fy=scale_y,
                           interpolation=cv2.INTER_NEAREST)
                for m in sample["gt_segm"]
            ]
            sample["gt_segm"] = np.asarray(masks, dtype=dtype)
        return sample


class BatchRandomResize:
    """Pick one random target size (+ optional random interp) for a whole batch."""

    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_NEAREST,
                 random_size=True, random_interp=False):
        if random_size and not isinstance(target_size, (list, tuple)):
            raise TypeError("target_size must be a list when random_size=True")
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples):
        if self.random_size:
            target_size = self.target_size[np.random.choice(len(self.target_size))]
        else:
            target_size = self.target_size
        interp = np.random.choice(_INTERPS) if self.random_interp else self.interp
        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=int(interp))
        return [resizer(s) for s in samples]


# --------------------------------------------------------------------------------------
# mask unpack + normalisation
# --------------------------------------------------------------------------------------
class UnpackMask:
    """Unpack packed masks to (N, H, W) binary masks + recompute bboxes."""

    _FILTER_KEYS = ("gt_class", "gt_score", "is_crowd", "difficult", "gt_areas", "gt_read_order")

    def __init__(self, remove_pack_info: bool = True, compute_bbox: bool = True):
        self.remove_pack_info = remove_pack_info
        self.compute_bbox = compute_bbox

    def __call__(self, sample):
        if "gt_segm" not in sample or len(sample["gt_segm"]) == 0:
            h = int(sample.get("im_shape", [1, 1])[0])
            w = int(sample.get("im_shape", [1, 1])[1])
            sample["gt_segm"] = np.zeros((0, max(h, 1), max(w, 1)), np.uint8)
            if self.compute_bbox:
                sample["gt_bbox"] = np.zeros((0, 4), np.float32)
            sample.pop("pack_indices", None)
            sample.pop("instance_ids", None)
            return sample

        if "pack_indices" not in sample:  # already unpacked (e.g. Poly2Mask path)
            return sample

        packed = sample["gt_segm"]
        pack_indices = sample["pack_indices"]
        instance_ids = sample["instance_ids"]
        n = len(pack_indices)
        h, w = packed.shape[1], packed.shape[2]

        masks, bboxes, valid = [], [], []
        for i in range(n):
            inst = (packed[pack_indices[i]] == instance_ids[i]).astype(np.uint8)
            if self.compute_bbox:
                bbox = _bbox_from_mask(inst)
                if bbox is None:
                    continue
                bboxes.append(bbox)
            masks.append(inst)
            valid.append(i)

        if masks:
            sample["gt_segm"] = np.stack(masks, axis=0)
        else:
            sample["gt_segm"] = np.zeros((0, h, w), np.uint8)
        if self.compute_bbox:
            sample["gt_bbox"] = (np.asarray(bboxes, np.float32) if bboxes
                                 else np.zeros((0, 4), np.float32))
        if len(valid) < n:
            valid = np.asarray(valid, dtype=np.int64)
            for k in self._FILTER_KEYS:
                if k in sample and len(sample[k]) > 0:
                    sample[k] = sample[k][valid]
            if "gt_poly" in sample and len(sample["gt_poly"]) > 0:
                sample["gt_poly"] = [sample["gt_poly"][i] for i in valid]
        if self.remove_pack_info:
            sample.pop("pack_indices", None)
            sample.pop("instance_ids", None)
        return sample


class NormalizeImage:
    """Port of ``NormalizeImage`` — optional /255 rescale + optional mean/std."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 is_scale=True, norm_type="mean_std"):
        assert norm_type in ("mean_std", "none")
        self.mean = list(mean)
        self.std = list(std)
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, sample):
        im = sample["image"].astype(np.float32, copy=False)
        if self.is_scale:
            im = im * (1.0 / 255.0)
        if self.norm_type == "mean_std":
            im = im - np.array(self.mean, np.float32)[None, None, :]
            im = im / np.array(self.std, np.float32)[None, None, :]
        sample["image"] = im
        return sample


class NormalizeBox:
    """Port of ``NormalizeBox`` — bbox coords -> [0, 1] using the current image size."""

    def __call__(self, sample):
        if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
            h, w = sample["image"].shape[:2]
            bbox = sample["gt_bbox"].astype(np.float32).copy()
            bbox[:, 0::2] /= w
            bbox[:, 1::2] /= h
            sample["gt_bbox"] = bbox
        return sample


class BboxXYXY2XYWH:
    """Port of ``BboxXYXY2XYWH`` — xyxy -> cxcywh (in place)."""

    def __call__(self, sample):
        if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
            bbox = sample["gt_bbox"].astype(np.float32).copy()
            bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
            bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.0
            sample["gt_bbox"] = bbox
        return sample


class Permute:
    """HWC -> CHW (port of ``Permute``)."""

    def __call__(self, sample):
        sample["image"] = np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
        return sample


# --------------------------------------------------------------------------------------
# composition + builders
# --------------------------------------------------------------------------------------
class Compose:
    """Apply a list of per-sample transforms in order."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class BatchCompose:
    """Apply batch-level transforms (a list of samples in, a list out)."""

    def __init__(self, batch_transforms, sample_transforms=None):
        self.batch_transforms = list(batch_transforms)
        # transforms that run per-sample *after* the batch op (UnpackMask, Normalize…)
        self.sample_transforms = list(sample_transforms or [])

    def __call__(self, samples):
        for t in self.batch_transforms:
            samples = t(samples)
        for i, s in enumerate(samples):
            for t in self.sample_transforms:
                s = t(s)
            samples[i] = s
        return samples


# A transform is "batch-level" if it consumes/returns a list of samples.
_BATCH_OPS = {"BatchRandomResize"}

_REGISTRY = {
    "Poly2MaskPack": Poly2MaskPack,
    "Poly2Mask": Poly2Mask,
    "RandomDistort": RandomDistort,
    "UpdateBBoxFromMask": UpdateBBoxFromMask,
    "Pad": Pad,
    "RandomExpand": RandomExpand,
    "RandomCrop": RandomCrop,
    "Resize": Resize,
    "BatchRandomResize": BatchRandomResize,
    "UnpackMask": UnpackMask,
    "NormalizeImage": NormalizeImage,
    "NormalizeBox": NormalizeBox,
    "BboxXYXY2XYWH": BboxXYXY2XYWH,
    "Permute": Permute,
}


def _build_one(spec: dict):
    """``spec`` is a one-key mapping ``{OpName: {kwargs}}`` (PaddleDetection style)."""
    assert isinstance(spec, dict) and len(spec) == 1, f"bad transform spec: {spec!r}"
    name, kwargs = next(iter(spec.items()))
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform {name!r}; known: {sorted(_REGISTRY)}")
    return name, _REGISTRY[name](**(kwargs or {}))


def build_sample_transforms(specs: list[dict] | None) -> Compose:
    """Build the per-sample :class:`Compose` from a list of ``{Op: kwargs}`` dicts."""
    return Compose([_build_one(s)[1] for s in (specs or [])])


def build_batch_transforms(specs: list[dict] | None) -> BatchCompose:
    """Build the :class:`BatchCompose` — splits batch-level ops from per-sample ones."""
    batch_ops, sample_ops = [], []
    for spec in (specs or []):
        name, op = _build_one(spec)
        (batch_ops if name in _BATCH_OPS else sample_ops).append(op)
    return BatchCompose(batch_ops, sample_ops)
