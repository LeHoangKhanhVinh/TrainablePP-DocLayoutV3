"""LabelMe-format dataset for PP-DocLayoutV3 fine-tuning.

Each JSON file follows the LabelMe schema (``imagePath``, ``imageWidth``,
``imageHeight``, ``shapes``). Per shape we expect ``label``, ``points`` (polygon
vertices in original-image coords), and ``reading_order`` (an int).
``shape_type == "linestrip"`` shapes (the curved reading-order traces) are
ignored — they are not class instances.

``__getitem__`` builds a PaddleDetection-style ``sample`` dict and runs the
configured *sample* transforms (see :mod:`src.transforms`); the *batch*
transforms (``BatchRandomResize`` + ``UnpackMask`` + normalisation) run inside
:class:`Collate`. This mirrors PaddleDetection's ``TrainReader`` /
``EvalReader`` split between ``sample_transforms`` and ``batch_transforms``.

If the image referenced by ``imagePath`` is missing on disk, we synthesize a
blank white image at the original ``(imageWidth, imageHeight)`` resolution.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import DEFAULT_LABEL_MAP, LabelMap
from .transforms import (
    BatchCompose,
    Compose,
    NormalizeImage,
    Permute,
    Poly2Mask,
    Resize,
)

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE = 800


def _default_eval_transforms(image_size: int) -> Compose:
    """EvalReader-equivalent pipeline (also rasterises masks for the eval loss)."""
    return Compose([
        Poly2Mask(del_poly=True),
        Resize(target_size=[image_size, image_size], keep_ratio=False, interp=2),
        NormalizeImage(mean=[0., 0., 0.], std=[1., 1., 1.], norm_type="none"),
        Permute(),
    ])


class LabelmeLayoutDataset(Dataset):
    """Reads LabelMe JSONs and returns PaddleDetection-style ``sample`` dicts.

    Each item is a dict with at least ``image`` (numpy, H,W,3 or 3,H,W after
    ``Permute``), ``gt_bbox``, ``gt_class``, ``gt_read_order``, ``im_shape`` and —
    after ``Poly2Mask`` / ``Poly2MaskPack`` — ``gt_segm`` (+ packing metadata).
    :class:`Collate` turns these into model-ready tensors.
    """

    def __init__(
        self,
        root: str,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        label_map: LabelMap | None = None,
        sample_transforms: Compose | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.sample_transforms = (
            sample_transforms if sample_transforms is not None
            else _default_eval_transforms(image_size)
        )
        self.json_paths = sorted(glob(os.path.join(root, "*.json")))
        if not self.json_paths:
            raise FileNotFoundError(f"No LabelMe *.json files found under {root!r}")
        self._unknown_labels_seen: set[str] = set()

    def __len__(self) -> int:
        return len(self.json_paths)

    def _load_or_synthesize_image(self, image_path: str | None, w: int, h: int) -> np.ndarray:
        if image_path:
            candidate = (
                image_path if os.path.isabs(image_path)
                else os.path.join(self.root, image_path)
            )
            if os.path.isfile(candidate):
                try:
                    return np.asarray(Image.open(candidate).convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read %s (%s); using synthetic image",
                                   candidate, exc)
        return np.full((h, w, 3), 255, dtype=np.uint8)

    def _build_sample(self, idx: int) -> dict:
        path = self.json_paths[idx]
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        orig_w = int(j["imageWidth"])
        orig_h = int(j["imageHeight"])
        image = self._load_or_synthesize_image(j.get("imagePath"), orig_w, orig_h)
        # Guard against JSON/image size mismatch.
        orig_h, orig_w = image.shape[:2]

        class_ids: list[int] = []
        boxes_xyxy: list[list[float]] = []
        polygons: list[list[list[float]]] = []
        order_values: list[int] = []

        for shape in j.get("shapes", []):
            if shape.get("shape_type", "polygon") == "linestrip":
                continue
            raw_label = shape.get("label", "")
            label = self.label_map.normalize(raw_label)
            if label is None:
                if raw_label not in self._unknown_labels_seen:
                    logger.warning("Skipping unknown label %r in %s", raw_label, path)
                    self._unknown_labels_seen.add(raw_label)
                continue

            pts = np.asarray(shape.get("points", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                continue
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            if x_max <= x_min or y_max <= y_min:
                continue

            class_ids.append(self.label_map.label2id[label])
            boxes_xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
            polygons.append([pts.reshape(-1).tolist()])  # one flat part per instance
            order_values.append(int(shape.get("reading_order", len(order_values))))

        n = len(class_ids)
        # Dense ranks: relative reading order is all the loss needs.
        if n:
            order_arr = np.asarray(order_values, dtype=np.int64)
            sorted_idx = np.argsort(order_arr, kind="stable")
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(n, dtype=np.int64)
        else:
            ranks = np.zeros((0,), dtype=np.int64)

        return {
            "image": image,
            "im_shape": np.asarray([orig_h, orig_w], dtype=np.float32),
            "scale_factor": np.asarray([1.0, 1.0], dtype=np.float32),
            "gt_class": np.asarray(class_ids, dtype=np.int64).reshape(-1, 1),
            "gt_bbox": np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4),
            "gt_poly": polygons,
            "gt_read_order": ranks,
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self._build_sample(idx)
        return self.sample_transforms(sample)


def _finalize(sample: dict) -> dict:
    """Turn a fully-transformed ``sample`` dict into model-ready tensors."""
    img = np.asarray(sample["image"], dtype=np.float32)
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:  # HWC -> CHW
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
    pixel_values = torch.from_numpy(np.ascontiguousarray(img)).float()
    _, h, w = pixel_values.shape

    gt_class = np.asarray(sample.get("gt_class", np.zeros((0, 1)))).reshape(-1)
    class_labels = torch.as_tensor(gt_class, dtype=torch.long)

    gt_bbox = np.asarray(sample.get("gt_bbox", np.zeros((0, 4))), dtype=np.float32)
    boxes = torch.as_tensor(gt_bbox, dtype=torch.float32).reshape(-1, 4)

    segm = sample.get("gt_segm", None)
    if segm is None or len(segm) == 0:
        masks = torch.zeros((boxes.shape[0], h, w), dtype=torch.float32)
    else:
        masks = torch.as_tensor(np.asarray(segm), dtype=torch.float32)

    order = np.asarray(sample.get("gt_read_order", np.zeros((boxes.shape[0],)))).reshape(-1)
    order_rank = torch.as_tensor(order, dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "labels": {
            "class_labels": class_labels,
            "boxes": boxes,
            "masks": masks,
            "order_rank": order_rank,
        },
    }


class Collate:
    """Collate function: runs the batch transforms, then stacks into tensors.

    ``batch_transforms`` is a :class:`~src.transforms.BatchCompose` (may be None,
    in which case samples are only finalised + stacked).
    """

    def __init__(self, batch_transforms: BatchCompose | None = None):
        self.batch_transforms = batch_transforms

    def __call__(self, batch: list[dict]) -> dict:
        if self.batch_transforms is not None:
            batch = self.batch_transforms(list(batch))
        finalized = [_finalize(s) for s in batch]
        pixel_values = torch.stack([f["pixel_values"] for f in finalized], dim=0)
        labels = [f["labels"] for f in finalized]
        return {"pixel_values": pixel_values, "labels": labels}


# Default collate (no batch transforms) — used by eval, where the per-sample
# eval pipeline already produces fixed-size, CHW, normalised images.
collate_fn = Collate(batch_transforms=None)


__all__ = ["LabelmeLayoutDataset", "Collate", "collate_fn"]
