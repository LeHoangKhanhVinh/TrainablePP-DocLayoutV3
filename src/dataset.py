"""LabelMe-format dataset for PP-DocLayoutV3 fine-tuning.

Each JSON file follows the LabelMe schema (``imagePath``, ``imageWidth``,
``imageHeight``, ``shapes``). Per shape we expect ``label``, ``points``
(polygon vertices in original-image coords), and ``reading_order`` (an int).
``shape_type == "linestrip"`` shapes (the curved reading-order traces) are
ignored — they are not class instances.

If the image referenced by ``imagePath`` is missing on disk, we synthesize a
blank white PIL image at the original ``(imageWidth, imageHeight)`` resolution,
as requested.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import DEFAULT_LABEL_MAP, LabelMap

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE = 800
_DEFAULT_MASK_SIZE = 200  # = image_size / 4 (matches the mask FPN output stride)


class LabelmeLayoutDataset(Dataset):
    """Reads LabelMe JSONs and returns tensors ready for PP-DocLayoutV3 training.

    Each item is a dict::

        {
            "pixel_values": torch.float32 (3, H, W) in [0, 1],
            "labels": {
                "class_labels": torch.long  (N,),
                "boxes":        torch.float (N, 4)     # cx, cy, w, h normalized
                "masks":        torch.float (N, mH, mW),
                "order_rank":   torch.long  (N,),      # dense rank within the kept shapes
            },
        }
    """

    def __init__(
        self,
        root: str,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        mask_size: int = _DEFAULT_MASK_SIZE,
        label_map: LabelMap | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.mask_size = mask_size
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.json_paths = sorted(glob(os.path.join(root, "*.json")))
        if not self.json_paths:
            raise FileNotFoundError(f"No LabelMe *.json files found under {root!r}")
        self._unknown_labels_seen: set[str] = set()

    def __len__(self) -> int:
        return len(self.json_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.json_paths[idx]
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        orig_w = int(j["imageWidth"])
        orig_h = int(j["imageHeight"])
        image = self._load_or_synthesize_image(j.get("imagePath"), orig_w, orig_h)

        # Resize image to (image_size, image_size). Mirrors PPDocLayoutV3ImageProcessor:
        # bicubic resize + 1/255 rescale; image_mean=[0,0,0], image_std=[1,1,1] => no normalize.
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        class_ids: list[int] = []
        boxes_xyxy: list[np.ndarray] = []
        polygons: list[np.ndarray] = []
        order_values: list[int] = []

        for shape in j.get("shapes", []):
            shape_type = shape.get("shape_type", "polygon")
            if shape_type == "linestrip":
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

            # Scale to image_size space.
            pts_scaled = pts.copy()
            pts_scaled[:, 0] *= scale_x
            pts_scaled[:, 1] *= scale_y

            x_min, y_min = pts_scaled.min(axis=0)
            x_max, y_max = pts_scaled.max(axis=0)
            if x_max <= x_min or y_max <= y_min:
                continue

            class_ids.append(self.label_map.label2id[label])
            boxes_xyxy.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
            polygons.append(pts_scaled)
            order_values.append(int(shape.get("reading_order", len(order_values))))

        n = len(class_ids)
        if n == 0:
            class_labels = torch.zeros((0,), dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, self.mask_size, self.mask_size), dtype=torch.float32)
            order_rank = torch.zeros((0,), dtype=torch.long)
        else:
            class_labels = torch.tensor(class_ids, dtype=torch.long)

            xyxy = np.stack(boxes_xyxy, axis=0)  # (N, 4)
            cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5 / self.image_size
            cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5 / self.image_size
            w = (xyxy[:, 2] - xyxy[:, 0]) / self.image_size
            h = (xyxy[:, 3] - xyxy[:, 1]) / self.image_size
            boxes = torch.from_numpy(np.stack([cx, cy, w, h], axis=1).astype(np.float32))

            mask_scale = self.mask_size / self.image_size
            masks_np = np.zeros((n, self.mask_size, self.mask_size), dtype=np.float32)
            for i, poly in enumerate(polygons):
                poly_mask = (poly * mask_scale).astype(np.int32)
                cv2.fillPoly(masks_np[i], [poly_mask], 1.0)
            masks = torch.from_numpy(masks_np)

            # Dense ranks within this image: argsort of order_values gives positions in
            # increasing-order; inverting gives the dense rank for each shape.
            order_arr = np.asarray(order_values, dtype=np.int64)
            sorted_idx = np.argsort(order_arr, kind="stable")
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(n, dtype=np.int64)
            order_rank = torch.from_numpy(ranks)

        return {
            "pixel_values": pixel_values,
            "labels": {
                "class_labels": class_labels,
                "boxes": boxes,
                "masks": masks,
                "order_rank": order_rank,
            },
        }

    def _load_or_synthesize_image(self, image_path: str | None, w: int, h: int) -> Image.Image:
        if image_path:
            candidate = image_path if os.path.isabs(image_path) else os.path.join(self.root, image_path)
            if os.path.isfile(candidate):
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception as exc:
                    logger.warning("Failed to read %s (%s); using synthetic image", candidate, exc)
        # Fallback: blank white image at the JSON's declared resolution.
        return Image.new("RGB", (w, h), color=(255, 255, 255))


def collate_fn(batch: list[dict]) -> dict:
    """Stack ``pixel_values``; keep ``labels`` as a list of dicts (RT-DETR style)."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


__all__ = ["LabelmeLayoutDataset", "collate_fn"]
