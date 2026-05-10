"""Evaluate a fine-tuned PP-DocLayoutV3 model on a test set.

Two save modes:

* ``labelme`` — writes a LabelMe-format JSON for each test sample, with the
  predicted shapes (bbox polygons), ``label``, and ``reading_order``.
* ``image``  — writes an annotated image with predicted bboxes, class labels,
  and reading-order arrows linking detections in predicted order.
* ``both``   — emit both.

Usage::

    python eval.py --config configs/default.yaml --checkpoint ./checkpoints/best
    python eval.py --checkpoint ./checkpoints/best --test-data ./datasets/foo/test \\
                   --out ./eval_out --save-mode both --threshold 0.5
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src import (
    LabelMap,
    LabelmeLayoutDataset,
    PPDocLayoutV3Loss,
    TrainablePPDocLayoutV3ForObjectDetection,
    collate_fn,
)


DEFAULTS: dict[str, Any] = {
    "data": "./datasets/vbhc_deduplicated_labeled",
    "test_data": None,
    "checkpoint": "./checkpoints/best",
    "out": "./eval_out",
    "save_mode": "both",       # labelme | image | both
    "threshold": 0.5,
    "batch_size": 1,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "label_list": None,
    "label_aliases": None,
    "image_size": 800,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML config (reuses training keys).")
    p.add_argument("--data", default=None,
                   help="Dataset root. If it contains a test/ subfolder it's auto-used.")
    p.add_argument("--test-data", default=None, dest="test_data")
    p.add_argument("--checkpoint", default=None,
                   help="Fine-tuned checkpoint directory (e.g. ./checkpoints/best).")
    p.add_argument("--out", default=None, help="Output directory for predictions.")
    p.add_argument("--save-mode", default=None, dest="save_mode",
                   choices=["labelme", "image", "both"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Min score to keep a detection.")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--device", default=None)
    return p


def resolve_config(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        if not isinstance(yaml_data, dict):
            raise ValueError(f"{args.config!r} must be a YAML mapping at the top level")
        # Only pick up keys that exist in DEFAULTS so training-only keys (lr, epochs, …) are ignored.
        for k, v in yaml_data.items():
            if k in DEFAULTS:
                cfg[k] = v
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(cli)
    return cfg


def resolve_test_split(cfg: dict[str, Any]) -> str:
    test_path = cfg.get("test_data")
    data_root = cfg.get("data")
    if not test_path and data_root:
        candidate = os.path.join(data_root, "test")
        if os.path.isdir(candidate):
            test_path = candidate
    if not test_path:
        raise SystemExit(
            "Need a test split. Either set `test_data`, or point `data` at a folder "
            "containing a test/ subdir."
        )
    if not os.path.isdir(test_path):
        raise SystemExit(f"test_data not found: {test_path}")
    return test_path


def post_process(outputs, target_sizes, threshold: float):
    """Decode raw model outputs to per-image detections sorted by reading order.

    Returns a list (one dict per image) with keys: ``scores``, ``labels``,
    ``boxes`` (xyxy in original-image pixels), ``order_seq`` (rank, 0 = first).
    """
    boxes = outputs.pred_boxes  # (B, Q, 4) cxcywh, normalized
    logits = outputs.logits     # (B, Q, C)
    order_logits = outputs.order_logits  # (B, Q, Q)

    cx, cy, w, h = boxes.unbind(-1)
    boxes_xyxy = torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )

    target_sizes_t = torch.as_tensor(target_sizes, device=boxes.device, dtype=boxes_xyxy.dtype)
    img_h, img_w = target_sizes_t.unbind(1)
    scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
    boxes_xyxy = boxes_xyxy * scale

    # Pairwise -> per-query rank via the same voting scheme as the official processor.
    order_scores = torch.sigmoid(order_logits)
    order_votes = (
        order_scores.triu(diagonal=1).sum(dim=1)
        + (1.0 - order_scores.transpose(1, 2)).tril(diagonal=-1).sum(dim=1)
    )
    order_pointers = torch.argsort(order_votes, dim=1)
    order_seq = torch.empty_like(order_pointers)
    ranks = torch.arange(
        order_pointers.shape[1], device=order_pointers.device, dtype=order_pointers.dtype
    ).expand(*order_pointers.shape)
    order_seq.scatter_(1, order_pointers, ranks)

    num_queries = logits.shape[1]
    num_classes = logits.shape[2]
    scores_flat = logits.sigmoid().flatten(1)
    scores, index = torch.topk(scores_flat, num_queries, dim=-1)
    labels = index % num_classes
    query_idx = index // num_classes
    boxes_xyxy = boxes_xyxy.gather(1, query_idx.unsqueeze(-1).expand(-1, -1, 4))
    order_seq = order_seq.gather(1, query_idx)

    results = []
    for s, l, b, o in zip(scores, labels, boxes_xyxy, order_seq):
        keep = s >= threshold
        s_k, l_k, b_k, o_k = s[keep], l[keep], b[keep], o[keep]
        if s_k.numel() == 0:
            results.append({
                "scores": s_k.cpu(),
                "labels": l_k.cpu(),
                "boxes": b_k.cpu(),
                "order_seq": o_k.cpu(),
            })
            continue
        o_sorted, idx = torch.sort(o_k)
        # Re-rank to dense 0..N-1 so downstream consumers see contiguous ranks.
        dense = torch.arange(o_sorted.numel(), dtype=o_sorted.dtype)
        results.append({
            "scores": s_k[idx].cpu(),
            "labels": l_k[idx].cpu(),
            "boxes": b_k[idx].cpu(),
            "order_seq": dense,
        })
    return results


def _palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def save_labelme(json_path: str, dst_dir: str, detections: dict, label_map: LabelMap) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    shapes = []
    boxes = detections["boxes"].tolist()
    labels = detections["labels"].tolist()
    orders = detections["order_seq"].tolist()
    for box, lbl, order in zip(boxes, labels, orders):
        x1, y1, x2, y2 = box
        shapes.append({
            "label": label_map.id2label.get(int(lbl), str(int(lbl))),
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "shape_type": "polygon",
            "reading_order": int(order),
            "group_id": None,
            "flags": {},
            "fillColor": None,
            "lineColor": None,
        })

    out = {
        "imageHeight": src.get("imageHeight"),
        "imageWidth": src.get("imageWidth"),
        "imagePath": src.get("imagePath"),
        "imageData": None,
        "flags": src.get("flags", {}),
        "shape_type": "polygon",
        "shapes": shapes,
    }
    name = os.path.splitext(os.path.basename(json_path))[0] + ".json"
    with open(os.path.join(dst_dir, name), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def save_annotated_image(
    json_path: str,
    dst_dir: str,
    detections: dict,
    label_map: LabelMap,
    palette: list[tuple[int, int, int]],
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    w = int(src["imageWidth"])
    h = int(src["imageHeight"])
    image_rel = src.get("imagePath")
    candidate = None
    if image_rel:
        candidate = image_rel if os.path.isabs(image_rel) else os.path.join(os.path.dirname(json_path), image_rel)
    if candidate and os.path.isfile(candidate):
        try:
            img = Image.open(candidate).convert("RGB")
        except Exception:
            img = Image.new("RGB", (w, h), (255, 255, 255))
    else:
        img = Image.new("RGB", (w, h), (255, 255, 255))

    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", max(14, int(min(w, h) * 0.012)))
    except Exception:
        font = ImageFont.load_default()
    line_w = max(2, int(min(w, h) * 0.0025))

    boxes = detections["boxes"].tolist()
    labels = detections["labels"].tolist()
    orders = detections["order_seq"].tolist()
    scores = detections["scores"].tolist()

    centers: list[tuple[float, float]] = []
    for box, lbl, order, score in zip(boxes, labels, orders, scores):
        x1, y1, x2, y2 = box
        cls_id = int(lbl)
        color = palette[cls_id % len(palette)]
        name = label_map.id2label.get(cls_id, str(cls_id))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        tag = f"{int(order)}: {name} {score:.2f}"
        try:
            tbox = draw.textbbox((x1, y1), tag, font=font)
        except Exception:
            tw = font.getsize(tag)[0] if hasattr(font, "getsize") else len(tag) * 7
            th = font.getsize(tag)[1] if hasattr(font, "getsize") else 12
            tbox = (x1, y1 - th - 4, x1 + tw + 6, y1)
        draw.rectangle(tbox, fill=color + (200,))
        draw.text((tbox[0] + 2, tbox[1] + 1), tag, fill=(255, 255, 255), font=font)
        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))

    # Reading-order trail: connect detection centers in predicted order.
    if len(centers) >= 2:
        order_color = (255, 0, 0, 200)
        draw.line(centers, fill=order_color, width=max(line_w, 3))
        # Arrowhead at each step.
        for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
            dx, dy = x1 - x0, y1 - y0
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                continue
            ux, uy = dx / n, dy / n
            size = max(8, int(min(w, h) * 0.006))
            ax = x1 - ux * size
            ay = y1 - uy * size
            px, py = -uy, ux
            draw.polygon(
                [(x1, y1), (ax + px * size * 0.5, ay + py * size * 0.5),
                 (ax - px * size * 0.5, ay - py * size * 0.5)],
                fill=order_color,
            )

    name = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
    img.save(os.path.join(dst_dir, name), quality=90)


def main() -> None:
    cfg = resolve_config()
    device = torch.device(cfg["device"])
    label_map = LabelMap.build(
        label_list=cfg.get("label_list"), aliases=cfg.get("label_aliases")
    )
    print(f"[labels] {label_map.num_classes} classes")

    print(f"[load] model from {cfg['checkpoint']}")
    model = TrainablePPDocLayoutV3ForObjectDetection.from_pretrained(cfg["checkpoint"])
    model.criterion = PPDocLayoutV3Loss(model.config)
    model.to(device)
    model.eval()

    test_path = resolve_test_split(cfg)
    print(f"[data] test {test_path}")
    test_set = LabelmeLayoutDataset(
        test_path, image_size=int(cfg["image_size"]), label_map=label_map
    )
    print(f"[data] {len(test_set)} samples")
    loader = DataLoader(
        test_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        drop_last=False,
    )

    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    save_mode = cfg["save_mode"]
    labelme_dir = os.path.join(out_dir, "labelme")
    image_dir = os.path.join(out_dir, "image")
    if save_mode in ("labelme", "both"):
        os.makedirs(labelme_dir, exist_ok=True)
    if save_mode in ("image", "both"):
        os.makedirs(image_dir, exist_ok=True)

    palette = _palette(label_map.num_classes)
    threshold = float(cfg["threshold"])

    total_loss = 0.0
    n_batches = 0
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()}
                for t in batch["labels"]
            ]
            # Run twice: once with labels (test loss), once without (predictions
            # without denoising slots in the query dimension).
            if model.criterion is not None:
                out_loss = model(pixel_values=pixel_values, labels=labels)
                total_loss += float(out_loss.loss.detach())
                n_batches += 1
            outputs = model(pixel_values=pixel_values)

            target_sizes = []
            json_paths = []
            for _ in range(pixel_values.shape[0]):
                jp = test_set.json_paths[sample_idx]
                with open(jp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                target_sizes.append((int(meta["imageHeight"]), int(meta["imageWidth"])))
                json_paths.append(jp)
                sample_idx += 1

            results = post_process(outputs, target_sizes, threshold=threshold)
            for jp, det in zip(json_paths, results):
                if save_mode in ("labelme", "both"):
                    save_labelme(jp, labelme_dir, det, label_map)
                if save_mode in ("image", "both"):
                    save_annotated_image(jp, image_dir, det, label_map, palette)
                print(f"[eval] {os.path.basename(jp)}: {det['boxes'].shape[0]} dets")

    if n_batches > 0:
        print(f"[eval] mean test loss: {total_loss / n_batches:.4f}")
    print(f"[eval] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
