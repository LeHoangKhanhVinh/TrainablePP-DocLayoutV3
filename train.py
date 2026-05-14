"""Fine-tune PP-DocLayoutV3 on a LabelMe-format layout dataset.

Usage::

    # Use a YAML config:
    python train.py --config configs/default.yaml

    # Pure CLI:
    python train.py --data ./datasets/foo --epochs 10 --batch-size 2

    # YAML + CLI override (CLI wins):
    python train.py --config configs/default.yaml --epochs 50 --batch-size 1

The YAML may include a ``label_list`` to override the default 25-class set.
When the new size differs from the pretrained checkpoint, the classification
heads (``enc_score_head``, ``decoder.class_embed``, ``denoising_class_embed``)
are reinitialized; everything else (backbone, encoder, decoder, mask + order
heads) keeps its pretrained weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection

from src import (
    Collate,
    DocLayoutV3PostProcess,
    LabelMap,
    LabelmeLayoutDataset,
    LayoutMetric,
    ModelEMA,
    PPDocLayoutV3Loss,
    TrainablePPDocLayoutV3ForObjectDetection,
    build_batch_transforms,
    build_param_groups,
    build_sample_transforms,
    collate_fn,
)


@contextmanager
def ema_weights(model, ema):
    """Temporarily load the EMA weights into ``model``; restore the raw weights on exit.

    PaddleDetection evaluates and snapshots the EMA model rather than the live
    one — this wraps the eval + save block to do exactly that. A no-op when
    ``ema is None``.
    """
    if ema is None:
        yield
        return
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.apply())
    try:
        yield
    finally:
        model.load_state_dict(raw)


def weighted_metric_score(metrics: dict, weights: dict[str, float]) -> float:
    """Combine eval metrics into a single scalar via (renormalised) weights.

    ``weights`` should sum to 1; if it does not it is renormalised. NaN metric
    values (e.g. ``order_acc`` with no matched pairs) contribute 0.
    """
    total_w = sum(float(w) for w in weights.values())
    if total_w <= 0:
        return 0.0
    score = 0.0
    for key, w in weights.items():
        v = metrics.get(key, 0.0)
        if v is None or v != v:  # None / NaN
            v = 0.0
        score += (float(w) / total_w) * float(v)
    return score


DEFAULTS: dict[str, Any] = {
    "data": "./datasets/vbhc_deduplicated_labeled",
    "train_data": None,
    "val_data": None,
    "checkpoint": "./models",
    "out": "./checkpoints",
    "epochs": 10,
    "batch_size": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "clip_grad": 0.1,
    "num_workers": 0,
    "log_every": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "max_steps": -1,
    "label_list": None,
    "label_aliases": None,
    "patience": 5,
    "min_delta": 0.0,
    "drop_last": True,
    "num_denoising": 0,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to a YAML config file.")
    p.add_argument("--data", default=None,
                   help="Dataset root. If it contains train/ and val/ subfolders, "
                        "they're auto-used unless --train-data/--val-data are set.")
    p.add_argument("--train-data", default=None, dest="train_data")
    p.add_argument("--val-data", default=None, dest="val_data")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    p.add_argument("--clip-grad", type=float, default=None, dest="clip_grad")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--log-every", type=int, default=None, dest="log_every")
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stop after this many epochs without val_loss improvement.")
    p.add_argument("--min-delta", type=float, default=None, dest="min_delta",
                   help="Minimum val_loss decrease to count as improvement.")
    return p


def resolve_splits(cfg: dict[str, Any]) -> tuple[str, str]:
    """Pick train/val directories from explicit keys or a `data` root with subfolders."""
    train_path = cfg.get("train_data")
    val_path = cfg.get("val_data")
    data_root = cfg.get("data")
    if not train_path and data_root:
        candidate = os.path.join(data_root, "train")
        if os.path.isdir(candidate):
            train_path = candidate
    if not val_path and data_root:
        candidate = os.path.join(data_root, "val")
        if os.path.isdir(candidate):
            val_path = candidate
    if not train_path or not val_path:
        raise SystemExit(
            "Need both train and val splits. Either set `train_data` and `val_data`, "
            "or point `data` at a folder containing train/ and val/ subdirs."
        )
    if not os.path.isdir(train_path):
        raise SystemExit(f"train_data not found: {train_path}")
    if not os.path.isdir(val_path):
        raise SystemExit(f"val_data not found: {val_path}")
    return train_path, val_path


@torch.no_grad()
def evaluate(model, loader, device, processor, num_classes) -> tuple[float, dict]:
    """Evaluate on the val loader: returns ``(mean_loss, metrics)``.

    A single forward per batch yields both the validation loss and the
    last-layer prediction tensors; the latter are decoded with ``processor`` and
    scored against the ground truth with :class:`~src.metrics.LayoutMetric`.
    Boxes are compared in normalised ``xyxy`` space (IoU is scale-invariant).
    """
    was_training = model.training
    model.eval()
    metric = LayoutMetric(num_classes=num_classes)
    total = 0.0
    n = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in batch["labels"]
        ]
        out = model(pixel_values=pixel_values, labels=labels)
        total += float(out.loss.detach())
        n += 1

        bs = pixel_values.shape[0]
        orig = torch.ones(bs, 2, device=device)  # decode in normalised coords
        head_out = (out.pred_boxes, out.logits, out.order_logits, out.out_masks)
        bbox_pred, _, _ = processor(head_out, orig)
        bbox_pred = bbox_pred.reshape(bs, -1, 7).cpu().numpy()

        preds, gts = [], []
        for i in range(bs):
            per = bbox_pred[i]
            preds.append({
                "boxes": per[:, 2:6],
                "scores": per[:, 1],
                "labels": per[:, 0].astype(np.int64),
                "order": per[:, 6],
            })
            g = labels[i]
            gb = g["boxes"].detach().cpu().numpy().reshape(-1, 4)  # cxcywh, normalised
            if len(gb):
                gxyxy = np.stack([
                    gb[:, 0] - gb[:, 2] / 2, gb[:, 1] - gb[:, 3] / 2,
                    gb[:, 0] + gb[:, 2] / 2, gb[:, 1] + gb[:, 3] / 2,
                ], axis=1)
            else:
                gxyxy = np.zeros((0, 4), dtype=np.float32)
            gts.append({
                "boxes": gxyxy,
                "labels": g["class_labels"].detach().cpu().numpy(),
                "order": g["order_rank"].detach().cpu().numpy(),
            })
        metric.update(preds, gts)

    if was_training:
        model.train()
    return total / max(n, 1), metric.compute()


def resolve_config(argv: list[str] | None = None) -> dict[str, Any]:
    """Merge YAML defaults with CLI overrides into a single dict.

    Precedence (highest first): CLI flag → ``--config`` YAML → built-in DEFAULTS.

    The YAML may carry ``loss`` / ``matcher`` / ``lr_schedule`` sub-mappings (see
    ``configs/default.yaml``); ``main()`` threads those into ``PPDocLayoutV3Loss``
    and the LR scheduler.
    """
    args = _build_parser().parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        if not isinstance(yaml_data, dict):
            raise ValueError(f"{args.config!r} must be a YAML mapping at the top level")
        cfg.update(yaml_data)

    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(cli)
    return cfg


def main() -> None:
    cfg = resolve_config()
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out"], exist_ok=True)
    device = torch.device(cfg["device"])

    print(f"[load] config + base weights from {cfg['checkpoint']}")
    pretrained_config = PPDocLayoutV3Config.from_pretrained(cfg["checkpoint"])

    # Build the LabelMap; default = 25-class PP-DocLayoutV3 list.
    label_map = LabelMap.build(label_list=cfg.get("label_list"), aliases=cfg.get("label_aliases"))
    print(f"[labels] {label_map.num_classes} classes: {label_map.label_list}")

    # Override the config to reflect the user's label set.
    custom_classes = label_map.num_classes != getattr(pretrained_config, "num_labels", 25)
    pretrained_config.id2label = {int(k): v for k, v in label_map.id2label.items()}
    pretrained_config.label2id = dict(label_map.label2id)

    # PP-DocLayoutV3 trains with contrastive denoising OFF; the HF checkpoint
    # config ships num_denoising=100, so override it to match the recipe.
    if cfg.get("num_denoising") is not None:
        nd = int(cfg["num_denoising"])
        if nd != getattr(pretrained_config, "num_denoising", nd):
            print(f"[config] num_denoising {pretrained_config.num_denoising} -> {nd} "
                  "(PP-DocLayoutV3 recipe)")
        pretrained_config.num_denoising = nd

    base = PPDocLayoutV3ForObjectDetection.from_pretrained(cfg["checkpoint"])

    print("[init] TrainablePPDocLayoutV3ForObjectDetection")
    model = TrainablePPDocLayoutV3ForObjectDetection(pretrained_config)
    base_state = base.state_dict()

    # Adapt ``denoising_class_embed.weight`` from the pretrained (num_labels, D)
    # to our resized (num_labels + 1, D) by copying old rows into the front.
    src_key = "model.denoising_class_embed.weight"
    if src_key in base_state and not custom_classes:
        old_w = base_state.pop(src_key)
        new_w = model.model.denoising_class_embed.weight.data
        n = min(old_w.shape[0], new_w.shape[0] - 1)
        new_w[:n].copy_(old_w[:n])
        base_state[src_key] = new_w

    # When the user changes label_list size, drop pretrained classification heads
    # so load_state_dict(strict=False) leaves them at their fresh (random) init.
    if custom_classes:
        n_new = label_map.num_classes
        old_n = getattr(base.config, "num_labels", n_new)
        print(
            f"[labels] custom label_list ({n_new} classes) differs from pretrained "
            f"({old_n}) — reinitializing classification heads."
        )
        for k in [
            "model.enc_score_head.weight",
            "model.enc_score_head.bias",
            "model.denoising_class_embed.weight",
        ]:
            base_state.pop(k, None)
        # Per-decoder-layer class_embed (RTDetrDecoder uses a single nn.Linear in
        # PPDocLayoutV3, but we strip any matching keys defensively).
        for k in list(base_state.keys()):
            if "decoder.class_embed" in k:
                base_state.pop(k)

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    del base, base_state

    loss_cfg = cfg.get("loss")
    matcher_cfg = cfg.get("matcher")
    if matcher_cfg is not None:
        # The matcher follows the loss's focal/sigmoid choice.
        matcher_cfg = dict(matcher_cfg)
        matcher_cfg.setdefault(
            "use_focal_loss",
            (loss_cfg or {}).get("use_focal_loss", True),
        )
    if loss_cfg is not None:
        print(f"[loss] loss_coeff={loss_cfg.get('loss_coeff')} "
              f"use_vfl={loss_cfg.get('use_vfl')} vfl_iou_type={loss_cfg.get('vfl_iou_type')}")
    if matcher_cfg is not None:
        print(f"[loss] matcher_coeff={matcher_cfg.get('matcher_coeff')}")
    model.criterion = PPDocLayoutV3Loss(
        model.config, loss_cfg=loss_cfg, matcher_cfg=matcher_cfg
    )
    model.to(device)

    train_path, val_path = resolve_splits(cfg)
    print(f"[data] loading train={train_path} val={val_path}")

    # PaddleDetection-style transform pipeline (see src/transforms.py).
    train_sample_tf = (
        build_sample_transforms(cfg["train_transforms"])
        if cfg.get("train_transforms") else None
    )
    eval_sample_tf = (
        build_sample_transforms(cfg["eval_transforms"])
        if cfg.get("eval_transforms") else None
    )
    train_batch_tf = (
        build_batch_transforms(cfg["train_batch_transforms"])
        if cfg.get("train_batch_transforms") else None
    )
    if train_sample_tf is not None:
        print(f"[data] train transforms: {len(cfg['train_transforms'])} sample ops "
              f"+ {len(cfg.get('train_batch_transforms') or [])} batch ops")

    train_set = LabelmeLayoutDataset(
        train_path, label_map=label_map, sample_transforms=train_sample_tf
    )
    val_set = LabelmeLayoutDataset(
        val_path, label_map=label_map, sample_transforms=eval_sample_tf
    )
    print(f"[data] train {len(train_set)} samples | val {len(val_set)} samples")
    train_collate = Collate(batch_transforms=train_batch_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=train_collate,
        drop_last=bool(cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("[optim] building param groups")
    param_groups = build_param_groups(
        model, base_lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Exponential moving average of the weights (PaddleDetection `use_ema`).
    ema_cfg = cfg.get("ema") or {}
    ema = None
    if ema_cfg.get("enabled", False):
        ema = ModelEMA(
            model,
            decay=float(ema_cfg.get("decay", 0.9999)),
            decay_type=str(ema_cfg.get("decay_type", "exponential")),
            gamma=float(ema_cfg.get("gamma", 2000)),
            filter_no_grad=bool(ema_cfg.get("filter_no_grad", True)),
        )
        print(f"[ema] enabled decay={ema.decay} type={ema.decay_type} "
              f"gamma={ema.gamma} filter_no_grad={ema.filter_no_grad} "
              f"(eval + best/last use EMA weights)")

    total_steps = max(cfg["epochs"] * max(len(train_loader), 1), 1)
    lr_schedule = cfg.get("lr_schedule")
    if lr_schedule and lr_schedule.get("milestones") is not None:
        steps_per_epoch = max(len(train_loader), 1)
        warmup_steps = int(lr_schedule.get("warmup_steps", 0))
        start_factor = float(lr_schedule.get("warmup_start_factor", 1.0))
        gamma = float(lr_schedule.get("gamma", 1.0))
        milestones = list(lr_schedule.get("milestones", []))

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return start_factor + (1.0 - start_factor) * step / warmup_steps
            factor = 1.0
            for m in milestones:
                if step >= m * steps_per_epoch:
                    factor *= gamma
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"[optim] warmup {warmup_steps} steps (start_factor {start_factor}), "
              f"piecewise decay gamma {gamma} at epoch milestones {milestones}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Post-processor used to decode val predictions for metric computation.
    pp_cfg = cfg.get("postprocess") or {}
    processor = DocLayoutV3PostProcess(
        num_classes=label_map.num_classes,
        num_top_queries=int(pp_cfg.get("num_top_queries", 300)),
        use_focal_loss=True,
        with_mask=True,
        mask_threshold=float(pp_cfg.get("mask_threshold", 0.5)),
        resize_mask=bool(pp_cfg.get("resize_mask", False)),
        use_avg_mask_score=bool(pp_cfg.get("use_avg_mask_score", False)),
    )

    # Best-model selection: 'loss' (lower is better) or 'metrics' (higher score).
    best_cfg = cfg.get("best_model") or {}
    criterion = str(best_cfg.get("criterion", "loss")).lower()
    if criterion not in ("loss", "metrics"):
        raise SystemExit(f"best_model.criterion must be 'loss' or 'metrics', got {criterion!r}")
    metric_weights = best_cfg.get("metric_weights") or {"mAP": 1.0}
    if criterion == "metrics":
        w_sum = sum(float(v) for v in metric_weights.values())
        print(f"[best] criterion=metrics weights={metric_weights}"
              + ("" if abs(w_sum - 1.0) < 1e-6 else f" (sum={w_sum:.3f}, will renormalise)"))
    else:
        print("[best] criterion=loss")

    patience = int(cfg["patience"])
    min_delta = float(cfg["min_delta"])
    best_score = float("inf") if criterion == "loss" else float("-inf")
    epochs_without_improvement = 0
    best_dir = os.path.join(cfg["out"], "best")
    last_dir = os.path.join(cfg["out"], "last")

    model.train()
    step_global = 0
    start = time.time()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()}
                for t in batch["labels"]
            ]
            out = model(pixel_values=pixel_values, labels=labels)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if cfg["clip_grad"] and cfg["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update()

            step_global += 1
            if step_global % cfg["log_every"] == 0 or step_global == 1:
                main_losses = {
                    k: float(v.detach())
                    for k, v in out.loss_dict.items()
                    if "_aux" not in k and "_dn" not in k
                }
                elapsed = time.time() - start
                print(
                    f"[train] epoch {epoch} step {step_global} "
                    f"loss {out.loss.detach().item():.4f} ({elapsed:.1f}s) {main_losses}"
                )
            if cfg["max_steps"] > 0 and step_global >= cfg["max_steps"]:
                ckpt_dir = os.path.join(cfg["out"], "smoke")
                with ema_weights(model, ema):
                    model.save_pretrained(ckpt_dir)
                print(f"[save] {ckpt_dir}")
                print("[train] reached --max-steps; stopping")
                return

        # Evaluate + snapshot on the EMA weights (PaddleDetection-style); training
        # itself continues with the raw weights, restored on context exit.
        with ema_weights(model, ema):
            val_loss, metrics = evaluate(
                model, val_loader, device, processor, label_map.num_classes
            )
            metric_score = weighted_metric_score(metrics, metric_weights)
            current = val_loss if criterion == "loss" else metric_score
            if criterion == "loss":
                improved = current < best_score - min_delta
            else:
                improved = current > best_score + min_delta

            order_score = metrics["order_score"]
            order_str = "n/a" if order_score != order_score else f"{order_score:.4f}"
            print(
                f"[eval] epoch {epoch} | val_loss {val_loss:.4f} | "
                f"mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
                f"AP75 {metrics['AP75']:.4f} order_score {order_str} | "
                f"metric_score {metric_score:.4f}"
                f"{' -> NEW BEST' if improved else ''}"
            )

            model.save_pretrained(last_dir)
            if improved:
                model.save_pretrained(best_dir)
                print(f"[save] best -> {best_dir} ({criterion}={current:.4f})")

        if improved:
            best_score = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[train] early stopping: no {criterion} improvement for "
                    f"{patience} epochs (best {criterion}={best_score:.4f})"
                )
                return

    print(f"[train] done. best {criterion}={best_score:.4f} -> {best_dir}")


if __name__ == "__main__":
    main()
