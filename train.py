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
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection

from src import (
    LabelMap,
    LabelmeLayoutDataset,
    PPDocLayoutV3Loss,
    TrainablePPDocLayoutV3ForObjectDetection,
    build_param_groups,
    collate_fn,
)


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
def evaluate(model, loader, device) -> float:
    """Return average loss over the val loader."""
    was_training = model.training
    model.eval()
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
    if was_training:
        model.train()
    return total / max(n, 1)


def resolve_config(argv: list[str] | None = None) -> dict[str, Any]:
    """Merge YAML defaults with CLI overrides into a single dict.

    Precedence (highest first): CLI flag → YAML file → built-in DEFAULTS.
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

    model.criterion = PPDocLayoutV3Loss(model.config)
    model.to(device)

    train_path, val_path = resolve_splits(cfg)
    print(f"[data] loading train={train_path} val={val_path}")
    train_set = LabelmeLayoutDataset(train_path, label_map=label_map)
    val_set = LabelmeLayoutDataset(val_path, label_map=label_map)
    print(f"[data] train {len(train_set)} samples | val {len(val_set)} samples")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
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
    total_steps = max(cfg["epochs"] * max(len(train_loader), 1), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val = float("inf")
    epochs_without_improvement = 0
    patience = int(cfg["patience"])
    min_delta = float(cfg["min_delta"])
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

            step_global += 1
            if step_global % cfg["log_every"] == 0 or step_global == 1:
                main_losses = {
                    k: float(v.detach())
                    for k, v in out.loss_dict.items()
                    if "_aux_" not in k and "_dn_" not in k and "_enc" not in k
                }
                elapsed = time.time() - start
                print(
                    f"[train] epoch {epoch} step {step_global} "
                    f"loss {out.loss.detach().item():.4f} ({elapsed:.1f}s) {main_losses}"
                )
            if cfg["max_steps"] > 0 and step_global >= cfg["max_steps"]:
                ckpt_dir = os.path.join(cfg["out"], "smoke")
                model.save_pretrained(ckpt_dir)
                print(f"[save] {ckpt_dir}")
                print("[train] reached --max-steps; stopping")
                return

        val_loss = evaluate(model, val_loader, device)
        improved = val_loss < best_val - min_delta
        print(
            f"[eval] epoch {epoch} val_loss {val_loss:.4f} "
            f"(best {best_val:.4f}{' -> NEW BEST' if improved else ''})"
        )

        model.save_pretrained(last_dir)
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(best_dir)
            print(f"[save] best -> {best_dir} (val_loss {best_val:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[train] early stopping: no val_loss improvement for "
                    f"{patience} epochs (best {best_val:.4f})"
                )
                return

    print(f"[train] done. best val_loss {best_val:.4f} -> {best_dir}")


if __name__ == "__main__":
    main()
