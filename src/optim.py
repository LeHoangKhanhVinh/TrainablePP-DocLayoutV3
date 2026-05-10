"""Parameter group construction honoring the backbone ``lr_mult_list``.

The PaddlePaddle recipe (encoded in ``config.backbone_config.lr_mult_list``)
freezes the stem (mult=0) and trains each HGNet-V2 stage at 0.05 of the base
LR. Encoder, decoder, and heads run at the full base LR. Frozen batch norms
(``freeze_norm=True``) are also enforced explicitly.

The backbone parameter names land under
``model.backbone.model.embedder.*`` (stem) and
``model.backbone.model.encoder.stages.{0..3}.*``.
"""

from __future__ import annotations

import re

import torch.nn as nn

# Stage index 0 in lr_mult_list refers to the stem; indices 1..4 to encoder stages 0..3.
_STAGE_RE = re.compile(r"backbone\.model\.encoder\.stages\.(\d+)\.")
_STEM_RE = re.compile(r"backbone\.model\.embedder\.")


def _classify(param_name: str) -> str:
    """Return one of ``"stem"`` / ``"stage{i}"`` / ``"other"``."""
    if _STEM_RE.search(param_name):
        return "stem"
    m = _STAGE_RE.search(param_name)
    if m:
        return f"stage{int(m.group(1))}"
    return "other"


def build_param_groups(
    model: nn.Module,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_mult_list: list[float] | None = None,
    freeze_batch_norms: bool = True,
) -> list[dict]:
    """Build AdamW parameter groups.

    ``lr_mult_list[0]`` applies to the stem, ``lr_mult_list[i+1]`` to encoder
    stage ``i``. A multiplier of ``0`` disables training of the corresponding
    parameters. If ``lr_mult_list`` is omitted, it is read from
    ``model.config.backbone_config.lr_mult_list``.
    """
    if lr_mult_list is None:
        lr_mult_list = list(model.config.backbone_config.lr_mult_list)

    if freeze_batch_norms:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                module.eval()
                for p in module.parameters(recurse=False):
                    p.requires_grad_(False)

    buckets: dict[str, list] = {"stem": [], "other": []}
    for i in range(len(lr_mult_list) - 1):
        buckets[f"stage{i}"] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        kind = _classify(name)
        # If stage idx is beyond what we expected, fall back to "other".
        if kind not in buckets:
            kind = "other"
        buckets[kind].append(param)

    # Stem freeze handling.
    if buckets["stem"] and lr_mult_list[0] == 0:
        for p in buckets["stem"]:
            p.requires_grad_(False)
        buckets["stem"] = []

    groups: list[dict] = []
    for i, mult in enumerate(lr_mult_list):
        key = "stem" if i == 0 else f"stage{i - 1}"
        params = buckets.get(key, [])
        if not params or mult == 0:
            continue
        groups.append({"params": params, "lr": base_lr * mult, "weight_decay": weight_decay})
    if buckets["other"]:
        groups.append({"params": buckets["other"], "lr": base_lr, "weight_decay": weight_decay})

    return groups


__all__ = ["build_param_groups"]
