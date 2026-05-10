# Trainable PP-DocLayoutV3

Fine-tune [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors)
on your own LabelMe-format layout dataset. Trains all three heads end-to-end:

1. **Detection** — class logits + bounding box regression
2. **Mask** — 200×200 prototype masks (focal + dice)
3. **Reading order** — pairwise BCE on the global-pointer logits

```bash
python train.py --config configs/default.yaml
```

---

## Why this repo exists

The shipped `transformers` v5.8 implementation of `PPDocLayoutV3ForObjectDetection`
is **inference-only**:

```python
# transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py:2044-2045
if labels is not None:
    raise ValueError("PPDocLayoutV3ForObjectDetection does not support training")
```

This repo adds:

* a trainable subclass that replaces that raise with a real multi-head loss,
* a LabelMe → PP-DocLayoutV3 dataset adapter (with synthetic image fallback),
* a custom HungarianMatcher-based loss for det + mask + reading order,
* a fix for an off-by-one bug in the upstream denoising path (see below),
* a YAML/CLI training entry point with first-class support for **custom label lists**.

The installed `transformers` package is **not** modified — everything is a
clean subclass / external module.

---

## What was actually added (and why)

### 1. The training path — `src/modeling.py`

`TrainablePPDocLayoutV3ForObjectDetection` subclasses the shipped
`PPDocLayoutV3ForObjectDetection` and overrides `forward` to:

* run the same model body,
* extract `intermediate_logits`, `intermediate_reference_points`,
  `out_masks`, and `out_order_logits` (one per decoder layer),
* split off the contrastive denoising slice when training (so the matched
  queries and the denoising queries get different supervision),
* call `self.criterion(...)` to compute the multi-head loss when `labels` is
  passed, and return both `loss` and per-component `loss_dict`.

### 2. Bug fix — denoising embedding OOB

The shipped class builds:

```python
self.model.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)  # <-- BUG
```

But the contrastive denoising helper (`get_contrastive_denoising_training_group`)
pads class slots with the sentinel value `num_labels` itself
(`modeling_rt_detr.py:1304`). On GPU this surfaces as:

```
CUDA error: vectorized gather kernel index out of bounds
```

(non-deterministic; depends on whether any sample in the batch has fewer
than `max_gt_num` ground-truth boxes — i.e. it triggers the moment you
have varying object counts across the batch).

RT-DETR itself sizes this embedding correctly — `num_labels + 1` rows with
`padding_idx=num_labels`. We replicate that in the subclass and copy the
pretrained 25 rows into rows `[0:25]` of the new 26-row table at load time:

```python
# src/modeling.py
self.model.denoising_class_embed = nn.Embedding(
    config.num_labels + 1, config.d_model, padding_idx=config.num_labels,
)
```

### 3. Multi-head loss — `src/losses.py`

`PPDocLayoutV3Loss` borrows RT-DETR's `RTDetrHungarianMatcher` for the bipartite
det matching, then computes:

| Component   | Loss                                       | Notes                                                                                                                                                  |
| ----------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `loss_vfl`  | Varifocal classification                   | Reuses RT-DETR's IoU-weighted BCE                                                                                                                      |
| `loss_bbox` | L1 on (cx, cy, w, h)                       |                                                                                                                                                        |
| `loss_giou` | 1 − GIoU                                   |                                                                                                                                                        |
| `loss_mask` | Sigmoid focal                              | Source masks gathered at matched queries, upsampled to target mask size                                                                                |
| `loss_dice` | DICE                                       | Same masks                                                                                                                                             |
| `loss_order` | Pairwise BCE on the upper triangle        | `pair_target[i, j] = 1 ↔ rank[i] < rank[j]`. Matches the inference-time voting in `_get_order_seqs` exactly (lower-tri is masked to −1e4 inside the model). |

Auxiliaries are applied per decoder layer + the encoder-stage proposals. CDN
denoising auxiliaries reuse `get_cdn_matched_indices` for det loss; mask and
order are skipped on aux/dn (RT-DETR convention — too costly).

### 4. Dataset adapter — `src/dataset.py`

`LabelmeLayoutDataset` reads `*.json` files in LabelMe format and yields:

```python
{
    "pixel_values": (3, 800, 800) in [0, 1],
    "labels": {
        "class_labels": (N,),           # int64
        "boxes":        (N, 4),         # (cx, cy, w, h) normalized
        "masks":        (N, 200, 200),  # rasterized polygons
        "order_rank":   (N,),           # dense rank from `reading_order`
    },
}
```

* Polygons are converted to axis-aligned bounding boxes for detection.
* The full polygon shape is preserved as a binary mask via `cv2.fillPoly`.
* `linestrip` shapes (the "reading_order" curves) are skipped — they are
  per-page reading traces, not class instances.
* **If the image referenced by `imagePath` is missing on disk, a blank white
  PIL image is synthesized at the original `(imageWidth, imageHeight)`** — so
  you can train on metadata-only datasets.

### 5. Custom label lists — `src/label_map.py`

The default 25-class list comes from `models/inference.yml` (the original
PaddlePaddle training labels — `id2label` in `config.json` has duplicates we
need to deduplicate). To fine-tune on a different label set, drop a list into
`configs/default.yaml`:

```yaml
label_list:
  - text
  - doc_title
  - paragraph_title
  - header
  - footer
  - image
  - chart
  - table
  - caption
  - formula
  - seal
```

When `len(label_list)` differs from the pretrained 25, **only** the
classification heads are reinitialized:

* `model.enc_score_head.{weight,bias}` — encoder-stage class head
* `model.decoder.class_embed.{weight,bias}` — decoder class head
* `model.denoising_class_embed.weight` — denoising embedding

Everything else (backbone, hybrid encoder, decoder, mask FPN, mask query head,
order head, global pointer) keeps its pretrained weights and is fine-tuned.

### 6. Backbone LR schedule — `src/optim.py`

`build_param_groups` honors `config.backbone_config.lr_mult_list`:

```python
[0, 0.05, 0.05, 0.05, 0.05]  # stem frozen, four HGNet-V2 stages at 0.05 × base_lr
```

Frozen batch norms are explicitly set to `eval()` and `requires_grad=False`.
Encoder/decoder/heads run at the full base LR.

### 7. Config-driven training — `train.py`

* Pure CLI: `python train.py --data ./datasets/foo --epochs 10 --batch-size 2`
* YAML only: `python train.py --config configs/default.yaml`
* Overlay: `python train.py --config configs/default.yaml --epochs 50 --batch-size 1`

Precedence (highest first): **CLI flag → YAML file → built-in defaults**.

---

## Project layout

```
DocLayout/
├── configs/
│   └── default.yaml              # all training args + optional label_list
├── datasets/
│   └── vbhc_deduplicated_labeled/  # 102 LabelMe JSONs (no images — fallback engages)
├── models/                       # pretrained PP-DocLayoutV3 (HF safetensors)
├── src/
│   ├── __init__.py
│   ├── label_map.py              # LabelMap.build(label_list=...)
│   ├── dataset.py                # LabelmeLayoutDataset, collate_fn
│   ├── losses.py                 # PPDocLayoutV3Loss (det + mask + order + aux + dn)
│   ├── modeling.py               # TrainablePPDocLayoutV3ForObjectDetection
│   └── optim.py                  # build_param_groups (lr_mult_list)
├── checkpoints/                  # written by train.py (epoch_N/ subdirs)
└── train.py
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
# (Windows + RTX 50-series: use the cu128 wheel from
#  https://download.pytorch.org/whl/cu128 — PP-DocLayoutV3 needs torch ≥ 2.5
#  and cu121 wheels don't ship sm_120 kernels.)

# 2. Smoke test (one optimizer step, batch 1)
python train.py --max-steps 1 --batch-size 1

# 3. Real training with the YAML default
python train.py --config configs/default.yaml

# 4. CLI override
python train.py --config configs/default.yaml --epochs 50 --batch-size 4 --lr 5e-5

# 5. Resume / inference from a checkpoint (uses the *base* class — training
#    additions are applied at .from_pretrained time)
python -c "
from transformers import PPDocLayoutV3ForObjectDetection, AutoImageProcessor
model = PPDocLayoutV3ForObjectDetection.from_pretrained('./checkpoints/epoch_9')
proc  = AutoImageProcessor.from_pretrained('./models')
"
```

---

## Configuration reference (`configs/default.yaml`)

| Key             | Default                                  | Notes                                             |
| --------------- | ---------------------------------------- | ------------------------------------------------- |
| `data`          | `./datasets/vbhc_deduplicated_labeled`   | LabelMe JSONs root                                |
| `checkpoint`    | `./models`                               | Pretrained PP-DocLayoutV3 (HF format)             |
| `out`           | `./checkpoints`                          | Per-epoch save destination                        |
| `epochs`        | `10`                                     |                                                   |
| `batch_size`    | `2`                                      |                                                   |
| `lr`            | `1e-4`                                   | Base LR; backbone uses `0.05 × lr` per stage      |
| `weight_decay`  | `1e-4`                                   |                                                   |
| `clip_grad`     | `0.1`                                    | Gradient norm clip                                |
| `num_workers`   | `0`                                      | DataLoader workers                                |
| `log_every`     | `10`                                     | Log every N steps                                 |
| `device`        | `cuda` if available else `cpu`           |                                                   |
| `seed`          | `42`                                     |                                                   |
| `max_steps`     | `-1`                                     | -1 = full epochs; > 0 = stop early                |
| `label_list`    | `null`                                   | Custom class list; `null` = 25-class default      |
| `label_aliases` | `null`                                   | Map dataset spellings → canonical names           |

---

## Caveats & out of scope

* No augmentations beyond an 800×800 bicubic resize. Add your own in
  `src/dataset.py` if needed.
* No mAP/eval loop ships with the trainer. Loss decay is the only signal.
* fp32 only — mixed precision left to the user.
* Reading-order supervision uses the per-shape `reading_order` integer in the
  JSON. The `linestrip` curves are *not* used as additional GT.
* Custom label lists reinitialize three small classifier tensors. The
  contrastive denoising loss may take a few epochs to recover from this — if
  it dominates and hurts the other heads, lower `weight_loss_*` for VFL via
  `PPDocLayoutV3Loss(...)` keyword args.

---

## Patched bug summary

| Symptom                                                | Root cause                                                                                                                       | Fix                                                                                              |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `raise ValueError("does not support training")`        | Upstream forward refuses any `labels`                                                                                            | `TrainablePPDocLayoutV3ForObjectDetection.forward` computes loss instead                         |
| `CUDA error: vectorized gather kernel index out of bounds` during training | `denoising_class_embed = nn.Embedding(num_labels, ...)` — sentinel `num_labels` is OOB                                | Resize to `num_labels + 1` with `padding_idx=num_labels`; copy old rows on load                  |
| Crash on samples with `imagePath` missing              | LabelMe JSON references a file we don't have                                                                                     | Synthesize a blank white PIL image at `(imageWidth, imageHeight)`                                |
