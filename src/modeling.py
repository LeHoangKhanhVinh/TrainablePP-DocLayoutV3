"""Trainable subclass of ``PPDocLayoutV3ForObjectDetection``.

The shipped class hard-raises when ``labels is not None``. We override the
forward to compute the full PP-DocLayoutV3 multi-head loss instead, while keeping
the inference-without-labels path identical to the parent.

To match PaddleDetection's loss *exactly*, the encoder-stage proposal must take
part in the auxiliary loss with its own mask + dice term. The HF
``PPDocLayoutV3Model`` computes ``enc_out_masks`` internally but discards it, so we
recover the two ingredients it is built from with lightweight forward hooks:

* ``mask_query_head`` — its **first** call per forward is the encoder-stage
  ``mask_query_embed``;
* ``encoder`` — its output carries ``mask_feat``.

and recompute ``enc_out_masks = bmm(mask_query_embed, mask_feat)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ForObjectDetectionOutput,
)

from .losses import PPDocLayoutV3Loss


@dataclass
class PPDocLayoutV3TrainOutput(PPDocLayoutV3ForObjectDetectionOutput):
    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None


class TrainablePPDocLayoutV3ForObjectDetection(PPDocLayoutV3ForObjectDetection):
    """Adds a training path to PP-DocLayoutV3.

    Wires a :class:`PPDocLayoutV3Loss` into ``self.criterion``; the train script
    is responsible for instantiating + assigning it after construction.
    """

    def __init__(self, config):
        super().__init__(config)
        # Bug fix: ``get_contrastive_denoising_training_group`` pads with the sentinel
        # value ``num_classes`` (== num_labels), but the shipped
        # ``PPDocLayoutV3ForObjectDetection`` builds ``denoising_class_embed`` with
        # exactly ``num_labels`` rows -> CUDA index OOB during training. Rebuild it
        # the way RT-DETR does, with ``num_labels + 1`` rows and a padding_idx.
        self.model.denoising_class_embed = nn.Embedding(
            config.num_labels + 1,
            config.d_model,
            padding_idx=config.num_labels,
        )
        self.criterion: PPDocLayoutV3Loss | None = None

        # ---- encoder-stage mask recovery (see module docstring) -------------------
        self._mask_query_head_calls: list[torch.Tensor] = []
        self._enc_mask_feat: torch.Tensor | None = None
        self.model.mask_query_head.register_forward_hook(self._capture_mask_query_head)
        self.model.encoder.register_forward_hook(self._capture_encoder)

    def _capture_mask_query_head(self, module, inputs, output):
        self._mask_query_head_calls.append(output)

    def _capture_encoder(self, module, inputs, output):
        mask_feat = getattr(output, "mask_feat", None)
        if mask_feat is None and isinstance(output, (tuple, list)):
            mask_feat = output[-1]
        self._enc_mask_feat = mask_feat

    def _compute_enc_masks(self) -> torch.Tensor | None:
        """Recompute the encoder-stage instance masks from the captured tensors."""
        if not self._mask_query_head_calls or self._enc_mask_feat is None:
            return None
        mask_query_embed = self._mask_query_head_calls[0]  # (B, num_queries, num_prototypes)
        mask_feat = self._enc_mask_feat                    # (B, num_prototypes, mh, mw)
        mh, mw = mask_feat.shape[-2:]
        return torch.bmm(
            mask_query_embed, mask_feat.flatten(start_dim=2)
        ).reshape(mask_query_embed.shape[0], -1, mh, mw)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs,
    ):
        # Reset hook capture buffers for this forward pass.
        self._mask_query_head_calls = []
        self._enc_mask_feat = None

        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            **kwargs,
        )

        intermediate_logits = outputs.intermediate_logits
        intermediate_refp = outputs.intermediate_reference_points
        order_logits = outputs.out_order_logits
        out_masks = outputs.out_masks

        # When denoising is active, the per-query dim contains denoising slots up front;
        # the "last layer" inference outputs strip them by taking the trailing num_queries.
        if outputs.denoising_meta_values is not None:
            num_dn = int(outputs.denoising_meta_values["dn_num_split"][0])
            last_logits = intermediate_logits[:, -1, num_dn:]
            last_pred_boxes = intermediate_refp[:, -1, num_dn:]
            last_out_masks = out_masks[:, -1, num_dn:]
        else:
            last_logits = intermediate_logits[:, -1]
            last_pred_boxes = intermediate_refp[:, -1]
            last_out_masks = out_masks[:, -1]
        last_order_logits = order_logits[:, -1]

        loss = None
        loss_dict = None
        if labels is not None:
            if self.criterion is None:
                raise RuntimeError(
                    "TrainablePPDocLayoutV3ForObjectDetection.criterion is not set; "
                    "assign a PPDocLayoutV3Loss before training."
                )
            enc_topk_masks = self._compute_enc_masks()
            loss, loss_dict = self.criterion(
                intermediate_logits=intermediate_logits,
                intermediate_reference_points=intermediate_refp,
                out_masks=out_masks,
                out_order_logits=order_logits,
                enc_topk_logits=outputs.enc_topk_logits,
                enc_topk_bboxes=outputs.enc_topk_bboxes,
                enc_topk_masks=enc_topk_masks,
                denoising_meta_values=outputs.denoising_meta_values,
                targets=labels,
            )

        # Release captured tensors so they are not held past the loss computation.
        self._mask_query_head_calls = []
        self._enc_mask_feat = None

        return PPDocLayoutV3TrainOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=last_logits,
            pred_boxes=last_pred_boxes,
            order_logits=last_order_logits,
            out_masks=last_out_masks,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            intermediate_predicted_corners=outputs.intermediate_predicted_corners,
            initial_reference_points=outputs.initial_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )


__all__ = ["TrainablePPDocLayoutV3ForObjectDetection", "PPDocLayoutV3TrainOutput"]
