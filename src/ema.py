"""Exponential moving average of model weights — port of PaddleDetection's ModelEMA.

Mirrors ``ppdet/optimizer/ema.py``: a shadow copy of the full ``state_dict``
(parameters *and* buffers) is updated after every optimizer step with
``shadow = decay * shadow + (1 - decay) * model``. The decay ramps up over
training; for ``decay_type='exponential'`` it is
``decay * (1 - exp(-(step + 1) / gamma))`` — which starts near 0 (so the
zero-initialised shadow tracks the model from the very first step) and anneals
up to ``decay``.

At snapshot time PaddleDetection evaluates and saves the EMA weights rather than
the raw ones; :func:`ModelEMA.apply` returns a state dict to ``load_state_dict``
for exactly that.

Pure PyTorch — no ``paddle`` dependency.
"""

from __future__ import annotations

import math

import torch

__all__ = ["ModelEMA"]


class ModelEMA:
    """Tracks an exponential moving average of ``model``'s ``state_dict``.

    Args:
        model: the live model being trained.
        decay: target EMA decay (the ceiling the ramped decay anneals to).
        decay_type: ``'exponential'`` (default, ramps via ``gamma``),
            ``'threshold'`` (``min(decay, (1+step)/(10+step))``) or ``'normal'``
            (constant ``decay``).
        gamma: ramp time-constant for ``decay_type='exponential'``.
        filter_no_grad: if True, parameters with ``requires_grad=False`` (e.g. a
            frozen backbone stem) are kept verbatim instead of being averaged.

    Non-floating-point buffers (e.g. ``num_batches_tracked``) are always kept
    verbatim — averaging them is meaningless and would corrupt their dtype.
    """

    def __init__(
        self,
        model,
        decay: float = 0.9999,
        decay_type: str = "exponential",
        gamma: float = 2000.0,
        filter_no_grad: bool = True,
    ) -> None:
        assert decay_type in ("exponential", "threshold", "normal"), decay_type
        self.decay = float(decay)
        self.decay_type = decay_type
        self.gamma = float(gamma)
        self.filter_no_grad = bool(filter_no_grad)
        self.step = 0
        self._model = model

        # entries kept verbatim (not averaged): frozen params + non-float buffers
        self.black_list: set[str] = set()
        if self.filter_no_grad:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    self.black_list.add(name)

        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if k in self.black_list or not torch.is_floating_point(v):
                self.black_list.add(k)
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k] = torch.zeros_like(v, dtype=torch.float32)

    def _current_decay(self) -> float:
        if self.decay_type == "threshold":
            return min(self.decay, (1 + self.step) / (10 + self.step))
        if self.decay_type == "exponential":
            return self.decay * (1.0 - math.exp(-(self.step + 1) / self.gamma))
        return self.decay

    @torch.no_grad()
    def update(self) -> None:
        """Fold the model's current weights into the shadow — call once per optimizer step."""
        decay = self._current_decay()
        msd = self._model.state_dict()
        for k, v in self.shadow.items():
            if k in self.black_list:
                continue
            v.mul_(decay).add_(msd[k].detach().to(torch.float32), alpha=1.0 - decay)
        self.step += 1

    @torch.no_grad()
    def apply(self) -> dict[str, torch.Tensor]:
        """Return a full state dict with the EMA weights (load it for eval / saving)."""
        msd = self._model.state_dict()
        if self.step == 0:  # nothing averaged yet — fall back to the raw weights
            return {k: v.detach().clone() for k, v in msd.items()}
        out: dict[str, torch.Tensor] = {}
        for k, v in self.shadow.items():
            if k in self.black_list:
                out[k] = v.detach().clone()
                continue
            vv = v
            if self.decay_type != "exponential":
                # de-bias the zero-initialised average (exponential type needs none,
                # since its decay starts near 0)
                vv = vv / (1.0 - self.decay ** self.step)
            out[k] = vv.detach().clone().to(msd[k].dtype)
        return out

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "shadow": self.shadow,
            "black_list": sorted(self.black_list),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step = int(sd["step"])
        self.shadow = sd["shadow"]
        self.black_list = set(sd["black_list"])
