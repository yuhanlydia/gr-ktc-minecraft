"""Low-rank residual-stream adapter used for causal reachability tests."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


class ResidualLoRAHook(AbstractContextManager):
    """Temporarily add ``B A h`` to one Transformer layer output.

    This realizes exactly the linear proxy fitted by ``fit_reachability``.  It
    is deliberately separate from PEFT's projection-specific LoRA so the
    mechanism experiment does not pretend residual features are o_proj inputs.
    """

    def __init__(self, layer: torch.nn.Module, a: torch.Tensor, b: torch.Tensor, *, scale: float = 1.0):
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError("expected A=[rank, hidden], B=[hidden, rank]")
        self.layer = layer
        self.a = a
        self.b = b
        self.scale = float(scale)
        self._handle: Any = None

    def _hook(self, module, inputs, output):
        hidden = inputs[0]
        a = self.a.to(device=hidden.device, dtype=hidden.dtype)
        b = self.b.to(device=hidden.device, dtype=hidden.dtype)
        delta = ((hidden @ a.T) @ b.T) * self.scale
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("hook already active")
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class ScheduledResidualStateHook(AbstractContextManager):
    """Apply a saved token-phase state correction during autoregressive decode.

    Prompt prefill is left unchanged. Each subsequent one-token layer call gets
    the next correction, matching the continuation semantics of KV prefix
    injection (which also leaves the first token ordinary in this codebase).
    """

    def __init__(self, layer: torch.nn.Module, correction: torch.Tensor, *, scale: float = 1.0):
        if correction.ndim != 2 or correction.shape[0] == 0:
            raise ValueError("correction must be [tokens, hidden] and non-empty")
        self.layer = layer
        self.correction = correction
        self.scale = float(scale)
        self._step = 0
        self._handle: Any = None

    def _hook(self, module, inputs, output):
        hidden = inputs[0]
        if hidden.shape[-2] != 1:
            return output
        index = min(self._step, self.correction.shape[0] - 1)
        delta = self.correction[index].to(
            device=hidden.device, dtype=hidden.dtype
        ).view(1, 1, -1) * self.scale
        self._step += 1
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("hook already active")
        self._step = 0
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class QueryResponsiveResidualMemoryHook(AbstractContextManager):
    """Retrieve unreachable corrections by the current hidden-state query."""

    def __init__(
        self,
        layer: torch.nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        temperature: float = 0.1,
        top_k: int = 4,
        scale: float = 1.0,
    ):
        if keys.ndim != 2 or values.ndim != 2 or keys.shape != values.shape:
            raise ValueError("keys and values must have equal [tokens, hidden] shapes")
        if temperature <= 0 or top_k < 1:
            raise ValueError("temperature and top_k must be positive")
        self.layer = layer
        self.keys = keys
        self.values = values
        self.temperature = float(temperature)
        self.top_k = min(top_k, keys.shape[0])
        self.scale = float(scale)
        self._handle: Any = None

    def _hook(self, module, inputs, output):
        hidden = inputs[0]
        if hidden.shape[-2] != 1:
            return output
        query = torch.nn.functional.normalize(hidden[0, 0].float(), dim=0)
        keys = torch.nn.functional.normalize(
            self.keys.to(hidden.device).float(), dim=-1
        )
        logits = keys @ query / self.temperature
        top = logits.topk(self.top_k)
        weights = torch.softmax(top.values, dim=0)
        values = self.values.to(device=hidden.device, dtype=hidden.dtype)
        delta = (weights.to(hidden.dtype)[:, None] * values[top.indices]).sum(0)
        delta = delta.view(1, 1, -1) * self.scale
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("hook already active")
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def text_layer(model: Any, layer_id: int) -> torch.nn.Module:
    """Resolve the text layer across base Qwen and PEFT wrappers."""
    candidates = [model]
    while candidates:
        current = candidates.pop(0)
        for path in (("model", "language_model", "layers"), ("language_model", "layers"), ("layers",)):
            node = current
            try:
                for name in path:
                    node = getattr(node, name)
                return node[layer_id]
            except (AttributeError, IndexError, TypeError):
                pass
        for name in ("base_model", "model"):
            child = getattr(current, name, None)
            if child is not None and child is not current and child not in candidates:
                candidates.append(child)
    raise AttributeError("could not resolve text transformer layers")
